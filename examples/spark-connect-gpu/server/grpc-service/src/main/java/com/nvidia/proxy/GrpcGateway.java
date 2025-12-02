package com.nvidia.proxy;

import org.apache.spark.connect.proto.*;
import org.sparkproject.connect.grpc.*;
import org.sparkproject.connect.grpc.stub.StreamObserver;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

public class GrpcGateway {
    private final ManagedChannel cpuChannel;
    private final ManagedChannel gpuChannel;
    private final SparkConnectServiceGrpc.SparkConnectServiceBlockingStub cpuService;
    private final SparkConnectServiceGrpc.SparkConnectServiceBlockingStub gpuService;
    private static int count;

    // Capture the server-side session ID and set it to empty initially. It will
    // be updated on the first response received of the first Spark Connect Server.
    // the clientServerSideSessionId will be sent back to client.
    private String clientServerSideSessionId = "";

    // Map the server side session id to the Spark Connect server.
    private Map<String, SparkConnectServiceGrpc.SparkConnectServiceBlockingStub> serverSideSessionIdToService;
    private Map<SparkConnectServiceGrpc.SparkConnectServiceBlockingStub, String> serviceToServerSideSessionId;

    public GrpcGateway(String cpuHost, String gpuHost) {
        this.cpuChannel = ManagedChannelBuilder.forAddress(cpuHost, 15002).usePlaintext().build();
        this.cpuService = SparkConnectServiceGrpc.newBlockingStub(cpuChannel);
        this.gpuChannel = ManagedChannelBuilder.forAddress(gpuHost, 15002).usePlaintext().build();
        this.gpuService = SparkConnectServiceGrpc.newBlockingStub(gpuChannel);

        serverSideSessionIdToService = new HashMap<>();
        serviceToServerSideSessionId = new HashMap<>();
    }

    private class ProxyService extends SparkConnectServiceGrpc.SparkConnectServiceImplBase {
//        @Override
//        public void config(ConfigRequest request, StreamObserver<ConfigResponse> responseObserver) {
//        }

        private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub chooseAService() {
            return (count++ % 2) == 1 ? cpuService : gpuService;
//            return cpuService;
        }

        @Override
        public void executePlan(ExecutePlanRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            String operationId = request.getOperationId();
            String userId = request.getUserContext().getUserId();
            String sessionId = request.getSessionId();
            String plan = request.getPlan().toString();
            String clientSideServerSideSessionId = request.getClientObservedServerSideSessionId();

            SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service = chooseAService();
            if (serviceToServerSideSessionId.containsKey(service)) {
                String serverSideSessionId = serviceToServerSideSessionId.get(service);
                request = request.toBuilder().setClientObservedServerSideSessionId(serverSideSessionId).build();
            } else {
                request = request.toBuilder().clearClientObservedServerSideSessionId().build();
            }

            System.out.println(
                    "userId: " + userId +
                            " sessionId: " + sessionId +
                            " clientSideServerSideSessionId " + clientSideServerSideSessionId +
                            " serverSide sessionId: " + request.getClientObservedServerSideSessionId() +
                            " operationId: " + operationId +
                            " plan:" + plan + "\nRunning on " + service.getChannel().toString());
            System.out.println("--------------------------------------------");

            try {
                // 1. Call upstream service (Blocking call)
                Iterator<ExecutePlanResponse> responses = service.executePlan(request);

                // 2. Iterate through results and stream them back to the client
                while (responses.hasNext()) {
                    ExecutePlanResponse response = responses.next();

                    String serverSideSessionId = response.getServerSideSessionId();
                    if (!serverSideSessionIdToService.containsKey(serverSideSessionId)) {
                        serverSideSessionIdToService.put(serverSideSessionId, service);
                        serviceToServerSideSessionId.put(service, serverSideSessionId);
                    }

                    if (Objects.equals(clientServerSideSessionId, "")) {
                        clientServerSideSessionId = serverSideSessionId;
                    } else {
                        response = response.toBuilder().setServerSideSessionId(clientServerSideSessionId).build();
                    }

                    System.out.println("Got Response: " + response);
                    responseObserver.onNext(response);
                }

                // 3. Signal that the stream is finished successfully
                responseObserver.onCompleted();

            } catch (Exception e) {
                // 4. Propagate any errors (upstream failures) back to the client
                responseObserver.onError(e);
            }
        }
    }

    public void start() throws IOException, InterruptedException {
        Server server = ServerBuilder.forPort(15002)
//                .addService(ServerInterceptors.intercept(new ProxyService(), getAuthInterceptor()))
                .addService(ServerInterceptors.intercept(new ProxyService()))
                .build()
                .start();
        System.out.println("Proxy Server started on 15002");
        server.awaitTermination();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        new GrpcGateway("spark-connect-server-cpu", "spark-connect-server").start();
    }

}
