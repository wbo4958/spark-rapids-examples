package com.nvidia.proxy;

import org.apache.spark.connect.proto.*;
import org.sparkproject.connect.grpc.*;
import org.sparkproject.connect.grpc.stub.StreamObserver;

import java.io.IOException;
import java.util.Iterator;

public class GrpcGateway {
    private final ManagedChannel cpuChannel;
    private final ManagedChannel gpuChannel;
    private final SparkConnectServiceGrpc.SparkConnectServiceBlockingStub cpuService;
    private final SparkConnectServiceGrpc.SparkConnectServiceBlockingStub gpuService;

    private static int count;

    public GrpcGateway(String cpuHost, String gpuHost) {
        this.cpuChannel = ManagedChannelBuilder.forAddress(cpuHost, 15002).usePlaintext().build();
        this.cpuService = SparkConnectServiceGrpc.newBlockingStub(cpuChannel);
        this.gpuChannel = ManagedChannelBuilder.forAddress(gpuHost, 15002).usePlaintext().build();
        this.gpuService = SparkConnectServiceGrpc.newBlockingStub(gpuChannel);
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

            SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service = chooseAService();

            System.out.println("userId: " + userId + " sessionId: " + sessionId +  " operationId: " + operationId +
                    " plan:" + plan + "\nRunning on " + service.getChannel().toString());
            System.out.println("--------------------------------------------");

            try {
                // 1. Call upstream service (Blocking call)
                Iterator<ExecutePlanResponse> responses = service.executePlan(request);

                // 2. Iterate through results and stream them back to the client
                while (responses.hasNext()) {
                    ExecutePlanResponse response = responses.next();
                    System.out.println("Got Response: " + response.toString());
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
