package com.nvidia.proxy;

import org.apache.spark.connect.proto.*;
import org.sparkproject.connect.grpc.*;
import org.sparkproject.connect.grpc.stub.StreamObserver;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class GrpcGateway {
    private static final Context.Key<String> UNIQ_ID_CONTEXT_KEY = Context.key("uniq_id");
    private static final Metadata.Key<String> UNIQ_ID_METADATA_KEY =
            Metadata.Key.of("uniq_id", Metadata.ASCII_STRING_MARSHALLER);

    private final SessionManager sessionManager;
    private final Router router;

    public GrpcGateway() {
        sessionManager = new SessionManager();
        router = new Router();
    }

    private class ProxyService extends SparkConnectServiceGrpc.SparkConnectServiceImplBase {

        @Override
        public void executePlan(ExecutePlanRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            String operationId = request.getOperationId();
            String userId = request.getUserContext().getUserId();
            String clientSessionId = request.getSessionId();
            String plan = request.getPlan().toString();
            String clientSideServerSideSessionId = request.getClientObservedServerSideSessionId();

            var service = router.routePerSession(UNIQ_ID_CONTEXT_KEY.get(), userId, clientSessionId);

            Optional<String> clientObservedServerSideSessionId = sessionManager.getServerSideSessionId(clientSessionId, service);
            System.out.println("xxxxx executePlan: uniq id: " + UNIQ_ID_CONTEXT_KEY.get() +
                    " => clientSessionId: " + clientSessionId +
                    " get ServerInterceptor: " + clientObservedServerSideSessionId.orElse("NO NO NO"));

            // Reset or clear the clientObservedServerSideSessionId field
            var builder = request.toBuilder();
            clientObservedServerSideSessionId.ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            request = builder.build();

            System.out.println(
                    "userId: " + userId +
                            "\nsessionId: " + clientSessionId +
                            "\nclientSideServerSideSessionId " + clientSideServerSideSessionId +
                            "\nserverSide sessionId: " + request.getClientObservedServerSideSessionId() +
                            "\noperationId: " + operationId +
                            "\nplan:" + plan +
                            "\nRunning on " + service.getChannel().toString());
            System.out.println("--------------------------------------------");

            try {
                // 1. Call upstream service (Blocking call)
                Iterator<ExecutePlanResponse> responses = service.executePlan(request);

                // 2. Iterate through results and stream them back to the client
                String consistentSessionId = null;
                while (responses.hasNext()) {
                    ExecutePlanResponse response = responses.next();
                    // For responses of the same query.
                    if (consistentSessionId == null) {
                        consistentSessionId = response.getServerSideSessionId();
                        String rawId = consistentSessionId;
                        sessionManager.storeSessionIds(clientSessionId, consistentSessionId, service);
                        consistentSessionId = sessionManager
                                .getServerSideSessionId(clientSessionId, consistentSessionId);
                        System.out.printf("xxxxx => Session ID resolved for client %s: %s → %s%n",
                                clientSessionId, rawId, consistentSessionId);
                    }
                    response = response.toBuilder().setServerSideSessionId(consistentSessionId).build();
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

    private ServerInterceptor getAuthInterceptor() {
        return new ServerInterceptor() {
            @Override
            public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                    ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {

                String token = headers.get(Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER));

                // Simple check: Token must be "Bearer spark-secret-token"
                if (token == null || !token.equals("Bearer spark-secret-token")) {
                    call.close(Status.UNAUTHENTICATED.withDescription("Invalid Token"), headers);
                    return new ServerCall.Listener<ReqT>() {
                    };
                }

                String uniqId = headers.get(UNIQ_ID_METADATA_KEY);  // Extract uniq_id
                // Create new context with uniq_id attached
                Context context = Context.current()
                        .withValue(UNIQ_ID_CONTEXT_KEY, Optional.ofNullable(uniqId).orElse(""));

                return Contexts.interceptCall(context, call, headers, next);
            }
        };
    }

    public void start() throws IOException, InterruptedException {
        File certChain = new File("/opt/spark/certs/server.crt");
        File privateKey = new File("/opt/spark/certs/server.key");

        Server server = ServerBuilder.forPort(15002)
                .addService(ServerInterceptors.intercept(new ProxyService(), getAuthInterceptor()))
                .useTransportSecurity(certChain, privateKey)
                .build()
                .start();
        System.out.println("Proxy Server started on 15002");
        server.awaitTermination();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        new GrpcGateway().start();
    }

}
