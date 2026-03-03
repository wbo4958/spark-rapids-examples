package com.nvidia.proxy;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.spark.connect.proto.*;
import org.sparkproject.connect.grpc.Context;
import org.sparkproject.connect.grpc.Contexts;
import org.sparkproject.connect.grpc.Metadata;
import org.sparkproject.connect.grpc.Server;
import org.sparkproject.connect.grpc.ServerBuilder;
import org.sparkproject.connect.grpc.ServerCall;
import org.sparkproject.connect.grpc.ServerCallHandler;
import org.sparkproject.connect.grpc.ServerInterceptor;
import org.sparkproject.connect.grpc.ServerInterceptors;
import org.sparkproject.connect.grpc.Status;
import org.sparkproject.connect.grpc.stub.StreamObserver;

public class GrpcGateway {
    private static final Logger LOG = Logger.getLogger(GrpcGateway.class.getName());

    private static final Context.Key<String> CONNECT_ID_CONTEXT_KEY = Context.key("connect_id");
    private static final Metadata.Key<String> CONNECT_ID_METADATA_KEY =
            Metadata.Key.of("connect_id", Metadata.ASCII_STRING_MARSHALLER);

    private static final Context.Key<String> CLUSTER_ID_CONTEXT_KEY = Context.key("cluster_id");
    private static final Metadata.Key<String> CLUSTER_ID_METADATA_KEY =
            Metadata.Key.of("cluster_id", Metadata.ASCII_STRING_MARSHALLER);

    // Default data directory if DATA_DIR environment variable is not set
    private static final String DEFAULT_DATA_DIR = "/data";

    private final SessionManager sessionManager;
    private final Router router;
    private final String eventLogBaseDir;
    /**
     * Cache of service-specific metadata for connected Spark Connect servers.
     *
     * Maps each SparkConnectServiceBlockingStub (representing a unique Spark Connect server connection)
     * to its associated ServiceInfo object, which may include properties such as event log directory,
     * session timeout, and other configuration or identification details required across requests.
     *
     * This allows for fast lookups and avoids the need to re-query service metadata on each request.
     * It is updated as new services are connected or new sessions are tracked.
     */
    private final Map<SparkConnectServiceGrpc.SparkConnectServiceBlockingStub, ServiceInfo> serviceInfos = new ConcurrentHashMap<>();

    public GrpcGateway() {
        sessionManager = new SessionManager();
        router = new Router();
        LOG.setLevel(Level.FINE);

        // Read DATA_DIR from environment variable, default to /data
        String dataDir = System.getenv("DATA_DIR");
        if (dataDir == null || dataDir.trim().isEmpty()) {
            dataDir = DEFAULT_DATA_DIR;
        }
        // Set base event log directory path (app ID will be appended per session)
        this.eventLogBaseDir = dataDir + "/spark-events";
        LOG.info("Event log base directory configured: " + eventLogBaseDir);

        // Set up session expiration callback to trigger releaseSession
        sessionManager.setOnSessionExpiredCallback(info -> {
            String connectId = info.getConnectId();
            String userId = info.getUserId();
            String sessionId = info.getSessionId();
            String eventLogDir = info.getEventLogDir();

            // Clear suggested configs on expiration
            info.setSuggestedConfs(null);
            info.setSuggestedConfigsApplied(true);

            LOG.info(String.format("[GrpcGateway] Session expired, releasing: connectId=%s, userId=%s, sessionId=%s",
                    connectId, userId, sessionId));

            // Trigger releaseSession on the router
            router.releaseSession(connectId, userId, sessionId, eventLogDir);
        });

        sessionManager.setOnJobFinishedCallback(jobInfo -> {
            router.releaseJob(jobInfo.getConnectId(), jobInfo.getUserId(), jobInfo.getSessionIds(), jobInfo.getEventLogDir());
        });
    }

    /**
     * Fetches multiple Spark configurations from the upstream Spark Connect server.
     * Uses executeUnaryCall pattern to properly maintain session mappings.
     *
     * @param session the session information containing service stub and session info
     * @param keys the configuration keys to retrieve
     * @return a map of configuration key-value pairs
     */
    private Map<String, String> getSparkConfigs(SessionInfo session, String... keys) {
        Map<String, String> configMap = new ConcurrentHashMap<>();
        
        if (keys == null || keys.length == 0) {
            LOG.warning("No configuration keys provided to getSparkConfigs");
            return configMap;
        }

        try {
            // Build a config request to get the specified keys
            var configReqBuilder = ConfigRequest.newBuilder()
                    .setSessionId(session.getSessionId())
                    .setUserContext(UserContext.newBuilder().setUserId(session.getUserId()).build())
                    .setOperation(
                            ConfigRequest.Operation.newBuilder()
                                    .setGet(
                                            ConfigRequest.Get.newBuilder()
                                                    .addAllKeys(java.util.Arrays.asList(keys))
                                                    .build()
                                    ).build()
                    );

            final var finalRequest = configReqBuilder.build();

            // Create an observer to capture the response
            var observer = new StreamObserver<ConfigResponse>() {
                @Override
                public void onNext(ConfigResponse response) {
                    // Extract all key-value pairs from response
                    for (KeyValue kv : response.getPairsList()) {
                        configMap.put(kv.getKey(), kv.getValue());
                        LOG.fine(String.format("Retrieved config: %s = %s", kv.getKey(), kv.getValue()));
                    }
                }

                @Override
                public void onError(Throwable throwable) {
                    LOG.log(Level.WARNING, "Failed to retrieve configurations: " + java.util.Arrays.toString(keys), throwable);
                }

                @Override
                public void onCompleted() {
                }
            };

            // Use executeUnaryCall to get configs
            executeUnaryCall(session, "GetSparkConfigs",
                    () -> session.getService().config(finalRequest),
                    observer);

            // Log any missing configs
            for (String key : keys) {
                if (!configMap.containsKey(key)) {
                    LOG.warning(String.format("Configuration key not found in response: %s", key));
                }
            }
            
            return configMap;
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Failed to retrieve configurations: " + java.util.Arrays.toString(keys), e);
            return configMap;
        }
    }

    private record ServiceInfo(
            String appId,
            String eventLog,
            long sessionTimeout
    ) {}

    /**
     * Returns (and caches) information about the upstream Spark Connect service associated with the given session.
     *
     * Computes the Spark application ID, event log directory, and session timeout for this service,
     * then stores them in a cache for future use. The information is retrieved from the Spark configuration
     * of the remote service using getSparkConfigs().
     *
     * This method is thread-safe; only one thread will compute and cache ServiceInfo for a given service at a time.
     *
     * @param session the SessionInfo containing the upstream service stub and headers/identifiers
     * @return ServiceInfo containing appId, event log path, and session timeout for the service
     */
    private synchronized ServiceInfo getServiceInfo(SessionInfo session) {
        if (!serviceInfos.containsKey(session.getService())) {
            var configs = getSparkConfigs(session, "spark.app.id");
            var appId = configs.getOrDefault("spark.app.id", "");
            var eventLog = eventLogBaseDir + "/eventlog_v2_" + appId;
            var defaultSessionTimeout = "5m";
            var timeout = Utils.timeStringAsMs(defaultSessionTimeout);
            LOG.info(String.format("[getServiceInfo] Determined appId: %s, eventLog: %s, sessionTimeout: %s(%ds) for service: %s",
                    appId, eventLog, defaultSessionTimeout, timeout, session.getService()));
            serviceInfos.put(session.getService(), new ServiceInfo(appId, eventLog, timeout));
        }
        return serviceInfos.get(session.getService());
    }

    /**
     * Creates or retrieves cached SessionInfo from common request fields.
     *
     * <p>This method first checks if a cached SessionInfo exists for the given session ID
     * via SessionManager. If found, it returns the cached session. Otherwise, it performs 
     * the following steps to build a new session:</p>
     * <ol>
     *   <li>Extracts the unique ID from the gRPC context (set by the auth interceptor)</li>
     *   <li>Extracts the cluster ID from the gRPC context (set by the auth interceptor)</li>
     *   <li>Determines which upstream Spark Connect server should handle this request
     *       based on connectId, userId, sessionId, and clusterId</li>
     *   <li>Tracks the session in SessionManager for expiration management</li>
     * </ol>
     *
     * @param sessionId The client-side session ID from the incoming gRPC request
     * @param userId    The user ID from the request's UserContext
     * @return A fully populated SessionInfo containing all information needed
     * to route and process the request
     */
    private SessionInfo createRequestContext(String sessionId, String userId) {
        Optional<SessionInfo> closedSession = sessionManager.getClosedSessionInfo(sessionId);
        if (closedSession.isPresent()) {
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[createRequestContext] Using a closed session for sessionId=%s", sessionId));
            }
            return closedSession.get();
        }

        // Check if we have a cached session
        Optional<SessionInfo> cachedSession = sessionManager.getSessionInfo(sessionId);
        if (cachedSession.isPresent()) {
            // Update access time for session expiration tracking
            SessionInfo session = cachedSession.get();
            session.updateAccessTime();
            
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[createRequestContext] Using cached session for sessionId=%s", sessionId));
            }
            return session;
        }
        
        // No cached session found, create a new one
        // Step 1: Extract unique ID from gRPC context (set by auth interceptor)
        String connectId = CONNECT_ID_CONTEXT_KEY.get();

        // Step 2: Extract cluster ID from gRPC context (set by auth interceptor)
        String clusterId = CLUSTER_ID_CONTEXT_KEY.get();

        // Step 3: Determine which upstream server should handle this request
        var determination = router.determineService(userId, sessionId, clusterId, connectId);
        var service = determination.service();

        // Step 4: Create a temporary session to get service info
        SessionInfo tempSession = new SessionInfo(connectId, userId, sessionId, clusterId, service,
                "", 0, determination.configurations(), false);
        var serviceInfo = getServiceInfo(tempSession);
        
        // Step 5: Create the actual session with proper eventLogDir and timeout
        SessionInfo session = new SessionInfo(connectId, userId, sessionId, clusterId, service,
                serviceInfo.eventLog(), serviceInfo.sessionTimeout(),
                determination.configurations(), false);
        
        // Step 6: Track session for expiration management
        sessionManager.trackSession(session);
        
        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[createRequestContext] Created new session for sessionId=%s", sessionId));
        }
        
        return session;
    }

    /**
     * Executes a unary gRPC call, passing through responses as-is from the upstream server.
     *
     * @param session          The session information
     * @param methodName       The API method name
     * @param serviceCall      Supplier that makes the actual service call
     * @param responseObserver The gRPC response observer
     */
    private <T> void executeUnaryCall(
            SessionInfo session,
            String methodName,
            Supplier<T> serviceCall,
            StreamObserver<T> responseObserver) {

        LOG.info(String.format("[%s] Request: connectId=%s, sessionId=%s, service=%s",
                methodName, session.getConnectId(), session.getSessionId(), session.getService().getChannel().authority()));

        try {
            T response = serviceCall.get();

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[%s] Response sent: sessionId=%s", methodName, session.getSessionId()));
            }
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOG.log(Level.SEVERE, String.format("[%s] Error: sessionId=%s", methodName, session.getSessionId()), e);
            responseObserver.onError(e);
        }
    }

    /**
     * Executes a streaming gRPC call, passing through responses as-is from the upstream server.
     *
     * <p>This method handles streaming API calls (like ExecutePlan and ReattachExecute)
     * where the upstream server returns multiple response messages.</p>
     *
     * @param session          The session information containing routing and session information
     * @param methodName       The API method name
     * @param serviceCall      Supplier that executes the upstream call and returns a response iterator
     * @param responseObserver The gRPC StreamObserver for sending responses back to the client
     */
    private void executeStreamingCall(
            SessionInfo session,
            String methodName,
            Supplier<Iterator<ExecutePlanResponse>> serviceCall,
            StreamObserver<ExecutePlanResponse> responseObserver) {

        LOG.info(String.format("[%s] Streaming request: connectId=%s, sessionId=%s, service=%s",
                methodName, session.getConnectId(), session.getSessionId(), session.getService().getChannel().authority()));

        try {
            // Execute upstream service call to get the response iterator
            Iterator<ExecutePlanResponse> responses = serviceCall.get();

            int responseCount = 0;

            // Stream each response back to the client as-is
            while (responses.hasNext()) {
                ExecutePlanResponse response = responses.next();
                responseCount++;

                if (responseCount == 1 && LOG.isLoggable(Level.FINE)) {
                    LOG.fine(String.format("[%s] Streaming response #%d: sessionId=%s",
                            methodName, responseCount, session.getSessionId()));
                }
                responseObserver.onNext(response);
            }

            LOG.info(String.format("[%s] Streaming completed: sessionId=%s, totalResponses=%d",
                    methodName, session.getSessionId(), responseCount));
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOG.log(Level.SEVERE, String.format("[%s] Streaming error: sessionId=%s", methodName, session.getSessionId()), e);
            responseObserver.onError(e);
        }
    }

    private class ProxyService extends SparkConnectServiceGrpc.SparkConnectServiceImplBase {

        /**
         * Applies suggested Spark configurations to a session if not already applied.
         * This is called once per session before the first query execution.
         */
        private void setSuggestedConfigurations(SessionInfo session) {
        // Ensure there's at least 1 recommended spark configuration.
        if (!session.isSuggestedConfigsApplied() && !session.getSuggestedConfs().isEmpty()) {
            LOG.info("setSuggestedConfigurations " + session.getSuggestedConfs());

            // Convert Map<String, String> to List<KeyValue>
                var keyValueList = session.getSuggestedConfs().entrySet().stream()
                        .map(e -> KeyValue.newBuilder().setKey(e.getKey()).setValue(e.getValue()).build())
                        .toList();

            final var finalRequest = ConfigRequest.newBuilder()
                    .setSessionId(session.getSessionId())
                    .setUserContext(UserContext.newBuilder().setUserId(session.getUserId()).build())
                    .setOperation(
                            ConfigRequest.Operation.newBuilder()
                                    .setSet(
                                            ConfigRequest.Set.newBuilder()
                                                    .addAllPairs(keyValueList)
                                                    .build()
                                    ).build()
                    ).build();

                // We don't care about the response, just fake an observer.
                var observer = new StreamObserver<ConfigResponse>() {
                    @Override
                    public void onNext(ConfigResponse configResponse) {
                    }

                    @Override
                    public void onError(Throwable throwable) {
                    }

                    @Override
                    public void onCompleted() {
                    }
                };

                executeUnaryCall(session, "Config", () -> session.getService().config(finalRequest),
                        observer);

                // Update the session with suggestedConfigsApplied = true
                session.setSuggestedConfigsApplied(true);
            }
        }

        @Override
        public void executePlan(ExecutePlanRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            setSuggestedConfigurations(session);

            executeStreamingCall(session, "ExecutePlan",
                    () -> session.getService().executePlan(request),
                    responseObserver);
        }

        @Override
        public void analyzePlan(AnalyzePlanRequest request, StreamObserver<AnalyzePlanResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            executeUnaryCall(session, "AnalyzePlan",
                    () -> session.getService().analyzePlan(request),
                    responseObserver);
        }

        @Override
        public void config(ConfigRequest request, StreamObserver<ConfigResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            executeUnaryCall(session, "Config",
                    () -> session.getService().config(request),
                    responseObserver);
        }

        @Override
        public StreamObserver<AddArtifactsRequest> addArtifacts(StreamObserver<AddArtifactsResponse> responseObserver) {
            return super.addArtifacts(responseObserver);
        }

        @Override
        public void artifactStatus(ArtifactStatusesRequest request, StreamObserver<ArtifactStatusesResponse> responseObserver) {
            super.artifactStatus(request, responseObserver);
        }

        @Override
        public void interrupt(InterruptRequest request, StreamObserver<InterruptResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            executeUnaryCall(session, "Interrupt",
                    () -> session.getService().interrupt(request),
                    responseObserver);
        }

        @Override
        public void reattachExecute(ReattachExecuteRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            executeStreamingCall(session, "ReattachExecute",
                    () -> session.getService().reattachExecute(request),
                    responseObserver);
        }

        @Override
        public void releaseExecute(ReleaseExecuteRequest request, StreamObserver<ReleaseExecuteResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            executeUnaryCall(session, "ReleaseExecute",
                    () -> session.getService().releaseExecute(request),
                    responseObserver);
        }

        @Override
        public void releaseSession(ReleaseSessionRequest request, StreamObserver<ReleaseSessionResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            String eventLogDir = session.getEventLogDir();
            LOG.info(String.format("[ReleaseSession] sessionId=%s, eventLogDir=%s", session.getSessionId(), eventLogDir));

            router.releaseSession(session.getConnectId(), session.getUserId(), session.getSessionId(), eventLogDir);

            // Remove session from tracking (explicit release, no need to wait for expiration)
            sessionManager.removeSession(session.getSessionId());

            executeUnaryCall(session, "ReleaseSession",
                    () -> session.getService().releaseSession(request),
                    responseObserver);
        }

        @Override
        public void fetchErrorDetails(FetchErrorDetailsRequest request, StreamObserver<FetchErrorDetailsResponse> responseObserver) {
            var session = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());
            executeUnaryCall(session, "FetchErrorDetails",
                    () -> session.getService().fetchErrorDetails(request),
                    responseObserver);
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

                // Extract connect_id from HTTP header
                String connectId = headers.get(CONNECT_ID_METADATA_KEY);
                // Extract cluster_id from HTTP header
                String clusterId = headers.get(CLUSTER_ID_METADATA_KEY);

                // Create new context with connect_id and cluster_id attached
                Context context = Context.current()
                        .withValue(CONNECT_ID_CONTEXT_KEY, Optional.ofNullable(connectId).orElse(""))
                        .withValue(CLUSTER_ID_CONTEXT_KEY, Optional.ofNullable(clusterId).orElse(""));

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
        LOG.info("GrpcGateway proxy server started on port 15002");
        server.awaitTermination();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        Logger proxyLogger = Logger.getLogger("GRPCGateway");
        proxyLogger.setLevel(Level.FINE);

        // Also set console handler to show FINE level
        for (Handler handler : Logger.getLogger("").getHandlers()) {
            handler.setLevel(Level.FINE);
        }
        new GrpcGateway().start();
    }

}
