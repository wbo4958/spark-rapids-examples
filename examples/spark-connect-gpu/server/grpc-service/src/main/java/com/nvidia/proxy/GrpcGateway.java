package com.nvidia.proxy;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
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

    private static final Context.Key<String> JOB_ID_CONTEXT_KEY = Context.key("job_id");
    private static final Metadata.Key<String> JOB_ID_METADATA_KEY =
            Metadata.Key.of("job_id", Metadata.ASCII_STRING_MARSHALLER);

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
    
    /**
     * Cache of RequestContext objects indexed by session ID.
     *
     * Maps session IDs to their corresponding RequestContext objects to avoid recreating
     * the context for each request from the same session. This improves performance by
     * reusing routing information, service stubs, and other metadata that remain constant
     * throughout a session's lifetime.
     *
     * The cache is thread-safe and automatically managed - entries are created on first
     * access and can be invalidated when sessions are released.
     */
    private final Map<String, RequestContext> requestContextCache = new ConcurrentHashMap<>();

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
            String jobId = info.getJobId();
            String userId = info.getUserId();
            String sessionId = info.getSessionId();
            String eventLogDir = info.getEventLogDir();

            // Session timeout, but the client side is not aware of this, so the session is still
            // available from the client point view. We need to route to the right connect server
            // to raise the exception.
            // TODO, create a set to store the expired sessions, and throw exception when creating request context.
            // requestContextCache.remove(sessionId);

            LOG.info(String.format("[GrpcGateway] Session expired, releasing: jobId=%s, userId=%s, sessionId=%s",
                    jobId, userId, sessionId));

            // Trigger releaseSession on the router
            router.releaseSession(jobId, userId, sessionId, eventLogDir);
        });
    }

    /**
     * Fetches the Spark application ID from the upstream Spark Connect server.
     * Uses executeUnaryCall pattern to properly maintain session mappings.
     *
     * @param ctx the request context containing service stub and session info
     * @return the application ID, or empty string if not available
     */
    private String getSparkAppId(RequestContext ctx) {
        var configs = getSparkConfigs(ctx, "spark.app.id");
        return configs.getOrDefault("spark.app.id", "");
    }

    /**
     * Fetches multiple Spark configurations from the upstream Spark Connect server.
     * Uses executeUnaryCall pattern to properly maintain session mappings.
     *
     * @param ctx the request context containing service stub and session info
     * @param keys the configuration keys to retrieve
     * @return a map of configuration key-value pairs
     */
    private Map<String, String> getSparkConfigs(RequestContext ctx, String... keys) {
        Map<String, String> configMap = new ConcurrentHashMap<>();
        
        if (keys == null || keys.length == 0) {
            LOG.warning("No configuration keys provided to getSparkConfigs");
            return configMap;
        }

        try {
            // Build a config request to get the specified keys
            var configReqBuilder = ConfigRequest.newBuilder()
                    .setSessionId(ctx.sessionId())
                    .setUserContext(UserContext.newBuilder().setUserId(ctx.userId()).build())
                    .setOperation(
                            ConfigRequest.Operation.newBuilder()
                                    .setGet(
                                            ConfigRequest.Get.newBuilder()
                                                    .addAllKeys(java.util.Arrays.asList(keys))
                                                    .build()
                                    ).build()
                    );

            // Set clientObservedServerSideSessionId if available
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    configReqBuilder::setClientObservedServerSideSessionId,
                    configReqBuilder::clearClientObservedServerSideSessionId
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

            // Use executeUnaryCall to maintain session mappings
            executeUnaryCall(ctx, "GetSparkConfigs",
                    () -> ctx.service().config(finalRequest),
                    ConfigResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
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
     * Returns (and caches) information about the upstream Spark Connect service associated with the given request context.
     *
     * Computes the Spark application ID, event log directory, and session timeout for this service,
     * then stores them in a cache for future use. The information is retrieved from the Spark configuration
     * of the remote service using getSparkConfigs().
     *
     * This method is thread-safe; only one thread will compute and cache ServiceInfo for a given service at a time.
     *
     * @param ctx the RequestContext containing the upstream service stub and headers/identifiers
     * @return ServiceInfo containing appId, event log path, and session timeout for the service
     */
    private synchronized ServiceInfo getServiceInfo(RequestContext ctx) {
        if (!serviceInfos.containsKey(ctx.service())) {
            var configs = getSparkConfigs(ctx, "spark.app.id", "spark.connect.session.manager.defaultSessionTimeout");
            var appId = configs.getOrDefault("spark.app.id", "");
            var eventLog = eventLogBaseDir + "/eventlog_v2_" + appId;
            var sessionTimeoutStr = configs.getOrDefault("spark.connect.session.manager.defaultSessionTimeout", "60m");
            var timeout = Utils.timeStringAsMs(sessionTimeoutStr);
            LOG.info(String.format("[getServiceInfo] Determined appId: %s, eventLog: %s, sessionTimeoutStr: %s(%ds) for service: %s",
                    appId, eventLog, sessionTimeoutStr, timeout, ctx.service()));
            serviceInfos.put(ctx.service(), new ServiceInfo(appId, eventLog, timeout));
        }
        return serviceInfos.get(ctx.service());
    }

    /**
     * Holds common request context information extracted from gRPC requests.
     * This record encapsulates all the information needed to process a request
     * and route it to the appropriate upstream Spark Connect server.
     *
     * <p>The context is created at the beginning of each API call and passed
     * through the request processing pipeline.</p>
     *
     * @param userId                            The user ID from the request's UserContext.
     *                                          Identifies who is making the request.
     * @param sessionId                         The client-side session ID from the incoming request.
     *                                          This is generated by the Spark Connect client.
     * @param clusterId                         The cluster ID extracted from gRPC metadata headers.
     *                                          Used to select CPU or GPU service ("cpu" or "gpu").
     * @param jobId                             The unique identifier extracted from gRPC metadata headers.
     *                                          Used for routing requests to the appropriate upstream server.
     * @param service                           The upstream Spark Connect service stub determined by the router.
     *                                          All API calls will be forwarded to this service.
     * @param clientObservedServerSideSessionId The previously observed server-side session ID for this client session.
     *                                          If present, it will be set on outgoing requests to maintain
     *                                          session consistency with the upstream server.
     *                                          Empty if this is the first request for the session.
     * @param suggestedConfs                    The suggested Spark configurations from the plugin.
     */
    private record RequestContext(
            String userId,
            String sessionId,
            String clusterId,
            String jobId,
            SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service,
            Optional<String> clientObservedServerSideSessionId,
            Map<String, String> suggestedConfs,
            boolean confIsSet
    ) {
    }

    /**
     * Creates or retrieves a cached RequestContext from common request fields.
     *
     * <p>This method first checks if a cached RequestContext exists for the given session ID.
     * If found, it updates the clientObservedServerSideSessionId and returns the cached context.
     * Otherwise, it performs the following steps to build a new context:</p>
     * <ol>
     *   <li>Extracts the unique ID from the gRPC context (set by the auth interceptor)</li>
     *   <li>Extracts the cluster ID from the gRPC context (set by the auth interceptor)</li>
     *   <li>Determines which upstream Spark Connect server should handle this request
     *       based on jobId, userId, sessionId, and clusterId</li>
     *   <li>Looks up any previously observed server-side session ID for this client session,
     *       which will be used to maintain session consistency with the upstream server</li>
     *   <li>Caches the created context for future requests</li>
     * </ol>
     *
     * @param sessionId The client-side session ID from the incoming gRPC request
     * @param userId    The user ID from the request's UserContext
     * @return A fully populated RequestContext containing all information needed
     * to route and process the request
     */
    private RequestContext createRequestContext(String sessionId, String userId) {
        // Check if we have a cached context for this session
        RequestContext cachedContext = requestContextCache.get(sessionId);

        if (cachedContext != null && cachedContext.clientObservedServerSideSessionId.isEmpty()) {
            // Update the clientObservedServerSideSessionId as it may have changed
            Optional<String> clientObservedServerSideSessionId = 
                    sessionManager.getServerSideSessionId(sessionId, cachedContext.service());
            
            // Create an updated context with the latest clientObservedServerSideSessionId
            var updatedContext = new RequestContext(
                    cachedContext.userId(),
                    cachedContext.sessionId(),
                    cachedContext.clusterId(),
                    cachedContext.jobId(),
                    cachedContext.service(),
                    clientObservedServerSideSessionId,
                    cachedContext.suggestedConfs(),
                    cachedContext.confIsSet());
            
            // Update the cache with the refreshed context
            requestContextCache.put(sessionId, updatedContext);
            
            // Track session for expiration management (updates access time if exists)
            var serviceInfo = getServiceInfo(updatedContext);
            sessionManager.trackSession(cachedContext.jobId(), userId, sessionId, 
                    serviceInfo.eventLog(), serviceInfo.sessionTimeout());
            
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[createRequestContext] Using cached context for sessionId=%s", sessionId));
            }
            return updatedContext;
        }
        
        // No cached context found, create a new one
        // Step 1: Extract unique ID from gRPC context (set by auth interceptor)
        String jobId = JOB_ID_CONTEXT_KEY.get();

        // Step 2: Extract cluster ID from gRPC context (set by auth interceptor)
        String clusterId = CLUSTER_ID_CONTEXT_KEY.get();

        // Step 3: Determine which upstream server should handle this request
        var determination = router.determineService(userId, sessionId, clusterId, jobId);
        var service = determination.service();

        // Step 4: Look up previously observed server-side session ID for session consistency
        Optional<String> clientObservedServerSideSessionId = sessionManager.getServerSideSessionId(sessionId, service);

        var context = new RequestContext(userId, sessionId, clusterId, jobId, service,
                clientObservedServerSideSessionId, determination.configurations(), false);
        var serviceInfo = getServiceInfo(context);
        // Step 5: Track session for expiration management (updates access time if exists)
        sessionManager.trackSession(jobId, userId, sessionId, serviceInfo.eventLog(), serviceInfo.sessionTimeout());
        
        // Step 6: Cache the context for future requests
        requestContextCache.put(sessionId, context);
        
        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[createRequestContext] Created new context for sessionId=%s", sessionId));
        }
        
        return context;
    }

    /**
     * Resolves and stores the serverSideSessionId, returning the consistent session ID.
     *
     * <p>This method ensures that the same client session always receives the same
     * server-side session ID, even if the upstream server returns different IDs
     * across multiple requests. This is important for session consistency in a
     * proxy environment where the client expects a stable session identifier.</p>
     *
     * <p>The resolution process:</p>
     * <ol>
     *   <li>Store the mapping between client session ID and server session ID
     *       (the first stored value wins for consistency)</li>
     *   <li>Retrieve the consistent session ID from the session manager</li>
     * </ol>
     *
     * @param clientSessionId     The client-side session ID from the original request
     * @param serverSideSessionId The server-side session ID returned by the upstream server
     * @param service             The upstream Spark Connect service that returned the response
     * @param methodName          The API method name
     * @return The consistent session ID that should be returned to the client.
     * This may differ from serverSideSessionId if a different ID was
     * previously stored for this client session.
     */
    private String resolveSessionId(String clientSessionId, String serverSideSessionId,
                                    SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service,
                                    String methodName) {
        // Store the session ID mapping (first stored value wins for consistency)
        sessionManager.storeSessionIds(clientSessionId, serverSideSessionId, service);

        // Retrieve the consistent session ID (may be different from serverSideSessionId)
        String consistentSessionId = sessionManager.getServerSideSessionId(clientSessionId, serverSideSessionId);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[%s] Session ID resolved: clientSessionId=%s, serverSideSessionId=%s, consistentSessionId=%s",
                    methodName, clientSessionId, serverSideSessionId, consistentSessionId));
        }

        return consistentSessionId;
    }

    /**
     * Executes a unary gRPC call with common session ID handling.
     *
     * @param ctx              The request context
     * @param methodName       The API method name
     * @param serviceCall      Supplier that makes the actual service call
     * @param getSessionId     Function to extract serverSideSessionId from response
     * @param updateResponse   Function to update response with consistent session ID
     * @param responseObserver The gRPC response observer
     */
    private <T> void executeUnaryCall(
            RequestContext ctx,
            String methodName,
            Supplier<T> serviceCall,
            Function<T, String> getSessionId,
            java.util.function.BiFunction<T, String, T> updateResponse,
            StreamObserver<T> responseObserver) {

        LOG.info(String.format("[%s] Request: jobId=%s, sessionId=%s, service=%s",
                methodName, ctx.jobId(), ctx.sessionId(), ctx.service().getChannel().authority()));

        try {
            T response = serviceCall.get();
            String serverSideSessionId = getSessionId.apply(response);
            String consistentSessionId = resolveSessionId(ctx.sessionId(), serverSideSessionId, ctx.service(), methodName);
            response = updateResponse.apply(response, consistentSessionId);

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[%s] Response sent: sessionId=%s", methodName, ctx.sessionId()));
            }
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOG.log(Level.SEVERE, String.format("[%s] Error: sessionId=%s", methodName, ctx.sessionId()), e);
            responseObserver.onError(e);
        }
    }

    /**
     * Executes a streaming gRPC call with common session ID handling.
     *
     * <p>This method handles streaming API calls (like ExecutePlan and ReattachExecute)
     * where the upstream server returns multiple response messages. It ensures that
     * all responses in the stream use the same consistent server-side session ID.</p>
     *
     * <p>The streaming call flow:</p>
     * <ol>
     *   <li>Execute the upstream service call to obtain a response iterator</li>
     *   <li>For the first response, resolve and store the server-side session ID</li>
     *   <li>For all responses (including first), update with the consistent session ID</li>
     *   <li>Stream each updated response back to the client</li>
     *   <li>Signal completion when all responses have been sent</li>
     *   <li>If any error occurs, propagate it to the client via onError</li>
     * </ol>
     *
     * <p><b>Important:</b> The session ID is resolved only from the first response
     * and reused for all subsequent responses. This ensures the client sees a
     * consistent session ID throughout the entire streaming response.</p>
     *
     * @param ctx              The request context containing routing and session information
     * @param methodName       The API method name
     * @param serviceCall      Supplier that executes the upstream call and returns a response iterator
     * @param responseObserver The gRPC StreamObserver for sending responses back to the client
     */
    private void executeStreamingCall(
            RequestContext ctx,
            String methodName,
            Supplier<Iterator<ExecutePlanResponse>> serviceCall,
            StreamObserver<ExecutePlanResponse> responseObserver) {

        LOG.info(String.format("[%s] Streaming request: jobId=%s, sessionId=%s, service=%s",
                methodName, ctx.jobId(), ctx.sessionId(), ctx.service().getChannel().authority()));

        try {
            // Execute upstream service call to get the response iterator
            Iterator<ExecutePlanResponse> responses = serviceCall.get();

            // Track consistent session ID (resolved from first response only)
            String consistentSessionId = null;
            int responseCount = 0;

            // Stream each response back to the client
            while (responses.hasNext()) {
                ExecutePlanResponse response = responses.next();
                responseCount++;

                // Resolve session ID from the first response only
                if (consistentSessionId == null) {
                    consistentSessionId = resolveSessionId(
                            ctx.sessionId(), response.getServerSideSessionId(), ctx.service(), methodName);
                }

                // Update response with consistent session ID and send to client
                response = response.toBuilder().setServerSideSessionId(consistentSessionId).build();
                if (responseCount == 1 && LOG.isLoggable(Level.FINE)) {
                    LOG.fine(String.format("[%s] Streaming response #%d: sessionId=%s",
                            methodName, responseCount, ctx.sessionId()));
                }
                responseObserver.onNext(response);
            }

            LOG.info(String.format("[%s] Streaming completed: sessionId=%s, totalResponses=%d",
                    methodName, ctx.sessionId(), responseCount));
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOG.log(Level.SEVERE, String.format("[%s] Streaming error: sessionId=%s", methodName, ctx.sessionId()), e);
            responseObserver.onError(e);
        }
    }

    private class ProxyService extends SparkConnectServiceGrpc.SparkConnectServiceImplBase {

        // TODO, set the Spark Configurations only once for a session
        private void setSuggestedConfigurations(RequestContext ctx) {
            // Ensure there's at least 1 recommended spark configuration.
            if (!ctx.suggestedConfs().isEmpty() && !ctx.confIsSet()) {
                // Convert Map<String, String> to List<KeyValue>
                var keyValueList = ctx.suggestedConfs().entrySet().stream()
                        .map(e -> KeyValue.newBuilder().setKey(e.getKey()).setValue(e.getValue()).build())
                        .toList();

                var configReq =
                        ConfigRequest.newBuilder()
                                .setSessionId(ctx.sessionId())
                                .setUserContext(UserContext.newBuilder().setUserId(ctx.userId()).build())
                                .setOperation(
                                        ConfigRequest.Operation.newBuilder()
                                                .setSet(
                                                        ConfigRequest.Set.newBuilder()
                                                                .addAllPairs(keyValueList)
                                                                .build()
                                                ).build()
                                ).build();

                // Set clientObservedServerSideSessionId on request
                var builder = configReq.toBuilder();
                ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                        builder::setClientObservedServerSideSessionId,
                        builder::clearClientObservedServerSideSessionId
                );

                final var finalRequest = builder.build();

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

                executeUnaryCall(ctx, "Config", () -> ctx.service().config(finalRequest),
                        ConfigResponse::getServerSideSessionId,
                        (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
                        observer);

                // Create an updated context with the latest true confIsSet
                var updatedContext = new RequestContext(
                        ctx.userId(),
                        ctx.sessionId(),
                        ctx.clusterId(),
                        ctx.jobId(),
                        ctx.service(),
                        ctx.clientObservedServerSideSessionId(),
                        ctx.suggestedConfs(),
                        true);

                // Update the cache with the refreshed context
                requestContextCache.put(ctx.sessionId(), updatedContext);
            }
        }

        @Override
        public void executePlan(ExecutePlanRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            setSuggestedConfigurations(ctx);

            // Set clientObservedServerSideSessionId on request
            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeStreamingCall(ctx, "ExecutePlan",
                    () -> ctx.service().executePlan(finalRequest),
                    responseObserver);
        }

        @Override
        public void releaseSession(ReleaseSessionRequest request, StreamObserver<ReleaseSessionResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            var serviceInfo = getServiceInfo(ctx);
            String eventLogDir = serviceInfo.eventLog();
            LOG.info(String.format("[ReleaseSession] sessionId=%s, eventLogDir=%s", ctx.sessionId(), eventLogDir));

            // Remove session from tracking (explicit release, no need to wait for expiration)
            sessionManager.removeSession(ctx.sessionId());
            
            // Remove the cached RequestContext for this session
            requestContextCache.remove(ctx.sessionId());
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[ReleaseSession] Removed cached context for sessionId=%s", ctx.sessionId()));
            }

            router.releaseSession(ctx.jobId(), ctx.userId(), ctx.sessionId(), eventLogDir);

            executeUnaryCall(ctx, "ReleaseSession",
                    () -> ctx.service().releaseSession(request),
                    ReleaseSessionResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
                    responseObserver);
        }


        @Override
        public void analyzePlan(AnalyzePlanRequest request, StreamObserver<AnalyzePlanResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            // Set clientObservedServerSideSessionId on request
            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeUnaryCall(ctx, "AnalyzePlan",
                    () -> ctx.service().analyzePlan(finalRequest),
                    AnalyzePlanResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
                    responseObserver);
        }

        @Override
        public void config(ConfigRequest request, StreamObserver<ConfigResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeUnaryCall(ctx, "Config",
                    () -> ctx.service().config(finalRequest),
                    ConfigResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
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
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeUnaryCall(ctx, "Interrupt",
                    () -> ctx.service().interrupt(finalRequest),
                    InterruptResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
                    responseObserver);
        }

        @Override
        public void reattachExecute(ReattachExecuteRequest request, StreamObserver<ExecutePlanResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            // Set clientObservedServerSideSessionId on request
            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeStreamingCall(ctx, "ReattachExecute",
                    () -> ctx.service().reattachExecute(finalRequest),
                    responseObserver);
        }

        @Override
        public void releaseExecute(ReleaseExecuteRequest request, StreamObserver<ReleaseExecuteResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeUnaryCall(ctx, "ReleaseExecute",
                    () -> ctx.service().releaseExecute(finalRequest),
                    ReleaseExecuteResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
                    responseObserver);
        }

        @Override
        public void fetchErrorDetails(FetchErrorDetailsRequest request, StreamObserver<FetchErrorDetailsResponse> responseObserver) {
            var ctx = createRequestContext(request.getSessionId(), request.getUserContext().getUserId());

            var builder = request.toBuilder();
            ctx.clientObservedServerSideSessionId().ifPresentOrElse(
                    builder::setClientObservedServerSideSessionId,
                    builder::clearClientObservedServerSideSessionId
            );
            final var finalRequest = builder.build();

            executeUnaryCall(ctx, "FetchErrorDetails",
                    () -> ctx.service().fetchErrorDetails(finalRequest),
                    FetchErrorDetailsResponse::getServerSideSessionId,
                    (resp, sessionId) -> resp.toBuilder().setServerSideSessionId(sessionId).build(),
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

                // Extract job_id from HTTP header
                String jobId = headers.get(JOB_ID_METADATA_KEY);
                // Extract cluster_id from HTTP header
                String clusterId = headers.get(CLUSTER_ID_METADATA_KEY);

                // Create new context with job_id and cluster_id attached
                Context context = Context.current()
                        .withValue(JOB_ID_CONTEXT_KEY, Optional.ofNullable(jobId).orElse(""))
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
