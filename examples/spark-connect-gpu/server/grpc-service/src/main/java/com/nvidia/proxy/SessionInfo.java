package com.nvidia.proxy;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import java.util.Map;

/**
 * Unified session information class that combines session tracking, routing, and request processing state.
 * 
 * <p>This class serves multiple purposes:
 * <ul>
 *   <li>Session lifecycle tracking and expiration management</li>
 *   <li>Request routing information (which upstream server to use)</li>
 *   <li>Request processing state (suggested configurations, etc.)</li>
 * </ul>
 * 
 * <p>Thread-safety: This class uses volatile fields for mutable state that may be
 * accessed from multiple threads. The service stub is immutable after construction.</p>
 */
public class SessionInfo {
    // Identification fields (immutable)
    private final String connectId;
    private final String userId;
    private final String sessionId;
    private final String clusterId;
    
    // Routing information (immutable)
    private final SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service;
    
    // Expiration tracking fields
    private final String eventLogDir;
    private final long sessionTimeout;
    private volatile long lastAccessTime;
    
    // Mutable request processing state
    private volatile Map<String, String> suggestedConfs;
    private volatile boolean suggestedConfigsApplied;
    
    /**
     * Constructs a new SessionInfo.
     *
     * @param connectId               The unique identifier extracted from gRPC metadata headers.
     * @param userId                  The user ID from the request's UserContext.
     * @param sessionId               The client-side session ID from the incoming request.
     * @param clusterId               The cluster ID extracted from gRPC metadata headers.
     * @param service                 The upstream Spark Connect service stub determined by the router.
     * @param eventLogDir             The event log directory (can be empty initially, will be updated).
     * @param sessionTimeout          The session timeout in milliseconds.
     * @param suggestedConfs          The suggested Spark configurations from the plugin.
     * @param suggestedConfigsApplied Whether the suggested configurations have been applied.
     */
    public SessionInfo(String connectId, String userId, String sessionId, String clusterId,
                      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service,
                      String eventLogDir, long sessionTimeout,
                      Map<String, String> suggestedConfs, boolean suggestedConfigsApplied) {
        this.connectId = connectId;
        this.userId = userId;
        this.sessionId = sessionId;
        this.clusterId = clusterId;
        this.service = service;
        this.eventLogDir = eventLogDir;
        this.sessionTimeout = sessionTimeout;
        this.lastAccessTime = System.currentTimeMillis();
        this.suggestedConfs = suggestedConfs;
        this.suggestedConfigsApplied = suggestedConfigsApplied;
    }
    
    // ============= Expiration Management Methods =============
    
    /**
     * Updates the last access time to the current time.
     * Should be called on every session access to keep the session alive.
     */
    public void updateAccessTime() {
        this.lastAccessTime = System.currentTimeMillis();
    }
    
    /**
     * Checks if this session has expired based on the session timeout.
     *
     * @return true if the session has expired, false otherwise
     */
    public boolean isExpired() {
        return System.currentTimeMillis() - lastAccessTime > sessionTimeout;
    }
    
    // ============= Getters for Immutable Fields =============
    
    public String getConnectId() {
        return connectId;
    }
    
    public String getUserId() {
        return userId;
    }
    
    public String getSessionId() {
        return sessionId;
    }
    
    public SparkConnectServiceGrpc.SparkConnectServiceBlockingStub getService() {
        return service;
    }
    
    public String getEventLogDir() {
        return eventLogDir;
    }
    
    public long getLastAccessTime() {
        return lastAccessTime;
    }
    
    // ============= Getters/Setters for Mutable Fields =============
    
    public Map<String, String> getSuggestedConfs() {
        return suggestedConfs;
    }
    
    public void setSuggestedConfs(Map<String, String> suggestedConfs) {
        this.suggestedConfs = suggestedConfs;
    }
    
    public boolean isSuggestedConfigsApplied() {
        return suggestedConfigsApplied;
    }
    
    public void setSuggestedConfigsApplied(boolean suggestedConfigsApplied) {
        this.suggestedConfigsApplied = suggestedConfigsApplied;
    }
}
