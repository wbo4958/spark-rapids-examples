package com.nvidia.proxy;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 1. Client creates session Id
 * 2. Client sends Request with client session id to Server
 * 3. Server creates session Id for the first time.
 * 4. Server sends Response with server session id back to Client.
 * 5. Client stores to server session id.
 *
 * <p>Session Timer Tracking:</p>
 * <ul>
 *   <li>Sessions are tracked with configurable timeout (default: 60 minutes)</li>
 *   <li>Last access time is updated on every session access</li>
 *   <li>When all sessions for a job_id expire, the onSessionExpired callback is triggered</li>
 *   <li>Configure via environment variables: SESSION_TIMEOUT_MS, SESSION_MAINTENANCE_INTERVAL_MS</li>
 * </ul>
 */
public class SessionManager implements Closeable {
    private static final Logger LOG = Logger.getLogger("SessionManager");

    // Default maintenance interval: 30 seconds
    private static final long DEFAULT_MAINTENANCE_INTERVAL_MS = 30 * 1000L;

    public static class JobInfo {
        private final String jobId;
        private final String userId;
        private final Set<String> sessionIds;
        private final String eventLogDir;

        JobInfo(String jobId, String userId, String eventLogDir) {
            this.jobId = jobId;
            this.userId = userId;
            this.eventLogDir = eventLogDir;
            this.sessionIds = new HashSet<>();
        }

        public void addSessions(Set<String> sessions) {
            this.sessionIds.addAll(sessions);
        }

        public String getJobId() {
            return jobId;
        }

        public String getUserId() {
            return userId;
        }

        public Set<String> getSessionIds() {
            return sessionIds;
        }

        public String getEventLogDir() {
            return eventLogDir;
        }
    }
    /**
     * Session metadata containing tracking information.
     * Public to allow external access in callbacks.
     */
    public static class SessionInfo {
        private final String jobId;
        private final String userId;
        private final String sessionId;
        private final String eventLogDir;
        private final long sessionTimeout;
        private volatile long lastAccessTime;

        SessionInfo(String jobId, String userId, String sessionId, String eventLogDir, long sessionTimeout) {
            this.jobId = jobId;
            this.userId = userId;
            this.sessionId = sessionId;
            this.eventLogDir = eventLogDir;
            this.sessionTimeout = sessionTimeout;
            this.lastAccessTime = System.currentTimeMillis();
        }

        void updateAccessTime() {
            this.lastAccessTime = System.currentTimeMillis();
        }

        boolean isExpired() {
            return System.currentTimeMillis() - lastAccessTime > sessionTimeout;
        }

        public String getJobId() {
            return jobId;
        }

        public String getUserId() {
            return userId;
        }

        public String getSessionId() {
            return sessionId;
        }

        public String getEventLogDir() {
            return eventLogDir;
        }

        public long getLastAccessTime() {
            return lastAccessTime;
        }
    }

    /**
     * Maintains a mapping between client-side session IDs and server-side session IDs for Spark Connect.
     *
     * <p>The server-side session ID is generated/updated by the first Spark Connect Server instance
     * that responds to a client request. This mapping ensures we can correlate client session IDs
     * with their corresponding server-side identifiers and return the server-side ID to the client.
     */
    private Map<String, String> clientSessionIdToServerSessionIdMap = new ConcurrentHashMap<>();

    // Map the services to the Map of client session id to server session id.
    private Map<SparkConnectServiceGrpc.SparkConnectServiceBlockingStub, Map<String, String>> serviceToSessionIdMap = new ConcurrentHashMap<>();

    // Map sessionId to SessionInfo for tracking expiration
    private final Map<String, SessionInfo> sessionInfoMap = new ConcurrentHashMap<>();

    // Map jobId to set of sessionIds for tracking when all sessions expire
    private final Map<String, Set<String>> jobIdToSessionIds = new ConcurrentHashMap<>();

    // Map jobId to set of all sessionIds for tracking when all sessions expire
    private final Map<String, Set<String>> jobIdToAllSessionIds = new ConcurrentHashMap<>();

    // Callback to trigger when all sessions for a job expire
    private Consumer<SessionInfo> onSessionExpiredCallback;

    private Consumer<JobInfo> onJobFinishedCallback;

    // Scheduled executor for session maintenance
    private final ScheduledExecutorService scheduler;

    public SessionManager() {
        LOG.setLevel(Level.FINE);
        LOG.info(String.format("[SessionManager] Initialized with maintenanceIntervalMs=%d", DEFAULT_MAINTENANCE_INTERVAL_MS));

        // Initialize scheduler if timeout is enabled (> 0)
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "SessionManager-Maintenance");
            t.setDaemon(true);
            return t;
        });

        // Schedule periodic maintenance task
        scheduler.scheduleAtFixedRate(
                this::performMaintenance,
                DEFAULT_MAINTENANCE_INTERVAL_MS,
                DEFAULT_MAINTENANCE_INTERVAL_MS,
                TimeUnit.MILLISECONDS
        );
        LOG.info("[SessionManager] Session expiration timer started");
    }

    /**
     * Sets the callback to be invoked when a session expires.
     * The callback receives SessionInfo containing jobId, userId, sessionId, and eventLogDir.
     *
     * @param callback the callback to invoke on session expiration
     */
    public void setOnSessionExpiredCallback(Consumer<SessionInfo> callback) {
        this.onSessionExpiredCallback = callback;
    }

    public void setOnJobFinishedCallback(Consumer<JobInfo> callback) {
        this.onJobFinishedCallback = callback;
    }

    /**
     * Register or update a session with its associated job metadata.
     * This should be called when a session is first accessed or on every access to update the timer.
     *
     * @param jobId       the job identifier
     * @param userId      the user identifier
     * @param sessionId   the session identifier
     * @param eventLogDir the event log directory (can be empty initially, will be updated)
     */
    public void trackSession(String jobId, String userId, String sessionId, String eventLogDir, long sessionTimeout) {
        SessionInfo existing = sessionInfoMap.get(sessionId);
        if (existing != null) {
            // Update access time for existing session
            existing.updateAccessTime();
            // Update eventLogDir if provided and different
            if (eventLogDir != null && !eventLogDir.isEmpty() && !eventLogDir.equals(existing.eventLogDir)) {
                // Create new SessionInfo with updated eventLogDir
                SessionInfo updated = new SessionInfo(jobId, userId, sessionId, eventLogDir, sessionTimeout);
                updated.lastAccessTime = existing.lastAccessTime;
                sessionInfoMap.put(sessionId, updated);
            }
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] Updated session access time: sessionId=%s, jobId=%s",
                        sessionId, jobId));
            }
        } else {
            // Create new session tracking
            SessionInfo info = new SessionInfo(jobId, userId, sessionId, eventLogDir, sessionTimeout);
            sessionInfoMap.put(sessionId, info);

            // Track session under jobId
            jobIdToSessionIds
                    .computeIfAbsent(jobId, k -> ConcurrentHashMap.newKeySet())
                    .add(sessionId);
            jobIdToAllSessionIds
                    .computeIfAbsent(jobId, k -> ConcurrentHashMap.newKeySet())
                    .add(sessionId);

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] New session tracked: sessionId=%s, jobId=%s, userId=%s",
                        sessionId, jobId, userId));
            }
        }
    }

    /**
     * Updates the last access time for a session.
     * Call this on every API request to keep the session alive.
     *
     * @param sessionId the session identifier
     */
    public void touchSession(String sessionId) {
        SessionInfo info = sessionInfoMap.get(sessionId);
        if (info != null) {
            info.updateAccessTime();
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] Session touched: sessionId=%s", sessionId));
            }
        }
    }

    /**
     * Manually remove a session (e.g., when releaseSession is called explicitly).
     *
     * @param sessionId the session identifier to remove
     */
    public void removeSession(String sessionId) {
        SessionInfo info = sessionInfoMap.remove(sessionId);
        if (info != null) {
            Set<String> sessions = jobIdToSessionIds.get(info.jobId);
            if (sessions != null) {
                sessions.remove(sessionId);
                if (sessions.isEmpty()) {
                    jobIdToSessionIds.remove(info.jobId);
                    var jobInfo = new JobInfo(info.jobId, info.userId, info.eventLogDir);
                    jobInfo.addSessions(jobIdToAllSessionIds.get(info.jobId));
                    jobIdToAllSessionIds.remove(info.jobId);
                    if (onJobFinishedCallback != null) {
                        onJobFinishedCallback.accept(jobInfo);
                    }
                }
            }
            // Clean up session ID mappings
            clientSessionIdToServerSessionIdMap.remove(sessionId);

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] Session removed: sessionId=%s, jobId=%s",
                        sessionId, info.jobId));
            }
        }
    }

    /**
     * Periodic maintenance task that checks for expired sessions.
     */
    private void performMaintenance() {
        try {
            List<SessionInfo> expiredSessions = new ArrayList<>();

            // Find all expired sessions
            for (Map.Entry<String, SessionInfo> entry : sessionInfoMap.entrySet()) {
                SessionInfo info = entry.getValue();
                if (info.isExpired()) {
                    expiredSessions.add(info);
                }
            }

            // Process expired sessions
            for (SessionInfo expired : expiredSessions) {
                LOG.info(String.format("[SessionManager] Session expired: sessionId=%s, jobId=%s, userId=%s, " +
                                "lastAccessTime=%d, age=%dms",
                        expired.sessionId, expired.jobId, expired.userId,
                        expired.lastAccessTime, System.currentTimeMillis() - expired.lastAccessTime));

                // Remove from tracking
                sessionInfoMap.remove(expired.sessionId);
                clientSessionIdToServerSessionIdMap.remove(expired.sessionId);

                // Trigger callback for job completion
                if (onSessionExpiredCallback != null) {
                    LOG.info(String.format("[SessionManager] All sessions for job expired, triggering releaseSession: jobId=%s",
                            expired.jobId));
                    try {
                        onSessionExpiredCallback.accept(expired);
                    } catch (Exception e) {
                        LOG.log(Level.WARNING,
                                String.format("[SessionManager] Error in session expired callback: jobId=%s", expired.jobId), e);
                    }
                }

                Set<String> jobSessions = jobIdToSessionIds.get(expired.jobId);
                if (jobSessions != null) {
                    jobSessions.remove(expired.sessionId);
                    // Check if all sessions for this job have expired
                    if (jobSessions.isEmpty()) {
                        jobIdToSessionIds.remove(expired.jobId);

                        if (LOG.isLoggable(Level.INFO)) {
                            LOG.info(String.format("[SessionManager] All sessions for jobId=%s have expired. Triggering onJobFinishedCallback.", expired.jobId));
                        }

                        var jobInfo = new JobInfo(expired.jobId, expired.userId, expired.eventLogDir);
                        jobInfo.addSessions(jobIdToAllSessionIds.get(expired.jobId));
                        jobIdToAllSessionIds.remove(expired.jobId);
                        if (onJobFinishedCallback != null) {
                            onJobFinishedCallback.accept(jobInfo);
                        }
                    }
                }
            }

            if (!expiredSessions.isEmpty()) {
                LOG.info(String.format("[SessionManager] Maintenance completed: expired=%d, remaining=%d",
                        expiredSessions.size(), sessionInfoMap.size()));
            }
        } catch (Exception e) {
            LOG.log(Level.WARNING, "[SessionManager] Error during maintenance", e);
        }
    }

    /**
     * Retrieves the server-side session ID associated with the given client-side session ID.
     * If this is the first time this mapping is seen, stores the provided server-side session ID.
     * Always updates session access time.
     *
     * @param clientSideSessionId the client-side session ID
     * @param serverSideSessionId the server-side session ID (as reported by the upstream server)
     * @return the stored (first-mapped) server-side session ID for the client session
     */
    public String getServerSideSessionId(String clientSideSessionId,
                                         String serverSideSessionId) {
        // Update session access time
        touchSession(clientSideSessionId);

        // Only capture the session id from the first response
        if (!clientSessionIdToServerSessionIdMap.containsKey(clientSideSessionId)) {
            clientSessionIdToServerSessionIdMap.put(clientSideSessionId, serverSideSessionId);
        }
        return clientSessionIdToServerSessionIdMap.get(clientSideSessionId);
    }

    /**
     * Get the server side sessionId according to the client side session id and service
     *
     * @param clientSideSessionId client side session id
     * @param service             the connect server
     * @return the server side session id
     */
    public Optional<String> getServerSideSessionId(String clientSideSessionId,
                                                   SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service) {
        // Update session access time
        touchSession(clientSideSessionId);

        Optional<String> result = Optional.ofNullable(serviceToSessionIdMap.get(service))
                .map(sessionIdMap -> sessionIdMap.get(clientSideSessionId));

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[SessionManager] getServerSideSessionId: clientSessionId=%s, found=%s, serverSessionId=%s",
                    clientSideSessionId, result.isPresent(), result.orElse("N/A")));
        }

        return result;
    }

    /**
     * Save the service and the corresponding session ids.
     * TODO: verify if the serverSideSessionId is matching with the previous one.
     *
     * @param clientSideSessionId client side session id
     * @param serverSideSessionId server side session id
     * @param service             the Spark Connect Server
     */
    public void storeSessionIds(String clientSideSessionId,
                                String serverSideSessionId,
                                SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service) {
        // Update session access time
        touchSession(clientSideSessionId);

        Map<String, String> clientToServerSessionIdMap = serviceToSessionIdMap
                .computeIfAbsent(service, k -> new ConcurrentHashMap<>());

        boolean isNewMapping = !clientToServerSessionIdMap.containsKey(clientSideSessionId);
        clientToServerSessionIdMap.put(clientSideSessionId, serverSideSessionId);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[SessionManager] storeSessionIds: clientSessionId=%s, serverSessionId=%s, isNew=%s, totalMappings=%d",
                    clientSideSessionId, serverSideSessionId, isNewMapping, clientToServerSessionIdMap.size()));
        }
    }

    /**
     * Get the session info for a given session ID.
     * Useful for retrieving job metadata when handling expiration.
     *
     * @param sessionId the session identifier
     * @return Optional containing SessionInfo if found
     */
    public Optional<SessionInfo> getSessionInfo(String sessionId) {
        return Optional.ofNullable(sessionInfoMap.get(sessionId));
    }

    /**
     * Get all active session IDs for a given job.
     *
     * @param jobId the job identifier
     * @return set of session IDs, or empty set if none
     */
    public Set<String> getSessionsForJob(String jobId) {
        Set<String> sessions = jobIdToSessionIds.get(jobId);
        return sessions != null ? Collections.unmodifiableSet(new HashSet<>(sessions)) : Collections.emptySet();
    }

    /**
     * Get the count of active tracked sessions.
     *
     * @return number of active sessions
     */
    public int getActiveSessionCount() {
        return sessionInfoMap.size();
    }

    @Override
    public void close() {
        if (scheduler != null) {
            LOG.info("[SessionManager] Shutting down session maintenance scheduler");
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

}
