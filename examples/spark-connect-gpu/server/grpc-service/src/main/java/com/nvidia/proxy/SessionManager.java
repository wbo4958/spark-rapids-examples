package com.nvidia.proxy;

import org.sparkproject.connect.guava.cache.Cache;
import org.sparkproject.connect.guava.cache.CacheBuilder;

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
 *   <li>When all sessions for a connect_id expire, the onSessionExpired callback is triggered</li>
 *   <li>Configure via environment variables: SESSION_TIMEOUT_MS, SESSION_MAINTENANCE_INTERVAL_MS</li>
 * </ul>
 */
public class SessionManager implements Closeable {
    private static final Logger LOG = Logger.getLogger("SessionManager");

    // Default maintenance interval: 30 seconds
    private static final long DEFAULT_MAINTENANCE_INTERVAL_MS = 30 * 1000L;

    public static class JobInfo {
        private final String connectId;
        private final String userId;
        private final Set<String> sessionIds;
        private final String eventLogDir;

        JobInfo(String connectId, String userId, String eventLogDir) {
            this.connectId = connectId;
            this.userId = userId;
            this.eventLogDir = eventLogDir;
            this.sessionIds = new HashSet<>();
        }

        public void addSessions(Set<String> sessions) {
            this.sessionIds.addAll(sessions);
        }

        public String getConnectId() {
            return connectId;
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
     * Helper class to track both active and historical sessions for a connectId.
     * Encapsulates the logic for managing session lifecycle.
     */
    private static class ConnectSessions {
        // Currently active sessions that haven't been removed
        private final Set<String> activeSessions = ConcurrentHashMap.newKeySet();
        // All sessions ever created for this connectId (for historical tracking)
        private final Set<String> allSessions = ConcurrentHashMap.newKeySet();
        
        void addSession(String sessionId) {
            activeSessions.add(sessionId);
            allSessions.add(sessionId);
        }
        
        boolean removeActiveSession(String sessionId) {
            return activeSessions.remove(sessionId);
        }
        
        boolean hasActiveSessions() {
            return !activeSessions.isEmpty();
        }
        
        Set<String> getAllSessions() {
            return new HashSet<>(allSessions);
        }
    }

    // Primary map for all session tracking (expiration, metadata, routing info)
    private final Map<String, SessionInfo> sessionStore = new ConcurrentHashMap<>();

    private final Cache<String, SessionInfo> closedSessionsCache = CacheBuilder.newBuilder()
            .maximumSize(10000)
            .build();

    // Single map tracking both active and historical sessions per connectId
    private final Map<String, ConnectSessions> connectIdToSessions = new ConcurrentHashMap<>();

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
     * The callback receives SessionInfo containing connectId, userId, sessionId, and eventLogDir.
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
     * Tracks an existing session or creates a new one if it doesn't exist.
     * For existing sessions, only updates the access time.
     * 
     * <p>Note: This method is called for session tracking/expiration purposes.
     * The full SessionInfo (including service stub and routing info) is managed
     * by the caller and stored in sessionStore if it's a new session.</p>
     *
     * @param sessionInfo the complete session information
     */
    public void trackSession(SessionInfo sessionInfo) {
        String sessionId = sessionInfo.getSessionId();
        SessionInfo existing = sessionStore.get(sessionId);
        
        if (existing != null) {
            // Update access time for existing session
            existing.updateAccessTime();
            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] Updated session access time: sessionId=%s, connectId=%s",
                        sessionId, sessionInfo.getConnectId()));
            }
        } else {
            // Register new session
            sessionStore.put(sessionId, sessionInfo);

            // Track session under connectId
            connectIdToSessions
                    .computeIfAbsent(sessionInfo.getConnectId(), k -> new ConnectSessions())
                    .addSession(sessionId);

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] New session tracked: sessionId=%s, connectId=%s, userId=%s",
                        sessionId, sessionInfo.getConnectId(), sessionInfo.getUserId()));
            }
        }
    }

    /**
     * Manually remove a session (e.g., when releaseSession is called explicitly).
     *
     * @param sessionId the session identifier to remove
     */
    public void removeSession(String sessionId) {
        SessionInfo info = sessionStore.remove(sessionId);
        if (info != null) {
            closedSessionsCache.put(info.getSessionId(), info);
            ConnectSessions sessions = connectIdToSessions.get(info.getConnectId());
            if (sessions != null) {
                sessions.removeActiveSession(sessionId);
                // Check if all sessions for this connectId are done
                if (!sessions.hasActiveSessions()) {
                    connectIdToSessions.remove(info.getConnectId());
                    var jobInfo = new JobInfo(info.getConnectId(), info.getUserId(), info.getEventLogDir());
                    jobInfo.addSessions(sessions.getAllSessions());
                    if (onJobFinishedCallback != null) {
                        onJobFinishedCallback.accept(jobInfo);
                    }
                }
            }

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format("[SessionManager] Session removed: sessionId=%s, connectId=%s",
                        sessionId, info.getConnectId()));
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
            for (Map.Entry<String, SessionInfo> entry : sessionStore.entrySet()) {
                SessionInfo info = entry.getValue();
                if (info.isExpired()) {
                    expiredSessions.add(info);
                }
            }

            // Process expired sessions
            for (SessionInfo expired : expiredSessions) {
                LOG.info(String.format("[SessionManager] Session expired: sessionId=%s, connectId=%s, userId=%s, " +
                                "lastAccessTime=%d, age=%dms",
                        expired.getSessionId(), expired.getConnectId(), expired.getUserId(),
                        expired.getLastAccessTime(), System.currentTimeMillis() - expired.getLastAccessTime()));

                closedSessionsCache.put(expired.getSessionId(), expired);

                // Remove from tracking
                sessionStore.remove(expired.getSessionId());

                // Trigger callback for individual session expiration
                if (onSessionExpiredCallback != null) {
                    LOG.info(String.format("[SessionManager] Session expired, triggering callback: connectId=%s",
                            expired.getConnectId()));
                    try {
                        onSessionExpiredCallback.accept(expired);
                    } catch (Exception e) {
                        LOG.log(Level.WARNING,
                                String.format("[SessionManager] Error in session expired callback: connectId=%s", expired.getConnectId()), e);
                    }
                }

                // Check if all sessions for this job have expired
                ConnectSessions jobSessions = connectIdToSessions.get(expired.getConnectId());
                if (jobSessions != null) {
                    jobSessions.removeActiveSession(expired.getSessionId());
                    // Check if all sessions for this job have expired
                    if (!jobSessions.hasActiveSessions()) {
                        connectIdToSessions.remove(expired.getConnectId());

                        if (LOG.isLoggable(Level.INFO)) {
                            LOG.info(String.format("[SessionManager] All sessions for connectId=%s have expired. Triggering onJobFinishedCallback.", expired.getConnectId()));
                        }

                        var jobInfo = new JobInfo(expired.getConnectId(), expired.getUserId(), expired.getEventLogDir());
                        jobInfo.addSessions(jobSessions.getAllSessions());
                        if (onJobFinishedCallback != null) {
                            onJobFinishedCallback.accept(jobInfo);
                        }
                    }
                }
            }

            if (!expiredSessions.isEmpty()) {
                LOG.info(String.format("[SessionManager] Maintenance completed: expired=%d, remaining=%d",
                        expiredSessions.size(), sessionStore.size()));
            }
        } catch (Exception e) {
            LOG.log(Level.WARNING, "[SessionManager] Error during maintenance", e);
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
        return Optional.ofNullable(sessionStore.get(sessionId));
    }

    public Optional<SessionInfo> getClosedSessionInfo(String sessionId) {
        return Optional.ofNullable(closedSessionsCache.getIfPresent(sessionId));
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
