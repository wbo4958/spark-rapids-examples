package com.nvidia.proxy;

import com.nvidia.proxy.beans.PluginSuggestion;

import java.util.Set;

public interface ConnectPlugin {

    /**
     * Suggest cluster type and Spark configurations based on user ID, session ID, connect ID, etc.
     * 
     * <p>This method provides routing decisions and configuration recommendations.
     * The returned {@link PluginSuggestion} contains:</p>
     * <ul>
     *   <li><b>clusterType</b>: "cpu" or "gpu" to control routing (null for default)</li>
     *   <li><b>sparkConfigurations</b>: Key-value pairs to configure the Spark session</li>
     * </ul>
     *
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     * @param connectId the connect identifier for the Spark application
     * @return a PluginSuggestion containing routing and configuration recommendations
     */
    PluginSuggestion suggestConfigurations(String userId,
                                           String sessionId,
                                           String connectId
    );

    /**
     * Releases the session associated with the given identifiers.
     * Called when a session is terminated or no longer needed.
     *
     * @param connectId   the connect id of the Spark applications.
     * @param userId      user id
     * @param sessionId   the spark session id
     * @param eventLogDir the directory path where Spark event logs are stored
     */
    void releaseSession(String connectId, String userId, String sessionId, String eventLogDir);

    /**
     * Releases all resources associated with a completed job.
     * Called when all sessions associated with a job have finished or expired.
     *
     * @param connectId   the connect id of the Spark application.
     * @param userId      user id associated with the job.
     * @param sessions    set of all session IDs that belonged to this job.
     * @param eventLogDir the directory path where Spark event logs for the job are stored.
     */
    void releaseJob(String connectId, String userId, Set<String> sessions, String eventLogDir);
}
