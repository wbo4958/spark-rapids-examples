package com.nvidia.proxy;

import com.nvidia.proxy.beans.PluginSuggestion;

public interface ConnectPlugin {

    /**
     * Suggest cluster type and Spark configurations based on user ID, session ID, job ID, etc.
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
     * @param jobId     the job identifier for the Spark application
     * @return a PluginSuggestion containing routing and configuration recommendations
     */
    PluginSuggestion suggestConfigurations(String userId,
                                           String sessionId,
                                           String jobId
    );

    /**
     * Releases the session associated with the given identifiers.
     * Called when a session is terminated or no longer needed.
     *
     * @param jobId       the job id of the Spark applications.
     * @param userId      user id
     * @param sessionId   the spark session id
     * @param eventLogDir the directory path where Spark event logs are stored
     */
    void releaseSession(String jobId, String userId, String sessionId, String eventLogDir);
}
