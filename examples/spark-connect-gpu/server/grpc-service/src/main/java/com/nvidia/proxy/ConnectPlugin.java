package com.nvidia.proxy;

import java.util.Map;

public interface ConnectPlugin {

    /**
     * Suggest specific Spark configurations based on unique ID, session ID, user ID, etc.
     *
     * @param uniqId   the unique identifier for the Spark application
     * @param userId   the user identifier
     * @param sessionId the Spark session identifier
     * @return a map of suggested Spark configuration properties
     */
    Map<String, String> suggestConfigurations(String uniqId,
                                              String userId,
                                              String sessionId
    );

    /**
     * Releases the session associated with the given identifiers.
     * Called when a session is terminated or no longer needed.
     *
     * @param uniqId    the unique id of the Spark applications.
     * @param userId    user id
     * @param sessionId the spark session id
     */
    void releaseSession(String uniqId, String userId, String sessionId);
}
