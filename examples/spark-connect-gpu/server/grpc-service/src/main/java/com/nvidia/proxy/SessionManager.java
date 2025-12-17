package com.nvidia.proxy;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 1. Client creates session Id
 * 2. Client sends Request with client session id to Server
 * 3. Server creates session Id for the first time.
 * 4. Server sends Response with server session id back to Client.
 * 5. Client stores to server session id.
 */
public class SessionManager {
    private static final Logger LOG = Logger.getLogger("SessionManager");

    /**
     * Maintains a mapping between client-side session IDs and server-side session IDs for Spark Connect.
     *
     * <p>The server-side session ID is generated/updated by the first Spark Connect Server instance
     * that responds to a client request. This mapping ensures we can correlate client session IDs
     * with their corresponding server-side identifiers and return the server-side ID to the client.
     */
    private Map<String, String> clientSessionIdToServerSessionIdMap = new HashMap<>();

    // Map the services to the Map of client session id to server session id.
    private Map<SparkConnectServiceGrpc.SparkConnectServiceBlockingStub, Map<String, String>> serviceToSessionIdMap = new HashMap<>();

    public SessionManager() {
        LOG.setLevel(Level.FINE);
    }
    /**
     * Get the server side session id according to the client session id.
     * @param clientSideSessionId the client side session id
     * @param serverSideSessionId the server side session id.
     * @return the server side session id stored for the first response.
     */
    public String getServerSideSessionId(String clientSideSessionId,
                                         String serverSideSessionId) {
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
     * @param service the Spark Connect Server
     */
    public void storeSessionIds(String clientSideSessionId,
                                String serverSideSessionId,
                                SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service) {
        Map<String, String> clientToServerSessionIdMap = serviceToSessionIdMap
                .computeIfAbsent(service, k -> new HashMap<>());

        boolean isNewMapping = !clientToServerSessionIdMap.containsKey(clientSideSessionId);
        clientToServerSessionIdMap.put(clientSideSessionId, serverSideSessionId);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[SessionManager] storeSessionIds: clientSessionId=%s, serverSessionId=%s, isNew=%s, totalMappings=%d",
                    clientSideSessionId, serverSideSessionId, isNewMapping, clientToServerSessionIdMap.size()));
        }
    }

}
