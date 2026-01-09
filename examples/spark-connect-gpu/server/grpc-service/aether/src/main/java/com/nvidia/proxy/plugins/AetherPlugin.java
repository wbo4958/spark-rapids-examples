package com.nvidia.proxy.plugins;

import com.nvidia.proxy.ConnectPlugin;
import com.nvidia.proxy.beans.PluginSuggestion;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple AetherPlugin that demonstrates how routing decisions can be made.
 * The logic is intentionally lightweight:
 * - Routing decisions are made when a session is first released
 * - Subsequent calls return the cached routing and configuration
 * - Additional Spark configuration defaults are returned to keep the example usable.
 */
public class AetherPlugin implements ConnectPlugin {

    private static final Logger LOG = Logger.getLogger("AetherPlugin");

    private Map<String, PluginSuggestion> connectIdToSuggestions = new HashMap<>();

    @Override
    public PluginSuggestion suggestConfigurations(String userId, String sessionId, String connectId) {
        PluginSuggestion suggestion = connectIdToSuggestions.getOrDefault(connectId, PluginSuggestion.empty());

        LOG.log(Level.INFO, String.format(
                "AetherPlugin suggestions for userId=%s, sessionId=%s, connectId=%s -> clusterType=%s, sparkConfigs=%s",
                Objects.toString(userId, "unknown"),
                Objects.toString(sessionId, "unknown"),
                Objects.toString(connectId, "unknown"),
                Objects.toString(suggestion.clusterType(), "default"),
                suggestion.sparkConfigurations()));

        return suggestion;
    }

    @Override
    public void releaseSession(String connectId, String userId, String sessionId, String eventLogDir) {
        LOG.log(Level.INFO, () -> String.format(
                "Releasing session for connectId=%s, userId=%s, sessionId=%s, eventLogDir=%s",
                Objects.toString(connectId, "unknown"),
                Objects.toString(userId, "unknown"),
                Objects.toString(sessionId, "unknown"),
                Objects.toString(eventLogDir, "unknown")));
    }

    @Override
    public void releaseJob(String connectId, String userId, Set<String> sessions, String eventLogDir) {
        LOG.log(Level.INFO, () -> String.format(
                "Releasing job for connectId=%s, userId=%s, sessions=%s, eventLogDir=%s",
                Objects.toString(connectId, "unknown"),
                Objects.toString(userId, "unknown"),
                Objects.toString(sessions, "unknown"),
                Objects.toString(eventLogDir, "unknown")));

        // Calculate suggestions for this connectId if not already cached.
        if (!connectIdToSuggestions.containsKey(connectId)) {
            // TODO: Calculate the suggestions for this connectId via Aether.
            String clusterType;
            Map<String, String> sparkConfigs = new HashMap<>();

            if (connectId.equals("hello-gpu")) {
                clusterType = "gpu";
                sparkConfigs.put("spark.rapids.hello.cluster.type", "gpu-cluster");
                sparkConfigs.put("spark.rapids.hello.cluster.name", "spark-connect-gpu-cluster");
                sparkConfigs.put("spark.rapids.hello.connectId", "hello-gpu");
            } else if (connectId.equals("hello-cpu")) {
                clusterType = "cpu";
                sparkConfigs.put("spark.rapids.hello.cluster.type", "cpu-cluster");
                sparkConfigs.put("spark.rapids.hello.cluster.name", "spark-connect-cpu-cluster");
                sparkConfigs.put("spark.rapids.hello.connectId", "hello-cpu");
            } else {
                clusterType = "cpu";
                sparkConfigs.put("spark.rapids.hello.connectId", connectId);
            }

            connectIdToSuggestions.put(connectId, new PluginSuggestion(clusterType, sparkConfigs));
        }
    }
}

