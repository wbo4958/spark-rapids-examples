package com.nvidia.proxy.plugins;

import com.nvidia.proxy.ConnectPlugin;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple AetherPlugin that demonstrates how routing decisions can be made.
 * The logic is intentionally lightweight:
 * - If the user id starts with "gpu-" the request is directed to the GPU cluster.
 * - Otherwise it defaults to the CPU cluster.
 * - Additional Spark configuration defaults are returned to keep the example usable.
 */
public class AetherPlugin implements ConnectPlugin {

    private static final Logger LOG = Logger.getLogger(AetherPlugin.class.getName());

    private Map<String, Map<String, String>> uniqIdToConfigs = new HashMap<>();

    @Override
    public Map<String, String> suggestConfigurations(String uniqId, String userId, String sessionId) {
        Map<String, String> configs = uniqIdToConfigs.getOrDefault(uniqId, Collections.emptyMap());

        LOG.log(Level.INFO, String.format(
                "AetherPlugin suggestions for uniqId=%s, userId=%s, sessionId=%s -> %s",
                Objects.toString(uniqId, "unknown"),
                Objects.toString(userId, "unknown"),
                Objects.toString(sessionId, "unknown"),
                configs));

        return configs;
    }

    @Override
    public void releaseSession(String uniqId, String userId, String sessionId) {
        LOG.log(Level.INFO, () -> String.format(
                "Releasing session for uniqId=%s, userId=%s, sessionId=%s",
                Objects.toString(uniqId, "unknown"),
                Objects.toString(userId, "unknown"),
                Objects.toString(sessionId, "unknown")));

        // Calculate configs for this uniqId anymore.
        if (!uniqIdToConfigs.containsKey(uniqId)) {
            // TODO: calcuate the configs for this uniqId via Aether.
            var configs = new HashMap<String, String>();
            if (uniqId.equals("hello-gpu")) {
                configs.put("cluster.type", "gpu");
                configs.put("spark.rapids.hello.cluster.type", "gpu-cluster");
                configs.put("spark.rapids.hello.cluster.name", "spark-connect-gpu-cluster");
                configs.put("spark.rapids.hello.uniqId", "hello-gpu");
            } else if (uniqId.equals("hello-cpu")) {
                configs.put("cluster.type", "cpu");
                configs.put("spark.rapids.hello.cluster.type", "cpu-cluster");
                configs.put("spark.rapids.hello.cluster.name", "spark-connect-cpu-cluster");
                configs.put("spark.rapids.hello.uniqId", "hello-cpu");
            } else {
                configs.put("cluster.type", "cpu");
                configs.put("spark.rapids.hello.uniqId", uniqId);
            }
            uniqIdToConfigs.put(uniqId, configs);
        }
    }
}

