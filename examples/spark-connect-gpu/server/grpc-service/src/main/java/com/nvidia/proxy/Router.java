package com.nvidia.proxy;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

class RoundRobinPolicy {
    private Map<String, Integer> uniqIdToServiceIndexMap;
    private Map<String, Set<String>> uniqIdToSessionIdMap;

    public RoundRobinPolicy() {
        this.uniqIdToServiceIndexMap = new HashMap<>();
        this.uniqIdToSessionIdMap = new HashMap<>();
    }

    public int getServiceIndex(String uniqId, String sessionId) {
        // Track the set of sessionIds for each uniqId
        uniqIdToSessionIdMap.computeIfAbsent(uniqId, k -> new HashSet<>()).add(sessionId);

        System.out.println("getServiceIndex " + uniqIdToSessionIdMap);
        // Return the mapped service index if it exists, otherwise default to 0 (CPU)
        return uniqIdToServiceIndexMap.getOrDefault(uniqId, 0);
    }


    public void releaseSession(String uniqId, String sessionId) {
        Set<String> sessionIds = uniqIdToSessionIdMap.get(uniqId);
        if (sessionIds != null) {
            sessionIds.remove(sessionId);
            if (sessionIds.isEmpty()) {
                uniqIdToServiceIndexMap.put(uniqId, 1);
            }
        } else {
            uniqIdToServiceIndexMap.put(uniqId, 1);
        }
        System.out.println("releaseSession " + uniqIdToServiceIndexMap);
    }

}

public class Router implements Closeable {
    private ManagedChannel cpuChannel;
    private ManagedChannel gpuChannel;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub cpuService;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub gpuService;

    private final Optional<ConnectPlugin> plugin;
    private RoundRobinPolicy inMemoryPolicy;

    public Router() {
        // TODO, discover the Spark Connect Server automatically.
        this.cpuChannel = ManagedChannelBuilder
                .forAddress("spark-connect-server-cpu", 15002)
                .usePlaintext()
                .build();
        this.cpuService = SparkConnectServiceGrpc.newBlockingStub(cpuChannel);

        this.gpuChannel = ManagedChannelBuilder
                .forAddress("spark-connect-server", 15002)
                .usePlaintext()
                .build();
        this.gpuService = SparkConnectServiceGrpc.newBlockingStub(gpuChannel);

        // Discover plugins via ServiceLoader
        this.plugin = discoverPlugins();
        this.inMemoryPolicy = new RoundRobinPolicy();
    }

    /**
     * Discover ConnectPlugin implementations using ServiceLoader.
     *
     * @return list of discovered plugins sorted by priority (highest first)
     */
    private Optional<ConnectPlugin> discoverPlugins() {
        ServiceLoader<ConnectPlugin> loader = ServiceLoader.load(ConnectPlugin.class);
        List<ConnectPlugin> discovered = new ArrayList<>();

        // Choose the first one.
        for (ConnectPlugin plugin : loader) {
            discovered.add(plugin);
            break;
        }
        return discovered.stream().findFirst();
    }

    /**
     * Determine the appropriate Spark Connect Service based on plugin configuration or routing policy.
     * If a plugin is present, it uses the plugin's suggested configurations to select between
     * CPU and GPU clusters. Otherwise, falls back to an in-memory round-robin policy.
     *
     * @param uniqId    the unique identifier for the Spark application
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     * @return the appropriate SparkConnectService blocking stub (CPU or GPU)
     */
    public SparkConnectServiceGrpc.SparkConnectServiceBlockingStub determineService(
            String uniqId,
            String userId,
            String sessionId) {

        if (plugin.isPresent()) {
            var configs = plugin.get().suggestConfigurations(uniqId, userId, sessionId);
            if (configs != null && configs.containsKey("cluster.type")) {
                if (Objects.equals(configs.get("cluster.type"), "cpu")) {
                    return cpuService;
                } else if (Objects.equals(configs.get("cluster.type"), "gpu")) {
                    return gpuService;
                }
            }
            return cpuService;
        } else {
            return inMemoryPolicy.getServiceIndex(uniqId, sessionId) % 2 == 0 ? cpuService : gpuService;
        }
    }

    /**
     * Release the session associated with the given identifiers.
     * If a plugin is present, delegates the release logic to the plugin.
     * Otherwise, falls back to the in-memory round-robin policy.
     *
     * @param uniqId    the unique identifier for the Spark application
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     */
    public void releaseSession(String uniqId, String userId, String sessionId) {
        if (plugin.isPresent()) {
            plugin.get().releaseSession(uniqId, userId, sessionId);
        } else {
            inMemoryPolicy.releaseSession(uniqId, sessionId);
        }
    }

    @Override
    public void close() throws IOException {
    }
}
