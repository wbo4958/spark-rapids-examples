package com.nvidia.proxy;

import com.nvidia.proxy.beans.ServiceDetermination;
import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Routing policy that directs first session to CPU, then all subsequent sessions to GPU.
 *
 * <p>This is the default fallback routing policy used when no plugin is configured.
 * It provides a simple "warm-up on CPU, then migrate to GPU" pattern per unique client.</p>
 *
 * <h3>Routing Behavior:</h3>
 * <ul>
 *   <li>First session for a uniqId → CPU (serviceIndex=0)</li>
 *   <li>After first session release → serviceIndex permanently set to 1 (GPU)</li>
 *   <li>All subsequent sessions for that uniqId → GPU (serviceIndex=1)</li>
 * </ul>
 *
 * <h3>Thread Safety:</h3>
 * <p>This class uses {@link java.util.concurrent.ConcurrentHashMap} for thread-safe
 * concurrent access from multiple gRPC request threads.</p>
 *
 * <h3>Use Case:</h3>
 * <p>Useful when you want clients to "warm up" on CPU before migrating to GPU,
 * or when GPU resources should be reserved for returning users.</p>
 */
class CpuFirstThenGpuPolicy {
    private static final Logger LOG = Logger.getLogger(CpuFirstThenGpuPolicy.class.getName());

    /**
     * Maps uniqId to service index (0=CPU, 1=GPU).
     * Default is 0 (CPU) for new uniqIds.
     */
    private final Map<String, Integer> uniqIdToServiceIndexMap;

    /**
     * Tracks active session IDs for each uniqId.
     * Used to determine when all sessions for a uniqId have been released.
     */
    private final Map<String, Set<String>> uniqIdToSessionIdMap;

    public CpuFirstThenGpuPolicy() {
        // Use ConcurrentHashMap for thread-safe concurrent access from gRPC threads
        this.uniqIdToServiceIndexMap = new java.util.concurrent.ConcurrentHashMap<>();
        this.uniqIdToSessionIdMap = new java.util.concurrent.ConcurrentHashMap<>();
    }

    /**
     * Gets the service index for routing a request.
     *
     * <p>Returns 0 (CPU) for new uniqIds, or the previously stored index.
     * Also tracks the sessionId as an active session for this uniqId.</p>
     *
     * @param uniqId    the unique client identifier
     * @param sessionId the Spark session identifier
     * @return 0 for CPU routing, 1 for GPU routing
     */
    public int getServiceIndex(String uniqId, String sessionId) {
        // Track active sessions using thread-safe set
        uniqIdToSessionIdMap
                .computeIfAbsent(uniqId, k -> java.util.concurrent.ConcurrentHashMap.newKeySet())
                .add(sessionId);

        // Return stored index, or default to 0 (CPU) for new uniqIds
        int serviceIndex = uniqIdToServiceIndexMap.getOrDefault(uniqId, 0);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] getServiceIndex: uniqId=%s, sessionId=%s, serviceIndex=%d, activeSessions=%s",
                    uniqId, sessionId, serviceIndex, uniqIdToSessionIdMap.get(uniqId)));
        }
        return serviceIndex;
    }

    /**
     * Releases a session and updates routing for future sessions.
     *
     * <p>When all sessions for a uniqId are released, the service index is set to 1 (GPU),
     * so all future sessions for this uniqId will be routed to GPU.</p>
     *
     * @param uniqId    the unique client identifier
     * @param sessionId the Spark session identifier being released
     */
    public void releaseSession(String uniqId, String sessionId) {
        Set<String> sessionIds = uniqIdToSessionIdMap.get(uniqId);

        if (sessionIds != null) {
            sessionIds.remove(sessionId);

            // When all sessions are released, switch to GPU for future sessions
            if (sessionIds.isEmpty()) {
                uniqIdToServiceIndexMap.put(uniqId, 1);
                // Clean up empty session set to prevent memory leak
                uniqIdToSessionIdMap.remove(uniqId);
            }
        } else {
            // No active sessions tracked (edge case), default to GPU
            uniqIdToServiceIndexMap.put(uniqId, 1);
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] releaseSession: uniqId=%s, sessionId=%s, newServiceIndex=%d",
                    uniqId, sessionId, uniqIdToServiceIndexMap.getOrDefault(uniqId, 0)));
        }
    }
}

public class Router implements Closeable {
    private static final Logger LOG = Logger.getLogger(Router.class.getName());

    private ManagedChannel cpuChannel;
    private ManagedChannel gpuChannel;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub cpuService;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub gpuService;

    private final Optional<ConnectPlugin> plugin;
    private CpuFirstThenGpuPolicy inMemoryPolicy;

    public Router() {
        // TODO, discover the Spark Connect Server automatically.
        this.cpuChannel = ManagedChannelBuilder
                .forAddress("spark-connect-server-cpu", 15002)
                .usePlaintext()
                .build();
        this.cpuService = SparkConnectServiceGrpc.newBlockingStub(cpuChannel);
        LOG.info("Router initialized CPU channel: spark-connect-server-cpu:15002");

        this.gpuChannel = ManagedChannelBuilder
                .forAddress("spark-connect-server", 15002)
                .usePlaintext()
                .build();
        this.gpuService = SparkConnectServiceGrpc.newBlockingStub(gpuChannel);
        LOG.info("Router initialized GPU channel: spark-connect-server:15002");

        // Discover plugins via ServiceLoader
        this.plugin = discoverPlugins();
        this.inMemoryPolicy = new CpuFirstThenGpuPolicy();

        LOG.info(String.format("Router initialized: pluginPresent=%s", plugin.isPresent()));
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
     * CPU and GPU clusters. Otherwise, falls back to the CpuFirstThenGpu policy (first session → CPU,
     * subsequent sessions → GPU).
     *
     * @param uniqId    the unique identifier for the Spark application
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     * @return the appropriate SparkConnectService blocking stub (CPU or GPU)
     */
    public ServiceDetermination determineService(
            String uniqId,
            String userId,
            String sessionId) {

        SparkConnectServiceGrpc.SparkConnectServiceBlockingStub selectedService;
        Map<String, String> configs = Collections.emptyMap();
        String clusterType;

        if (plugin.isPresent()) {
            configs = plugin.get().suggestConfigurations(uniqId, userId, sessionId);
            if (configs != null && configs.containsKey("cluster.type")) {
                if (Objects.equals(configs.get("cluster.type"), "cpu")) {
                    selectedService = cpuService;
                    clusterType = "cpu";
                } else if (Objects.equals(configs.get("cluster.type"), "gpu")) {
                    selectedService = gpuService;
                    clusterType = "gpu";
                } else {
                    selectedService = cpuService;
                    clusterType = "cpu(default)";
                }
                configs.remove("cluster.type");
            } else {
                selectedService = cpuService;
                clusterType = "cpu(no-config)";
            }
        } else {
            int serviceIndex = inMemoryPolicy.getServiceIndex(uniqId, sessionId);
            if (serviceIndex % 2 == 0) {
                selectedService = cpuService;
                clusterType = "cpu(cpu-first-then-gpu)";
            } else {
                selectedService = gpuService;
                clusterType = "gpu(cpu-first-then-gpu)";
            }
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[Router] determineService: uniqId=%s, userId=%s, sessionId=%s, clusterType=%s",
                    uniqId, userId, sessionId, clusterType));
        }
        if (configs == null) {
            configs = Collections.emptyMap();
        }
        return new ServiceDetermination(selectedService, configs);
    }

    /**
     * Release the session associated with the given identifiers.
     * If a plugin is present, delegates the release logic to the plugin.
     * Otherwise, falls back to the in-memory CpuFirstThenGpu policy.
     *
     * @param uniqId    the unique identifier for the Spark application
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     */
    public void releaseSession(String uniqId, String userId, String sessionId) {
        LOG.info(String.format("[Router] releaseSession: uniqId=%s, userId=%s, sessionId=%s",
                uniqId, userId, sessionId));

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
