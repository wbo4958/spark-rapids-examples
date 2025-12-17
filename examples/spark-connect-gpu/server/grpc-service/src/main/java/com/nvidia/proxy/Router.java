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
 *   <li>First session for a jobId → CPU (serviceIndex=0)</li>
 *   <li>After first session release → serviceIndex permanently set to 1 (GPU)</li>
 *   <li>All subsequent sessions for that jobId → GPU (serviceIndex=1)</li>
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
    private static final Logger LOG = Logger.getLogger("CpuFirstThenGpuPolicy");

    /**
     * Maps jobId to service index (0=CPU, 1=GPU).
     * Default is 0 (CPU) for new jobIds.
     */
    private final Map<String, Integer> jobIdToServiceIndexMap;

    /**
     * Tracks active session IDs for each jobId.
     * Used to determine when all sessions for a jobId have been released.
     */
    private final Map<String, Set<String>> jobIdToSessionIdMap;

    public CpuFirstThenGpuPolicy() {
        // Use ConcurrentHashMap for thread-safe concurrent access from gRPC threads
        this.jobIdToServiceIndexMap = new java.util.concurrent.ConcurrentHashMap<>();
        this.jobIdToSessionIdMap = new java.util.concurrent.ConcurrentHashMap<>();
    }

    /**
     * Gets the service index for routing a request.
     *
     * <p>Returns 0 (CPU) for new jobIds, or the previously stored index.
     * Also tracks the sessionId as an active session for this jobId.</p>
     *
     * @param jobId    the unique client identifier
     * @param sessionId the Spark session identifier
     * @return 0 for CPU routing, 1 for GPU routing
     */
    public int getServiceIndex(String jobId, String sessionId) {
        // Track active sessions using thread-safe set
        jobIdToSessionIdMap
                .computeIfAbsent(jobId, k -> java.util.concurrent.ConcurrentHashMap.newKeySet())
                .add(sessionId);

        // Return stored index, or default to 0 (CPU) for new jobIds
        int serviceIndex = jobIdToServiceIndexMap.getOrDefault(jobId, 0);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] getServiceIndex: jobId=%s, sessionId=%s, serviceIndex=%d, activeSessions=%s",
                    jobId, sessionId, serviceIndex, jobIdToSessionIdMap.get(jobId)));
        }
        return serviceIndex;
    }

    /**
     * Releases a session and updates routing for future sessions.
     *
     * <p>When all sessions for a jobId are released, the service index is set to 1 (GPU),
     * so all future sessions for this jobId will be routed to GPU.</p>
     *
     * @param jobId    the unique client identifier
     * @param sessionId the Spark session identifier being released
     */
    public void releaseSession(String jobId, String sessionId) {
        Set<String> sessionIds = jobIdToSessionIdMap.get(jobId);

        if (sessionIds != null) {
            sessionIds.remove(sessionId);

            // When all sessions are released, switch to GPU for future sessions
            if (sessionIds.isEmpty()) {
                jobIdToServiceIndexMap.put(jobId, 1);
                // Clean up empty session set to prevent memory leak
                jobIdToSessionIdMap.remove(jobId);
            }
        } else {
            // No active sessions tracked (edge case), default to GPU
            jobIdToServiceIndexMap.put(jobId, 1);
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] releaseSession: jobId=%s, sessionId=%s, newServiceIndex=%d",
                    jobId, sessionId, jobIdToServiceIndexMap.getOrDefault(jobId, 0)));
        }
    }
}

public class Router implements Closeable {
    private static final Logger LOG = Logger.getLogger("Router");

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
     * Determine the appropriate Spark Connect Service based on cluster ID, plugin configuration, or routing policy.
     * <p>
     * Priority order for service selection:
     * <ol>
     *   <li>If clusterId is "cpu" or "gpu", use that directly</li>
     *   <li>If a plugin is present, use the plugin's suggested configurations</li>
     *   <li>Otherwise, fall back to the CpuFirstThenGpu policy (first session → CPU, subsequent sessions → GPU)</li>
     * </ol>
     *
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     * @param clusterId the cluster ID from HTTP header ("cpu" or "gpu"), may be null or empty
     * @param jobId     the unique identifier for the Spark application
     * @return the appropriate SparkConnectService blocking stub (CPU or GPU)
     */
    public ServiceDetermination determineService(
            String userId,
            String sessionId,
            String clusterId,
            String jobId) {

        SparkConnectServiceGrpc.SparkConnectServiceBlockingStub selectedService;
        Map<String, String> configs = Collections.emptyMap();
        String clusterType;

        if (jobId == null || jobId.trim().isEmpty()) {
            // Priority 1: Check if clusterId is explicitly specified
            if (Objects.equals(clusterId, "cpu")) {
                selectedService = cpuService;
                clusterType = "cpu(header)";
            } else if (Objects.equals(clusterId, "gpu")) {
                selectedService = gpuService;
                clusterType = "gpu(header)";
            } else {
                // Unknown clusterId, fall through to other logic
                selectedService = cpuService;
                clusterType = "cpu(default)";
            }

            if (LOG.isLoggable(Level.FINE)) {
                LOG.fine(String.format(
                        "[Router] determineService: jobId=%s, userId=%s, sessionId=%s, clusterId=%s, clusterType=%s",
                        jobId, userId, sessionId, clusterId, clusterType));
            }
            return new ServiceDetermination(selectedService, configs);
        }

        // Priority 2: Use plugin if present
        if (plugin.isPresent()) {
            var suggestion = plugin.get().suggestConfigurations(userId, sessionId, jobId);
            
            // Use the plugin's cluster type recommendation for routing
            String pluginClusterType = suggestion.clusterType();
            if (pluginClusterType != null) {
                if (Objects.equals(pluginClusterType, "cpu")) {
                    selectedService = cpuService;
                    clusterType = "cpu(plugin)";
                } else if (Objects.equals(pluginClusterType, "gpu")) {
                    selectedService = gpuService;
                    clusterType = "gpu(plugin)";
                } else {
                    selectedService = cpuService;
                    clusterType = "cpu(plugin-default)";
                }
            } else {
                // No cluster type specified, default to CPU
                selectedService = cpuService;
                clusterType = "cpu(plugin-no-type)";
            }
            
            // Get Spark configurations (already clean - no routing metadata mixed in)
            configs = suggestion.sparkConfigurations();
        } else {
            // Priority 3: Fall back to in-memory policy
            int serviceIndex = inMemoryPolicy.getServiceIndex(jobId, sessionId);
            if (serviceIndex % 2 == 0) {
                selectedService = cpuService;
                clusterType = "cpu(cpu-first-then-gpu)";
            } else {
                selectedService = gpuService;
                clusterType = "gpu(cpu-first-then-gpu)";
            }
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[Router] determineService: jobId=%s, userId=%s, sessionId=%s, clusterId=%s, clusterType=%s",
                    jobId, userId, sessionId, clusterId, clusterType));
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
     * @param jobId    the unique identifier for the Spark application
     * @param userId    the user identifier
     * @param sessionId the Spark session identifier
     */
    public void releaseSession(String jobId, String userId, String sessionId) {
        LOG.info(String.format("[Router] releaseSession: jobId=%s, userId=%s, sessionId=%s",
                jobId, userId, sessionId));

        if (plugin.isPresent()) {
            plugin.get().releaseSession(jobId, userId, sessionId);
        } else {
            inMemoryPolicy.releaseSession(jobId, sessionId);
        }
    }

    @Override
    public void close() throws IOException {
    }
}
