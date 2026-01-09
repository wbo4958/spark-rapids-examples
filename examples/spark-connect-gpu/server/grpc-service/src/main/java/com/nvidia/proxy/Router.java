package com.nvidia.proxy;

import com.nvidia.proxy.beans.PluginSuggestion;
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
 *   <li>First session for a connectId → CPU (serviceIndex=0)</li>
 *   <li>After first session release → serviceIndex permanently set to 1 (GPU)</li>
 *   <li>All subsequent sessions for that connectId → GPU (serviceIndex=1)</li>
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
class CpuFirstThenGpuPolicy implements ConnectPlugin {
    private static final Logger LOG = Logger.getLogger("CpuFirstThenGpuPolicy");

    /**
     * Maps connectId to service index (0=CPU, 1=GPU).
     * Default is 0 (CPU) for new connectIds.
     */
    private final Map<String, Integer> connectIdToServiceIndexMap;

    /**
     * Tracks active session IDs for each connectId.
     * Used to determine when all sessions for a connectId have been released.
     */
    private final Map<String, Set<String>> connectIdToSessionIdMap;

    public CpuFirstThenGpuPolicy() {
        // Use ConcurrentHashMap for thread-safe concurrent access from gRPC threads
        this.connectIdToServiceIndexMap = new java.util.concurrent.ConcurrentHashMap<>();
        this.connectIdToSessionIdMap = new java.util.concurrent.ConcurrentHashMap<>();
    }

    /**
     * Gets the service index for routing a request.
     *
     * <p>Returns 0 (CPU) for new connectIds, or the previously stored index.
     * Also tracks the sessionId as an active session for this connectId.</p>
     *
     * @param connectId the unique client identifier
     * @param sessionId the Spark session identifier
     * @return 0 for CPU routing, 1 for GPU routing
     */
    public int getServiceIndex(String connectId, String sessionId) {
        // Track active sessions using thread-safe set
        connectIdToSessionIdMap
                .computeIfAbsent(connectId, k -> java.util.concurrent.ConcurrentHashMap.newKeySet())
                .add(sessionId);

        // Return stored index, or default to 0 (CPU) for new connectIds
        int serviceIndex = connectIdToServiceIndexMap.getOrDefault(connectId, 0);

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] getServiceIndex: connectId=%s, sessionId=%s, serviceIndex=%d, activeSessions=%s",
                    connectId, sessionId, serviceIndex, connectIdToSessionIdMap.get(connectId)));
        }
        return serviceIndex;
    }

    @Override
    public PluginSuggestion suggestConfigurations(String userId, String sessionId, String connectId) {
        return null;
    }

    /**
     * Releases a session and updates routing for future sessions.
     *
     * <p>When all sessions for a connectId are released, the service index is set to 1 (GPU),
     * so all future sessions for this connectId will be routed to GPU.</p>
     *
     * @param connectId the unique client identifier
     * @param sessionId the Spark session identifier being released
     */
    @Override
    public void releaseSession(String connectId, String userId, String sessionId, String eventLogDir) {
        Set<String> sessionIds = connectIdToSessionIdMap.get(connectId);

        if (sessionIds != null) {
            sessionIds.remove(sessionId);

            // When all sessions are released, switch to GPU for future sessions
            if (sessionIds.isEmpty()) {
                connectIdToServiceIndexMap.put(connectId, 1);
                // Clean up empty session set to prevent memory leak
                connectIdToSessionIdMap.remove(connectId);
            }
        } else {
            // No active sessions tracked (edge case), default to GPU
            connectIdToServiceIndexMap.put(connectId, 1);
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[CpuFirstThenGpu] releaseSession: connectId=%s, sessionId=%s, newServiceIndex=%d",
                    connectId, sessionId, connectIdToServiceIndexMap.getOrDefault(connectId, 0)));
        }
    }

    @Override
    public void releaseJob(String connectId, String userId, Set<String> sessions, String eventLogDir) {

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
     * @param connectId the unique identifier for the Spark application
     * @return the appropriate SparkConnectService blocking stub (CPU or GPU)
     */
    public ServiceDetermination determineService(
            String userId,
            String sessionId,
            String clusterId,
            String connectId) {

        SparkConnectServiceGrpc.SparkConnectServiceBlockingStub selectedService;
        Map<String, String> configs = Collections.emptyMap();
        String clusterType;

        if (connectId == null || connectId.trim().isEmpty()) {
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
                        "[Router] determineService: connectId=%s, userId=%s, sessionId=%s, clusterId=%s, clusterType=%s",
                        connectId, userId, sessionId, clusterId, clusterType));
            }
            return new ServiceDetermination(selectedService, configs);
        }

        // Priority 2: Use plugin if present
        if (plugin.isPresent()) {
            var suggestion = plugin.get().suggestConfigurations(userId, sessionId, connectId);
            
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
            int serviceIndex = inMemoryPolicy.getServiceIndex(connectId, sessionId);
            if (serviceIndex % 2 == 0) {
                selectedService = cpuService;
                clusterType = "cpu(cpu-first-then-gpu)";
            } else {
                selectedService = gpuService;
                clusterType = "gpu(cpu-first-then-gpu)";
            }
        }

        if (LOG.isLoggable(Level.FINE)) {
            LOG.fine(String.format("[Router] determineService: connectId=%s, userId=%s, sessionId=%s, clusterId=%s, clusterType=%s",
                    connectId, userId, sessionId, clusterId, clusterType));
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
     * @param connectId   the unique identifier for the Spark application
     * @param userId      the user identifier
     * @param sessionId   the Spark session identifier
     * @param eventLogDir the directory path where Spark event logs are stored
     */
    public void releaseSession(String connectId, String userId, String sessionId, String eventLogDir) {
        LOG.info(String.format("[Router] releaseSession: connectId=%s, userId=%s, sessionId=%s, eventLogDir=%s",
                connectId, userId, sessionId, eventLogDir));

        if (plugin.isPresent()) {
            plugin.get().releaseSession(connectId, userId, sessionId, eventLogDir);
        } else {
            inMemoryPolicy.releaseSession(connectId, userId, sessionId, eventLogDir);
        }
    }

    public void releaseJob(String connectId, String userId, Set<String> sessions, String eventLogDir) {
        LOG.info(String.format("[Router] releaseJob: connectId=%s, userId=%s, sessionId=%s, eventLogDir=%s",
                connectId, userId, sessions, eventLogDir));

        if (plugin.isPresent()) {
            plugin.get().releaseJob(connectId, userId, sessions, eventLogDir);
        } else {
            inMemoryPolicy.releaseJob(connectId, userId, sessions, eventLogDir);
        }
    }

    @Override
    public void close() throws IOException {
    }
}
