package com.nvidia.proxy.beans;

import java.util.Collections;
import java.util.Map;

/**
 * Represents a plugin's routing and configuration suggestions.
 * 
 * <p>This record encapsulates two distinct pieces of information:</p>
 * <ul>
 *   <li><b>clusterType</b>: Routing metadata indicating which backend to use ("cpu" or "gpu")</li>
 *   <li><b>sparkConfigurations</b>: Spark configuration properties to be applied to the session</li>
 * </ul>
 * 
 * <p>By separating routing metadata from Spark configurations, this design ensures
 * that the Router can make routing decisions without needing to mutate the
 * configuration map returned by the plugin.</p>
 *
 * @param clusterType The type of cluster to route to ("cpu", "gpu", or null for default)
 * @param sparkConfigurations Immutable map of Spark configuration key-value pairs
 */
public record PluginSuggestion(
        String clusterType,
        Map<String, String> sparkConfigurations
) {
    /**
     * Creates a PluginSuggestion with empty Spark configurations.
     * 
     * @param clusterType The type of cluster to route to
     * @return A new PluginSuggestion with the specified cluster type and no configurations
     */
    public static PluginSuggestion ofClusterType(String clusterType) {
        return new PluginSuggestion(clusterType, Collections.emptyMap());
    }
    
    /**
     * Creates a PluginSuggestion with default routing (null cluster type).
     * 
     * @param sparkConfigurations The Spark configurations to apply
     * @return A new PluginSuggestion with default routing and the specified configurations
     */
    public static PluginSuggestion ofConfigurations(Map<String, String> sparkConfigurations) {
        return new PluginSuggestion(null, sparkConfigurations);
    }
    
    /**
     * Creates a PluginSuggestion with no routing override and no configurations.
     * 
     * @return An empty PluginSuggestion
     */
    public static PluginSuggestion empty() {
        return new PluginSuggestion(null, Collections.emptyMap());
    }
}

