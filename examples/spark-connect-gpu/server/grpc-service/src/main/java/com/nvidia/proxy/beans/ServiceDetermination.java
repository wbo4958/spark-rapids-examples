package com.nvidia.proxy.beans;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;

import java.util.Map;

/**
 * Result of service determination containing both the selected service stub
 * and any configuration properties to be applied.
 */
public record ServiceDetermination(
        SparkConnectServiceGrpc.SparkConnectServiceBlockingStub service,
        Map<String, String> configurations
) {}
