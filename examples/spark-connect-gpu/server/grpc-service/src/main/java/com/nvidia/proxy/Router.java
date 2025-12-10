package com.nvidia.proxy;

import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Router implements Closeable {
    private ManagedChannel cpuChannel;
    private ManagedChannel gpuChannel;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub cpuService;
    private SparkConnectServiceGrpc.SparkConnectServiceBlockingStub gpuService;

    private Map<String, SparkConnectServiceGrpc.SparkConnectServiceBlockingStub> sessionIdToServiceMap;
    private int count = 0;

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

        this.sessionIdToServiceMap = new HashMap<>();
    }

    /**
     * Choose a Spark Connect Service.
     *
     * @param userId    the user id.
     * @param sessionId the session id.
     * @return the Spark Connect Service.
     */
    public SparkConnectServiceGrpc.SparkConnectServiceBlockingStub routePerSession(String userId,
                                                                                   String sessionId) {
        if (sessionIdToServiceMap.containsKey(sessionId)) {
            return sessionIdToServiceMap.get(sessionId);
        }

        if (count++ % 2 == 0) {
            sessionIdToServiceMap.put(sessionId, cpuService);
        } else {
            sessionIdToServiceMap.put(sessionId, gpuService);
        }
        return sessionIdToServiceMap.get(sessionId);
    }

    @Override
    public void close() throws IOException {
    }
}
