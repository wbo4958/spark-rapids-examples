# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build stage
FROM maven:3.8-openjdk-17 AS builder
# Set platform to linux/amd64
ENV DOCKER_DEFAULT_PLATFORM=linux/amd64
# Copy the grpc-service project
COPY src /build/src
COPY pom.xml /build/pom.xml
# Copy the aether plugin project
COPY aether /build/aether
WORKDIR /build
# Build all modules
RUN mvn clean install -DskipTests \
 && mvn -f aether/pom.xml clean package -DskipTests


# Spark connect client image
FROM apache/spark:3.5.6-java17

USER root
COPY --from=builder /build/target/com.nvidia.grpc-proxy-1.0-SNAPSHOT.jar /opt/spark/jars/
RUN chown spark:spark /opt/spark/jars/com.nvidia.grpc-proxy-1.0-SNAPSHOT.jar
COPY --from=builder /build/aether/target/connect-plugin-1.0-SNAPSHOT.jar /opt/spark/jars/
RUN chown spark:spark /opt/spark/jars/connect-plugin-1.0-SNAPSHOT.jar
COPY spark-connect_2.12-3.5.6.jar /opt/spark/jars/
RUN chown spark:spark /opt/spark/jars/spark-connect_2.12-3.5.6.jar
COPY certs /opt/spark/certs
RUN chown -R spark:spark /opt/spark/certs

USER spark
SHELL [ "/bin/bash", "-c" ]
ENV SHELL=/bin/bash