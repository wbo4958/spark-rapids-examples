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
# Copy the entire project
COPY auth /build
WORKDIR /build
# Build all modules
RUN mvn clean install -DskipTests

FROM apache/spark:4.0.0

ARG CUDA_VERSION
ARG RAPIDS_VERSION
ARG REPO_URL

USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY --from=builder /build/target/connect-auth-1.0-SNAPSHOT.jar /opt/spark-rapids/jars/


RUN mkdir -p /opt/spark-rapids/jars
RUN chown -R spark:spark /opt/spark-rapids

USER spark

ENV CUDA_VERSION=${CUDA_VERSION}
ENV RAPIDS_VERSION=${RAPIDS_VERSION}
ENV REPO_URL=${REPO_URL}

RUN wget -q ${REPO_URL}/com/nvidia/rapids-4-spark_2.13/${RAPIDS_VERSION}/rapids-4-spark_2.13-${RAPIDS_VERSION}-cuda${CUDA_VERSION}.jar -O /opt/spark-rapids/jars/rapids-4-spark-sql.jar
COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf
COPY spark-env.sh /opt/spark/conf/spark-env.sh
COPY credentials.txt /opt/spark/conf/credentials.txt
