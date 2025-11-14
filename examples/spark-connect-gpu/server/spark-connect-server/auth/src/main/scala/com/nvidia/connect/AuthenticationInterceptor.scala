/**
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package com.nvidia.connect

import com.google.protobuf.Message
import io.grpc.ForwardingServerCallListener.SimpleForwardingServerCallListener
import io.grpc.{Metadata, ServerCall, ServerCallHandler, ServerInterceptor, Status}
import org.apache.spark.connect.proto

import scala.io.Source
import scala.util.Using

class AuthenticationInterceptor extends ServerInterceptor {
  private val CREDENTIALS_FILE = "/opt/spark/conf/credentials.txt"

  private val accessKeyMetadataKey = Metadata.Key.of("access_key", Metadata.ASCII_STRING_MARSHALLER)

  // Read file and convert to Map[String, String]
  private val userKeys: Map[String, String] = Using.resource(Source.fromFile(CREDENTIALS_FILE)) { source =>
    source.getLines()
      .filter(_.contains(":"))
      .map { line =>
        val Array(key, value) = line.split(":", 2)
        key -> value
      }
      .toMap
  }

  override def interceptCall[ReqT, RespT](call: ServerCall[ReqT, RespT],
                                          metadata: Metadata,
                                          next: ServerCallHandler[ReqT, RespT]): ServerCall.Listener[ReqT] = {
    val accessKeyHeaderValue = Option(metadata.get(accessKeyMetadataKey)).getOrElse {
      call.close(Status.UNAUTHENTICATED.withDescription("No access key provided"), new Metadata())
      return new ServerCall.Listener[ReqT]() {}
    }

    val listener = next.startCall(call, metadata)

    new SimpleForwardingServerCallListener[ReqT](listener) {
      override def onMessage(message: ReqT): Unit = {
        message match {
          case msg: Message =>
            Option(msg.getDescriptorForType.findFieldByName("user_context")) match {
              case Some(userContextField) =>
                val userContext = msg.getField(userContextField).asInstanceOf[proto.UserContext]
                val userId = userContext.getUserId
                if (userId.isEmpty) {
                  call.close(Status.PERMISSION_DENIED.withDescription(
                    "User ID is missing"), new Metadata())
                } else {
                  userKeys.get(userId) match {
                    case Some(expectedAccessKey) =>
                      if (expectedAccessKey != accessKeyHeaderValue) {
                        call.close(Status.UNAUTHENTICATED.withDescription(
                          s"Invalid access key for user $userId"), new Metadata())
                      }
                    case None =>
                      call.close(Status.PERMISSION_DENIED.withDescription(
                        s"User ($userId) is not allowed to access the Spark service"), new Metadata())
                  }
                }
              case None =>
              // Proceed without auth check if no user_context field (assuming not all messages require it)
            }
          case _ =>
          // Non-protobuf message; proceed or handle as needed (unlikely in this context)
        }
        super.onMessage(message)
      }
    }
  }
}