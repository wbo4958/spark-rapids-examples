# ConnectPlugin Example

This module builds a standalone JAR that implements `com.nvidia.proxy.ConnectPlugin`
and registers itself with `ServiceLoader`.

## Build

```bash
# Install the grpc-proxy artifact locally so the interface is available
mvn -f ../grpc-service/pom.xml install

# Build the plugin JAR
mvn -f ./pom.xml package
```

## Deploy

Copy the built JAR (`target/com.nvidia.connect-plugin-1.0-SNAPSHOT.jar`) into the
class path of the gRPC proxy service (for example, alongside `grpc-service` JARs
inside the container or image). On startup, the proxy uses `ServiceLoader` to
discover and load this plugin automatically.

