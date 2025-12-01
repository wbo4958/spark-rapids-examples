package com.nvidia.proxy;

import io.grpc.*;
import io.grpc.stub.ClientCalls;
import io.grpc.stub.ServerCalls;
import io.grpc.stub.StreamObserver;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class GrpcGateway {

    // Load Balancer State
    private final List<ManagedChannel> backends;
    private final AtomicInteger counter = new AtomicInteger(0);

    public GrpcGateway(List<String> targets) {
        this.backends = targets.stream()
                .map(target -> ManagedChannelBuilder.forTarget(target).usePlaintext().build())
                .toList(); // JDK 17+ syntax
    }

    private ManagedChannel getNextBackend() {
        if (backends.isEmpty()) throw new IllegalStateException("No backends available");
        return backends.get(Math.abs(counter.getAndIncrement() % backends.size()));
    }

    public void start() throws Exception {
        // 1. Define Interceptor to capture Headers
        ServerInterceptor headerCapturer = new ServerInterceptor() {
            @Override
            public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                    ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
                Context ctx = Context.current().withValue(ContextUtils.HEADERS, headers);
                return Contexts.interceptCall(ctx, call, headers, next);
            }
        };

        // 2. Build Server
        Server server = ServerBuilder.forPort(50051)
                .intercept(headerCapturer)
                .fallbackHandlerRegistry(this.new TransparentRegistry())
                .build()
                .start();

        System.out.println("Gateway started on :50051. Waiting for HTTP/2 frames...");

        // Cleanup hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down...");
            backends.forEach(ManagedChannel::shutdownNow);
            server.shutdown();
        }));

        server.awaitTermination();
    }

    // --- INNER CLASS: REGISTRY ---
    private class TransparentRegistry extends HandlerRegistry {
        @Override
        public ServerMethodDefinition<?, ?> lookupMethod(String methodName, String authority) {
            MethodDescriptor<byte[], byte[]> methodDesc = MethodDescriptor.<byte[], byte[]>newBuilder()
                    .setType(MethodDescriptor.MethodType.BIDI_STREAMING)
                    .setFullMethodName(methodName)
                    .setRequestMarshaller(byteMarshaller())
                    .setResponseMarshaller(byteMarshaller())
                    .build();

            // *** FIX IS HERE ***
            // We must convert our 'BidiStreamingMethod' implementation into a 'ServerCallHandler'
            // using the ServerCalls utility.
            return ServerMethodDefinition.create(
                    methodDesc,
                    ServerCalls.asyncBidiStreamingCall(new ForwardingCallHandler(methodName))
            );
        }
    }

    // --- INNER CLASS: THE HANDLER ---
    private class ForwardingCallHandler implements ServerCalls.BidiStreamingMethod<byte[], byte[]> {
        private final String methodName;

        public ForwardingCallHandler(String methodName) {
            this.methodName = methodName;
        }

        @Override
        public StreamObserver<byte[]> invoke(StreamObserver<byte[]> responseObserver) {
            Metadata clientHeaders = ContextUtils.HEADERS.get();
            String token = clientHeaders.get(Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER));
            System.out.println("Gateway: Intercepted " + methodName + " | Token: " + token);
            if (token == null || !token.equals("my-secret-token")) {
//                responseObserver.onError(Status.UNAUTHENTICATED
//                        .withDescription("Missing or Invalid Authorization Header").asRuntimeException());
//                return new StreamObserver<>() {
//                    public void onNext(byte[] v) {}
//                    public void onError(Throwable t) {}
//                    public void onCompleted() {}
//                };
            }

            ManagedChannel backend = getNextBackend();

            MethodDescriptor<byte[], byte[]> outgoingMethod = MethodDescriptor.<byte[], byte[]>newBuilder()
                    .setType(MethodDescriptor.MethodType.BIDI_STREAMING)
                    .setFullMethodName(methodName)
                    .setRequestMarshaller(byteMarshaller())
                    .setResponseMarshaller(byteMarshaller())
                    .build();

            ClientCall<byte[], byte[]> call = backend.newCall(outgoingMethod, CallOptions.DEFAULT);

            call.start(new ClientCall.Listener<byte[]>() {
                @Override
                public void onMessage(byte[] message) {
                    responseObserver.onNext(message);
                    call.request(1); // Ask for next
                }

                @Override
                public void onClose(Status status, Metadata trailers) {
                    if (status.isOk()) responseObserver.onCompleted();
                    else responseObserver.onError(status.asRuntimeException(trailers));
                }
            }, clientHeaders);

            call.request(1); // Ask for first

            return new StreamObserver<>() {
                @Override
                public void onNext(byte[] value) {
                    call.sendMessage(value);
                }

                @Override
                public void onError(Throwable t) {
                    call.cancel("Proxy cancelled", t);
                }

                @Override
                public void onCompleted() {
                    call.halfClose();
                }
            };
        }
    }

    // Helper: Marshals raw bytes (JDK 9+ compatible)
    private static MethodDescriptor.Marshaller<byte[]> byteMarshaller() {
        return new MethodDescriptor.Marshaller<>() {
            public InputStream stream(byte[] value) {
                return new ByteArrayInputStream(value);
            }

            public byte[] parse(InputStream stream) {
                try {
                    return stream.readAllBytes(); // JDK 9+ Method
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        };
    }

    static class ContextUtils {
        public static final Context.Key<Metadata> HEADERS = Context.key("grpc-headers");
    }

    public static void main(String[] args) throws Exception {
//        List<String> targets = List.of("localhost:50052", "localhost:50053");
        List<String> targets = List.of("localhost:50052");
        new GrpcGateway(targets).start();
    }
}