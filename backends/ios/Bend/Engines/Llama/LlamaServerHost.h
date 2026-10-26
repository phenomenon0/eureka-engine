#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs the real llama.cpp llama-server (tools/server, statically linked from llama.xcframework)
/// inside this process on a background thread, listening on 127.0.0.1:<port>.
/// One server at a time (llama-server keeps process-global state).
@interface LlamaServerHost : NSObject
/// args are llama-server CLI arguments, e.g. @[@"-m", path, @"--port", @"8931"]. Returns NO if one is already running.
+ (BOOL)startWithArguments:(NSArray<NSString *> *)args;
/// Ask the server to shut down and block until llama_server() has returned. Returns its exit code.
+ (int)stop;
+ (BOOL)isRunning;
@end

NS_ASSUME_NONNULL_END
