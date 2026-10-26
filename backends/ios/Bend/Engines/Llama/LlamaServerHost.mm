#import "LlamaServerHost.h"
#include "llama_server_embed.h"
#include <pthread.h>
#include <string>
#include <vector>

static std::vector<std::string> g_args;
static pthread_t g_thread;
static bool g_running = false;
static int g_exit_code = 0;
static NSCondition * g_cond;

static void * server_main(void *) {
    std::vector<char *> argv;
    for (auto & a : g_args) argv.push_back(a.data());
    argv.push_back(nullptr);
    int rc = llama_server((int) g_args.size(), argv.data());
    [g_cond lock];
    g_exit_code = rc;
    g_running = false;
    [g_cond broadcast];
    [g_cond unlock];
    NSLog(@"BEND engine llama_server returned %d", rc);
    return nullptr;
}

@implementation LlamaServerHost

+ (void)initialize { if (!g_cond) g_cond = [NSCondition new]; }

+ (BOOL)startWithArguments:(NSArray<NSString *> *)args {
    [g_cond lock];
    if (g_running) { [g_cond unlock]; return NO; }
    g_args.clear();
    g_args.emplace_back("llama-server");
    for (NSString * a in args) g_args.emplace_back(a.UTF8String);
    g_running = true;
    [g_cond unlock];

    // llama-server expects a main-thread-sized stack; secondary threads default to 512 KB on iOS
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
    int err = pthread_create(&g_thread, &attr, server_main, nullptr);
    pthread_attr_destroy(&attr);
    if (err != 0) {
        [g_cond lock]; g_running = false; [g_cond unlock];
        return NO;
    }
    return YES;
}

+ (int)stop {
    [g_cond lock];
    bool running = g_running;
    [g_cond unlock];
    if (!running) return g_exit_code;
    // only valid while llama_server() is still on the stack (its shutdown handler captures locals)
    llama_server_terminate();
    pthread_join(g_thread, nullptr);
    return g_exit_code;
}

+ (BOOL)isRunning {
    [g_cond lock];
    bool r = g_running;
    [g_cond unlock];
    return r;
}

@end
