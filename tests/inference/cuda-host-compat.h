// Test-only shims for compiling the Linux HTTP front on macOS. No GPU code is linked.
#include <unistd.h>
#include <fcntl.h>
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
static inline int inference_test_pipe2(int p[2], int flags) {
    int result = pipe(p);
    if (!result && (flags & O_CLOEXEC)) {
        fcntl(p[0], F_SETFD, FD_CLOEXEC);
        fcntl(p[1], F_SETFD, FD_CLOEXEC);
    }
    return result;
}
#define pipe2 inference_test_pipe2
