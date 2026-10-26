// metrics_demo.c: CPU-only exercise of the metrics API + standalone listener.
// metrics_demo <port> <n_requests> <linger_s>: n requests with real sleeps
// (queue 1 ms, prefill 2 ms + 0.1 ms/prompt token, decode 8-12 ms/token), then
// serves /metrics for linger_s seconds so curl can scrape it.
#define _POSIX_C_SOURCE 200809L
#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void nap(double s) {
    struct timespec t = {(time_t)s, (long)((s - (time_t)s) * 1e9)};
    nanosleep(&t, NULL);
}

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : 9432, n = argc > 2 ? atoi(argv[2]) : 20;
    double linger = argc > 3 ? atof(argv[3]) : 5;
    if (llb_metrics_listen("127.0.0.1", port)) { perror("listen"); return 1; }
    for (int i = 0; i < n; i++) {
        llb_req r;
        llb_req_begin(&r);
        llb_metrics_queue_depth(n - 1 - i);
        nap(.001);
        int prompt = 20 + 7 * i, cached = i % 5 == 4 ? prompt : (i % 3 == 0 ? prompt / 2 : 0);
        llb_req_start(&r, prompt, cached);
        if (cached == prompt) {       // full cache hit: answered without the engine
            llb_req_end(&r, "generate", "retained");
            continue;
        }
        nap(.002 + 1e-4 * (prompt - cached));
        int gen = 4 + i % 6;
        for (int k = 0; k < gen; k++) {
            if (k) nap(.008 + .001 * (k % 5));
            llb_req_token(&r);
        }
        if (i % 4 == 1) llb_req_spec(&r, 8, 6);
        llb_req_end(&r, "generate", "ok");
        if (i == 0) {
            char t[1024];
            llb_req_timings_json(&r, t, sizeof t);
            printf("timings[0] %s\n", t);
            fflush(stdout);
        }
    }
    llb_metrics_queue_depth(0);
    printf("done %d requests; serving for %.1f s\n", n, linger);
    fflush(stdout);
    nap(linger);
    return 0;
}
