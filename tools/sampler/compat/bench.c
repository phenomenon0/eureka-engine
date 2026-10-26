#define _POSIX_C_SOURCE 199309L
// CPU microbench: ms per q35s_sample call on V=248320, single thread.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qwen35_sampler.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void run(const char *name, Q35Params P, const float *lg, int V, int hist, int iters) {
    Q35S *S = q35s_new(&P, V);
    for (int i = 0; i < hist; i++) q35s_accept(S, (i * 7919) % V);
    float *sp = malloc(4 * (size_t)V); int *si = malloc(4 * (size_t)V);
    double best = 1e9, tot = 0; int sink = 0;
    for (int i = 0; i < iters; i++) { double t0 = now(); sink += q35s_sample(S, lg, sp, si); double d = now() - t0; tot += d; if (d < best) best = d; }
    printf("%-44s mean %.3f ms  min %.3f ms  (sink %d)\n", name, tot / iters, best, sink);
    q35s_free(S); free(sp); free(si);
}
int main(void) {
    int V = 248320; float *lg = malloc(4 * (size_t)V);
    srand(7); for (int i = 0; i < V; i++) lg[i] = ((rand() / (float)RAND_MAX) + (rand() / (float)RAND_MAX) + (rand() / (float)RAND_MAX) - 1.5f) * 6.0f;
    Q35Params P;
    q35_params_default(&P, 0);                run("greedy (no penalties)", P, lg, V, 0, 500);
    P.top_k = 20; P.top_p = 0.95f; P.temp = 1.0f; P.seed = 1; run("top_k=20 + top_p=0.95 + dist", P, lg, V, 0, 500);
    q35_params_default(&P, 1); P.seed = 1;    run("qwen thinking chain, pres 1.5, history 512 (win 64)", P, lg, V, 512, 500);
    P.repeat_last_n = 512; P.repeat_penalty = 1.1f; P.frequency = 0.1f;
    int bid[3] = {5, 99, 1234}; float bv[3] = {-100, 2, -1}; P.n_bias = 3; P.bias_ids = bid; P.bias_vals = bv; P.min_p = 0.05f;
    run("FULL: bias3+pen(512)+top_k20+top_p+min_p+temp+dist", P, lg, V, 512, 500);
    P.top_k = 0; run("FULL but top_k off (top_p over all V)", P, lg, V, 512, 200);
    P.top_k = 400; run("FULL top_k=400 (full-sort path)", P, lg, V, 512, 50);
    // peaked logits (LLM-like: ~40 dominant tokens): top_k off, top_p resolves inside the 256-prefix
    float *pk = malloc(4 * (size_t)V);
    for (int i = 0; i < V; i++) pk[i] = lg[i] * 0.25f;
    for (int j = 0; j < 40; j++) pk[(j * 6151) % V] = 12.0f + (j % 7) * 0.5f;
    P.top_k = 0;
    run("FULL top_k off, PEAKED logits", P, pk, V, 512, 200);
    return 0;
}
