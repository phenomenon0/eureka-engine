// tools/qwen35_sampler.c — see qwen35_sampler.h
#include <math.h>
#include <stdlib.h>
#include "qwen35_sampler.h"

void q35_sampler_init(Q35Sampler *s, float temp, float top_p, uint64_t seed) {
    s->temp = temp;
    s->top_p = top_p;
    s->rng = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static uint64_t xorshift64(uint64_t *st) {
    uint64_t x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    return x;
}

static int spl_argmax(const float *z, int n) {
    int bi = 0; float bv = z[0];
    for (int i = 1; i < n; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
    return bi;
}

typedef struct { int i; float p; } ProbId;
static int cmp_desc(const void *a, const void *b) {
    float pa = ((const ProbId *)a)->p, pb = ((const ProbId *)b)->p;
    if (pa < pb) return 1;
    if (pa > pb) return -1;
    return ((const ProbId *)a)->i - ((const ProbId *)b)->i;
}

int q35_sample(Q35Sampler *s, const float *logits, int V, float *probs, int *idx) {
    if (s->temp <= 0.0f) return spl_argmax(logits, V);

    // top-p over softmax(logits) at temp = 1 (order: TOP_P before TEMPERATURE)
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) { probs[i] = (float)exp((double)(logits[i] - mx)); sum += probs[i]; }
    for (int i = 0; i < V; i++) probs[i] = (float)(probs[i] / sum);

    int kept = V;
    if (s->top_p < 1.0f) {
        ProbId *order = (ProbId *)malloc(sizeof(ProbId) * V);
        for (int i = 0; i < V; i++) { order[i].i = i; order[i].p = probs[i]; }
        qsort(order, V, sizeof(ProbId), cmp_desc);
        double cum = 0.0;
        kept = V;
        for (int i = 0; i < V; i++) {
            cum += order[i].p;
            if (cum >= (double)s->top_p) { kept = i + 1; break; }   // min_keep = 1
        }
        for (int i = 0; i < kept; i++) idx[i] = order[i].i;
        free(order);
    } else {
        for (int i = 0; i < V; i++) idx[i] = i;
    }

    // temperature then softmax over the kept set
    float kmx = logits[idx[0]] / s->temp;
    for (int i = 1; i < kept; i++) { float v = logits[idx[i]] / s->temp; if (v > kmx) kmx = v; }
    double ksum = 0.0;
    for (int i = 0; i < kept; i++) { probs[i] = (float)exp((double)(logits[idx[i]] / s->temp - kmx)); ksum += probs[i]; }

    // dist: cumulative sample
    double r = (double)(xorshift64(&s->rng) >> 11) * (1.0 / 9007199254740992.0);
    double acc = 0.0;
    for (int i = 0; i < kept; i++) {
        acc += probs[i] / ksum;
        if (r < acc) return idx[i];
    }
    return idx[kept - 1];
}
