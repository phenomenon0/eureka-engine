// tools/qwen35_sampler.h — deterministic greedy + temperature/top-p.
//
// Order of operations matches llama.cpp's default sampler chain
// (common/common.h): ... TOP_P -> TEMPERATURE -> DIST. So top-p is applied to
// softmax(logits) at temp=1; the kept logits are then divided by temp and
// sampled from softmax(logits/temp). min_keep = 1, matching llama.cpp.
// Default is greedy (temp <= 0). The RNG is a 64-bit xorshift (NOT llama.cpp's
// std::mt19937), so equal seeds do not imply equal draws.
#ifndef QWEN35_SAMPLER_H
#define QWEN35_SAMPLER_H

#include <stdint.h>

typedef struct {
    float temp;       // <= 0 means greedy
    float top_p;      // 1.0 disables
    uint64_t rng;
} Q35Sampler;

void q35_sampler_init(Q35Sampler *s, float temp, float top_p, uint64_t seed);

// Returns the chosen token id. `probs`/`idx` are caller scratch of size V.
int q35_sample(Q35Sampler *s, const float *logits, int V, float *probs, int *idx);

#endif
