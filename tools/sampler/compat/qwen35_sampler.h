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


// ===================== sampler v2 =====================================
// Full llama.cpp default chain, written from llama.cpp src/llama-sampler.cpp and
// common/sampling.cpp (see reports/llb-sampler2.md for file:line):
//   logit_bias -> penalties -> top_k -> top_p -> min_p -> temperature -> dist
// A stage whose parameter is "off" is a no-op (llama's "empty" sampler):
//   top_k<=0, top_p>=1, min_p<=0, temp==1, penalties off when last_n==0 or
//   (repeat==1 && freq==0 && presence==0).  temp<=0 = greedy (first max wins).
// TIE RULE: equal logits -> lowest token id first (llama.cpp's own tie order at a
// top_k boundary is unspecified: std::partial_sort).  RNG is splitmix64, NOT mt19937.
// Not handled (like llama.cpp, undefined): NaN logits.  Token ids outside [0,V) are
// ignored by bias/accept.
typedef struct {
    float temp, top_p, min_p;
    int top_k;
    float repeat_penalty;
    int repeat_last_n;           // <=0 disables penalties (llama: max(n,0))
    float presence, frequency;
    uint64_t seed;
    int min_keep;                // llama's common chain passes 0
    int n_bias;                  // logit_bias list, copied by q35s_new
    const int *bias_ids;
    const float *bias_vals;
} Q35Params;

// mode 0 greedy, 1 qwen thinking, 2 qwen non-thinking (tools/sampler2/presets.json)
void q35_params_default(Q35Params *p, int mode);

typedef struct Q35S Q35S;
Q35S *q35s_new(const Q35Params *p, int V);
void q35s_free(Q35S *s);
// History ring for penalties.  Like llama.cpp's completion loop (completion.cpp:695),
// feed the PROMPT tokens too, then every generated token.  No-op when penalties are off.
void q35s_accept(Q35S *s, int token);
// scratch_probs / scratch_idx: caller arrays of V floats / V ints.  Returns the token.
int q35s_sample(Q35S *s, const float *logits, float *scratch_probs, int *scratch_idx);
// Test hook: run stages 0..upto (0 bias,1 penalties,2 top_k,3 top_p,4 min_p,5 temp,
// 6 dist-probabilities-only).  Fills ids/lg (V-sized) with the survivors, returns n.
// If upto==6, p_dist (V floats, may be NULL) gets llama's dist normalised probabilities.
int q35s_trace(Q35S *s, const float *logits, int upto, int *ids, float *lg, float *p_dist);
// Top-n of log_softmax(RAW logits): before bias/penalties/temperature (the OpenAI
// "logprobs" are of the model distribution).  Ties: lowest id first.
void q35s_topn_logprobs(const float *logits, int V, int n, int *ids, float *lp);

// Stop-string matcher over detokenised bytes.  Holds back (a) any suffix that is a
// proper prefix of a stop string and (b) an incomplete trailing UTF-8 sequence.
// On a match the output is trimmed at the match start and stopped=1.
typedef struct Q35Stop Q35Stop;
Q35Stop *q35stop_new(const char *const *stops, int n);
void q35stop_free(Q35Stop *s);
// Feed bytes; returns a pointer (valid until the next call) to the bytes now safe to
// emit, length in *outlen.  q35stop_stopped() tells whether a stop matched.
const char *q35stop_feed(Q35Stop *s, const char *bytes, int len, int *outlen);
// End of generation without a match: release held-back bytes.
const char *q35stop_flush(Q35Stop *s, int *outlen);
int q35stop_stopped(const Q35Stop *s);

#endif
