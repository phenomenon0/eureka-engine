// host/llb_sampler.h — the sampler pack (W1-SAMPLER): llama.cpp-default-chain sampling,
// seed control, logprobs, stop strings, and the lossless speculative verify step.
//
// Chain (llama.cpp common/sampling.cpp default order, the off-by-default stages omitted):
//   logit_bias -> penalties -> top_k -> top_p -> min_p -> temperature -> dist
// A stage whose parameter is "off" is a no-op: top_k<=0, top_p>=1, min_p<=0, temp==1,
// penalties off when repeat_last_n<=0 or (repeat==1 && frequency==0 && presence==0).
// temp<=0 = greedy: the first max (lowest id on ties) of the biased/penalised logits,
// which is the device argmax (gpu_amax, first index) when bias and penalties are off.
// The stage code is tools/qwen35_sampler.c's sampler v2, gated stage-by-stage against
// libllama (reports/llb-sampler2.md, 177 cases); this file renames it, keys the RNG by
// position and adds verify / history / parsing.
//
// RNG: counter-based. Every draw is splitmix64(seed, pos, k) — the k-th draw for the token
// generated at stream position pos. Equal (seed, pos, k) -> equal draw, no hidden state, so a
// stream is deterministic given the seed regardless of how many earlier calls happened
// (prefill tier, continuation, a re-run cycle). NOT llama.cpp's mt19937: equal seeds do not
// give llama.cpp's tokens; distributions are compared instead (tools/sampler/README.md).
//
// Spec decode (llb_smp_verify): Leviathan, Kalman & Matias 2023, "Fast Inference from
// Transformers via Speculative Decoding", Algorithm 1 / Appendix A.1 (also Chen et al. 2023
// "Accelerating LLM Decoding with Speculative Sampling"): accept the draft x with probability
// min(1, p(x)/q(x)); on reject sample from norm(max(0, p - q)). The emitted token is then
// distributed exactly as p, whatever q is. Our drafters (prompt lookup, MTP argmax) are
// deterministic, so q is one-hot at the draft: accept w.p. p(x), else sample p with x removed.
// Under greedy the rule degenerates to "accept iff x == argmax", the old served-row check.
//
// Every function is plain C11, no allocation on the per-token path except llb_smp_new.
// The engine inlines this header and llb_sampler_impl.h into the emitted C (Bend comp.ts
// inlines #include "llb_*.h"); standalone users compile host/llb_sampler.c.
#ifndef LLB_SAMPLER_H
#define LLB_SAMPLER_H

#include <stdint.h>

#define LLB_SMP_MAX_BIAS 64
#define LLB_SMP_MAX_LOGPROBS 20

typedef struct {
  float temp, top_p, min_p;   // temp<=0 greedy; top_p>=1 off; min_p<=0 off
  int top_k;                  // <=0 off
  float repeat_penalty;       // 1 off
  int repeat_last_n;          // window (prompt tokens count, as llama.cpp's completion); <=0 off
  float presence, frequency;  // 0 off
  uint64_t seed;
  int min_keep;               // llama's common chain passes 0
  int n_logprobs;             // top-n logprobs of the RAW logits to report, 0..LLB_SMP_MAX_LOGPROBS
  int max_tokens;             // a policy field for the caller (0 = caller default); unused here
  int n_bias;
  int bias_ids[LLB_SMP_MAX_BIAS];
  float bias_vals[LLB_SMP_MAX_BIAS];
} LlbSmpParams;

// greedy, every stage off, seed 0, no logprobs
void llb_smp_defaults(LlbSmpParams* p);
// "key=val,key=val" over: temp top_p top_k min_p repeat_penalty repeat_last_n presence
// frequency seed min_keep logprobs max_tokens bias=ID:VAL[;ID:VAL...]. Also preset=greedy|
// qwen_thinking|qwen_non_thinking (applied where it appears, later keys override).
// Returns 0, or -1 with a message in err.
int llb_smp_parse(LlbSmpParams* p, const char* spec, char* err, int errlen);
// 1 when sampling leaves the device argmax untouched: greedy, no bias, penalties off.
int llb_smp_is_argmax(const LlbSmpParams* p);

typedef struct LlbSmp LlbSmp;
LlbSmp* llb_smp_new(const LlbSmpParams* p, int n_vocab);
void llb_smp_free(LlbSmp* s);
const LlbSmpParams* llb_smp_params(const LlbSmp* s);
// penalty window = the last repeat_last_n of toks[0..n) (prompt + generated so far)
void llb_smp_history(LlbSmp* s, const int* toks, int n);
// streaming alternative: push one token into the window
void llb_smp_accept(LlbSmp* s, int tok);

// The token for stream position pos (the position it will occupy), from that row's logits.
int llb_smp_sample(LlbSmp* s, const float* logits, uint64_t pos);
// Speculative verify at pos: draft = the drafter's token for pos; q = the drafter's
// distribution (V floats) or NULL for a deterministic drafter (one-hot). Returns the token
// to emit; *accepted = 1 iff it is the draft. Exactly p-distributed (Leviathan Alg. 1).
int llb_smp_verify(LlbSmp* s, const float* logits, int draft, const float* q, uint64_t pos, int* accepted);
// The target distribution the two above draw from: survivors into ids/p (V-sized), returns n.
int llb_smp_dist(LlbSmp* s, const float* logits, int* ids, float* p);
// Test hook: stages 0..upto (0 bias, 1 penalties, 2 top_k, 3 top_p, 4 min_p, 5 temp,
// 6 dist-probabilities) as reports/llb-sampler2.md gate 1.
int llb_smp_trace(LlbSmp* s, const float* logits, int upto, int* ids, float* lg, float* p_dist);

// top-n of log_softmax(RAW logits) (before bias/penalties/temp: the model distribution, as
// OpenAI "logprobs"); ties lowest id first. Returns the log-sum-exp; lp_tok (may be NULL)
// gets tok's logprob.
double llb_smp_logprobs(const float* logits, int V, int n, int* ids, float* lp, int tok, float* lp_tok);

// Stop strings over detokenised bytes, streaming-safe: holds back any suffix that is a
// proper prefix of a stop string and any incomplete trailing UTF-8 sequence. On a match the
// output is trimmed at the (leftmost) match start and stopped = 1.
typedef struct LlbStop LlbStop;
LlbStop* llb_stop_new(const char* const* stops, int n);
void llb_stop_free(LlbStop* s);
const char* llb_stop_feed(LlbStop* s, const char* bytes, int len, int* outlen);
const char* llb_stop_flush(LlbStop* s, int* outlen);
int llb_stop_stopped(const LlbStop* s);

#endif
