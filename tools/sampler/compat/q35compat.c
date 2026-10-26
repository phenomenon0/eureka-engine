// q35compat.c: the sampler2 gate API (tools/qwen35_sampler.h "sampler v2") over the new
// host/llb_sampler module, so reports/llb-sampler2.md's libllama-oracle gates 1-4 re-run
// unchanged on the code the engine now inlines. Draws use a per-sampler counter as the
// position (llb_smp_draw is keyed by (seed, pos, k)), so consecutive calls draw fresh numbers.
#include "qwen35_sampler.h"
#include "../../../host/llb_sampler_impl.h"

void q35_params_default(Q35Params *p, int mode) {
    memset(p, 0, sizeof *p);
    p->temp = 0.0f; p->top_p = 1.0f; p->repeat_penalty = 1.0f; p->repeat_last_n = 64;
    if (mode == 1) { p->temp = 1.0f; p->top_p = 0.95f; p->top_k = 20; p->presence = 1.5f; }
    else if (mode == 2) { p->temp = 0.7f; p->top_p = 0.8f; p->top_k = 20; p->presence = 1.5f; }
}

struct Q35S { LlbSmp *s; uint64_t n; };

Q35S *q35s_new(const Q35Params *P, int V) {
    LlbSmpParams L;
    llb_smp_defaults(&L);
    L.temp = P->temp; L.top_p = P->top_p; L.min_p = P->min_p; L.top_k = P->top_k;
    L.repeat_penalty = P->repeat_penalty; L.repeat_last_n = P->repeat_last_n;
    L.presence = P->presence; L.frequency = P->frequency; L.seed = P->seed; L.min_keep = P->min_keep;
    L.n_bias = P->n_bias < LLB_SMP_MAX_BIAS ? P->n_bias : LLB_SMP_MAX_BIAS;
    for (int i = 0; i < L.n_bias; i++) { L.bias_ids[i] = P->bias_ids[i]; L.bias_vals[i] = P->bias_vals[i]; }
    Q35S *q = (Q35S *)calloc(1, sizeof *q);
    q->s = llb_smp_new(&L, V);
    return q;
}
void q35s_free(Q35S *q) { if (q) { llb_smp_free(q->s); free(q); } }
void q35s_accept(Q35S *q, int t) { llb_smp_accept(q->s, t); }
int q35s_sample(Q35S *q, const float *lg, float *sp, int *si) { (void)sp; (void)si; return llb_smp_sample(q->s, lg, q->n++); }
int q35s_trace(Q35S *q, const float *lg, int upto, int *ids, float *out, float *pd) { return llb_smp_trace(q->s, lg, upto, ids, out, pd); }
void q35s_topn_logprobs(const float *lg, int V, int n, int *ids, float *lp) {
    llb_smp_logprobs(lg, V, n, ids, lp, -1, NULL);
}
Q35Stop *q35stop_new(const char *const *s, int n) { return (Q35Stop *)llb_stop_new(s, n); }
void q35stop_free(Q35Stop *s) { llb_stop_free((LlbStop *)s); }
const char *q35stop_feed(Q35Stop *s, const char *b, int len, int *ol) { return llb_stop_feed((LlbStop *)s, b, len, ol); }
const char *q35stop_flush(Q35Stop *s, int *ol) { return llb_stop_flush((LlbStop *)s, ol); }
int q35stop_stopped(const Q35Stop *s) { return llb_stop_stopped((const LlbStop *)s); }
