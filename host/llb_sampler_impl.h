// host/llb_sampler_impl.h — the sampler pack's implementation (API: llb_sampler.h).
// Stage code ported from tools/qwen35_sampler.c sampler v2 (gated vs libllama,
// reports/llb-sampler2.md); float arithmetic mirrors llama.cpp on purpose (float cum sums,
// expf): build with -ffp-contract=off (the engine's remap_cc.sh does). MUT_* macros are gate
// mutants (tools/sampler/mutants.sh), never defined in a real build.
#ifndef LLB_SAMPLER_IMPL_H
#define LLB_SAMPLER_IMPL_H
#include "llb_sampler.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void llb_smp_defaults(LlbSmpParams* p) {
  memset(p, 0, sizeof *p);
  p->temp = 0.0f; p->top_p = 1.0f; p->repeat_penalty = 1.0f; p->repeat_last_n = 64;
}

static void llb_smp_preset(LlbSmpParams* p, const char* v) {
  // Qwen3.5 card presets (tools/sampler2/presets.json); greedy = defaults
  if (!strcmp(v, "qwen_thinking")) { p->temp = 1.0f; p->top_p = 0.95f; p->top_k = 20; p->presence = 1.5f; }
  else if (!strcmp(v, "qwen_non_thinking")) { p->temp = 0.7f; p->top_p = 0.8f; p->top_k = 20; p->presence = 1.5f; }
}

int llb_smp_parse(LlbSmpParams* p, const char* spec, char* err, int errlen) {
  char buf[1024];
  if (!spec) return 0;
  if (strlen(spec) >= sizeof buf) { snprintf(err, (size_t)errlen, "sampler spec too long"); return -1; }
  strcpy(buf, spec);
  for (char* kv = strtok(buf, ","); kv; kv = strtok(NULL, ",")) {
    char* eq = strchr(kv, '=');
    if (!eq) { snprintf(err, (size_t)errlen, "sampler: '%s' is not key=value", kv); return -1; }
    *eq = 0;
    const char *k = kv, *v = eq + 1;
    char* end = NULL;
    double d = strtod(v, &end);
    int num = end != v && *end == 0;
    if (!strcmp(k, "preset")) {
      if (strcmp(v, "greedy") && strcmp(v, "qwen_thinking") && strcmp(v, "qwen_non_thinking")) {
        snprintf(err, (size_t)errlen, "sampler: unknown preset '%s'", v); return -1;
      }
      if (!strcmp(v, "greedy")) llb_smp_defaults(p); else llb_smp_preset(p, v);
      continue;
    }
    if (!strcmp(k, "bias")) {             // ID:VAL;ID:VAL
      for (char *e = (char*)v, *nx; e && *e; e = nx) {
        nx = strchr(e, ';');
        if (nx) *nx++ = 0;
        char* c = strchr(e, ':');
        if (!c || p->n_bias >= LLB_SMP_MAX_BIAS) { snprintf(err, (size_t)errlen, "sampler: bad bias '%s'", e); return -1; }
        *c = 0;
        p->bias_ids[p->n_bias] = atoi(e);
        p->bias_vals[p->n_bias] = !strcmp(c + 1, "-inf") ? -INFINITY : strtof(c + 1, NULL);
        p->n_bias += 1;
      }
      continue;
    }
    if (!num) { snprintf(err, (size_t)errlen, "sampler: '%s=%s' is not a number", k, v); return -1; }
    if (!strcmp(k, "temp") || !strcmp(k, "temperature")) p->temp = (float)d;
    else if (!strcmp(k, "top_p")) p->top_p = (float)d;
    else if (!strcmp(k, "top_k")) p->top_k = (int)d;
    else if (!strcmp(k, "min_p")) p->min_p = (float)d;
    else if (!strcmp(k, "repeat_penalty")) p->repeat_penalty = (float)d;
    else if (!strcmp(k, "repeat_last_n")) p->repeat_last_n = (int)d;
    else if (!strcmp(k, "presence") || !strcmp(k, "presence_penalty")) p->presence = (float)d;
    else if (!strcmp(k, "frequency") || !strcmp(k, "frequency_penalty")) p->frequency = (float)d;
    else if (!strcmp(k, "seed")) p->seed = strtoull(v, NULL, 10);
    else if (!strcmp(k, "min_keep")) p->min_keep = (int)d;
    else if (!strcmp(k, "logprobs")) p->n_logprobs = d < 0 ? 0 : d > LLB_SMP_MAX_LOGPROBS ? LLB_SMP_MAX_LOGPROBS : (int)d;
    else if (!strcmp(k, "max_tokens")) p->max_tokens = (int)d;
    else { snprintf(err, (size_t)errlen, "sampler: unknown key '%s'", k); return -1; }
  }
  return 0;
}

static int llb_smp_pen_off(const LlbSmpParams* P) {
  return P->repeat_last_n <= 0 || (P->repeat_penalty == 1.0f && P->frequency == 0.0f && P->presence == 0.0f);
}

int llb_smp_is_argmax(const LlbSmpParams* p) {
  return p->temp <= 0.0f && p->n_bias == 0 && llb_smp_pen_off(p);
}

struct LlbSmp {
  LlbSmpParams P;
  int V;
  int *ring, ring_head, ring_size;        // penalty window (capacity P.repeat_last_n)
  int* cnt;                               // per-token count inside the window
  int* ids; float *lg, *p;                // V-sized scratch
  uint64_t seq;                           // draw counter for callers without a position
};

LlbSmp* llb_smp_new(const LlbSmpParams* P, int V) {
  LlbSmp* s = (LlbSmp*)calloc(1, sizeof *s);
  if (!s) return NULL;
  s->P = *P; s->V = V;
  if (s->P.repeat_last_n < 0) s->P.repeat_last_n = 0;
  if (!llb_smp_pen_off(&s->P)) {
    s->ring = (int*)calloc((size_t)s->P.repeat_last_n, sizeof(int));
    s->cnt = (int*)calloc((size_t)V, sizeof(int));
  }
  s->ids = (int*)malloc(sizeof(int) * (size_t)V);
  s->lg = (float*)malloc(sizeof(float) * (size_t)V);
  s->p = (float*)malloc(sizeof(float) * (size_t)V);
  if (!s->ids || !s->lg || !s->p || (!llb_smp_pen_off(&s->P) && (!s->ring || !s->cnt))) {
    llb_smp_free(s);
    return NULL;
  }
  return s;
}

void llb_smp_free(LlbSmp* s) {
  if (!s) return;
  free(s->ring); free(s->cnt); free(s->ids); free(s->lg); free(s->p); free(s);
}

const LlbSmpParams* llb_smp_params(const LlbSmp* s) { return &s->P; }

void llb_smp_accept(LlbSmp* s, int t) {
  if (!s->ring || t < 0 || t >= s->V) return;
  int cap = s->P.repeat_last_n;
  s->cnt[t]++;
  if (s->ring_size >= cap) {              // full: drop the oldest first
    s->cnt[s->ring[s->ring_head]]--;
    s->ring[s->ring_head] = t;
    s->ring_head = (s->ring_head + 1) % cap;
  } else {
    s->ring[(s->ring_head + s->ring_size) % cap] = t;
    s->ring_size++;
  }
}

void llb_smp_history(LlbSmp* s, const int* toks, int n) {
  if (!s->ring) return;
  for (int k = 0; k < s->ring_size; k++) s->cnt[s->ring[(s->ring_head + k) % s->P.repeat_last_n]] = 0;
  s->ring_head = 0; s->ring_size = 0;
  int i0 = n > s->P.repeat_last_n ? n - s->P.repeat_last_n : 0;
  for (int i = i0; i < n; i++) llb_smp_accept(s, toks[i]);
}

// splitmix64 of (seed, pos, k) -> [0,1) with 53 bits
static double llb_smp_draw(const LlbSmp* s, uint64_t pos, uint32_t k) {
  uint64_t z = s->P.seed ^ 0x6A09E667F3BCC909ull;
  z += (pos * 4 + k + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
#ifdef MUT_RNG_BIAS
  return (double)(z >> 11) * (1.0 / 9007199254740992.0) * 0.9;
#else
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
#endif
}

typedef struct { float l; int id; } LlbSmpCand;
static int llb_smp_cmp(const void* a, const void* b) {   // logit desc, id asc (total order)
  const LlbSmpCand *x = (const LlbSmpCand*)a, *y = (const LlbSmpCand*)b;
  if (x->l > y->l) return -1;
  if (x->l < y->l) return 1;
  return (x->id > y->id) - (x->id < y->id);
}

static void llb_smp_sort_all(int* ids, float* lg, int n) {
  LlbSmpCand* c = (LlbSmpCand*)malloc(sizeof(LlbSmpCand) * (size_t)n);
  for (int i = 0; i < n; i++) { c[i].l = lg[i]; c[i].id = ids[i]; }
  qsort(c, (size_t)n, sizeof(LlbSmpCand), llb_smp_cmp);
  for (int i = 0; i < n; i++) { lg[i] = c[i].l; ids[i] = c[i].id; }
  free(c);
}

#define LLB_SMP_SEL 256
// top-m (m <= LLB_SMP_SEL) of an unsorted array in candidate order into out[] sorted desc;
// earlier position wins ties. Partial selection, O(n).
static void llb_smp_select(const int* ids, const float* lg, int n, int m, LlbSmpCand* out) {
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    float l = lg[i];
    if (cnt == m && !(l > out[m - 1].l)) continue;
    int pos = cnt < m ? cnt++ : m - 1;
    while (pos > 0 && out[pos - 1].l < l) { out[pos] = out[pos - 1]; pos--; }
    out[pos].l = l; out[pos].id = ids[i];
  }
}

static void llb_smp_top_k(int* ids, float* lg, int* n, int* sorted, int k) {
  if (k <= 0) return;
  if (k > *n) k = *n;
  if (!*sorted) {
    if (k <= LLB_SMP_SEL) {
      LlbSmpCand top[LLB_SMP_SEL];
      llb_smp_select(ids, lg, *n, k, top);
      for (int i = 0; i < k; i++) { ids[i] = top[i].id; lg[i] = top[i].l; }
    } else {
      llb_smp_sort_all(ids, lg, *n);
    }
    *sorted = 1;
  }
  *n = k;
}

// llama_sampler_top_p_apply: float softmax, float cumulative sum over the sorted prefix,
// keep through the first index where cum >= p (and i+1 >= min_keep)
static void llb_smp_top_p(int* ids, float* lg, int* n, int* sorted, float p, int min_keep) {
  if (p >= 1.0f) return;
  int N = *n;
  float mx = lg[0];
  if (!*sorted) for (int i = 1; i < N; i++) if (lg[i] > mx) mx = lg[i];
  float sum = 0.0f;
  for (int i = 0; i < N; i++) sum += expf(lg[i] - mx);
  int last = N;
  if (*sorted) {
    float cum = 0.0f;
    for (int i = 0; i < N; i++) {
      cum += expf(lg[i] - mx) / sum;
#ifdef MUT_TOPP_CUTOFF
      if (cum > p && i + 1 >= min_keep) { last = i + 1; break; }
#else
      if (cum >= p && i + 1 >= min_keep) { last = i + 1; break; }
#endif
    }
    *n = last;
    return;
  }
  int m = N < LLB_SMP_SEL ? N : LLB_SMP_SEL;
  LlbSmpCand top[LLB_SMP_SEL];
  llb_smp_select(ids, lg, N, m, top);
  float cum = 0.0f;
  int found = 0;
  for (int i = 0; i < m; i++) {
    cum += expf(top[i].l - mx) / sum;
#ifdef MUT_TOPP_CUTOFF
    if (cum > p && i + 1 >= min_keep) { last = i + 1; found = 1; break; }
#else
    if (cum >= p && i + 1 >= min_keep) { last = i + 1; found = 1; break; }
#endif
  }
  if (found) {
    for (int i = 0; i < last; i++) { ids[i] = top[i].id; lg[i] = top[i].l; }
  } else if (m < N) {                     // prefix of 256 not enough: full sort, rescan
    llb_smp_sort_all(ids, lg, N);
    cum = 0.0f;
    for (int i = 0; i < N; i++) {
      cum += expf(lg[i] - mx) / sum;
      if (cum >= p && i + 1 >= min_keep) { last = i + 1; break; }
    }
  } else {                                // N <= 256, never reached p: keep all, sorted
    for (int i = 0; i < N; i++) { ids[i] = top[i].id; lg[i] = top[i].l; }
  }
  *n = last;
  *sorted = 1;
}

// llama_sampler_min_p_apply
static void llb_smp_min_p(int* ids, float* lg, int* n, int* sorted, float p, int min_keep) {
  if (p <= 0.0f || *n == 0) return;
  int N = *n;
  if (!*sorted) {                         // unsorted path: filter, keep candidate order
    float mxl = -FLT_MAX;
    for (int i = 0; i < N; i++) if (lg[i] > mxl) mxl = lg[i];
    float min_logit = mxl + logf(p);
    int w = 0;
    for (int i = 0; i < N; i++) if (lg[i] >= min_logit) w++;
    if (w > 0 && w >= min_keep) {
      w = 0;
      for (int i = 0; i < N; i++) if (lg[i] >= min_logit) { ids[w] = ids[i]; lg[w] = lg[i]; w++; }
      *n = w;
      return;
    }
    llb_smp_sort_all(ids, lg, N);         // fallback: sorted path
    *sorted = 1;
  }
  float min_logit = lg[0] + logf(p);
  int i = 1;                              // first token always matches
  for (; i < N; i++) {
#ifdef MUT_DROP_MINKEEP
    if (lg[i] < min_logit) break;
#else
    if (lg[i] < min_logit && i >= min_keep) break;
#endif
  }
  *n = i;
}

// llama_sampler_temp_ext_apply with delta == 0
static void llb_smp_temp(float* lg, int n, float t) {
  if (n == 0 || t == 1.0f) return;
  if (t <= 0.0f) {                        // keep the first max, everything else -inf
    int mi = 0; float ml = lg[0];
    for (int i = 1; i < n; i++) {
      if (lg[i] > ml) { lg[mi] = -INFINITY; mi = i; ml = lg[i]; }
      else lg[i] = -INFINITY;
    }
    return;
  }
#ifdef MUT_DIST_NOTEMP
  return;
#endif
  for (int i = 0; i < n; i++) lg[i] /= t;
}

// llama_sampler_dist_apply's probabilities: p = expf(l - max) / sum (sum in double)
static double llb_smp_probs(const float* lg, int n, int sorted, float* p) {
  float mx = lg[0];
  if (!sorted) for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
  double sum = 0.0;
  for (int i = 0; i < n; i++) { float e = expf(lg[i] - mx); p[i] = e; sum += e; }
  return sum;
}

// llama's draw: walk the UNnormalised p against sum * rnd; normalise p. The selected position.
static int llb_smp_walk(const float* lg, int n, int sorted, float* p, double rnd) {
  if (n <= 0) return -1;
  if (n == 1) { p[0] = 1.0f; return 0; }
  double sum = llb_smp_probs(lg, n, sorted, p), tgt = sum * rnd, run = 0.0;
  int sel = -1;
  for (int i = 0; i < n; i++) {
    if (sel < 0) { run += p[i]; if (run >= tgt) sel = i; }
    p[i] = (float)((double)p[i] / sum);
  }
  return sel < 0 ? n - 1 : sel;
}

// stages 0..upto into s->ids / s->lg; *sorted_out as llama's `sorted` flag; survivors
static int llb_smp_chain(LlbSmp* s, const float* logits, int upto, int* sorted_out) {
  const LlbSmpParams* P = &s->P;
  int V = s->V, n = V, sorted = 0, *ids = s->ids;
  float* lg = s->lg;
  memcpy(lg, logits, sizeof(float) * (size_t)V);
  for (int i = 0; i < V; i++) ids[i] = i;
  if (upto >= 0) {                        // logit_bias first, as common/sampling.cpp
    for (int j = 0; j < P->n_bias; j++) {
      int t = P->bias_ids[j];
      if (t >= 0 && t < V) lg[t] += P->bias_vals[j];
    }
  }
  if (upto >= 1 && s->ring) {             // penalties: once per distinct token in the window
    int cap = P->repeat_last_n;
    for (int k = 0; k < s->ring_size; k++) {
      int t = s->ring[(s->ring_head + k) % cap], c = s->cnt[t];
      if (c <= 0) continue;
      if (lg[t] <= 0) lg[t] *= P->repeat_penalty; else lg[t] /= P->repeat_penalty;
#ifdef MUT_PEN_SIGN
      lg[t] += (float)c * P->frequency + (float)(c > 0) * P->presence;
#else
      lg[t] -= (float)c * P->frequency + (float)(c > 0) * P->presence;
#endif
      s->cnt[t] = -c;
    }
    for (int k = 0; k < s->ring_size; k++) {
      int t = s->ring[(s->ring_head + k) % cap];
      if (s->cnt[t] < 0) s->cnt[t] = -s->cnt[t];
    }
  }
  if (upto >= 2) llb_smp_top_k(ids, lg, &n, &sorted, P->top_k);
  if (upto >= 3) llb_smp_top_p(ids, lg, &n, &sorted, P->top_p, P->min_keep);
  if (upto >= 4) llb_smp_min_p(ids, lg, &n, &sorted, P->min_p, P->min_keep);
  if (upto >= 5) llb_smp_temp(lg, n, P->temp);
  *sorted_out = sorted;
  return n;
}

int llb_smp_trace(LlbSmp* s, const float* logits, int upto, int* ids, float* lg, float* p_dist) {
  int sorted, n = llb_smp_chain(s, logits, upto > 5 ? 5 : upto, &sorted);
  memcpy(ids, s->ids, sizeof(int) * (size_t)n);
  memcpy(lg, s->lg, sizeof(float) * (size_t)n);
  if (upto >= 6 && p_dist) llb_smp_walk(lg, n, sorted, p_dist, llb_smp_draw(s, s->seq++, 0));
  return n;
}

// greedy: top_k/top_p/min_p never remove the first max and temp<=0 keeps it, so the chain
// collapses to bias + penalties then the first max (lowest id on ties)
static int llb_smp_argmax(LlbSmp* s, const float* logits) {
  const float* src = logits;
  if (s->P.n_bias > 0 || s->ring) {
    int sorted;
    llb_smp_chain(s, logits, 1, &sorted);
    src = s->lg;
  }
  int bi = 0; float bv = src[0];
  for (int i = 1; i < s->V; i++) if (src[i] > bv) { bv = src[i]; bi = i; }
  return bi;
}

int llb_smp_sample(LlbSmp* s, const float* logits, uint64_t pos) {
  if (s->P.temp <= 0.0f) return llb_smp_argmax(s, logits);
  int sorted, n = llb_smp_chain(s, logits, 5, &sorted);
  int sel = llb_smp_walk(s->lg, n, sorted, s->p, llb_smp_draw(s, pos, 0));
  return s->ids[sel];
}

int llb_smp_dist(LlbSmp* s, const float* logits, int* ids, float* p) {
  if (s->P.temp <= 0.0f) { ids[0] = llb_smp_argmax(s, logits); p[0] = 1.0f; return 1; }
  int sorted, n = llb_smp_chain(s, logits, 5, &sorted);
  double sum = llb_smp_probs(s->lg, n, sorted, p);
  for (int i = 0; i < n; i++) { ids[i] = s->ids[i]; p[i] = (float)((double)p[i] / sum); }
  return n;
}

int llb_smp_verify(LlbSmp* s, const float* logits, int draft, const float* q, uint64_t pos, int* accepted) {
  if (s->P.temp <= 0.0f) {                // greedy: accept iff the draft is the argmax
    int t = llb_smp_argmax(s, logits);
    *accepted = t == draft;
    return t;
  }
  int sorted, n = llb_smp_chain(s, logits, 5, &sorted);
  double sum = llb_smp_probs(s->lg, n, sorted, s->p);
  double pd = 0.0;
  for (int i = 0; i < n; i++) if (s->ids[i] == draft) pd = (double)s->p[i] / sum;
  double qd = q ? (double)q[draft] : 1.0;
  // accept w.p. min(1, p(x) / q(x)) (q(x) > 0: the drafter proposed x)
  if (qd > 0.0 && llb_smp_draw(s, pos, 1) * qd < pd) {
    *accepted = 1;
    return draft;
  }
  *accepted = 0;
#ifdef MUT_SPEC_NORESID
  {                                       // mutant: resample p itself (biases toward the draft)
    int sel = llb_smp_walk(s->lg, n, sorted, s->p, llb_smp_draw(s, pos, 2));
    return s->ids[sel];
  }
#endif
  // residual norm(max(0, p - q)) over the survivors (outside them p = 0, so the residual is 0)
  double rs = 0.0;
  for (int i = 0; i < n; i++) {
    double pi = (double)s->p[i] / sum, qi = q ? (double)q[s->ids[i]] : (s->ids[i] == draft ? 1.0 : 0.0);
    double r = pi > qi ? pi - qi : 0.0;
    s->p[i] = (float)r;
    rs += r;
  }
  if (!(rs > 0.0)) {                      // p == q exactly: unreachable after a reject; fall back to p
    int sel = llb_smp_walk(s->lg, n, sorted, s->p, llb_smp_draw(s, pos, 2));
    return s->ids[sel];
  }
  double tgt = rs * llb_smp_draw(s, pos, 2), run = 0.0;
  int last = -1;
  for (int i = 0; i < n; i++) {
    if (s->p[i] <= 0.0f) continue;
    last = i;
    run += s->p[i];
    if (run >= tgt) return s->ids[i];
  }
  return s->ids[last];
}

double llb_smp_logprobs(const float* logits, int V, int n, int* ids, float* lp, int tok, float* lp_tok) {
  float mx = logits[0];
  for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
  double sum = 0.0;
  for (int i = 0; i < V; i++) sum += exp((double)(logits[i] - mx));
  double lse = (double)mx + log(sum);
  if (n > V) n = V;
  if (n > LLB_SMP_MAX_LOGPROBS) n = LLB_SMP_MAX_LOGPROBS;
  LlbSmpCand c[LLB_SMP_MAX_LOGPROBS];
  int cnt = 0;
  for (int i = 0; n > 0 && i < V; i++) {  // insertion top-n, earlier id wins ties
    float l = logits[i];
    if (cnt == n && !(l > c[n - 1].l)) continue;
    int pos = cnt < n ? cnt++ : n - 1;
    while (pos > 0 && c[pos - 1].l < l) { c[pos] = c[pos - 1]; pos--; }
    c[pos].l = l; c[pos].id = i;
  }
  for (int i = 0; i < cnt; i++) { ids[i] = c[i].id; lp[i] = (float)((double)c[i].l - lse); }
  if (lp_tok && tok >= 0 && tok < V) *lp_tok = (float)((double)logits[tok] - lse);
  return lse;
}

// ---- stop strings ---------------------------------------------------------
struct LlbStop {
  int n; char** stop; int* slen;
  char* pend; int plen, pcap;             // bytes received, not yet emitted
  char* out; int ocap;
  int stopped;
};

LlbStop* llb_stop_new(const char* const* stops, int n) {
  LlbStop* s = (LlbStop*)calloc(1, sizeof *s);
  s->stop = (char**)calloc((size_t)n + 1, sizeof(char*));
  s->slen = (int*)calloc((size_t)n + 1, sizeof(int));
  for (int i = 0; i < n; i++) {
    size_t L = strlen(stops[i]);
    if (L == 0) continue;                 // an empty stop would match everywhere
    s->stop[s->n] = (char*)malloc(L + 1);
    memcpy(s->stop[s->n], stops[i], L + 1);
    s->slen[s->n++] = (int)L;
  }
  return s;
}

void llb_stop_free(LlbStop* s) {
  if (!s) return;
  for (int i = 0; i < s->n; i++) free(s->stop[i]);
  free(s->stop); free(s->slen); free(s->pend); free(s->out); free(s);
}

int llb_stop_stopped(const LlbStop* s) { return s->stopped; }

static const char* llb_stop_emit(LlbStop* s, int cnt, int* outlen) {
  if (cnt + 1 > s->ocap) { s->ocap = cnt + 64; s->out = (char*)realloc(s->out, (size_t)s->ocap); }
  memcpy(s->out, s->pend, (size_t)cnt);
  memmove(s->pend, s->pend + cnt, (size_t)(s->plen - cnt));
  s->plen -= cnt;
  *outlen = cnt;
  return s->out;
}

// bytes of an incomplete UTF-8 sequence at the end of b[0..len)
static int llb_stop_utf8_tail(const unsigned char* b, int len) {
  for (int k = 1; k <= 3 && k <= len; k++) {
    unsigned char c = b[len - k];
    if ((c & 0xC0) == 0x80) continue;     // continuation byte, keep looking back
    int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return need > k ? k : 0;
  }
  return 0;
}

const char* llb_stop_feed(LlbStop* s, const char* bytes, int len, int* outlen) {
  if (s->stopped) { *outlen = 0; return s->out ? s->out : ""; }
  if (s->plen + len > s->pcap) { s->pcap = s->plen + len + 64; s->pend = (char*)realloc(s->pend, (size_t)s->pcap); }
  memcpy(s->pend + s->plen, bytes, (size_t)len);
  s->plen += len;
  int best = -1;                          // earliest match start
  for (int k = 0; k < s->n; k++) {
    for (int i = 0; i + s->slen[k] <= s->plen; i++) {
      if (best >= 0 && i >= best) break;
      if (memcmp(s->pend + i, s->stop[k], (size_t)s->slen[k]) == 0) { best = i; break; }
    }
  }
  if (best >= 0) {
    s->stopped = 1;
#ifdef MUT_STOP_TRIM
    best++;                               // off by one: leaks the first stop byte
#endif
    const char* r = llb_stop_emit(s, best, outlen);
    s->plen = 0;
    return r;
  }
  int hold = llb_stop_utf8_tail((const unsigned char*)s->pend, s->plen);
  for (int k = 0; k < s->n; k++) {        // longest suffix that is a proper prefix of a stop
    int m = s->slen[k] - 1 < s->plen ? s->slen[k] - 1 : s->plen;
    for (; m > hold; m--) {
      if (memcmp(s->pend + s->plen - m, s->stop[k], (size_t)m) == 0) { hold = m; break; }
    }
  }
  return llb_stop_emit(s, s->plen - hold, outlen);
}

const char* llb_stop_flush(LlbStop* s, int* outlen) {
  if (s->stopped) { *outlen = 0; return s->out ? s->out : ""; }
  return llb_stop_emit(s, s->plen, outlen);
}

#endif
