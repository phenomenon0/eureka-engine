// spec_stat.c — W1-SAMPLER gate S: speculative decoding under sampling is LOSSLESS.
// A toy LM (V = 8, logits = a fixed pseudo-random function of (last two tokens, position))
// is decoded two ways from the same prompt:
//   direct: x_t = llb_smp_sample(p_t)                                 (the plain engine path)
//   spec  : the engine's cycle — a drafter proposes k tokens, llb_smp_verify serves them in
//           order until the first reject (its residual draw is emitted), then the bonus row
//           is sampled directly when all k were accepted (host/llb_host.c spec_pass shape)
// Both empirical joint distributions of the first L = 3 generated tokens (512 bins) are tested
// against the EXACT joint computed by enumeration from llb_smp_dist (the chain's own target),
// N draws each (seed = draw index). Thresholds FIXED here before the first run:
//   chi2 (bins with expected >= 5, the rest pooled) <= df + 5 * sqrt(2 df)   and
//   zero draws on an exact-zero-probability sequence; the spec run must see both accepts and
//   rejects (acceptance strictly inside (0.05, 0.95)) or the configuration is not exercising
//   the rejection branch.
// Plus G (greedy): temp = 0 spec tokens == direct tokens, byte-identical, 2000 toy models x 32
// tokens; and D (determinism): same seed -> same 32-token stream (direct and spec), and seed
// s vs s+1 differ in >= 90% of 2000 pairs.
// usage: spec_stat [N]   (default 400000)
#include "../../host/llb_sampler_impl.h"
#include <math.h>

#define V 8
#define L 3
#define PROMPT 2
#define MAXT 64

static uint64_t g_model = 1;              // toy model id
static float g_scale = 2.0f;

static uint64_t mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// the toy LM's next-token logits after ctx[0..n)
static void model(const int* ctx, int n, float* lg) {
  uint64_t h = mix(g_model * 1315423911ull + (uint64_t)ctx[n - 1] * 31 + (uint64_t)ctx[n - 2] * 977 + (uint64_t)n * 7919);
  for (int i = 0; i < V; i++) {
    h = mix(h + (uint64_t)i);
    double u1 = ((h >> 11) + 0.5) / 9007199254740992.0, u2 = ((mix(h) >> 11) + 0.5) / 9007199254740992.0;
    lg[i] = g_scale * (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
  }
}
// deterministic drafter: argmax of a perturbed copy of the model (right often, not always)
static int draft_det(const int* ctx, int n) {
  float lg[V];
  model(ctx, n, lg);
  uint64_t h = mix(0xD1Bull + (uint64_t)ctx[n - 1] * 13 + (uint64_t)n);
  int b = 0;
  for (int i = 0; i < V; i++) {
    lg[i] += (float)((mix(h + (uint64_t)i) >> 40) % 1000) / 400.0f;
    if (lg[i] > lg[b]) b = i;
  }
  return b;
}
// stochastic drafter: q = softmax(perturbed logits / 1.3), draw with its own counter RNG
static void draft_q(const int* ctx, int n, float* q) {
  float lg[V];
  model(ctx, n, lg);
  double s = 0;
  for (int i = 0; i < V; i++) {
    lg[i] = lg[i] / 1.3f + (float)((mix((uint64_t)n * 31 + (uint64_t)i + (uint64_t)ctx[n - 1] * 101) >> 40) % 1000) / 1000.0f;
    q[i] = expf(lg[i]);
    s += q[i];
  }
  for (int i = 0; i < V; i++) q[i] = (float)(q[i] / s);
}
static int draw_q(const float* q, uint64_t seed, uint64_t pos) {
  double u = (mix(seed * 0x51ED27ull + pos * 977 + 12345) >> 11) * (1.0 / 9007199254740992.0), c = 0;
  for (int i = 0; i < V; i++) { c += q[i]; if (u < c) return i; }
  return V - 1;
}

static const int prompt[PROMPT] = { 3, 5 };

static void direct(LlbSmp* s, int len, int* out) {
  int ctx[MAXT];
  memcpy(ctx, prompt, sizeof prompt);
  for (int t = 0; t < len; t++) {
    float lg[V];
    int n = PROMPT + t;
    model(ctx, n, lg);
    llb_smp_history(s, ctx, n);
    ctx[n] = out[t] = llb_smp_sample(s, lg, (uint64_t)n);
  }
}

// one spec decode: cycles of k drafts; mode 0 deterministic drafter, 1 stochastic (q)
static long acc_n, dft_n;
static void spec(LlbSmp* s, int len, int k, int mode, int* out) {
  int ctx[MAXT];
  memcpy(ctx, prompt, sizeof prompt);
  int n = PROMPT;
  uint64_t seed = llb_smp_params(s)->seed;
  while (n < PROMPT + len) {
    int d[8], dc[MAXT];
    float q[8][V];
    memcpy(dc, ctx, sizeof(int) * (size_t)n);
    for (int j = 0; j < k; j++) {          // the drafter chains on its own drafts
      if (mode == 0) d[j] = draft_det(dc, n + j);
      else { draft_q(dc, n + j, q[j]); d[j] = draw_q(q[j], seed, (uint64_t)(n + j)); }
      dc[n + j] = d[j];
    }
    int j = 0;
    for (; j < k && n < PROMPT + len; j++) {  // verify rows in order (spec_pass serving)
      float lg[V];
      int a;
      model(ctx, n, lg);
      llb_smp_history(s, ctx, n);
      int t = llb_smp_verify(s, lg, d[j], mode ? q[j] : NULL, (uint64_t)n, &a);
      dft_n += 1; acc_n += a;
      ctx[n] = out[n - PROMPT] = t;
      n += 1;
      if (!a) break;
    }
    if (j == k && n < PROMPT + len) {      // all accepted: the bonus row
      float lg[V];
      model(ctx, n, lg);
      llb_smp_history(s, ctx, n);
      ctx[n] = out[n - PROMPT] = llb_smp_sample(s, lg, (uint64_t)n);
      n += 1;
    }
  }
}

// exact joint over V^L from the chain's own target distribution
static void exact(LlbSmp* s, double* P) {
  int ids[V];
  float p[V];
  for (int c = 0; c < V * V * V; c++) {
    int x[L] = { c / (V * V), (c / V) % V, c % V }, ctx[PROMPT + L];
    memcpy(ctx, prompt, sizeof prompt);
    double pr = 1;
    for (int t = 0; t < L && pr > 0; t++) {
      float lg[V];
      int n = PROMPT + t, m;
      model(ctx, n, lg);
      llb_smp_history(s, ctx, n);
      m = llb_smp_dist(s, lg, ids, p);
      double pt = 0;
      for (int i = 0; i < m; i++) if (ids[i] == x[t]) pt = p[i];
      pr *= pt;
      ctx[n] = x[t];
    }
    P[c] = pr;
  }
}

static int chi(const char* tag, const double* P, const long* obs, long N, double* chi2o, int* dfo) {
  double chi2 = 0, pe = 0, po = 0;
  int df = -1;
  long zero_hits = 0;
  for (int c = 0; c < V * V * V; c++) {
    double e = P[c] * (double)N;
    if (P[c] == 0) { zero_hits += obs[c]; continue; }
    if (e >= 5) { chi2 += (obs[c] - e) * (obs[c] - e) / e; df += 1; }
    else { pe += e; po += (double)obs[c]; }
  }
  if (pe > 0) { chi2 += (po - pe) * (po - pe) / pe; df += 1; }
  double lim = df + 5 * sqrt(2.0 * df);
  int ok = chi2 <= lim && zero_hits == 0;
  printf("  %-26s chi2 %9.2f df %4d limit %8.2f zero-prob hits %ld  %s\n", tag, chi2, df, lim, zero_hits, ok ? "PASS" : "FAIL");
  *chi2o = chi2; *dfo = df;
  return ok;
}

int main(int argc, char** argv) {
  long N = argc > 1 ? atol(argv[1]) : 400000;
  int fails = 0;
  struct { const char* name; const char* spec; int k; int mode; } C[] = {
    { "T1 plain temp=1", "temp=1", 3, 0 },
    { "T2 chain k/p/min_p", "temp=0.7,top_k=5,top_p=0.9,min_p=0.05", 3, 0 },
    { "T3 chain+penalties", "temp=0.9,top_k=6,top_p=0.95,presence=0.8,frequency=0.3,repeat_penalty=1.2,repeat_last_n=4", 2, 0 },
    { "T4 stochastic drafter q", "temp=0.8,top_k=6", 3, 1 },
    { "T5 bias + k=1", "temp=1.1,bias=2:1.5;6:-inf", 1, 0 },
  };
  printf("gate S: spec-decode output distribution == direct sampling == exact target (V=%d, L=%d, N=%ld)\n", V, L, N);
  for (unsigned ci = 0; ci < sizeof C / sizeof C[0]; ci++) {
    LlbSmpParams P;
    char err[128];
    llb_smp_defaults(&P);
    if (llb_smp_parse(&P, C[ci].spec, err, sizeof err)) { printf("%s\n", err); return 2; }
    static double ex[V * V * V];
    static long od[V * V * V], os[V * V * V];
    memset(od, 0, sizeof od); memset(os, 0, sizeof os);
    LlbSmp* s = llb_smp_new(&P, V);
    exact(s, ex);
    acc_n = dft_n = 0;
    for (long i = 0; i < N; i++) {
      int a[L], b[L];
      LlbSmp* r;
      P.seed = (uint64_t)i * 2654435761u + 17;
      r = llb_smp_new(&P, V);
      direct(r, L, a);
      spec(r, L, C[ci].k, C[ci].mode, b);
      llb_smp_free(r);
      od[(a[0] * V + a[1]) * V + a[2]] += 1;
      os[(b[0] * V + b[1]) * V + b[2]] += 1;
    }
    double c2; int df;
    double ar = dft_n ? (double)acc_n / dft_n : 0;
    printf("%s  [%s] k=%d drafter=%s  acceptance %.3f (%ld/%ld)\n", C[ci].name, C[ci].spec, C[ci].k,
      C[ci].mode ? "stochastic-q" : "deterministic", ar, acc_n, dft_n);
    fails += !chi("direct vs exact", ex, od, N, &c2, &df);
    fails += !chi("spec   vs exact", ex, os, N, &c2, &df);
    int both = ar > 0.05 && ar < 0.95;
    printf("  rejection branch exercised: %s\n", both ? "PASS" : "FAIL");
    fails += !both;
    llb_smp_free(s);
  }
  // G: greedy spec == greedy direct, byte-identical, many toy models
  {
    LlbSmpParams P;
    llb_smp_defaults(&P);
    long bad = 0, acc = 0;
    acc_n = dft_n = 0;
    for (int m = 0; m < 2000; m++) {
      int a[32], b[32];
      g_model = 1000 + (uint64_t)m;
      LlbSmp* s = llb_smp_new(&P, V);
      direct(s, 32, a);
      spec(s, 32, 1 + m % 4, 0, b);
      bad += memcmp(a, b, sizeof a) != 0;
      llb_smp_free(s);
    }
    acc = acc_n;
    printf("G greedy: spec == direct byte-identical on %d/2000 toy models x 32 tokens (acceptance %.3f)  %s\n",
      2000 - (int)bad, dft_n ? (double)acc / dft_n : 0, bad ? "FAIL" : "PASS");
    fails += bad != 0;
    g_model = 1;
  }
  // D: determinism matrix
  {
    LlbSmpParams P;
    char err[64];
    llb_smp_defaults(&P);
    llb_smp_parse(&P, "temp=0.9,top_k=6,top_p=0.95", err, sizeof err);
    int same_d = 1, same_s = 1, differ = 0;
    for (int i = 0; i < 2000; i++) {
      int a[32], a2[32], b[32], b2[32], c[32];
      P.seed = (uint64_t)i;
      LlbSmp *s1 = llb_smp_new(&P, V), *s2 = llb_smp_new(&P, V);
      direct(s1, 32, a); direct(s2, 32, a2);
      spec(s1, 32, 3, 0, b); spec(s2, 32, 3, 0, b2);
      same_d &= !memcmp(a, a2, sizeof a);
      same_s &= !memcmp(b, b2, sizeof b);
      llb_smp_free(s1); llb_smp_free(s2);
      P.seed = (uint64_t)i + 1;
      s1 = llb_smp_new(&P, V);
      direct(s1, 32, c);
      differ += memcmp(a, c, sizeof a) != 0;
      llb_smp_free(s1);
    }
    printf("D determinism: same seed same stream direct %s, spec %s; seed s vs s+1 differ %d/2000  %s\n",
      same_d ? "yes" : "NO", same_s ? "yes" : "NO", differ, same_d && same_s && differ >= 1800 ? "PASS" : "FAIL");
    fails += !(same_d && same_s && differ >= 1800);
  }
  printf("gate S: %s (%d failing checks)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
