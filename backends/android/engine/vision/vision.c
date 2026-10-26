// vision.c: see vision.h. A port of the Mac reference tower (vit.c) that keeps the F16 mmproj mmap'd (no F32 widening:
// the phone has ~2.9 GB free) and runs every GEMM as F16-weight x F16-activation -> F32 accumulate (NEON, OpenMP).
// Graph (llama.cpp qwen3vl.cpp, no DeepStack): patch conv x2 (temporal taps) -> +bias -> +bilinear pos -> L x [LN ->
// QKV -> 2D vision rope -> bidirectional MHA -> out -> +res -> LN -> up -> GELU(tanh) -> down -> +res] -> post LN ->
// 2x2 merge -> mm.0 -> GELU(tanh) -> mm.2.
#define _GNU_SOURCE
#include "vision.h"
#include "vpre.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define ERR(...) (snprintf(err, 256, __VA_ARGS__), -1)

// ---------------------------------------------------------------- F16
static inline float h2f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
static inline uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; memcpy(&h, &x, 2); return h; }

// ---------------------------------------------------------------- mmproj (GGUF v3, mmap'd)
typedef struct { char name[96]; uint32_t ty; uint64_t n; const uint8_t *p; } Ten;  // ty 0 = F32, 1 = F16

static struct {
  uint8_t *map; size_t len;
  Ten *t; int nt;
  uint32_t E, F, L, H, PD, G, P, M;  // width, ffn, layers, heads, proj dim, pos grid side, patch, merge
  float eps, mean[3], std[3];
  int open;
} V;

typedef struct { const uint8_t *p, *end; int bad; } Rd;
static void rd(Rd *r, void *dst, size_t n) { if ((size_t)(r->end - r->p) < n) { r->bad = 1; memset(dst, 0, n); return; } memcpy(dst, r->p, n); r->p += n; }
static uint32_t rd32(Rd *r) { uint32_t v; rd(r, &v, 4); return v; }
static uint64_t rd64(Rd *r) { uint64_t v; rd(r, &v, 8); return v; }
static size_t rdstr(Rd *r, char *dst, size_t cap) {  // copies up to cap-1 bytes, skips the rest
  uint64_t n = rd64(r);
  if ((uint64_t)(r->end - r->p) < n) { r->bad = 1; return 0; }
  size_t c = n < cap - 1 ? (size_t)n : cap - 1;
  if (dst) memcpy(dst, r->p, c), dst[c] = 0;
  r->p += n;
  return c;
}
static const int KVSZ[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
// one value: numbers into *u / *f, strings into str, f32[3] arrays into a3, array of bools counts trues
static void rdval(Rd *r, uint32_t t, uint64_t *u, float *f, char *str, size_t cap, float *a3, uint32_t *ntrue) {
  if (t == 8) { rdstr(r, str, str ? cap : 1); return; }
  if (t == 9) {
    uint32_t et = rd32(r); uint64_t n = rd64(r);
    if (et == 9 || et > 12) { r->bad = 1; return; }
    for (uint64_t i = 0; i < n && !r->bad; i++) {
      if (et == 8) { rdstr(r, NULL, 1); continue; }
      uint64_t v = 0; rd(r, &v, (size_t)KVSZ[et]);
      if (et == 6 && a3 && i < 3) { uint32_t w = (uint32_t)v; memcpy(&a3[i], &w, 4); }
      if (et == 7 && ntrue && v) (*ntrue)++;
    }
    return;
  }
  if (t > 12) { r->bad = 1; return; }
  uint64_t v = 0; rd(r, &v, (size_t)KVSZ[t]);
  if (u) *u = v;
  if (f && t == 6) { uint32_t w = (uint32_t)v; memcpy(f, &w, 4); }
}

static const Ten *find(const char *name) {
  for (int i = 0; i < V.nt; i++) if (!strcmp(V.t[i].name, name)) return &V.t[i];
  return NULL;
}
// F32 view (an F16 tensor is widened into a malloc'd copy; used for the small tensors only)
static float *f32of(const char *name, size_t want, char *err) {
  const Ten *t = find(name);
  if (!t) return (snprintf(err, 256, "mmproj: no tensor %s", name), NULL);
  if (t->n != want) return (snprintf(err, 256, "mmproj: %s has %llu elements, want %zu", name, (unsigned long long)t->n, want), NULL);
  float *d = malloc(want * 4);
  if (!d) return (snprintf(err, 256, "out of memory"), NULL);
  if (t->ty == 0) memcpy(d, t->p, want * 4);
  else for (size_t i = 0; i < want; i++) d[i] = h2f(((const uint16_t *)t->p)[i]);
  return d;
}
static const uint16_t *w16(const char *name, size_t want, char *err) {  // GEMM weights stay F16 in the map
  const Ten *t = find(name);
  if (!t) return (snprintf(err, 256, "mmproj: no tensor %s", name), NULL);
  if (t->n != want || t->ty != 1) return (snprintf(err, 256, "mmproj: %s must be F16 with %zu elements", name, want), NULL);
  return (const uint16_t *)t->p;
}

typedef struct {
  float *ln1_w, *ln1_b, *qkv_b, *o_b, *ln2_w, *ln2_b, *up_b, *down_b;
  const uint16_t *qkv_w, *o_w, *up_w, *down_w;
} Lay;
static struct {
  const uint16_t *pe0, *pe1, *mm0_w, *mm2_w;
  float *pe_b, *pos, *post_w, *post_b, *mm0_b, *mm2_b;
  Lay *l;
} W;

int vis_open(const char *path, int proj_dim, char *err) {
  if (V.open) return 0;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return ERR("cannot open %s", path);
  struct stat st;
  if (fstat(fd, &st) || st.st_size < 64) { close(fd); return ERR("%s: unreadable", path); }
  V.len = (size_t)st.st_size;
  V.map = mmap(NULL, V.len, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (V.map == MAP_FAILED) { V.map = NULL; return ERR("%s: mmap failed", path); }
  Rd r = {V.map, V.map + V.len, 0};
  char magic[4]; rd(&r, magic, 4);
  uint32_t ver = rd32(&r);
  if (memcmp(magic, "GGUF", 4) || (ver != 2 && ver != 3)) return ERR("%s: not a GGUF", path);
  uint64_t nt = rd64(&r), nkv = rd64(&r), align = 32;
  if (nt > 4096 || nkv > 4096) return ERR("%s: corrupt header", path);
  char proj[64] = "", key[160];
  uint32_t E = 0, F = 0, L = 0, H = 0, PD = 0, IS = 0, P = 0, M = 2, nds = 0;
  float eps = 1e-6f, mean[3] = {.5f, .5f, .5f}, std[3] = {.5f, .5f, .5f};
  for (uint64_t i = 0; i < nkv && !r.bad; i++) {
    rdstr(&r, key, sizeof key);
    uint32_t t = rd32(&r); uint64_t u = 0;
    if (!strcmp(key, "clip.projector_type")) rdval(&r, t, NULL, NULL, proj, sizeof proj, NULL, NULL);
    else if (!strcmp(key, "general.alignment")) { rdval(&r, t, &u, NULL, NULL, 0, NULL, NULL); align = u; }
#define U(K, X) else if (!strcmp(key, K)) { rdval(&r, t, &u, NULL, NULL, 0, NULL, NULL); X = (uint32_t)u; }
    U("clip.vision.embedding_length", E) U("clip.vision.feed_forward_length", F) U("clip.vision.block_count", L)
    U("clip.vision.attention.head_count", H) U("clip.vision.projection_dim", PD) U("clip.vision.image_size", IS)
    U("clip.vision.patch_size", P) U("clip.vision.spatial_merge_size", M)
#undef U
    else if (!strcmp(key, "clip.vision.attention.layer_norm_epsilon")) rdval(&r, t, NULL, &eps, NULL, 0, NULL, NULL);
    else if (!strcmp(key, "clip.vision.image_mean")) rdval(&r, t, NULL, NULL, NULL, 0, mean, NULL);
    else if (!strcmp(key, "clip.vision.image_std")) rdval(&r, t, NULL, NULL, NULL, 0, std, NULL);
    else if (!strcmp(key, "clip.vision.is_deepstack_layers")) rdval(&r, t, NULL, NULL, NULL, 0, NULL, &nds);
    else rdval(&r, t, NULL, NULL, NULL, 0, NULL, NULL);
  }
  if (r.bad) return ERR("%s: corrupt metadata", path);
  if (strcmp(proj, "qwen3vl_merger")) return ERR("%s: projector '%s', this build runs qwen3vl_merger only", path, proj);
  if (nds) return ERR("%s: DeepStack layers are not supported", path);
  if (!E || !F || !L || !H || !PD || !IS || !P || E % H || (E / H) % 4 || M != 2) return ERR("%s: unsupported tower hparams", path);
  if ((int)PD != proj_dim) return ERR("%s: projects to %u dims, the language model is %d wide", path, PD, proj_dim);
  V.t = calloc(nt, sizeof(Ten));
  uint64_t *off = calloc(nt, 8);
  for (uint64_t i = 0; i < nt && !r.bad; i++) {
    Ten *t = &V.t[i];
    rdstr(&r, t->name, sizeof t->name);
    uint32_t nd = rd32(&r); uint64_t n = 1;
    if (nd < 1 || nd > 4) { r.bad = 1; break; }
    for (uint32_t d = 0; d < nd; d++) n *= rd64(&r);
    t->ty = rd32(&r); t->n = n; off[i] = rd64(&r);
    if (t->ty > 1) return ERR("%s: tensor %s is not F32/F16", path, t->name);
  }
  if (r.bad) return ERR("%s: corrupt tensor table", path);
  size_t data = ((size_t)(r.p - V.map) + align - 1) / align * align;
  for (uint64_t i = 0; i < nt; i++) {
    Ten *t = &V.t[i];
    size_t b = t->n * (t->ty ? 2 : 4);
    if (data + off[i] + b > V.len) return ERR("%s: truncated (tensor %s runs past the end; re-push the file)", path, t->name);
    t->p = V.map + data + off[i];
  }
  free(off);
  V.nt = (int)nt; V.E = E; V.F = F; V.L = L; V.H = H; V.PD = PD; V.P = P; V.M = M; V.G = IS / P; V.eps = eps;
  memcpy(V.mean, mean, 12); memcpy(V.std, std, 12);
  const size_t e = E, f = F, K = 3 * (size_t)P * P;
  W.pe0 = w16("v.patch_embd.weight", e * K, err); if (!W.pe0) return -1;
  W.pe1 = w16("v.patch_embd.weight.1", e * K, err); if (!W.pe1) return -1;
  if (!(W.pe_b = f32of("v.patch_embd.bias", e, err))) return -1;
  if (!(W.pos = f32of("v.position_embd.weight", e * V.G * V.G, err))) return -1;
  W.l = calloc(L, sizeof(Lay));
  char nm[96];
#define LF(f_, s, n) do { snprintf(nm, sizeof nm, "v.blk.%u.%s", i, s); if (!(l->f_ = f32of(nm, n, err))) return -1; } while (0)
#define LW(f_, s, n) do { snprintf(nm, sizeof nm, "v.blk.%u.%s", i, s); if (!(l->f_ = w16(nm, n, err))) return -1; } while (0)
  for (uint32_t i = 0; i < L; i++) {
    Lay *l = &W.l[i];
    LF(ln1_w, "ln1.weight", e); LF(ln1_b, "ln1.bias", e); LW(qkv_w, "attn_qkv.weight", 3 * e * e); LF(qkv_b, "attn_qkv.bias", 3 * e);
    LW(o_w, "attn_out.weight", e * e); LF(o_b, "attn_out.bias", e); LF(ln2_w, "ln2.weight", e); LF(ln2_b, "ln2.bias", e);
    LW(up_w, "ffn_up.weight", f * e); LF(up_b, "ffn_up.bias", f); LW(down_w, "ffn_down.weight", e * f); LF(down_b, "ffn_down.bias", e);
  }
#undef LF
#undef LW
  if (!(W.post_w = f32of("v.post_ln.weight", e, err)) || !(W.post_b = f32of("v.post_ln.bias", e, err))) return -1;
  if (!(W.mm0_w = w16("mm.0.weight", 16 * e * e, err)) || !(W.mm0_b = f32of("mm.0.bias", 4 * e, err))) return -1;
  if (!(W.mm2_w = w16("mm.2.weight", 4 * e * PD, err)) || !(W.mm2_b = f32of("mm.2.bias", PD, err))) return -1;
  V.open = 1;
  return 0;
}

// ---------------------------------------------------------------- GEMM: y[n][o] = x[n][k] . W[o][k] (+ b[o]), x and W F16
#ifdef __ARM_NEON
static inline float32x4_t ld4(const uint16_t *p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }
// 4 x rows against 4 W rows over k (k % 4 == 0): 16 vector accumulators, reduced at the end
static void tile44(const uint16_t *x, const uint16_t *w, int k, float *y, int ldy, const float *b) {
  float32x4_t a[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i][j] = vdupq_n_f32(0);
  for (int c = 0; c < k; c += 4) {
    float32x4_t xv[4], wv[4];
    for (int i = 0; i < 4; i++) xv[i] = ld4(x + (size_t)i * k + c);
    for (int j = 0; j < 4; j++) wv[j] = ld4(w + (size_t)j * k + c);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i][j] = vfmaq_f32(a[i][j], xv[i], wv[j]);
  }
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) y[(size_t)i * ldy + j] = vaddvq_f32(a[i][j]) + (b ? b[j] : 0.f);
}
static void tile14(const uint16_t *x, const uint16_t *w, int k, float *y, const float *b) {
  float32x4_t a[4];
  for (int j = 0; j < 4; j++) a[j] = vdupq_n_f32(0);
  for (int c = 0; c < k; c += 4) {
    float32x4_t xv = ld4(x + c);
    for (int j = 0; j < 4; j++) a[j] = vfmaq_f32(a[j], xv, ld4(w + (size_t)j * k + c));
  }
  for (int j = 0; j < 4; j++) y[j] = vaddvq_f32(a[j]) + (b ? b[j] : 0.f);
}
#else
static void tile44(const uint16_t *x, const uint16_t *w, int k, float *y, int ldy, const float *b) {
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
    float s = 0;
    for (int c = 0; c < k; c++) s += h2f(x[(size_t)i * k + c]) * h2f(w[(size_t)j * k + c]);
    y[(size_t)i * ldy + j] = s + (b ? b[j] : 0.f);
  }
}
static void tile14(const uint16_t *x, const uint16_t *w, int k, float *y, const float *b) {
  for (int j = 0; j < 4; j++) {
    float s = 0;
    for (int c = 0; c < k; c++) s += h2f(x[c]) * h2f(w[(size_t)j * k + c]);
    y[j] = s + (b ? b[j] : 0.f);
  }
}
#endif

#define XT 16  // x rows per job
#define WT 64  // W rows per job
static void gemm(const uint16_t *x, int n, int k, const uint16_t *w, int o, const float *b, float *y) {
  const int nxt = (n + XT - 1) / XT, nwt = (o + WT - 1) / WT;
#pragma omp parallel for schedule(dynamic, 1)
  for (int job = 0; job < nxt * nwt; job++) {
    const int xt = job % nxt, wt = job / nxt;
    const int i0 = xt * XT, i1 = i0 + XT < n ? i0 + XT : n, j0 = wt * WT, j1 = j0 + WT < o ? j0 + WT : o;
    for (int j = j0; j < j1; j += 4) {
      int i = i0;
      for (; i + 4 <= i1; i += 4) tile44(x + (size_t)i * k, w + (size_t)j * k, k, y + (size_t)i * o + j, o, b ? b + j : NULL);
      for (; i < i1; i++) tile14(x + (size_t)i * k, w + (size_t)j * k, k, y + (size_t)i * o + j, b ? b + j : NULL);
    }
  }
}

// ---------------------------------------------------------------- elementwise ops
// LN (ggml_norm: mean/var in double) * w + b -> F16 (the only consumer is a GEMM, which rounds its input to F16)
static void layernorm16(const float *x, int n, int e, const float *w, const float *b, uint16_t *y) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < n; r++) {
    const float *xr = x + (size_t)r * e;
    double s = 0;
    for (int i = 0; i < e; i++) s += xr[i];
    const float mean = (float)(s / e);
    double v = 0;
    for (int i = 0; i < e; i++) { const float d = xr[i] - mean; v += (double)d * d; }
    const float scale = 1.0f / sqrtf((float)(v / e) + V.eps);
    for (int i = 0; i < e; i++) y[(size_t)r * e + i] = f2h(((xr[i] - mean) * scale) * w[i] + b[i]);
  }
}
static void gelu16(const float *x, size_t n, uint16_t *y) {  // tanh form; n elements
#pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)n; i++) {
    const float v = x[i];
    y[i] = f2h(0.5f * v * (1.0f + tanhf(0.79788456080286535588f * v * (1.0f + 0.044715f * v * v))));
  }
}
static void addto(float *y, const float *a, size_t n) {
#pragma omp parallel for schedule(static)
  for (long i = 0; i < (long)n; i++) y[i] += a[i];
}

// GGML_ROPE_TYPE_VISION, n_dims = D/2, sections {D/4 x4}: pair (i, i+D/2); i < D/4 -> row pos, else column pos
static void rope_tables(int D, const int *py, const int *px, int n, float *cs, float *sn) {
  const int nd = D / 2, s0 = D / 4;
  const float inv = -1.f / nd;
#pragma omp parallel for schedule(static)
  for (int r = 0; r < n; r++)
    for (int ic = 0; ic < nd; ic++) {
      const float p = ic < s0 ? (float)ic : (float)(ic - s0);
      const float base = ic < s0 ? (float)py[r] : (float)px[r];
      const float th = base * powf(10000.0f, 2.0f * inv * p);
      cs[(size_t)r * nd + ic] = cosf(th), sn[(size_t)r * nd + ic] = sinf(th);
    }
}
// rotate Q (off 0) and K (off E) heads of qkv rows [n][3E]; then write K,V rounded to F16 into kv[n][2E] (as float)
static void rope_kv(float *qkv, int n, int E, int H, int D, const float *cs, const float *sn, float *kv) {
  const int nd = D / 2;
#pragma omp parallel for schedule(static)
  for (int r = 0; r < n; r++) {
    for (int part = 0; part < 2; part++)
      for (int h = 0; h < H; h++) {
        float *v = qkv + (size_t)r * 3 * E + (size_t)part * E + (size_t)h * D;
        for (int ic = 0; ic < nd; ic++) {
          const float c = cs[(size_t)r * nd + ic], s = sn[(size_t)r * nd + ic], x0 = v[ic], x1 = v[ic + nd];
          v[ic] = x0 * c - x1 * s;
          v[ic + nd] = x0 * s + x1 * c;
        }
      }
    for (int i = 0; i < 2 * E; i++) kv[(size_t)r * 2 * E + i] = h2f(f2h(qkv[(size_t)r * 3 * E + E + i]));
  }
}

static inline float dotf(const float *a, const float *b, int n) {
#ifdef __ARM_NEON
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  for (int i = 0; i < n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#else
  float s = 0;
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  return s;
#endif
}
// bidirectional MHA: out16[n][E] = softmax(scale * Q K^T) V, per head; K/V from kv (F16-rounded floats), Q from qkv
static void attention16(const float *qkv, const float *kv, int n, int E, int H, int D, uint16_t *out16) {
  const float scale = 1.0f / sqrtf((float)D);
#pragma omp parallel
  {
    float *S = malloc(sizeof(float) * (size_t)n), *o = malloc(sizeof(float) * (size_t)D);
#pragma omp for schedule(dynamic, 4) collapse(2)
    for (int h = 0; h < H; h++)
      for (int i = 0; i < n; i++) {
        const float *q = qkv + (size_t)i * 3 * E + (size_t)h * D;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
          const float s = dotf(q, kv + (size_t)j * 2 * E + (size_t)h * D, D) * scale;
          S[j] = s;
          if (s > mx) mx = s;
        }
        double sum = 0;
        for (int j = 0; j < n; j++) { S[j] = expf(S[j] - mx); sum += S[j]; }
        const float inv = (float)(1.0 / sum);
        for (int d = 0; d < D; d++) o[d] = 0;
        for (int j = 0; j < n; j++) {
          const float p = S[j] * inv;
          const float *v = kv + (size_t)j * 2 * E + E + (size_t)h * D;
          for (int d = 0; d < D; d++) o[d] += p * v[d];
        }
        for (int d = 0; d < D; d++) out16[(size_t)i * E + (size_t)h * D + d] = f2h(o[d]);
      }
    free(S); free(o);
  }
}

// ggml_interpolate(BILINEAR | ALIGN_CORNERS) of the G x G table to px x py (Metal kernel_upscale_bilinear_f32, no antialias)
static void pos_resize(const float *tab, int G, int E, int px, int py, float *out) {
  if (px == G && py == G) { memcpy(out, tab, sizeof(float) * (size_t)G * G * E); return; }
  const float sf0 = px > 1 && G > 1 ? (float)(px - 1) / (G - 1) : (float)px / G;
  const float sf1 = py > 1 && G > 1 ? (float)(py - 1) / (G - 1) : (float)py / G;
  for (int y = 0; y < py; y++) {
    const float f01 = (float)y / sf1;
    long i01 = (long)floorf(f01);
    if (i01 < 0) i01 = 0;
    if (i01 > G - 1) i01 = G - 1;
    const long i01p = i01 + 1 > G - 1 ? G - 1 : i01 + 1;
    float fd1 = f01 - (float)i01;
    fd1 = fd1 < 0 ? 0 : fd1 > 1 ? 1 : fd1;
    for (int x = 0; x < px; x++) {
      const float f00 = (float)x / sf0;
      long i00 = (long)floorf(f00);
      if (i00 < 0) i00 = 0;
      if (i00 > G - 1) i00 = G - 1;
      const long i00p = i00 + 1 > G - 1 ? G - 1 : i00 + 1;
      float fd0 = f00 - (float)i00;
      fd0 = fd0 < 0 ? 0 : fd0 > 1 ? 1 : fd0;
      const float *s00 = tab + ((size_t)i01 * G + i00) * E, *s10 = tab + ((size_t)i01 * G + i00p) * E;
      const float *s01 = tab + ((size_t)i01p * G + i00) * E, *s11 = tab + ((size_t)i01p * G + i00p) * E;
      float *o = out + ((size_t)y * px + x) * E;
      for (int e = 0; e < E; e++)
        o[e] = s00[e] * (1.0f - fd0) * (1.0f - fd1) + s10[e] * fd0 * (1.0f - fd1) + s01[e] * (1.0f - fd0) * fd1 + s11[e] * fd0 * fd1;
    }
  }
}

// ---------------------------------------------------------------- the tower
int vis_tower(const float *planar, int nx, int ny, float *embd, char *err) {
  if (!V.open) return ERR("vision: mmproj not open");
  const int E = V.E, H = V.H, D = E / V.H, F = V.F, P = V.P, M = V.M, PD = V.PD;
  if (nx % (P * M) || ny % (P * M)) return ERR("image %dx%d is not a multiple of %d", nx, ny, P * M);
  const int px = nx / P, py = ny / P, N = px * py, K = 3 * P * P;
  float *x = malloc(sizeof(float) * (size_t)N * E), *t1 = malloc(sizeof(float) * (size_t)N * E);
  float *qkv = malloc(sizeof(float) * (size_t)N * 3 * E), *up = malloc(sizeof(float) * (size_t)N * (F > 4 * E ? F : 4 * E));
  float *kv = malloc(sizeof(float) * (size_t)N * 2 * E), *cs = malloc(sizeof(float) * (size_t)N * (D / 2)), *sn = malloc(sizeof(float) * (size_t)N * (D / 2));
  uint16_t *a16 = malloc(2 * (size_t)N * (F > 4 * E ? F : 4 * E)), *b16 = malloc(2 * (size_t)N * E);
  int *pyi = malloc(sizeof(int) * N), *pxi = malloc(sizeof(int) * N);
  int rc = 0;
  if (!x || !t1 || !qkv || !up || !kv || !cs || !sn || !a16 || !b16 || !pyi || !pxi) { rc = ERR("out of memory in the vision tower"); goto done; }

  {  // patch embed: im2col (2x2-merge row order, F16) -> conv(W0) + conv(W1) -> + bias
    int r = 0;
    for (int y = 0; y < py; y += M)
      for (int xx = 0; xx < px; xx += M)
        for (int dy = 0; dy < M; dy++)
          for (int dx = 0; dx < M; dx++, r++) {
            uint16_t *row = a16 + (size_t)r * K;
            const int y0 = (y + dy) * P, x0 = (xx + dx) * P;
            pyi[r] = y + dy, pxi[r] = xx + dx;
            for (int ch = 0; ch < 3; ch++)
              for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++)
                  row[((size_t)ch * P + ky) * P + kx] = f2h(planar[(size_t)ch * nx * ny + (size_t)(y0 + ky) * nx + x0 + kx]);
          }
    gemm(a16, N, K, W.pe0, E, NULL, x);
    gemm(a16, N, K, W.pe1, E, NULL, t1);
    addto(x, t1, (size_t)N * E);
#pragma omp parallel for
    for (int r2 = 0; r2 < N; r2++) for (int i = 0; i < E; i++) x[(size_t)r2 * E + i] += W.pe_b[i];
    float *pr = malloc(sizeof(float) * (size_t)N * E);
    if (!pr) { rc = ERR("out of memory in the vision tower"); goto done; }
    pos_resize(W.pos, V.G, E, px, py, pr);
#pragma omp parallel for
    for (int r2 = 0; r2 < N; r2++) {
      const float *s = pr + ((size_t)pyi[r2] * px + pxi[r2]) * E;
      for (int i = 0; i < E; i++) x[(size_t)r2 * E + i] += s[i];
    }
    free(pr);
  }
  rope_tables(D, pyi, pxi, N, cs, sn);

  for (uint32_t il = 0; il < V.L; il++) {
    const Lay *l = &W.l[il];
    layernorm16(x, N, E, l->ln1_w, l->ln1_b, b16);
    gemm(b16, N, E, l->qkv_w, 3 * E, l->qkv_b, qkv);
    rope_kv(qkv, N, E, H, D, cs, sn, kv);
    attention16(qkv, kv, N, E, H, D, b16);
    gemm(b16, N, E, l->o_w, E, l->o_b, t1);
    addto(x, t1, (size_t)N * E);
    layernorm16(x, N, E, l->ln2_w, l->ln2_b, b16);
    gemm(b16, N, E, l->up_w, F, l->up_b, up);
    gelu16(up, (size_t)N * F, a16);
    gemm(a16, N, F, l->down_w, E, l->down_b, t1);
    addto(x, t1, (size_t)N * E);
  }
  layernorm16(x, N, E, W.post_w, W.post_b, b16);
  {  // merger: 4 consecutive (merge-ordered) patches are one 4E row
    const int T = N / 4;
    gemm(b16, T, 4 * E, W.mm0_w, 4 * E, W.mm0_b, up);
    gelu16(up, (size_t)T * 4 * E, a16);
    gemm(a16, T, 4 * E, W.mm2_w, PD, W.mm2_b, embd);
  }
done:
  free(x), free(t1), free(qkv), free(up), free(kv), free(cs), free(sn), free(a16), free(b16), free(pyi), free(pxi);
  return rc;
}

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int vis_encode(const char *img, int max_tokens, float **embd, int *ntok, int *gx, int *gy, double *ms, char *err) {
  if (!V.open) return ERR("vision: mmproj not open");
  VPreCfg c;
  vpre_cfg_default(&c);
  c.patch = V.P, c.merge = V.M;
  for (int i = 0; i < 3; i++) c.mean[i] = V.mean[i], c.std[i] = V.std[i];
  if (max_tokens > 0) c.max_px = max_tokens * (V.P * V.M) * (V.P * V.M);
  float *planar = NULL;
  int nx = 0, ny = 0;
  if (vpre_image(&c, img, &planar, &nx, &ny, err)) return -1;
  const int n = (nx / (V.P * V.M)) * (ny / (V.P * V.M));
  *embd = malloc(sizeof(float) * (size_t)n * V.PD);
  if (!*embd) { free(planar); return ERR("out of memory"); }
  double t0 = now_ms();
  int rc = vis_tower(planar, nx, ny, *embd, err);
  if (ms) *ms = now_ms() - t0;
  free(planar);
  if (rc) { free(*embd); *embd = NULL; return -1; }
  *ntok = n; *gx = nx / (V.P * V.M); *gy = ny / (V.P * V.M);
  return 0;
}
