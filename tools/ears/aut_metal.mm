// aut_metal.mm: IOS-EARS "metaud", the Qwen3 AuT audio tower (mtmd PROJECTOR_TYPE_QWEN3A, bf16 mmproj) on Metal.
// A port of EARS-G1 aut.cu (cudaud) op by op, same aut.h API, same graph (llama.cpp 2ca15f5 qwen3a.cpp + build_vit):
//   mel window [128][n_len] -> 100-frame chunks -> 3 x [conv2d 3x3 s2 p1 (im2col, F16) + b -> GELU-erf]
//   -> [16 freq x 480 ch] -> conv_out (F16) -> + pos rows 0..12 per chunk
//   -> L x [LN -> Q,K,V (+b) -> bidirectional MHA D=64 -> out+b -> +res -> LN -> up+b -> GELU-erf -> down+b -> +res]
//   -> post LN -> mm.a.mlp.1+b -> GELU-erf -> mm.a.mlp.2+b
// Staging = the CPU golden's (and the ears-fix k_attn32's): conv / conv_out operands F16; block + projector GEMMs
// BF16 x BF16; all F32 accumulate; attention Q F32, K/V F16, P F32, F32 softmax + accumulate.
// Metal specifics: one command buffer per window; all buffers shared (unified memory); MSL compiled at load with
// fast math OFF; erf is a positive-term series (no Metal erf), ~1-2 ulp.
#include "aut.h"
#include "gguf.h"
#include "ggml.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

enum { EP_BIAS = 1, EP_GELU = 2, EP_RES = 4 };
enum { T_LN1W, T_LN1B, T_QW, T_QB, T_KW, T_KB, T_VW, T_VB, T_OW, T_OB, T_LN2W, T_LN2B, T_UPW, T_UPB, T_DNW, T_DNB, T_NL };
static const char *const LNAME[T_NL] = {"ln1.weight", "ln1.bias", "attn_q.weight", "attn_q.bias", "attn_k.weight",
  "attn_k.bias", "attn_v.weight", "attn_v.bias", "attn_out.weight", "attn_out.bias", "ln2.weight", "ln2.bias",
  "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};
enum { G_C1W, G_C1B, G_C2W, G_C2B, G_C3W, G_C3B, G_COW, G_POS, G_POSTW, G_POSTB, G_M1W, G_M1B, G_M2W, G_M2B, G_N };
enum { B_MEL, B_C1, B_C2, B_C3, B_X, B_CUR, B_TMP, B_Q, B_K, B_V, B_AO, B_UP, B_EMBD, B_ROWS, B_B16, B_IN, B_N };
enum { K_CHUNKS, K_IM2COL, K_GATHER, K_ADDPOS, K_EPI, K_LN, K_ATTN, K_GEMM16, K_GEMMBF, K_N };
static const char *const KNAME[K_N] = {"k_chunks", "k_im2col3", "k_gather", "k_addpos", "k_epi", "k_ln", "k_attn",
                                       "k_gemm_f16", "k_gemm_bf16"};

static const char *MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define SQRT_2_INV 0.70710678118654752440084436210484f

// erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (1*3*...*(2n+1)): all terms positive, no cancellation
static inline float erf_ser(float x) {
  const float a = fabs(x);
  if (a >= 3.9f) return copysign(1.0f, x);
  const float x2 = a * a;
  float term = a, sum = a;
  for (int n = 1; n < 96; n++) {
    term = term * (2.0f * x2) / (float)(2 * n + 1);
    sum = sum + term;
    if (term < sum * 1e-9f) break;
  }
  const float r = 1.1283791670955126f * precise::exp(-x2) * sum;
  return copysign(min(r, 1.0f), x);
}
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erf_ser(x * SQRT_2_INV)); }
static inline ushort f2bf(float f) {  // round to nearest even
  const uint u = as_type<uint>(f);
  return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float bf2f(ushort b) { return as_type<float>((uint)b << 16); }

kernel void k_chunks(device const float *mel [[buffer(0)]], device float *dst [[buffer(1)]],
                     constant int &n_mel [[buffer(2)]], constant int &n_len [[buffer(3)]], uint t [[thread_position_in_grid]]) {
  if ((long)t >= (long)n_mel * n_len) return;
  const int w = t % 100, h = (t / 100) % n_mel, n = t / (100 * n_mel);
  dst[t] = mel[(long)h * n_len + n * 100 + w];
}
struct ConvP { int N, IH, IW, IC, OH, OW; };
kernel void k_im2col3(device const float *src [[buffer(0)]], device half *rows [[buffer(1)]], constant ConvP &p [[buffer(2)]],
                      uint t [[thread_position_in_grid]]) {
  const long K = (long)p.IC * 9;
  if ((long)t >= (long)p.N * p.OH * p.OW * K) return;
  const long r = t / K;
  const int k = (int)(t % K), ic = k / 9, kh = (k / 3) % 3, kw = k % 3;
  const int ow = (int)(r % p.OW), oh = (int)((r / p.OW) % p.OH), n = (int)(r / ((long)p.OW * p.OH));
  const int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
  float v = 0.0f;
  if (ih >= 0 && ih < p.IH && iw >= 0 && iw < p.IW) v = src[(((long)n * p.IH + ih) * p.IW + iw) * p.IC + ic];
  rows[t] = (half)v;
}
kernel void k_gather(device const float *src [[buffer(0)]], device half *rows [[buffer(1)]], constant ConvP &p [[buffer(2)]],
                     uint t [[thread_position_in_grid]]) {  // NHWC [N][OH=16][OW=13][C=IC] -> rows [N*OW][C*OH]
  const int OH = p.OH, OW = p.OW, C = p.IC;
  if ((long)t >= (long)p.N * OW * C * OH) return;
  const int f = t % OH, c = (t / OH) % C;
  const long tok = t / ((long)OH * C);
  const int ow = (int)(tok % OW), n = (int)(tok / OW);
  rows[t] = (half)src[(((long)n * OH + f) * OW + ow) * C + c];
}
kernel void k_addpos(device float *x [[buffer(0)]], device const float *pos [[buffer(1)]], constant int &n [[buffer(2)]],
                     constant int &E [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if ((int)i >= n) return;
  const int t = i / E;
  x[i] = x[i] + pos[(t % 13) * E + i % E];
}
struct EpiP { int n, o, flags, hb; };
kernel void k_epi(device const float *acc [[buffer(0)]], device float *dst [[buffer(1)]], device const float *bias [[buffer(2)]],
                  device ushort *hb [[buffer(3)]], constant EpiP &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if ((int)i >= p.n) return;
  float v = acc[i];
  if (p.flags & 1) v = v + bias[i % p.o];
  if (p.flags & 2) v = gelu_erf(v);
  if (p.flags & 4) v = v + dst[i];
  dst[i] = v;
  if (p.hb) hb[i] = f2bf(v);
}
// LayerNorm, one row per 256-thread group (two-pass mean / var as aut.cu k_ln)
kernel void k_ln(device const float *src [[buffer(0)]], device const float *w [[buffer(1)]], device const float *b [[buffer(2)]],
                 device float *dst [[buffer(3)]], device ushort *h [[buffer(4)]], constant int &E [[buffer(5)]],
                 constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const float *xr = src + (long)row * E;
  float s = 0.0f;
  for (int i = tid; i < E; i += 256) s += xr[i];
  s = simd_sum(s);
  if (ln == 0) red[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = ln < 8 ? red[ln] : 0.0f;
  s = simd_sum(s);
  const float mean = s / E;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float v = 0.0f;
  for (int i = tid; i < E; i += 256) { const float d = xr[i] - mean; v += d * d; }
  v = simd_sum(v);
  if (ln == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = ln < 8 ? red[ln] : 0.0f;
  v = simd_sum(v);
  const float scale = 1.0f / precise::sqrt(v / E + eps);
  for (int i = tid; i < E; i += 256) {
    const float y = ((xr[i] - mean) * scale) * w[i] + b[i];
    dst[(long)row * E + i] = y;
    h[(long)row * E + i] = f2bf(y);
  }
}
// attention, one 128-thread group per (query, head): Q F32, K/V F16, scores / softmax / P / accumulate F32.
// Softmax as ggml soft_max: p = exp(s - max), then scaled by 1/sum before P.V. n <= 128 (one 800-frame window = 104).
struct AttP { int n, E, D; float scale; };
kernel void k_attn(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                   device float *dst [[buffer(3)]], device ushort *dh [[buffer(4)]], constant AttP &p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  threadgroup float qs[64], ps[128], red[4];
  const int qi = tg.x, hh = tg.y, n = p.n, E = p.E;
  if (tid < 64) qs[tid] = q[(long)qi * E + hh * 64 + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = -INFINITY;
  if ((int)tid < n) {
    device const float *kr = k + (long)tid * E + hh * 64;
    float a = 0.0f;
    for (int d = 0; d < 64; d++) a += qs[d] * (float)(half)kr[d];
    s = a * p.scale;
  }
  float m = simd_max(s);
  if (ln == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(red[0], red[1]), max(red[2], red[3]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float e = (int)tid < n ? precise::exp(s - m) : 0.0f;
  float sum = simd_sum(e);
  if (ln == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = (red[0] + red[1]) + (red[2] + red[3]);
  ps[tid] = e * (1.0f / sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 64) {
    float o = 0.0f;
    for (int j = 0; j < n; j++) o += ps[j] * (float)(half)v[(long)j * E + hh * 64 + tid];
    dst[(long)qi * E + hh * 64 + tid] = o;
    dh[(long)qi * E + hh * 64 + tid] = f2bf(o);
  }
}
// Y[n][o] = X[n][k] . W[o][k]^T, operands converted exactly to F32, F32 simdgroup MMA accumulate.
// 128 threads = 4 simdgroups, tile 32 (n) x 32 (o) x 32 (k); each simdgroup a 16 x 16 block (2 x 2 frags).
struct GemmP { int n, k, o; };
template <typename T> static inline float cvt(T x);
template <> inline float cvt<half>(half x) { return (float)x; }
template <> inline float cvt<ushort>(ushort x) { return bf2f(x); }
#if defined(GEMM_SCALAR) || defined(GEMM_KAHAN)  // diagnostic variants: one thread per output, sequential k (plain / compensated)
template <typename T>
static void gemm_body(device const T *X, device const T *W, device float *Y, constant GemmP &p, uint2 tg, uint tid, uint sg,
                      threadgroup float *As, threadgroup float *Bs) {
  const int n0 = tg.y * 32, o0 = tg.x * 32;
  for (int e = tid; e < 32 * 32; e += 128) {
    const int r = n0 + e / 32, c = o0 + e % 32;
    if (r >= p.n || c >= p.o) continue;
    device const T *xr = X + (long)r * p.k; device const T *wr = W + (long)c * p.k;
    float s = 0.0f, comp = 0.0f;
    for (int k = 0; k < p.k; k++) {
      const float t = cvt(xr[k]) * cvt(wr[k]);  // exact: bf16*bf16 and f16*f16 products fit in f32
#ifdef GEMM_KAHAN
      const float y = t - comp, u = s + y; comp = (u - s) - y; s = u;
#else
      s = s + t;
#endif
    }
    Y[(long)r * p.o + c] = s;
  }
}
#else
template <typename T>
static void gemm_body(device const T *X, device const T *W, device float *Y, constant GemmP &p, uint2 tg, uint tid, uint sg,
                      threadgroup float *As, threadgroup float *Bs) {
  const int n0 = tg.y * 32, o0 = tg.x * 32;
  const int r0 = (sg / 2) * 16, c0 = (sg % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int kb = 0; kb < p.k; kb += 32) {
    for (int e = tid; e < 32 * 32; e += 128) {
      const int r = e / 32, kk = e % 32, kg = kb + kk;
      As[e] = (n0 + r < p.n && kg < p.k) ? cvt(X[(long)(n0 + r) * p.k + kg]) : 0.0f;
      Bs[e] = (o0 + r < p.o && kg < p.k) ? cvt(W[(long)(o0 + r) * p.k + kg]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 32; kk += 8) {
      simdgroup_float8x8 a[2], b[2];
      for (int i = 0; i < 2; i++) simdgroup_load(a[i], As + (r0 + 8 * i) * 32 + kk, 32);
      for (int j = 0; j < 2; j++) simdgroup_load(b[j], Bs + (c0 + 8 * j) * 32 + kk, 32, ulong2(0, 0), true);
      for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) simdgroup_store(acc[i][j], As + (r0 + 8 * i) * 32 + c0 + 8 * j, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = tid; e < 32 * 32; e += 128) {
    const int r = e / 32, c = e % 32;
    if (n0 + r < p.n && o0 + c < p.o) Y[(long)(n0 + r) * p.o + o0 + c] = As[e];
  }
}
#endif
kernel void k_gemm_f16(device const half *X [[buffer(0)]], device const half *W [[buffer(1)]], device float *Y [[buffer(2)]],
                       constant GemmP &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float As[32 * 32], Bs[32 * 32];
  gemm_body<half>(X, W, Y, p, tg, tid, sg, As, Bs);
}
kernel void k_gemm_bf16(device const ushort *X [[buffer(0)]], device const ushort *W [[buffer(1)]], device float *Y [[buffer(2)]],
                        constant GemmP &p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float As[32 * 32], Bs[32 * 32];
  gemm_body<ushort>(X, W, Y, p, tg, tid, sg, As, Bs);
}
)MSL";

struct ConvP { int N, IH, IW, IC, OH, OW; };
struct EpiP { int n, o, flags, hb; };
struct AttP { int n, E, D; float scale; };
struct GemmP { int n, k, o; };

struct AutW {
  int E, H, D, F, L, PD, C;
  float eps;
  size_t wbytes, og[G_N], *ol;
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLBuffer> wb, b[B_N];
  id<MTLComputePipelineState> ps[K_N];
  size_t cap;
  double gpu_ms;
};

extern "C" void aut_free(AutW *w) {
  if (!w) return;
  @autoreleasepool {
    w->wb = nil, w->q = nil, w->dev = nil;
    for (int i = 0; i < B_N; i++) w->b[i] = nil;
    for (int i = 0; i < K_N; i++) w->ps[i] = nil;
  }
  free(w->ol);
  delete w;
}
extern "C" int aut_proj_dim(const AutW *w) { return w->PD; }
extern "C" int aut_n_tokens(int n_len) { return n_len / 100 * 13; }
extern "C" double aut_gpu_ms(const AutW *w) { return w->gpu_ms; }
extern "C" size_t aut_weight_bytes(const AutW *w) { return w->wbytes; }

static size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

extern "C" AutW *aut_load(const char *path, char *err) {
  @autoreleasepool {
    ggml_context *meta = NULL;
    gguf_init_params gp = {false, &meta};
    gguf_context *g = gguf_init_from_file(path, gp);
    if (!g) { snprintf(err, 256, "aut: cannot read %s", path); return NULL; }
    AutW *c = new AutW();
    auto kv_u32 = [&](const char *k) -> int { int64_t i = gguf_find_key(g, k); return i < 0 ? -1 : (int)gguf_get_val_u32(g, i); };
    const int64_t ip = gguf_find_key(g, "clip.audio.projector_type");
    if (ip < 0 || strcmp(gguf_get_val_str(g, ip), "qwen3a") != 0) {
      snprintf(err, 256, "aut: clip.audio.projector_type is not qwen3a");
      gguf_free(g), ggml_free(meta), aut_free(c);
      return NULL;
    }
    c->E = kv_u32("clip.audio.embedding_length"), c->F = kv_u32("clip.audio.feed_forward_length");
    c->L = kv_u32("clip.audio.block_count"), c->H = kv_u32("clip.audio.attention.head_count");
    c->PD = kv_u32("clip.audio.projection_dim");
    { int64_t ie = gguf_find_key(g, "clip.audio.attention.layer_norm_epsilon"); c->eps = ie < 0 ? 1e-5f : gguf_get_val_f32(g, ie); }
    c->D = c->H > 0 ? c->E / c->H : 0, c->C = 480;
    if (c->D != 64 || c->H * c->D != c->E) {
      snprintf(err, 256, "aut: head dim %d (E %d / H %d): k_attn needs 64", c->D, c->E, c->H);
      gguf_free(g), ggml_free(meta), aut_free(c);
      return NULL;
    }
    const size_t E = c->E, F = c->F, C = c->C;
    struct Want { std::string name; size_t *slot; ggml_type ty; size_t n; };
    std::vector<Want> ws = {
      {"a.conv2d.1.weight", &c->og[G_C1W], GGML_TYPE_F16, C * 9}, {"a.conv2d.1.bias", &c->og[G_C1B], GGML_TYPE_F32, C},
      {"a.conv2d.2.weight", &c->og[G_C2W], GGML_TYPE_F16, C * C * 9}, {"a.conv2d.2.bias", &c->og[G_C2B], GGML_TYPE_F32, C},
      {"a.conv2d.3.weight", &c->og[G_C3W], GGML_TYPE_F16, C * C * 9}, {"a.conv2d.3.bias", &c->og[G_C3B], GGML_TYPE_F32, C},
      {"a.conv_out.weight", &c->og[G_COW], GGML_TYPE_F16, C * 16 * E}, {"a.position_embd.weight", &c->og[G_POS], GGML_TYPE_F32, 0},
      {"a.post_ln.weight", &c->og[G_POSTW], GGML_TYPE_F32, E}, {"a.post_ln.bias", &c->og[G_POSTB], GGML_TYPE_F32, E},
      {"mm.a.mlp.1.weight", &c->og[G_M1W], GGML_TYPE_BF16, E * E}, {"mm.a.mlp.1.bias", &c->og[G_M1B], GGML_TYPE_F32, E},
      {"mm.a.mlp.2.weight", &c->og[G_M2W], GGML_TYPE_BF16, E * c->PD}, {"mm.a.mlp.2.bias", &c->og[G_M2B], GGML_TYPE_F32, (size_t)c->PD}};
    const size_t ln[T_NL] = {E, E, E * E, E, E * E, E, E * E, E, E * E, E, E, E, F * E, F, E * F, E};
    c->ol = (size_t *)calloc((size_t)c->L * T_NL, sizeof(size_t));
    for (int il = 0; il < c->L; il++)
      for (int t = 0; t < T_NL; t++) {
        char nm[96];
        snprintf(nm, sizeof nm, "a.blk.%d.%s", il, LNAME[t]);
        const bool mat = t == T_QW || t == T_KW || t == T_VW || t == T_OW || t == T_UPW || t == T_DNW;
        ws.push_back({nm, &c->ol[(size_t)il * T_NL + t], mat ? GGML_TYPE_BF16 : GGML_TYPE_F32, ln[t]});
      }
    size_t off = 0;
    for (auto &w : ws) {
      ggml_tensor *t = ggml_get_tensor(meta, w.name.c_str());
      if (!t || t->type != w.ty || (w.n && (size_t)ggml_nelements(t) != w.n)) {
        snprintf(err, 256, "aut: %s: %s type %s n %lld (want %s n %zu)%s", w.name.c_str(), t ? "found" : "missing",
                 t ? ggml_type_name(t->type) : "-", t ? (long long)ggml_nelements(t) : 0LL, ggml_type_name(w.ty), w.n,
                 t && t->type == GGML_TYPE_Q8_0 ? " [Q8_0 mmproj: not in scope]" : "");
        gguf_free(g), ggml_free(meta), aut_free(c);
        return NULL;
      }
      *w.slot = off;
      off += al256(ggml_nbytes(t));
    }
    c->wbytes = off;
    c->dev = MTLCreateSystemDefaultDevice();
    if (!c->dev) { snprintf(err, 256, "aut: no Metal device"); gguf_free(g), ggml_free(meta), aut_free(c); return NULL; }
    c->wb = [c->dev newBufferWithLength:off options:MTLResourceStorageModeShared];
    if (!c->wb) { snprintf(err, 256, "aut: weight buffer of %zu bytes failed", off); gguf_free(g), ggml_free(meta), aut_free(c); return NULL; }
    char *stage = (char *)c->wb.contents;
    for (auto &w : ws) { ggml_tensor *t = ggml_get_tensor(meta, w.name.c_str()); memcpy(stage + *w.slot, t->data, ggml_nbytes(t)); }
    gguf_free(g), ggml_free(meta);
    MTLCompileOptions *co = [MTLCompileOptions new];
    if (@available(macOS 15.0, iOS 18.0, *)) co.mathMode = MTLMathModeSafe; else co.fastMathEnabled = NO;
    NSError *ne = nil;
    std::string msl = MSL;
#ifdef AUT_DIAG  // diagnostic builds: $AUT_MSL_PREFIX is prepended (variant #defines / #pragma METAL fp contract(off))
    if (const char *pre = getenv("AUT_MSL_PREFIX")) msl = std::string(pre) + "\n" + msl;
#endif
    id<MTLLibrary> lib = [c->dev newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:co error:&ne];
    if (!lib) { snprintf(err, 256, "aut: MSL compile: %s", ne.localizedDescription.UTF8String); aut_free(c); return NULL; }
    for (int i = 0; i < K_N; i++) {
      id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:KNAME[i]]];
      c->ps[i] = f ? [c->dev newComputePipelineStateWithFunction:f error:&ne] : nil;
      if (!c->ps[i]) { snprintf(err, 256, "aut: pipeline %s failed", KNAME[i]); aut_free(c); return NULL; }
    }
    c->q = [c->dev newCommandQueue];
    return c;
  }
}

static int ensure(AutW *c, size_t ntok, char *err) {
  if (ntok <= c->cap) return 0;
  const size_t nch = (ntok + 12) / 13, E = c->E, C = c->C;
  const size_t big = nch * 64 * 50 * C;
  const size_t wide = ntok * (size_t)(c->F > (int)(C * 16) ? c->F : C * 16);
  const size_t rows = nch * 32 * 25 * C * 9 > big ? nch * 32 * 25 * C * 9 : big;
  const size_t sz[B_N] = {nch * 100 * 128 * 4, big * 4, nch * 32 * 25 * C * 4, nch * 16 * 13 * C * 4, ntok * E * 4,
                          ntok * E * 4, (wide > big ? wide : big) * 4, ntok * E * 4, ntok * E * 4, ntok * E * 4, ntok * E * 4,
                          wide * 4, ntok * (size_t)c->PD * 4, rows * 2, wide * 2, nch * 100 * 128 * 4};
  for (int i = 0; i < B_N; i++) {
    c->b[i] = [c->dev newBufferWithLength:sz[i] options:MTLResourceStorageModeShared];
    if (!c->b[i]) { snprintf(err, 256, "aut: activation buffer %d (%zu bytes) failed", i, sz[i]); c->cap = 0; return -1; }
  }
  c->cap = ntok;
  return 0;
}

static int wants(const AutOpt *o, const char *name) { return o && o->dump && (!o->want || o->want(o->ud, name)); }

// a recorder: one encoder per command buffer; flush() commits + waits (dumps need host-visible results mid-graph)
struct Rec {
  AutW *c;
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> enc;
  double gpu_s = 0;
  void open() { cb = [c->q commandBuffer]; enc = [cb computeCommandEncoder]; }
  int flush(char *err) {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      snprintf(err, 256, "aut: command buffer failed: %s", cb.error.localizedDescription.UTF8String);
      return -1;
    }
    gpu_s += cb.GPUEndTime - cb.GPUStartTime;
    open();
    return 0;
  }
  void lin(int k, long n) {  // 1-D grid
    [enc setComputePipelineState:c->ps[k]];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  void buf(id<MTLBuffer> b, size_t off, int i) { [enc setBuffer:b offset:off atIndex:i]; }
  template <typename T> void val(const T &v, int i) { [enc setBytes:&v length:sizeof v atIndex:i]; }
};

extern "C" int aut_encode(AutW *c, const float *mel, int n_mel, int n_len, const AutOpt *o, float *embd, char *err) {
  @autoreleasepool {
    const int E = c->E, H = c->H, D = c->D, F = c->F, PD = c->PD, C = c->C;
    if (n_mel != 128 || n_len % 100) { snprintf(err, 256, "aut: window %dx%d (want 128 x k*100)", n_mel, n_len); return -1; }
    const int nch = n_len / 100, N = nch * 13;
    if (N > 128) { snprintf(err, 256, "aut: %d tokens per window (k_attn max 128)", N); return -1; }
    if (ensure(c, (size_t)N, err)) return -1;
    __strong id<MTLBuffer> *b = c->b;
    Rec r{c};
    r.open();
    char nm[64];
    auto F32 = [&](int i) { return (const float *)b[i].contents; };
    auto dump = [&](const char *name, int bi, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) -> int {
      if (r.flush(err)) return -1;
      const int64_t ne[4] = {ne0, ne1, ne2, ne3};
      o->dump(o->ud, name, F32(bi), ne);
      return 0;
    };
#define DUMP2(name, bi, ne0, ne1) do { if (wants(o, name) && dump(name, bi, ne0, ne1, 1, 1)) return -1; } while (0)
#define LDUMP(base, il, bi, ne0, ne1) do { snprintf(nm, sizeof nm, "%s-%d", base, il); DUMP2(nm, bi, ne0, ne1); } while (0)
    auto gemm = [&](bool bf, size_t woff, int bx, int by, int n, int k, int oo) {
      [r.enc setComputePipelineState:c->ps[bf ? K_GEMMBF : K_GEMM16]];
      r.buf(b[bx], 0, 0), r.buf(c->wb, woff, 1), r.buf(b[by], 0, 2);
      const GemmP p = {n, k, oo};
      r.val(p, 3);
      [r.enc dispatchThreadgroups:MTLSizeMake((oo + 31) / 32, (n + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    // epi(acc buffer -> dst buffer), bias at weight offset (or none), optional BF16 copy into B_B16
    auto epi = [&](int bacc, int bdst, long boff, long n, int oo, int flags, bool hb) {
      r.buf(b[bacc], 0, 0), r.buf(b[bdst], 0, 1);
      r.buf(c->wb, boff < 0 ? 0 : (size_t)boff, 2), r.buf(b[B_B16], 0, 3);
      const EpiP p = {(int)n, oo, flags, hb ? 1 : 0};
      r.val(p, 4);
      r.lin(K_EPI, n);
    };
    auto lnorm = [&](size_t woff, size_t boff) {
      [r.enc setComputePipelineState:c->ps[K_LN]];
      r.buf(b[B_X], 0, 0), r.buf(c->wb, woff, 1), r.buf(c->wb, boff, 2), r.buf(b[B_CUR], 0, 3), r.buf(b[B_B16], 0, 4);
      r.val(E, 5), r.val(c->eps, 6);
      [r.enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    // --- conv front end, NHWC (h = mel, w = time)
    memcpy(b[B_IN].contents, mel, sizeof(float) * 128 * (size_t)n_len);
    r.buf(b[B_IN], 0, 0), r.buf(b[B_MEL], 0, 1), r.val(n_mel, 2), r.val(n_len, 3);
    r.lin(K_CHUNKS, 128L * n_len);
    if (wants(o, "inp_chunks") && dump("inp_chunks", B_MEL, 100, 128, 1, nch)) return -1;
    struct { int IH, IW, IC, src, dst; size_t w, bb; } cv[3] = {
      {128, 100, 1, B_MEL, B_C1, c->og[G_C1W], c->og[G_C1B]}, {64, 50, C, B_C1, B_C2, c->og[G_C2W], c->og[G_C2B]},
      {32, 25, C, B_C2, B_C3, c->og[G_C3W], c->og[G_C3B]}};
    for (int i = 0; i < 3; i++) {
      const int OH = (cv[i].IH - 1) / 2 + 1, OW = (cv[i].IW - 1) / 2 + 1, K = cv[i].IC * 9;
      const long R = (long)nch * OH * OW;
      const ConvP p = {nch, cv[i].IH, cv[i].IW, cv[i].IC, OH, OW};
      r.buf(b[cv[i].src], 0, 0), r.buf(b[B_ROWS], 0, 1), r.val(p, 2);
      r.lin(K_IM2COL, R * K);
      gemm(false, cv[i].w, B_ROWS, B_TMP, (int)R, K, C);
      epi(B_TMP, cv[i].dst, (long)cv[i].bb, R * C, C, EP_BIAS | EP_GELU, false);
    }
    if (wants(o, "after_conv_blocks")) {  // NHWC [nch][16][13][C] -> ggml [OW=13][OH=16][C][nch]
      if (r.flush(err)) return -1;
      const float *a = F32(B_C3);
      std::vector<float> t((size_t)nch * 16 * 13 * C);
      for (int n = 0; n < nch; n++)
        for (int h = 0; h < 16; h++)
          for (int w = 0; w < 13; w++)
            for (int ch = 0; ch < C; ch++) t[(((size_t)n * C + ch) * 16 + h) * 13 + w] = a[(((size_t)n * 16 + h) * 13 + w) * C + ch];
      const int64_t ne[4] = {13, 16, C, nch};
      o->dump(o->ud, "after_conv_blocks", t.data(), ne);
    }
    { const ConvP p = {nch, 16, 13, C, 16, 13}; r.buf(b[B_C3], 0, 0), r.buf(b[B_ROWS], 0, 1), r.val(p, 2); r.lin(K_GATHER, (long)N * C * 16); }
    gemm(false, c->og[G_COW], B_ROWS, B_X, N, C * 16, E);
    DUMP2("after_conv_out", B_X, E, N);
    const int NE = N * E;
    r.buf(b[B_X], 0, 0), r.buf(c->wb, c->og[G_POS], 1), r.val(NE, 2), r.val(E, 3);
    r.lin(K_ADDPOS, NE);

    // --- transformer
    const AttP ap = {N, E, D, 1.0f / sqrtf((float)D)};
#ifdef AUT_DIAG
    static int diag_win = 0;
    const char *cpw = getenv("AUT_TF_CALLS_PER_WIN");  // cudaud-cli: --repeat plain encodes + 1 instrumented per window
    const int win = diag_win++ / (cpw ? atoi(cpw) : 2);
#endif
    for (int il = 0; il < c->L; il++) {
      const size_t *lo = c->ol + (size_t)il * T_NL;
#ifdef AUT_DIAG  // teacher forcing (diagnostic builds only): x_in of layer il := $AUT_TF_DIR/layer_out-(il-1).wW.f32
      if (const char *tf = getenv("AUT_TF_DIR"); tf && il > 0) {
        if (r.flush(err)) return -1;
        char pth[1024];
        snprintf(pth, sizeof pth, "%s/layer_out-%d.w%d.f32", tf, il - 1, win);
        FILE *fp = fopen(pth, "rb");
        if (!fp || fread(b[B_X].contents, sizeof(float), (size_t)NE, fp) != (size_t)NE) { snprintf(err, 256, "aut: TF read %s", pth); return -1; }
        fclose(fp);
      }
#endif
      lnorm(lo[T_LN1W], lo[T_LN1B]);
      LDUMP("layer_inp_normed", il, B_CUR, E, N);
      gemm(true, lo[T_QW], B_B16, B_TMP, N, E, E);
      epi(B_TMP, B_Q, (long)lo[T_QB], NE, E, EP_BIAS, false);
      gemm(true, lo[T_KW], B_B16, B_TMP, N, E, E);
      epi(B_TMP, B_K, (long)lo[T_KB], NE, E, EP_BIAS, false);
      gemm(true, lo[T_VW], B_B16, B_TMP, N, E, E);
      epi(B_TMP, B_V, (long)lo[T_VB], NE, E, EP_BIAS, false);
      snprintf(nm, sizeof nm, "Qcur-%d", il);
      if (wants(o, nm) && dump(nm, B_Q, D, H, N, 1)) return -1;
      [r.enc setComputePipelineState:c->ps[K_ATTN]];
      r.buf(b[B_Q], 0, 0), r.buf(b[B_K], 0, 1), r.buf(b[B_V], 0, 2), r.buf(b[B_AO], 0, 3), r.buf(b[B_B16], 0, 4), r.val(ap, 5);
      [r.enc dispatchThreadgroups:MTLSizeMake(N, H, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      LDUMP("kqv_out", il, B_AO, E, N);
      gemm(true, lo[T_OW], B_B16, B_TMP, N, E, E);
      epi(B_TMP, B_CUR, (long)lo[T_OB], NE, E, EP_BIAS, false);
      LDUMP("attn_out", il, B_CUR, E, N);
      epi(B_CUR, B_X, -1, NE, E, EP_RES, false);
      lnorm(lo[T_LN2W], lo[T_LN2B]);
      gemm(true, lo[T_UPW], B_B16, B_TMP, N, E, F);
      epi(B_TMP, B_UP, (long)lo[T_UPB], (long)N * F, F, EP_BIAS | EP_GELU, true);
      gemm(true, lo[T_DNW], B_B16, B_TMP, N, F, E);
      epi(B_TMP, B_CUR, (long)lo[T_DNB], NE, E, EP_BIAS, false);
      LDUMP("ffn_out", il, B_CUR, E, N);
      epi(B_CUR, B_X, -1, NE, E, EP_RES, false);
      LDUMP("layer_out", il, B_X, E, N);
    }
    lnorm(c->og[G_POSTW], c->og[G_POSTB]);
    DUMP2("after_transformer", B_CUR, E, N);
    // --- projector: mlp.1 + b -> GELU-erf -> mlp.2 + b
    gemm(true, c->og[G_M1W], B_B16, B_TMP, N, E, E);
    epi(B_TMP, B_UP, (long)c->og[G_M1B], NE, E, EP_BIAS | EP_GELU, true);
    gemm(true, c->og[G_M2W], B_B16, B_TMP, N, E, PD);
    epi(B_TMP, B_EMBD, (long)c->og[G_M2B], (long)N * PD, PD, EP_BIAS, false);
    [r.enc endEncoding];
    [r.cb commit];
    [r.cb waitUntilCompleted];
    if (r.cb.status != MTLCommandBufferStatusCompleted) {
      snprintf(err, 256, "aut: command buffer failed: %s", r.cb.error.localizedDescription.UTF8String);
      return -1;
    }
    r.gpu_s += r.cb.GPUEndTime - r.cb.GPUStartTime;
    c->gpu_ms = r.gpu_s * 1e3;
    memcpy(embd, F32(B_EMBD), sizeof(float) * (size_t)N * PD);
    if (wants(o, "projected")) { const int64_t ne[4] = {PD, N, 1, 1}; o->dump(o->ud, "projected", embd, ne); }
#undef DUMP2
#undef LDUMP
    return 0;
  }
}
