// aut.cu: EARS-G1 "cudaud", the Qwen3 AuT audio tower (mtmd PROJECTOR_TYPE_QWEN3A) on CUDA (sm_86). Graph = llama.cpp
// 2ca15f5 tools/mtmd/models/qwen3a.cpp + clip.cpp build_vit, op by op, for ONE 800-frame window per call:
//   mel window [128][n_len] -> 100-frame chunks -> 3 x [conv2d 3x3 s2 p1 (overlapping im2col, F16) + b -> GELU-erf]
//   -> [16 freq x 480 ch] = 7680 features/token -> conv_out (F16, no bias) -> + pos rows 0..12 per chunk
//   -> L x [LN -> Q,K,V (+b) -> bidirectional MHA D=64 over the whole window -> out+b -> +res -> LN -> up+b -> GELU-erf
//   -> down+b -> +res] -> post LN -> mm.a.mlp.1+b -> GELU-erf -> mm.a.mlp.2+b
// Precision staging mirrors the llama.cpp CPU goldens: conv / conv_out operands F16 (ggml_conv_2d im2col dst = kernel
// type F16; vec_dot_type(F16) = F16); block + projector GEMMs BF16 x BF16 (vec_dot_type(BF16) = BF16), F32 accumulate
// (cuBLAS); attention K/V F16 (the clip graph's flash-attn path casts K/V to F16), Q and softmax P F32, F32 accumulate:
// the staging the CPU goldens were produced with (ears-fix faemu2: mean 1e-7 vs CPU kqv_out; the older k_fa64 rounded Q and P
// to F16 for the tensor-core MMA, the systematic gap behind the G1 ch770 failures). Attention = k_attn32 (D = 64).
// Built with -fmad=false (kernels evaluate float expressions as written). Windows are split on the host (aud_mel).
#include "aut.h"
#include "gguf.h"
#include "ggml.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define SQRT_2_INV 0.70710678118654752440084436210484f
enum { EP_BIAS = 1, EP_GELU = 2, EP_RES = 4 };
enum { T_LN1W, T_LN1B, T_QW, T_QB, T_KW, T_KB, T_VW, T_VB, T_OW, T_OB, T_LN2W, T_LN2B, T_UPW, T_UPB, T_DNW, T_DNB, T_NL };
static const char *const LNAME[T_NL] = {"ln1.weight", "ln1.bias", "attn_q.weight", "attn_q.bias", "attn_k.weight",
  "attn_k.bias", "attn_v.weight", "attn_v.bias", "attn_out.weight", "attn_out.bias", "ln2.weight", "ln2.bias",
  "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};
enum { G_C1W, G_C1B, G_C2W, G_C2B, G_C3W, G_C3B, G_COW, G_POS, G_POSTW, G_POSTB, G_M1W, G_M1B, G_M2W, G_M2B, G_N };

struct AutW {
  int E, H, D, F, L, PD, C;  // C = conv channels (480)
  float eps;
  char *wb;
  size_t wbytes, og[G_N], *ol;
  cublasHandle_t cb;
  cudaEvent_t e0, e1;
  size_t cap;  // tokens
  float *mel, *c1, *c2, *c3, *x, *cur, *tmp, *q, *k, *v, *ao, *up, *embd;
  __half *rows, *h16, *qh, *kh, *vh;
  __nv_bfloat16 *b16;
  double gpu_ms;
  // (Stage 2, gemm streams) side streams for the per-window GEMMs, one cuBLAS handle each; created on first use
  int nss;
  cudaStream_t ss[8];
  cublasHandle_t cbs[8];
  cudaEvent_t ef, ej[8];
};

#define CK(call, what)                                                                                   \
  do {                                                                                                   \
    cudaError_t e_ = (call);                                                                             \
    if (e_ != cudaSuccess) { snprintf(err, 256, "aut: %s: %s", what, cudaGetErrorString(e_)); return -1; } \
  } while (0)

// ---------------------------------------------------------------- kernels

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * SQRT_2_INV)); }

// overlapping im2col of a 3x3 stride-2 pad-1 conv: src NHWC [N][IH][IW][IC] f32 -> rows [N*OH*OW][IC*9] f16,
// k = (ic*3 + kh)*3 + kw (= the ggml kernel layout ne [KW, KH, IC, OC]); h = mel axis, w = time axis
__global__ void k_im2col3(const float *src, __half *rows, int N, int IH, int IW, int IC, int OH, int OW) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long K = (long)IC * 9;
  if (t >= (long)N * OH * OW * K) return;
  const long r = t / K;
  const int k = (int)(t % K), ic = k / 9, kh = (k / 3) % 3, kw = k % 3;
  const int ow = (int)(r % OW), oh = (int)((r / OW) % OH), n = (int)(r / ((long)OW * OH));
  const int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
  float v = 0.0f;
  if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) v = src[(((long)n * IH + ih) * IW + iw) * IC + ic];
  rows[t] = __float2half_rn(v);
}

// mel window [128][n_len] -> conv1 input NHWC [nch][128][100][1] (= golden inp_chunks memory order)
__global__ void k_chunks(const float *mel, float *dst, int n_mel, int n_len) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)n_mel * n_len) return;
  const int w = (int)(t % 100), h = (int)((t / 100) % n_mel), n = (int)(t / (100L * n_mel));
  dst[t] = mel[(long)h * n_len + n * 100 + w];
}

// conv3 out NHWC [N][16][13][C] -> conv_out rows [N*13][C*16] f16, feature = c*16 + f (the qwen3a permute)
__global__ void k_gather(const float *src, __half *rows, int N, int OH, int OW, int C) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)N * OW * C * OH) return;
  const int f = (int)(t % OH), c = (int)((t / OH) % C);
  const long tok = t / ((long)OH * C);
  const int ow = (int)(tok % OW), n = (int)(tok / OW);
  rows[t] = __float2half_rn(src[(((long)n * OH + f) * OW + ow) * C + c]);
}

// x[t][e] = x[t][e] + pos[t % 13][e]
__global__ void k_addpos(float *x, const float *pos, long n, int E, int per) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long t = i / E;
  x[i] = x[i] + pos[(t % per) * E + i % E];
}

// dst = epi(acc): + bias[col], GELU-erf, + dst (residual), in that order; optional BF16 / F16 copies for the next GEMM
__global__ void k_epi(const float *acc, float *dst, const float *bias, __nv_bfloat16 *hb, long n, int o, int flags) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = acc[i];
  if (flags & EP_BIAS) v = v + bias[i % o];
  if (flags & EP_GELU) v = gelu_erf(v);
  if (flags & EP_RES) v = v + dst[i];
  dst[i] = v;
  if (hb) hb[i] = __float2bfloat16_rn(v);
}

// LayerNorm (weight, bias), one row per block; F32 out + BF16 copy (cudavis k_ln)
__global__ void k_ln(const float *src, const float *w, const float *b, float *dst, __nv_bfloat16 *h, int E, float eps) {
  __shared__ float red[32];
  const long row = blockIdx.x;
  const float *xr = src + row * E;
  float s = 0.0f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) s += xr[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  s = (threadIdx.x & 31) < (blockDim.x >> 5) ? red[threadIdx.x & 31] : 0.0f;
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / E;
  __syncthreads();
  float v = 0.0f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = (threadIdx.x & 31) < (blockDim.x >> 5) ? red[threadIdx.x & 31] : 0.0f;
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const float scale = 1.0f / sqrtf(v / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const float y = ((xr[i] - mean) * scale) * w[i] + b[i];
    dst[row * E + i] = y;
    h[row * E + i] = __float2bfloat16_rn(y);
  }
}

// q, k, v rows [n][E] f32 -> F16 [H][npad][D]
__global__ void k_split(const float *q, const float *k, const float *v, __half *qh, __half *kh, __half *vh, int n, int E,
                        int D, int npad) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)n * E) return;
  const int r = (int)(t / E), e = (int)(t % E), hh = e / D, d = e % D;
  const long o = ((long)hh * npad + r) * D + d;
  qh[o] = __float2half_rn(q[t]), kh[o] = __float2half_rn(k[t]), vh[o] = __float2half_rn(v[t]);
}

// cudavis R1 k_fa64, unchanged except the activation copy (BF16: the out-proj weight is BF16)
#define FA_D 64
#define FA_LH 72
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *(uint32_t *)&h;
}
__global__ void __launch_bounds__(128) k_fa64(const __half *qh, const __half *kh, const __half *vh, float *dst,
                                               __nv_bfloat16 *dh, int n, int npad, int E, float scale) {
  __shared__ __align__(16) __half sK[64 * FA_LH], sVt[64 * FA_LH];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, hh = blockIdx.y;
  const int qr = blockIdx.x * 64 + warp * 16;
  const __half *kg = kh + (long)hh * npad * FA_D, *vg = vh + (long)hh * npad * FA_D;
  uint32_t qa[4][4];
  {
    const __half *q0 = qh + ((long)hh * npad + qr + g) * FA_D, *q8 = q0 + 8 * FA_D;
    for (int kc = 0; kc < 4; kc++) {
      qa[kc][0] = *(const uint32_t *)(q0 + kc * 16 + 2 * t), qa[kc][1] = *(const uint32_t *)(q8 + kc * 16 + 2 * t);
      qa[kc][2] = *(const uint32_t *)(q0 + kc * 16 + 8 + 2 * t), qa[kc][3] = *(const uint32_t *)(q8 + kc * 16 + 8 + 2 * t);
    }
  }
  float o[8][4] = {}, M0 = -FLT_MAX / 2, M1 = -FLT_MAX / 2, l0 = 0.0f, l1 = 0.0f;
  for (int kb = 0; kb < n; kb += 64) {
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 8; i += 128) {
      const int r = i >> 3, c = (i & 7) * 8;
      *(uint4 *)(sK + r * FA_LH + c) = *(const uint4 *)(kg + (long)(kb + r) * FA_D + c);
      const uint4 v = *(const uint4 *)(vg + (long)(kb + r) * FA_D + c);
      const __half *vv = (const __half *)&v;
      for (int j = 0; j < 8; j++) sVt[(c + j) * FA_LH + r] = vv[j];
    }
    __syncthreads();
    float s[8][4];
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
      const __half *kr = sK + (nt * 8 + g) * FA_LH + 2 * t;
      for (int kc = 0; kc < 4; kc++) mma16816(s[nt], qa[kc], *(const uint32_t *)(kr + kc * 16), *(const uint32_t *)(kr + kc * 16 + 8));
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
    for (int nt = 0; nt < 8; nt++)
      for (int j = 0; j < 2; j++) {
        const bool ok = kb + nt * 8 + 2 * t + j < n;
        s[nt][j] = ok ? s[nt][j] * scale : -INFINITY;
        s[nt][2 + j] = ok ? s[nt][2 + j] * scale : -INFINITY;
        mx0 = fmaxf(mx0, s[nt][j]), mx1 = fmaxf(mx1, s[nt][2 + j]);
      }
    for (int x = 1; x <= 2; x <<= 1) mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x)), mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
    const float Mn0 = fmaxf(M0, mx0), Mn1 = fmaxf(M1, mx1), ms0 = expf(M0 - Mn0), ms1 = expf(M1 - Mn1);
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int nt = 0; nt < 8; nt++)
      for (int j = 0; j < 2; j++) {
        s[nt][j] = expf(s[nt][j] - Mn0), sum0 += s[nt][j];
        s[nt][2 + j] = expf(s[nt][2 + j] - Mn1), sum1 += s[nt][2 + j];
      }
    for (int x = 1; x <= 2; x <<= 1) sum0 += __shfl_xor_sync(0xffffffffu, sum0, x), sum1 += __shfl_xor_sync(0xffffffffu, sum1, x);
    l0 = l0 * ms0 + sum0, l1 = l1 * ms1 + sum1, M0 = Mn0, M1 = Mn1;
    for (int dt = 0; dt < 8; dt++) o[dt][0] *= ms0, o[dt][1] *= ms0, o[dt][2] *= ms1, o[dt][3] *= ms1;
    for (int kc = 0; kc < 4; kc++) {
      const uint32_t pa[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]),
                              pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
      for (int dt = 0; dt < 8; dt++) {
        const __half *vr = sVt + (dt * 8 + g) * FA_LH + kc * 16 + 2 * t;
        mma16816(o[dt], pa, *(const uint32_t *)vr, *(const uint32_t *)(vr + 8));
      }
    }
  }
  const float inv0 = l0 == 0.0f ? 0.0f : 1.0f / l0, inv1 = l1 == 0.0f ? 0.0f : 1.0f / l1;
  for (int half8 = 0; half8 < 2; half8++) {
    const int q = qr + g + 8 * half8;
    if (q >= n) continue;
    const float inv = half8 ? inv1 : inv0;
    float *d = dst + (long)q * E + hh * FA_D;
    __nv_bfloat16 *dq = dh + (long)q * E + hh * FA_D;
    for (int dt = 0; dt < 8; dt++) {
      const float v0 = o[dt][2 * half8] * inv, v1 = o[dt][2 * half8 + 1] * inv;
      const int col = dt * 8 + 2 * t;
      *(float2 *)(d + col) = make_float2(v0, v1);
      dq[col] = __float2bfloat16_rn(v0), dq[col + 1] = __float2bfloat16_rn(v1);
    }
  }
}

// exact-staging attention: one block per (query, head), D = 64 threads. Q F32, K/V F16 -> F32, scores/softmax/P F32,
// F32 accumulate, normalise at the end. Shared: q[64] + scores[n]. Activation copy BF16 (out-proj weight is BF16).
// (CHAIN-2-1-3 Stage 2) body shared by the single launch and the batched one: query row i, keys/values kg/vg = this
// head's first key row of the query's window, n keys.
__device__ __forceinline__ void attn32_body(const float *q, const __half *kg, const __half *vg, float *dst,
                                            __nv_bfloat16 *dh, int i, int n, int E, float scale) {
  extern __shared__ float sm[];
  float *qs = sm, *sc = sm + 64;
  __shared__ float red[2];
  const int hh = blockIdx.y, d = threadIdx.x;
  qs[d] = q[(long)i * E + hh * 64 + d];
  __syncthreads();
  for (int j = d; j < n; j += 64) {
    const __half *kr = kg + (long)j * 64;
    float s = 0.0f;
    for (int t = 0; t < 64; t++) s = s + qs[t] * __half2float(kr[t]);
    sc[j] = s * scale;
  }
  __syncthreads();
  if (d == 0) {
    float M = -INFINITY;
    for (int j = 0; j < n; j++) M = fmaxf(M, sc[j]);
    red[0] = M;
  }
  __syncthreads();
  for (int j = d; j < n; j += 64) sc[j] = expf(sc[j] - red[0]);
  __syncthreads();
  if (d == 0) {
    float S = 0.0f;
    for (int j = 0; j < n; j++) S = S + sc[j];
    red[1] = S;
  }
  __syncthreads();
  float acc = 0.0f;
  for (int j = 0; j < n; j++) acc = acc + sc[j] * __half2float(vg[(long)j * 64 + d]);
  const float v = acc / red[1];
  dst[(long)i * E + hh * 64 + d] = v;
  dh[(long)i * E + hh * 64 + d] = __float2bfloat16_rn(v);
}
__global__ void __launch_bounds__(64) k_attn32(const float *q, const __half *kh, const __half *vh, float *dst,
                                               __nv_bfloat16 *dh, int n, int npad, int E, float scale) {
  const long ho = (long)blockIdx.y * npad * 64;
  attn32_body(q, kh + ho, vh + ho, dst, dh, blockIdx.x, n, E, scale);
}
// batched windows: row i belongs to the window starting at row s0[i] with sn[i] rows (bidirectional within it only)
__global__ void __launch_bounds__(64) k_attn32_seg(const float *q, const __half *kh, const __half *vh, float *dst,
                                                   __nv_bfloat16 *dh, const int *s0, const int *sn, int npad, int E,
                                                   float scale) {
  const int i = blockIdx.x;
  const long ho = ((long)blockIdx.y * npad + s0[i]) * 64;
  attn32_body(q, kh + ho, vh + ho, dst, dh, i, sn[i], E, scale);
}

// ---------------------------------------------------------------- host

static size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

extern "C" void aut_free(AutW *w) {
  if (!w) return;
  void *bufs[] = {w->wb, w->mel, w->c1, w->c2, w->c3, w->x, w->cur, w->tmp, w->q, w->k, w->v, w->ao, w->up, w->embd,
                  w->rows, w->h16, w->qh, w->kh, w->vh, w->b16};
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) cudaFree(bufs[i]);
  if (w->cb) cublasDestroy(w->cb);
  if (w->e0) cudaEventDestroy(w->e0), cudaEventDestroy(w->e1);
  for (int i = 0; i < w->nss; i++) cublasDestroy(w->cbs[i]), cudaStreamDestroy(w->ss[i]), cudaEventDestroy(w->ej[i]);
  if (w->nss) cudaEventDestroy(w->ef);
  free(w->ol), free(w);
}
extern "C" int aut_proj_dim(const AutW *w) { return w->PD; }
extern "C" int aut_n_tokens(int n_len) { return n_len / 100 * 13; }
extern "C" double aut_gpu_ms(const AutW *w) { return w->gpu_ms; }
extern "C" size_t aut_weight_bytes(const AutW *w) { return w->wbytes; }

extern "C" AutW *aut_load(const char *path, char *err) {
  ggml_context *meta = NULL;
  gguf_init_params gp = {false, &meta};
  gguf_context *g = gguf_init_from_file(path, gp);
  if (!g) { snprintf(err, 256, "aut: cannot read %s", path); return NULL; }
  AutW *c = (AutW *)calloc(1, sizeof *c);
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
    snprintf(err, 256, "aut: head dim %d (E %d / H %d): k_fa64 needs 64", c->D, c->E, c->H);
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
               t && t->type == GGML_TYPE_Q8_0 ? " [Q8_0 mmproj: not in G1 scope]" : "");
      gguf_free(g), ggml_free(meta), aut_free(c);
      return NULL;
    }
    *w.slot = off;
    off += al256(ggml_nbytes(t));
  }
  c->wbytes = off;
  char *stage = (char *)calloc(1, off);
  for (auto &w : ws) { ggml_tensor *t = ggml_get_tensor(meta, w.name.c_str()); memcpy(stage + *w.slot, t->data, ggml_nbytes(t)); }
  gguf_free(g), ggml_free(meta);
  if (cudaMalloc(&c->wb, off) != cudaSuccess || cudaMemcpy(c->wb, stage, off, cudaMemcpyHostToDevice) != cudaSuccess) {
    snprintf(err, 256, "aut: weight upload of %zu bytes failed", off);
    free(stage), aut_free(c);
    return NULL;
  }
  free(stage);
  if (cublasCreate(&c->cb) != CUBLAS_STATUS_SUCCESS) { snprintf(err, 256, "aut: cublasCreate failed"); aut_free(c); return NULL; }
  cudaEventCreate(&c->e0), cudaEventCreate(&c->e1);
  return c;
}

static int ensure(AutW *c, size_t ntok, char *err) {
  if (ntok <= c->cap) return 0;
  void **bufs[] = {(void **)&c->mel, (void **)&c->c1, (void **)&c->c2, (void **)&c->c3, (void **)&c->x, (void **)&c->cur,
                   (void **)&c->tmp, (void **)&c->q, (void **)&c->k, (void **)&c->v, (void **)&c->ao, (void **)&c->up,
                   (void **)&c->embd, (void **)&c->rows, (void **)&c->h16, (void **)&c->qh, (void **)&c->kh, (void **)&c->vh,
                   (void **)&c->b16};
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) cudaFree(*bufs[i]), *bufs[i] = NULL;
  const size_t nch = (ntok + 12) / 13, E = c->E, C = c->C, npad = (ntok + 63) / 64 * 64;
  const size_t big = nch * 64 * 50 * C;  // conv1 output
  const size_t wide = ntok * (size_t)(c->F > (int)(C * 16) ? c->F : C * 16);
  const size_t rows = nch * 32 * 25 * C * 9 > big ? nch * 32 * 25 * C * 9 : big;
  const size_t sz[] = {nch * 100 * 128 * 4, big * 4, nch * 32 * 25 * C * 4, nch * 16 * 13 * C * 4, ntok * E * 4, ntok * E * 4,
                       (wide > big ? wide : big) * 4, ntok * E * 4, ntok * E * 4, ntok * E * 4, ntok * E * 4, wide * 4,
                       ntok * (size_t)c->PD * 4, rows * 2, wide * 2, c->H * npad * 64 * 2, c->H * npad * 64 * 2,
                       c->H * npad * 64 * 2, wide * 2};
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) CK(cudaMalloc(bufs[i], sz[i]), "alloc activations");
  c->cap = ntok;
  return 0;
}

// Y[n][o] = X[n][k] . W[o][k]^T, F32 accumulate; X and W both F16 or both BF16
static int gemm_h(AutW *c, cublasHandle_t cb, size_t woff, const void *X, cudaDataType ty, float *Y, int n, int k, int o,
                  char *err) {
  const float one = 1.0f, zero = 0.0f;
  const cublasStatus_t s = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, o, n, k, &one, c->wb + woff, ty, k, X, ty, k,
                                        &zero, Y, CUDA_R_32F, o, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (s != CUBLAS_STATUS_SUCCESS) { snprintf(err, 256, "aut: cublasGemmEx %dx%dx%d: status %d", n, k, o, (int)s); return -1; }
  return 0;
}
static int gemm(AutW *c, size_t woff, const void *X, cudaDataType ty, float *Y, int n, int k, int o, char *err) {
  return gemm_h(c, c->cb, woff, X, ty, Y, n, k, o, err);
}
// (Stage 2, gemm streams) side streams + handles, created once (non-blocking streams, joined by events)
static int streams_init(AutW *c, char *err) {
  if (c->nss) return 0;
  if (cudaEventCreateWithFlags(&c->ef, cudaEventDisableTiming) != cudaSuccess) { snprintf(err, 256, "aut: stream event"); return -1; }
  for (int i = 0; i < 8; i++) {
    if (cudaStreamCreateWithFlags(&c->ss[i], cudaStreamNonBlocking) != cudaSuccess || cublasCreate(&c->cbs[i]) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(c->cbs[i], c->ss[i]) != CUBLAS_STATUS_SUCCESS || cudaEventCreateWithFlags(&c->ej[i], cudaEventDisableTiming) != cudaSuccess) {
      snprintf(err, 256, "aut: side stream %d setup failed", i); return -1;
    }
    c->nss = i + 1;
  }
  return 0;
}

static int wants(const AutOpt *o, const char *name) { return o && o->dump && (!o->want || o->want(o->ud, name)); }
static unsigned nb(long n) { return (unsigned)((n + 255) / 256); }

extern "C" int aut_encode(AutW *c, const float *mel, int n_mel, int n_len, const AutOpt *o, float *embd, char *err) {
  const int E = c->E, H = c->H, D = c->D, F = c->F, PD = c->PD, C = c->C;
  if (n_mel != 128 || n_len % 100) { snprintf(err, 256, "aut: window %dx%d (want 128 x k*100)", n_mel, n_len); return -1; }
  const int nch = n_len / 100, N = nch * 13, npad = (N + 63) / 64 * 64;
  if (ensure(c, (size_t)N, err)) return -1;
  std::vector<float> hbuf;
  char nm[64];
  auto dump = [&](const char *name, const float *dev, size_t n, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) -> int {
    hbuf.resize(n);
    CK(cudaMemcpy(hbuf.data(), dev, n * 4, cudaMemcpyDeviceToHost), "dump copy");
    const int64_t ne[4] = {ne0, ne1, ne2, ne3};
    o->dump(o->ud, name, hbuf.data(), ne);
    return 0;
  };
#define DUMP2(name, buf, ne0, ne1) do { if (wants(o, name) && dump(name, buf, (size_t)(ne0) * (ne1), ne0, ne1, 1, 1)) return -1; } while (0)
#define LDUMP(base, il, buf, ne0, ne1) do { snprintf(nm, sizeof nm, "%s-%d", base, il); DUMP2(nm, buf, ne0, ne1); } while (0)
#define G16(woff, X, Y, n, k, oo) do { if (gemm(c, woff, X, CUDA_R_16F, Y, n, k, oo, err)) return -1; } while (0)
#define GB(woff, X, Y, n, k, oo) do { if (gemm(c, woff, X, CUDA_R_16BF, Y, n, k, oo, err)) return -1; } while (0)
#define W32(off) ((const float *)(c->wb + (off)))

  cudaEventRecord(c->e0);
  // --- conv front end, NHWC (h = mel, w = time)
  CK(cudaMemcpy(c->tmp, mel, sizeof(float) * 128 * (size_t)n_len, cudaMemcpyHostToDevice), "mel upload");
  k_chunks<<<nb(128L * n_len), 256>>>(c->tmp, c->mel, 128, n_len);
  if (wants(o, "inp_chunks") && dump("inp_chunks", c->mel, 128 * (size_t)n_len, 100, 128, 1, nch)) return -1;
  struct { int IH, IW, IC; float *src, *dst; size_t w, b; } cv[3] = {
    {128, 100, 1, c->mel, c->c1, c->og[G_C1W], c->og[G_C1B]}, {64, 50, C, c->c1, c->c2, c->og[G_C2W], c->og[G_C2B]},
    {32, 25, C, c->c2, c->c3, c->og[G_C3W], c->og[G_C3B]}};
  for (int i = 0; i < 3; i++) {
    const int OH = (cv[i].IH - 1) / 2 + 1, OW = (cv[i].IW - 1) / 2 + 1, K = cv[i].IC * 9;
    const long R = (long)nch * OH * OW;
    k_im2col3<<<nb(R * K), 256>>>(cv[i].src, c->rows, nch, cv[i].IH, cv[i].IW, cv[i].IC, OH, OW);
    G16(cv[i].w, c->rows, c->tmp, (int)R, K, C);
    k_epi<<<nb(R * C), 256>>>(c->tmp, cv[i].dst, W32(cv[i].b), NULL, R * C, C, EP_BIAS | EP_GELU);
  }
  if (wants(o, "after_conv_blocks")) {  // NHWC [nch][16][13][C] -> ggml [OW=13][OH=16][C][nch]
    std::vector<float> a((size_t)nch * 16 * 13 * C), b(a.size());
    CK(cudaMemcpy(a.data(), c->c3, a.size() * 4, cudaMemcpyDeviceToHost), "dump copy");
    for (int n = 0; n < nch; n++)
      for (int h = 0; h < 16; h++)
        for (int w = 0; w < 13; w++)
          for (int ch = 0; ch < C; ch++) b[(((size_t)n * C + ch) * 16 + h) * 13 + w] = a[(((size_t)n * 16 + h) * 13 + w) * C + ch];
    const int64_t ne[4] = {13, 16, C, nch};
    o->dump(o->ud, "after_conv_blocks", b.data(), ne);
  }
  k_gather<<<nb((long)N * C * 16), 256>>>(c->c3, c->rows, nch, 16, 13, C);
  G16(c->og[G_COW], c->rows, c->x, N, C * 16, E);
  DUMP2("after_conv_out", c->x, E, N);
  const long NE = (long)N * E;
  k_addpos<<<nb(NE), 256>>>(c->x, W32(c->og[G_POS]), NE, E, 13);

  // --- transformer
  const float scale = 1.0f / sqrtf((float)D);
  const size_t hsz = (size_t)H * npad * D * 2;
  cudaMemsetAsync(c->qh, 0, hsz), cudaMemsetAsync(c->kh, 0, hsz), cudaMemsetAsync(c->vh, 0, hsz);
  for (int il = 0; il < c->L; il++) {
    const size_t *lo = c->ol + (size_t)il * T_NL;
    k_ln<<<N, 256>>>(c->x, W32(lo[T_LN1W]), W32(lo[T_LN1B]), c->cur, c->b16, E, c->eps);
    LDUMP("layer_inp_normed", il, c->cur, E, N);
    GB(lo[T_QW], c->b16, c->tmp, N, E, E);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->q, W32(lo[T_QB]), NULL, NE, E, EP_BIAS);
    GB(lo[T_KW], c->b16, c->tmp, N, E, E);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->k, W32(lo[T_KB]), NULL, NE, E, EP_BIAS);
    GB(lo[T_VW], c->b16, c->tmp, N, E, E);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->v, W32(lo[T_VB]), NULL, NE, E, EP_BIAS);
    snprintf(nm, sizeof nm, "Qcur-%d", il);
    if (wants(o, nm) && dump(nm, c->q, NE, D, H, N, 1)) return -1;
    k_split<<<nb(NE), 256>>>(c->q, c->k, c->v, c->qh, c->kh, c->vh, N, E, D, npad);
    k_attn32<<<dim3(N, H), 64, (64 + N) * sizeof(float)>>>(c->q, c->kh, c->vh, c->ao, c->b16, N, npad, E, scale);
    LDUMP("kqv_out", il, c->ao, E, N);
    GB(lo[T_OW], c->b16, c->tmp, N, E, E);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_OB]), NULL, NE, E, EP_BIAS);
    LDUMP("attn_out", il, c->cur, E, N);
    k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
    k_ln<<<N, 256>>>(c->x, W32(lo[T_LN2W]), W32(lo[T_LN2B]), c->cur, c->b16, E, c->eps);
    GB(lo[T_UPW], c->b16, c->tmp, N, E, F);
    k_epi<<<nb((long)N * F), 256>>>(c->tmp, c->up, W32(lo[T_UPB]), c->b16, (long)N * F, F, EP_BIAS | EP_GELU);
    GB(lo[T_DNW], c->b16, c->tmp, N, F, E);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_DNB]), NULL, NE, E, EP_BIAS);
    LDUMP("ffn_out", il, c->cur, E, N);
    k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
    LDUMP("layer_out", il, c->x, E, N);
  }
  k_ln<<<N, 256>>>(c->x, W32(c->og[G_POSTW]), W32(c->og[G_POSTB]), c->cur, c->b16, E, c->eps);
  DUMP2("after_transformer", c->cur, E, N);
  // --- projector: mlp.1 + b -> GELU-erf -> mlp.2 + b
  GB(c->og[G_M1W], c->b16, c->tmp, N, E, E);
  k_epi<<<nb(NE), 256>>>(c->tmp, c->up, W32(c->og[G_M1B]), c->b16, NE, E, EP_BIAS | EP_GELU);
  GB(c->og[G_M2W], c->b16, c->tmp, N, E, PD);
  k_epi<<<nb((long)N * PD), 256>>>(c->tmp, c->embd, W32(c->og[G_M2B]), NULL, (long)N * PD, PD, EP_BIAS);
  CK(cudaMemcpy(embd, c->embd, sizeof(float) * (size_t)N * PD, cudaMemcpyDeviceToHost), "embd copy");
  cudaEventRecord(c->e1);
  CK(cudaEventSynchronize(c->e1), "encode");
  CK(cudaGetLastError(), "kernel launch");
  float ms = 0;
  cudaEventElapsedTime(&ms, c->e0, c->e1);
  c->gpu_ms = ms;
  if (wants(o, "projected")) { const int64_t ne[4] = {PD, N, 1, 1}; o->dump(o->ud, "projected", embd, ne); }
#undef DUMP2
#undef LDUMP
#undef G16
#undef GB
#undef W32
  return 0;
}

// ---------------------------------------------------------------- batch (CHAIN-2-1-3 Stage 2)
// nwin mel windows in one pass: the windows' 100-frame chunks are concatenated along the chunk axis (the conv front end
// is per chunk), the tokens along the row axis (LN, GEMMs, epilogues, pos rows t % 13 are per row), and attention is
// block-diagonal per window (k_attn32_seg). gemm_seg 1 issues each GEMM per window at the single encode's shape;
// 2 issues those same GEMMs on up to 8 side streams (one cuBLAS handle each); 0 issues one GEMM over all rows.
// embd[i] gets window i's rows. No dumps.
extern "C" int aut_encode_batch(AutW *c, int nwin, const float *const *mel, const int *n_len, int gemm_seg,
                                float *const *embd, char *err) {
  const int E = c->E, H = c->H, D = c->D, F = c->F, PD = c->PD, C = c->C;
  if (nwin < 1) { snprintf(err, 256, "aut batch: no windows"); return -1; }
  std::vector<int> ch0(nwin + 1), t0(nwin + 1), s0, sn;
  int nch = 0, maxn = 0;
  for (int i = 0; i < nwin; i++) {
    if (n_len[i] % 100 || n_len[i] <= 0) { snprintf(err, 256, "aut batch: window %d n_len %d (want k*100)", i, n_len[i]); return -1; }
    ch0[i] = nch, t0[i] = nch * 13, nch += n_len[i] / 100;
    if (n_len[i] / 100 * 13 > maxn) maxn = n_len[i] / 100 * 13;
  }
  ch0[nwin] = nch, t0[nwin] = nch * 13;
  const int N = nch * 13, npad = (N + 63) / 64 * 64;
  for (int i = 0; i < nwin; i++)
    for (int t = t0[i]; t < t0[i + 1]; t++) s0.push_back(t0[i]), sn.push_back(t0[i + 1] - t0[i]);
  if (ensure(c, (size_t)N, err)) return -1;
  int *segd = NULL;
  CK(cudaMalloc(&segd, sizeof(int) * 2 * (size_t)N), "alloc seg");
  int rc = -1;
  if (gemm_seg == 2 && streams_init(c, err)) { cudaFree(segd); return -1; }
#define BCK(call, what) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { snprintf(err, 256, "aut batch: %s: %s", what, cudaGetErrorString(e_)); goto out; } } while (0)
#define BG(woff, X, ty, esz, Y, k, oo, offs)                                                                     \
  do {                                                                                                            \
    if (gemm_seg == 2) { /* fork: side streams wait for the default stream; GEMM i on stream i % 8; join */       \
      const int ns_ = nwin < 8 ? nwin : 8;                                                                        \
      cudaEventRecord(c->ef, 0);                                                                                  \
      for (int s_ = 0; s_ < ns_; s_++) cudaStreamWaitEvent(c->ss[s_], c->ef, 0);                                  \
      for (int i_ = 0; i_ < nwin; i_++)                                                                           \
        if (gemm_h(c, c->cbs[i_ % 8], woff, (const char *)(X) + (size_t)offs(i_) * (k) * (esz), ty,               \
                   (Y) + (size_t)offs(i_) * (oo), offs(i_ + 1) - offs(i_), k, oo, err)) goto out;                 \
      for (int s_ = 0; s_ < ns_; s_++) cudaEventRecord(c->ej[s_], c->ss[s_]), cudaStreamWaitEvent(0, c->ej[s_], 0); \
    } else if (gemm_seg) {                                                                                        \
      for (int i_ = 0; i_ < nwin; i_++)                                                                           \
        if (gemm(c, woff, (const char *)(X) + (size_t)offs(i_) * (k) * (esz), ty, (Y) + (size_t)offs(i_) * (oo),  \
                 offs(i_ + 1) - offs(i_), k, oo, err)) goto out;                                                  \
    } else if (gemm(c, woff, X, ty, Y, offs(nwin), k, oo, err)) goto out;                                         \
  } while (0)
#define W32(off) ((const float *)(c->wb + (off)))
#define TOK(i) t0[i]
  {
    cudaEventRecord(c->e0);
    BCK(cudaMemcpyAsync(segd, s0.data(), sizeof(int) * (size_t)N, cudaMemcpyHostToDevice), "seg upload");
    BCK(cudaMemcpyAsync(segd + N, sn.data(), sizeof(int) * (size_t)N, cudaMemcpyHostToDevice), "seg upload");
    for (int i = 0; i < nwin; i++) {
      float *src = c->tmp + (size_t)ch0[i] * 100 * 128;
      BCK(cudaMemcpy(src, mel[i], sizeof(float) * 128 * (size_t)n_len[i], cudaMemcpyHostToDevice), "mel upload");
      k_chunks<<<nb(128L * n_len[i]), 256>>>(src, c->mel + (size_t)ch0[i] * 100 * 128, 128, n_len[i]);
    }
    struct { int IH, IW, IC; float *src, *dst; size_t w, b; } cv[3] = {
      {128, 100, 1, c->mel, c->c1, c->og[G_C1W], c->og[G_C1B]}, {64, 50, C, c->c1, c->c2, c->og[G_C2W], c->og[G_C2B]},
      {32, 25, C, c->c2, c->c3, c->og[G_C3W], c->og[G_C3B]}};
    for (int j = 0; j < 3; j++) {
      const int OH = (cv[j].IH - 1) / 2 + 1, OW = (cv[j].IW - 1) / 2 + 1, K = cv[j].IC * 9;
      const long R = (long)nch * OH * OW;
      k_im2col3<<<nb(R * K), 256>>>(cv[j].src, c->rows, nch, cv[j].IH, cv[j].IW, cv[j].IC, OH, OW);
#define CONVR(i) (ch0[i] * OH * OW)
      BG(cv[j].w, c->rows, CUDA_R_16F, 2, c->tmp, K, C, CONVR);
#undef CONVR
      k_epi<<<nb(R * C), 256>>>(c->tmp, cv[j].dst, W32(cv[j].b), NULL, R * C, C, EP_BIAS | EP_GELU);
    }
    k_gather<<<nb((long)N * C * 16), 256>>>(c->c3, c->rows, nch, 16, 13, C);
    BG(c->og[G_COW], c->rows, CUDA_R_16F, 2, c->x, C * 16, E, TOK);
    const long NE = (long)N * E;
    k_addpos<<<nb(NE), 256>>>(c->x, W32(c->og[G_POS]), NE, E, 13);
    const float scale = 1.0f / sqrtf((float)D);
    const size_t hsz = (size_t)H * npad * D * 2;
    cudaMemsetAsync(c->qh, 0, hsz), cudaMemsetAsync(c->kh, 0, hsz), cudaMemsetAsync(c->vh, 0, hsz);
    for (int il = 0; il < c->L; il++) {
      const size_t *lo = c->ol + (size_t)il * T_NL;
      k_ln<<<N, 256>>>(c->x, W32(lo[T_LN1W]), W32(lo[T_LN1B]), c->cur, c->b16, E, c->eps);
      BG(lo[T_QW], c->b16, CUDA_R_16BF, 2, c->tmp, E, E, TOK);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->q, W32(lo[T_QB]), NULL, NE, E, EP_BIAS);
      BG(lo[T_KW], c->b16, CUDA_R_16BF, 2, c->tmp, E, E, TOK);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->k, W32(lo[T_KB]), NULL, NE, E, EP_BIAS);
      BG(lo[T_VW], c->b16, CUDA_R_16BF, 2, c->tmp, E, E, TOK);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->v, W32(lo[T_VB]), NULL, NE, E, EP_BIAS);
      k_split<<<nb(NE), 256>>>(c->q, c->k, c->v, c->qh, c->kh, c->vh, N, E, D, npad);
      k_attn32_seg<<<dim3(N, H), 64, (64 + maxn) * sizeof(float)>>>(c->q, c->kh, c->vh, c->ao, c->b16, segd, segd + N, npad, E, scale);
      BG(lo[T_OW], c->b16, CUDA_R_16BF, 2, c->tmp, E, E, TOK);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_OB]), NULL, NE, E, EP_BIAS);
      k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
      k_ln<<<N, 256>>>(c->x, W32(lo[T_LN2W]), W32(lo[T_LN2B]), c->cur, c->b16, E, c->eps);
      BG(lo[T_UPW], c->b16, CUDA_R_16BF, 2, c->tmp, E, F, TOK);
      k_epi<<<nb((long)N * F), 256>>>(c->tmp, c->up, W32(lo[T_UPB]), c->b16, (long)N * F, F, EP_BIAS | EP_GELU);
      BG(lo[T_DNW], c->b16, CUDA_R_16BF, 2, c->tmp, F, E, TOK);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_DNB]), NULL, NE, E, EP_BIAS);
      k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
    }
    k_ln<<<N, 256>>>(c->x, W32(c->og[G_POSTW]), W32(c->og[G_POSTB]), c->cur, c->b16, E, c->eps);
    BG(c->og[G_M1W], c->b16, CUDA_R_16BF, 2, c->tmp, E, E, TOK);
    k_epi<<<nb(NE), 256>>>(c->tmp, c->up, W32(c->og[G_M1B]), c->b16, NE, E, EP_BIAS | EP_GELU);
    BG(c->og[G_M2W], c->b16, CUDA_R_16BF, 2, c->tmp, E, PD, TOK);
    k_epi<<<nb((long)N * PD), 256>>>(c->tmp, c->embd, W32(c->og[G_M2B]), NULL, (long)N * PD, PD, EP_BIAS);
    for (int i = 0; i < nwin; i++)
      BCK(cudaMemcpy(embd[i], c->embd + (size_t)t0[i] * PD, sizeof(float) * (size_t)(t0[i + 1] - t0[i]) * PD, cudaMemcpyDeviceToHost), "embd copy");
    cudaEventRecord(c->e1);
    BCK(cudaEventSynchronize(c->e1), "encode");
    BCK(cudaGetLastError(), "kernel launch");
    float ms = 0;
    cudaEventElapsedTime(&ms, c->e0, c->e1);
    c->gpu_ms = ms;
    rc = 0;
  }
out:
  if (rc) cudaDeviceSynchronize();
  cudaFree(segd);
  return rc;
#undef BCK
#undef BG
#undef W32
#undef TOK
}
extern "C" void aut_mem_info(size_t *fr, size_t *tot) { cudaMemGetInfo(fr, tot); }
