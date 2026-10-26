// vitc.cu: see vitc.h. The CUDA tower: the mmproj's tensors are uploaded as stored (F16 matrices + conv kernels, F32
// biases/norms) into one device buffer; GEMMs are cuBLAS GemmEx (A = F16 weights, B = F16-staged activations, F32
// accumulate: the same staging as llama.cpp Metal's kernel_mul_mm_f16_f32, which the goldens came from); every other
// op is an explicit kernel below (one per op, the vitm.metal set). Kernels are built with -fmad=false so their float
// expressions are evaluated as written; the host math (pos resize) with -ffp-contract=off.
#include "vitc.h"
extern "C" {
#include "vmm.h"
}
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace nvcuda;

#define GELU_COEF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
enum { EP_BIAS = 1, EP_GELU = 2, EP_RES = 4 };
enum { T_LN1W, T_LN1B, T_QKVW, T_QKVB, T_OW, T_OB, T_LN2W, T_LN2B, T_UPW, T_UPB, T_DOWNW, T_DOWNB, T_NL };
static const char *const LNAME[T_NL] = {"ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
  "attn_out.bias", "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};

struct VitcW {
  VMmH h;
  int E, H, D, DP, F, L, P, M, PD, G;  // DP = D padded to a multiple of 16 (the FA tile; 64 -> 64, 72 -> 80)
  char *wb;  // device: all weights, 256-aligned offsets
  size_t wbytes, o_pe0, o_pe1, o_peb, o_postw, o_postb, o_mm0w, o_mm0b, o_mm2w, o_mm2b, *o_l;
  float *pos;  // host: v.position_embd [G*G][E]
  cublasHandle_t cb;
  cudaEvent_t e0, e1;
  size_t ncap, pixcap, actbytes;
  float *planar, *x, *cur, *tmp, *tmp2, *qkv, *ao, *up, *qr, *kd, *pos_d, *embd;
  __half *rows, *h16, *qh, *kh, *vh;
  int *ppos;
  int pos_px, pos_py;
  double gpu_ms;
  // (Stage 2, --gemm streams) side streams for the per-image GEMMs, one cuBLAS handle each; created on first use
  int nss;
  cudaStream_t ss[8];
  cublasHandle_t cbs[8];
  cudaEvent_t ef, ej[8];
};

#define CK(call, what)                                                                      \
  do {                                                                                      \
    cudaError_t e_ = (call);                                                                \
    if (e_ != cudaSuccess) { snprintf(err, 256, "vitc: %s: %s", what, cudaGetErrorString(e_)); return -1; } \
  } while (0)

// ---------------------------------------------------------------- kernels

__device__ __forceinline__ float gelu1(float x) { return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x))); }

// rows [n][768] half in 2x2-merge order from planar [3][ny][nx] F32 (= vt_im2col)
__global__ void k_im2col(const float *planar, __half *rows, int nx, int ny, int px, int n) {
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)n * 768) return;
  const int r = (int)(t / 768), kk = (int)(t % 768);
  const int b = r / 4, dy = (r % 4) / 2, dx = r % 2;
  const int y = 2 * (b / (px / 2)) + dy, x = 2 * (b % (px / 2)) + dx;
  const int ch = kk / 256, ky = (kk / 16) % 16, kx = kk % 16;
  rows[t] = __float2half_rn(planar[(long)ch * nx * ny + (long)(y * 16 + ky) * nx + x * 16 + kx]);
}

// x = ((c0 + c1) + bias[e]) + pos  (vt_embed, then vt_add of the position rows); xb = x before pos (patch_bias)
__global__ void k_embed(float *x, float *xb, const float *c0, const float *c1, const float *b, const float *pos, long n, int E) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float s = c0[i] + c1[i];
  const float v = s + b[i % E];
  if (xb) xb[i] = v;
  x[i] = v + pos[i];
}

// dst = epi(acc): + bias[col], GELU, + dst (residual), in that order; optional F16 copy for the next GEMM's input
__global__ void k_epi(const float *acc, float *dst, const float *bias, __half *h, long n, int o, int flags) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = acc[i];
  if (flags & EP_BIAS) v = v + bias[i % o];
  if (flags & EP_GELU) v = gelu1(v);
  if (flags & EP_RES) v = v + dst[i];
  dst[i] = v;
  if (h) h[i] = __float2half_rn(v);
}

// LayerNorm with weight and bias, one row per block (vt_ln: F32 sums, mean, centered variance)
__global__ void k_ln(const float *src, const float *w, const float *b, float *dst, __half *h, int E, float eps) {
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
    h[row * E + i] = __float2half_rn(y);
  }
}

// 2D vision rope (GGML_ROPE_TYPE_VISION, n_dims D/2, sections D/4 + D/4) on Q and K + the F16 casts of Q, K, V into
// [H][npad][D]; optional F32 Q / K rows [n][E] for dumps. One thread per (patch, head, pair).
// (Stage 2) prow: NULL = row r is padded row r (single); else the padded row of row r (batched segments)
__global__ void k_rope(const float *qkv, const int *pos, __half *qh, __half *kh, __half *vh, float *qd, float *kd, int n,
                       int E, int H, int D, int DP, int npad, const int *prow) {
  const int ND = D / 2, S0 = D / 4;
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long)n * H * ND) return;
  const int ic = (int)(t % ND), hh = (int)((t / ND) % H), r = (int)(t / ((long)ND * H));
  const float inv_ndims = -1.f / ND;
  const float p = ic < S0 ? (float)ic : (float)(ic - S0);
  const float base = ic < S0 ? (float)pos[r] : (float)pos[r + n];
  const float theta = base * powf(10000.0f, 2.0f * inv_ndims * p);
  const float c = cosf(theta), s = sinf(theta);
  const float *qs = qkv + (long)r * 3 * E + hh * D, *ks = qs + E, *vs = qs + 2 * E;
  const long o = ((long)hh * npad + (prow ? prow[r] : r)) * DP;  // dims D..DP-1 stay zero (memset per encode)
  float x0 = qs[ic], x1 = qs[ic + ND];
  const float q0 = x0 * c - x1 * s, q1 = x0 * s + x1 * c;
  x0 = ks[ic], x1 = ks[ic + ND];
  const float k0 = x0 * c - x1 * s, k1 = x0 * s + x1 * c;
  qh[o + ic] = __float2half_rn(q0), qh[o + ic + ND] = __float2half_rn(q1);
  kh[o + ic] = __float2half_rn(k0), kh[o + ic + ND] = __float2half_rn(k1);
  vh[o + ic] = __float2half_rn(vs[ic]), vh[o + ic + ND] = __float2half_rn(vs[ic + ND]);
  if (qd) { float *qq = qd + (long)r * E + hh * D; qq[ic] = q0, qq[ic + ND] = q1; }
  if (kd) { float *kq = kd + (long)r * E + hh * D; kq[ic] = k0, kq[ic + ND] = k1; }
}

// bidirectional flash attention (FA2 register form), head dim D stored padded to DP (a multiple of 16; the pad lanes of
// Q/K/V are zero, so Q K^T and P V are unchanged): block = 4 warps x 16 queries, one head (blockIdx.y); key blocks of 64
// staged in shared memory (K as is, V transposed). Per warp, mma.sync m16n8k16 (F16 in, F32 acc): S = Q K^T -> * scale,
// online softmax in F32 registers (keys >= n excluded) -> P rounded to F16 (the S accumulators re-packed as the A
// operand) -> O += P V in F32 registers -> O / l. Writes the first D columns as F32 rows [n][E] at h*D + their F16 copy.
// DP 64 = the 4B tower (R1's D-64 kernel, same arithmetic), DP 80 = the 9B/27B tower (D 72).
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *(uint32_t *)&h;
}
// (CHAIN-2-1-3 Stage 2) the body takes one head's Q/K/V planes (qg/kg/vg = that head's row 0) and the rows' dst/dh
// base, so the single launch (k_fa) and the batched one (k_fa_seg: one image segment per blockIdx.z) run the same code.
template <int DP>
__device__ __forceinline__ void fa_body(const __half *qg, const __half *kg, const __half *vg, float *dst, __half *dh,
                                        int n, int E, int D, float scale, int hh) {
  constexpr int KC = DP / 16, DT = DP / 8, LK = DP + 8, LV = 72;  // k-chunks of Q.K^T, 8-wide O tiles, smem strides
  __shared__ __align__(16) __half sK[64 * LK], sVt[DP * LV];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int qr = blockIdx.x * 64 + warp * 16;  // this warp's first query row
  uint32_t qa[KC][4];  // Q A-fragments, KC k-chunks of 16 dims
  {
    const __half *q0 = qg + (long)(qr + g) * DP, *q8 = q0 + 8 * DP;
    for (int kc = 0; kc < KC; kc++) {
      qa[kc][0] = *(const uint32_t *)(q0 + kc * 16 + 2 * t), qa[kc][1] = *(const uint32_t *)(q8 + kc * 16 + 2 * t);
      qa[kc][2] = *(const uint32_t *)(q0 + kc * 16 + 8 + 2 * t), qa[kc][3] = *(const uint32_t *)(q8 + kc * 16 + 8 + 2 * t);
    }
  }
  float o[DT][4] = {}, M0 = -FLT_MAX / 2, M1 = -FLT_MAX / 2, l0 = 0.0f, l1 = 0.0f;  // rows g (o[.][0..1]) and g+8 (o[.][2..3])

  for (int kb = 0; kb < n; kb += 64) {
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * DT; i += 128) {
      const int r = i / DT, c = (i % DT) * 8;
      *(uint4 *)(sK + r * LK + c) = *(const uint4 *)(kg + (long)(kb + r) * DP + c);
      const uint4 v = *(const uint4 *)(vg + (long)(kb + r) * DP + c);
      const __half *vv = (const __half *)&v;
      for (int j = 0; j < 8; j++) sVt[(c + j) * LV + r] = vv[j];
    }
    __syncthreads();
    float s[8][4];  // 8 tiles of 8 keys
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
      const __half *kr = sK + (nt * 8 + g) * LK + 2 * t;
      for (int kc = 0; kc < KC; kc++) mma16816(s[nt], qa[kc], *(const uint32_t *)(kr + kc * 16), *(const uint32_t *)(kr + kc * 16 + 8));
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
    for (int dt = 0; dt < DT; dt++) o[dt][0] *= ms0, o[dt][1] *= ms0, o[dt][2] *= ms1, o[dt][3] *= ms1;
    for (int kc = 0; kc < 4; kc++) {  // keys 16kc..16kc+15 = S tiles 2kc, 2kc+1
      const uint32_t pa[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]),
                              pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
      for (int dt = 0; dt < DT; dt++) {
        const __half *vr = sVt + (dt * 8 + g) * LV + kc * 16 + 2 * t;
        mma16816(o[dt], pa, *(const uint32_t *)vr, *(const uint32_t *)(vr + 8));
      }
    }
  }
  const float inv0 = l0 == 0.0f ? 0.0f : 1.0f / l0, inv1 = l1 == 0.0f ? 0.0f : 1.0f / l1;
  for (int half8 = 0; half8 < 2; half8++) {
    const int q = qr + g + 8 * half8;
    if (q >= n) continue;
    const float inv = half8 ? inv1 : inv0;
    float *d = dst + (long)q * E + hh * D;
    __half *dq = dh + (long)q * E + hh * D;
    for (int dt = 0; dt < DT; dt++) {
      const float v0 = o[dt][2 * half8] * inv, v1 = o[dt][2 * half8 + 1] * inv;
      const int col = dt * 8 + 2 * t;
      if (col >= D) continue;  // pad lanes (D even, col even: col + 1 < D too)
      *(float2 *)(d + col) = make_float2(v0, v1);
      *(__half2 *)(dq + col) = __floats2half2_rn(v0, v1);
    }
  }
}
template <int DP>
__global__ void __launch_bounds__(128) k_fa(const __half *qh, const __half *kh, const __half *vh, float *dst, __half *dh,
                                            int n, int npad, int E, int D, float scale) {
  const int hh = blockIdx.y;
  const long ho = (long)hh * npad * DP;
  fa_body<DP>(qh + ho, kh + ho, vh + ho, dst, dh, n, E, D, scale, hh);
}
// segment z: rows row0[z] .. row0[z]+sn[z]-1 of dst/dh, padded rows pr0[z] .. of the [H][npad][DP] planes
template <int DP>
__global__ void __launch_bounds__(128) k_fa_seg(const __half *qh, const __half *kh, const __half *vh, float *dst, __half *dh,
                                                const int *row0, const int *pr0, const int *sn, int npad, int E, int D,
                                                float scale) {
  const int hh = blockIdx.y, z = blockIdx.z, n = sn[z];
  if (blockIdx.x * 64 >= n) return;  // whole block: this segment has fewer query blocks than the widest one
  const long ho = ((long)hh * npad + pr0[z]) * DP, ro = (long)row0[z] * E;
  fa_body<DP>(qh + ho, kh + ho, vh + ho, dst + ro, dh + ro, n, E, D, scale, hh);
}

// ---------------------------------------------------------------- host

static size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

extern "C" void vitc_free(VitcW *w) {
  if (!w) return;
  void *bufs[] = {w->wb, w->planar, w->x, w->cur, w->tmp, w->tmp2, w->qkv, w->ao, w->up, w->qr, w->kd, w->pos_d, w->embd,
                  w->rows, w->h16, w->qh, w->kh, w->vh, w->ppos};
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) cudaFree(bufs[i]);
  if (w->cb) cublasDestroy(w->cb);
  if (w->e0) cudaEventDestroy(w->e0), cudaEventDestroy(w->e1);
  for (int i = 0; i < w->nss; i++) cublasDestroy(w->cbs[i]), cudaStreamDestroy(w->ss[i]), cudaEventDestroy(w->ej[i]);
  if (w->nss) cudaEventDestroy(w->ef);
  free(w->o_l), free(w->pos), free(w);
}

extern "C" int vitc_n_tokens(const VitcW *w, int nx, int ny) { return (nx / (w->P * w->M)) * (ny / (w->P * w->M)); }
extern "C" int vitc_proj_dim(const VitcW *w) { return w->PD; }
extern "C" double vitc_gpu_ms(const VitcW *w) { return w->gpu_ms; }
extern "C" size_t vitc_weight_bytes(const VitcW *w) { return w->wbytes; }
extern "C" size_t vitc_act_bytes(const VitcW *w) { return w->actbytes; }

// Stage GGUF BF16 matrices as F16 for the existing cuBLAS path. This is the
// same round-to-nearest-even conversion used by mmproj_to_f16.py, in memory.
static int stage_half(VMm *m, const char *name, int ty, size_t n, char *dst, char *err) {
  if (ty == 1) return vmm_raw(m, name, dst, err);
  if (ty != 0 && ty != 30) { snprintf(err, 256, "vitc: unsupported weight type %d", ty); return -1; }
  const size_t bytes = n * (ty == 0 ? 4 : 2);
  void *raw = malloc(bytes);
  if (!raw) { snprintf(err, 256, "vitc: weight staging allocation failed"); return -1; }
  if (vmm_raw(m, name, raw, err)) { free(raw); return -1; }
  for (size_t j = 0; j < n; j++) {
    float f;
    if (ty == 0) memcpy(&f, (char *)raw + j * 4, 4);
    else {
      uint32_t bits = (uint32_t)((uint16_t *)raw)[j] << 16;
      memcpy(&f, &bits, 4);
    }
    __half h = __float2half_rn(f);
    memcpy(dst + j * 2, &h, 2);
  }
  free(raw);
  return 0;
}

extern "C" VitcW *vitc_load(const char *mmproj, uint32_t n_embd_llm, char *err) {
  VMm *m = vmm_open(mmproj, n_embd_llm, err);
  if (!m) return NULL;
  VitcW *c = (VitcW *)calloc(1, sizeof *c);
  vmm_hparams(m, &c->h);
  c->E = (int)c->h.n_embd, c->H = (int)c->h.n_head, c->D = c->E / c->H, c->F = (int)c->h.n_ff, c->L = (int)c->h.n_layer;
  c->P = (int)c->h.patch, c->M = (int)c->h.n_merge, c->PD = (int)c->h.proj_dim, c->G = (int)(c->h.image_size / c->h.patch);
  c->DP = (c->D + 15) / 16 * 16;
  // FA instantiations: DP 64 (the 4B tower, D 64) and DP 80 (9B/27B/35B/FN, D 72)
  if ((c->DP != 64 && c->DP != 80) || c->D % 2 || c->H * c->D != c->E || c->P != 16 || c->M != 2) {
    snprintf(err, 256, "vitc: unsupported tower (head dim %d, heads %d, width %d, patch %d, merge %d): D 64 or 72 only", c->D, c->H, c->E, c->P, c->M);
    vmm_close(m), vitc_free(c);
    return NULL;
  }
  const size_t E = c->E, F = c->F, K = 3 * (size_t)c->P * c->P;
  enum { NG = 9 };
  struct { const char *name; size_t *slot; int f16; size_t n; } g[NG] = {
    {"v.patch_embd.weight", &c->o_pe0, 1, E * K}, {"v.patch_embd.weight.1", &c->o_pe1, 1, E * K},
    {"v.patch_embd.bias", &c->o_peb, 0, E}, {"v.post_ln.weight", &c->o_postw, 0, E}, {"v.post_ln.bias", &c->o_postb, 0, E},
    {"mm.0.weight", &c->o_mm0w, 1, 16 * E * E}, {"mm.0.bias", &c->o_mm0b, 0, 4 * E},
    {"mm.2.weight", &c->o_mm2w, 1, 4 * E * (size_t)c->PD}, {"mm.2.bias", &c->o_mm2b, 0, (size_t)c->PD}};
  const size_t ln[T_NL] = {E, E, 3 * E * E, 3 * E, E * E, E, E, E, F * E, F, E * F, E};
  c->o_l = (size_t *)calloc((size_t)c->L * T_NL, sizeof *c->o_l);
  char nm[96], *stage = NULL;
  for (int pass = 0; pass < 2; pass++) {
    size_t off = 0;
    for (int i = 0; i < NG + c->L * T_NL; i++) {
      const char *name;
      size_t *slot, n;
      int f16;
      if (i < NG) name = g[i].name, slot = g[i].slot, n = g[i].n, f16 = g[i].f16;
      else {
        const int il = (i - NG) / T_NL, t = (i - NG) % T_NL;
        snprintf(nm, sizeof nm, "v.blk.%d.%s", il, LNAME[t]);
        name = nm, slot = &c->o_l[(size_t)il * T_NL + t], n = ln[t];
        f16 = t == T_QKVW || t == T_OW || t == T_UPW || t == T_DOWNW;
      }
      int ty = -1;
      size_t ne = 0, bytes = vmm_bytes(m, name, &ty, &ne);
      const int half_ok = f16 && (ty == 1 || ty == 30 || (i < 2 && ty == 0));
      if (!bytes || ne != n || (f16 ? !half_ok : ty != 0)) {
        snprintf(err, 256, "vitc: %s: %zu elements type %d, want %zu type %d", name, ne, ty, n, f16);
        free(stage), vmm_close(m), vitc_free(c);
        return NULL;
      }
      if (pass == 0) *slot = off;
      else if ((f16 ? stage_half(m, name, ty, n, stage + *slot, err) : vmm_raw(m, name, stage + *slot, err))) {
        free(stage), vmm_close(m), vitc_free(c); return NULL;
      }
      off += al256(n * (f16 ? 2 : 4));
    }
    if (pass == 0) {
      c->wbytes = off;
      stage = (char *)calloc(1, off);
      if (!stage || cudaMalloc(&c->wb, off) != cudaSuccess) {
        snprintf(err, 256, "vitc: weight buffer of %zu bytes failed", off);
        free(stage), vmm_close(m), vitc_free(c);
        return NULL;
      }
    }
  }
  const cudaError_t ce = cudaMemcpy(c->wb, stage, c->wbytes, cudaMemcpyHostToDevice);
  free(stage);
  c->pos = vmm_f32(m, "v.position_embd.weight", E * c->G * c->G, err);
  vmm_close(m);
  if (ce != cudaSuccess || !c->pos) { if (ce != cudaSuccess) snprintf(err, 256, "vitc: weight upload: %s", cudaGetErrorString(ce)); vitc_free(c); return NULL; }
  if (cublasCreate(&c->cb) != CUBLAS_STATUS_SUCCESS) { snprintf(err, 256, "vitc: cublasCreate failed"); vitc_free(c); return NULL; }
  cudaEventCreate(&c->e0), cudaEventCreate(&c->e1);
  c->pos_px = c->pos_py = -1;
  return c;
}

// ggml_interpolate(BILINEAR | ALIGN_CORNERS) of the G x G table to px x py: vit.c / vitm.m's pos_resize, verbatim
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
        o[e] = s00[e] * (1.0f - fd0) * (1.0f - fd1) + s10[e] * fd0 * (1.0f - fd1) + s01[e] * (1.0f - fd0) * fd1 +
               s11[e] * fd0 * fd1;
    }
  }
}

static int ensure(VitcW *c, size_t N, size_t pix, char *err) {
  if (pix > c->pixcap) {
    cudaFree(c->planar), c->planar = NULL, c->pixcap = 0;
    CK(cudaMalloc(&c->planar, 3 * pix * sizeof(float)), "alloc planar");
    c->pixcap = pix;
  }
  if (N <= c->ncap) return 0;
  void **bufs[] = {(void **)&c->x, (void **)&c->cur, (void **)&c->tmp, (void **)&c->tmp2, (void **)&c->qkv, (void **)&c->ao,
                   (void **)&c->up, (void **)&c->qr, (void **)&c->kd, (void **)&c->pos_d, (void **)&c->embd, (void **)&c->rows,
                   (void **)&c->h16, (void **)&c->qh, (void **)&c->kh, (void **)&c->vh, (void **)&c->ppos};
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) cudaFree(*bufs[i]), *bufs[i] = NULL;
  c->ncap = 0, c->pos_px = c->pos_py = -1;
  const size_t E = c->E, wide = N * (size_t)(c->F > 4 * c->E ? c->F : 4 * c->E), npad = (N + 63) / 64 * 64;
  const size_t sz[] = {N * E * 4, N * E * 4, (wide > 3 * N * E ? wide : 3 * N * E) * 4, N * E * 4, N * 3 * E * 4, N * E * 4,
                       wide * 4, N * E * 4, N * E * 4, N * E * 4, N / 4 * c->PD * 4, N * 768 * 2,
                       wide * 2, c->H * npad * c->DP * 2, c->H * npad * c->DP * 2, c->H * npad * c->DP * 2, 2 * N * sizeof(int)};
  c->actbytes = 3 * pix * sizeof(float);
  for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
    CK(cudaMalloc(bufs[i], sz[i]), "alloc activations");
    c->actbytes += sz[i];
  }
  c->ncap = N;
  return 0;
}

// Y[n][o] = X16[n][k] . W[o][k]^T, F32 accumulate (column-major view: Y^T = W . X^T)
static int gemm_h(VitcW *c, cublasHandle_t cb, size_t woff, const __half *X, float *Y, int n, int k, int o, char *err) {
  const float one = 1.0f, zero = 0.0f;
  const cublasStatus_t s = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, o, n, k, &one, c->wb + woff, CUDA_R_16F, k, X,
                                        CUDA_R_16F, k, &zero, Y, CUDA_R_32F, o, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (s != CUBLAS_STATUS_SUCCESS) { snprintf(err, 256, "vitc: cublasGemmEx %dx%dx%d: status %d", n, k, o, (int)s); return -1; }
  return 0;
}
static int gemm(VitcW *c, size_t woff, const __half *X, float *Y, int n, int k, int o, char *err) {
  return gemm_h(c, c->cb, woff, X, Y, n, k, o, err);
}
// (Stage 2, --gemm streams) side streams + handles, created once (non-blocking streams, joined by events)
static int streams_init(VitcW *c, char *err) {
  if (c->nss) return 0;
  if (cudaEventCreateWithFlags(&c->ef, cudaEventDisableTiming) != cudaSuccess) { snprintf(err, 256, "vitc: stream event"); return -1; }
  for (int i = 0; i < 8; i++) {
    if (cudaStreamCreateWithFlags(&c->ss[i], cudaStreamNonBlocking) != cudaSuccess || cublasCreate(&c->cbs[i]) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(c->cbs[i], c->ss[i]) != CUBLAS_STATUS_SUCCESS || cudaEventCreateWithFlags(&c->ej[i], cudaEventDisableTiming) != cudaSuccess) {
      snprintf(err, 256, "vitc: side stream %d setup failed", i); return -1;
    }
    c->nss = i + 1;
  }
  return 0;
}

static int wants(const VitOpt *o, const char *name) { return o && o->dump && (!o->want || o->want(o->dump_ud, name)); }
static unsigned nb(long n) { return (unsigned)((n + 255) / 256); }

extern "C" int vitc_encode(VitcW *c, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err) {
  const int E = c->E, H = c->H, D = c->D, F = c->F, P = c->P, M = c->M, PD = c->PD;
  if (o && !o->f16act) { snprintf(err, 256, "vitc: only the registered F16-staged mode"); return -1; }
  if (nx % (P * M) || ny % (P * M)) { snprintf(err, 256, "image %dx%d not a multiple of %d", nx, ny, P * M); return -1; }
  const int px = nx / P, py = ny / P;
  const int N = px * py, T = N / 4, npad = (N + 63) / 64 * 64, K = 3 * P * P;
  if (ensure(c, (size_t)N, (size_t)nx * ny, err)) return -1;
  if (c->pos_px != px || c->pos_py != py) {  // the position rows (merge order) and the rope positions: once per size
    float *pr = (float *)malloc(sizeof(float) * (size_t)N * E), *dst = (float *)malloc(sizeof(float) * (size_t)N * E);
    int *pp = (int *)malloc(sizeof(int) * 2 * (size_t)N);
    if (!pr || !dst || !pp) { snprintf(err, 256, "out of memory"); return -1; }
    pos_resize(c->pos, c->G, E, px, py, pr);
    size_t r = 0;
    for (int y = 0; y < py; y += M)
      for (int x = 0; x < px; x += M)
        for (int dy = 0; dy < M; dy++)
          for (int dx = 0; dx < M; dx++, r++) {
            memcpy(dst + r * E, pr + ((size_t)(y + dy) * px + x + dx) * E, sizeof(float) * E);
            pp[r] = y + dy, pp[N + r] = x + dx;
          }
    CK(cudaMemcpy(c->pos_d, dst, sizeof(float) * (size_t)N * E, cudaMemcpyHostToDevice), "pos upload");
    CK(cudaMemcpy(c->ppos, pp, sizeof(int) * 2 * (size_t)N, cudaMemcpyHostToDevice), "ppos upload");
    free(pr), free(dst), free(pp);
    c->pos_px = px, c->pos_py = py;
  }
  float *dbuf = NULL;
  size_t dcap = 0;
  char nm[64];
#define DUMP(name, buf, ne0, rows)                                                                          \
  do {                                                                                                      \
    if (wants(o, name)) {                                                                                   \
      const size_t ne_ = (size_t)(ne0) * (rows);                                                            \
      if (ne_ > dcap) { free(dbuf); dbuf = (float *)malloc(ne_ * 4); dcap = ne_; }                         \
      CK(cudaMemcpy(dbuf, buf, ne_ * 4, cudaMemcpyDeviceToHost), "dump copy");                            \
      o->dump(o->dump_ud, name, dbuf, (ne0), (int)(rows));                                                  \
    }                                                                                                       \
  } while (0)
#define LDUMP(base, il, buf, ne0, rows) do { snprintf(nm, sizeof nm, "%s-%d", base, il); DUMP(nm, buf, ne0, rows); } while (0)
#define G(woff, X, Y, n, k, oo) do { if (gemm(c, woff, X, Y, n, k, oo, err)) return -1; } while (0)
#define W32(off) ((const float *)(c->wb + (off)))
#define W16(off) (off)

  cudaEventRecord(c->e0);
  CK(cudaMemcpyAsync(c->planar, planar, sizeof(float) * 3 * (size_t)nx * ny, cudaMemcpyHostToDevice), "planar upload");
  const size_t hsz = (size_t)H * npad * c->DP * 2;
  cudaMemsetAsync(c->qh, 0, hsz), cudaMemsetAsync(c->kh, 0, hsz), cudaMemsetAsync(c->vh, 0, hsz);
  // patch embed
  k_im2col<<<nb((long)N * K), 256>>>(c->planar, c->rows, nx, ny, px, N);
  G(W16(c->o_pe0), c->rows, c->tmp, N, K, E);
  G(W16(c->o_pe1), c->rows, c->tmp2, N, K, E);
  k_embed<<<nb((long)N * E), 256>>>(c->x, wants(o, "patch_bias") ? c->cur : NULL, c->tmp, c->tmp2, W32(c->o_peb), c->pos_d, (long)N * E, E);
  DUMP("patch_bias", c->cur, E, N);
  DUMP("inp_pos_emb", c->x, E, N);

  const float scale = 1.0f / sqrtf((float)D);
  const long NE = (long)N * E;
  for (int il = 0; il < c->L; il++) {
    const size_t *lo = c->o_l + (size_t)il * T_NL;
    k_ln<<<N, 256>>>(c->x, W32(lo[T_LN1W]), W32(lo[T_LN1B]), c->cur, c->h16, E, c->h.eps);
    LDUMP("ln1", il, c->cur, E, N);
    G(W16(lo[T_QKVW]), c->h16, c->tmp, N, E, 3 * E);
    k_epi<<<nb(3 * NE), 256>>>(c->tmp, c->qkv, W32(lo[T_QKVB]), NULL, 3 * NE, 3 * E, EP_BIAS);
    snprintf(nm, sizeof nm, "Qcur_rope-%d", il);
    float *qd = wants(o, nm) ? c->qr : NULL;
    snprintf(nm, sizeof nm, "Kcur_rope-%d", il);
    float *kd = wants(o, nm) ? c->kd : NULL;
    k_rope<<<nb((long)N * H * (D / 2)), 256>>>(c->qkv, c->ppos, c->qh, c->kh, c->vh, qd, kd, N, E, H, D, c->DP, npad, NULL);
    LDUMP("Qcur_rope", il, c->qr, E, N);
    LDUMP("Kcur_rope", il, c->kd, E, N);
    if (c->DP == 64) k_fa<64><<<dim3(npad / 64, H), 128>>>(c->qh, c->kh, c->vh, c->ao, c->h16, N, npad, E, D, scale);
    else k_fa<80><<<dim3(npad / 64, H), 128>>>(c->qh, c->kh, c->vh, c->ao, c->h16, N, npad, E, D, scale);
    LDUMP("kqv_out", il, c->ao, E, N);
    G(W16(lo[T_OW]), c->h16, c->tmp, N, E, E);
    snprintf(nm, sizeof nm, "attn_out-%d", il);
    if (wants(o, nm)) {
      k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_OB]), NULL, NE, E, EP_BIAS);
      DUMP(nm, c->cur, E, N);
      k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
    } else {
      k_epi<<<nb(NE), 256>>>(c->tmp, c->x, W32(lo[T_OB]), NULL, NE, E, EP_BIAS | EP_RES);
    }
    LDUMP("ffn_inp", il, c->x, E, N);
    k_ln<<<N, 256>>>(c->x, W32(lo[T_LN2W]), W32(lo[T_LN2B]), c->cur, c->h16, E, c->h.eps);
    LDUMP("ffn_inp_normed", il, c->cur, E, N);
    G(W16(lo[T_UPW]), c->h16, c->tmp, N, E, F);
    snprintf(nm, sizeof nm, "ffn_up_b-%d", il);
    if (wants(o, nm)) {
      k_epi<<<nb((long)N * F), 256>>>(c->tmp, c->up, W32(lo[T_UPB]), NULL, (long)N * F, F, EP_BIAS);
      DUMP(nm, c->up, F, N);
      k_epi<<<nb((long)N * F), 256>>>(c->up, c->up, NULL, c->h16, (long)N * F, F, EP_GELU);
    } else {
      k_epi<<<nb((long)N * F), 256>>>(c->tmp, c->up, W32(lo[T_UPB]), c->h16, (long)N * F, F, EP_BIAS | EP_GELU);
    }
    LDUMP("ffn_gelu", il, c->up, F, N);
    G(W16(lo[T_DOWNW]), c->h16, c->tmp, N, F, E);
    snprintf(nm, sizeof nm, "ffn_out-%d", il);
    if (wants(o, nm)) {
      k_epi<<<nb(NE), 256>>>(c->tmp, c->cur, W32(lo[T_DOWNB]), NULL, NE, E, EP_BIAS);
      DUMP(nm, c->cur, E, N);
      k_epi<<<nb(NE), 256>>>(c->cur, c->x, NULL, NULL, NE, E, EP_RES);
    } else {
      k_epi<<<nb(NE), 256>>>(c->tmp, c->x, W32(lo[T_DOWNB]), NULL, NE, E, EP_BIAS | EP_RES);
    }
    LDUMP("layer_out", il, c->x, E, N);
  }
  k_ln<<<N, 256>>>(c->x, W32(c->o_postw), W32(c->o_postb), c->cur, c->h16, E, c->h.eps);
  LDUMP("norm_b", c->L, c->cur, E, N);
  // merger: 4 consecutive (merge-ordered) patches = one 4E row of h16 -> mm.0 + b -> GELU -> mm.2 + b
  const long T4 = (long)T * 4 * E;
  G(W16(c->o_mm0w), c->h16, c->tmp, T, 4 * E, 4 * E);
  if (wants(o, "ffn_up_b")) {
    k_epi<<<nb(T4), 256>>>(c->tmp, c->up, W32(c->o_mm0b), NULL, T4, 4 * E, EP_BIAS);
    DUMP("ffn_up_b", c->up, 4 * E, T);
    k_epi<<<nb(T4), 256>>>(c->up, c->up, NULL, c->h16, T4, 4 * E, EP_GELU);
  } else {
    k_epi<<<nb(T4), 256>>>(c->tmp, c->up, W32(c->o_mm0b), c->h16, T4, 4 * E, EP_BIAS | EP_GELU);
  }
  DUMP("ffn_gelu", c->up, 4 * E, T);
  G(W16(c->o_mm2w), c->h16, c->tmp, T, 4 * E, PD);
  k_epi<<<nb((long)T * PD), 256>>>(c->tmp, c->embd, W32(c->o_mm2b), NULL, (long)T * PD, PD, EP_BIAS);
  CK(cudaMemcpyAsync(embd, c->embd, sizeof(float) * (size_t)T * PD, cudaMemcpyDeviceToHost), "embd copy");
  cudaEventRecord(c->e1);
  CK(cudaEventSynchronize(c->e1), "encode");
  CK(cudaGetLastError(), "kernel launch");
  float ms = 0;
  cudaEventElapsedTime(&ms, c->e0, c->e1);
  c->gpu_ms = ms;
  free(dbuf);
  if (wants(o, "embd")) o->dump(o->dump_ud, "embd", embd, PD, T);
#undef DUMP
#undef LDUMP
#undef G
  return 0;
}

// ---------------------------------------------------------------- batch (CHAIN-2-1-3 Stage 2)
// n images in one pass: every row-local op (im2col per image, embed, LN, epilogues, rope, merger rows) runs once over
// the concatenated rows; attention is block-diagonal (k_fa_seg, image z attends to its own rows only, its planes padded
// to 64 as in a single encode). gemm_seg = 1 issues each GEMM per image with the single encode's shape (n_i rows), so
// cuBLAS sees the same problem as vitc_encode; 2 = the same per-image GEMMs on up to 8 side streams (one cuBLAS handle
// each), so small images' GEMMs overlap; 0 issues one GEMM over all rows. Same F16 staging; no dumps.
extern "C" int vitc_encode_batch(VitcW *c, int nimg, const float *const *planar, const int *nxs, const int *nys,
                                 int gemm_seg, float *const *embd, char *err) {
  const int E = c->E, H = c->H, D = c->D, F = c->F, P = c->P, M = c->M, PD = c->PD, K = 3 * P * P;
  if (nimg < 1) { snprintf(err, 256, "vitc batch: no images"); return -1; }
  int *row0 = (int *)malloc(sizeof(int) * 3 * (nimg + 1)), *pr0 = row0 + nimg + 1, *sn = pr0 + nimg + 1;
  size_t pix = 0;
  int N = 0, NP = 0, maxpad = 0;
  for (int i = 0; i < nimg; i++) {
    if (nxs[i] % (P * M) || nys[i] % (P * M)) { snprintf(err, 256, "image %d: %dx%d not a multiple of %d", i, nxs[i], nys[i], P * M); free(row0); return -1; }
    const int n = (nxs[i] / P) * (nys[i] / P), np = (n + 63) / 64 * 64;
    row0[i] = N, pr0[i] = NP, sn[i] = n;
    N += n, NP += np, pix += (size_t)nxs[i] * nys[i];
    if (np > maxpad) maxpad = np;
  }
  row0[nimg] = N, pr0[nimg] = NP;
  // activation buffers sized for max(N, NP) rows: the [H][npad][DP] planes need NP padded rows (npad of NP = NP)
  if (ensure(c, (size_t)(NP > N ? NP : N), pix, err)) { free(row0); return -1; }
  const int npad = (int)(((size_t)(NP > N ? NP : N) + 63) / 64 * 64);
  float *pr = (float *)malloc(sizeof(float) * (size_t)N * E), *pos = (float *)malloc(sizeof(float) * (size_t)N * E);
  int *pp = (int *)malloc(sizeof(int) * 3 * (size_t)N), *prow = pp + 2 * (size_t)N;
  if (!pr || !pos || !pp) { snprintf(err, 256, "out of memory"); free(row0), free(pr), free(pos), free(pp); return -1; }
  for (int i = 0; i < nimg; i++) {  // = vitc_encode's per-size block, per image, at its row offset
    const int px = nxs[i] / P, py = nys[i] / P, n = sn[i];
    pos_resize(c->pos, c->G, E, px, py, pr);
    size_t r = 0;
    for (int y = 0; y < py; y += M)
      for (int x = 0; x < px; x += M)
        for (int dy = 0; dy < M; dy++)
          for (int dx = 0; dx < M; dx++, r++) {
            const size_t g = (size_t)row0[i] + r;
            memcpy(pos + g * E, pr + ((size_t)(y + dy) * px + x + dx) * E, sizeof(float) * E);
            pp[g] = y + dy, pp[N + g] = x + dx, prow[g] = pr0[i] + (int)r;
          }
    (void)n;
  }
  free(pr);
  c->pos_px = c->pos_py = -1;  // pos_d / ppos now hold the batch's rows: the single path re-uploads its own
  int *segd = NULL, *prowd = NULL;
#define BCK(call, what) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { snprintf(err, 256, "vitc batch: %s: %s", what, cudaGetErrorString(e_)); goto fail; } } while (0)
#define BG(woff, X, Y, k, oo, rowsof)                                                                         \
  do {                                                                                                        \
    if (gemm_seg == 2) { /* fork: side streams wait for the default stream; GEMM i on stream i % 8; join */   \
      const int ns_ = nimg < 8 ? nimg : 8;                                                                    \
      cudaEventRecord(c->ef, 0);                                                                              \
      for (int s_ = 0; s_ < ns_; s_++) cudaStreamWaitEvent(c->ss[s_], c->ef, 0);                              \
      for (int i_ = 0; i_ < nimg; i_++)                                                                       \
        if (gemm_h(c, c->cbs[i_ % 8], woff, (X) + (size_t)rowsof(i_, row0) * (k), (Y) + (size_t)rowsof(i_, row0) * (oo), rowsof(i_ + 1, row0) - rowsof(i_, row0), k, oo, err)) goto fail; \
      for (int s_ = 0; s_ < ns_; s_++) cudaEventRecord(c->ej[s_], c->ss[s_]), cudaStreamWaitEvent(0, c->ej[s_], 0); \
    } else if (gemm_seg) {                                                                                    \
      for (int i_ = 0; i_ < nimg; i_++)                                                                       \
        if (gemm(c, woff, (X) + (size_t)rowsof(i_, row0) * (k), (Y) + (size_t)rowsof(i_, row0) * (oo), rowsof(i_ + 1, row0) - rowsof(i_, row0), k, oo, err)) goto fail; \
    } else if (gemm(c, woff, X, Y, rowsof(nimg, row0), k, oo, err)) goto fail;                               \
  } while (0)
#define RP(i, a) ((a)[i])
#define RT(i, a) ((a)[i] / 4)
#define W32(off) ((const float *)(c->wb + (off)))
  if (gemm_seg == 2 && streams_init(c, err)) goto fail;
  {
    BCK(cudaMalloc(&segd, sizeof(int) * 3 * (nimg + 1)), "alloc seg");
    BCK(cudaMalloc(&prowd, sizeof(int) * (size_t)N), "alloc prow");
    cudaEventRecord(c->e0);
    BCK(cudaMemcpyAsync(segd, row0, sizeof(int) * 3 * (nimg + 1), cudaMemcpyHostToDevice), "seg upload");
    BCK(cudaMemcpyAsync(prowd, prow, sizeof(int) * (size_t)N, cudaMemcpyHostToDevice), "prow upload");
    BCK(cudaMemcpy(c->pos_d, pos, sizeof(float) * (size_t)N * E, cudaMemcpyHostToDevice), "pos upload");
    BCK(cudaMemcpy(c->ppos, pp, sizeof(int) * 2 * (size_t)N, cudaMemcpyHostToDevice), "ppos upload");
    const size_t hsz = (size_t)H * npad * c->DP * 2;
    cudaMemsetAsync(c->qh, 0, hsz), cudaMemsetAsync(c->kh, 0, hsz), cudaMemsetAsync(c->vh, 0, hsz);
    size_t po = 0;
    for (int i = 0; i < nimg; i++) {
      const size_t np_ = (size_t)nxs[i] * nys[i];
      BCK(cudaMemcpyAsync(c->planar + 3 * po, planar[i], sizeof(float) * 3 * np_, cudaMemcpyHostToDevice), "planar upload");
      k_im2col<<<nb((long)sn[i] * K), 256>>>(c->planar + 3 * po, c->rows + (size_t)row0[i] * K, nxs[i], nys[i], nxs[i] / P, sn[i]);
      po += np_;
    }
    BG(c->o_pe0, c->rows, c->tmp, K, E, RP);
    BG(c->o_pe1, c->rows, c->tmp2, K, E, RP);
    k_embed<<<nb((long)N * E), 256>>>(c->x, NULL, c->tmp, c->tmp2, W32(c->o_peb), c->pos_d, (long)N * E, E);
    const float scale = 1.0f / sqrtf((float)D);
    const long NE = (long)N * E;
    const dim3 fag(maxpad / 64, H, nimg);
    for (int il = 0; il < c->L; il++) {
      const size_t *lo = c->o_l + (size_t)il * T_NL;
      k_ln<<<N, 256>>>(c->x, W32(lo[T_LN1W]), W32(lo[T_LN1B]), c->cur, c->h16, E, c->h.eps);
      BG(lo[T_QKVW], c->h16, c->tmp, E, 3 * E, RP);
      k_epi<<<nb(3 * NE), 256>>>(c->tmp, c->qkv, W32(lo[T_QKVB]), NULL, 3 * NE, 3 * E, EP_BIAS);
      k_rope<<<nb((long)N * H * (D / 2)), 256>>>(c->qkv, c->ppos, c->qh, c->kh, c->vh, NULL, NULL, N, E, H, D, c->DP, npad, prowd);
      if (c->DP == 64) k_fa_seg<64><<<fag, 128>>>(c->qh, c->kh, c->vh, c->ao, c->h16, segd, segd + nimg + 1, segd + 2 * (nimg + 1), npad, E, D, scale);
      else k_fa_seg<80><<<fag, 128>>>(c->qh, c->kh, c->vh, c->ao, c->h16, segd, segd + nimg + 1, segd + 2 * (nimg + 1), npad, E, D, scale);
      BG(lo[T_OW], c->h16, c->tmp, E, E, RP);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->x, W32(lo[T_OB]), NULL, NE, E, EP_BIAS | EP_RES);
      k_ln<<<N, 256>>>(c->x, W32(lo[T_LN2W]), W32(lo[T_LN2B]), c->cur, c->h16, E, c->h.eps);
      BG(lo[T_UPW], c->h16, c->tmp, E, F, RP);
      k_epi<<<nb((long)N * F), 256>>>(c->tmp, c->up, W32(lo[T_UPB]), c->h16, (long)N * F, F, EP_BIAS | EP_GELU);
      BG(lo[T_DOWNW], c->h16, c->tmp, F, E, RP);
      k_epi<<<nb(NE), 256>>>(c->tmp, c->x, W32(lo[T_DOWNB]), NULL, NE, E, EP_BIAS | EP_RES);
    }
    k_ln<<<N, 256>>>(c->x, W32(c->o_postw), W32(c->o_postb), c->cur, c->h16, E, c->h.eps);
    const int T = N / 4;
    const long T4 = (long)T * 4 * E;
    BG(c->o_mm0w, c->h16, c->tmp, 4 * E, 4 * E, RT);
    k_epi<<<nb(T4), 256>>>(c->tmp, c->up, W32(c->o_mm0b), c->h16, T4, 4 * E, EP_BIAS | EP_GELU);
    BG(c->o_mm2w, c->h16, c->tmp, 4 * E, PD, RT);
    k_epi<<<nb((long)T * PD), 256>>>(c->tmp, c->embd, W32(c->o_mm2b), NULL, (long)T * PD, PD, EP_BIAS);
    for (int i = 0; i < nimg; i++)
      BCK(cudaMemcpyAsync(embd[i], c->embd + (size_t)(row0[i] / 4) * PD, sizeof(float) * (size_t)(sn[i] / 4) * PD, cudaMemcpyDeviceToHost), "embd copy");
    cudaEventRecord(c->e1);
    BCK(cudaEventSynchronize(c->e1), "encode");
    BCK(cudaGetLastError(), "kernel launch");
    float ms = 0;
    cudaEventElapsedTime(&ms, c->e0, c->e1);
    c->gpu_ms = ms;
  }
  cudaFree(segd), cudaFree(prowd), free(row0), free(pos), free(pp);
  return 0;
fail:
  cudaDeviceSynchronize();
  cudaFree(segd), cudaFree(prowd), free(row0), free(pos), free(pp);
  return -1;
#undef BCK
#undef BG
#undef RP
#undef RT
#undef W32
}
extern "C" void vitc_mem_info(size_t *fr, size_t *tot) { cudaMemGetInfo(fr, tot); }
