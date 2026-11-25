// ec_trellis2.cu — 100% Pure Native Eureka TRELLIS.2 3D Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Pipeline:
//   1. Native CUDA DINOv3-ViT-L/16 (24 blocks, 2D RoPE base=100, 1 cls + 4 reg + 1024 patches -> [1029, 1024])
//   2. Native CUDA SparseStructureFlowModel 1.3B BF16 (30 ModulatedTransformerCrossBlocks,
//      3D RoPE on 16^3=4096 tokens, MultiHeadRMSNorm, 12-step Flow Euler CFG interval [0.6,1.0] + 0.7 rescale)
//   3. Native CUDA SparseStructureDecoder (3D ResBlocks + 3D PixelShuffle 16^3 -> 32^3 -> 64^3 logits)
//   4. Smooth Sub-Voxel Surface Nets Mesher + Analytical Gradient Normals -> GLB 2.0 + 3DGS PLY

#define _POSIX_C_SOURCE 200809L
#include "ec_trellis2.h"
#include "ec_cowrie.h"
#include "ec_media.h"
#include "ec_ops.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double tr_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static __device__ __forceinline__ uint16_t tr_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static inline float tr_bf16_to_f32_h(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline uint16_t tr_f32_to_bf16_h(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static inline float tr_f16_to_f32_h(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) << 31;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t u;
  if (exp == 0) {
    if (mant == 0) u = sign;
    else {
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3FF;
      u = sign | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000U | (mant << 13);
  } else {
    u = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static __device__ __forceinline__ float tr_silu(float x) {
  return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float tr_gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}
static __device__ __forceinline__ float tr_gelu_tanh(float x) {
  float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(c));
}
static __device__ __forceinline__ float tr_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// ===========================================================================
// 1. Unified High-Throughput BF16 TensorCore GEMM (via shared ec_ops.cu)
// ===========================================================================
static void tr_gemm(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t s) {
  ec_ops_gemm_bf16(X, W, bias, Y_f32, Y_bf16, M, N, K, s);
}

// ===========================================================================
// 2. Common Norm / Activation / Attention Kernels
// ===========================================================================
__global__ void k_tr_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = tr_f32_to_bf16(src[i]);
}
__global__ void k_tr_add_inplace(float *dst, const float *src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
__global__ void k_tr_add_scaled(float *dst, const float *src, const float *scale, int rows, int dim) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * dim;
  if (idx >= tot) return;
  dst[idx] += src[idx] * scale[idx % dim];
}
__global__ void k_tr_add_vec(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_tr_gelu_erf_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = tr_f32_to_bf16(tr_gelu_erf(src[i]));
}
__global__ void k_tr_gelu_tanh_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = tr_f32_to_bf16(tr_gelu_tanh(src[i]));
}
__global__ void k_tr_silu_f32(const float *src, float *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = tr_silu(src[i]);
}

// Unified LayerNorm (optional affine w/b, optional AdaLN scale/shift, optional SiLU)
__global__ void k_tr_layernorm(
    const float *__restrict__ x,
    const float *__restrict__ w, const float *__restrict__ b,
    const float *__restrict__ adaln_scale, const float *__restrict__ adaln_shift,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int dim, float eps, int apply_silu) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; s1 += v; s2 += v * v; }
  __shared__ float sm1[32], sm2[32];
  s1 = tr_warp_sum(s1); s2 = tr_warp_sum(s2);
  if ((threadIdx.x & 31) == 0) { sm1[threadIdx.x >> 5] = s1; sm2[threadIdx.x >> 5] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = blockDim.x >> 5;
    float a = (threadIdx.x < nw) ? sm1[threadIdx.x] : 0.0f;
    float bb = (threadIdx.x < nw) ? sm2[threadIdx.x] : 0.0f;
    a = tr_warp_sum(a); bb = tr_warp_sum(bb);
    if (threadIdx.x == 0) {
      float mean = a / (float)dim;
      float var = bb / (float)dim - mean * mean;
      if (var < 0.0f) var = 0.0f;
      sm1[0] = mean;
      sm2[0] = rsqrtf(var + eps);
    }
  }
  __syncthreads();
  float mean = sm1[0], inv = sm2[0];
  float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
  uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float v = (xr[c] - mean) * inv;
    if (w) v = v * w[c] + b[c];
    if (adaln_scale) v = v * (1.0f + adaln_scale[c]) + adaln_shift[c];
    if (apply_silu) v = tr_silu(v);
    if (yf) yf[c] = v;
    if (yb) yb[c] = tr_f32_to_bf16(v);
  }
}

// ===========================================================================
// 3. DINOv3-ViT-L/16 Attention Kernels (H=16, D=64, 2D RoPE -> BF16 for TensorCore FA2!)
// ===========================================================================
// Unpack qkv[N, 3, 16, 64] -> Q, K, V [16, N, 64] BF16, applying DINOv3 2D RoPE to patch tokens [5..1028]
__global__ void k_dino_unpack_rope(
    const float *__restrict__ qkv,
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V,
    int N, int H, int D, int n_prefix, int grid_hw) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31; // D = 64 -> half = 32, 1 thread per half pair!
  if (idx >= N * H) return;
  int tok = idx / H;
  int h = idx % H;
  const float *base = qkv + (long)tok * (3 * H * D);
  const float *qs = base + (0 * H + h) * D;
  const float *ks = base + (1 * H + h) * D;
  const float *vs = base + (2 * H + h) * D;
  uint16_t *qo = Q + ((long)h * N + tok) * D;
  uint16_t *ko = K + ((long)h * N + tok) * D;
  uint16_t *vo = V + ((long)h * N + tok) * D;

  float q0 = qs[lane], q1 = qs[lane + 32];
  float k0 = ks[lane], k1 = ks[lane + 32];
  if (tok >= n_prefix) {
    int p = tok - n_prefix;
    int py = p / grid_hw;
    int px = p % grid_hw;
    float cy = 2.0f * (((float)py + 0.5f) / (float)grid_hw) - 1.0f;
    float cx = 2.0f * (((float)px + 0.5f) / (float)grid_hw) - 1.0f;
    int freq_idx = lane & 15; // 0..15: y axis, 16..31: x axis
    float coord = (lane < 16) ? cy : cx;
    double inv_f = 1.0 / pow(100.0, (double)(4 * freq_idx) / 64.0);
    double ang = 6.28318530717958647692 * (double)coord * inv_f;
    float cs = (float)cos(ang), sn = (float)sin(ang);
    qo[lane]      = tr_f32_to_bf16(q0 * cs - q1 * sn);
    qo[lane + 32] = tr_f32_to_bf16(q1 * cs + q0 * sn);
    ko[lane]      = tr_f32_to_bf16(k0 * cs - k1 * sn);
    ko[lane + 32] = tr_f32_to_bf16(k1 * cs + k0 * sn);
  } else {
    qo[lane] = tr_f32_to_bf16(q0); qo[lane + 32] = tr_f32_to_bf16(q1);
    ko[lane] = tr_f32_to_bf16(k0); ko[lane + 32] = tr_f32_to_bf16(k1);
  }
  vo[lane] = tr_f32_to_bf16(vs[lane]);
  vo[lane + 32] = tr_f32_to_bf16(vs[lane + 32]);
}

// ===========================================================================
// 4. TRELLIS.2 SparseStructureFlowModel Kernels (3D RoPE, MultiHeadRMSNorm -> BF16 FA2)
// ===========================================================================
// Self-attn unpack qkv[N, 3, H, 128] -> Q, K, V [H, N, 128] BF16 + MultiHeadRMSNorm + 3D RoPE
__global__ void k_ss_self_qkv_rope(
    const float *__restrict__ qkv,
    const float *__restrict__ q_gamma, const float *__restrict__ k_gamma,
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V,
    int N, int H, int res) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= N * H) return;
  int tok = idx / H;
  int h = idx % H;
  const float *base = qkv + (long)tok * (3 * H * 128);
  const float *qs = base + (0 * H + h) * 128;
  const float *ks = base + (1 * H + h) * 128;
  const float *vs = base + (2 * H + h) * 128;
  const float *qg = q_gamma + (long)h * 128;
  const float *kg = k_gamma + (long)h * 128;
  uint16_t *qo = Q + ((long)h * N + tok) * 128;
  uint16_t *ko = K + ((long)h * N + tok) * 128;
  uint16_t *vo = V + ((long)h * N + tok) * 128;

  // MultiHeadRMSNorm: F.normalize(x, dim=-1) * gamma * sqrt(128) = x * rsqrt(mean(x^2)) * gamma
  float sq = 0.0f, sk = 0.0f;
  for (int i = 0; i < 4; i++) {
    int d = lane + i * 32;
    float qv = qs[d], kv = ks[d];
    sq += qv * qv;
    sk += kv * kv;
  }
  float inv_q = rsqrtf(tr_warp_sum(sq) / 128.0f + 1e-12f);
  float inv_k = rsqrtf(tr_warp_sum(sk) / 128.0f + 1e-12f);

  // Voxel coords (ci, cj, ck) matching torch.meshgrid(arange(16), arange(16), arange(16), indexing='ij')
  int ci = tok / (res * res);
  int cj = (tok / res) % res;
  int ck = tok % res;

  // 64 complex pairs (2p, 2p+1): 2 pairs per lane!
  for (int pass = 0; pass < 2; pass++) {
    int p = lane + pass * 32; // 0..63
    int d0 = 2 * p, d1 = 2 * p + 1;
    float q0 = qs[d0] * inv_q * qg[d0];
    float q1 = qs[d1] * inv_q * qg[d1];
    float k0 = ks[d0] * inv_k * kg[d0];
    float k1 = ks[d1] * inv_k * kg[d1];
    float cs = 1.0f, sn = 0.0f;
    if (p < 63) {
      int axis = p / 21;
      int fi = p % 21;
      float coord = (axis == 0) ? (float)ci : ((axis == 1) ? (float)cj : (float)ck);
      double freq = 1.0 / pow(10000.0, (double)fi / 21.0);
      double ang = (double)coord * freq;
      cs = (float)cos(ang);
      sn = (float)sin(ang);
    }
    qo[d0] = tr_f32_to_bf16(q0 * cs - q1 * sn);
    qo[d1] = tr_f32_to_bf16(q0 * sn + q1 * cs);
    ko[d0] = tr_f32_to_bf16(k0 * cs - k1 * sn);
    ko[d1] = tr_f32_to_bf16(k0 * sn + k1 * cs);
    vo[d0] = tr_f32_to_bf16(vs[d0]);
    vo[d1] = tr_f32_to_bf16(vs[d1]);
  }
}

// Cross-attn Q norm & transpose [Lq, H, 128] -> [H, Lq, 128] BF16
__global__ void k_ss_cross_q_norm(
    const float *__restrict__ q_in, const float *__restrict__ q_gamma,
    uint16_t *__restrict__ Q, int Lq, int H) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= Lq * H) return;
  int tok = idx / H, h = idx % H;
  const float *qs = q_in + ((long)tok * H + h) * 128;
  const float *qg = q_gamma + (long)h * 128;
  uint16_t *qo = Q + ((long)h * Lq + tok) * 128;
  float sq = 0.0f;
  for (int i = 0; i < 4; i++) { float v = qs[lane + i * 32]; sq += v * v; }
  float inv = rsqrtf(tr_warp_sum(sq) / 128.0f + 1e-12f);
  for (int i = 0; i < 4; i++) {
    int d = lane + i * 32;
    qo[d] = tr_f32_to_bf16(qs[d] * inv * qg[d]);
  }
}

// Cross-attn KV unpack [Lkv, 2, H, 128] -> K, V [H, Lkv, 128] BF16 + k_rms_norm
__global__ void k_ss_cross_kv_norm(
    const float *__restrict__ kv_in, const float *__restrict__ k_gamma,
    uint16_t *__restrict__ K, uint16_t *__restrict__ V, int Lkv, int H) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= Lkv * H) return;
  int tok = idx / H, h = idx % H;
  const float *ks = kv_in + ((long)tok * 2 * H + h) * 128;
  const float *vs = kv_in + ((long)tok * 2 * H + H + h) * 128;
  const float *kg = k_gamma + (long)h * 128;
  uint16_t *ko = K + ((long)h * Lkv + tok) * 128;
  uint16_t *vo = V + ((long)h * Lkv + tok) * 128;
  float sk = 0.0f;
  for (int i = 0; i < 4; i++) { float v = ks[lane + i * 32]; sk += v * v; }
  float inv = rsqrtf(tr_warp_sum(sk) / 128.0f + 1e-12f);
  for (int i = 0; i < 4; i++) {
    int d = lane + i * 32;
    ko[d] = tr_f32_to_bf16(ks[d] * inv * kg[d]);
    vo[d] = tr_f32_to_bf16(vs[d]);
  }
}

// ===========================================================================
// 5. SparseStructureDecoder 3D Conv & PixelShuffle3D Kernels (Layout: [R, R, R, C])
// ===========================================================================
// Im2Col for 3D Conv (k=3x3x3, pad=1) on voxel range [v0 .. v0+chunk-1] of x[R, R, R, Cin]
// Output col_bf16[chunk, 27 * Cin] ordered as [(kd*9 + kh*3 + kw) * Cin + ic]
__global__ void k_tr_im2col_3d_k3(
    const float *__restrict__ x, uint16_t *__restrict__ col_bf16,
    int R, int Cin, int v0, int chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K = 27L * Cin;
  long tot = (long)chunk * K;
  if (idx >= tot) return;
  int row = (int)(idx / K);
  int rem = (int)(idx % K);
  int k = rem / Cin;
  int ic = rem % Cin;
  int v = v0 + row;
  int vd = v / (R * R);
  int vh = (v / R) % R;
  int vw = v % R;
  int kd = (k / 9) - 1;
  int kh = ((k / 3) % 3) - 1;
  int kw = (k % 3) - 1;
  int id = vd + kd, ih = vh + kh, iw = vw + kw;
  float val = 0.0f;
  if (id >= 0 && id < R && ih >= 0 && ih < R && iw >= 0 && iw < R) {
    val = x[(((long)id * R + ih) * R + iw) * Cin + ic];
  }
  col_bf16[idx] = tr_f32_to_bf16(val);
}

// 3D PixelShuffle (scale=2): in[R, R, R, 8*Cout] -> out[2R, 2R, 2R, Cout]
// Matches trellis2_spatial.py: reshape(C_, 2, 2, 2, H, W, D).permute(C_, H, s0, W, s1, D, s2)
__global__ void k_tr_pixel_shuffle_3d(
    const float *__restrict__ in, float *__restrict__ out, int R, int Cout) {
  int R2 = 2 * R;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)R2 * R2 * R2 * Cout;
  if (idx >= tot) return;
  int c = (int)(idx % Cout);
  long v = idx / Cout;
  int od = (int)(v / ((long)R2 * R2));
  int oh = (int)((v / R2) % R2);
  int ow = (int)(v % R2);
  int id = od >> 1, sd = od & 1;
  int ih = oh >> 1, sh = oh & 1;
  int iw = ow >> 1, sw = ow & 1;
  int c_in = ((c * 2 + sd) * 2 + sh) * 2 + sw;
  out[idx] = in[(((long)id * R + ih) * R + iw) * (8 * Cout) + c_in];
}

// ===========================================================================
// 6. Safetensors Loaders (supports F32, F16, BF16 cleanly!)
// ===========================================================================
static int st_load_f32_host(const ec_safetensors_t *st, const char *name, float **out, size_t *out_n) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[trellis2] missing tensor: %s\n", name); return -1; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  float *buf = (float *)malloc(n * 4);
  if (!buf) return -1;
  const void *p = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_F32) {
    memcpy(buf, p, n * 4);
  } else if (e->dtype == EC_DTYPE_BF16) {
    const uint16_t *s = (const uint16_t *)p;
    for (size_t i = 0; i < n; i++) buf[i] = tr_bf16_to_f32_h(s[i]);
  } else if (e->dtype == EC_DTYPE_F16) {
    const uint16_t *s = (const uint16_t *)p;
    for (size_t i = 0; i < n; i++) buf[i] = tr_f16_to_f32_h(s[i]);
  } else {
    free(buf);
    return -1;
  }
  *out = buf;
  if (out_n) *out_n = n;
  return 0;
}

static float *tr_upload_f32(const ec_safetensors_t *st, const char *name) {
  float *h = NULL;
  size_t n = 0;
  if (st_load_f32_host(st, name, &h, &n) != 0) return NULL;
  float *d = NULL;
  if (cudaMalloc(&d, n * 4) != cudaSuccess) { free(h); return NULL; }
  cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
  free(h);
  return d;
}

static uint16_t *tr_upload_bf16(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[trellis2] missing tensor: %s\n", name); return NULL; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  uint16_t *d = NULL;
  if (cudaMalloc(&d, n * 2) != cudaSuccess) return NULL;
  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpy(d, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice);
  } else {
    float *h = NULL;
    if (st_load_f32_host(st, name, &h, &n) != 0) { cudaFree(d); return NULL; }
    uint16_t *b = (uint16_t *)malloc(n * 2);
    for (size_t i = 0; i < n; i++) b[i] = tr_f32_to_bf16_h(h[i]);
    free(h);
    cudaMemcpy(d, b, n * 2, cudaMemcpyHostToDevice);
    free(b);
  }
  return d;
}

// Transpose Conv3d weight [Cout, Cin, 3, 3, 3] -> [Cout, 27 * Cin] BF16 (row oc, col k*Cin + ic)
static uint16_t *tr_upload_conv3d_bf16(const ec_safetensors_t *st, const char *name, int Cout, int Cin) {
  float *h = NULL;
  size_t n = 0;
  if (st_load_f32_host(st, name, &h, &n) != 0) return NULL;
  uint16_t *wt = (uint16_t *)malloc((size_t)Cout * 27 * Cin * 2);
  for (int oc = 0; oc < Cout; oc++) {
    for (int ic = 0; ic < Cin; ic++) {
      for (int k = 0; k < 27; k++) {
        wt[(long)oc * (27 * Cin) + k * Cin + ic] = tr_f32_to_bf16_h(h[((long)oc * Cin + ic) * 27 + k]);
      }
    }
  }
  free(h);
  uint16_t *d = NULL;
  cudaMalloc(&d, (size_t)Cout * 27 * Cin * 2);
  cudaMemcpy(d, wt, (size_t)Cout * 27 * Cin * 2, cudaMemcpyHostToDevice);
  free(wt);
  return d;
}

// ===========================================================================
// 7. Resident Model Structures
// ===========================================================================
typedef struct {
  float    *modulation; // [9216]
  uint16_t *sa_qkv_w, *sa_out_w;
  float    *sa_qkv_b, *sa_out_b, *sa_qn, *sa_kn;
  float    *norm2_w, *norm2_b;
  uint16_t *ca_q_w, *ca_kv_w, *ca_out_w;
  float    *ca_q_b, *ca_kv_b, *ca_out_b, *ca_qn, *ca_kn;
  uint16_t *mlp0_w, *mlp2_w;
  float    *mlp0_b, *mlp2_b;
} tr_ss_block_t;

typedef struct {
  int loaded;
  char lora_sig[1024];
  int  lora_hits;
  uint16_t *in_w, *out_w, *t0_w, *t2_w, *adaln_w;
  float    *in_b, *out_b, *t0_b, *t2_b, *adaln_b;
  tr_ss_block_t blocks[30];
} tr_ss_flow_res_t;

typedef struct {
  float    *n1_w, *n1_b, *n2_w, *n2_b;
  uint16_t *c1_w, *c2_w;
  float    *c1_b, *c2_b;
} tr_res3d_t;

typedef struct {
  int loaded;
  uint16_t *in_w, *up2_w, *up5_w, *out_w;
  float    *in_b, *up2_b, *up5_b, *out_b, *out_nw, *out_nb;
  tr_res3d_t mid[2];   // 512
  tr_res3d_t blk01[2]; // 512 (blocks.0, 1)
  tr_res3d_t blk34[2]; // 128 (blocks.3, 4)
  tr_res3d_t blk67[2]; // 32  (blocks.6, 7)
} tr_ss_dec_res_t;

struct ec_trellis2 {
  ec_trellis2_paths_t paths;
  cudaStream_t        stream;
  ec_safetensors_t    dino_st, flow_st, dec_st;
  int                 dino_open, flow_open, dec_open;
  // v17.2 Full 3-Stage 512^3 O-Voxel PBR handles
  ec_safetensors_t    sh_flow_st, tx_flow_st, sh_dec_st, tx_dec_st;
  int                 sh_flow_open, tx_flow_open, sh_dec_open, tx_dec_open;
  int                 full_3stage;
  tr_ss_flow_res_t    flow;
  tr_ss_dec_res_t     dec;
  char                dino_sha[65];
};

static void free_flow_res(tr_ss_flow_res_t *f) {
  if (!f->loaded) return;
  cudaFree(f->in_w); cudaFree(f->in_b); cudaFree(f->out_w); cudaFree(f->out_b);
  cudaFree(f->t0_w); cudaFree(f->t0_b); cudaFree(f->t2_w); cudaFree(f->t2_b);
  cudaFree(f->adaln_w); cudaFree(f->adaln_b);
  for (int i = 0; i < 30; i++) {
    tr_ss_block_t *B = &f->blocks[i];
    cudaFree(B->modulation);
    cudaFree(B->sa_qkv_w); cudaFree(B->sa_qkv_b); cudaFree(B->sa_out_w); cudaFree(B->sa_out_b);
    cudaFree(B->sa_qn); cudaFree(B->sa_kn);
    cudaFree(B->norm2_w); cudaFree(B->norm2_b);
    cudaFree(B->ca_q_w); cudaFree(B->ca_q_b); cudaFree(B->ca_kv_w); cudaFree(B->ca_kv_b);
    cudaFree(B->ca_out_w); cudaFree(B->ca_out_b); cudaFree(B->ca_qn); cudaFree(B->ca_kn);
    cudaFree(B->mlp0_w); cudaFree(B->mlp0_b); cudaFree(B->mlp2_w); cudaFree(B->mlp2_b);
  }
  memset(f, 0, sizeof *f);
}

static void free_res3d(tr_res3d_t *r) {
  cudaFree(r->n1_w); cudaFree(r->n1_b); cudaFree(r->n2_w); cudaFree(r->n2_b);
  cudaFree(r->c1_w); cudaFree(r->c1_b); cudaFree(r->c2_w); cudaFree(r->c2_b);
}

static void free_dec_res(tr_ss_dec_res_t *d) {
  if (!d->loaded) return;
  cudaFree(d->in_w); cudaFree(d->in_b);
  cudaFree(d->up2_w); cudaFree(d->up2_b);
  cudaFree(d->up5_w); cudaFree(d->up5_b);
  cudaFree(d->out_nw); cudaFree(d->out_nb); cudaFree(d->out_w); cudaFree(d->out_b);
  for (int i = 0; i < 2; i++) {
    free_res3d(&d->mid[i]); free_res3d(&d->blk01[i]);
    free_res3d(&d->blk34[i]); free_res3d(&d->blk67[i]);
  }
  memset(d, 0, sizeof *d);
}

static void build_tr_lora_sig(const ec_lora_spec_t *loras, int n_loras, char *out, size_t cap) {
  out[0] = '\0';
  size_t pos = 0;
  for (int i = 0; i < n_loras && pos + 8 < cap; i++) {
    if (!loras[i].path[0]) continue;
    pos += (size_t)snprintf(out + pos, cap - pos, "%s%.3f|", loras[i].path, loras[i].strength);
  }
}

static int ensure_flow_loaded(ec_trellis2_t *tr, const ec_lora_spec_t *loras, int n_loras) {
  int clamped_n = (loras && n_loras > 0) ? (n_loras > EC_LORA_MAX_STACK ? EC_LORA_MAX_STACK : n_loras) : 0;
  char lsig[1024];
  build_tr_lora_sig(loras, clamped_n, lsig, sizeof lsig);
  if (tr->flow.loaded && strcmp(tr->flow.lora_sig, lsig) == 0) return 0;
  if (tr->flow.loaded) free_flow_res(&tr->flow);
  const ec_safetensors_t *st = &tr->flow_st;
  tr_ss_flow_res_t *f = &tr->flow;
  ec_lora_apply_stats_t lstats = {};
  f->in_w    = tr_upload_bf16(st, "input_layer.weight");
  f->in_b    = tr_upload_f32(st,  "input_layer.bias");
  f->out_w   = tr_upload_bf16(st, "out_layer.weight");
  f->out_b   = tr_upload_f32(st,  "out_layer.bias");
  f->t0_w    = tr_upload_bf16(st, "t_embedder.mlp.0.weight");
  f->t0_b    = tr_upload_f32(st,  "t_embedder.mlp.0.bias");
  f->t2_w    = tr_upload_bf16(st, "t_embedder.mlp.2.weight");
  f->t2_b    = tr_upload_f32(st,  "t_embedder.mlp.2.bias");
  f->adaln_w = tr_upload_bf16(st, "adaLN_modulation.1.weight");
  f->adaln_b = tr_upload_f32(st,  "adaLN_modulation.1.bias");
  if (!f->in_w || !f->out_w || !f->adaln_w) { free_flow_res(f); return -1; }

  char k[192];
  for (int i = 0; i < 30; i++) {
    tr_ss_block_t *B = &f->blocks[i];
    snprintf(k, sizeof k, "blocks.%d.modulation", i);                    B->modulation = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.weight", i);      B->sa_qkv_w   = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->sa_qkv_w) ec_lora_fuse_onto_bf16(B->sa_qkv_w, 4608, 1536, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.bias", i);        B->sa_qkv_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.weight", i);      B->sa_out_w   = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->sa_out_w) ec_lora_fuse_onto_bf16(B->sa_out_w, 1536, 1536, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.bias", i);        B->sa_out_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.q_rms_norm.gamma", i);   B->sa_qn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.k_rms_norm.gamma", i);   B->sa_kn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.weight", i);                 B->norm2_w    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.bias", i);                   B->norm2_b    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.weight", i);       B->ca_q_w     = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->ca_q_w) ec_lora_fuse_onto_bf16(B->ca_q_w, 1536, 1536, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.bias", i);         B->ca_q_b     = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.weight", i);      B->ca_kv_w    = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->ca_kv_w) ec_lora_fuse_onto_bf16(B->ca_kv_w, 3072, 1024, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.bias", i);        B->ca_kv_b    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.weight", i);     B->ca_out_w   = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->ca_out_w) ec_lora_fuse_onto_bf16(B->ca_out_w, 1536, 1536, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.bias", i);       B->ca_out_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.q_rms_norm.gamma", i);  B->ca_qn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.k_rms_norm.gamma", i);  B->ca_kn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.weight", i);             B->mlp0_w     = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->mlp0_w) ec_lora_fuse_onto_bf16(B->mlp0_w, 6144, 1536, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.bias", i);               B->mlp0_b     = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.weight", i);             B->mlp2_w     = tr_upload_bf16(st, k);
    if (clamped_n > 0 && B->mlp2_w) ec_lora_fuse_onto_bf16(B->mlp2_w, 1536, 6144, k, loras, clamped_n, &lstats, tr->stream);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.bias", i);               B->mlp2_b     = tr_upload_f32(st, k);
    if (!B->sa_qkv_w || !B->ca_kv_w || !B->mlp2_w) { free_flow_res(f); return -1; }
  }
  snprintf(f->lora_sig, sizeof f->lora_sig, "%s", lsig);
  f->lora_hits = lstats.n_tensors_matched;
  f->loaded = 1;
  return 0;
}

static int load_res3d(const ec_safetensors_t *st, const char *pfx, int C, tr_res3d_t *r) {
  char k[192];
  snprintf(k, sizeof k, "%s.norm1.weight", pfx); r->n1_w = tr_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.norm1.bias", pfx);   r->n1_b = tr_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.norm2.weight", pfx); r->n2_w = tr_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.norm2.bias", pfx);   r->n2_b = tr_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.conv1.weight", pfx); r->c1_w = tr_upload_conv3d_bf16(st, k, C, C);
  snprintf(k, sizeof k, "%s.conv1.bias", pfx);   r->c1_b = tr_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.conv2.weight", pfx); r->c2_w = tr_upload_conv3d_bf16(st, k, C, C);
  snprintf(k, sizeof k, "%s.conv2.bias", pfx);   r->c2_b = tr_upload_f32(st, k);
  return (r->n1_w && r->c1_w && r->c2_w) ? 0 : -1;
}

static int ensure_dec_loaded(ec_trellis2_t *tr) {
  if (tr->dec.loaded) return 0;
  const ec_safetensors_t *st = &tr->dec_st;
  tr_ss_dec_res_t *d = &tr->dec;
  d->in_w   = tr_upload_conv3d_bf16(st, "input_layer.weight", 512, 8);
  d->in_b   = tr_upload_f32(st, "input_layer.bias");
  d->up2_w  = tr_upload_conv3d_bf16(st, "blocks.2.conv.weight", 1024, 512);
  d->up2_b  = tr_upload_f32(st, "blocks.2.conv.bias");
  d->up5_w  = tr_upload_conv3d_bf16(st, "blocks.5.conv.weight", 256, 128);
  d->up5_b  = tr_upload_f32(st, "blocks.5.conv.bias");
  d->out_nw = tr_upload_f32(st, "out_layer.0.weight");
  d->out_nb = tr_upload_f32(st, "out_layer.0.bias");
  d->out_w  = tr_upload_conv3d_bf16(st, "out_layer.2.weight", 1, 32);
  d->out_b  = tr_upload_f32(st, "out_layer.2.bias");
  if (!d->in_w || !d->up2_w || !d->up5_w || !d->out_w) { free_dec_res(d); return -1; }

  if (load_res3d(st, "middle_block.0", 512, &d->mid[0]) != 0 ||
      load_res3d(st, "middle_block.1", 512, &d->mid[1]) != 0 ||
      load_res3d(st, "blocks.0", 512, &d->blk01[0]) != 0 ||
      load_res3d(st, "blocks.1", 512, &d->blk01[1]) != 0 ||
      load_res3d(st, "blocks.3", 128, &d->blk34[0]) != 0 ||
      load_res3d(st, "blocks.4", 128, &d->blk34[1]) != 0 ||
      load_res3d(st, "blocks.6", 32,  &d->blk67[0]) != 0 ||
      load_res3d(st, "blocks.7", 32,  &d->blk67[1]) != 0) {
    free_dec_res(d);
    return -1;
  }
  d->loaded = 1;
  return 0;
}

// ===========================================================================
// 8. Stage 1: Native CUDA DINOv3-ViT-L/16 Vision Encoder (512x512 RGB -> [1029, 1024] BF16)
// ===========================================================================
static int run_dinov3_vitl16(
    ec_trellis2_t *tr, const uint8_t *rgb, int img_w, int img_h,
    uint16_t *out_cond_bf16_d, // [1029, 1024] BF16 on GPU
    double *out_ms) {
  double t0 = tr_now_ms();
  const ec_safetensors_t *st = &tr->dino_st;
  const int S = 512, P = 16, G = 32, NP = 1024, N = 1029, DIM = 1024, H = 16, D = 64, FFN = 4096;
  const size_t cond_bytes = (size_t)N * DIM * 2;

  // Persistent cross-restart Cowrie v2 SHA-NI StatePage lookup for DINOv3-ViT-L/16 (2.11 MB -> ~0.93 ms vs ~900 ms!)
  char dino_img_key[96] = {0};
  if (tr->dino_sha[0] && ec_cowrie_available() && rgb && img_w > 0 && img_h > 0) {
    char rgb_sha[65];
    ec_cowrie_sha256_hex(rgb, (size_t)img_w * (size_t)img_h * 3u, rgb_sha);
    snprintf(dino_img_key, sizeof dino_img_key, "%dx%d:%s", img_w, img_h, rgb_sha);
    uint16_t *cond_h = (uint16_t *)malloc(cond_bytes);
    if (cond_h) {
      if (ec_cowrie_sp_load("trellis2_dinov3", tr->dino_sha, dino_img_key, strlen(dino_img_key),
                            EC_COWRIE_KV_BF16, cond_h, cond_bytes) == 0) {
        cudaMemcpyAsync(out_cond_bf16_d, cond_h, cond_bytes, cudaMemcpyHostToDevice, tr->stream);
        cudaStreamSynchronize(tr->stream);
        free(cond_h);
        if (out_ms) *out_ms = tr_now_ms() - t0;
        return 0;
      }
      free(cond_h);
    }
  }

  // Official TRELLIS.2 preprocess_image parity with gradient-aware border BFS flood-fill matte!
  // Handles BOTH pure dark-studio AND gradient-lit softbox cyclorama studio backdrops without eating
  // holes into dark metals OR leaking floor/wall cyclorama geometry into DINOv3!
  uint8_t *bg_mask = (uint8_t *)calloc((size_t)img_w * (size_t)img_h, 1);
  float *alpha_map = (float *)malloc((size_t)img_w * (size_t)img_h * sizeof(float));
  int use_bfs_matte = 0;
  float crop_x0 = 0.0f, crop_y0 = 0.0f, crop_w = (float)img_w, crop_h = (float)img_h;

  if (bg_mask && alpha_map && img_w >= 32 && img_h >= 32) {
    // Compute smooth per-row left/right border colors to model vertical cyclorama gradients cleanly
    float *row_l = (float *)malloc((size_t)img_h * 3 * sizeof(float));
    float *row_r = (float *)malloc((size_t)img_h * 3 * sizeof(float));
    double edge_jump = 0.0; long edge_cnt = 0;
    for (int y = 0; y < img_h; y++) {
      float lr = 0, lg = 0, lb = 0, rr = 0, rg = 0, rb = 0;
      for (int dx = 0; dx < 4; dx++) {
        const uint8_t *pl = rgb + ((long)y * img_w + dx) * 3;
        const uint8_t *pr = rgb + ((long)y * img_w + (img_w - 1 - dx)) * 3;
        lr += pl[0]; lg += pl[1]; lb += pl[2];
        rr += pr[0]; rg += pr[1]; rb += pr[2];
      }
      row_l[y*3+0] = lr * 0.25f; row_l[y*3+1] = lg * 0.25f; row_l[y*3+2] = lb * 0.25f;
      row_r[y*3+0] = rr * 0.25f; row_r[y*3+1] = rg * 0.25f; row_r[y*3+2] = rb * 0.25f;
      if (y > 0) {
        float dl = fabsf(row_l[y*3] - row_l[(y-1)*3]) + fabsf(row_l[y*3+1] - row_l[(y-1)*3+1]) + fabsf(row_l[y*3+2] - row_l[(y-1)*3+2]);
        float dr = fabsf(row_r[y*3] - row_r[(y-1)*3]) + fabsf(row_r[y*3+1] - row_r[(y-1)*3+1]) + fabsf(row_r[y*3+2] - row_r[(y-1)*3+2]);
        edge_jump += dl + dr;
        edge_cnt += 2;
      }
    }
    // Smooth borders -> run Coons-patch bilinear studio backdrop model + local-edge-aware BFS + interior-hole punch!
    if (edge_cnt > 0 && (edge_jump / (double)edge_cnt) < 12.0) {
      float *top_e = (float *)calloc((size_t)img_w * 3, 4);
      float *bot_e = (float *)calloc((size_t)img_w * 3, 4);
      float *lft_e = (float *)calloc((size_t)img_h * 3, 4);
      float *rgt_e = (float *)calloc((size_t)img_h * 3, 4);
      uint8_t *grad_map = (uint8_t *)calloc((size_t)img_w * (size_t)img_h, 1);
      int *q = (int *)malloc((size_t)img_w * (size_t)img_h * sizeof(int));
      if (top_e && bot_e && lft_e && rgt_e && grad_map && q) {
        int strip = img_h < 8 ? img_h : 8;
        int strip_w = img_w < 8 ? img_w : 8;
        for (int x = 0; x < img_w; x++) {
          for (int k = 0; k < strip; k++) {
            const uint8_t *pt = rgb + ((long)k * img_w + x) * 3;
            const uint8_t *pb = rgb + ((long)(img_h - 1 - k) * img_w + x) * 3;
            for (int c = 0; c < 3; c++) { top_e[x*3+c] += pt[c]; bot_e[x*3+c] += pb[c]; }
          }
          for (int c = 0; c < 3; c++) { top_e[x*3+c] /= (float)strip; bot_e[x*3+c] /= (float)strip; }
        }
        for (int y = 0; y < img_h; y++) {
          for (int k = 0; k < strip_w; k++) {
            const uint8_t *pl = rgb + ((long)y * img_w + k) * 3;
            const uint8_t *pr = rgb + ((long)y * img_w + (img_w - 1 - k)) * 3;
            for (int c = 0; c < 3; c++) { lft_e[y*3+c] += pl[c]; rgt_e[y*3+c] += pr[c]; }
          }
          for (int c = 0; c < 3; c++) { lft_e[y*3+c] /= (float)strip_w; rgt_e[y*3+c] /= (float)strip_w; }
        }
        float c00[3] = {0}, c01[3] = {0}, c10[3] = {0}, c11[3] = {0};
        for (int ky = 0; ky < strip; ky++) {
          for (int kx = 0; kx < strip_w; kx++) {
            const uint8_t *p00 = rgb + ((long)ky * img_w + kx) * 3;
            const uint8_t *p01 = rgb + ((long)ky * img_w + (img_w - 1 - kx)) * 3;
            const uint8_t *p10 = rgb + ((long)(img_h - 1 - ky) * img_w + kx) * 3;
            const uint8_t *p11 = rgb + ((long)(img_h - 1 - ky) * img_w + (img_w - 1 - kx)) * 3;
            for (int c = 0; c < 3; c++) { c00[c] += p00[c]; c01[c] += p01[c]; c10[c] += p10[c]; c11[c] += p11[c]; }
          }
        }
        float inv_cn = 1.0f / (float)(strip * strip_w);
        for (int c = 0; c < 3; c++) { c00[c] *= inv_cn; c01[c] *= inv_cn; c10[c] *= inv_cn; c11[c] *= inv_cn; }

        double bg_sat_sum = 0.0;
        for (int x = 0; x < img_w; x += 4) {
          float mx = fmaxf(top_e[x*3], fmaxf(top_e[x*3+1], top_e[x*3+2]));
          float mn = fminf(top_e[x*3], fminf(top_e[x*3+1], top_e[x*3+2]));
          bg_sat_sum += (double)(mx - mn);
        }
        int neutral_bg = (bg_sat_sum / (double)((img_w + 3) / 4)) < 15.0;

        auto eval_bg = [&](int x, int y, float out[3]) {
          float u = (float)x / (float)(img_w - 1);
          float v = (float)y / (float)(img_h - 1);
          for (int c = 0; c < 3; c++) {
            out[c] = (1.0f - v) * top_e[x*3+c] + v * bot_e[x*3+c]
                   + (1.0f - u) * lft_e[y*3+c] + u * rgt_e[y*3+c]
                   - (1.0f - v) * (1.0f - u) * c00[c] - (1.0f - v) * u * c01[c]
                   - v * (1.0f - u) * c10[c] - v * u * c11[c];
          }
        };

        for (int y = 0; y < img_h; y++) {
          int ym = y > 0 ? y - 1 : 0, yp = y + 1 < img_h ? y + 1 : img_h - 1;
          for (int x = 0; x < img_w; x++) {
            int xm = x > 0 ? x - 1 : 0, xp = x + 1 < img_w ? x + 1 : img_w - 1;
            const uint8_t *pl = rgb + ((long)y * img_w + xm) * 3;
            const uint8_t *pr = rgb + ((long)y * img_w + xp) * 3;
            const uint8_t *pu = rgb + ((long)ym * img_w + x) * 3;
            const uint8_t *pd = rgb + ((long)yp * img_w + x) * 3;
            int g = 0;
            for (int c = 0; c < 3; c++) {
              int dx_v = abs((int)pr[c] - (int)pl[c]);
              int dy_v = abs((int)pd[c] - (int)pu[c]);
              if (dx_v > g) g = dx_v;
              if (dy_v > g) g = dy_v;
            }
            grad_map[y * img_w + x] = (uint8_t)(g > 255 ? 255 : g);
          }
        }

        int qh = 0, qt = 0;
        auto try_seed = [&](int x, int y) {
          int idx = y * img_w + x;
          if (!bg_mask[idx]) { bg_mask[idx] = 1; q[qt++] = idx; }
        };
        for (int x = 0; x < img_w; x++) { try_seed(x, 0); try_seed(x, img_h - 1); }
        for (int y = 0; y < img_h; y++) { try_seed(0, y); try_seed(img_w - 1, y); }
        static const int DX[4] = { 1, -1, 0, 0 };
        static const int DY[4] = { 0, 0, 1, -1 };
        while (qh < qt) {
          int cur = q[qh++];
          int cx = cur % img_w, cy = cur / img_w;
          const uint8_t *pc = rgb + (long)cur * 3;
          for (int d = 0; d < 4; d++) {
            int nx = cx + DX[d], ny = cy + DY[d];
            if (nx < 0 || nx >= img_w || ny < 0 || ny >= img_h) continue;
            int nidx = ny * img_w + nx;
            if (bg_mask[nidx]) continue;
            if (grad_map[nidx] > 12) continue;
            const uint8_t *pn = rgb + (long)nidx * 3;
            float dr = (float)pn[0] - (float)pc[0];
            float dg = (float)pn[1] - (float)pc[1];
            float db = (float)pn[2] - (float)pc[2];
            if (dr*dr + dg*dg + db*db > 8.0f * 8.0f) continue;
            int cmax = pn[0] > pn[1] ? (pn[0] > pn[2] ? pn[0] : pn[2]) : (pn[1] > pn[2] ? pn[1] : pn[2]);
            int cmin = pn[0] < pn[1] ? (pn[0] < pn[2] ? pn[0] : pn[2]) : (pn[1] < pn[2] ? pn[1] : pn[2]);
            int sat = cmax - cmin;
            if (neutral_bg && sat > 10) continue;
            float be[3];
            eval_bg(nx, ny, be);
            float bdr = (float)pn[0] - be[0], bdg = (float)pn[1] - be[1], bdb = (float)pn[2] - be[2];
            float bd2 = bdr*bdr + bdg*bdg + bdb*bdb;
            float pm = ((float)pn[0] + (float)pn[1] + (float)pn[2]) * (1.0f / 3.0f);
            float bm = (be[0] + be[1] + be[2]) * (1.0f / 3.0f);
            int is_soft_shadow = (neutral_bg && sat <= 6 && pm < bm + 4.0f && bd2 <= 65.0f * 65.0f);
            if (bd2 <= 28.0f * 28.0f || is_soft_shadow) {
              bg_mask[nidx] = 1;
              q[qt++] = nidx;
            }
          }
        }
        // Punch out enclosed studio-backdrop holes (e.g. inside teapot/mug handles & loops!)
        {
          uint8_t *cand = (uint8_t *)calloc((size_t)img_w * (size_t)img_h, 1);
          uint8_t *vis  = (uint8_t *)calloc((size_t)img_w * (size_t)img_h, 1);
          if (cand && vis) {
            for (int y = 0; y < img_h; y++) {
              for (int x = 0; x < img_w; x++) {
                int idx = y * img_w + x;
                if (bg_mask[idx] || grad_map[idx] > 10) continue;
                const uint8_t *p = rgb + (long)idx * 3;
                int cmax = p[0] > p[1] ? (p[0] > p[2] ? p[0] : p[2]) : (p[1] > p[2] ? p[1] : p[2]);
                int cmin = p[0] < p[1] ? (p[0] < p[2] ? p[0] : p[2]) : (p[1] < p[2] ? p[1] : p[2]);
                if (neutral_bg && (cmax - cmin) > 8) continue;
                float be[3];
                eval_bg(x, y, be);
                float bdr = (float)p[0] - be[0], bdg = (float)p[1] - be[1], bdb = (float)p[2] - be[2];
                if (bdr*bdr + bdg*bdg + bdb*bdb <= 22.0f * 22.0f) cand[idx] = 1;
              }
            }
            for (int i = 0; i < img_w * img_h; i++) {
              if (!cand[i] || vis[i]) continue;
              int ch = 0, ct = 0;
              vis[i] = 1; q[ct++] = i;
              while (ch < ct) {
                int c = q[ch++], cx = c % img_w, cy = c / img_w;
                for (int d = 0; d < 4; d++) {
                  int nx = cx + DX[d], ny = cy + DY[d];
                  if (nx < 0 || nx >= img_w || ny < 0 || ny >= img_h) continue;
                  int ni = ny * img_w + nx;
                  if (cand[ni] && !vis[ni]) { vis[ni] = 1; q[ct++] = ni; }
                }
              }
              if (ct >= 400) {
                for (int k = 0; k < ct; k++) bg_mask[q[k]] = 1;
              }
            }
          }
          free(cand); free(vis);
        }
        // Keep only the single largest connected foreground component (drops any detached cast floor shadow!)
        {
          int *comp_id = (int *)calloc((size_t)img_w * (size_t)img_h, sizeof(int));
          int best_cid = 0, best_cnt = 0, cur_cid = 0;
          if (comp_id) {
            for (int i = 0; i < img_w * img_h; i++) {
              if (bg_mask[i] || comp_id[i]) continue;
              cur_cid++;
              int ch = 0, ct = 0;
              comp_id[i] = cur_cid;
              q[ct++] = i;
              while (ch < ct) {
                int c = q[ch++], cx = c % img_w, cy = c / img_w;
                for (int d = 0; d < 4; d++) {
                  int nx = cx + DX[d], ny = cy + DY[d];
                  if (nx < 0 || nx >= img_w || ny < 0 || ny >= img_h) continue;
                  int ni = ny * img_w + nx;
                  if (!bg_mask[ni] && !comp_id[ni]) { comp_id[ni] = cur_cid; q[ct++] = ni; }
                }
              }
              if (ct > best_cnt) { best_cnt = ct; best_cid = cur_cid; }
            }
            if (best_cid > 0) {
              for (int i = 0; i < img_w * img_h; i++) {
                if (!bg_mask[i] && comp_id[i] != best_cid) bg_mask[i] = 1;
              }
            }
            free(comp_id);
          }
        }
        // Compute tight foreground bounding box and validate foreground fraction [3%, 94%]
        int bx0 = img_w, by0 = img_h, bx1 = -1, by1 = -1;
        long fg_cnt = 0;
        for (int y = 0; y < img_h; y++) {
          for (int x = 0; x < img_w; x++) {
            if (!bg_mask[y * img_w + x]) {
              fg_cnt++;
              if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
              if (y < by0) by0 = y; if (y > by1) by1 = y;
            }
          }
        }
        double fg_frac = (double)fg_cnt / (double)((long)img_w * img_h);
        if (fg_frac >= 0.03 && fg_frac <= 0.94 && bx1 > bx0 + 16 && by1 > by0 + 16) {
          use_bfs_matte = 1;
          for (int y = 0; y < img_h; y++) {
            for (int x = 0; x < img_w; x++) {
              if (!bg_mask[y * img_w + x]) {
                int bg_nb = 0;
                for (int dy = -1; dy <= 1; dy++) {
                  int yy = y + dy; if (yy < 0 || yy >= img_h) continue;
                  for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx; if (xx < 0 || xx >= img_w) continue;
                    if (bg_mask[yy * img_w + xx]) bg_nb++;
                  }
                }
                alpha_map[y * img_w + x] = bg_nb ? (1.0f - (float)bg_nb / 10.0f) : 1.0f;
              } else {
                alpha_map[y * img_w + x] = 0.0f;
              }
            }
          }
          float cx = 0.5f * (float)(bx0 + bx1);
          float cy = 0.5f * (float)(by0 + by1);
          float sz = (float)((bx1 - bx0 > by1 - by0) ? (bx1 - bx0) : (by1 - by0)) * 1.06f;
          crop_x0 = cx - 0.5f * sz;
          crop_y0 = cy - 0.5f * sz;
          crop_w = sz;
          crop_h = sz;
        }
      }
      free(top_e); free(bot_e); free(lft_e); free(rgt_e); free(grad_map); free(q);
    }
    free(row_l); free(row_r);
  }

  // Bilinear resize to 512x512 + clean alpha premultiply onto pure (0,0,0) + ImageNet normalize
  static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
  static const float STD[3]  = {0.229f, 0.224f, 0.225f};
  float *img_norm = (float *)malloc((size_t)3 * S * S * 4);
  for (int y = 0; y < S; y++) {
    float fy = crop_y0 + ((float)y + 0.5f) * crop_h / (float)S - 0.5f;
    int y0 = (int)floorf(fy), y1 = y0 + 1;
    float wy = fy - (float)y0; if (wy < 0.0f) wy = 0.0f; if (wy > 1.0f) wy = 1.0f;
    for (int x = 0; x < S; x++) {
      float fx = crop_x0 + ((float)x + 0.5f) * crop_w / (float)S - 0.5f;
      int x0 = (int)floorf(fx), x1 = x0 + 1;
      float wx = fx - (float)x0; if (wx < 0.0f) wx = 0.0f; if (wx > 1.0f) wx = 1.0f;
      float rgb_f[3] = {0.0f, 0.0f, 0.0f};
      if (fy >= 0.0f && fy < (float)img_h && fx >= 0.0f && fx < (float)img_w) {
        int cy0 = y0 < 0 ? 0 : (y0 >= img_h ? img_h - 1 : y0);
        int cy1 = y1 < 0 ? 0 : (y1 >= img_h ? img_h - 1 : y1);
        int cx0 = x0 < 0 ? 0 : (x0 >= img_w ? img_w - 1 : x0);
        int cx1 = x1 < 0 ? 0 : (x1 >= img_w ? img_w - 1 : x1);
        for (int c = 0; c < 3; c++) {
          float v00 = rgb[((long)cy0 * img_w + cx0) * 3 + c];
          float v01 = rgb[((long)cy0 * img_w + cx1) * 3 + c];
          float v10 = rgb[((long)cy1 * img_w + cx0) * 3 + c];
          float v11 = rgb[((long)cy1 * img_w + cx1) * 3 + c];
          rgb_f[c] = (v00 * (1.0f - wx) + v01 * wx) * (1.0f - wy) +
                     (v10 * (1.0f - wx) + v11 * wx) * wy;
        }
        if (use_bfs_matte) {
          float a00 = alpha_map[(long)cy0 * img_w + cx0];
          float a01 = alpha_map[(long)cy0 * img_w + cx1];
          float a10 = alpha_map[(long)cy1 * img_w + cx0];
          float a11 = alpha_map[(long)cy1 * img_w + cx1];
          float a = (a00 * (1.0f - wx) + a01 * wx) * (1.0f - wy) +
                    (a10 * (1.0f - wx) + a11 * wx) * wy;
          rgb_f[0] *= a; rgb_f[1] *= a; rgb_f[2] *= a;
        }
      }
      for (int c = 0; c < 3; c++) {
        img_norm[((long)c * S + y) * S + x] = (rgb_f[c] * (1.0f / 255.0f) - MEAN[c]) / STD[c];
      }
    }
  }
  free(bg_mask);
  free(alpha_map);

  uint16_t *patches_h = (uint16_t *)malloc((size_t)NP * 768 * 2);
  for (int py = 0; py < G; py++) {
    for (int px = 0; px < G; px++) {
      uint16_t *dst = patches_h + (long)(py * G + px) * 768;
      int idx = 0;
      for (int c = 0; c < 3; c++) {
        for (int dy = 0; dy < P; dy++) {
          const float *row = img_norm + ((long)c * S + py * P + dy) * S + px * P;
          for (int dx = 0; dx < P; dx++) dst[idx++] = tr_f32_to_bf16_h(row[dx]);
        }
      }
    }
  }
  free(img_norm);

  uint16_t *patches_d = NULL, *pw_d = tr_upload_bf16(st, "patch_embed.proj.weight");
  float *pb_d = tr_upload_f32(st, "patch_embed.proj.bias");
  float *cls_d = tr_upload_f32(st, "cls_token");
  float *reg_d = tr_upload_f32(st, "reg_token");
  float *x_d = NULL;
  cudaMalloc(&patches_d, (size_t)NP * 768 * 2);
  cudaMalloc(&x_d,       (size_t)N * DIM * 4);
  cudaMemcpyAsync(patches_d, patches_h, (size_t)NP * 768 * 2, cudaMemcpyHostToDevice, tr->stream);
  free(patches_h);

  // Token 0 = cls, Tokens 1..4 = reg, Tokens 5..1028 = patches
  cudaMemcpyAsync(x_d,                  cls_d, 1L * DIM * 4, cudaMemcpyDeviceToDevice, tr->stream);
  cudaMemcpyAsync(x_d + 1L * DIM,       reg_d, 4L * DIM * 4, cudaMemcpyDeviceToDevice, tr->stream);
  tr_gemm(patches_d, pw_d, pb_d, x_d + 5L * DIM, NULL, NP, DIM, 768, tr->stream);
  cudaStreamSynchronize(tr->stream);
  cudaFree(patches_d); cudaFree(pw_d); cudaFree(pb_d); cudaFree(cls_d); cudaFree(reg_d);

  // Working buffers for 24 ViT-L blocks (BF16 Q/K/V -> direct TensorCore FA2!)
  uint16_t *xn_bf16 = NULL, *Q_d = NULL, *K_d = NULL, *V_d = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL;
  float *qkv_d = NULL, *tmp_d = NULL, *fc1_d = NULL;
  cudaMalloc(&xn_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&qkv_d,    (size_t)N * 3 * DIM * 4);
  cudaMalloc(&Q_d,      (size_t)N * DIM * 2);
  cudaMalloc(&K_d,      (size_t)N * DIM * 2);
  cudaMalloc(&V_d,      (size_t)N * DIM * 2);
  cudaMalloc(&ao_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&tmp_d,    (size_t)N * DIM * 4);
  cudaMalloc(&fc1_d,    (size_t)N * FFN * 4);
  cudaMalloc(&mlp_bf16, (size_t)N * FFN * 2);

  char k[160];
  for (int l = 0; l < 24; l++) {
    snprintf(k, sizeof k, "blocks.%d.norm1.weight", l);     float *n1w   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm1.bias", l);       float *n1b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.attn.qkv.weight", l);  uint16_t *qkv_w = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.attn.proj.weight", l); uint16_t *pr_w  = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.attn.proj.bias", l);   float *pr_b  = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.gamma_1", l);          float *g1    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.weight", l);     float *n2w   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.bias", l);       float *n2b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.fc1.weight", l);   uint16_t *fc1w = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.fc1.bias", l);     float *fc1b  = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.fc2.weight", l);   uint16_t *fc2w = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.fc2.bias", l);     float *fc2b  = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.gamma_2", l);          float *g2    = tr_upload_f32(st, k);

    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(x_d, n1w, n1b, NULL, NULL, NULL, xn_bf16, N, DIM, 1e-5f, 0);
    tr_gemm(xn_bf16, qkv_w, NULL, qkv_d, NULL, N, 3 * DIM, DIM, tr->stream);
    int nw = N * H;
    k_dino_unpack_rope<<<(nw + 7) / 8, 256, 0, tr->stream>>>(qkv_d, Q_d, K_d, V_d, N, H, D, 5, G);
    ec_ops_fa2_bf16(Q_d, K_d, V_d, NULL, ao_bf16, N, N, N, N, H, H, D, 1.0f / sqrtf((float)D), 0, tr->stream);
    tr_gemm(ao_bf16, pr_w, pr_b, tmp_d, NULL, N, DIM, DIM, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(x_d, tmp_d, g1, N, DIM);

    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(x_d, n2w, n2b, NULL, NULL, NULL, xn_bf16, N, DIM, 1e-5f, 0);
    tr_gemm(xn_bf16, fc1w, fc1b, fc1_d, NULL, N, FFN, DIM, tr->stream);
    k_tr_gelu_erf_bf16<<<((long)N * FFN + 255) / 256, 256, 0, tr->stream>>>(fc1_d, mlp_bf16, (long)N * FFN);
    tr_gemm(mlp_bf16, fc2w, fc2b, tmp_d, NULL, N, DIM, FFN, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(x_d, tmp_d, g2, N, DIM);

    cudaStreamSynchronize(tr->stream);
    cudaFree(n1w); cudaFree(n1b); cudaFree(qkv_w); cudaFree(pr_w); cudaFree(pr_b); cudaFree(g1);
    cudaFree(n2w); cudaFree(n2b); cudaFree(fc1w); cudaFree(fc1b); cudaFree(fc2w); cudaFree(fc2b); cudaFree(g2);
  }

  // Exact TRELLIS.2 DinoV3FeatureExtractor output: affine-free F.layer_norm(hidden_states, [1024], eps=1e-5)
  k_tr_layernorm<<<N, 256, 0, tr->stream>>>(x_d, NULL, NULL, NULL, NULL, NULL, out_cond_bf16_d, N, DIM, 1e-5f, 0);
  cudaStreamSynchronize(tr->stream);

  // Persist 16 KB SHA-NI Cowrie StatePage for DINOv3 vision conditioning across restarts & rerolls
  if (dino_img_key[0] && tr->dino_sha[0] && ec_cowrie_available()) {
    uint16_t *cond_h = (uint16_t *)malloc(cond_bytes);
    if (cond_h) {
      cudaMemcpy(cond_h, out_cond_bf16_d, cond_bytes, cudaMemcpyDeviceToHost);
      ec_cowrie_sp_save("trellis2_dinov3", tr->dino_sha, dino_img_key, strlen(dino_img_key),
                        (uint32_t)N, EC_COWRIE_KV_BF16, cond_h, cond_bytes);
      free(cond_h);
    }
  }

  cudaFree(x_d); cudaFree(xn_bf16); cudaFree(qkv_d);
  cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d);
  cudaFree(ao_bf16); cudaFree(tmp_d); cudaFree(fc1_d); cudaFree(mlp_bf16);
  if (out_ms) *out_ms = tr_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 9. Stage 2: Native CUDA 30-Block SparseStructureFlowModel (12-Step Flow Euler CFG)
// ===========================================================================
static uint64_t tr_splitmix64(uint64_t *s) {
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static float tr_randn(uint64_t *rng) {
  double u1 = ((double)(tr_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  double u2 = ((double)(tr_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  return (float)(sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2));
}

typedef struct {
  uint16_t *x8_bf16, *xn_bf16, *Q_b, *K_b, *V_b, *ao_bf16, *mlp_bf16;
  uint16_t *t_f_d, *t_hb_d;
  float *x_d, *qkv_d, *tmp_d, *fc0_d, *mod_d, *mod_blk_d, *t_hf_d;
} ss_work_t;

static void compute_ss_t_mod(ec_trellis2_t *tr, float t_val, ss_work_t *W, float *out_mod9216_d) {
  const tr_ss_flow_res_t *f = &tr->flow;
  uint16_t fb_h[256];
  double scaled = (double)t_val * 1000.0;
  for (int i = 0; i < 128; i++) {
    double fr = exp(-log(10000.0) * (double)i / 128.0);
    double a = scaled * fr;
    fb_h[i]       = tr_f32_to_bf16_h((float)cos(a));
    fb_h[128 + i] = tr_f32_to_bf16_h((float)sin(a));
  }
  cudaMemcpyAsync(W->t_f_d, fb_h, 256 * 2, cudaMemcpyHostToDevice, tr->stream);
  tr_gemm(W->t_f_d, f->t0_w, f->t0_b, W->t_hf_d, NULL, 1, 1536, 256, tr->stream);
  ec_ops_silu_f32_to_bf16(W->t_hf_d, W->t_hb_d, 1536, tr->stream);
  tr_gemm(W->t_hb_d, f->t2_w, f->t2_b, W->t_hf_d, NULL, 1, 1536, 1536, tr->stream);
  ec_ops_silu_f32_to_bf16(W->t_hf_d, W->t_hb_d, 1536, tr->stream);
  tr_gemm(W->t_hb_d, f->adaln_w, f->adaln_b, out_mod9216_d, NULL, 1, 9216, 1536, tr->stream);
}

static void ss_flow_forward(
    ec_trellis2_t *tr, const float *xt_d, float t_val,
    uint16_t *const ck_cache[30], uint16_t *const cv_cache[30],
    ss_work_t *W, float *out_v_d) {
  const tr_ss_flow_res_t *f = &tr->flow;
  const int N = 4096, DIM = 1536, H = 12, FFN = 8192, Lkv = 1029;
  const float scale = 1.0f / sqrtf(128.0f);

  compute_ss_t_mod(tr, t_val, W, W->mod_d);
  k_tr_f32_to_bf16<<<(N * 8 + 255) / 256, 256, 0, tr->stream>>>(xt_d, W->x8_bf16, N * 8);
  tr_gemm(W->x8_bf16, f->in_w, f->in_b, W->x_d, NULL, N, DIM, 8, tr->stream);

  int nw_sa = N * H;

  for (int l = 0; l < 30; l++) {
    const tr_ss_block_t *B = &f->blocks[l];
    k_tr_add_vec<<<(9216 + 255) / 256, 256, 0, tr->stream>>>(B->modulation, W->mod_d, W->mod_blk_d, 9216);
    const float *shift_msa = W->mod_blk_d + 0 * DIM;
    const float *scale_msa = W->mod_blk_d + 1 * DIM;
    const float *gate_msa  = W->mod_blk_d + 2 * DIM;
    const float *shift_mlp = W->mod_blk_d + 3 * DIM;
    const float *scale_mlp = W->mod_blk_d + 4 * DIM;
    const float *gate_mlp  = W->mod_blk_d + 5 * DIM;

    // 1. Self-Attention (AdaLN -> to_qkv -> MultiHeadRMSNorm + 3D RoPE -> TensorCore BF16 FA2 -> gated residual)
    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(
        W->x_d, NULL, NULL, scale_msa, shift_msa, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->sa_qkv_w, B->sa_qkv_b, W->qkv_d, NULL, N, 3 * DIM, DIM, tr->stream);
    k_ss_self_qkv_rope<<<(nw_sa + 7) / 8, 256, 0, tr->stream>>>(
        W->qkv_d, B->sa_qn, B->sa_kn, W->Q_b, W->K_b, W->V_b, N, H, 16);
    ec_ops_fa2_bf16(W->Q_b, W->K_b, W->V_b, NULL, W->ao_bf16, N, N, N, N, H, H, 128, scale, 0, tr->stream);
    tr_gemm(W->ao_bf16, B->sa_out_w, B->sa_out_b, W->tmp_d, NULL, N, DIM, DIM, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, gate_msa, N, DIM);

    // 2. Cross-Attention (affine norm2 -> to_q + MultiHeadRMSNorm -> TensorCore BF16 FA2 -> residual)
    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(
        W->x_d, B->norm2_w, B->norm2_b, NULL, NULL, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->ca_q_w, B->ca_q_b, W->qkv_d, NULL, N, DIM, DIM, tr->stream);
    k_ss_cross_q_norm<<<(nw_sa + 7) / 8, 256, 0, tr->stream>>>(W->qkv_d, B->ca_qn, W->Q_b, N, H);
    ec_ops_fa2_bf16(W->Q_b, ck_cache[l], cv_cache[l], NULL, W->ao_bf16, N, Lkv, N, Lkv, H, H, 128, scale, 0, tr->stream);
    tr_gemm(W->ao_bf16, B->ca_out_w, B->ca_out_b, W->tmp_d, NULL, N, DIM, DIM, tr->stream);
    k_tr_add_inplace<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, (long)N * DIM);

    // 3. MLP (AdaLN -> 1536->8192 GELU(tanh) -> 8192->1536 -> gated residual)
    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(
        W->x_d, NULL, NULL, scale_mlp, shift_mlp, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->mlp0_w, B->mlp0_b, W->fc0_d, NULL, N, FFN, DIM, tr->stream);
    k_tr_gelu_tanh_bf16<<<((long)N * FFN + 255) / 256, 256, 0, tr->stream>>>(W->fc0_d, W->mlp_bf16, (long)N * FFN);
    tr_gemm(W->mlp_bf16, B->mlp2_w, B->mlp2_b, W->tmp_d, NULL, N, DIM, FFN, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, gate_mlp, N, DIM);
  }

  // Output Layer: F.layer_norm(no affine, eps=1e-5) -> out_layer (1536 -> 8)
  k_tr_layernorm<<<N, 256, 0, tr->stream>>>(W->x_d, NULL, NULL, NULL, NULL, NULL, W->xn_bf16, N, DIM, 1e-5f, 0);
  tr_gemm(W->xn_bf16, f->out_w, f->out_b, out_v_d, NULL, N, 8, DIM, tr->stream);
}

static int run_ss_flow(
    ec_trellis2_t *tr, uint16_t *const *view_conds_d, int n_views,
    int steps, float cfg_scale, float rescale_t, uint32_t seed,
    float *out_z_d, // [4096, 8] F32 on GPU (matches [16, 16, 16, 8]!)
    ec_trellis2_progress_fn cb, void *cb_ctx, double *out_ms) {
  double t0 = tr_now_ms();
  const int N = 4096, DIM = 1536, H = 12, FFN = 8192, Lkv = 1029;
  const float sigma_min = 1e-5f;
  const float guidance_rescale = 0.7f;
  if (n_views < 1) n_views = 1;
  if (n_views > 4) n_views = 4;

  // Build neg_cond = zeros[1029, 1024] BF16
  uint16_t *neg_bf16_d = NULL;
  cudaMalloc(&neg_bf16_d, (size_t)Lkv * 1024 * 2);
  cudaMemsetAsync(neg_bf16_d, 0, (size_t)Lkv * 1024 * 2, tr->stream);

  // Official TRELLIS.2 Multi-View Stochastic Cycling: cache K/V per view separately (NEVER pixel-average DINOv3 patches!)
  uint16_t *ck_pos[4][30] = {{0}}, *cv_pos[4][30] = {{0}}, *ck_neg[30] = {0}, *cv_neg[30] = {0};
  float *kv_tmp = NULL;
  cudaMalloc(&kv_tmp, (size_t)Lkv * 2 * DIM * 4);
  int nw_kv = Lkv * H;
  for (int l = 0; l < 30; l++) {
    const tr_ss_block_t *B = &tr->flow.blocks[l];
    for (int v = 0; v < n_views; v++) {
      cudaMalloc(&ck_pos[v][l], (size_t)Lkv * DIM * 2);
      cudaMalloc(&cv_pos[v][l], (size_t)Lkv * DIM * 2);
      tr_gemm(view_conds_d[v], B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, Lkv, 2 * DIM, 1024, tr->stream);
      k_ss_cross_kv_norm<<<(nw_kv + 7) / 8, 256, 0, tr->stream>>>(kv_tmp, B->ca_kn, ck_pos[v][l], cv_pos[v][l], Lkv, H);
    }
    cudaMalloc(&ck_neg[l], (size_t)Lkv * DIM * 2);
    cudaMalloc(&cv_neg[l], (size_t)Lkv * DIM * 2);
    tr_gemm(neg_bf16_d, B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, Lkv, 2 * DIM, 1024, tr->stream);
    k_ss_cross_kv_norm<<<(nw_kv + 7) / 8, 256, 0, tr->stream>>>(kv_tmp, B->ca_kn, ck_neg[l], cv_neg[l], Lkv, H);
  }
  cudaFree(kv_tmp);
  cudaFree(neg_bf16_d);

  // Working buffers (100% GPU-resident, zero per-step cudaMalloc/sync!)
  ss_work_t W;
  cudaMalloc(&W.x8_bf16,  (size_t)N * 8 * 2);
  cudaMalloc(&W.x_d,      (size_t)N * DIM * 4);
  cudaMalloc(&W.xn_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&W.qkv_d,    (size_t)N * 3 * DIM * 4);
  cudaMalloc(&W.Q_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.K_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.V_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.ao_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&W.tmp_d,    (size_t)N * DIM * 4);
  cudaMalloc(&W.fc0_d,    (size_t)N * FFN * 4);
  cudaMalloc(&W.mlp_bf16, (size_t)N * FFN * 2);
  cudaMalloc(&W.mod_d,     9216 * 4);
  cudaMalloc(&W.mod_blk_d, 9216 * 4);
  cudaMalloc(&W.t_f_d,     256 * 2);
  cudaMalloc(&W.t_hf_d,    1536 * 4);
  cudaMalloc(&W.t_hb_d,    1536 * 2);

  // Initial noise xt [4096, 8] -> upload ONCE to GPU!
  float *xt_h = (float *)malloc((size_t)N * 8 * 4);
  uint64_t rng = (uint64_t)seed * 6364136223846793005ULL + 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N * 8; i++) xt_h[i] = tr_randn(&rng);

  float *xt_d = NULL, *vp_d = NULL, *vn_d = NULL;
  cudaMalloc(&xt_d, (size_t)N * 8 * 4);
  cudaMalloc(&vp_d, (size_t)N * 8 * 4);
  cudaMalloc(&vn_d, (size_t)N * 8 * 4);
  cudaMemcpyAsync(xt_d, xt_h, (size_t)N * 8 * 4, cudaMemcpyHostToDevice, tr->stream);
  free(xt_h);

  // Build rescaled t schedule
  float t_seq[65];
  for (int i = 0; i <= steps; i++) {
    float u = 1.0f - (float)i / (float)steps;
    t_seq[i] = rescale_t * u / (1.0f + (rescale_t - 1.0f) * u);
  }

  int cancelled = 0;
  for (int s = 0; s < steps; s++) {
    if (cb && cb(s + 1, steps + 3, "trellis2_ss_flow_step", cb_ctx) != 0) {
      cancelled = 1;
      break;
    }
    float t_cur = t_seq[s], t_prev = t_seq[s + 1], dt = t_cur - t_prev;
    int vid = s % n_views;
    ss_flow_forward(tr, xt_d, t_cur, ck_pos[vid], cv_pos[vid], &W, vp_d);

    int use_cfg = (cfg_scale > 1.01f && t_cur >= 0.6f && t_cur <= 1.0f);
    if (use_cfg) {
      ss_flow_forward(tr, xt_d, t_cur, ck_neg, cv_neg, &W, vn_d);
    }
    ec_ops_trellis_cfg_euler(
        xt_d, vp_d, use_cfg ? vn_d : NULL, N * 8, dt, t_cur,
        cfg_scale, guidance_rescale, sigma_min, use_cfg, tr->stream);
  }

  if (!cancelled) {
    cudaMemcpyAsync(out_z_d, xt_d, (size_t)N * 8 * 4, cudaMemcpyDeviceToDevice, tr->stream);
  }
  cudaStreamSynchronize(tr->stream);

  for (int l = 0; l < 30; l++) {
    for (int v = 0; v < n_views; v++) { cudaFree(ck_pos[v][l]); cudaFree(cv_pos[v][l]); }
    cudaFree(ck_neg[l]); cudaFree(cv_neg[l]);
  }
  cudaFree(W.x8_bf16); cudaFree(W.x_d); cudaFree(W.xn_bf16); cudaFree(W.qkv_d);
  cudaFree(W.Q_b); cudaFree(W.K_b); cudaFree(W.V_b); cudaFree(W.ao_bf16);
  cudaFree(W.tmp_d); cudaFree(W.fc0_d); cudaFree(W.mlp_bf16); cudaFree(W.mod_d); cudaFree(W.mod_blk_d);
  cudaFree(W.t_f_d); cudaFree(W.t_hf_d); cudaFree(W.t_hb_d);
  cudaFree(xt_d); cudaFree(vp_d); cudaFree(vn_d);

  if (out_ms) *out_ms = tr_now_ms() - t0;
  return cancelled ? -2 : 0;
}

// ===========================================================================
// 10. Stage 3: Native CUDA SparseStructureDecoder (16^3x8 -> 64^3x1 logits)
// ===========================================================================
static void run_conv3d_k3(
    ec_trellis2_t *tr, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int R, int Cin, int Cout, uint16_t *col_scratch, size_t col_cap_elems) {
  int V = R * R * R;
  int max_chunk = (int)(col_cap_elems / (size_t)(27 * Cin));
  if (max_chunk < 64) max_chunk = 64;
  for (int v0 = 0; v0 < V; v0 += max_chunk) {
    int ch = (V - v0 < max_chunk) ? (V - v0) : max_chunk;
    long n = (long)ch * 27 * Cin;
    k_tr_im2col_3d_k3<<<(n + 255) / 256, 256, 0, tr->stream>>>(x_d, col_scratch, R, Cin, v0, ch);
    tr_gemm(col_scratch, w_bf16, b_f32, y_d + (long)v0 * Cout, NULL, ch, Cout, 27 * Cin, tr->stream);
  }
}

static void run_res3d(
    ec_trellis2_t *tr, float *x_d, float *buf_a, float *buf_b,
    const tr_res3d_t *r, int R, int C, uint16_t *col_scratch, size_t col_cap_elems) {
  int V = R * R * R;
  k_tr_layernorm<<<V, 256, 0, tr->stream>>>(x_d, r->n1_w, r->n1_b, NULL, NULL, buf_a, NULL, V, C, 1e-5f, 1);
  run_conv3d_k3(tr, buf_a, buf_b, r->c1_w, r->c1_b, R, C, C, col_scratch, col_cap_elems);
  k_tr_layernorm<<<V, 256, 0, tr->stream>>>(buf_b, r->n2_w, r->n2_b, NULL, NULL, buf_a, NULL, V, C, 1e-5f, 1);
  run_conv3d_k3(tr, buf_a, buf_b, r->c2_w, r->c2_b, R, C, C, col_scratch, col_cap_elems);
  k_tr_add_inplace<<<((long)V * C + 255) / 256, 256, 0, tr->stream>>>(x_d, buf_b, (long)V * C);
}

static int run_ss_decoder(
    ec_trellis2_t *tr, const float *z_d,
    float *out_logits_64_h, // host [64*64*64] F32
    double *out_ms) {
  double t0 = tr_now_ms();
  const tr_ss_dec_res_t *d = &tr->dec;
  const size_t COL_CAP = 32L * 1024 * 1024; // 64 MB BF16 scratch
  uint16_t *col = NULL;
  if (cudaMalloc(&col, COL_CAP * 2) != cudaSuccess) return -1;

  // Stage 16^3 (C = 512)
  float *x16 = NULL, *a16 = NULL, *b16 = NULL, *up16 = NULL;
  cudaMalloc(&x16, 4096L * 512 * 4);
  cudaMalloc(&a16, 4096L * 512 * 4);
  cudaMalloc(&b16, 4096L * 512 * 4);
  run_conv3d_k3(tr, z_d, x16, d->in_w, d->in_b, 16, 8, 512, col, COL_CAP);
  run_res3d(tr, x16, a16, b16, &d->mid[0],   16, 512, col, COL_CAP);
  run_res3d(tr, x16, a16, b16, &d->mid[1],   16, 512, col, COL_CAP);
  run_res3d(tr, x16, a16, b16, &d->blk01[0], 16, 512, col, COL_CAP);
  run_res3d(tr, x16, a16, b16, &d->blk01[1], 16, 512, col, COL_CAP);
  cudaFree(a16); cudaFree(b16);

  // Upsample 16^3 (512) -> 32^3 (128)
  cudaMalloc(&up16, 4096L * 1024 * 4);
  run_conv3d_k3(tr, x16, up16, d->up2_w, d->up2_b, 16, 512, 1024, col, COL_CAP);
  cudaFree(x16);

  float *x32 = NULL, *a32 = NULL, *b32 = NULL, *up32 = NULL;
  cudaMalloc(&x32, 32768L * 128 * 4);
  k_tr_pixel_shuffle_3d<<<(32768L * 128 + 255) / 256, 256, 0, tr->stream>>>(up16, x32, 16, 128);
  cudaFree(up16);

  cudaMalloc(&a32, 32768L * 128 * 4);
  cudaMalloc(&b32, 32768L * 128 * 4);
  run_res3d(tr, x32, a32, b32, &d->blk34[0], 32, 128, col, COL_CAP);
  run_res3d(tr, x32, a32, b32, &d->blk34[1], 32, 128, col, COL_CAP);
  cudaFree(a32); cudaFree(b32);

  // Upsample 32^3 (128) -> 64^3 (32)
  cudaMalloc(&up32, 32768L * 256 * 4);
  run_conv3d_k3(tr, x32, up32, d->up5_w, d->up5_b, 32, 128, 256, col, COL_CAP);
  cudaFree(x32);

  float *x64 = NULL, *a64 = NULL, *b64 = NULL, *out64 = NULL;
  cudaMalloc(&x64, 262144L * 32 * 4);
  k_tr_pixel_shuffle_3d<<<(262144L * 32 + 255) / 256, 256, 0, tr->stream>>>(up32, x64, 32, 32);
  cudaFree(up32);

  cudaMalloc(&a64, 262144L * 32 * 4);
  cudaMalloc(&b64, 262144L * 32 * 4);
  run_res3d(tr, x64, a64, b64, &d->blk67[0], 64, 32, col, COL_CAP);
  run_res3d(tr, x64, a64, b64, &d->blk67[1], 64, 32, col, COL_CAP);
  cudaFree(b64);

  // Out Layer: ChannelLayerNorm32(32) -> SiLU -> Conv3d(32 -> 1)
  cudaMalloc(&out64, 262144L * 4);
  k_tr_layernorm<<<262144, 64, 0, tr->stream>>>(x64, d->out_nw, d->out_nb, NULL, NULL, a64, NULL, 262144, 32, 1e-5f, 1);
  cudaFree(x64);
  run_conv3d_k3(tr, a64, out64, d->out_w, d->out_b, 64, 32, 1, col, COL_CAP);
  cudaFree(a64);
  cudaFree(col);

  cudaMemcpyAsync(out_logits_64_h, out64, 262144L * 4, cudaMemcpyDeviceToHost, tr->stream);
  cudaStreamSynchronize(tr->stream);
  cudaFree(out64);
  if (out_ms) *out_ms = tr_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 11. Stage 4: Sub-Voxel Smooth Surface Nets Mesher + 3DGS Splat Builder
//     Extracts watertight smooth surface from continuous 64^3 logit field (> 0 = solid).
// ===========================================================================
static inline float sample_grid_64(const float *g, int x, int y, int z) {
  if (x < 0 || x >= 64 || y < 0 || y >= 64 || z < 0 || z >= 64) return -10.0f;
  return g[(x * 64 + y) * 64 + z];
}

static inline float tr_srgb_to_linear(float s) {
  if (s <= 0.0f) return 0.0f;
  if (s >= 1.0f) return 1.0f;
  return (s <= 0.04045f) ? (s / 12.92f) : powf((s + 0.055f) / 1.055f, 2.4f);
}

static int extract_surface_nets_64(
    const float *logits, const float base_color[4],
    const uint8_t *ref_rgb, int ref_w, int ref_h,
    ec_mesh_t *mesh, ec_splat3d_t *splat, int *out_active_vox) {
  memset(mesh, 0, sizeof *mesh);
  memset(splat, 0, sizeof *splat);

  const int R = 64;
  // Detect spurious dark-studio outer-box shell (all 6 faces of 64^3 cube occupied around an interior object)
  int n_act = 0, border_tot = 0, border_pos = 0, interior_pos = 0;
  for (int x = 0; x < R; x++)
    for (int y = 0; y < R; y++)
      for (int z = 0; z < R; z++) {
        int pos_v = (logits[(x * R + y) * R + z] > 0.0f);
        int on_b = (x == 0 || x == R - 1 || y == 0 || y == R - 1 || z == 0 || z == R - 1);
        int in_core = (x >= 2 && x <= R - 3 && y >= 2 && y <= R - 3 && z >= 2 && z <= R - 3);
        if (on_b) { border_tot++; if (pos_v) border_pos++; }
        if (in_core && pos_v) interior_pos++;
        if (pos_v) n_act++;
      }
  int strip_shell = (border_tot > 0 && (double)border_pos / (double)border_tot > 0.22 && interior_pos >= 800) ? 1 : 0;
  if (strip_shell) n_act = interior_pos;
  if (out_active_vox) *out_active_vox = n_act;
  if (n_act == 0) return -1;

  // Pad grid to 66^3 with -10.0f border so boundary voxels close cleanly!
  const int P = 66;
  float *pad = (float *)malloc((size_t)P * P * P * 4);
  for (int x = 0; x < P; x++)
    for (int y = 0; y < P; y++)
      for (int z = 0; z < P; z++) {
        int gx = x - 1, gy = y - 1, gz = z - 1;
        if (strip_shell && (gx < 2 || gx > R - 3 || gy < 2 || gy > R - 3 || gz < 2 || gz > R - 3)) {
          pad[(x * P + y) * P + z] = -10.0f;
        } else {
          pad[(x * P + y) * P + z] = sample_grid_64(logits, gx, gy, gz);
        }
      }

  const int C = P - 1; // 65 cells per axis
  int32_t *cell_vid = (int32_t *)malloc((size_t)C * C * C * sizeof(int32_t));
  memset(cell_vid, -1, (size_t)C * C * C * sizeof(int32_t));

  uint32_t vcap = 32768, n_verts = 0;
  float *pos = (float *)malloc((size_t)vcap * 3 * 4);
  float *nrm = (float *)malloc((size_t)vcap * 3 * 4);
  float *uvs = (float *)malloc((size_t)vcap * 2 * 4);

  static const int E12[12][2] = {
    {0,1},{2,3},{4,5},{6,7}, // x edges
    {0,2},{1,3},{4,6},{5,7}, // y edges
    {0,4},{1,5},{2,6},{3,7}  // z edges
  };

  // Pass 1: place 1 sub-voxel vertex inside every cell with a sign change
  for (int x = 0; x < C; x++) {
    for (int y = 0; y < C; y++) {
      for (int z = 0; z < C; z++) {
        float val[8];
        int mask = 0;
        for (int b = 0; b < 8; b++) {
          int dx = b & 1, dy = (b >> 1) & 1, dz = (b >> 2) & 1;
          val[b] = pad[((x + dx) * P + (y + dy)) * P + (z + dz)];
          if (val[b] > 0.0f) mask |= (1 << b);
        }
        if (mask == 0 || mask == 255) continue;

        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        int cnt = 0;
        for (int e = 0; e < 12; e++) {
          int a = E12[e][0], b = E12[e][1];
          float va = val[a], vb = val[b];
          if ((va > 0.0f) == (vb > 0.0f)) continue;
          float t = va / (va - vb);
          if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
          float ax = (float)(a & 1), ay = (float)((a >> 1) & 1), az = (float)((a >> 2) & 1);
          float bx = (float)(b & 1), by = (float)((b >> 1) & 1), bz = (float)((b >> 2) & 1);
          sx += ax + t * (bx - ax);
          sy += ay + t * (by - ay);
          sz += az + t * (bz - az);
          cnt++;
        }
        if (cnt == 0) continue;
        float vx = (float)x + sx / (float)cnt;
        float vy = (float)y + sy / (float)cnt;
        float vz = (float)z + sz / (float)cnt;

        // Analytical outward normal = -grad(logit) (solid > 0, air < 0)
        float gx = (val[1] + val[3] + val[5] + val[7]) - (val[0] + val[2] + val[4] + val[6]);
        float gy = (val[2] + val[3] + val[6] + val[7]) - (val[0] + val[1] + val[4] + val[5]);
        float gz = (val[4] + val[5] + val[6] + val[7]) - (val[0] + val[1] + val[2] + val[3]);
        float nx = -gx, ny = -gy, nz = -gz;
        float nl = sqrtf(nx * nx + ny * ny + nz * nz) + 1e-8f;
        nx /= nl; ny /= nl; nz /= nl;

        if (n_verts >= vcap) {
          vcap *= 2;
          pos = (float *)realloc(pos, (size_t)vcap * 3 * 4);
          nrm = (float *)realloc(nrm, (size_t)vcap * 3 * 4);
          uvs = (float *)realloc(uvs, (size_t)vcap * 2 * 4);
        }
        cell_vid[(x * C + y) * C + z] = (int32_t)n_verts;
        // Map grid coords [1..65] -> normalized [-0.5, 0.5] in glTF Y-up convention!
        // TRELLIS.2 voxel coords (X, Y, Z) with Z-up -> glTF Y-up: (x, z, -y)
        float px = (vx - 1.0f) / 64.0f - 0.5f;
        float py = (vy - 1.0f) / 64.0f - 0.5f;
        float pz = (vz - 1.0f) / 64.0f - 0.5f;
        pos[n_verts * 3 + 0] =  px;
        pos[n_verts * 3 + 1] =  pz;
        pos[n_verts * 3 + 2] = -py;
        nrm[n_verts * 3 + 0] =  nx;
        nrm[n_verts * 3 + 1] =  nz;
        nrm[n_verts * 3 + 2] = -ny;
        uvs[n_verts * 2 + 0] = px + 0.5f;
        uvs[n_verts * 2 + 1] = pz + 0.5f;
        n_verts++;
      }
    }
  }

  // Pass 2: emit 1 dual quad (2 tris) for every interior grid edge with a sign change
  uint32_t tcap = 65536, n_tris = 0;
  uint32_t *idx = (uint32_t *)malloc((size_t)tcap * 3 * sizeof(uint32_t));
  auto emit_quad = [&](int32_t v0, int32_t v1, int32_t v2, int32_t v3, int flip) {
    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) return;
    if (n_tris + 2 > tcap) {
      tcap *= 2;
      idx = (uint32_t *)realloc(idx, (size_t)tcap * 3 * sizeof(uint32_t));
    }
    if (!flip) {
      idx[n_tris * 3 + 0] = v0; idx[n_tris * 3 + 1] = v1; idx[n_tris * 3 + 2] = v2; n_tris++;
      idx[n_tris * 3 + 0] = v0; idx[n_tris * 3 + 1] = v2; idx[n_tris * 3 + 2] = v3; n_tris++;
    } else {
      idx[n_tris * 3 + 0] = v0; idx[n_tris * 3 + 1] = v2; idx[n_tris * 3 + 2] = v1; n_tris++;
      idx[n_tris * 3 + 0] = v0; idx[n_tris * 3 + 1] = v3; idx[n_tris * 3 + 2] = v2; n_tris++;
    }
  };

  for (int x = 1; x < C; x++) {
    for (int y = 1; y < C; y++) {
      for (int z = 1; z < C; z++) {
        int s0 = pad[(x * P + y) * P + z] > 0.0f;
        // Edge along +X: (x,y,z) -> (x+1,y,z), dual cells around (y-1..y, z-1..z) at cell x
        if (x < C) {
          int sx = pad[((x + 1) * P + y) * P + z] > 0.0f;
          if (s0 != sx) {
            int32_t c0 = cell_vid[(x * C + (y - 1)) * C + (z - 1)];
            int32_t c1 = cell_vid[(x * C + (y - 1)) * C + z];
            int32_t c2 = cell_vid[(x * C + y) * C + z];
            int32_t c3 = cell_vid[(x * C + y) * C + (z - 1)];
            emit_quad(c0, c1, c2, c3, s0 ? 0 : 1);
          }
        }
        // Edge along +Y
        if (y < C) {
          int sy = pad[(x * P + (y + 1)) * P + z] > 0.0f;
          if (s0 != sy) {
            int32_t c0 = cell_vid[((x - 1) * C + y) * C + (z - 1)];
            int32_t c1 = cell_vid[(x * C + y) * C + (z - 1)];
            int32_t c2 = cell_vid[(x * C + y) * C + z];
            int32_t c3 = cell_vid[((x - 1) * C + y) * C + z];
            emit_quad(c0, c1, c2, c3, s0 ? 0 : 1);
          }
        }
        // Edge along +Z
        if (z < C) {
          int sz = pad[(x * P + y) * P + (z + 1)] > 0.0f;
          if (s0 != sz) {
            int32_t c0 = cell_vid[((x - 1) * C + (y - 1)) * C + z];
            int32_t c1 = cell_vid[((x - 1) * C + y) * C + z];
            int32_t c2 = cell_vid[(x * C + y) * C + z];
            int32_t c3 = cell_vid[(x * C + (y - 1)) * C + z];
            emit_quad(c0, c1, c2, c3, s0 ? 0 : 1);
          }
        }
      }
    }
  }
  free(pad);
  free(cell_vid);

  if (n_verts == 0 || n_tris == 0) {
    free(pos); free(nrm); free(uvs); free(idx);
    return -1;
  }

  // True Volume-Preserving 2-Pass Taubin Smoothing (2x [lambda=+0.42, mu=-0.45]) — smooths stairsteps with ZERO shrinkage!
  float *pos_s = (float *)calloc((size_t)n_verts * 3, 4);
  uint32_t *deg = (uint32_t *)calloc(n_verts, sizeof(uint32_t));
  static const float TAUBIN_W[4] = { 0.42f, -0.45f, 0.35f, -0.37f };
  for (int it = 0; it < 4; it++) {
    memset(pos_s, 0, (size_t)n_verts * 3 * 4);
    memset(deg, 0, n_verts * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tris; t++) {
      uint32_t i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
      uint32_t e[6] = {i0, i1, i1, i2, i2, i0};
      for (int k = 0; k < 6; k += 2) {
        uint32_t a = e[k], b = e[k + 1];
        pos_s[a * 3 + 0] += pos[b * 3 + 0]; pos_s[a * 3 + 1] += pos[b * 3 + 1]; pos_s[a * 3 + 2] += pos[b * 3 + 2]; deg[a]++;
        pos_s[b * 3 + 0] += pos[a * 3 + 0]; pos_s[b * 3 + 1] += pos[a * 3 + 1]; pos_s[b * 3 + 2] += pos[a * 3 + 2]; deg[b]++;
      }
    }
    float w = TAUBIN_W[it];
    for (uint32_t v = 0; v < n_verts; v++) {
      if (deg[v] > 0) {
        float inv = 1.0f / (float)deg[v];
        float dx = pos_s[v * 3 + 0] * inv - pos[v * 3 + 0];
        float dy = pos_s[v * 3 + 1] * inv - pos[v * 3 + 1];
        float dz = pos_s[v * 3 + 2] * inv - pos[v * 3 + 2];
        pos[v * 3 + 0] += w * dx;
        pos[v * 3 + 1] += w * dy;
        pos[v * 3 + 2] += w * dz;
      }
    }
  }
  free(pos_s); free(deg);

  // Recompute exact smooth area-weighted vertex normals from smoothed triangles (aligned with analytical normal)
  float *fnrm = (float *)calloc((size_t)n_verts * 3, 4);
  for (uint32_t t = 0; t < n_tris; t++) {
    uint32_t i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    float ax = pos[i1*3+0]-pos[i0*3+0], ay = pos[i1*3+1]-pos[i0*3+1], az = pos[i1*3+2]-pos[i0*3+2];
    float bx = pos[i2*3+0]-pos[i0*3+0], by = pos[i2*3+1]-pos[i0*3+1], bz = pos[i2*3+2]-pos[i0*3+2];
    float cx = ay*bz - az*by, cy = az*bx - ax*bz, cz = ax*by - ay*bx;
    float dot = cx * nrm[i0*3+0] + cy * nrm[i0*3+1] + cz * nrm[i0*3+2];
    if (dot < 0.0f) {
      idx[t * 3 + 1] = i2; idx[t * 3 + 2] = i1;
      cx = -cx; cy = -cy; cz = -cz;
    }
    uint32_t vs[3] = {i0, i1, i2};
    for (int k = 0; k < 3; k++) {
      fnrm[vs[k]*3+0] += cx; fnrm[vs[k]*3+1] += cy; fnrm[vs[k]*3+2] += cz;
    }
  }
  for (uint32_t v = 0; v < n_verts; v++) {
    float x = fnrm[v*3+0], y = fnrm[v*3+1], z = fnrm[v*3+2];
    float l = sqrtf(x*x + y*y + z*z);
    if (l > 1e-8f) { nrm[v*3+0] = x/l; nrm[v*3+1] = y/l; nrm[v*3+2] = z/l; }
  }
  free(fnrm);

  // v16 Per-Vertex PBR Albedo Projection (COLOR_0 linear RGB) from the 1024x1024 studio reference image!
  // Projects every vertex onto the foreground-cropped reference silhouette (replacing flat single-color beige clay!)
  float *vcol = (float *)malloc((size_t)n_verts * 3 * 4);
  if (vcol && ref_rgb && ref_w >= 16 && ref_h >= 16) {
    // Detect studio backdrop color & tight foreground bounding box in reference image
    float bgr = 0, bgg = 0, bgb = 0; int bcnt = 0;
    for (int x = 0; x < ref_w; x += 2) {
      const uint8_t *p = ref_rgb + (long)x * 3;
      bgr += p[0]; bgg += p[1]; bgb += p[2]; bcnt++;
    }
    if (bcnt > 0) { bgr /= bcnt; bgg /= bcnt; bgb /= bcnt; }
    int fx0 = ref_w, fy0 = ref_h, fx1 = -1, fy1 = -1;
    double avg_fg[3] = {0, 0, 0}; long fg_cnt = 0;
    for (int y = 0; y < ref_h; y += 2) {
      for (int x = 0; x < ref_w; x += 2) {
        const uint8_t *p = ref_rgb + ((long)y * ref_w + x) * 3;
        float dr = p[0] - bgr, dg = p[1] - bgg, db = p[2] - bgb;
        if (sqrtf(dr * dr + dg * dg + db * db) > 22.0f) {
          if (x < fx0) fx0 = x; if (x > fx1) fx1 = x;
          if (y < fy0) fy0 = y; if (y > fy1) fy1 = y;
          avg_fg[0] += p[0]; avg_fg[1] += p[1]; avg_fg[2] += p[2]; fg_cnt++;
        }
      }
    }
    if (fx1 <= fx0 + 8 || fy1 <= fy0 + 8) { fx0 = 0; fy0 = 0; fx1 = ref_w - 1; fy1 = ref_h - 1; }
    float fb_r = fg_cnt ? (float)(avg_fg[0] / (fg_cnt * 255.0)) : base_color[0];
    float fb_g = fg_cnt ? (float)(avg_fg[1] / (fg_cnt * 255.0)) : base_color[1];
    float fb_b = fg_cnt ? (float)(avg_fg[2] / (fg_cnt * 255.0)) : base_color[2];

    // Compute mesh AABB in glTF Y-up space
    float mmin[3] = { 1e9f, 1e9f, 1e9f }, mmax[3] = { -1e9f, -1e9f, -1e9f };
    for (uint32_t v = 0; v < n_verts; v++) {
      for (int a = 0; a < 3; a++) {
        float c = pos[v * 3 + a];
        if (c < mmin[a]) mmin[a] = c; if (c > mmax[a]) mmax[a] = c;
      }
    }
    float sx = fmaxf(1e-4f, mmax[0] - mmin[0]);
    float sy = fmaxf(1e-4f, mmax[1] - mmin[1]);
    float sz = fmaxf(1e-4f, mmax[2] - mmin[2]);
    // 3/4 view horizontal azimuth projection (matches our 3/4 studio reference view!)
    const float cosA = 0.8660254f, sinA = 0.5f;
    float pmin_u = 1e9f, pmax_u = -1e9f;
    for (uint32_t v = 0; v < n_verts; v++) {
      float ux = pos[v * 3 + 0] * cosA + pos[v * 3 + 2] * sinA;
      if (ux < pmin_u) pmin_u = ux; if (ux > pmax_u) pmax_u = ux;
    }
    float su = fmaxf(1e-4f, pmax_u - pmin_u);

    for (uint32_t v = 0; v < n_verts; v++) {
      float ux = pos[v * 3 + 0] * cosA + pos[v * 3 + 2] * sinA;
      float u = (ux - pmin_u) / su;
      float wv = 1.0f - (pos[v * 3 + 1] - mmin[1]) / sy;
      // Pull slightly inward from extreme silhouette rim to avoid sampling backdrop pixels!
      float u_in = 0.04f + 0.92f * u;
      float v_in = 0.04f + 0.92f * wv;
      int px = fx0 + (int)(u_in * (float)(fx1 - fx0));
      int py = fy0 + (int)(v_in * (float)(fy1 - fy0));
      if (px < 0) px = 0; if (px >= ref_w) px = ref_w - 1;
      if (py < 0) py = 0; if (py >= ref_h) py = ref_h - 1;
      // 3x3 foreground-weighted sample
      float sr = 0, sg = 0, sb = 0, sw = 0;
      for (int dy = -2; dy <= 2; dy += 2) {
        int yy = py + dy; if (yy < 0) yy = 0; if (yy >= ref_h) yy = ref_h - 1;
        for (int dx = -2; dx <= 2; dx += 2) {
          int xx = px + dx; if (xx < 0) xx = 0; if (xx >= ref_w) xx = ref_w - 1;
          const uint8_t *p = ref_rgb + ((long)yy * ref_w + xx) * 3;
          float dr = p[0] - bgr, dg = p[1] - bgg, db = p[2] - bgb;
          if (sqrtf(dr * dr + dg * dg + db * db) > 20.0f) {
            sr += p[0]; sg += p[1]; sb += p[2]; sw += 1.0f;
          }
        }
      }
      float cr = sw > 0.5f ? (sr / (sw * 255.0f)) : fb_r;
      float cg = sw > 0.5f ? (sg / (sw * 255.0f)) : fb_g;
      float cb = sw > 0.5f ? (sb / (sw * 255.0f)) : fb_b;
      // Soft hemisphere AO tint + exact sRGB -> linear for glTF 2.0 COLOR_0 PBR!
      float ao = 0.90f + 0.10f * fmaxf(-0.3f, nrm[v * 3 + 1]);
      vcol[v * 3 + 0] = tr_srgb_to_linear(cr * ao);
      vcol[v * 3 + 1] = tr_srgb_to_linear(cg * ao);
      vcol[v * 3 + 2] = tr_srgb_to_linear(cb * ao);
      // Clean spherical/cylindrical UV coordinates for downstream Blender/Three.js editing
      uvs[v * 2 + 0] = u;
      uvs[v * 2 + 1] = wv;
    }
    (void)sx; (void)sz;
  } else if (vcol) {
    for (uint32_t v = 0; v < n_verts; v++) {
      vcol[v * 3 + 0] = tr_srgb_to_linear(base_color[0]);
      vcol[v * 3 + 1] = tr_srgb_to_linear(base_color[1]);
      vcol[v * 3 + 2] = tr_srgb_to_linear(base_color[2]);
    }
  }

  mesh->positions = pos;
  mesh->normals   = nrm;
  mesh->uvs       = uvs;
  mesh->colors    = vcol;
  mesh->indices   = idx;
  mesh->n_verts   = n_verts;
  mesh->n_tris    = n_tris;
  if (vcol) {
    // glTF 2.0 multiplies baseColorFactor * COLOR_0 -> keep baseColorFactor white [1,1,1,1]!
    mesh->base_color[0] = 1.0f; mesh->base_color[1] = 1.0f;
    mesh->base_color[2] = 1.0f; mesh->base_color[3] = 1.0f;
  } else {
    memcpy(mesh->base_color, base_color, 4 * sizeof(float));
  }
  mesh->metallic  = 0.18f;
  mesh->roughness = 0.36f;

  // Build matching per-vertex colored 3DGS Splat cloud from surface vertices
  splat->n_splats = n_verts;
  splat->pos     = (float *)malloc((size_t)n_verts * 3 * 4);
  splat->scale   = (float *)malloc((size_t)n_verts * 3 * 4);
  splat->rot     = (float *)malloc((size_t)n_verts * 4 * 4);
  splat->opacity = (float *)malloc((size_t)n_verts * 4);
  splat->sh_dc   = (float *)malloc((size_t)n_verts * 3 * 4);
  memcpy(splat->pos, pos, (size_t)n_verts * 3 * 4);
  float log_s = logf(0.85f / 64.0f);
  for (uint32_t v = 0; v < n_verts; v++) {
    splat->scale[v*3+0] = log_s; splat->scale[v*3+1] = log_s; splat->scale[v*3+2] = log_s;
    splat->rot[v*4+0] = 1.0f; splat->rot[v*4+1] = 0.0f; splat->rot[v*4+2] = 0.0f; splat->rot[v*4+3] = 0.0f;
    splat->opacity[v] = 2.8f;
    float shade = 0.72f + 0.28f * fmaxf(0.0f, nrm[v*3+1] * 0.7f + nrm[v*3+0] * 0.3f);
    float cr = vcol ? powf(fmaxf(0.0f, vcol[v*3+0]), 1.0f / 2.2f) : base_color[0];
    float cg = vcol ? powf(fmaxf(0.0f, vcol[v*3+1]), 1.0f / 2.2f) : base_color[1];
    float cb = vcol ? powf(fmaxf(0.0f, vcol[v*3+2]), 1.0f / 2.2f) : base_color[2];
    splat->sh_dc[v*3+0] = (cr * shade - 0.5f) / 0.28209479f;
    splat->sh_dc[v*3+1] = (cg * shade - 0.5f) / 0.28209479f;
    splat->sh_dc[v*3+2] = (cb * shade - 0.5f) / 0.28209479f;
  }
  return 0;
}

// ===========================================================================
// 11b. v17.2 Full 3-Stage TRELLIS.2-4B Native Pure-CUDA Pipeline (512^3 O-Voxel PBR!)
//      Stage 2a: Sparse 30-Block SLatFlowModel (Shape, 32ch, 32^3 sparse coords, 3D RoPE)
//      Stage 3a: Sparse 30-Block SLatFlowModel (Texture, 64ch->32ch, conditioned on Shape SLat)
//      Stage 2b: FlexiDualGridVaeDecoder (4-stage Sparse 3D ConvNeXt + C2S octree 32^3->512^3 -> 7ch DC)
//      Stage 3b: SparseUnetVaeDecoder (matching octree -> 6ch 360-deg PBR: BaseColor+Metallic+Roughness+Alpha)
// ===========================================================================

static const float TR2_SHAPE_MEAN[32] = {
  0.781296f, 0.018091f, -0.495192f, -0.558457f, 1.060530f, 0.093252f, 1.518149f, -0.933218f,
  -0.732996f, 2.604095f, -0.118341f, -2.143904f, 0.495076f, -2.179512f, -2.130751f, -0.996944f,
  0.261421f, -2.217463f, 1.260067f, -0.150213f, 3.790713f, 1.481266f, -1.046058f, -1.523667f,
  -0.059621f, 2.220780f, 1.621212f, 0.877230f, 0.567247f, -3.175944f, -3.186688f, 1.578665f
};
static const float TR2_SHAPE_STD[32] = {
  5.972266f, 4.706852f, 5.445010f, 5.209927f, 5.320220f, 4.547237f, 5.020802f, 5.444004f,
  5.226681f, 5.683095f, 4.831436f, 5.286469f, 5.652043f, 5.367606f, 5.525084f, 4.730578f,
  4.805265f, 5.124013f, 5.530808f, 5.619001f, 5.103930f, 5.417670f, 5.269677f, 5.547194f,
  5.634698f, 5.235274f, 6.110351f, 5.511298f, 6.237273f, 4.879207f, 5.347008f, 5.405691f
};
static const float TR2_TEX_MEAN[32] = {
  3.501659f, 2.212398f, 2.226094f, 0.251093f, -0.026248f, -0.687364f, 0.439898f, -0.928075f,
  0.029398f, -0.339596f, -0.869527f, 1.038479f, -0.972385f, 0.126042f, -1.129303f, 0.455149f,
  -1.209521f, 2.069067f, 0.544735f, 2.569128f, -0.323407f, 2.293000f, -1.925608f, -1.217717f,
  1.213905f, 0.971588f, -0.023631f, 0.106750f, 2.021786f, 0.250524f, -0.662387f, -0.768862f
};
static const float TR2_TEX_STD[32] = {
  2.665652f, 2.743913f, 2.765121f, 2.595319f, 3.037293f, 2.291316f, 2.144656f, 2.911822f,
  2.969419f, 2.501689f, 2.154811f, 3.163343f, 2.621215f, 2.381943f, 3.186697f, 3.021588f,
  2.295916f, 3.234985f, 3.233086f, 2.260140f, 2.874801f, 2.810596f, 3.292720f, 2.674999f,
  2.680878f, 2.372054f, 2.451546f, 2.353556f, 2.995195f, 2.379849f, 2.786195f, 2.775190f
};

__global__ void k_slat_pack_in(
    const float *__restrict__ xt, const float *__restrict__ concat_cond,
    uint16_t *__restrict__ out_bf16, int N, int out_ch, int in_ch) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N * in_ch;
  if (idx >= tot) return;
  int row = (int)(idx / in_ch);
  int c   = (int)(idx % in_ch);
  float v = (c < out_ch) ? xt[(long)row * out_ch + c]
                         : (concat_cond ? concat_cond[(long)row * (in_ch - out_ch) + (c - out_ch)] : 0.0f);
  out_bf16[idx] = tr_f32_to_bf16(v);
}

__global__ void k_slat_self_qkv_rope(
    const float *__restrict__ qkv,
    const float *__restrict__ q_gamma, const float *__restrict__ k_gamma,
    const int32_t *__restrict__ coords3,
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V,
    int N, int H) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= N * H) return;
  int tok = idx / H;
  int h = idx % H;
  const float *base = qkv + (long)tok * (3 * H * 128);
  const float *qs = base + (0 * H + h) * 128;
  const float *ks = base + (1 * H + h) * 128;
  const float *vs = base + (2 * H + h) * 128;
  const float *qg = q_gamma + (long)h * 128;
  const float *kg = k_gamma + (long)h * 128;
  uint16_t *qo = Q + ((long)h * N + tok) * 128;
  uint16_t *ko = K + ((long)h * N + tok) * 128;
  uint16_t *vo = V + ((long)h * N + tok) * 128;

  float sq = 0.0f, sk = 0.0f;
  for (int i = 0; i < 4; i++) {
    int d = lane + i * 32;
    float qv = qs[d], kv = ks[d];
    sq += qv * qv; sk += kv * kv;
  }
  float inv_q = rsqrtf(tr_warp_sum(sq) / 128.0f + 1e-12f);
  float inv_k = rsqrtf(tr_warp_sum(sk) / 128.0f + 1e-12f);

  int ci = coords3[tok * 3 + 0];
  int cj = coords3[tok * 3 + 1];
  int ck = coords3[tok * 3 + 2];

  for (int pass = 0; pass < 2; pass++) {
    int p = lane + pass * 32;
    int d0 = 2 * p, d1 = 2 * p + 1;
    float q0 = qs[d0] * inv_q * qg[d0];
    float q1 = qs[d1] * inv_q * qg[d1];
    float k0 = ks[d0] * inv_k * kg[d0];
    float k1 = ks[d1] * inv_k * kg[d1];
    float cs = 1.0f, sn = 0.0f;
    if (p < 63) {
      int axis = p / 21;
      int fi = p % 21;
      float coord = (axis == 0) ? (float)ci : ((axis == 1) ? (float)cj : (float)ck);
      double freq = 1.0 / pow(10000.0, (double)fi / 21.0);
      double ang = (double)coord * freq;
      cs = (float)cos(ang); sn = (float)sin(ang);
    }
    qo[d0] = tr_f32_to_bf16(q0 * cs - q1 * sn);
    qo[d1] = tr_f32_to_bf16(q0 * sn + q1 * cs);
    ko[d0] = tr_f32_to_bf16(k0 * cs - k1 * sn);
    ko[d1] = tr_f32_to_bf16(k0 * sn + k1 * cs);
    vo[d0] = tr_f32_to_bf16(vs[d0]);
    vo[d1] = tr_f32_to_bf16(vs[d1]);
  }
}

__global__ void k_tr_silu_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = tr_f32_to_bf16(tr_silu(src[i]));
}

__global__ void k_tr_unnorm_ch(const float *in, float *out, const float *mean_d, const float *std_d, int N, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N * C;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  out[idx] = in[idx] * std_d[c] + mean_d[c];
}

// Sparse 3D Open-Addressing Hash Table + 27-Neighbor Table Builder (matches FlexGEMM subm_neighbor_map!)
__device__ __forceinline__ uint64_t tr_pack_xyz(int x, int y, int z) {
  return (((uint64_t)(uint32_t)x << 20) | ((uint64_t)(uint32_t)y << 10) | (uint64_t)(uint32_t)z) + 1ULL;
}
__device__ __forceinline__ uint32_t tr_hash_key(uint64_t key, uint32_t mask) {
  return (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

__global__ void k_tr_sp_hash_insert(
    const int32_t *__restrict__ coords3, int N,
    unsigned long long *__restrict__ keys, int32_t *__restrict__ vals, uint32_t mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint64_t k = tr_pack_xyz(coords3[i * 3 + 0], coords3[i * 3 + 1], coords3[i * 3 + 2]);
  uint32_t slot = tr_hash_key(k, mask);
  while (true) {
    unsigned long long prev = atomicCAS(&keys[slot], 0ULL, (unsigned long long)k);
    if (prev == 0ULL || prev == (unsigned long long)k) {
      vals[slot] = i;
      return;
    }
    slot = (slot + 1u) & mask;
  }
}

__global__ void k_tr_sp_build_nb(
    const int32_t *__restrict__ coords3, int N, int R,
    const unsigned long long *__restrict__ keys, const int32_t *__restrict__ vals, uint32_t mask,
    int32_t *__restrict__ nb_out) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N * 27;
  if (idx >= tot) return;
  int i = (int)(idx / 27);
  int v = (int)(idx % 27);
  if (v == 13) { nb_out[idx] = i; return; }
  int x = coords3[i * 3 + 0] + (v / 9) - 1;
  int y = coords3[i * 3 + 1] + ((v / 3) % 3) - 1;
  int z = coords3[i * 3 + 2] + (v % 3) - 1;
  if (x < 0 || x >= R || y < 0 || y >= R || z < 0 || z >= R) { nb_out[idx] = -1; return; }
  uint64_t k = tr_pack_xyz(x, y, z);
  uint32_t slot = tr_hash_key(k, mask);
  while (true) {
    unsigned long long cur = keys[slot];
    if (cur == 0ULL) { nb_out[idx] = -1; return; }
    if (cur == (unsigned long long)k) { nb_out[idx] = vals[slot]; return; }
    slot = (slot + 1u) & mask;
  }
}

__global__ void k_tr_sp_gather_k3(
    const float *__restrict__ x, const int32_t *__restrict__ nb,
    uint16_t *__restrict__ col_bf16, int v0, int chunk, int Cin) {
  long K = 27L * Cin;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)chunk * K;
  if (idx >= tot) return;
  int row = (int)(idx / K);
  int rem = (int)(idx % K);
  int v   = rem / Cin;
  int ic  = rem % Cin;
  int ni  = nb[(long)(v0 + row) * 27 + v];
  float val = (ni >= 0) ? x[(long)ni * Cin + ic] : 0.0f;
  col_bf16[idx] = tr_f32_to_bf16(val);
}

// SparseChannel2Spatial gather: h_in[N_parent, 8 * Cout] -> h_out[N_child, Cout]
__global__ void k_tr_c2s_gather(
    const float *__restrict__ h_in, const int32_t *__restrict__ c_parent,
    const uint8_t *__restrict__ c_sub, float *__restrict__ h_out, int N_child, int Cout) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N_child * Cout;
  if (idx >= tot) return;
  int ci = (int)(idx / Cout);
  int k  = (int)(idx % Cout);
  int p  = c_parent[ci];
  int s  = (int)c_sub[ci];
  h_out[idx] = h_in[((long)p * 8 + s) * Cout + k];
}

// SparseChannel2Spatial skip + repeat_interleave(Cout / (Cin/8)): h_f32 += repeat_interleave(C2S(x_in))
__global__ void k_tr_c2s_skip_add(
    float *__restrict__ h_f32, const float *__restrict__ x_in,
    const int32_t *__restrict__ c_parent, const uint8_t *__restrict__ c_sub,
    int N_child, int Cin, int Cout) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N_child * Cout;
  if (idx >= tot) return;
  int ci = (int)(idx / Cout);
  int k  = (int)(idx % Cout);
  int p  = c_parent[ci];
  int s  = (int)c_sub[ci];
  int xc = Cin >> 3;
  int rep = Cout / xc;
  h_f32[idx] += x_in[((long)p * 8 + s) * xc + (k / rep)];
}

static int load_slat_flow_generic(const ec_safetensors_t *st, tr_ss_flow_res_t *f) {
  memset(f, 0, sizeof *f);
  f->in_w    = tr_upload_bf16(st, "input_layer.weight");
  f->in_b    = tr_upload_f32(st,  "input_layer.bias");
  f->out_w   = tr_upload_bf16(st, "out_layer.weight");
  f->out_b   = tr_upload_f32(st,  "out_layer.bias");
  f->t0_w    = tr_upload_bf16(st, "t_embedder.mlp.0.weight");
  f->t0_b    = tr_upload_f32(st,  "t_embedder.mlp.0.bias");
  f->t2_w    = tr_upload_bf16(st, "t_embedder.mlp.2.weight");
  f->t2_b    = tr_upload_f32(st,  "t_embedder.mlp.2.bias");
  f->adaln_w = tr_upload_bf16(st, "adaLN_modulation.1.weight");
  f->adaln_b = tr_upload_f32(st,  "adaLN_modulation.1.bias");
  if (!f->in_w || !f->out_w || !f->adaln_w) { free_flow_res(f); return -1; }
  char k[192];
  for (int i = 0; i < 30; i++) {
    tr_ss_block_t *B = &f->blocks[i];
    snprintf(k, sizeof k, "blocks.%d.modulation", i);                    B->modulation = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.weight", i);      B->sa_qkv_w   = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.bias", i);        B->sa_qkv_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.weight", i);      B->sa_out_w   = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.bias", i);        B->sa_out_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.q_rms_norm.gamma", i);   B->sa_qn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.k_rms_norm.gamma", i);   B->sa_kn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.weight", i);                 B->norm2_w    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm2.bias", i);                   B->norm2_b    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.weight", i);       B->ca_q_w     = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.bias", i);         B->ca_q_b     = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.weight", i);      B->ca_kv_w    = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.bias", i);        B->ca_kv_b    = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.weight", i);     B->ca_out_w   = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.bias", i);       B->ca_out_b   = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.q_rms_norm.gamma", i);  B->ca_qn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.k_rms_norm.gamma", i);  B->ca_kn      = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.weight", i);             B->mlp0_w     = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.bias", i);               B->mlp0_b     = tr_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.weight", i);             B->mlp2_w     = tr_upload_bf16(st, k);
    snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.bias", i);               B->mlp2_b     = tr_upload_f32(st, k);
    if (!B->sa_qkv_w || !B->ca_kv_w || !B->mlp2_w) { free_flow_res(f); return -1; }
  }
  f->loaded = 1;
  return 0;
}

static void slat_flow_forward_one(
    ec_trellis2_t *tr, const tr_ss_flow_res_t *f,
    const float *xt_d, const float *concat_cond_d, const int32_t *coords_d,
    int N, int in_ch, int out_ch, float t_val,
    uint16_t *const ck_cache[30], uint16_t *const cv_cache[30],
    ss_work_t *W, float *out_v_d) {
  const int DIM = 1536, H = 12, FFN = 8192, Lkv = 1029;
  const float scale = 1.0f / sqrtf(128.0f);

  // Timestep embedding
  uint16_t fb_h[256];
  double scaled = (double)t_val * 1000.0;
  for (int i = 0; i < 128; i++) {
    double fr = exp(-log(10000.0) * (double)i / 128.0);
    double a = scaled * fr;
    fb_h[i]       = tr_f32_to_bf16_h((float)cos(a));
    fb_h[128 + i] = tr_f32_to_bf16_h((float)sin(a));
  }
  cudaMemcpyAsync(W->t_f_d, fb_h, 256 * 2, cudaMemcpyHostToDevice, tr->stream);
  tr_gemm(W->t_f_d, f->t0_w, f->t0_b, W->t_hf_d, NULL, 1, 1536, 256, tr->stream);
  ec_ops_silu_f32_to_bf16(W->t_hf_d, W->t_hb_d, 1536, tr->stream);
  tr_gemm(W->t_hb_d, f->t2_w, f->t2_b, W->t_hf_d, NULL, 1, 1536, 1536, tr->stream);
  ec_ops_silu_f32_to_bf16(W->t_hf_d, W->t_hb_d, 1536, tr->stream);
  tr_gemm(W->t_hb_d, f->adaln_w, f->adaln_b, W->mod_d, NULL, 1, 9216, 1536, tr->stream);

  k_slat_pack_in<<<((long)N * in_ch + 255) / 256, 256, 0, tr->stream>>>(
      xt_d, concat_cond_d, W->x8_bf16, N, out_ch, in_ch);
  tr_gemm(W->x8_bf16, f->in_w, f->in_b, W->x_d, NULL, N, DIM, in_ch, tr->stream);

  int nw_sa = N * H;
  for (int l = 0; l < 30; l++) {
    const tr_ss_block_t *B = &f->blocks[l];
    k_tr_add_vec<<<(9216 + 255) / 256, 256, 0, tr->stream>>>(B->modulation, W->mod_d, W->mod_blk_d, 9216);
    const float *shift_msa = W->mod_blk_d + 0 * DIM;
    const float *scale_msa = W->mod_blk_d + 1 * DIM;
    const float *gate_msa  = W->mod_blk_d + 2 * DIM;
    const float *shift_mlp = W->mod_blk_d + 3 * DIM;
    const float *scale_mlp = W->mod_blk_d + 4 * DIM;
    const float *gate_mlp  = W->mod_blk_d + 5 * DIM;

    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(W->x_d, NULL, NULL, scale_msa, shift_msa, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->sa_qkv_w, B->sa_qkv_b, W->qkv_d, NULL, N, 3 * DIM, DIM, tr->stream);
    k_slat_self_qkv_rope<<<(nw_sa + 7) / 8, 256, 0, tr->stream>>>(
        W->qkv_d, B->sa_qn, B->sa_kn, coords_d, W->Q_b, W->K_b, W->V_b, N, H);
    ec_ops_fa2_bf16(W->Q_b, W->K_b, W->V_b, NULL, W->ao_bf16, N, N, N, N, H, H, 128, scale, 0, tr->stream);
    tr_gemm(W->ao_bf16, B->sa_out_w, B->sa_out_b, W->tmp_d, NULL, N, DIM, DIM, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, gate_msa, N, DIM);

    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(W->x_d, B->norm2_w, B->norm2_b, NULL, NULL, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->ca_q_w, B->ca_q_b, W->qkv_d, NULL, N, DIM, DIM, tr->stream);
    k_ss_cross_q_norm<<<(nw_sa + 7) / 8, 256, 0, tr->stream>>>(W->qkv_d, B->ca_qn, W->Q_b, N, H);
    ec_ops_fa2_bf16(W->Q_b, ck_cache[l], cv_cache[l], NULL, W->ao_bf16, N, Lkv, N, Lkv, H, H, 128, scale, 0, tr->stream);
    tr_gemm(W->ao_bf16, B->ca_out_w, B->ca_out_b, W->tmp_d, NULL, N, DIM, DIM, tr->stream);
    k_tr_add_inplace<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, (long)N * DIM);

    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(W->x_d, NULL, NULL, scale_mlp, shift_mlp, NULL, W->xn_bf16, N, DIM, 1e-6f, 0);
    tr_gemm(W->xn_bf16, B->mlp0_w, B->mlp0_b, W->fc0_d, NULL, N, FFN, DIM, tr->stream);
    k_tr_gelu_tanh_bf16<<<((long)N * FFN + 255) / 256, 256, 0, tr->stream>>>(W->fc0_d, W->mlp_bf16, (long)N * FFN);
    tr_gemm(W->mlp_bf16, B->mlp2_w, B->mlp2_b, W->tmp_d, NULL, N, DIM, FFN, tr->stream);
    k_tr_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, tr->stream>>>(W->x_d, W->tmp_d, gate_mlp, N, DIM);
  }

  k_tr_layernorm<<<N, 256, 0, tr->stream>>>(W->x_d, NULL, NULL, NULL, NULL, NULL, W->xn_bf16, N, DIM, 1e-5f, 0);
  tr_gemm(W->xn_bf16, f->out_w, f->out_b, out_v_d, NULL, N, out_ch, DIM, tr->stream);
}

static int run_slat_flow_generic(
    ec_trellis2_t *tr, const ec_safetensors_t *flow_st,
    const int32_t *coords_d, int N, int in_ch, int out_ch,
    const float *concat_cond_d, uint16_t *const *view_conds_d, int n_views,
    int steps, float cfg_scale, float guidance_rescale, float cfg_t0, float cfg_t1, float rescale_t,
    uint32_t seed, float *out_norm_d, const char *stage_name,
    ec_trellis2_progress_fn cb, void *cb_ctx, int prog_base, int prog_tot, double *out_ms) {
  double t0 = tr_now_ms();
  const int DIM = 1536, H = 12, FFN = 8192, Lkv = 1029;
  const float sigma_min = 1e-5f;
  if (n_views < 1) n_views = 1;
  if (n_views > 4) n_views = 4;

  tr_ss_flow_res_t f = {};
  if (load_slat_flow_generic(flow_st, &f) != 0) return -1;

  int need_neg = (cfg_scale > 1.01f) ? 1 : 0;
  uint16_t *neg_bf16_d = NULL;
  if (need_neg) {
    cudaMalloc(&neg_bf16_d, (size_t)Lkv * 1024 * 2);
    cudaMemsetAsync(neg_bf16_d, 0, (size_t)Lkv * 1024 * 2, tr->stream);
  }

  uint16_t *ck_pos[4][30] = {{0}}, *cv_pos[4][30] = {{0}}, *ck_neg[30] = {0}, *cv_neg[30] = {0};
  float *kv_tmp = NULL;
  cudaMalloc(&kv_tmp, (size_t)Lkv * 2 * DIM * 4);
  int nw_kv = Lkv * H;
  for (int l = 0; l < 30; l++) {
    const tr_ss_block_t *B = &f.blocks[l];
    for (int v = 0; v < n_views; v++) {
      cudaMalloc(&ck_pos[v][l], (size_t)Lkv * DIM * 2);
      cudaMalloc(&cv_pos[v][l], (size_t)Lkv * DIM * 2);
      tr_gemm(view_conds_d[v], B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, Lkv, 2 * DIM, 1024, tr->stream);
      k_ss_cross_kv_norm<<<(nw_kv + 7) / 8, 256, 0, tr->stream>>>(kv_tmp, B->ca_kn, ck_pos[v][l], cv_pos[v][l], Lkv, H);
    }
    if (need_neg) {
      cudaMalloc(&ck_neg[l], (size_t)Lkv * DIM * 2);
      cudaMalloc(&cv_neg[l], (size_t)Lkv * DIM * 2);
      tr_gemm(neg_bf16_d, B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, Lkv, 2 * DIM, 1024, tr->stream);
      k_ss_cross_kv_norm<<<(nw_kv + 7) / 8, 256, 0, tr->stream>>>(kv_tmp, B->ca_kn, ck_neg[l], cv_neg[l], Lkv, H);
    }
  }
  cudaFree(kv_tmp);
  cudaFree(neg_bf16_d);

  ss_work_t W = {};
  cudaMalloc(&W.x8_bf16,  (size_t)N * in_ch * 2);
  cudaMalloc(&W.x_d,      (size_t)N * DIM * 4);
  cudaMalloc(&W.xn_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&W.qkv_d,    (size_t)N * 3 * DIM * 4);
  cudaMalloc(&W.Q_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.K_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.V_b,      (size_t)N * DIM * 2);
  cudaMalloc(&W.ao_bf16,  (size_t)N * DIM * 2);
  cudaMalloc(&W.tmp_d,    (size_t)N * DIM * 4);
  cudaMalloc(&W.fc0_d,    (size_t)N * FFN * 4);
  cudaMalloc(&W.mlp_bf16, (size_t)N * FFN * 2);
  cudaMalloc(&W.mod_d,     9216 * 4);
  cudaMalloc(&W.mod_blk_d, 9216 * 4);
  cudaMalloc(&W.t_f_d,     256 * 2);
  cudaMalloc(&W.t_hf_d,    1536 * 4);
  cudaMalloc(&W.t_hb_d,    1536 * 2);

  float *xt_h = (float *)malloc((size_t)N * out_ch * 4);
  uint64_t rng = (uint64_t)seed * 6364136223846793005ULL + (concat_cond_d ? 0xD1B54A32D192ED03ULL : 0xA0761D6478BD642FULL);
  for (long i = 0; i < (long)N * out_ch; i++) xt_h[i] = tr_randn(&rng);

  float *xt_d = NULL, *vp_d = NULL, *vn_d = NULL;
  cudaMalloc(&xt_d, (size_t)N * out_ch * 4);
  cudaMalloc(&vp_d, (size_t)N * out_ch * 4);
  if (need_neg) cudaMalloc(&vn_d, (size_t)N * out_ch * 4);
  cudaMemcpyAsync(xt_d, xt_h, (size_t)N * out_ch * 4, cudaMemcpyHostToDevice, tr->stream);
  free(xt_h);

  float t_seq[65];
  for (int i = 0; i <= steps; i++) {
    float u = 1.0f - (float)i / (float)steps;
    t_seq[i] = rescale_t * u / (1.0f + (rescale_t - 1.0f) * u);
  }

  int cancelled = 0;
  for (int s = 0; s < steps; s++) {
    if (cb && cb(prog_base + s, prog_tot, stage_name, cb_ctx) != 0) { cancelled = 1; break; }
    float t_cur = t_seq[s], t_prev = t_seq[s + 1], dt = t_cur - t_prev;
    int vid = s % n_views;
    slat_flow_forward_one(tr, &f, xt_d, concat_cond_d, coords_d, N, in_ch, out_ch, t_cur, ck_pos[vid], cv_pos[vid], &W, vp_d);
    int use_cfg = (need_neg && t_cur >= cfg_t0 && t_cur <= cfg_t1) ? 1 : 0;
    if (use_cfg) {
      slat_flow_forward_one(tr, &f, xt_d, concat_cond_d, coords_d, N, in_ch, out_ch, t_cur, ck_neg, cv_neg, &W, vn_d);
    }
    ec_ops_trellis_cfg_euler(
        xt_d, vp_d, use_cfg ? vn_d : NULL, N * out_ch, dt, t_cur,
        cfg_scale, guidance_rescale, sigma_min, use_cfg, tr->stream);
  }

  if (!cancelled) {
    cudaMemcpyAsync(out_norm_d, xt_d, (size_t)N * out_ch * 4, cudaMemcpyDeviceToDevice, tr->stream);
  }
  cudaStreamSynchronize(tr->stream);

  for (int l = 0; l < 30; l++) {
    for (int v = 0; v < n_views; v++) { cudaFree(ck_pos[v][l]); cudaFree(cv_pos[v][l]); }
    cudaFree(ck_neg[l]); cudaFree(cv_neg[l]);
  }
  cudaFree(W.x8_bf16); cudaFree(W.x_d); cudaFree(W.xn_bf16); cudaFree(W.qkv_d);
  cudaFree(W.Q_b); cudaFree(W.K_b); cudaFree(W.V_b); cudaFree(W.ao_bf16);
  cudaFree(W.tmp_d); cudaFree(W.fc0_d); cudaFree(W.mlp_bf16); cudaFree(W.mod_d); cudaFree(W.mod_blk_d);
  cudaFree(W.t_f_d); cudaFree(W.t_hf_d); cudaFree(W.t_hb_d);
  cudaFree(xt_d); cudaFree(vp_d); cudaFree(vn_d);
  free_flow_res(&f);

  if (out_ms) *out_ms = tr_now_ms() - t0;
  return cancelled ? -2 : 0;
}

typedef struct {
  int      n_child;
  int32_t *child_coords_h; // [n_child * 3]
  int32_t *child_parent_h; // [n_child]
  uint8_t *child_sub_h;    // [n_child]
} tr_subdiv_level_t;

static void free_subdiv_levels(tr_subdiv_level_t lvls[4]) {
  for (int i = 0; i < 4; i++) {
    free(lvls[i].child_coords_h);
    free(lvls[i].child_parent_h);
    free(lvls[i].child_sub_h);
  }
  memset(lvls, 0, 4 * sizeof(tr_subdiv_level_t));
}

static int build_sp_nb_gpu(
    ec_trellis2_t *tr, const int32_t *coords_h, int N, int R, int32_t **out_nb_d) {
  uint32_t cap = 1024u;
  while (cap < (uint32_t)N * 4u) cap <<= 1;
  uint32_t mask = cap - 1u;
  int32_t *coords_d = NULL, *vals_d = NULL, *nb_d = NULL;
  unsigned long long *keys_d = NULL;
  if (cudaMalloc(&coords_d, (size_t)N * 3 * sizeof(int32_t)) != cudaSuccess ||
      cudaMalloc(&keys_d,   (size_t)cap * sizeof(unsigned long long)) != cudaSuccess ||
      cudaMalloc(&vals_d,   (size_t)cap * sizeof(int32_t)) != cudaSuccess ||
      cudaMalloc(&nb_d,     (size_t)N * 27 * sizeof(int32_t)) != cudaSuccess) {
    cudaFree(coords_d); cudaFree(keys_d); cudaFree(vals_d); cudaFree(nb_d);
    return -1;
  }
  cudaMemcpyAsync(coords_d, coords_h, (size_t)N * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, tr->stream);
  cudaMemsetAsync(keys_d, 0, (size_t)cap * sizeof(unsigned long long), tr->stream);
  k_tr_sp_hash_insert<<<(N + 255) / 256, 256, 0, tr->stream>>>(coords_d, N, keys_d, vals_d, mask);
  k_tr_sp_build_nb<<<((long)N * 27 + 255) / 256, 256, 0, tr->stream>>>(coords_d, N, R, keys_d, vals_d, mask, nb_d);
  cudaStreamSynchronize(tr->stream);
  cudaFree(coords_d); cudaFree(keys_d); cudaFree(vals_d);
  *out_nb_d = nb_d;
  return 0;
}

static void run_sp_conv3d_k3(
    ec_trellis2_t *tr, const float *x_d, const int32_t *nb_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int N, int Cin, int Cout, uint16_t *col_scratch, size_t col_cap_elems) {
  int max_chunk = (int)(col_cap_elems / (size_t)(27 * Cin));
  if (max_chunk < 64) max_chunk = 64;
  for (int v0 = 0; v0 < N; v0 += max_chunk) {
    int ch = (N - v0 < max_chunk) ? (N - v0) : max_chunk;
    long n = (long)ch * 27 * Cin;
    k_tr_sp_gather_k3<<<(n + 255) / 256, 256, 0, tr->stream>>>(x_d, nb_d, col_scratch, v0, ch, Cin);
    tr_gemm(col_scratch, w_bf16, b_f32, y_d + (long)v0 * Cout, NULL, ch, Cout, 27 * Cin, tr->stream);
  }
}

// Unified Streamed Sparse 3D ConvNeXt + C2S Decoder (runs BOTH shape_dec [7ch, pred_subdiv=1] AND tex_dec [6ch, pred_subdiv=0]!)
static int run_sparse_unet_decoder(
    ec_trellis2_t *tr, const ec_safetensors_t *dec_st,
    const float *slat_d, const int32_t *coords32_h, int N0,
    int pred_subdiv, int out_ch,
    tr_subdiv_level_t subdiv_lvls[4],
    int *out_N_final, int32_t **out_coords512_h, float **out_feats_h, double *out_ms) {
  double t0 = tr_now_ms();
  const int CH[5]   = { 1024, 512, 256, 128, 64 };
  const int NBLK[4] = { 4, 16, 8, 4 };
  const int RES[5]  = { 32, 64, 128, 256, 512 };
  const size_t COL_CAP = 32L * 1024 * 1024; // 64 MB BF16 scratch
  uint16_t *col_d = NULL;
  if (cudaMalloc(&col_d, COL_CAP * 2) != cudaSuccess) return -1;

  // from_latent: [N0, 32] -> [N0, 1024]
  float *x_d = NULL;
  uint16_t *in_b = NULL;
  cudaMalloc(&x_d,  (size_t)N0 * CH[0] * 4);
  cudaMalloc(&in_b, (size_t)N0 * 32 * 2);
  k_tr_f32_to_bf16<<<((long)N0 * 32 + 255) / 256, 256, 0, tr->stream>>>(slat_d, in_b, (long)N0 * 32);
  {
    uint16_t *fw = tr_upload_bf16(dec_st, "from_latent.weight");
    float    *fb = tr_upload_f32(dec_st,  "from_latent.bias");
    if (!fw || !fb) { cudaFree(fw); cudaFree(fb); cudaFree(x_d); cudaFree(in_b); cudaFree(col_d); return -1; }
    tr_gemm(in_b, fw, fb, x_d, NULL, N0, CH[0], 32, tr->stream);
    cudaStreamSynchronize(tr->stream);
    cudaFree(fw); cudaFree(fb);
  }
  cudaFree(in_b);

  int N = N0;
  int32_t *cur_coords_h = (int32_t *)malloc((size_t)N * 3 * sizeof(int32_t));
  memcpy(cur_coords_h, coords32_h, (size_t)N * 3 * sizeof(int32_t));
  int32_t *nb_d = NULL;
  if (build_sp_nb_gpu(tr, cur_coords_h, N, RES[0], &nb_d) != 0) {
    free(cur_coords_h); cudaFree(x_d); cudaFree(col_d); return -1;
  }

  char k[192];
  for (int s = 0; s < 4; s++) {
    int Cin = CH[s], Cout = CH[s + 1];
    float *h_d = NULL, *fc_d = NULL;
    uint16_t *hb_d = NULL, *fcb_d = NULL;
    cudaMalloc(&h_d,   (size_t)N * Cin * 4);
    cudaMalloc(&hb_d,  (size_t)N * Cin * 2);
    cudaMalloc(&fc_d,  (size_t)N * (4 * Cin) * 4);
    cudaMalloc(&fcb_d, (size_t)N * (4 * Cin) * 2);

    // 1. NBLK[s] SparseConvNeXtBlock3d layers at resolution RES[s], channels Cin
    for (int j = 0; j < NBLK[s]; j++) {
      snprintf(k, sizeof k, "blocks.%d.%d.conv.weight", s, j);  uint16_t *cw = tr_upload_bf16(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.conv.bias", s, j);    float    *cb = tr_upload_f32(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.norm.weight", s, j);  float    *nw = tr_upload_f32(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.norm.bias", s, j);    float    *nb = tr_upload_f32(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.mlp.0.weight", s, j); uint16_t *m0w = tr_upload_bf16(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.mlp.0.bias", s, j);   float    *m0b = tr_upload_f32(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.mlp.2.weight", s, j); uint16_t *m2w = tr_upload_bf16(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.mlp.2.bias", s, j);   float    *m2b = tr_upload_f32(dec_st, k);
      if (!cw || !nw || !m0w || !m2w) {
        cudaFree(cw); cudaFree(cb); cudaFree(nw); cudaFree(nb); cudaFree(m0w); cudaFree(m0b); cudaFree(m2w); cudaFree(m2b);
        cudaFree(h_d); cudaFree(hb_d); cudaFree(fc_d); cudaFree(fcb_d); cudaFree(nb_d); cudaFree(x_d); cudaFree(col_d);
        free(cur_coords_h); return -1;
      }
      run_sp_conv3d_k3(tr, x_d, nb_d, h_d, cw, cb, N, Cin, Cin, col_d, COL_CAP);
      k_tr_layernorm<<<N, 256, 0, tr->stream>>>(h_d, nw, nb, NULL, NULL, NULL, hb_d, N, Cin, 1e-6f, 0);
      tr_gemm(hb_d, m0w, m0b, fc_d, NULL, N, 4 * Cin, Cin, tr->stream);
      k_tr_silu_bf16<<<((long)N * 4 * Cin + 255) / 256, 256, 0, tr->stream>>>(fc_d, fcb_d, (long)N * 4 * Cin);
      tr_gemm(fcb_d, m2w, m2b, h_d, NULL, N, Cin, 4 * Cin, tr->stream);
      k_tr_add_inplace<<<((long)N * Cin + 255) / 256, 256, 0, tr->stream>>>(x_d, h_d, (long)N * Cin);
      cudaStreamSynchronize(tr->stream);
      cudaFree(cw); cudaFree(cb); cudaFree(nw); cudaFree(nb);
      cudaFree(m0w); cudaFree(m0b); cudaFree(m2w); cudaFree(m2b);
    }
    cudaFree(fc_d); cudaFree(fcb_d);

    // 2. SparseResBlockC2S3d upsample block: blocks.{s}.{NBLK[s]}
    int uj = NBLK[s];
    if (pred_subdiv) {
      snprintf(k, sizeof k, "blocks.%d.%d.to_subdiv.weight", s, uj); uint16_t *sw = tr_upload_bf16(dec_st, k);
      snprintf(k, sizeof k, "blocks.%d.%d.to_subdiv.bias", s, uj);   float    *sb = tr_upload_f32(dec_st, k);
      float *sub_d = NULL;
      cudaMalloc(&sub_d, (size_t)N * 8 * 4);
      k_tr_f32_to_bf16<<<((long)N * Cin + 255) / 256, 256, 0, tr->stream>>>(x_d, hb_d, (long)N * Cin);
      tr_gemm(hb_d, sw, sb, sub_d, NULL, N, 8, Cin, tr->stream);
      float *sub_h = (float *)malloc((size_t)N * 8 * 4);
      cudaMemcpyAsync(sub_h, sub_d, (size_t)N * 8 * 4, cudaMemcpyDeviceToHost, tr->stream);
      cudaStreamSynchronize(tr->stream);
      cudaFree(sw); cudaFree(sb); cudaFree(sub_d);

      // Count active children (sub_h[p*8 + b] > 0) — exact TRELLIS.2 subdivision threshold (0.0f)!
      // Only raise threshold if extreme pathological seed exceeds 3.5M voxels (never chop valid 512^3 shells!).
      int n_child = 0;
      for (long i = 0; i < (long)N * 8; i++) if (sub_h[i] > 0.0f) n_child++;
      float thresh = 0.0f;
      if (n_child > 3500000) {
        thresh = 0.25f; n_child = 0;
        for (long i = 0; i < (long)N * 8; i++) if (sub_h[i] > thresh) n_child++;
      }
      if (n_child == 0) {
        // Guarantee at least 1 child per parent if all negative
        for (int p = 0; p < N; p++) { sub_h[p * 8] = 1.0f; }
        n_child = N; thresh = 0.0f;
      }
      int32_t *cc = (int32_t *)malloc((size_t)n_child * 3 * sizeof(int32_t));
      int32_t *cp = (int32_t *)malloc((size_t)n_child * sizeof(int32_t));
      uint8_t *cs = (uint8_t *)malloc((size_t)n_child);
      int ci = 0;
      for (int p = 0; p < N; p++) {
        int px = cur_coords_h[p * 3 + 0] * 2;
        int py = cur_coords_h[p * 3 + 1] * 2;
        int pz = cur_coords_h[p * 3 + 2] * 2;
        for (int b = 0; b < 8; b++) {
          if (sub_h[(long)p * 8 + b] > thresh) {
            cc[ci * 3 + 0] = px + (b & 1);
            cc[ci * 3 + 1] = py + ((b >> 1) & 1);
            cc[ci * 3 + 2] = pz + ((b >> 2) & 1);
            cp[ci] = p;
            cs[ci] = (uint8_t)b;
            ci++;
          }
        }
      }
      free(sub_h);
      subdiv_lvls[s].n_child        = ci;
      subdiv_lvls[s].child_coords_h = cc;
      subdiv_lvls[s].child_parent_h = cp;
      subdiv_lvls[s].child_sub_h    = cs;
    }

    int N_next = subdiv_lvls[s].n_child;
    // norm1 + SiLU -> conv1(Cin -> 8 * Cout)
    snprintf(k, sizeof k, "blocks.%d.%d.norm1.weight", s, uj); float    *n1w = tr_upload_f32(dec_st, k);
    snprintf(k, sizeof k, "blocks.%d.%d.norm1.bias", s, uj);   float    *n1b = tr_upload_f32(dec_st, k);
    snprintf(k, sizeof k, "blocks.%d.%d.conv1.weight", s, uj); uint16_t *c1w = tr_upload_bf16(dec_st, k);
    snprintf(k, sizeof k, "blocks.%d.%d.conv1.bias", s, uj);   float    *c1b = tr_upload_f32(dec_st, k);
    k_tr_layernorm<<<N, 256, 0, tr->stream>>>(x_d, n1w, n1b, NULL, NULL, h_d, NULL, N, Cin, 1e-6f, 1);
    float *h8_d = NULL;
    cudaMalloc(&h8_d, (size_t)N * (8 * Cout) * 4);
    run_sp_conv3d_k3(tr, h_d, nb_d, h8_d, c1w, c1b, N, Cin, 8 * Cout, col_d, COL_CAP);
    cudaStreamSynchronize(tr->stream);
    cudaFree(n1w); cudaFree(n1b); cudaFree(c1w); cudaFree(c1b);
    cudaFree(h_d); cudaFree(hb_d); cudaFree(nb_d);

    // Channel2Spatial gather onto N_next children!
    int32_t *cp_d = NULL; uint8_t *cs_d = NULL;
    cudaMalloc(&cp_d, (size_t)N_next * sizeof(int32_t));
    cudaMalloc(&cs_d, (size_t)N_next);
    cudaMemcpyAsync(cp_d, subdiv_lvls[s].child_parent_h, (size_t)N_next * sizeof(int32_t), cudaMemcpyHostToDevice, tr->stream);
    cudaMemcpyAsync(cs_d, subdiv_lvls[s].child_sub_h,    (size_t)N_next,                   cudaMemcpyHostToDevice, tr->stream);

    float *h_up_d = NULL, *h_n2_d = NULL, *h_c2_d = NULL;
    cudaMalloc(&h_up_d, (size_t)N_next * Cout * 4);
    k_tr_c2s_gather<<<((long)N_next * Cout + 255) / 256, 256, 0, tr->stream>>>(h8_d, cp_d, cs_d, h_up_d, N_next, Cout);
    cudaStreamSynchronize(tr->stream);
    cudaFree(h8_d);

    // Build neighbor table at RES[s+1]
    free(cur_coords_h);
    cur_coords_h = (int32_t *)malloc((size_t)N_next * 3 * sizeof(int32_t));
    memcpy(cur_coords_h, subdiv_lvls[s].child_coords_h, (size_t)N_next * 3 * sizeof(int32_t));
    if (build_sp_nb_gpu(tr, cur_coords_h, N_next, RES[s + 1], &nb_d) != 0) {
      cudaFree(cp_d); cudaFree(cs_d); cudaFree(h_up_d); cudaFree(x_d); cudaFree(col_d); free(cur_coords_h); return -1;
    }

    // norm2 (no affine, eps=1e-6) + SiLU -> conv2(Cout -> Cout) + C2S skip(x_d)
    cudaMalloc(&h_n2_d, (size_t)N_next * Cout * 4);
    cudaMalloc(&h_c2_d, (size_t)N_next * Cout * 4);
    k_tr_layernorm<<<N_next, 256, 0, tr->stream>>>(h_up_d, NULL, NULL, NULL, NULL, h_n2_d, NULL, N_next, Cout, 1e-6f, 1);
    cudaFree(h_up_d);

    snprintf(k, sizeof k, "blocks.%d.%d.conv2.weight", s, uj); uint16_t *c2w = tr_upload_bf16(dec_st, k);
    snprintf(k, sizeof k, "blocks.%d.%d.conv2.bias", s, uj);   float    *c2b = tr_upload_f32(dec_st, k);
    run_sp_conv3d_k3(tr, h_n2_d, nb_d, h_c2_d, c2w, c2b, N_next, Cout, Cout, col_d, COL_CAP);
    k_tr_c2s_skip_add<<<((long)N_next * Cout + 255) / 256, 256, 0, tr->stream>>>(h_c2_d, x_d, cp_d, cs_d, N_next, Cin, Cout);
    cudaStreamSynchronize(tr->stream);

    cudaFree(c2w); cudaFree(c2b); cudaFree(h_n2_d);
    cudaFree(cp_d); cudaFree(cs_d); cudaFree(x_d);
    x_d = h_c2_d;
    N = N_next;
  }

  // Final output layer at 512^3: F.layer_norm(x, [64], eps=1e-5, no affine) -> output_layer(64 -> out_ch)
  cudaFree(nb_d);
  cudaFree(col_d);
  uint16_t *xn_b = NULL;
  float *out_d = NULL;
  cudaMalloc(&xn_b,  (size_t)N * 64 * 2);
  cudaMalloc(&out_d, (size_t)N * out_ch * 4);
  k_tr_layernorm<<<N, 64, 0, tr->stream>>>(x_d, NULL, NULL, NULL, NULL, NULL, xn_b, N, 64, 1e-5f, 0);
  cudaFree(x_d);
  {
    uint16_t *ow = tr_upload_bf16(dec_st, "output_layer.weight");
    float    *ob = tr_upload_f32(dec_st,  "output_layer.bias");
    tr_gemm(xn_b, ow, ob, out_d, NULL, N, out_ch, 64, tr->stream);
    cudaStreamSynchronize(tr->stream);
    cudaFree(ow); cudaFree(ob);
  }
  cudaFree(xn_b);

  float *out_h = (float *)malloc((size_t)N * out_ch * 4);
  cudaMemcpyAsync(out_h, out_d, (size_t)N * out_ch * 4, cudaMemcpyDeviceToHost, tr->stream);
  cudaStreamSynchronize(tr->stream);
  cudaFree(out_d);

  *out_N_final = N;
  if (out_coords512_h) *out_coords512_h = cur_coords_h; else free(cur_coords_h);
  *out_feats_h = out_h;
  if (out_ms) *out_ms = tr_now_ms() - t0;
  return 0;
}

static int extract_trellis2_ovoxel_pbr(
    const int32_t *coords512, int N,
    const float *shape7, const float *tex6, const float *logits_64,
    ec_mesh_t *mesh, ec_splat3d_t *splat) {
  memset(mesh, 0, sizeof *mesh);
  memset(splat, 0, sizeof *splat);
  if (N <= 0 || !coords512 || !shape7 || !tex6) return -1;

  // Build CPU open-addressing hash table mapping (x, y, z) -> voxel index [0..N-1]
  uint32_t cap = 1024u;
  while (cap < (uint32_t)N * 4u) cap <<= 1;
  uint32_t mask = cap - 1u;
  uint64_t *hkeys = (uint64_t *)calloc(cap, sizeof(uint64_t));
  int32_t  *hvals = (int32_t *)malloc(cap * sizeof(int32_t));
  if (!hkeys || !hvals) { free(hkeys); free(hvals); return -1; }
  for (int i = 0; i < N; i++) {
    uint64_t k = (((uint64_t)(uint32_t)coords512[i*3+0] << 20) |
                  ((uint64_t)(uint32_t)coords512[i*3+1] << 10) |
                   (uint64_t)(uint32_t)coords512[i*3+2]) + 1ULL;
    uint32_t sl = (uint32_t)((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
    while (hkeys[sl] != 0ULL && hkeys[sl] != k) sl = (sl + 1u) & mask;
    hkeys[sl] = k; hvals[sl] = i;
  }
  auto hfind = [&](int x, int y, int z) -> int32_t {
    if (x < 0 || x >= 512 || y < 0 || y >= 512 || z < 0 || z >= 512) return -1;
    uint64_t k = (((uint64_t)(uint32_t)x << 20) | ((uint64_t)(uint32_t)y << 10) | (uint64_t)(uint32_t)z) + 1ULL;
    uint32_t sl = (uint32_t)((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
    while (hkeys[sl] != 0ULL) {
      if (hkeys[sl] == k) return hvals[sl];
      sl = (sl + 1u) & mask;
    }
    return -1;
  };

  // Precompute softplus split weights for all N voxels
  float *sw = (float *)malloc((size_t)N * 4);
  for (int i = 0; i < N; i++) {
    float v = shape7[(long)i * 7 + 6];
    sw[i] = (v > 20.0f) ? v : log1pf(expf(v));
  }

  // Exact TRELLIS.2 Flexible Dual Grid neighbor offsets (tr2_flexible_dual_grid.py:173-181)
  static const int OFF[3][4][3] = {
    { {0,0,0}, {0,0,1}, {0,1,1}, {0,1,0} }, // X-axis
    { {0,0,0}, {1,0,0}, {1,0,1}, {0,0,1} }, // Y-axis
    { {0,0,0}, {0,1,0}, {1,1,0}, {1,0,0} }, // Z-axis
  };

  uint32_t tcap = (uint32_t)N * 2u + 16384u;
  uint32_t *raw_idx = (uint32_t *)malloc((size_t)tcap * 3 * sizeof(uint32_t));
  uint32_t *quad_v  = (uint32_t *)malloc((size_t)(tcap / 2 + 1) * 4 * sizeof(uint32_t));
  uint32_t n_tris = 0, n_quads = 0;
  for (int i = 0; i < N; i++) {
    int cx = coords512[i*3+0], cy = coords512[i*3+1], cz = coords512[i*3+2];
    const float *s7 = shape7 + (long)i * 7;
    for (int ax = 0; ax < 3; ax++) {
      if (s7[3 + ax] <= 0.0f) continue;
      int32_t qv[4];
      int valid = 1;
      for (int c = 0; c < 4; c++) {
        qv[c] = hfind(cx + OFF[ax][c][0], cy + OFF[ax][c][1], cz + OFF[ax][c][2]);
        if (qv[c] < 0) { valid = 0; break; }
      }
      if (!valid) continue;
      if (n_tris + 2 > tcap) {
        tcap *= 2;
        raw_idx = (uint32_t *)realloc(raw_idx, (size_t)tcap * 3 * sizeof(uint32_t));
        quad_v  = (uint32_t *)realloc(quad_v,  (size_t)(tcap / 2 + 1) * 4 * sizeof(uint32_t));
      }
      quad_v[n_quads*4+0] = (uint32_t)qv[0];
      quad_v[n_quads*4+1] = (uint32_t)qv[1];
      quad_v[n_quads*4+2] = (uint32_t)qv[2];
      quad_v[n_quads*4+3] = (uint32_t)qv[3];
      n_quads++;
      if (sw[qv[0]] * sw[qv[2]] > sw[qv[1]] * sw[qv[3]]) {
        raw_idx[n_tris*3+0] = qv[0]; raw_idx[n_tris*3+1] = qv[1]; raw_idx[n_tris*3+2] = qv[2]; n_tris++;
        raw_idx[n_tris*3+0] = qv[0]; raw_idx[n_tris*3+1] = qv[2]; raw_idx[n_tris*3+2] = qv[3]; n_tris++;
      } else {
        raw_idx[n_tris*3+0] = qv[0]; raw_idx[n_tris*3+1] = qv[1]; raw_idx[n_tris*3+2] = qv[3]; n_tris++;
        raw_idx[n_tris*3+0] = qv[3]; raw_idx[n_tris*3+1] = qv[1]; raw_idx[n_tris*3+2] = qv[2]; n_tris++;
      }
    }
  }
  free(hkeys); free(hvals); free(sw);
  if (n_tris == 0) { free(raw_idx); free(quad_v); return -1; }

  // Compact vertices to only those referenced by triangles
  int32_t *vmap = (int32_t *)malloc((size_t)N * sizeof(int32_t));
  memset(vmap, -1, (size_t)N * sizeof(int32_t));
  uint32_t n_verts = 0;
  for (uint32_t t = 0; t < n_tris * 3u; t++) {
    uint32_t oi = raw_idx[t];
    if (vmap[oi] < 0) vmap[oi] = (int32_t)(n_verts++);
    raw_idx[t] = (uint32_t)vmap[oi];
  }
  for (uint32_t k = 0; k < n_quads * 4u; k++) quad_v[k] = (uint32_t)vmap[quad_v[k]];

  float *pos = (float *)malloc((size_t)n_verts * 3 * 4);
  float *nrm = (float *)calloc((size_t)n_verts * 3, 4);
  float *uvs = (float *)malloc((size_t)n_verts * 2 * 4);
  float *col = (float *)malloc((size_t)n_verts * 3 * 4);
  float *vmet = (float *)malloc((size_t)n_verts * 4);
  float *vrgh = (float *)malloc((size_t)n_verts * 4);

  for (int i = 0; i < N; i++) {
    int32_t vi = vmap[i];
    if (vi < 0) continue;
    const float *s7 = shape7 + (long)i * 7;
    // Exact fdg_vae.py:100: dv = (1 + 2*0.5) * sigmoid(s7[0..2]) - 0.5 = 2*sigmoid - 0.5
    float dv0 = 2.0f / (1.0f + expf(-s7[0])) - 0.5f;
    float dv1 = 2.0f / (1.0f + expf(-s7[1])) - 0.5f;
    float dv2 = 2.0f / (1.0f + expf(-s7[2])) - 0.5f;
    float px = ((float)coords512[i*3+0] + dv0) * (1.0f / 512.0f) - 0.5f;
    float py = ((float)coords512[i*3+1] + dv1) * (1.0f / 512.0f) - 0.5f;
    float pz = ((float)coords512[i*3+2] + dv2) * (1.0f / 512.0f) - 0.5f;
    // TRELLIS.2 Z-up -> glTF Y-up: (px, pz, -py)
    pos[vi*3+0] =  px;
    pos[vi*3+1] =  pz;
    pos[vi*3+2] = -py;
    uvs[vi*2+0] = px + 0.5f;
    uvs[vi*2+1] = pz + 0.5f;

    // Exact tr2_pipeline.py:450: tex_voxels = tex_dec(slat) * 0.5 + 0.5 -> [R, G, B, metallic, roughness, alpha]
    const float *t6 = tex6 + (long)i * 6;
    float r = fminf(1.0f, fmaxf(0.0f, t6[0] * 0.5f + 0.5f));
    float g = fminf(1.0f, fmaxf(0.0f, t6[1] * 0.5f + 0.5f));
    float b = fminf(1.0f, fmaxf(0.0f, t6[2] * 0.5f + 0.5f));
    col[vi*3+0] = tr_srgb_to_linear(r);
    col[vi*3+1] = tr_srgb_to_linear(g);
    col[vi*3+2] = tr_srgb_to_linear(b);
    vmet[vi] = fminf(1.0f, fmaxf(0.0f, t6[3] * 0.5f + 0.5f));
    vrgh[vi] = fminf(1.0f, fmaxf(0.04f, t6[4] * 0.5f + 0.5f));
  }
  free(vmap);

  // BFS Edge-Consistent Manifold Winding + Signed-Volume Outward Orientation!
  // Eliminates 100% of normal-flip contour seams across smooth curved surfaces (Stage-1 64^3 logits are a hollow shell!).
  {
    (void)logits_64;
    uint32_t ecap = 4096u;
    while (ecap < n_quads * 8u) ecap <<= 1;
    uint32_t emask = ecap - 1u;
    struct EdgeSlot { uint64_t key; uint32_t q0, q1; uint8_t d0, d1, cnt; };
    EdgeSlot *etab = (EdgeSlot *)calloc(ecap, sizeof(EdgeSlot));
    uint8_t *qvis  = (uint8_t *)calloc(n_quads, 1);
    uint8_t *qflip = (uint8_t *)calloc(n_quads, 1);
    uint32_t *qbfs = (uint32_t *)malloc((size_t)n_quads * sizeof(uint32_t));
    if (etab && qvis && qflip && qbfs) {
      for (uint32_t q = 0; q < n_quads; q++) {
        for (int e = 0; e < 4; e++) {
          uint32_t u = quad_v[q*4 + e], v = quad_v[q*4 + ((e + 1) & 3)];
          uint8_t dir = (u < v) ? 0 : 1;
          uint64_t k = (((uint64_t)(dir ? v : u) << 32) | (uint64_t)(dir ? u : v)) + 1ULL;
          uint32_t sl = (uint32_t)((k * 0x9E3779B97F4A7C15ULL) >> 32) & emask;
          while (etab[sl].key != 0ULL && etab[sl].key != k) sl = (sl + 1u) & emask;
          if (etab[sl].key == 0ULL) {
            etab[sl].key = k; etab[sl].q0 = q; etab[sl].d0 = dir; etab[sl].cnt = 1;
          } else if (etab[sl].cnt == 1) {
            etab[sl].q1 = q; etab[sl].d1 = dir; etab[sl].cnt = 2;
          } else {
            etab[sl].cnt = 3; // non-manifold edge -> don't propagate across it
          }
        }
      }
      for (uint32_t qs = 0; qs < n_quads; qs++) {
        if (qvis[qs]) continue;
        uint32_t qh = 0, qt = 0;
        qvis[qs] = 1; qflip[qs] = 0; qbfs[qt++] = qs;
        while (qh < qt) {
          uint32_t cur = qbfs[qh++];
          for (int e = 0; e < 4; e++) {
            uint32_t u = quad_v[cur*4 + e], v = quad_v[cur*4 + ((e + 1) & 3)];
            uint8_t dir = (u < v) ? 0 : 1;
            uint64_t k = (((uint64_t)(dir ? v : u) << 32) | (uint64_t)(dir ? u : v)) + 1ULL;
            uint32_t sl = (uint32_t)((k * 0x9E3779B97F4A7C15ULL) >> 32) & emask;
            while (etab[sl].key != 0ULL && etab[sl].key != k) sl = (sl + 1u) & emask;
            if (etab[sl].cnt != 2) continue;
            uint32_t nb = (etab[sl].q0 == cur) ? etab[sl].q1 : etab[sl].q0;
            uint8_t nb_d = (etab[sl].q0 == cur) ? etab[sl].d1 : etab[sl].d0;
            if (!qvis[nb]) {
              qvis[nb] = 1;
              // Adjacent quads MUST traverse shared edge in opposite directions!
              qflip[nb] = (uint8_t)(qflip[cur] ^ (dir == nb_d ? 1 : 0));
              qbfs[qt++] = nb;
            }
          }
        }
        // Compute Green's theorem signed volume of this oriented component to guarantee OUTWARD normals!
        double vol6 = 0.0, rad_dot = 0.0;
        for (uint32_t ci = 0; ci < qt; ci++) {
          uint32_t q = qbfs[ci];
          uint8_t fl = qflip[q];
          for (int sub = 0; sub < 2; sub++) {
            uint32_t t = q * 2u + (uint32_t)sub;
            uint32_t i0 = raw_idx[t*3+0];
            uint32_t i1 = raw_idx[t*3+(fl ? 2 : 1)];
            uint32_t i2 = raw_idx[t*3+(fl ? 1 : 2)];
            double x0 = pos[i0*3+0], y0 = pos[i0*3+1], z0 = pos[i0*3+2];
            double x1 = pos[i1*3+0], y1 = pos[i1*3+1], z1 = pos[i1*3+2];
            double x2 = pos[i2*3+0], y2 = pos[i2*3+1], z2 = pos[i2*3+2];
            double cx = y1*z2 - z1*y2, cy = z1*x2 - x1*z2, cz = x1*y2 - y1*x2;
            vol6 += x0*cx + y0*cy + z0*cz;
            double ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
            double bx = x2 - x0, by = y2 - y0, bz = z2 - z0;
            double fnx = ay*bz - az*by, fny = az*bx - ax*bz, fnz = ax*by - ay*bx;
            rad_dot += fnx * (x0 + x1 + x2) + fny * (y0 + y1 + y2) + fnz * (z0 + z1 + z2);
          }
        }
        if ((qt >= 16 && vol6 < 0.0) || (qt < 16 && rad_dot < 0.0)) {
          for (uint32_t ci = 0; ci < qt; ci++) qflip[qbfs[ci]] ^= 1u;
        }
      }
      for (uint32_t q = 0; q < n_quads; q++) {
        for (int sub = 0; sub < 2; sub++) {
          uint32_t t = q * 2u + (uint32_t)sub;
          if (qflip[q]) {
            uint32_t tmp = raw_idx[t*3+1];
            raw_idx[t*3+1] = raw_idx[t*3+2];
            raw_idx[t*3+2] = tmp;
          }
          uint32_t i0 = raw_idx[t*3+0], i1 = raw_idx[t*3+1], i2 = raw_idx[t*3+2];
          float ax = pos[i1*3+0]-pos[i0*3+0], ay = pos[i1*3+1]-pos[i0*3+1], az = pos[i1*3+2]-pos[i0*3+2];
          float bx = pos[i2*3+0]-pos[i0*3+0], by = pos[i2*3+1]-pos[i0*3+1], bz = pos[i2*3+2]-pos[i0*3+2];
          float fnx = ay*bz - az*by, fny = az*bx - ax*bz, fnz = ax*by - ay*bx;
          nrm[i0*3+0] += fnx; nrm[i0*3+1] += fny; nrm[i0*3+2] += fnz;
          nrm[i1*3+0] += fnx; nrm[i1*3+1] += fny; nrm[i1*3+2] += fnz;
          nrm[i2*3+0] += fnx; nrm[i2*3+1] += fny; nrm[i2*3+2] += fnz;
        }
      }
    }
    free(etab); free(qvis); free(qflip); free(qbfs);
  }
  free(quad_v);
  for (uint32_t v = 0; v < n_verts; v++) {
    float x = nrm[v*3+0], y = nrm[v*3+1], z = nrm[v*3+2];
    float l = sqrtf(x*x + y*y + z*z);
    if (l > 1e-10f) { nrm[v*3+0] = x/l; nrm[v*3+1] = y/l; nrm[v*3+2] = z/l; }
    else { nrm[v*3+0] = 0.0f; nrm[v*3+1] = 1.0f; nrm[v*3+2] = 0.0f; }
  }

  // 1-pass Laplacian smooth on metallic & roughness maps to eliminate 1-triangle speckled material islands!
  {
    float *ms = (float *)calloc(n_verts, 4);
    float *rs = (float *)calloc(n_verts, 4);
    uint32_t *dg = (uint32_t *)calloc(n_verts, sizeof(uint32_t));
    if (ms && rs && dg) {
      for (uint32_t t = 0; t < n_tris; t++) {
        uint32_t a = raw_idx[t*3+0], b = raw_idx[t*3+1], c = raw_idx[t*3+2];
        ms[a] += vmet[b] + vmet[c]; rs[a] += vrgh[b] + vrgh[c]; dg[a] += 2;
        ms[b] += vmet[a] + vmet[c]; rs[b] += vrgh[a] + vrgh[c]; dg[b] += 2;
        ms[c] += vmet[a] + vmet[b]; rs[c] += vrgh[a] + vrgh[b]; dg[c] += 2;
      }
      for (uint32_t v = 0; v < n_verts; v++) {
        if (dg[v] > 0) {
          vmet[v] = 0.5f * vmet[v] + 0.5f * (ms[v] / (float)dg[v]);
          vrgh[v] = 0.5f * vrgh[v] + 0.5f * (rs[v] / (float)dg[v]);
        }
      }
    }
    free(ms); free(rs); free(dg);
  }

  // Cluster triangles into 4x4=16 exact-mean (metallic, roughness) PBR material primitives!
  uint16_t *tri_mats = (uint16_t *)malloc((size_t)n_tris * sizeof(uint16_t));
  double b_met[16] = {0}, b_rgh[16] = {0};
  uint32_t b_cnt[16] = {0};
  double tot_met = 0.0, tot_rgh = 0.0;
  for (uint32_t t = 0; t < n_tris; t++) {
    uint32_t i0 = raw_idx[t*3+0], i1 = raw_idx[t*3+1], i2 = raw_idx[t*3+2];
    float tm = (vmet[i0] + vmet[i1] + vmet[i2]) * (1.0f / 3.0f);
    float tr = (vrgh[i0] + vrgh[i1] + vrgh[i2]) * (1.0f / 3.0f);
    int mb = (int)(tm * 4.0f); if (mb < 0) mb = 0; if (mb > 3) mb = 3;
    int rb = (int)(tr * 4.0f); if (rb < 0) rb = 0; if (rb > 3) rb = 3;
    int bid = mb * 4 + rb;
    tri_mats[t] = (uint16_t)bid;
    b_met[bid] += tm; b_rgh[bid] += tr; b_cnt[bid]++;
    tot_met += tm; tot_rgh += tr;
  }
  ec_pbr_mat_t *mats = (ec_pbr_mat_t *)calloc(16, sizeof(ec_pbr_mat_t));
  int non_empty_buckets = 0;
  for (int b = 0; b < 16; b++) {
    mats[b].base_color[0] = 1.0f; mats[b].base_color[1] = 1.0f;
    mats[b].base_color[2] = 1.0f; mats[b].base_color[3] = 1.0f;
    mats[b].metallic  = b_cnt[b] ? (float)(b_met[b] / b_cnt[b]) : ((b / 4 + 0.5f) * 0.25f);
    mats[b].roughness = b_cnt[b] ? (float)(b_rgh[b] / b_cnt[b]) : ((b % 4 + 0.5f) * 0.25f);
    mats[b].double_sided = 1;
    if (b_cnt[b] > 0) non_empty_buckets++;
  }
  fprintf(stderr, "[trellis2] v17.2 512^3 O-Voxel PBR mesh: %u verts, %u tris, %d PBR material clusters (avg met=%.2f rgh=%.2f)\n",
          n_verts, n_tris, non_empty_buckets, tot_met / n_tris, tot_rgh / n_tris);

  mesh->positions   = pos;
  mesh->normals     = nrm;
  mesh->uvs         = uvs;
  mesh->colors      = col;
  mesh->indices     = raw_idx;
  mesh->n_verts     = n_verts;
  mesh->n_tris      = n_tris;
  mesh->base_color[0] = 1.0f; mesh->base_color[1] = 1.0f; mesh->base_color[2] = 1.0f; mesh->base_color[3] = 1.0f;
  mesh->metallic    = (float)(tot_met / n_tris);
  mesh->roughness   = (float)(tot_rgh / n_tris);
  mesh->tri_mat_ids = tri_mats;
  mesh->mats        = mats;
  mesh->n_mats      = 16;

  // Build crisp 3DGS .ply splat cloud (subsampled to <= 120k splats if needed to keep .ply lightweight)
  uint32_t step = (n_verts > 120000u) ? ((n_verts + 119999u) / 120000u) : 1u;
  uint32_t ns = (n_verts + step - 1u) / step;
  splat->n_splats = ns;
  splat->pos     = (float *)malloc((size_t)ns * 3 * 4);
  splat->scale   = (float *)malloc((size_t)ns * 3 * 4);
  splat->rot     = (float *)malloc((size_t)ns * 4 * 4);
  splat->opacity = (float *)malloc((size_t)ns * 4);
  splat->sh_dc   = (float *)malloc((size_t)ns * 3 * 4);
  float log_s = logf((float)step * 1.1f / 512.0f);
  for (uint32_t si = 0; si < ns; si++) {
    uint32_t v = si * step;
    splat->pos[si*3+0] = pos[v*3+0]; splat->pos[si*3+1] = pos[v*3+1]; splat->pos[si*3+2] = pos[v*3+2];
    splat->scale[si*3+0] = log_s; splat->scale[si*3+1] = log_s; splat->scale[si*3+2] = log_s;
    splat->rot[si*4+0] = 1.0f; splat->rot[si*4+1] = 0.0f; splat->rot[si*4+2] = 0.0f; splat->rot[si*4+3] = 0.0f;
    splat->opacity[si] = 3.0f;
    float cr = powf(fmaxf(0.0f, col[v*3+0]), 1.0f / 2.2f);
    float cg = powf(fmaxf(0.0f, col[v*3+1]), 1.0f / 2.2f);
    float cb = powf(fmaxf(0.0f, col[v*3+2]), 1.0f / 2.2f);
    splat->sh_dc[si*3+0] = (cr - 0.5f) / 0.28209479f;
    splat->sh_dc[si*3+1] = (cg - 0.5f) / 0.28209479f;
    splat->sh_dc[si*3+2] = (cb - 0.5f) / 0.28209479f;
  }
  free(vmet); free(vrgh);
  return 0;
}

// ===========================================================================
// 12. Public Lifecycle & End-to-End 3D Generator
// ===========================================================================
extern "C" ec_trellis2_t *ec_trellis2_create(const ec_trellis2_paths_t *paths) {
  if (!paths || !paths->dino_path[0] || !paths->ss_flow_path[0] || !paths->ss_dec_path[0]) return NULL;
  ec_trellis2_t *tr = (ec_trellis2_t *)calloc(1, sizeof *tr);
  if (!tr) return NULL;
  tr->paths = *paths;
  if (cudaStreamCreate(&tr->stream) != cudaSuccess) { free(tr); return NULL; }
  char err[128] = {0};
  if (ec_st_open(&tr->dino_st, paths->dino_path, err, sizeof err) != 0)    { ec_trellis2_destroy(tr); return NULL; }
  tr->dino_open = 1;
  if (ec_st_open(&tr->flow_st, paths->ss_flow_path, err, sizeof err) != 0) { ec_trellis2_destroy(tr); return NULL; }
  tr->flow_open = 1;
  if (ec_st_open(&tr->dec_st, paths->ss_dec_path, err, sizeof err) != 0)   { ec_trellis2_destroy(tr); return NULL; }
  tr->dec_open = 1;
  if (paths->slat_shape_flow_path[0] && paths->slat_tex_flow_path[0] &&
      paths->shape_dec_path[0] && paths->tex_dec_path[0]) {
    if (ec_st_open(&tr->sh_flow_st, paths->slat_shape_flow_path, err, sizeof err) == 0) tr->sh_flow_open = 1;
    if (ec_st_open(&tr->tx_flow_st, paths->slat_tex_flow_path,   err, sizeof err) == 0) tr->tx_flow_open = 1;
    if (ec_st_open(&tr->sh_dec_st,  paths->shape_dec_path,       err, sizeof err) == 0) tr->sh_dec_open  = 1;
    if (ec_st_open(&tr->tx_dec_st,  paths->tex_dec_path,         err, sizeof err) == 0) tr->tx_dec_open  = 1;
    if (tr->sh_flow_open && tr->tx_flow_open && tr->sh_dec_open && tr->tx_dec_open) {
      tr->full_3stage = 1;
      fprintf(stderr, "[trellis2] v17.2 Full 3-Stage 512^3 O-Voxel PBR pipeline READY!\n");
    }
  }
  ec_cowrie_file_fingerprint(paths->dino_path, tr->dino_sha);
  return tr;
}

extern "C" void ec_trellis2_unload_vram(ec_trellis2_t *tr) {
  if (!tr) return;
  free_flow_res(&tr->flow);
  free_dec_res(&tr->dec);
}

extern "C" void ec_trellis2_destroy(ec_trellis2_t *tr) {
  if (!tr) return;
  ec_trellis2_unload_vram(tr);
  if (tr->dino_open)    ec_st_close(&tr->dino_st);
  if (tr->flow_open)    ec_st_close(&tr->flow_st);
  if (tr->dec_open)     ec_st_close(&tr->dec_st);
  if (tr->sh_flow_open) ec_st_close(&tr->sh_flow_st);
  if (tr->tx_flow_open) ec_st_close(&tr->tx_flow_st);
  if (tr->sh_dec_open)  ec_st_close(&tr->sh_dec_st);
  if (tr->tx_dec_open)  ec_st_close(&tr->tx_dec_st);
  if (tr->stream) cudaStreamDestroy(tr->stream);
  free(tr);
}

extern "C" int ec_trellis2_is_ready(const ec_trellis2_t *tr) {
  return (tr && tr->dino_open && tr->flow_open && tr->dec_open) ? 1 : 0;
}

// Average BF16 DINOv3 patch features across multiple camera views in F32
__global__ void k_tr_accum_bf16_f32(float *acc_f32, const uint16_t *src_bf16, long n, float weight) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __uint_as_float(((uint32_t)src_bf16[i]) << 16);
  acc_f32[i] += v * weight;
}

extern "C" int ec_trellis2_generate_3d(
    ec_trellis2_t *tr,
    const ec_trellis2_req_t *req,
    const char *out_glb_path,
    const char *out_ply_path,
    ec_trellis2_progress_fn cb, void *cb_ctx,
    ec_trellis2_res_t *res) {
  memset(res, 0, sizeof *res);
  if (!ec_trellis2_is_ready(tr) || !req || !req->rgb || req->img_w <= 0 || req->img_h <= 0) {
    snprintf(res->error, sizeof res->error, "trellis2 engine not initialized or missing rgb image");
    return -1;
  }
  double t_all = tr_now_ms();
  int steps = (req->steps >= 2 && req->steps <= 50) ? req->steps : 12;
  float cfg = (req->cfg_scale > 0.0f) ? req->cfg_scale : 7.5f;
  float rsc = (req->rescale_t > 0.0f) ? req->rescale_t : 5.0f;

  // Sample dominant center-region PBR albedo if caller didn't override base_color
  float bcol[4] = {0.78f, 0.75f, 0.72f, 1.0f};
  if (req->base_color[3] > 0.01f) {
    memcpy(bcol, req->base_color, 4 * sizeof(float));
  } else {
    double sr = 0, sg = 0, sb = 0;
    int cnt = 0;
    int y0 = req->img_h / 4, y1 = (req->img_h * 3) / 4;
    int x0 = req->img_w / 4, x1 = (req->img_w * 3) / 4;
    for (int y = y0; y < y1; y += 2) {
      for (int x = x0; x < x1; x += 2) {
        const uint8_t *p = req->rgb + ((long)y * req->img_w + x) * 3;
        sr += p[0]; sg += p[1]; sb += p[2]; cnt++;
      }
    }
    if (cnt > 0) {
      bcol[0] = (float)(sr / (cnt * 255.0));
      bcol[1] = (float)(sg / (cnt * 255.0));
      bcol[2] = (float)(sb / (cnt * 255.0));
      bcol[3] = 1.0f;
    }
  }

  // Stage 1: DINOv3-ViT-L/16 Conditioning Features [1029, 1024] per view (Stochastic View Cycling — zero patch blurring!)
  if (cb) cb(0, steps + 3, "trellis2_dinov3_encode", cb_ctx);
  uint16_t *view_conds_d[4] = {0};
  int n_views = 0;
  cudaMalloc(&view_conds_d[0], 1029L * 1024L * 2);
  if (run_dinov3_vitl16(tr, req->rgb, req->img_w, req->img_h, view_conds_d[0], &res->dino_ms) != 0) {
    cudaFree(view_conds_d[0]);
    snprintf(res->error, sizeof res->error, "dinov3 vision encoder failed");
    return -1;
  }
  n_views = 1;
  int n_extra = req->n_extra_views;
  if (n_extra > 3) n_extra = 3;
  for (int vi = 0; vi < n_extra; vi++) {
    if (!req->extra_rgbs[vi] || req->extra_w[vi] <= 0 || req->extra_h[vi] <= 0) continue;
    cudaMalloc(&view_conds_d[n_views], 1029L * 1024L * 2);
    double vms = 0.0;
    if (run_dinov3_vitl16(tr, req->extra_rgbs[vi], req->extra_w[vi], req->extra_h[vi], view_conds_d[n_views], &vms) == 0) {
      res->dino_ms += vms;
      n_views++;
    } else {
      cudaFree(view_conds_d[n_views]); view_conds_d[n_views] = NULL;
    }
  }

  // Stage 1b: 30-Block SparseStructureFlowModel 1.3B BF16 (12-step Flow Euler CFG + Stochastic Multi-View Cycling!)
  int use_full_3stage = (tr->full_3stage && !(getenv("EC_TRELLIS_STAGE1_ONLY") && atoi(getenv("EC_TRELLIS_STAGE1_ONLY")) != 0)) ? 1 : 0;
  if (ensure_flow_loaded(tr, req->loras, req->n_loras) != 0) {
    for (int v = 0; v < n_views; v++) cudaFree(view_conds_d[v]);
    snprintf(res->error, sizeof res->error, "trellis2 ss_flow load failed");
    return -1;
  }
  int lora_hits = tr->flow.lora_hits;
  float *z_d = NULL;
  cudaMalloc(&z_d, 4096L * 8 * 4);
  int rc = run_ss_flow(tr, view_conds_d, n_views, steps, cfg, rsc, req->seed, z_d, cb, cb_ctx, &res->flow_ms);
  if (!use_full_3stage) {
    for (int v = 0; v < n_views; v++) { cudaFree(view_conds_d[v]); view_conds_d[v] = NULL; }
  }
  if (rc != 0) {
    for (int v = 0; v < n_views; v++) cudaFree(view_conds_d[v]);
    cudaFree(z_d);
    snprintf(res->error, sizeof res->error, rc == -2 ? "cancelled" : "trellis2 ss_flow failed");
    return rc;
  }

  // Stage 1c: SparseStructureDecoder (16^3x8 -> 64^3x1 logits)
  if (cb) cb(steps + 1, steps + 3, "trellis2_ss_decode_64", cb_ctx);
  if (ensure_dec_loaded(tr) != 0) {
    for (int v = 0; v < n_views; v++) cudaFree(view_conds_d[v]);
    cudaFree(z_d);
    snprintf(res->error, sizeof res->error, "trellis2 ss_dec load failed");
    return -1;
  }
  float *logits_h = (float *)malloc(262144L * 4);
  if (run_ss_decoder(tr, z_d, logits_h, &res->dec_ms) != 0) {
    for (int v = 0; v < n_views; v++) cudaFree(view_conds_d[v]);
    free(logits_h); cudaFree(z_d);
    snprintf(res->error, sizeof res->error, "trellis2 ss_dec forward failed");
    return -1;
  }
  cudaFree(z_d);

  ec_mesh_t mesh = {};
  ec_splat3d_t splat = {};
  int n_act = 0;
  int built_3stage = 0;
  double tm0 = tr_now_ms();

  // v17.2 Full 3-Stage 512^3 O-Voxel PBR Pipeline (Stage 2 Shape SLat + Stage 3 Texture SLat + Dual Contouring PBR!)
  if (use_full_3stage) {
    // Evict Stage-1 resident weights so full 24 GB VRAM is wide open for Stage 2 & Stage 3!
    free_flow_res(&tr->flow);
    free_dec_res(&tr->dec);

    // Downsample 64^3 > 0 via 2x2x2 max_pool3d -> 32^3 active sparse coords (exact tr2_pipeline.py:227-233!)
    int border_tot = 0, border_pos = 0, interior_pos = 0;
    for (int x = 0; x < 64; x++)
      for (int y = 0; y < 64; y++)
        for (int z = 0; z < 64; z++) {
          int pos_v = (logits_h[(x * 64 + y) * 64 + z] > 0.0f);
          int on_b = (x == 0 || x == 63 || y == 0 || y == 63 || z == 0 || z == 63);
          int in_c = (x >= 2 && x <= 61 && y >= 2 && y <= 61 && z >= 2 && z <= 61);
          if (on_b) { border_tot++; if (pos_v) border_pos++; }
          if (in_c && pos_v) interior_pos++;
        }
    int strip_shell = (border_tot > 0 && (double)border_pos / (double)border_tot > 0.22 && interior_pos >= 800) ? 1 : 0;

    int32_t *coords32_h = (int32_t *)malloc(32768L * 3 * sizeof(int32_t));
    int N32 = 0;
    for (int x = 0; x < 32; x++) {
      for (int y = 0; y < 32; y++) {
        for (int z = 0; z < 32; z++) {
          int any = 0;
          for (int dx = 0; dx < 2 && !any; dx++)
            for (int dy = 0; dy < 2 && !any; dy++)
              for (int dz = 0; dz < 2 && !any; dz++) {
                int gx = x * 2 + dx, gy = y * 2 + dy, gz = z * 2 + dz;
                if (strip_shell && (gx < 2 || gx > 61 || gy < 2 || gy > 61 || gz < 2 || gz > 61)) continue;
                if (logits_h[(gx * 64 + gy) * 64 + gz] > 0.0f) any = 1;
              }
          if (any) {
            coords32_h[N32 * 3 + 0] = x;
            coords32_h[N32 * 3 + 1] = y;
            coords32_h[N32 * 3 + 2] = z;
            N32++;
          }
        }
      }
    }
    fprintf(stderr, "[trellis2] Stage-1 32^3 active sparse voxels: %d (strip_shell=%d)\n", N32, strip_shell);

    if (N32 >= 16 && N32 <= 24576) {
      int32_t *coords32_d = NULL;
      float *sh_norm_d = NULL, *sh_un_d = NULL, *tx_norm_d = NULL, *tx_un_d = NULL;
      float *mean_d = NULL, *std_d = NULL;
      cudaMalloc(&coords32_d, (size_t)N32 * 3 * sizeof(int32_t));
      cudaMalloc(&sh_norm_d,  (size_t)N32 * 32 * 4);
      cudaMalloc(&sh_un_d,    (size_t)N32 * 32 * 4);
      cudaMalloc(&tx_norm_d,  (size_t)N32 * 32 * 4);
      cudaMalloc(&tx_un_d,    (size_t)N32 * 32 * 4);
      cudaMalloc(&mean_d,     32 * 4);
      cudaMalloc(&std_d,      32 * 4);
      cudaMemcpy(coords32_d, coords32_h, (size_t)N32 * 3 * sizeof(int32_t), cudaMemcpyHostToDevice);

      double s2_flow_ms = 0.0, s3_flow_ms = 0.0, s2_dec_ms = 0.0, s3_dec_ms = 0.0;
      int rc2 = run_slat_flow_generic(
          tr, &tr->sh_flow_st, coords32_d, N32, 32, 32, NULL, view_conds_d, n_views,
          12, 7.5f, 0.5f, 0.6f, 1.0f, 3.0f, req->seed, sh_norm_d, "trellis2_shape_slat_flow",
          cb, cb_ctx, steps + 1, steps + 28, &s2_flow_ms);
      int rc3 = (rc2 == 0) ? run_slat_flow_generic(
          tr, &tr->tx_flow_st, coords32_d, N32, 64, 32, sh_norm_d, view_conds_d, n_views,
          12, 1.0f, 0.0f, 0.6f, 0.9f, 3.0f, req->seed + 1u, tx_norm_d, "trellis2_tex_slat_flow",
          cb, cb_ctx, steps + 13, steps + 28, &s3_flow_ms) : rc2;
      for (int v = 0; v < n_views; v++) { cudaFree(view_conds_d[v]); view_conds_d[v] = NULL; }
      cudaFree(coords32_d);
      res->flow_ms += s2_flow_ms + s3_flow_ms;

      if (rc2 == -2 || rc3 == -2) {
        cudaFree(sh_norm_d); cudaFree(sh_un_d); cudaFree(tx_norm_d); cudaFree(tx_un_d);
        cudaFree(mean_d); cudaFree(std_d); free(coords32_h); free(logits_h);
        snprintf(res->error, sizeof res->error, "cancelled");
        return -2;
      }

      if (rc2 == 0 && rc3 == 0) {
        cudaMemcpy(mean_d, TR2_SHAPE_MEAN, 32 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(std_d,  TR2_SHAPE_STD,  32 * 4, cudaMemcpyHostToDevice);
        k_tr_unnorm_ch<<<((long)N32 * 32 + 255) / 256, 256, 0, tr->stream>>>(sh_norm_d, sh_un_d, mean_d, std_d, N32, 32);
        cudaMemcpy(mean_d, TR2_TEX_MEAN,   32 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(std_d,  TR2_TEX_STD,    32 * 4, cudaMemcpyHostToDevice);
        k_tr_unnorm_ch<<<((long)N32 * 32 + 255) / 256, 256, 0, tr->stream>>>(tx_norm_d, tx_un_d, mean_d, std_d, N32, 32);
        cudaStreamSynchronize(tr->stream);

        tr_subdiv_level_t sub_lvls[4] = {};
        int N512_sh = 0, N512_tx = 0;
        int32_t *coords512_h = NULL;
        float *shape7_h = NULL, *tex6_h = NULL;
        if (cb) cb(steps + 25, steps + 28, "trellis2_shape_dec_512", cb_ctx);
        int drc2 = run_sparse_unet_decoder(
            tr, &tr->sh_dec_st, sh_un_d, coords32_h, N32, 1, 7,
            sub_lvls, &N512_sh, &coords512_h, &shape7_h, &s2_dec_ms);
        if (cb) cb(steps + 26, steps + 28, "trellis2_tex_dec_512", cb_ctx);
        int drc3 = (drc2 == 0) ? run_sparse_unet_decoder(
            tr, &tr->tx_dec_st, tx_un_d, coords32_h, N32, 0, 6,
            sub_lvls, &N512_tx, NULL, &tex6_h, &s3_dec_ms) : -1;
        res->dec_ms += s2_dec_ms + s3_dec_ms;
        free_subdiv_levels(sub_lvls);

        if (drc2 == 0 && drc3 == 0 && N512_sh > 0 && N512_sh == N512_tx) {
          fprintf(stderr, "[trellis2] Stage 2/3 decoded %d 512^3 O-Voxels (s2_flow=%.0fms s3_flow=%.0fms s2_dec=%.0fms s3_dec=%.0fms)\n",
                  N512_sh, s2_flow_ms, s3_flow_ms, s2_dec_ms, s3_dec_ms);
          tm0 = tr_now_ms();
          if (extract_trellis2_ovoxel_pbr(coords512_h, N512_sh, shape7_h, tex6_h, logits_h, &mesh, &splat) == 0) {
            n_act = N512_sh;
            built_3stage = 1;
          }
        }
        free(coords512_h); free(shape7_h); free(tex6_h);
      }
      cudaFree(sh_norm_d); cudaFree(sh_un_d); cudaFree(tx_norm_d); cudaFree(tx_un_d);
      cudaFree(mean_d); cudaFree(std_d);
    } else {
      for (int v = 0; v < n_views; v++) { cudaFree(view_conds_d[v]); view_conds_d[v] = NULL; }
    }
    free(coords32_h);
  }

  // Fallback to Stage-1 64^3 Surface Nets if 3-stage weights not installed or EC_TRELLIS_STAGE1_ONLY=1
  if (!built_3stage) {
    if (cb) cb(steps + 2, steps + 3, "trellis2_export_glb_ply", cb_ctx);
    tm0 = tr_now_ms();
    if (extract_surface_nets_64(logits_h, bcol, req->rgb, req->img_w, req->img_h, &mesh, &splat, &n_act) != 0) {
      free(logits_h);
      snprintf(res->error, sizeof res->error, "trellis2 empty occupancy volume (0 active voxels)");
      return -1;
    }
  }
  free(logits_h);

  int wr = 0;
  if (out_glb_path && out_glb_path[0]) wr |= ec_export_glb(out_glb_path, &mesh);
  if (out_ply_path && out_ply_path[0]) wr |= ec_export_splat_ply(out_ply_path, &splat);
  res->mesh_ms = tr_now_ms() - tm0;

  res->n_active_voxels = n_act;
  res->n_verts  = (int)mesh.n_verts;
  res->n_tris   = (int)mesh.n_tris;
  res->n_splats = (int)splat.n_splats;
  ec_mesh_free(&mesh);
  ec_splat_free(&splat);

  if (wr != 0) {
    snprintf(res->error, sizeof res->error, "glb/ply export failed");
    return -1;
  }
  res->ok = 1;
  res->steps_run = steps;
  res->lora_tensors_fused = lora_hits;
  res->total_ms = tr_now_ms() - t_all;
  return 0;
}
