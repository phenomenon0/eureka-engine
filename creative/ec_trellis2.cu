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
  tr_ss_flow_res_t    flow;
  tr_ss_dec_res_t     dec;
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

static int ensure_flow_loaded(ec_trellis2_t *tr) {
  if (tr->flow.loaded) return 0;
  const ec_safetensors_t *st = &tr->flow_st;
  tr_ss_flow_res_t *f = &tr->flow;
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

  // Optional TRELLIS.2 preprocess_image parity for dark-studio isolated hero renders:
  // If borders are near-black (avg < 22), clamp dark background noise to exact (0,0,0)
  // and tightly center-crop square around the foreground bounding box (with 5% margin).
  double border_sum = 0.0;
  long border_cnt = 0;
  for (int x = 0; x < img_w; x++) {
    for (int c = 0; c < 3; c++) {
      border_sum += rgb[(long)x * 3 + c];
      border_sum += rgb[((long)(img_h - 1) * img_w + x) * 3 + c];
    }
    border_cnt += 6;
  }
  for (int y = 0; y < img_h; y++) {
    for (int c = 0; c < 3; c++) {
      border_sum += rgb[((long)y * img_w) * 3 + c];
      border_sum += rgb[((long)y * img_w + (img_w - 1)) * 3 + c];
    }
    border_cnt += 6;
  }
  int dark_bg = (border_cnt > 0 && (border_sum / (double)border_cnt) < 22.0) ? 1 : 0;
  float crop_x0 = 0.0f, crop_y0 = 0.0f, crop_w = (float)img_w, crop_h = (float)img_h;
  if (dark_bg) {
    int bx0 = img_w, by0 = img_h, bx1 = -1, by1 = -1;
    for (int y = 0; y < img_h; y++) {
      for (int x = 0; x < img_w; x++) {
        const uint8_t *px = rgb + ((long)y * img_w + x) * 3;
        int m = px[0]; if (px[1] > m) m = px[1]; if (px[2] > m) m = px[2];
        if (m >= 26) {
          if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
          if (y < by0) by0 = y; if (y > by1) by1 = y;
        }
      }
    }
    if (bx1 > bx0 + 16 && by1 > by0 + 16) {
      float cx = 0.5f * (float)(bx0 + bx1);
      float cy = 0.5f * (float)(by0 + by1);
      float sz = (float)((bx1 - bx0 > by1 - by0) ? (bx1 - bx0) : (by1 - by0)) * 1.10f;
      if (sz > (float)(img_w < img_h ? img_w : img_h)) sz = (float)(img_w < img_h ? img_w : img_h);
      crop_x0 = cx - 0.5f * sz;
      crop_y0 = cy - 0.5f * sz;
      crop_w = sz;
      crop_h = sz;
    }
  }

  // Bilinear resize to 512x512 + ImageNet normalize -> pack 1024 patches of 3*16*16=768 BF16
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
        if (dark_bg) {
          float mx = fmaxf(rgb_f[0], fmaxf(rgb_f[1], rgb_f[2]));
          if (mx < 18.0f) {
            rgb_f[0] = rgb_f[1] = rgb_f[2] = 0.0f;
          } else if (mx < 28.0f) {
            float alpha = (mx - 18.0f) * 0.1f;
            rgb_f[0] *= alpha; rgb_f[1] *= alpha; rgb_f[2] *= alpha;
          }
        }
      }
      for (int c = 0; c < 3; c++) {
        img_norm[((long)c * S + y) * S + x] = (rgb_f[c] * (1.0f / 255.0f) - MEAN[c]) / STD[c];
      }
    }
  }

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
    ec_trellis2_t *tr, const uint16_t *cond_bf16_d,
    int steps, float cfg_scale, float rescale_t, uint32_t seed,
    float *out_z_d, // [4096, 8] F32 on GPU (matches [16, 16, 16, 8]!)
    ec_trellis2_progress_fn cb, void *cb_ctx, double *out_ms) {
  double t0 = tr_now_ms();
  const int N = 4096, DIM = 1536, H = 12, FFN = 8192, Lkv = 1029;
  const float sigma_min = 1e-5f;
  const float guidance_rescale = 0.7f;

  // Build neg_cond = zeros[1029, 1024] BF16
  uint16_t *neg_bf16_d = NULL;
  cudaMalloc(&neg_bf16_d, (size_t)Lkv * 1024 * 2);
  cudaMemsetAsync(neg_bf16_d, 0, (size_t)Lkv * 1024 * 2, tr->stream);

  // Precompute cross-attn K/V cache in BF16 for pos & neg cond across all 30 blocks (~360 MB total!)
  uint16_t *ck_pos[30], *cv_pos[30], *ck_neg[30], *cv_neg[30];
  float *kv_tmp = NULL;
  cudaMalloc(&kv_tmp, (size_t)Lkv * 2 * DIM * 4);
  int nw_kv = Lkv * H;
  for (int l = 0; l < 30; l++) {
    const tr_ss_block_t *B = &tr->flow.blocks[l];
    cudaMalloc(&ck_pos[l], (size_t)Lkv * DIM * 2);
    cudaMalloc(&cv_pos[l], (size_t)Lkv * DIM * 2);
    cudaMalloc(&ck_neg[l], (size_t)Lkv * DIM * 2);
    cudaMalloc(&cv_neg[l], (size_t)Lkv * DIM * 2);
    tr_gemm(cond_bf16_d, B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, Lkv, 2 * DIM, 1024, tr->stream);
    k_ss_cross_kv_norm<<<(nw_kv + 7) / 8, 256, 0, tr->stream>>>(kv_tmp, B->ca_kn, ck_pos[l], cv_pos[l], Lkv, H);
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
    ss_flow_forward(tr, xt_d, t_cur, ck_pos, cv_pos, &W, vp_d);

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
    cudaFree(ck_pos[l]); cudaFree(cv_pos[l]); cudaFree(ck_neg[l]); cudaFree(cv_neg[l]);
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

static int extract_surface_nets_64(
    const float *logits, const float base_color[4],
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

  // Laplacian Taubin-style mild smoothing pass (1 step lambda=0.35) for ultra-clean silhouette!
  float *pos_s = (float *)calloc((size_t)n_verts * 3, 4);
  uint32_t *deg = (uint32_t *)calloc(n_verts, sizeof(uint32_t));
  for (uint32_t t = 0; t < n_tris; t++) {
    uint32_t i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    uint32_t e[6] = {i0, i1, i1, i2, i2, i0};
    for (int k = 0; k < 6; k += 2) {
      uint32_t a = e[k], b = e[k + 1];
      pos_s[a * 3 + 0] += pos[b * 3 + 0]; pos_s[a * 3 + 1] += pos[b * 3 + 1]; pos_s[a * 3 + 2] += pos[b * 3 + 2]; deg[a]++;
      pos_s[b * 3 + 0] += pos[a * 3 + 0]; pos_s[b * 3 + 1] += pos[a * 3 + 1]; pos_s[b * 3 + 2] += pos[a * 3 + 2]; deg[b]++;
    }
  }
  for (uint32_t v = 0; v < n_verts; v++) {
    if (deg[v] > 0) {
      float inv = 1.0f / (float)deg[v];
      pos[v * 3 + 0] = 0.65f * pos[v * 3 + 0] + 0.35f * pos_s[v * 3 + 0] * inv;
      pos[v * 3 + 1] = 0.65f * pos[v * 3 + 1] + 0.35f * pos_s[v * 3 + 1] * inv;
      pos[v * 3 + 2] = 0.65f * pos[v * 3 + 2] + 0.35f * pos_s[v * 3 + 2] * inv;
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
    // Ensure triangle winding agrees with outward analytical gradient normal
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

  mesh->positions = pos;
  mesh->normals   = nrm;
  mesh->uvs       = uvs;
  mesh->indices   = idx;
  mesh->n_verts   = n_verts;
  mesh->n_tris    = n_tris;
  memcpy(mesh->base_color, base_color, 4 * sizeof(float));
  mesh->metallic  = 0.15f;
  mesh->roughness = 0.42f;

  // Build matching 3DGS Splat cloud from surface vertices
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
    float shade = 0.65f + 0.35f * fmaxf(0.0f, nrm[v*3+1] * 0.7f + nrm[v*3+0] * 0.3f);
    splat->sh_dc[v*3+0] = (base_color[0] * shade - 0.5f) / 0.28209479f;
    splat->sh_dc[v*3+1] = (base_color[1] * shade - 0.5f) / 0.28209479f;
    splat->sh_dc[v*3+2] = (base_color[2] * shade - 0.5f) / 0.28209479f;
  }
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
  if (tr->dino_open) ec_st_close(&tr->dino_st);
  if (tr->flow_open) ec_st_close(&tr->flow_st);
  if (tr->dec_open)  ec_st_close(&tr->dec_st);
  if (tr->stream) cudaStreamDestroy(tr->stream);
  free(tr);
}

extern "C" int ec_trellis2_is_ready(const ec_trellis2_t *tr) {
  return (tr && tr->dino_open && tr->flow_open && tr->dec_open) ? 1 : 0;
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

  // Stage 1: DINOv3-ViT-L/16 Conditioning Features [1029, 1024]
  if (cb) cb(0, steps + 3, "trellis2_dinov3_encode", cb_ctx);
  uint16_t *cond_bf16_d = NULL;
  cudaMalloc(&cond_bf16_d, 1029L * 1024L * 2);
  if (run_dinov3_vitl16(tr, req->rgb, req->img_w, req->img_h, cond_bf16_d, &res->dino_ms) != 0) {
    cudaFree(cond_bf16_d);
    snprintf(res->error, sizeof res->error, "dinov3 vision encoder failed");
    return -1;
  }

  // Stage 2: 30-Block SparseStructureFlowModel 1.3B BF16 (12-step Flow Euler CFG)
  if (ensure_flow_loaded(tr) != 0) {
    cudaFree(cond_bf16_d);
    snprintf(res->error, sizeof res->error, "trellis2 ss_flow load failed");
    return -1;
  }
  float *z_d = NULL;
  cudaMalloc(&z_d, 4096L * 8 * 4);
  int rc = run_ss_flow(tr, cond_bf16_d, steps, cfg, rsc, req->seed, z_d, cb, cb_ctx, &res->flow_ms);
  cudaFree(cond_bf16_d);
  if (rc != 0) {
    cudaFree(z_d);
    snprintf(res->error, sizeof res->error, rc == -2 ? "cancelled" : "trellis2 ss_flow failed");
    return rc;
  }

  // Stage 3: SparseStructureDecoder (16^3x8 -> 64^3x1 logits)
  if (cb) cb(steps + 1, steps + 3, "trellis2_ss_decode_64", cb_ctx);
  if (ensure_dec_loaded(tr) != 0) {
    cudaFree(z_d);
    snprintf(res->error, sizeof res->error, "trellis2 ss_dec load failed");
    return -1;
  }
  float *logits_h = (float *)malloc(262144L * 4);
  if (run_ss_decoder(tr, z_d, logits_h, &res->dec_ms) != 0) {
    free(logits_h); cudaFree(z_d);
    snprintf(res->error, sizeof res->error, "trellis2 ss_dec forward failed");
    return -1;
  }
  cudaFree(z_d);

  // Stage 4: Sub-Voxel Surface Nets Mesh + 3DGS Splat Extraction & Export
  if (cb) cb(steps + 2, steps + 3, "trellis2_export_glb_ply", cb_ctx);
  double tm0 = tr_now_ms();
  ec_mesh_t mesh;
  ec_splat3d_t splat;
  int n_act = 0;
  if (extract_surface_nets_64(logits_h, bcol, &mesh, &splat, &n_act) != 0) {
    free(logits_h);
    snprintf(res->error, sizeof res->error, "trellis2 empty occupancy volume (0 active voxels)");
    return -1;
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
  res->total_ms = tr_now_ms() - t_all;
  return 0;
}
