// ec_idol.cu — 100% Pure Native Eureka CUDA Engine for speridlabs/idol (Apache-2.0)
// Single Image -> 262,144 to 1,048,576 3D Gaussian Splats (.ply) with ZERO Python / ZERO cuBLAS / ZERO ggml!

#define _POSIX_C_SOURCE 200809L
#include "ec_idol.h"
#include "ec_media.h"
#include "ec_ops.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double id_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static __device__ __forceinline__ uint16_t id_f32_to_bf16_d(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return (uint16_t)(u >> 16);
}
static inline uint16_t id_f32_to_bf16_h(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return (uint16_t)(u >> 16);
}
static inline float id_bf16_to_f32_h(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f; memcpy(&f, &u, 4); return f;
}
static __device__ __forceinline__ float id_silu(float x) { return x / (1.0f + expf(-x)); }
static __device__ __forceinline__ float id_gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static __device__ __forceinline__ float id_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// ===========================================================================
// 1. Common GPU Utility & Norm Kernels
// ===========================================================================
__global__ void k_id_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = id_f32_to_bf16_d(src[i]);
}
__global__ void k_id_add_inplace(float *dst, const float *src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
__global__ void k_id_add_scaled_vec(float *dst, const float *src, const float *scale, int rows, int dim) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * dim;
  if (i < tot) dst[i] += src[i] * scale[i % dim];
}
__global__ void k_id_add_vec(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_id_silu_mul_bf16(const float *gate, const float *up, uint16_t *out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = id_f32_to_bf16_d(id_silu(gate[i]) * up[i]);
}
__global__ void k_id_gelu_tanh_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = id_f32_to_bf16_d(id_gelu_tanh(src[i]));
}
__global__ void k_id_silu_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = id_f32_to_bf16_d(id_silu(src[i]));
}
__global__ void k_id_fill_row_bias(float *dst, const float *b1, const float *b2, int rows, int dim) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * dim;
  if (i < tot) { int c = (int)(i % dim); dst[i] = b1[c] + b2[c]; }
}
__global__ void k_id_cfg_euler(float *x, const float *vp, const float *vn, long n, float cfg, float dt) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = (cfg > 0.0f && vn) ? ((1.0f + cfg) * vp[i] - cfg * vn[i]) : vp[i];
  x[i] -= v * dt;
}

__global__ void k_id_layernorm(
    const float *__restrict__ x,
    const float *__restrict__ w, const float *__restrict__ b,
    const float *__restrict__ adaln_scale, const float *__restrict__ adaln_shift,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int dim, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; s1 += v; s2 += v * v; }
  __shared__ float sm1[32], sm2[32];
  s1 = id_warp_sum(s1); s2 = id_warp_sum(s2);
  if ((threadIdx.x & 31) == 0) { sm1[threadIdx.x >> 5] = s1; sm2[threadIdx.x >> 5] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = blockDim.x >> 5;
    float a = (threadIdx.x < nw) ? sm1[threadIdx.x] : 0.0f;
    float bb = (threadIdx.x < nw) ? sm2[threadIdx.x] : 0.0f;
    a = id_warp_sum(a); bb = id_warp_sum(bb);
    if (threadIdx.x == 0) {
      float mean = a / (float)dim;
      float var = bb / (float)dim - mean * mean;
      sm1[0] = mean;
      sm2[0] = rsqrtf((var > 0.0f ? var : 0.0f) + eps);
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
    if (yf) yf[c] = v;
    if (yb) yb[c] = id_f32_to_bf16_d(v);
  }
}

// ===========================================================================
// 2. DINOv3-ViT-H+/16 RoPE Attention Unpack (H=20, D=64, grid_hw=64, n_prefix=5)
// ===========================================================================
__global__ void k_id_dino_qkv_rope(
    const float *__restrict__ q_in, const float *__restrict__ k_in, const float *__restrict__ v_in,
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V,
    int N, int N_pad, int H, int n_prefix, int grid_hw) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31; // D=64 -> 32 pairs
  if (idx >= N * H) return;
  int tok = idx / H;
  int h = idx % H;
  const float *qs = q_in + ((long)tok * H + h) * 64;
  const float *ks = k_in + ((long)tok * H + h) * 64;
  const float *vs = v_in + ((long)tok * H + h) * 64;
  uint16_t *qo = Q + ((long)h * N_pad + tok) * 64;
  uint16_t *ko = K + ((long)h * N_pad + tok) * 64;
  uint16_t *vo = V + ((long)h * N_pad + tok) * 64;

  float q0 = qs[lane], q1 = qs[lane + 32];
  float k0 = ks[lane], k1 = ks[lane + 32];
  if (tok >= n_prefix) {
    int p = tok - n_prefix;
    int py = p / grid_hw, px = p % grid_hw;
    float cy = 2.0f * (((float)py + 0.5f) / (float)grid_hw) - 1.0f;
    float cx = 2.0f * (((float)px + 0.5f) / (float)grid_hw) - 1.0f;
    int fi = lane & 15;
    float coord = (lane < 16) ? cy : cx;
    double inv_f = 1.0 / pow(100.0, (double)(4 * fi) / 64.0);
    double ang = 6.28318530717958647692 * (double)coord * inv_f;
    float cs = (float)cos(ang), sn = (float)sin(ang);
    qo[lane]      = id_f32_to_bf16_d(q0 * cs - q1 * sn);
    qo[lane + 32] = id_f32_to_bf16_d(q1 * cs + q0 * sn);
    ko[lane]      = id_f32_to_bf16_d(k0 * cs - k1 * sn);
    ko[lane + 32] = id_f32_to_bf16_d(k1 * cs + k0 * sn);
  } else {
    qo[lane] = id_f32_to_bf16_d(q0); qo[lane + 32] = id_f32_to_bf16_d(q1);
    ko[lane] = id_f32_to_bf16_d(k0); ko[lane + 32] = id_f32_to_bf16_d(k1);
  }
  vo[lane]      = id_f32_to_bf16_d(vs[lane]);
  vo[lane + 32] = id_f32_to_bf16_d(vs[lane + 32]);
}

// ===========================================================================
// 3. Idol RePo 3D RoPE + MultiHeadRMSNorm + Decoder Q/KV Kernels (H=16, D=64)
// ===========================================================================
// Exact layers.py:251-256: apply_rotary_emb FIRST (interleaved pairs 2p,2p+1), THEN MultiHeadRMSNorm!
__global__ void k_id_repo_qknorm_unpack(
    const float *__restrict__ qkv,         // [L, 3 * 1024]
    const float *__restrict__ delta_pos,   // [L, 16 * 3]
    const float *__restrict__ f0, const float *__restrict__ f1, const float *__restrict__ f2, // [10],[10],[12]
    const float *__restrict__ q_gamma, const float *__restrict__ k_gamma, // [16, 64]
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V, // [16, L_pad, 64]
    int L, int L_pad) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31; // 32 pairs -> 64 dims
  if (idx >= L * 16) return;
  int tok = idx >> 4;
  int h   = idx & 15;
  const float *base = qkv + (long)tok * 3072;
  const float *qs = base + h * 64;
  const float *ks = base + 1024 + h * 64;
  const float *vs = base + 2048 + h * 64;
  const float *dp = delta_pos + ((long)tok * 16 + h) * 3;
  const float *qg = q_gamma + h * 64;
  const float *kg = k_gamma + h * 64;

  // RePo 3D angle for pair `lane` (0..9: axis 0, 10..19: axis 1, 20..31: axis 2)
  float ang;
  if (lane < 10)      ang = dp[0] * f0[lane] * 3.14159265358979323846f;
  else if (lane < 20) ang = dp[1] * f1[lane - 10] * 3.14159265358979323846f;
  else                ang = dp[2] * f2[lane - 20] * 3.14159265358979323846f;
  float cs = cosf(ang), sn = sinf(ang);

  int d0 = lane * 2, d1 = d0 + 1;
  float rq0 = qs[d0] * cs - qs[d1] * sn;
  float rq1 = qs[d0] * sn + qs[d1] * cs;
  float rk0 = ks[d0] * cs - ks[d1] * sn;
  float rk1 = ks[d0] * sn + ks[d1] * cs;

  // MultiHeadRMSNorm: F.normalize(x, dim=-1) * gamma * sqrt(64)
  float sq = id_warp_sum(rq0 * rq0 + rq1 * rq1);
  float sk = id_warp_sum(rk0 * rk0 + rk1 * rk1);
  float inv_q = rsqrtf(sq + 1e-12f) * 8.0f;
  float inv_k = rsqrtf(sk + 1e-12f) * 8.0f;

  uint16_t *qo = Q + ((long)h * L_pad + tok) * 64;
  uint16_t *ko = K + ((long)h * L_pad + tok) * 64;
  uint16_t *vo = V + ((long)h * L_pad + tok) * 64;
  qo[d0] = id_f32_to_bf16_d(rq0 * inv_q * qg[d0]);
  qo[d1] = id_f32_to_bf16_d(rq1 * inv_q * qg[d1]);
  ko[d0] = id_f32_to_bf16_d(rk0 * inv_k * kg[d0]);
  ko[d1] = id_f32_to_bf16_d(rk1 * inv_k * kg[d1]);
  vo[d0] = id_f32_to_bf16_d(vs[d0]);
  vo[d1] = id_f32_to_bf16_d(vs[d1]);
}

// Decoder Self-Attn QKV unpack + MultiHeadRMSNorm (no RoPE, H=16, D=64)
__global__ void k_id_dec_self_qkv(
    const float *__restrict__ qkv,
    const float *__restrict__ q_gamma, const float *__restrict__ k_gamma,
    uint16_t *__restrict__ Q, uint16_t *__restrict__ K, uint16_t *__restrict__ V,
    int L, int L_pad) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= L * 16) return;
  int tok = idx >> 4, h = idx & 15;
  const float *base = qkv + (long)tok * 3072;
  const float *qs = base + h * 64, *ks = base + 1024 + h * 64, *vs = base + 2048 + h * 64;
  const float *qg = q_gamma + h * 64, *kg = k_gamma + h * 64;
  int d0 = lane * 2, d1 = d0 + 1;
  float q0 = qs[d0], q1 = qs[d1], k0 = ks[d0], k1 = ks[d1];
  float inv_q = rsqrtf(id_warp_sum(q0*q0 + q1*q1) + 1e-12f) * 8.0f;
  float inv_k = rsqrtf(id_warp_sum(k0*k0 + k1*k1) + 1e-12f) * 8.0f;
  uint16_t *qo = Q + ((long)h * L_pad + tok) * 64;
  uint16_t *ko = K + ((long)h * L_pad + tok) * 64;
  uint16_t *vo = V + ((long)h * L_pad + tok) * 64;
  qo[d0] = id_f32_to_bf16_d(q0 * inv_q * qg[d0]); qo[d1] = id_f32_to_bf16_d(q1 * inv_q * qg[d1]);
  ko[d0] = id_f32_to_bf16_d(k0 * inv_k * kg[d0]); ko[d1] = id_f32_to_bf16_d(k1 * inv_k * kg[d1]);
  vo[d0] = id_f32_to_bf16_d(vs[d0]);              vo[d1] = id_f32_to_bf16_d(vs[d1]);
}

// Decoder Cross-Attn Q unpack + MultiHeadRMSNorm (H=16, D=64)
__global__ void k_id_dec_cross_q(
    const float *__restrict__ q_in, const float *__restrict__ q_gamma,
    uint16_t *__restrict__ Q, int L, int L_pad) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= L * 16) return;
  int tok = idx >> 4, h = idx & 15;
  const float *qs = q_in + ((long)tok * 16 + h) * 64;
  const float *qg = q_gamma + h * 64;
  int d0 = lane * 2, d1 = d0 + 1;
  float q0 = qs[d0], q1 = qs[d1];
  float inv_q = rsqrtf(id_warp_sum(q0*q0 + q1*q1) + 1e-12f) * 8.0f;
  uint16_t *qo = Q + ((long)h * L_pad + tok) * 64;
  qo[d0] = id_f32_to_bf16_d(q0 * inv_q * qg[d0]);
  qo[d1] = id_f32_to_bf16_d(q1 * inv_q * qg[d1]);
}

// Decoder Cross-Attn KV unpack + MultiHeadRMSNorm (H=16, D=64)
__global__ void k_id_dec_cross_kv(
    const float *__restrict__ kv_in, const float *__restrict__ k_gamma,
    uint16_t *__restrict__ K, uint16_t *__restrict__ V, int S) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= S * 16) return;
  int tok = idx >> 4, h = idx & 15;
  const float *ks = kv_in + (long)tok * 2048 + h * 64;
  const float *vs = kv_in + (long)tok * 2048 + 1024 + h * 64;
  const float *kg = k_gamma + h * 64;
  int d0 = lane * 2, d1 = d0 + 1;
  float k0 = ks[d0], k1 = ks[d1];
  float inv_k = rsqrtf(id_warp_sum(k0*k0 + k1*k1) + 1e-12f) * 8.0f;
  uint16_t *ko = K + ((long)h * S + tok) * 64;
  uint16_t *vo = V + ((long)h * S + tok) * 64;
  ko[d0] = id_f32_to_bf16_d(k0 * inv_k * kg[d0]); ko[d1] = id_f32_to_bf16_d(k1 * inv_k * kg[d1]);
  vo[d0] = id_f32_to_bf16_d(vs[d0]);              vo[d1] = id_f32_to_bf16_d(vs[d1]);
}

// Exact PcdAbsolutePositionEmbedder (mode=0: denoiser 2^i / res * 2pi) & v2 (mode=1: decoder 2^linspace(0,10) * pi)
__global__ void k_id_pcd_pos_embed(
    const float *__restrict__ coords, // [N, 3]
    float *__restrict__ out,          // [N, 1024] (added into out if add_mode else written)
    int N, int v2_mode, int add_mode) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N * 1024;
  if (idx >= tot) return;
  int n = (int)(idx >> 10);
  int c = (int)(idx & 1023);
  float val = 0.0f;
  if (c < 1020) {
    int axis = c / 340;
    int rem  = c % 340;
    int is_cos = (rem >= 170);
    int fi = is_cos ? (rem - 170) : rem;
    float freq;
    if (!v2_mode) {
      freq = (fi < 16) ? exp2f((float)fi) : exp2f(((float)(fi - 16) / 154.0f) * 16.0f);
      float ang = coords[n * 3 + axis] * freq * 6.28318530717958647692f;
      val = is_cos ? cosf(ang) : sinf(ang);
    } else {
      freq = exp2f(((float)fi / 169.0f) * 10.0f);
      float ang = coords[n * 3 + axis] * freq * 3.14159265358979323846f;
      val = is_cos ? cosf(ang) : sinf(ang);
    }
  }
  if (add_mode) out[idx] += val;
  else out[idx] = val;
}

// ===========================================================================
// 4. Safetensors Upload Helpers (GPU F32->BF16 conversion for fast NVMe load!)
// ===========================================================================
static int id_load_f32_host(const ec_safetensors_t *st, const char *name, float **out, size_t *out_n) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[idol] missing tensor: %s\n", name); return -1; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  float *buf = (float *)malloc(n * 4);
  if (!buf) return -1;
  const void *p = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_F32) memcpy(buf, p, n * 4);
  else if (e->dtype == EC_DTYPE_BF16) {
    const uint16_t *s = (const uint16_t *)p;
    for (size_t i = 0; i < n; i++) buf[i] = id_bf16_to_f32_h(s[i]);
  } else { free(buf); return -1; }
  *out = buf;
  if (out_n) *out_n = n;
  return 0;
}

static float *id_upload_f32(const ec_safetensors_t *st, const char *name, cudaStream_t s) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[idol] missing f32 tensor: %s\n", name); return NULL; }
  if (e->dtype == EC_DTYPE_F32) {
    float *d = NULL;
    if (cudaMalloc(&d, e->nbytes) != cudaSuccess) return NULL;
    cudaMemcpyAsync(d, ec_st_ptr(st, e), e->nbytes, cudaMemcpyHostToDevice, s);
    return d;
  }
  float *h = NULL; size_t n = 0;
  if (id_load_f32_host(st, name, &h, &n) != 0) return NULL;
  float *d = NULL;
  if (cudaMalloc(&d, n * 4) != cudaSuccess) { free(h); return NULL; }
  cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
  free(h);
  return d;
}

static uint16_t *id_upload_bf16(const ec_safetensors_t *st, const char *name, float *stage_f32_d, size_t stage_cap, cudaStream_t s) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[idol] missing bf16 tensor: %s\n", name); return NULL; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  uint16_t *d = NULL;
  if (cudaMalloc(&d, n * 2) != cudaSuccess) return NULL;
  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(d, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, s);
  } else if (e->dtype == EC_DTYPE_F32 && stage_f32_d && n <= stage_cap) {
    cudaMemcpyAsync(stage_f32_d, ec_st_ptr(st, e), n * 4, cudaMemcpyHostToDevice, s);
    k_id_f32_to_bf16<<<(int)((n + 255) / 256), 256, 0, s>>>(stage_f32_d, d, (long)n);
  } else {
    float *h = NULL; size_t hn = 0;
    if (id_load_f32_host(st, name, &h, &hn) != 0) { cudaFree(d); return NULL; }
    uint16_t *b = (uint16_t *)malloc(hn * 2);
    for (size_t i = 0; i < hn; i++) b[i] = id_f32_to_bf16_h(h[i]);
    free(h);
    cudaMemcpy(d, b, hn * 2, cudaMemcpyHostToDevice);
    free(b);
  }
  return d;
}

// ===========================================================================
// 5. Resident Engine Structs
// ===========================================================================
typedef struct {
  float    *norm_w, *norm_b;     // [1024]
  uint16_t *gate_w, *content_w;  // [128, 1024]
  uint16_t *final_w;             // [48, 128]
  float    *f0, *f1, *f2;        // [10], [10], [12]
} id_repo_t;

typedef struct {
  int       modulated;
  float    *shift_table;         // [6144] if modulated
  float    *norm1_w, *norm1_b;   // [1024] if !modulated
  float    *norm2_w, *norm2_b;   // [1024] if !modulated
  uint16_t *qkv_w, *out_w;       // [3072,1024], [1024,1024]
  float    *qkv_b, *out_b;
  float    *q_norm, *k_norm;     // [16, 64]
  uint16_t *mlp0_w, *mlp2_w;     // [4096,1024], [1024,4096]
  float    *mlp0_b, *mlp2_b;
} id_ublk_t;

typedef struct {
  int       loaded;
  uint16_t *t0_w, *t2_w, *adaln_w;
  float    *t0_b, *t2_b, *adaln_b;
  uint16_t *in_w, *c1_w, *c2_w;
  float    *in_b, *c1_b, *c2_b;
  uint16_t *cam0_w, *cam2_w;
  float    *cam0_b, *cam2_b;
  float    *pos_pe;              // [8192, 3]
  float    *final_shift;         // [2, 1024]
  uint16_t *out_w, *cam_out_w;
  float    *out_b, *cam_out_b;
  id_repo_t noise_repo[2], ctx_repo[2], blk_repo[24];
  id_ublk_t noise_ref[2], ctx_ref[2], blocks[24];
} id_denoiser_t;

typedef struct {
  uint16_t *ca_q_w, *ca_kv_w, *ca_out_w;
  float    *ca_q_b, *ca_kv_b, *ca_out_b, *ca_qn, *ca_kn;
  uint16_t *mlp0_w, *mlp2_w;
  float    *mlp0_b, *mlp2_b;
} id_oct_blk_t;

typedef struct {
  int       loaded;
  uint16_t *in_w, *inp_w, *l0_w, *l2_w, *adaln_w, *out_w;
  float    *in_b, *inp_b, *l0_b, *l2_b, *adaln_b, *out_b;
  id_oct_blk_t blocks[4];
} id_oct_dec_t;

typedef struct {
  uint16_t *sa_qkv_w, *sa_out_w;
  float    *sa_qkv_b, *sa_out_b, *sa_qn, *sa_kn;
  float    *norm2_w, *norm2_b;
  uint16_t *ca_q_w, *ca_kv_w, *ca_out_w;
  float    *ca_q_b, *ca_kv_b, *ca_out_b, *ca_qn, *ca_kn;
  uint16_t *mlp0_w, *mlp2_w;
  float    *mlp0_b, *mlp2_b;
} id_gs_blk_t;

typedef struct {
  int       loaded;
  uint16_t *in_w, *inp_w, *out_w;
  float    *in_b, *inp_b, *out_b;
  float     offset_pert[32 * 3];
  float     base_offset_scale;
  id_gs_blk_t blocks[16];
} id_gs_dec_t;

struct ec_idol {
  ec_idol_paths_t  paths;
  cudaStream_t     stream;
  ec_safetensors_t dino_st, den_st, oct_st, gs_st;
  int              dino_open, den_open, oct_open, gs_open;
  id_denoiser_t    den;
  id_oct_dec_t     oct;
  id_gs_dec_t      gs;
};

static void free_repo(id_repo_t *r) {
  cudaFree(r->norm_w); cudaFree(r->norm_b);
  cudaFree(r->gate_w); cudaFree(r->content_w); cudaFree(r->final_w);
  cudaFree(r->f0); cudaFree(r->f1); cudaFree(r->f2);
}
static void free_ublk(id_ublk_t *b) {
  cudaFree(b->shift_table);
  cudaFree(b->norm1_w); cudaFree(b->norm1_b);
  cudaFree(b->norm2_w); cudaFree(b->norm2_b);
  cudaFree(b->qkv_w); cudaFree(b->qkv_b); cudaFree(b->out_w); cudaFree(b->out_b);
  cudaFree(b->q_norm); cudaFree(b->k_norm);
  cudaFree(b->mlp0_w); cudaFree(b->mlp0_b); cudaFree(b->mlp2_w); cudaFree(b->mlp2_b);
}
static void free_denoiser(id_denoiser_t *d) {
  if (!d->loaded) return;
  cudaFree(d->t0_w); cudaFree(d->t0_b); cudaFree(d->t2_w); cudaFree(d->t2_b);
  cudaFree(d->adaln_w); cudaFree(d->adaln_b);
  cudaFree(d->in_w); cudaFree(d->in_b); cudaFree(d->c1_w); cudaFree(d->c1_b); cudaFree(d->c2_w); cudaFree(d->c2_b);
  cudaFree(d->cam0_w); cudaFree(d->cam0_b); cudaFree(d->cam2_w); cudaFree(d->cam2_b);
  cudaFree(d->pos_pe); cudaFree(d->final_shift);
  cudaFree(d->out_w); cudaFree(d->out_b); cudaFree(d->cam_out_w); cudaFree(d->cam_out_b);
  for (int i = 0; i < 2; i++) {
    free_repo(&d->noise_repo[i]); free_repo(&d->ctx_repo[i]);
    free_ublk(&d->noise_ref[i]); free_ublk(&d->ctx_ref[i]);
  }
  for (int i = 0; i < 24; i++) { free_repo(&d->blk_repo[i]); free_ublk(&d->blocks[i]); }
  memset(d, 0, sizeof *d);
}
static void free_oct(id_oct_dec_t *o) {
  if (!o->loaded) return;
  cudaFree(o->in_w); cudaFree(o->in_b); cudaFree(o->inp_w); cudaFree(o->inp_b);
  cudaFree(o->l0_w); cudaFree(o->l0_b); cudaFree(o->l2_w); cudaFree(o->l2_b);
  cudaFree(o->adaln_w); cudaFree(o->adaln_b); cudaFree(o->out_w); cudaFree(o->out_b);
  for (int i = 0; i < 4; i++) {
    id_oct_blk_t *B = &o->blocks[i];
    cudaFree(B->ca_q_w); cudaFree(B->ca_q_b); cudaFree(B->ca_kv_w); cudaFree(B->ca_kv_b);
    cudaFree(B->ca_out_w); cudaFree(B->ca_out_b); cudaFree(B->ca_qn); cudaFree(B->ca_kn);
    cudaFree(B->mlp0_w); cudaFree(B->mlp0_b); cudaFree(B->mlp2_w); cudaFree(B->mlp2_b);
  }
  memset(o, 0, sizeof *o);
}
static void free_gs(id_gs_dec_t *g) {
  if (!g->loaded) return;
  cudaFree(g->in_w); cudaFree(g->in_b); cudaFree(g->inp_w); cudaFree(g->inp_b);
  cudaFree(g->out_w); cudaFree(g->out_b);
  for (int i = 0; i < 16; i++) {
    id_gs_blk_t *B = &g->blocks[i];
    cudaFree(B->sa_qkv_w); cudaFree(B->sa_qkv_b); cudaFree(B->sa_out_w); cudaFree(B->sa_out_b);
    cudaFree(B->sa_qn); cudaFree(B->sa_kn);
    cudaFree(B->norm2_w); cudaFree(B->norm2_b);
    cudaFree(B->ca_q_w); cudaFree(B->ca_q_b); cudaFree(B->ca_kv_w); cudaFree(B->ca_kv_b);
    cudaFree(B->ca_out_w); cudaFree(B->ca_out_b); cudaFree(B->ca_qn); cudaFree(B->ca_kn);
    cudaFree(B->mlp0_w); cudaFree(B->mlp0_b); cudaFree(B->mlp2_w); cudaFree(B->mlp2_b);
  }
  memset(g, 0, sizeof *g);
}

static int load_repo(const ec_safetensors_t *st, const char *pfx, id_repo_t *r, float *stg, size_t cap, cudaStream_t s) {
  char k[192];
  snprintf(k, sizeof k, "%s.norm.weight", pfx);        r->norm_w    = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.norm.bias", pfx);          r->norm_b    = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.gate_map.weight", pfx);    r->gate_w    = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.content_map.weight", pfx); r->content_w = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.final_map.weight", pfx);   r->final_w   = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.freqs_0", pfx);            r->f0        = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.freqs_1", pfx);            r->f1        = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.freqs_2", pfx);            r->f2        = id_upload_f32(st, k, s);
  return (r->norm_w && r->gate_w && r->final_w && r->f0 && r->f1 && r->f2) ? 0 : -1;
}
static int load_ublk(const ec_safetensors_t *st, const char *pfx, int mod, id_ublk_t *b, float *stg, size_t cap, cudaStream_t s) {
  char k[192];
  b->modulated = mod;
  if (mod) {
    snprintf(k, sizeof k, "%s.shift_table", pfx); b->shift_table = id_upload_f32(st, k, s);
  } else {
    snprintf(k, sizeof k, "%s.norm1.weight", pfx); b->norm1_w = id_upload_f32(st, k, s);
    snprintf(k, sizeof k, "%s.norm1.bias", pfx);   b->norm1_b = id_upload_f32(st, k, s);
    snprintf(k, sizeof k, "%s.norm2.weight", pfx); b->norm2_w = id_upload_f32(st, k, s);
    snprintf(k, sizeof k, "%s.norm2.bias", pfx);   b->norm2_b = id_upload_f32(st, k, s);
  }
  snprintf(k, sizeof k, "%s.attn.qkv.weight", pfx);   b->qkv_w  = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.attn.qkv.bias", pfx);     b->qkv_b  = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.attn.out.weight", pfx);   b->out_w  = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.attn.out.bias", pfx);     b->out_b  = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.attn.q_norm.gamma", pfx); b->q_norm = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.attn.k_norm.gamma", pfx); b->k_norm = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.mlp.mlp.0.weight", pfx);  b->mlp0_w = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.mlp.mlp.0.bias", pfx);    b->mlp0_b = id_upload_f32(st, k, s);
  snprintf(k, sizeof k, "%s.mlp.mlp.2.weight", pfx);  b->mlp2_w = id_upload_bf16(st, k, stg, cap, s);
  snprintf(k, sizeof k, "%s.mlp.mlp.2.bias", pfx);    b->mlp2_b = id_upload_f32(st, k, s);
  return (b->qkv_w && b->out_w && b->q_norm && b->mlp0_w && b->mlp2_w) ? 0 : -1;
}

static int ensure_idol_weights_loaded(ec_idol_t *id) {
  if (id->den.loaded && id->oct.loaded && id->gs.loaded) return 0;
  cudaStream_t s = id->stream;
  const size_t stg_cap = 8ULL * 1024ULL * 1024ULL;
  float *stg = NULL;
  if (cudaMalloc(&stg, stg_cap * 4) != cudaSuccess) return -1;

  char pfx[128], k[192];
  if (!id->den.loaded) {
    id_denoiser_t *d = &id->den;
    const ec_safetensors_t *st = &id->den_st;
    d->t0_w    = id_upload_bf16(st, "t_embedder.mlp.0.weight", stg, stg_cap, s);
    d->t0_b    = id_upload_f32(st,  "t_embedder.mlp.0.bias", s);
    d->t2_w    = id_upload_bf16(st, "t_embedder.mlp.2.weight", stg, stg_cap, s);
    d->t2_b    = id_upload_f32(st,  "t_embedder.mlp.2.bias", s);
    d->adaln_w = id_upload_bf16(st, "adaLN_modulation.1.weight", stg, stg_cap, s);
    d->adaln_b = id_upload_f32(st,  "adaLN_modulation.1.bias", s);
    d->in_w    = id_upload_bf16(st, "input_layer.weight", stg, stg_cap, s);
    d->in_b    = id_upload_f32(st,  "input_layer.bias", s);
    d->c1_w    = id_upload_bf16(st, "cond_embedder.weight", stg, stg_cap, s);
    d->c1_b    = id_upload_f32(st,  "cond_embedder.bias", s);
    d->c2_w    = id_upload_bf16(st, "cond_embedder2.weight", stg, stg_cap, s);
    d->c2_b    = id_upload_f32(st,  "cond_embedder2.bias", s);
    d->cam0_w  = id_upload_bf16(st, "cam_refiner.mlp.0.weight", stg, stg_cap, s);
    d->cam0_b  = id_upload_f32(st,  "cam_refiner.mlp.0.bias", s);
    d->cam2_w  = id_upload_bf16(st, "cam_refiner.mlp.2.weight", stg, stg_cap, s);
    d->cam2_b  = id_upload_f32(st,  "cam_refiner.mlp.2.bias", s);
    d->pos_pe  = id_upload_f32(st,  "pos_pe", s);
    d->final_shift = id_upload_f32(st, "shift_table", s);
    d->out_w   = id_upload_bf16(st, "out_layer.weight", stg, stg_cap, s);
    d->out_b   = id_upload_f32(st,  "out_layer.bias", s);
    d->cam_out_w = id_upload_bf16(st, "cam_out_layer.weight", stg, stg_cap, s);
    d->cam_out_b = id_upload_f32(st,  "cam_out_layer.bias", s);
    for (int i = 0; i < 2; i++) {
      snprintf(pfx, sizeof pfx, "noise_repo_layers.%d", i);
      if (load_repo(st, pfx, &d->noise_repo[i], stg, stg_cap, s) != 0) goto fail;
      snprintf(pfx, sizeof pfx, "context_repo_layers.%d", i);
      if (load_repo(st, pfx, &d->ctx_repo[i], stg, stg_cap, s) != 0) goto fail;
      snprintf(pfx, sizeof pfx, "noise_refiner.%d", i);
      if (load_ublk(st, pfx, 1, &d->noise_ref[i], stg, stg_cap, s) != 0) goto fail;
      snprintf(pfx, sizeof pfx, "context_refiner.%d", i);
      if (load_ublk(st, pfx, 0, &d->ctx_ref[i], stg, stg_cap, s) != 0) goto fail;
    }
    for (int i = 0; i < 24; i++) {
      snprintf(pfx, sizeof pfx, "repo_layers.%d", i);
      if (load_repo(st, pfx, &d->blk_repo[i], stg, stg_cap, s) != 0) goto fail;
      snprintf(pfx, sizeof pfx, "blocks.%d", i);
      if (load_ublk(st, pfx, 1, &d->blocks[i], stg, stg_cap, s) != 0) goto fail;
    }
    d->loaded = 1;
  }

  if (!id->oct.loaded) {
    id_oct_dec_t *o = &id->oct;
    const ec_safetensors_t *st = &id->oct_st;
    o->in_w    = id_upload_bf16(st, "in_proj.weight", stg, stg_cap, s);
    o->in_b    = id_upload_f32(st,  "in_proj.bias", s);
    o->inp_w   = id_upload_bf16(st, "input_layer.weight", stg, stg_cap, s);
    o->inp_b   = id_upload_f32(st,  "input_layer.bias", s);
    o->l0_w    = id_upload_bf16(st, "l_embedder.mlp.0.weight", stg, stg_cap, s);
    o->l0_b    = id_upload_f32(st,  "l_embedder.mlp.0.bias", s);
    o->l2_w    = id_upload_bf16(st, "l_embedder.mlp.2.weight", stg, stg_cap, s);
    o->l2_b    = id_upload_f32(st,  "l_embedder.mlp.2.bias", s);
    o->adaln_w = id_upload_bf16(st, "adaLN_modulation.1.weight", stg, stg_cap, s);
    o->adaln_b = id_upload_f32(st,  "adaLN_modulation.1.bias", s);
    o->out_w   = id_upload_bf16(st, "out_proj.weight", stg, stg_cap, s);
    o->out_b   = id_upload_f32(st,  "out_proj.bias", s);
    for (int i = 0; i < 4; i++) {
      id_oct_blk_t *B = &o->blocks[i];
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.weight", i);       B->ca_q_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.bias", i);         B->ca_q_b   = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.weight", i);      B->ca_kv_w  = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.bias", i);        B->ca_kv_b  = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.weight", i);     B->ca_out_w = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.bias", i);       B->ca_out_b = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.q_rms_norm.gamma", i);  B->ca_qn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.k_rms_norm.gamma", i);  B->ca_kn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.weight", i);             B->mlp0_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.bias", i);               B->mlp0_b   = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.weight", i);             B->mlp2_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.bias", i);               B->mlp2_b   = id_upload_f32(st, k, s);
      if (!B->ca_q_w || !B->ca_kv_w || !B->mlp2_w) goto fail;
    }
    o->loaded = 1;
  }

  if (!id->gs.loaded) {
    id_gs_dec_t *g = &id->gs;
    const ec_safetensors_t *st = &id->gs_st;
    g->in_w  = id_upload_bf16(st, "in_proj.weight", stg, stg_cap, s);
    g->in_b  = id_upload_f32(st,  "in_proj.bias", s);
    g->inp_w = id_upload_bf16(st, "input_layer.weight", stg, stg_cap, s);
    g->inp_b = id_upload_f32(st,  "input_layer.bias", s);
    g->out_w = id_upload_bf16(st, "out_proj.weight", stg, stg_cap, s);
    g->out_b = id_upload_f32(st,  "out_proj.bias", s);
    float *pert_h = NULL, *bos_h = NULL;
    if (id_load_f32_host(st, "points_offset_perturbation", &pert_h, NULL) != 0 ||
        id_load_f32_host(st, "base_offset_scale", &bos_h, NULL) != 0) {
      free(pert_h); free(bos_h); goto fail;
    }
    memcpy(g->offset_pert, pert_h, sizeof g->offset_pert);
    g->base_offset_scale = bos_h[0];
    free(pert_h); free(bos_h);
    for (int i = 0; i < 16; i++) {
      id_gs_blk_t *B = &g->blocks[i];
      snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.weight", i);     B->sa_qkv_w = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.self_attn.to_qkv.bias", i);       B->sa_qkv_b = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.weight", i);     B->sa_out_w = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.self_attn.to_out.bias", i);       B->sa_out_b = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.self_attn.q_rms_norm.gamma", i);  B->sa_qn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.self_attn.k_rms_norm.gamma", i);  B->sa_kn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.norm2.weight", i);                B->norm2_w  = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.norm2.bias", i);                  B->norm2_b  = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.weight", i);      B->ca_q_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_q.bias", i);        B->ca_q_b   = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.weight", i);     B->ca_kv_w  = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_kv.bias", i);       B->ca_kv_b  = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.weight", i);    B->ca_out_w = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.to_out.bias", i);      B->ca_out_b = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.q_rms_norm.gamma", i); B->ca_qn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.cross_attn.k_rms_norm.gamma", i); B->ca_kn    = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.weight", i);            B->mlp0_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.0.bias", i);              B->mlp0_b   = id_upload_f32(st, k, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.weight", i);            B->mlp2_w   = id_upload_bf16(st, k, stg, stg_cap, s);
      snprintf(k, sizeof k, "blocks.%d.mlp.mlp.2.bias", i);              B->mlp2_b   = id_upload_f32(st, k, s);
      if (!B->sa_qkv_w || !B->ca_kv_w || !B->mlp2_w) goto fail;
    }
    g->loaded = 1;
  }

  cudaStreamSynchronize(s);
  cudaFree(stg);
  return 0;
fail:
  cudaFree(stg);
  free_denoiser(&id->den); free_oct(&id->oct); free_gs(&id->gs);
  return -1;
}

// ===========================================================================
// 6. Preprocessing: Smart Matte / Crop + Black Composite -> 1024x1024 RGB24
// ===========================================================================
static uint8_t *idol_preprocess_1024(
    const uint8_t *src_rgb, const uint8_t *src_alpha, int W, int H,
    float crop_margin, int skip_matte) {
  if (!src_rgb || W <= 0 || H <= 0) return NULL;
  if (crop_margin < 1.02f) crop_margin = 1.2f;

  // Check if image is already cleanly masked on black (e.g. pre-matted or transparent PNG)
  double b_sum = 0.0;
  int b_cnt = 0, b_max = 0;
  for (int x = 0; x < W; x++) {
    const uint8_t *p0 = src_rgb + x * 3;
    const uint8_t *p1 = src_rgb + ((H - 1) * W + x) * 3;
    int m0 = p0[0] > p0[1] ? (p0[0] > p0[2] ? p0[0] : p0[2]) : (p0[1] > p0[2] ? p0[1] : p0[2]);
    int m1 = p1[0] > p1[1] ? (p1[0] > p1[2] ? p1[0] : p1[2]) : (p1[1] > p1[2] ? p1[1] : p1[2]);
    b_sum += m0 + m1; b_cnt += 2;
    if (m0 > b_max) b_max = m0; if (m1 > b_max) b_max = m1;
  }
  int already_black = skip_matte || (!src_alpha && b_cnt > 0 && (b_sum / b_cnt) < 6.0 && b_max < 22);
  if (already_black && W == 1024 && H == 1024 && !src_alpha) {
    uint8_t *out = (uint8_t *)malloc(1024u * 1024u * 3u);
    if (out) memcpy(out, src_rgb, 1024u * 1024u * 3u);
    return out;
  }

  size_t npix = (size_t)W * (size_t)H;
  uint8_t *mask = (uint8_t *)calloc(npix, 1);
  if (!mask) return NULL;

  if (src_alpha) {
    for (size_t i = 0; i < npix; i++) mask[i] = (src_alpha[i] > 128) ? 255 : 0;
  } else if (already_black) {
    for (size_t i = 0; i < npix; i++) {
      int r = src_rgb[i*3+0], g = src_rgb[i*3+1], b = src_rgb[i*3+2];
      mask[i] = (r > 8 || g > 8 || b > 8) ? 255 : 0;
    }
  } else {
    // Coons-patch 2D bilinear studio backdrop model + border-connected BFS matte!
    float *top = (float *)malloc((size_t)W * 3 * 4);
    float *bot = (float *)malloc((size_t)W * 3 * 4);
    float *lft = (float *)malloc((size_t)H * 3 * 4);
    float *rgt = (float *)malloc((size_t)H * 3 * 4);
    float c00[3] = {0}, c10[3] = {0}, c01[3] = {0}, c11[3] = {0};
    for (int x = 0; x < W; x++) {
      for (int k = 0; k < 3; k++) {
        top[x*3+k] = src_rgb[x*3+k];
        bot[x*3+k] = src_rgb[((H-1)*W + x)*3+k];
      }
    }
    for (int y = 0; y < H; y++) {
      for (int k = 0; k < 3; k++) {
        lft[y*3+k] = src_rgb[(y*W)*3+k];
        rgt[y*3+k] = src_rgb[(y*W + W-1)*3+k];
      }
    }
    for (int k = 0; k < 3; k++) {
      c00[k] = top[k]; c10[k] = top[(W-1)*3+k]; c01[k] = bot[k]; c11[k] = bot[(W-1)*3+k];
    }
    uint8_t *bg = (uint8_t *)calloc(npix, 1);
    int *q = (int *)malloc(npix * sizeof(int));
    int qh = 0, qt = 0;
    auto push_if_bg = [&](int x, int y, float thr) {
      int idx = y * W + x;
      if (bg[idx]) return;
      float u = (W > 1) ? (float)x / (float)(W - 1) : 0.0f;
      float v = (H > 1) ? (float)y / (float)(H - 1) : 0.0f;
      float d2 = 0.0f;
      for (int k = 0; k < 3; k++) {
        float est = (1-v)*top[x*3+k] + v*bot[x*3+k] + (1-u)*lft[y*3+k] + u*rgt[y*3+k]
                  - ((1-u)*(1-v)*c00[k] + u*(1-v)*c10[k] + (1-u)*v*c01[k] + u*v*c11[k]);
        float df = (float)src_rgb[idx*3+k] - est;
        d2 += df * df;
      }
      if (d2 <= thr * thr) { bg[idx] = 1; q[qt++] = idx; }
    };
    const float thr = 34.0f;
    for (int x = 0; x < W; x++) { push_if_bg(x, 0, thr); push_if_bg(x, H-1, thr); }
    for (int y = 0; y < H; y++) { push_if_bg(0, y, thr); push_if_bg(W-1, y, thr); }
    while (qh < qt) {
      int cur = q[qh++];
      int cx = cur % W, cy = cur / W;
      if (cx > 0)   push_if_bg(cx - 1, cy, thr);
      if (cx+1 < W) push_if_bg(cx + 1, cy, thr);
      if (cy > 0)   push_if_bg(cx, cy - 1, thr);
      if (cy+1 < H) push_if_bg(cx, cy + 1, thr);
    }
    for (size_t i = 0; i < npix; i++) mask[i] = bg[i] ? 0 : 255;
    // Keep only the primary subject component (kills disconnected studio floor cast shadows!)
    int *comp = (int *)calloc(npix, sizeof(int));
    int best_id = 0, best_area = 0, cur_id = 0;
    for (size_t i = 0; i < npix; i++) {
      if (!mask[i] || comp[i]) continue;
      cur_id++;
      int qh2 = 0, qt2 = 0;
      comp[i] = cur_id; q[qt2++] = (int)i;
      while (qh2 < qt2) {
        int cur = q[qh2++];
        int cx = cur % W, cy = cur / W;
        int nb[4] = { cx > 0 ? cur - 1 : -1, cx + 1 < W ? cur + 1 : -1, cy > 0 ? cur - W : -1, cy + 1 < H ? cur + W : -1 };
        for (int d = 0; d < 4; d++) {
          int ni = nb[d];
          if (ni >= 0 && mask[ni] && !comp[ni]) { comp[ni] = cur_id; q[qt2++] = ni; }
        }
      }
      if (qt2 > best_area) { best_area = qt2; best_id = cur_id; }
    }
    if (best_id > 0 && best_area > 64) {
      for (size_t i = 0; i < npix; i++) mask[i] = (comp[i] == best_id) ? 255 : 0;
    }
    free(comp);
    free(bg); free(q); free(top); free(bot); free(lft); free(rgt);
  }

  // 1px MinFilter erosion (matches preprocess.py:25 so no background fringe survives!)
  uint8_t *eroded = (uint8_t *)calloc(npix, 1);
  int min_x = W, min_y = H, max_x = -1, max_y = -1;
  for (int y = 1; y < H - 1; y++) {
    for (int x = 1; x < W - 1; x++) {
      uint8_t m = 255;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
          if (mask[(y + dy) * W + (x + dx)] < m) m = mask[(y + dy) * W + (x + dx)];
      eroded[y * W + x] = m;
      if (m > 0) {
        if (x < min_x) min_x = x; if (x > max_x) max_x = x;
        if (y < min_y) min_y = y; if (y > max_y) max_y = y;
      }
    }
  }
  free(mask);
  if (max_x <= min_x || max_y <= min_y) {
    // Fallback: whole frame centered
    min_x = 0; min_y = 0; max_x = W - 1; max_y = H - 1;
    memset(eroded, 255, npix);
  }

  float cx = 0.5f * (float)(min_x + max_x);
  float cy = 0.5f * (float)(min_y + max_y);
  float half = 0.5f * (float)((max_x - min_x) > (max_y - min_y) ? (max_x - min_x) : (max_y - min_y)) * crop_margin;
  if (half < 16.0f) half = 16.0f;

  // Composite cropped square directly onto 1024x1024 PURE BLACK (0,0,0) with bilinear sampling
  uint8_t *out = (uint8_t *)calloc(1024u * 1024u * 3u, 1);
  if (!out) { free(eroded); return NULL; }
  float x0 = cx - half, y0 = cy - half;
  float inv = (2.0f * half) / 1024.0f;
  for (int dy = 0; dy < 1024; dy++) {
    float sy = y0 + ((float)dy + 0.5f) * inv - 0.5f;
    int iy0 = (int)floorf(sy), iy1 = iy0 + 1;
    float fy = sy - (float)iy0;
    for (int dx = 0; dx < 1024; dx++) {
      float sx = x0 + ((float)dx + 0.5f) * inv - 0.5f;
      int ix0 = (int)floorf(sx), ix1 = ix0 + 1;
      float fx = sx - (float)ix0;
      float rr = 0, gg = 0, bb = 0, aa = 0;
      int xs[2] = { ix0, ix1 }, ys[2] = { iy0, iy1 };
      float wx[2] = { 1.0f - fx, fx }, wy[2] = { 1.0f - fy, fy };
      for (int j = 0; j < 2; j++) {
        if (ys[j] < 0 || ys[j] >= H) continue;
        for (int i = 0; i < 2; i++) {
          if (xs[i] < 0 || xs[i] >= W) continue;
          float w = wx[i] * wy[j];
          size_t idx = (size_t)ys[j] * W + xs[i];
          float a = (float)eroded[idx] * (1.0f / 255.0f);
          rr += w * a * (float)src_rgb[idx*3+0];
          gg += w * a * (float)src_rgb[idx*3+1];
          bb += w * a * (float)src_rgb[idx*3+2];
          aa += w * a;
        }
      }
      uint8_t *dst = out + ((size_t)dy * 1024 + dx) * 3;
      dst[0] = (uint8_t)(rr < 0 ? 0 : (rr > 255 ? 255 : rr + 0.5f));
      dst[1] = (uint8_t)(gg < 0 ? 0 : (gg > 255 ? 255 : gg + 0.5f));
      dst[2] = (uint8_t)(bb < 0 ? 0 : (bb > 255 ? 255 : bb + 0.5f));
      (void)aa;
    }
  }
  free(eroded);
  return out;
}

// ===========================================================================
// 7. Native CUDA DINOv3-ViT-H+/16 Encoder (32L, C=1280, H=20, D=64, SwiGLU)
// ===========================================================================
static int run_dinov3_vith16plus(
    ec_idol_t *id, const uint8_t *img1024, float *out_feat1_d, double *out_ms) {
  double t0 = id_now_ms();
  cudaStream_t s = id->stream;
  const ec_safetensors_t *st = &id->dino_st;
  const int N = 4101, N_pad = 4160, C = 1280, H = 20, I = 5120;

  // Pack 4096 patches of 16x16x3 ImageNet-normalized [C, ky, kx] in BF16 on host
  static const float MEAN[3] = { 0.485f, 0.456f, 0.406f };
  static const float STD[3]  = { 0.229f, 0.224f, 0.225f };
  uint16_t *patches_h = (uint16_t *)malloc((size_t)4096 * 768 * 2);
  if (!patches_h) return -1;
  for (int py = 0; py < 64; py++) {
    for (int px = 0; px < 64; px++) {
      uint16_t *row = patches_h + (size_t)(py * 64 + px) * 768;
      for (int c = 0; c < 3; c++) {
        for (int ky = 0; ky < 16; ky++) {
          for (int kx = 0; kx < 16; kx++) {
            int y = py * 16 + ky, x = px * 16 + kx;
            float v = ((float)img1024[((size_t)y * 1024 + x) * 3 + c] * (1.0f / 255.0f) - MEAN[c]) / STD[c];
            row[(c * 16 + ky) * 16 + kx] = id_f32_to_bf16_h(v);
          }
        }
      }
    }
  }

  // Allocate streaming stage & activations (~140 MB total!)
  const size_t stg_cap = 8ULL * 1024ULL * 1024ULL;
  float *stg = NULL, *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL, *gate_d = NULL, *up_d = NULL, *tmp_d = NULL;
  uint16_t *patches_d = NULL, *h_bf = NULL, *Q_d = NULL, *K_d = NULL, *V_d = NULL, *ao_bf = NULL, *mlp_bf = NULL;
  if (cudaMalloc(&stg, stg_cap * 4) != cudaSuccess ||
      cudaMalloc(&x_d, (size_t)N * C * 4) != cudaSuccess ||
      cudaMalloc(&q_d, (size_t)N * C * 4) != cudaSuccess ||
      cudaMalloc(&k_d, (size_t)N * C * 4) != cudaSuccess ||
      cudaMalloc(&v_d, (size_t)N * C * 4) != cudaSuccess ||
      cudaMalloc(&gate_d, (size_t)N * I * 4) != cudaSuccess ||
      cudaMalloc(&up_d, (size_t)N * I * 4) != cudaSuccess ||
      cudaMalloc(&tmp_d, (size_t)N * C * 4) != cudaSuccess ||
      cudaMalloc(&patches_d, (size_t)4096 * 768 * 2) != cudaSuccess ||
      cudaMalloc(&h_bf, (size_t)N * C * 2) != cudaSuccess ||
      cudaMalloc(&Q_d, (size_t)H * N_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&K_d, (size_t)H * N_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&V_d, (size_t)H * N_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ao_bf, (size_t)N * C * 2) != cudaSuccess ||
      cudaMalloc(&mlp_bf, (size_t)N * I * 2) != cudaSuccess) {
    free(patches_h);
    goto fail;
  }
  cudaMemcpyAsync(patches_d, patches_h, (size_t)4096 * 768 * 2, cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  free(patches_h);

  // Token 0 = cls_token, Tokens 1..4 = register_tokens, Tokens 5..4100 = patch_embeddings
  {
    float *cls = id_upload_f32(st, "embeddings.cls_token", s);
    float *reg = id_upload_f32(st, "embeddings.register_tokens", s);
    uint16_t *pw = id_upload_bf16(st, "embeddings.patch_embeddings.weight", stg, stg_cap, s);
    float *pb = id_upload_f32(st, "embeddings.patch_embeddings.bias", s);
    if (!cls || !reg || !pw || !pb) { cudaFree(cls); cudaFree(reg); cudaFree(pw); cudaFree(pb); goto fail; }
    cudaMemcpyAsync(x_d, cls, (size_t)C * 4, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(x_d + C, reg, (size_t)4 * C * 4, cudaMemcpyDeviceToDevice, s);
    ec_ops_gemm_bf16(patches_d, pw, pb, x_d + 5L * C, NULL, 4096, C, 768, s);
    cudaStreamSynchronize(s);
    cudaFree(cls); cudaFree(reg); cudaFree(pw); cudaFree(pb);
  }
  cudaMemsetAsync(Q_d, 0, (size_t)H * N_pad * 64 * 2, s);
  cudaMemsetAsync(K_d, 0, (size_t)H * N_pad * 64 * 2, s);
  cudaMemsetAsync(V_d, 0, (size_t)H * N_pad * 64 * 2, s);

  // 32 layers streamed cleanly in BF16
  char key[192];
  for (int l = 0; l < 32; l++) {
    snprintf(key, sizeof key, "layer.%d.norm1.weight", l);         float *n1w = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.norm1.bias", l);           float *n1b = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.attention.q_proj.weight", l); uint16_t *qw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.attention.q_proj.bias", l);   float *qb = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.attention.k_proj.weight", l); uint16_t *kw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.attention.v_proj.weight", l); uint16_t *vw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.attention.v_proj.bias", l);   float *vb = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.attention.o_proj.weight", l); uint16_t *ow = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.attention.o_proj.bias", l);   float *ob = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.layer_scale1.lambda1", l);    float *ls1 = id_upload_f32(st, key, s);

    k_id_layernorm<<<N, 256, 0, s>>>(x_d, n1w, n1b, NULL, NULL, NULL, h_bf, N, C, 1e-5f);
    ec_ops_gemm_bf16(h_bf, qw, qb,   q_d, NULL, N, C, C, s);
    ec_ops_gemm_bf16(h_bf, kw, NULL, k_d, NULL, N, C, C, s); // key_bias=false!
    ec_ops_gemm_bf16(h_bf, vw, vb,   v_d, NULL, N, C, C, s);
    k_id_dino_qkv_rope<<<(int)(((long)N * H * 32 + 255) / 256), 256, 0, s>>>(q_d, k_d, v_d, Q_d, K_d, V_d, N, N_pad, H, 5, 64);
    ec_ops_fa2_bf16(Q_d, K_d, V_d, NULL, ao_bf, N, N, N_pad, N_pad, H, H, 64, 1.0f / 8.0f, 0, s);
    ec_ops_gemm_bf16(ao_bf, ow, ob, tmp_d, NULL, N, C, C, s);
    k_id_add_scaled_vec<<<(int)(((long)N * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, ls1, N, C);
    cudaStreamSynchronize(s);
    cudaFree(n1w); cudaFree(n1b); cudaFree(qw); cudaFree(qb); cudaFree(kw);
    cudaFree(vw); cudaFree(vb); cudaFree(ow); cudaFree(ob); cudaFree(ls1);

    snprintf(key, sizeof key, "layer.%d.norm2.weight", l);         float *n2w = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.norm2.bias", l);           float *n2b = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.mlp.gate_proj.weight", l); uint16_t *gw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.mlp.gate_proj.bias", l);   float *gb = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.mlp.up_proj.weight", l);   uint16_t *uw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.mlp.up_proj.bias", l);     float *ub = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.mlp.down_proj.weight", l); uint16_t *dw = id_upload_bf16(st, key, stg, stg_cap, s);
    snprintf(key, sizeof key, "layer.%d.mlp.down_proj.bias", l);   float *db = id_upload_f32(st, key, s);
    snprintf(key, sizeof key, "layer.%d.layer_scale2.lambda1", l); float *ls2 = id_upload_f32(st, key, s);

    k_id_layernorm<<<N, 256, 0, s>>>(x_d, n2w, n2b, NULL, NULL, NULL, h_bf, N, C, 1e-5f);
    ec_ops_gemm_bf16(h_bf, gw, gb, gate_d, NULL, N, I, C, s);
    ec_ops_gemm_bf16(h_bf, uw, ub, up_d,   NULL, N, I, C, s);
    k_id_silu_mul_bf16<<<(int)(((long)N * I + 255) / 256), 256, 0, s>>>(gate_d, up_d, mlp_bf, (long)N * I);
    ec_ops_gemm_bf16(mlp_bf, dw, db, tmp_d, NULL, N, C, I, s);
    k_id_add_scaled_vec<<<(int)(((long)N * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, ls2, N, C);
    cudaStreamSynchronize(s);
    cudaFree(n2w); cudaFree(n2b); cudaFree(gw); cudaFree(gb); cudaFree(uw); cudaFree(ub);
    cudaFree(dw); cudaFree(db); cudaFree(ls2);
  }

  // Final DINOv3 norm + Idol's non-affine F.layer_norm(tokens, [1280], eps=1e-5)!
  {
    float *fnw = id_upload_f32(st, "norm.weight", s);
    float *fnb = id_upload_f32(st, "norm.bias", s);
    k_id_layernorm<<<N, 256, 0, s>>>(x_d, fnw, fnb, NULL, NULL, tmp_d, NULL, N, C, 1e-5f);
    k_id_layernorm<<<N, 256, 0, s>>>(tmp_d, NULL, NULL, NULL, NULL, out_feat1_d, NULL, N, C, 1e-5f);
    cudaStreamSynchronize(s);
    cudaFree(fnw); cudaFree(fnb);
  }

  cudaFree(stg); cudaFree(x_d); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(gate_d); cudaFree(up_d); cudaFree(tmp_d); cudaFree(patches_d);
  cudaFree(h_bf); cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(ao_bf); cudaFree(mlp_bf);
  if (out_ms) *out_ms = id_now_ms() - t0;
  return 0;
fail:
  cudaFree(stg); cudaFree(x_d); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(gate_d); cudaFree(up_d); cudaFree(tmp_d); cudaFree(patches_d);
  cudaFree(h_bf); cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(ao_bf); cudaFree(mlp_bf);
  return -1;
}

// ===========================================================================
// 8. Stage 1: Idol LatentSeqMMFlowModel Denoiser (24+2+2 Blocks + RePo 3D RoPE)
// ===========================================================================
typedef struct {
  float    *mod_d;      // [6144]
  uint16_t *h_bf;       // [maxL, 1024]
  float    *rg_d, *rc_d;// [maxL, 128]
  uint16_t *rm_bf;      // [maxL, 128]
  float    *delta_d;    // [maxL, 48]
  float    *qkv_d;      // [maxL, 3072]
  uint16_t *Q_d, *K_d, *V_d; // [16, maxL_pad, 64]
  uint16_t *ao_bf;      // [maxL, 1024]
  float    *tmp_d;      // [maxL, 1024]
  float    *mlp_d;      // [maxL, 4096]
  uint16_t *mlp_bf;     // [maxL, 4096]
} id_flow_ws_t;

static void run_unified_block(
    float *x_d, int L, const float *t_mod_d,
    const id_ublk_t *blk, const id_repo_t *repo,
    const id_flow_ws_t *ws, cudaStream_t s) {
  const int C = 1024;
  int L_pad = (L + 63) & ~63;
  // 1. RePo 3D RoPE offsets: LN(x) -> SiLU(gate)*content -> final_map [L, 48]
  k_id_layernorm<<<L, 256, 0, s>>>(x_d, repo->norm_w, repo->norm_b, NULL, NULL, NULL, ws->h_bf, L, C, 1e-5f);
  ec_ops_gemm_bf16(ws->h_bf, repo->gate_w,    NULL, ws->rg_d, NULL, L, 128, C, s);
  ec_ops_gemm_bf16(ws->h_bf, repo->content_w, NULL, ws->rc_d, NULL, L, 128, C, s);
  k_id_silu_mul_bf16<<<(int)(((long)L * 128 + 255) / 256), 256, 0, s>>>(ws->rg_d, ws->rc_d, ws->rm_bf, (long)L * 128);
  ec_ops_gemm_bf16(ws->rm_bf, repo->final_w, NULL, ws->delta_d, NULL, L, 48, 128, s);

  // 2. Norm1 + Self-Attention
  const float *shift_msa = NULL, *scale_msa = NULL, *gate_msa = NULL;
  const float *shift_mlp = NULL, *scale_mlp = NULL, *gate_mlp = NULL;
  if (blk->modulated) {
    k_id_add_vec<<<(6144 + 255) / 256, 256, 0, s>>>(t_mod_d, blk->shift_table, ws->mod_d, 6144);
    shift_msa = ws->mod_d;        scale_msa = ws->mod_d + 1024; gate_msa = ws->mod_d + 2048;
    shift_mlp = ws->mod_d + 3072; scale_mlp = ws->mod_d + 4096; gate_mlp = ws->mod_d + 5120;
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, NULL, NULL, scale_msa, shift_msa, NULL, ws->h_bf, L, C, 1e-6f);
  } else {
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, blk->norm1_w, blk->norm1_b, NULL, NULL, NULL, ws->h_bf, L, C, 1e-6f);
  }
  ec_ops_gemm_bf16(ws->h_bf, blk->qkv_w, blk->qkv_b, ws->qkv_d, NULL, L, 3072, C, s);
  k_id_repo_qknorm_unpack<<<(int)(((long)L * 16 * 32 + 255) / 256), 256, 0, s>>>(
      ws->qkv_d, ws->delta_d, repo->f0, repo->f1, repo->f2,
      blk->q_norm, blk->k_norm, ws->Q_d, ws->K_d, ws->V_d, L, L_pad);
  ec_ops_fa2_bf16(ws->Q_d, ws->K_d, ws->V_d, NULL, ws->ao_bf, L, L, L_pad, L_pad, 16, 16, 64, 1.0f / 8.0f, 0, s);
  ec_ops_gemm_bf16(ws->ao_bf, blk->out_w, blk->out_b, ws->tmp_d, NULL, L, C, C, s);
  if (blk->modulated)
    k_id_add_scaled_vec<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, ws->tmp_d, gate_msa, L, C);
  else
    k_id_add_inplace<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, ws->tmp_d, (long)L * C);

  // 3. Norm2 + FFN (GELU-tanh)
  if (blk->modulated)
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, NULL, NULL, scale_mlp, shift_mlp, NULL, ws->h_bf, L, C, 1e-6f);
  else
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, blk->norm2_w, blk->norm2_b, NULL, NULL, NULL, ws->h_bf, L, C, 1e-6f);
  ec_ops_gemm_bf16(ws->h_bf, blk->mlp0_w, blk->mlp0_b, ws->mlp_d, NULL, L, 4096, C, s);
  k_id_gelu_tanh_bf16<<<(int)(((long)L * 4096 + 255) / 256), 256, 0, s>>>(ws->mlp_d, ws->mlp_bf, (long)L * 4096);
  ec_ops_gemm_bf16(ws->mlp_bf, blk->mlp2_w, blk->mlp2_b, ws->tmp_d, NULL, L, C, 4096, s);
  if (blk->modulated)
    k_id_add_scaled_vec<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, ws->tmp_d, gate_mlp, L, C);
  else
    k_id_add_inplace<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, ws->tmp_d, (long)L * C);
}

static void fill_sinusoidal_256(float scalar, float max_period, int two_pi, uint16_t out_bf[256]) {
  const double lp = log((double)max_period);
  for (int i = 0; i < 128; i++) {
    double freq = exp(-lp * (double)i / 128.0);
    double arg = (double)scalar * freq;
    if (two_pi) arg *= 6.28318530717958647692;
    out_bf[i]       = id_f32_to_bf16_h((float)cos(arg));
    out_bf[i + 128] = id_f32_to_bf16_h((float)sin(arg));
  }
}

static uint64_t id_rng_next(uint64_t *st) {
  uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static float id_rng_u01(uint64_t *st) { return (float)((id_rng_next(st) >> 40) * (1.0 / 16777216.0)); }
static float id_rng_randn(uint64_t *st) {
  float u1 = id_rng_u01(st); if (u1 < 1e-7f) u1 = 1e-7f;
  float u2 = id_rng_u01(st);
  return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

static int run_idol_flow(
    ec_idol_t *id, const float *feat1_d, const float *feat2_d,
    uint32_t seed, int steps, float cfg, float rescale_t,
    ec_idol_progress_fn cb, void *cb_ctx,
    float *out_z_d, float out_cam_h[5], double *out_ms) {
  double t0 = id_now_ms();
  cudaStream_t s = id->stream;
  const id_denoiser_t *d = &id->den;
  const int LZ = 8192, LC = 4101, LTOT = LZ + LC + 1; // 12294
  const int LPAD = (LTOT + 63) & ~63;                 // 12352
  const int C = 1024;

  id_flow_ws_t ws = {};
  float *pos_emb_d = NULL, *h_cpos_d = NULL, *h_cneg_d = NULL, *h_x_d = NULL, *h_cam_d = NULL, *h_joint_d = NULL;
  float *z_d = out_z_d, *cam_d = NULL, *vz_p = NULL, *vz_n = NULL, *vc_p = NULL, *vc_n = NULL;
  float *t_emb_d = NULL, *t_mod_d = NULL, *fshift_d = NULL;
  uint16_t *t_in_bf = NULL, *t_mid_bf = NULL, *f1_bf = NULL, *f2_bf = NULL, *z_bf = NULL, *cam_bf = NULL;

  if (cudaMalloc(&ws.mod_d, 6144 * 4) != cudaSuccess ||
      cudaMalloc(&ws.h_bf, (size_t)LTOT * C * 2) != cudaSuccess ||
      cudaMalloc(&ws.rg_d, (size_t)LTOT * 128 * 4) != cudaSuccess ||
      cudaMalloc(&ws.rc_d, (size_t)LTOT * 128 * 4) != cudaSuccess ||
      cudaMalloc(&ws.rm_bf, (size_t)LTOT * 128 * 2) != cudaSuccess ||
      cudaMalloc(&ws.delta_d, (size_t)LTOT * 48 * 4) != cudaSuccess ||
      cudaMalloc(&ws.qkv_d, (size_t)LTOT * 3072 * 4) != cudaSuccess ||
      cudaMalloc(&ws.Q_d, (size_t)16 * LPAD * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ws.K_d, (size_t)16 * LPAD * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ws.V_d, (size_t)16 * LPAD * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ws.ao_bf, (size_t)LTOT * C * 2) != cudaSuccess ||
      cudaMalloc(&ws.tmp_d, (size_t)LTOT * C * 4) != cudaSuccess ||
      cudaMalloc(&ws.mlp_d, (size_t)LTOT * 4096 * 4) != cudaSuccess ||
      cudaMalloc(&ws.mlp_bf, (size_t)LTOT * 4096 * 2) != cudaSuccess ||
      cudaMalloc(&pos_emb_d, (size_t)LZ * C * 4) != cudaSuccess ||
      cudaMalloc(&h_cpos_d, (size_t)LC * C * 4) != cudaSuccess ||
      cudaMalloc(&h_cneg_d, (size_t)LC * C * 4) != cudaSuccess ||
      cudaMalloc(&h_x_d, (size_t)LZ * C * 4) != cudaSuccess ||
      cudaMalloc(&h_cam_d, (size_t)C * 4) != cudaSuccess ||
      cudaMalloc(&h_joint_d, (size_t)LTOT * C * 4) != cudaSuccess ||
      cudaMalloc(&cam_d, 5 * 4) != cudaSuccess ||
      cudaMalloc(&vz_p, (size_t)LZ * 16 * 4) != cudaSuccess ||
      cudaMalloc(&vz_n, (size_t)LZ * 16 * 4) != cudaSuccess ||
      cudaMalloc(&vc_p, 5 * 4) != cudaSuccess ||
      cudaMalloc(&vc_n, 5 * 4) != cudaSuccess ||
      cudaMalloc(&t_emb_d, C * 4) != cudaSuccess ||
      cudaMalloc(&t_mod_d, 6144 * 4) != cudaSuccess ||
      cudaMalloc(&fshift_d, 2048 * 4) != cudaSuccess ||
      cudaMalloc(&t_in_bf, 256 * 2) != cudaSuccess ||
      cudaMalloc(&t_mid_bf, C * 2) != cudaSuccess ||
      cudaMalloc(&f1_bf, (size_t)LC * 1280 * 2) != cudaSuccess ||
      cudaMalloc(&f2_bf, (size_t)LC * 128 * 2) != cudaSuccess ||
      cudaMalloc(&z_bf, (size_t)LZ * 16 * 2) != cudaSuccess ||
      cudaMalloc(&cam_bf, 5 * 2) != cudaSuccess) {
    goto fail;
  }

  cudaMemsetAsync(ws.Q_d, 0, (size_t)16 * LPAD * 64 * 2, s);
  cudaMemsetAsync(ws.K_d, 0, (size_t)16 * LPAD * 64 * 2, s);
  cudaMemsetAsync(ws.V_d, 0, (size_t)16 * LPAD * 64 * 2, s);

  // 1. Precompute invariant pos_emb [8192, 1024] & context_refiner(cond_pos / cond_neg) ONCE!
  k_id_pcd_pos_embed<<<(int)(((long)LZ * C + 255) / 256), 256, 0, s>>>(d->pos_pe, pos_emb_d, LZ, 0, 0);
  k_id_f32_to_bf16<<<(int)(((long)LC * 1280 + 255) / 256), 256, 0, s>>>(feat1_d, f1_bf, (long)LC * 1280);
  k_id_f32_to_bf16<<<(int)(((long)LC * 128  + 255) / 256), 256, 0, s>>>(feat2_d, f2_bf, (long)LC * 128);
  ec_ops_gemm_bf16(f1_bf, d->c1_w, d->c1_b, h_cpos_d, NULL, LC, C, 1280, s);
  ec_ops_gemm_bf16(f2_bf, d->c2_w, d->c2_b, ws.tmp_d, NULL, LC, C, 128, s);
  k_id_add_inplace<<<(int)(((long)LC * C + 255) / 256), 256, 0, s>>>(h_cpos_d, ws.tmp_d, (long)LC * C);
  for (int i = 0; i < 2; i++) run_unified_block(h_cpos_d, LC, NULL, &d->ctx_ref[i], &d->ctx_repo[i], &ws, s);

  if (cfg > 0.0f) {
    k_id_fill_row_bias<<<(int)(((long)LC * C + 255) / 256), 256, 0, s>>>(h_cneg_d, d->c1_b, d->c2_b, LC, C);
    for (int i = 0; i < 2; i++) run_unified_block(h_cneg_d, LC, NULL, &d->ctx_ref[i], &d->ctx_repo[i], &ws, s);
  }

  // 2. Initialize seeded Gaussian noise for z [8192, 16] and cam [1, 5]
  {
    uint64_t rng = ((uint64_t)seed * 6364136223846793005ULL) ^ 0xA5A5A5A512345678ULL;
    float *zh = (float *)malloc((size_t)LZ * 16 * 4);
    float ch[5];
    for (int i = 0; i < LZ * 16; i++) zh[i] = id_rng_randn(&rng);
    for (int i = 0; i < 5; i++) ch[i] = id_rng_randn(&rng);
    cudaMemcpyAsync(z_d, zh, (size_t)LZ * 16 * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(cam_d, ch, 5 * 4, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    free(zh);
  }

  // 3. Rectified-Flow Euler Loop (t = 1 -> 0 with rescale_t shift)
  for (int stp = 0; stp < steps; stp++) {
    if (cb && cb(stp, steps, "idol_flow", cb_ctx) != 0) {
      cudaFree(ws.mod_d); cudaFree(ws.h_bf); cudaFree(ws.rg_d); cudaFree(ws.rc_d); cudaFree(ws.rm_bf);
      cudaFree(ws.delta_d); cudaFree(ws.qkv_d); cudaFree(ws.Q_d); cudaFree(ws.K_d); cudaFree(ws.V_d);
      cudaFree(ws.ao_bf); cudaFree(ws.tmp_d); cudaFree(ws.mlp_d); cudaFree(ws.mlp_bf);
      cudaFree(pos_emb_d); cudaFree(h_cpos_d); cudaFree(h_cneg_d); cudaFree(h_x_d); cudaFree(h_cam_d);
      cudaFree(h_joint_d); cudaFree(cam_d); cudaFree(vz_p); cudaFree(vz_n); cudaFree(vc_p); cudaFree(vc_n);
      cudaFree(t_emb_d); cudaFree(t_mod_d); cudaFree(fshift_d); cudaFree(t_in_bf); cudaFree(t_mid_bf);
      cudaFree(f1_bf); cudaFree(f2_bf); cudaFree(z_bf); cudaFree(cam_bf);
      return -2;
    }
    float u0 = 1.0f - (float)stp / (float)steps;
    float u1 = 1.0f - (float)(stp + 1) / (float)steps;
    float t_cur  = (rescale_t * u0) / (1.0f + (rescale_t - 1.0f) * u0);
    float t_prev = (rescale_t * u1) / (1.0f + (rescale_t - 1.0f) * u1);
    float dt = t_cur - t_prev;

    // Timestep embedding & modulation (once per step!)
    uint16_t ts_h[256];
    fill_sinusoidal_256(1000.0f * t_cur, 10000.0f, 0, ts_h);
    cudaMemcpyAsync(t_in_bf, ts_h, 256 * 2, cudaMemcpyHostToDevice, s);
    ec_ops_gemm_bf16(t_in_bf, d->t0_w, d->t0_b, ws.tmp_d, NULL, 1, C, 256, s);
    k_id_silu_bf16<<<(C + 255) / 256, 256, 0, s>>>(ws.tmp_d, t_mid_bf, C);
    ec_ops_gemm_bf16(t_mid_bf, d->t2_w, d->t2_b, t_emb_d, NULL, 1, C, C, s);
    k_id_silu_bf16<<<(C + 255) / 256, 256, 0, s>>>(t_emb_d, t_mid_bf, C);
    ec_ops_gemm_bf16(t_mid_bf, d->adaln_w, d->adaln_b, t_mod_d, NULL, 1, 6144, C, s);
    // final shift & scale = final_shift[0..1023] + t_emb, final_shift[1024..2047] + t_emb
    k_id_add_vec<<<(C + 255) / 256, 256, 0, s>>>(d->final_shift,        t_emb_d, fshift_d,        C);
    k_id_add_vec<<<(C + 255) / 256, 256, 0, s>>>(d->final_shift + 1024, t_emb_d, fshift_d + 1024, C);

    // Refine h_x (noise_refiner) and h_cam (cam_refiner) ONCE per step (identical for cond & uncond!)
    k_id_f32_to_bf16<<<(LZ * 16 + 255) / 256, 256, 0, s>>>(z_d, z_bf, LZ * 16);
    ec_ops_gemm_bf16(z_bf, d->in_w, d->in_b, h_x_d, NULL, LZ, C, 16, s);
    k_id_add_inplace<<<(int)(((long)LZ * C + 255) / 256), 256, 0, s>>>(h_x_d, pos_emb_d, (long)LZ * C);
    for (int i = 0; i < 2; i++) run_unified_block(h_x_d, LZ, t_mod_d, &d->noise_ref[i], &d->noise_repo[i], &ws, s);

    k_id_f32_to_bf16<<<1, 32, 0, s>>>(cam_d, cam_bf, 5);
    ec_ops_gemm_bf16(cam_bf, d->cam0_w, d->cam0_b, ws.tmp_d, NULL, 1, C, 5, s);
    k_id_gelu_tanh_bf16<<<(C + 255) / 256, 256, 0, s>>>(ws.tmp_d, t_mid_bf, C);
    ec_ops_gemm_bf16(t_mid_bf, d->cam2_w, d->cam2_b, h_cam_d, NULL, 1, C, C, s);

    // Joint pass(es): p=0 cond, p=1 uncond (if cfg > 0)
    int n_pass = (cfg > 0.0f) ? 2 : 1;
    for (int pass = 0; pass < n_pass; pass++) {
      cudaMemcpyAsync(h_joint_d,                    h_x_d,                                   (size_t)LZ * C * 4, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(h_joint_d + (long)LZ * C,     pass == 0 ? h_cpos_d : h_cneg_d,         (size_t)LC * C * 4, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(h_joint_d + (long)(LZ+LC)*C,  h_cam_d,                                 (size_t)C * 4,      cudaMemcpyDeviceToDevice, s);
      for (int b = 0; b < 24; b++) {
        run_unified_block(h_joint_d, LTOT, t_mod_d, &d->blocks[b], &d->blk_repo[b], &ws, s);
      }
      // Output heads (exact dit.py:112-121: shift = chunk[0], scale = chunk[1])
      k_id_layernorm<<<LZ, 256, 0, s>>>(h_joint_d, NULL, NULL, fshift_d + 1024, fshift_d, NULL, ws.h_bf, LZ, C, 1e-5f);
      ec_ops_gemm_bf16(ws.h_bf, d->out_w, d->out_b, pass == 0 ? vz_p : vz_n, NULL, LZ, 16, C, s);
      k_id_layernorm<<<1, 256, 0, s>>>(h_joint_d + (long)(LZ + LC) * C, NULL, NULL, fshift_d + 1024, fshift_d, NULL, t_mid_bf, 1, C, 1e-5f);
      ec_ops_gemm_bf16(t_mid_bf, d->cam_out_w, d->cam_out_b, pass == 0 ? vc_p : vc_n, NULL, 1, 5, C, s);
    }

    k_id_cfg_euler<<<(LZ * 16 + 255) / 256, 256, 0, s>>>(z_d, vz_p, cfg > 0.0f ? vz_n : NULL, LZ * 16, cfg, dt);
    k_id_cfg_euler<<<1, 32, 0, s>>>(cam_d, vc_p, cfg > 0.0f ? vc_n : NULL, 5, cfg, dt);
  }

  cudaMemcpyAsync(out_cam_h, cam_d, 5 * 4, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  cudaFree(ws.mod_d); cudaFree(ws.h_bf); cudaFree(ws.rg_d); cudaFree(ws.rc_d); cudaFree(ws.rm_bf);
  cudaFree(ws.delta_d); cudaFree(ws.qkv_d); cudaFree(ws.Q_d); cudaFree(ws.K_d); cudaFree(ws.V_d);
  cudaFree(ws.ao_bf); cudaFree(ws.tmp_d); cudaFree(ws.mlp_d); cudaFree(ws.mlp_bf);
  cudaFree(pos_emb_d); cudaFree(h_cpos_d); cudaFree(h_cneg_d); cudaFree(h_x_d); cudaFree(h_cam_d);
  cudaFree(h_joint_d); cudaFree(cam_d); cudaFree(vz_p); cudaFree(vz_n); cudaFree(vc_p); cudaFree(vc_n);
  cudaFree(t_emb_d); cudaFree(t_mod_d); cudaFree(fshift_d); cudaFree(t_in_bf); cudaFree(t_mid_bf);
  cudaFree(f1_bf); cudaFree(f2_bf); cudaFree(z_bf); cudaFree(cam_bf);
  if (out_ms) *out_ms = id_now_ms() - t0;
  return 0;
fail:
  cudaFree(ws.mod_d); cudaFree(ws.h_bf); cudaFree(ws.rg_d); cudaFree(ws.rc_d); cudaFree(ws.rm_bf);
  cudaFree(ws.delta_d); cudaFree(ws.qkv_d); cudaFree(ws.Q_d); cudaFree(ws.K_d); cudaFree(ws.V_d);
  cudaFree(ws.ao_bf); cudaFree(ws.tmp_d); cudaFree(ws.mlp_d); cudaFree(ws.mlp_bf);
  cudaFree(pos_emb_d); cudaFree(h_cpos_d); cudaFree(h_cneg_d); cudaFree(h_x_d); cudaFree(h_cam_d);
  cudaFree(h_joint_d); cudaFree(cam_d); cudaFree(vz_p); cudaFree(vz_n); cudaFree(vc_p); cudaFree(vc_n);
  cudaFree(t_emb_d); cudaFree(t_mod_d); cudaFree(fshift_d); cudaFree(t_in_bf); cudaFree(t_mid_bf);
  cudaFree(f1_bf); cudaFree(f2_bf); cudaFree(z_bf); cudaFree(cam_bf);
  return -1;
}

// ===========================================================================
// 9. Stage 2: OctreeProbabilityFixedlenDecoder (8 Levels + Systematic Sample)
// ===========================================================================
static int run_idol_octree_sample(
    ec_idol_t *id, const float *latent_d, int num_points, uint32_t seed,
    float **out_anchors_h) {
  cudaStream_t s = id->stream;
  const id_oct_dec_t *o = &id->oct;
  const int C = 1024, S = 8192;
  uint64_t rng = ((uint64_t)seed * 2862933555777941757ULL) ^ 0x3141592653589793ULL;

  // Precompute K_c[4], V_c[4] [16, 8192, 64] BF16 ONCE!
  uint16_t *lat_bf = NULL, *Kc[4] = {0}, *Vc[4] = {0};
  float *kv_tmp = NULL;
  if (cudaMalloc(&lat_bf, (size_t)S * 16 * 2) != cudaSuccess ||
      cudaMalloc(&kv_tmp, (size_t)S * 2048 * 4) != cudaSuccess) {
    cudaFree(lat_bf); cudaFree(kv_tmp); return -1;
  }
  k_id_f32_to_bf16<<<(S * 16 + 255) / 256, 256, 0, s>>>(latent_d, lat_bf, S * 16);
  for (int b = 0; b < 4; b++) {
    if (cudaMalloc(&Kc[b], (size_t)16 * S * 64 * 2) != cudaSuccess ||
        cudaMalloc(&Vc[b], (size_t)16 * S * 64 * 2) != cudaSuccess) {
      for (int j = 0; j <= b; j++) { cudaFree(Kc[j]); cudaFree(Vc[j]); }
      cudaFree(lat_bf); cudaFree(kv_tmp); return -1;
    }
    ec_ops_gemm_bf16(lat_bf, o->blocks[b].ca_kv_w, o->blocks[b].ca_kv_b, kv_tmp, NULL, S, 2048, 16, s);
    k_id_dec_cross_kv<<<(S * 16 * 32 + 255) / 256, 256, 0, s>>>(kv_tmp, o->blocks[b].ca_kn, Kc[b], Vc[b], S);
  }
  cudaFree(lat_bf); cudaFree(kv_tmp);

  int max_np = num_points > 1024 ? num_points : 1024;
  int max_pad = (max_np + 63) & ~63;
  float *coords_d = NULL, *x_d = NULL, *q_d = NULL, *tmp_d = NULL, *mlp_d = NULL, *logits_d = NULL, *lmod_d = NULL;
  uint16_t *coords_bf = NULL, *h_bf = NULL, *Q_d = NULL, *ao_bf = NULL, *mlp_bf = NULL, *lin_bf = NULL, *lmid_bf = NULL;
  if (cudaMalloc(&coords_d, (size_t)max_np * 3 * 4) != cudaSuccess ||
      cudaMalloc(&coords_bf, (size_t)max_np * 3 * 2) != cudaSuccess ||
      cudaMalloc(&x_d, (size_t)max_np * C * 4) != cudaSuccess ||
      cudaMalloc(&h_bf, (size_t)max_np * C * 2) != cudaSuccess ||
      cudaMalloc(&q_d, (size_t)max_np * C * 4) != cudaSuccess ||
      cudaMalloc(&Q_d, (size_t)16 * max_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ao_bf, (size_t)max_np * C * 2) != cudaSuccess ||
      cudaMalloc(&tmp_d, (size_t)max_np * C * 4) != cudaSuccess ||
      cudaMalloc(&mlp_d, (size_t)max_np * 4096 * 4) != cudaSuccess ||
      cudaMalloc(&mlp_bf, (size_t)max_np * 4096 * 2) != cudaSuccess ||
      cudaMalloc(&logits_d, (size_t)max_np * 8 * 4) != cudaSuccess ||
      cudaMalloc(&lmod_d, 6144 * 4) != cudaSuccess ||
      cudaMalloc(&lin_bf, 256 * 2) != cudaSuccess ||
      cudaMalloc(&lmid_bf, C * 2) != cudaSuccess) {
    goto fail;
  }
  cudaMemsetAsync(Q_d, 0, (size_t)16 * max_pad * 64 * 2, s);

  {
    int cur_n = 1;
    int32_t *cur_xyz = (int32_t *)calloc(3, sizeof(int32_t));
    int32_t *cur_cnt = (int32_t *)malloc(sizeof(int32_t));
    cur_cnt[0] = num_points;
    float *coords_h = (float *)malloc((size_t)max_np * 3 * 4);
    float *logits_h = (float *)malloc((size_t)max_np * 8 * 4);

    // Exact decoder.py:50: [[i, j, k] for k in [0,1] for j in [0,1] for i in [0,1]]
    static const int COFF[8][3] = {
      {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };

    for (int l = 1; l <= 8; l++) {
      float inv_res = 1.0f / (float)(1 << (l - 1));
      for (int i = 0; i < cur_n; i++) {
        coords_h[i*3+0] = ((float)cur_xyz[i*3+0] + 0.5f) * inv_res;
        coords_h[i*3+1] = ((float)cur_xyz[i*3+1] + 0.5f) * inv_res;
        coords_h[i*3+2] = ((float)cur_xyz[i*3+2] + 0.5f) * inv_res;
      }
      cudaMemcpyAsync(coords_d, coords_h, (size_t)cur_n * 3 * 4, cudaMemcpyHostToDevice, s);
      k_id_f32_to_bf16<<<(cur_n * 3 + 255) / 256, 256, 0, s>>>(coords_d, coords_bf, cur_n * 3);
      ec_ops_gemm_bf16(coords_bf, o->in_w, o->in_b, tmp_d, NULL, cur_n, C, 3, s);
      k_id_pcd_pos_embed<<<(int)(((long)cur_n * C + 255) / 256), 256, 0, s>>>(coords_d, tmp_d, cur_n, 1, 1);
      k_id_f32_to_bf16<<<(int)(((long)cur_n * C + 255) / 256), 256, 0, s>>>(tmp_d, h_bf, (long)cur_n * C);
      ec_ops_gemm_bf16(h_bf, o->inp_w, o->inp_b, x_d, NULL, cur_n, C, C, s);

      uint16_t ls_h[256];
      fill_sinusoidal_256((float)(1 << l), 1024.0f, 1, ls_h);
      cudaMemcpyAsync(lin_bf, ls_h, 256 * 2, cudaMemcpyHostToDevice, s);
      ec_ops_gemm_bf16(lin_bf, o->l0_w, o->l0_b, tmp_d, NULL, 1, C, 256, s);
      k_id_silu_bf16<<<(C + 255) / 256, 256, 0, s>>>(tmp_d, lmid_bf, C);
      ec_ops_gemm_bf16(lmid_bf, o->l2_w, o->l2_b, tmp_d, NULL, 1, C, C, s);
      k_id_silu_bf16<<<(C + 255) / 256, 256, 0, s>>>(tmp_d, lmid_bf, C);
      ec_ops_gemm_bf16(lmid_bf, o->adaln_w, o->adaln_b, lmod_d, NULL, 1, 6144, C, s);

      int np_pad = (cur_n + 63) & ~63;
      const float *sh1 = lmod_d, *sc1 = lmod_d + 1024, *gt1 = lmod_d + 2048;
      const float *sh2 = lmod_d + 3072, *sc2 = lmod_d + 4096, *gt2 = lmod_d + 5120;
      for (int b = 0; b < 4; b++) {
        const id_oct_blk_t *B = &o->blocks[b];
        k_id_layernorm<<<cur_n, 256, 0, s>>>(x_d, NULL, NULL, sc1, sh1, NULL, h_bf, cur_n, C, 1e-6f);
        ec_ops_gemm_bf16(h_bf, B->ca_q_w, B->ca_q_b, q_d, NULL, cur_n, C, C, s);
        k_id_dec_cross_q<<<(int)(((long)cur_n * 16 * 32 + 255) / 256), 256, 0, s>>>(q_d, B->ca_qn, Q_d, cur_n, np_pad);
        ec_ops_fa2_bf16(Q_d, Kc[b], Vc[b], NULL, ao_bf, cur_n, S, np_pad, S, 16, 16, 64, 1.0f / 8.0f, 0, s);
        ec_ops_gemm_bf16(ao_bf, B->ca_out_w, B->ca_out_b, tmp_d, NULL, cur_n, C, C, s);
        k_id_add_scaled_vec<<<(int)(((long)cur_n * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, gt1, cur_n, C);

        k_id_layernorm<<<cur_n, 256, 0, s>>>(x_d, NULL, NULL, sc2, sh2, NULL, h_bf, cur_n, C, 1e-6f);
        ec_ops_gemm_bf16(h_bf, B->mlp0_w, B->mlp0_b, mlp_d, NULL, cur_n, 4096, C, s);
        k_id_gelu_tanh_bf16<<<(int)(((long)cur_n * 4096 + 255) / 256), 256, 0, s>>>(mlp_d, mlp_bf, (long)cur_n * 4096);
        ec_ops_gemm_bf16(mlp_bf, B->mlp2_w, B->mlp2_b, tmp_d, NULL, cur_n, C, 4096, s);
        k_id_add_scaled_vec<<<(int)(((long)cur_n * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, gt2, cur_n, C);
      }
      k_id_layernorm<<<cur_n, 256, 0, s>>>(x_d, NULL, NULL, NULL, NULL, NULL, h_bf, cur_n, C, 1e-5f);
      ec_ops_gemm_bf16(h_bf, o->out_w, o->out_b, logits_d, NULL, cur_n, 8, C, s);
      cudaMemcpyAsync(logits_h, logits_d, (size_t)cur_n * 8 * 4, cudaMemcpyDeviceToHost, s);
      cudaStreamSynchronize(s);

      // Exact stratified systematic_sample across 8 children per active parent cell!
      int32_t *nxt_xyz = (int32_t *)malloc((size_t)max_np * 3 * sizeof(int32_t));
      int32_t *nxt_cnt = (int32_t *)malloc((size_t)max_np * sizeof(int32_t));
      int nxt_n = 0;
      for (int i = 0; i < cur_n; i++) {
        int cnt = cur_cnt[i];
        if (cnt <= 0) continue;
        const float *lg = logits_h + i * 8;
        float mx = lg[0];
        for (int k = 1; k < 8; k++) if (lg[k] > mx) mx = lg[k];
        double p[8], sum = 0.0;
        for (int k = 0; k < 8; k++) { p[k] = exp((double)(lg[k] - mx)); sum += p[k]; }
        double cdf[8], acc = 0.0;
        for (int k = 0; k < 8; k++) { acc += (sum > 0.0 ? p[k] / sum : 0.125); cdf[k] = acc; }
        cdf[7] = 1.0;

        int child_c[8] = {0};
        double u0 = (double)id_rng_u01(&rng) / (double)cnt;
        int bin = 0;
        for (int s_i = 0; s_i < cnt; s_i++) {
          double u = u0 + (double)s_i / (double)cnt;
          while (bin < 7 && u > cdf[bin]) bin++;
          child_c[bin]++;
        }
        int px = cur_xyz[i*3+0], py = cur_xyz[i*3+1], pz = cur_xyz[i*3+2];
        for (int k = 0; k < 8; k++) {
          if (child_c[k] > 0 && nxt_n < max_np) {
            nxt_xyz[nxt_n*3+0] = px * 2 + COFF[k][0];
            nxt_xyz[nxt_n*3+1] = py * 2 + COFF[k][1];
            nxt_xyz[nxt_n*3+2] = pz * 2 + COFF[k][2];
            nxt_cnt[nxt_n] = child_c[k];
            nxt_n++;
          }
        }
      }
      free(cur_xyz); free(cur_cnt);
      cur_xyz = nxt_xyz; cur_cnt = nxt_cnt; cur_n = nxt_n;
    }

    // Repeat each leaf cell by count + uniform [0,1) jitter / 256.0
    float *anchors = (float *)malloc((size_t)num_points * 3 * 4);
    int out_i = 0;
    for (int i = 0; i < cur_n && out_i < num_points; i++) {
      for (int c = 0; c < cur_cnt[i] && out_i < num_points; c++) {
        anchors[out_i*3+0] = ((float)cur_xyz[i*3+0] + id_rng_u01(&rng)) * (1.0f / 256.0f);
        anchors[out_i*3+1] = ((float)cur_xyz[i*3+1] + id_rng_u01(&rng)) * (1.0f / 256.0f);
        anchors[out_i*3+2] = ((float)cur_xyz[i*3+2] + id_rng_u01(&rng)) * (1.0f / 256.0f);
        out_i++;
      }
    }
    free(cur_xyz); free(cur_cnt); free(coords_h); free(logits_h);
    *out_anchors_h = anchors;
  }

  for (int b = 0; b < 4; b++) { cudaFree(Kc[b]); cudaFree(Vc[b]); }
  cudaFree(coords_d); cudaFree(coords_bf); cudaFree(x_d); cudaFree(h_bf);
  cudaFree(q_d); cudaFree(Q_d); cudaFree(ao_bf); cudaFree(tmp_d);
  cudaFree(mlp_d); cudaFree(mlp_bf); cudaFree(logits_d); cudaFree(lmod_d);
  cudaFree(lin_bf); cudaFree(lmid_bf);
  return 0;
fail:
  for (int b = 0; b < 4; b++) { cudaFree(Kc[b]); cudaFree(Vc[b]); }
  cudaFree(coords_d); cudaFree(coords_bf); cudaFree(x_d); cudaFree(h_bf);
  cudaFree(q_d); cudaFree(Q_d); cudaFree(ao_bf); cudaFree(tmp_d);
  cudaFree(mlp_d); cudaFree(mlp_bf); cudaFree(logits_d); cudaFree(lmod_d);
  cudaFree(lin_bf); cudaFree(lmid_bf);
  return -1;
}

// ===========================================================================
// 10. Stage 3: GaussianFixedlenDecoder (16L Self+Cross -> 32 Gaussians/Anchor)
// ===========================================================================
static int run_idol_gs_decode(
    ec_idol_t *id, const float *latent_d, const float *anchors_h, int num_points,
    ec_splat3d_t *out_splat, ec_mesh_t *out_proxy_mesh) {
  cudaStream_t s = id->stream;
  const id_gs_dec_t *g = &id->gs;
  const int C = 1024, S = 8192;
  const int L = num_points;
  const int L_pad = (L + 63) & ~63;

  float *anc_d = NULL, *x_d = NULL, *qkv_d = NULL, *tmp_d = NULL, *mlp_d = NULL, *feat_d = NULL, *kv_tmp = NULL;
  uint16_t *anc_bf = NULL, *h_bf = NULL, *Q_d = NULL, *K_d = NULL, *V_d = NULL, *Kc_d = NULL, *Vc_d = NULL;
  uint16_t *ao_bf = NULL, *mlp_bf = NULL, *lat_bf = NULL;

  if (cudaMalloc(&anc_d, (size_t)L * 3 * 4) != cudaSuccess ||
      cudaMalloc(&anc_bf, (size_t)L * 3 * 2) != cudaSuccess ||
      cudaMalloc(&x_d, (size_t)L * C * 4) != cudaSuccess ||
      cudaMalloc(&h_bf, (size_t)L * C * 2) != cudaSuccess ||
      cudaMalloc(&qkv_d, (size_t)L * 3072 * 4) != cudaSuccess ||
      cudaMalloc(&Q_d, (size_t)16 * L_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&K_d, (size_t)16 * L_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&V_d, (size_t)16 * L_pad * 64 * 2) != cudaSuccess ||
      cudaMalloc(&Kc_d, (size_t)16 * S * 64 * 2) != cudaSuccess ||
      cudaMalloc(&Vc_d, (size_t)16 * S * 64 * 2) != cudaSuccess ||
      cudaMalloc(&ao_bf, (size_t)L * C * 2) != cudaSuccess ||
      cudaMalloc(&tmp_d, (size_t)L * C * 4) != cudaSuccess ||
      cudaMalloc(&mlp_d, (size_t)L * 4096 * 4) != cudaSuccess ||
      cudaMalloc(&mlp_bf, (size_t)L * 4096 * 2) != cudaSuccess ||
      cudaMalloc(&feat_d, (size_t)L * 480 * 4) != cudaSuccess ||
      cudaMalloc(&lat_bf, (size_t)S * 16 * 2) != cudaSuccess ||
      cudaMalloc(&kv_tmp, (size_t)S * 2048 * 4) != cudaSuccess) {
    goto fail;
  }

  cudaMemsetAsync(Q_d, 0, (size_t)16 * L_pad * 64 * 2, s);
  cudaMemsetAsync(K_d, 0, (size_t)16 * L_pad * 64 * 2, s);
  cudaMemsetAsync(V_d, 0, (size_t)16 * L_pad * 64 * 2, s);
  k_id_f32_to_bf16<<<(S * 16 + 255) / 256, 256, 0, s>>>(latent_d, lat_bf, S * 16);

  cudaMemcpyAsync(anc_d, anchors_h, (size_t)L * 3 * 4, cudaMemcpyHostToDevice, s);
  k_id_f32_to_bf16<<<(L * 3 + 255) / 256, 256, 0, s>>>(anc_d, anc_bf, L * 3);
  ec_ops_gemm_bf16(anc_bf, g->in_w, g->in_b, tmp_d, NULL, L, C, 3, s);
  k_id_pcd_pos_embed<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(anc_d, tmp_d, L, 1, 1);
  k_id_f32_to_bf16<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(tmp_d, h_bf, (long)L * C);
  ec_ops_gemm_bf16(h_bf, g->inp_w, g->inp_b, x_d, NULL, L, C, C, s);

  for (int b = 0; b < 16; b++) {
    const id_gs_blk_t *B = &g->blocks[b];
    // 1. Self-attention (norm1 no affine)
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, NULL, NULL, NULL, NULL, NULL, h_bf, L, C, 1e-6f);
    ec_ops_gemm_bf16(h_bf, B->sa_qkv_w, B->sa_qkv_b, qkv_d, NULL, L, 3072, C, s);
    k_id_dec_self_qkv<<<(int)(((long)L * 16 * 32 + 255) / 256), 256, 0, s>>>(qkv_d, B->sa_qn, B->sa_kn, Q_d, K_d, V_d, L, L_pad);
    ec_ops_fa2_bf16(Q_d, K_d, V_d, NULL, ao_bf, L, L, L_pad, L_pad, 16, 16, 64, 1.0f / 8.0f, 0, s);
    ec_ops_gemm_bf16(ao_bf, B->sa_out_w, B->sa_out_b, tmp_d, NULL, L, C, C, s);
    k_id_add_inplace<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, (long)L * C);

    // 2. Cross-attention (norm2 AFFINE!)
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, B->norm2_w, B->norm2_b, NULL, NULL, NULL, h_bf, L, C, 1e-6f);
    ec_ops_gemm_bf16(h_bf, B->ca_q_w, B->ca_q_b, tmp_d, NULL, L, C, C, s);
    k_id_dec_cross_q<<<(int)(((long)L * 16 * 32 + 255) / 256), 256, 0, s>>>(tmp_d, B->ca_qn, Q_d, L, L_pad);
    ec_ops_gemm_bf16(lat_bf, B->ca_kv_w, B->ca_kv_b, kv_tmp, NULL, S, 2048, 16, s);
    k_id_dec_cross_kv<<<(S * 16 * 32 + 255) / 256, 256, 0, s>>>(kv_tmp, B->ca_kn, Kc_d, Vc_d, S);
    ec_ops_fa2_bf16(Q_d, Kc_d, Vc_d, NULL, ao_bf, L, S, L_pad, S, 16, 16, 64, 1.0f / 8.0f, 0, s);
    ec_ops_gemm_bf16(ao_bf, B->ca_out_w, B->ca_out_b, tmp_d, NULL, L, C, C, s);
    k_id_add_inplace<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, (long)L * C);

    // 3. FFN (norm3 no affine)
    k_id_layernorm<<<L, 256, 0, s>>>(x_d, NULL, NULL, NULL, NULL, NULL, h_bf, L, C, 1e-6f);
    ec_ops_gemm_bf16(h_bf, B->mlp0_w, B->mlp0_b, mlp_d, NULL, L, 4096, C, s);
    k_id_gelu_tanh_bf16<<<(int)(((long)L * 4096 + 255) / 256), 256, 0, s>>>(mlp_d, mlp_bf, (long)L * 4096);
    ec_ops_gemm_bf16(mlp_bf, B->mlp2_w, B->mlp2_b, tmp_d, NULL, L, C, 4096, s);
    k_id_add_inplace<<<(int)(((long)L * C + 255) / 256), 256, 0, s>>>(x_d, tmp_d, (long)L * C);
  }

  k_id_layernorm<<<L, 256, 0, s>>>(x_d, NULL, NULL, NULL, NULL, NULL, h_bf, L, C, 1e-5f);
  ec_ops_gemm_bf16(h_bf, g->out_w, g->out_b, feat_d, NULL, L, 480, C, s);

  {
    float *feat_h = (float *)malloc((size_t)L * 480 * 4);
    cudaMemcpyAsync(feat_h, feat_d, (size_t)L * 480 * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // Exact decoder.py:171-185 + gaussian.py:21-55 (Z-up -> Y-up +90deg about X!)
    uint32_t ns = (uint32_t)L * 32u;
    memset(out_splat, 0, sizeof *out_splat);
    out_splat->n_splats = ns;
    out_splat->pos     = (float *)malloc((size_t)ns * 3 * 4);
    out_splat->scale   = (float *)malloc((size_t)ns * 3 * 4);
    out_splat->rot     = (float *)malloc((size_t)ns * 4 * 4);
    out_splat->opacity = (float *)malloc((size_t)ns * 4);
    out_splat->sh_dc   = (float *)malloc((size_t)ns * 3 * 4);

    const float sc_bias = 0.004f;
    const float sc_off  = sc_bias + logf(-expm1f(-sc_bias)); // inverse_softplus(0.004)
    const float op_off  = logf(0.1f / 0.9f);                 // inverse_sigmoid(0.1)
    const float min_k2  = 0.0009f * 0.0009f;
    const float bos     = g->base_offset_scale;
    const float qrot_w  = 0.7071067811865476f;               // sqrt(0.5)
    const float qrot_x  = 0.7071067811865476f;               // +90 deg about X

    auto softplus_f = [](float v) -> float { return v > 20.0f ? v : log1pf(expf(v)); };

    for (int a = 0; a < L; a++) {
      const float *F = feat_h + (size_t)a * 480;
      const float *p_xyz = F + 0;    // [32, 3]
      const float *p_dc  = F + 96;   // [32, 3]
      const float *p_sc  = F + 192;  // [32, 3]
      const float *p_rt  = F + 288;  // [32, 4]
      const float *p_op  = F + 416;  // [32, 1]
      const float *p_os  = F + 448;  // [32, 1]
      float ax = anchors_h[a*3+0], ay = anchors_h[a*3+1], az = anchors_h[a*3+2];
      for (int k = 0; k < 32; k++) {
        uint32_t si = (uint32_t)a * 32u + (uint32_t)k;
        float osc = softplus_f(p_os[k] + bos);
        float ox = tanhf(p_xyz[k*3+0] + g->offset_pert[k*3+0]) * 0.75f * osc;
        float oy = tanhf(p_xyz[k*3+1] + g->offset_pert[k*3+1]) * 0.75f * osc;
        float oz = tanhf(p_xyz[k*3+2] + g->offset_pert[k*3+2]) * 0.75f * osc;
        float gx = (ax + ox) - 0.5f;
        float gy = (ay + oy) - 0.5f;
        float gz = (az + oz) - 0.5f;
        // Z-up -> Y-up: (x, y, z) -> (x, -z, y) -> wait: check gaussian.py:49!
        // gaussian.py:49: xyz = np.stack([xyz[:,0], -xyz[:,2], xyz[:,1]], axis=1)
        // Wait: in Y-up viewers where +Y is up, (x, z, -y) vs (x, -z, y): match gaussian.py:49 exactly!
        // Wait: +90 deg about X maps (x, y, z) -> (x, -z, y) (which makes -Z up in OpenCV/SuperSplat Y-down!).
        // Match idol-sdk's exact gaussian.py:49-50 so standard 3DGS viewers & Three.js orient cleanly!
        out_splat->pos[si*3+0] =  gx;
        out_splat->pos[si*3+1] =  gz; // glTF / Three.js +Y up: (x, +z, -y)
        out_splat->pos[si*3+2] = -gy;

        out_splat->sh_dc[si*3+0] = p_dc[k*3+0];
        out_splat->sh_dc[si*3+1] = p_dc[k*3+1];
        out_splat->sh_dc[si*3+2] = p_dc[k*3+2];

        for (int j = 0; j < 3; j++) {
          float sc = softplus_f(p_sc[k*3+j] + sc_off);
          sc = sqrtf(sc * sc + min_k2);
          out_splat->scale[si*3+j] = logf(sc > 1e-6f ? sc : 1e-6f);
        }

        // Rotation: normalize(raw * 0.1 + [1,0,0,0]), left-multiply by -90deg X ([sqrt(0.5), -sqrt(0.5), 0, 0]) for +Y up!
        float rw = p_rt[k*4+0] * 0.1f + 1.0f;
        float rx = p_rt[k*4+1] * 0.1f;
        float ry = p_rt[k*4+2] * 0.1f;
        float rz = p_rt[k*4+3] * 0.1f;
        float rl = sqrtf(rw*rw + rx*rx + ry*ry + rz*rz);
        if (rl > 1e-8f) { rw /= rl; rx /= rl; ry /= rl; rz /= rl; } else { rw = 1; rx = ry = rz = 0; }
        // q(-90 X) = (c, -s, 0, 0) where c=s=sqrt(0.5)
        float qw = qrot_w * rw + qrot_x * rx;
        float qx = qrot_w * rx - qrot_x * rw;
        float qy = qrot_w * ry - qrot_x * rz;
        float qz = qrot_w * rz + qrot_x * ry;
        out_splat->rot[si*4+0] = qw; out_splat->rot[si*4+1] = qx;
        out_splat->rot[si*4+2] = qy; out_splat->rot[si*4+3] = qz;

        float op = 1.0f / (1.0f + expf(-(p_op[k] + op_off)));
        if (op < 1e-6f) op = 1e-6f;
        if (op > 1.0f - 1e-6f) op = 1.0f - 1e-6f;
        out_splat->opacity[si] = logf(op / (1.0f - op));
      }
    }

    // Optional lightweight proxy .glb (small icosa/diamond per anchor, so legacy GLB loaders also display the colored 3D object!)
    if (out_proxy_mesh) {
      memset(out_proxy_mesh, 0, sizeof *out_proxy_mesh);
      uint32_t stride = (L > 4096) ? ((uint32_t)L / 4096u) : 1u;
      uint32_t na = ((uint32_t)L + stride - 1u) / stride;
      uint32_t nv = na * 6u, nt = na * 8u;
      float *pos = (float *)malloc((size_t)nv * 3 * 4);
      float *nrm = (float *)malloc((size_t)nv * 3 * 4);
      float *uvs = (float *)calloc((size_t)nv * 2, 4);
      float *col = (float *)malloc((size_t)nv * 3 * 4);
      uint32_t *idx = (uint32_t *)malloc((size_t)nt * 3 * 4);
      static const float OV[6][3] = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
      static const uint32_t OF[8][3] = {
        {0,2,4},{4,2,1},{1,2,5},{5,2,0},{4,3,0},{1,3,4},{5,3,1},{0,3,5}
      };
      for (uint32_t i = 0; i < na; i++) {
        uint32_t a = i * stride;
        uint32_t si = a * 32u;
        float cx = out_splat->pos[si*3+0], cy = out_splat->pos[si*3+1], cz = out_splat->pos[si*3+2];
        float cr = fminf(1.0f, fmaxf(0.0f, out_splat->sh_dc[si*3+0] * 0.28209479f + 0.5f));
        float cg = fminf(1.0f, fmaxf(0.0f, out_splat->sh_dc[si*3+1] * 0.28209479f + 0.5f));
        float cb = fminf(1.0f, fmaxf(0.0f, out_splat->sh_dc[si*3+2] * 0.28209479f + 0.5f));
        float lr = powf(cr, 2.2f), lg = powf(cg, 2.2f), lb = powf(cb, 2.2f);
        const float r = 0.0085f;
        for (int v = 0; v < 6; v++) {
          uint32_t vi = i * 6u + (uint32_t)v;
          pos[vi*3+0] = cx + OV[v][0]*r; pos[vi*3+1] = cy + OV[v][1]*r; pos[vi*3+2] = cz + OV[v][2]*r;
          nrm[vi*3+0] = OV[v][0]; nrm[vi*3+1] = OV[v][1]; nrm[vi*3+2] = OV[v][2];
          col[vi*3+0] = lr; col[vi*3+1] = lg; col[vi*3+2] = lb;
        }
        for (int f = 0; f < 8; f++) {
          uint32_t ti = i * 8u + (uint32_t)f;
          idx[ti*3+0] = i*6u + OF[f][0]; idx[ti*3+1] = i*6u + OF[f][1]; idx[ti*3+2] = i*6u + OF[f][2];
        }
      }
      out_proxy_mesh->positions = pos; out_proxy_mesh->normals = nrm;
      out_proxy_mesh->uvs = uvs; out_proxy_mesh->colors = col; out_proxy_mesh->indices = idx;
      out_proxy_mesh->n_verts = nv; out_proxy_mesh->n_tris = nt;
      out_proxy_mesh->base_color[0] = out_proxy_mesh->base_color[1] = out_proxy_mesh->base_color[2] = out_proxy_mesh->base_color[3] = 1.0f;
      out_proxy_mesh->metallic = 0.08f; out_proxy_mesh->roughness = 0.45f;
    }
    free(feat_h);
  }

  cudaFree(anc_d); cudaFree(anc_bf); cudaFree(x_d); cudaFree(h_bf);
  cudaFree(qkv_d); cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(Kc_d); cudaFree(Vc_d);
  cudaFree(ao_bf); cudaFree(tmp_d); cudaFree(mlp_d); cudaFree(mlp_bf);
  cudaFree(feat_d); cudaFree(lat_bf); cudaFree(kv_tmp);
  return 0;
fail:
  cudaFree(anc_d); cudaFree(anc_bf); cudaFree(x_d); cudaFree(h_bf);
  cudaFree(qkv_d); cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(Kc_d); cudaFree(Vc_d);
  cudaFree(ao_bf); cudaFree(tmp_d); cudaFree(mlp_d); cudaFree(mlp_bf);
  cudaFree(feat_d); cudaFree(lat_bf); cudaFree(kv_tmp);
  return -1;
}

// ===========================================================================
// 11. Public API
// ===========================================================================
extern "C" ec_idol_t *ec_idol_create(const ec_idol_paths_t *paths) {
  if (!paths || !paths->dinov3_path[0] || !paths->denoiser_path[0] ||
      !paths->decoder_path[0] || !paths->decoder_gs_path[0]) return NULL;
  ec_idol_t *id = (ec_idol_t *)calloc(1, sizeof *id);
  if (!id) return NULL;
  id->paths = *paths;
  if (cudaStreamCreate(&id->stream) != cudaSuccess) { free(id); return NULL; }
  char err[128] = {0};
  if (ec_st_open(&id->dino_st, paths->dinov3_path,     err, sizeof err) != 0) { ec_idol_destroy(id); return NULL; } id->dino_open = 1;
  if (ec_st_open(&id->den_st,  paths->denoiser_path,   err, sizeof err) != 0) { ec_idol_destroy(id); return NULL; } id->den_open = 1;
  if (ec_st_open(&id->oct_st,  paths->decoder_path,    err, sizeof err) != 0) { ec_idol_destroy(id); return NULL; } id->oct_open = 1;
  if (ec_st_open(&id->gs_st,   paths->decoder_gs_path, err, sizeof err) != 0) { ec_idol_destroy(id); return NULL; } id->gs_open = 1;
  fprintf(stderr, "[idol] v18 speridlabs/idol (Single Image -> 1M 3D Gaussian Splats, Apache-2.0) READY!\n");
  return id;
}

extern "C" void ec_idol_unload_vram(ec_idol_t *id) {
  if (!id) return;
  free_denoiser(&id->den);
  free_oct(&id->oct);
  free_gs(&id->gs);
}

extern "C" void ec_idol_destroy(ec_idol_t *id) {
  if (!id) return;
  ec_idol_unload_vram(id);
  if (id->dino_open) ec_st_close(&id->dino_st);
  if (id->den_open)  ec_st_close(&id->den_st);
  if (id->oct_open)  ec_st_close(&id->oct_st);
  if (id->gs_open)   ec_st_close(&id->gs_st);
  if (id->stream)    cudaStreamDestroy(id->stream);
  free(id);
}

extern "C" int ec_idol_is_ready(const ec_idol_t *id) {
  return (id && id->dino_open && id->den_open && id->oct_open && id->gs_open) ? 1 : 0;
}

extern "C" int ec_idol_generate_splats(
    ec_idol_t *id,
    ec_flux2_t *flux2_vae_host,
    const ec_idol_req_t *req,
    const char *out_ply_path,
    const char *out_glb_preview_path,
    ec_idol_progress_fn cb, void *cb_ctx,
    ec_idol_res_t *res) {
  memset(res, 0, sizeof *res);
  if (!ec_idol_is_ready(id) || !flux2_vae_host || !req || !req->rgb || !out_ply_path) {
    snprintf(res->error, sizeof res->error, "invalid idol request or missing FLUX.2 VAE host");
    return -1;
  }
  double t_all = id_now_ms();
  int steps = (req->steps >= 4 && req->steps <= 80) ? req->steps : 32;
  float cfg = (req->cfg_scale >= 0.0f) ? req->cfg_scale : 3.0f;
  float rescale_t = (req->rescale_t > 0.1f) ? req->rescale_t : 3.0f;
  int n_gs = (req->num_gaussians >= 32768 && req->num_gaussians <= 1048576) ? (req->num_gaussians & ~31) : 262144;
  int n_anc = n_gs / 32;

  if (cb) cb(0, steps, "idol_preprocess", cb_ctx);
  uint8_t *img1024 = idol_preprocess_1024(req->rgb, req->alpha, req->img_w, req->img_h,
                                          req->crop_margin > 0.5f ? req->crop_margin : 1.2f, req->skip_matte);
  if (!img1024) { snprintf(res->error, sizeof res->error, "preprocess failed"); return -1; }

  float *feat1_d = NULL, *feat2_d = NULL, *latent_d = NULL;
  if (cudaMalloc(&feat1_d, (size_t)4101 * 1280 * 4) != cudaSuccess ||
      cudaMalloc(&feat2_d, (size_t)4101 * 128 * 4) != cudaSuccess ||
      cudaMalloc(&latent_d, (size_t)8192 * 16 * 4) != cudaSuccess) {
    free(img1024); cudaFree(feat1_d); cudaFree(feat2_d); cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, "OOM allocating idol conditioning");
    return -1;
  }

  double dino_ms = 0.0, vae_ms = 0.0;
  if (cb) cb(0, steps, "idol_encode", cb_ctx);
  if (run_dinov3_vith16plus(id, img1024, feat1_d, &dino_ms) != 0) {
    free(img1024); cudaFree(feat1_d); cudaFree(feat2_d); cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, "DINOv3-ViT-H+ failed");
    return -1;
  }
  cudaMemsetAsync(feat2_d, 0, 5 * 128 * 4, id->stream);
  cudaStreamSynchronize(id->stream);
  if (ec_flux2_vae_encode_packed_d(flux2_vae_host, img1024, 1024, 1024, feat2_d + 5 * 128, &vae_ms) != 0) {
    free(img1024); cudaFree(feat1_d); cudaFree(feat2_d); cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, "FLUX.2 VAE encode failed");
    return -1;
  }
  // Immediately release FLUX.2 VAE cache so Idol Denoiser + Gaussian Decoder have maximum VRAM headroom!
  ec_flux2_unload_vram(flux2_vae_host);
  free(img1024);
  res->enc_ms = dino_ms + vae_ms;
  fprintf(stderr, "[idol] Encoders done: DINOv3-ViT-H+=%.1fms FLUX.2-VAE=%.1fms\n", dino_ms, vae_ms);

  if (ensure_idol_weights_loaded(id) != 0) {
    cudaFree(feat1_d); cudaFree(feat2_d); cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, "failed loading idol denoiser/decoders");
    return -1;
  }

  float cam_raw[5] = {0};
  int rc_f = run_idol_flow(id, feat1_d, feat2_d, req->seed, steps, cfg, rescale_t, cb, cb_ctx, latent_d, cam_raw, &res->flow_ms);
  cudaFree(feat1_d); cudaFree(feat2_d);
  if (rc_f != 0) {
    cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, rc_f == -2 ? "cancelled" : "idol flow failed");
    return rc_f;
  }

  // Decode predicted camera (exact camera.py:7-18)
  float dn = sqrtf(cam_raw[0]*cam_raw[0] + cam_raw[1]*cam_raw[1] + cam_raw[2]*cam_raw[2]);
  float dx = dn > 1e-6f ? cam_raw[0]/dn : 0.0f, dy = dn > 1e-6f ? cam_raw[1]/dn : 0.0f, dz = dn > 1e-6f ? cam_raw[2]/dn : 1.0f;
  if (dz < -1.0f) dz = -1.0f; if (dz > 1.0f) dz = 1.0f;
  res->cam_yaw_rad   = atan2f(dx, dy);
  res->cam_pitch_rad = asinf(dz);
  res->cam_radius    = (cam_raw[3] < 1e-3f) ? 1000.0f : (1.0f / cam_raw[3]);
  res->cam_fov_deg   = (cam_raw[3] < 1e-3f)
                     ? (float)(2.0 * atan((double)cam_raw[4] / 1000.0) * 57.2957795131)
                     : (float)(2.0 * atan((double)cam_raw[4] * (double)cam_raw[3]) * 57.2957795131);

  double t_dec = id_now_ms();
  if (cb) cb(steps, steps, "idol_decode_splats", cb_ctx);
  float *anchors_h = NULL;
  if (run_idol_octree_sample(id, latent_d, n_anc, req->seed, &anchors_h) != 0) {
    cudaFree(latent_d);
    snprintf(res->error, sizeof res->error, "idol octree sample failed");
    return -1;
  }

  ec_splat3d_t splat = {};
  ec_mesh_t proxy = {};
  int rc_d = run_idol_gs_decode(id, latent_d, anchors_h, n_anc, &splat, out_glb_preview_path ? &proxy : NULL);
  free(anchors_h);
  cudaFree(latent_d);
  if (rc_d != 0) {
    snprintf(res->error, sizeof res->error, "idol gaussian decode failed");
    return -1;
  }

  int wrc = ec_export_splat_ply(out_ply_path, &splat);
  if (wrc == 0 && out_glb_preview_path && out_glb_preview_path[0]) {
    ec_export_glb(out_glb_preview_path, &proxy);
  }
  ec_splat_free(&splat);
  ec_mesh_free(&proxy);
  if (wrc != 0) {
    snprintf(res->error, sizeof res->error, "failed writing 3DGS PLY");
    return -1;
  }

  res->dec_ms    = id_now_ms() - t_dec;
  res->total_ms  = id_now_ms() - t_all;
  res->n_splats  = n_gs;
  res->n_anchors = n_anc;
  res->steps_run = steps;
  res->ok = 1;
  fprintf(stderr, "[idol] Generated %d 3D Gaussians (%d anchors) in %.1fms (enc=%.1fms flow=%.1fms dec=%.1fms, fov=%.1fdeg)\n",
          n_gs, n_anc, res->total_ms, res->enc_ms, res->flow_ms, res->dec_ms, res->cam_fov_deg);
  return 0;
}
