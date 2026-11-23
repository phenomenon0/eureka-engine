// ec_acestep.cu — 100% Pure Native Eureka ACE-Step 1.5 Turbo Music Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Pipeline:
//   1. Native CUDA Qwen3-0.6B Text Encoder (28L GQA, dim=1024 -> caption hidden + lyric embeds)
//   2. Native CUDA AceStepConditionEncoder (text_proj + 8L lyric_encoder + 4L timbre_encoder)
//   3. Native CUDA 24-Layer AceStepDiTModel (dim=2048, 16Q/8KV, 8-step Shift=3.0 Turbo ODE,
//      alternating sliding-window W=128 / full self-attn, cached cross-attn K/V)
//   4. Native CUDA AutoencoderOobleck 1D Stereo VAE (weight-norm Conv1d/ConvTranspose1d +
//      Snake1d + 5×3 dilated ResUnits, 1920x upsampling -> 48 kHz stereo PCM WAV + 4 stems)

#define _POSIX_C_SOURCE 200809L
#include "ec_acestep.h"
#include "ec_bpe.h"
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

static double as_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static __device__ __forceinline__ uint16_t as_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static inline float as_bf16_to_f32_host(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline uint16_t as_f32_to_bf16_host(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float as_silu(float x) {
  return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float as_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// ===========================================================================
// 1. Unified High-Throughput BF16 TensorCore GEMM (via shared ec_ops.cu)
// ===========================================================================
static void as_gemm(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t s) {
  ec_ops_gemm_bf16(X, W, bias, Y_f32, Y_bf16, M, N, K, s);
}

// ===========================================================================
// 2. Transformer & Conditioning Kernels
// ===========================================================================
__global__ void k_as_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = as_f32_to_bf16(src[i]);
}
__global__ void k_as_add_inplace(float *dst, const float *src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
__global__ void k_as_add_gated(float *dst, const float *src, const float *gate, int rows, int dim) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * dim;
  if (idx >= tot) return;
  int c = (int)(idx % dim);
  dst[idx] += src[idx] * gate[c];
}
__global__ void k_as_swiglu_pair_bf16(const float *gate, const float *up, uint16_t *out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = as_f32_to_bf16(as_silu(gate[i]) * up[i]);
}

// Qwen3RMSNorm -> optional AdaLN: (rmsnorm(x, w) * (1 + scale) + shift) -> BF16
__global__ void k_as_rmsnorm_adaln(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ scale, const float *__restrict__ shift,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int dim, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; ss += v * v; }
  __shared__ float sm[32];
  ss = as_warp_sum(ss);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    t = as_warp_sum(t);
    if (threadIdx.x == 0) sm[0] = rsqrtf(t / (float)dim + eps);
  }
  __syncthreads();
  float inv = sm[0];
  float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
  uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float v = xr[c] * inv * (w ? w[c] : 1.0f);
    if (scale) v = v * (1.0f + scale[c]) + shift[c];
    if (yf) yf[c] = v;
    if (yb) yb[c] = as_f32_to_bf16(v);
  }
}

// QK-RMSNorm + Split-Half 1D RoPE + Transpose [L, H, D] -> [H, L, D] BF16 (direct TensorCore FA2 input!)
__global__ void k_as_qk_norm_rope_bf16(
    const float *__restrict__ q_in, const float *__restrict__ k_in, const float *__restrict__ v_in,
    const float *__restrict__ q_norm_w, const float *__restrict__ k_norm_w,
    uint16_t *__restrict__ q_out, uint16_t *__restrict__ k_out, uint16_t *__restrict__ v_out,
    int Lq, int Lkv, int Hq, int Hkv, int D, float theta, float eps, int apply_rope) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  int total_q = q_in ? (Lq * Hq) : 0;
  int total_kv = k_in ? (Lkv * Hkv) : 0;
  int half = D >> 1;

  if (idx < total_q) {
    int pos = idx / Hq, h = idx % Hq;
    const float *qs = q_in + ((long)pos * Hq + h) * D;
    uint16_t *qo = q_out + ((long)h * Lq + pos) * D;
    float sq = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = qs[d]; sq += v * v; }
    sq = as_warp_sum(sq);
    float inv = rsqrtf(sq / (float)D + eps);
    for (int j = lane; j < half; j += 32) {
      float x0 = qs[j] * inv * q_norm_w[j];
      float x1 = qs[j + half] * inv * q_norm_w[j + half];
      if (apply_rope) {
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        qo[j]        = as_f32_to_bf16(x0 * cs - x1 * sn);
        qo[j + half] = as_f32_to_bf16(x1 * cs + x0 * sn);
      } else {
        qo[j]        = as_f32_to_bf16(x0);
        qo[j + half] = as_f32_to_bf16(x1);
      }
    }
  } else if (idx < total_q + total_kv) {
    int kidx = idx - total_q;
    int pos = kidx / Hkv, h = kidx % Hkv;
    const float *ks = k_in + ((long)pos * Hkv + h) * D;
    const float *vs = v_in + ((long)pos * Hkv + h) * D;
    uint16_t *ko = k_out + ((long)h * Lkv + pos) * D;
    uint16_t *vo = v_out + ((long)h * Lkv + pos) * D;
    float sk = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = ks[d]; sk += v * v; }
    sk = as_warp_sum(sk);
    float inv = rsqrtf(sk / (float)D + eps);
    for (int j = lane; j < half; j += 32) {
      float x0 = ks[j] * inv * k_norm_w[j];
      float x1 = ks[j + half] * inv * k_norm_w[j + half];
      if (apply_rope) {
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        ko[j]        = as_f32_to_bf16(x0 * cs - x1 * sn);
        ko[j + half] = as_f32_to_bf16(x1 * cs + x0 * sn);
      } else {
        ko[j]        = as_f32_to_bf16(x0);
        ko[j + half] = as_f32_to_bf16(x1);
      }
      vo[j]        = as_f32_to_bf16(vs[j]);
      vo[j + half] = as_f32_to_bf16(vs[j + half]);
    }
  }
}

// QK-RMSNorm + Split-Half 1D RoPE + Transpose [L, H, D] -> [H, L, D]
// apply_rope=0 skips RoPE (for cross-attention K/V & Q).
__global__ void k_as_qk_norm_rope(
    const float *__restrict__ q_in, const float *__restrict__ k_in, const float *__restrict__ v_in,
    const float *__restrict__ q_norm_w, const float *__restrict__ k_norm_w,
    float *__restrict__ q_out, float *__restrict__ k_out, float *__restrict__ v_out,
    int Lq, int Lkv, int Hq, int Hkv, int D, float theta, float eps, int apply_rope) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  int total_q = q_in ? (Lq * Hq) : 0;
  int total_kv = k_in ? (Lkv * Hkv) : 0;
  int half = D >> 1;

  if (idx < total_q) {
    int pos = idx / Hq, h = idx % Hq;
    const float *qs = q_in + ((long)pos * Hq + h) * D;
    float *qo = q_out + ((long)h * Lq + pos) * D;
    float sq = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = qs[d]; sq += v * v; }
    sq = as_warp_sum(sq);
    float inv = rsqrtf(sq / (float)D + eps);
    for (int j = lane; j < half; j += 32) {
      float x0 = qs[j] * inv * q_norm_w[j];
      float x1 = qs[j + half] * inv * q_norm_w[j + half];
      if (apply_rope) {
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        qo[j]        = x0 * cs - x1 * sn;
        qo[j + half] = x1 * cs + x0 * sn;
      } else {
        qo[j]        = x0;
        qo[j + half] = x1;
      }
    }
  } else if (idx < total_q + total_kv) {
    int kidx = idx - total_q;
    int pos = kidx / Hkv, h = kidx % Hkv;
    const float *ks = k_in + ((long)pos * Hkv + h) * D;
    const float *vs = v_in + ((long)pos * Hkv + h) * D;
    float *ko = k_out + ((long)h * Lkv + pos) * D;
    float *vo = v_out + ((long)h * Lkv + pos) * D;
    float sk = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = ks[d]; sk += v * v; }
    sk = as_warp_sum(sk);
    float inv = rsqrtf(sk / (float)D + eps);
    for (int j = lane; j < half; j += 32) {
      float x0 = ks[j] * inv * k_norm_w[j];
      float x1 = ks[j + half] * inv * k_norm_w[j + half];
      if (apply_rope) {
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        ko[j]        = x0 * cs - x1 * sn;
        ko[j + half] = x1 * cs + x0 * sn;
      } else {
        ko[j]        = x0;
        ko[j + half] = x1;
      }
      vo[j]        = vs[j];
      vo[j + half] = vs[j + half];
    }
  }
}

// Unified F32 Flash-style GQA Attention (D=128, 1 warp per (qpos, hq)):
//   mode=0: causal (k <= q)
//   mode=1: bidirectional full (0 <= k < Lkv)
//   mode=2: bidirectional sliding window (|q - k| <= window)
__global__ void k_as_gqa_attn(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    uint16_t *__restrict__ O_bf16,
    int Lq, int Lkv, int Hq, int Hkv, int D, float scale, int mode, int window) {
  int pair = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (pair >= Lq * Hq) return;
  int qpos = pair / Hq;
  int hq = pair % Hq;
  int hkv = hq / (Hq / Hkv);

  const float *qvec = Q + ((long)hq * Lq + qpos) * D;
  const float *kbase = K + (long)hkv * Lkv * D;
  const float *vbase = V + (long)hkv * Lkv * D;

  float qr[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) qr[i] = qvec[lane + i * 32];

  int k_start = 0, k_end = Lkv - 1;
  if (mode == 0) {
    k_end = qpos < Lkv ? qpos : (Lkv - 1);
  } else if (mode == 2) {
    k_start = qpos - window;
    if (k_start < 0) k_start = 0;
    k_end = qpos + window;
    if (k_end >= Lkv) k_end = Lkv - 1;
  }

  float m_max = -1e30f;
  float l_sum = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kpos = k_start; kpos <= k_end; kpos++) {
    const float *kv = kbase + (long)kpos * D;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) dot += qr[i] * kv[lane + i * 32];
    dot = as_warp_sum(dot) * scale;

    float new_max = fmaxf(m_max, dot);
    float alpha = expf(m_max - new_max);
    float p = expf(dot - new_max);
    l_sum = l_sum * alpha + p;
    m_max = new_max;
    const float *vv = vbase + (long)kpos * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = acc[i] * alpha + p * vv[lane + i * 32];
  }

  float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  uint16_t *out = O_bf16 + ((long)qpos * Hq + hq) * D;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    out[lane + i * 32] = as_f32_to_bf16(acc[i] * inv_l);
  }
}

// Pack [ctx[T, 128], xt[T, 64]] -> patchified BF16 [T_p, 384] (zero-padded if T is odd)
__global__ void k_as_pack_patch_in(
    const float *__restrict__ ctx, const float *__restrict__ xt,
    uint16_t *__restrict__ out_bf16, int T, int T_p) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)T_p * 384;
  if (idx >= tot) return;
  int p = (int)(idx / 384);
  int col = (int)(idx % 384);
  int sub = col / 192;   // 0 or 1
  int c = col % 192;
  int t = p * 2 + sub;
  float v = 0.0f;
  if (t < T) {
    v = (c < 128) ? ctx[(long)t * 128 + c] : xt[(long)t * 64 + (c - 128)];
  }
  out_bf16[idx] = as_f32_to_bf16(v);
}

// ODE step update: xt[i] -= vt[i] * dt
__global__ void k_as_ode_step(float *xt, const float *vt, float dt, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) xt[i] -= vt[i] * dt;
}

// ===========================================================================
// 3. AutoencoderOobleck 1D Stereo VAE Kernels (Time-Major [L, C] Layout)
// ===========================================================================
// Snake1d(logscale=True): y = x + (1 / (exp(beta) + 1e-9)) * sin(exp(alpha) * x)^2
__global__ void k_as_snake1d(
    const float *__restrict__ x, const float *__restrict__ exp_alpha,
    const float *__restrict__ inv_beta, float *__restrict__ y,
    int L, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)L * C;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  float v = x[idx];
  float s = sinf(exp_alpha[c] * v);
  y[idx] = v + inv_beta[c] * s * s;
}

// Im2Col for dilated 1D Conv (k=7, pad=3*dil) on chunk [t0 .. t0+chunk-1] of x[L, Cin]
// Output: col_bf16[chunk, 7 * Cin] ordered as [k=0..6][ic=0..Cin-1]
__global__ void k_as_im2col_1d_k7(
    const float *__restrict__ x, uint16_t *__restrict__ col_bf16,
    int L, int Cin, int dil, int t0, int chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)chunk * 7 * Cin;
  if (idx >= tot) return;
  int row = (int)(idx / (7 * Cin));
  int rem = (int)(idx % (7 * Cin));
  int k = rem / Cin;
  int ic = rem % Cin;
  int t = t0 + row + (k - 3) * dil;
  float v = (t >= 0 && t < L) ? x[(long)t * Cin + ic] : 0.0f;
  col_bf16[idx] = as_f32_to_bf16(v);
}

// Build input pairs for ConvTranspose1d (k=2*s, stride=s, pad=s/2) on chunk [q0 .. q0+q_chunk-1]
// Row r = cat(x[q], x[q-1]) of width 2*Cin, zero outside [0, L_in-1].
__global__ void k_as_convt_pairs(
    const float *__restrict__ x, uint16_t *__restrict__ pairs_bf16,
    int L_in, int Cin, int q0, int q_chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)q_chunk * (2 * Cin);
  if (idx >= tot) return;
  int row = (int)(idx / (2 * Cin));
  int col = (int)(idx % (2 * Cin));
  int q = q0 + row;
  int t_in = (col < Cin) ? q : (q - 1);
  int ic = (col < Cin) ? col : (col - Cin);
  float v = (t_in >= 0 && t_in < L_in) ? x[(long)t_in * Cin + ic] : 0.0f;
  pairs_bf16[idx] = as_f32_to_bf16(v);
}

// ===========================================================================
// 4. Safetensors Weight Loading Helpers
// ===========================================================================
static uint16_t *st_upload_bf16(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) { fprintf(stderr, "[acestep] missing tensor: %s\n", name); return NULL; }
  size_t nelem = t->nbytes / (t->dtype == EC_DTYPE_F32 ? 4 : 2);
  uint16_t *d = NULL;
  if (cudaMalloc(&d, nelem * 2) != cudaSuccess) return NULL;
  if (t->dtype == EC_DTYPE_BF16) {
    cudaMemcpy(d, ec_st_ptr(st, t), nelem * 2, cudaMemcpyHostToDevice);
  } else if (t->dtype == EC_DTYPE_F32) {
    const float *src = (const float *)ec_st_ptr(st, t);
    uint16_t *tmp = (uint16_t *)malloc(nelem * 2);
    for (size_t i = 0; i < nelem; i++) tmp[i] = as_f32_to_bf16_host(src[i]);
    cudaMemcpy(d, tmp, nelem * 2, cudaMemcpyHostToDevice);
    free(tmp);
  } else {
    cudaFree(d);
    return NULL;
  }
  return d;
}

static float *st_upload_f32(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) { fprintf(stderr, "[acestep] missing tensor: %s\n", name); return NULL; }
  size_t nelem = t->nbytes / (t->dtype == EC_DTYPE_F32 ? 4 : 2);
  float *tmp = (float *)malloc(nelem * 4);
  if (t->dtype == EC_DTYPE_BF16) {
    const uint16_t *src = (const uint16_t *)ec_st_ptr(st, t);
    for (size_t i = 0; i < nelem; i++) tmp[i] = as_bf16_to_f32_host(src[i]);
  } else {
    memcpy(tmp, ec_st_ptr(st, t), nelem * 4);
  }
  float *d = NULL;
  if (cudaMalloc(&d, nelem * 4) != cudaSuccess) { free(tmp); return NULL; }
  cudaMemcpy(d, tmp, nelem * 4, cudaMemcpyHostToDevice);
  free(tmp);
  return d;
}

static int st_read_host_f32(const ec_safetensors_t *st, const char *name, float **out, size_t *out_n) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) { fprintf(stderr, "[acestep] missing tensor: %s\n", name); return -1; }
  size_t nelem = t->nbytes / (t->dtype == EC_DTYPE_F32 ? 4 : 2);
  float *buf = (float *)malloc(nelem * 4);
  if (!buf) return -1;
  if (t->dtype == EC_DTYPE_BF16) {
    const uint16_t *src = (const uint16_t *)ec_st_ptr(st, t);
    for (size_t i = 0; i < nelem; i++) buf[i] = as_bf16_to_f32_host(src[i]);
  } else {
    memcpy(buf, ec_st_ptr(st, t), nelem * 4);
  }
  *out = buf;
  if (out_n) *out_n = nelem;
  return 0;
}

// ===========================================================================
// 5. Resident Model Structs (ACE-Step DiT + AutoencoderOobleck Decoder)
// ===========================================================================
typedef struct {
  float    *self_norm_w, *cross_norm_w, *mlp_norm_w;
  float    *self_q_norm, *self_k_norm, *cross_q_norm, *cross_k_norm;
  uint16_t *self_q, *self_k, *self_v, *self_o;
  uint16_t *cross_q, *cross_k, *cross_v, *cross_o;
  uint16_t *mlp_gate, *mlp_up, *mlp_down;
  float    *scale_shift; // [6 * 2048]
} as_dit_layer_t;

typedef struct {
  uint16_t *l1_w, *l2_w, *proj_w;
  float    *l1_b, *l2_b, *proj_b;
} as_time_emb_t;

typedef struct {
  int loaded;
  uint16_t *proj_in_w;     // transposed [2048, 384] BF16
  float    *proj_in_b;     // [2048]
  uint16_t *cond_emb_w;    // [2048, 2048]
  float    *cond_emb_b;    // [2048]
  as_time_emb_t time_embed;
  as_time_emb_t time_embed_r;
  as_dit_layer_t layers[24];
  float    *norm_out_w;    // [2048]
  float    *scale_shift_out; // [2 * 2048]
  uint16_t *proj_out_w;    // transposed [128, 2048] BF16
  float    *proj_out_b2;   // [128] (bias[64] duplicated twice)
} as_dit_resident_t;

typedef struct {
  float *exp_alpha; // [C]
  float *inv_beta;  // [C] = 1 / (exp(beta) + 1e-9)
} as_snake_t;

typedef struct {
  as_snake_t s1, s2;
  uint16_t *c1_w; // [C, 7*C] BF16 (weight-norm folded, transposed to [Cout, K, Cin])
  float    *c1_b; // [C]
  uint16_t *c2_w; // [C, C] BF16
  float    *c2_b; // [C]
} as_oobleck_res_t;

typedef struct {
  int in_c, out_c, stride;
  as_snake_t snake1;
  uint16_t *conv_t_w; // [stride * out_c, 2 * in_c] BF16
  float    *conv_t_b; // [stride * out_c] (bias tiled stride times!)
  as_oobleck_res_t res[3]; // dil = 1, 3, 9
} as_oobleck_block_t;

typedef struct {
  int loaded;
  uint16_t *conv1_w; // [2048, 7*64] BF16
  float    *conv1_b; // [2048]
  as_oobleck_block_t blocks[5];
  as_snake_t final_snake;
  uint16_t *conv2_w; // [2, 7*128] BF16 (no bias)
} as_vae_resident_t;

struct ec_acestep {
  ec_acestep_paths_t paths;
  cudaStream_t       stream;
  ec_bpe_t          *bpe;
  ec_safetensors_t       dit_st, vae_st, te_st, sil_st;
  int                dit_open, vae_open, te_open, sil_open;
  float             *silence_T64_h; // host [15000, 64] F32 (transposed from [1, 64, 15000])
  int                silence_frames;
  as_dit_resident_t  dit;
  as_vae_resident_t  vae;
};

// ===========================================================================
// 6. Memory Management (Unload VRAM cleanly for cross-engine sharing!)
// ===========================================================================
static void free_time_emb(as_time_emb_t *te) {
  cudaFree(te->l1_w); cudaFree(te->l2_w); cudaFree(te->proj_w);
  cudaFree(te->l1_b); cudaFree(te->l2_b); cudaFree(te->proj_b);
  memset(te, 0, sizeof *te);
}

static void free_dit_resident(as_dit_resident_t *d) {
  if (!d->loaded) return;
  cudaFree(d->proj_in_w); cudaFree(d->proj_in_b);
  cudaFree(d->cond_emb_w); cudaFree(d->cond_emb_b);
  free_time_emb(&d->time_embed);
  free_time_emb(&d->time_embed_r);
  for (int i = 0; i < 24; i++) {
    as_dit_layer_t *L = &d->layers[i];
    cudaFree(L->self_norm_w); cudaFree(L->cross_norm_w); cudaFree(L->mlp_norm_w);
    cudaFree(L->self_q_norm); cudaFree(L->self_k_norm);
    cudaFree(L->cross_q_norm); cudaFree(L->cross_k_norm);
    cudaFree(L->self_q); cudaFree(L->self_k); cudaFree(L->self_v); cudaFree(L->self_o);
    cudaFree(L->cross_q); cudaFree(L->cross_k); cudaFree(L->cross_v); cudaFree(L->cross_o);
    cudaFree(L->mlp_gate); cudaFree(L->mlp_up); cudaFree(L->mlp_down);
    cudaFree(L->scale_shift);
  }
  cudaFree(d->norm_out_w); cudaFree(d->scale_shift_out);
  cudaFree(d->proj_out_w); cudaFree(d->proj_out_b2);
  memset(d, 0, sizeof *d);
}

static void free_snake(as_snake_t *s) {
  cudaFree(s->exp_alpha); cudaFree(s->inv_beta);
  s->exp_alpha = NULL; s->inv_beta = NULL;
}

static void free_vae_resident(as_vae_resident_t *v) {
  if (!v->loaded) return;
  cudaFree(v->conv1_w); cudaFree(v->conv1_b);
  for (int b = 0; b < 5; b++) {
    free_snake(&v->blocks[b].snake1);
    cudaFree(v->blocks[b].conv_t_w);
    cudaFree(v->blocks[b].conv_t_b);
    for (int r = 0; r < 3; r++) {
      free_snake(&v->blocks[b].res[r].s1);
      free_snake(&v->blocks[b].res[r].s2);
      cudaFree(v->blocks[b].res[r].c1_w); cudaFree(v->blocks[b].res[r].c1_b);
      cudaFree(v->blocks[b].res[r].c2_w); cudaFree(v->blocks[b].res[r].c2_b);
    }
  }
  free_snake(&v->final_snake);
  cudaFree(v->conv2_w);
  memset(v, 0, sizeof *v);
}

// ===========================================================================
// 7. Loaders for DiT & Weight-Norm Folded AutoencoderOobleck VAE
// ===========================================================================
static int load_time_emb(const ec_safetensors_t *st, const char *pfx, as_time_emb_t *te) {
  char k[160];
  snprintf(k, sizeof k, "%s.linear_1.weight", pfx);  te->l1_w   = st_upload_bf16(st, k);
  snprintf(k, sizeof k, "%s.linear_1.bias", pfx);    te->l1_b   = st_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.linear_2.weight", pfx);  te->l2_w   = st_upload_bf16(st, k);
  snprintf(k, sizeof k, "%s.linear_2.bias", pfx);    te->l2_b   = st_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.time_proj.weight", pfx); te->proj_w = st_upload_bf16(st, k);
  snprintf(k, sizeof k, "%s.time_proj.bias", pfx);   te->proj_b = st_upload_f32(st, k);
  return (te->l1_w && te->l1_b && te->l2_w && te->l2_b && te->proj_w && te->proj_b) ? 0 : -1;
}

static int ensure_dit_loaded(ec_acestep_t *as) {
  if (as->dit.loaded) return 0;
  const ec_safetensors_t *st = &as->dit_st;
  as_dit_resident_t *d = &as->dit;

  // proj_in.1.weight [2048, 192, 2] -> transpose to [2048, 384] where [oc, k*192 + ic] = W[oc, ic, k]
  float *w_in_h = NULL;
  if (st_read_host_f32(st, "decoder.proj_in.1.weight", &w_in_h, NULL) != 0) return -1;
  uint16_t *w_in_t = (uint16_t *)malloc(2048L * 384L * 2);
  for (int oc = 0; oc < 2048; oc++) {
    for (int ic = 0; ic < 192; ic++) {
      w_in_t[(long)oc * 384 + ic]       = as_f32_to_bf16_host(w_in_h[((long)oc * 192 + ic) * 2 + 0]);
      w_in_t[(long)oc * 384 + 192 + ic] = as_f32_to_bf16_host(w_in_h[((long)oc * 192 + ic) * 2 + 1]);
    }
  }
  free(w_in_h);
  cudaMalloc(&d->proj_in_w, 2048L * 384L * 2);
  cudaMemcpy(d->proj_in_w, w_in_t, 2048L * 384L * 2, cudaMemcpyHostToDevice);
  free(w_in_t);
  d->proj_in_b = st_upload_f32(st, "decoder.proj_in.1.bias");

  d->cond_emb_w = st_upload_bf16(st, "decoder.condition_embedder.weight");
  d->cond_emb_b = st_upload_f32(st, "decoder.condition_embedder.bias");
  if (load_time_emb(st, "decoder.time_embed", &d->time_embed) != 0) return -1;
  if (load_time_emb(st, "decoder.time_embed_r", &d->time_embed_r) != 0) return -1;

  char k[192];
  for (int i = 0; i < 24; i++) {
    as_dit_layer_t *L = &d->layers[i];
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn_norm.weight", i);    L->self_norm_w  = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn_norm.weight", i);   L->cross_norm_w = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.mlp_norm.weight", i);          L->mlp_norm_w   = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.q_norm.weight", i);  L->self_q_norm  = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.k_norm.weight", i);  L->self_k_norm  = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.q_norm.weight", i); L->cross_q_norm = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.k_norm.weight", i); L->cross_k_norm = st_upload_f32(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.q_proj.weight", i);  L->self_q  = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.k_proj.weight", i);  L->self_k  = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.v_proj.weight", i);  L->self_v  = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.self_attn.o_proj.weight", i);  L->self_o  = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.q_proj.weight", i); L->cross_q = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.k_proj.weight", i); L->cross_k = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.v_proj.weight", i); L->cross_v = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.cross_attn.o_proj.weight", i); L->cross_o = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.mlp.gate_proj.weight", i);     L->mlp_gate = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.mlp.up_proj.weight", i);       L->mlp_up   = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.mlp.down_proj.weight", i);     L->mlp_down = st_upload_bf16(st, k);
    snprintf(k, sizeof k, "decoder.layers.%d.scale_shift_table", i);        L->scale_shift = st_upload_f32(st, k);
    if (!L->self_q || !L->cross_q || !L->mlp_down || !L->scale_shift) {
      free_dit_resident(d);
      return -1;
    }
  }

  d->norm_out_w      = st_upload_f32(st, "decoder.norm_out.weight");
  d->scale_shift_out = st_upload_f32(st, "decoder.scale_shift_table");

  // proj_out.1.weight [2048, 64, 2] -> transpose to [128, 2048] where row (k*64 + oc), col ic = W[ic, oc, k]
  float *w_out_h = NULL, *b_out_h = NULL;
  if (st_read_host_f32(st, "decoder.proj_out.1.weight", &w_out_h, NULL) != 0) return -1;
  if (st_read_host_f32(st, "decoder.proj_out.1.bias", &b_out_h, NULL) != 0) { free(w_out_h); return -1; }
  uint16_t *w_out_t = (uint16_t *)malloc(128L * 2048L * 2);
  for (int k_i = 0; k_i < 2; k_i++) {
    for (int oc = 0; oc < 64; oc++) {
      for (int ic = 0; ic < 2048; ic++) {
        w_out_t[(long)(k_i * 64 + oc) * 2048 + ic] = as_f32_to_bf16_host(w_out_h[((long)ic * 64 + oc) * 2 + k_i]);
      }
    }
  }
  float b_out2[128];
  for (int oc = 0; oc < 64; oc++) { b_out2[oc] = b_out_h[oc]; b_out2[64 + oc] = b_out_h[oc]; }
  free(w_out_h); free(b_out_h);
  cudaMalloc(&d->proj_out_w, 128L * 2048L * 2);
  cudaMemcpy(d->proj_out_w, w_out_t, 128L * 2048L * 2, cudaMemcpyHostToDevice);
  free(w_out_t);
  cudaMalloc(&d->proj_out_b2, 128 * sizeof(float));
  cudaMemcpy(d->proj_out_b2, b_out2, 128 * sizeof(float), cudaMemcpyHostToDevice);

  d->loaded = 1;
  return 0;
}

static int load_snake(const ec_safetensors_t *st, const char *pfx, int C, as_snake_t *s) {
  char ka[192], kb[192];
  snprintf(ka, sizeof ka, "%s.alpha", pfx);
  snprintf(kb, sizeof kb, "%s.beta", pfx);
  float *ah = NULL, *bh = NULL;
  if (st_read_host_f32(st, ka, &ah, NULL) != 0) return -1;
  if (st_read_host_f32(st, kb, &bh, NULL) != 0) { free(ah); return -1; }
  for (int c = 0; c < C; c++) {
    ah[c] = expf(ah[c]);
    bh[c] = 1.0f / (expf(bh[c]) + 1e-9f);
  }
  cudaMalloc(&s->exp_alpha, (size_t)C * 4);
  cudaMalloc(&s->inv_beta,  (size_t)C * 4);
  cudaMemcpy(s->exp_alpha, ah, (size_t)C * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(s->inv_beta,  bh, (size_t)C * 4, cudaMemcpyHostToDevice);
  free(ah); free(bh);
  return 0;
}

// Weight-norm fold Conv1d(Cout, Cin, K) -> upload BF16 transposed to [Cout, K * Cin] (row oc, col k*Cin + ic)
static uint16_t *load_wn_conv1d_bf16(const ec_safetensors_t *st, const char *pfx, int Cout, int Cin, int K) {
  char kg[192], kv[192];
  snprintf(kg, sizeof kg, "%s.weight_g", pfx);
  snprintf(kv, sizeof kv, "%s.weight_v", pfx);
  float *g = NULL, *v = NULL;
  if (st_read_host_f32(st, kg, &g, NULL) != 0) return NULL;
  if (st_read_host_f32(st, kv, &v, NULL) != 0) { free(g); return NULL; }

  uint16_t *w_t = (uint16_t *)malloc((size_t)Cout * K * Cin * 2);
  for (int oc = 0; oc < Cout; oc++) {
    double ss = 0.0;
    const float *vo = v + (long)oc * Cin * K;
    for (int i = 0; i < Cin * K; i++) ss += (double)vo[i] * (double)vo[i];
    double scale = (double)g[oc] / (sqrt(ss) + 1e-12);
    for (int ic = 0; ic < Cin; ic++) {
      for (int k = 0; k < K; k++) {
        float wf = (float)((double)vo[(long)ic * K + k] * scale);
        w_t[(long)oc * (K * Cin) + k * Cin + ic] = as_f32_to_bf16_host(wf);
      }
    }
  }
  free(g); free(v);
  uint16_t *d = NULL;
  cudaMalloc(&d, (size_t)Cout * K * Cin * 2);
  cudaMemcpy(d, w_t, (size_t)Cout * K * Cin * 2, cudaMemcpyHostToDevice);
  free(w_t);
  return d;
}

// Weight-norm fold ConvTranspose1d(Cin, Cout, K=2*s) -> upload BF16 [s * Cout, 2 * Cin]
// For u = q*s + r (r=0..s-1), output y[u, oc] = sum_{ic} (x[q, ic] * W[ic, oc, r] + x[q-1, ic] * W[ic, oc, r+s])
static uint16_t *load_wn_convt1d_bf16(const ec_safetensors_t *st, const char *pfx, int Cin, int Cout, int s) {
  int K = 2 * s;
  char kg[192], kv[192];
  snprintf(kg, sizeof kg, "%s.weight_g", pfx);
  snprintf(kv, sizeof kv, "%s.weight_v", pfx);
  float *g = NULL, *v = NULL;
  if (st_read_host_f32(st, kg, &g, NULL) != 0) return NULL;
  if (st_read_host_f32(st, kv, &v, NULL) != 0) { free(g); return NULL; }

  // Weight-norm dim=0 -> norm over (Cout * K) for each ic in 0..Cin-1
  float *scale_ic = (float *)malloc((size_t)Cin * 4);
  for (int ic = 0; ic < Cin; ic++) {
    double ss = 0.0;
    const float *vi = v + (long)ic * Cout * K;
    for (int i = 0; i < Cout * K; i++) ss += (double)vi[i] * (double)vi[i];
    scale_ic[ic] = (float)((double)g[ic] / (sqrt(ss) + 1e-12));
  }

  uint16_t *w_t = (uint16_t *)malloc((size_t)(s * Cout) * (2 * Cin) * 2);
  for (int r = 0; r < s; r++) {
    for (int oc = 0; oc < Cout; oc++) {
      uint16_t *row = w_t + (long)(r * Cout + oc) * (2 * Cin);
      for (int ic = 0; ic < Cin; ic++) {
        const float *vic = v + ((long)ic * Cout + oc) * K;
        row[ic]       = as_f32_to_bf16_host(vic[r]     * scale_ic[ic]);
        row[Cin + ic] = as_f32_to_bf16_host(vic[r + s] * scale_ic[ic]);
      }
    }
  }
  free(scale_ic); free(g); free(v);
  uint16_t *d = NULL;
  cudaMalloc(&d, (size_t)(s * Cout) * (2 * Cin) * 2);
  cudaMemcpy(d, w_t, (size_t)(s * Cout) * (2 * Cin) * 2, cudaMemcpyHostToDevice);
  free(w_t);
  return d;
}

static int ensure_vae_loaded(ec_acestep_t *as) {
  if (as->vae.loaded) return 0;
  const ec_safetensors_t *st = &as->vae_st;
  as_vae_resident_t *v = &as->vae;

  v->conv1_w = load_wn_conv1d_bf16(st, "decoder.conv1", 2048, 64, 7);
  v->conv1_b = st_upload_f32(st, "decoder.conv1.bias");
  if (!v->conv1_w || !v->conv1_b) return -1;

  const int in_cs[5]  = {2048, 1024, 512, 256, 128};
  const int out_cs[5] = {1024,  512, 256, 128, 128};
  const int strds[5]  = {10, 6, 4, 4, 2};
  char pfx[192], kb[192];

  for (int b = 0; b < 5; b++) {
    as_oobleck_block_t *B = &v->blocks[b];
    B->in_c = in_cs[b]; B->out_c = out_cs[b]; B->stride = strds[b];

    snprintf(pfx, sizeof pfx, "decoder.block.%d.snake1", b);
    if (load_snake(st, pfx, B->in_c, &B->snake1) != 0) { free_vae_resident(v); return -1; }

    snprintf(pfx, sizeof pfx, "decoder.block.%d.conv_t1", b);
    B->conv_t_w = load_wn_convt1d_bf16(st, pfx, B->in_c, B->out_c, B->stride);
    snprintf(kb, sizeof kb, "%s.bias", pfx);
    float *bh = NULL;
    if (!B->conv_t_w || st_read_host_f32(st, kb, &bh, NULL) != 0) { free_vae_resident(v); return -1; }
    float *bt = (float *)malloc((size_t)B->stride * B->out_c * 4);
    for (int r = 0; r < B->stride; r++) memcpy(bt + (long)r * B->out_c, bh, (size_t)B->out_c * 4);
    free(bh);
    cudaMalloc(&B->conv_t_b, (size_t)B->stride * B->out_c * 4);
    cudaMemcpy(B->conv_t_b, bt, (size_t)B->stride * B->out_c * 4, cudaMemcpyHostToDevice);
    free(bt);

    for (int u = 0; u < 3; u++) {
      as_oobleck_res_t *R = &B->res[u];
      int C = B->out_c;
      snprintf(pfx, sizeof pfx, "decoder.block.%d.res_unit%d.snake1", b, u + 1);
      if (load_snake(st, pfx, C, &R->s1) != 0) { free_vae_resident(v); return -1; }
      snprintf(pfx, sizeof pfx, "decoder.block.%d.res_unit%d.conv1", b, u + 1);
      R->c1_w = load_wn_conv1d_bf16(st, pfx, C, C, 7);
      snprintf(kb, sizeof kb, "%s.bias", pfx);
      R->c1_b = st_upload_f32(st, kb);
      snprintf(pfx, sizeof pfx, "decoder.block.%d.res_unit%d.snake2", b, u + 1);
      if (load_snake(st, pfx, C, &R->s2) != 0) { free_vae_resident(v); return -1; }
      snprintf(pfx, sizeof pfx, "decoder.block.%d.res_unit%d.conv2", b, u + 1);
      R->c2_w = load_wn_conv1d_bf16(st, pfx, C, C, 1);
      snprintf(kb, sizeof kb, "%s.bias", pfx);
      R->c2_b = st_upload_f32(st, kb);
      if (!R->c1_w || !R->c1_b || !R->c2_w || !R->c2_b) { free_vae_resident(v); return -1; }
    }
  }

  if (load_snake(st, "decoder.snake1", 128, &v->final_snake) != 0) { free_vae_resident(v); return -1; }
  v->conv2_w = load_wn_conv1d_bf16(st, "decoder.conv2", 2, 128, 7);
  if (!v->conv2_w) { free_vae_resident(v); return -1; }

  v->loaded = 1;
  return 0;
}

// ===========================================================================
// 8. Stage 1: Native CUDA Qwen3-0.6B Text & Lyric Encoder (Streamed 1 Layer/Time)
// ===========================================================================
static int run_qwen3_06b(
    ec_acestep_t *as,
    const int32_t *text_ids, int L_txt,
    const int32_t *lyric_ids, int L_lyr,
    uint16_t *out_text_bf16_d,   // [L_txt, 1024] BF16 on GPU
    uint16_t *out_lyric_bf16_d,  // [L_lyr, 1024] BF16 on GPU
    double *out_ms) {
  double t0 = as_now_ms();
  const ec_safetensors_t *st = &as->te_st;
  const ec_st_entry_t *emb_t = ec_st_find(st, "embed_tokens.weight");
  if (!emb_t) return -1;
  const uint16_t *emb_ptr = (const uint16_t *)ec_st_ptr(st, emb_t);
  const int vocab = (int)emb_t->shape[0];
  const int DIM = 1024, HQ = 16, HKV = 8, D = 128, FFN = 3072;

  // Gather lyric embeddings directly from embed_tokens.weight
  uint16_t *lyr_h = (uint16_t *)malloc((size_t)L_lyr * DIM * 2);
  for (int i = 0; i < L_lyr; i++) {
    int id = lyric_ids[i];
    if (id < 0 || id >= vocab) id = 0;
    memcpy(lyr_h + (long)i * DIM, emb_ptr + (long)id * DIM, DIM * 2);
  }
  cudaMemcpyAsync(out_lyric_bf16_d, lyr_h, (size_t)L_lyr * DIM * 2, cudaMemcpyHostToDevice, as->stream);
  free(lyr_h);

  // Gather text embeddings -> F32 on GPU
  float *txt_h = (float *)malloc((size_t)L_txt * DIM * 4);
  for (int i = 0; i < L_txt; i++) {
    int id = text_ids[i];
    if (id < 0 || id >= vocab) id = 0;
    const uint16_t *row = emb_ptr + (long)id * DIM;
    for (int c = 0; c < DIM; c++) txt_h[(long)i * DIM + c] = as_bf16_to_f32_host(row[c]);
  }

  float *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL;
  float *qr_d = NULL, *kr_d = NULL, *vr_d = NULL, *tmp_d = NULL, *gate_d = NULL, *up_d = NULL;
  uint16_t *xn_bf16 = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL;
  cudaMalloc(&x_d,      (size_t)L_txt * DIM * 4);
  cudaMalloc(&xn_bf16,  (size_t)L_txt * DIM * 2);
  cudaMalloc(&q_d,      (size_t)L_txt * HQ * D * 4);
  cudaMalloc(&k_d,      (size_t)L_txt * HKV * D * 4);
  cudaMalloc(&v_d,      (size_t)L_txt * HKV * D * 4);
  cudaMalloc(&qr_d,     (size_t)L_txt * HQ * D * 4);
  cudaMalloc(&kr_d,     (size_t)L_txt * HKV * D * 4);
  cudaMalloc(&vr_d,     (size_t)L_txt * HKV * D * 4);
  cudaMalloc(&ao_bf16,  (size_t)L_txt * HQ * D * 2);
  cudaMalloc(&tmp_d,    (size_t)L_txt * DIM * 4);
  cudaMalloc(&gate_d,   (size_t)L_txt * FFN * 4);
  cudaMalloc(&up_d,     (size_t)L_txt * FFN * 4);
  cudaMalloc(&mlp_bf16, (size_t)L_txt * FFN * 2);
  cudaMemcpyAsync(x_d, txt_h, (size_t)L_txt * DIM * 4, cudaMemcpyHostToDevice, as->stream);
  free(txt_h);

  // Stream 28 Qwen3-0.6B layers (only ~40 MB temp GPU weight per layer!)
  char key[160];
  for (int l = 0; l < 28; l++) {
    snprintf(key, sizeof key, "layers.%d.input_layernorm.weight", l);          float *ln1 = st_upload_f32(st, key);
    snprintf(key, sizeof key, "layers.%d.post_attention_layernorm.weight", l); float *ln2 = st_upload_f32(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.q_norm.weight", l);         float *qn  = st_upload_f32(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.k_norm.weight", l);         float *kn  = st_upload_f32(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.q_proj.weight", l);         uint16_t *wq = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.k_proj.weight", l);         uint16_t *wk = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.v_proj.weight", l);         uint16_t *wv = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.self_attn.o_proj.weight", l);         uint16_t *wo = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.mlp.gate_proj.weight", l);            uint16_t *wg = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.mlp.up_proj.weight", l);              uint16_t *wu = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "layers.%d.mlp.down_proj.weight", l);            uint16_t *wd = st_upload_bf16(st, key);

    k_as_rmsnorm_adaln<<<L_txt, 256, 0, as->stream>>>(x_d, ln1, NULL, NULL, NULL, xn_bf16, L_txt, DIM, 1e-6f);
    as_gemm(xn_bf16, wq, NULL, q_d, NULL, L_txt, HQ * D,  DIM, as->stream);
    as_gemm(xn_bf16, wk, NULL, k_d, NULL, L_txt, HKV * D, DIM, as->stream);
    as_gemm(xn_bf16, wv, NULL, v_d, NULL, L_txt, HKV * D, DIM, as->stream);

    int n_warps = L_txt * (HQ + HKV);
    k_as_qk_norm_rope<<<(n_warps + 7) / 8, 256, 0, as->stream>>>(
        q_d, k_d, v_d, qn, kn, qr_d, kr_d, vr_d, L_txt, L_txt, HQ, HKV, D, 1e6f, 1e-6f, 1);
    int q_warps = L_txt * HQ;
    k_as_gqa_attn<<<(q_warps + 7) / 8, 256, 0, as->stream>>>(
        qr_d, kr_d, vr_d, ao_bf16, L_txt, L_txt, HQ, HKV, D, 1.0f / sqrtf((float)D), 0, 0);
    as_gemm(ao_bf16, wo, NULL, tmp_d, NULL, L_txt, DIM, HQ * D, as->stream);
    k_as_add_inplace<<<((long)L_txt * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, (long)L_txt * DIM);

    k_as_rmsnorm_adaln<<<L_txt, 256, 0, as->stream>>>(x_d, ln2, NULL, NULL, NULL, xn_bf16, L_txt, DIM, 1e-6f);
    as_gemm(xn_bf16, wg, NULL, gate_d, NULL, L_txt, FFN, DIM, as->stream);
    as_gemm(xn_bf16, wu, NULL, up_d,   NULL, L_txt, FFN, DIM, as->stream);
    k_as_swiglu_pair_bf16<<<((long)L_txt * FFN + 255) / 256, 256, 0, as->stream>>>(
        gate_d, up_d, mlp_bf16, (long)L_txt * FFN);
    as_gemm(mlp_bf16, wd, NULL, tmp_d, NULL, L_txt, DIM, FFN, as->stream);
    k_as_add_inplace<<<((long)L_txt * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, (long)L_txt * DIM);

    cudaStreamSynchronize(as->stream);
    cudaFree(ln1); cudaFree(ln2); cudaFree(qn); cudaFree(kn);
    cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo);
    cudaFree(wg); cudaFree(wu); cudaFree(wd);
  }

  float *norm_w = st_upload_f32(st, "norm.weight");
  k_as_rmsnorm_adaln<<<L_txt, 256, 0, as->stream>>>(x_d, norm_w, NULL, NULL, NULL, out_text_bf16_d, L_txt, DIM, 1e-6f);
  cudaStreamSynchronize(as->stream);

  cudaFree(norm_w);
  cudaFree(x_d); cudaFree(xn_bf16); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(qr_d); cudaFree(kr_d); cudaFree(vr_d); cudaFree(ao_bf16);
  cudaFree(tmp_d); cudaFree(gate_d); cudaFree(up_d); cudaFree(mlp_bf16);
  if (out_ms) *out_ms = as_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 9. Stage 2: Native CUDA AceStepConditionEncoder (Text Proj + 8L Lyric + 4L Timbre)
// ===========================================================================
static int run_encoder_stack(
    ec_acestep_t *as, const char *prefix, int n_layers,
    const uint16_t *in_bf16, int L, int in_dim,
    float *out_f32) {
  const ec_safetensors_t *st = &as->dit_st;
  const int DIM = 2048, HQ = 16, HKV = 8, D = 128, FFN = 6144;
  char key[192];

  snprintf(key, sizeof key, "%s.embed_tokens.weight", prefix); uint16_t *ew = st_upload_bf16(st, key);
  snprintf(key, sizeof key, "%s.embed_tokens.bias", prefix);   float    *eb = st_upload_f32(st, key);
  if (!ew || !eb) { cudaFree(ew); cudaFree(eb); return -1; }

  float *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL;
  float *qr_d = NULL, *kr_d = NULL, *vr_d = NULL, *tmp_d = NULL, *gate_d = NULL, *up_d = NULL;
  uint16_t *xn_bf16 = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL;
  cudaMalloc(&x_d,      (size_t)L * DIM * 4);
  cudaMalloc(&xn_bf16,  (size_t)L * DIM * 2);
  cudaMalloc(&q_d,      (size_t)L * HQ * D * 4);
  cudaMalloc(&k_d,      (size_t)L * HKV * D * 4);
  cudaMalloc(&v_d,      (size_t)L * HKV * D * 4);
  cudaMalloc(&qr_d,     (size_t)L * HQ * D * 4);
  cudaMalloc(&kr_d,     (size_t)L * HKV * D * 4);
  cudaMalloc(&vr_d,     (size_t)L * HKV * D * 4);
  cudaMalloc(&ao_bf16,  (size_t)L * HQ * D * 2);
  cudaMalloc(&tmp_d,    (size_t)L * DIM * 4);
  cudaMalloc(&gate_d,   (size_t)L * FFN * 4);
  cudaMalloc(&up_d,     (size_t)L * FFN * 4);
  cudaMalloc(&mlp_bf16, (size_t)L * FFN * 2);

  as_gemm(in_bf16, ew, eb, x_d, NULL, L, DIM, in_dim, as->stream);
  cudaStreamSynchronize(as->stream);
  cudaFree(ew); cudaFree(eb);

  for (int l = 0; l < n_layers; l++) {
    snprintf(key, sizeof key, "%s.layers.%d.input_layernorm.weight", prefix, l);          float *ln1 = st_upload_f32(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.post_attention_layernorm.weight", prefix, l); float *ln2 = st_upload_f32(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.q_norm.weight", prefix, l);         float *qn  = st_upload_f32(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.k_norm.weight", prefix, l);         float *kn  = st_upload_f32(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.q_proj.weight", prefix, l);         uint16_t *wq = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.k_proj.weight", prefix, l);         uint16_t *wk = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.v_proj.weight", prefix, l);         uint16_t *wv = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.self_attn.o_proj.weight", prefix, l);         uint16_t *wo = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.mlp.gate_proj.weight", prefix, l);            uint16_t *wg = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.mlp.up_proj.weight", prefix, l);              uint16_t *wu = st_upload_bf16(st, key);
    snprintf(key, sizeof key, "%s.layers.%d.mlp.down_proj.weight", prefix, l);            uint16_t *wd = st_upload_bf16(st, key);

    k_as_rmsnorm_adaln<<<L, 256, 0, as->stream>>>(x_d, ln1, NULL, NULL, NULL, xn_bf16, L, DIM, 1e-6f);
    as_gemm(xn_bf16, wq, NULL, q_d, NULL, L, HQ * D,  DIM, as->stream);
    as_gemm(xn_bf16, wk, NULL, k_d, NULL, L, HKV * D, DIM, as->stream);
    as_gemm(xn_bf16, wv, NULL, v_d, NULL, L, HKV * D, DIM, as->stream);

    int n_warps = L * (HQ + HKV);
    k_as_qk_norm_rope<<<(n_warps + 7) / 8, 256, 0, as->stream>>>(
        q_d, k_d, v_d, qn, kn, qr_d, kr_d, vr_d, L, L, HQ, HKV, D, 1e6f, 1e-6f, 1);
    int q_warps = L * HQ;
    int mode = (l % 2 == 0) ? 2 : 1; // even: sliding window 128, odd: full bidirectional
    k_as_gqa_attn<<<(q_warps + 7) / 8, 256, 0, as->stream>>>(
        qr_d, kr_d, vr_d, ao_bf16, L, L, HQ, HKV, D, 1.0f / sqrtf((float)D), mode, 128);
    as_gemm(ao_bf16, wo, NULL, tmp_d, NULL, L, DIM, HQ * D, as->stream);
    k_as_add_inplace<<<((long)L * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, (long)L * DIM);

    k_as_rmsnorm_adaln<<<L, 256, 0, as->stream>>>(x_d, ln2, NULL, NULL, NULL, xn_bf16, L, DIM, 1e-6f);
    as_gemm(xn_bf16, wg, NULL, gate_d, NULL, L, FFN, DIM, as->stream);
    as_gemm(xn_bf16, wu, NULL, up_d,   NULL, L, FFN, DIM, as->stream);
    k_as_swiglu_pair_bf16<<<((long)L * FFN + 255) / 256, 256, 0, as->stream>>>(
        gate_d, up_d, mlp_bf16, (long)L * FFN);
    as_gemm(mlp_bf16, wd, NULL, tmp_d, NULL, L, DIM, FFN, as->stream);
    k_as_add_inplace<<<((long)L * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, (long)L * DIM);

    cudaStreamSynchronize(as->stream);
    cudaFree(ln1); cudaFree(ln2); cudaFree(qn); cudaFree(kn);
    cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo);
    cudaFree(wg); cudaFree(wu); cudaFree(wd);
  }

  snprintf(key, sizeof key, "%s.norm.weight", prefix);
  float *nw = st_upload_f32(st, key);
  k_as_rmsnorm_adaln<<<L, 256, 0, as->stream>>>(x_d, nw, NULL, NULL, out_f32, NULL, L, DIM, 1e-6f);
  cudaStreamSynchronize(as->stream);

  cudaFree(nw);
  cudaFree(x_d); cudaFree(xn_bf16); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(qr_d); cudaFree(kr_d); cudaFree(vr_d); cudaFree(ao_bf16);
  cudaFree(tmp_d); cudaFree(gate_d); cudaFree(up_d); cudaFree(mlp_bf16);
  return 0;
}

static int run_condition_encoder(
    ec_acestep_t *as,
    const uint16_t *text_bf16_d, int L_txt,
    const uint16_t *lyric_bf16_d, int L_lyr,
    uint16_t *out_packed_bf16_d, // [L_lyr + 1 + L_txt, 2048] BF16
    double *out_ms) {
  double t0 = as_now_ms();
  const int DIM = 2048;

  // 1. Lyric encoder (8L) -> [L_lyr, 2048]
  float *lyr_f32 = NULL;
  cudaMalloc(&lyr_f32, (size_t)L_lyr * DIM * 4);
  if (run_encoder_stack(as, "encoder.lyric_encoder", 8, lyric_bf16_d, L_lyr, 1024, lyr_f32) != 0) {
    cudaFree(lyr_f32); return -1;
  }

  // 2. Timbre encoder (4L on silence_latent[0..749, 0..63]) -> take row 0 [1, 2048]
  int T_timbre = as->silence_frames < 750 ? as->silence_frames : 750;
  uint16_t *sil_bf16_h = (uint16_t *)malloc((size_t)T_timbre * 64 * 2);
  for (int i = 0; i < T_timbre * 64; i++) sil_bf16_h[i] = as_f32_to_bf16_host(as->silence_T64_h[i]);
  uint16_t *sil_bf16_d = NULL;
  cudaMalloc(&sil_bf16_d, (size_t)T_timbre * 64 * 2);
  cudaMemcpy(sil_bf16_d, sil_bf16_h, (size_t)T_timbre * 64 * 2, cudaMemcpyHostToDevice);
  free(sil_bf16_h);

  float *timbre_f32 = NULL;
  cudaMalloc(&timbre_f32, (size_t)T_timbre * DIM * 4);
  int rc = run_encoder_stack(as, "encoder.timbre_encoder", 4, sil_bf16_d, T_timbre, 64, timbre_f32);
  cudaFree(sil_bf16_d);
  if (rc != 0) { cudaFree(lyr_f32); cudaFree(timbre_f32); return -1; }

  // 3. Text projector: [L_txt, 1024] -> [L_txt, 2048]
  uint16_t *tp_w = st_upload_bf16(&as->dit_st, "encoder.text_projector.weight");
  float *txt_f32 = NULL;
  cudaMalloc(&txt_f32, (size_t)L_txt * DIM * 4);
  as_gemm(text_bf16_d, tp_w, NULL, txt_f32, NULL, L_txt, DIM, 1024, as->stream);

  // 4. Pack in exact order: [lyric_enc, timbre_enc[0], text_proj] -> BF16
  k_as_f32_to_bf16<<<((long)L_lyr * DIM + 255) / 256, 256, 0, as->stream>>>(
      lyr_f32, out_packed_bf16_d, (long)L_lyr * DIM);
  k_as_f32_to_bf16<<<(DIM + 255) / 256, 256, 0, as->stream>>>(
      timbre_f32, out_packed_bf16_d + (long)L_lyr * DIM, DIM);
  k_as_f32_to_bf16<<<((long)L_txt * DIM + 255) / 256, 256, 0, as->stream>>>(
      txt_f32, out_packed_bf16_d + (long)(L_lyr + 1) * DIM, (long)L_txt * DIM);

  cudaStreamSynchronize(as->stream);
  cudaFree(tp_w); cudaFree(lyr_f32); cudaFree(timbre_f32); cudaFree(txt_f32);
  if (out_ms) *out_ms = as_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 10. Stage 3: Native CUDA 24-Layer AceStepDiTModel (8-Step Shift=3.0 Turbo ODE)
// ===========================================================================
static void compute_time_embed_gpu(
    ec_acestep_t *as, const as_time_emb_t *te, float t_val,
    uint16_t *f_d, float *h_f32, uint16_t *h_bf16,
    float *out_temb_d, float *out_tproj_d) {
  uint16_t freq_bf16_h[256];
  double scaled = (double)t_val * 1000.0;
  for (int i = 0; i < 128; i++) {
    double f = exp(-log(10000.0) * (double)i / 128.0);
    double a = scaled * f;
    freq_bf16_h[i]       = as_f32_to_bf16_host((float)cos(a));
    freq_bf16_h[128 + i] = as_f32_to_bf16_host((float)sin(a));
  }
  cudaMemcpyAsync(f_d, freq_bf16_h, 256 * 2, cudaMemcpyHostToDevice, as->stream);
  as_gemm(f_d, te->l1_w, te->l1_b, h_f32, NULL, 1, 2048, 256, as->stream);
  ec_ops_silu_f32_to_bf16(h_f32, h_bf16, 2048, as->stream);
  as_gemm(h_bf16, te->l2_w, te->l2_b, out_temb_d, NULL, 1, 2048, 2048, as->stream);
  ec_ops_silu_f32_to_bf16(out_temb_d, h_bf16, 2048, as->stream);
  as_gemm(h_bf16, te->proj_w, te->proj_b, out_tproj_d, NULL, 1, 12288, 2048, as->stream);
}

__global__ void k_as_add_vec(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_as_add_out_mod(const float *table2, const float *temb, float *out2, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 2 * dim) out2[i] = table2[i] + temb[i % dim];
}

static uint64_t as_splitmix64(uint64_t *s) {
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static float as_randn(uint64_t *rng) {
  double u1 = ((double)(as_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  double u2 = ((double)(as_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  return (float)(sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2));
}

static int run_acestep_dit(
    ec_acestep_t *as,
    const uint16_t *enc_packed_bf16_d, int L_enc,
    int T, int steps, float shift, uint32_t seed,
    float *out_x0_d, // [T, 64] F32 on GPU
    ec_acestep_progress_fn cb, void *cb_ctx,
    double *out_ms) {
  double t0 = as_now_ms();
  const as_dit_resident_t *d = &as->dit;
  const int DIM = 2048, HQ = 16, HKV = 8, D = 128, FFN = 6144;
  int T_pad = (T + 1) & ~1;
  int T_p = T_pad / 2;

  // Build valid Turbo schedule
  static const float S1[8] = {1.0f, 0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f};
  static const float S2[8] = {1.0f, 0.93333333f, 0.85714286f, 0.76923077f, 0.66666667f, 0.54545454f, 0.4f, 0.22222222f};
  static const float S3[8] = {1.0f, 0.95454545f, 0.9f, 0.83333333f, 0.75f, 0.64285714f, 0.5f, 0.3f};
  const float *base_sched = (shift <= 1.5f) ? S1 : ((shift <= 2.5f) ? S2 : S3);
  int n_steps = (steps >= 1 && steps <= 8) ? steps : 8;
  float sched[8];
  if (n_steps == 8) {
    memcpy(sched, base_sched, sizeof sched);
  } else {
    for (int i = 0; i < n_steps; i++) {
      int idx = (i * 8) / n_steps;
      sched[i] = base_sched[idx];
    }
  }

  // Project encoder_hidden_states ONCE through decoder.condition_embedder
  uint16_t *cond_bf16_d = NULL;
  cudaMalloc(&cond_bf16_d, (size_t)L_enc * DIM * 2);
  as_gemm(enc_packed_bf16_d, d->cond_emb_w, d->cond_emb_b, NULL, cond_bf16_d, L_enc, DIM, DIM, as->stream);

  // Precompute & cache cross-attention K/V in BF16 for all 24 layers (direct TensorCore FA2!)
  uint16_t *ck_cache[24] = {0}, *cv_cache[24] = {0};
  float *ck_tmp = NULL, *cv_tmp = NULL;
  cudaMalloc(&ck_tmp, (size_t)L_enc * HKV * D * 4);
  cudaMalloc(&cv_tmp, (size_t)L_enc * HKV * D * 4);
  for (int l = 0; l < 24; l++) {
    cudaMalloc(&ck_cache[l], (size_t)HKV * L_enc * D * 2);
    cudaMalloc(&cv_cache[l], (size_t)HKV * L_enc * D * 2);
    as_gemm(cond_bf16_d, d->layers[l].cross_k, NULL, ck_tmp, NULL, L_enc, HKV * D, DIM, as->stream);
    as_gemm(cond_bf16_d, d->layers[l].cross_v, NULL, cv_tmp, NULL, L_enc, HKV * D, DIM, as->stream);
    int kv_warps = L_enc * HKV;
    k_as_qk_norm_rope_bf16<<<(kv_warps + 7) / 8, 256, 0, as->stream>>>(
        NULL, ck_tmp, cv_tmp, NULL, d->layers[l].cross_k_norm,
        NULL, ck_cache[l], cv_cache[l], 0, L_enc, HQ, HKV, D, 1e6f, 1e-6f, 0);
  }
  cudaFree(ck_tmp); cudaFree(cv_tmp); cudaFree(cond_bf16_d);

  // Precompute constant time_embed_r(0.0) using pre-allocated zero-sync scratch
  float *temb_r_d = NULL, *tproj_r_d = NULL;
  float *temb_t_d = NULL, *tproj_t_d = NULL;
  float *temb_d = NULL, *tproj_d = NULL, *mod_d = NULL, *out_mod_d = NULL;
  uint16_t *te_f_d = NULL, *te_hb_d = NULL; float *te_hf_d = NULL;
  cudaMalloc(&temb_r_d,  DIM * 4);
  cudaMalloc(&tproj_r_d, 6 * DIM * 4);
  cudaMalloc(&temb_t_d,  DIM * 4);
  cudaMalloc(&tproj_t_d, 6 * DIM * 4);
  cudaMalloc(&temb_d,    DIM * 4);
  cudaMalloc(&tproj_d,   6 * DIM * 4);
  cudaMalloc(&mod_d,     6 * DIM * 4);
  cudaMalloc(&out_mod_d, 2 * DIM * 4);
  cudaMalloc(&te_f_d,    256 * 2);
  cudaMalloc(&te_hf_d,   2048 * 4);
  cudaMalloc(&te_hb_d,   2048 * 2);
  compute_time_embed_gpu(as, &d->time_embed_r, 0.0f, te_f_d, te_hf_d, te_hb_d, temb_r_d, tproj_r_d);

  // Prepare context_latents [T, 128] = cat(silence_latent[0..T-1], ones) and initial noise xt [T, 64]
  float *ctx_h = (float *)malloc((size_t)T * 128 * 4);
  float *xt_h  = (float *)malloc((size_t)T * 64 * 4);
  uint64_t rng = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (int t = 0; t < T; t++) {
    int st_idx = t % as->silence_frames;
    memcpy(ctx_h + (long)t * 128, as->silence_T64_h + (long)st_idx * 64, 64 * 4);
    for (int c = 0; c < 64; c++) ctx_h[(long)t * 128 + 64 + c] = 1.0f;
    for (int c = 0; c < 64; c++) xt_h[(long)t * 64 + c] = as_randn(&rng);
  }
  float *ctx_d = NULL, *xt_d = NULL, *vt_d = NULL;
  cudaMalloc(&ctx_d, (size_t)T * 128 * 4);
  cudaMalloc(&xt_d,  (size_t)T * 64 * 4);
  cudaMalloc(&vt_d,  (size_t)T_pad * 64 * 4);
  cudaMemcpyAsync(ctx_d, ctx_h, (size_t)T * 128 * 4, cudaMemcpyHostToDevice, as->stream);
  cudaMemcpyAsync(xt_d,  xt_h,  (size_t)T * 64 * 4,  cudaMemcpyHostToDevice, as->stream);
  free(ctx_h); free(xt_h);

  // Working buffers for DiT forward [T_p, DIM] (BF16 Q/K/V for TensorCore FA2!)
  uint16_t *patch_in_bf16 = NULL, *xn_bf16 = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL;
  uint16_t *qr_b = NULL, *kr_b = NULL, *vr_b = NULL;
  float *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL;
  float *tmp_d = NULL, *gate_d = NULL, *up_d = NULL;
  cudaMalloc(&patch_in_bf16, (size_t)T_p * 384 * 2);
  cudaMalloc(&x_d,           (size_t)T_p * DIM * 4);
  cudaMalloc(&xn_bf16,       (size_t)T_p * DIM * 2);
  cudaMalloc(&q_d,           (size_t)T_p * HQ * D * 4);
  cudaMalloc(&k_d,           (size_t)T_p * HKV * D * 4);
  cudaMalloc(&v_d,           (size_t)T_p * HKV * D * 4);
  cudaMalloc(&qr_b,          (size_t)T_p * HQ * D * 2);
  cudaMalloc(&kr_b,          (size_t)T_p * HKV * D * 2);
  cudaMalloc(&vr_b,          (size_t)T_p * HKV * D * 2);
  cudaMalloc(&ao_bf16,       (size_t)T_p * HQ * D * 2);
  cudaMalloc(&tmp_d,         (size_t)T_p * DIM * 4);
  cudaMalloc(&gate_d,        (size_t)T_p * FFN * 4);
  cudaMalloc(&up_d,          (size_t)T_p * FFN * 4);
  cudaMalloc(&mlp_bf16,      (size_t)T_p * FFN * 2);

  const float attn_scale = 1.0f / sqrtf((float)D);
  int cancelled = 0;
  for (int step = 0; step < n_steps; step++) {
    if (cb && cb(step + 2, n_steps + 3, "acestep_dit_step", cb_ctx) != 0) {
      cancelled = 1;
      break;
    }
    float t_cur = sched[step];
    compute_time_embed_gpu(as, &d->time_embed, t_cur, te_f_d, te_hf_d, te_hb_d, temb_t_d, tproj_t_d);
    k_as_add_vec<<<(DIM + 255) / 256, 256, 0, as->stream>>>(temb_t_d, temb_r_d, temb_d, DIM);
    k_as_add_vec<<<(6 * DIM + 255) / 256, 256, 0, as->stream>>>(tproj_t_d, tproj_r_d, tproj_d, 6 * DIM);

    // Pack + proj_in
    k_as_pack_patch_in<<<((long)T_p * 384 + 255) / 256, 256, 0, as->stream>>>(
        ctx_d, xt_d, patch_in_bf16, T, T_p);
    as_gemm(patch_in_bf16, d->proj_in_w, d->proj_in_b, x_d, NULL, T_p, DIM, 384, as->stream);

    // 24 DiT Layers
    for (int l = 0; l < 24; l++) {
      const as_dit_layer_t *L = &d->layers[l];
      k_as_add_vec<<<(6 * DIM + 255) / 256, 256, 0, as->stream>>>(L->scale_shift, tproj_d, mod_d, 6 * DIM);
      const float *shift_msa = mod_d + 0 * DIM;
      const float *scale_msa = mod_d + 1 * DIM;
      const float *gate_msa  = mod_d + 2 * DIM;
      const float *c_shift   = mod_d + 3 * DIM;
      const float *c_scale   = mod_d + 4 * DIM;
      const float *c_gate    = mod_d + 5 * DIM;

      // 1. Self-Attention (AdaLN -> GQA w/ RoPE -> TensorCore BF16 FA2 -> gated residual)
      k_as_rmsnorm_adaln<<<T_p, 256, 0, as->stream>>>(
          x_d, L->self_norm_w, scale_msa, shift_msa, NULL, xn_bf16, T_p, DIM, 1e-6f);
      as_gemm(xn_bf16, L->self_q, NULL, q_d, NULL, T_p, HQ * D,  DIM, as->stream);
      as_gemm(xn_bf16, L->self_k, NULL, k_d, NULL, T_p, HKV * D, DIM, as->stream);
      as_gemm(xn_bf16, L->self_v, NULL, v_d, NULL, T_p, HKV * D, DIM, as->stream);
      int n_warps = T_p * (HQ + HKV);
      k_as_qk_norm_rope_bf16<<<(n_warps + 7) / 8, 256, 0, as->stream>>>(
          q_d, k_d, v_d, L->self_q_norm, L->self_k_norm, qr_b, kr_b, vr_b,
          T_p, T_p, HQ, HKV, D, 1e6f, 1e-6f, 1);
      int win = (l % 2 == 0) ? 128 : 0; // even: sliding W=128, odd: full
      ec_ops_fa2_bf16(qr_b, kr_b, vr_b, NULL, ao_bf16, T_p, T_p, T_p, T_p, HQ, HKV, D, attn_scale, win, as->stream);
      as_gemm(ao_bf16, L->self_o, NULL, tmp_d, NULL, T_p, DIM, HQ * D, as->stream);
      k_as_add_gated<<<((long)T_p * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, gate_msa, T_p, DIM);

      // 2. Cross-Attention (RMSNorm -> TensorCore BF16 GQA FA2 against cached K/V)
      k_as_rmsnorm_adaln<<<T_p, 256, 0, as->stream>>>(
          x_d, L->cross_norm_w, NULL, NULL, NULL, xn_bf16, T_p, DIM, 1e-6f);
      as_gemm(xn_bf16, L->cross_q, NULL, q_d, NULL, T_p, HQ * D, DIM, as->stream);
      int q_warps = T_p * HQ;
      k_as_qk_norm_rope_bf16<<<(q_warps + 7) / 8, 256, 0, as->stream>>>(
          q_d, NULL, NULL, L->cross_q_norm, NULL, qr_b, NULL, NULL,
          T_p, 0, HQ, HKV, D, 1e6f, 1e-6f, 0);
      ec_ops_fa2_bf16(qr_b, ck_cache[l], cv_cache[l], NULL, ao_bf16, T_p, L_enc, T_p, L_enc, HQ, HKV, D, attn_scale, 0, as->stream);
      as_gemm(ao_bf16, L->cross_o, NULL, tmp_d, NULL, T_p, DIM, HQ * D, as->stream);
      k_as_add_inplace<<<((long)T_p * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, (long)T_p * DIM);

      // 3. SwiGLU MLP (AdaLN -> gate/up -> down -> gated residual)
      k_as_rmsnorm_adaln<<<T_p, 256, 0, as->stream>>>(
          x_d, L->mlp_norm_w, c_scale, c_shift, NULL, xn_bf16, T_p, DIM, 1e-6f);
      as_gemm(xn_bf16, L->mlp_gate, NULL, gate_d, NULL, T_p, FFN, DIM, as->stream);
      as_gemm(xn_bf16, L->mlp_up,   NULL, up_d,   NULL, T_p, FFN, DIM, as->stream);
      k_as_swiglu_pair_bf16<<<((long)T_p * FFN + 255) / 256, 256, 0, as->stream>>>(
          gate_d, up_d, mlp_bf16, (long)T_p * FFN);
      as_gemm(mlp_bf16, L->mlp_down, NULL, tmp_d, NULL, T_p, DIM, FFN, as->stream);
      k_as_add_gated<<<((long)T_p * DIM + 255) / 256, 256, 0, as->stream>>>(x_d, tmp_d, c_gate, T_p, DIM);
    }

    // Output Head: AdaLN + ConvTranspose1d(2048 -> 64, k=2, s=2) -> vt [T_pad, 64]
    k_as_add_out_mod<<<(2 * DIM + 255) / 256, 256, 0, as->stream>>>(d->scale_shift_out, temb_d, out_mod_d, DIM);
    k_as_rmsnorm_adaln<<<T_p, 256, 0, as->stream>>>(
        x_d, d->norm_out_w, out_mod_d + DIM, out_mod_d, NULL, xn_bf16, T_p, DIM, 1e-6f);
    as_gemm(xn_bf16, d->proj_out_w, d->proj_out_b2, vt_d, NULL, T_p, 128, DIM, as->stream);

    float dt = (step == n_steps - 1) ? t_cur : (t_cur - sched[step + 1]);
    k_as_ode_step<<<((long)T * 64 + 255) / 256, 256, 0, as->stream>>>(xt_d, vt_d, dt, (long)T * 64);
  }

  if (!cancelled) {
    cudaMemcpyAsync(out_x0_d, xt_d, (size_t)T * 64 * 4, cudaMemcpyDeviceToDevice, as->stream);
  }
  cudaStreamSynchronize(as->stream);

  for (int l = 0; l < 24; l++) { cudaFree(ck_cache[l]); cudaFree(cv_cache[l]); }
  cudaFree(temb_r_d); cudaFree(tproj_r_d); cudaFree(temb_t_d); cudaFree(tproj_t_d);
  cudaFree(temb_d); cudaFree(tproj_d); cudaFree(mod_d); cudaFree(out_mod_d);
  cudaFree(te_f_d); cudaFree(te_hf_d); cudaFree(te_hb_d);
  cudaFree(ctx_d); cudaFree(xt_d); cudaFree(vt_d);
  cudaFree(patch_in_bf16); cudaFree(x_d); cudaFree(xn_bf16);
  cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(qr_b); cudaFree(kr_b); cudaFree(vr_b);
  cudaFree(ao_bf16); cudaFree(tmp_d); cudaFree(gate_d); cudaFree(up_d); cudaFree(mlp_bf16);

  if (out_ms) *out_ms = as_now_ms() - t0;
  return cancelled ? -2 : 0;
}

// ===========================================================================
// 11. Stage 4: Native CUDA AutoencoderOobleck 1D Stereo VAE Decoder
// ===========================================================================
static void run_conv1d_k7_chunked(
    ec_acestep_t *as, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int L, int Cin, int Cout, int dil,
    uint16_t *col_scratch, int max_chunk) {
  for (int t0 = 0; t0 < L; t0 += max_chunk) {
    int ch = (L - t0 < max_chunk) ? (L - t0) : max_chunk;
    long n_col = (long)ch * 7 * Cin;
    k_as_im2col_1d_k7<<<(n_col + 255) / 256, 256, 0, as->stream>>>(
        x_d, col_scratch, L, Cin, dil, t0, ch);
    as_gemm(col_scratch, w_bf16, b_f32, y_d + (long)t0 * Cout, NULL, ch, Cout, 7 * Cin, as->stream);
  }
}

static void run_conv1d_k1_chunked(
    ec_acestep_t *as, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int L, int Cin, int Cout,
    uint16_t *bf16_scratch, int max_chunk) {
  for (int t0 = 0; t0 < L; t0 += max_chunk) {
    int ch = (L - t0 < max_chunk) ? (L - t0) : max_chunk;
    long n = (long)ch * Cin;
    k_as_f32_to_bf16<<<(n + 255) / 256, 256, 0, as->stream>>>(x_d + (long)t0 * Cin, bf16_scratch, n);
    as_gemm(bf16_scratch, w_bf16, b_f32, y_d + (long)t0 * Cout, NULL, ch, Cout, Cin, as->stream);
  }
}

static void run_oobleck_resunit(
    ec_acestep_t *as, float *x_d, float *buf_a, float *buf_b,
    const as_oobleck_res_t *R, int L, int C, int dil,
    uint16_t *col_scratch, int max_chunk) {
  long tot = (long)L * C;
  k_as_snake1d<<<(tot + 255) / 256, 256, 0, as->stream>>>(
      x_d, R->s1.exp_alpha, R->s1.inv_beta, buf_a, L, C);
  run_conv1d_k7_chunked(as, buf_a, buf_b, R->c1_w, R->c1_b, L, C, C, dil, col_scratch, max_chunk);
  k_as_snake1d<<<(tot + 255) / 256, 256, 0, as->stream>>>(
      buf_b, R->s2.exp_alpha, R->s2.inv_beta, buf_a, L, C);
  run_conv1d_k1_chunked(as, buf_a, buf_b, R->c2_w, R->c2_b, L, C, C, col_scratch, max_chunk * 4);
  k_as_add_inplace<<<(tot + 255) / 256, 256, 0, as->stream>>>(x_d, buf_b, tot);
}

static int run_oobleck_vae_decode(
    ec_acestep_t *as, const float *x0_d, int T,
    float *out_pcm_lr_h, // host [T * 1920 * 2] interleaved stereo F32
    double *out_ms) {
  double t0 = as_now_ms();
  const as_vae_resident_t *v = &as->vae;

  // Scratch im2col buffer: 64 MB BF16 (handles any chunk cleanly!)
  const size_t COL_CAP_ELEMS = 32L * 1024 * 1024;
  uint16_t *col_scratch = NULL;
  if (cudaMalloc(&col_scratch, COL_CAP_ELEMS * 2) != cudaSuccess) return -1;

  // decoder.conv1: [T, 64] -> [T, 2048]
  float *cur_d = NULL;
  cudaMalloc(&cur_d, (size_t)T * 2048 * 4);
  int ch0 = (int)(COL_CAP_ELEMS / (7 * 64));
  run_conv1d_k7_chunked(as, x0_d, cur_d, v->conv1_w, v->conv1_b, T, 64, 2048, 1, col_scratch, ch0);

  int L = T;
  const int dils[3] = {1, 3, 9};
  for (int b = 0; b < 5; b++) {
    const as_oobleck_block_t *B = &v->blocks[b];
    int L_out = L * B->stride;
    int s = B->stride;
    int Cin = B->in_c, Cout = B->out_c;

    // 1. Snake1d in-place into buf_s
    float *buf_s = NULL;
    cudaMalloc(&buf_s, (size_t)L * Cin * 4);
    k_as_snake1d<<<((long)L * Cin + 255) / 256, 256, 0, as->stream>>>(
        cur_d, B->snake1.exp_alpha, B->snake1.inv_beta, buf_s, L, Cin);
    cudaFree(cur_d);

    // 2. ConvTranspose1d(Cin -> Cout, k=2*s, stride=s, pad=s/2) -> [L_out, Cout]
    // Allocate (L + 1) * s * Cout, compute chunked GEMMs, then slice starting at (s/2) * Cout!
    float *ct_full = NULL;
    cudaMalloc(&ct_full, (size_t)(L + 1) * s * Cout * 4);
    int q_max = (int)(COL_CAP_ELEMS / (2 * Cin));
    if (q_max < 1) q_max = 1;
    for (int q0 = 0; q0 < L + 1; q0 += q_max) {
      int qch = (L + 1 - q0 < q_max) ? (L + 1 - q0) : q_max;
      long n_p = (long)qch * 2 * Cin;
      k_as_convt_pairs<<<(n_p + 255) / 256, 256, 0, as->stream>>>(buf_s, col_scratch, L, Cin, q0, qch);
      as_gemm(col_scratch, B->conv_t_w, B->conv_t_b,
              ct_full + (long)q0 * s * Cout, NULL, qch, s * Cout, 2 * Cin, as->stream);
    }
    cudaFree(buf_s);

    // Copy contiguous valid slice [s/2 .. s/2 + L_out - 1] into next_d [L_out, Cout]
    float *next_d = NULL, *buf_a = NULL, *buf_b = NULL;
    cudaMalloc(&next_d, (size_t)L_out * Cout * 4);
    cudaMemcpyAsync(next_d, ct_full + (long)(s / 2) * Cout, (size_t)L_out * Cout * 4,
                    cudaMemcpyDeviceToDevice, as->stream);
    cudaFree(ct_full);

    // 3. Three dilated Residual Units (dil = 1, 3, 9)
    cudaMalloc(&buf_a, (size_t)L_out * Cout * 4);
    cudaMalloc(&buf_b, (size_t)L_out * Cout * 4);
    int r_chunk = (int)(COL_CAP_ELEMS / (7 * Cout));
    if (r_chunk < 256) r_chunk = 256;
    for (int u = 0; u < 3; u++) {
      run_oobleck_resunit(as, next_d, buf_a, buf_b, &B->res[u], L_out, Cout, dils[u], col_scratch, r_chunk);
    }
    cudaFree(buf_a);
    cudaFree(buf_b);

    cur_d = next_d;
    L = L_out;
  }

  // Final decoder.snake1(128) -> decoder.conv2(128 -> 2, k=7, no bias) -> [L, 2]
  float *sn_d = NULL, *pcm_d = NULL;
  cudaMalloc(&sn_d,  (size_t)L * 128 * 4);
  cudaMalloc(&pcm_d, (size_t)L * 2 * 4);
  k_as_snake1d<<<((long)L * 128 + 255) / 256, 256, 0, as->stream>>>(
      cur_d, v->final_snake.exp_alpha, v->final_snake.inv_beta, sn_d, L, 128);
  cudaFree(cur_d);

  int fin_chunk = (int)(COL_CAP_ELEMS / (7 * 128));
  run_conv1d_k7_chunked(as, sn_d, pcm_d, v->conv2_w, NULL, L, 128, 2, 1, col_scratch, fin_chunk);
  cudaFree(sn_d);
  cudaFree(col_scratch);

  cudaMemcpyAsync(out_pcm_lr_h, pcm_d, (size_t)L * 2 * 4, cudaMemcpyDeviceToHost, as->stream);
  cudaStreamSynchronize(as->stream);
  cudaFree(pcm_d);

  if (out_ms) *out_ms = as_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 12. Complementary 4-Band Stem Separator for Studio Live Mixer
//     (Clean 1-pole crossover cascade: stems sum identically to the AI mix!)
// ===========================================================================
static void export_mixer_stems(
    const float *pcm_lr, int n_samples, int sr, const char *const stem_paths[4]) {
  if (!stem_paths) return;
  int any = 0;
  for (int i = 0; i < 4; i++) if (stem_paths[i] && stem_paths[i][0]) any = 1;
  if (!any) return;

  float *stems[4];
  for (int i = 0; i < 4; i++) stems[i] = (float *)malloc((size_t)n_samples * 2 * 4);

  // One-pole complementary LP splits at 160 Hz (bass), 900 Hz (drums/low-mid), 3200 Hz (harmony/mid), >3200 Hz (lead/air)
  float a1 = 1.0f - expf(-2.0f * 3.14159265f * 160.0f  / (float)sr);
  float a2 = 1.0f - expf(-2.0f * 3.14159265f * 900.0f  / (float)sr);
  float a3 = 1.0f - expf(-2.0f * 3.14159265f * 3200.0f / (float)sr);
  float lp1[2] = {0, 0}, lp2[2] = {0, 0}, lp3[2] = {0, 0};

  for (int n = 0; n < n_samples; n++) {
    for (int ch = 0; ch < 2; ch++) {
      float x = pcm_lr[(long)n * 2 + ch];
      lp1[ch] += a1 * (x - lp1[ch]);
      lp2[ch] += a2 * (x - lp2[ch]);
      lp3[ch] += a3 * (x - lp3[ch]);
      float bass    = lp1[ch];
      float drums   = lp2[ch] - lp1[ch];
      float harmony = lp3[ch] - lp2[ch];
      float lead    = x - lp3[ch];
      // Order matches ec_studio: 0=drums, 1=bass, 2=harmony, 3=lead
      stems[0][(long)n * 2 + ch] = drums;
      stems[1][(long)n * 2 + ch] = bass;
      stems[2][(long)n * 2 + ch] = harmony;
      stems[3][(long)n * 2 + ch] = lead;
    }
  }

  for (int i = 0; i < 4; i++) {
    if (stem_paths[i] && stem_paths[i][0]) {
      ec_audio_t au = {stems[i], (uint32_t)n_samples, (uint32_t)sr, 2};
      ec_export_wav(stem_paths[i], &au);
    }
    free(stems[i]);
  }
}

// ===========================================================================
// 13. Public Lifecycle & End-to-End Generator
// ===========================================================================
extern "C" ec_acestep_t *ec_acestep_create(const ec_acestep_paths_t *paths) {
  if (!paths || !paths->dit_path[0] || !paths->vae_path[0] ||
      !paths->text_encoder_path[0] || !paths->silence_latent_path[0] || !paths->tokenizer_path[0]) {
    return NULL;
  }
  ec_acestep_t *as = (ec_acestep_t *)calloc(1, sizeof *as);
  if (!as) return NULL;
  as->paths = *paths;
  if (cudaStreamCreate(&as->stream) != cudaSuccess) { free(as); return NULL; }

  char err[128] = {0};
  as->bpe = ec_bpe_load(paths->tokenizer_path);
  if (!as->bpe) { ec_acestep_destroy(as); return NULL; }
  if (ec_st_open(&as->dit_st, paths->dit_path, err, sizeof err) != 0) { ec_acestep_destroy(as); return NULL; }
  as->dit_open = 1;
  if (ec_st_open(&as->vae_st, paths->vae_path, err, sizeof err) != 0) { ec_acestep_destroy(as); return NULL; }
  as->vae_open = 1;
  if (ec_st_open(&as->te_st, paths->text_encoder_path, err, sizeof err) != 0) { ec_acestep_destroy(as); return NULL; }
  as->te_open = 1;
  if (ec_st_open(&as->sil_st, paths->silence_latent_path, err, sizeof err) != 0) { ec_acestep_destroy(as); return NULL; }
  as->sil_open = 1;

  // Load & transpose silence_latent [1, 64, 15000] -> [15000, 64] on host
  const ec_st_entry_t *st_sil = ec_st_find(&as->sil_st, "silence_latent");
  if (!st_sil || st_sil->rank < 3) { ec_acestep_destroy(as); return NULL; }
  int C_s = (int)st_sil->shape[1];
  int T_s = (int)st_sil->shape[2];
  if (C_s != 64 || T_s < 750) { ec_acestep_destroy(as); return NULL; }
  as->silence_frames = T_s;
  as->silence_T64_h = (float *)malloc((size_t)T_s * 64 * 4);
  if (st_sil->dtype == EC_DTYPE_F32) {
    const float *src = (const float *)ec_st_ptr(&as->sil_st, st_sil);
    for (int c = 0; c < 64; c++)
      for (int t = 0; t < T_s; t++)
        as->silence_T64_h[(long)t * 64 + c] = src[(long)c * T_s + t];
  } else {
    const uint16_t *src = (const uint16_t *)ec_st_ptr(&as->sil_st, st_sil);
    for (int c = 0; c < 64; c++)
      for (int t = 0; t < T_s; t++)
        as->silence_T64_h[(long)t * 64 + c] = as_bf16_to_f32_host(src[(long)c * T_s + t]);
  }
  return as;
}

extern "C" void ec_acestep_unload_vram(ec_acestep_t *as) {
  if (!as) return;
  free_dit_resident(&as->dit);
  free_vae_resident(&as->vae);
}

extern "C" void ec_acestep_destroy(ec_acestep_t *as) {
  if (!as) return;
  ec_acestep_unload_vram(as);
  if (as->dit_open) ec_st_close(&as->dit_st);
  if (as->vae_open) ec_st_close(&as->vae_st);
  if (as->te_open)  ec_st_close(&as->te_st);
  if (as->sil_open) ec_st_close(&as->sil_st);
  if (as->bpe) ec_bpe_free(as->bpe);
  free(as->silence_T64_h);
  if (as->stream) cudaStreamDestroy(as->stream);
  free(as);
}

extern "C" int ec_acestep_is_ready(const ec_acestep_t *as) {
  return (as && as->bpe && as->dit_open && as->vae_open && as->te_open && as->sil_open && as->silence_T64_h) ? 1 : 0;
}

extern "C" int ec_acestep_generate_wav(
    ec_acestep_t *as,
    const ec_acestep_req_t *req,
    const char *out_wav_path,
    const char *const stem_paths[4],
    float **out_pcm_lr,
    ec_acestep_progress_fn cb, void *cb_ctx,
    ec_acestep_res_t *res) {
  memset(res, 0, sizeof *res);
  if (!ec_acestep_is_ready(as) || !req) {
    snprintf(res->error, sizeof res->error, "acestep engine not initialized");
    return -1;
  }
  double t_all = as_now_ms();

  int bpm = (req->bpm >= 40 && req->bpm <= 240) ? req->bpm : 120;
  float dur = (req->duration_sec >= 2.0f && req->duration_sec <= 60.0f) ? req->duration_sec : 10.0f;
  int steps = (req->steps >= 1 && req->steps <= 8) ? req->steps : 8;
  float shift = (req->shift >= 0.5f && req->shift <= 4.0f) ? req->shift : 3.0f;
  int T = (int)roundf(dur * 25.0f);
  if (T < 25) T = 25;
  int n_samples = T * 1920;

  // Format exact SFT_GEN_PROMPT caption & lyrics strings
  const char *caption = (req->prompt && req->prompt[0]) ? req->prompt : "Warm cinematic instrumental synthwave";
  const char *lyrics  = (req->lyrics && req->lyrics[0]) ? req->lyrics : "[Instrumental]";
  const char *lang    = (req->lyrics && req->lyrics[0]) ? "en" : "unknown";
  char cap_buf[2048], lyr_buf[2048];
  snprintf(cap_buf, sizeof cap_buf,
           "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n"
           "# Caption\n%s\n\n"
           "# Metas\n- bpm: %d\n- timesignature: 4\n- keyscale: N/A\n- duration: %d seconds\n<|endoftext|>\n",
           caption, bpm, (int)roundf(dur));
  snprintf(lyr_buf, sizeof lyr_buf, "# Languages\n%s\n\n# Lyric\n%s<|endoftext|>", lang, lyrics);

  int32_t text_ids[256], lyric_ids[1024];
  int L_txt = ec_bpe_encode(as->bpe, cap_buf, text_ids, 256);
  int L_lyr = ec_bpe_encode(as->bpe, lyr_buf, lyric_ids, 1024);
  if (L_txt <= 0 || L_lyr <= 0) {
    snprintf(res->error, sizeof res->error, "bpe tokenization failed");
    return -1;
  }

  // Stage 1: Qwen3-0.6B Text + Lyric Embeddings
  if (cb) cb(0, steps + 3, "acestep_qwen3_encode", cb_ctx);
  uint16_t *text_bf16_d = NULL, *lyric_bf16_d = NULL;
  cudaMalloc(&text_bf16_d,  (size_t)L_txt * 1024 * 2);
  cudaMalloc(&lyric_bf16_d, (size_t)L_lyr * 1024 * 2);
  if (run_qwen3_06b(as, text_ids, L_txt, lyric_ids, L_lyr, text_bf16_d, lyric_bf16_d, &res->text_enc_ms) != 0) {
    cudaFree(text_bf16_d); cudaFree(lyric_bf16_d);
    snprintf(res->error, sizeof res->error, "qwen3-0.6b text encoder failed");
    return -1;
  }

  // Stage 2: Condition Encoder (Text Proj + 8L Lyric + 4L Timbre)
  if (cb) cb(1, steps + 3, "acestep_cond_encode", cb_ctx);
  int L_enc = L_lyr + 1 + L_txt;
  uint16_t *enc_packed_d = NULL;
  cudaMalloc(&enc_packed_d, (size_t)L_enc * 2048 * 2);
  if (run_condition_encoder(as, text_bf16_d, L_txt, lyric_bf16_d, L_lyr, enc_packed_d, &res->cond_enc_ms) != 0) {
    cudaFree(text_bf16_d); cudaFree(lyric_bf16_d); cudaFree(enc_packed_d);
    snprintf(res->error, sizeof res->error, "condition encoder failed");
    return -1;
  }
  cudaFree(text_bf16_d);
  cudaFree(lyric_bf16_d);

  // Stage 3: Load & Run 24-Layer AceStepDiTModel (8-step Shift=3.0 Turbo ODE)
  if (ensure_dit_loaded(as) != 0) {
    cudaFree(enc_packed_d);
    snprintf(res->error, sizeof res->error, "acestep dit weight load failed");
    return -1;
  }
  float *x0_d = NULL;
  cudaMalloc(&x0_d, (size_t)T * 64 * 4);
  int rc = run_acestep_dit(as, enc_packed_d, L_enc, T, steps, shift, req->seed, x0_d, cb, cb_ctx, &res->dit_ms);
  cudaFree(enc_packed_d);
  if (rc != 0) {
    cudaFree(x0_d);
    snprintf(res->error, sizeof res->error, rc == -2 ? "cancelled" : "acestep dit denoising failed");
    return rc;
  }

  // Stage 4: Load & Run AutoencoderOobleck 1D Stereo Audio VAE -> 48 kHz Stereo PCM
  if (cb) cb(steps + 2, steps + 3, "acestep_oobleck_vae", cb_ctx);
  if (ensure_vae_loaded(as) != 0) {
    cudaFree(x0_d);
    snprintf(res->error, sizeof res->error, "oobleck vae weight load failed");
    return -1;
  }
  float *pcm_lr = (float *)malloc((size_t)n_samples * 2 * 4);
  if (run_oobleck_vae_decode(as, x0_d, T, pcm_lr, &res->vae_ms) != 0) {
    free(pcm_lr); cudaFree(x0_d);
    snprintf(res->error, sizeof res->error, "oobleck vae decode failed");
    return -1;
  }
  cudaFree(x0_d);

  // Peak normalize softly only if peak > 0.99 (preserve genuine studio dynamics!), compute RMS/peak
  float peak = 0.0f;
  double sum_sq = 0.0;
  for (long i = 0; i < (long)n_samples * 2; i++) {
    float v = pcm_lr[i];
    if (!isfinite(v)) v = 0.0f;
    float a = fabsf(v);
    if (a > peak) peak = a;
    sum_sq += (double)v * (double)v;
  }
  if (peak > 0.99f) {
    float gain = 0.98f / peak;
    sum_sq = 0.0;
    for (long i = 0; i < (long)n_samples * 2; i++) {
      pcm_lr[i] *= gain;
      sum_sq += (double)pcm_lr[i] * (double)pcm_lr[i];
    }
    peak = 0.98f;
  }
  double rms = sqrt(sum_sq / (double)((long)n_samples * 2) + 1e-12);

  if (out_wav_path && out_wav_path[0]) {
    ec_audio_t au = {pcm_lr, (uint32_t)n_samples, 48000, 2};
    if (ec_export_wav(out_wav_path, &au) != 0) {
      free(pcm_lr);
      snprintf(res->error, sizeof res->error, "wav export failed");
      return -1;
    }
  }
  export_mixer_stems(pcm_lr, n_samples, 48000, stem_paths);

  if (out_pcm_lr) *out_pcm_lr = pcm_lr;
  else free(pcm_lr);

  res->ok = 1;
  res->sample_rate = 48000;
  res->channels = 2;
  res->num_samples = n_samples;
  res->latent_frames = T;
  res->steps_run = steps;
  res->peak_amp = peak;
  res->rms_db = (float)(20.0 * log10(rms + 1e-12));
  res->total_ms = as_now_ms() - t_all;
  return 0;
}
