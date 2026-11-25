// ec_ltx2.cu — 100% Pure Native CUDA Lightricks LTX-2.3 (22B Distilled FP8) Joint Synchronized A/V Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML libraries.
// Complete 4-Stage Pipeline on RTX 3090 (24 GB VRAM):
//   Stage 1: Gemma-3-12B-IT FP8 (48L, hybrid local/global RoPE, zero-centered RMSNorm)
//            + FeatureExtractorV2 (49-layer per-token RMSNorm, (d,l) interleave, dual aggregate_embed)
//            + 8-Layer Video & Audio Embeddings1DConnector (128 learnable registers, split float64 RoPE, gated attn)
//   Stage 2: 8-Step Distilled Dual-Stream AVTransformer3DModel (48L, cfg=1.0, deterministic Euler)
//            Hybrid 34-block VRAM cache + 14-block PCIe streaming, FP8 E4M3->BF16 LUT dequant + m16n8k16 TensorCores
//   Stage 3: Non-Causal 3D Conv Video Decoder (up_blocks.0..8, PixelNorm, D2S, 4x4 Unpatchify -> RGB24)
//   Stage 4: Causal 2D Audio Mel Decoder + Pure IEEE FP32 BigVGAN v2 AMP1 Stereo Vocoder + 16k->48k BWE -> 48kHz WAV + NVENC MP4!

#define _POSIX_C_SOURCE 200809L
#include "ec_ltx2.h"
#include "ec_bpe.h"
#include "ec_media.h"
#include "ec_nvenc.h"
#include "ec_ops.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// 0. Helpers, Timing, FP8 E4M3FN LUT, Deterministic PRNG
// ===========================================================================

static double ltx_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static inline float ltx_bf16_to_f32_host(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline uint16_t ltx_f32_to_bf16_host(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return (uint16_t)(u >> 16);
}

static inline uint64_t ltx_splitmix64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void ltx_fill_randn(float *dst, size_t n, uint64_t seed) {
  uint64_t st = seed ^ 0x4C545832334156ULL;
  for (size_t i = 0; i < n; i += 2) {
    double u1 = ((double)(ltx_splitmix64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = ((double)(ltx_splitmix64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double th = 2.0 * M_PI * u2;
    dst[i] = (float)(r * cos(th));
    if (i + 1 < n) dst[i + 1] = (float)(r * sin(th));
  }
}

__constant__ float c_fp8_e4m3_lut[256];
static int g_fp8_lut_ready = 0;

static void ltx_init_fp8_lut(void) {
  if (g_fp8_lut_ready) return;
  float lut[256];
  for (int v = 0; v < 256; v++) {
    int s = (v >> 7) & 1;
    int e = (v >> 3) & 15;
    int m = v & 7;
    float val = 0.0f;
    if (e == 0) {
      val = (float)m * (1.0f / 512.0f);
    } else if (e == 15 && m == 7) {
      val = 0.0f; // NaN sentinel in OCP E4M3FN
    } else {
      val = (1.0f + (float)m * 0.125f) * ldexpf(1.0f, e - 7);
    }
    lut[v] = s ? -val : val;
  }
  cudaMemcpyToSymbol(c_fp8_e4m3_lut, lut, sizeof(lut));
  g_fp8_lut_ready = 1;
}

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
  return __uint_as_float((uint32_t)b << 16);
}

__device__ __forceinline__ uint16_t d_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float d_gelu_tanh(float x) {
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x3)));
}

__device__ __forceinline__ float d_silu(float x) {
  return x / (1.0f + expf(-x));
}

// ===========================================================================
// 1. Core Native CUDA Kernels (FP8 Dequant, Norms, RoPE, Attention, AdaLN)
// ===========================================================================

__global__ void k_fp8_to_bf16(const uint8_t *__restrict__ src, float scale, uint16_t *__restrict__ dst, size_t n) {
  size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx + 3 < n) {
    uint32_t b4 = *(const uint32_t *)(src + idx);
    uint16_t o0 = d_f32_to_bf16(c_fp8_e4m3_lut[b4 & 0xFFu] * scale);
    uint16_t o1 = d_f32_to_bf16(c_fp8_e4m3_lut[(b4 >> 8) & 0xFFu] * scale);
    uint16_t o2 = d_f32_to_bf16(c_fp8_e4m3_lut[(b4 >> 16) & 0xFFu] * scale);
    uint16_t o3 = d_f32_to_bf16(c_fp8_e4m3_lut[(b4 >> 24) & 0xFFu] * scale);
    uint32_t p0 = (uint32_t)o0 | ((uint32_t)o1 << 16);
    uint32_t p1 = (uint32_t)o2 | ((uint32_t)o3 << 16);
    *(uint2 *)(dst + idx) = make_uint2(p0, p1);
  } else {
    for (size_t i = idx; i < n && i < idx + 4; i++) {
      dst[i] = d_f32_to_bf16(c_fp8_e4m3_lut[src[i]] * scale);
    }
  }
}

__global__ void k_f32_to_bf16(const float *__restrict__ src, uint16_t *__restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = d_f32_to_bf16(src[i]);
}

__global__ void k_bf16_to_f32(const uint16_t *__restrict__ src, float *__restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = d_bf16_to_f32(src[i]);
}

__global__ void k_f32_add_inplace(float *__restrict__ dst, const float *__restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void k_f32_add_bias_rows(float *__restrict__ y, const float *__restrict__ b, int rows, int cols) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)rows * cols;
  if (idx < tot) y[idx] += b[idx % cols];
}

__global__ void k_f32_euler_step(float *__restrict__ x, const float *__restrict__ v, float dt, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += dt * v[i];
}

// Gemma-3 zero-centered RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1.0f + w)
__global__ void k_gemma3_rmsnorm(
    const float *__restrict__ x, const float *__restrict__ w,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int cols, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (size_t)r * cols;
  __shared__ float sm[256];
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = xr[c];
    sum += v * v;
  }
  sm[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / (float)cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = xr[c] * inv * (1.0f + w[c]);
    if (y_f32)  y_f32[(size_t)r * cols + c] = v;
    if (y_bf16) y_bf16[(size_t)r * cols + c] = d_f32_to_bf16(v);
  }
}

// Gemma-3 split-half RoPE on [K, H*D] (D=256)
__global__ void k_gemma3_rope_inplace(float *__restrict__ x, int K, int H, int D, double theta, double pos_scale) {
  int t = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;
  int half = D >> 1;
  if (t >= K || h >= H || d >= half) return;
  double inv_freq = 1.0 / pow(theta, (double)(2 * d) / (double)D);
  double ang = ((double)t * pos_scale) * inv_freq;
  float c = (float)cos(ang);
  float s = (float)sin(ang);
  float *p = x + ((size_t)t * H + h) * D;
  float x0 = p[d], x1 = p[d + half];
  p[d]        = x0 * c - x1 * s;
  p[d + half] = x1 * c + x0 * s;
}

// Gemma-3 exact causal GQA attention (HQ=16, HKV=8, D=256, K <= 1024, scale = 1/16)
__global__ void k_gemma3_causal_gqa(
    const float *__restrict__ Q, const float *__restrict__ K_, const float *__restrict__ V,
    float *__restrict__ O_f32, uint16_t *__restrict__ O_bf16,
    int S, int HQ, int HKV, int D, float scale) {
  int q_pos = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x; // blockDim.x == D (256)
  if (q_pos >= S || h >= HQ || d >= D) return;
  int hkv = h / (HQ / HKV);
  const float *qp = Q + ((size_t)q_pos * HQ + h) * D;
  float qv = qp[d];
  __shared__ float sm[256];

  float m_max = -1e30f;
  float l_sum = 0.0f;
  float acc = 0.0f;
  for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
    const float *kp = K_ + ((size_t)k_pos * HKV + hkv) * D;
    sm[d] = qv * kp[d];
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (d < s) sm[d] += sm[d + s];
      __syncthreads();
    }
    float score = sm[0] * scale;
    float new_m = fmaxf(m_max, score);
    float alpha = expf(m_max - new_m);
    float p = expf(score - new_m);
    l_sum = l_sum * alpha + p;
    const float *vp = V + ((size_t)k_pos * HKV + hkv) * D;
    acc = acc * alpha + p * vp[d];
    m_max = new_m;
    __syncthreads();
  }
  float out = acc / (l_sum + 1e-12f);
  size_t off = ((size_t)q_pos * HQ + h) * D + d;
  if (O_f32)  O_f32[off] = out;
  if (O_bf16) O_bf16[off] = d_f32_to_bf16(out);
}

__global__ void k_gelu_tanh_mul_bf16(const float *__restrict__ gate, const float *__restrict__ up, uint16_t *__restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = d_f32_to_bf16(d_gelu_tanh(gate[i]) * up[i]);
}

__global__ void k_gelu_tanh_to_bf16(const float *__restrict__ src, uint16_t *__restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = d_f32_to_bf16(d_gelu_tanh(src[i]));
}

// LTX FeatureExtractorV2: per-token per-layer RMSNorm + interleave (d, l) + rescale
// hs_all: [49, K, 3840] F32 -> out_bf16 [K, 188160] BF16 (d*49 + l)
__global__ void k_ltx_fev2_full(
    const float *__restrict__ hs_all, const float *__restrict__ inv_rms,
    uint16_t *__restrict__ out_bf16,
    int K, float rescale) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)K * 188160;
  if (idx >= tot) return;
  int t = (int)(idx / 188160);
  int c = (int)(idx % 188160);
  int d = c / 49;
  int l = c % 49;
  float v = hs_all[((size_t)l * K + t) * 3840 + d] * inv_rms[l * K + t] * rescale;
  out_bf16[idx] = d_f32_to_bf16(v);
}

__global__ void k_scatter_cols_f32(
    const float *__restrict__ src, float *__restrict__ dst,
    int K, int r0, int R_ch, int N_out) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)K * R_ch;
  if (idx >= tot) return;
  int t = (int)(idx / R_ch);
  int c = (int)(idx % R_ch);
  dst[(size_t)t * N_out + r0 + c] = src[idx];
}

__global__ void k_ltx_layer_tok_inv_rms(const float *__restrict__ hs_all, float *__restrict__ inv_rms, int K) {
  int l = blockIdx.y;
  int t = blockIdx.x;
  if (l >= 49 || t >= K) return;
  const float *p = hs_all + ((size_t)l * K + t) * 3840;
  __shared__ float sm[256];
  float s = 0.0f;
  for (int d = threadIdx.x; d < 3840; d += 256) { float v = p[d]; s += v * v; }
  sm[threadIdx.x] = s;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k]; __syncthreads(); }
  if (threadIdx.x == 0) inv_rms[l * K + t] = rsqrtf(sm[0] / 3840.0f + 1e-6f);
}

// General LTX RMSNorm (optional weight, optional AdaLN shift/scale -> F32 and/or BF16)
__global__ void k_ltx_rmsnorm(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ scale, const float *__restrict__ shift,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int cols, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (size_t)r * cols;
  __shared__ float sm[256];
  float s = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) { float v = xr[c]; s += v * v; }
  sm[threadIdx.x] = s;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k]; __syncthreads(); }
  float inv = rsqrtf(sm[0] / (float)cols + eps);
  for (int c = threadIdx.x; c < cols; c += 256) {
    float v = xr[c] * inv;
    if (w) v *= w[c];
    if (scale) v = v * (1.0f + scale[c]) + shift[c];
    if (y_f32)  y_f32[(size_t)r * cols + c] = v;
    if (y_bf16) y_bf16[(size_t)r * cols + c] = d_f32_to_bf16(v);
  }
}

// Affine-free LayerNorm + AdaLN scale/shift (used for LTX final norm_out!)
__global__ void k_ltx_layernorm_adaln(
    const float *__restrict__ x, const float *__restrict__ scale, const float *__restrict__ shift,
    uint16_t *__restrict__ y_bf16, int rows, int cols, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (size_t)r * cols;
  __shared__ float sm_m[256], sm_v[256];
  float sum = 0.0f, sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) { float v = xr[c]; sum += v; sq += v * v; }
  sm_m[threadIdx.x] = sum; sm_v[threadIdx.x] = sq;
  __syncthreads();
  for (int k = 128; k > 0; k >>= 1) {
    if (threadIdx.x < k) { sm_m[threadIdx.x] += sm_m[threadIdx.x + k]; sm_v[threadIdx.x] += sm_v[threadIdx.x + k]; }
    __syncthreads();
  }
  float mean = sm_m[0] / (float)cols;
  float var = fmaxf(0.0f, sm_v[0] / (float)cols - mean * mean);
  float inv = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < cols; c += 256) {
    float v = (xr[c] - mean) * inv;
    v = v * (1.0f + scale[c]) + shift[c];
    y_bf16[(size_t)r * cols + c] = d_f32_to_bf16(v);
  }
}

// Modulate context tensor: dst_bf16[t, c] = ctx[t, c] * (1 + scale[c]) + shift[c]
__global__ void k_ltx_modulate_bf16(
    const float *__restrict__ ctx, const float *__restrict__ scale, const float *__restrict__ shift,
    uint16_t *__restrict__ dst_bf16, int T, int D) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)T * D;
  if (idx >= tot) return;
  int c = (int)(idx % D);
  dst_bf16[idx] = d_f32_to_bf16(ctx[idx] * (1.0f + scale[c]) + shift[c]);
}

// Fused: Full-dim Q/K RMSNorm + LTX Split RoPE + transpose [T, H*D] -> [H, T_pad, D] BF16 for ec_ops_fa2_bf16!
// cos_pe / sin_pe: [H, T, D/2] F32 (may be NULL for cross-attn without RoPE!)
__global__ void k_ltx_qk_norm_rope_pack(
    const float *__restrict__ src, const float *__restrict__ norm_w,
    const float *__restrict__ cos_pe, const float *__restrict__ sin_pe,
    uint16_t *__restrict__ dst_htd,
    int T, int T_pad, int H, int D, float eps) {
  int t = blockIdx.x;
  if (t >= T_pad) return;
  int inner = H * D;
  if (t >= T) {
    for (int i = threadIdx.x; i < inner; i += 256) {
      int h = i / D, d = i % D;
      dst_htd[((size_t)h * T_pad + t) * D + d] = 0;
    }
    return;
  }
  const float *xr = src + (size_t)t * inner;
  __shared__ float sm[256];
  float inv = 1.0f;
  if (norm_w) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < inner; i += 256) { float v = xr[i]; s += v * v; }
    sm[threadIdx.x] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k]; __syncthreads(); }
    inv = rsqrtf(sm[0] / (float)inner + eps);
  }
  int half = D >> 1;
  int n_pairs = H * half;
  for (int p = threadIdx.x; p < n_pairs; p += 256) {
    int h = p / half;
    int d = p % half;
    int i0 = h * D + d;
    int i1 = i0 + half;
    float x0 = xr[i0] * inv * (norm_w ? norm_w[i0] : 1.0f);
    float x1 = xr[i1] * inv * (norm_w ? norm_w[i1] : 1.0f);
    if (cos_pe && sin_pe) {
      size_t pe_off = ((size_t)h * T + t) * half + d;
      float c = cos_pe[pe_off];
      float s = sin_pe[pe_off];
      float y0 = x0 * c - x1 * s;
      float y1 = x1 * c + x0 * s;
      x0 = y0; x1 = y1;
    }
    size_t dst_base = ((size_t)h * T_pad + t) * D;
    dst_htd[dst_base + d]        = d_f32_to_bf16(x0);
    dst_htd[dst_base + d + half] = d_f32_to_bf16(x1);
  }
}

// Pack V: [T, H*D] F32 -> [H, T_pad, D] BF16
__global__ void k_ltx_v_pack(const float *__restrict__ src, uint16_t *__restrict__ dst_htd, int T, int T_pad, int H, int D) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)H * T_pad * D;
  if (idx >= tot) return;
  int d = (int)(idx % D);
  size_t rem = idx / D;
  int t = (int)(rem % T_pad);
  int h = (int)(rem / T_pad);
  dst_htd[idx] = (t < T) ? d_f32_to_bf16(src[(size_t)t * (H * D) + h * D + d]) : 0;
}

// Apply per-head attention gate: out_bf16[t, h, d] = out_f32[t, h, d] * (2 * sigmoid(gate_logits[t, h]))
__global__ void k_ltx_apply_head_gate_bf16(
    const float *__restrict__ out_f32, const float *__restrict__ gate_logits,
    uint16_t *__restrict__ out_bf16, int T, int H, int D) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)T * H * D;
  if (idx >= tot) return;
  int hd = (int)(idx % (H * D));
  int t = (int)(idx / (H * D));
  int h = hd / D;
  float g = 2.0f / (1.0f + expf(-gate_logits[(size_t)t * H + h]));
  out_bf16[idx] = d_f32_to_bf16(out_f32[idx] * g);
}

// Residual add with broadcast channel gate: x[t, c] += delta[t, c] * gate[c]
__global__ void k_ltx_res_add_gate(float *__restrict__ x, const float *__restrict__ delta, const float *__restrict__ gate, int T, int D) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)T * D;
  if (idx >= tot) return;
  int c = (int)(idx % D);
  x[idx] += delta[idx] * gate[c];
}

// ===========================================================================
// 2. Engine Struct, Weight Loading, Hybrid VRAM/PCIe Block Cache
// ===========================================================================

typedef struct {
  const ec_st_entry_t *w_e;
  const ec_st_entry_t *s_e; // optional F32 scalar weight_scale
  const ec_st_entry_t *b_e; // optional bias
  float   scale;
  int     is_fp8;
  int     N, K;
  void   *w_d;   // if cached in VRAM: uint8_t* (FP8) or uint16_t* (BF16)
  float  *b_d;   // if cached in VRAM: float* [N]
} ltx_lin_t;

typedef struct {
  // Video self-attn, text cross-attn, FF
  ltx_lin_t a1_q, a1_k, a1_v, a1_g, a1_o;
  float *a1_qn, *a1_kn;
  ltx_lin_t a2_q, a2_k, a2_v, a2_g, a2_o;
  float *a2_qn, *a2_kn;
  ltx_lin_t ff_0, ff_2;
  // Audio self-attn, text cross-attn, FF
  ltx_lin_t aa1_q, aa1_k, aa1_v, aa1_g, aa1_o;
  float *aa1_qn, *aa1_kn;
  ltx_lin_t aa2_q, aa2_k, aa2_v, aa2_g, aa2_o;
  float *aa2_qn, *aa2_kn;
  ltx_lin_t aff_0, aff_2;
  // Bidirectional A/V cross-attn
  ltx_lin_t a2v_q, a2v_k, a2v_v, a2v_g, a2v_o;
  float *a2v_qn, *a2v_kn;
  ltx_lin_t v2a_q, v2a_k, v2a_v, v2a_g, v2a_o;
  float *v2a_qn, *v2a_kn;
  // AdaLN tables [9,4096], [2,4096], [9,2048], [2,2048], [5,4096], [5,2048]
  float *sst_v, *psst_v, *sst_a, *psst_a, *ca_sst_v, *ca_sst_a;
  int resident; // 1 = weights in VRAM, 0 = stream from mmap
} ltx_dit_block_t;

struct ec_ltx2 {
  ec_safetensors_t main_st;
  ec_safetensors_t gemma_st;
  ec_bpe_t        *tok;
  cudaStream_t     stream;
  int              ready;

  // Scratch buffers for FP8->BF16 weight dequant & streaming (reused across all linears!)
  uint8_t  *fp8_stage_d;   // 68 MB max (single FP8 matrix)
  uint16_t *bf16_stage_d;  // 136 MB max (dequanted BF16 matrix up to 16384*4096)
  float    *bias_stage_d;  // 256 KB max

  // 48-layer DiT hybrid cache (blocks 0..cache_limit-1 resident in VRAM when active)
  ltx_dit_block_t blocks[48];
  int             dit_cached;
};

static int ltx_find_lin(const ec_safetensors_t *st, const char *prefix, ltx_lin_t *out) {
  memset(out, 0, sizeof *out);
  char k[192];
  snprintf(k, sizeof k, "%s.weight", prefix);
  out->w_e = ec_st_find(st, k);
  if (!out->w_e || out->w_e->rank != 2) return -1;
  out->N = (int)out->w_e->shape[0];
  out->K = (int)out->w_e->shape[1];
  out->is_fp8 = (out->w_e->dtype == EC_DTYPE_FP8_E4M3) ? 1 : 0;
  out->scale = 1.0f;
  if (out->is_fp8) {
    snprintf(k, sizeof k, "%s.weight_scale", prefix);
    out->s_e = ec_st_find(st, k);
    if (out->s_e) {
      const void *p = ec_st_ptr(st, out->s_e);
      if (out->s_e->dtype == EC_DTYPE_F32) out->scale = *(const float *)p;
      else if (out->s_e->dtype == EC_DTYPE_BF16) out->scale = ltx_bf16_to_f32_host(*(const uint16_t *)p);
    }
  }
  snprintf(k, sizeof k, "%s.bias", prefix);
  out->b_e = ec_st_find(st, k);
  return 0;
}

static float *ltx_upload_vec_f32(const ec_safetensors_t *st, const char *name, cudaStream_t stream) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) return NULL;
  size_t ne = 1;
  for (int i = 0; i < e->rank; i++) ne *= (size_t)e->shape[i];
  float *h = (float *)malloc(ne * 4);
  if (!h) return NULL;
  const void *p = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_F32) {
    memcpy(h, p, ne * 4);
  } else if (e->dtype == EC_DTYPE_BF16) {
    const uint16_t *b = (const uint16_t *)p;
    for (size_t i = 0; i < ne; i++) h[i] = ltx_bf16_to_f32_host(b[i]);
  } else {
    free(h); return NULL;
  }
  float *d = NULL;
  if (cudaMalloc(&d, ne * 4) != cudaSuccess) { free(h); return NULL; }
  cudaMemcpyAsync(d, h, ne * 4, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  free(h);
  return d;
}

static int ltx_upload_lin_cache(const ec_safetensors_t *st, ltx_lin_t *lin, cudaStream_t stream) {
  if (!lin->w_e) return -1;
  size_t wb = (size_t)lin->w_e->nbytes;
  if (cudaMalloc(&lin->w_d, wb) != cudaSuccess) { lin->w_d = NULL; return -1; }
  cudaMemcpyAsync(lin->w_d, ec_st_ptr(st, lin->w_e), wb, cudaMemcpyHostToDevice, stream);
  if (lin->b_e) {
    size_t ne = (size_t)lin->N;
    float *hb = (float *)malloc(ne * 4);
    const void *bp = ec_st_ptr(st, lin->b_e);
    if (lin->b_e->dtype == EC_DTYPE_F32) memcpy(hb, bp, ne * 4);
    else {
      const uint16_t *b16 = (const uint16_t *)bp;
      for (size_t i = 0; i < ne; i++) hb[i] = ltx_bf16_to_f32_host(b16[i]);
    }
    if (cudaMalloc(&lin->b_d, ne * 4) != cudaSuccess) { free(hb); cudaFree(lin->w_d); lin->w_d = NULL; return -1; }
    cudaMemcpyAsync(lin->b_d, hb, ne * 4, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    free(hb);
  }
  return 0;
}

static void ltx_free_lin_cache(ltx_lin_t *lin) {
  if (lin->w_d) { cudaFree(lin->w_d); lin->w_d = NULL; }
  if (lin->b_d) { cudaFree(lin->b_d); lin->b_d = NULL; }
}

// Execute linear layer Y = X @ W^T + b (supports cached/streamed BF16 or FP8+weight_scale!)
static int ltx_run_lin(
    ec_ltx2_t *eng, const ec_safetensors_t *st, const ltx_lin_t *lin,
    const uint16_t *X_bf16, float *Y_f32, uint16_t *Y_bf16, int M) {
  if (!lin->w_e) return -1;
  const uint16_t *W_bf16 = NULL;
  const float *bias_d = lin->b_d;
  size_t ne = (size_t)lin->N * (size_t)lin->K;

  if (lin->is_fp8) {
    const uint8_t *w_u8 = (const uint8_t *)lin->w_d;
    if (!w_u8) {
      cudaMemcpyAsync(eng->fp8_stage_d, ec_st_ptr(st, lin->w_e), ne, cudaMemcpyHostToDevice, eng->stream);
      w_u8 = eng->fp8_stage_d;
    }
    int grid = (int)(((ne + 3) / 4 + 255) / 256);
    k_fp8_to_bf16<<<grid, 256, 0, eng->stream>>>(w_u8, lin->scale, eng->bf16_stage_d, ne);
    W_bf16 = eng->bf16_stage_d;
  } else {
    if (lin->w_d) {
      W_bf16 = (const uint16_t *)lin->w_d;
    } else {
      cudaMemcpyAsync(eng->bf16_stage_d, ec_st_ptr(st, lin->w_e), ne * 2, cudaMemcpyHostToDevice, eng->stream);
      W_bf16 = eng->bf16_stage_d;
    }
  }

  if (lin->b_e && !bias_d) {
    float *hb = (float *)malloc((size_t)lin->N * 4);
    const void *bp = ec_st_ptr(st, lin->b_e);
    if (lin->b_e->dtype == EC_DTYPE_F32) memcpy(hb, bp, (size_t)lin->N * 4);
    else {
      const uint16_t *b16 = (const uint16_t *)bp;
      for (int i = 0; i < lin->N; i++) hb[i] = ltx_bf16_to_f32_host(b16[i]);
    }
    cudaMemcpyAsync(eng->bias_stage_d, hb, (size_t)lin->N * 4, cudaMemcpyHostToDevice, eng->stream);
    cudaStreamSynchronize(eng->stream);
    free(hb);
    bias_d = eng->bias_stage_d;
  }

  return ec_ops_gemm_bf16(X_bf16, W_bf16, bias_d, Y_f32, Y_bf16, M, lin->N, lin->K, eng->stream);
}

// ===========================================================================
// 3. Exact LTX-2.3 Float64 Split RoPE Precomputation (matches rope.py:86-230)
// ===========================================================================

// positions_h: [n_pos_dims, T, 2] if use_middle else [n_pos_dims, T]
// Outputs cos_d, sin_d of shape [H, T, D/2] F32 on GPU!
static int ltx_precompute_split_rope(
    const float *positions_h, int n_pos_dims, int T,
    const int *max_pos, int use_middle,
    int inner_dim, int H, double theta,
    float **out_cos_d, float **out_sin_d, cudaStream_t stream) {
  int half = inner_dim / 2;
  int d_head_half = half / H;
  int n_elem = 2 * n_pos_dims;
  int Kf = inner_dim / n_elem;
  int current_freqs = Kf * n_pos_dims;
  int pad_size = half - current_freqs;

  double *indices = (double *)malloc((size_t)Kf * sizeof(double));
  for (int k = 0; k < Kf; k++) {
    double frac = (Kf > 1) ? ((double)k / (double)(Kf - 1)) : 0.0;
    indices[k] = pow(theta, frac) * (M_PI * 0.5);
  }

  float *cos_h = (float *)malloc((size_t)H * T * d_head_half * sizeof(float));
  float *sin_h = (float *)malloc((size_t)H * T * d_head_half * sizeof(float));
  if (!cos_h || !sin_h) { free(indices); free(cos_h); free(sin_h); return -1; }

  for (int t = 0; t < T; t++) {
    double fpos[4];
    for (int p = 0; p < n_pos_dims; p++) {
      double pv;
      if (use_middle) {
        size_t off = ((size_t)p * T + t) * 2;
        pv = 0.5 * ((double)positions_h[off] + (double)positions_h[off + 1]);
      } else {
        pv = (double)positions_h[(size_t)p * T + t];
      }
      fpos[p] = (pv / (double)max_pos[p]) * 2.0 - 1.0;
    }
    for (int f = 0; f < half; f++) {
      float cv, sv;
      if (f < pad_size) {
        cv = 1.0f; sv = 0.0f;
      } else {
        int fi = f - pad_size;
        // Exact rope.py:164: .transpose(-1, -2).flatten(2) -> p is fast index, k is slow index!
        int k = fi / n_pos_dims;
        int p = fi % n_pos_dims;
        double ang = (double)((float)indices[k]) * fpos[p];
        cv = cosf((float)ang);
        sv = sinf((float)ang);
      }
      int h = f / d_head_half;
      int d = f % d_head_half;
      size_t dst = ((size_t)h * T + t) * d_head_half + d;
      cos_h[dst] = cv;
      sin_h[dst] = sv;
    }
  }
  free(indices);

  size_t bytes = (size_t)H * T * d_head_half * sizeof(float);
  if (cudaMalloc(out_cos_d, bytes) != cudaSuccess || cudaMalloc(out_sin_d, bytes) != cudaSuccess) {
    free(cos_h); free(sin_h); return -1;
  }
  cudaMemcpyAsync(*out_cos_d, cos_h, bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(*out_sin_d, sin_h, bytes, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  free(cos_h); free(sin_h);
  return 0;
}

// ===========================================================================
// 4. Stage 1: Gemma-3-12B-IT FP8 + FeatureExtractorV2 + 8L Connectors
// ===========================================================================

static int run_ltx_connector_8l(
    ec_ltx2_t *eng, const char *prefix,
    const float *feats_k_d, int K, int dim, int H,
    float *out_ctx_d) {
  const int S = 1024;
  const int D = dim / H;
  const int S_pad = S; // 1024 is already multiple of 64!

  // Build initial [S, dim] F32: [0..K-1] from feats_k_d, [K..S-1] from learnable_registers[t % 128]
  char kname[192];
  snprintf(kname, sizeof kname, "%s.learnable_registers", prefix);
  float *reg_d = ltx_upload_vec_f32(&eng->main_st, kname, eng->stream);
  if (!reg_d) return -1;

  float *x_h = (float *)malloc((size_t)S * dim * 4);
  float *reg_h = (float *)malloc((size_t)128 * dim * 4);
  cudaMemcpy(x_h, feats_k_d, (size_t)K * dim * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(reg_h, reg_d, (size_t)128 * dim * 4, cudaMemcpyDeviceToHost);
  cudaFree(reg_d);
  for (int t = K; t < S; t++) {
    memcpy(x_h + (size_t)t * dim, reg_h + (size_t)(t % 128) * dim, (size_t)dim * 4);
  }
  free(reg_h);

  float *x_d = NULL, *tmp_f = NULL, *q_f = NULL, *k_f = NULL, *v_f = NULL, *g_f = NULL, *ff_f = NULL;
  uint16_t *xn_b = NULL, *qp_b = NULL, *kp_b = NULL, *vp_b = NULL, *ob_b = NULL, *ff_b = NULL;
  cudaMalloc(&x_d,   (size_t)S * dim * 4);
  cudaMalloc(&tmp_f, (size_t)S * dim * 4);
  cudaMalloc(&q_f,   (size_t)S * dim * 4);
  cudaMalloc(&k_f,   (size_t)S * dim * 4);
  cudaMalloc(&v_f,   (size_t)S * dim * 4);
  cudaMalloc(&g_f,   (size_t)S * H * 4);
  cudaMalloc(&ff_f,  (size_t)S * (dim * 4) * 4);
  cudaMalloc(&xn_b,  (size_t)S * dim * 2);
  cudaMalloc(&qp_b,  (size_t)H * S_pad * D * 2);
  cudaMalloc(&kp_b,  (size_t)H * S_pad * D * 2);
  cudaMalloc(&vp_b,  (size_t)H * S_pad * D * 2);
  cudaMalloc(&ob_b,  (size_t)S * dim * 2);
  cudaMalloc(&ff_b,  (size_t)S * (dim * 4) * 2);
  cudaMemcpyAsync(x_d, x_h, (size_t)S * dim * 4, cudaMemcpyHostToDevice, eng->stream);
  cudaStreamSynchronize(eng->stream);
  free(x_h);

  // Precompute 1D split float64 RoPE on indices 0..1023, max_pos=[4096], use_middle=False
  float *pos_h = (float *)malloc((size_t)S * 4);
  for (int t = 0; t < S; t++) pos_h[t] = (float)t;
  int max_p[1] = {4096};
  float *cos_d = NULL, *sin_d = NULL;
  ltx_precompute_split_rope(pos_h, 1, S, max_p, 0, dim, H, 10000.0, &cos_d, &sin_d, eng->stream);
  free(pos_h);

  float scale = 1.0f / sqrtf((float)D);
  for (int l = 0; l < 8; l++) {
    char bp[192], ln[220];
    snprintf(bp, sizeof bp, "%s.transformer_1d_blocks.%d", prefix, l);

    // 1. Pre-attn affine-free RMSNorm
    k_ltx_rmsnorm<<<S, 256, 0, eng->stream>>>(x_d, NULL, NULL, NULL, NULL, xn_b, S, dim, 1e-6f);

    // 2. Self-attention (Q, K, V, gate_logits, Q/K norm + split RoPE + FA2 + head gate + to_out)
    ltx_lin_t lq, lk, lv, lg, lo;
    snprintf(ln, sizeof ln, "%s.attn1.to_q", bp); ltx_find_lin(&eng->main_st, ln, &lq);
    snprintf(ln, sizeof ln, "%s.attn1.to_k", bp); ltx_find_lin(&eng->main_st, ln, &lk);
    snprintf(ln, sizeof ln, "%s.attn1.to_v", bp); ltx_find_lin(&eng->main_st, ln, &lv);
    snprintf(ln, sizeof ln, "%s.attn1.to_gate_logits", bp); ltx_find_lin(&eng->main_st, ln, &lg);
    snprintf(ln, sizeof ln, "%s.attn1.to_out.0", bp); ltx_find_lin(&eng->main_st, ln, &lo);
    snprintf(ln, sizeof ln, "%s.attn1.q_norm.weight", bp); float *qnw = ltx_upload_vec_f32(&eng->main_st, ln, eng->stream);
    snprintf(ln, sizeof ln, "%s.attn1.k_norm.weight", bp); float *knw = ltx_upload_vec_f32(&eng->main_st, ln, eng->stream);

    ltx_run_lin(eng, &eng->main_st, &lq, xn_b, q_f, NULL, S);
    ltx_run_lin(eng, &eng->main_st, &lk, xn_b, k_f, NULL, S);
    ltx_run_lin(eng, &eng->main_st, &lv, xn_b, v_f, NULL, S);
    ltx_run_lin(eng, &eng->main_st, &lg, xn_b, g_f, NULL, S);

    k_ltx_qk_norm_rope_pack<<<S, 256, 0, eng->stream>>>(q_f, qnw, cos_d, sin_d, qp_b, S, S_pad, H, D, 1e-6f);
    k_ltx_qk_norm_rope_pack<<<S, 256, 0, eng->stream>>>(k_f, knw, cos_d, sin_d, kp_b, S, S_pad, H, D, 1e-6f);
    k_ltx_v_pack<<<((size_t)H * S_pad * D + 255) / 256, 256, 0, eng->stream>>>(v_f, vp_b, S, S_pad, H, D);
    ec_ops_fa2_bf16(qp_b, kp_b, vp_b, tmp_f, NULL, S, S, S_pad, S_pad, H, H, D, scale, 0, eng->stream);
    k_ltx_apply_head_gate_bf16<<<((size_t)S * dim + 255) / 256, 256, 0, eng->stream>>>(tmp_f, g_f, ob_b, S, H, D);
    ltx_run_lin(eng, &eng->main_st, &lo, ob_b, tmp_f, NULL, S);
    k_f32_add_inplace<<<((size_t)S * dim + 255) / 256, 256, 0, eng->stream>>>(x_d, tmp_f, (size_t)S * dim);
    cudaStreamSynchronize(eng->stream);
    cudaFree(qnw); cudaFree(knw);

    // 3. Pre-FF affine-free RMSNorm + GELU-tanh FF
    k_ltx_rmsnorm<<<S, 256, 0, eng->stream>>>(x_d, NULL, NULL, NULL, NULL, xn_b, S, dim, 1e-6f);
    ltx_lin_t lf0, lf2;
    snprintf(ln, sizeof ln, "%s.ff.net.0.proj", bp); ltx_find_lin(&eng->main_st, ln, &lf0);
    snprintf(ln, sizeof ln, "%s.ff.net.2", bp);      ltx_find_lin(&eng->main_st, ln, &lf2);
    ltx_run_lin(eng, &eng->main_st, &lf0, xn_b, ff_f, NULL, S);
    k_gelu_tanh_to_bf16<<<((size_t)S * dim * 4 + 255) / 256, 256, 0, eng->stream>>>(ff_f, ff_b, (size_t)S * dim * 4);
    ltx_run_lin(eng, &eng->main_st, &lf2, ff_b, tmp_f, NULL, S);
    k_f32_add_inplace<<<((size_t)S * dim + 255) / 256, 256, 0, eng->stream>>>(x_d, tmp_f, (size_t)S * dim);
  }

  // Final affine-free RMSNorm
  k_ltx_rmsnorm<<<S, 256, 0, eng->stream>>>(x_d, NULL, NULL, NULL, out_ctx_d, NULL, S, dim, 1e-6f);
  cudaStreamSynchronize(eng->stream);

  cudaFree(cos_d); cudaFree(sin_d);
  cudaFree(x_d); cudaFree(tmp_f); cudaFree(q_f); cudaFree(k_f); cudaFree(v_f); cudaFree(g_f); cudaFree(ff_f);
  cudaFree(xn_b); cudaFree(qp_b); cudaFree(kp_b); cudaFree(vp_b); cudaFree(ob_b); cudaFree(ff_b);
  return 0;
}

static int run_gemma3_and_connectors(
    ec_ltx2_t *eng, const char *prompt,
    float *video_ctx_d, float *audio_ctx_d, double *out_ms) {
  double t0 = ltx_now_ms();
  int32_t ids[1024];
  int K = ec_bpe_encode_gemma3(eng->tok, prompt ? prompt : "", ids, 1024);
  if (K <= 0) return -1;
  if (K > 1024) K = 1024;
  fprintf(stderr, "[ltx2] Gemma-3 tokenized prompt -> %d tokens (bos=%d)\n", K, ids[0]);

  const int DIM = 3840, HQ = 16, HKV = 8, D = 256, FFN = 15360;
  const ec_st_entry_t *emb_e = ec_st_find(&eng->gemma_st, "model.embed_tokens.weight");
  if (!emb_e) return -1;
  const uint16_t *emb_base = (const uint16_t *)ec_st_ptr(&eng->gemma_st, emb_e);
  float emb_scale = sqrtf((float)DIM);

  float *x_h = (float *)malloc((size_t)K * DIM * 4);
  for (int t = 0; t < K; t++) {
    int id = ids[t];
    if (id < 0 || id >= (int)emb_e->shape[0]) id = 0;
    const uint16_t *row = emb_base + (size_t)id * DIM;
    for (int d = 0; d < DIM; d++) {
      // Round scaled embed through BF16 exactly like HF Gemma3TextModel
      float v = ltx_bf16_to_f32_host(row[d]) * emb_scale;
      x_h[(size_t)t * DIM + d] = ltx_bf16_to_f32_host(ltx_f32_to_bf16_host(v));
    }
  }

  float *hs_all_d = NULL, *x_d = NULL, *tmp_f = NULL, *q_f = NULL, *k_f = NULL, *v_f = NULL, *g_f = NULL, *u_f = NULL;
  uint16_t *xn_b = NULL, *ao_b = NULL, *ff_b = NULL;
  cudaMalloc(&hs_all_d, (size_t)49 * K * DIM * 4);
  cudaMalloc(&x_d,      (size_t)K * DIM * 4);
  cudaMalloc(&tmp_f,    (size_t)K * DIM * 4);
  cudaMalloc(&q_f,      (size_t)K * (HQ * D) * 4);
  cudaMalloc(&k_f,      (size_t)K * (HKV * D) * 4);
  cudaMalloc(&v_f,      (size_t)K * (HKV * D) * 4);
  cudaMalloc(&g_f,      (size_t)K * FFN * 4);
  cudaMalloc(&u_f,      (size_t)K * FFN * 4);
  cudaMalloc(&xn_b,     (size_t)K * DIM * 2);
  cudaMalloc(&ao_b,     (size_t)K * (HQ * D) * 2);
  cudaMalloc(&ff_b,     (size_t)K * FFN * 2);

  cudaMemcpyAsync(x_d, x_h, (size_t)K * DIM * 4, cudaMemcpyHostToDevice, eng->stream);
  cudaMemcpyAsync(hs_all_d, x_d, (size_t)K * DIM * 4, cudaMemcpyDeviceToDevice, eng->stream);
  cudaStreamSynchronize(eng->stream);
  free(x_h);

  const float attn_scale = 1.0f / sqrtf(256.0f); // query_pre_attn_scalar = 256
  for (int l = 0; l < 48; l++) {
    char pfx[96], nm[160];
    snprintf(pfx, sizeof pfx, "model.layers.%d", l);

    snprintf(nm, sizeof nm, "%s.input_layernorm.weight", pfx);
    float *w_in = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);
    snprintf(nm, sizeof nm, "%s.self_attn.q_norm.weight", pfx);
    float *w_qn = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);
    snprintf(nm, sizeof nm, "%s.self_attn.k_norm.weight", pfx);
    float *w_kn = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);
    snprintf(nm, sizeof nm, "%s.post_attention_layernorm.weight", pfx);
    float *w_pa = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);
    snprintf(nm, sizeof nm, "%s.pre_feedforward_layernorm.weight", pfx);
    float *w_pf = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);
    snprintf(nm, sizeof nm, "%s.post_feedforward_layernorm.weight", pfx);
    float *w_pof = ltx_upload_vec_f32(&eng->gemma_st, nm, eng->stream);

    ltx_lin_t lq, lk, lv, lo, lg, lu, ld;
    snprintf(nm, sizeof nm, "%s.self_attn.q_proj", pfx); ltx_find_lin(&eng->gemma_st, nm, &lq);
    snprintf(nm, sizeof nm, "%s.self_attn.k_proj", pfx); ltx_find_lin(&eng->gemma_st, nm, &lk);
    snprintf(nm, sizeof nm, "%s.self_attn.v_proj", pfx); ltx_find_lin(&eng->gemma_st, nm, &lv);
    snprintf(nm, sizeof nm, "%s.self_attn.o_proj", pfx); ltx_find_lin(&eng->gemma_st, nm, &lo);
    snprintf(nm, sizeof nm, "%s.mlp.gate_proj", pfx);    ltx_find_lin(&eng->gemma_st, nm, &lg);
    snprintf(nm, sizeof nm, "%s.mlp.up_proj", pfx);      ltx_find_lin(&eng->gemma_st, nm, &lu);
    snprintf(nm, sizeof nm, "%s.mlp.down_proj", pfx);    ltx_find_lin(&eng->gemma_st, nm, &ld);

    // 1. Attention
    k_gemma3_rmsnorm<<<K, 256, 0, eng->stream>>>(x_d, w_in, NULL, xn_b, K, DIM, 1e-6f);
    ltx_run_lin(eng, &eng->gemma_st, &lq, xn_b, q_f, NULL, K);
    ltx_run_lin(eng, &eng->gemma_st, &lk, xn_b, k_f, NULL, K);
    ltx_run_lin(eng, &eng->gemma_st, &lv, xn_b, v_f, NULL, K);
    k_gemma3_rmsnorm<<<K * HQ,  256, 0, eng->stream>>>(q_f, w_qn, q_f, NULL, K * HQ,  D, 1e-6f);
    k_gemma3_rmsnorm<<<K * HKV, 256, 0, eng->stream>>>(k_f, w_kn, k_f, NULL, K * HKV, D, 1e-6f);

    int is_global = (((l + 1) % 6) == 0);
    double r_theta = is_global ? 1000000.0 : 10000.0;
    double r_scale = is_global ? (1.0 / 8.0) : 1.0;
    k_gemma3_rope_inplace<<<dim3(K, HQ),  D / 2, 0, eng->stream>>>(q_f, K, HQ,  D, r_theta, r_scale);
    k_gemma3_rope_inplace<<<dim3(K, HKV), D / 2, 0, eng->stream>>>(k_f, K, HKV, D, r_theta, r_scale);
    k_gemma3_causal_gqa<<<dim3(K, HQ), D, 0, eng->stream>>>(q_f, k_f, v_f, NULL, ao_b, K, HQ, HKV, D, attn_scale);
    ltx_run_lin(eng, &eng->gemma_st, &lo, ao_b, tmp_f, NULL, K);
    k_gemma3_rmsnorm<<<K, 256, 0, eng->stream>>>(tmp_f, w_pa, tmp_f, NULL, K, DIM, 1e-6f);
    k_f32_add_inplace<<<((size_t)K * DIM + 255) / 256, 256, 0, eng->stream>>>(x_d, tmp_f, (size_t)K * DIM);

    // 2. MLP (GeGLU with gelu_pytorch_tanh)
    k_gemma3_rmsnorm<<<K, 256, 0, eng->stream>>>(x_d, w_pf, NULL, xn_b, K, DIM, 1e-6f);
    ltx_run_lin(eng, &eng->gemma_st, &lg, xn_b, g_f, NULL, K);
    ltx_run_lin(eng, &eng->gemma_st, &lu, xn_b, u_f, NULL, K);
    k_gelu_tanh_mul_bf16<<<((size_t)K * FFN + 255) / 256, 256, 0, eng->stream>>>(g_f, u_f, ff_b, (size_t)K * FFN);
    ltx_run_lin(eng, &eng->gemma_st, &ld, ff_b, tmp_f, NULL, K);
    k_gemma3_rmsnorm<<<K, 256, 0, eng->stream>>>(tmp_f, w_pof, tmp_f, NULL, K, DIM, 1e-6f);
    k_f32_add_inplace<<<((size_t)K * DIM + 255) / 256, 256, 0, eng->stream>>>(x_d, tmp_f, (size_t)K * DIM);

    if (l < 47) {
      cudaMemcpyAsync(hs_all_d + (size_t)(l + 1) * K * DIM, x_d, (size_t)K * DIM * 4, cudaMemcpyDeviceToDevice, eng->stream);
    } else {
      // HF Gemma3TextModel applies final model.norm before appending hidden_states[48]!
      float *w_fn = ltx_upload_vec_f32(&eng->gemma_st, "model.norm.weight", eng->stream);
      k_gemma3_rmsnorm<<<K, 256, 0, eng->stream>>>(x_d, w_fn, hs_all_d + (size_t)48 * K * DIM, NULL, K, DIM, 1e-6f);
      cudaStreamSynchronize(eng->stream);
      cudaFree(w_fn);
    }
    cudaStreamSynchronize(eng->stream);
    cudaFree(w_in); cudaFree(w_qn); cudaFree(w_kn); cudaFree(w_pa); cudaFree(w_pf); cudaFree(w_pof);
  }

  cudaFree(x_d); cudaFree(tmp_f); cudaFree(q_f); cudaFree(k_f); cudaFree(v_f); cudaFree(g_f); cudaFree(u_f);
  cudaFree(xn_b); cudaFree(ao_b); cudaFree(ff_b);

  // FeatureExtractorV2: per-token per-layer RMSNorm -> interleave (d,l) -> rescale -> video & audio aggregate_embed!
  // Row-chunked 100% SEQUENTIAL reads (512 contiguous rows = 192 MB/chunk -> zero HDD seeks!)
  fprintf(stderr, "[ltx2] Gemma-3 48L done in %.1fs; running FeatureExtractorV2...\n", (ltx_now_ms() - t0) / 1000.0);
  float *inv_rms_d = NULL, *v_feats_d = NULL, *a_feats_d = NULL;
  cudaMalloc(&inv_rms_d, (size_t)49 * K * 4);
  cudaMalloc(&v_feats_d, (size_t)K * 4096 * 4);
  cudaMalloc(&a_feats_d, (size_t)K * 2048 * 4);
  k_ltx_layer_tok_inv_rms<<<dim3(K, 49), 256, 0, eng->stream>>>(hs_all_d, inv_rms_d, K);

  const int TOTAL_C = 188160;
  const int ROW_CH  = 512; // 512 * 188160 * 2 = 192.6 MB contiguous BF16 per chunk!
  uint16_t *x_full_b = NULL, *w_ch_b = NULL;
  float *part_f = NULL;
  cudaMalloc(&x_full_b, (size_t)K * TOTAL_C * 2);
  cudaMalloc(&w_ch_b,   (size_t)ROW_CH * TOTAL_C * 2);
  cudaMalloc(&part_f,   (size_t)K * ROW_CH * 4);

  auto run_agg = [&](const char *wname, const char *bname, int N_out, float *dst_f) {
    const ec_st_entry_t *we = ec_st_find(&eng->main_st, wname);
    const uint16_t *wbase = (const uint16_t *)ec_st_ptr(&eng->main_st, we);
    float *bd = ltx_upload_vec_f32(&eng->main_st, bname, eng->stream);
    float rescale = sqrtf((float)N_out / 3840.0f);
    k_ltx_fev2_full<<<((size_t)K * TOTAL_C + 255) / 256, 256, 0, eng->stream>>>(
        hs_all_d, inv_rms_d, x_full_b, K, rescale);
    for (int r0 = 0; r0 < N_out; r0 += ROW_CH) {
      int rc = (r0 + ROW_CH <= N_out) ? ROW_CH : (N_out - r0);
      cudaMemcpyAsync(w_ch_b, wbase + (size_t)r0 * TOTAL_C, (size_t)rc * TOTAL_C * 2, cudaMemcpyHostToDevice, eng->stream);
      ec_ops_gemm_bf16(x_full_b, w_ch_b, bd ? (bd + r0) : NULL, part_f, NULL, K, rc, TOTAL_C, eng->stream);
      k_scatter_cols_f32<<<((size_t)K * rc + 255) / 256, 256, 0, eng->stream>>>(part_f, dst_f, K, r0, rc, N_out);
    }
    cudaStreamSynchronize(eng->stream);
    if (bd) cudaFree(bd);
  };
  run_agg("text_embedding_projection.video_aggregate_embed.weight",
          "text_embedding_projection.video_aggregate_embed.bias", 4096, v_feats_d);
  run_agg("text_embedding_projection.audio_aggregate_embed.weight",
          "text_embedding_projection.audio_aggregate_embed.bias", 2048, a_feats_d);
  fprintf(stderr, "[ltx2] FeatureExtractorV2 done (%.1fs); running 8L Video+Audio Connectors...\n", (ltx_now_ms() - t0) / 1000.0);

  cudaFree(x_full_b); cudaFree(w_ch_b); cudaFree(part_f); cudaFree(hs_all_d); cudaFree(inv_rms_d);

  // Run 8-layer Video & Audio Embeddings1DConnectors (output: [1024, 4096] & [1024, 2048])
  int rc_v = run_ltx_connector_8l(eng, "model.diffusion_model.video_embeddings_connector", v_feats_d, K, 4096, 32, video_ctx_d);
  int rc_a = run_ltx_connector_8l(eng, "model.diffusion_model.audio_embeddings_connector", a_feats_d, K, 2048, 32, audio_ctx_d);
  cudaFree(v_feats_d); cudaFree(a_feats_d);
  fprintf(stderr, "[ltx2] Text + 8L Connectors complete in %.1fs\n", (ltx_now_ms() - t0) / 1000.0);

  if (out_ms) *out_ms = ltx_now_ms() - t0;
  return (rc_v == 0 && rc_a == 0) ? 0 : -1;
}

// ===========================================================================
// 5. Stage 2: 8-Step Distilled Dual-Stream AVTransformer3DModel (48 Layers)
// ===========================================================================

static void ltx_free_dit_cache(ec_ltx2_t *eng) {
  for (int b = 0; b < 48; b++) {
    ltx_dit_block_t *blk = &eng->blocks[b];
    ltx_free_lin_cache(&blk->a1_q); ltx_free_lin_cache(&blk->a1_k); ltx_free_lin_cache(&blk->a1_v);
    ltx_free_lin_cache(&blk->a1_g); ltx_free_lin_cache(&blk->a1_o);
    ltx_free_lin_cache(&blk->a2_q); ltx_free_lin_cache(&blk->a2_k); ltx_free_lin_cache(&blk->a2_v);
    ltx_free_lin_cache(&blk->a2_g); ltx_free_lin_cache(&blk->a2_o);
    ltx_free_lin_cache(&blk->ff_0); ltx_free_lin_cache(&blk->ff_2);
    ltx_free_lin_cache(&blk->aa1_q); ltx_free_lin_cache(&blk->aa1_k); ltx_free_lin_cache(&blk->aa1_v);
    ltx_free_lin_cache(&blk->aa1_g); ltx_free_lin_cache(&blk->aa1_o);
    ltx_free_lin_cache(&blk->aa2_q); ltx_free_lin_cache(&blk->aa2_k); ltx_free_lin_cache(&blk->aa2_v);
    ltx_free_lin_cache(&blk->aa2_g); ltx_free_lin_cache(&blk->aa2_o);
    ltx_free_lin_cache(&blk->aff_0); ltx_free_lin_cache(&blk->aff_2);
    ltx_free_lin_cache(&blk->a2v_q); ltx_free_lin_cache(&blk->a2v_k); ltx_free_lin_cache(&blk->a2v_v);
    ltx_free_lin_cache(&blk->a2v_g); ltx_free_lin_cache(&blk->a2v_o);
    ltx_free_lin_cache(&blk->v2a_q); ltx_free_lin_cache(&blk->v2a_k); ltx_free_lin_cache(&blk->v2a_v);
    ltx_free_lin_cache(&blk->v2a_g); ltx_free_lin_cache(&blk->v2a_o);
    float **vecs[14] = {
      &blk->a1_qn, &blk->a1_kn, &blk->a2_qn, &blk->a2_kn,
      &blk->aa1_qn, &blk->aa1_kn, &blk->aa2_qn, &blk->aa2_kn,
      &blk->a2v_qn, &blk->a2v_kn, &blk->v2a_qn, &blk->v2a_kn,
      &blk->sst_v, &blk->psst_v
    };
    for (int i = 0; i < 14; i++) { if (*vecs[i]) { cudaFree(*vecs[i]); *vecs[i] = NULL; } }
    if (blk->sst_a)    { cudaFree(blk->sst_a);    blk->sst_a = NULL; }
    if (blk->psst_a)   { cudaFree(blk->psst_a);   blk->psst_a = NULL; }
    if (blk->ca_sst_v) { cudaFree(blk->ca_sst_v); blk->ca_sst_v = NULL; }
    if (blk->ca_sst_a) { cudaFree(blk->ca_sst_a); blk->ca_sst_a = NULL; }
    blk->resident = 0;
  }
  eng->dit_cached = 0;
}

static int ltx_ensure_dit_cache(ec_ltx2_t *eng) {
  if (eng->dit_cached) return 0;
  size_t free_b = 0, tot_b = 0;
  cudaMemGetInfo(&free_b, &tot_b);
  // Keep 4.4 GB VRAM headroom for activations, attention Q/K/V, and 302MB adaln BF16 staging (~40/48 FP8 blocks in VRAM!)
  size_t reserve_b = 4400ULL * 1024ULL * 1024ULL;
  size_t budget_b = (free_b > reserve_b) ? (free_b - reserve_b) : 0;
  size_t used_b = 0;
  int n_res = 0;

  for (int b = 0; b < 48; b++) {
    ltx_dit_block_t *blk = &eng->blocks[b];
    memset(blk, 0, sizeof *blk);
    char bp[128], ln[192];
    snprintf(bp, sizeof bp, "model.diffusion_model.transformer_blocks.%d", b);

    auto fl = [&](const char *sub, ltx_lin_t *dst) {
      snprintf(ln, sizeof ln, "%s.%s", bp, sub);
      return ltx_find_lin(&eng->main_st, ln, dst);
    };
    if (fl("attn1.to_q", &blk->a1_q) || fl("attn1.to_k", &blk->a1_k) || fl("attn1.to_v", &blk->a1_v) ||
        fl("attn1.to_gate_logits", &blk->a1_g) || fl("attn1.to_out.0", &blk->a1_o) ||
        fl("attn2.to_q", &blk->a2_q) || fl("attn2.to_k", &blk->a2_k) || fl("attn2.to_v", &blk->a2_v) ||
        fl("attn2.to_gate_logits", &blk->a2_g) || fl("attn2.to_out.0", &blk->a2_o) ||
        fl("ff.net.0.proj", &blk->ff_0) || fl("ff.net.2", &blk->ff_2) ||
        fl("audio_attn1.to_q", &blk->aa1_q) || fl("audio_attn1.to_k", &blk->aa1_k) || fl("audio_attn1.to_v", &blk->aa1_v) ||
        fl("audio_attn1.to_gate_logits", &blk->aa1_g) || fl("audio_attn1.to_out.0", &blk->aa1_o) ||
        fl("audio_attn2.to_q", &blk->aa2_q) || fl("audio_attn2.to_k", &blk->aa2_k) || fl("audio_attn2.to_v", &blk->aa2_v) ||
        fl("audio_attn2.to_gate_logits", &blk->aa2_g) || fl("audio_attn2.to_out.0", &blk->aa2_o) ||
        fl("audio_ff.net.0.proj", &blk->aff_0) || fl("audio_ff.net.2", &blk->aff_2) ||
        fl("audio_to_video_attn.to_q", &blk->a2v_q) || fl("audio_to_video_attn.to_k", &blk->a2v_k) ||
        fl("audio_to_video_attn.to_v", &blk->a2v_v) || fl("audio_to_video_attn.to_gate_logits", &blk->a2v_g) ||
        fl("audio_to_video_attn.to_out.0", &blk->a2v_o) ||
        fl("video_to_audio_attn.to_q", &blk->v2a_q) || fl("video_to_audio_attn.to_k", &blk->v2a_k) ||
        fl("video_to_audio_attn.to_v", &blk->v2a_v) || fl("video_to_audio_attn.to_gate_logits", &blk->v2a_g) ||
        fl("video_to_audio_attn.to_out.0", &blk->v2a_o)) {
      fprintf(stderr, "[ltx2] Missing tensor in transformer_blocks.%d\n", b);
      ltx_free_dit_cache(eng);
      return -1;
    }

    auto uv = [&](const char *sub) {
      snprintf(ln, sizeof ln, "%s.%s", bp, sub);
      return ltx_upload_vec_f32(&eng->main_st, ln, eng->stream);
    };
    blk->a1_qn = uv("attn1.q_norm.weight"); blk->a1_kn = uv("attn1.k_norm.weight");
    blk->a2_qn = uv("attn2.q_norm.weight"); blk->a2_kn = uv("attn2.k_norm.weight");
    blk->aa1_qn = uv("audio_attn1.q_norm.weight"); blk->aa1_kn = uv("audio_attn1.k_norm.weight");
    blk->aa2_qn = uv("audio_attn2.q_norm.weight"); blk->aa2_kn = uv("audio_attn2.k_norm.weight");
    blk->a2v_qn = uv("audio_to_video_attn.q_norm.weight"); blk->a2v_kn = uv("audio_to_video_attn.k_norm.weight");
    blk->v2a_qn = uv("video_to_audio_attn.q_norm.weight"); blk->v2a_kn = uv("video_to_audio_attn.k_norm.weight");
    blk->sst_v = uv("scale_shift_table");
    blk->psst_v = uv("prompt_scale_shift_table");
    blk->sst_a = uv("audio_scale_shift_table");
    blk->psst_a = uv("audio_prompt_scale_shift_table");
    blk->ca_sst_v = uv("scale_shift_table_a2v_ca_video");
    blk->ca_sst_a = uv("scale_shift_table_a2v_ca_audio");

    // Compute exact byte size of this block's linear weights; prefer caching FP8 blocks (2x denser in VRAM!)
    ltx_lin_t *all_lins[34] = {
      &blk->a1_q, &blk->a1_k, &blk->a1_v, &blk->a1_g, &blk->a1_o,
      &blk->a2_q, &blk->a2_k, &blk->a2_v, &blk->a2_g, &blk->a2_o,
      &blk->ff_0, &blk->ff_2,
      &blk->aa1_q, &blk->aa1_k, &blk->aa1_v, &blk->aa1_g, &blk->aa1_o,
      &blk->aa2_q, &blk->aa2_k, &blk->aa2_v, &blk->aa2_g, &blk->aa2_o,
      &blk->aff_0, &blk->aff_2,
      &blk->a2v_q, &blk->a2v_k, &blk->a2v_v, &blk->a2v_g, &blk->a2v_o,
      &blk->v2a_q, &blk->v2a_k, &blk->v2a_v, &blk->v2a_g, &blk->v2a_o
    };
    size_t blk_bytes = 0;
    for (int i = 0; i < 34; i++) {
      blk_bytes += (size_t)all_lins[i]->w_e->nbytes + (all_lins[i]->b_e ? (size_t)all_lins[i]->N * 4 : 0);
    }
    if (blk->a1_q.is_fp8 && used_b + blk_bytes <= budget_b) {
      int ok = 1;
      for (int i = 0; i < 34; i++) {
        if (ltx_upload_lin_cache(&eng->main_st, all_lins[i], eng->stream) != 0) { ok = 0; break; }
      }
      if (ok) {
        blk->resident = 1;
        used_b += blk_bytes;
        n_res++;
      } else {
        for (int i = 0; i < 34; i++) ltx_free_lin_cache(all_lins[i]);
      }
    }
  }
  eng->dit_cached = 1;
  fprintf(stderr, "[ltx2] DiT hybrid cache ready: %d/48 blocks resident in %.2f GB VRAM (streaming %d blocks)\n",
          n_res, (double)used_b / 1e9, 48 - n_res);
  return 0;
}

// Compute PixArtAlpha AdaLayerNormSingle(t_scaled) -> (modulation [coeff*dim], embedded [dim]) on GPU!
static int ltx_eval_adaln_single(
    ec_ltx2_t *eng, const char *prefix, float t_scaled, int dim, int coeff,
    float *out_mod_d, float *out_emb_d) {
  float emb256_h[256];
  const double log_max = log(10000.0);
  for (int i = 0; i < 128; i++) {
    double freq = exp(-log_max * (double)i / 128.0);
    double arg = (double)t_scaled * freq;
    // flip_sin_to_cos = True: [cos, sin]
    emb256_h[i]       = (float)cos(arg);
    emb256_h[128 + i] = (float)sin(arg);
  }
  uint16_t emb256_b[256];
  for (int i = 0; i < 256; i++) emb256_b[i] = ltx_f32_to_bf16_host(emb256_h[i]);

  uint16_t *in_b = NULL, *h_b = NULL;
  float *h_f = NULL, *e_f = NULL;
  cudaMalloc(&in_b, 256 * 2);
  cudaMalloc(&h_b,  (size_t)dim * 2);
  cudaMalloc(&h_f,  (size_t)dim * 4);
  cudaMalloc(&e_f,  (size_t)dim * 4);
  cudaMemcpyAsync(in_b, emb256_b, 256 * 2, cudaMemcpyHostToDevice, eng->stream);

  char ln[192];
  ltx_lin_t l1, l2, lo;
  snprintf(ln, sizeof ln, "%s.emb.timestep_embedder.linear_1", prefix); ltx_find_lin(&eng->main_st, ln, &l1);
  snprintf(ln, sizeof ln, "%s.emb.timestep_embedder.linear_2", prefix); ltx_find_lin(&eng->main_st, ln, &l2);
  snprintf(ln, sizeof ln, "%s.linear", prefix);                         ltx_find_lin(&eng->main_st, ln, &lo);

  ltx_run_lin(eng, &eng->main_st, &l1, in_b, h_f, NULL, 1);
  ec_ops_silu_f32_to_bf16(h_f, h_b, dim, eng->stream);
  ltx_run_lin(eng, &eng->main_st, &l2, h_b, e_f, NULL, 1);
  if (out_emb_d) cudaMemcpyAsync(out_emb_d, e_f, (size_t)dim * 4, cudaMemcpyDeviceToDevice, eng->stream);
  ec_ops_silu_f32_to_bf16(e_f, h_b, dim, eng->stream);
  ltx_run_lin(eng, &eng->main_st, &lo, h_b, out_mod_d, NULL, 1);
  cudaStreamSynchronize(eng->stream);
  (void)coeff;

  cudaFree(in_b); cudaFree(h_b); cudaFree(h_f); cudaFree(e_f);
  return 0;
}

__global__ void k_vec_add2(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

// Helper: execute one LTX Attention module (self-attn, text cross-attn, or A/V cross-attn)
typedef struct {
  float    *q_f, *k_f, *v_f, *g_f, *ao_f;
  uint16_t *qp_b, *kp_b, *vp_b, *ao_b;
} ltx_attn_ws_t;

static void ltx_run_attention(
    ec_ltx2_t *eng,
    const ltx_lin_t *lq, const ltx_lin_t *lk, const ltx_lin_t *lv, const ltx_lin_t *lg, const ltx_lin_t *lo,
    const float *qn_w, const float *kn_w,
    const uint16_t *x_q_bf16, const uint16_t *x_kv_bf16,
    int Lq, int Lkv, int H, int D,
    const float *q_cos, const float *q_sin, const float *k_cos, const float *k_sin,
    ltx_attn_ws_t *ws, float *out_f32) {
  int Lq_pad  = ((Lq  + 63) / 64) * 64;
  int Lkv_pad = ((Lkv + 63) / 64) * 64;
  int inner = H * D;
  float scale = 1.0f / sqrtf((float)D);

  ltx_run_lin(eng, &eng->main_st, lq, x_q_bf16,  ws->q_f, NULL, Lq);
  ltx_run_lin(eng, &eng->main_st, lk, x_kv_bf16, ws->k_f, NULL, Lkv);
  ltx_run_lin(eng, &eng->main_st, lv, x_kv_bf16, ws->v_f, NULL, Lkv);
  ltx_run_lin(eng, &eng->main_st, lg, x_q_bf16,  ws->g_f, NULL, Lq);

  k_ltx_qk_norm_rope_pack<<<Lq_pad,  256, 0, eng->stream>>>(ws->q_f, qn_w, q_cos, q_sin, ws->qp_b, Lq,  Lq_pad,  H, D, 1e-6f);
  k_ltx_qk_norm_rope_pack<<<Lkv_pad, 256, 0, eng->stream>>>(ws->k_f, kn_w, k_cos, k_sin, ws->kp_b, Lkv, Lkv_pad, H, D, 1e-6f);
  k_ltx_v_pack<<<((size_t)H * Lkv_pad * D + 255) / 256, 256, 0, eng->stream>>>(ws->v_f, ws->vp_b, Lkv, Lkv_pad, H, D);

  ec_ops_fa2_bf16(ws->qp_b, ws->kp_b, ws->vp_b, ws->ao_f, NULL, Lq, Lkv, Lq_pad, Lkv_pad, H, H, D, scale, 0, eng->stream);
  k_ltx_apply_head_gate_bf16<<<((size_t)Lq * inner + 255) / 256, 256, 0, eng->stream>>>(ws->ao_f, ws->g_f, ws->ao_b, Lq, H, D);
  ltx_run_lin(eng, &eng->main_st, lo, ws->ao_b, out_f32, NULL, Lq);
}

static int run_ltx2_dit_8step(
    ec_ltx2_t *eng,
    const float *video_ctx_d, const float *audio_ctx_d,
    int F_lat, int H_lat, int W_lat, int T_a, int fps,
    uint32_t seed, int steps,
    float *v_lat_d, float *a_lat_d,
    ec_ltx2_progress_cb_t cb, void *cb_ctx, double *out_ms) {
  double t0 = ltx_now_ms();
  if (ltx_ensure_dit_cache(eng) != 0) return -1;

  const int Nv = F_lat * H_lat * W_lat;
  const int Na = T_a;
  const int S_ctx = 1024;
  const int VD = 4096, AD = 2048;

  // Initialize deterministic Gaussian noise latents
  {
    float *vn_h = (float *)malloc((size_t)Nv * 128 * 4);
    float *an_h = (float *)malloc((size_t)Na * 128 * 4);
    ltx_fill_randn(vn_h, (size_t)Nv * 128, (uint64_t)seed);
    ltx_fill_randn(an_h, (size_t)Na * 128, (uint64_t)seed ^ 0xA0D10CAFEBEEFULL);
    cudaMemcpyAsync(v_lat_d, vn_h, (size_t)Nv * 128 * 4, cudaMemcpyHostToDevice, eng->stream);
    cudaMemcpyAsync(a_lat_d, an_h, (size_t)Na * 128 * 4, cudaMemcpyHostToDevice, eng->stream);
    cudaStreamSynchronize(eng->stream);
    free(vn_h); free(an_h);
  }

  // Precompute all 4 Split Float64 RoPE tables: v_pe (3D), a_pe (1D), v_cross_pe (1D temporal), a_cross_pe (= a_pe)
  float *v_cos_d = NULL, *v_sin_d = NULL;
  float *a_cos_d = NULL, *a_sin_d = NULL;
  float *vc_cos_d = NULL, *vc_sin_d = NULL;
  {
    // Video 3D pixel bounds with causal_fix=True, temporal axis divided by fps (units = seconds!)
    float *vpos = (float *)malloc((size_t)3 * Nv * 2 * 4);
    float *vtpos = (float *)malloc((size_t)1 * Nv * 2 * 4);
    int idx = 0;
    for (int f = 0; f < F_lat; f++) {
      float ts = fmaxf(0.0f, (float)(f * 8 + 1 - 8)) / (float)fps;
      float te = fmaxf(0.0f, (float)((f + 1) * 8 + 1 - 8)) / (float)fps;
      for (int h = 0; h < H_lat; h++) {
        float ys = (float)(h * 32), ye = (float)((h + 1) * 32);
        for (int w = 0; w < W_lat; w++) {
          float xs = (float)(w * 32), xe = (float)((w + 1) * 32);
          vpos[((size_t)0 * Nv + idx) * 2 + 0] = ts; vpos[((size_t)0 * Nv + idx) * 2 + 1] = te;
          vpos[((size_t)1 * Nv + idx) * 2 + 0] = ys; vpos[((size_t)1 * Nv + idx) * 2 + 1] = ye;
          vpos[((size_t)2 * Nv + idx) * 2 + 0] = xs; vpos[((size_t)2 * Nv + idx) * 2 + 1] = xe;
          vtpos[(size_t)idx * 2 + 0] = ts;           vtpos[(size_t)idx * 2 + 1] = te;
          idx++;
        }
      }
    }
    int vmax[3] = {20, 2048, 2048};
    int tmax[1] = {20};
    ltx_precompute_split_rope(vpos,  3, Nv, vmax, 1, 4096, 32, 10000.0, &v_cos_d,  &v_sin_d,  eng->stream);
    ltx_precompute_split_rope(vtpos, 1, Nv, tmax, 1, 2048, 32, 10000.0, &vc_cos_d, &vc_sin_d, eng->stream);
    free(vpos); free(vtpos);

    // Audio 1D causal timing bounds in seconds (patchifiers.py:232-278)
    float *apos = (float *)malloc((size_t)Na * 2 * 4);
    for (int t = 0; t < Na; t++) {
      float ts = fmaxf(0.0f, (float)(t * 4 + 1 - 4)) * (160.0f / 16000.0f);
      float te = fmaxf(0.0f, (float)((t + 1) * 4 + 1 - 4)) * (160.0f / 16000.0f);
      apos[(size_t)t * 2 + 0] = ts;
      apos[(size_t)t * 2 + 1] = te;
    }
    ltx_precompute_split_rope(apos, 1, Na, tmax, 1, 2048, 32, 10000.0, &a_cos_d, &a_sin_d, eng->stream);
    free(apos);
  }

  // Allocate reusable DiT step workspaces (fits comfortably inside 6.2 GB reserved headroom!)
  int maxL = Nv > S_ctx ? Nv : S_ctx;
  if (Na > maxL) maxL = Na;
  int maxL_pad = ((maxL + 63) / 64) * 64;

  float *vx_d = NULL, *ax_d = NULL, *vx_pre_d = NULL, *ax_pre_d = NULL;
  float *v_tmp_d = NULL, *a_tmp_d = NULL, *ff_f = NULL, *v_vel_d = NULL, *a_vel_d = NULL;
  uint16_t *v_in_b = NULL, *a_in_b = NULL, *v_n_b = NULL, *a_n_b = NULL, *ctx_mod_b = NULL, *ff_b = NULL;
  ltx_attn_ws_t aws = {};
  cudaMalloc(&vx_d,     (size_t)Nv * VD * 4);
  cudaMalloc(&vx_pre_d, (size_t)Nv * VD * 4);
  cudaMalloc(&v_tmp_d,  (size_t)Nv * VD * 4);
  cudaMalloc(&ax_d,     (size_t)Na * AD * 4);
  cudaMalloc(&ax_pre_d, (size_t)Na * AD * 4);
  cudaMalloc(&a_tmp_d,  (size_t)Na * AD * 4);
  cudaMalloc(&ff_f,     (size_t)Nv * (VD * 4) * 4);
  cudaMalloc(&ff_b,     (size_t)Nv * (VD * 4) * 2);
  cudaMalloc(&v_vel_d,  (size_t)Nv * 128 * 4);
  cudaMalloc(&a_vel_d,  (size_t)Na * 128 * 4);
  cudaMalloc(&v_in_b,   (size_t)Nv * 128 * 2);
  cudaMalloc(&a_in_b,   (size_t)Na * 128 * 2);
  cudaMalloc(&v_n_b,    (size_t)Nv * VD * 2);
  cudaMalloc(&a_n_b,    (size_t)Na * AD * 2);
  cudaMalloc(&ctx_mod_b,(size_t)S_ctx * VD * 2);

  cudaMalloc(&aws.q_f,  (size_t)maxL * VD * 4);
  cudaMalloc(&aws.k_f,  (size_t)maxL * VD * 4);
  cudaMalloc(&aws.v_f,  (size_t)maxL * VD * 4);
  cudaMalloc(&aws.g_f,  (size_t)maxL * 32 * 4);
  cudaMalloc(&aws.ao_f, (size_t)maxL * VD * 4);
  cudaMalloc(&aws.qp_b, (size_t)maxL_pad * VD * 2);
  cudaMalloc(&aws.kp_b, (size_t)maxL_pad * VD * 2);
  cudaMalloc(&aws.vp_b, (size_t)maxL_pad * VD * 2);
  cudaMalloc(&aws.ao_b, (size_t)maxL * VD * 2);

  // Per-step AdaLN modulation buffers + per-block summed vectors
  float *v_mod_d = NULL, *v_emb_d = NULL, *vp_mod_d = NULL;
  float *a_mod_d = NULL, *a_emb_d = NULL, *ap_mod_d = NULL;
  float *v_ca_ss_d = NULL, *v_ca_g_d = NULL, *a_ca_ss_d = NULL, *a_ca_g_d = NULL;
  float *ada_v_d = NULL, *ada_a_d = NULL, *ada_p_d = NULL, *ada_cv_d = NULL, *ada_ca_d = NULL;
  cudaMalloc(&v_mod_d,   9 * VD * 4); cudaMalloc(&v_emb_d,  VD * 4); cudaMalloc(&vp_mod_d, 2 * VD * 4);
  cudaMalloc(&a_mod_d,   9 * AD * 4); cudaMalloc(&a_emb_d,  AD * 4); cudaMalloc(&ap_mod_d, 2 * AD * 4);
  cudaMalloc(&v_ca_ss_d, 4 * VD * 4); cudaMalloc(&v_ca_g_d, VD * 4);
  cudaMalloc(&a_ca_ss_d, 4 * AD * 4); cudaMalloc(&a_ca_g_d, AD * 4);
  cudaMalloc(&ada_v_d,   9 * VD * 4); cudaMalloc(&ada_a_d,  9 * AD * 4); cudaMalloc(&ada_p_d, 2 * VD * 4);
  cudaMalloc(&ada_cv_d,  5 * VD * 4); cudaMalloc(&ada_ca_d, 5 * AD * 4);

  ltx_lin_t l_vpatch, l_apatch, l_vout, l_aout;
  ltx_find_lin(&eng->main_st, "model.diffusion_model.patchify_proj",       &l_vpatch);
  ltx_find_lin(&eng->main_st, "model.diffusion_model.audio_patchify_proj", &l_apatch);
  ltx_find_lin(&eng->main_st, "model.diffusion_model.proj_out",            &l_vout);
  ltx_find_lin(&eng->main_st, "model.diffusion_model.audio_proj_out",      &l_aout);
  float *out_sst_v = ltx_upload_vec_f32(&eng->main_st, "model.diffusion_model.scale_shift_table", eng->stream);
  float *out_sst_a = ltx_upload_vec_f32(&eng->main_st, "model.diffusion_model.audio_scale_shift_table", eng->stream);

  // Official LTX-2.3 Distilled 8-Step Schedule (constants.py:17, cfg=1.0, deterministic Euler!)
  static const float DISTILLED_SIGMAS[9] = {
    1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f
  };
  int n_steps = (steps > 0 && steps <= 8) ? steps : 8;
  float sig_sched[9];
  if (n_steps == 8) {
    memcpy(sig_sched, DISTILLED_SIGMAS, sizeof sig_sched);
  } else {
    for (int i = 0; i <= n_steps; i++) {
      float pos = (float)i * 8.0f / (float)n_steps;
      int i0 = (int)pos; if (i0 >= 8) i0 = 8;
      int i1 = (i0 < 8) ? i0 + 1 : 8;
      float w = pos - (float)i0;
      sig_sched[i] = DISTILLED_SIGMAS[i0] * (1.0f - w) + DISTILLED_SIGMAS[i1] * w;
    }
  }

  for (int st = 0; st < n_steps; st++) {
    if (cb && cb(st, n_steps, "ltx2_dit", cb_ctx) != 0) {
      fprintf(stderr, "[ltx2] Cancelled at step %d\n", st);
      break;
    }
    float sig_cur = sig_sched[st];
    float sig_nxt = sig_sched[st + 1];
    float t_scaled = sig_cur * 1000.0f;

    // 1. Evaluate all AdaLN-Single timestep MLPs
    ltx_eval_adaln_single(eng, "model.diffusion_model.adaln_single",                        t_scaled, VD, 9, v_mod_d,   v_emb_d);
    ltx_eval_adaln_single(eng, "model.diffusion_model.prompt_adaln_single",                 t_scaled, VD, 2, vp_mod_d,  NULL);
    ltx_eval_adaln_single(eng, "model.diffusion_model.audio_adaln_single",                  t_scaled, AD, 9, a_mod_d,   a_emb_d);
    ltx_eval_adaln_single(eng, "model.diffusion_model.audio_prompt_adaln_single",           t_scaled, AD, 2, ap_mod_d,  NULL);
    ltx_eval_adaln_single(eng, "model.diffusion_model.av_ca_video_scale_shift_adaln_single",t_scaled, VD, 4, v_ca_ss_d, NULL);
    ltx_eval_adaln_single(eng, "model.diffusion_model.av_ca_a2v_gate_adaln_single",         t_scaled, VD, 1, v_ca_g_d,  NULL);
    ltx_eval_adaln_single(eng, "model.diffusion_model.av_ca_audio_scale_shift_adaln_single",t_scaled, AD, 4, a_ca_ss_d, NULL);
    ltx_eval_adaln_single(eng, "model.diffusion_model.av_ca_v2a_gate_adaln_single",         t_scaled, AD, 1, a_ca_g_d,  NULL);

    // 2. Patchify input projections: [Nv, 128] -> [Nv, 4096], [Na, 128] -> [Na, 2048]
    k_f32_to_bf16<<<((size_t)Nv * 128 + 255) / 256, 256, 0, eng->stream>>>(v_lat_d, v_in_b, (size_t)Nv * 128);
    k_f32_to_bf16<<<((size_t)Na * 128 + 255) / 256, 256, 0, eng->stream>>>(a_lat_d, a_in_b, (size_t)Na * 128);
    ltx_run_lin(eng, &eng->main_st, &l_vpatch, v_in_b, vx_d, NULL, Nv);
    ltx_run_lin(eng, &eng->main_st, &l_apatch, a_in_b, ax_d, NULL, Na);

    // 3. 48 Dual-Stream BasicAVTransformerBlocks!
    for (int b = 0; b < 48; b++) {
      const ltx_dit_block_t *blk = &eng->blocks[b];

      // --- Video Self-Attn + Cross-Attn AdaLN (shift=0, scale=1, gate=2; ca_shift=6, ca_scale=7, ca_gate=8) ---
      k_vec_add2<<<(9 * VD + 255) / 256, 256, 0, eng->stream>>>(blk->sst_v, v_mod_d, ada_v_d, 9 * VD);
      k_ltx_rmsnorm<<<Nv, 256, 0, eng->stream>>>(vx_d, NULL, ada_v_d + 1 * VD, ada_v_d + 0 * VD, NULL, v_n_b, Nv, VD, 1e-6f);
      ltx_run_attention(eng, &blk->a1_q, &blk->a1_k, &blk->a1_v, &blk->a1_g, &blk->a1_o,
                        blk->a1_qn, blk->a1_kn, v_n_b, v_n_b, Nv, Nv, 32, 128,
                        v_cos_d, v_sin_d, v_cos_d, v_sin_d, &aws, v_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Nv * VD + 255) / 256, 256, 0, eng->stream>>>(vx_d, v_tmp_d, ada_v_d + 2 * VD, Nv, VD);

      // Video Text Cross-Attention with Cross-Attention AdaLN (transformer.py:188-203, 398-404)
      k_ltx_rmsnorm<<<Nv, 256, 0, eng->stream>>>(vx_d, NULL, ada_v_d + 7 * VD, ada_v_d + 6 * VD, NULL, v_n_b, Nv, VD, 1e-6f);
      k_vec_add2<<<(2 * VD + 255) / 256, 256, 0, eng->stream>>>(blk->psst_v, vp_mod_d, ada_p_d, 2 * VD);
      k_ltx_modulate_bf16<<<((size_t)S_ctx * VD + 255) / 256, 256, 0, eng->stream>>>(
          video_ctx_d, ada_p_d + 1 * VD, ada_p_d + 0 * VD, ctx_mod_b, S_ctx, VD);
      ltx_run_attention(eng, &blk->a2_q, &blk->a2_k, &blk->a2_v, &blk->a2_g, &blk->a2_o,
                        blk->a2_qn, blk->a2_kn, v_n_b, ctx_mod_b, Nv, S_ctx, 32, 128,
                        NULL, NULL, NULL, NULL, &aws, v_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Nv * VD + 255) / 256, 256, 0, eng->stream>>>(vx_d, v_tmp_d, ada_v_d + 8 * VD, Nv, VD);

      // --- Audio Self-Attn + Cross-Attn AdaLN ---
      k_vec_add2<<<(9 * AD + 255) / 256, 256, 0, eng->stream>>>(blk->sst_a, a_mod_d, ada_a_d, 9 * AD);
      k_ltx_rmsnorm<<<Na, 256, 0, eng->stream>>>(ax_d, NULL, ada_a_d + 1 * AD, ada_a_d + 0 * AD, NULL, a_n_b, Na, AD, 1e-6f);
      ltx_run_attention(eng, &blk->aa1_q, &blk->aa1_k, &blk->aa1_v, &blk->aa1_g, &blk->aa1_o,
                        blk->aa1_qn, blk->aa1_kn, a_n_b, a_n_b, Na, Na, 32, 64,
                        a_cos_d, a_sin_d, a_cos_d, a_sin_d, &aws, a_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Na * AD + 255) / 256, 256, 0, eng->stream>>>(ax_d, a_tmp_d, ada_a_d + 2 * AD, Na, AD);

      k_ltx_rmsnorm<<<Na, 256, 0, eng->stream>>>(ax_d, NULL, ada_a_d + 7 * AD, ada_a_d + 6 * AD, NULL, a_n_b, Na, AD, 1e-6f);
      k_vec_add2<<<(2 * AD + 255) / 256, 256, 0, eng->stream>>>(blk->psst_a, ap_mod_d, ada_p_d, 2 * AD);
      k_ltx_modulate_bf16<<<((size_t)S_ctx * AD + 255) / 256, 256, 0, eng->stream>>>(
          audio_ctx_d, ada_p_d + 1 * AD, ada_p_d + 0 * AD, ctx_mod_b, S_ctx, AD);
      ltx_run_attention(eng, &blk->aa2_q, &blk->aa2_k, &blk->aa2_v, &blk->aa2_g, &blk->aa2_o,
                        blk->aa2_qn, blk->aa2_kn, a_n_b, ctx_mod_b, Na, S_ctx, 32, 64,
                        NULL, NULL, NULL, NULL, &aws, a_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Na * AD + 255) / 256, 256, 0, eng->stream>>>(ax_d, a_tmp_d, ada_a_d + 8 * AD, Na, AD);

      // --- Bidirectional Audio-Video Cross-Attention (snapshot BEFORE A2V mutates vx!) ---
      // CRITICAL: scale_shift_table_a2v_ca_* ordering is [0=scale_a2v, 1=shift_a2v, 2=scale_v2a, 3=shift_v2a, 4=gate]!
      cudaMemcpyAsync(vx_pre_d, vx_d, (size_t)Nv * VD * 4, cudaMemcpyDeviceToDevice, eng->stream);
      cudaMemcpyAsync(ax_pre_d, ax_d, (size_t)Na * AD * 4, cudaMemcpyDeviceToDevice, eng->stream);
      k_vec_add2<<<(4 * VD + 255) / 256, 256, 0, eng->stream>>>(blk->ca_sst_v, v_ca_ss_d, ada_cv_d, 4 * VD);
      k_vec_add2<<<(1 * VD + 255) / 256, 256, 0, eng->stream>>>(blk->ca_sst_v + 4 * VD, v_ca_g_d, ada_cv_d + 4 * VD, VD);
      k_vec_add2<<<(4 * AD + 255) / 256, 256, 0, eng->stream>>>(blk->ca_sst_a, a_ca_ss_d, ada_ca_d, 4 * AD);
      k_vec_add2<<<(1 * AD + 255) / 256, 256, 0, eng->stream>>>(blk->ca_sst_a + 4 * AD, a_ca_g_d, ada_ca_d + 4 * AD, AD);

      // A2V: Q = Video (4096->2048), K,V = Audio (2048->2048), Out -> 4096
      k_ltx_rmsnorm<<<Nv, 256, 0, eng->stream>>>(vx_pre_d, NULL, ada_cv_d + 0 * VD, ada_cv_d + 1 * VD, NULL, v_n_b, Nv, VD, 1e-6f);
      k_ltx_rmsnorm<<<Na, 256, 0, eng->stream>>>(ax_pre_d, NULL, ada_ca_d + 0 * AD, ada_ca_d + 1 * AD, NULL, a_n_b, Na, AD, 1e-6f);
      ltx_run_attention(eng, &blk->a2v_q, &blk->a2v_k, &blk->a2v_v, &blk->a2v_g, &blk->a2v_o,
                        blk->a2v_qn, blk->a2v_kn, v_n_b, a_n_b, Nv, Na, 32, 64,
                        vc_cos_d, vc_sin_d, a_cos_d, a_sin_d, &aws, v_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Nv * VD + 255) / 256, 256, 0, eng->stream>>>(vx_d, v_tmp_d, ada_cv_d + 4 * VD, Nv, VD);

      // V2A: Q = Audio (2048->2048), K,V = Video (4096->2048), Out -> 2048
      k_ltx_rmsnorm<<<Na, 256, 0, eng->stream>>>(ax_pre_d, NULL, ada_ca_d + 2 * AD, ada_ca_d + 3 * AD, NULL, a_n_b, Na, AD, 1e-6f);
      k_ltx_rmsnorm<<<Nv, 256, 0, eng->stream>>>(vx_pre_d, NULL, ada_cv_d + 2 * VD, ada_cv_d + 3 * VD, NULL, v_n_b, Nv, VD, 1e-6f);
      ltx_run_attention(eng, &blk->v2a_q, &blk->v2a_k, &blk->v2a_v, &blk->v2a_g, &blk->v2a_o,
                        blk->v2a_qn, blk->v2a_kn, a_n_b, v_n_b, Na, Nv, 32, 64,
                        a_cos_d, a_sin_d, vc_cos_d, vc_sin_d, &aws, a_tmp_d);
      k_ltx_res_add_gate<<<((size_t)Na * AD + 255) / 256, 256, 0, eng->stream>>>(ax_d, a_tmp_d, ada_ca_d + 4 * AD, Na, AD);

      // --- Video & Audio Feed-Forward (shift=3, scale=4, gate=5) ---
      k_ltx_rmsnorm<<<Nv, 256, 0, eng->stream>>>(vx_d, NULL, ada_v_d + 4 * VD, ada_v_d + 3 * VD, NULL, v_n_b, Nv, VD, 1e-6f);
      ltx_run_lin(eng, &eng->main_st, &blk->ff_0, v_n_b, ff_f, NULL, Nv);
      k_gelu_tanh_to_bf16<<<((size_t)Nv * VD * 4 + 255) / 256, 256, 0, eng->stream>>>(ff_f, ff_b, (size_t)Nv * VD * 4);
      ltx_run_lin(eng, &eng->main_st, &blk->ff_2, ff_b, v_tmp_d, NULL, Nv);
      k_ltx_res_add_gate<<<((size_t)Nv * VD + 255) / 256, 256, 0, eng->stream>>>(vx_d, v_tmp_d, ada_v_d + 5 * VD, Nv, VD);

      k_ltx_rmsnorm<<<Na, 256, 0, eng->stream>>>(ax_d, NULL, ada_a_d + 4 * AD, ada_a_d + 3 * AD, NULL, a_n_b, Na, AD, 1e-6f);
      ltx_run_lin(eng, &eng->main_st, &blk->aff_0, a_n_b, ff_f, NULL, Na);
      k_gelu_tanh_to_bf16<<<((size_t)Na * AD * 4 + 255) / 256, 256, 0, eng->stream>>>(ff_f, ff_b, (size_t)Na * AD * 4);
      ltx_run_lin(eng, &eng->main_st, &blk->aff_2, ff_b, a_tmp_d, NULL, Na);
      k_ltx_res_add_gate<<<((size_t)Na * AD + 255) / 256, 256, 0, eng->stream>>>(ax_d, a_tmp_d, ada_a_d + 5 * AD, Na, AD);
    }

    // 4. Final output head: affine-free LayerNorm * (1 + scale) + shift -> proj_out
    k_vec_add2<<<(VD + 255) / 256, 256, 0, eng->stream>>>(out_sst_v + 0 * VD, v_emb_d, ada_p_d + 0 * VD, VD);
    k_vec_add2<<<(VD + 255) / 256, 256, 0, eng->stream>>>(out_sst_v + 1 * VD, v_emb_d, ada_p_d + 1 * VD, VD);
    k_ltx_layernorm_adaln<<<Nv, 256, 0, eng->stream>>>(vx_d, ada_p_d + 1 * VD, ada_p_d + 0 * VD, v_n_b, Nv, VD, 1e-6f);
    ltx_run_lin(eng, &eng->main_st, &l_vout, v_n_b, v_vel_d, NULL, Nv);

    k_vec_add2<<<(AD + 255) / 256, 256, 0, eng->stream>>>(out_sst_a + 0 * AD, a_emb_d, ada_p_d + 0 * AD, AD);
    k_vec_add2<<<(AD + 255) / 256, 256, 0, eng->stream>>>(out_sst_a + 1 * AD, a_emb_d, ada_p_d + 1 * AD, AD);
    k_ltx_layernorm_adaln<<<Na, 256, 0, eng->stream>>>(ax_d, ada_p_d + 1 * AD, ada_p_d + 0 * AD, a_n_b, Na, AD, 1e-6f);
    ltx_run_lin(eng, &eng->main_st, &l_aout, a_n_b, a_vel_d, NULL, Na);

    // 5. Deterministic Euler update: x_next = x + (sigma_next - sigma_cur) * v_pred
    float dt = sig_nxt - sig_cur;
    k_f32_euler_step<<<((size_t)Nv * 128 + 255) / 256, 256, 0, eng->stream>>>(v_lat_d, v_vel_d, dt, (size_t)Nv * 128);
    k_f32_euler_step<<<((size_t)Na * 128 + 255) / 256, 256, 0, eng->stream>>>(a_lat_d, a_vel_d, dt, (size_t)Na * 128);
    cudaError_t ce = cudaStreamSynchronize(eng->stream);
    if (ce != cudaSuccess) {
      fprintf(stderr, "[ltx2] CUDA ERROR at DiT step %d: %s\n", st + 1, cudaGetErrorString(ce));
      break;
    }
    fprintf(stderr, "[ltx2] DiT step %d/%d sigma=%.5f->%.5f elapsed=%.1fs\n",
            st + 1, n_steps, sig_cur, sig_nxt, (ltx_now_ms() - t0) / 1000.0);
  }

  cudaFree(out_sst_v); cudaFree(out_sst_a);
  cudaFree(v_cos_d); cudaFree(v_sin_d); cudaFree(a_cos_d); cudaFree(a_sin_d); cudaFree(vc_cos_d); cudaFree(vc_sin_d);
  cudaFree(vx_d); cudaFree(vx_pre_d); cudaFree(v_tmp_d); cudaFree(ax_d); cudaFree(ax_pre_d); cudaFree(a_tmp_d);
  cudaFree(ff_f); cudaFree(ff_b); cudaFree(v_vel_d); cudaFree(a_vel_d);
  cudaFree(v_in_b); cudaFree(a_in_b); cudaFree(v_n_b); cudaFree(a_n_b); cudaFree(ctx_mod_b);
  cudaFree(aws.q_f); cudaFree(aws.k_f); cudaFree(aws.v_f); cudaFree(aws.g_f); cudaFree(aws.ao_f);
  cudaFree(aws.qp_b); cudaFree(aws.kp_b); cudaFree(aws.vp_b); cudaFree(aws.ao_b);
  cudaFree(v_mod_d); cudaFree(v_emb_d); cudaFree(vp_mod_d); cudaFree(a_mod_d); cudaFree(a_emb_d); cudaFree(ap_mod_d);
  cudaFree(v_ca_ss_d); cudaFree(v_ca_g_d); cudaFree(a_ca_ss_d); cudaFree(a_ca_g_d);
  cudaFree(ada_v_d); cudaFree(ada_a_d); cudaFree(ada_p_d); cudaFree(ada_cv_d); cudaFree(ada_ca_d);

  if (out_ms) *out_ms = ltx_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 6. Stage 3: Non-Causal 3D Conv Video Decoder (vae.decoder.* -> RGB24)
// ===========================================================================

__global__ void k_vvae_unnorm_bf16(
    const float *__restrict__ lat, const float *__restrict__ std_c, const float *__restrict__ mean_c,
    uint16_t *__restrict__ dst_bf16, int Nv, int C) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)Nv * C;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  dst_bf16[idx] = d_f32_to_bf16(lat[idx] * std_c[c] + mean_c[c]);
}

// PixelNorm (eps=1e-8) + SiLU on channel-last [N, C] BF16 -> BF16
__global__ void k_pixelnorm_silu_bf16(const uint16_t *__restrict__ x, uint16_t *__restrict__ y, int N, int C) {
  int r = blockIdx.x;
  if (r >= N) return;
  const uint16_t *xr = x + (size_t)r * C;
  uint16_t *yr = y + (size_t)r * C;
  __shared__ float sm[128];
  float s = 0.0f;
  for (int c = threadIdx.x; c < C; c += 128) {
    float v = d_bf16_to_f32(xr[c]);
    s += v * v;
  }
  sm[threadIdx.x] = s;
  __syncthreads();
  for (int k = 64; k > 0; k >>= 1) { if (threadIdx.x < k) sm[threadIdx.x] += sm[threadIdx.x + k]; __syncthreads(); }
  float inv = rsqrtf(sm[0] / (float)C + 1e-8f);
  for (int c = threadIdx.x; c < C; c += 128) {
    float v = d_bf16_to_f32(xr[c]) * inv;
    yr[c] = d_f32_to_bf16(d_silu(v));
  }
}

// Non-causal 3x3x3 im2col for 1 output temporal slice f:
// Temporal padding: edge-frame replication (front=1, back=1); Spatial padding: zeros (pad=1)!
__global__ void k_conv3d_im2col_frame_bf16(
    const uint16_t *__restrict__ x, uint16_t *__restrict__ col,
    int F, int H, int W, int Cin, int f) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t col_stride = (size_t)27 * Cin;
  size_t tot = (size_t)H * W * col_stride;
  if (idx >= tot) return;
  int rem = (int)(idx % col_stride);
  int hw  = (int)(idx / col_stride);
  int hy = hw / W, wx = hw % W;
  int tap = rem / Cin;
  int ci  = rem % Cin;
  int kt = tap / 9;
  int rem9 = tap % 9;
  int ky = rem9 / 3;
  int kx = rem9 % 3;
  int sf = f + kt - 1;
  if (sf < 0) sf = 0;
  if (sf >= F) sf = F - 1;
  int sy = hy + ky - 1;
  int sx = wx + kx - 1;
  if (sy < 0 || sy >= H || sx < 0 || sx >= W) {
    col[idx] = 0;
  } else {
    col[idx] = x[(((size_t)sf * H + sy) * W + sx) * Cin + ci];
  }
}

__global__ void k_bf16_add_inplace(uint16_t *__restrict__ dst, const uint16_t *__restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = d_f32_to_bf16(d_bf16_to_f32(dst[i]) + d_bf16_to_f32(src[i]));
}

// DepthToSpaceUpsample (sampling.py:116-124):
// in: [F, H, W, Cout * p1 * p2 * p3] -> out: [F_out, H*p2, W*p3, Cout]
// Channel order: ((co * p1 + dt) * p2 + dh) * p3 + dw; if p1==2, drop global time index 0!
__global__ void k_d2s_3d_bf16(
    const uint16_t *__restrict__ src, uint16_t *__restrict__ dst,
    int F, int H, int W, int Cout, int p1, int p2, int p3) {
  int F_out = (p1 == 2) ? (F * 2 - 1) : F;
  int H_out = H * p2;
  int W_out = W * p3;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)F_out * H_out * W_out * Cout;
  if (idx >= tot) return;
  int co = (int)(idx % Cout);
  size_t rem = idx / Cout;
  int ox = (int)(rem % W_out); rem /= W_out;
  int oy = (int)(rem % H_out); rem /= H_out;
  int ot = (int)rem;
  int gt = (p1 == 2) ? (ot + 1) : ot;
  int it = gt / p1, dt = gt % p1;
  int iy = oy / p2, dh = oy % p2;
  int ix = ox / p3, dw = ox % p3;
  int Cin = Cout * p1 * p2 * p3;
  int ic = ((co * p1 + dt) * p2 + dh) * p3 + dw;
  dst[idx] = src[(((size_t)it * H + iy) * W + ix) * Cin + ic];
}

// Exact LTX unpatchify(patch_size_hw=4, patch_size_t=1) -> RGB24 [F, H*4, W*4, 3]
// ops.py:69: "b (c p r q) f h w -> b c (f p) (h q) (w r)" -> ic = (rgb * 4 + dw) * 4 + dh!
__global__ void k_vvae_unpatchify_rgb24(
    const uint16_t *__restrict__ src, uint8_t *__restrict__ dst_rgb,
    int F, int H, int W) {
  int H4 = H * 4, W4 = W * 4;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)F * H4 * W4 * 3;
  if (idx >= tot) return;
  int rgb = (int)(idx % 3);
  size_t rem = idx / 3;
  int ox = (int)(rem % W4); rem /= W4;
  int oy = (int)(rem % H4); rem /= H4;
  int f  = (int)rem;
  int iy = oy >> 2, dh = oy & 3;
  int ix = ox >> 2, dw = ox & 3;
  int ic = (rgb * 4 + dw) * 4 + dh;
  float v = d_bf16_to_f32(src[(((size_t)f * H + iy) * W + ix) * 48 + ic]);
  v = (v + 1.0f) * 0.5f;
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
  dst_rgb[idx] = (uint8_t)lrintf(v * 255.0f);
}

static int ltx_vvae_conv3d(
    ec_ltx2_t *eng, const char *prefix,
    const uint16_t *x_d, uint16_t *y_d,
    int F, int H, int W, int Cin, int Cout,
    uint16_t *col_scratch_d) {
  char wn[192], bn[192];
  snprintf(wn, sizeof wn, "%s.conv.weight", prefix);
  snprintf(bn, sizeof bn, "%s.conv.bias", prefix);
  const ec_st_entry_t *we = ec_st_find(&eng->main_st, wn);
  if (!we || we->rank != 5) { fprintf(stderr, "[ltx2] Missing %s\n", wn); return -1; }

  // Permute [Cout, Cin, 3, 3, 3] -> [Cout, 27*Cin] BF16 on host and upload once!
  size_t k_dim = (size_t)27 * Cin;
  uint16_t *wh = (uint16_t *)malloc((size_t)Cout * k_dim * 2);
  const uint16_t *ws = (const uint16_t *)ec_st_ptr(&eng->main_st, we);
  for (int co = 0; co < Cout; co++) {
    for (int ci = 0; ci < Cin; ci++) {
      const uint16_t *s27 = ws + ((size_t)co * Cin + ci) * 27;
      for (int tap = 0; tap < 27; tap++) {
        wh[(size_t)co * k_dim + (size_t)tap * Cin + ci] = s27[tap];
      }
    }
  }
  uint16_t *wd = NULL;
  cudaMalloc(&wd, (size_t)Cout * k_dim * 2);
  cudaMemcpyAsync(wd, wh, (size_t)Cout * k_dim * 2, cudaMemcpyHostToDevice, eng->stream);
  float *bd = ltx_upload_vec_f32(&eng->main_st, bn, eng->stream);
  cudaStreamSynchronize(eng->stream);
  free(wh);

  int HW = H * W;
  size_t col_elems = (size_t)HW * k_dim;
  for (int f = 0; f < F; f++) {
    k_conv3d_im2col_frame_bf16<<<(col_elems + 255) / 256, 256, 0, eng->stream>>>(x_d, col_scratch_d, F, H, W, Cin, f);
    ec_ops_gemm_bf16(col_scratch_d, wd, bd, NULL, y_d + (size_t)f * HW * Cout, HW, Cout, (int)k_dim, eng->stream);
  }
  cudaStreamSynchronize(eng->stream);
  cudaFree(wd);
  if (bd) cudaFree(bd);
  return 0;
}

static int run_ltx2_video_decoder(
    ec_ltx2_t *eng, const float *v_lat_d, int F_lat, int H_lat, int W_lat,
    uint8_t **out_rgb24_h, int *out_F, int *out_H, int *out_W, double *out_ms) {
  double t0 = ltx_now_ms();
  int F = F_lat, H = H_lat, W = W_lat, C = 128;

  // Peak 1-frame im2col buffer: max over stages of H*W*27*Cin*2 bytes
  // At 768x512: stage 8 has H=128, W=192, Cin=128 -> 128*192*27*128*2 = 170 MB;
  // stage 6 has H=64, W=96, Cin=256 -> 85 MB; stage 4 has H=64, W=96, Cin=512 -> 170 MB.
  size_t max_col = (size_t)(H_lat * 8) * (W_lat * 8) * 27 * 128;
  size_t alt_col = (size_t)(H_lat * 4) * (W_lat * 4) * 27 * 512;
  if (alt_col > max_col) max_col = alt_col;
  uint16_t *col_d = NULL;
  if (cudaMalloc(&col_d, max_col * 2) != cudaSuccess) return -1;

  // Un-normalize latents via vae.per_channel_statistics
  float *std_d  = ltx_upload_vec_f32(&eng->main_st, "vae.per_channel_statistics.std-of-means", eng->stream);
  float *mean_d = ltx_upload_vec_f32(&eng->main_st, "vae.per_channel_statistics.mean-of-means", eng->stream);
  uint16_t *cur_d = NULL;
  cudaMalloc(&cur_d, (size_t)F * H * W * 128 * 2);
  k_vvae_unnorm_bf16<<<((size_t)F * H * W * 128 + 255) / 256, 256, 0, eng->stream>>>(v_lat_d, std_d, mean_d, cur_d, F * H * W, 128);
  cudaStreamSynchronize(eng->stream);
  cudaFree(std_d); cudaFree(mean_d);

  // conv_in: 128 -> 1024
  uint16_t *nxt_d = NULL;
  cudaMalloc(&nxt_d, (size_t)F * H * W * 1024 * 2);
  ltx_vvae_conv3d(eng, "vae.decoder.conv_in", cur_d, nxt_d, F, H, W, 128, 1024, col_d);
  cudaFree(cur_d); cur_d = nxt_d; C = 1024;

  auto run_res_group = [&](int ub_idx, int n_res) {
    uint16_t *h1 = NULL, *h2 = NULL;
    size_t n_vox = (size_t)F * H * W;
    cudaMalloc(&h1, n_vox * C * 2);
    cudaMalloc(&h2, n_vox * C * 2);
    for (int r = 0; r < n_res; r++) {
      char p1[128], p2[128];
      snprintf(p1, sizeof p1, "vae.decoder.up_blocks.%d.res_blocks.%d.conv1", ub_idx, r);
      snprintf(p2, sizeof p2, "vae.decoder.up_blocks.%d.res_blocks.%d.conv2", ub_idx, r);
      k_pixelnorm_silu_bf16<<<(int)n_vox, 128, 0, eng->stream>>>(cur_d, h1, (int)n_vox, C);
      ltx_vvae_conv3d(eng, p1, h1, h2, F, H, W, C, C, col_d);
      k_pixelnorm_silu_bf16<<<(int)n_vox, 128, 0, eng->stream>>>(h2, h1, (int)n_vox, C);
      ltx_vvae_conv3d(eng, p2, h1, h2, F, H, W, C, C, col_d);
      k_bf16_add_inplace<<<(n_vox * C + 255) / 256, 256, 0, eng->stream>>>(cur_d, h2, n_vox * C);
    }
    cudaStreamSynchronize(eng->stream);
    cudaFree(h1); cudaFree(h2);
  };

  auto run_d2s_up = [&](int ub_idx, int p1, int p2, int p3, int mult) {
    int c_conv = C * (p1 * p2 * p3) / mult;
    int c_out  = C / mult;
    int F_out  = (p1 == 2) ? (F * 2 - 1) : F;
    int H_out  = H * p2;
    int W_out  = W * p3;
    char pn[128];
    snprintf(pn, sizeof pn, "vae.decoder.up_blocks.%d.conv", ub_idx);
    uint16_t *conv_o = NULL, *up_o = NULL;
    cudaMalloc(&conv_o, (size_t)F * H * W * c_conv * 2);
    ltx_vvae_conv3d(eng, pn, cur_d, conv_o, F, H, W, C, c_conv, col_d);
    cudaFree(cur_d);
    cudaMalloc(&up_o, (size_t)F_out * H_out * W_out * c_out * 2);
    size_t tot = (size_t)F_out * H_out * W_out * c_out;
    k_d2s_3d_bf16<<<(tot + 255) / 256, 256, 0, eng->stream>>>(conv_o, up_o, F, H, W, c_out, p1, p2, p3);
    cudaStreamSynchronize(eng->stream);
    cudaFree(conv_o);
    cur_d = up_o; F = F_out; H = H_out; W = W_out; C = c_out;
  };

  // Exact reversed(decoder_blocks) schedule for LTX-2.3!
  run_res_group(0, 2);                // 1024
  run_d2s_up(1, 2, 2, 2, 2);          // -> 512, 2x T,H,W
  run_res_group(2, 2);                // 512
  run_d2s_up(3, 2, 2, 2, 1);          // -> 512, 2x T,H,W
  run_res_group(4, 4);                // 512
  run_d2s_up(5, 2, 1, 1, 2);          // -> 256, 2x T
  run_res_group(6, 6);                // 256
  run_d2s_up(7, 1, 2, 2, 2);          // -> 128, 2x H,W
  run_res_group(8, 4);                // 128

  // Final PixelNorm -> SiLU -> conv_out (128 -> 48) -> unpatchify(4) -> RGB24!
  size_t n_vox = (size_t)F * H * W;
  uint16_t *norm_d = NULL, *out48_d = NULL;
  cudaMalloc(&norm_d,  n_vox * 128 * 2);
  cudaMalloc(&out48_d, n_vox * 48 * 2);
  k_pixelnorm_silu_bf16<<<(int)n_vox, 128, 0, eng->stream>>>(cur_d, norm_d, (int)n_vox, 128);
  cudaFree(cur_d);
  ltx_vvae_conv3d(eng, "vae.decoder.conv_out", norm_d, out48_d, F, H, W, 128, 48, col_d);
  cudaFree(norm_d); cudaFree(col_d);

  int H_px = H * 4, W_px = W * 4;
  size_t rgb_bytes = (size_t)F * H_px * W_px * 3;
  uint8_t *rgb_d = NULL;
  cudaMalloc(&rgb_d, rgb_bytes);
  k_vvae_unpatchify_rgb24<<<(rgb_bytes + 255) / 256, 256, 0, eng->stream>>>(out48_d, rgb_d, F, H, W);
  uint8_t *rgb_h = (uint8_t *)malloc(rgb_bytes);
  cudaMemcpyAsync(rgb_h, rgb_d, rgb_bytes, cudaMemcpyDeviceToHost, eng->stream);
  cudaStreamSynchronize(eng->stream);
  cudaFree(out48_d); cudaFree(rgb_d);

  *out_rgb24_h = rgb_h;
  *out_F = F; *out_H = H_px; *out_W = W_px;
  if (out_ms) *out_ms = ltx_now_ms() - t0;
  fprintf(stderr, "[ltx2] Video 3D VAE decoded %dx%dx%d RGB24 in %.1f ms\n", W_px, H_px, F, ltx_now_ms() - t0);
  return 0;
}

// ===========================================================================
// 7. Stage 4: Causal 2D Audio Mel Decoder + Pure IEEE FP32 BigVGAN v2 AMP1 + BWE
// ===========================================================================

// Un-normalize [T_a, 128] and unpatchify (c=8, f=16) -> channel-last [T_a, 16, 8] BF16
__global__ void k_avae_unnorm_unpatch(
    const float *__restrict__ lat, const float *__restrict__ std_c, const float *__restrict__ mean_c,
    uint16_t *__restrict__ dst_tfc, int T_a) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)T_a * 128;
  if (idx >= tot) return;
  int t = (int)(idx / 128);
  int cf = (int)(idx % 128);
  int c = cf / 16;
  int f = cf % 16;
  float v = lat[idx] * std_c[cf] + mean_c[cf];
  dst_tfc[((size_t)t * 16 + f) * 8 + c] = d_f32_to_bf16(v);
}

// Causal 2D im2col (causality_axis=height=time: pad_top=2, pad_bot=0, pad_lr=1, zero-padded!)
__global__ void k_avae_causal_im2col(
    const uint16_t *__restrict__ x, uint16_t *__restrict__ col,
    int T, int F, int Cin) {
  size_t col_s = (size_t)9 * Cin;
  size_t tot = (size_t)T * F * col_s;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  int rem = (int)(idx % col_s);
  int tf  = (int)(idx / col_s);
  int t = tf / F, f = tf % F;
  int tap = rem / Cin, ci = rem % Cin;
  int kt = tap / 3, kf = tap % 3;
  int st = t + kt - 2; // causal along time: pad_top=2, pad_bot=0 -> kernel looks at t-2, t-1, t!
  int sf = f + kf - 1; // symmetric along freq: pad_l=1, pad_r=1
  if (st < 0 || st >= T || sf < 0 || sf >= F) col[idx] = 0;
  else col[idx] = x[((size_t)st * F + sf) * Cin + ci];
}

__global__ void k_avae_upsample2x_nearest(
    const uint16_t *__restrict__ x, uint16_t *__restrict__ y, int T, int F, int C) {
  int T2 = T * 2, F2 = F * 2;
  size_t tot = (size_t)T2 * F2 * C;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  size_t rem = idx / C;
  int f = (int)(rem % F2), t = (int)(rem / F2);
  y[idx] = x[((size_t)(t >> 1) * F + (f >> 1)) * C + c];
}

static int ltx_avae_conv2d(
    ec_ltx2_t *eng, const char *prefix, int is_1x1,
    const uint16_t *x_d, uint16_t *y_d, int T, int F, int Cin, int Cout) {
  char wn[192], bn[192];
  snprintf(wn, sizeof wn, "%s.conv.weight", prefix);
  snprintf(bn, sizeof bn, "%s.conv.bias", prefix);
  const ec_st_entry_t *we = ec_st_find(&eng->main_st, wn);
  if (!we) {
    snprintf(wn, sizeof wn, "%s.weight", prefix);
    snprintf(bn, sizeof bn, "%s.bias", prefix);
    we = ec_st_find(&eng->main_st, wn);
  }
  if (!we) { fprintf(stderr, "[ltx2] Missing audio conv %s\n", prefix); return -1; }
  const uint16_t *ws = (const uint16_t *)ec_st_ptr(&eng->main_st, we);
  float *bd = ltx_upload_vec_f32(&eng->main_st, bn, eng->stream);

  int TF = T * F;
  if (is_1x1) {
    uint16_t *wd = NULL;
    cudaMalloc(&wd, (size_t)Cout * Cin * 2);
    cudaMemcpyAsync(wd, ws, (size_t)Cout * Cin * 2, cudaMemcpyHostToDevice, eng->stream);
    ec_ops_gemm_bf16(x_d, wd, bd, NULL, y_d, TF, Cout, Cin, eng->stream);
    cudaStreamSynchronize(eng->stream);
    cudaFree(wd);
  } else {
    size_t k_dim = (size_t)9 * Cin;
    uint16_t *wh = (uint16_t *)malloc((size_t)Cout * k_dim * 2);
    for (int co = 0; co < Cout; co++) {
      for (int ci = 0; ci < Cin; ci++) {
        const uint16_t *s9 = ws + ((size_t)co * Cin + ci) * 9;
        for (int tap = 0; tap < 9; tap++) wh[(size_t)co * k_dim + (size_t)tap * Cin + ci] = s9[tap];
      }
    }
    uint16_t *wd = NULL, *col = NULL;
    cudaMalloc(&wd, (size_t)Cout * k_dim * 2);
    cudaMalloc(&col, (size_t)TF * k_dim * 2);
    cudaMemcpyAsync(wd, wh, (size_t)Cout * k_dim * 2, cudaMemcpyHostToDevice, eng->stream);
    free(wh);
    k_avae_causal_im2col<<<((size_t)TF * k_dim + 255) / 256, 256, 0, eng->stream>>>(x_d, col, T, F, Cin);
    ec_ops_gemm_bf16(col, wd, bd, NULL, y_d, TF, Cout, (int)k_dim, eng->stream);
    cudaStreamSynchronize(eng->stream);
    cudaFree(wd); cudaFree(col);
  }
  if (bd) cudaFree(bd);
  return 0;
}

// --- Pure IEEE FP32 BigVGAN v2 AMP1 & BWE Kernels ---

__global__ void k_f32_transpose_2d(const float *__restrict__ src, float *__restrict__ dst, int R, int C) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)R * C;
  if (idx >= tot) return;
  int c = (int)(idx % C), r = (int)(idx / C);
  dst[(size_t)c * R + r] = src[idx];
}

// Replicate-padded 1D depthwise ConvTranspose1d for UpSample1d (vocoder.py:143-148):
// y_raw has length (L + 2*pad - 1)*ratio + K; we output y[pad_left .. y_raw_len - pad_right - 1] (length = L * ratio)
__global__ void k_voc_upsample1d_f32(
    const float *__restrict__ x, const float *__restrict__ filt, float *__restrict__ y,
    int C, int L, int ratio, int K, int pad, int pad_left) {
  int L_out = L * ratio;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)C * L_out;
  if (idx >= tot) return;
  int t_out = (int)(idx % L_out);
  int c     = (int)(idx / L_out);
  int t_raw = t_out + pad_left;
  const float *xc = x + (size_t)c * L;
  // Find all k in [0, K-1] such that (t_raw - k) % ratio == 0
  float acc = 0.0f;
  int k_start = t_raw % ratio;
  if (k_start < 0) k_start += ratio;
  int L_pad = L + 2 * pad;
  for (int k = k_start; k < K; k += ratio) {
    int p_in = (t_raw - k) / ratio;
    if (p_in < 0 || p_in >= L_pad) continue;
    int s_in = p_in - pad;
    if (s_in < 0) s_in = 0;
    if (s_in >= L) s_in = L - 1;
    acc += xc[s_in] * filt[k];
  }
  y[idx] = acc * (float)ratio;
}

// Exact SnakeBeta (vocoder.py:224-230): x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2
__global__ void k_voc_snakebeta_f32(float *__restrict__ x, const float *__restrict__ alpha, const float *__restrict__ beta, int C, int L) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)C * L;
  if (idx >= tot) return;
  int c = (int)(idx / L);
  float a = expf(alpha[c]);
  float b = expf(beta[c]);
  float v = x[idx];
  float s = sinf(v * a);
  x[idx] = v + (s * s) / (b + 1e-9f);
}

// Replicate-padded depthwise Conv1d for DownSample1d (vocoder.py:97-102):
// pad_left = K/2 - (K%2==0 ? 1 : 0), pad_right = K/2, stride = ratio
__global__ void k_voc_downsample1d_f32(
    const float *__restrict__ x, const float *__restrict__ filt, float *__restrict__ y,
    int C, int L_in, int ratio, int K) {
  int L_out = L_in / ratio;
  int pad_left = K / 2 - ((K % 2 == 0) ? 1 : 0);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = (size_t)C * L_out;
  if (idx >= tot) return;
  int t_out = (int)(idx % L_out);
  int c     = (int)(idx / L_out);
  const float *xc = x + (size_t)c * L_in;
  float acc = 0.0f;
  int base = t_out * ratio - pad_left;
  for (int k = 0; k < K; k++) {
    int s = base + k;
    if (s < 0) s = 0;
    if (s >= L_in) s = L_in - 1;
    acc += xc[s] * filt[k];
  }
  y[idx] = acc;
}

// Zero-padded Conv1d im2col: x [Cin, L] -> col [L, K * Cin] F32
__global__ void k_voc_conv1d_im2col_f32(
    const float *__restrict__ x, float *__restrict__ col,
    int Cin, int L, int K, int dil, int pad) {
  size_t k_dim = (size_t)K * Cin;
  size_t tot = (size_t)L * k_dim;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  int rem = (int)(idx % k_dim);
  int t   = (int)(idx / k_dim);
  int k   = rem / Cin;
  int ci  = rem % Cin;
  int s   = t + k * dil - pad;
  col[idx] = (s >= 0 && s < L) ? x[(size_t)ci * L + s] : 0.0f;
}

// Overlap-add gather for ConvTranspose1d: Y_lck [L_in, Cout * K] -> out [Cout, L_out] + bias
__global__ void k_voc_convt1d_gather_f32(
    const float *__restrict__ y_lck, const float *__restrict__ bias, float *__restrict__ out,
    int L_in, int Cout, int K, int stride, int pad, int L_out) {
  size_t tot = (size_t)Cout * L_out;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  int t_out = (int)(idx % L_out);
  int co    = (int)(idx / L_out);
  float acc = bias ? bias[co] : 0.0f;
  int num = t_out + pad;
  int k_start = num % stride;
  if (k_start < 0) k_start += stride;
  size_t ck_stride = (size_t)Cout * K;
  for (int k = k_start; k < K; k += stride) {
    int t_in = (num - k) / stride;
    if (t_in >= 0 && t_in < L_in) {
      acc += y_lck[(size_t)t_in * ck_stride + (size_t)co * K + k];
    }
  }
  out[idx] = acc;
}

__global__ void k_f32_scale_add(float *__restrict__ dst, const float *__restrict__ src, float scale, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i] * scale;
}

__global__ void k_f32_clamp(float *__restrict__ x, float lo, float hi, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    x[i] = v;
  }
}

static int ltx_voc_conv1d_f32(
    ec_ltx2_t *eng, const char *prefix,
    const float *x_d, float *y_d, int Cin, int Cout, int L, int K, int dil) {
  char wn[192], bn[192];
  snprintf(wn, sizeof wn, "%s.weight", prefix);
  snprintf(bn, sizeof bn, "%s.bias", prefix);
  const ec_st_entry_t *we = ec_st_find(&eng->main_st, wn);
  if (!we) { fprintf(stderr, "[ltx2] Missing voc conv1d %s\n", wn); return -1; }
  const uint16_t *ws16 = (const uint16_t *)ec_st_ptr(&eng->main_st, we);
  const float *ws32 = (const float *)ec_st_ptr(&eng->main_st, we);
  int is_f32 = (we->dtype == EC_DTYPE_F32);

  size_t k_dim = (size_t)K * Cin;
  float *wh = (float *)malloc((size_t)Cout * k_dim * 4);
  for (int co = 0; co < Cout; co++) {
    for (int ci = 0; ci < Cin; ci++) {
      for (int k = 0; k < K; k++) {
        size_t s = ((size_t)co * Cin + ci) * K + k;
        wh[(size_t)co * k_dim + (size_t)k * Cin + ci] = is_f32 ? ws32[s] : ltx_bf16_to_f32_host(ws16[s]);
      }
    }
  }
  float *wd = NULL, *col = NULL, *ylc = NULL;
  cudaMalloc(&wd,  (size_t)Cout * k_dim * 4);
  cudaMalloc(&col, (size_t)L * k_dim * 4);
  cudaMalloc(&ylc, (size_t)L * Cout * 4);
  cudaMemcpyAsync(wd, wh, (size_t)Cout * k_dim * 4, cudaMemcpyHostToDevice, eng->stream);
  free(wh);
  float *bd = ltx_upload_vec_f32(&eng->main_st, bn, eng->stream);

  int pad = (K * dil - dil) / 2;
  k_voc_conv1d_im2col_f32<<<((size_t)L * k_dim + 255) / 256, 256, 0, eng->stream>>>(x_d, col, Cin, L, K, dil, pad);
  ec_ops_gemm_f32(col, wd, bd, ylc, L, Cout, (int)k_dim, eng->stream);
  k_f32_transpose_2d<<<((size_t)L * Cout + 255) / 256, 256, 0, eng->stream>>>(ylc, y_d, L, Cout);
  cudaStreamSynchronize(eng->stream);
  cudaFree(wd); cudaFree(col); cudaFree(ylc);
  if (bd) cudaFree(bd);
  return 0;
}

static int ltx_voc_convt1d_f32(
    ec_ltx2_t *eng, const char *prefix,
    const float *x_d, float *y_d, int Cin, int Cout, int L_in, int K, int stride, int *out_L) {
  char wn[192], bn[192];
  snprintf(wn, sizeof wn, "%s.weight", prefix);
  snprintf(bn, sizeof bn, "%s.bias", prefix);
  const ec_st_entry_t *we = ec_st_find(&eng->main_st, wn);
  if (!we) { fprintf(stderr, "[ltx2] Missing voc convT1d %s\n", wn); return -1; }
  const uint16_t *ws16 = (const uint16_t *)ec_st_ptr(&eng->main_st, we);
  const float *ws32 = (const float *)ec_st_ptr(&eng->main_st, we);
  int is_f32 = (we->dtype == EC_DTYPE_F32);

  int pad = (K - stride) / 2;
  int L_out = (L_in - 1) * stride - 2 * pad + K;
  if (out_L) *out_L = L_out;

  // Permute [Cin, Cout, K] -> [Cout * K, Cin] F32
  size_t ck = (size_t)Cout * K;
  float *wh = (float *)malloc(ck * Cin * 4);
  for (int ci = 0; ci < Cin; ci++) {
    for (int co = 0; co < Cout; co++) {
      for (int k = 0; k < K; k++) {
        size_t s = ((size_t)ci * Cout + co) * K + k;
        wh[((size_t)co * K + k) * Cin + ci] = is_f32 ? ws32[s] : ltx_bf16_to_f32_host(ws16[s]);
      }
    }
  }
  float *wd = NULL, *xlc = NULL, *ylck = NULL;
  cudaMalloc(&wd,   ck * Cin * 4);
  cudaMalloc(&xlc,  (size_t)L_in * Cin * 4);
  cudaMalloc(&ylck, (size_t)L_in * ck * 4);
  cudaMemcpyAsync(wd, wh, ck * Cin * 4, cudaMemcpyHostToDevice, eng->stream);
  free(wh);
  float *bd = ltx_upload_vec_f32(&eng->main_st, bn, eng->stream);

  k_f32_transpose_2d<<<((size_t)Cin * L_in + 255) / 256, 256, 0, eng->stream>>>(x_d, xlc, Cin, L_in);
  ec_ops_gemm_f32(xlc, wd, NULL, ylck, L_in, (int)ck, Cin, eng->stream);
  k_voc_convt1d_gather_f32<<<((size_t)Cout * L_out + 255) / 256, 256, 0, eng->stream>>>(
      ylck, bd, y_d, L_in, Cout, K, stride, pad, L_out);
  cudaStreamSynchronize(eng->stream);
  cudaFree(wd); cudaFree(xlc); cudaFree(ylck);
  if (bd) cudaFree(bd);
  return 0;
}

static int ltx_voc_activation1d_f32(
    ec_ltx2_t *eng, const char *prefix, const float *x_d, float *y_d, int C, int L) {
  char k_up[192], k_dn[192], k_al[192], k_be[192];
  snprintf(k_up, sizeof k_up, "%s.upsample.filter", prefix);
  snprintf(k_dn, sizeof k_dn, "%s.downsample.lowpass.filter", prefix);
  snprintf(k_al, sizeof k_al, "%s.act.alpha", prefix);
  snprintf(k_be, sizeof k_be, "%s.act.beta", prefix);
  float *f_up = ltx_upload_vec_f32(&eng->main_st, k_up, eng->stream);
  float *f_dn = ltx_upload_vec_f32(&eng->main_st, k_dn, eng->stream);
  float *al_d = ltx_upload_vec_f32(&eng->main_st, k_al, eng->stream);
  float *be_d = ltx_upload_vec_f32(&eng->main_st, k_be, eng->stream);
  if (!f_up || !f_dn || !al_d || !be_d) return -1;

  // BigVGAN v2 default Activation1d: ratio=2, K=12 -> pad=5, pad_left=5*2+(12-2)/2 = 15
  int L2 = L * 2;
  float *up_d = NULL;
  cudaMalloc(&up_d, (size_t)C * L2 * 4);
  k_voc_upsample1d_f32<<<((size_t)C * L2 + 255) / 256, 256, 0, eng->stream>>>(x_d, f_up, up_d, C, L, 2, 12, 5, 15);
  k_voc_snakebeta_f32<<<((size_t)C * L2 + 255) / 256, 256, 0, eng->stream>>>(up_d, al_d, be_d, C, L2);
  k_voc_downsample1d_f32<<<((size_t)C * L + 255) / 256, 256, 0, eng->stream>>>(up_d, f_dn, y_d, C, L2, 2, 12);
  cudaStreamSynchronize(eng->stream);
  cudaFree(up_d); cudaFree(f_up); cudaFree(f_dn); cudaFree(al_d); cudaFree(be_d);
  return 0;
}

static int ltx_run_bigvgan_amp1_f32(
    ec_ltx2_t *eng, const char *prefix,
    const float *mel128_d, int T_mel,
    int init_ch, const int *up_rates, const int *up_ks, int n_ups,
    int apply_final_clamp,
    float **out_wav2_d, int *out_L) {
  static const int RES_KS[3] = {3, 7, 11};
  static const int RES_DILS[3] = {1, 3, 5};
  char p[220];

  int C = init_ch, L = T_mel;
  float *cur_d = NULL;
  cudaMalloc(&cur_d, (size_t)C * L * 4);
  snprintf(p, sizeof p, "%s.conv_pre", prefix);
  if (ltx_voc_conv1d_f32(eng, p, mel128_d, cur_d, 128, C, L, 7, 1) != 0) { cudaFree(cur_d); return -1; }

  for (int i = 0; i < n_ups; i++) {
    int C_next = C / 2;
    int L_next = L * up_rates[i];
    float *up_d = NULL;
    cudaMalloc(&up_d, (size_t)C_next * L_next * 4);
    snprintf(p, sizeof p, "%s.ups.%d", prefix, i);
    if (ltx_voc_convt1d_f32(eng, p, cur_d, up_d, C, C_next, L, up_ks[i], up_rates[i], &L_next) != 0) {
      cudaFree(cur_d); cudaFree(up_d); return -1;
    }
    cudaFree(cur_d);
    C = C_next; L = L_next;

    // Mean of 3 independent AMPBlock1 branches (vocoder.py:426-430)
    float *acc_d = NULL, *br_d = NULL, *t1_d = NULL, *t2_d = NULL;
    size_t ne = (size_t)C * L;
    cudaMalloc(&acc_d, ne * 4); cudaMemsetAsync(acc_d, 0, ne * 4, eng->stream);
    cudaMalloc(&br_d,  ne * 4);
    cudaMalloc(&t1_d,  ne * 4);
    cudaMalloc(&t2_d,  ne * 4);
    for (int j = 0; j < 3; j++) {
      int ridx = i * 3 + j;
      int rk = RES_KS[j];
      cudaMemcpyAsync(br_d, up_d, ne * 4, cudaMemcpyDeviceToDevice, eng->stream);
      for (int d = 0; d < 3; d++) {
        snprintf(p, sizeof p, "%s.resblocks.%d.acts1.%d", prefix, ridx, d);
        ltx_voc_activation1d_f32(eng, p, br_d, t1_d, C, L);
        snprintf(p, sizeof p, "%s.resblocks.%d.convs1.%d", prefix, ridx, d);
        ltx_voc_conv1d_f32(eng, p, t1_d, t2_d, C, C, L, rk, RES_DILS[d]);
        snprintf(p, sizeof p, "%s.resblocks.%d.acts2.%d", prefix, ridx, d);
        ltx_voc_activation1d_f32(eng, p, t2_d, t1_d, C, L);
        snprintf(p, sizeof p, "%s.resblocks.%d.convs2.%d", prefix, ridx, d);
        ltx_voc_conv1d_f32(eng, p, t1_d, t2_d, C, C, L, rk, 1);
        k_f32_add_inplace<<<(ne + 255) / 256, 256, 0, eng->stream>>>(br_d, t2_d, ne);
      }
      k_f32_scale_add<<<(ne + 255) / 256, 256, 0, eng->stream>>>(acc_d, br_d, 1.0f / 3.0f, ne);
    }
    cudaStreamSynchronize(eng->stream);
    cudaFree(up_d); cudaFree(br_d); cudaFree(t1_d); cudaFree(t2_d);
    cur_d = acc_d;
  }

  float *post_d = NULL, *wav_d = NULL;
  cudaMalloc(&post_d, (size_t)C * L * 4);
  cudaMalloc(&wav_d,  (size_t)2 * L * 4);
  snprintf(p, sizeof p, "%s.act_post", prefix);
  ltx_voc_activation1d_f32(eng, p, cur_d, post_d, C, L);
  cudaFree(cur_d);
  snprintf(p, sizeof p, "%s.conv_post", prefix);
  ltx_voc_conv1d_f32(eng, p, post_d, wav_d, C, 2, L, 7, 1);
  cudaFree(post_d);
  if (apply_final_clamp) {
    k_f32_clamp<<<((size_t)2 * L + 255) / 256, 256, 0, eng->stream>>>(wav_d, -1.0f, 1.0f, (size_t)2 * L);
  }
  cudaStreamSynchronize(eng->stream);
  *out_wav2_d = wav_d;
  *out_L = L;
  return 0;
}

// Compute exact causal MelSTFT from 16kHz stereo [2, L_pad] -> [128, T_bwe] F32 on CPU/GPU using checkpoint bases!
static int ltx_compute_bwe_mel_f32(
    ec_ltx2_t *eng, const float *wav16_d, int L_pad, float **out_mel128_d, int *out_T_bwe) {
  const int WIN = 512, HOP = 80, LEFT_PAD = 432, NFREQ = 257, NMEL = 64;
  int T_frames = (L_pad + LEFT_PAD - WIN) / HOP + 1;
  float *fb_d = ltx_upload_vec_f32(&eng->main_st, "vocoder.mel_stft.stft_fn.forward_basis", eng->stream); // [514, 512]
  float *mb_d = ltx_upload_vec_f32(&eng->main_st, "vocoder.mel_stft.mel_basis", eng->stream);             // [64, 257]
  if (!fb_d || !mb_d) return -1;

  float *wav_h = (float *)malloc((size_t)2 * L_pad * 4);
  float *mb_h  = (float *)malloc((size_t)NMEL * NFREQ * 4);
  cudaMemcpy(wav_h, wav16_d, (size_t)2 * L_pad * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(mb_h, mb_d, (size_t)NMEL * NFREQ * 4, cudaMemcpyDeviceToHost);
  cudaFree(mb_d);

  // Build framed matrix [2 * T_frames, 512] -> single FP32 GEMM with forward_basis [514, 512]!
  float *frames_h = (float *)calloc((size_t)2 * T_frames * WIN, 4);
  for (int ch = 0; ch < 2; ch++) {
    const float *xc = wav_h + (size_t)ch * L_pad;
    for (int t = 0; t < T_frames; t++) {
      float *dst = frames_h + ((size_t)ch * T_frames + t) * WIN;
      int start = t * HOP - LEFT_PAD;
      for (int k = 0; k < WIN; k++) {
        int s = start + k;
        if (s >= 0 && s < L_pad) dst[k] = xc[s];
      }
    }
  }
  free(wav_h);

  float *frames_d = NULL, *spec_d = NULL;
  cudaMalloc(&frames_d, (size_t)2 * T_frames * WIN * 4);
  cudaMalloc(&spec_d,   (size_t)2 * T_frames * 514 * 4);
  cudaMemcpyAsync(frames_d, frames_h, (size_t)2 * T_frames * WIN * 4, cudaMemcpyHostToDevice, eng->stream);
  ec_ops_gemm_f32(frames_d, fb_d, NULL, spec_d, 2 * T_frames, 514, WIN, eng->stream);
  float *spec_h = (float *)malloc((size_t)2 * T_frames * 514 * 4);
  cudaMemcpyAsync(spec_h, spec_d, (size_t)2 * T_frames * 514 * 4, cudaMemcpyDeviceToHost, eng->stream);
  cudaStreamSynchronize(eng->stream);
  cudaFree(fb_d); cudaFree(frames_d); cudaFree(spec_d); free(frames_h);

  // Magnitude -> mel_basis -> log(clamp(mel, 1e-5)) -> [128, T_frames]
  float *mel128_h = (float *)malloc((size_t)128 * T_frames * 4);
  for (int ch = 0; ch < 2; ch++) {
    for (int t = 0; t < T_frames; t++) {
      const float *sp = spec_h + ((size_t)ch * T_frames + t) * 514;
      float mag[257];
      for (int f = 0; f < NFREQ; f++) mag[f] = sqrtf(sp[f] * sp[f] + sp[NFREQ + f] * sp[NFREQ + f]);
      for (int m = 0; m < NMEL; m++) {
        const float *mrow = mb_h + (size_t)m * NFREQ;
        float s = 0.0f;
        for (int f = 0; f < NFREQ; f++) s += mrow[f] * mag[f];
        if (s < 1e-5f) s = 1e-5f;
        mel128_h[(size_t)(ch * NMEL + m) * T_frames + t] = logf(s);
      }
    }
  }
  free(spec_h); free(mb_h);

  float *mel128_d = NULL;
  cudaMalloc(&mel128_d, (size_t)128 * T_frames * 4);
  cudaMemcpyAsync(mel128_d, mel128_h, (size_t)128 * T_frames * 4, cudaMemcpyHostToDevice, eng->stream);
  cudaStreamSynchronize(eng->stream);
  free(mel128_h);
  *out_mel128_d = mel128_d;
  *out_T_bwe = T_frames;
  return 0;
}

static int run_ltx2_audio_decoder_and_vocoder(
    ec_ltx2_t *eng, const float *a_lat_d, int T_a,
    float **out_pcm48_lr, int *out_n_samples, double *out_ms) {
  double t0 = ltx_now_ms();
  int T = T_a, F = 16, C = 8;

  // 1. Un-normalize & unpatchify [T_a, 128] -> [T_a, 16, 8] BF16
  float *std_d  = ltx_upload_vec_f32(&eng->main_st, "audio_vae.per_channel_statistics.std-of-means", eng->stream);
  float *mean_d = ltx_upload_vec_f32(&eng->main_st, "audio_vae.per_channel_statistics.mean-of-means", eng->stream);
  uint16_t *cur_d = NULL;
  cudaMalloc(&cur_d, (size_t)T * F * C * 2);
  k_avae_unnorm_unpatch<<<((size_t)T * 128 + 255) / 256, 256, 0, eng->stream>>>(a_lat_d, std_d, mean_d, cur_d, T_a);
  cudaStreamSynchronize(eng->stream);
  cudaFree(std_d); cudaFree(mean_d);

  // conv_in: 8 -> 512
  uint16_t *nxt_d = NULL;
  cudaMalloc(&nxt_d, (size_t)T * F * 512 * 2);
  ltx_avae_conv2d(eng, "audio_vae.decoder.conv_in", 0, cur_d, nxt_d, T, F, 8, 512);
  cudaFree(cur_d); cur_d = nxt_d; C = 512;

  auto run_res2d = [&](const char *pfx, int cin, int cout) {
    size_t tf = (size_t)T * F;
    uint16_t *h1 = NULL, *h2 = NULL, *sc = NULL;
    cudaMalloc(&h1, tf * cin * 2);
    cudaMalloc(&h2, tf * cout * 2);
    k_pixelnorm_silu_bf16<<<(int)tf, 128, 0, eng->stream>>>(cur_d, h1, (int)tf, cin);
    char p1[160], p2[160], psc[160];
    snprintf(p1, sizeof p1, "%s.conv1", pfx);
    snprintf(p2, sizeof p2, "%s.conv2", pfx);
    ltx_avae_conv2d(eng, p1, 0, h1, h2, T, F, cin, cout);
    cudaFree(h1);
    cudaMalloc(&h1, tf * cout * 2);
    k_pixelnorm_silu_bf16<<<(int)tf, 128, 0, eng->stream>>>(h2, h1, (int)tf, cout);
    ltx_avae_conv2d(eng, p2, 0, h1, h2, T, F, cout, cout);
    if (cin != cout) {
      cudaMalloc(&sc, tf * cout * 2);
      snprintf(psc, sizeof psc, "%s.nin_shortcut", pfx);
      ltx_avae_conv2d(eng, psc, 1, cur_d, sc, T, F, cin, cout);
      k_bf16_add_inplace<<<(tf * cout + 255) / 256, 256, 0, eng->stream>>>(sc, h2, tf * cout);
      cudaFree(cur_d); cur_d = sc; C = cout;
    } else {
      k_bf16_add_inplace<<<(tf * cout + 255) / 256, 256, 0, eng->stream>>>(cur_d, h2, tf * cout);
    }
    cudaStreamSynchronize(eng->stream);
    cudaFree(h1); cudaFree(h2);
  };

  auto run_up2d = [&](int level) {
    int T2 = T * 2, F2 = F * 2;
    uint16_t *u1 = NULL, *u2 = NULL, *dr = NULL;
    cudaMalloc(&u1, (size_t)T2 * F2 * C * 2);
    cudaMalloc(&u2, (size_t)T2 * F2 * C * 2);
    k_avae_upsample2x_nearest<<<((size_t)T2 * F2 * C + 255) / 256, 256, 0, eng->stream>>>(cur_d, u1, T, F, C);
    cudaFree(cur_d);
    char pu[160];
    snprintf(pu, sizeof pu, "audio_vae.decoder.up.%d.upsample.conv", level);
    ltx_avae_conv2d(eng, pu, 0, u1, u2, T2, F2, C, C);
    cudaFree(u1);
    // Drop FIRST time frame (causality_axis=height, upsample.py:47)!
    int T_drop = T2 - 1;
    cudaMalloc(&dr, (size_t)T_drop * F2 * C * 2);
    cudaMemcpyAsync(dr, u2 + (size_t)F2 * C, (size_t)T_drop * F2 * C * 2, cudaMemcpyDeviceToDevice, eng->stream);
    cudaStreamSynchronize(eng->stream);
    cudaFree(u2);
    cur_d = dr; T = T_drop; F = F2;
  };

  // mid.block_1, mid.block_2 (512 -> 512)
  run_res2d("audio_vae.decoder.mid.block_1", 512, 512);
  run_res2d("audio_vae.decoder.mid.block_2", 512, 512);
  // up.2: 3 blocks @ 512 + upsample
  run_res2d("audio_vae.decoder.up.2.block.0", 512, 512);
  run_res2d("audio_vae.decoder.up.2.block.1", 512, 512);
  run_res2d("audio_vae.decoder.up.2.block.2", 512, 512);
  run_up2d(2);
  // up.1: 512->256, 256, 256 + upsample
  run_res2d("audio_vae.decoder.up.1.block.0", 512, 256);
  run_res2d("audio_vae.decoder.up.1.block.1", 256, 256);
  run_res2d("audio_vae.decoder.up.1.block.2", 256, 256);
  run_up2d(1);
  // up.0: 256->128, 128, 128
  run_res2d("audio_vae.decoder.up.0.block.0", 256, 128);
  run_res2d("audio_vae.decoder.up.0.block.1", 128, 128);
  run_res2d("audio_vae.decoder.up.0.block.2", 128, 128);

  // Final PixelNorm -> SiLU -> conv_out(128 -> 2)
  size_t tf_fin = (size_t)T * F;
  uint16_t *fn_d = NULL, *mel2_d = NULL;
  cudaMalloc(&fn_d,   tf_fin * 128 * 2);
  cudaMalloc(&mel2_d, tf_fin * 2 * 2);
  k_pixelnorm_silu_bf16<<<(int)tf_fin, 128, 0, eng->stream>>>(cur_d, fn_d, (int)tf_fin, 128);
  cudaFree(cur_d);
  ltx_avae_conv2d(eng, "audio_vae.decoder.conv_out", 0, fn_d, mel2_d, T, F, 128, 2);
  cudaFree(fn_d);

  // Adjust shape to [2, T_target=4*T_a-3, 64] and flatten into vocoder input [128, T_target] F32!
  int T_target = T_a * 4 - 3; if (T_target < 1) T_target = 1;
  uint16_t *mel2_h = (uint16_t *)malloc(tf_fin * 2 * 2);
  cudaMemcpy(mel2_h, mel2_d, tf_fin * 2 * 2, cudaMemcpyDeviceToHost);
  cudaFree(mel2_d);

  float *voc_in_h = (float *)calloc((size_t)128 * T_target, 4);
  int t_copy = (T < T_target) ? T : T_target;
  int f_copy = (F < 64) ? F : 64;
  for (int t = 0; t < t_copy; t++) {
    for (int f = 0; f < f_copy; f++) {
      for (int s = 0; s < 2; s++) {
        // mel2_h layout is [T, F, 2]; vocoder expects [s*64 + f, t]
        float v = ltx_bf16_to_f32_host(mel2_h[((size_t)t * F + f) * 2 + s]);
        voc_in_h[(size_t)(s * 64 + f) * T_target + t] = v;
      }
    }
  }
  free(mel2_h);

  float *voc_in_d = NULL;
  cudaMalloc(&voc_in_d, (size_t)128 * T_target * 4);
  cudaMemcpyAsync(voc_in_d, voc_in_h, (size_t)128 * T_target * 4, cudaMemcpyHostToDevice, eng->stream);
  cudaStreamSynchronize(eng->stream);
  free(voc_in_h);

  // 2. Pure IEEE FP32 BigVGAN v2 AMP1 Stereo Vocoder (16 kHz)
  static const int VOC_RATES[6] = {5, 2, 2, 2, 2, 2};
  static const int VOC_KS[6]    = {11, 4, 4, 4, 4, 4};
  float *wav16_d = NULL;
  int L16 = 0;
  if (ltx_run_bigvgan_amp1_f32(eng, "vocoder.vocoder", voc_in_d, T_target,
                               1536, VOC_RATES, VOC_KS, 6, 1, &wav16_d, &L16) != 0) {
    cudaFree(voc_in_d); return -1;
  }
  cudaFree(voc_in_d);

  // 3. Pure IEEE FP32 16kHz -> 48kHz Bandwidth Extension (BWE) + 3x Hann-Sinc Skip!
  int out_len48 = L16 * 3;
  int rem = L16 % 80;
  int L16_pad = rem ? (L16 + 80 - rem) : L16;
  float *wav16_pad_d = NULL;
  cudaMalloc(&wav16_pad_d, (size_t)2 * L16_pad * 4);
  cudaMemsetAsync(wav16_pad_d, 0, (size_t)2 * L16_pad * 4, eng->stream);
  cudaMemcpyAsync(wav16_pad_d,           wav16_d,       (size_t)L16 * 4, cudaMemcpyDeviceToDevice, eng->stream);
  cudaMemcpyAsync(wav16_pad_d + L16_pad, wav16_d + L16, (size_t)L16 * 4, cudaMemcpyDeviceToDevice, eng->stream);
  cudaFree(wav16_d);

  float *bwe_mel_d = NULL;
  int T_bwe = 0;
  if (ltx_compute_bwe_mel_f32(eng, wav16_pad_d, L16_pad, &bwe_mel_d, &T_bwe) != 0) {
    cudaFree(wav16_pad_d); return -1;
  }
  static const int BWE_RATES[5] = {6, 5, 2, 2, 2};
  static const int BWE_KS[5]    = {12, 11, 4, 4, 4};
  float *res48_d = NULL;
  int L48_pad = 0;
  if (ltx_run_bigvgan_amp1_f32(eng, "vocoder.bwe_generator", bwe_mel_d, T_bwe,
                               512, BWE_RATES, BWE_KS, 5, 0, &res48_d, &L48_pad) != 0) {
    cudaFree(wav16_pad_d); cudaFree(bwe_mel_d); return -1;
  }
  cudaFree(bwe_mel_d);

  // Build exact 3x Hann-windowed sinc filter (vocoder.py:116-128): width=7, K=43, pad=7, pad_left=42
  float hann_h[43];
  for (int k = 0; k < 43; k++) {
    double t = ((double)k / 3.0 - 7.0) * 0.99;
    double tc = (t < -6.0) ? -6.0 : ((t > 6.0) ? 6.0 : t);
    double w = cos(tc * M_PI / 12.0);
    w = w * w;
    double sc = (fabs(t) < 1e-12) ? 1.0 : (sin(M_PI * t) / (M_PI * t));
    hann_h[k] = (float)(sc * w * 0.99 / 3.0);
  }
  float *hann_d = NULL, *skip48_d = NULL;
  cudaMalloc(&hann_d, 43 * 4);
  cudaMalloc(&skip48_d, (size_t)2 * L48_pad * 4);
  cudaMemcpyAsync(hann_d, hann_h, 43 * 4, cudaMemcpyHostToDevice, eng->stream);
  k_voc_upsample1d_f32<<<((size_t)2 * L48_pad + 255) / 256, 256, 0, eng->stream>>>(
      wav16_pad_d, hann_d, skip48_d, 2, L16_pad, 3, 43, 7, 42);
  k_f32_add_inplace<<<((size_t)2 * L48_pad + 255) / 256, 256, 0, eng->stream>>>(res48_d, skip48_d, (size_t)2 * L48_pad);
  k_f32_clamp<<<((size_t)2 * L48_pad + 255) / 256, 256, 0, eng->stream>>>(res48_d, -1.0f, 1.0f, (size_t)2 * L48_pad);

  float *ch_h = (float *)malloc((size_t)2 * L48_pad * 4);
  cudaMemcpyAsync(ch_h, res48_d, (size_t)2 * L48_pad * 4, cudaMemcpyDeviceToHost, eng->stream);
  cudaStreamSynchronize(eng->stream);
  cudaFree(wav16_pad_d); cudaFree(hann_d); cudaFree(skip48_d); cudaFree(res48_d);

  // Interleave stereo [L0, R0, L1, R1, ...] truncated to exact out_len48
  float *pcm_lr = (float *)malloc((size_t)out_len48 * 2 * 4);
  for (int i = 0; i < out_len48; i++) {
    pcm_lr[i * 2 + 0] = ch_h[i];
    pcm_lr[i * 2 + 1] = ch_h[L48_pad + i];
  }
  free(ch_h);

  *out_pcm48_lr = pcm_lr;
  *out_n_samples = out_len48;
  if (out_ms) *out_ms = ltx_now_ms() - t0;
  fprintf(stderr, "[ltx2] Audio VAE + Pure FP32 BigVGAN v2 + BWE decoded %d samples @ 48kHz (%.2fs) in %.1f ms\n",
          out_len48, (double)out_len48 / 48000.0, ltx_now_ms() - t0);
  return 0;
}

// ===========================================================================
// 8. Public C API: Create / Destroy / Unload / Generate
// ===========================================================================

extern "C" ec_ltx2_t *ec_ltx2_create(const ec_ltx2_paths_t *paths) {
  if (!paths || !paths->checkpoint_path[0] || !paths->text_encoder_path[0] || !paths->tokenizer_path[0]) return NULL;
  ltx_init_fp8_lut();
  ec_ltx2_t *eng = (ec_ltx2_t *)calloc(1, sizeof *eng);
  if (!eng) return NULL;

  char err[256] = {0};
  if (ec_st_open(&eng->main_st, paths->checkpoint_path, err, sizeof err) != 0) {
    fprintf(stderr, "[ltx2] Failed opening main checkpoint %s: %s\n", paths->checkpoint_path, err);
    ec_ltx2_destroy(eng); return NULL;
  }
  if (ec_st_open(&eng->gemma_st, paths->text_encoder_path, err, sizeof err) != 0) {
    fprintf(stderr, "[ltx2] Failed opening Gemma-3 encoder %s: %s\n", paths->text_encoder_path, err);
    ec_ltx2_destroy(eng); return NULL;
  }
  eng->tok = ec_bpe_load(paths->tokenizer_path);
  if (!eng->tok) {
    fprintf(stderr, "[ltx2] Failed loading Gemma-3 tokenizer %s\n", paths->tokenizer_path);
    ec_ltx2_destroy(eng); return NULL;
  }
  // Verify key tensors exist
  if (!ec_st_find(&eng->main_st, "model.diffusion_model.transformer_blocks.47.attn1.to_q.weight") ||
      !ec_st_find(&eng->main_st, "vae.decoder.conv_out.conv.weight") ||
      !ec_st_find(&eng->main_st, "vocoder.bwe_generator.conv_post.weight")) {
    fprintf(stderr, "[ltx2] Main checkpoint missing required LTX-2.3 tensors!\n");
    ec_ltx2_destroy(eng); return NULL;
  }
  if (cudaStreamCreate(&eng->stream) != cudaSuccess) {
    ec_ltx2_destroy(eng); return NULL;
  }
  eng->ready = 1;
  fprintf(stderr, "[ltx2] v17.3 Lightricks LTX-2.3 (22B Distilled FP8 Joint A/V + Gemma-3-12B-IT + 48kHz BigVGAN-BWE) READY!\n");
  return eng;
}

extern "C" void ec_ltx2_unload_vram(ec_ltx2_t *eng) {
  if (!eng) return;
  ltx_free_dit_cache(eng);
  if (eng->fp8_stage_d)  { cudaFree(eng->fp8_stage_d);  eng->fp8_stage_d = NULL; }
  if (eng->bf16_stage_d) { cudaFree(eng->bf16_stage_d); eng->bf16_stage_d = NULL; }
  if (eng->bias_stage_d) { cudaFree(eng->bias_stage_d); eng->bias_stage_d = NULL; }
}

extern "C" void ec_ltx2_destroy(ec_ltx2_t *eng) {
  if (!eng) return;
  ec_ltx2_unload_vram(eng);
  if (eng->stream) cudaStreamDestroy(eng->stream);
  ec_st_close(&eng->main_st);
  ec_st_close(&eng->gemma_st);
  if (eng->tok) ec_bpe_free(eng->tok);
  free(eng);
}

extern "C" int ec_ltx2_is_ready(const ec_ltx2_t *eng) {
  return (eng && eng->ready) ? 1 : 0;
}

extern "C" int ec_ltx2_generate(
    ec_ltx2_t *eng,
    const ec_ltx2_req_t *req,
    const char *out_mp4_path,
    const char *out_wav_path,
    const char *out_keyframe0_png,
    const char *out_last_frame_png,
    ec_ltx2_progress_cb_t cb,
    void *cb_ctx,
    ec_ltx2_stats_t *stats) {
  if (!eng || !eng->ready || !req || !out_mp4_path) return -1;
  double t_all0 = ltx_now_ms();
  if (stats) memset(stats, 0, sizeof *stats);

  // Snap resolution to multiples of 32 and frames to 8*k + 1
  int W = req->width  > 0 ? ((req->width  + 16) / 32) * 32 : 768;
  int H = req->height > 0 ? ((req->height + 16) / 32) * 32 : 512;
  if (W < 256) W = 256; if (W > 1280) W = 1280;
  if (H < 256) H = 256; if (H > 768)  H = 768;
  int fps = (req->fps > 0 && req->fps <= 60) ? req->fps : 24;
  int n_frames_req = (req->n_frames > 0) ? req->n_frames : 49;
  int F_lat = (n_frames_req - 1 + 4) / 8 + 1;
  if (F_lat < 2) F_lat = 2;
  if (F_lat > 17) F_lat = 17; // up to 129 frames (~5.3s @ 24fps)
  int F_px = (F_lat - 1) * 8 + 1;
  int H_lat = H / 32, W_lat = W / 32;
  int T_a = (int)lrintf(((float)F_px / (float)fps) * 25.0f);
  if (T_a < 4) T_a = 4;
  int steps = (req->steps > 0 && req->steps <= 8) ? req->steps : 8;

  // Allocate reusable FP8/BF16 linear staging buffers:
  // Largest DiT BF16 linear is adaln_single.linear.weight [36864, 4096] = 302 MB BF16!
  if (!eng->fp8_stage_d)  cudaMalloc(&eng->fp8_stage_d,  (size_t)40960 * 4096);
  if (!eng->bf16_stage_d) cudaMalloc(&eng->bf16_stage_d, (size_t)40960 * 4096 * 2);
  if (!eng->bias_stage_d) cudaMalloc(&eng->bias_stage_d, (size_t)65536 * 4);

  // Stage 1: Gemma-3-12B-IT FP8 + FeatureExtractorV2 + 8L Video & Audio Connectors
  if (cb) cb(0, steps, "ltx2_gemma3_encode", cb_ctx);
  float *v_ctx_d = NULL, *a_ctx_d = NULL;
  cudaMalloc(&v_ctx_d, (size_t)1024 * 4096 * 4);
  cudaMalloc(&a_ctx_d, (size_t)1024 * 2048 * 4);
  double text_ms = 0.0;
  if (run_gemma3_and_connectors(eng, req->prompt, v_ctx_d, a_ctx_d, &text_ms) != 0) {
    cudaFree(v_ctx_d); cudaFree(a_ctx_d); return -1;
  }

  // Stage 2: 8-Step Distilled Dual-Stream AVTransformer3DModel
  float *v_lat_d = NULL, *a_lat_d = NULL;
  cudaMalloc(&v_lat_d, (size_t)F_lat * H_lat * W_lat * 128 * 4);
  cudaMalloc(&a_lat_d, (size_t)T_a * 128 * 4);
  double dit_ms = 0.0;
  int rc = run_ltx2_dit_8step(eng, v_ctx_d, a_ctx_d, F_lat, H_lat, W_lat, T_a, fps,
                              req->seed, steps, v_lat_d, a_lat_d, cb, cb_ctx, &dit_ms);
  cudaFree(v_ctx_d); cudaFree(a_ctx_d);
  // Release DiT VRAM cache before decoders so full 22+ GB VRAM is free for 3D Video VAE + FP32 BigVGAN-BWE!
  ltx_free_dit_cache(eng);
  if (rc != 0) { cudaFree(v_lat_d); cudaFree(a_lat_d); return -1; }

  // Stage 3: Video 3D Decoder -> RGB24 frames
  if (cb) cb(steps, steps, "ltx2_video_vae", cb_ctx);
  uint8_t *rgb_h = NULL;
  int dec_F = 0, dec_H = 0, dec_W = 0;
  double vvae_ms = 0.0;
  rc = run_ltx2_video_decoder(eng, v_lat_d, F_lat, H_lat, W_lat, &rgb_h, &dec_F, &dec_H, &dec_W, &vvae_ms);
  cudaFree(v_lat_d);
  if (rc != 0 || !rgb_h) { cudaFree(a_lat_d); return -1; }

  // Stage 4: Audio 2D Causal Mel Decoder + Pure FP32 BigVGAN v2 AMP1 + 16k->48k BWE -> 48kHz Studio Stereo WAV!
  double avae_ms = 0.0;
  float *pcm48_lr = NULL;
  int n_audio_samples = 0;
  char wav_tmp[512] = {0};
  const char *wav_use = NULL;
  if (req->with_audio) {
    if (cb) cb(steps, steps, "ltx2_audio_vocoder", cb_ctx);
    rc = run_ltx2_audio_decoder_and_vocoder(eng, a_lat_d, T_a, &pcm48_lr, &n_audio_samples, &avae_ms);
    if (rc == 0 && pcm48_lr && n_audio_samples > 0) {
      // Gentle broadcast true-peak safety guard (-1.0 dBTP, preserve natural dynamics & foley realism!)
      float lufs_m = 0.0f, peak_m = 0.0f;
      ec_audio_master_studio(pcm48_lr, n_audio_samples, 48000, -16.0f, -1.0f, &lufs_m, &peak_m);
      fprintf(stderr, "[ltx2] Mastered synced 48kHz audio: LUFS=%.1f dBTP=%.1f\n", lufs_m, peak_m);
      if (out_wav_path && out_wav_path[0]) {
        wav_use = out_wav_path;
      } else {
        snprintf(wav_tmp, sizeof wav_tmp, "%s.ltx2_tmp.wav", out_mp4_path);
        wav_use = wav_tmp;
      }
      ec_audio_t aud = { pcm48_lr, (uint32_t)n_audio_samples, 48000, 2 };
      if (ec_export_wav(wav_use, &aud) != 0) wav_use = NULL;
      free(pcm48_lr);
    }
  }
  cudaFree(a_lat_d);

  // Save keyframe 0 and final keyframe PNGs
  size_t frame_bytes = (size_t)dec_W * dec_H * 3;
  if (out_keyframe0_png && out_keyframe0_png[0]) {
    ec_export_png(out_keyframe0_png, rgb_h, (uint32_t)dec_W, (uint32_t)dec_H);
  }
  if (out_last_frame_png && out_last_frame_png[0]) {
    ec_export_png(out_last_frame_png, rgb_h + (size_t)(dec_F - 1) * frame_bytes, (uint32_t)dec_W, (uint32_t)dec_H);
  }

  // Hardware NVENC BT.709/tv H.264 + 48kHz AAC MP4 Mux!
  char enc_used[32] = {0};
  ec_mp4_writer_t *mw = ec_mp4_open(out_mp4_path, wav_use, (uint32_t)dec_W, (uint32_t)dec_H,
                                    (uint32_t)fps, 1, enc_used, sizeof enc_used);
  if (!mw) {
    free(rgb_h);
    if (wav_tmp[0]) remove(wav_tmp);
    return -1;
  }
  for (int f = 0; f < dec_F; f++) {
    ec_mp4_write_rgb24(mw, rgb_h + (size_t)f * frame_bytes, frame_bytes);
  }
  int mrc = ec_mp4_close(mw);
  free(rgb_h);
  if (wav_tmp[0]) remove(wav_tmp);
  if (mrc != 0) return -1;

  if (stats) {
    stats->width = dec_W;
    stats->height = dec_H;
    stats->n_frames = dec_F;
    stats->fps = fps;
    stats->steps_run = steps;
    stats->sample_rate = 48000;
    stats->audio_frames = n_audio_samples;
    stats->text_enc_ms = text_ms;
    stats->dit_ms = dit_ms;
    stats->video_vae_ms = vvae_ms;
    stats->audio_vae_ms = avae_ms;
    stats->total_ms = ltx_now_ms() - t_all0;
    snprintf(stats->encoder_used, sizeof stats->encoder_used, "%s", enc_used);
  }
  return 0;
}
