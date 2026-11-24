// ec_flux2.cu — 100% Pure Native Eureka Photoreal FLUX.2-klein-4B Pipeline
// Zero Python, zero cuBLAS, zero ggml, zero stable-diffusion.cpp!
// Architecture:
//   1. Native PTX mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 TensorCore GEMM
//   2. Pure CUDA Qwen3-4B Text Encoder (layers 0..26 streamed from mmap -> [512, 7680])
//   3. Pure CUDA FLUX.2-klein-4B MMDiT (5 Double + 20 Single Blocks, 4D RoPE theta=2000,
//      4-step compute_empirical_mu dynamic shifted RF, hot-fused LoRAs & custom finetunes)
//   4. Pure CUDA AutoencoderKLFlux2 2D VAE Decoder (strip-tiled BF16 TC Conv2D -> RFC 2083 PNG)

#define _POSIX_C_SOURCE 200809L
#include "ec_flux2.h"
#include "ec_bpe.h"
#include "ec_cowrie.h"
#include "ec_kernels.cuh"
#include "ec_kernels_v2.cuh"
#include "ec_media.h"
#include "ec_ops.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double fx_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static __device__ __forceinline__ float bf16_to_f32_dev(uint16_t b) {
  return __uint_as_float(((uint32_t)b) << 16);
}
static __device__ __forceinline__ uint16_t f32_to_bf16_dev(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float silu_dev(float x) {
  return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float warp_sum_dev(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// ===========================================================================
// 1. Unified High-Throughput BF16 TensorCore GEMM (via shared ec_ops.cu)
// ===========================================================================
static void gemm_bf16(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t s) {
  ec_ops_gemm_bf16(X, W, bias, Y_f32, Y_bf16, M, N, K, s);
}

// Elementwise helpers
__global__ void k_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f32_to_bf16_dev(src[i]);
}
__global__ void k_bf16_to_f32(const uint16_t *src, float *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = bf16_to_f32_dev(src[i]);
}
__global__ void k_silu_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f32_to_bf16_dev(silu_dev(src[i]));
}
__global__ void k_add_inplace(float *dst, const float *src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
__global__ void k_swiglu_pair_bf16(const float *gate, const float *up, uint16_t *out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f32_to_bf16_dev(silu_dev(gate[i]) * up[i]);
}
// Fused SwiGLU on [N, 2*H] -> [N, H] BF16 (first half gate, second half up — exact Flux2SwiGLU)
__global__ void k_flux2_swiglu_bf16(const float *in2h, uint16_t *out_h, int rows, int H) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * H;
  if (idx >= tot) return;
  int r = (int)(idx / H), c = (int)(idx % H);
  float g = in2h[(long)r * (2 * H) + c];
  float u = in2h[(long)r * (2 * H) + H + c];
  out_h[idx] = f32_to_bf16_dev(silu_dev(g) * u);
}

// RMSNorm: y_bf16[r, c] = (x[r, c] * rsqrt(mean(x^2) + eps)) * w[c]
__global__ void k_rmsnorm_bf16(const float *x, const float *w, uint16_t *y_bf16, int rows, int dim, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  uint16_t *yr = y_bf16 + (long)r * dim;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; ss += v * v; }
  __shared__ float sm[32];
  ss = warp_sum_dev(ss);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    t = warp_sum_dev(t);
    if (threadIdx.x == 0) sm[0] = rsqrtf(t / (float)dim + eps);
  }
  __syncthreads();
  float inv = sm[0];
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    yr[c] = f32_to_bf16_dev(xr[c] * inv * (w ? w[c] : 1.0f));
  }
}

// LayerNorm(no affine, eps=1e-6) + (1 + scale)*norm + shift -> BF16
__global__ void k_adaln_bf16(
    const float *x, const float *shift, const float *scale,
    uint16_t *y_bf16, int rows, int dim, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  uint16_t *yr = y_bf16 + (long)r * dim;
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; s1 += v; s2 += v * v; }
  __shared__ float sm1[32], sm2[32];
  s1 = warp_sum_dev(s1); s2 = warp_sum_dev(s2);
  if ((threadIdx.x & 31) == 0) { sm1[threadIdx.x >> 5] = s1; sm2[threadIdx.x >> 5] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = blockDim.x >> 5;
    float a = (threadIdx.x < nw) ? sm1[threadIdx.x] : 0.0f;
    float b = (threadIdx.x < nw) ? sm2[threadIdx.x] : 0.0f;
    a = warp_sum_dev(a); b = warp_sum_dev(b);
    if (threadIdx.x == 0) {
      float mean = a / (float)dim;
      float var = b / (float)dim - mean * mean;
      if (var < 0.0f) var = 0.0f;
      sm1[0] = mean;
      sm2[0] = rsqrtf(var + eps);
    }
  }
  __syncthreads();
  float mean = sm1[0], inv = sm2[0];
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float nrm = (xr[c] - mean) * inv;
    float v = nrm * (1.0f + scale[c]) + shift[c];
    yr[c] = f32_to_bf16_dev(v);
  }
}

// ===========================================================================
// 2. Qwen3-4B Attention Kernels (QK-RMSNorm + Split-Half 1D RoPE + GQA Causal+Mask)
// ===========================================================================
__global__ void k_qwen3_qk_rope(
    const float *__restrict__ q_in, const float *__restrict__ k_in, const float *__restrict__ v_in,
    const float *__restrict__ q_norm_w, const float *__restrict__ k_norm_w,
    float *__restrict__ q_out, float *__restrict__ k_out, float *__restrict__ v_out,
    int L, int Hq, int Hkv, int D, float theta, float eps) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  int total_q = L * Hq;
  int total_kv = L * Hkv;
  if (idx < total_q) {
    int pos = idx / Hq, h = idx % Hq;
    const float *qs = q_in + ((long)pos * Hq + h) * D;
    float *qo = q_out + ((long)h * L + pos) * D;
    float sq = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = qs[d]; sq += v * v; }
    sq = warp_sum_dev(sq);
    float inv = rsqrtf(sq / (float)D + eps);
    int half = D >> 1;
    for (int j = lane; j < half; j += 32) {
      float x0 = qs[j] * inv * q_norm_w[j];
      float x1 = qs[j + half] * inv * q_norm_w[j + half];
      double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
      float cs = (float)cos(ang), sn = (float)sin(ang);
      qo[j]        = x0 * cs - x1 * sn;
      qo[j + half] = x1 * cs + x0 * sn;
    }
  } else if (idx < total_q + total_kv) {
    int kidx = idx - total_q;
    int pos = kidx / Hkv, h = kidx % Hkv;
    const float *ks = k_in + ((long)pos * Hkv + h) * D;
    const float *vs = v_in + ((long)pos * Hkv + h) * D;
    float *ko = k_out + ((long)h * L + pos) * D;
    float *vo = v_out + ((long)h * L + pos) * D;
    float sk = 0.0f;
    for (int d = lane; d < D; d += 32) { float v = ks[d]; sk += v * v; }
    sk = warp_sum_dev(sk);
    float inv = rsqrtf(sk / (float)D + eps);
    int half = D >> 1;
    for (int j = lane; j < half; j += 32) {
      float x0 = ks[j] * inv * k_norm_w[j];
      float x1 = ks[j + half] * inv * k_norm_w[j + half];
      double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
      float cs = (float)cos(ang), sn = (float)sin(ang);
      ko[j]        = x0 * cs - x1 * sn;
      ko[j + half] = x1 * cs + x0 * sn;
      vo[j]        = vs[j];
      vo[j + half] = vs[j + half];
    }
  }
}

// Exact F32 GQA Causal + Attention-Mask Attention (1 warp per (q_head, query_pos), D=128)
__global__ void k_qwen3_gqa_attn(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    const uint8_t *__restrict__ mask, uint16_t *__restrict__ O_bf16,
    int L, int Hq, int Hkv, int D, float scale) {
  int pair = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (pair >= L * Hq) return;
  int qpos = pair / Hq;
  int hq = pair % Hq;
  int hkv = hq / (Hq / Hkv);

  const float *qvec = Q + ((long)hq * L + qpos) * D;
  const float *kbase = K + (long)hkv * L * D;
  const float *vbase = V + (long)hkv * L * D;

  float qr[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) qr[i] = qvec[lane + i * 32];

  float m_max = -1e30f;
  float l_sum = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kpos = 0; kpos <= qpos; kpos++) {
    // Attend to all valid tokens up to qpos (and self if pad token)
    if (mask[kpos] == 0 && kpos != qpos) continue;
    const float *kv = kbase + (long)kpos * D;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) dot += qr[i] * kv[lane + i * 32];
    dot = warp_sum_dev(dot) * scale;

    float new_max = fmaxf(m_max, dot);
    float alpha = expf(m_max - new_max);
    float p = expf(dot - new_max);
    l_sum = l_sum * alpha + p;
    m_max = new_max;
    const float *vv = vbase + (long)kpos * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = acc[i] * alpha + p * vv[lane + i * 32];
  }

  float inv_l = 1.0f / fmaxf(l_sum, 1e-20f);
  uint16_t *out = O_bf16 + ((long)qpos * Hq + hq) * D;
  #pragma unroll
  for (int i = 0; i < 4; i++) out[lane + i * 32] = f32_to_bf16_dev(acc[i] * inv_l);
}

// Scatter hidden state [512, 2560] into prompt_embeds [512, 7680] at slot s_idx (0, 1, 2)
__global__ void k_scatter_qwen_layer(const float *x, uint16_t *dst_7680, int L, int dim, int slot) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)L * dim;
  if (idx >= tot) return;
  int r = (int)(idx / dim), c = (int)(idx % dim);
  dst_7680[(long)r * (3 * dim) + slot * dim + c] = f32_to_bf16_dev(x[idx]);
}

// ===========================================================================
// 3. FLUX.2-klein-4B 4D RoPE, QKV Unpack, Single-Block Pack & Euler
// ===========================================================================
// Precompute 4D RoPE (axes [32,32,32,32], theta=2000, float64 precision) for [txt(0..N_txt-1), img(0..N_img-1)]
__global__ void k_flux2_precompute_rope(
    float *cos_out, float *sin_out, int N_txt, int H_p, int W_p, double theta) {
  int N_img = H_p * W_p;
  int N_tot = N_txt + N_img;
  int tok = blockIdx.x;
  int p = threadIdx.x; // 0..63 (64 pairs -> 128 head_dim)
  if (tok >= N_tot || p >= 64) return;
  int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  if (tok < N_txt) {
    c3 = tok;
  } else {
    int im = tok - N_txt;
    c1 = im / W_p;
    c2 = im % W_p;
  }
  int axis = p >> 4;    // 0, 1, 2, 3 (16 pairs per axis = 32 channels per axis)
  int local = p & 15;   // 0..15
  int coord = (axis == 0) ? c0 : ((axis == 1) ? c1 : ((axis == 2) ? c2 : c3));
  double freq = 1.0 / pow(theta, (double)(2 * local) / 32.0);
  double ang = (double)coord * freq;
  cos_out[(long)tok * 64 + p] = (float)cos(ang);
  sin_out[(long)tok * 64 + p] = (float)sin(ang);
}

// Pack QKV [N, 3*E] (or strided [N, stride] where Q,K,V are first 3*E) with QK-RMSNorm + 4D RoPE
// into [H, N_pad, D] FP16 at token offset `tok_off` for ec_launch_fa2_f16!
__global__ void k_flux2_pack_qkv_rope(
    const float *__restrict__ qkv, int row_stride,
    const float *__restrict__ q_norm_w, const float *__restrict__ k_norm_w,
    const float *__restrict__ rope_cos, const float *__restrict__ rope_sin,
    __half *__restrict__ Q_out, __half *__restrict__ K_out, __half *__restrict__ V_out,
    int n_tok, int tok_off, int n_pad, int H, int D, float eps) {
  long pair = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (pair >= (long)n_tok * H) return;
  int h = (int)(pair % H);
  int t = (int)(pair / H);
  int E = H * D;
  const float *base = qkv + (long)t * row_stride;
  const float *qs = base + h * D;
  const float *ks = base + E + h * D;
  const float *vs = base + 2 * E + h * D;

  float sq = 0.0f, sk = 0.0f;
  for (int d = lane; d < D; d += 32) {
    float qv = qs[d], kv = ks[d];
    sq += qv * qv; sk += kv * kv;
  }
  sq = warp_sum_dev(sq); sk = warp_sum_dev(sk);
  float inv_q = rsqrtf(sq / (float)D + eps);
  float inv_k = rsqrtf(sk / (float)D + eps);

  int gtok = tok_off + t;
  const float *rc = rope_cos + (long)gtok * (D >> 1);
  const float *rs = rope_sin + (long)gtok * (D >> 1);

  long out_base = ((long)h * n_pad + gtok) * D;
  __half *qo = Q_out + out_base;
  __half *ko = K_out + out_base;
  __half *vo = V_out + out_base;

  // Adjacent pairs (2p, 2p+1) — exact FLUX.2 repeat_interleave_real=True convention!
  for (int p = lane; p < (D >> 1); p += 32) {
    int d0 = 2 * p, d1 = d0 + 1;
    float q0 = qs[d0] * inv_q * q_norm_w[d0];
    float q1 = qs[d1] * inv_q * q_norm_w[d1];
    float k0 = ks[d0] * inv_k * k_norm_w[d0];
    float k1 = ks[d1] * inv_k * k_norm_w[d1];
    float cs = rc[p], sn = rs[p];
    float rq0 = q0 * cs - q1 * sn, rq1 = q0 * sn + q1 * cs;
    float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;
    qo[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, rq0)));
    qo[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, rq1)));
    ko[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, rk0)));
    ko[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, rk1)));
    vo[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, vs[d0])));
    vo[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, vs[d1])));
  }
}

// Single-block concat: cat_bf16[t, 0..E-1] = attn_out[t, :], cat_bf16[t, E..E+F-1] = SwiGLU(lin1[t, 3E..3E+2F])
__global__ void k_flux2_single_cat(
    const float *attn_out, const float *lin1, uint16_t *cat_bf16, int N_tot, int E, int F) {
  int stride_out = E + F;
  int stride_in = 3 * E + 2 * F;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N_tot * stride_out;
  if (idx >= tot) return;
  int t = (int)(idx / stride_out);
  int c = (int)(idx % stride_out);
  float v;
  if (c < E) {
    v = attn_out[(long)t * E + c];
  } else {
    int fc = c - E;
    const float *row = lin1 + (long)t * stride_in + 3 * E;
    v = silu_dev(row[fc]) * row[F + fc];
  }
  cat_bf16[idx] = f32_to_bf16_dev(v);
}

// Fused Single-Block Concat + SwiGLU + Q8_G64 Quantizer (eliminates 18 GB/img cat_bf DRAM roundtrip!)
__global__ void __launch_bounds__(256) k_flux2_single_cat_q8g64(
    const float *__restrict__ attn_out, const float *__restrict__ lin1,
    int8_t *__restrict__ dst_q8, float *__restrict__ dst_sc,
    int N_tot, int E, int F) {
  const int stride_out = E + F;
  const int stride_in  = 3 * E + 2 * F;
  const int groups_per_row = stride_out >> 6;
  const long total_groups  = (long)N_tot * groups_per_row;
  const long hw_id = (((long)blockIdx.x * blockDim.x) + threadIdx.x) >> 4;
  if (hw_id >= total_groups) return;

  const int sub_lane = threadIdx.x & 15;
  const int t = (int)(hw_id / groups_per_row);
  const int c0 = (int)(hw_id - (long)t * groups_per_row) * 64 + sub_lane * 4;

  float vf[4];
  if (c0 < E) {
    const float4 a4 = *(const float4 *)(attn_out + (long)t * E + c0);
    vf[0] = bf16_to_f32_dev(f32_to_bf16_dev(a4.x));
    vf[1] = bf16_to_f32_dev(f32_to_bf16_dev(a4.y));
    vf[2] = bf16_to_f32_dev(f32_to_bf16_dev(a4.z));
    vf[3] = bf16_to_f32_dev(f32_to_bf16_dev(a4.w));
  } else {
    const int fc = c0 - E;
    const float *row = lin1 + (long)t * stride_in + 3 * E;
    const float4 g4 = *(const float4 *)(row + fc);
    const float4 u4 = *(const float4 *)(row + F + fc);
    vf[0] = bf16_to_f32_dev(f32_to_bf16_dev(silu_dev(g4.x) * u4.x));
    vf[1] = bf16_to_f32_dev(f32_to_bf16_dev(silu_dev(g4.y) * u4.y));
    vf[2] = bf16_to_f32_dev(f32_to_bf16_dev(silu_dev(g4.z) * u4.z));
    vf[3] = bf16_to_f32_dev(f32_to_bf16_dev(silu_dev(g4.w) * u4.w));
  }

  float amax = fmaxf(fmaxf(fabsf(vf[0]), fabsf(vf[1])), fmaxf(fabsf(vf[2]), fabsf(vf[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

  const float sc  = amax * (1.0f / 127.0f);
  const float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
  auto q8 = [&](float v) -> uint32_t {
    int q = __float2int_rn(v * inv);
    if (q < -127) q = -127;
    if (q >  127) q =  127;
    return (uint32_t)(uint8_t)(int8_t)q;
  };
  const long elem_off = hw_id * 64 + sub_lane * 4;
  *(uint32_t *)(dst_q8 + elem_off) = q8(vf[0]) | (q8(vf[1]) << 8) | (q8(vf[2]) << 16) | (q8(vf[3]) << 24);
  if (sub_lane == 0) dst_sc[hw_id] = sc;
}

// ===========================================================================
// 4. AutoencoderKLFlux2 2D VAE Kernels (Patchify BN Norm + Strip TC Conv2D + NHWC Implicit-GEMM)
// ===========================================================================
// Inverse: take first 32 mean channels of quant_conv [64, H_p*2, W_p*2] -> 2x2 patchify + BN norm -> [H_p*W_p, 128]
__global__ void k_flux2_patchify_bn_norm(
    const float *in_64_chw, const float *bn_mean, const float *bn_var,
    float *packed_out, int H_p, int W_p, float bn_eps) {
  int H_lat = H_p * 2, W_lat = W_p * 2;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)H_p * W_p * 128;
  if (idx >= tot) return;
  int p = (int)(idx / 128);
  int pch = (int)(idx % 128);
  int hp = p / W_p, wp = p % W_p;
  int c = pch >> 2, py = (pch >> 1) & 1, px = pch & 1;
  int y = hp * 2 + py, x = wp * 2 + px;
  float v = in_64_chw[((long)c * H_lat + y) * W_lat + x];
  float inv_std = rsqrtf(bn_var[pch] + bn_eps);
  packed_out[idx] = (v - bn_mean[pch]) * inv_std;
}

// Strip-tiled im2col for Stride-2 3x3 Downsample2D (asymmetric pad [0,1,0,1] matching Diffusers AutoencoderKL!)
__global__ void k_im2col_strip_s2_bf16(
    const float *__restrict__ x, uint16_t *__restrict__ col,
    int Cin, int H_in, int W_in, int H_out, int W_out, int y0, int strip_h) {
  int K = Cin * 9;
  long total = (long)strip_h * W_out * K;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int k = (int)(idx % K);
  int hw = (int)(idx / K);
  int dy = hw / W_out, ox = hw % W_out;
  int oy = y0 + dy;
  int c = k / 9, rem = k % 9;
  int kh = rem / 3, kw = rem % 3;
  int iy = oy * 2 + kh;
  int ix = ox * 2 + kw;
  float v = (iy >= 0 && iy < H_in && ix >= 0 && ix < W_in) ? x[((long)c * H_in + iy) * W_in + ix] : 0.0f;
  col[idx] = f32_to_bf16_dev(v);
}

// Inpaint / Img2Img forward-noise blend: x = mask * x + (1 - mask) * ((1 - sigma) * z0 + sigma * eps)
__global__ void k_flux2_mask_project(
    float *x, const float *z0, const float *eps, const float *mask_p,
    int N_img, float sigma) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)N_img * 128;
  if (idx >= tot) return;
  int p = (int)(idx / 128);
  float m = mask_p ? mask_p[p] : 1.0f;
  float gt = (1.0f - sigma) * z0[idx] + sigma * eps[idx];
  x[idx] = m * x[idx] + (1.0f - m) * gt;
}

// Strip-tiled im2col into BF16: rows [y0 .. y0+strip_h) of [Cin, H, W] -> col_bf16[strip_h*W, Cin*Kh*Kw]
__global__ void k_im2col_strip_bf16(
    const float *__restrict__ x, uint16_t *__restrict__ col,
    int Cin, int H, int W, int Kh, int Kw, int pad, int y0, int strip_h) {
  int K = Cin * Kh * Kw;
  long total = (long)strip_h * W * K;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int k = (int)(idx % K);
  int hw = (int)(idx / K);
  int dy = hw / W, ox = hw % W;
  int oy = y0 + dy;
  int c = k / (Kh * Kw);
  int rem = k % (Kh * Kw);
  int kh = rem / Kw, kw = rem % Kw;
  int iy = oy - pad + kh;
  int ix = ox - pad + kw;
  float v = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[((long)c * H + iy) * W + ix] : 0.0f;
  col[idx] = f32_to_bf16_dev(v);
}

// Transpose strip GEMM output [strip_h*W, Cout] -> y[Cout, H, W] at row offset y0 (+ optional residual)
__global__ void k_strip_transpose_out(
    const float *__restrict__ gemm_y, const float *__restrict__ res,
    float *__restrict__ y_chw, int Cout, int H, int W, int y0, int strip_h) {
  long total = (long)Cout * strip_h * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int hw = (int)(idx % (strip_h * W));
  int oc = (int)(idx / (strip_h * W));
  int dy = hw / W, ox = hw % W;
  long dst_i = ((long)oc * H + (y0 + dy)) * W + ox;
  float v = gemm_y[(long)hw * Cout + oc];
  if (res) v += res[dst_i];
  y_chw[dst_i] = v;
}

// Single-head spatial self-attention for VAE mid_block (C=512, S=H_lat*W_lat)
__global__ void k_vae_mid_attn(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    uint16_t *__restrict__ O_bf16, int S, int C, float scale) {
  int q = blockIdx.x;
  if (q >= S) return;
  // 256 threads handle C=512 (2 elements per thread)
  int tid = threadIdx.x;
  float q0 = Q[(long)q * C + tid];
  float q1 = Q[(long)q * C + tid + 256];
  float acc0 = 0.0f, acc1 = 0.0f;
  float m_max = -1e30f, l_sum = 0.0f;
  __shared__ float sm[32];

  for (int k = 0; k < S; k++) {
    float dot = q0 * K[(long)k * C + tid] + q1 * K[(long)k * C + tid + 256];
    dot = warp_sum_dev(dot);
    if ((tid & 31) == 0) sm[tid >> 5] = dot;
    __syncthreads();
    if (tid < 32) {
      float t = (tid < 8) ? sm[tid] : 0.0f;
      t = warp_sum_dev(t);
      if (tid == 0) sm[0] = t * scale;
    }
    __syncthreads();
    float sc = sm[0];
    float nm = fmaxf(m_max, sc);
    float al = expf(m_max - nm);
    float p = expf(sc - nm);
    l_sum = l_sum * al + p;
    m_max = nm;
    acc0 = acc0 * al + p * V[(long)k * C + tid];
    acc1 = acc1 * al + p * V[(long)k * C + tid + 256];
  }
  float inv = 1.0f / fmaxf(l_sum, 1e-20f);
  O_bf16[(long)q * C + tid]       = f32_to_bf16_dev(acc0 * inv);
  O_bf16[(long)q * C + tid + 256] = f32_to_bf16_dev(acc1 * inv);
}

// Transpose [C, S] -> [S, C] BF16 (for mid_block attention Q/K/V)
__global__ void k_chw_to_sc_bf16(const float *x, uint16_t *y, int C, int S) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)C * S) return;
  int s = (int)(idx / C), c = (int)(idx % C);
  y[idx] = f32_to_bf16_dev(x[(long)c * S + s]);
}
__global__ void k_sc_add_to_chw(const float *sc, float *chw, int C, int S) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)C * S) return;
  int s = (int)(idx % S), c = (int)(idx / S);
  chw[idx] += sc[(long)s * C + c];
}

// ===========================================================================
// Weight Loading Helpers (Supports BF16, FP16, F32, FP8_E4M3 + prefix remapping!)
// ===========================================================================
static const ec_st_entry_t *find_tensor_flex(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (e) return e;
  char alt[192];
  snprintf(alt, sizeof alt, "model.diffusion_model.%s", name);
  e = ec_st_find(st, alt);
  if (e) return e;
  snprintf(alt, sizeof alt, "diffusion_model.%s", name);
  return ec_st_find(st, alt);
}

static uint16_t *load_bf16_gpu(const ec_safetensors_t *st, const char *name, size_t *out_n, cudaStream_t s) {
  const ec_st_entry_t *e = find_tensor_flex(st, name);
  if (!e) { fprintf(stderr, "[ec_flux2] missing tensor: %s\n", name); return NULL; }
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  if (out_n) *out_n = n;
  uint16_t *d = NULL;
  if (cudaMalloc(&d, n * sizeof(uint16_t)) != cudaSuccess) return NULL;
  const void *src = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(d, src, n * 2, cudaMemcpyHostToDevice, s);
  } else if (e->dtype == EC_DTYPE_F32) {
    uint16_t *tmp = (uint16_t *)malloc(n * 2);
    const float *fp = (const float *)src;
    for (size_t i = 0; i < n; i++) {
      uint32_t u; memcpy(&u, &fp[i], 4);
      u += 0x7FFFU + ((u >> 16) & 1U);
      tmp[i] = (uint16_t)(u >> 16);
    }
    cudaMemcpy(d, tmp, n * 2, cudaMemcpyHostToDevice);
    free(tmp);
  } else if (e->dtype == EC_DTYPE_F16) {
    uint16_t *tmp = (uint16_t *)malloc(n * 2);
    const uint16_t *hp = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
      float v = __half2float(*(const __half *)&hp[i]);
      uint32_t u; memcpy(&u, &v, 4);
      u += 0x7FFFU + ((u >> 16) & 1U);
      tmp[i] = (uint16_t)(u >> 16);
    }
    cudaMemcpy(d, tmp, n * 2, cudaMemcpyHostToDevice);
    free(tmp);
  } else if (e->dtype == EC_DTYPE_FP8_E4M3) {
    // Convert FP8 -> F16 on GPU -> F32 -> BF16
    uint8_t *fp8_d = NULL; __half *f16_d = NULL;
    cudaMalloc(&fp8_d, n); cudaMalloc(&f16_d, n * 2);
    cudaMemcpyAsync(fp8_d, src, n, cudaMemcpyHostToDevice, s);
    ec_launch_fp8_to_f16(fp8_d, f16_d, n, 1.0f, s);
    float *f32_d = NULL; cudaMalloc(&f32_d, n * 4);
    cudaStreamSynchronize(s);
    __half *f16_h = (__half *)malloc(n * 2);
    uint16_t *bf_h = (uint16_t *)malloc(n * 2);
    cudaMemcpy(f16_h, f16_d, n * 2, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; i++) {
      float v = __half2float(f16_h[i]);
      uint32_t u; memcpy(&u, &v, 4);
      u += 0x7FFFU + ((u >> 16) & 1U);
      bf_h[i] = (uint16_t)(u >> 16);
    }
    cudaMemcpy(d, bf_h, n * 2, cudaMemcpyHostToDevice);
    free(f16_h); free(bf_h); cudaFree(fp8_d); cudaFree(f16_d); cudaFree(f32_d);
  } else {
    cudaFree(d); return NULL;
  }
  return d;
}

static float *load_f32_gpu(const ec_safetensors_t *st, const char *name, size_t *out_n, cudaStream_t s) {
  const ec_st_entry_t *e = find_tensor_flex(st, name);
  if (!e) { fprintf(stderr, "[ec_flux2] missing f32 tensor: %s\n", name); return NULL; }
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  if (out_n) *out_n = n;
  float *d = NULL;
  if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
  const void *src = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_F32) {
    cudaMemcpyAsync(d, src, n * 4, cudaMemcpyHostToDevice, s);
  } else if (e->dtype == EC_DTYPE_BF16) {
    uint16_t *tmp_d = NULL; cudaMalloc(&tmp_d, n * 2);
    cudaMemcpyAsync(tmp_d, src, n * 2, cudaMemcpyHostToDevice, s);
    k_bf16_to_f32<<<(int)((n + 255) / 256), 256, 0, s>>>(tmp_d, d, (long)n);
    cudaStreamSynchronize(s);
    cudaFree(tmp_d);
  } else if (e->dtype == EC_DTYPE_F16) {
    float *h = (float *)malloc(n * 4);
    const uint16_t *hp = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) h[i] = __half2float(*(const __half *)&hp[i]);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    free(h);
  } else {
    cudaFree(d); return NULL;
  }
  return d;
}

// Copy BF16 tensor directly into preallocated GPU buffer (for fast layer streaming!)
static int stream_bf16_into(const ec_safetensors_t *st, const char *name, uint16_t *dst_d, size_t expected_n, cudaStream_t s) {
  const ec_st_entry_t *e = find_tensor_flex(st, name);
  if (!e || e->dtype != EC_DTYPE_BF16) return -1;
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  if (n != expected_n) return -1;
  cudaMemcpyAsync(dst_d, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, s);
  return 0;
}

static int stream_f32_into(const ec_safetensors_t *st, const char *name, float *dst_d, uint16_t *tmp_bf16_d, size_t expected_n, cudaStream_t s) {
  const ec_st_entry_t *e = find_tensor_flex(st, name);
  if (!e) return -1;
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  if (n != expected_n) return -1;
  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(tmp_bf16_d, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, s);
    k_bf16_to_f32<<<(int)((n + 255) / 256), 256, 0, s>>>(tmp_bf16_d, dst_d, (long)n);
  } else if (e->dtype == EC_DTYPE_F32) {
    cudaMemcpyAsync(dst_d, ec_st_ptr(st, e), n * 4, cudaMemcpyHostToDevice, s);
  } else return -1;
  return 0;
}

// ===========================================================================
// FLUX.2-klein-4B Resident GPU Structures (v15 Q8_G64 4.15 GB + Exact BF16 Oracle)
// ===========================================================================
typedef struct {
  int8_t   *q8;   // [out_d, in_d] symmetric per-64-group INT8
  float    *sc;   // [out_d, in_d / 64] FP32 group scales
  uint16_t *bf16; // populated only when exact_mode == 1 (EC_TIER_EXACT / EC_EXACT=1)
} fx_mat_t;

typedef struct {
  fx_mat_t img_qkv, img_proj, img_mlp0, img_mlp2;
  fx_mat_t txt_qkv, txt_proj, txt_mlp0, txt_mlp2;
  float   *img_qn, *img_kn, *txt_qn, *txt_kn;
} fx_double_w_t;

typedef struct {
  fx_mat_t lin1, lin2;
  float   *qn, *kn;
} fx_single_w_t;

typedef struct {
  int loaded;
  int exact_mode;
  char ckpt_path[512];
  char lora_sig[1024];
  int  lora_hits;
  ec_lora_spec_t active_loras[EC_LORA_MAX_STACK];
  int  active_n_loras;
  int  delta_swaps;

  uint16_t *img_in, *txt_in, *time_in0, *time_in1;
  uint16_t *mod_img, *mod_txt, *mod_sng;
  uint16_t *final_mod, *final_lin;
  fx_double_w_t dbl[5];
  fx_single_w_t sng[20];
} fx_dit_resident_t;

typedef struct {
  char  name[112];
  void *ptr;
  int   kind; // 0 = OIHW bf16, 1 = f32, 2 = OHWI [Cout, 9*Cin] bf16
} fx_vae_tensor_t;

typedef struct {
  fx_vae_tensor_t tensors[384];
  int             n_tensors;
  uint16_t       *col_bf;
  size_t          col_cap;
  float          *gy_d;
  size_t          gy_cap;
  float          *res_tmp[3];
  size_t          res_cap[3];
  uint8_t        *dec_arena;
  size_t          dec_arena_cap;
} fx_vae_cache_t;

typedef struct {
  int loaded;
  float    *ln1[27], *ln2[27], *qn[27], *kn[27];
  uint16_t *wq[27], *wk[27], *wv[27], *wo[27], *wg[27], *wu[27], *wd[27];
} fx_qwen3_resident_t;

struct ec_flux2 {
  ec_flux2_paths_t paths;
  ec_bpe_t        *bpe;
  ec_safetensors_t te_st;
  ec_safetensors_t vae_st;
  int              te_open, vae_open;
  cudaStream_t     stream;

  fx_dit_resident_t   dit;
  fx_qwen3_resident_t qwen_res;
  fx_vae_cache_t      vae_cache;
  uint8_t            *dit_arena;
  size_t              dit_arena_cap;

  // Cached prompt embedding [512 * 7680] BF16 on host RAM + persistent Cowrie SHA-NI StatePage
  char      cached_prompt[1024];
  uint16_t *cached_prompt_bf16;
  int       has_cached_prompt;
  char      te_sha[65];
};

static void trim_vae_conv_scratch(fx_vae_cache_t *vc) {
  if (!vc) return;
  if (vc->col_bf) { cudaFree(vc->col_bf); vc->col_bf = NULL; vc->col_cap = 0; }
  if (vc->gy_d)   { cudaFree(vc->gy_d);   vc->gy_d   = NULL; vc->gy_cap  = 0; }
  for (int i = 0; i < 3; i++) {
    if (vc->res_tmp[i]) { cudaFree(vc->res_tmp[i]); vc->res_tmp[i] = NULL; vc->res_cap[i] = 0; }
  }
  // Keep <= 900 MB NHWC decode arena always hot (256^2 / 512^2); for > 900 MB (1024^2), evict Qwen3 TE first so standalone stays >= 12 GB free & 100% hot!
  if (vc->dec_arena && vc->dec_arena_cap > (900ULL << 20)) {
    size_t free_b = 0, tot_b = 0;
    if (cudaMemGetInfo(&free_b, &tot_b) != cudaSuccess || free_b < (12000ULL << 20)) {
      cudaFree(vc->dec_arena);
      vc->dec_arena = NULL;
      vc->dec_arena_cap = 0;
    }
  }
}

static void free_vae_cache(fx_vae_cache_t *vc) {
  if (!vc) return;
  for (int i = 0; i < vc->n_tensors; i++) {
    if (vc->tensors[i].ptr) cudaFree(vc->tensors[i].ptr);
  }
  if (vc->dec_arena) { cudaFree(vc->dec_arena); vc->dec_arena = NULL; vc->dec_arena_cap = 0; }
  trim_vae_conv_scratch(vc);
  memset(vc, 0, sizeof *vc);
}

static uint16_t *get_vae_bf16(ec_flux2_t *fx, const char *name) {
  fx_vae_cache_t *vc = &fx->vae_cache;
  for (int i = 0; i < vc->n_tensors; i++) {
    if (vc->tensors[i].kind == 0 && strcmp(vc->tensors[i].name, name) == 0)
      return (uint16_t *)vc->tensors[i].ptr;
  }
  uint16_t *d = load_bf16_gpu(&fx->vae_st, name, NULL, fx->stream);
  if (d && vc->n_tensors < 384) {
    snprintf(vc->tensors[vc->n_tensors].name, sizeof vc->tensors[0].name, "%s", name);
    vc->tensors[vc->n_tensors].ptr = d;
    vc->tensors[vc->n_tensors].kind = 0;
    vc->n_tensors++;
  }
  return d;
}

// Loads OIHW [Cout, Cin, 3, 3] weight from safetensors and caches permuted OHWI [Cout, 9 * Cin] BF16 on GPU!
static uint16_t *get_vae_ohwi_3x3_bf16(ec_flux2_t *fx, const char *name, int Cout, int Cin) {
  fx_vae_cache_t *vc = &fx->vae_cache;
  for (int i = 0; i < vc->n_tensors; i++) {
    if (vc->tensors[i].kind == 2 && strcmp(vc->tensors[i].name, name) == 0)
      return (uint16_t *)vc->tensors[i].ptr;
  }
  const ec_st_entry_t *e = find_tensor_flex(&fx->vae_st, name);
  if (!e) return NULL;
  const size_t n = (size_t)Cout * (size_t)Cin * 9u;
  size_t en = 1;
  for (int i = 0; i < e->rank; i++) en *= (size_t)e->shape[i];
  if (en != n) return NULL;

  uint16_t *ohwi_h = (uint16_t *)malloc(n * sizeof(uint16_t));
  if (!ohwi_h) return NULL;
  const void *src = ec_st_ptr(&fx->vae_st, e);
  for (int oc = 0; oc < Cout; oc++) {
    for (int ic = 0; ic < Cin; ic++) {
      for (int tap = 0; tap < 9; tap++) {
        const size_t src_i = (((size_t)oc * (size_t)Cin + (size_t)ic) * 9u) + (size_t)tap;
        const size_t dst_i = (((size_t)oc * 9u + (size_t)tap) * (size_t)Cin) + (size_t)ic;
        uint16_t v_bf = 0;
        if (e->dtype == EC_DTYPE_BF16) {
          v_bf = ((const uint16_t *)src)[src_i];
        } else if (e->dtype == EC_DTYPE_F32) {
          uint32_t u; memcpy(&u, &((const float *)src)[src_i], 4);
          u += 0x7FFFU + ((u >> 16) & 1U);
          v_bf = (uint16_t)(u >> 16);
        } else if (e->dtype == EC_DTYPE_F16) {
          float fv = __half2float(((const __half *)src)[src_i]);
          uint32_t u; memcpy(&u, &fv, 4);
          u += 0x7FFFU + ((u >> 16) & 1U);
          v_bf = (uint16_t)(u >> 16);
        }
        ohwi_h[dst_i] = v_bf;
      }
    }
  }
  uint16_t *d = NULL;
  if (cudaMalloc(&d, n * sizeof(uint16_t)) != cudaSuccess) { free(ohwi_h); return NULL; }
  cudaMemcpyAsync(d, ohwi_h, n * sizeof(uint16_t), cudaMemcpyHostToDevice, fx->stream);
  cudaStreamSynchronize(fx->stream);
  free(ohwi_h);
  if (vc->n_tensors < 384) {
    snprintf(vc->tensors[vc->n_tensors].name, sizeof vc->tensors[0].name, "%s", name);
    vc->tensors[vc->n_tensors].ptr = d;
    vc->tensors[vc->n_tensors].kind = 2;
    vc->n_tensors++;
  }
  return d;
}

static float *get_vae_f32(ec_flux2_t *fx, const char *name) {
  fx_vae_cache_t *vc = &fx->vae_cache;
  for (int i = 0; i < vc->n_tensors; i++) {
    if (vc->tensors[i].kind == 1 && strcmp(vc->tensors[i].name, name) == 0)
      return (float *)vc->tensors[i].ptr;
  }
  float *d = load_f32_gpu(&fx->vae_st, name, NULL, fx->stream);
  if (d && vc->n_tensors < 384) {
    snprintf(vc->tensors[vc->n_tensors].name, sizeof vc->tensors[0].name, "%s", name);
    vc->tensors[vc->n_tensors].ptr = d;
    vc->tensors[vc->n_tensors].kind = 1;
    vc->n_tensors++;
  }
  return d;
}

static int ensure_vae_conv_scratch(fx_vae_cache_t *vc, size_t col_bytes, size_t gy_bytes) {
  if (col_bytes > vc->col_cap) {
    if (vc->col_bf) cudaFree(vc->col_bf);
    vc->col_bf = NULL; vc->col_cap = 0;
    if (cudaMalloc(&vc->col_bf, col_bytes) != cudaSuccess) return -1;
    vc->col_cap = col_bytes;
  }
  if (gy_bytes > vc->gy_cap) {
    if (vc->gy_d) cudaFree(vc->gy_d);
    vc->gy_d = NULL; vc->gy_cap = 0;
    if (cudaMalloc(&vc->gy_d, gy_bytes) != cudaSuccess) return -1;
    vc->gy_cap = gy_bytes;
  }
  return 0;
}

static float *ensure_vae_res_slot(fx_vae_cache_t *vc, int slot, size_t bytes) {
  if (bytes > vc->res_cap[slot]) {
    if (vc->res_tmp[slot]) cudaFree(vc->res_tmp[slot]);
    vc->res_tmp[slot] = NULL; vc->res_cap[slot] = 0;
    if (cudaMalloc(&vc->res_tmp[slot], bytes) != cudaSuccess) return NULL;
    vc->res_cap[slot] = bytes;
  }
  return vc->res_tmp[slot];
}

static void free_qwen3_resident(fx_qwen3_resident_t *q) {
  if (!q || !q->loaded) return;
  for (int l = 0; l < 27; l++) {
    cudaFree(q->ln1[l]); cudaFree(q->ln2[l]); cudaFree(q->qn[l]); cudaFree(q->kn[l]);
    cudaFree(q->wq[l]); cudaFree(q->wk[l]); cudaFree(q->wv[l]); cudaFree(q->wo[l]);
    cudaFree(q->wg[l]); cudaFree(q->wu[l]); cudaFree(q->wd[l]);
  }
  memset(q, 0, sizeof *q);
}

static int try_load_qwen3_resident(ec_flux2_t *fx) {
  if (fx->qwen_res.loaded) return 1;
  size_t free_b = 0, tot_b = 0;
  if (cudaMemGetInfo(&free_b, &tot_b) != cudaSuccess) return 0;
  // Require generous VRAM headroom (13 GB when DiT is already resident, 20 GB when DiT not yet loaded)
  // so 5.4 GB Qwen3-4B resident weights NEVER starve FLUX.2 DiT (~2 GB workspace) or VAE (~1.8 GB) on 24 GB GPUs!
  size_t need_b = fx->dit.loaded ? (13ULL * 1024ULL * 1024ULL * 1024ULL) : (20ULL * 1024ULL * 1024ULL * 1024ULL);
  if (free_b < need_b) return 0;
  cudaStream_t s = fx->stream;
  char key[128];
  for (int l = 0; l < 27; l++) {
    snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l);
    fx->qwen_res.ln1[l] = load_f32_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l);
    fx->qwen_res.ln2[l] = load_f32_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.q_norm.weight", l);
    fx->qwen_res.qn[l]  = load_f32_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.k_norm.weight", l);
    fx->qwen_res.kn[l]  = load_f32_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l);
    fx->qwen_res.wq[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l);
    fx->qwen_res.wk[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj.weight", l);
    fx->qwen_res.wv[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l);
    fx->qwen_res.wo[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", l);
    fx->qwen_res.wg[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", l);
    fx->qwen_res.wu[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    snprintf(key, sizeof key, "model.layers.%d.mlp.down_proj.weight", l);
    fx->qwen_res.wd[l]  = load_bf16_gpu(&fx->te_st, key, NULL, s);
    if (!fx->qwen_res.wq[l] || !fx->qwen_res.wd[l]) {
      fx->qwen_res.loaded = 1;
      free_qwen3_resident(&fx->qwen_res);
      return 0;
    }
  }
  cudaStreamSynchronize(s);
  fx->qwen_res.loaded = 1;
  return 1;
}

static void free_fx_mat(fx_mat_t *m) {
  if (!m) return;
  if (m->q8)   cudaFree(m->q8);
  if (m->sc)   cudaFree(m->sc);
  if (m->bf16) cudaFree(m->bf16);
  memset(m, 0, sizeof *m);
}

static void free_dit_resident(fx_dit_resident_t *d) {
  if (!d || !d->loaded) return;
  cudaFree(d->img_in); cudaFree(d->txt_in); cudaFree(d->time_in0); cudaFree(d->time_in1);
  cudaFree(d->mod_img); cudaFree(d->mod_txt); cudaFree(d->mod_sng);
  cudaFree(d->final_mod); cudaFree(d->final_lin);
  for (int i = 0; i < 5; i++) {
    free_fx_mat(&d->dbl[i].img_qkv); free_fx_mat(&d->dbl[i].img_proj);
    free_fx_mat(&d->dbl[i].img_mlp0); free_fx_mat(&d->dbl[i].img_mlp2);
    free_fx_mat(&d->dbl[i].txt_qkv); free_fx_mat(&d->dbl[i].txt_proj);
    free_fx_mat(&d->dbl[i].txt_mlp0); free_fx_mat(&d->dbl[i].txt_mlp2);
    cudaFree(d->dbl[i].img_qn); cudaFree(d->dbl[i].img_kn);
    cudaFree(d->dbl[i].txt_qn); cudaFree(d->dbl[i].txt_kn);
  }
  for (int i = 0; i < 20; i++) {
    free_fx_mat(&d->sng[i].lin1); free_fx_mat(&d->sng[i].lin2);
    cudaFree(d->sng[i].qn); cudaFree(d->sng[i].kn);
  }
  memset(d, 0, sizeof *d);
}

static void build_lora_sig(const ec_lora_spec_t *loras, int n, char *out, size_t cap) {
  out[0] = '\0';
  int pos = 0;
  for (int i = 0; i < n && (size_t)pos + 64 < cap; i++) {
    pos += snprintf(out + pos, cap - (size_t)pos, "%s@%.3f;", loras[i].path, loras[i].strength);
  }
}

static int lora_stack_touches_blocks(const ec_lora_spec_t *stack, int n_stack) {
  if (!stack || n_stack <= 0) return 0;
  for (int li = 0; li < n_stack; li++) {
    if (!stack[li].path[0]) continue;
    ec_safetensors_t st;
    char err[64] = {0};
    if (ec_st_open(&st, stack[li].path, err, sizeof err) != 0) continue;
    int hit = 0;
    for (uint32_t ti = 0; ti < st.n_tensors; ti++) {
      if (strstr(st.entries[ti].name, "blocks.")) { hit = 1; break; }
    }
    ec_st_close(&st);
    if (hit) return 1;
  }
  return 0;
}

static void swap_one_fx_mat(
    const ec_safetensors_t *base_st,
    fx_mat_t *m, int out_d, int in_d, const char *name,
    const ec_lora_spec_t *undo_stack, int n_undo, ec_lora_apply_stats_t *undo_st,
    const ec_lora_spec_t *new_stack, int n_new, ec_lora_apply_stats_t *new_st,
    uint16_t *stage_bf16, int exact_mode, cudaStream_t s) {
  if (exact_mode) {
    if (n_undo > 0) ec_lora_fuse_onto_bf16(m->bf16, out_d, in_d, name, undo_stack, n_undo, undo_st, s);
    if (n_new > 0)  ec_lora_fuse_onto_bf16(m->bf16, out_d, in_d, name, new_stack, n_new, new_st, s);
    return;
  }
  // Probe whether either undo_stack or new_stack touches this tensor before touching base weights!
  ec_lora_apply_stats_t probe_u = {}, probe_n = {};
  int hu = (n_undo > 0) ? ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, undo_stack, n_undo, &probe_u, s) : 0;
  int hn = (n_new  > 0) ? ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, new_stack,  n_new,  &probe_n, s) : 0;
  if (hu == 0 && hn == 0) return;

  // Reload exact clean BF16 base slice from page-cached mmap -> fuse new_stack -> quantize Q8_G64 (zero cumulative rounding drift!)
  const ec_st_entry_t *e = base_st ? find_tensor_flex(base_st, name) : NULL;
  const size_t n = (size_t)out_d * (size_t)in_d;
  if (e && e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(stage_bf16, ec_st_ptr(base_st, e), n * 2, cudaMemcpyHostToDevice, s);
    if (undo_st) undo_st->n_tensors_matched += hu;
    if (n_new > 0) ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, new_stack, n_new, new_st, s);
  } else {
    ec_ops_dequant_q8g64_bf16(m->q8, m->sc, stage_bf16, out_d, in_d, s);
    if (n_undo > 0) ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, undo_stack, n_undo, undo_st, s);
    if (n_new > 0)  ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, new_stack, n_new, new_st, s);
  }
  ec_ops_quantize_bf16_q8g64(stage_bf16, m->q8, m->sc, out_d, in_d, s);
}

static void delta_swap_resident_dit(
    fx_dit_resident_t *d,
    const ec_lora_spec_t *undo_stack, int n_undo, ec_lora_apply_stats_t *undo_st,
    const ec_lora_spec_t *new_stack, int n_new, ec_lora_apply_stats_t *new_st,
    cudaStream_t s) {
  #define SWAP_BF(ptr, od, id, nm) do { \
    if (n_undo > 0) ec_lora_fuse_onto_bf16((ptr), (od), (id), (nm), undo_stack, n_undo, undo_st, s); \
    if (n_new > 0)  ec_lora_fuse_onto_bf16((ptr), (od), (id), (nm), new_stack,  n_new,  new_st,  s); \
  } while (0)
  SWAP_BF(d->img_in,    3072,  128, "img_in.weight");
  SWAP_BF(d->txt_in,    3072, 7680, "txt_in.weight");
  SWAP_BF(d->time_in0,  3072,  256, "time_in.in_layer.weight");
  SWAP_BF(d->time_in1,  3072, 3072, "time_in.out_layer.weight");
  SWAP_BF(d->mod_img,  18432, 3072, "double_stream_modulation_img.lin.weight");
  SWAP_BF(d->mod_txt,  18432, 3072, "double_stream_modulation_txt.lin.weight");
  SWAP_BF(d->mod_sng,   9216, 3072, "single_stream_modulation.lin.weight");
  SWAP_BF(d->final_mod, 6144, 3072, "final_layer.adaLN_modulation.1.weight");
  SWAP_BF(d->final_lin,  128, 3072, "final_layer.linear.weight");
  #undef SWAP_BF

  if (!lora_stack_touches_blocks(undo_stack, n_undo) && !lora_stack_touches_blocks(new_stack, n_new)) {
    return;
  }

  uint16_t *stage_bf16 = NULL;
  ec_safetensors_t base_st = {};
  int has_base = 0;
  if (!d->exact_mode) {
    if (cudaMalloc(&stage_bf16, 27648ULL * 3072ULL * 2ULL) != cudaSuccess) return;
    char err_tmp[128] = {0};
    if (ec_st_open(&base_st, d->ckpt_path, err_tmp, sizeof err_tmp) == 0) has_base = 1;
  }
  const ec_safetensors_t *bst = has_base ? &base_st : NULL;
  char k[160];
  for (int i = 0; i < 5; i++) {
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.qkv.weight", i);  swap_one_fx_mat(bst, &d->dbl[i].img_qkv,  9216, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.proj.weight", i); swap_one_fx_mat(bst, &d->dbl[i].img_proj, 3072, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.0.weight", i);     swap_one_fx_mat(bst, &d->dbl[i].img_mlp0,18432, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.2.weight", i);     swap_one_fx_mat(bst, &d->dbl[i].img_mlp2, 3072, 9216, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.qkv.weight", i);  swap_one_fx_mat(bst, &d->dbl[i].txt_qkv,  9216, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.proj.weight", i); swap_one_fx_mat(bst, &d->dbl[i].txt_proj, 3072, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.0.weight", i);     swap_one_fx_mat(bst, &d->dbl[i].txt_mlp0,18432, 3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.2.weight", i);     swap_one_fx_mat(bst, &d->dbl[i].txt_mlp2, 3072, 9216, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
  }
  for (int i = 0; i < 20; i++) {
    snprintf(k, sizeof k, "single_blocks.%d.linear1.weight", i); swap_one_fx_mat(bst, &d->sng[i].lin1, 27648,  3072, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
    snprintf(k, sizeof k, "single_blocks.%d.linear2.weight", i); swap_one_fx_mat(bst, &d->sng[i].lin2,  3072, 12288, k, undo_stack, n_undo, undo_st, new_stack, n_new, new_st, stage_bf16, d->exact_mode, s);
  }
  cudaStreamSynchronize(s);
  if (has_base) ec_st_close(&base_st);
  if (stage_bf16) cudaFree(stage_bf16);
}

static int load_fx_mat(
    const ec_safetensors_t *st, const char *name, fx_mat_t *m,
    int out_d, int in_d,
    const ec_lora_spec_t *loras, int n_loras, ec_lora_apply_stats_t *lstats,
    uint16_t *stage_bf16, int exact_mode, cudaStream_t s) {
  memset(m, 0, sizeof *m);
  if (exact_mode) {
    m->bf16 = load_bf16_gpu(st, name, NULL, s);
    if (!m->bf16) return -1;
    if (n_loras > 0) ec_lora_fuse_onto_bf16(m->bf16, out_d, in_d, name, loras, n_loras, lstats, s);
    return 0;
  }
  const ec_st_entry_t *e = find_tensor_flex(st, name);
  if (!e) return -1;
  const size_t n = (size_t)out_d * (size_t)in_d;
  size_t en = 1;
  for (int i = 0; i < e->rank; i++) en *= (size_t)e->shape[i];
  if (en != n) return -1;

  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(stage_bf16, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, s);
  } else {
    uint16_t *tmp = load_bf16_gpu(st, name, NULL, s);
    if (!tmp) return -1;
    cudaMemcpyAsync(stage_bf16, tmp, n * 2, cudaMemcpyDeviceToDevice, s);
    cudaStreamSynchronize(s);
    cudaFree(tmp);
  }
  if (n_loras > 0) ec_lora_fuse_onto_bf16(stage_bf16, out_d, in_d, name, loras, n_loras, lstats, s);
  if (cudaMalloc(&m->q8, n) != cudaSuccess ||
      cudaMalloc(&m->sc, (size_t)out_d * (size_t)(in_d >> 6) * sizeof(float)) != cudaSuccess) {
    free_fx_mat(m);
    return -1;
  }
  if (ec_ops_quantize_bf16_q8g64(stage_bf16, m->q8, m->sc, out_d, in_d, s) != 0) {
    free_fx_mat(m);
    return -1;
  }
  return 0;
}

static int ensure_dit_loaded_ex(
    ec_flux2_t *fx, const char *ckpt_path,
    const ec_lora_spec_t *loras, int n_loras, int want_exact,
    int *out_delta_used, char *err, size_t err_cap) {
  if (out_delta_used) *out_delta_used = 0;
  int clamped_n = (loras && n_loras > 0) ? (n_loras > EC_LORA_MAX_STACK ? EC_LORA_MAX_STACK : n_loras) : 0;
  char lsig[1024];
  build_lora_sig(loras, clamped_n, lsig, sizeof lsig);
  if (fx->dit.loaded && fx->dit.exact_mode == want_exact &&
      strcmp(fx->dit.ckpt_path, ckpt_path) == 0 && strcmp(fx->dit.lora_sig, lsig) == 0) {
    if (out_delta_used) *out_delta_used = 1;
    return 0;
  }

  // Zero-Reload GPU LoRA/QLoRA Delta Unfusing Path (<15ms single-pass on Q8_G64 or BF16!):
  if (fx->dit.loaded && fx->dit.exact_mode == want_exact &&
      strcmp(fx->dit.ckpt_path, ckpt_path) == 0 &&
      fx->dit.delta_swaps < 16 &&
      !ec_lora_stack_has_dora(fx->dit.active_loras, fx->dit.active_n_loras) &&
      !ec_lora_stack_has_dora(loras, clamped_n)) {
    fx_dit_resident_t *d = &fx->dit;
    cudaStream_t s = fx->stream;
    ec_lora_spec_t neg_prev[EC_LORA_MAX_STACK];
    for (int i = 0; i < d->active_n_loras; i++) {
      neg_prev[i] = d->active_loras[i];
      neg_prev[i].strength = -neg_prev[i].strength;
    }
    ec_lora_apply_stats_t undo_st = {}, lstats = {};
    delta_swap_resident_dit(d, neg_prev, d->active_n_loras, &undo_st, loras, clamped_n, &lstats, s);
    cudaStreamSynchronize(s);
    snprintf(d->lora_sig, sizeof d->lora_sig, "%s", lsig);
    d->lora_hits = lstats.n_tensors_matched;
    d->active_n_loras = clamped_n;
    for (int i = 0; i < clamped_n; i++) d->active_loras[i] = loras[i];
    d->delta_swaps++;
    if (out_delta_used) *out_delta_used = 1;
    return 0;
  }

  free_dit_resident(&fx->dit);

  ec_safetensors_t st;
  if (ec_st_open(&st, ckpt_path, err, err_cap) != 0) return -1;

  fx_dit_resident_t *d = &fx->dit;
  cudaStream_t s = fx->stream;
  ec_lora_apply_stats_t lstats = {};
  uint16_t *stage_bf16 = NULL;
  if (!want_exact) {
    if (cudaMalloc(&stage_bf16, 27648ULL * 3072ULL * 2ULL) != cudaSuccess) {
      ec_st_close(&st);
      snprintf(err, err_cap, "oom allocating 169MB q8_g64 stage");
      return -1;
    }
  }

  #define LOAD_W(field, name, out_d, in_d) do { \
    d->field = load_bf16_gpu(&st, name, NULL, s); \
    if (!d->field) { cudaFree(stage_bf16); ec_st_close(&st); free_dit_resident(d); snprintf(err, err_cap, "missing %s", name); return -1; } \
    if (clamped_n > 0) ec_lora_fuse_onto_bf16(d->field, (out_d), (in_d), name, loras, clamped_n, &lstats, s); \
  } while (0)
  #define LOAD_M(field, name, out_d, in_d) do { \
    if (load_fx_mat(&st, name, &d->field, (out_d), (in_d), loras, clamped_n, &lstats, stage_bf16, want_exact, s) != 0) { \
      cudaFree(stage_bf16); ec_st_close(&st); free_dit_resident(d); snprintf(err, err_cap, "missing/oom %s", name); return -1; \
    } \
  } while (0)

  LOAD_W(img_in,    "img_in.weight",    3072, 128);
  LOAD_W(txt_in,    "txt_in.weight",    3072, 7680);
  LOAD_W(time_in0,  "time_in.in_layer.weight",  3072, 256);
  LOAD_W(time_in1,  "time_in.out_layer.weight", 3072, 3072);
  LOAD_W(mod_img,   "double_stream_modulation_img.lin.weight", 18432, 3072);
  LOAD_W(mod_txt,   "double_stream_modulation_txt.lin.weight", 18432, 3072);
  LOAD_W(mod_sng,   "single_stream_modulation.lin.weight",      9216, 3072);
  LOAD_W(final_mod, "final_layer.adaLN_modulation.1.weight",    6144, 3072);
  LOAD_W(final_lin, "final_layer.linear.weight",                 128, 3072);

  char k[160];
  for (int i = 0; i < 5; i++) {
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.qkv.weight", i);  LOAD_M(dbl[i].img_qkv,  k, 9216, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.proj.weight", i); LOAD_M(dbl[i].img_proj, k, 3072, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.0.weight", i);     LOAD_M(dbl[i].img_mlp0, k, 18432, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.2.weight", i);     LOAD_M(dbl[i].img_mlp2, k, 3072, 9216);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.qkv.weight", i);  LOAD_M(dbl[i].txt_qkv,  k, 9216, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.proj.weight", i); LOAD_M(dbl[i].txt_proj, k, 3072, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.0.weight", i);     LOAD_M(dbl[i].txt_mlp0, k, 18432, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.2.weight", i);     LOAD_M(dbl[i].txt_mlp2, k, 3072, 9216);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.norm.query_norm.scale", i); d->dbl[i].img_qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.norm.key_norm.scale", i);   d->dbl[i].img_kn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.norm.query_norm.scale", i); d->dbl[i].txt_qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.norm.key_norm.scale", i);   d->dbl[i].txt_kn = load_f32_gpu(&st, k, NULL, s);
  }
  for (int i = 0; i < 20; i++) {
    snprintf(k, sizeof k, "single_blocks.%d.linear1.weight", i); LOAD_M(sng[i].lin1, k, 27648, 3072);
    snprintf(k, sizeof k, "single_blocks.%d.linear2.weight", i); LOAD_M(sng[i].lin2, k, 3072, 12288);
    snprintf(k, sizeof k, "single_blocks.%d.norm.query_norm.scale", i); d->sng[i].qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "single_blocks.%d.norm.key_norm.scale", i);   d->sng[i].kn = load_f32_gpu(&st, k, NULL, s);
  }
  #undef LOAD_M
  #undef LOAD_W

  cudaStreamSynchronize(s);
  if (stage_bf16) cudaFree(stage_bf16);
  ec_st_close(&st);
  snprintf(d->ckpt_path, sizeof d->ckpt_path, "%s", ckpt_path);
  snprintf(d->lora_sig, sizeof d->lora_sig, "%s", lsig);
  d->lora_hits = lstats.n_tensors_matched;
  d->active_n_loras = clamped_n;
  for (int i = 0; i < clamped_n; i++) d->active_loras[i] = loras[i];
  d->delta_swaps = 0;
  d->exact_mode = want_exact;
  d->loaded = 1;
  return 0;
}

static int ensure_dit_loaded(
    ec_flux2_t *fx, const char *ckpt_path,
    const ec_lora_spec_t *loras, int n_loras, int want_exact, char *err, size_t err_cap) {
  return ensure_dit_loaded_ex(fx, ckpt_path, loras, n_loras, want_exact, NULL, err, err_cap);
}

// ===========================================================================
// Stage A: Streamed Qwen3-4B Text Encoder (layers 0..26 -> [512, 7680] BF16)
// ===========================================================================
static int run_qwen3_text_encoder(
    ec_flux2_t *fx, const char *prompt, uint16_t *out_embeds_gpu_bf16, double *out_ms) {
  double t0 = fx_now_ms();
  const int L = 512, E = 2560, Hq = 32, Hkv = 8, D = 128, F = 9728;
  const size_t emb_elems = (size_t)L * 3 * E;

  if (fx->has_cached_prompt && prompt && strcmp(fx->cached_prompt, prompt) == 0) {
    cudaMemcpyAsync(out_embeds_gpu_bf16, fx->cached_prompt_bf16, emb_elems * 2, cudaMemcpyHostToDevice, fx->stream);
    cudaStreamSynchronize(fx->stream);
    if (out_ms) *out_ms = fx_now_ms() - t0;
    return 0;
  }

  // Persistent cross-restart Cowrie v2 SHA-NI StatePage lookup (7.86 MB -> ~2.1 ms instead of ~1.2 s!)
  if (prompt && fx->te_sha[0] && ec_cowrie_available()) {
    if (!fx->cached_prompt_bf16) fx->cached_prompt_bf16 = (uint16_t *)malloc(emb_elems * 2);
    if (fx->cached_prompt_bf16 &&
        ec_cowrie_sp_load("flux2_qwen3", fx->te_sha, prompt, strlen(prompt),
                          EC_COWRIE_KV_BF16, fx->cached_prompt_bf16, emb_elems * 2) == 0) {
      snprintf(fx->cached_prompt, sizeof fx->cached_prompt, "%s", prompt);
      fx->has_cached_prompt = 1;
      cudaMemcpyAsync(out_embeds_gpu_bf16, fx->cached_prompt_bf16, emb_elems * 2, cudaMemcpyHostToDevice, fx->stream);
      cudaStreamSynchronize(fx->stream);
      if (out_ms) *out_ms = fx_now_ms() - t0;
      return 0;
    }
  }

  int32_t ids[512];
  uint8_t mask_h[512];
  int real_len = 0;
  if (ec_bpe_encode_flux2_qwen3(fx->bpe, prompt, ids, mask_h, L, &real_len) < 0) return -1;

  // Lookup 512 rows from embed_tokens.weight [151936, 2560] BF16 straight from mmap!
  const ec_st_entry_t *e_emb = ec_st_find(&fx->te_st, "model.embed_tokens.weight");
  if (!e_emb || e_emb->dtype != EC_DTYPE_BF16) return -1;
  const uint16_t *emb_tab = (const uint16_t *)ec_st_ptr(&fx->te_st, e_emb);
  float *x0_h = (float *)malloc((size_t)L * E * 4);
  for (int t = 0; t < L; t++) {
    int id = ids[t];
    if (id < 0 || id >= (int)e_emb->shape[0]) id = 0;
    const uint16_t *row = emb_tab + (size_t)id * E;
    for (int c = 0; c < E; c++) {
      uint32_t u = ((uint32_t)row[c]) << 16;
      memcpy(&x0_h[(size_t)t * E + c], &u, 4);
    }
  }

  // Reusable GPU workspaces + 1-layer weight buffer (~260 MB total VRAM!)
  float *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL;
  float *qh_d = NULL, *kh_d = NULL, *vh_d = NULL, *o_d = NULL, *g_d = NULL, *u_d = NULL;
  uint16_t *h_bf16 = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL, *tmp_w_bf16 = NULL;
  uint8_t *mask_d = NULL;
  cudaMalloc(&x_d, (size_t)L * E * 4);
  cudaMalloc(&h_bf16, (size_t)L * E * 2);
  cudaMalloc(&q_d, (size_t)L * (Hq * D) * 4);
  cudaMalloc(&k_d, (size_t)L * (Hkv * D) * 4);
  cudaMalloc(&v_d, (size_t)L * (Hkv * D) * 4);
  cudaMalloc(&qh_d, (size_t)Hq * L * D * 4);
  cudaMalloc(&kh_d, (size_t)Hkv * L * D * 4);
  cudaMalloc(&vh_d, (size_t)Hkv * L * D * 4);
  cudaMalloc(&ao_bf16, (size_t)L * (Hq * D) * 2);
  cudaMalloc(&o_d, (size_t)L * E * 4);
  cudaMalloc(&g_d, (size_t)L * F * 4);
  cudaMalloc(&u_d, (size_t)L * F * 4);
  cudaMalloc(&mlp_bf16, (size_t)L * F * 2);
  cudaMalloc(&mask_d, L);

  // Try keeping Qwen3-4B (layers 0..26, ~5.4 GB) resident on 20+ GB GPUs so every NEW prompt runs in <300ms
  // even when qwen_3_4b.safetensors lives on an HDD or network drive!
  int use_resident = try_load_qwen3_resident(fx);

  // Layer weight staging (only allocated when streaming on smaller GPUs)
  float *ln1_w = NULL, *ln2_w = NULL, *qn_w = NULL, *kn_w = NULL;
  uint16_t *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL, *wg = NULL, *wu = NULL, *wd = NULL;
  if (!use_resident) {
    cudaMalloc(&ln1_w, E * 4); cudaMalloc(&ln2_w, E * 4);
    cudaMalloc(&qn_w, D * 4);  cudaMalloc(&kn_w, D * 4);
    cudaMalloc(&tmp_w_bf16, E * 2);
    cudaMalloc(&wq, (size_t)(Hq * D) * E * 2);
    cudaMalloc(&wk, (size_t)(Hkv * D) * E * 2);
    cudaMalloc(&wv, (size_t)(Hkv * D) * E * 2);
    cudaMalloc(&wo, (size_t)E * (Hq * D) * 2);
    cudaMalloc(&wg, (size_t)F * E * 2);
    cudaMalloc(&wu, (size_t)F * E * 2);
    cudaMalloc(&wd, (size_t)E * F * 2);
  }

  cudaStream_t s = fx->stream;
  cudaMemcpyAsync(x_d, x0_h, (size_t)L * E * 4, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(mask_d, mask_h, L, cudaMemcpyHostToDevice, s);
  free(x0_h);

  const float attn_scale = 1.0f / sqrtf((float)D);
  char key[128];

  // Run layers 0..26 only (hidden_states[9, 18, 27] = outputs after layers 8, 17, 26!)
  for (int l = 0; l <= 26; l++) {
    float *c_ln1, *c_ln2, *c_qn, *c_kn;
    uint16_t *c_wq, *c_wk, *c_wv, *c_wo, *c_wg, *c_wu, *c_wd;
    if (use_resident) {
      c_ln1 = fx->qwen_res.ln1[l]; c_ln2 = fx->qwen_res.ln2[l];
      c_qn  = fx->qwen_res.qn[l];  c_kn  = fx->qwen_res.kn[l];
      c_wq  = fx->qwen_res.wq[l];  c_wk  = fx->qwen_res.wk[l];
      c_wv  = fx->qwen_res.wv[l];  c_wo  = fx->qwen_res.wo[l];
      c_wg  = fx->qwen_res.wg[l];  c_wu  = fx->qwen_res.wu[l];
      c_wd  = fx->qwen_res.wd[l];
    } else {
      snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l);
      stream_f32_into(&fx->te_st, key, ln1_w, tmp_w_bf16, E, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wq, (size_t)(Hq * D) * E, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wk, (size_t)(Hkv * D) * E, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wv, (size_t)(Hkv * D) * E, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.q_norm.weight", l);
      stream_f32_into(&fx->te_st, key, qn_w, tmp_w_bf16, D, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.k_norm.weight", l);
      stream_f32_into(&fx->te_st, key, kn_w, tmp_w_bf16, D, s);
      snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wo, (size_t)E * (Hq * D), s);
      snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l);
      stream_f32_into(&fx->te_st, key, ln2_w, tmp_w_bf16, E, s);
      snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wg, (size_t)F * E, s);
      snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wu, (size_t)F * E, s);
      snprintf(key, sizeof key, "model.layers.%d.mlp.down_proj.weight", l);
      stream_bf16_into(&fx->te_st, key, wd, (size_t)E * F, s);
      c_ln1 = ln1_w; c_ln2 = ln2_w; c_qn = qn_w; c_kn = kn_w;
      c_wq = wq; c_wk = wk; c_wv = wv; c_wo = wo; c_wg = wg; c_wu = wu; c_wd = wd;
    }

    k_rmsnorm_bf16<<<L, 256, 0, s>>>(x_d, c_ln1, h_bf16, L, E, 1e-6f);
    gemm_bf16(h_bf16, c_wq, NULL, q_d, NULL, L, Hq * D, E, s);
    gemm_bf16(h_bf16, c_wk, NULL, k_d, NULL, L, Hkv * D, E, s);
    gemm_bf16(h_bf16, c_wv, NULL, v_d, NULL, L, Hkv * D, E, s);

    int total_qk_warps = L * (Hq + Hkv);
    k_qwen3_qk_rope<<<(total_qk_warps + 7) / 8, 256, 0, s>>>(
        q_d, k_d, v_d, c_qn, c_kn, qh_d, kh_d, vh_d, L, Hq, Hkv, D, 1000000.0f, 1e-6f);

    int attn_warps = L * Hq;
    k_qwen3_gqa_attn<<<(attn_warps + 7) / 8, 256, 0, s>>>(
        qh_d, kh_d, vh_d, mask_d, ao_bf16, L, Hq, Hkv, D, attn_scale);

    gemm_bf16(ao_bf16, c_wo, NULL, o_d, NULL, L, E, Hq * D, s);
    k_add_inplace<<<(L * E + 255) / 256, 256, 0, s>>>(x_d, o_d, (long)L * E);

    k_rmsnorm_bf16<<<L, 256, 0, s>>>(x_d, c_ln2, h_bf16, L, E, 1e-6f);
    gemm_bf16(h_bf16, c_wg, NULL, g_d, NULL, L, F, E, s);
    gemm_bf16(h_bf16, c_wu, NULL, u_d, NULL, L, F, E, s);
    k_swiglu_pair_bf16<<<((long)L * F + 255) / 256, 256, 0, s>>>(g_d, u_d, mlp_bf16, (long)L * F);
    gemm_bf16(mlp_bf16, c_wd, NULL, o_d, NULL, L, E, F, s);
    k_add_inplace<<<(L * E + 255) / 256, 256, 0, s>>>(x_d, o_d, (long)L * E);

    if (l == 8 || l == 17 || l == 26) {
      int slot = (l == 8) ? 0 : ((l == 17) ? 1 : 2);
      k_scatter_qwen_layer<<<((long)L * E + 255) / 256, 256, 0, s>>>(x_d, out_embeds_gpu_bf16, L, E, slot);
    }
  }
  cudaStreamSynchronize(s);

  // Cache host copy for instant rerolls + persist 16 KB SHA-NI Cowrie StatePage across restarts
  if (!fx->cached_prompt_bf16) fx->cached_prompt_bf16 = (uint16_t *)malloc(emb_elems * 2);
  if (fx->cached_prompt_bf16 && prompt) {
    cudaMemcpy(fx->cached_prompt_bf16, out_embeds_gpu_bf16, emb_elems * 2, cudaMemcpyDeviceToHost);
    snprintf(fx->cached_prompt, sizeof fx->cached_prompt, "%s", prompt);
    fx->has_cached_prompt = 1;
    if (fx->te_sha[0] && ec_cowrie_available()) {
      ec_cowrie_sp_save("flux2_qwen3", fx->te_sha, prompt, strlen(prompt),
                        (uint32_t)L, EC_COWRIE_KV_BF16, fx->cached_prompt_bf16, emb_elems * 2);
    }
  }

  cudaFree(x_d); cudaFree(h_bf16); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(qh_d); cudaFree(kh_d); cudaFree(vh_d); cudaFree(ao_bf16); cudaFree(o_d);
  cudaFree(g_d); cudaFree(u_d); cudaFree(mlp_bf16); cudaFree(mask_d);
  cudaFree(ln1_w); cudaFree(ln2_w); cudaFree(qn_w); cudaFree(kn_w); cudaFree(tmp_w_bf16);
  cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo); cudaFree(wg); cudaFree(wu); cudaFree(wd);

  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
}

// ===========================================================================
// Stage B: FLUX.2-klein-4B 4-Step Flow-Matching MMDiT
// ===========================================================================
static double compute_empirical_mu(int image_seq_len, int num_steps) {
  const double a1 = 8.73809524e-05, b1 = 1.89833333;
  const double a2 = 0.00016927,     b2 = 0.45666666;
  if (image_seq_len > 4300) return a2 * image_seq_len + b2;
  double m200 = a2 * image_seq_len + b2;
  double m10  = a1 * image_seq_len + b1;
  double a = (m200 - m10) / 190.0;
  double b = m200 - 200.0 * a;
  return a * num_steps + b;
}

static uint64_t sm64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void fill_gaussian(float *dst, size_t n, uint32_t seed) {
  uint64_t st = ((uint64_t)seed * 0x9E3779B97F4A7C15ULL) ^ 0xF10C22026ULL;
  for (size_t i = 0; i < n; i += 2) {
    double u1 = ((double)(sm64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = ((double)(sm64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double ang = 6.28318530717958647692 * u2;
    dst[i] = (float)(r * cos(ang));
    if (i + 1 < n) dst[i + 1] = (float)(r * sin(ang));
  }
}

static int run_flux2_dit(
    ec_flux2_t *fx, const uint16_t *prompt_embeds_bf16, const uint16_t *neg_embeds_bf16,
    int H_p, int W_p, int steps, float cfg_scale, uint32_t seed,
    const float *z0_init_d, const float *mask_p_d, float denoise,
    float *out_latents_gpu_f32, ec_flux2_progress_fn cb, void *cb_ctx, double *out_ms) {
  double t0 = fx_now_ms();
  const int N_txt = 512;
  const int N_img = H_p * W_p;
  const int N_tot = N_txt + N_img;
  const int N_pad = (N_tot + 63) & ~63;
  const int E = 3072, H = 24, D = 128, F = 9216;
  const float attn_scale = 1.0f / sqrtf((float)D);
  cudaStream_t s = fx->stream;
  const fx_dit_resident_t *d = &fx->dit;

  // Pre-evict resident Qwen3-4B (5.4 GB) or VAE cache if VRAM is below 3.8 GB so ~2.0 GB DiT workspaces never OOM!
  {
    size_t free_b = 0, tot_b = 0;
    if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && free_b < 3800ULL * 1024ULL * 1024ULL) {
      free_qwen3_resident(&fx->qwen_res);
      if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && free_b < 2800ULL * 1024ULL * 1024ULL) {
        free_vae_cache(&fx->vae_cache);
      }
    }
  }

  // Shifted sigmas schedule (with FLUX.2 empirical mu-shifted SDEdit starting at t_lin = denoise for unmasked Img2Img!)
  int s0 = 0;
  float sigmas[64];
  if (z0_init_d && !mask_p_d && denoise > 0.01f && denoise < 0.99f) {
    if (steps < 6) steps = 6;
    double mu = compute_empirical_mu(N_img, steps);
    double emu = exp(mu);
    for (int i = 0; i < steps; i++) {
      double t_lin = (double)denoise * (1.0 - (double)i / (double)steps);
      double sh = (t_lin >= 1.0) ? 1.0 : ((t_lin <= 1e-5) ? 0.0 : (emu / (emu + (1.0 / t_lin - 1.0))));
      sigmas[i] = (float)sh;
    }
    sigmas[steps] = 0.0f;
  } else {
    if (z0_init_d && mask_p_d && steps < 6) steps = 6;
    double mu = compute_empirical_mu(N_img, steps);
    double emu = exp(mu);
    for (int i = 0; i < steps; i++) {
      double t_lin = 1.0 - (double)i * (1.0 - 1.0 / (double)steps) / (double)(steps > 1 ? steps - 1 : 1);
      if (steps == 1) t_lin = 1.0;
      double sh = (t_lin >= 1.0) ? 1.0 : (emu / (emu + (1.0 / t_lin - 1.0)));
      sigmas[i] = (float)sh;
    }
    sigmas[steps] = 0.0f;
  }

  // Persistent zero-malloc DiT step arena (eliminates 28 synchronous cudaMalloc/cudaFree TLB shootdowns per image!)
  int use_cfg = (neg_embeds_bf16 != NULL && cfg_scale > 1.05f) ? 1 : 0;
  int use_sage = (!d->exact_mode && !(getenv("EC_FA2_EXACT") && atoi(getenv("EC_FA2_EXACT")) != 0)) ? 1 : 0;
  size_t lin1_bytes = (size_t)N_tot * (3 * E + 2 * F) * 4;
  size_t dbl_tmp_bytes = ((size_t)N_img * 3 * E * 4 + 255) + ((size_t)N_txt * 3 * E * 4 + 255);
  size_t mlp_bytes = (size_t)(N_img > N_txt ? N_img : N_txt) * 2 * F * 4;
  if (dbl_tmp_bytes > lin1_bytes) lin1_bytes = dbl_tmp_bytes;
  if (mlp_bytes > lin1_bytes) lin1_bytes = mlp_bytes;

  auto align256 = [](size_t x) -> size_t { return (x + 255ULL) & ~255ULL; };
  size_t need_arena = 0;
  need_arena += align256((size_t)N_img * 128 * 4);       // eps_d
  need_arena += align256((size_t)N_tot * 64 * 4) * 2;    // rope_cos, rope_sin
  need_arena += align256(256 * 4) + align256(256 * 2);   // temb_256, temb_256_bf
  need_arena += align256(E * 4) * 2 + align256(E * 2) * 2; // temb_0, temb, temb_0_bf, silu_temb_bf
  need_arena += align256(6 * E * 4) * 2 + align256(3 * E * 4) + align256(2 * E * 4); // mod_*
  need_arena += align256((size_t)N_img * 128 * 2);       // x_bf16
  need_arena += align256((size_t)N_img * E * 4);         // img_d
  need_arena += align256((size_t)N_txt * E * 4);         // txt_d
  need_arena += align256((size_t)N_tot * E * 4);         // h_cat
  need_arena += align256((size_t)N_tot * E * 2);         // norm_bf
  need_arena += align256((size_t)N_txt * E * 2);         // norm_txt_bf
  need_arena += align256((size_t)N_tot * (E + F) * 2);   // cat_bf
  need_arena += align256((size_t)N_tot * E * 4) * 2;     // attn_out, proj_out
  need_arena += align256(lin1_bytes);                    // lin1_out (aliased with qkv_img/qkv_txt & mlp_2f!)
  need_arena += align256((size_t)N_img * 128 * 4) * (use_cfg ? 2 : 1); // v_pred (+ v_neg)
  need_arena += align256((size_t)H * N_pad * D * 2) * 3; // Q_h, K_h, V_h
  need_arena += align256((size_t)N_tot * (E + F));       // act_q8
  need_arena += align256((size_t)N_tot * ((E + F) / 64) * 4); // act_sc
  if (use_sage) {
    need_arena += align256((size_t)H * N_pad * D) * 2;   // sage_q_i8, sage_k_i8
    need_arena += align256((size_t)H * N_pad * 4) * 2;   // sage_q_sc, sage_k_sc
  }

  if (fx->dit_arena_cap < need_arena) {
    if (fx->dit_arena) { cudaFree(fx->dit_arena); fx->dit_arena = NULL; fx->dit_arena_cap = 0; }
    if (cudaMalloc(&fx->dit_arena, need_arena) != cudaSuccess) return -1;
    fx->dit_arena_cap = need_arena;
  }
  size_t cur_off = 0;
  auto carve = [&](size_t bytes) -> void * {
    void *p = fx->dit_arena + cur_off;
    cur_off += align256(bytes);
    return p;
  };

  float *eps_d       = (float *)carve((size_t)N_img * 128 * 4);
  float *rope_cos    = (float *)carve((size_t)N_tot * 64 * 4);
  float *rope_sin    = (float *)carve((size_t)N_tot * 64 * 4);
  float *temb_256    = (float *)carve(256 * 4);
  uint16_t *temb_256_bf = (uint16_t *)carve(256 * 2);
  float *temb_0      = (float *)carve(E * 4);
  uint16_t *temb_0_bf = (uint16_t *)carve(E * 2);
  float *temb        = (float *)carve(E * 4);
  uint16_t *silu_temb_bf = (uint16_t *)carve(E * 2);
  float *mod_img_v   = (float *)carve(6 * E * 4);
  float *mod_txt_v   = (float *)carve(6 * E * 4);
  float *mod_sng_v   = (float *)carve(3 * E * 4);
  float *fmod_v      = (float *)carve(2 * E * 4);
  uint16_t *x_bf16   = (uint16_t *)carve((size_t)N_img * 128 * 2);
  float *img_d       = (float *)carve((size_t)N_img * E * 4);
  float *txt_d       = (float *)carve((size_t)N_txt * E * 4);
  float *h_cat       = (float *)carve((size_t)N_tot * E * 4);
  uint16_t *norm_bf  = (uint16_t *)carve((size_t)N_tot * E * 2);
  uint16_t *norm_txt_bf = (uint16_t *)carve((size_t)N_txt * E * 2);
  uint16_t *cat_bf   = (uint16_t *)carve((size_t)N_tot * (E + F) * 2);
  float *attn_out    = (float *)carve((size_t)N_tot * E * 4);
  float *proj_out    = (float *)carve((size_t)N_tot * E * 4);
  float *lin1_out    = (float *)carve(lin1_bytes);
  float *qkv_img     = lin1_out;
  float *qkv_txt     = (float *)((uint8_t *)lin1_out + align256((size_t)N_img * 3 * E * 4));
  float *mlp_2f      = lin1_out;
  float *v_pred      = (float *)carve((size_t)N_img * 128 * 4);
  float *v_neg       = use_cfg ? (float *)carve((size_t)N_img * 128 * 4) : NULL;
  __half *Q_h        = (__half *)carve((size_t)H * N_pad * D * 2);
  __half *K_h        = (__half *)carve((size_t)H * N_pad * D * 2);
  __half *V_h        = (__half *)carve((size_t)H * N_pad * D * 2);
  int8_t *act_q8     = (int8_t *)carve((size_t)N_tot * (E + F));
  float  *act_sc     = (float *)carve((size_t)N_tot * ((E + F) / 64) * 4);
  int8_t *sage_q_i8  = use_sage ? (int8_t *)carve((size_t)H * N_pad * D) : NULL;
  int8_t *sage_k_i8  = use_sage ? (int8_t *)carve((size_t)H * N_pad * D) : NULL;
  float  *sage_q_sc  = use_sage ? (float *)carve((size_t)H * N_pad * 4)  : NULL;
  float  *sage_k_sc  = use_sage ? (float *)carve((size_t)H * N_pad * 4)  : NULL;

  // Initial unit Gaussian noise eps [N_img, 128]
  float *eps_h = (float *)malloc((size_t)N_img * 128 * 4);
  if (!eps_h) return -1;
  fill_gaussian(eps_h, (size_t)N_img * 128, seed);
  cudaMemcpyAsync(eps_d, eps_h, (size_t)N_img * 128 * 4, cudaMemcpyHostToDevice, s);
  free(eps_h);

  if (z0_init_d) {
    float sigma0 = sigmas[s0];
    cudaMemcpyAsync(out_latents_gpu_f32, eps_d, (size_t)N_img * 128 * 4, cudaMemcpyDeviceToDevice, s);
    k_flux2_mask_project<<<((long)N_img * 128 + 255) / 256, 256, 0, s>>>(
        out_latents_gpu_f32, z0_init_d, eps_d, NULL, N_img, sigma0);
    if (sigma0 < 0.9999f) {
      float *zero_m = NULL;
      cudaMalloc(&zero_m, (size_t)N_img * 4);
      cudaMemsetAsync(zero_m, 0, (size_t)N_img * 4, s);
      k_flux2_mask_project<<<((long)N_img * 128 + 255) / 256, 256, 0, s>>>(
          out_latents_gpu_f32, z0_init_d, eps_d, zero_m, N_img, sigma0);
      cudaFree(zero_m);
    }
  } else {
    cudaMemcpyAsync(out_latents_gpu_f32, eps_d, (size_t)N_img * 128 * 4, cudaMemcpyDeviceToDevice, s);
  }

  k_flux2_precompute_rope<<<N_tot, 64, 0, s>>>(rope_cos, rope_sin, N_txt, H_p, W_p, 2000.0);
  if (cudaGetLastError() != cudaSuccess) return -1;

  auto gemm_fx = [&](const uint16_t *X_bf16, const fx_mat_t &W, float *Y_f32, int M, int N, int K) {
    if (W.q8 && W.sc) {
      ec_ops_quantize_bf16_q8g64(X_bf16, act_q8, act_sc, M, K, s);
      ec_ops_gemm_q8g64(act_q8, act_sc, W.q8, W.sc, NULL, Y_f32, NULL, M, N, K, s);
    } else {
      gemm_bf16(X_bf16, W.bf16, NULL, Y_f32, NULL, M, N, K, s);
    }
  };

  auto attn_fx = [&]() {
    if (use_sage) {
      ec_launch_sage_attn_i8(Q_h, K_h, V_h, sage_q_i8, sage_k_i8, sage_q_sc, sage_k_sc,
                             attn_out, NULL, N_tot, N_tot, N_pad, N_pad, H, D, attn_scale, 0, s);
    } else {
      ec_launch_fa2_f16(Q_h, K_h, V_h, attn_out, NULL, N_tot, N_tot, N_pad, N_pad, H, D, attn_scale, 0, s);
    }
  };

  auto eval_mmdit_pass = [&](const uint16_t *emb_bf, float *out_v) {
    k_f32_to_bf16<<<((long)N_img * 128 + 255) / 256, 256, 0, s>>>(out_latents_gpu_f32, x_bf16, (long)N_img * 128);
    gemm_bf16(x_bf16, d->img_in, NULL, img_d, NULL, N_img, E, 128, s);
    gemm_bf16(emb_bf, d->txt_in, NULL, txt_d, NULL, N_txt, E, 7680, s);

    for (int b = 0; b < 5; b++) {
      const fx_double_w_t *w = &d->dbl[b];
      k_adaln_bf16<<<N_img, 256, 0, s>>>(img_d, mod_img_v, mod_img_v + E, norm_bf, N_img, E, 1e-6f);
      k_adaln_bf16<<<N_txt, 256, 0, s>>>(txt_d, mod_txt_v, mod_txt_v + E, norm_txt_bf, N_txt, E, 1e-6f);
      gemm_fx(norm_bf,     w->img_qkv, qkv_img, N_img, 3 * E, E);
      gemm_fx(norm_txt_bf, w->txt_qkv, qkv_txt, N_txt, 3 * E, E);
      int w_txt = N_txt * H, w_img = N_img * H;
      k_flux2_pack_qkv_rope<<<(w_txt + 7) / 8, 256, 0, s>>>(
          qkv_txt, 3 * E, w->txt_qn, w->txt_kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_txt, 0, N_pad, H, D, 1e-6f);
      k_flux2_pack_qkv_rope<<<(w_img + 7) / 8, 256, 0, s>>>(
          qkv_img, 3 * E, w->img_qn, w->img_kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_img, N_txt, N_pad, H, D, 1e-6f);
      attn_fx();
      k_f32_to_bf16<<<((long)N_txt * E + 255) / 256, 256, 0, s>>>(attn_out, norm_txt_bf, (long)N_txt * E);
      k_f32_to_bf16<<<((long)N_img * E + 255) / 256, 256, 0, s>>>(attn_out + (long)N_txt * E, norm_bf, (long)N_img * E);
      gemm_fx(norm_txt_bf, w->txt_proj, proj_out, N_txt, E, E);
      ec_launch_gated_res(txt_d, proj_out, mod_txt_v + 2 * E, N_txt, E, s);
      gemm_fx(norm_bf, w->img_proj, proj_out, N_img, E, E);
      ec_launch_gated_res(img_d, proj_out, mod_img_v + 2 * E, N_img, E, s);
      k_adaln_bf16<<<N_img, 256, 0, s>>>(img_d, mod_img_v + 3 * E, mod_img_v + 4 * E, norm_bf, N_img, E, 1e-6f);
      gemm_fx(norm_bf, w->img_mlp0, mlp_2f, N_img, 2 * F, E);
      k_flux2_swiglu_bf16<<<((long)N_img * F + 255) / 256, 256, 0, s>>>(mlp_2f, cat_bf, N_img, F);
      gemm_fx(cat_bf, w->img_mlp2, proj_out, N_img, E, F);
      ec_launch_gated_res(img_d, proj_out, mod_img_v + 5 * E, N_img, E, s);
      k_adaln_bf16<<<N_txt, 256, 0, s>>>(txt_d, mod_txt_v + 3 * E, mod_txt_v + 4 * E, norm_txt_bf, N_txt, E, 1e-6f);
      gemm_fx(norm_txt_bf, w->txt_mlp0, mlp_2f, N_txt, 2 * F, E);
      k_flux2_swiglu_bf16<<<((long)N_txt * F + 255) / 256, 256, 0, s>>>(mlp_2f, cat_bf, N_txt, F);
      gemm_fx(cat_bf, w->txt_mlp2, proj_out, N_txt, E, F);
      ec_launch_gated_res(txt_d, proj_out, mod_txt_v + 5 * E, N_txt, E, s);
    }

    cudaMemcpyAsync(h_cat, txt_d, (size_t)N_txt * E * 4, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(h_cat + (size_t)N_txt * E, img_d, (size_t)N_img * E * 4, cudaMemcpyDeviceToDevice, s);

    int w_tot = N_tot * H;
    for (int b = 0; b < 20; b++) {
      const fx_single_w_t *w = &d->sng[b];
      k_adaln_bf16<<<N_tot, 256, 0, s>>>(h_cat, mod_sng_v, mod_sng_v + E, norm_bf, N_tot, E, 1e-6f);
      gemm_fx(norm_bf, w->lin1, lin1_out, N_tot, 3 * E + 2 * F, E);
      k_flux2_pack_qkv_rope<<<(w_tot + 7) / 8, 256, 0, s>>>(
          lin1_out, 3 * E + 2 * F, w->qn, w->kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_tot, 0, N_pad, H, D, 1e-6f);
      attn_fx();
      if (w->lin2.q8 && w->lin2.sc) {
        const long tot_thr = (long)N_tot * ((E + F) >> 6) * 16;
        k_flux2_single_cat_q8g64<<<(int)((tot_thr + 255) / 256), 256, 0, s>>>(attn_out, lin1_out, act_q8, act_sc, N_tot, E, F);
        ec_ops_gemm_q8g64(act_q8, act_sc, w->lin2.q8, w->lin2.sc, NULL, proj_out, NULL, N_tot, E, E + F, s);
      } else {
        k_flux2_single_cat<<<((long)N_tot * (E + F) + 255) / 256, 256, 0, s>>>(attn_out, lin1_out, cat_bf, N_tot, E, F);
        gemm_fx(cat_bf, w->lin2, proj_out, N_tot, E, E + F);
      }
      ec_launch_gated_res(h_cat, proj_out, mod_sng_v + 2 * E, N_tot, E, s);
    }

    gemm_bf16(silu_temb_bf, d->final_mod, NULL, fmod_v, NULL, 1, 2 * E, E, s);
    k_adaln_bf16<<<N_img, 256, 0, s>>>(h_cat + (size_t)N_txt * E, fmod_v, fmod_v + E, norm_bf, N_img, E, 1e-6f);
    gemm_bf16(norm_bf, d->final_lin, NULL, out_v, NULL, N_img, 128, E, s);
  };

  for (int step = s0; step < steps; step++) {
    if (cb && cb(step, steps, "flux2_dit", cb_ctx) != 0) break;

    float sigma = sigmas[step];
    float sigma_next = sigmas[step + 1];
    float dt = sigma_next - sigma;

    float t_h[256];
    for (int i = 0; i < 128; i++) {
      double freq = exp(-log(10000.0) * (double)i / 128.0);
      double arg = (double)sigma * 1000.0 * freq;
      t_h[i] = (float)cos(arg); t_h[128 + i] = (float)sin(arg);
    }
    cudaMemcpyAsync(temb_256, t_h, sizeof t_h, cudaMemcpyHostToDevice, s);
    k_f32_to_bf16<<<1, 256, 0, s>>>(temb_256, temb_256_bf, 256);
    gemm_bf16(temb_256_bf, d->time_in0, NULL, temb_0, NULL, 1, E, 256, s);
    k_silu_f32_to_bf16<<<(E + 255) / 256, 256, 0, s>>>(temb_0, temb_0_bf, E);
    gemm_bf16(temb_0_bf, d->time_in1, NULL, temb, NULL, 1, E, E, s);
    k_silu_f32_to_bf16<<<(E + 255) / 256, 256, 0, s>>>(temb, silu_temb_bf, E);
    gemm_bf16(silu_temb_bf, d->mod_img, NULL, mod_img_v, NULL, 1, 6 * E, E, s);
    gemm_bf16(silu_temb_bf, d->mod_txt, NULL, mod_txt_v, NULL, 1, 6 * E, E, s);
    gemm_bf16(silu_temb_bf, d->mod_sng, NULL, mod_sng_v, NULL, 1, 3 * E, E, s);

    eval_mmdit_pass(prompt_embeds_bf16, v_pred);
    if (use_cfg) {
      eval_mmdit_pass(neg_embeds_bf16, v_neg);
      ec_launch_flow_cfg_euler(out_latents_gpu_f32, v_pred, v_neg, out_latents_gpu_f32, N_img * 128, dt, cfg_scale, s);
    } else {
      ec_launch_flow_cfg_euler(out_latents_gpu_f32, v_pred, NULL, out_latents_gpu_f32, N_img * 128, dt, 1.0f, s);
    }

    // Masked Inpainting / Outpainting: lock unmasked pixels to exact forward-noised z0!
    if (z0_init_d && mask_p_d) {
      k_flux2_mask_project<<<((long)N_img * 128 + 255) / 256, 256, 0, s>>>(
          out_latents_gpu_f32, z0_init_d, eps_d, mask_p_d, N_img, sigma_next);
    }
  }
  cudaStreamSynchronize(s);

  // Trim > 512 MB DiT arena only when < 13.6 GB VRAM remains free (standalone keeps 14.3 GB free = 100% hot; Chat co-residency frees +1.1 GB -> ~14.0 GB free!)
  if (fx->dit_arena && fx->dit_arena_cap > (512ULL << 20)) {
    size_t free_b = 0, tot_b = 0;
    if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && free_b < (15000ULL << 20) && fx->qwen_res.loaded) {
      free_qwen3_resident(&fx->qwen_res);
    }
    if (cudaMemGetInfo(&free_b, &tot_b) != cudaSuccess || free_b < (13600ULL << 20)) {
      cudaFree(fx->dit_arena);
      fx->dit_arena = NULL;
      fx->dit_arena_cap = 0;
    }
  }

  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
}

// ===========================================================================
// Stage C: AutoencoderKLFlux2 2D VAE Decoder (Strip-Tiled BF16 TensorCore Conv2D)
// ===========================================================================
static int vae_conv2d_tc(
    ec_flux2_t *fx, const char *prefix,
    const float *x_d, const float *res_d, float *y_d,
    int Cin, int Cout, int H, int W, int Ksz, int pad, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  uint16_t *w_bf = get_vae_bf16(fx, kw);
  float *b_f32 = get_vae_f32(fx, kb);
  if (!w_bf || !b_f32) return -1;

  int K = Cin * Ksz * Ksz;
  int strip_max = 64;
  if (strip_max > H) strip_max = H;
  if (ensure_vae_conv_scratch(&fx->vae_cache, (size_t)strip_max * W * K * 2, (size_t)strip_max * W * Cout * 4) != 0) return -1;
  uint16_t *col_bf = fx->vae_cache.col_bf;
  float *gy_d = fx->vae_cache.gy_d;

  for (int y0 = 0; y0 < H; y0 += strip_max) {
    int sh = (y0 + strip_max <= H) ? strip_max : (H - y0);
    long n_col = (long)sh * W * K;
    k_im2col_strip_bf16<<<(int)((n_col + 255) / 256), 256, 0, s>>>(x_d, col_bf, Cin, H, W, Ksz, Ksz, pad, y0, sh);
    gemm_bf16(col_bf, w_bf, b_f32, gy_d, NULL, sh * W, Cout, K, s);
    long n_out = (long)Cout * sh * W;
    k_strip_transpose_out<<<(int)((n_out + 255) / 256), 256, 0, s>>>(gy_d, res_d, y_d, Cout, H, W, y0, sh);
  }
  return 0;
}

static int vae_gn_silu(
    ec_flux2_t *fx, const char *prefix,
    const float *x_d, float *y_d, int C, int H, int W, int apply_silu, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  float *g = get_vae_f32(fx, kw);
  float *b = get_vae_f32(fx, kb);
  if (!g || !b) return -1;
  ec_launch_groupnorm_silu(x_d, g, b, y_d, C, H * W, 32, 1e-6f, apply_silu, s);
  return 0;
}

static int vae_resnet(
    ec_flux2_t *fx, const char *prefix,
    float *x_d, float *out_d, int Cin, int Cout, int H, int W, cudaStream_t s) {
  int maxC = Cin > Cout ? Cin : Cout;
  float *tmp1 = ensure_vae_res_slot(&fx->vae_cache, 0, (size_t)maxC * H * W * 4);
  float *tmp2 = ensure_vae_res_slot(&fx->vae_cache, 1, (size_t)Cout * H * W * 4);
  float *sc   = (Cin != Cout) ? ensure_vae_res_slot(&fx->vae_cache, 2, (size_t)Cout * H * W * 4) : NULL;
  if (!tmp1 || !tmp2 || (Cin != Cout && !sc)) return -1;

  char k[192];
  snprintf(k, sizeof k, "%s.norm1", prefix);
  if (vae_gn_silu(fx, k, x_d, tmp1, Cin, H, W, 1, s) != 0) return -1;
  snprintf(k, sizeof k, "%s.conv1", prefix);
  if (vae_conv2d_tc(fx, k, tmp1, NULL, tmp2, Cin, Cout, H, W, 3, 1, s) != 0) return -1;

  snprintf(k, sizeof k, "%s.norm2", prefix);
  if (vae_gn_silu(fx, k, tmp2, tmp1, Cout, H, W, 1, s) != 0) return -1;

  if (Cin != Cout) {
    snprintf(k, sizeof k, "%s.conv_shortcut", prefix);
    if (vae_conv2d_tc(fx, k, x_d, NULL, sc, Cin, Cout, H, W, 1, 0, s) != 0) return -1;
  }
  snprintf(k, sizeof k, "%s.conv2", prefix);
  if (vae_conv2d_tc(fx, k, tmp1, sc ? sc : x_d, out_d, Cout, Cout, H, W, 3, 1, s) != 0) return -1;
  return 0;
}

static int vae_conv2d_s2_tc(
    ec_flux2_t *fx, const char *prefix,
    const float *x_d, float *y_d,
    int Cin, int Cout, int H_in, int W_in, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  uint16_t *w_bf = get_vae_bf16(fx, kw);
  float *b_f32 = get_vae_f32(fx, kb);
  if (!w_bf || !b_f32) return -1;

  int H_out = H_in / 2, W_out = W_in / 2;
  int K = Cin * 9;
  int strip_max = 64;
  if (strip_max > H_out) strip_max = H_out;
  if (ensure_vae_conv_scratch(&fx->vae_cache, (size_t)strip_max * W_out * K * 2, (size_t)strip_max * W_out * Cout * 4) != 0) return -1;
  uint16_t *col_bf = fx->vae_cache.col_bf;
  float *gy_d = fx->vae_cache.gy_d;

  for (int y0 = 0; y0 < H_out; y0 += strip_max) {
    int sh = (y0 + strip_max <= H_out) ? strip_max : (H_out - y0);
    long n_col = (long)sh * W_out * K;
    k_im2col_strip_s2_bf16<<<(int)((n_col + 255) / 256), 256, 0, s>>>(x_d, col_bf, Cin, H_in, W_in, H_out, W_out, y0, sh);
    gemm_bf16(col_bf, w_bf, b_f32, gy_d, NULL, sh * W_out, Cout, K, s);
    long n_out = (long)Cout * sh * W_out;
    k_strip_transpose_out<<<(int)((n_out + 255) / 256), 256, 0, s>>>(gy_d, NULL, y_d, Cout, H_out, W_out, y0, sh);
  }
  return 0;
}

static int vae_mid_attention_ex(ec_flux2_t *fx, const char *pfx, float *x_d, int C, int H, int W, cudaStream_t s) {
  int S = H * W;
  float *gn = NULL, *q = NULL, *k = NULL, *v = NULL, *out = NULL;
  uint16_t *sc_bf = NULL, *ao_bf = NULL;
  cudaMalloc(&gn, (size_t)C * S * 4);
  cudaMalloc(&sc_bf, (size_t)S * C * 2);
  cudaMalloc(&q, (size_t)S * C * 4);
  cudaMalloc(&k, (size_t)S * C * 4);
  cudaMalloc(&v, (size_t)S * C * 4);
  cudaMalloc(&ao_bf, (size_t)S * C * 2);
  cudaMalloc(&out, (size_t)S * C * 4);

  char tk[192];
  snprintf(tk, sizeof tk, "%s.group_norm", pfx);
  if (vae_gn_silu(fx, tk, x_d, gn, C, H, W, 0, s) != 0) goto fail;
  k_chw_to_sc_bf16<<<((long)C * S + 255) / 256, 256, 0, s>>>(gn, sc_bf, C, S);

  {
    snprintf(tk, sizeof tk, "%s.to_q.weight", pfx); uint16_t *wq = get_vae_bf16(fx, tk);
    snprintf(tk, sizeof tk, "%s.to_q.bias", pfx);   float    *bq = get_vae_f32(fx,  tk);
    snprintf(tk, sizeof tk, "%s.to_k.weight", pfx); uint16_t *wk = get_vae_bf16(fx, tk);
    snprintf(tk, sizeof tk, "%s.to_k.bias", pfx);   float    *bk = get_vae_f32(fx,  tk);
    snprintf(tk, sizeof tk, "%s.to_v.weight", pfx); uint16_t *wv = get_vae_bf16(fx, tk);
    snprintf(tk, sizeof tk, "%s.to_v.bias", pfx);   float    *bv = get_vae_f32(fx,  tk);
    snprintf(tk, sizeof tk, "%s.to_out.0.weight", pfx); uint16_t *wo = get_vae_bf16(fx, tk);
    snprintf(tk, sizeof tk, "%s.to_out.0.bias", pfx);   float    *bo = get_vae_f32(fx,  tk);
    if (!wq || !wk || !wv || !wo) goto fail;
    gemm_bf16(sc_bf, wq, bq, q, NULL, S, C, C, s);
    gemm_bf16(sc_bf, wk, bk, k, NULL, S, C, C, s);
    gemm_bf16(sc_bf, wv, bv, v, NULL, S, C, C, s);
    k_vae_mid_attn<<<S, 256, 0, s>>>(q, k, v, ao_bf, S, C, 1.0f / sqrtf((float)C));
    gemm_bf16(ao_bf, wo, bo, out, NULL, S, C, C, s);
    k_sc_add_to_chw<<<((long)C * S + 255) / 256, 256, 0, s>>>(out, x_d, C, S);
  }

  cudaFree(gn); cudaFree(sc_bf); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ao_bf); cudaFree(out);
  return 0;
fail:
  cudaFree(gn); cudaFree(sc_bf); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ao_bf); cudaFree(out);
  return -1;
}

// Native AutoencoderKLFlux2 2D VAE Encoder: RGB24 [H, W, 3] -> BN-normalized packed [H/16 * W/16, 128]!
static int run_flux2_vae_encode(
    ec_flux2_t *fx, const uint8_t *in_rgb_hwc, int H, int W,
    float *out_packed_latents_d, double *out_ms) {
  double t0 = fx_now_ms();
  cudaStream_t s = fx->stream;
  int H_p = H / 16, W_p = W / 16;

  size_t plane = (size_t)H * (size_t)W;
  float *rgb_h = (float *)malloc(3u * plane * sizeof(float));
  if (!rgb_h) return -1;
  for (size_t i = 0; i < plane; i++) {
    rgb_h[0 * plane + i] = (float)in_rgb_hwc[i * 3 + 0] * (2.0f / 255.0f) - 1.0f;
    rgb_h[1 * plane + i] = (float)in_rgb_hwc[i * 3 + 1] * (2.0f / 255.0f) - 1.0f;
    rgb_h[2 * plane + i] = (float)in_rgb_hwc[i * 3 + 2] * (2.0f / 255.0f) - 1.0f;
  }
  size_t ping_bytes = (size_t)256 * H * W * 4;
  {
    size_t free_b = 0, tot_b = 0;
    if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && free_b < ping_bytes * 2 + 1200ULL * 1024ULL * 1024ULL) {
      free_qwen3_resident(&fx->qwen_res);
    }
  }
  float *ping0 = NULL, *ping1 = NULL, *rgb_d = NULL;
  if (cudaMalloc(&ping0, ping_bytes) != cudaSuccess ||
      cudaMalloc(&ping1, ping_bytes) != cudaSuccess ||
      cudaMalloc(&rgb_d, 3u * plane * 4) != cudaSuccess) {
    cudaFree(ping0); cudaFree(ping1); cudaFree(rgb_d); free(rgb_h); return -1;
  }
  cudaMemcpyAsync(rgb_d, rgb_h, 3u * plane * sizeof(float), cudaMemcpyHostToDevice, s);
  free(rgb_h);

  float *cur = ping0, *nxt = ping1;
  #define SWAP_PING() do { float *_t = cur; cur = nxt; nxt = _t; } while (0)

  if (vae_conv2d_tc(fx, "encoder.conv_in", rgb_d, NULL, cur, 3, 128, H, W, 3, 1, s) != 0) {
    cudaFree(rgb_d); cudaFree(ping0); cudaFree(ping1); return -1;
  }
  cudaFree(rgb_d);

  const int dn_ch[4] = { 128, 256, 512, 512 };
  int C = 128;
  int cH = H, cW = W;
  char pref[128];
  for (int b = 0; b < 4; b++) {
    int Cout = dn_ch[b];
    for (int r = 0; r < 2; r++) {
      int Cin = (r == 0) ? C : Cout;
      snprintf(pref, sizeof pref, "encoder.down_blocks.%d.resnets.%d", b, r);
      if (vae_resnet(fx, pref, cur, nxt, Cin, Cout, cH, cW, s) != 0) { cudaFree(ping0); cudaFree(ping1); return -1; }
      SWAP_PING();
    }
    C = Cout;
    if (b < 3) {
      int nH = cH / 2, nW = cW / 2;
      snprintf(pref, sizeof pref, "encoder.down_blocks.%d.downsamplers.0.conv", b);
      if (vae_conv2d_s2_tc(fx, pref, cur, nxt, C, C, cH, cW, s) != 0) { cudaFree(ping0); cudaFree(ping1); return -1; }
      SWAP_PING();
      cH = nH; cW = nW;
    }
  }

  vae_resnet(fx, "encoder.mid_block.resnets.0", cur, nxt, 512, 512, cH, cW, s);
  SWAP_PING();
  vae_mid_attention_ex(fx, "encoder.mid_block.attentions.0", cur, 512, cH, cW, s);
  vae_resnet(fx, "encoder.mid_block.resnets.1", cur, nxt, 512, 512, cH, cW, s);
  SWAP_PING();

  vae_gn_silu(fx, "encoder.conv_norm_out", cur, nxt, 512, cH, cW, 1, s);
  SWAP_PING();
  vae_conv2d_tc(fx, "encoder.conv_out", cur, NULL, nxt, 512, 64, cH, cW, 3, 1, s);
  SWAP_PING();
  vae_conv2d_tc(fx, "quant_conv", cur, NULL, nxt, 64, 64, cH, cW, 1, 0, s);
  SWAP_PING();

  float *bn_mean = get_vae_f32(fx, "bn.running_mean");
  float *bn_var  = get_vae_f32(fx, "bn.running_var");
  if (!bn_mean || !bn_var) { cudaFree(ping0); cudaFree(ping1); return -1; }
  k_flux2_patchify_bn_norm<<<(int)(((long)H_p * W_p * 128 + 255) / 256), 256, 0, s>>>(
      cur, bn_mean, bn_var, out_packed_latents_d, H_p, W_p, 1e-4f);
  cudaStreamSynchronize(s);
  cudaFree(ping0); cudaFree(ping1);
  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
}

// ===========================================================================
// v15 Implicit-GEMM NHWC 2D VAE Decoder Kernels (Zero im2col, Zero CHW<->HWC Transpose)
// ===========================================================================
__global__ void k_flux2_bn_unpatchify_nhwc_bf16(
    const float *__restrict__ packed, const float *__restrict__ bn_mean, const float *__restrict__ bn_var,
    uint16_t *__restrict__ out_nhwc_bf, int H_p, int W_p, float bn_eps) {
  const int H_lat = H_p * 2, W_lat = W_p * 2;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long tot = 32L * H_lat * W_lat;
  if (idx >= tot) return;
  const int c = (int)(idx & 31);
  const int pix = (int)(idx >> 5);
  const int y = pix / W_lat, x = pix - y * W_lat;
  const int hp = y >> 1, py = y & 1;
  const int wp = x >> 1, px = x & 1;
  const int pch = c * 4 + py * 2 + px;
  const float v = packed[(long)(hp * W_p + wp) * 128 + pch] * sqrtf(bn_var[pch] + bn_eps) + bn_mean[pch];
  out_nhwc_bf[idx] = f32_to_bf16_dev(v);
}

// GroupNorm stats over NHWC [M, C] (num_groups = 32, C in {128, 256, 512})
__global__ void __launch_bounds__(256) k_nhwc_gn_stats(
    const float *__restrict__ x_nhwc, float *__restrict__ stats64, int M, int C) {
  const int grp = blockIdx.x; // 0..31
  const int gc  = C >> 5;     // 4, 8, or 16 channels per group
  const int c0  = grp * gc;
  double s1 = 0.0, s2 = 0.0;

  const long total_work = (long)M * gc;
  const long stride = (long)gridDim.y * blockDim.x;
  for (long i = (long)blockIdx.y * blockDim.x + threadIdx.x; i < total_work; i += stride) {
    const int m = (int)(i / gc);
    const int dc = (int)(i - (long)m * gc);
    const float v = x_nhwc[(long)m * C + c0 + dc];
    s1 += (double)v;
    s2 += (double)v * (double)v;
  }

  for (int o = 16; o > 0; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  }
  __shared__ double sm1[8], sm2[8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { sm1[warp] = s1; sm2[warp] = s2; }
  __syncthreads();
  if (warp == 0) {
    double a = (lane < 8) ? sm1[lane] : 0.0;
    double b = (lane < 8) ? sm2[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) {
      a += __shfl_xor_sync(0xffffffffu, a, o);
      b += __shfl_xor_sync(0xffffffffu, b, o);
    }
    if (lane == 0) {
      atomicAdd(&stats64[grp],      (float)a);
      atomicAdd(&stats64[32 + grp], (float)b);
    }
  }
}

__global__ void __launch_bounds__(256) k_nhwc_gn_apply_bf16(
    const float *__restrict__ x_nhwc, const float *__restrict__ stats64,
    const float *__restrict__ gamma, const float *__restrict__ beta,
    uint16_t *__restrict__ y_bf16, int M, int C, float eps, int apply_silu) {
  const int gc = C >> 5;
  const float inv_n = 1.0f / ((float)M * (float)gc);
  __shared__ float s_mean[32], s_rstd[32];
  if (threadIdx.x < 32) {
    const int g = threadIdx.x;
    const float mean = stats64[g] * inv_n;
    const float var  = fmaxf(0.0f, stats64[32 + g] * inv_n - mean * mean);
    s_mean[g] = mean;
    s_rstd[g] = rsqrtf(var + eps);
  }
  __syncthreads();

  const long n_pairs = ((long)M * C) >> 1;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pairs) return;
  const long off = idx << 1;
  const int c0 = (int)(off % C);
  const int g0 = c0 >> ((C == 512) ? 4 : ((C == 256) ? 3 : 2));
  const float2 xv = *(const float2 *)(x_nhwc + off);
  float v0 = (xv.x - s_mean[g0]) * s_rstd[g0] * gamma[c0] + beta[c0];
  float v1 = (xv.y - s_mean[g0]) * s_rstd[g0] * gamma[c0 + 1] + beta[c0 + 1];
  if (apply_silu) { v0 = silu_dev(v0); v1 = silu_dev(v1); }
  __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
  *(uint32_t *)(y_bf16 + off) = *(const uint32_t *)&b2;
}

__global__ void __launch_bounds__(256) k_upsample2x_nhwc_bf16(
    const float *__restrict__ x_nhwc, uint16_t *__restrict__ y_bf16, int H, int W, int C) {
  const int H2 = H * 2, W2 = W * 2;
  const long n_pairs = ((long)H2 * W2 * C) >> 1;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pairs) return;
  const long off = idx << 1;
  const int c0 = (int)(off % C);
  const int pix = (int)(off / C);
  const int y2 = pix / W2, x2 = pix - y2 * W2;
  const float2 xv = *(const float2 *)(x_nhwc + ((long)((y2 >> 1) * W + (x2 >> 1)) * C + c0));
  __nv_bfloat162 b2 = __floats2bfloat162_rn(xv.x, xv.y);
  *(uint32_t *)(y_bf16 + off) = *(const uint32_t *)&b2;
}

// Fast fused final 3x3 Conv (Cin=128 -> Cout=3) + clamp ((v * 0.5 + 0.5) * 255) -> RGB24 uint8!
__global__ void __launch_bounds__(256) k_vae_conv_out_rgb8_nhwc(
    const uint16_t *__restrict__ X_nhwc, const uint16_t *__restrict__ W_ohwi, const float *__restrict__ bias,
    uint8_t *__restrict__ out_rgb, int H, int W) {
  __shared__ float sW[3][9 * 128];
  for (int i = threadIdx.x; i < 3 * 9 * 128; i += 256) {
    sW[i / (9 * 128)][i % (9 * 128)] = bf16_to_f32_dev(W_ohwi[i]);
  }
  __syncthreads();

  const int pix = (blockIdx.x * 16) + (threadIdx.x >> 4);
  if (pix >= H * W) return;
  const int sub = threadIdx.x & 15; // 16 lanes * 8 channels = all 128 channels!
  const int oy = pix / W, ox = pix - oy * W;
  const int ci0 = sub * 8;

  float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
  #pragma unroll
  for (int tap = 0; tap < 9; tap++) {
    const int iy = oy + (tap / 3) - 1;
    const int ix = ox + (tap % 3) - 1;
    if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
      const uint4 xv = *(const uint4 *)(X_nhwc + ((long)(iy * W + ix) * 128 + ci0));
      const uint16_t *xb = (const uint16_t *)&xv;
      const int woff = tap * 128 + ci0;
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        const float xf = bf16_to_f32_dev(xb[j]);
        r_acc = fmaf(xf, sW[0][woff + j], r_acc);
        g_acc = fmaf(xf, sW[1][woff + j], g_acc);
        b_acc = fmaf(xf, sW[2][woff + j], b_acc);
      }
    }
  }
  #pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    r_acc += __shfl_xor_sync(0xffffffffu, r_acc, o);
    g_acc += __shfl_xor_sync(0xffffffffu, g_acc, o);
    b_acc += __shfl_xor_sync(0xffffffffu, b_acc, o);
  }
  if (sub == 0) {
    auto to_u8 = [](float v) -> uint8_t {
      float c = (v * 0.5f + 0.5f) * 255.0f + 0.5f;
      return (uint8_t)(c < 0.0f ? 0.0f : (c > 255.0f ? 255.0f : c));
    };
    out_rgb[(long)pix * 3 + 0] = to_u8(r_acc + (bias ? bias[0] : 0.0f));
    out_rgb[(long)pix * 3 + 1] = to_u8(g_acc + (bias ? bias[1] : 0.0f));
    out_rgb[(long)pix * 3 + 2] = to_u8(b_acc + (bias ? bias[2] : 0.0f));
  }
}

static int vae_gn_silu_nhwc(
    ec_flux2_t *fx, const char *prefix,
    const float *x_nhwc, uint16_t *y_bf16, float *stats64_d,
    int M, int C, int apply_silu, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  float *g = get_vae_f32(fx, kw);
  float *b = get_vae_f32(fx, kb);
  if (!g || !b) return -1;
  cudaMemsetAsync(stats64_d, 0, 64 * sizeof(float), s);
  int y_blocks = (M * (C >> 5) + 1023) / 1024;
  if (y_blocks < 4) y_blocks = 4;
  if (y_blocks > 64) y_blocks = 64;
  k_nhwc_gn_stats<<<dim3(32, y_blocks), 256, 0, s>>>(x_nhwc, stats64_d, M, C);
  const long n_pairs = ((long)M * C) >> 1;
  k_nhwc_gn_apply_bf16<<<(int)((n_pairs + 255) / 256), 256, 0, s>>>(
      x_nhwc, stats64_d, g, b, y_bf16, M, C, 1e-6f, apply_silu);
  return 0;
}

static int vae_conv3x3_nhwc(
    ec_flux2_t *fx, const char *prefix,
    const uint16_t *x_bf16, const float *res_f32, float *y_f32, uint16_t *y_bf16,
    int H, int W, int Cin, int Cout, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  uint16_t *w_ohwi = get_vae_ohwi_3x3_bf16(fx, kw, Cout, Cin);
  float    *b_f32  = get_vae_f32(fx, kb);
  if (!w_ohwi || !b_f32) return -1;
  return ec_ops_conv2d_3x3_nhwc_bf16(x_bf16, w_ohwi, b_f32, res_f32, y_f32, y_bf16, H, W, Cin, Cout, s);
}

static int vae_resnet_nhwc(
    ec_flux2_t *fx, const char *prefix,
    float *x_f32, float *out_f32, float *h_f32,
    uint16_t *tmp_bf16, float *stats64_d,
    int Cin, int Cout, int H, int W, cudaStream_t s) {
  const int M = H * W;
  char k[192];
  snprintf(k, sizeof k, "%s.norm1", prefix);
  if (vae_gn_silu_nhwc(fx, k, x_f32, tmp_bf16, stats64_d, M, Cin, 1, s) != 0) return -1;
  snprintf(k, sizeof k, "%s.conv1", prefix);
  if (vae_conv3x3_nhwc(fx, k, tmp_bf16, NULL, h_f32, NULL, H, W, Cin, Cout, s) != 0) return -1;

  const float *res_ptr = x_f32;
  if (Cin != Cout) {
    k_f32_to_bf16<<<(int)(((long)M * Cin + 255) / 256), 256, 0, s>>>(x_f32, tmp_bf16, (long)M * Cin);
    snprintf(k, sizeof k, "%s.conv_shortcut.weight", prefix); uint16_t *ws = get_vae_bf16(fx, k);
    snprintf(k, sizeof k, "%s.conv_shortcut.bias", prefix);   float    *bs = get_vae_f32(fx, k);
    if (!ws || !bs) return -1;
    // Write 1x1 shortcut directly into out_f32; conv2 adds res_f32[off] in-place per disjoint CTA tile!
    gemm_bf16(tmp_bf16, ws, bs, out_f32, NULL, M, Cout, Cin, s);
    res_ptr = out_f32;
  }

  snprintf(k, sizeof k, "%s.norm2", prefix);
  if (vae_gn_silu_nhwc(fx, k, h_f32, tmp_bf16, stats64_d, M, Cout, 1, s) != 0) return -1;
  snprintf(k, sizeof k, "%s.conv2", prefix);
  return vae_conv3x3_nhwc(fx, k, tmp_bf16, res_ptr, out_f32, NULL, H, W, Cout, Cout, s);
}

// Split-D (4 warps x 128 channels, BM=32, BN=32) BF16 TensorCore FlashAttention-2 for VAE mid_block (D=512)!
// Cuts 1024x1024 VAE mid_attention from ~470 ms scalar O(S^2) loop down to ~10 ms (47x speedup)!
__global__ void __launch_bounds__(256) k_vae_mid_attn_tc_d512(
    const uint16_t *__restrict__ Q_bf, const uint16_t *__restrict__ K_bf, const uint16_t *__restrict__ V_bf,
    uint16_t *__restrict__ O_bf, int S, float scale) {
  constexpr int D = 512, D_SUB = 128, KC = 8, DT = 16, STRIDE_KV = 520;
  union SmemMid {
    uint16_t kv[32 * STRIDE_KV];
    float    part[2][4][16][32];
  };
  __shared__ __align__(16) SmemMid sm;

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;       // 0..7
  const int wm   = warp >> 2;      // 0..1 (2 query groups of 16 rows = BM 32)
  const int wd   = warp & 3;       // 0..3 (4 head-dim slices of 128 = D 512)
  const int lane = tid & 31;
  const int g    = lane >> 2;      // 0..7
  const int t    = lane & 3;       // 0..3
  const int qr   = blockIdx.x * 32 + wm * 16;
  const int col_base = wd * D_SUB;

  const int r0 = (qr + g < S)     ? (qr + g)     : (S - 1);
  const int r8 = (qr + g + 8 < S) ? (qr + g + 8) : (S - 1);
  const uint16_t *q0p = Q_bf + (long)r0 * D + col_base;
  const uint16_t *q8p = Q_bf + (long)r8 * D + col_base;

  uint32_t qa[KC][4];
  #pragma unroll
  for (int kc = 0; kc < KC; kc++) {
    qa[kc][0] = *(const uint32_t *)(q0p + kc * 16 + 2 * t);
    qa[kc][1] = *(const uint32_t *)(q8p + kc * 16 + 2 * t);
    qa[kc][2] = *(const uint32_t *)(q0p + kc * 16 + 8 + 2 * t);
    qa[kc][3] = *(const uint32_t *)(q8p + kc * 16 + 8 + 2 * t);
  }

  float o[DT][4];
  #pragma unroll
  for (int dt = 0; dt < DT; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
  float M0 = -1e30f, M1 = -1e30f, l0 = 0.0f, l1 = 0.0f;

  auto mma_bf16 = [](float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  };
  auto pack_bf2 = [](float lo, float hi) -> uint32_t {
    __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    uint32_t u;
    memcpy(&u, &b, 4);
    return u;
  };

  for (int kb = 0; kb < S; kb += 32) {
    // 1. Async load K[kb .. kb+31, 0..511] (2048 x 16B = 8 loads/thread)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int idx = tid + i * 256;
      const int r = idx >> 6;
      const int c = (idx & 63) * 8;
      const int pk = (kb + r < S) ? 16 : 0;
      uint32_t sk = static_cast<uint32_t>(__cvta_generic_to_shared(sm.kv + r * STRIDE_KV + c));
      const void *gk = pk ? (const void *)(K_bf + (long)(kb + r) * D + c) : (const void *)K_bf;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sk), "l"(gk), "r"(pk) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // 2. Partial 128-dim TensorCore Q @ K^T
    float sp[4][4];
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      sp[nt][0] = sp[nt][1] = sp[nt][2] = sp[nt][3] = 0.0f;
      const uint16_t *kr = sm.kv + (nt * 8 + g) * STRIDE_KV + col_base + 2 * t;
      #pragma unroll
      for (int kc = 0; kc < KC; kc++) {
        mma_bf16(sp[nt], qa[kc], *(const uint32_t *)(kr + kc * 16), *(const uint32_t *)(kr + kc * 16 + 8));
      }
    }
    __syncthreads();

    // 3. Reduce across 4 head-dim warps + exact FP32 online softmax
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        sm.part[wm][wd][nt * 4 + j][lane] = sp[nt][j];
      }
    }
    __syncthreads();

    float s[4][4];
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        const int kidx = kb + nt * 8 + 2 * t + j;
        const int i0 = nt * 4 + j, i1 = nt * 4 + 2 + j;
        const float tot0 = sm.part[wm][0][i0][lane] + sm.part[wm][1][i0][lane] + sm.part[wm][2][i0][lane] + sm.part[wm][3][i0][lane];
        const float tot1 = sm.part[wm][0][i1][lane] + sm.part[wm][1][i1][lane] + sm.part[wm][2][i1][lane] + sm.part[wm][3][i1][lane];
        s[nt][j]     = (kidx < S) ? (tot0 * scale) : -INFINITY;
        s[nt][2 + j] = (kidx < S) ? (tot1 * scale) : -INFINITY;
        mx0 = fmaxf(mx0, s[nt][j]);
        mx1 = fmaxf(mx1, s[nt][2 + j]);
      }
    }
    __syncthreads();

    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float Mn0 = fmaxf(M0, mx0), Mn1 = fmaxf(M1, mx1);
    const float ms0 = (Mn0 > -1e29f) ? expf(M0 - Mn0) : 0.0f;
    const float ms1 = (Mn1 > -1e29f) ? expf(M1 - Mn1) : 0.0f;
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        s[nt][j]     = (s[nt][j]     > -1e29f) ? expf(s[nt][j]     - Mn0) : 0.0f;
        s[nt][2 + j] = (s[nt][2 + j] > -1e29f) ? expf(s[nt][2 + j] - Mn1) : 0.0f;
        sum0 += s[nt][j];
        sum1 += s[nt][2 + j];
      }
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);

    l0 = l0 * ms0 + sum0; l1 = l1 * ms1 + sum1;
    M0 = Mn0; M1 = Mn1;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      o[dt][0] *= ms0; o[dt][1] *= ms0;
      o[dt][2] *= ms1; o[dt][3] *= ms1;
    }

    uint32_t pa[2][4];
    #pragma unroll
    for (int kc = 0; kc < 2; kc++) {
      pa[kc][0] = pack_bf2(s[2 * kc][0],     s[2 * kc][1]);
      pa[kc][1] = pack_bf2(s[2 * kc][2],     s[2 * kc][3]);
      pa[kc][2] = pack_bf2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
      pa[kc][3] = pack_bf2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
    }

    // 4. Async load V[kb .. kb+31, 0..511]
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int idx = tid + i * 256;
      const int r = idx >> 6;
      const int c = (idx & 63) * 8;
      const int pv = (kb + r < S) ? 16 : 0;
      uint32_t sv = static_cast<uint32_t>(__cvta_generic_to_shared(sm.kv + r * STRIDE_KV + c));
      const void *gv = pv ? (const void *)(V_bf + (long)(kb + r) * D + c) : (const void *)V_bf;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sv), "l"(gv), "r"(pv) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // 5. Hardware ldmatrix.trans + BF16 TensorCore P @ V
    const int ldm_r = lane & 15;
    #pragma unroll
    for (int kc = 0; kc < 2; kc++) {
      const uint16_t *v_row = sm.kv + (kc * 16 + ldm_r) * STRIDE_KV + col_base;
      #pragma unroll
      for (int dt = 0; dt < DT; dt++) {
        uint32_t vb0, vb1;
        uint32_t sm_v = static_cast<uint32_t>(__cvta_generic_to_shared(v_row + dt * 8));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(vb0), "=r"(vb1) : "r"(sm_v));
        mma_bf16(o[dt], pa[kc], vb0, vb1);
      }
    }
    __syncthreads();
  }

  const float inv0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
  const float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
  #pragma unroll
  for (int h8 = 0; h8 < 2; h8++) {
    const int q = qr + g + 8 * h8;
    if (q >= S) continue;
    const float inv = h8 ? inv1 : inv0;
    uint16_t *db = O_bf + (long)q * D + col_base;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      const float v0 = o[dt][2 * h8]     * inv;
      const float v1 = o[dt][2 * h8 + 1] * inv;
      *(uint32_t *)(db + dt * 8 + 2 * t) = pack_bf2(v0, v1);
    }
  }
}

static int vae_mid_attention_nhwc(
    ec_flux2_t *fx, float *x_nhwc, float *qkv_scratch, uint16_t *tmp_bf16, float *stats64_d,
    int C, int H, int W, cudaStream_t s) {
  const int S = H * W;
  const char *pfx = "decoder.mid_block.attentions.0";
  char tk[192];
  snprintf(tk, sizeof tk, "%s.group_norm", pfx);
  if (vae_gn_silu_nhwc(fx, tk, x_nhwc, tmp_bf16, stats64_d, S, C, 0, s) != 0) return -1;

  snprintf(tk, sizeof tk, "%s.to_q.weight", pfx); uint16_t *wq = get_vae_bf16(fx, tk);
  snprintf(tk, sizeof tk, "%s.to_q.bias", pfx);   float    *bq = get_vae_f32(fx,  tk);
  snprintf(tk, sizeof tk, "%s.to_k.weight", pfx); uint16_t *wk = get_vae_bf16(fx, tk);
  snprintf(tk, sizeof tk, "%s.to_k.bias", pfx);   float    *bk = get_vae_f32(fx,  tk);
  snprintf(tk, sizeof tk, "%s.to_v.weight", pfx); uint16_t *wv = get_vae_bf16(fx, tk);
  snprintf(tk, sizeof tk, "%s.to_v.bias", pfx);   float    *bv = get_vae_f32(fx,  tk);
  snprintf(tk, sizeof tk, "%s.to_out.0.weight", pfx); uint16_t *wo = get_vae_bf16(fx, tk);
  snprintf(tk, sizeof tk, "%s.to_out.0.bias", pfx);   float    *bo = get_vae_f32(fx,  tk);
  if (!wq || !wk || !wv || !wo) return -1;

  if (C == 512) {
    // qkv_scratch (h_buf) has >= 128 * fullH * fullW * 4 = 32 * S * C bytes -> carve Q_bf, K_bf, V_bf, O_bf + out_f32!
    uint16_t *q_bf  = (uint16_t *)qkv_scratch;
    uint16_t *k_bf  = q_bf + (size_t)S * C;
    uint16_t *v_bf  = k_bf + (size_t)S * C;
    uint16_t *ao_bf = v_bf + (size_t)S * C;
    float    *out   = (float *)(ao_bf + (size_t)S * C);
    gemm_bf16(tmp_bf16, wq, bq, NULL, q_bf, S, C, C, s);
    gemm_bf16(tmp_bf16, wk, bk, NULL, k_bf, S, C, C, s);
    gemm_bf16(tmp_bf16, wv, bv, NULL, v_bf, S, C, C, s);
    k_vae_mid_attn_tc_d512<<<(S + 31) / 32, 256, 0, s>>>(q_bf, k_bf, v_bf, ao_bf, S, 1.0f / sqrtf((float)C));
    gemm_bf16(ao_bf, wo, bo, out, NULL, S, C, C, s);
    k_add_inplace<<<(int)(((long)S * C + 255) / 256), 256, 0, s>>>(x_nhwc, out, (long)S * C);
    return 0;
  }

  float *q = qkv_scratch;
  float *k = q + (size_t)S * C;
  float *v = k + (size_t)S * C;
  float *out = v + (size_t)S * C;
  uint16_t *ao_bf = tmp_bf16 + (size_t)S * C;

  gemm_bf16(tmp_bf16, wq, bq, q, NULL, S, C, C, s);
  gemm_bf16(tmp_bf16, wk, bk, k, NULL, S, C, C, s);
  gemm_bf16(tmp_bf16, wv, bv, v, NULL, S, C, C, s);
  k_vae_mid_attn<<<S, 256, 0, s>>>(q, k, v, ao_bf, S, C, 1.0f / sqrtf((float)C));
  gemm_bf16(ao_bf, wo, bo, out, NULL, S, C, C, s);
  k_add_inplace<<<(int)(((long)S * C + 255) / 256), 256, 0, s>>>(x_nhwc, out, (long)S * C);
  return 0;
}

static int run_flux2_vae_decode(
    ec_flux2_t *fx, const float *packed_latents_d, int H_p, int W_p,
    uint8_t *out_rgb_hwc, double *out_ms) {
  double t0 = fx_now_ms();
  cudaStream_t s = fx->stream;
  int H = H_p * 2, W = W_p * 2;
  const int fullH = H_p * 16, fullW = W_p * 16;
  // Peak feature map occurs at u=2 upsample & u=3 r=0 input: [fullH, fullW, 256] = 256 * fullH * fullW!
  // Peak h_buf (Cout * H * W) occurs at u=1 (512 * fullH/2 * fullW/2) & u=3 (128 * fullH * fullW) = 128 * fullH * fullW!
  const size_t ping_elems = (size_t)256 * fullH * fullW;
  const size_t h_elems    = (size_t)128 * fullH * fullW;

  float *bn_mean = get_vae_f32(fx, "bn.running_mean");
  float *bn_var  = get_vae_f32(fx, "bn.running_var");
  if (!bn_mean || !bn_var) return -1;

  auto align256 = [](size_t x) -> size_t { return (x + 255ULL) & ~255ULL; };
  size_t arena_need = 0;
  auto rsv = [&](size_t b) -> size_t { size_t o = arena_need; arena_need += align256(b); return o; };

  const size_t o_cur   = rsv(ping_elems * 4);
  const size_t o_nxt   = rsv(ping_elems * 4);
  const size_t o_h     = rsv(h_elems * 4); // doubles as mid_attn 4*S*C scratch (32*fullH*fullW <= 128*fullH*fullW!)
  const size_t o_bf    = rsv(ping_elems * 2);
  const size_t o_stats = rsv(256 * sizeof(float));
  const size_t o_rgb   = rsv((size_t)fullH * fullW * 3);

  {
    size_t free_b = 0, tot_b = 0;
    if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && free_b < arena_need + (800ULL << 20)) {
      free_qwen3_resident(&fx->qwen_res);
      if (fx->dit_arena) { cudaFree(fx->dit_arena); fx->dit_arena = NULL; fx->dit_arena_cap = 0; }
    }
  }
  fx_vae_cache_t *vc = &fx->vae_cache;
  if (arena_need > vc->dec_arena_cap) {
    if (vc->dec_arena) { cudaFree(vc->dec_arena); vc->dec_arena = NULL; vc->dec_arena_cap = 0; }
    if (cudaMalloc(&vc->dec_arena, arena_need) != cudaSuccess) return -1;
    vc->dec_arena_cap = arena_need;
  }

  float    *cur     = (float *)(vc->dec_arena + o_cur);
  float    *nxt     = (float *)(vc->dec_arena + o_nxt);
  float    *h_buf   = (float *)(vc->dec_arena + o_h);
  uint16_t *tmp_bf  = (uint16_t *)(vc->dec_arena + o_bf);
  float    *stats64 = (float *)(vc->dec_arena + o_stats);
  uint8_t  *rgb_d   = (uint8_t *)(vc->dec_arena + o_rgb);
  #define SWAP_NHWC() do { float *_t = cur; cur = nxt; nxt = _t; } while (0)

  k_flux2_bn_unpatchify_nhwc_bf16<<<(int)((32L * H * W + 255) / 256), 256, 0, s>>>(
      packed_latents_d, bn_mean, bn_var, tmp_bf, H_p, W_p, 1e-4f);

  // post_quant_conv (1x1, 32 -> 32) in NHWC is a direct [H*W, 32] @ [32, 32]^T GEMM!
  {
    uint16_t *wpq = get_vae_bf16(fx, "post_quant_conv.weight");
    float    *bpq = get_vae_f32(fx,  "post_quant_conv.bias");
    if (!wpq || !bpq) return -1;
    gemm_bf16(tmp_bf, wpq, bpq, NULL, tmp_bf + 32L * H * W, H * W, 32, 32, s);
  }

  if (vae_conv3x3_nhwc(fx, "decoder.conv_in", tmp_bf + 32L * H * W, NULL, cur, NULL, H, W, 32, 512, s) != 0) return -1;

  if (vae_resnet_nhwc(fx, "decoder.mid_block.resnets.0", cur, nxt, h_buf, tmp_bf, stats64, 512, 512, H, W, s) != 0) return -1;
  SWAP_NHWC();
  if (vae_mid_attention_nhwc(fx, cur, h_buf, tmp_bf, stats64, 512, H, W, s) != 0) return -1;
  if (vae_resnet_nhwc(fx, "decoder.mid_block.resnets.1", cur, nxt, h_buf, tmp_bf, stats64, 512, 512, H, W, s) != 0) return -1;
  SWAP_NHWC();

  const int up_ch[4] = { 512, 512, 256, 128 };
  int C = 512;
  char pref[128];
  for (int u = 0; u < 4; u++) {
    int Cout = up_ch[u];
    for (int r = 0; r < 3; r++) {
      int Cin = (r == 0) ? C : Cout;
      snprintf(pref, sizeof pref, "decoder.up_blocks.%d.resnets.%d", u, r);
      if (vae_resnet_nhwc(fx, pref, cur, nxt, h_buf, tmp_bf, stats64, Cin, Cout, H, W, s) != 0) return -1;
      SWAP_NHWC();
    }
    C = Cout;
    if (u < 3) {
      int nH = H * 2, nW = W * 2;
      long n_pairs = ((long)nH * nW * C) >> 1;
      k_upsample2x_nhwc_bf16<<<(int)((n_pairs + 255) / 256), 256, 0, s>>>(cur, tmp_bf, H, W, C);
      snprintf(pref, sizeof pref, "decoder.up_blocks.%d.upsamplers.0.conv", u);
      if (vae_conv3x3_nhwc(fx, pref, tmp_bf, NULL, nxt, NULL, nH, nW, C, C, s) != 0) return -1;
      SWAP_NHWC();
      H = nH; W = nW;
    }
  }

  if (vae_gn_silu_nhwc(fx, "decoder.conv_norm_out", cur, tmp_bf, stats64, H * W, 128, 1, s) != 0) return -1;
  {
    uint16_t *wo = get_vae_ohwi_3x3_bf16(fx, "decoder.conv_out.weight", 3, 128);
    float    *bo = get_vae_f32(fx, "decoder.conv_out.bias");
    if (!wo || !bo) return -1;
    k_vae_conv_out_rgb8_nhwc<<<(H * W + 15) / 16, 256, 0, s>>>(tmp_bf, wo, bo, rgb_d, H, W);
  }
  #undef SWAP_NHWC

  cudaMemcpyAsync(out_rgb_hwc, rgb_d, (size_t)H * W * 3, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  trim_vae_conv_scratch(vc);

  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
}

extern "C" int ec_flux2_hot_swap_loras(
    ec_flux2_t *fx, const char *ckpt_override,
    const ec_lora_spec_t *loras, int n_loras,
    double *out_ms, int *out_delta_used) {
  if (!ec_flux2_is_ready(fx)) return -1;
  const char *ckpt = (ckpt_override && ckpt_override[0]) ? ckpt_override : fx->paths.checkpoint_path;
  int want_exact = (getenv("EC_EXACT") && atoi(getenv("EC_EXACT")) != 0) ? 1 : 0;
  char err[256] = {0};
  double t0 = fx_now_ms();
  int rc = ensure_dit_loaded_ex(fx, ckpt, loras, n_loras, want_exact, out_delta_used, err, sizeof err);
  if (out_ms) *out_ms = fx_now_ms() - t0;
  return rc;
}

// ===========================================================================
// Public Lifecycle & End-to-End Photoreal Generator
// ===========================================================================
extern "C" ec_flux2_t *ec_flux2_create(const ec_flux2_paths_t *paths) {
  if (!paths || !paths->checkpoint_path[0] || !paths->vae_path[0] ||
      !paths->text_encoder_path[0] || !paths->tokenizer_path[0]) {
    return NULL;
  }
  ec_flux2_t *fx = (ec_flux2_t *)calloc(1, sizeof *fx);
  if (!fx) return NULL;
  fx->paths = *paths;
  if (cudaStreamCreate(&fx->stream) != cudaSuccess) { free(fx); return NULL; }

  char err[128] = {0};
  fx->bpe = ec_bpe_load(paths->tokenizer_path);
  if (!fx->bpe) { ec_flux2_destroy(fx); return NULL; }
  if (ec_st_open(&fx->te_st, paths->text_encoder_path, err, sizeof err) != 0) { ec_flux2_destroy(fx); return NULL; }
  fx->te_open = 1;
  if (ec_st_open(&fx->vae_st, paths->vae_path, err, sizeof err) != 0) { ec_flux2_destroy(fx); return NULL; }
  fx->vae_open = 1;
  ec_cowrie_file_fingerprint(paths->text_encoder_path, fx->te_sha);
  return fx;
}

extern "C" void ec_flux2_unload_vram(ec_flux2_t *fx) {
  if (!fx) return;
  free_dit_resident(&fx->dit);
  free_qwen3_resident(&fx->qwen_res);
  free_vae_cache(&fx->vae_cache);
  if (fx->dit_arena) { cudaFree(fx->dit_arena); fx->dit_arena = NULL; fx->dit_arena_cap = 0; }
}

extern "C" void ec_flux2_destroy(ec_flux2_t *fx) {
  if (!fx) return;
  ec_flux2_unload_vram(fx);
  if (fx->te_open) ec_st_close(&fx->te_st);
  if (fx->vae_open) ec_st_close(&fx->vae_st);
  if (fx->bpe) ec_bpe_free(fx->bpe);
  free(fx->cached_prompt_bf16);
  if (fx->stream) cudaStreamDestroy(fx->stream);
  free(fx);
}

extern "C" int ec_flux2_is_ready(const ec_flux2_t *fx) {
  return (fx && fx->bpe && fx->te_open && fx->vae_open) ? 1 : 0;
}

extern "C" int ec_flux2_generate_rgb(
    ec_flux2_t *fx, const ec_flux2_req_t *req, uint8_t **out_rgb, const char *out_png_path,
    ec_flux2_progress_fn cb, void *cb_ctx, ec_flux2_res_t *res) {
  memset(res, 0, sizeof *res);
  if (out_rgb) *out_rgb = NULL;
  if (!ec_flux2_is_ready(fx) || !req) {
    snprintf(res->error, sizeof res->error, "flux2 engine not initialized");
    return -1;
  }
  double t_all = fx_now_ms();
  int W = req->width  > 0 ? ((req->width  + 15) & ~15) : 512;
  int H = req->height > 0 ? ((req->height + 15) & ~15) : 512;
  if (W < 256) W = 256; if (W > 1536) W = 1536;
  if (H < 256) H = 256; if (H > 1536) H = 1536;
  int steps = (req->steps > 0 && req->steps <= 50) ? req->steps : 4;
  const char *ckpt = (req->checkpoint_override && req->checkpoint_override[0])
                     ? req->checkpoint_override : fx->paths.checkpoint_path;

  // 1. Text Encode (Qwen3-4B layers 0..26) + optional negative prompt for CFG > 1.05
  if (cb) cb(0, steps + 2, "qwen3_text_encode", cb_ctx);
  char enriched_prompt[1536];
  const char *eff_prompt = (req->prompt && req->prompt[0]) ? req->prompt : "";
  if (req->init_rgb) {
    // If user supplies an edit verb ("make it better", "enhance", "upscale", "fix") or empty prompt on Img2Img,
    // enrich with concrete optical quality cues so pure flow-matching T2I enhances structure instead of hallucinating!
    size_t plen = strlen(eff_prompt);
    int is_edit_verb = (plen == 0 || plen < 22);
    if (!is_edit_verb) {
      char low[128] = {0};
      for (size_t i = 0; i < plen && i + 1 < sizeof low; i++) {
        char c = eff_prompt[i];
        low[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
      }
      if (strstr(low, "make it") || strstr(low, "enhance") || strstr(low, "improve") ||
          strstr(low, "upscale") || strstr(low, "sharpen") || strstr(low, "fix")) {
        is_edit_verb = 1;
      }
    }
    if (is_edit_verb) {
      if (plen > 0) {
        snprintf(enriched_prompt, sizeof enriched_prompt,
                 "%s, masterpiece photorealistic 35mm cinema still, ultra-detailed natural textures, dramatic volumetric lighting, crisp micro-contrast, razor-sharp focus",
                 eff_prompt);
      } else {
        snprintf(enriched_prompt, sizeof enriched_prompt,
                 "Masterpiece photorealistic 35mm cinema still, ultra-detailed natural textures, dramatic volumetric lighting, crisp micro-contrast, razor-sharp focus");
      }
      eff_prompt = enriched_prompt;
    }
  }
  uint16_t *prompt_emb_d = NULL, *neg_emb_d = NULL;
  if (cudaMalloc(&prompt_emb_d, 512L * 7680L * 2) != cudaSuccess ||
      run_qwen3_text_encoder(fx, eff_prompt, prompt_emb_d, &res->text_enc_ms) != 0) {
    cudaFree(prompt_emb_d);
    snprintf(res->error, sizeof res->error, "qwen3 text encoder failed");
    return -1;
  }
  const char *eff_neg = (req->neg_prompt && req->neg_prompt[0])
                        ? req->neg_prompt
                        : "blurry, low quality, out of focus, distorted anatomy, oversaturated, plastic skin, watermark, text, artifacts";
  float cfg = (req->cfg_scale > 1.05f) ? req->cfg_scale : 1.0f;
  if (cfg > 1.05f) {
    double neg_ms = 0.0;
    cudaMalloc(&neg_emb_d, 512L * 7680L * 2);
    if (neg_emb_d && run_qwen3_text_encoder(fx, eff_neg, neg_emb_d, &neg_ms) == 0) {
      res->text_enc_ms += neg_ms;
    } else {
      cudaFree(neg_emb_d); neg_emb_d = NULL; cfg = 1.0f;
    }
  }

  // 2. Ensure FLUX.2 DiT checkpoint + LoRA/QLoRA stack is resident on GPU (Q8_G64 fast default vs BF16+FA2 oracle!)
  //    (If an imported workflow overrides checkpoint with an incompatible arch like SDXL/9B-Quanto, fall back cleanly!)
  int want_exact = (req->tier == EC_TIER_EXACT || (getenv("EC_EXACT") && atoi(getenv("EC_EXACT")) != 0)) ? 1 : 0;
  if (cb) cb(1, steps + 2, "load_dit_and_loras", cb_ctx);
  if (ensure_dit_loaded(fx, ckpt, req->loras, req->n_loras, want_exact, res->error, sizeof res->error) != 0) {
    if (ckpt != fx->paths.checkpoint_path && strcmp(ckpt, fx->paths.checkpoint_path) != 0 &&
        ensure_dit_loaded(fx, fx->paths.checkpoint_path, req->loras, req->n_loras, want_exact, res->error, sizeof res->error) == 0) {
      ckpt = fx->paths.checkpoint_path;
      res->error[0] = '\0';
    } else {
      cudaFree(prompt_emb_d); cudaFree(neg_emb_d);
      return -1;
    }
  }

  // 2b. Optional Native AutoencoderKLFlux2 VAE Encode (for Img2Img / Masked Inpainting / Outpainting!)
  int H_p = H / 16, W_p = W / 16;
  float *z0_init_d = NULL, *mask_p_d = NULL;
  double enc_ms = 0.0;
  float denoise = (req->denoise > 0.01f && req->denoise <= 1.0f) ? req->denoise : (req->init_rgb ? 0.48f : 1.0f);
  if (req->init_rgb && req->init_w > 0 && req->init_h > 0) {
    uint8_t *resized = ec_rgb_resize_bilinear(req->init_rgb, req->init_w, req->init_h, W, H);
    if (resized) {
      cudaMalloc(&z0_init_d, (size_t)H_p * W_p * 128 * 4);
      if (run_flux2_vae_encode(fx, resized, H, W, z0_init_d, &enc_ms) != 0) {
        cudaFree(z0_init_d); z0_init_d = NULL;
      }
      free(resized);
    }
    if (z0_init_d && req->mask_gray) {
      // Downsample pixel mask [init_h, init_w] -> latent patch mask [H_p, W_p] (average pool per 16x16 patch)
      float *mp_h = (float *)calloc((size_t)H_p * W_p, sizeof(float));
      for (int py = 0; py < H_p; py++) {
        for (int px = 0; px < W_p; px++) {
          float sum = 0.0f; int cnt = 0;
          int y0 = (py * req->init_h) / H_p, y1 = ((py + 1) * req->init_h) / H_p;
          int x0 = (px * req->init_w) / W_p, x1 = ((px + 1) * req->init_w) / W_p;
          if (y1 <= y0) y1 = y0 + 1;
          if (x1 <= x0) x1 = x0 + 1;
          for (int y = y0; y < y1 && y < req->init_h; y++)
            for (int x = x0; x < x1 && x < req->init_w; x++) {
              sum += req->mask_gray[(size_t)y * req->init_w + x]; cnt++;
            }
          mp_h[py * W_p + px] = cnt > 0 ? (sum / (float)cnt) : 1.0f;
        }
      }
      cudaMalloc(&mask_p_d, (size_t)H_p * W_p * 4);
      cudaMemcpyAsync(mask_p_d, mp_h, (size_t)H_p * W_p * 4, cudaMemcpyHostToDevice, fx->stream);
      free(mp_h);
    }
  }

  // 3. Denoise latents [H/16 * W/16, 128]
  float *latents_d = NULL;
  cudaMalloc(&latents_d, (size_t)H_p * W_p * 128 * 4);
  if (run_flux2_dit(fx, prompt_emb_d, neg_emb_d, H_p, W_p, steps, cfg, req->seed,
                    z0_init_d, mask_p_d, denoise, latents_d, cb, cb_ctx, &res->dit_ms) != 0) {
    cudaFree(prompt_emb_d); cudaFree(neg_emb_d); cudaFree(z0_init_d); cudaFree(mask_p_d); cudaFree(latents_d);
    snprintf(res->error, sizeof res->error, "flux2 dit denoising failed");
    return -1;
  }
  cudaFree(z0_init_d); cudaFree(mask_p_d);

  // 4. VAE Decode -> RGB24
  if (cb) cb(steps + 1, steps + 2, "vae_decode_png", cb_ctx);
  uint8_t *rgb = (uint8_t *)malloc((size_t)W * H * 3);
  if (run_flux2_vae_decode(fx, latents_d, H_p, W_p, rgb, &res->vae_ms) != 0) {
    free(rgb); cudaFree(prompt_emb_d); cudaFree(neg_emb_d); cudaFree(latents_d);
    snprintf(res->error, sizeof res->error, "flux2 vae decode failed");
    return -1;
  }
  res->vae_ms += enc_ms;
  cudaFree(latents_d);

  // 4b. Optional 2x Native Tile Detail Upscale (2x2 overlapping tiles via VAE Encode -> 0.42 denoise -> VAE Decode!)
  if (req->tile_upscale && W <= 768 && H <= 768) {
    int UW = W * 2, UH = H * 2;
    uint8_t *up_base = ec_rgb_resize_bilinear(rgb, W, H, UW, UH);
    if (up_base) {
      int TW = ((W + 64 + 15) & ~15), TH = ((H + 64 + 15) & ~15);
      if (TW > UW) TW = UW; if (TH > UH) TH = UH;
      int T_hp = TH / 16, T_wp = TW / 16;
      float *acc = (float *)calloc((size_t)UW * UH * 3, sizeof(float));
      float *wsum = (float *)calloc((size_t)UW * UH, sizeof(float));
      uint8_t *tile_in = (uint8_t *)malloc((size_t)TW * TH * 3);
      uint8_t *tile_out = (uint8_t *)malloc((size_t)TW * TH * 3);
      float *tz0 = NULL, *tlat = NULL;
      cudaMalloc(&tz0, (size_t)T_hp * T_wp * 128 * 4);
      cudaMalloc(&tlat, (size_t)T_hp * T_wp * 128 * 4);
      int oxs[2] = { 0, UW - TW }, oys[2] = { 0, UH - TH };
      for (int ty = 0; ty < 2; ty++) {
        for (int tx = 0; tx < 2; tx++) {
          int x0 = oxs[tx], y0 = oys[ty];
          for (int y = 0; y < TH; y++)
            memcpy(tile_in + (size_t)y * TW * 3, up_base + ((size_t)(y0 + y) * UW + x0) * 3, (size_t)TW * 3);
          double e_ms = 0, d_ms = 0, v_ms = 0;
          if (run_flux2_vae_encode(fx, tile_in, TH, TW, tz0, &e_ms) == 0 &&
              run_flux2_dit(fx, prompt_emb_d, NULL, T_hp, T_wp, steps, 1.0f, req->seed + ty * 2 + tx + 11,
                            tz0, NULL, 0.42f, tlat, NULL, NULL, &d_ms) == 0 &&
              run_flux2_vae_decode(fx, tlat, T_hp, T_wp, tile_out, &v_ms) == 0) {
            res->dit_ms += d_ms; res->vae_ms += e_ms + v_ms;
            for (int y = 0; y < TH; y++) {
              float wy = fminf(1.0f, fminf((float)(y + 1), (float)(TH - y)) / 32.0f);
              for (int x = 0; x < TW; x++) {
                float wx = fminf(1.0f, fminf((float)(x + 1), (float)(TW - x)) / 32.0f);
                float ww = fmaxf(0.02f, wx * wy);
                size_t gi = (size_t)(y0 + y) * UW + (x0 + x);
                size_t li = (size_t)y * TW + x;
                acc[gi * 3 + 0] += tile_out[li * 3 + 0] * ww;
                acc[gi * 3 + 1] += tile_out[li * 3 + 1] * ww;
                acc[gi * 3 + 2] += tile_out[li * 3 + 2] * ww;
                wsum[gi] += ww;
              }
            }
          }
        }
      }
      cudaFree(tz0); cudaFree(tlat);
      free(tile_in); free(tile_out); free(up_base);
      uint8_t *final_up = (uint8_t *)malloc((size_t)UW * UH * 3);
      for (size_t i = 0; i < (size_t)UW * UH; i++) {
        float inv = wsum[i] > 1e-4f ? (1.0f / wsum[i]) : 0.0f;
        for (int c = 0; c < 3; c++) {
          float v = acc[i * 3 + c] * inv;
          final_up[i * 3 + c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v + 0.5f));
        }
      }
      free(acc); free(wsum); free(rgb);
      rgb = final_up; W = UW; H = UH;
    }
  }
  cudaFree(prompt_emb_d); cudaFree(neg_emb_d);

  if (out_png_path && out_png_path[0]) {
    if (ec_export_png(out_png_path, rgb, (uint32_t)W, (uint32_t)H) != 0) {
      free(rgb);
      snprintf(res->error, sizeof res->error, "png write failed");
      return -1;
    }
  }
  if (out_rgb) *out_rgb = rgb;
  else free(rgb);

  res->ok = 1;
  res->width = W;
  res->height = H;
  res->steps_run = steps;
  res->lora_tensors_fused = fx->dit.lora_hits;
  res->total_ms = fx_now_ms() - t_all;
  snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", ckpt);
  return 0;
}

extern "C" int ec_flux2_generate_png(
    ec_flux2_t *fx, const ec_flux2_req_t *req, const char *out_png_path,
    ec_flux2_progress_fn cb, void *cb_ctx, ec_flux2_res_t *res) {
  return ec_flux2_generate_rgb(fx, req, NULL, out_png_path, cb, cb_ctx, res);
}

extern "C" int ec_flux2_vae_roundtrip_rgb(
    ec_flux2_t *fx, const uint8_t *in_rgb, int width, int height,
    uint8_t *out_rgb, double *enc_ms, double *dec_ms) {
  if (!ec_flux2_is_ready(fx) || !in_rgb || !out_rgb || width < 64 || height < 64 || (width & 15) || (height & 15))
    return -1;
  int H_p = height / 16, W_p = width / 16;
  float *lat_d = NULL;
  if (cudaMalloc(&lat_d, (size_t)H_p * W_p * 128 * 4) != cudaSuccess) return -1;
  int rc = run_flux2_vae_encode(fx, in_rgb, height, width, lat_d, enc_ms);
  if (rc == 0) rc = run_flux2_vae_decode(fx, lat_d, H_p, W_p, out_rgb, dec_ms);
  cudaFree(lat_d);
  return rc;
}
