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
#include "ec_kernels.cuh"
#include "ec_kernels_v2.cuh"
#include "ec_media.h"
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

static __device__ __forceinline__ void mma_bf16_m16n8k16(
    float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ===========================================================================
// 1. Native PTX BF16 TensorCore GEMM: Y[M, N] = X_bf16[M, K] @ W_bf16[N, K]^T + bias
//    Zero cuBLAS, zero FP16 65504 overflow, exact F32 accumulation!
// ===========================================================================
__global__ void __launch_bounds__(128) k_gemm_bf16_tc(
    const uint16_t *__restrict__ X, const uint16_t *__restrict__ W,
    const float *__restrict__ bias,
    float *__restrict__ Y_f32, uint16_t *__restrict__ Y_bf16,
    int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 32, STRIDE = 40;
  __shared__ __align__(16) uint16_t sX[BM * STRIDE];
  __shared__ __align__(16) uint16_t sW[BN * STRIDE];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wm = (warp >> 1) * 32;
  const int wn = (warp & 1) * 32;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  float acc[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.0f;

  const bool fast_k = ((K & 7) == 0);
  for (int kb = 0; kb < K; kb += BK) {
    if (fast_k && (kb + BK <= K)) {
      #pragma unroll
      for (int iter = 0; iter < 2; iter++) {
        const int idx = threadIdx.x + iter * 128;
        const int r = idx >> 2;
        const int c = (idx & 3) * 8;
        uint4 vx = make_uint4(0, 0, 0, 0);
        if (m0 + r < M) vx = *(const uint4 *)(X + (long)(m0 + r) * K + kb + c);
        *(uint4 *)(sX + r * STRIDE + c) = vx;
        uint4 vw = make_uint4(0, 0, 0, 0);
        if (n0 + r < N) vw = *(const uint4 *)(W + (long)(n0 + r) * K + kb + c);
        *(uint4 *)(sW + r * STRIDE + c) = vw;
      }
    } else {
      for (int idx = threadIdx.x; idx < BM * BK; idx += 128) {
        const int r = idx / BK, c = idx % BK;
        sX[r * STRIDE + c] = (m0 + r < M && kb + c < K) ? X[(long)(m0 + r) * K + kb + c] : 0;
        sW[r * STRIDE + c] = (n0 + r < N && kb + c < K) ? W[(long)(n0 + r) * K + kb + c] : 0;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      const int ko = ks * 16;
      uint32_t a_frag[2][4];
      #pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        const uint16_t *xr0 = sX + (wm + mi * 16 + g) * STRIDE + ko;
        const uint16_t *xr8 = xr0 + 8 * STRIDE;
        a_frag[mi][0] = *(const uint32_t *)(xr0 + 2 * t);
        a_frag[mi][1] = *(const uint32_t *)(xr8 + 2 * t);
        a_frag[mi][2] = *(const uint32_t *)(xr0 + 8 + 2 * t);
        a_frag[mi][3] = *(const uint32_t *)(xr8 + 8 + 2 * t);
      }
      #pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const uint16_t *wr = sW + (wn + ni * 8 + g) * STRIDE + ko;
        const uint32_t b0 = *(const uint32_t *)(wr + 2 * t);
        const uint32_t b1 = *(const uint32_t *)(wr + 8 + 2 * t);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          mma_bf16_m16n8k16(acc[mi][ni], a_frag[mi], b0, b1);
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int mi = 0; mi < 2; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = m0 + wm + mi * 16 + g + 8 * (e >> 1);
        const int c = n0 + wn + ni * 8 + 2 * t + (e & 1);
        if (r < M && c < N) {
          float v = acc[mi][ni][e] + (bias ? bias[c] : 0.0f);
          if (Y_f32) Y_f32[(long)r * N + c] = v;
          if (Y_bf16) Y_bf16[(long)r * N + c] = f32_to_bf16_dev(v);
        }
      }
    }
  }
}

static void gemm_bf16(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t s) {
  dim3 grd((N + 63) / 64, (M + 63) / 64);
  k_gemm_bf16_tc<<<grd, 128, 0, s>>>(X, W, bias, Y_f32, Y_bf16, M, N, K);
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

// ===========================================================================
// 4. AutoencoderKLFlux2 2D VAE Kernels (BN Denorm + Unpatchify + Strip TC Conv2D + Mid Attn)
// ===========================================================================
// Unpack [H_p*W_p, 128] -> BN denorm -> unpatchify 2x2 -> [32, H_p*2, W_p*2] CHW
__global__ void k_flux2_bn_unpatchify(
    const float *packed, const float *bn_mean, const float *bn_var,
    float *out_chw, int H_p, int W_p, float bn_eps) {
  int H_lat = H_p * 2, W_lat = W_p * 2;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = 32L * H_lat * W_lat;
  if (idx >= tot) return;
  int c = (int)(idx / (H_lat * W_lat));
  int rem = (int)(idx % (H_lat * W_lat));
  int y = rem / W_lat, x = rem % W_lat;
  int hp = y >> 1, py = y & 1;
  int wp = x >> 1, px = x & 1;
  int pch = c * 4 + py * 2 + px;
  float v = packed[(long)(hp * W_p + wp) * 128 + pch];
  float std_v = sqrtf(bn_var[pch] + bn_eps);
  out_chw[idx] = v * std_v + bn_mean[pch];
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
// FLUX.2-klein-4B Resident GPU Structures
// ===========================================================================
typedef struct {
  uint16_t *img_qkv, *img_proj, *img_mlp0, *img_mlp2;
  uint16_t *txt_qkv, *txt_proj, *txt_mlp0, *txt_mlp2;
  float    *img_qn, *img_kn, *txt_qn, *txt_kn;
} fx_double_w_t;

typedef struct {
  uint16_t *lin1, *lin2;
  float    *qn, *kn;
} fx_single_w_t;

typedef struct {
  int loaded;
  char ckpt_path[512];
  char lora_sig[1024];
  int  lora_hits;

  uint16_t *img_in, *txt_in, *time_in0, *time_in1;
  uint16_t *mod_img, *mod_txt, *mod_sng;
  uint16_t *final_mod, *final_lin;
  fx_double_w_t dbl[5];
  fx_single_w_t sng[20];
} fx_dit_resident_t;

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

  // Cached prompt embedding [512 * 7680] BF16 on host RAM -> 0.0 ms repeat encode!
  char      cached_prompt[1024];
  uint16_t *cached_prompt_bf16;
  int       has_cached_prompt;
};

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
  // Need ~5.5 GB for 27 BF16 layers + 8 GB for DiT if not loaded yet + 2.5 GB workspace headroom
  size_t need_b = fx->dit.loaded ? (8ULL * 1024ULL * 1024ULL * 1024ULL) : (16ULL * 1024ULL * 1024ULL * 1024ULL);
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

static void free_dit_resident(fx_dit_resident_t *d) {
  if (!d || !d->loaded) return;
  cudaFree(d->img_in); cudaFree(d->txt_in); cudaFree(d->time_in0); cudaFree(d->time_in1);
  cudaFree(d->mod_img); cudaFree(d->mod_txt); cudaFree(d->mod_sng);
  cudaFree(d->final_mod); cudaFree(d->final_lin);
  for (int i = 0; i < 5; i++) {
    cudaFree(d->dbl[i].img_qkv); cudaFree(d->dbl[i].img_proj);
    cudaFree(d->dbl[i].img_mlp0); cudaFree(d->dbl[i].img_mlp2);
    cudaFree(d->dbl[i].txt_qkv); cudaFree(d->dbl[i].txt_proj);
    cudaFree(d->dbl[i].txt_mlp0); cudaFree(d->dbl[i].txt_mlp2);
    cudaFree(d->dbl[i].img_qn); cudaFree(d->dbl[i].img_kn);
    cudaFree(d->dbl[i].txt_qn); cudaFree(d->dbl[i].txt_kn);
  }
  for (int i = 0; i < 20; i++) {
    cudaFree(d->sng[i].lin1); cudaFree(d->sng[i].lin2);
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

static int ensure_dit_loaded(
    ec_flux2_t *fx, const char *ckpt_path,
    const ec_lora_spec_t *loras, int n_loras, char *err, size_t err_cap) {
  char lsig[1024];
  build_lora_sig(loras, n_loras, lsig, sizeof lsig);
  if (fx->dit.loaded && strcmp(fx->dit.ckpt_path, ckpt_path) == 0 && strcmp(fx->dit.lora_sig, lsig) == 0) {
    return 0;
  }
  free_dit_resident(&fx->dit);

  ec_safetensors_t st;
  if (ec_st_open(&st, ckpt_path, err, err_cap) != 0) return -1;

  fx_dit_resident_t *d = &fx->dit;
  cudaStream_t s = fx->stream;
  ec_lora_apply_stats_t lstats = {};

  #define LOAD_W(field, name, out_d, in_d) do { \
    d->field = load_bf16_gpu(&st, name, NULL, s); \
    if (!d->field) { ec_st_close(&st); free_dit_resident(d); snprintf(err, err_cap, "missing %s", name); return -1; } \
    if (n_loras > 0) ec_lora_fuse_onto_bf16(d->field, (out_d), (in_d), name, loras, n_loras, &lstats, s); \
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
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.qkv.weight", i);  LOAD_W(dbl[i].img_qkv,  k, 9216, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.proj.weight", i); LOAD_W(dbl[i].img_proj, k, 3072, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.0.weight", i);     LOAD_W(dbl[i].img_mlp0, k, 18432, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.img_mlp.2.weight", i);     LOAD_W(dbl[i].img_mlp2, k, 3072, 9216);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.qkv.weight", i);  LOAD_W(dbl[i].txt_qkv,  k, 9216, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.proj.weight", i); LOAD_W(dbl[i].txt_proj, k, 3072, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.0.weight", i);     LOAD_W(dbl[i].txt_mlp0, k, 18432, 3072);
    snprintf(k, sizeof k, "double_blocks.%d.txt_mlp.2.weight", i);     LOAD_W(dbl[i].txt_mlp2, k, 3072, 9216);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.norm.query_norm.scale", i); d->dbl[i].img_qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.img_attn.norm.key_norm.scale", i);   d->dbl[i].img_kn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.norm.query_norm.scale", i); d->dbl[i].txt_qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "double_blocks.%d.txt_attn.norm.key_norm.scale", i);   d->dbl[i].txt_kn = load_f32_gpu(&st, k, NULL, s);
  }
  for (int i = 0; i < 20; i++) {
    snprintf(k, sizeof k, "single_blocks.%d.linear1.weight", i); LOAD_W(sng[i].lin1, k, 27648, 3072);
    snprintf(k, sizeof k, "single_blocks.%d.linear2.weight", i); LOAD_W(sng[i].lin2, k, 3072, 12288);
    snprintf(k, sizeof k, "single_blocks.%d.norm.query_norm.scale", i); d->sng[i].qn = load_f32_gpu(&st, k, NULL, s);
    snprintf(k, sizeof k, "single_blocks.%d.norm.key_norm.scale", i);   d->sng[i].kn = load_f32_gpu(&st, k, NULL, s);
  }
  #undef LOAD_W

  cudaStreamSynchronize(s);
  ec_st_close(&st);
  snprintf(d->ckpt_path, sizeof d->ckpt_path, "%s", ckpt_path);
  snprintf(d->lora_sig, sizeof d->lora_sig, "%s", lsig);
  d->lora_hits = lstats.n_tensors_matched;
  d->loaded = 1;
  return 0;
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

  // Cache host copy for instant rerolls
  if (!fx->cached_prompt_bf16) fx->cached_prompt_bf16 = (uint16_t *)malloc(emb_elems * 2);
  if (fx->cached_prompt_bf16 && prompt) {
    cudaMemcpy(fx->cached_prompt_bf16, out_embeds_gpu_bf16, emb_elems * 2, cudaMemcpyDeviceToHost);
    snprintf(fx->cached_prompt, sizeof fx->cached_prompt, "%s", prompt);
    fx->has_cached_prompt = 1;
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
    ec_flux2_t *fx, const uint16_t *prompt_embeds_bf16,
    int H_p, int W_p, int steps, uint32_t seed,
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

  // Initial unit Gaussian latents [N_img, 128]
  float *x_h = (float *)malloc((size_t)N_img * 128 * 4);
  fill_gaussian(x_h, (size_t)N_img * 128, seed);
  cudaMemcpyAsync(out_latents_gpu_f32, x_h, (size_t)N_img * 128 * 4, cudaMemcpyHostToDevice, s);
  free(x_h);

  // Shifted sigmas schedule
  double mu = compute_empirical_mu(N_img, steps);
  double emu = exp(mu);
  float sigmas[64];
  for (int i = 0; i < steps; i++) {
    double t_lin = 1.0 - (double)i * (1.0 - 1.0 / (double)steps) / (double)(steps > 1 ? steps - 1 : 1);
    if (steps == 1) t_lin = 1.0;
    double sh = (t_lin >= 1.0) ? 1.0 : (emu / (emu + (1.0 / t_lin - 1.0)));
    sigmas[i] = (float)sh;
  }
  sigmas[steps] = 0.0f;

  // Allocate reusable step workspaces
  float *rope_cos = NULL, *rope_sin = NULL;
  float *temb_256 = NULL, *temb_0 = NULL, *temb = NULL, *mod_img_v = NULL, *mod_txt_v = NULL, *mod_sng_v = NULL, *fmod_v = NULL;
  uint16_t *temb_256_bf = NULL, *temb_0_bf = NULL, *silu_temb_bf = NULL;
  float *img_d = NULL, *txt_d = NULL, *h_cat = NULL;
  uint16_t *x_bf16 = NULL, *norm_bf = NULL, *norm_txt_bf = NULL, *cat_bf = NULL;
  float *qkv_img = NULL, *qkv_txt = NULL, *attn_out = NULL, *proj_out = NULL, *mlp_2f = NULL;
  float *lin1_out = NULL, *v_pred = NULL;
  __half *Q_h = NULL, *K_h = NULL, *V_h = NULL;

  cudaMalloc(&rope_cos, (size_t)N_tot * 64 * 4);
  cudaMalloc(&rope_sin, (size_t)N_tot * 64 * 4);
  k_flux2_precompute_rope<<<N_tot, 64, 0, s>>>(rope_cos, rope_sin, N_txt, H_p, W_p, 2000.0);

  cudaMalloc(&temb_256, 256 * 4); cudaMalloc(&temb_256_bf, 256 * 2);
  cudaMalloc(&temb_0, E * 4);     cudaMalloc(&temb_0_bf, E * 2);
  cudaMalloc(&temb, E * 4);       cudaMalloc(&silu_temb_bf, E * 2);
  cudaMalloc(&mod_img_v, 6 * E * 4); cudaMalloc(&mod_txt_v, 6 * E * 4);
  cudaMalloc(&mod_sng_v, 3 * E * 4); cudaMalloc(&fmod_v, 2 * E * 4);

  cudaMalloc(&x_bf16, (size_t)N_img * 128 * 2);
  cudaMalloc(&img_d, (size_t)N_img * E * 4);
  cudaMalloc(&txt_d, (size_t)N_txt * E * 4);
  cudaMalloc(&h_cat, (size_t)N_tot * E * 4);
  cudaMalloc(&norm_bf, (size_t)N_tot * E * 2);
  cudaMalloc(&norm_txt_bf, (size_t)N_txt * E * 2);
  cudaMalloc(&cat_bf, (size_t)N_tot * (E + F) * 2);
  cudaMalloc(&qkv_img, (size_t)N_img * 3 * E * 4);
  cudaMalloc(&qkv_txt, (size_t)N_txt * 3 * E * 4);
  cudaMalloc(&attn_out, (size_t)N_tot * E * 4);
  cudaMalloc(&proj_out, (size_t)N_tot * E * 4);
  cudaMalloc(&mlp_2f, (size_t)(N_img > N_txt ? N_img : N_txt) * 2 * F * 4);
  cudaMalloc(&lin1_out, (size_t)N_tot * (3 * E + 2 * F) * 4);
  cudaMalloc(&v_pred, (size_t)N_img * 128 * 4);
  cudaMalloc(&Q_h, (size_t)H * N_pad * D * 2);
  cudaMalloc(&K_h, (size_t)H * N_pad * D * 2);
  cudaMalloc(&V_h, (size_t)H * N_pad * D * 2);

  for (int step = 0; step < steps; step++) {
    if (cb && cb(step, steps, "flux2_dit", cb_ctx) != 0) break;

    float sigma = sigmas[step];
    float dt = sigmas[step + 1] - sigma;

    // 1. Timestep embedding (t = sigma * 1000) -> [cos, sin] -> time_in
    float t_h[256];
    for (int i = 0; i < 128; i++) {
      double freq = exp(-log(10000.0) * (double)i / 128.0);
      double arg = (double)sigma * 1000.0 * freq;
      t_h[i]       = (float)cos(arg);
      t_h[128 + i] = (float)sin(arg);
    }
    cudaMemcpyAsync(temb_256, t_h, sizeof t_h, cudaMemcpyHostToDevice, s);
    k_f32_to_bf16<<<1, 256, 0, s>>>(temb_256, temb_256_bf, 256);
    gemm_bf16(temb_256_bf, d->time_in0, NULL, temb_0, NULL, 1, E, 256, s);
    k_silu_f32_to_bf16<<<(E + 255) / 256, 256, 0, s>>>(temb_0, temb_0_bf, E);
    gemm_bf16(temb_0_bf, d->time_in1, NULL, temb, NULL, 1, E, E, s);
    k_silu_f32_to_bf16<<<(E + 255) / 256, 256, 0, s>>>(temb, silu_temb_bf, E);

    // Shared modulations
    gemm_bf16(silu_temb_bf, d->mod_img, NULL, mod_img_v, NULL, 1, 6 * E, E, s);
    gemm_bf16(silu_temb_bf, d->mod_txt, NULL, mod_txt_v, NULL, 1, 6 * E, E, s);
    gemm_bf16(silu_temb_bf, d->mod_sng, NULL, mod_sng_v, NULL, 1, 3 * E, E, s);

    // 2. Input projections
    k_f32_to_bf16<<<((long)N_img * 128 + 255) / 256, 256, 0, s>>>(out_latents_gpu_f32, x_bf16, (long)N_img * 128);
    gemm_bf16(x_bf16, d->img_in, NULL, img_d, NULL, N_img, E, 128, s);
    gemm_bf16(prompt_embeds_bf16, d->txt_in, NULL, txt_d, NULL, N_txt, E, 7680, s);

    // 3. 5 Double-Stream Blocks
    for (int b = 0; b < 5; b++) {
      const fx_double_w_t *w = &d->dbl[b];
      k_adaln_bf16<<<N_img, 256, 0, s>>>(img_d, mod_img_v, mod_img_v + E, norm_bf, N_img, E, 1e-6f);
      k_adaln_bf16<<<N_txt, 256, 0, s>>>(txt_d, mod_txt_v, mod_txt_v + E, norm_txt_bf, N_txt, E, 1e-6f);

      gemm_bf16(norm_bf,     w->img_qkv, NULL, qkv_img, NULL, N_img, 3 * E, E, s);
      gemm_bf16(norm_txt_bf, w->txt_qkv, NULL, qkv_txt, NULL, N_txt, 3 * E, E, s);

      // Pack [txt (0..N_txt-1), img (N_txt..N_tot-1)] into [H, N_pad, D] FP16 with QK-RMSNorm + 4D RoPE
      int w_txt = N_txt * H, w_img = N_img * H;
      k_flux2_pack_qkv_rope<<<(w_txt + 7) / 8, 256, 0, s>>>(
          qkv_txt, 3 * E, w->txt_qn, w->txt_kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_txt, 0, N_pad, H, D, 1e-6f);
      k_flux2_pack_qkv_rope<<<(w_img + 7) / 8, 256, 0, s>>>(
          qkv_img, 3 * E, w->img_qn, w->img_kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_img, N_txt, N_pad, H, D, 1e-6f);

      ec_launch_fa2_f16(Q_h, K_h, V_h, attn_out, NULL, N_tot, N_tot, N_pad, N_pad, H, D, attn_scale, 0, s);

      // Output projections + gated residual (txt = 0..N_txt-1, img = N_txt..N_tot-1)
      k_f32_to_bf16<<<((long)N_txt * E + 255) / 256, 256, 0, s>>>(attn_out, norm_txt_bf, (long)N_txt * E);
      k_f32_to_bf16<<<((long)N_img * E + 255) / 256, 256, 0, s>>>(attn_out + (long)N_txt * E, norm_bf, (long)N_img * E);
      gemm_bf16(norm_txt_bf, w->txt_proj, NULL, proj_out, NULL, N_txt, E, E, s);
      ec_launch_gated_res(txt_d, proj_out, mod_txt_v + 2 * E, N_txt, E, s);
      gemm_bf16(norm_bf, w->img_proj, NULL, proj_out, NULL, N_img, E, E, s);
      ec_launch_gated_res(img_d, proj_out, mod_img_v + 2 * E, N_img, E, s);

      // Image SwiGLU MLP
      k_adaln_bf16<<<N_img, 256, 0, s>>>(img_d, mod_img_v + 3 * E, mod_img_v + 4 * E, norm_bf, N_img, E, 1e-6f);
      gemm_bf16(norm_bf, w->img_mlp0, NULL, mlp_2f, NULL, N_img, 2 * F, E, s);
      k_flux2_swiglu_bf16<<<((long)N_img * F + 255) / 256, 256, 0, s>>>(mlp_2f, cat_bf, N_img, F);
      gemm_bf16(cat_bf, w->img_mlp2, NULL, proj_out, NULL, N_img, E, F, s);
      ec_launch_gated_res(img_d, proj_out, mod_img_v + 5 * E, N_img, E, s);

      // Text SwiGLU MLP
      k_adaln_bf16<<<N_txt, 256, 0, s>>>(txt_d, mod_txt_v + 3 * E, mod_txt_v + 4 * E, norm_txt_bf, N_txt, E, 1e-6f);
      gemm_bf16(norm_txt_bf, w->txt_mlp0, NULL, mlp_2f, NULL, N_txt, 2 * F, E, s);
      k_flux2_swiglu_bf16<<<((long)N_txt * F + 255) / 256, 256, 0, s>>>(mlp_2f, cat_bf, N_txt, F);
      gemm_bf16(cat_bf, w->txt_mlp2, NULL, proj_out, NULL, N_txt, E, F, s);
      ec_launch_gated_res(txt_d, proj_out, mod_txt_v + 5 * E, N_txt, E, s);
    }

    // 4. Concatenate [txt, img] -> h_cat [N_tot, E]
    cudaMemcpyAsync(h_cat, txt_d, (size_t)N_txt * E * 4, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(h_cat + (size_t)N_txt * E, img_d, (size_t)N_img * E * 4, cudaMemcpyDeviceToDevice, s);

    // 5. 20 Single-Stream Parallel Blocks
    int w_tot = N_tot * H;
    for (int b = 0; b < 20; b++) {
      const fx_single_w_t *w = &d->sng[b];
      k_adaln_bf16<<<N_tot, 256, 0, s>>>(h_cat, mod_sng_v, mod_sng_v + E, norm_bf, N_tot, E, 1e-6f);
      gemm_bf16(norm_bf, w->lin1, NULL, lin1_out, NULL, N_tot, 3 * E + 2 * F, E, s);
      k_flux2_pack_qkv_rope<<<(w_tot + 7) / 8, 256, 0, s>>>(
          lin1_out, 3 * E + 2 * F, w->qn, w->kn, rope_cos, rope_sin, Q_h, K_h, V_h, N_tot, 0, N_pad, H, D, 1e-6f);
      ec_launch_fa2_f16(Q_h, K_h, V_h, attn_out, NULL, N_tot, N_tot, N_pad, N_pad, H, D, attn_scale, 0, s);
      k_flux2_single_cat<<<((long)N_tot * (E + F) + 255) / 256, 256, 0, s>>>(attn_out, lin1_out, cat_bf, N_tot, E, F);
      gemm_bf16(cat_bf, w->lin2, NULL, proj_out, NULL, N_tot, E, E + F, s);
      ec_launch_gated_res(h_cat, proj_out, mod_sng_v + 2 * E, N_tot, E, s);
    }

    // 6. Final Layer (BFL native: shift = fmod[0..E-1], scale = fmod[E..2E-1])
    gemm_bf16(silu_temb_bf, d->final_mod, NULL, fmod_v, NULL, 1, 2 * E, E, s);
    k_adaln_bf16<<<N_img, 256, 0, s>>>(h_cat + (size_t)N_txt * E, fmod_v, fmod_v + E, norm_bf, N_img, E, 1e-6f);
    gemm_bf16(norm_bf, d->final_lin, NULL, v_pred, NULL, N_img, 128, E, s);

    // 7. Euler step: x += dt * v_pred
    ec_launch_flow_cfg_euler(out_latents_gpu_f32, v_pred, NULL, out_latents_gpu_f32, N_img * 128, dt, 1.0f, s);
  }
  cudaStreamSynchronize(s);

  cudaFree(rope_cos); cudaFree(rope_sin);
  cudaFree(temb_256); cudaFree(temb_256_bf); cudaFree(temb_0); cudaFree(temb_0_bf);
  cudaFree(temb); cudaFree(silu_temb_bf);
  cudaFree(mod_img_v); cudaFree(mod_txt_v); cudaFree(mod_sng_v); cudaFree(fmod_v);
  cudaFree(x_bf16); cudaFree(img_d); cudaFree(txt_d); cudaFree(h_cat);
  cudaFree(norm_bf); cudaFree(norm_txt_bf); cudaFree(cat_bf);
  cudaFree(qkv_img); cudaFree(qkv_txt); cudaFree(attn_out); cudaFree(proj_out);
  cudaFree(mlp_2f); cudaFree(lin1_out); cudaFree(v_pred);
  cudaFree(Q_h); cudaFree(K_h); cudaFree(V_h);

  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
}

// ===========================================================================
// Stage C: AutoencoderKLFlux2 2D VAE Decoder (Strip-Tiled BF16 TensorCore Conv2D)
// ===========================================================================
static int vae_conv2d_tc(
    const ec_safetensors_t *vst, const char *prefix,
    const float *x_d, const float *res_d, float *y_d,
    int Cin, int Cout, int H, int W, int Ksz, int pad, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  uint16_t *w_bf = load_bf16_gpu(vst, kw, NULL, s);
  float *b_f32 = load_f32_gpu(vst, kb, NULL, s);
  if (!w_bf || !b_f32) { cudaFree(w_bf); cudaFree(b_f32); return -1; }

  int K = Cin * Ksz * Ksz;
  int strip_max = 64;
  if (strip_max > H) strip_max = H;
  uint16_t *col_bf = NULL;
  float *gy_d = NULL;
  cudaMalloc(&col_bf, (size_t)strip_max * W * K * 2);
  cudaMalloc(&gy_d,   (size_t)strip_max * W * Cout * 4);

  for (int y0 = 0; y0 < H; y0 += strip_max) {
    int sh = (y0 + strip_max <= H) ? strip_max : (H - y0);
    long n_col = (long)sh * W * K;
    k_im2col_strip_bf16<<<(int)((n_col + 255) / 256), 256, 0, s>>>(x_d, col_bf, Cin, H, W, Ksz, Ksz, pad, y0, sh);
    gemm_bf16(col_bf, w_bf, b_f32, gy_d, NULL, sh * W, Cout, K, s);
    long n_out = (long)Cout * sh * W;
    k_strip_transpose_out<<<(int)((n_out + 255) / 256), 256, 0, s>>>(gy_d, res_d, y_d, Cout, H, W, y0, sh);
  }
  cudaStreamSynchronize(s);
  cudaFree(col_bf); cudaFree(gy_d); cudaFree(w_bf); cudaFree(b_f32);
  return 0;
}

static int vae_gn_silu(
    const ec_safetensors_t *vst, const char *prefix,
    const float *x_d, float *y_d, int C, int H, int W, int apply_silu, cudaStream_t s) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", prefix);
  snprintf(kb, sizeof kb, "%s.bias", prefix);
  float *g = load_f32_gpu(vst, kw, NULL, s);
  float *b = load_f32_gpu(vst, kb, NULL, s);
  if (!g || !b) { cudaFree(g); cudaFree(b); return -1; }
  ec_launch_groupnorm_silu(x_d, g, b, y_d, C, H * W, 32, 1e-6f, apply_silu, s);
  cudaStreamSynchronize(s);
  cudaFree(g); cudaFree(b);
  return 0;
}

static int vae_resnet(
    const ec_safetensors_t *vst, const char *prefix,
    float *x_d, float *out_d, int Cin, int Cout, int H, int W, cudaStream_t s) {
  float *tmp1 = NULL, *tmp2 = NULL, *sc = NULL;
  cudaMalloc(&tmp1, (size_t)Cin * H * W * 4);
  cudaMalloc(&tmp2, (size_t)Cout * H * W * 4);

  char k[192];
  snprintf(k, sizeof k, "%s.norm1", prefix);
  if (vae_gn_silu(vst, k, x_d, tmp1, Cin, H, W, 1, s) != 0) goto fail;
  snprintf(k, sizeof k, "%s.conv1", prefix);
  if (vae_conv2d_tc(vst, k, tmp1, NULL, tmp2, Cin, Cout, H, W, 3, 1, s) != 0) goto fail;

  cudaFree(tmp1); tmp1 = NULL;
  cudaMalloc(&tmp1, (size_t)Cout * H * W * 4);
  snprintf(k, sizeof k, "%s.norm2", prefix);
  if (vae_gn_silu(vst, k, tmp2, tmp1, Cout, H, W, 1, s) != 0) goto fail;

  if (Cin != Cout) {
    cudaMalloc(&sc, (size_t)Cout * H * W * 4);
    snprintf(k, sizeof k, "%s.conv_shortcut", prefix);
    if (vae_conv2d_tc(vst, k, x_d, NULL, sc, Cin, Cout, H, W, 1, 0, s) != 0) goto fail;
  }
  snprintf(k, sizeof k, "%s.conv2", prefix);
  if (vae_conv2d_tc(vst, k, tmp1, sc ? sc : x_d, out_d, Cout, Cout, H, W, 3, 1, s) != 0) goto fail;

  cudaFree(tmp1); cudaFree(tmp2); cudaFree(sc);
  return 0;
fail:
  cudaFree(tmp1); cudaFree(tmp2); cudaFree(sc);
  return -1;
}

static int vae_mid_attention(const ec_safetensors_t *vst, float *x_d, int C, int H, int W, cudaStream_t s) {
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

  if (vae_gn_silu(vst, "decoder.mid_block.attentions.0.group_norm", x_d, gn, C, H, W, 0, s) != 0) goto fail;
  k_chw_to_sc_bf16<<<((long)C * S + 255) / 256, 256, 0, s>>>(gn, sc_bf, C, S);

  {
    uint16_t *wq = load_bf16_gpu(vst, "decoder.mid_block.attentions.0.to_q.weight", NULL, s);
    float    *bq = load_f32_gpu(vst,  "decoder.mid_block.attentions.0.to_q.bias", NULL, s);
    uint16_t *wk = load_bf16_gpu(vst, "decoder.mid_block.attentions.0.to_k.weight", NULL, s);
    float    *bk = load_f32_gpu(vst,  "decoder.mid_block.attentions.0.to_k.bias", NULL, s);
    uint16_t *wv = load_bf16_gpu(vst, "decoder.mid_block.attentions.0.to_v.weight", NULL, s);
    float    *bv = load_f32_gpu(vst,  "decoder.mid_block.attentions.0.to_v.bias", NULL, s);
    uint16_t *wo = load_bf16_gpu(vst, "decoder.mid_block.attentions.0.to_out.0.weight", NULL, s);
    float    *bo = load_f32_gpu(vst,  "decoder.mid_block.attentions.0.to_out.0.bias", NULL, s);
    if (!wq || !wk || !wv || !wo) {
      cudaFree(wq); cudaFree(bq); cudaFree(wk); cudaFree(bk); cudaFree(wv); cudaFree(bv); cudaFree(wo); cudaFree(bo);
      goto fail;
    }
    gemm_bf16(sc_bf, wq, bq, q, NULL, S, C, C, s);
    gemm_bf16(sc_bf, wk, bk, k, NULL, S, C, C, s);
    gemm_bf16(sc_bf, wv, bv, v, NULL, S, C, C, s);
    k_vae_mid_attn<<<S, 256, 0, s>>>(q, k, v, ao_bf, S, C, 1.0f / sqrtf((float)C));
    gemm_bf16(ao_bf, wo, bo, out, NULL, S, C, C, s);
    k_sc_add_to_chw<<<((long)C * S + 255) / 256, 256, 0, s>>>(out, x_d, C, S);
    cudaStreamSynchronize(s);
    cudaFree(wq); cudaFree(bq); cudaFree(wk); cudaFree(bk); cudaFree(wv); cudaFree(bv); cudaFree(wo); cudaFree(bo);
  }

  cudaFree(gn); cudaFree(sc_bf); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ao_bf); cudaFree(out);
  return 0;
fail:
  cudaFree(gn); cudaFree(sc_bf); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ao_bf); cudaFree(out);
  return -1;
}

static int run_flux2_vae_decode(
    ec_flux2_t *fx, const float *packed_latents_d, int H_p, int W_p,
    uint8_t *out_rgb_hwc, double *out_ms) {
  double t0 = fx_now_ms();
  cudaStream_t s = fx->stream;
  const ec_safetensors_t *vst = &fx->vae_st;
  int H = H_p * 2, W = W_p * 2;

  float *bn_mean = load_f32_gpu(vst, "bn.running_mean", NULL, s);
  float *bn_var  = load_f32_gpu(vst, "bn.running_var", NULL, s);
  if (!bn_mean || !bn_var) { cudaFree(bn_mean); cudaFree(bn_var); return -1; }

  float *z_d = NULL, *z2_d = NULL;
  cudaMalloc(&z_d,  32L * H * W * 4);
  cudaMalloc(&z2_d, 32L * H * W * 4);
  k_flux2_bn_unpatchify<<<(int)((32L * H * W + 255) / 256), 256, 0, s>>>(
      packed_latents_d, bn_mean, bn_var, z_d, H_p, W_p, 1e-4f);
  cudaFree(bn_mean); cudaFree(bn_var);

  // post_quant_conv (32 -> 32, 1x1)
  if (vae_conv2d_tc(vst, "post_quant_conv", z_d, NULL, z2_d, 32, 32, H, W, 1, 0, s) != 0) {
    cudaFree(z_d); cudaFree(z2_d); return -1;
  }
  cudaFree(z_d);

  // conv_in (32 -> 512, 3x3)
  float *cur = NULL, *nxt = NULL;
  cudaMalloc(&cur, 512L * H * W * 4);
  if (vae_conv2d_tc(vst, "decoder.conv_in", z2_d, NULL, cur, 32, 512, H, W, 3, 1, s) != 0) {
    cudaFree(z2_d); cudaFree(cur); return -1;
  }
  cudaFree(z2_d);

  // mid_block: resnet0 -> attention -> resnet1
  cudaMalloc(&nxt, 512L * H * W * 4);
  vae_resnet(vst, "decoder.mid_block.resnets.0", cur, nxt, 512, 512, H, W, s);
  cudaFree(cur); cur = nxt; nxt = NULL;
  vae_mid_attention(vst, cur, 512, H, W, s);
  cudaMalloc(&nxt, 512L * H * W * 4);
  vae_resnet(vst, "decoder.mid_block.resnets.1", cur, nxt, 512, 512, H, W, s);
  cudaFree(cur); cur = nxt; nxt = NULL;

  // up_blocks: 0 (512->512, up), 1 (512->512, up), 2 (512->256, up), 3 (256->128, no up)
  const int up_ch[4] = { 512, 512, 256, 128 };
  int C = 512;
  char pref[128];
  for (int u = 0; u < 4; u++) {
    int Cout = up_ch[u];
    for (int r = 0; r < 3; r++) {
      int Cin = (r == 0) ? C : Cout;
      cudaMalloc(&nxt, (size_t)Cout * H * W * 4);
      snprintf(pref, sizeof pref, "decoder.up_blocks.%d.resnets.%d", u, r);
      vae_resnet(vst, pref, cur, nxt, Cin, Cout, H, W, s);
      cudaFree(cur); cur = nxt; nxt = NULL;
    }
    C = Cout;
    if (u < 3) {
      int nH = H * 2, nW = W * 2;
      float *up = NULL;
      cudaMalloc(&up,  (size_t)C * nH * nW * 4);
      cudaMalloc(&nxt, (size_t)C * nH * nW * 4);
      ec_launch_upsample2x_2d(cur, up, C, H, W, s);
      cudaFree(cur);
      snprintf(pref, sizeof pref, "decoder.up_blocks.%d.upsamplers.0.conv", u);
      vae_conv2d_tc(vst, pref, up, NULL, nxt, C, C, nH, nW, 3, 1, s);
      cudaFree(up);
      cur = nxt; nxt = NULL;
      H = nH; W = nW;
    }
  }

  // conv_norm_out (128, GN32 + SiLU) -> conv_out (128 -> 3, 3x3)
  float *gn_out = NULL, *rgb_d = NULL;
  cudaMalloc(&gn_out, (size_t)128 * H * W * 4);
  cudaMalloc(&rgb_d,  (size_t)3 * H * W * 4);
  vae_gn_silu(vst, "decoder.conv_norm_out", cur, gn_out, 128, H, W, 1, s);
  cudaFree(cur);
  vae_conv2d_tc(vst, "decoder.conv_out", gn_out, NULL, rgb_d, 128, 3, H, W, 3, 1, s);
  cudaFree(gn_out);

  float *rgb_h = (float *)malloc((size_t)3 * H * W * 4);
  cudaMemcpy(rgb_h, rgb_d, (size_t)3 * H * W * 4, cudaMemcpyDeviceToHost);
  cudaFree(rgb_d);

  size_t plane = (size_t)H * W;
  for (size_t i = 0; i < plane; i++) {
    for (int c = 0; c < 3; c++) {
      float v = (rgb_h[c * plane + i] * 0.5f + 0.5f);
      if (v < 0.0f) v = 0.0f;
      if (v > 1.0f) v = 1.0f;
      out_rgb_hwc[i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
    }
  }
  free(rgb_h);
  if (out_ms) *out_ms = fx_now_ms() - t0;
  return 0;
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
  return fx;
}

extern "C" void ec_flux2_unload_vram(ec_flux2_t *fx) {
  if (!fx) return;
  free_dit_resident(&fx->dit);
  free_qwen3_resident(&fx->qwen_res);
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

  // 1. Text Encode (Qwen3-4B layers 0..26) BEFORE loading DiT if DiT isn't loaded yet -> minimal peak VRAM!
  if (cb) cb(0, steps + 2, "qwen3_text_encode", cb_ctx);
  uint16_t *prompt_emb_d = NULL;
  cudaMalloc(&prompt_emb_d, 512L * 7680L * 2);
  if (run_qwen3_text_encoder(fx, req->prompt ? req->prompt : "", prompt_emb_d, &res->text_enc_ms) != 0) {
    cudaFree(prompt_emb_d);
    snprintf(res->error, sizeof res->error, "qwen3 text encoder failed");
    return -1;
  }

  // 2. Ensure FLUX.2 DiT checkpoint + LoRA stack is resident on GPU
  if (cb) cb(1, steps + 2, "load_dit_and_loras", cb_ctx);
  if (ensure_dit_loaded(fx, ckpt, req->loras, req->n_loras, res->error, sizeof res->error) != 0) {
    cudaFree(prompt_emb_d);
    return -1;
  }

  // 3. Denoise latents [H/16 * W/16, 128]
  int H_p = H / 16, W_p = W / 16;
  float *latents_d = NULL;
  cudaMalloc(&latents_d, (size_t)H_p * W_p * 128 * 4);
  if (run_flux2_dit(fx, prompt_emb_d, H_p, W_p, steps, req->seed, latents_d, cb, cb_ctx, &res->dit_ms) != 0) {
    cudaFree(prompt_emb_d); cudaFree(latents_d);
    snprintf(res->error, sizeof res->error, "flux2 dit denoising failed");
    return -1;
  }
  cudaFree(prompt_emb_d);

  // 4. VAE Decode -> RGB24 (+ optional RFC 2083 PNG)
  if (cb) cb(steps + 1, steps + 2, "vae_decode_png", cb_ctx);
  uint8_t *rgb = (uint8_t *)malloc((size_t)W * H * 3);
  if (run_flux2_vae_decode(fx, latents_d, H_p, W_p, rgb, &res->vae_ms) != 0) {
    free(rgb); cudaFree(latents_d);
    snprintf(res->error, sizeof res->error, "flux2 vae decode failed");
    return -1;
  }
  cudaFree(latents_d);

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
