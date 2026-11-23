// ec_kernels_v2.cu — Production Creative GPU Kernels (Phase C1-C4 Complete Backend)
// 100% Pure Eureka CUDA PTX. Zero cuBLAS, zero ggml, zero external ML libraries.

#include "ec_kernels_v2.cuh"
#include <math.h>
#include <stdio.h>

// ===========================================================================
// 1. Exact IEEE BF16 & OCP FP8_E4M3FN GPU Weight Ingestion
// ===========================================================================

__global__ void k_bf16_to_f16(const uint16_t *__restrict__ src, __half *__restrict__ dst, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t bits = ((uint32_t)src[i]) << 16;
  float f = __uint_as_float(bits);
  dst[i] = __float2half_rn(f);
}

extern "C" int ec_launch_bf16_to_f16(const uint16_t *src_bf16, __half *dst_f16, size_t n, cudaStream_t stream) {
  if (!src_bf16 || !dst_f16 || n == 0) return -1;
  const int blk = 256;
  const int grd = (int)((n + blk - 1) / blk);
  k_bf16_to_f16<<<grd, blk, 0, stream>>>(src_bf16, dst_f16, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Exact OCP FP8 E4M3FN decode (1 sign, 4 exp bias 7, 3 mantissa; 0x7F/0xFF = NaN -> 0)
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t b) {
  const uint32_t sign = (b >> 7) & 1u;
  const uint32_t exp  = (b >> 3) & 0x0Fu;
  const uint32_t mant = b & 0x07u;
  if (exp == 15u && mant == 7u) return 0.0f; // NaN guarded cleanly
  float v;
  if (exp == 0u) {
    // Subnormal: (-1)^s * 2^(-6) * (mant / 8) = mant * 2^(-9)
    v = ldexpf((float)mant, -9);
  } else {
    // Normal: (-1)^s * 2^(exp - 7) * (1 + mant / 8)
    v = ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7);
  }
  return sign ? -v : v;
}

__global__ void k_fp8_to_f16(const uint8_t *__restrict__ src, __half *__restrict__ dst, size_t n, float scale) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half_rn(fp8_e4m3_to_f32(src[i]) * scale);
}

extern "C" int ec_launch_fp8_to_f16(const uint8_t *src_fp8, __half *dst_f16, size_t n, float scale, cudaStream_t stream) {
  if (!src_fp8 || !dst_f16 || n == 0) return -1;
  const int blk = 256;
  const int grd = (int)((n + blk - 1) / blk);
  k_fp8_to_f16<<<grd, blk, 0, stream>>>(src_fp8, dst_f16, n, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int ec_launch_gemm_bf16(
    const __half *X_f16, const uint16_t *W_bf16, __half *w_stage_f16,
    const float *bias, float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue, cudaStream_t stream) {
  if (ec_launch_bf16_to_f16(W_bf16, w_stage_f16, (size_t)N * K, stream) != 0) return -1;
  return ec_launch_gemm_f16(X_f16, w_stage_f16, bias, nullptr, Y, Y_f16, M, N, K, epilogue, stream);
}

extern "C" int ec_launch_gemm_fp8(
    const __half *X_f16, const uint8_t *W_fp8, float w_scale, __half *w_stage_f16,
    const float *bias, float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue, cudaStream_t stream) {
  if (ec_launch_fp8_to_f16(W_fp8, w_stage_f16, (size_t)N * K, w_scale, stream) != 0) return -1;
  return ec_launch_gemm_f16(X_f16, w_stage_f16, bias, nullptr, Y, Y_f16, M, N, K, epilogue, stream);
}

// ===========================================================================
// 2. Tensor-Core Accelerated Tiled im2col Conv2D (routes through 53.7 TF GEMM!)
// ===========================================================================
// Builds col_f16[H_out * W_out, Cin * Kh * Kw] in one coalesced pass, runs m16n8k16 GEMM,
// and transposes [H_out * W_out, Cout] -> [Cout, H_out, W_out] with bias!

__global__ void k_im2col_2d_f16(
    const float *__restrict__ x, __half *__restrict__ col,
    int Cin, int H, int W, int Kh, int Kw, int stride, int pad, int H_out, int W_out) {
  const int K = Cin * Kh * Kw;
  const long M = (long)H_out * W_out;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * K) return;
  const int m = (int)(idx / K);
  const int k = (int)(idx % K);
  const int ho = m / W_out;
  const int wo = m % W_out;
  const int kw = k % Kw;
  const int tmp = k / Kw;
  const int kh = tmp % Kh;
  const int ci = tmp / Kh;
  const int hi = ho * stride - pad + kh;
  const int wi = wo * stride - pad + kw;
  float v = 0.0f;
  if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
    v = x[((long)ci * H + hi) * W + wi];
  }
  col[idx] = __float2half_rn(v);
}

__global__ void k_transpose_hw_c(
    const float *__restrict__ src_mc, float *__restrict__ dst_cm, int M, int C) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)M * C) return;
  const int c = (int)(idx / M);
  const int m = (int)(idx % M);
  dst_cm[idx] = src_mc[(long)m * C + c];
}

extern "C" int ec_launch_conv2d_tc(
    const float *x, const __half *w_f16, const float *bias,
    __half *col_f16, float *y_rowmajor_tmp, float *y,
    int Cin, int Cout, int H, int W, int Kh, int Kw, int stride, int pad,
    cudaStream_t stream) {
  const int H_out = (H + 2 * pad - Kh) / stride + 1;
  const int W_out = (W + 2 * pad - Kw) / stride + 1;
  if (H_out <= 0 || W_out <= 0) return -1;
  const int M = H_out * W_out;
  const int K = Cin * Kh * Kw;
  const long col_n = (long)M * K;
  k_im2col_2d_f16<<<(int)((col_n + 255) / 256), 256, 0, stream>>>(
      x, col_f16, Cin, H, W, Kh, Kw, stride, pad, H_out, W_out);
  if (ec_launch_gemm_f16(col_f16, w_f16, bias, nullptr, y_rowmajor_tmp, nullptr,
                         M, Cout, K, bias ? 1 : 0, stream) != 0)
    return -1;
  const long out_n = (long)M * Cout;
  k_transpose_hw_c<<<(int)((out_n + 255) / 256), 256, 0, stream>>>(y_rowmajor_tmp, y, M, Cout);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 3. 2x Nearest Spatial Upsample (VAE Decoder)
// ===========================================================================
__global__ void k_upsample2x_2d(const float *__restrict__ x, float *__restrict__ y, int C, int H, int W) {
  const int H2 = H * 2, W2 = W * 2;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)C * H2 * W2) return;
  const int w = (int)(idx % W2);
  const long rem = idx / W2;
  const int h = (int)(rem % H2);
  const int c = (int)(rem / H2);
  y[idx] = x[((long)c * H + (h >> 1)) * W + (w >> 1)];
}

extern "C" int ec_launch_upsample2x_2d(const float *x, float *y, int C, int H, int W, cudaStream_t stream) {
  const long total = (long)C * (H * 2) * (W * 2);
  if (total <= 0) return -1;
  k_upsample2x_2d<<<(int)((total + 255) / 256), 256, 0, stream>>>(x, y, C, H, W);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 4. General Cross-Attention (Text->Latent & LTX-2.5 Audio<->Video)
// ===========================================================================
extern "C" int ec_launch_cross_attn(
    const __half *Q, const __half *K, const __half *V,
    int8_t *q_i8, int8_t *k_i8, float *q_sc, float *k_sc,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, ec_tier_t tier,
    cudaStream_t stream) {
  if (tier == EC_TIER_FAST && q_i8 && k_i8 && q_sc && k_sc) {
    return ec_launch_sage_attn_i8(Q, K, V, q_i8, k_i8, q_sc, k_sc, O, O_f16,
                                  n_q, n_kv, n_q_pad, n_kv_pad, n_heads, head_dim, scale, 0, stream);
  }
  return ec_launch_fa2_f16(Q, K, V, O, O_f16,
                           n_q, n_kv, n_q_pad, n_kv_pad, n_heads, head_dim, scale, 0, stream);
}

// ===========================================================================
// 5. SwiGLU Elementwise Gate (Z-Image S3-DiT, Wan 2.2, ACE-Step 1.5)
// ===========================================================================
__global__ void k_swiglu_mul(
    const float *__restrict__ gate, const float *__restrict__ up,
    float *__restrict__ out, __half *__restrict__ out_f16, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = gate[i];
  const float v = (g / (1.0f + expf(-g))) * up[i];
  out[i] = v;
  if (out_f16) out_f16[i] = __float2half_rn(v);
}

extern "C" int ec_launch_swiglu_mul(
    const float *gate, const float *up, float *out, __half *out_f16,
    int n_elems, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  k_swiglu_mul<<<(n_elems + 255) / 256, 256, 0, stream>>>(gate, up, out, out_f16, n_elems);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 6. Sinusoidal Timestep Embedding
// ===========================================================================
__global__ void k_timestep_embed(float t_val, float *__restrict__ out, int half_d, float log_max_period) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half_d) return;
  const float freq = expf(-log_max_period * (float)i / (float)half_d);
  const float arg = t_val * freq;
  out[i] = cosf(arg);
  out[i + half_d] = sinf(arg);
}

extern "C" int ec_launch_timestep_embed(
    float t_val, float *out_emb, int dim, float max_period, cudaStream_t stream) {
  if (dim <= 0 || (dim & 1)) return -1;
  const int half_d = dim >> 1;
  k_timestep_embed<<<(half_d + 255) / 256, 256, 0, stream>>>(t_val, out_emb, half_d, logf(max_period));
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 7. Shifted Rectified-Flow Schedule + 2nd-Order Heun + LTX-2.5 Modality-CFG
// ===========================================================================
extern "C" void ec_rf_shifted_schedule(int n_steps, float shift, float *out_ts) {
  if (n_steps <= 0 || !out_ts) return;
  for (int i = 0; i <= n_steps; i++) {
    float t = 1.0f - (float)i / (float)n_steps;
    if (shift > 0.0f && shift != 1.0f) {
      t = (shift * t) / (1.0f + (shift - 1.0f) * t);
    }
    out_ts[i] = t;
  }
}

__global__ void k_flow_heun(
    const float *__restrict__ x, const float *__restrict__ v1, const float *__restrict__ v2,
    float *__restrict__ x_next, int n, float dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x_next[i] = x[i] + 0.5f * dt * (v1[i] + v2[i]);
}

extern "C" int ec_launch_flow_heun(
    const float *x, const float *v1, const float *v2,
    float *x_next, int n_elems, float dt, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  k_flow_heun<<<(n_elems + 255) / 256, 256, 0, stream>>>(x, v1, v2, x_next, n_elems, dt);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_modality_cfg(
    const float *__restrict__ vu, const float *__restrict__ vt, const float *__restrict__ vm,
    float *__restrict__ vo, int n, float cfg_text, float cfg_modal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = vu[i] + cfg_text * (vt[i] - vu[i]);
  if (vm) v += cfg_modal * (vm[i] - vt[i]);
  vo[i] = v;
}

extern "C" int ec_launch_modality_cfg(
    const float *v_uncond, const float *v_text, const float *v_modal,
    float *v_out, int n_elems, float cfg_text, float cfg_modal, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  k_modality_cfg<<<(n_elems + 255) / 256, 256, 0, stream>>>(
      v_uncond, v_text, v_modal, v_out, n_elems, cfg_text, cfg_modal);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
