// ec_ops_common.h — Shared Coherent/Unified-Memory Auxiliary Operators for Metal & Vulkan Backends
// Heavy matrix/attention/conv ops (GEMM_BF16, GEMM_F16, GEMM_Q8_G64, Quantize_Q8_G64, FA2_BF16,
// FA2_F16, Conv2D_3x3_NHWC_BF16, Conv2D_TC, CrossAttn) execute on the GPU in ec_backend_{metal,vulkan};
// this header provides exact parallel implementations for auxiliary 1D/3D/sparse/Euler ops,
// marking outputs via ec_hal_mark_host_dirty so discrete Vulkan DEVICE_LOCAL mirrors stay coherent.

#pragma once

#include "../ec_hal.h"
#include "../ec_kernels.cuh"
#include "../ec_kernels_v2.cuh"
#include "../ec_ops.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float ec_cmn_bf16_to_f32(uint16_t b) {
  return __uint_as_float(((uint32_t)b) << 16);
}
static inline uint16_t ec_cmn_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline float ec_cmn_silu(float x) {
  return x / (1.0f + expf(-x));
}
static inline float ec_cmn_fp8_e4m3_to_f32(uint8_t b) {
  const uint32_t sign = (b >> 7) & 1u;
  const uint32_t exp  = (b >> 3) & 0x0Fu;
  const uint32_t mant = b & 0x07u;
  if (exp == 15u && mant == 7u) return 0.0f;
  float v = (exp == 0u) ? ldexpf((float)mant, -9) : ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7);
  return sign ? -v : v;
}

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EC_BACKEND_VULKAN
int ec_ops_dequant_q8g64_bf16(
    const int8_t *src_q8, const float *src_sc, uint16_t *dst_bf16,
    int rows, int cols, cudaStream_t stream) {
  if (!src_q8 || !src_sc || !dst_bf16 || rows <= 0 || cols <= 0 || (cols & 63)) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int groups = cols >> 6;
  const long total_g = (long)rows * groups;
  EC_PAR_FOR_BEGIN(g, total_g)
    const float sc = src_sc[g];
    const int8_t *sq = src_q8 + g * 64;
    uint16_t *db = dst_bf16 + g * 64;
    for (int i = 0; i < 64; i++) db[i] = ec_cmn_f32_to_bf16((float)sq[i] * sc);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst_bf16);
  return 0;
}

int ec_ops_silu_f32_to_bf16(const float *src, uint16_t *dst, long n, cudaStream_t stream) {
  if (n <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n) dst[i] = ec_cmn_f32_to_bf16(ec_cmn_silu(src[i])); EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst);
  return 0;
}
#endif

int ec_ops_wan_cfg_euler(
    float *xt, const float *vp, const float *vn, float *delta_cache,
    long n, float dt, float cfg_scale, int mode, cudaStream_t stream) {
  if (n <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n)
    float v;
    if (mode == 0) {
      v = vp[i];
    } else if (mode == 1) {
      float d = vp[i] - vn[i];
      if (delta_cache) delta_cache[i] = d;
      v = vn[i] + cfg_scale * d;
    } else {
      v = vp[i] + (cfg_scale - 1.0f) * delta_cache[i];
    }
    xt[i] -= dt * v;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(xt);
  if (delta_cache) ec_hal_mark_host_dirty(delta_cache);
  return 0;
}

int ec_ops_trellis_cfg_euler(
    float *xt, const float *vp, const float *vn,
    int n, float dt, float t_cur, float cfg_scale,
    float guidance_rescale, float sigma_min, int use_cfg,
    cudaStream_t stream) {
  if (n <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  if (!use_cfg || !vn) {
    EC_PAR_FOR_BEGIN(i, n) xt[i] -= dt * vp[i]; EC_PAR_FOR_END
    ec_hal_mark_host_dirty(xt);
    return 0;
  }
  const float a = 1.0f - sigma_min;
  const float b = sigma_min + a * t_cur;
  double mp = 0.0, mc = 0.0;
  for (int i = 0; i < n; i++) {
    float x = xt[i], p = vp[i], u = vn[i];
    float vc = cfg_scale * p + (1.0f - cfg_scale) * u;
    mp += (double)(a * x - b * p);
    mc += (double)(a * x - b * vc);
  }
  double mean_p = mp / (double)n, mean_c = mc / (double)n;
  double vp_acc = 0.0, vc_acc = 0.0;
  for (int i = 0; i < n; i++) {
    float x = xt[i], p = vp[i], u = vn[i];
    float vc = cfg_scale * p + (1.0f - cfg_scale) * u;
    double dp = (double)(a * x - b * p) - mean_p;
    double dc = (double)(a * x - b * vc) - mean_c;
    vp_acc += dp * dp;
    vc_acc += dc * dc;
  }
  float ratio = (float)(sqrt(vp_acc / (double)n + 1e-12) / sqrt(vc_acc / (double)n + 1e-12));
  float inv_b = 1.0f / b;
  EC_PAR_FOR_BEGIN(i, n)
    float x = xt[i], p = vp[i], u = vn[i];
    float vc = cfg_scale * p + (1.0f - cfg_scale) * u;
    float x0_c = a * x - b * vc;
    float x0_fin = guidance_rescale * (x0_c * ratio) + (1.0f - guidance_rescale) * x0_c;
    float v_fin = (a * x - x0_fin) * inv_b;
    xt[i] = x - dt * v_fin;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(xt);
  return 0;
}

#ifndef EC_BACKEND_VULKAN
int ec_ops_rmsnorm_adaln(
    const float *x, const float *w, const float *scale, const float *shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, cudaStream_t stream) {
  if (!x || rows <= 0 || dim <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float ss = 0.0f;
    for (int c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float inv = rsqrtf(ss / (float)dim + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = xr[c] * inv * (w ? w[c] : 1.0f);
      if (scale) v = v * (1.0f + scale[c]);
      if (shift) v += shift[c];
      if (yf) yf[c] = v;
      if (yb) yb[c] = ec_cmn_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
  return 0;
}

int ec_ops_layernorm_adaln(
    const float *x, const float *w, const float *b, const float *scale, const float *shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, cudaStream_t stream) {
  if (!x || rows <= 0 || dim <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = 0; c < dim; c++) { float v = xr[c]; s1 += v; s2 += v * v; }
    float mean = s1 / (float)dim;
    float var  = s2 / (float)dim - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv = rsqrtf(var + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = (xr[c] - mean) * inv;
      if (w) v *= w[c];
      if (b) v += b[c];
      if (scale) v = v * (1.0f + scale[c]);
      if (shift) v += shift[c];
      if (yf) yf[c] = v;
      if (yb) yb[c] = ec_cmn_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
  return 0;
}

int ec_launch_adaln_zero(
    const float *x, const float *norm_w, const float *shift, const float *scale,
    float *y, __half *y_f16, int n_rows, int dim, float eps, int norm_mode,
    cudaStream_t stream) {
  if (n_rows <= 0 || dim <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(r, n_rows)
    const float *xr = x + (long)r * dim;
    float mean = 0.0f;
    if (norm_mode == 0) {
      float s = 0.0f;
      for (int i = 0; i < dim; i++) s += xr[i];
      mean = s / (float)dim;
    }
    float v = 0.0f;
    for (int i = 0; i < dim; i++) { float d = xr[i] - mean; v += d * d; }
    float inv = rsqrtf(v / (float)dim + eps);
    float *yr = y + (long)r * dim;
    __half *yh = y_f16 ? (y_f16 + (long)r * dim) : NULL;
    for (int i = 0; i < dim; i++) {
      float nrm = (xr[i] - mean) * inv;
      if (norm_mode == 2 && norm_w) nrm *= norm_w[i];
      float sc = scale ? (1.0f + scale[i]) : 1.0f;
      float sh = shift ? shift[i] : 0.0f;
      float out = nrm * sc + sh;
      yr[i] = out;
      if (yh) yh[i] = __float2half_rn(out);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  ec_hal_mark_host_dirty(y_f16);
  return 0;
}

int ec_launch_gated_res(float *x, const float *h, const float *gate, int n_rows, int dim, cudaStream_t stream) {
  long total = (long)n_rows * dim;
  if (total <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, total) x[i] += gate[i % dim] * h[i]; EC_PAR_FOR_END
  ec_hal_mark_host_dirty(x);
  return 0;
}
#endif

int ec_launch_rope_nd(
    const float *q_in, const float *k_in, const float *v_in,
    const int32_t *coords, const float *qk_norm_w_q, const float *qk_norm_w_k,
    __half *q_out, __half *k_out, __half *v_out,
    int n_tokens, int n_pad, int n_heads, int head_dim,
    int ndims, const int32_t axis_dims[3], float theta_base, float eps,
    cudaStream_t stream) {
  if (n_tokens <= 0 || n_heads <= 0 || (head_dim & 1) || ndims < 1 || ndims > 3) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const long pairs = (long)n_tokens * n_heads;
  const int E = n_heads * head_dim;
  const int half_d = head_dim >> 1;
  const int ax0 = axis_dims[0], ax1 = axis_dims[1], ax2 = axis_dims[2];
  const int p0 = ax0 >> 1, p1 = ax1 >> 1, p2 = ax2 >> 1;
  EC_PAR_FOR_BEGIN(pair_idx, pairs)
    int h = (int)(pair_idx % n_heads);
    int tok = (int)(pair_idx / n_heads);
    const float *qs = q_in + (long)tok * E + h * head_dim;
    const float *ks = k_in + (long)tok * E + h * head_dim;
    const float *vs = v_in ? (v_in + (long)tok * E + h * head_dim) : NULL;
    float inv_q = 1.0f, inv_k = 1.0f;
    if (qk_norm_w_q || qk_norm_w_k) {
      float sq = 0.0f, sk = 0.0f;
      for (int d = 0; d < head_dim; d++) { sq += qs[d] * qs[d]; sk += ks[d] * ks[d]; }
      if (qk_norm_w_q) inv_q = rsqrtf(sq / (float)head_dim + eps);
      if (qk_norm_w_k) inv_k = rsqrtf(sk / (float)head_dim + eps);
    }
    int32_t c0 = coords[(long)tok * ndims + 0];
    int32_t c1 = (ndims >= 2) ? coords[(long)tok * ndims + 1] : 0;
    int32_t c2 = (ndims >= 3) ? coords[(long)tok * ndims + 2] : 0;
    long out_base = ((long)h * n_pad + tok) * head_dim;
    __half *qo = q_out + out_base;
    __half *ko = k_out + out_base;
    __half *vo = v_out ? (v_out + out_base) : NULL;
    for (int p = 0; p < half_d; p++) {
      int d0 = 2 * p, d1 = d0 + 1;
      float q0 = qs[d0] * inv_q * (qk_norm_w_q ? qk_norm_w_q[d0] : 1.0f);
      float q1 = qs[d1] * inv_q * (qk_norm_w_q ? qk_norm_w_q[d1] : 1.0f);
      float k0 = ks[d0] * inv_k * (qk_norm_w_k ? qk_norm_w_k[d0] : 1.0f);
      float k1 = ks[d1] * inv_k * (qk_norm_w_k ? qk_norm_w_k[d1] : 1.0f);
      float angle = 0.0f;
      if (p < p0 && p0 > 0) angle = (float)c0 * powf(theta_base, -2.0f * (float)p / (float)ax0);
      else if (ndims >= 2 && p < p0 + p1 && p1 > 0) angle = (float)c1 * powf(theta_base, -2.0f * (float)(p - p0) / (float)ax1);
      else if (ndims >= 3 && p < p0 + p1 + p2 && p2 > 0) angle = (float)c2 * powf(theta_base, -2.0f * (float)(p - p0 - p1) / (float)ax2);
      float cs = cosf(angle), sn = sinf(angle);
      qo[d0] = __float2half_rn(q0 * cs - q1 * sn);
      qo[d1] = __float2half_rn(q0 * sn + q1 * cs);
      ko[d0] = __float2half_rn(k0 * cs - k1 * sn);
      ko[d1] = __float2half_rn(k0 * sn + k1 * cs);
      if (vo && vs) { vo[d0] = __float2half_rn(vs[d0]); vo[d1] = __float2half_rn(vs[d1]); }
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(q_out);
  ec_hal_mark_host_dirty(k_out);
  ec_hal_mark_host_dirty(v_out);
  return 0;
}

int ec_launch_lora_add(
    const __half *X, const ec_lora_t *lora,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (!lora || lora->rank <= 0) return 0;
  const int R = lora->rank;
  if (ec_launch_gemm_f16(X, (const __half *)lora->A_d, NULL, NULL, tmp_f32, tmp_f16, M, R, K, 0, stream) != 0)
    return -1;
  ec_hal_sync_all_for_cpu(stream);
  const __half *B = (const __half *)lora->B_d;
  const long total = (long)M * N;
  EC_PAR_FOR_BEGIN(idx, total)
    int r = (int)(idx / N), c = (int)(idx % N);
    const __half *hr = tmp_f16 + (long)r * R;
    const __half *br = B + (long)c * R;
    float s = 0.0f;
    for (int k = 0; k < R; k++) s += __half2float(hr[k]) * __half2float(br[k]);
    Y[idx] += lora->scale * s;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(Y);
  return 0;
}

#if !defined(EC_BACKEND_VULKAN) && !defined(EC_BACKEND_METAL)
int ec_launch_gemm_i8(
    const int8_t *X_i8, const float *x_sc,
    const int8_t *W_i8, const float *w_sc,
    const float *bias, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      const int8_t *xr = X_i8 + (long)m * K;
      const int8_t *wr = W_i8 + (long)n * K;
      int32_t acc = 0;
      for (int k = 0; k < K; k++) acc += (int32_t)xr[k] * (int32_t)wr[k];
      float v = (float)acc * (x_sc[m] * w_sc[n]);
      if (bias) v += bias[n];
      Y[(long)m * N + n] = v;
    }
  }
  ec_hal_mark_host_dirty(Y);
  return 0;
}
#endif

int ec_launch_gemm_svd4(
    const __half *X_f16, const uint8_t *W_i4, const float *w_sc,
    const ec_lora_t *outlier_branch,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || (K & 31)) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int ng = K / 32;
  #pragma omp parallel for collapse(2) schedule(static)
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      const __half *xr = X_f16 + (long)r * K;
      const uint8_t *wr = W_i4 + (long)c * (K / 2);
      const float *ws = w_sc + (long)c * ng;
      float acc = 0.0f;
      for (int g = 0; g < ng; g++) {
        float xv[32], amax = 0.0f;
        for (int i = 0; i < 32; i++) {
          xv[i] = __half2float(xr[g * 32 + i]);
          if (fabsf(xv[i]) > amax) amax = fabsf(xv[i]);
        }
        float xs = (amax > 1e-8f) ? (amax / 7.0f) : 0.0f;
        float inv_x = (amax > 1e-8f) ? (7.0f / amax) : 0.0f;
        int32_t idot = 0;
        const uint8_t *wg = wr + g * 16;
        for (int b = 0; b < 16; b++) {
          uint8_t byte = wg[b];
          int8_t w0 = (int8_t)((byte & 0x0Fu) - 8);
          int8_t w1 = (int8_t)((byte >> 4) - 8);
          int8_t q0 = (int8_t)lrintf(xv[2 * b + 0] * inv_x);
          int8_t q1 = (int8_t)lrintf(xv[2 * b + 1] * inv_x);
          idot += (int32_t)w0 * (int32_t)q0 + (int32_t)w1 * (int32_t)q1;
        }
        acc += (float)idot * (xs * ws[g]);
      }
      Y[(long)r * N + c] = acc;
    }
  }
  ec_hal_mark_host_dirty(Y);
  if (outlier_branch && outlier_branch->rank > 0) {
    return ec_launch_lora_add(X_f16, outlier_branch, tmp_f32, tmp_f16, Y, M, N, K, stream);
  }
  return 0;
}

#ifndef EC_BACKEND_VULKAN
int ec_launch_groupnorm_silu(
    const float *x, const float *gamma, const float *beta,
    float *y, int C, int spatial, int num_groups, float eps, int apply_silu,
    cudaStream_t stream) {
  if (C <= 0 || spatial <= 0 || num_groups <= 0 || (C % num_groups)) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int cpg = C / num_groups;
  const long n = (long)cpg * spatial;
  EC_PAR_FOR_BEGIN(g, num_groups)
    int c0 = g * cpg;
    const float *xg = x + (long)c0 * spatial;
    float *yg = y + (long)c0 * spatial;
    double s = 0.0;
    for (long i = 0; i < n; i++) s += xg[i];
    float mean = (float)(s / (double)n);
    double v = 0.0;
    for (long i = 0; i < n; i++) { double d = xg[i] - mean; v += d * d; }
    float inv = rsqrtf((float)(v / (double)n) + eps);
    for (long i = 0; i < n; i++) {
      int c = c0 + (int)(i / spatial);
      float val = (xg[i] - mean) * inv;
      if (gamma) val *= gamma[c];
      if (beta)  val += beta[c];
      if (apply_silu) val = ec_cmn_silu(val);
      yg[i] = val;
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}
#endif

int ec_launch_conv1d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int L, int K, int stride, int pad, int dilation, int causal,
    cudaStream_t stream) {
  const int pad_left = causal ? (K - 1) * dilation : pad;
  const int pad_total = causal ? (K - 1) * dilation : 2 * pad;
  const int L_out = (L + pad_total - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const long total = (long)Cout * L_out;
  EC_PAR_FOR_BEGIN(idx, total)
    int co = (int)(idx / L_out), lo = (int)(idx % L_out);
    float acc = bias ? bias[co] : 0.0f;
    int base_l = lo * stride - pad_left;
    for (int ci = 0; ci < Cin; ci++) {
      const float *xc = x + (long)ci * L;
      const float *wc = w + ((long)co * Cin + ci) * K;
      for (int k = 0; k < K; k++) {
        int li = base_l + k * dilation;
        if (li >= 0 && li < L) acc += xc[li] * wc[k];
      }
    }
    y[idx] = acc;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

int ec_launch_conv2d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int H, int W, int Kh, int Kw, int stride, int pad,
    cudaStream_t stream) {
  const int H_out = (H + 2 * pad - Kh) / stride + 1;
  const int W_out = (W + 2 * pad - Kw) / stride + 1;
  if (H_out <= 0 || W_out <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const long hw_out = (long)H_out * W_out;
  const long total = (long)Cout * hw_out;
  EC_PAR_FOR_BEGIN(idx, total)
    int co = (int)(idx / hw_out);
    long rem = idx % hw_out;
    int ho = (int)(rem / W_out), wo = (int)(rem % W_out);
    float acc = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < Cin; ci++) {
      const float *xc = x + (long)ci * H * W;
      const float *wc = w + ((long)co * Cin + ci) * Kh * Kw;
      for (int kh = 0; kh < Kh; kh++) {
        int hi = ho * stride - pad + kh;
        if (hi < 0 || hi >= H) continue;
        for (int kw = 0; kw < Kw; kw++) {
          int wi = wo * stride - pad + kw;
          if (wi >= 0 && wi < W) acc += xc[(long)hi * W + wi] * wc[kh * Kw + kw];
        }
      }
    }
    y[idx] = acc;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

int ec_launch_conv3d_causal_f32(
    const float *x, const float *prev_cache, const float *w, const float *bias, float *y,
    int Cin, int Cout, int T, int H, int W, int Kt, int Kh, int Kw, int pad_hw,
    cudaStream_t stream) {
  const int H_out = H + 2 * pad_hw - Kh + 1;
  const int W_out = W + 2 * pad_hw - Kw + 1;
  if (T <= 0 || H_out <= 0 || W_out <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const long thw_out = (long)T * H_out * W_out;
  const long total = (long)Cout * thw_out;
  const int past = Kt - 1;
  EC_PAR_FOR_BEGIN(idx, total)
    int co = (int)(idx / thw_out);
    long rem = idx % thw_out;
    int to = (int)(rem / ((long)H_out * W_out));
    int rem2 = (int)(rem % ((long)H_out * W_out));
    int ho = rem2 / W_out, wo = rem2 % W_out;
    float acc = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < Cin; ci++) {
      const float *xc = x + (long)ci * T * H * W;
      const float *pc = prev_cache ? (prev_cache + (long)ci * past * H * W) : NULL;
      const float *wc = w + ((long)co * Cin + ci) * Kt * Kh * Kw;
      for (int kt = 0; kt < Kt; kt++) {
        int ti = to - past + kt;
        const float *plane = (ti >= 0) ? (xc + (long)ti * H * W) : (pc ? (pc + (long)(ti + past) * H * W) : NULL);
        if (!plane) continue;
        for (int kh = 0; kh < Kh; kh++) {
          int hi = ho - pad_hw + kh;
          if (hi < 0 || hi >= H) continue;
          for (int kw = 0; kw < Kw; kw++) {
            int wi = wo - pad_hw + kw;
            if (wi >= 0 && wi < W) acc += plane[(long)hi * W + wi] * wc[(kt * Kh + kh) * Kw + kw];
          }
        }
      }
    }
    y[idx] = acc;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

static inline uint64_t ec_cmn_ov_pack(int32_t x, int32_t y, int32_t z) {
  uint64_t ux = (uint64_t)(x + 32768) & 0x1FFFFFULL;
  uint64_t uy = (uint64_t)(y + 32768) & 0x1FFFFFULL;
  uint64_t uz = (uint64_t)(z + 32768) & 0x1FFFFFULL;
  return ((ux << 42) | (uy << 21) | uz) + 1ULL;
}
static inline uint32_t ec_cmn_ov_hash(uint64_t k, uint32_t mask) {
  k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
  k ^= k >> 27; k *= 0x94d049bb133111ebULL;
  k ^= k >> 31;
  return (uint32_t)k & mask;
}
static inline int32_t ec_cmn_ov_lookup(const uint64_t *keys, const int32_t *vals, uint32_t mask, int32_t x, int32_t y, int32_t z) {
  uint64_t key = ec_cmn_ov_pack(x, y, z);
  uint32_t slot = ec_cmn_ov_hash(key, mask);
  while (1) {
    uint64_t cur = keys[slot];
    if (cur == key) return vals[slot];
    if (cur == 0ULL) return -1;
    slot = (slot + 1u) & mask;
  }
}

int ec_launch_ovoxel_hash_build(ec_ovoxel_t *ov, cudaStream_t stream) {
  if (!ov || ov->n_active <= 0 || ov->hash_cap < 2 * ov->n_active || (ov->hash_cap & (ov->hash_cap - 1))) return -1;
  ec_hal_sync_all_for_cpu(stream);
  memset(ov->hash_keys_d, 0, (size_t)ov->hash_cap * sizeof(uint64_t));
  memset(ov->hash_vals_d, 0xFF, (size_t)ov->hash_cap * sizeof(int32_t));
  const uint32_t mask = (uint32_t)(ov->hash_cap - 1);
  for (int i = 0; i < ov->n_active; i++) {
    uint64_t key = ec_cmn_ov_pack(ov->coords_d[3 * i + 0], ov->coords_d[3 * i + 1], ov->coords_d[3 * i + 2]);
    uint32_t slot = ec_cmn_ov_hash(key, mask);
    while (ov->hash_keys_d[slot] != 0ULL && ov->hash_keys_d[slot] != key) slot = (slot + 1u) & mask;
    ov->hash_keys_d[slot] = key;
    ov->hash_vals_d[slot] = i;
  }
  ec_hal_mark_host_dirty(ov->hash_keys_d);
  ec_hal_mark_host_dirty(ov->hash_vals_d);
  return 0;
}

int ec_launch_ovoxel_sparse_conv3d(
    const ec_ovoxel_t *ov, const float *feats_in,
    const float *w, const float *bias, float *feats_out,
    int Cout, cudaStream_t stream) {
  if (!ov || ov->n_active <= 0 || Cout <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int Cin = ov->channels;
  const uint32_t mask = (uint32_t)(ov->hash_cap - 1);
  const long total = (long)ov->n_active * Cout;
  EC_PAR_FOR_BEGIN(idx, total)
    int vi = (int)(idx / Cout), co = (int)(idx % Cout);
    int32_t x = ov->coords_d[3 * vi + 0], y = ov->coords_d[3 * vi + 1], z = ov->coords_d[3 * vi + 2];
    float acc = bias ? bias[co] : 0.0f;
    int k = 0;
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++, k++) {
          int32_t nb = (dx == 0 && dy == 0 && dz == 0) ? vi : ec_cmn_ov_lookup(ov->hash_keys_d, ov->hash_vals_d, mask, x + dx, y + dy, z + dz);
          if (nb < 0) continue;
          const float *fn = feats_in + (long)nb * Cin;
          const float *wk = w + ((long)k * Cout + co) * Cin;
          for (int ci = 0; ci < Cin; ci++) acc += fn[ci] * wk[ci];
        }
    feats_out[idx] = acc;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(feats_out);
  return 0;
}

#ifndef EC_BACKEND_VULKAN
int ec_launch_flow_cfg_euler(
    const float *x, const float *v_cond, const float *v_uncond,
    float *x_next, int n_elems, float dt, float cfg_scale, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems)
    float v = v_uncond ? (v_uncond[i] + cfg_scale * (v_cond[i] - v_uncond[i])) : v_cond[i];
    x_next[i] = x[i] + dt * v;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(x_next);
  return 0;
}
#endif

int ec_launch_l1_rel_diff(const float *a, const float *b, float *out_stats_d, int n_elems, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  double ld = 0.0, lr = 0.0;
  #pragma omp parallel for reduction(+:ld,lr) schedule(static)
  for (int i = 0; i < n_elems; i++) {
    ld += fabs((double)a[i] - (double)b[i]);
    lr += fabs((double)b[i]);
  }
  out_stats_d[0] = (float)ld;
  out_stats_d[1] = (float)lr;
  ec_hal_mark_host_dirty(out_stats_d);
  return 0;
}

int ec_launch_kv_ring_append(
    ec_kv_ring_t *ring, const __half *chunk_k, const __half *chunk_v,
    int chunk_tokens, cudaStream_t stream) {
  if (!ring || chunk_tokens <= 0 || chunk_tokens > ring->cap_tokens) return -1;
  ec_hal_sync_all_for_cpu(stream);
  __half *kr = (__half *)ring->k_ring_d;
  __half *vr = (__half *)ring->v_ring_d;
  const long total = (long)ring->n_heads * chunk_tokens * ring->head_dim;
  EC_PAR_FOR_BEGIN(idx, total)
    int d = (int)(idx % ring->head_dim);
    long rem = idx / ring->head_dim;
    int t = (int)(rem % chunk_tokens);
    int h = (int)(rem / chunk_tokens);
    int dst_t = (ring->write_pos + t) % ring->cap_tokens;
    long dst_o = ((long)h * ring->cap_tokens + dst_t) * ring->head_dim + d;
    kr[dst_o] = chunk_k[idx];
    vr[dst_o] = chunk_v[idx];
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(ring->k_ring_d);
  ec_hal_mark_host_dirty(ring->v_ring_d);
  ring->write_pos = (ring->write_pos + chunk_tokens) % ring->cap_tokens;
  ring->valid_tokens += chunk_tokens;
  if (ring->valid_tokens > ring->cap_tokens) ring->valid_tokens = ring->cap_tokens;
  return 0;
}

int ec_launch_bf16_to_f16(const uint16_t *src_bf16, __half *dst_f16, size_t n, cudaStream_t stream) {
  if (!src_bf16 || !dst_f16 || n == 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n) dst_f16[i] = __float2half_rn(ec_cmn_bf16_to_f32(src_bf16[i])); EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst_f16);
  return 0;
}

int ec_launch_fp8_to_f16(const uint8_t *src_fp8, __half *dst_f16, size_t n, float scale, cudaStream_t stream) {
  if (!src_fp8 || !dst_f16 || n == 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n) dst_f16[i] = __float2half_rn(ec_cmn_fp8_e4m3_to_f32(src_fp8[i]) * scale); EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst_f16);
  return 0;
}

int ec_launch_gemm_bf16(
    const __half *X_f16, const uint16_t *W_bf16, __half *w_stage_f16,
    const float *bias, float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue, cudaStream_t stream) {
  if (ec_launch_bf16_to_f16(W_bf16, w_stage_f16, (size_t)N * K, stream) != 0) return -1;
  return ec_launch_gemm_f16(X_f16, w_stage_f16, bias, NULL, Y, Y_f16, M, N, K, epilogue, stream);
}

int ec_launch_gemm_fp8(
    const __half *X_f16, const uint8_t *W_fp8, float w_scale, __half *w_stage_f16,
    const float *bias, float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue, cudaStream_t stream) {
  if (ec_launch_fp8_to_f16(W_fp8, w_stage_f16, (size_t)N * K, w_scale, stream) != 0) return -1;
  return ec_launch_gemm_f16(X_f16, w_stage_f16, bias, NULL, Y, Y_f16, M, N, K, epilogue, stream);
}

int ec_launch_conv2d_tc(
    const float *x, const __half *w_f16, const float *bias,
    __half *col_f16, float *y_rowmajor_tmp, float *y,
    int Cin, int Cout, int H, int W, int Kh, int Kw, int stride, int pad,
    cudaStream_t stream) {
  const int H_out = (H + 2 * pad - Kh) / stride + 1;
  const int W_out = (W + 2 * pad - Kw) / stride + 1;
  if (H_out <= 0 || W_out <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int M = H_out * W_out;
  const int K = Cin * Kh * Kw;
  const long col_n = (long)M * K;
  EC_PAR_FOR_BEGIN(idx, col_n)
    int m = (int)(idx / K), k = (int)(idx % K);
    int ho = m / W_out, wo = m % W_out;
    int kw = k % Kw, tmp = k / Kw;
    int kh = tmp % Kh, ci = tmp / Kh;
    int hi = ho * stride - pad + kh, wi = wo * stride - pad + kw;
    float v = (hi >= 0 && hi < H && wi >= 0 && wi < W) ? x[((long)ci * H + hi) * W + wi] : 0.0f;
    col_f16[idx] = __float2half_rn(v);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(col_f16);
  if (ec_launch_gemm_f16(col_f16, w_f16, bias, NULL, y_rowmajor_tmp, NULL, M, Cout, K, bias ? 1 : 0, stream) != 0)
    return -1;
  ec_hal_sync_all_for_cpu(stream);
  const long out_n = (long)M * Cout;
  EC_PAR_FOR_BEGIN(idx, out_n)
    int c = (int)(idx / M), m = (int)(idx % M);
    y[idx] = y_rowmajor_tmp[(long)m * Cout + c];
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

int ec_launch_upsample2x_2d(const float *x, float *y, int C, int H, int W, cudaStream_t stream) {
  int H2 = H * 2, W2 = W * 2;
  long total = (long)C * H2 * W2;
  if (total <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(idx, total)
    int w = (int)(idx % W2);
    long rem = idx / W2;
    int h = (int)(rem % H2);
    int c = (int)(rem / H2);
    y[idx] = x[((long)c * H + (h >> 1)) * W + (w >> 1)];
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y);
  return 0;
}

int ec_launch_sage_attn_i8(
    const __half *Q, const __half *K, const __half *V,
    int8_t *q_i8, int8_t *k_i8, float *q_sc, float *k_sc,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size,
    cudaStream_t stream) {
  (void)q_i8; (void)k_i8; (void)q_sc; (void)k_sc;
  return ec_launch_fa2_f16(Q, K, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, head_dim, scale, block_causal_size, stream);
}

int ec_launch_cross_attn(
    const __half *Q, const __half *K, const __half *V,
    int8_t *q_i8, int8_t *k_i8, float *q_sc, float *k_sc,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, ec_tier_t tier,
    cudaStream_t stream) {
  (void)q_i8; (void)k_i8; (void)q_sc; (void)k_sc; (void)tier;
  return ec_launch_fa2_f16(Q, K, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, head_dim, scale, 0, stream);
}

#ifndef EC_BACKEND_VULKAN
int ec_launch_swiglu_mul(const float *gate, const float *up, float *out, __half *out_f16, int n_elems, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems)
    float v = ec_cmn_silu(gate[i]) * up[i];
    out[i] = v;
    if (out_f16) out_f16[i] = __float2half_rn(v);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(out);
  ec_hal_mark_host_dirty(out_f16);
  return 0;
}
#endif

int ec_launch_timestep_embed(float t_val, float *out_emb, int dim, float max_period, cudaStream_t stream) {
  if (dim <= 0 || (dim & 1)) return -1;
  ec_hal_sync_all_for_cpu(stream);
  const int half_d = dim >> 1;
  const float lmp = logf(max_period);
  for (int i = 0; i < half_d; i++) {
    float freq = expf(-lmp * (float)i / (float)half_d);
    float arg = t_val * freq;
    out_emb[i] = cosf(arg);
    out_emb[i + half_d] = sinf(arg);
  }
  ec_hal_mark_host_dirty(out_emb);
  return 0;
}

void ec_rf_shifted_schedule(int n_steps, float shift, float *out_ts) {
  if (n_steps <= 0 || !out_ts) return;
  for (int i = 0; i <= n_steps; i++) {
    float t = 1.0f - (float)i / (float)n_steps;
    if (shift > 0.0f && shift != 1.0f) t = (shift * t) / (1.0f + (shift - 1.0f) * t);
    out_ts[i] = t;
  }
}

int ec_launch_flow_heun(const float *x, const float *v1, const float *v2, float *x_next, int n_elems, float dt, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems) x_next[i] = x[i] + 0.5f * dt * (v1[i] + v2[i]); EC_PAR_FOR_END
  ec_hal_mark_host_dirty(x_next);
  return 0;
}

int ec_launch_modality_cfg(
    const float *v_uncond, const float *v_text, const float *v_modal,
    float *v_out, int n_elems, float cfg_text, float cfg_modal, cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  ec_hal_sync_all_for_cpu(stream);
  EC_PAR_FOR_BEGIN(i, n_elems)
    float v = v_uncond[i] + cfg_text * (v_text[i] - v_uncond[i]);
    if (v_modal) v += cfg_modal * (v_modal[i] - v_text[i]);
    v_out[i] = v;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(v_out);
  return 0;
}

#ifdef __cplusplus
}
#endif
