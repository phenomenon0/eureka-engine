// ec_model_kernels_portable.h — Barrier-Free Portable Implementations of the 25 Warp/Shared Model Helpers
// Used ONLY when compiling without nvcc (EC_BACKEND_METAL & EC_BACKEND_VULKAN).
// Routes norms & GQA/FA2 attention through GPU ec_ops_* where applicable, and marks CPU-written
// outputs via ec_hal_mark_host_dirty so discrete Vulkan DEVICE_LOCAL VRAM mirrors stay coherent.

#pragma once
#ifndef __CUDACC__

#include "../ec_hal.h"
#include "../ec_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef EC_BACKEND_VULKAN
#ifdef __cplusplus
extern "C" {
#endif
int ec_vk_k_flux2_pack_qkv_rope(const float *qkv, int row_stride, const float *q_norm_w, const float *k_norm_w,
    const float *rope_cos, const float *rope_sin, __half *Q_out, __half *K_out, __half *V_out,
    int n_tok, int tok_off, int n_pad, int H, int D, float eps);
int ec_vk_k_flux2_single_cat_q8g64(const float *attn_out, const float *lin1, int8_t *dst_q8, float *dst_sc,
    int N_tot, int E, int F);
int ec_vk_k_nhwc_gn_stats(const float *x_nhwc, float *stats64, int M, int C);
int ec_vk_k_nhwc_gn_apply_bf16(const float *x_nhwc, const float *stats64, const float *gamma, const float *beta,
    uint16_t *y_bf16, int M, int C, float eps, int apply_silu);
int ec_vk_k_upsample2x_nhwc_bf16(const float *x_nhwc, uint16_t *y_bf16, int H, int W, int C);
int ec_vk_k_vae_mid_attn(const void *Q, const void *K, const void *V, uint16_t *O_bf16,
    int S, int C, float scale, int is_bf16);
int ec_vk_k_qwen3_qk_rope(const float *q_in, const float *k_in, const float *v_in,
    const float *q_norm_w, const float *k_norm_w, float *q_out, float *k_out, float *v_out,
    int L, int Hq, int Hkv, int D, float theta, float eps);
int ec_vk_k_qwen3_gqa_attn(const float *Q, const float *K, const float *V, const uint8_t *mask,
    uint16_t *O_bf16, int L, int Hq, int Hkv, int D, float scale);
int ec_vk_f32_to_bf16(const float *src, uint16_t *dst, long n);
int ec_vk_add_inplace(float *dst, const float *src, long n);
int ec_vk_swiglu_pair_bf16(const float *gate, const float *up, uint16_t *out, long n);
int ec_vk_flux2_swiglu_bf16(const float *in2h, uint16_t *out_h, int rows, int H);
#ifdef __cplusplus
}
#endif
#endif

static inline float ecp_bf16_to_f32(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
static inline uint16_t ecp_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline float ecp_silu(float x) { return x / (1.0f + expf(-x)); }

// --- 1. Shared Transformer Norms (route to ec_ops_*) ---
static inline void ec_port_k_rmsnorm_bf16(
    cudaStream_t s, const float *x, const float *w, uint16_t *y_bf16, int rows, int dim, float eps) {
  ec_ops_rmsnorm_adaln(x, w, NULL, NULL, NULL, y_bf16, rows, dim, eps, s);
}

static inline void ec_port_k_adaln_bf16(
    cudaStream_t s, const float *x, const float *shift, const float *scale,
    uint16_t *y_bf16, int rows, int dim, float eps) {
  ec_ops_layernorm_adaln(x, NULL, NULL, scale, shift, NULL, y_bf16, rows, dim, eps, s);
}

static inline void ec_port_k_as_rmsnorm_adaln(
    cudaStream_t s, const float *x, const float *w, const float *scale, const float *shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps) {
  ec_ops_rmsnorm_adaln(x, w, scale, shift, y_f32, y_bf16, rows, dim, eps, s);
}

static inline void ec_port_k_tr_layernorm(
    cudaStream_t s, const float *x, const float *w, const float *b,
    const float *adaln_scale, const float *adaln_shift,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, int apply_silu) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float s1 = 0.0f, s2 = 0.0f;
    for (int c = 0; c < dim; c++) { float v = xr[c]; s1 += v; s2 += v * v; }
    float mean = s1 / (float)dim;
    float var = fmaxf(0.0f, s2 / (float)dim - mean * mean);
    float inv = rsqrtf(var + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = (xr[c] - mean) * inv;
      if (w) v = v * w[c] + b[c];
      if (adaln_scale) v = v * (1.0f + adaln_scale[c]) + adaln_shift[c];
      if (apply_silu) v = ecp_silu(v);
      if (yf) yf[c] = v;
      if (yb) yb[c] = ecp_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
}

static inline void ec_port_k_wn_rmsnorm(
    cudaStream_t s, const float *x, const float *w,
    float *y_f32, uint16_t *y_bf16, int rows, int dim, float eps, int apply_silu) {
  if (!apply_silu) {
    ec_ops_rmsnorm_adaln(x, w, NULL, NULL, y_f32, y_bf16, rows, dim, eps, s);
    return;
  }
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(r, rows)
    const float *xr = x + (long)r * dim;
    float ss = 0.0f;
    for (int c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float inv = rsqrtf(ss / (float)dim + eps);
    float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
    uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
    for (int c = 0; c < dim; c++) {
      float v = ecp_silu(xr[c] * inv * (w ? w[c] : 1.0f));
      if (yf) yf[c] = v;
      if (yb) yb[c] = ecp_f32_to_bf16(v);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_f32);
  ec_hal_mark_host_dirty(y_bf16);
}

static inline void ec_port_k_wn_layernorm(
    cudaStream_t s, const float *x, const float *w, const float *b,
    const float *adaln_scale, const float *adaln_shift,
    uint16_t *y_bf16, int rows, int dim, float eps) {
  ec_ops_layernorm_adaln(x, w, b, adaln_scale, adaln_shift, NULL, y_bf16, rows, dim, eps, s);
}

// --- 2. FLUX.2 Per-Head RoPE, Q8 Single-Cat, NHWC GroupNorm, Upsample & VAE Mid-Attn ---
static inline void ec_port_k_qwen3_qk_rope(
    cudaStream_t s, const float *q_in, const float *k_in, const float *v_in,
    const float *q_norm_w, const float *k_norm_w,
    float *q_out, float *k_out, float *v_out,
    int L, int Hq, int Hkv, int D, float theta, float eps) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_qwen3_qk_rope(q_in, k_in, v_in, q_norm_w, k_norm_w, q_out, k_out, v_out, L, Hq, Hkv, D, theta, eps) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int total_q = L * Hq, total_kv = L * Hkv, half = D >> 1;
  EC_PAR_FOR_BEGIN(idx, total_q + total_kv)
    if (idx < total_q) {
      int pos = idx / Hq, h = idx % Hq;
      const float *qs = q_in + ((long)pos * Hq + h) * D;
      float *qo = q_out + ((long)h * L + pos) * D;
      float sq = 0.0f;
      for (int d = 0; d < D; d++) sq += qs[d] * qs[d];
      float inv = rsqrtf(sq / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = qs[j] * inv * q_norm_w[j];
        float x1 = qs[j + half] * inv * q_norm_w[j + half];
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        qo[j] = x0 * cs - x1 * sn;
        qo[j + half] = x1 * cs + x0 * sn;
      }
    } else {
      int kidx = idx - total_q;
      int pos = kidx / Hkv, h = kidx % Hkv;
      const float *ks = k_in + ((long)pos * Hkv + h) * D;
      const float *vs = v_in + ((long)pos * Hkv + h) * D;
      float *ko = k_out + ((long)h * L + pos) * D;
      float *vo = v_out + ((long)h * L + pos) * D;
      float sk = 0.0f;
      for (int d = 0; d < D; d++) sk += ks[d] * ks[d];
      float inv = rsqrtf(sk / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = ks[j] * inv * k_norm_w[j];
        float x1 = ks[j + half] * inv * k_norm_w[j + half];
        double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
        float cs = (float)cos(ang), sn = (float)sin(ang);
        ko[j] = x0 * cs - x1 * sn;
        ko[j + half] = x1 * cs + x0 * sn;
        vo[j] = vs[j];
        vo[j + half] = vs[j + half];
      }
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(q_out);
  ec_hal_mark_host_dirty(k_out);
  ec_hal_mark_host_dirty(v_out);
}

static inline void ec_port_k_qwen3_gqa_attn(
    cudaStream_t s, const float *Q, const float *K, const float *V,
    const uint8_t *mask, uint16_t *O_bf16, int L, int Hq, int Hkv, int D, float scale) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_qwen3_gqa_attn(Q, K, V, mask, O_bf16, L, Hq, Hkv, D, scale) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(pair, L * Hq)
    int qpos = pair / Hq, hq = pair % Hq, hkv = hq / (Hq / Hkv);
    const float *qvec = Q + ((long)hq * L + qpos) * D;
    const float *kbase = K + (long)hkv * L * D;
    const float *vbase = V + (long)hkv * L * D;
    float acc[128] = {0};
    float m_max = -1e30f, l_sum = 0.0f;
    for (int kpos = 0; kpos <= qpos; kpos++) {
      if (mask[kpos] == 0 && kpos != qpos) continue;
      const float *kv = kbase + (long)kpos * D;
      float dot = 0.0f;
      for (int d = 0; d < D; d++) dot += qvec[d] * kv[d];
      dot *= scale;
      float nm = fmaxf(m_max, dot);
      float al = expf(m_max - nm), p = expf(dot - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const float *vv = vbase + (long)kpos * D;
      for (int d = 0; d < D; d++) acc[d] = acc[d] * al + p * vv[d];
    }
    float inv = 1.0f / fmaxf(l_sum, 1e-20f);
    uint16_t *out = O_bf16 + ((long)qpos * Hq + hq) * D;
    for (int d = 0; d < D; d++) out[d] = ecp_f32_to_bf16(acc[d] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(O_bf16);
}

static inline void ec_port_k_flux2_pack_qkv_rope(
    cudaStream_t s, const float *qkv, int row_stride,
    const float *q_norm_w, const float *k_norm_w,
    const float *rope_cos, const float *rope_sin,
    __half *Q_out, __half *K_out, __half *V_out,
    int n_tok, int tok_off, int n_pad, int H, int D, float eps) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_flux2_pack_qkv_rope(qkv, row_stride, q_norm_w, k_norm_w, rope_cos, rope_sin,
                                  Q_out, K_out, V_out, n_tok, tok_off, n_pad, H, D, eps) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int E = H * D;
  EC_PAR_FOR_BEGIN(pair, (long)n_tok * H)
    int h = (int)(pair % H), t = (int)(pair / H);
    const float *base = qkv + (long)t * row_stride;
    const float *qs = base + h * D, *ks = base + E + h * D, *vs = base + 2 * E + h * D;
    float sq = 0.0f, sk = 0.0f;
    for (int d = 0; d < D; d++) { sq += qs[d] * qs[d]; sk += ks[d] * ks[d]; }
    float inv_q = rsqrtf(sq / (float)D + eps), inv_k = rsqrtf(sk / (float)D + eps);
    int gtok = tok_off + t;
    const float *rc = rope_cos + (long)gtok * (D >> 1);
    const float *rs = rope_sin + (long)gtok * (D >> 1);
    long out_base = ((long)h * n_pad + gtok) * D;
    __half *qo = Q_out + out_base, *ko = K_out + out_base, *vo = V_out + out_base;
    for (int p = 0; p < (D >> 1); p++) {
      int d0 = 2 * p, d1 = d0 + 1;
      float q0 = qs[d0] * inv_q * q_norm_w[d0], q1 = qs[d1] * inv_q * q_norm_w[d1];
      float k0 = ks[d0] * inv_k * k_norm_w[d0], k1 = ks[d1] * inv_k * k_norm_w[d1];
      float cs = rc[p], sn = rs[p];
      qo[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, q0 * cs - q1 * sn)));
      qo[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, q0 * sn + q1 * cs)));
      ko[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, k0 * cs - k1 * sn)));
      ko[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, k0 * sn + k1 * cs)));
      vo[d0] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, vs[d0])));
      vo[d1] = __float2half_rn(fmaxf(-65000.0f, fminf(65000.0f, vs[d1])));
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(Q_out);
  ec_hal_mark_host_dirty(K_out);
  ec_hal_mark_host_dirty(V_out);
}

static inline void ec_port_k_flux2_single_cat_q8g64(
    cudaStream_t s, const float *attn_out, const float *lin1,
    int8_t *dst_q8, float *dst_sc, int N_tot, int E, int F) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_flux2_single_cat_q8g64(attn_out, lin1, dst_q8, dst_sc, N_tot, E, F) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int stride_out = E + F, stride_in = 3 * E + 2 * F;
  const int gpr = stride_out >> 6;
  const long total_g = (long)N_tot * gpr;
  EC_PAR_FOR_BEGIN(g, total_g)
    int t = (int)(g / gpr);
    int c0 = (int)(g - (long)t * gpr) * 64;
    float vf[64], amax = 0.0f;
    for (int i = 0; i < 64; i++) {
      int c = c0 + i;
      float v;
      if (c < E) v = attn_out[(long)t * E + c];
      else {
        int fc = c - E;
        const float *row = lin1 + (long)t * stride_in + 3 * E;
        v = ecp_silu(row[fc]) * row[F + fc];
      }
      v = ecp_bf16_to_f32(ecp_f32_to_bf16(v));
      vf[i] = v;
      if (fabsf(v) > amax) amax = fabsf(v);
    }
    float sc = amax / 127.0f;
    float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
    dst_sc[g] = sc;
    int8_t *oq = dst_q8 + g * 64;
    for (int i = 0; i < 64; i++) {
      long q = lrintf(vf[i] * inv);
      if (q < -127) q = -127;
      if (q > 127) q = 127;
      oq[i] = (int8_t)q;
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(dst_q8);
  ec_hal_mark_host_dirty(dst_sc);
}

static inline void ec_port_k_vae_mid_attn(
    cudaStream_t s, const float *Q, const float *K, const float *V,
    uint16_t *O_bf16, int S, int C, float scale) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_vae_mid_attn(Q, K, V, O_bf16, S, C, scale, 0) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(q, S)
    const float *qv = Q + (long)q * C;
    float acc[512] = {0};
    float m_max = -1e30f, l_sum = 0.0f;
    for (int k = 0; k < S; k++) {
      const float *kv = K + (long)k * C;
      float dot = 0.0f;
      for (int c = 0; c < C; c++) dot += qv[c] * kv[c];
      float sc = dot * scale;
      float nm = fmaxf(m_max, sc);
      float al = expf(m_max - nm), p = expf(sc - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const float *vv = V + (long)k * C;
      for (int c = 0; c < C; c++) acc[c] = acc[c] * al + p * vv[c];
    }
    float inv = 1.0f / fmaxf(l_sum, 1e-20f);
    uint16_t *out = O_bf16 + (long)q * C;
    for (int c = 0; c < C; c++) out[c] = ecp_f32_to_bf16(acc[c] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(O_bf16);
}

static inline void ec_port_k_vae_mid_attn_tc_d512(
    cudaStream_t s, const uint16_t *Q_bf, const uint16_t *K_bf, const uint16_t *V_bf,
    uint16_t *O_bf, int S, float scale) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_vae_mid_attn(Q_bf, K_bf, V_bf, O_bf, S, 512, scale, 1) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(q, S)
    const uint16_t *qv = Q_bf + (long)q * 512;
    float qf[512], acc[512];
    for (int c = 0; c < 512; c++) { qf[c] = ecp_bf16_to_f32(qv[c]); acc[c] = 0.0f; }
    float m_max = -1e30f, l_sum = 0.0f;
    for (int k = 0; k < S; k++) {
      const uint16_t *kv = K_bf + (long)k * 512;
      float dot = 0.0f;
      for (int c = 0; c < 512; c++) dot += qf[c] * ecp_bf16_to_f32(kv[c]);
      float sc = dot * scale;
      float nm = fmaxf(m_max, sc);
      float al = expf(m_max - nm), p = expf(sc - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const uint16_t *vv = V_bf + (long)k * 512;
      for (int c = 0; c < 512; c++) acc[c] = acc[c] * al + p * ecp_bf16_to_f32(vv[c]);
    }
    float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint16_t *out = O_bf + (long)q * 512;
    for (int c = 0; c < 512; c++) out[c] = ecp_f32_to_bf16(acc[c] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(O_bf);
}

static inline void ec_port_k_nhwc_gn_stats(
    cudaStream_t s, const float *x_nhwc, float *stats64, int M, int C) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_nhwc_gn_stats(x_nhwc, stats64, M, C) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int gc = C >> 5;
  EC_PAR_FOR_BEGIN(grp, 32)
    int c0 = grp * gc;
    double s1 = 0.0, s2 = 0.0;
    for (int m = 0; m < M; m++) {
      const float *xm = x_nhwc + (long)m * C + c0;
      for (int dc = 0; dc < gc; dc++) { double v = xm[dc]; s1 += v; s2 += v * v; }
    }
    stats64[grp] = (float)s1;
    stats64[32 + grp] = (float)s2;
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(stats64);
}

static inline void ec_port_k_nhwc_gn_apply_bf16(
    cudaStream_t s, const float *x_nhwc, const float *stats64,
    const float *gamma, const float *beta,
    uint16_t *y_bf16, int M, int C, float eps, int apply_silu) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_nhwc_gn_apply_bf16(x_nhwc, stats64, gamma, beta, y_bf16, M, C, eps, apply_silu) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int gc = C >> 5;
  const float inv_n = 1.0f / ((float)M * (float)gc);
  float s_mean[32], s_rstd[32];
  for (int g = 0; g < 32; g++) {
    float mean = stats64[g] * inv_n;
    float var = fmaxf(0.0f, stats64[32 + g] * inv_n - mean * mean);
    s_mean[g] = mean;
    s_rstd[g] = rsqrtf(var + eps);
  }
  const long tot = (long)M * C;
  EC_PAR_FOR_BEGIN(i, tot)
    int c = (int)(i % C);
    int g = c / gc;
    float v = (x_nhwc[i] - s_mean[g]) * s_rstd[g] * gamma[c] + beta[c];
    if (apply_silu) v = ecp_silu(v);
    y_bf16[i] = ecp_f32_to_bf16(v);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_bf16);
}

static inline void ec_port_k_upsample2x_nhwc_bf16(
    cudaStream_t s, const float *x_nhwc, uint16_t *y_bf16, int H, int W, int C) {
#ifdef EC_BACKEND_VULKAN
  if (ec_vk_k_upsample2x_nhwc_bf16(x_nhwc, y_bf16, H, W, C) == 0) return;
#endif
  ec_hal_sync_all_for_cpu(s);
  const int H2 = H * 2, W2 = W * 2;
  EC_PAR_FOR_BEGIN(pix, H2 * W2)
    int y2 = pix / W2, x2 = pix - y2 * W2;
    const float *src = x_nhwc + (long)((y2 >> 1) * W + (x2 >> 1)) * C;
    uint16_t *dst = y_bf16 + (long)pix * C;
    for (int c = 0; c < C; c++) dst[c] = ecp_f32_to_bf16(src[c]);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(y_bf16);
}

static inline void ec_port_k_vae_conv_out_rgb8_nhwc(
    cudaStream_t s, const uint16_t *X_nhwc, const uint16_t *W_ohwi, const float *bias,
    uint8_t *out_rgb, int H, int W) {
  ec_hal_sync_all_for_cpu(s);
  float sW[3][9 * 128];
  for (int i = 0; i < 3 * 9 * 128; i++) sW[i / (9 * 128)][i % (9 * 128)] = ecp_bf16_to_f32(W_ohwi[i]);
  EC_PAR_FOR_BEGIN(pix, H * W)
    int oy = pix / W, ox = pix - oy * W;
    float r = bias ? bias[0] : 0.0f, g = bias ? bias[1] : 0.0f, b = bias ? bias[2] : 0.0f;
    for (int tap = 0; tap < 9; tap++) {
      int iy = oy + (tap / 3) - 1, ix = ox + (tap % 3) - 1;
      if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
        const uint16_t *xp = X_nhwc + (long)(iy * W + ix) * 128;
        const int woff = tap * 128;
        for (int c = 0; c < 128; c++) {
          float xf = ecp_bf16_to_f32(xp[c]);
          r += xf * sW[0][woff + c];
          g += xf * sW[1][woff + c];
          b += xf * sW[2][woff + c];
        }
      }
    }
    auto to_u8 = [](float v) -> uint8_t {
      float c = (v * 0.5f + 0.5f) * 255.0f + 0.5f;
      return (uint8_t)(c < 0.0f ? 0.0f : (c > 255.0f ? 255.0f : c));
    };
    out_rgb[(long)pix * 3 + 0] = to_u8(r);
    out_rgb[(long)pix * 3 + 1] = to_u8(g);
    out_rgb[(long)pix * 3 + 2] = to_u8(b);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(out_rgb);
}

// --- 3. ACE-Step 1.5 Per-Head RoPE & GQA ---
static inline void ec_port_k_as_qk_norm_rope_bf16(
    cudaStream_t s, const float *q_in, const float *k_in, const float *v_in,
    const float *q_norm_w, const float *k_norm_w,
    uint16_t *q_out, uint16_t *k_out, uint16_t *v_out,
    int Lq, int Lkv, int Hq, int Hkv, int D, float theta, float eps, int apply_rope) {
  ec_hal_sync_all_for_cpu(s);
  const int total_q = q_in ? (Lq * Hq) : 0, total_kv = k_in ? (Lkv * Hkv) : 0, half = D >> 1;
  EC_PAR_FOR_BEGIN(idx, total_q + total_kv)
    if (idx < total_q) {
      int pos = idx / Hq, h = idx % Hq;
      const float *qs = q_in + ((long)pos * Hq + h) * D;
      uint16_t *qo = q_out + ((long)h * Lq + pos) * D;
      float sq = 0.0f;
      for (int d = 0; d < D; d++) sq += qs[d] * qs[d];
      float inv = rsqrtf(sq / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = qs[j] * inv * q_norm_w[j], x1 = qs[j + half] * inv * q_norm_w[j + half];
        if (apply_rope) {
          double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          qo[j] = ecp_f32_to_bf16(x0 * cs - x1 * sn);
          qo[j + half] = ecp_f32_to_bf16(x1 * cs + x0 * sn);
        } else {
          qo[j] = ecp_f32_to_bf16(x0);
          qo[j + half] = ecp_f32_to_bf16(x1);
        }
      }
    } else {
      int kidx = idx - total_q, pos = kidx / Hkv, h = kidx % Hkv;
      const float *ks = k_in + ((long)pos * Hkv + h) * D, *vs = v_in + ((long)pos * Hkv + h) * D;
      uint16_t *ko = k_out + ((long)h * Lkv + pos) * D, *vo = v_out + ((long)h * Lkv + pos) * D;
      float sk = 0.0f;
      for (int d = 0; d < D; d++) sk += ks[d] * ks[d];
      float inv = rsqrtf(sk / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = ks[j] * inv * k_norm_w[j], x1 = ks[j + half] * inv * k_norm_w[j + half];
        if (apply_rope) {
          double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          ko[j] = ecp_f32_to_bf16(x0 * cs - x1 * sn);
          ko[j + half] = ecp_f32_to_bf16(x1 * cs + x0 * sn);
        } else {
          ko[j] = ecp_f32_to_bf16(x0);
          ko[j + half] = ecp_f32_to_bf16(x1);
        }
        vo[j] = ecp_f32_to_bf16(vs[j]);
        vo[j + half] = ecp_f32_to_bf16(vs[j + half]);
      }
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(q_out);
  ec_hal_mark_host_dirty(k_out);
  ec_hal_mark_host_dirty(v_out);
}

static inline void ec_port_k_as_qk_norm_rope(
    cudaStream_t s, const float *q_in, const float *k_in, const float *v_in,
    const float *q_norm_w, const float *k_norm_w,
    float *q_out, float *k_out, float *v_out,
    int Lq, int Lkv, int Hq, int Hkv, int D, float theta, float eps, int apply_rope) {
  ec_hal_sync_all_for_cpu(s);
  const int total_q = q_in ? (Lq * Hq) : 0, total_kv = k_in ? (Lkv * Hkv) : 0, half = D >> 1;
  EC_PAR_FOR_BEGIN(idx, total_q + total_kv)
    if (idx < total_q) {
      int pos = idx / Hq, h = idx % Hq;
      const float *qs = q_in + ((long)pos * Hq + h) * D;
      float *qo = q_out + ((long)h * Lq + pos) * D;
      float sq = 0.0f;
      for (int d = 0; d < D; d++) sq += qs[d] * qs[d];
      float inv = rsqrtf(sq / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = qs[j] * inv * q_norm_w[j], x1 = qs[j + half] * inv * q_norm_w[j + half];
        if (apply_rope) {
          double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          qo[j] = x0 * cs - x1 * sn; qo[j + half] = x1 * cs + x0 * sn;
        } else {
          qo[j] = x0; qo[j + half] = x1;
        }
      }
    } else {
      int kidx = idx - total_q, pos = kidx / Hkv, h = kidx % Hkv;
      const float *ks = k_in + ((long)pos * Hkv + h) * D, *vs = v_in + ((long)pos * Hkv + h) * D;
      float *ko = k_out + ((long)h * Lkv + pos) * D, *vo = v_out + ((long)h * Lkv + pos) * D;
      float sk = 0.0f;
      for (int d = 0; d < D; d++) sk += ks[d] * ks[d];
      float inv = rsqrtf(sk / (float)D + eps);
      for (int j = 0; j < half; j++) {
        float x0 = ks[j] * inv * k_norm_w[j], x1 = ks[j + half] * inv * k_norm_w[j + half];
        if (apply_rope) {
          double ang = (double)pos / pow((double)theta, (double)(2 * j) / (double)D);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          ko[j] = x0 * cs - x1 * sn; ko[j + half] = x1 * cs + x0 * sn;
        } else {
          ko[j] = x0; ko[j + half] = x1;
        }
        vo[j] = vs[j]; vo[j + half] = vs[j + half];
      }
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(q_out);
  ec_hal_mark_host_dirty(k_out);
  ec_hal_mark_host_dirty(v_out);
}

static inline void ec_port_k_as_gqa_attn(
    cudaStream_t s, const float *Q, const float *K, const float *V,
    uint16_t *O_bf16, int Lq, int Lkv, int Hq, int Hkv, int D, float scale, int mode, int window) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(pair, Lq * Hq)
    int qpos = pair / Hq, hq = pair % Hq, hkv = hq / (Hq / Hkv);
    const float *qvec = Q + ((long)hq * Lq + qpos) * D;
    const float *kbase = K + (long)hkv * Lkv * D, *vbase = V + (long)hkv * Lkv * D;
    int k_start = 0, k_end = Lkv - 1;
    if (mode == 0) k_end = qpos < Lkv ? qpos : (Lkv - 1);
    else if (mode == 2) {
      k_start = qpos - window; if (k_start < 0) k_start = 0;
      k_end = qpos + window;   if (k_end >= Lkv) k_end = Lkv - 1;
    }
    float acc[128] = {0};
    float m_max = -1e30f, l_sum = 0.0f;
    for (int kpos = k_start; kpos <= k_end; kpos++) {
      const float *kv = kbase + (long)kpos * D;
      float dot = 0.0f;
      for (int d = 0; d < D; d++) dot += qvec[d] * kv[d];
      dot *= scale;
      float nm = fmaxf(m_max, dot), al = expf(m_max - nm), p = expf(dot - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const float *vv = vbase + (long)kpos * D;
      for (int d = 0; d < D; d++) acc[d] = acc[d] * al + p * vv[d];
    }
    float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint16_t *out = O_bf16 + ((long)qpos * Hq + hq) * D;
    for (int d = 0; d < D; d++) out[d] = ecp_f32_to_bf16(acc[d] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(O_bf16);
}

// --- 4. TRELLIS.2 Per-Head 3D RoPE & Cross-Norm ---
static inline void ec_port_k_ss_self_qkv_rope(
    cudaStream_t s, const float *qkv, const float *q_gamma, const float *k_gamma,
    uint16_t *Q, uint16_t *K, uint16_t *V, int N, int H, int res) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(idx, N * H)
    int tok = idx / H, h = idx % H;
    const float *base = qkv + (long)tok * (3 * H * 128);
    const float *qs = base + (0 * H + h) * 128, *ks = base + (1 * H + h) * 128, *vs = base + (2 * H + h) * 128;
    const float *qg = q_gamma + (long)h * 128, *kg = k_gamma + (long)h * 128;
    uint16_t *qo = Q + ((long)h * N + tok) * 128, *ko = K + ((long)h * N + tok) * 128, *vo = V + ((long)h * N + tok) * 128;
    float sq = 0.0f, sk = 0.0f;
    for (int d = 0; d < 128; d++) { sq += qs[d] * qs[d]; sk += ks[d] * ks[d]; }
    float inv_q = rsqrtf(sq / 128.0f + 1e-12f), inv_k = rsqrtf(sk / 128.0f + 1e-12f);
    int ci = tok / (res * res), cj = (tok / res) % res, ck = tok % res;
    for (int p = 0; p < 64; p++) {
      int d0 = 2 * p, d1 = d0 + 1;
      float q0 = qs[d0] * inv_q * qg[d0], q1 = qs[d1] * inv_q * qg[d1];
      float k0 = ks[d0] * inv_k * kg[d0], k1 = ks[d1] * inv_k * kg[d1];
      float cs = 1.0f, sn = 0.0f;
      if (p < 63) {
        int axis = p / 21, fi = p % 21;
        float coord = (axis == 0) ? (float)ci : ((axis == 1) ? (float)cj : (float)ck);
        double ang = (double)coord / pow(10000.0, (double)fi / 21.0);
        cs = (float)cos(ang); sn = (float)sin(ang);
      }
      qo[d0] = ecp_f32_to_bf16(q0 * cs - q1 * sn); qo[d1] = ecp_f32_to_bf16(q0 * sn + q1 * cs);
      ko[d0] = ecp_f32_to_bf16(k0 * cs - k1 * sn); ko[d1] = ecp_f32_to_bf16(k0 * sn + k1 * cs);
      vo[d0] = ecp_f32_to_bf16(vs[d0]); vo[d1] = ecp_f32_to_bf16(vs[d1]);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(Q);
  ec_hal_mark_host_dirty(K);
  ec_hal_mark_host_dirty(V);
}

static inline void ec_port_k_ss_cross_q_norm(
    cudaStream_t s, const float *q_in, const float *q_gamma, uint16_t *Q, int Lq, int H) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(idx, Lq * H)
    int tok = idx / H, h = idx % H;
    const float *qs = q_in + ((long)tok * H + h) * 128, *qg = q_gamma + (long)h * 128;
    uint16_t *qo = Q + ((long)h * Lq + tok) * 128;
    float sq = 0.0f;
    for (int d = 0; d < 128; d++) sq += qs[d] * qs[d];
    float inv = rsqrtf(sq / 128.0f + 1e-12f);
    for (int d = 0; d < 128; d++) qo[d] = ecp_f32_to_bf16(qs[d] * inv * qg[d]);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(Q);
}

static inline void ec_port_k_ss_cross_kv_norm(
    cudaStream_t s, const float *kv_in, const float *k_gamma, uint16_t *K, uint16_t *V, int Lkv, int H) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(idx, Lkv * H)
    int tok = idx / H, h = idx % H;
    const float *ks = kv_in + ((long)tok * 2 * H + h) * 128;
    const float *vs = kv_in + ((long)tok * 2 * H + H + h) * 128;
    const float *kg = k_gamma + (long)h * 128;
    uint16_t *ko = K + ((long)h * Lkv + tok) * 128, *vo = V + ((long)h * Lkv + tok) * 128;
    float sk = 0.0f;
    for (int d = 0; d < 128; d++) sk += ks[d] * ks[d];
    float inv = rsqrtf(sk / 128.0f + 1e-12f);
    for (int d = 0; d < 128; d++) {
      ko[d] = ecp_f32_to_bf16(ks[d] * inv * kg[d]);
      vo[d] = ecp_f32_to_bf16(vs[d]);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(K);
  ec_hal_mark_host_dirty(V);
}

// --- 5. Wan 2.2 UMT5 Relative-Bias Attention, 3D RoPE & VAE Spatial Attention ---
static inline int ecp_t5_rel_bucket(int rel_pos) {
  int nb = 16, bucket = (rel_pos > 0) ? nb : 0;
  int a = (rel_pos < 0) ? -rel_pos : rel_pos, max_exact = 8;
  if (a < max_exact) bucket += a;
  else {
    float val = (float)max_exact + logf((float)a / (float)max_exact) / logf(128.0f / (float)max_exact) * (float)(nb - max_exact);
    int iv = (int)val; if (iv > nb - 1) iv = nb - 1;
    bucket += iv;
  }
  return bucket;
}

static inline void ec_port_k_wn_t5_attn_d64(
    cudaStream_t s, const float *Q, const float *K, const float *V,
    const float *rel_bias, uint16_t *O_bf16, int L, int H) {
  ec_hal_sync_all_for_cpu(s);
  EC_PAR_FOR_BEGIN(pair, L * H)
    int qpos = pair / H, h = pair % H;
    const float *qv = Q + ((long)qpos * H + h) * 64;
    float acc[64] = {0};
    float m_max = -1e30f, l_sum = 0.0f;
    for (int kpos = 0; kpos < L; kpos++) {
      const float *kv = K + ((long)kpos * H + h) * 64;
      float dot = 0.0f;
      for (int d = 0; d < 64; d++) dot += qv[d] * kv[d];
      dot += rel_bias[ecp_t5_rel_bucket(kpos - qpos) * H + h];
      float nm = fmaxf(m_max, dot), al = expf(m_max - nm), p = expf(dot - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const float *vv = V + ((long)kpos * H + h) * 64;
      for (int d = 0; d < 64; d++) acc[d] = acc[d] * al + p * vv[d];
    }
    float inv = 1.0f / (l_sum + 1e-12f);
    uint16_t *out = O_bf16 + ((long)qpos * H + h) * 64;
    for (int d = 0; d < 64; d++) out[d] = ecp_f32_to_bf16(acc[d] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(O_bf16);
}

static inline void ec_port_k_wn_sa_qkv_norm_rope_bf16(
    cudaStream_t s, const float *qkv, const float *nq_w, const float *nk_w,
    uint16_t *qr_b, uint16_t *kr_b, uint16_t *vr_b, int N, int F, int Hp, int Wp) {
  (void)F;
  ec_hal_sync_all_for_cpu(s);
  const int DIM = 3072;
  EC_PAR_FOR_BEGIN(tok, N)
    const float *qs = qkv + (long)tok * (3 * DIM), *ks = qs + DIM, *vs = qs + 2 * DIM;
    float sq = 0.0f, sk = 0.0f;
    for (int c = 0; c < DIM; c++) { sq += qs[c] * qs[c]; sk += ks[c] * ks[c]; }
    float inv_q = rsqrtf(sq / (float)DIM + 1e-6f), inv_k = rsqrtf(sk / (float)DIM + 1e-6f);
    int f = tok / (Hp * Wp), hp = (tok / Wp) % Hp, wp = tok % Wp;
    for (int gp = 0; gp < 1536; gp++) {
      int h = gp >> 6, p = gp & 63, d0 = 2 * p, d1 = d0 + 1;
      int c0 = h * 128 + d0, c1 = c0 + 1;
      float q0 = qs[c0] * inv_q * nq_w[c0], q1 = qs[c1] * inv_q * nq_w[c1];
      float k0 = ks[c0] * inv_k * nk_w[c0], k1 = ks[c1] * inv_k * nk_w[c1];
      double coord, denom; int fi;
      if (p < 22)      { coord = (double)f;  fi = p;      denom = 44.0; }
      else if (p < 43) { coord = (double)hp; fi = p - 22; denom = 42.0; }
      else             { coord = (double)wp; fi = p - 43; denom = 42.0; }
      double ang = coord / pow(10000.0, (double)(2 * fi) / denom);
      float cs = (float)cos(ang), sn = (float)sin(ang);
      long out_off = ((long)h * N + tok) * 128;
      qr_b[out_off + d0] = ecp_f32_to_bf16(q0 * cs - q1 * sn);
      qr_b[out_off + d1] = ecp_f32_to_bf16(q0 * sn + q1 * cs);
      kr_b[out_off + d0] = ecp_f32_to_bf16(k0 * cs - k1 * sn);
      kr_b[out_off + d1] = ecp_f32_to_bf16(k0 * sn + k1 * cs);
      vr_b[out_off + d0] = ecp_f32_to_bf16(vs[c0]);
      vr_b[out_off + d1] = ecp_f32_to_bf16(vs[c1]);
    }
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(qr_b);
  ec_hal_mark_host_dirty(kr_b);
  ec_hal_mark_host_dirty(vr_b);
}

static inline void ec_port_k_wn_vae_attn_1h(
    cudaStream_t s, const float *qkv, uint16_t *out_bf16, int HW, int C) {
  ec_hal_sync_all_for_cpu(s);
  const float scale = rsqrtf((float)C);
  EC_PAR_FOR_BEGIN(qpos, HW)
    const float *qv = qkv + (long)qpos * (3 * C);
    float acc[1024] = {0};
    float m_max = -1e30f, l_sum = 0.0f;
    for (int kpos = 0; kpos < HW; kpos++) {
      const float *kv = qkv + (long)kpos * (3 * C) + C;
      float dot = 0.0f;
      for (int c = 0; c < C; c++) dot += qv[c] * kv[c];
      float sc = dot * scale;
      float nm = fmaxf(m_max, sc), al = expf(m_max - nm), p = expf(sc - nm);
      l_sum = l_sum * al + p; m_max = nm;
      const float *vv = qkv + (long)kpos * (3 * C) + 2 * C;
      for (int c = 0; c < C; c++) acc[c] = acc[c] * al + p * vv[c];
    }
    float inv = 1.0f / (l_sum + 1e-12f);
    uint16_t *out = out_bf16 + (long)qpos * C;
    for (int c = 0; c < C; c++) out[c] = ecp_f32_to_bf16(acc[c] * inv);
  EC_PAR_FOR_END
  ec_hal_mark_host_dirty(out_bf16);
}

#endif // !__CUDACC__

#ifdef __CUDACC__
  #define EC_DISPATCH_PORT(kern, grid, block, smem, stream, ...) \
    kern<<<(grid), (block), (smem), (stream)>>>(__VA_ARGS__)
#else
  #define EC_DISPATCH_PORT(kern, grid, block, smem, stream, ...) \
    ec_port_##kern((stream), __VA_ARGS__)
#endif
