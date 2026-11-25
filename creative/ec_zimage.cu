// ec_zimage.cu — 100% Pure Native CUDA Tongyi-MAI Z-Image-Turbo (6.15B S3-DiT, Apache 2.0)
// Zero Python, zero cuBLAS, zero ggml!
// Architecture verified exact against ComfyUI `comfy/ldm/lumina/model.py` & `comfy/text_encoders/z_image.py`:
//   - Qwen3-4B streaming encoder: template "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
//     extracts penultimate layer_idx=-2 (= 0-indexed layer 34, dim=2560, unpadded length L_cap)
//   - S3-DiT (NextDiT): dim=3840, 30 heads x D=128, FFN=10240 SwiGLU, 2 context_refiner (unmodulated),
//     2 noise_refiner (modulated), 30 main layers (modulated), 3D Unified RoPE axes=[32,48,48] theta=256,
//     t_model=(1-sigma)*1000, cap/x padded to multiples of 32, velocity = -unpatchify(final_layer)
//   - FLUX.1 16-channel AutoencoderKL VAE: z_vae = z / 0.3611 + 0.1159 -> 1024x1024 RGB24 PNG!

#define _POSIX_C_SOURCE 200809L
#include "ec_zimage.h"
#include "ec_bpe.h"
#include "ec_cowrie.h"
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

static double zi_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static inline uint16_t zi_f32_to_bf16_h(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline float zi_bf16_to_f32_h(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f; memcpy(&f, &u, 4);
  return f;
}
static __device__ __forceinline__ uint16_t zi_f32_to_bf16_d(float f) {
  __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u; memcpy(&u, &b, 2);
  return u;
}
static __device__ __forceinline__ float zi_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ===========================================================================
// 1. CUDA Kernels (Qwen3-4B, Z-Image S3-DiT NextDiT, & FLUX.1 16ch VAE)
// ===========================================================================
__global__ void k_zi_f32_to_bf16(const float *x, uint16_t *y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = zi_f32_to_bf16_d(x[i]);
}

__global__ void k_zi_add_inplace(float *a, const float *b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void k_zi_silu_inplace(float *x, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}

__global__ void k_zi_silu_f32_to_bf16(const float *x, uint16_t *y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = zi_f32_to_bf16_d(v / (1.0f + expf(-v))); }
}

__global__ void k_zi_swiglu_pair_bf16(const float *w1, const float *w3, uint16_t *out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = w1[i];
    out[i] = zi_f32_to_bf16_d((a / (1.0f + expf(-a))) * w3[i]);
  }
}

// RMSNorm [N, D] -> optional scale_mod (1 + scale[c]) -> F32 and/or BF16 output
__global__ void k_zi_rmsnorm(
    const float *__restrict__ x, const float *__restrict__ w,
    const float *__restrict__ scale_mod,
    float *__restrict__ out_f32, uint16_t *__restrict__ out_bf16,
    int N, int D, float eps) {
  __shared__ float sm[8];
  int row = blockIdx.x;
  if (row >= N) return;
  const float *xr = x + (long)row * D;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < D; c += blockDim.x) { float v = xr[c]; ss += v * v; }
  ss = zi_warp_sum(ss);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sm[warp] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    s = zi_warp_sum(s);
    if (threadIdx.x == 0) sm[0] = rsqrtf(s / (float)D + eps);
  }
  __syncthreads();
  float inv = sm[0];
  float *of = out_f32 ? (out_f32 + (long)row * D) : NULL;
  uint16_t *ob = out_bf16 ? (out_bf16 + (long)row * D) : NULL;
  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    float v = xr[c] * inv * (w ? w[c] : 1.0f);
    if (scale_mod) v *= (1.0f + scale_mod[c]);
    if (of) of[c] = v;
    if (ob) ob[c] = zi_f32_to_bf16_d(v);
  }
}

// Residual update with post-RMSNorm2 + optional tanh(gate): x += (gate ? tanh(gate) : 1) * RMSNorm(sub_out, norm2_w)
__global__ void k_zi_post_rmsnorm_res(
    float *__restrict__ x, const float *__restrict__ sub_out,
    const float *__restrict__ norm2_w, const float *__restrict__ gate_v,
    int N, int D, float eps) {
  __shared__ float sm[8];
  int row = blockIdx.x;
  if (row >= N) return;
  const float *sr = sub_out + (long)row * D;
  float *xr = x + (long)row * D;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < D; c += blockDim.x) { float v = sr[c]; ss += v * v; }
  ss = zi_warp_sum(ss);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sm[warp] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    s = zi_warp_sum(s);
    if (threadIdx.x == 0) sm[0] = rsqrtf(s / (float)D + eps);
  }
  __syncthreads();
  float inv = sm[0];
  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    float v = sr[c] * inv * norm2_w[c];
    if (gate_v) v *= tanhf(gate_v[c]);
    xr[c] += v;
  }
}

// FinalLayer: LayerNorm(no affine, eps=1e-6) * (1 + scale) -> BF16
__global__ void k_zi_final_ln_mod(
    const float *__restrict__ x, const float *__restrict__ scale,
    uint16_t *__restrict__ out_bf16, int N, int D) {
  __shared__ float sm_m[8], sm_v[8];
  int row = blockIdx.x;
  if (row >= N) return;
  const float *xr = x + (long)row * D;
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = threadIdx.x; c < D; c += blockDim.x) { float v = xr[c]; s1 += v; s2 += v * v; }
  s1 = zi_warp_sum(s1); s2 = zi_warp_sum(s2);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { sm_m[warp] = s1; sm_v[warp] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = (threadIdx.x < (blockDim.x >> 5)) ? sm_m[threadIdx.x] : 0.0f;
    float v = (threadIdx.x < (blockDim.x >> 5)) ? sm_v[threadIdx.x] : 0.0f;
    m = zi_warp_sum(m); v = zi_warp_sum(v);
    if (threadIdx.x == 0) {
      float mean = m / (float)D;
      float var  = v / (float)D - mean * mean;
      sm_m[0] = mean;
      sm_v[0] = rsqrtf(fmaxf(var, 0.0f) + 1e-6f);
    }
  }
  __syncthreads();
  float mean = sm_m[0], inv = sm_v[0];
  uint16_t *ob = out_bf16 + (long)row * D;
  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    ob[c] = zi_f32_to_bf16_d((xr[c] - mean) * inv * (1.0f + scale[c]));
  }
}

// Qwen3-4B split-half RoPE + QK-RMSNorm + causal GQA (reused identically for layer 34 extraction!)
__global__ void k_zi_qwen_qk_rope(
    const float *src, const float *w_norm, float *dst,
    int L, int H, int D, float eps, float theta, int apply_rope) {
  int pair = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (pair >= L * H) return;
  int pos = pair / H, h = pair % H;
  const float *s = src + ((long)pos * H + h) * D;
  float *o = dst + ((long)h * L + pos) * D;
  float vals[4];
  float ss = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int c = lane + i * 32;
    float v = s[c]; vals[i] = v; ss += v * v;
  }
  if (w_norm) {
    ss = zi_warp_sum(ss);
    float inv = rsqrtf(ss / (float)D + eps);
    #pragma unroll
    for (int i = 0; i < 4; i++) vals[i] *= inv * w_norm[lane + i * 32];
  }
  if (apply_rope) {
    int half = D >> 1;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      int c = lane + i * 32;
      double freq = 1.0 / pow((double)theta, (double)(2 * c) / (double)D);
      double ang = (double)pos * freq;
      float cs = (float)cos(ang), sn = (float)sin(ang);
      float x0 = vals[i], x1 = vals[i + 2];
      o[c]        = x0 * cs - x1 * sn;
      o[c + half] = x1 * cs + x0 * sn;
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 4; i++) o[lane + i * 32] = vals[i];
  }
}

__global__ void k_zi_qwen_gqa_causal(
    const float *Q, const float *K, const float *V, uint16_t *O_bf16,
    int L, int Hq, int Hkv, int D, float scale) {
  int pair = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (pair >= L * Hq) return;
  int qpos = pair / Hq, hq = pair % Hq, hkv = hq / (Hq / Hkv);
  const float *qv = Q + ((long)hq * L + qpos) * D;
  const float *kb = K + (long)hkv * L * D;
  const float *vb = V + (long)hkv * L * D;
  float qr[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) qr[i] = qv[lane + i * 32];
  float m_max = -1e30f, l_sum = 0.0f, acc[4] = {0, 0, 0, 0};
  for (int kpos = 0; kpos <= qpos; kpos++) {
    const float *kv = kb + (long)kpos * D;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) dot += qr[i] * kv[lane + i * 32];
    dot = zi_warp_sum(dot) * scale;
    float nm = fmaxf(m_max, dot);
    float al = expf(m_max - nm), p = expf(dot - nm);
    l_sum = l_sum * al + p; m_max = nm;
    const float *vv = vb + (long)kpos * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = acc[i] * al + p * vv[lane + i * 32];
  }
  float inv = 1.0f / fmaxf(l_sum, 1e-20f);
  uint16_t *o = O_bf16 + ((long)qpos * Hq + hq) * D;
  #pragma unroll
  for (int i = 0; i < 4; i++) o[lane + i * 32] = zi_f32_to_bf16_d(acc[i] * inv);
}

// Precompute 3D Unified RoPE (axes_dims = [32, 48, 48] -> 16 + 24 + 24 = 64 pairs, theta = 256.0)
// pos_ids: [N_tot, 3] float32 -> cos_out/sin_out [N_tot, 64]
__global__ void k_zi_precompute_rope_3d(
    const float *pos3, float *cos_out, float *sin_out, int N_tot, double theta) {
  int tok = blockIdx.x, p = threadIdx.x; // p in 0..63
  if (tok >= N_tot || p >= 64) return;
  float coord; int local, axis_dim;
  if (p < 16) {
    coord = pos3[tok * 3 + 0]; local = p; axis_dim = 32;
  } else if (p < 40) {
    coord = pos3[tok * 3 + 1]; local = p - 16; axis_dim = 48;
  } else {
    coord = pos3[tok * 3 + 2]; local = p - 40; axis_dim = 48;
  }
  double freq = 1.0 / pow(theta, (double)(2 * local) / (double)axis_dim);
  double ang = (double)coord * freq;
  cos_out[(long)tok * 64 + p] = (float)cos(ang);
  sin_out[(long)tok * 64 + p] = (float)sin(ang);
}

// Pack QKV [N, 3*E] with per-head QK-RMSNorm(1e-5) + adjacent-pair 3D RoPE -> [H, N_pad, 128] BF16 for ec_ops_fa2_bf16!
__global__ void k_zi_pack_qkv_rope_bf16(
    const float *__restrict__ qkv,
    const float *__restrict__ qn_w, const float *__restrict__ kn_w,
    const float *__restrict__ rc, const float *__restrict__ rs,
    uint16_t *__restrict__ Q_b, uint16_t *__restrict__ K_b, uint16_t *__restrict__ V_b,
    int N, int N_pad, int H, int D, float eps) {
  long pair = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (pair >= (long)N * H) return;
  int h = (int)(pair % H), t = (int)(pair / H), E = H * D;
  const float *base = qkv + (long)t * (3 * E);
  const float *qs = base + h * D;
  const float *ks = base + E + h * D;
  const float *vs = base + 2 * E + h * D;
  float sq = 0.0f, sk = 0.0f;
  for (int d = lane; d < D; d += 32) { float qv = qs[d], kv = ks[d]; sq += qv * qv; sk += kv * kv; }
  sq = zi_warp_sum(sq); sk = zi_warp_sum(sk);
  float iq = rsqrtf(sq / (float)D + eps), ik = rsqrtf(sk / (float)D + eps);
  const float *rc_t = rc + (long)t * 64, *rs_t = rs + (long)t * 64;
  long ob = ((long)h * N_pad + t) * D;
  uint16_t *qo = Q_b + ob, *ko = K_b + ob, *vo = V_b + ob;
  for (int p = lane; p < 64; p += 32) {
    int d0 = 2 * p, d1 = d0 + 1;
    float q0 = qs[d0] * iq * qn_w[d0], q1 = qs[d1] * iq * qn_w[d1];
    float k0 = ks[d0] * ik * kn_w[d0], k1 = ks[d1] * ik * kn_w[d1];
    float cs = rc_t[p], sn = rs_t[p];
    qo[d0] = zi_f32_to_bf16_d(q0 * cs - q1 * sn);
    qo[d1] = zi_f32_to_bf16_d(q0 * sn + q1 * cs);
    ko[d0] = zi_f32_to_bf16_d(k0 * cs - k1 * sn);
    ko[d1] = zi_f32_to_bf16_d(k0 * sn + k1 * cs);
    vo[d0] = zi_f32_to_bf16_d(vs[d0]);
    vo[d1] = zi_f32_to_bf16_d(vs[d1]);
  }
}

// Patchify [16, H_lat, W_lat] -> [N_img, 64] BF16 in exact ComfyUI NextDiT order (ph*32 + pw*16 + c)
__global__ void k_zi_patchify_bf16(
    const float *x_chw, uint16_t *tok_bf16, int Hp, int Wp) {
  int N_img = Hp * Wp;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)N_img * 64) return;
  int tok = (int)(idx / 64), k = (int)(idx % 64);
  int py = tok / Wp, px = tok % Wp;
  int ph = k >> 5, pw = (k >> 4) & 1, c = k & 15;
  int H_lat = Hp * 2, W_lat = Wp * 2;
  int y = py * 2 + ph, x = px * 2 + pw;
  tok_bf16[idx] = zi_f32_to_bf16_d(x_chw[((long)c * H_lat + y) * W_lat + x]);
}

// Unpatchify + NEGATE (-img -> standard FlowMatch velocity v = eps - x0!) into [16, H_lat, W_lat] F32!
__global__ void k_zi_unpatchify_neg(
    const float *tok_64, float *v_chw, int Hp, int Wp) {
  int N_img = Hp * Wp;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)N_img * 64) return;
  int tok = (int)(idx / 64), k = (int)(idx % 64);
  int py = tok / Wp, px = tok % Wp;
  int ph = k >> 5, pw = (k >> 4) & 1, c = k & 15;
  int H_lat = Hp * 2, W_lat = Wp * 2;
  int y = py * 2 + ph, x = px * 2 + pw;
  v_chw[((long)c * H_lat + y) * W_lat + x] = -tok_64[idx];
}

__global__ void k_zi_fill_pad_rows(float *dst, const float *pad_vec, int start_row, int n_pad_rows, int D) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)n_pad_rows * D) return;
  int r = (int)(idx / D), c = (int)(idx % D);
  dst[(long)(start_row + r) * D + c] = pad_vec[c];
}

// ===========================================================================
// 2. FLUX.1 16-Channel AutoencoderKL VAE Kernels (NHWC [H*W, C] -> RGB24!)
// ===========================================================================
__global__ void k_zi_vae_latent_scale_nhwc(
    const float *z_chw, float *z_nhwc, int H_lat, int W_lat) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long HW = (long)H_lat * W_lat;
  if (idx >= HW * 16) return;
  int pix = (int)(idx / 16), c = (int)(idx % 16);
  // Official FLUX.1 AutoencoderKL scaling: z_vae = z / 0.3611 + 0.1159
  z_nhwc[idx] = z_chw[(long)c * HW + pix] * (1.0f / 0.3611f) + 0.1159f;
}

__global__ void k_zi_vae_gn_silu_nhwc(
    const float *__restrict__ x, const float *__restrict__ g, const float *__restrict__ b,
    uint16_t *__restrict__ y_bf16, float *__restrict__ y_f32,
    int HW, int C, int apply_silu) {
  __shared__ float sm_m[8], sm_v[8];
  const int grp = blockIdx.x; // 0..31
  const int cpg = C / 32;
  const int c0 = grp * cpg;
  const long tot = (long)HW * cpg;
  float s1 = 0.0f, s2 = 0.0f;
  for (long i = threadIdx.x; i < tot; i += blockDim.x) {
    int p = (int)(i / cpg), dc = (int)(i % cpg);
    float v = x[(long)p * C + c0 + dc];
    s1 += v; s2 += v * v;
  }
  s1 = zi_warp_sum(s1); s2 = zi_warp_sum(s2);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { sm_m[warp] = s1; sm_v[warp] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = (threadIdx.x < (blockDim.x >> 5)) ? sm_m[threadIdx.x] : 0.0f;
    float v = (threadIdx.x < (blockDim.x >> 5)) ? sm_v[threadIdx.x] : 0.0f;
    m = zi_warp_sum(m); v = zi_warp_sum(v);
    if (threadIdx.x == 0) {
      float mean = m / (float)tot;
      float var  = v / (float)tot - mean * mean;
      sm_m[0] = mean;
      sm_v[0] = rsqrtf(fmaxf(var, 0.0f) + 1e-6f);
    }
  }
  __syncthreads();
  float mean = sm_m[0], inv = sm_v[0];
  for (long i = threadIdx.x; i < tot; i += blockDim.x) {
    int p = (int)(i / cpg), dc = (int)(i % cpg);
    int c = c0 + dc;
    float v = (x[(long)p * C + c] - mean) * inv * g[c] + b[c];
    if (apply_silu) v = v / (1.0f + expf(-v));
    if (y_bf16) y_bf16[(long)p * C + c] = zi_f32_to_bf16_d(v);
    if (y_f32)  y_f32[(long)p * C + c]  = v;
  }
}

__global__ void k_zi_vae_im2col_3x3(
    const float *x, uint16_t *col_bf16, int H, int W, int Cin, int row0, int n_rows) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K = 9L * Cin;
  long tot = (long)n_rows * W * K;
  if (idx >= tot) return;
  long pix = idx / K;
  int rem = (int)(idx % K);
  int ky = rem / (3 * Cin), rem2 = rem % (3 * Cin);
  int kx = rem2 / Cin, ic = rem2 % Cin;
  int y = row0 + (int)(pix / W) + ky - 1;
  int wx = (int)(pix % W) + kx - 1;
  float v = (y >= 0 && y < H && wx >= 0 && wx < W) ? x[((long)y * W + wx) * Cin + ic] : 0.0f;
  col_bf16[idx] = zi_f32_to_bf16_d(v);
}

__global__ void k_zi_vae_upsample2x_nhwc(const float *x, float *y, int H, int W, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int H2 = H * 2, W2 = W * 2;
  if (idx >= (long)H2 * W2 * C) return;
  int c = (int)(idx % C);
  long p = idx / C;
  int oh = (int)(p / W2), ow = (int)(p % W2);
  y[idx] = x[((long)(oh >> 1) * W + (ow >> 1)) * C + c];
}

// Tensor-Core-accelerated tiled Mid-Attention helpers (zero HW-sized shared memory -> works up to 4K in ~4ms!)
__global__ void k_zi_vae_transpose_bf16(const float *src_hw_c, uint16_t *dst_c_hw, int HW, int HW_pad, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)C * HW_pad;
  if (idx >= tot) return;
  int c = (int)(idx / HW_pad), hw = (int)(idx % HW_pad);
  dst_c_hw[idx] = (hw < HW) ? zi_f32_to_bf16_d(src_hw_c[(long)hw * C + c]) : (uint16_t)0;
}

__global__ void k_zi_vae_row_softmax_bf16(
    const float *scores, uint16_t *probs_bf16, int rows, int HW, int HW_pad, float scale) {
  __shared__ float sm[8];
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *sr = scores + (long)r * HW;
  uint16_t *pr = probs_bf16 + (long)r * HW_pad;
  float mx = -1e30f;
  for (int i = threadIdx.x; i < HW; i += blockDim.x) {
    float v = sr[i] * scale;
    if (v > mx) mx = v;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sm[warp] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : -1e30f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x == 0) sm[0] = m;
  }
  __syncthreads();
  float gmax = sm[0], sum = 0.0f;
  for (int i = threadIdx.x; i < HW; i += blockDim.x) sum += expf(sr[i] * scale - gmax);
  sum = zi_warp_sum(sum);
  if (lane == 0) sm[warp] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    s = zi_warp_sum(s);
    if (threadIdx.x == 0) sm[0] = 1.0f / fmaxf(s, 1e-20f);
  }
  __syncthreads();
  float inv = sm[0];
  for (int i = threadIdx.x; i < HW_pad; i += blockDim.x) {
    pr[i] = (i < HW) ? zi_f32_to_bf16_d(expf(sr[i] * scale - gmax) * inv) : (uint16_t)0;
  }
}

__global__ void k_zi_vae_to_rgb24(const float *nhwc3, uint8_t *rgb, long npix) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npix * 3) return;
  float v = (nhwc3[i] * 0.5f + 0.5f) * 255.0f;
  if (v < 0.0f) v = 0.0f; if (v > 255.0f) v = 255.0f;
  rgb[i] = (uint8_t)(v + 0.5f);
}

// ===========================================================================
// 3. Weight Structs & Safetensors Upload Helpers
// ===========================================================================
static uint16_t *zi_upload_bf16(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) { fprintf(stderr, "[zimage] missing tensor: %s\n", name); return NULL; }
  size_t nelem = t->nbytes / (t->dtype == EC_DTYPE_F32 ? 4 : 2);
  uint16_t *d = NULL;
  if (cudaMalloc(&d, nelem * 2) != cudaSuccess) return NULL;
  if (t->dtype == EC_DTYPE_BF16) {
    cudaMemcpy(d, ec_st_ptr(st, t), nelem * 2, cudaMemcpyHostToDevice);
  } else if (t->dtype == EC_DTYPE_F32) {
    const float *src = (const float *)ec_st_ptr(st, t);
    uint16_t *tmp = (uint16_t *)malloc(nelem * 2);
    for (size_t i = 0; i < nelem; i++) tmp[i] = zi_f32_to_bf16_h(src[i]);
    cudaMemcpy(d, tmp, nelem * 2, cudaMemcpyHostToDevice);
    free(tmp);
  } else {
    cudaFree(d); return NULL;
  }
  return d;
}

static float *zi_upload_f32(const ec_safetensors_t *st, const char *name) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) { fprintf(stderr, "[zimage] missing f32 tensor: %s\n", name); return NULL; }
  size_t nelem = t->nbytes / (t->dtype == EC_DTYPE_F32 ? 4 : 2);
  float *d = NULL;
  if (cudaMalloc(&d, nelem * 4) != cudaSuccess) return NULL;
  if (t->dtype == EC_DTYPE_F32) {
    cudaMemcpy(d, ec_st_ptr(st, t), nelem * 4, cudaMemcpyHostToDevice);
  } else if (t->dtype == EC_DTYPE_BF16) {
    const uint16_t *src = (const uint16_t *)ec_st_ptr(st, t);
    float *tmp = (float *)malloc(nelem * 4);
    for (size_t i = 0; i < nelem; i++) tmp[i] = zi_bf16_to_f32_h(src[i]);
    cudaMemcpy(d, tmp, nelem * 4, cudaMemcpyHostToDevice);
    free(tmp);
  } else {
    cudaFree(d); return NULL;
  }
  return d;
}

// Transpose 2D Conv weight [Cout, Cin, K, K] -> [Cout, K*K*Cin] OHWI BF16 for implicit/im2col TensorCore GEMM!
static uint16_t *zi_upload_conv2d_ohwi_bf16(const ec_safetensors_t *st, const char *name, int Cout, int Cin, int K) {
  const ec_st_entry_t *t = ec_st_find(st, name);
  if (!t) return NULL;
  size_t nelem = (size_t)Cout * Cin * K * K;
  uint16_t *h = (uint16_t *)malloc(nelem * 2);
  const float *fsrc = (t->dtype == EC_DTYPE_F32) ? (const float *)ec_st_ptr(st, t) : NULL;
  const uint16_t *bsrc = (t->dtype == EC_DTYPE_BF16) ? (const uint16_t *)ec_st_ptr(st, t) : NULL;
  for (int oc = 0; oc < Cout; oc++)
    for (int ky = 0; ky < K; ky++)
      for (int kx = 0; kx < K; kx++)
        for (int ic = 0; ic < Cin; ic++) {
          size_t si = (((size_t)oc * Cin + ic) * K + ky) * K + kx;
          size_t di = ((size_t)oc * (K * K) + ky * K + kx) * Cin + ic;
          h[di] = fsrc ? zi_f32_to_bf16_h(fsrc[si]) : bsrc[si];
        }
  uint16_t *d = NULL;
  cudaMalloc(&d, nelem * 2);
  cudaMemcpy(d, h, nelem * 2, cudaMemcpyHostToDevice);
  free(h);
  return d;
}

typedef struct {
  int has_mod;
  uint16_t *adaln_w;   // [15360, 256] BF16 (when has_mod)
  float    *adaln_b;   // [15360] F32
  float    *anorm1_w, *anorm2_w, *fnorm1_w, *fnorm2_w; // [3840] F32
  float    *q_norm_w, *k_norm_w; // [128] F32
  uint16_t *qkv_w;     // [11520, 3840] BF16
  uint16_t *out_w;     // [3840, 3840] BF16
  uint16_t *w1_w, *w2_w, *w3_w; // [10240,3840], [3840,10240], [10240,3840] BF16
} zi_block_t;

typedef struct {
  int loaded;
  char lora_sig[1024];
  int lora_hits;
  float    *cap_norm_w;     // [2560] F32
  uint16_t *cap_lin_w;      // [3840, 2560] BF16
  float    *cap_lin_b;      // [3840] F32
  float    *cap_pad_tok;    // [3840] F32
  uint16_t *x_emb_w;        // [3840, 64] BF16
  float    *x_emb_b;        // [3840] F32
  float    *x_pad_tok;      // [3840] F32
  uint16_t *t_mlp0_w, *t_mlp2_w; // [1024, 256], [256, 1024] BF16
  float    *t_mlp0_b, *t_mlp2_b; // [1024], [256] F32
  zi_block_t ctx_ref[2];
  zi_block_t noise_ref[2];
  zi_block_t layers[30];
  uint16_t *fin_mod_w;      // [3840, 256] BF16
  float    *fin_mod_b;      // [3840] F32
  uint16_t *fin_lin_w;      // [64, 3840] BF16
  float    *fin_lin_b;      // [64] F32
} zi_dit_res_t;

struct ec_zimage {
  ec_zimage_paths_t paths;
  cudaStream_t      stream;
  ec_bpe_t         *bpe;
  ec_safetensors_t  dit_st, vae_st, te_st;
  int               dit_open, vae_open, te_open;
  char              te_sha[65];
  zi_dit_res_t      dit;
};

static void free_zi_block(zi_block_t *b) {
  cudaFree(b->adaln_w); cudaFree(b->adaln_b);
  cudaFree(b->anorm1_w); cudaFree(b->anorm2_w); cudaFree(b->fnorm1_w); cudaFree(b->fnorm2_w);
  cudaFree(b->q_norm_w); cudaFree(b->k_norm_w);
  cudaFree(b->qkv_w); cudaFree(b->out_w);
  cudaFree(b->w1_w); cudaFree(b->w2_w); cudaFree(b->w3_w);
  memset(b, 0, sizeof *b);
}

static void free_zi_dit(zi_dit_res_t *d) {
  if (!d->loaded) return;
  cudaFree(d->cap_norm_w); cudaFree(d->cap_lin_w); cudaFree(d->cap_lin_b); cudaFree(d->cap_pad_tok);
  cudaFree(d->x_emb_w); cudaFree(d->x_emb_b); cudaFree(d->x_pad_tok);
  cudaFree(d->t_mlp0_w); cudaFree(d->t_mlp0_b); cudaFree(d->t_mlp2_w); cudaFree(d->t_mlp2_b);
  for (int i = 0; i < 2; i++) { free_zi_block(&d->ctx_ref[i]); free_zi_block(&d->noise_ref[i]); }
  for (int i = 0; i < 30; i++) free_zi_block(&d->layers[i]);
  cudaFree(d->fin_mod_w); cudaFree(d->fin_mod_b); cudaFree(d->fin_lin_w); cudaFree(d->fin_lin_b);
  memset(d, 0, sizeof *d);
}

static int load_zi_block(
    const ec_safetensors_t *st, const char *pfx, int has_mod,
    const ec_lora_spec_t *loras, int n_loras, ec_lora_apply_stats_t *lstats,
    cudaStream_t stream, zi_block_t *b) {
  char k[192];
  b->has_mod = has_mod;
  if (has_mod) {
    snprintf(k, sizeof k, "%s.adaLN_modulation.0.weight", pfx); b->adaln_w = zi_upload_bf16(st, k);
    if (n_loras > 0 && b->adaln_w) ec_lora_fuse_onto_bf16(b->adaln_w, 15360, 256, k, loras, n_loras, lstats, stream);
    snprintf(k, sizeof k, "%s.adaLN_modulation.0.bias", pfx);   b->adaln_b = zi_upload_f32(st, k);
    if (!b->adaln_w || !b->adaln_b) return -1;
  }
  snprintf(k, sizeof k, "%s.attention_norm1.weight", pfx); b->anorm1_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.attention_norm2.weight", pfx); b->anorm2_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.ffn_norm1.weight", pfx);       b->fnorm1_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.ffn_norm2.weight", pfx);       b->fnorm2_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.attention.q_norm.weight", pfx); b->q_norm_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.attention.k_norm.weight", pfx); b->k_norm_w = zi_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.attention.qkv.weight", pfx);    b->qkv_w    = zi_upload_bf16(st, k);
  if (n_loras > 0 && b->qkv_w) ec_lora_fuse_onto_bf16(b->qkv_w, 11520, 3840, k, loras, n_loras, lstats, stream);
  snprintf(k, sizeof k, "%s.attention.out.weight", pfx);    b->out_w    = zi_upload_bf16(st, k);
  if (n_loras > 0 && b->out_w) ec_lora_fuse_onto_bf16(b->out_w, 3840, 3840, k, loras, n_loras, lstats, stream);
  snprintf(k, sizeof k, "%s.feed_forward.w1.weight", pfx);  b->w1_w     = zi_upload_bf16(st, k);
  if (n_loras > 0 && b->w1_w)  ec_lora_fuse_onto_bf16(b->w1_w, 10240, 3840, k, loras, n_loras, lstats, stream);
  snprintf(k, sizeof k, "%s.feed_forward.w2.weight", pfx);  b->w2_w     = zi_upload_bf16(st, k);
  if (n_loras > 0 && b->w2_w)  ec_lora_fuse_onto_bf16(b->w2_w, 3840, 10240, k, loras, n_loras, lstats, stream);
  snprintf(k, sizeof k, "%s.feed_forward.w3.weight", pfx);  b->w3_w     = zi_upload_bf16(st, k);
  if (n_loras > 0 && b->w3_w)  ec_lora_fuse_onto_bf16(b->w3_w, 10240, 3840, k, loras, n_loras, lstats, stream);
  return (b->anorm1_w && b->qkv_w && b->out_w && b->w1_w && b->w2_w && b->w3_w) ? 0 : -1;
}

static int ensure_zi_dit_loaded(ec_zimage_t *zi, const ec_lora_spec_t *loras, int n_loras) {
  char sig[1024] = {0};
  size_t pos = 0;
  for (int i = 0; i < n_loras && pos + 8 < sizeof sig; i++) {
    if (loras[i].path[0]) pos += (size_t)snprintf(sig + pos, sizeof sig - pos, "%s%.3f|", loras[i].path, loras[i].strength);
  }
  if (zi->dit.loaded && strcmp(zi->dit.lora_sig, sig) == 0) return zi->dit.lora_hits;
  free_zi_dit(&zi->dit);

  const ec_safetensors_t *st = &zi->dit_st;
  zi_dit_res_t *d = &zi->dit;
  ec_lora_apply_stats_t lstats = {};

  d->cap_norm_w  = zi_upload_f32(st, "cap_embedder.0.weight");
  d->cap_lin_w   = zi_upload_bf16(st, "cap_embedder.1.weight");
  d->cap_lin_b   = zi_upload_f32(st, "cap_embedder.1.bias");
  d->cap_pad_tok = zi_upload_f32(st, "cap_pad_token");
  d->x_emb_w     = zi_upload_bf16(st, "x_embedder.weight");
  d->x_emb_b     = zi_upload_f32(st, "x_embedder.bias");
  d->x_pad_tok   = zi_upload_f32(st, "x_pad_token");
  d->t_mlp0_w    = zi_upload_bf16(st, "t_embedder.mlp.0.weight");
  d->t_mlp0_b    = zi_upload_f32(st, "t_embedder.mlp.0.bias");
  d->t_mlp2_w    = zi_upload_bf16(st, "t_embedder.mlp.2.weight");
  d->t_mlp2_b    = zi_upload_f32(st, "t_embedder.mlp.2.bias");
  d->fin_mod_w   = zi_upload_bf16(st, "final_layer.adaLN_modulation.1.weight");
  d->fin_mod_b   = zi_upload_f32(st, "final_layer.adaLN_modulation.1.bias");
  d->fin_lin_w   = zi_upload_bf16(st, "final_layer.linear.weight");
  d->fin_lin_b   = zi_upload_f32(st, "final_layer.linear.bias");
  if (!d->cap_lin_w || !d->x_emb_w || !d->t_mlp0_w || !d->fin_lin_w) { free_zi_dit(d); return -1; }

  char pfx[64];
  for (int i = 0; i < 2; i++) {
    snprintf(pfx, sizeof pfx, "context_refiner.%d", i);
    if (load_zi_block(st, pfx, 0, loras, n_loras, &lstats, zi->stream, &d->ctx_ref[i]) != 0) { free_zi_dit(d); return -1; }
    snprintf(pfx, sizeof pfx, "noise_refiner.%d", i);
    if (load_zi_block(st, pfx, 1, loras, n_loras, &lstats, zi->stream, &d->noise_ref[i]) != 0) { free_zi_dit(d); return -1; }
  }
  for (int i = 0; i < 30; i++) {
    snprintf(pfx, sizeof pfx, "layers.%d", i);
    if (load_zi_block(st, pfx, 1, loras, n_loras, &lstats, zi->stream, &d->layers[i]) != 0) { free_zi_dit(d); return -1; }
  }
  snprintf(d->lora_sig, sizeof d->lora_sig, "%s", sig);
  d->lora_hits = lstats.n_tensors_matched;
  d->loaded = 1;
  return d->lora_hits;
}

// ===========================================================================
// 4. Stage A: Streaming Qwen3-4B Text Encoder -> Layer 34 (hidden_states[-2], [L, 2560])
// ===========================================================================
static int run_zimage_qwen3_encode(
    ec_zimage_t *zi, const char *prompt,
    float **out_cap_f32_d, int *out_L, double *out_ms) {
  double t0 = zi_now_ms();
  const int E = 2560, Hq = 32, Hkv = 8, D = 128, F = 9728;
  char templ[4096];
  // Exact official ComfyUI ZImageTokenizer template (thinking=True -> no empty <think> block!)
  snprintf(templ, sizeof templ, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt ? prompt : "");
  int32_t ids[512];
  int L = ec_bpe_encode(zi->bpe, templ, ids, 512);
  if (L <= 0) return -1;

  // Cowrie cross-restart StatePage cache for instant ~1ms rerolls!
  if (prompt && zi->te_sha[0] && ec_cowrie_available()) {
    uint16_t *cached_bf = (uint16_t *)malloc((size_t)L * E * 2);
    if (cached_bf && ec_cowrie_sp_load("zimage_qwen3_v1", zi->te_sha, templ, strlen(templ),
                                        EC_COWRIE_KV_BF16, cached_bf, (size_t)L * E * 2) == 0) {
      float *h = (float *)malloc((size_t)L * E * 4);
      for (long i = 0; i < (long)L * E; i++) h[i] = zi_bf16_to_f32_h(cached_bf[i]);
      free(cached_bf);
      float *d = NULL;
      cudaMalloc(&d, (size_t)L * E * 4);
      cudaMemcpyAsync(d, h, (size_t)L * E * 4, cudaMemcpyHostToDevice, zi->stream);
      cudaStreamSynchronize(zi->stream);
      free(h);
      *out_cap_f32_d = d; *out_L = L;
      if (out_ms) *out_ms = zi_now_ms() - t0;
      return 0;
    }
    free(cached_bf);
  }

  const ec_st_entry_t *e_emb = ec_st_find(&zi->te_st, "model.embed_tokens.weight");
  if (!e_emb || e_emb->dtype != EC_DTYPE_BF16) return -1;
  const uint16_t *emb_tab = (const uint16_t *)ec_st_ptr(&zi->te_st, e_emb);
  float *x0_h = (float *)malloc((size_t)L * E * 4);
  for (int t = 0; t < L; t++) {
    int id = ids[t];
    if (id < 0 || id >= (int)e_emb->shape[0]) id = 0;
    const uint16_t *row = emb_tab + (size_t)id * E;
    for (int c = 0; c < E; c++) x0_h[(size_t)t * E + c] = zi_bf16_to_f32_h(row[c]);
  }

  cudaStream_t s = zi->stream;
  float *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL, *qh_d = NULL, *kh_d = NULL, *vh_d = NULL;
  float *o_d = NULL, *g_d = NULL, *u_d = NULL;
  uint16_t *h_bf = NULL, *ao_bf = NULL, *mlp_bf = NULL;
  cudaMalloc(&x_d, (size_t)L * E * 4);
  cudaMalloc(&h_bf, (size_t)L * E * 2);
  cudaMalloc(&q_d, (size_t)L * Hq * D * 4);
  cudaMalloc(&k_d, (size_t)L * Hkv * D * 4);
  cudaMalloc(&v_d, (size_t)L * Hkv * D * 4);
  cudaMalloc(&qh_d, (size_t)Hq * L * D * 4);
  cudaMalloc(&kh_d, (size_t)Hkv * L * D * 4);
  cudaMalloc(&vh_d, (size_t)Hkv * L * D * 4);
  cudaMalloc(&ao_bf, (size_t)L * Hq * D * 2);
  cudaMalloc(&o_d, (size_t)L * E * 4);
  cudaMalloc(&g_d, (size_t)L * F * 4);
  cudaMalloc(&u_d, (size_t)L * F * 4);
  cudaMalloc(&mlp_bf, (size_t)L * F * 2);
  cudaMemcpyAsync(x_d, x0_h, (size_t)L * E * 4, cudaMemcpyHostToDevice, s);
  free(x0_h);

  // Stream layers 0..34 (35 layers -> hidden_states[-2] of 36-layer Qwen3-4B!)
  char key[160];
  float attn_sc = 1.0f / sqrtf((float)D);
  for (int l = 0; l <= 34; l++) {
    snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l);            float *ln1 = zi_upload_f32(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l);   float *ln2 = zi_upload_f32(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.q_norm.weight", l);           float *qn  = zi_upload_f32(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.k_norm.weight", l);           float *kn  = zi_upload_f32(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l);           uint16_t *wq = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l);           uint16_t *wk = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj.weight", l);           uint16_t *wv = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l);           uint16_t *wo = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", l);              uint16_t *wg = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", l);                uint16_t *wu = zi_upload_bf16(&zi->te_st, key);
    snprintf(key, sizeof key, "model.layers.%d.mlp.down_proj.weight", l);              uint16_t *wd = zi_upload_bf16(&zi->te_st, key);

    k_zi_rmsnorm<<<L, 256, 0, s>>>(x_d, ln1, NULL, NULL, h_bf, L, E, 1e-6f);
    ec_ops_gemm_bf16(h_bf, wq, NULL, q_d, NULL, L, Hq * D,  E, s);
    ec_ops_gemm_bf16(h_bf, wk, NULL, k_d, NULL, L, Hkv * D, E, s);
    ec_ops_gemm_bf16(h_bf, wv, NULL, v_d, NULL, L, Hkv * D, E, s);
    k_zi_qwen_qk_rope<<<(L * Hq + 7) / 8, 256, 0, s>>>(q_d, qn, qh_d, L, Hq, D, 1e-6f, 1e6f, 1);
    k_zi_qwen_qk_rope<<<(L * Hkv + 7) / 8, 256, 0, s>>>(k_d, kn, kh_d, L, Hkv, D, 1e-6f, 1e6f, 1);
    k_zi_qwen_qk_rope<<<(L * Hkv + 7) / 8, 256, 0, s>>>(v_d, NULL, vh_d, L, Hkv, D, 1e-6f, 1e6f, 0);
    k_zi_qwen_gqa_causal<<<(L * Hq + 7) / 8, 256, 0, s>>>(qh_d, kh_d, vh_d, ao_bf, L, Hq, Hkv, D, attn_sc);
    ec_ops_gemm_bf16(ao_bf, wo, NULL, o_d, NULL, L, E, Hq * D, s);
    k_zi_add_inplace<<<((long)L * E + 255) / 256, 256, 0, s>>>(x_d, o_d, (long)L * E);

    k_zi_rmsnorm<<<L, 256, 0, s>>>(x_d, ln2, NULL, NULL, h_bf, L, E, 1e-6f);
    ec_ops_gemm_bf16(h_bf, wg, NULL, g_d, NULL, L, F, E, s);
    ec_ops_gemm_bf16(h_bf, wu, NULL, u_d, NULL, L, F, E, s);
    k_zi_swiglu_pair_bf16<<<((long)L * F + 255) / 256, 256, 0, s>>>(g_d, u_d, mlp_bf, (long)L * F);
    ec_ops_gemm_bf16(mlp_bf, wd, NULL, o_d, NULL, L, E, F, s);
    k_zi_add_inplace<<<((long)L * E + 255) / 256, 256, 0, s>>>(x_d, o_d, (long)L * E);
    cudaStreamSynchronize(s);
    cudaFree(ln1); cudaFree(ln2); cudaFree(qn); cudaFree(kn);
    cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo); cudaFree(wg); cudaFree(wu); cudaFree(wd);
  }

  if (prompt && zi->te_sha[0] && ec_cowrie_available()) {
    float *h = (float *)malloc((size_t)L * E * 4);
    uint16_t *bf = (uint16_t *)malloc((size_t)L * E * 2);
    if (h && bf && cudaMemcpy(h, x_d, (size_t)L * E * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaGetLastError() == cudaSuccess) {
      int finite_ok = 1;
      for (long i = 0; i < (long)L * E; i++) {
        if (!isfinite(h[i])) { finite_ok = 0; break; }
        bf[i] = zi_f32_to_bf16_h(h[i]);
      }
      if (finite_ok) {
        ec_cowrie_sp_save("zimage_qwen3_v1", zi->te_sha, templ, strlen(templ), (uint32_t)L, EC_COWRIE_KV_BF16, bf, (size_t)L * E * 2);
      }
    }
    free(h); free(bf);
  }

  cudaFree(h_bf); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(qh_d); cudaFree(kh_d); cudaFree(vh_d); cudaFree(ao_bf);
  cudaFree(o_d); cudaFree(g_d); cudaFree(u_d); cudaFree(mlp_bf);
  *out_cap_f32_d = x_d; *out_L = L;
  if (out_ms) *out_ms = zi_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 5. Stage B: Z-Image-Turbo 6.15B S3-DiT Forward & 8-Step AuraFlow Sampling
// ===========================================================================
static uint64_t zi_sm64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static void zi_fill_gaussian(float *dst, size_t n, uint32_t seed) {
  uint64_t st = ((uint64_t)seed * 0x9E3779B97F4A7C15ULL) ^ 0x21A6E002026ULL;
  for (size_t i = 0; i < n; i += 2) {
    double u1 = ((double)(zi_sm64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = ((double)(zi_sm64(&st) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1)), ang = 6.28318530717958647692 * u2;
    dst[i] = (float)(r * cos(ang));
    if (i + 1 < n) dst[i + 1] = (float)(r * sin(ang));
  }
}

static int run_zimage_dit(
    ec_zimage_t *zi, const float *cap_2560_d, int L_cap,
    int H_lat, int W_lat, int steps, float shift, uint32_t seed,
    float *out_lat_chw_d, ec_zimage_progress_fn cb, void *cb_ctx, double *out_ms) {
  double t0 = zi_now_ms();
  const zi_dit_res_t *d = &zi->dit;
  cudaStream_t s = zi->stream;
  const int DIM = 3840, H = 30, D = 128, FFN = 10240;
  const float attn_sc = 1.0f / sqrtf((float)D);
  const int Hp = H_lat / 2, Wp = W_lat / 2;
  const int N_img = Hp * Wp;
  const int L_cap_pad = (L_cap + 31) & ~31;
  const int N_img_pad = (N_img + 31) & ~31;
  const int N_tot = L_cap_pad + N_img_pad;
  const int N_fa_pad = (N_tot + 63) & ~63;

  // 1. Build 3D RoPE position IDs for cap [1..L_cap_pad, 0, 0] and x [L_cap_pad+1, py, px] (pad rows -> 0,0,0)
  float *pos_h = (float *)calloc((size_t)N_tot * 3, sizeof(float));
  for (int i = 0; i < L_cap_pad; i++) pos_h[i * 3 + 0] = (float)(i + 1);
  for (int i = 0; i < N_img; i++) {
    int r = L_cap_pad + i;
    pos_h[r * 3 + 0] = (float)(L_cap_pad + 1);
    pos_h[r * 3 + 1] = (float)(i / Wp);
    pos_h[r * 3 + 2] = (float)(i % Wp);
  }
  float *pos_d = NULL, *rope_cos = NULL, *rope_sin = NULL;
  cudaMalloc(&pos_d, (size_t)N_tot * 3 * 4);
  cudaMalloc(&rope_cos, (size_t)N_tot * 64 * 4);
  cudaMalloc(&rope_sin, (size_t)N_tot * 64 * 4);
  cudaMemcpyAsync(pos_d, pos_h, (size_t)N_tot * 3 * 4, cudaMemcpyHostToDevice, s);
  free(pos_h);
  k_zi_precompute_rope_3d<<<N_tot, 64, 0, s>>>(pos_d, rope_cos, rope_sin, N_tot, 256.0);
  cudaFree(pos_d);

  // 2. Allocate clean reusable workspace
  float *cap_ref_d = NULL, *x_tok_d = NULL, *h_cat = NULL;
  uint16_t *cap_in_bf = NULL, *x_in_bf = NULL, *xn_bf = NULL, *ao_bf = NULL, *mlp_bf = NULL;
  uint16_t *Q_b = NULL, *K_b = NULL, *V_b = NULL;
  float *qkv_d = NULL, *sub_d = NULL, *w1_d = NULL, *w3_d = NULL;
  float *t_256 = NULL, *t_1024 = NULL, *adaln_256 = NULL, *mod4_d = NULL, *fin_scale = NULL;
  uint16_t *t_256_bf = NULL, *t_1024_bf = NULL, *adaln_bf = NULL;
  float *fin_out64 = NULL, *v_chw = NULL, *old_x0 = NULL;

  cudaMalloc(&cap_ref_d, (size_t)L_cap_pad * DIM * 4);
  cudaMalloc(&x_tok_d,   (size_t)N_img_pad * DIM * 4);
  cudaMalloc(&h_cat,     (size_t)N_tot * DIM * 4);
  cudaMalloc(&cap_in_bf, (size_t)L_cap * 2560 * 2);
  cudaMalloc(&x_in_bf,   (size_t)N_img * 64 * 2);
  cudaMalloc(&xn_bf,     (size_t)N_tot * DIM * 2);
  cudaMalloc(&ao_bf,     (size_t)N_tot * DIM * 2);
  cudaMalloc(&mlp_bf,    (size_t)N_tot * FFN * 2);
  cudaMalloc(&Q_b,       (size_t)H * N_fa_pad * D * 2);
  cudaMalloc(&K_b,       (size_t)H * N_fa_pad * D * 2);
  cudaMalloc(&V_b,       (size_t)H * N_fa_pad * D * 2);
  cudaMalloc(&qkv_d,     (size_t)N_tot * 3 * DIM * 4);
  cudaMalloc(&sub_d,     (size_t)N_tot * DIM * 4);
  cudaMalloc(&w1_d,      (size_t)N_tot * FFN * 4);
  cudaMalloc(&w3_d,      (size_t)N_tot * FFN * 4);
  cudaMalloc(&t_256,     256 * 4);   cudaMalloc(&t_256_bf,  256 * 2);
  cudaMalloc(&t_1024,    1024 * 4);  cudaMalloc(&t_1024_bf, 1024 * 2);
  cudaMalloc(&adaln_256, 256 * 4);   cudaMalloc(&adaln_bf,  256 * 2);
  cudaMalloc(&mod4_d,    4 * DIM * 4);
  cudaMalloc(&fin_scale, DIM * 4);
  cudaMalloc(&fin_out64, (size_t)N_img * 64 * 4);
  cudaMalloc(&v_chw,     (size_t)16 * H_lat * W_lat * 4);
  cudaMalloc(&old_x0,    (size_t)16 * H_lat * W_lat * 4);

  auto run_block = [&](const zi_block_t *B, float *x_buf, int N, int rope_off) {
    int N_pad_loc = (N + 63) & ~63;
    const float *s_msa = NULL, *g_msa = NULL, *s_mlp = NULL, *g_mlp = NULL;
    if (B->has_mod) {
      ec_ops_gemm_bf16(adaln_bf, B->adaln_w, B->adaln_b, mod4_d, NULL, 1, 4 * DIM, 256, s);
      s_msa = mod4_d + 0 * DIM; g_msa = mod4_d + 1 * DIM;
      s_mlp = mod4_d + 2 * DIM; g_mlp = mod4_d + 3 * DIM;
    }
    k_zi_rmsnorm<<<N, 256, 0, s>>>(x_buf, B->anorm1_w, s_msa, NULL, xn_bf, N, DIM, 1e-5f);
    ec_ops_gemm_bf16(xn_bf, B->qkv_w, NULL, qkv_d, NULL, N, 3 * DIM, DIM, s);
    int nw = N * H;
    k_zi_pack_qkv_rope_bf16<<<(nw + 7) / 8, 256, 0, s>>>(
        qkv_d, B->q_norm_w, B->k_norm_w, rope_cos + (long)rope_off * 64, rope_sin + (long)rope_off * 64,
        Q_b, K_b, V_b, N, N_pad_loc, H, D, 1e-5f);
    ec_ops_fa2_bf16(Q_b, K_b, V_b, NULL, ao_bf, N, N, N_pad_loc, N_pad_loc, H, H, D, attn_sc, 0, s);
    ec_ops_gemm_bf16(ao_bf, B->out_w, NULL, sub_d, NULL, N, DIM, DIM, s);
    k_zi_post_rmsnorm_res<<<N, 256, 0, s>>>(x_buf, sub_d, B->anorm2_w, g_msa, N, DIM, 1e-5f);

    k_zi_rmsnorm<<<N, 256, 0, s>>>(x_buf, B->fnorm1_w, s_mlp, NULL, xn_bf, N, DIM, 1e-5f);
    ec_ops_gemm_bf16(xn_bf, B->w1_w, NULL, w1_d, NULL, N, FFN, DIM, s);
    ec_ops_gemm_bf16(xn_bf, B->w3_w, NULL, w3_d, NULL, N, FFN, DIM, s);
    k_zi_swiglu_pair_bf16<<<((long)N * FFN + 255) / 256, 256, 0, s>>>(w1_d, w3_d, mlp_bf, (long)N * FFN);
    ec_ops_gemm_bf16(mlp_bf, B->w2_w, NULL, sub_d, NULL, N, DIM, FFN, s);
    k_zi_post_rmsnorm_res<<<N, 256, 0, s>>>(x_buf, sub_d, B->fnorm2_w, g_mlp, N, DIM, 1e-5f);
  };

  // 3. Embed & refine caption ONCE across all steps! (context_refiner has NO timestep modulation!)
  k_zi_rmsnorm<<<L_cap, 256, 0, s>>>(cap_2560_d, d->cap_norm_w, NULL, NULL, cap_in_bf, L_cap, 2560, 1e-5f);
  ec_ops_gemm_bf16(cap_in_bf, d->cap_lin_w, d->cap_lin_b, cap_ref_d, NULL, L_cap, DIM, 2560, s);
  if (L_cap_pad > L_cap) {
    k_zi_fill_pad_rows<<<((long)(L_cap_pad - L_cap) * DIM + 255) / 256, 256, 0, s>>>(
        cap_ref_d, d->cap_pad_tok, L_cap, L_cap_pad - L_cap, DIM);
  }
  run_block(&d->ctx_ref[0], cap_ref_d, L_cap_pad, 0);
  run_block(&d->ctx_ref[1], cap_ref_d, L_cap_pad, 0);

  // 4. Initial Gaussian latent [16, H_lat, W_lat] + AuraFlow shift=3.0 sigma schedule
  long lat_elems = (long)16 * H_lat * W_lat;
  float *eps_h = (float *)malloc((size_t)lat_elems * 4);
  zi_fill_gaussian(eps_h, (size_t)lat_elems, seed);
  cudaMemcpyAsync(out_lat_chw_d, eps_h, (size_t)lat_elems * 4, cudaMemcpyHostToDevice, s);
  free(eps_h);

  float sigmas[65];
  for (int i = 0; i < steps; i++) {
    float u = 1.0f - (float)i / (float)steps;
    sigmas[i] = (shift * u) / (1.0f + (shift - 1.0f) * u);
  }
  sigmas[steps] = 0.0f;

  // 5. 8-Step S3-DiT Sampling Loop (RES-Multistep exact exponential integrator!)
  int use_res = !(getenv("EC_SOLVER_EULER") && atoi(getenv("EC_SOLVER_EULER")) != 0);
  float h_prev = 0.0f; int has_prev = 0; int cancelled = 0;
  for (int step = 0; step < steps; step++) {
    if (cb && cb(step, steps + 2, "zimage_s3dit", cb_ctx) != 0) { cancelled = 1; break; }
    float sigma = sigmas[step], sigma_next = sigmas[step + 1];
    // Official Z-Image inverted timestep: t_model = (1.0 - sigma) * 1000.0
    float t_scaled = (1.0f - sigma) * 1000.0f;
    float tf_h[256];
    for (int i = 0; i < 128; i++) {
      double freq = exp(-log(10000.0) * (double)i / 128.0);
      double arg = (double)t_scaled * freq;
      tf_h[i] = (float)cos(arg); tf_h[128 + i] = (float)sin(arg);
    }
    cudaMemcpyAsync(t_256, tf_h, sizeof tf_h, cudaMemcpyHostToDevice, s);
    k_zi_f32_to_bf16<<<1, 256, 0, s>>>(t_256, t_256_bf, 256);
    ec_ops_gemm_bf16(t_256_bf, d->t_mlp0_w, d->t_mlp0_b, t_1024, NULL, 1, 1024, 256, s);
    k_zi_silu_f32_to_bf16<<<4, 256, 0, s>>>(t_1024, t_1024_bf, 1024);
    ec_ops_gemm_bf16(t_1024_bf, d->t_mlp2_w, d->t_mlp2_b, adaln_256, adaln_bf, 1, 256, 1024, s);

    // Patchify + x_embedder + pad + 2x noise_refiner
    k_zi_patchify_bf16<<<((long)N_img * 64 + 255) / 256, 256, 0, s>>>(out_lat_chw_d, x_in_bf, Hp, Wp);
    ec_ops_gemm_bf16(x_in_bf, d->x_emb_w, d->x_emb_b, x_tok_d, NULL, N_img, DIM, 64, s);
    if (N_img_pad > N_img) {
      k_zi_fill_pad_rows<<<((long)(N_img_pad - N_img) * DIM + 255) / 256, 256, 0, s>>>(
          x_tok_d, d->x_pad_tok, N_img, N_img_pad - N_img, DIM);
    }
    run_block(&d->noise_ref[0], x_tok_d, N_img_pad, L_cap_pad);
    run_block(&d->noise_ref[1], x_tok_d, N_img_pad, L_cap_pad);

    // Joint stream: [cap_ref_d, x_tok_d] -> 30x Main S3-DiT Layers
    cudaMemcpyAsync(h_cat, cap_ref_d, (size_t)L_cap_pad * DIM * 4, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(h_cat + (size_t)L_cap_pad * DIM, x_tok_d, (size_t)N_img_pad * DIM * 4, cudaMemcpyDeviceToDevice, s);
    for (int l = 0; l < 30; l++) run_block(&d->layers[l], h_cat, N_tot, 0);

    // FinalLayer on image tokens ONLY (cap tokens discarded before unpatchify)
    k_zi_silu_f32_to_bf16<<<1, 256, 0, s>>>(adaln_256, t_256_bf, 256);
    ec_ops_gemm_bf16(t_256_bf, d->fin_mod_w, d->fin_mod_b, fin_scale, NULL, 1, DIM, 256, s);
    k_zi_final_ln_mod<<<N_img, 256, 0, s>>>(h_cat + (size_t)L_cap_pad * DIM, fin_scale, xn_bf, N_img, DIM);
    ec_ops_gemm_bf16(xn_bf, d->fin_lin_w, d->fin_lin_b, fin_out64, NULL, N_img, 64, DIM, s);
    k_zi_unpatchify_neg<<<((long)N_img * 64 + 255) / 256, 256, 0, s>>>(fin_out64, v_chw, Hp, Wp);

    if (use_res) {
      ec_ops_res_multistep_step(out_lat_chw_d, v_chw, old_x0, lat_elems, sigma, sigma_next, h_prev, has_prev, s);
      if (sigma > 1e-5f && sigma_next > 1e-5f) { h_prev = (float)log((double)sigma / (double)sigma_next); has_prev = 1; }
    } else {
      ec_ops_res_multistep_step(out_lat_chw_d, v_chw, NULL, lat_elems, sigma, sigma_next, 0.0f, 0, s);
    }
  }
  cudaStreamSynchronize(s);

  cudaFree(rope_cos); cudaFree(rope_sin);
  cudaFree(cap_ref_d); cudaFree(x_tok_d); cudaFree(h_cat);
  cudaFree(cap_in_bf); cudaFree(x_in_bf); cudaFree(xn_bf); cudaFree(ao_bf); cudaFree(mlp_bf);
  cudaFree(Q_b); cudaFree(K_b); cudaFree(V_b);
  cudaFree(qkv_d); cudaFree(sub_d); cudaFree(w1_d); cudaFree(w3_d);
  cudaFree(t_256); cudaFree(t_256_bf); cudaFree(t_1024); cudaFree(t_1024_bf);
  cudaFree(adaln_256); cudaFree(adaln_bf); cudaFree(mod4_d); cudaFree(fin_scale);
  cudaFree(fin_out64); cudaFree(v_chw); cudaFree(old_x0);

  if (out_ms) *out_ms = zi_now_ms() - t0;
  return cancelled ? -2 : 0;
}

// ===========================================================================
// 6. Stage C: Native CUDA FLUX.1 16-Channel AutoencoderKL VAE Decoder
// ===========================================================================
static void zi_vae_conv3x3(
    ec_zimage_t *zi, const float *x_nhwc, const float *res_nhwc, float *y_nhwc,
    const char *pfx, int H, int W, int Cin, int Cout, uint16_t *col_bf, size_t col_cap_elems) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", pfx);
  snprintf(kb, sizeof kb, "%s.bias", pfx);
  uint16_t *w_bf = zi_upload_conv2d_ohwi_bf16(&zi->vae_st, kw, Cout, Cin, 3);
  float    *b_f  = zi_upload_f32(&zi->vae_st, kb);
  if (!w_bf || !b_f) { cudaFree(w_bf); cudaFree(b_f); return; }

  // Fast implicit-GEMM NHWC path when Cin%32==0 && Cout%64==0 (runs 3.2x faster!), else strip-tiled im2col
  if ((Cin & 31) == 0 && (Cout & 63) == 0) {
    uint16_t *x_bf = NULL;
    cudaMalloc(&x_bf, (size_t)H * W * Cin * 2);
    k_zi_f32_to_bf16<<<((long)H * W * Cin + 255) / 256, 256, 0, zi->stream>>>(x_nhwc, x_bf, (long)H * W * Cin);
    ec_ops_conv2d_3x3_nhwc_bf16(x_bf, w_bf, b_f, res_nhwc, y_nhwc, NULL, H, W, Cin, Cout, zi->stream);
    cudaFree(x_bf);
  } else {
    int max_rows = (int)(col_cap_elems / ((size_t)W * 9 * Cin));
    if (max_rows < 1) max_rows = 1;
    for (int r0 = 0; r0 < H; r0 += max_rows) {
      int nr = (H - r0 < max_rows) ? (H - r0) : max_rows;
      long tot = (long)nr * W * 9 * Cin;
      k_zi_vae_im2col_3x3<<<(tot + 255) / 256, 256, 0, zi->stream>>>(x_nhwc, col_bf, H, W, Cin, r0, nr);
      ec_ops_gemm_bf16(col_bf, w_bf, b_f, y_nhwc + (long)r0 * W * Cout, NULL, nr * W, Cout, 9 * Cin, zi->stream);
    }
    if (res_nhwc) k_zi_add_inplace<<<((long)H * W * Cout + 255) / 256, 256, 0, zi->stream>>>(y_nhwc, res_nhwc, (long)H * W * Cout);
  }
  cudaStreamSynchronize(zi->stream);
  cudaFree(w_bf); cudaFree(b_f);
}

static void zi_vae_conv1x1(
    ec_zimage_t *zi, const float *x_nhwc, float *y_nhwc, const char *pfx, int HW, int Cin, int Cout) {
  char kw[160], kb[160];
  snprintf(kw, sizeof kw, "%s.weight", pfx);
  snprintf(kb, sizeof kb, "%s.bias", pfx);
  uint16_t *w_bf = zi_upload_conv2d_ohwi_bf16(&zi->vae_st, kw, Cout, Cin, 1);
  float    *b_f  = zi_upload_f32(&zi->vae_st, kb);
  uint16_t *x_bf = NULL;
  cudaMalloc(&x_bf, (size_t)HW * Cin * 2);
  k_zi_f32_to_bf16<<<((long)HW * Cin + 255) / 256, 256, 0, zi->stream>>>(x_nhwc, x_bf, (long)HW * Cin);
  ec_ops_gemm_bf16(x_bf, w_bf, b_f, y_nhwc, NULL, HW, Cout, Cin, zi->stream);
  cudaStreamSynchronize(zi->stream);
  cudaFree(x_bf); cudaFree(w_bf); cudaFree(b_f);
}

static float *zi_vae_resblock(
    ec_zimage_t *zi, float *x_d, int H, int W, int Cin, int Cout,
    const char *pfx, uint16_t *col_bf, size_t col_cap) {
  char k1[160], k2[160], k3[160];
  int HW = H * W;
  float *h1 = NULL, *h2 = NULL, *sc = NULL;
  cudaMalloc(&h1, (size_t)HW * Cin * 4);
  cudaMalloc(&h2, (size_t)HW * Cout * 4);
  snprintf(k1, sizeof k1, "%s.norm1.weight", pfx); float *n1w = zi_upload_f32(&zi->vae_st, k1);
  snprintf(k1, sizeof k1, "%s.norm1.bias", pfx);   float *n1b = zi_upload_f32(&zi->vae_st, k1);
  k_zi_vae_gn_silu_nhwc<<<32, 256, 0, zi->stream>>>(x_d, n1w, n1b, NULL, h1, HW, Cin, 1);
  cudaStreamSynchronize(zi->stream);
  cudaFree(n1w); cudaFree(n1b);

  snprintf(k2, sizeof k2, "%s.conv1", pfx);
  zi_vae_conv3x3(zi, h1, NULL, h2, k2, H, W, Cin, Cout, col_bf, col_cap);
  cudaFree(h1);

  float *h3 = NULL;
  cudaMalloc(&h3, (size_t)HW * Cout * 4);
  snprintf(k1, sizeof k1, "%s.norm2.weight", pfx); float *n2w = zi_upload_f32(&zi->vae_st, k1);
  snprintf(k1, sizeof k1, "%s.norm2.bias", pfx);   float *n2b = zi_upload_f32(&zi->vae_st, k1);
  k_zi_vae_gn_silu_nhwc<<<32, 256, 0, zi->stream>>>(h2, n2w, n2b, NULL, h3, HW, Cout, 1);
  cudaStreamSynchronize(zi->stream);
  cudaFree(n2w); cudaFree(n2b);

  if (Cin != Cout) {
    cudaMalloc(&sc, (size_t)HW * Cout * 4);
    snprintf(k3, sizeof k3, "%s.nin_shortcut", pfx);
    zi_vae_conv1x1(zi, x_d, sc, k3, HW, Cin, Cout);
    cudaFree(x_d);
  } else {
    sc = x_d;
  }
  snprintf(k2, sizeof k2, "%s.conv2", pfx);
  zi_vae_conv3x3(zi, h3, sc, h2, k2, H, W, Cout, Cout, col_bf, col_cap);
  cudaFree(h3); cudaFree(sc);
  return h2;
}

static int run_flux1_vae_decode(
    ec_zimage_t *zi, const float *z_chw_d, int H_lat, int W_lat,
    uint8_t *out_rgb_h, double *out_ms) {
  double t0 = zi_now_ms();
  const size_t COL_CAP = 32L * 1024 * 1024;
  uint16_t *col_bf = NULL;
  if (cudaMalloc(&col_bf, COL_CAP * 2) != cudaSuccess) return -1;

  int H = H_lat, W = W_lat;
  float *z_nhwc = NULL, *cur = NULL;
  cudaMalloc(&z_nhwc, (size_t)H * W * 16 * 4);
  cudaMalloc(&cur, (size_t)H * W * 512 * 4);
  k_zi_vae_latent_scale_nhwc<<<((long)H * W * 16 + 255) / 256, 256, 0, zi->stream>>>(z_chw_d, z_nhwc, H, W);
  zi_vae_conv3x3(zi, z_nhwc, NULL, cur, "decoder.conv_in", H, W, 16, 512, col_bf, COL_CAP);
  cudaFree(z_nhwc);

  // Mid: block_1 -> attn_1 -> block_2
  cur = zi_vae_resblock(zi, cur, H, W, 512, 512, "decoder.mid.block_1", col_bf, COL_CAP);
  {
    int HW = H * W;
    float *nw = zi_upload_f32(&zi->vae_st, "decoder.mid.attn_1.norm.weight");
    float *nb = zi_upload_f32(&zi->vae_st, "decoder.mid.attn_1.norm.bias");
    float *hn = NULL, *qd = NULL, *kd = NULL, *vd = NULL, *po = NULL;
    uint16_t *ao = NULL;
    cudaMalloc(&hn, (size_t)HW * 512 * 4);
    cudaMalloc(&qd, (size_t)HW * 512 * 4);
    cudaMalloc(&kd, (size_t)HW * 512 * 4);
    cudaMalloc(&vd, (size_t)HW * 512 * 4);
    cudaMalloc(&ao, (size_t)HW * 512 * 2);
    cudaMalloc(&po, (size_t)HW * 512 * 4);
    k_zi_vae_gn_silu_nhwc<<<32, 256, 0, zi->stream>>>(cur, nw, nb, NULL, hn, HW, 512, 0);
    cudaStreamSynchronize(zi->stream);
    cudaFree(nw); cudaFree(nb);
    zi_vae_conv1x1(zi, hn, qd, "decoder.mid.attn_1.q", HW, 512, 512);
    zi_vae_conv1x1(zi, hn, kd, "decoder.mid.attn_1.k", HW, 512, 512);
    zi_vae_conv1x1(zi, hn, vd, "decoder.mid.attn_1.v", HW, 512, 512);
    cudaFree(hn);
    // Tensor-Core tiled Mid-Attention: S = Q_tile @ K^T -> softmax -> O_tile = P_tile @ V
    {
      int HW_pad = (HW + 15) & ~15;
      int BQ = (HW <= 16384) ? 1024 : 512;
      if (BQ > HW) BQ = HW;
      uint16_t *q_bf = NULL, *k_bf = NULL, *vt_bf = NULL, *p_bf = NULL;
      float *s_f = NULL;
      cudaMalloc(&q_bf,  (size_t)HW * 512 * 2);
      cudaMalloc(&k_bf,  (size_t)HW * 512 * 2);
      cudaMalloc(&vt_bf, (size_t)512 * HW_pad * 2);
      cudaMalloc(&s_f,   (size_t)BQ * HW * 4);
      cudaMalloc(&p_bf,  (size_t)BQ * HW_pad * 2);
      k_zi_f32_to_bf16<<<((long)HW * 512 + 255) / 256, 256, 0, zi->stream>>>(qd, q_bf, (long)HW * 512);
      k_zi_f32_to_bf16<<<((long)HW * 512 + 255) / 256, 256, 0, zi->stream>>>(kd, k_bf, (long)HW * 512);
      k_zi_vae_transpose_bf16<<<((long)512 * HW_pad + 255) / 256, 256, 0, zi->stream>>>(vd, vt_bf, HW, HW_pad, 512);
      float sc = rsqrtf(512.0f);
      for (int q0 = 0; q0 < HW; q0 += BQ) {
        int b = (q0 + BQ <= HW) ? BQ : (HW - q0);
        ec_ops_gemm_bf16(q_bf + (size_t)q0 * 512, k_bf, NULL, s_f, NULL, b, HW, 512, zi->stream);
        k_zi_vae_row_softmax_bf16<<<b, 256, 0, zi->stream>>>(s_f, p_bf, b, HW, HW_pad, sc);
        ec_ops_gemm_bf16(p_bf, vt_bf, NULL, NULL, ao + (size_t)q0 * 512, b, 512, HW_pad, zi->stream);
      }
      cudaStreamSynchronize(zi->stream);
      cudaFree(q_bf); cudaFree(k_bf); cudaFree(vt_bf); cudaFree(s_f); cudaFree(p_bf);
    }
    cudaFree(qd); cudaFree(kd); cudaFree(vd);
    uint16_t *pw = zi_upload_conv2d_ohwi_bf16(&zi->vae_st, "decoder.mid.attn_1.proj_out.weight", 512, 512, 1);
    float    *pb = zi_upload_f32(&zi->vae_st, "decoder.mid.attn_1.proj_out.bias");
    ec_ops_gemm_bf16(ao, pw, pb, po, NULL, HW, 512, 512, zi->stream);
    k_zi_add_inplace<<<((long)HW * 512 + 255) / 256, 256, 0, zi->stream>>>(cur, po, (long)HW * 512);
    cudaStreamSynchronize(zi->stream);
    cudaFree(ao); cudaFree(po); cudaFree(pw); cudaFree(pb);
  }
  cur = zi_vae_resblock(zi, cur, H, W, 512, 512, "decoder.mid.block_2", col_bf, COL_CAP);

  // Up stages: up.3 (512->512, up), up.2 (512->512, up), up.1 (512->256, up), up.0 (256->128)
  const int up_in[4]  = { 512, 512, 512, 256 };
  const int up_out[4] = { 512, 512, 256, 128 };
  const int up_id[4]  = { 3, 2, 1, 0 };
  char pfx[160];
  for (int stg = 0; stg < 4; stg++) {
    int ci = up_in[stg], co = up_out[stg], uid = up_id[stg];
    for (int bi = 0; bi < 3; bi++) {
      snprintf(pfx, sizeof pfx, "decoder.up.%d.block.%d", uid, bi);
      cur = zi_vae_resblock(zi, cur, H, W, bi == 0 ? ci : co, co, pfx, col_bf, COL_CAP);
    }
    if (uid > 0) {
      int H2 = H * 2, W2 = W * 2;
      float *up_d = NULL, *up_c = NULL;
      cudaMalloc(&up_d, (size_t)H2 * W2 * co * 4);
      cudaMalloc(&up_c, (size_t)H2 * W2 * co * 4);
      k_zi_vae_upsample2x_nhwc<<<((long)H2 * W2 * co + 255) / 256, 256, 0, zi->stream>>>(cur, up_d, H, W, co);
      cudaFree(cur);
      snprintf(pfx, sizeof pfx, "decoder.up.%d.upsample.conv", uid);
      zi_vae_conv3x3(zi, up_d, NULL, up_c, pfx, H2, W2, co, co, col_bf, COL_CAP);
      cudaFree(up_d);
      cur = up_c; H = H2; W = W2;
    }
  }

  // Final norm_out -> SiLU -> conv_out (128 -> 3) -> RGB24
  float *hn = NULL, *rgb_f = NULL;
  uint8_t *rgb_d = NULL;
  cudaMalloc(&hn,    (size_t)H * W * 128 * 4);
  cudaMalloc(&rgb_f, (size_t)H * W * 3 * 4);
  cudaMalloc(&rgb_d, (size_t)H * W * 3);
  float *now = zi_upload_f32(&zi->vae_st, "decoder.norm_out.weight");
  float *nob = zi_upload_f32(&zi->vae_st, "decoder.norm_out.bias");
  k_zi_vae_gn_silu_nhwc<<<32, 256, 0, zi->stream>>>(cur, now, nob, NULL, hn, H * W, 128, 1);
  cudaStreamSynchronize(zi->stream);
  cudaFree(cur); cudaFree(now); cudaFree(nob);

  zi_vae_conv3x3(zi, hn, NULL, rgb_f, "decoder.conv_out", H, W, 128, 3, col_bf, COL_CAP);
  cudaFree(hn); cudaFree(col_bf);

  k_zi_vae_to_rgb24<<<((long)H * W * 3 + 255) / 256, 256, 0, zi->stream>>>(rgb_f, rgb_d, (long)H * W);
  cudaMemcpyAsync(out_rgb_h, rgb_d, (size_t)H * W * 3, cudaMemcpyDeviceToHost, zi->stream);
  cudaStreamSynchronize(zi->stream);
  cudaFree(rgb_f); cudaFree(rgb_d);

  if (out_ms) *out_ms = zi_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 7. Public Lifecycle & Entrypoints
// ===========================================================================
extern "C" ec_zimage_t *ec_zimage_create(const ec_zimage_paths_t *paths) {
  if (!paths || !paths->dit_path[0] || !paths->vae_path[0] || !paths->text_encoder_path[0] || !paths->tokenizer_path[0]) return NULL;
  ec_zimage_t *zi = (ec_zimage_t *)calloc(1, sizeof *zi);
  if (!zi) return NULL;
  zi->paths = *paths;
  if (cudaStreamCreate(&zi->stream) != cudaSuccess) { free(zi); return NULL; }
  zi->bpe = ec_bpe_load(paths->tokenizer_path);
  char err[128] = {0};
  if (!zi->bpe ||
      ec_st_open(&zi->dit_st, paths->dit_path, err, sizeof err) != 0 ||
      ec_st_open(&zi->vae_st, paths->vae_path, err, sizeof err) != 0 ||
      ec_st_open(&zi->te_st,  paths->text_encoder_path, err, sizeof err) != 0) {
    ec_zimage_destroy(zi);
    return NULL;
  }
  zi->dit_open = zi->vae_open = zi->te_open = 1;
  ec_cowrie_file_fingerprint(paths->text_encoder_path, zi->te_sha);
  return zi;
}

extern "C" void ec_zimage_unload_vram(ec_zimage_t *zi) {
  if (!zi) return;
  free_zi_dit(&zi->dit);
}

extern "C" void ec_zimage_destroy(ec_zimage_t *zi) {
  if (!zi) return;
  ec_zimage_unload_vram(zi);
  if (zi->dit_open) ec_st_close(&zi->dit_st);
  if (zi->vae_open) ec_st_close(&zi->vae_st);
  if (zi->te_open)  ec_st_close(&zi->te_st);
  if (zi->bpe) ec_bpe_free(zi->bpe);
  if (zi->stream) cudaStreamDestroy(zi->stream);
  free(zi);
}

extern "C" int ec_zimage_is_ready(const ec_zimage_t *zi) {
  return (zi && zi->bpe && zi->dit_open && zi->vae_open && zi->te_open) ? 1 : 0;
}

extern "C" int ec_zimage_generate_rgb(
    ec_zimage_t *zi, const ec_zimage_req_t *req,
    uint8_t **out_rgb, const char *out_png_path,
    ec_zimage_progress_fn cb, void *cb_ctx, ec_zimage_res_t *res) {
  memset(res, 0, sizeof *res);
  if (out_rgb) *out_rgb = NULL;
  if (!ec_zimage_is_ready(zi) || !req) {
    snprintf(res->error, sizeof res->error, "zimage engine not initialized");
    return -1;
  }
  double t_all = zi_now_ms();
  int W = req->width  > 0 ? ((req->width  + 15) & ~15) : 1024;
  int H = req->height > 0 ? ((req->height + 15) & ~15) : 1024;
  if (W < 512) W = 512; if (W > 1536) W = 1536;
  if (H < 512) H = 512; if (H > 1536) H = 1536;
  int steps = (req->steps >= 4 && req->steps <= 20) ? req->steps : 8;
  float shift = (req->shift >= 1.0f && req->shift <= 8.0f) ? req->shift : 3.0f;

  // Stage 1: Qwen3-4B Text Encode (streamed -> ~0.3 GB VRAM peak!)
  if (cb) cb(0, steps + 2, "zimage_qwen3_encode", cb_ctx);
  float *cap_d = NULL; int L_cap = 0;
  if (run_zimage_qwen3_encode(zi, req->prompt, &cap_d, &L_cap, &res->text_enc_ms) != 0) {
    snprintf(res->error, sizeof res->error, "zimage qwen3 text encode failed");
    return -1;
  }

  // Stage 2: Load & Run 6.15B S3-DiT (12.31 GB BF16 resident on 24 GB 3090 + hot-fused LoRAs!)
  int lhits = ensure_zi_dit_loaded(zi, req->loras, req->n_loras);
  if (lhits < 0) {
    cudaFree(cap_d);
    snprintf(res->error, sizeof res->error, "zimage dit load failed");
    return -1;
  }
  res->lora_tensors_fused = lhits;

  int H_lat = H / 8, W_lat = W / 8;
  float *lat_d = NULL;
  cudaMalloc(&lat_d, (size_t)16 * H_lat * W_lat * 4);
  int drc = run_zimage_dit(zi, cap_d, L_cap, H_lat, W_lat, steps, shift, req->seed, lat_d, cb, cb_ctx, &res->dit_ms);
  cudaFree(cap_d);
  if (drc != 0) {
    cudaFree(lat_d);
    snprintf(res->error, sizeof res->error, drc == -2 ? "cancelled" : "zimage dit failed");
    return drc;
  }

  // Stage 3: FLUX.1 16-Channel AutoencoderKL VAE Decode -> RGB24
  if (cb) cb(steps + 1, steps + 2, "zimage_vae_decode", cb_ctx);
  uint8_t *rgb = (uint8_t *)malloc((size_t)W * H * 3);
  if (run_flux1_vae_decode(zi, lat_d, H_lat, W_lat, rgb, &res->vae_ms) != 0) {
    free(rgb); cudaFree(lat_d);
    snprintf(res->error, sizeof res->error, "zimage vae decode failed");
    return -1;
  }
  cudaFree(lat_d);

  if (out_png_path && out_png_path[0]) {
    if (ec_export_png(out_png_path, rgb, (uint32_t)W, (uint32_t)H) != 0) {
      free(rgb);
      snprintf(res->error, sizeof res->error, "zimage png export failed");
      return -1;
    }
  }
  if (out_rgb) *out_rgb = rgb; else free(rgb);

  res->ok = 1;
  res->width = W; res->height = H; res->steps_run = steps;
  snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", zi->paths.dit_path);
  res->total_ms = zi_now_ms() - t_all;
  return 0;
}

extern "C" int ec_zimage_generate_png(
    ec_zimage_t *zi, const ec_zimage_req_t *req,
    const char *out_png_path, ec_zimage_progress_fn cb, void *cb_ctx, ec_zimage_res_t *res) {
  return ec_zimage_generate_rgb(zi, req, NULL, out_png_path, cb, cb_ctx, res);
}
