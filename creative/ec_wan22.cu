// ec_wan22.cu — 100% Pure Native Eureka Wan 2.2 TI2V-5B Video Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Pipeline:
//   1. Native C11 SentencePiece Unigram Tokenizer (ec_unigram.c -> umt5_tokenizer.json)
//   2. Native CUDA UMT5-XXL FP8-E4M3FN-Scaled 24-Layer Text Encoder (streamed 1 block/time)
//   3. Native CUDA 30-Block WanModel TI2V-5B DiT (dim=3072, H=24, D=128, ffn=14336,
//      3D RoPE [22,21,21], WanRMSNorm(3072), Flow Shift=5.0 Euler + CFG=5.0)
//   4. Native CUDA Causal 3D Wan2_2_VAE Streaming Decoder (CACHE_T=2, DupUp3D, unpatchify(2))
//   5. Hardware h264_nvenc MP4 Export (+ optional synced ACE-Step 1.5 AAC audio mux)

#define _POSIX_C_SOURCE 200809L
#include "ec_wan22.h"
#include "ec_media.h"
#include "ec_nvenc.h"
#include "ec_ops.h"
#include "ec_safetensors.h"
#include "ec_unigram.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wn_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static __device__ __forceinline__ uint16_t wn_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static inline uint16_t wn_f32_to_bf16_h(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  uint32_t lsb = (u >> 16) & 1U;
  u += 0x7FFFU + lsb;
  return (uint16_t)(u >> 16);
}
static inline float wn_f16_to_f32_h(uint16_t h) {
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
static inline float wn_bf16_to_f32_h(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// Exact OCP FP8 E4M3FN -> F32 LUT (bias=7, no inf, 0x7F/0xFF=NaN -> 0.0)
static float g_fp8_lut[256];
static int   g_fp8_lut_ready = 0;
static void wn_init_fp8_lut(void) {
  if (g_fp8_lut_ready) return;
  for (int i = 0; i < 256; i++) {
    int sign = (i >> 7) & 1;
    int exp  = (i >> 3) & 0xF;
    int mant = i & 0x7;
    float v = 0.0f;
    if (exp == 15 && mant == 7) {
      v = 0.0f;
    } else if (exp == 0) {
      v = ldexpf((float)mant, -9); // subnormal: mant * 2^-9
    } else {
      v = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    g_fp8_lut[i] = sign ? -v : v;
  }
  g_fp8_lut_ready = 1;
}

static __device__ __forceinline__ float wn_silu(float x) {
  return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float wn_gelu_tanh(float x) {
  float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(c));
}
static __device__ __forceinline__ float wn_warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// ===========================================================================
// 1. Unified High-Throughput BF16 TensorCore GEMM (via shared ec_ops.cu)
// ===========================================================================
static void wn_gemm(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t s) {
  ec_ops_gemm_bf16(X, W, bias, Y_f32, Y_bf16, M, N, K, s);
}

// ===========================================================================
// 2. General Norm, Elementwise, T5, DiT & VAE Kernels
// ===========================================================================
__global__ void k_wn_f32_to_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = wn_f32_to_bf16(src[i]);
}
__global__ void k_wn_add_inplace(float *dst, const float *src, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
__global__ void k_wn_add_scaled(float *dst, const float *src, const float *gate, int rows, int dim) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)rows * dim;
  if (idx >= tot) return;
  dst[idx] += src[idx] * gate[idx % dim];
}
__global__ void k_wn_add_vec(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_wn_add_head_mod(const float *mod2, const float *e, float *out2, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 2 * dim) out2[i] = mod2[i] + e[i % dim];
}
__global__ void k_wn_gelu_tanh_bf16(const float *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = wn_f32_to_bf16(wn_gelu_tanh(src[i]));
}
__global__ void k_wn_t5_geglu_bf16(const float *wi0, const float *wi1, uint16_t *out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = wn_f32_to_bf16(wn_gelu_tanh(wi0[i]) * wi1[i]);
}
__global__ void k_wn_fp8_to_bf16_scaled(
    const uint8_t *__restrict__ src, const float *__restrict__ lut,
    float scale, uint16_t *__restrict__ dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = wn_f32_to_bf16(lut[src[i]] * scale);
}

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w  (optionally + SiLU for VAE!)
__global__ void k_wn_rmsnorm(
    const float *__restrict__ x, const float *__restrict__ w,
    float *__restrict__ y_f32, uint16_t *__restrict__ y_bf16,
    int rows, int dim, float eps, int apply_silu) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; ss += v * v; }
  __shared__ float sm[32];
  ss = wn_warp_sum(ss);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    t = wn_warp_sum(t);
    if (threadIdx.x == 0) sm[0] = rsqrtf(t / (float)dim + eps);
  }
  __syncthreads();
  float inv = sm[0];
  float *yf = y_f32 ? (y_f32 + (long)r * dim) : NULL;
  uint16_t *yb = y_bf16 ? (y_bf16 + (long)r * dim) : NULL;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float v = xr[c] * inv * (w ? w[c] : 1.0f);
    if (apply_silu) v = wn_silu(v);
    if (yf) yf[c] = v;
    if (yb) yb[c] = wn_f32_to_bf16(v);
  }
}

// LayerNorm (optional affine w/b, optional AdaLN scale/shift) -> BF16
__global__ void k_wn_layernorm(
    const float *__restrict__ x,
    const float *__restrict__ w, const float *__restrict__ b,
    const float *__restrict__ adaln_scale, const float *__restrict__ adaln_shift,
    uint16_t *__restrict__ y_bf16, int rows, int dim, float eps) {
  int r = blockIdx.x;
  if (r >= rows) return;
  const float *xr = x + (long)r * dim;
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) { float v = xr[c]; s1 += v; s2 += v * v; }
  __shared__ float sm1[32], sm2[32];
  s1 = wn_warp_sum(s1); s2 = wn_warp_sum(s2);
  if ((threadIdx.x & 31) == 0) { sm1[threadIdx.x >> 5] = s1; sm2[threadIdx.x >> 5] = s2; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = blockDim.x >> 5;
    float a = (threadIdx.x < nw) ? sm1[threadIdx.x] : 0.0f;
    float bb = (threadIdx.x < nw) ? sm2[threadIdx.x] : 0.0f;
    a = wn_warp_sum(a); bb = wn_warp_sum(bb);
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
  uint16_t *yb = y_bf16 + (long)r * dim;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float v = (xr[c] - mean) * inv;
    if (w) v = v * w[c] + b[c];
    if (adaln_scale) v = v * (1.0f + adaln_scale[c]) + adaln_shift[c];
    yb[c] = wn_f32_to_bf16(v);
  }
}

// ===========================================================================
// 3. UMT5-XXL Attention Kernel (H=64, D=64, NO 1/sqrt(D) scaling, Per-Block Relative Bias!)
// ===========================================================================
__device__ __forceinline__ int t5_rel_bucket(int rel_pos) {
  // bidirectional=True, num_buckets=32, max_dist=128 (wan22_t5.py lines 245-264)
  int nb = 16;
  int bucket = (rel_pos > 0) ? nb : 0;
  int a = (rel_pos < 0) ? -rel_pos : rel_pos;
  int max_exact = 8;
  if (a < max_exact) {
    bucket += a;
  } else {
    float val = (float)max_exact + logf((float)a / (float)max_exact) / logf(128.0f / (float)max_exact) * (float)(nb - max_exact);
    int iv = (int)val;
    if (iv > nb - 1) iv = nb - 1;
    bucket += iv;
  }
  return bucket;
}

__global__ void k_wn_t5_attn_d64(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    const float *__restrict__ rel_bias, // [32, 64]
    uint16_t *__restrict__ O_bf16, int L, int H) {
  int pair = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (pair >= L * H) return;
  int qpos = pair / H, h = pair % H;
  const float *qv = Q + ((long)qpos * H + h) * 64;
  float q0 = qv[lane], q1 = qv[lane + 32];
  float m_max = -1e30f, l_sum = 0.0f, a0 = 0.0f, a1 = 0.0f;
  for (int kpos = 0; kpos < L; kpos++) {
    const float *kv = K + ((long)kpos * H + h) * 64;
    float dot = wn_warp_sum(q0 * kv[lane] + q1 * kv[lane + 32]); // T5: NO scaling!
    int bkt = t5_rel_bucket(kpos - qpos);
    dot += rel_bias[bkt * H + h];
    float nm = fmaxf(m_max, dot);
    float al = expf(m_max - nm);
    float p  = expf(dot - nm);
    l_sum = l_sum * al + p;
    m_max = nm;
    const float *vv = V + ((long)kpos * H + h) * 64;
    a0 = a0 * al + p * vv[lane];
    a1 = a1 * al + p * vv[lane + 32];
  }
  float inv = 1.0f / (l_sum + 1e-12f);
  uint16_t *out = O_bf16 + ((long)qpos * H + h) * 64;
  out[lane]      = wn_f32_to_bf16(a0 * inv);
  out[lane + 32] = wn_f32_to_bf16(a1 * inv);
}

// ===========================================================================
// 4. WanModel 3D RoPE, Patchify/Unpatchify & Shared-Mem FlashAttention (D=128)
// ===========================================================================
// Pack latent z[F, H_lat, W_lat, 48] -> patches_bf16[F*Hp*Wp, 192] matching Conv3d weight [3072, 48, 1, 2, 2]
__global__ void k_wn_patchify(
    const float *__restrict__ z, uint16_t *__restrict__ out_bf16,
    int F, int Hp, int Wp) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long N = (long)F * Hp * Wp;
  if (idx >= N * 192) return;
  long tok = idx / 192;
  int col = (int)(idx % 192);
  int c = col >> 2;
  int dh = (col >> 1) & 1;
  int dw = col & 1;
  int f  = (int)(tok / (Hp * Wp));
  int hp = (int)((tok / Wp) % Hp);
  int wp = (int)(tok % Wp);
  int hl = hp * 2 + dh, wl = wp * 2 + dw;
  int Hl = Hp * 2, Wl = Wp * 2;
  float v = z[(((long)f * Hl + hl) * Wl + wl) * 48 + c];
  out_bf16[idx] = wn_f32_to_bf16(v);
}

// Unpatchify head_out[F*Hp*Wp, 192] -> z[F, H_lat, W_lat, 48]
// Matches wan22_model.py line 518: view(F, Hp, Wp, 1, 2, 2, 48) -> [dh, dw, c] order!
__global__ void k_wn_unpatchify(
    const float *__restrict__ head_out, float *__restrict__ z_out,
    int F, int Hp, int Wp) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long N = (long)F * Hp * Wp;
  if (idx >= N * 192) return;
  long tok = idx / 192;
  int col = (int)(idx % 192);
  int c = col % 48;
  int sub = col / 48;
  int dh = sub >> 1;
  int dw = sub & 1;
  int f  = (int)(tok / (Hp * Wp));
  int hp = (int)((tok / Wp) % Hp);
  int wp = (int)(tok % Wp);
  int Hl = Hp * 2, Wl = Wp * 2;
  z_out[(((long)f * Hl + (hp * 2 + dh)) * Wl + (wp * 2 + dw)) * 48 + c] = head_out[idx];
}

// Fused Self-Attention QKV WanRMSNorm(3072) + 3D RoPE ([22,21,21]) + Transpose [N, 3*3072] -> [24, N, 128] BF16
// Replaces 2 separate RMSNorms + 3 separate transposes with 1 single bandwidth-fused kernel!
__global__ void k_wn_sa_qkv_norm_rope_bf16(
    const float *__restrict__ qkv,   // [N, 3 * DIM] where DIM = 3072 = 24 * 128
    const float *__restrict__ nq_w,  // [DIM]
    const float *__restrict__ nk_w,  // [DIM]
    uint16_t *__restrict__ qr_b,     // [H, N, 128] BF16
    uint16_t *__restrict__ kr_b,     // [H, N, 128] BF16
    uint16_t *__restrict__ vr_b,     // [H, N, 128] BF16
    int N, int F, int Hp, int Wp) {
  const int tok = blockIdx.x;
  if (tok >= N) return;
  constexpr int DIM = 3072;
  const float *qs = qkv + (long)tok * (3 * DIM);
  const float *ks = qs + DIM;
  const float *vs = qs + 2 * DIM;

  // 1. Block-wide WanRMSNorm(3072) on Q and K simultaneously
  float sq = 0.0f, sk = 0.0f;
  for (int c = threadIdx.x; c < DIM; c += blockDim.x) {
    float qv = qs[c], kv = ks[c];
    sq += qv * qv;
    sk += kv * kv;
  }
  __shared__ float smq[32], smk[32];
  sq = wn_warp_sum(sq); sk = wn_warp_sum(sk);
  if ((threadIdx.x & 31) == 0) { smq[threadIdx.x >> 5] = sq; smk[threadIdx.x >> 5] = sk; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = blockDim.x >> 5;
    float aq = (threadIdx.x < nw) ? smq[threadIdx.x] : 0.0f;
    float ak = (threadIdx.x < nw) ? smk[threadIdx.x] : 0.0f;
    aq = wn_warp_sum(aq); ak = wn_warp_sum(ak);
    if (threadIdx.x == 0) {
      smq[0] = rsqrtf(aq / (float)DIM + 1e-6f);
      smk[0] = rsqrtf(ak / (float)DIM + 1e-6f);
    }
  }
  __syncthreads();
  const float inv_q = smq[0], inv_k = smk[0];

  // 2. Apply 3D RoPE ([22,21,21]) & transpose directly into BF16 [24, N, 128]
  const int f  = tok / (Hp * Wp);
  const int hp = (tok / Wp) % Hp;
  const int wp = tok % Wp;

  for (int gp = threadIdx.x; gp < 1536; gp += blockDim.x) {
    const int h  = gp >> 6;  // 0..23
    const int p  = gp & 63;  // 0..63 complex pairs
    const int d0 = 2 * p, d1 = d0 + 1;
    const int c0 = h * 128 + d0, c1 = c0 + 1;

    const float q0 = qs[c0] * inv_q * nq_w[c0];
    const float q1 = qs[c1] * inv_q * nq_w[c1];
    const float k0 = ks[c0] * inv_k * nk_w[c0];
    const float k1 = ks[c1] * inv_k * nk_w[c1];

    double coord, denom;
    int fi;
    if (p < 22)      { coord = (double)f;  fi = p;      denom = 44.0; }
    else if (p < 43) { coord = (double)hp; fi = p - 22; denom = 42.0; }
    else             { coord = (double)wp; fi = p - 43; denom = 42.0; }
    double ang = coord / pow(10000.0, (double)(2 * fi) / denom);
    float cs = (float)cos(ang), sn = (float)sin(ang);

    const long out_off = ((long)h * N + tok) * 128;
    qr_b[out_off + d0] = wn_f32_to_bf16(q0 * cs - q1 * sn);
    qr_b[out_off + d1] = wn_f32_to_bf16(q0 * sn + q1 * cs);
    kr_b[out_off + d0] = wn_f32_to_bf16(k0 * cs - k1 * sn);
    kr_b[out_off + d1] = wn_f32_to_bf16(k0 * sn + k1 * cs);
    vr_b[out_off + d0] = wn_f32_to_bf16(vs[c0]);
    vr_b[out_off + d1] = wn_f32_to_bf16(vs[c1]);
  }
}

// Transpose [L, H, 128] F32 -> [H, L, 128] BF16 (for cross-attention Q and cached K/V!)
__global__ void k_wn_transpose_bf16(
    const float *__restrict__ in, uint16_t *__restrict__ out, int L, int H) {
  int idx = (int)(((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (idx >= L * H) return;
  int tok = idx / H, h = idx % H;
  const float *src = in + ((long)tok * H + h) * 128;
  uint16_t *dst = out + ((long)h * L + tok) * 128;
  #pragma unroll
  for (int i = 0; i < 4; i++) dst[lane + i * 32] = wn_f32_to_bf16(src[lane + i * 32]);
}

// ===========================================================================
// 5. Causal 3D Wan2_2_VAE Decoder Kernels (Frame-Major Layout: [T, H, W, C])
// ===========================================================================
// Im2Col for CausalConv3d(Cin -> Cout, kt=3, kh=3, kw=3) on padded input [T_in, H, W, Cin]
// producing output frames [0 .. T_out-1] (T_out = T_in - 2), chunked over pixels [p0 .. p0+chunk-1]
__global__ void k_wn_im2col_c3d(
    const float *__restrict__ x_pad, uint16_t *__restrict__ col_bf16,
    int T_out, int H, int W, int Cin, long p0, int chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K = 27L * Cin;
  long tot = (long)chunk * K;
  if (idx >= tot) return;
  int row = (int)(idx / K);
  int rem = (int)(idx % K);
  int k = rem / Cin;
  int ic = rem % Cin;
  long p = p0 + row;
  int t = (int)(p / ((long)H * W));
  int h = (int)((p / W) % H);
  int w = (int)(p % W);
  int kt = k / 9;       // 0, 1, 2 (causal: x_pad already has 2 prepended frames!)
  int kh = ((k / 3) % 3) - 1;
  int kw = (k % 3) - 1;
  int it = t + kt, ih = h + kh, iw = w + kw;
  float v = 0.0f;
  if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
    v = x_pad[(((long)it * H + ih) * W + iw) * Cin + ic];
  }
  col_bf16[idx] = wn_f32_to_bf16(v);
}

// Im2Col for 2D Conv3x3 (pad=1) on [T, H, W, Cin]
__global__ void k_wn_im2col_2d_k3(
    const float *__restrict__ x, uint16_t *__restrict__ col_bf16,
    int T, int H, int W, int Cin, long p0, int chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K = 9L * Cin;
  long tot = (long)chunk * K;
  if (idx >= tot) return;
  int row = (int)(idx / K);
  int rem = (int)(idx % K);
  int k = rem / Cin;
  int ic = rem % Cin;
  long p = p0 + row;
  int t = (int)(p / ((long)H * W));
  int h = (int)((p / W) % H);
  int w = (int)(p % W);
  int kh = (k / 3) - 1;
  int kw = (k % 3) - 1;
  int ih = h + kh, iw = w + kw;
  float v = 0.0f;
  if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
    v = x[(((long)t * H + ih) * W + iw) * Cin + ic];
  }
  col_bf16[idx] = wn_f32_to_bf16(v);
}

// Im2Col for Causal Time-Conv3x1x1 on padded [T_out+2, H, W, Cin] -> [T_out*H*W, 3*Cin]
__global__ void k_wn_im2col_t3(
    const float *__restrict__ x_pad, uint16_t *__restrict__ col_bf16,
    int T_out, int H, int W, int Cin, long p0, int chunk) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K = 3L * Cin;
  long tot = (long)chunk * K;
  if (idx >= tot) return;
  int row = (int)(idx / K);
  int rem = (int)(idx % K);
  int kt = rem / Cin;
  int ic = rem % Cin;
  long p = p0 + row;
  int t = (int)(p / ((long)H * W));
  int hw = (int)(p % ((long)H * W));
  col_bf16[idx] = wn_f32_to_bf16(x_pad[((long)(t + kt) * H * W + hw) * Cin + ic]);
}

// Interleave time_conv output [T, H, W, 2*C] -> [2*T, H, W, C] (matches wan22_vae.py lines 148-151)
__global__ void k_wn_time_unshuffle(
    const float *__restrict__ in, float *__restrict__ out, int T, int H, int W, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)(2 * T) * H * W * C;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  long p = idx / C;
  int ot = (int)(p / ((long)H * W));
  int hw = (int)(p % ((long)H * W));
  int it = ot >> 1, sub = ot & 1;
  out[idx] = in[((long)it * H * W + hw) * (2 * C) + sub * C + c];
}

// Nearest 2x spatial upsample: [T, H, W, C] -> [T, 2H, 2W, C]
__global__ void k_wn_upsample2d_nearest(
    const float *__restrict__ in, float *__restrict__ out, int T, int H, int W, int C) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)T * (2 * H) * (2 * W) * C;
  if (idx >= tot) return;
  int c = (int)(idx % C);
  long p = idx / C;
  int t  = (int)(p / ((long)(2 * H) * (2 * W)));
  int oh = (int)((p / (2 * W)) % (2 * H));
  int ow = (int)(p % (2 * W));
  out[idx] = in[(((long)t * H + (oh >> 1)) * W + (ow >> 1)) * C + c];
}

// DupUp3D shortcut (exact match to wan22_vae.py lines 370-412!)
// in[T, H, W, in_c] -> out[T_out, 2H, 2W, out_c] where T_out = first_chunk ? (T*ft - (ft-1)) : (T*ft)
__global__ void k_wn_dupup3d(
    const float *__restrict__ in, float *__restrict__ out,
    int T, int H, int W, int in_c, int out_c, int ft, int first_chunk) {
  int T_full = T * ft;
  int t_skip = first_chunk ? (ft - 1) : 0;
  int T_out = T_full - t_skip;
  int H2 = 2 * H, W2 = 2 * W;
  int repeats = (out_c * ft * 4) / in_c;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)T_out * H2 * W2 * out_c;
  if (idx >= tot) return;
  int oc = (int)(idx % out_c);
  long p = idx / out_c;
  int ot = (int)(p / ((long)H2 * W2)) + t_skip;
  int oh = (int)((p / W2) % H2);
  int ow = (int)(p % W2);
  int it = ot / ft, st = ot % ft;
  int ih = oh >> 1, sh = oh & 1;
  int iw = ow >> 1, sw = ow & 1;
  int flat_c = ((oc * ft + st) * 2 + sh) * 2 + sw;
  int ic = flat_c / repeats;
  out[idx] = in[(((long)it * H + ih) * W + iw) * in_c + ic];
}

// Single-head spatial self-attention for VAE middle.1 (C=1024, [H*W, 3072] qkv -> [H*W, 1024] BF16)
__global__ void k_wn_vae_attn_1h(
    const float *__restrict__ qkv, uint16_t *__restrict__ out_bf16, int HW, int C) {
  int qpos = blockIdx.x;
  if (qpos >= HW) return;
  const float *qv = qkv + (long)qpos * (3 * C);
  float scale = rsqrtf((float)C);

  extern __shared__ float smem[]; // [HW] attention weights + [32] warp scratch
  float *scores = smem;
  float *wred   = smem + HW;

  float m_local = -1e30f;
  for (int kpos = 0; kpos < HW; kpos++) {
    const float *kv = qkv + (long)kpos * (3 * C) + C;
    float dot = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) dot += qv[c] * kv[c];
    dot = wn_warp_sum(dot);
    if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = dot;
    __syncthreads();
    if (threadIdx.x < 32) {
      float t = (threadIdx.x < (blockDim.x >> 5)) ? wred[threadIdx.x] : 0.0f;
      t = wn_warp_sum(t);
      if (threadIdx.x == 0) {
        float s = t * scale;
        scores[kpos] = s;
        if (s > m_local) m_local = s;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) wred[0] = m_local;
  __syncthreads();
  float mmax = wred[0];

  float lsum = 0.0f;
  for (int kpos = threadIdx.x; kpos < HW; kpos += blockDim.x) {
    float e = expf(scores[kpos] - mmax);
    scores[kpos] = e;
    lsum += e;
  }
  lsum = wn_warp_sum(lsum);
  if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = lsum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = (threadIdx.x < (blockDim.x >> 5)) ? wred[threadIdx.x] : 0.0f;
    t = wn_warp_sum(t);
    if (threadIdx.x == 0) wred[0] = 1.0f / (t + 1e-12f);
  }
  __syncthreads();
  float inv_l = wred[0];

  uint16_t *out = out_bf16 + (long)qpos * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.0f;
    for (int kpos = 0; kpos < HW; kpos++) {
      acc += scores[kpos] * qkv[(long)kpos * (3 * C) + 2 * C + c];
    }
    out[c] = wn_f32_to_bf16(acc * inv_l);
  }
}

// Unpatchify(patch_size=2): [T, H, W, 12] -> RGB24 [T, 2H, 2W, 3]
// Matches wan22_vae.py line 309: (c r q) -> (h q)(w r) => channel index = (c*2 + r)*2 + q
__global__ void k_wn_unpatchify_rgb24(
    const float *__restrict__ in12, uint8_t *__restrict__ rgb_out, int T, int H, int W) {
  int H2 = 2 * H, W2 = 2 * W;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = (long)T * H2 * W2 * 3;
  if (idx >= tot) return;
  int c = (int)(idx % 3);
  long p = idx / 3;
  int t  = (int)(p / ((long)H2 * W2));
  int oh = (int)((p / W2) % H2);
  int ow = (int)(p % W2);
  int ih = oh >> 1, q = oh & 1;
  int iw = ow >> 1, r = ow & 1;
  int ch12 = (c * 2 + r) * 2 + q;
  float v = in12[(((long)t * H + ih) * W + iw) * 12 + ch12];
  if (v < -1.0f) v = -1.0f;
  if (v >  1.0f) v =  1.0f;
  rgb_out[idx] = (uint8_t)lrintf((v * 0.5f + 0.5f) * 255.0f);
}

// ===========================================================================
// 6. Safetensors Weight Helpers (F16 / BF16 / F32 / FP8_E4M3-scaled)
// ===========================================================================
static int wn_load_f32_host(const ec_safetensors_t *st, const char *name, float **out, size_t *out_n) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[wan22] missing tensor: %s\n", name); return -1; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  float *buf = (float *)malloc(n * 4);
  const void *p = ec_st_ptr(st, e);
  if (e->dtype == EC_DTYPE_F32) memcpy(buf, p, n * 4);
  else if (e->dtype == EC_DTYPE_F16) {
    const uint16_t *s = (const uint16_t *)p;
    for (size_t i = 0; i < n; i++) buf[i] = wn_f16_to_f32_h(s[i]);
  } else if (e->dtype == EC_DTYPE_BF16) {
    const uint16_t *s = (const uint16_t *)p;
    for (size_t i = 0; i < n; i++) buf[i] = wn_bf16_to_f32_h(s[i]);
  } else { free(buf); return -1; }
  *out = buf;
  if (out_n) *out_n = n;
  return 0;
}

static float *wn_upload_f32(const ec_safetensors_t *st, const char *name) {
  float *h = NULL; size_t n = 0;
  if (wn_load_f32_host(st, name, &h, &n) != 0) return NULL;
  float *d = NULL;
  if (cudaMalloc(&d, n * 4) != cudaSuccess) { free(h); return NULL; }
  cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
  free(h);
  return d;
}

// Fast GPU F16->BF16 conversion kernel so loading 9.4 GB WanModel F16 weights takes < 300ms!
__global__ void k_wn_f16_to_bf16(const __half *src, uint16_t *dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = wn_f32_to_bf16(__half2float(src[i]));
}

static uint16_t *wn_upload_bf16(const ec_safetensors_t *st, const char *name, cudaStream_t stream) {
  const ec_st_entry_t *e = ec_st_find(st, name);
  if (!e) { fprintf(stderr, "[wan22] missing tensor: %s\n", name); return NULL; }
  size_t esz = (e->dtype == EC_DTYPE_F32) ? 4 : 2;
  size_t n = e->nbytes / esz;
  uint16_t *d = NULL;
  if (cudaMalloc(&d, n * 2) != cudaSuccess) return NULL;
  if (e->dtype == EC_DTYPE_BF16) {
    cudaMemcpyAsync(d, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, stream);
  } else if (e->dtype == EC_DTYPE_F16) {
    __half *tmp = NULL;
    if (cudaMalloc(&tmp, n * 2) != cudaSuccess) { cudaFree(d); return NULL; }
    cudaMemcpyAsync(tmp, ec_st_ptr(st, e), n * 2, cudaMemcpyHostToDevice, stream);
    k_wn_f16_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(tmp, d, (long)n);
    cudaStreamSynchronize(stream);
    cudaFree(tmp);
  } else {
    float *h = NULL;
    if (wn_load_f32_host(st, name, &h, &n) != 0) { cudaFree(d); return NULL; }
    uint16_t *b = (uint16_t *)malloc(n * 2);
    for (size_t i = 0; i < n; i++) b[i] = wn_f32_to_bf16_h(h[i]);
    free(h);
    cudaMemcpy(d, b, n * 2, cudaMemcpyHostToDevice);
    free(b);
  }
  return d;
}

// Upload FP8_E4M3 linear weight + scalar F32 scale_weight -> BF16 on GPU
static uint16_t *wn_upload_fp8_bf16(
    const ec_safetensors_t *st, const char *w_name, const char *s_name,
    const float *lut_d, cudaStream_t stream) {
  const ec_st_entry_t *ew = ec_st_find(st, w_name);
  const ec_st_entry_t *es = ec_st_find(st, s_name);
  if (!ew || !es) { fprintf(stderr, "[wan22] missing fp8 pair: %s\n", w_name); return NULL; }
  float scale = *(const float *)ec_st_ptr(st, es);
  size_t n = ew->nbytes;
  uint8_t *raw_d = NULL;
  uint16_t *out_d = NULL;
  if (cudaMalloc(&raw_d, n) != cudaSuccess) return NULL;
  if (cudaMalloc(&out_d, n * 2) != cudaSuccess) { cudaFree(raw_d); return NULL; }
  cudaMemcpyAsync(raw_d, ec_st_ptr(st, ew), n, cudaMemcpyHostToDevice, stream);
  k_wn_fp8_to_bf16_scaled<<<(n + 255) / 256, 256, 0, stream>>>(raw_d, lut_d, scale, out_d, (long)n);
  cudaStreamSynchronize(stream);
  cudaFree(raw_d);
  return out_d;
}

// Transpose Conv weight [Cout, Cin, K] -> [Cout, K * Cin] BF16 (row oc, col k*Cin + ic)
static uint16_t *wn_upload_conv_k_bf16(const ec_safetensors_t *st, const char *name, int Cout, int Cin, int K) {
  float *h = NULL; size_t n = 0;
  if (wn_load_f32_host(st, name, &h, &n) != 0) return NULL;
  uint16_t *wt = (uint16_t *)malloc((size_t)Cout * K * Cin * 2);
  for (int oc = 0; oc < Cout; oc++)
    for (int ic = 0; ic < Cin; ic++)
      for (int k = 0; k < K; k++)
        wt[(long)oc * (K * Cin) + k * Cin + ic] = wn_f32_to_bf16_h(h[((long)oc * Cin + ic) * K + k]);
  free(h);
  uint16_t *d = NULL;
  cudaMalloc(&d, (size_t)Cout * K * Cin * 2);
  cudaMemcpy(d, wt, (size_t)Cout * K * Cin * 2, cudaMemcpyHostToDevice);
  free(wt);
  return d;
}

// ===========================================================================
// 7. Resident Model Structures (WanModel 5B + Wan2_2_VAE Decoder)
// ===========================================================================
typedef struct {
  float    *mod; // [6 * 3072]
  uint16_t *sa_qkv_w, *sa_o_w; // Fused QKV [3*3072, 3072] BF16 = 1 GEMM instead of 3!
  float    *sa_qkv_b, *sa_o_b, *sa_nq, *sa_nk;
  float    *norm3_w, *norm3_b;
  uint16_t *ca_q_w, *ca_k_w, *ca_v_w, *ca_o_w;
  float    *ca_q_b, *ca_k_b, *ca_v_b, *ca_o_b, *ca_nq, *ca_nk;
  uint16_t *ffn0_w, *ffn2_w;
  float    *ffn0_b, *ffn2_b;
} wn_dit_blk_t;

typedef struct {
  int loaded;
  uint16_t *patch_w, *txt0_w, *txt2_w, *t0_w, *t2_w, *tproj_w, *head_w;
  float    *patch_b, *txt0_b, *txt2_b, *t0_b, *t2_b, *tproj_b, *head_b, *head_mod;
  wn_dit_blk_t blocks[30];
} wn_dit_res_t;

typedef struct {
  int in_c, out_c, has_sc;
  float    *g0, *g3, *b2, *b6, *sc_b;
  uint16_t *w2, *w6, *sc_w;
} wn_vae_resblk_t;

typedef struct {
  int in_c, out_c, up_flag, t_up_flag;
  wn_vae_resblk_t r[3];
  uint16_t *sp_w, *tm_w;
  float    *sp_b, *tm_b;
} wn_vae_up_t;

typedef struct {
  int loaded;
  uint16_t *conv2_w, *conv1_w, *head_w;
  float    *conv2_b, *conv1_b, *head_g, *head_b;
  wn_vae_resblk_t mid0, mid2;
  float    *attn_g, *attn_qkv_b, *attn_pr_b;
  uint16_t *attn_qkv_w, *attn_pr_w;
  wn_vae_up_t up[4];
} wn_vae_res_t;

#define WN_T5_CACHE_SLOTS 4

struct ec_wan22 {
  ec_wan22_paths_t paths;
  cudaStream_t     stream;
  ec_unigram_t    *ug;
  ec_safetensors_t dit_st, vae_st, te_st;
  int              dit_open, vae_open, te_open;
  wn_dit_res_t     dit;
  wn_vae_res_t     vae;
  // 4-slot Host RAM UMT5-XXL Prompt Embedding Cache (16 MB host RAM -> 0.5 ms warm hits!)
  char             t5_keys[WN_T5_CACHE_SLOTS][1024];
  uint16_t        *t5_vals[WN_T5_CACHE_SLOTS];
  int              t5_valid[WN_T5_CACHE_SLOTS];
  int              t5_next_slot;
};

static void free_dit_res(wn_dit_res_t *d) {
  if (!d->loaded) return;
  cudaFree(d->patch_w); cudaFree(d->patch_b);
  cudaFree(d->txt0_w);  cudaFree(d->txt0_b);  cudaFree(d->txt2_w); cudaFree(d->txt2_b);
  cudaFree(d->t0_w);    cudaFree(d->t0_b);    cudaFree(d->t2_w);   cudaFree(d->t2_b);
  cudaFree(d->tproj_w); cudaFree(d->tproj_b);
  cudaFree(d->head_w);  cudaFree(d->head_b);  cudaFree(d->head_mod);
  for (int i = 0; i < 30; i++) {
    wn_dit_blk_t *B = &d->blocks[i];
    cudaFree(B->mod);
    cudaFree(B->sa_qkv_w); cudaFree(B->sa_qkv_b); cudaFree(B->sa_o_w); cudaFree(B->sa_o_b);
    cudaFree(B->sa_nq);  cudaFree(B->sa_nk);
    cudaFree(B->norm3_w); cudaFree(B->norm3_b);
    cudaFree(B->ca_q_w); cudaFree(B->ca_q_b); cudaFree(B->ca_k_w); cudaFree(B->ca_k_b);
    cudaFree(B->ca_v_w); cudaFree(B->ca_v_b); cudaFree(B->ca_o_w); cudaFree(B->ca_o_b);
    cudaFree(B->ca_nq);  cudaFree(B->ca_nk);
    cudaFree(B->ffn0_w); cudaFree(B->ffn0_b); cudaFree(B->ffn2_w); cudaFree(B->ffn2_b);
  }
  memset(d, 0, sizeof *d);
}

static void free_vae_rb(wn_vae_resblk_t *r) {
  cudaFree(r->g0); cudaFree(r->g3); cudaFree(r->w2); cudaFree(r->b2);
  cudaFree(r->w6); cudaFree(r->b6); cudaFree(r->sc_w); cudaFree(r->sc_b);
}

static void free_vae_res(wn_vae_res_t *v) {
  if (!v->loaded) return;
  cudaFree(v->conv2_w); cudaFree(v->conv2_b);
  cudaFree(v->conv1_w); cudaFree(v->conv1_b);
  cudaFree(v->head_g);  cudaFree(v->head_w); cudaFree(v->head_b);
  free_vae_rb(&v->mid0); free_vae_rb(&v->mid2);
  cudaFree(v->attn_g); cudaFree(v->attn_qkv_w); cudaFree(v->attn_qkv_b);
  cudaFree(v->attn_pr_w); cudaFree(v->attn_pr_b);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) free_vae_rb(&v->up[i].r[j]);
    cudaFree(v->up[i].sp_w); cudaFree(v->up[i].sp_b);
    cudaFree(v->up[i].tm_w); cudaFree(v->up[i].tm_b);
  }
  memset(v, 0, sizeof *v);
}

static int ensure_dit_loaded(ec_wan22_t *w) {
  if (w->dit.loaded) return 0;
  const ec_safetensors_t *st = &w->dit_st;
  wn_dit_res_t *d = &w->dit;

  d->patch_w  = wn_upload_bf16(st, "patch_embedding.weight", w->stream);
  d->patch_b  = wn_upload_f32(st,  "patch_embedding.bias");
  d->txt0_w   = wn_upload_bf16(st, "text_embedding.0.weight", w->stream);
  d->txt0_b   = wn_upload_f32(st,  "text_embedding.0.bias");
  d->txt2_w   = wn_upload_bf16(st, "text_embedding.2.weight", w->stream);
  d->txt2_b   = wn_upload_f32(st,  "text_embedding.2.bias");
  d->t0_w     = wn_upload_bf16(st, "time_embedding.0.weight", w->stream);
  d->t0_b     = wn_upload_f32(st,  "time_embedding.0.bias");
  d->t2_w     = wn_upload_bf16(st, "time_embedding.2.weight", w->stream);
  d->t2_b     = wn_upload_f32(st,  "time_embedding.2.bias");
  d->tproj_w  = wn_upload_bf16(st, "time_projection.1.weight", w->stream);
  d->tproj_b  = wn_upload_f32(st,  "time_projection.1.bias");
  d->head_w   = wn_upload_bf16(st, "head.head.weight", w->stream);
  d->head_b   = wn_upload_f32(st,  "head.head.bias");
  d->head_mod = wn_upload_f32(st,  "head.modulation");
  if (!d->patch_w || !d->txt0_w || !d->tproj_w || !d->head_w) { free_dit_res(d); return -1; }

  char k[192];
  for (int i = 0; i < 30; i++) {
    wn_dit_blk_t *B = &d->blocks[i];
    snprintf(k, sizeof k, "blocks.%d.modulation", i);             B->mod     = wn_upload_f32(st, k);
    // Fuse self-attn Q, K, V weights & biases into contiguous [3*3072, 3072] BF16 + [3*3072] F32!
    snprintf(k, sizeof k, "blocks.%d.self_attn.q.weight", i);     uint16_t *qw = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.k.weight", i);     uint16_t *kw = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.v.weight", i);     uint16_t *vw = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.q.bias", i);       float *qb = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.k.bias", i);       float *kb = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.v.bias", i);       float *vb = wn_upload_f32(st, k);
    if (qw && kw && vw && qb && kb && vb) {
      const size_t W_BYTES = (size_t)3072 * 3072 * 2;
      const size_t B_BYTES = (size_t)3072 * 4;
      if (cudaMalloc(&B->sa_qkv_w, 3 * W_BYTES) == cudaSuccess &&
          cudaMalloc(&B->sa_qkv_b, 3 * B_BYTES) == cudaSuccess) {
        cudaMemcpyAsync(B->sa_qkv_w,                  qw, W_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaMemcpyAsync(B->sa_qkv_w + 3072L * 3072L,  kw, W_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaMemcpyAsync(B->sa_qkv_w + 6144L * 3072L,  vw, W_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaMemcpyAsync(B->sa_qkv_b,                  qb, B_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaMemcpyAsync(B->sa_qkv_b + 3072L,          kb, B_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaMemcpyAsync(B->sa_qkv_b + 6144L,          vb, B_BYTES, cudaMemcpyDeviceToDevice, w->stream);
        cudaStreamSynchronize(w->stream);
      }
    }
    cudaFree(qw); cudaFree(kw); cudaFree(vw); cudaFree(qb); cudaFree(kb); cudaFree(vb);
    snprintf(k, sizeof k, "blocks.%d.self_attn.o.weight", i);     B->sa_o_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.self_attn.o.bias", i);       B->sa_o_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.norm_q.weight", i); B->sa_nq  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.self_attn.norm_k.weight", i); B->sa_nk  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm3.weight", i);           B->norm3_w = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.norm3.bias", i);             B->norm3_b = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.q.weight", i);    B->ca_q_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.q.bias", i);      B->ca_q_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.k.weight", i);    B->ca_k_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.k.bias", i);      B->ca_k_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.v.weight", i);    B->ca_v_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.v.bias", i);      B->ca_v_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.o.weight", i);    B->ca_o_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.o.bias", i);      B->ca_o_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.norm_q.weight", i); B->ca_nq = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.cross_attn.norm_k.weight", i); B->ca_nk = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.ffn.0.weight", i);           B->ffn0_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.ffn.0.bias", i);             B->ffn0_b  = wn_upload_f32(st, k);
    snprintf(k, sizeof k, "blocks.%d.ffn.2.weight", i);           B->ffn2_w  = wn_upload_bf16(st, k, w->stream);
    snprintf(k, sizeof k, "blocks.%d.ffn.2.bias", i);             B->ffn2_b  = wn_upload_f32(st, k);
    if (!B->sa_qkv_w || !B->sa_qkv_b || !B->ca_q_w || !B->ffn2_w) { free_dit_res(d); return -1; }
  }
  d->loaded = 1;
  return 0;
}

static int load_vae_rb(const ec_safetensors_t *st, const char *pfx, int in_c, int out_c, wn_vae_resblk_t *r) {
  char k[192];
  r->in_c = in_c; r->out_c = out_c; r->has_sc = (in_c != out_c);
  snprintf(k, sizeof k, "%s.residual.0.gamma", pfx);  r->g0 = wn_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.residual.2.weight", pfx); r->w2 = wn_upload_conv_k_bf16(st, k, out_c, in_c, 27);
  snprintf(k, sizeof k, "%s.residual.2.bias", pfx);   r->b2 = wn_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.residual.3.gamma", pfx);  r->g3 = wn_upload_f32(st, k);
  snprintf(k, sizeof k, "%s.residual.6.weight", pfx); r->w6 = wn_upload_conv_k_bf16(st, k, out_c, out_c, 27);
  snprintf(k, sizeof k, "%s.residual.6.bias", pfx);   r->b6 = wn_upload_f32(st, k);
  if (r->has_sc) {
    snprintf(k, sizeof k, "%s.shortcut.weight", pfx); r->sc_w = wn_upload_conv_k_bf16(st, k, out_c, in_c, 1);
    snprintf(k, sizeof k, "%s.shortcut.bias", pfx);   r->sc_b = wn_upload_f32(st, k);
  }
  return (r->g0 && r->w2 && r->g3 && r->w6 && (!r->has_sc || r->sc_w)) ? 0 : -1;
}

static int ensure_vae_loaded(ec_wan22_t *w) {
  if (w->vae.loaded) return 0;
  const ec_safetensors_t *st = &w->vae_st;
  wn_vae_res_t *v = &w->vae;

  v->conv2_w    = wn_upload_conv_k_bf16(st, "conv2.weight", 48, 48, 1);
  v->conv2_b    = wn_upload_f32(st, "conv2.bias");
  v->conv1_w    = wn_upload_conv_k_bf16(st, "decoder.conv1.weight", 1024, 48, 27);
  v->conv1_b    = wn_upload_f32(st, "decoder.conv1.bias");
  v->attn_g     = wn_upload_f32(st, "decoder.middle.1.norm.gamma");
  v->attn_qkv_w = wn_upload_conv_k_bf16(st, "decoder.middle.1.to_qkv.weight", 3072, 1024, 1);
  v->attn_qkv_b = wn_upload_f32(st, "decoder.middle.1.to_qkv.bias");
  v->attn_pr_w  = wn_upload_conv_k_bf16(st, "decoder.middle.1.proj.weight", 1024, 1024, 1);
  v->attn_pr_b  = wn_upload_f32(st, "decoder.middle.1.proj.bias");
  v->head_g     = wn_upload_f32(st, "decoder.head.0.gamma");
  v->head_w     = wn_upload_conv_k_bf16(st, "decoder.head.2.weight", 12, 256, 27);
  v->head_b     = wn_upload_f32(st, "decoder.head.2.bias");
  if (!v->conv2_w || !v->conv1_w || !v->attn_qkv_w || !v->head_w) { free_vae_res(v); return -1; }
  if (load_vae_rb(st, "decoder.middle.0", 1024, 1024, &v->mid0) != 0 ||
      load_vae_rb(st, "decoder.middle.2", 1024, 1024, &v->mid2) != 0) { free_vae_res(v); return -1; }

  const int ic[4] = {1024, 1024, 1024, 512};
  const int oc[4] = {1024, 1024,  512, 256};
  const int uf[4] = {1, 1, 1, 0};
  const int tf[4] = {1, 1, 0, 0};
  char pfx[192], k[192];
  for (int i = 0; i < 4; i++) {
    wn_vae_up_t *U = &v->up[i];
    U->in_c = ic[i]; U->out_c = oc[i]; U->up_flag = uf[i]; U->t_up_flag = tf[i];
    for (int j = 0; j < 3; j++) {
      snprintf(pfx, sizeof pfx, "decoder.upsamples.%d.upsamples.%d", i, j);
      if (load_vae_rb(st, pfx, j == 0 ? ic[i] : oc[i], oc[i], &U->r[j]) != 0) {
        free_vae_res(v); return -1;
      }
    }
    if (uf[i]) {
      snprintf(k, sizeof k, "decoder.upsamples.%d.upsamples.3.resample.1.weight", i);
      U->sp_w = wn_upload_conv_k_bf16(st, k, oc[i], oc[i], 9);
      snprintf(k, sizeof k, "decoder.upsamples.%d.upsamples.3.resample.1.bias", i);
      U->sp_b = wn_upload_f32(st, k);
      if (!U->sp_w || !U->sp_b) { free_vae_res(v); return -1; }
      if (tf[i]) {
        snprintf(k, sizeof k, "decoder.upsamples.%d.upsamples.3.time_conv.weight", i);
        U->tm_w = wn_upload_conv_k_bf16(st, k, 2 * oc[i], oc[i], 3);
        snprintf(k, sizeof k, "decoder.upsamples.%d.upsamples.3.time_conv.bias", i);
        U->tm_b = wn_upload_f32(st, k);
        if (!U->tm_w || !U->tm_b) { free_vae_res(v); return -1; }
      }
    }
  }
  v->loaded = 1;
  return 0;
}

// ===========================================================================
// 8. Stage 1: Native CUDA UMT5-XXL FP8-Scaled 24-Layer Text Encoder
// ===========================================================================
static int run_umt5_xxl(
    ec_wan22_t *w, const char *prompt,
    uint16_t *out_512_bf16_d, // [512, 4096] BF16 on GPU (zero-padded beyond L_txt!)
    double *out_ms) {
  double t0 = wn_now_ms();
  const char *pkey = prompt ? prompt : "";
  for (int i = 0; i < WN_T5_CACHE_SLOTS; i++) {
    if (w->t5_valid[i] && w->t5_vals[i] && strcmp(w->t5_keys[i], pkey) == 0) {
      cudaMemcpyAsync(out_512_bf16_d, w->t5_vals[i], 512L * 4096 * 2, cudaMemcpyHostToDevice, w->stream);
      cudaStreamSynchronize(w->stream);
      if (out_ms) *out_ms = wn_now_ms() - t0;
      return 0;
    }
  }
  wn_init_fp8_lut();
  const ec_safetensors_t *st = &w->te_st;
  const int DIM = 4096, H = 64, FFN = 10240;

  int32_t ids[512];
  int L = ec_unigram_encode(w->ug, prompt ? prompt : "", ids, 512);
  if (L <= 0) return -1;

  // Lookup token embeddings from shared.weight F16 [256384, 4096]
  const ec_st_entry_t *es = ec_st_find(st, "shared.weight");
  if (!es) return -1;
  const uint16_t *emb = (const uint16_t *)ec_st_ptr(st, es);
  int vocab = (int)es->shape[0];
  float *x_h = (float *)malloc((size_t)L * DIM * 4);
  for (int i = 0; i < L; i++) {
    int id = ids[i];
    if (id < 0 || id >= vocab) id = 3;
    const uint16_t *row = emb + (long)id * DIM;
    for (int c = 0; c < DIM; c++) x_h[(long)i * DIM + c] = wn_f16_to_f32_h(row[c]);
  }

  float *lut_d = NULL, *x_d = NULL, *q_d = NULL, *k_d = NULL, *v_d = NULL;
  float *tmp_d = NULL, *w0_d = NULL, *w1_d = NULL;
  uint16_t *xn_bf16 = NULL, *ao_bf16 = NULL, *mlp_bf16 = NULL;
  cudaMalloc(&lut_d,    256 * 4);
  cudaMalloc(&x_d,      (size_t)L * DIM * 4);
  cudaMalloc(&xn_bf16,  (size_t)L * DIM * 2);
  cudaMalloc(&q_d,      (size_t)L * DIM * 4);
  cudaMalloc(&k_d,      (size_t)L * DIM * 4);
  cudaMalloc(&v_d,      (size_t)L * DIM * 4);
  cudaMalloc(&ao_bf16,  (size_t)L * DIM * 2);
  cudaMalloc(&tmp_d,    (size_t)L * DIM * 4);
  cudaMalloc(&w0_d,     (size_t)L * FFN * 4);
  cudaMalloc(&w1_d,     (size_t)L * FFN * 4);
  cudaMalloc(&mlp_bf16, (size_t)L * FFN * 2);
  cudaMemcpyAsync(lut_d, g_fp8_lut, 256 * 4, cudaMemcpyHostToDevice, w->stream);
  cudaMemcpyAsync(x_d, x_h, (size_t)L * DIM * 4, cudaMemcpyHostToDevice, w->stream);
  free(x_h);

  // Stream 24 UMT5-XXL FP8-scaled blocks
  char kw[192], ks[192];
  for (int l = 0; l < 24; l++) {
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.layer_norm.weight", l);
    float *ln0 = wn_upload_f32(st, kw);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.SelfAttention.relative_attention_bias.weight", l);
    float *rb  = wn_upload_f32(st, kw);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.SelfAttention.q.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.0.SelfAttention.q.scale_weight", l);
    uint16_t *wq = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.SelfAttention.k.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.0.SelfAttention.k.scale_weight", l);
    uint16_t *wk = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.SelfAttention.v.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.0.SelfAttention.v.scale_weight", l);
    uint16_t *wv = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.0.SelfAttention.o.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.0.SelfAttention.o.scale_weight", l);
    uint16_t *wo = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);

    k_wn_rmsnorm<<<L, 256, 0, w->stream>>>(x_d, ln0, NULL, xn_bf16, L, DIM, 1e-6f, 0);
    wn_gemm(xn_bf16, wq, NULL, q_d, NULL, L, DIM, DIM, w->stream);
    wn_gemm(xn_bf16, wk, NULL, k_d, NULL, L, DIM, DIM, w->stream);
    wn_gemm(xn_bf16, wv, NULL, v_d, NULL, L, DIM, DIM, w->stream);
    int nw = L * H;
    k_wn_t5_attn_d64<<<(nw + 7) / 8, 256, 0, w->stream>>>(q_d, k_d, v_d, rb, ao_bf16, L, H);
    wn_gemm(ao_bf16, wo, NULL, tmp_d, NULL, L, DIM, DIM, w->stream);
    k_wn_add_inplace<<<((long)L * DIM + 255) / 256, 256, 0, w->stream>>>(x_d, tmp_d, (long)L * DIM);
    cudaStreamSynchronize(w->stream);
    cudaFree(ln0); cudaFree(rb); cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo);

    snprintf(kw, sizeof kw, "encoder.block.%d.layer.1.layer_norm.weight", l);
    float *ln1 = wn_upload_f32(st, kw);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.1.DenseReluDense.wi_0.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.1.DenseReluDense.wi_0.scale_weight", l);
    uint16_t *w0 = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.1.DenseReluDense.wi_1.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.1.DenseReluDense.wi_1.scale_weight", l);
    uint16_t *w1 = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);
    snprintf(kw, sizeof kw, "encoder.block.%d.layer.1.DenseReluDense.wo.weight", l);
    snprintf(ks, sizeof ks, "encoder.block.%d.layer.1.DenseReluDense.wo.scale_weight", l);
    uint16_t *wd = wn_upload_fp8_bf16(st, kw, ks, lut_d, w->stream);

    k_wn_rmsnorm<<<L, 256, 0, w->stream>>>(x_d, ln1, NULL, xn_bf16, L, DIM, 1e-6f, 0);
    wn_gemm(xn_bf16, w0, NULL, w0_d, NULL, L, FFN, DIM, w->stream);
    wn_gemm(xn_bf16, w1, NULL, w1_d, NULL, L, FFN, DIM, w->stream);
    k_wn_t5_geglu_bf16<<<((long)L * FFN + 255) / 256, 256, 0, w->stream>>>(w0_d, w1_d, mlp_bf16, (long)L * FFN);
    wn_gemm(mlp_bf16, wd, NULL, tmp_d, NULL, L, DIM, FFN, w->stream);
    k_wn_add_inplace<<<((long)L * DIM + 255) / 256, 256, 0, w->stream>>>(x_d, tmp_d, (long)L * DIM);
    cudaStreamSynchronize(w->stream);
    cudaFree(ln1); cudaFree(w0); cudaFree(w1); cudaFree(wd);
  }

  float *fnorm = wn_upload_f32(st, "encoder.final_layer_norm.weight");
  cudaMemsetAsync(out_512_bf16_d, 0, 512L * DIM * 2, w->stream);
  k_wn_rmsnorm<<<L, 256, 0, w->stream>>>(x_d, fnorm, NULL, out_512_bf16_d, L, DIM, 1e-6f, 0);
  cudaStreamSynchronize(w->stream);

  // Store host copy in 4-slot UMT5-XXL prompt cache (eliminates ~2.5s neg-prompt re-encode on every run!)
  if (strlen(pkey) < sizeof(w->t5_keys[0])) {
    int slot = w->t5_next_slot % WN_T5_CACHE_SLOTS;
    if (!w->t5_vals[slot]) w->t5_vals[slot] = (uint16_t *)malloc(512L * DIM * 2);
    if (w->t5_vals[slot]) {
      cudaMemcpy(w->t5_vals[slot], out_512_bf16_d, 512L * DIM * 2, cudaMemcpyDeviceToHost);
      snprintf(w->t5_keys[slot], sizeof(w->t5_keys[slot]), "%s", pkey);
      w->t5_valid[slot] = 1;
      w->t5_next_slot = (slot + 1) % WN_T5_CACHE_SLOTS;
    }
  }

  cudaFree(fnorm); cudaFree(lut_d);
  cudaFree(x_d); cudaFree(xn_bf16); cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(ao_bf16); cudaFree(tmp_d); cudaFree(w0_d); cudaFree(w1_d); cudaFree(mlp_bf16);
  if (out_ms) *out_ms = wn_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 9. Stage 2: Native CUDA 30-Block WanModel TI2V-5B DiT (Shift=5.0 Euler + CFG)
// ===========================================================================
static uint64_t wn_splitmix64(uint64_t *s) {
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static float wn_randn(uint64_t *rng) {
  double u1 = ((double)(wn_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  double u2 = ((double)(wn_splitmix64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  return (float)(sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2));
}

typedef struct {
  uint16_t *patch_bf16, *xn_bf16, *qr_b, *kr_b, *vr_b, *ao_bf16, *ffn_bf16;
  uint16_t *t_f_d, *t_hb_d;
  float *x_d, *qkv_d, *tmp_d, *ffn_d, *head_d;
  float *e_d, *e0_d, *mod_d, *hmod_d, *t_hf_d;
} wn_dit_work_t;

static void compute_wan_time(ec_wan22_t *w, float t_val, wn_dit_work_t *W, float *out_e_d, float *out_e0_d) {
  const wn_dit_res_t *d = &w->dit;
  uint16_t fb_h[256];
  double tv = (double)t_val;
  for (int i = 0; i < 128; i++) {
    double a = tv * pow(10000.0, -(double)i / 128.0);
    fb_h[i]       = wn_f32_to_bf16_h((float)cos(a));
    fb_h[128 + i] = wn_f32_to_bf16_h((float)sin(a));
  }
  cudaMemcpyAsync(W->t_f_d, fb_h, 256 * 2, cudaMemcpyHostToDevice, w->stream);
  wn_gemm(W->t_f_d, d->t0_w, d->t0_b, W->t_hf_d, NULL, 1, 3072, 256, w->stream);
  ec_ops_silu_f32_to_bf16(W->t_hf_d, W->t_hb_d, 3072, w->stream);
  wn_gemm(W->t_hb_d, d->t2_w, d->t2_b, out_e_d, NULL, 1, 3072, 3072, w->stream);
  ec_ops_silu_f32_to_bf16(out_e_d, W->t_hb_d, 3072, w->stream);
  wn_gemm(W->t_hb_d, d->tproj_w, d->tproj_b, out_e0_d, NULL, 1, 18432, 3072, w->stream);
}

static void wan_dit_forward(
    ec_wan22_t *w, const float *xt_d, float t_input,
    int F, int Hp, int Wp,
    uint16_t *const ck_cache[30], uint16_t *const cv_cache[30],
    wn_dit_work_t *W, float *out_v_d) {
  const wn_dit_res_t *d = &w->dit;
  const int N = F * Hp * Wp, DIM = 3072, H = 24, FFN = 14336, Lctx = 512;
  const float scale = 1.0f / sqrtf(128.0f);

  compute_wan_time(w, t_input, W, W->e_d, W->e0_d);
  k_wn_patchify<<<((long)N * 192 + 255) / 256, 256, 0, w->stream>>>(xt_d, W->patch_bf16, F, Hp, Wp);
  wn_gemm(W->patch_bf16, d->patch_w, d->patch_b, W->x_d, NULL, N, DIM, 192, w->stream);

  int nw = N * H;

  for (int l = 0; l < 30; l++) {
    const wn_dit_blk_t *B = &d->blocks[l];
    k_wn_add_vec<<<(18432 + 255) / 256, 256, 0, w->stream>>>(B->mod, W->e0_d, W->mod_d, 18432);
    const float *m0 = W->mod_d + 0 * DIM;
    const float *m1 = W->mod_d + 1 * DIM;
    const float *m2 = W->mod_d + 2 * DIM;
    const float *m3 = W->mod_d + 3 * DIM;
    const float *m4 = W->mod_d + 4 * DIM;
    const float *m5 = W->mod_d + 5 * DIM;

    // 1. Self-Attention: LN + AdaLN -> 1 Fused QKV GEMM -> 1 Fused WanRMSNorm+3DRoPE+BF16 -> TensorCore BF16 FA2!
    k_wn_layernorm<<<N, 256, 0, w->stream>>>(W->x_d, NULL, NULL, m1, m0, W->xn_bf16, N, DIM, 1e-6f);
    wn_gemm(W->xn_bf16, B->sa_qkv_w, B->sa_qkv_b, W->qkv_d, NULL, N, 3 * DIM, DIM, w->stream);
    k_wn_sa_qkv_norm_rope_bf16<<<N, 256, 0, w->stream>>>(
        W->qkv_d, B->sa_nq, B->sa_nk, W->qr_b, W->kr_b, W->vr_b, N, F, Hp, Wp);
    ec_ops_fa2_bf16(W->qr_b, W->kr_b, W->vr_b, NULL, W->ao_bf16, N, N, N, N, H, H, 128, scale, 0, w->stream);
    wn_gemm(W->ao_bf16, B->sa_o_w, B->sa_o_b, W->tmp_d, NULL, N, DIM, DIM, w->stream);
    k_wn_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, w->stream>>>(W->x_d, W->tmp_d, m2, N, DIM);

    // 2. Cross-Attention: affine norm3 -> q + WanRMSNorm(3072) -> TensorCore BF16 FA2 against cached KV -> residual
    k_wn_layernorm<<<N, 256, 0, w->stream>>>(W->x_d, B->norm3_w, B->norm3_b, NULL, NULL, W->xn_bf16, N, DIM, 1e-6f);
    wn_gemm(W->xn_bf16, B->ca_q_w, B->ca_q_b, W->qkv_d, NULL, N, DIM, DIM, w->stream);
    k_wn_rmsnorm<<<N, 256, 0, w->stream>>>(W->qkv_d, B->ca_nq, W->qkv_d, NULL, N, DIM, 1e-6f, 0);
    k_wn_transpose_bf16<<<(nw + 7) / 8, 256, 0, w->stream>>>(W->qkv_d, W->qr_b, N, H);
    ec_ops_fa2_bf16(W->qr_b, ck_cache[l], cv_cache[l], NULL, W->ao_bf16, N, Lctx, N, Lctx, H, H, 128, scale, 0, w->stream);
    wn_gemm(W->ao_bf16, B->ca_o_w, B->ca_o_b, W->tmp_d, NULL, N, DIM, DIM, w->stream);
    k_wn_add_inplace<<<((long)N * DIM + 255) / 256, 256, 0, w->stream>>>(W->x_d, W->tmp_d, (long)N * DIM);

    // 3. FFN: LN(no affine) + AdaLN -> 3072->14336 GELU(tanh) -> 14336->3072 -> gated residual
    k_wn_layernorm<<<N, 256, 0, w->stream>>>(W->x_d, NULL, NULL, m4, m3, W->xn_bf16, N, DIM, 1e-6f);
    wn_gemm(W->xn_bf16, B->ffn0_w, B->ffn0_b, W->ffn_d, NULL, N, FFN, DIM, w->stream);
    k_wn_gelu_tanh_bf16<<<((long)N * FFN + 255) / 256, 256, 0, w->stream>>>(W->ffn_d, W->ffn_bf16, (long)N * FFN);
    wn_gemm(W->ffn_bf16, B->ffn2_w, B->ffn2_b, W->tmp_d, NULL, N, DIM, FFN, w->stream);
    k_wn_add_scaled<<<((long)N * DIM + 255) / 256, 256, 0, w->stream>>>(W->x_d, W->tmp_d, m5, N, DIM);
  }

  // Head + Unpatchify
  k_wn_add_head_mod<<<(2 * DIM + 255) / 256, 256, 0, w->stream>>>(d->head_mod, W->e_d, W->hmod_d, DIM);
  k_wn_layernorm<<<N, 256, 0, w->stream>>>(W->x_d, NULL, NULL, W->hmod_d + DIM, W->hmod_d, W->xn_bf16, N, DIM, 1e-6f);
  wn_gemm(W->xn_bf16, d->head_w, d->head_b, W->head_d, NULL, N, 192, DIM, w->stream);
  k_wn_unpatchify<<<((long)N * 192 + 255) / 256, 256, 0, w->stream>>>(W->head_d, out_v_d, F, Hp, Wp);
}

static int run_wan_dit(
    ec_wan22_t *w, const uint16_t *t5_pos_bf16, const uint16_t *t5_neg_bf16,
    int F, int Hp, int Wp, int steps, float cfg_scale, float shift, uint32_t seed,
    float *out_z_d, // [F, 2*Hp, 2*Wp, 48] F32 on GPU
    ec_wan22_progress_fn cb, void *cb_ctx, double *out_ms) {
  double t0 = wn_now_ms();
  const wn_dit_res_t *d = &w->dit;
  const int N = F * Hp * Wp, DIM = 3072, H = 24, FFN = 14336, Lctx = 512;
  const long Z_ELEMS = (long)N * 192;

  // Project text embeddings ONCE: 4096 -> 3072 -> GELU -> 3072
  float *t_mid = NULL;
  uint16_t *t_mid_b = NULL, *ctx_pos_b = NULL, *ctx_neg_b = NULL;
  cudaMalloc(&t_mid,     (size_t)Lctx * DIM * 4);
  cudaMalloc(&t_mid_b,   (size_t)Lctx * DIM * 2);
  cudaMalloc(&ctx_pos_b, (size_t)Lctx * DIM * 2);
  cudaMalloc(&ctx_neg_b, (size_t)Lctx * DIM * 2);

  wn_gemm(t5_pos_bf16, d->txt0_w, d->txt0_b, t_mid, NULL, Lctx, DIM, 4096, w->stream);
  k_wn_gelu_tanh_bf16<<<((long)Lctx * DIM + 255) / 256, 256, 0, w->stream>>>(t_mid, t_mid_b, (long)Lctx * DIM);
  wn_gemm(t_mid_b, d->txt2_w, d->txt2_b, NULL, ctx_pos_b, Lctx, DIM, DIM, w->stream);

  wn_gemm(t5_neg_bf16, d->txt0_w, d->txt0_b, t_mid, NULL, Lctx, DIM, 4096, w->stream);
  k_wn_gelu_tanh_bf16<<<((long)Lctx * DIM + 255) / 256, 256, 0, w->stream>>>(t_mid, t_mid_b, (long)Lctx * DIM);
  wn_gemm(t_mid_b, d->txt2_w, d->txt2_b, NULL, ctx_neg_b, Lctx, DIM, DIM, w->stream);
  cudaFree(t_mid); cudaFree(t_mid_b);

  // Precompute cross-attn K/V cache in BF16 across all 30 blocks (~360 MB total — HALF previous VRAM!)
  uint16_t *ck_pos[30], *cv_pos[30], *ck_neg[30], *cv_neg[30];
  float *kv_tmp = NULL;
  cudaMalloc(&kv_tmp, (size_t)Lctx * DIM * 4);
  int nw_c = Lctx * H;
  for (int l = 0; l < 30; l++) {
    const wn_dit_blk_t *B = &d->blocks[l];
    cudaMalloc(&ck_pos[l], (size_t)Lctx * DIM * 2);
    cudaMalloc(&cv_pos[l], (size_t)Lctx * DIM * 2);
    cudaMalloc(&ck_neg[l], (size_t)Lctx * DIM * 2);
    cudaMalloc(&cv_neg[l], (size_t)Lctx * DIM * 2);

    wn_gemm(ctx_pos_b, B->ca_k_w, B->ca_k_b, kv_tmp, NULL, Lctx, DIM, DIM, w->stream);
    k_wn_rmsnorm<<<Lctx, 256, 0, w->stream>>>(kv_tmp, B->ca_nk, kv_tmp, NULL, Lctx, DIM, 1e-6f, 0);
    k_wn_transpose_bf16<<<(nw_c + 7) / 8, 256, 0, w->stream>>>(kv_tmp, ck_pos[l], Lctx, H);
    wn_gemm(ctx_pos_b, B->ca_v_w, B->ca_v_b, kv_tmp, NULL, Lctx, DIM, DIM, w->stream);
    k_wn_transpose_bf16<<<(nw_c + 7) / 8, 256, 0, w->stream>>>(kv_tmp, cv_pos[l], Lctx, H);

    wn_gemm(ctx_neg_b, B->ca_k_w, B->ca_k_b, kv_tmp, NULL, Lctx, DIM, DIM, w->stream);
    k_wn_rmsnorm<<<Lctx, 256, 0, w->stream>>>(kv_tmp, B->ca_nk, kv_tmp, NULL, Lctx, DIM, 1e-6f, 0);
    k_wn_transpose_bf16<<<(nw_c + 7) / 8, 256, 0, w->stream>>>(kv_tmp, ck_neg[l], Lctx, H);
    wn_gemm(ctx_neg_b, B->ca_v_w, B->ca_v_b, kv_tmp, NULL, Lctx, DIM, DIM, w->stream);
    k_wn_transpose_bf16<<<(nw_c + 7) / 8, 256, 0, w->stream>>>(kv_tmp, cv_neg[l], Lctx, H);
  }
  cudaFree(kv_tmp); cudaFree(ctx_pos_b); cudaFree(ctx_neg_b);

  // Working buffers (100% GPU-resident, zero per-step cudaMalloc/sync!)
  wn_dit_work_t W;
  cudaMalloc(&W.patch_bf16, (size_t)Z_ELEMS * 2);
  cudaMalloc(&W.x_d,        (size_t)N * DIM * 4);
  cudaMalloc(&W.xn_bf16,    (size_t)N * DIM * 2);
  cudaMalloc(&W.qkv_d,      (size_t)N * 3 * DIM * 4);
  cudaMalloc(&W.qr_b,       (size_t)N * DIM * 2);
  cudaMalloc(&W.kr_b,       (size_t)N * DIM * 2);
  cudaMalloc(&W.vr_b,       (size_t)N * DIM * 2);
  cudaMalloc(&W.ao_bf16,    (size_t)N * DIM * 2);
  cudaMalloc(&W.tmp_d,      (size_t)N * DIM * 4);
  cudaMalloc(&W.ffn_d,      (size_t)N * FFN * 4);
  cudaMalloc(&W.ffn_bf16,   (size_t)N * FFN * 2);
  cudaMalloc(&W.head_d,     (size_t)Z_ELEMS * 4);
  cudaMalloc(&W.e_d,         DIM * 4);
  cudaMalloc(&W.e0_d,        6 * DIM * 4);
  cudaMalloc(&W.mod_d,       6 * DIM * 4);
  cudaMalloc(&W.hmod_d,      2 * DIM * 4);
  cudaMalloc(&W.t_f_d,       256 * 2);
  cudaMalloc(&W.t_hf_d,      3072 * 4);
  cudaMalloc(&W.t_hb_d,      3072 * 2);

  // Initial noise xt -> upload ONCE to GPU!
  float *xt_h = (float *)malloc((size_t)Z_ELEMS * 4);
  uint64_t rng = (uint64_t)seed * 6364136223846793005ULL + 0xfeedfacecafebeefULL;
  for (long i = 0; i < Z_ELEMS; i++) xt_h[i] = wn_randn(&rng);

  float *xt_d = NULL, *vp_d = NULL, *vn_d = NULL, *delta_d = NULL;
  cudaMalloc(&xt_d,    (size_t)Z_ELEMS * 4);
  cudaMalloc(&vp_d,    (size_t)Z_ELEMS * 4);
  cudaMalloc(&vn_d,    (size_t)Z_ELEMS * 4);
  cudaMalloc(&delta_d, (size_t)Z_ELEMS * 4);
  cudaMemcpyAsync(xt_d, xt_h, (size_t)Z_ELEMS * 4, cudaMemcpyHostToDevice, w->stream);
  free(xt_h);

  // Shifted Flow Matching Euler Schedule
  float sig[65];
  for (int i = 0; i <= steps; i++) {
    float u = 1.0f - (float)i / (float)steps;
    sig[i] = shift * u / (1.0f + (shift - 1.0f) * u);
  }

  // Tolerance Doctrine: EC_EXACT=1 forces full 2*steps CFG oracle; default enables FasterCache mid-step CFG delta reuse!
  const char *env_ex = getenv("EC_EXACT");
  const int exact_mode = (env_ex && atoi(env_ex) == 1) ? 1 : 0;

  int cancelled = 0;
  for (int s = 0; s < steps; s++) {
    if (cb && cb(s + 1, steps + 3, "wan22_dit_step", cb_ctx) != 0) {
      cancelled = 1;
      break;
    }
    float t_cur = sig[s], t_next = sig[s + 1], dt = t_cur - t_next;
    wan_dit_forward(w, xt_d, t_cur * 1000.0f, F, Hp, Wp, ck_pos, cv_pos, &W, vp_d);

    if (cfg_scale > 1.01f) {
      // FasterCache (ICLR 2025): full CFG on warmup 0..3 & final 2 steps, alternating uncond delta reuse on mid steps
      int reuse_delta = (!exact_mode && steps >= 10 && s >= 4 && s < steps - 2 && (s & 1) == 1);
      if (!reuse_delta) {
        wan_dit_forward(w, xt_d, t_cur * 1000.0f, F, Hp, Wp, ck_neg, cv_neg, &W, vn_d);
        ec_ops_wan_cfg_euler(xt_d, vp_d, vn_d, delta_d, Z_ELEMS, dt, cfg_scale, 1, w->stream);
      } else {
        ec_ops_wan_cfg_euler(xt_d, vp_d, NULL, delta_d, Z_ELEMS, dt, cfg_scale, 2, w->stream);
      }
    } else {
      ec_ops_wan_cfg_euler(xt_d, vp_d, NULL, NULL, Z_ELEMS, dt, 1.0f, 0, w->stream);
    }
  }

  if (!cancelled) {
    cudaMemcpyAsync(out_z_d, xt_d, (size_t)Z_ELEMS * 4, cudaMemcpyDeviceToDevice, w->stream);
  }
  cudaStreamSynchronize(w->stream);

  for (int l = 0; l < 30; l++) {
    cudaFree(ck_pos[l]); cudaFree(cv_pos[l]); cudaFree(ck_neg[l]); cudaFree(cv_neg[l]);
  }
  cudaFree(W.patch_bf16); cudaFree(W.x_d); cudaFree(W.xn_bf16); cudaFree(W.qkv_d);
  cudaFree(W.qr_b); cudaFree(W.kr_b); cudaFree(W.vr_b);
  cudaFree(W.ao_bf16); cudaFree(W.tmp_d); cudaFree(W.ffn_d); cudaFree(W.ffn_bf16);
  cudaFree(W.head_d); cudaFree(W.e_d); cudaFree(W.e0_d); cudaFree(W.mod_d); cudaFree(W.hmod_d);
  cudaFree(W.t_f_d); cudaFree(W.t_hf_d); cudaFree(W.t_hb_d);
  cudaFree(xt_d); cudaFree(vp_d); cudaFree(vn_d); cudaFree(delta_d);

  if (out_ms) *out_ms = wn_now_ms() - t0;
  return cancelled ? -2 : 0;
}

// ===========================================================================
// 10. Stage 3: Native CUDA Causal 3D Wan2_2_VAE Streaming Decoder
// ===========================================================================
static const float WAN22_LATENT_MEAN[48] = {
  -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f,  0.0174f,  0.1838f,  0.1557f,
  -0.1382f,  0.0542f,  0.2813f,  0.0891f,  0.1570f, -0.0098f,  0.0375f, -0.1825f,
  -0.2246f, -0.1207f, -0.0698f,  0.5109f,  0.2665f, -0.2108f, -0.2158f,  0.2502f,
  -0.2055f, -0.0322f,  0.1109f,  0.1567f, -0.0729f,  0.0899f, -0.2799f, -0.1230f,
  -0.0313f, -0.1649f,  0.0117f,  0.0723f, -0.2839f, -0.2083f, -0.0520f,  0.3748f,
   0.0152f,  0.1957f,  0.1433f, -0.2944f,  0.3573f, -0.0548f, -0.1681f, -0.0667f
};
static const float WAN22_LATENT_STD[48] = {
   0.4765f,  1.0364f,  0.4514f,  1.1677f,  0.5313f,  0.4990f,  0.4818f,  0.5013f,
   0.8158f,  1.0344f,  0.5894f,  1.0901f,  0.6885f,  0.6165f,  0.8454f,  0.4978f,
   0.5759f,  0.3523f,  0.7135f,  0.6804f,  0.5833f,  1.4146f,  0.8986f,  0.5659f,
   0.7069f,  0.5338f,  0.4889f,  0.4917f,  0.4069f,  0.4999f,  0.6866f,  0.4093f,
   0.5709f,  0.6065f,  0.6415f,  0.4944f,  0.5726f,  1.2042f,  0.5458f,  1.6887f,
   0.3971f,  1.0600f,  0.3943f,  0.5537f,  0.5444f,  0.4089f,  0.7468f,  0.7744f
};

__global__ void k_wn_denorm_latent(float *z, const float *mean, const float *std, long n_pix) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = n_pix * 48;
  if (idx >= tot) return;
  int c = (int)(idx % 48);
  z[idx] = z[idx] * std[c] + mean[c];
}

// Per-CausalConv3d streaming cache slot (holds up to 2 temporal frames on GPU)
typedef struct {
  float *frames_d; // [2, H, W, C] F32 (or NULL if empty)
  int    n_valid;  // 0, 1, or 2; -1 means "Rep" sentinel for upsample3d time_conv
} wn_cache_slot_t;

static void vae_causal_conv3d(
    ec_wan22_t *w, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int T, int H, int W, int Cin, int Cout,
    wn_cache_slot_t *slot, uint16_t *col_scratch, size_t col_cap) {
  long frame_elems = (long)H * W * Cin;
  // Build temporally padded input [T + 2, H, W, Cin]
  float *x_pad = NULL;
  cudaMalloc(&x_pad, (size_t)(T + 2) * frame_elems * 4);
  if (slot->n_valid == 0 || !slot->frames_d) {
    cudaMemsetAsync(x_pad, 0, (size_t)2 * frame_elems * 4, w->stream);
  } else if (slot->n_valid == 1) {
    cudaMemsetAsync(x_pad, 0, (size_t)frame_elems * 4, w->stream);
    cudaMemcpyAsync(x_pad + frame_elems, slot->frames_d, (size_t)frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
  } else {
    cudaMemcpyAsync(x_pad, slot->frames_d, (size_t)2 * frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
  }
  cudaMemcpyAsync(x_pad + 2 * frame_elems, x_d, (size_t)T * frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);

  // Update cache slot with last 2 frames of (prev_cache + x_d)
  if (!slot->frames_d) cudaMalloc(&slot->frames_d, (size_t)2 * frame_elems * 4);
  if (T >= 2) {
    cudaMemcpyAsync(slot->frames_d, x_d + (long)(T - 2) * frame_elems, (size_t)2 * frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
    slot->n_valid = 2;
  } else {
    // T == 1: shift prev last frame into slot 0, new frame into slot 1
    if (slot->n_valid == 0) {
      cudaMemcpyAsync(slot->frames_d, x_d, (size_t)frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
      slot->n_valid = 1;
    } else if (slot->n_valid == 1) {
      cudaMemcpyAsync(slot->frames_d + frame_elems, x_d, (size_t)frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
      slot->n_valid = 2;
    } else {
      // copy slot[1] -> slot[0] safely via x_pad[1]
      cudaMemcpyAsync(slot->frames_d, x_pad + frame_elems, (size_t)frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
      cudaMemcpyAsync(slot->frames_d + frame_elems, x_d, (size_t)frame_elems * 4, cudaMemcpyDeviceToDevice, w->stream);
      slot->n_valid = 2;
    }
  }

  // Run chunked im2col + BF16 GEMM
  long total_p = (long)T * H * W;
  int max_ch = (int)(col_cap / (size_t)(27 * Cin));
  if (max_ch < 64) max_ch = 64;
  for (long p0 = 0; p0 < total_p; p0 += max_ch) {
    int ch = (total_p - p0 < max_ch) ? (int)(total_p - p0) : max_ch;
    long n = (long)ch * 27 * Cin;
    k_wn_im2col_c3d<<<(n + 255) / 256, 256, 0, w->stream>>>(x_pad, col_scratch, T, H, W, Cin, p0, ch);
    wn_gemm(col_scratch, w_bf16, b_f32, y_d + p0 * Cout, NULL, ch, Cout, 27 * Cin, w->stream);
  }
  cudaFree(x_pad);
}

static void vae_conv1x1(
    ec_wan22_t *w, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    long total_p, int Cin, int Cout, uint16_t *col_scratch, size_t col_cap) {
  int max_ch = (int)(col_cap / (size_t)Cin);
  if (max_ch < 256) max_ch = 256;
  for (long p0 = 0; p0 < total_p; p0 += max_ch) {
    int ch = (total_p - p0 < max_ch) ? (int)(total_p - p0) : max_ch;
    k_wn_f32_to_bf16<<<((long)ch * Cin + 255) / 256, 256, 0, w->stream>>>(x_d + p0 * Cin, col_scratch, (long)ch * Cin);
    wn_gemm(col_scratch, w_bf16, b_f32, y_d + p0 * Cout, NULL, ch, Cout, Cin, w->stream);
  }
}

static void vae_conv2d_k3(
    ec_wan22_t *w, const float *x_d, float *y_d,
    const uint16_t *w_bf16, const float *b_f32,
    int T, int H, int W, int C, uint16_t *col_scratch, size_t col_cap) {
  long total_p = (long)T * H * W;
  int max_ch = (int)(col_cap / (size_t)(9 * C));
  if (max_ch < 64) max_ch = 64;
  for (long p0 = 0; p0 < total_p; p0 += max_ch) {
    int ch = (total_p - p0 < max_ch) ? (int)(total_p - p0) : max_ch;
    long n = (long)ch * 9 * C;
    k_wn_im2col_2d_k3<<<(n + 255) / 256, 256, 0, w->stream>>>(x_d, col_scratch, T, H, W, C, p0, ch);
    wn_gemm(col_scratch, w_bf16, b_f32, y_d + p0 * C, NULL, ch, C, 9 * C, w->stream);
  }
}

static float *vae_run_resblk(
    ec_wan22_t *w, float *x_d, int T, int H, int W,
    const wn_vae_resblk_t *R, wn_cache_slot_t *caches, int *cidx,
    uint16_t *col_scratch, size_t col_cap) {
  long total_p = (long)T * H * W;
  float *h_d = NULL, *a_d = NULL, *b_d = NULL;
  cudaMalloc(&a_d, (size_t)total_p * R->in_c * 4);
  cudaMalloc(&b_d, (size_t)total_p * R->out_c * 4);
  if (R->has_sc) {
    cudaMalloc(&h_d, (size_t)total_p * R->out_c * 4);
    vae_conv1x1(w, x_d, h_d, R->sc_w, R->sc_b, total_p, R->in_c, R->out_c, col_scratch, col_cap);
  } else {
    h_d = x_d;
  }

  k_wn_rmsnorm<<<total_p, 256, 0, w->stream>>>(x_d, R->g0, a_d, NULL, (int)total_p, R->in_c, 1e-12f, 1);
  vae_causal_conv3d(w, a_d, b_d, R->w2, R->b2, T, H, W, R->in_c, R->out_c, &caches[(*cidx)++], col_scratch, col_cap);
  cudaFree(a_d);
  cudaMalloc(&a_d, (size_t)total_p * R->out_c * 4);
  k_wn_rmsnorm<<<total_p, 256, 0, w->stream>>>(b_d, R->g3, a_d, NULL, (int)total_p, R->out_c, 1e-12f, 1);
  vae_causal_conv3d(w, a_d, b_d, R->w6, R->b6, T, H, W, R->out_c, R->out_c, &caches[(*cidx)++], col_scratch, col_cap);
  k_wn_add_inplace<<<(total_p * R->out_c + 255) / 256, 256, 0, w->stream>>>(h_d, b_d, total_p * R->out_c);
  cudaFree(a_d); cudaFree(b_d);
  if (R->has_sc) cudaFree(x_d);
  return h_d;
}

static int run_wan_vae_decode(
    ec_wan22_t *w, float *z_d, int F_lat, int H_lat, int W_lat,
    uint8_t *out_rgb24_h, // host [(1 + 4*(F_lat-1)) * (16*H_lat) * (16*W_lat) * 3]
    double *out_ms) {
  double t0 = wn_now_ms();
  const wn_vae_res_t *v = &w->vae;
  const size_t COL_CAP = 32L * 1024 * 1024; // 64 MB BF16 scratch
  uint16_t *col = NULL;
  if (cudaMalloc(&col, COL_CAP * 2) != cudaSuccess) return -1;

  // 1. De-normalize z and run conv2 (1x1x1 48->48)
  float *mean_d = NULL, *std_d = NULL, *z_c2 = NULL;
  cudaMalloc(&mean_d, 48 * 4);
  cudaMalloc(&std_d,  48 * 4);
  cudaMemcpyAsync(mean_d, WAN22_LATENT_MEAN, 48 * 4, cudaMemcpyHostToDevice, w->stream);
  cudaMemcpyAsync(std_d,  WAN22_LATENT_STD,  48 * 4, cudaMemcpyHostToDevice, w->stream);
  long z_pix = (long)F_lat * H_lat * W_lat;
  k_wn_denorm_latent<<<(z_pix * 48 + 255) / 256, 256, 0, w->stream>>>(z_d, mean_d, std_d, z_pix);
  cudaFree(mean_d); cudaFree(std_d);

  cudaMalloc(&z_c2, (size_t)z_pix * 48 * 4);
  vae_conv1x1(w, z_d, z_c2, v->conv2_w, v->conv2_b, z_pix, 48, 48, col, COL_CAP);

  // 2. Allocate 48 streaming cache slots (actual count is 32)
  wn_cache_slot_t caches[48];
  memset(caches, 0, sizeof caches);

  int out_H = H_lat * 16, out_W = W_lat * 16;
  long rgb_frame_bytes = (long)out_H * out_W * 3;
  int out_frame_cursor = 0;

  // 3. Stream frame-by-frame i = 0 .. F_lat-1 (exact WanVAE_.decode!)
  for (int fi = 0; fi < F_lat; fi++) {
    int cidx = 0;
    int first_chunk = (fi == 0) ? 1 : 0;
    int T = 1, H = H_lat, W = W_lat;

    // decoder.conv1: 48 -> 1024
    float *cur = NULL;
    cudaMalloc(&cur, (size_t)H * W * 1024 * 4);
    vae_causal_conv3d(w, z_c2 + (long)fi * H * W * 48, cur, v->conv1_w, v->conv1_b,
                      T, H, W, 48, 1024, &caches[cidx++], col, COL_CAP);

    // decoder.middle.0 (ResBlock 1024)
    cur = vae_run_resblk(w, cur, T, H, W, &v->mid0, caches, &cidx, col, COL_CAP);

    // decoder.middle.1 (1-head spatial self-attention on T*H*W, C=1024)
    {
      int HW = H * W;
      float *norm_d = NULL, *qkv_d = NULL, *pr_d = NULL;
      uint16_t *ao_b = NULL;
      cudaMalloc(&norm_d, (size_t)HW * 1024 * 4);
      cudaMalloc(&qkv_d,  (size_t)HW * 3072 * 4);
      cudaMalloc(&ao_b,   (size_t)HW * 1024 * 2);
      cudaMalloc(&pr_d,   (size_t)HW * 1024 * 4);
      k_wn_rmsnorm<<<HW, 256, 0, w->stream>>>(cur, v->attn_g, norm_d, NULL, HW, 1024, 1e-12f, 0);
      vae_conv1x1(w, norm_d, qkv_d, v->attn_qkv_w, v->attn_qkv_b, HW, 1024, 3072, col, COL_CAP);
      size_t smem_bytes = (size_t)(HW + 32) * 4;
      k_wn_vae_attn_1h<<<HW, 256, smem_bytes, w->stream>>>(qkv_d, ao_b, HW, 1024);
      wn_gemm(ao_b, v->attn_pr_w, v->attn_pr_b, pr_d, NULL, HW, 1024, 1024, w->stream);
      k_wn_add_inplace<<<((long)HW * 1024 + 255) / 256, 256, 0, w->stream>>>(cur, pr_d, (long)HW * 1024);
      cudaFree(norm_d); cudaFree(qkv_d); cudaFree(ao_b); cudaFree(pr_d);
    }

    // decoder.middle.2 (ResBlock 1024)
    cur = vae_run_resblk(w, cur, T, H, W, &v->mid2, caches, &cidx, col, COL_CAP);

    // decoder.upsamples.0..3
    for (int u = 0; u < 4; u++) {
      const wn_vae_up_t *U = &v->up[u];
      float *x_in_copy = NULL;
      int T_in = T, H_in = H, W_in = W;
      if (U->up_flag) {
        cudaMalloc(&x_in_copy, (size_t)T * H * W * U->in_c * 4);
        cudaMemcpyAsync(x_in_copy, cur, (size_t)T * H * W * U->in_c * 4, cudaMemcpyDeviceToDevice, w->stream);
      }

      for (int r = 0; r < 3; r++) {
        cur = vae_run_resblk(w, cur, T, H, W, &U->r[r], caches, &cidx, col, COL_CAP);
      }

      if (U->up_flag) {
        int C = U->out_c;
        if (U->t_up_flag) {
          wn_cache_slot_t *tslot = &caches[cidx++];
          if (tslot->n_valid == 0) {
            // first_chunk: mark "Rep" sentinel (-1), skip time_conv!
            tslot->n_valid = -1;
          } else {
            long fe = (long)H * W * C;
            float *x_pad = NULL;
            cudaMalloc(&x_pad, (size_t)(T + 2) * fe * 4);
            if (tslot->n_valid == -1) {
              // cache == "Rep" -> prepend 2 zero frames!
              cudaMemsetAsync(x_pad, 0, (size_t)2 * fe * 4, w->stream);
            } else {
              cudaMemcpyAsync(x_pad, tslot->frames_d, (size_t)2 * fe * 4, cudaMemcpyDeviceToDevice, w->stream);
            }
            cudaMemcpyAsync(x_pad + 2 * fe, cur, (size_t)T * fe * 4, cudaMemcpyDeviceToDevice, w->stream);

            // Update cache for next chunk (matches wan22_vae.py lines 121-146!)
            if (!tslot->frames_d) cudaMalloc(&tslot->frames_d, (size_t)2 * fe * 4);
            if (T >= 2) {
              cudaMemcpyAsync(tslot->frames_d, cur + (long)(T - 2) * fe, (size_t)2 * fe * 4, cudaMemcpyDeviceToDevice, w->stream);
            } else {
              // T == 1
              if (tslot->n_valid == -1) {
                cudaMemsetAsync(tslot->frames_d, 0, (size_t)fe * 4, w->stream);
                cudaMemcpyAsync(tslot->frames_d + fe, cur, (size_t)fe * 4, cudaMemcpyDeviceToDevice, w->stream);
              } else {
                cudaMemcpyAsync(tslot->frames_d, x_pad + fe, (size_t)fe * 4, cudaMemcpyDeviceToDevice, w->stream);
                cudaMemcpyAsync(tslot->frames_d + fe, cur, (size_t)fe * 4, cudaMemcpyDeviceToDevice, w->stream);
              }
            }
            tslot->n_valid = 2;

            // Run time_conv (3x1x1 C -> 2*C)
            float *tc_out = NULL, *unshuf = NULL;
            cudaMalloc(&tc_out, (size_t)T * H * W * (2 * C) * 4);
            long total_p = (long)T * H * W;
            int max_ch = (int)(COL_CAP / (size_t)(3 * C));
            if (max_ch < 64) max_ch = 64;
            for (long p0 = 0; p0 < total_p; p0 += max_ch) {
              int ch = (total_p - p0 < max_ch) ? (int)(total_p - p0) : max_ch;
              k_wn_im2col_t3<<<((long)ch * 3 * C + 255) / 256, 256, 0, w->stream>>>(x_pad, col, T, H, W, C, p0, ch);
              wn_gemm(col, U->tm_w, U->tm_b, tc_out + p0 * (2 * C), NULL, ch, 2 * C, 3 * C, w->stream);
            }
            cudaFree(x_pad);
            cudaFree(cur);

            cudaMalloc(&unshuf, (size_t)(2 * T) * H * W * C * 4);
            k_wn_time_unshuffle<<<((long)(2 * T) * H * W * C + 255) / 256, 256, 0, w->stream>>>(tc_out, unshuf, T, H, W, C);
            cudaFree(tc_out);
            cur = unshuf;
            T *= 2;
          }
        }

        // Spatial 2x nearest + Conv2d 3x3
        float *up2d = NULL, *sp_out = NULL;
        cudaMalloc(&up2d,   (size_t)T * (2 * H) * (2 * W) * C * 4);
        cudaMalloc(&sp_out, (size_t)T * (2 * H) * (2 * W) * C * 4);
        k_wn_upsample2d_nearest<<<((long)T * 4 * H * W * C + 255) / 256, 256, 0, w->stream>>>(cur, up2d, T, H, W, C);
        cudaFree(cur);
        H *= 2; W *= 2;
        vae_conv2d_k3(w, up2d, sp_out, U->sp_w, U->sp_b, T, H, W, C, col, COL_CAP);
        cudaFree(up2d);

        // Add DupUp3D shortcut!
        float *sc_up = NULL;
        cudaMalloc(&sc_up, (size_t)T * H * W * C * 4);
        k_wn_dupup3d<<<((long)T * H * W * C + 255) / 256, 256, 0, w->stream>>>(
            x_in_copy, sc_up, T_in, H_in, W_in, U->in_c, C, U->t_up_flag ? 2 : 1, first_chunk);
        cudaFree(x_in_copy);
        k_wn_add_inplace<<<((long)T * H * W * C + 255) / 256, 256, 0, w->stream>>>(sp_out, sc_up, (long)T * H * W * C);
        cudaFree(sc_up);
        cur = sp_out;
      }
    }

    // decoder.head: RMS_norm(256) -> SiLU -> CausalConv3d(256->12) -> unpatchify(2) -> RGB24!
    long tot_p = (long)T * H * W;
    float *hnorm = NULL, *out12 = NULL;
    cudaMalloc(&hnorm, (size_t)tot_p * 256 * 4);
    cudaMalloc(&out12, (size_t)tot_p * 12 * 4);
    k_wn_rmsnorm<<<tot_p, 256, 0, w->stream>>>(cur, v->head_g, hnorm, NULL, (int)tot_p, 256, 1e-12f, 1);
    cudaFree(cur);
    vae_causal_conv3d(w, hnorm, out12, v->head_w, v->head_b, T, H, W, 256, 12, &caches[cidx++], col, COL_CAP);
    cudaFree(hnorm);

    uint8_t *rgb_d = NULL;
    cudaMalloc(&rgb_d, (size_t)T * rgb_frame_bytes);
    k_wn_unpatchify_rgb24<<<((long)T * rgb_frame_bytes + 255) / 256, 256, 0, w->stream>>>(out12, rgb_d, T, H, W);
    cudaFree(out12);

    cudaMemcpyAsync(out_rgb24_h + (long)out_frame_cursor * rgb_frame_bytes, rgb_d,
                    (size_t)T * rgb_frame_bytes, cudaMemcpyDeviceToHost, w->stream);
    cudaStreamSynchronize(w->stream);
    cudaFree(rgb_d);
    out_frame_cursor += T;
  }

  for (int i = 0; i < 48; i++) cudaFree(caches[i].frames_d);
  cudaFree(z_c2);
  cudaFree(col);
  if (out_ms) *out_ms = wn_now_ms() - t0;
  return 0;
}

// ===========================================================================
// 11. Public Lifecycle & End-to-End Video Generator
// ===========================================================================
extern "C" ec_wan22_t *ec_wan22_create(const ec_wan22_paths_t *paths) {
  if (!paths || !paths->dit_path[0] || !paths->vae_path[0] ||
      !paths->text_encoder_path[0] || !paths->tokenizer_path[0]) return NULL;
  ec_wan22_t *w = (ec_wan22_t *)calloc(1, sizeof *w);
  if (!w) return NULL;
  w->paths = *paths;
  if (cudaStreamCreate(&w->stream) != cudaSuccess) { free(w); return NULL; }
  char err[128] = {0};
  w->ug = ec_unigram_load(paths->tokenizer_path);
  if (!w->ug) { ec_wan22_destroy(w); return NULL; }
  if (ec_st_open(&w->dit_st, paths->dit_path, err, sizeof err) != 0)          { ec_wan22_destroy(w); return NULL; }
  w->dit_open = 1;
  if (ec_st_open(&w->vae_st, paths->vae_path, err, sizeof err) != 0)          { ec_wan22_destroy(w); return NULL; }
  w->vae_open = 1;
  if (ec_st_open(&w->te_st, paths->text_encoder_path, err, sizeof err) != 0)  { ec_wan22_destroy(w); return NULL; }
  w->te_open = 1;
  return w;
}

extern "C" void ec_wan22_unload_vram(ec_wan22_t *w) {
  if (!w) return;
  free_dit_res(&w->dit);
  free_vae_res(&w->vae);
}

extern "C" void ec_wan22_destroy(ec_wan22_t *w) {
  if (!w) return;
  ec_wan22_unload_vram(w);
  for (int i = 0; i < WN_T5_CACHE_SLOTS; i++) free(w->t5_vals[i]);
  if (w->dit_open) ec_st_close(&w->dit_st);
  if (w->vae_open) ec_st_close(&w->vae_st);
  if (w->te_open)  ec_st_close(&w->te_st);
  if (w->ug) ec_unigram_free(w->ug);
  if (w->stream) cudaStreamDestroy(w->stream);
  free(w);
}

extern "C" int ec_wan22_is_ready(const ec_wan22_t *w) {
  return (w && w->ug && w->dit_open && w->vae_open && w->te_open) ? 1 : 0;
}

extern "C" int ec_wan22_generate_mp4(
    ec_wan22_t *w,
    const ec_wan22_req_t *req,
    const char *out_mp4_path,
    const char *out_keyframe_png,
    const char *audio_wav_path,
    ec_wan22_progress_fn cb, void *cb_ctx,
    ec_wan22_res_t *res) {
  memset(res, 0, sizeof *res);
  if (!ec_wan22_is_ready(w) || !req || !out_mp4_path) {
    snprintf(res->error, sizeof res->error, "wan22 engine not initialized");
    return -1;
  }
  double t_all = wn_now_ms();

  int W = req->width  > 0 ? ((req->width  + 31) & ~31) : 832;
  int H = req->height > 0 ? ((req->height + 31) & ~31) : 480;
  if (W < 640) W = 640; if (W > 1280) W = 1280;
  if (H < 384) H = 384; if (H > 768)  H = 768;
  int req_f = req->num_frames > 0 ? req->num_frames : 17;
  int F_lat = (req_f - 1) / 4 + 1;
  if (F_lat < 2) F_lat = 2;
  if (F_lat > 21) F_lat = 21;
  int n_frames = 1 + 4 * (F_lat - 1);
  int fps   = (req->fps >= 8 && req->fps <= 60) ? req->fps : 16;
  int steps = (req->steps >= 4 && req->steps <= 50) ? req->steps : 24;
  float cfg = (req->cfg_scale > 0.0f) ? req->cfg_scale : 5.0f;
  float shf = (req->shift > 0.0f) ? req->shift : ((W * H <= 832 * 480) ? 3.0f : 5.0f);

  int H_lat = H / 16, W_lat = W / 16;
  int Hp = H_lat / 2, Wp = W_lat / 2;

  const char *pos_p = (req->prompt && req->prompt[0]) ? req->prompt : "A cinematic drone shot over snow-capped mountains at sunrise";
  const char *neg_p = (req->neg_prompt && req->neg_prompt[0])
      ? req->neg_prompt
      : "blurry, low quality, distorted, static, overexposed, watermark, text, ugly, jitter";

  // Stage 1: UMT5-XXL FP8-Scaled Text Encoding (before loading 9.4 GB DiT -> minimal peak VRAM!)
  if (cb) cb(0, steps + 3, "wan22_umt5_encode", cb_ctx);
  uint16_t *t5_pos_d = NULL, *t5_neg_d = NULL;
  cudaMalloc(&t5_pos_d, 512L * 4096 * 2);
  cudaMalloc(&t5_neg_d, 512L * 4096 * 2);
  double ms_p = 0, ms_n = 0;
  if (run_umt5_xxl(w, pos_p, t5_pos_d, &ms_p) != 0 ||
      run_umt5_xxl(w, neg_p, t5_neg_d, &ms_n) != 0) {
    cudaFree(t5_pos_d); cudaFree(t5_neg_d);
    snprintf(res->error, sizeof res->error, "umt5-xxl text encoder failed");
    return -1;
  }
  res->text_enc_ms = ms_p + ms_n;

  // Stage 2: Load & Run 30-Block WanModel TI2V-5B DiT
  if (ensure_dit_loaded(w) != 0) {
    cudaFree(t5_pos_d); cudaFree(t5_neg_d);
    snprintf(res->error, sizeof res->error, "wan22 dit load failed");
    return -1;
  }
  float *z_d = NULL;
  cudaMalloc(&z_d, (size_t)F_lat * H_lat * W_lat * 48 * 4);
  int rc = run_wan_dit(w, t5_pos_d, t5_neg_d, F_lat, Hp, Wp, steps, cfg, shf, req->seed, z_d, cb, cb_ctx, &res->dit_ms);
  cudaFree(t5_pos_d); cudaFree(t5_neg_d);
  if (rc != 0) {
    cudaFree(z_d);
    snprintf(res->error, sizeof res->error, rc == -2 ? "cancelled" : "wan22 dit denoising failed");
    return rc;
  }

  // Stage 3: Unload 9.4 GB DiT if VRAM is tight? Wait: VAE is only 1.4 GB + 300 MB scratch -> fits alongside DiT!
  // Let's keep DiT resident unless VAE load fails, in which case auto-evict DiT and retry!
  if (cb) cb(steps + 1, steps + 3, "wan22_causal_vae_decode", cb_ctx);
  if (ensure_vae_loaded(w) != 0) {
    free_dit_res(&w->dit);
    if (ensure_vae_loaded(w) != 0) {
      cudaFree(z_d);
      snprintf(res->error, sizeof res->error, "wan22 vae load failed");
      return -1;
    }
  }
  uint8_t *rgb_all = (uint8_t *)malloc((size_t)n_frames * W * H * 3);
  if (run_wan_vae_decode(w, z_d, F_lat, H_lat, W_lat, rgb_all, &res->vae_ms) != 0) {
    free(rgb_all); cudaFree(z_d);
    snprintf(res->error, sizeof res->error, "wan22 vae decode failed");
    return -1;
  }
  cudaFree(z_d);

  // Stage 4: Export Keyframe PNG + Hardware NVENC H.264 (+ optional AAC) MP4!
  if (cb) cb(steps + 2, steps + 3, "wan22_nvenc_mp4", cb_ctx);
  if (out_keyframe_png && out_keyframe_png[0]) {
    ec_export_png(out_keyframe_png, rgb_all, (uint32_t)W, (uint32_t)H);
  }
  double te0 = wn_now_ms();
  ec_mp4_writer_t *enc = ec_mp4_open(
      out_mp4_path, (audio_wav_path && audio_wav_path[0]) ? audio_wav_path : NULL,
      (uint32_t)W, (uint32_t)H, (uint32_t)fps, 1,
      res->encoder_used, sizeof res->encoder_used);
  if (!enc) {
    free(rgb_all);
    snprintf(res->error, sizeof res->error, "nvenc mp4 encoder open failed");
    return -1;
  }
  for (int f = 0; f < n_frames; f++) {
    ec_mp4_write_rgb24(enc, rgb_all + (size_t)f * W * H * 3, (size_t)W * H * 3u);
  }
  int close_rc = ec_mp4_close(enc);
  res->encode_ms = wn_now_ms() - te0;
  free(rgb_all);
  if (close_rc != 0) {
    snprintf(res->error, sizeof res->error, "nvenc mp4 finalize failed");
    return -1;
  }

  res->ok = 1;
  res->width = W;
  res->height = H;
  res->num_frames = n_frames;
  res->latent_frames = F_lat;
  res->steps_run = steps;
  res->total_ms = wn_now_ms() - t_all;
  return 0;
}
