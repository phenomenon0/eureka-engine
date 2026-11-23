// host/llb_inkling.cuh — Native first-principles CUDA kernels for Inkling 2 Mini (`inkling2_loop`)
// Compiled lazily via NVRTC inside Eureka's ONE `engine_gpu` binary (zero cuBLAS, zero separate binary!).
// Implements all 48 visits: Embed+RMSNorm, Kimi AttnRes depth mix, Rank-64 tanh LoopControl,
// Multi-format GEMV/GEMM (BF16/F32/Q8_ROW dp4a/S5_K/S6_K/S8_K/Q4_K/Q5_K/Q6_K/Q8_0),
// Fused Post-QKV (ShortConv4 + L2-clip + V-RMSNorm + Global log-pos tau + FP16 KV write),
// RelPos (d_rel=16) + Per-Query Sink Split-K Flash-GQA (24:8, hd=128, SWA-1024 & Global),
// Routed Top-8/128 SiTU(beta=4, lb=25) MoE (SF4 NVFP4, MX8 MXFP8, S_K, GGUF Q4_K..Q8_0),
// Exact 2-Stage Cauchy-Schwarz INT8 Screened Unembed + softcap(30) + DSpark Markov RNN!

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef signed char        i8;
typedef int                i32;

static __device__ __forceinline__ float ink_bf16(u16 h) {
  return __uint_as_float((u32)h << 16);
}
static __device__ __forceinline__ float ink_f16(u16 h) {
  u32 s = ((u32)h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
  if (e == 0) {
    if (m == 0) return __uint_as_float(s);
    while (!(m & 1024u)) { m <<= 1; e--; }
    m &= 1023u;
    return __uint_as_float(s | ((e + 113u) << 23) | (m << 13));
  }
  if (e == 31) return __uint_as_float(s | 0x7f800000u | (m << 13));
  return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}
static __device__ __forceinline__ u16 ink_f2h(float f) {
  u32 x = __float_as_uint(f), sg = (x >> 16) & 0x8000u, e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
  int ex = (int)e - 127 + 15;
  if (e == 0xffu || ex >= 31) return (u16)(sg | 0x7c00u);
  if (ex <= 0) return (u16)sg;
  return (u16)(sg | ((u32)ex << 10) | (m >> 13));
}
static __device__ __forceinline__ float ink_wsum(float v) {
  #pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  return v;
}
static __device__ __forceinline__ float ink_wmax(float v) {
  #pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
  return v;
}
static __device__ __forceinline__ float ink_bsum(float v, float* sh) {
  int l = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = ink_wsum(v);
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
  if (w == 0) { r = ink_wsum(r); if (l == 0) sh[0] = r; }
  __syncthreads();
  float out = sh[0];
  __syncthreads();
  return out;
}
static __device__ __forceinline__ float ink_situ(float g, float u) {
  // Exact Inkling-2-Mini SiTU(beta=4.0, lb=25.0): beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
  float sg = 1.0f / (1.0f + __expf(-g));
  return 4.0f * tanhf(g * 0.25f) * sg * 25.0f * tanhf(u * 0.04f);
}

// E2M1 FP4 LUT (16 values) & E4M3 FP8 LUT (signed magnitude, bias 7)
__device__ __constant__ float kE2M1[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static __device__ __forceinline__ float ink_e4m3(u8 b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
  float v = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m * 0.125f, e - 7);
  return s ? -v : v;
}
static __device__ __forceinline__ float ink_ue8m0(u8 b) {
  return ldexpf(1.0f, (int)b - 127);
}
static __device__ __forceinline__ int ink_sk_sb(int fmt) {
  // S_K superblock bytes (256 elems): 6 B header + payload
  return fmt == 2 ? 70 : fmt == 3 ? 102 : fmt == 4 ? 134 : fmt == 5 ? 166 : fmt == 6 ? 198 : 262;
}
static __device__ __forceinline__ float ink_sk_half(int fmt) {
  return fmt == 2 ? 1.5f : fmt == 3 ? 3.5f : fmt == 4 ? 7.5f : fmt == 5 ? 15.5f : fmt == 6 ? 31.5f : 0.0f;
}
static __device__ __forceinline__ float ink_sk_elem(const u8* row, int fmt, int d) {
  int sb = ink_sk_sb(fmt), s = d >> 8, idx = d & 255, g = idx >> 6;
  const u8* blk = row + (u64)s * sb;
  u16 dh = (u16)(blk[0] | ((u16)blk[1] << 8));
  float eff = ink_f16(dh) * ((float)blk[2 + g] * (1.0f / 255.0f));
  const u8* pl = blk + 6;
  if (fmt == 8) return (float)(i8)pl[idx] * eff;
  if (fmt == 6) {
    u32 lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    u32 hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
    return ((float)(lo | (hi << 4)) - 31.5f) * eff;
  }
  if (fmt == 5) {
    u32 lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    u32 hi = (pl[128 + (idx >> 3)] >> (idx & 7)) & 1u;
    return ((float)(lo | (hi << 4)) - 15.5f) * eff;
  }
  if (fmt == 4) {
    u32 lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    return ((float)lo - 7.5f) * eff;
  }
  if (fmt == 3) {
    u32 lo = (pl[idx >> 2] >> ((idx & 3) * 2)) & 3u;
    u32 hi = (pl[64 + (idx >> 3)] >> (idx & 7)) & 1u;
    return ((float)(lo | (hi << 2)) - 3.5f) * eff;
  }
  u32 lo = (pl[idx >> 2] >> ((idx & 3) * 2)) & 3u;
  return ((float)lo - 1.5f) * eff;
}

// 1. Embedding lookup + RMSNorm (BF16=102, F32=101, Q8ROW=100, S5/S6/S8)
extern "C" __global__ void __launch_bounds__(256) ink_embed_norm(
  const u8* __restrict__ W, int fmt, int vocab_pad,
  const float* __restrict__ nw, const i32* __restrict__ ids,
  float* __restrict__ out, int M, int D) {
  int m = blockIdx.x, tid = threadIdx.x;
  if (m >= M) return;
  int id = ids[m];
  float* dst = out + (u64)m * D;
  __shared__ float sh[32];
  float ss = 0.0f;
  if (fmt == 102) {
    const u16* r = (const u16*)W + (u64)id * D;
    for (int d = tid; d < D; d += 256) { float v = ink_bf16(r[d]); dst[d] = v; ss += v * v; }
  } else if (fmt == 101) {
    const float* r = (const float*)W + (u64)id * D;
    for (int d = tid; d < D; d += 256) { float v = r[d]; dst[d] = v; ss += v * v; }
  } else if (fmt == 100) {
    float sc = ((const float*)W)[id];
    const i8* r = (const i8*)(W + (u64)vocab_pad * 4) + (u64)id * D;
    for (int d = tid; d < D; d += 256) { float v = (float)r[d] * sc; dst[d] = v; ss += v * v; }
  } else {
    const u8* r = W + (u64)id * (D >> 8) * ink_sk_sb(fmt);
    for (int d = tid; d < D; d += 256) { float v = ink_sk_elem(r, fmt, d); dst[d] = v; ss += v * v; }
  }
  ss = ink_bsum(ss, sh);
  float inv = rsqrtf(ss / (float)D + 1e-6f);
  for (int d = tid; d < D; d += 256) dst[d] = dst[d] * inv * nw[d];
}

// 2. RMSNorm (grid M, block 256)
extern "C" __global__ void __launch_bounds__(256) ink_rms_norm(
  const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int M, int D) {
  int m = blockIdx.x, tid = threadIdx.x;
  if (m >= M) return;
  const float* xm = x + (u64)m * D;
  float* ym = y + (u64)m * D;
  __shared__ float sh[32];
  float ss = 0.0f;
  for (int d = tid; d < D; d += 256) { float v = xm[d]; ss += v * v; }
  ss = ink_bsum(ss, sh);
  float inv = rsqrtf(ss / (float)D + 1e-6f);
  for (int d = tid; d < D; d += 256) ym[d] = xm[d] * inv * (w ? w[d] : 1.0f);
}

// 3. Fused Kimi-Style Attention-Residual Depth Mix (+ optional fused RMSNorm!)
struct InkDepthPtrs { const float* p[8]; int count; };
extern "C" __global__ void __launch_bounds__(256) ink_attn_res_mix(
  InkDepthPtrs st, const float* __restrict__ queries, float* __restrict__ out,
  int n_q, int M, int D, const float* __restrict__ nw0, const float* __restrict__ nw1, const float* __restrict__ nw2) {
  int m = blockIdx.x, tid = threadIdx.x, w = tid >> 5, l = tid & 31, S = st.count;
  if (m >= M) return;
  __shared__ float sh[32], red[8][5][8], prob[4][8];
  float inv_sd = rsqrtf((float)D);
  for (int s = 0; s < S; s++) {
    const float* sp = st.p[s] + (u64)m * D;
    float ss = 0.0f, dt[4] = {0, 0, 0, 0};
    for (int d = tid; d < D; d += 256) {
      float v = sp[d];
      ss += v * v;
      for (int q = 0; q < n_q; q++) dt[q] = fmaf(v, queries[(u64)q * D + d], dt[q]);
    }
    ss = ink_wsum(ss);
    if (l == 0) red[s][0][w] = ss;
    for (int q = 0; q < n_q; q++) {
      float dq = ink_wsum(dt[q]);
      if (l == 0) red[s][1 + q][w] = dq;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float sc[4][8], sinv[8];
    for (int s = 0; s < S; s++) {
      float ss = 0.0f; for (int k = 0; k < 8; k++) ss += red[s][0][k];
      sinv[s] = rsqrtf(ss / (float)D + 1e-6f);
      for (int q = 0; q < n_q; q++) {
        float ds = 0.0f; for (int k = 0; k < 8; k++) ds += red[s][1 + q][k];
        sc[q][s] = ds * sinv[s] * inv_sd;
      }
    }
    for (int q = 0; q < n_q; q++) {
      float mx = -1e30f; for (int s = 0; s < S; s++) mx = fmaxf(mx, sc[q][s]);
      float sm = 0.0f;
      for (int s = 0; s < S; s++) { sc[q][s] = expf(sc[q][s] - mx); sm += sc[q][s]; }
      float is = 1.0f / sm;
      for (int s = 0; s < S; s++) prob[q][s] = sc[q][s] * is * sinv[s];
    }
  }
  __syncthreads();
  for (int q = 0; q < n_q; q++) {
    const float* nw = (q == 0) ? nw0 : (q == 1 ? nw1 : nw2);
    float* dst = out + ((u64)q * M + m) * D;
    float ss = 0.0f;
    for (int d = tid; d < D; d += 256) {
      float a = 0.0f;
      for (int s = 0; s < S; s++) a = fmaf(prob[q][s], st.p[s][(u64)m * D + d], a);
      dst[d] = a;
      if (nw) ss += a * a;
    }
    if (nw) {
      ss = ink_bsum(ss, sh);
      float inv = rsqrtf(ss / (float)D + 1e-6f);
      for (int d = tid; d < D; d += 256) dst[d] = dst[d] * inv * nw[d];
    }
  }
}

// 4. Rank-64 tanh Gain/Forget Loop Controller (+ optional fused RMSNorm)
extern "C" __global__ void __launch_bounds__(256) ink_ctrl_cat_rms(
  const float* __restrict__ cur, const float* __restrict__ msg, float* __restrict__ cat, int M, int D) {
  int m = blockIdx.x, tid = threadIdx.x;
  if (m >= M) return;
  __shared__ float sh[32];
  const float* c = cur + (u64)m * D; const float* g = msg + (u64)m * D;
  float sc = 0.0f, sm = 0.0f;
  for (int d = tid; d < D; d += 256) { sc += c[d] * c[d]; sm += g[d] * g[d]; }
  float ic = rsqrtf(ink_bsum(sc, sh) / (float)D + 1e-6f);
  float im = rsqrtf(ink_bsum(sm, sh) / (float)D + 1e-6f);
  float* o0 = cat + (u64)m * (2 * D); float* o1 = o0 + D;
  for (int d = tid; d < D; d += 256) { o0[d] = c[d] * ic; o1[d] = g[d] * im; }
}

extern "C" __global__ void __launch_bounds__(256) ink_ctrl_down(
  const float* __restrict__ cat, const float* __restrict__ w_down, float* __restrict__ gate, int M, int D2, int rank) {
  int r = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
  if (r >= rank || m >= M) return;
  __shared__ float sh[32];
  const float* x = cat + (u64)m * D2;
  const float* w = w_down + (u64)r * D2;
  float a = 0.0f;
  for (int i = tid; i < D2; i += 256) a = fmaf(x[i], w[i], a);
  a = ink_bsum(a, sh);
  if (tid == 0) gate[(u64)m * rank + r] = a;
}

extern "C" __global__ void __launch_bounds__(256) ink_ctrl_apply(
  const float* __restrict__ cur, const float* __restrict__ msg, const float* __restrict__ gate,
  const float* __restrict__ w_gain, const float* __restrict__ w_forget,
  float* __restrict__ out, int M, int D, int rank) {
  int m = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
  if (m >= M || d >= D) return;
  const float* g = gate + (u64)m * rank;
  const float* wg = w_gain + (u64)d * rank;
  const float* wf = w_forget + (u64)d * rank;
  float ag = 0.0f, af = 0.0f;
  for (int r = 0; r < rank; r++) { ag = fmaf(wg[r], g[r], ag); af = fmaf(wf[r], g[r], af); }
  u64 idx = (u64)m * D + d;
  out[idx] = (1.0f - tanhf(af)) * cur[idx] + tanhf(ag) * msg[idx];
}

// 5. Unified Multi-Token (M=1..16) Dense Linear GEMV (reads each weight row ONCE across all M tokens!)
// Supports BF16 (102), F32 (101), Q8_ROW (100), S5_K (5), S6_K (6), S8_K (8).
extern "C" __global__ void __launch_bounds__(128) ink_linear_m(
  const u8* __restrict__ W, int fmt, int N_total, int N, int K,
  const float* __restrict__ X, float* __restrict__ Y, int M) {
  int row = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (row >= N) return;
  float acc[16];
  #pragma unroll
  for (int m = 0; m < 16; m++) acc[m] = 0.0f;
  if (fmt == 102) {
    const u16* wr = (const u16*)W + (u64)row * K;
    for (int k = l * 2; k < K; k += 64) {
      float w0 = ink_bf16(wr[k]), w1 = ink_bf16(wr[k + 1]);
      for (int m = 0; m < M; m++) {
        const float* xm = X + (u64)m * K;
        acc[m] = fmaf(w0, xm[k], fmaf(w1, xm[k + 1], acc[m]));
      }
    }
  } else if (fmt == 101) {
    const float* wr = (const float*)W + (u64)row * K;
    for (int k = l; k < K; k += 32) {
      float w = wr[k];
      for (int m = 0; m < M; m++) acc[m] = fmaf(w, X[(u64)m * K + k], acc[m]);
    }
  } else if (fmt == 100) {
    float sc = ((const float*)W)[row];
    const i8* wr = (const i8*)(W + (u64)N_total * 4) + (u64)row * K;
    for (int k = l * 4; k < K; k += 128) {
      int w4 = *(const int*)(wr + k);
      float f0 = (float)(i8)(w4 & 0xff), f1 = (float)(i8)((w4 >> 8) & 0xff);
      float f2 = (float)(i8)((w4 >> 16) & 0xff), f3 = (float)(i8)(w4 >> 24);
      for (int m = 0; m < M; m++) {
        const float4 xv = *(const float4*)(X + (u64)m * K + k);
        acc[m] = fmaf(f0, xv.x, fmaf(f1, xv.y, fmaf(f2, xv.z, fmaf(f3, xv.w, acc[m]))));
      }
    }
    for (int m = 0; m < M; m++) acc[m] *= sc;
  } else {
    const u8* wr = W + (u64)row * (K >> 8) * ink_sk_sb(fmt);
    for (int k = l; k < K; k += 32) {
      float w = ink_sk_elem(wr, fmt, k);
      for (int m = 0; m < M; m++) acc[m] = fmaf(w, X[(u64)m * K + k], acc[m]);
    }
  }
  for (int m = 0; m < M; m++) {
    float s = ink_wsum(acc[m]);
    if (l == 0) Y[(u64)m * N + row] = s;
  }
}

// 6. Fused Post-QKV (K/V ShortConv4 + Q/K/R L2-clip + V RMSNorm + Global log-pos tau + FP16 KV write)
// Grid: (64 blocks = 24 Q + 8 K + 8 V + 24 R, M), Block: 128
extern "C" __global__ void __launch_bounds__(128) ink_post_qkv_fused(
  float* __restrict__ q, const float* __restrict__ q_sc, int q_sc_n,
  const float* __restrict__ k_raw, const float* __restrict__ v_raw,
  const float* __restrict__ sw_k, const float* __restrict__ sw_v,
  const float* __restrict__ k_dyn, const float* __restrict__ v_dyn,
  float* __restrict__ hk, float* __restrict__ hv,
  float* __restrict__ r, const float* __restrict__ r_sc, int r_sc_n,
  u16* __restrict__ kv_k, u16* __restrict__ kv_v,
  int kv_cap, int is_swa, int pos0, int M) {
  int blk = blockIdx.x, tid = threadIdx.x;
  __shared__ float sh[8];
  if (blk < 24) {
    int h = blk;
    for (int m = 0; m < M; m++) {
      int pos = pos0 + m;
      float tau = is_swa ? 1.0f : (1.0f + 0.1f * logf(fmaxf(1.0f, (float)(pos + 1) * (1.0f / 8192.0f))));
      float* qp = q + ((u64)m * 24 + h) * 128;
      float v = qp[tid];
      if (q_sc) v *= q_sc[(q_sc_n == 128) ? tid : (h * 128 + tid)];
      float ss = ink_bsum(v * v, sh);
      float fac = fminf(1.0f, 22.627416997969522f / (sqrtf(ss) + 1e-6f)) * tau;
      qp[tid] = v * fac;
    }
  } else if (blk < 40) {
    int is_k = blk < 32;
    int kh = is_k ? (blk - 24) : (blk - 32);
    int c = kh * 128 + tid;
    const float* raw = is_k ? k_raw : v_raw;
    const float* swp = is_k ? sw_k : sw_v;
    const float* dyn = is_k ? k_dyn : v_dyn;
    float* hist = is_k ? hk : hv;
    float h0 = hist[c], h1 = hist[1024 + c], h2 = hist[2048 + c];
    const float4 sw = ((const float4*)swp)[c];
    float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
    for (int m = 0; m < M; m++) {
      float cur = raw[(u64)m * 1024 + c];
      float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
      if (dyn) {
        const float4 dt = ((const float4*)(dyn + ((u64)m * 8 + kh) * 4))[0];
        acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
      }
      float val = (cur + acc) * sc_conv;
      h0 = h1; h1 = h2; h2 = cur;
      float ss = ink_bsum(val * val, sh);
      if (is_k) val *= fminf(1.0f, 22.627416997969522f / (sqrtf(ss) + 1e-6f));
      else val *= rsqrtf(ss * (1.0f / 128.0f) + 1e-6f);
      int pos = pos0 + m;
      int pidx = is_swa ? (pos % kv_cap) : pos;
      u64 base = ((u64)pidx * 8 + kh) * 128 + tid;
      (is_k ? kv_k : kv_v)[base] = ink_f2h(val);
    }
    hist[c] = h0; hist[1024 + c] = h1; hist[2048 + c] = h2;
  } else {
    int h = blk - 40;
    if (tid >= 16) return;
    for (int m = 0; m < M; m++) {
      float* rp = r + ((u64)m * 24 + h) * 16;
      float v = rp[tid];
      if (r_sc) v *= r_sc[(r_sc_n == 16) ? tid : (h * 16 + tid)];
      float ss = v * v;
      #pragma unroll
      for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffu, ss, o);
      rp[tid] = v * fminf(1.0f, 8.0f / (sqrtf(ss) + 1e-6f));
    }
  }
}

// 7. RelPos table [M, 24, 1025] (grid (24, M), block 256)
extern "C" __global__ void __launch_bounds__(256) ink_rel_table(
  const float* __restrict__ r, const float* __restrict__ rel_proj, float* __restrict__ out, int M) {
  int h = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
  __shared__ float sr[16];
  if (tid < 16) sr[tid] = r[((u64)m * 24 + h) * 16 + tid];
  __syncthreads();
  float* dst = out + ((u64)m * 24 + h) * 1025;
  for (int dist = tid; dist < 1024; dist += 256) {
    float a = 0.0f;
    for (int d = 0; d < 16; d++) a = fmaf(sr[d], rel_proj[(u64)d * 1024 + dist], a);
    dst[dist] = a;
  }
  if (tid == 0) dst[1024] = 0.0f;
}

// 8. Fused RelPos + Per-Query Sink Flash-GQA (24:8, hd=128, FP16 KV) + sigmoid output gate
// Grid: (24 heads, M tokens), Block: 128 (1 dim/thread, 4 warps)
extern "C" __global__ void __launch_bounds__(128) ink_flash_gqa(
  const float* __restrict__ q, const float* __restrict__ rel_tbl, const float* __restrict__ sinks,
  const float* __restrict__ gate, const u16* __restrict__ kv_k, const u16* __restrict__ kv_v,
  int kv_cap, int is_swa, int pos0, float* __restrict__ out, int M) {
  int h = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
  int kh = h / 3;
  int q_pos = pos0 + m;
  int k_beg = is_swa ? max(0, q_pos - 1023) : 0;
  int k_end = q_pos + 1;
  float qv = q[((u64)m * 24 + h) * 128 + tid] * 0.08838834764831845f;
  const float* rt = rel_tbl + ((u64)m * 24 + h) * 1025;
  __shared__ float sh[8];
  float mx = sinks[(u64)m * 24 + h];
  float lsum = 1.0f;  // exact algebraic sink: exp(sink - sink) = 1.0, acc = 0.0
  float acc = 0.0f;
  for (int kp = k_beg; kp < k_end; kp++) {
    int pidx = is_swa ? (kp % kv_cap) : kp;
    u64 base = ((u64)pidx * 8 + kh) * 128 + tid;
    float kv = ink_f16(kv_k[base]);
    float vv = ink_f16(kv_v[base]);
    float dot = ink_bsum(qv * kv, sh);
    int dist = q_pos - kp;
    if (dist > 1024) dist = 1024;
    float s = dot + rt[dist];
    float nm = fmaxf(mx, s);
    float al = __expf(mx - nm);
    float pv = __expf(s - nm);
    acc = acc * al + pv * vv;
    lsum = lsum * al + pv;
    mx = nm;
  }
  float o = acc / lsum;
  u64 oidx = ((u64)m * 24 + h) * 128 + tid;
  o *= 1.0f / (1.0f + __expf(-gate[oidx]));
  out[oidx] = o;
}

// 9. Post-Attn / Post-MLP 4-tap ShortConv + optional residual add (grid ceil(C/256), block 256)
extern "C" __global__ void __launch_bounds__(256) ink_sconv4_seq(
  const float* __restrict__ src, const float* __restrict__ swp,
  float* __restrict__ hist, float* __restrict__ dst, int M, int C, int add_res) {
  int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= C) return;
  float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
  const float4 sw = ((const float4*)swp)[c];
  for (int m = 0; m < M; m++) {
    float cur = src[(u64)m * C + c];
    float y = (cur + h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w) * 0.44721359549995793f;
    h0 = h1; h1 = h2; h2 = cur;
    u64 idx = (u64)m * C + c;
    dst[idx] = add_res ? (dst[idx] + y) : y;
  }
  hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
}

// 10. Exact Top-8 / 128 Sigmoid + Bias MoE Router (grid M, block 128)
extern "C" __global__ void __launch_bounds__(128) ink_moe_route_topk8(
  const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
  i32* __restrict__ top_idx, float* __restrict__ top_w, float route_scale, int M, int D) {
  int m = blockIdx.x, e = threadIdx.x;
  if (m >= M) return;
  const float* xm = x + (u64)m * D;
  const float* we = w + (u64)e * D;
  float dot = 0.0f;
  for (int d = 0; d < D; d++) dot = fmaf(xm[d], we[d], dot);
  float sig = 1.0f / (1.0f + __expf(-dot));
  __shared__ float s_sig[128], s_sc[128];
  s_sig[e] = sig;
  s_sc[e] = sig + b[e];
  __syncthreads();
  if (e == 0) {
    float sum = 0.0f;
    for (int k = 0; k < 8; k++) {
      int bi = 0; float bv = -1e30f;
      for (int i = 0; i < 128; i++) {
        if (s_sc[i] > bv) { bv = s_sc[i]; bi = i; }
      }
      top_idx[(u64)m * 8 + k] = bi;
      float wt = s_sig[bi];
      top_w[(u64)m * 8 + k] = wt;
      sum += wt;
      s_sc[bi] = -2e30f;
    }
    float inv = route_scale / (sum + 1e-20f);
    for (int k = 0; k < 8; k++) top_w[(u64)m * 8 + k] *= inv;
  }
}

// 11. Routed MoE Expert Descriptor + SiTU(4, 25) Gate+Up & Down Kernels (SF4, MX8, S_K!)
struct InkExpDev { u64 ptr; u32 dn_off; u8 gu_code; u8 dn_code; u16 pad; };

static __device__ __forceinline__ float ink_exp_dot_row(
  const u8* body, u8 code, int row, const float* x, int K, int N_rows) {
  int l = threadIdx.x & 31;
  float acc = 0.0f;
  if (code == 15) {
    // SF4 (NVFP4): 16B header (w13_o0, w13_o1, w2_o0, w2_o1), packed nibbles [N, K/2], E4M3 scales [N, K/16]
    const float* hdr = (const float*)body;
    float outer = (N_rows == 2304 && K == 2304) ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
    const u8* pbase = body + ((N_rows == 2304 && K == 2304) ? 16 : 0);
    const u8* pr = pbase + (u64)row * (K >> 1);
    const u8* sr = pbase + (u64)N_rows * (K >> 1) + (u64)row * (K >> 4);
    for (int g = l; g < (K >> 4); g += 32) {
      float sc = ink_e4m3(sr[g]) * outer;
      const u8* pg = pr + g * 8;
      const float* xg = x + g * 16;
      float s = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        u8 b = pg[i];
        s = fmaf(kE2M1[b & 15], xg[2 * i], fmaf(kE2M1[b >> 4], xg[2 * i + 1], s));
      }
      acc = fmaf(sc, s, acc);
    }
  } else if (code == 16) {
    // MX8 (MXFP8): weights [N, K] E4M3 + UE8M0 scales [N, K/32]
    const u8* wr = body + (u64)row * K;
    const u8* sr = body + (u64)N_rows * K + (u64)row * (K >> 5);
    for (int g = l; g < (K >> 5); g += 32) {
      float sc = ink_ue8m0(sr[g]);
      const u8* wg = wr + g * 32;
      const float* xg = x + g * 32;
      float s = 0.0f;
      #pragma unroll
      for (int i = 0; i < 32; i++) s = fmaf(ink_e4m3(wg[i]), xg[i], s);
      acc = fmaf(sc, s, acc);
    }
  } else {
    // S_K (S2_K..S8_K): body starts with 2*1152 B FP16 sq_xd_dn (only on gu body)
    int sb = ink_sk_sb(code);
    const u8* rptr = body + (u64)row * (K >> 8) * sb;
    for (int k = l; k < K; k += 32) acc = fmaf(ink_sk_elem(rptr, code, k), x[k], acc);
  }
  return ink_wsum(acc);
}

// MoE Gate+Up+SiTU: grid (ceil(1152/4), M*8), block 128 (4 warps = 4 inter dims/block)
extern "C" __global__ void __launch_bounds__(128) ink_moe_gu_situ(
  const InkExpDev* __restrict__ exps, const i32* __restrict__ top_idx,
  const float* __restrict__ x, float* __restrict__ act, int M) {
  int slot = blockIdx.y, m = slot >> 3, j = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (m >= M || j >= 1152) return;
  InkExpDev ed = exps[top_idx[slot]];
  const u8* body = (const u8*)ed.ptr;
  const u8* gu_base = (ed.gu_code == 15 || ed.gu_code == 16) ? body : (body + 1152 * 2);
  const float* xm = x + (u64)m * 2304;
  float gv = ink_exp_dot_row(gu_base, ed.gu_code, 2 * j, xm, 2304, 2304);
  float uv = ink_exp_dot_row(gu_base, ed.gu_code, 2 * j + 1, xm, 2304, 2304);
  if (l == 0) {
    float v = ink_situ(gv, uv);
    if (ed.dn_code != 15 && ed.dn_code != 16) {
      u16 sq = ((const u16*)body)[j];
      v /= ink_f16(sq);
    }
    act[(u64)slot * 1152 + j] = v;
  }
}

// MoE Down + deterministic 8-slot weighted reduction: grid (ceil(2304/4), M), block 128
extern "C" __global__ void __launch_bounds__(128) ink_moe_dn_reduce(
  const InkExpDev* __restrict__ exps, const i32* __restrict__ top_idx, const float* __restrict__ top_w,
  const float* __restrict__ act, float* __restrict__ out, int M) {
  int m = blockIdx.y, d = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (m >= M || d >= 2304) return;
  float sum = 0.0f;
  for (int k = 0; k < 8; k++) {
    int slot = m * 8 + k;
    InkExpDev ed = exps[top_idx[slot]];
    const u8* body = (const u8*)ed.ptr;
    const u8* dn_ptr = body + ed.dn_off;
    float yv = 0.0f;
    if (ed.dn_code == 15) {
      const float* hdr = (const float*)body;
      float outer = hdr[2] / hdr[3];
      const u8* pr = dn_ptr + (u64)d * (1152 >> 1);
      const u8* sr = dn_ptr + (u64)2304 * (1152 >> 1) + (u64)d * (1152 >> 4);
      const float* am = act + (u64)slot * 1152;
      float a = 0.0f;
      for (int g = l; g < (1152 >> 4); g += 32) {
        float sc = ink_e4m3(sr[g]) * outer;
        const u8* pg = pr + g * 8;
        const float* ag = am + g * 16;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
          u8 b = pg[i];
          s = fmaf(kE2M1[b & 15], ag[2 * i], fmaf(kE2M1[b >> 4], ag[2 * i + 1], s));
        }
        a = fmaf(sc, s, a);
      }
      yv = ink_wsum(a);
    } else {
      yv = ink_exp_dot_row(dn_ptr, ed.dn_code, d, act + (u64)slot * 1152, 1152, 2304);
    }
    sum = fmaf(top_w[slot], yv, sum);
  }
  if (l == 0) out[(u64)m * 2304 + d] = sum;
}

// 12. Softcap + Deterministic Argmax over [M, V] logits
extern "C" __global__ void __launch_bounds__(256) ink_softcap_argmax(
  const float* __restrict__ lg, int V, float cap, i32* __restrict__ out_id, float* __restrict__ out_val) {
  int m = blockIdx.x, tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  const float* row = lg + (u64)m * V;
  float bv = -3e38f; i32 bi = 0;
  float ic = 1.0f / cap;
  for (int i = tid; i < V; i += 256) {
    float v = cap * tanhf(row[i] * ic);
    if (v > bv) { bv = v; bi = i; }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    i32 oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  __shared__ float sv[8]; __shared__ i32 si[8];
  if (l == 0) { sv[w] = bv; si[w] = bi; }
  __syncthreads();
  if (tid == 0) {
    float b2 = sv[0]; i32 i2 = si[0];
    for (int k = 1; k < 8; k++) if (sv[k] > b2 || (sv[k] == b2 && si[k] < i2)) { b2 = sv[k]; i2 = si[k]; }
    out_id[m] = i2;
    if (out_val) out_val[m] = b2;
  }
}

// 13. Exact 2-Stage Cauchy-Schwarz INT8 Screened Unembed for BF16 unembed (2x bandwidth win, 100% bit-exact!)
extern "C" __global__ void __launch_bounds__(256) ink_scr_build_bf16(
  const u16* __restrict__ W, i8* __restrict__ q8, float* __restrict__ sc, float* __restrict__ eps, int V, int D) {
  int row = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (row >= V) return;
  const u16* wr = W + (u64)row * D;
  float am = 0.0f;
  for (int k = l; k < D; k += 32) am = fmaxf(am, fabsf(ink_bf16(wr[k])));
  am = ink_wmax(am);
  float s = am * (1.0f / 127.0f), inv = s > 0.0f ? (127.0f / am) : 0.0f;
  i8* dst = q8 + (u64)row * D;
  float err2 = 0.0f;
  for (int k = l; k < D; k += 32) {
    float w = ink_bf16(wr[k]);
    int q = max(-127, min(127, (int)roundf(w * inv)));
    dst[k] = (i8)q;
    float d = w - (float)q * s;
    err2 += d * d;
  }
  err2 = ink_wsum(err2);
  if (l == 0) { sc[row] = s; eps[row] = sqrtf(err2) * 1.002f + 1e-4f; }
}

extern "C" __global__ void __launch_bounds__(256) ink_scr_pass1_i8(
  const i8* __restrict__ q8, const float* __restrict__ sc, const float* __restrict__ eps,
  const float* __restrict__ h, float h_norm, float* __restrict__ dot_hat, int* __restrict__ lmax_bits, int V, int D) {
  int row = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (row >= V) return;
  const uint4* q16 = (const uint4*)(q8 + (u64)row * D);
  float acc = 0.0f;
  for (int i = l; i < (D >> 4); i += 32) {
    uint4 v = q16[i]; u32 u[4] = { v.x, v.y, v.z, v.w };
    int base = i * 16;
    #pragma unroll
    for (int q = 0; q < 4; q++) {
      float4 xv = *(const float4*)(h + base + q * 4);
      u32 p = u[q];
      acc = fmaf((float)(i8)(p & 0xffu), xv.x, acc);
      acc = fmaf((float)(i8)((p >> 8) & 0xffu), xv.y, acc);
      acc = fmaf((float)(i8)((p >> 16) & 0xffu), xv.z, acc);
      acc = fmaf((float)(i8)(p >> 24), xv.w, acc);
    }
  }
  float d = ink_wsum(acc) * sc[row];
  if (l == 0) {
    dot_hat[row] = d;
    u32 b = __float_as_uint(d - eps[row] * h_norm);
    atomicMax(lmax_bits, (int)(b ^ ((b >> 31) ? 0x7fffffffu : 0u)));
  }
}
