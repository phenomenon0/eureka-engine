// ec_kernels.cu — 100% Native Eureka Creative CUDA Kernels
// Zero cuBLAS, zero ggml, zero external ML libraries.
// Built on pure PTX Tensor Cores (m16n8k16 FP16 + m16n8k32 INT8) in the exact idiom of
// vision/vitc.cu and tools/pf27/pfmm.cuh.

#include "ec_kernels.cuh"
#include <float.h>
#include <math.h>
#include <stdio.h>

// ===========================================================================
// Low-level PTX Tensor Core & Bit-Pack Intrinsics
// ===========================================================================

__device__ __forceinline__ void ec_mma_f16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ec_mma_s8(int32_t *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ec_pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  uint32_t u;
  memcpy(&u, &h, 4);
  return u;
}

__device__ __forceinline__ float ec_silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ec_gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// ===========================================================================
// 1. Fused adaLN-Zero & Gated Residual
// ===========================================================================

__global__ void k_adaln_zero(
    const float *__restrict__ x, const float *__restrict__ norm_w,
    const float *__restrict__ shift, const float *__restrict__ scale,
    float *__restrict__ y, __half *__restrict__ y_f16,
    int dim, float eps, int norm_mode) {
  __shared__ float red[32];
  const long row = blockIdx.x;
  const float *xr = x + row * dim;
  const int tid = threadIdx.x;
  const int nwarps = blockDim.x >> 5;

  float mean = 0.0f;
  if (norm_mode == 0) {
    float s = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) s += xr[i];
    s = warp_reduce_sum(s);
    if ((tid & 31) == 0) red[tid >> 5] = s;
    __syncthreads();
    float bs = (tid < nwarps) ? red[tid] : 0.0f;
    if (tid < 32) bs = warp_reduce_sum(bs);
    if (tid == 0) red[0] = bs / (float)dim;
    __syncthreads();
    mean = red[0];
  }

  float v = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    float d = xr[i] - mean;
    v += d * d;
  }
  v = warp_reduce_sum(v);
  if ((tid & 31) == 0) red[tid >> 5] = v;
  __syncthreads();
  float bv = (tid < nwarps) ? red[tid] : 0.0f;
  if (tid < 32) bv = warp_reduce_sum(bv);
  if (tid == 0) red[0] = rsqrtf(bv / (float)dim + eps);
  __syncthreads();
  const float inv = red[0];

  float *yr = y + row * dim;
  __half *yh = y_f16 ? (y_f16 + row * dim) : nullptr;
  for (int i = tid; i < dim; i += blockDim.x) {
    float nrm = (xr[i] - mean) * inv;
    if (norm_mode == 2 && norm_w) nrm *= norm_w[i];
    const float sc = scale ? (1.0f + scale[i]) : 1.0f;
    const float sh = shift ? shift[i] : 0.0f;
    const float out = nrm * sc + sh;
    yr[i] = out;
    if (yh) yh[i] = __float2half_rn(out);
  }
}

extern "C" int ec_launch_adaln_zero(
    const float *x, const float *norm_w,
    const float *shift, const float *scale,
    float *y, __half *y_f16,
    int n_rows, int dim, float eps, int norm_mode,
    cudaStream_t stream) {
  if (n_rows <= 0 || dim <= 0) return -1;
  int threads = 256;
  if (dim >= 2048) threads = 512;
  k_adaln_zero<<<n_rows, threads, 0, stream>>>(x, norm_w, shift, scale, y, y_f16, dim, eps, norm_mode);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_gated_res(
    float *__restrict__ x, const float *__restrict__ h, const float *__restrict__ gate,
    long total, int dim) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  x[i] = x[i] + gate[i % dim] * h[i];
}

extern "C" int ec_launch_gated_res(
    float *x, const float *h, const float *gate,
    int n_rows, int dim, cudaStream_t stream) {
  const long total = (long)n_rows * dim;
  if (total <= 0) return -1;
  const int blk = 256;
  const int grd = (int)((total + blk - 1) / blk);
  k_gated_res<<<grd, blk, 0, stream>>>(x, h, gate, total, dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 2. Generalized N-D RoPE (1D Audio, 2D Image, 3D Video/O-Voxel) + Optional QK-RMSNorm
// ===========================================================================

__global__ void k_rope_nd(
    const float *__restrict__ q_in, const float *__restrict__ k_in, const float *__restrict__ v_in,
    const int32_t *__restrict__ coords,
    const float *__restrict__ qk_norm_w_q, const float *__restrict__ qk_norm_w_k,
    __half *__restrict__ q_out, __half *__restrict__ k_out, __half *__restrict__ v_out,
    int n_tokens, int n_pad, int n_heads, int head_dim,
    int ndims, int ax0, int ax1, int ax2, float theta_base, float eps) {
  // One warp handles one (token, head) pair for fast optional QK-RMSNorm + RoPE!
  const long pair_idx = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (pair_idx >= (long)n_tokens * n_heads) return;

  const int h = (int)(pair_idx % n_heads);
  const int tok = (int)(pair_idx / n_heads);
  const int E = n_heads * head_dim;
  const float *qs = q_in + (long)tok * E + h * head_dim;
  const float *ks = k_in + (long)tok * E + h * head_dim;
  const float *vs = v_in ? (v_in + (long)tok * E + h * head_dim) : nullptr;

  float inv_q = 1.0f, inv_k = 1.0f;
  if (qk_norm_w_q || qk_norm_w_k) {
    float sq = 0.0f, sk = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
      float qv = qs[d], kv = ks[d];
      sq += qv * qv;
      sk += kv * kv;
    }
    sq = warp_reduce_sum(sq);
    sk = warp_reduce_sum(sk);
    if (qk_norm_w_q) inv_q = rsqrtf(sq / (float)head_dim + eps);
    if (qk_norm_w_k) inv_k = rsqrtf(sk / (float)head_dim + eps);
  }

  const int32_t c0 = coords[(long)tok * ndims + 0];
  const int32_t c1 = (ndims >= 2) ? coords[(long)tok * ndims + 1] : 0;
  const int32_t c2 = (ndims >= 3) ? coords[(long)tok * ndims + 2] : 0;

  const int half_d = head_dim >> 1;
  const int p0 = ax0 >> 1;
  const int p1 = ax1 >> 1;
  const int p2 = ax2 >> 1;

  const long out_base = ((long)h * n_pad + tok) * head_dim;
  __half *qo = q_out + out_base;
  __half *ko = k_out + out_base;
  __half *vo = v_out ? (v_out + out_base) : nullptr;

  // Interleaved pair convention (standard in FLUX / Wan / LTX / Qwen-Image / TRELLIS / ACE-Step):
  // pairs (2*p, 2*p+1) rotated by axis frequency.
  for (int p = lane; p < half_d; p += 32) {
    const int d0 = 2 * p, d1 = d0 + 1;
    float q0 = qs[d0] * inv_q * (qk_norm_w_q ? qk_norm_w_q[d0] : 1.0f);
    float q1 = qs[d1] * inv_q * (qk_norm_w_q ? qk_norm_w_q[d1] : 1.0f);
    float k0 = ks[d0] * inv_k * (qk_norm_w_k ? qk_norm_w_k[d0] : 1.0f);
    float k1 = ks[d1] * inv_k * (qk_norm_w_k ? qk_norm_w_k[d1] : 1.0f);

    float angle = 0.0f;
    if (p < p0 && p0 > 0) {
      angle = (float)c0 * powf(theta_base, -2.0f * (float)p / (float)ax0);
    } else if (ndims >= 2 && p < p0 + p1 && p1 > 0) {
      angle = (float)c1 * powf(theta_base, -2.0f * (float)(p - p0) / (float)ax1);
    } else if (ndims >= 3 && p < p0 + p1 + p2 && p2 > 0) {
      angle = (float)c2 * powf(theta_base, -2.0f * (float)(p - p0 - p1) / (float)ax2);
    }

    float cs = cosf(angle), sn = sinf(angle);
    qo[d0] = __float2half_rn(q0 * cs - q1 * sn);
    qo[d1] = __float2half_rn(q0 * sn + q1 * cs);
    ko[d0] = __float2half_rn(k0 * cs - k1 * sn);
    ko[d1] = __float2half_rn(k0 * sn + k1 * cs);
    if (vo && vs) {
      vo[d0] = __float2half_rn(vs[d0]);
      vo[d1] = __float2half_rn(vs[d1]);
    }
  }
}

extern "C" int ec_launch_rope_nd(
    const float *q_in, const float *k_in, const float *v_in,
    const int32_t *coords,
    const float *qk_norm_w_q, const float *qk_norm_w_k,
    __half *q_out, __half *k_out, __half *v_out,
    int n_tokens, int n_pad, int n_heads, int head_dim,
    int ndims, const int32_t axis_dims[3], float theta_base, float eps,
    cudaStream_t stream) {
  if (n_tokens <= 0 || n_heads <= 0 || (head_dim & 1) || ndims < 1 || ndims > 3) return -1;
  const long pairs = (long)n_tokens * n_heads;
  const int warps_per_blk = 8; // 256 threads
  const int grd = (int)((pairs + warps_per_blk - 1) / warps_per_blk);
  k_rope_nd<<<grd, warps_per_blk * 32, 0, stream>>>(
      q_in, k_in, v_in, coords, qk_norm_w_q, qk_norm_w_k,
      q_out, k_out, v_out, n_tokens, n_pad, n_heads, head_dim,
      ndims, axis_dims[0], axis_dims[1], axis_dims[2], theta_base, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 3. Shared-Memory Tiled Tensor-Core FP16 GEMM (mma.sync.aligned.m16n8k16)
//    Y[M, N] = X[M, K] @ W[N, K]^T + optional epilogue
// ===========================================================================
// Block: 128 threads (4 warps), 64x64 output tile, BK=32 staged in smem.
// Each warp computes 32x32 = 2x4 MMA tiles (m16n8k16). Zero cuBLAS!

__global__ void __launch_bounds__(128) k_gemm_f16(
    const __half *__restrict__ X, const __half *__restrict__ W,
    const float *__restrict__ bias, const float *__restrict__ res,
    float *__restrict__ Y, __half *__restrict__ Y_f16,
    int M, int N, int K, int epilogue) {
  constexpr int BM = 64, BN = 64, BK = 32, STRIDE = 40; // +8 pad avoids bank conflicts
  __shared__ __align__(16) __half sX[BM * STRIDE];
  __shared__ __align__(16) __half sW[BN * STRIDE];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wm = (warp >> 1) * 32; // 0 or 32
  const int wn = (warp & 1) * 32;  // 0 or 32

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
    // Cooperative load 64x32 halfs = 256 uint4s -> 2 uint4s/thread
    if (fast_k && (kb + BK <= K)) {
      #pragma unroll
      for (int iter = 0; iter < 2; iter++) {
        const int idx = threadIdx.x + iter * 128; // 0..255
        const int r = idx >> 2;                   // 0..63
        const int c = (idx & 3) * 8;              // 0, 8, 16, 24
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
        sX[r * STRIDE + c] = (m0 + r < M && kb + c < K) ? X[(long)(m0 + r) * K + kb + c] : __float2half(0.0f);
        sW[r * STRIDE + c] = (n0 + r < N && kb + c < K) ? W[(long)(n0 + r) * K + kb + c] : __float2half(0.0f);
      }
    }
    __syncthreads();

    // 2 K-substeps of 16
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      const int ko = ks * 16;
      uint32_t a_frag[2][4];
      #pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        const __half *xr0 = sX + (wm + mi * 16 + g) * STRIDE + ko;
        const __half *xr8 = xr0 + 8 * STRIDE;
        a_frag[mi][0] = *(const uint32_t *)(xr0 + 2 * t);
        a_frag[mi][1] = *(const uint32_t *)(xr8 + 2 * t);
        a_frag[mi][2] = *(const uint32_t *)(xr0 + 8 + 2 * t);
        a_frag[mi][3] = *(const uint32_t *)(xr8 + 8 + 2 * t);
      }
      #pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const __half *wr = sW + (wn + ni * 8 + g) * STRIDE + ko;
        const uint32_t b0 = *(const uint32_t *)(wr + 2 * t);
        const uint32_t b1 = *(const uint32_t *)(wr + 8 + 2 * t);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          ec_mma_f16(acc[mi][ni], a_frag[mi], b0, b1);
        }
      }
    }
    __syncthreads();
  }

  // Store with fused epilogue
  #pragma unroll
  for (int mi = 0; mi < 2; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = m0 + wm + mi * 16 + g + 8 * (e >> 1);
        const int c = n0 + wn + ni * 8 + 2 * t + (e & 1);
        if (r < M && c < N) {
          float v = acc[mi][ni][e];
          if (epilogue >= 1 && bias) v += bias[c];
          if (epilogue == 2) v = ec_silu(v);
          else if (epilogue == 3) v = ec_gelu_tanh(v);
          else if (epilogue == 4 && res) v += res[(long)r * N + c];
          Y[(long)r * N + c] = v;
          if (Y_f16) Y_f16[(long)r * N + c] = __float2half_rn(v);
        }
      }
    }
  }
}

extern "C" int ec_launch_gemm_f16(
    const __half *X, const __half *W, const float *bias, const float *res,
    float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue,
    cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -1;
  dim3 grd((N + 63) / 64, (M + 63) / 64);
  k_gemm_f16<<<grd, 128, 0, stream>>>(X, W, bias, res, Y, Y_f16, M, N, K, epilogue);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_lora_b_accum(
    const __half *__restrict__ H, const __half *__restrict__ B,
    float *__restrict__ Y, int M, int N, int R, float scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)M * N) return;
  const int r = (int)(idx / N);
  const int c = (int)(idx % N);
  const __half *hr = H + (long)r * R;
  const __half *br = B + (long)c * R;
  float s = 0.0f;
  for (int k = 0; k < R; k++) s += __half2float(hr[k]) * __half2float(br[k]);
  Y[idx] += scale * s;
}

// Hot-fused LoRA: Y[M, N] += scale * ((X[M, K] @ A[R, K]^T) @ B[N, R]^T)
extern "C" int ec_launch_lora_add(
    const __half *X, const ec_lora_t *lora,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (!lora || lora->rank <= 0) return 0;
  const int R = lora->rank;
  if (ec_launch_gemm_f16(X, (const __half *)lora->A_d, nullptr, nullptr, tmp_f32, tmp_f16, M, R, K, 0, stream))
    return -1;
  const long total = (long)M * N;
  const int blk = 256;
  const int grd = (int)((total + blk - 1) / blk);
  k_lora_b_accum<<<grd, blk, 0, stream>>>(tmp_f16, (const __half *)lora->B_d, Y, M, N, R, lora->scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 3B. Native INT8 W8A8 Tensor-Core GEMM (PTX m16n8k32 s8 -> s32) — 100+ TOPS Zero cuBLAS
// ===========================================================================
// Block: 128 threads (4 warps), 64x64 output tile, BK=64 staged in smem (2 k32 steps/load).
__global__ void __launch_bounds__(128) k_gemm_i8(
    const int8_t *__restrict__ X, const float *__restrict__ x_sc,
    const int8_t *__restrict__ W, const float *__restrict__ w_sc,
    const float *__restrict__ bias, float *__restrict__ Y,
    int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 64, STRIDE = 80; // +16B pad avoids smem bank conflicts
  __shared__ __align__(16) int8_t sX[BM * STRIDE];
  __shared__ __align__(16) int8_t sW[BN * STRIDE];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wm = (warp >> 1) * 32;
  const int wn = (warp & 1) * 32;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  int32_t acc[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;

  const bool fast_k = ((K & 15) == 0);
  for (int kb = 0; kb < K; kb += BK) {
    if (fast_k && (kb + BK <= K)) {
      // 64 rows x 64 bytes = 256 uint4s -> 2 uint4 loads/thread
      #pragma unroll
      for (int iter = 0; iter < 2; iter++) {
        const int idx = threadIdx.x + iter * 128;
        const int r = idx >> 2;
        const int c = (idx & 3) * 16;
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
      const int ko = ks * 32;
      uint32_t a_frag[2][4];
      #pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        const int8_t *xr0 = sX + (wm + mi * 16 + g) * STRIDE + ko;
        const int8_t *xr8 = xr0 + 8 * STRIDE;
        a_frag[mi][0] = *(const uint32_t *)(xr0 + 4 * t);
        a_frag[mi][1] = *(const uint32_t *)(xr8 + 4 * t);
        a_frag[mi][2] = *(const uint32_t *)(xr0 + 16 + 4 * t);
        a_frag[mi][3] = *(const uint32_t *)(xr8 + 16 + 4 * t);
      }
      #pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const int8_t *wr = sW + (wn + ni * 8 + g) * STRIDE + ko;
        const uint32_t b0 = *(const uint32_t *)(wr + 4 * t);
        const uint32_t b1 = *(const uint32_t *)(wr + 16 + 4 * t);
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          ec_mma_s8(acc[mi][ni], a_frag[mi], b0, b1);
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
          float v = (float)acc[mi][ni][e] * x_sc[r] * w_sc[c];
          if (bias) v += bias[c];
          Y[(long)r * N + c] = v;
        }
      }
    }
  }
}

extern "C" int ec_launch_gemm_i8(
    const int8_t *X_i8, const float *x_sc,
    const int8_t *W_i8, const float *w_sc,
    const float *bias, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -1;
  dim3 grd((N + 63) / 64, (M + 63) / 64);
  k_gemm_i8<<<grd, 128, 0, stream>>>(X_i8, x_sc, W_i8, w_sc, bias, Y, M, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 3C. Native SVDQuant W4A4 + FP16 Low-Rank Outlier GEMM (100% Native 4-bit DiT Path)
// ===========================================================================
// Quantizes X_f16 per-32-group to int4 (-7..+7) on-the-fly in registers and performs
// dp4a (__dp4a) dot product against packed W_i4[N, K/2] with per-32-group scales,
// then adds the FP16 low-rank outlier branch (rank 16/32) via ec_launch_lora_add!
__global__ void k_gemm_w4a4_core(
    const __half *__restrict__ X, const uint8_t *__restrict__ W_i4, const float *__restrict__ w_sc,
    float *__restrict__ Y, int M, int N, int K) {
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M || c >= N) return;
  const __half *xr = X + (long)r * K;
  const uint8_t *wr = W_i4 + (long)c * (K / 2);
  const float *ws = w_sc + (long)c * (K / 32);
  const int ng = K / 32;
  float acc = 0.0f;
  for (int g = 0; g < ng; g++) {
    const __half *xg = xr + g * 32;
    float amax = 0.0f;
    float xv[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
      float v = __half2float(xg[i]);
      xv[i] = v;
      amax = fmaxf(amax, fabsf(v));
    }
    const float xs = (amax > 1e-8f) ? (amax / 7.0f) : 0.0f;
    const float inv_x = (amax > 1e-8f) ? (7.0f / amax) : 0.0f;
    int32_t idot = 0;
    const uint8_t *wg = wr + g * 16;
    #pragma unroll
    for (int b = 0; b < 16; b += 2) {
      uint8_t b0 = wg[b], b1 = wg[b + 1];
      int8_t w0 = (int8_t)((b0 & 0x0Fu) - 8);
      int8_t w1 = (int8_t)((b0 >> 4) - 8);
      int8_t w2 = (int8_t)((b1 & 0x0Fu) - 8);
      int8_t w3 = (int8_t)((b1 >> 4) - 8);
      int8_t q0 = (int8_t)__float2int_rn(xv[2 * b + 0] * inv_x);
      int8_t q1 = (int8_t)__float2int_rn(xv[2 * b + 1] * inv_x);
      int8_t q2 = (int8_t)__float2int_rn(xv[2 * b + 2] * inv_x);
      int8_t q3 = (int8_t)__float2int_rn(xv[2 * b + 3] * inv_x);
      int32_t wp = ((uint32_t)(uint8_t)w0) | (((uint32_t)(uint8_t)w1) << 8) |
                   (((uint32_t)(uint8_t)w2) << 16) | (((uint32_t)(uint8_t)w3) << 24);
      int32_t qp = ((uint32_t)(uint8_t)q0) | (((uint32_t)(uint8_t)q1) << 8) |
                   (((uint32_t)(uint8_t)q2) << 16) | (((uint32_t)(uint8_t)q3) << 24);
      idot = __dp4a(wp, qp, idot);
    }
    acc += (float)idot * (xs * ws[g]);
  }
  Y[(long)r * N + c] = acc;
}

extern "C" int ec_launch_gemm_svd4(
    const __half *X_f16,
    const uint8_t *W_i4, const float *w_sc,
    const ec_lora_t *outlier_branch,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || (K & 31)) return -1;
  dim3 grd((N + 127) / 128, M);
  k_gemm_w4a4_core<<<grd, 128, 0, stream>>>(X_f16, W_i4, w_sc, Y, M, N, K);
  if (cudaGetLastError() != cudaSuccess) return -1;
  if (outlier_branch && outlier_branch->rank > 0) {
    return ec_launch_lora_add(X_f16, outlier_branch, tmp_f32, tmp_f16, Y, M, N, K, stream);
  }
  return 0;
}

// ===========================================================================
// 4A. Exact Oracle Tier: FlashAttention-2 Register Form (FP16 m16n8k16, D=64/128)
//     Supports both Bidirectional (DiT) and Block-Causal (Self-Forcing Realtime Video)
// ===========================================================================

template <int D>
__global__ void __launch_bounds__(128) k_fa2_f16(
    const __half *__restrict__ Q, const __half *__restrict__ K, const __half *__restrict__ V,
    float *__restrict__ O, __half *__restrict__ O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, float scale, int block_causal_size) {
  constexpr int KC = D / 16, DT = D / 8, LK = D + 8, LV = 72;
  __shared__ __align__(16) __half sK[64 * LK];
  __shared__ __align__(16) __half sVt[D * LV];

  const int hh = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int qr = blockIdx.x * 64 + warp * 16;
  const int E = n_heads * D;

  const __half *qg = Q + (long)hh * n_q_pad * D;
  const __half *kg = K + (long)hh * n_kv_pad * D;
  const __half *vg = V + (long)hh * n_kv_pad * D;

  // Compute max visible key for this block and for this lane's two query rows (q0, q1)
  int kv_limit_blk = n_kv;
  int kv_lim0 = n_kv, kv_lim1 = n_kv;
  if (block_causal_size > 0) {
    const int max_q_blk = min(blockIdx.x * 64 + 63, n_q - 1);
    kv_limit_blk = min(n_kv, ((max_q_blk / block_causal_size) + 1) * block_causal_size);
    const int q0 = qr + g, q1 = qr + g + 8;
    kv_lim0 = (q0 < n_q) ? min(n_kv, ((q0 / block_causal_size) + 1) * block_causal_size) : 0;
    kv_lim1 = (q1 < n_q) ? min(n_kv, ((q1 / block_causal_size) + 1) * block_causal_size) : 0;
  }

  uint32_t qa[KC][4];
  {
    const int r0 = min(qr + g, n_q_pad - 1);
    const int r8 = min(qr + g + 8, n_q_pad - 1);
    const __half *q0p = qg + (long)r0 * D;
    const __half *q8p = qg + (long)r8 * D;
    #pragma unroll
    for (int kc = 0; kc < KC; kc++) {
      qa[kc][0] = *(const uint32_t *)(q0p + kc * 16 + 2 * t);
      qa[kc][1] = *(const uint32_t *)(q8p + kc * 16 + 2 * t);
      qa[kc][2] = *(const uint32_t *)(q0p + kc * 16 + 8 + 2 * t);
      qa[kc][3] = *(const uint32_t *)(q8p + kc * 16 + 8 + 2 * t);
    }
  }

  float o[DT][4];
  #pragma unroll
  for (int dt = 0; dt < DT; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
  float M0 = -FLT_MAX * 0.5f, M1 = -FLT_MAX * 0.5f, l0 = 0.0f, l1 = 0.0f;

  for (int kb = 0; kb < kv_limit_blk; kb += 64) {
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * DT; i += 128) {
      const int r = i / DT, c = (i % DT) * 8;
      uint4 kv_val = make_uint4(0, 0, 0, 0), vv_val = make_uint4(0, 0, 0, 0);
      if (kb + r < n_kv) {
        kv_val = *(const uint4 *)(kg + (long)(kb + r) * D + c);
        vv_val = *(const uint4 *)(vg + (long)(kb + r) * D + c);
      }
      *(uint4 *)(sK + r * LK + c) = kv_val;
      const __half *vv = (const __half *)&vv_val;
      #pragma unroll
      for (int j = 0; j < 8; j++) sVt[(c + j) * LV + r] = vv[j];
    }
    __syncthreads();

    float s[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
      const __half *kr = sK + (nt * 8 + g) * LK + 2 * t;
      #pragma unroll
      for (int kc = 0; kc < KC; kc++) {
        ec_mma_f16(s[nt], qa[kc], *(const uint32_t *)(kr + kc * 16), *(const uint32_t *)(kr + kc * 16 + 8));
      }
    }

    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        const int kidx = kb + nt * 8 + 2 * t + j;
        s[nt][j]     = (kidx < kv_lim0) ? (s[nt][j] * scale)     : -INFINITY;
        s[nt][2 + j] = (kidx < kv_lim1) ? (s[nt][2 + j] * scale) : -INFINITY;
        mx0 = fmaxf(mx0, s[nt][j]);
        mx1 = fmaxf(mx1, s[nt][2 + j]);
      }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float Mn0 = fmaxf(M0, mx0), Mn1 = fmaxf(M1, mx1);
    const float ms0 = (Mn0 > -1e30f) ? expf(M0 - Mn0) : 0.0f;
    const float ms1 = (Mn1 > -1e30f) ? expf(M1 - Mn1) : 0.0f;
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        s[nt][j]     = (s[nt][j] > -1e30f)     ? expf(s[nt][j] - Mn0)     : 0.0f;
        s[nt][2 + j] = (s[nt][2 + j] > -1e30f) ? expf(s[nt][2 + j] - Mn1) : 0.0f;
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

    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
      const uint32_t pa[4] = {
        ec_pack_h2(s[2 * kc][0],     s[2 * kc][1]),
        ec_pack_h2(s[2 * kc][2],     s[2 * kc][3]),
        ec_pack_h2(s[2 * kc + 1][0], s[2 * kc + 1][1]),
        ec_pack_h2(s[2 * kc + 1][2], s[2 * kc + 1][3]),
      };
      #pragma unroll
      for (int dt = 0; dt < DT; dt++) {
        const __half *vr = sVt + (dt * 8 + g) * LV + kc * 16 + 2 * t;
        ec_mma_f16(o[dt], pa, *(const uint32_t *)vr, *(const uint32_t *)(vr + 8));
      }
    }
  }

  const float inv0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
  const float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
  #pragma unroll
  for (int half8 = 0; half8 < 2; half8++) {
    const int q = qr + g + 8 * half8;
    if (q >= n_q) continue;
    const float inv = half8 ? inv1 : inv0;
    float *d = O + (long)q * E + hh * D;
    __half *dq = O_f16 ? (O_f16 + (long)q * E + hh * D) : nullptr;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      const float v0 = o[dt][2 * half8] * inv;
      const float v1 = o[dt][2 * half8 + 1] * inv;
      const int col = dt * 8 + 2 * t;
      *(float2 *)(d + col) = make_float2(v0, v1);
      if (dq) *(__half2 *)(dq + col) = __floats2half2_rn(v0, v1);
    }
  }
}

extern "C" int ec_launch_fa2_f16(
    const __half *Q, const __half *K, const __half *V,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size,
    cudaStream_t stream) {
  if (n_q <= 0 || n_kv <= 0 || n_heads <= 0) return -1;
  dim3 grd((n_q + 63) / 64, n_heads);
  if (head_dim == 64) {
    k_fa2_f16<64><<<grd, 128, 0, stream>>>(Q, K, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, scale, block_causal_size);
  } else if (head_dim == 128) {
    k_fa2_f16<128><<<grd, 128, 0, stream>>>(Q, K, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, scale, block_causal_size);
  } else {
    return -2;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 4B. Fast Tier: Native SageAttention INT8 Q/K (PTX m16n8k32 s8) + FP16 P*V
// ===========================================================================
// Step 1: Quantize each token row of [H, n_pad, D] F16 -> INT8 + F32 per-row scale (1 warp / row)

__global__ void k_quant_rows_i8(
    const __half *__restrict__ src, int8_t *__restrict__ dst, float *__restrict__ sc,
    int total_rows, int D) {
  const long row = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= total_rows) return;
  const __half *sr = src + row * D;
  int8_t *dr = dst + row * D;

  float amax = 0.0f;
  for (int d = lane; d < D; d += 32) {
    amax = fmaxf(amax, fabsf(__half2float(sr[d])));
  }
  amax = warp_reduce_max(amax);
  const float s = (amax > 1e-8f) ? (amax / 127.0f) : 1.0f;
  const float inv = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;
  if (lane == 0) sc[row] = s;
  for (int d = lane; d < D; d += 32) {
    int q = __float2int_rn(__half2float(sr[d]) * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dr[d] = (int8_t)q;
  }
}

template <int D>
__global__ void __launch_bounds__(128) k_sage_attn_i8(
    const int8_t *__restrict__ Qi, const float *__restrict__ Qs,
    const int8_t *__restrict__ Ki, const float *__restrict__ Ks,
    const __half *__restrict__ V,
    float *__restrict__ O, __half *__restrict__ O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, float scale, int block_causal_size) {
  constexpr int KC8 = D / 32;  // 2 for D=64, 4 for D=128
  constexpr int DT  = D / 8;
  constexpr int LK8 = D + 16;  // int8 stride padded by 16B for conflict-free access
  constexpr int LV  = 72;

  __shared__ __align__(16) int8_t sK8[64 * LK8];
  __shared__ __align__(16) __half sVt[D * LV];
  __shared__ float sKsc[64];

  const int hh = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int qr = blockIdx.x * 64 + warp * 16;
  const int E = n_heads * D;

  const int8_t *qg = Qi + (long)hh * n_q_pad * D;
  const float  *qsg = Qs + (long)hh * n_q_pad;
  const int8_t *kg = Ki + (long)hh * n_kv_pad * D;
  const float  *ksg = Ks + (long)hh * n_kv_pad;
  const __half *vg = V  + (long)hh * n_kv_pad * D;

  int kv_limit_blk = n_kv;
  int kv_lim0 = n_kv, kv_lim1 = n_kv;
  if (block_causal_size > 0) {
    const int max_q_blk = min(blockIdx.x * 64 + 63, n_q - 1);
    kv_limit_blk = min(n_kv, ((max_q_blk / block_causal_size) + 1) * block_causal_size);
    const int q0 = qr + g, q1 = qr + g + 8;
    kv_lim0 = (q0 < n_q) ? min(n_kv, ((q0 / block_causal_size) + 1) * block_causal_size) : 0;
    kv_lim1 = (q1 < n_q) ? min(n_kv, ((q1 / block_causal_size) + 1) * block_causal_size) : 0;
  }

  // Load INT8 Q fragments into registers (HALF the register footprint of FP16!)
  const int r0 = min(qr + g, n_q_pad - 1);
  const int r8 = min(qr + g + 8, n_q_pad - 1);
  const float qs0 = qsg[r0] * scale;
  const float qs1 = qsg[r8] * scale;

  uint32_t qa[KC8][4];
  {
    const int8_t *q0p = qg + (long)r0 * D;
    const int8_t *q8p = qg + (long)r8 * D;
    #pragma unroll
    for (int kc = 0; kc < KC8; kc++) {
      qa[kc][0] = *(const uint32_t *)(q0p + kc * 32 + 4 * t);
      qa[kc][1] = *(const uint32_t *)(q8p + kc * 32 + 4 * t);
      qa[kc][2] = *(const uint32_t *)(q0p + kc * 32 + 16 + 4 * t);
      qa[kc][3] = *(const uint32_t *)(q8p + kc * 32 + 16 + 4 * t);
    }
  }

  float o[DT][4];
  #pragma unroll
  for (int dt = 0; dt < DT; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
  float M0 = -FLT_MAX * 0.5f, M1 = -FLT_MAX * 0.5f, l0 = 0.0f, l1 = 0.0f;

  for (int kb = 0; kb < kv_limit_blk; kb += 64) {
    __syncthreads();
    // Load K_i8 (16B uint4 chunks = 16 int8s -> D/16 chunks per row)
    constexpr int K8_CHUNKS = D / 16;
    for (int i = threadIdx.x; i < 64 * K8_CHUNKS; i += 128) {
      const int r = i / K8_CHUNKS, c = (i % K8_CHUNKS) * 16;
      uint4 kv8 = make_uint4(0, 0, 0, 0);
      if (kb + r < n_kv) kv8 = *(const uint4 *)(kg + (long)(kb + r) * D + c);
      *(uint4 *)(sK8 + r * LK8 + c) = kv8;
    }
    if (threadIdx.x < 64) {
      sKsc[threadIdx.x] = (kb + threadIdx.x < n_kv) ? ksg[kb + threadIdx.x] : 0.0f;
    }
    // Load V transposed into sVt
    for (int i = threadIdx.x; i < 64 * DT; i += 128) {
      const int r = i / DT, c = (i % DT) * 8;
      uint4 vv_val = make_uint4(0, 0, 0, 0);
      if (kb + r < n_kv) vv_val = *(const uint4 *)(vg + (long)(kb + r) * D + c);
      const __half *vv = (const __half *)&vv_val;
      #pragma unroll
      for (int j = 0; j < 8; j++) sVt[(c + j) * LV + r] = vv[j];
    }
    __syncthreads();

    // INT8 Tensor Core QK^T (m16n8k32 = 2x throughput vs FP16!)
    float s[8][4];
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      int32_t si[4] = {0, 0, 0, 0};
      const int8_t *kr = sK8 + (nt * 8 + g) * LK8 + 4 * t;
      #pragma unroll
      for (int kc = 0; kc < KC8; kc++) {
        ec_mma_s8(si, qa[kc], *(const uint32_t *)(kr + kc * 32), *(const uint32_t *)(kr + kc * 32 + 16));
      }
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        const int kl = nt * 8 + 2 * t + j;
        const int kidx = kb + kl;
        const float ksc = sKsc[kl];
        s[nt][j]     = (kidx < kv_lim0) ? ((float)si[j]     * qs0 * ksc) : -INFINITY;
        s[nt][2 + j] = (kidx < kv_lim1) ? ((float)si[2 + j] * qs1 * ksc) : -INFINITY;
        mx0 = fmaxf(mx0, s[nt][j]);
        mx1 = fmaxf(mx1, s[nt][2 + j]);
      }
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float Mn0 = fmaxf(M0, mx0), Mn1 = fmaxf(M1, mx1);
    const float ms0 = (Mn0 > -1e30f) ? expf(M0 - Mn0) : 0.0f;
    const float ms1 = (Mn1 > -1e30f) ? expf(M1 - Mn1) : 0.0f;
    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        s[nt][j]     = (s[nt][j] > -1e30f)     ? expf(s[nt][j] - Mn0)     : 0.0f;
        s[nt][2 + j] = (s[nt][2 + j] > -1e30f) ? expf(s[nt][2 + j] - Mn1) : 0.0f;
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

    #pragma unroll
    for (int kc = 0; kc < 4; kc++) {
      const uint32_t pa[4] = {
        ec_pack_h2(s[2 * kc][0],     s[2 * kc][1]),
        ec_pack_h2(s[2 * kc][2],     s[2 * kc][3]),
        ec_pack_h2(s[2 * kc + 1][0], s[2 * kc + 1][1]),
        ec_pack_h2(s[2 * kc + 1][2], s[2 * kc + 1][3]),
      };
      #pragma unroll
      for (int dt = 0; dt < DT; dt++) {
        const __half *vr = sVt + (dt * 8 + g) * LV + kc * 16 + 2 * t;
        ec_mma_f16(o[dt], pa, *(const uint32_t *)vr, *(const uint32_t *)(vr + 8));
      }
    }
  }

  const float inv0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
  const float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
  #pragma unroll
  for (int half8 = 0; half8 < 2; half8++) {
    const int q = qr + g + 8 * half8;
    if (q >= n_q) continue;
    const float inv = half8 ? inv1 : inv0;
    float *d = O + (long)q * E + hh * D;
    __half *dq = O_f16 ? (O_f16 + (long)q * E + hh * D) : nullptr;
    #pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      const float v0 = o[dt][2 * half8] * inv;
      const float v1 = o[dt][2 * half8 + 1] * inv;
      const int col = dt * 8 + 2 * t;
      *(float2 *)(d + col) = make_float2(v0, v1);
      if (dq) *(__half2 *)(dq + col) = __floats2half2_rn(v0, v1);
    }
  }
}

extern "C" int ec_launch_sage_attn_i8(
    const __half *Q, const __half *K, const __half *V,
    int8_t *q_i8, int8_t *k_i8, float *q_sc, float *k_sc,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size,
    cudaStream_t stream) {
  if (n_q <= 0 || n_kv <= 0 || n_heads <= 0) return -1;
  const int rows_q = n_heads * n_q_pad;
  const int rows_k = n_heads * n_kv_pad;
  k_quant_rows_i8<<<(rows_q + 7) / 8, 256, 0, stream>>>(Q, q_i8, q_sc, rows_q, head_dim);
  k_quant_rows_i8<<<(rows_k + 7) / 8, 256, 0, stream>>>(K, k_i8, k_sc, rows_k, head_dim);

  dim3 grd((n_q + 63) / 64, n_heads);
  if (head_dim == 64) {
    k_sage_attn_i8<64><<<grd, 128, 0, stream>>>(q_i8, q_sc, k_i8, k_sc, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, scale, block_causal_size);
  } else if (head_dim == 128) {
    k_sage_attn_i8<128><<<grd, 128, 0, stream>>>(q_i8, q_sc, k_i8, k_sc, V, O, O_f16, n_q, n_kv, n_q_pad, n_kv_pad, n_heads, scale, block_causal_size);
  } else {
    return -2;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 5. Autoencoder Primitives: GroupNorm+SiLU, Conv1D, Conv2D, Causal Conv3D
// ===========================================================================

__global__ void k_groupnorm_silu(
    const float *__restrict__ x, const float *__restrict__ gamma, const float *__restrict__ beta,
    float *__restrict__ y, int C, int spatial, int num_groups, float eps, int apply_silu) {
  __shared__ float red[32];
  const int g = blockIdx.x;
  const int cpg = C / num_groups;
  const int c0 = g * cpg;
  const long n = (long)cpg * spatial;
  const float *xg = x + (long)c0 * spatial;
  float *yg = y + (long)c0 * spatial;
  const int tid = threadIdx.x;
  const int nwarps = blockDim.x >> 5;

  float s = 0.0f;
  for (long i = tid; i < n; i += blockDim.x) s += xg[i];
  s = warp_reduce_sum(s);
  if ((tid & 31) == 0) red[tid >> 5] = s;
  __syncthreads();
  float bs = (tid < nwarps) ? red[tid] : 0.0f;
  if (tid < 32) bs = warp_reduce_sum(bs);
  if (tid == 0) red[0] = bs / (float)n;
  __syncthreads();
  const float mean = red[0];

  float v = 0.0f;
  for (long i = tid; i < n; i += blockDim.x) {
    float d = xg[i] - mean;
    v += d * d;
  }
  v = warp_reduce_sum(v);
  if ((tid & 31) == 0) red[tid >> 5] = v;
  __syncthreads();
  float bv = (tid < nwarps) ? red[tid] : 0.0f;
  if (tid < 32) bv = warp_reduce_sum(bv);
  if (tid == 0) red[0] = rsqrtf(bv / (float)n + eps);
  __syncthreads();
  const float inv = red[0];

  for (long i = tid; i < n; i += blockDim.x) {
    const int c = c0 + (int)(i / spatial);
    float val = (xg[i] - mean) * inv;
    if (gamma) val *= gamma[c];
    if (beta)  val += beta[c];
    if (apply_silu) val = ec_silu(val);
    yg[i] = val;
  }
}

extern "C" int ec_launch_groupnorm_silu(
    const float *x, const float *gamma, const float *beta,
    float *y, int C, int spatial, int num_groups, float eps, int apply_silu,
    cudaStream_t stream) {
  if (C <= 0 || spatial <= 0 || num_groups <= 0 || (C % num_groups)) return -1;
  k_groupnorm_silu<<<num_groups, 256, 0, stream>>>(x, gamma, beta, y, C, spatial, num_groups, eps, apply_silu);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_conv1d(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ bias,
    float *__restrict__ y, int Cin, int Cout, int L, int L_out,
    int K, int stride, int pad_left, int dilation) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)Cout * L_out) return;
  const int co = (int)(idx / L_out);
  const int lo = (int)(idx % L_out);
  float acc = bias ? bias[co] : 0.0f;
  const int base_l = lo * stride - pad_left;
  for (int ci = 0; ci < Cin; ci++) {
    const float *xc = x + (long)ci * L;
    const float *wc = w + ((long)co * Cin + ci) * K;
    for (int k = 0; k < K; k++) {
      const int li = base_l + k * dilation;
      if (li >= 0 && li < L) acc += xc[li] * wc[k];
    }
  }
  y[idx] = acc;
}

extern "C" int ec_launch_conv1d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int L, int K, int stride, int pad, int dilation, int causal,
    cudaStream_t stream) {
  const int pad_left = causal ? (K - 1) * dilation : pad;
  const int pad_total = causal ? (K - 1) * dilation : 2 * pad;
  const int L_out = (L + pad_total - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) return -1;
  const long total = (long)Cout * L_out;
  k_conv1d<<<(int)((total + 255) / 256), 256, 0, stream>>>(x, w, bias, y, Cin, Cout, L, L_out, K, stride, pad_left, dilation);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_conv2d(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ bias,
    float *__restrict__ y, int Cin, int Cout, int H, int W, int H_out, int W_out,
    int Kh, int Kw, int stride, int pad) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long hw_out = (long)H_out * W_out;
  if (idx >= (long)Cout * hw_out) return;
  const int co = (int)(idx / hw_out);
  const int rem = (int)(idx % hw_out);
  const int ho = rem / W_out, wo = rem % W_out;
  float acc = bias ? bias[co] : 0.0f;
  for (int ci = 0; ci < Cin; ci++) {
    const float *xc = x + (long)ci * H * W;
    const float *wc = w + ((long)co * Cin + ci) * Kh * Kw;
    for (int kh = 0; kh < Kh; kh++) {
      const int hi = ho * stride - pad + kh;
      if (hi < 0 || hi >= H) continue;
      for (int kw = 0; kw < Kw; kw++) {
        const int wi = wo * stride - pad + kw;
        if (wi >= 0 && wi < W) acc += xc[(long)hi * W + wi] * wc[kh * Kw + kw];
      }
    }
  }
  y[idx] = acc;
}

extern "C" int ec_launch_conv2d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int H, int W, int Kh, int Kw, int stride, int pad,
    cudaStream_t stream) {
  const int H_out = (H + 2 * pad - Kh) / stride + 1;
  const int W_out = (W + 2 * pad - Kw) / stride + 1;
  if (H_out <= 0 || W_out <= 0) return -1;
  const long total = (long)Cout * H_out * W_out;
  k_conv2d<<<(int)((total + 255) / 256), 256, 0, stream>>>(x, w, bias, y, Cin, Cout, H, W, H_out, W_out, Kh, Kw, stride, pad);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Causal Conv3D for Video VAE streaming chunks (zero future lookahead; optional temporal boundary cache)
__global__ void k_conv3d_causal(
    const float *__restrict__ x, const float *__restrict__ prev_cache,
    const float *__restrict__ w, const float *__restrict__ bias, float *__restrict__ y,
    int Cin, int Cout, int T, int H, int W, int H_out, int W_out,
    int Kt, int Kh, int Kw, int pad_hw) {
  const long thw_out = (long)T * H_out * W_out;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)Cout * thw_out) return;
  const int co = (int)(idx / thw_out);
  const long rem = idx % thw_out;
  const int to = (int)(rem / ((long)H_out * W_out));
  const int rem2 = (int)(rem % ((long)H_out * W_out));
  const int ho = rem2 / W_out, wo = rem2 % W_out;

  float acc = bias ? bias[co] : 0.0f;
  const int past = Kt - 1;
  for (int ci = 0; ci < Cin; ci++) {
    const float *xc = x + (long)ci * T * H * W;
    const float *pc = prev_cache ? (prev_cache + (long)ci * past * H * W) : nullptr;
    const float *wc = w + ((long)co * Cin + ci) * Kt * Kh * Kw;
    for (int kt = 0; kt < Kt; kt++) {
      const int ti = to - past + kt; // causal: only attends to ti <= to
      const float *plane = nullptr;
      if (ti >= 0) {
        plane = xc + (long)ti * H * W;
      } else if (pc) {
        plane = pc + (long)(ti + past) * H * W;
      } else {
        continue; // zero-pad past if no prev_cache
      }
      for (int kh = 0; kh < Kh; kh++) {
        const int hi = ho - pad_hw + kh;
        if (hi < 0 || hi >= H) continue;
        for (int kw = 0; kw < Kw; kw++) {
          const int wi = wo - pad_hw + kw;
          if (wi >= 0 && wi < W) {
            acc += plane[(long)hi * W + wi] * wc[(kt * Kh + kh) * Kw + kw];
          }
        }
      }
    }
  }
  y[idx] = acc;
}

extern "C" int ec_launch_conv3d_causal_f32(
    const float *x, const float *prev_cache, const float *w, const float *bias, float *y,
    int Cin, int Cout, int T, int H, int W, int Kt, int Kh, int Kw, int pad_hw,
    cudaStream_t stream) {
  const int H_out = H + 2 * pad_hw - Kh + 1;
  const int W_out = W + 2 * pad_hw - Kw + 1;
  if (T <= 0 || H_out <= 0 || W_out <= 0) return -1;
  const long total = (long)Cout * T * H_out * W_out;
  k_conv3d_causal<<<(int)((total + 255) / 256), 256, 0, stream>>>(
      x, prev_cache, w, bias, y, Cin, Cout, T, H, W, H_out, W_out, Kt, Kh, Kw, pad_hw);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 6. Sparse 3D O-Voxel Hash Table + Submanifold 3x3x3 Conv (TRELLIS.2 Native 3D)
// ===========================================================================

__device__ __forceinline__ uint64_t ov_pack_key(int32_t x, int32_t y, int32_t z) {
  const uint64_t ux = (uint64_t)(x + 32768) & 0x1FFFFFULL;
  const uint64_t uy = (uint64_t)(y + 32768) & 0x1FFFFFULL;
  const uint64_t uz = (uint64_t)(z + 32768) & 0x1FFFFFULL;
  return ((ux << 42) | (uy << 21) | uz) + 1ULL; // 0 = empty sentinel
}

__device__ __forceinline__ uint32_t ov_hash(uint64_t k, uint32_t mask) {
  k ^= k >> 30;
  k *= 0xbf58476d1ce4e5b9ULL;
  k ^= k >> 27;
  k *= 0x94d049bb133111ebULL;
  k ^= k >> 31;
  return (uint32_t)k & mask;
}

__global__ void k_ov_hash_insert(
    const int32_t *__restrict__ coords, uint64_t *__restrict__ keys, int32_t *__restrict__ vals,
    int n_active, uint32_t mask) {
  const int i = (int)((long)blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_active) return;
  const uint64_t key = ov_pack_key(coords[3 * i + 0], coords[3 * i + 1], coords[3 * i + 2]);
  uint32_t slot = ov_hash(key, mask);
  while (true) {
    unsigned long long prev = atomicCAS((unsigned long long *)&keys[slot], 0ULL, (unsigned long long)key);
    if (prev == 0ULL || prev == (unsigned long long)key) {
      vals[slot] = i;
      return;
    }
    slot = (slot + 1u) & mask;
  }
}

extern "C" int ec_launch_ovoxel_hash_build(ec_ovoxel_t *ov, cudaStream_t stream) {
  if (!ov || ov->n_active <= 0 || ov->hash_cap < 2 * ov->n_active || (ov->hash_cap & (ov->hash_cap - 1)))
    return -1;
  cudaMemsetAsync(ov->hash_keys_d, 0, (size_t)ov->hash_cap * sizeof(uint64_t), stream);
  cudaMemsetAsync(ov->hash_vals_d, 0xFF, (size_t)ov->hash_cap * sizeof(int32_t), stream);
  const int grd = (ov->n_active + 255) / 256;
  k_ov_hash_insert<<<grd, 256, 0, stream>>>(ov->coords_d, ov->hash_keys_d, ov->hash_vals_d, ov->n_active, (uint32_t)(ov->hash_cap - 1));
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__device__ __forceinline__ int32_t ov_lookup(
    const uint64_t *__restrict__ keys, const int32_t *__restrict__ vals,
    uint32_t mask, int32_t x, int32_t y, int32_t z) {
  const uint64_t key = ov_pack_key(x, y, z);
  uint32_t slot = ov_hash(key, mask);
  while (true) {
    const uint64_t cur = keys[slot];
    if (cur == key) return vals[slot];
    if (cur == 0ULL) return -1;
    slot = (slot + 1u) & mask;
  }
}

__global__ void k_ov_sparse_conv3d(
    const int32_t *__restrict__ coords,
    const uint64_t *__restrict__ keys, const int32_t *__restrict__ vals,
    const float *__restrict__ feats_in, const float *__restrict__ w, const float *__restrict__ bias,
    float *__restrict__ feats_out,
    int n_active, int Cin, int Cout, uint32_t mask) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)n_active * Cout) return;
  const int vi = (int)(idx / Cout);
  const int co = (int)(idx % Cout);
  const int32_t x = coords[3 * vi + 0];
  const int32_t y = coords[3 * vi + 1];
  const int32_t z = coords[3 * vi + 2];

  float acc = bias ? bias[co] : 0.0f;
  int k = 0;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++, k++) {
        const int32_t nb = (dx == 0 && dy == 0 && dz == 0) ? vi : ov_lookup(keys, vals, mask, x + dx, y + dy, z + dz);
        if (nb < 0) continue;
        const float *fn = feats_in + (long)nb * Cin;
        const float *wk = w + ((long)k * Cout + co) * Cin;
        for (int ci = 0; ci < Cin; ci++) acc += fn[ci] * wk[ci];
      }
    }
  }
  feats_out[idx] = acc;
}

extern "C" int ec_launch_ovoxel_sparse_conv3d(
    const ec_ovoxel_t *ov, const float *feats_in,
    const float *w, const float *bias, float *feats_out,
    int Cout, cudaStream_t stream) {
  if (!ov || ov->n_active <= 0 || Cout <= 0) return -1;
  const long total = (long)ov->n_active * Cout;
  k_ov_sparse_conv3d<<<(int)((total + 255) / 256), 256, 0, stream>>>(
      ov->coords_d, ov->hash_keys_d, ov->hash_vals_d,
      feats_in, w, bias, feats_out, ov->n_active, ov->channels, Cout, (uint32_t)(ov->hash_cap - 1));
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 7. Flow-Matching CFG Euler Solver + MagCache L1 Probe
// ===========================================================================

__global__ void k_flow_cfg_euler(
    const float *__restrict__ x, const float *__restrict__ vc, const float *__restrict__ vu,
    float *__restrict__ x_next, long n, float dt, float cfg) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = vu ? (vu[i] + cfg * (vc[i] - vu[i])) : vc[i];
  x_next[i] = x[i] + dt * v;
}

extern "C" int ec_launch_flow_cfg_euler(
    const float *x, const float *v_cond, const float *v_uncond,
    float *x_next, int n_elems, float dt, float cfg_scale,
    cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  k_flow_cfg_euler<<<(n_elems + 255) / 256, 256, 0, stream>>>(x, v_cond, v_uncond, x_next, n_elems, dt, cfg_scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_l1_rel_diff(
    const float *__restrict__ a, const float *__restrict__ b,
    float *__restrict__ stats, int n) {
  __shared__ float sd[32], sr[32];
  const int tid = threadIdx.x;
  float ld = 0.0f, lr = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
    ld += fabsf(a[i] - b[i]);
    lr += fabsf(b[i]);
  }
  ld = warp_reduce_sum(ld);
  lr = warp_reduce_sum(lr);
  if ((tid & 31) == 0) { sd[tid >> 5] = ld; sr[tid >> 5] = lr; }
  __syncthreads();
  const int nw = blockDim.x >> 5;
  if (tid < 32) {
    float bd = (tid < nw) ? sd[tid] : 0.0f;
    float br = (tid < nw) ? sr[tid] : 0.0f;
    bd = warp_reduce_sum(bd);
    br = warp_reduce_sum(br);
    if (tid == 0) {
      atomicAdd(&stats[0], bd);
      atomicAdd(&stats[1], br);
    }
  }
}

extern "C" int ec_launch_l1_rel_diff(
    const float *a, const float *b, float *out_stats_d, int n_elems,
    cudaStream_t stream) {
  if (n_elems <= 0) return -1;
  cudaMemsetAsync(out_stats_d, 0, 2 * sizeof(float), stream);
  int grd = (n_elems + 1023) / 1024;
  if (grd > 256) grd = 256;
  k_l1_rel_diff<<<grd, 256, 0, stream>>>(a, b, out_stats_d, n_elems);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ===========================================================================
// 8. Realtime Video Self-Forcing Rolling KV-Cache Ring Append
// ===========================================================================

__global__ void k_kv_ring_append(
    __half *__restrict__ k_ring, __half *__restrict__ v_ring,
    const __half *__restrict__ ck, const __half *__restrict__ cv,
    int n_heads, int cap_tokens, int head_dim, int write_pos, int chunk_tokens) {
  const long total = (long)n_heads * chunk_tokens * head_dim;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int d = (int)(idx % head_dim);
  const long rem = idx / head_dim;
  const int t = (int)(rem % chunk_tokens);
  const int h = (int)(rem / chunk_tokens);

  const int dst_t = (write_pos + t) % cap_tokens;
  const long dst_o = ((long)h * cap_tokens + dst_t) * head_dim + d;
  k_ring[dst_o] = ck[idx];
  v_ring[dst_o] = cv[idx];
}

extern "C" int ec_launch_kv_ring_append(
    ec_kv_ring_t *ring,
    const __half *chunk_k, const __half *chunk_v,
    int chunk_tokens, cudaStream_t stream) {
  if (!ring || chunk_tokens <= 0 || chunk_tokens > ring->cap_tokens) return -1;
  const long total = (long)ring->n_heads * chunk_tokens * ring->head_dim;
  k_kv_ring_append<<<(int)((total + 255) / 256), 256, 0, stream>>>(
      (__half *)ring->k_ring_d, (__half *)ring->v_ring_d,
      chunk_k, chunk_v, ring->n_heads, ring->cap_tokens, ring->head_dim,
      ring->write_pos, chunk_tokens);
  ring->write_pos = (ring->write_pos + chunk_tokens) % ring->cap_tokens;
  ring->valid_tokens += chunk_tokens;
  if (ring->valid_tokens > ring->cap_tokens) ring->valid_tokens = ring->cap_tokens;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
