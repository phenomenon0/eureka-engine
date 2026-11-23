// ec_ops.h — Unified High-Throughput Native PTX Ampere+ Operator Core
// Shared across FLUX.2-klein-4B, ACE-Step 1.5, TRELLIS.2, and Wan 2.2.
// 100% pure C11/CUDA PTX (m16n8k16 BF16 Tensor Cores + cp.async double buffering).
// Zero cuBLAS, zero ggml, zero external ML libraries.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1. Unified High-Throughput BF16 TensorCore GEMM:
//    Y[M, N] = X_bf16[M, K] @ W_bf16[N, K]^T + optional bias[N]
//    Auto-dispatches:
//      - Vectorized warp-reduced BF16 GEMV for M <= 4 (timestep/modulation MLPs)
//      - Double-buffered cp.async BM=128xBN=128xBK=32 (8-warp, L2-swizzled) for M>=64, N>=64
//      - Bank-conflict-free BM=64xBN=64xBK=32 fallback for small/unaligned/odd-K shapes
int ec_ops_gemm_bf16(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16,
    int M, int N, int K, cudaStream_t stream);

// 2. Unified Native BF16 TensorCore FlashAttention-2 (D = 64 or 128, full / GQA / sliding-window):
//    Q: [Hq, Lq_pad, D] BF16, K/V: [Hkv, Lkv_pad, D] BF16
//    -> O_f32: optional [Lq, Hq*D] F32, O_bf16: optional [Lq, Hq*D] BF16
//    Supports arbitrary Lq != Lkv (self- and cross-attention), GQA (Hq % Hkv == 0),
//    and optional symmetric sliding window (window > 0: |k - q| <= window; window <= 0: full).
int ec_ops_fa2_bf16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V,
    float *O_f32, uint16_t *O_bf16,
    int Lq, int Lkv, int Lq_pad, int Lkv_pad,
    int Hq, int Hkv, int D, float scale, int window,
    cudaStream_t stream);

// 3. GPU-Resident Activation & Flow-Matching CFG/Euler Kernels (Zero Host Roundtrips!)
int ec_ops_silu_f32_to_bf16(const float *src, uint16_t *dst, long n, cudaStream_t stream);

// Wan 2.2 GPU-resident CFG + Euler + FasterCache delta reuse:
//   mode=0 (no CFG):          xt -= dt * vp
//   mode=1 (full CFG + save): d = vp - vn; delta_cache = d; xt -= dt * (vn + cfg * d)
//   mode=2 (FasterCache):     xt -= dt * (vp + (cfg - 1) * delta_cache)
int ec_ops_wan_cfg_euler(
    float *xt, const float *vp, const float *vn, float *delta_cache,
    long n, float dt, float cfg_scale, int mode, cudaStream_t stream);

// TRELLIS.2 GPU-resident CFG + x0-space variance rescale (guidance_rescale=0.7) + Euler step:
//   use_cfg=0: xt -= dt * vp
//   use_cfg=1: single-CTA exact double-precision variance rescale + Euler in-place on GPU
int ec_ops_trellis_cfg_euler(
    float *xt, const float *vp, const float *vn,
    int n, float dt, float t_cur, float cfg_scale,
    float guidance_rescale, float sigma_min, int use_cfg,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
