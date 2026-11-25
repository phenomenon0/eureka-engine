// ec_kernels.cuh — 100% Native Eureka Creative CUDA Kernel API
// Zero cuBLAS, zero ggml, zero external ML libraries.
// Covers all 8 core GPU primitive families required for Image, Video, Realtime Video,
// 3D Sparse O-Voxel, and Music DiT + Autoencoder inference.

#pragma once

#include "ec_hal.h"
#include "ec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 1. Fused adaLN-Zero (RMSNorm or LayerNorm + shift/scale) & Gated Residual
// ---------------------------------------------------------------------------
// y[row, c] = norm(x[row, c]) * (1 + scale[c]) + shift[c]
// norm_mode: 0 = LayerNorm (no affine), 1 = RMSNorm (no affine), 2 = RMSNorm * w[c]
// Optional y_f16 != NULL writes F16 copy in the same memory pass for Tensor Core GEMM input.
int ec_launch_adaln_zero(
    const float *x, const float *norm_w,
    const float *shift, const float *scale,
    float *y, __half *y_f16,
    int n_rows, int dim, float eps, int norm_mode,
    cudaStream_t stream);

// x[row, c] += gate[c] * h[row, c]
int ec_launch_gated_res(
    float *x, const float *h, const float *gate,
    int n_rows, int dim, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 2. Generalized N-D RoPE (1D Audio, 2D Image, 3D Video/O-Voxel)
// ---------------------------------------------------------------------------
// Applies RoPE to Q and K in-place (or writes to q_out/k_out F16 heads [H, N_pad, D]).
// ndims: 1, 2, or 3 axes; axis_dims[a] = even channel count for axis a (sum <= D).
// coords: [N, ndims] int32 coordinates (e.g., [t], [y, x], or [t, y, x]).
// Optional qk_norm_w_q / qk_norm_w_k: per-head RMSNorm weights [D] fused BEFORE RoPE!
int ec_launch_rope_nd(
    const float *q_in, const float *k_in, const float *v_in,
    const int32_t *coords,
    const float *qk_norm_w_q, const float *qk_norm_w_k,
    __half *q_out, __half *k_out, __half *v_out,
    int n_tokens, int n_pad, int n_heads, int head_dim,
    int ndims, const int32_t axis_dims[3], float theta_base, float eps,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 3. Tensor-Core FP16 GEMM (mma.sync.aligned.m16n8k16) + Epilogue + Hot LoRA
// ---------------------------------------------------------------------------
// Computes Y[M, N] = X[M, K] @ W[N, K]^T + optional bias/SiLU/GELU/residual.
// W is row-major [N, K] F16 (matches Safetensors/GGUF Linear weight layout directly!).
// epilogue: 0=none, 1=bias, 2=bias+SiLU, 3=bias+GELU, 4=bias+residual(Y += res)
int ec_launch_gemm_f16(
    const __half *X, const __half *W, const float *bias, const float *res,
    float *Y, __half *Y_f16,
    int M, int N, int K, int epilogue,
    cudaStream_t stream);

// Hot-fused LoRA delta: Y[M, N] += (X[M, K] @ A[R, K]^T) @ B[N, R]^T * scale
// Uses scratch buffer tmp_f16 [M * R] and tmp_f32 [M * R]. Zero base weight reload.
int ec_launch_lora_add(
    const __half *X, const ec_lora_t *lora,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream);

// Native INT8 W8A8 Tensor-Core GEMM (PTX mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32)
// Y[M, N] = (X_i8[M, K] @ W_i8[N, K]^T) * (x_sc[M] * w_sc[N]) + optional bias[N]
// Targets 100+ TOPS on RTX 3090 with zero cuBLAS!
int ec_launch_gemm_i8(
    const int8_t *X_i8, const float *x_sc,
    const int8_t *W_i8, const float *w_sc,
    const float *bias, float *Y,
    int M, int N, int K, cudaStream_t stream);

// Native SVDQuant W4A4 + FP16 Low-Rank Outlier GEMM (100% own-coded Nunchaku-class 4-bit DiT path)
// W_i4 packed as [N, K/2] (low nibble = even k, high nibble = odd k, signed -8..7 via offset +8)
// w_sc: per-32-group scales [N, K/32]; low-rank outlier branch passed as ec_lora_t (rank 16/32)
int ec_launch_gemm_svd4(
    const __half *X_f16,
    const uint8_t *W_i4, const float *w_sc,
    const ec_lora_t *outlier_branch,
    float *tmp_f32, __half *tmp_f16, float *Y,
    int M, int N, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 4. Dual-Tier FlashAttention-2 (Exact FP16 Oracle) & SageAttention (Fast INT8 Q/K)
// ---------------------------------------------------------------------------
// Q: [H, n_q_pad, D] F16, K/V: [H, n_kv_pad, D] F16 -> O: [n_q, H*D] F32 (+ optional F16)
// Supports D = 64 or 128.
// block_causal_size: 0 = full bidirectional (Image/Video/3D/Music DiT);
//                    >0 = block-causal chunk size (Self-Forcing realtime video chunks!)
int ec_launch_fa2_f16(
    const __half *Q, const __half *K, const __half *V,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size,
    cudaStream_t stream);

// SageAttention Fast Tier: quantizes Q and K per 64-token block to INT8 with per-row scales,
// computes S = Q_i8 @ K_i8^T via PTX mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32,
// online softmax in F32, and P @ V via mma.sync.aligned.m16n8k16!
// Scratch buffers: q_i8 [H * n_q_pad * D], k_i8 [H * n_kv_pad * D],
//                  q_sc [H * n_q_pad], k_sc [H * n_kv_pad].
int ec_launch_sage_attn_i8(
    const __half *Q, const __half *K, const __half *V,
    int8_t *q_i8, int8_t *k_i8, float *q_sc, float *k_sc,
    float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad,
    int n_heads, int head_dim, float scale, int block_causal_size,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 5. Autoencoder Primitives (GroupNorm+SiLU, Conv1D Audio, Conv2D Image, Causal Conv3D Video)
// ---------------------------------------------------------------------------
// Input/Output layout: [C, S] where S = L (1D), H*W (2D), or T*H*W (3D).
int ec_launch_groupnorm_silu(
    const float *x, const float *gamma, const float *beta,
    float *y, int C, int spatial, int num_groups, float eps, int apply_silu,
    cudaStream_t stream);

// Conv1D: x [Cin, L] -> y [Cout, L_out], w [Cout, Cin, K] (Audio DCAE for ACE-Step/DiffRhythm)
// causal=1 pads left by (K-1)*dilation, causal=0 uses symmetric pad.
int ec_launch_conv1d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int L, int K, int stride, int pad, int dilation, int causal,
    cudaStream_t stream);

// Conv2D: x [Cin, H, W] -> y [Cout, H_out, W_out], w [Cout, Cin, Kh, Kw] (Image VAE)
int ec_launch_conv2d_f32(
    const float *x, const float *w, const float *bias, float *y,
    int Cin, int Cout, int H, int W, int Kh, int Kw, int stride, int pad,
    cudaStream_t stream);

// Causal Conv3D: x [Cin, T, H, W] -> y [Cout, T, H_out, W_out], w [Cout, Cin, Kt, Kh, Kw]
// Causal in time (pads past by Kt-1, zero future lookahead — enables streaming Video VAE!)
// Optional prev_cache [Cin, Kt-1, H, W] supplies exact temporal boundary from previous chunk!
int ec_launch_conv3d_causal_f32(
    const float *x, const float *prev_cache, const float *w, const float *bias, float *y,
    int Cin, int Cout, int T, int H, int W, int Kt, int Kh, int Kw, int pad_hw,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 6. Sparse 3D O-Voxel Primitives (TRELLIS.2 Native 3D Engine)
// ---------------------------------------------------------------------------
// Builds open-addressing GPU spatial hash table from coords_d [n_active, 3]
int ec_launch_ovoxel_hash_build(ec_ovoxel_t *ov, cudaStream_t stream);

// Submanifold 3x3x3 Sparse Conv: feats_in [N, Cin] -> feats_out [N, Cout]
// Weight layout: w [27, Cout, Cin] F32, bias [Cout] F32 (optional).
int ec_launch_ovoxel_sparse_conv3d(
    const ec_ovoxel_t *ov, const float *feats_in,
    const float *w, const float *bias, float *feats_out,
    int Cout, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 7. Rectified-Flow / Flow-Matching Solver + MagCache Residual Probe
// ---------------------------------------------------------------------------
// x_next = x + dt * (v_uncond + cfg_scale * (v_cond - v_uncond))
// If v_uncond == NULL, runs guidance-distilled path: x_next = x + dt * v_cond.
int ec_launch_flow_cfg_euler(
    const float *x, const float *v_cond, const float *v_uncond,
    float *x_next, int n_elems, float dt, float cfg_scale,
    cudaStream_t stream);

// Computes L1 norm of (a - b) and L1 norm of b on GPU into out_stats[2] = {diff_l1, ref_l1}
// Drives MagCache / TeaCache adaptive residual-skip decisions with minimal sync.
int ec_launch_l1_rel_diff(
    const float *a, const float *b, float *out_stats_d, int n_elems,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// 8. Realtime Video Self-Forcing Rolling KV-Cache Ring
// ---------------------------------------------------------------------------
// Appends chunk_k, chunk_v ([H, chunk_tokens, D] F16) into rolling ring buffer.
int ec_launch_kv_ring_append(
    ec_kv_ring_t *ring,
    const __half *chunk_k, const __half *chunk_v,
    int chunk_tokens, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
