// ec_types.h — 100% Native Eureka Creative Engine Core Types & Generator Contract
// Implements the "Generator" module class from docs/reference/module-contract.md
// across all 5 creative modalities: Image, Video, Realtime Video, 3D (O-Voxel/Mesh/3DGS), Music.
// Zero third-party ML frameworks (no ggml, no PyTorch, no cuBLAS).

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 1. Dtypes & Execution Tiers (Tolerance Doctrine)
// ---------------------------------------------------------------------------

typedef enum {
  EC_DTYPE_F32     = 0,
  EC_DTYPE_F16     = 1,
  EC_DTYPE_BF16    = 2,
  EC_DTYPE_FP8_E4M3 = 3,
  EC_DTYPE_I8      = 4,
  EC_DTYPE_U8      = 5,
  EC_DTYPE_I32     = 6,
  EC_DTYPE_U32     = 7,
  EC_DTYPE_Q8_0    = 8,
  EC_DTYPE_Q4_K    = 9,
  EC_DTYPE_Q6_K    = 10,
  EC_DTYPE_SVD4    = 11,  // W4A4 + FP16 low-rank outlier branch (100% native)
  EC_DTYPE_I64     = 12,
  EC_DTYPE_F64     = 13,
} ec_dtype_t;

typedef enum {
  EC_TIER_EXACT = 0,  // Oracle reference: FP16/F32 FlashAttention-2, full steps, no lossy skips
  EC_TIER_FAST  = 1,  // Production fly tier: SageAttention INT8 Q/K + CUDA Graph + MagCache
} ec_tier_t;

typedef enum {
  EC_MODALITY_IMAGE         = 0,  // 2D RoPE DiT + 2D VAE (Z-Image, FLUX.2, Qwen-Image)
  EC_MODALITY_VIDEO         = 1,  // 3D RoPE DiT + Causal 3D VAE (Wan 2.2, LTX-2.5)
  EC_MODALITY_VIDEO_RT      = 2,  // Block-causal KV-cached Self-Forcing streaming DiT
  EC_MODALITY_3D_OVOXEL     = 3,  // Sparse 3D O-Voxel DiT + SC-VAE (TRELLIS.2)
  EC_MODALITY_MUSIC         = 4,  // 1D RoPE DiT + Audio DCAE (ACE-Step 1.5, DiffRhythm 2)
} ec_modality_t;

// ---------------------------------------------------------------------------
// 2. Unified Typed Tensor & Creative Object Handles (DLPack-aligned semantics)
// ---------------------------------------------------------------------------

typedef enum {
  EC_DEV_CPU    = 0,
  EC_DEV_CUDA   = 1,
  EC_DEV_METAL  = 2,
  EC_DEV_VULKAN = 3,
} ec_device_t;

typedef struct {
  void       *data;
  ec_dtype_t  dtype;
  ec_device_t device;
  int32_t     rank;
  int64_t     shape[5];     // up to 5D: [B, C, T, H, W] or [N, H, D]
  int64_t     stride[5];    // element strides
  size_t      nbytes;
  uint64_t    content_hash; // content-addressed DAG identity
} ec_tensor_t;

// Sparse 3D O-Voxel structured latent (TRELLIS.2 native representation)
typedef struct {
  int32_t    *coords_d;     // [n_active, 3] int32 (x, y, z) on device
  void       *feats_d;      // [n_active, channels] F16/F32 on device
  uint64_t   *hash_keys_d;  // open-addressing spatial hash table keys (packed xyz + 1)
  int32_t    *hash_vals_d;  // open-addressing hash table active index (-1 = empty)
  int32_t     n_active;
  int32_t     channels;
  int32_t     grid_res;     // e.g. 64, 128, 256, 512
  int32_t     hash_cap;     // power-of-2 >= 2 * n_active
} ec_ovoxel_t;

// Native 3D Mesh with PBR maps (exports directly to GLB 2.0)
typedef struct {
  float      *positions;    // [n_verts * 3]
  float      *normals;      // [n_verts * 3] (optional)
  float      *uvs;          // [n_verts * 2] (optional)
  uint32_t   *indices;      // [n_tris * 3]
  uint32_t    n_verts;
  uint32_t    n_tris;
  float       base_color[4];
  float       metallic;
  float       roughness;
} ec_mesh_t;

// Native 3D Gaussian Splat cloud (exports directly to .ply / WebGPU splat viewer)
typedef struct {
  float      *pos;          // [n_splats * 3]
  float      *scale;        // [n_splats * 3] log-scale
  float      *rot;          // [n_splats * 4] quaternion (w,x,y,z)
  float      *opacity;      // [n_splats] logit opacity
  float      *sh_dc;        // [n_splats * 3] RGB DC spherical harmonic
  uint32_t    n_splats;
} ec_splat3d_t;

// Native Audio buffer (Music, Stems, Foley, Voiceover — exports to WAV)
typedef struct {
  float      *pcm;          // interleaved [-1, 1], length = n_frames * channels
  uint32_t    n_frames;
  uint32_t    sample_rate;  // e.g. 44100 or 48000
  uint16_t    channels;     // 1 = mono, 2 = stereo
} ec_audio_t;

// Hot-fused LoRA adapter (no base weight reload)
typedef struct {
  const void *A_d;          // [rank, K] F16
  const void *B_d;          // [N, rank] F16
  int32_t     rank;
  float       scale;        // alpha / rank
} ec_lora_t;

// ---------------------------------------------------------------------------
// 3. MagCache State & Realtime Self-Forcing Rolling KV-Cache
// ---------------------------------------------------------------------------

typedef struct {
  int32_t     n_steps;
  float       accum_err;
  float       err_thresh;     // frozen tolerance threshold (e.g. 0.08)
  int32_t     max_consec_skip;
  int32_t     consec_skips;
  int32_t     warmup_steps;   // always compute first N steps exactly
  int32_t     cooldown_steps; // always compute final N steps exactly
  int32_t     skipped_count;
  float      *cached_res_d;   // [N * D] cached block residual on GPU
  const float *mag_ratios;    // optional calibrated magnitude decay table [n_steps]
} ec_magcache_t;

typedef struct {
  void       *k_ring_d;       // [n_heads, cap_tokens, head_dim] F16
  void       *v_ring_d;       // [n_heads, cap_tokens, head_dim] F16
  int32_t     n_heads;
  int32_t     cap_tokens;     // sliding window capacity (e.g., 4..16 chunks)
  int32_t     head_dim;
  int32_t     write_pos;      // next ring write index
  int32_t     valid_tokens;   // current filled count (<= cap_tokens)
} ec_kv_ring_t;

static inline size_t ec_dtype_size(ec_dtype_t dt) {
  switch (dt) {
    case EC_DTYPE_I64: case EC_DTYPE_F64: return 8;
    case EC_DTYPE_F32: case EC_DTYPE_I32: case EC_DTYPE_U32: return 4;
    case EC_DTYPE_F16: case EC_DTYPE_BF16: return 2;
    case EC_DTYPE_FP8_E4M3: case EC_DTYPE_I8: case EC_DTYPE_U8: return 1;
    default: return 0; // block-quantized
  }
}

#ifdef __cplusplus
}
#endif
