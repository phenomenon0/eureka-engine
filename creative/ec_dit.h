// ec_dit.h — Unified 100% Native Eureka Multi-Modal Flow-Matching DiT Engine
// One engine core runs Image (2D), Video (3D), Realtime Video (Block-Causal / KV-Ring),
// 3D Sparse O-Voxel (3D), and Music (1D) DiT blocks with CUDA Graph replay & MagCache.

#pragma once

#include "ec_hal.h"
#include "ec_kernels.cuh"
#include "ec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t       dim;            // hidden size E = n_heads * head_dim
  int32_t       n_heads;
  int32_t       head_dim;       // 64 or 128
  int32_t       ffn_dim;        // e.g. 4 * dim
  int32_t       ndims;          // 1 (Music), 2 (Image), 3 (Video / 3D O-Voxel)
  int32_t       axis_dims[3];   // RoPE dims per axis
  float         rope_theta;
  float         eps;
  int32_t       norm_mode;      // 0=LN, 1=RMSNorm, 2=RMSNorm*w
  int32_t       block_causal;   // 0=bidirectional, >0=chunk size (Self-Forcing realtime video)
  ec_modality_t modality;
} ec_dit_cfg_t;

// Device weight pointers for a single DiT transformer block (F16 matrices, F32 norms/modulations)
typedef struct {
  __half *w_qkv;    // [3 * E, E]
  float  *b_qkv;    // [3 * E] (optional)
  __half *w_proj;   // [E, E]
  float  *b_proj;   // [E] (optional)
  __half *w_fc1;    // [ffn_dim, E]
  float  *b_fc1;    // [ffn_dim] (optional)
  __half *w_fc2;    // [E, ffn_dim]
  float  *b_fc2;    // [E] (optional)
  float  *qk_norm_q;// [head_dim] (optional)
  float  *qk_norm_k;// [head_dim] (optional)
  // adaLN-Zero conditioning vectors [E] for current timestep
  float  *shift_msa;
  float  *scale_msa;
  float  *gate_msa;
  float  *shift_mlp;
  float  *scale_mlp;
  float  *gate_mlp;
  ec_lora_t *lora_qkv; // optional hot-fused LoRA (NULL if none)
} ec_dit_block_w_t;

// Reusable device workspace for zero-malloc step execution & CUDA Graph capture
typedef struct {
  int32_t n_tokens;
  int32_t n_pad;
  int32_t dim;
  int32_t ffn_dim;
  int32_t max_lora_rank;

  float  *norm_f32;   // [N, E]
  __half *norm_f16;   // [N, E]
  float  *qkv_f32;    // [N, 3*E]
  __half *q_h;        // [H, n_pad, D]
  __half *k_h;        // [H, n_pad, D]
  __half *v_h;        // [H, n_pad, D]
  int8_t *q_i8;       // [H, n_pad, D] (SageAttention fast tier)
  int8_t *k_i8;       // [H, n_pad, D]
  float  *q_sc;       // [H, n_pad]
  float  *k_sc;       // [H, n_pad]
  float  *attn_f32;   // [N, E]
  __half *attn_f16;   // [N, E]
  float  *proj_f32;   // [N, E]
  float  *ffn_f32;    // [N, ffn_dim]
  __half *ffn_f16;    // [N, ffn_dim]
  float  *lora_f32;   // [N, max_lora_rank]
  __half *lora_f16;   // [N, max_lora_rank]
  float  *mag_stats;  // [2] GPU L1 stats

  // Captured CUDA Graph for replay (plus portable replay bindings when stream capture is unsupported)
  cudaGraph_t        graph;
  cudaGraphExec_t    graph_exec;
  int                graph_valid;
  ec_dit_cfg_t       cap_cfg;
  ec_dit_block_w_t  *cap_blocks;
  int32_t            cap_n_blocks;
  float             *cap_x_d;
  const int32_t     *cap_coords_d;
  ec_tier_t          cap_tier;
} ec_dit_ws_t;

int  ec_dit_ws_alloc(ec_dit_ws_t *ws, const ec_dit_cfg_t *cfg, int32_t n_tokens, int32_t max_lora_rank);
void ec_dit_ws_free(ec_dit_ws_t *ws);

// Execute one DiT block in-place on x_d [N, E] (F32)
int ec_dit_block_forward(
    const ec_dit_cfg_t *cfg, const ec_dit_block_w_t *w, ec_dit_ws_t *ws,
    float *x_d, const int32_t *coords_d, ec_tier_t tier, cudaStream_t stream);

// Capture multi-block forward pass into a CUDA Graph and replay it with minimal host overhead
int ec_dit_capture_graph(
    const ec_dit_cfg_t *cfg, const ec_dit_block_w_t *blocks, int32_t n_blocks,
    ec_dit_ws_t *ws, float *x_d, const int32_t *coords_d, ec_tier_t tier, cudaStream_t stream);

int ec_dit_replay_graph(ec_dit_ws_t *ws, cudaStream_t stream);

// MagCache decision helper: returns 1 if step should skip DiT eval and reuse cached_res_d
int ec_magcache_should_skip(ec_magcache_t *mc, int32_t step, float rel_delta);

#ifdef __cplusplus
}
#endif
