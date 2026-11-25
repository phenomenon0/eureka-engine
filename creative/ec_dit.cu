// ec_dit.cu — Unified Multi-Modal Flow-Matching DiT Runner + CUDA Graph Capture + MagCache
// 100% Native Eureka CUDA. Zero external ML frameworks.

#include "ec_dit.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define EC_CK(call) do { if ((call) != cudaSuccess) return -1; } while (0)

int ec_dit_ws_alloc(ec_dit_ws_t *ws, const ec_dit_cfg_t *cfg, int32_t n_tokens, int32_t max_lora_rank) {
  memset(ws, 0, sizeof *ws);
  ws->n_tokens = n_tokens;
  ws->n_pad = (n_tokens + 63) / 64 * 64;
  ws->dim = cfg->dim;
  ws->ffn_dim = cfg->ffn_dim;
  ws->max_lora_rank = max_lora_rank > 0 ? max_lora_rank : 16;

  const size_t N = (size_t)n_tokens;
  const size_t NP = (size_t)ws->n_pad;
  const size_t E = (size_t)cfg->dim;
  const size_t F = (size_t)cfg->ffn_dim;
  const size_t H = (size_t)cfg->n_heads;
  const size_t D = (size_t)cfg->head_dim;

  EC_CK(cudaMalloc(&ws->norm_f32, N * E * sizeof(float)));
  EC_CK(cudaMalloc(&ws->norm_f16, N * E * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->qkv_f32,  N * 3 * E * sizeof(float)));
  EC_CK(cudaMalloc(&ws->q_h,      H * NP * D * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->k_h,      H * NP * D * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->v_h,      H * NP * D * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->q_i8,     H * NP * D * sizeof(int8_t)));
  EC_CK(cudaMalloc(&ws->k_i8,     H * NP * D * sizeof(int8_t)));
  EC_CK(cudaMalloc(&ws->q_sc,     H * NP * sizeof(float)));
  EC_CK(cudaMalloc(&ws->k_sc,     H * NP * sizeof(float)));
  EC_CK(cudaMalloc(&ws->attn_f32, N * E * sizeof(float)));
  EC_CK(cudaMalloc(&ws->attn_f16, N * E * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->proj_f32, N * E * sizeof(float)));
  EC_CK(cudaMalloc(&ws->ffn_f32,  N * F * sizeof(float)));
  EC_CK(cudaMalloc(&ws->ffn_f16,  N * F * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->lora_f32, N * (size_t)ws->max_lora_rank * sizeof(float)));
  EC_CK(cudaMalloc(&ws->lora_f16, N * (size_t)ws->max_lora_rank * sizeof(__half)));
  EC_CK(cudaMalloc(&ws->mag_stats, 2 * sizeof(float)));

  // Zero padded tail of Q/K/V planes once so pad rows never contain NaNs
  EC_CK(cudaMemset(ws->q_h, 0, H * NP * D * sizeof(__half)));
  EC_CK(cudaMemset(ws->k_h, 0, H * NP * D * sizeof(__half)));
  EC_CK(cudaMemset(ws->v_h, 0, H * NP * D * sizeof(__half)));
  return 0;
}

void ec_dit_ws_free(ec_dit_ws_t *ws) {
  if (!ws) return;
  if (ws->graph_exec) cudaGraphExecDestroy(ws->graph_exec);
  if (ws->graph) cudaGraphDestroy(ws->graph);
  free(ws->cap_blocks);
  void *ptrs[] = {
    ws->norm_f32, ws->norm_f16, ws->qkv_f32, ws->q_h, ws->k_h, ws->v_h,
    ws->q_i8, ws->k_i8, ws->q_sc, ws->k_sc, ws->attn_f32, ws->attn_f16,
    ws->proj_f32, ws->ffn_f32, ws->ffn_f16, ws->lora_f32, ws->lora_f16, ws->mag_stats
  };
  for (size_t i = 0; i < sizeof(ptrs) / sizeof(ptrs[0]); i++) cudaFree(ptrs[i]);
  memset(ws, 0, sizeof *ws);
}

// Helper to split interleaved [N, 3*E] QKV output into [N, E] Q, K, V pointers:
// Note: ec_launch_rope_nd expects contiguous Q[N,E], K[N,E], V[N,E].
// So we run a lightweight strided split OR layout w_qkv as 3 stacked blocks [E,E],[E,E],[E,E]
// and compute Q, K, V directly into qkv_f32 + {0, N*E, 2*N*E}! Zero extra transpose kernel!
int ec_dit_block_forward(
    const ec_dit_cfg_t *cfg, const ec_dit_block_w_t *w, ec_dit_ws_t *ws,
    float *x_d, const int32_t *coords_d, ec_tier_t tier, cudaStream_t stream) {
  const int N = ws->n_tokens;
  const int E = cfg->dim;
  const int F = cfg->ffn_dim;
  const float attn_scale = 1.0f / sqrtf((float)cfg->head_dim);

  // 1. Pre-Attention Fused adaLN-Zero -> norm_f16 [N, E]
  if (ec_launch_adaln_zero(x_d, nullptr, w->shift_msa, w->scale_msa,
                           ws->norm_f32, ws->norm_f16, N, E, cfg->eps, cfg->norm_mode, stream))
    return -1;

  // 2. Q, K, V Projections into contiguous planes Q=[0..N*E), K=[N*E..2*N*E), V=[2*N*E..3*N*E)
  float *q_buf = ws->qkv_f32;
  float *k_buf = ws->qkv_f32 + (size_t)N * E;
  float *v_buf = ws->qkv_f32 + (size_t)2 * N * E;
  const __half *wq = w->w_qkv;
  const __half *wk = w->w_qkv + (size_t)E * E;
  const __half *wv = w->w_qkv + (size_t)2 * E * E;
  const float *bq = w->b_qkv ? w->b_qkv : nullptr;
  const float *bk = w->b_qkv ? (w->b_qkv + E) : nullptr;
  const float *bv = w->b_qkv ? (w->b_qkv + 2 * E) : nullptr;

  if (ec_launch_gemm_f16(ws->norm_f16, wq, bq, nullptr, q_buf, nullptr, N, E, E, bq ? 1 : 0, stream)) return -1;
  if (ec_launch_gemm_f16(ws->norm_f16, wk, bk, nullptr, k_buf, nullptr, N, E, E, bk ? 1 : 0, stream)) return -1;
  if (ec_launch_gemm_f16(ws->norm_f16, wv, bv, nullptr, v_buf, nullptr, N, E, E, bv ? 1 : 0, stream)) return -1;

  // Optional Hot-Fused LoRA on Q projection (zero base-weight reload)
  if (w->lora_qkv && w->lora_qkv->rank > 0) {
    if (ec_launch_lora_add(ws->norm_f16, w->lora_qkv, ws->lora_f32, ws->lora_f16, q_buf, N, E, E, stream))
      return -1;
  }

  // 3. N-D RoPE (1D Audio / 2D Image / 3D Video & O-Voxel) + Optional QK-RMSNorm -> [H, n_pad, D] F16
  if (ec_launch_rope_nd(q_buf, k_buf, v_buf, coords_d, w->qk_norm_q, w->qk_norm_k,
                        ws->q_h, ws->k_h, ws->v_h, N, ws->n_pad, cfg->n_heads, cfg->head_dim,
                        cfg->ndims, cfg->axis_dims, cfg->rope_theta, cfg->eps, stream))
    return -1;

  // 4. Dual-Tier Attention (Exact FP16 FA2 Oracle vs Fast SageAttention INT8 Q/K)
  if (tier == EC_TIER_FAST) {
    if (ec_launch_sage_attn_i8(ws->q_h, ws->k_h, ws->v_h,
                               ws->q_i8, ws->k_i8, ws->q_sc, ws->k_sc,
                               ws->attn_f32, ws->attn_f16,
                               N, N, ws->n_pad, ws->n_pad, cfg->n_heads, cfg->head_dim,
                               attn_scale, cfg->block_causal, stream))
      return -1;
  } else {
    if (ec_launch_fa2_f16(ws->q_h, ws->k_h, ws->v_h,
                          ws->attn_f32, ws->attn_f16,
                          N, N, ws->n_pad, ws->n_pad, cfg->n_heads, cfg->head_dim,
                          attn_scale, cfg->block_causal, stream))
      return -1;
  }

  // 5. Attention Output Projection + Gated Residual
  if (ec_launch_gemm_f16(ws->attn_f16, w->w_proj, w->b_proj, nullptr,
                         ws->proj_f32, nullptr, N, E, E, w->b_proj ? 1 : 0, stream))
    return -1;
  if (ec_launch_gated_res(x_d, ws->proj_f32, w->gate_msa, N, E, stream)) return -1;

  // 6. Pre-MLP Fused adaLN-Zero -> norm_f16 [N, E]
  if (ec_launch_adaln_zero(x_d, nullptr, w->shift_mlp, w->scale_mlp,
                           ws->norm_f32, ws->norm_f16, N, E, cfg->eps, cfg->norm_mode, stream))
    return -1;

  // 7. FFN Up (fused bias+GELU -> F16) -> FFN Down -> Gated Residual
  if (ec_launch_gemm_f16(ws->norm_f16, w->w_fc1, w->b_fc1, nullptr,
                         ws->ffn_f32, ws->ffn_f16, N, F, E, 3, stream))
    return -1;
  if (ec_launch_gemm_f16(ws->ffn_f16, w->w_fc2, w->b_fc2, nullptr,
                         ws->proj_f32, nullptr, N, E, F, w->b_fc2 ? 1 : 0, stream))
    return -1;
  if (ec_launch_gated_res(x_d, ws->proj_f32, w->gate_mlp, N, E, stream)) return -1;

  return 0;
}

int ec_dit_capture_graph(
    const ec_dit_cfg_t *cfg, const ec_dit_block_w_t *blocks, int32_t n_blocks,
    ec_dit_ws_t *ws, float *x_d, const int32_t *coords_d, ec_tier_t tier, cudaStream_t stream) {
  if (ws->graph_exec) { cudaGraphExecDestroy(ws->graph_exec); ws->graph_exec = nullptr; }
  if (ws->graph)      { cudaGraphDestroy(ws->graph);          ws->graph = nullptr; }
  free(ws->cap_blocks); ws->cap_blocks = nullptr;
  ws->graph_valid = 0;

  cudaError_t cap_err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (cap_err == cudaErrorNotSupported) {
    ws->cap_cfg = *cfg;
    ws->cap_n_blocks = n_blocks;
    ws->cap_blocks = (ec_dit_block_w_t *)malloc((size_t)n_blocks * sizeof(ec_dit_block_w_t));
    if (!ws->cap_blocks) return -1;
    memcpy(ws->cap_blocks, blocks, (size_t)n_blocks * sizeof(ec_dit_block_w_t));
    ws->cap_x_d = x_d;
    ws->cap_coords_d = coords_d;
    ws->cap_tier = tier;
    ws->graph_valid = 2;
    return 0;
  }
  if (cap_err != cudaSuccess) return -1;

  for (int32_t b = 0; b < n_blocks; b++) {
    if (ec_dit_block_forward(cfg, &blocks[b], ws, x_d, coords_d, tier, stream) != 0) {
      cudaGraph_t tmp = nullptr;
      cudaStreamEndCapture(stream, &tmp);
      if (tmp) cudaGraphDestroy(tmp);
      return -1;
    }
  }
  EC_CK(cudaStreamEndCapture(stream, &ws->graph));
  EC_CK(cudaGraphInstantiate(&ws->graph_exec, ws->graph, nullptr, nullptr, 0));
  ws->graph_valid = 1;
  return 0;
}

int ec_dit_replay_graph(ec_dit_ws_t *ws, cudaStream_t stream) {
  if (!ws || !ws->graph_valid) return -1;
  if (ws->graph_valid == 2) {
    for (int32_t b = 0; b < ws->cap_n_blocks; b++) {
      if (ec_dit_block_forward(&ws->cap_cfg, &ws->cap_blocks[b], ws,
                               ws->cap_x_d, ws->cap_coords_d, ws->cap_tier, stream) != 0)
        return -1;
    }
    return 0;
  }
  if (!ws->graph_exec) return -1;
  return cudaGraphLaunch(ws->graph_exec, stream) == cudaSuccess ? 0 : -1;
}

int ec_magcache_should_skip(ec_magcache_t *mc, int32_t step, float rel_delta) {
  if (!mc) return 0;
  // Always compute warmup & final cooldown steps exactly for sharp high-freq details
  if (step < mc->warmup_steps || step >= mc->n_steps - mc->cooldown_steps) {
    mc->accum_err = 0.0f;
    mc->consec_skips = 0;
    return 0;
  }
  float step_err = rel_delta;
  if (mc->mag_ratios && step < mc->n_steps) {
    step_err = fabsf(1.0f - mc->mag_ratios[step]);
  }
  if (mc->consec_skips < mc->max_consec_skip && (mc->accum_err + step_err) <= mc->err_thresh) {
    mc->accum_err += step_err;
    mc->consec_skips++;
    mc->skipped_count++;
    return 1;
  }
  mc->accum_err = 0.0f;
  mc->consec_skips = 0;
  return 0;
}
