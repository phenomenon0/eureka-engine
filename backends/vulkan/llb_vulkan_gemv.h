// backends/vulkan/llb_vulkan_gemv.h — Native Cross-Vendor Vulkan 1.2/1.3+ Turbo LLM Backend
// Accelerates Qwen3.5 (0.8B..27B) on AMD Radeon, Intel Arc, and NVIDIA GPUs via:
//   - T1: Full GPU-resident single-submit forward graph (zero per-GEMV submits/syncs)
//   - T2: Batched prefill GEMM (BM=8) + subgroup & VK_KHR_shader_integer_dot_product (dp4a) GEMV
//   - T5: Disk-backed VkPipelineCache + dedicated VK_QUEUE_TRANSFER_BIT weight DMA

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t off;
  uint64_t bytes;
} llb_vk_tensor_span_t;

typedef struct {
  int L, D, H, KV, HDIM, dff, V, INTERVAL, KC, SZ, GK, TR, VHD, KDC, VDC, CONVD, ROPEDIM;
  float EPS, ROPE_BASE;
  int cap_pos;
} llb_vk_model_cfg_t;

typedef struct {
  uint64_t off;
  uint64_t bytes;
  int rows, cols, rb, fmt;
} llb_vk_tensor_desc_t;

typedef struct {
  int is_full;
  int attn_norm, post_attn_norm;
  int attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v, attn_output;
  int attn_qkv, attn_gate, ssm_beta, ssm_alpha, ssm_dt_bias, ssm_a, ssm_conv1d, ssm_norm, ssm_out;
  int ffn_gate, ffn_up, ffn_down;
} llb_vk_layer_ids_t;

typedef struct {
  int token_embd;
  int output_norm;
  int output_w;
  int n_layers;
  llb_vk_layer_ids_t layers[64];
} llb_vk_model_ids_t;

// Initialize Vulkan device and upload model weight tensors (from gPack) into <=1 GiB DEVICE_LOCAL slabs.
// Returns 0 if Vulkan GPU acceleration is active, -1 on fallback to CPU workers (never fatal).
int llb_vk_init(const uint8_t *pack_base, size_t pack_bytes,
                const llb_vk_tensor_span_t *spans, int n_spans,
                int max_rows, int max_cols);

// Fallback per-GEMV entry point (used when full GPU graph is unavailable or disabled).
int llb_vk_gemv(int fmt, const uint8_t *W, int rb, int rows, int cols, const float *x, float *y);

// Initialize full-graph GPU execution (T1/T2 single-submit decode + batched prefill).
// Returns 0 if active, -1 to fall back to per-GEMV path.
int llb_vk_graph_init(const llb_vk_model_cfg_t *cfg,
                      const llb_vk_tensor_desc_t *tensors, int n_tensors,
                      const llb_vk_model_ids_t *ids);

int llb_vk_graph_active(void);

// Reset recurrent state (gS, gConv) on GPU.
void llb_vk_graph_reset(void);

// Single-token full forward pass in ONE vkQueueSubmit.
// If compute_logits != 0, also executes output_norm + output.weight GEMV:
//   logits_out (optional): receives V float logits.
//   top2_out   (optional): receives 4 uint32s [i1, fbits(v1), i2, fbits(v2)] via GPU argmax (16B D2H!).
int llb_vk_graph_step(int tok, int pos, int compute_logits, float *logits_out, uint32_t *top2_out);

// Batched prefill for n_tok tokens starting at pos0 (chunks of up to 32 tokens per submit).
// If compute_last_logits != 0, computes final logits/top2 for the last token in the batch.
int llb_vk_graph_prefill(const int *toks, int n_tok, int pos0,
                         int compute_last_logits, float *logits_out, uint32_t *top2_out);

// Compute final logits/top2 from the current GPU hidden state X (used after prefill or CLEF).
int llb_vk_graph_final_logits(float *logits_out, uint32_t *top2_out);

// Copy normalized hidden state HH = rms_norm(X, output_norm) to host (for CLEF hidden dumps).
int llb_vk_graph_read_norm_hidden(float *hh_out);

// Human-readable device summary string (empty if not active).
const char *llb_vk_device_info(void);

// Query Vulkan device VRAM (free, total) in bytes. Returns 0 on success, -1 if unavailable.
int llb_vk_mem_info(size_t *free_b, size_t *total_b);

#ifdef __cplusplus
}
#endif
