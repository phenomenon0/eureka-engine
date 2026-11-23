// ec_lora.h — Universal Native LoRA Adapter Loader & GPU Weight Fuser
// Accepts Civitai, HuggingFace, Kohya, Diffusers & AI-Toolkit .safetensors LoRAs.
// Supports stacking multiple LoRAs with arbitrary strength (-2.0..+2.0) and auto-maps
// Diffusers/ComfyUI/BFL key prefixes onto native Eureka GPU BF16/FP16 weights.

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_LORA_MAX_STACK 8

typedef struct {
  char  path[512];
  float strength;
} ec_lora_spec_t;

typedef struct {
  int  n_applied;
  int  n_tensors_matched;
  int  n_tensors_total;
  int  rank_first;
  char format_detected[64];
} ec_lora_apply_stats_t;

// Apply a stack of LoRA .safetensors files onto a named GPU BF16 weight matrix W_d [out_dim, in_dim].
// Matches BFL keys ("double_blocks.0.img_attn.qkv"), Diffusers keys ("transformer_blocks.0.attn.to_q/k/v"),
// and Kohya/ComfyUI prefixes ("lora_unet_...", "diffusion_model.", "transformer.").
// For fused projections (e.g. qkv or single_blocks.linear1), automatically fuses split sub-LoRAs
// into their exact row slice [row_offset .. row_offset + sub_out)!
int ec_lora_fuse_onto_bf16(
    uint16_t *W_bf16_d, int out_dim, int in_dim,
    const char *bfl_tensor_name,
    const ec_lora_spec_t *stack, int n_stack,
    ec_lora_apply_stats_t *stats,
    cudaStream_t stream);

// Inspect a .safetensors LoRA file without applying it (reports tensor count, rank, arch hint).
int ec_lora_inspect(const char *path, int *out_pairs, int *out_rank, char *arch_hint, size_t arch_cap);

#ifdef __cplusplus
}
#endif
