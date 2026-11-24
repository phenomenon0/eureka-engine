// ec_lora.h — Universal Native LoRA / QLoRA (NF4/FP8/INT8) / DoRA / LoHa / LoKr GPU Weight Fuser
// Accepts Civitai, HuggingFace, Kohya, Diffusers, LyCORIS, BitsAndBytes QLoRA & AI-Toolkit .safetensors.
// Supports stacking up to 8 adapters with arbitrary strength (-2.0..+2.0) and auto-maps
// BFL / Diffusers / Kohya / Wan 2.2 / ACE-Step 1.5 / TRELLIS.2 keys onto native Eureka GPU BF16 weights.

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
  char format_detected[64]; // "lora", "qlora_nf4", "qlora_fp8", "dora", "loha", "lokr"
} ec_lora_apply_stats_t;

// Apply a stack of LoRA/QLoRA/DoRA/LoHa/LoKr .safetensors files onto a named GPU BF16 weight matrix
// W_bf16_d [out_dim, in_dim]. Auto-maps FLUX.2, Wan 2.2, ACE-Step 1.5, and TRELLIS.2 keys and fuses
// split sub-LoRAs (e.g. q/k/v onto fused qkv) into their exact row slice!
int ec_lora_fuse_onto_bf16(
    uint16_t *W_bf16_d, int out_dim, int in_dim,
    const char *tensor_name,
    const ec_lora_spec_t *stack, int n_stack,
    ec_lora_apply_stats_t *stats,
    cudaStream_t stream);

// Inspect a .safetensors adapter file without applying it (reports tensor pairs, rank, arch hint).
int ec_lora_inspect(const char *path, int *out_pairs, int *out_rank, char *arch_hint, size_t arch_cap);

// Extended inspector also reporting adapter format ("lora"|"qlora_nf4"|"qlora_fp8"|"dora"|"loha"|"lokr").
int ec_lora_inspect_ex(
    const char *path, int *out_pairs, int *out_rank,
    char *arch_hint, size_t arch_cap,
    char *format_out, size_t format_cap);

// Returns 1 if any adapter in `stack[0..n_stack-1]` uses nonlinear DoRA row-magnitude rescaling
// (which requires a clean base weight reload), or 0 when all adapters are purely additive
// (standard LoRA, NF4 QLoRA, FP8 QLoRA, LoHa, LoKr — eligible for <8ms GPU delta unfusing!).
int ec_lora_stack_has_dora(const ec_lora_spec_t *stack, int n_stack);

#ifdef __cplusplus
}
#endif
