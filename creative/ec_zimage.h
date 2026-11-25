// ec_zimage.h — 100% Pure Native CUDA Tongyi-MAI Z-Image-Turbo (6.15B S3-DiT, Apache 2.0)
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Reuses Eureka's existing Qwen3-4B (penultimate hidden_states[-2] = layer 34, dim=2560),
// ec_ops_gemm_bf16 + ec_ops_fa2_bf16 (30 heads x D=128 = 3840), hot-fused LoRAs, and FLUX.1 16ch VAE!

#pragma once

#include "ec_lora.h"
#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_zimage ec_zimage_t;

typedef struct {
  char dit_path[512];          // z_image_turbo_bf16.safetensors (12.31 GB BF16)
  char vae_path[512];          // flux1_ae.safetensors (335 MB F32)
  char text_encoder_path[512]; // qwen_3_4b.safetensors (shared with FLUX.2!)
  char tokenizer_path[512];    // qwen3_tokenizer.json
} ec_zimage_paths_t;

typedef struct {
  const char           *prompt;
  uint32_t              seed;
  int                   width;         // multiple of 16 (default 1024)
  int                   height;        // multiple of 16 (default 1024)
  int                   steps;         // 8 default (4..20 for Turbo; 28..50 for Base)
  float                 shift;         // 3.0 default (3.0..7.0 AuraFlow shift)
  const ec_lora_spec_t *loras;
  int                   n_loras;
  int                   tile_upscale;  // 1 -> 1.5x Lanczos3 Pixel-Space HiRes Detail Refine
} ec_zimage_req_t;

typedef struct {
  int    ok;
  int    width;
  int    height;
  int    steps_run;
  int    lora_tensors_fused;
  double text_enc_ms;
  double dit_ms;
  double vae_ms;
  double total_ms;
  char   checkpoint_used[256];
  char   error[256];
} ec_zimage_res_t;

typedef int (*ec_zimage_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_zimage_t *ec_zimage_create(const ec_zimage_paths_t *paths);
void         ec_zimage_destroy(ec_zimage_t *zi);
int          ec_zimage_is_ready(const ec_zimage_t *zi);
void         ec_zimage_unload_vram(ec_zimage_t *zi);

int ec_zimage_generate_rgb(
    ec_zimage_t *zi,
    const ec_zimage_req_t *req,
    uint8_t **out_rgb,
    const char *out_png_path,
    ec_zimage_progress_fn cb, void *cb_ctx,
    ec_zimage_res_t *res);

int ec_zimage_generate_png(
    ec_zimage_t *zi,
    const ec_zimage_req_t *req,
    const char *out_png_path,
    ec_zimage_progress_fn cb, void *cb_ctx,
    ec_zimage_res_t *res);

#ifdef __cplusplus
}
#endif
