// ec_flux2.h — 100% Pure Native Eureka Photoreal FLUX.2-klein-4B Pipeline
// Zero Python, zero cuBLAS, zero ggml, zero stable-diffusion.cpp!
// Runs native Qwen3-4B text encoder (layers 0..26 -> [512, 7680]),
// native FLUX.2-klein-4B MMDiT (5 double + 20 single blocks, 4-step shifted RF, BF16 PTX MMA,
// dynamic finetune checkpoints & stacked LoRAs), and full AutoencoderKLFlux2 2D VAE -> RGB PNG!

#pragma once

#include "ec_lora.h"
#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_flux2 ec_flux2_t;

typedef struct {
  char checkpoint_path[512];   // e.g. flux2_klein_4b_bf16.safetensors or custom finetune!
  char vae_path[512];          // e.g. flux2_klein_vae.safetensors
  char text_encoder_path[512]; // e.g. qwen_3_4b.safetensors (can live on /mnt/storage or network drive!)
  char tokenizer_path[512];    // e.g. qwen3_tokenizer.json
} ec_flux2_paths_t;

typedef struct {
  const char           *prompt;
  uint32_t              seed;
  int                   width;         // multiple of 16 (e.g. 512, 768, 1024)
  int                   height;        // multiple of 16 (e.g. 512, 768, 1024)
  int                   steps;         // 4 default for FLUX.2-klein-4B distilled (1..50)
  const char           *checkpoint_override; // optional finetune checkpoint path
  const ec_lora_spec_t *loras;
  int                   n_loras;
} ec_flux2_req_t;

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
} ec_flux2_res_t;

typedef int (*ec_flux2_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_flux2_t *ec_flux2_create(const ec_flux2_paths_t *paths);
void        ec_flux2_destroy(ec_flux2_t *fx);
int         ec_flux2_is_ready(const ec_flux2_t *fx);

// Releases resident GPU weights (keeps lightweight mmap & BPE open) so VRAM shares cleanly across engines.
void ec_flux2_unload_vram(ec_flux2_t *fx);

// Generates a real photorealistic RGB image into out_png_path (RFC 2083 PNG).
int ec_flux2_generate_png(
    ec_flux2_t *fx,
    const ec_flux2_req_t *req,
    const char *out_png_path,
    ec_flux2_progress_fn cb, void *cb_ctx,
    ec_flux2_res_t *res);

// Generates photoreal RGB24 buffer (out_rgb allocated via malloc, caller frees) and optionally writes out_png_path.
int ec_flux2_generate_rgb(
    ec_flux2_t *fx,
    const ec_flux2_req_t *req,
    uint8_t **out_rgb,
    const char *out_png_path,
    ec_flux2_progress_fn cb, void *cb_ctx,
    ec_flux2_res_t *res);

#ifdef __cplusplus
}
#endif
