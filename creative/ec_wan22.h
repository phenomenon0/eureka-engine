// ec_wan22.h — 100% Pure Native Eureka Wan 2.2 TI2V-5B Video Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Runs native C11 Unigram tokenizer (umt5_tokenizer.json), native CUDA UMT5-XXL FP8-scaled
// 24-layer text encoder, full 30-block WanModel 3D RoPE DiT (dim=3072, H=24, D=128, shift=5.0,
// CFG=5.0), and full Causal 3D Wan2_2_VAE streaming decoder -> hardware h264_nvenc MP4!

#pragma once

#include "ec_lora.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_wan22 ec_wan22_t;

typedef struct {
  char dit_path[512];          // wan2.2_ti2v_5b_fp16.safetensors
  char vae_path[512];          // wan2.2_vae.safetensors
  char text_encoder_path[512]; // umt5_xxl_fp8_e4m3fn_scaled.safetensors
  char tokenizer_path[512];    // umt5_tokenizer.json
} ec_wan22_paths_t;

typedef struct {
  const char           *prompt;
  const char           *neg_prompt;    // optional (NULL -> default clean negative prompt)
  uint32_t              seed;
  int                   width;         // multiple of 32, min 320 (default 832 for native 480p)
  int                   height;        // multiple of 32, min 320 (default 480 for native 480p)
  int                   num_frames;    // 4*k + 1 (e.g. 5, 9, 17, 21, 25, 33; default 17)
  int                   fps;           // default 16
  int                   steps;         // default 20 (4..50)
  float                 cfg_scale;     // default 5.0
  float                 shift;         // 0 -> auto (3.0 for <=480p, 5.0 for 720p)
  // v8 Native Start-Frame I2V, Start+End FLF2V, Seamless Loop, Camera Motion & LoRA/QLoRA Stack
  const uint8_t        *start_rgb;     // optional RGB24 start frame [start_h * start_w * 3]
  int                   start_w;
  int                   start_h;
  const uint8_t        *end_rgb;       // optional RGB24 end frame [end_h * end_w * 3] (FLF2V!)
  int                   end_w;
  int                   end_h;
  float                 denoise;       // 0.1..1.0 (default 1.0)
  const char           *camera_motion; // optional: "orbit"|"dolly_in"|"pan_left"|"pan_right"|"crane_up"|"fpv"|"static"
  const ec_lora_spec_t *loras;
  int                   n_loras;
} ec_wan22_req_t;

typedef struct {
  int    ok;
  int    width;
  int    height;
  int    num_frames;
  int    latent_frames;
  int    steps_run;
  int    lora_tensors_fused;
  char   encoder_used[32];
  double text_enc_ms;
  double dit_ms;
  double vae_ms;
  double encode_ms;
  double total_ms;
  char   error[256];
} ec_wan22_res_t;

typedef int (*ec_wan22_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_wan22_t *ec_wan22_create(const ec_wan22_paths_t *paths);
void        ec_wan22_destroy(ec_wan22_t *w);
int         ec_wan22_is_ready(const ec_wan22_t *w);

// Releases resident 9.4 GB DiT + VAE weights from VRAM (keeps mmap & tokenizer open).
void ec_wan22_unload_vram(ec_wan22_t *w);

// Generates a real H.264 NVENC MP4 video (+ optional keyframe0 PNG and optional audio WAV mux).
int ec_wan22_generate_mp4(
    ec_wan22_t *w,
    const ec_wan22_req_t *req,
    const char *out_mp4_path,
    const char *out_keyframe_png, // optional
    const char *audio_wav_path,   // optional synced audio track to mux into MP4
    ec_wan22_progress_fn cb, void *cb_ctx,
    ec_wan22_res_t *res);

// Extended generator also exporting the final decoded RGB frame PNG (`out_lastframe_png`) for 1-click Storyboard Chaining!
int ec_wan22_generate_mp4_ex(
    ec_wan22_t *w,
    const ec_wan22_req_t *req,
    const char *out_mp4_path,
    const char *out_keyframe_png,  // optional
    const char *out_lastframe_png, // optional final frame PNG for 1-click Extend
    const char *audio_wav_path,    // optional synced audio track to mux into MP4
    ec_wan22_progress_fn cb, void *cb_ctx,
    ec_wan22_res_t *res);

// Native Wan2.2 Causal 3D VAE single-frame RGB -> normalized [H/16, W/16, 48] latent helper (for G-C7!)
int ec_wan22_vae_encode_image(
    ec_wan22_t *w, const uint8_t *rgb, int width, int height,
    float **out_latent_h, int *out_h_lat, int *out_w_lat, double *out_ms);

#ifdef __cplusplus
}
#endif
