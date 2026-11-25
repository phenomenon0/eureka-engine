// ec_studio.h — Pure C11 Backend-Agnostic Interface to Eureka Native Creative Studio (v3)
// Zero CUDA/Metal headers exposed to HTTP server, supervisor, or Tauri host callers!
// Includes: Real Photoreal FLUX.2-klein-4B + LoRA Stack + Custom Finetunes, Multi-Drive &
// Network-Drive Storage Vault, Curated Photoreal Prompt Library & Recipes, and ComfyUI Importer.

#pragma once

#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_studio ec_studio_t;

#define EC_STUDIO_MAX_LORAS 8

typedef struct {
  char  path[512];
  float strength;
} ec_studio_lora_t;

typedef struct {
  char   artifact_dir[256];
  char   models_dir[256];
  char   backend[16]; // "cuda", "metal", or "vulkan"
  char   gpu_name[128];
  int    sm_major, sm_minor;
  double vram_total_gb;
  double vram_free_mb;
  double vram_pool_used_mb;
  double vram_pool_budget_mb;
  int    engines_resident_mask; // bit0=flux2, bit1=acestep, bit2=trellis2, bit3=wan22
  int    models_discovered;
  int    photoreal_ready;
  int    acestep_ready;
  int    trellis2_ready;
  int    wan22_ready;
  int    storage_roots;
  int    loras_discovered;
  int    checkpoints_discovered;
  int    recipes_count;
  int    cowrie_enabled;
  int    cowrie_shani;
  uint64_t cowrie_sp_hits;
  uint64_t cowrie_sp_saves;
  uint64_t cowrie_sp_mismatches;
  int    cowrie_trust_entries;
} ec_studio_info_t;

typedef struct {
  char   file_main[256];
  char   file_aux[256];       // .ply for 3D, keyframe0 .png for video
  char   file_last[256];      // final frame .png for 1-click Video Storyboard Extend!
  char   file_mp4[256];       // hardware NVENC H.264 (+AAC) .mp4 for video / video_av
  char   encoder_used[32];    // "h264_nvenc" or "libx264"
  char   stem_files[8][256];  // music: up to 8 flexible stems (0=full, 2=vocals/inst, 4=4stem, 8=8stem)
  char   stem_names[8][24];   // stem labels corresponding to stem_files[0..n_stems-1]
  int    n_stems;
  char   engine_mode[32];     // "flux2_photoreal", "acestep_1.5_turbo", "trellis2_1.3b", "wan2.2_ti2v_5b", or "dit_smoke"
  char   checkpoint_used[256];
  int    width, height, frames;
  int    sample_rate;         // 48000 for ACE-Step 1.5, 44100 for smoke
  int    n_verts, n_tris, n_splats;
  int    steps_run, steps_skipped;
  int    lora_tensors_fused;
  int    cancelled;
  double gpu_ms;
  double text_enc_ms, dit_ms, vae_ms;
  double fps;
  int    dag_executed, dag_cache_hits, dag_evictions;
} ec_studio_result_t;

// v8 Multimodal Request Structs (all optional media fields resolve either absolute paths or /artifacts/<name>!)
typedef struct {
  const char             *prompt;
  const char             *neg_prompt;
  uint32_t                seed;
  int                     width, height, steps;
  float                   cfg_scale;
  ec_tier_t               tier;
  int                     photoreal;
  const char             *checkpoint_query;
  const ec_studio_lora_t *loras;
  int                     n_loras;
  const char             *init_image_path;   // Img2Img / Inpaint / Outpaint input image
  const char             *mask_image_path;   // Inpaint mask image (white=repaint, black=keep)
  float                   denoise;           // 0.05..1.0
  int                     outpaint_pad[4];   // left, top, right, bottom pixels
  int                     tile_upscale;      // 1 -> 2x overlapping tiled latent detail upscale
} ec_studio_image_v8_t;

typedef struct {
  const char             *prompt;
  const char             *neg_prompt;
  uint32_t                seed;
  int                     n_frames, width, height, fps, steps, with_audio, full_model;
  float                   cfg_scale, shift, denoise;
  ec_tier_t               tier;
  const char             *start_frame_path;  // I2V start frame
  const char             *end_frame_path;    // FLF2V end frame (set == start_frame for Seamless Loop!)
  const char             *camera_motion;     // auto|orbit|dolly_in|pan_left|pan_right|crane_up|fpv|static
  const ec_studio_lora_t *loras;
  int                     n_loras;
} ec_studio_video_v8_t;

typedef struct {
  const char *prompt;
  uint32_t    seed;
  int         grid_res, steps, full_model, isolate_fg;
  float       cfg_scale, rescale_t;
  float       base_color[4];
  ec_tier_t   tier;
  const char             *input_images[4]; // 1..4 direct photo uploads -> DINOv3 multi-view (skips FLUX.2!)
  int                     n_input_images;
  const ec_studio_lora_t *loras;
  int                     n_loras;
} ec_studio_3d_v8_t;

typedef struct {
  const char             *prompt;
  const char             *lyrics;
  uint32_t                seed;
  int                     bpm, steps, stem_mode, full_model; // stem_mode: 0=full, 2, 4, 8
  float                   duration_sec, shift, denoise, repaint_start_sec, repaint_end_sec;
  ec_tier_t               tier;
  const char             *key_scale;
  const char             *time_sig;
  const char             *vocal_mode;
  const char             *audio_mode;        // generate|cover|repaint|extend
  const char             *source_audio_path; // WAV/MP3/FLAC upload for Cover/Repaint/Extend
  const ec_studio_lora_t *loras;
  int                     n_loras;
} ec_studio_music_v8_t;

// Live progress callback for SSE streaming & cooperative cancellation (return non-zero to cancel)
typedef int (*ec_progress_cb_t)(int step, int total_steps, int skipped, const char *stage, void *ctx);

ec_studio_t *ec_studio_create(const char *artifact_dir, const char *models_dir, size_t vram_budget_mb);
void         ec_studio_destroy(ec_studio_t *st);
void         ec_studio_get_info(const ec_studio_t *st, ec_studio_info_t *out);

// Evicts all resident GPU weights across FLUX.2, ACE-Step, TRELLIS.2, and Wan 2.2 in <5ms
// (preserving mmap handles & tokenizers) for Cross-Process GPU VRAM Lease coordination with llb-serve!
int ec_studio_unload_all_vram(ec_studio_t *st, double *out_ms, double *out_free_mb);

// 1-6. All Creative Modality Pipelines (thread-safe, progress + cancel aware)
int ec_studio_gen_image(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

// Extended photoreal image generation accepting custom finetune checkpoint + stacked LoRAs!
int ec_studio_gen_image_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, int photoreal,
    const char *checkpoint_query,
    const ec_studio_lora_t *loras, int n_loras,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_3d(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_3d_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, int full_model, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_music(
    ec_studio_t *st, const char *prompt, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_music_ex(
    ec_studio_t *st, const char *prompt, const char *lyrics, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, int full_model, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

// with_audio=1 runs synced stereo audio (ACE-Step 1.5 in full_model) + NVENC H.264/AAC MP4 mux!
int ec_studio_gen_video(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    int fps, int with_audio, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_video_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    int fps, int steps, int with_audio, ec_tier_t tier, int full_model,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

// v8 Native Multimodal Entrypoints (Img2Img/Inpaint/Outpaint/Upscale, I2V/FLF2V/Loop, Photo-to-3D, 8-Stem/Repaint/Cover/Extend)
int ec_studio_gen_image_v8(ec_studio_t *st, const ec_studio_image_v8_t *req, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);
int ec_studio_gen_video_v8(ec_studio_t *st, const ec_studio_video_v8_t *req, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);
int ec_studio_gen_3d_v8(ec_studio_t *st, const ec_studio_3d_v8_t *req, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);
int ec_studio_gen_music_v8(ec_studio_t *st, const ec_studio_music_v8_t *req, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);
int ec_studio_lora_catalog_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_upload_media(ec_studio_t *st, const uint8_t *bytes, size_t len, const char *ext, char *out_filename, size_t cap);

int ec_studio_realtime_burst(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames,
    const char *shm_name, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

// Live per-node ComfyUI DAG telemetry callback (status: "cached" | "start" | "progress" | "done")
typedef int (*ec_flow_ui_event_cb_t)(
    const char *node_id, const char *class_type, const char *status,
    int step, int total_steps, int cache_hits, int executed,
    double node_ms, const char *artifact_url, void *ctx);

int ec_studio_run_flow_json(
    ec_studio_t *st, const char *comfy_json, ec_studio_result_t *res, char *err, size_t err_cap);
int ec_studio_run_flow_ex(
    ec_studio_t *st, const char *comfy_json,
    ec_flow_ui_event_cb_t event_cb, void *cb_ctx,
    ec_studio_result_t *res, char *node_outputs_json, size_t node_outputs_cap,
    char *err, size_t err_cap);
int ec_studio_flow_object_info_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_flow_templates_json(ec_studio_t *st, char *buf, size_t cap);
void ec_studio_flow_clear_cache(ec_studio_t *st);
int ec_studio_flow_upload_image(
    ec_studio_t *st, const uint8_t *bytes, size_t len, const char *ext,
    char *out_filename, size_t cap);

// 7. Live Multi-Architecture Safetensors Model Registry API
int ec_studio_models_scan(ec_studio_t *st);
int ec_studio_models_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_model_load(ec_studio_t *st, const char *id, double *probe_ms, char *route, size_t rcap, char *err, size_t ecap);
int ec_studio_model_unload(ec_studio_t *st, const char *id);

// 8. Multi-Drive & Network-Drive Storage Vault, LoRAs & Finetune Checkpoints
int ec_studio_storage_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_storage_add_root(ec_studio_t *st, const char *path, const char *label);
int ec_studio_storage_remove_root(ec_studio_t *st, const char *path);
int ec_studio_storage_rescan(ec_studio_t *st);
int ec_studio_loras_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_checkpoints_json(ec_studio_t *st, char *buf, size_t cap);

// 9. Photoreal Prompt Library, User Recipes & Universal ComfyUI Importer (JSON + PNG!)
int ec_studio_recipes_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_prompts_json(char *buf, size_t cap);
int ec_studio_recipe_save(
    ec_studio_t *st, const char *id, const char *name, const char *category,
    const char *modality, const char *prompt, const char *camera,
    const char *checkpoint, const char *lora, float lora_strength,
    int width, int height, int steps, uint32_t seed);
int ec_studio_recipe_delete(ec_studio_t *st, const char *id);

// Import ComfyUI API JSON, LiteGraph UI JSON, OR raw PNG bytes (if is_png=1), resolve models
// across all local/network storage roots, frame under Cowrie v2 Trust Membrane, and optionally execute.
int ec_studio_import_comfy(
    ec_studio_t *st, const uint8_t *payload, size_t payload_len, int is_png, int execute_now,
    char *out_json, size_t out_cap);
int ec_studio_import_comfy_ex(
    ec_studio_t *st, const uint8_t *payload, size_t payload_len, int is_png, int execute_now,
    const char *source_tag, char *out_json, size_t out_cap);

// 10. Tessera / Cowrie v2 Director LLM Lenses (R0/R1/R2) & Trust Membrane Gate
int ec_studio_cowrie_lens_json(
    ec_studio_t *st, const char *target, int tier, char *out_json, size_t out_cap);
int ec_studio_cowrie_trust_gate_json(
    ec_studio_t *st, const char *action, const char *const *arg_ids, int n_args,
    int owner_override, char *out_json, size_t out_cap);
int ec_studio_cowrie_trust_triage_json(ec_studio_t *st, char *out_json, size_t out_cap);

#ifdef __cplusplus
}
#endif
