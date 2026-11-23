// ec_studio.h — Pure C11 Backend-Agnostic Interface to Eureka Native Creative Studio (v2)
// Zero CUDA/Metal headers exposed to HTTP server, supervisor, or Tauri host callers!

#pragma once

#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_studio ec_studio_t;

typedef struct {
  char   artifact_dir[256];
  char   models_dir[256];
  char   gpu_name[128];
  int    sm_major, sm_minor;
  double vram_total_gb;
  double vram_pool_used_mb;
  double vram_pool_budget_mb;
  int    models_discovered;
} ec_studio_info_t;

typedef struct {
  char   file_main[256];
  char   file_aux[256];       // .ply for 3D, keyframe0 .png for video
  char   file_mp4[256];       // hardware NVENC H.264 (+AAC) .mp4 for video / video_av
  char   encoder_used[32];    // "h264_nvenc" or "libx264"
  char   stem_files[4][256];  // music: drums, bass, harmony, lead
  int    width, height, frames;
  int    n_verts, n_tris, n_splats;
  int    steps_run, steps_skipped;
  int    cancelled;
  double gpu_ms;
  double fps;
  int    dag_executed, dag_cache_hits, dag_evictions;
} ec_studio_result_t;

// Live progress callback for SSE streaming & cooperative cancellation (return non-zero to cancel)
typedef int (*ec_progress_cb_t)(int step, int total_steps, int skipped, const char *stage, void *ctx);

ec_studio_t *ec_studio_create(const char *artifact_dir, const char *models_dir, size_t vram_budget_mb);
void         ec_studio_destroy(ec_studio_t *st);
void         ec_studio_get_info(const ec_studio_t *st, ec_studio_info_t *out);

// 1-6. All Creative Modality Pipelines (thread-safe, progress + cancel aware)
int ec_studio_gen_image(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_3d(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_gen_music(
    ec_studio_t *st, const char *prompt, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

// with_audio=1 runs LTX-2.5 style synced stereo audio + NVENC H.264/AAC MP4 mux!
int ec_studio_gen_video(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    int fps, int with_audio, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_realtime_burst(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames,
    const char *shm_name, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res);

int ec_studio_run_flow_json(
    ec_studio_t *st, const char *comfy_json, ec_studio_result_t *res, char *err, size_t err_cap);

// 7. Live Multi-Architecture Safetensors Model Registry API
int ec_studio_models_scan(ec_studio_t *st);
int ec_studio_models_json(ec_studio_t *st, char *buf, size_t cap);
int ec_studio_model_load(ec_studio_t *st, const char *id, double *probe_ms, char *route, size_t rcap, char *err, size_t ecap);
int ec_studio_model_unload(ec_studio_t *st, const char *id);

#ifdef __cplusplus
}
#endif
