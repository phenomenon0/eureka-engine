// ec_studio.h — Pure C11 Backend-Agnostic Interface to Eureka Native Creative Studio
// Zero CUDA/Metal headers exposed to HTTP server or Tauri host callers!

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
  char   gpu_name[128];
  int    sm_major, sm_minor;
  double vram_total_gb;
  double vram_pool_used_mb;
  double vram_pool_budget_mb;
} ec_studio_info_t;

typedef struct {
  char   file_main[256];
  char   file_aux[256];
  char   stem_files[4][256];
  int    width, height, frames;
  int    n_verts, n_tris, n_splats;
  int    steps_run, steps_skipped;
  double gpu_ms;
  double fps;
  int    dag_executed, dag_cache_hits, dag_evictions;
} ec_studio_result_t;

ec_studio_t *ec_studio_create(const char *artifact_dir, size_t vram_budget_mb);
void         ec_studio_destroy(ec_studio_t *st);
void         ec_studio_get_info(const ec_studio_t *st, ec_studio_info_t *out);

int ec_studio_gen_image(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, ec_studio_result_t *res);

int ec_studio_gen_3d(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, ec_studio_result_t *res);

int ec_studio_gen_music(
    ec_studio_t *st, const char *prompt, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, ec_studio_result_t *res);

int ec_studio_gen_video(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    ec_tier_t tier, ec_studio_result_t *res);

int ec_studio_realtime_burst(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames,
    const char *shm_name, ec_studio_result_t *res);

int ec_studio_run_flow_json(
    ec_studio_t *st, const char *comfy_json, ec_studio_result_t *res, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
