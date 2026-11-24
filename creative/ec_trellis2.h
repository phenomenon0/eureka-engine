// ec_trellis2.h — 100% Pure Native Eureka TRELLIS.2 Image/Text-to-3D Generation Engine
// Zero Python, zero cuBLAS, zero ggml, zero external ML frameworks!
// Runs native CUDA DINOv3-ViT-L/16 vision encoder (512x512 RGB -> 1029x1024 features),
// 30-block SparseStructureFlowModel 1.3B BF16 DiT (3D RoPE, 12-step Flow Euler CFG interval),
// full 3D Conv SparseStructureDecoder (16^3x8 -> 64^3x1 continuous occupancy logits), and
// sub-voxel smooth Surface Nets mesh + 3DGS splat extraction -> GLB 2.0 + PLY!

#pragma once

#include "ec_lora.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_trellis2 ec_trellis2_t;

typedef struct {
  char dino_path[512];     // dinov3_vitl16.safetensors
  char ss_flow_path[512];  // trellis2_ss_flow_1.3b_bf16.safetensors
  char ss_dec_path[512];   // trellis2_ss_dec_16l8_fp16.safetensors
} ec_trellis2_paths_t;

typedef struct {
  const uint8_t *rgb;        // primary conditioning RGB image (H * W * 3), required!
  int            img_w;
  int            img_h;
  // v8 Multi-View Conditioning (up to 3 extra RGB views fused cleanly in DINOv3 patch feature space!)
  const uint8_t *extra_rgbs[3];
  int            extra_w[3];
  int            extra_h[3];
  int            n_extra_views;
  uint32_t       seed;
  int            steps;      // default 12
  float          cfg_scale;  // default 7.5
  float                 rescale_t;     // default 5.0
  float                 base_color[4]; // optional PBR tint (if alpha <= 0, auto-sampled from conditioning image!)
  const ec_lora_spec_t *loras;
  int                   n_loras;
} ec_trellis2_req_t;

typedef struct {
  int    ok;
  int    n_active_voxels;
  int    n_verts;
  int    n_tris;
  int    n_splats;
  int    steps_run;
  int    lora_tensors_fused;
  double dino_ms;
  double flow_ms;
  double dec_ms;
  double mesh_ms;
  double total_ms;
  char   error[256];
} ec_trellis2_res_t;

typedef int (*ec_trellis2_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_trellis2_t *ec_trellis2_create(const ec_trellis2_paths_t *paths);
void           ec_trellis2_destroy(ec_trellis2_t *tr);
int            ec_trellis2_is_ready(const ec_trellis2_t *tr);

// Releases resident GPU weights (keeps lightweight mmap open) so RTX 3090 VRAM shares cleanly.
void ec_trellis2_unload_vram(ec_trellis2_t *tr);

// Generates a genuine 3D binary glTF 2.0 (.glb) mesh and optional 3DGS (.ply) splat cloud
// from a conditioning RGB image (e.g. from FLUX.2-klein-4B or user upload).
int ec_trellis2_generate_3d(
    ec_trellis2_t *tr,
    const ec_trellis2_req_t *req,
    const char *out_glb_path,
    const char *out_ply_path,
    ec_trellis2_progress_fn cb, void *cb_ctx,
    ec_trellis2_res_t *res);

#ifdef __cplusplus
}
#endif
