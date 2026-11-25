// ec_idol.h — 100% Pure Native Eureka CUDA Engine for speridlabs/idol (Apache-2.0)
// Single Image -> 262,144 to 1,048,576 3D Gaussian Splats (.ply) + Preview .glb!
// Zero Python, zero cuBLAS, zero ggml, zero torch.compile cold boot!
// Pipeline:
//   1. Coons-patch / black-alpha 1024x1024 foreground crop + black composite
//   2. Dual Image Conditioners: DINOv3-ViT-H+/16 (32L, 1280d, SwiGLU, 2D RoPE) + AutoencoderKLFlux2 (128ch BN-packed)
//   3. LatentSeqMMFlowModel (24 joint + 2+2 refiner blocks, RePo 3D RoPE, Rectified-Flow Euler CFG)
//   4. OctreeProbabilityFixedlenDecoder (4L cross-attn, 8-level octree stratified systematic sampling)
//   5. GaussianFixedlenDecoder (16L self+cross-attn, 32 Gaussians/anchor -> Y-up standard 3DGS .ply)

#pragma once

#include "ec_flux2.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ec_idol ec_idol_t;

typedef struct {
  char dinov3_path[512];     // dinov3_vith16plus.safetensors
  char denoiser_path[512];   // idol/denoiser.safetensors
  char decoder_path[512];    // idol/decoder.safetensors
  char decoder_gs_path[512]; // idol/decoder_gs.safetensors
} ec_idol_paths_t;

typedef struct {
  const uint8_t *rgb;        // Input RGB24 image [img_h * img_w * 3]
  const uint8_t *alpha;      // Optional alpha channel [img_h * img_w] (NULL if RGB-only)
  int            img_w;
  int            img_h;
  uint32_t       seed;
  int            steps;          // Default 32 (20..50)
  float          cfg_scale;      // Default 3.0
  float          rescale_t;      // Default 3.0
  float          crop_margin;    // Default 1.2 (synthetic) or 1.5 (real photo)
  int            num_gaussians;  // 32768..1048576 (multiple of 32; default 262144)
  int            skip_matte;     // 1 if image is already cleanly masked on black
} ec_idol_req_t;

typedef struct {
  int    ok;
  int    n_splats;
  int    n_anchors;
  int    steps_run;
  float  cam_yaw_rad, cam_pitch_rad, cam_radius, cam_fov_deg;
  double enc_ms;
  double flow_ms;
  double dec_ms;
  double total_ms;
  char   error[256];
} ec_idol_res_t;

typedef int (*ec_idol_progress_fn)(int step, int total_steps, const char *stage, void *ctx);

ec_idol_t *ec_idol_create(const ec_idol_paths_t *paths);
void       ec_idol_destroy(ec_idol_t *id);
int        ec_idol_is_ready(const ec_idol_t *id);
void       ec_idol_unload_vram(ec_idol_t *id);

int ec_idol_generate_splats(
    ec_idol_t *id,
    ec_flux2_t *flux2_vae_host, // Reuses native AutoencoderKLFlux2 encoder!
    const ec_idol_req_t *req,
    const char *out_ply_path,
    const char *out_glb_preview_path, // Optional proxy GLB for legacy viewers
    ec_idol_progress_fn cb, void *cb_ctx,
    ec_idol_res_t *res);

#ifdef __cplusplus
}
#endif
