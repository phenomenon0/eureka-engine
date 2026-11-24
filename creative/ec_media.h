// ec_media.h — Native Zero-Dependency Creative Media Exporters & Zero-Copy SHM Realtime Video Ring
// Writes real GLB 2.0 (3D PBR mesh), 3DGS .ply (splats), 16-bit Stereo WAV (music), PPM (images),
// and streams 60+ FPS RGBA frames via lock-free triple-buffered POSIX shared memory.

#pragma once

#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1. Native GLB 2.0 (glTF Binary) 3D PBR Mesh Exporter
int ec_export_glb(const char *path, const ec_mesh_t *mesh);

// 2. Native 3D Gaussian Splat (.ply) Exporter
int ec_export_splat_ply(const char *path, const ec_splat3d_t *splat);

// 3. Native 16-bit PCM Stereo/Mono WAV Exporter, Loader & Multi-Stem Splitter (0/2/4/8 Stems)
int ec_export_wav(const char *path, const ec_audio_t *audio);
int ec_load_wav_stereo_f32(const char *path, int target_sr, float **out_pcm_lr, int *out_samples);
// Splits stereo PCM into 0 (full mix), 2 (vocals/instrumental), 4 (drums/bass/harmony/lead),
// or 8 (drums/bass/vocals/guitar/piano/synth/strings/fx) complementary stems summing to master!
int ec_audio_split_stems(
    const float *pcm_lr, int n_samples, int sr, int stem_mode,
    const char *dir_prefix, char out_paths[8][256], char out_names[8][24], int *out_n_stems);

// 4. Native Image Exporters, Universal Loaders (PNG/JPG/PPM), Resizer, Mask & Outpaint Pad
int ec_export_ppm(const char *path, const uint8_t *rgb, uint32_t width, uint32_t height);
int ec_export_png(const char *path, const uint8_t *rgb, uint32_t width, uint32_t height);
int ec_load_image_rgb(const char *path, uint8_t **out_rgb, int *out_w, int *out_h);
uint8_t *ec_rgb_resize_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh);
int ec_load_mask_gray(const char *path, int target_w, int target_h, float **out_mask);
int ec_rgb_pad_outpaint(
    const uint8_t *src, int sw, int sh,
    int pad_left, int pad_top, int pad_right, int pad_bottom,
    int target_w, int target_h,
    uint8_t **out_rgb, float **out_mask);
void ec_rgb_isolate_foreground(uint8_t *rgb, int w, int h);
int ec_png_inject_comfy_metadata(const char *src_png, const char *dst_png, const char *prompt_json, const char *workflow_json);
int ec_png_postprocess_grade(
    const char *src_png, const char *dst_png,
    float contrast, float warmth, float saturation, float vignette,
    float sharpen, int grayscale, int invert);

// 4b. Native O-Voxel Iso-Surface Mesh & Gaussian Splat Builders (from GPU sparse coords/feats)
int ec_ovoxel_build_mesh(
    const int32_t *coords_h, const float *feats_h, int n_active, int channels, int grid_res,
    ec_mesh_t *out_mesh);
void ec_mesh_free(ec_mesh_t *m);

int ec_ovoxel_build_splats(
    const int32_t *coords_h, const float *feats_h, int n_active, int channels, int grid_res,
    ec_splat3d_t *out_splat);
void ec_splat_free(ec_splat3d_t *s);

// 5. Lock-Free Triple-Buffered POSIX SHM Zero-Copy Realtime Video Ring (60+ FPS to Tauri/WebGPU)
#define EC_RING_SLOTS 3
#define EC_RING_MAGIC 0x45435256u /* "ECRV" */

typedef struct {
  uint32_t magic;
  uint32_t width;
  uint32_t height;
  uint32_t stride_bytes;   // width * 4 (RGBA8)
  uint64_t frame_bytes;
  uint64_t write_seq;      // monotonic atomic sequence counter
  uint64_t read_seq;
  uint64_t slot_seq[EC_RING_SLOTS];
  uint64_t slot_pts_us[EC_RING_SLOTS];
  uint32_t slot_crc[EC_RING_SLOTS];
  uint32_t _pad;
} ec_shm_ring_hdr_t;

typedef struct {
  char              shm_name[64];
  int               fd;
  int               is_owner;
  size_t            map_bytes;
  ec_shm_ring_hdr_t *hdr;
  uint8_t           *slots_base;
} ec_shm_ring_t;

int  ec_shm_ring_create(ec_shm_ring_t *r, const char *name, uint32_t w, uint32_t h);
int  ec_shm_ring_open(ec_shm_ring_t *r, const char *name);
int  ec_shm_ring_push(ec_shm_ring_t *r, const uint8_t *rgba, uint64_t pts_us);
// Non-blocking pop of next unread frame into dst_rgba; returns 1 if got frame, 0 if empty, <0 on error
int  ec_shm_ring_pop(ec_shm_ring_t *r, uint8_t *dst_rgba, uint64_t *out_seq, uint64_t *out_pts_us, uint32_t *out_crc);
void ec_shm_ring_close(ec_shm_ring_t *r);

uint32_t ec_crc32_fast(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif
