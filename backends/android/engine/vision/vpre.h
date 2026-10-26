// vpre.h: ENGINE-VISION V2 image preprocessing for the Qwen3-VL tower (qwen3vl_merger), a C port of
// llama.cpp mtmd's dyn-size path (tools/mtmd/mtmd-image.cpp @ fn-run 2690873):
//   decode (stb_image, as mtmd-helper; or a binary PPM) -> calc_size_preserved_ratio (align 32, min/max pixels)
//   -> img_tool::resize (PAD_CEIL: Pillow fixed-point bicubic to the fitted size, black pad, centered)
//   -> u8/255, (x - mean)/std -> planar [3][H][W] f32 (= clip's "inp_raw")
// plus the patch gather the tower's patch-embed GEMM eats:
//   im2col 2x16x16 with the temporal duplicate (a still image is frame 0 and frame 1), rows in 2x2-merge order.
// Must be compiled with -ffp-contract=off (the reference does no FMA contraction in normalize).
#pragma once
#include <stdint.h>

typedef struct {
  int nx, ny;
  uint8_t *rgb;  // nx*ny*3, row-major RGB
} VImg;

typedef struct {
  int patch, merge;       // 16, 2
  int min_px, max_px;     // pixel bounds (clip.cpp qwen3vl defaults: 8 and 4096 tokens -> 8192, 4194304)
  float mean[3], std[3];  // from the mmproj (0.5)
} VPreCfg;

void vpre_cfg_default(VPreCfg *c);
int vpre_decode(const char *path, VImg *out, char *err);  // 0 ok; PPM (P6, maxval 255) natively, else stb_image
int vpre_write_ppm(const char *path, const VImg *img);
void vpre_free(VImg *img);
void vpre_target(const VPreCfg *c, int w, int h, int *tw, int *th);       // calc_size_preserved_ratio
int vpre_resize(const VImg *src, int tw, int th, VImg *dst, char *err);  // img_tool::resize, PAD_CEIL, black
void vpre_normalize(const VPreCfg *c, const VImg *img, float *planar);   // [3][ny][nx]
// planar [3][ny][nx] -> rows [n_patches][2*3*patch*patch]; row r = merged token r/4, sub-patch (dy,dx) = r%4 (dy major);
// within a row: [t][c][ky][kx], t = 0,1 identical (the temporal duplicate)
void vpre_im2col(const VPreCfg *c, const float *planar, int nx, int ny, float *rows);
// the whole host path: decode -> target -> resize -> normalize. *planar malloc'd, *nx/*ny set.
int vpre_image(const VPreCfg *c, const char *path, float **planar, int *nx, int *ny, char *err);
