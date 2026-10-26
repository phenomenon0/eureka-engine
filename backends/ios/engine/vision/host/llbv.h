// llbv.h: ENGINE-VISION2, the vision rows the engine splices (see llbv.c). Declared again (extern) inside the
// emitted C by patches/metal-vision.patch; keep the two in step.
#pragma once
#include <stdint.h>

// 1 if the environment asks for an image (LLB_VIS_IMAGE or LLB_VIS_EMBD)
int llbv_requested(void);
// the image's rows: *rows malloc'd [n][d] F32, n = nx * ny (the merged grid). 0 ok, else err (256 B) set
int llbv_rows(uint32_t d, float **rows, uint32_t *n, uint32_t *nx, uint32_t *ny, char *err);
// (ENGINE-IPHONEVIS) the library path (no environment): run the tower for backbone width d AHEAD of the engine run and
// hold the rows (*n = their count, may be NULL); the next llbv_rows takes them. The tower's buffers are freed before it
// returns. 0 ok, else err (256 B) set. llbv_forget drops held rows (e.g. the request was cancelled before the run).
int llbv_prepare(const char *image, const char *mmproj, uint32_t d, uint32_t *n, char *err);
void llbv_forget(void);
// (IMGPERF) the rows llbv holds for the next run (read only; NULL when none), and a malloc'd copy handed back as held
const float *llbv_peek(uint32_t *d, uint32_t *n, uint32_t *nx, uint32_t *ny);
void llbv_hold(float *rows, uint32_t d, uint32_t n, uint32_t nx, uint32_t ny);
// the merged grid an image will produce (vpre's target size / 32), without the tower: the prompt carries nx * ny pads
int llbv_grid(const char *image, uint32_t *nx, uint32_t *ny, char *err);
// (PHONECAM) the camera path: the tower stays loaded (weights + pipelines resident) between frames, and a frame comes
// in memory. llbv_open loads it once for backbone width d (0 ok, else err set); llbv_close frees it. llbv_prepare_rgb:
// RGB bytes (w x h, row stride w*3) -> vpre's target/resize/normalize (the same functions vpre_image runs after its
// decode) -> the tower (the resident one if open, else loaded and freed as llbv_prepare) -> rows held for the next run,
// as llbv_prepare. ms (may be NULL): [0] preprocess, [1] encode (host wall), [2] the tower's GPU time
int llbv_open(const char *mmproj, uint32_t d, char *err);
void llbv_close(void);
int llbv_is_open(void);
int llbv_prepare_rgb(const uint8_t *rgb, int w, int h, uint32_t d, uint32_t *n, uint32_t *nx, uint32_t *ny, double *ms,
                     char *err);
