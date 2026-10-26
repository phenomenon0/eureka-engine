// vision.h: the phone's vision tower (qwen3vl_merger mmproj, F16 weights mmap'd, NEON/OpenMP) + preprocessing.
// Numerics follow the Mac reference (vit.c "f16act"): GEMM inputs / im2col / attention K,V rounded to F16,
// F32 accumulate; tanh GELU; 2D vision rope base 10000. Only the GEMM summation order differs.
#pragma once
// map + validate the mmproj; proj_dim must equal the backbone width. 0 ok, else err (<= 255 chars).
int vis_open(const char *mmproj, int proj_dim, char *err);
// image file -> embd [*ntok][proj_dim] (malloc'd, caller frees); merged-token grid is *gx x *gy (row-major).
// max_tokens caps the merged-token count (the preprocessor fits the image under it). ms = tower wall ms.
int vis_encode(const char *img, int max_tokens, float **embd, int *ntok, int *gx, int *gy, double *ms, char *err);
// tower on an already-preprocessed planar [3][ny][nx] (tests)
int vis_tower(const float *planar, int nx, int ny, float *embd, char *err);
