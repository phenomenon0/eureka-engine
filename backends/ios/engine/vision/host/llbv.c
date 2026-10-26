// llbv.c: ENGINE-VISION2 V5/V6, the engine's face on the V1-V3 vision host code (linked into sys1mac only in a
// VISION=1 build; patches/metal-vision.patch calls llbv_rows). One image per process, chosen by environment:
//   LLB_VIS_IMAGE=<png|jpeg|ppm> + LLB_VIS_MMPROJ=<mmproj-F16.gguf>  the tower IN PROCESS: vpre (V2) -> the tower in
//       the registered gate mode (F16 staging, tanh GELU). (ENGINE-VISION3 V7) LLB_VIS_TOWER picks it:
//         metal (default): vitm.m, the Metal tower;
//         cpu: vit.c (V3, OpenBLAS), the reference; LLB_VIS_THREADS (default 20). Not in a build with -DLLBV_NO_CPU
//         (the iOS slice links no OpenBLAS), where it is refused
//   LLB_VIS_EMBD=<rows.f32>:<nx>:<ny>  precomputed rows (nx * ny rows of d floats; e.g. a V0 golden embd.f32): the V5
//       splice gate and the LLM-only control
// Optional LLB_VIS_SAVE=<file>: the rows the engine will splice, written as F32 (the V5 gate compares them).
// (ENGINE-IPHONEVIS) the library path: llbv_prepare(image, mmproj, d) runs the tower AHEAD of the engine run and holds
// the rows; the run's llbv_rows takes them (no environment needed). The tower's GPU buffers are freed before
// llbv_prepare returns, so the tower and the engine's span are never resident together (peak = max, not sum).
#include "llbv.h"
#include "vit.h"
#include "vitm.h"
#include "vpre.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double llbv_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// (ENGINE-VISION3 V7) the tower behind LLB_VIS_TOWER: Metal (vitm.m) or the CPU reference (vit.c, absent under
// -DLLBV_NO_CPU). Same calls, same F16-staged gate mode
typedef struct {
  int cpu;
  VitW *c;
  VitmW *m;
} Tower;

static int tower_load(Tower *t, const char *mm, uint32_t d, char *err) {
#ifdef LLBV_NO_CPU
  if (t->cpu) {
    snprintf(err, 256, "LLB_VIS_TOWER=cpu: this build has no CPU tower (no OpenBLAS); use metal");
    return 1;
  }
#else
  if (t->cpu) return (t->c = vit_load(mm, d, err)) == NULL;
#endif
  return (t->m = vitm_load(mm, d, err)) == NULL;
}

static int tower_tokens(const Tower *t, int px, int py) {
#ifndef LLBV_NO_CPU
  if (t->cpu) return vit_n_tokens(t->c, px, py);
#endif
  return vitm_n_tokens(t->m, px, py);
}

static int tower_proj_dim(const Tower *t) {
#ifndef LLBV_NO_CPU
  if (t->cpu) return vit_proj_dim(t->c);
#endif
  return vitm_proj_dim(t->m);
}

static int tower_encode(Tower *t, const float *planar, int px, int py, const VitOpt *o, float *r, char *err) {
#ifndef LLBV_NO_CPU
  if (t->cpu) return vit_encode(t->c, planar, px, py, o, r, err);
#endif
  return vitm_encode(t->m, planar, px, py, o, r, err);
}

static void tower_free(Tower *t) {
#ifndef LLBV_NO_CPU
  vit_free(t->c);
#endif
  vitm_free(t->m);
}

// (ENGINE-IPHONEVIS) rows held by llbv_prepare for the next run
static float *llbv_held;
static uint32_t llbv_held_d, llbv_held_n, llbv_held_nx, llbv_held_ny;

int llbv_requested(void) {
  return llbv_held != NULL || getenv("LLB_VIS_IMAGE") != NULL || getenv("LLB_VIS_EMBD") != NULL;
}

// the tower in process: image -> vpre -> the tower (LLB_VIS_TOWER) -> rows [n][d]; the tower freed before return
static int tower_rows(const char *img, const char *mm, uint32_t d, float **rows, uint32_t *n, uint32_t *nx, uint32_t *ny, char *err);

int llbv_rows(uint32_t d, float **rows, uint32_t *n, uint32_t *nx, uint32_t *ny, char *err) {
  const char *img = getenv("LLB_VIS_IMAGE"), *mm = getenv("LLB_VIS_MMPROJ"), *ef = getenv("LLB_VIS_EMBD");
  *rows = NULL, *n = *nx = *ny = 0;
  if (img != NULL && ef != NULL) {
    snprintf(err, 256, "LLB_VIS_IMAGE and LLB_VIS_EMBD are exclusive");
    return 1;
  }
  if (ef != NULL) {
    char path[4096];
    unsigned ax = 0, ay = 0;
    const char *c2 = strrchr(ef, ':'), *c1 = NULL;
    for (const char *p = ef; c2 != NULL && p < c2; p++) {
      if (*p == ':') c1 = p;
    }
    if (c1 == NULL || (size_t)(c1 - ef) >= sizeof path || sscanf(c1, ":%u:%u", &ax, &ay) != 2 || ax == 0 || ay == 0) {
      snprintf(err, 256, "LLB_VIS_EMBD wants <rows.f32>:<nx>:<ny>");
      return 1;
    }
    memcpy(path, ef, (size_t)(c1 - ef));
    path[c1 - ef] = 0;
    size_t want = (size_t)ax * ay * d;
    float *r = malloc(want * sizeof *r);
    FILE *f = fopen(path, "rb");
    size_t got = f && r ? fread(r, sizeof *r, want, f) : 0;
    int extra = f ? fgetc(f) != EOF : 0;
    if (f) fclose(f);
    if (got != want || extra) {
      free(r);
      snprintf(err, 256, "LLB_VIS_EMBD %s: not %u x %u rows of %u floats", path, ax, ay, d);
      return 1;
    }
    *rows = r, *n = ax * ay, *nx = ax, *ny = ay;
    fprintf(stderr, "vision: rows from %s: %u tokens (%u x %u), d %u\n", path, *n, ax, ay, d);
  } else if (llbv_held != NULL) {  // (ENGINE-IPHONEVIS) rows prepared ahead of the run (llbv_prepare)
    if (img != NULL) {
      snprintf(err, 256, "rows prepared by llbv_prepare and LLB_VIS_IMAGE are exclusive");
      return 1;
    }
    if (llbv_held_d != d) {
      snprintf(err, 256, "rows prepared for width %u, the model's is %u", llbv_held_d, d);
      return 1;
    }
    *rows = llbv_held, *n = llbv_held_n, *nx = llbv_held_nx, *ny = llbv_held_ny;
    llbv_held = NULL;
    fprintf(stderr, "vision: rows prepared ahead of the run: %u tokens (%u x %u), d %u\n", *n, *nx, *ny, d);
  } else {
    if (tower_rows(img, mm, d, rows, n, nx, ny, err)) return 1;
  }
  if (*nx * *ny != *n) {
    snprintf(err, 256, "rows %u != grid %u x %u", *n, *nx, *ny);
    return 1;
  }
  const char *sv = getenv("LLB_VIS_SAVE");
  if (sv != NULL) {
    FILE *f = fopen(sv, "wb");
    if (f == NULL || fwrite(*rows, sizeof(float), (size_t)*n * d, f) != (size_t)*n * d) {
      snprintf(err, 256, "LLB_VIS_SAVE %s: write failed", sv);
      if (f) fclose(f);
      return 1;
    }
    fclose(f);
  }
  return 0;
}

static int tower_rows(const char *img, const char *mm, uint32_t d, float **rows, uint32_t *n, uint32_t *nx, uint32_t *ny, char *err) {
  if (mm == NULL) {
    snprintf(err, 256, "LLB_VIS_IMAGE needs LLB_VIS_MMPROJ");
    return 1;
  }
  const char *th = getenv("LLB_VIS_THREADS"), *tw = getenv("LLB_VIS_TOWER");
  Tower t = {tw != NULL && strcmp(tw, "cpu") == 0, NULL, NULL};
  if (tw != NULL && !t.cpu && strcmp(tw, "metal") != 0) {
    snprintf(err, 256, "LLB_VIS_TOWER=%s: want metal or cpu", tw);
    return 1;
  }
  VitOpt o = {1, th ? atoi(th) : 20, NULL, NULL, NULL};
  double t0 = llbv_ms();
  if (tower_load(&t, mm, d, err)) return 1;  // V1's gg_mm_check: refuses all but qwen3vl_merger at projection_dim == d
  double t1 = llbv_ms();
  VPreCfg pc;
  vpre_cfg_default(&pc);
  float *planar = NULL;
  int px = 0, py = 0;
  if (vpre_image(&pc, img, &planar, &px, &py, err)) {
    tower_free(&t);
    return 1;
  }
  double t2 = llbv_ms();
  int nt = tower_tokens(&t, px, py);
  float *r = malloc((size_t)nt * d * sizeof *r);
  if (r == NULL || tower_proj_dim(&t) != (int)d || tower_encode(&t, planar, px, py, &o, r, err)) {
    if (r != NULL && tower_proj_dim(&t) != (int)d) snprintf(err, 256, "tower projection %d != d %u", tower_proj_dim(&t), d);
    free(r), free(planar), tower_free(&t);
    return 1;
  }
  double t3 = llbv_ms();
  const double gpu = t.cpu ? 0 : vitm_gpu_ms(t.m);
  const double wmib = t.cpu ? 0 : vitm_weight_bytes(t.m) / 1048576.0, amib = t.cpu ? 0 : vitm_act_bytes(t.m) / 1048576.0;
  free(planar);
  tower_free(&t);
  *rows = r, *n = (uint32_t)nt, *nx = (uint32_t)px / 32, *ny = (uint32_t)py / 32;
  if (t.cpu)
    fprintf(stderr, "vision: tower %s -> %dx%d px -> %u tokens (%u x %u) | load %.1f ms, preprocess %.1f ms, "
      "encode %.1f ms (CPU, %d threads, F16-staged gate mode)\n", img, px, py, *n, *nx, *ny, t1 - t0, t2 - t1,
      t3 - t2, o.threads);
  else
    fprintf(stderr, "vision: tower %s -> %dx%d px -> %u tokens (%u x %u) | load %.1f ms, preprocess %.1f ms, "
      "encode %.1f ms (Metal, GPU %.1f ms, F16-staged gate mode; weights %.1f MiB, activations %.1f MiB)\n", img,
      px, py, *n, *nx, *ny, t1 - t0, t2 - t1, t3 - t2, gpu, wmib, amib);
  fprintf(stderr, "LLBVIS tower_load_ms=%.1f preprocess_ms=%.1f encode_ms=%.1f tokens=%u tower=%s gpu_ms=%.1f\n",
    t1 - t0, t2 - t1, t3 - t2, *n, t.cpu ? "cpu" : "metal", gpu);
  return 0;
}

int llbv_prepare(const char *image, const char *mmproj, uint32_t d, uint32_t *n, char *err) {
  llbv_forget();
  float *r = NULL;
  uint32_t nn = 0, ax = 0, ay = 0;
  if (image == NULL || mmproj == NULL || d == 0) {
    snprintf(err, 256, "llbv_prepare wants an image, an mmproj and the backbone width");
    return 1;
  }
  if (tower_rows(image, mmproj, d, &r, &nn, &ax, &ay, err)) return 1;
  llbv_held = r, llbv_held_d = d, llbv_held_n = nn, llbv_held_nx = ax, llbv_held_ny = ay;
  if (n != NULL) *n = nn;
  return 0;
}

void llbv_forget(void) {
  free(llbv_held);
  llbv_held = NULL, llbv_held_d = llbv_held_n = llbv_held_nx = llbv_held_ny = 0;
}

// (IMGPERF) the held rows, read only (the wrapper keeps a copy per photo), or NULL; and rows handed back as held (a
// malloc'd copy the next run takes and frees, as llbv_prepare_rgb's own)
const float *llbv_peek(uint32_t *d, uint32_t *n, uint32_t *nx, uint32_t *ny) {
  if (llbv_held == NULL) return NULL;
  *d = llbv_held_d, *n = llbv_held_n, *nx = llbv_held_nx, *ny = llbv_held_ny;
  return llbv_held;
}
void llbv_hold(float *rows, uint32_t d, uint32_t n, uint32_t nx, uint32_t ny) {
  llbv_forget();
  llbv_held = rows, llbv_held_d = d, llbv_held_n = n, llbv_held_nx = nx, llbv_held_ny = ny;
}

int llbv_grid(const char *image, uint32_t *nx, uint32_t *ny, char *err) {
  VImg im;
  VPreCfg c;
  int tw = 0, th = 0;
  if (vpre_decode(image, &im, err)) return 1;
  vpre_cfg_default(&c);
  vpre_target(&c, im.nx, im.ny, &tw, &th);
  vpre_free(&im);
  *nx = (uint32_t)tw / 32, *ny = (uint32_t)th / 32;
  return 0;
}

// (PHONECAM) the resident tower (the camera path, llbv.h)
static Tower llbv_res;
static int llbv_res_on;
static uint32_t llbv_res_d;

int llbv_open(const char *mmproj, uint32_t d, char *err) {
  if (llbv_res_on && llbv_res_d == d) return 0;
  llbv_close();
  if (mmproj == NULL || d == 0) {
    snprintf(err, 256, "llbv_open wants an mmproj and the backbone width");
    return 1;
  }
  Tower t = {0, NULL, NULL};
  if (tower_load(&t, mmproj, d, err)) return 1;
  if (tower_proj_dim(&t) != (int)d) {
    snprintf(err, 256, "tower projection %d != d %u", tower_proj_dim(&t), d);
    tower_free(&t);
    return 1;
  }
  llbv_res = t, llbv_res_on = 1, llbv_res_d = d;
  return 0;
}

void llbv_close(void) {
  if (llbv_res_on) tower_free(&llbv_res);
  llbv_res = (Tower){0, NULL, NULL}, llbv_res_on = 0, llbv_res_d = 0;
}

int llbv_is_open(void) { return llbv_res_on; }

int llbv_prepare_rgb(const uint8_t *rgb, int w, int h, uint32_t d, uint32_t *n, uint32_t *nx, uint32_t *ny, double *ms,
                     char *err) {
  llbv_forget();
  if (rgb == NULL || w < 1 || h < 1 || d == 0) {
    snprintf(err, 256, "llbv_prepare_rgb wants RGB bytes, a size and the backbone width");
    return 1;
  }
  if (!llbv_res_on || llbv_res_d != d) {
    snprintf(err, 256, "llbv_prepare_rgb: no resident tower for width %u (llbv_open)", d);
    return 1;
  }
  double t0 = llbv_ms();
  VPreCfg pc;
  vpre_cfg_default(&pc);
  VImg src = {w, h, (uint8_t *)rgb}, rs;
  int px = 0, py = 0;
  vpre_target(&pc, w, h, &px, &py);
  if (vpre_resize(&src, px, py, &rs, err)) return 1;
  float *planar = malloc(sizeof(float) * 3 * (size_t)px * py);
  if (planar == NULL) {
    vpre_free(&rs);
    snprintf(err, 256, "out of memory (planar)");
    return 1;
  }
  vpre_normalize(&pc, &rs, planar);
  vpre_free(&rs);
  double t1 = llbv_ms();
  int nt = tower_tokens(&llbv_res, px, py);
  float *r = malloc((size_t)nt * d * sizeof *r);
  VitOpt o = {1, 1, NULL, NULL, NULL};
  if (r == NULL || tower_encode(&llbv_res, planar, px, py, &o, r, err)) {
    free(r), free(planar);
    return 1;
  }
  double t2 = llbv_ms();
  free(planar);
  llbv_held = r, llbv_held_d = d, llbv_held_n = (uint32_t)nt, llbv_held_nx = (uint32_t)px / 32,
  llbv_held_ny = (uint32_t)py / 32;
  if (n != NULL) *n = llbv_held_n;
  if (nx != NULL) *nx = llbv_held_nx;
  if (ny != NULL) *ny = llbv_held_ny;
  if (ms != NULL) ms[0] = t1 - t0, ms[1] = t2 - t1, ms[2] = vitm_gpu_ms(llbv_res.m);
  return 0;
}
