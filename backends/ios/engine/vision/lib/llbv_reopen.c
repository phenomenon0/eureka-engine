// (ENGINE-IPHONEVIS) llbv_reopen: the PHONE's library path for an image request, on the Mac slice of the vision library
// (build/vis-macos/libllbengine-vis.a = the iOS recipe: llb_engine.m over the VISION emitted C + llbv/vpre/vmm/vitm,
// no OpenBLAS). One process, three runs through the public API at llb_span()'s default (the phone's chat span):
//   1. image: LLB_VIS_IMAGE + LLB_VIS_MMPROJ set for the call, the image prompt ids, n_new ids
//   2. text:  the variables unset, a text prompt (bend_close must have reset every vision static)
//   3. image: run 1 again (the reopened engine reproduces it)
// Prints "run<k> ids ..." and the release report per run. The ids files are argv[3] (image) and argv[4] (text).
//   llbv_reopen <model_dir> <cache_dir> <image_ids.txt> <text_ids.txt> <image> <mmproj> <n_new> [env|prepare]
// env (default): the image through LLB_VIS_IMAGE/LLB_VIS_MMPROJ (the tower runs INSIDE the engine run, next to the span);
// prepare: llbv_prepare(image, mmproj, 2560) BEFORE llb_load (the tower runs and is freed before the span exists) and
// llbv_grid's pad count checked against the ids file. Compare the two modes' peak footprint with /usr/bin/time -l.
#include "llb_engine.h"
#include "llbv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_ids(const char *p, uint32_t *ids, int cap) {
  FILE *f = fopen(p, "r");
  int n = 0;
  unsigned long v;
  while (f && n < cap && fscanf(f, "%lu", &v) == 1) ids[n++] = (uint32_t)v;
  if (f) fclose(f);
  return n;
}

static int run(int k, const char *dir, const char *cache, const uint32_t *ids, int n, int n_new, uint32_t *out) {
  if (llb_load(dir, cache) != 0) { fprintf(stderr, "run%d: llb_load: %s\n", k, llb_error()); return -1; }
  llb_stats st;
  int got = llb_generate(ids, n, n_new, out, &st);
  printf("run%d ids", k);
  for (int i = 0; i < got; i++) printf(" %u", out[i]);
  printf("\nrun%d got %d exit %d span_mb %llu load_ms %.0f prefill_ms %.0f total_ms %.0f | %s | err '%s'\n", k, got, st.exit_code,
         (unsigned long long)(st.span_bytes >> 20), st.load_ms, st.prefill_ms, st.total_ms, llb_release_report(), llb_error());
  const char *lg = llb_log();
  for (const char *l = lg; l && *l;) {  // the engine's vision lines
    const char *e = strchr(l, '\n');
    size_t len = e ? (size_t)(e - l) : strlen(l);
    if (!strncmp(l, "vision:", 7) || !strncmp(l, "LLBVIS", 6)) printf("run%d %.*s\n", k, (int)len, l);
    l = e ? e + 1 : l + len;
  }
  llb_free();
  fflush(stdout);
  return got;
}

int main(int argc, char **argv) {
  const int prep = argc == 9 && !strcmp(argv[8], "prepare");
  if (argc != 8 && !(argc == 9 && (prep || !strcmp(argv[8], "env")))) { fprintf(stderr, "usage: llbv_reopen <model_dir> <cache_dir> <image_ids> <text_ids> <image> <mmproj> <n_new>\n"); return 2; }
  static uint32_t im[8192], tx[8192], o1[512], o2[512], o3[512];
  const int ni = read_ids(argv[3], im, 8192), nt = read_ids(argv[4], tx, 8192), nn = atoi(argv[7]);
  if (ni < 1 || nt < 1 || nn < 1 || nn > 512) { fprintf(stderr, "bad ids / n_new\n"); return 2; }
  printf("device %s\n", llb_device_info());
  char err[256] = {0};
  uint32_t gx = 0, gy = 0, pads = 0, rn = 0;
  for (int i = 0; i < ni; i++) pads += im[i] == 248056u;
  if (llbv_grid(argv[5], &gx, &gy, err)) { fprintf(stderr, "llbv_grid: %s\n", err); return 2; }
  printf("mode %s | llbv_grid %u x %u = %u pads; the ids file carries %u\n", prep ? "prepare" : "env", gx, gy, gx * gy, pads);
  // prepare: the tower before llb_load; env: the engine runs it inside the run
#define IMAGE_ON() do { if (prep) { if (llbv_prepare(argv[5], argv[6], 2560, &rn, err)) { fprintf(stderr, "llbv_prepare: %s\n", err); return 2; } \
    printf("llbv_prepare: %u rows held\n", rn); } else setenv("LLB_VIS_IMAGE", argv[5], 1), setenv("LLB_VIS_MMPROJ", argv[6], 1); } while (0)
  IMAGE_ON();
  const int g1 = run(1, argv[1], argv[2], im, ni, nn, o1);
  unsetenv("LLB_VIS_IMAGE"), unsetenv("LLB_VIS_MMPROJ");
  const int g2 = run(2, argv[1], argv[2], tx, nt, nn, o2);
  IMAGE_ON();
  const int g3 = run(3, argv[1], argv[2], im, ni, nn, o3);
  const int same = g1 == g3 && g1 > 0 && !memcmp(o1, o3, sizeof(uint32_t) * (size_t)g1);
  printf("image run 3 == run 1: %s\n", same ? "BYTE-IDENTICAL ids" : "DIFFERS");
  return g1 == nn && g2 == nn && same ? 0 : 1;
}
