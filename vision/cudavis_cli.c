// cudavis-cli: image -> Qwen3.5 vision embedding rows on CUDA (vitc.cu). Driver shape = the Mac's vitm_run.c.
//   cudavis-cli IMAGE MMPROJ [--out EMBD.f32] [--dump DIR [--nodes a,b | --nodes-from MANIFEST]] [--repeat R]
//   cudavis-cli --pre PRE.f32 NX NY MMPROJ ...   (a golden's preprocessed planar tensor instead of an image)
// --dump writes NAME.f32 + manifest.tsv (the goldens' format) from the first encode; embd is always dumped there.
// --batch: see batch_main (Stage 2).
// --repeat R: R encodes in one process; prints each, the warm median (2..R) and whether every embd is BYTE-IDENTICAL.
#include "vitc.h"
#include "vpre.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void vitc_mem_info(size_t *fr, size_t *tot);  // vitc.cu: cudaMemGetInfo

typedef struct { const char *dir; FILE *man; char **names; int nn; } Ctx;

static int want(void *ud, const char *name) {
  Ctx *c = ud;
  if (!strcmp(name, "embd") || !c->nn) return 1;
  for (int i = 0; i < c->nn; i++) if (!strcmp(c->names[i], name)) return 1;
  return 0;
}
static void dumpf(void *ud, const char *name, const float *x, int ne0, int rows) {
  Ctx *c = ud;
  char p[1024];
  snprintf(p, sizeof p, "%s/%s.f32", c->dir, name);
  FILE *f = fopen(p, "wb");
  fwrite(x, sizeof(float), (size_t)ne0 * rows, f), fclose(f);
  fprintf(c->man, "%s\t%d\t%d\t1\t1\n", name, ne0, rows), fflush(c->man);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// (CHAIN-2-1-3 Stage 2) cudavis-cli --batch MMPROJ OUTDIR [--gemm seg|streams|cat] [--repeat R] [--max-batch B] IMG...
// Encodes every image alone (vitc_encode, written OUTDIR/single-K.f32) and then in batches of up to B (default all)
// in one pass (vitc_encode_batch, OUTDIR/batch-K.f32); per image prints whether the batch rows are BYTE-IDENTICAL to
// its single rows. --repeat R times R rounds of each (serial singles vs batched) and prints the warm medians, images/s,
// and the activation / device memory of each mode.
static int batch_main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: cudavis-cli --batch MMPROJ OUTDIR [--gemm seg|streams|cat] [--repeat R] [--max-batch B] IMG...\n"); return 1; }
  const char *mm = argv[2], *dir = argv[3];
  int seg = 1, rep = 1, mb = 0, i = 4;
  for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
    if (!strcmp(argv[i], "--gemm") && i + 1 < argc) i++, seg = !strcmp(argv[i], "cat") ? 0 : !strcmp(argv[i], "streams") ? 2 : 1;
    else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) rep = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--max-batch") && i + 1 < argc) mb = atoi(argv[++i]);
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  const int n = argc - i;
  if (n < 1) { fprintf(stderr, "--batch: no images\n"); return 1; }
  if (mb <= 0 || mb > n) mb = n;
  char err[256] = {0}, p[1200];
  snprintf(p, sizeof p, "mkdir -p '%s'", dir);
  if (system(p)) return 2;
  VitcW *w = vitc_load(mm, 0, err);
  if (!w) { fprintf(stderr, "vitc_load: %s\n", err); return 2; }
  float **pl = calloc(n, sizeof *pl), **es = calloc(n, sizeof *es), **eb = calloc(n, sizeof *eb);
  int *nx = calloc(n, sizeof *nx), *ny = calloc(n, sizeof *ny);
  size_t *ne = calloc(n, sizeof *ne);
  long tok = 0;
  VPreCfg pc;
  vpre_cfg_default(&pc);
  for (int k = 0; k < n; k++) {
    if (vpre_image(&pc, argv[i + k], &pl[k], &nx[k], &ny[k], err)) { fprintf(stderr, "%s: %s\n", argv[i + k], err); return 2; }
    ne[k] = (size_t)vitc_n_tokens(w, nx[k], ny[k]) * vitc_proj_dim(w), tok += vitc_n_tokens(w, nx[k], ny[k]);
    es[k] = malloc(sizeof(float) * ne[k]), eb[k] = malloc(sizeof(float) * ne[k]);
  }
  VitOpt o = {1, 0, NULL, NULL, NULL};
  double *ts = malloc(sizeof(double) * rep), *tb = malloc(sizeof(double) * rep);
  size_t act_s = 0, act_b = 0, fr0, fr1, tot;
  int same = 0;
  for (int r = 0; r < rep; r++) {
    double a = now();
    for (int k = 0; k < n; k++)
      if (vitc_encode(w, pl[k], nx[k], ny[k], &o, es[k], err)) { fprintf(stderr, "vitc_encode %d: %s\n", k, err); return 2; }
    ts[r] = now() - a;
    if (vitc_act_bytes(w) > act_s) act_s = vitc_act_bytes(w);
    a = now();
    for (int k0 = 0; k0 < n; k0 += mb) {
      const int m = n - k0 < mb ? n - k0 : mb;
      if (vitc_encode_batch(w, m, (const float *const *)pl + k0, nx + k0, ny + k0, seg, eb + k0, err)) { fprintf(stderr, "vitc_encode_batch: %s\n", err); return 2; }
    }
    tb[r] = now() - a;
    if (vitc_act_bytes(w) > act_b) act_b = vitc_act_bytes(w);
    if (r == 0) {
      vitc_mem_info(&fr0, &tot);
      for (int k = 0; k < n; k++) {
        const int eq = !memcmp(es[k], eb[k], sizeof(float) * ne[k]);
        same += eq;
        FILE *f;
        snprintf(p, sizeof p, "%s/single-%d.f32", dir, k);
        if ((f = fopen(p, "wb"))) fwrite(es[k], 4, ne[k], f), fclose(f);
        snprintf(p, sizeof p, "%s/batch-%d.f32", dir, k);
        if ((f = fopen(p, "wb"))) fwrite(eb[k], 4, ne[k], f), fclose(f);
        printf("img\t%d\t%s\t%dx%d\t%d tokens\t%s\n", k, argv[i + k], nx[k], ny[k], vitc_n_tokens(w, nx[k], ny[k]),
               eq ? "BYTE-IDENTICAL" : "DIFFER");
      }
    }
  }
  vitc_mem_info(&fr1, &tot);
  qsort(ts, rep, sizeof *ts, cmpd), qsort(tb, rep, sizeof *tb, cmpd);
  const double ms = ts[(rep - 1) / 2], mb_ = tb[(rep - 1) / 2];
  printf("verdict\t%d/%d BYTE-IDENTICAL\tgemm %s\tmax-batch %d\n", same, n, seg == 2 ? "streams" : seg ? "seg" : "cat", mb);
  printf("time\tserial %.2f ms (%.2f img/s)\tbatch %.2f ms (%.2f img/s)\tspeedup %.3fx\t(median of %d, %ld tokens)\n",
         ms * 1e3, n / ms, mb_ * 1e3, n / mb_, ms / mb_, rep, tok);
  printf("mem\tweights %.1f MiB\tact single-max %.1f MiB\tact after batch %.1f MiB\tdevice free %.1f MiB of %.1f\n",
         vitc_weight_bytes(w) / 1048576.0, act_s / 1048576.0, act_b / 1048576.0, fr1 / 1048576.0, tot / 1048576.0);
  (void)fr0;
  vitc_free(w);
  return same == n ? 0 : 3;
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "--batch")) return batch_main(argc, argv);
  const char *img = NULL, *pre = NULL, *mm = NULL, *out = NULL;
  int nx = 0, ny = 0, rep = 1, i = 1;
  char *list = NULL;
  Ctx c = {NULL, NULL, NULL, 0};
  if (argc > 1 && !strcmp(argv[1], "--pre") && argc > 5) pre = argv[2], nx = atoi(argv[3]), ny = atoi(argv[4]), mm = argv[5], i = 6;
  else if (argc > 2) img = argv[1], mm = argv[2], i = 3;
  else {
    fprintf(stderr, "usage: cudavis-cli IMAGE MMPROJ [--out EMBD.f32] [--dump DIR [--nodes a,b | --nodes-from MANIFEST]] [--repeat R]\n"
                    "       cudavis-cli --pre PRE.f32 NX NY MMPROJ [same options]\n");
    return 1;
  }
  for (; i < argc; i++) {
    if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) c.dir = argv[++i];
    else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) rep = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) list = strdup(argv[++i]);
    else if (!strcmp(argv[i], "--nodes-from") && i + 1 < argc) {  // the names column of a goldens manifest.tsv
      FILE *f = fopen(argv[++i], "r");
      char line[512];
      list = calloc(1 << 16, 1);
      while (f && fgets(line, sizeof line, f)) {
        char *t = strchr(line, '\t');
        if (!t || !strncmp(line, "name\t", 5)) continue;
        *t = 0, strcat(list, line), strcat(list, ",");
      }
      if (f) fclose(f);
    } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  if (list) for (char *s = strtok(list, ","); s; s = strtok(NULL, ",")) { c.names = realloc(c.names, sizeof(char *) * (c.nn + 1)); c.names[c.nn++] = s; }
  VitOpt o = {1, 0, NULL, &c, want};
  if (c.dir) {
    char cmd[1100], mp[1100];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", c.dir);
    if (system(cmd)) return 2;
    snprintf(mp, sizeof mp, "%s/manifest.tsv", c.dir);
    c.man = fopen(mp, "w");
    fprintf(c.man, "name\tne0\tne1\tne2\tne3\n");
    o.dump = dumpf;
  }
  char err[256] = {0};
  const double t0 = now();
  VitcW *w = vitc_load(mm, 0, err);
  if (!w) { fprintf(stderr, "vitc_load: %s\n", err); return 2; }
  const double t1 = now();
  float *planar = NULL;
  if (img) {
    VPreCfg pc;
    vpre_cfg_default(&pc);
    if (vpre_image(&pc, img, &planar, &nx, &ny, err)) { fprintf(stderr, "%s\n", err); return 2; }
  } else {
    const size_t n = 3 * (size_t)nx * ny;
    planar = malloc(sizeof(float) * n);
    FILE *f = fopen(pre, "rb");
    if (!f || fread(planar, sizeof(float), n, f) != n) { fprintf(stderr, "bad --pre\n"); return 2; }
    fclose(f);
  }
  const double t2 = now();
  if (c.dir && want(&c, "pre")) dumpf(&c, "pre", planar, nx, 3 * ny);  // the planar input, for byte-compare with clip's inp_raw
  const int nt = vitc_n_tokens(w, nx, ny), pd = vitc_proj_dim(w);
  const size_t ne = (size_t)nt * pd;
  float *embd = malloc(sizeof(float) * ne);
  if (vitc_encode(w, planar, nx, ny, &o, embd, err)) { fprintf(stderr, "vitc_encode: %s\n", err); return 2; }
  const double t3 = now();
  fprintf(stderr, "cudavis-cli: %dx%d -> %d tokens (%d patches) x %d | CUDA, f16act(gate) | load %.3f s, preprocess %.3f s, "
          "encode %.1f ms (GPU %.1f ms) | weights %.1f MiB, activations %.1f MiB\n", nx, ny, nt, nt * 4, pd, t1 - t0, t2 - t1,
          (t3 - t2) * 1e3, vitc_gpu_ms(w), vitc_weight_bytes(w) / 1048576.0, vitc_act_bytes(w) / 1048576.0);
  if (out) {
    FILE *f = fopen(out, "wb");
    if (!f || fwrite(embd, sizeof(float), ne, f) != ne) { fprintf(stderr, "write %s failed\n", out); return 2; }
    fclose(f);
  }
  if (rep > 1) {
    float *e2 = malloc(sizeof(float) * ne);
    double *ms = malloc(sizeof(double) * rep), *gms = malloc(sizeof(double) * rep);
    int same = 1;
    VitOpt o2 = o;
    o2.dump = NULL;
    for (int r = 1; r < rep; r++) {
      const double a = now();
      if (vitc_encode(w, planar, nx, ny, &o2, e2, err)) { fprintf(stderr, "vitc_encode: %s\n", err); return 2; }
      ms[r - 1] = (now() - a) * 1e3, gms[r - 1] = vitc_gpu_ms(w);
      same &= !memcmp(e2, embd, sizeof(float) * ne);
      fprintf(stderr, "cudavis-cli: repeat %d: encode %.2f ms (GPU %.2f ms)\n", r, ms[r - 1], gms[r - 1]);
    }
    qsort(ms, rep - 1, sizeof *ms, cmpd), qsort(gms, rep - 1, sizeof *gms, cmpd);
    fprintf(stderr, "cudavis-cli: warm median of %d: encode %.2f ms (GPU %.2f ms) | repeats vs first: embd %s\n", rep - 1,
            ms[(rep - 2) / 2], gms[(rep - 2) / 2], same ? "BYTE-IDENTICAL" : "DIFFER");
  }
  if (c.man) fclose(c.man);
  vitc_free(w);
  return 0;
}
