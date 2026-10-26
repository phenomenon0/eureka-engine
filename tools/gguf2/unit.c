// GGUF2 S1 unit test: host/llb_gguf.c, CPU only, against a pack made by gguf2pack.
//
//   unit MODEL.gguf PACK_DIR [--mutate] [--threads N]
//
// Every tensor the loader produces (each layer region as the engine asks for
// it, then every non-layer tensor as a virtual pack.bin range) must hash to the
// pack manifest's sha256; each layer's size must equal table.bin word 0; the
// GGUF's pack_bytes must equal the manifest's. A virtual range cut mid-row
// inside every layer must equal the same slice of the layer read (Q8_0 partial
// rows). --mutate then breaks the loader three ways (a flipped output byte,
// Q8_0 copied raw without the row-split, one tensor read 1 byte off) and
// passes only if every mutation is caught. Exit 0 = PASS.
//
// Built with -DGG_CUDA (tools/gguf2/g4.sh, GPU: under the bench lock): every
// region is first put on the device the way the engine's io_put does
// (cuMemAllocManaged + cuMemcpyHtoD), the host copy poisoned, and the hashes
// are taken of a cuMemcpyDtoH readback: the device bytes.

#include "../../host/llb_gguf.c"

#include <time.h>

#ifdef GG_CUDA
#include <cuda.h>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { fprintf(stderr, "%s: CUDA error %d\n", #x, r_); exit(1); } } while (0)
// host -> managed device block (io_put's cuMemcpyHtoD) -> poison host -> readback
static void dev_roundtrip(uint8_t *d, size_t n) {
  CUdeviceptr p;
  CK(cuMemAllocManaged(&p, n, CU_MEM_ATTACH_GLOBAL));
  CK(cuMemcpyHtoD(p, d, n));
  memset(d, 0xA5, n);
  CK(cuMemcpyDtoH(d, p, n));
  CK(cuMemFree(p));
}
#else
#define dev_roundtrip(d, n) ((void)0)
#endif

static char *man;
static uint32_t *table;
static int verbose = 1;

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

// sha every whole tensor of g in dst = pack [lo, lo + n); returns mismatches
static int check(const GgFile *g, const uint8_t *dst, uint64_t lo, uint64_t n, size_t *seen) {
  int bad = 0;
  for (size_t i = 0; i < g->nt; i++) {
    const GgT *t = &g->t[i];
    if (t->off < lo || t->off + t->bytes > lo + n) continue;
    GgSha s;
    char got[65], want[65];
    gg_sha_init(&s);
    gg_sha_update(&s, dst + (t->off - lo), (size_t)t->bytes);
    gg_sha_hex(&s, got);
    if (!gg_manifest_sha(man, t->name, want)) {
      printf("  %s: not in the manifest\n", t->name);
      bad++;
    } else if (strcmp(got, want)) {
      if (verbose) printf("  %s: sha256 %s, manifest %s\n", t->name, got, want);
      bad++;
    }
    (*seen)++;
  }
  return bad;
}

// all checks; returns the number of failures
static int run(GgFile *g, int threads, int mutate) {
  char err[256];
  int bad = 0;
  size_t seen = 0;
  uint64_t total = 0;
  double t0 = now_s();
  for (int L = 0; L < g->nl; L++) {
    uint64_t n = g->hi[L] - g->lo[L], lo;
    if (table[L * 32] != n) printf("  layer %d: %llu bytes, table.bin says %u\n", L, (unsigned long long)n, table[L * 32]), bad++;
    if (gg_region(g, (uint32_t)L, 0, n, &lo, err)) { printf("  %s\n", err); return bad + 1; }
    uint8_t *d = malloc(n);
    if (!d || gg_fill(g, lo, n, d, threads, err)) { printf("  layer %d: %s\n", L, d ? err : "oom"); free(d); return bad + 1; }
    if (!mutate) dev_roundtrip(d, n);
    if (mutate == 1 && L == g->nl / 2) d[n / 3] ^= 1;
    total += n;
    bad += check(g, d, lo, n, &seen);
    // a virtual range cut mid-row at both ends equals the same slice
    uint64_t a = lo + 12345 < lo + n ? 12345 : 0, b = n > a + 777 ? n - 777 : n;
    uint8_t *v = malloc(b - a);
    if (!v || gg_fill(g, lo + a, b - a, v, threads, err)) { printf("  layer %d virtual: %s\n", L, v ? err : "oom"); bad++; }
    else if (memcmp(v, d + a, b - a)) printf("  layer %d: virtual [%llu, %llu) differs from the layer read\n", L,
                                             (unsigned long long)(lo + a), (unsigned long long)(lo + b)), bad++;
    free(v), free(d);
  }
  for (size_t i = 0; i < g->nt; i++) {
    const GgT *t = &g->t[i];
    if (t->layer >= 0) continue;
    uint8_t *d = malloc(t->bytes);
    if (!d || gg_fill(g, t->off, t->bytes, d, threads, err)) { printf("  %s: %s\n", t->name, d ? err : "oom"); free(d); return bad + 1; }
    if (!mutate) dev_roundtrip(d, t->bytes);
    total += t->bytes;
    bad += check(g, d, t->off, t->bytes, &seen);
    free(d);
  }
  double dt = now_s() - t0;
  if (verbose || !mutate)
    printf("  %zu of %zu tensors hashed, %d failures, %.2f GB in %.1f s (%.0f MB/s, %d threads)\n", seen, g->nt, bad,
           total / 1e9, dt, total / dt / 1e6, threads);
  if (seen != g->nt) bad++;
  return bad;
}

int main(int argc, char **argv) {
  const char *src = NULL, *dir = NULL;
  int mutate = 0, threads = gg_threads();
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mutate")) mutate = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!src) src = argv[i];
    else if (!dir) dir = argv[i];
    else src = NULL;
  }
  if (!src || !dir) { fputs("usage: unit MODEL.gguf PACK_DIR [--mutate] [--threads N]\n", stderr); return 2; }
  char p[4096], err[256];
  snprintf(p, sizeof p, "%s/manifest.json", dir);
  if (!(man = gg_slurp(p))) { fprintf(stderr, "read %s failed\n", p); return 2; }
  snprintf(p, sizeof p, "%s/table.bin", dir);
  table = (uint32_t *)gg_slurp(p);
#ifdef GG_CUDA
  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  printf("device mode: hashes of cuMemcpyDtoH readbacks of managed blocks\n");
#endif
  GgFile g;
  double t0 = now_s();
  if (gg_open(&g, src, err)) { fprintf(stderr, "gg_open: %s\n", err); return 1; }
  printf("%s: %zu tensors, %d layers, pack_bytes %llu, header parsed in %.3f s\n", src, g.nt, g.nl,
         (unsigned long long)g.pack_bytes, now_s() - t0);
  const char *pb = strstr(man, "\n \"pack_bytes\": ");
  int fails = 0;
  if (!pb || strtoull(pb + 16, NULL, 10) != g.pack_bytes) printf("  pack_bytes differs from the manifest\n"), fails++;
  if (!table) printf("  no table.bin\n"), fails++;
  else fails += run(&g, threads, 0);
  printf("clean: %s\n", fails ? "FAIL" : "PASS");
  if (fails || !mutate) return fails ? 1 : 0;

  // mutations: each must be caught (run() > 0)
  verbose = 0;
  int caught = 0, n = 0;
  n++, caught += run(&g, threads, 1) > 0;
  printf("mutation 1 (one output byte flipped): %s\n", caught == n ? "caught" : "MISSED");
  int hasq8 = 0;
  for (size_t i = 0; i < g.nt; i++)
    if (g.t[i].q8) g.t[i].q8 = 0, hasq8 = 1;
  if (hasq8) {
    n++, caught += run(&g, threads, 0) > 0;
    printf("mutation 2 (Q8_0 copied raw, no row-split): %s\n", caught == n ? "caught" : "MISSED");
    for (size_t i = 0; i < g.nt; i++) g.t[i].q8 = g.t[i].rb != 0;
  } else {
    printf("mutation 2 (Q8_0 raw): skipped, no Q8_0 tensors in this GGUF\n");
  }
  g.t[g.nt / 2].src += 1;
  n++, caught += run(&g, threads, 0) > 0;
  printf("mutation 3 (%s read 1 byte off): %s\n", g.t[g.nt / 2].name, caught == n ? "caught" : "MISSED");
  printf("mutate: %s (%d of %d caught)\n", caught == n ? "PASS" : "FAIL", caught, n);
  return caught == n ? 0 : 1;
}
