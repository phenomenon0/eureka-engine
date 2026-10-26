// vtest MMPROJ IMAGE [MAX_TOKENS] [GOLDEN.f32]: tower on one image; prints n, ms, per-row cosine vs golden, dumps /tmp/vtest.f32
#include "vision.h"
#include "vpre.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char err[256] = "";
  if (argc < 3) return 2;
  if (vis_open(argv[1], 2560, err)) { printf("open: %s\n", err); return 1; }
  { VPreCfg c; vpre_cfg_default(&c); c.max_px = (argc > 3 ? atoi(argv[3]) : 256) * 1024; float *pl; int nx, ny;
    if (vpre_image(&c, argv[2], &pl, &nx, &ny, err)) { printf("pre: %s\n", err); return 1; }
    uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < (size_t)3 * nx * ny; i++) { uint32_t u; memcpy(&u, &pl[i], 4); h = (h ^ u) * 1099511628211ull; }
    printf("planar %dx%d fnv=%016llx\n", nx, ny, (unsigned long long)h); }
  float *e; int n, gx, gy; double ms;
  if (vis_encode(argv[2], argc > 3 ? atoi(argv[3]) : 256, &e, &n, &gx, &gy, &ms, err)) { printf("encode: %s\n", err); return 1; }
  printf("tokens=%d grid=%dx%d tower_ms=%.0f\n", n, gx, gy, ms);
  if (argc > 4) {
    FILE *f = fopen(argv[4], "rb");
    if (!f) { printf("no golden\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    int gn = (int)(sz / 4 / 2560);
    float *g = malloc(sz);
    if (fread(g, 1, sz, f) != (size_t)sz) return 1;
    printf("golden rows=%d (ours %d)\n", gn, n);
    if (gn == n) {
      double mn = 2, sum = 0, maxabs = 0;
      for (int r = 0; r < n; r++) {
        double d = 0, a = 0, b = 0;
        for (int i = 0; i < 2560; i++) {
          double x = e[(size_t)r * 2560 + i], y = g[(size_t)r * 2560 + i];
          d += x * y; a += x * x; b += y * y;
          if (fabs(x - y) > maxabs) maxabs = fabs(x - y);
        }
        double c = d / sqrt(a * b);
        if (c < 0.99) printf("  row %d cos=%.4f |ours|=%.2f |gold|=%.2f\n", r, c, sqrt(a), sqrt(b));
        if (c < mn) mn = c;
        sum += c;
      }
      printf("cosine min=%.6f mean=%.6f maxabs=%.4g\n", mn, sum / n, maxabs);
    }
  }
  return 0;
}
