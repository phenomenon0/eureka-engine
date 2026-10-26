// vpre_dump: FUS-A V2 harness. IMAGE -> vpre_image (decode, calc_size_preserved_ratio, Pillow-style bicubic, normalize)
// -> OUT.f32 (planar [3][ny][nx], = clip's inp_raw), and optionally the im2col rows (2x16x16, temporal duplicate,
// 2x2-merge order) -> ROWS.f32. Prints "nx ny" to stdout. Build with -ffp-contract=off.
#include "vpre.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: vpre_dump IMAGE OUT.f32 [ROWS.f32]\n"); return 1; }
  VPreCfg c; vpre_cfg_default(&c);
  float *pl = NULL; int nx = 0, ny = 0; char err[256] = {0};
  if (vpre_image(&c, argv[1], &pl, &nx, &ny, err)) { fprintf(stderr, "%s\n", err); return 2; }
  FILE *f = fopen(argv[2], "wb"); fwrite(pl, sizeof(float), (size_t)3 * nx * ny, f); fclose(f);
  if (argc > 3) {
    const size_t np = (size_t)(nx / c.patch) * (ny / c.patch), k = 2 * 3 * (size_t)c.patch * c.patch;
    float *rows = malloc(np * k * sizeof *rows);
    vpre_im2col(&c, pl, nx, ny, rows);
    f = fopen(argv[3], "wb"); fwrite(rows, sizeof(float), np * k, f); fclose(f); free(rows);
  }
  printf("%d %d\n", nx, ny);
  free(pl);
  return 0;
}
