// vpre.c: see vpre.h. Every arithmetic choice below copies mtmd-image.cpp / clip-impl.h line for line; the comments
// name the C++ it mirrors. The gate (V2) is byte equality of vpre_image's output with clip's inp_raw.
#include "vpre.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"  // v2.30, the same file as llama.cpp vendor/stb (sha256 594c2fe3...)

void vpre_cfg_default(VPreCfg *c) {
  c->patch = 16, c->merge = 2;
  c->min_px = 8 * 32 * 32;     // set_limit_image_tokens(8, 4096) x patch_area (= (patch*merge)^2)
  c->max_px = 4096 * 32 * 32;
  for (int i = 0; i < 3; i++) c->mean[i] = 0.5f, c->std[i] = 0.5f;
}

void vpre_free(VImg *img) { free(img->rgb); img->rgb = NULL; }

static int ppm_tok(FILE *f) {  // next header integer (skips whitespace and # comments)
  int ch, v = 0, any = 0;
  for (;;) {
    ch = fgetc(f);
    if (ch == '#') { while (ch != '\n' && ch != EOF) ch = fgetc(f); continue; }
    if (ch == EOF) return -1;
    if (ch >= '0' && ch <= '9') break;
  }
  while (ch >= '0' && ch <= '9') v = v * 10 + (ch - '0'), any = 1, ch = fgetc(f);
  return any ? v : -1;  // the single whitespace after maxval is consumed here
}

int vpre_decode(const char *path, VImg *out, char *err) {
  memset(out, 0, sizeof *out);
  FILE *f = fopen(path, "rb");
  if (!f) return snprintf(err, 256, "open %s failed", path), -1;
  char m[2] = {0};
  if (fread(m, 1, 2, f) == 2 && m[0] == 'P' && m[1] == '6') {
    int w = ppm_tok(f), h = ppm_tok(f), mx = ppm_tok(f);
    if (w <= 0 || h <= 0 || mx != 255) { fclose(f); return snprintf(err, 256, "%s: PPM must be P6 maxval 255", path), -1; }
    out->nx = w, out->ny = h, out->rgb = malloc((size_t)w * h * 3);
    size_t got = out->rgb ? fread(out->rgb, 1, (size_t)w * h * 3, f) : 0;
    fclose(f);
    if (got != (size_t)w * h * 3) { vpre_free(out); return snprintf(err, 256, "%s: PPM truncated", path), -1; }
    return 0;
  }
  // everything else: stb_image exactly as mtmd_helper_bitmap_init_from_buf (stbi_load_from_memory, req_comp 3)
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *buf = malloc(n > 0 ? (size_t)n : 1);
  size_t got = buf ? fread(buf, 1, (size_t)n, f) : 0;
  fclose(f);
  if (got != (size_t)n) { free(buf); return snprintf(err, 256, "%s: read failed", path), -1; }
  int nx, ny, nc;
  unsigned char *d = stbi_load_from_memory(buf, (int)n, &nx, &ny, &nc, 3);
  free(buf);
  if (!d) return snprintf(err, 256, "%s: stb_image: %s", path, stbi_failure_reason()), -1;
  out->nx = nx, out->ny = ny, out->rgb = malloc((size_t)nx * ny * 3);
  memcpy(out->rgb, d, (size_t)nx * ny * 3);
  stbi_image_free(d);
  return 0;
}

int vpre_write_ppm(const char *path, const VImg *img) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  fprintf(f, "P6\n%d %d\n255\n", img->nx, img->ny);
  size_t n = fwrite(img->rgb, 1, (size_t)img->nx * img->ny * 3, f);
  fclose(f);
  return n == (size_t)img->nx * img->ny * 3 ? 0 : -1;
}

// img_tool::calc_size_preserved_ratio (longest_edge 0): float math as in the C++ (x is float, f cast to float)
void vpre_target(const VPreCfg *c, int width, int height, int *tw, int *th) {
  const int f = c->patch * c->merge;
#define ROUND_BY(x) ((int)roundf((x) / (float)f) * f)
#define CEIL_BY(x) ((int)ceilf((x) / (float)f) * f)
#define FLOOR_BY(x) ((int)floorf((x) / (float)f) * f)
  int w_bar = ROUND_BY((float)width), h_bar = ROUND_BY((float)height);
  if (w_bar < f) w_bar = f;
  if (h_bar < f) h_bar = f;
  if (c->max_px > 0 && h_bar * w_bar > c->max_px) {
    const float beta = sqrtf((float)height * width / c->max_px);  // static_cast<float>(height) * width / max_pixels
    h_bar = FLOOR_BY(height / beta), w_bar = FLOOR_BY(width / beta);
    if (h_bar < f) h_bar = f;
    if (w_bar < f) w_bar = f;
  } else if (c->min_px > 0 && h_bar * w_bar < c->min_px) {
    const float beta = sqrtf((float)c->min_px / ((float)height * width));
    h_bar = CEIL_BY(height * beta), w_bar = CEIL_BY(width * beta);
  }
#undef ROUND_BY
#undef CEIL_BY
#undef FLOOR_BY
  *tw = w_bar, *th = h_bar;
}

// ---- resize_pillow (RESIZE_ALGO_BICUBIC), fixed point, 22 fractional bits
#define PREC 22

static double bicubic(double x) {
  if (x < 0.0) x = -x;
  const double a = -0.5;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

static uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }

// precompute_weights: returns ksize; bounds[2*out], weights[out*ksize]
static int weights(int in, int out, int **bounds, int32_t **kk) {
  double support, scale, filterscale, center, ww, ss;
  filterscale = scale = (double)in / out;
  if (filterscale < 1.0) filterscale = 1.0;
  support = 2.0 * filterscale;
  int ksize = (int)ceil(support) * 2 + 1;
  double *pre = malloc(sizeof(double) * (size_t)out * ksize);
  *bounds = malloc(sizeof(int) * (size_t)out * 2);
  *kk = malloc(sizeof(int32_t) * (size_t)out * ksize);
  for (int xx = 0; xx < out; xx++) {
    center = (xx + 0.5) * scale;
    ww = 0.0;
    ss = 1.0 / filterscale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5);
    if (xmax > in) xmax = in;
    xmax -= xmin;
    int x;
    for (x = 0; x < xmax; x++) {
      double w = bicubic((x + xmin - center + 0.5) * ss);
      pre[(size_t)xx * ksize + x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++)
      if (ww != 0.0) pre[(size_t)xx * ksize + x] /= ww;
    for (; x < ksize; x++) pre[(size_t)xx * ksize + x] = 0;
    (*bounds)[xx * 2 + 0] = xmin;
    (*bounds)[xx * 2 + 1] = xmax;
  }
  const double fxp = ldexp(1.0, PREC);
  for (size_t i = 0; i < (size_t)out * ksize; i++) {
    const double r = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
    (*kk)[i] = (int32_t)r;
  }
  free(pre);
  return ksize;
}

static uint8_t *horiz(const uint8_t *src, int in_nx, int in_ny, int out_nx, int ksize, const int *b, const int32_t *w) {
  uint8_t *out = malloc((size_t)out_nx * in_ny * 3);
  for (int yy = 0; yy < in_ny; yy++) {
    const uint8_t *row = src + (size_t)yy * in_nx * 3;
    uint8_t *dst = out + (size_t)yy * out_nx * 3;
    for (int xx = 0; xx < out_nx; xx++) {
      const int xmin = b[xx * 2], xcnt = b[xx * 2 + 1];
      const int32_t *k = &w[xx * ksize];
      const uint8_t *p = row + (size_t)xmin * 3;
      int32_t s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
      for (int x = 0; x < xcnt; x++, p += 3) s0 += p[0] * k[x], s1 += p[1] * k[x], s2 += p[2] * k[x];
      dst[xx * 3 + 0] = clip8(s0 >> PREC), dst[xx * 3 + 1] = clip8(s1 >> PREC), dst[xx * 3 + 2] = clip8(s2 >> PREC);
    }
  }
  return out;
}

static uint8_t *vert(const uint8_t *src, int in_nx, int out_ny, int ksize, const int *b, const int32_t *w) {
  const size_t re = (size_t)in_nx * 3;
  uint8_t *out = malloc(re * out_ny);
  int32_t *acc = malloc(re * sizeof *acc);
  for (int yy = 0; yy < out_ny; yy++) {
    const int ymin = b[yy * 2], ycnt = b[yy * 2 + 1];
    const int32_t *k = &w[yy * ksize];
    for (size_t i = 0; i < re; i++) acc[i] = 1 << (PREC - 1);
    for (int y = 0; y < ycnt; y++) {
      const uint8_t *row = src + (size_t)(ymin + y) * re;
      for (size_t i = 0; i < re; i++) acc[i] += row[i] * k[y];
    }
    for (size_t i = 0; i < re; i++) out[(size_t)yy * re + i] = clip8(acc[i] >> PREC);
  }
  free(acc);
  return out;
}

static uint8_t *pillow(const VImg *src, int tw, int th) {
  const int sw = src->nx, sh = src->ny;
  int *bh = NULL, *bv = NULL;
  int32_t *wh = NULL, *wv = NULL;
  int kh = 0, kv = 0;
  const int need_h = tw != sw, need_v = th != sh;
  if (need_h) kh = weights(sw, tw, &bh, &wh);
  if (need_v) kv = weights(sh, th, &bv, &wv);
  uint8_t *out;
  if (need_h && need_v) {
    uint8_t *tmp = horiz(src->rgb, sw, sh, tw, kh, bh, wh);
    out = vert(tmp, tw, th, kv, bv, wv);
    free(tmp);
  } else if (need_h) out = horiz(src->rgb, sw, sh, tw, kh, bh, wh);
  else if (need_v) out = vert(src->rgb, sw, th, kv, bv, wv);
  else out = malloc((size_t)sw * sh * 3), memcpy(out, src->rgb, (size_t)sw * sh * 3);
  free(bh), free(bv), free(wh), free(wv);
  return out;
}

// img_tool::resize(src, dst, target, BICUBIC, PAD_CEIL, {0,0,0})
int vpre_resize(const VImg *src, int tw, int th, VImg *dst, char *err) {
  if (tw <= 0 || tw > 65536 || th <= 0 || th > 65536) return snprintf(err, 256, "resize target %dx%d out of range", tw, th), -1;
  dst->nx = tw, dst->ny = th;
  if (src->nx == tw && src->ny == th) {  // simple copy
    dst->rgb = malloc((size_t)tw * th * 3);
    memcpy(dst->rgb, src->rgb, (size_t)tw * th * 3);
    return 0;
  }
  const float scale_w = (float)tw / src->nx, scale_h = (float)th / src->ny;
  const float scale = scale_w < scale_h ? scale_w : scale_h;
  int nw = (int)ceilf(src->nx * scale), nh = (int)ceilf(src->ny * scale);  // PAD_CEIL
  if (nw > tw) nw = tw;
  if (nh > th) nh = th;
  uint8_t *r = pillow(src, nw, nh);
  dst->rgb = calloc((size_t)tw * th * 3, 1);  // fill(pad_color = 0,0,0)
  const int ox = (tw - nw) / 2, oy = (th - nh) / 2;
  for (int y = 0; y < nh; y++)
    for (int x = 0; x < nw; x++) {
      const int dx = x + ox, dy = y + oy;
      if (dx < 0 || dy < 0 || dx >= tw || dy >= th) continue;
      memcpy(dst->rgb + ((size_t)dy * tw + dx) * 3, r + ((size_t)y * nw + x) * 3, 3);
    }
  free(r);
  return 0;
}

// clip_image_f32::from_u8 + normalize, then clip.cpp's planar copy into inp_raw
void vpre_normalize(const VPreCfg *c, const VImg *img, float *planar) {
  const size_t n = (size_t)img->nx * img->ny;
  for (size_t i = 0; i < n; i++)
    for (int ch = 0; ch < 3; ch++) {
      float v = (float)img->rgb[i * 3 + ch] / 255.0f;
      v = (v - c->mean[ch]) / c->std[ch];
      planar[ch * n + i] = v;
    }
}

void vpre_im2col(const VPreCfg *c, const float *planar, int nx, int ny, float *rows) {
  const int P = c->patch, M = c->merge, px = nx / P, py = ny / P, K = 3 * P * P;
  const size_t n = (size_t)nx * ny;
  size_t r = 0;
  for (int y = 0; y < py; y += M)
    for (int x = 0; x < px; x += M)
      for (int dy = 0; dy < M; dy++)
        for (int dx = 0; dx < M; dx++, r++) {
          float *row = rows + r * 2 * K;
          const int y0 = (y + dy) * P, x0 = (x + dx) * P;
          for (int ch = 0; ch < 3; ch++)
            for (int ky = 0; ky < P; ky++)
              for (int kx = 0; kx < P; kx++)
                row[(ch * P + ky) * P + kx] = planar[ch * n + (size_t)(y0 + ky) * nx + x0 + kx];
          memcpy(row + K, row, sizeof(float) * K);  // temporal duplicate: frame 1 = frame 0
        }
}

int vpre_image(const VPreCfg *c, const char *path, float **planar, int *nx, int *ny, char *err) {
  VImg src, rs;
  if (vpre_decode(path, &src, err)) return -1;
  int tw, th;
  vpre_target(c, src.nx, src.ny, &tw, &th);
  if (vpre_resize(&src, tw, th, &rs, err)) { vpre_free(&src); return -1; }
  vpre_free(&src);
  *planar = malloc(sizeof(float) * 3 * (size_t)tw * th);
  vpre_normalize(c, &rs, *planar);
  vpre_free(&rs);
  *nx = tw, *ny = th;
  return 0;
}
