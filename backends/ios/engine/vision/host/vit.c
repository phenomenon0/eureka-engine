// vit.c: see vit.h. CPU reference tower: OpenBLAS sgemm for every GEMM, libdispatch for the row-parallel ops.
// Numerics follow the reference's precision where llama.cpp makes a choice (registered in vision/TOLERANCES.md):
// with f16act, GEMM inputs (and conv im2col) are rounded to F16 and accumulated in F32 (Metal kernel_mul_mm_f16_f32
// stages B as half), attention K/V are rounded to F16 (clip.cpp casts them for flash-attn), softmax is F32. Vision
// rope uses Metal's direct theta (pos * base^(-2p/36)), GELU is the tanh form for both FFN and merger.
// Build: -ffp-contract=off -fblocks.
#include "vit.h"
#include "vmm.h"
#include <cblas.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GELU_COEF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

typedef struct {
  float *ln1_w, *ln1_b, *qkv_w, *qkv_b, *o_w, *o_b, *ln2_w, *ln2_b, *up_w, *up_b, *down_w, *down_b;
} VitL;

struct VitW {
  VMmH h;
  int E, H, D, F, L, P, M, PD, G;  // width, heads, head dim, ffn, layers, patch, merge, proj dim, pos grid side
  float *pe0, *pe1, *pe_b, *pos;    // patch embed [E][3*P*P] x2, bias [E], pos table [G*G][E]
  VitL *l;
  float *post_w, *post_b, *mm0_w, *mm0_b, *mm2_w, *mm2_b;
};

static float r16(float x) { return (float)(_Float16)x; }

static void free_l(VitL *l) {
  free(l->ln1_w), free(l->ln1_b), free(l->qkv_w), free(l->qkv_b), free(l->o_w), free(l->o_b);
  free(l->ln2_w), free(l->ln2_b), free(l->up_w), free(l->up_b), free(l->down_w), free(l->down_b);
}

void vit_free(VitW *w) {
  if (!w) return;
  if (w->l) for (int i = 0; i < w->L; i++) free_l(&w->l[i]);
  free(w->l), free(w->pe0), free(w->pe1), free(w->pe_b), free(w->pos);
  free(w->post_w), free(w->post_b), free(w->mm0_w), free(w->mm0_b), free(w->mm2_w), free(w->mm2_b);
  free(w);
}

VitW *vit_load(const char *mmproj, uint32_t n_embd_llm, char *err) {
  VMm *m = vmm_open(mmproj, n_embd_llm, err);
  if (!m) return NULL;
  VitW *w = calloc(1, sizeof *w);
  vmm_hparams(m, &w->h);
  w->E = (int)w->h.n_embd, w->H = (int)w->h.n_head, w->D = w->E / w->H, w->F = (int)w->h.n_ff, w->L = (int)w->h.n_layer;
  w->P = (int)w->h.patch, w->M = (int)w->h.n_merge, w->PD = (int)w->h.proj_dim, w->G = (int)(w->h.image_size / w->h.patch);
  if (w->D % 4 || w->M != 2) { snprintf(err, 256, "unsupported head dim %d / merge %d", w->D, w->M); vmm_close(m); vit_free(w); return NULL; }
  const size_t E = w->E, F = w->F, K = 3 * (size_t)w->P * w->P;
  int bad = 0;
#define LD(dst, name, n) do { if (!bad && !((dst) = vmm_f32(m, name, n, err))) bad = 1; } while (0)
  LD(w->pe0, "v.patch_embd.weight", E * K);
  LD(w->pe1, "v.patch_embd.weight.1", E * K);
  LD(w->pe_b, "v.patch_embd.bias", E);
  LD(w->pos, "v.position_embd.weight", E * w->G * w->G);
  w->l = calloc((size_t)w->L, sizeof *w->l);
  char nm[96];
  for (int i = 0; i < w->L && !bad; i++) {
    VitL *l = &w->l[i];
#define LDL(f, suffix, n) do { snprintf(nm, sizeof nm, "v.blk.%d.%s", i, suffix); LD(l->f, nm, n); } while (0)
    LDL(ln1_w, "ln1.weight", E); LDL(ln1_b, "ln1.bias", E);
    LDL(qkv_w, "attn_qkv.weight", 3 * E * E); LDL(qkv_b, "attn_qkv.bias", 3 * E);
    LDL(o_w, "attn_out.weight", E * E); LDL(o_b, "attn_out.bias", E);
    LDL(ln2_w, "ln2.weight", E); LDL(ln2_b, "ln2.bias", E);
    LDL(up_w, "ffn_up.weight", F * E); LDL(up_b, "ffn_up.bias", F);
    LDL(down_w, "ffn_down.weight", E * F); LDL(down_b, "ffn_down.bias", E);
#undef LDL
  }
  LD(w->post_w, "v.post_ln.weight", E);
  LD(w->post_b, "v.post_ln.bias", E);
  LD(w->mm0_w, "mm.0.weight", 16 * E * E);
  LD(w->mm0_b, "mm.0.bias", 4 * E);
  LD(w->mm2_w, "mm.2.weight", 4 * E * (size_t)w->PD);
  LD(w->mm2_b, "mm.2.bias", (size_t)w->PD);
#undef LD
  vmm_close(m);
  if (bad) { vit_free(w); return NULL; }
  return w;
}

int vit_n_tokens(const VitW *w, int nx, int ny) { return (nx / (w->P * w->M)) * (ny / (w->P * w->M)); }
int vit_proj_dim(const VitW *w) { return w->PD; }

// ---------------------------------------------------------------- ops (each = one future Metal kernel)

typedef struct { int threads; dispatch_queue_t q; } Par;

// rows [0,n) in ~4*threads chunks
#define PAR_ROWS(par, n, ...)                                                              \
  do {                                                                                     \
    const size_t n_ = (size_t)(n), ch_ = (n_ + 4 * (par)->threads - 1) / (4 * (par)->threads); \
    dispatch_apply((n_ + ch_ - 1) / ch_, (par)->q, ^(size_t c_) {                          \
      const size_t r0 = c_ * ch_, r1 = r0 + ch_ < n_ ? r0 + ch_ : n_;                      \
      for (size_t r = r0; r < r1; r++) { __VA_ARGS__; }                                    \
    });                                                                                    \
  } while (0)

// y[n][o] = round?(x)[n][k] . W[o][k]^T (+ b[o])
static void gemm(const Par *par, int f16act, const float *x, size_t n, size_t k, const float *W, size_t o, const float *b,
                 float *y, float *scratch) {
  const float *xa = x;
  if (f16act) {
    PAR_ROWS(par, n, for (size_t i = 0; i < k; i++) scratch[r * k + i] = r16(x[r * k + i]));
    xa = scratch;
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)n, (int)o, (int)k, 1.0f, xa, (int)k, W, (int)k, 0.0f, y, (int)o);
  if (b) PAR_ROWS(par, n, for (size_t i = 0; i < o; i++) y[r * o + i] += b[i]);
}

// ggml_norm (eps) then * w + b; mean/variance accumulated in double
static void layernorm(const Par *par, const float *x, size_t n, size_t e, float eps, const float *w, const float *b, float *y) {
  PAR_ROWS(par, n, {
    const float *xr = x + r * e;
    float *yr = y + r * e;
    double s = 0;
    for (size_t i = 0; i < e; i++) s += xr[i];
    const float mean = (float)(s / (double)e);
    double v = 0;
    for (size_t i = 0; i < e; i++) { const float d = xr[i] - mean; v += (double)d * d; }
    const float scale = 1.0f / sqrtf((float)(v / (double)e) + eps);
    for (size_t i = 0; i < e; i++) yr[i] = ((xr[i] - mean) * scale) * w[i] + b[i];
  });
}

static void gelu_tanh(const Par *par, float *x, size_t n, size_t e) {
  PAR_ROWS(par, n, {
    float *xr = x + r * e;
    for (size_t i = 0; i < e; i++) {
      const float v = xr[i];
      xr[i] = 0.5f * v * (1.0f + tanhf(SQRT_2_OVER_PI * v * (1.0f + GELU_COEF_A * v * v)));
    }
  });
}

static void add(const Par *par, float *y, const float *a, size_t n, size_t e) {
  PAR_ROWS(par, n, for (size_t i = 0; i < e; i++) y[r * e + i] += a[r * e + i]);
}

// GGML_ROPE_TYPE_VISION, n_dims = D/2, sections {D/4 x4}: pair (i, i+D/2); i < D/4 -> row pos, else column pos
static void rope_tables(int D, const int *py, const int *px, size_t n, float *cs, float *sn) {
  const int nd = D / 2, s0 = D / 4;
  const float inv_ndims = -1.f / nd;
  for (size_t r = 0; r < n; r++)
    for (int ic = 0; ic < nd; ic++) {
      const float p = ic < s0 ? (float)ic : (float)(ic - s0);
      const float base = ic < s0 ? (float)py[r] : (float)px[r];
      const float theta = base * powf(10000.0f, 2.0f * inv_ndims * p);
      cs[r * nd + ic] = cosf(theta), sn[r * nd + ic] = sinf(theta);
    }
}

// rotate the Q (off 0) or K (off E) heads inside qkv rows [n][3E]
static void rope_apply(const Par *par, float *qkv, size_t n, int E, int H, int D, size_t off, const float *cs, const float *sn) {
  const int nd = D / 2;
  PAR_ROWS(par, n, {
    for (int h = 0; h < H; h++) {
      float *v = qkv + r * 3 * (size_t)E + off + (size_t)h * D;
      for (int ic = 0; ic < nd; ic++) {
        const float c = cs[r * nd + ic], s = sn[r * nd + ic], x0 = v[ic], x1 = v[ic + nd];
        v[ic] = x0 * c - x1 * s;
        v[ic + nd] = x0 * s + x1 * c;
      }
    }
  });
}

// bidirectional MHA over all n patches: out[n][H*D]; q/k/v are rows of qkv [n][3E] (K, V optionally F16-rounded)
static void attention(const Par *par, int f16act, const float *qkv, size_t n, int E, int H, int D, float *out, float *kv16) {
  const float scale = 1.0f / sqrtf((float)D);
  const size_t ld = 3 * (size_t)E;
  const float *kb = qkv + E, *vb = qkv + 2 * (size_t)E;
  size_t ldk = ld;
  if (f16act) {  // kv16 = [n][2E]: K | V rounded
    PAR_ROWS(par, n, for (size_t i = 0; i < 2 * (size_t)E; i++) kv16[r * 2 * E + i] = r16(qkv[r * ld + E + i]));
    kb = kv16, vb = kv16 + E, ldk = 2 * (size_t)E;
  }
  const size_t B = 256, nb = (n + B - 1) / B;
  openblas_set_num_threads(1);
  dispatch_apply((size_t)H * nb, par->q, ^(size_t job) {
    const size_t h = job / nb, q0 = (job % nb) * B, bq = q0 + B < n ? B : n - q0;
    float *S = malloc(sizeof(float) * bq * n);
    // S = Q_h[q0:q0+bq] . K_h^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)bq, (int)n, D, 1.0f, qkv + q0 * ld + h * D, (int)ld,
                kb + h * D, (int)ldk, 0.0f, S, (int)n);
    for (size_t i = 0; i < bq; i++) {  // softmax(scale * S) in F32
      float *s = S + i * n, mx = -INFINITY;
      for (size_t j = 0; j < n; j++) { s[j] *= scale; if (s[j] > mx) mx = s[j]; }
      double sum = 0;
      for (size_t j = 0; j < n; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
      const float inv = (float)(1.0 / sum);
      for (size_t j = 0; j < n; j++) s[j] *= inv;
    }
    // O = P . V_h
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)bq, D, (int)n, 1.0f, S, (int)n, vb + h * D, (int)ldk,
                0.0f, out + q0 * E + h * D, E);
    free(S);
  });
  openblas_set_num_threads(par->threads);
}

// ggml_interpolate(BILINEAR | ALIGN_CORNERS) of the G x G table to px x py, exactly Metal's kernel_upscale_bilinear_f32
// (no antialias): out[(y*px + x)][E]
static void pos_resize(const float *tab, int G, int E, int px, int py, float *out) {
  if (px == G && py == G) { memcpy(out, tab, sizeof(float) * (size_t)G * G * E); return; }
  const float sf0 = px > 1 && G > 1 ? (float)(px - 1) / (G - 1) : (float)px / G;
  const float sf1 = py > 1 && G > 1 ? (float)(py - 1) / (G - 1) : (float)py / G;
  for (int y = 0; y < py; y++) {
    const float f01 = (float)y / sf1;
    long i01 = (long)floorf(f01);
    if (i01 < 0) i01 = 0;
    if (i01 > G - 1) i01 = G - 1;
    const long i01p = i01 + 1 > G - 1 ? G - 1 : i01 + 1;
    float fd1 = f01 - (float)i01;
    fd1 = fd1 < 0 ? 0 : fd1 > 1 ? 1 : fd1;
    for (int x = 0; x < px; x++) {
      const float f00 = (float)x / sf0;
      long i00 = (long)floorf(f00);
      if (i00 < 0) i00 = 0;
      if (i00 > G - 1) i00 = G - 1;
      const long i00p = i00 + 1 > G - 1 ? G - 1 : i00 + 1;
      float fd0 = f00 - (float)i00;
      fd0 = fd0 < 0 ? 0 : fd0 > 1 ? 1 : fd0;
      const float *s00 = tab + ((size_t)i01 * G + i00) * E, *s10 = tab + ((size_t)i01 * G + i00p) * E;
      const float *s01 = tab + ((size_t)i01p * G + i00) * E, *s11 = tab + ((size_t)i01p * G + i00p) * E;
      float *o = out + ((size_t)y * px + x) * E;
      for (int e = 0; e < E; e++)
        o[e] = s00[e] * (1.0f - fd0) * (1.0f - fd1) + s10[e] * fd0 * (1.0f - fd1) + s01[e] * (1.0f - fd0) * fd1 +
               s11[e] * fd0 * fd1;
    }
  }
}

// ---------------------------------------------------------------- the tower

static void dump(const VitOpt *o, const char *name, const float *x, int ne0, size_t rows) {
  if (o && o->dump && (!o->want || o->want(o->dump_ud, name))) o->dump(o->dump_ud, name, x, ne0, (int)rows);
}
static int wants(const VitOpt *o, const char *name) { return o && o->dump && (!o->want || o->want(o->dump_ud, name)); }

int vit_encode(const VitW *w, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err) {
  const int E = w->E, H = w->H, D = w->D, F = w->F, P = w->P, M = w->M, PD = w->PD, f16 = o ? o->f16act : 1;
  if (nx % (P * M) || ny % (P * M)) { snprintf(err, 256, "image %dx%d not a multiple of %d", nx, ny, P * M); return -1; }
  const int px = nx / P, py = ny / P;
  const size_t N = (size_t)px * py, K = 3 * (size_t)P * P;
  Par par = {o && o->threads > 0 ? o->threads : 20, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0)};
  openblas_set_num_threads(par.threads);
  const size_t big = N * (size_t)(F > 4 * E ? F : 4 * E);
  float *x = malloc(sizeof(float) * N * E), *cur = malloc(sizeof(float) * N * E), *t2 = malloc(sizeof(float) * N * E);
  float *qkv = malloc(sizeof(float) * N * 3 * E), *up = malloc(sizeof(float) * big), *scr = malloc(sizeof(float) * big);
  float *kv16 = malloc(sizeof(float) * N * 2 * E), *rows = malloc(sizeof(float) * N * 2 * K);
  float *cs = malloc(sizeof(float) * N * (D / 2)), *sn = malloc(sizeof(float) * N * (D / 2));
  int *pyi = malloc(sizeof(int) * N), *pxi = malloc(sizeof(int) * N);
  if (!x || !cur || !t2 || !qkv || !up || !scr || !kv16 || !rows || !cs || !sn || !pyi || !pxi) { snprintf(err, 256, "out of memory"); return -1; }
  char nm[64];

  // patch embed: im2col (2x2-merge row order, temporal duplicate) -> conv(W0) + conv(W1) -> + bias
  {
    size_t r = 0;
    for (int y = 0; y < py; y += M)
      for (int xx = 0; xx < px; xx += M)
        for (int dy = 0; dy < M; dy++)
          for (int dx = 0; dx < M; dx++, r++) {
            float *row = rows + r * K;  // one frame; frame 1 is the same values
            const int y0 = (y + dy) * P, x0 = (xx + dx) * P;
            pyi[r] = y + dy, pxi[r] = xx + dx;
            for (int ch = 0; ch < 3; ch++)
              for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++) {
                  const float v = planar[(size_t)ch * nx * ny + (size_t)(y0 + ky) * nx + x0 + kx];
                  row[((size_t)ch * P + ky) * P + kx] = f16 ? r16(v) : v;  // ggml im2col output type = kernel type (F16)
                }
          }
    gemm(&par, 0, rows, N, K, w->pe0, E, NULL, x, NULL);
    gemm(&par, 0, rows, N, K, w->pe1, E, NULL, cur, NULL);
    add(&par, x, cur, N, E);
    PAR_ROWS(&par, N, for (int i = 0; i < E; i++) x[r * E + i] += w->pe_b[i]);
    dump(o, "patch_bias", x, E, N);
  }
  // + learned pos embed, bilinear (align corners) to px x py, then the same 2x2-merge order
  {
    float *pr = malloc(sizeof(float) * N * E);
    pos_resize(w->pos, w->G, E, px, py, pr);
    PAR_ROWS(&par, N, { const float *s = pr + ((size_t)pyi[r] * px + pxi[r]) * E; for (int i = 0; i < E; i++) x[r * E + i] += s[i]; });
    free(pr);
    dump(o, "inp_pos_emb", x, E, N);
  }
  rope_tables(D, pyi, pxi, N, cs, sn);

  for (int il = 0; il < w->L; il++) {
    const VitL *l = &w->l[il];
    layernorm(&par, x, N, E, w->h.eps, l->ln1_w, l->ln1_b, cur);
    snprintf(nm, sizeof nm, "ln1-%d", il); dump(o, nm, cur, E, N);
    gemm(&par, f16, cur, N, E, l->qkv_w, 3 * (size_t)E, l->qkv_b, qkv, scr);
    rope_apply(&par, qkv, N, E, H, D, 0, cs, sn);
    rope_apply(&par, qkv, N, E, H, D, E, cs, sn);
    snprintf(nm, sizeof nm, "Qcur_rope-%d", il);
    if (wants(o, nm)) { PAR_ROWS(&par, N, memcpy(t2 + r * E, qkv + r * 3 * E, sizeof(float) * E)); dump(o, nm, t2, E, N); }
    snprintf(nm, sizeof nm, "Kcur_rope-%d", il);
    if (wants(o, nm)) { PAR_ROWS(&par, N, memcpy(t2 + r * E, qkv + r * 3 * E + E, sizeof(float) * E)); dump(o, nm, t2, E, N); }
    attention(&par, f16, qkv, N, E, H, D, t2, kv16);
    snprintf(nm, sizeof nm, "kqv_out-%d", il); dump(o, nm, t2, E, N);
    gemm(&par, f16, t2, N, E, l->o_w, E, l->o_b, cur, scr);
    snprintf(nm, sizeof nm, "attn_out-%d", il); dump(o, nm, cur, E, N);
    add(&par, x, cur, N, E);
    snprintf(nm, sizeof nm, "ffn_inp-%d", il); dump(o, nm, x, E, N);
    layernorm(&par, x, N, E, w->h.eps, l->ln2_w, l->ln2_b, cur);
    snprintf(nm, sizeof nm, "ffn_inp_normed-%d", il); dump(o, nm, cur, E, N);
    gemm(&par, f16, cur, N, E, l->up_w, F, l->up_b, up, scr);
    snprintf(nm, sizeof nm, "ffn_up_b-%d", il); dump(o, nm, up, F, N);
    gelu_tanh(&par, up, N, F);
    snprintf(nm, sizeof nm, "ffn_gelu-%d", il); dump(o, nm, up, F, N);
    gemm(&par, f16, up, N, F, l->down_w, E, l->down_b, cur, scr);
    snprintf(nm, sizeof nm, "ffn_out-%d", il); dump(o, nm, cur, E, N);
    add(&par, x, cur, N, E);
    snprintf(nm, sizeof nm, "layer_out-%d", il); dump(o, nm, x, E, N);
  }
  layernorm(&par, x, N, E, w->h.eps, w->post_w, w->post_b, cur);
  snprintf(nm, sizeof nm, "norm_b-%d", w->L); dump(o, nm, cur, E, N);
  // merger: 4 consecutive (merge-ordered) patches -> one 4E row -> mm.0 + b -> GELU(tanh) -> mm.2 + b
  const size_t T = N / 4;
  gemm(&par, f16, cur, T, 4 * (size_t)E, w->mm0_w, 4 * (size_t)E, w->mm0_b, up, scr);
  dump(o, "ffn_up_b", up, 4 * E, T);
  gelu_tanh(&par, up, T, 4 * (size_t)E);
  dump(o, "ffn_gelu", up, 4 * E, T);
  gemm(&par, f16, up, T, 4 * (size_t)E, w->mm2_w, PD, w->mm2_b, embd, scr);
  dump(o, "embd", embd, PD, T);
  free(x), free(cur), free(t2), free(qkv), free(up), free(scr), free(kv16), free(rows), free(cs), free(sn), free(pyi), free(pxi);
  return 0;
}
