// vmm.c: a non-static face on the mmproj section of host/llb_gguf.c (patches/engine-mmproj.patch), so host tools
// can link the engine's own loader. Compile with -I<patched tree>/host (vision/v1/run-v1.sh materializes it).
#define GG_MM_VISION_BF16 1
#include "llb_gguf.c"
#include "vmm.h"

struct VMm { GgMm m; };

VMm *vmm_open(const char *path, uint32_t n_embd_llm, char *err) {
  VMm *v = calloc(1, sizeof *v);
  if (!v) { snprintf(err, 256, "out of memory"); return NULL; }
  if (gg_mm_open(&v->m, path, err) || gg_mm_check(&v->m, n_embd_llm, err)) { gg_mm_close(&v->m); free(v); return NULL; }
  return v;
}

void vmm_hparams(const VMm *v, VMmH *h) {
  const GgMm *m = &v->m;
  h->n_embd = m->n_embd, h->n_ff = m->n_ff, h->n_layer = m->n_layer, h->n_head = m->n_head, h->proj_dim = m->proj_dim;
  h->image_size = m->image_size, h->patch = m->patch, h->n_merge = m->n_merge ? m->n_merge : 2;
  h->min_px = m->min_px, h->max_px = m->max_px, h->eps = m->eps;
  for (int i = 0; i < 3; i++) h->mean[i] = m->mean[i], h->std[i] = m->std[i];
}

float *vmm_f32(const VMm *v, const char *name, size_t want_n, char *err) {
  const GgMmT *t = gg_mm_find(&v->m, name);
  if (!t) { snprintf(err, 256, "no tensor %s", name); return NULL; }
  size_t n = (size_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
  if (want_n && n != want_n) { snprintf(err, 256, "%s: %zu elements, want %zu", name, n, want_n); return NULL; }
  float *d = malloc(n * sizeof *d);
  if (!d || gg_mm_f32(&v->m, t, d, err)) { free(d); return NULL; }
  return d;
}

// (ENGINE-VISION3 V7) the tensor's bytes as stored (F16 stays F16, for the Metal tower): size/type, then the read
size_t vmm_bytes(const VMm *v, const char *name, int *f16, size_t *n) {
  const GgMmT *t = gg_mm_find(&v->m, name);
  if (!t) return 0;
  if (f16) *f16 = (int)t->ty;
  if (n) *n = (size_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
  return (size_t)t->bytes;
}

int vmm_raw(const VMm *v, const char *name, void *dst, char *err) {
  const GgMmT *t = gg_mm_find(&v->m, name);
  if (!t) { snprintf(err, 256, "no tensor %s", name); return -1; }
  return gg_mm_raw(&v->m, t, dst, err);
}

void vmm_close(VMm *v) { if (v) { gg_mm_close(&v->m); free(v); } }
