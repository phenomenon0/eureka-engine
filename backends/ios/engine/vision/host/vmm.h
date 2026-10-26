// vmm.h: the mmproj loader (engine-mmproj.patch's gg_mm_*) for host tools. See vmm.c.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct VMm VMm;
typedef struct {
  uint32_t n_embd, n_ff, n_layer, n_head, proj_dim, image_size, patch, n_merge, min_px, max_px;
  float eps, mean[3], std[3];
} VMmH;

VMm *vmm_open(const char *path, uint32_t n_embd_llm, char *err);  // open + gg_mm_check (refuses all but qwen3vl_merger)
void vmm_hparams(const VMm *v, VMmH *h);
float *vmm_f32(const VMm *v, const char *name, size_t want_n, char *err);  // malloc'd F32 copy (F16 widened exactly)
// (ENGINE-VISION3 V7) raw bytes as stored: vmm_bytes = byte size (0 = no such tensor), *f16 = 1 if F16, *n = elements
size_t vmm_bytes(const VMm *v, const char *name, int *f16, size_t *n);
int vmm_raw(const VMm *v, const char *name, void *dst, char *err);  // 0 ok
void vmm_close(VMm *v);
