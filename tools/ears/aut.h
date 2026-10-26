// aut.h: EARS-G1 cudaud, the Qwen3 AuT audio tower (qwen3a mmproj, bf16) on CUDA. See aut.cu.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AutW AutW;
// called with each intermediate under llama.cpp's qwen3a node name; x is contiguous f32 in ggml order (ne0 fastest)
typedef void (*AutDump)(void *ud, const char *name, const float *x, const int64_t ne[4]);
typedef struct {
  AutDump dump;
  void *ud;
  int (*want)(void *ud, const char *name);  // NULL = all
} AutOpt;
AutW *aut_load(const char *mmproj, char *err);
void aut_free(AutW *w);
int aut_proj_dim(const AutW *w);
int aut_n_tokens(int n_len);  // tokens of one window of n_len (x100) mel frames = n_len/100*13
// one 800-frame window: mel [128][n_len] -> embd [n_tokens][proj_dim]. 0 ok.
int aut_encode(AutW *w, const float *mel, int n_mel, int n_len, const AutOpt *o, float *embd, char *err);
// (CHAIN-2-1-3 Stage 2, opt-in) nwin windows in one pass; embd[i] gets window i's rows, BYTE-IDENTICAL to aut_encode of
// that window alone (the gate). gemm_seg 1: each GEMM per window at the single encode's shape; 2: those same GEMMs on
// up to 8 side streams (one cuBLAS handle each); 0: one GEMM over all rows (not exact: cuBLAS picks other kernels).
int aut_encode_batch(AutW *w, int nwin, const float *const *mel, const int *n_len, int gemm_seg, float *const *embd,
                     char *err);
void aut_mem_info(size_t *free_b, size_t *total_b);
double aut_gpu_ms(const AutW *w);
size_t aut_weight_bytes(const AutW *w);
#ifdef __cplusplus
}
#endif
