// vitc.h: CUDA-VISION R1, the Qwen3-VL / Qwen3.5 vision tower (qwen3vl_merger, no DeepStack) on CUDA (sm_86).
// Same graph, API shape and dump names as the Mac's vit.h (CPU reference) / vitm.h (Metal), op by op:
//   im2col (F16, 2x2-merge order) -> conv(W0)+conv(W1)+bias -> +pos (bilinear, align corners) ->
//   L x [LN -> QKV+b -> 2D vision rope -> bidirectional MHA -> out+b +res -> LN -> up+b -> GELU(tanh) -> down+b +res]
//   -> post LN -> 4 patches/row -> mm.0+b -> GELU(tanh) -> mm.2+b
// Precision staging = the registered gate mode (f16act): F16 weights, GEMM activations rounded to F16, F32 accumulate;
// attention Q/K/V F16, scores + softmax F32, P rounded to F16 for the PV product (tensor cores), O F32.
#pragma once
#include "vit.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct VitcW VitcW;
VitcW *vitc_load(const char *mmproj, uint32_t n_embd_llm, char *err);
void vitc_free(VitcW *w);
int vitc_n_tokens(const VitcW *w, int nx, int ny);
int vitc_proj_dim(const VitcW *w);
// planar [3][ny][nx] -> embd [n_tokens][proj_dim]. 0 ok. VitOpt.threads ignored; f16act must be 1.
int vitc_encode(VitcW *w, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err);
// (CHAIN-2-1-3 Stage 2, opt-in) nimg images in one pass; embd[i] gets image i's rows, BYTE-IDENTICAL to vitc_encode of
// that image alone (the gate). gemm_seg 1: each GEMM per image at the single encode's shape; 2: those same GEMMs on up
// to 8 side streams (one cuBLAS handle each); 0: one GEMM over all rows (not exact: cuBLAS picks other kernels).
int vitc_encode_batch(VitcW *w, int nimg, const float *const *planar, const int *nx, const int *ny, int gemm_seg,
                      float *const *embd, char *err);
double vitc_gpu_ms(const VitcW *w);  // cudaEvent time of the last encode (H2D planar .. D2H embd)
size_t vitc_weight_bytes(const VitcW *w);
size_t vitc_act_bytes(const VitcW *w);
#ifdef __cplusplus
}
#endif
