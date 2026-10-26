// vit.h: ENGINE-VISION V3, the Qwen3-VL vision tower (qwen3vl_merger, no DeepStack) on the host CPU.
// Graph = llama.cpp tools/mtmd/models/qwen3vl.cpp (fn-run 2690873), op by op:
//   conv(W0,img)+conv(W1,img) (the temporal duplicate) -> 2x2-merge order -> +patch bias -> +bilinear pos-embed ->
//   27 x [LN(w,b) -> QKV+b -> 2D vision rope -> bidirectional MHA (16x72) -> out+b -> +res -> LN -> up+b ->
//   GELU(tanh) -> down+b -> +res] -> post LN -> 4 patches -> mm.0+b -> GELU(tanh) -> mm.2+b
// These are the reference kernels for the engine's Metal port (V7): each op is one function below.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct VitW VitW;

// called with each intermediate under llama.cpp's node name (patch_bias, inp_pos_emb, ln1-L, Qcur_rope-L, ...,
// layer_out-L, norm_b-27, ffn_up_b, ffn_gelu, embd); x is [rows][ne0] contiguous f32
typedef void (*VitDump)(void *ud, const char *name, const float *x, int ne0, int rows);

typedef struct {
  int f16act;        // 1 = the registered gate mode: GEMM inputs, im2col and attention K/V rounded to F16
  int threads;       // BLAS + elementwise threads (0 = all P cores)
  VitDump dump;      // optional
  void *dump_ud;
  int (*want)(void *ud, const char *name);  // optional filter: dump only names it accepts (NULL = all)
} VitOpt;

VitW *vit_load(const char *mmproj, uint32_t n_embd_llm, char *err);
void vit_free(VitW *w);
int vit_n_tokens(const VitW *w, int nx, int ny);
int vit_proj_dim(const VitW *w);
// planar [3][ny][nx] (vpre_image's output) -> embd [n_tokens][proj_dim]. 0 ok.
int vit_encode(const VitW *w, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err);
