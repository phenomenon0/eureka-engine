// vitm.h: ENGINE-VISION3 V7, the Qwen3-VL vision tower on Metal. Same graph, same API shape and same dump names as
// vit.h (the CPU reference, kept). Kernels: vitm.metal (one per op, llama.cpp Metal's precision staging); host: vitm.m.
// Weights stay F16 on the GPU (one shared buffer, ~0.86 GiB); the position-embed resize runs on the host once per image
// size. The rows of one encode are produced by one command buffer (split only where a dump is asked for).
// Only the registered F16-staged mode exists here (VitOpt.f16act must be 1; VitOpt.threads is ignored).
#pragma once
#include "vit.h"

typedef struct VitmW VitmW;

VitmW *vitm_load(const char *mmproj, uint32_t n_embd_llm, char *err);  // V1's gg_mm_check, then the GPU upload
void vitm_free(VitmW *w);
int vitm_n_tokens(const VitmW *w, int nx, int ny);
int vitm_proj_dim(const VitmW *w);
// planar [3][ny][nx] (vpre_image's output) -> embd [n_tokens][proj_dim]. 0 ok.
int vitm_encode(VitmW *w, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err);
double vitm_gpu_ms(const VitmW *w);         // GPU time of the last encode (sum over its command buffers)
size_t vitm_weight_bytes(const VitmW *w);   // the F16/F32 weight buffer
size_t vitm_act_bytes(const VitmW *w);      // the activation buffers currently allocated
