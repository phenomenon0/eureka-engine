// No-CUDA stand-ins for what host/llb_host.c uses from the emitted runtime (tools/step2a_neutral_check.sh). Syntax only: values are arbitrary.
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
typedef uint32_t u32; typedef uint64_t u64; typedef int32_t i32;
typedef u64 llb_dptr; typedef struct llb_kernel_* llb_kernel;
bool llb_launch_stub(llb_kernel, u32, u32, u32, void**);
#define LLB_LAUNCH(k, gx, gy, bx, ...) llb_launch_stub((k), (gx), (gy), (bx), (void*[]){ __VA_ARGS__ })
#define LLB_LAUNCH_SM(k, gx, gy, bx, sm, ...) llb_launch_stub((k), (gx), (gy), (bx) + 0 * (sm), (void*[]){ __VA_ARGS__ })
llb_dptr llb_alloc_zero(size_t);
bool llb_zero(llb_dptr, size_t);
bool llb_read(void*, llb_dptr, size_t);
bool llb_copy(llb_dptr, llb_dptr, size_t);
bool llb_write(llb_dptr, const void*, size_t);
#define llb_dev(p) ((llb_dptr)(uintptr_t)(p))
#define LINE 512
#define WARP_MAX (LINE - 1)
typedef struct { u32 d, f, nh, nkv, hd, lkh, lvh, sd; } LlbShape;
typedef struct { u64 w, x, y, k, v, wn, r, qk; u32 kind, n, d, rb, c, fm, gm;
  u32 bq, bg, bb, ba, bsa, bdt, bcv, bsn, bso, ban, brg, bro; } WarpOp;
typedef struct Mail { u32 front, err, done, m, bad; u64 pw[64]; u64 tbk, wlk, skk; u32 pt[64][20]; LlbShape sh; WarpOp op[WARP_MAX]; } Mail;
_Noreturn void err_fail(const char*);
static llb_kernel gpu_pso;
static llb_kernel gpu_warp;
static llb_kernel gpu_warpq;
static llb_kernel gpu_warph;
static llb_kernel gpu_warpk;
static llb_kernel gpu_warp6;
static llb_kernel gpu_warp5;
static llb_kernel gpu_warpke;
static llb_kernel gpu_warp6e;
static llb_kernel gpu_warpkba;
static llb_kernel gpu_q8x;
static llb_kernel gpu_q4dpgu;
static llb_kernel gpu_q4dpre;
static llb_kernel gpu_embedk;
static llb_kernel gpu_warp3;
static llb_kernel gpu_warp2;
static llb_kernel gpu_warpq2;
static llb_kernel gpu_warp4;
static llb_kernel gpu_attn;
static llb_kernel gpu_rms;
static llb_kernel gpu_mailk;
static llb_kernel gpu_dlv;
static llb_kernel gpu_add;
static llb_kernel gpu_rope;
static llb_kernel gpu_silu;
static llb_kernel gpu_qknorm;
static llb_kernel gpu_l2n;
static llb_kernel gpu_conv;
static llb_kernel gpu_grn;
static llb_kernel gpu_gdl;
static llb_kernel gpu_rnorm;
static llb_kernel gpu_vadd;
static llb_kernel gpu_vmul;
static llb_kernel gpu_og;
static llb_kernel gpu_qkr;
static llb_kernel gpu_slot;
static llb_kernel gpu_amax;
static llb_kernel gpu_amaxp;
static llb_kernel gpu_embed;
static llb_kernel gpu_attn4;
static llb_kernel gpu_sgu[9], gpu_sre[9], gpu_sq8[9], gpu_sq6[9], gpu_sq4s, gpu_sgdl;
static llb_kernel gpu_mid;
static llb_kernel gpu_rnadd, gpu_l2qk, gpu_midconv, gpu_warpqr, gpu_warpqt, gpu_warpqc, gpu_warpqc11;
static llb_kernel gpu_siluq, gpu_rnaddq, gpu_rnormq;
static llb_kernel gpu_q8x2, gpu_q4dp2gu, gpu_q4dp2re, gpu_q8dp2, gpu_q8dp2c, gpu_q6dp2;
static llb_kernel gpu_pfq8x, gpu_pfq4m, gpu_pfq4mr, gpu_pfq4mg, gpu_pfq8m, gpu_pfq8mr, gpu_pfq6w, gpu_pfq6wr;
static llb_kernel gpu_pfrms, gpu_pfmc, gpu_pfring, gpu_pfl2qk, gpu_pfgdl, gpu_pfgrn, gpu_pfqkr, gpu_pfslot, gpu_pfog, gpu_pfattn, gpu_pfmark;
static llb_dptr gpu_act;
static llb_dptr gpu_blk;
static llb_dptr gpu_q8b;
static llb_dptr gpu_q8c;
static llb_dptr gpu_logit;
static llb_dptr gpu_headh;
static llb_dptr gpu_atts;
static llb_dptr gpu_att;
static volatile struct Mail* gpu_mail;
static llb_dptr gpu_maild;
#include <stdlib.h>
#define AMAX_G 128u
#define AMAX_T 256u
typedef u64* Corpus;
