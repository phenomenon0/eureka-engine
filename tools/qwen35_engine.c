// tools/qwen35_engine.c — Qwen3.5-9B hybrid decode engine (CPU twin).
//
// Reads a llama-bend Q8 pack (tools/q8_pack_qwen35.py + tools/pack_index.py):
// mmap pack.bin, index.tsv for offsets/shapes/config. Layer i is full
// attention iff (i+1) % full_attention_interval == 0, else a gated delta net
// (linear attention). Mirrors tools/llb_qwen35_ref.py:
//   rms_norm, l2_norm (no +eps inside sqrt), partial NEOX rope (64/256, base
//   1e7), gated delta recurrence, gated RMSNorm, conv1d+silu, qk-norm, output
//   gate.
//
// All math in F32; transcendentals through double (host C and NVRTC both call
// the double function and round once); reductions in a fixed order, so the CPU
// and CUDA lanes can be made bit-identical.
//
// usage: qwen35_engine <packdir> <tok>... [--steps N] [--layers L]
// stdout: generated ids, one per line, then `wall_ms <ms>` and `tok_s <x>`.
// stderr: `logits <pos> <id1> <bits1> <id2> <bits2>` and `step <pos> <ms>`.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llb_platform.h"
#include "../host/llb_formats.h"
#include "qwen35_tokenizer.c"
#include "qwen35_template.c"
#include "qwen35_sampler.c"

typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD static inline
#define DEVI static __host__ __device__
#else
#define HD static inline
#define DEVI static inline
#endif

// ---- transcendentals through double -------------------------------------
HD float hf_exp(float x) { return (float)exp((double)x); }
HD float hf_log1p(float x) { return (float)log1p((double)x); }
HD float hf_sqrt(float x) { return (float)sqrt((double)x); }
HD float hf_sigmoid(float x) { return 1.0f / (1.0f + hf_exp(-x)); }
HD float hf_silu(float x) { return x / (1.0f + hf_exp(-x)); }
HD float hf_softplus(float x) {
  return x > 20.0f ? x : hf_log1p(hf_exp(x < 88.0f ? x : 88.0f));
}
HD float hf_pow(float b, float e) { return (float)pow((double)b, (double)e); }
HD float hf_cos(float x) { return (float)cos((double)x); }
HD float hf_sin(float x) { return (float)sin((double)x); }

// ---- quant doors (fm tags; storage is llama.cpp's bytes, verbatim) --------
// 0 F32 | 1 Q8_0 row-split | 3 Q4_K | 4 Q6_K | 5 Q3_K | 6 Q2_0 | 7 PQ2_0 | 8 Q5_K
DEVI float half_to_float(uint16_t h);
DEVI float h2f(const uint8_t *p) {
  return half_to_float((uint16_t)(p[0] | (p[1] << 8)));
}

// ggml get_scale_min_k4: 6-bit scale/min from the 12 packed bytes of Q4_K/Q5_K
DEVI void gsm4(int j, const uint8_t *q, int *sd, int *sm) {
  if (j < 4) { *sd = q[j] & 63; *sm = q[j + 4] & 63; }
  else {
    *sd = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *sm = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

DEVI float dot_q4k(const uint8_t *row, int cols, const float *x) {
  float acc = 0.0f;
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk = row + (size_t)b * 144;
    float d = h2f(blk), dmin = h2f(blk + 2);
    const uint8_t *sc = blk + 4, *qs = blk + 16;
    for (int g = 0; g < 4; g++) {
      int sd, sm;
      gsm4(2 * g, sc, &sd, &sm);
      float d1 = d * (float)sd, m1 = dmin * (float)sm;
      gsm4(2 * g + 1, sc, &sd, &sm);
      float d2 = d * (float)sd, m2 = dmin * (float)sm;
      const uint8_t *q = qs + g * 32;
      const float *xg = x + b * 256 + g * 64;
      for (int l = 0; l < 32; l++) acc += (d1 * (float)(q[l] & 0xF) - m1) * xg[l];
      for (int l = 0; l < 32; l++) acc += (d2 * (float)(q[l] >> 4) - m2) * xg[32 + l];
    }
  }
  return acc;
}

DEVI float dot_q5k(const uint8_t *row, int cols, const float *x) {
  float acc = 0.0f;
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk = row + (size_t)b * 176;
    float d = h2f(blk), dmin = h2f(blk + 2);
    const uint8_t *sc = blk + 4, *qh = blk + 16, *qs = blk + 48;
    for (int g = 0; g < 4; g++) {
      int sd, sm;
      gsm4(2 * g, sc, &sd, &sm);
      float d1 = d * (float)sd, m1 = dmin * (float)sm;
      gsm4(2 * g + 1, sc, &sd, &sm);
      float d2 = d * (float)sd, m2 = dmin * (float)sm;
      const uint8_t *q = qs + g * 32;
      const float *xg = x + b * 256 + g * 64;
      for (int l = 0; l < 32; l++)
        acc += (d1 * (float)((q[l] & 0xF) + ((qh[l] >> (2 * g)) & 1) * 16) - m1) * xg[l];
      for (int l = 0; l < 32; l++)
        acc += (d2 * (float)((q[l] >> 4) + ((qh[l] >> (2 * g + 1)) & 1) * 16) - m2) * xg[32 + l];
    }
  }
  return acc;
}

DEVI float dot_q6k(const uint8_t *row, int cols, const float *x) {
  float acc = 0.0f;
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk = row + (size_t)b * 210;
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t *scb = (const int8_t *)(blk + 192);
    float d = h2f(blk + 208);
    const float *xb = x + b * 256;
    for (int half = 0; half < 2; half++) {
      const uint8_t *qlh = ql + half * 64, *qhh = qh + half * 32;
      const int8_t *sch = scb + half * 8;
      for (int qd = 0; qd < 4; qd++) {
        const uint8_t *qq = qlh + (qd & 1) * 32;
        const float *xq = xb + half * 128 + qd * 32;
        for (int l = 0; l < 32; l++) {
          int nib = (qd < 2) ? (qq[l] & 0xF) : (qq[l] >> 4);
          int code = (nib | (((qhh[l] >> (2 * qd)) & 3) << 4)) - 32;
          float dl = d * (float)sch[2 * qd + (l >> 4)];
          acc += (dl * (float)code) * xq[l];
        }
      }
    }
  }
  return acc;
}

// ggml's 12 scale bytes -> 16 int8 scales (Q3_K)
DEVI void q3k_scales(const uint8_t *s, int8_t *out) {
  u32 a0 = (u32)s[0] | ((u32)s[1] << 8) | ((u32)s[2] << 16) | ((u32)s[3] << 24);
  u32 a1 = (u32)s[4] | ((u32)s[5] << 8) | ((u32)s[6] << 16) | ((u32)s[7] << 24);
  u32 a2 = (u32)s[8] | ((u32)s[9] << 8) | ((u32)s[10] << 16) | ((u32)s[11] << 24);
  const u32 k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
  u32 b0 = (a0 & k2) | (((a2 >> 0) & k1) << 4);
  u32 b1 = (a1 & k2) | (((a2 >> 2) & k1) << 4);
  u32 b2 = ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4);
  u32 b3 = ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4);
  u32 bb[4] = {b0, b1, b2, b3};
  for (int w = 0; w < 4; w++)
    for (int j = 0; j < 4; j++) out[w * 4 + j] = (int8_t)((bb[w] >> (8 * j)) & 0xFF);
}

DEVI float dot_q3k(const uint8_t *row, int cols, const float *x) {
  float acc = 0.0f;
  for (int b = 0; b < cols / 256; b++) {
    const uint8_t *blk = row + (size_t)b * 110;
    const uint8_t *hm = blk, *qs = blk + 32;
    float d = h2f(blk + 108);
    int8_t s16[16];
    q3k_scales(blk + 96, s16);
    const float *xb = x + b * 256;
    for (int n = 0; n < 2; n++)
      for (int j = 0; j < 4; j++)
        for (int w16 = 0; w16 < 2; w16++) {
          float dl = d * (float)(s16[n * 8 + j * 2 + w16] - 32);
          const float *xj = xb + n * 128 + j * 32 + w16 * 16;
          for (int lb = 0; lb < 16; lb++) {
            int bi = n * 32 + lb + 16 * w16, hbi = lb + 16 * w16;
            int code = ((qs[bi] >> (2 * j)) & 3) - (((hm[hbi] >> (j + 4 * n)) & 1) ? 0 : 4);
            acc += (dl * (float)code) * xj[lb];
          }
        }
  }
  return acc;
}

// Q2_0 (qk=64) and PQ2_0 (qk=128): w = (q-1)*d, 2 bits per weight
DEVI float dot_q20(const uint8_t *row, int cols, const float *x, int qk) {
  float acc = 0.0f;
  int bb = 2 + qk / 4;
  for (int b = 0; b < cols / qk; b++) {
    const uint8_t *blk = row + (size_t)b * bb;
    float d = h2f(blk);
    const uint8_t *qs = blk + 2;
    const float *xb = x + b * qk;
    for (int c = 0; c < qk; c++) {
      int q = (qs[c >> 2] >> ((c & 3) * 2)) & 3;
      acc += (((float)q - 1.0f) * d) * xb[c];
    }
  }
  return acc;
}

DEVI float dot_q8(const uint8_t *row, int cols, const float *x) {
  const i8 *q = (const i8 *)row;
  const uint8_t *d = row + cols;
  float acc = 0.0f;
  for (int b = 0; b < cols / 32; b++) {
    float dd = h2f(d + 2 * b);
    for (int j = 0; j < 32; j++) acc += ((float)q[b * 32 + j] * dd) * x[b * 32 + j];
  }
  return acc;
}

DEVI float dot_f32(const uint8_t *row, int cols, const float *x) {
  const float *M = (const float *)row;
  float acc = 0.0f;
  for (int c = 0; c < cols; c++) acc += M[c] * x[c];
  return acc;
}

DEVI float dot_fmt(int fmt, const uint8_t *row, int cols, const float *x) {
  switch (fmt) {
    case 0: return dot_f32(row, cols, x);
    case 1: return dot_q8(row, cols, x);
    case 3: return dot_q4k(row, cols, x);
    case 4: return dot_q6k(row, cols, x);
    case 5: return dot_q3k(row, cols, x);
    case 6: return dot_q20(row, cols, x, 64);
    case 7: return dot_q20(row, cols, x, 128);
    case LLB_FM_Q5_K: return dot_q5k(row, cols, x);
  }
  return 0.0f;
}

// Dequantize one whole row (used only to fetch a token_embd row).
DEVI void deq_row(int fmt, const uint8_t *row, int cols, float *out) {
  if (fmt == 0) { memcpy(out, row, (size_t)cols * 4); return; }
  if (fmt == 1) {
    const i8 *q = (const i8 *)row;
    const uint8_t *d = row + cols;
    for (int b = 0; b < cols / 32; b++) {
      float dd = h2f(d + 2 * b);
      for (int j = 0; j < 32; j++) out[b * 32 + j] = (float)q[b * 32 + j] * dd;
    }
    return;
  }
  if (fmt == 6 || fmt == 7) {
    int qk = (fmt == 6) ? 64 : 128, bb = 2 + qk / 4;
    for (int b = 0; b < cols / qk; b++) {
      const uint8_t *blk = row + (size_t)b * bb;
      float d = h2f(blk);
      const uint8_t *qs = blk + 2;
      for (int c = 0; c < qk; c++) {
        int q = (qs[c >> 2] >> ((c & 3) * 2)) & 3;
        out[b * qk + c] = ((float)q - 1.0f) * d;
      }
    }
    return;
  }
  for (int b = 0; b < cols / 256; b++) {
    if (fmt == 3) {
      const uint8_t *blk = row + (size_t)b * 144;
      float d = h2f(blk), dmin = h2f(blk + 2);
      const uint8_t *sc = blk + 4, *qs = blk + 16;
      for (int g = 0; g < 4; g++) {
        int sd, sm;
        gsm4(2 * g, sc, &sd, &sm);
        float d1 = d * (float)sd, m1 = dmin * (float)sm;
        gsm4(2 * g + 1, sc, &sd, &sm);
        float d2 = d * (float)sd, m2 = dmin * (float)sm;
        const uint8_t *q = qs + g * 32;
        float *o = out + b * 256 + g * 64;
        for (int l = 0; l < 32; l++) o[l] = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) o[32 + l] = d2 * (float)(q[l] >> 4) - m2;
      }
    } else if (fmt == LLB_FM_Q5_K) {
      const uint8_t *blk = row + (size_t)b * 176;
      float d = h2f(blk), dmin = h2f(blk + 2);
      const uint8_t *sc = blk + 4, *qh = blk + 16, *qs = blk + 48;
      for (int g = 0; g < 4; g++) {
        int sd, sm;
        gsm4(2 * g, sc, &sd, &sm);
        float d1 = d * (float)sd, m1 = dmin * (float)sm;
        gsm4(2 * g + 1, sc, &sd, &sm);
        float d2 = d * (float)sd, m2 = dmin * (float)sm;
        const uint8_t *q = qs + g * 32;
        float *o = out + b * 256 + g * 64;
        for (int l = 0; l < 32; l++)
          o[l] = d1 * (float)((q[l] & 0xF) + ((qh[l] >> (2 * g)) & 1) * 16) - m1;
        for (int l = 0; l < 32; l++)
          o[32 + l] = d2 * (float)((q[l] >> 4) + ((qh[l] >> (2 * g + 1)) & 1) * 16) - m2;
      }
    } else if (fmt == 4) {
      const uint8_t *blk = row + (size_t)b * 210;
      const uint8_t *ql = blk, *qh = blk + 128;
      const int8_t *scb = (const int8_t *)(blk + 192);
      float d = h2f(blk + 208);
      for (int half = 0; half < 2; half++) {
        const uint8_t *qlh = ql + half * 64, *qhh = qh + half * 32;
        const int8_t *sch = scb + half * 8;
        for (int qd = 0; qd < 4; qd++) {
          const uint8_t *qq = qlh + (qd & 1) * 32;
          float *o = out + b * 256 + half * 128 + qd * 32;
          for (int l = 0; l < 32; l++) {
            int nib = (qd < 2) ? (qq[l] & 0xF) : (qq[l] >> 4);
            int code = (nib | (((qhh[l] >> (2 * qd)) & 3) << 4)) - 32;
            o[l] = (d * (float)sch[2 * qd + (l >> 4)]) * (float)code;
          }
        }
      }
    } else if (fmt == 5) {
      const uint8_t *blk = row + (size_t)b * 110;
      const uint8_t *hm = blk, *qs = blk + 32;
      float d = h2f(blk + 108);
      int8_t s16[16];
      q3k_scales(blk + 96, s16);
      for (int n = 0; n < 2; n++)
        for (int j = 0; j < 4; j++)
          for (int w16 = 0; w16 < 2; w16++) {
            float dl = d * (float)(s16[n * 8 + j * 2 + w16] - 32);
            float *o = out + b * 256 + n * 128 + j * 32 + w16 * 16;
            for (int lb = 0; lb < 16; lb++) {
              int bi = n * 32 + lb + 16 * w16, hbi = lb + 16 * w16;
              int code = ((qs[bi] >> (2 * j)) & 3) - (((hm[hbi] >> (j + 4 * n)) & 1) ? 0 : 4);
              o[lb] = dl * (float)code;
            }
          }
    }
  }
}

DEVI float half_to_float(uint16_t h) {
  u32 sign = (u32)(h >> 15) & 1u, exp = (u32)(h >> 10) & 0x1fu, mant = (u32)h & 0x3ffu, f;
  if (exp == 0) {
    if (mant == 0) f = sign << 31;
    else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  union { u32 u; float f; } v;
  v.u = f;
  return v.f;
}

// ---- pack + index --------------------------------------------------------
typedef struct { char name[64]; u64 off; u64 bytes; int rows, cols; int is_f32; int row_bytes; int fmt; } TInfo;
static TInfo gT[1024];
static int gNT = 0;
static const uint8_t *gPack = NULL;
static size_t gMapSz = 0;  // nonzero: gPack is mapped; drop its pages per layer (RSS <= one layer)
static const char *gDumpLayers = NULL;

static int L, D, H, KV, HDIM, dff, V, INTERVAL, KC, SZ, GK, TR, VHD, KDC, VDC, CONVD, ROPEDIM;
static float EPS, ROPE_BASE;

static int fmt_of_type(const char *t) {
#define FM_OF(n, fm, bb, be) if (!strcmp(t, #n)) return fm;
  LLB_FORMATS(FM_OF)
#undef FM_OF
  return -1;
}

static int find(const char *name) {
  for (int i = 0; i < gNT; i++) if (strcmp(gT[i].name, name) == 0) return i;
  return -1;
}
HD const uint8_t *tptr(int id) { return gPack + gT[id].off; }
HD const float *fvec(int id) { return (const float *)tptr(id); }
HD int trows(int id) { return gT[id].rows; }
HD int tcols(int id) { return gT[id].cols; }
HD int tf32(int id) { return gT[id].is_f32; }
HD int trb(int id) { return gT[id].row_bytes; }
HD int tfmt(int id) { return gT[id].fmt; }

static void load_index(const char *dir) {
  char path[4096];
  snprintf(path, sizeof path, "%s/index.tsv", dir);
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char line[1024];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#') {
      if (strncmp(line, "#cfg\t", 5) == 0) {
        char key[64], val[128];
        if (sscanf(line + 5, "%63[^\t]\t%127[^\n]", key, val) == 2) {
          if (!strcmp(key, "n_layers")) L = atoi(val);
          else if (!strcmp(key, "n_heads")) H = atoi(val);
          else if (!strcmp(key, "n_kv_heads")) KV = atoi(val);
          else if (!strcmp(key, "d_model")) D = atoi(val);
          else if (!strcmp(key, "head_dim")) HDIM = atoi(val);
          else if (!strcmp(key, "d_ff")) dff = atoi(val);
          else if (!strcmp(key, "vocab_size")) V = atoi(val);
          else if (!strcmp(key, "full_attention_interval")) INTERVAL = atoi(val);
          else if (!strcmp(key, "ssm_conv_kernel")) KC = atoi(val);
          else if (!strcmp(key, "ssm_state_size")) SZ = atoi(val);
          else if (!strcmp(key, "ssm_group_count")) GK = atoi(val);
          else if (!strcmp(key, "ssm_time_step_rank")) TR = atoi(val);
          else if (!strcmp(key, "ssm_inner_size")) VDC = atoi(val);
          else if (!strcmp(key, "linear_key_dim")) KDC = atoi(val);
          else if (!strcmp(key, "linear_value_head_dim")) VHD = atoi(val);
          else if (!strcmp(key, "linear_conv_dim")) CONVD = atoi(val);
          else if (!strcmp(key, "rope_dim")) ROPEDIM = atoi(val);
          else if (!strcmp(key, "rope_freq_base")) ROPE_BASE = strtof(val, NULL);
          else if (!strcmp(key, "rms_norm_eps")) EPS = strtof(val, NULL);
        }
      }
      continue;
    }
    if (line[0] == '\n' || line[0] == 0) continue;
    TInfo t;
    char type[16];
    int n = sscanf(line, "%63[^\t]\t%lu\t%lu\t%d\t%d\t%15[^\t]\t%d\t%d",
                   t.name, (unsigned long *)&t.off, (unsigned long *)&t.bytes,
                   &t.rows, &t.cols, type, &t.row_bytes, &t.fmt);
    if (n < 6) continue;
    t.is_f32 = strcmp(type, "F32") == 0;
    if (n < 7) t.row_bytes = 0;
    if (n < 8) t.fmt = fmt_of_type(type);  // older index.tsv has no fmt column
    if (t.fmt != fmt_of_type(type)) {  // pre-registry packs wrote Q5_K = 8
      fprintf(stderr, "%s: fmt %d != registry %d for %s (re-run tools/pack_index.py)\n",
              t.name, t.fmt, fmt_of_type(type), type);
      exit(1);
    }
    if (t.fmt < 0) { fprintf(stderr, "unknown tensor type %s for %s\n", type, t.name); exit(1); }
    if (gNT >= (int)(sizeof gT / sizeof gT[0])) { fprintf(stderr, "index: more than %d tensors\n", gNT); exit(1); }
    gT[gNT++] = t;
  }
  fclose(f);
}

// ---- ops -----------------------------------------------------------------

// GPU gemv: one thread per row, sequential over cols — the same order as the
// CPU gemv, so the two lanes are bit-identical (no FMA: -ffp-contract=off /
// -fmad=false). The whole pack is resident on the device.
static int gUseGpu = 0;

#ifdef __METAL__
int metal_init(size_t psz, int maxrows, int maxcols);
void metal_gemv(int tid, const float *x, float *y);
#endif

#ifdef __CUDACC__
__global__ void k_gemv(const uint8_t *W, int rows, int cols, int rb, int fmt,
                       const float *x, float *y) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  y[r] = dot_fmt(fmt, W + (size_t)r * rb, cols, x);
}

static uint8_t *gdPack = NULL;
static float *gdX = NULL, *gdY = NULL;
static void gemv_gpu(int id, const float *x, float *y);

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static void cuda_init(size_t psz, int maxrows, int maxcols) {
  CUDA_CHECK(cudaMalloc(&gdPack, psz));
  CUDA_CHECK(cudaMemcpy(gdPack, gPack, psz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&gdX, (size_t)maxcols * 4));
  CUDA_CHECK(cudaMalloc(&gdY, (size_t)maxrows * 4));
  fprintf(stderr, "gpu: %zu MB weights resident, maxrows %d maxcols %d\n", psz >> 20, maxrows, maxcols);
}
#endif

// y[r] = sum_c W[r,c] x[c]; the row layout is per-format (see dot_fmt).
HD void gemv(int id, const uint8_t *W, const float *x, float *y) {
#ifdef __CUDACC__
  if (gUseGpu) { gemv_gpu(id, x, y); return; }
#endif
#ifdef __METAL__
  if (gUseGpu) { metal_gemv(id, x, y); return; }
#endif
  int rows = trows(id), cols = tcols(id), fmt = tfmt(id);
  int rb = (fmt == 0) ? cols * 4 : trb(id);
  for (int r = 0; r < rows; r++)
    y[r] = dot_fmt(fmt, W + (size_t)r * rb, cols, x);
}

#ifdef __CUDACC__
static void gemv_gpu(int id, const float *x, float *y) {
  int rows = trows(id), cols = tcols(id), fmt = tfmt(id);
  int rb = (fmt == 0) ? cols * 4 : trb(id);
  CUDA_CHECK(cudaMemcpy(gdX, x, (size_t)cols * 4, cudaMemcpyHostToDevice));
  int threads = 256, blocks = (rows + threads - 1) / threads;
  k_gemv<<<blocks, threads>>>(gdPack + gT[id].off, rows, cols, rb, fmt, gdX, gdY);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(y, gdY, (size_t)rows * 4, cudaMemcpyDeviceToHost));
}
#endif

HD void row_f32(int id, int r, float *out) {
  int cols = tcols(id), fmt = tfmt(id);
  int rb = (fmt == 0) ? cols * 4 : trb(id);
  deq_row(fmt, tptr(id) + (size_t)r * rb, cols, out);
}

HD void rms_norm(float *h, const float *x, const float *w, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float s = 1.0f / hf_sqrt(ss / (float)n + EPS);
  for (int i = 0; i < n; i++) h[i] = x[i] * s * w[i];
}

// 1/max(sqrt(sum), eps)  — the l2_norm scale (ggml: no +eps inside)
HD float l2_scale(float sumsq) {
  float s = hf_sqrt(sumsq);
  return 1.0f / (s > EPS ? s : EPS);
}

HD void rope_neox(float *x, int pos, int nd, int hd, float base) {
  int half = nd / 2;
  (void)hd;
  for (int i = 0; i < half; i++) {
    float theta = (float)((double)pos * pow((double)base, -2.0 * i / nd));
    float c = hf_cos(theta), s = hf_sin(theta);
    float a = x[i], b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
  }
}

// ---- model state ---------------------------------------------------------
static float *gS, *gConv, *gK, *gV;
static int gCap = 0;
static int gFull[64];
static int is_full(int i) { return (i + 1) % INTERVAL == 0; }

static void state_alloc(int cap) {
  gCap = cap;
  gS = (float *)calloc((size_t)L * TR * SZ * SZ, 4);
  gConv = (float *)calloc((size_t)L * (KC - 1) * CONVD, 4);
  gK = (float *)calloc((size_t)L * cap * KV * HDIM, 4);
  gV = (float *)calloc((size_t)L * cap * KV * HDIM, 4);
  if (!gS || !gConv || !gK || !gV) { fprintf(stderr, "state alloc failed\n"); exit(1); }
  for (int i = 0; i < L; i++) gFull[i] = is_full(i);
}

// named scratch buffers (no aliasing across an op)
static float *X, *HH, *QKV, *Z, *BETA, *ALPHA, *GATE_PRE, *Y, *CORE;
static float *QGN, *QN, *KN, *KB, *VB, *ATTN, *PROBS, *QT, *KT, *SCR, *OUT0, *LOGITS, *EMBROW;

static void buf_alloc(void) {
  int cap = gCap;
  X = (float *)calloc(D, 4); HH = (float *)calloc(D, 4);
  QKV = (float *)calloc(CONVD, 4); Z = (float *)calloc(VDC, 4);
  BETA = (float *)calloc(TR, 4); ALPHA = (float *)calloc(TR, 4); GATE_PRE = (float *)calloc(TR, 4);
  CORE = (float *)calloc((size_t)TR * SZ, 4);
  Y = (float *)calloc(CONVD > dff ? CONVD : dff, 4);
  QGN = (float *)calloc(2 * HDIM * H, 4); QN = (float *)calloc(H * HDIM, 4);
  KN = (float *)calloc(KV * HDIM, 4); KB = (float *)calloc(KV * HDIM, 4); VB = (float *)calloc(KV * HDIM, 4);
  ATTN = (float *)calloc(H * HDIM, 4); PROBS = (float *)calloc(cap, 4);
  QT = (float *)calloc(TR * SZ, 4); KT = (float *)calloc(TR * SZ, 4);
  SCR = (float *)calloc(dff > D ? dff : D, 4); OUT0 = (float *)calloc(VDC > D ? VDC : D, 4);
  LOGITS = (float *)calloc(V, 4); EMBROW = (float *)calloc(D, 4);
}

// ---- forward -------------------------------------------------------------

static void dl_dump(const char *name, const float *p, int n) {
  if (!gDumpLayers) return;
  char fn[512];
  snprintf(fn, sizeof fn, "%s.%s.f32", gDumpLayers, name);
  FILE *f = fopen(fn, "wb");
  fwrite(p, 4, n, f);
  fclose(f);
}

static void linear_attn(int il, const char *p, const float *h, int pos) {
  char nm[80]; int id;
#define N(s) (snprintf(nm, sizeof nm, "%s%s", p, s), id = find(nm))
  int kd = KDC;

  N("attn_qkv.weight"); gemv(id, tptr(id), h, QKV);
  N("attn_gate.weight"); gemv(id, tptr(id), h, Z);
  N("ssm_beta.weight"); gemv(id, tptr(id), h, SCR);
  for (int i = 0; i < TR; i++) BETA[i] = hf_sigmoid(SCR[i]);
  N("ssm_alpha.weight"); gemv(id, tptr(id), h, SCR);
  N("ssm_dt.bias"); const float *dt = fvec(id);
  N("ssm_a"); const float *alog = fvec(id);
  for (int i = 0; i < TR; i++) GATE_PRE[i] = hf_softplus(SCR[i] + dt[i]) * alog[i];

  N("ssm_conv1d.weight"); const float *cw = fvec(id);
  float *cs = gConv + (size_t)il * (KC - 1) * CONVD;
  for (int c = 0; c < CONVD; c++) {
    float acc = 0.0f;
    for (int k = 0; k < KC - 1; k++) acc += cs[k * CONVD + c] * cw[c * KC + k];
    acc += QKV[c] * cw[c * KC + KC - 1];
    Y[c] = hf_silu(acc);
  }
  if (il == 0) { dl_dump("qkv", QKV, CONVD); dl_dump("conv_y", Y, CONVD);
    dl_dump("beta", BETA, TR); dl_dump("gate_pre", GATE_PRE, TR); }
  for (int c = 0; c < CONVD; c++) {
    for (int k = 0; k < KC - 2; k++) cs[k * CONVD + c] = cs[(k + 1) * CONVD + c];
    cs[(KC - 2) * CONVD + c] = QKV[c];
  }

  const float *yq = Y, *yk = Y + kd, *yv = Y + 2 * kd;
  // q/k are per-group l2-normalized, then tiled GK->TR (np.tile: value head hh
  // reads key group hh % GK, i.e. the whole group axis repeated rep times)
  float invq[64], invk[64];
  for (int g = 0; g < GK; g++) {
    const float *qg = yq + g * SZ, *kg = yk + g * SZ;
    float sq = 0.0f, sk = 0.0f;
    for (int i = 0; i < SZ; i++) { sq += qg[i] * qg[i]; sk += kg[i] * kg[i]; }
    invq[g] = l2_scale(sq) / hf_sqrt((float)SZ);
    invk[g] = l2_scale(sk);
  }
  for (int hh = 0; hh < TR; hh++) {
    int g = hh % GK;
    const float *qg = yq + g * SZ, *kg = yk + g * SZ;
    float *qd = QT + hh * SZ, *kdd = KT + hh * SZ;
    for (int i = 0; i < SZ; i++) { qd[i] = qg[i] * invq[g]; kdd[i] = kg[i] * invk[g]; }
  }

  N("ssm_norm.weight"); const float *sn = fvec(id);
  if (il == 0) { dl_dump("qt", QT, TR * SZ); dl_dump("kt", KT, TR * SZ); }
  float *sbase = gS + (size_t)il * TR * SZ * SZ;
  for (int hh = 0; hh < TR; hh++) {
    float *S = sbase + (size_t)hh * SZ * SZ;
    const float *qh = QT + hh * SZ, *kh = KT + hh * SZ;
    const float *vh = yv + hh * VHD;
    float decay = hf_exp(GATE_PRE[hh]);
    float beta = BETA[hh];
    for (int j = 0; j < SZ; j++) {
      float *srow = S + j * SZ;
      float kvacc = 0.0f;
      for (int i = 0; i < SZ; i++) { srow[i] *= decay; kvacc += srow[i] * kh[i]; }
      float delta = (vh[j] - kvacc) * beta;
      for (int i = 0; i < SZ; i++) srow[i] += kh[i] * delta;
      float o = 0.0f;
      for (int i = 0; i < SZ; i++) o += srow[i] * qh[i];
      SCR[j] = o;
    }
    if (gDumpLayers) memcpy(CORE + (size_t)hh * SZ, SCR, SZ * 4);
    float ss = 0.0f;
    for (int i = 0; i < SZ; i++) ss += SCR[i] * SCR[i];
    float rms = 1.0f / hf_sqrt(ss / (float)SZ + EPS);
    const float *zh = Z + hh * VHD;
    float *oh = OUT0 + hh * VHD;
    for (int i = 0; i < SZ; i++) oh[i] = SCR[i] * rms * sn[i] * hf_silu(zh[i]);
  }
  if (il == 0) { dl_dump("core", CORE, TR * SZ); }
  if (il == 0) dl_dump("out_pre_wo", OUT0, VDC);
  N("ssm_out.weight"); gemv(id, tptr(id), OUT0, SCR);
  memcpy(OUT0, SCR, D * 4);
  if (il == 0) dl_dump("out", OUT0, D);
  (void)pos;
#undef N
}

static void full_attn(int il, const char *p, const float *h, int tpos) {
  char nm[80]; int id;
#define N(s) (snprintf(nm, sizeof nm, "%s%s", p, s), id = find(nm))
  N("attn_q.weight"); gemv(id, tptr(id), h, QGN);      // [H * 2hd]
  N("attn_q_norm.weight"); const float *qnw = fvec(id);
  N("attn_k_norm.weight"); const float *knw = fvec(id);
  for (int hh = 0; hh < H; hh++) {
    const float *q = QGN + hh * 2 * HDIM;
    float ss = 0.0f;
    for (int i = 0; i < HDIM; i++) ss += q[i] * q[i];
    float s = 1.0f / hf_sqrt(ss / (float)HDIM + EPS);
    for (int i = 0; i < HDIM; i++) QN[hh * HDIM + i] = q[i] * s * qnw[i];
    rope_neox(QN + hh * HDIM, tpos, ROPEDIM, HDIM, ROPE_BASE);
  }
  N("attn_k.weight"); gemv(id, tptr(id), h, KB);
  for (int g = 0; g < KV; g++) {
    const float *k = KB + g * HDIM;
    float ss = 0.0f;
    for (int i = 0; i < HDIM; i++) ss += k[i] * k[i];
    float s = 1.0f / hf_sqrt(ss / (float)HDIM + EPS);
    for (int i = 0; i < HDIM; i++) KN[g * HDIM + i] = k[i] * s * knw[i];
    rope_neox(KN + g * HDIM, tpos, ROPEDIM, HDIM, ROPE_BASE);
  }
  N("attn_v.weight"); gemv(id, tptr(id), h, VB);
  size_t base = ((size_t)il * gCap + tpos) * KV * HDIM;
  memcpy(gK + base, KN, KV * HDIM * 4);
  memcpy(gV + base, VB, KV * HDIM * 4);

  int grp = H / KV;
  float scale = 1.0f / hf_sqrt((float)HDIM);
  for (int hh = 0; hh < H; hh++) {
    int kvh = hh / grp;
    const float *qh = QN + hh * HDIM;
    float mx = -INFINITY;
    for (int tp = 0; tp <= tpos; tp++) {
      const float *kh = gK + (((size_t)il * gCap + tp) * KV + kvh) * HDIM;
      float s = 0.0f;
      for (int i = 0; i < HDIM; i++) s += kh[i] * qh[i];
      s *= scale;
      PROBS[tp] = s;
      if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int tp = 0; tp <= tpos; tp++) { float e = hf_exp(PROBS[tp] - mx); PROBS[tp] = e; sum += e; }
    for (int i = 0; i < HDIM; i++) {
      float acc = 0.0f;
      for (int tp = 0; tp <= tpos; tp++) {
        const float *vh = gV + (((size_t)il * gCap + tp) * KV + kvh) * HDIM;
        acc += (PROBS[tp] / sum) * vh[i];
      }
      ATTN[hh * HDIM + i] = acc;
    }
  }
  for (int hh = 0; hh < H; hh++) {
    const float *g = QGN + hh * 2 * HDIM + HDIM;
    for (int i = 0; i < HDIM; i++) ATTN[hh * HDIM + i] *= hf_sigmoid(g[i]);
  }
  N("attn_output.weight"); gemv(id, tptr(id), ATTN, SCR);
  memcpy(OUT0, SCR, D * 4);
#undef N
}

static void forward_token(int tok, int pos) {
  int id; char p[32], nm[80];
  row_f32(find("token_embd.weight"), tok, X);
  if (gDumpLayers) {
    char fn[512]; snprintf(fn, sizeof fn, "%s.-1.f32", gDumpLayers);
    FILE *f = fopen(fn, "wb"); fwrite(X, 4, D, f); fclose(f);
  }
  for (int il = 0; il < L; il++) {
    snprintf(p, sizeof p, "blk.%d.", il);
#define N(s) (snprintf(nm, sizeof nm, "%s%s", p, s), id = find(nm))
    N("attn_norm.weight"); rms_norm(HH, X, fvec(id), D);
    if (il == 0) dl_dump("h", HH, D);
    if (gFull[il]) full_attn(il, p, HH, pos);
    else linear_attn(il, p, HH, pos);
    for (int i = 0; i < D; i++) X[i] += OUT0[i];
    N("post_attention_norm.weight"); rms_norm(HH, X, fvec(id), D);
    N("ffn_gate.weight"); gemv(id, tptr(id), HH, SCR);
    N("ffn_up.weight"); gemv(id, tptr(id), HH, Y);
    for (int i = 0; i < dff; i++) SCR[i] = hf_silu(SCR[i]) * Y[i];
    N("ffn_down.weight"); gemv(id, tptr(id), SCR, Y);
    for (int i = 0; i < D; i++) X[i] += Y[i];
    if (gDumpLayers) {
      char fn[512]; snprintf(fn, sizeof fn, "%s.%d.f32", gDumpLayers, il);
      FILE *f = fopen(fn, "wb"); fwrite(X, 4, D, f); fclose(f);
    }
#undef N
    if (gMapSz) llb_drop_pages(gPack, gMapSz);
  }
}

static void final_logits(void) {
  int onorm = find("output_norm.weight");
  rms_norm(HH, X, fvec(onorm), D);
  int ow = find("output.weight");
  gemv(ow, tptr(ow), HH, LOGITS);
}

static int argmax_first(const float *z, int n) {
  int bi = 0; float bv = z[0];
  for (int i = 1; i < n; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
  return bi;
}
static u32 fbits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static void top2(int *i1, float *v1, int *i2, float *v2) {
  *i1 = argmax_first(LOGITS, V);
  *v1 = LOGITS[*i1];
  float save = LOGITS[*i1];
  LOGITS[*i1] = -INFINITY;
  *i2 = argmax_first(LOGITS, V);
  *v2 = LOGITS[*i2];
  LOGITS[*i1] = save;
}

static u64 now_ms(void) {
  struct timespec ts;
  llb_clock_monotonic(&ts);
  return (u64)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void usage(const char *p) {
  fprintf(stderr,
    "usage: %s <packdir> [tok...] [options]\n"
    "  text in (the normal way):\n"
    "    --prompt \"text\"      raw text prompt (tokenized, parse_special)\n"
    "    --chat \"text\"        render the Qwen3.5 chat template around one user msg\n"
    "    --system \"text\"      system message for --chat\n"
    "    --template chat|raw  (default: chat if --chat, raw if --prompt)\n"
    "    --think / --no-think inject <think>\\n / empty think block at gen prompt\n"
    "  model / backend:\n"
    "    --pack DIR           pack directory (overrides positional)\n"
    "    --quant NAME         use sibling pack-NAME next to the positional dir\n"
    "    --gpu | --metal       offload GEMV (CUDA / Metal build)\n"
    "    --cpu                host GEMV (default)\n"
    "  decoding:\n"
    "    --steps N            number of tokens to generate (default 8)\n"
    "    --temp T --top-p P   sampling (default greedy; T<=0 => greedy)\n"
    "    --seed S             RNG seed\n"
    "    --ids | --text       emit ids / streamed text (default by mode)\n"
    "    --stop ID            extra stop token (repeatable)\n"
    "    --layers L --preload --dump-layers DIR --selftest\n", p);
}

static int is_stop(Q35Tokenizer *t, int id, const int *stops, int nstops) {
  if (id == t->eos) return 1;
  int len; const char *s = q35_tok_text(t, id, &len);
  if ((len == 10 && !memcmp(s, "<|im_end|>", 10)) || (len == 13 && !memcmp(s, "<|endoftext|>", 13))) return 1;
  for (int i = 0; i < nstops; i++) if (id == stops[i]) return 1;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { usage(argv[0]); return 2; }
  const char *dir = argv[1];
  int ids[8192], nids = 0, steps = 8, layers = 0, preload = 0, selftest = 0;
  const char *prompt = NULL, *chat = NULL, *system = NULL, *pack = NULL, *quant = NULL;
  int emit = 0;              // 0 auto, 1 ids, 2 text
  int think = 0;             // 0 none, 1 <think>, 2 empty think block
  int tmpl_raw = -1;         // -1 auto, 0 chat, 1 raw
  float temp = 0.0f, top_p = 1.0f;
  unsigned long seed = 0;
  int stops[64], nstops = 0;

  for (int a = 2; a < argc; a++) {
    if (!strcmp(argv[a], "--steps") && a + 1 < argc) steps = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--layers") && a + 1 < argc) layers = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--preload")) preload = 1;
    else if (!strcmp(argv[a], "--gpu") || !strcmp(argv[a], "--metal")) gUseGpu = 1;
    else if (!strcmp(argv[a], "--cpu")) gUseGpu = 0;
    else if (!strcmp(argv[a], "--selftest")) selftest = 1;
    else if (!strcmp(argv[a], "--dump-layers") && a + 1 < argc) gDumpLayers = argv[++a];
    else if (!strcmp(argv[a], "--pack") && a + 1 < argc) pack = argv[++a];
    else if (!strcmp(argv[a], "--quant") && a + 1 < argc) quant = argv[++a];
    else if (!strcmp(argv[a], "--prompt") && a + 1 < argc) prompt = argv[++a];
    else if (!strcmp(argv[a], "--chat") && a + 1 < argc) chat = argv[++a];
    else if (!strcmp(argv[a], "--system") && a + 1 < argc) system = argv[++a];
    else if (!strcmp(argv[a], "--template") && a + 1 < argc) { const char *m = argv[++a]; tmpl_raw = !strcmp(m, "raw"); }
    else if (!strcmp(argv[a], "--temp") && a + 1 < argc) temp = strtof(argv[++a], NULL);
    else if (!strcmp(argv[a], "--top-p") && a + 1 < argc) top_p = strtof(argv[++a], NULL);
    else if (!strcmp(argv[a], "--seed") && a + 1 < argc) seed = strtoul(argv[++a], NULL, 10);
    else if (!strcmp(argv[a], "--ids")) emit = 1;
    else if (!strcmp(argv[a], "--text")) emit = 2;
    else if (!strcmp(argv[a], "--think")) think = 1;
    else if (!strcmp(argv[a], "--no-think")) think = 2;
    else if (!strcmp(argv[a], "--stop") && a + 1 < argc) stops[nstops++] = atoi(argv[++a]);
    else if (argv[a][0] == '-') { fprintf(stderr, "unknown option %s\n", argv[a]); usage(argv[0]); return 2; }
    else ids[nids++] = atoi(argv[a]);
  }

  if (tmpl_raw == 1 && chat != NULL && prompt == NULL) { prompt = chat; chat = NULL; }
  if (tmpl_raw == 0 && prompt != NULL && chat == NULL) { chat = prompt; prompt = NULL; }

  static char dircopy[4096];
  if (quant) {
    // sibling pack-<quant> next to the positional dir
    const char *slash = strrchr(dir, '/');
    int n = slash ? (int)(slash - dir) : 0;
    snprintf(dircopy, sizeof dircopy, "%.*s/pack-%s", n, dir, quant);
    dir = dircopy;
  }
  if (pack) dir = pack;

  load_index(dir);
  if (layers > 0 && layers < L) L = layers;
  char ppath[4096];
  snprintf(ppath, sizeof ppath, "%s/pack.bin", dir);
  int fd = llb_open_read(ppath);
  if (fd < 0) { llb_perror("open pack"); return 1; }
  int64_t fsz = llb_fdsize(fd);
  if (fsz < 0) { llb_perror("stat pack"); return 1; }
  size_t psz = (size_t)fsz;
  if (preload) {
    uint8_t *buf = (uint8_t *)malloc(psz);
    if (!buf) { fprintf(stderr, "preload malloc %zu failed\n", psz); return 1; }
    size_t off = 0;
    while (off < psz) {
      int64_t got = llb_pread(fd, buf + off, psz - off, off);
      if (got <= 0) { llb_perror("pread pack"); return 1; }
      off += got;
      if ((off & ((64u << 20) - 1)) < (size_t)got) fprintf(stderr, "preload %zu/%zu MB\n", off >> 20, psz >> 20);
    }
    gPack = buf;
    fprintf(stderr, "preloaded pack %zu bytes into RAM\n", psz);
  } else {
    // 1: mapped, drop pages per layer; 0: Windows only, mapping failed and the
    // pack was read into RAM (no page drop); -1: failed
    int mapped = llb_map_read(fd, psz, &gPack);
    if (mapped < 0) { llb_perror("mmap"); return 1; }
    if (mapped) gMapSz = psz;
  }
  fprintf(stderr, "loaded %d tensors, pack %zu bytes; L=%d D=%d H=%d KV=%d hd=%d dff=%d V=%d\n",
          gNT, psz, L, D, H, KV, HDIM, dff, V);
#if defined(__CUDACC__) || defined(__METAL__)
  if (gUseGpu || selftest) {
    int maxrows = V > dff ? V : dff;
    int maxcols = dff > CONVD ? dff : CONVD;
    if (D > maxcols) maxcols = D;
#ifdef __CUDACC__
    cuda_init(psz, maxrows, maxcols);
#else
    metal_init(psz, maxrows, maxcols);
#endif
  }
  if (selftest) {
    int pdiff[16] = {0}, pn[16] = {0}, maxrows = 0, maxcols = 0;
    for (int i = 0; i < gNT; i++) {
      if (gT[i].rows > maxrows) maxrows = gT[i].rows;
      if (gT[i].cols > maxcols) maxcols = gT[i].cols;
    }
    float *x = (float *)malloc((size_t)maxcols * 4);
    float *y1 = (float *)malloc((size_t)maxrows * 4);
    float *y2 = (float *)malloc((size_t)maxrows * 4);
    int total = 0, tested = 0;
    for (int id = 0; id < gNT; id++) {
      int rows = trows(id), cols = tcols(id), fmt = tfmt(id);
      if (rows < 1 || cols < 1) continue;
      for (int i = 0; i < cols; i++) x[i] = hf_sin((float)i) * 0.01f;
      gUseGpu = 0; gemv(id, tptr(id), x, y1);
      gUseGpu = 1; gemv(id, tptr(id), x, y2);
      int diff = 0;
      for (int i = 0; i < rows; i++) if (fbits(y1[i]) != fbits(y2[i])) diff++;
      pdiff[fmt] += diff; pn[fmt]++; total += diff; tested++;
    }
    for (int f = 0; f < 16; f++)
      if (pn[f]) fprintf(stderr, "selftest fmt %d tensors=%d bits_diff=%d\n", f, pn[f], pdiff[f]);
    fprintf(stderr, "selftest TOTAL tensors=%d bits_diff=%d\n", tested, total);
    free(x); free(y1); free(y2);
    return total ? 1 : 0;
  }
#endif

  int text_mode = (prompt != NULL || chat != NULL);
  Q35Tokenizer tok;
  int have_tok = 0;
  int pin[8192], npin = 0;
  if (text_mode) {
    if (q35_tok_load(&tok, dir)) return 1;
    have_tok = 1;
    char *rendered = NULL;
    if (chat) {
      Q35Msg msgs[2];
      int nm = 0;
      if (system) { msgs[nm].role = "system"; msgs[nm].content = system; nm++; }
      msgs[nm].role = "user"; msgs[nm].content = chat; nm++;
      int need = q35_chat_render(msgs, nm, 1, think, NULL, 0);
      rendered = (char *)malloc(need + 1);
      q35_chat_render(msgs, nm, 1, think, rendered, need + 1);
    } else {
      rendered = (char *)malloc(strlen(prompt) + 1);
      strcpy(rendered, prompt);
    }
    npin = q35_tok_encode(&tok, rendered, 1, pin, 8192);
    if (npin > 8192) { fprintf(stderr, "prompt too long\n"); return 1; }
    if (emit == 0) emit = 2;
    if (tmpl_raw == -1) tmpl_raw = chat ? 0 : 1;
    fprintf(stderr, "prompt_text %s\n", rendered);
    fprintf(stderr, "prompt_ids");
    for (int i = 0; i < npin; i++) fprintf(stderr, " %d", pin[i]);
    fprintf(stderr, "\n");
    free(rendered);
  } else {
    npin = nids;
    for (int i = 0; i < nids; i++) pin[i] = ids[i];
    if (emit == 0) emit = 1;
  }

  int cap = npin + steps + 2;
  state_alloc(cap);
  buf_alloc();
  Q35Sampler smp;
  q35_sampler_init(&smp, temp, top_p, seed);
  float *probs = (float *)malloc(sizeof(float) * V);
  int *pidx = (int *)malloc(sizeof(int) * V);

  u64 t0 = now_ms();
  for (int t = 0; t < npin; t++) forward_token(pin[t], t);
  final_logits();
  int i1, i2; float v1, v2;
  top2(&i1, &v1, &i2, &v2);
  fprintf(stderr, "logits %d %d %u %d %u\n", npin - 1, i1, fbits(v1), i2, fbits(v2));
  int cur = (temp <= 0.0f) ? i1 : q35_sample(&smp, LOGITS, V, probs, pidx);
  int gen = 0;
  for (int s = 0; s < steps; s++) {
    if (have_tok && is_stop(&tok, cur, stops, nstops)) break;
    if (emit == 1) {
      printf("%d\n", cur);
    } else {
      char piece[65536];
      int nb = q35_tok_piece(&tok, cur, piece, sizeof piece);
      fwrite(piece, 1, nb, stdout);
      fflush(stdout);
    }
    gen++;
    if (s + 1 >= steps) break;
    u64 s0 = now_ms();
    forward_token(cur, npin + s);
    final_logits();
    top2(&i1, &v1, &i2, &v2);
    fprintf(stderr, "logits %d %d %u %d %u\n", npin + s, i1, fbits(v1), i2, fbits(v2));
    fprintf(stderr, "step %d %llu\n", npin + s, (unsigned long long)(now_ms() - s0));
    cur = (temp <= 0.0f) ? i1 : q35_sample(&smp, LOGITS, V, probs, pidx);
  }
  u64 t1 = now_ms();
  if (emit == 1) {
    printf("wall_ms %llu\n", (unsigned long long)(t1 - t0));
    printf("tok_s %.4f\n", gen > 0 ? (double)gen * 1000.0 / (double)(t1 - t0) : 0.0);
  } else {
    fputc('\n', stdout);
    fprintf(stderr, "\nwall_ms %llu tok_s %.4f gen %d\n",
            (unsigned long long)(t1 - t0), gen > 0 ? (double)gen * 1000.0 / (double)(t1 - t0) : 0.0, gen);
  }
  if (have_tok) q35_tok_free(&tok);
  (void)tmpl_raw;
  return 0;
}
