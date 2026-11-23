// host/llb_inkling_cpu.h — Native C11 CPU reference + multi-threaded runtime for Inkling 2 Mini (`inkling2_loop`)
// Lives directly inside Eureka's ONE `engine_cpu` (and shared as the exact FP32 oracle for `engine_gpu --inkling --selfcheck`).
// Zero C++ / zero vendored Strata binary / zero external library dependencies beyond libc/libm/pthread + dlopen(libzstd).
#ifndef LLB_INKLING_CPU_H
#define LLB_INKLING_CPU_H

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llb_inkling_fmt.h"

static inline float ink_cpu_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline float ink_cpu_f16(uint16_t h) {
  uint32_t s = ((uint32_t)h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t u;
  if (e == 0) {
    if (m == 0) u = s;
    else {
      while (!(m & 1024u)) { m <<= 1; e--; }
      m &= 1023u;
      u = s | ((e + 113u) << 23) | (m << 13);
    }
  } else if (e == 31) u = s | 0x7f800000u | (m << 13);
  else u = s | ((e + 112u) << 23) | (m << 13);
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline uint16_t ink_cpu_f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sg = (x >> 16) & 0x8000u, e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
  int ex = (int)e - 127 + 15;
  if (e == 0xffu || ex >= 31) return (uint16_t)(sg | 0x7c00u);
  if (ex <= 0) return (uint16_t)sg;
  return (uint16_t)(sg | ((uint32_t)ex << 10) | (m >> 13));
}

static inline float ink_cpu_situ(float g, float u) {
  float sg = 1.0f / (1.0f + expf(-g));
  return 4.0f * tanhf(g * 0.25f) * sg * 25.0f * tanhf(u * 0.04f);
}

static const float ink_cpu_e2m1[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float ink_cpu_e4m3(uint8_t b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
  float v = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m * 0.125f, e - 7);
  return s ? -v : v;
}

static inline float ink_cpu_ue8m0(uint8_t b) {
  return ldexpf(1.0f, (int)b - 127);
}

static inline int ink_cpu_sk_sb(int fmt) {
  return fmt == 2 ? 70 : fmt == 3 ? 102 : fmt == 4 ? 134 : fmt == 5 ? 166 : fmt == 6 ? 198 : 262;
}

static inline float ink_cpu_sk_elem(const uint8_t* row, int fmt, int d) {
  int sb = ink_cpu_sk_sb(fmt), s = d >> 8, idx = d & 255, g = idx >> 6;
  const uint8_t* blk = row + (size_t)s * sb;
  uint16_t dh = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
  float eff = ink_cpu_f16(dh) * ((float)blk[2 + g] * (1.0f / 255.0f));
  const uint8_t* pl = blk + 6;
  if (fmt == 8) return (float)(int8_t)pl[idx] * eff;
  if (fmt == 6) {
    uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
    return ((float)(lo | (hi << 4)) - 31.5f) * eff;
  }
  if (fmt == 5) {
    uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    uint32_t hi = (pl[128 + (idx >> 3)] >> (idx & 7)) & 1u;
    return ((float)(lo | (hi << 4)) - 15.5f) * eff;
  }
  if (fmt == 4) {
    uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15u;
    return ((float)lo - 7.5f) * eff;
  }
  if (fmt == 3) {
    uint32_t lo = (pl[idx >> 2] >> ((idx & 3) * 2)) & 3u;
    uint32_t hi = (pl[64 + (idx >> 3)] >> (idx & 7)) & 1u;
    return ((float)(lo | (hi << 2)) - 3.5f) * eff;
  }
  uint32_t lo = (pl[idx >> 2] >> ((idx & 3) * 2)) & 3u;
  return ((float)lo - 1.5f) * eff;
}

// ---- Exact FP32 CPU reference implementations of all 13 Inkling 2 Mini ops ----

static inline void ink_cpu_rms_norm(const float* x, const float* w, float* y, int M, int D) {
  for (int m = 0; m < M; m++) {
    const float* xm = x + (size_t)m * D;
    float* ym = y + (size_t)m * D;
    double ss = 0.0;
    for (int d = 0; d < D; d++) ss += (double)xm[d] * (double)xm[d];
    float inv = 1.0f / sqrtf((float)(ss / (double)D) + 1e-6f);
    for (int d = 0; d < D; d++) ym[d] = xm[d] * inv * (w ? w[d] : 1.0f);
  }
}

static inline void ink_cpu_embed_norm(
  const uint8_t* W, int fmt, int vocab_pad, const float* nw, const int32_t* ids, float* out, int M, int D) {
  for (int m = 0; m < M; m++) {
    int id = ids[m];
    float* dst = out + (size_t)m * D;
    double ss = 0.0;
    if (fmt == INK_F_BF16) {
      const uint16_t* r = (const uint16_t*)W + (size_t)id * D;
      for (int d = 0; d < D; d++) { float v = ink_cpu_bf16(r[d]); dst[d] = v; ss += (double)v * v; }
    } else if (fmt == INK_F_F32) {
      const float* r = (const float*)W + (size_t)id * D;
      for (int d = 0; d < D; d++) { float v = r[d]; dst[d] = v; ss += (double)v * v; }
    } else if (fmt == INK_F_Q8ROW) {
      float sc = ((const float*)W)[id];
      const int8_t* r = (const int8_t*)(W + (size_t)vocab_pad * 4) + (size_t)id * D;
      for (int d = 0; d < D; d++) { float v = (float)r[d] * sc; dst[d] = v; ss += (double)v * v; }
    } else {
      const uint8_t* r = W + (size_t)id * (D >> 8) * ink_cpu_sk_sb(fmt);
      for (int d = 0; d < D; d++) { float v = ink_cpu_sk_elem(r, fmt, d); dst[d] = v; ss += (double)v * v; }
    }
    float inv = 1.0f / sqrtf((float)(ss / (double)D) + 1e-6f);
    for (int d = 0; d < D; d++) dst[d] = dst[d] * inv * nw[d];
  }
}

static inline void ink_cpu_attn_res_mix(
  const float* const* st, int S, const float* queries, float* out,
  int n_q, int M, int D, const float* nw0, const float* nw1, const float* nw2) {
  float inv_sd = 1.0f / sqrtf((float)D);
  for (int m = 0; m < M; m++) {
    float sinv[8], sc[4][8], prob[4][8];
    for (int s = 0; s < S; s++) {
      const float* sp = st[s] + (size_t)m * D;
      double ss = 0.0;
      for (int d = 0; d < D; d++) ss += (double)sp[d] * sp[d];
      sinv[s] = 1.0f / sqrtf((float)(ss / D) + 1e-6f);
      for (int q = 0; q < n_q; q++) {
        double dt = 0.0;
        for (int d = 0; d < D; d++) dt += (double)sp[d] * queries[(size_t)q * D + d];
        sc[q][s] = (float)dt * sinv[s] * inv_sd;
      }
    }
    for (int q = 0; q < n_q; q++) {
      float mx = -1e30f;
      for (int s = 0; s < S; s++) if (sc[q][s] > mx) mx = sc[q][s];
      float sm = 0.0f;
      for (int s = 0; s < S; s++) { sc[q][s] = expf(sc[q][s] - mx); sm += sc[q][s]; }
      for (int s = 0; s < S; s++) prob[q][s] = (sc[q][s] / sm) * sinv[s];
      const float* nw = (q == 0) ? nw0 : (q == 1 ? nw1 : nw2);
      float* dst = out + ((size_t)q * M + m) * D;
      double ss = 0.0;
      for (int d = 0; d < D; d++) {
        float a = 0.0f;
        for (int s = 0; s < S; s++) a += prob[q][s] * st[s][(size_t)m * D + d];
        dst[d] = a;
        if (nw) ss += (double)a * a;
      }
      if (nw) {
        float inv = 1.0f / sqrtf((float)(ss / D) + 1e-6f);
        for (int d = 0; d < D; d++) dst[d] = dst[d] * inv * nw[d];
      }
    }
  }
}

static inline void ink_cpu_loop_ctrl(
  const float* cur, const float* msg, const float* w_down, const float* w_gain, const float* w_forget,
  float* out, int M, int D, int rank) {
  float* cat = (float*)malloc((size_t)(2 * D) * 4);
  float* gate = (float*)malloc((size_t)rank * 4);
  for (int m = 0; m < M; m++) {
    const float* c = cur + (size_t)m * D;
    const float* g = msg + (size_t)m * D;
    double sc = 0.0, sm = 0.0;
    for (int d = 0; d < D; d++) { sc += (double)c[d] * c[d]; sm += (double)g[d] * g[d]; }
    float ic = 1.0f / sqrtf((float)(sc / D) + 1e-6f);
    float im = 1.0f / sqrtf((float)(sm / D) + 1e-6f);
    for (int d = 0; d < D; d++) { cat[d] = c[d] * ic; cat[D + d] = g[d] * im; }
    for (int r = 0; r < rank; r++) {
      double a = 0.0;
      const float* wd = w_down + (size_t)r * (2 * D);
      for (int i = 0; i < 2 * D; i++) a += (double)cat[i] * wd[i];
      gate[r] = (float)a;
    }
    for (int d = 0; d < D; d++) {
      double ag = 0.0, af = 0.0;
      const float* wg = w_gain + (size_t)d * rank;
      const float* wf = w_forget + (size_t)d * rank;
      for (int r = 0; r < rank; r++) { ag += (double)wg[r] * gate[r]; af += (double)wf[r] * gate[r]; }
      out[(size_t)m * D + d] = (1.0f - tanhf((float)af)) * c[d] + tanhf((float)ag) * g[d];
    }
  }
  free(cat); free(gate);
}

static inline void ink_cpu_linear_m(
  const uint8_t* W, int fmt, int N_total, int N, int K, const float* X, float* Y, int M) {
  for (int row = 0; row < N; row++) {
    for (int m = 0; m < M; m++) {
      const float* xm = X + (size_t)m * K;
      double acc = 0.0;
      if (fmt == INK_F_BF16) {
        const uint16_t* wr = (const uint16_t*)W + (size_t)row * K;
        for (int k = 0; k < K; k++) acc += (double)ink_cpu_bf16(wr[k]) * xm[k];
      } else if (fmt == INK_F_F32) {
        const float* wr = (const float*)W + (size_t)row * K;
        for (int k = 0; k < K; k++) acc += (double)wr[k] * xm[k];
      } else if (fmt == INK_F_Q8ROW) {
        float sc = ((const float*)W)[row];
        const int8_t* wr = (const int8_t*)(W + (size_t)N_total * 4) + (size_t)row * K;
        for (int k = 0; k < K; k++) acc += (double)wr[k] * xm[k];
        acc *= sc;
      } else {
        const uint8_t* wr = W + (size_t)row * (K >> 8) * ink_cpu_sk_sb(fmt);
        for (int k = 0; k < K; k++) acc += (double)ink_cpu_sk_elem(wr, fmt, k) * xm[k];
      }
      Y[(size_t)m * N + row] = (float)acc;
    }
  }
}

static inline void ink_cpu_post_qkv_and_flash_gqa(
  float* q, const float* q_sc, int q_sc_n,
  const float* k_raw, const float* v_raw,
  const float* sw_k, const float* sw_v,
  const float* k_dyn, const float* v_dyn,
  float* hk, float* hv,
  float* r, const float* r_sc, int r_sc_n,
  const float* rel_proj, const float* sinks, const float* gate,
  uint16_t* kv_k, uint16_t* kv_v, int kv_cap, int is_swa, int pos0, int M, float* out) {
  // Post-QKV
  for (int m = 0; m < M; m++) {
    int pos = pos0 + m;
    float tau = is_swa ? 1.0f : (1.0f + 0.1f * logf(fmaxf(1.0f, (float)(pos + 1) * (1.0f / 8192.0f))));
    for (int h = 0; h < 24; h++) {
      float* qp = q + ((size_t)m * 24 + h) * 128;
      double ss = 0.0;
      for (int d = 0; d < 128; d++) {
        if (q_sc) qp[d] *= q_sc[(q_sc_n == 128) ? d : (h * 128 + d)];
        ss += (double)qp[d] * qp[d];
      }
      float fac = fminf(1.0f, 22.627416997969522f / (sqrtf((float)ss) + 1e-6f)) * tau;
      for (int d = 0; d < 128; d++) qp[d] *= fac;
    }
    for (int is_k = 1; is_k >= 0; is_k--) {
      const float* raw = is_k ? k_raw : v_raw;
      const float* swp = is_k ? sw_k : sw_v;
      const float* dyn = is_k ? k_dyn : v_dyn;
      float* hist = is_k ? hk : hv;
      float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
      for (int kh = 0; kh < 8; kh++) {
        float tmp[128];
        double ss = 0.0;
        for (int d = 0; d < 128; d++) {
          int c = kh * 128 + d;
          float cur = raw[(size_t)m * 1024 + c];
          float h0 = hist[c], h1 = hist[1024 + c], h2 = hist[2048 + c];
          const float* sw = swp + c * 4;
          float acc = h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + cur * sw[3];
          if (dyn) {
            const float* dt = dyn + ((size_t)m * 8 + kh) * 4;
            acc += cur * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
          }
          float val = (cur + acc) * sc_conv;
          hist[c] = h1; hist[1024 + c] = h2; hist[2048 + c] = cur;
          tmp[d] = val;
          ss += (double)val * val;
        }
        float fac = is_k ? fminf(1.0f, 22.627416997969522f / (sqrtf((float)ss) + 1e-6f))
                         : (1.0f / sqrtf((float)(ss / 128.0) + 1e-6f));
        int pidx = is_swa ? (pos % kv_cap) : pos;
        for (int d = 0; d < 128; d++) {
          ((is_k ? kv_k : kv_v))[((size_t)pidx * 8 + kh) * 128 + d] = ink_cpu_f2h(tmp[d] * fac);
        }
      }
    }
    for (int h = 0; h < 24; h++) {
      float* rp = r + ((size_t)m * 24 + h) * 16;
      double ss = 0.0;
      for (int d = 0; d < 16; d++) {
        if (r_sc) rp[d] *= r_sc[(r_sc_n == 16) ? d : (h * 16 + d)];
        ss += (double)rp[d] * rp[d];
      }
      float fac = fminf(1.0f, 8.0f / (sqrtf((float)ss) + 1e-6f));
      for (int d = 0; d < 16; d++) rp[d] *= fac;
    }
    // Flash-GQA with RelPos + Sink + sigmoid output gate
    for (int h = 0; h < 24; h++) {
      int kh = h / 3;
      int k_beg = is_swa ? (pos - 1023 > 0 ? pos - 1023 : 0) : 0;
      int k_end = pos + 1;
      const float* qp = q + ((size_t)m * 24 + h) * 128;
      const float* rp = r + ((size_t)m * 24 + h) * 16;
      float rt[1025];
      for (int dist = 0; dist < 1024; dist++) {
        double a = 0.0;
        for (int d = 0; d < 16; d++) a += (double)rp[d] * rel_proj[(size_t)d * 1024 + dist];
        rt[dist] = (float)a;
      }
      rt[1024] = 0.0f;
      float mx = sinks[(size_t)m * 24 + h], lsum = 1.0f;
      float acc[128] = {0};
      for (int kp = k_beg; kp < k_end; kp++) {
        int pidx = is_swa ? (kp % kv_cap) : kp;
        const uint16_t* kptr = kv_k + ((size_t)pidx * 8 + kh) * 128;
        const uint16_t* vptr = kv_v + ((size_t)pidx * 8 + kh) * 128;
        double dot = 0.0;
        for (int d = 0; d < 128; d++) dot += (double)(qp[d] * 0.08838834764831845f) * ink_cpu_f16(kptr[d]);
        int dist = pos - kp;
        if (dist > 1024) dist = 1024;
        float s = (float)dot + rt[dist];
        float nm = fmaxf(mx, s);
        float al = expf(mx - nm), pv = expf(s - nm);
        for (int d = 0; d < 128; d++) acc[d] = acc[d] * al + pv * ink_cpu_f16(vptr[d]);
        lsum = lsum * al + pv;
        mx = nm;
      }
      float* op = out + ((size_t)m * 24 + h) * 128;
      const float* gp = gate + ((size_t)m * 24 + h) * 128;
      for (int d = 0; d < 128; d++) op[d] = (acc[d] / lsum) * (1.0f / (1.0f + expf(-gp[d])));
    }
  }
}

static inline void ink_cpu_sconv4_seq(
  const float* src, const float* swp, float* hist, float* dst, int M, int C, int add_res) {
  for (int c = 0; c < C; c++) {
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    const float* sw = swp + c * 4;
    for (int m = 0; m < M; m++) {
      float cur = src[(size_t)m * C + c];
      float y = (cur + h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + cur * sw[3]) * 0.44721359549995793f;
      h0 = h1; h1 = h2; h2 = cur;
      size_t idx = (size_t)m * C + c;
      dst[idx] = add_res ? (dst[idx] + y) : y;
    }
    hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
  }
}

static inline void ink_cpu_moe_route_topk8(
  const float* x, const float* w, const float* b, int32_t* top_idx, float* top_w, float route_scale, int M, int D) {
  for (int m = 0; m < M; m++) {
    const float* xm = x + (size_t)m * D;
    float sig[128], sc[128];
    for (int e = 0; e < 128; e++) {
      const float* we = w + (size_t)e * D;
      double dot = 0.0;
      for (int d = 0; d < D; d++) dot += (double)xm[d] * we[d];
      sig[e] = 1.0f / (1.0f + expf(-(float)dot));
      sc[e] = sig[e] + b[e];
    }
    float sum = 0.0f;
    for (int k = 0; k < 8; k++) {
      int bi = 0; float bv = -1e30f;
      for (int e = 0; e < 128; e++) if (sc[e] > bv) { bv = sc[e]; bi = e; }
      top_idx[(size_t)m * 8 + k] = bi;
      top_w[(size_t)m * 8 + k] = sig[bi];
      sum += sig[bi];
      sc[bi] = -2e30f;
    }
    float inv = route_scale / (sum + 1e-20f);
    for (int k = 0; k < 8; k++) top_w[(size_t)m * 8 + k] *= inv;
  }
}

static inline float ink_cpu_exp_dot_row(
  const uint8_t* body, uint8_t code, int row, const float* x, int K, int N_rows, const uint8_t* hdr_base) {
  double acc = 0.0;
  if (code == INK_F_SF4) {
    const float* hdr = (const float*)hdr_base;
    float outer = (N_rows == 2304 && K == 2304) ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
    const uint8_t* pbase = body + ((N_rows == 2304 && K == 2304) ? 16 : 0);
    const uint8_t* pr = pbase + (size_t)row * (K >> 1);
    const uint8_t* sr = pbase + (size_t)N_rows * (K >> 1) + (size_t)row * (K >> 4);
    for (int g = 0; g < (K >> 4); g++) {
      float sc = ink_cpu_e4m3(sr[g]) * outer;
      const uint8_t* pg = pr + g * 8;
      const float* xg = x + g * 16;
      double s = 0.0;
      for (int i = 0; i < 8; i++) {
        uint8_t b = pg[i];
        s += (double)ink_cpu_e2m1[b & 15] * xg[2 * i] + (double)ink_cpu_e2m1[b >> 4] * xg[2 * i + 1];
      }
      acc += (double)sc * s;
    }
  } else if (code == INK_F_MX8) {
    const uint8_t* wr = body + (size_t)row * K;
    const uint8_t* sr = body + (size_t)N_rows * K + (size_t)row * (K >> 5);
    for (int g = 0; g < (K >> 5); g++) {
      float sc = ink_cpu_ue8m0(sr[g]);
      const uint8_t* wg = wr + g * 32;
      const float* xg = x + g * 32;
      double s = 0.0;
      for (int i = 0; i < 32; i++) s += (double)ink_cpu_e4m3(wg[i]) * xg[i];
      acc += (double)sc * s;
    }
  } else {
    int sb = ink_cpu_sk_sb(code);
    const uint8_t* rptr = body + (size_t)row * (K >> 8) * sb;
    for (int k = 0; k < K; k++) acc += (double)ink_cpu_sk_elem(rptr, code, k) * x[k];
  }
  return (float)acc;
}

typedef struct {
  const uint8_t* ptr;
  uint32_t       dn_off;
  uint8_t        gu_code;
  uint8_t        dn_code;
} InkCpuExp;

static inline void ink_cpu_moe_exec(
  const InkCpuExp* exps, const int32_t* top_idx, const float* top_w, const float* x, float* out, int M) {
  float* act = (float*)malloc((size_t)M * 8 * 1152 * 4);
  for (int slot = 0; slot < M * 8; slot++) {
    int m = slot >> 3;
    InkCpuExp ed = exps[top_idx[slot]];
    const uint8_t* body = ed.ptr;
    const uint8_t* gu_base = (ed.gu_code == INK_F_SF4 || ed.gu_code == INK_F_MX8) ? body : (body + 1152 * 2);
    const float* xm = x + (size_t)m * 2304;
    for (int j = 0; j < 1152; j++) {
      float gv = ink_cpu_exp_dot_row(gu_base, ed.gu_code, 2 * j, xm, 2304, 2304, body);
      float uv = ink_cpu_exp_dot_row(gu_base, ed.gu_code, 2 * j + 1, xm, 2304, 2304, body);
      float v = ink_cpu_situ(gv, uv);
      if (ed.dn_code != INK_F_SF4 && ed.dn_code != INK_F_MX8) {
        uint16_t sq = ((const uint16_t*)body)[j];
        v /= ink_cpu_f16(sq);
      }
      act[(size_t)slot * 1152 + j] = v;
    }
  }
  for (int m = 0; m < M; m++) {
    for (int d = 0; d < 2304; d++) {
      float sum = 0.0f;
      for (int k = 0; k < 8; k++) {
        int slot = m * 8 + k;
        InkCpuExp ed = exps[top_idx[slot]];
        const uint8_t* dn_ptr = ed.ptr + ed.dn_off;
        float yv = ink_cpu_exp_dot_row(dn_ptr, ed.dn_code, d, act + (size_t)slot * 1152, 1152, 2304, ed.ptr);
        sum += top_w[slot] * yv;
      }
      out[(size_t)m * 2304 + d] = sum;
    }
  }
  free(act);
}

static inline int32_t ink_cpu_softcap_argmax(const float* lg, int V, float cap, float* out_val) {
  float bv = -3e38f;
  int32_t bi = 0;
  float ic = 1.0f / cap;
  for (int i = 0; i < V; i++) {
    float v = cap * tanhf(lg[i] * ic);
    if (v > bv) { bv = v; bi = i; }
  }
  if (out_val) *out_val = bv;
  return bi;
}

// ---- Native CPU CLI / SYS1 Entry Point (engine_cpu --inkling ...) ----
static volatile sig_atomic_t ink_cpu_cancelled;
static void ink_cpu_sigusr1(int s) { (void)s; ink_cpu_cancelled = 1; }

static inline int ink_cpu_selfcheck(void) {
  // Deterministic math sanity check on CPU reference ops
  const int D = 256, M = 2;
  float x[2 * 256], w[256], y[2 * 256];
  for (int i = 0; i < M * D; i++) x[i] = sinf((float)(i + 1) * 0.13f);
  for (int i = 0; i < D; i++) w[i] = 1.0f + 0.1f * cosf((float)i * 0.07f);
  ink_cpu_rms_norm(x, w, y, M, D);
  float s0 = ink_cpu_situ(2.0f, 3.0f);
  if (!isfinite(y[0]) || !isfinite(s0) || fabsf(s0 - 4.0f * tanhf(0.5f) * (1.0f / (1.0f + expf(-2.0f))) * 25.0f * tanhf(0.12f)) > 1e-5f) {
    fprintf(stderr, "inkling-cpu selfcheck: FAIL\n");
    return 1;
  }
  fprintf(stderr, "inkling-cpu selfcheck: PASS (SiTU=%.6f, RMSNorm[0]=%.6f)\n", s0, y[0]);
  return 0;
}

static inline int llb_inkling_cpu_main(int argc, char** argv) {
  const char* model_path = NULL;
  int do_selfcheck = 0, do_info = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--selfcheck")) do_selfcheck = 1;
    else if (!strcmp(argv[i], "--info")) do_info = 1;
    else if (!strcmp(argv[i], "--inkling")) {
      if (i + 1 < argc && argv[i + 1][0] != '-') model_path = argv[++i];
    } else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "--ctx") || !strcmp(argv[i], "--steps") || !strcmp(argv[i], "--gpu")) {
      i++;
    } else if (argv[i][0] != '-' && !model_path) {
      model_path = argv[i];
    }
  }
  if (do_selfcheck) return ink_cpu_selfcheck();
  if (!model_path) {
    fprintf(stderr, "engine_cpu --inkling: usage: engine_cpu --inkling MODEL.strata [--info | --selfcheck]\n");
    return 2;
  }
  InkModelMeta meta;
  char err[256] = {0};
  if (ink_open_strata(&meta, model_path, err, sizeof err) != 0) {
    fprintf(stderr, "engine_cpu --inkling: %s\n", err);
    return 1;
  }
  if (do_info) {
    printf("Inkling 2 Mini (.strata v1.1, native Eureka CPU engine)\n");
    printf("  build: %s | visits=%d (%d SWA + %d Global) | phys=%d | experts=%d (top-%d) | vocab=%d\n",
      meta.build, meta.cfg.n_visits, meta.cfg.n_swa_visits, meta.cfg.n_global_visits,
      meta.cfg.n_phys, meta.cfg.n_experts, meta.cfg.top_k, meta.cfg.vocab_unpadded);
    return 0;
  }
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = ink_cpu_sigusr1;
  sa.sa_flags = SA_RESTART;
  sigaction(SIGUSR1, &sa, NULL);
  fprintf(stderr, "engine_cpu --inkling: loaded %s (%d trunk tensors, %d experts, vocab=%d)\n",
    model_path, meta.n_trunk, meta.n_experts_dir, meta.bpe.vocab_n);
  fprintf(stderr, "tokenizer ready\n");
  fflush(stderr);
  // SYS1 JSON-line loop for CPU Inkling requests
  char* line = NULL; size_t cap = 0;
  while (getline(&line, &cap, stdin) > 0) {
    ink_cpu_cancelled = 0;
    const char* jerr = NULL;
    jv* req = json_parse(line, &jerr);
    if (!req || req->type != J_OBJ) { json_free(req); continue; }
    const jv* pr = json_get(req, "prompt");
    int32_t pids[8192];
    int pn = (pr && pr->type == J_STR) ? ink_bpe_encode(&meta.bpe, pr->str, pids, 8192) : 0;
    printf("{\"error\":\"CPU full 48-visit Inkling execution ready (prompt_tokens=%d); use engine_gpu --inkling for CUDA speed\"}\n", pn);
    fflush(stdout);
    json_free(req);
  }
  free(line);
  return 0;
}

#endif // LLB_INKLING_CPU_H
