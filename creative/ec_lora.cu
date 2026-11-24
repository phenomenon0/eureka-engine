// ec_lora.cu — Universal Native LoRA / QLoRA (NF4/FP8/INT8) / DoRA / LoHa / LoKr GPU BF16 Fuser
// 100% Own CUDA + C11. Zero Python, zero cuBLAS, zero PEFT, zero bitsandbytes.
// Supports: FLUX.2, Wan 2.2 TI2V-5B, ACE-Step 1.5 Turbo, TRELLIS.2 1.3B.

#include "ec_lora.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Exact QLoRA NormalFloat4 (Dettmers et al., NeurIPS 2023) 16-entry symmetric quantile table
static const float NF4_LUT[16] = {
  -1.00000000f, -0.69619280f, -0.52507305f, -0.39491749f,
  -0.28444138f, -0.18477343f, -0.09105004f,  0.00000000f,
   0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
   0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f
};

static inline float fp8_e4m3fn_to_f32_host(uint8_t byte) {
  int s = (byte >> 7) & 1;
  int e = (byte >> 3) & 15;
  int m = byte & 7;
  if (e == 15 && m == 7) return 0.0f; // NaN -> 0
  float v = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m * 0.125f, e - 7);
  return s ? -v : v;
}

// ===========================================================================
// 1. GPU Fusion Kernels: LoRA, DoRA, LoHa (Hadamard), LoKr (Kronecker)
// ===========================================================================
static __device__ __forceinline__ uint16_t dev_f32_to_bf16(float v) {
  uint32_t out_b = __float_as_uint(v);
  out_b += 0x7FFFU + ((out_b >> 16) & 1U);
  return (uint16_t)(out_b >> 16);
}

// Standard LoRA / QLoRA-dequantized fusion: W += scale * (B[out_sub, R] @ A[R, in_dim])
__global__ void k_lora_fuse_bf16(
    uint16_t *__restrict__ W_bf16, int row_off, int out_sub, int in_dim, int rank,
    const float *__restrict__ B_f32, const float *__restrict__ A_f32, float scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long total = (long)out_sub * in_dim;
  if (idx >= total) return;
  const int r = (int)(idx / in_dim);
  const int c = (int)(idx % in_dim);
  const float *brow = B_f32 + (long)r * rank;
  float acc = 0.0f;
  for (int k = 0; k < rank; k++) acc += brow[k] * A_f32[(long)k * in_dim + c];
  const long widx = (long)(row_off + r) * in_dim + c;
  float wv = __uint_as_float(((uint32_t)W_bf16[widx]) << 16) + scale * acc;
  W_bf16[widx] = dev_f32_to_bf16(wv);
}

// DoRA (Weight-Decomposed Low-Rank Adaptation, ICML 2024):
// After W' = W + scale * (B @ A), rescale each row i by mag[i] / ||W'_i||_2!
__global__ void k_dora_rescale_rows_bf16(
    uint16_t *__restrict__ W_bf16, int row_off, int out_sub, int in_dim,
    const float *__restrict__ mag_f32) {
  int r = blockIdx.x;
  if (r >= out_sub) return;
  uint16_t *row = W_bf16 + (long)(row_off + r) * in_dim;
  float ss = 0.0f;
  for (int c = threadIdx.x; c < in_dim; c += blockDim.x) {
    float v = __uint_as_float(((uint32_t)row[c]) << 16);
    ss += v * v;
  }
  __shared__ float sm[32];
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x >> 5)) ? sm[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0) sm[0] = mag_f32[r] * rsqrtf(v + 1e-12f);
  }
  __syncthreads();
  float ratio = sm[0];
  for (int c = threadIdx.x; c < in_dim; c += blockDim.x) {
    float v = __uint_as_float(((uint32_t)row[c]) << 16);
    row[c] = dev_f32_to_bf16(v * ratio);
  }
}

// LoHa (Hadamard Product Low-Rank Adaptation, LyCORIS):
// W += scale * ((W1a @ W1b) * (W2a @ W2b))
__global__ void k_loha_fuse_bf16(
    uint16_t *__restrict__ W_bf16, int row_off, int out_sub, int in_dim, int rank1, int rank2,
    const float *__restrict__ W1a, const float *__restrict__ W1b,
    const float *__restrict__ W2a, const float *__restrict__ W2b, float scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long total = (long)out_sub * in_dim;
  if (idx >= total) return;
  const int r = (int)(idx / in_dim);
  const int c = (int)(idx % in_dim);
  float a1 = 0.0f, a2 = 0.0f;
  for (int k = 0; k < rank1; k++) a1 += W1a[(long)r * rank1 + k] * W1b[(long)k * in_dim + c];
  for (int k = 0; k < rank2; k++) a2 += W2a[(long)r * rank2 + k] * W2b[(long)k * in_dim + c];
  const long widx = (long)(row_off + r) * in_dim + c;
  float wv = __uint_as_float(((uint32_t)W_bf16[widx]) << 16) + scale * (a1 * a2);
  W_bf16[widx] = dev_f32_to_bf16(wv);
}

// LoKr (Kronecker Product Low-Rank Adaptation, LyCORIS & ACE-Step 1.5!):
// W += scale * kron(W1[m1, n1], W2[m2, n2]) where m1*m2 == out_sub and n1*n2 == in_dim!
__global__ void k_lokr_fuse_bf16(
    uint16_t *__restrict__ W_bf16, int row_off, int out_sub, int in_dim,
    const float *__restrict__ W1, int m1, int n1,
    const float *__restrict__ W2, int m2, int n2, float scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long total = (long)out_sub * in_dim;
  if (idx >= total) return;
  const int r = (int)(idx / in_dim);
  const int c = (int)(idx % in_dim);
  int r1 = r / m2, r2 = r % m2;
  int c1 = c / n2, c2 = c % n2;
  float delta = W1[(long)r1 * n1 + c1] * W2[(long)r2 * n2 + c2];
  const long widx = (long)(row_off + r) * in_dim + c;
  float wv = __uint_as_float(((uint32_t)W_bf16[widx]) << 16) + scale * delta;
  W_bf16[widx] = dev_f32_to_bf16(wv);
}

// ===========================================================================
// 2. Universal Tensor Reader (F32, BF16, F16, FP8_E4M3FN+scale, INT8+scale, QLoRA NF4+absmax!)
// ===========================================================================
static float *entry_to_host_f32_ex(
    const ec_safetensors_t *st, const ec_st_entry_t *e,
    size_t *out_n, char *fmt_tag, size_t fmt_cap) {
  if (!e) return NULL;
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  const void *src = ec_st_ptr(st, e);

  // Check for companion scale / QLoRA NF4 absmax tensor!
  char k_sc[256];
  const ec_st_entry_t *e_absmax = NULL;
  const ec_st_entry_t *e_scale  = NULL;
  snprintf(k_sc, sizeof k_sc, "%s.absmax", e->name);
  e_absmax = ec_st_find(st, k_sc);
  if (!e_absmax) {
    // Also check stem + ".absmax" when tensor name ends with ".weight"
    size_t nl = strlen(e->name);
    if (nl > 7 && !strcmp(e->name + nl - 7, ".weight")) {
      char stem[220];
      snprintf(stem, sizeof stem, "%.*s", (int)(nl - 7), e->name);
      snprintf(k_sc, sizeof k_sc, "%s.absmax", stem);
      e_absmax = ec_st_find(st, k_sc);
      if (!e_absmax) { snprintf(k_sc, sizeof k_sc, "%s.weight_scale", stem); e_scale = ec_st_find(st, k_sc); }
      if (!e_scale)  { snprintf(k_sc, sizeof k_sc, "%s.scale_weight", stem); e_scale = ec_st_find(st, k_sc); }
      if (!e_scale)  { snprintf(k_sc, sizeof k_sc, "%s.scale", stem);        e_scale = ec_st_find(st, k_sc); }
    }
  }
  if (!e_scale) { snprintf(k_sc, sizeof k_sc, "%s.scale", e->name);        e_scale = ec_st_find(st, k_sc); }
  if (!e_scale) { snprintf(k_sc, sizeof k_sc, "%s_scale", e->name);        e_scale = ec_st_find(st, k_sc); }

  // QLoRA NF4 packed 4-bit (2 weights per U8 byte + blockwise absmax, standard blocksize=64!)
  if (e->dtype == EC_DTYPE_U8 && e_absmax) {
    size_t n_abs = 1;
    for (int i = 0; i < e_absmax->rank; i++) n_abs *= (size_t)e_absmax->shape[i];
    float *abs_h = entry_to_host_f32_ex(st, e_absmax, NULL, NULL, 0);
    if (!abs_h || n_abs == 0) { free(abs_h); return NULL; }
    // Optional custom 16-entry quant_map; otherwise exact QLoRA NF4_LUT
    float lut[16];
    memcpy(lut, NF4_LUT, sizeof lut);
    snprintf(k_sc, sizeof k_sc, "%s.quant_map", e->name);
    const ec_st_entry_t *e_qm = ec_st_find(st, k_sc);
    if (e_qm) {
      size_t nq = 0;
      float *qm = entry_to_host_f32_ex(st, e_qm, &nq, NULL, 0);
      if (qm && nq == 16) memcpy(lut, qm, sizeof lut);
      free(qm);
    }
    size_t total_vals = n * 2;
    size_t blocksize = (total_vals + n_abs - 1) / n_abs;
    if (blocksize == 0) blocksize = 64;
    float *dst = (float *)malloc(total_vals * sizeof(float));
    if (!dst) { free(abs_h); return NULL; }
    const uint8_t *p = (const uint8_t *)src;
    for (size_t bi = 0; bi < n; bi++) {
      uint8_t byte = p[bi];
      size_t i0 = bi * 2, i1 = bi * 2 + 1;
      size_t b0 = i0 / blocksize; if (b0 >= n_abs) b0 = n_abs - 1;
      size_t b1 = i1 / blocksize; if (b1 >= n_abs) b1 = n_abs - 1;
      // BitsAndBytes NF4 packing: high nibble first, low nibble second
      dst[i0] = lut[(byte >> 4) & 0x0F] * abs_h[b0];
      dst[i1] = lut[byte & 0x0F]        * abs_h[b1];
    }
    free(abs_h);
    if (out_n) *out_n = total_vals;
    if (fmt_tag && fmt_cap > 0) snprintf(fmt_tag, fmt_cap, "qlora_nf4");
    return dst;
  }

  float *dst = (float *)malloc(n * sizeof(float));
  if (!dst) return NULL;
  if (e->dtype == EC_DTYPE_F32) {
    memcpy(dst, src, n * sizeof(float));
  } else if (e->dtype == EC_DTYPE_BF16) {
    const uint16_t *p = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
      uint32_t u = ((uint32_t)p[i]) << 16;
      memcpy(&dst[i], &u, 4);
    }
  } else if (e->dtype == EC_DTYPE_F16) {
    const uint16_t *p = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
      uint16_t h = p[i];
      uint32_t s = (uint32_t)(h >> 15) << 31;
      uint32_t exp = (h >> 10) & 0x1F;
      uint32_t man = h & 0x3FF;
      uint32_t u = 0;
      if (exp == 0) {
        if (man == 0) u = s;
        else { float v = (float)man * (1.0f / 16777216.0f); memcpy(&u, &v, 4); u |= s; }
      } else if (exp == 31) u = s | 0x7F800000U | (man << 13);
      else u = s | ((exp + 112) << 23) | (man << 13);
      memcpy(&dst[i], &u, 4);
    }
  } else if (e->dtype == EC_DTYPE_FP8_E4M3) {
    const uint8_t *p = (const uint8_t *)src;
    float sc_scalar = 1.0f;
    float *sc_vec = NULL; size_t sc_n = 0;
    if (e_scale) sc_vec = entry_to_host_f32_ex(st, e_scale, &sc_n, NULL, 0);
    if (sc_vec && sc_n == 1) sc_scalar = sc_vec[0];
    size_t blk = (sc_vec && sc_n > 1) ? ((n + sc_n - 1) / sc_n) : n;
    for (size_t i = 0; i < n; i++) {
      float sc = (sc_vec && sc_n > 1) ? sc_vec[i / blk < sc_n ? i / blk : sc_n - 1] : sc_scalar;
      dst[i] = fp8_e4m3fn_to_f32_host(p[i]) * sc;
    }
    free(sc_vec);
    if (fmt_tag && fmt_cap > 0) snprintf(fmt_tag, fmt_cap, "qlora_fp8");
  } else if (e->dtype == EC_DTYPE_I8) {
    const int8_t *p = (const int8_t *)src;
    float sc_scalar = 1.0f / 127.0f;
    float *sc_vec = NULL; size_t sc_n = 0;
    if (e_scale) sc_vec = entry_to_host_f32_ex(st, e_scale, &sc_n, NULL, 0);
    if (sc_vec && sc_n == 1) sc_scalar = sc_vec[0];
    size_t blk = (sc_vec && sc_n > 1) ? ((n + sc_n - 1) / sc_n) : n;
    for (size_t i = 0; i < n; i++) {
      float sc = (sc_vec && sc_n > 1) ? sc_vec[i / blk < sc_n ? i / blk : sc_n - 1] : sc_scalar;
      dst[i] = (float)p[i] * sc;
    }
    free(sc_vec);
    if (fmt_tag && fmt_cap > 0) snprintf(fmt_tag, fmt_cap, "qlora_int8");
  } else {
    free(dst);
    return NULL;
  }
  if (out_n) *out_n = n;
  return dst;
}

static float read_alpha_or_default(const ec_safetensors_t *st, const char *prefix, const char *alt, float def_alpha) {
  char kal[256];
  snprintf(kal, sizeof kal, "%s%s.alpha", prefix, alt);
  const ec_st_entry_t *e = ec_st_find(st, kal);
  if (!e) return def_alpha;
  size_t na = 0;
  float *ah = entry_to_host_f32_ex(st, e, &na, NULL, 0);
  float v = (ah && na >= 1) ? ah[0] : def_alpha;
  free(ah);
  return v;
}

// ===========================================================================
// 3. Unified Pair Applicator (LoRA + QLoRA NF4/FP8/INT8 + DoRA + LoHa + LoKr!)
// ===========================================================================
static int apply_one_pair(
    const ec_safetensors_t *st, const char *base_key,
    uint16_t *W_bf16_d, int row_off, int expected_out, int expected_in,
    float strength, ec_lora_apply_stats_t *stats, cudaStream_t stream) {
  const char *prefixes[] = {
    "", "diffusion_model.", "model.diffusion_model.", "transformer.", "base_model.model.",
    "base_model.model.base_model.model.",
    "base_model.model.base_model.model.base_model.model.base_model.model.",
    "base_model.model.decoder.", "decoder.", "lora_unet_", "lycoris_"
  };
  const char *a_sufs[] = { ".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight" };
  const char *b_sufs[] = { ".lora_B.weight", ".lora_up.weight",   ".lora_B.default.weight" };

  char ka[256], kb[256], km[256], alt[256];

  for (size_t pi = 0; pi < sizeof prefixes / sizeof prefixes[0]; pi++) {
    snprintf(alt, sizeof alt, "%s", base_key);
    if (!strcmp(prefixes[pi], "lora_unet_") || !strcmp(prefixes[pi], "lycoris_")) {
      for (char *p = alt; *p; p++) if (*p == '.') *p = '_';
    }

    // --- Path A: Standard LoRA / QLoRA (NF4 / FP8 / INT8) + Optional DoRA Magnitude Vector ---
    for (size_t si = 0; si < 3; si++) {
      snprintf(ka, sizeof ka, "%s%s%s", prefixes[pi], alt, a_sufs[si]);
      snprintf(kb, sizeof kb, "%s%s%s", prefixes[pi], alt, b_sufs[si]);
      const ec_st_entry_t *ea = ec_st_find(st, ka);
      const ec_st_entry_t *eb = ec_st_find(st, kb);
      if (!ea || !eb) continue;

      char fmt[64] = "lora";
      size_t na = 0, nb = 0;
      float *Ah = entry_to_host_f32_ex(st, ea, &na, fmt, sizeof fmt);
      float *Bh = entry_to_host_f32_ex(st, eb, &nb, fmt, sizeof fmt);
      if (!Ah || !Bh) { free(Ah); free(Bh); continue; }

      // Derive rank cleanly even for flattened QLoRA NF4 tensors [N/2, 1] -> na == rank * in_dim!
      int rank = 0, in_d = 0, out_d = 0;
      if (ea->rank == 2 && ea->dtype != EC_DTYPE_U8) {
        rank = (int)ea->shape[0]; in_d = (int)ea->shape[1];
      } else if (expected_in > 0 && na % (size_t)expected_in == 0) {
        rank = (int)(na / (size_t)expected_in); in_d = expected_in;
      }
      if (eb->rank == 2 && eb->dtype != EC_DTYPE_U8) {
        out_d = (int)eb->shape[0];
      } else if (rank > 0 && nb % (size_t)rank == 0) {
        out_d = (int)(nb / (size_t)rank);
      }
      if (rank <= 0 || in_d != expected_in || out_d != expected_out || nb != (size_t)out_d * (size_t)rank) {
        free(Ah); free(Bh);
        continue;
      }

      float alpha = read_alpha_or_default(st, prefixes[pi], alt, (float)rank);
      float scale = strength * (alpha / (float)rank);
      if (fabsf(scale) < 1e-8f) { free(Ah); free(Bh); return 0; }

      // Optional DoRA magnitude vector
      snprintf(km, sizeof km, "%s%s.lora_magnitude_vector.default.weight", prefixes[pi], alt);
      const ec_st_entry_t *em = ec_st_find(st, km);
      if (!em) { snprintf(km, sizeof km, "%s%s.lora_magnitude_vector", prefixes[pi], alt); em = ec_st_find(st, km); }
      if (!em) { snprintf(km, sizeof km, "%s%s.dora_scale", prefixes[pi], alt);            em = ec_st_find(st, km); }
      float *Mh = NULL; size_t nm = 0;
      if (em) {
        Mh = entry_to_host_f32_ex(st, em, &nm, NULL, 0);
        if (Mh && (int)nm == out_d) snprintf(fmt, sizeof fmt, "dora");
        else { free(Mh); Mh = NULL; }
      }

      float *Ad = NULL, *Bd = NULL, *Md = NULL;
      cudaMalloc(&Ad, na * sizeof(float));
      cudaMalloc(&Bd, nb * sizeof(float));
      cudaMemcpyAsync(Ad, Ah, na * sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(Bd, Bh, nb * sizeof(float), cudaMemcpyHostToDevice, stream);
      long total = (long)out_d * in_d;
      k_lora_fuse_bf16<<<(int)((total + 255) / 256), 256, 0, stream>>>(
          W_bf16_d, row_off, out_d, in_d, rank, Bd, Ad, scale);
      if (Mh) {
        cudaMalloc(&Md, (size_t)out_d * sizeof(float));
        cudaMemcpyAsync(Md, Mh, (size_t)out_d * sizeof(float), cudaMemcpyHostToDevice, stream);
        k_dora_rescale_rows_bf16<<<out_d, 256, 0, stream>>>(W_bf16_d, row_off, out_d, in_d, Md);
      }
      cudaStreamSynchronize(stream);
      cudaFree(Ad); cudaFree(Bd); cudaFree(Md);
      free(Ah); free(Bh); free(Mh);

      if (stats) {
        stats->n_tensors_matched++;
        if (stats->rank_first == 0) stats->rank_first = rank;
        if (!stats->format_detected[0] || !strcmp(stats->format_detected, "lora"))
          snprintf(stats->format_detected, sizeof stats->format_detected, "%s", fmt);
      }
      return 1;
    }

    // --- Path B: LoHa (Hadamard Product: hada_w1_a/b * hada_w2_a/b) ---
    {
      char k1a[256], k1b[256], k2a[256], k2b[256];
      snprintf(k1a, sizeof k1a, "%s%s.hada_w1_a", prefixes[pi], alt);
      snprintf(k1b, sizeof k1b, "%s%s.hada_w1_b", prefixes[pi], alt);
      snprintf(k2a, sizeof k2a, "%s%s.hada_w2_a", prefixes[pi], alt);
      snprintf(k2b, sizeof k2b, "%s%s.hada_w2_b", prefixes[pi], alt);
      const ec_st_entry_t *e1a = ec_st_find(st, k1a), *e1b = ec_st_find(st, k1b);
      const ec_st_entry_t *e2a = ec_st_find(st, k2a), *e2b = ec_st_find(st, k2b);
      if (e1a && e1b && e2a && e2b && e1a->rank == 2 && e1b->rank == 2 && e2a->rank == 2 && e2b->rank == 2) {
        int out1 = (int)e1a->shape[0], r1 = (int)e1a->shape[1], in1 = (int)e1b->shape[1];
        int out2 = (int)e2a->shape[0], r2 = (int)e2a->shape[1], in2 = (int)e2b->shape[1];
        if (out1 == expected_out && out2 == expected_out && in1 == expected_in && in2 == expected_in &&
            (int)e1b->shape[0] == r1 && (int)e2b->shape[0] == r2) {
          float alpha = read_alpha_or_default(st, prefixes[pi], alt, (float)r1);
          float scale = strength * (alpha / (float)r1);
          size_t n1a, n1b, n2a, n2b;
          float *h1a = entry_to_host_f32_ex(st, e1a, &n1a, NULL, 0);
          float *h1b = entry_to_host_f32_ex(st, e1b, &n1b, NULL, 0);
          float *h2a = entry_to_host_f32_ex(st, e2a, &n2a, NULL, 0);
          float *h2b = entry_to_host_f32_ex(st, e2b, &n2b, NULL, 0);
          if (h1a && h1b && h2a && h2b) {
            float *d1a, *d1b, *d2a, *d2b;
            cudaMalloc(&d1a, n1a * 4); cudaMalloc(&d1b, n1b * 4);
            cudaMalloc(&d2a, n2a * 4); cudaMalloc(&d2b, n2b * 4);
            cudaMemcpyAsync(d1a, h1a, n1a * 4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d1b, h1b, n1b * 4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d2a, h2a, n2a * 4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d2b, h2b, n2b * 4, cudaMemcpyHostToDevice, stream);
            long total = (long)expected_out * expected_in;
            k_loha_fuse_bf16<<<(int)((total + 255) / 256), 256, 0, stream>>>(
                W_bf16_d, row_off, expected_out, expected_in, r1, r2, d1a, d1b, d2a, d2b, scale);
            cudaStreamSynchronize(stream);
            cudaFree(d1a); cudaFree(d1b); cudaFree(d2a); cudaFree(d2b);
            free(h1a); free(h1b); free(h2a); free(h2b);
            if (stats) {
              stats->n_tensors_matched++;
              if (stats->rank_first == 0) stats->rank_first = r1;
              snprintf(stats->format_detected, sizeof stats->format_detected, "loha");
            }
            return 1;
          }
          free(h1a); free(h1b); free(h2a); free(h2b);
        }
      }
    }

    // --- Path C: LoKr (Kronecker Product: lokr_w1 * lokr_w2 or low-rank lokr_w2_a @ lokr_w2_b!) ---
    {
      char k1[256], k2[256], k2a[256], k2b[256];
      snprintf(k1,  sizeof k1,  "%s%s.lokr_w1",   prefixes[pi], alt);
      snprintf(k2,  sizeof k2,  "%s%s.lokr_w2",   prefixes[pi], alt);
      snprintf(k2a, sizeof k2a, "%s%s.lokr_w2_a", prefixes[pi], alt);
      snprintf(k2b, sizeof k2b, "%s%s.lokr_w2_b", prefixes[pi], alt);
      const ec_st_entry_t *e1  = ec_st_find(st, k1);
      const ec_st_entry_t *e2  = ec_st_find(st, k2);
      const ec_st_entry_t *e2a = ec_st_find(st, k2a);
      const ec_st_entry_t *e2b = ec_st_find(st, k2b);
      if (e1 && e1->rank == 2 && (e2 || (e2a && e2b))) {
        int m1 = (int)e1->shape[0], n1 = (int)e1->shape[1];
        int m2 = 0, n2 = 0, rk = 1;
        float *h1 = entry_to_host_f32_ex(st, e1, NULL, NULL, 0);
        float *h2 = NULL;
        if (e2 && e2->rank == 2) {
          m2 = (int)e2->shape[0]; n2 = (int)e2->shape[1];
          h2 = entry_to_host_f32_ex(st, e2, NULL, NULL, 0);
          rk = m2 < n2 ? m2 : n2;
        } else if (e2a && e2b && e2a->rank == 2 && e2b->rank == 2) {
          m2 = (int)e2a->shape[0]; rk = (int)e2a->shape[1]; n2 = (int)e2b->shape[1];
          float *ha = entry_to_host_f32_ex(st, e2a, NULL, NULL, 0);
          float *hb = entry_to_host_f32_ex(st, e2b, NULL, NULL, 0);
          if (ha && hb && (int)e2b->shape[0] == rk) {
            h2 = (float *)calloc((size_t)m2 * n2, sizeof(float));
            for (int r = 0; r < m2; r++)
              for (int c = 0; c < n2; c++) {
                float s = 0.0f;
                for (int kk = 0; kk < rk; kk++) s += ha[(long)r * rk + kk] * hb[(long)kk * n2 + c];
                h2[(long)r * n2 + c] = s;
              }
          }
          free(ha); free(hb);
        }
        if (h1 && h2 && m1 * m2 == expected_out && n1 * n2 == expected_in) {
          float alpha = read_alpha_or_default(st, prefixes[pi], alt, (float)rk);
          float scale = strength * (alpha / (float)(rk > 0 ? rk : 1));
          float *d1, *d2;
          cudaMalloc(&d1, (size_t)m1 * n1 * 4);
          cudaMalloc(&d2, (size_t)m2 * n2 * 4);
          cudaMemcpyAsync(d1, h1, (size_t)m1 * n1 * 4, cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(d2, h2, (size_t)m2 * n2 * 4, cudaMemcpyHostToDevice, stream);
          long total = (long)expected_out * expected_in;
          k_lokr_fuse_bf16<<<(int)((total + 255) / 256), 256, 0, stream>>>(
              W_bf16_d, row_off, expected_out, expected_in, d1, m1, n1, d2, m2, n2, scale);
          cudaStreamSynchronize(stream);
          cudaFree(d1); cudaFree(d2);
          free(h1); free(h2);
          if (stats) {
            stats->n_tensors_matched++;
            if (stats->rank_first == 0) stats->rank_first = rk;
            snprintf(stats->format_detected, sizeof stats->format_detected, "lokr");
          }
          return 1;
        }
        free(h1); free(h2);
      }
    }
  }
  return 0;
}

// ===========================================================================
// 4. Multi-Architecture Key Mapper (FLUX.2, Wan 2.2, ACE-Step 1.5, TRELLIS.2)
// ===========================================================================
extern "C" int ec_lora_fuse_onto_bf16(
    uint16_t *W_bf16_d, int out_dim, int in_dim,
    const char *tensor_name,
    const ec_lora_spec_t *stack, int n_stack,
    ec_lora_apply_stats_t *stats,
    cudaStream_t stream) {
  if (!W_bf16_d || !tensor_name || !stack || n_stack <= 0) return 0;

  char stem[192];
  snprintf(stem, sizeof stem, "%s", tensor_name);
  size_t sl = strlen(stem);
  if (sl > 7 && strcmp(stem + sl - 7, ".weight") == 0) stem[sl - 7] = '\0';

  int total_hits = 0;
  for (int li = 0; li < n_stack; li++) {
    if (!stack[li].path[0] || fabsf(stack[li].strength) < 1e-6f) continue;
    ec_safetensors_t st;
    char err[128] = {0};
    if (ec_st_open(&st, stack[li].path, err, sizeof err) != 0) continue;

    // 1. Direct stem match (works natively for BFL FLUX.2, Wan 2.2 native, ACE-Step 1.5, TRELLIS.2!)
    int hit = apply_one_pair(&st, stem, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);

    // 2. Architecture-specific Diffusers / ComfyUI / PEFT key remapping
    if (!hit) {
      int bi = -1;
      char sub[128];
      // 2a. FLUX.2 Double Blocks (Diffusers)
      if (sscanf(stem, "double_blocks.%d.%127s", &bi, sub) == 2) {
        char dk[192];
        if (strcmp(sub, "img_attn.qkv") == 0 && out_dim == 3 * in_dim) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.to_q", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, in_dim, in_dim, stack[li].strength, stats, stream);
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.to_k", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, in_dim, in_dim, in_dim, stack[li].strength, stats, stream);
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.to_v", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 2 * in_dim, in_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "txt_attn.qkv") == 0 && out_dim == 3 * in_dim) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.add_q_proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, in_dim, in_dim, stack[li].strength, stats, stream);
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.add_k_proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, in_dim, in_dim, in_dim, stack[li].strength, stats, stream);
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.add_v_proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 2 * in_dim, in_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "img_attn.proj") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.to_out.0", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "txt_attn.proj") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.attn.to_add_out", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "img_mlp.0") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.ff.linear_in", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "img_mlp.2") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.ff.linear_out", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "txt_mlp.0") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.ff_context.linear_in", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "txt_mlp.2") == 0) {
          snprintf(dk, sizeof dk, "transformer_blocks.%d.ff_context.linear_out", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        }
      }
      // 2b. FLUX.2 Single Blocks (Diffusers)
      else if (sscanf(stem, "single_blocks.%d.%127s", &bi, sub) == 2) {
        char dk[192];
        if (strcmp(sub, "linear1") == 0) {
          snprintf(dk, sizeof dk, "single_transformer_blocks.%d.attn.to_qkv_mlp_proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "linear2") == 0) {
          snprintf(dk, sizeof dk, "single_transformer_blocks.%d.attn.to_out", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        }
      }
      // 2c. Wan 2.2 TI2V-5B Blocks (Diffusers attn1/attn2/ffn.net + fused QKV support!)
      else if (sscanf(stem, "blocks.%d.%127s", &bi, sub) == 2) {
        char dk[192];
        if (!strcmp(sub, "self_attn.q")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn1.to_q", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "self_attn.k")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn1.to_k", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "self_attn.v")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn1.to_v", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "self_attn.o")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn1.to_out.0", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "cross_attn.q")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn2.to_q", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "cross_attn.k")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn2.to_k", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "cross_attn.v")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn2.to_v", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "cross_attn.o")) {
          snprintf(dk, sizeof dk, "blocks.%d.attn2.to_out.0", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "ffn.0")) {
          snprintf(dk, sizeof dk, "blocks.%d.ffn.net.0.proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (!strcmp(sub, "ffn.2")) {
          snprintf(dk, sizeof dk, "blocks.%d.ffn.net.2", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        }
      }
      // 2d. ACE-Step 1.5 Turbo DiT Layers ("decoder.layers.N.*" <-> "layers.N.*")
      else if (sscanf(stem, "decoder.layers.%d.%127s", &bi, sub) == 2) {
        char dk[192];
        snprintf(dk, sizeof dk, "layers.%d.%s", bi, sub);
        hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
      }
    }
    total_hits += hit;
    ec_st_close(&st);
  }
  return total_hits;
}

extern "C" int ec_lora_inspect_ex(
    const char *path, int *out_pairs, int *out_rank,
    char *arch_hint, size_t arch_cap,
    char *format_out, size_t format_cap) {
  ec_safetensors_t st;
  char err[128] = {0};
  if (ec_st_open(&st, path, err, sizeof err) != 0) return -1;
  int pairs = 0, rank = 0;
  int has_flux2 = 0, has_qwen = 0, has_wan = 0, has_flux1 = 0, has_acestep = 0, has_trellis = 0;
  int has_nf4 = 0, has_fp8 = 0, has_dora = 0, has_loha = 0, has_lokr = 0;
  for (uint32_t i = 0; i < st.n_tensors; i++) {
    const ec_st_entry_t *e = &st.entries[i];
    if (strstr(e->name, "lora_A") || strstr(e->name, "lora_down")) {
      pairs++;
      if (rank == 0 && e->rank == 2 && e->dtype != EC_DTYPE_U8) rank = (int)e->shape[0];
    } else if (strstr(e->name, "hada_w1_a")) {
      pairs++; has_loha = 1;
      if (rank == 0 && e->rank == 2) rank = (int)e->shape[1];
    } else if (strstr(e->name, "lokr_w1")) {
      pairs++; has_lokr = 1;
      if (rank == 0 && e->rank == 2) rank = (int)e->shape[0];
    }
    if (strstr(e->name, ".absmax") || strstr(e->name, "quant_state")) has_nf4 = 1;
    if (e->dtype == EC_DTYPE_FP8_E4M3) has_fp8 = 1;
    if (strstr(e->name, "lora_magnitude_vector") || strstr(e->name, "dora_scale")) has_dora = 1;

    if (strstr(e->name, "to_qkv_mlp_proj") || strstr(e->name, "double_stream_modulation") ||
        (strstr(e->name, "double_blocks") && strstr(e->name, "img_attn.qkv"))) has_flux2 = 1;
    if (strstr(e->name, "double_blocks") || strstr(e->name, "single_blocks")) has_flux1 = 1;
    if (strstr(e->name, "transformer_blocks") && strstr(e->name, "img_mod")) has_qwen = 1;
    if ((strstr(e->name, "self_attn") && strstr(e->name, "cross_attn") && strstr(e->name, "blocks.")) ||
        strstr(e->name, "attn1.to_q")) has_wan = 1;
    if (strstr(e->name, "decoder.layers.") || strstr(e->name, "lyric_encoder") || strstr(e->name, "acestep") ||
        (strstr(e->name, "layers.") && strstr(e->name, "cross_attn.k_proj"))) has_acestep = 1;
    if (strstr(e->name, "ss_flow") || strstr(e->name, "trellis") ||
        strstr(e->name, "cross_attn.to_kv")) has_trellis = 1;
  }
  if (out_pairs) *out_pairs = pairs;
  if (out_rank) *out_rank = rank;
  if (arch_hint && arch_cap > 0) {
    if (has_flux2)        snprintf(arch_hint, arch_cap, "flux2_lora");
    else if (has_acestep) snprintf(arch_hint, arch_cap, "acestep_music_lora");
    else if (has_wan)     snprintf(arch_hint, arch_cap, "wan2_video_lora");
    else if (has_trellis) snprintf(arch_hint, arch_cap, "trellis2_3d_lora");
    else if (has_qwen)    snprintf(arch_hint, arch_cap, "qwen_image_lora");
    else if (has_flux1)   snprintf(arch_hint, arch_cap, "flux_lora");
    else                  snprintf(arch_hint, arch_cap, "generic_lora");
  }
  if (format_out && format_cap > 0) {
    if (has_nf4)       snprintf(format_out, format_cap, "qlora_nf4");
    else if (has_fp8)  snprintf(format_out, format_cap, "qlora_fp8");
    else if (has_dora) snprintf(format_out, format_cap, "dora");
    else if (has_loha) snprintf(format_out, format_cap, "loha");
    else if (has_lokr) snprintf(format_out, format_cap, "lokr");
    else               snprintf(format_out, format_cap, "lora");
  }
  ec_st_close(&st);
  return 0;
}

extern "C" int ec_lora_inspect(const char *path, int *out_pairs, int *out_rank, char *arch_hint, size_t arch_cap) {
  return ec_lora_inspect_ex(path, out_pairs, out_rank, arch_hint, arch_cap, NULL, 0);
}

extern "C" int ec_lora_stack_has_dora(const ec_lora_spec_t *stack, int n_stack) {
  if (!stack || n_stack <= 0) return 0;
  for (int i = 0; i < n_stack && i < EC_LORA_MAX_STACK; i++) {
    if (!stack[i].path[0]) continue;
    char fmt[32] = {0};
    if (ec_lora_inspect_ex(stack[i].path, NULL, NULL, NULL, 0, fmt, sizeof fmt) == 0) {
      if (strcmp(fmt, "dora") == 0) return 1;
    }
  }
  return 0;
}
