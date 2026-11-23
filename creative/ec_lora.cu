// ec_lora.cu — Universal Native LoRA Adapter Loader & GPU BF16 Weight Fuser
// 100% Own CUDA + C11. Zero Python, zero cuBLAS, zero PEFT.

#include "ec_lora.h"
#include "ec_safetensors.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Fuses delta = scale * (B[out_sub, R] @ A[R, in_dim]) into W_bf16_d[row_off : row_off+out_sub, :]
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
  for (int k = 0; k < rank; k++) {
    acc += brow[k] * A_f32[(long)k * in_dim + c];
  }
  const long widx = (long)(row_off + r) * in_dim + c;
  uint32_t bits = ((uint32_t)W_bf16[widx]) << 16;
  float wv = __uint_as_float(bits) + scale * acc;
  // Round-to-nearest-even float32 -> BF16
  uint32_t out_b = __float_as_uint(wv);
  uint32_t lsb = (out_b >> 16) & 1U;
  out_b += 0x7FFFU + lsb;
  W_bf16[widx] = (uint16_t)(out_b >> 16);
}

static float *entry_to_host_f32(const ec_safetensors_t *st, const ec_st_entry_t *e, size_t *out_n) {
  if (!e) return NULL;
  size_t n = 1;
  for (int i = 0; i < e->rank; i++) n *= (size_t)e->shape[i];
  float *dst = (float *)malloc(n * sizeof(float));
  if (!dst) return NULL;
  const void *src = ec_st_ptr(st, e);
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
      // Exact IEEE FP16 -> F32 on host
      uint16_t h = p[i];
      uint32_t s = (uint32_t)(h >> 15) << 31;
      uint32_t exp = (h >> 10) & 0x1F;
      uint32_t man = h & 0x3FF;
      uint32_t u = 0;
      if (exp == 0) {
        if (man == 0) u = s;
        else {
          float v = (float)man * (1.0f / 16777216.0f);
          memcpy(&u, &v, 4);
          u |= s;
        }
      } else if (exp == 31) {
        u = s | 0x7F800000U | (man << 13);
      } else {
        u = s | ((exp + 112) << 23) | (man << 13);
      }
      memcpy(&dst[i], &u, 4);
    }
  } else {
    free(dst);
    return NULL;
  }
  if (out_n) *out_n = n;
  return dst;
}

static int apply_one_pair(
    const ec_safetensors_t *st, const char *base_key,
    uint16_t *W_bf16_d, int row_off, int expected_out, int expected_in,
    float strength, ec_lora_apply_stats_t *stats, cudaStream_t stream) {
  const char *prefixes[] = {
    "", "diffusion_model.", "model.diffusion_model.", "transformer.", "base_model.model.", "lora_unet_"
  };
  const char *a_sufs[] = { ".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight" };
  const char *b_sufs[] = { ".lora_B.weight", ".lora_up.weight",   ".lora_B.default.weight" };

  const ec_st_entry_t *ea = NULL, *eb = NULL, *ealpha = NULL;
  char ka[256], kb[256], kal[256], alt[256];

  for (size_t pi = 0; pi < sizeof prefixes / sizeof prefixes[0] && !ea; pi++) {
    // Also convert dots to underscores for Kohya lora_unet_ style
    snprintf(alt, sizeof alt, "%s", base_key);
    if (strcmp(prefixes[pi], "lora_unet_") == 0) {
      for (char *p = alt; *p; p++) if (*p == '.') *p = '_';
    }
    for (size_t si = 0; si < 3; si++) {
      snprintf(ka, sizeof ka, "%s%s%s", prefixes[pi], alt, a_sufs[si]);
      snprintf(kb, sizeof kb, "%s%s%s", prefixes[pi], alt, b_sufs[si]);
      ea = ec_st_find(st, ka);
      eb = ec_st_find(st, kb);
      if (ea && eb) {
        snprintf(kal, sizeof kal, "%s%s.alpha", prefixes[pi], alt);
        ealpha = ec_st_find(st, kal);
        break;
      }
    }
  }
  if (!ea || !eb || ea->rank != 2 || eb->rank != 2) return 0;

  int rank = (int)ea->shape[0];
  int in_d = (int)ea->shape[1];
  int out_d = (int)eb->shape[0];
  if ((int)eb->shape[1] != rank || in_d != expected_in || out_d != expected_out) return 0;

  float alpha = (float)rank;
  if (ealpha) {
    size_t na = 0;
    float *ah = entry_to_host_f32(st, ealpha, &na);
    if (ah && na >= 1) alpha = ah[0];
    free(ah);
  }
  float scale = strength * (alpha / (float)rank);
  if (fabsf(scale) < 1e-8f) return 0;

  size_t na = 0, nb = 0;
  float *Ah = entry_to_host_f32(st, ea, &na);
  float *Bh = entry_to_host_f32(st, eb, &nb);
  if (!Ah || !Bh) { free(Ah); free(Bh); return 0; }

  float *Ad = NULL, *Bd = NULL;
  cudaMalloc(&Ad, na * sizeof(float));
  cudaMalloc(&Bd, nb * sizeof(float));
  cudaMemcpyAsync(Ad, Ah, na * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Bd, Bh, nb * sizeof(float), cudaMemcpyHostToDevice, stream);

  long total = (long)out_d * in_d;
  int blk = 256;
  int grd = (int)((total + blk - 1) / blk);
  k_lora_fuse_bf16<<<grd, blk, 0, stream>>>(W_bf16_d, row_off, out_d, in_d, rank, Bd, Ad, scale);
  cudaStreamSynchronize(stream);

  cudaFree(Ad);
  cudaFree(Bd);
  free(Ah);
  free(Bh);

  if (stats) {
    stats->n_tensors_matched++;
    if (stats->rank_first == 0) stats->rank_first = rank;
  }
  return 1;
}

extern "C" int ec_lora_fuse_onto_bf16(
    uint16_t *W_bf16_d, int out_dim, int in_dim,
    const char *bfl_tensor_name,
    const ec_lora_spec_t *stack, int n_stack,
    ec_lora_apply_stats_t *stats,
    cudaStream_t stream) {
  if (!W_bf16_d || !bfl_tensor_name || !stack || n_stack <= 0) return 0;

  // Strip trailing ".weight" to get module stem
  char stem[192];
  snprintf(stem, sizeof stem, "%s", bfl_tensor_name);
  size_t sl = strlen(stem);
  if (sl > 7 && strcmp(stem + sl - 7, ".weight") == 0) stem[sl - 7] = '\0';

  int total_hits = 0;
  for (int li = 0; li < n_stack; li++) {
    if (!stack[li].path[0] || fabsf(stack[li].strength) < 1e-6f) continue;
    ec_safetensors_t st;
    char err[128] = {0};
    if (ec_st_open(&st, stack[li].path, err, sizeof err) != 0) continue;

    // 1. Direct BFL stem match
    int hit = apply_one_pair(&st, stem, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);

    // 2. Auto-map Diffusers FLUX.2 keys onto BFL tensors!
    if (!hit) {
      int bi = -1;
      char sub[128];
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
      } else if (sscanf(stem, "single_blocks.%d.%127s", &bi, sub) == 2) {
        char dk[192];
        if (strcmp(sub, "linear1") == 0) {
          snprintf(dk, sizeof dk, "single_transformer_blocks.%d.attn.to_qkv_mlp_proj", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        } else if (strcmp(sub, "linear2") == 0) {
          snprintf(dk, sizeof dk, "single_transformer_blocks.%d.attn.to_out", bi);
          hit += apply_one_pair(&st, dk, W_bf16_d, 0, out_dim, in_dim, stack[li].strength, stats, stream);
        }
      }
    }
    total_hits += hit;
    ec_st_close(&st);
  }
  return total_hits;
}

extern "C" int ec_lora_inspect(const char *path, int *out_pairs, int *out_rank, char *arch_hint, size_t arch_cap) {
  ec_safetensors_t st;
  char err[128] = {0};
  if (ec_st_open(&st, path, err, sizeof err) != 0) return -1;
  int pairs = 0, rank = 0;
  int has_flux2 = 0, has_qwen = 0, has_wan = 0, has_flux1 = 0;
  for (uint32_t i = 0; i < st.n_tensors; i++) {
    const ec_st_entry_t *e = &st.entries[i];
    if (strstr(e->name, "lora_A") || strstr(e->name, "lora_down")) {
      pairs++;
      if (rank == 0 && e->rank == 2) rank = (int)e->shape[0];
    }
    if (strstr(e->name, "to_qkv_mlp_proj") || strstr(e->name, "double_stream_modulation")) has_flux2 = 1;
    if (strstr(e->name, "double_blocks") || strstr(e->name, "single_blocks")) has_flux1 = 1;
    if (strstr(e->name, "transformer_blocks") && strstr(e->name, "img_mod")) has_qwen = 1;
    if (strstr(e->name, "self_attn") && strstr(e->name, "cross_attn")) has_wan = 1;
  }
  if (out_pairs) *out_pairs = pairs;
  if (out_rank) *out_rank = rank;
  if (arch_hint && arch_cap > 0) {
    if (has_flux2) snprintf(arch_hint, arch_cap, "flux2_lora");
    else if (has_qwen) snprintf(arch_hint, arch_cap, "qwen_image_lora");
    else if (has_wan) snprintf(arch_hint, arch_cap, "wan2_video_lora");
    else if (has_flux1) snprintf(arch_hint, arch_cap, "flux_lora");
    else snprintf(arch_hint, arch_cap, "generic_lora");
  }
  ec_st_close(&st);
  return 0;
}
