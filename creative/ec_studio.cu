// ec_studio.cu — 100% Native Eureka End-to-End Multi-Modal Creative Studio Runtime (v2)
// Thread-safe GPU execution, Shifted RF + MagCache + TensorCore Conv2D + NVENC H.264/AAC MP4,
// Live Multi-Architecture Safetensors Registry (Z-Image, ACE-Step 1.5, TRELLIS.2, Wan 2.2, LTX-2.5),
// Cooperative SSE Progress & Instant Cancellation.

#define _POSIX_C_SOURCE 200809L
#include "ec_studio.h"
#include "ec_acestep.h"
#include "ec_cowrie.h"
#include "ec_dit.h"
#include "ec_flow.h"
#include "ec_flux2.h"
#include "ec_kernels_v2.cuh"
#include "ec_lora.h"
#include "ec_media.h"
#include "ec_models.h"
#include "ec_nvenc.h"
#include "ec_recipes.h"
#include "ec_storage.h"
#include "ec_trellis2.h"
#include "ec_vram_pool.h"
#include "ec_wan22.h"
#include "ec_ltx2.h"
#include "ec_zimage.h"
#include <cuda_fp16.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

struct ec_studio {
  char   artifact_dir[256];
  char   models_dir[256];
  char   gpu_name[128];
  int    sm_major, sm_minor;
  double vram_total_gb;

  pthread_mutex_t gpu_mu;   // serializes GPU workspace access while CPU I/O & HTTP run in parallel
  ec_vram_pool_t  vram_pool;
  int             slot_dit_weights;
  int             slot_vae_weights;
  int             slot_3d_weights;
  int             slot_music_weights;

  ec_dit_cfg_t     dit_cfg;
  ec_dit_block_w_t dit_w;
  ec_dit_ws_t      dit_ws;
  int32_t         *coords_d;
  float           *x_d;
  float           *v_d;
  cudaStream_t     compute_stream;
  ec_flow_graph_t     flow_graph;
  ec_model_registry_t registry;
  ec_storage_vault_t  vault;
  ec_recipe_lib_t     recipes;
  ec_flux2_t         *flux2;
  ec_zimage_t        *zimage;
  ec_acestep_t       *acestep;
  ec_trellis2_t      *trellis2;
  ec_wan22_t         *wan22;
  ec_ltx2_t          *ltx2;
  char                dag_model_sha[65];
  uint64_t            artifact_seq;
  int                 active_engine_mask;
  int                 initialized;
};

// Evicts resident GPU weights from all flagship engines EXCEPT keep_id (1=FLUX.2, 2=ACE-Step, 3=TRELLIS.2, 4=Wan2.2, 5=Z-Image, 6=LTX-2.3, 0=ALL).
// Lightweight mmap handles & tokenizers stay open so switching modalities incurs zero file re-parsing!
static void studio_evict_other_engines_locked(ec_studio_t *st, int keep_id) {
  if (keep_id != 1 && st->flux2)    ec_flux2_unload_vram(st->flux2);
  if (keep_id != 2 && st->acestep)  ec_acestep_unload_vram(st->acestep);
  if (keep_id != 3 && st->trellis2) ec_trellis2_unload_vram(st->trellis2);
  if (keep_id != 4 && st->wan22)    ec_wan22_unload_vram(st->wan22);
  if (keep_id != 5 && st->zimage)   ec_zimage_unload_vram(st->zimage);
  if (keep_id != 6 && st->ltx2)     ec_ltx2_unload_vram(st->ltx2);
  st->active_engine_mask = (keep_id >= 1 && keep_id <= 6) ? (1 << (keep_id - 1)) : 0;
}

extern "C" int ec_studio_unload_all_vram(ec_studio_t *st, double *out_ms, double *out_free_mb) {
  if (!st || !st->initialized) return -1;
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  pthread_mutex_lock(&st->gpu_mu);
  studio_evict_other_engines_locked(st, 0);
  cudaDeviceSynchronize();
  size_t free_b = 0, tot_b = 0;
  if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess && out_free_mb) {
    *out_free_mb = (double)free_b / (1024.0 * 1024.0);
  }
  pthread_mutex_unlock(&st->gpu_mu);
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  if (out_ms) *out_ms = ((double)(ts1.tv_sec - ts0.tv_sec) * 1000.0) + ((double)(ts1.tv_nsec - ts0.tv_nsec) / 1e6);
  return 0;
}

static void studio_try_init_all_engines_locked(ec_studio_t *st) {
  // 1. Photoreal Image: FLUX.2-klein-4B
  if (!st->flux2 || !ec_flux2_is_ready(st->flux2)) {
    const ec_storage_asset_t *ckpt = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "flux2_klein_4b");
    const ec_storage_asset_t *vae  = ec_storage_find(&st->vault, EC_ASSET_VAE, "flux2_klein_vae");
    if (!vae) vae = ec_storage_find(&st->vault, EC_ASSET_VAE, "flux2");
    const ec_storage_asset_t *te   = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "qwen_3_4b");
    if (!te) te = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "qwen3_4b");
    const ec_storage_asset_t *tok  = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "qwen3_tokenizer");
    if (!tok) tok = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "tokenizer");
    if (ckpt && vae && te && tok) {
      ec_flux2_paths_t p = {};
      snprintf(p.checkpoint_path,   sizeof p.checkpoint_path,   "%s", ckpt->path);
      snprintf(p.vae_path,          sizeof p.vae_path,          "%s", vae->path);
      snprintf(p.text_encoder_path, sizeof p.text_encoder_path, "%s", te->path);
      snprintf(p.tokenizer_path,    sizeof p.tokenizer_path,    "%s", tok->path);
      if (st->flux2) { ec_flux2_destroy(st->flux2); st->flux2 = NULL; }
      st->flux2 = ec_flux2_create(&p);
    }
  }

  // 1b. Photoreal Image: Tongyi-MAI Z-Image-Turbo (6.15B S3-DiT + FLUX.1 16ch VAE + shared Qwen3-4B!)
  if (!st->zimage || !ec_zimage_is_ready(st->zimage)) {
    const ec_storage_asset_t *dit = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "z_image_turbo");
    const ec_storage_asset_t *vae = ec_storage_find(&st->vault, EC_ASSET_VAE, "flux1_ae");
    if (!vae) vae = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "flux1_ae");
    const ec_storage_asset_t *te  = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "qwen_3_4b");
    if (!te) te = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "qwen3_4b");
    const ec_storage_asset_t *tok = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "qwen3_tokenizer");
    if (!tok) tok = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "tokenizer");
    if (dit && vae && te && tok) {
      ec_zimage_paths_t zp = {};
      snprintf(zp.dit_path,          sizeof zp.dit_path,          "%s", dit->path);
      snprintf(zp.vae_path,          sizeof zp.vae_path,          "%s", vae->path);
      snprintf(zp.text_encoder_path, sizeof zp.text_encoder_path, "%s", te->path);
      snprintf(zp.tokenizer_path,    sizeof zp.tokenizer_path,    "%s", tok->path);
      if (st->zimage) { ec_zimage_destroy(st->zimage); st->zimage = NULL; }
      st->zimage = ec_zimage_create(&zp);
    }
  }

  // 2. AI Music + 4 Stems: ACE-Step 1.5 Turbo
  if (!st->acestep || !ec_acestep_is_ready(st->acestep)) {
    const ec_storage_asset_t *dit = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "acestep_v15_turbo");
    const ec_storage_asset_t *vae = ec_storage_find(&st->vault, EC_ASSET_VAE, "acestep_v15_vae");
    const ec_storage_asset_t *te  = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "acestep_qwen3_0.6b");
    const ec_storage_asset_t *sil = ec_storage_find(&st->vault, EC_ASSET_OTHER, "acestep_silence_latent");
    const ec_storage_asset_t *tok = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "qwen3_tokenizer");
    if (dit && vae && te && sil && tok) {
      ec_acestep_paths_t p = {};
      snprintf(p.dit_path,            sizeof p.dit_path,            "%s", dit->path);
      snprintf(p.vae_path,            sizeof p.vae_path,            "%s", vae->path);
      snprintf(p.text_encoder_path,   sizeof p.text_encoder_path,   "%s", te->path);
      snprintf(p.silence_latent_path, sizeof p.silence_latent_path, "%s", sil->path);
      snprintf(p.tokenizer_path,      sizeof p.tokenizer_path,      "%s", tok->path);
      if (st->acestep) { ec_acestep_destroy(st->acestep); st->acestep = NULL; }
      st->acestep = ec_acestep_create(&p);
    }
  }

  // 3. Native 3D Asset Generation: TRELLIS.2 (Stage-1 1.3B + optional v17.2 Full 3-Stage 512^3 O-Voxel PBR!)
  if (!st->trellis2 || !ec_trellis2_is_ready(st->trellis2)) {
    const ec_storage_asset_t *dino = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "dinov3_vitl16");
    const ec_storage_asset_t *flow = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_ss_flow");
    const ec_storage_asset_t *dec  = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_ss_dec");
    if (dino && flow && dec) {
      ec_trellis2_paths_t p = {};
      snprintf(p.dino_path,    sizeof p.dino_path,    "%s", dino->path);
      snprintf(p.ss_flow_path, sizeof p.ss_flow_path, "%s", flow->path);
      snprintf(p.ss_dec_path,  sizeof p.ss_dec_path,  "%s", dec->path);
      const ec_storage_asset_t *s_fl = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_slat_shape_flow");
      const ec_storage_asset_t *t_fl = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_slat_tex_flow");
      const ec_storage_asset_t *s_dc = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_shape_dec");
      const ec_storage_asset_t *t_dc = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_tex_dec");
      if (s_fl && t_fl && s_dc && t_dc) {
        snprintf(p.slat_shape_flow_path, sizeof p.slat_shape_flow_path, "%s", s_fl->path);
        snprintf(p.slat_tex_flow_path,   sizeof p.slat_tex_flow_path,   "%s", t_fl->path);
        snprintf(p.shape_dec_path,       sizeof p.shape_dec_path,       "%s", s_dc->path);
        snprintf(p.tex_dec_path,         sizeof p.tex_dec_path,         "%s", t_dc->path);
      }
      if (st->trellis2) { ec_trellis2_destroy(st->trellis2); st->trellis2 = NULL; }
      st->trellis2 = ec_trellis2_create(&p);
    }
  }

  // 4. Native Cinema Video: Wan 2.2 TI2V-5B (UMT5-XXL FP8 + 30L 3D RoPE DiT + Causal 3D VAE)
  if (!st->wan22 || !ec_wan22_is_ready(st->wan22)) {
    const ec_storage_asset_t *dit = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "wan2.2_ti2v_5b");
    const ec_storage_asset_t *vae = ec_storage_find(&st->vault, EC_ASSET_VAE, "wan2.2_vae");
    const ec_storage_asset_t *te  = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "umt5_xxl");
    const ec_storage_asset_t *tok = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "umt5_tokenizer");
    if (dit && vae && te && tok) {
      ec_wan22_paths_t p = {};
      snprintf(p.dit_path,          sizeof p.dit_path,          "%s", dit->path);
      snprintf(p.vae_path,          sizeof p.vae_path,          "%s", vae->path);
      snprintf(p.text_encoder_path, sizeof p.text_encoder_path, "%s", te->path);
      snprintf(p.tokenizer_path,    sizeof p.tokenizer_path,    "%s", tok->path);
      if (st->wan22) { ec_wan22_destroy(st->wan22); st->wan22 = NULL; }
      st->wan22 = ec_wan22_create(&p);
    }
  }

  // 5. v17.3 Joint Synchronized Audio+Video Frontier: Lightricks LTX-2.3 (22B Distilled FP8 + Gemma-3-12B-IT)
  if (!st->ltx2 || !ec_ltx2_is_ready(st->ltx2)) {
    const ec_storage_asset_t *ckpt = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "ltx-2.3");
    if (!ckpt) ckpt = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "ltx2");
    const ec_storage_asset_t *te   = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "gemma_3_12B");
    if (!te) te = ec_storage_find(&st->vault, EC_ASSET_TEXT_ENC, "gemma");
    const ec_storage_asset_t *tok  = ec_storage_find(&st->vault, EC_ASSET_TOKENIZER, "gemma3_tokenizer");
    if (ckpt && te && tok) {
      ec_ltx2_paths_t lp = {};
      snprintf(lp.checkpoint_path,   sizeof lp.checkpoint_path,   "%s", ckpt->path);
      snprintf(lp.text_encoder_path, sizeof lp.text_encoder_path, "%s", te->path);
      snprintf(lp.tokenizer_path,    sizeof lp.tokenizer_path,    "%s", tok->path);
      if (st->ltx2) { ec_ltx2_destroy(st->ltx2); st->ltx2 = NULL; }
      st->ltx2 = ec_ltx2_create(&lp);
    }
  }

  // Compute composite 64-hex model contract SHA for persistent Cowrie ComfyUI DAG StatePages
  char fp_flux[65] = "none", fp_ace[65] = "none", fp_tr[65] = "none", fp_wan[65] = "none";
  const ec_storage_asset_t *a_fl = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "flux2_klein_4b");
  const ec_storage_asset_t *a_ac = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "acestep_v15_turbo");
  const ec_storage_asset_t *a_tr = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_ss_flow");
  const ec_storage_asset_t *a_wn = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "wan2.2_ti2v_5b");
  if (a_fl) ec_cowrie_file_fingerprint(a_fl->path, fp_flux);
  if (a_ac) ec_cowrie_file_fingerprint(a_ac->path, fp_ace);
  if (a_tr) ec_cowrie_file_fingerprint(a_tr->path, fp_tr);
  if (a_wn) ec_cowrie_file_fingerprint(a_wn->path, fp_wan);
  const char *parts[5] = { "eureka_dag_v7", fp_flux, fp_ace, fp_tr, fp_wan };
  ec_cowrie_composite_sha(parts, 5, st->dag_model_sha);
}

static double st_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static uint64_t hash_prompt_seed(const char *prompt, uint32_t seed) {
  uint64_t h = 14695981039346656037ULL ^ ((uint64_t)seed * 0x9E3779B97F4A7C15ULL);
  if (prompt) {
    for (const unsigned char *p = (const unsigned char *)prompt; *p; p++) {
      h ^= (uint64_t)*p;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

static uint64_t hash_mix_str(uint64_t h, const char *s) {
  if (!s || !s[0]) return h;
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    h ^= (uint64_t)*p;
    h *= 1099511628211ULL;
  }
  return h;
}

static uint32_t next_artifact_hash32(ec_studio_t *st, uint64_t h) {
  uint64_t seq = __sync_add_and_fetch(&st->artifact_seq, 1ULL);
  h ^= seq * 0xBF58476D1CE4E5B9ULL;
  h ^= (uint64_t)time(NULL) * 0x94D049BB133111EBULL;
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 27;
  return (uint32_t)(h ^ (h >> 32));
}

static int ci_substr(const char *hay, const char *needle) {
  if (!hay || !needle || !*needle) return 0;
  size_t nl = strlen(needle);
  for (const char *p = hay; *p; p++) if (strncasecmp(p, needle, nl) == 0) return 1;
  return 0;
}

static float prng_next(uint64_t *s) {
  uint64_t x = *s;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  *s = x;
  uint32_t v = (uint32_t)((x * 2685821657736338717ULL) >> 40);
  return ((float)v / 8388608.0f) - 1.0f;
}

static int ec_studio_init_impl(ec_studio_t *st, const char *artifact_dir, const char *models_dir, size_t vram_budget_mb) {
  memset(st, 0, sizeof *st);
  snprintf(st->artifact_dir, sizeof st->artifact_dir, "%s", artifact_dir ? artifact_dir : "/tmp/eureka-creative-build/artifacts");
  snprintf(st->models_dir, sizeof st->models_dir, "%s", models_dir ? models_dir : "");
  pthread_mutex_init(&st->gpu_mu, NULL);

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;
  snprintf(st->gpu_name, sizeof st->gpu_name, "%s", prop.name);
  st->sm_major = prop.major;
  st->sm_minor = prop.minor;
  st->vram_total_gb = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);

  if (cudaStreamCreate(&st->compute_stream) != cudaSuccess) return -1;
  if (ec_vram_pool_init(&st->vram_pool, vram_budget_mb * 1024ULL * 1024ULL) != 0) return -1;

  st->slot_dit_weights   = ec_vram_register(&st->vram_pool, "dit_core_fp16",     16ULL << 20, NULL, 1);
  st->slot_vae_weights   = ec_vram_register(&st->vram_pool, "vae_2d_3d_causal",   8ULL << 20, NULL, 0);
  st->slot_3d_weights    = ec_vram_register(&st->vram_pool, "trellis_ovoxel_3d",  8ULL << 20, NULL, 0);
  st->slot_music_weights = ec_vram_register(&st->vram_pool, "acestep_dcae_1d",    8ULL << 20, NULL, 0);

  st->dit_cfg.dim = 512; st->dit_cfg.n_heads = 8; st->dit_cfg.head_dim = 64; st->dit_cfg.ffn_dim = 2048;
  st->dit_cfg.ndims = 3; st->dit_cfg.axis_dims[0] = 16; st->dit_cfg.axis_dims[1] = 24; st->dit_cfg.axis_dims[2] = 24;
  st->dit_cfg.rope_theta = 10000.0f; st->dit_cfg.eps = 1e-6f; st->dit_cfg.norm_mode = 0;

  const int N = 256, E = st->dit_cfg.dim, F = st->dit_cfg.ffn_dim;
  if (ec_dit_ws_alloc(&st->dit_ws, &st->dit_cfg, N, 16) != 0) return -1;

  cudaMalloc(&st->dit_w.w_qkv,  (size_t)3 * E * E * 2);
  cudaMalloc(&st->dit_w.w_proj, (size_t)E * E * 2);
  cudaMalloc(&st->dit_w.w_fc1,  (size_t)F * E * 2);
  cudaMalloc(&st->dit_w.w_fc2,  (size_t)E * F * 2);
  cudaMalloc(&st->dit_w.shift_msa, E * 4); cudaMalloc(&st->dit_w.scale_msa, E * 4); cudaMalloc(&st->dit_w.gate_msa, E * 4);
  cudaMalloc(&st->dit_w.shift_mlp, E * 4); cudaMalloc(&st->dit_w.scale_mlp, E * 4); cudaMalloc(&st->dit_w.gate_mlp, E * 4);
  cudaMalloc(&st->coords_d, (size_t)N * 3 * 4);
  cudaMalloc(&st->x_d,      (size_t)N * E * 4);
  cudaMalloc(&st->v_d,      (size_t)N * E * 4);

  uint64_t rs = 0xE0ECA2026ULL;
  __half *hw = (__half *)malloc((size_t)F * E * 2);
  float  *hv = (float *)malloc(E * 4);
  for (size_t i = 0; i < (size_t)3 * E * E; i++) hw[i] = __float2half_rn(prng_next(&rs) * 0.03f);
  cudaMemcpy(st->dit_w.w_qkv, hw, (size_t)3 * E * E * 2, cudaMemcpyHostToDevice);
  for (size_t i = 0; i < (size_t)E * E; i++) hw[i] = __float2half_rn(prng_next(&rs) * 0.03f);
  cudaMemcpy(st->dit_w.w_proj, hw, (size_t)E * E * 2, cudaMemcpyHostToDevice);
  for (size_t i = 0; i < (size_t)F * E; i++) hw[i] = __float2half_rn(prng_next(&rs) * 0.03f);
  cudaMemcpy(st->dit_w.w_fc1, hw, (size_t)F * E * 2, cudaMemcpyHostToDevice);
  for (size_t i = 0; i < (size_t)E * F; i++) hw[i] = __float2half_rn(prng_next(&rs) * 0.03f);
  cudaMemcpy(st->dit_w.w_fc2, hw, (size_t)E * F * 2, cudaMemcpyHostToDevice);
  for (int i = 0; i < E; i++) hv[i] = 0.04f;
  cudaMemcpy(st->dit_w.shift_msa, hv, E * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(st->dit_w.scale_msa, hv, E * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(st->dit_w.shift_mlp, hv, E * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(st->dit_w.scale_mlp, hv, E * 4, cudaMemcpyHostToDevice);
  for (int i = 0; i < E; i++) hv[i] = 0.25f;
  cudaMemcpy(st->dit_w.gate_msa, hv, E * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(st->dit_w.gate_mlp, hv, E * 4, cudaMemcpyHostToDevice);
  free(hw); free(hv);

  ec_flow_init(&st->flow_graph);
  ec_registry_init(&st->registry, st->models_dir);
  if (st->models_dir[0]) ec_registry_scan(&st->registry);

  char cfg_storage[512], cfg_recipes[512], cw_cache[512];
  snprintf(cfg_storage, sizeof cfg_storage, "%s/storage_roots.json", st->artifact_dir);
  snprintf(cfg_recipes, sizeof cfg_recipes, "%s/recipes_user.json", st->artifact_dir);
  snprintf(cw_cache,    sizeof cw_cache,    "%s/cowrie_cache",       st->artifact_dir);
  ec_cowrie_init();
  ec_cowrie_set_cache_dir(cw_cache);
  ec_storage_init(&st->vault, st->models_dir, cfg_storage);
  ec_recipes_init(&st->recipes, cfg_recipes);
  studio_try_init_all_engines_locked(st);

  st->initialized = 1;
  return 0;
}

extern "C" ec_studio_t *ec_studio_create(const char *artifact_dir, const char *models_dir, size_t vram_budget_mb) {
  ec_studio_t *st = (ec_studio_t *)calloc(1, sizeof(ec_studio_t));
  if (!st) return NULL;
  if (ec_studio_init_impl(st, artifact_dir, models_dir, vram_budget_mb) != 0) { free(st); return NULL; }
  return st;
}

extern "C" void ec_studio_destroy(ec_studio_t *st) {
  if (!st) return;
  if (st->initialized) {
    pthread_mutex_lock(&st->gpu_mu);
    if (st->flux2)    { ec_flux2_destroy(st->flux2);       st->flux2 = NULL; }
    if (st->zimage)   { ec_zimage_destroy(st->zimage);     st->zimage = NULL; }
    if (st->acestep)  { ec_acestep_destroy(st->acestep);   st->acestep = NULL; }
    if (st->trellis2) { ec_trellis2_destroy(st->trellis2); st->trellis2 = NULL; }
    if (st->wan22)    { ec_wan22_destroy(st->wan22);       st->wan22 = NULL; }
    if (st->ltx2)     { ec_ltx2_destroy(st->ltx2);         st->ltx2 = NULL; }
    ec_registry_free(&st->registry);
    ec_flow_free(&st->flow_graph);
    ec_vram_pool_free(&st->vram_pool);
    ec_dit_ws_free(&st->dit_ws);
    cudaFree(st->dit_w.w_qkv); cudaFree(st->dit_w.w_proj); cudaFree(st->dit_w.w_fc1); cudaFree(st->dit_w.w_fc2);
    cudaFree(st->dit_w.shift_msa); cudaFree(st->dit_w.scale_msa); cudaFree(st->dit_w.gate_msa);
    cudaFree(st->dit_w.shift_mlp); cudaFree(st->dit_w.scale_mlp); cudaFree(st->dit_w.gate_mlp);
    cudaFree(st->coords_d); cudaFree(st->x_d); cudaFree(st->v_d);
    cudaStreamDestroy(st->compute_stream);
    pthread_mutex_unlock(&st->gpu_mu);
    pthread_mutex_destroy(&st->gpu_mu);
  }
  free(st);
}

extern "C" void ec_studio_get_info(const ec_studio_t *st, ec_studio_info_t *out) {
  memset(out, 0, sizeof *out);
  if (!st) return;
  snprintf(out->artifact_dir, sizeof out->artifact_dir, "%s", st->artifact_dir);
  snprintf(out->models_dir, sizeof out->models_dir, "%s", st->models_dir);
  snprintf(out->gpu_name, sizeof out->gpu_name, "%s", st->gpu_name);
  out->sm_major = st->sm_major;
  out->sm_minor = st->sm_minor;
  out->vram_total_gb = st->vram_total_gb;
  size_t free_b = 0, tot_b = 0;
  if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess) {
    out->vram_free_mb = (double)free_b / (1024.0 * 1024.0);
  }
  out->vram_pool_used_mb = (double)st->vram_pool.used_bytes / (1024.0 * 1024.0);
  out->vram_pool_budget_mb = (double)st->vram_pool.budget_bytes / (1024.0 * 1024.0);
  out->engines_resident_mask = st->active_engine_mask;
  out->models_discovered = st->registry.n_models;
  out->photoreal_ready = (st->flux2    && ec_flux2_is_ready(st->flux2))       ? 1 : 0;
  out->zimage_ready    = (st->zimage   && ec_zimage_is_ready(st->zimage))     ? 1 : 0;
  out->acestep_ready   = (st->acestep  && ec_acestep_is_ready(st->acestep))   ? 1 : 0;
  out->trellis2_ready  = (st->trellis2 && ec_trellis2_is_ready(st->trellis2)) ? 1 : 0;
  out->wan22_ready     = (st->wan22    && ec_wan22_is_ready(st->wan22))       ? 1 : 0;
  out->ltx2_ready      = (st->ltx2     && ec_ltx2_is_ready(st->ltx2))         ? 1 : 0;
  out->storage_roots = st->vault.n_roots;
  out->recipes_count = st->recipes.count;
  for (int i = 0; i < st->vault.n_assets; i++) {
    if (st->vault.assets[i].kind == EC_ASSET_LORA) out->loras_discovered++;
    else if (st->vault.assets[i].kind == EC_ASSET_CHECKPOINT) out->checkpoints_discovered++;
  }
  ec_cowrie_stats_t cws;
  ec_cowrie_get_stats(&cws);
  out->cowrie_enabled       = cws.enabled;
  out->cowrie_shani         = cws.shani;
  out->cowrie_sp_hits       = cws.sp_hits;
  out->cowrie_sp_saves      = cws.sp_saves;
  out->cowrie_sp_mismatches = cws.sp_mismatches;
  out->cowrie_trust_entries = cws.trust_entries;
}

typedef struct {
  ec_progress_cb_t cb;
  void            *cb_ctx;
} fx_cb_bridge_t;

static int fx_progress_bridge(int step, int total, const char *stage, void *ctx) {
  fx_cb_bridge_t *b = (fx_cb_bridge_t *)ctx;
  return (b && b->cb) ? b->cb(step, total, 0, stage, b->cb_ctx) : 0;
}
static int tr_ref_progress_bridge(int step, int total, const char *stage, void *ctx);
static int tr_flow_progress_bridge(int step, int total, const char *stage, void *ctx);

static int resolve_studio_media_path(const ec_studio_t *st, const char *raw, char *out, size_t cap) {
  out[0] = '\0';
  if (!raw || !raw[0]) return -1;
  if (!strncmp(raw, "/artifacts/", 11)) {
    const char *bn = raw + 11;
    if (strstr(bn, "..") || strchr(bn, '/')) return -1;
    snprintf(out, cap, "%s/%s", st->artifact_dir, bn);
    return access(out, R_OK) == 0 ? 0 : -1;
  }
  if (raw[0] == '/') {
    snprintf(out, cap, "%s", raw);
    return access(out, R_OK) == 0 ? 0 : -1;
  }
  if (!strstr(raw, "..") && !strchr(raw, '/')) {
    snprintf(out, cap, "%s/%s", st->artifact_dir, raw);
    return access(out, R_OK) == 0 ? 0 : -1;
  }
  return -1;
}

static int resolve_lora_stack(
    const ec_studio_t *st, const ec_studio_lora_t *in, int n_in, ec_lora_spec_t out[EC_LORA_MAX_STACK]) {
  int nl = 0;
  for (int i = 0; i < n_in && nl < EC_LORA_MAX_STACK; i++) {
    if (!in[i].path[0]) continue;
    const ec_storage_asset_t *la = ec_storage_find(&st->vault, EC_ASSET_LORA, in[i].path);
    snprintf(out[nl].path, sizeof out[nl].path, "%s", la ? la->path : in[i].path);
    out[nl].strength = in[i].strength;
    nl++;
  }
  return nl;
}

extern "C" int ec_studio_gen_image_v8(
    ec_studio_t *st, const ec_studio_image_v8_t *req,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized || !req) return -1;

  // Route Tongyi-MAI Z-Image-Turbo (6.15B S3-DiT) when explicitly selected and no Img2Img/Inpaint input!
  int want_zimage = (req->photoreal && st->zimage && ec_zimage_is_ready(st->zimage) &&
                     req->checkpoint_query &&
                     (ci_substr(req->checkpoint_query, "z_image") || ci_substr(req->checkpoint_query, "zimage")) &&
                     !(req->init_image_path && req->init_image_path[0]) &&
                     !(req->mask_image_path && req->mask_image_path[0]));
  if (want_zimage) {
    ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
    int nl = resolve_lora_stack(st, req->loras, req->n_loras, lstack);
    int W = (req->width  >= 512) ? req->width  : 1024;
    int H = (req->height >= 512) ? req->height : 1024;
    if (req->tile_upscale) {
      W = (W * 3) / 2; H = (H * 3) / 2;
      if (W > 1536) W = 1536; if (H > 1536) H = 1536;
    }
    ec_zimage_req_t zreq = {};
    zreq.prompt = req->prompt;
    zreq.seed = req->seed;
    zreq.width = W; zreq.height = H;
    zreq.steps = (req->steps >= 4 && req->steps <= 20) ? req->steps : 8;
    zreq.shift = 3.0f;
    zreq.loras = nl > 0 ? lstack : NULL;
    zreq.n_loras = nl;
    zreq.tile_upscale = req->tile_upscale;

    uint64_t h = hash_prompt_seed(req->prompt, req->seed);
    h = hash_mix_str(h, req->checkpoint_query);
    for (int i = 0; i < nl; i++) h = hash_mix_str(h, lstack[i].path);
    h ^= ((uint64_t)W << 32) ^ ((uint64_t)H << 16) ^ (uint64_t)zreq.steps;
    uint32_t tag32 = next_artifact_hash32(st, h);
    snprintf(res->file_main, sizeof res->file_main, "%s/zimage_%u_%08x.png", st->artifact_dir, req->seed, tag32);

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_zimage_res_t zres = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 5);
    int rc = ec_zimage_generate_png(st->zimage, &zreq, res->file_main, cb ? fx_progress_bridge : NULL, &bridge, &zres);
    pthread_mutex_unlock(&st->gpu_mu);
    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && zres.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "z_image_turbo_6b");
      snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", zres.checkpoint_used);
      res->width = zres.width; res->height = zres.height;
      res->steps_run = zres.steps_run; res->lora_tensors_fused = zres.lora_tensors_fused;
      res->text_enc_ms = zres.text_enc_ms; res->dit_ms = zres.dit_ms;
      res->vae_ms = zres.vae_ms; res->gpu_ms = zres.total_ms;
      return 0;
    }
    return -1;
  }

  if (req->photoreal && st->flux2 && ec_flux2_is_ready(st->flux2)) {
    char ckpt_path[512] = {0};
    if (req->checkpoint_query && req->checkpoint_query[0]) {
      const ec_storage_asset_t *a = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, req->checkpoint_query);
      snprintf(ckpt_path, sizeof ckpt_path, "%s", a ? a->path : req->checkpoint_query);
    }
    ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
    int nl = resolve_lora_stack(st, req->loras, req->n_loras, lstack);

    int W = (req->width  >= 256) ? req->width  : 1024;
    int H = (req->height >= 256) ? req->height : 1024;
    uint8_t *init_rgb = NULL;
    int init_w = 0, init_h = 0;
    float *mask_gray = NULL;

    char ipath[512] = {0}, mpath[512] = {0};
    if (req->init_image_path && resolve_studio_media_path(st, req->init_image_path, ipath, sizeof ipath) == 0) {
      ec_load_image_rgb(ipath, &init_rgb, &init_w, &init_h);
    }
    int has_outpaint = (init_rgb && (req->outpaint_pad[0] > 0 || req->outpaint_pad[1] > 0 ||
                                     req->outpaint_pad[2] > 0 || req->outpaint_pad[3] > 0));
    if (has_outpaint) {
      uint8_t *pad_rgb = NULL; float *pad_mask = NULL;
      if (ec_rgb_pad_outpaint(init_rgb, init_w, init_h,
                              req->outpaint_pad[0], req->outpaint_pad[1], req->outpaint_pad[2], req->outpaint_pad[3],
                              W, H, &pad_rgb, &pad_mask) == 0) {
        free(init_rgb); init_rgb = pad_rgb; init_w = W; init_h = H; mask_gray = pad_mask;
      }
    } else if (init_rgb && req->mask_image_path && resolve_studio_media_path(st, req->mask_image_path, mpath, sizeof mpath) == 0) {
      ec_load_mask_gray(mpath, init_w, init_h, &mask_gray);
    }

    ec_flux2_req_t freq = {};
    freq.prompt = req->prompt;
    freq.neg_prompt = req->neg_prompt;
    freq.seed = req->seed;
    freq.width = W;
    freq.height = H;
    freq.steps = (req->steps >= 1 && req->steps <= 64) ? req->steps : 4;
    freq.cfg_scale = req->cfg_scale;
    freq.checkpoint_override = ckpt_path[0] ? ckpt_path : NULL;
    freq.loras = nl > 0 ? lstack : NULL;
    freq.n_loras = nl;
    freq.init_rgb = init_rgb;
    freq.init_w = init_w;
    freq.init_h = init_h;
    freq.mask_gray = mask_gray;
    freq.denoise = req->denoise;
    freq.tile_upscale = req->tile_upscale;
    freq.tier = req->tier;

    uint64_t tag = hash_prompt_seed(req->prompt, req->seed);
    tag = hash_mix_str(tag, ckpt_path);
    tag = hash_mix_str(tag, req->neg_prompt);
    for (int i = 0; i < nl; i++) tag = hash_mix_str(tag, lstack[i].path);
    tag ^= ((uint64_t)W << 32) ^ ((uint64_t)H << 16) ^ ((uint64_t)freq.steps << 8) ^ (uint64_t)(int)(req->cfg_scale * 100.0f);
    if (init_rgb) tag ^= 0x12101210ULL;
    if (mask_gray) tag ^= 0xA1B2C3D4ULL;
    if (req->tile_upscale) tag ^= 0x20002000ULL;
    uint32_t tag32 = next_artifact_hash32(st, tag);
    snprintf(res->file_main, sizeof res->file_main, "%s/flux2_%u_%08x.png",
             st->artifact_dir, req->seed, tag32);

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_flux2_res_t fres = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 1);
    int rc = ec_flux2_generate_png(st->flux2, &freq, res->file_main, cb ? fx_progress_bridge : NULL, &bridge, &fres);
    pthread_mutex_unlock(&st->gpu_mu);
    free(init_rgb); free(mask_gray);
    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && fres.ok) {
      int is_base = ci_substr(fres.checkpoint_used, "base");
      snprintf(res->engine_mode, sizeof res->engine_mode, "%s",
               has_outpaint ? "flux2_outpaint" : (mask_gray ? "flux2_inpaint" : (init_rgb ? "flux2_img2img" : (is_base ? "flux2_base_4b" : "flux2_photoreal"))));
      snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", fres.checkpoint_used);
      res->width = fres.width; res->height = fres.height;
      res->steps_run = fres.steps_run; res->lora_tensors_fused = fres.lora_tensors_fused;
      res->text_enc_ms = fres.text_enc_ms; res->dit_ms = fres.dit_ms;
      res->vae_ms = fres.vae_ms; res->gpu_ms = fres.total_ms;
      return 0;
    }
    return -1;
  }

  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return ec_studio_gen_image(st, req->prompt, req->seed, req->width, req->height, req->steps, req->tier, cb, cb_ctx, res);
}

extern "C" int ec_studio_gen_3d_v8(
    ec_studio_t *st, const ec_studio_3d_v8_t *req,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized || !req) return -1;

  if (req->full_model && st->trellis2 && ec_trellis2_is_ready(st->trellis2)) {
    uint8_t *views[4] = {0};
    int vw[4] = {0}, vh[4] = {0}, n_loaded = 0;
    for (int i = 0; i < req->n_input_images && i < 4; i++) {
      char pbuf[512];
      if (req->input_images[i] && resolve_studio_media_path(st, req->input_images[i], pbuf, sizeof pbuf) == 0) {
        if (ec_load_image_rgb(pbuf, &views[n_loaded], &vw[n_loaded], &vh[n_loaded]) == 0) {
          // NOTE: Do NOT call legacy radial ec_rgb_isolate_foreground() here — ec_trellis2's
          // Coons-patch 2D bilinear studio backdrop matte + hole punch + largest-CC handles
          // background isolation cleanly without carving a circular rad=0.35 halo into the photo!
          n_loaded++;
        }
      }
    }

    uint64_t tag = hash_prompt_seed(req->prompt, req->seed) ^ ((uint64_t)n_loaded * 0x9E3779B9ULL);
    uint32_t tag32 = next_artifact_hash32(st, tag);
    snprintf(res->file_main, sizeof res->file_main, "%s/trellis2_%u_%08x.glb",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->file_aux, sizeof res->file_aux, "%s/trellis2_%u_%08x.ply",
             st->artifact_dir, req->seed, tag32);

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    double ref_ms = 0.0;
    if (n_loaded == 0 && ((st->zimage && ec_zimage_is_ready(st->zimage)) || (st->flux2 && ec_flux2_is_ready(st->flux2)))) {
      char hero_prompt[1536], ref_png[256];
      // v16 Studio 3D Reference Standard: 1024x1024 Telephoto 3/4 Softbox floating cutout on flat shadowless neutral gray!
      snprintf(hero_prompt, sizeof hero_prompt,
          "Single isolated 3D turntable product cutout of %s, three-quarter front-elevated view, 100mm telephoto lens with zero distortion, full object floating cleanly centered with 18%% empty margin on all four sides, even shadowless neutral softbox lighting, flat solid uniform neutral medium-gray #888888 void background with zero gradient and zero cast shadows, crisp true material albedo, no studio wall, no table, no floor, no horizon, no props, no text",
          (req->prompt && req->prompt[0]) ? req->prompt : "sculptural matte-white porcelain teapot with bamboo handle");
      snprintf(ref_png, sizeof ref_png, "%s/trellis2_%u_%08x_ref.png",
               st->artifact_dir, req->seed, tag32);
      int rc_ref = -1;
      if (st->zimage && ec_zimage_is_ready(st->zimage)) {
        ec_zimage_req_t zreq = {};
        zreq.prompt = hero_prompt; zreq.seed = req->seed; zreq.width = 1024; zreq.height = 1024; zreq.steps = 8; zreq.shift = 3.0f;
        ec_zimage_res_t zres = {};
        pthread_mutex_lock(&st->gpu_mu);
        studio_evict_other_engines_locked(st, 5);
        rc_ref = ec_zimage_generate_rgb(st->zimage, &zreq, &views[0], ref_png,
                                        cb ? tr_ref_progress_bridge : NULL, &bridge, &zres);
        ec_zimage_unload_vram(st->zimage);
        pthread_mutex_unlock(&st->gpu_mu);
        if (rc_ref == 0 && views[0]) { vw[0] = zres.width; vh[0] = zres.height; n_loaded = 1; ref_ms = zres.total_ms; }
      } else {
        ec_flux2_req_t freq = {};
        freq.prompt = hero_prompt; freq.seed = req->seed; freq.width = 1024; freq.height = 1024; freq.steps = 4; freq.tier = EC_TIER_EXACT;
        ec_flux2_res_t fres = {};
        pthread_mutex_lock(&st->gpu_mu);
        studio_evict_other_engines_locked(st, 1);
        rc_ref = ec_flux2_generate_rgb(st->flux2, &freq, &views[0], ref_png,
                                       cb ? tr_ref_progress_bridge : NULL, &bridge, &fres);
        ec_flux2_unload_vram(st->flux2);
        pthread_mutex_unlock(&st->gpu_mu);
        if (rc_ref == 0 && views[0]) { vw[0] = fres.width; vh[0] = fres.height; n_loaded = 1; ref_ms = fres.total_ms; }
      }
      if (rc_ref != 0 || !views[0]) {
        free(views[0]);
        if (rc_ref == -2) { res->cancelled = 1; return 0; }
        return -1;
      }
    }

    if (n_loaded > 0) {
      ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
      int nl = resolve_lora_stack(st, req->loras, req->n_loras, lstack);
      ec_trellis2_req_t treq = {};
      treq.rgb = views[0]; treq.img_w = vw[0]; treq.img_h = vh[0];
      for (int i = 1; i < n_loaded; i++) {
        treq.extra_rgbs[i - 1] = views[i]; treq.extra_w[i - 1] = vw[i]; treq.extra_h[i - 1] = vh[i];
      }
      treq.n_extra_views = n_loaded - 1;
      treq.seed = req->seed;
      treq.steps = (req->steps >= 2 && req->steps <= 50) ? req->steps : 12;
      treq.cfg_scale = (req->cfg_scale > 0.0f) ? req->cfg_scale : 7.5f;
      treq.rescale_t = (req->rescale_t > 0.0f) ? req->rescale_t : 5.0f;
      memcpy(treq.base_color, req->base_color, sizeof treq.base_color);
      treq.loras = nl > 0 ? lstack : NULL;
      treq.n_loras = nl;

      ec_trellis2_res_t tres = {};
      pthread_mutex_lock(&st->gpu_mu);
      studio_evict_other_engines_locked(st, 3);
      int rc_3d = ec_trellis2_generate_3d(st->trellis2, &treq, res->file_main, res->file_aux,
                                          cb ? tr_flow_progress_bridge : NULL, &bridge, &tres);
      pthread_mutex_unlock(&st->gpu_mu);
      for (int i = 0; i < n_loaded; i++) free(views[i]);
      if (rc_3d == -2) { res->cancelled = 1; return 0; }
      if (rc_3d == 0 && tres.ok) {
        snprintf(res->engine_mode, sizeof res->engine_mode, "%s",
                 req->n_input_images > 1 ? "trellis2_multiview" : (req->n_input_images == 1 ? "trellis2_photo" : "trellis2_1.3b"));
        res->n_verts = tres.n_verts; res->n_tris = tres.n_tris; res->n_splats = tres.n_splats;
        res->steps_run = tres.steps_run;
        res->lora_tensors_fused = tres.lora_tensors_fused;
        res->text_enc_ms = ref_ms; res->dit_ms = tres.dino_ms + tres.flow_ms;
        res->vae_ms = tres.dec_ms + tres.mesh_ms; res->gpu_ms = ref_ms + tres.total_ms;
        return 0;
      }
      return -1;
    }
  }
  int rc = ec_studio_gen_3d(st, req->prompt, req->seed, req->grid_res, req->tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return rc;
}

extern "C" int ec_studio_gen_music_v8(
    ec_studio_t *st, const ec_studio_music_v8_t *req,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized || !req) return -1;

  if (req->full_model && st->acestep && ec_acestep_is_ready(st->acestep)) {
    char ckpt_path[512] = {0};
    if (req->checkpoint_query && req->checkpoint_query[0]) {
      const ec_storage_asset_t *a = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, req->checkpoint_query);
      snprintf(ckpt_path, sizeof ckpt_path, "%s", a ? a->path : req->checkpoint_query);
    }
    int is_sft = ci_substr(ckpt_path, "sft") || (req->checkpoint_query && ci_substr(req->checkpoint_query, "sft"));
    uint64_t tag = hash_prompt_seed(req->prompt, req->seed) ^ ((uint64_t)req->stem_mode << 12);
    tag = hash_mix_str(tag, ckpt_path);
    tag = hash_mix_str(tag, req->lyrics);
    uint32_t tag32 = next_artifact_hash32(st, tag);
    snprintf(res->file_main, sizeof res->file_main, "%s/acestep_%u_%08x_master.wav",
             st->artifact_dir, req->seed, tag32);

    float *src_pcm = NULL; int src_samples = 0;
    char apath[512] = {0};
    if (req->source_audio_path && resolve_studio_media_path(st, req->source_audio_path, apath, sizeof apath) == 0) {
      ec_load_wav_stereo_f32(apath, 48000, &src_pcm, &src_samples);
    }
    ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
    int nl = resolve_lora_stack(st, req->loras, req->n_loras, lstack);

    ec_acestep_req_t areq = {};
    areq.prompt = (req->prompt && req->prompt[0]) ? req->prompt : "warm jazz house groove, electric rhodes piano, crisp percussion, analog bassline";
    areq.lyrics = (req->lyrics && req->lyrics[0]) ? req->lyrics : "";
    areq.seed = req->seed;
    areq.bpm = (req->bpm >= 40 && req->bpm <= 240) ? req->bpm : 120;
    areq.duration_sec = (req->duration_sec >= 4.0f && req->duration_sec <= 120.0f) ? req->duration_sec : 20.0f;
    areq.steps = (req->steps >= 1 && req->steps <= 64) ? req->steps : (is_sft ? 32 : 8);
    areq.shift = (req->shift >= 0.5f && req->shift <= 6.0f) ? req->shift : 3.0f;
    areq.cfg_scale = req->cfg_scale;
    areq.checkpoint_override = ckpt_path[0] ? ckpt_path : NULL;
    areq.key_scale = req->key_scale;
    areq.time_sig = req->time_sig;
    areq.vocal_mode = req->vocal_mode;
    areq.audio_mode = req->audio_mode;
    areq.source_pcm_lr = src_pcm;
    areq.source_samples = src_samples;
    areq.denoise = req->denoise;
    areq.repaint_start_sec = req->repaint_start_sec;
    areq.repaint_end_sec = req->repaint_end_sec;
    areq.stem_mode = req->stem_mode;
    areq.loras = nl > 0 ? lstack : NULL;
    areq.n_loras = nl;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_acestep_res_t ares = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 2);
    int rc = ec_acestep_generate_wav(st->acestep, &areq, res->file_main, NULL, NULL,
                                     cb ? fx_progress_bridge : NULL, &bridge, &ares);
    pthread_mutex_unlock(&st->gpu_mu);
    free(src_pcm);

    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && ares.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "%s", is_sft ? "acestep_1.5_sft" : "acestep_1.5_turbo");
      snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", ckpt_path[0] ? ckpt_path : "acestep_v15_turbo_bf16.safetensors");
      res->sample_rate = ares.sample_rate;
      res->frames = ares.num_samples;
      res->steps_run = ares.steps_run;
      res->lora_tensors_fused = ares.lora_tensors_fused;
      res->n_stems = ares.n_stems;
      for (int i = 0; i < ares.n_stems && i < 8; i++) {
        snprintf(res->stem_files[i], sizeof res->stem_files[i], "%s", ares.stem_paths[i]);
        snprintf(res->stem_names[i], sizeof res->stem_names[i], "%s", ares.stem_names[i]);
      }
      res->text_enc_ms = ares.text_enc_ms + ares.cond_enc_ms;
      res->dit_ms = ares.dit_ms;
      res->vae_ms = ares.vae_ms;
      res->gpu_ms = ares.total_ms;
      return 0;
    }
    return -1;
  }
  int rc = ec_studio_gen_music(st, req->prompt, req->seed, req->bpm, req->duration_sec, req->tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  res->sample_rate = 44100;
  if (rc == 0 && !res->cancelled) {
    if (req->stem_mode == 0) {
      res->n_stems = 0;
      for (int i = 0; i < 8; i++) { res->stem_files[i][0] = '\0'; res->stem_names[i][0] = '\0'; }
    } else if (req->stem_mode == 2 || req->stem_mode == 8) {
      float *pcm = NULL; int ns = 0;
      if (ec_load_wav_stereo_f32(res->file_main, 44100, &pcm, &ns) == 0 && pcm && ns > 0) {
        char pfx[240];
        snprintf(pfx, sizeof pfx, "%s/music_%u_v8", st->artifact_dir, req->seed);
        for (int i = 0; i < 8; i++) { res->stem_files[i][0] = '\0'; res->stem_names[i][0] = '\0'; }
        ec_audio_split_stems(pcm, ns, 44100, req->stem_mode, pfx, res->stem_files, res->stem_names, &res->n_stems);
        free(pcm);
      }
    } else {
      static const char *L4[4] = { "drums", "bass", "harmony", "lead" };
      res->n_stems = 4;
      for (int i = 0; i < 4; i++) snprintf(res->stem_names[i], sizeof res->stem_names[i], "%s", L4[i]);
    }
  }
  return rc;
}

extern "C" int ec_studio_gen_video_v8(
    ec_studio_t *st, const ec_studio_video_v8_t *req,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized || !req) return -1;

  // v17.3 Joint Synchronized Audio+Video Frontier: Lightricks LTX-2.3 (22B Distilled FP8)
  int want_ltx2 = (req->checkpoint_query && ci_substr(req->checkpoint_query, "ltx")) ? 1 : 0;
  if (req->full_model && st->ltx2 && ec_ltx2_is_ready(st->ltx2) &&
      (want_ltx2 || (!st->wan22 && !req->start_frame_path && !req->end_frame_path))) {
    int W = (req->width  >= 256) ? ((req->width  + 16) / 32) * 32 : 768;
    int H = (req->height >= 256) ? ((req->height + 16) / 32) * 32 : 512;
    int F = (req->n_frames >= 9) ? req->n_frames : 49;
    int afps = (req->fps >= 8 && req->fps <= 60) ? req->fps : 24;
    int asteps = (req->steps >= 2 && req->steps <= 8) ? req->steps : 8;

    uint64_t tag = hash_prompt_seed(req->prompt, req->seed) ^ 0x4C54583233ULL;
    tag ^= ((uint64_t)W << 32) ^ ((uint64_t)H << 16) ^ (uint64_t)F;
    uint32_t tag32 = next_artifact_hash32(st, tag);
    snprintf(res->file_mp4, sizeof res->file_mp4, "%s/ltx23_%u_%08x%s.mp4",
             st->artifact_dir, req->seed, tag32, req->with_audio ? "_av" : "");
    snprintf(res->file_aux, sizeof res->file_aux, "%s/ltx23_%u_%08x_kf0.png",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->file_last, sizeof res->file_last, "%s/ltx23_%u_%08x_last.png",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->stem_files[0], sizeof res->stem_files[0], "%s/ltx23_%u_%08x_48k.wav",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->file_main, sizeof res->file_main, "%s", res->file_mp4);

    ec_ltx2_req_t lreq = {};
    lreq.prompt = (req->prompt && req->prompt[0]) ? req->prompt : "Dramatic ocean waves crashing onto volcanic basalt cliffs at sunset with roaring surf and mist";
    lreq.seed = req->seed;
    lreq.width = W; lreq.height = H;
    lreq.n_frames = F; lreq.fps = afps;
    lreq.steps = asteps; lreq.with_audio = req->with_audio ? 1 : 0;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_ltx2_stats_t lstats = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 6);
    int rc = ec_ltx2_generate(st->ltx2, &lreq, res->file_mp4,
                              req->with_audio ? res->stem_files[0] : NULL,
                              res->file_aux, res->file_last,
                              cb ? fx_progress_bridge : NULL, &bridge, &lstats);
    pthread_mutex_unlock(&st->gpu_mu);
    if (rc == 0) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "ltx2.3_22b_joint_av");
      snprintf(res->encoder_used, sizeof res->encoder_used, "%s", lstats.encoder_used);
      res->width = lstats.width; res->height = lstats.height; res->frames = lstats.n_frames;
      res->steps_run = lstats.steps_run;
      res->sample_rate = lstats.sample_rate;
      res->text_enc_ms = lstats.text_enc_ms; res->dit_ms = lstats.dit_ms;
      res->vae_ms = lstats.video_vae_ms + lstats.audio_vae_ms;
      res->gpu_ms = lstats.total_ms;
      res->fps = (double)lstats.n_frames / ((lstats.total_ms > 1.0 ? lstats.total_ms : 1.0) * 1e-3);
      return 0;
    }
    return -1;
  }

  if (req->full_model && st->wan22 && ec_wan22_is_ready(st->wan22)) {
    int W = (req->width  >= 320) ? ((req->width  + 31) & ~31) : 1280;
    if (W > 1280) W = 1280;
    int H = (req->height >= 320) ? ((req->height + 31) & ~31) : 704;
    if (H > 1024) H = 1024;
    int req_f = (req->n_frames >= 5) ? req->n_frames : 49;
    int F_lat = (req_f - 1) / 4 + 1;
    if (F_lat < 2) F_lat = 2; if (F_lat > 31) F_lat = 31;
    int actual_f = 1 + 4 * (F_lat - 1);
    int afps = (req->fps >= 8 && req->fps <= 60) ? req->fps : 24;
    int asteps = (req->steps >= 4 && req->steps <= 50) ? req->steps : 30;

    char wav_path[256] = {0};
    double audio_ms = 0.0;
    if (req->with_audio) {
      if (cb && cb(0, asteps + 4, 0, "acestep_synced_audio", cb_ctx) != 0) { res->cancelled = 1; return 0; }
      ec_studio_result_t ares = {};
      float dur = (float)actual_f / (float)afps;
      if (dur < 4.0f) dur = 4.0f;
      int use_acestep = (st->acestep && ec_acestep_is_ready(st->acestep)) ? 1 : 0;
      if (ec_studio_gen_music_ex(st, req->prompt, "", req->seed + 17, 116, dur, req->tier, use_acestep, NULL, NULL, &ares) == 0 && !ares.cancelled) {
        snprintf(wav_path, sizeof wav_path, "%s", ares.file_main);
        audio_ms = ares.gpu_ms;
      }
    }

    uint8_t *start_rgb = NULL, *end_rgb = NULL;
    int sw = 0, sh = 0, ew = 0, eh = 0, used_hero_i2v = 0;
    double hero_ms = 0.0;
    char spath[512] = {0}, epath[512] = {0};
    if (req->start_frame_path && resolve_studio_media_path(st, req->start_frame_path, spath, sizeof spath) == 0) {
      ec_load_image_rgb(spath, &start_rgb, &sw, &sh);
    }
    if (req->end_frame_path && resolve_studio_media_path(st, req->end_frame_path, epath, sizeof epath) == 0) {
      ec_load_image_rgb(epath, &end_rgb, &ew, &eh);
    }
    // Optional v16 Hero-Keyframe -> Wan 2.2 I2V Cinema Mode: synthesize ultra-sharp photoreal opening frame first!
    const char *env_hero = getenv("EC_VIDEO_HERO_I2V");
    int want_hero = (!start_rgb && (req->hero_keyframe_i2v || (env_hero && env_hero[0] == '1')));
    if (want_hero && ((st->zimage && ec_zimage_is_ready(st->zimage)) || (st->flux2 && ec_flux2_is_ready(st->flux2)))) {
      if (cb && cb(0, asteps + 4, 0, "hero_keyframe_gen", cb_ctx) != 0) { res->cancelled = 1; return 0; }
      const char *hp = (req->prompt && req->prompt[0]) ? req->prompt : "35mm cinema still frame, natural lighting";
      int rc_h = -1;
      if (st->zimage && ec_zimage_is_ready(st->zimage)) {
        ec_zimage_req_t zreq = {};
        // Z-Image S3-DiT is trained at ~1 MP (e.g. 1280x720 or 1024x576); generate hero keyframe at >=1024 long edge
        // matching the target video aspect ratio, then Lanczos-3 downsample crisply into Wan 2.2 I2V!
        int zw = W, zh = H;
        if (zw < 1024 && zh < 1024) {
          if (zw >= zh) { zh = ((H * 1024 / W + 15) & ~15); zw = 1024; }
          else          { zw = ((W * 1024 / H + 15) & ~15); zh = 1024; }
        }
        if (zw < 512) zw = 512; if (zh < 512) zh = 512;
        zreq.prompt = hp; zreq.seed = req->seed; zreq.width = zw; zreq.height = zh; zreq.steps = 8; zreq.shift = 3.0f;
        ec_zimage_res_t zres = {};
        pthread_mutex_lock(&st->gpu_mu);
        studio_evict_other_engines_locked(st, 5);
        rc_h = ec_zimage_generate_rgb(st->zimage, &zreq, &start_rgb, NULL, NULL, NULL, &zres);
        ec_zimage_unload_vram(st->zimage);
        pthread_mutex_unlock(&st->gpu_mu);
        if (rc_h == 0 && start_rgb) { sw = zres.width; sh = zres.height; hero_ms = zres.total_ms; used_hero_i2v = 1; }
      } else {
        ec_flux2_req_t freq = {};
        freq.prompt = hp; freq.seed = req->seed; freq.width = W; freq.height = H; freq.steps = 4; freq.tier = EC_TIER_EXACT;
        ec_flux2_res_t fres = {};
        pthread_mutex_lock(&st->gpu_mu);
        studio_evict_other_engines_locked(st, 1);
        rc_h = ec_flux2_generate_rgb(st->flux2, &freq, &start_rgb, NULL, NULL, NULL, &fres);
        ec_flux2_unload_vram(st->flux2);
        pthread_mutex_unlock(&st->gpu_mu);
        if (rc_h == 0 && start_rgb) { sw = fres.width; sh = fres.height; hero_ms = fres.total_ms; used_hero_i2v = 1; }
      }
    }
    ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
    int nl = resolve_lora_stack(st, req->loras, req->n_loras, lstack);

    uint64_t tag = hash_prompt_seed(req->prompt, req->seed) ^ (start_rgb ? 0x1111ULL : 0ULL) ^ (end_rgb ? 0x2222ULL : 0ULL);
    tag ^= ((uint64_t)W << 32) ^ ((uint64_t)H << 16) ^ (uint64_t)actual_f;
    uint32_t tag32 = next_artifact_hash32(st, tag);
    snprintf(res->file_mp4, sizeof res->file_mp4, "%s/wan22_%u_%08x%s.mp4",
             st->artifact_dir, req->seed, tag32, req->with_audio ? "_av" : "");
    snprintf(res->file_aux, sizeof res->file_aux, "%s/wan22_%u_%08x_kf0.png",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->file_last, sizeof res->file_last, "%s/wan22_%u_%08x_last.png",
             st->artifact_dir, req->seed, tag32);
    snprintf(res->file_main, sizeof res->file_main, "%s", res->file_mp4);

    ec_wan22_req_t wreq = {};
    wreq.prompt = (req->prompt && req->prompt[0]) ? req->prompt : "Aerial drone footage flying slowly along black basalt sea cliffs in Iceland at sunset, 35mm cinema";
    wreq.neg_prompt = req->neg_prompt;
    wreq.seed = req->seed;
    wreq.width = W; wreq.height = H; wreq.num_frames = actual_f;
    wreq.fps = afps; wreq.steps = asteps;
    wreq.cfg_scale = (req->cfg_scale > 0.0f) ? req->cfg_scale : 4.0f;
    wreq.shift = req->shift; // 0 -> ec_wan22 picks 5.0f cleanly!
    wreq.start_rgb = start_rgb; wreq.start_w = sw; wreq.start_h = sh;
    wreq.end_rgb = end_rgb; wreq.end_w = ew; wreq.end_h = eh;
    wreq.denoise = req->denoise;
    wreq.camera_motion = req->camera_motion;
    wreq.loras = nl > 0 ? lstack : NULL;
    wreq.n_loras = nl;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_wan22_res_t wres = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 4);
    int rc = ec_wan22_generate_mp4_ex(st->wan22, &wreq, res->file_mp4, res->file_aux, res->file_last,
                                      wav_path[0] ? wav_path : NULL,
                                      cb ? fx_progress_bridge : NULL, &bridge, &wres);
    pthread_mutex_unlock(&st->gpu_mu);
    free(start_rgb); free(end_rgb);

    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && wres.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "%s",
               used_hero_i2v ? "wan2.2_hero_i2v" : ((start_rgb && end_rgb) ? "wan2.2_flf2v" : (start_rgb ? "wan2.2_i2v" : "wan2.2_ti2v_5b")));
      snprintf(res->encoder_used, sizeof res->encoder_used, "%s", wres.encoder_used);
      res->width = wres.width; res->height = wres.height; res->frames = wres.num_frames;
      res->steps_run = wres.steps_run;
      res->lora_tensors_fused = wres.lora_tensors_fused;
      res->text_enc_ms = wres.text_enc_ms + hero_ms; res->dit_ms = wres.dit_ms;
      res->vae_ms = wres.vae_ms; res->gpu_ms = wres.total_ms + audio_ms + hero_ms;
      res->fps = (double)wres.num_frames / ((wres.total_ms > 1.0 ? wres.total_ms : 1.0) * 1e-3);
      return 0;
    }
    return -1;
  }
  int rc = ec_studio_gen_video(st, req->prompt, req->seed, req->n_frames, req->width, req->height, req->fps, req->with_audio, req->tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return rc;
}

extern "C" int ec_studio_gen_image_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, int photoreal,
    const char *checkpoint_query,
    const ec_studio_lora_t *loras, int n_loras,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  ec_studio_image_v8_t req = {};
  req.prompt = prompt;
  req.seed = seed;
  req.width = width;
  req.height = height;
  req.steps = steps;
  req.tier = tier;
  req.photoreal = photoreal;
  req.checkpoint_query = checkpoint_query;
  req.loras = loras;
  req.n_loras = n_loras;
  return ec_studio_gen_image_v8(st, &req, cb, cb_ctx, res);
}

// ===========================================================================
// 1. Image Generation: Shifted RF + 2D RoPE DiT + MagCache + TensorCore Conv2D VAE -> PNG (Fast Smoke)
// ===========================================================================
extern "C" int ec_studio_gen_image(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;
  if (width < 64) width = 256; if (width > 1024) width = 512;
  if (height < 64) height = 256; if (height > 1024) height = 512;
  if (steps < 2) steps = 8; if (steps > 40) steps = 20;

  double t0 = st_now_ms();
  const int N = 256, E = st->dit_cfg.dim;
  float *x0 = (float *)malloc((size_t)N * E * 4);
  uint64_t rs = hash_prompt_seed(prompt, seed);
  for (int i = 0; i < N * E; i++) x0[i] = prng_next(&rs);

  float rgb16[3 * 256];
  float ts[44];
  ec_rf_shifted_schedule(steps, 3.0f, ts); // Z-Image / FLUX shifted RF schedule

  pthread_mutex_lock(&st->gpu_mu);
  ec_vram_acquire(&st->vram_pool, st->slot_dit_weights, st->compute_stream);
  ec_vram_prefetch_async(&st->vram_pool, st->slot_vae_weights);

  st->dit_cfg.ndims = 2;
  st->dit_cfg.axis_dims[0] = 32; st->dit_cfg.axis_dims[1] = 32; st->dit_cfg.axis_dims[2] = 0;
  st->dit_cfg.block_causal = 0;
  st->dit_cfg.modality = EC_MODALITY_IMAGE;

  int32_t coords[256 * 2];
  for (int i = 0; i < 256; i++) { coords[2 * i] = i / 16; coords[2 * i + 1] = i % 16; }
  cudaMemcpyAsync(st->coords_d, coords, sizeof coords, cudaMemcpyHostToDevice, st->compute_stream);
  cudaMemcpyAsync(st->x_d, x0, (size_t)N * E * 4, cudaMemcpyHostToDevice, st->compute_stream);

  ec_magcache_t mc = {};
  mc.n_steps = steps; mc.err_thresh = 0.045f; mc.max_consec_skip = 2;
  mc.warmup_steps = 2; mc.cooldown_steps = 2;

  for (int s = 0; s < steps; s++) {
    if (cb && cb(s, steps, res->steps_skipped, "dit_sample", cb_ctx) != 0) {
      res->cancelled = 1;
      break;
    }
    const float dt = ts[s + 1] - ts[s];
    int skip = (tier == EC_TIER_FAST) ? ec_magcache_should_skip(&mc, s, 0.018f) : 0;
    if (!skip) {
      cudaMemcpyAsync(st->v_d, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToDevice, st->compute_stream);
      ec_dit_block_forward(&st->dit_cfg, &st->dit_w, &st->dit_ws, st->v_d, st->coords_d, tier, st->compute_stream);
      res->steps_run++;
    } else {
      res->steps_skipped++;
    }
    ec_launch_flow_cfg_euler(st->x_d, st->v_d, nullptr, st->x_d, N * E, dt, 1.0f, st->compute_stream);
  }

  if (!res->cancelled) {
    // VAE decode using TensorCore im2col Conv2D!
    ec_vram_acquire(&st->vram_pool, st->slot_vae_weights, st->compute_stream);
    cudaMemcpyAsync(x0, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);

    float lat_h[8 * 256];
    for (int c = 0; c < 8; c++)
      for (int i = 0; i < 256; i++) lat_h[c * 256 + i] = x0[i * E + c];
    __half cw16[3 * 8 * 9];
    for (int i = 0; i < 3 * 8 * 9; i++) cw16[i] = __float2half_rn(prng_next(&rs) * 0.25f);

    float *lat_d, *gn_d, *rgb16_d, *y_tmp_d; __half *cw_d, *col_d;
    cudaMalloc(&lat_d, sizeof lat_h); cudaMalloc(&gn_d, sizeof lat_h);
    cudaMalloc(&rgb16_d, sizeof rgb16); cudaMalloc(&y_tmp_d, sizeof rgb16);
    cudaMalloc(&cw_d, sizeof cw16); cudaMalloc(&col_d, 256 * 8 * 9 * 2);
    cudaMemcpyAsync(lat_d, lat_h, sizeof lat_h, cudaMemcpyHostToDevice, st->compute_stream);
    cudaMemcpyAsync(cw_d, cw16, sizeof cw16, cudaMemcpyHostToDevice, st->compute_stream);

    ec_launch_groupnorm_silu(lat_d, nullptr, nullptr, gn_d, 8, 256, 4, 1e-6f, 1, st->compute_stream);
    ec_launch_conv2d_tc(gn_d, cw_d, nullptr, col_d, y_tmp_d, rgb16_d, 8, 3, 16, 16, 3, 3, 1, 1, st->compute_stream);
    cudaMemcpyAsync(rgb16, rgb16_d, sizeof rgb16, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    cudaFree(lat_d); cudaFree(gn_d); cudaFree(rgb16_d); cudaFree(y_tmp_d); cudaFree(cw_d); cudaFree(col_d);
  }
  res->gpu_ms = st_now_ms() - t0;
  pthread_mutex_unlock(&st->gpu_mu);
  free(x0);
  if (res->cancelled) return 0;

  if (cb) cb(steps, steps, res->steps_skipped, "encode_png", cb_ctx);

  // CPU PNG encoding runs OUTSIDE gpu_mu so other GPU requests run in parallel!
  uint8_t *rgb = (uint8_t *)malloc((size_t)width * height * 3u);
  const float hue_shift = (float)((hash_prompt_seed(prompt, seed) & 0xFFu)) / 255.0f * 6.283185f;
  for (int y = 0; y < height; y++) {
    float fy = (float)y / (float)(height - 1) * 15.0f;
    int y0 = (int)fy, y1 = (y0 < 15) ? (y0 + 1) : 15; float wy = fy - (float)y0;
    for (int x = 0; x < width; x++) {
      float fx = (float)x / (float)(width - 1) * 15.0f;
      int x0i = (int)fx, x1i = (x0i < 15) ? (x0i + 1) : 15; float wx = fx - (float)x0i;
      for (int c = 0; c < 3; c++) {
        const float *p = rgb16 + c * 256;
        float v = (p[y0*16+x0i]*(1-wx) + p[y0*16+x1i]*wx)*(1-wy) + (p[y1*16+x0i]*(1-wx) + p[y1*16+x1i]*wx)*wy;
        float nx = (float)x / width * 2.0f - 1.0f, ny = (float)y / height * 2.0f - 1.0f;
        float glow = expf(-1.6f * (nx * nx + ny * ny));
        float out = 0.12f + 0.78f * (0.65f * (0.5f + 0.5f * sinf(v * 2.2f + hue_shift + c * 2.094f + glow * 2.5f)) + 0.35f * glow);
        if (out < 0.0f) out = 0.0f; if (out > 1.0f) out = 1.0f;
        rgb[((size_t)y * width + x) * 3u + c] = (uint8_t)(out * 255.0f);
      }
    }
  }
  snprintf(res->file_main, sizeof res->file_main, "%s/image_%u.png", st->artifact_dir, seed);
  int rc = ec_export_png(res->file_main, rgb, (uint32_t)width, (uint32_t)height);
  free(rgb);
  res->width = width; res->height = height;
  return rc;
}

// ===========================================================================
// 2. 3D Object Generation: TRELLIS.2 1.3B (FLUX.2 Hero -> DINOv3 -> 30L Flow -> 64^3 3D Conv -> GLB + PLY)
// ===========================================================================
static int tr_ref_progress_bridge(int step, int total, const char *stage, void *ctx) {
  fx_cb_bridge_t *b = (fx_cb_bridge_t *)ctx;
  (void)total;
  return (b && b->cb) ? b->cb(step, 20, 0, stage ? stage : "flux2_3d_ref", b->cb_ctx) : 0;
}

static int tr_flow_progress_bridge(int step, int total, const char *stage, void *ctx) {
  fx_cb_bridge_t *b = (fx_cb_bridge_t *)ctx;
  return (b && b->cb) ? b->cb(5 + step, 5 + total, 0, stage ? stage : "trellis2_flow", b->cb_ctx) : 0;
}

extern "C" int ec_studio_gen_3d_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, int full_model, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  ec_studio_3d_v8_t req = {};
  req.prompt = prompt;
  req.seed = seed;
  req.grid_res = grid_res;
  req.steps = 12;
  req.cfg_scale = 7.5f;
  req.rescale_t = 5.0f;
  req.tier = tier;
  req.full_model = full_model;
  return ec_studio_gen_3d_v8(st, &req, cb, cb_ctx, res);
}

extern "C" int ec_studio_gen_3d(
    ec_studio_t *st, const char *prompt, uint32_t seed, int grid_res,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;
  if (grid_res < 12) grid_res = 20; if (grid_res > 32) grid_res = 32;
  if (cb && cb(0, 3, 0, "ovoxel_hash", cb_ctx) != 0) { res->cancelled = 1; return 0; }

  double t0 = st_now_ms();
  uint64_t rs = hash_prompt_seed(prompt, seed);
  const float freq = 2.0f + (float)(rs & 3u);
  int32_t *coords_h = (int32_t *)malloc((size_t)grid_res * grid_res * grid_res * 3u * sizeof(int32_t));
  int n_act = 0;
  const float half = (float)(grid_res - 1) * 0.5f;
  for (int z = 0; z < grid_res; z++)
    for (int y = 0; y < grid_res; y++)
      for (int x = 0; x < grid_res; x++) {
        float fx = ((float)x - half) / half, fy = ((float)y - half) / half, fz = ((float)z - half) / half;
        float r_xy = sqrtf(fx * fx + fz * fz);
        float d_tor = sqrtf((r_xy - 0.58f) * (r_xy - 0.58f) + fy * fy) - 0.24f;
        float d_sph = sqrtf(fx * fx + fy * fy + fz * fz) - (0.36f + 0.08f * sinf(freq * fx) * cosf(freq * fz));
        float sdf = fminf(d_tor, d_sph);
        if (sdf <= 0.0f && sdf >= -0.26f) {
          coords_h[3 * n_act + 0] = x; coords_h[3 * n_act + 1] = y; coords_h[3 * n_act + 2] = z; n_act++;
        }
      }
  if (n_act == 0) { free(coords_h); return -1; }

  const int Cin = 8, Cout = 8;
  int hcap = 1; while (hcap < 4 * n_act) hcap <<= 1;
  float *fin_h = (float *)malloc((size_t)n_act * Cin * 4);
  float *fout_h = (float *)malloc((size_t)n_act * Cout * 4);
  float *w_h = (float *)malloc((size_t)27 * Cout * Cin * 4);
  for (int i = 0; i < n_act * Cin; i++) fin_h[i] = prng_next(&rs) * 0.8f;
  for (int i = 0; i < 27 * Cout * Cin; i++) w_h[i] = prng_next(&rs) * 0.15f;

  pthread_mutex_lock(&st->gpu_mu);
  ec_vram_acquire(&st->vram_pool, st->slot_3d_weights, st->compute_stream);
  ec_ovoxel_t ov = {};
  ov.n_active = n_act; ov.channels = Cin; ov.grid_res = grid_res; ov.hash_cap = hcap;
  float *fin_d, *fout_d, *w_d;
  cudaMalloc(&ov.coords_d, (size_t)n_act * 3 * 4);
  cudaMalloc(&ov.hash_keys_d, (size_t)hcap * 8);
  cudaMalloc(&ov.hash_vals_d, (size_t)hcap * 4);
  cudaMalloc(&fin_d, (size_t)n_act * Cin * 4); cudaMalloc(&fout_d, (size_t)n_act * Cout * 4);
  cudaMalloc(&w_d, (size_t)27 * Cout * Cin * 4);
  cudaMemcpyAsync(ov.coords_d, coords_h, (size_t)n_act * 3 * 4, cudaMemcpyHostToDevice, st->compute_stream);
  cudaMemcpyAsync(fin_d, fin_h, (size_t)n_act * Cin * 4, cudaMemcpyHostToDevice, st->compute_stream);
  cudaMemcpyAsync(w_d, w_h, (size_t)27 * Cout * Cin * 4, cudaMemcpyHostToDevice, st->compute_stream);

  ec_launch_ovoxel_hash_build(&ov, st->compute_stream);
  ec_launch_ovoxel_sparse_conv3d(&ov, fin_d, w_d, nullptr, fout_d, Cout, st->compute_stream);
  if (cb) cb(1, 3, 0, "slat_dit_3d", cb_ctx);

  st->dit_cfg.ndims = 3;
  st->dit_cfg.axis_dims[0] = 16; st->dit_cfg.axis_dims[1] = 24; st->dit_cfg.axis_dims[2] = 24;
  st->dit_cfg.block_causal = 0; st->dit_cfg.modality = EC_MODALITY_3D_OVOXEL;
  int32_t c256[256 * 3];
  for (int i = 0; i < 256; i++) {
    int src = (i < n_act) ? i : (n_act - 1);
    c256[3*i] = coords_h[3*src]; c256[3*i+1] = coords_h[3*src+1]; c256[3*i+2] = coords_h[3*src+2];
  }
  cudaMemcpyAsync(st->coords_d, c256, sizeof c256, cudaMemcpyHostToDevice, st->compute_stream);
  ec_dit_block_forward(&st->dit_cfg, &st->dit_w, &st->dit_ws, st->x_d, st->coords_d, tier, st->compute_stream);
  cudaMemcpyAsync(fout_h, fout_d, (size_t)n_act * Cout * 4, cudaMemcpyDeviceToHost, st->compute_stream);
  cudaStreamSynchronize(st->compute_stream);
  res->gpu_ms = st_now_ms() - t0;

  cudaFree(ov.coords_d); cudaFree(ov.hash_keys_d); cudaFree(ov.hash_vals_d);
  cudaFree(fin_d); cudaFree(fout_d); cudaFree(w_d);
  pthread_mutex_unlock(&st->gpu_mu);

  if (cb) cb(2, 3, 0, "export_glb_ply", cb_ctx);
  ec_mesh_t mesh; ec_splat3d_t splat;
  ec_ovoxel_build_mesh(coords_h, fout_h, n_act, Cout, grid_res, &mesh);
  ec_ovoxel_build_splats(coords_h, fout_h, n_act, Cout, grid_res, &splat);
  snprintf(res->file_main, sizeof res->file_main, "%s/asset3d_%u.glb", st->artifact_dir, seed);
  snprintf(res->file_aux,  sizeof res->file_aux,  "%s/asset3d_%u.ply", st->artifact_dir, seed);
  int rc = ec_export_glb(res->file_main, &mesh) | ec_export_splat_ply(res->file_aux, &splat);
  res->n_verts = (int)mesh.n_verts; res->n_tris = (int)mesh.n_tris; res->n_splats = (int)splat.n_splats;
  res->steps_run = 1;
  ec_mesh_free(&mesh); ec_splat_free(&splat);
  free(coords_h); free(fin_h); free(fout_h); free(w_h);
  return rc;
}

// ===========================================================================
// 3. Music + 4 Stems: ACE-Step 1.5 Turbo (48 kHz Stereo + 4 Stems) / Fast Smoke
// ===========================================================================
extern "C" int ec_studio_gen_music_ex(
    ec_studio_t *st, const char *prompt, const char *lyrics, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, int full_model, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;

  if (full_model && st->acestep && ec_acestep_is_ready(st->acestep)) {
    uint64_t tag = hash_prompt_seed(prompt, seed);
    static const char *snames[4] = { "drums", "bass", "harmony", "lead" };
    snprintf(res->file_main, sizeof res->file_main, "%s/acestep_%u_%04x_master.wav",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));
    const char *spaths[4];
    for (int i = 0; i < 4; i++) {
      snprintf(res->stem_files[i], sizeof res->stem_files[i], "%s/acestep_%u_%04x_stem_%s.wav",
               st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu), snames[i]);
      spaths[i] = res->stem_files[i];
    }

    ec_acestep_req_t areq = {};
    areq.prompt = (prompt && prompt[0]) ? prompt : "warm jazz house groove, electric rhodes piano, crisp percussion, analog bassline";
    areq.lyrics = (lyrics && lyrics[0]) ? lyrics : "";
    areq.seed = seed;
    areq.bpm = (bpm >= 40 && bpm <= 240) ? bpm : 120;
    areq.duration_sec = (duration_sec >= 4.0f && duration_sec <= 60.0f) ? duration_sec : 10.0f;
    areq.steps = 8;
    areq.shift = 3.0f;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_acestep_res_t ares = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 2);
    int rc = ec_acestep_generate_wav(st->acestep, &areq, res->file_main, spaths, NULL,
                                     cb ? fx_progress_bridge : NULL, &bridge, &ares);
    pthread_mutex_unlock(&st->gpu_mu);

    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && ares.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "acestep_1.5_turbo");
      res->sample_rate = ares.sample_rate;
      res->frames = ares.num_samples;
      res->steps_run = ares.steps_run;
      res->text_enc_ms = ares.text_enc_ms + ares.cond_enc_ms;
      res->dit_ms = ares.dit_ms;
      res->vae_ms = ares.vae_ms;
      res->gpu_ms = ares.total_ms;
      return 0;
    }
    return -1;
  }

  int rc = ec_studio_gen_music(st, prompt, seed, bpm, duration_sec, tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  res->sample_rate = 44100;
  return rc;
}

extern "C" int ec_studio_gen_music(
    ec_studio_t *st, const char *prompt, uint32_t seed, int bpm, float duration_sec,
    ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;
  if (bpm < 60 || bpm > 200) bpm = 120;
  if (duration_sec < 1.0f) duration_sec = 3.0f; if (duration_sec > 15.0f) duration_sec = 8.0f;

  double t0 = st_now_ms();
  uint64_t rs = hash_prompt_seed(prompt, seed);
  const int N = 256, E = st->dit_cfg.dim;
  float *x0 = (float *)malloc((size_t)N * E * 4);
  for (int i = 0; i < N * E; i++) x0[i] = prng_next(&rs);
  float env[4 * 256];

  pthread_mutex_lock(&st->gpu_mu);
  ec_vram_acquire(&st->vram_pool, st->slot_music_weights, st->compute_stream);
  st->dit_cfg.ndims = 1; st->dit_cfg.axis_dims[0] = 64; st->dit_cfg.axis_dims[1] = 0; st->dit_cfg.axis_dims[2] = 0;
  st->dit_cfg.block_causal = 0; st->dit_cfg.modality = EC_MODALITY_MUSIC;
  int32_t c1d[256]; for (int i = 0; i < 256; i++) c1d[i] = i;
  cudaMemcpyAsync(st->coords_d, c1d, sizeof c1d, cudaMemcpyHostToDevice, st->compute_stream);
  cudaMemcpyAsync(st->x_d, x0, (size_t)N * E * 4, cudaMemcpyHostToDevice, st->compute_stream);

  for (int s = 0; s < 4; s++) {
    if (cb && cb(s, 5, 0, "music_dit_1d", cb_ctx) != 0) { res->cancelled = 1; break; }
    cudaMemcpyAsync(st->v_d, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToDevice, st->compute_stream);
    ec_dit_block_forward(&st->dit_cfg, &st->dit_w, &st->dit_ws, st->v_d, st->coords_d, tier, st->compute_stream);
    ec_launch_flow_cfg_euler(st->x_d, st->v_d, nullptr, st->x_d, N * E, -0.25f, 1.0f, st->compute_stream);
    res->steps_run++;
  }

  if (!res->cancelled) {
    cudaMemcpyAsync(x0, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    float lat_h[8 * 256], w1d_h[4 * 8 * 5];
    for (int c = 0; c < 8; c++) for (int i = 0; i < 256; i++) lat_h[c * 256 + i] = x0[i * E + c];
    for (int i = 0; i < 4 * 8 * 5; i++) w1d_h[i] = prng_next(&rs) * 0.2f;
    float *lat_d, *w1d_d, *env_d;
    cudaMalloc(&lat_d, sizeof lat_h); cudaMalloc(&w1d_d, sizeof w1d_h); cudaMalloc(&env_d, sizeof env);
    cudaMemcpyAsync(lat_d, lat_h, sizeof lat_h, cudaMemcpyHostToDevice, st->compute_stream);
    cudaMemcpyAsync(w1d_d, w1d_h, sizeof w1d_h, cudaMemcpyHostToDevice, st->compute_stream);
    ec_launch_conv1d_f32(lat_d, w1d_d, nullptr, env_d, 8, 4, 256, 5, 1, 0, 1, 1, st->compute_stream);
    cudaMemcpyAsync(env, env_d, sizeof env, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    cudaFree(lat_d); cudaFree(w1d_d); cudaFree(env_d);
  }
  res->gpu_ms = st_now_ms() - t0;
  pthread_mutex_unlock(&st->gpu_mu);
  free(x0);
  if (res->cancelled) return 0;

  if (cb) cb(4, 5, 0, "render_stems_wav", cb_ctx);
  const uint32_t SR = 44100, NF = (uint32_t)(duration_sec * (float)SR);
  const double beat_sec = 60.0 / (double)bpm;
  static const double scale_hz[8] = { 220.0, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33 };
  static const char *stem_names[4] = { "drums", "bass", "harmony", "lead" };
  float *master = (float *)calloc((size_t)NF * 2u, sizeof(float));
  float *stem_buf = (float *)malloc((size_t)NF * 2u * sizeof(float));
  int rc = 0;
  for (int si = 0; si < 4; si++) {
    for (uint32_t i = 0; i < NF; i++) {
      double t = (double)i / SR;
      int ti = (int)((t / duration_sec) * 255.0); if (ti > 255) ti = 255;
      double lm = 0.55 + 0.45 * tanh((double)env[si * 256 + ti]);
      double bp = fmod(t, beat_sec) / beat_sec;
      int s16 = (int)(t / (beat_sec * 0.25));
      double L = 0.0, R = 0.0;
      if (si == 0) {
        double k = sin(2.0 * 3.14159265 * (52.0 + 95.0 * exp(-bp * 28.0)) * bp * beat_sec) * exp(-bp * 10.0);
        double hp = fmod(t, beat_sec * 0.5) / (beat_sec * 0.5);
        double hat = (((s16 * 1103515245 + 12345) & 0xFF) / 128.0 - 1.0) * 0.14 * exp(-hp * 35.0);
        L = (0.34 * k + hat) * lm; R = (0.34 * k - hat * 0.8) * lm;
      } else if (si == 1) {
        int ch = ((int)(t / (beat_sec * 2.0))) & 3;
        double rt = scale_hz[ch] * 0.25, eb = exp(-fmod(t, beat_sec * 0.5) * 3.5);
        L = R = (0.7 * sin(2.0 * 3.14159265 * rt * t) + 0.3 * sin(4.0 * 3.14159265 * rt * t)) * eb * 0.32 * lm;
      } else if (si == 2) {
        int ch = ((int)(t / (beat_sec * 2.0))) & 3;
        double sw = 0.65 + 0.35 * sin(2.0 * 3.14159265 * 0.25 * t);
        L = 0.14 * sw * lm * (sin(2.0 * 3.14159265 * scale_hz[ch] * t) + sin(2.0 * 3.14159265 * scale_hz[(ch+2)&7] * t));
        R = 0.14 * sw * lm * (sin(2.0 * 3.14159265 * scale_hz[ch] * t * 1.003) + sin(2.0 * 3.14159265 * scale_hz[(ch+4)&7] * t));
      } else {
        int ni = (int)(fabs(env[3 * 256 + ((s16 * 7) & 255)]) * 11.0 + s16) & 7;
        double fl = scale_hz[ni] * 2.0, plk = exp(-fmod(t, beat_sec * 0.25) / (beat_sec * 0.25) * 7.5);
        double pan = 0.5 + 0.35 * sin(t * 1.7);
        double sig = 0.24 * plk * lm * (sin(2.0 * 3.14159265 * fl * t) + 0.35 * sin(6.0 * 3.14159265 * fl * t));
        L = sig * (1.0 - pan); R = sig * pan;
      }
      stem_buf[2*i] = (float)L; stem_buf[2*i+1] = (float)R;
      master[2*i] += (float)L; master[2*i+1] += (float)R;
    }
    ec_audio_t sa = { stem_buf, NF, SR, 2 };
    snprintf(res->stem_files[si], sizeof res->stem_files[si], "%s/music_%u_stem_%s.wav", st->artifact_dir, seed, stem_names[si]);
    rc |= ec_export_wav(res->stem_files[si], &sa);
  }
  for (uint32_t i = 0; i < NF * 2u; i++) master[i] = tanhf(master[i]);
  ec_audio_t ma = { master, NF, SR, 2 };
  snprintf(res->file_main, sizeof res->file_main, "%s/music_%u_master.wav", st->artifact_dir, seed);
  rc |= ec_export_wav(res->file_main, &ma);
  free(master); free(stem_buf);
  res->frames = (int)NF;
  res->sample_rate = (int)SR;
  if (!res->engine_mode[0]) snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return rc;
}

// ===========================================================================
// 4. Video + Joint A/V: Wan 2.2 TI2V-5B (+ ACE-Step 1.5 Synced AAC Audio) -> NVENC H.264 MP4!
// ===========================================================================
extern "C" int ec_studio_gen_video_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    int fps, int steps, int with_audio, ec_tier_t tier, int full_model,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;

  if (full_model && st->wan22 && ec_wan22_is_ready(st->wan22)) {
    int W = (width >= 640) ? ((width + 31) & ~31) : 832;
    if (W > 1280) W = 1280;
    int H = (height >= 384) ? ((height + 31) & ~31) : 480;
    if (H > 768) H = 768;
    int req_f = (n_frames >= 5) ? n_frames : 17;
    int F_lat = (req_f - 1) / 4 + 1;
    if (F_lat < 2) F_lat = 2;
    if (F_lat > 21) F_lat = 21;
    int actual_f = 1 + 4 * (F_lat - 1);
    int afps = (fps >= 8 && fps <= 60) ? fps : 16;
    int asteps = (steps >= 4 && steps <= 50) ? steps : 20;

    // Optional synced stereo audio track via native ACE-Step 1.5 Turbo FIRST (evicts cleanly before Wan 2.2!)
    char wav_path[256] = {0};
    double audio_ms = 0.0;
    if (with_audio) {
      if (cb && cb(0, asteps + 4, 0, "acestep_synced_audio", cb_ctx) != 0) {
        res->cancelled = 1;
        return 0;
      }
      ec_studio_result_t ares = {};
      float dur = (float)actual_f / (float)afps;
      if (dur < 4.0f) dur = 4.0f;
      int use_acestep = (st->acestep && ec_acestep_is_ready(st->acestep)) ? 1 : 0;
      if (ec_studio_gen_music_ex(st, prompt, "", seed + 17, 116, dur, tier, use_acestep, NULL, NULL, &ares) == 0 && !ares.cancelled) {
        snprintf(wav_path, sizeof wav_path, "%s", ares.file_main);
        audio_ms = ares.gpu_ms;
      }
    }

    uint64_t tag = hash_prompt_seed(prompt, seed);
    snprintf(res->file_mp4, sizeof res->file_mp4, "%s/wan22_%u_%04x%s.mp4",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu), with_audio ? "_av" : "");
    snprintf(res->file_aux, sizeof res->file_aux, "%s/wan22_%u_%04x_kf0.png",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));
    snprintf(res->file_main, sizeof res->file_main, "%s", res->file_mp4);

    ec_wan22_req_t wreq = {};
    wreq.prompt = (prompt && prompt[0]) ? prompt : "Aerial drone footage flying slowly along dramatic black volcanic sea cliffs in Iceland at golden sunset, turquoise ocean waves crashing on shore, cinema 35mm";
    wreq.seed = seed;
    wreq.width = W;
    wreq.height = H;
    wreq.num_frames = actual_f;
    wreq.fps = afps;
    wreq.steps = asteps;
    wreq.cfg_scale = 5.0f;
    wreq.shift = 5.0f;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_wan22_res_t wres = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 4);
    int rc = ec_wan22_generate_mp4(st->wan22, &wreq, res->file_mp4, res->file_aux,
                                   wav_path[0] ? wav_path : NULL,
                                   cb ? fx_progress_bridge : NULL, &bridge, &wres);
    pthread_mutex_unlock(&st->gpu_mu);

    if (rc == -2) { res->cancelled = 1; return 0; }
    if (rc == 0 && wres.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "wan2.2_ti2v_5b");
      snprintf(res->encoder_used, sizeof res->encoder_used, "%s", wres.encoder_used);
      res->width = wres.width;
      res->height = wres.height;
      res->frames = wres.num_frames;
      res->steps_run = wres.steps_run;
      res->text_enc_ms = wres.text_enc_ms;
      res->dit_ms = wres.dit_ms;
      res->vae_ms = wres.vae_ms;
      res->gpu_ms = wres.total_ms + audio_ms;
      res->fps = (double)wres.num_frames / ((wres.total_ms > 1.0 ? wres.total_ms : 1.0) * 1e-3);
      return 0;
    }
    return -1;
  }

  int rc = ec_studio_gen_video(st, prompt, seed, n_frames, width, height, fps, with_audio, tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return rc;
}

extern "C" int ec_studio_gen_video(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames, int width, int height,
    int fps, int with_audio, ec_tier_t tier, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;
  if (n_frames < 8) n_frames = 24; if (n_frames > 120) n_frames = 48;
  if (width < 128) width = 320; width &= ~1;
  if (height < 128) height = 180; height &= ~1;
  if (fps < 8 || fps > 60) fps = 24;

  // Optional LTX-2.5 joint synced stereo audio track first
  char wav_path[256] = {0};
  if (with_audio) {
    ec_studio_result_t ares;
    float dur = (float)n_frames / (float)fps;
    if (dur < 1.0f) dur = 1.0f;
    if (ec_studio_gen_music(st, prompt, seed + 17, 120, dur, tier, NULL, NULL, &ares) == 0) {
      snprintf(wav_path, sizeof wav_path, "%s", ares.file_main);
    }
  }

  double t0 = st_now_ms();
  uint64_t rs = hash_prompt_seed(prompt, seed);
  const int N = 256, E = st->dit_cfg.dim;
  float *x0 = (float *)malloc((size_t)N * E * 4);
  for (int i = 0; i < N * E; i++) x0[i] = prng_next(&rs);
  float vrgb[3 * 4 * 64];

  pthread_mutex_lock(&st->gpu_mu);
  ec_vram_acquire(&st->vram_pool, st->slot_dit_weights, st->compute_stream);
  ec_vram_acquire(&st->vram_pool, st->slot_vae_weights, st->compute_stream);

  st->dit_cfg.ndims = 3;
  st->dit_cfg.axis_dims[0] = 16; st->dit_cfg.axis_dims[1] = 24; st->dit_cfg.axis_dims[2] = 24;
  st->dit_cfg.block_causal = 0; st->dit_cfg.modality = EC_MODALITY_VIDEO;
  int32_t c3d[256 * 3];
  for (int i = 0; i < 256; i++) { c3d[3*i] = i / 64; c3d[3*i+1] = (i / 8) % 8; c3d[3*i+2] = i % 8; }
  cudaMemcpyAsync(st->coords_d, c3d, sizeof c3d, cudaMemcpyHostToDevice, st->compute_stream);
  cudaMemcpyAsync(st->x_d, x0, (size_t)N * E * 4, cudaMemcpyHostToDevice, st->compute_stream);

  for (int s = 0; s < 4; s++) {
    if (cb && cb(s, 6, 0, "video_dit_3d", cb_ctx) != 0) { res->cancelled = 1; break; }
    cudaMemcpyAsync(st->v_d, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToDevice, st->compute_stream);
    ec_dit_block_forward(&st->dit_cfg, &st->dit_w, &st->dit_ws, st->v_d, st->coords_d, tier, st->compute_stream);
    ec_launch_flow_cfg_euler(st->x_d, st->v_d, nullptr, st->x_d, N * E, -0.25f, 1.0f, st->compute_stream);
    res->steps_run++;
  }

  if (!res->cancelled) {
    cudaMemcpyAsync(x0, st->x_d, (size_t)N * E * 4, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    float vlat_h[4 * 256], vw_h[3 * 4 * 27];
    for (int c = 0; c < 4; c++) for (int i = 0; i < 256; i++) vlat_h[c * 256 + i] = x0[i * E + c];
    for (int i = 0; i < 3 * 4 * 27; i++) vw_h[i] = prng_next(&rs) * 0.2f;
    float *vlat_d, *vw_d, *vrgb_d;
    cudaMalloc(&vlat_d, sizeof vlat_h); cudaMalloc(&vw_d, sizeof vw_h); cudaMalloc(&vrgb_d, sizeof vrgb);
    cudaMemcpyAsync(vlat_d, vlat_h, sizeof vlat_h, cudaMemcpyHostToDevice, st->compute_stream);
    cudaMemcpyAsync(vw_d, vw_h, sizeof vw_h, cudaMemcpyHostToDevice, st->compute_stream);
    ec_launch_conv3d_causal_f32(vlat_d, nullptr, vw_d, nullptr, vrgb_d, 4, 3, 4, 8, 8, 3, 3, 3, 1, st->compute_stream);
    cudaMemcpyAsync(vrgb, vrgb_d, sizeof vrgb, cudaMemcpyDeviceToHost, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    cudaFree(vlat_d); cudaFree(vw_d); cudaFree(vrgb_d);
  }
  res->gpu_ms = st_now_ms() - t0;
  pthread_mutex_unlock(&st->gpu_mu);
  free(x0);
  if (res->cancelled) return 0;

  if (cb) cb(5, 6, 0, "nvenc_mux_mp4", cb_ctx);
  snprintf(res->file_mp4, sizeof res->file_mp4, "%s/video_%u%s.mp4",
           st->artifact_dir, seed, with_audio ? "_av" : "");
  snprintf(res->file_aux, sizeof res->file_aux, "%s/video_%u_kf0.png", st->artifact_dir, seed);
  snprintf(res->file_last, sizeof res->file_last, "%s/video_%u_last.png", st->artifact_dir, seed);
  snprintf(res->file_main, sizeof res->file_main, "%s", res->file_mp4);

  ec_mp4_writer_t *mp4 = ec_mp4_open(res->file_mp4, wav_path[0] ? wav_path : NULL,
                                     (uint32_t)width, (uint32_t)height, (uint32_t)fps, 1,
                                     res->encoder_used, sizeof res->encoder_used);
  uint8_t *rgb = (uint8_t *)malloc((size_t)width * height * 3u);
  int rc = (mp4 != NULL) ? 0 : -1;
  for (int f = 0; f < n_frames && rc == 0; f++) {
    int tf = (f * 3) / (n_frames > 1 ? (n_frames - 1) : 1);
    float phase = (float)f / (float)n_frames * 6.283185f;
    for (int y = 0; y < height; y++) {
      int gy = (y * 8) / height;
      for (int x = 0; x < width; x++) {
        int gx = (x * 8) / width;
        size_t o = ((size_t)y * width + x) * 3u;
        for (int c = 0; c < 3; c++) {
          float lv = vrgb[((c * 4 + tf) * 8 + gy) * 8 + gx];
          float val = 0.5f + 0.45f * sinf(lv * 1.8f + phase + (float)c * 1.9f + (float)x * 0.015f);
          rgb[o + c] = (uint8_t)(val * 255.0f);
        }
      }
    }
    if (f == 0) ec_export_png(res->file_aux, rgb, (uint32_t)width, (uint32_t)height);
    if (f == n_frames - 1) ec_export_png(res->file_last, rgb, (uint32_t)width, (uint32_t)height);
    rc |= ec_mp4_write_rgb24(mp4, rgb, (size_t)width * height * 3u);
  }
  free(rgb);
  if (mp4) rc |= ec_mp4_close(mp4);
  res->width = width; res->height = height; res->frames = n_frames;
  res->fps = (double)n_frames / (res->gpu_ms * 1e-3);
  return rc;
}

// ===========================================================================
// 5. Realtime Video Burst: Block-Causal Self-Forcing CUDA Graph + Zero-Copy SHM Ring
// ===========================================================================
extern "C" int ec_studio_realtime_burst(
    ec_studio_t *st, const char *prompt, uint32_t seed, int n_frames,
    const char *shm_name, ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;
  if (n_frames < 4) n_frames = 30; if (n_frames > 120) n_frames = 60;
  (void)prompt;

  const uint32_t W = 640, H = 360;
  ec_shm_ring_t ring;
  const char *rname = (shm_name && shm_name[0]) ? shm_name : "/eureka_rt_studio_live";
  if (ec_shm_ring_create(&ring, rname, W, H) != 0) return -1;
  uint8_t *frame = (uint8_t *)malloc((size_t)W * H * 4u);

  pthread_mutex_lock(&st->gpu_mu);
  st->dit_cfg.ndims = 3;
  st->dit_cfg.axis_dims[0] = 16; st->dit_cfg.axis_dims[1] = 24; st->dit_cfg.axis_dims[2] = 24;
  st->dit_cfg.block_causal = 64; st->dit_cfg.modality = EC_MODALITY_VIDEO_RT;
  ec_dit_capture_graph(&st->dit_cfg, &st->dit_w, 1, &st->dit_ws, st->x_d, st->coords_d, EC_TIER_FAST, st->compute_stream);

  double t0 = st_now_ms();
  for (int f = 0; f < n_frames; f++) {
    if (cb && (f % 10 == 0) && cb(f, n_frames, 0, "realtime_shm", cb_ctx) != 0) { res->cancelled = 1; break; }
    ec_dit_replay_graph(&st->dit_ws, st->compute_stream);
    cudaStreamSynchronize(st->compute_stream);
    memset(frame, (int)((seed + f * 9) & 255), (size_t)W * H * 4u);
    ec_shm_ring_push(&ring, frame, (uint64_t)f * 16667ULL);
    res->steps_run++;
  }
  double elapsed = st_now_ms() - t0;
  pthread_mutex_unlock(&st->gpu_mu);

  ec_shm_ring_close(&ring);
  free(frame);
  res->width = (int)W; res->height = (int)H; res->frames = res->steps_run;
  res->gpu_ms = elapsed;
  res->fps = (double)res->steps_run / (elapsed * 1e-3);
  snprintf(res->file_main, sizeof res->file_main, "%s", rname);
  return 0;
}

// ===========================================================================
// 6. Full ComfyUI-Level Compiled Native DAG Runtime (24 Typed Nodes, Merkle Cache,
//    Chained LoRAs, Native Post-FX, PNG tEXt Roundtrip, Live Per-Node SSE!)
// ===========================================================================
typedef struct {
  char modality[16];     // "image", "music", "video", "3d"
  char checkpoint[512];
  ec_studio_lora_t loras[EC_STUDIO_MAX_LORAS];
  int  n_loras;
  char prompt[1024];
  char neg_prompt[1024];
  char lyrics[1024];
  int  width, height, frames, steps, bpm, full_model;
  uint32_t seed;
  float cfg, duration_sec;
  char image_path[256];
  char end_image_path[256];
  char mask_path[256];
  char audio_path[256];
  char video_path[256];
  char glb_path[256];
  char ply_path[256];
  char stem_paths[8][256];
  char camera_motion[32];
  char key_scale[32];
  char vocal_mode[24];
  char audio_mode[24];
  float denoise, repaint_start, repaint_end;
  int  stem_mode, stem_mode_set, tile_upscale, isolate_fg;
  char engine_mode[32];
  double gpu_ms;
} ec_node_val_t;

typedef struct {
  ec_studio_t          *st;
  ec_studio_result_t   *res;
  const char           *api_json;
  ec_flow_ui_event_cb_t event_cb;
  void                 *cb_ctx;
  char                 *node_outs_json;
  size_t                node_outs_cap;
  int                   node_outs_pos;
  int                   node_outs_cnt;
  char                  cur_node_id[32];
  char                  cur_class_type[64];
} flow_exec_ctx_t;

static const char *flow_bname(const char *p) {
  if (!p || !p[0]) return "";
  const char *s = strrchr(p, '/');
  return s ? s + 1 : p;
}

static const ec_node_val_t *flow_get_up_val(
    const ec_flow_graph_t *g, const ec_flow_node_t *nd,
    const ec_flow_cache_entry_t *ups[], int32_t n_up, const char *key) {
  int32_t ui = 0;
  for (int32_t k = 0; k < nd->n_inputs; k++) {
    if (nd->inputs[k].kind == EC_VAL_LINK) {
      if (ui < n_up && !strcmp(nd->inputs[k].key, key)) {
        if (ups[ui] && ups[ui]->payload && ups[ui]->payload_bytes == sizeof(ec_node_val_t)) {
          return (const ec_node_val_t *)ups[ui]->payload;
        }
      }
      ui++;
    }
  }
  (void)g;
  return NULL;
}

static void flow_merge_val(ec_node_val_t *dst, const ec_node_val_t *src) {
  if (!dst || !src) return;
  if (src->modality[0] && !dst->modality[0]) snprintf(dst->modality, sizeof dst->modality, "%s", src->modality);
  if (src->checkpoint[0] && !dst->checkpoint[0]) snprintf(dst->checkpoint, sizeof dst->checkpoint, "%s", src->checkpoint);
  for (int i = 0; i < src->n_loras && dst->n_loras < EC_STUDIO_MAX_LORAS; i++) {
    int dup = 0;
    for (int j = 0; j < dst->n_loras; j++) if (!strcmp(dst->loras[j].path, src->loras[i].path)) dup = 1;
    if (!dup) dst->loras[dst->n_loras++] = src->loras[i];
  }
  if (src->prompt[0] && !dst->prompt[0]) snprintf(dst->prompt, sizeof dst->prompt, "%s", src->prompt);
  if (src->neg_prompt[0] && !dst->neg_prompt[0]) snprintf(dst->neg_prompt, sizeof dst->neg_prompt, "%s", src->neg_prompt);
  if (src->lyrics[0] && !dst->lyrics[0]) snprintf(dst->lyrics, sizeof dst->lyrics, "%s", src->lyrics);
  if (src->width > 0 && dst->width <= 0) dst->width = src->width;
  if (src->height > 0 && dst->height <= 0) dst->height = src->height;
  if (src->frames > 0 && dst->frames <= 0) dst->frames = src->frames;
  if (src->steps > 0 && dst->steps <= 0) dst->steps = src->steps;
  if (src->bpm > 0 && dst->bpm <= 0) dst->bpm = src->bpm;
  if (src->duration_sec > 0.0f && dst->duration_sec <= 0.0f) dst->duration_sec = src->duration_sec;
  if (src->seed > 0 && dst->seed == 0) dst->seed = src->seed;
  if (src->cfg > 0.0f && dst->cfg <= 0.0f) dst->cfg = src->cfg;
  if (src->full_model && !dst->full_model) dst->full_model = src->full_model;
  if (src->image_path[0] && !dst->image_path[0]) snprintf(dst->image_path, sizeof dst->image_path, "%s", src->image_path);
  if (src->end_image_path[0] && !dst->end_image_path[0]) snprintf(dst->end_image_path, sizeof dst->end_image_path, "%s", src->end_image_path);
  if (src->mask_path[0] && !dst->mask_path[0]) snprintf(dst->mask_path, sizeof dst->mask_path, "%s", src->mask_path);
  if (src->audio_path[0] && !dst->audio_path[0]) snprintf(dst->audio_path, sizeof dst->audio_path, "%s", src->audio_path);
  if (src->video_path[0] && !dst->video_path[0]) snprintf(dst->video_path, sizeof dst->video_path, "%s", src->video_path);
  if (src->glb_path[0] && !dst->glb_path[0]) snprintf(dst->glb_path, sizeof dst->glb_path, "%s", src->glb_path);
  if (src->ply_path[0] && !dst->ply_path[0]) snprintf(dst->ply_path, sizeof dst->ply_path, "%s", src->ply_path);
  if (src->camera_motion[0] && !dst->camera_motion[0]) snprintf(dst->camera_motion, sizeof dst->camera_motion, "%s", src->camera_motion);
  if (src->key_scale[0] && !dst->key_scale[0]) snprintf(dst->key_scale, sizeof dst->key_scale, "%s", src->key_scale);
  if (src->vocal_mode[0] && !dst->vocal_mode[0]) snprintf(dst->vocal_mode, sizeof dst->vocal_mode, "%s", src->vocal_mode);
  if (src->audio_mode[0] && !dst->audio_mode[0]) snprintf(dst->audio_mode, sizeof dst->audio_mode, "%s", src->audio_mode);
  if (src->denoise > 0.0f && dst->denoise <= 0.0f) dst->denoise = src->denoise;
  if (src->repaint_start > 0.0f && dst->repaint_start <= 0.0f) dst->repaint_start = src->repaint_start;
  if (src->repaint_end > 0.0f && dst->repaint_end <= 0.0f) dst->repaint_end = src->repaint_end;
  if (src->stem_mode_set && !dst->stem_mode_set) { dst->stem_mode = src->stem_mode; dst->stem_mode_set = 1; }
  if (src->tile_upscale && !dst->tile_upscale) dst->tile_upscale = src->tile_upscale;
  if (src->isolate_fg && !dst->isolate_fg) dst->isolate_fg = src->isolate_fg;
  for (int i = 0; i < 8; i++) {
    if (src->stem_paths[i][0] && !dst->stem_paths[i][0]) snprintf(dst->stem_paths[i], sizeof dst->stem_paths[i], "%s", src->stem_paths[i]);
  }
  if (src->engine_mode[0] && !dst->engine_mode[0]) snprintf(dst->engine_mode, sizeof dst->engine_mode, "%s", src->engine_mode);
}

static const char *flow_primary_artifact_url(const ec_node_val_t *v, char *buf, size_t cap) {
  buf[0] = '\0';
  if (!v) return buf;
  if (v->video_path[0])      snprintf(buf, cap, "/artifacts/%s", flow_bname(v->video_path));
  else if (v->glb_path[0])   snprintf(buf, cap, "/artifacts/%s", flow_bname(v->glb_path));
  else if (v->audio_path[0]) snprintf(buf, cap, "/artifacts/%s", flow_bname(v->audio_path));
  else if (v->image_path[0]) snprintf(buf, cap, "/artifacts/%s", flow_bname(v->image_path));
  return buf;
}

static int flow_sub_progress_cb(int step, int total, int skipped, const char *stage, void *ctx) {
  (void)skipped; (void)stage;
  flow_exec_ctx_t *fc = (flow_exec_ctx_t *)ctx;
  if (!fc || !fc->event_cb) return 0;
  return fc->event_cb(fc->cur_node_id, fc->cur_class_type, "progress",
                      step, total, fc->st->flow_graph.last_cache_hits,
                      fc->st->flow_graph.last_executed, 0.0, "", fc->cb_ctx);
}

static int studio_dag_cb(
    const ec_flow_graph_t *g, int32_t idx,
    const ec_flow_cache_entry_t *ups[], int32_t n_up,
    ec_flow_cache_entry_t *out, void *uctx) {
  flow_exec_ctx_t *fc = (flow_exec_ctx_t *)uctx;
  const ec_flow_node_t *nd = &g->nodes[idx];
  const char *ct = nd->class_type;
  uint64_t tag = nd->merkle_hash;
  for (int32_t i = 0; i < n_up; i++) tag ^= ups[i]->result_tag * 0x9E3779B97F4A7C15ULL;
  out->result_tag = tag;

  snprintf(fc->cur_node_id, sizeof fc->cur_node_id, "%s", nd->id);
  snprintf(fc->cur_class_type, sizeof fc->cur_class_type, "%s", nd->class_type);

  ec_node_val_t *val = (ec_node_val_t *)calloc(1, sizeof(ec_node_val_t));
  if (!val) return -1;
  out->payload = val;
  out->payload_bytes = sizeof(ec_node_val_t);

  // Merge all linked upstream typed values first (preserving explicit slot precedence below)
  for (int32_t i = 0; i < n_up; i++) {
    if (ups[i] && ups[i]->payload && ups[i]->payload_bytes == sizeof(ec_node_val_t)) {
      flow_merge_val(val, (const ec_node_val_t *)ups[i]->payload);
    }
  }
  // Explicit positive vs negative conditioning slot resolution
  const ec_node_val_t *pos_v = flow_get_up_val(g, nd, ups, n_up, "positive");
  const ec_node_val_t *neg_v = flow_get_up_val(g, nd, ups, n_up, "negative");
  if (pos_v && pos_v->prompt[0]) snprintf(val->prompt, sizeof val->prompt, "%s", pos_v->prompt);
  if (neg_v && neg_v->prompt[0]) snprintf(val->neg_prompt, sizeof val->neg_prompt, "%s", neg_v->prompt);

  // Local scalar input helpers
  const ec_flow_input_t *in_p = ec_flow_get_input(nd, "prompt");
  if (!in_p) in_p = ec_flow_get_input(nd, "text");
  if (!in_p) in_p = ec_flow_get_input(nd, "tags");
  if (!in_p) in_p = ec_flow_get_input(nd, "value");
  if (in_p && in_p->kind == EC_VAL_STRING && in_p->str_val[0]) {
    snprintf(val->prompt, sizeof val->prompt, "%s", in_p->str_val);
  }
  const ec_flow_input_t *in_lyr = ec_flow_get_input(nd, "lyrics");
  if (in_lyr && in_lyr->kind == EC_VAL_STRING && in_lyr->str_val[0]) {
    snprintf(val->lyrics, sizeof val->lyrics, "%s", in_lyr->str_val);
  }
  const ec_flow_input_t *in_s = ec_flow_get_input(nd, "seed");
  if (!in_s) in_s = ec_flow_get_input(nd, "noise_seed");
  if (in_s && in_s->kind == EC_VAL_NUMBER) val->seed = (uint32_t)in_s->num_val;
  const ec_flow_input_t *in_w = ec_flow_get_input(nd, "width");
  if (in_w && in_w->kind == EC_VAL_NUMBER && in_w->num_val > 0) val->width = (int)in_w->num_val;
  const ec_flow_input_t *in_h = ec_flow_get_input(nd, "height");
  if (in_h && in_h->kind == EC_VAL_NUMBER && in_h->num_val > 0) val->height = (int)in_h->num_val;
  const ec_flow_input_t *in_st = ec_flow_get_input(nd, "steps");
  if (in_st && in_st->kind == EC_VAL_NUMBER && in_st->num_val > 0) val->steps = (int)in_st->num_val;
  const ec_flow_input_t *in_cfg = ec_flow_get_input(nd, "cfg");
  if (!in_cfg) in_cfg = ec_flow_get_input(nd, "guidance");
  if (in_cfg && in_cfg->kind == EC_VAL_NUMBER && in_cfg->num_val > 0) val->cfg = (float)in_cfg->num_val;
  const ec_flow_input_t *in_bpm = ec_flow_get_input(nd, "bpm");
  if (in_bpm && in_bpm->kind == EC_VAL_NUMBER && in_bpm->num_val > 0) val->bpm = (int)in_bpm->num_val;
  const ec_flow_input_t *in_dur = ec_flow_get_input(nd, "seconds");
  if (!in_dur) in_dur = ec_flow_get_input(nd, "duration");
  if (!in_dur) in_dur = ec_flow_get_input(nd, "duration_sec");
  if (in_dur && in_dur->kind == EC_VAL_NUMBER && in_dur->num_val > 0) val->duration_sec = (float)in_dur->num_val;
  const ec_flow_input_t *in_fr = ec_flow_get_input(nd, "length");
  if (!in_fr) in_fr = ec_flow_get_input(nd, "frames");
  if (in_fr && in_fr->kind == EC_VAL_NUMBER && in_fr->num_val > 0) val->frames = (int)in_fr->num_val;
  const ec_flow_input_t *in_fm = ec_flow_get_input(nd, "full_model");
  if (!in_fm) in_fm = ec_flow_get_input(nd, "photoreal");
  if (in_fm && in_fm->kind == EC_VAL_NUMBER) val->full_model = (int)in_fm->num_val;

  // --- 1. MODEL & LORA LOADERS (0 ms Merkle-cached on rerun!) ---
  if (!strcmp(ct, "CheckpointLoaderSimple") || !strcmp(ct, "UNETLoader") || !strcmp(ct, "DiffusionModelLoader")) {
    const ec_flow_input_t *in_ck = ec_flow_get_input(nd, "ckpt_name");
    if (!in_ck) in_ck = ec_flow_get_input(nd, "unet_name");
    if (in_ck && in_ck->kind == EC_VAL_STRING && in_ck->str_val[0]) {
      pthread_mutex_lock(&fc->st->gpu_mu);
      const ec_storage_asset_t *a = ec_storage_find(&fc->st->vault, EC_ASSET_CHECKPOINT, in_ck->str_val);
      pthread_mutex_unlock(&fc->st->gpu_mu);
      snprintf(val->checkpoint, sizeof val->checkpoint, "%s", a ? a->path : in_ck->str_val);
      if (strstr(in_ck->str_val, "acestep") || strstr(in_ck->str_val, "ace_step")) snprintf(val->modality, sizeof val->modality, "music");
      else if (strstr(in_ck->str_val, "wan2.2") || strstr(in_ck->str_val, "wan22")) snprintf(val->modality, sizeof val->modality, "video");
      else if (strstr(in_ck->str_val, "trellis")) snprintf(val->modality, sizeof val->modality, "3d");
      else snprintf(val->modality, sizeof val->modality, "image");
    }
    return 0;
  }
  if (!strcmp(ct, "LoraLoader") || !strcmp(ct, "LoraLoaderModelOnly") || !strcmp(ct, "QLoRALoader")) {
    const ec_flow_input_t *in_ln = ec_flow_get_input(nd, "lora_name");
    const ec_flow_input_t *in_ls = ec_flow_get_input(nd, "strength_model");
    if (!in_ls) in_ls = ec_flow_get_input(nd, "strength");
    float str = (in_ls && in_ls->kind == EC_VAL_NUMBER) ? (float)in_ls->num_val : 0.85f;
    if (in_ln && in_ln->kind == EC_VAL_STRING && in_ln->str_val[0] &&
        strcmp(in_ln->str_val, "None") != 0 && val->n_loras < EC_STUDIO_MAX_LORAS) {
      pthread_mutex_lock(&fc->st->gpu_mu);
      const ec_storage_asset_t *a = ec_storage_find(&fc->st->vault, EC_ASSET_LORA, in_ln->str_val);
      pthread_mutex_unlock(&fc->st->gpu_mu);
      const char *lp = a ? a->path : in_ln->str_val;
      if (lp[0]) {
        snprintf(val->loras[val->n_loras].path, sizeof val->loras[val->n_loras].path, "%s", lp);
        val->loras[val->n_loras].strength = str;
        val->n_loras++;
      }
    }
    return 0;
  }
  if (!strcmp(ct, "LoadImage")) {
    const ec_flow_input_t *in_im = ec_flow_get_input(nd, "image");
    if (in_im && in_im->kind == EC_VAL_STRING && in_im->str_val[0]) {
      if (in_im->str_val[0] == '/') snprintf(val->image_path, sizeof val->image_path, "%s", in_im->str_val);
      else snprintf(val->image_path, sizeof val->image_path, "%s/%s", fc->st->artifact_dir, flow_bname(in_im->str_val));
    }
    return 0;
  }
  if (!strcmp(ct, "LoadAudio")) {
    const ec_flow_input_t *in_au = ec_flow_get_input(nd, "audio");
    if (in_au && in_au->kind == EC_VAL_STRING && in_au->str_val[0]) {
      if (in_au->str_val[0] == '/') snprintf(val->audio_path, sizeof val->audio_path, "%s", in_au->str_val);
      else snprintf(val->audio_path, sizeof val->audio_path, "%s/%s", fc->st->artifact_dir, flow_bname(in_au->str_val));
    }
    return 0;
  }
  if (!strcmp(ct, "VAEEncode") || !strcmp(ct, "ImageInpaintMask")) {
    if (!val->modality[0]) snprintf(val->modality, sizeof val->modality, "image");
    const ec_flow_input_t *dn = ec_flow_get_input(nd, "denoise");
    if (dn && dn->kind == EC_VAL_NUMBER) val->denoise = (float)dn->num_val;
    const ec_flow_input_t *mk = ec_flow_get_input(nd, "mask");
    if (mk && mk->kind == EC_VAL_STRING && mk->str_val[0]) {
      if (mk->str_val[0] == '/') snprintf(val->mask_path, sizeof val->mask_path, "%s", mk->str_val);
      else snprintf(val->mask_path, sizeof val->mask_path, "%s/%s", fc->st->artifact_dir, flow_bname(mk->str_val));
    }
    return 0;
  }
  if (!strcmp(ct, "AceStepAudioEncode") || !strcmp(ct, "AceStepRepaint") || !strcmp(ct, "AceStepExtend")) {
    snprintf(val->modality, sizeof val->modality, "music");
    if (!strcmp(ct, "AceStepRepaint")) snprintf(val->audio_mode, sizeof val->audio_mode, "repaint");
    else if (!strcmp(ct, "AceStepExtend")) snprintf(val->audio_mode, sizeof val->audio_mode, "extend");
    else {
      const ec_flow_input_t *am = ec_flow_get_input(nd, "audio_mode");
      snprintf(val->audio_mode, sizeof val->audio_mode, "%s", (am && am->kind == EC_VAL_STRING && am->str_val[0]) ? am->str_val : "cover");
    }
    const ec_flow_input_t *dn = ec_flow_get_input(nd, "denoise");
    if (dn && dn->kind == EC_VAL_NUMBER) val->denoise = (float)dn->num_val;
    const ec_flow_input_t *rs = ec_flow_get_input(nd, "start_sec");
    if (rs && rs->kind == EC_VAL_NUMBER) val->repaint_start = (float)rs->num_val;
    const ec_flow_input_t *re = ec_flow_get_input(nd, "end_sec");
    if (re && re->kind == EC_VAL_NUMBER) val->repaint_end = (float)re->num_val;
    const ec_flow_input_t *ts = ec_flow_get_input(nd, "target_seconds");
    if (ts && ts->kind == EC_VAL_NUMBER && ts->num_val > 0) val->duration_sec = (float)ts->num_val;
    return 0;
  }
  if (!strcmp(ct, "WanFirstLastFrameToVideo")) {
    snprintf(val->modality, sizeof val->modality, "video");
    const ec_node_val_t *sv = flow_get_up_val(g, nd, ups, n_up, "start_image");
    const ec_node_val_t *ev = flow_get_up_val(g, nd, ups, n_up, "end_image");
    if (sv && sv->image_path[0]) snprintf(val->image_path, sizeof val->image_path, "%s", sv->image_path);
    if (ev && ev->image_path[0]) snprintf(val->end_image_path, sizeof val->end_image_path, "%s", ev->image_path);
    const ec_flow_input_t *cm = ec_flow_get_input(nd, "camera_motion");
    if (cm && cm->kind == EC_VAL_STRING && cm->str_val[0]) snprintf(val->camera_motion, sizeof val->camera_motion, "%s", cm->str_val);
    if (val->width <= 0) val->width = 832; if (val->height <= 0) val->height = 480; if (val->frames <= 0) val->frames = 17;
    return 0;
  }

  // --- 2. CONDITIONING & LATENT SPEC NODES ---
  if (!strcmp(ct, "CLIPTextEncode") || !strcmp(ct, "TextEncodeQwenImage") ||
      !strcmp(ct, "PrimitiveStringMultiline") || !strcmp(ct, "FluxGuidance") ||
      !strcmp(ct, "CFGGuider") || !strcmp(ct, "BasicGuider") ||
      !strcmp(ct, "RandomNoise") || !strcmp(ct, "BasicScheduler") || !strcmp(ct, "Flux2Scheduler")) {
    return 0;
  }
  if (!strcmp(ct, "ConditioningZeroOut")) {
    val->prompt[0] = '\0';
    return 0;
  }
  if (!strcmp(ct, "TextEncodeAceStepAudio") || !strcmp(ct, "TextEncodeAceStepAudio1.5")) {
    snprintf(val->modality, sizeof val->modality, "music");
    const ec_flow_input_t *in_ks = ec_flow_get_input(nd, "key_scale");
    if (in_ks && in_ks->kind == EC_VAL_STRING && in_ks->str_val[0]) snprintf(val->key_scale, sizeof val->key_scale, "%s", in_ks->str_val);
    const ec_flow_input_t *in_vm = ec_flow_get_input(nd, "vocal_mode");
    if (in_vm && in_vm->kind == EC_VAL_STRING && in_vm->str_val[0]) snprintf(val->vocal_mode, sizeof val->vocal_mode, "%s", in_vm->str_val);
    const ec_flow_input_t *in_sm = ec_flow_get_input(nd, "stem_mode");
    if (in_sm && in_sm->kind == EC_VAL_NUMBER) { val->stem_mode = (int)in_sm->num_val; val->stem_mode_set = 1; }
    else if (in_sm && in_sm->kind == EC_VAL_STRING) {
      val->stem_mode_set = 1;
      if (!strcmp(in_sm->str_val, "8stem") || !strcmp(in_sm->str_val, "8")) val->stem_mode = 8;
      else if (!strcmp(in_sm->str_val, "4stem") || !strcmp(in_sm->str_val, "4")) val->stem_mode = 4;
      else if (!strcmp(in_sm->str_val, "2stem") || !strcmp(in_sm->str_val, "2")) val->stem_mode = 2;
      else if (!strcmp(in_sm->str_val, "full_mix") || !strcmp(in_sm->str_val, "0")) val->stem_mode = 0;
    }
    return 0;
  }
  if (!strcmp(ct, "EmptyLatentImage") || !strcmp(ct, "EmptySD3LatentImage") || !strcmp(ct, "EmptyFlux2LatentImage")) {
    if (!val->modality[0]) snprintf(val->modality, sizeof val->modality, "image");
    if (val->width <= 0) val->width = 512;
    if (val->height <= 0) val->height = 512;
    return 0;
  }
  if (!strcmp(ct, "EmptyAceStepLatentAudio") || !strcmp(ct, "EmptyAceStep1.5LatentAudio")) {
    snprintf(val->modality, sizeof val->modality, "music");
    if (val->duration_sec <= 0.0f) val->duration_sec = 10.0f;
    return 0;
  }
  if (!strcmp(ct, "Wan22ImageToVideoLatent")) {
    snprintf(val->modality, sizeof val->modality, "video");
    if (val->width <= 0) val->width = 832;
    if (val->height <= 0) val->height = 480;
    if (val->frames <= 0) val->frames = 17;
    return 0;
  }

  // --- 3. SAMPLING & NATIVE FLAGSHIP ENGINE EXECUTION ---
  const char *prompt = val->prompt[0] ? val->prompt : "A cinematic 35mm editorial studio photograph, natural light";
  uint32_t seed = val->seed ? val->seed : (uint32_t)(tag & 0xFFFFu);
  ec_studio_result_t sub = {};

  if (!strcmp(ct, "KSampler") || !strcmp(ct, "KSamplerAdvanced") || !strcmp(ct, "SamplerCustomAdvanced")) {
    const char *mod = val->modality[0] ? val->modality : "image";
    const ec_flow_input_t *in_dn = ec_flow_get_input(nd, "denoise");
    if (in_dn && in_dn->kind == EC_VAL_NUMBER) val->denoise = (float)in_dn->num_val;
    const ec_flow_input_t *in_tu = ec_flow_get_input(nd, "tile_upscale");
    if (in_tu && in_tu->kind == EC_VAL_NUMBER) val->tile_upscale = (int)in_tu->num_val;
    if (!strcmp(mod, "music")) {
      ec_studio_music_v8_t mq = {};
      mq.prompt = prompt; mq.lyrics = val->lyrics; mq.seed = seed;
      mq.bpm = (val->bpm >= 40 && val->bpm <= 240) ? val->bpm : 124;
      mq.duration_sec = (val->duration_sec >= 2.0f && val->duration_sec <= 120.0f) ? val->duration_sec : 20.0f;
      mq.steps = (val->steps >= 1 && val->steps <= 64) ? val->steps : 8;
      mq.cfg_scale = val->cfg;
      mq.checkpoint_query = val->checkpoint[0] ? val->checkpoint : NULL;
      mq.stem_mode = val->stem_mode_set ? val->stem_mode : 4;
      mq.key_scale = val->key_scale[0] ? val->key_scale : NULL;
      mq.vocal_mode = val->vocal_mode[0] ? val->vocal_mode : NULL;
      mq.audio_mode = val->audio_mode[0] ? val->audio_mode : NULL;
      mq.source_audio_path = val->audio_path[0] ? val->audio_path : NULL;
      mq.denoise = val->denoise; mq.repaint_start_sec = val->repaint_start; mq.repaint_end_sec = val->repaint_end;
      mq.loras = val->n_loras > 0 ? val->loras : NULL; mq.n_loras = val->n_loras;
      mq.tier = EC_TIER_FAST;
      mq.full_model = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : (fc->st->acestep ? 1 : 0);
      int rc = ec_studio_gen_music_v8(fc->st, &mq, flow_sub_progress_cb, fc, &sub);
      if (sub.cancelled) return -2;
      if (rc != 0) return -1;
      snprintf(val->audio_path, sizeof val->audio_path, "%s", sub.file_main);
      for (int i = 0; i < sub.n_stems && i < 8; i++) snprintf(val->stem_paths[i], sizeof val->stem_paths[i], "%s", sub.stem_files[i]);
      snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
      val->gpu_ms = sub.gpu_ms;
      snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", sub.file_main);
      fc->res->gpu_ms += sub.gpu_ms;
      return 0;
    } else if (!strcmp(mod, "video")) {
      ec_studio_video_v8_t vq = {};
      vq.prompt = prompt; vq.neg_prompt = val->neg_prompt[0] ? val->neg_prompt : NULL; vq.seed = seed;
      vq.width = val->width >= 320 ? val->width : 832;
      vq.height = val->height >= 180 ? val->height : 480;
      vq.n_frames = (val->frames >= 5 && val->frames <= 81) ? val->frames : 17;
      vq.steps = (val->steps >= 4 && val->steps <= 40) ? val->steps : 20;
      vq.cfg_scale = val->cfg > 0.0f ? val->cfg : 5.0f;
      vq.full_model = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : ((vq.width >= 320 && fc->st->wan22) ? 1 : 0);
      vq.fps = vq.full_model ? 16 : 24;
      vq.with_audio = ec_flow_get_input(nd, "with_audio") ? (int)ec_flow_get_input(nd, "with_audio")->num_val : 1;
      vq.start_frame_path = val->image_path[0] ? val->image_path : NULL;
      vq.end_frame_path = val->end_image_path[0] ? val->end_image_path : NULL;
      vq.camera_motion = val->camera_motion[0] ? val->camera_motion : NULL;
      vq.loras = val->n_loras > 0 ? val->loras : NULL; vq.n_loras = val->n_loras;
      vq.tier = EC_TIER_FAST;
      int rc = ec_studio_gen_video_v8(fc->st, &vq, flow_sub_progress_cb, fc, &sub);
      if (sub.cancelled) return -2;
      if (rc != 0) return -1;
      snprintf(val->video_path, sizeof val->video_path, "%s", sub.file_mp4);
      snprintf(val->image_path, sizeof val->image_path, "%s", sub.file_aux);
      snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
      val->gpu_ms = sub.gpu_ms;
      snprintf(fc->res->file_mp4, sizeof fc->res->file_mp4, "%s", sub.file_mp4);
      fc->res->gpu_ms += sub.gpu_ms;
      return 0;
    } else if (!strcmp(mod, "3d")) {
      ec_studio_3d_v8_t tq = {};
      tq.prompt = prompt; tq.seed = seed; tq.tier = EC_TIER_FAST;
      tq.full_model = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : (fc->st->trellis2 ? 1 : 0);
      tq.grid_res = tq.full_model ? 64 : 18;
      if (val->image_path[0]) { tq.input_images[0] = val->image_path; tq.n_input_images = 1; }
      int rc = ec_studio_gen_3d_v8(fc->st, &tq, flow_sub_progress_cb, fc, &sub);
      if (sub.cancelled) return -2;
      if (rc != 0) return -1;
      snprintf(val->glb_path, sizeof val->glb_path, "%s", sub.file_main);
      snprintf(val->ply_path, sizeof val->ply_path, "%s", sub.file_aux);
      snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
      val->gpu_ms = sub.gpu_ms;
      snprintf(fc->res->file_aux, sizeof fc->res->file_aux, "%s", sub.file_main);
      fc->res->n_verts = sub.n_verts; fc->res->n_tris = sub.n_tris; fc->res->n_splats = sub.n_splats;
      fc->res->gpu_ms += sub.gpu_ms;
      return 0;
    } else {
      ec_studio_image_v8_t iq = {};
      iq.prompt = prompt; iq.neg_prompt = val->neg_prompt[0] ? val->neg_prompt : NULL; iq.seed = seed;
      iq.width = val->width >= 256 ? val->width : 1024;
      iq.height = val->height >= 256 ? val->height : 1024;
      iq.steps = (val->steps >= 1 && val->steps <= 64) ? val->steps : 4;
      iq.cfg_scale = val->cfg;
      iq.tier = EC_TIER_EXACT;
      iq.photoreal = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : ((iq.width >= 512 && fc->st->flux2) ? 1 : 0);
      iq.checkpoint_query = val->checkpoint[0] ? val->checkpoint : NULL;
      iq.loras = val->n_loras > 0 ? val->loras : NULL; iq.n_loras = val->n_loras;
      iq.init_image_path = val->image_path[0] ? val->image_path : NULL;
      iq.mask_image_path = val->mask_path[0] ? val->mask_path : NULL;
      iq.denoise = val->denoise; iq.tile_upscale = val->tile_upscale;
      int rc = ec_studio_gen_image_v8(fc->st, &iq, flow_sub_progress_cb, fc, &sub);
      if (sub.cancelled) return -2;
      if (rc != 0) return -1;
      snprintf(val->image_path, sizeof val->image_path, "%s", sub.file_main);
      snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
      val->gpu_ms = sub.gpu_ms;
      snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", sub.file_main);
      fc->res->lora_tensors_fused += sub.lora_tensors_fused;
      fc->res->gpu_ms += sub.gpu_ms;
      return 0;
    }
  }

  // --- 4. DECODERS & DIRECT MULTI-MODAL EUREKA NODES ---
  if (!strcmp(ct, "VAEDecode") || !strcmp(ct, "EurekaImageGen")) {
    if (val->image_path[0] && !strcmp(ct, "VAEDecode")) {
      // Upstream KSampler already decoded image cleanly! Pass through in 0 ms.
      snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", val->image_path);
      return 0;
    }
    int iw = val->width >= 256 ? val->width : 256;
    int ih = val->height >= 256 ? val->height : 256;
    int is = val->steps > 0 ? val->steps : (iw >= 512 ? 4 : 6);
    int pr = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : ((iw >= 512 && fc->st->flux2) ? 1 : 0);
    int rc = ec_studio_gen_image_ex(
        fc->st, prompt, seed, iw, ih, is, EC_TIER_FAST, pr,
        val->checkpoint[0] ? val->checkpoint : NULL,
        val->n_loras > 0 ? val->loras : NULL, val->n_loras,
        flow_sub_progress_cb, fc, &sub);
    if (sub.cancelled) return -2;
    if (rc != 0) return -1;
    snprintf(val->image_path, sizeof val->image_path, "%s", sub.file_main);
    snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
    val->gpu_ms = sub.gpu_ms;
    snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", sub.file_main);
    fc->res->gpu_ms += sub.gpu_ms;
    return 0;
  }
  if (!strcmp(ct, "VAEDecodeAudio") || !strcmp(ct, "EurekaMusicGen") || !strcmp(ct, "ACEStepMusic")) {
    if (val->audio_path[0] && !strcmp(ct, "VAEDecodeAudio")) {
      snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", val->audio_path);
      return 0;
    }
    ec_studio_music_v8_t mq = {};
    mq.prompt = prompt;
    mq.lyrics = val->lyrics;
    mq.seed = seed;
    mq.bpm = (val->bpm >= 40 && val->bpm <= 240) ? val->bpm : 120;
    mq.duration_sec = (val->duration_sec >= 1.0f && val->duration_sec <= 120.0f) ? val->duration_sec : 10.0f;
    mq.steps = (val->steps >= 1 && val->steps <= 8) ? val->steps : 8;
    const ec_flow_input_t *in_sm = ec_flow_get_input(nd, "stem_mode");
    int sm = val->stem_mode > 0 ? val->stem_mode : 4;
    if (in_sm && in_sm->kind == EC_VAL_NUMBER) sm = (int)in_sm->num_val;
    else if (in_sm && in_sm->kind == EC_VAL_STRING) {
      if (!strcmp(in_sm->str_val, "8stem") || !strcmp(in_sm->str_val, "8")) sm = 8;
      else if (!strcmp(in_sm->str_val, "4stem") || !strcmp(in_sm->str_val, "4")) sm = 4;
      else if (!strcmp(in_sm->str_val, "2stem") || !strcmp(in_sm->str_val, "2")) sm = 2;
      else if (!strcmp(in_sm->str_val, "full_mix") || !strcmp(in_sm->str_val, "0")) sm = 0;
    }
    mq.stem_mode = sm;
    mq.key_scale = val->key_scale[0] ? val->key_scale : NULL;
    mq.vocal_mode = val->vocal_mode[0] ? val->vocal_mode : NULL;
    mq.audio_mode = val->audio_mode[0] ? val->audio_mode : NULL;
    mq.source_audio_path = val->audio_path[0] ? val->audio_path : NULL;
    mq.denoise = val->denoise;
    mq.repaint_start_sec = val->repaint_start;
    mq.repaint_end_sec = val->repaint_end;
    mq.loras = val->n_loras > 0 ? val->loras : NULL;
    mq.n_loras = val->n_loras;
    mq.tier = EC_TIER_FAST;
    mq.full_model = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : ((mq.duration_sec >= 4.0f && fc->st->acestep) ? 1 : 0);
    int rc = ec_studio_gen_music_v8(fc->st, &mq, flow_sub_progress_cb, fc, &sub);
    if (sub.cancelled) return -2;
    if (rc != 0) return -1;
    snprintf(val->audio_path, sizeof val->audio_path, "%s", sub.file_main);
    for (int i = 0; i < sub.n_stems && i < 8; i++) snprintf(val->stem_paths[i], sizeof val->stem_paths[i], "%s", sub.stem_files[i]);
    snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
    val->gpu_ms = sub.gpu_ms;
    snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", sub.file_main);
    fc->res->gpu_ms += sub.gpu_ms;
    return 0;
  }
  if (!strcmp(ct, "Eureka3DGen") || !strcmp(ct, "TRELLIS3D") || !strcmp(ct, "TRELLIS2ImageTo3D") || !strcmp(ct, "TrellisImageTo3D")) {
    int fm = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : (((!strcmp(ct, "TRELLIS2ImageTo3D") || !strcmp(ct, "TrellisImageTo3D")) && fc->st->trellis2) ? 1 : 0);
    ec_studio_3d_v8_t tq = {};
    tq.prompt = prompt; tq.seed = seed; tq.grid_res = fm ? 64 : 18;
    tq.steps = val->steps > 0 ? val->steps : 12;
    tq.cfg_scale = val->cfg > 0.0f ? val->cfg : 7.5f;
    tq.tier = EC_TIER_FAST; tq.full_model = fm;
    tq.isolate_fg = ec_flow_get_input(nd, "isolate_fg") ? (int)ec_flow_get_input(nd, "isolate_fg")->num_val : 1;
    if (val->image_path[0]) { tq.input_images[0] = val->image_path; tq.n_input_images = 1; }
    int rc = ec_studio_gen_3d_v8(fc->st, &tq, flow_sub_progress_cb, fc, &sub);
    if (sub.cancelled) return -2;
    if (rc != 0) return -1;
    snprintf(val->glb_path, sizeof val->glb_path, "%s", sub.file_main);
    snprintf(val->ply_path, sizeof val->ply_path, "%s", sub.file_aux);
    snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
    val->gpu_ms = sub.gpu_ms;
    snprintf(fc->res->file_aux, sizeof fc->res->file_aux, "%s", sub.file_main);
    fc->res->n_verts = sub.n_verts; fc->res->n_tris = sub.n_tris; fc->res->n_splats = sub.n_splats;
    fc->res->gpu_ms += sub.gpu_ms;
    return 0;
  }
  if (!strcmp(ct, "EurekaVideoGen") || !strcmp(ct, "LTXVideoAV") || !strcmp(ct, "Wan22VideoGen")) {
    int fm = (in_fm && in_fm->kind == EC_VAL_NUMBER) ? (int)in_fm->num_val : ((!strcmp(ct, "Wan22VideoGen") && fc->st->wan22) ? 1 : 0);
    ec_studio_video_v8_t vq = {};
    vq.prompt = prompt; vq.neg_prompt = val->neg_prompt[0] ? val->neg_prompt : NULL; vq.seed = seed;
    vq.width = val->width >= 320 ? val->width : (fm ? 832 : 320);
    vq.height = val->height >= 180 ? val->height : (fm ? 480 : 180);
    vq.n_frames = val->frames >= 5 ? val->frames : (fm ? 17 : 16);
    vq.steps = val->steps >= 4 ? val->steps : (fm ? 20 : 4);
    vq.fps = fm ? 16 : 24; vq.full_model = fm; vq.tier = EC_TIER_FAST;
    vq.with_audio = ec_flow_get_input(nd, "with_audio") ? (int)ec_flow_get_input(nd, "with_audio")->num_val : 1;
    vq.start_frame_path = val->image_path[0] ? val->image_path : NULL;
    vq.end_frame_path = val->end_image_path[0] ? val->end_image_path : NULL;
    vq.camera_motion = val->camera_motion[0] ? val->camera_motion : NULL;
    vq.loras = val->n_loras > 0 ? val->loras : NULL; vq.n_loras = val->n_loras;
    int rc = ec_studio_gen_video_v8(fc->st, &vq, flow_sub_progress_cb, fc, &sub);
    if (sub.cancelled) return -2;
    if (rc != 0) return -1;
    snprintf(val->video_path, sizeof val->video_path, "%s", sub.file_mp4);
    snprintf(val->image_path, sizeof val->image_path, "%s", sub.file_aux);
    snprintf(val->engine_mode, sizeof val->engine_mode, "%s", sub.engine_mode);
    val->gpu_ms = sub.gpu_ms;
    snprintf(fc->res->file_mp4, sizeof fc->res->file_mp4, "%s", sub.file_mp4);
    fc->res->gpu_ms += sub.gpu_ms;
    return 0;
  }

  // --- 5. REAL NATIVE PIXEL POST-FX NODES (Proves Merkle Cache Hits on Downstream Tweaks!) ---
  if (!strcmp(ct, "ImageColorGrade") || !strcmp(ct, "ImageSharpen") ||
      !strcmp(ct, "ImageGrayscale") || !strcmp(ct, "ImageInvert")) {
    if (!val->image_path[0]) return 0;
    float contrast   = ec_flow_get_input(nd, "contrast")   ? (float)ec_flow_get_input(nd, "contrast")->num_val   : 1.0f;
    float warmth     = ec_flow_get_input(nd, "warmth")     ? (float)ec_flow_get_input(nd, "warmth")->num_val     : 0.0f;
    float saturation = ec_flow_get_input(nd, "saturation") ? (float)ec_flow_get_input(nd, "saturation")->num_val : 1.0f;
    float vignette   = ec_flow_get_input(nd, "vignette")   ? (float)ec_flow_get_input(nd, "vignette")->num_val   : 0.0f;
    float sharpen    = ec_flow_get_input(nd, "sharpen")    ? (float)ec_flow_get_input(nd, "sharpen")->num_val    : (!strcmp(ct, "ImageSharpen") ? 0.65f : 0.0f);
    int grayscale    = !strcmp(ct, "ImageGrayscale") ? 1 : (ec_flow_get_input(nd, "grayscale") ? (int)ec_flow_get_input(nd, "grayscale")->num_val : 0);
    int invert       = !strcmp(ct, "ImageInvert") ? 1 : (ec_flow_get_input(nd, "invert") ? (int)ec_flow_get_input(nd, "invert")->num_val : 0);
    char dst[256];
    snprintf(dst, sizeof dst, "%s/flow_fx_%s_%04x.png", fc->st->artifact_dir, nd->id, (unsigned)(tag & 0xFFFFu));
    if (ec_png_postprocess_grade(val->image_path, dst, contrast, warmth, saturation, vignette, sharpen, grayscale, invert) == 0) {
      snprintf(val->image_path, sizeof val->image_path, "%s", dst);
      snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", dst);
    }
    return 0;
  }

  // --- 6. OUTPUT NODES (SaveImage embeds full ComfyUI JSON into PNG tEXt chunk!) ---
  if (!strcmp(ct, "SaveImage") || !strcmp(ct, "PreviewImage")) {
    if (val->image_path[0]) {
      const ec_flow_input_t *in_pre = ec_flow_get_input(nd, "filename_prefix");
      const char *pre = (in_pre && in_pre->kind == EC_VAL_STRING && in_pre->str_val[0]) ? in_pre->str_val : "EurekaFlow";
      char safe_pre[48];
      size_t j = 0;
      for (size_t i = 0; pre[i] && j + 1 < sizeof safe_pre; i++) {
        char ch = pre[i];
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')
          safe_pre[j++] = ch;
      }
      safe_pre[j] = '\0';
      if (!safe_pre[0]) snprintf(safe_pre, sizeof safe_pre, "EurekaFlow");
      char dst[256];
      snprintf(dst, sizeof dst, "%s/%s_%s_%04x.png", fc->st->artifact_dir, safe_pre, nd->id, (unsigned)(tag & 0xFFFFu));
      if (fc->api_json && ec_png_inject_comfy_metadata(val->image_path, dst, fc->api_json, NULL) == 0) {
        snprintf(val->image_path, sizeof val->image_path, "%s", dst);
      }
      snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", val->image_path);
    }
    return 0;
  }
  if (!strcmp(ct, "SaveAudio") && val->audio_path[0]) {
    snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", val->audio_path);
    return 0;
  }
  if (!strcmp(ct, "SaveVideo") && val->video_path[0]) {
    snprintf(fc->res->file_mp4, sizeof fc->res->file_mp4, "%s", val->video_path);
    return 0;
  }
  if (!strcmp(ct, "Save3D") && val->glb_path[0]) {
    snprintf(fc->res->file_aux, sizeof fc->res->file_aux, "%s", val->glb_path);
    return 0;
  }
  return 0;
}

static void st_json_esc(const char *s, char *d, size_t cap);

static int flow_val_artifacts_exist(const ec_node_val_t *v) {
  if (!v) return 0;
  if (v->image_path[0] && access(v->image_path, R_OK) != 0) return 0;
  if (v->audio_path[0] && access(v->audio_path, R_OK) != 0) return 0;
  if (v->video_path[0] && access(v->video_path, R_OK) != 0) return 0;
  if (v->glb_path[0]   && access(v->glb_path,   R_OK) != 0) return 0;
  if (v->ply_path[0]   && access(v->ply_path,   R_OK) != 0) return 0;
  return 1;
}

static int studio_dag_event_cb(
    const ec_flow_graph_t *g, int32_t step, int32_t total_steps, int32_t node_idx,
    const char *status, const ec_flow_cache_entry_t *entry, double node_ms, void *uctx) {
  flow_exec_ctx_t *fc = (flow_exec_ctx_t *)uctx;
  const ec_flow_node_t *nd = &g->nodes[node_idx];
  const ec_node_val_t *v = (entry && entry->payload && entry->payload_bytes == sizeof(ec_node_val_t))
                           ? (const ec_node_val_t *)entry->payload : NULL;
  char art_url[300] = {0};
  flow_primary_artifact_url(v, art_url, sizeof art_url);

  // Persist freshly executed DAG node into 16 KB SHA-NI Cowrie StatePage for cross-restart hits!
  if (v && !strcmp(status, "done") && nd->merkle_cid[0] && fc->st->dag_model_sha[0] && ec_cowrie_available()) {
    ec_cowrie_sp_save("dag", fc->st->dag_model_sha, nd->merkle_cid, strlen(nd->merkle_cid),
                      1u, EC_COWRIE_KV_DAG_NODEVAL, v, sizeof(ec_node_val_t));
  }

  // Sync top-level outputs even on cached hits!
  if (v && (!strcmp(status, "cached") || !strcmp(status, "done"))) {
    if (v->image_path[0]) snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", v->image_path);
    if (v->audio_path[0]) snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", v->audio_path);
    if (v->video_path[0]) snprintf(fc->res->file_mp4, sizeof fc->res->file_mp4, "%s", v->video_path);
    if (v->glb_path[0])   snprintf(fc->res->file_aux, sizeof fc->res->file_aux, "%s", v->glb_path);

    if (fc->node_outs_json && (size_t)fc->node_outs_pos + 512 < fc->node_outs_cap) {
      char e_id[64], e_ct[96], e_url[320], e_glb[320], e_ply[320], e_aud[320], e_vid[320];
      st_json_esc(nd->id, e_id, sizeof e_id);
      st_json_esc(nd->class_type, e_ct, sizeof e_ct);
      st_json_esc(art_url, e_url, sizeof e_url);
      snprintf(e_glb, sizeof e_glb, "%s%s", v->glb_path[0] ? "/artifacts/" : "", v->glb_path[0] ? flow_bname(v->glb_path) : "");
      snprintf(e_ply, sizeof e_ply, "%s%s", v->ply_path[0] ? "/artifacts/" : "", v->ply_path[0] ? flow_bname(v->ply_path) : "");
      snprintf(e_aud, sizeof e_aud, "%s%s", v->audio_path[0] ? "/artifacts/" : "", v->audio_path[0] ? flow_bname(v->audio_path) : "");
      snprintf(e_vid, sizeof e_vid, "%s%s", v->video_path[0] ? "/artifacts/" : "", v->video_path[0] ? flow_bname(v->video_path) : "");
      fc->node_outs_pos += snprintf(
          fc->node_outs_json + fc->node_outs_pos, fc->node_outs_cap - (size_t)fc->node_outs_pos,
          "%s\"%s\":{\"class_type\":\"%s\",\"status\":\"%s\",\"ms\":%.1f,\"artifact_url\":\"%s\","
          "\"glb_url\":\"%s\",\"ply_url\":\"%s\",\"audio_url\":\"%s\",\"video_url\":\"%s\",\"engine_mode\":\"%s\"}",
          fc->node_outs_cnt ? "," : "", e_id, e_ct, status, node_ms, e_url,
          e_glb, e_ply, e_aud, e_vid, v->engine_mode);
      fc->node_outs_cnt++;
    }
  }

  if (fc->event_cb) {
    return fc->event_cb(nd->id, nd->class_type, status, step, total_steps,
                        g->last_cache_hits, g->last_executed, node_ms, art_url, fc->cb_ctx);
  }
  return 0;
}

extern "C" int ec_studio_run_flow_ex(
    ec_studio_t *st, const char *comfy_json,
    ec_flow_ui_event_cb_t event_cb, void *cb_ctx,
    ec_studio_result_t *res, char *node_outputs_json, size_t node_outputs_cap,
    char *err, size_t err_cap) {
  memset(res, 0, sizeof *res);
  if (node_outputs_json && node_outputs_cap > 2) snprintf(node_outputs_json, node_outputs_cap, "{}");
  if (!st || !st->initialized || !comfy_json) return -1;

  // Auto-normalize LiteGraph UI JSON (including ComfyUI 2026 subgraphs!) into API DAG JSON if needed
  char *norm_buf = NULL;
  const char *api_src = comfy_json;
  if (strstr(comfy_json, "\"nodes\"") && strstr(comfy_json, "\"links\"")) {
    norm_buf = (char *)malloc(262144);
    if (norm_buf) {
      ec_comfy_import_info_t info;
      pthread_mutex_lock(&st->gpu_mu);
      int nrc = ec_comfy_normalize_and_resolve(comfy_json, &st->vault, norm_buf, 262144, &info);
      pthread_mutex_unlock(&st->gpu_mu);
      if (nrc == 0 && info.ok && norm_buf[0] == '{') api_src = norm_buf;
    }
  }

  if (ec_flow_parse_comfy_json(&st->flow_graph, api_src, err, err_cap) != 0) { free(norm_buf); return -1; }
  if (ec_flow_compile(&st->flow_graph, err, err_cap) != 0) { free(norm_buf); return -1; }

  // Hydrate missing RAM Merkle slots from persistent 16 KB SHA-NI Cowrie StatePages (cross-restart hits!)
  if (ec_cowrie_available() && st->dag_model_sha[0]) {
    for (int32_t step = 0; step < st->flow_graph.n_nodes; step++) {
      int32_t idx = st->flow_graph.topo_order[step];
      const ec_flow_node_t *nd = &st->flow_graph.nodes[idx];
      if (!nd->merkle_cid[0] || ec_flow_cache_has(&st->flow_graph, nd->merkle_hash)) continue;
      ec_node_val_t disk_v;
      if (ec_cowrie_sp_load("dag", st->dag_model_sha, nd->merkle_cid, strlen(nd->merkle_cid),
                            EC_COWRIE_KV_DAG_NODEVAL, &disk_v, sizeof disk_v) == 0 &&
          flow_val_artifacts_exist(&disk_v)) {
        ec_flow_preload_cache_entry(&st->flow_graph, nd->merkle_hash, nd->merkle_hash, &disk_v, sizeof disk_v);
      }
    }
  }

  flow_exec_ctx_t fc = {};
  fc.st = st;
  fc.res = res;
  fc.api_json = api_src;
  fc.event_cb = event_cb;
  fc.cb_ctx = cb_ctx;
  if (node_outputs_json && node_outputs_cap > 4) {
    fc.node_outs_json = node_outputs_json;
    fc.node_outs_cap = node_outputs_cap - 2;
    fc.node_outs_json[0] = '{';
    fc.node_outs_pos = 1;
  }

  int rc = ec_flow_execute_ex(&st->flow_graph, studio_dag_cb, studio_dag_event_cb, &fc, err, err_cap);
  if (fc.node_outs_json) {
    fc.node_outs_json[fc.node_outs_pos++] = '}';
    fc.node_outs_json[fc.node_outs_pos] = '\0';
  }
  free(norm_buf);

  res->dag_executed   = st->flow_graph.last_executed;
  res->dag_cache_hits = st->flow_graph.last_cache_hits;
  res->dag_evictions  = st->flow_graph.last_evictions;
  if (rc == -2) { res->cancelled = 1; return 0; }
  return rc;
}

extern "C" int ec_studio_run_flow_json(
    ec_studio_t *st, const char *comfy_json, ec_studio_result_t *res, char *err, size_t err_cap) {
  return ec_studio_run_flow_ex(st, comfy_json, NULL, NULL, res, NULL, 0, err, err_cap);
}

extern "C" void ec_studio_flow_clear_cache(ec_studio_t *st) {
  if (!st || !st->initialized) return;
  pthread_mutex_lock(&st->gpu_mu);
  ec_flow_clear_cache(&st->flow_graph);
  ec_cowrie_sp_purge_prefix("dag_");
  pthread_mutex_unlock(&st->gpu_mu);
}

extern "C" int ec_studio_upload_media(
    ec_studio_t *st, const uint8_t *bytes, size_t len, const char *ext,
    char *out_filename, size_t cap) {
  if (!st || !st->initialized || !bytes || len < 8 || len > 32 * 1024 * 1024) return -1;
  uint32_t crc = ec_crc32_fast(bytes, len);
  const char *e = "png";
  if (ext) {
    if (!strcmp(ext, "jpg") || !strcmp(ext, "jpeg")) e = "jpg";
    else if (!strcmp(ext, "webp")) e = "webp";
    else if (!strcmp(ext, "wav"))  e = "wav";
    else if (!strcmp(ext, "mp3"))  e = "mp3";
    else if (!strcmp(ext, "flac")) e = "flac";
    else if (!strcmp(ext, "ogg"))  e = "ogg";
  }
  char fname[96], fpath[384];
  snprintf(fname, sizeof fname, "upload_%08x.%s", crc, e);
  snprintf(fpath, sizeof fpath, "%s/%s", st->artifact_dir, fname);
  FILE *f = fopen(fpath, "wb");
  if (!f) return -1;
  size_t wr = fwrite(bytes, 1, len, f);
  fclose(f);
  if (wr != len) return -1;
  if (out_filename && cap > 0) snprintf(out_filename, cap, "%s", fname);
  return 0;
}

extern "C" int ec_studio_flow_upload_image(
    ec_studio_t *st, const uint8_t *bytes, size_t len, const char *ext,
    char *out_filename, size_t cap) {
  return ec_studio_upload_media(st, bytes, len, ext, out_filename, cap);
}

extern "C" int ec_studio_lora_catalog_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized || !buf) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_lora_catalog_to_json(&st->vault, buf, cap);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

// Emits ComfyUI-compatible typed node schema catalog (/v1/flows/object_info) hydrated with live Vault models!
extern "C" int ec_studio_flow_object_info_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized || !buf || cap < 2048) return -1;

  // Build live dropdown arrays for checkpoints & LoRAs from Multi-Drive Storage Vault
  char ckpt_opts[8192], lora_opts[4096];
  int cp = snprintf(ckpt_opts, sizeof ckpt_opts, "[\"z_image_turbo_bf16.safetensors\",\"flux2_klein_4b_bf16.safetensors\",\"flux2_klein_base_4b_bf16.safetensors\",\"acestep_v15_sft_bf16.safetensors\",\"acestep_v15_turbo_bf16.safetensors\",\"trellis2_ss_flow_1.3b_bf16.safetensors\",\"wan2.2_ti2v_5b_fp16.safetensors\"");
  int lp = snprintf(lora_opts, sizeof lora_opts, "[\"None\"");
  pthread_mutex_lock(&st->gpu_mu);
  for (int i = 0; i < st->vault.n_assets; i++) {
    const ec_storage_asset_t *a = &st->vault.assets[i];
    char esc[256];
    st_json_esc(a->name, esc, sizeof esc);
    if (a->kind == EC_ASSET_CHECKPOINT && !strstr(ckpt_opts, esc) && cp + 300 < (int)sizeof(ckpt_opts)) {
      cp += snprintf(ckpt_opts + cp, sizeof(ckpt_opts) - (size_t)cp, ",\"%s\"", esc);
    } else if (a->kind == EC_ASSET_LORA && !strstr(lora_opts, esc) && lp + 300 < (int)sizeof(lora_opts)) {
      lp += snprintf(lora_opts + lp, sizeof(lora_opts) - (size_t)lp, ",\"%s\"", esc);
    }
  }
  pthread_mutex_unlock(&st->gpu_mu);
  snprintf(ckpt_opts + cp, sizeof(ckpt_opts) - (size_t)cp, "]");
  snprintf(lora_opts + lp, sizeof(lora_opts) - (size_t)lp, "]");

  return snprintf(buf, cap,
      "{"
      "\"UNETLoader\":{\"category\":\"loaders\",\"title\":\"Load Diffusion Model\",\"outputs\":[\"MODEL\"],\"inputs\":{\"unet_name\":{\"type\":\"COMBO\",\"options\":%s,\"default\":\"flux2_klein_4b_bf16.safetensors\"}}},"
      "\"CheckpointLoaderSimple\":{\"category\":\"loaders\",\"title\":\"Load Checkpoint\",\"outputs\":[\"MODEL\",\"CLIP\",\"VAE\"],\"inputs\":{\"ckpt_name\":{\"type\":\"COMBO\",\"options\":%s,\"default\":\"flux2_klein_4b_bf16.safetensors\"}}},"
      "\"LoraLoader\":{\"category\":\"loaders\",\"title\":\"Load LoRA / QLoRA (BF16/NF4/FP8/DoRA/LoKr)\",\"outputs\":[\"MODEL\",\"CLIP\"],\"inputs\":{\"model\":{\"type\":\"MODEL\"},\"lora_name\":{\"type\":\"COMBO\",\"options\":%s,\"default\":\"None\"},\"strength_model\":{\"type\":\"FLOAT\",\"default\":0.85,\"min\":-2,\"max\":2,\"step\":0.05}}},"
      "\"QLoRALoader\":{\"category\":\"loaders\",\"title\":\"Load QLoRA / DoRA / LoHa / LoKr\",\"outputs\":[\"MODEL\"],\"inputs\":{\"model\":{\"type\":\"MODEL\"},\"lora_name\":{\"type\":\"COMBO\",\"options\":%s,\"default\":\"None\"},\"strength_model\":{\"type\":\"FLOAT\",\"default\":0.85,\"min\":-2,\"max\":2,\"step\":0.05}}},"
      "\"LoadImage\":{\"category\":\"loaders\",\"title\":\"Load Image\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"image\":{\"type\":\"IMAGE_UPLOAD\",\"default\":\"\"}}},"
      "\"LoadAudio\":{\"category\":\"audio\",\"title\":\"Load Audio (WAV/MP3/FLAC)\",\"outputs\":[\"AUDIO\"],\"inputs\":{\"audio\":{\"type\":\"AUDIO_UPLOAD\",\"default\":\"\"}}},"
      "\"VAEEncode\":{\"category\":\"latent\",\"title\":\"VAE Encode (FLUX.2 Img2Img)\",\"outputs\":[\"LATENT\"],\"inputs\":{\"pixels\":{\"type\":\"IMAGE\"},\"denoise\":{\"type\":\"FLOAT\",\"default\":0.65,\"min\":0.05,\"max\":1.0,\"step\":0.05}}},"
      "\"ImageInpaintMask\":{\"category\":\"latent\",\"title\":\"Image + Inpaint Mask\",\"outputs\":[\"LATENT\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"mask\":{\"type\":\"IMAGE_UPLOAD\",\"default\":\"\"}}},"
      "\"WanFirstLastFrameToVideo\":{\"category\":\"video\",\"title\":\"Wan 2.2 First + Last Frame (FLF2V)\",\"outputs\":[\"LATENT\"],\"inputs\":{\"start_image\":{\"type\":\"IMAGE\"},\"end_image\":{\"type\":\"IMAGE\"},\"width\":{\"type\":\"INT\",\"default\":832,\"min\":480,\"max\":1280,\"step\":32},\"height\":{\"type\":\"INT\",\"default\":480,\"min\":480,\"max\":1024,\"step\":32},\"length\":{\"type\":\"INT\",\"default\":17,\"min\":17,\"max\":81,\"step\":4},\"camera_motion\":{\"type\":\"COMBO\",\"options\":[\"auto\",\"orbit\",\"dolly_in\",\"pan_left\",\"pan_right\",\"crane_up\",\"fpv\",\"static\"],\"default\":\"auto\"}}},"
      "\"TrellisImageTo3D\":{\"category\":\"3d\",\"title\":\"TRELLIS.2 Direct Photo -> 3D\",\"outputs\":[\"MESH3D\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"steps\":{\"type\":\"INT\",\"default\":12,\"min\":4,\"max\":30},\"cfg\":{\"type\":\"FLOAT\",\"default\":7.5,\"min\":1,\"max\":12,\"step\":0.5},\"isolate_fg\":{\"type\":\"BOOL\",\"default\":true},\"seed\":{\"type\":\"INT\",\"default\":101},\"full_model\":{\"type\":\"BOOL\",\"default\":true}}},"
      "\"AceStepAudioEncode\":{\"category\":\"audio\",\"title\":\"ACE-Step Oobleck VAE Encode (Cover)\",\"outputs\":[\"LATENT\"],\"inputs\":{\"audio\":{\"type\":\"AUDIO\"},\"audio_mode\":{\"type\":\"COMBO\",\"options\":[\"cover\",\"repaint\",\"extend\"],\"default\":\"cover\"},\"denoise\":{\"type\":\"FLOAT\",\"default\":0.65,\"min\":0.1,\"max\":1.0,\"step\":0.05}}},"
      "\"AceStepRepaint\":{\"category\":\"audio\",\"title\":\"ACE-Step Audio Repaint [t0..t1]\",\"outputs\":[\"LATENT\"],\"inputs\":{\"audio\":{\"type\":\"AUDIO\"},\"start_sec\":{\"type\":\"FLOAT\",\"default\":4.0,\"min\":0,\"max\":115,\"step\":0.5},\"end_sec\":{\"type\":\"FLOAT\",\"default\":8.0,\"min\":0.5,\"max\":120,\"step\":0.5}}},"
      "\"AceStepExtend\":{\"category\":\"audio\",\"title\":\"ACE-Step Song Extender (up to 120s)\",\"outputs\":[\"LATENT\"],\"inputs\":{\"audio\":{\"type\":\"AUDIO\"},\"target_seconds\":{\"type\":\"FLOAT\",\"default\":45.0,\"min\":8,\"max\":120,\"step\":1}}},"
      "\"CLIPTextEncode\":{\"category\":\"conditioning\",\"title\":\"CLIP Text Encode (Prompt)\",\"outputs\":[\"CONDITIONING\"],\"inputs\":{\"clip\":{\"type\":\"CLIP\"},\"text\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Candid editorial portrait in a sunlit Kyoto ceramic workshop, 85mm f/1.4, Kodak Portra 400, photorealistic\"}}},"
      "\"ConditioningZeroOut\":{\"category\":\"conditioning\",\"title\":\"Conditioning Zero Out\",\"outputs\":[\"CONDITIONING\"],\"inputs\":{\"conditioning\":{\"type\":\"CONDITIONING\"}}},"
      "\"FluxGuidance\":{\"category\":\"conditioning\",\"title\":\"FLUX Guidance\",\"outputs\":[\"CONDITIONING\"],\"inputs\":{\"conditioning\":{\"type\":\"CONDITIONING\"},\"guidance\":{\"type\":\"FLOAT\",\"default\":3.5,\"min\":1,\"max\":12,\"step\":0.5}}},"
      "\"TextEncodeAceStepAudio\":{\"category\":\"audio\",\"title\":\"ACE-Step 1.5 Tags + Lyrics\",\"outputs\":[\"CONDITIONING\"],\"inputs\":{\"tags\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"warm upbeat jazz house groove, electric rhodes chords, funky bassline, crisp acoustic percussion\"},\"lyrics\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"\"},\"bpm\":{\"type\":\"INT\",\"default\":124,\"min\":50,\"max\":220}}},"
      "\"EmptyFlux2LatentImage\":{\"category\":\"latent\",\"title\":\"Empty FLUX.2 Latent\",\"outputs\":[\"LATENT\"],\"inputs\":{\"width\":{\"type\":\"INT\",\"default\":512,\"min\":256,\"max\":1536,\"step\":64},\"height\":{\"type\":\"INT\",\"default\":512,\"min\":256,\"max\":1536,\"step\":64}}},"
      "\"EmptyLatentImage\":{\"category\":\"latent\",\"title\":\"Empty Latent Image\",\"outputs\":[\"LATENT\"],\"inputs\":{\"width\":{\"type\":\"INT\",\"default\":512,\"min\":256,\"max\":1536,\"step\":64},\"height\":{\"type\":\"INT\",\"default\":512,\"min\":256,\"max\":1536,\"step\":64}}},"
      "\"EmptyAceStepLatentAudio\":{\"category\":\"audio\",\"title\":\"Empty ACE-Step Latent Audio\",\"outputs\":[\"LATENT\"],\"inputs\":{\"seconds\":{\"type\":\"FLOAT\",\"default\":10,\"min\":2,\"max\":30,\"step\":1}}},"
      "\"Wan22ImageToVideoLatent\":{\"category\":\"video\",\"title\":\"Wan 2.2 TI2V Latent\",\"outputs\":[\"LATENT\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"width\":{\"type\":\"INT\",\"default\":832,\"min\":480,\"max\":1280,\"step\":32},\"height\":{\"type\":\"INT\",\"default\":480,\"min\":480,\"max\":1024,\"step\":32},\"length\":{\"type\":\"INT\",\"default\":17,\"min\":17,\"max\":81,\"step\":4}}},"
      "\"KSampler\":{\"category\":\"sampling\",\"title\":\"KSampler (Native CUDA PTX)\",\"outputs\":[\"LATENT\"],\"inputs\":{\"model\":{\"type\":\"MODEL\"},\"positive\":{\"type\":\"CONDITIONING\"},\"negative\":{\"type\":\"CONDITIONING\"},\"latent_image\":{\"type\":\"LATENT\"},\"seed\":{\"type\":\"INT\",\"default\":1042,\"min\":0,\"max\":9999999},\"steps\":{\"type\":\"INT\",\"default\":4,\"min\":1,\"max\":64},\"cfg\":{\"type\":\"FLOAT\",\"default\":1.0,\"min\":1,\"max\":12,\"step\":0.5},\"full_model\":{\"type\":\"BOOL\",\"default\":true}}},"
      "\"VAEDecode\":{\"category\":\"decoders\",\"title\":\"VAE Decode (Image)\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"samples\":{\"type\":\"LATENT\"},\"vae\":{\"type\":\"VAE\"}}},"
      "\"VAEDecodeAudio\":{\"category\":\"audio\",\"title\":\"Oobleck 48kHz Audio Decode\",\"outputs\":[\"AUDIO\"],\"inputs\":{\"samples\":{\"type\":\"LATENT\"}}},"
      "\"ImageColorGrade\":{\"category\":\"image_fx\",\"title\":\"Film Color Grade (Pure C11)\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"contrast\":{\"type\":\"FLOAT\",\"default\":1.08,\"min\":0.4,\"max\":2.0,\"step\":0.02},\"warmth\":{\"type\":\"FLOAT\",\"default\":0.15,\"min\":-1,\"max\":1,\"step\":0.05},\"saturation\":{\"type\":\"FLOAT\",\"default\":1.12,\"min\":0,\"max\":2.0,\"step\":0.05},\"vignette\":{\"type\":\"FLOAT\",\"default\":0.25,\"min\":0,\"max\":1,\"step\":0.05},\"sharpen\":{\"type\":\"FLOAT\",\"default\":0.30,\"min\":0,\"max\":1.5,\"step\":0.05}}},"
      "\"ImageGrayscale\":{\"category\":\"image_fx\",\"title\":\"Monochrome B&W Film\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"contrast\":{\"type\":\"FLOAT\",\"default\":1.18,\"min\":0.5,\"max\":2.0,\"step\":0.02}}},"
      "\"ImageInvert\":{\"category\":\"image_fx\",\"title\":\"Invert Image\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"}}},"
      "\"TRELLIS2ImageTo3D\":{\"category\":\"3d\",\"title\":\"TRELLIS.2 1.3B (Image/Text -> 3D)\",\"outputs\":[\"MESH3D\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"sculptural matte-white porcelain teapot with curved spout and warm bamboo handle\"},\"seed\":{\"type\":\"INT\",\"default\":777,\"min\":0,\"max\":9999999},\"full_model\":{\"type\":\"BOOL\",\"default\":true}}},"
      "\"ACEStepMusic\":{\"category\":\"audio\",\"title\":\"ACE-Step 1.5 Direct Music\",\"outputs\":[\"AUDIO\"],\"inputs\":{\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"warm jazz house groove, rhodes chords, funky bass\"},\"bpm\":{\"type\":\"INT\",\"default\":124,\"min\":50,\"max\":220},\"duration_sec\":{\"type\":\"FLOAT\",\"default\":10,\"min\":2,\"max\":30},\"seed\":{\"type\":\"INT\",\"default\":4242},\"full_model\":{\"type\":\"BOOL\",\"default\":true}}},"
      "\"Wan22VideoGen\":{\"category\":\"video\",\"title\":\"Wan 2.2 TI2V-5B Cinema Video\",\"outputs\":[\"VIDEO\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"audio\":{\"type\":\"AUDIO\"},\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Aerial drone footage along black basalt sea cliffs in Iceland at golden sunset, 35mm film\"},\"width\":{\"type\":\"INT\",\"default\":832},\"height\":{\"type\":\"INT\",\"default\":480},\"frames\":{\"type\":\"INT\",\"default\":17},\"steps\":{\"type\":\"INT\",\"default\":20},\"seed\":{\"type\":\"INT\",\"default\":42},\"with_audio\":{\"type\":\"BOOL\",\"default\":true},\"full_model\":{\"type\":\"BOOL\",\"default\":true}}},"
      "\"SaveImage\":{\"category\":\"output\",\"title\":\"Save Image (+Embed ComfyUI PNG)\",\"outputs\":[],\"inputs\":{\"images\":{\"type\":\"IMAGE\"},\"filename_prefix\":{\"type\":\"STRING\",\"default\":\"EurekaFlow\"}}},"
      "\"SaveAudio\":{\"category\":\"output\",\"title\":\"Save 48kHz Audio + Stems\",\"outputs\":[],\"inputs\":{\"audio\":{\"type\":\"AUDIO\"}}},"
      "\"Save3D\":{\"category\":\"output\",\"title\":\"Save 3D GLB + 3DGS PLY\",\"outputs\":[],\"inputs\":{\"mesh\":{\"type\":\"MESH3D\"}}},"
      "\"SaveVideo\":{\"category\":\"output\",\"title\":\"Save NVENC H.264 MP4\",\"outputs\":[],\"inputs\":{\"video\":{\"type\":\"VIDEO\"}}},"
      "\"EurekaImageGen\":{\"category\":\"sampling\",\"title\":\"Eureka Photoreal Image\",\"outputs\":[\"IMAGE\"],\"inputs\":{\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Brutalist concrete architectural pavilion at dawn\"},\"seed\":{\"type\":\"INT\",\"default\":2026},\"width\":{\"type\":\"INT\",\"default\":512},\"height\":{\"type\":\"INT\",\"default\":512},\"full_model\":{\"type\":\"BOOL\",\"default\":false}}},"
      "\"Eureka3DGen\":{\"category\":\"3d\",\"title\":\"Eureka 3D O-Voxel / TRELLIS.2\",\"outputs\":[\"MESH3D\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Sculptural architectural artifact\"},\"seed\":{\"type\":\"INT\",\"default\":2026},\"full_model\":{\"type\":\"BOOL\",\"default\":false}}},"
      "\"EurekaMusicGen\":{\"category\":\"audio\",\"title\":\"Eureka 4-Stem Music\",\"outputs\":[\"AUDIO\"],\"inputs\":{\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Nordic ambient analog synth soundtrack\"},\"seed\":{\"type\":\"INT\",\"default\":2026},\"bpm\":{\"type\":\"INT\",\"default\":118},\"duration_sec\":{\"type\":\"FLOAT\",\"default\":2},\"full_model\":{\"type\":\"BOOL\",\"default\":false}}},"
      "\"EurekaVideoGen\":{\"category\":\"video\",\"title\":\"Eureka NVENC Cinema Video\",\"outputs\":[\"VIDEO\"],\"inputs\":{\"image\":{\"type\":\"IMAGE\"},\"audio\":{\"type\":\"AUDIO\"},\"prompt\":{\"type\":\"STRING\",\"multiline\":true,\"default\":\"Architectural dawn timelapse\"},\"seed\":{\"type\":\"INT\",\"default\":2026},\"width\":{\"type\":\"INT\",\"default\":320},\"height\":{\"type\":\"INT\",\"default\":180},\"frames\":{\"type\":\"INT\",\"default\":16},\"full_model\":{\"type\":\"BOOL\",\"default\":false}}}"
      "}",
      ckpt_opts, ckpt_opts, lora_opts, lora_opts);
}

// 24 Pre-Seeded Ready-to-Run Multi-Modal ComfyUI Workflow Templates (6 per modality!)
extern "C" int ec_studio_flow_templates_json(ec_studio_t *st, char *buf, size_t cap) {
  (void)st;
  return snprintf(buf, cap, "%s",
      "{\"templates\":["
      // ==================== PHOTOREAL IMAGE (6) ====================
      "{\"id\":\"flux2-photoreal-grade\",\"name\":\"FLUX.2-klein-4B + LoRA + Film Color Grade\",\"modality\":\"image\","
      "\"desc\":\"Full native ComfyUI photoreal pipeline with hot-fused LoRA/QLoRA, pure-C11 Film Color Grade (instant Merkle cache hits!), and PNG tEXt workflow embedding.\","
      "\"positions\":{\"1\":[40,40],\"2\":[310,40],\"3\":[40,210],\"4\":[310,270],\"5\":[610,110],\"6\":[900,110],\"7\":[1150,90],\"8\":[1440,110]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoraLoader\",\"inputs\":{\"model\":[\"1\",0],\"lora_name\":\"None\",\"strength_model\":0.85}},"
      "\"3\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Candid editorial portrait of a ceramic artist in a sunlit Kyoto workshop, natural window sidelight, 85mm f/1.4, Kodak Portra 400, photorealistic\"}},"
      "\"4\":{\"class_type\":\"EmptyFlux2LatentImage\",\"inputs\":{\"width\":768,\"height\":1024}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"2\",0],\"positive\":[\"3\",0],\"latent_image\":[\"4\",0],\"seed\":1042,\"steps\":4,\"cfg\":1.0,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"VAEDecode\",\"inputs\":{\"samples\":[\"5\",0]}},"
      "\"7\":{\"class_type\":\"ImageColorGrade\",\"inputs\":{\"image\":[\"6\",0],\"contrast\":1.08,\"warmth\":0.12,\"saturation\":1.10,\"vignette\":0.20,\"sharpen\":0.25}},"
      "\"8\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"7\",0],\"filename_prefix\":\"Flux2Studio\"}}"
      "}},"
      "{\"id\":\"flux2-img2img-restyle\",\"name\":\"FLUX.2 Photo-to-Photo Img2Img Restyle\",\"modality\":\"image\","
      "\"desc\":\"Load any photograph, encode through native AutoencoderKLFlux2 VAE Encoder, and re-render style & lighting with adjustable denoise strength.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[340,210],\"4\":[340,40],\"5\":[660,110],\"6\":[950,110],\"7\":[1210,110]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"3\":{\"class_type\":\"VAEEncode\",\"inputs\":{\"pixels\":[\"2\",0],\"denoise\":0.65}},"
      "\"4\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Transformed into golden-hour 35mm Leica cinema still with warm sun flare and Kodak Portra 400 grain, photorealistic\"}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"4\",0],\"latent_image\":[\"3\",0],\"seed\":2042,\"steps\":6,\"cfg\":1.0,\"denoise\":0.65,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"VAEDecode\",\"inputs\":{\"samples\":[\"5\",0]}},"
      "\"7\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"6\",0],\"filename_prefix\":\"Flux2Img2Img\"}}"
      "}},"
      "{\"id\":\"flux2-masked-inpaint\",\"name\":\"FLUX.2 Precision Masked Inpainting\",\"modality\":\"image\","
      "\"desc\":\"Paint a mask region over any photo and selectively repaint only the masked area via exact flow-matching latent projection.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[340,210],\"4\":[340,40],\"5\":[660,110],\"6\":[960,110]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"3\":{\"class_type\":\"ImageInpaintMask\",\"inputs\":{\"image\":[\"2\",0],\"mask\":\"\"}},"
      "\"4\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Replace masked region seamlessly with weathered handmade stoneware vase holding wild eucalyptus branches, matching ambient lighting\"}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"4\",0],\"latent_image\":[\"3\",0],\"seed\":3033,\"steps\":6,\"cfg\":1.0,\"denoise\":0.92,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"5\",0],\"filename_prefix\":\"Flux2Inpaint\"}}"
      "}},"
      "{\"id\":\"flux2-2x-tile-upscale\",\"name\":\"FLUX.2 2x Latent Detail Tile Upscaler\",\"modality\":\"image\","
      "\"desc\":\"Generates or loads a base frame and runs overlapping 2x tiled VAE-encode + flow-detail refinement for razor-sharp 1536px+ editorial prints.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[340,120],\"4\":[650,120],\"5\":[960,120],\"6\":[1240,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Commercial macro photograph of a brushed titanium mechanical chronograph on dark wet slate, strip softbox lighting, Canon R5 100mm macro\"}},"
      "\"3\":{\"class_type\":\"EmptyFlux2LatentImage\",\"inputs\":{\"width\":512,\"height\":512}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":777,\"steps\":4,\"tile_upscale\":true,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"ImageColorGrade\",\"inputs\":{\"image\":[\"4\",0],\"contrast\":1.05,\"sharpen\":0.30}},"
      "\"6\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"5\",0],\"filename_prefix\":\"Flux2Upscale2x\"}}"
      "}},"
      "{\"id\":\"flux2-monochrome-darkroom\",\"name\":\"Leica Monochrom Silver Gelatin Darkroom\",\"modality\":\"image\","
      "\"desc\":\"High-contrast documentary street photography pipeline with native C11 monochrome silver-gelatin darkroom finishing.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[340,240],\"4\":[640,120],\"5\":[940,120],\"6\":[1220,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Street documentary photograph of a jazz saxophonist under a rainy Paris cafe awning at twilight, Leica M11 50mm Summilux f/1.4, Ilford HP5 film grain\"}},"
      "\"3\":{\"class_type\":\"EmptyFlux2LatentImage\",\"inputs\":{\"width\":1024,\"height\":768}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":314,\"steps\":4,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"ImageGrayscale\",\"inputs\":{\"image\":[\"4\",0],\"contrast\":1.22}},"
      "\"6\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"5\",0],\"filename_prefix\":\"LeicaMono\"}}"
      "}},"
      "{\"id\":\"flux2-qlora-style-stack\",\"name\":\"FLUX.2 Multi-LoRA / QLoRA NF4 Style Stack\",\"modality\":\"image\","
      "\"desc\":\"Hot-fuses chained LoRA / QLoRA NF4 / DoRA / LoKr adapters on GPU into BF16 DiT weights with zero Python or PEFT overhead.\","
      "\"positions\":{\"1\":[40,40],\"2\":[310,40],\"3\":[40,210],\"4\":[310,270],\"5\":[610,110],\"6\":[910,110]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"QLoRALoader\",\"inputs\":{\"model\":[\"1\",0],\"lora_name\":\"None\",\"strength_model\":0.85}},"
      "\"3\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Anamorphic 65mm film still of a lone astronaut walking across an obsidian volcanic plain under two crescent moons, ARRI Alexa 65\"}},"
      "\"4\":{\"class_type\":\"EmptyFlux2LatentImage\",\"inputs\":{\"width\":1024,\"height\":576}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"2\",0],\"positive\":[\"3\",0],\"latent_image\":[\"4\",0],\"seed\":2049,\"steps\":4,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"SaveImage\",\"inputs\":{\"images\":[\"5\",0],\"filename_prefix\":\"Flux2QLoRA\"}}"
      "}},"
      // ==================== CINEMA VIDEO (6) ====================
      "{\"id\":\"wan22-cinema-av\",\"name\":\"Wan 2.2 TI2V-5B Cinema Video + Synced Audio\",\"modality\":\"video\","
      "\"desc\":\"30-block 3D RoPE Wan 2.2 TI2V-5B with isolated positive/negative prompts, 832x480x17f latent, and hardware NVENC H.264 + 48 kHz AAC mux.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,190],\"3\":[40,390],\"4\":[370,320],\"5\":[680,140],\"6\":[990,140]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"wan2.2_ti2v_5b_fp16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Aerial drone footage flying slowly along dramatic black volcanic basalt sea cliffs in Iceland at golden sunset, turquoise ocean waves, cinematic 35mm film\"}},"
      "\"3\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"blurry, static, overexposed, distorted, watermark, low quality\"}},"
      "\"4\":{\"class_type\":\"Wan22ImageToVideoLatent\",\"inputs\":{\"width\":832,\"height\":480,\"length\":17}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"negative\":[\"3\",0],\"latent_image\":[\"4\",0],\"seed\":42,\"steps\":20,\"cfg\":5.0,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"5\",0]}}"
      "}},"
      "{\"id\":\"wan22-start-frame-i2v\",\"name\":\"Wan 2.2 Start-Frame Image-to-Video (I2V)\",\"modality\":\"video\","
      "\"desc\":\"Animate any uploaded photo or FLUX.2 hero still by encoding Frame 0 via native Causal 3D VAE and anchoring the opening frame cleanly.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[340,210],\"4\":[340,40],\"5\":[680,120],\"6\":[990,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"wan2.2_ti2v_5b_fp16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"3\":{\"class_type\":\"Wan22ImageToVideoLatent\",\"inputs\":{\"image\":[\"2\",0],\"width\":832,\"height\":480,\"length\":17,\"camera_motion\":\"dolly_in\"}},"
      "\"4\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Gentle atmospheric motion, soft breeze and shifting natural cinema light, smooth camera dolly-in, 35mm film\"}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"4\",0],\"latent_image\":[\"3\",0],\"seed\":808,\"steps\":20,\"cfg\":5.0,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"5\",0]}}"
      "}},"
      "{\"id\":\"wan22-first-last-frame\",\"name\":\"Wan 2.2 First + Last Frame Keyframe Bridge (FLF2V)\",\"modality\":\"video\","
      "\"desc\":\"Pin BOTH Start Frame and End Frame in Wan 2.2 Causal 3D latent space to synthesize seamless transitions, camera orbits, or morph shots.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,190],\"3\":[40,360],\"4\":[360,240],\"5\":[690,140],\"6\":[1000,140]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"wan2.2_ti2v_5b_fp16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"3\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"4\":{\"class_type\":\"WanFirstLastFrameToVideo\",\"inputs\":{\"start_image\":[\"2\",0],\"end_image\":[\"3\",0],\"width\":832,\"height\":480,\"length\":17,\"camera_motion\":\"orbit\"}},"
      "\"5\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"latent_image\":[\"4\",0],\"prompt\":\"Seamless continuous camera glide transitioning fluidly from start frame to closing frame, 35mm cinema\",\"seed\":515,\"steps\":20,\"cfg\":5.0,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"5\",0]}}"
      "}},"
      "{\"id\":\"wan22-seamless-loop\",\"name\":\"Wan 2.2 Seamless Infinite Loop (Cinemagraph)\",\"modality\":\"video\","
      "\"desc\":\"Locks Start Frame == End Frame on a 1:1 canvas for hypnotic infinitely-looping ambient backdrops and product cinemagraphs.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[350,150],\"4\":[670,120],\"5\":[980,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"wan2.2_ti2v_5b_fp16.safetensors\"}},"
      "\"2\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"3\":{\"class_type\":\"WanFirstLastFrameToVideo\",\"inputs\":{\"start_image\":[\"2\",0],\"end_image\":[\"2\",0],\"width\":640,\"height\":640,\"length\":17,\"camera_motion\":\"static\"}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"latent_image\":[\"3\",0],\"prompt\":\"Hypnotic seamless loop, gentle ambient light shimmer and curling steam, locked tripod camera\",\"seed\":108,\"steps\":18,\"cfg\":5.0,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"wan22-vertical-9x16\",\"name\":\"Wan 2.2 Vertical 9:16 Social Reel (480x832)\",\"modality\":\"video\","
      "\"desc\":\"Native vertical 480x832 portrait cinema pipeline with synced ACE-Step 1.5 soundtrack for Reels, Shorts, and fashion stories.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[350,240],\"4\":[670,120],\"5\":[980,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"wan2.2_ti2v_5b_fp16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Vertical 9:16 editorial fashion film of silk trench coat billowing on a sunlit Milan rooftop, slow tracking shot, 35mm cinema\"}},"
      "\"3\":{\"class_type\":\"Wan22ImageToVideoLatent\",\"inputs\":{\"width\":480,\"height\":832,\"length\":17,\"camera_motion\":\"orbit\"}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":916,\"steps\":20,\"cfg\":5.0,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"flux2-to-wan22-hero\",\"name\":\"FLUX.2 Concept Still -> Wan 2.2 I2V Cinema Shot\",\"modality\":\"video\","
      "\"desc\":\"Two-model director chain: synthesize a crisp FLUX.2 hero still first, then animate it live through Wan 2.2 Causal 3D VAE I2V.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[350,120],\"4\":[660,120],\"5\":[980,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"Rain-beaded vintage bronze astrolabe glowing softly by warm candlelight inside a wood-paneled observatory, 35mm cinema\"}},"
      "\"3\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"width\":832,\"height\":480,\"seed\":303,\"steps\":4,\"full_model\":true}},"
      "\"4\":{\"class_type\":\"Wan22VideoGen\",\"inputs\":{\"image\":[\"3\",0],\"prompt\":\"Slow orbital camera dolly around glowing bronze astrolabe, flickering candlelight reflections\",\"width\":832,\"height\":480,\"frames\":17,\"steps\":18,\"seed\":303,\"with_audio\":true,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"4\",0]}}"
      "}},"
      // ==================== 3D ASSETS (6) ====================
      "{\"id\":\"trellis2-hero-3d\",\"name\":\"FLUX.2 Hero -> TRELLIS.2 1.3B 3D GLB + PLY\",\"modality\":\"3d\","
      "\"desc\":\"Two-stage multimodal graph: FLUX.2 generates a studio concept image, then TRELLIS.2 lifts it to a watertight GLB 2.0 + 3DGS PLY.\","
      "\"positions\":{\"1\":[40,60],\"2\":[40,230],\"3\":[360,120],\"4\":[660,120],\"5\":[960,100],\"6\":[1280,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"UNETLoader\",\"inputs\":{\"unet_name\":\"flux2_klein_4b_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"CLIPTextEncode\",\"inputs\":{\"text\":\"sculptural matte-white porcelain teapot with curved spout and warm bamboo handle\"}},"
      "\"3\":{\"class_type\":\"EmptyFlux2LatentImage\",\"inputs\":{\"width\":512,\"height\":512}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":777,\"steps\":4,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"TRELLIS2ImageTo3D\",\"inputs\":{\"image\":[\"4\",0],\"prompt\":\"sculptural matte-white porcelain teapot with curved spout and warm bamboo handle\",\"seed\":777,\"full_model\":true}},"
      "\"6\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"5\",0]}}"
      "}},"
      "{\"id\":\"trellis2-direct-photo\",\"name\":\"Direct Photo-to-3D Upload (Skip FLUX.2!)\",\"modality\":\"3d\","
      "\"desc\":\"Upload any product photo directly into DINOv3-ViT-L/16 + TRELLIS.2 1.3B with automatic studio backdrop isolation -> watertight GLB 2.0 + 3DGS PLY.\","
      "\"positions\":{\"1\":[40,120],\"2\":[380,120],\"3\":[740,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"\"}},"
      "\"2\":{\"class_type\":\"TrellisImageTo3D\",\"inputs\":{\"image\":[\"1\",0],\"steps\":12,\"cfg\":7.5,\"isolate_fg\":true,\"seed\":101,\"full_model\":true}},"
      "\"3\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"2\",0]}}"
      "}},"
      "{\"id\":\"trellis2-game-prop-cad\",\"name\":\"Hard-Surface Sci-Fi Game Prop (GLB + PLY)\",\"modality\":\"3d\","
      "\"desc\":\"Clean symmetric hard-surface prop generation tuned for Unity, Unreal Engine 5, and Blender PBR asset pipelines.\","
      "\"positions\":{\"1\":[40,120],\"2\":[420,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"TRELLIS2ImageTo3D\",\"inputs\":{\"prompt\":\"hard-surface orbital pilot helmet in weathered titanium and dark amber visor glass, studio product render\",\"seed\":2099,\"steps\":14,\"full_model\":true}},"
      "\"2\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"1\",0]}}"
      "}},"
      "{\"id\":\"trellis2-archviz-furniture\",\"name\":\"ArchViz Mid-Century Furniture 3D Asset\",\"modality\":\"3d\","
      "\"desc\":\"Architectural visualization furniture generator exporting clean glTF 2.0 meshes with auto-sampled PBR base albedo.\","
      "\"positions\":{\"1\":[40,120],\"2\":[420,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"TRELLIS2ImageTo3D\",\"inputs\":{\"prompt\":\"Danish mid-century sculpted walnut lounge chair with saddle tan leather cushion, neutral studio turntable\",\"seed\":1958,\"steps\":12,\"full_model\":true}},"
      "\"2\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"1\",0]}}"
      "}},"
      "{\"id\":\"trellis2-fast-ovoxel-draft\",\"name\":\"Fast O-Voxel 3D Concept Draft (< 150ms)\",\"modality\":\"3d\","
      "\"desc\":\"Ultra-fast GPU sparse O-Voxel hash-grid + Surface Nets draft for instant silhouette ideation before committing to full 1.3B flow.\","
      "\"positions\":{\"1\":[40,120],\"2\":[420,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"Eureka3DGen\",\"inputs\":{\"prompt\":\"Brutalist concrete architectural sculpture\",\"seed\":2026,\"full_model\":false}},"
      "\"2\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"1\",0]}}"
      "}},"
      "{\"id\":\"director-4-modality\",\"name\":\"4-Modality Director Master Graph (Image+3D+Music+MP4)\",\"modality\":\"3d\","
      "\"desc\":\"All 4 creative modalities linked in one compiled Merkle-cached DAG — tweak any downstream node and upstream nodes hit cache in 0 ms.\","
      "\"positions\":{\"1\":[40,150],\"2\":[360,30],\"3\":[360,270],\"4\":[690,150],\"5\":[980,30],\"6\":[980,270]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"EurekaImageGen\",\"inputs\":{\"prompt\":\"Brutalist concrete architectural pavilion over misty Norwegian fjord at dawn\",\"seed\":2026,\"width\":256,\"height\":256,\"full_model\":false}},"
      "\"2\":{\"class_type\":\"Eureka3DGen\",\"inputs\":{\"image\":[\"1\",0],\"prompt\":\"Brutalist fjord pavilion sculpture\",\"seed\":2026,\"full_model\":false}},"
      "\"3\":{\"class_type\":\"EurekaMusicGen\",\"inputs\":{\"prompt\":\"Nordic ambient analog synth soundtrack\",\"seed\":2026,\"bpm\":118,\"duration_sec\":2.0,\"full_model\":false}},"
      "\"4\":{\"class_type\":\"EurekaVideoGen\",\"inputs\":{\"image\":[\"1\",0],\"audio\":[\"3\",0],\"prompt\":\"Brutalist fjord pavilion dawn timelapse\",\"seed\":2026,\"width\":320,\"height\":180,\"frames\":16,\"full_model\":false}},"
      "\"5\":{\"class_type\":\"Save3D\",\"inputs\":{\"mesh\":[\"2\",0]}},"
      "\"6\":{\"class_type\":\"SaveVideo\",\"inputs\":{\"video\":[\"4\",0]}}"
      "}},"
      // ==================== MUSIC & AUDIO (6) ====================
      "{\"id\":\"acestep-15-8stem-band\",\"name\":\"ACE-Step 1.5 Turbo — 8-Stem Full Band Master\",\"modality\":\"music\","
      "\"desc\":\"24L GQA Sliding-Window DiT + Oobleck 48 kHz stereo decoder exporting 8 complementary stems (Drums, Bass, Vocals, Guitar, Piano, Synth, Strings, FX).\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[360,310],\"4\":[660,120],\"5\":[950,120],\"6\":[1210,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{\"ckpt_name\":\"acestep_v15_turbo_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"warm upbeat jazz house groove, electric rhodes chords, funky bassline, crisp acoustic percussion, studio mix\",\"lyrics\":\"\",\"bpm\":124,\"key_scale\":\"D minor\",\"stem_mode\":8}},"
      "\"3\":{\"class_type\":\"EmptyAceStepLatentAudio\",\"inputs\":{\"seconds\":15}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":4242,\"steps\":8,\"cfg\":1.0,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"VAEDecodeAudio\",\"inputs\":{\"samples\":[\"4\",0]}},"
      "\"6\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"5\",0]}}"
      "}},"
      "{\"id\":\"acestep-15-vocal-2stem\",\"name\":\"ACE-Step 1.5 Vocal Song — 2-Stem (Vocals + Inst)\",\"modality\":\"music\","
      "\"desc\":\"Structured lyric-to-song pipeline with [Verse]/[Chorus] tags and clean 2-stem Vocals + Instrumental separation.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[360,310],\"4\":[660,120],\"5\":[950,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{\"ckpt_name\":\"acestep_v15_turbo_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"shimmering indie dream-pop anthem, airy female vocals, jangly electric guitar, warm analog synths\",\"lyrics\":\"[Verse 1]\\nNeon rain on window glass tonight\\nChasing echoes in the city light\\n[Chorus]\\nHold the skyline till the morning comes\",\"bpm\":118,\"key_scale\":\"A major\",\"vocal_mode\":\"female\",\"stem_mode\":2}},"
      "\"3\":{\"class_type\":\"EmptyAceStepLatentAudio\",\"inputs\":{\"seconds\":20}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"1\",0],\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"seed\":404,\"steps\":8,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"acestep-15-audio-cover\",\"name\":\"ACE-Step 1.5 Audio-to-Audio Cover / Remix\",\"modality\":\"music\","
      "\"desc\":\"Upload a reference WAV/MP3 loop, encode via native Oobleck 1D Audio VAE, and transform genre & instrumentation while keeping melody & groove.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[350,210],\"4\":[680,120],\"5\":[980,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"LoadAudio\",\"inputs\":{\"audio\":\"\"}},"
      "\"2\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"lush orchestral jazz arrangement with warm Fender Rhodes, brushed snare, upright bass, vinyl warmth\",\"bpm\":112,\"stem_mode\":8}},"
      "\"3\":{\"class_type\":\"AceStepAudioEncode\",\"inputs\":{\"audio\":[\"1\",0],\"audio_mode\":\"cover\",\"denoise\":0.65}},"
      "\"4\":{\"class_type\":\"ACEStepMusic\",\"inputs\":{\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"duration_sec\":15,\"seed\":505,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"acestep-15-repaint-section\",\"name\":\"ACE-Step 1.5 Precision Audio Repaint [t0..t1]\",\"modality\":\"music\","
      "\"desc\":\"Keep intro and outro 100% locked while repainting a target time window (e.g. 4s–8s) using native ACE-Step ctx_d mask conditioning.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[360,140],\"4\":[690,140],\"5\":[980,140]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"LoadAudio\",\"inputs\":{\"audio\":\"\"}},"
      "\"2\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"explosive synth lead solo with crisp drum fill transition\",\"bpm\":124,\"stem_mode\":4}},"
      "\"3\":{\"class_type\":\"AceStepRepaint\",\"inputs\":{\"audio\":[\"1\",0],\"start_sec\":4.0,\"end_sec\":8.0}},"
      "\"4\":{\"class_type\":\"ACEStepMusic\",\"inputs\":{\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"duration_sec\":12,\"seed\":606,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"acestep-15-extend-track\",\"name\":\"ACE-Step 1.5 Long-Form Song Extender (up to 120s)\",\"modality\":\"music\","
      "\"desc\":\"Lock an existing 10s–30s clip in context latents and seamlessly compose a continuous chorus, bridge, and outro up to 2 minutes.\","
      "\"positions\":{\"1\":[40,40],\"2\":[40,210],\"3\":[360,140],\"4\":[690,140],\"5\":[980,140]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"LoadAudio\",\"inputs\":{\"audio\":\"\"}},"
      "\"2\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"evolving build into soaring euphoric chorus and warm fadeout outro\",\"bpm\":124,\"stem_mode\":8}},"
      "\"3\":{\"class_type\":\"AceStepExtend\",\"inputs\":{\"audio\":[\"1\",0],\"target_seconds\":45.0}},"
      "\"4\":{\"class_type\":\"ACEStepMusic\",\"inputs\":{\"positive\":[\"2\",0],\"latent_image\":[\"3\",0],\"duration_sec\":45,\"seed\":707,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"4\",0]}}"
      "}},"
      "{\"id\":\"acestep-15-lokr-master\",\"name\":\"ACE-Step 1.5 + LoKr Mastering Adapter (Full Mix)\",\"modality\":\"music\","
      "\"desc\":\"Clean full-mix 48 kHz studio track with hot-fused Kronecker LoKr / QLoRA adapter support and zero stem split overhead.\","
      "\"positions\":{\"1\":[40,40],\"2\":[310,40],\"3\":[40,210],\"4\":[620,120],\"5\":[940,120]},"
      "\"workflow\":{"
      "\"1\":{\"class_type\":\"CheckpointLoaderSimple\",\"inputs\":{\"ckpt_name\":\"acestep_v15_turbo_bf16.safetensors\"}},"
      "\"2\":{\"class_type\":\"QLoRALoader\",\"inputs\":{\"model\":[\"1\",0],\"lora_name\":\"None\",\"strength_model\":0.85}},"
      "\"3\":{\"class_type\":\"TextEncodeAceStepAudio\",\"inputs\":{\"tags\":\"cinematic Hans Zimmer orchestral trailer, deep resonant cellos, French horns, taiko drums, soaring piano\",\"bpm\":96,\"key_scale\":\"G minor\",\"stem_mode\":0}},"
      "\"4\":{\"class_type\":\"KSampler\",\"inputs\":{\"model\":[\"2\",0],\"positive\":[\"3\",0],\"duration_sec\":20,\"seed\":609,\"steps\":8,\"full_model\":true}},"
      "\"5\":{\"class_type\":\"SaveAudio\",\"inputs\":{\"audio\":[\"4\",0]}}"
      "}}"
      "]}");
}

// ===========================================================================
// 7. Live Multi-Architecture Safetensors Model Registry Bridge
// ===========================================================================
extern "C" int ec_studio_models_scan(ec_studio_t *st) {
  if (!st || !st->initialized) return 0;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_registry_scan(&st->registry);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_models_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_registry_to_json(&st->registry, buf, cap);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_model_load(
    ec_studio_t *st, const char *id, double *probe_ms, char *route, size_t rcap, char *err, size_t ecap) {
  if (!st || !st->initialized || !id) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_registry_load(&st->registry, id, err, ecap);
  if (rc == 0) {
    ec_loaded_model_t *lm = ec_registry_find_loaded(&st->registry, id);
    if (lm) {
      float norm = 0.0f;
      rc = ec_model_forward_probe(lm, 0.75f, EC_TIER_FAST, probe_ms, &norm, route, rcap);
    }
  }
  pthread_mutex_unlock(&st->gpu_mu);
  return rc;
}

extern "C" int ec_studio_model_unload(ec_studio_t *st, const char *id) {
  if (!st || !st->initialized || !id) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_registry_unload(&st->registry, id);
  pthread_mutex_unlock(&st->gpu_mu);
  return rc;
}

// ===========================================================================
// 8. Multi-Drive & Network-Drive Storage Vault, LoRAs & Finetune Checkpoints
// ===========================================================================
extern "C" int ec_studio_storage_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_storage_to_json(&st->vault, buf, cap);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_storage_add_root(ec_studio_t *st, const char *path, const char *label) {
  if (!st || !st->initialized || !path) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_storage_add_root(&st->vault, path, label, 1);
  if (rc >= 0) {
    ec_storage_scan(&st->vault);
    studio_try_init_all_engines_locked(st);
  }
  pthread_mutex_unlock(&st->gpu_mu);
  return rc >= 0 ? 0 : -1;
}

extern "C" int ec_studio_storage_remove_root(ec_studio_t *st, const char *path) {
  if (!st || !st->initialized || !path) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_storage_remove_root(&st->vault, path);
  if (rc == 0) ec_storage_scan(&st->vault);
  pthread_mutex_unlock(&st->gpu_mu);
  return rc;
}

extern "C" int ec_studio_storage_rescan(ec_studio_t *st) {
  if (!st || !st->initialized) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_storage_scan(&st->vault);
  studio_try_init_all_engines_locked(st);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_loras_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized || !buf) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = snprintf(buf, cap, "{\"loras\":[");
  int cnt = 0;
  for (int i = 0; i < st->vault.n_assets && (size_t)n < cap - 320; i++) {
    const ec_storage_asset_t *a = &st->vault.assets[i];
    if (a->kind != EC_ASSET_LORA) continue;
    int pairs = 0, rank = 0;
    char arch[64] = "lora", fmt[32] = "lora";
    ec_lora_inspect_ex(a->path, &pairs, &rank, arch, sizeof arch, fmt, sizeof fmt);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"path\":\"%s\",\"root\":\"%s\","
        "\"bytes\":%llu,\"is_network\":%s,\"pairs\":%d,\"rank\":%d,\"arch\":\"%s\",\"format\":\"%s\"}",
        cnt ? "," : "", a->id, a->name, a->path, a->root_label,
        (unsigned long long)a->bytes, a->is_network ? "true" : "false", pairs, rank, arch, fmt);
    cnt++;
  }
  n += snprintf(buf + n, cap - (size_t)n, "],\"count\":%d}", cnt);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

static int inspect_supported_ckpt(
    const char *path, const char *name,
    const char **out_family, const char **out_mod, const char **out_variant,
    int *out_rec_steps, float *out_rec_cfg, int *out_supports_neg) {
  if (!path || !path[0]) return 0;
  ec_safetensors_t st;
  char err[128] = {0};
  if (ec_st_open(&st, path, err, sizeof err) != 0) return 0;
  int ok = 0;
  const ec_st_entry_t *qkv = ec_st_find(&st, "double_blocks.0.img_attn.qkv.weight");
  const ec_st_entry_t *s19 = ec_st_find(&st, "single_blocks.19.linear1.weight");
  if (qkv && s19 && qkv->rank == 2 && qkv->shape[0] == 9216 && qkv->shape[1] == 3072 &&
      s19->rank == 2 && s19->shape[0] == 27648 && s19->shape[1] == 3072 &&
      (qkv->dtype == EC_DTYPE_BF16 || qkv->dtype == EC_DTYPE_F16 || qkv->dtype == EC_DTYPE_F32)) {
    int is_base = ci_substr(name, "base");
    *out_family = "flux2"; *out_mod = "image";
    *out_variant = is_base ? "base" : "distilled";
    *out_rec_steps = is_base ? 28 : 4;
    *out_rec_cfg = is_base ? 4.0f : 1.0f;
    *out_supports_neg = is_base ? 1 : 0;
    ok = 1;
  } else {
    const ec_st_entry_t *zi_qkv = ec_st_find(&st, "layers.0.attention.qkv.weight");
    const ec_st_entry_t *zi_cap = ec_st_find(&st, "cap_embedder.1.weight");
    if (zi_qkv && zi_cap && zi_qkv->rank == 2 && zi_qkv->shape[0] == 11520 && zi_qkv->shape[1] == 3840) {
      int is_base = ci_substr(name, "base");
      *out_family = "zimage"; *out_mod = "image";
      *out_variant = is_base ? "base" : "turbo";
      *out_rec_steps = is_base ? 30 : 8;
      *out_rec_cfg = is_base ? 4.0f : 1.0f;
      *out_supports_neg = is_base ? 1 : 0;
      ok = 1;
    } else {
      const ec_st_entry_t *as_q = ec_st_find(&st, "decoder.layers.0.self_attn.q_proj.weight");
      if (as_q && as_q->rank == 2 &&
          ((as_q->shape[0] == 2048 && as_q->shape[1] == 2048) ||
           (as_q->shape[0] == 4096 && as_q->shape[1] == 2560))) {
        int is_xl  = (as_q->shape[1] == 2560) ? 1 : 0;
        int is_sft = ci_substr(name, "sft") || (ec_st_find(&st, "null_condition_emb") != NULL && !ci_substr(name, "turbo"));
        *out_family = "acestep"; *out_mod = "music";
        *out_variant = is_xl ? (is_sft ? "xl_sft" : "xl_turbo") : (is_sft ? "sft" : "turbo");
        *out_rec_steps = is_sft ? (is_xl ? 40 : 32) : 8;
        *out_rec_cfg = is_sft ? (is_xl ? 7.0f : 6.5f) : 1.0f;
        *out_supports_neg = 0;
        ok = 1;
      } else if (ec_st_find(&st, "model.diffusion_model.transformer_blocks.47.attn1.to_q.weight") &&
                 ec_st_find(&st, "vocoder.bwe_generator.conv_post.weight")) {
        *out_family = "ltx2"; *out_mod = "video";
        *out_variant = "distilled_av";
        *out_rec_steps = 8;
        *out_rec_cfg = 1.0f;
        *out_supports_neg = 0;
        ok = 1;
      } else if (ec_st_find(&st, "blocks.0.self_attn.q.weight") &&
                 ec_st_find(&st, "patch_embedding.weight")) {
        *out_family = "wan22"; *out_mod = "video";
        *out_variant = "ti2v_5b";
        *out_rec_steps = 30;
        *out_rec_cfg = 4.0f;
        *out_supports_neg = 1;
        ok = 1;
      }
    }
  }
  ec_st_close(&st);
  return ok;
}

extern "C" int ec_studio_checkpoints_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized || !buf) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = snprintf(buf, cap, "{\"checkpoints\":[");
  int cnt = 0;
  char seen_ids[64][128];
  int n_seen = 0;
  for (int i = 0; i < st->vault.n_assets && (size_t)n < cap - 420; i++) {
    const ec_storage_asset_t *a = &st->vault.assets[i];
    if (a->kind != EC_ASSET_CHECKPOINT) continue;
    int dup = 0;
    for (int j = 0; j < n_seen; j++) {
      if (!strcmp(seen_ids[j], a->id)) { dup = 1; break; }
    }
    if (dup) continue;
    const char *fam = "", *mod = "", *var = "";
    int rec_steps = 4, supports_neg = 0;
    float rec_cfg = 1.0f;
    if (!inspect_supported_ckpt(a->path, a->name, &fam, &mod, &var, &rec_steps, &rec_cfg, &supports_neg)) continue;
    if (n_seen < 64) snprintf(seen_ids[n_seen++], sizeof seen_ids[0], "%s", a->id);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"path\":\"%s\",\"root\":\"%s\","
        "\"bytes\":%llu,\"is_network\":%s,\"family\":\"%s\",\"modality\":\"%s\","
        "\"variant\":\"%s\",\"rec_steps\":%d,\"rec_cfg\":%.1f,\"supports_neg\":%s}",
        cnt ? "," : "", a->id, a->name, a->path, a->root_label,
        (unsigned long long)a->bytes, a->is_network ? "true" : "false",
        fam, mod, var, rec_steps, rec_cfg, supports_neg ? "true" : "false");
    cnt++;
  }
  n += snprintf(buf + n, cap - (size_t)n, "],\"count\":%d}", cnt);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

// ===========================================================================
// 9. Photoreal Prompt Library, User Recipes & Universal ComfyUI Importer
// ===========================================================================
extern "C" int ec_studio_recipes_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = ec_recipes_to_json(&st->recipes, buf, cap);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_prompts_json(char *buf, size_t cap) {
  return ec_prompt_library_to_json(buf, cap);
}

extern "C" int ec_studio_recipe_save(
    ec_studio_t *st, const char *id, const char *name, const char *category,
    const char *modality, const char *prompt, const char *camera,
    const char *checkpoint, const char *lora, float lora_strength,
    int width, int height, int steps, uint32_t seed) {
  if (!st || !st->initialized) return -1;
  ec_recipe_t r = {};
  snprintf(r.id, sizeof r.id, "%s", id ? id : "");
  snprintf(r.name, sizeof r.name, "%s", name ? name : "");
  snprintf(r.category, sizeof r.category, "%s", (category && category[0]) ? category : "Custom");
  snprintf(r.modality, sizeof r.modality, "%s", (modality && modality[0]) ? modality : "image");
  snprintf(r.prompt, sizeof r.prompt, "%s", prompt ? prompt : "");
  snprintf(r.camera, sizeof r.camera, "%s", camera ? camera : "");
  snprintf(r.checkpoint, sizeof r.checkpoint, "%s", checkpoint ? checkpoint : "");
  snprintf(r.lora, sizeof r.lora, "%s", lora ? lora : "");
  r.lora_strength = lora_strength;
  r.width = width > 0 ? width : 512;
  r.height = height > 0 ? height : 512;
  r.steps = steps > 0 ? steps : 4;
  r.seed = seed;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_recipes_save_user(&st->recipes, &r);
  pthread_mutex_unlock(&st->gpu_mu);
  return rc;
}

extern "C" int ec_studio_recipe_delete(ec_studio_t *st, const char *id) {
  if (!st || !st->initialized) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int rc = ec_recipes_delete_user(&st->recipes, id);
  pthread_mutex_unlock(&st->gpu_mu);
  return rc;
}

static void st_json_esc(const char *src, char *dst, size_t cap) {
  size_t ej = 0;
  for (size_t i = 0; src && src[i] && ej + 2 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (c == '"' || c == '\\') { dst[ej++] = '\\'; dst[ej++] = (char)c; }
    else if (c == '\n') { dst[ej++] = '\\'; dst[ej++] = 'n'; }
    else if (c == '\r') { dst[ej++] = '\\'; dst[ej++] = 'r'; }
    else if (c == '\t') { dst[ej++] = '\\'; dst[ej++] = 't'; }
    else if (c >= 0x20) dst[ej++] = (char)c;
  }
  dst[ej] = '\0';
}

extern "C" int ec_studio_import_comfy_ex(
    ec_studio_t *st, const uint8_t *payload, size_t payload_len, int is_png, int execute_now,
    const char *source_tag, char *out_json, size_t out_cap) {
  if (!st || !st->initialized || !payload || !payload_len || !out_json) return -1;
  char *raw_json = (char *)malloc(262144);
  char *api_json = (char *)malloc(262144);
  if (!raw_json || !api_json) { free(raw_json); free(api_json); return -1; }

  if (is_png) {
    int n = ec_png_extract_comfy_json(payload, payload_len, raw_json, 262144);
    if (n <= 0) {
      snprintf(out_json, out_cap, "{\"ok\":false,\"error\":\"no embedded ComfyUI prompt/workflow chunk found in PNG\"}");
      free(raw_json); free(api_json);
      return -1;
    }
  } else {
    size_t cp = payload_len < 262143 ? payload_len : 262143;
    memcpy(raw_json, payload, cp);
    raw_json[cp] = '\0';
  }

  ec_comfy_import_info_t info;
  pthread_mutex_lock(&st->gpu_mu);
  int nrc = ec_comfy_normalize_and_resolve(raw_json, &st->vault, api_json, 262144, &info);
  pthread_mutex_unlock(&st->gpu_mu);
  free(raw_json);

  if (nrc != 0 || !info.ok) {
    snprintf(out_json, out_cap, "{\"ok\":false,\"error\":\"%s\"}", info.error[0] ? info.error : "invalid ComfyUI workflow");
    free(api_json);
    return -1;
  }

  char artifact_url[256] = {0};
  double exec_ms = 0.0;
  char exec_mode[32] = "";
  if (execute_now) {
    ec_studio_result_t r = {};
    int grc = -1;
    const char *run_prompt = info.extracted_prompt[0] ? info.extracted_prompt
                             : (info.input_image_hint[0] ? info.input_image_hint : "eureka imported studio scene");
    if (strcmp(info.modality, "music") == 0) {
      float dur = (info.duration_sec >= 2.0f && info.duration_sec <= 15.0f) ? info.duration_sec : 10.0f;
      int bpm = (info.bpm >= 40 && info.bpm <= 240) ? info.bpm : 124;
      grc = ec_studio_gen_music_ex(st, run_prompt, info.extracted_lyrics, info.seed, bpm, dur, EC_TIER_FAST, 1, NULL, NULL, &r);
      if (grc == 0) {
        const char *bn = strrchr(r.file_main, '/');
        snprintf(artifact_url, sizeof artifact_url, "/artifacts/%s", bn ? bn + 1 : r.file_main);
      }
    } else if (strcmp(info.modality, "video") == 0) {
      int vw = (info.width >= 640 && info.width <= 1280) ? info.width : 832;
      int vh = (info.height >= 384 && info.height <= 768) ? info.height : 480;
      if (vw > 832) { vw = 832; vh = 480; }
      int vf = (info.frames >= 5 && info.frames <= 33) ? info.frames : 17;
      int vs = (info.steps >= 8 && info.steps <= 24) ? info.steps : 20;
      grc = ec_studio_gen_video_ex(st, run_prompt, info.seed, vf, vw, vh, 16, vs, 1, EC_TIER_FAST, 1, NULL, NULL, &r);
      if (grc == 0) {
        const char *bn = strrchr(r.file_mp4, '/');
        snprintf(artifact_url, sizeof artifact_url, "/artifacts/%s", bn ? bn + 1 : r.file_mp4);
      }
    } else if (strcmp(info.modality, "3d") == 0) {
      grc = ec_studio_gen_3d_ex(st, run_prompt, info.seed, 64, EC_TIER_FAST, 1, NULL, NULL, &r);
      if (grc == 0) {
        const char *bn = strrchr(r.file_main, '/');
        snprintf(artifact_url, sizeof artifact_url, "/artifacts/%s", bn ? bn + 1 : r.file_main);
      }
    } else if (info.extracted_prompt[0]) {
      ec_studio_lora_t loras[4] = {};
      int nl = 0;
      for (int i = 0; i < info.n_loras && nl < 4; i++) {
        if (info.resolved_loras[i][0]) {
          snprintf(loras[nl].path, sizeof loras[nl].path, "%s", info.resolved_loras[i]);
          loras[nl].strength = info.extracted_lora_strengths[i];
          nl++;
        }
      }
      grc = ec_studio_gen_image_ex(
          st, info.extracted_prompt, info.seed,
          info.width >= 256 ? info.width : 512,
          info.height >= 256 ? info.height : 512,
          info.steps > 0 && info.steps <= 12 ? info.steps : 4,
          EC_TIER_FAST, 1,
          info.resolved_checkpoint[0] ? info.resolved_checkpoint : NULL,
          nl > 0 ? loras : NULL, nl, NULL, NULL, &r);
      if (grc == 0) {
        const char *bn = strrchr(r.file_main, '/');
        snprintf(artifact_url, sizeof artifact_url, "/artifacts/%s", bn ? bn + 1 : r.file_main);
      }
    }
    if (grc == 0) {
      exec_ms = r.gpu_ms;
      snprintf(exec_mode, sizeof exec_mode, "%s", r.engine_mode);
    }
  }

  char ep[1536], eneg[1536], elyr[2048], ehint[384];
  st_json_esc(info.extracted_prompt, ep, sizeof ep);
  st_json_esc(info.extracted_neg_prompt, eneg, sizeof eneg);
  st_json_esc(info.extracted_lyrics, elyr, sizeof elyr);
  st_json_esc(info.input_image_hint, ehint, sizeof ehint);

  int pos = snprintf(out_json, out_cap,
      "{\"ok\":true,\"source_format\":\"%s\",\"modality\":\"%s\",\"node_count\":%d,\"subgraph_count\":%d,"
      "\"extracted\":{\"modality\":\"%s\",\"prompt\":\"%s\",\"neg_prompt\":\"%s\",\"lyrics\":\"%s\","
      "\"checkpoint\":\"%s\",\"resolved_checkpoint\":\"%s\","
      "\"width\":%d,\"height\":%d,\"steps\":%d,\"frames\":%d,\"bpm\":%d,\"duration_sec\":%.1f,"
      "\"needs_image_input\":%s,\"input_image_hint\":\"%s\",\"seed\":%u,\"loras\":[",
      info.source_format, info.modality, info.node_count, info.subgraph_count,
      info.modality, ep, eneg, elyr,
      info.extracted_checkpoint, info.resolved_checkpoint,
      info.width, info.height, info.steps, info.frames, info.bpm, info.duration_sec,
      info.needs_image_input ? "true" : "false", ehint, info.seed);
  for (int i = 0; i < info.n_loras && (size_t)pos < out_cap - 512; i++) {
    pos += snprintf(out_json + pos, out_cap - (size_t)pos,
        "%s{\"name\":\"%s\",\"resolved\":\"%s\",\"strength\":%.2f}",
        i ? "," : "", info.extracted_loras[i], info.resolved_loras[i], info.extracted_lora_strengths[i]);
  }
  // Frame normalized ComfyUI workflow under Cowrie v2 Trust Membrane (UNTRUSTED_EXT = 0x04 unless local verified)
  const char *src_t = (source_tag && source_tag[0]) ? source_tag : "external";
  uint8_t base_auth = (!strcmp(src_t, "local") || !strcmp(src_t, "template"))
                      ? EC_COWRIE_AUTH_VERIFIED_TOOL : EC_COWRIE_AUTH_UNTRUSTED_EXT;
  char cw_label[96];
  snprintf(cw_label, sizeof cw_label, "%s:%s", info.source_format, info.modality);
  ec_cowrie_trust_meta_t cw_meta = {};
  ec_cowrie_trust_register_json(api_json, strlen(api_json), base_auth, src_t, cw_label, NULL, 0, &cw_meta);
  char e_r0[320], e_r1[1536], e_src_t[64];
  st_json_esc(cw_meta.r0, e_r0, sizeof e_r0);
  st_json_esc(cw_meta.r1, e_r1, sizeof e_r1);
  st_json_esc(src_t, e_src_t, sizeof e_src_t);

  pos += snprintf(out_json + pos, out_cap - (size_t)pos,
      "]},\"executed\":%s,\"artifact_url\":\"%s\",\"engine_mode\":\"%s\",\"gpu_ms\":%.2f,"
      "\"cowrie\":{\"enabled\":%s,\"shani\":%s,\"cid\":\"%s\",\"handle\":\"%s\","
      "\"authority\":%u,\"tainted\":%s,\"source\":\"%s\",\"raw_bytes\":%zu,\"cowr_bytes\":%zu,"
      "\"r0\":\"%s\",\"r1\":\"%s\"},"
      "\"api_workflow\":%s}",
      artifact_url[0] ? "true" : "false", artifact_url, exec_mode, exec_ms,
      ec_cowrie_available() ? "true" : "false",
      ec_cowrie_has_shani() ? "true" : "false",
      cw_meta.cid_hex, cw_meta.handle, (unsigned)cw_meta.authority,
      cw_meta.tainted ? "true" : "false", e_src_t, cw_meta.raw_bytes, cw_meta.cowr_bytes,
      e_r0, e_r1,
      strlen(api_json) < (out_cap - (size_t)pos - 32) ? api_json : "{}");
  free(api_json);
  return 0;
}

extern "C" int ec_studio_import_comfy(
    ec_studio_t *st, const uint8_t *payload, size_t payload_len, int is_png, int execute_now,
    char *out_json, size_t out_cap) {
  return ec_studio_import_comfy_ex(st, payload, payload_len, is_png, execute_now, "external", out_json, out_cap);
}

// ===========================================================================
// 10. Tessera / Cowrie v2 Director LLM Lenses (R0/R1/R2) & Trust Membrane Gate
// ===========================================================================
extern "C" int ec_studio_cowrie_lens_json(
    ec_studio_t *st, const char *target, int tier, char *out_json, size_t out_cap) {
  if (!st || !st->initialized || !out_json || out_cap < 512) return -1;
  const char *tgt = (target && target[0]) ? target : "vault";
  char *raw = (char *)malloc(262144);
  if (!raw) return -1;
  raw[0] = '\0';

  int rc = -1;
  if (!strcmp(tgt, "vault") || !strcmp(tgt, "storage")) {
    rc = ec_studio_storage_json(st, raw, 262144) >= 0 ? 0 : -1;
  } else if (!strcmp(tgt, "recipes")) {
    rc = ec_studio_recipes_json(st, raw, 262144) >= 0 ? 0 : -1;
  } else if (!strcmp(tgt, "models")) {
    rc = ec_studio_models_json(st, raw, 262144) >= 0 ? 0 : -1;
  } else if (!strcmp(tgt, "templates")) {
    rc = ec_studio_flow_templates_json(st, raw, 262144) >= 0 ? 0 : -1;
  } else if (!strcmp(tgt, "prompts")) {
    rc = ec_studio_prompts_json(raw, 262144) >= 0 ? 0 : -1;
  } else if (!strcmp(tgt, "triage")) {
    rc = ec_cowrie_trust_triage_json(raw, 262144) >= 0 ? 0 : -1;
  } else {
    free(raw);
    snprintf(out_json, out_cap, "{\"ok\":false,\"error\":\"unknown lens target (use vault|recipes|models|templates|prompts|triage)\"}");
    return -1;
  }

  if (rc != 0 || !raw[0]) {
    free(raw);
    snprintf(out_json, out_cap, "{\"ok\":false,\"error\":\"failed to gather lens target payload\"}");
    return -1;
  }
  int n = ec_cowrie_lens_response_json(tgt, raw, strlen(raw), EC_COWRIE_AUTH_VERIFIED_TOOL, tier, out_json, out_cap);
  free(raw);
  return n >= 0 ? 0 : -1;
}

extern "C" int ec_studio_cowrie_trust_gate_json(
    ec_studio_t *st, const char *action, const char *const *arg_ids, int n_args,
    int owner_override, char *out_json, size_t out_cap) {
  (void)st;
  return ec_cowrie_trust_gate_json(action, arg_ids, n_args, owner_override, out_json, out_cap);
}

extern "C" int ec_studio_cowrie_trust_triage_json(ec_studio_t *st, char *out_json, size_t out_cap) {
  (void)st;
  return ec_cowrie_trust_triage_json(out_json, out_cap);
}
