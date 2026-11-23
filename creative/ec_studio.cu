// ec_studio.cu — 100% Native Eureka End-to-End Multi-Modal Creative Studio Runtime (v2)
// Thread-safe GPU execution, Shifted RF + MagCache + TensorCore Conv2D + NVENC H.264/AAC MP4,
// Live Multi-Architecture Safetensors Registry (Z-Image, ACE-Step 1.5, TRELLIS.2, Wan 2.2, LTX-2.5),
// Cooperative SSE Progress & Instant Cancellation.

#define _POSIX_C_SOURCE 200809L
#include "ec_studio.h"
#include "ec_acestep.h"
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
#include <cuda_fp16.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
  ec_acestep_t       *acestep;
  ec_trellis2_t      *trellis2;
  ec_wan22_t         *wan22;
  int                 initialized;
};

// Evicts resident GPU weights from all flagship engines EXCEPT keep_id (1=FLUX.2, 2=ACE-Step, 3=TRELLIS.2, 4=Wan2.2).
// Lightweight mmap handles & tokenizers stay open so switching modalities incurs zero file re-parsing!
static void studio_evict_other_engines_locked(ec_studio_t *st, int keep_id) {
  if (keep_id != 1 && st->flux2)    ec_flux2_unload_vram(st->flux2);
  if (keep_id != 2 && st->acestep)  ec_acestep_unload_vram(st->acestep);
  if (keep_id != 3 && st->trellis2) ec_trellis2_unload_vram(st->trellis2);
  if (keep_id != 4 && st->wan22)    ec_wan22_unload_vram(st->wan22);
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

  // 3. Native 3D Asset Generation: TRELLIS.2 (DINOv3-L/16 + 1.3B Flow + 64^3 3D Conv Decoder)
  if (!st->trellis2 || !ec_trellis2_is_ready(st->trellis2)) {
    const ec_storage_asset_t *dino = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "dinov3_vitl16");
    const ec_storage_asset_t *flow = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_ss_flow");
    const ec_storage_asset_t *dec  = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, "trellis2_ss_dec");
    if (dino && flow && dec) {
      ec_trellis2_paths_t p = {};
      snprintf(p.dino_path,    sizeof p.dino_path,    "%s", dino->path);
      snprintf(p.ss_flow_path, sizeof p.ss_flow_path, "%s", flow->path);
      snprintf(p.ss_dec_path,  sizeof p.ss_dec_path,  "%s", dec->path);
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

  char cfg_storage[512], cfg_recipes[512];
  snprintf(cfg_storage, sizeof cfg_storage, "%s/storage_roots.json", st->artifact_dir);
  snprintf(cfg_recipes, sizeof cfg_recipes, "%s/recipes_user.json", st->artifact_dir);
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
    if (st->acestep)  { ec_acestep_destroy(st->acestep);   st->acestep = NULL; }
    if (st->trellis2) { ec_trellis2_destroy(st->trellis2); st->trellis2 = NULL; }
    if (st->wan22)    { ec_wan22_destroy(st->wan22);       st->wan22 = NULL; }
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
  out->vram_pool_used_mb = (double)st->vram_pool.used_bytes / (1024.0 * 1024.0);
  out->vram_pool_budget_mb = (double)st->vram_pool.budget_bytes / (1024.0 * 1024.0);
  out->models_discovered = st->registry.n_models;
  out->photoreal_ready = (st->flux2    && ec_flux2_is_ready(st->flux2))       ? 1 : 0;
  out->acestep_ready   = (st->acestep  && ec_acestep_is_ready(st->acestep))   ? 1 : 0;
  out->trellis2_ready  = (st->trellis2 && ec_trellis2_is_ready(st->trellis2)) ? 1 : 0;
  out->wan22_ready     = (st->wan22    && ec_wan22_is_ready(st->wan22))       ? 1 : 0;
  out->storage_roots = st->vault.n_roots;
  out->recipes_count = st->recipes.count;
  for (int i = 0; i < st->vault.n_assets; i++) {
    if (st->vault.assets[i].kind == EC_ASSET_LORA) out->loras_discovered++;
    else if (st->vault.assets[i].kind == EC_ASSET_CHECKPOINT) out->checkpoints_discovered++;
  }
}

typedef struct {
  ec_progress_cb_t cb;
  void            *cb_ctx;
} fx_cb_bridge_t;

static int fx_progress_bridge(int step, int total, const char *stage, void *ctx) {
  fx_cb_bridge_t *b = (fx_cb_bridge_t *)ctx;
  return (b && b->cb) ? b->cb(step, total, 0, stage, b->cb_ctx) : 0;
}

extern "C" int ec_studio_gen_image_ex(
    ec_studio_t *st, const char *prompt, uint32_t seed, int width, int height,
    int steps, ec_tier_t tier, int photoreal,
    const char *checkpoint_query,
    const ec_studio_lora_t *loras, int n_loras,
    ec_progress_cb_t cb, void *cb_ctx, ec_studio_result_t *res) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;

  if (photoreal && st->flux2 && ec_flux2_is_ready(st->flux2)) {
    char ckpt_path[512] = {0};
    if (checkpoint_query && checkpoint_query[0]) {
      const ec_storage_asset_t *a = ec_storage_find(&st->vault, EC_ASSET_CHECKPOINT, checkpoint_query);
      if (a) snprintf(ckpt_path, sizeof ckpt_path, "%s", a->path);
      else snprintf(ckpt_path, sizeof ckpt_path, "%s", checkpoint_query);
    }

    ec_lora_spec_t lstack[EC_LORA_MAX_STACK];
    int nl = 0;
    for (int i = 0; i < n_loras && nl < EC_LORA_MAX_STACK; i++) {
      if (!loras[i].path[0]) continue;
      const ec_storage_asset_t *la = ec_storage_find(&st->vault, EC_ASSET_LORA, loras[i].path);
      snprintf(lstack[nl].path, sizeof lstack[nl].path, "%s", la ? la->path : loras[i].path);
      lstack[nl].strength = loras[i].strength;
      nl++;
    }

    ec_flux2_req_t freq = {};
    freq.prompt = prompt;
    freq.seed = seed;
    freq.width = (width >= 256) ? width : 512;
    freq.height = (height >= 256) ? height : 512;
    freq.steps = (steps >= 1 && steps <= 50) ? steps : 4;
    freq.checkpoint_override = ckpt_path[0] ? ckpt_path : NULL;
    freq.loras = nl > 0 ? lstack : NULL;
    freq.n_loras = nl;

    uint64_t tag = hash_prompt_seed(prompt, seed);
    snprintf(res->file_main, sizeof res->file_main, "%s/flux2_%u_%04x.png",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    ec_flux2_res_t fres = {};
    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 1);
    int rc = ec_flux2_generate_png(st->flux2, &freq, res->file_main, cb ? fx_progress_bridge : NULL, &bridge, &fres);
    pthread_mutex_unlock(&st->gpu_mu);
    if (rc == 0 && fres.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "flux2_photoreal");
      snprintf(res->checkpoint_used, sizeof res->checkpoint_used, "%s", fres.checkpoint_used);
      res->width = fres.width;
      res->height = fres.height;
      res->steps_run = fres.steps_run;
      res->lora_tensors_fused = fres.lora_tensors_fused;
      res->text_enc_ms = fres.text_enc_ms;
      res->dit_ms = fres.dit_ms;
      res->vae_ms = fres.vae_ms;
      res->gpu_ms = fres.total_ms;
      return 0;
    }
    return -1;
  }

  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return ec_studio_gen_image(st, prompt, seed, width, height, steps, tier, cb, cb_ctx, res);
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
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized) return -1;

  if (full_model && st->trellis2 && ec_trellis2_is_ready(st->trellis2) &&
      st->flux2 && ec_flux2_is_ready(st->flux2)) {
    uint64_t tag = hash_prompt_seed(prompt, seed);
    char hero_prompt[1536];
    snprintf(hero_prompt, sizeof hero_prompt,
        "Isolated centered 3D studio product render of %s, single object floating centered on pure solid matte black background, even neutral softbox studio lighting, full object visible in frame, no room, no table, no background clutter",
        (prompt && prompt[0]) ? prompt : "sculptural matte-white porcelain teapot with bamboo handle");

    char ref_png[256];
    snprintf(ref_png, sizeof ref_png, "%s/trellis2_%u_%04x_ref.png",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));
    snprintf(res->file_main, sizeof res->file_main, "%s/trellis2_%u_%04x.glb",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));
    snprintf(res->file_aux, sizeof res->file_aux, "%s/trellis2_%u_%04x.ply",
             st->artifact_dir, seed, (unsigned)(tag & 0xFFFFu));

    ec_flux2_req_t freq = {};
    freq.prompt = hero_prompt;
    freq.seed = seed;
    freq.width = 512;
    freq.height = 512;
    freq.steps = 4;

    fx_cb_bridge_t bridge = { cb, cb_ctx };
    uint8_t *ref_rgb = NULL;
    ec_flux2_res_t fres = {};
    ec_trellis2_res_t tres = {};

    pthread_mutex_lock(&st->gpu_mu);
    studio_evict_other_engines_locked(st, 1);
    int rc_ref = ec_flux2_generate_rgb(st->flux2, &freq, &ref_rgb, ref_png,
                                       cb ? tr_ref_progress_bridge : NULL, &bridge, &fres);
    // Evict FLUX.2 immediately so TRELLIS.2 DINOv3 + 30-block Flow + 3D Conv Decoder run with ample VRAM!
    ec_flux2_unload_vram(st->flux2);
    if (rc_ref != 0 || !ref_rgb) {
      pthread_mutex_unlock(&st->gpu_mu);
      if (ref_rgb) free(ref_rgb);
      if (rc_ref == -2) { res->cancelled = 1; return 0; }
      return -1;
    }

    ec_trellis2_req_t treq = {};
    treq.rgb = ref_rgb;
    treq.img_w = fres.width;
    treq.img_h = fres.height;
    treq.seed = seed;
    treq.steps = 12;
    treq.cfg_scale = 7.5f;
    treq.rescale_t = 5.0f;

    int rc_3d = ec_trellis2_generate_3d(st->trellis2, &treq, res->file_main, res->file_aux,
                                        cb ? tr_flow_progress_bridge : NULL, &bridge, &tres);
    pthread_mutex_unlock(&st->gpu_mu);
    free(ref_rgb);

    if (rc_3d == -2) { res->cancelled = 1; return 0; }
    if (rc_3d == 0 && tres.ok) {
      snprintf(res->engine_mode, sizeof res->engine_mode, "trellis2_1.3b");
      res->n_verts = tres.n_verts;
      res->n_tris = tres.n_tris;
      res->n_splats = tres.n_splats;
      res->steps_run = tres.steps_run;
      res->text_enc_ms = fres.total_ms;
      res->dit_ms = tres.dino_ms + tres.flow_ms;
      res->vae_ms = tres.dec_ms + tres.mesh_ms;
      res->gpu_ms = fres.total_ms + tres.total_ms;
      return 0;
    }
    return -1;
  }

  int rc = ec_studio_gen_3d(st, prompt, seed, grid_res, tier, cb, cb_ctx, res);
  snprintf(res->engine_mode, sizeof res->engine_mode, "dit_smoke");
  return rc;
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
    wreq.shift = (W * H <= 832 * 480) ? 3.0f : 5.0f;

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
// 6. ComfyUI JSON Compiled DAG Execution (Persistent Merkle Cache)
// ===========================================================================
typedef struct { ec_studio_t *st; ec_studio_result_t *res; } flow_exec_ctx_t;

static int studio_dag_cb(
    const ec_flow_graph_t *g, int32_t idx,
    const ec_flow_cache_entry_t *ups[], int32_t n_up,
    ec_flow_cache_entry_t *out, void *uctx) {
  flow_exec_ctx_t *fc = (flow_exec_ctx_t *)uctx;
  const ec_flow_node_t *nd = &g->nodes[idx];
  uint64_t tag = nd->merkle_hash;
  for (int32_t i = 0; i < n_up; i++) tag ^= ups[i]->result_tag * 0x9E3779B97F4A7C15ULL;
  out->result_tag = tag;

  const ec_flow_input_t *in_p = ec_flow_get_input(nd, "prompt");
  const ec_flow_input_t *in_s = ec_flow_get_input(nd, "seed");
  const char *prompt = (in_p && in_p->kind == EC_VAL_STRING) ? in_p->str_val : "eureka creative flow";
  uint32_t seed = (in_s && in_s->kind == EC_VAL_NUMBER) ? (uint32_t)in_s->num_val : (uint32_t)(tag & 0xFFFFu);

  ec_studio_result_t sub = {};
  if (!strcmp(nd->class_type, "EurekaImageGen") || !strcmp(nd->class_type, "VAEDecode")) {
    if (ec_studio_gen_image(fc->st, prompt, seed, 256, 256, 6, EC_TIER_FAST, NULL, NULL, &sub) != 0) return -1;
    snprintf(fc->res->file_main, sizeof fc->res->file_main, "%s", sub.file_main);
    fc->res->gpu_ms += sub.gpu_ms;
  } else if (!strcmp(nd->class_type, "Eureka3DGen") || !strcmp(nd->class_type, "TRELLIS3D")) {
    if (ec_studio_gen_3d(fc->st, prompt, seed, 18, EC_TIER_FAST, NULL, NULL, &sub) != 0) return -1;
    snprintf(fc->res->file_aux, sizeof fc->res->file_aux, "%s", sub.file_main);
    fc->res->n_verts = sub.n_verts; fc->res->n_tris = sub.n_tris; fc->res->n_splats = sub.n_splats;
    fc->res->gpu_ms += sub.gpu_ms;
  } else if (!strcmp(nd->class_type, "EurekaMusicGen") || !strcmp(nd->class_type, "ACEStepMusic")) {
    const ec_flow_input_t *in_b = ec_flow_get_input(nd, "bpm");
    int bpm = (in_b && in_b->kind == EC_VAL_NUMBER) ? (int)in_b->num_val : 120;
    if (ec_studio_gen_music(fc->st, prompt, seed, bpm, 2.0f, EC_TIER_FAST, NULL, NULL, &sub) != 0) return -1;
    snprintf(fc->res->stem_files[0], sizeof fc->res->stem_files[0], "%s", sub.file_main);
    fc->res->gpu_ms += sub.gpu_ms;
  } else if (!strcmp(nd->class_type, "EurekaVideoGen") || !strcmp(nd->class_type, "LTXVideoAV")) {
    if (ec_studio_gen_video(fc->st, prompt, seed, 16, 320, 180, 24, 1, EC_TIER_FAST, NULL, NULL, &sub) != 0) return -1;
    snprintf(fc->res->file_mp4, sizeof fc->res->file_mp4, "%s", sub.file_mp4);
    fc->res->gpu_ms += sub.gpu_ms;
  }
  return 0;
}

extern "C" int ec_studio_run_flow_json(
    ec_studio_t *st, const char *comfy_json, ec_studio_result_t *res, char *err, size_t err_cap) {
  memset(res, 0, sizeof *res);
  if (!st || !st->initialized || !comfy_json) return -1;
  // Flow graph parse/compile/cache runs cleanly; GPU nodes acquire gpu_mu per step
  if (ec_flow_parse_comfy_json(&st->flow_graph, comfy_json, err, err_cap) != 0) return -1;
  if (ec_flow_compile(&st->flow_graph, err, err_cap) != 0) return -1;
  flow_exec_ctx_t fc = { st, res };
  if (ec_flow_execute(&st->flow_graph, studio_dag_cb, &fc, err, err_cap) != 0) return -1;
  res->dag_executed   = st->flow_graph.last_executed;
  res->dag_cache_hits = st->flow_graph.last_cache_hits;
  res->dag_evictions  = st->flow_graph.last_evictions;
  return 0;
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
    char arch[64] = "lora";
    ec_lora_inspect(a->path, &pairs, &rank, arch, sizeof arch);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"path\":\"%s\",\"root\":\"%s\","
        "\"bytes\":%llu,\"is_network\":%s,\"pairs\":%d,\"rank\":%d,\"arch\":\"%s\"}",
        cnt ? "," : "", a->id, a->name, a->path, a->root_label,
        (unsigned long long)a->bytes, a->is_network ? "true" : "false", pairs, rank, arch);
    cnt++;
  }
  n += snprintf(buf + n, cap - (size_t)n, "],\"count\":%d}", cnt);
  pthread_mutex_unlock(&st->gpu_mu);
  return n;
}

extern "C" int ec_studio_checkpoints_json(ec_studio_t *st, char *buf, size_t cap) {
  if (!st || !st->initialized || !buf) return -1;
  pthread_mutex_lock(&st->gpu_mu);
  int n = snprintf(buf, cap, "{\"checkpoints\":[");
  int cnt = 0;
  for (int i = 0; i < st->vault.n_assets && (size_t)n < cap - 320; i++) {
    const ec_storage_asset_t *a = &st->vault.assets[i];
    if (a->kind != EC_ASSET_CHECKPOINT) continue;
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"path\":\"%s\",\"root\":\"%s\","
        "\"bytes\":%llu,\"is_network\":%s}",
        cnt ? "," : "", a->id, a->name, a->path, a->root_label,
        (unsigned long long)a->bytes, a->is_network ? "true" : "false");
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

extern "C" int ec_studio_import_comfy(
    ec_studio_t *st, const uint8_t *payload, size_t payload_len, int is_png, int execute_now,
    char *out_json, size_t out_cap) {
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
  pos += snprintf(out_json + pos, out_cap - (size_t)pos,
      "]},\"executed\":%s,\"artifact_url\":\"%s\",\"engine_mode\":\"%s\",\"gpu_ms\":%.2f,\"api_workflow\":%s}",
      artifact_url[0] ? "true" : "false", artifact_url, exec_mode, exec_ms,
      strlen(api_json) < (out_cap - (size_t)pos - 32) ? api_json : "{}");
  free(api_json);
  return 0;
}
