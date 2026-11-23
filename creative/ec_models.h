// ec_models.h — 100% Native 2026 HuggingFace Safetensors Multi-Architecture Model Registry & Loaders
// Auto-detects and loads real BF16 / FP8_E4M3 / F16 / F32 Safetensors checkpoints across all 5 flagship families:
//   1. Z-Image-Turbo (6B S3-DiT + Qwen3 text encoder) & FLUX.2 Klein (4B MMDiT)
//   2. ACE-Step 1.5 (MIT Music 1D DiT + Cross-Attn + 25Hz DCAE)
//   3. TRELLIS.2 (4B MIT Sparse Structure + O-Voxel SLAT 3D DiT)
//   4. Wan 2.2 (Timestep-Routed High/Low-Noise MoE Video DiT + Causal 3D VAE)
//   5. LTX-2.5 (Asymmetric Dual-Stream Audio+Video DiT + Bidirectional A/V Cross-Attn + Modality-CFG)

#pragma once

#include "ec_safetensors.h"
#include "ec_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_REG_MAX_MODELS 64

typedef enum {
  EC_ARCH_UNKNOWN       = 0,
  EC_ARCH_ZIMAGE_S3DIT  = 1,  // Z-Image-Turbo (S3-DiT + Qwen3)
  EC_ARCH_FLUX2_MMDIT   = 2,  // FLUX.2 Klein / Dev (Dual+Single Stream MMDiT)
  EC_ARCH_ACESTEP_15    = 3,  // ACE-Step 1.5 (1D DiT + Cross-Attn + DCAE)
  EC_ARCH_TRELLIS2_3D   = 4,  // TRELLIS.2 (Sparse O-Voxel 3D Flow DiT)
  EC_ARCH_WAN22_MOE     = 5,  // Wan 2.2 (High/Low-Noise Timestep MoE Video DiT)
  EC_ARCH_LTX25_AV      = 6,  // LTX-2.5 (Asymmetric Dual-Stream Audio+Video DiT)
  EC_ARCH_SELFFORCING   = 7,  // Self-Forcing (Block-Causal KV-Ring Realtime Video)
} ec_arch_t;

const char *ec_arch_name(ec_arch_t a);
ec_modality_t ec_arch_modality(ec_arch_t a);

typedef struct {
  char          id[64];
  char          path[384];
  ec_arch_t     arch;
  ec_modality_t modality;
  ec_dtype_t    weight_dtype;   // detected primary dtype (BF16 / FP8_E4M3 / F16 / F32)
  uint64_t      file_bytes;
  uint32_t      n_tensors;
  int32_t       n_layers;
  int32_t       dim;
  int32_t       ffn_dim;
  int32_t       n_heads;
  int32_t       head_dim;
  int32_t       audio_dim;      // LTX-2.5 secondary audio stream dim (0 for non-A/V)
  int32_t       has_moe_split;  // Wan 2.2 high_noise + low_noise experts detected
  int           loaded;
} ec_model_info_t;

// Loaded live GPU model handle (BF16/FP8 converted on-GPU into resident F16 TensorCore weights!)
typedef struct ec_loaded_model ec_loaded_model_t;

typedef struct {
  char              models_dir[256];
  ec_model_info_t   models[EC_REG_MAX_MODELS];
  int               n_models;
  ec_loaded_model_t *active[5]; // one resident active loaded model per modality slot
} ec_model_registry_t;

void ec_registry_init(ec_model_registry_t *reg, const char *models_dir);
void ec_registry_free(ec_model_registry_t *reg);

// Inspect a single .safetensors file & detect its exact 2026 HF architecture + hyperparams
int ec_model_probe_safetensors(const char *path, ec_model_info_t *out, char *err, size_t err_cap);

// Scan models_dir for all .safetensors checkpoints (non-destructive; keeps loaded state)
int ec_registry_scan(ec_model_registry_t *reg);

// Hot-load a discovered model by id (uploads & converts BF16/FP8->F16 directly on GPU!)
int ec_registry_load(ec_model_registry_t *reg, const char *id, char *err, size_t err_cap);
int ec_registry_unload(ec_model_registry_t *reg, const char *id);
ec_loaded_model_t *ec_registry_find_loaded(ec_model_registry_t *reg, const char *id);

// Execute a real forward verification step on a loaded architecture model on GPU!
// For Wan 2.2: routes through high_noise (t >= 0.5) vs low_noise (t < 0.5) expert!
// For LTX-2.5: runs true asymmetric Video + Audio dual streams with bidirectional A<->V Cross-Attention!
int ec_model_forward_probe(
    ec_loaded_model_t *m, float t_val, ec_tier_t tier,
    double *out_gpu_ms, float *out_norm_val, char *out_route, size_t route_cap);

// Serialize registry as JSON into caller buffer
int ec_registry_to_json(const ec_model_registry_t *reg, char *buf, size_t cap);

// Writes 5 miniature HuggingFace-schema-accurate BF16 & FP8 Safetensors test checkpoints
// (Z-Image-Turbo, ACE-Step 1.5, TRELLIS.2, Wan 2.2 MoE, LTX-2.5 Dual-Stream A/V) into out_dir.
int ec_models_write_hf_fixtures(const char *out_dir);

#ifdef __cplusplus
}
#endif
