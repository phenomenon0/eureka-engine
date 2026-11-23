// ec_storage.h — Multi-Drive & Network-Drive Creative Model Vault Manager
// Supports local NVMe, secondary HDDs (/mnt/storage), and mounted network drives
// (NFS, SMB/CIFS, SSHFS, GVFS, Tailscale mounts) with live statvfs space telemetry
// and automatic categorization of Checkpoints, LoRAs, VAEs, Text Encoders & Tokenizers.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_STORAGE_MAX_ROOTS 16
#define EC_STORAGE_MAX_ASSETS 256

typedef enum {
  EC_ASSET_CHECKPOINT = 0, // Diffusion DiT / UNet / Finetune checkpoint
  EC_ASSET_LORA       = 1, // LoRA / LyCORIS / DoRA adapter
  EC_ASSET_VAE        = 2, // 2D/3D/Audio VAE
  EC_ASSET_TEXT_ENC   = 3, // Qwen3 / UMT5 / CLIP text encoder
  EC_ASSET_TOKENIZER  = 4, // tokenizer.json
  EC_ASSET_OTHER      = 5
} ec_asset_kind_t;

typedef struct {
  char     path[512];
  char     label[64];
  int      is_network;
  int      writable;
  int      exists;
  double   total_gb;
  double   free_gb;
  double   used_pct;
  char     fs_type[32];
} ec_storage_root_t;

typedef struct {
  char            id[128];
  char            name[128];
  char            path[512];
  char            root_label[64];
  ec_asset_kind_t kind;
  uint64_t        bytes;
  int             is_network;
} ec_storage_asset_t;

typedef struct {
  char               config_path[512];
  ec_storage_root_t  roots[EC_STORAGE_MAX_ROOTS];
  int                n_roots;
  ec_storage_asset_t assets[EC_STORAGE_MAX_ASSETS];
  int                n_assets;
} ec_storage_vault_t;

void ec_storage_init(ec_storage_vault_t *v, const char *primary_models_dir, const char *config_path);
int  ec_storage_add_root(ec_storage_vault_t *v, const char *path, const char *label, int persist);
int  ec_storage_remove_root(ec_storage_vault_t *v, const char *path);
int  ec_storage_scan(ec_storage_vault_t *v);

// Pick the writable root with the most free space (preferring local if >= min_free_gb, else largest network/storage root)
const ec_storage_root_t *ec_storage_best_root(const ec_storage_vault_t *v, double min_free_gb);

// Resolve an asset by filename, id, substring, or full path across ALL mounted local + network roots
const ec_storage_asset_t *ec_storage_find(const ec_storage_vault_t *v, ec_asset_kind_t kind, const char *query);

const char *ec_asset_kind_name(ec_asset_kind_t k);

// Format vault status + roots + assets as JSON
int ec_storage_to_json(const ec_storage_vault_t *v, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif
