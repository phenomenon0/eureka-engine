// ec_recipes.h — Photoreal Prompt Libraries, Recipes & Universal ComfyUI Importer
// Zero-dependency C11: curated photoreal recipes, user-saved recipes, prompt fragments,
// PNG tEXt/iTXt ComfyUI metadata extraction, and LiteGraph UI JSON -> API DAG conversion.

#pragma once

#include "ec_storage.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EC_MAX_RECIPES 64

typedef struct {
  char id[64];
  char name[96];
  char category[48];
  char modality[24];      // "image", "3d", "video", "music"
  char prompt[768];
  char camera[128];
  char checkpoint[128];
  char lora[128];
  float lora_strength;
  int  width;
  int  height;
  int  steps;
  uint32_t seed;
  int  builtin;
} ec_recipe_t;

typedef struct {
  char        store_path[512];
  ec_recipe_t items[EC_MAX_RECIPES];
  int         count;
} ec_recipe_lib_t;

void ec_recipes_init(ec_recipe_lib_t *lib, const char *user_store_json_path);
int  ec_recipes_save_user(ec_recipe_lib_t *lib, const ec_recipe_t *r);
int  ec_recipes_delete_user(ec_recipe_lib_t *lib, const char *id);
int  ec_recipes_to_json(const ec_recipe_lib_t *lib, char *buf, size_t cap);
int  ec_prompt_library_to_json(char *buf, size_t cap);

// Extract embedded ComfyUI "prompt" or "workflow" JSON from a raw PNG byte buffer (tEXt / iTXt).
// Returns bytes written to out_json, or <=0 if no ComfyUI chunk found.
int ec_png_extract_comfy_json(const uint8_t *png_bytes, size_t png_len, char *out_json, size_t out_cap);

typedef struct {
  int  ok;
  char source_format[32];    // "comfy_api" or "comfy_litegraph_ui"
  char modality[24];         // "image", "video", "music", "3d"
  int  node_count;
  int  subgraph_count;
  char extracted_prompt[768];
  char extracted_neg_prompt[768];
  char extracted_lyrics[1024];
  char extracted_checkpoint[192];
  char resolved_checkpoint[512];
  char extracted_loras[4][192];
  char resolved_loras[4][512];
  float extracted_lora_strengths[4];
  int  n_loras;
  int  width, height, steps, frames, bpm;
  float duration_sec;
  int  needs_image_input;
  char input_image_hint[192];
  uint32_t seed;
  char error[256];
} ec_comfy_import_info_t;

// Normalize EITHER ComfyUI API JSON OR ComfyUI LiteGraph UI JSON into canonical executable API JSON
// and resolve referenced checkpoints/LoRAs against the Multi-Drive Storage Vault!
int ec_comfy_normalize_and_resolve(
    const char *raw_json,
    const ec_storage_vault_t *vault,
    char *out_api_json, size_t out_api_cap,
    ec_comfy_import_info_t *info);

#ifdef __cplusplus
}
#endif
