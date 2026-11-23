// ec_recipes.c — Photoreal Prompt Libraries, Recipes & Universal ComfyUI Importer
#define _POSIX_C_SOURCE 200809L
#include "ec_recipes.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static void json_esc(const char *src, char *dst, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; src && src[i] && j + 2 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (c == '"' || c == '\\') { dst[j++] = '\\'; dst[j++] = (char)c; }
    else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
    else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
    else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
    else if (c >= 0x20) dst[j++] = (char)c;
  }
  dst[j] = '\0';
}

static const ec_recipe_t BUILTIN_RECIPES[] = {
  {
    "editorial-portrait-85mm", "Editorial Portrait — 85mm f/1.4", "Portrait", "image",
    "Candid editorial portrait of a weathered ceramic artist in a sunlit Kyoto workshop, natural window sidelight catching dust motes and fine skin texture, shot on Hasselblad X2D 100C, 85mm f/1.4, shallow depth of field, Kodak Portra 400 color science, authentic candid expression, photorealistic",
    "Hasselblad X2D · 85mm f/1.4 · Portra 400", "flux2_klein_4b_bf16", "", 0.8f, 768, 1024, 4, 1042, 1
  },
  {
    "tokyo-rain-automotive", "Automotive Night Rain — Tokyo 35mm", "Automotive", "image",
    "A cinematic 35mm photograph of a crimson Ferrari F40 parked on a rain-slicked neon Tokyo alley at midnight, crisp wet asphalt reflections of kanji signs, mist rising from warm engine vents, anamorphic lens flare, shot on Leica SL3 35mm f/1.4, Cinestill 800T grain, photorealistic, ultra detailed",
    "Leica SL3 · 35mm f/1.4 · Cinestill 800T", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 42, 1
  },
  {
    "architectural-brutalist", "Architectural Monograph — 24mm Tilt-Shift", "Architecture", "image",
    "Architectural photography of a serene concrete and cedar pavilion cantilevered over a misty Norwegian fjord at dawn, crisp rectilinear verticals, calm mirror water reflections, warm interior tungsten glow contrasting cool blue twilight, shot on Phase One IQ4 24mm tilt-shift f/8, magazine editorial",
    "Phase One IQ4 · 24mm Tilt-Shift f/8", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 2026, 1
  },
  {
    "luxury-watch-macro", "Luxury Product Macro — 100mm Studio", "Product", "image",
    "Commercial macro studio photograph of a brushed titanium mechanical chronograph watch resting on wet dark slate, dramatic sculpted rim lighting from overhead strip softbox, sapphire crystal anti-reflective glint, microscopic gear detail, shot on Canon R5 100mm macro f/5.6, focus stacked, photorealistic",
    "Canon R5 · 100mm Macro f/5.6 · Strip Softbox", "flux2_klein_4b_bf16", "", 0.8f, 768, 768, 4, 777, 1
  },
  {
    "golden-hour-wildlife", "Golden Hour Wildlife — 400mm Telephoto", "Nature", "image",
    "National Geographic photograph of a golden retriever sitting calmly on a weathered wooden lake pier at golden hour, backlit fur glowing in warm low sun, glassy alpine mountain reflections in soft bokeh background, shot on Sony A1 400mm f/2.8, crisp eye catchlight, natural photography",
    "Sony A1 · 400mm f/2.8 · Natural Backlight", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 888, 1
  },
  {
    "leica-street-doc", "Street Documentary — Rainy Paris 50mm", "Street", "image",
    "Timeless street documentary photograph of a jazz saxophonist under a warm glowing Paris cafe awning during a light evening rain, wet cobblestones, subtle motion blur of passing umbrellas, Leica M11 50mm Summilux f/1.4, authentic film grain, candid human moment",
    "Leica M11 · 50mm Summilux f/1.4", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 314, 1
  },
  {
    "nordic-interior-sun", "Interior Design — Nordic Morning Light", "Interior", "image",
    "Editorial interior photograph of a minimalist Copenhagen living room bathed in crisp morning sunlight, oak herringbone floor, travertine coffee table with stoneware espresso cup, linen sofa, soft architectural shadows across limewash wall, shot on Fujifilm GFX 100 II 32mm f/5.6",
    "Fuji GFX 100 II · 32mm f/5.6 · Daylight", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 512, 1
  },
  {
    "culinary-editorial", "Culinary Editorial — Overhead Artisanal", "Food", "image",
    "Overhead editorial food photograph of freshly baked sourdough boule sliced on an aged walnut board beside cultured butter, flaky sea salt, and wild figs, raking natural daylight from side window, subtle flour dust suspended in air, shot on Nikon Z8 50mm f/2.8, Bon Appetit style",
    "Nikon Z8 · 50mm f/2.8 · Raking Daylight", "flux2_klein_4b_bf16", "", 0.8f, 768, 1024, 4, 909, 1
  },
  {
    "scifi-cinema-still", "Cinema Sci-Fi Film Still — 65mm IMAX", "Cinema", "image",
    "Anamorphic 65mm film still of a lone astronaut walking across an obsidian volcanic plain under two pale crescent moons, subtle dust storm catching amber helmet visor lights, Denis Villeneuve cinematography, atmospheric scale, muted naturalistic palette, shot on ARRI Alexa 65",
    "ARRI Alexa 65 · Prime DNA 40mm · Naturalistic", "flux2_klein_4b_bf16", "", 0.8f, 1024, 576, 4, 2049, 1
  },
  {
    "hero-3d-prop", "3D Hero PBR Prop — Studio Turntable", "3D Asset", "3d",
    "Isometric studio turntable asset of a hand-crafted brass and matte ceramic espresso machine with clean watertight silhouette, neutral softbox lighting, subtle bevel wear, ready for TRELLIS.2 O-Voxel GLB 2.0 and 3DGS splat export",
    "Neutral 3-Point Turntable · PBR Clean", "trellis2_ss_flow_1.3b_bf16", "", 0.0f, 512, 512, 12, 77, 1
  },
  {
    "cinema-video-dolly", "Cinema Video — 35mm Orbital Dolly", "Video", "video",
    "Slow cinematic 35mm orbital camera dolly around a rain-beaded vintage bronze astrolabe glowing softly by candlelight inside an observatory, smooth temporal continuity, shallow focus pull, 24fps NVENC H.264 + stereo score",
    "24fps Orbital Dolly · Wan 2.2 TI2V-5B", "wan2.2_ti2v_5b_fp16", "", 0.0f, 512, 288, 12, 303, 1
  },
  {
    "cinema-synth-stems", "Full-Song 4-Stem Score — Analog Cinema", "Music", "music",
    "Warm analog Moog bassline, crisp acoustic brushed drums, lush Juno-106 pad harmonies, and expressive Rhodes electric piano lead arpeggio at 124 BPM in D minor, studio mastered 44.1kHz stereo with isolated stems",
    "124 BPM · D Minor · 4 Stems (ACE-Step 1.5)", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 101, 1
  }
};

static void load_user_recipes(ec_recipe_lib_t *lib) {
  if (!lib->store_path[0]) return;
  FILE *fp = fopen(lib->store_path, "rb");
  if (!fp) return;
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (sz <= 0 || sz > 512 * 1024) { fclose(fp); return; }
  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf || fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return; }
  buf[sz] = '\0';
  fclose(fp);

  const char *p = buf;
  while ((p = strchr(p, '{')) != NULL && lib->count < EC_MAX_RECIPES) {
    const char *end = strchr(p, '}');
    if (!end) break;
    ec_recipe_t r;
    memset(&r, 0, sizeof r);
    r.width = 512; r.height = 512; r.steps = 4; r.lora_strength = 0.8f; r.builtin = 0;

    #define GRAB_S(key, dst) do { \
      const char *k = strstr(p, "\"" key "\""); \
      if (k && k < end) { \
        k = strchr(k, ':'); if (k && k < end) { \
          while (k < end && *k != '"') k++; \
          if (k < end && *k == '"') { \
            k++; size_t j = 0; \
            while (k < end && *k != '"' && j + 1 < sizeof(dst)) { \
              if (*k == '\\' && k + 1 < end) k++; \
              (dst)[j++] = *k++; \
            } \
            (dst)[j] = '\0'; \
          } \
        } \
      } \
    } while (0)
    #define GRAB_I(key, dst) do { \
      const char *k = strstr(p, "\"" key "\""); \
      if (k && k < end) { k = strchr(k, ':'); if (k && k < end) (dst) = (int)strtol(k + 1, NULL, 10); } \
    } while (0)

    GRAB_S("id", r.id);
    GRAB_S("name", r.name);
    GRAB_S("category", r.category);
    GRAB_S("modality", r.modality);
    GRAB_S("prompt", r.prompt);
    GRAB_S("camera", r.camera);
    GRAB_S("checkpoint", r.checkpoint);
    GRAB_S("lora", r.lora);
    GRAB_I("width", r.width);
    GRAB_I("height", r.height);
    GRAB_I("steps", r.steps);
    int sd = 42; GRAB_I("seed", sd); r.seed = (uint32_t)sd;
    #undef GRAB_S
    #undef GRAB_I
    if (r.id[0] && r.name[0]) lib->items[lib->count++] = r;
    p = end + 1;
  }
  free(buf);
}

static void persist_user_recipes(const ec_recipe_lib_t *lib) {
  if (!lib->store_path[0]) return;
  FILE *fp = fopen(lib->store_path, "wb");
  if (!fp) return;
  fprintf(fp, "[\n");
  int first = 1;
  for (int i = 0; i < lib->count; i++) {
    const ec_recipe_t *r = &lib->items[i];
    if (r->builtin) continue;
    char ep[1536], en[192], ec[256], eck[256], el[256];
    json_esc(r->prompt, ep, sizeof ep);
    json_esc(r->name, en, sizeof en);
    json_esc(r->camera, ec, sizeof ec);
    json_esc(r->checkpoint, eck, sizeof eck);
    json_esc(r->lora, el, sizeof el);
    fprintf(fp, "%s  {\"id\":\"%s\",\"name\":\"%s\",\"category\":\"%s\",\"modality\":\"%s\","
                "\"prompt\":\"%s\",\"camera\":\"%s\",\"checkpoint\":\"%s\",\"lora\":\"%s\","
                "\"lora_strength\":%.2f,\"width\":%d,\"height\":%d,\"steps\":%d,\"seed\":%u}\n",
            first ? "" : ",", r->id, en, r->category, r->modality, ep, ec, eck, el,
            r->lora_strength, r->width, r->height, r->steps, r->seed);
    first = 0;
  }
  fprintf(fp, "]\n");
  fclose(fp);
}

void ec_recipes_init(ec_recipe_lib_t *lib, const char *user_store_json_path) {
  memset(lib, 0, sizeof *lib);
  if (user_store_json_path && user_store_json_path[0]) {
    snprintf(lib->store_path, sizeof lib->store_path, "%s", user_store_json_path);
  }
  size_t nb = sizeof BUILTIN_RECIPES / sizeof BUILTIN_RECIPES[0];
  for (size_t i = 0; i < nb && lib->count < EC_MAX_RECIPES; i++) {
    lib->items[lib->count++] = BUILTIN_RECIPES[i];
  }
  load_user_recipes(lib);
}

int ec_recipes_save_user(ec_recipe_lib_t *lib, const ec_recipe_t *r) {
  if (!lib || !r || !r->id[0] || !r->name[0]) return -1;
  for (int i = 0; i < lib->count; i++) {
    if (strcmp(lib->items[i].id, r->id) == 0) {
      if (lib->items[i].builtin) return -2;
      lib->items[i] = *r;
      lib->items[i].builtin = 0;
      persist_user_recipes(lib);
      return 0;
    }
  }
  if (lib->count >= EC_MAX_RECIPES) return -1;
  lib->items[lib->count] = *r;
  lib->items[lib->count].builtin = 0;
  lib->count++;
  persist_user_recipes(lib);
  return 0;
}

int ec_recipes_delete_user(ec_recipe_lib_t *lib, const char *id) {
  if (!lib || !id) return -1;
  for (int i = 0; i < lib->count; i++) {
    if (strcmp(lib->items[i].id, id) == 0 && !lib->items[i].builtin) {
      for (int j = i; j + 1 < lib->count; j++) lib->items[j] = lib->items[j + 1];
      lib->count--;
      persist_user_recipes(lib);
      return 0;
    }
  }
  return -1;
}

int ec_recipes_to_json(const ec_recipe_lib_t *lib, char *buf, size_t cap) {
  int n = snprintf(buf, cap, "{\"count\":%d,\"recipes\":[", lib->count);
  for (int i = 0; i < lib->count && (size_t)n < cap - 1024; i++) {
    const ec_recipe_t *r = &lib->items[i];
    char ep[1536], en[192], ec[256], eck[256], el[256];
    json_esc(r->prompt, ep, sizeof ep);
    json_esc(r->name, en, sizeof en);
    json_esc(r->camera, ec, sizeof ec);
    json_esc(r->checkpoint, eck, sizeof eck);
    json_esc(r->lora, el, sizeof el);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"category\":\"%s\",\"modality\":\"%s\","
        "\"prompt\":\"%s\",\"camera\":\"%s\",\"checkpoint\":\"%s\",\"lora\":\"%s\","
        "\"lora_strength\":%.2f,\"width\":%d,\"height\":%d,\"steps\":%d,\"seed\":%u,\"builtin\":%s}",
        i ? "," : "", r->id, en, r->category, r->modality, ep, ec, eck, el,
        r->lora_strength, r->width, r->height, r->steps, r->seed, r->builtin ? "true" : "false");
  }
  n += snprintf(buf + n, cap - (size_t)n, "]}");
  return n;
}

int ec_prompt_library_to_json(char *buf, size_t cap) {
  return snprintf(buf, cap,
    "{"
    "\"cameras\":[\"Hasselblad X2D 100C\",\"Leica M11 Summilux\",\"ARRI Alexa 65\",\"Phase One IQ4\",\"Sony A1\",\"Fujifilm GFX 100 II\",\"Canon EOS R5\"],"
    "\"lenses\":[\"24mm tilt-shift f/8\",\"35mm anamorphic f/1.4\",\"50mm Summilux f/1.4\",\"85mm portrait f/1.4\",\"100mm macro f/5.6\",\"400mm telephoto f/2.8\"],"
    "\"lighting\":[\"natural window sidelight\",\"golden hour backlight\",\"rain-slicked neon night reflections\",\"overhead studio strip softbox\",\"overcast diffused Nordic daylight\",\"chiaroscuro tungsten rim light\"],"
    "\"film_stocks\":[\"Kodak Portra 400 color science\",\"Cinestill 800T halation\",\"Ilford HP5 monochrome grain\",\"Fujifilm Velvia 50 vivid landscape\",\"ARRI LogC naturalistic film grade\"],"
    "\"compositions\":[\"shallow depth of field bokeh\",\"rule of thirds editorial framing\",\"rectilinear architectural verticals\",\"overhead flat-lay tabletop\",\"sweeping anamorphic wide shot\",\"intimate macro focus stack\"],"
    "\"music_palettes\":[\"warm analog Moog bass + brushed drums 124 BPM\",\"lush strings + solo cello cinematic 84 BPM\",\"lo-fi Rhodes chords + dusty vinyl breaks 92 BPM\",\"driving Berlin techno arpeggio + sub 132 BPM\"],"
    "\"materials_3d\":[\"brushed titanium + sapphire glass PBR\",\"hand-thrown matte stoneware ceramic\",\"weathered patinated bronze + walnut wood\",\"translucent frosted acrylic + anodized aluminum\"]"
    "}");
}

// ===========================================================================
// Pure-C11 PNG Embedded ComfyUI Metadata Extractor (tEXt / iTXt, zero libpng!)
// ===========================================================================
static uint32_t rd_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

int ec_png_extract_comfy_json(const uint8_t *png, size_t len, char *out_json, size_t out_cap) {
  static const uint8_t SIG[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
  if (!png || len < 33 || memcmp(png, SIG, 8) != 0 || !out_json || out_cap < 32) return -1;

  size_t off = 8;
  size_t best_len = 0;
  int found_prompt = 0;

  while (off + 12 <= len) {
    uint32_t clen = rd_be32(png + off);
    const char *ctype = (const char *)(png + off + 4);
    if (off + 12 + clen > len) break;
    const uint8_t *data = png + off + 8;

    if (memcmp(ctype, "tEXt", 4) == 0 && clen > 7) {
      const uint8_t *nul = (const uint8_t *)memchr(data, 0, clen);
      if (nul) {
        size_t klen = (size_t)(nul - data);
        int is_prompt = (klen == 6 && memcmp(data, "prompt", 6) == 0);
        int is_wf     = (klen == 8 && memcmp(data, "workflow", 8) == 0);
        if (is_prompt || (is_wf && !found_prompt)) {
          size_t vlen = clen - klen - 1;
          if (vlen > 2 && vlen < out_cap) {
            memcpy(out_json, nul + 1, vlen);
            out_json[vlen] = '\0';
            best_len = vlen;
            if (is_prompt) found_prompt = 1;
          }
        }
      }
    } else if (memcmp(ctype, "iTXt", 4) == 0 && clen > 10) {
      const uint8_t *nul = (const uint8_t *)memchr(data, 0, clen);
      if (nul && (size_t)(nul - data) + 5 < clen) {
        size_t klen = (size_t)(nul - data);
        uint8_t comp_flag = nul[1];
        if (comp_flag == 0) {
          int is_prompt = (klen == 6 && memcmp(data, "prompt", 6) == 0);
          int is_wf     = (klen == 8 && memcmp(data, "workflow", 8) == 0);
          if (is_prompt || (is_wf && !found_prompt)) {
            // Skip compression_flag(1), compression_method(1), language\0, translated\0
            const uint8_t *q = nul + 3;
            const uint8_t *end = data + clen;
            while (q < end && *q != 0) q++;
            if (q < end) q++;
            while (q < end && *q != 0) q++;
            if (q < end) q++;
            size_t vlen = (size_t)(end - q);
            if (vlen > 2 && vlen < out_cap) {
              memcpy(out_json, q, vlen);
              out_json[vlen] = '\0';
              best_len = vlen;
              if (is_prompt) found_prompt = 1;
            }
          }
        }
      }
    }
    off += 12 + (size_t)clen;
  }
  return (int)best_len;
}

// ===========================================================================
// Universal ComfyUI JSON Normalizer (API format OR LiteGraph UI format!)
// ===========================================================================
static void skip_json_ws(const char **pp) {
  while (**pp && (**pp == ' ' || **pp == '\n' || **pp == '\r' || **pp == '\t')) (*pp)++;
}

static int grab_json_string(const char **pp, char *dst, size_t cap) {
  skip_json_ws(pp);
  if (**pp != '"') return -1;
  (*pp)++;
  size_t j = 0;
  while (**pp && **pp != '"') {
    if (**pp == '\\' && (*pp)[1]) {
      (*pp)++;
      char c = **pp;
      if (c == 'n') c = '\n';
      else if (c == 't') c = '\t';
      if (j + 1 < cap) dst[j++] = c;
      (*pp)++;
    } else {
      if (j + 1 < cap) dst[j++] = **pp;
      (*pp)++;
    }
  }
  if (**pp != '"') return -1;
  (*pp)++;
  dst[j] = '\0';
  return 0;
}

static void skip_json_val(const char **pp) {
  skip_json_ws(pp);
  char c = **pp;
  if (c == '"') {
    char tmp[8];
    grab_json_string(pp, tmp, sizeof tmp);
  } else if (c == '{' || c == '[') {
    char open = c, close = (c == '{') ? '}' : ']';
    int depth = 1;
    (*pp)++;
    while (**pp && depth > 0) {
      if (**pp == '"') {
        (*pp)++;
        while (**pp && **pp != '"') { if (**pp == '\\' && (*pp)[1]) (*pp)++; (*pp)++; }
        if (**pp == '"') (*pp)++;
      } else {
        if (**pp == open) depth++;
        else if (**pp == close) depth--;
        (*pp)++;
      }
    }
  } else {
    while (**pp && **pp != ',' && **pp != '}' && **pp != ']' && !isspace((unsigned char)**pp)) (*pp)++;
  }
}

static int ci_has(const char *h, const char *n) {
  if (!h || !n) return 0;
  size_t nl = strlen(n);
  for (const char *p = h; *p; p++) if (strncasecmp(p, n, nl) == 0) return 1;
  return 0;
}

int ec_comfy_normalize_and_resolve(
    const char *raw_json,
    const ec_storage_vault_t *vault,
    char *out_api_json, size_t out_api_cap,
    ec_comfy_import_info_t *info) {
  memset(info, 0, sizeof *info);
  info->width = 512; info->height = 512; info->steps = 4; info->seed = 42;
  if (!raw_json || !out_api_json || out_api_cap < 64) {
    snprintf(info->error, sizeof info->error, "empty workflow json");
    return -1;
  }

  int is_litegraph = (strstr(raw_json, "\"nodes\"") != NULL && strstr(raw_json, "\"links\"") != NULL);
  snprintf(info->source_format, sizeof info->source_format, "%s",
           is_litegraph ? "comfy_litegraph_ui" : "comfy_api");

  if (!is_litegraph) {
    // Already API format: scan for prompt, checkpoint, lora, width, height, steps, seed
    snprintf(out_api_json, out_api_cap, "%s", raw_json);
    const char *p = raw_json;
    while ((p = strstr(p, "\"class_type\"")) != NULL) {
      info->node_count++;
      const char *colon = strchr(p, ':');
      if (!colon) break;
      const char *q = colon + 1;
      char ct[128] = {0};
      grab_json_string(&q, ct, sizeof ct);

      // Scan forward up to next class_type (or end) for inputs
      const char *next_ct = strstr(q, "\"class_type\"");
      const char *limit = next_ct ? next_ct : (raw_json + strlen(raw_json));

      if (ci_has(ct, "CLIPTextEncode") || ci_has(ct, "Prompt")) {
        const char *tp = strstr(q, "\"text\"");
        if (tp && tp < limit) {
          tp = strchr(tp, ':');
          if (tp && tp < limit) {
            const char *v = tp + 1;
            char txt[768] = {0};
            if (grab_json_string(&v, txt, sizeof txt) == 0 && strlen(txt) > strlen(info->extracted_prompt)) {
              snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", txt);
            }
          }
        }
      }
      if (ci_has(ct, "CheckpointLoader") || ci_has(ct, "UNETLoader") || ci_has(ct, "DiffusionModel")) {
        const char *kp = strstr(q, "\"ckpt_name\"");
        if (!kp || kp >= limit) kp = strstr(q, "\"unet_name\"");
        if (kp && kp < limit) {
          kp = strchr(kp, ':');
          if (kp && kp < limit) {
            const char *v = kp + 1;
            grab_json_string(&v, info->extracted_checkpoint, sizeof info->extracted_checkpoint);
          }
        }
      }
      if (ci_has(ct, "LoraLoader") && info->n_loras < 4) {
        const char *lp = strstr(q, "\"lora_name\"");
        if (lp && lp < limit) {
          lp = strchr(lp, ':');
          if (lp && lp < limit) {
            const char *v = lp + 1;
            if (grab_json_string(&v, info->extracted_loras[info->n_loras], sizeof info->extracted_loras[0]) == 0 &&
                info->extracted_loras[info->n_loras][0]) {
              float str_v = 0.85f;
              const char *sp = strstr(q, "\"strength_model\"");
              if (sp && sp < limit) { sp = strchr(sp, ':'); if (sp) str_v = strtof(sp + 1, NULL); }
              info->extracted_lora_strengths[info->n_loras] = str_v;
              info->n_loras++;
            }
          }
        }
      }
      if (ci_has(ct, "Empty") && ci_has(ct, "Latent")) {
        const char *wp = strstr(q, "\"width\""), *hp = strstr(q, "\"height\"");
        if (wp && wp < limit) { wp = strchr(wp, ':'); if (wp) info->width  = (int)strtol(wp + 1, NULL, 10); }
        if (hp && hp < limit) { hp = strchr(hp, ':'); if (hp) info->height = (int)strtol(hp + 1, NULL, 10); }
      }
      if (ci_has(ct, "Sampler") || ci_has(ct, "Scheduler")) {
        const char *stp = strstr(q, "\"steps\""), *sdp = strstr(q, "\"seed\"");
        if (!sdp || sdp >= limit) sdp = strstr(q, "\"noise_seed\"");
        if (stp && stp < limit) { stp = strchr(stp, ':'); if (stp) info->steps = (int)strtol(stp + 1, NULL, 10); }
        if (sdp && sdp < limit) { sdp = strchr(sdp, ':'); if (sdp) info->seed  = (uint32_t)strtoull(sdp + 1, NULL, 10); }
      }
      p = q;
    }
  } else {
    // LiteGraph UI JSON: parse "links" [[link_id, from_id, from_slot, to_id, to_slot, ...]] & "nodes" [{id, type, widgets_values, inputs}]
    typedef struct { int id, from_node, from_slot; } lg_link_t;
    lg_link_t links[512];
    int n_links = 0;
    const char *lp = strstr(raw_json, "\"links\"");
    if (lp) {
      lp = strchr(lp, '[');
      if (lp) {
        lp++;
        while (*lp && *lp != ']' && n_links < 512) {
          skip_json_ws(&lp);
          if (*lp == ',') { lp++; continue; }
          if (*lp == '[') {
            lp++;
            int lid = (int)strtol(lp, (char **)&lp, 10);
            if (*lp == ',') lp++;
            int fn = (int)strtol(lp, (char **)&lp, 10);
            if (*lp == ',') lp++;
            int fs = (int)strtol(lp, (char **)&lp, 10);
            links[n_links].id = lid; links[n_links].from_node = fn; links[n_links].from_slot = fs;
            n_links++;
            while (*lp && *lp != ']') lp++;
            if (*lp == ']') lp++;
          } else {
            break;
          }
        }
      }
    }

    int out_n = snprintf(out_api_json, out_api_cap, "{");
    const char *np = strstr(raw_json, "\"nodes\"");
    if (np) np = strchr(np, '[');
    if (np) {
      np++;
      while (*np && *np != ']') {
        skip_json_ws(&np);
        if (*np == ',') { np++; continue; }
        if (*np != '{') break;
        const char *obj_start = np;
        skip_json_val(&np);
        const char *obj_end = np;

        int nid = -1;
        char ntype[128] = {0};
        char wstrs[6][512];
        int  n_wstrs = 0;
        double wnums[8] = {0};
        int  n_wnums = 0;
        char in_names[12][48];
        int  in_link_ids[12];
        int  n_ins = 0;

        // Parse top-level fields of node object
        const char *q = obj_start + 1;
        while (q < obj_end) {
          skip_json_ws(&q);
          if (*q == '}' || *q == '\0') break;
          if (*q == ',') { q++; continue; }
          char fk[64] = {0};
          if (grab_json_string(&q, fk, sizeof fk) != 0) break;
          skip_json_ws(&q);
          if (*q == ':') q++;
          skip_json_ws(&q);
          if (strcmp(fk, "id") == 0) {
            nid = (int)strtol(q, (char **)&q, 10);
          } else if (strcmp(fk, "type") == 0) {
            grab_json_string(&q, ntype, sizeof ntype);
          } else if (strcmp(fk, "widgets_values") == 0 && *q == '[') {
            q++;
            while (q < obj_end && *q != ']') {
              skip_json_ws(&q);
              if (*q == ',') { q++; continue; }
              if (*q == '"') {
                char tmp[512] = {0};
                grab_json_string(&q, tmp, sizeof tmp);
                if (n_wstrs < 6) { snprintf(wstrs[n_wstrs], sizeof wstrs[0], "%s", tmp); n_wstrs++; }
              } else if (*q == '-' || isdigit((unsigned char)*q)) {
                double dv = strtod(q, (char **)&q);
                if (n_wnums < 8) wnums[n_wnums++] = dv;
              } else {
                skip_json_val(&q);
              }
            }
            if (q < obj_end && *q == ']') q++;
          } else if (strcmp(fk, "inputs") == 0 && *q == '[') {
            q++;
            while (q < obj_end && *q != ']') {
              skip_json_ws(&q);
              if (*q == ',') { q++; continue; }
              if (*q == '{') {
                const char *is = q;
                skip_json_val(&q);
                const char *ie = q;
                const char *np2 = strstr(is, "\"name\"");
                const char *lk2 = strstr(is, "\"link\"");
                if (np2 && np2 < ie && lk2 && lk2 < ie && n_ins < 12) {
                  np2 = strchr(np2, ':'); lk2 = strchr(lk2, ':');
                  if (np2 && lk2) {
                    const char *nv = np2 + 1;
                    char iname[48] = {0};
                    grab_json_string(&nv, iname, sizeof iname);
                    const char *lv = lk2 + 1; skip_json_ws(&lv);
                    if (*lv == '-' || isdigit((unsigned char)*lv)) {
                      int lid = (int)strtol(lv, NULL, 10);
                      snprintf(in_names[n_ins], sizeof in_names[0], "%s", iname);
                      in_link_ids[n_ins] = lid;
                      n_ins++;
                    }
                  }
                }
              } else {
                skip_json_val(&q);
              }
            }
            if (q < obj_end && *q == ']') q++;
          } else {
            skip_json_val(&q);
          }
        }

        if (nid >= 0 && ntype[0]) {
          // Extract semantic fields for direct photoreal execution!
          if ((ci_has(ntype, "CLIPTextEncode") || ci_has(ntype, "Prompt")) && n_wstrs > 0 &&
              strlen(wstrs[0]) > strlen(info->extracted_prompt)) {
            snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", wstrs[0]);
          }
          if ((ci_has(ntype, "CheckpointLoader") || ci_has(ntype, "UNETLoader") || ci_has(ntype, "DiffusionModel")) && n_wstrs > 0) {
            snprintf(info->extracted_checkpoint, sizeof info->extracted_checkpoint, "%s", wstrs[0]);
          }
          if (ci_has(ntype, "LoraLoader") && n_wstrs > 0 && info->n_loras < 4) {
            snprintf(info->extracted_loras[info->n_loras], sizeof info->extracted_loras[0], "%s", wstrs[0]);
            info->extracted_lora_strengths[info->n_loras] = (n_wnums > 0) ? (float)wnums[0] : 0.85f;
            info->n_loras++;
          }
          if (ci_has(ntype, "Empty") && ci_has(ntype, "Latent") && n_wnums >= 2) {
            info->width  = (int)wnums[0];
            info->height = (int)wnums[1];
          }
          if (ci_has(ntype, "KSampler") && n_wnums >= 2) {
            info->seed  = (uint32_t)wnums[0];
            info->steps = (int)wnums[1];
          }

          // Emit normalized ComfyUI API node entry
          char ect[192];
          json_esc(ntype, ect, sizeof ect);
          out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
              "%s\"%d\":{\"class_type\":\"%s\",\"inputs\":{",
              info->node_count ? "," : "", nid, ect);
          int first_in = 1;
          for (int k = 0; k < n_ins; k++) {
            for (int li = 0; li < n_links; li++) {
              if (links[li].id == in_link_ids[k]) {
                out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
                    "%s\"%s\":[\"%d\",%d]", first_in ? "" : ",", in_names[k], links[li].from_node, links[li].from_slot);
                first_in = 0;
                break;
              }
            }
          }
          if (n_wstrs > 0) {
            char ew[1024];
            json_esc(wstrs[0], ew, sizeof ew);
            out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
                "%s\"value\":\"%s\"", first_in ? "" : ",", ew);
          }
          out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n, "}}");
          info->node_count++;
        }
      }
    }
    snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n, "}");
  }

  // Resolve checkpoints and LoRAs across ALL local + network storage roots!
  if (vault) {
    if (info->extracted_checkpoint[0]) {
      const ec_storage_asset_t *a = ec_storage_find(vault, EC_ASSET_CHECKPOINT, info->extracted_checkpoint);
      if (a) snprintf(info->resolved_checkpoint, sizeof info->resolved_checkpoint, "%s", a->path);
    }
    for (int i = 0; i < info->n_loras; i++) {
      const ec_storage_asset_t *a = ec_storage_find(vault, EC_ASSET_LORA, info->extracted_loras[i]);
      if (a) snprintf(info->resolved_loras[i], sizeof info->resolved_loras[0], "%s", a->path);
    }
  }
  info->ok = (info->node_count > 0) ? 1 : 0;
  if (!info->ok) snprintf(info->error, sizeof info->error, "no valid ComfyUI nodes found");
  return info->ok ? 0 : -1;
}
