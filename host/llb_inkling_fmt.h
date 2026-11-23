// host/llb_inkling_fmt.h — Native C11 container (.strata v1.1 + GGUF) & HF BPE tokenizer loader
// for Eureka's unified single-engine Inkling 2 Mini (`inkling2_loop`) runtime.
// Zero C++ / zero vendored Strata binary; uses engine's own tools/clef3/json.h + dlopen(libzstd).
#ifndef LLB_INKLING_FMT_H
#define LLB_INKLING_FMT_H

#include <ctype.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tools/clef3/json.h"

#define INK_N_PHYS       24
#define INK_N_VISITS     48
#define INK_N_EXPERTS    128
#define INK_TOP_K        8
#define INK_HIDDEN       2304
#define INK_INTER        1152
#define INK_N_HEAD       24
#define INK_N_KV         8
#define INK_HEAD_DIM     128
#define INK_D_REL        16
#define INK_REL_EXTENT   1024
#define INK_SWA_WIN      1024
#define INK_GATE_RANK    64
#define INK_MAX_TRUNK_T  1024
#define INK_MAX_DSPARK_T 128

enum InkFmt {
  INK_F_S1_25    = 0,
  INK_F_S1_5     = 1,
  INK_F_S2       = 2,
  INK_F_S3       = 3,
  INK_F_S4       = 4,
  INK_F_S5       = 5,
  INK_F_S6       = 6,
  INK_F_S8       = 8,
  INK_F_SF4      = 15,
  INK_F_MX8      = 16,
  INK_F_GGUF_RAW = 20,
  INK_F_Q8ROW    = 100,
  INK_F_F32      = 101,
  INK_F_BF16     = 102,
  INK_F_F16      = 103,
  INK_F_Q4K      = 112,
  INK_F_Q5K      = 113,
  INK_F_Q6K      = 114,
  INK_F_Q8_0     = 108,
};

#pragma pack(push, 1)
typedef struct {
  uint64_t file_offset;
  uint64_t byte_length;
  uint8_t  sha256[32];
  uint32_t section_id;
  uint32_t flags;
  uint64_t reserved;
} InkSectionEntry;

typedef struct {
  uint16_t expert_id;
  uint8_t  gu_code_id;
  uint8_t  dn_code_id;
  uint16_t cluster_id;
  uint16_t flags;
  uint64_t body_offset;
  uint32_t body_bytes;
  uint32_t dn_offset;
  uint32_t aux_offsets[2];
} InkExpertDirEntry;
#pragma pack(pop)

typedef struct {
  char     name[128];
  int      fmt;
  int64_t  rows;
  int64_t  cols;
  uint64_t off;
  uint64_t bytes;
} InkTensor;

typedef struct {
  int   n_phys;
  int   core_start;
  int   core_end;
  int   num_passes;
  int   gate_rank;
  int   block_size;
  int   hidden;
  int   inter;
  int   n_head;
  int   n_kv;
  int   head_dim;
  int   d_rel;
  int   rel_extent;
  int   swa_win;
  int   n_experts;
  int   top_k;
  int   vocab_padded;
  int   vocab_unpadded;
  int   eos_token_id;
  int   eot_token_id;
  float rms_eps;
  float qk_l2_clip;
  float rel_l2_clip;
  int   log_scale_floor;
  float log_scale_alpha;
  float situ_beta;
  float situ_lb;
  float route_scale;
  float logit_softcap;
  uint8_t is_swa[INK_N_PHYS];
  int     visit_to_phys[INK_N_VISITS];
  int     visit_to_pass[INK_N_VISITS];
  uint8_t visit_is_swa[INK_N_VISITS];
  int     visit_kv_slot[INK_N_VISITS];
  int     n_visits;
  int     n_swa_visits;
  int     n_global_visits;
} InkCfg;

// Compact HF BPE tokenizer (o200k / ByteLevel BPE)
typedef struct {
  char*    str;
  int32_t  id;
} InkTokKv;

typedef struct {
  char**    id_to_tok;
  int       vocab_cap;
  int       vocab_n;
  InkTokKv* vocab_ht;
  uint32_t  vocab_hmask;
  InkTokKv* merge_ht;
  uint32_t  merge_hmask;
  char**    specials;
  int32_t*  special_ids;
  int       n_specials;
  int       byte_level;
  int       b2u_cp[256];
  char      b2u_utf8[256][4];
  int       b2u_len[256];
  int       u2b_tbl[512];
} InkBpe;

typedef struct {
  FILE*              fp;
  int                is_strata;  // 1 = .strata v1.1, 0 = GGUF
  uint32_t           flags;
  uint16_t           profile;
  InkSectionEntry    sections[6];
  InkCfg             cfg;
  char               build[64];
  int                has_dspark;
  int                dspark_rank;
  InkTensor          trunk[INK_MAX_TRUNK_T];
  int                n_trunk;
  InkTensor          dspark[INK_MAX_DSPARK_T];
  int                n_dspark;
  InkExpertDirEntry* experts;
  uint16_t*          vram_order;
  int                n_experts_dir;
  char*              tokenizer_json;
  char*              chat_template;
  InkBpe             bpe;
} InkModelMeta;

static inline int ink_fmt_from_str(const char* s) {
  if (!s) return -1;
  if (!strcmp(s, "F32") || !strcmp(s, "float32")) return INK_F_F32;
  if (!strcmp(s, "BF16") || !strcmp(s, "bfloat16")) return INK_F_BF16;
  if (!strcmp(s, "F16") || !strcmp(s, "float16")) return INK_F_F16;
  if (!strcmp(s, "Q8_ROW")) return INK_F_Q8ROW;
  if (!strcmp(s, "S1_25")) return INK_F_S1_25;
  if (!strcmp(s, "S1_5")) return INK_F_S1_5;
  if (!strcmp(s, "S2_K")) return INK_F_S2;
  if (!strcmp(s, "S3_K")) return INK_F_S3;
  if (!strcmp(s, "S4_K")) return INK_F_S4;
  if (!strcmp(s, "S5_K")) return INK_F_S5;
  if (!strcmp(s, "S6_K")) return INK_F_S6;
  if (!strcmp(s, "S8_K")) return INK_F_S8;
  if (!strcmp(s, "SF4")) return INK_F_SF4;
  if (!strcmp(s, "MX8")) return INK_F_MX8;
  return -1;
}

static inline void ink_cfg_defaults(InkCfg* c) {
  memset(c, 0, sizeof *c);
  c->n_phys = 24; c->core_start = 6; c->core_end = 18; c->num_passes = 3;
  c->gate_rank = 64; c->block_size = 6; c->hidden = 2304; c->inter = 1152;
  c->n_head = 24; c->n_kv = 8; c->head_dim = 128; c->d_rel = 16;
  c->rel_extent = 1024; c->swa_win = 1024; c->n_experts = 128; c->top_k = 8;
  c->vocab_padded = 201024; c->vocab_unpadded = 200058;
  c->eos_token_id = 200006; c->eot_token_id = 199999;
  c->rms_eps = 1e-6f; c->qk_l2_clip = 22.627416997969522f; c->rel_l2_clip = 8.0f;
  c->log_scale_floor = 8192; c->log_scale_alpha = 0.1f;
  c->situ_beta = 4.0f; c->situ_lb = 25.0f; c->route_scale = 8.0f; c->logit_softcap = 30.0f;
  for (int i = 0; i < INK_N_PHYS; i++) c->is_swa[i] = (uint8_t)(((i + 1) % 6) != 0);
}

static inline void ink_cfg_build_visits(InkCfg* c) {
  int core_len = c->core_end - c->core_start;
  c->n_visits = c->core_start + c->num_passes * core_len + (c->n_phys - c->core_end);
  if (c->n_visits > INK_N_VISITS) c->n_visits = INK_N_VISITS;
  int swa_i = 0, glob_i = 0;
  for (int v = 0; v < c->n_visits; v++) {
    int ip = 0, p = 0;
    if (v < c->core_start) { ip = v; p = 0; }
    else if (v < c->core_start + c->num_passes * core_len) {
      ip = c->core_start + ((v - c->core_start) % core_len);
      p = (v - c->core_start) / core_len;
    } else {
      ip = c->core_end + (v - c->core_start - c->num_passes * core_len);
      p = 0;
    }
    c->visit_to_phys[v] = ip;
    c->visit_to_pass[v] = p;
    uint8_t swa = (ip < INK_N_PHYS) ? c->is_swa[ip] : 1;
    c->visit_is_swa[v] = swa;
    c->visit_kv_slot[v] = swa ? swa_i++ : glob_i++;
  }
  c->n_swa_visits = swa_i;
  c->n_global_visits = glob_i;
}

static inline const InkTensor* ink_find_trunk(const InkModelMeta* m, const char* name) {
  for (int i = 0; i < m->n_trunk; i++) {
    if (!strcmp(m->trunk[i].name, name)) return &m->trunk[i];
  }
  return NULL;
}

static inline const InkTensor* ink_find_dspark(const InkModelMeta* m, const char* name) {
  for (int i = 0; i < m->n_dspark; i++) {
    if (!strcmp(m->dspark[i].name, name)) return &m->dspark[i];
    if (!strncmp(m->dspark[i].name, "dspark.", 7) && !strcmp(m->dspark[i].name + 7, name)) return &m->dspark[i];
  }
  return NULL;
}

// Dynamic ZSTD decompression via dlopen("libzstd.so.1") so engine_gpu needs no extra static link dep
static inline int ink_zstd_decompress(const uint8_t* src, size_t src_sz, char** out, size_t* out_sz, char* err, size_t err_cap) {
  typedef unsigned long long (*fn_fcs)(const void*, size_t);
  typedef size_t (*fn_dec)(void*, size_t, const void*, size_t);
  typedef unsigned (*fn_iserr)(size_t);
  void* lib = dlopen("libzstd.so.1", RTLD_LAZY);
  if (!lib) lib = dlopen("libzstd.so", RTLD_LAZY);
  if (!lib) { snprintf(err, err_cap, "libzstd.so.1 not found (needed for .strata manifest)"); return -1; }
  fn_fcs get_fcs = (fn_fcs)dlsym(lib, "ZSTD_getFrameContentSize");
  fn_dec decomp = (fn_dec)dlsym(lib, "ZSTD_decompress");
  fn_iserr iserr = (fn_iserr)dlsym(lib, "ZSTD_isError");
  if (!get_fcs || !decomp || !iserr) { snprintf(err, err_cap, "libzstd symbols missing"); return -1; }
  unsigned long long rsz = get_fcs(src, src_sz);
  if (rsz == (unsigned long long)-2 || rsz == (unsigned long long)-1 || rsz > (256ull << 20)) {
    snprintf(err, err_cap, "invalid or streaming ZSTD manifest size");
    return -1;
  }
  char* buf = (char*)malloc((size_t)rsz + 1);
  if (!buf) { snprintf(err, err_cap, "out of memory for manifest"); return -1; }
  size_t got = decomp(buf, (size_t)rsz, src, src_sz);
  if (iserr(got)) { free(buf); snprintf(err, err_cap, "ZSTD_decompress failed"); return -1; }
  buf[got] = 0;
  *out = buf;
  *out_sz = got;
  return 0;
}

static inline int ink_parse_tensor_map(const jv* obj, InkTensor* dst, int max_n) {
  if (!obj || obj->type != J_OBJ) return 0;
  int n = 0;
  for (size_t i = 0; i < obj->n && n < max_n; i++) {
    const jv* v = obj->items[i];
    if (!v || v->type != J_OBJ) continue;
    InkTensor* t = &dst[n++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", obj->keys[i]);
    const jv* fm = json_get(v, "fmt");
    if (!fm) fm = json_get(v, "format");
    t->fmt = (fm && fm->type == J_STR) ? ink_fmt_from_str(fm->str) : INK_F_F32;
    const jv* of = json_get(v, "off");
    if (!of) of = json_get(v, "offset");
    if (of && of->type == J_NUM) t->off = (uint64_t)of->num;
    const jv* by = json_get(v, "bytes");
    if (by && by->type == J_NUM) t->bytes = (uint64_t)by->num;
    const jv* sh = json_get(v, "shape");
    if (sh && sh->type == J_ARR && sh->n > 0) {
      if (sh->n == 1) { t->rows = 1; t->cols = (int64_t)sh->items[0]->num; }
      else { t->rows = (int64_t)sh->items[0]->num; t->cols = (int64_t)sh->items[sh->n - 1]->num; }
    }
  }
  return n;
}

// ---- Clean C11 ByteLevel HF BPE tokenizer (o200k / Inkling 2 Mini) ----
static inline uint32_t ink_fnv1a(const char* s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u;
  return h;
}

static inline void ink_bpe_init_tables(InkBpe* b) {
  int seen[256] = {0};
  for (int c = 0; c < 256; c++) {
    if ((c >= 33 && c <= 126) || (c >= 161 && c <= 172) || (c >= 174 && c <= 255)) {
      b->b2u_cp[c] = c; seen[c] = 1;
    }
  }
  int n = 0;
  for (int c = 0; c < 256; c++) if (!seen[c]) b->b2u_cp[c] = 256 + n++;
  for (int i = 0; i < 512; i++) b->u2b_tbl[i] = -1;
  for (int c = 0; c < 256; c++) {
    int cp = b->b2u_cp[c];
    b->u2b_tbl[cp] = c;
    if (cp < 0x80) { b->b2u_utf8[c][0] = (char)cp; b->b2u_len[c] = 1; }
    else { b->b2u_utf8[c][0] = (char)(0xC0 | (cp >> 6)); b->b2u_utf8[c][1] = (char)(0x80 | (cp & 0x3F)); b->b2u_len[c] = 2; }
  }
}

static inline void ink_ht_put(InkTokKv* ht, uint32_t mask, char* key, int32_t id) {
  uint32_t h = ink_fnv1a(key, strlen(key)) & mask;
  while (ht[h].str != NULL) {
    if (!strcmp(ht[h].str, key)) { ht[h].id = id; return; }
    h = (h + 1) & mask;
  }
  ht[h].str = key;
  ht[h].id = id;
}

static inline int32_t ink_ht_get(const InkTokKv* ht, uint32_t mask, const char* key, size_t len) {
  if (!ht) return -1;
  uint32_t h = ink_fnv1a(key, len) & mask;
  while (ht[h].str != NULL) {
    if (strlen(ht[h].str) == len && memcmp(ht[h].str, key, len) == 0) return ht[h].id;
    h = (h + 1) & mask;
  }
  return -1;
}

static inline int ink_bpe_load_json(InkBpe* b, const char* json_str) {
  memset(b, 0, sizeof *b);
  ink_bpe_init_tables(b);
  if (!json_str || !*json_str) return -1;
  const char* jerr = NULL;
  jv* root = json_parse(json_str, &jerr);
  if (!root || root->type != J_OBJ) { json_free(root); return -1; }
  b->byte_level = 1;
  const jv* model = json_get(root, "model");
  const jv* vocab = model ? json_get(model, "vocab") : NULL;
  const jv* merges = model ? json_get(model, "merges") : NULL;
  const jv* added = json_get(root, "added_tokens");
  int max_id = 201024;
  b->vocab_cap = max_id + 1024;
  b->id_to_tok = (char**)calloc((size_t)b->vocab_cap, sizeof(char*));
  uint32_t vsz = 1u << 19;  // 524288 slots for 200K vocab
  b->vocab_ht = (InkTokKv*)calloc(vsz, sizeof(InkTokKv));
  b->vocab_hmask = vsz - 1;
  if (vocab && vocab->type == J_OBJ) {
    for (size_t i = 0; i < vocab->n; i++) {
      int32_t id = (int32_t)vocab->items[i]->num;
      if (id >= 0 && id < b->vocab_cap) {
        char* s = strdup(vocab->keys[i]);
        b->id_to_tok[id] = s;
        ink_ht_put(b->vocab_ht, b->vocab_hmask, s, id);
        if (id + 1 > b->vocab_n) b->vocab_n = id + 1;
      }
    }
  }
  if (added && added->type == J_ARR && added->n > 0) {
    b->specials = (char**)calloc(added->n, sizeof(char*));
    b->special_ids = (int32_t*)calloc(added->n, sizeof(int32_t));
    for (size_t i = 0; i < added->n; i++) {
      const jv* a = added->items[i];
      const jv* c = json_get(a, "content");
      const jv* idv = json_get(a, "id");
      if (c && c->type == J_STR && idv && idv->type == J_NUM) {
        int32_t id = (int32_t)idv->num;
        if (id >= 0 && id < b->vocab_cap) {
          char* s = strdup(c->str);
          if (!b->id_to_tok[id]) b->id_to_tok[id] = s;
          ink_ht_put(b->vocab_ht, b->vocab_hmask, s, id);
          b->specials[b->n_specials] = s;
          b->special_ids[b->n_specials] = id;
          b->n_specials++;
          if (id + 1 > b->vocab_n) b->vocab_n = id + 1;
        }
      }
    }
    // Sort specials longest-first for greedy matching
    for (int i = 0; i < b->n_specials; i++) {
      for (int j = i + 1; j < b->n_specials; j++) {
        if (strlen(b->specials[j]) > strlen(b->specials[i])) {
          char* ts = b->specials[i]; b->specials[i] = b->specials[j]; b->specials[j] = ts;
          int32_t ti = b->special_ids[i]; b->special_ids[i] = b->special_ids[j]; b->special_ids[j] = ti;
        }
      }
    }
  }
  if (merges && merges->type == J_ARR && merges->n > 0) {
    uint32_t msz = 1u << 19;
    while (msz < merges->n * 2) msz <<= 1;
    b->merge_ht = (InkTokKv*)calloc(msz, sizeof(InkTokKv));
    b->merge_hmask = msz - 1;
    for (size_t i = 0; i < merges->n; i++) {
      const jv* m = merges->items[i];
      if (m && m->type == J_STR) {
        ink_ht_put(b->merge_ht, b->merge_hmask, strdup(m->str), (int32_t)i);
      } else if (m && m->type == J_ARR && m->n == 2 && m->items[0]->type == J_STR && m->items[1]->type == J_STR) {
        size_t l0 = strlen(m->items[0]->str), l1 = strlen(m->items[1]->str);
        char* k = (char*)malloc(l0 + l1 + 2);
        memcpy(k, m->items[0]->str, l0); k[l0] = ' '; memcpy(k + l0 + 1, m->items[1]->str, l1 + 1);
        ink_ht_put(b->merge_ht, b->merge_hmask, k, (int32_t)i);
      }
    }
  }
  json_free(root);
  return 0;
}

static inline int ink_bpe_encode_chunk(const InkBpe* b, const char* s, size_t n, int32_t* out, int cap) {
  if (n == 0) return 0;
  // Convert bytes -> ByteLevel symbols
  size_t cap_sym = n * 2 + 8;
  char** syms = (char**)malloc(cap_sym * sizeof(char*));
  size_t ns = 0;
  for (size_t i = 0; i < n; i++) {
    uint8_t c = (uint8_t)s[i];
    char* p = (char*)malloc((size_t)b->b2u_len[c] + 1);
    memcpy(p, b->b2u_utf8[c], (size_t)b->b2u_len[c]);
    p[b->b2u_len[c]] = 0;
    syms[ns++] = p;
  }
  char pair[512];
  while (ns > 1) {
    int32_t best_r = 0x7fffffff;
    size_t best_i = (size_t)-1;
    for (size_t i = 0; i + 1 < ns; i++) {
      size_t l0 = strlen(syms[i]), l1 = strlen(syms[i + 1]);
      if (l0 + l1 + 2 >= sizeof pair) continue;
      memcpy(pair, syms[i], l0); pair[l0] = ' '; memcpy(pair + l0 + 1, syms[i + 1], l1 + 1);
      int32_t r = ink_ht_get(b->merge_ht, b->merge_hmask, pair, l0 + l1 + 1);
      if (r >= 0 && r < best_r) { best_r = r; best_i = i; }
    }
    if (best_i == (size_t)-1) break;
    size_t l0 = strlen(syms[best_i]), l1 = strlen(syms[best_i + 1]);
    char* merged = (char*)malloc(l0 + l1 + 1);
    memcpy(merged, syms[best_i], l0); memcpy(merged + l0, syms[best_i + 1], l1 + 1);
    free(syms[best_i]); free(syms[best_i + 1]);
    syms[best_i] = merged;
    memmove(&syms[best_i + 1], &syms[best_i + 2], (ns - best_i - 2) * sizeof(char*));
    ns--;
  }
  int out_n = 0;
  for (size_t i = 0; i < ns && out_n < cap; i++) {
    int32_t id = ink_ht_get(b->vocab_ht, b->vocab_hmask, syms[i], strlen(syms[i]));
    if (id >= 0) out[out_n++] = id;
    free(syms[i]);
  }
  free(syms);
  return out_n;
}

static inline int ink_bpe_encode(const InkBpe* b, const char* text, int32_t* out, int cap) {
  int n_out = 0;
  const char* p = text;
  while (*p && n_out < cap) {
    int matched = 0;
    for (int s = 0; s < b->n_specials; s++) {
      size_t sl = strlen(b->specials[s]);
      if (sl > 0 && strncmp(p, b->specials[s], sl) == 0) {
        out[n_out++] = b->special_ids[s];
        p += sl;
        matched = 1;
        break;
      }
    }
    if (matched) continue;
    // Find next special or word boundary
    const char* seg = p;
    while (*p) {
      int sp = 0;
      for (int s = 0; s < b->n_specials; s++) {
        size_t sl = strlen(b->specials[s]);
        if (sl > 0 && strncmp(p, b->specials[s], sl) == 0) { sp = 1; break; }
      }
      if (sp) break;
      p++;
      if ((p - seg) >= 128 && (*p == ' ' || *p == '\n')) break;
    }
    // Split segment into coarse word chunks (space + alnum / punct / newline)
    const char* q = seg;
    while (q < p && n_out < cap) {
      const char* w = q;
      if (*q == '\r' || *q == '\n') {
        while (q < p && (*q == '\r' || *q == '\n')) q++;
      } else if (*q == ' ' && q + 1 < p && (isalpha((unsigned char)q[1]) || (unsigned char)q[1] >= 128)) {
        q++;
        while (q < p && (isalpha((unsigned char)*q) || (unsigned char)*q >= 128)) q++;
      } else if (isalpha((unsigned char)*q) || (unsigned char)*q >= 128) {
        while (q < p && (isalpha((unsigned char)*q) || (unsigned char)*q >= 128)) q++;
      } else if (isdigit((unsigned char)*q)) {
        int d = 0;
        while (q < p && isdigit((unsigned char)*q) && d < 3) { q++; d++; }
      } else if (*q == ' ') {
        while (q < p && *q == ' ') q++;
      } else {
        q++;
        while (q < p && !isspace((unsigned char)*q) && !isalnum((unsigned char)*q) && (unsigned char)*q < 128) q++;
      }
      n_out += ink_bpe_encode_chunk(b, w, (size_t)(q - w), out + n_out, cap - n_out);
    }
  }
  return n_out;
}

static inline size_t ink_bpe_decode_id(const InkBpe* b, int32_t id, char* out, size_t cap) {
  if (id < 0 || id >= b->vocab_cap || !b->id_to_tok[id]) return 0;
  const char* s = b->id_to_tok[id];
  // Special tokens decode literally so serve.c can parse <|content_thinking|>/<|content_text|>/<|end_message|>
  for (int i = 0; i < b->n_specials; i++) {
    if (b->special_ids[i] == id) {
      size_t n = strlen(s);
      if (n >= cap) n = cap - 1;
      memcpy(out, s, n); out[n] = 0;
      return n;
    }
  }
  size_t k = 0;
  const uint8_t* p = (const uint8_t*)s;
  while (*p && k + 1 < cap) {
    uint32_t cp = 0;
    if (*p < 0x80) { cp = *p++; }
    else if ((*p & 0xE0) == 0xC0 && p[1]) { cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F); p += 2; }
    else if ((*p & 0xF0) == 0xE0 && p[1] && p[2]) { cp = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); p += 3; }
    else { out[k++] = (char)*p++; continue; }
    if (cp < 512 && b->u2b_tbl[cp] >= 0) out[k++] = (char)b->u2b_tbl[cp];
    else if (cp < 0x80) out[k++] = (char)cp;
  }
  out[k] = 0;
  return k;
}

static inline int ink_open_strata(InkModelMeta* m, const char* path, char* err, size_t err_cap) {
  memset(m, 0, sizeof *m);
  ink_cfg_defaults(&m->cfg);
  m->fp = fopen(path, "rb");
  if (!m->fp) { snprintf(err, err_cap, "cannot open %s", path); return -1; }
  uint8_t frame[4096];
  if (fread(frame, 1, 4096, m->fp) != 4096) { snprintf(err, err_cap, "short .strata header"); return -1; }
  static const uint8_t magic[8] = {'S','T','R','A','T','A',0,1};
  if (memcmp(frame, magic, 8) != 0) { snprintf(err, err_cap, "bad .strata magic"); return -1; }
  m->is_strata = 1;
  memcpy(&m->flags, frame + 8, 4);
  memcpy(&m->profile, frame + 12, 2);
  uint16_t nsec = 0;
  memcpy(&nsec, frame + 14, 2);
  if (nsec != 6) { snprintf(err, err_cap, "unexpected .strata sections=%u", nsec); return -1; }
  for (int i = 0; i < 6; i++) memcpy(&m->sections[i], frame + 0x10 + i * sizeof(InkSectionEntry), sizeof(InkSectionEntry));

  // Section 0: ZSTD manifest JSON
  if (m->sections[0].byte_length > 0) {
    uint8_t* zbuf = (uint8_t*)malloc((size_t)m->sections[0].byte_length);
    if (!zbuf || fseeko(m->fp, (off_t)m->sections[0].file_offset, SEEK_SET) != 0
      || fread(zbuf, 1, (size_t)m->sections[0].byte_length, m->fp) != (size_t)m->sections[0].byte_length) {
      free(zbuf); snprintf(err, err_cap, "cannot read section 0"); return -1;
    }
    char* mjson = NULL; size_t mlen = 0;
    int rc = ink_zstd_decompress(zbuf, (size_t)m->sections[0].byte_length, &mjson, &mlen, err, err_cap);
    free(zbuf);
    if (rc != 0) return -1;
    const char* jerr = NULL;
    jv* root = json_parse(mjson, &jerr);
    free(mjson);
    if (!root || root->type != J_OBJ) { snprintf(err, err_cap, "bad manifest JSON"); json_free(root); return -1; }
    const jv* bld = json_get(root, "build");
    if (bld && bld->type == J_STR) snprintf(m->build, sizeof m->build, "%s", bld->str);
    const jv* cfg = json_get(root, "config");
    const jv* tc = cfg ? json_get(cfg, "text_config") : NULL;
    if (tc && tc->type == J_OBJ) {
      const jv* v;
      if ((v = json_get(tc, "num_hidden_layers")) && v->type == J_NUM) m->cfg.n_phys = (int)v->num;
      if ((v = json_get(tc, "hidden_size")) && v->type == J_NUM) m->cfg.hidden = (int)v->num;
      if ((v = json_get(tc, "intermediate_size")) && v->type == J_NUM) m->cfg.inter = (int)v->num;
      if ((v = json_get(tc, "num_attention_heads")) && v->type == J_NUM) m->cfg.n_head = (int)v->num;
      if ((v = json_get(tc, "num_key_value_heads")) && v->type == J_NUM) m->cfg.n_kv = (int)v->num;
      if ((v = json_get(tc, "head_dim")) && v->type == J_NUM) m->cfg.head_dim = (int)v->num;
      if ((v = json_get(tc, "n_routed_experts")) && v->type == J_NUM) m->cfg.n_experts = (int)v->num;
      if ((v = json_get(tc, "num_experts_per_tok")) && v->type == J_NUM) m->cfg.top_k = (int)v->num;
      if ((v = json_get(tc, "vocab_size")) && v->type == J_NUM) m->cfg.vocab_padded = (int)v->num;
      if ((v = json_get(tc, "unpadded_vocab_size")) && v->type == J_NUM) m->cfg.vocab_unpadded = (int)v->num;
      if ((v = json_get(tc, "eos_token_id")) && v->type == J_NUM) m->cfg.eos_token_id = (int)v->num;
      if ((v = json_get(tc, "route_scale")) && v->type == J_NUM) m->cfg.route_scale = (float)v->num;
      if ((v = json_get(tc, "final_logit_softcapping")) && v->type == J_NUM) m->cfg.logit_softcap = (float)v->num;
      const jv* lt = json_get(tc, "layer_types");
      if (lt && lt->type == J_ARR) {
        for (size_t i = 0; i < lt->n && i < INK_N_PHYS; i++) {
          m->cfg.is_swa[i] = (lt->items[i]->type == J_STR && strstr(lt->items[i]->str, "sliding") != NULL) ? 1 : 0;
        }
      }
      const jv* lp = json_get(tc, "loop_config");
      if (lp && lp->type == J_OBJ) {
        if ((v = json_get(lp, "core_start")) && v->type == J_NUM) m->cfg.core_start = (int)v->num;
        if ((v = json_get(lp, "core_end")) && v->type == J_NUM) m->cfg.core_end = (int)v->num;
        if ((v = json_get(lp, "num_passes")) && v->type == J_NUM) m->cfg.num_passes = (int)v->num;
        if ((v = json_get(lp, "gate_rank")) && v->type == J_NUM) m->cfg.gate_rank = (int)v->num;
      }
    }
    ink_cfg_build_visits(&m->cfg);
    m->n_trunk = ink_parse_tensor_map(json_get(root, "trunk_tensors"), m->trunk, INK_MAX_TRUNK_T);
    const jv* ds = json_get(root, "dspark_tensors");
    if (ds && ds->type == J_OBJ && ds->n > 0) {
      m->n_dspark = ink_parse_tensor_map(ds, m->dspark, INK_MAX_DSPARK_T);
      m->has_dspark = 1;
      m->dspark_rank = 256;
    }
    const jv* tj = json_get(root, "tokenizer_json");
    if (tj && tj->type == J_STR) m->tokenizer_json = strdup(tj->str);
    const jv* ct = json_get(root, "chat_template");
    if (ct && ct->type == J_STR) m->chat_template = strdup(ct->str);
    json_free(root);
  }
  // Section 3: expert directory (+ optional v1.1 loop-weighted vram_load_order)
  if (m->sections[3].byte_length > 0) {
    size_t s3b = (size_t)m->sections[3].byte_length;
    uint8_t* s3 = (uint8_t*)malloc(s3b);
    if (!s3 || fseeko(m->fp, (off_t)m->sections[3].file_offset, SEEK_SET) != 0 || fread(s3, 1, s3b, m->fp) != s3b) {
      free(s3); snprintf(err, err_cap, "cannot read section 3 expert dir"); return -1;
    }
    int has_vo = (m->sections[3].flags & 1u) != 0;
    size_t stride = sizeof(InkExpertDirEntry) + (has_vo ? 2 : 0);
    int n = (int)(s3b / stride);
    m->n_experts_dir = n;
    m->experts = (InkExpertDirEntry*)malloc((size_t)n * sizeof(InkExpertDirEntry));
    memcpy(m->experts, s3, (size_t)n * sizeof(InkExpertDirEntry));
    m->vram_order = (uint16_t*)malloc((size_t)n * sizeof(uint16_t));
    if (has_vo) memcpy(m->vram_order, s3 + (size_t)n * sizeof(InkExpertDirEntry), (size_t)n * 2);
    else for (int i = 0; i < n; i++) m->vram_order[i] = (uint16_t)i;
    free(s3);
  }
  if (m->tokenizer_json) ink_bpe_load_json(&m->bpe, m->tokenizer_json);
  return 0;
}

static inline int ink_read_at(const InkModelMeta* m, uint64_t off, void* dst, size_t n) {
  if (!m->fp || fseeko(m->fp, (off_t)off, SEEK_SET) != 0) return -1;
  return fread(dst, 1, n, m->fp) == n ? 0 : -1;
}

#endif // LLB_INKLING_FMT_H
