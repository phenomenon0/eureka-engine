// ec_bpe.c — 100% Pure C11 HuggingFace tokenizer.json Byte-Level BPE Tokenizer
// Directly parses HuggingFace tokenizer.json (model.vocab, model.merges, added_tokens)
// into compact open-addressing FNV-1a hash tables for sub-millisecond encoding.

#define _POSIX_C_SOURCE 200809L
#include "ec_bpe.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t off;
  uint16_t len;
  int32_t  id;
} ec_vocab_slot_t;

typedef struct {
  uint32_t off;  // "left\0right" packed in merge_pool as "left right" (space-separated)
  uint16_t len;
  int32_t  rank;
} ec_merge_slot_t;

typedef struct {
  char    text[64];
  int     len;
  int32_t id;
} ec_special_t;

struct ec_bpe {
  char            *vocab_pool;
  size_t           vocab_pool_len, vocab_pool_cap;
  ec_vocab_slot_t *vocab_tab;
  uint32_t         vocab_mask;
  int              n_vocab;

  char            *merge_pool;
  size_t           merge_pool_len, merge_pool_cap;
  ec_merge_slot_t *merge_tab;
  uint32_t         merge_mask;
  int              n_merges;

  ec_special_t    *specials;
  int              n_specials, cap_specials;

  char             byte_to_utf8[256][4];
  uint8_t          byte_to_len[256];
  int32_t          pad_id;
};

static uint64_t fnv1a_bytes(const char *s, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static int utf8_encode_cp(uint32_t cp, char out[4]) {
  if (cp < 0x80) { out[0] = (char)cp; return 1; }
  if (cp < 0x800) {
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  out[0] = (char)(0xF0 | (cp >> 18));
  out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  out[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

static void init_gpt2_byte_table(ec_bpe_t *t) {
  int direct[256] = {0};
  for (int b = '!'; b <= '~'; b++) direct[b] = 1;
  for (int b = 0xA1; b <= 0xAC; b++) direct[b] = 1;
  for (int b = 0xAE; b <= 0xFF; b++) direct[b] = 1;
  uint32_t extra = 256;
  for (int b = 0; b < 256; b++) {
    uint32_t cp = direct[b] ? (uint32_t)b : extra++;
    t->byte_to_len[b] = (uint8_t)utf8_encode_cp(cp, t->byte_to_utf8[b]);
  }
}

static uint32_t pool_append(char **pool, size_t *len, size_t *cap, const char *s, size_t n) {
  if (*len + n + 1 > *cap) {
    size_t nc = (*cap == 0) ? (1ULL << 20) : (*cap * 2);
    while (nc < *len + n + 1) nc *= 2;
    *pool = (char *)realloc(*pool, nc);
    *cap = nc;
  }
  uint32_t off = (uint32_t)*len;
  memcpy(*pool + off, s, n);
  (*pool)[off + n] = '\0';
  *len += n + 1;
  return off;
}

static void vocab_put(ec_bpe_t *t, const char *s, size_t n, int32_t id) {
  if (n > 65535) return;
  uint32_t off = pool_append(&t->vocab_pool, &t->vocab_pool_len, &t->vocab_pool_cap, s, n);
  uint32_t idx = (uint32_t)(fnv1a_bytes(s, n) & t->vocab_mask);
  while (t->vocab_tab[idx].id != -1) {
    if (t->vocab_tab[idx].len == n && memcmp(t->vocab_pool + t->vocab_tab[idx].off, s, n) == 0) {
      t->vocab_tab[idx].id = id;
      return;
    }
    idx = (idx + 1) & t->vocab_mask;
  }
  t->vocab_tab[idx].off = off;
  t->vocab_tab[idx].len = (uint16_t)n;
  t->vocab_tab[idx].id = id;
  t->n_vocab++;
}

static int32_t vocab_get(const ec_bpe_t *t, const char *s, size_t n) {
  uint32_t idx = (uint32_t)(fnv1a_bytes(s, n) & t->vocab_mask);
  while (t->vocab_tab[idx].id != -1) {
    if (t->vocab_tab[idx].len == n && memcmp(t->vocab_pool + t->vocab_tab[idx].off, s, n) == 0)
      return t->vocab_tab[idx].id;
    idx = (idx + 1) & t->vocab_mask;
  }
  return -1;
}

static void merge_put(ec_bpe_t *t, const char *s, size_t n, int32_t rank) {
  if (n > 65535) return;
  uint32_t off = pool_append(&t->merge_pool, &t->merge_pool_len, &t->merge_pool_cap, s, n);
  uint32_t idx = (uint32_t)(fnv1a_bytes(s, n) & t->merge_mask);
  while (t->merge_tab[idx].rank != -1) {
    idx = (idx + 1) & t->merge_mask;
  }
  t->merge_tab[idx].off = off;
  t->merge_tab[idx].len = (uint16_t)n;
  t->merge_tab[idx].rank = rank;
  t->n_merges++;
}

static int32_t merge_get(const ec_bpe_t *t, const char *a, size_t la, const char *b, size_t lb) {
  char key[512];
  if (la + 1 + lb >= sizeof key) return -1;
  memcpy(key, a, la);
  key[la] = ' ';
  memcpy(key + la + 1, b, lb);
  size_t n = la + 1 + lb;
  uint32_t idx = (uint32_t)(fnv1a_bytes(key, n) & t->merge_mask);
  while (t->merge_tab[idx].rank != -1) {
    if (t->merge_tab[idx].len == n && memcmp(t->merge_pool + t->merge_tab[idx].off, key, n) == 0)
      return t->merge_tab[idx].rank;
    idx = (idx + 1) & t->merge_mask;
  }
  return -1;
}

static int hex_val(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parse a JSON string starting at *pp (which points to opening '"') into out[cap].
// Handles \uXXXX (including UTF-16 surrogate pairs) and standard JSON escapes.
static int parse_json_str(const char **pp, char *out, size_t cap, size_t *out_len) {
  const char *p = *pp;
  if (*p != '"') return -1;
  p++;
  size_t n = 0;
  while (*p && *p != '"') {
    if (*p == '\\') {
      p++;
      if (!*p) return -1;
      char c = *p++;
      if (c == 'n') { if (n + 1 < cap) out[n++] = '\n'; }
      else if (c == 'r') { if (n + 1 < cap) out[n++] = '\r'; }
      else if (c == 't') { if (n + 1 < cap) out[n++] = '\t'; }
      else if (c == 'b') { if (n + 1 < cap) out[n++] = '\b'; }
      else if (c == 'f') { if (n + 1 < cap) out[n++] = '\f'; }
      else if (c == '"' || c == '\\' || c == '/') { if (n + 1 < cap) out[n++] = c; }
      else if (c == 'u') {
        if (!p[0] || !p[1] || !p[2] || !p[3]) return -1;
        uint32_t cp = (uint32_t)((hex_val(p[0]) << 12) | (hex_val(p[1]) << 8) | (hex_val(p[2]) << 4) | hex_val(p[3]));
        p += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
          p += 2;
          uint32_t lo = (uint32_t)((hex_val(p[0]) << 12) | (hex_val(p[1]) << 8) | (hex_val(p[2]) << 4) | hex_val(p[3]));
          p += 4;
          cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
        }
        char u[4];
        int ul = utf8_encode_cp(cp, u);
        if (n + (size_t)ul < cap) { memcpy(out + n, u, (size_t)ul); n += (size_t)ul; }
      }
    } else {
      if (n + 1 < cap) out[n++] = *p;
      p++;
    }
  }
  if (*p != '"') return -1;
  out[n] = '\0';
  if (out_len) *out_len = n;
  *pp = p + 1;
  return 0;
}

static const char *skip_ws(const char *p) {
  while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
  return p;
}

static int cmp_special_desc(const void *a, const void *b) {
  return ((const ec_special_t *)b)->len - ((const ec_special_t *)a)->len;
}

ec_bpe_t *ec_bpe_load(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (sz <= 0 || sz > 64 * 1024 * 1024) { fclose(fp); return NULL; }
  char *js = (char *)malloc((size_t)sz + 1);
  if (!js || fread(js, 1, (size_t)sz, fp) != (size_t)sz) { free(js); fclose(fp); return NULL; }
  js[sz] = '\0';
  fclose(fp);

  ec_bpe_t *t = (ec_bpe_t *)calloc(1, sizeof *t);
  init_gpt2_byte_table(t);
  t->pad_id = 151643;
  // 524,288 slots for vocab (~152k entries -> load 0.29) and merges (~151k entries)
  uint32_t vcap = 1U << 19;
  t->vocab_mask = vcap - 1;
  t->vocab_tab = (ec_vocab_slot_t *)malloc(vcap * sizeof(ec_vocab_slot_t));
  for (uint32_t i = 0; i < vcap; i++) t->vocab_tab[i].id = -1;

  uint32_t mcap = 1U << 19;
  t->merge_mask = mcap - 1;
  t->merge_tab = (ec_merge_slot_t *)malloc(mcap * sizeof(ec_merge_slot_t));
  for (uint32_t i = 0; i < mcap; i++) t->merge_tab[i].rank = -1;

  // 1. Parse "added_tokens": [ { "id": ..., "content": ... }, ... ]
  const char *ap = strstr(js, "\"added_tokens\"");
  if (ap) {
    ap = strchr(ap, '[');
    if (ap) {
      ap++;
      while (*ap && *ap != ']') {
        ap = skip_ws(ap);
        if (*ap != '{') { ap++; continue; }
        const char *end = strchr(ap, '}');
        if (!end) break;
        const char *idp = strstr(ap, "\"id\"");
        const char *cp = strstr(ap, "\"content\"");
        if (idp && cp && idp < end && cp < end) {
          idp = strchr(idp, ':');
          int32_t id = idp ? (int32_t)strtol(idp + 1, NULL, 10) : -1;
          cp = strchr(cp, ':');
          if (cp) {
            cp = skip_ws(cp + 1);
            char sbuf[128]; size_t slen = 0;
            if (parse_json_str(&cp, sbuf, sizeof sbuf, &slen) == 0 && id >= 0 && slen > 0 && slen < 63) {
              if (t->n_specials == t->cap_specials) {
                t->cap_specials = t->cap_specials ? t->cap_specials * 2 : 64;
                t->specials = (ec_special_t *)realloc(t->specials, (size_t)t->cap_specials * sizeof(ec_special_t));
              }
              memcpy(t->specials[t->n_specials].text, sbuf, slen + 1);
              t->specials[t->n_specials].len = (int)slen;
              t->specials[t->n_specials].id = id;
              t->n_specials++;
              vocab_put(t, sbuf, slen, id);
              if (strcmp(sbuf, "<|endoftext|>") == 0) t->pad_id = id;
            }
          }
        }
        ap = end + 1;
      }
      if (t->n_specials > 1) qsort(t->specials, (size_t)t->n_specials, sizeof(ec_special_t), cmp_special_desc);
    }
  }

  // 2. Parse "vocab": { "tok": id, ... } inside "model"
  const char *mp = strstr(js, "\"model\"");
  const char *vp = mp ? strstr(mp, "\"vocab\"") : strstr(js, "\"vocab\"");
  if (vp) {
    vp = strchr(vp, '{');
    if (vp) {
      vp++;
      char key[512]; size_t klen = 0;
      while (*vp) {
        vp = skip_ws(vp);
        if (*vp == '}') { vp++; break; }
        if (*vp == ',') { vp++; continue; }
        if (parse_json_str(&vp, key, sizeof key, &klen) != 0) break;
        vp = skip_ws(vp);
        if (*vp != ':') break;
        vp = skip_ws(vp + 1);
        char *ep = NULL;
        long id = strtol(vp, &ep, 10);
        if (!ep || ep == vp) break;
        vp = ep;
        vocab_put(t, key, klen, (int32_t)id);
      }
    }
  }

  // 3. Parse "merges": [ "a b", ... ] OR [ ["a","b"], ... ]
  const char *mg = mp ? strstr(mp, "\"merges\"") : strstr(js, "\"merges\"");
  if (mg) {
    mg = strchr(mg, '[');
    if (mg) {
      mg++;
      int32_t rank = 0;
      char s1[256], s2[256], comb[512];
      size_t l1 = 0, l2 = 0;
      while (*mg) {
        mg = skip_ws(mg);
        if (*mg == ']') break;
        if (*mg == ',') { mg++; continue; }
        if (*mg == '"') {
          if (parse_json_str(&mg, comb, sizeof comb, &l1) != 0) break;
          merge_put(t, comb, l1, rank++);
        } else if (*mg == '[') {
          mg = skip_ws(mg + 1);
          if (parse_json_str(&mg, s1, sizeof s1, &l1) != 0) break;
          mg = skip_ws(mg);
          if (*mg == ',') mg = skip_ws(mg + 1);
          if (parse_json_str(&mg, s2, sizeof s2, &l2) != 0) break;
          mg = skip_ws(mg);
          if (*mg == ']') mg++;
          if (l1 + 1 + l2 < sizeof comb) {
            memcpy(comb, s1, l1);
            comb[l1] = ' ';
            memcpy(comb + l1 + 1, s2, l2);
            comb[l1 + 1 + l2] = '\0';
            merge_put(t, comb, l1 + 1 + l2, rank++);
          }
        } else {
          break;
        }
      }
    }
  }

  free(js);
  if (t->n_vocab < 1000) { ec_bpe_free(t); return NULL; }
  return t;
}

void ec_bpe_free(ec_bpe_t *t) {
  if (!t) return;
  free(t->vocab_pool);
  free(t->vocab_tab);
  free(t->merge_pool);
  free(t->merge_tab);
  free(t->specials);
  free(t);
}

// Run GPT-2 byte-level BPE on a single pre-tokenized chunk [s, s+len)
static int bpe_encode_chunk(const ec_bpe_t *t, const char *s, size_t len, int32_t *out, int max_out) {
  if (len == 0 || max_out <= 0) return 0;
  // Convert bytes to GPT-2 unicode symbols
  enum { MAX_SYMS = 512 };
  if (len > MAX_SYMS) len = MAX_SYMS;
  char *syms[MAX_SYMS];
  size_t slens[MAX_SYMS];
  char arena[4096];
  size_t arena_used = 0;

  int n = 0;
  for (size_t i = 0; i < len; i++) {
    unsigned char b = (unsigned char)s[i];
    uint8_t ul = t->byte_to_len[b];
    if (arena_used + ul + 1 > sizeof arena) break;
    syms[n] = arena + arena_used;
    memcpy(syms[n], t->byte_to_utf8[b], ul);
    syms[n][ul] = '\0';
    slens[n] = ul;
    arena_used += ul + 1;
    n++;
  }

  // Greedy lowest-rank BPE merge loop
  while (n > 1) {
    int best_i = -1;
    int32_t best_rank = 0x7fffffff;
    for (int i = 0; i + 1 < n; i++) {
      int32_t r = merge_get(t, syms[i], slens[i], syms[i + 1], slens[i + 1]);
      if (r >= 0 && r < best_rank) {
        best_rank = r;
        best_i = i;
      }
    }
    if (best_i < 0) break;
    size_t nl = slens[best_i] + slens[best_i + 1];
    if (arena_used + nl + 1 > sizeof arena) break;
    char *merged = arena + arena_used;
    memcpy(merged, syms[best_i], slens[best_i]);
    memcpy(merged + slens[best_i], syms[best_i + 1], slens[best_i + 1]);
    merged[nl] = '\0';
    arena_used += nl + 1;
    syms[best_i] = merged;
    slens[best_i] = nl;
    for (int j = best_i + 1; j + 1 < n; j++) {
      syms[j] = syms[j + 1];
      slens[j] = slens[j + 1];
    }
    n--;
  }

  int written = 0;
  for (int i = 0; i < n && written < max_out; i++) {
    int32_t id = vocab_get(t, syms[i], slens[i]);
    if (id >= 0) out[written++] = id;
  }
  return written;
}

static int is_ascii_letter(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; }
static int is_ascii_digit(unsigned char c)  { return (c >= '0' && c <= '9'); }
static int is_space_ch(unsigned char c)     { return c == ' ' || c == '\t'; }

// Qwen2/Qwen3 pre-tokenizer splitter (faithful ASCII + UTF-8 letter/digit/punct/newline segmentation)
static int encode_plain_segment(const ec_bpe_t *t, const char *s, size_t len, int32_t *out, int max_out) {
  int total = 0;
  size_t i = 0;
  while (i < len && total < max_out) {
    size_t start = i;
    unsigned char c = (unsigned char)s[i];
    // Contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
    if (c == '\'' && i + 1 < len) {
      char c1 = (char)tolower((unsigned char)s[i + 1]);
      char c2 = (i + 2 < len) ? (char)tolower((unsigned char)s[i + 2]) : 0;
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        i += 2;
      } else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
        i += 3;
      } else {
        i++;
      }
    } else if (c == '\r' || c == '\n') {
      while (i < len && (s[i] == '\r' || s[i] == '\n')) i++;
    } else if (is_ascii_digit(c)) {
      // Qwen3 splits digits individually (\p{N})!
      i++;
    } else if (is_ascii_letter(c)) {
      while (i < len && is_ascii_letter((unsigned char)s[i])) i++;
    } else if (is_space_ch(c)) {
      // Leading single space attaches to following letters or punctuation (standard GPT-2 / Qwen3 rule)
      if (i + 1 < len && is_ascii_letter((unsigned char)s[i + 1])) {
        i++;
        while (i < len && is_ascii_letter((unsigned char)s[i])) i++;
      } else if (i + 1 < len && !is_space_ch((unsigned char)s[i + 1]) && s[i + 1] != '\r' && s[i + 1] != '\n' && !is_ascii_digit((unsigned char)s[i + 1])) {
        i++;
        while (i < len && !is_space_ch((unsigned char)s[i]) && !is_ascii_letter((unsigned char)s[i]) &&
               !is_ascii_digit((unsigned char)s[i]) && s[i] != '\r' && s[i] != '\n') i++;
      } else {
        while (i < len && is_space_ch((unsigned char)s[i])) i++;
      }
    } else {
      // Punctuation run
      while (i < len && !is_space_ch((unsigned char)s[i]) && !is_ascii_letter((unsigned char)s[i]) &&
             !is_ascii_digit((unsigned char)s[i]) && s[i] != '\r' && s[i] != '\n' && s[i] != '\'') i++;
      if (i == start) i++;
    }
    total += bpe_encode_chunk(t, s + start, i - start, out + total, max_out - total);
  }
  return total;
}

int ec_bpe_encode(const ec_bpe_t *t, const char *text, int32_t *out_ids, int max_out) {
  if (!t || !text || !out_ids || max_out <= 0) return -1;
  size_t len = strlen(text);
  int total = 0;
  size_t pos = 0, seg_start = 0;
  while (pos < len && total < max_out) {
    int matched = -1;
    for (int s = 0; s < t->n_specials; s++) {
      int sl = t->specials[s].len;
      if (pos + (size_t)sl <= len && memcmp(text + pos, t->specials[s].text, (size_t)sl) == 0) {
        matched = s;
        break;
      }
    }
    if (matched >= 0) {
      if (pos > seg_start) {
        total += encode_plain_segment(t, text + seg_start, pos - seg_start, out_ids + total, max_out - total);
      }
      if (total < max_out) out_ids[total++] = t->specials[matched].id;
      pos += (size_t)t->specials[matched].len;
      seg_start = pos;
    } else {
      pos++;
    }
  }
  if (pos > seg_start && total < max_out) {
    total += encode_plain_segment(t, text + seg_start, pos - seg_start, out_ids + total, max_out - total);
  }
  return total;
}

int ec_bpe_encode_flux2_qwen3(
    const ec_bpe_t *tok, const char *prompt,
    int32_t *out_ids, uint8_t *out_mask, int fixed_len, int *out_real_len) {
  if (!tok || !out_ids || fixed_len <= 0) return -1;
  char templ[4096];
  snprintf(templ, sizeof templ,
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
      prompt ? prompt : "");
  int n = ec_bpe_encode(tok, templ, out_ids, fixed_len);
  if (n < 0) return -1;
  if (out_real_len) *out_real_len = n;
  for (int i = 0; i < fixed_len; i++) {
    if (i < n) {
      if (out_mask) out_mask[i] = 1;
    } else {
      out_ids[i] = tok->pad_id;
      if (out_mask) out_mask[i] = 0;
    }
  }
  return n;
}
