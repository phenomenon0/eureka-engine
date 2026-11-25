// ec_unigram.c — 100% Pure C11 HuggingFace Unigram / SentencePiece Tokenizer
// Zero external dependencies. Parses umt5_tokenizer.json "model.vocab" array directly
// into an open-addressing hash table and runs exact Viterbi max-log-prob segmentation.

#define _POSIX_C_SOURCE 200809L
#include "ec_unigram.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t hash;
  uint32_t off;    // offset in arena
  uint16_t len;    // byte length of UTF-8 piece
  int32_t  id;     // -1 = empty slot
  float    score;
} ug_slot_t;

struct ec_unigram {
  char      *arena;
  size_t     arena_len;
  ug_slot_t *table;
  size_t     table_mask; // power-of-2 minus 1
  int        max_piece_len;
  int32_t    unk_id;
  int32_t    eos_id;
  int32_t    byte_ids[256]; // SentencePiece <0x00>..<0xFF> UTF-8 byte fallback IDs (-1 if absent)
};

static inline uint64_t ug_fnv1a(const char *s, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (uint8_t)s[i];
    h *= 1099511628211ULL;
  }
  return h ? h : 1;
}

static int hex_val(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static size_t encode_utf8(uint32_t cp, char *out) {
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

// Parse JSON string starting at *p (which points at opening '"'). Advances *p past closing '"'.
static int parse_json_str(const char **p, const char *end, char *out, size_t cap, size_t *out_len) {
  const char *s = *p;
  if (s >= end || *s != '"') return -1;
  s++;
  size_t n = 0;
  while (s < end && *s != '"') {
    char c = *s++;
    if (c == '\\') {
      if (s >= end) return -1;
      char e = *s++;
      if (e == '"' || e == '\\' || e == '/') { if (n + 1 < cap) out[n++] = e; }
      else if (e == 'b') { if (n + 1 < cap) out[n++] = '\b'; }
      else if (e == 'f') { if (n + 1 < cap) out[n++] = '\f'; }
      else if (e == 'n') { if (n + 1 < cap) out[n++] = '\n'; }
      else if (e == 'r') { if (n + 1 < cap) out[n++] = '\r'; }
      else if (e == 't') { if (n + 1 < cap) out[n++] = '\t'; }
      else if (e == 'u') {
        if (s + 4 > end) return -1;
        uint32_t cp = 0;
        for (int i = 0; i < 4; i++) {
          int hv = hex_val(s[i]);
          if (hv < 0) return -1;
          cp = (cp << 4) | (uint32_t)hv;
        }
        s += 4;
        // Handle UTF-16 surrogate pair
        if (cp >= 0xD800 && cp <= 0xDBFF && s + 6 <= end && s[0] == '\\' && s[1] == 'u') {
          uint32_t lo = 0;
          int ok = 1;
          for (int i = 0; i < 4; i++) {
            int hv = hex_val(s[2 + i]);
            if (hv < 0) { ok = 0; break; }
            lo = (lo << 4) | (uint32_t)hv;
          }
          if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
            s += 6;
          }
        }
        char u8[4];
        size_t ul = encode_utf8(cp, u8);
        if (n + ul < cap) { memcpy(out + n, u8, ul); n += ul; }
      }
    } else {
      if (n + 1 < cap) out[n++] = c;
    }
  }
  if (s >= end || *s != '"') return -1;
  s++;
  out[n] = '\0';
  *out_len = n;
  *p = s;
  return 0;
}

static inline void skip_ws(const char **p, const char *end) {
  while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) (*p)++;
}

static void ug_insert(ec_unigram_t *ug, uint32_t off, uint16_t len, int32_t id, float score) {
  const char *s = ug->arena + off;
  uint64_t h = ug_fnv1a(s, len);
  size_t idx = (size_t)h & ug->table_mask;
  while (ug->table[idx].id >= 0) {
    if (ug->table[idx].hash == h && ug->table[idx].len == len &&
        memcmp(ug->arena + ug->table[idx].off, s, len) == 0) {
      // Keep higher-scoring duplicate if any
      if (score > ug->table[idx].score) {
        ug->table[idx].id = id;
        ug->table[idx].score = score;
      }
      return;
    }
    idx = (idx + 1) & ug->table_mask;
  }
  ug->table[idx].hash  = h;
  ug->table[idx].off   = off;
  ug->table[idx].len   = len;
  ug->table[idx].id    = id;
  ug->table[idx].score = score;
  if ((int)len > ug->max_piece_len) ug->max_piece_len = (int)len;
}

static const ug_slot_t *ug_lookup(const ec_unigram_t *ug, const char *s, size_t len) {
  uint64_t h = ug_fnv1a(s, len);
  size_t idx = (size_t)h & ug->table_mask;
  while (ug->table[idx].id >= 0) {
    if (ug->table[idx].hash == h && ug->table[idx].len == len &&
        memcmp(ug->arena + ug->table[idx].off, s, len) == 0) {
      return &ug->table[idx];
    }
    idx = (idx + 1) & ug->table_mask;
  }
  return NULL;
}

ec_unigram_t *ec_unigram_load(const char *tokenizer_json_path) {
  FILE *f = fopen(tokenizer_json_path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long fsz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (fsz <= 0 || fsz > 128L * 1024 * 1024) { fclose(f); return NULL; }
  char *buf = (char *)malloc((size_t)fsz + 1);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) { fclose(f); free(buf); return NULL; }
  buf[fsz] = '\0';
  fclose(f);

  // Locate "\"model\"" then "\"vocab\""
  const char *end = buf + fsz;
  const char *mpos = strstr(buf, "\"model\"");
  if (!mpos) { free(buf); return NULL; }
  const char *vpos = strstr(mpos, "\"vocab\"");
  if (!vpos) { free(buf); return NULL; }
  const char *p = strchr(vpos + 7, '[');
  if (!p) { free(buf); return NULL; }
  p++; // inside vocab array

  ec_unigram_t *ug = (ec_unigram_t *)calloc(1, sizeof *ug);
  ug->unk_id = 2; // official google/umt5-xxl: <pad>=0, </s>=1, <unk>=2
  ug->eos_id = 1;
  for (int b = 0; b < 256; b++) ug->byte_ids[b] = -1;
  size_t table_cap = 1 << 20; // 1,048,576 slots (~25% load factor for 256k vocab!)
  ug->table_mask = table_cap - 1;
  ug->table = (ug_slot_t *)malloc(table_cap * sizeof(ug_slot_t));
  for (size_t i = 0; i < table_cap; i++) ug->table[i].id = -1;

  size_t arena_cap = 8 * 1024 * 1024;
  ug->arena = (char *)malloc(arena_cap);
  ug->arena_len = 0;

  char piece[512];
  int32_t id = 0;
  while (p < end) {
    skip_ws(&p, end);
    if (p >= end || *p == ']') break;
    if (*p == ',') { p++; continue; }
    if (*p != '[') break;
    p++;
    skip_ws(&p, end);
    size_t plen = 0;
    if (parse_json_str(&p, end, piece, sizeof piece, &plen) != 0) break;
    skip_ws(&p, end);
    if (p < end && *p == ',') p++;
    skip_ws(&p, end);
    char *num_end = NULL;
    float score = strtof(p, &num_end);
    if (!num_end || num_end == p) break;
    p = num_end;
    skip_ws(&p, end);
    if (p < end && *p == ']') p++;

    // Detect <unk> & SentencePiece <0xXX> UTF-8 byte fallback tokens!
    if (plen == 5 && memcmp(piece, "<unk>", 5) == 0) {
      ug->unk_id = id;
    } else if (plen == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
      int hi = hex_val(piece[3]), lo = hex_val(piece[4]);
      if (hi >= 0 && lo >= 0) ug->byte_ids[(hi << 4) | lo] = id;
    }

    // Skip special/control tokens 0..2 from matching raw text, insert all valid pieces
    if (id >= 3 && plen > 0 && plen < 256 && !(plen == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>')) {
      if (ug->arena_len + plen + 1 > arena_cap) {
        arena_cap *= 2;
        ug->arena = (char *)realloc(ug->arena, arena_cap);
      }
      uint32_t off = (uint32_t)ug->arena_len;
      memcpy(ug->arena + off, piece, plen);
      ug->arena[off + plen] = '\0';
      ug->arena_len += plen + 1;
      ug_insert(ug, off, (uint16_t)plen, id, score);
    }
    id++;
  }
  free(buf);
  if (id < 1000) { ec_unigram_free(ug); return NULL; }
  return ug;
}

void ec_unigram_free(ec_unigram_t *ug) {
  if (!ug) return;
  free(ug->arena);
  free(ug->table);
  free(ug);
}

static size_t utf8_char_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

int ec_unigram_encode(const ec_unigram_t *ug, const char *text, int32_t *out_ids, int max_out) {
  if (!ug || !out_ids || max_out <= 0) return -1;
  if (!text) text = "";

  // 1. Whitespace normalize (collapse runs of whitespace to single ' ', trim ends)
  size_t raw_len = strlen(text);
  char *norm = (char *)malloc(raw_len * 3 + 8);
  size_t nl = 0;
  int in_ws = 1;
  for (size_t i = 0; i < raw_len; i++) {
    unsigned char c = (unsigned char)text[i];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      if (!in_ws) {
        norm[nl++] = '\xE2'; norm[nl++] = '\x96'; norm[nl++] = '\x81'; // '▁'
        in_ws = 1;
      }
    } else {
      if (nl == 0) {
        // add_prefix_space = true
        norm[nl++] = '\xE2'; norm[nl++] = '\x96'; norm[nl++] = '\x81';
      }
      norm[nl++] = (char)c;
      in_ws = 0;
    }
  }
  // Trim trailing '▁' if original ended in whitespace
  while (nl >= 3 && (uint8_t)norm[nl-3] == 0xE2 && (uint8_t)norm[nl-2] == 0x96 && (uint8_t)norm[nl-1] == 0x81) {
    nl -= 3;
  }
  norm[nl] = '\0';

  if (nl == 0) {
    out_ids[0] = ug->eos_id; // </s> = 1
    free(norm);
    return 1;
  }

  // 2. Exact Viterbi DP over byte positions 0..nl
  double *dp   = (double *)malloc((nl + 1) * sizeof(double));
  int    *prev = (int *)malloc((nl + 1) * sizeof(int));
  int32_t *tok = (int32_t *)malloc((nl + 1) * sizeof(int32_t));
  for (size_t i = 0; i <= nl; i++) { dp[i] = -1e30; prev[i] = -1; tok[i] = ug->unk_id; }
  dp[0] = 0.0;

  for (size_t i = 0; i < nl; i++) {
    if (dp[i] < -1e29) continue;
    int max_l = ug->max_piece_len;
    if (i + (size_t)max_l > nl) max_l = (int)(nl - i);
    int matched = 0;
    for (int l = 1; l <= max_l; l++) {
      const ug_slot_t *s = ug_lookup(ug, norm + i, (size_t)l);
      if (s) {
        double cand = dp[i] + (double)s->score;
        if (cand > dp[i + l]) {
          dp[i + l] = cand;
          prev[i + l] = (int)i;
          tok[i + l] = s->id;
        }
        matched = 1;
      }
    }
    // Always permit 1 UTF-8 codepoint <unk> fallback so DP always reaches nl
    size_t cl = utf8_char_len((unsigned char)norm[i]);
    if (i + cl > nl) cl = nl - i;
    double unk_cand = dp[i] + (matched ? -100.0 : -20.0);
    if (unk_cand > dp[i + cl]) {
      dp[i + cl] = unk_cand;
      prev[i + cl] = (int)i;
      tok[i + cl] = ug->unk_id;
    }
  }

  // 3. Backtrack & reverse (with exact SentencePiece <0xXX> UTF-8 byte-fallback for unknown codepoints!)
  int32_t *rev = (int32_t *)malloc((nl * 4 + 4) * sizeof(int32_t));
  int count = 0;
  int cur = (int)nl;
  while (cur > 0 && prev[cur] >= 0) {
    int pcur = prev[cur];
    if (tok[cur] == ug->unk_id) {
      int all_bytes_known = 1;
      for (int bi = pcur; bi < cur; bi++) {
        if (ug->byte_ids[(unsigned char)norm[bi]] < 0) { all_bytes_known = 0; break; }
      }
      if (all_bytes_known && cur > pcur) {
        // Push in reverse order so reversal emits forward UTF-8 bytes
        for (int bi = cur - 1; bi >= pcur; bi--) {
          rev[count++] = ug->byte_ids[(unsigned char)norm[bi]];
        }
      } else {
        rev[count++] = ug->unk_id;
      }
    } else {
      rev[count++] = tok[cur];
    }
    cur = pcur;
  }

  int out_n = 0;
  for (int i = count - 1; i >= 0 && out_n < max_out - 1; i--) {
    // Collapse consecutive <unk> tokens like SentencePiece
    if (out_n > 0 && rev[i] == ug->unk_id && out_ids[out_n - 1] == ug->unk_id) continue;
    out_ids[out_n++] = rev[i];
  }
  out_ids[out_n++] = ug->eos_id; // append </s> = 1

  free(rev); free(dp); free(prev); free(tok); free(norm);
  return out_n;
}
