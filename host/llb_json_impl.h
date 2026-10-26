// host/llb_json_impl.h — definitions for host/llb_json.h (inlined by the engine, compiled
// standalone via host/llb_json.c).
#ifndef LLB_JSON_IMPL_H
#define LLB_JSON_IMPL_H

#include "llb_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void llb_buf_put(LlbBuf* b, const char* s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t c = b->cap ? b->cap : 256;
    while (c < b->n + n + 1) c *= 2;
    char* p = realloc(b->p, c);
    if (!p) abort();
    b->p = p;
    b->cap = c;
  }
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = 0;
}

void llb_buf_puts(LlbBuf* b, const char* s) { llb_buf_put(b, s, strlen(s)); }

void llb_json_quote(LlbBuf* b, const char* s, size_t n) {
  llb_buf_put(b, "\"", 1);
  for (size_t i = 0; i < n; i += 1) {
    unsigned char c = (unsigned char)s[i];
    char e[8];
    switch (c) {
      case '"': llb_buf_put(b, "\\\"", 2); break;
      case '\\': llb_buf_put(b, "\\\\", 2); break;
      case '\n': llb_buf_put(b, "\\n", 2); break;
      case '\r': llb_buf_put(b, "\\r", 2); break;
      case '\t': llb_buf_put(b, "\\t", 2); break;
      case '\b': llb_buf_put(b, "\\b", 2); break;
      case '\f': llb_buf_put(b, "\\f", 2); break;
      default:
        if (c < 0x20) {
          snprintf(e, sizeof e, "\\u%04x", c);
          llb_buf_put(b, e, 6);
        } else {
          llb_buf_put(b, (const char*)&s[i], 1);
        }
    }
  }
  llb_buf_put(b, "\"", 1);
}

typedef struct {
  const char* p;
  const char* e;
  char* err;
  int errlen;
  int depth;
} LlbJP;

static int llb_jp_fail(LlbJP* P, const char* m) {
  if (P->err && P->errlen > 0 && !P->err[0]) snprintf(P->err, (size_t)P->errlen, "json: %s", m);
  return -1;
}

static void llb_jp_ws(LlbJP* P) {
  while (P->p < P->e && (*P->p == ' ' || *P->p == '\t' || *P->p == '\n' || *P->p == '\r')) P->p += 1;
}

static void llb_utf8_put(LlbBuf* b, unsigned cp) {
  char u[4];
  int n;
  if (cp < 0x80) { u[0] = (char)cp; n = 1; }
  else if (cp < 0x800) { u[0] = (char)(0xC0 | cp >> 6); u[1] = (char)(0x80 | (cp & 63)); n = 2; }
  else if (cp < 0x10000) { u[0] = (char)(0xE0 | cp >> 12); u[1] = (char)(0x80 | (cp >> 6 & 63)); u[2] = (char)(0x80 | (cp & 63)); n = 3; }
  else { u[0] = (char)(0xF0 | cp >> 18); u[1] = (char)(0x80 | (cp >> 12 & 63)); u[2] = (char)(0x80 | (cp >> 6 & 63)); u[3] = (char)(0x80 | (cp & 63)); n = 4; }
  llb_buf_put(b, u, (size_t)n);
}

static int llb_jp_hex4(LlbJP* P, unsigned* v) {
  if (P->e - P->p < 4) return llb_jp_fail(P, "short \\u escape");
  *v = 0;
  for (int i = 0; i < 4; i += 1) {
    char c = P->p[i];
    unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0') : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10) : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 99;
    if (d > 15) return llb_jp_fail(P, "bad \\u escape");
    *v = *v * 16 + d;
  }
  P->p += 4;
  return 0;
}

// the string at P->p (after the opening quote) into b; P->p ends past the closing quote
static int llb_jp_str(LlbJP* P, LlbBuf* b) {
  llb_buf_put(b, "", 0);
  while (P->p < P->e) {
    unsigned char c = (unsigned char)*P->p++;
    if (c == '"') return 0;
    if (c < 0x20) return llb_jp_fail(P, "control character in string");
    if (c != '\\') {
      llb_buf_put(b, (const char*)&c, 1);
      continue;
    }
    if (P->p >= P->e) break;
    char x = *P->p++;
    switch (x) {
      case '"': llb_buf_put(b, "\"", 1); break;
      case '\\': llb_buf_put(b, "\\", 1); break;
      case '/': llb_buf_put(b, "/", 1); break;
      case 'b': llb_buf_put(b, "\b", 1); break;
      case 'f': llb_buf_put(b, "\f", 1); break;
      case 'n': llb_buf_put(b, "\n", 1); break;
      case 'r': llb_buf_put(b, "\r", 1); break;
      case 't': llb_buf_put(b, "\t", 1); break;
      case 'u': {
        unsigned v, w;
        if (llb_jp_hex4(P, &v)) return -1;
        if (v >= 0xD800 && v < 0xDC00) {
          if (P->e - P->p < 6 || P->p[0] != '\\' || P->p[1] != 'u') return llb_jp_fail(P, "lone high surrogate");
          P->p += 2;
          if (llb_jp_hex4(P, &w)) return -1;
          if (w < 0xDC00 || w >= 0xE000) return llb_jp_fail(P, "bad low surrogate");
          v = 0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00);
        } else if (v >= 0xDC00 && v < 0xE000) {
          return llb_jp_fail(P, "lone low surrogate");
        }
        llb_utf8_put(b, v);
        break;
      }
      default: return llb_jp_fail(P, "bad escape");
    }
  }
  return llb_jp_fail(P, "unterminated string");
}

static LlbJson* llb_jp_val(LlbJP* P);

static LlbJson* llb_jnew(LlbJType t) {
  LlbJson* j = calloc(1, sizeof *j);
  if (!j) abort();
  j->t = t;
  return j;
}

static void llb_jadd(LlbJson* j, char* key, LlbJson* kid) {
  j->kids = realloc(j->kids, (j->n + 1) * sizeof *j->kids);
  if (j->t == LLB_J_OBJ) j->keys = realloc(j->keys, (j->n + 1) * sizeof *j->keys);
  if (!j->kids || (j->t == LLB_J_OBJ && !j->keys)) abort();
  if (j->t == LLB_J_OBJ) j->keys[j->n] = key;
  j->kids[j->n++] = kid;
}

static LlbJson* llb_jp_val(LlbJP* P) {
  llb_jp_ws(P);
  if (P->p >= P->e) { llb_jp_fail(P, "unexpected end"); return NULL; }
  if (++P->depth > 512) { llb_jp_fail(P, "nesting too deep"); return NULL; }
  char c = *P->p;
  LlbJson* j = NULL;
  if (c == '{' || c == '[') {
    int obj = c == '{';
    char close = obj ? '}' : ']';
    j = llb_jnew(obj ? LLB_J_OBJ : LLB_J_ARR);
    P->p += 1;
    llb_jp_ws(P);
    if (P->p < P->e && *P->p == close) {
      P->p += 1;
      P->depth -= 1;
      return j;
    }
    for (;;) {
      char* key = NULL;
      if (obj) {
        llb_jp_ws(P);
        if (P->p >= P->e || *P->p != '"') { llb_jp_fail(P, "expected key"); goto bad; }
        P->p += 1;
        LlbBuf kb = {0};
        if (llb_jp_str(P, &kb)) { free(kb.p); goto bad; }
        key = kb.p;
        llb_jp_ws(P);
        if (P->p >= P->e || *P->p != ':') { free(key); llb_jp_fail(P, "expected ':'"); goto bad; }
        P->p += 1;
      }
      LlbJson* v = llb_jp_val(P);
      if (!v) { free(key); goto bad; }
      llb_jadd(j, key, v);
      llb_jp_ws(P);
      if (P->p < P->e && *P->p == ',') { P->p += 1; continue; }
      if (P->p < P->e && *P->p == close) { P->p += 1; break; }
      llb_jp_fail(P, obj ? "expected ',' or '}'" : "expected ',' or ']'");
      goto bad;
    }
  } else if (c == '"') {
    P->p += 1;
    LlbBuf b = {0};
    if (llb_jp_str(P, &b)) { free(b.p); P->depth -= 1; return NULL; }
    j = llb_jnew(LLB_J_STR);
    j->s = b.p;
    j->n = b.n;
  } else if (c == 't' && P->e - P->p >= 4 && !memcmp(P->p, "true", 4)) {
    j = llb_jnew(LLB_J_BOOL); j->b = 1; P->p += 4;
  } else if (c == 'f' && P->e - P->p >= 5 && !memcmp(P->p, "false", 5)) {
    j = llb_jnew(LLB_J_BOOL); P->p += 5;
  } else if (c == 'n' && P->e - P->p >= 4 && !memcmp(P->p, "null", 4)) {
    j = llb_jnew(LLB_J_NULL); P->p += 4;
  } else if (c == '-' || (c >= '0' && c <= '9')) {
    const char* s = P->p;
    if (*P->p == '-') P->p += 1;
    if (P->p < P->e && *P->p == '0') P->p += 1;
    else if (P->p < P->e && *P->p >= '1' && *P->p <= '9') while (P->p < P->e && *P->p >= '0' && *P->p <= '9') P->p += 1;
    else { llb_jp_fail(P, "bad number"); P->depth -= 1; return NULL; }
    if (P->p < P->e && *P->p == '.') {
      P->p += 1;
      if (P->p >= P->e || *P->p < '0' || *P->p > '9') { llb_jp_fail(P, "bad fraction"); P->depth -= 1; return NULL; }
      while (P->p < P->e && *P->p >= '0' && *P->p <= '9') P->p += 1;
    }
    if (P->p < P->e && (*P->p == 'e' || *P->p == 'E')) {
      P->p += 1;
      if (P->p < P->e && (*P->p == '+' || *P->p == '-')) P->p += 1;
      if (P->p >= P->e || *P->p < '0' || *P->p > '9') { llb_jp_fail(P, "bad exponent"); P->depth -= 1; return NULL; }
      while (P->p < P->e && *P->p >= '0' && *P->p <= '9') P->p += 1;
    }
    j = llb_jnew(LLB_J_NUM);
    j->n = (size_t)(P->p - s);
    j->s = malloc(j->n + 1);
    if (!j->s) abort();
    memcpy(j->s, s, j->n);
    j->s[j->n] = 0;
  } else {
    llb_jp_fail(P, "unexpected character");
  }
  P->depth -= 1;
  return j;
bad:
  llb_json_free(j);
  P->depth -= 1;
  return NULL;
}

LlbJson* llb_json_parse(const char* text, size_t len, char* err, int errlen) {
  LlbJP P = {text, text + len, err, errlen, 0};
  if (err && errlen > 0) err[0] = 0;
  LlbJson* j = llb_jp_val(&P);
  if (!j) return NULL;
  llb_jp_ws(&P);
  if (P.p != P.e) {
    llb_jp_fail(&P, "trailing characters");
    llb_json_free(j);
    return NULL;
  }
  return j;
}

void llb_json_free(LlbJson* j) {
  if (!j) return;
  for (size_t i = 0; i < j->n && (j->t == LLB_J_ARR || j->t == LLB_J_OBJ); i += 1) {
    llb_json_free(j->kids[i]);
    if (j->keys) free(j->keys[i]);
  }
  free(j->kids);
  free(j->keys);
  free(j->s);
  free(j);
}

const LlbJson* llb_json_get(const LlbJson* o, const char* key) {
  if (!o || o->t != LLB_J_OBJ) return NULL;
  for (size_t i = o->n; i-- > 0;) {
    if (!strcmp(o->keys[i], key)) return o->kids[i];
  }
  return NULL;
}

double llb_json_num(const LlbJson* j) { return j && j->t == LLB_J_NUM ? strtod(j->s, NULL) : 0.0; }

static void llb_json_dump_to(LlbBuf* b, const LlbJson* j, int compact) {
  switch (j->t) {
    case LLB_J_NULL: llb_buf_puts(b, "null"); break;
    case LLB_J_BOOL: llb_buf_puts(b, j->b ? "true" : "false"); break;
    case LLB_J_NUM: llb_buf_put(b, j->s, j->n); break;
    case LLB_J_STR: llb_json_quote(b, j->s, j->n); break;
    case LLB_J_ARR:
    case LLB_J_OBJ:
      llb_buf_puts(b, j->t == LLB_J_OBJ ? "{" : "[");
      for (size_t i = 0; i < j->n; i += 1) {
        if (i) llb_buf_puts(b, compact ? "," : ", ");
        if (j->t == LLB_J_OBJ) {
          llb_json_quote(b, j->keys[i], strlen(j->keys[i]));
          llb_buf_puts(b, compact ? ":" : ": ");
        }
        llb_json_dump_to(b, j->kids[i], compact);
      }
      llb_buf_puts(b, j->t == LLB_J_OBJ ? "}" : "]");
      break;
  }
}

char* llb_json_dump(const LlbJson* j, int compact) {
  LlbBuf b = {0};
  llb_buf_put(&b, "", 0);
  llb_json_dump_to(&b, j, compact);
  return b.p;
}

#endif
