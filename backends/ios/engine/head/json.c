// head/json.c — see json.h. Strict RFC 8259 parser (as nlohmann's lexer: no comments, no trailing
// commas, control characters must be escaped, strings must be valid UTF-8).
#include "json.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- string buffer ---------------------------------------------------------

void sb_putn(sbuf *b, const char *s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t c = b->cap ? b->cap : 256;
    while (c < b->n + n + 1) c *= 2;
    b->p = (char *)realloc(b->p, c);
    b->cap = c;
  }
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = 0;
}
void sb_puts(sbuf *b, const char *s) { sb_putn(b, s, strlen(s)); }
void sb_putc(sbuf *b, char c) { sb_putn(b, &c, 1); }
void sb_printf(sbuf *b, const char *fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
  va_end(ap);
  if (n < (int)sizeof tmp) { sb_putn(b, tmp, (size_t)n); return; }
  char *big = (char *)malloc((size_t)n + 1);
  va_start(ap, fmt);
  vsnprintf(big, (size_t)n + 1, fmt, ap);
  va_end(ap);
  sb_putn(b, big, (size_t)n);
  free(big);
}
char *sb_take(sbuf *b) {
  if (!b->p) sb_putn(b, "", 0);
  char *p = b->p;
  b->p = NULL; b->n = b->cap = 0;
  return p;
}

// ---- parser ----------------------------------------------------------------

typedef struct { const char *p, *end, *start; char *err; size_t errn; int depth; } P;

static int fail(P *s, const char *msg) {
  if (s->err && !s->err[0]) snprintf(s->err, s->errn, "json: %s at byte %td", msg, s->p - s->start);
  return -1;
}
static void ws(P *s) {
  while (s->p < s->end && (*s->p == ' ' || *s->p == '\t' || *s->p == '\n' || *s->p == '\r')) s->p++;
}
static jv *mk(jtype t) { jv *v = (jv *)calloc(1, sizeof(jv)); v->t = t; return v; }

static void put_utf8(sbuf *b, uint32_t c) {
  char o[4];
  if (c < 0x80) { o[0] = (char)c; sb_putn(b, o, 1); }
  else if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 63)); sb_putn(b, o, 2); }
  else if (c < 0x10000) {
    o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 63)); o[2] = (char)(0x80 | (c & 63));
    sb_putn(b, o, 3);
  } else {
    o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 63));
    o[2] = (char)(0x80 | ((c >> 6) & 63)); o[3] = (char)(0x80 | (c & 63));
    sb_putn(b, o, 4);
  }
}
static int hex4(P *s, uint32_t *out) {
  if (s->end - s->p < 4) return -1;
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) {
    char c = s->p[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
    else return -1;
  }
  s->p += 4;
  *out = v;
  return 0;
}
// one UTF-8 sequence starting at s->p (first byte >= 0x80), copied verbatim; -1 if ill-formed
static int utf8_seq(P *s, sbuf *b) {
  const unsigned char *u = (const unsigned char *)s->p;
  size_t left = (size_t)(s->end - s->p);
  int n; uint32_t c, lo;
  if (u[0] >= 0xC2 && u[0] <= 0xDF) { n = 2; c = u[0] & 31; lo = 0x80; }
  else if (u[0] >= 0xE0 && u[0] <= 0xEF) { n = 3; c = u[0] & 15; lo = 0x800; }
  else if (u[0] >= 0xF0 && u[0] <= 0xF4) { n = 4; c = u[0] & 7; lo = 0x10000; }
  else return -1;
  if (left < (size_t)n) return -1;
  for (int i = 1; i < n; i++) {
    if ((u[i] & 0xC0) != 0x80) return -1;
    c = (c << 6) | (u[i] & 63);
  }
  if (c < lo || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return -1;
  sb_putn(b, s->p, (size_t)n);
  s->p += n;
  return 0;
}
static int str_raw(P *s, char **out, size_t *outn) {
  if (s->p >= s->end || *s->p != '"') return fail(s, "expected string");
  s->p++;
  sbuf b = {0};
  for (;;) {
    if (s->p >= s->end) { free(b.p); return fail(s, "unterminated string"); }
    unsigned char c = (unsigned char)*s->p;
    if (c == '"') { s->p++; break; }
    if (c < 0x20) { free(b.p); return fail(s, "control character in string"); }
    if (c == '\\') {
      s->p++;
      if (s->p >= s->end) { free(b.p); return fail(s, "bad escape"); }
      char e = *s->p++;
      switch (e) {
        case '"': sb_putc(&b, '"'); break;
        case '\\': sb_putc(&b, '\\'); break;
        case '/': sb_putc(&b, '/'); break;
        case 'b': sb_putc(&b, '\b'); break;
        case 'f': sb_putc(&b, '\f'); break;
        case 'n': sb_putc(&b, '\n'); break;
        case 'r': sb_putc(&b, '\r'); break;
        case 't': sb_putc(&b, '\t'); break;
        case 'u': {
          uint32_t cp;
          if (hex4(s, &cp)) { free(b.p); return fail(s, "bad \\u escape"); }
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            uint32_t lo;
            if (s->end - s->p < 6 || s->p[0] != '\\' || s->p[1] != 'u') { free(b.p); return fail(s, "lone surrogate"); }
            s->p += 2;
            if (hex4(s, &lo) || lo < 0xDC00 || lo > 0xDFFF) { free(b.p); return fail(s, "bad surrogate pair"); }
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) { free(b.p); return fail(s, "lone surrogate"); }
          put_utf8(&b, cp);
          break;
        }
        default: free(b.p); return fail(s, "bad escape");
      }
      continue;
    }
    if (c < 0x80) { sb_putc(&b, (char)c); s->p++; continue; }
    if (utf8_seq(s, &b)) { free(b.p); return fail(s, "invalid UTF-8 in string"); }
  }
  *outn = b.n;
  *out = sb_take(&b);
  return 0;
}

static jv *value(P *s);

static jv *number(P *s) {
  const char *st = s->p, *q = s->p;
  int is_int = 1;
  if (q < s->end && *q == '-') q++;
  if (q >= s->end) { fail(s, "bad number"); return NULL; }
  if (*q == '0') q++;
  else if (*q >= '1' && *q <= '9') { while (q < s->end && *q >= '0' && *q <= '9') q++; }
  else { fail(s, "bad number"); return NULL; }
  if (q < s->end && *q == '.') {
    is_int = 0; q++;
    if (q >= s->end || *q < '0' || *q > '9') { fail(s, "bad number"); return NULL; }
    while (q < s->end && *q >= '0' && *q <= '9') q++;
  }
  if (q < s->end && (*q == 'e' || *q == 'E')) {
    is_int = 0; q++;
    if (q < s->end && (*q == '+' || *q == '-')) q++;
    if (q >= s->end || *q < '0' || *q > '9') { fail(s, "bad number"); return NULL; }
    while (q < s->end && *q >= '0' && *q <= '9') q++;
  }
  char tmp[512];
  size_t n = (size_t)(q - st);
  if (n >= sizeof tmp) { fail(s, "number too long"); return NULL; }
  memcpy(tmp, st, n); tmp[n] = 0;
  s->p = q;
  jv *v;
  if (is_int) {
    errno = 0;
    if (tmp[0] == '-') {
      long long x = strtoll(tmp, NULL, 10);
      if (errno == 0) { v = mk(JN_INT); v->i = x; return v; }
    } else {
      unsigned long long x = strtoull(tmp, NULL, 10);
      if (errno == 0) { v = mk(JN_UINT); v->u = x; return v; }
    }
  }
  v = mk(JN_DBL);
  v->d = strtod(tmp, NULL);
  if (!isfinite(v->d)) { free(v); fail(s, "number out of range"); return NULL; }
  return v;
}

static int lit(P *s, const char *w) {
  size_t n = strlen(w);
  if ((size_t)(s->end - s->p) < n || memcmp(s->p, w, n)) return -1;
  s->p += n;
  return 0;
}

static void push(jv *v, char *key, jv *item) {
  if (v->t == JN_OBJ) {
    for (size_t i = 0; i < v->n; i++)
      if (!strcmp(v->keys[i], key)) { jv_free(v->items[i]); v->items[i] = item; free(key); return; }
  }
  v->items = (jv **)realloc(v->items, sizeof(jv *) * (v->n + 1));
  v->items[v->n] = item;
  if (v->t == JN_OBJ) {
    v->keys = (char **)realloc(v->keys, sizeof(char *) * (v->n + 1));
    v->keys[v->n] = key;
  }
  v->n++;
}

static jv *value(P *s) {
  ws(s);
  if (s->p >= s->end) { fail(s, "unexpected end"); return NULL; }
  if (++s->depth > 512) { fail(s, "nesting too deep"); return NULL; }
  jv *v = NULL;
  char c = *s->p;
  if (c == '{' || c == '[') {
    int obj = c == '{';
    v = mk(obj ? JN_OBJ : JN_ARR);
    s->p++;
    ws(s);
    if (s->p < s->end && *s->p == (obj ? '}' : ']')) { s->p++; s->depth--; return v; }
    for (;;) {
      char *key = NULL; size_t kn;
      if (obj) {
        ws(s);
        if (str_raw(s, &key, &kn)) { jv_free(v); return NULL; }
        ws(s);
        if (s->p >= s->end || *s->p != ':') { free(key); jv_free(v); fail(s, "expected ':'"); return NULL; }
        s->p++;
      }
      jv *it = value(s);
      if (!it) { free(key); jv_free(v); return NULL; }
      push(v, key, it);
      ws(s);
      if (s->p < s->end && *s->p == ',') { s->p++; continue; }
      if (s->p < s->end && *s->p == (obj ? '}' : ']')) { s->p++; break; }
      jv_free(v); fail(s, obj ? "expected ',' or '}'" : "expected ',' or ']'"); return NULL;
    }
  } else if (c == '"') {
    v = mk(JN_STR);
    if (str_raw(s, &v->s, &v->slen)) { free(v); return NULL; }
  } else if (c == 't' || c == 'f') {
    v = mk(JN_BOOL); v->b = c == 't';
    if (lit(s, v->b ? "true" : "false")) { free(v); fail(s, "bad literal"); return NULL; }
  } else if (c == 'n') {
    v = mk(JN_NULL);
    if (lit(s, "null")) { free(v); fail(s, "bad literal"); return NULL; }
  } else {
    v = number(s);
  }
  s->depth--;
  return v;
}

jv *jv_parse(const char *text, size_t len, char *err, size_t errn) {
  if (err && errn) err[0] = 0;
  P s = {text, text + len, text, err, errn, 0};
  jv *v = value(&s);
  if (!v) return NULL;
  ws(&s);
  if (s.p != s.end) { jv_free(v); fail(&s, "trailing data"); return NULL; }
  return v;
}

void jv_free(jv *v) {
  if (!v) return;
  for (size_t i = 0; i < v->n; i++) {
    jv_free(v->items[i]);
    if (v->keys) free(v->keys[i]);
  }
  free(v->items); free(v->keys); free(v->s); free(v);
}

const jv *jv_get(const jv *o, const char *key) {
  if (!o || o->t != JN_OBJ) return NULL;
  for (size_t i = 0; i < o->n; i++) if (!strcmp(o->keys[i], key)) return o->items[i];
  return NULL;
}

// ---- serializer ------------------------------------------------------------

void jv_dump_str(sbuf *b, const char *s, size_t n) {
  sb_putc(b, '"');
  for (size_t i = 0; i < n; i++) {
    unsigned char c = (unsigned char)s[i];
    switch (c) {
      case '"': sb_puts(b, "\\\""); break;
      case '\\': sb_puts(b, "\\\\"); break;
      case '\b': sb_puts(b, "\\b"); break;
      case '\f': sb_puts(b, "\\f"); break;
      case '\n': sb_puts(b, "\\n"); break;
      case '\r': sb_puts(b, "\\r"); break;
      case '\t': sb_puts(b, "\\t"); break;
      default:
        if (c < 0x20) sb_printf(b, "\\u%04x", c);
        else sb_putc(b, (char)c);
    }
  }
  sb_putc(b, '"');
}

// nlohmann::detail::to_chars: shortest round-trip digits, then format_buffer(min_exp=-4, max_exp=15)
void jv_dump_double(sbuf *b, double d) {
  if (!isfinite(d)) { sb_puts(b, "null"); return; }
  if (d == 0) { sb_puts(b, signbit(d) ? "-0.0" : "0.0"); return; }
  if (d < 0) { sb_putc(b, '-'); d = -d; }
  char e[40];
  for (int p = 1; p <= 17; p++) {
    snprintf(e, sizeof e, "%.*e", p - 1, d);
    if (strtod(e, NULL) == d) break;
  }
  // e = "D[.DDD]e±XX"
  char dig[24]; int k = 0;
  const char *q = e;
  for (; *q && *q != 'e'; q++) if (*q >= '0' && *q <= '9') dig[k++] = *q;
  while (k > 1 && dig[k - 1] == '0') k--;
  int n = atoi(q + 1) + 1;  // position of the decimal point relative to the digit string
  if (k <= n && n <= 15) {
    sb_putn(b, dig, (size_t)k);
    for (int i = k; i < n; i++) sb_putc(b, '0');
    sb_puts(b, ".0");
  } else if (0 < n && n <= 15) {
    sb_putn(b, dig, (size_t)n); sb_putc(b, '.'); sb_putn(b, dig + n, (size_t)(k - n));
  } else if (-4 < n && n <= 0) {
    sb_puts(b, "0.");
    for (int i = 0; i < -n; i++) sb_putc(b, '0');
    sb_putn(b, dig, (size_t)k);
  } else {
    sb_putc(b, dig[0]);
    if (k > 1) { sb_putc(b, '.'); sb_putn(b, dig + 1, (size_t)(k - 1)); }
    int x = n - 1;
    sb_printf(b, "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
  }
}

void jv_dump(sbuf *b, const jv *v) {
  switch (v->t) {
    case JN_NULL: sb_puts(b, "null"); break;
    case JN_BOOL: sb_puts(b, v->b ? "true" : "false"); break;
    case JN_INT: sb_printf(b, "%lld", (long long)v->i); break;
    case JN_UINT: sb_printf(b, "%llu", (unsigned long long)v->u); break;
    case JN_DBL: jv_dump_double(b, v->d); break;
    case JN_STR: jv_dump_str(b, v->s, v->slen); break;
    case JN_ARR:
    case JN_OBJ:
      sb_putc(b, v->t == JN_OBJ ? '{' : '[');
      for (size_t i = 0; i < v->n; i++) {
        if (i) sb_putc(b, ',');
        if (v->t == JN_OBJ) { jv_dump_str(b, v->keys[i], strlen(v->keys[i])); sb_putc(b, ':'); }
        jv_dump(b, v->items[i]);
      }
      sb_putc(b, v->t == JN_OBJ ? '}' : ']');
      break;
  }
}
