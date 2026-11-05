#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

void sb_addn(sb *s, const char *t, size_t n) {
    if (s->n + n + 1 > s->cap) { s->cap = (s->n + n + 1) * 2; s->p = realloc(s->p, s->cap); }
    memcpy(s->p + s->n, t, n); s->n += n; s->p[s->n] = 0;
}
void sb_add(sb *s, const char *t) { sb_addn(s, t, strlen(t)); }
void sb_addc(sb *s, char c) { sb_addn(s, &c, 1); }
void sb_fmt(sb *s, const char *fmt, ...) {
    char buf[512]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap); sb_add(s, buf);
}

typedef struct { const char *p; const char *err; } P;
static jv *pv(P *c, unsigned depth);
static void ws(P *c) { while (*c->p == ' ' || *c->p == '\n' || *c->p == '\t' || *c->p == '\r') c->p++; }
static jv *mk(jtype t) { jv *v = calloc(1, sizeof *v); v->type = t; return v; }
static void put_utf8(sb *s, unsigned cp) {
    if (cp < 0x80) sb_addc(s, cp);
    else if (cp < 0x800) { sb_addc(s, 0xC0 | cp >> 6); sb_addc(s, 0x80 | (cp & 63)); }
    else if (cp < 0x10000) { sb_addc(s, 0xE0 | cp >> 12); sb_addc(s, 0x80 | (cp >> 6 & 63)); sb_addc(s, 0x80 | (cp & 63)); }
    else { sb_addc(s, 0xF0 | cp >> 18); sb_addc(s, 0x80 | (cp >> 12 & 63)); sb_addc(s, 0x80 | (cp >> 6 & 63)); sb_addc(s, 0x80 | (cp & 63)); }
}
static int hex4(const char *p, unsigned *o) {
    unsigned v = 0; for (int i = 0; i < 4; i++) { char c = p[i]; v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10; else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else return 0; }
    *o = v; return 1;
}
static char *pstr(P *c) {
    sb s = {0}; c->p++; // opening quote
    for (;;) {
        char ch = *c->p;
        if (!ch) { c->err = "unterminated string"; free(s.p); return NULL; }
        c->p++;
        if (ch == '"') break;
        if ((unsigned char)ch < 0x20) { c->err = "control character in string"; free(s.p); return NULL; }
        if (ch != '\\') { sb_addc(&s, ch); continue; }
        ch = *c->p++;
        switch (ch) {
        case 'n': sb_addc(&s, '\n'); break; case 't': sb_addc(&s, '\t'); break; case 'r': sb_addc(&s, '\r'); break;
        case 'b': sb_addc(&s, '\b'); break; case 'f': sb_addc(&s, '\f'); break;
        case '/': case '\\': case '"': sb_addc(&s, ch); break;
        case 'u': { unsigned cp, lo;
            if (!hex4(c->p, &cp)) { c->err = "bad \\u"; free(s.p); return NULL; }
            c->p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && c->p[0] == '\\' && c->p[1] == 'u' && hex4(c->p + 2, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); c->p += 6; }
            if (!cp) { c->err = "NUL strings unsupported"; free(s.p); return NULL; }
            put_utf8(&s, cp); break; }
        default: c->err = "bad escape"; free(s.p); return NULL;
        }
    }
    if (!s.p) s.p = strdup("");
    return s.p;
}
static jv *pv(P *c, unsigned depth) {
    if (depth > 128) { c->err = "JSON nesting too deep"; return NULL; }
    ws(c);
    char ch = *c->p;
    if (ch == '"') { jv *v = mk(J_STR); v->str = pstr(c); if (!v->str) { free(v); return NULL; } return v; }
    if (ch == '{' || ch == '[') {
        int obj = ch == '{'; jv *v = mk(obj ? J_OBJ : J_ARR); c->p++; ws(c);
        size_t cap = 0;
        if (*c->p == (obj ? '}' : ']')) { c->p++; return v; }
        for (;;) {
            char *k = NULL; ws(c);
            if (obj) { if (*c->p != '"') { c->err = "expected key"; json_free(v); return NULL; }
                k = pstr(c); if (!k) { json_free(v); return NULL; } ws(c);
                if (*c->p++ != ':') { c->err = "expected :"; free(k); json_free(v); return NULL; } }
            jv *e = pv(c, depth + 1); if (!e) { free(k); json_free(v); return NULL; }
            if (v->n == cap) { cap = cap ? cap * 2 : 8; v->items = realloc(v->items, cap * sizeof *v->items); if (obj) v->keys = realloc(v->keys, cap * sizeof *v->keys); }
            v->items[v->n] = e; if (obj) v->keys[v->n] = k; v->n++;
            ws(c);
            if (*c->p == ',') { c->p++; continue; }
            if (*c->p == (obj ? '}' : ']')) { c->p++; return v; }
            c->err = "expected , or close"; json_free(v); return NULL;
        }
    }
    if (!strncmp(c->p, "true", 4))  { c->p += 4; jv *v = mk(J_BOOL); v->b = 1; return v; }
    if (!strncmp(c->p, "false", 5)) { c->p += 5; return mk(J_BOOL); }
    if (!strncmp(c->p, "null", 4))  { c->p += 4; return mk(J_NULL); }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
        const char *s = c->p;
        if (*c->p == '-') c->p++;
        if (*c->p == '0') c->p++;
        else if (*c->p >= '1' && *c->p <= '9') while (*c->p >= '0' && *c->p <= '9') c->p++;
        else { c->err = "invalid number"; return NULL; }
        if (*c->p == '.') { c->p++; if (*c->p < '0' || *c->p > '9') { c->err = "invalid number"; return NULL; } while (*c->p >= '0' && *c->p <= '9') c->p++; }
        if (*c->p == 'e' || *c->p == 'E') { c->p++; if (*c->p == '+' || *c->p == '-') c->p++; if (*c->p < '0' || *c->p > '9') { c->err = "invalid number"; return NULL; } while (*c->p >= '0' && *c->p <= '9') c->p++; }
        jv *v = mk(J_NUM); v->raw = strndup(s, c->p - s); v->num = strtod(v->raw, NULL);
        if (!isfinite(v->num)) { c->err = "number out of range"; json_free(v); return NULL; } return v;
    }
    c->err = "unexpected character"; return NULL;
}
jv *json_parse(const char *s, const char **err) {
    P c = { s, NULL }; jv *v = pv(&c, 0);
    if (v) { ws(&c); if (*c.p) { c.err = "trailing data"; json_free(v); v = NULL; } }
    if (!v && err) *err = c.err ? c.err : "parse error";
    return v;
}
jv *json_get(const jv *o, const char *key) {
    if (!o || o->type != J_OBJ) return NULL;
    for (size_t i = 0; i < o->n; i++) if (!strcmp(o->keys[i], key)) return o->items[i];
    return NULL;
}
void json_free(jv *v) {
    if (!v) return;
    for (size_t i = 0; i < v->n; i++) { json_free(v->items[i]); if (v->keys) free(v->keys[i]); }
    free(v->items); free(v->keys); free(v->str); free(v->raw); free(v);
}
char *json_read_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); if (fread(b, 1, n, f) != (size_t) n) { fclose(f); free(b); return NULL; } b[n] = 0; fclose(f); return b;
}
void json_quote(sb *s, const char *t) {
    sb_addc(s, '"');
    for (; *t; t++) {
        unsigned char c = *t;
        if (c == '"') sb_add(s, "\\\""); else if (c == '\\') sb_add(s, "\\\\");
        else if (c == '\n') sb_add(s, "\\n"); else if (c == '\t') sb_add(s, "\\t"); else if (c == '\r') sb_add(s, "\\r");
        else if (c == '\b') sb_add(s, "\\b"); else if (c == '\f') sb_add(s, "\\f");
        else if (c < 0x20) sb_fmt(s, "\\u%04x", c); else sb_addc(s, c);
    }
    sb_addc(s, '"');
}
// Python repr-compatible shortest decimal (reference renderer); adapted from SYS1A dump() of a number: integers as written, other values as the shortest text that reads back equal
static void num_text(sb *s, const jv *v) {
    const char *r = v->raw ? v->raw : "0";
    if (!strpbrk(r, ".eE")) { sb_add(s, r); return; }
    char b[40]; int p = 1; for (; p < 17; p++) { snprintf(b, sizeof b, "%.*e", p - 1, v->num); if (strtod(b, NULL) == v->num) break; }
    snprintf(b, sizeof b, "%.*e", p - 1, v->num);
    int neg = b[0] == '-'; char dg[24]; int nd = 0, ex = atoi(strchr(b, 'e') + 1);
    for (char *c = b + neg; *c && *c != 'e'; c++) if (*c != '.') dg[nd++] = *c;
    while (nd > 1 && dg[nd - 1] == '0') nd--;
    if (neg) sb_addc(s, '-');
    if (ex >= -4 && ex < 16) {
        if (ex < 0) { sb_add(s, "0."); for (int i = -1; i > ex; i--) sb_addc(s, '0'); sb_addn(s, dg, nd); }
        else { for (int i = 0; i <= ex; i++) sb_addc(s, i < nd ? dg[i] : '0'); sb_addc(s, '.'); if (nd > ex + 1) sb_addn(s, dg + ex + 1, nd - ex - 1); else sb_addc(s, '0'); }
    } else { sb_addc(s, dg[0]); if (nd > 1) { sb_addc(s, '.'); sb_addn(s, dg + 1, nd - 1); } sb_fmt(s, "e%c%02d", ex < 0 ? '-' : '+', abs(ex)); }
}
void json_dump(sb *s, const jv *v) {
    switch (v->type) {
    case J_NULL: sb_add(s, "null"); break;
    case J_BOOL: sb_add(s, v->b ? "true" : "false"); break;
    case J_NUM:  num_text(s, v); break;
    case J_STR:  json_quote(s, v->str); break;
    case J_ARR:  sb_addc(s, '['); for (size_t i = 0; i < v->n; i++) { if (i) sb_addc(s, ','); json_dump(s, v->items[i]); } sb_addc(s, ']'); break;
    case J_OBJ:  sb_addc(s, '{'); for (size_t i = 0; i < v->n; i++) { if (i) sb_addc(s, ','); json_quote(s, v->keys[i]); sb_addc(s, ':'); json_dump(s, v->items[i]); } sb_addc(s, '}'); break;
    }
}
