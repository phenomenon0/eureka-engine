// host/llb_grammar_impl.h — definitions for host/llb_grammar.h (inlined by the engine,
// compiled standalone via host/llb_grammar.c). Plain C11.
#ifndef LLB_GRAMMAR_IMPL_H
#define LLB_GRAMMAR_IMPL_H

#include "llb_grammar.h"
#include "llb_json_impl.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- compiled grammar --------------------------------------------------------------
// elems: every alternative is a run of elements closed by END. TERM = a 256-bit byte set,
// REF = a rule. A position is an index into elems.
#define LLB_GE_END 0
#define LLB_GE_KIND(e) ((e) & 3)
#define LLB_GE_ARG(e) ((e) >> 2)
#define LLB_GE_TERM(i) (((i) << 2) | 1)
#define LLB_GE_REF(r) (((r) << 2) | 2)

typedef struct { uint8_t b[32]; } LlbGSet;

typedef struct { int* e; int n, cap; } LlbGSeq;
typedef struct { char* name; LlbGSeq* alts; int nalt, capalt; int defined; } LlbGRule;

struct LlbGram {
  int* elems;
  int nelems;
  int nrules;
  int* rule_alt0;   // first alternative's index in alt_start
  int* rule_nalt;
  int* alt_start;   // position of each alternative
  LlbGSet* sets;
  int nsets;
  int root;
};

static void* llb_g_grow(void* p, int* cap, int need, size_t sz) {
  if (need <= *cap) return p;
  int c = *cap ? *cap : 8;
  while (c < need) c *= 2;
  p = realloc(p, (size_t)c * sz);
  if (!p) abort();
  *cap = c;
  return p;
}

static void llb_gseq_push(LlbGSeq* s, int e) {
  s->e = llb_g_grow(s->e, &s->cap, s->n + 1, sizeof(int));
  s->e[s->n++] = e;
}

// ---- GBNF parser ---------------------------------------------------------------------
typedef struct {
  const char* p;
  char* err;
  int errlen;
  int failed;
  LlbGRule* rules;
  int nrules, caprules;
  LlbGSet* sets;
  int nsets, capsets;
} LlbGP;

static int llb_gp_fail(LlbGP* P, const char* fmt, ...) {
  if (!P->failed && P->err && P->errlen > 0) {
    va_list ap;
    va_start(ap, fmt);
    int k = snprintf(P->err, (size_t)P->errlen, "grammar: ");
    if (k < P->errlen) vsnprintf(P->err + k, (size_t)(P->errlen - k), fmt, ap);
    va_end(ap);
  }
  P->failed = 1;
  return -1;
}

static int llb_gp_rule(LlbGP* P, const char* name, int len) {
  for (int i = 0; i < P->nrules; i += 1) {
    if ((int)strlen(P->rules[i].name) == len && !memcmp(P->rules[i].name, name, (size_t)len)) return i;
  }
  P->rules = llb_g_grow(P->rules, &P->caprules, P->nrules + 1, sizeof *P->rules);
  LlbGRule* r = &P->rules[P->nrules];
  memset(r, 0, sizeof *r);
  r->name = malloc((size_t)len + 1);
  if (!r->name) abort();
  memcpy(r->name, name, (size_t)len);
  r->name[len] = 0;
  return P->nrules++;
}

static int llb_gp_anon(LlbGP* P, int base) {
  char nm[160];
  int k = 1;
  for (;; k += 1) {
    snprintf(nm, sizeof nm, "%s_%d", P->rules[base].name, k);
    int dup = 0;
    for (int i = 0; i < P->nrules && !dup; i += 1) dup = !strcmp(P->rules[i].name, nm);
    if (!dup) break;
  }
  int r = llb_gp_rule(P, nm, (int)strlen(nm));
  P->rules[r].defined = 1;
  return r;
}

static LlbGSeq* llb_gp_alt(LlbGP* P, int r) {
  LlbGRule* R = &P->rules[r];
  R->alts = llb_g_grow(R->alts, &R->capalt, R->nalt + 1, sizeof *R->alts);
  memset(&R->alts[R->nalt], 0, sizeof R->alts[0]);
  return &R->alts[R->nalt++];
}

static int llb_gp_set(LlbGP* P, const LlbGSet* s) {
  P->sets = llb_g_grow(P->sets, &P->capsets, P->nsets + 1, sizeof *P->sets);
  P->sets[P->nsets] = *s;
  return P->nsets++;
}

static int llb_gp_byte(LlbGP* P, int lo, int hi) {
  LlbGSet s;
  memset(&s, 0, sizeof s);
  for (int c = lo; c <= hi; c += 1) s.b[c >> 3] |= (uint8_t)(1u << (c & 7));
  return llb_gp_set(P, &s);
}

static void llb_gp_space(LlbGP* P, int newline_ok) {
  for (;;) {
    char c = *P->p;
    if (c == ' ' || c == '\t' || (newline_ok && (c == '\n' || c == '\r'))) P->p += 1;
    else if (c == '#') { while (*P->p && *P->p != '\n' && *P->p != '\r') P->p += 1; }
    else break;
  }
}

static int llb_gp_is_word(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

static int llb_gp_hex(LlbGP* P, int n, unsigned* v) {
  *v = 0;
  for (int i = 0; i < n; i += 1) {
    char c = *P->p;
    unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0') : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10) : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 99;
    if (d > 15) return llb_gp_fail(P, "bad hex escape");
    *v = *v * 16 + d;
    P->p += 1;
  }
  return 0;
}

// one (possibly escaped) character as a code point
static int llb_gp_char(LlbGP* P, unsigned* cp) {
  const unsigned char* s = (const unsigned char*)P->p;
  if (!*s) return llb_gp_fail(P, "unexpected end of input");
  if (*s == '\\') {
    P->p += 2;
    switch (s[1]) {
      case 'x': return llb_gp_hex(P, 2, cp);
      case 'u': return llb_gp_hex(P, 4, cp);
      case 'U': return llb_gp_hex(P, 8, cp);
      case 't': *cp = '\t'; return 0;
      case 'r': *cp = '\r'; return 0;
      case 'n': *cp = '\n'; return 0;
      case '\\': case '"': case '[': case ']': *cp = s[1]; return 0;
      default: return llb_gp_fail(P, "unknown escape \\%c", s[1] ? s[1] : '0');
    }
  }
  int n = *s < 0x80 ? 1 : (*s >> 5) == 6 ? 2 : (*s >> 4) == 14 ? 3 : (*s >> 3) == 30 ? 4 : 0;
  if (!n) return llb_gp_fail(P, "bad UTF-8 in grammar");
  unsigned v = n == 1 ? *s : n == 2 ? (*s & 31u) : n == 3 ? (*s & 15u) : (*s & 7u);
  for (int i = 1; i < n; i += 1) {
    if ((s[i] & 0xC0) != 0x80) return llb_gp_fail(P, "bad UTF-8 in grammar");
    v = v << 6 | (s[i] & 63u);
  }
  P->p += n;
  *cp = v;
  return 0;
}

static int llb_u8enc(unsigned cp, uint8_t* u) {
  if (cp < 0x80) { u[0] = (uint8_t)cp; return 1; }
  if (cp < 0x800) { u[0] = (uint8_t)(0xC0 | cp >> 6); u[1] = (uint8_t)(0x80 | (cp & 63)); return 2; }
  if (cp < 0x10000) { u[0] = (uint8_t)(0xE0 | cp >> 12); u[1] = (uint8_t)(0x80 | (cp >> 6 & 63)); u[2] = (uint8_t)(0x80 | (cp & 63)); return 3; }
  u[0] = (uint8_t)(0xF0 | cp >> 18); u[1] = (uint8_t)(0x80 | (cp >> 12 & 63)); u[2] = (uint8_t)(0x80 | (cp >> 6 & 63)); u[3] = (uint8_t)(0x80 | (cp & 63));
  return 4;
}

// code-point range [lo, hi] (no surrogates, one encoded length after the split) -> the
// UTF-8 byte-range sequences covering exactly it; each sequence becomes one alternative
static void llb_gp_u8seq(LlbGP* P, int rule, unsigned lo, unsigned hi) {
  static const unsigned mx[3] = {0x7F, 0x7FF, 0xFFFF};
  if (lo > hi) return;
  for (int i = 0; i < 3; i += 1) {
    if (lo <= mx[i] && hi > mx[i]) {
      llb_gp_u8seq(P, rule, lo, mx[i]);
      llb_gp_u8seq(P, rule, mx[i] + 1, hi);
      return;
    }
  }
  for (int i = 1; i < 4; i += 1) {
    unsigned m = (1u << (6 * i)) - 1;
    if ((lo & ~m) != (hi & ~m)) {
      if ((lo & m) != 0) {
        llb_gp_u8seq(P, rule, lo, lo | m);
        llb_gp_u8seq(P, rule, (lo | m) + 1, hi);
        return;
      }
      if ((hi & m) != m) {
        llb_gp_u8seq(P, rule, lo, (hi & ~m) - 1);
        llb_gp_u8seq(P, rule, hi & ~m, hi);
        return;
      }
    }
  }
  uint8_t a[4], b[4];
  int n = llb_u8enc(lo, a);
  llb_u8enc(hi, b);
  LlbGSeq* alt = llb_gp_alt(P, rule);
  for (int k = 0; k < n; k += 1) llb_gseq_push(alt, LLB_GE_TERM(llb_gp_byte(P, a[k], b[k])));
}

typedef struct { unsigned lo, hi; } LlbGRange;

static int llb_gp_rcmp(const void* a, const void* b) {
  const LlbGRange* x = a;
  const LlbGRange* y = b;
  return x->lo < y->lo ? -1 : x->lo > y->lo;
}

// a code-point class -> one TERM (all ASCII) or a REF to a rule of byte sequences
static void llb_gp_class(LlbGP* P, LlbGSeq* out, int base, LlbGRange* r, int n, int neg) {
  qsort(r, (size_t)n, sizeof *r, llb_gp_rcmp);
  int m = 0;
  for (int i = 0; i < n; i += 1) {
    if (r[i].lo > r[i].hi) continue;
    if (m && r[i].lo <= r[m - 1].hi + 1) { if (r[i].hi > r[m - 1].hi) r[m - 1].hi = r[i].hi; }
    else r[m++] = r[i];
  }
  LlbGRange* c = malloc(sizeof *c * (size_t)(m + 3));
  if (!c) abort();
  int k = 0;
  if (neg) {
    unsigned next = 0;
    for (int i = 0; i < m; i += 1) {
      if (r[i].lo > next) c[k++] = (LlbGRange){next, r[i].lo - 1};
      next = r[i].hi + 1;
    }
    if (next <= 0x10FFFF) c[k++] = (LlbGRange){next, 0x10FFFF};
  } else {
    for (int i = 0; i < m; i += 1) c[k++] = r[i];
  }
  // drop surrogates, clamp to Unicode
  LlbGRange* d = malloc(sizeof *d * (size_t)(2 * k + 1));
  if (!d) abort();
  int nd = 0;
  for (int i = 0; i < k; i += 1) {
    unsigned lo = c[i].lo, hi = c[i].hi > 0x10FFFF ? 0x10FFFF : c[i].hi;
    if (lo > hi) continue;
    if (lo < 0xD800 && hi >= 0xD800) { d[nd++] = (LlbGRange){lo, 0xD7FF}; lo = 0xE000; }
    else if (lo >= 0xD800 && lo <= 0xDFFF) lo = 0xE000;
    if (lo <= hi) d[nd++] = (LlbGRange){lo, hi};
  }
  free(c);
  LlbGSet ascii;
  memset(&ascii, 0, sizeof ascii);
  int has_ascii = 0, has_multi = 0;
  for (int i = 0; i < nd; i += 1) {
    for (unsigned x = d[i].lo; x <= d[i].hi && x < 0x80; x += 1) { ascii.b[x >> 3] |= (uint8_t)(1u << (x & 7)); has_ascii = 1; }
    if (d[i].hi >= 0x80) has_multi = 1;
  }
  if (!has_multi) {
    llb_gseq_push(out, LLB_GE_TERM(llb_gp_set(P, &ascii)));   // an empty class never matches
  } else {
    int rr = llb_gp_anon(P, base);
    if (has_ascii) llb_gseq_push(llb_gp_alt(P, rr), LLB_GE_TERM(llb_gp_set(P, &ascii)));
    for (int i = 0; i < nd; i += 1) {
      if (d[i].hi >= 0x80) llb_gp_u8seq(P, rr, d[i].lo < 0x80 ? 0x80 : d[i].lo, d[i].hi);
    }
    llb_gseq_push(out, LLB_GE_REF(rr));
  }
  free(d);
}

static int llb_gp_alternates(LlbGP* P, int rule, int nested);

// x{min,max} (max < 0 = unbounded) on out->e[start..n), llama.cpp's expansion
static void llb_gp_repeat(LlbGP* P, LlbGSeq* out, int start, int base, int mn, int mx) {
  int unit;
  if (out->n - start == 1 && LLB_GE_KIND(out->e[start]) != LLB_GE_END) {
    unit = out->e[start];
  } else {
    int u = llb_gp_anon(P, base);
    LlbGSeq* a = llb_gp_alt(P, u);
    for (int i = start; i < out->n; i += 1) llb_gseq_push(a, out->e[i]);
    unit = LLB_GE_REF(u);
  }
  out->n = start;
  for (int i = 0; i < mn; i += 1) llb_gseq_push(out, unit);
  if (mx < 0) {
    int r = llb_gp_anon(P, base);
    LlbGSeq* a = llb_gp_alt(P, r);   // r ::= unit r | ""
    llb_gseq_push(a, unit);
    llb_gseq_push(a, LLB_GE_REF(r));
    llb_gp_alt(P, r);
    llb_gseq_push(out, LLB_GE_REF(r));
    return;
  }
  int last = -1;
  for (int i = mn; i < mx; i += 1) {   // r_k ::= unit r_{k-1}? | ""
    int r = llb_gp_anon(P, base);
    LlbGSeq* a = llb_gp_alt(P, r);
    llb_gseq_push(a, unit);
    if (last >= 0) llb_gseq_push(a, LLB_GE_REF(last));
    llb_gp_alt(P, r);
    last = r;
  }
  if (last >= 0) llb_gseq_push(out, LLB_GE_REF(last));
}

static int llb_gp_int(LlbGP* P, int* v) {
  if (*P->p < '0' || *P->p > '9') return llb_gp_fail(P, "expected a number in {}");
  long x = 0;
  while (*P->p >= '0' && *P->p <= '9') { x = x * 10 + (*P->p - '0'); P->p += 1; if (x > 100000) return llb_gp_fail(P, "repetition count too large"); }
  *v = (int)x;
  return 0;
}

static int llb_gp_sequence(LlbGP* P, int rule, LlbGSeq* out, int nested) {
  int last = -1;
  while (*P->p && !P->failed) {
    char c = *P->p;
    if (c == '"') {
      P->p += 1;
      last = out->n;
      while (*P->p != '"') {
        unsigned cp;
        if (!*P->p) return llb_gp_fail(P, "unterminated literal");
        if (llb_gp_char(P, &cp)) return -1;
        uint8_t u[4];
        int k = llb_u8enc(cp, u);
        for (int i = 0; i < k; i += 1) llb_gseq_push(out, LLB_GE_TERM(llb_gp_byte(P, u[i], u[i])));
      }
      P->p += 1;
    } else if (c == '[' || c == '.') {
      LlbGRange* rg = NULL;
      int nr = 0, cap = 0, neg = 0;
      if (c == '.') {
        neg = 1;   // any character = the complement of nothing
        P->p += 1;
      } else {
        P->p += 1;
        if (*P->p == '^') { neg = 1; P->p += 1; }
        while (*P->p != ']') {
          unsigned lo, hi;
          if (!*P->p) { free(rg); return llb_gp_fail(P, "unterminated character class"); }
          if (llb_gp_char(P, &lo)) { free(rg); return -1; }
          hi = lo;
          if (P->p[0] == '-' && P->p[1] && P->p[1] != ']') {
            P->p += 1;
            if (llb_gp_char(P, &hi)) { free(rg); return -1; }
          }
          rg = llb_g_grow(rg, &cap, nr + 1, sizeof *rg);
          rg[nr++] = (LlbGRange){lo, hi};
        }
        P->p += 1;
      }
      last = out->n;
      llb_gp_class(P, out, rule, rg, nr, neg);
      free(rg);
    } else if (llb_gp_is_word(c)) {
      const char* s = P->p;
      while (llb_gp_is_word(*P->p)) P->p += 1;
      last = out->n;
      llb_gseq_push(out, LLB_GE_REF(llb_gp_rule(P, s, (int)(P->p - s))));
    } else if (c == '(') {
      P->p += 1;
      llb_gp_space(P, 1);
      int r = llb_gp_anon(P, rule);
      if (llb_gp_alternates(P, r, 1)) return -1;
      if (*P->p != ')') return llb_gp_fail(P, "expected ')'");
      P->p += 1;
      last = out->n;
      llb_gseq_push(out, LLB_GE_REF(r));
    } else if (c == '<') {
      return llb_gp_fail(P, "token references <[id]> are not supported");
    } else if (c == '*' || c == '+' || c == '?' || c == '{') {
      if (last < 0) return llb_gp_fail(P, "repetition without a preceding item");
      int mn = 0, mx = -1;
      P->p += 1;
      if (c == '+') mn = 1;
      else if (c == '?') mx = 1;
      else if (c == '{') {
        llb_gp_space(P, 0);
        if (llb_gp_int(P, &mn)) return -1;
        llb_gp_space(P, 0);
        mx = mn;
        if (*P->p == ',') {
          P->p += 1;
          llb_gp_space(P, 0);
          mx = -1;
          if (*P->p >= '0' && *P->p <= '9' && llb_gp_int(P, &mx)) return -1;
          llb_gp_space(P, 0);
        }
        if (*P->p != '}') return llb_gp_fail(P, "expected '}'");
        P->p += 1;
        if (mx >= 0 && mx < mn) return llb_gp_fail(P, "bad repetition {%d,%d}", mn, mx);
      }
      llb_gp_repeat(P, out, last, rule, mn, mx);
      last = -1;
    } else {
      break;
    }
    llb_gp_space(P, nested);
  }
  return P->failed ? -1 : 0;
}

static int llb_gp_alternates(LlbGP* P, int rule, int nested) {
  for (;;) {
    LlbGSeq* alt = llb_gp_alt(P, rule);
    // alt may move when sequence parsing adds alternatives to other rules (not this one),
    // but this rule's alts array is only grown here, so the pointer stays valid
    LlbGSeq tmp = *alt;
    int ai = P->rules[rule].nalt - 1;
    memset(alt, 0, sizeof *alt);
    if (llb_gp_sequence(P, rule, &tmp, nested)) { free(tmp.e); return -1; }
    P->rules[rule].alts[ai] = tmp;
    if (*P->p != '|') break;
    P->p += 1;
    llb_gp_space(P, 1);
  }
  return 0;
}

static void llb_gp_free(LlbGP* P) {
  for (int i = 0; i < P->nrules; i += 1) {
    for (int a = 0; a < P->rules[i].nalt; a += 1) free(P->rules[i].alts[a].e);
    free(P->rules[i].alts);
    free(P->rules[i].name);
  }
  free(P->rules);
  free(P->sets);
}

void llb_gram_free(LlbGram* g) {
  if (!g) return;
  free(g->elems);
  free(g->rule_alt0);
  free(g->rule_nalt);
  free(g->alt_start);
  free(g->sets);
  free(g);
}

// left recursion: rule r reaches itself through leftmost, nullable-prefixed references
static int llb_gram_leftrec(const LlbGram* g, char** names, char* err, int errlen) {
  int n = g->nrules;
  uint8_t* nul = calloc((size_t)n, 1);
  uint8_t* adj = calloc((size_t)n * (size_t)n, 1);
  if (!nul || !adj) abort();
  for (int ch = 1; ch;) {
    ch = 0;
    for (int r = 0; r < n; r += 1) {
      if (nul[r]) continue;
      for (int a = 0; a < g->rule_nalt[r] && !nul[r]; a += 1) {
        int p = g->alt_start[g->rule_alt0[r] + a], ok = 1;
        for (; g->elems[p] != LLB_GE_END && ok; p += 1) {
          int e = g->elems[p];
          ok = LLB_GE_KIND(e) == 2 && nul[LLB_GE_ARG(e)];
        }
        if (ok) { nul[r] = 1; ch = 1; }
      }
    }
  }
  for (int r = 0; r < n; r += 1) {
    for (int a = 0; a < g->rule_nalt[r]; a += 1) {
      for (int p = g->alt_start[g->rule_alt0[r] + a]; g->elems[p] != LLB_GE_END; p += 1) {
        int e = g->elems[p];
        if (LLB_GE_KIND(e) != 2) break;
        adj[(size_t)r * (size_t)n + (size_t)LLB_GE_ARG(e)] = 1;
        if (!nul[LLB_GE_ARG(e)]) break;
      }
    }
  }
  // transitive closure (rule counts are small)
  for (int k = 0; k < n; k += 1)
    for (int i = 0; i < n; i += 1)
      if (adj[(size_t)i * (size_t)n + (size_t)k])
        for (int j = 0; j < n; j += 1)
          if (adj[(size_t)k * (size_t)n + (size_t)j]) adj[(size_t)i * (size_t)n + (size_t)j] = 1;
  int bad = -1;
  for (int r = 0; r < n && bad < 0; r += 1)
    if (adj[(size_t)r * (size_t)n + (size_t)r]) bad = r;
  if (bad >= 0 && err) snprintf(err, (size_t)errlen, "grammar: left recursion in rule '%s'", names[bad]);
  free(nul);
  free(adj);
  return bad >= 0 ? -1 : 0;
}

LlbGram* llb_gram_from_gbnf(const char* src, const char* root, char* err, int errlen) {
  LlbGP P;
  memset(&P, 0, sizeof P);
  P.p = src;
  P.err = err;
  P.errlen = errlen;
  if (err && errlen > 0) err[0] = 0;
  llb_gp_space(&P, 1);
  while (*P.p && !P.failed) {
    const char* s = P.p;
    while (llb_gp_is_word(*P.p)) P.p += 1;
    if (P.p == s) { llb_gp_fail(&P, "expected a rule name near '%.20s'", s); break; }
    int r = llb_gp_rule(&P, s, (int)(P.p - s));
    if (P.rules[r].defined) { llb_gp_fail(&P, "rule '%s' defined twice", P.rules[r].name); break; }
    P.rules[r].defined = 1;
    llb_gp_space(&P, 0);
    if (strncmp(P.p, "::=", 3)) { llb_gp_fail(&P, "expected '::=' after '%s'", P.rules[r].name); break; }
    P.p += 3;
    llb_gp_space(&P, 1);
    if (llb_gp_alternates(&P, r, 0)) break;
    if (*P.p && *P.p != '\n' && *P.p != '\r') { llb_gp_fail(&P, "unexpected '%.20s'", P.p); break; }
    llb_gp_space(&P, 1);
  }
  const char* rn = root ? root : "root";
  int ri = -1;
  for (int i = 0; i < P.nrules && !P.failed; i += 1) {
    if (!P.rules[i].defined) llb_gp_fail(&P, "undefined rule '%s'", P.rules[i].name);
    if (!strcmp(P.rules[i].name, rn)) ri = i;
  }
  if (!P.failed && ri < 0) llb_gp_fail(&P, "no rule '%s'", rn);
  if (P.failed) { llb_gp_free(&P); return NULL; }

  LlbGram* g = calloc(1, sizeof *g);
  if (!g) abort();
  int nalt = 0, nel = 0;
  for (int i = 0; i < P.nrules; i += 1) {
    nalt += P.rules[i].nalt;
    for (int a = 0; a < P.rules[i].nalt; a += 1) nel += P.rules[i].alts[a].n + 1;
  }
  g->nrules = P.nrules;
  g->rule_alt0 = malloc(sizeof(int) * (size_t)P.nrules);
  g->rule_nalt = malloc(sizeof(int) * (size_t)P.nrules);
  g->alt_start = malloc(sizeof(int) * (size_t)(nalt + 1));
  g->elems = malloc(sizeof(int) * (size_t)(nel + 1));
  if (!g->rule_alt0 || !g->rule_nalt || !g->alt_start || !g->elems) abort();
  int ai = 0, ei = 0;
  for (int i = 0; i < P.nrules; i += 1) {
    g->rule_alt0[i] = ai;
    g->rule_nalt[i] = P.rules[i].nalt;
    for (int a = 0; a < P.rules[i].nalt; a += 1) {
      g->alt_start[ai++] = ei;
      for (int k = 0; k < P.rules[i].alts[a].n; k += 1) g->elems[ei++] = P.rules[i].alts[a].e[k];
      g->elems[ei++] = LLB_GE_END;
    }
  }
  g->nelems = ei;
  g->sets = P.sets;
  g->nsets = P.nsets;
  P.sets = NULL;
  g->root = ri;
  char** names = malloc(sizeof(char*) * (size_t)P.nrules);
  if (!names) abort();
  for (int i = 0; i < P.nrules; i += 1) names[i] = P.rules[i].name;
  int lr = llb_gram_leftrec(g, names, err, errlen);
  free(names);
  llb_gp_free(&P);
  if (lr) { llb_gram_free(g); return NULL; }
  return g;
}

// ---- vocabulary + byte trie ---------------------------------------------------------
struct LlbGVocab {
  int n;
  unsigned char* buf;
  size_t* off;
  int* len;
  uint8_t* eog;
  // trie in DFS preorder; node 0 = root
  int tn;
  uint8_t* tbyte;
  int* tpar;
  int* tskip;    // first node after this node's subtree
  int* ttok0;    // tokens ending here: ttoks[ttok0 .. ttok0 + ttokn)
  int* ttokn;
  int* ttoks;
};

static const LlbGVocab* llb_gv_sort_v;

static int llb_gv_cmp(const void* a, const void* b) {
  const LlbGVocab* v = llb_gv_sort_v;
  int x = *(const int*)a, y = *(const int*)b;
  int lx = v->len[x], ly = v->len[y];
  int c = memcmp(v->buf + v->off[x], v->buf + v->off[y], (size_t)(lx < ly ? lx : ly));
  if (c) return c;
  if (lx != ly) return lx < ly ? -1 : 1;
  return x < y ? -1 : x > y;
}

static void llb_gv_build_trie(LlbGVocab* v) {
  int* ids = malloc(sizeof(int) * (size_t)(v->n + 1));
  if (!ids) abort();
  int m = 0;
  for (int i = 0; i < v->n; i += 1) {
    if (!v->eog[i] && v->len[i] > 0 && v->buf[v->off[i]] != 0) ids[m++] = i;
  }
  llb_gv_sort_v = v;
  qsort(ids, (size_t)m, sizeof(int), llb_gv_cmp);
  size_t total = 1;
  for (int i = 0; i < m; i += 1) total += (size_t)v->len[ids[i]];
  v->tbyte = malloc(total);
  v->tpar = malloc(sizeof(int) * total);
  v->tskip = malloc(sizeof(int) * total);
  v->ttok0 = malloc(sizeof(int) * total);
  v->ttokn = calloc(total, sizeof(int));
  v->ttoks = ids;
  int* path = malloc(sizeof(int) * 65538);
  if (!v->tbyte || !v->tpar || !v->tskip || !v->ttok0 || !v->ttokn || !path) abort();
  v->tn = 1;
  v->tbyte[0] = 0;
  v->tpar[0] = -1;
  v->ttok0[0] = 0;
  int depth = 0;   // path[0..depth] = the current root..node chain
  path[0] = 0;
  const unsigned char* prev = NULL;
  int prevlen = 0;
  for (int k = 0; k < m; k += 1) {
    int t = ids[k];
    const unsigned char* s = v->buf + v->off[t];
    int l = v->len[t], lcp = 0;
    while (prev && lcp < l && lcp < prevlen && lcp < depth && s[lcp] == prev[lcp]) lcp += 1;
    while (depth > lcp) { v->tskip[path[depth]] = v->tn; depth -= 1; }
    while (depth < l) {
      int nd = v->tn++;
      v->tbyte[nd] = s[depth];
      v->tpar[nd] = path[depth];
      v->ttok0[nd] = k;
      v->ttokn[nd] = 0;
      path[++depth] = nd;
    }
    int at = path[depth];
    if (!v->ttokn[at]) v->ttok0[at] = k;
    v->ttokn[at] += 1;
    prev = s;
    prevlen = l;
  }
  while (depth > 0) { v->tskip[path[depth]] = v->tn; depth -= 1; }
  v->tskip[0] = v->tn;
  free(path);
}

static LlbGVocab* llb_gv_alloc(int n) {
  LlbGVocab* v = calloc(1, sizeof *v);
  if (!v) abort();
  v->n = n;
  v->off = malloc(sizeof(size_t) * (size_t)(n + 1));
  v->len = malloc(sizeof(int) * (size_t)(n + 1));
  v->eog = calloc((size_t)n + 1, 1);
  if (!v->off || !v->len || !v->eog) abort();
  return v;
}

LlbGVocab* llb_gv_new(int n, const unsigned char* const* bytes, const int* lens) {
  LlbGVocab* v = llb_gv_alloc(n);
  size_t tot = 0;
  for (int i = 0; i < n; i += 1) tot += (size_t)lens[i];
  v->buf = malloc(tot + 1);
  if (!v->buf) abort();
  size_t o = 0;
  for (int i = 0; i < n; i += 1) {
    v->off[i] = o;
    v->len[i] = lens[i];
    memcpy(v->buf + o, bytes[i], (size_t)lens[i]);
    o += (size_t)lens[i];
  }
  llb_gv_build_trie(v);
  return v;
}

static void llb_gv_free_trie(LlbGVocab* v) {
  free(v->tbyte); free(v->tpar); free(v->tskip); free(v->ttok0); free(v->ttokn); free(v->ttoks);
  v->tbyte = NULL; v->tpar = v->tskip = v->ttok0 = v->ttokn = v->ttoks = NULL;
}

void llb_gv_set_eog(LlbGVocab* v, const int* ids, int n) {
  memset(v->eog, 0, (size_t)v->n);
  for (int i = 0; i < n; i += 1) if (ids[i] >= 0 && ids[i] < v->n) v->eog[ids[i]] = 1;
  llb_gv_free_trie(v);
  llb_gv_build_trie(v);
}

int llb_gv_n_vocab(const LlbGVocab* v) { return v->n; }

int llb_gv_piece(const LlbGVocab* v, int id, const unsigned char** bytes) {
  if (id < 0 || id >= v->n) { *bytes = NULL; return 0; }
  *bytes = v->buf + v->off[id];
  return v->len[id];
}

void llb_gv_free(LlbGVocab* v) {
  if (!v) return;
  llb_gv_free_trie(v);
  free(v->buf); free(v->off); free(v->len); free(v->eog);
  free(v);
}

// GPT-2 byte-level map: the code point a byte is written as in the vocab -> the byte
static int llb_gv_unbyte(unsigned cp) {
  if ((cp >= '!' && cp <= '~') || (cp >= 0xA1 && cp <= 0xAC) || (cp >= 0xAE && cp <= 0xFF)) return (int)cp;
  if (cp < 256 || cp >= 256 + 68) return -1;
  int n = 0;
  for (int b = 0; b < 256; b += 1) {
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) continue;
    if (n++ == (int)(cp - 256)) return b;
  }
  return -1;
}

LlbGVocab* llb_gv_load(const char* path, char* err, int errlen) {
  FILE* f = fopen(path, "rb");
  if (!f) { snprintf(err, (size_t)errlen, "grammar: cannot open %s", path); return NULL; }
  char magic[8];
  uint32_t h[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "Q35TOK01", 8) || fread(h, 4, 8, f) != 8) {
    fclose(f);
    snprintf(err, (size_t)errlen, "grammar: %s is not a Q35TOK01 tokenizer.bin", path);
    return NULL;
  }
  int n = (int)h[0];   // n_vocab n_merges max_len bos eos pad add_bos n_special
  if (fseek(f, (long)h[7] * 4, SEEK_CUR)) { fclose(f); snprintf(err, (size_t)errlen, "grammar: short tokenizer.bin"); return NULL; }
  LlbGVocab* v = llb_gv_alloc(n);
  size_t cap = (size_t)n * 8 + 64, o = 0;
  v->buf = malloc(cap);
  unsigned char* tmp = malloc(65536);
  if (!v->buf || !tmp) abort();
  int bad = 0;
  for (int i = 0; i < n && !bad; i += 1) {
    uint16_t l;
    uint8_t ty;
    if (fread(&l, 2, 1, f) != 1 || fread(&ty, 1, 1, f) != 1 || fread(tmp, 1, l, f) != l) { bad = 1; break; }
    if (o + l + 1 > cap) {
      while (o + l + 1 > cap) cap *= 2;
      v->buf = realloc(v->buf, cap);
      if (!v->buf) abort();
    }
    v->off[i] = o;
    if (ty == 1) {   // normal: byte-level BPE text -> raw bytes (as q35_tok_piece / llama)
      int j = 0;
      while (j < l) {
        unsigned char c = tmp[j];
        int k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        unsigned cp = k == 1 ? c : k == 2 ? (c & 31u) : k == 3 ? (c & 15u) : (c & 7u);
        for (int q = 1; q < k && j + q < l; q += 1) cp = cp << 6 | (tmp[j + q] & 63u);
        int b = llb_gv_unbyte(cp);
        if (b >= 0) v->buf[o++] = (unsigned char)b;
        j += k;
      }
    } else if (ty == 2 || ty == 3 || ty == 4) {   // unknown / control / user-defined: raw text
      memcpy(v->buf + o, tmp, l);
      o += l;
    }   // unused / byte: no piece (never allowed by a grammar)
    v->len[i] = (int)(o - v->off[i]);
  }
  free(tmp);
  fclose(f);
  if (bad) { llb_gv_free(v); snprintf(err, (size_t)errlen, "grammar: short tokenizer.bin"); return NULL; }
  int eog[2], ne = 0;
  static const char* names[2] = {"<|im_end|>", "<|endoftext|>"};
  for (int k = 0; k < 2; k += 1) {
    for (int i = n - 1; i >= 0; i -= 1) {
      if (v->len[i] == (int)strlen(names[k]) && !memcmp(v->buf + v->off[i], names[k], (size_t)v->len[i])) { eog[ne++] = i; break; }
    }
  }
  for (int i = 0; i < ne; i += 1) v->eog[eog[i]] = 1;
  llb_gv_build_trie(v);
  return v;
}

// ---- the matcher -----------------------------------------------------------------------
// Node = one stack: (position, parent node); node 0 = the empty stack (the grammar ended).
// Set = a sorted array of nodes (each a normalised stack: its top is a TERM, or it is 0);
// set 0 = the empty set (dead).
struct LlbGMatch {
  const LlbGram* g;
  const LlbGVocab* v;
  int *npos, *npar, nn, ncap;
  int* nh;
  uint32_t nhcap;
  int *soff, *slen, ns, scap;
  int* sdata;
  int sdn, sdcap;
  int* sh;
  uint32_t shcap;
  uint8_t* sacc;
  int32_t** trans;
  uint32_t** mcache;
  int* mcache_n;
  int nmcache;
  int *tmp, ntmp, tmpcap;
  int *seen, seencap, stamp;
  int32_t* st;
  int init, cur, done;
};

static uint32_t llb_gm_hash2(uint32_t a, uint32_t b) {
  uint64_t x = ((uint64_t)a << 32 | b) * 0x9E3779B97F4A7C15ull;
  return (uint32_t)(x >> 29) ^ (uint32_t)x;
}

static int llb_gm_node(LlbGMatch* m, int pos, int par) {
  if ((uint32_t)(m->nn * 2 + 2) > m->nhcap) {
    uint32_t c = m->nhcap ? m->nhcap * 2 : 1024;
    free(m->nh);
    m->nh = malloc(sizeof(int) * c);
    if (!m->nh) abort();
    memset(m->nh, -1, sizeof(int) * c);
    m->nhcap = c;
    for (int i = 1; i < m->nn; i += 1) {
      uint32_t h = llb_gm_hash2((uint32_t)m->npos[i], (uint32_t)m->npar[i]) & (c - 1);
      while (m->nh[h] >= 0) h = (h + 1) & (c - 1);
      m->nh[h] = i;
    }
  }
  uint32_t h = llb_gm_hash2((uint32_t)pos, (uint32_t)par) & (m->nhcap - 1);
  while (m->nh[h] >= 0) {
    int i = m->nh[h];
    if (m->npos[i] == pos && m->npar[i] == par) return i;
    h = (h + 1) & (m->nhcap - 1);
  }
  m->npos = llb_g_grow(m->npos, &m->ncap, m->nn + 1, sizeof(int));
  m->npar = realloc(m->npar, sizeof(int) * (size_t)m->ncap);
  if (!m->npar) abort();
  m->npos[m->nn] = pos;
  m->npar[m->nn] = par;
  m->nh[h] = m->nn;
  return m->nn++;
}

static uint32_t llb_gm_hashv(const int* a, int n) {
  uint32_t h = 2166136261u;
  for (int i = 0; i < n; i += 1) h = llb_gm_hash2(h, (uint32_t)a[i]);
  return h;
}

static int llb_gm_set(LlbGMatch* m, const int* a, int n) {
  if ((uint32_t)(m->ns * 2 + 2) > m->shcap) {
    uint32_t c = m->shcap ? m->shcap * 2 : 1024;
    free(m->sh);
    m->sh = malloc(sizeof(int) * c);
    if (!m->sh) abort();
    memset(m->sh, -1, sizeof(int) * c);
    m->shcap = c;
    for (int i = 0; i < m->ns; i += 1) {
      uint32_t h = llb_gm_hashv(m->sdata + m->soff[i], m->slen[i]) & (c - 1);
      while (m->sh[h] >= 0) h = (h + 1) & (c - 1);
      m->sh[h] = i;
    }
  }
  uint32_t h = llb_gm_hashv(a, n) & (m->shcap - 1);
  while (m->sh[h] >= 0) {
    int i = m->sh[h];
    if (m->slen[i] == n && !memcmp(m->sdata + m->soff[i], a, sizeof(int) * (size_t)n)) return i;
    h = (h + 1) & (m->shcap - 1);
  }
  int oc = m->scap;
  m->soff = llb_g_grow(m->soff, &m->scap, m->ns + 1, sizeof(int));
  if (m->scap != oc) {
    m->slen = realloc(m->slen, sizeof(int) * (size_t)m->scap);
    m->sacc = realloc(m->sacc, (size_t)m->scap);
    m->trans = realloc(m->trans, sizeof(int32_t*) * (size_t)m->scap);
    m->mcache = realloc(m->mcache, sizeof(uint32_t*) * (size_t)m->scap);
    m->mcache_n = realloc(m->mcache_n, sizeof(int) * (size_t)m->scap);
    if (!m->slen || !m->sacc || !m->trans || !m->mcache || !m->mcache_n) abort();
  }
  m->sdata = llb_g_grow(m->sdata, &m->sdcap, m->sdn + n + 1, sizeof(int));
  if (n) memcpy(m->sdata + m->sdn, a, sizeof(int) * (size_t)n);
  int id = m->ns++;
  m->soff[id] = m->sdn;
  m->slen[id] = n;
  m->sdn += n;
  m->sacc[id] = n > 0 && a[0] == 0;
  m->trans[id] = NULL;
  m->mcache[id] = NULL;
  m->mcache_n[id] = 0;
  m->sh[h] = id;
  return id;
}

static void llb_gm_emit(LlbGMatch* m, int node) {
  if (node >= m->seencap) {
    int oc = m->seencap;
    m->seen = llb_g_grow(m->seen, &m->seencap, node + 1, sizeof(int));
    memset(m->seen + oc, 0, sizeof(int) * (size_t)(m->seencap - oc));
  }
  if (m->seen[node] == m->stamp) return;
  m->seen[node] = m->stamp;
  if (node == 0) {
    m->tmp = llb_g_grow(m->tmp, &m->tmpcap, m->ntmp + 1, sizeof(int));
    m->tmp[m->ntmp++] = 0;
    return;
  }
  int pos = m->npos[node], par = m->npar[node];
  int e = m->g->elems[pos];
  if (e == LLB_GE_END) {
    llb_gm_emit(m, par);
  } else if (LLB_GE_KIND(e) == 1) {
    m->tmp = llb_g_grow(m->tmp, &m->tmpcap, m->ntmp + 1, sizeof(int));
    m->tmp[m->ntmp++] = node;
  } else {
    int r = LLB_GE_ARG(e);
    int ret = m->g->elems[pos + 1] == LLB_GE_END ? par : llb_gm_node(m, pos + 1, par);
    for (int a = 0; a < m->g->rule_nalt[r]; a += 1) {
      llb_gm_emit(m, llb_gm_node(m, m->g->alt_start[m->g->rule_alt0[r] + a], ret));
    }
  }
}

static int llb_gm_icmp(const void* a, const void* b) {
  int x = *(const int*)a, y = *(const int*)b;
  return x < y ? -1 : x > y;
}

static int llb_gm_close(LlbGMatch* m) {
  qsort(m->tmp, (size_t)m->ntmp, sizeof(int), llb_gm_icmp);
  return llb_gm_set(m, m->tmp, m->ntmp);
}

static void llb_gm_start(LlbGMatch* m) {
  m->ntmp = 0;
  m->stamp += 1;
  int r = m->g->root;
  for (int a = 0; a < m->g->rule_nalt[r]; a += 1) llb_gm_emit(m, llb_gm_node(m, m->g->alt_start[m->g->rule_alt0[r] + a], 0));
}

static void llb_gm_tables_init(LlbGMatch* m) {
  m->nn = 0;
  m->ns = 0;
  m->sdn = 0;
  if (m->nh) memset(m->nh, -1, sizeof(int) * m->nhcap);
  if (m->sh) memset(m->sh, -1, sizeof(int) * m->shcap);
  llb_gm_node(m, -1, -1);   // node 0: the empty stack
  llb_gm_set(m, NULL, 0);   // set 0: dead
}

static void llb_gm_tables_drop(LlbGMatch* m) {
  for (int i = 0; i < m->ns; i += 1) {
    free(m->trans[i]);
    free(m->mcache[i]);
  }
  m->nmcache = 0;
}

static int32_t llb_gm_step(LlbGMatch* m, int s, int b) {
  if (s == 0) return 0;
  if (!m->trans[s]) {
    m->trans[s] = malloc(sizeof(int32_t) * 256);
    if (!m->trans[s]) abort();
    for (int i = 0; i < 256; i += 1) m->trans[s][i] = -1;
  }
  int32_t t = m->trans[s][b];
  if (t >= 0) return t;
  m->ntmp = 0;
  m->stamp += 1;
  for (int i = 0; i < m->slen[s]; i += 1) {
    int nd = m->sdata[m->soff[s] + i];
    if (nd == 0) continue;
    int pos = m->npos[nd];
    const LlbGSet* bs = &m->g->sets[LLB_GE_ARG(m->g->elems[pos])];
    if (bs->b[b >> 3] >> (b & 7) & 1) llb_gm_emit(m, llb_gm_node(m, pos + 1, m->npar[nd]));
  }
  t = llb_gm_close(m);
  m->trans[s][b] = t;   // trans[] may have moved: index afresh
  return t;
}

// bound the tables: re-intern only the current set's stacks when they grow too large
static void llb_gm_compact(LlbGMatch* m) {
  if (m->nn < (1 << 21) && m->ns < (1 << 15)) return;
  int n = m->slen[m->cur];
  int total = 0;
  for (int i = 0; i < n; i += 1)
    for (int nd = m->sdata[m->soff[m->cur] + i]; nd > 0; nd = m->npar[nd]) total += 1;
  int* chains = malloc(sizeof(int) * (size_t)(total + n + 1));
  if (!chains) abort();
  int k = 0;
  for (int i = 0; i < n; i += 1) {
    int len = 0, nd = m->sdata[m->soff[m->cur] + i];
    for (int x = nd; x > 0; x = m->npar[x]) len += 1;
    chains[k++] = len;
    int at = k + len;
    for (int x = nd; x > 0; x = m->npar[x]) chains[--at] = m->npos[x];   // bottom first
    k += len;
  }
  llb_gm_tables_drop(m);
  llb_gm_tables_init(m);
  m->ntmp = 0;
  for (int i = 0, q = 0; i < n; i += 1) {
    int len = chains[q++], nd = 0;
    for (int j = 0; j < len; j += 1) nd = llb_gm_node(m, chains[q + j], nd);
    q += len;
    m->tmp = llb_g_grow(m->tmp, &m->tmpcap, m->ntmp + 1, sizeof(int));
    m->tmp[m->ntmp++] = nd;
  }
  free(chains);
  m->cur = llb_gm_close(m);
}

LlbGMatch* llb_gm_new(const LlbGram* g, const LlbGVocab* v) {
  LlbGMatch* m = calloc(1, sizeof *m);
  if (!m) abort();
  m->g = g;
  m->v = v;
  m->st = malloc(sizeof(int32_t) * (size_t)(v->tn + 1));
  if (!m->st) abort();
  llb_gm_tables_init(m);
  llb_gm_reset(m);
  return m;
}

void llb_gm_reset(LlbGMatch* m) {
  llb_gm_start(m);
  m->cur = llb_gm_close(m);
  m->done = 0;
}

void llb_gm_free(LlbGMatch* m) {
  if (!m) return;
  llb_gm_tables_drop(m);
  free(m->npos); free(m->npar); free(m->nh);
  free(m->soff); free(m->slen); free(m->sdata); free(m->sh); free(m->sacc);
  free(m->trans); free(m->mcache); free(m->mcache_n);
  free(m->tmp); free(m->seen); free(m->st);
  free(m);
}

int llb_gm_can_end(const LlbGMatch* m) { return m->done || m->sacc[m->cur]; }
int llb_gm_done(const LlbGMatch* m) { return m->done; }

static int llb_gm_walk(LlbGMatch* m, const unsigned char* s, int n) {
  int st = m->cur;
  for (int i = 0; i < n && st; i += 1) st = llb_gm_step(m, st, s[i]);
  return st;
}

int llb_gm_accept_bytes(LlbGMatch* m, const char* bytes, int len) {
  if (m->done) return len ? -1 : 0;
  llb_gm_compact(m);
  int st = llb_gm_walk(m, (const unsigned char*)bytes, len);
  if (!st) return -1;
  m->cur = st;
  return 0;
}

int llb_gm_allows(LlbGMatch* m, int tok) {
  const LlbGVocab* v = m->v;
  if (tok < 0 || tok >= v->n) return 0;
  if (v->eog[tok]) return llb_gm_can_end(m);
  if (m->done || v->len[tok] == 0 || v->buf[v->off[tok]] == 0) return 0;
  return llb_gm_walk(m, v->buf + v->off[tok], v->len[tok]) != 0;
}

int llb_gm_accept(LlbGMatch* m, int tok) {
  const LlbGVocab* v = m->v;
  if (tok < 0 || tok >= v->n) return -1;
  if (v->eog[tok]) {
    if (!llb_gm_can_end(m)) return -1;
    m->done = 1;
    return 0;
  }
  if (m->done || v->len[tok] == 0 || v->buf[v->off[tok]] == 0) return -1;
  return llb_gm_accept_bytes(m, (const char*)v->buf + v->off[tok], v->len[tok]);
}

void llb_gm_apply(const uint32_t* bits, float* logits, int n) {
  for (int i = 0; i < n; i += 1) {
    if (!(bits[i >> 5] >> (i & 31) & 1)) logits[i] = -INFINITY;
  }
}

int llb_gm_mask(LlbGMatch* m, uint32_t* bits, int n_logits) {
  const LlbGVocab* v = m->v;
  int nw = (n_logits + 31) / 32;
  memset(bits, 0, sizeof(uint32_t) * (size_t)nw);
  int cnt = 0;
  if (m->done) {
    for (int i = 0; i < v->n && i < n_logits; i += 1)
      if (v->eog[i]) { bits[i >> 5] |= 1u << (i & 31); cnt += 1; }
    return cnt;
  }
  llb_gm_compact(m);
  int s = m->cur;
  if (m->mcache[s] && m->mcache_n[s] == n_logits) {
    memcpy(bits, m->mcache[s], sizeof(uint32_t) * (size_t)nw);
    for (int i = 0; i < nw; i += 1) cnt += __builtin_popcount(bits[i]);
    return cnt;
  }
  int32_t* st = m->st;
  st[0] = s;
  for (int i = 1; i < v->tn;) {
    int32_t ns = llb_gm_step(m, st[v->tpar[i]], v->tbyte[i]);
    if (!ns) { i = v->tskip[i]; continue; }
    st[i] = ns;
    for (int k = 0; k < v->ttokn[i]; k += 1) {
      int t = v->ttoks[v->ttok0[i] + k];
      if (t < n_logits) { bits[t >> 5] |= 1u << (t & 31); cnt += 1; }
    }
    i += 1;
  }
  if (m->sacc[s]) {
    for (int i = 0; i < v->n && i < n_logits; i += 1)
      if (v->eog[i]) { bits[i >> 5] |= 1u << (i & 31); cnt += 1; }
  }
  if (m->nmcache < 256) {
    m->mcache[s] = malloc(sizeof(uint32_t) * (size_t)nw);
    if (m->mcache[s]) {
      memcpy(m->mcache[s], bits, sizeof(uint32_t) * (size_t)nw);
      m->mcache_n[s] = n_logits;
      m->nmcache += 1;
    }
  }
  return cnt;
}

// ---- JSON schema -> GBNF ----------------------------------------------------------------
typedef struct {
  LlbBuf out;       // rule text
  const LlbJson* root;
  char* err;
  int errlen;
  int failed;
  int nrule;
  uint32_t prim;    // primitives already emitted
  char** refname;   // $ref -> rule name memo
  char** refrule;
  int nref;
} LlbGS;

#define LLB_GS_SPACE 1u
#define LLB_GS_CHAR 2u
#define LLB_GS_STRING 4u
#define LLB_GS_NUMBER 8u
#define LLB_GS_INTEGER 16u
#define LLB_GS_BOOLEAN 32u
#define LLB_GS_NULL 64u
#define LLB_GS_VALUE 128u
#define LLB_GS_PARTS 256u

static const char* llb_gs_fail(LlbGS* S, const char* fmt, const char* a) {
  if (!S->failed && S->err && S->errlen > 0) {
    int k = snprintf(S->err, (size_t)S->errlen, "json_schema: ");
    if (k < S->errlen) snprintf(S->err + k, (size_t)(S->errlen - k), fmt, a);
  }
  S->failed = 1;
  return "null";
}

static const char* llb_gs_prim(LlbGS* S, uint32_t which) {
  static const struct { uint32_t f; const char* name; const char* body; uint32_t deps; } P[] = {
    {LLB_GS_SPACE, "space", "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}", 0},
    {LLB_GS_CHAR, "char", "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", 0},
    {LLB_GS_STRING, "string", "\"\\\"\" char* \"\\\"\"", LLB_GS_CHAR},
    {LLB_GS_PARTS, "integral-part", "[0] | [1-9] [0-9]{0,15}", 0},
    {LLB_GS_NUMBER, "number", "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", LLB_GS_PARTS},
    {LLB_GS_INTEGER, "integer", "(\"-\"? integral-part)", LLB_GS_PARTS},
    {LLB_GS_BOOLEAN, "boolean", "(\"true\" | \"false\")", 0},
    {LLB_GS_NULL, "null", "\"null\"", 0},
    {LLB_GS_VALUE, "value", "object | array | string | number | boolean | null",
     LLB_GS_STRING | LLB_GS_NUMBER | LLB_GS_BOOLEAN | LLB_GS_NULL | LLB_GS_SPACE},
  };
  const char* name = NULL;
  for (size_t i = 0; i < sizeof P / sizeof P[0]; i += 1) {
    if (P[i].f != which) continue;
    name = P[i].name;
    if (S->prim & which) break;
    S->prim |= which;
    for (uint32_t d = 1; d <= LLB_GS_PARTS; d <<= 1)
      if (P[i].deps & d) llb_gs_prim(S, d);
    llb_buf_puts(&S->out, P[i].name);
    llb_buf_puts(&S->out, " ::= ");
    llb_buf_puts(&S->out, P[i].body);
    llb_buf_puts(&S->out, "\n");
    if (which == LLB_GS_PARTS) llb_buf_puts(&S->out, "decimal-part ::= [0-9]{1,16}\n");
    if (which == LLB_GS_VALUE) {
      llb_buf_puts(&S->out, "object ::= \"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"\n");
      llb_buf_puts(&S->out, "array ::= \"[\" space ( value (\",\" space value)* )? space \"]\"\n");
    }
    break;
  }
  return name;
}

// a GBNF string literal of the given bytes
static void llb_gs_lit(LlbBuf* b, const char* s, size_t n) {
  llb_buf_puts(b, "\"");
  for (size_t i = 0; i < n; i += 1) {
    unsigned char c = (unsigned char)s[i];
    char e[8];
    if (c == '"' || c == '\\') { e[0] = '\\'; e[1] = (char)c; llb_buf_put(b, e, 2); }
    else if (c == '\n') llb_buf_puts(b, "\\n");
    else if (c == '\r') llb_buf_puts(b, "\\r");
    else if (c == '\t') llb_buf_puts(b, "\\t");
    else if (c < 0x20 || c == 0x7F) { snprintf(e, sizeof e, "\\x%02X", c); llb_buf_puts(b, e); }
    else llb_buf_put(b, (const char*)&s[i], 1);
  }
  llb_buf_puts(b, "\"");
}

static char* llb_gs_name(LlbGS* S) {
  char* nm = malloc(24);
  if (!nm) abort();
  snprintf(nm, 24, "r%d", S->nrule++);
  return nm;
}

// rule NAME ::= BODY (body already complete)
static void llb_gs_rule(LlbGS* S, const char* name, const char* body) {
  llb_buf_puts(&S->out, name);
  llb_buf_puts(&S->out, " ::= ");
  llb_buf_puts(&S->out, body);
  llb_buf_puts(&S->out, "\n");
}

static char* llb_gs_visit(LlbGS* S, const LlbJson* sch);

static char* llb_gs_dup(const char* s) {
  size_t n = strlen(s);
  char* d = malloc(n + 1);
  if (!d) abort();
  memcpy(d, s, n + 1);
  return d;
}

// {m,n} repetition of ITEM with SEP between (min 0 / max -1 handled as llama.cpp)
static void llb_gs_rep(LlbBuf* b, const char* item, const char* sep, int mn, int mx) {
  char r[64];
  if (mx == 0) { llb_buf_puts(b, "\"\""); return; }
  if (mn == 0) llb_buf_puts(b, "( ");
  llb_buf_puts(b, item);
  int lo = mn > 0 ? mn - 1 : 0, hi = mx < 0 ? -1 : mx - 1;
  if (!(hi == 0)) {
    llb_buf_puts(b, " ( ");
    llb_buf_puts(b, sep);
    llb_buf_puts(b, " ");
    llb_buf_puts(b, item);
    llb_buf_puts(b, " )");
    if (lo == 0 && hi < 0) llb_buf_puts(b, "*");
    else if (hi < 0) { snprintf(r, sizeof r, "{%d,}", lo); llb_buf_puts(b, r); }
    else if (lo == hi) { snprintf(r, sizeof r, "{%d}", lo); llb_buf_puts(b, r); }
    else { snprintf(r, sizeof r, "{%d,%d}", lo, hi); llb_buf_puts(b, r); }
  }
  if (mn == 0) llb_buf_puts(b, " )?");
}

static int llb_gs_int(const LlbJson* j, int dflt) {
  if (!j || j->t != LLB_J_NUM) return dflt;
  double d = llb_json_num(j);
  return d < 0 ? 0 : d > 100000 ? 100000 : (int)d;
}

static const char* const llb_gs_unsupported[] = {
  "pattern", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
  "uniqueItems", "allOf", "not", "if", "then", "else", "patternProperties", "propertyNames",
  "minProperties", "maxProperties", "prefixItems", "contains", "minContains", "maxContains",
  "dependentRequired", "dependentSchemas", "unevaluatedProperties", "unevaluatedItems", NULL};

static char* llb_gs_typed(LlbGS* S, const LlbJson* sch, const char* type) {
  LlbBuf b = {0};
  llb_buf_put(&b, "", 0);
  char* nm = NULL;
  if (!strcmp(type, "string")) {
    const LlbJson* mn = llb_json_get(sch, "minLength");
    const LlbJson* mx = llb_json_get(sch, "maxLength");
    if (!mn && !mx) { free(b.p); return llb_gs_dup(llb_gs_prim(S, LLB_GS_STRING)); }
    llb_gs_prim(S, LLB_GS_CHAR);
    int lo = llb_gs_int(mn, 0), hi = mx ? llb_gs_int(mx, 0) : -1;
    if (hi >= 0 && hi < lo) { free(b.p); llb_gs_fail(S, "%s", "maxLength < minLength"); return llb_gs_dup("null"); }
    char r[64];
    if (hi < 0) snprintf(r, sizeof r, "\"\\\"\" char{%d,} \"\\\"\"", lo);
    else snprintf(r, sizeof r, "\"\\\"\" char{%d,%d} \"\\\"\"", lo, hi);
    llb_buf_puts(&b, r);
  } else if (!strcmp(type, "number")) { free(b.p); return llb_gs_dup(llb_gs_prim(S, LLB_GS_NUMBER)); }
  else if (!strcmp(type, "integer")) { free(b.p); return llb_gs_dup(llb_gs_prim(S, LLB_GS_INTEGER)); }
  else if (!strcmp(type, "boolean")) { free(b.p); return llb_gs_dup(llb_gs_prim(S, LLB_GS_BOOLEAN)); }
  else if (!strcmp(type, "null")) { free(b.p); return llb_gs_dup(llb_gs_prim(S, LLB_GS_NULL)); }
  else if (!strcmp(type, "array")) {
    const LlbJson* it = llb_json_get(sch, "items");
    char* item = it ? llb_gs_visit(S, it) : llb_gs_dup(llb_gs_prim(S, LLB_GS_VALUE));
    llb_gs_prim(S, LLB_GS_SPACE);
    int lo = llb_gs_int(llb_json_get(sch, "minItems"), 0);
    const LlbJson* mxj = llb_json_get(sch, "maxItems");
    int hi = mxj ? llb_gs_int(mxj, 0) : -1;
    if (hi >= 0 && hi < lo) { free(item); free(b.p); llb_gs_fail(S, "%s", "maxItems < minItems"); return llb_gs_dup("null"); }
    llb_buf_puts(&b, "\"[\" space ");
    llb_gs_rep(&b, item, "\",\" space", lo, hi);
    llb_buf_puts(&b, " space \"]\"");
    free(item);
  } else if (!strcmp(type, "object")) {
    const LlbJson* props = llb_json_get(sch, "properties");
    const LlbJson* req = llb_json_get(sch, "required");
    const LlbJson* ap = llb_json_get(sch, "additionalProperties");
    if (props && props->t != LLB_J_OBJ) { free(b.p); llb_gs_fail(S, "%s", "properties must be an object"); return llb_gs_dup("null"); }
    if (req && req->t != LLB_J_ARR) { free(b.p); llb_gs_fail(S, "%s", "required must be an array"); return llb_gs_dup("null"); }
    llb_gs_prim(S, LLB_GS_SPACE);
    int np = props ? (int)props->n : 0;
    char** kv = calloc((size_t)np + 1, sizeof(char*));
    uint8_t* isreq = calloc((size_t)np + 1, 1);
    if (!kv || !isreq) abort();
    for (size_t r = 0; req && r < req->n; r += 1) {
      const LlbJson* k = req->kids[r];
      int found = 0;
      for (int i = 0; k->t == LLB_J_STR && i < np; i += 1)
        if (!strcmp(props->keys[i], k->s)) { isreq[i] = 1; found = 1; }
      if (!found) { llb_gs_fail(S, "required property '%s' is not in properties", k->t == LLB_J_STR ? k->s : "?"); }
    }
    for (int i = 0; i < np && !S->failed; i += 1) {   // kv_i ::= "\"key\"" space ":" space value
      char* val = llb_gs_visit(S, props->kids[i]);
      LlbBuf kb = {0}, lit = {0};
      llb_buf_put(&kb, "", 0);
      llb_json_quote(&lit, props->keys[i], strlen(props->keys[i]));
      llb_gs_lit(&kb, lit.p, lit.n);
      llb_buf_puts(&kb, " space \":\" space ");
      llb_buf_puts(&kb, val);
      kv[i] = llb_gs_name(S);
      llb_gs_rule(S, kv[i], kb.p);
      free(kb.p); free(lit.p); free(val);
    }
    char* apkv = NULL;
    // G16: a bare object (no properties, no additionalProperties) is any object, as
    // llama.cpp's converter (its primitive "object") and JSON Schema; it was "{}" only
    int bare = !props && !ap;
    if ((bare || (ap && !(ap->t == LLB_J_BOOL && !ap->b))) && !S->failed) {
      char* val = (bare || ap->t == LLB_J_BOOL) ? llb_gs_dup(llb_gs_prim(S, LLB_GS_VALUE)) : llb_gs_visit(S, ap);
      LlbBuf kb = {0};
      llb_buf_puts(&kb, llb_gs_prim(S, LLB_GS_STRING));
      llb_buf_puts(&kb, " \":\" space ");
      llb_buf_puts(&kb, val);
      apkv = llb_gs_name(S);
      llb_gs_rule(S, apkv, kb.p);
      free(kb.p); free(val);
    }
    // optional tail: T_i ::= kv_i ( "," space T_{i+1} )? | T_{i+1}  over the optional props
    int* opt = malloc(sizeof(int) * (size_t)(np + 1));
    if (!opt) abort();
    int no = 0, nr = 0;
    for (int i = 0; i < np; i += 1) { if (isreq[i]) nr += 1; else opt[no++] = i; }
    char* tail = NULL;
    for (int k = no - 1; k >= 0 && !S->failed; k -= 1) {
      LlbBuf tb = {0};
      llb_buf_puts(&tb, kv[opt[k]]);
      if (tail) { llb_buf_puts(&tb, " ( \",\" space "); llb_buf_puts(&tb, tail); llb_buf_puts(&tb, " )? | "); llb_buf_puts(&tb, tail); }
      char* t = llb_gs_name(S);
      llb_gs_rule(S, t, tb.p);
      free(tb.p);
      free(tail);
      tail = t;
    }
    llb_buf_puts(&b, "\"{\" space ");
    if (nr) {
      int first = 1;
      for (int i = 0; i < np; i += 1) {
        if (!isreq[i]) continue;
        if (!first) llb_buf_puts(&b, " \",\" space ");
        llb_buf_puts(&b, kv[i]);
        first = 0;
      }
      if (tail) { llb_buf_puts(&b, " ( \",\" space "); llb_buf_puts(&b, tail); llb_buf_puts(&b, " )?"); }
      if (apkv) { llb_buf_puts(&b, " ( \",\" space "); llb_buf_puts(&b, apkv); llb_buf_puts(&b, " )*"); }
    } else if (tail || apkv) {
      llb_buf_puts(&b, "( ");
      if (tail) { llb_buf_puts(&b, tail); if (apkv) { llb_buf_puts(&b, " ( \",\" space "); llb_buf_puts(&b, apkv); llb_buf_puts(&b, " )*"); } }
      if (tail && apkv) llb_buf_puts(&b, " | ");
      if (apkv) { llb_buf_puts(&b, apkv); llb_buf_puts(&b, " ( \",\" space "); llb_buf_puts(&b, apkv); llb_buf_puts(&b, " )*"); }
      llb_buf_puts(&b, " )?");
    }
    llb_buf_puts(&b, " space \"}\"");
    for (int i = 0; i < np; i += 1) free(kv[i]);
    free(kv); free(isreq); free(opt); free(tail); free(apkv);
  } else {
    free(b.p);
    llb_gs_fail(S, "unknown type '%s'", type);
    return llb_gs_dup("null");
  }
  nm = llb_gs_name(S);
  llb_gs_rule(S, nm, b.p);
  free(b.p);
  return nm;
}

// JSON value -> a rule matching exactly its serialisation (+ space)
static void llb_gs_const(LlbGS* S, LlbBuf* b, const LlbJson* v) {
  char* d = llb_json_dump(v, 0);
  llb_gs_lit(b, d, strlen(d));
  free(d);
  (void)S;
}

static char* llb_gs_visit(LlbGS* S, const LlbJson* sch) {
  if (S->failed) return llb_gs_dup("null");
  if (sch->t == LLB_J_BOOL) {
    if (sch->b) return llb_gs_dup(llb_gs_prim(S, LLB_GS_VALUE));
    llb_gs_fail(S, "%s", "schema false matches nothing");
    return llb_gs_dup("null");
  }
  if (sch->t != LLB_J_OBJ) { llb_gs_fail(S, "%s", "a schema must be an object or a boolean"); return llb_gs_dup("null"); }
  for (int i = 0; llb_gs_unsupported[i]; i += 1) {
    if (llb_json_get(sch, llb_gs_unsupported[i])) { llb_gs_fail(S, "unsupported keyword '%s'", llb_gs_unsupported[i]); return llb_gs_dup("null"); }
  }
  const LlbJson* ref = llb_json_get(sch, "$ref");
  if (ref) {
    if (ref->t != LLB_J_STR) { llb_gs_fail(S, "%s", "$ref must be a string"); return llb_gs_dup("null"); }
    for (int i = 0; i < S->nref; i += 1)
      if (!strcmp(S->refname[i], ref->s)) return llb_gs_dup(S->refrule[i]);
    const char* rest = NULL;
    const char* sect = NULL;
    if (!strncmp(ref->s, "#/$defs/", 8)) { sect = "$defs"; rest = ref->s + 8; }
    else if (!strncmp(ref->s, "#/definitions/", 14)) { sect = "definitions"; rest = ref->s + 14; }
    const LlbJson* tgt = sect ? llb_json_get(llb_json_get(S->root, sect), rest) : NULL;
    if (!tgt) { llb_gs_fail(S, "unresolvable $ref '%s' (only local #/$defs/ and #/definitions/)", ref->s); return llb_gs_dup("null"); }
    char* nm = llb_gs_name(S);   // reserve the name first: recursive schemas refer to it
    S->refname = realloc(S->refname, sizeof(char*) * (size_t)(S->nref + 1));
    S->refrule = realloc(S->refrule, sizeof(char*) * (size_t)(S->nref + 1));
    if (!S->refname || !S->refrule) abort();
    S->refname[S->nref] = llb_gs_dup(ref->s);
    S->refrule[S->nref] = llb_gs_dup(nm);
    S->nref += 1;
    char* body = llb_gs_visit(S, tgt);
    llb_gs_rule(S, nm, body);
    free(body);
    return nm;
  }
  const LlbJson* en = llb_json_get(sch, "enum");
  const LlbJson* cn = llb_json_get(sch, "const");
  if (en || cn) {
    if (en && (en->t != LLB_J_ARR || en->n == 0)) { llb_gs_fail(S, "%s", "enum must be a non-empty array"); return llb_gs_dup("null"); }
    LlbBuf b = {0};
    llb_buf_puts(&b, "( ");
    if (cn) llb_gs_const(S, &b, cn);
    for (size_t i = 0; en && i < en->n; i += 1) {
      if (i) llb_buf_puts(&b, " | ");
      llb_gs_const(S, &b, en->kids[i]);
    }
    llb_buf_puts(&b, " )");
    char* nm = llb_gs_name(S);
    llb_gs_rule(S, nm, b.p);
    free(b.p);
    return nm;
  }
  const LlbJson* any = llb_json_get(sch, "anyOf");
  if (!any) any = llb_json_get(sch, "oneOf");
  if (any) {   // oneOf as anyOf (exclusivity is not enforced; same as llama.cpp)
    if (any->t != LLB_J_ARR || any->n == 0) { llb_gs_fail(S, "%s", "anyOf/oneOf must be a non-empty array"); return llb_gs_dup("null"); }
    LlbBuf b = {0};
    llb_buf_puts(&b, "( ");
    for (size_t i = 0; i < any->n; i += 1) {
      char* r = llb_gs_visit(S, any->kids[i]);
      if (i) llb_buf_puts(&b, " | ");
      llb_buf_puts(&b, r);
      free(r);
    }
    llb_buf_puts(&b, " )");
    char* nm = llb_gs_name(S);
    llb_gs_rule(S, nm, b.p);
    free(b.p);
    return nm;
  }
  const LlbJson* ty = llb_json_get(sch, "type");
  if (!ty) {
    if (llb_json_get(sch, "properties") || llb_json_get(sch, "additionalProperties")) return llb_gs_typed(S, sch, "object");
    if (llb_json_get(sch, "items")) return llb_gs_typed(S, sch, "array");
    return llb_gs_dup(llb_gs_prim(S, LLB_GS_VALUE));
  }
  if (ty->t == LLB_J_STR) return llb_gs_typed(S, sch, ty->s);
  if (ty->t == LLB_J_ARR && ty->n > 0) {
    LlbBuf b = {0};
    llb_buf_puts(&b, "( ");
    for (size_t i = 0; i < ty->n; i += 1) {
      if (ty->kids[i]->t != LLB_J_STR) { free(b.p); llb_gs_fail(S, "%s", "type list entries must be strings"); return llb_gs_dup("null"); }
      char* r = llb_gs_typed(S, sch, ty->kids[i]->s);
      if (i) llb_buf_puts(&b, " | ");
      llb_buf_puts(&b, r);
      free(r);
    }
    llb_buf_puts(&b, " )");
    char* nm = llb_gs_name(S);
    llb_gs_rule(S, nm, b.p);
    free(b.p);
    return nm;
  }
  llb_gs_fail(S, "%s", "bad 'type'");
  return llb_gs_dup("null");
}

char* llb_gram_schema_to_gbnf(const char* schema_json, char* err, int errlen) {
  LlbJson* j = llb_json_parse(schema_json, strlen(schema_json), err, errlen);
  if (!j) return NULL;
  LlbGS S;
  memset(&S, 0, sizeof S);
  S.root = j;
  S.err = err;
  S.errlen = errlen;
  llb_buf_put(&S.out, "", 0);
  char* top = llb_gs_visit(&S, j);
  llb_buf_puts(&S.out, "root ::= ");
  llb_buf_puts(&S.out, top);
  llb_buf_puts(&S.out, "\n");
  free(top);
  for (int i = 0; i < S.nref; i += 1) { free(S.refname[i]); free(S.refrule[i]); }
  free(S.refname);
  free(S.refrule);
  llb_json_free(j);
  if (S.failed) { free(S.out.p); return NULL; }
  return S.out.p;
}

#endif
