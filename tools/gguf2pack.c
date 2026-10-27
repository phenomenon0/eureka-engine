// tools/gguf2pack.c: Qwen3.5 GGUF -> llama-bend pack directory, no Python (GGUF1).
//
//   gguf2pack MODEL.gguf OUT_DIR [--format q8|quant] [--head-only]
//   gguf2pack MODEL.gguf --verify-against PACK_DIR [--format q8|quant]
//   gguf2pack MODEL.gguf --sidecar-only OUT_DIR [--sha] [--format q8|quant] [--target cuda|metal]
//   gguf2pack MODEL.gguf --probe
//
// qwen35moe: MODEL.gguf OUT_DIR [--format quant] writes heads, layer files,
// table.bin, experts.idx and the MoE manifest; routed experts stay in GGUF.
// The other conversion flags below are dense-only.
//
// --sidecar-only (GGUF2): manifest.json + table.bin only, from the header; no
// weight is read and the sha256 fields are empty. --sha fills them with one
// read-only pass. The engine reads the weights from the GGUF (LLB_GGUF).
//
// Writes the bytes tools/q8_pack_qwen35.py (every tensor Q8_0/F32: format
// llama-bend-pack-q8/v0) or tools/quant_pack_qwen35.py (else: q35-quant/v0)
// write, plus pack_index.py's index.tsv and pack_table.py's table.bin, with
// SYS1B's cls.output.* tensors and tied-embedding `output.weight` alias. Those
// Python tools are the reference oracle; --format overrides the auto choice.
// Not written: ops.md (a human doc of the q8 tool, no reader).
//
// pack.bin: GGUF tensors in file order, each after zero padding to 64. F32 and
// every K-quant / ternary tensor verbatim; Q8_0 row-split per row:
// [cols int8 q][cols/32 f16 d]. layers/blk.N.bin = pack.bin[lo, hi) of layer N.
// One streaming pass (pread 16 MiB chunks, RAM bounded by two buffers).
//
// --verify-against: the same pass writing NOTHING; compares the manifest.json,
// index.tsv and table.bin it would write with PACK_DIR's, byte for byte, and
// names every tensor whose sha256 differs. --head-only: pack.bin is sparse:
// only the non-blk tensors (head, norm, embedding) are written, the blk ranges
// are holes (they live in layers/); the GPU engine reads nothing else, but the
// CPU twin / KAT tools that mmap pack.bin would read zeros (HEAD_ONLY.txt).
//
// C11 + POSIX, no dependencies. Little-endian hosts only (checked at start).

#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../host/llb_formats.h"

static void die(const char *fmt, ...) {
  va_list a;
  va_start(a, fmt);
  fputs("gguf2pack: ", stderr);
  vfprintf(stderr, fmt, a);
  fputc('\n', stderr);
  va_end(a);
  exit(1);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) die("out of memory (%zu bytes)", n);
  return p;
}

// ---------------------------------------------------------------- sha256

typedef struct {
  uint32_t h[8];
  uint64_t n;
  uint8_t b[64];
  size_t k;
} Sha;

static const uint32_t SK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, r) ((x) >> (r) | (x) << (32 - (r)))

static void sha_block(uint32_t *h, const uint8_t *p) {
  uint32_t w[64], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + SK[i] + w[i];
    uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
}

static void sha_init(Sha *s) {
  static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  memcpy(s->h, h0, sizeof h0);
  s->n = 0, s->k = 0;
}

static void sha_update(Sha *s, const uint8_t *p, size_t n) {
  s->n += n;
  if (s->k) {
    while (n && s->k < 64) s->b[s->k++] = *p++, n--;
    if (s->k < 64) return;
    sha_block(s->h, s->b), s->k = 0;
  }
  for (; n >= 64; p += 64, n -= 64) sha_block(s->h, p);
  memcpy(s->b, p, n), s->k = n;
}

static void sha_hex(Sha *s, char out[65]) {
  uint64_t bits = s->n * 8;
  uint8_t pad = 0x80, z = 0, len[8];
  sha_update(s, &pad, 1);
  while (s->k != 56) sha_update(s, &z, 1);
  for (int i = 0; i < 8; i++) len[i] = (uint8_t)(bits >> (56 - 8 * i));
  sha_update(s, len, 8);
  for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", s->h[i]);
}

// ---------------------------------------------------------------- text buffer

typedef struct {
  char *p;
  size_t n, cap;
} Buf;

static void bput(Buf *b, const char *s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    b->cap = (b->n + n + 1) * 2;
    b->p = realloc(b->p, b->cap);
    if (!b->p) die("out of memory");
  }
  memcpy(b->p + b->n, s, n), b->n += n, b->p[b->n] = 0;
}

static void bs(Buf *b, const char *s) { bput(b, s, strlen(s)); }

static void bf(Buf *b, const char *fmt, ...) {
  char t[512];
  va_list a;
  va_start(a, fmt);
  int n = vsnprintf(t, sizeof t, fmt, a);
  va_end(a);
  if (n < 0 || n >= (int)sizeof t) die("internal: format overflow");
  bput(b, t, (size_t)n);
}

// Python repr(float): the shortest round-trip digits; exponent form iff the
// decimal exponent is < -4 or >= 16. ponytail: shortest = the first %.*e that
// strtod()s back, the same digits as Python's dtoa except possibly at exotic
// power-of-two boundaries (not hit by any GGUF config value we read).
static void pyrepr(Buf *b, double x) {
  char t[64], dg[32];
  int p, e, nd = 0;
  if (x != x || x - x != 0) die("non-finite float in config");
  for (p = 1; p <= 17; p++) {
    snprintf(t, sizeof t, "%.*e", p - 1, x);
    if (strtod(t, NULL) == x) break;
  }
  char *s = t;
  if (*s == '-') bs(b, "-"), s++;
  for (; *s != 'e'; s++)
    if (*s != '.') dg[nd++] = *s;
  e = atoi(s + 1);
  while (nd > 1 && dg[nd - 1] == '0') nd--;
  dg[nd] = 0;
  if (e < -4 || e >= 16) {
    bput(b, dg, 1);
    if (nd > 1) bs(b, "."), bs(b, dg + 1);
    bf(b, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
  } else if (e < 0) {
    bs(b, "0.");
    for (int i = 0; i < -e - 1; i++) bs(b, "0");
    bs(b, dg);
  } else if (e + 1 >= nd) {
    bs(b, dg);
    for (int i = nd; i < e + 1; i++) bs(b, "0");
    bs(b, ".0");
  } else {
    bput(b, dg, (size_t)e + 1), bs(b, "."), bs(b, dg + e + 1);
  }
}

// json.dumps string with ensure_ascii. Invalid UTF-8 bytes: GGUF strings were
// decoded errors="replace" (U+FFFD), argv paths surrogateescape (U+DC80+b).
// ponytail: one U+FFFD per invalid byte; Python folds some truncated sequences.
static void bjstr(Buf *b, const char *str, int argv) {
  const uint8_t *s = (const uint8_t *)str;
  bs(b, "\"");
  while (*s) {
    uint32_t c = *s, n = 0, lo = 0x80, hi = 0xbf;
    if (c >= 0xc2 && c <= 0xdf) n = 1, c &= 0x1f;
    else if (c >= 0xe0 && c <= 0xef) n = 2, lo = c == 0xe0 ? 0xa0 : 0x80, hi = c == 0xed ? 0x9f : 0xbf, c &= 0x0f;
    else if (c >= 0xf0 && c <= 0xf4) n = 3, lo = c == 0xf0 ? 0x90 : 0x80, hi = c == 0xf4 ? 0x8f : 0xbf, c &= 0x07;
    int ok = c < 0x80 || n;
    for (uint32_t i = 1; ok && i <= n; i++) {
      uint8_t x = s[i];
      if (x < (i == 1 ? lo : 0x80) || x > (i == 1 ? hi : 0xbf)) ok = 0;
      else c = c << 6 | (x & 0x3f);
    }
    if (!ok) c = argv ? 0xdc00 | *s : 0xfffd, n = 0;
    s += n + 1;
    if (c == '"') bs(b, "\\\"");
    else if (c == '\\') bs(b, "\\\\");
    else if (c == '\n') bs(b, "\\n");
    else if (c == '\r') bs(b, "\\r");
    else if (c == '\t') bs(b, "\\t");
    else if (c == '\b') bs(b, "\\b");
    else if (c == '\f') bs(b, "\\f");
    else if (c >= 0x20 && c < 0x7f) { char ch = (char)c; bput(b, &ch, 1); }
    else if (c < 0x10000) bf(b, "\\u%04x", c);
    else c -= 0x10000, bf(b, "\\u%04x\\u%04x", 0xd800 | c >> 10, 0xdc00 | (c & 0x3ff));
  }
  bs(b, "\"");
}

// ---------------------------------------------------------------- values

enum { VN, VI, VU, VF, VB, VS, VL };
typedef struct {
  int k;
  int64_t i;  // VI / VB / VU (as bits)
  double f;
  const char *s;
  const int64_t *l;  // VL
  int n;
} Val;

static void bindent(Buf *b, int d) {
  bs(b, "\n");
  for (int i = 0; i < d; i++) bs(b, " ");
}

// json.dumps(v, indent=1) at depth d (pretty) or separators=(",", ":") (!pretty)
static void bjval(Buf *b, Val v, int d, int pretty) {
  switch (v.k) {
    case VN: bs(b, "null"); break;
    case VI: bf(b, "%lld", (long long)v.i); break;
    case VU: bf(b, "%llu", (unsigned long long)v.i); break;
    case VF: pyrepr(b, v.f); break;
    case VB: bs(b, v.i ? "true" : "false"); break;
    case VS: bjstr(b, v.s, 0); break;
    case VL:
      if (!v.n) { bs(b, "[]"); break; }
      bs(b, "[");
      for (int i = 0; i < v.n; i++) {
        if (i) bs(b, ",");
        if (pretty) bindent(b, d + 1);
        bf(b, "%lld", (long long)v.l[i]);
      }
      if (pretty) bindent(b, d);
      bs(b, "]");
  }
}

// Python str(v) (index.tsv #cfg lines)
static void bpystr(Buf *b, Val v) {
  switch (v.k) {
    case VN: bs(b, "None"); break;
    case VB: bs(b, v.i ? "True" : "False"); break;
    case VS: bs(b, v.s); break;
    case VL:
      bs(b, "[");
      for (int i = 0; i < v.n; i++) bf(b, i ? ", %lld" : "%lld", (long long)v.l[i]);
      bs(b, "]");
      break;
    default: bjval(b, v, 0, 0);
  }
}

// ---------------------------------------------------------------- GGUF reader

typedef struct {
  int fd;
  uint64_t off;  // file offset of buf[0]
  uint8_t *buf;
  size_t n, p;
} Rd;

static void rd_fill(Rd *r) {
  r->off += r->n, r->p = 0;
  ssize_t k = pread(r->fd, r->buf, 1 << 20, (off_t)r->off);
  if (k <= 0) die("GGUF header truncated at byte %llu", (unsigned long long)r->off);
  r->n = (size_t)k;
}

static void rd_get(Rd *r, void *dst, size_t k) {
  uint8_t *d = dst;
  while (k) {
    if (r->p == r->n) rd_fill(r);
    size_t c = r->n - r->p < k ? r->n - r->p : k;
    if (d) memcpy(d, r->buf + r->p, c), d += c;
    r->p += c, k -= c;
  }
}

static uint64_t rd_pos(Rd *r) { return r->off + r->p; }
static uint32_t rd_u32(Rd *r) { uint32_t v; rd_get(r, &v, 4); return v; }
static uint64_t rd_u64(Rd *r) { uint64_t v; rd_get(r, &v, 8); return v; }

static char *rd_str(Rd *r, int keep) {
  uint64_t n = rd_u64(r);
  if (n > (1ull << 31)) die("GGUF string length %llu: corrupt header", (unsigned long long)n);
  if (!keep) { rd_get(r, NULL, n); return NULL; }
  char *s = xmalloc(n + 1);
  rd_get(r, s, n), s[n] = 0;
  if (strlen(s) != n) die("GGUF string with a NUL byte");
  return s;
}

// GGUF metadata value types: size of the numeric ones (0 = not numeric)
static const int KV_SIZE[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

typedef struct {
  char *key;
  uint32_t t;  // GGUF value type
  Val v;       // scalars, strings, small int arrays
  uint64_t an; // array length
} KV;

static int64_t rd_int(Rd *r, uint32_t t) {
  uint8_t x[8] = {0};
  rd_get(r, x, (size_t)KV_SIZE[t]);
  switch (t) {
    case 0: case 7: return x[0];
    case 1: return (int8_t)x[0];
    case 2: { uint16_t v; memcpy(&v, x, 2); return v; }
    case 3: { int16_t v; memcpy(&v, x, 2); return v; }
    case 4: { uint32_t v; memcpy(&v, x, 4); return v; }
    case 5: { int32_t v; memcpy(&v, x, 4); return v; }
    default: { int64_t v; memcpy(&v, x, 8); return v; }  // 10 u64 (bits), 11 i64
  }
}

static Val rd_val(Rd *r, uint32_t t, uint64_t *an) {
  Val v = {VN, 0, 0, NULL, NULL, 0};
  if (t > 12) die("GGUF metadata value type %u unknown", t);
  if (t == 8) v.k = VS, v.s = rd_str(r, 1);
  else if (t == 6) { float f; rd_get(r, &f, 4); v.k = VF, v.f = f; }
  else if (t == 12) { double f; rd_get(r, &f, 8); v.k = VF, v.f = f; }
  else if (t == 9) {
    uint32_t et = rd_u32(r);
    uint64_t n = rd_u64(r);
    *an = n;
    if (et == 9 || et > 12) die("GGUF nested/unknown array type %u", et);
    int isint = KV_SIZE[et] && et != 6 && et != 7 && et != 12;
    if (isint && n <= 64) {
      int64_t *l = xmalloc(n * 8);
      for (uint64_t i = 0; i < n; i++) l[i] = rd_int(r, et);
      v.k = VL, v.l = l, v.n = (int)n;
    } else if (et == 8) {
      for (uint64_t i = 0; i < n; i++) rd_str(r, 0);
    } else {
      rd_get(r, NULL, n * (uint64_t)KV_SIZE[et]);
    }
  } else {
    v.i = rd_int(r, t);
    v.k = t == 7 ? VB : t == 10 ? VU : VI;
  }
  return v;
}

// ggml type id -> registry name (fm / block size from host/llb_formats.h)
static const struct { uint32_t id; const char *name; } GGML[] = {
    {0, "F32"}, {1, "F16"}, {8, "Q8_0"}, {11, "Q3_K"}, {12, "Q4_K"}, {13, "Q5_K"},
    {14, "Q6_K"}, {23, "IQ4_XS"}, {30, "BF16"}, {42, "Q2_0"}, {142, "PQ2_0"}};

typedef struct { const char *name; int fm, bytes, elems; } Fmt;
#define X(n, f, b, e) {#n, f, b, e},
static const Fmt FMTS[] = {LLB_FORMATS(X)};
#undef X

static const Fmt *fmt_of(const char *name) {
  for (size_t i = 0; i < sizeof FMTS / sizeof *FMTS; i++)
    if (!strcmp(FMTS[i].name, name)) return &FMTS[i];
  return NULL;
}

// quant_pack_qwen35.DOORS; scale (f16) byte offset inside each block, -1 none
static const struct { const char *name; int d0, d1; } DOORS[] = {
    {"F32", -1, -1}, {"Q8_0", 0, -1}, {"Q4_K", 0, 2}, {"Q6_K", 208, -1},
    {"Q3_K", 108, -1}, {"Q5_K", 0, 2}, {"Q2_0", 0, -1}, {"PQ2_0", 0, -1}};

typedef struct {
  char *name;
  const char *tn;  // ggml type name
  const Fmt *f;    // NULL if the registry lacks it
  int door;        // index in DOORS, -1 not packable
  int nd;
  int64_t sh[4];   // row-major shape (GGUF ne reversed)
  uint64_t src, bytes, off, rb;  // GGUF data offset, bytes, pack offset, row_bytes
  int layer;       // blk.N -> N, else -1
  char sha[65];
} T;

typedef struct {
  KV *kv;
  size_t nkv;
  T *t;
  size_t nt;
  uint32_t version;
  uint64_t align, data, fsize;
} Gguf;

static KV *kv_get(Gguf *g, const char *key) {
  for (size_t i = 0; i < g->nkv; i++)
    if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
  return NULL;
}

static Gguf gguf_read(int fd) {
  Gguf g = {0};
  struct stat st;
  if (fstat(fd, &st)) die("stat: %s", strerror(errno));
  g.fsize = (uint64_t)st.st_size;
  Rd r = {fd, 0, xmalloc(1 << 20), 0, 0};
  char magic[4];
  rd_get(&r, magic, 4);
  g.version = rd_u32(&r);
  if (memcmp(magic, "GGUF", 4) || (g.version != 2 && g.version != 3))
    die("not a GGUF v2/v3 file (magic %.4s, version %u)", magic, g.version);
  g.nt = rd_u64(&r), g.nkv = rd_u64(&r);
  if (g.nt > 1000000 || g.nkv > 1000000) die("GGUF header counts implausible: corrupt file");
  g.kv = xmalloc(g.nkv * sizeof *g.kv);
  for (size_t i = 0; i < g.nkv; i++) {
    KV *k = &g.kv[i];
    k->key = rd_str(&r, 1), k->t = rd_u32(&r), k->an = 0;
    k->v = rd_val(&r, k->t, &k->an);
  }
  g.t = xmalloc(g.nt * sizeof *g.t);
  for (size_t i = 0; i < g.nt; i++) {
    T *t = &g.t[i];
    memset(t, 0, sizeof *t);
    t->name = rd_str(&r, 1);
    uint32_t nd = rd_u32(&r);
    if (nd < 1 || nd > 4) die("%s: %u dims", t->name, nd);
    int64_t ne[4];
    for (uint32_t j = 0; j < nd; j++) ne[j] = (int64_t)rd_u64(&r);
    t->nd = (int)nd;
    for (uint32_t j = 0; j < nd; j++) t->sh[j] = ne[nd - 1 - j];
    uint32_t ty = rd_u32(&r);
    uint64_t rel = rd_u64(&r);
    t->tn = NULL;
    for (size_t j = 0; j < sizeof GGML / sizeof *GGML; j++)
      if (GGML[j].id == ty) t->tn = GGML[j].name;
    if (!t->tn) die("%s: ggml type %u unsupported (known: F32 F16 Q8_0 Q3_K Q4_K Q5_K Q6_K IQ4_XS BF16 Q2_0 PQ2_0)", t->name, ty);
    t->f = fmt_of(t->tn);
    t->door = -1;
    for (size_t j = 0; j < sizeof DOORS / sizeof *DOORS; j++)
      if (!strcmp(DOORS[j].name, t->tn)) t->door = (int)j;
    int elems = 1, bb = 2;  // BF16 is not in the registry
    if (t->f) elems = t->f->elems, bb = t->f->bytes;
    uint64_t n = 1;
    for (int j = 0; j < t->nd; j++) n *= (uint64_t)t->sh[j];
    if (n % (uint64_t)elems) die("%s: %llu elements not a multiple of the %s block", t->name, (unsigned long long)n, t->tn);
    t->bytes = n / (uint64_t)elems * (uint64_t)bb;
    t->src = rel;  // made absolute below
    t->layer = -1;
    unsigned L;
    int c = 0;
    if (sscanf(t->name, "blk.%u.%n", &L, &c) == 1 && c > 0) t->layer = (int)L;
  }
  KV *a = kv_get(&g, "general.alignment");
  g.align = a && (a->v.k == VI || a->v.k == VU) ? (uint64_t)a->v.i : 32;
  if (!g.align) die("general.alignment 0");
  g.data = (rd_pos(&r) + g.align - 1) / g.align * g.align;
  for (size_t i = 0; i < g.nt; i++) {
    g.t[i].src += g.data;
    if (g.t[i].src + g.t[i].bytes > g.fsize)
      die("%s: data [%llu, +%llu) past the end of the file (%llu B): truncated GGUF", g.t[i].name,
          (unsigned long long)g.t[i].src, (unsigned long long)g.t[i].bytes, (unsigned long long)g.fsize);
  }
  free(r.buf);
  return g;
}

// ---------------------------------------------------------------- config

typedef struct { const char *k; Val v; } CfgE;
typedef struct { CfgE e[48]; int n; } Cfg;

static void cfg_put(Cfg *c, const char *k, Val v) {
  if (c->n == 48) die("internal: config full");
  c->e[c->n].k = k, c->e[c->n].v = v, c->n++;
}

static Val cfg_get(Cfg *c, const char *k) {
  for (int i = 0; i < c->n; i++)
    if (!strcmp(c->e[i].k, k)) return c->e[i].v;
  return (Val){VN, 0, 0, NULL, NULL, 0};
}

static int64_t cfg_int(Cfg *c, const char *k) {
  Val v = cfg_get(c, k);
  if (v.k != VI && v.k != VU) die("config %s is not an integer", k);
  return v.i;
}

static Val kvv(Gguf *g, const char *key, int need) {
  KV *k = kv_get(g, key);
  if (!k && need) die("GGUF has no %s (the pack needs it)", key);
  return k ? k->v : (Val){VN, 0, 0, NULL, NULL, 0};
}

static Val vint(int64_t i) { return (Val){VI, i, 0, NULL, NULL, 0}; }

static int has_tensor(Gguf *g, const char *name) {
  for (size_t i = 0; i < g->nt; i++)
    if (!strcmp(g->t[i].name, name)) return 1;
  return 0;
}

// Clef stores the same hybrid backbone metadata under clef.*.
static KV *backbone_kv(Gguf *g, const char *suffix) {
  Val arch = kvv(g, "general.architecture", 1);
  char key[160];
  snprintf(key, sizeof key, "%s.%s", arch.s, suffix);
  return kv_get(g, key);
}

static Val backbone_val(Gguf *g, const char *suffix, int need) {
  KV *k = backbone_kv(g, suffix);
  if (!k && need) die("GGUF has no %s.%s (the pack needs it)", kvv(g, "general.architecture", 1).s, suffix);
  return k ? k->v : (Val){VN, 0, 0, NULL, NULL, 0};
}

// gguf_convert.build_config_qwen35, key order and defaults included
static Cfg build_config(Gguf *g) {
  Cfg c = {.n = 0};
  Val arch = kvv(g, "general.architecture", 1);
  if (arch.k != VS || (strcmp(arch.s, "qwen35") && strcmp(arch.s, "clef") && strcmp(arch.s, "qwen35moe")))
    die("architecture %s: only qwen35 or clef (Qwen3.5 hybrid backbone) is packed", arch.k == VS ? arch.s : "?");
  // qwen35moe (--sidecar-only, moe_sidecar_cuda): the same backbone keys; d_ff is the wider of the
  // routed and the shared expert widths (the dense scratch the shared expert runs in)
  int moe = !strcmp(arch.s, "qwen35moe");
  static const int64_t zero4[4] = {0, 0, 0, 0};
  KV *tok = kv_get(g, "tokenizer.ggml.tokens");
  if (!tok || tok->t != 9) die("GGUF has no tokenizer.ggml.tokens array");
  Val hd = backbone_val(g, "attention.key_length", 1), v;
  cfg_put(&c, "architecture", (Val){VS, 0, 0, moe ? "qwen3_5_moe" : "qwen3_5", NULL, 0});
  cfg_put(&c, "n_layers", backbone_val(g, "block_count", 1));
  cfg_put(&c, "n_heads", backbone_val(g, "attention.head_count", 1));
  cfg_put(&c, "n_kv_heads", backbone_val(g, "attention.head_count_kv", 1));
  cfg_put(&c, "d_model", backbone_val(g, "embedding_length", 1));
  cfg_put(&c, "head_dim", hd);
  if (moe) {
    Val ef = backbone_val(g, "expert_feed_forward_length", 1), sf = backbone_val(g, "expert_shared_feed_forward_length", 1);
    if ((ef.k != VI && ef.k != VU) || (sf.k != VI && sf.k != VU))
      die("qwen35moe: expert_feed_forward_length / expert_shared_feed_forward_length are not integers");
    cfg_put(&c, "d_ff", vint(ef.i > sf.i ? ef.i : sf.i));
  } else {
    cfg_put(&c, "d_ff", backbone_val(g, "feed_forward_length", 1));
  }
  v = backbone_val(g, "vocab_size", 0);
  cfg_put(&c, "vocab_size", v.k == VN && !backbone_kv(g, "vocab_size") ? vint((int64_t)tok->an) : v);
  cfg_put(&c, "context_length", backbone_val(g, "context_length", 1));
  v = backbone_val(g, "rope.freq_base", 0);
  cfg_put(&c, "rope_freq_base", backbone_kv(g, "rope.freq_base") ? v : (Val){VF, 0, 10000.0, NULL, NULL, 0});
  cfg_put(&c, "rope_dim", backbone_kv(g, "rope.dimension_count") ? backbone_val(g, "rope.dimension_count", 1) : hd);
  KV *rs = backbone_kv(g, "rope.dimension_sections");
  if (rs && rs->v.k != VL) die("%s.rope.dimension_sections is not a small int array", arch.s);
  cfg_put(&c, "rope_sections", rs ? rs->v : (Val){VL, 0, 0, NULL, zero4, 4});
  cfg_put(&c, "rope_type", (Val){VS, 0, 0, "neox_mrope_text", NULL, 0});
  cfg_put(&c, "rms_norm_eps", backbone_val(g, "attention.layer_norm_rms_epsilon", 1));
  cfg_put(&c, "full_attention_interval", backbone_val(g, "full_attention_interval", 1));
  cfg_put(&c, "ssm_conv_kernel", backbone_val(g, "ssm.conv_kernel", 1));
  cfg_put(&c, "ssm_state_size", backbone_val(g, "ssm.state_size", 1));
  cfg_put(&c, "ssm_group_count", backbone_val(g, "ssm.group_count", 1));
  cfg_put(&c, "ssm_time_step_rank", backbone_val(g, "ssm.time_step_rank", 1));
  cfg_put(&c, "ssm_inner_size", backbone_val(g, "ssm.inner_size", 1));
  cfg_put(&c, "tied_embeddings", (Val){VB, !has_tensor(g, "output.weight"), 0, NULL, NULL, 0});
  cfg_put(&c, "bos_token_id", kvv(g, "tokenizer.ggml.bos_token_id", 0));
  cfg_put(&c, "eos_token_id", kvv(g, "tokenizer.ggml.eos_token_id", 0));
  cfg_put(&c, "eot_token_id", kvv(g, "tokenizer.ggml.eot_token_id", 0));
  cfg_put(&c, "pad_token_id", kvv(g, "tokenizer.ggml.padding_token_id", 0));
  cfg_put(&c, "add_bos_token", kvv(g, "tokenizer.ggml.add_bos_token", 0));
  cfg_put(&c, "tokenizer_model", kvv(g, "tokenizer.ggml.model", 0));
  cfg_put(&c, "tokenizer_pre", kvv(g, "tokenizer.ggml.pre", 0));
  cfg_put(&c, "source_file_type", kvv(g, "general.file_type", 0));
  int64_t S = cfg_int(&c, "ssm_state_size"), kd = S * cfg_int(&c, "ssm_group_count");
  int64_t inner = cfg_int(&c, "ssm_inner_size"), tr = cfg_int(&c, "ssm_time_step_rank");
  if (tr <= 0) die("ssm_time_step_rank %lld", (long long)tr);
  cfg_put(&c, "linear_key_dim", vint(kd));
  cfg_put(&c, "linear_value_dim", vint(inner));
  cfg_put(&c, "linear_conv_dim", vint(2 * kd + inner));
  cfg_put(&c, "linear_value_head_dim", vint(inner / tr));  // Python // (positive here)
  return c;
}

// The decision layers are not part of autoregressive backbone inference.
// Keep token_types.weight as an unused, shape-checked question-type table.
static void drop_clef_head(Gguf *g) {
  size_t o = 0, dec = 0, decision = 0;
  for (size_t i = 0; i < g->nt; i++) {
    if (!strncmp(g->t[i].name, "dec.", 4)) dec++;
    else if (!strncmp(g->t[i].name, "decision.", 9)) decision++;
    else g->t[o++] = g->t[i];
  }
  fprintf(stderr, "gguf2pack: Clef decision head skipped (%zu dec.* + %zu decision.* = %zu tensors; backbone only)\n",
          dec, decision, dec + decision);
  if (has_tensor(g, "token_types.weight"))
    fprintf(stderr, "gguf2pack: token_types.weight retained (question-type embedding; unused by backbone)\n");
  g->nt = o;
}

// Some finetunes carry the speculative-decoding "NextN" (MTP) block as
// blk.<block_count-1>.* and count it in qwen35.block_count. Plain decode never
// reads it. If that trailing block is EXACTLY the registered MTP tensor set
// (15 tensors) with the shapes this config implies, drop those tensors and
// decrement n_layers, so no downstream step (shape check, offsets, table,
// manifest, writes) ever counts the block. Anything else still fails as before.
static void drop_mtp_block(Gguf *g, Cfg *c) {
  int64_t L0 = cfg_int(c, "n_layers");
  if (L0 <= 1 || L0 > 4096) return;
  char nxt[80];
  snprintf(nxt, sizeof nxt, "blk.%lld.nextn.eh_proj.weight", (long long)(L0 - 1));
  if (!has_tensor(g, nxt)) return;
  int64_t D = cfg_int(c, "d_model"), F = cfg_int(c, "d_ff");
  int64_t H = cfg_int(c, "n_heads"), KVh = cfg_int(c, "n_kv_heads"), hd = cfg_int(c, "head_dim");
  struct { const char *n; int nd; int64_t a, b; } req[] = {
      {"attn_norm.weight", 1, D, 0}, {"post_attention_norm.weight", 1, D, 0},
      {"attn_q.weight", 2, 2 * hd * H, D}, {"attn_k.weight", 2, hd * KVh, D},
      {"attn_v.weight", 2, hd * KVh, D}, {"attn_output.weight", 2, D, hd * H},
      {"attn_q_norm.weight", 1, hd, 0}, {"attn_k_norm.weight", 1, hd, 0},
      {"ffn_gate.weight", 2, F, D}, {"ffn_up.weight", 2, F, D}, {"ffn_down.weight", 2, D, F},
      {"nextn.eh_proj.weight", 2, D, 2 * D}, {"nextn.enorm.weight", 1, D, 0},
      {"nextn.hnorm.weight", 1, D, 0}, {"nextn.shared_head_norm.weight", 1, D, 0}};
  size_t nreq = sizeof req / sizeof *req, inblk = 0;
  char pfx[64];
  snprintf(pfx, sizeof pfx, "blk.%lld.", (long long)(L0 - 1));
  size_t pfxlen = strlen(pfx);
  for (size_t i = 0; i < g->nt; i++)
    if (!strncmp(g->t[i].name, pfx, pfxlen)) inblk++;
  if (inblk != nreq) return;
  for (size_t j = 0; j < nreq; j++) {
    char nm[96];
    snprintf(nm, sizeof nm, "%s%s", pfx, req[j].n);
    T *t = NULL;
    for (size_t i = 0; i < g->nt; i++)
      if (!strcmp(g->t[i].name, nm)) t = &g->t[i];
    if (!t || t->nd != req[j].nd || t->sh[0] != req[j].a || (req[j].nd == 2 && t->sh[1] != req[j].b)) return;
  }
  size_t o = 0;
  for (size_t i = 0; i < g->nt; i++)
    if (strncmp(g->t[i].name, pfx, pfxlen)) g->t[o++] = g->t[i];
  fprintf(stderr, "gguf2pack: MTP block %s* skipped (%zu tensors; plain decode)\n", pfx, g->nt - o);
  g->nt = o;
  for (int i = 0; i < c->n; i++)
    if (!strcmp(c->e[i].k, "n_layers")) c->e[i].v = vint(L0 - 1);
}

// gguf_convert.check_shapes_qwen35 (+ SYS1B's cls.output.*): every tensor's
// shape derivable from the config, nothing unexpected but rope_*
typedef struct { char name[96]; int nd; int64_t sh[2]; } Want;

static int check_shapes(Gguf *g, Cfg *c) {
  int64_t D = cfg_int(c, "d_model"), F = cfg_int(c, "d_ff"), V = cfg_int(c, "vocab_size");
  int64_t H = cfg_int(c, "n_heads"), KVh = cfg_int(c, "n_kv_heads"), hd = cfg_int(c, "head_dim");
  int64_t TR = cfg_int(c, "ssm_time_step_rank"), kd = cfg_int(c, "linear_key_dim");
  int64_t vd = cfg_int(c, "linear_value_dim"), conv = cfg_int(c, "linear_conv_dim");
  int64_t kc = cfg_int(c, "ssm_conv_kernel"), vhd = cfg_int(c, "linear_value_head_dim");
  int64_t L = cfg_int(c, "n_layers"), iv = cfg_int(c, "full_attention_interval");
  if (L < 1 || L > 4096 || iv < 1) die("n_layers %lld / full_attention_interval %lld", (long long)L, (long long)iv);
  size_t cap = 8 + (size_t)L * 16, n = 0;
  Want *w = xmalloc(cap * sizeof *w);
#define W(nm, d, a, b) (snprintf(w[n].name, sizeof w[n].name, "%s", nm), w[n].nd = d, w[n].sh[0] = a, w[n].sh[1] = b, n++)
  int tied = cfg_get(c, "tied_embeddings").i;
  W("token_embd.weight", 2, V, D);
  W("output_norm.weight", 1, D, 0);
  if (!tied) W("output.weight", 2, V, D);
  Val arch = kvv(g, "general.architecture", 1);
  if (arch.k == VS && !strcmp(arch.s, "clef") && has_tensor(g, "token_types.weight"))
    W("token_types.weight", 2, 3, 1024);
  for (int64_t i = 0; i < L; i++) {
    char p[160];
#define P(s, d, a, b) (snprintf(p, sizeof p, "blk.%lld.%s", (long long)i, s), W(p, d, a, b))
    P("attn_norm.weight", 1, D, 0);
    P("post_attention_norm.weight", 1, D, 0);
    P("ffn_gate.weight", 2, F, D);
    P("ffn_up.weight", 2, F, D);
    P("ffn_down.weight", 2, D, F);
    if ((i + 1) % iv == 0) {
      P("attn_q.weight", 2, 2 * hd * H, D);
      P("attn_k.weight", 2, hd * KVh, D);
      P("attn_v.weight", 2, hd * KVh, D);
      P("attn_output.weight", 2, D, hd * H);
      P("attn_q_norm.weight", 1, hd, 0);
      P("attn_k_norm.weight", 1, hd, 0);
    } else {
      P("attn_qkv.weight", 2, 2 * kd + vd, D);
      P("attn_gate.weight", 2, vd, D);
      P("ssm_conv1d.weight", 2, conv, kc);
      P("ssm_dt.bias", 1, TR, 0);
      P("ssm_a", 1, TR, 0);
      P("ssm_beta.weight", 2, TR, D);
      P("ssm_alpha.weight", 2, TR, D);
      P("ssm_norm.weight", 1, vhd, 0);
      P("ssm_out.weight", 2, D, vd);
    }
#undef P
  }
  for (size_t i = 0; i < g->nt; i++)
    if (!strcmp(g->t[i].name, "cls.output.weight")) {  // decision-model head (Kev)
      W("cls.output.weight", 2, g->t[i].sh[0], D);
      W("cls.output.bias", 1, g->t[i].sh[0], 0);
    }
#undef W
  int bad = 0;
  for (size_t j = 0; j < n; j++) {
    T *t = NULL;
    for (size_t i = 0; i < g->nt; i++)
      if (!strcmp(g->t[i].name, w[j].name)) t = &g->t[i];
    int ok = t && t->nd == w[j].nd && t->sh[0] == w[j].sh[0] && (w[j].nd == 1 || t->sh[1] == w[j].sh[1]);
    if (!ok) {
      fprintf(stderr, "  %s: want [%lld%s", w[j].name, (long long)w[j].sh[0], w[j].nd == 2 ? ", " : "]");
      if (w[j].nd == 2) fprintf(stderr, "%lld]", (long long)w[j].sh[1]);
      if (!t) fprintf(stderr, " got None\n");
      else {
        fprintf(stderr, " got [");
        for (int k = 0; k < t->nd; k++) fprintf(stderr, k ? ", %lld" : "%lld", (long long)t->sh[k]);
        fprintf(stderr, "]\n");
      }
      bad++;
    }
  }
  for (size_t i = 0; i < g->nt; i++) {
    int known = !strncmp(g->t[i].name, "rope_", 5);
    for (size_t j = 0; j < n && !known; j++) known = !strcmp(g->t[i].name, w[j].name);
    if (!known) fprintf(stderr, "  %s: unexpected tensor\n", g->t[i].name), bad++;
    for (size_t j = 0; j < i; j++)
      if (!strcmp(g->t[i].name, g->t[j].name)) fprintf(stderr, "  %s: duplicate tensor\n", g->t[i].name), bad++;
  }
  int64_t rd = cfg_int(c, "rope_dim");
  if (rd > hd || rd % 2) fprintf(stderr, "  rope_dim %lld incompatible with head_dim %lld\n", (long long)rd, (long long)hd), bad++;
  free(w);
  return bad;
}

// ---------------------------------------------------------------- pack plan

// pack_table.OFFS: the engine's per-layer tensor order
static const char *OFFS[20] = {
    "attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight",
    "ffn_down.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight",
    "ssm_alpha.weight", "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight",
    "ssm_out.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight"};

#define ALIGN 64

typedef struct { uint64_t lo, hi; int fd; } Layer;

static int pread_all(int fd, void *p, size_t n, uint64_t off) {
  uint8_t *b = p;
  while (n) {
    ssize_t k = pread(fd, b, n, (off_t)off);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    b += k, n -= (size_t)k, off += (uint64_t)k;
  }
  return 0;
}

static void pwrite_all(int fd, const void *p, size_t n, uint64_t off, const char *what) {
  const uint8_t *b = p;
  while (n) {
    ssize_t k = pwrite(fd, b, n, (off_t)off);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) die("write %s: %s", what, k < 0 ? strerror(errno) : "short write");
    b += k, n -= (size_t)k, off += (uint64_t)k;
  }
}

typedef struct {
  int pack_fd, head_only, cur_layer, nl;
  Layer *ly;
  uint64_t pos;
  Sha pack, ten;
} Out;

static void emit(Out *o, const uint8_t *p, size_t n, int data) {
  sha_update(&o->pack, p, n);
  if (data) sha_update(&o->ten, p, n);
  if (o->pack_fd >= 0 && !(o->head_only && o->cur_layer >= 0)) pwrite_all(o->pack_fd, p, n, o->pos, "pack.bin");
  for (int i = 0; i < o->nl; i++) {
    Layer *l = &o->ly[i];
    if (l->fd < 0) continue;
    uint64_t a = o->pos > l->lo ? o->pos : l->lo, b = o->pos + n < l->hi ? o->pos + n : l->hi;
    if (a < b) pwrite_all(l->fd, p + (a - o->pos), b - a, a - l->lo, "layers/blk.N.bin");
  }
  o->pos += n;
}

static int f16_finite(const uint8_t *p) { return (((p[1] << 8 | p[0]) & 0x7c00) != 0x7c00); }

static void rowsplit(const uint8_t *in, uint8_t *out, size_t n, size_t cols, size_t unit) {
  for (size_t r = 0; r < n; r += unit) {
    const uint8_t *s = in + r;
    uint8_t *q = out + r, *d = out + r + cols;
    for (size_t j = 0; j < cols / 32; j++)
      memcpy(q + 32 * j, s + 34 * j + 2, 32), memcpy(d + 2 * j, s + 34 * j, 2);
  }
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long vm_hwm_kb(void) {
  FILE *f = fopen("/proc/self/status", "r");
  char ln[256];
  long v = -1;
  if (!f) return -1;
  while (fgets(ln, sizeof ln, f))
    if (!strncmp(ln, "VmHWM:", 6)) v = atol(ln + 6);
  fclose(f);
  return v;
}

// ---------------------------------------------------------------- outputs

static void manifest_tensor(Buf *b, T *t, int q8, const char *alias_of, const char *as) {
  bs(b, "  "), bjstr(b, as ? as : t->name, 0), bs(b, ": {\n   \"shape\": [");
  for (int i = 0; i < t->nd; i++) bf(b, "%s\n    %lld", i ? "," : "", (long long)t->sh[i]);
  bf(b, "\n   ],\n   \"offset\": %llu,\n   \"bytes\": %llu,\n   \"src_type\": \"%s\",\n",
     (unsigned long long)t->off, (unsigned long long)t->bytes, t->tn);
  if (!q8) bf(b, "   \"fmt\": %d,\n", t->f->fm);
  bf(b, "   \"sha256\": \"%s\"", t->sha);
  if (strcmp(t->tn, "F32")) bf(b, ",\n   \"row_bytes\": %llu", (unsigned long long)t->rb);
  if (alias_of) bs(b, ",\n   \"alias_of\": "), bjstr(b, alias_of, 0);
  bs(b, "\n  }");
}

static const char *IDX_KEYS[22] = {
    "architecture", "n_layers", "n_heads", "n_kv_heads", "d_model", "head_dim",
    "d_ff", "vocab_size", "context_length", "rope_freq_base", "rope_dim",
    "rms_norm_eps", "full_attention_interval", "ssm_conv_kernel", "ssm_state_size",
    "ssm_group_count", "ssm_time_step_rank", "ssm_inner_size", "linear_key_dim",
    "linear_value_dim", "linear_conv_dim", "linear_value_head_dim"};

static void index_line(Buf *b, T *t, const char *as) {
  int64_t rows = t->nd == 2 ? t->sh[0] : 1, cols = t->nd == 2 ? t->sh[1] : t->sh[0];
  bf(b, "%s\t%llu\t%llu\t%lld\t%lld\t%s\t%llu\t%d\n", as ? as : t->name, (unsigned long long)t->off,
     (unsigned long long)t->bytes, (long long)rows, (long long)cols, t->tn,
     (unsigned long long)(strcmp(t->tn, "F32") ? t->rb : 0), t->f->fm);
}

static Buf read_file(const char *dir, const char *name, int *ok) {
  char p[4096];
  Buf b = {0};
  snprintf(p, sizeof p, "%s/%s", dir, name);
  FILE *f = fopen(p, "rb");
  *ok = f != NULL;
  if (!f) return b;
  char t[65536];
  size_t k;
  while ((k = fread(t, 1, sizeof t, f)) > 0) bput(&b, t, k);
  fclose(f);
  if (!b.p) bput(&b, "", 0);
  return b;
}

static void write_file(const char *dir, const char *name, const void *p, size_t n) {
  char path[4096];
  snprintf(path, sizeof path, "%s/%s", dir, name);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die("open %s: %s", path, strerror(errno));
  pwrite_all(fd, p, n, 0, name);
  if (close(fd)) die("close %s: %s", path, strerror(errno));
}

// compare generated vs on-disk; returns 1 if identical
static int compare(const char *dir, const char *name, Buf *mine) {
  int ok;
  Buf old = read_file(dir, name, &ok);
  if (!ok) { printf("  %-14s MISSING in %s\n", name, dir); return 0; }
  int same = old.n == mine->n && !memcmp(old.p, mine->p, mine->n);
  if (same) printf("  %-14s identical (%zu bytes)\n", name, mine->n);
  else {
    size_t i = 0, line = 1;
    while (i < old.n && i < mine->n && old.p[i] == mine->p[i]) line += old.p[i++] == '\n';
    printf("  %-14s DIFFERS: %zu vs %zu bytes, first difference at byte %zu (line %zu)\n", name, mine->n, old.n, i, line);
  }
  free(old.p);
  return same;
}

// the sha256 recorded for tensor `name` in an indent=1 manifest, or NULL
static int old_sha(const char *m, const char *name, char out[65]) {
  char key[256];
  snprintf(key, sizeof key, "\n  \"%s\": {", name);
  const char *p = strstr(m, key), *s;
  if (!p || !(s = strstr(p, "\"sha256\": \"")) || strchr(p + 1, '}') < s) return 0;
  memcpy(out, s + 11, 64), out[64] = 0;
  return 1;
}

// ---------------------------------------------------------------- qwen35moe
// tools/moe2/pack_moe.py layout: physical heads, virtual layer offsets, experts
// retained in the source GGUF. Keep this branch separate from the dense format.
static Cfg moe_config(Gguf *g) {
  static const char *keys[][2] = {
      {"n_layers", "block_count"}, {"n_heads", "attention.head_count"},
      {"n_kv_heads", "attention.head_count_kv"}, {"d_model", "embedding_length"},
      {"head_dim", "attention.key_length"}, {"d_ff", "expert_feed_forward_length"},
      {"n_experts", "expert_count"}, {"n_experts_used", "expert_used_count"},
      {"vocab_size", NULL}, {"rope_freq_base", "rope.freq_base"},
      {"rope_dim", "rope.dimension_count"}, {"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
      {"full_attention_interval", "full_attention_interval"}, {"ssm_conv_kernel", "ssm.conv_kernel"},
      {"ssm_state_size", "ssm.state_size"}, {"ssm_group_count", "ssm.group_count"},
      {"ssm_time_step_rank", "ssm.time_step_rank"}, {"ssm_inner_size", "ssm.inner_size"}};
  Cfg c = {0};
  cfg_put(&c, "architecture", (Val){VS, 0, 0, "qwen35moe", NULL, 0});
  KV *tok = kv_get(g, "tokenizer.ggml.tokens");
  if (!tok || tok->t != 9) die("GGUF has no tokenizer.ggml.tokens array");
  for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
    char key[128];
    snprintf(key, sizeof key, "qwen35moe.%s", keys[i][1] ? keys[i][1] : "");
    cfg_put(&c, keys[i][0], keys[i][1] ? kvv(g, key, 1) : vint((int64_t)tok->an));
  }
  Val shared = kvv(g, "qwen35moe.expert_shared_feed_forward_length", 1);
  if ((shared.k != VI && shared.k != VU) || shared.i != cfg_int(&c, "d_ff"))
    die("qwen35moe: expert/shared feed-forward lengths must match");
  for (int i = 1; i < c.n; i++)
    if (strcmp(c.e[i].k, "rope_freq_base") && strcmp(c.e[i].k, "rms_norm_eps") && cfg_int(&c, c.e[i].k) <= 0)
      die("qwen35moe: %s must be positive", c.e[i].k);
  if (cfg_int(&c, "n_layers") > 4096 || cfg_int(&c, "n_experts_used") > cfg_int(&c, "n_experts"))
    die("qwen35moe: invalid layer/expert counts");
  return c;
}

static T *moe_tensor(Gguf *g, const char *name) {
  for (size_t i = 0; i < g->nt; i++)
    if (!strcmp(g->t[i].name, name)) return &g->t[i];
  die("qwen35moe: missing %s", name);
  return NULL;
}

static int source_order(const void *a, const void *b) {
  const T *x = a, *y = b;
  return x->src < y->src ? -1 : x->src > y->src;
}

static int moe_pack(Gguf *g, int fd, const char *src, const char *out) {
  static const char *heads[] = {"output.weight", "output_norm.weight", "token_embd.weight"};
  static const char *exps[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
  static const char *shared[] = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
  static const char *gates[] = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"};
  Cfg cfg = moe_config(g), shapes = cfg;
  int nl = (int)cfg_int(&cfg, "n_layers");
  int64_t ne = cfg_int(&cfg, "n_experts"), D = cfg_int(&cfg, "d_model"), F = cfg_int(&cfg, "d_ff");
  int64_t inner = cfg_int(&cfg, "ssm_inner_size"), tr = cfg_int(&cfg, "ssm_time_step_rank");
  int64_t kd = cfg_int(&cfg, "ssm_state_size") * cfg_int(&cfg, "ssm_group_count");
  cfg_put(&shapes, "linear_key_dim", vint(kd));
  cfg_put(&shapes, "linear_value_dim", vint(inner));
  cfg_put(&shapes, "linear_conv_dim", vint(2 * kd + inner));
  cfg_put(&shapes, "linear_value_head_dim", vint(inner / tr));
  // Validate the same trunk/shared shapes as dense, excluding the two routers.
  Gguf trunk = *g;
  trunk.t = xmalloc(g->nt * sizeof(T)), trunk.nt = 0;
  T *ts = xmalloc(g->nt * sizeof(T));
  size_t nt = 0;
  for (size_t i = 0; i < g->nt; i++)
    for (size_t j = 0; j < i; j++)
      if (!strcmp(g->t[i].name, g->t[j].name)) die("%s: duplicate tensor", g->t[i].name);
  for (int i = 0; i < 3; i++) ts[nt++] = *moe_tensor(g, heads[i]);
  Buf ix = {0};
  bs(&ix, "gguf "), bs(&ix, src), bs(&ix, "\n");
  if (strchr(src, '\n') || strchr(src, '\r')) die("qwen35moe: newline in GGUF path cannot be indexed");
  for (int L = 0; L < nl; L++) {
    size_t start = nt;
    for (size_t i = 0; i < g->nt; i++) {
      T t = g->t[i];
      if (t.layer != L) continue;
      const char *s = strchr(t.name + 4, '.') + 1;
      int expert = 0;
      for (int j = 0; j < 3; j++) expert |= !strcmp(s, exps[j]);
      if (expert) continue;
      for (int j = 0; j < 3; j++)
        if (!strcmp(s, shared[j])) {
          t.name = xmalloc(96);
          snprintf(t.name, 96, "blk.%d.%s", L, OFFS[2 + j]);
        }
      ts[nt++] = t;
    }
    qsort(ts + start, nt - start, sizeof(T), source_order);
    char name[128];
    for (int j = 0; j < 3; j++) {
      snprintf(name, sizeof name, "blk.%d.%s", L, shared[j]);
      moe_tensor(g, name);
    }
    for (int j = 0; j < 2; j++) {
      snprintf(name, sizeof name, "blk.%d.%s", L, gates[j]);
      T *t = moe_tensor(g, name);
      if (strcmp(t->tn, "F32") || (j ? t->nd != 1 || t->sh[0] != D :
          t->nd != 2 || t->sh[0] != ne || t->sh[1] != D))
        die("%s: invalid F32 router shape", name);
    }
    bf(&ix, "%d", L);
    for (int j = 0; j < 3; j++) {
      snprintf(name, sizeof name, "blk.%d.%s", L, exps[j]);
      T *t = moe_tensor(g, name);
      if (!t->f || t->f->elems != 256 || t->nd != 3 || t->sh[0] != ne ||
          t->sh[1] != (j == 2 ? D : F) || t->sh[2] != (j == 2 ? F : D) || t->sh[2] % 256)
        die("%s: invalid expert shape/type (need 256-element blocks)", name);
      bf(&ix, " %llu %d %llu", (unsigned long long)t->src, t->f->fm, (unsigned long long)(t->bytes / (uint64_t)ne));
    }
    bs(&ix, "\n");
  }
  if (nt + (size_t)nl * 3 != g->nt) die("qwen35moe: unexpected tensors outside heads/layers");
  for (size_t i = 0; i < nt; i++) {
    T *t = &ts[i];
    const char *s = t->layer < 0 ? "" : strchr(t->name + 4, '.') + 1;
    if (strcmp(s, gates[0]) && strcmp(s, gates[1])) trunk.t[trunk.nt++] = *t;
    if (strcmp(t->tn, "F32") && strcmp(t->tn, "Q8_0")) die("%s: %s outside the trunk doors (F32 / Q8_0)", t->name, t->tn);
    if (!strcmp(t->tn, "Q8_0")) {
      if (t->nd != 2 || t->sh[1] <= 0 || t->sh[1] % 32) die("%s: invalid Q8_0 row shape", t->name);
      t->rb = (uint64_t)t->sh[1] / 32 * 34;
    }
  }
  if (check_shapes(&trunk, &shapes)) die("qwen35moe shape check failed (above)");
  free(trunk.t);

  Layer *ly = xmalloc((size_t)nl * sizeof *ly);
  uint32_t *table = xmalloc((size_t)nl * 128);
  memset(table, 0, (size_t)nl * 128);
  for (int L = 0; L < nl; L++) ly[L] = (Layer){UINT64_MAX, 0, -1};
  uint64_t pos = 0, head_bytes = 0;
  size_t chunk = 16u << 20;
  for (size_t i = 0; i < nt; i++) {
    T *t = &ts[i];
    pos += (ALIGN - pos % ALIGN) % ALIGN;
    t->off = pos, pos += t->bytes;
    if (t->rb > chunk) chunk = t->rb;
    if (t->layer < 0) head_bytes = pos;
    else {
      Layer *l = &ly[t->layer];
      if (l->lo == UINT64_MAX) l->lo = t->off;
      l->hi = pos;
      const char *s = strchr(t->name + 4, '.') + 1;
      int j = 0;
      while (j < 22 && strcmp(s, j < 20 ? OFFS[j] : gates[j - 20])) j++;
      if (j == 22) die("%s: outside MoE table", t->name);
      if (pos - l->lo >= 1u << 28) die("layer %d: bytes >= 2^28", t->layer);
      table[t->layer * 32 + 1 + j] = (uint32_t)(t->off - l->lo) | (uint32_t)t->f->fm << 28;
      table[t->layer * 32] = (uint32_t)(l->hi - l->lo);
    }
  }
  char path[4096];
  if (strlen(out) > sizeof path - 64) die("output path too long");
  if (mkdir(out, 0755) && errno != EEXIST) die("mkdir %s: %s", out, strerror(errno));
  snprintf(path, sizeof path, "%s/manifest.json", out);
  if (!access(path, F_OK)) die("%s exists; refusing to overwrite a pack", path);
  snprintf(path, sizeof path, "%s/layers", out);
  if (mkdir(path, 0755) && errno != EEXIST) die("mkdir %s: %s", path, strerror(errno));
  Out o = {.head_only = 1, .cur_layer = -1, .nl = nl, .ly = ly};
  snprintf(path, sizeof path, "%s/pack.bin", out);
  o.pack_fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (o.pack_fd < 0) die("create %s: %s (refusing to overwrite)", path, strerror(errno));
  for (int L = 0; L < nl; L++) {
    snprintf(path, sizeof path, "%s/layers/blk.%d.bin", out, L);
    ly[L].fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (ly[L].fd < 0) die("create %s: %s (refusing to overwrite)", path, strerror(errno));
  }
  uint8_t *in = xmalloc(chunk), *tmp = xmalloc(chunk), zero[ALIGN] = {0};
  char pack_sha[65] = "";
  sha_init(&o.pack);
  double t0 = now_s();
  for (size_t i = 0; i < nt; i++) {
    T *t = &ts[i];
    o.cur_layer = t->layer;
    emit(&o, zero, (size_t)(t->off - o.pos), 0);
    sha_init(&o.ten);
    size_t unit = t->rb ? (size_t)t->rb : 4, step = chunk / unit * unit;
    for (uint64_t k = 0; k < t->bytes; k += step) {
      size_t n = t->bytes - k < step ? (size_t)(t->bytes - k) : step;
      if (pread_all(fd, in, n, t->src + k)) die("%s: read failed at %llu", t->name, (unsigned long long)(t->src + k));
      if (t->rb) {
        for (size_t b = 0; b < n; b += 34)
          if (!f16_finite(in + b)) die("%s: non-finite f16 scale in block %llu", t->name, (unsigned long long)((k + b) / 34));
        rowsplit(in, tmp, n, (size_t)t->sh[1], unit);
      }
      emit(&o, t->rb ? tmp : in, n, 1);
    }
    sha_hex(&o.ten, t->sha);
    if (i == 2) { Sha h = o.pack; sha_hex(&h, pack_sha); }
    if (i + 1 == nt || ts[i + 1].layer != t->layer)
      fprintf(stderr, "gguf2pack: qwen35moe %s complete, %.1f s\n", t->layer < 0 ? "heads" : t->name, now_s() - t0);
  }
  if (o.pos != pos) die("qwen35moe: internal virtual size mismatch");
  if (close(o.pack_fd)) die("close pack.bin: %s", strerror(errno));
  for (int L = 0; L < nl; L++)
    if (close(ly[L].fd)) die("close layer %d: %s", L, strerror(errno));
  Buf m = {0};
  bs(&m, "{\n \"format\": \"llama-bend-pack-q35-quant/v0\",\n \"alignment\": 64,\n \"dtype\": \"mixed-quant\",\n \"source\": {\n  \"file\": ");
  bjstr(&m, src, 1);
  bs(&m, "\n },\n \"config\": {");
  for (int i = 0; i < cfg.n; i++) {
    bs(&m, i ? ",\n  " : "\n  "), bjstr(&m, cfg.e[i].k, 0), bs(&m, ": ");
    bjval(&m, cfg.e[i].v, 2, 1);
  }
  bf(&m, "\n },\n \"layer_files\": %d,\n \"tensors\": {\n", nl);
  for (size_t i = 0; i < nt; i++) {
    if (i) bs(&m, ",\n");
    manifest_tensor(&m, &ts[i], 1, NULL, NULL); // MoE oracle omits fmt
  }
  bf(&m, "\n },\n \"pack_bytes\": %llu,\n \"pack_sha256\": \"%s\"\n}\n", (unsigned long long)head_bytes, pack_sha);
  write_file(out, "experts.idx", ix.p, ix.n);
  write_file(out, "table.bin", table, (size_t)nl * 128);
  write_file(out, "manifest.json", m.p, m.n);
  printf("qwen35moe: %zu tensors, head %llu bytes, %d layer files, pack_sha256 %s\nwrote %s\n",
         nt, (unsigned long long)head_bytes, nl, pack_sha, out);
  fprintf(stderr, "gguf2pack: %.1f s, VmHWM %ld kB\n", now_s() - t0, vm_hwm_kb());
  free(in), free(tmp), free(ts), free(ly), free(table), free(ix.p), free(m.p);
  return 0;
}

// ---------------------------------------------------------------- qwen35moe sidecar (CUDA GGUF load)
// The CUDA engine's direct GGUF load of any qwen35moe shape (tools/payload-moeany.patch): the
// sidecar only (manifest.json + table.bin), the weights stay in the GGUF. The virtual pack layout
// is host/llb_gguf.c gg_open's rule for a file with routed experts: the non-layer tensors, then
// each layer's trunk (every tensor but the *_exps stacks), file order within each, then every
// *_exps stack (layer order). A layer's range, the Bytes the engine keeps on the device, is its
// trunk only; the host loads the expert stacks itself (llb_host.c llb_moe_load: device, pinned
// host or streamed from the mmapped GGUF), by name and 64-bit file offset. table.bin rows:
// words 1..20 as dense with ffn_gate / ffn_up / ffn_down = the shared expert, then
//   21 ffn_gate_inp (router)   offset | fm << 28   (F32)
//   22 ffn_gate_inp_shexp      offset | fm << 28   (F32)
//   26 the expert formats: gate | up << 4 | down << 8, bit 12 fused gate+up, bit 13 set
//   27 expert_count, 28 expert ff, 29 shared ff, 30 expert_used_count
static int is_exps(const char *name) {
  size_t n = strlen(name);
  return n > 12 && !strcmp(name + n - 12, "_exps.weight");
}

// host/llb_gguf.c gg_open's order: a stable sort of the file order by (is_exps, layer), heads first
static void moe_order(T *t, size_t n) {
  for (size_t i = 1; i < n; i++) {
    T v = t[i];
    size_t j = i;
    int lv = v.layer, ev = lv >= 0 && is_exps(v.name);
    while (j > 0) {
      int lu = t[j - 1].layer, eu = lu >= 0 && is_exps(t[j - 1].name);
      if (eu < ev || (eu == ev && lu <= lv)) break;
      t[j] = t[j - 1], j--;
    }
    t[j] = v;
  }
}

static int moe_sidecar_cuda(Gguf *g, const char *src, const char *out) {
  static const char *fmn[] = {"Q8_0", "Q4_K", "Q5_K", "Q6_K"};
  Cfg cfg = build_config(g);
  int64_t L0 = cfg_int(&cfg, "n_layers"), D = cfg_int(&cfg, "d_model");
  Val vne = backbone_val(g, "expert_count", 1), vk = backbone_val(g, "expert_used_count", 1);
  Val vef = backbone_val(g, "expert_feed_forward_length", 1), vsf = backbone_val(g, "expert_shared_feed_forward_length", 1);
  Val vmt = backbone_val(g, "nextn_predict_layers", 0);
  int64_t ne = vne.i, nk = vk.i, EF = vef.i, SF = vsf.i, mtp = vmt.k == VN ? 0 : vmt.i;
  if (ne < 1 || ne > UINT32_MAX) die("qwen35moe: expert_count %lld", (long long)ne);
  if (nk < 1 || nk > ne) die("qwen35moe: expert_used_count %lld (1..expert_count)", (long long)nk);
  if (EF < 1 || SF < 1) die("qwen35moe: expert_feed_forward_length %lld / expert_shared_feed_forward_length %lld (> 0)",
                           (long long)EF, (long long)SF);
  if (mtp < 0 || mtp >= L0) die("qwen35moe: nextn_predict_layers %lld (0..block_count - 1)", (long long)mtp);
  cfg_put(&cfg, "n_experts", vint(ne));
  cfg_put(&cfg, "n_experts_used", vint(nk));
  cfg_put(&cfg, "expert_ff", vint(EF));
  cfg_put(&cfg, "shared_ff", vint(SF));
  cfg_put(&cfg, "nextn_predict_layers", vint(mtp));
  int nl = (int)(L0 - mtp);
  for (int i = 0; i < cfg.n; i++)
    if (!strcmp(cfg.e[i].k, "n_layers")) cfg.e[i].v = vint(nl);
  if (mtp) fprintf(stderr, "gguf2pack: qwen35moe: nextn_predict_layers %lld: blk.%d..%lld (MTP) skipped (plain decode)\n",
                   (long long)mtp, nl, (long long)L0 - 1);
  for (size_t i = 0; i < g->nt; i++)
    for (size_t j = 0; j < i; j++)
      if (!strcmp(g->t[i].name, g->t[j].name)) die("%s: duplicate tensor", g->t[i].name);
  // the virtual layout over every tensor (the MTP blocks too: gg_open sees them), then the plan
  T *ts = xmalloc(g->nt * sizeof(T));
  memcpy(ts, g->t, g->nt * sizeof(T));
  moe_order(ts, g->nt);
  uint64_t pos = 0;
  for (size_t i = 0; i < g->nt; i++) {
    pos += (ALIGN - pos % ALIGN) % ALIGN;
    ts[i].off = pos, pos += ts[i].bytes;
    if (ts[i].f && strcmp(ts[i].tn, "F32")) {
      int64_t cols = ts[i].sh[ts[i].nd - 1];
      if (cols % ts[i].f->elems) die("%s: cols %lld not a multiple of %d", ts[i].name, (long long)cols, ts[i].f->elems);
      ts[i].rb = (uint64_t)cols / (uint64_t)ts[i].f->elems * (uint64_t)ts[i].f->bytes;
    }
  }
  uint64_t pack_bytes = pos;
  // the trunk + shared expert under the dense names, for check_shapes at d_ff = shared ff
  Gguf tr = *g;
  tr.t = xmalloc(g->nt * sizeof(T)), tr.nt = 0;
  Cfg sc = cfg;
  for (int i = 0; i < sc.n; i++)
    if (!strcmp(sc.e[i].k, "d_ff")) sc.e[i].v = vint(SF);
  static const char *shn[3] = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
  // per layer: router, shexp gate, gate_exps, up_exps, gate_up_exps, down_exps
  T *(*rt)[6] = xmalloc((size_t)nl * sizeof *rt);
  memset(rt, 0, (size_t)nl * sizeof *rt);
  for (size_t i = 0; i < g->nt; i++) {
    T *t = &ts[i];
    if (t->layer >= nl) continue;
    if (t->layer < 0) { tr.t[tr.nt++] = *t; continue; }
    const char *s = strchr(t->name + 4, '.') + 1;
    static const char *mo[6] = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_exps.weight",
                                "ffn_up_exps.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight"};
    int j = 0;
    while (j < 6 && strcmp(s, mo[j])) j++;
    if (j < 6) { rt[t->layer][j] = t; continue; }
    T c = *t;
    for (int k = 0; k < 3; k++)
      if (!strcmp(s, shn[k])) {
        c.name = xmalloc(96);
        snprintf(c.name, 96, "blk.%d.%s", t->layer, OFFS[2 + k]);
      }
    tr.t[tr.nt++] = c;
  }
  if (check_shapes(&tr, &sc)) die("qwen35moe shape check failed (above; ffn_gate/up/down = the *_shexp tensors)");
  // the routed experts and the two gates, refused by name
  for (int L = 0; L < nl; L++) {
    T **r = rt[L];
    char nm[128];
#define NEED(j, what) do { if (!r[j]) { snprintf(nm, sizeof nm, "blk.%d.%s", L, what); die("qwen35moe: missing %s", nm); } } while (0)
    NEED(0, "ffn_gate_inp.weight");
    NEED(1, "ffn_gate_inp_shexp.weight");
    NEED(5, "ffn_down_exps.weight");
    if (!r[4]) {
      NEED(2, "ffn_gate_exps.weight (or ffn_gate_up_exps.weight)");
      NEED(3, "ffn_up_exps.weight");
    }
    if (r[4] && (r[2] || r[3])) die("blk.%d: both ffn_gate_up_exps and ffn_gate_exps / ffn_up_exps", L);
#undef NEED
    if (strcmp(r[0]->tn, "F32") || r[0]->nd != 2 || r[0]->sh[0] != ne || r[0]->sh[1] != D)
      die("%s: %s [%lld x %lld]: the CUDA router needs F32 [expert_count %lld x d_model %lld]", r[0]->name, r[0]->tn,
          (long long)r[0]->sh[0], (long long)(r[0]->nd > 1 ? r[0]->sh[1] : 0), (long long)ne, (long long)D);
    if (strcmp(r[1]->tn, "F32") || !((r[1]->nd == 1 && r[1]->sh[0] == D) || (r[1]->nd == 2 && r[1]->sh[0] == 1 && r[1]->sh[1] == D)))
      die("%s: %s: the CUDA shared-expert gate needs F32 [d_model %lld]", r[1]->name, r[1]->tn, (long long)D);
    for (int j = 2; j < 6; j++) {
      T *t = r[j];
      if (!t) continue;
      int64_t rows = j == 5 ? D : j == 4 ? 2 * EF : EF, cols = j == 5 ? EF : D;
      if (t->nd != 3 || t->sh[0] != ne || t->sh[1] != rows || t->sh[2] != cols)
        die("%s: shape [%lld, %lld, %lld], want [expert_count %lld, %lld, %lld]", t->name, (long long)t->sh[0],
            (long long)(t->nd > 1 ? t->sh[1] : 0), (long long)(t->nd > 2 ? t->sh[2] : 0), (long long)ne, (long long)rows, (long long)cols);
      int ok = 0;
      for (int k = 0; k < 4; k++) ok |= !strcmp(t->tn, fmn[k]);
      if (!ok) die("%s: type %s: the CUDA expert kernels take Q8_0, Q4_K, Q5_K, Q6_K", t->name, t->tn);
    }
  }
  // layer ranges and table rows
  Layer *ly = xmalloc((size_t)nl * sizeof *ly);
  for (int i = 0; i < nl; i++) ly[i].lo = UINT64_MAX, ly[i].hi = 0, ly[i].fd = -1;
  for (size_t i = 0; i < g->nt; i++) {
    T *t = &ts[i];
    if (t->layer < 0 || t->layer >= nl || is_exps(t->name)) continue;
    Layer *l = &ly[t->layer];
    if (t->off < l->lo) l->lo = t->off;
    if (t->off + t->bytes > l->hi) l->hi = t->off + t->bytes;
  }
  uint32_t *table = xmalloc((size_t)nl * 128);
  memset(table, 0, (size_t)nl * 128);
  for (int L = 0; L < nl; L++) {
    uint64_t n = ly[L].hi - ly[L].lo;
    if (n > (1ull << 31))
      die("layer %d: trunk %llu bytes > 2 GiB (the engine's per-layer Bytes)", L, (unsigned long long)n);
    table[L * 32] = (uint32_t)n;
  }
  for (size_t i = 0; i < g->nt; i++) {
    T *t = &ts[i];
    if (t->layer < 0 || t->layer >= nl) continue;
    const char *s = strchr(t->name + 4, '.') + 1;
    uint64_t rel = t->off - ly[t->layer].lo;
    uint32_t *w = table + t->layer * 32;
    int j = 0;
    while (j < 20 && strcmp(OFFS[j], s)) j++;
    for (int k = 0; k < 3 && j == 20; k++)
      if (!strcmp(s, shn[k])) j = 2 + k;
    if (j < 20 || !strcmp(s, "ffn_gate_inp.weight") || !strcmp(s, "ffn_gate_inp_shexp.weight")) {
      if (j == 20) j = !strcmp(s, "ffn_gate_inp.weight") ? 20 : 21;
      if (rel >= 1u << 28) die("%s: offset %llu in layer >= 2^28", t->name, (unsigned long long)rel);
      w[1 + j] = (uint32_t)rel | (uint32_t)t->f->fm << 28;
      continue;
    }
    int x = !strcmp(s, "ffn_gate_exps.weight") || !strcmp(s, "ffn_gate_up_exps.weight") ? 0
      : !strcmp(s, "ffn_up_exps.weight") ? 1 : !strcmp(s, "ffn_down_exps.weight") ? 2 : -1;
    if (x < 0) die("%s: tensor outside the MoE table", t->name);
    (void)rel;  // the stacks are outside the layer's range: the host finds them by name
    w[26] |= (uint32_t)t->f->fm << (4 * x) | (!strcmp(s, "ffn_gate_up_exps.weight") ? 1u << 12 : 0) | 1u << 13;
  }
  for (int L = 0; L < nl; L++) {
    uint32_t *w = table + L * 32;
    if (w[26] >> 12 & 1) w[26] |= (w[26] & 15) << 4;  // fused: up has gate's format
    w[27] = (uint32_t)ne, w[28] = (uint32_t)EF, w[29] = (uint32_t)SF, w[30] = (uint32_t)nk;
  }
  // outputs: manifest.json (the dense sidecar's format; the tensors in virtual order) + table.bin
  int allq8 = 1;
  for (size_t i = 0; i < g->nt; i++) allq8 &= !strcmp(ts[i].tn, "Q8_0") || !strcmp(ts[i].tn, "F32");
  char p[4096];
  if (mkdir(out, 0755) && errno != EEXIST) die("mkdir %s: %s", out, strerror(errno));
  snprintf(p, sizeof p, "%s/manifest.json", out);
  if (!access(p, F_OK)) die("%s exists; refusing to overwrite a pack", p);
  Buf m = {0};
  bf(&m, "{\n \"format\": \"%s\",\n \"alignment\": %d,\n \"dtype\": \"%s\",\n \"source\": {\n  \"file\": ",
     allq8 ? "llama-bend-pack-q8/v0" : "llama-bend-pack-q35-quant/v0", ALIGN, allq8 ? "q8_0-rowsplit" : "mixed-quant");
  bjstr(&m, src, 1);
  bs(&m, "\n },\n \"config\": {");
  for (int i = 0; i < cfg.n; i++) {
    bs(&m, i ? ",\n  " : "\n  "), bjstr(&m, cfg.e[i].k, 0), bs(&m, ": ");
    bjval(&m, cfg.e[i].v, 2, 1);
  }
  bs(&m, "\n },\n \"tensors\": {\n");
  T *embd = NULL;
  for (size_t i = 0; i < g->nt; i++) {
    if (i) bs(&m, ",\n");
    manifest_tensor(&m, &ts[i], 0, NULL, NULL);
    if (!strcmp(ts[i].name, "token_embd.weight")) embd = &ts[i];
  }
  if (cfg_get(&cfg, "tied_embeddings").i) bs(&m, ",\n"), manifest_tensor(&m, embd, 0, "token_embd.weight", "output.weight");
  bf(&m, "\n },\n \"pack_bytes\": %llu,\n \"pack_sha256\": \"\"\n}\n", (unsigned long long)pack_bytes);
  write_file(out, "table.bin", table, (size_t)nl * 128);
  write_file(out, "manifest.json", m.p, m.n);
  printf("sidecar %s: manifest.json + table.bin for %s (qwen35moe, %d layers, %lld experts / %lld used, expert ff %lld, "
         "shared ff %lld, pack_bytes %llu)\n", out, src, nl, (long long)ne, (long long)nk, (long long)EF, (long long)SF,
         (unsigned long long)pack_bytes);
  free(ts), free(tr.t), free(rt), free(ly), free(table), free(m.p);
  return 0;
}

// ---------------------------------------------------------------- qwen35moe sidecar (Metal GGUF load)
// MODEL.gguf --sidecar-only OUT_DIR --target metal, any qwen35moe shape, for the Metal engine's general MoE path
// (backends/metal, METAL-MOE; the CUDA sidecar above is the default and has another layout): manifest.json + table.bin only. The engine reads
// the trunk from the GGUF (LLB_GGUF, host/llb_gguf.c's MoE virtual layout: heads, then each layer's non-expert
// tensors in source order, 64-aligned) and maps the routed experts from the same file itself, so nothing else
// is written. Unlike the registered 35B pack above: the trunk may be F32 / Q8_0 / Q4_K / Q5_K / Q6_K, the
// shared expert's width may differ from the routed experts', the experts may be fused (ffn_gate_up_exps, gate
// rows first), and the NextN (MTP) blocks are dropped. The config says "experts": "gguf" (the general MoE path).
static int moe_known_expert(const char *tn) {
  static const char *ok[] = {"F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K"};
  for (size_t i = 0; i < sizeof ok / sizeof *ok; i++)
    if (!strcmp(ok[i], tn)) return 1;
  return 0;
}

static int moe_sidecar_metal(Gguf *g, const char *src, const char *out) {
  static const char *heads[] = {"output.weight", "output_norm.weight", "token_embd.weight"};
  static const char *shared[] = {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
  static const char *gates[] = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"};
  static const char *keys[][2] = {
      {"n_layers", "block_count"}, {"n_heads", "attention.head_count"},
      {"n_kv_heads", "attention.head_count_kv"}, {"d_model", "embedding_length"},
      {"head_dim", "attention.key_length"}, {"d_ff", "expert_feed_forward_length"},
      {"d_ff_shared", "expert_shared_feed_forward_length"},
      {"n_experts", "expert_count"}, {"n_experts_used", "expert_used_count"},
      {"vocab_size", NULL}, {"rope_freq_base", "rope.freq_base"},
      {"rope_dim", "rope.dimension_count"}, {"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
      {"full_attention_interval", "full_attention_interval"}, {"ssm_conv_kernel", "ssm.conv_kernel"},
      {"ssm_state_size", "ssm.state_size"}, {"ssm_group_count", "ssm.group_count"},
      {"ssm_time_step_rank", "ssm.time_step_rank"}, {"ssm_inner_size", "ssm.inner_size"}};
  Cfg cfg = {0};
  cfg_put(&cfg, "architecture", (Val){VS, 0, 0, "qwen35moe", NULL, 0});
  KV *tok = kv_get(g, "tokenizer.ggml.tokens");
  if (!tok || tok->t != 9) die("GGUF has no tokenizer.ggml.tokens array");
  for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
    char key[128];
    snprintf(key, sizeof key, "qwen35moe.%s", keys[i][1] ? keys[i][1] : "");
    Val v = keys[i][1] ? kvv(g, key, 0) : vint((int64_t)tok->an);
    if (v.k == VN) die("qwen35moe: GGUF has no %s (the general MoE path needs it)", key);
    cfg_put(&cfg, keys[i][0], v);
  }
  // the routing the engine runs: softmax over every expert, top-k, renormalized, no scale. Refuse files that
  // declare anything else (llama.cpp's expert_weights_scale / expert_weights_norm / expert_gating_func keys)
  Val sc = kvv(g, "qwen35moe.expert_weights_scale", 0), nw = kvv(g, "qwen35moe.expert_weights_norm", 0);
  Val gf = kvv(g, "qwen35moe.expert_gating_func", 0);
  if (sc.k == VF && sc.f != 0.0 && sc.f != 1.0) die("qwen35moe: expert_weights_scale %g (the engine applies none)", sc.f);
  if (nw.k == VB && !nw.i) die("qwen35moe: expert_weights_norm false (the engine renormalizes the top-k weights)");
  if ((gf.k == VI || gf.k == VU) && gf.i != 1) die("qwen35moe: expert_gating_func %lld (the engine's router is softmax, 1)", (long long)gf.i);
  for (int i = 1; i < cfg.n; i++)
    if (strcmp(cfg.e[i].k, "rope_freq_base") && strcmp(cfg.e[i].k, "rms_norm_eps") && cfg_int(&cfg, cfg.e[i].k) <= 0)
      die("qwen35moe: %s must be positive", cfg.e[i].k);
  int64_t L0 = cfg_int(&cfg, "n_layers");
  Val mtp = kvv(g, "qwen35moe.nextn_predict_layers", 0);
  int64_t nmtp = mtp.k == VI || mtp.k == VU ? mtp.i : 0;
  if (nmtp < 0 || nmtp >= L0) die("qwen35moe: nextn_predict_layers %lld of %lld blocks", (long long)nmtp, (long long)L0);
  int nl = (int)(L0 - nmtp);
  for (int i = 0; i < cfg.n; i++)
    if (!strcmp(cfg.e[i].k, "n_layers")) cfg.e[i].v = vint(nl);
  if (nmtp) fprintf(stderr, "gguf2pack: qwen35moe: %lld NextN (MTP) block(s) dropped (blk.%d..; plain decode)\n", (long long)nmtp, nl);
  if (nl > 4096 || cfg_int(&cfg, "n_experts_used") > cfg_int(&cfg, "n_experts")) die("qwen35moe: invalid layer/expert counts");
  cfg_put(&cfg, "experts", (Val){VS, 0, 0, "gguf", NULL, 0});
  int64_t ne = cfg_int(&cfg, "n_experts"), D = cfg_int(&cfg, "d_model"), F = cfg_int(&cfg, "d_ff");
  int64_t FS = cfg_int(&cfg, "d_ff_shared");
  Cfg shapes = cfg;
  int64_t inner = cfg_int(&cfg, "ssm_inner_size"), tr = cfg_int(&cfg, "ssm_time_step_rank");
  int64_t kd = cfg_int(&cfg, "ssm_state_size") * cfg_int(&cfg, "ssm_group_count");
  for (int i = 0; i < shapes.n; i++)
    if (!strcmp(shapes.e[i].k, "d_ff")) shapes.e[i].v = vint(FS);  // the trunk's ffn_* are the shared expert
  cfg_put(&shapes, "linear_key_dim", vint(kd));
  cfg_put(&shapes, "linear_value_dim", vint(inner));
  cfg_put(&shapes, "linear_conv_dim", vint(2 * kd + inner));
  cfg_put(&shapes, "linear_value_head_dim", vint(inner / tr));
  for (size_t i = 0; i < g->nt; i++)
    for (size_t j = 0; j < i; j++)
      if (!strcmp(g->t[i].name, g->t[j].name)) die("%s: duplicate tensor", g->t[i].name);
  T *ts = xmalloc(g->nt * sizeof(T));
  size_t nt = 0, nexp = 0, nmtpt = 0;
  int *fz = xmalloc((size_t)nl * sizeof *fz);
  for (int i = 0; i < 3; i++) ts[nt++] = *moe_tensor(g, heads[i]);
  for (int L = 0; L < nl; L++) {
    size_t start = nt;
    char name[128];
    snprintf(name, sizeof name, "blk.%d.ffn_gate_up_exps.weight", L);
    int fused = fz[L] = has_tensor(g, name);
    for (size_t i = 0; i < g->nt; i++) {
      T t = g->t[i];
      if (t.layer != L) continue;
      const char *s = strchr(t.name + 4, '.') + 1;
      if (!strcmp(s, "ffn_gate_exps.weight") || !strcmp(s, "ffn_up_exps.weight") ||
          !strcmp(s, "ffn_down_exps.weight") || !strcmp(s, "ffn_gate_up_exps.weight")) {
        int dn = !strcmp(s, "ffn_down_exps.weight"), gu = !strcmp(s, "ffn_gate_up_exps.weight");
        if (fused && !dn && !gu) die("%s: separate gate/up experts beside ffn_gate_up_exps", t.name);
        if (!moe_known_expert(t.tn))
          die("%s: expert type %s (the general MoE path reads F32 F16 BF16 Q8_0 Q4_K Q5_K Q6_K experts)", t.name, t.tn);
        int64_t rows = dn ? D : gu ? 2 * F : F, cols = dn ? F : D;
        int el = t.f ? t.f->elems : 1;
        if (t.nd != 3 || t.sh[0] != ne || t.sh[1] != rows || t.sh[2] != cols || cols % el || cols % 32)
          die("%s: shape [%lld, %lld, %lld], want [%lld, %lld, %lld] (whole %d-element blocks a row)", t.name,
              (long long)t.sh[0], (long long)t.sh[1], (long long)(t.nd > 2 ? t.sh[2] : 0), (long long)ne,
              (long long)rows, (long long)cols, el);
        nexp++;
        continue;
      }
      for (int j = 0; j < 3; j++)
        if (!strcmp(s, shared[j])) {
          t.name = xmalloc(96);
          snprintf(t.name, 96, "blk.%d.%s", L, OFFS[2 + j]);
        }
      ts[nt++] = t;
    }
    qsort(ts + start, nt - start, sizeof(T), source_order);
    for (int j = 0; j < 3; j++) {
      snprintf(name, sizeof name, "blk.%d.%s", L, shared[j]);
      if (!has_tensor(g, name)) die("qwen35moe: %s missing (the general MoE path needs the shared expert)", name);
    }
    for (int j = 0; j < 2; j++) {
      snprintf(name, sizeof name, "blk.%d.%s", L, gates[j]);
      T *t = moe_tensor(g, name);
      if (strcmp(t->tn, "F32") || (j ? t->nd != 1 || t->sh[0] != D : t->nd != 2 || t->sh[0] != ne || t->sh[1] != D))
        die("%s: the router must be F32 [%lld, %lld] (shared gate [%lld])", name, (long long)ne, (long long)D, (long long)D);
    }
  }
  for (size_t i = 0; i < g->nt; i++) nmtpt += g->t[i].layer >= nl;
  if (nt + nexp + nmtpt != g->nt) die("qwen35moe: unexpected tensors outside heads/layers");
  Gguf trunk = *g;
  trunk.t = xmalloc(g->nt * sizeof(T)), trunk.nt = 0;
  for (size_t i = 0; i < nt; i++) {
    T *t = &ts[i];
    const char *s = t->layer < 0 ? "" : strchr(t->name + 4, '.') + 1;
    if (strcmp(s, gates[0]) && strcmp(s, gates[1])) trunk.t[trunk.nt++] = *t;
    if (strcmp(t->tn, "F32") && strcmp(t->tn, "Q8_0") && strcmp(t->tn, "Q4_K") && strcmp(t->tn, "Q5_K") && strcmp(t->tn, "Q6_K"))
      die("%s: %s outside the trunk formats (F32 Q8_0 Q4_K Q5_K Q6_K; requantize, e.g. BF16 -> Q8_0)", t->name, t->tn);
    if (strcmp(t->tn, "F32")) {
      if (t->nd != 2 || t->sh[1] % t->f->elems) die("%s: %s needs 2-D whole blocks", t->name, t->tn);
      t->rb = (uint64_t)t->sh[1] / (uint64_t)t->f->elems * (uint64_t)t->f->bytes;
    }
  }
  if (check_shapes(&trunk, &shapes)) die("qwen35moe shape check failed (above)");
  free(trunk.t);
  uint32_t *table = xmalloc((size_t)nl * 128);
  memset(table, 0, (size_t)nl * 128);
  uint64_t pos = 0, lo = 0;
  int cur = -1;
  for (size_t i = 0; i < nt; i++) {
    T *t = &ts[i];
    pos += (ALIGN - pos % ALIGN) % ALIGN;
    t->off = pos, pos += t->bytes;
    if (t->layer < 0) continue;
    if (t->layer != cur) cur = t->layer, lo = t->off;
    const char *s = strchr(t->name + 4, '.') + 1;
    int j = 0;
    while (j < 22 && strcmp(s, j < 20 ? OFFS[j] : gates[j - 20])) j++;
    if (j == 22) die("%s: outside the MoE table", t->name);
    if (pos - lo >= 1u << 28) die("layer %d: bytes >= 2^28", t->layer);
    table[t->layer * 32 + 1 + j] = (uint32_t)(t->off - lo) | (uint32_t)t->f->fm << 28;
    table[t->layer * 32] = (uint32_t)(pos - lo);
  }
  // the MoE words the host reads (every row): 24 "MXG1", 25 n_experts, 26 top-k, 27 expert ff, 28 shared ff,
  // 29 bit 0 a fused ffn_gate_up_exps
  int64_t K = cfg_int(&cfg, "n_experts_used");
  if (ne > UINT32_MAX || F > UINT32_MAX || FS > UINT32_MAX) die("qwen35moe: n_experts / expert ff / shared ff above 2^32");
  for (int L = 0; L < nl; L++) {
    table[L * 32 + 24] = 0x3147584Du;
    table[L * 32 + 25] = (uint32_t)ne;
    table[L * 32 + 26] = (uint32_t)K;
    table[L * 32 + 27] = (uint32_t)F;
    table[L * 32 + 28] = (uint32_t)FS;
    table[L * 32 + 29] = (uint32_t)fz[L];
  }
  free(fz);
  char path[4096];
  if (strlen(out) > sizeof path - 64) die("output path too long");
  if (mkdir(out, 0755) && errno != EEXIST) die("mkdir %s: %s", out, strerror(errno));
  snprintf(path, sizeof path, "%s/manifest.json", out);
  if (!access(path, F_OK)) die("%s exists; refusing to overwrite a pack", path);
  Buf m = {0};
  bs(&m, "{\n \"format\": \"llama-bend-pack-q35-quant/v0\",\n \"alignment\": 64,\n \"dtype\": \"mixed-quant\",\n \"source\": {\n  \"file\": ");
  bjstr(&m, src, 1);
  bs(&m, "\n },\n \"config\": {");
  for (int i = 0; i < cfg.n; i++) {
    bs(&m, i ? ",\n  " : "\n  "), bjstr(&m, cfg.e[i].k, 0), bs(&m, ": ");
    bjval(&m, cfg.e[i].v, 2, 1);
  }
  bs(&m, "\n },\n \"tensors\": {\n");
  for (size_t i = 0; i < nt; i++) {
    if (i) bs(&m, ",\n");
    manifest_tensor(&m, &ts[i], 0, NULL, NULL);
  }
  bf(&m, "\n },\n \"pack_bytes\": %llu,\n \"pack_sha256\": \"\"\n}\n", (unsigned long long)pos);
  write_file(out, "table.bin", table, (size_t)nl * 128);
  write_file(out, "manifest.json", m.p, m.n);
  printf("sidecar %s: manifest.json + table.bin for %s (qwen35moe, %d layers, %lld experts top %lld, expert ff %lld,"
         " shared ff %lld, %zu trunk tensors, pack_bytes %llu)\n", out, src, nl, (long long)ne,
         (long long)cfg_int(&cfg, "n_experts_used"), (long long)F, (long long)FS, nt, (unsigned long long)pos);
  free(ts), free(table), free(m.p);
  return 0;
}

// ---------------------------------------------------------------- main

static void usage(void) {
  fputs("usage: gguf2pack MODEL.gguf OUT_DIR [--format q8|quant] [--head-only]\n"
        "       gguf2pack MODEL.gguf --verify-against PACK_DIR [--format q8|quant]\n"
        "       gguf2pack MODEL.gguf --sidecar-only OUT_DIR [--sha] [--format q8|quant] [--target cuda|metal]\n"
        "       gguf2pack MODEL.gguf --probe\n", stderr);
  exit(2);
}

int main(int argc, char **argv) {
  uint16_t one = 1;
  if (*(uint8_t *)&one != 1) die("big-endian host: GGUF and the pack are little-endian; unsupported");
  const char *src = NULL, *out = NULL, *vdir = NULL, *fmtarg = NULL;
  int probe = 0, head_only = 0, sidecar = 0, shas = 0, metal = 0;
  const char *target = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--probe")) probe = 1;
    else if (!strcmp(argv[i], "--head-only")) head_only = 1;
    else if (!strcmp(argv[i], "--sidecar-only")) sidecar = 1;
    else if (!strcmp(argv[i], "--sha")) shas = 1;
    else if (!strcmp(argv[i], "--verify-against") && i + 1 < argc) vdir = argv[++i];
    else if (!strcmp(argv[i], "--format") && i + 1 < argc) fmtarg = argv[++i];
    else if (!strcmp(argv[i], "--target") && i + 1 < argc) target = argv[++i];
    else if (argv[i][0] == '-' && argv[i][1] == '-') usage();
    else if (!src) src = argv[i];
    else if (!out) out = argv[i];
    else usage();
  }
  if (!src || (probe + !!vdir + !!out) != 1 || (head_only && !out) || (sidecar && (!out || head_only)) || (shas && !sidecar)) usage();
  if (fmtarg && strcmp(fmtarg, "q8") && strcmp(fmtarg, "quant")) usage();
  // --target: whose qwen35moe sidecar (cuda, the default, or metal); a dense sidecar is the same for both
  if (target && (!sidecar || (strcmp(target, "cuda") && strcmp(target, "metal")))) usage();
  metal = target && !strcmp(target, "metal");

  int fd = open(src, O_RDONLY);
  if (fd < 0) die("open %s: %s", src, strerror(errno));
  Gguf g = gguf_read(fd);

  if (probe) {
    KV *a = kv_get(&g, "general.architecture");
    printf("GGUF v%u, %zu tensors, %zu kv, alignment %llu, data at %llu, file %llu B, architecture %s\n",
           g.version, g.nt, g.nkv, (unsigned long long)g.align, (unsigned long long)g.data,
           (unsigned long long)g.fsize, a && a->v.k == VS ? a->v.s : "?");
    printf("# name\ttype\tshape\tgguf_offset\tbytes\n");
    for (size_t i = 0; i < g.nt; i++) {
      T *t = &g.t[i];
      printf("%s\t%s\t", t->name, t->tn);
      for (int j = 0; j < t->nd; j++) printf(j ? "x%lld" : "%lld", (long long)t->sh[j]);
      printf("\t%llu\t%llu\n", (unsigned long long)t->src, (unsigned long long)t->bytes);
    }
    return 0;
  }

  Val arch = kvv(&g, "general.architecture", 1);
  if (arch.k == VS && !strcmp(arch.s, "qwen35moe")) {
    // --sidecar-only: the CUDA engine's direct GGUF load, any qwen35moe shape (moe_sidecar_cuda); with
    // --target metal the Metal engine's (moe_sidecar_metal: experts left out of the layout, MoE words 24 .. 29)
    if (sidecar && !shas && !fmtarg) return metal ? moe_sidecar_metal(&g, src, out) : moe_sidecar_cuda(&g, src, out);
    if (sidecar || head_only || vdir || (fmtarg && strcmp(fmtarg, "quant")))
      die("qwen35moe: use MODEL.gguf OUT_DIR [--format quant], or MODEL.gguf --sidecar-only OUT_DIR; other dense flags are unsupported");
    return moe_pack(&g, fd, src, out);
  }
  Cfg cfg = build_config(&g);
  int clef = arch.k == VS && !strcmp(arch.s, "clef");
  uint64_t clef_virtual_bytes = 0;
  if (clef && sidecar) {
    // Linux 0.1.2's GGUF loader derives virtual offsets from ALL tensors.
    // Save that layout before filtering; a compact sidecar would misaddress
    // output/embedding bytes. Full packs below still use compact offsets.
    if (shas) die("clef --sidecar-only --sha is unsupported: skipped head ranges have no pack checksum; omit --sha");
    for (size_t i = 0; i < g.nt; i++) {
      clef_virtual_bytes += (ALIGN - clef_virtual_bytes % ALIGN) % ALIGN;
      g.t[i].off = clef_virtual_bytes;
      clef_virtual_bytes += g.t[i].bytes;
    }
  }
  if (clef) drop_clef_head(&g);
  drop_mtp_block(&g, &cfg);  // finetunes may carry the NextN block; skip it
  if (check_shapes(&g, &cfg)) die("shape check failed (above)");
  int allq8 = 1;
  for (size_t i = 0; i < g.nt; i++) {
    T *t = &g.t[i];
    if (t->door < 0) die("%s: type %s unsupported for packing (doors F32 Q8_0 Q4_K Q5_K Q6_K Q3_K Q2_0 PQ2_0)", t->name, t->tn);
    allq8 &= !strcmp(t->tn, "Q8_0") || !strcmp(t->tn, "F32");
  }
  int q8 = fmtarg ? !strcmp(fmtarg, "q8") : allq8;
  if (q8 && !allq8) die("--format q8 takes Q8_0 + F32 only");

  // plan: pack offsets, row bytes, layer ranges, table rows
  int nl = (int)cfg_int(&cfg, "n_layers");
  Layer *ly = xmalloc((size_t)nl * sizeof *ly);
  for (int i = 0; i < nl; i++) ly[i].lo = UINT64_MAX, ly[i].hi = 0, ly[i].fd = -1;
  uint64_t pos = 0;
  size_t maxrb = 0;
  for (size_t i = 0; i < g.nt; i++) {
    T *t = &g.t[i];
    if (clef_virtual_bytes) pos = t->off;
    else {
      pos += (ALIGN - pos % ALIGN) % ALIGN;
      t->off = pos;
    }
    pos += t->bytes;
    if (strcmp(t->tn, "F32")) {
      if (t->nd != 2) die("%s: %s needs a 2-D shape", t->name, t->tn);
      if (t->sh[1] % t->f->elems) die("%s: cols %lld not a multiple of %d", t->name, (long long)t->sh[1], t->f->elems);
      t->rb = (uint64_t)t->sh[1] / (uint64_t)t->f->elems * (uint64_t)t->f->bytes;
      if (t->rb > maxrb) maxrb = t->rb;
    }
    if (t->layer >= 0) {
      Layer *l = &ly[t->layer];
      if (t->off < l->lo) l->lo = t->off;
      if (t->off + t->bytes > l->hi) l->hi = t->off + t->bytes;
    }
  }
  uint64_t pack_bytes = clef_virtual_bytes ? clef_virtual_bytes : pos;
  uint32_t *table = xmalloc((size_t)nl * 128);
  memset(table, 0, (size_t)nl * 128);
  for (int L = 0; L < nl; L++) {
    table[L * 32] = (uint32_t)(ly[L].hi - ly[L].lo);
    if (ly[L].hi - ly[L].lo > UINT32_MAX) die("layer %d: %llu bytes > u32", L, (unsigned long long)(ly[L].hi - ly[L].lo));
  }
  for (size_t i = 0; i < g.nt; i++) {
    T *t = &g.t[i];
    if (t->layer < 0) continue;
    const char *sfx = strchr(t->name + 4, '.') + 1;
    int j = 0;
    while (j < 20 && strcmp(OFFS[j], sfx)) j++;
    if (j == 20) die("%s: tensor outside table.bin's 20 (pack_table.OFFS)", t->name);
    uint64_t rel = t->off - ly[t->layer].lo;
    if (rel >= 1u << 28) die("%s: offset %llu in layer >= 2^28", t->name, (unsigned long long)rel);
    table[t->layer * 32 + 1 + j] = (uint32_t)rel | (uint32_t)t->f->fm << 28;
  }

  // outputs
  Out o = {.pack_fd = -1, .head_only = head_only, .cur_layer = -1, .nl = nl, .ly = ly};
  sha_init(&o.pack);
  if (out) {
    char p[4096];
    if (mkdir(out, 0755) && errno != EEXIST) die("mkdir %s: %s", out, strerror(errno));
    snprintf(p, sizeof p, "%s/manifest.json", out);
    if (!access(p, F_OK)) die("%s exists; refusing to overwrite a pack", p);
    if (sidecar) goto made;
    snprintf(p, sizeof p, "%s/layers", out);
    if (mkdir(p, 0755) && errno != EEXIST) die("mkdir %s: %s", p, strerror(errno));
    snprintf(p, sizeof p, "%s/pack.bin", out);
    o.pack_fd = open(p, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (o.pack_fd < 0) die("create %s: %s (refusing to overwrite)", p, strerror(errno));
    for (int L = 0; L < nl; L++) {
      snprintf(p, sizeof p, "%s/layers/blk.%d.bin", out, L);
      if ((ly[L].fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644)) < 0) die("create %s: %s", p, strerror(errno));
    }
  made:;
  }
#ifdef POSIX_FADV_SEQUENTIAL
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
  size_t CH = 16u << 20;
  if (maxrb * 2 > CH) CH = maxrb * 2;
  uint8_t *in = xmalloc(CH), *tmp = xmalloc(CH), zero[ALIGN] = {0};
  uint64_t total = 0, done = 0, next = 1ull << 30;
  for (size_t i = 0; i < g.nt; i++) total += g.t[i].bytes;
  double t0 = now_s();
  fprintf(stderr, "gguf2pack: %s -> %s, format %s, %zu tensors, %.2f GB%s\n", src, out ? out : "(verify, nothing written)",
          q8 ? "llama-bend-pack-q8/v0" : "llama-bend-pack-q35-quant/v0", g.nt, total / 1e9, head_only ? ", head-only pack.bin" : "");
  if (sidecar && !shas) o.pos = pack_bytes;  // header only: no weight is read, the sha fields stay empty
  for (size_t i = 0; i < g.nt && !(sidecar && !shas); i++) {
    T *t = &g.t[i];
    o.cur_layer = t->layer;
    emit(&o, zero, t->off - o.pos, 0);
    sha_init(&o.ten);
    int q8t = !strcmp(t->tn, "Q8_0"), d0 = DOORS[t->door].d0, d1 = DOORS[t->door].d1;
    size_t bb = (size_t)t->f->bytes;
    // chunk = whole rows for Q8_0 (row-split is per row), whole blocks otherwise
    size_t unit = q8t ? (size_t)t->rb : bb, step = CH / unit * unit;
    for (uint64_t k = 0; k < t->bytes; k += step) {
      size_t n = t->bytes - k < step ? (size_t)(t->bytes - k) : step;
      if (pread_all(fd, in, n, t->src + k)) die("%s: read failed at %llu: %s", t->name, (unsigned long long)(t->src + k), strerror(errno));
      if (d0 >= 0)
        for (size_t b = 0; b < n; b += bb)
          if (!f16_finite(in + b + d0) || (d1 >= 0 && !f16_finite(in + b + d1)))
            die("%s: non-finite f16 scale in block %llu", t->name, (unsigned long long)((k + b) / bb));
      if (q8t) {
        rowsplit(in, tmp, n, (size_t)t->sh[1], unit);
        emit(&o, tmp, n, 1);
      } else {
        emit(&o, in, n, 1);
      }
      done += n;
      if (done >= next || done == total) {
        double dt = now_s() - t0;
        fprintf(stderr, "  %5.1f%% %6.2f / %.2f GB  %6.1f MB/s  %s\n", 100.0 * done / total, done / 1e9, total / 1e9,
                dt > 0 ? done / dt / 1e6 : 0, t->name);
        while (next <= done) next += 1ull << 30;
      }
    }
    sha_hex(&o.ten, t->sha);
  }
  if (o.pos != pack_bytes) die("internal: wrote %llu of %llu", (unsigned long long)o.pos, (unsigned long long)pack_bytes);
  char pack_sha[65] = "";
  if (!sidecar || shas) sha_hex(&o.pack, pack_sha);

  // manifest.json (json.dumps(indent=1) + "\n"), index.tsv, table.bin
  Buf m = {0}, ix = {0};
  bf(&m, "{\n \"format\": \"%s\",\n \"alignment\": %d,\n \"dtype\": \"%s\",\n \"source\": {\n  \"file\": ",
     q8 ? "llama-bend-pack-q8/v0" : "llama-bend-pack-q35-quant/v0", ALIGN, q8 ? "q8_0-rowsplit" : "mixed-quant");
  bjstr(&m, src, 1);
  bs(&m, ",\n  \"types\": {");
  for (size_t i = 0, first = 1; i < g.nt; i++) {
    size_t j = 0, cnt = 0;
    while (j < i && strcmp(g.t[j].tn, g.t[i].tn)) j++;
    if (j < i) continue;
    for (j = i; j < g.nt; j++) cnt += !strcmp(g.t[j].tn, g.t[i].tn);
    bf(&m, "%s\n   \"%s\": %zu", first ? "" : ",", g.t[i].tn, cnt), first = 0;
  }
  bs(&m, "\n  }\n },\n \"config\": {");
  for (int i = 0; i < cfg.n; i++) {
    bs(&m, i ? ",\n  " : "\n  "), bjstr(&m, cfg.e[i].k, 0), bs(&m, ": ");
    bjval(&m, cfg.e[i].v, 2, 1);
  }
  bs(&m, "\n },\n \"tensors\": {\n");
  T *embd = NULL;
  for (size_t i = 0; i < g.nt; i++) {
    if (i) bs(&m, ",\n");
    manifest_tensor(&m, &g.t[i], q8, NULL, NULL);
    if (!strcmp(g.t[i].name, "token_embd.weight")) embd = &g.t[i];
  }
  int tied = cfg_get(&cfg, "tied_embeddings").i;
  if (tied) bs(&m, ",\n"), manifest_tensor(&m, embd, q8, "token_embd.weight", "output.weight");
  bf(&m, "\n },\n \"pack_bytes\": %llu,\n \"pack_sha256\": \"%s\"\n}\n", (unsigned long long)pack_bytes, pack_sha);

  for (int i = 0; i < 22; i++) bf(&ix, "#cfg\t%s\t", IDX_KEYS[i]), bpystr(&ix, cfg_get(&cfg, IDX_KEYS[i])), bs(&ix, "\n");
  bs(&ix, "# config\t{");
  for (int i = 0; i < 22; i++) {
    bs(&ix, i ? "," : ""), bjstr(&ix, IDX_KEYS[i], 0), bs(&ix, ":");
    bjval(&ix, cfg_get(&cfg, IDX_KEYS[i]), 0, 0);
  }
  bf(&ix, "}\n# pack_bytes\t%llu\n# pack_sha256\t%s\n# format\t%s\n# name\toffset\tbytes\trows\tcols\tsrc_type\trow_bytes\n",
     (unsigned long long)pack_bytes, pack_sha, q8 ? "llama-bend-pack-q8/v0" : "llama-bend-pack-q35-quant/v0");
  for (size_t i = 0; i < g.nt; i++) index_line(&ix, &g.t[i], NULL);
  if (tied) index_line(&ix, embd, "output.weight");
  Buf tb = {0};
  bput(&tb, (const char *)table, (size_t)nl * 128);  // u32 LE (host checked LE)

  double dt = now_s() - t0;
  int rc = 0;
  if (out && sidecar) {
    write_file(out, "table.bin", tb.p, tb.n);
    write_file(out, "manifest.json", m.p, m.n);
    printf("sidecar %s: manifest.json + table.bin for %s (%zu tensors, pack_bytes %llu, sha256 %s)\n", out, src, g.nt,
           (unsigned long long)pack_bytes, shas ? "filled" : "empty");
  } else if (out) {
    if (head_only && ftruncate(o.pack_fd, (off_t)pack_bytes)) die("ftruncate pack.bin: %s", strerror(errno));
    if (close(o.pack_fd)) die("close pack.bin: %s", strerror(errno));
    for (int L = 0; L < nl; L++)
      if (close(ly[L].fd)) die("close layer %d: %s", L, strerror(errno));
    write_file(out, "table.bin", tb.p, tb.n);
    write_file(out, "index.tsv", ix.p, ix.n);
    if (head_only) {
      const char *h = "pack.bin is HEAD-ONLY (gguf2pack --head-only): the non-blk tensors (head, norm, embedding)\n"
                      "are real; every blk range is a hole that reads as zeros. layers/ holds the blk bytes.\n"
                      "The GPU engine reads nothing else. pack_sha256 is of the full pack, not this file.\n"
                      "The CPU twin and KAT/bench tools that mmap pack.bin need a full pack (no --head-only).\n";
      write_file(out, "HEAD_ONLY.txt", h, strlen(h));
    }
    write_file(out, "manifest.json", m.p, m.n);  // last: its presence marks a complete pack
    printf("%zu tensors%s, %llu bytes, pack_sha256 %s\nwrote %s (pack.bin%s, %d layer files, manifest.json, index.tsv, table.bin)\n",
           g.nt, tied ? " (+ output.weight alias)" : "", (unsigned long long)pack_bytes, pack_sha, out,
           head_only ? " head-only" : "", nl);
  } else {
    int ok;
    Buf old = read_file(vdir, "manifest.json", &ok);
    size_t same = 0, diff = 0, miss = 0;
    char os[65];
    for (size_t i = 0; i < g.nt; i++) {
      if (!ok || !old_sha(old.p, g.t[i].name, os)) { miss++; continue; }
      if (!strcmp(os, g.t[i].sha)) same++;
      else if (diff++ < 20) printf("  tensor %s: sha256 %s, %s has %s\n", g.t[i].name, g.t[i].sha, vdir, os);
    }
    printf("verify-against %s: %zu tensors, %zu sha256 identical, %zu differ, %zu not in its manifest; pack_sha256 %s\n",
           vdir, g.nt, same, diff, miss, pack_sha);
    int a = compare(vdir, "manifest.json", &m), b = compare(vdir, "index.tsv", &ix), c = compare(vdir, "table.bin", &tb);
    rc = !(a && b && c && same == g.nt);
    printf("verify-against: %s\n", rc ? "FAIL" : "PASS (manifest.json, index.tsv, table.bin byte-identical)");
    free(old.p);
  }
  if (!sidecar || shas) fprintf(stderr, "gguf2pack: %.1f s, %.1f MB/s, VmHWM %ld kB\n", dt, dt > 0 ? total / dt / 1e6 : 0, vm_hwm_kb());
  return rc;
}
