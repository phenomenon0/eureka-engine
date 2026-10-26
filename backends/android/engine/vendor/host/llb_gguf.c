// llb_gguf.c: the engine reads its weights straight from a GGUF (LLB_GGUF=path).
//
// The engine still asks for pack bytes: one layer file (layers/blk.N.bin) or a
// range of the virtual pack.bin. This file derives the pack layout from the
// GGUF header alone (the layout tools/gguf2pack.c writes: tensors in file
// order, each at the next 64-aligned pack offset, zero gaps) and fills the
// asked range from the GGUF: pread for every type, plus the Q8_0 row-split
// ([f16 d][32 q] blocks -> [cols q][cols/32 d] rows, gguf2pack.c's loop) on N
// worker threads. Nothing is written to disk; no whole-model buffer.
//
// Two halves. The loader (gg_*) is C11 + POSIX + pthreads with no runtime
// dependency, so tools/gguf2/unit.c compiles this file alone. The effect glue
// (Gguf.read in gguf.bend) is under #ifdef CID_GGUF_READ: the Bend compiler
// splices this file after the runtime, which defines CID_GGUF_READ.
//
// Env: LLB_GGUF        the GGUF (required by the effect)
//      LLB_GGUF_THREADS worker threads (default 8)
//      LLB_GGUF_VERIFY  a manifest.json with sha256s (gguf2pack --sidecar-only
//                       --sha, or a pack's): after each region is in its block,
//                       read the block back (cuMemcpyDtoH on the GPU build),
//                       sha256 every whole tensor in it, print one line each,
//                       fail the load on a mismatch. Off the timed path.

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define GG_ALIGN 64            // pack alignment (gguf2pack ALIGN)
#define GG_UNIT (16u << 20)    // work unit: bytes per worker read
#define GG_VIRT 0xFFFFFFFFu    // which = virtual pack.bin

// ---------------------------------------------------------------- sha256 (gguf2pack.c)

typedef struct {
  uint32_t h[8];
  uint64_t n;
  uint8_t b[64];
  size_t k;
} GgSha;

static const uint32_t GG_SK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define GG_ROR(x, r) ((x) >> (r) | (x) << (32 - (r)))

static void gg_sha_block(uint32_t *h, const uint8_t *p) {
  uint32_t w[64], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = GG_ROR(w[i - 15], 7) ^ GG_ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = GG_ROR(w[i - 2], 17) ^ GG_ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (GG_ROR(e, 6) ^ GG_ROR(e, 11) ^ GG_ROR(e, 25)) + ((e & f) ^ (~e & g)) + GG_SK[i] + w[i];
    uint32_t t2 = (GG_ROR(a, 2) ^ GG_ROR(a, 13) ^ GG_ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
}

static void gg_sha_init(GgSha *s) {
  static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  memcpy(s->h, h0, sizeof h0);
  s->n = 0, s->k = 0;
}

static void gg_sha_update(GgSha *s, const uint8_t *p, size_t n) {
  s->n += n;
  if (s->k) {
    while (n && s->k < 64) s->b[s->k++] = *p++, n--;
    if (s->k < 64) return;
    gg_sha_block(s->h, s->b), s->k = 0;
  }
  for (; n >= 64; p += 64, n -= 64) gg_sha_block(s->h, p);
  memcpy(s->b, p, n), s->k = n;
}

static void gg_sha_hex(GgSha *s, char out[65]) {
  uint64_t bits = s->n * 8;
  uint8_t pad = 0x80, z = 0, len[8];
  gg_sha_update(s, &pad, 1);
  while (s->k != 56) gg_sha_update(s, &z, 1);
  for (int i = 0; i < 8; i++) len[i] = (uint8_t)(bits >> (56 - 8 * i));
  gg_sha_update(s, len, 8);
  for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", s->h[i]);
}

// ---------------------------------------------------------------- the GGUF header

typedef struct {
  char *name;
  int q8;            // Q8_0: row-split on the way in
  uint64_t src;      // absolute GGUF offset
  uint64_t off;      // pack offset
  uint64_t bytes;    // same in both layouts
  uint64_t rb;       // row bytes (Q8_0 only)
  int layer;         // blk.N -> N, else -1
} GgT;

typedef struct {
  int fd;
  GgT *t;
  size_t nt;
  int nl;
  uint64_t *lo, *hi;  // layer N = pack [lo[N], hi[N])
  uint64_t pack_bytes;
} GgFile;

typedef struct {
  int fd, bad;
  uint64_t off;
  uint8_t *buf;
  size_t n, p;
} GgRd;

static void gg_rd_get(GgRd *r, void *dst, uint64_t k) {
  uint8_t *d = dst;
  while (k && !r->bad) {
    if (r->p == r->n) {
      r->off += r->n, r->p = 0;
      ssize_t got = pread(r->fd, r->buf, 1 << 20, (off_t)r->off);
      if (got <= 0) { r->bad = 1; r->n = 0; break; }
      r->n = (size_t)got;
    }
    size_t c = r->n - r->p < k ? r->n - r->p : (size_t)k;
    if (d) memcpy(d, r->buf + r->p, c), d += c;
    r->p += c, k -= c;
  }
}

static uint32_t gg_u32(GgRd *r) { uint32_t v = 0; gg_rd_get(r, &v, 4); return v; }
static uint64_t gg_u64(GgRd *r) { uint64_t v = 0; gg_rd_get(r, &v, 8); return v; }

// a GGUF string: kept (malloc, NUL-terminated) or skipped
static char *gg_str(GgRd *r, int keep) {
  uint64_t n = gg_u64(r);
  if (n > (1ull << 31)) { r->bad = 1; return NULL; }
  if (!keep) { gg_rd_get(r, NULL, n); return NULL; }
  char *s = malloc(n + 1);
  if (!s) { r->bad = 1; return NULL; }
  gg_rd_get(r, s, n), s[n] = 0;
  return s;
}

// GGUF metadata value types: size of the numeric ones (0 = string / array)
static const int GG_KV[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

// skip one value of type t; the integer value if it is one of <= 8 bytes
static uint64_t gg_val(GgRd *r, uint32_t t) {
  if (t > 12) { r->bad = 1; return 0; }
  if (t == 8) { gg_str(r, 0); return 0; }
  if (t == 9) {
    uint32_t et = gg_u32(r);
    uint64_t n = gg_u64(r);
    if (et == 9 || et > 12) { r->bad = 1; return 0; }
    if (et == 8) for (uint64_t i = 0; i < n && !r->bad; i++) gg_str(r, 0);
    else gg_rd_get(r, NULL, n * (uint64_t)GG_KV[et]);
    return 0;
  }
  uint64_t v = 0;
  gg_rd_get(r, &v, (uint64_t)GG_KV[t]);
  return v;
}

// ggml type id -> block bytes / elements (the types gguf2pack knows)
static int gg_type(uint32_t ty, int *bb, int *el) {
  static const struct { uint32_t id; int bb, el; } TY[] = {
      {0, 4, 1}, {1, 2, 1}, {8, 34, 32}, {11, 110, 256}, {12, 144, 256}, {13, 176, 256},
      {14, 210, 256}, {23, 136, 256}, {30, 2, 1}, {42, 18, 64}, {142, 34, 128}};
  for (size_t i = 0; i < sizeof TY / sizeof *TY; i++)
    if (TY[i].id == ty) { *bb = TY[i].bb, *el = TY[i].el; return 1; }
  return 0;
}

static int gg_fail(char *err, const char *fmt, ...) {
  va_list a;
  va_start(a, fmt);
  vsnprintf(err, 256, fmt, a);
  va_end(a);
  return -1;
}

// Parse the header of `path` into g: tensor table, pack offsets, layer
// ranges. Reads the header only. 0, or -1 with err (256 bytes) set.
static int gg_open(GgFile *g, const char *path, char *err) {
  memset(g, 0, sizeof *g);
  uint16_t one = 1;
  if (*(uint8_t *)&one != 1) return gg_fail(err, "big-endian host unsupported");
  g->fd = open(path, O_RDONLY);
  if (g->fd < 0) return gg_fail(err, "open %s: %s", path, strerror(errno));
  struct stat st;
  if (fstat(g->fd, &st)) return gg_fail(err, "stat %s: %s", path, strerror(errno));
  GgRd r = {g->fd, 0, 0, malloc(1 << 20), 0, 0};
  if (!r.buf) return gg_fail(err, "out of memory");
  char magic[4] = {0};
  gg_rd_get(&r, magic, 4);
  uint32_t ver = gg_u32(&r);
  if (memcmp(magic, "GGUF", 4) || (ver != 2 && ver != 3)) { free(r.buf); return gg_fail(err, "%s: not a GGUF v2/v3", path); }
  uint64_t nt = gg_u64(&r), nkv = gg_u64(&r), align = 32;
  if (nt > 1000000 || nkv > 1000000) { free(r.buf); return gg_fail(err, "%s: corrupt header counts", path); }
  for (uint64_t i = 0; i < nkv && !r.bad; i++) {
    char *key = gg_str(&r, 1);
    uint32_t t = gg_u32(&r);
    uint64_t v = gg_val(&r, t);
    if (key && !strcmp(key, "general.alignment") && t != 8 && t != 9) align = v;
    free(key);
  }
  g->nt = (size_t)nt;
  g->t = calloc(nt ? nt : 1, sizeof *g->t);
  if (!g->t) { free(r.buf); return gg_fail(err, "out of memory"); }
  for (size_t i = 0; i < g->nt && !r.bad; i++) {
    GgT *t = &g->t[i];
    t->name = gg_str(&r, 1);
    uint32_t nd = gg_u32(&r);
    if (r.bad || nd < 1 || nd > 4) { r.bad = 1; break; }
    uint64_t ne[4], n = 1;
    for (uint32_t j = 0; j < nd; j++) ne[j] = gg_u64(&r), n *= ne[j];
    uint32_t ty = gg_u32(&r);
    t->src = gg_u64(&r);
    int bb, el;
    if (!gg_type(ty, &bb, &el)) { free(r.buf); return gg_fail(err, "%s: ggml type %u unsupported", t->name, ty); }
    if (n % (uint64_t)el) { free(r.buf); return gg_fail(err, "%s: not a whole number of blocks", t->name); }
    t->bytes = n / (uint64_t)el * (uint64_t)bb;
    t->q8 = ty == 8;
    if (t->q8) {
      if (nd != 2 || ne[0] % 32) { free(r.buf); return gg_fail(err, "%s: Q8_0 needs 2-D, cols %% 32", t->name); }
      t->rb = ne[0] / 32 * 34;  // ne[0] = cols (GGUF ne is reversed)
    }
    unsigned L;
    int c = 0;
    t->layer = sscanf(t->name, "blk.%u.%n", &L, &c) == 1 && c > 0 ? (int)L : -1;
    if (t->layer >= g->nl) g->nl = t->layer + 1;
  }
  int bad = r.bad;
  uint64_t data = r.off + r.p;
  free(r.buf);
  if (bad) return gg_fail(err, "%s: header truncated or corrupt", path);
  if (!align) return gg_fail(err, "general.alignment 0");
  data = (data + align - 1) / align * align;
  g->lo = malloc((size_t)(g->nl ? g->nl : 1) * 8), g->hi = malloc((size_t)(g->nl ? g->nl : 1) * 8);
  if (!g->lo || !g->hi) return gg_fail(err, "out of memory");
  for (int L = 0; L < g->nl; L++) g->lo[L] = UINT64_MAX, g->hi[L] = 0;
  uint64_t pos = 0;
  for (size_t i = 0; i < g->nt; i++) {
    GgT *t = &g->t[i];
    t->src += data;
    if (t->src + t->bytes > (uint64_t)st.st_size) return gg_fail(err, "%s: data past the end of %s (truncated)", t->name, path);
    pos += (GG_ALIGN - pos % GG_ALIGN) % GG_ALIGN;
    t->off = pos, pos += t->bytes;
    if (t->layer >= 0) {
      if (t->off < g->lo[t->layer]) g->lo[t->layer] = t->off;
      if (t->off + t->bytes > g->hi[t->layer]) g->hi[t->layer] = t->off + t->bytes;
    }
  }
  g->pack_bytes = pos;
  for (int L = 0; L < g->nl; L++)
    if (g->lo[L] == UINT64_MAX) return gg_fail(err, "layer %d has no tensors", L);
  return 0;
}

// ---------------------------------------------------------------- filling a region

// one unit of work: pack bytes [a, b) of tensor t
typedef struct { const GgT *t; uint64_t a, b; } GgU;

typedef struct {
  int fd;
  uint64_t lo;
  uint8_t *dst;
  GgU *u;
  size_t nu;
  size_t next;  // shared, under mu
  pthread_mutex_t mu;
  int err;      // first errno
} GgJob;

static int gg_pread(int fd, uint8_t *p, size_t n, uint64_t at) {
  while (n) {
    ssize_t k = pread(fd, p, n, (off_t)at);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return k < 0 ? errno : EIO;
    p += k, n -= (size_t)k, at += (uint64_t)k;
  }
  return 0;
}

// Q8_0 row: [f16 d][32 q] x nb -> [32 q x nb][f16 d x nb] (gguf2pack.c)
static void gg_q8_row(uint8_t *o, const uint8_t *s, size_t rb) {
  size_t nb = rb / 34, cols = nb * 32;
  uint8_t *q = o, *d = o + cols;
  for (size_t j = 0; j < nb; j++) memcpy(q + 32 * j, s + 34 * j + 2, 32), memcpy(d + 2 * j, s + 34 * j, 2);
}

static int gg_unit(GgJob *J, const GgU *u, uint8_t *scr, uint8_t *row) {
  const GgT *t = u->t;
  uint8_t *out = J->dst + (u->a - J->lo);
  uint64_t a = u->a - t->off, b = u->b - t->off;  // tensor-relative
  if (!t->q8) return gg_pread(J->fd, out, (size_t)(b - a), t->src + a);
  // whole rows r0..r1 cover [a, b); rows are the same size in both layouts
  uint64_t rb = t->rb, r0 = a / rb, r1 = (b + rb - 1) / rb;
  int e = gg_pread(J->fd, scr, (size_t)((r1 - r0) * rb), t->src + r0 * rb);
  if (e) return e;
  for (uint64_t r = r0; r < r1; r++) {
    uint64_t ra = r * rb, rz = ra + rb;
    const uint8_t *s = scr + (r - r0) * rb;
    if (ra >= a && rz <= b) gg_q8_row(out + (ra - a), s, (size_t)rb);
    else {  // a partial row at either end of a virtual range
      gg_q8_row(row, s, (size_t)rb);
      uint64_t x = ra > a ? ra : a, y = rz < b ? rz : b;
      memcpy(out + (x - a), row + (x - ra), (size_t)(y - x));
    }
  }
  return 0;
}

static void *gg_worker(void *arg) {
  GgJob *J = arg;
  uint64_t maxrb = 0;
  for (size_t i = 0; i < J->nu; i++)
    if (J->u[i].t->q8 && J->u[i].t->rb > maxrb) maxrb = J->u[i].t->rb;
  uint8_t *scr = NULL, *row = NULL;
  if (maxrb && (!(scr = malloc(GG_UNIT + 2 * maxrb)) || !(row = malloc(maxrb)))) {
    pthread_mutex_lock(&J->mu);
    if (!J->err) J->err = ENOMEM;
    pthread_mutex_unlock(&J->mu);
  }
  for (;;) {
    pthread_mutex_lock(&J->mu);
    size_t i = J->err ? J->nu : J->next++;
    pthread_mutex_unlock(&J->mu);
    if (i >= J->nu) break;
    int e = gg_unit(J, &J->u[i], scr, row);
    if (e) {
      pthread_mutex_lock(&J->mu);
      if (!J->err) J->err = e;
      pthread_mutex_unlock(&J->mu);
    }
  }
  free(scr), free(row);
  return NULL;
}

// Pack bytes [lo, lo + n) into dst: tensors by pread (+ row-split for Q8_0)
// on `threads` workers, the alignment gaps zeroed. 0 or -1 with err.
static int gg_fill(const GgFile *g, uint64_t lo, uint64_t n, uint8_t *dst, int threads, char *err) {
  uint64_t hi = lo + n;
  if (hi > g->pack_bytes || hi < lo) return gg_fail(err, "range [%llu, +%llu) past the pack (%llu B)",
                                                    (unsigned long long)lo, (unsigned long long)n, (unsigned long long)g->pack_bytes);
  size_t cap = 0, nu = 0;
  GgU *u = NULL;
  uint64_t pos = lo;  // first byte not yet covered
  for (size_t i = 0; i < g->nt; i++) {
    const GgT *t = &g->t[i];
    uint64_t a = t->off > lo ? t->off : lo, b = t->off + t->bytes < hi ? t->off + t->bytes : hi;
    if (a >= b) continue;
    if (a > pos) memset(dst + (pos - lo), 0, (size_t)(a - pos));
    pos = b;
    // split into units of ~GG_UNIT; for Q8_0 at row boundaries
    uint64_t step = t->q8 ? (GG_UNIT / t->rb ? GG_UNIT / t->rb : 1) * t->rb : GG_UNIT;
    for (uint64_t x = a; x < b;) {
      uint64_t y = t->q8 ? t->off + ((x - t->off) / step + 1) * step : x + step;
      if (y > b) y = b;
      if (nu == cap) {
        GgU *v = realloc(u, (cap = cap ? cap * 2 : 256) * sizeof *u);
        if (!v) { free(u); return gg_fail(err, "out of memory"); }
        u = v;
      }
      u[nu++] = (GgU){t, x, y};
      x = y;
    }
  }
  if (hi > pos) memset(dst + (pos - lo), 0, (size_t)(hi - pos));
  GgJob J = {g->fd, lo, dst, u, nu, 0, PTHREAD_MUTEX_INITIALIZER, 0};
  if (threads < 1) threads = 1;
  if (threads > 64) threads = 64;
  pthread_t tid[64];
  int nt = 0;
  for (; nt < threads - 1; nt++)
    if (pthread_create(&tid[nt], NULL, gg_worker, &J)) break;
  gg_worker(&J);
  for (int i = 0; i < nt; i++) pthread_join(tid[i], NULL);
  free(u);
  if (J.err) return gg_fail(err, "read: %s", strerror(J.err));
  return 0;
}

// Resolve a request: layer `which` (its whole range; n must match, else the
// sidecar is for another GGUF), or GG_VIRT = virtual pack.bin [off, off + n).
static int gg_region(const GgFile *g, uint32_t which, uint64_t off, uint64_t n, uint64_t *lo, char *err) {
  if (which == GG_VIRT) {
    if (off + n > g->pack_bytes) return gg_fail(err, "pack.bin [%llu, +%llu) past its %llu bytes: stale sidecar?",
                                                (unsigned long long)off, (unsigned long long)n, (unsigned long long)g->pack_bytes);
    *lo = off;
    return 0;
  }
  if (which >= (uint32_t)g->nl) return gg_fail(err, "layer %u: the GGUF has %d", which, g->nl);
  if (g->hi[which] - g->lo[which] != n)
    return gg_fail(err, "layer %u: %llu bytes asked, the GGUF has %llu: stale sidecar?", which, (unsigned long long)n,
                   (unsigned long long)(g->hi[which] - g->lo[which]));
  *lo = g->lo[which];
  return 0;
}

// The sha256 recorded for `name` in an indent=1 manifest (gguf2pack.c old_sha).
static int gg_manifest_sha(const char *m, const char *name, char out[65]) {
  char key[512];
  snprintf(key, sizeof key, "\n  \"%s\": {", name);
  const char *p = strstr(m, key), *s;
  if (!p || !(s = strstr(p, "\"sha256\": \"")) || strchr(p + 1, '}') < s) return 0;
  memcpy(out, s + 11, 64), out[64] = 0;
  return out[0] != '"';  // an empty sha (sidecar without --sha) is no sha
}

static char *gg_slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  size_t n = 0, cap = 1 << 16, k;
  char *p = malloc(cap);
  while (p && (k = fread(p + n, 1, cap - n - 1, f)) > 0)
    if ((n += k) + 1 == cap) p = realloc(p, cap *= 2);
  fclose(f);
  if (p) p[n] = 0;
  return p;
}

static int gg_threads(void) {
  const char *s = getenv("LLB_GGUF_THREADS");
  int n = s ? atoi(s) : 8;
  return n > 0 ? n : 8;
}

// ---------------------------------------------------------------- the effect

#ifdef CID_GGUF_READ

// Gguf.read(which, off, n): one region into a Bytes. Like File.read_buf: the
// helper fills a malloc buffer, the loop puts it into the block (io_buf;
// cuMemcpyHtoD for a big one, so the CPU never writes the managed heap).

static GgFile gg_file;
static int gg_state;  // 0 unopened, 1 open, -1 failed
static char gg_err0[256];
static pthread_mutex_t gg_mu = PTHREAD_MUTEX_INITIALIZER;

static void gguf_read_call(IoWork* w) {
  char err[256] = "";
  uint32_t which = (uint32_t)w->hand;
  uint64_t n = w->size, lo = 0;
  pthread_mutex_lock(&gg_mu);
  if (!gg_state) {
    const char* path = getenv("LLB_GGUF");
    gg_state = !path || !*path ? (snprintf(gg_err0, sizeof gg_err0, "LLB_GGUF unset"), -1)
                               : gg_open(&gg_file, path, gg_err0) ? -1 : 1;
    if (gg_state > 0)
      fprintf(stderr, "gguf: weights from %s (%zu tensors, %d layers, pack_bytes %llu)\n", path, gg_file.nt,
              gg_file.nl, (unsigned long long)gg_file.pack_bytes);
  }
  int ok = gg_state > 0 ? !gg_region(&gg_file, which, (uint64_t)w->made, n, &lo, err)
                       : (snprintf(err, sizeof err, "%s", gg_err0), 0);
  pthread_mutex_unlock(&gg_mu);
  if (ok) ok = !gg_fill(&gg_file, lo, n, (uint8_t*)w->data, gg_threads(), err);
  w->made = (intptr_t)lo;
  if (!ok) {
    w->code = EIO;
    w->text = strdup(err);
  }
}

// LLB_GGUF_VERIFY: the block read back, every whole tensor in [lo, lo + n)
// hashed and compared with the manifest's sha256. 0 ok, else mismatches.
static int gguf_verify(Env e, Term blk, uint64_t lo, uint64_t n, const char* man) {
  static char* m;
  if (!m && !(m = gg_slurp(man))) {
    fprintf(stderr, "gguf verify: cannot read %s\n", man);
    return 1;
  }
  const u8* p = (const u8*)(e.mem + term_peek(e, blk));
  size_t CH = 64u << 20;
  u8* tmp = malloc(CH);
  int bad = 0, seen = 0;
  for (size_t i = 0; i < gg_file.nt && tmp; i++) {
    const GgT* t = &gg_file.t[i];
    if (t->off < lo || t->off + t->bytes > lo + n) continue;
    GgSha s;
    gg_sha_init(&s);
    for (uint64_t k = 0; k < t->bytes; k += CH) {
      size_t c = t->bytes - k < CH ? (size_t)(t->bytes - k) : CH;
      const u8* at = p + (t->off - lo) + k;
#if BEND_CUDA
      if (io_gpu && cuMemcpyDtoH(tmp, (CUdeviceptr)(uintptr_t)at, c) == CUDA_SUCCESS) at = tmp;
#endif
      gg_sha_update(&s, at, c);
    }
    char got[65], want[65];
    gg_sha_hex(&s, got);
    int have = gg_manifest_sha(m, t->name, want), same = have && !strcmp(got, want);
    fprintf(stderr, "g1 %s %s %s\n", t->name, got, !have ? "NO-SHA" : same ? "ok" : "MISMATCH");
    bad += !same, seen++;
  }
  free(tmp);
  return bad + !tmp;
}

static Term gguf_read_pack(Env e, IoWork* w) {
  Term r;
  if (w->code) {
    r = io_fail(e, w->code, w->text);
  } else {
    StrParts out = str_alloc(e, w->size, 2);
    if (out.data && !err_seen(e.mem)) {
      io_put((u8*)(e.mem + term_peek(e, out.data)), w->data, w->size);
      const char* man = getenv("LLB_GGUF_VERIFY");
      if (man && *man && gguf_verify(e, out.data, (uint64_t)w->made, w->size, man)) {
        fprintf(stderr, "gguf verify: device bytes differ from %s\n", man);
        exit(1);
      }
    }
    r = io_done(e, str_view_owned(e, out));
  }
  free(w->data);
  free(w->text);
  w->text = NULL;
  return r;
}

Term gguf_read_run(Env e, Term* f, IoWork* w) {
  (void)e;
  w->hand = (intptr_t)(u32)f[0];
  w->made = (intptr_t)(u32)f[1];
  w->size = (u32)f[2];
  w->code = 0;
  w->text = NULL;
  if (w->size > (1ull << 31)) {
    return io_fail(e, EINVAL, "Gguf.read: region over 2^31 bytes");
  }
  w->data = io_mem(malloc(w->size ? w->size : 1));
  return io_work(w, gguf_read_call, gguf_read_pack);
}

static void __attribute__((constructor)) gguf_read_use(void) {
  io_eff(CID_GGUF_READ, gguf_read_run, 0);
}

#endif
