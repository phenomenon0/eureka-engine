// head/gguf.c — see gguf.h. Little-endian hosts only (as the GGUF and every engine here).
#define _FILE_OFFSET_BITS 64
#include "gguf.h"

#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }
static int rd_u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }
static int rd_u64(FILE *f, uint64_t *v) { return rd(f, v, 8); }
static char *rd_str(FILE *f) {
  uint64_t n;
  if (rd_u64(f, &n) || n > (1u << 30)) return NULL;
  char *s = (char *)malloc(n + 1);
  if (rd(f, s, n)) { free(s); return NULL; }
  s[n] = 0;
  return s;
}
static const int scalar_size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

static int rd_scalar(FILE *f, int type, double *out) {
  unsigned char b[8];
  if (type < 0 || type > 12 || type == GG_STR || type == GG_ARR) return -1;
  if (rd(f, b, (size_t)scalar_size[type])) return -1;
  switch (type) {
    case GG_U8: *out = b[0]; break;
    case GG_I8: *out = (int8_t)b[0]; break;
    case GG_BOOL: *out = b[0] != 0; break;
    case GG_U16: { uint16_t v; memcpy(&v, b, 2); *out = v; break; }
    case GG_I16: { int16_t v; memcpy(&v, b, 2); *out = v; break; }
    case GG_U32: { uint32_t v; memcpy(&v, b, 4); *out = v; break; }
    case GG_I32: { int32_t v; memcpy(&v, b, 4); *out = v; break; }
    case GG_F32: { float v; memcpy(&v, b, 4); *out = v; break; }
    case GG_U64: { uint64_t v; memcpy(&v, b, 8); *out = (double)v; break; }
    case GG_I64: { int64_t v; memcpy(&v, b, 8); *out = (double)v; break; }
    case GG_F64: { double v; memcpy(&v, b, 8); *out = v; break; }
  }
  return 0;
}

int gguf_open(gguf *g, const char *path, char *err, size_t errn) {
  memset(g, 0, sizeof *g);
  g->f = fopen(path, "rb");
  if (!g->f) { snprintf(err, errn, "gguf: cannot open %s", path); return -1; }
  char magic[4]; uint32_t ver; uint64_t nt, nkv;
  if (rd(g->f, magic, 4) || memcmp(magic, "GGUF", 4) || rd_u32(g->f, &ver) || ver < 2 || ver > 3 ||
      rd_u64(g->f, &nt) || rd_u64(g->f, &nkv) || nt > 100000 || nkv > 100000) {
    snprintf(err, errn, "gguf: %s is not a GGUF v2/v3 file", path);
    goto bad;
  }
  g->kv = (gg_kv *)calloc(nkv ? nkv : 1, sizeof(gg_kv));
  g->t = (gg_tensor *)calloc(nt ? nt : 1, sizeof(gg_tensor));
  uint32_t align = 32;
  for (uint64_t i = 0; i < nkv; i++) {
    gg_kv *kv = &g->kv[g->n_kv++];
    uint32_t type;
    if (!(kv->key = rd_str(g->f)) || rd_u32(g->f, &type)) goto trunc;
    kv->type = (int)type;
    if (type == GG_STR) {
      if (!(kv->str = rd_str(g->f))) goto trunc;
    } else if (type == GG_ARR) {
      uint32_t et; uint64_t n;
      if (rd_u32(g->f, &et) || rd_u64(g->f, &n)) goto trunc;
      kv->num = (double)n;
      for (uint64_t j = 0; j < n; j++) {
        if (et == GG_STR) {
          uint64_t sl;
          if (rd_u64(g->f, &sl) || fseeko(g->f, (off_t)sl, SEEK_CUR)) goto trunc;
        } else if (et < 13 && et != GG_ARR) {
          if (fseeko(g->f, (off_t)scalar_size[et] * (off_t)(n - j), SEEK_CUR)) goto trunc;
          break;
        } else goto trunc;
      }
    } else if (rd_scalar(g->f, (int)type, &kv->num)) goto trunc;
    if (!strcmp(kv->key, "general.alignment")) align = (uint32_t)kv->num;
  }
  for (uint64_t i = 0; i < nt; i++) {
    gg_tensor *t = &g->t[g->n_t++];
    uint32_t nd, type;
    if (!(t->name = rd_str(g->f)) || rd_u32(g->f, &nd) || nd > 4) goto trunc;
    t->n_dims = (int)nd;
    for (int d = 0; d < 4; d++) t->ne[d] = 1;
    for (uint32_t d = 0; d < nd; d++) if (rd_u64(g->f, &t->ne[d])) goto trunc;
    if (rd_u32(g->f, &type) || rd_u64(g->f, &t->off)) goto trunc;
    t->type = (int)type;
  }
  off_t pos = ftello(g->f);
  off_t base = (pos + align - 1) / align * align;
  for (int i = 0; i < g->n_t; i++) g->t[i].off += (uint64_t)base;
  return 0;
trunc:
  snprintf(err, errn, "gguf: %s: truncated or malformed header", path);
bad:
  gguf_close(g);
  return -1;
}

void gguf_close(gguf *g) {
  if (g->f) fclose(g->f);
  for (int i = 0; i < g->n_kv; i++) { free(g->kv[i].key); free(g->kv[i].str); }
  for (int i = 0; i < g->n_t; i++) free(g->t[i].name);
  free(g->kv); free(g->t);
  memset(g, 0, sizeof *g);
}

const gg_kv *gguf_kv(const gguf *g, const char *key) {
  for (int i = 0; i < g->n_kv; i++) if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
  return NULL;
}
const gg_tensor *gguf_tensor(const gguf *g, const char *name) {
  for (int i = 0; i < g->n_t; i++) if (!strcmp(g->t[i].name, name)) return &g->t[i];
  return NULL;
}
int gguf_read(const gguf *g, const gg_tensor *t, void *dst, size_t n) {
  if (fseeko(g->f, (off_t)t->off, SEEK_SET)) return -1;
  return rd(g->f, dst, n);
}

size_t ggml_row_bytes(int type, uint64_t n) {
  switch (type) {
    case GGML_F32: return (size_t)n * 4;
    case GGML_F16:
    case GGML_BF16: return (size_t)n * 2;
    case GGML_Q8_0: return n % 32 ? 0 : (size_t)(n / 32) * 34;
  }
  return 0;
}

float gg_f16(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, u;
  if (e == 0) {
    if (m == 0) u = s;
    else {  // subnormal
      e = 113;
      while (!(m & 1024)) { m <<= 1; e--; }
      u = s | (e << 23) | ((m & 1023) << 13);
    }
  } else if (e == 31) u = s | 0x7F800000u | (m << 13);
  else u = s | ((e + 112) << 23) | (m << 13);
  float f; memcpy(&f, &u, 4);
  return f;
}
float gg_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f; memcpy(&f, &u, 4);
  return f;
}
