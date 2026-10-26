// auxpack.c — AUX1 pack reader (twoheads S6-a). Standalone C, host idioms
// (mirrors host/llb_gguf.c's validate-while-parsing discipline). No engine or
// GPU dependency: the engine-side consumer lands in a later increment.
//
//   auxpack --show <file>                 header dump
//   auxpack --selftest <fixture> [<bad>]  read/validate round-trip + negative tests
//
// AUX1 layout (little-endian): magic "AUX1", u32 version, u32 n, u32 flags,
// n * 72B entries {char name[32]; u32 dtype; u32 rank; u32 ne[4]; u64 off; u64 bytes},
// payloads 256-aligned. rank 0 = opaque bytes.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AUX_MAX 128
#define AUX_ENTRY 72

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

enum { DT_F32 = 0, DT_BF16 = 1, DT_Q8_0 = 2, DT_Q4_K = 3, DT_Q6_K = 4, DT_U32 = 5, DT_BLOB = 6 };
static const char* dt_name(u32 d) {
  switch (d) {
    case DT_F32: return "F32"; case DT_BF16: return "BF16"; case DT_Q8_0: return "Q8_0";
    case DT_Q4_K: return "Q4_K"; case DT_Q6_K: return "Q6_K"; case DT_U32: return "U32";
    case DT_BLOB: return "BLOB"; default: return "?";
  }
}

typedef struct {
  char name[33];
  u32 dtype, rank, ne[4];
  u64 off, bytes;
} AuxE;

typedef struct {
  u32 ver, n, flags;
  u64 size;
  AuxE e[AUX_MAX];
} Aux;

static u32 rd32(const u8* p) { u32 v; memcpy(&v, p, 4); return v; }
static u64 rd64(const u8* p) { u64 v; memcpy(&v, p, 8); return v; }

static int aux_open(Aux* a, const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "aux: cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  a->size = (u64)ftell(f);
  fseek(f, 0, SEEK_SET);
  u8 hd[16];
  if (fread(hd, 1, 16, f) != 16 || memcmp(hd, "AUX1", 4)) { fprintf(stderr, "aux: bad magic in %s\n", path); fclose(f); return 1; }
  a->ver = rd32(hd + 4); a->n = rd32(hd + 8); a->flags = rd32(hd + 12);
  if (a->ver != 1 || a->n == 0 || a->n > AUX_MAX) { fprintf(stderr, "aux: unsupported version/count (%u/%u)\n", a->ver, a->n); fclose(f); return 1; }
  u64 need = 16ull + (u64)AUX_ENTRY * a->n;
  if (need > a->size) { fprintf(stderr, "aux: header past EOF\n"); fclose(f); return 1; }
  u64 en = (u64)AUX_ENTRY * a->n;
  u8* hdr = malloc((size_t)en);
  if (!hdr || fread(hdr, 1, (size_t)en, f) != (size_t)en) { fprintf(stderr, "aux: short header\n"); free(hdr); fclose(f); return 1; }
  for (u32 i = 0; i < a->n; i += 1) {
    const u8* p = hdr + (u64)AUX_ENTRY * i;
    AuxE* e = &a->e[i];
    memcpy(e->name, p, 32);
    e->name[32] = 0;
    e->dtype = rd32(p + 32); e->rank = rd32(p + 36);
    for (int d = 0; d < 4; d += 1) e->ne[d] = rd32(p + 40 + 4 * d);
    e->off = rd64(p + 56); e->bytes = rd64(p + 64);
    if ((e->off & 255) || e->off + e->bytes > a->size || e->bytes == 0) {
      fprintf(stderr, "aux: entry %u (%s) out of bounds (off=%llu bytes=%llu size=%llu)\n",
              i, e->name, (unsigned long long)e->off, (unsigned long long)e->bytes, (unsigned long long)a->size);
      free(hdr); fclose(f); return 1;
    }
    for (u32 j = 0; j < i; j += 1) {
      if (!strcmp(e->name, a->e[j].name)) { fprintf(stderr, "aux: duplicate name %s\n", e->name); free(hdr); fclose(f); return 1; }
    }
  }
  free(hdr); fclose(f);
  return 0;
}

static AuxE* aux_find(Aux* a, const char* name) {
  for (u32 i = 0; i < a->n; i += 1) if (!strcmp(a->e[i].name, name)) return &a->e[i];
  return NULL;
}

// expected dtype/shape check; rank 0 expected = opaque (bytes pinned by caller)
static int aux_expect(Aux* a, const char* name, u32 dtype, u32 rank, const u32* ne) {
  AuxE* e = aux_find(a, name);
  if (!e) { fprintf(stderr, "aux: MISSING %s\n", name); return 1; }
  if (e->dtype != dtype) { fprintf(stderr, "aux: %s dtype %s != %s\n", name, dt_name(e->dtype), dt_name(dtype)); return 1; }
  if (e->rank != rank) { fprintf(stderr, "aux: %s rank %u != %u\n", name, e->rank, rank); return 1; }
  for (u32 d = 0; d < rank && d < 4; d += 1) {
    if (e->ne[d] != ne[d]) { fprintf(stderr, "aux: %s ne[%u] %u != %u\n", name, d, e->ne[d], ne[d]); return 1; }
  }
  return 0;
}

static int show(const char* path) {
  Aux a;
  if (aux_open(&a, path)) return 1;
  printf("AUX1 v%u n=%u flags=%u size=%llu\n", a.ver, a.n, a.flags, (unsigned long long)a.size);
  for (u32 i = 0; i < a.n; i += 1) {
    AuxE* e = &a.e[i];
    printf("  [%2u] %-20s %-5s rank=%u ne=(%u,%u,%u,%u) off=%llu bytes=%llu\n",
           i, e->name, dt_name(e->dtype), e->rank, e->ne[0], e->ne[1], e->ne[2], e->ne[3],
           (unsigned long long)e->off, (unsigned long long)e->bytes);
  }
  return 0;
}

static int selftest(const char* fixture, const char* bad) {
  int fail = 0;
  Aux a;
  if (aux_open(&a, fixture)) { printf("FAIL open fixture\n"); return 1; }
  u32 nea[4] = { 8, 0, 0, 0 }, neb[4] = { 4, 0, 0, 0 }, nec[4] = { 0, 0, 0, 0 };
  fail |= aux_expect(&a, "fixture.a", DT_F32, 1, nea);
  fail |= aux_expect(&a, "fixture.b", DT_U32, 1, neb);
  fail |= aux_expect(&a, "fixture.c", DT_BLOB, 0, nec);
  printf(fail ? "FAIL expect fixture\n" : "PASS read+validate fixture (3 named entries)\n");
  if (aux_find(&a, "fixture.zz")) { printf("FAIL missing-name lookup returned an entry\n"); fail = 1; }
  else printf("PASS missing-name lookup => NULL\n");
  if (bad) {
    Aux b;
    if (aux_open(&b, bad)) printf("FAIL open bad fixture\n"), fail = 1;
    else {
      int r = aux_expect(&b, "fixture.a", DT_F32, 1, nea);
      printf(r ? "PASS tampered shape rejected (ne declared 9 != expected 8)\n" : "FAIL tampered shape accepted\n");
      fail |= !r;
    }
  }
  return fail;
}

int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "--show")) return show(argv[2]);
  if (argc >= 3 && !strcmp(argv[1], "--selftest")) return selftest(argv[2], argc >= 4 ? argv[3] : NULL);
  fprintf(stderr, "usage: auxpack --show <file> | --selftest <fixture> [<bad-fixture>]\n");
  return 2;
}
