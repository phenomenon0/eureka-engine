#!/usr/bin/env python3
# (G3) host-only edits to the emitted C (after host-prefix.patch): no kernel or device-program text changes.
#   1. LLB_PACK_MAP=<file>: the packed weight regions kept in a file and later mapped into the heap block (mmap-class load)
#   2. the prefix restore checks a kept slot's layer count and bytes (slots can now come from a file: llb_state_load)
# usage: g3_emit.py <sys1mac.c>  (in place; exits 3 if an anchor is missing)
import sys

p = sys.argv[1]
s = open(p).read()


def sub(old, new, count=1):
    global s
    if s.count(old) != count:
        sys.exit(f"g3_emit: anchor found {s.count(old)}x (want {count}): {old[:70]!r}")
    s = s.replace(old, new)


# 2. restore guard
sub("    if (llb_px_match(&llb_px[k], ids, nt, dT) && (best == NULL || llb_px[k].p > best->p)) {",
    "    if (llb_px_match(&llb_px[k], ids, nt, dT) && llb_px[k].nl == S0->nl\n"
    "      && llb_px[k].nb >= llb_px_bytes(llb_px[k].p, S0) && (best == NULL || llb_px[k].p > best->p)) {")

# 1. pack map: state + helpers before gguf_read_call
PM = r'''
// (G3) LLB_PACK_MAP=<file>: the mmap-class load. The packed regions (gg_fill's bytes: Q8_0 row-split, alignment gaps)
// are not the GGUF's bytes, so the GGUF itself cannot be mapped; instead each region is written once to <file> as it
// is read (the first load), at a file offset congruent to its heap block's address mod the page, and indexed in
// <file>.idx (magic, pack_bytes, the GGUF's size + mtime, then one {lo, n, off} per region). A later load maps the
// region's whole pages MAP_PRIVATE | MAP_FIXED from <file> over the heap block (clean file pages: not dirty memory)
// and preads only the partial head and tail page; a block whose address is not congruent with its region's offset
// falls back to pread (counted). Off (unset): the reads and copies are the engine's own, unchanged.
typedef struct { uint64_t lo, n, off; } G3PmEnt;
static int       g3_pm_mode = -1;      // -1 unread, 0 off, 1 on
static int       g3_pm_fd = -1, g3_pm_ix = -1;
static G3PmEnt*  g3_pm_ent;
static size_t    g3_pm_n, g3_pm_cap;
static uint64_t  g3_pm_mapped, g3_pm_copied, g3_pm_wrote, g3_pm_end;
static pthread_mutex_t g3_pm_mu = PTHREAD_MUTEX_INITIALIZER;

static uint64_t g3_pm_page(void) { return (uint64_t)getpagesize(); }

// (G18) the state is keyed to (LLB_PACK_MAP, LLB_GGUF, its size + mtime, pack_bytes), read at every region: a load of
// another GGUF in the same process (the app's Qwen <-> Kev swap) re-reads the env and opens that model's own pack
// file. Before G18 the env was read once per process and regions were matched by (lo, n) alone, so a second model of
// the same layout would have mapped the first model's weights.
static char g3_pm_key[2400];

static void g3_pm_open(uint64_t pack_bytes) {  // under g3_pm_mu
  const char* path = getenv("LLB_PACK_MAP");
  const char* gg = getenv("LLB_GGUF");
  struct stat gs;
  int have = gg != NULL && stat(gg, &gs) == 0;
  char key[sizeof g3_pm_key];
  snprintf(key, sizeof key, "%s|%s|%lld|%lld|%llu", path ? path : "", gg ? gg : "", have ? (long long)gs.st_size : -1LL,
    have ? (long long)gs.st_mtime : -1LL, (unsigned long long)pack_bytes);
  if (g3_pm_mode >= 0 && strcmp(key, g3_pm_key) == 0) return;
  if (g3_pm_mode >= 0) {  // (G18) re-keyed: the previous file's pages stay valid where mapped (mmap holds the file)
    fprintf(stderr, "packmap: re-keyed for %s\n", gg ? gg : "(no gguf)");
  }
  if (g3_pm_fd >= 0) close(g3_pm_fd);
  if (g3_pm_ix >= 0) close(g3_pm_ix);
  g3_pm_fd = g3_pm_ix = -1;
  free(g3_pm_ent);
  g3_pm_ent = NULL;
  g3_pm_n = g3_pm_cap = 0;
  g3_pm_end = g3_pm_mapped = g3_pm_copied = g3_pm_wrote = 0;
  snprintf(g3_pm_key, sizeof g3_pm_key, "%s", key);
  g3_pm_mode = 0;
  if (!path || !*path || !have) return;
  char ix[1100];
  snprintf(ix, sizeof ix, "%s.idx", path);
  uint64_t want[4] = {0x31304b4341503347ull /* G3PACK01 */, pack_bytes, (uint64_t)gs.st_size, (uint64_t)gs.st_mtime};
  g3_pm_fd = open(path, O_RDWR | O_CREAT, 0644);
  g3_pm_ix = open(ix, O_RDWR | O_CREAT, 0644);
  if (g3_pm_fd < 0 || g3_pm_ix < 0) {
    fprintf(stderr, "packmap: cannot open %s (off)\n", path);
    return;
  }
  uint64_t got[4] = {0};
  struct stat is;
  fstat(g3_pm_ix, &is);
  if (pread(g3_pm_ix, got, sizeof got, 0) != (ssize_t)sizeof got || memcmp(got, want, sizeof want) != 0) {
    // new or stale (another GGUF / pack): start over
    if (ftruncate(g3_pm_fd, 0) != 0 || ftruncate(g3_pm_ix, 0) != 0
      || pwrite(g3_pm_ix, want, sizeof want, 0) != (ssize_t)sizeof want) {
      fprintf(stderr, "packmap: cannot reset %s (off)\n", path);
      return;
    }
    is.st_size = sizeof want;
    fprintf(stderr, "packmap: %s new (pack_bytes %llu)\n", path, (unsigned long long)pack_bytes);
  }
  size_t ne = ((size_t)is.st_size - sizeof want) / sizeof(G3PmEnt);
  g3_pm_cap = ne + 64;
  g3_pm_ent = calloc(g3_pm_cap, sizeof(G3PmEnt));
  if (ne && pread(g3_pm_ix, g3_pm_ent, ne * sizeof(G3PmEnt), sizeof want) != (ssize_t)(ne * sizeof(G3PmEnt))) ne = 0;
  g3_pm_n = ne;
  struct stat ds;
  fstat(g3_pm_fd, &ds);
  g3_pm_end = (uint64_t)ds.st_size;
  for (size_t i = 0; i < g3_pm_n; i++) {  // an entry past the data (a torn write) is dropped
    if (g3_pm_ent[i].off + g3_pm_ent[i].n > g3_pm_end) g3_pm_n = i;
  }
  g3_pm_mode = 1;
  fprintf(stderr, "packmap: %s, %zu regions indexed\n", path, g3_pm_n);
}

static const G3PmEnt* g3_pm_find(uint64_t lo, uint64_t n) {  // under g3_pm_mu
  for (size_t i = 0; i < g3_pm_n; i++) {
    if (g3_pm_ent[i].lo == lo && g3_pm_ent[i].n == n) return &g3_pm_ent[i];
  }
  return NULL;
}

static int g3_pm_pread(void* dst, uint64_t n, uint64_t off) {
  uint8_t* p = dst;
  while (n) {
    ssize_t k = pread(g3_pm_fd, p, n > (1u << 30) ? (1u << 30) : n, (off_t)off);
    if (k <= 0) return -1;
    p += k, n -= (uint64_t)k, off += (uint64_t)k;
  }
  return 0;
}

// a heap block d of n bytes from entry E: whole pages mapped, head / tail read. 0, or -1 (the caller reads the GGUF)
static int g3_pm_place(uint8_t* d, const G3PmEnt* E) {
  uint64_t pg = g3_pm_page(), a = ((uint64_t)d + pg - 1) / pg * pg, b = ((uint64_t)d + E->n) / pg * pg;
  if ((uint64_t)d % pg != E->off % pg || b <= a) {
    if (g3_pm_pread(d, E->n, E->off) != 0) return -1;
    g3_pm_copied += E->n;
    return 0;
  }
  uint64_t head = a - (uint64_t)d, tail = (uint64_t)d + E->n - b;
  if (head && g3_pm_pread(d, head, E->off) != 0) return -1;
  if (tail && g3_pm_pread((uint8_t*)b, tail, E->off + (b - (uint64_t)d)) != 0) return -1;
  void* m = mmap((void*)a, b - a, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, g3_pm_fd, (off_t)(E->off + head));
  if (m == MAP_FAILED) return -1;
  g3_pm_mapped += b - a;
  g3_pm_copied += head + tail;
  return 0;
}

// the first load: the region just copied into d, written at an offset congruent to d, and indexed
static void g3_pm_keep(const uint8_t* d, const uint8_t* src, uint64_t lo, uint64_t n) {  // under g3_pm_mu
  uint64_t pg = g3_pm_page(), off = (g3_pm_end + pg - 1) / pg * pg + (uint64_t)d % pg;
  const uint8_t* p = src;
  for (uint64_t k = 0; k < n;) {
    ssize_t w = pwrite(g3_pm_fd, p + k, n - k > (1u << 30) ? (1u << 30) : n - k, (off_t)(off + k));
    if (w <= 0) return;
    k += (uint64_t)w;
  }
  if (g3_pm_n == g3_pm_cap) {
    g3_pm_cap *= 2;
    g3_pm_ent = realloc(g3_pm_ent, g3_pm_cap * sizeof(G3PmEnt));
  }
  G3PmEnt E = {lo, n, off};
  if (pwrite(g3_pm_ix, &E, sizeof E, (off_t)(32 + g3_pm_n * sizeof E)) != (ssize_t)sizeof E) return;
  g3_pm_ent[g3_pm_n++] = E;
  g3_pm_end = off + n;
  g3_pm_wrote += n;
}

// the span's Metal buffer made again over the same addresses: a buffer made before the map still resolves the
// anonymous pages the map replaced (the GPU read zeros); encoders take gpu_buf afresh, so the next dispatch sees the file
static void g3_pm_rebind(void) {
#if BEND_METAL && defined(__OBJC__)
  if (gpu_buf != nil) {
    id<MTLBuffer> b = [gpu_dev newBufferWithBytesNoCopy:CORPUS length:[gpu_buf length]
      options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
    if (!b) {
      err_fail("packmap: the span's buffer could not be made again");
    }
    gpu_buf = b;
  }
#endif
}

// one line for the report (llb_resident_report's packmap=)
static void g3_pm_report(char* out, size_t cap) {
  snprintf(out, cap, "%s mapped_mb=%.0f copied_mb=%.1f wrote_mb=%.0f", g3_pm_mode == 1 ? "on" : "off",
    g3_pm_mapped / 1048576.0, g3_pm_copied / 1048576.0, g3_pm_wrote / 1048576.0);
}

static void gguf_read_call(IoWork* w) {'''
sub("\nstatic void gguf_read_call(IoWork* w) {", PM)

# in the worker: a region the file holds is not read from the GGUF (marked for gguf_read_pack)
sub('''  pthread_mutex_unlock(&gg_mu);
  if (ok) ok = !gg_fill(&gg_file, lo, n, (uint8_t*)w->data, gg_threads(), err);
  w->made = (intptr_t)lo;''',
'''  uint64_t pack_bytes = gg_file.pack_bytes;
  pthread_mutex_unlock(&gg_mu);
  int g3_hit = 0;
  if (ok) {  // (G3) LLB_PACK_MAP
    pthread_mutex_lock(&g3_pm_mu);
    g3_pm_open(pack_bytes);
    g3_hit = g3_pm_mode == 1 && g3_pm_find(lo, n) != NULL;
    pthread_mutex_unlock(&g3_pm_mu);
  }
  if (ok && !g3_hit) ok = !gg_fill(&gg_file, lo, n, (uint8_t*)w->data, gg_threads(), err);
  if (ok && g3_hit) w->text = strdup("g3:packmap");
  w->made = (intptr_t)lo;''')

sub('''    StrParts out = str_alloc(e, w->size, 2);
    if (out.data && !err_seen(e.mem)) {
      memcpy((u8*)(e.mem + term_peek(e, out.data)), w->data, w->size);''',
'''    StrParts out = str_alloc(e, w->size, 2);
    // (G3) LLB_PACK_MAP: the region from the pack file (mapped), or kept into it (the first load)
    bool g3_hit = w->text != NULL && strcmp(w->text, "g3:packmap") == 0, g3_done = false;
    if (out.data && !err_seen(e.mem) && g3_hit) {
      pthread_mutex_lock(&g3_pm_mu);
      const G3PmEnt* E = g3_pm_find((uint64_t)w->made, w->size);
      g3_done = E != NULL && g3_pm_place((u8*)(e.mem + term_peek(e, out.data)), E) == 0;
      pthread_mutex_unlock(&g3_pm_mu);
      if (g3_done) {
        g3_pm_rebind();
      }
      if (!g3_done) {
        char err[256] = "";
        if (gg_fill(&gg_file, (uint64_t)w->made, w->size, (uint8_t*)w->data, gg_threads(), err) != 0) {
          fprintf(stderr, "packmap: fallback read failed: %s\\n", err);
          exit(1);
        }
      }
    }
    if (out.data && !err_seen(e.mem) && !g3_done) {
      memcpy((u8*)(e.mem + term_peek(e, out.data)), w->data, w->size);
      if (g3_pm_mode == 1 && !g3_hit) {
        pthread_mutex_lock(&g3_pm_mu);
        g3_pm_keep((u8*)(e.mem + term_peek(e, out.data)), (const uint8_t*)w->data, (uint64_t)w->made, w->size);
        pthread_mutex_unlock(&g3_pm_mu);
      }''')

open(p, "w").write(s)
print("g3_emit: ok")
