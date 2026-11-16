/* backends/windows/engine_cpu_gguf.c: engine_cpu's weights straight from the GGUF (XPLAT-WIN).
 *
 * host/llb_gguf.c is the loader the Bend engines use for LLB_GGUF: it derives the
 * virtual pack.bin layout (gguf2pack's: tensors in file order, 64-aligned, Q8_0
 * row-split) from the GGUF header and fills any range of it on N threads. This file
 * compiles it unchanged (its Bend effect half is under CID_GGUF_READ, which only the
 * emitted engine defines) and exports the two calls engine_cpu.c needs. Built with
 * -I backends/windows/compat (pthread, pread, open).
 */
#include "../../host/llb_gguf.c"

/* Open `path`, report every tensor (name, pack offset, bytes) to `each` (a nonzero return stops the load
 * with that tensor named), then read the whole virtual pack.bin into a new buffer on `threads` workers.
 * 0, or -1 with err (256 bytes) set. */
int ecpu_gguf_load(const char *path, int threads, uint8_t **pack, uint64_t *pack_bytes,
                   int (*each)(void *ctx, const char *name, uint64_t off, uint64_t bytes), void *ctx, char *err) {
  GgFile g;
  if (gg_open(&g, path, err)) return -1;
  for (size_t i = 0; i < g.nt; i++)
    if (each && each(ctx, g.t[i].name, g.t[i].off, g.t[i].bytes)) {
      snprintf(err, 256, "%s: tensor %s does not match the sidecar's manifest.json (another GGUF's sidecar?)", path,
               g.t[i].name);
      return -1;
    }
  uint8_t *buf = malloc(g.pack_bytes ? (size_t)g.pack_bytes : 1);
  if (!buf) { snprintf(err, 256, "out of memory for %llu weight bytes", (unsigned long long)g.pack_bytes); return -1; }
  uint64_t lo = 0;
  if (gg_region(&g, GG_VIRT, 0, g.pack_bytes, &lo, err) || gg_fill(&g, lo, g.pack_bytes, buf, threads, err)) {
    free(buf);
    return -1;
  }
  close(g.fd);
  *pack = buf;
  *pack_bytes = g.pack_bytes;
  return 0;
}

/* 1 when the GGUF carries a CLEF decide head (tools/clef3/weights.c: dec.* / decision.* tensors), else 0;
 * -1 with err if it cannot be read. */
int ecpu_gguf_has_head(const char *path, char *err) {
  GgFile g;
  if (gg_open(&g, path, err)) return -1;
  int head = 0;
  for (size_t i = 0; i < g.nt; i++)
    if (!strncmp(g.t[i].name, "dec.", 4) || !strncmp(g.t[i].name, "decision.", 9)) head = 1;
  close(g.fd);
  return head;
}
