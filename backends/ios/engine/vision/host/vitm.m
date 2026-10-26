// vitm.m: see vitm.h. The host side of the Metal tower: the mmproj's tensors are copied as stored (F16 matrices and
// conv kernels, F32 biases and norms) into one shared MTLBuffer, the kernels (vitm.metal, embedded as vitm_msl.h by
// the build) are compiled once per load with default options (as ggml compiles its library), and vitm_encode
// records the whole tower (prologue, the layers (27 on the 9B tower, 24 on the 4B), merger: 8 dispatches per layer when
// fused) into one command buffer.
// Graph and dump names = vit.c. Build: -fobjc-arc, Metal + Foundation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "vitm.h"
#include "vitm_msl.h"
#include "vmm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int k, o, n, ldb, ldd, flags; } VtMmArgs;
typedef struct { int n, npad, ldq, ldo; float scale; } VtFaArgs;
typedef struct { int ne00, ne00_t; float eps; } VtLnArgs;
typedef struct { int n, E, H, npad, dbg; } VtRopeArgs;
typedef struct { int nx, ny, px, n; } VtI2cArgs;
enum { MM_BIAS = 1, MM_GELU = 2, MM_RES = 4 };

enum { T_LN1W, T_LN1B, T_QKVW, T_QKVB, T_OW, T_OB, T_LN2W, T_LN2B, T_UPW, T_UPB, T_DOWNW, T_DOWNB, T_NL };
static const char *const LNAME[T_NL] = {"ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
  "attn_out.bias", "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"};

@interface VitmCtx : NSObject {
 @public
  VMmH h;
  int E, H, D, F, L, P, M, PD, G;
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLBuffer> wb;  // all weights, 256-aligned offsets
  size_t o_pe0, o_pe1, o_peb, o_postw, o_postb, o_mm0w, o_mm0b, o_mm2w, o_mm2b, *o_l;  // o_l[il * T_NL + t]
  float *pos;        // v.position_embd [G*G][E] (F32 in the file), host side
  id<MTLComputePipelineState> p_mm32, p_mm16, p_fa, p_ln, p_rope, p_gelu, p_add, p_embed, p_i2c;
  size_t ncap, pixcap;  // activation capacity (patches), planar capacity (pixels)
  id<MTLBuffer> b_planar, b_rows, b_x, b_cur, b_ao, b_qr, b_qkv, b_up, b_kh, b_vh, b_kd, b_pos, b_ppos, b_embd;
  int pos_px, pos_py;  // the image size b_pos / b_ppos were built for
  double gpu_ms;
}
@end
@implementation VitmCtx
- (void)dealloc {
  free(o_l);
  free(pos);
}
@end

static VitmCtx *ctx(const VitmW *w) { return (__bridge VitmCtx *)(const void *)w; }

void vitm_free(VitmW *w) {
  if (w) { VitmCtx *c = (__bridge_transfer VitmCtx *)(void *)w; (void)c; }
}

int vitm_n_tokens(const VitmW *w, int nx, int ny) { VitmCtx *c = ctx(w); return (nx / (c->P * c->M)) * (ny / (c->P * c->M)); }
int vitm_proj_dim(const VitmW *w) { return ctx(w)->PD; }
double vitm_gpu_ms(const VitmW *w) { return ctx(w)->gpu_ms; }
size_t vitm_weight_bytes(const VitmW *w) { return ctx(w)->wb.length; }
size_t vitm_act_bytes(const VitmW *w) {
  VitmCtx *c = ctx(w);
  id<MTLBuffer> bs[] = {c->b_planar, c->b_rows, c->b_x, c->b_cur, c->b_ao, c->b_qr, c->b_qkv, c->b_up, c->b_kh, c->b_vh,
                        c->b_kd, c->b_pos, c->b_ppos, c->b_embd};
  size_t s = 0;
  for (size_t i = 0; i < sizeof bs / sizeof bs[0]; i++) s += bs[i].length;  // (a nil buffer's length is 0)
  return s;
}

static size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

static id<MTLComputePipelineState> vt_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name, char *err) {
  id<MTLFunction> f = [lib newFunctionWithName:@(name)];
  if (!f) { snprintf(err, 256, "vitm: no kernel %s", name); return nil; }
  NSError *e = nil;
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
  if (!p) snprintf(err, 256, "vitm: pipeline %s: %s", name, e.localizedDescription.UTF8String);
  return p;
}

VitmW *vitm_load(const char *mmproj, uint32_t n_embd_llm, char *err) {
  @autoreleasepool {
    VMm *m = vmm_open(mmproj, n_embd_llm, err);
    if (!m) return NULL;
    VitmCtx *c = [VitmCtx new];
    vmm_hparams(m, &c->h);
    c->E = (int)c->h.n_embd, c->H = (int)c->h.n_head, c->D = c->E / c->H, c->F = (int)c->h.n_ff, c->L = (int)c->h.n_layer;
    c->P = (int)c->h.patch, c->M = (int)c->h.n_merge, c->PD = (int)c->h.proj_dim, c->G = (int)(c->h.image_size / c->h.patch);
    // the kernels bake the 16x16x3 patch and the 2x2 merge; the head dim is a template (vitm.metal): 72 (the 9B's
    // tower, 2 x 36 rope pairs) or 64 (ENGINE-IPHONEVIS: the 4B's, 2 x 32)
    if ((c->D != 72 && c->D != 64) || c->H * c->D != c->E || c->P != 16 || c->M != 2 || c->E % 4) {
      snprintf(err, 256, "vitm: unsupported tower (head dim %d, heads %d, width %d, patch %d, merge %d)", c->D, c->H, c->E, c->P, c->M);
      vmm_close(m);
      return NULL;
    }
    c->dev = MTLCreateSystemDefaultDevice();
    if (!c->dev) { snprintf(err, 256, "vitm: no Metal device"); vmm_close(m); return NULL; }
    // the weight layout: (name, slot, F16?, elements)
    const size_t E = c->E, F = c->F, K = 3 * (size_t)c->P * c->P;
    enum { NG = 9 };
    struct { const char *name; size_t *slot; int f16; size_t n; } g[NG] = {
      {"v.patch_embd.weight", &c->o_pe0, 1, E * K}, {"v.patch_embd.weight.1", &c->o_pe1, 1, E * K},
      {"v.patch_embd.bias", &c->o_peb, 0, E}, {"v.post_ln.weight", &c->o_postw, 0, E}, {"v.post_ln.bias", &c->o_postb, 0, E},
      {"mm.0.weight", &c->o_mm0w, 1, 16 * E * E}, {"mm.0.bias", &c->o_mm0b, 0, 4 * E},
      {"mm.2.weight", &c->o_mm2w, 1, 4 * E * (size_t)c->PD}, {"mm.2.bias", &c->o_mm2b, 0, (size_t)c->PD}};
    const size_t ln[T_NL] = {E, E, 3 * E * E, 3 * E, E * E, E, E, E, F * E, F, E * F, E};
    c->o_l = calloc((size_t)c->L * T_NL, sizeof *c->o_l);
    size_t total = 0;
    char nm[96];
    for (int pass = 0; pass < 2; pass++) {
      size_t off = 0;
      for (int i = 0; i < NG + c->L * T_NL; i++) {
        const char *name;
        size_t *slot, n;
        int f16;
        if (i < NG) name = g[i].name, slot = g[i].slot, n = g[i].n, f16 = g[i].f16;
        else {
          const int il = (i - NG) / T_NL, t = (i - NG) % T_NL;
          snprintf(nm, sizeof nm, "v.blk.%d.%s", il, LNAME[t]);
          name = nm, slot = &c->o_l[(size_t)il * T_NL + t], n = ln[t];
          f16 = t == T_QKVW || t == T_OW || t == T_UPW || t == T_DOWNW;
        }
        int ty = -1;
        size_t ne = 0, bytes = vmm_bytes(m, name, &ty, &ne);
        if (!bytes || ne != n || ty != f16) {
          snprintf(err, 256, "vitm: %s: %zu elements type %d, want %zu type %d", name, ne, ty, n, f16);
          vmm_close(m);
          return NULL;
        }
        if (pass == 0) *slot = off;
        else if (vmm_raw(m, name, (char *)c->wb.contents + *slot, err)) { vmm_close(m); return NULL; }
        off += al256(bytes);
      }
      if (pass == 0) {
        total = off;
        c->wb = [c->dev newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!c->wb) { snprintf(err, 256, "vitm: weight buffer of %zu bytes failed", total); vmm_close(m); return NULL; }
      }
    }
    c->pos = vmm_f32(m, "v.position_embd.weight", E * c->G * c->G, err);
    vmm_close(m);
    if (!c->pos) return NULL;

    NSString *src = [[NSString alloc] initWithBytes:vitm_metal length:vitm_metal_len encoding:NSUTF8StringEncoding];
    NSError *e = nil;
    id<MTLLibrary> lib = [c->dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&e];
    if (!lib) {
      fprintf(stderr, "vitm: library: %s\n", e.localizedDescription.UTF8String);
      snprintf(err, 256, "vitm: library compile failed (above)");
      return NULL;
    }
    if (!(c->p_mm32 = vt_pipe(c->dev, lib, "vt_mm_f32", err)) || !(c->p_mm16 = vt_pipe(c->dev, lib, "vt_mm_f16", err)) ||
        !(c->p_fa = vt_pipe(c->dev, lib, c->D == 64 ? "vt_fa_dk64" : "vt_fa_dk72", err)) || !(c->p_ln = vt_pipe(c->dev, lib, "vt_ln", err)) ||
        !(c->p_rope = vt_pipe(c->dev, lib, c->D == 64 ? "vt_rope_d64" : "vt_rope_d72", err)) || !(c->p_gelu = vt_pipe(c->dev, lib, "vt_gelu", err)) ||
        !(c->p_add = vt_pipe(c->dev, lib, "vt_add", err)) || !(c->p_embed = vt_pipe(c->dev, lib, "vt_embed", err)) ||
        !(c->p_i2c = vt_pipe(c->dev, lib, "vt_im2col", err)))
      return NULL;
    if (c->p_ln.maxTotalThreadsPerThreadgroup < (NSUInteger)c->E / 4 || c->p_mm32.maxTotalThreadsPerThreadgroup < 128 ||
        c->p_fa.maxTotalThreadsPerThreadgroup < 128) {
      snprintf(err, 256, "vitm: threadgroup limits too small (ln %lu, mm %lu, fa %lu)", (unsigned long)c->p_ln.maxTotalThreadsPerThreadgroup,
               (unsigned long)c->p_mm32.maxTotalThreadsPerThreadgroup, (unsigned long)c->p_fa.maxTotalThreadsPerThreadgroup);
      return NULL;
    }
    c->q = [c->dev newCommandQueue];
    c->pos_px = c->pos_py = -1;
    return (VitmW *)(__bridge_retained void *)c;
  }
}

// ggml_interpolate(BILINEAR | ALIGN_CORNERS) of the G x G table to px x py: vit.c's pos_resize, the same expressions
static void pos_resize(const float *tab, int G, int E, int px, int py, float *out) {
  if (px == G && py == G) { memcpy(out, tab, sizeof(float) * (size_t)G * G * E); return; }
  const float sf0 = px > 1 && G > 1 ? (float)(px - 1) / (G - 1) : (float)px / G;
  const float sf1 = py > 1 && G > 1 ? (float)(py - 1) / (G - 1) : (float)py / G;
  for (int y = 0; y < py; y++) {
    const float f01 = (float)y / sf1;
    long i01 = (long)floorf(f01);
    if (i01 < 0) i01 = 0;
    if (i01 > G - 1) i01 = G - 1;
    const long i01p = i01 + 1 > G - 1 ? G - 1 : i01 + 1;
    float fd1 = f01 - (float)i01;
    fd1 = fd1 < 0 ? 0 : fd1 > 1 ? 1 : fd1;
    for (int x = 0; x < px; x++) {
      const float f00 = (float)x / sf0;
      long i00 = (long)floorf(f00);
      if (i00 < 0) i00 = 0;
      if (i00 > G - 1) i00 = G - 1;
      const long i00p = i00 + 1 > G - 1 ? G - 1 : i00 + 1;
      float fd0 = f00 - (float)i00;
      fd0 = fd0 < 0 ? 0 : fd0 > 1 ? 1 : fd0;
      const float *s00 = tab + ((size_t)i01 * G + i00) * E, *s10 = tab + ((size_t)i01 * G + i00p) * E;
      const float *s01 = tab + ((size_t)i01p * G + i00) * E, *s11 = tab + ((size_t)i01p * G + i00p) * E;
      float *o = out + ((size_t)y * px + x) * E;
      for (int e = 0; e < E; e++)
        o[e] = s00[e] * (1.0f - fd0) * (1.0f - fd1) + s10[e] * fd0 * (1.0f - fd1) + s01[e] * (1.0f - fd0) * fd1 +
               s11[e] * fd0 * fd1;
    }
  }
}

static id<MTLBuffer> nbuf(VitmCtx *c, size_t bytes) {
  return [c->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static int ensure(VitmCtx *c, size_t N, size_t pix, char *err) {
  const size_t E = c->E, F = c->F, H = c->H, D = c->D;
  if (pix > c->pixcap) {
    c->b_planar = nbuf(c, 3 * pix * sizeof(float));
    if (!c->b_planar) { snprintf(err, 256, "vitm: out of GPU memory (planar)"); return -1; }
    c->pixcap = pix;
  }
  if (N <= c->ncap) return 0;
  const size_t npad = (N + 63) / 64 * 64, big = N * (F > 4 * E ? F : 4 * E);
  c->b_rows = nbuf(c, N * 3 * c->P * c->P * sizeof(uint16_t));
  c->b_x = nbuf(c, N * E * 4), c->b_cur = nbuf(c, N * E * 4), c->b_ao = nbuf(c, N * E * 4), c->b_qr = nbuf(c, N * E * 4);
  c->b_kd = nbuf(c, N * E * 4), c->b_qkv = nbuf(c, N * 3 * E * 4), c->b_up = nbuf(c, big * 4);
  // K, V: [H][npad][D] half; + 256 halves of slack (at D 72 the PV step reads columns up to 128 of the last row block)
  c->b_kh = nbuf(c, (H * npad * D + 256) * 2), c->b_vh = nbuf(c, (H * npad * D + 256) * 2);
  c->b_pos = nbuf(c, N * E * 4), c->b_ppos = nbuf(c, 2 * N * sizeof(int)), c->b_embd = nbuf(c, N / 4 * c->PD * 4);
  if (!c->b_rows || !c->b_x || !c->b_cur || !c->b_ao || !c->b_qr || !c->b_kd || !c->b_qkv || !c->b_up || !c->b_kh ||
      !c->b_vh || !c->b_pos || !c->b_ppos || !c->b_embd) {
    c->ncap = 0;
    snprintf(err, 256, "vitm: out of GPU memory (%zu patches)", N);
    return -1;
  }
  memset(c->b_kh.contents, 0, c->b_kh.length), memset(c->b_vh.contents, 0, c->b_vh.length);
  c->ncap = N, c->pos_px = c->pos_py = -1;
  return 0;
}

// ---------------------------------------------------------------- dispatch helpers (one per kernel)

static void d_mm(VitmCtx *c, id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, size_t woff, id<MTLBuffer> B,
                 id<MTLBuffer> Dst, size_t boff, int k, int o, int n, int flags) {
  const VtMmArgs a = {k, o, n, k, o, flags};
  [enc setComputePipelineState:p];
  [enc setBytes:&a length:sizeof a atIndex:0];
  [enc setBuffer:c->wb offset:woff atIndex:1];
  [enc setBuffer:B offset:0 atIndex:2];
  [enc setBuffer:Dst offset:0 atIndex:3];
  [enc setBuffer:c->wb offset:(flags & MM_BIAS) ? boff : 0 atIndex:4];
  [enc setThreadgroupMemoryLength:8192 atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 31) / 32, ((NSUInteger)o + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void d_ln(VitmCtx *c, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, size_t woff, size_t boff, id<MTLBuffer> dst, int n) {
  const VtLnArgs a = {c->E, c->E / 4, c->h.eps};
  [enc setComputePipelineState:c->p_ln];
  [enc setBytes:&a length:sizeof a atIndex:0];
  [enc setBuffer:src offset:0 atIndex:1];
  [enc setBuffer:c->wb offset:woff atIndex:2];
  [enc setBuffer:c->wb offset:boff atIndex:3];
  [enc setBuffer:dst offset:0 atIndex:4];
  [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
  // ggml's nth: the smallest power of two >= ne00_t, capped at ne00_t rounded up to 32 (= 288 at width 1152, 256 at 1024)
  NSUInteger nth = 32;
  while (nth < (NSUInteger)a.ne00_t && nth < c->p_ln.maxTotalThreadsPerThreadgroup) nth *= 2;
  if (nth > c->p_ln.maxTotalThreadsPerThreadgroup) nth = c->p_ln.maxTotalThreadsPerThreadgroup;
  if (nth > ((NSUInteger)a.ne00_t + 31) / 32 * 32) nth = ((NSUInteger)a.ne00_t + 31) / 32 * 32;
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
}

static void d_elt(id<MTLComputeCommandEncoder> enc, NSUInteger n) {
  [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void d_add(VitmCtx *c, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> y, id<MTLBuffer> a, size_t n) {
  const uint32_t n4 = (uint32_t)(n / 4);
  [enc setComputePipelineState:c->p_add];
  [enc setBuffer:y offset:0 atIndex:0];
  [enc setBuffer:a offset:0 atIndex:1];
  [enc setBytes:&n4 length:sizeof n4 atIndex:2];
  d_elt(enc, n4);
}

static void d_gelu(VitmCtx *c, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, size_t n) {
  const uint32_t n4 = (uint32_t)(n / 4);
  [enc setComputePipelineState:c->p_gelu];
  [enc setBuffer:x offset:0 atIndex:0];
  [enc setBytes:&n4 length:sizeof n4 atIndex:1];
  d_elt(enc, n4);
}

// ---------------------------------------------------------------- the tower

static int wants(const VitOpt *o, const char *name) { return o && o->dump && (!o->want || o->want(o->dump_ud, name)); }

int vitm_encode(VitmW *w, const float *planar, int nx, int ny, const VitOpt *o, float *embd, char *err) {
  @autoreleasepool {
    VitmCtx *c = ctx(w);
    const int E = c->E, H = c->H, D = c->D, F = c->F, P = c->P, M = c->M, PD = c->PD;
    if (o && !o->f16act) { snprintf(err, 256, "vitm: only the registered F16-staged mode (vit.c has the plain-F32 mode)"); return -1; }
    if (nx % (P * M) || ny % (P * M)) { snprintf(err, 256, "image %dx%d not a multiple of %d", nx, ny, P * M); return -1; }
    const int px = nx / P, py = ny / P;
    const size_t N = (size_t)px * py, T = N / 4, npad = (N + 63) / 64 * 64, K = 3 * (size_t)P * P;
    if (ensure(c, N, (size_t)nx * ny, err)) return -1;
    const int unfused_all = getenv("VITM_UNFUSED") != NULL;  // test hook: the fused epilogues vs separate kernels
    memcpy(c->b_planar.contents, planar, sizeof(float) * 3 * (size_t)nx * ny);
    // the position rows (merge order) and the rope positions: once per image size
    if (c->pos_px != px || c->pos_py != py) {
      float *pr = malloc(sizeof(float) * N * E), *dst = c->b_pos.contents;
      int *pp = c->b_ppos.contents;
      if (!pr) { snprintf(err, 256, "out of memory"); return -1; }
      pos_resize(c->pos, c->G, E, px, py, pr);
      size_t r = 0;
      for (int y = 0; y < py; y += M)
        for (int x = 0; x < px; x += M)
          for (int dy = 0; dy < M; dy++)
            for (int dx = 0; dx < M; dx++, r++) {
              memcpy(dst + r * E, pr + ((size_t)(y + dy) * px + x + dx) * E, sizeof(float) * E);
              pp[r] = y + dy, pp[N + r] = x + dx;
            }
      free(pr);
      c->pos_px = px, c->pos_py = py;
    }
    // K, V rows [N, npad) of every head must be zero (a larger earlier image may have left values there)
    for (int hh = 0; hh < H && npad > N; hh++) {
      memset((uint16_t *)c->b_kh.contents + ((size_t)hh * npad + N) * D, 0, (npad - N) * D * 2);
      memset((uint16_t *)c->b_vh.contents + ((size_t)hh * npad + N) * D, 0, (npad - N) * D * 2);
    }

    c->gpu_ms = 0;
    __block id<MTLCommandBuffer> cb = [c->q commandBuffer];
    __block id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    __block int failed = 0;
    // end the command buffer here and wait (a dump reads the shared buffers), then open the next one
    int (^flush)(void) = ^int(void) {
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        snprintf(err, 256, "vitm: command buffer failed: %s", cb.error.localizedDescription.UTF8String ?: "?");
        failed = 1;
        return -1;
      }
      c->gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      cb = [c->q commandBuffer];
      enc = [cb computeCommandEncoder];
      return 0;
    };
    char nm[64];
#define DUMP(name, buf, ne0, rows)                                                                     \
  do {                                                                                                 \
    if (wants(o, name)) {                                                                              \
      if (flush()) return -1;                                                                          \
      o->dump(o->dump_ud, name, (const float *)(buf).contents, (ne0), (int)(rows));                   \
    }                                                                                                  \
  } while (0)
#define LDUMP(base, il, buf, ne0, rows) do { snprintf(nm, sizeof nm, "%s-%d", base, il); DUMP(nm, buf, ne0, rows); } while (0)

    // patch embed: im2col (F16, merge order) -> conv(W0), conv(W1) -> sum + bias -> + pos
    {
      const VtI2cArgs a = {nx, ny, px, (int)N};
      [enc setComputePipelineState:c->p_i2c];
      [enc setBytes:&a length:sizeof a atIndex:0];
      [enc setBuffer:c->b_planar offset:0 atIndex:1];
      [enc setBuffer:c->b_rows offset:0 atIndex:2];
      d_elt(enc, N * K);
      d_mm(c, enc, c->p_mm16, c->o_pe0, c->b_rows, c->b_cur, 0, (int)K, E, (int)N, 0);
      d_mm(c, enc, c->p_mm16, c->o_pe1, c->b_rows, c->b_ao, 0, (int)K, E, (int)N, 0);
      const uint32_t ne[2] = {(uint32_t)(N * E / 4), (uint32_t)(E / 4)};
      [enc setComputePipelineState:c->p_embed];
      [enc setBuffer:c->b_x offset:0 atIndex:0];
      [enc setBuffer:c->b_cur offset:0 atIndex:1];
      [enc setBuffer:c->b_ao offset:0 atIndex:2];
      [enc setBuffer:c->wb offset:c->o_peb atIndex:3];
      [enc setBytes:ne length:sizeof ne atIndex:4];
      d_elt(enc, ne[0]);
      DUMP("patch_bias", c->b_x, E, N);
      d_add(c, enc, c->b_x, c->b_pos, N * E);
      DUMP("inp_pos_emb", c->b_x, E, N);
    }

    const float scale = 1.0f / sqrtf((float)D);
    for (int il = 0; il < c->L; il++) {
      const size_t *lo = c->o_l + (size_t)il * T_NL;
      int unf = unfused_all;  // a dump between the GEMM and its epilogue needs the separate kernels
      static const char *const split[] = {"attn_out", "ffn_inp", "ffn_up_b", "ffn_gelu", "ffn_out"};
      for (int s = 0; s < 5 && !unf; s++) { snprintf(nm, sizeof nm, "%s-%d", split[s], il); unf = wants(o, nm); }
      snprintf(nm, sizeof nm, "Kcur_rope-%d", il);
      const int kdbg = wants(o, nm);

      d_ln(c, enc, c->b_x, lo[T_LN1W], lo[T_LN1B], c->b_cur, (int)N);
      LDUMP("ln1", il, c->b_cur, E, N);
      d_mm(c, enc, c->p_mm32, lo[T_QKVW], c->b_cur, c->b_qkv, lo[T_QKVB], E, 3 * E, (int)N, MM_BIAS);
      {
        const VtRopeArgs a = {(int)N, E, H, (int)npad, kdbg};
        [enc setComputePipelineState:c->p_rope];
        [enc setBytes:&a length:sizeof a atIndex:0];
        [enc setBuffer:c->b_qkv offset:0 atIndex:1];
        [enc setBuffer:c->b_ppos offset:0 atIndex:2];
        [enc setBuffer:c->b_qr offset:0 atIndex:3];
        [enc setBuffer:c->b_kh offset:0 atIndex:4];
        [enc setBuffer:c->b_vh offset:0 atIndex:5];
        [enc setBuffer:c->b_kd offset:0 atIndex:6];
        d_elt(enc, N * H * (D / 2));
      }
      LDUMP("Qcur_rope", il, c->b_qr, E, N);
      LDUMP("Kcur_rope", il, c->b_kd, E, N);
      {
        const VtFaArgs a = {(int)N, (int)npad, E, E, scale};
        [enc setComputePipelineState:c->p_fa];
        [enc setBytes:&a length:sizeof a atIndex:0];
        [enc setBuffer:c->b_qr offset:0 atIndex:1];
        [enc setBuffer:c->b_kh offset:0 atIndex:2];
        [enc setBuffer:c->b_vh offset:0 atIndex:3];
        [enc setBuffer:c->b_ao offset:0 atIndex:4];
        // Q (8) x (DK + 2 PV halves + 2 C floats): sq + so, then ss; PV = D padded to 64
        [enc setThreadgroupMemoryLength:8 * (D + 2 * ((D + 63) / 64 * 64) + 4 * 64) * 2 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((N + 7) / 8, (NSUInteger)H, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
      }
      LDUMP("kqv_out", il, c->b_ao, E, N);
      if (unf) {
        d_mm(c, enc, c->p_mm32, lo[T_OW], c->b_ao, c->b_cur, lo[T_OB], E, E, (int)N, MM_BIAS);
        LDUMP("attn_out", il, c->b_cur, E, N);
        d_add(c, enc, c->b_x, c->b_cur, N * E);
      } else {
        d_mm(c, enc, c->p_mm32, lo[T_OW], c->b_ao, c->b_x, lo[T_OB], E, E, (int)N, MM_BIAS | MM_RES);
      }
      LDUMP("ffn_inp", il, c->b_x, E, N);
      d_ln(c, enc, c->b_x, lo[T_LN2W], lo[T_LN2B], c->b_cur, (int)N);
      LDUMP("ffn_inp_normed", il, c->b_cur, E, N);
      if (unf) {
        d_mm(c, enc, c->p_mm32, lo[T_UPW], c->b_cur, c->b_up, lo[T_UPB], E, F, (int)N, MM_BIAS);
        LDUMP("ffn_up_b", il, c->b_up, F, N);
        d_gelu(c, enc, c->b_up, N * F);
        LDUMP("ffn_gelu", il, c->b_up, F, N);
        d_mm(c, enc, c->p_mm32, lo[T_DOWNW], c->b_up, c->b_cur, lo[T_DOWNB], F, E, (int)N, MM_BIAS);
        LDUMP("ffn_out", il, c->b_cur, E, N);
        d_add(c, enc, c->b_x, c->b_cur, N * E);
      } else {
        d_mm(c, enc, c->p_mm32, lo[T_UPW], c->b_cur, c->b_up, lo[T_UPB], E, F, (int)N, MM_BIAS | MM_GELU);
        d_mm(c, enc, c->p_mm32, lo[T_DOWNW], c->b_up, c->b_x, lo[T_DOWNB], F, E, (int)N, MM_BIAS | MM_RES);
      }
      LDUMP("layer_out", il, c->b_x, E, N);
    }
    d_ln(c, enc, c->b_x, c->o_postw, c->o_postb, c->b_cur, (int)N);
    LDUMP("norm_b", c->L, c->b_cur, E, N);
    // merger: 4 consecutive (merge-ordered) patches = one 4E row -> mm.0 + b -> GELU -> mm.2 + b
    if (unfused_all || wants(o, "ffn_up_b") || wants(o, "ffn_gelu")) {
      d_mm(c, enc, c->p_mm32, c->o_mm0w, c->b_cur, c->b_up, c->o_mm0b, 4 * E, 4 * E, (int)T, MM_BIAS);
      DUMP("ffn_up_b", c->b_up, 4 * E, T);
      d_gelu(c, enc, c->b_up, T * 4 * E);
      DUMP("ffn_gelu", c->b_up, 4 * E, T);
    } else {
      d_mm(c, enc, c->p_mm32, c->o_mm0w, c->b_cur, c->b_up, c->o_mm0b, 4 * E, 4 * E, (int)T, MM_BIAS | MM_GELU);
    }
    d_mm(c, enc, c->p_mm32, c->o_mm2w, c->b_up, c->b_embd, c->o_mm2b, 4 * E, PD, (int)T, MM_BIAS);
    if (flush() || failed) return -1;
    [enc endEncoding];  // the empty buffer flush() opened, never committed
    memcpy(embd, c->b_embd.contents, sizeof(float) * T * PD);
    if (wants(o, "embd")) o->dump(o->dump_ud, "embd", embd, PD, (int)T);
#undef DUMP
#undef LDUMP
    return 0;
  }
}
