#if defined(BEND_CUDA) && defined(LLB_LAUNCH)
// LLB_ATTN_SK8=1 (default off): decode attention split over 8 key segments per head, then
// combined (llb_attn4_sk8 / _combine, tools/payload-sk8add.patch). llb_attn4 stays the default.
static bool llb_sk8_enabled(void) {
  const char* e = getenv("LLB_ATTN_SK8"); return e && !strcmp(e, "1");
}
// its private planes hold 2^17 slots a head for <= 128 heads; past that the canonical
// llb_attn4 runs (its planes grow with dT, tools/payload-anyshape.patch)
static bool llb_sk8_fits(u32 c, u32 nh) { return nh <= 128 && (c & 31) <= 17; }
static bool llb_sk8_launch(CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr out,
  u32 c, u32 grp, u32 nh) {
  static CUdeviceptr tmp, sc, pr;
  // Private planes, allocated only for the opt-in; default launch is unchanged.
  if (!tmp && cuMemAlloc(&tmp, 128 * 8 * 258 * sizeof(float)) != CUDA_SUCCESS) return false;
  if (!sc && cuMemAlloc(&sc, 128ull * (1u << 17) * sizeof(float)) != CUDA_SUCCESS) return false;
  if (!pr && cuMemAlloc(&pr, 128ull * (1u << 17) * sizeof(float)) != CUDA_SUCCESS) return false;
  return LLB_LAUNCH(gpu_sk8, nh * 8, 1, 1024, &q, &k, &v, &tmp, &sc, &pr, &c, &grp, &nh)
    && LLB_LAUNCH(gpu_sk8_combine, nh, 1, 256, &tmp, &out);
}
#endif
#if defined(LLB_LAUNCH) && defined(LLB_WARP_FID)
#include "llb_backend.h"
#include "llb_formats.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>
// The fbn / linb / fulb chains (tools/payload-fbn, -linb, -fulb.patch) as
// functions of the op, its output y and the GEMV grid: their warp_serve
// branches and the pass bang (tools/payload-pass.patch) run the same launches
// the device's Q8Cat as launch bytes: 4 + 4 device pointers, 4 row ends
typedef struct {
  u64 W[4];
  u64 y[4];
  u32 end[4];
} Q8Cat;

// The shape record (comp.ts LlbShape): the pass bang's, from the manifest
// (pass_mail refuses one outside the kernels' envelope); the 9B's until then,
// so the legacy kinds 17 / 18 / 22 run the 9B
static LlbShape llb_sh = { 4096, 12288, 16, 4, 256, 16, 32, 128 };

// tools/payload-anyshape.patch: a full-attention layer is one whose table row holds attn_q
// (Offs word 14), pass_mail's rule, so any full_attention_interval (was (i + 1) % 4 == 0)
#define llb_full(i) (gpu_mail->pt[(i)][14] != 0)

// the engine's cdepth: a Q8 GEMV's column depth, ceil(log2(cols)) - 2
static u32 llb_cdepth(u32 cols) {
  u32 d = 0;
  while ((1u << d) < cols) {
    d += 1;
  }
  return d - 2;
}

// tools/payload-anyshape.patch: whether a tuned GEMV body covers a cols-wide row of format fm
// -- Q8_0 llb_warp_q8's trees at d 7..12 (6 carry slots), F32 llb_warp at d 9..12, Q4_K /
// Q6_K the pair body (whole superblock quads, d 8..13) or the tile body (d 9..11), Q5_K the
// pair body only. Elsewhere llb_gemv_any runs the row (any width; not the fixed tree).
static bool llb_tuned(u32 fm, u32 cols) {
  u32 d = llb_cdepth(cols), cg = cols / 4;
  if (fm == LLB_FM_Q8_0) {
    return d >= 7 && d <= 12;
  }
  if (fm == LLB_FM_F32) {
    return d >= 9 && d <= 12;
  }
  if (fm == LLB_FM_Q4_K || fm == LLB_FM_Q6_K) {
    return (cg % 256 == 0 && d >= 8 && d <= 13) || (d >= 9 && d <= 11);
  }
  return fm == LLB_FM_Q5_K && cg % 256 == 0 && d >= 8 && d <= 13;
}

// y = W x by llb_gemv_any: n rows of cols in format fm (F32, Q8_0, Q4_K, Q6_K, Q5_K)
static void llb_gany(u32 fm, u64 w, u32 cols, u64 x, u64 y, u32 n) {
  u32 rb = LLB_FM_RB(fm, cols), cg = cols / 4, g = (n + 3) / 4;
  g = g < 4096 ? g : 4096;
  if (fm != LLB_FM_F32 && fm != LLB_FM_Q8_0 && fm != LLB_FM_Q4_K && fm != LLB_FM_Q6_K && fm != LLB_FM_Q5_K) {
    err_fail("warp pass: no GEMV body for a table format");
  }
  // reports/cuda-catchup: K-quant rows on llb_gemv_anyk (moe_dotn: the same leaf and order, two rows a
  // warp, loads first)
  bool kq = fm == LLB_FM_Q4_K || fm == LLB_FM_Q6_K || fm == LLB_FM_Q5_K;
  if (kq) {
    g = (n + 7) / 8;
    g = g < 4096 ? g : 4096;
  }
  if (!LLB_LAUNCH(kq ? gpu_ganyk : gpu_gany, g, 1, 128, &w, &rb, &fm, &cg, &x, &y, &n)) {
    err_fail("device launch failed");
  }
}

// one GEMV of a pass tensor in its own format (the table's fm): n rows of
// cols, the engine tree at llb_cdepth(cols); Q8_0 / Q4_K / Q6_K / Q5_K bodies.
// LLB_Q4_DP4A bit 32: Q6_K on the two planes (llb_q6dp2, twin ref_q6row2)
static u32 llb_q4dp;
static void llb_kgemv(u32 fm, u64 w, u32 cols, u64 x, u64 y, u32 n, u32 grid) {
  u32 rb = LLB_FM_RB(fm, cols), cg = cols / 4, d = llb_cdepth(cols);
  // pass_mail admits F32 GEMVs only for recurrent ssm_beta / ssm_alpha.
  // llb_warp addresses float4 rows, so its stride is cols / 4, not bytes.
  if (fm == LLB_FM_F32) {
    u32 rq = cols / 4;
    if (!llb_tuned(fm, cols)) {
      llb_gany(fm, w, cols, x, y, n);
      return;
    }
    if (!LLB_LAUNCH(gpu_warp, grid, 1, 128, &w, &rq, &cg, &x, &y, &n, &d)) {
      err_fail("device launch failed");
    }
    return;
  }
  if (fm == LLB_FM_Q6_K && (llb_q4dp & 32) && cols % 256 == 0) {
    llb_dptr qx = gpu_q8c, dx = qx + 2 * cols;
    u32 q8G = cols / 256, g6 = (n + 3) / 4;
    if (!LLB_LAUNCH(gpu_q8x2, q8G, 1, 256, &x, &qx, &dx, &cols)
      || !LLB_LAUNCH(gpu_q6dp2, g6, 1, 128, &w, &rb, &qx, &dx, &y, &n)) {
      err_fail("device launch failed");
    }
    return;
  }
  if (fm != LLB_FM_Q8_0 && fm != LLB_FM_Q4_K && fm != LLB_FM_Q6_K && fm != LLB_FM_Q5_K) {
    err_fail("warp pass: no GEMV body for a table format");
  }
  if (!llb_tuned(fm, cols)) {
    llb_gany(fm, w, cols, x, y, n);
    return;
  }
  // Q4_K / Q6_K at the default grid take 984 = 3 waves of 82 SMs x 4 blocks
  // (Q4B M5; Q5_K is best at 768). The grid moves rows between warps only
  if ((fm == LLB_FM_Q4_K || fm == LLB_FM_Q6_K) && grid == 768) {
    grid = 984;
  }
  if (!LLB_LAUNCH(fm == LLB_FM_Q5_K ? gpu_warp5 : fm == LLB_FM_Q6_K ? gpu_warp6 : fm == LLB_FM_Q4_K ? gpu_warpk : gpu_warpq,
    grid, 1, 128, &w, &rb, &cg, &x, &y, &n, &d)) {
    err_fail("device launch failed");
  }
}

// llb_kgemv with the residual in its store (tools/payload-q4glue.patch):
// y[r] = x1[r] + row, Q4_K / Q6_K pair body; false (nothing launched) where
// only the plain GEMV applies.
// llb_q4glue (env LLB_Q4GLUE, default 3): bit 1 z + beta + alpha, 2 down +
// residual; a cleared bit runs that glue as its own launch (A/B)
static u32 llb_q4glue = 3;
// llb_q4dp (env LLB_Q4_DP4A=1, default 0; tools/payload-dp4a.patch): the
// FFN's Q4_K GEMVs on int8 x and dp4a (llama.cpp's mmvq numerics, the K1 twin
// tools/q4dp/q4dp_ref.h) where cols is whole superblocks and d is 9..13.
// Bits (q27c bisect): 1 gate+up, 2 down; LLB_Q4_DP4A 1 = both, 2 = gate+up
// only, 3 = down only. Bits 4 / 8: that class on two int8 planes (tools/
// payload-dp4b.patch, twin tools/q27c/q4dp2_ref.h); 4 = both two-plane, 5 = down
// two-plane only, 6 = gate+up two-plane only (the other class one plane).
// Bit 16 (q27c stage 3, LLB_Q4_DP4A 7 = 4 + this): the Q8_0 cats and ssm_out
// on the two planes (llb_q8dp2 / _cat, twin ref_q8row2). Bit 32: Q6_K (llb_kgemv);
// LLB_Q4_DP4A 8 = 7 + this
static u32 llb_q4dp = 0;
static bool llb_q4dp_ok(u32 cols, u32 bit) {
  u32 d = llb_cdepth(cols);
  return (llb_q4dp & bit) && cols % 256 == 0 && d >= 9 && d <= 13;
}

// LLB_DUMP=dir (q27c diagnosis, default off): per step of prompt k (a new one
// at pos 0) the layer outputs a, x into dir/p<k>_h<pos>.bin (2 d f32 a
// layer), the logits into dir/p<k>_l<pos>.bin; at pos LLB_DUMP_POS each
// layer's FFN I/O x1, h2, g, u, silu, y into dir/p<k>_f<pos>_<i>.bin
static const char* llb_dump;
static int llb_dump_pos = -1, llb_dump_k = -1;
static void llb_dump_put(const char* tag, u32 pos, int i, const char* mode, const llb_dptr* p, const u32* n, u32 m) {
  char path[512];
  snprintf(path, sizeof path, i < 0 ? "%s/p%d_%s%u.bin" : "%s/p%d_%s%u_%d.bin", llb_dump, llb_dump_k, tag, pos, i);
  FILE* f = fopen(path, mode);
  if (!f) {
    err_fail("LLB_DUMP: cannot open a dump file");
  }
  for (u32 j = 0; j < m; j += 1) {
    static float buf[1 << 18];
    if (n[j] > (1u << 18) || !llb_read(buf, p[j], (size_t)n[j] * 4) || fwrite(buf, 4, n[j], f) != n[j]) {
      err_fail("LLB_DUMP: read / write failed");
    }
  }
  fclose(f);
}
// CLEF-3: exact prompt rows, after output RMS norm; no device arithmetic changes.
static const char* llb_hidden_path;
static unsigned char llb_hidden_positions[16384];
static u32 llb_hidden_tokens;
static bool llb_hidden_ready, llb_hidden_written;
static void llb_hidden_init(void) {
  if (llb_hidden_ready) return;
  llb_hidden_ready = true;
  const char* path = getenv("LLB_DUMP_HIDDEN");
  if (!path || !*path) return;
  const char* count = getenv("LLB_DUMP_HIDDEN_TOKENS");
  char* end;
  long n = count ? strtol(count, &end, 10) : 0;
  if (n < 1 || n > 16384 || *end) err_fail("LLB_DUMP_HIDDEN: invalid prompt count");
  const char* pf = getenv("LLB_PREFILL");
  if ((pf && !strcmp(pf, "fast")) || getenv("LLB_SPEC") || getenv("LLB_MTP"))
    err_fail("LLB_DUMP_HIDDEN: exact pass only, no speculation");
  llb_hidden_path = path;
  llb_hidden_tokens = (u32)n;
  const char* list = getenv("LLB_DUMP_HIDDEN_POS");
  if (!list || !*list) memset(llb_hidden_positions, 1, n);
  else while (*list) {
    long pos = strtol(list, &end, 10);
    if (end == list || pos < 0 || pos >= n || (*end && *end != ','))
      err_fail("LLB_DUMP_HIDDEN: invalid position list");
    llb_hidden_positions[pos] = 1;
    if (*end && !end[1]) err_fail("LLB_DUMP_HIDDEN: trailing comma");
    list = *end ? end + 1 : end;
  }
}
static void llb_hidden_put(u32 pos, u32 width, llb_dptr row) {
  if (!llb_hidden_path || pos >= llb_hidden_tokens || !llb_hidden_positions[pos]) return;
  static float buf[32768];
  u32 header[2] = {pos, width};
  if (width > 32768 || !llb_read(buf, row, (size_t)width * 4))
    err_fail("LLB_DUMP_HIDDEN: read failed");
  FILE* f = fopen(llb_hidden_path, llb_hidden_written ? "ab" : "wb");
  if (!f || fwrite(header, 4, 2, f) != 2 || fwrite(buf, 4, width, f) != width)
    err_fail("LLB_DUMP_HIDDEN: write failed");
  if (fclose(f)) err_fail("LLB_DUMP_HIDDEN: close failed");
  llb_hidden_written = true;
}
static bool llb_kgemv_res(u32 fm, u64 w, u32 cols, u64 x, u64 y, u32 n, u32 grid, u64 x1) {
  u32 rb = LLB_FM_RB(fm, cols), cg = cols / 4, d = llb_cdepth(cols);
  // (reports/cuda-catchup: whole superblocks, cg % 64, the pair body's partial quad; was cg % 256.
  // A partial-quad Q6_K row under LLB_Q4_DP4A bit 32 stays llb_kgemv's two-plane dp4a GEMV + vadd,
  // as before: only rows the tile body ran move here)
  if (!(llb_q4glue & 2) || (fm != LLB_FM_Q4_K && fm != LLB_FM_Q6_K) || cg % 64 != 0 || d < 9 || d > 13
    || !llb_tuned(fm, cols) || (cg % 256 != 0 && fm == LLB_FM_Q6_K && (llb_q4dp & 32))) {
    return false;
  }
  if (!LLB_LAUNCH(fm == LLB_FM_Q6_K ? gpu_warp6e : gpu_warpke, grid == 768 ? 984 : grid, 1, 128,
    &w, &rb, &cg, &x, &y, &n, &d, &x1)) {
    err_fail("device launch failed");
  }
  return true;
}

// tw: the pass's table row (tools/payload-table.patch), each tensor's fm in
// bits 28..31; NULL (the legacy kinds) = every GEMV Q8_0
static void llb_fbn_run(volatile WarpOp* o, u64 yp, u32 grid, const volatile u32* tw) {
  // the FFN block bang (tools/payload-fbn.patch): rms(post, x + a), the
  // gate and up GEMVs, silu, the down GEMV, the residual add -- one task
  u64 postp17 = o->w, fgp17 = o->k, fup17 = o->v, fdp17 = o->wn;
  u64 xp17 = o->x, ap17 = o->r;
  u32 rg17 = o->rb, rd17 = o->d;
  u32 cgu = rg17 / 17 * 4, cgd = rd17 / 17 * 4;
  u32 dm = llb_sh.d, df = llb_sh.f;
  llb_dptr x1b = gpu_blk, h2b = x1b + dm * 4;
  llb_dptr ggb = h2b + dm * 4, uub = ggb + df * 4;
  llb_dptr aab = uub + df * 4, ddb = aab + df * 4;
  u32 n4 = dm, ngu = df, dgu = llb_cdepth(dm), dfd = llb_cdepth(df);
  u32 siluG = (df + 255) / 256, vaddG = (dm + 1023) / 1024;
  void* gg17[] = { &fgp17, &rg17, &cgu, &h2b, &ggb, &ngu, &dgu };
  void* uu17[] = { &fup17, &rg17, &cgu, &h2b, &uub, &ngu, &dgu };
  Q8Cat c17 = { { fgp17, fup17, 0, 0 }, { ggb, uub, 0, 0 }, { ngu, 2 * ngu, 2 * ngu, 2 * ngu } };
  (void)gg17; (void)uu17;
  (void)dfd; (void)ddb; (void)vaddG;
  u32 fg = tw ? tw[2] >> 28 : 1, fu = tw ? tw[3] >> 28 : 1, fd = tw ? tw[4] >> 28 : 1;
  // bit 64: the two-plane quantizers in rnadd's / silu's stores (gpu_q8c)
  bool fzg = (llb_q4dp & 68) == 68 && fg == LLB_FM_Q4_K && fu == LLB_FM_Q4_K && llb_q4dp_ok(dm, 1);
  bool fzd = (llb_q4dp & 72) == 72 && fd == LLB_FM_Q4_K && llb_q4dp_ok(df, 2);
  llb_dptr qxz = gpu_q8c, dzg = qxz + 2 * dm, dzd = qxz + 2 * df;
  if (!(fzg ? LLB_LAUNCH(gpu_rnaddq, 1, 1, 1024, &xp17, &ap17, &postp17, &x1b, &h2b, &n4, &qxz, &dzg)
    : LLB_LAUNCH(gpu_rnadd, 1, 1, 1024, &xp17, &ap17, &postp17, &x1b, &h2b, &n4))) {
    err_fail("device launch failed");
  }
  // warpqc is the d 10 cat, warpqc11 the d 11 one: other depths (d 8, the 0.8B,
  // tools/payload-w1024.patch; past 8192 cols) take the GEMVs below (tools/payload-anyshape.patch)
  if (fg == 1 && fu == 1 && (dgu == 10 || dgu == 11)) {
    if (!LLB_LAUNCH(dgu == 11 ? gpu_warpqc11 : gpu_warpqc, grid, 1, 128, &rg17, &cgu, &h2b, &c17)) {
      err_fail("device launch failed");
    }
  } else if (fg == LLB_FM_Q4_K && fu == LLB_FM_Q4_K && llb_q4dp_ok(dm, 1)) {
    // the int8 x in gpu_q8b: dm bytes, then dm / 32 d's
    // (two planes: gpu_q8c, 2 dm bytes, then the e's)
    bool two = llb_q4dp & 4;
    llb_dptr qxg = two ? gpu_q8c : gpu_q8b, dxg = qxg + (two ? 2 : 1) * dm;
    u32 rbk = LLB_FM_RB(fg, dm), q8G = dm / 256;
    if ((!fzg && !LLB_LAUNCH(two ? gpu_q8x2 : gpu_q8x, q8G, 1, 256, &h2b, &qxg, &dxg, &n4))
      || !LLB_LAUNCH(two ? gpu_q4dp2gu : gpu_q4dpgu, ngu, 1, 128, &fgp17, &fup17, &rbk, &qxg, &dxg, &ggb, &uub, &ngu)) {
      err_fail("device launch failed");
    }
  } else {
    llb_kgemv(fg, fgp17, dm, h2b, ggb, ngu, grid);
    llb_kgemv(fu, fup17, dm, h2b, uub, ngu, grid);
  }
  if (!(fzd ? LLB_LAUNCH(gpu_siluq, ngu / 256, 1, 256, &ggb, &uub, &aab, &ngu, &qxz, &dzd)
    : LLB_LAUNCH(gpu_silu, siluG, 1, 256, &ggb, &uub, &aab, &ngu))) {
    err_fail("device launch failed");
  }
  // llb_warp_q8_res is the d 12 body over 24 real vblocks: 8192 < f <= 12288 (the 9B's);
  // other Q8_0 down widths run llb_kgemv + vadd below (tools/payload-anyshape.patch)
  if (fd == 1 && dfd == 12 && df <= 12288) {
    if (!LLB_LAUNCH(gpu_warpqr, 82 * 8, 1, 192, &fdp17, &rd17, &cgd, &aab, &yp, &n4, &x1b)) {
      err_fail("device launch failed");
    }
  } else if (fd == LLB_FM_Q4_K && llb_q4dp_ok(df, 2)) {
    bool two = llb_q4dp & 8;
    llb_dptr qxd = two ? gpu_q8c : gpu_q8b, dxd = qxd + (two ? 2 : 1) * df;
    u32 rbk = LLB_FM_RB(fd, df), q8G = df / 256;
    if ((!fzd && !LLB_LAUNCH(two ? gpu_q8x2 : gpu_q8x, q8G, 1, 256, &aab, &qxd, &dxd, &ngu))
      || !LLB_LAUNCH(two ? gpu_q4dp2re : gpu_q4dpre, n4, 1, 128, &fdp17, &rbk, &qxd, &dxd, &yp, &n4, &x1b)) {
      err_fail("device launch failed");
    }
  } else if (!llb_kgemv_res(fd, fdp17, df, aab, yp, n4, grid, x1b)) {
    // down into ggb (dead after silu; ddb is aab's pad tail at d 12, which
    // the pad vote reads), then y = x1b + ggb as warpqr's residual
    llb_kgemv(fd, fdp17, df, aab, ggb, n4, grid);
    if (!LLB_LAUNCH(gpu_vadd, vaddG, 1, 1024, &x1b, &ggb, &yp, &n4)) {
      err_fail("device launch failed");
    }
  }
}

// tools/payload-moeany.patch: a routed-expert layer's FFN (the table row's words 21..30, pt[20..29]:
// gguf2pack.c moe_sidecar; pass_mail checked them). rms(post, x + a) as fbn; the router and the
// shared expert's gate (F32 GEMVs); llb_moe_topk (softmax over all experts, the top k, weights
// renormalized); the shared expert (gate / up / silu / down on llb_kgemv's bodies, sf rows, into
// shd); llb_moe_gu (the k experts' gate + up + silu); llb_moe_dn (their down, weighted, + the gated
// shared expert + the residual) into y. Expert ids and weights stay on the device except in the
// stream residency, which reads the k ids back to copy those experts in.
//
// The expert stacks are not in the layer's Bytes: llb_moe_load reads them from the GGUF by name
// (llb_gg_tensor, 64-bit offsets and sizes) into one of three residencies, picked from the device
// actually present and the file (LLB_MOE_EXPERTS=auto, the default, or device / host / stream):
//   device  every stack in VRAM (cuMemAlloc per stack) when they fit the free VRAM less a reserve;
//   host    every stack in pinned, device-mapped host RAM (the kernels read the k experts' rows
//           over the bus, nothing else) when they fit MemAvailable less a reserve;
//   stream  the GGUF mmapped; per layer and token the k chosen experts are copied into a device
//           staging area (k slots), so a model larger than RAM runs at disk / page-cache speed.
// LLB_MOE_VRAM / LLB_MOE_RAM (bytes) stand in for the free VRAM / MemAvailable the auto choice reads
// (tests force the host and stream paths on a small file with them).
int llb_gg_tensor(const char* name, uint64_t* src, uint64_t* bytes, uint32_t* ty, int* fd, const char** path);
static void llb_scratch(llb_dptr* p, size_t* have, size_t n, const char* what);
typedef struct {
  llb_dptr g, u, d;  // device-visible bases (device / host); stream: 0
  u64 sg, su, sd;    // file offsets (stream)
  u64 bg, bu, bd;    // bytes
} MoeL;
static MoeL moe_l[LLB_MAXL];
static int moe_mode = -1;  // 0 device, 1 host, 2 stream
static const unsigned char* moe_map;
static llb_dptr moe_stg, moe_stu, moe_std, moe_slot;
static u32* moe_hid;
// reports/cuda-moe-speed: the stream residency's copies. The k experts' bytes go from the mmapped
// GGUF into pinned host slots by MOE_NT threads (each a slice of every piece), and each slot is sent
// to the device staging area by an async copy on the null stream while the next is filled; the
// expert kernels queue behind the copies. The bytes are the same, so are the results
#define MOE_NT 8
static unsigned char* moe_pin;
static struct { unsigned char* d[3]; const unsigned char* s[3]; u64 n[3]; } moe_job;
static pthread_mutex_t moe_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t moe_go = PTHREAD_COND_INITIALIZER, moe_done = PTHREAD_COND_INITIALIZER;
static u32 moe_gen, moe_left;
static void* moe_worker(void* a) {
  u32 i = (u32)(uintptr_t)a, seen = 0;
  for (;;) {
    pthread_mutex_lock(&moe_mx);
    while (moe_gen == seen) {
      pthread_cond_wait(&moe_go, &moe_mx);
    }
    seen = moe_gen;
    pthread_mutex_unlock(&moe_mx);
    for (u32 k = 0; k < 3; k += 1) {
      u64 n = moe_job.n[k], a0 = n * i / MOE_NT, a1 = n * (i + 1) / MOE_NT;
      if (a1 > a0) {
        memcpy(moe_job.d[k] + a0, moe_job.s[k] + a0, (size_t)(a1 - a0));
      }
    }
    pthread_mutex_lock(&moe_mx);
    if (--moe_left == 0) {
      pthread_cond_signal(&moe_done);
    }
    pthread_mutex_unlock(&moe_mx);
  }
  return NULL;
}
// up to three pieces (n 0: none) copied by the pool; returns when all are in place
static void moe_pcopy(unsigned char* d0, const unsigned char* s0, u64 n0, unsigned char* d1, const unsigned char* s1,
  u64 n1, unsigned char* d2, const unsigned char* s2, u64 n2) {
  pthread_mutex_lock(&moe_mx);
  moe_job.d[0] = d0; moe_job.s[0] = s0; moe_job.n[0] = n0;
  moe_job.d[1] = d1; moe_job.s[1] = s1; moe_job.n[1] = n1;
  moe_job.d[2] = d2; moe_job.s[2] = s2; moe_job.n[2] = n2;
  moe_left = MOE_NT;
  moe_gen += 1;
  pthread_cond_broadcast(&moe_go);
  while (moe_left != 0) {
    pthread_cond_wait(&moe_done, &moe_mx);
  }
  pthread_mutex_unlock(&moe_mx);
}
static llb_dptr gpu_moe;  // lg[ne] sg pv[ne] sgo pad pad ids[k] wt[k] shd[d] act[k ef]
static size_t gpu_moe_have;
static bool llb_moe(u32 i) {
  return gpu_mail->pt[i][26] != 0;
}

static u32 moe_fm(u32 ty) {  // a ggml type id as the registry's format (llb_formats.h)
  return ty == 8 ? LLB_FM_Q8_0 : ty == 12 ? LLB_FM_Q4_K : ty == 13 ? LLB_FM_Q5_K : ty == 14 ? LLB_FM_Q6_K : 99;
}

static u64 moe_env(const char* k, u64 dflt) {
  const char* e = getenv(k);
  return e != NULL && *e ? strtoull(e, NULL, 10) : dflt;
}

static u64 moe_memavail(void) {
  FILE* f = fopen("/proc/meminfo", "r");
  char ln[256];
  u64 kb = 0;
  while (f != NULL && fgets(ln, sizeof ln, f) != NULL) {
    if (strncmp(ln, "MemAvailable:", 13) == 0) {
      kb = strtoull(ln + 13, NULL, 10);
    }
  }
  if (f != NULL) {
    fclose(f);
  }
  return kb * 1024;
}

// one stack: device (cuMemAlloc + pread in 64 MiB pieces) or host (pinned, mapped, pread in place)
static llb_dptr moe_fill(int fd, u64 src, u64 n, int mode, const char* name) {
  if (mode == 1) {
    void* h = NULL;
    CUdeviceptr dp = 0;
    if (cuMemHostAlloc(&h, n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS
      || cuMemHostGetDevicePointer(&dp, h, 0) != CUDA_SUCCESS) {
      fprintf(stderr, "moe: cannot pin %llu bytes of host RAM for %s\n", (unsigned long long)n, name);
      err_fail("moe: pinned host allocation failed (LLB_MOE_EXPERTS=stream needs no pinned RAM)");
    }
    for (u64 k = 0; k < n;) {
      ssize_t r = pread(fd, (unsigned char*)h + k, n - k > (1u << 30) ? (1u << 30) : (size_t)(n - k), (off_t)(src + k));
      if (r <= 0) {
        err_fail("moe: GGUF read failed");
      }
      k += (u64)r;
    }
    return dp;
  }
  llb_dptr d = 0;
  llb_graph_bail();
  if (cuMemAlloc(&d, n) != CUDA_SUCCESS) {
    fprintf(stderr, "moe: cannot allocate %llu bytes of VRAM for %s\n", (unsigned long long)n, name);
    err_fail("moe: out of device memory for the experts (LLB_MOE_EXPERTS=host or stream keeps them off the GPU)");
  }
  static unsigned char* buf;
  const u64 CH = 64u << 20;
  if (buf == NULL && (buf = malloc(CH)) == NULL) {
    err_fail("moe: out of host memory");
  }
  for (u64 k = 0; k < n; k += CH) {
    size_t c = n - k < CH ? (size_t)(n - k) : (size_t)CH;
    for (size_t q = 0; q < c;) {
      ssize_t r = pread(fd, buf + q, c - q, (off_t)(src + k + q));
      if (r <= 0) {
        err_fail("moe: GGUF read failed");
      }
      q += (size_t)r;
    }
    if (!llb_write(d + k, buf, c)) {
      err_fail("moe: cannot upload an expert stack");
    }
  }
  return d;
}

// the stacks of every MoE layer, once: names, sizes and formats against the table, then the
// residency, then the bytes
static void llb_moe_load(LlbShape sh) {
  int fd = -1;
  const char* path = NULL;
  u64 tot = 0, stg = 0, std_ = 0;
  u32 nkmax = 0, nmoe = 0;
  for (u32 i = 0; i < LLB_MAXL; i += 1) {
    const volatile u32* tw = gpu_mail->pt[i];
    if (tw[26] == 0) {
      continue;
    }
    u32 fx = tw[25], fz = fx >> 12 & 1, ne = tw[26], ef = tw[27], nk = tw[29];
    const char* nm[3] = { fz ? "ffn_gate_up_exps" : "ffn_gate_exps", fz ? NULL : "ffn_up_exps", "ffn_down_exps" };
    u64 want[3] = { (u64)ne * (fz ? 2 : 1) * ef * LLB_FM_RB(fx & 15, sh.d), fz ? 0 : (u64)ne * ef * LLB_FM_RB(fx >> 4 & 15, sh.d),
      (u64)ne * sh.d * LLB_FM_RB(fx >> 8 & 15, ef) };
    u64 src[3] = { 0, 0, 0 }, by[3] = { 0, 0, 0 };
    for (u32 j = 0; j < 3; j += 1) {
      if (nm[j] == NULL) {
        continue;
      }
      char name[96];
      uint32_t ty = 0;
      snprintf(name, sizeof name, "blk.%u.%s.weight", i, nm[j]);
      if (llb_gg_tensor(name, &src[j], &by[j], &ty, &fd, &path) != 0) {
        fprintf(stderr, "moe: %s not found\n", name);
        err_fail("moe: the routed experts load from the GGUF: run with LLB_GGUF=MODEL.gguf and its gguf2pack --sidecar-only dir");
      }
      if (moe_fm(ty) != (fx >> (4 * j) & 15) || by[j] != want[j]) {
        fprintf(stderr, "moe: %s: type %u, %llu bytes; the sidecar says format %u, %llu bytes\n", name, ty,
          (unsigned long long)by[j], fx >> (4 * j) & 15, (unsigned long long)want[j]);
        err_fail("moe: the sidecar is for another GGUF (rebuild it with gguf2pack --sidecar-only)");
      }
    }
    moe_l[i] = (MoeL){ 0, 0, 0, src[0], src[1], src[2], by[0], by[1], by[2] };
    tot += by[0] + by[1] + by[2];
    stg = stg > by[0] / ne ? stg : by[0] / ne;
    std_ = std_ > by[2] / ne ? std_ : by[2] / ne;
    nkmax = nkmax > nk ? nkmax : nk;
    nmoe += 1;
  }
  size_t fr = 0, tv = 0;
  cuMemGetInfo(&fr, &tv);
  u64 vram = moe_env("LLB_MOE_VRAM", fr), ram = moe_env("LLB_MOE_RAM", moe_memavail());
  u64 rv = (u64)1 << 30, rr = (u64)4 << 30;  // the reserves: KV growth and scratch / the rest of the system
  const char* e = getenv("LLB_MOE_EXPERTS");
  moe_mode = e != NULL && strcmp(e, "device") == 0 ? 0 : e != NULL && strcmp(e, "host") == 0 ? 1
    : e != NULL && strcmp(e, "stream") == 0 ? 2 : tot + rv <= vram ? 0 : tot + rr <= ram ? 1 : 2;
  static const char* mn[3] = { "device (VRAM)", "host (pinned, device-mapped RAM)", "stream (mmapped GGUF, the k experts copied per token)" };
  fprintf(stderr, "moe: %u layers, experts %.2f GiB -> %s [free VRAM %.2f GiB%s, MemAvailable %.2f GiB%s]\n", nmoe,
    tot / 1073741824.0, mn[moe_mode], vram / 1073741824.0, getenv("LLB_MOE_VRAM") ? " (LLB_MOE_VRAM)" : "",
    ram / 1073741824.0, getenv("LLB_MOE_RAM") ? " (LLB_MOE_RAM)" : "");
  if (moe_mode == 2) {
    struct stat st;
    if (fstat(fd, &st) != 0 || (moe_map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0)) == MAP_FAILED) {
      err_fail("moe: cannot mmap the GGUF");
    }
    moe_stg = llb_alloc_zero(stg * nkmax);
    moe_stu = llb_alloc_zero(stg * nkmax);
    moe_std = llb_alloc_zero(std_ * nkmax);
    moe_slot = llb_alloc_zero((size_t)nkmax * 4);
    moe_hid = malloc((size_t)nkmax * 4);
    if (cuMemHostAlloc((void**)&moe_pin, (size_t)(2 * stg + std_) * nkmax, 0) != CUDA_SUCCESS) {
      err_fail("moe: cannot pin the stream copy slots");
    }
    for (u32 t = 0; t < MOE_NT; t += 1) {
      pthread_t th;
      if (pthread_create(&th, NULL, moe_worker, (void*)(uintptr_t)t) != 0) {
        err_fail("moe: cannot start the stream copy threads");
      }
      pthread_detach(th);
    }
    u32* sl = malloc((size_t)nkmax * 4);
    for (u32 j = 0; sl != NULL && j < nkmax; j += 1) {
      sl[j] = j;
    }
    if (!moe_stg || !moe_stu || !moe_std || !moe_slot || !moe_hid || !sl || !llb_write(moe_slot, sl, (size_t)nkmax * 4)) {
      err_fail("moe: cannot allocate the stream staging area");
    }
    free(sl);
    return;
  }
  for (u32 i = 0; i < LLB_MAXL; i += 1) {
    MoeL* m = &moe_l[i];
    if (gpu_mail->pt[i][26] == 0) {
      continue;
    }
    m->g = moe_fill(fd, m->sg, m->bg, moe_mode, "a gate stack");
    m->u = m->bu ? moe_fill(fd, m->su, m->bu, moe_mode, "an up stack") : 0;
    m->d = moe_fill(fd, m->sd, m->bd, moe_mode, "a down stack");
  }
  (void)path;
}

// reports/cuda-catchup: the int8 expert kernels (host/llb_pf.cuh llb_mx_*), shared by the dp4a
// decode tier (LLB_MOE_DP4A=1, default off) and the fast MoE prefill (LLB_PREFILL=fast); loaded on
// first use
static CUfunction mx_fgu, mx_fdn, mx_fcomb, mx_froute, mx_ftopk, mx_fgroup, mx_fgather, mx_frg;
static CUfunction pfx_q4g, pfx_q4m, pfx_q5g, pfx_q5m, pfx_q6m;  // the grouped pfmm leaves (fast prefill)
static void mx_load(void) {
  if (mx_fgu != NULL) {
    return;
  }
  const char* nm[13] = { "llb_mx_gu", "llb_mx_dn", "llb_mx_comb", "llb_mx_route", "llb_mx_topk", "llb_mx_group",
    "llb_mx_gather", "llb_mx_rgemm", "pfx_q4k_g", "pfx_q4k_m", "pfx_q5k_g", "pfx_q5k_m", "pfx_q6k_m" };
  CUfunction* fp[13] = { &mx_fgu, &mx_fdn, &mx_fcomb, &mx_froute, &mx_ftopk, &mx_fgroup, &mx_fgather, &mx_frg,
    &pfx_q4g, &pfx_q4m, &pfx_q5g, &pfx_q5m, &pfx_q6m };
  for (u32 i = 0; i < 13; i += 1) {
    if (cuModuleGetFunction(fp[i], gpu_lib, nm[i]) != CUDA_SUCCESS
      || (i >= 8 && cuFuncSetAttribute(*fp[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 99 * 1024) != CUDA_SUCCESS)) {
      err_fail("moe: cannot load the int8 expert kernels");
    }
  }
}
static bool moe_dp4a(void) {
  static int v = -1;
  if (v < 0) {
    const char* e = getenv("LLB_MOE_DP4A");
    v = e != NULL && atoi(e) == 1;
    if (v) {
      fprintf(stderr, "moe: LLB_MOE_DP4A=1 -- routed experts on int8 activations (dp4a tier, not exact)\n");
    }
  }
  return v;
}
static llb_dptr mx_dq;
static size_t mx_dq_have;

static void llb_moe_run(volatile WarpOp* o, u64 yp, u32 grid, const volatile u32* tw, u32 li) {
  u64 postp = o->w, fgp = o->k, fup = o->v, fdp = o->wn, xp = o->x, ap = o->r, base = o->qk;  // qk: the layer base
  u32 dm = llb_sh.d, ne = tw[26], nk = tw[29], ef = tw[27], sf = tw[28], fx = tw[25];
  u32 fg = fx & 15, fu = fx >> 4 & 15, fd = fx >> 8 & 15, fz = fx >> 12 & 1;
  u32 sg = tw[2] >> 28, su = tw[3] >> 28, sd = tw[4] >> 28;
  u64 rp = base + (tw[20] & 0x0fffffffu), gp = base + (tw[21] & 0x0fffffffu);
  u32 rbg = LLB_FM_RB(fg, dm), rbu = LLB_FM_RB(fu, dm), rbd = LLB_FM_RB(fd, ef);
  if (moe_mode < 0) {
    llb_moe_load(llb_sh);
  }
  const MoeL* m = &moe_l[li];
  u64 esg = (u64)(fz ? 2 : 1) * ef * rbg, esd = (u64)dm * rbd;
  size_t need = ((size_t)2 * ne + 4 + 2 * (size_t)nk + dm + (size_t)nk * ef) * 4;
  llb_scratch(&gpu_moe, &gpu_moe_have, need, "gpu_moe");
  // reports/cuda-moe-speed: the router's logits and the shared gate's next to each other (lg, sg), so
  // the two F32 GEMVs are one launch where their rows are adjacent in the layer (gguf2pack's name
  // order puts ffn_gate_inp_shexp right after ffn_gate_inp); the shared gate and up likewise (ggb, uub)
  llb_dptr lg = gpu_moe, sgl = lg + (u64)ne * 4, pvp = sgl + 4, sgo = pvp + (u64)ne * 4, idp = sgo + 3 * 4;
  llb_dptr wtp = idp + (u64)nk * 4, shd = wtp + (u64)nk * 4, act = shd + (u64)dm * 4;
  llb_dptr x1b = gpu_blk, h2b = x1b + dm * 4, ggb = h2b + dm * 4, uub = ggb + (u64)sf * 4, aab = uub + (u64)sf * 4;
  u32 n4 = dm, nsf = sf, cgd = dm / 4, cge = ef / 4, tkG = 1 + (sf + 1023) / 1024;
  // a GEMV's rows are independent of the grid (it moves rows between warps only): the F32 ones take
  // a warp a row, not the default grid's idle warps
  u32 gr1 = (ne + 4) / 4 < grid ? (ne + 4) / 4 : grid;
  if (!LLB_LAUNCH(gpu_rnadd, 1, 1, 1024, &xp, &ap, &postp, &x1b, &h2b, &n4)) {
    err_fail("device launch failed");
  }
  if (gp == rp + (u64)ne * dm * 4) {
    llb_kgemv(LLB_FM_F32, rp, dm, h2b, lg, ne + 1, gr1);
  } else {
    llb_kgemv(LLB_FM_F32, rp, dm, h2b, lg, ne, gr1);
    llb_kgemv(LLB_FM_F32, gp, dm, h2b, sgl, 1, 1);
  }
  if (su == sg && fup == fgp + (u64)sf * LLB_FM_RB(sg, dm)) {
    llb_kgemv(sg, fgp, dm, h2b, ggb, 2 * sf, grid);
  } else {
    llb_kgemv(sg, fgp, dm, h2b, ggb, sf, grid);
    llb_kgemv(su, fup, dm, h2b, uub, sf, grid);
  }
  // the top k, and the shared expert's silu in the same launch (blocks 1..)
  if (!LLB_LAUNCH(gpu_moetk, tkG, 1, 1024, &lg, &ne, &nk, &sgl, &pvp, &idp, &wtp, &sgo, &ggb, &uub, &aab, &nsf)) {
    err_fail("device launch failed");
  }
  // the shared expert's down inside llb_moe_dn where llb_kgemv would take llb_gemv_any for it (the
  // same leaf and order: no tuned body for sf columns, a K-quant, not LLB_Q4_DP4A's Q6_K planes);
  // else its own GEMV into shd
  bool dq = moe_dp4a();
  bool sfold = !dq && !llb_tuned(sd, sf) && (sd == LLB_FM_Q4_K || sd == LLB_FM_Q5_K || (sd == LLB_FM_Q6_K && !(llb_q4dp & 32)));
  u64 SDp = sfold ? fdp : 0;
  u32 rbs = LLB_FM_RB(sd, sf), cgs = sf / 4;
  if (!sfold) {
    llb_kgemv(sd, fdp, sf, aab, shd, dm, grid);
  }
  u64 Gp = m->g, Up = fz ? m->g + (u64)ef * rbg : m->u, Dp = m->d;
  llb_dptr ids = idp;
  if (moe_mode == 2) {
    // the k chosen experts into the staging slots (gate (+ up), down), then slot j is expert j
    if (!llb_read(moe_hid, idp, (size_t)nk * 4)) {
      err_fail("moe: cannot read the chosen experts back");
    }
    u64 eu = fz ? 0 : esg, sl = esg + eu + esd;
    for (u32 j = 0; j < nk; j += 1) {
      u64 e = moe_hid[j];
      unsigned char* hp = moe_pin + j * sl;
      if (e >= ne) {
        err_fail("moe: cannot stage an expert");
      }
      moe_pcopy(hp, moe_map + m->sg + e * esg, esg, hp + esg, fz ? NULL : moe_map + m->su + e * esg, eu,
        hp + esg + eu, moe_map + m->sd + e * esd, esd);
      if (cuMemcpyHtoDAsync(moe_stg + j * esg, hp, esg, NULL) != CUDA_SUCCESS
        || (!fz && cuMemcpyHtoDAsync(moe_stu + j * esg, hp + esg, esg, NULL) != CUDA_SUCCESS)
        || cuMemcpyHtoDAsync(moe_std + j * esd, hp + esg + eu, esd, NULL) != CUDA_SUCCESS) {
        err_fail("moe: cannot stage an expert");
      }
    }
    Gp = moe_stg, Up = fz ? moe_stg + (u64)ef * rbg : moe_stu, Dp = moe_std, ids = moe_slot;
  }
  if (dq) {
    // the dp4a tier: h and the k experts' act rows to q8_1, the int8 gate / up / down, then the
    // weighted rank-order sum + the gated shared expert + the residual (llb_mx_comb)
    mx_load();
    size_t qn = (size_t)dm + dm / 32 * 8 + (size_t)nk * ef + (size_t)nk * ef / 32 * 8 + (size_t)nk * dm * 4 + 256;
    llb_scratch(&mx_dq, &mx_dq_have, qn, "moe dp4a");
    llb_dptr hq = mx_dq, hd = hq + dm, aq = hd + dm / 32 * 8, ad = aq + (u64)nk * ef, out = ad + (u64)nk * ef / 32 * 8;
    out = (out + 15) & ~(llb_dptr)15;
    u64 nul = 0;
    u32 one = 1, gq = dm / 256, ga = nk * ef / 256, gu = (ef + 3) / 4, gd = ef / 32 <= 16 ? (dm + 7) / 8 : (dm + 3) / 4;
    u32 gc = (dm + 255) / 256;
    if (!LLB_LAUNCH(gpu_pfq8x, gq, 1, 256, &h2b, &hq, &hd)
      || !LLB_LAUNCH(mx_fgu, gu, nk, 128, &Gp, &Up, &esg, &rbg, &rbu, &fg, &fu, &dm, &hq, &hd, &ids, &nul, &nul, &act, &ef)
      || !LLB_LAUNCH(gpu_pfq8x, ga, 1, 256, &act, &aq, &ad)
      || !LLB_LAUNCH(mx_fdn, gd, nk, 128, &Dp, &esd, &rbd, &fd, &ef, &aq, &ad, &ids, &nul, &out, &dm)
      || !LLB_LAUNCH(mx_fcomb, gc, one, 256, &out, &wtp, &nul, &nk, &shd, &sgo, &x1b, &yp, &dm)) {
      err_fail("device launch failed");
    }
    return;
  }
  if (!LLB_LAUNCH(gpu_moegu, (ef + 3) / 4, nk, 128, &Gp, &Up, &esg, &rbg, &rbu, &fg, &fu, &cgd, &h2b, &ids, &act, &ef)
    || !LLB_LAUNCH(gpu_moedn, (dm + 3) / 4, 1, 128, &Dp, &esd, &rbd, &fd, &cge, &act, &ids, &wtp, &nk, &ef,
      &x1b, &shd, &sgo, &yp, &n4, &SDp, &rbs, &sd, &cgs, &aab)) {
    err_fail("device launch failed");
  }
}

static void llb_linb_run(volatile WarpOp* o, u64 yp, u32 grid, const volatile u32* tw) {
  // the linear attention block bang (tools/payload-linb.patch): the
  // z/beta/alpha GEMVs, the 32-element middleware, conv1d, the two
  // l2norms, gated_delta (S in place), gated_rmsnorm, the ssm_out GEMV
  u64 base18 = o->w;
  u64 xp18 = o->x, qkvp18 = o->qk, h0p = o->k, h1p = o->v, h2p = o->wn, Sp = o->r;
  u64 anp18 = base18 + o->ban, wqkvp18 = base18 + o->c;
  u64 gtp = base18 + o->bq, bop = base18 + o->bg, aop = base18 + o->bb;
  u64 sap = base18 + o->ba, dtp = base18 + o->bsa, cvp = base18 + o->bdt;
  u64 snp = base18 + o->bcv, sop = base18 + o->bsn;
  // ssm_out's Q8_0 row bytes from its own width lv (o->bro is attn_output's, nh * hd columns:
  // the same on every shape with nh hd = lvh sd; tools/payload-anyshape.patch)
  u32 rg18 = o->brg, cg18 = rg18 / 17 * 4, ro18 = llb_sh.lvh * llb_sh.sd / 32 * 34, co18 = ro18 / 17 * 4;
  // conv's [q lk][k lk][v lv] channels; qt / kt hold lk and its copies, lv in
  // all (lvh = 2 lkh the 9B, 3 the 27B); hb18 past them (39040 the 9B's)
  u32 lk = llb_sh.lkh * llb_sh.sd, lv = llb_sh.lvh * llb_sh.sd, cv = 2 * lk + lv;
  u32 rep = llb_sh.lvh / llb_sh.lkh;
  llb_dptr zz = gpu_blk, bpre = zz + lv * 4, apre = bpre + llb_sh.lvh * 4;
  llb_dptr beta = apre + llb_sh.lvh * 4, dec = beta + llb_sh.lvh * 4;
  llb_dptr cnv = dec + llb_sh.lvh * 4, cnvk = cnv + lk * 4, vn = cnv + 2 * lk * 4;
  llb_dptr gout = cnv + cv * 4, qt = gout + lv * 4;
  llb_dptr qt2 = qt + lk * 4, kt = qt + lv * 4;
  llb_dptr kt2 = kt + lk * 4, normed = kt + lv * 4;
  u32 end18 = 4 * lv + 4 * llb_sh.lvh + cv;
  llb_dptr hb18 = gpu_blk + (end18 > 39040 ? end18 : 39040) * 4;
  // sq = sqrt(sd) for sd 128 (the envelope)
  union { u32 u; float f; } sq = { 0x413504f3u }, one = { 0x3f800000u };
  u32 n4 = llb_sh.d, n8192 = cv, n32 = llb_sh.lvh, d10 = llb_cdepth(llb_sh.d);
  u32 hs16 = llb_sh.lkh, hs32 = llb_sh.lvh, gdy = llb_sh.sd / 4;
  u32 nz = lv, dso = llb_cdepth(lv);
  u32 gconv = (cv + 255) / 256;
  void* q18[] = { &wqkvp18, &rg18, &cg18, &hb18, &qkvp18, &n8192, &d10 };
  void* z18[] = { &gtp, &rg18, &cg18, &hb18, &zz, &nz, &d10 };
  void* ba18[] = { &bop, &aop, &rg18, &cg18, &hb18, &bpre, &apre, &n32 };
  Q8Cat qc18 = { { wqkvp18, gtp, bop, aop }, { qkvp18, zz, bpre, apre },
    { n8192, n8192 + nz, n8192 + nz + n32, n8192 + nz + 2 * n32 } };
  (void)q18; (void)z18; (void)ba18;
  (void)qt2; (void)kt2; (void)one;
  u32 fq = tw ? tw[5] >> 28 : 1, fz = tw ? tw[6] >> 28 : 1;
  u32 fb = tw ? tw[7] >> 28 : 1, fa = tw ? tw[8] >> 28 : 1, fo = tw ? tw[13] >> 28 : 1;
  // the Q8_0 cats are d 10 / 11: d 8 (the 0.8B) runs each GEMV (tools/payload-w1024.patch), as
  // does any depth past 11 (tools/payload-anyshape.patch)
  bool q8 = fq == 1 && fz == 1 && fb == 1 && fa == 1 && (d10 == 10 || d10 == 11);
  // mixed: per tensor, beta + alpha one 2-member cat when both Q8_0
  Q8Cat ba = { { bop, aop, 0, 0 }, { bpre, apre, 0, 0 }, { n32, 2 * n32, 2 * n32, 2 * n32 } };
  bool dp18 = q8 && (llb_q4dp & 16) && n4 % 256 == 0;
  llb_dptr qx18 = gpu_q8c, dx18 = qx18 + 2 * n4;
  u32 q8G18 = n4 / 256, nc18 = qc18.end[3];
  bool fz18 = dp18 && (llb_q4dp & 64);
  if (!(fz18 ? LLB_LAUNCH(gpu_rnormq, 1, 1, 1024, &xp18, &anp18, &hb18, &n4, &qx18, &dx18)
      : LLB_LAUNCH(gpu_rnorm, 1, 1, 1024, &xp18, &anp18, &hb18, &n4))
    || (dp18 && ((!fz18 && !LLB_LAUNCH(gpu_q8x2, q8G18, 1, 256, &hb18, &qx18, &dx18, &n4))
      || !LLB_LAUNCH(gpu_q8dp2c, nc18, 1, 128, &rg18, &qx18, &dx18, &qc18)))
    || (q8 && !dp18 && !LLB_LAUNCH(d10 == 11 ? gpu_warpqc11 : gpu_warpqc, grid, 1, 128, &rg18, &cg18, &hb18, &qc18))) {
    err_fail("device launch failed");
  }
  if (!q8) {
    llb_kgemv(fq, wqkvp18, n4, hb18, qkvp18, n8192, grid);
    // Q4_K z with beta + alpha (Q8_0, d 10) in its launch: 2 n32 blocks a row each first
    u32 rz = LLB_FM_RB(fz, n4), cz = n4 / 4, nba = 2 * n32 + (grid == 768 ? 984 : grid);
    if ((llb_q4glue & 1) && fz == LLB_FM_Q4_K && fb == 1 && fa == 1 && d10 == 10 && cz % 256 == 0) {
      if (!LLB_LAUNCH(gpu_warpkba, nba, 1, 128, &gtp, &rz, &cz, &hb18, &zz, &nz, &d10, &bop, &aop,
        &rg18, &cg18, &bpre, &apre, &n32)) {
        err_fail("device launch failed");
      }
    } else {
      llb_kgemv(fz, gtp, n4, hb18, zz, nz, grid);
      // warpqc's cat is d 10 only; the 0.8B's d 8 (tools/payload-w1024.patch) runs each GEMV
      if (fb == 1 && fa == 1 && (d10 == 10 || d10 == 11)) {
        if (!LLB_LAUNCH(d10 == 11 ? gpu_warpqc11 : gpu_warpqc, grid, 1, 128, &rg18, &cg18, &hb18, &ba)) {
          err_fail("device launch failed");
        }
      } else {
        llb_kgemv(fb, bop, n4, hb18, bpre, n32, grid);
        llb_kgemv(fa, aop, n4, hb18, apre, n32, grid);
      }
    }
  }
  llb_dptr dxo = qx18 + 2 * lv;
  u32 q8Go = lv / 256;
  if (!LLB_LAUNCH(gpu_midconv, 1 + gconv, 1, 256, &bpre, &apre, &dtp, &sap, &beta, &dec, &n32,
    &cvp, &qkvp18, &h0p, &h1p, &h2p, &cnv, &n8192)
    || !LLB_LAUNCH(gpu_l2qk, 2 * hs16, 1, 64, &cnv, &cnvk, &qt, &kt, &sq.f, &hs16, &rep)
    || !LLB_LAUNCH(gpu_gdl, hs32, gdy, 128, &Sp, &qt, &kt, &vn, &dec, &beta, &gout, &hs32)
    || !LLB_LAUNCH(gpu_grn, hs32, 1, 64, &gout, &snp, &zz, &normed, &hs32)
    || (fo == 1 && (llb_q4dp & 16) && lv % 256 == 0
      ? !LLB_LAUNCH(gpu_q8x2, q8Go, 1, 256, &normed, &qx18, &dxo, &nz)
        || !LLB_LAUNCH(gpu_q8dp2, n4, 1, 128, &sop, &ro18, &qx18, &dxo, &yp, &n4)
      : fo == 1 && llb_tuned(fo, lv) && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &sop, &ro18, &co18, &normed, &yp, &n4, &dso))) {
    err_fail("device launch failed");
  }
  if (fo != 1 || (!((llb_q4dp & 16) && lv % 256 == 0) && !llb_tuned(fo, lv))) {
    llb_kgemv(fo, sop, lv, normed, yp, n4, grid);
  }
}

// llb_attn4 as the ATTN4S split (host/llb_pf.cuh llb_a4s_*): the same bits over NC x nkv
// blocks instead of nh. LLB_ATTN4_SPLIT=0 keeps the one-block-a-head kernel. nh q heads, grp
// a kv head; the args as llb_attn4's.
static CUfunction gpu_a4s_score, gpu_a4s_exp, gpu_a4s_wv, gpu_a4s_out;
static bool llb_attn4_run(u32 nh, u64 q, u64 kc, u64 vc, u64 att, u64 s, u64 p, u32 c, u32 grp) {
  static int on = -1;
  if (on < 0) {
    const char* e = getenv("LLB_ATTN4_SPLIT");
    on = !(e != NULL && e[0] == '0')
      && cuModuleGetFunction(&gpu_a4s_score, gpu_lib, "llb_a4s_score") == CUDA_SUCCESS
      && cuModuleGetFunction(&gpu_a4s_exp, gpu_lib, "llb_a4s_exp") == CUDA_SUCCESS
      && cuModuleGetFunction(&gpu_a4s_wv, gpu_lib, "llb_a4s_wv") == CUDA_SUCCESS
      && cuModuleGetFunction(&gpu_a4s_out, gpu_lib, "llb_a4s_out") == CUDA_SUCCESS;
  }
  if (!on || grp == 0 || grp > 8 || nh % grp != 0 || (kc & 15) != 0) {
    return LLB_LAUNCH(gpu_attn4, nh, 1, 1024, &q, &kc, &vc, &att, &s, &p, &c, &grp);
  }
  const u32 dT = c & 31, pos = c >> 5, T = 1u << dT, n = pos < T ? pos + 1 : T;
  u32 T2 = 1;
  while (T2 < n) {
    T2 <<= 1;
  }
  const u32 C = T2 <= 256 ? T2 : (T2 >> 8 > 256 ? T2 >> 8 : 256), NC = T2 / C, nkv = nh / grp;
  return LLB_LAUNCH(gpu_a4s_score, NC, nkv, 256, &q, &kc, &s, &p, &c, &grp)
    && LLB_LAUNCH(gpu_a4s_exp, NC, nkv, 256, &s, &p, &c, &grp)
    && LLB_LAUNCH(gpu_a4s_wv, NC, nkv, 1024, &vc, &att, &s, &p, &c, &grp)
    && (NC == 1 || LLB_LAUNCH(gpu_a4s_out, nh, 1, 256, &att, &p, &c));
}

static void llb_fulb_run(volatile WarpOp* o, u64 yp, u32 grid, const volatile u32* tw) {
  u64 base22 = o->w, xp22 = o->x, kcp22 = o->k, vcp22 = o->v;
  u64 anp22 = base22 + o->ban, wqp22 = base22 + o->bq, wkp22 = base22 + o->bg;
  u64 wvp22 = base22 + o->bb, wqnp22 = base22 + o->ba, wknp22 = base22 + o->bsa;
  u64 wop22 = base22 + o->bdt;
  u32 c22 = o->c, pos22 = c22 >> 5, rg22 = o->brg, cg22 = rg22 / 17 * 4;
  u32 ro22 = o->bro, co22 = ro22 / 17 * 4;
  // q heads then gates (qa each), k and v (kv each)
  u32 qa = llb_sh.nh * llb_sh.hd, kv = llb_sh.nkv * llb_sh.hd;
  u64 kslot22 = kcp22 + (u64)pos22 * kv * 4, vslot22 = vcp22 + (u64)pos22 * kv * 4;
  // the score planes: 2^max(dT, 17) floats a head (llb_attn4; tools/payload-anyshape.patch)
  u32 pl22 = (c22 & 31) > 17 ? (c22 & 31) : 17;
  u64 s22 = gpu_atts, p22 = gpu_atts + ((u64)llb_sh.nh << pl22) * 4;
  llb_dptr attn22 = gpu_blk, gated22 = attn22 + qa * 4;
  llb_dptr h22 = gpu_blk + 2 * qa * 4, qgn22 = h22 + llb_sh.d * 4;
  llb_dptr qg22 = qgn22 + 2 * qa * 4, gate22 = qg22 + qa * 4;
  llb_dptr kb22 = qg22 + 2 * qa * 4, kn22 = kb22 + kv * 4, vb22 = kn22 + kv * 4;
  u32 n4 = llb_sh.d, n8192 = 2 * qa, n1024 = kv, d10 = llb_cdepth(llb_sh.d), one = 1, zero = 0;
  u32 nqa = qa, dqa = llb_cdepth(qa), ah22 = llb_sh.nh;
  u32 rq22 = llb_sh.nh, rk22 = llb_sh.nkv, gog22 = (qa + 1023) / 1024;
  u32 grp22 = llb_sh.nh / llb_sh.nkv;
  void* q22[] = { &wqp22, &rg22, &cg22, &h22, &qgn22, &n8192, &d10 };
  void* k22[] = { &wkp22, &rg22, &cg22, &h22, &kb22, &n1024, &d10 };
  void* v22[] = { &wvp22, &rg22, &cg22, &h22, &vb22, &n1024, &d10 };
  Q8Cat qc22 = { { wqp22, wkp22, wvp22, 0 }, { qgn22, kb22, vb22, 0 },
    { n8192, n8192 + n1024, n8192 + 2 * n1024, n8192 + 2 * n1024 } };
  (void)q22; (void)k22; (void)v22;
  u32 fq = tw ? tw[14] >> 28 : 1, fk = tw ? tw[15] >> 28 : 1;
  u32 fv = tw ? tw[16] >> 28 : 1, fo = tw ? tw[19] >> 28 : 1;
  bool q8 = fq == 1 && fk == 1 && fv == 1 && (d10 == 10 || d10 == 11);
  bool dp22 = q8 && (llb_q4dp & 16) && n4 % 256 == 0;
  llb_dptr qx22 = gpu_q8c, dx22 = qx22 + 2 * n4;
  u32 q8G22 = n4 / 256, nc22 = qc22.end[3];
  bool fz22 = dp22 && (llb_q4dp & 64);
  if (!(fz22 ? LLB_LAUNCH(gpu_rnormq, 1, 1, 1024, &xp22, &anp22, &h22, &n4, &qx22, &dx22)
      : LLB_LAUNCH(gpu_rnorm, 1, 1, 1024, &xp22, &anp22, &h22, &n4))
    || (dp22 && ((!fz22 && !LLB_LAUNCH(gpu_q8x2, q8G22, 1, 256, &h22, &qx22, &dx22, &n4))
      || !LLB_LAUNCH(gpu_q8dp2c, nc22, 1, 128, &rg22, &qx22, &dx22, &qc22)))
    || (q8 && !dp22 && !LLB_LAUNCH(d10 == 11 ? gpu_warpqc11 : gpu_warpqc, grid, 1, 128, &rg22, &cg22, &h22, &qc22))) {
    err_fail("device launch failed");
  }
  if (!q8) {
    llb_kgemv(fq, wqp22, n4, h22, qgn22, n8192, grid);
    llb_kgemv(fk, wkp22, n4, h22, kb22, n1024, grid);
    llb_kgemv(fv, wvp22, n4, h22, vb22, n1024, grid);
  }
  if (!LLB_LAUNCH(gpu_qkr, rq22, 1, 256, &qgn22, &wqnp22, &qg22, &n8192, &one, &pos22)
    || !LLB_LAUNCH(gpu_qkr, rk22, 1, 256, &kb22, &wknp22, &kn22, &n1024, &zero, &pos22)
    // a slot is kv floats: 1024 every 4-kv-head model, 512 the 0.8B (tools/payload-w1024.patch);
    // past 1024, blocks of 256 (tools/payload-anyshape.patch)
    || !LLB_LAUNCH(gpu_slot, kv > 1024 ? kv / 256 : 1, 1, kv > 1024 ? 256 : kv, &kn22, &kslot22)
    || !LLB_LAUNCH(gpu_slot, kv > 1024 ? kv / 256 : 1, 1, kv > 1024 ? 256 : kv, &vb22, &vslot22)
    // LLB_ATTN_SK8=1: split-key decode attention (tools/payload-sk8add.patch), where it fits
    || !(llb_sk8_enabled() && llb_sk8_fits(c22, ah22) ? llb_sk8_launch(qg22, kcp22, vcp22, attn22, c22, grp22, ah22) :
      llb_attn4_run(ah22, qg22, kcp22, vcp22, attn22, s22, p22, c22, grp22))
    || !LLB_LAUNCH(gpu_og, gog22, 1, 1024, &attn22, &gate22, &gated22, &nqa)
    || (fo == 1 && llb_tuned(fo, qa) && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &wop22, &ro22, &co22, &gated22, &yp, &n4, &dqa))) {
    err_fail("device launch failed");
  }
  if (fo != 1 || !llb_tuned(fo, qa)) {
    llb_kgemv(fo, wop22, qa, gated22, yp, n4, grid);
  }
}

// The whole-pass bang (tools/payload-pass.patch): layers s..s+k-1 of one
// step, each the linb (SSM) or fulb (full, (i + 1) % 4 == 0) chain into a,
// then the fbn chain (x, a) into the next x; the last into y. The layer state
// lives here, not in the interpreter: per SSM layer a ring of four 8192 conv
// buffers (h0, h1, h2, qkva = b[r .. r + 3 mod 4], r += 1 a step: lin.go's
// (h1, h2, qkva) rotation; qkva's GEMV writes all 8192 rows before conv reads
// it) and S (2^19); per full layer kc, vc (2^(dT + 10)). Zeroed at first use
// as Array.new(F32, _, 0.0) was; each layer must then see pos, pos + 1, ...
// a and the x ping-pong are fully written (4096 rows) before each read.
// tools/payload-anyshape.patch: the caches are allocated by position (pass_kv), up to the
// pass's KV capacity (pass_cap, the config word: prompt + steps), not 2^dT up front; the layer
// arrays hold LLB_MAXL layers (comp.ts, was 64).
static llb_dptr pass_b[LLB_MAXL], pass_S[LLB_MAXL], pass_kc[LLB_MAXL], pass_vc[LLB_MAXL], pass_x[3];
static u32  pass_r[LLB_MAXL], pass_pos[LLB_MAXL], pass_dT;
static size_t pass_kcap[LLB_MAXL];  // slots allocated in pass_kc[i] / pass_vc[i]
static u32  pass_cap;               // the KV capacity in positions (Mail cfg[3])
static bool pass_on[LLB_MAXL];

static llb_dptr pass_zero(size_t n) {
  llb_dptr p = llb_alloc_zero(n * 4);
  if (!p) {
    err_fail("warp pass: cannot allocate the layer state");
  }
  return p;
}

// W1-KVPFX: cross-request prefix reuse (env LLB_PREFIX_CACHE=1; unset = off, nothing below runs).
// Qwen3.5 is hybrid. A full layer's KV slots are by position: slots < K stay valid while the
// ids before K are unchanged, and slots >= pos are masked. A GatedDeltaNet layer's state (the
// conv ring + S) folds the whole history, so reuse at K needs that state as it was at K: the
// host snapshots every GDN layer's (ring, S, r) when it reaches a chunk boundary K = j * C
// inside the known ids, keyed by a hash chain over ids[0, K). The request layer
// (tools/sys1dual/dual.c) calls llb_pfx_begin with a request's full ids before the engine runs
// it: the longest valid boundary whose chain AND ids match is restored and only ids[K..] run
// at --pos K. Stale guard: every snapshot past the request's start is dropped there (its ids
// may change from that position on), and a hit also compares the ids themselves.
// LLB_PREFIX_CHUNK (C, default 128), LLB_PREFIX_SLOTS (default 8, <= 64; the lowest boundary is
// evicted first). Off under LLB_SPEC / LLB_MTP (they keep their own copies of the state).
typedef struct { u32 pos; u64 h; bool done[LLB_MAXL]; u32 r[LLB_MAXL]; llb_dptr b[LLB_MAXL], S[LLB_MAXL]; } PfxSnap;
static int pfx_on = -1;  // -1 unread, 0 off, 1 on
static u32 pfx_C = 128, pfx_ns = 8, pfx_n, pfx_cap, pfx_K;
static int pfx_arm = -1;
static u32* pfx_ids;
static unsigned char pfx_kind[LLB_MAXL];  // 1 GDN, 2 full (as pass_layer last saw layer i)
static PfxSnap pfx_s[64];

int llb_pfx_on(void) {
  if (pfx_on < 0) {
    const char* e = getenv("LLB_PREFIX_CACHE");
    const char* c = getenv("LLB_PREFIX_CHUNK");
    const char* s = getenv("LLB_PREFIX_SLOTS");
    pfx_on = e != NULL && strcmp(e, "1") == 0 && !getenv("LLB_SPEC") && !getenv("LLB_MTP");
    if (c != NULL && atoi(c) >= 16) {
      pfx_C = (u32)atoi(c);
    }
    if (s != NULL && atoi(s) >= 1 && atoi(s) <= 64) {
      pfx_ns = (u32)atoi(s);
    }
    if (pfx_on) {
      fprintf(stderr, "prefix_cache on chunk %u slots %u\n", pfx_C, pfx_ns);
    }
  }
  return pfx_on;
}

static u64 pfx_hash(const u32* ids, u32 k) {
  u64 h = 0xcbf29ce484222325ull;  // FNV-1a over the ids, one link per chunk of C
  for (u32 t = 0; t < k; t += 1) {
    h = (h ^ ids[t]) * 0x100000001b3ull;
    if ((t + 1) % pfx_C == 0) {
      h = (h ^ (t + 1)) * 0x100000001b3ull;
    }
  }
  return h;
}

static bool pfx_valid(const PfxSnap* s) {
  if (s->pos == 0) {
    return false;
  }
  for (u32 i = 0; i < LLB_MAXL; i += 1) {
    if (pfx_kind[i] == 1 && !s->done[i]) {
      return false;
    }
  }
  return true;
}

// before the engine runs ids[0, n) from start: returns the start it should use (the cached count)
unsigned llb_pfx_begin(const int* ids, unsigned n, unsigned start, int reuse) {
  if (llb_pfx_on() != 1) {
    return start;
  }
  if (n > pfx_cap) {
    pfx_cap = n < 4096 ? 4096 : n;
    pfx_ids = realloc(pfx_ids, (size_t)pfx_cap * 4);
    if (!pfx_ids) {
      err_fail("prefix cache: cannot allocate the resident ids");
    }
  }
  pfx_arm = -1;
  if (reuse && start == 0) {
    static u32 q[65536];
    u32 m = n < 65536 ? n : 65536;
    for (u32 t = 0; t < m; t += 1) {
      q[t] = (u32)ids[t];
    }
    for (u32 j = 0; j < pfx_ns; j += 1) {
      PfxSnap* s = &pfx_s[j];
      if (pfx_valid(s) && s->pos + 1 <= m && s->pos <= pfx_n && s->pos > start
          && s->h == pfx_hash(q, s->pos) && memcmp(q, pfx_ids, (size_t)s->pos * 4) == 0) {
        start = s->pos;
        pfx_arm = (int)j;
      }
    }
    pfx_K = start;
  }
  for (u32 j = 0; j < pfx_ns; j += 1) {
    if (pfx_s[j].pos > start || !pfx_valid(&pfx_s[j])) {
      pfx_s[j].pos = 0;
    }
  }
  for (u32 t = 0; t < n; t += 1) {
    pfx_ids[t] = (u32)ids[t];
  }
  pfx_n = n;
  return start;
}

// pass_layer, before the guard: the armed snapshot back into layer i at its start pos K
static void pfx_restore(u32 i, bool full, u32 pos, LlbShape sh) {
  if (pfx_arm < 0 || pos != pfx_K) {
    return;
  }
  PfxSnap* s = &pfx_s[pfx_arm];
  if (!pass_on[i] || s->pos != pos) {
    err_fail("prefix cache: the armed snapshot does not match the layer");
  }
  if (!full) {
    u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
    if (!s->done[i] || !llb_copy(pass_b[i], s->b[i], 4 * (size_t)cv * 4)
        || !llb_copy(pass_S[i], s->S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4)) {
      err_fail("prefix cache: cannot restore a layer's state");
    }
    pass_r[i] = s->r[i];
  }
  pass_pos[i] = pos - 1;
}

// pass_layer, after the guard: layer i's state over ids[0, pos) at a boundary inside the known ids
static void pfx_take(u32 i, bool full, u32 pos, LlbShape sh) {
  pfx_kind[i] = full ? 2 : 1;
  if (pfx_on != 1 || full || pos == 0 || pos % pfx_C != 0 || pos > pfx_n) {
    return;
  }
  PfxSnap* s = NULL;
  for (u32 j = 0; j < pfx_ns && !s; j += 1) {
    s = pfx_s[j].pos == pos ? &pfx_s[j] : NULL;
  }
  if (!s) {
    for (u32 j = 0; j < pfx_ns; j += 1) {
      if (!s || pfx_s[j].pos < s->pos) {
        s = &pfx_s[j];  // a free slot (pos 0), else the lowest boundary
      }
    }
    s->pos = pos;
    s->h = pfx_hash(pfx_ids, pos);
    memset(s->done, 0, sizeof s->done);
  }
  if (s->done[i]) {
    return;
  }
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  size_t nb = 4 * (size_t)cv * 4, ns = (size_t)sh.lvh * sh.sd * sh.sd * 4;
  if (!s->b[i]) {
    s->b[i] = llb_alloc_zero(nb);
    s->S[i] = llb_alloc_zero(ns);
    if (!s->b[i] || !s->S[i]) {
      err_fail("prefix cache: cannot allocate a snapshot");
    }
  }
  if (!llb_copy(s->b[i], pass_b[i], nb) || !llb_copy(s->S[i], pass_S[i], ns)) {
    err_fail("prefix cache: cannot snapshot a layer's state");
  }
  s->r[i] = pass_r[i];
  s->done[i] = true;
}

// G20 (S1 banks; env LLB_BANKS=B, 2..16; unset or 1 = off, nothing below runs): B independent
// sequence states behind ONE copy of the weights. A bank is everything the host keeps for one
// sequence: every layer's KV slots (full) or conv ring + S + ring index (GDN), the per-layer pos
// guard, and the prefix cache's snapshots and resident ids. llb_bank_select(j) parks the live
// views in the current bank and loads bank j's (host pointers and counters; no device copy, no
// kernel change). A bank's device buffers are allocated on its first pass (pass_layer at pos 0),
// so an unused bank costs nothing. The x ping-pong and the shape record stay shared: a pass
// writes them fully before reading. The request layer (tools/sys1dual/dual.c) selects the
// bank before llb_pfx_begin, between requests only.
typedef struct {
  llb_dptr b[LLB_MAXL], S[LLB_MAXL], kc[LLB_MAXL], vc[LLB_MAXL];
  size_t kcap[LLB_MAXL];
  u32 r[LLB_MAXL], pos[LLB_MAXL];
  bool on[LLB_MAXL];
  u32 pfx_n, pfx_cap, pfx_K;
  int pfx_arm;
  u32* pfx_ids;
  PfxSnap pfx_s[64];
} LlbBank;
static LlbBank* llb_bank;
static int llb_nbank = -1, llb_bcur;

int llb_banks(void) {
  if (llb_nbank < 0) {
    const char* e = getenv("LLB_BANKS");
    llb_nbank = e != NULL && atoi(e) >= 2 && atoi(e) <= 16 ? atoi(e) : 1;
    if (llb_nbank > 1) {
      llb_bank = calloc((size_t)llb_nbank, sizeof *llb_bank);
      if (!llb_bank) {
        err_fail("banks: cannot allocate the bank table");
      }
      for (int j = 0; j < llb_nbank; j += 1) {
        llb_bank[j].pfx_arm = -1;
      }
      fprintf(stderr, "banks on %d (one weights copy, %d sequence states)\n", llb_nbank, llb_nbank);
    }
  }
  return llb_nbank;
}

// 0 = bank j is live; -1 = no such bank
int llb_bank_select(int j) {
  if (j < 0 || j >= llb_banks()) {
    return -1;
  }
  if (j == llb_bcur) {
    return 0;
  }
  LlbBank* o = &llb_bank[llb_bcur];
  memcpy(o->b, pass_b, sizeof pass_b), memcpy(o->S, pass_S, sizeof pass_S);
  memcpy(o->kc, pass_kc, sizeof pass_kc), memcpy(o->vc, pass_vc, sizeof pass_vc), memcpy(o->kcap, pass_kcap, sizeof pass_kcap);
  memcpy(o->r, pass_r, sizeof pass_r), memcpy(o->pos, pass_pos, sizeof pass_pos), memcpy(o->on, pass_on, sizeof pass_on);
  o->pfx_n = pfx_n, o->pfx_cap = pfx_cap, o->pfx_K = pfx_K, o->pfx_arm = pfx_arm, o->pfx_ids = pfx_ids;
  memcpy(o->pfx_s, pfx_s, sizeof pfx_s);
  LlbBank* n = &llb_bank[j];
  memcpy(pass_b, n->b, sizeof pass_b), memcpy(pass_S, n->S, sizeof pass_S);
  memcpy(pass_kc, n->kc, sizeof pass_kc), memcpy(pass_vc, n->vc, sizeof pass_vc), memcpy(pass_kcap, n->kcap, sizeof pass_kcap);
  memcpy(pass_r, n->r, sizeof pass_r), memcpy(pass_pos, n->pos, sizeof pass_pos), memcpy(pass_on, n->on, sizeof pass_on);
  pfx_n = n->pfx_n, pfx_cap = n->pfx_cap, pfx_K = n->pfx_K, pfx_arm = n->pfx_arm, pfx_ids = n->pfx_ids;
  memcpy(pfx_s, n->pfx_s, sizeof pfx_s);
  llb_bcur = j;
  return 0;
}

// layer i's state at pos: allocated (pos 0), zeroed (pos 0 again: the next prompt of a
// multi-prompt run, engine ids cut at ","), else pos must follow the last
static void pf_dump(u32 i, bool full, u32 pos, LlbShape sh);  // llb_pf.h: G3's LLB_PF_DUMP
static int spec_k;                 // LLB_SPEC's K (-1 off; defined below), set in warp_serve
static const char* spec_mtp;       // LLB_MTP
int llb_bx_max(void);
// tools/payload-anyshape.patch: full layer i's KV slots by position. A step at pos may write
// slots up to pos + 4096 (a fast prefill chunk) + 64 (spec drafts); the layer grows to
// min(pass_cap, max(that, 2x)) slots, zeroed past the copied ones, one layer's copy in flight
// at a time. LLB_SPEC / LLB_MTP / LLB_BATCH (whose kernels read every slot of the 2^dT tree)
// take all of pass_cap at once.
static void pass_kv(u32 i, u32 pos, LlbShape sh) {
  size_t row = (size_t)sh.nkv * sh.hd, need = (size_t)pos + 4096 + 64;
  if (spec_k >= 0 || spec_mtp || llb_bx_max() >= 2 || need > pass_cap) {
    need = pass_cap;
  }
  if (pass_kcap[i] >= need) {
    return;
  }
  size_t n = 2 * pass_kcap[i] > need ? 2 * pass_kcap[i] : need;
  n = n < pass_cap ? n : pass_cap;
  llb_dptr k = llb_alloc_zero(n * row * 4), v = llb_alloc_zero(n * row * 4);
  if (!k || !v) {
    fprintf(stderr, "warp pass: cannot allocate %zu KV slots (%.1f MiB) for layer %u; free VRAM or lower the context\n",
      n, 2.0 * n * row * 4 / 1048576.0, i);
    err_fail("warp pass: out of device memory for the KV cache");
  }
  if (pass_kcap[i] != 0) {
    if (!llb_copy(k, pass_kc[i], pass_kcap[i] * row * 4) || !llb_copy(v, pass_vc[i], pass_kcap[i] * row * 4)
      || !llb_sync() || cuMemFree(pass_kc[i]) != CUDA_SUCCESS || cuMemFree(pass_vc[i]) != CUDA_SUCCESS) {
      err_fail("warp pass: cannot grow the KV cache");
    }
  }
  pass_kc[i] = k;
  pass_vc[i] = v;
  pass_kcap[i] = n;
}

static void pass_layer(u32 i, bool full, u32 pos, u32 dT, LlbShape sh) {
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  (void)dT;
  pf_dump(i, full, pos, sh);
  pfx_restore(i, full, pos, sh);  // W1-KVPFX
  if (i >= LLB_MAXL) {
    err_fail("warp pass: more layers than LLB_MAXL");
  }
  if (full) {
    pass_kv(i, pos, sh);
  }
  if (!pass_on[i]) {
    if (pos != 0) {
      err_fail("warp pass: a layer's state must start at pos 0");
    }
    if (!full) {
      pass_b[i] = pass_zero(4 * (size_t)cv);
      pass_S[i] = pass_zero((size_t)sh.lvh * sh.sd * sh.sd);
    }
    pass_on[i] = true;
  } else if (pos == 0) {
    // the next prompt of a multi-prompt run (engine ids cut at ","): fresh state
    // === S2 twoheads SOURCE-SPEC — UNVERIFIED-BY-REEMIT (bend toolchain down) ===
    // Retention delta to apply with the re-emit: zero the state here only on an
    // explicit reset (a new/absent session); a same-conversation request starts
    // at the retained pos + 1 and skips this branch, passing the guard below
    // unchanged (pass_pos[i] + 1 == pos). Served request-side today in
    // tools/sys1dual/dual.c (retained session record). === end S2 SOURCE-SPEC ===
    bool z = full ? llb_zero(pass_kc[i], pass_kcap[i] * sh.nkv * sh.hd * 4)
        && llb_zero(pass_vc[i], pass_kcap[i] * sh.nkv * sh.hd * 4)
      : llb_zero(pass_b[i], 4 * (size_t)cv * 4) && llb_zero(pass_S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4);
    if (!z) {
      err_fail("warp pass: cannot reset the layer state");
    }
  } else if (pass_pos[i] + 1 != pos) {
    err_fail("warp pass: a layer served out of order");
  }
  pass_pos[i] = pos;

  pfx_take(i, full, pos, sh);  // W1-KVPFX
}

#include "llb_sampler_impl.h"
// W1-SAMPLER (host/llb_sampler.h): LLB_SAMPLE="temp=..,top_k=..,top_p=..,min_p=..,seed=..,
// logprobs=n,..." turns the head's device argmax into a host draw from the copied logits
// (one D2H of the vocab row a token). Unset, or argmax-equivalent (greedy, no bias, no
// penalties) with logprobs=0, it never runs: every path stays byte-identical. Under LLB_SPEC
// a served row's token is llb_smp_verify's (Leviathan et al. 2023 Alg. 1 with the drafter's
// deterministic proposal: accept w.p. p(draft), else a draw from p without it), so spec
// decode stays distribution-exact (tools/sampler/README.md, gate S). The penalty window reads
// smp_hist (every position's input token, both prefill tiers). llb_sampler_configure: the
// per-request C ABI (a server swaps params between requests; NULL or "" = back to argmax).
static LlbSmp* smp;
static int smp_env;
static u32* smp_hist;
static u32 smp_cap;
static float* smp_lg;
static u32 smp_nv;
static LlbSmpParams smp_par;

int llb_sampler_configure(const char* spec) {
  char err[160];
  LlbSmpParams p;
  llb_smp_defaults(&p);
  smp_env = 1;
  if (spec && *spec && llb_smp_parse(&p, spec, err, sizeof err) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  llb_smp_free(smp);
  smp = NULL;
  smp_par = p;
  return 0;
}

#include "llb_grammar_impl.h"
// G1 (host/llb_grammar.h): grammar-constrained decoding. LLB_GRAMMAR=<file.gbnf> or
// LLB_JSON_SCHEMA=<file.json>, with LLB_GRAMMAR_AT=<prompt token count> (the first
// generated position: the host never sees the prompt length) and LLB_TOKENIZER=<the pack's
// tokenizer.bin>; or llb_grammar_configure per request. From that position on, smp_pick
// masks a copy of the vocab row before the W1-SAMPLER chain (llama.cpp's grammar-first
// order: a bias stage ahead of it) and advances the match by the picked token; logprobs
// and the written logit stay the raw model's. Unset (the default) gr_on() is false and
// smp_on() is the W1-SAMPLER expression: every path stays byte-identical. Positions before
// LLB_GRAMMAR_AT keep the device argmax untouched when the sampler is argmax-equivalent.
// Refused with LLB_SPEC (a cycle verifies T rows at once; the mask is per accepted token).
static LlbGram* gr;
static LlbGVocab* gr_v;
static LlbGMatch* gr_m;
static int gr_env;
static u32 gr_at;
static uint32_t* gr_bits;
static float* gr_lg;
static u32 gr_nv;

// kind "gbnf" | "json_schema" (text = the grammar / schema), NULL or "" = off; at = the
// first generated position. 0, or -1 with a message on stderr (the grammar is then off).
int llb_grammar_configure(const char* kind, const char* text, u32 at) {
  char err[256];
  gr_env = 1;
  llb_gm_free(gr_m);
  gr_m = NULL;
  llb_gram_free(gr);
  gr = NULL;
  if (!kind || !*kind || !text) {
    return 0;
  }
  const char* sp = getenv("LLB_SPEC");
  if (sp && atoi(sp) > 0) {
    fprintf(stderr, "grammar: not with LLB_SPEC\n");
    return -1;
  }
  char* gb = NULL;
  if (!strcmp(kind, "json_schema")) {
    gb = llb_gram_schema_to_gbnf(text, err, sizeof err);
    if (!gb) {
      fprintf(stderr, "%s\n", err);
      return -1;
    }
  } else if (strcmp(kind, "gbnf")) {
    fprintf(stderr, "grammar: kind must be gbnf or json_schema\n");
    return -1;
  }
  gr = llb_gram_from_gbnf(gb ? gb : text, NULL, err, sizeof err);
  free(gb);
  if (!gr) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  if (!gr_v) {
    const char* tp = getenv("LLB_TOKENIZER");
    gr_v = tp ? llb_gv_load(tp, err, sizeof err) : NULL;
    if (!gr_v) {
      fprintf(stderr, "%s\n", tp ? err : "grammar: set LLB_TOKENIZER to the pack's tokenizer.bin");
      llb_gram_free(gr);
      gr = NULL;
      return -1;
    }
  }
  gr_m = llb_gm_new(gr, gr_v);
  gr_at = at;
  return 0;
}

static bool gr_on(void) {
  if (!gr_env) {
    gr_env = 1;
    const char* g = getenv("LLB_GRAMMAR");
    const char* j = getenv("LLB_JSON_SCHEMA");
    if (g || j) {
      const char* at = getenv("LLB_GRAMMAR_AT");
      FILE* f = fopen(g ? g : j, "rb");
      char* t = NULL;
      long n = -1;
      if (f && !fseek(f, 0, SEEK_END) && (n = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET) && (t = malloc((size_t)n + 1))) {
        n = (long)fread(t, 1, (size_t)n, f);
        t[n] = 0;
      }
      if (f) {
        fclose(f);
      }
      if (!t || !at || llb_grammar_configure(g ? "gbnf" : "json_schema", t, (u32)atoi(at)) != 0) {
        err_fail(!at ? "LLB_GRAMMAR/LLB_JSON_SCHEMA: set LLB_GRAMMAR_AT to the prompt's token count"
                     : "LLB_GRAMMAR/LLB_JSON_SCHEMA: cannot set up the grammar");
      }
      free(t);
    }
  }
  return gr_m != NULL;
}

static bool smp_on(void) {
  if (!smp_env && llb_sampler_configure(getenv("LLB_SAMPLE")) != 0) {
    err_fail("LLB_SAMPLE: bad sampler spec");
  }
  return !(llb_smp_is_argmax(&smp_par) && smp_par.n_logprobs == 0) || gr_on();
}

// position pos's input token (prompt or generated) into the penalty history
static void smp_note(u32 pos, u32 tok) {
  if (!smp_env) {
    smp_on();
  }
  if (pos == 0 && gr_m) {
    llb_gm_reset(gr_m);   // the next prompt of a multi-prompt run
  }
  if (llb_smp_is_argmax(&smp_par) && smp_par.n_logprobs == 0) {
    return;
  }
  if (pos >= smp_cap) {
    u32 c = smp_cap ? smp_cap : 4096;
    while (c <= pos) c *= 2;
    smp_hist = realloc(smp_hist, (size_t)c * 4);
    if (!smp_hist) err_fail("LLB_SAMPLE: cannot grow the history");
    memset(smp_hist + smp_cap, 0, (size_t)(c - smp_cap) * 4);
    smp_cap = c;
  }
  smp_hist[pos] = tok;
}

// the token for pos + 1 from the nv logits at lg (device): a draw, or (draft >= 0) the spec
// verify of the drafter's token; its id and logit replace words 1 and 0 of the four words at yp
static void smp_pick(u64 yp, llb_dptr lg, u32 nv, u32 pos, int draft) {
  bool gram = gr_on() && pos + 1 >= gr_at;
  if (!gram && llb_smp_is_argmax(&smp_par) && smp_par.n_logprobs == 0) {
    return;   // only the grammar turned the host pick on, and not yet: the device argmax
  }
  if (!smp || nv != smp_nv) {
    llb_smp_free(smp);
    smp = llb_smp_new(&smp_par, (int)nv);
    smp_lg = realloc(smp_lg, (size_t)nv * 4);
    smp_nv = nv;
    if (!smp || !smp_lg) err_fail("LLB_SAMPLE: cannot allocate the sampler");
  }
  float y[4];
  if (!llb_read(smp_lg, lg, (size_t)nv * 4) || !llb_read(y, yp, 16)) {
    err_fail("LLB_SAMPLE: cannot read the logits");
  }
  llb_smp_history(smp, (const int*)smp_hist, (int)pos + 1);
  int acc = 0;
  int t;
  if (gram) {
    // greedy fast path: the raw argmax, when the grammar allows it, is the masked argmax
    int fast = -1;
    if (llb_smp_is_argmax(&smp_par)) {
      int a = llb_smp_sample(smp, smp_lg, (uint64_t)pos + 1);
      fast = llb_gm_allows(gr_m, a) ? a : -1;
    }
    if (fast >= 0) {
      t = fast;
    } else {
      if (nv != gr_nv) {
        gr_bits = realloc(gr_bits, (size_t)(nv + 31) / 32 * 4);
        gr_lg = realloc(gr_lg, (size_t)nv * 4);
        gr_nv = nv;
        if (!gr_bits || !gr_lg) err_fail("grammar: cannot allocate the mask");
      }
      if (llb_gm_mask(gr_m, gr_bits, (int)nv) == 0) {
        err_fail("grammar: dead end (no token can continue the grammar)");
      }
      memcpy(gr_lg, smp_lg, (size_t)nv * 4);
      llb_gm_apply(gr_bits, gr_lg, (int)nv);
      t = llb_smp_sample(smp, gr_lg, (uint64_t)pos + 1);
    }
    if (llb_gm_accept(gr_m, t) != 0) {
      err_fail("grammar: the picked token is not allowed");
    }
  } else {
    t = draft >= 0 ? llb_smp_verify(smp, smp_lg, draft, NULL, (uint64_t)pos + 1, &acc)
                   : llb_smp_sample(smp, smp_lg, (uint64_t)pos + 1);
  }
  y[0] = smp_lg[t];
  y[1] = (float)t;
  if (!llb_write(yp, y, 16)) {
    err_fail("LLB_SAMPLE: cannot write the token");
  }
  if (smp_par.n_logprobs > 0) {
    int ids[LLB_SMP_MAX_LOGPROBS];
    float lp[LLB_SMP_MAX_LOGPROBS], lt = 0.0f;
    llb_smp_logprobs(smp_lg, (int)nv, smp_par.n_logprobs, ids, lp, t, &lt);
    fprintf(stderr, "logprobs %u %d %.6f", pos + 1, t, (double)lt);
    for (int i = 0; i < smp_par.n_logprobs && i < (int)nv; i += 1) {
      fprintf(stderr, " %d:%.6f", ids[i], (double)lp[i]);
    }
    fprintf(stderr, "\n");
  }
}

#include "llb_pf.h"
// MTP27 (tools/payload-spec.patch; env LLB_SPEC=k, 0..7, default unset = off):
// speculative decoding on the folded pass. A cycle at (pos, tok) runs T = 1 + the
// drafts (<= k) rows, positions pos .. pos + T - 1, in one pass: each weight GEMV
// once for T tokens (the llb_spec_* T-row leaves; row tt of a T launch == the T = 1
// leaf, the mode-9 one), the small kernels per row as llb_pass_run's. Row r's four
// words are served while the engine's next (pos, tok) is (pos + r, row r's input);
// otherwise a new cycle starts there. Unserved rows cost nothing to drop: gated
// delta is out of place (a ring of k + 2 states a layer, row r reads slot sc + r),
// conv a 16-slot ring by position, KV slots rewritten. Stage 1 drafts by prompt
// lookup over the host's token history; LLB_SPEC=0 is the T = 1 identity path.
// Mode 9 formats only (Q8_0 attention / SSM GEMVs, Q4_K FFN, Q6_K attn_out and head).
// MTP28 (tools/payload-spec2.patch): the small kernels run T rows a launch (their
// llb_t_* twins, row r = blockIdx.y); only the embeddings still loop per row.
static int  spec_k = -1;
static u32  spec_R, spec_T, spec_p0, spec_nv, spec_sc, spec_cm, spec_tin[8], *spec_hist;
static u64  spec_cyc, spec_nrow, spec_srv, spec_dft[8], spec_acc[8];
static llb_dptr sp_ring[64], sp_S[64], sp_kc[64], sp_vc[64];
static llb_dptr sp_x, sp_a, sp_x1, sp_h2, sp_g, sp_u, sp_s, sp_qx, sp_dx, sp_S4, sp_z, sp_bp, sp_ap;
static llb_dptr sp_qgn, sp_kb, sp_vb, sp_lg, sp_y, sp_hb, sp_bet, sp_dec, sp_cnv, sp_qt, sp_go;
static llb_dptr sp_nm, sp_qg, sp_at, sp_hn, sp_att;
// LLB_MTP=file (tools/mtp27/mtp_pack.py): the NextN drafter instead of the lookup; its 15
// tensors, the draft head slice and ids, its own KV. LLB_SPEC_ADAPT=1: k adapts in 1 .. k
static const char* spec_mtp;
static int  spec_adapt;
static u32  spec_ke, mt_nv, *mt_ids;
static llb_dptr mt_w[17], mt_kc, mt_vc, mt_hn, mt_z, mt_cat, mt_y;

// the cycle ending: rows served of T; position j's draft accepted if row j was served
static void spec_end(void) {
  for (u32 j = 1; j < spec_T; j += 1) {
    spec_dft[j] += 1;
    spec_acc[j] += spec_nv > j;
  }
  spec_srv += spec_nv;
  if (spec_T > 1) {
    spec_ke = spec_nv == spec_T ? spec_ke + (spec_ke < (u32)spec_k) : spec_ke - (spec_ke > 1);
  }
}

static void spec_stats(void) {
  spec_end();
  spec_T = 0;
  fprintf(stderr, "spec: %s%s k %d cycles %llu rows %llu served %llu (%.3f a cycle); accepted/drafted by position:",
    mt_ids ? "mtp" : "lookup", spec_adapt ? " adaptive" : "", spec_k, (unsigned long long)spec_cyc, (unsigned long long)spec_nrow, (unsigned long long)spec_srv,
    spec_cyc ? (double)spec_srv / spec_cyc : 0.0);
  for (int j = 1; j <= spec_k; j += 1) {
    fprintf(stderr, " %llu/%llu", (unsigned long long)spec_acc[j], (unsigned long long)spec_dft[j]);
  }
  fprintf(stderr, "\n");
}

// LLB_MTP's file onto the device. Two containers:
//  MTP1: "MTP1", u32 N = 17, N x (u64 off, u64 bytes) - entries in tools/mtp27/mtp_pack.py ORDER.
//  AUX1: "AUX1", u32 ver, u32 n, u32 flags, n x 72B {char name[32]; u32 dtype; u32 rank;
//        u32 ne[4]; u64 off; u64 bytes} - the 17 engine slots bind BY NAME (tools/auxpack/,
//        S6-b); extra entries are ignored; duplicates refused.
// Each slot's byte size is checked against the same shape math in both formats.
static void mtp_load(volatile WarpOp* o, LlbShape sh, u32 dT) {
  u32 d = sh.d, f = sh.f, qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd, k4 = LLB_FM_Q4_K, k6 = LLB_FM_Q6_K;
  FILE* fp = fopen(spec_mtp, "rb");
  u32 hd[4] = { 0, 0, 0, 0 };
  u64 tab[17][2];
  if (!fp || fread(hd, 4, 4, fp) != 4) {
    err_fail("LLB_MTP: cannot read the mtp pack header");
  }
  if (hd[0] == 0x3150544du) {                       /* "MTP1": legacy positional table */
    if (hd[1] != 17 || fseek(fp, 8, SEEK_SET) != 0 || fread(tab, 16, 17, fp) != 17) {
      err_fail("LLB_MTP: cannot read the MTP1 header");
    }
  } else if (hd[0] == 0x31585541u) {                /* "AUX1": named entries (S6-b) */
    static const char* mtp_nm[17] = {
      "nextn.enorm", "nextn.hnorm", "nextn.eh_proj", "attn_norm", "attn_q", "attn_k", "attn_v",
      "attn_q_norm", "attn_k_norm", "attn_output", "post_attention_norm", "ffn_gate", "ffn_up",
      "ffn_down", "nextn.shared_head_norm", "mtp.head_slice", "mtp.ids" };
    u32 n = hd[2];
    bool got[17] = { false };
    if (hd[1] != 1 || n < 17 || n > 64) {
      err_fail("LLB_MTP: cannot read the AUX1 header");
    }
    for (u32 i = 0; i < n; i += 1) {
      char e[72], nm[33];
      if (fread(e, 1, 72, fp) != 72) {
        err_fail("LLB_MTP: cannot read the AUX1 entries");
      }
      memcpy(nm, e, 32); nm[32] = 0;
      for (u32 j = 0; j < 17; j += 1) {
        if (!strcmp(nm, mtp_nm[j])) {
          if (got[j]) err_fail("LLB_MTP: duplicate AUX1 entry");
          memcpy(&tab[j][0], e + 56, 8);   /* off (u64 LE) */
          memcpy(&tab[j][1], e + 64, 8);   /* bytes (u64 LE) */
          got[j] = true;
        }
      }
    }
    for (u32 j = 0; j < 17; j += 1) {
      if (!got[j]) err_fail("LLB_MTP: AUX1 pack is missing a named entry");
    }
  } else {
    err_fail("LLB_MTP: cannot read the mtp pack header");
  }
  u64 hb = (u64)LLB_FM_RB(k6, d), nv = tab[16][1] / 4;
  u64 want[17] = { 4ull * d, 4ull * d, (u64)d * LLB_FM_RB(k4, 2 * d), 4ull * d, 2ull * qa * hb, kv * hb, kv * hb,
    4ull * sh.hd, 4ull * sh.hd, (u64)d * LLB_FM_RB(k6, qa), 4ull * d, (u64)f * LLB_FM_RB(k4, d),
    (u64)f * LLB_FM_RB(k4, d), (u64)d * LLB_FM_RB(k4, f), 4ull * d, nv * hb, nv * 4 };
  bool ok = o->bg == LLB_FM_Q6_K && nv % 16 == 0 && nv + 4 * AMAX_G <= (1u << 18) && (2 * d) % 256 == 0;
  for (u32 j = 0; j < 17; j += 1) {
    ok = ok && tab[j][1] == want[j];
  }
  u64 tot = tab[16][0] + tab[16][1];
  void* h = ok ? malloc(tot) : NULL;
  if (!h || fseek(fp, 0, SEEK_SET) != 0 || fread(h, 1, tot, fp) != tot) {
    err_fail("LLB_MTP: bad entry sizes or a short file");
  }
  fclose(fp);
  llb_dptr b = pass_zero((tot + 3) / 4);
  mt_ids = malloc(nv * 4);
  if (!b || !mt_ids || !llb_write(b, h, tot)) {
    err_fail("LLB_MTP: cannot load the drafter");
  }
  memcpy(mt_ids, (char*)h + tab[16][0], nv * 4);
  free(h);
  for (u32 j = 0; j < 17; j += 1) {
    mt_w[j] = b + tab[j][0];
  }
  mt_nv = (u32)nv;
  u32 T = (u32)spec_k + 1;
  mt_kc = pass_zero(((size_t)1 << dT) * kv);
  mt_vc = pass_zero(((size_t)1 << dT) * kv);
  mt_hn = pass_zero(T * d); mt_z = pass_zero(T * d); mt_cat = pass_zero(T * 2 * d); mt_y = pass_zero(4);
}

static void spec_init(volatile WarpOp* o, LlbShape sh, u32 dT, u32 nl) {
  u32 lk = sh.lkh * sh.sd, lv = sh.lvh * sh.sd, cv = 2 * lk + lv, qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd;
  u32 T = (u32)spec_k + 1, nh = o->gm;
  bool ok = spec_k >= 0 && spec_k <= 7 && llb_q4dp == 127 && sh.sd == 128 && o->bb == LLB_FM_Q4_K
    && o->bg == LLB_FM_Q6_K && sh.d % 256 == 0 && sh.f % 256 == 0 && lv % 256 == 0 && qa % 256 == 0
    && nh % 16 == 0 && nh + 4 * AMAX_G <= (1u << 18) && llb_q4dp_ok(sh.d, 1) && llb_q4dp_ok(sh.f, 2);
  for (u32 i = 0; i < nl; i += 1) {
    const volatile u32* t = gpu_mail->pt[i];
    ok = ok && t[2] >> 28 == LLB_FM_Q4_K && t[3] >> 28 == LLB_FM_Q4_K && t[4] >> 28 == LLB_FM_Q4_K;
    ok = ok && ((i + 1) % 4 == 0
      ? t[14] >> 28 == 1 && t[15] >> 28 == 1 && t[16] >> 28 == 1 && t[19] >> 28 == LLB_FM_Q6_K
      : t[5] >> 28 == 1 && t[6] >> 28 == 1 && t[7] >> 28 == 1 && t[8] >> 28 == 1 && t[13] >> 28 == 1);
  }
  if (!ok) {
    err_fail("LLB_SPEC: needs k 0..7, LLB_Q4_DP4A=9 and the mode-9 formats / shapes");
  }
  spec_R = T + 1;
  spec_cm = sh.d > sh.f ? sh.d : sh.f;
  spec_cm = spec_cm > lv ? spec_cm : lv;
  spec_cm = spec_cm > qa ? spec_cm : qa;
  for (u32 i = 0; i < nl; i += 1) {
    if ((i + 1) % 4 == 0) {
      sp_kc[i] = pass_zero(((size_t)1 << dT) * kv);
      sp_vc[i] = pass_zero(((size_t)1 << dT) * kv);
    } else {
      sp_ring[i] = pass_zero(16 * (size_t)cv);
      sp_S[i] = pass_zero((size_t)spec_R * sh.lvh * sh.sd * sh.sd);
    }
  }
  sp_x = pass_zero(T * sh.d); sp_a = pass_zero(T * sh.d); sp_x1 = pass_zero(T * sh.d); sp_h2 = pass_zero(T * sh.d);
  sp_g = pass_zero(T * sh.f); sp_u = pass_zero(T * sh.f); sp_s = pass_zero(T * sh.f);
  sp_qx = pass_zero(T * spec_cm / 2); sp_dx = pass_zero(T * spec_cm / 32); sp_S4 = pass_zero(T * spec_cm / 8);
  sp_z = pass_zero(T * lv); sp_bp = pass_zero(T * sh.lvh); sp_ap = pass_zero(T * sh.lvh);
  sp_qgn = pass_zero(T * 2 * qa); sp_kb = pass_zero(T * kv); sp_vb = pass_zero(T * kv);
  sp_lg = pass_zero((size_t)T << 18); sp_y = pass_zero(T * 4);
  sp_hb = pass_zero(T * sh.d); sp_bet = pass_zero(T * sh.lvh); sp_dec = pass_zero(T * sh.lvh);
  sp_cnv = pass_zero(T * cv); sp_qt = pass_zero(T * 2 * lv); sp_go = pass_zero(T * lv); sp_nm = pass_zero(T * lv);
  sp_qg = pass_zero(T * 2 * qa); sp_at = pass_zero(T * qa); sp_att = pass_zero(((size_t)2 * T * sh.nh) << dT);
  sp_hn = pass_zero(T * sh.d);
  spec_ke = (u32)spec_k;
  if (spec_mtp && spec_k > 0) {
    mtp_load(o, sh, dT);
  }
  spec_hist = malloc((size_t)4 << dT);
  if (!spec_hist) {
    err_fail("LLB_SPEC: cannot allocate the token history");
  }
  atexit(spec_stats);
}

// prompt lookup: the latest earlier match of the history's last L = 3, 2, 1 tokens,
// the (<= k) tokens after it into spec_tin[1 ..]; the count. ponytail: a linear scan
// a cycle (O(pos)); an n-gram index if contexts get long
static u32 spec_draft(u32 pos) {
  for (int L = 3; L >= 1; L -= 1) {
    for (int j = (int)pos - 1; j >= L - 1; j -= 1) {
      int m = 0;
      while (m < L && spec_hist[j - m] == spec_hist[pos - m]) {
        m += 1;
      }
      if (m == L) {
        u32 n = 0;
        while ((int)n < spec_k && j + 1 + (int)n <= (int)pos) {
          spec_tin[1 + n] = spec_hist[j + 1 + n];
          n += 1;
        }
        return n;
      }
    }
  }
  return 0;
}

#define SPEC_GO(...) do { if (!LLB_LAUNCH(__VA_ARGS__)) err_fail("device launch failed"); } while (0)
// row r's quantized x for a GEMV of c columns: qx [T][2c] bytes, dx [T][c / 32]
#define QX(r, c) (sp_qx + (u64)(r) * 2 * (c))
#define DX(r, c) (sp_dx + (u64)(r) * ((c) / 32) * 4)

// full attention for T rows at p0 after the q / k / v GEMVs (sp_qgn, sp_kb, sp_vb): norms,
// rope, the KV slots, attention, the gate; each row's quantized x for attn_out
static void spec_attn(LlbShape sh, u32 dT, u32 T, u32 p0, llb_dptr wqn, llb_dptr wkn, llb_dptr kc, llb_dptr vc) {
  u32 qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd, q2 = 2 * qa, one = 1, zero = 0, grp = sh.nh / sh.nkv, c = dT | p0 << 5;
  llb_dptr gate = sp_qg + qa * 4, ks = kc + (u64)p0 * kv * 4, vs = vc + (u64)p0 * kv * 4;
  llb_dptr s2 = sp_att, p2 = sp_att + (((u64)T * sh.nh) << dT) * 4, qx = sp_qx, dx = sp_dx;
  // k rows normed straight into their KV slots, v rows copied there (positions p0 .. p0 + T - 1)
  SPEC_GO(gpu_tqkr, sh.nh, T, 256, &sp_qgn, &wqn, &sp_qg, &q2, &one, &p0);
  SPEC_GO(gpu_tqkr, sh.nkv, T, 256, &sp_kb, &wkn, &ks, &kv, &zero, &p0);
  if (!llb_copy(vs, sp_vb, (size_t)T * kv * 4)) {
    err_fail("LLB_SPEC: device copy failed");
  }
  SPEC_GO(gpu_tattn4, sh.nh, T, 1024, &sp_qg, &kc, &vc, &sp_at, &s2, &p2, &c, &grp);
  SPEC_GO(gpu_togq, qa / 256, T, 256, &sp_at, &gate, &qa, &qx, &dx);
}

// the FFN block for T rows (sp_x += attn sp_a, then the FFN); x in place (rnaddq has
// read every row's x before the down GEMV)
static void spec_ffn(LlbShape sh, u32 T, llb_dptr post, llb_dptr wg, llb_dptr wu, llb_dptr wd) {
  u32 d = sh.d, f = sh.f, gd4 = (d / 8 + 255) / 256, gf4 = (f / 8 + 255) / 256;
  u32 rbg = LLB_FM_RB(LLB_FM_Q4_K, d), rbd = LLB_FM_RB(LLB_FM_Q4_K, f);
  llb_dptr qx = sp_qx, dx = sp_dx, S4 = sp_S4;
  SPEC_GO(gpu_trnaddq, 1, T, 1024, &sp_x, &sp_a, &post, &sp_x1, &sp_h2, &d, &qx, &dx);
  SPEC_GO(gpu_sq4s, gd4, T, 256, &qx, &S4, &d);
  SPEC_GO(gpu_sgu[T], f / 2, 1, 128, &wg, &wu, &rbg, &qx, &dx, &S4, &sp_g, &sp_u, &f);
  SPEC_GO(gpu_tsiluq, f / 256, T, 256, &sp_g, &sp_u, &sp_s, &f, &qx, &dx);
  SPEC_GO(gpu_sq4s, gf4, T, 256, &qx, &S4, &f);
  SPEC_GO(gpu_sre[T], d / 4, 1, 128, &wd, &rbd, &qx, &dx, &S4, &sp_x, &d, &sp_x1);
}

// one cycle's T rows: embeddings, the layers, the head into sp_y (four words a row)
static void spec_rows(volatile WarpOp* o, LlbShape sh, u32 dT, u32 nl) {
  u32 T = spec_T, p0 = spec_p0, d = sh.d, n32 = sh.lvh, hs16 = sh.lkh, rep = sh.lvh / sh.lkh;
  u32 lk = sh.lkh * sh.sd, lv = sh.lvh * sh.sd, cv = 2 * lk + lv, qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd;
  u32 q2 = 2 * qa, gconv = (cv + 255) / 256, gdy = sh.sd / 4;
  u32 rbe = LLB_FM_RB(o->bb, d);
  u32 rbo = LLB_FM_RB(LLB_FM_Q6_K, qa), rbh = LLB_FM_RB(LLB_FM_Q6_K, d), rg = o->brg, ro = o->bro;
  u32 nh = o->gm, ys = 1u << 18, ring = 1, nr = 0;
  u32 lv2 = 2 * sh.lvh * sh.sd;
  u64 Ssz = (u64)sh.lvh * sh.sd * sh.sd;
  union { u32 u; float f; } sq = { 0x413504f3u };
  llb_dptr qt = sp_qt, kt = sp_qt + lv * 4, cnvk = sp_cnv + lk * 4, vn = sp_cnv + 2 * lk * 4;
  llb_dptr qx = sp_qx, dx = sp_dx;
  for (u32 r = 0; r < T; r += 1) {
    llb_dptr er = o->r + (u64)spec_tin[r] * rbe, xr = sp_x + (u64)r * d * 4;
    SPEC_GO(gpu_embedk, (d + 255) / 256, 1, 256, &er, &d, &xr);
  }
  for (u32 i = 0; i < nl; i += 1) {
    u64 base = gpu_mail->pw[i];
    u32 p[20];
    for (u32 fi = 0; fi < 20; fi += 1) {
      p[fi] = gpu_mail->pt[i][fi] & 0x0fffffffu;
    }
    llb_dptr an = base + p[0];
    SPEC_GO(gpu_trnormq, 1, T, 1024, &sp_x, &an, &sp_hb, &d, &qx, &dx);
    if ((i + 1) % 4 == 0) {
      llb_dptr wqn = base + p[17], wkn = base + p[18], wo = base + p[19];
      Q8Cat qc = { { base + p[14], base + p[15], base + p[16], 0 }, { sp_qgn, sp_kb, sp_vb, 0 },
        { q2, q2 + kv, q2 + 2 * kv, q2 + 2 * kv } };
      SPEC_GO(gpu_sq8[T], qc.end[3], 1, 128, &rg, &qx, &dx, &qc, &nr, &p0);
      spec_attn(sh, dT, T, p0, wqn, wkn, sp_kc[i], sp_vc[i]);
      SPEC_GO(gpu_sq6[T], d / 16, 1, 128, &wo, &rbo, &qx, &dx, &sp_a, &d, &d);
    } else {
      llb_dptr dt = base + p[10], sa = base + p[9], cvw = base + p[11], sn = base + p[12];
      Q8Cat qc = { { base + p[5], base + p[6], base + p[7], base + p[8] }, { sp_ring[i], sp_z, sp_bp, sp_ap },
        { cv, cv + lv, cv + lv + n32, cv + lv + 2 * n32 } };
      SPEC_GO(gpu_sq8[T], qc.end[3], 1, 128, &rg, &qx, &dx, &qc, &ring, &p0);
      llb_dptr Si = sp_S[i];
      SPEC_GO(gpu_tmidconv, 1 + gconv, T, 256, &sp_bp, &sp_ap, &dt, &sa, &sp_bet, &sp_dec, &n32,
        &cvw, &sp_ring[i], &p0, &sp_cnv, &cv);
      SPEC_GO(gpu_tl2qk, 2 * hs16, T, 64, &sp_cnv, &cnvk, &qt, &kt, &sq.f, &hs16, &rep, &cv, &lv2);
      SPEC_GO(gpu_tgdl, n32, gdy, 128, &Si, &Ssz, &spec_sc, &spec_R, &T, &qt, &kt, &vn, &cv, &sp_dec, &sp_bet,
        &sp_go, &n32);
      SPEC_GO(gpu_tgrn, n32, T, 64, &sp_go, &sn, &sp_z, &sp_nm, &n32);
      SPEC_GO(gpu_tq8x2, lv / 256, T, 256, &sp_nm, &qx, &dx, &lv);
      Q8Cat oc = { { base + p[13], base + p[13], base + p[13], base + p[13] }, { sp_a, sp_a, sp_a, sp_a },
        { d, d, d, d } };
      SPEC_GO(gpu_sq8[T], d, 1, 128, &ro, &qx, &dx, &oc, &nr, &p0);
    }
    spec_ffn(sh, T, base + p[1], base + p[2], base + p[3], base + p[4]);
  }
  llb_dptr wn = o->k, wh = o->v;
  SPEC_GO(gpu_trnormq, 1, T, 1024, &sp_x, &wn, &sp_hn, &d, &qx, &dx);
  SPEC_GO(gpu_sq6[T], nh / 16, 1, 128, &wh, &rbh, &qx, &dx, &sp_lg, &nh, &ys);
  SPEC_GO(gpu_tamaxp, AMAX_G, T, AMAX_T, &sp_lg, &nh);
  SPEC_GO(gpu_tamax, 1, T, AMAX_G, &sp_lg, &nh, &sp_y);
}

// the NextN block (llama.cpp src/models/qwen35.cpp graph_mtp) for n rows at pa .. pa + n - 1:
// row r pairs token tk[r] with the hidden hs row r (h of position pa + r - 1): eh_proj of
// [rms(e) enorm, rms(h) hnorm], full attention on its own KV, the FFN, shared_head_norm
// into mt_hn (the next chain step's h); the head slice on the last row, its argmax's id
static u32 mtp_rows(volatile WarpOp* o, LlbShape sh, u32 dT, u32 n, u32 pa, const u32* tk, llb_dptr hs) {
  u32 d = sh.d, d2 = 2 * d, qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd, q2 = 2 * qa, ys = 1u << 18;
  u32 rbe = LLB_FM_RB(o->bb, d), rbx = LLB_FM_RB(LLB_FM_Q4_K, d2), rbh = LLB_FM_RB(LLB_FM_Q6_K, d);
  u32 rbo = LLB_FM_RB(LLB_FM_Q6_K, qa), g2 = (d2 / 8 + 255) / 256;
  llb_dptr qx = sp_qx, dx = sp_dx, S4 = sp_S4;
  for (u32 r = 0; r < n; r += 1) {
    llb_dptr er = o->r + (u64)tk[r] * rbe, xr = sp_x + (u64)r * d * 4, hr = hs + (u64)r * d * 4;
    llb_dptr ce = mt_cat + (u64)r * d2 * 4, ch = ce + (u64)d * 4, qr = QX(r, d2), dr = DX(r, d2);
    SPEC_GO(gpu_embedk, (d + 255) / 256, 1, 256, &er, &d, &xr);
    SPEC_GO(gpu_rnorm, 1, 1, 1024, &xr, &mt_w[0], &ce, &d);
    SPEC_GO(gpu_rnorm, 1, 1, 1024, &hr, &mt_w[1], &ch, &d);
    SPEC_GO(gpu_q8x2, d2 / 256, 1, 256, &ce, &qr, &dr, &d2);
  }
  SPEC_GO(gpu_sq4s, g2, n, 256, &qx, &S4, &d2);
  SPEC_GO(gpu_sre[n], d / 4, 1, 128, &mt_w[2], &rbx, &qx, &dx, &S4, &sp_x, &d, &mt_z);
  SPEC_GO(gpu_trnormq, 1, n, 1024, &sp_x, &mt_w[3], &sp_hb, &d, &qx, &dx);
  SPEC_GO(gpu_sq6[n], q2 / 16, 1, 128, &mt_w[4], &rbh, &qx, &dx, &sp_qgn, &q2, &q2);
  SPEC_GO(gpu_sq6[n], kv / 16, 1, 128, &mt_w[5], &rbh, &qx, &dx, &sp_kb, &kv, &kv);
  SPEC_GO(gpu_sq6[n], kv / 16, 1, 128, &mt_w[6], &rbh, &qx, &dx, &sp_vb, &kv, &kv);
  spec_attn(sh, dT, n, pa, mt_w[7], mt_w[8], mt_kc, mt_vc);
  SPEC_GO(gpu_sq6[n], d / 16, 1, 128, &mt_w[9], &rbo, &qx, &dx, &sp_a, &d, &d);
  spec_ffn(sh, n, mt_w[10], mt_w[11], mt_w[12], mt_w[13]);
  // the last row's quantized h is the head slice's x (row n - 1 of qx / dx)
  SPEC_GO(gpu_trnormq, 1, n, 1024, &sp_x, &mt_w[14], &mt_hn, &d, &qx, &dx);
  llb_dptr ql = QX(n - 1, d), dl = DX(n - 1, d);
  SPEC_GO(gpu_sq6[1], mt_nv / 16, 1, 128, &mt_w[15], &rbh, &ql, &dl, &sp_lg, &mt_nv, &ys);
  SPEC_GO(gpu_amaxp, AMAX_G, 1, AMAX_T, &sp_lg, &mt_nv, &mt_y);
  SPEC_GO(gpu_amax, 1, 1, AMAX_G, &sp_lg, &mt_nv, &mt_y);
  float y[4];
  if (!llb_read(y, mt_y, 16) || !(y[1] >= 0.0f && y[1] < (float)mt_nv)) {
    err_fail("LLB_MTP: cannot read the draft");
  }
  return mt_ids[(u32)y[1]];
}

// MTP drafts at (pos, tok): catch the drafter's KV up over the rows the last cycle served
// (positions q0 + 1 .. pos, their trunk h; at pos 0 the one row with h = 0, as llama.cpp's
// zero pending h), the last row's argmax the first draft; then chain on the drafter's own
// h (llama.cpp common/speculative.cpp:1294, the (h_p, x_p+1) pairing)
static u32 spec_mdraft(volatile WarpOp* o, LlbShape sh, u32 dT, u32 pos, u32 nprev) {
  u32 n = pos == 0 ? 1 : nprev, pa = pos + 1 - n, ke = spec_adapt ? spec_ke : (u32)spec_k;
  if (pos == 0 && !llb_zero(sp_hn, (size_t)sh.d * 4)) {
    err_fail("LLB_MTP: cannot reset h");
  }
  ke = pos + ke < (1u << dT) ? ke : (1u << dT) - 1 - pos;
  if (ke == 0) {
    return 0;
  }
  spec_tin[1] = mtp_rows(o, sh, dT, n, pa, spec_hist + pa, sp_hn);
  for (u32 j = 2; j <= ke; j += 1) {
    llb_dptr hl = mt_hn + (u64)(j == 2 ? n - 1 : 0) * sh.d * 4;
    spec_tin[j] = mtp_rows(o, sh, dT, 1, pos + j - 1, spec_tin + j - 1, hl);
  }
  return ke;
}

// INTEG1: LLB_SPEC after LLB_PREFILL=fast. The fast chunks (llb_pf.h) fill the exact path's
// layer state (pass_*); a cycle reads its own (sp_*). spec_pf, after each chunk: its ids into
// the history; with LLB_MTP the drafter's KV rows of the chunk, batched like a pf layer:
// [rms(e_p) enorm, rms(h_p-1) hnorm] -> eh_proj -> attn_norm -> k / v -> k norm + rope -> slots
// (q, attention, FFN and head skipped: a later row reads only the KV), h_p = rms(x_p) out norm
// from the chunk's last layer, the chunk's last h carried in pf_hn row 0 (zero at pos 0).
// ponytail: e / cat rows by per-row launches (3 T a chunk); one strided kernel if it shows.
static bool spec_pfnew;
static llb_dptr pf_hn, pf_cat;
static void spec_pf(u32 T, llb_dptr x, LlbShape sh) {
  if (spec_k < 0) {
    return;
  }
  memcpy(spec_hist + pf_pos0, pf_tk, (size_t)T * 4);
  spec_pfnew = true;
  if (!mt_ids) {
    return;
  }
  u32 d = sh.d, d2 = 2 * d, kv = sh.nkv * sh.hd, zero = 0, pos0 = pf_pos0;
  if (!pf_hn) {
    pf_hn = pf_alloc(((size_t)pf_T + 1) * d * 4);
    pf_cat = pf_alloc((size_t)pf_T * d2 * 4);
  }
  llb_dptr h1 = pf_hn + (u64)d * 4, wn = pf_wn;
  if ((pos0 == 0 && !llb_zero(pf_hn, (size_t)d * 4)) || !LLB_LAUNCH(gpu_pfrms, T, 1, 1024, &x, &wn, &h1, &d)) {
    err_fail("LLB_MTP: fast prefill h failed");
  }
  for (u32 r = 0; r < T; r += 1) {
    llb_dptr er = pf_er[r], xr = pf_h + (u64)r * d * 4, hr = pf_hn + (u64)r * d * 4;
    llb_dptr ce = pf_cat + (u64)r * d2 * 4, ch = ce + (u64)d * 4;
    SPEC_GO(gpu_embedk, (d + 255) / 256, 1, 256, &er, &d, &xr);
    SPEC_GO(gpu_rnorm, 1, 1, 1024, &xr, &mt_w[0], &ce, &d);
    SPEC_GO(gpu_rnorm, 1, 1, 1024, &hr, &mt_w[1], &ch, &d);
  }
  pf_q8(pf_cat, d2, T);
  pf_mm(LLB_FM_Q4_K, mt_w[2], d, d2, T, pf_x1, 0);
  SPEC_GO(gpu_pfrms, T, 1, 1024, &pf_x1, &mt_w[3], &pf_h, &d);
  pf_q8(pf_h, d, T);
  pf_mm(LLB_FM_Q6_K, mt_w[5], kv, d, T, pf_kb, 0);
  pf_mm(LLB_FM_Q6_K, mt_w[6], kv, d, T, pf_vb, 0);
  SPEC_GO(gpu_pfqkr, sh.nkv, T, 256, &pf_kb, &mt_w[8], &pf_kn, &kv, &zero, &pos0);
  SPEC_GO(gpu_pfslot, T, 2, 1024, &pf_kn, &pf_vb, &mt_kc, &mt_vc, &pos0, &kv);
  if (!llb_copy(pf_hn, pf_hn + (u64)T * d * 4, (size_t)d * 4)) {
    err_fail("LLB_MTP: fast prefill h carry failed");
  }
}

// the first cycle after fast chunks (pos = the prompt's last token): pass_* over 0 .. pos - 1
// into sp_* -- KV rows; S into slot 0; the conv rows of tokens pos - 3 .. pos - 1 from the
// exact ring (token t at ring index pass_r - (pos - t), slot (index + 3) & 3) to slot t & 15
static void spec_take(LlbShape sh, u32 nl, u32 pos) {
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd;
  size_t Ssz = (size_t)sh.lvh * sh.sd * sh.sd * 4;
  bool ok = true;
  for (u32 i = 0; i < nl; i += 1) {
    if ((i + 1) % 4 == 0) {
      ok = ok && llb_copy(sp_kc[i], pass_kc[i], (size_t)pos * kv * 4) && llb_copy(sp_vc[i], pass_vc[i], (size_t)pos * kv * 4);
    } else {
      ok = ok && llb_zero(sp_ring[i], 16 * (size_t)cv * 4) && llb_copy(sp_S[i], pass_S[i], Ssz);
      for (u32 t = pos >= 3 ? pos - 3 : 0; t < pos; t += 1) {
        ok = ok && llb_copy(sp_ring[i] + (u64)(t & 15) * cv * 4, pass_b[i] + (u64)((pass_r[i] + t - pos + 3) & 3) * cv * 4, (size_t)cv * 4);
      }
    }
  }
  if (!ok || (mt_ids && !llb_copy(sp_hn, pf_hn, (size_t)sh.d * 4))) {
    err_fail("LLB_SPEC: cannot take the fast prefill's state");
  }
  spec_sc = 0;
}

// the pass op under LLB_SPEC: serve a row of the running cycle, or start one at (pos, tok)
static void spec_pass(volatile WarpOp* o, u64 yp, LlbShape sh, u32 dT, u32 pos) {
  u32 nl = o->d;
  u32 tok = (u32)((o->w - o->r) / LLB_FM_RB(o->bb, sh.d));
  if (pos >> dT) {
    err_fail("LLB_SPEC: pos past the context");
  }
  spec_hist[pos] = tok;
  smp_note(pos, tok);
  u32 nprev = spec_pfnew ? 1 : spec_nv;
  if (spec_pfnew || pos == 0 || pos != spec_p0 + spec_nv || spec_nv == spec_T || tok != spec_tin[spec_nv]) {
    if (spec_pfnew) {
      spec_take(sh, nl, pos);
      spec_pfnew = false;
    } else if (pos == 0) {
      u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd;
      for (u32 i = 0; i < nl; i += 1) {
        bool z = (i + 1) % 4 == 0
          ? llb_zero(sp_kc[i], ((size_t)1 << dT) * kv * 4) && llb_zero(sp_vc[i], ((size_t)1 << dT) * kv * 4)
          : llb_zero(sp_ring[i], 16 * (size_t)cv * 4) && llb_zero(sp_S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4);
        if (!z) {
          err_fail("LLB_SPEC: cannot reset the layer state");
        }
      }
      if (mt_ids && !(llb_zero(mt_kc, ((size_t)1 << dT) * kv * 4) && llb_zero(mt_vc, ((size_t)1 << dT) * kv * 4))) {
        err_fail("LLB_MTP: cannot reset the drafter KV");
      }
      spec_sc = 0;
    } else if (pos != spec_p0 + spec_nv) {
      err_fail("LLB_SPEC: a step out of order");
    } else {
      spec_sc = (spec_sc + spec_nv) % spec_R;
    }
    spec_end();
    spec_tin[0] = tok;
    spec_T = 1 + (mt_ids ? spec_mdraft(o, sh, dT, pos, nprev) : spec_draft(pos));
    spec_p0 = pos;
    spec_nv = 0;
    spec_cyc += 1;
    spec_nrow += spec_T;
    spec_rows(o, sh, dT, nl);
  }
  if (!llb_copy(yp, sp_y + (u64)spec_nv * 16, 16)) {
    err_fail("LLB_SPEC: device copy failed");
  }
  if (smp_on()) {
    smp_pick(yp, sp_lg + ((u64)spec_nv << 18) * 4, o->gm, pos, spec_nv + 1 < spec_T ? (int)spec_tin[spec_nv + 1] : -1);
  }
  spec_nv += 1;
}

// (G4 F1, FUS-V46 V4/V5) image rows in place of <|image_pad|> embeddings, from fus-a-wip
// fc7af11e. LLB_VIS_EMBD=<rows.f32>:<nx>:<ny> (nx * ny rows of d floats) and the prompt carries
// them as one run of nx * ny <|image_pad|> ids (LLB_IMG_PAD); each such slot takes its row in
// place of the embedding launch. (FOLD-CONTINUE MIMG) K images: LLB_VIS_EMBD is a ';'-separated
// list <a.f32>:<nx>:<ny>;<b.f32>:<nx>:<ny>;... (1 .. LLB_IMG_KMAX entries; a path may hold ':' but
// not ';'), and the prompt then holds K runs of <|image_pad|>, run k exactly image k's nx * ny ids,
// in list order. One entry = the old single-image path exactly. The CLI names the rope with LLB_VIS_ROPE:
//   seq    (G4 F1) connector rows keep the ordinary text rope, as the connector was trained
//          (inputs_embeds at text positions);
//   imrope (default; FUS-V46 V4) the Qwen3.5 tower's rows (vision/build/cudavis-cli IMAGE MMPROJ --out
//          rows.f32; tools/vision/f1.sh runs both): image k's first pad sets the rope's image map
//          llb_mr = {s0, n, nx, sh, p0} (tools/payload-imrope.patch), mtmd's MROPE: s0 = image k's first
//          KV slot S_k, p0 = its mtmd position base S_k - P_k, sh = P_k+1, where P_k = sum over the
//          earlier images j of (n_j - max(nx_j, ny_j)) (mtmd's n_pos = max(nx, ny)). KV slot = Bend
//          pos, unchanged; only the angle moves. The exact tier runs one slot a step and the KV holds
//          the earlier slots' rotated K, so switching the map at each image's first pad is exact. The
//          map stays zero otherwise: the old rope. One image: p0 = s0, sh = n - max(nx, ny), as before.
// Unset: no device call, the text path as before. SYS1 (served): the image is per request,
// set by dual.c's "image" field through llb_vis_request (the env is not read there), always
// seq; or (MIMG) by dual.c's "images" list through llb_vis_request_list, always imrope (tower
// rows). LLB_VIS_DUMPX=<file> (CLI): the step input rows of slots 0 .. LLB_VIS_DUMPN - 1
// (default the last image's end + 16), F32, written at exit (the V5 gate).
#define LLB_IMG_PAD 248056u
#define LLB_IMG_KMAX 16u
static int llb_img_state;  // 0 unread, 1 no image, 2 rows on the device
static u32 llb_img_n, llb_img_nx, llb_img_ny, llb_img_seen, llb_img_s0, llb_img_d;
static u32 llb_img_k, llb_img_cur, llb_img_P;  // images, the current one, its P_k
static u32 llb_img_kn[LLB_IMG_KMAX], llb_img_knx[LLB_IMG_KMAX], llb_img_kny[LLB_IMG_KMAX];
static size_t llb_img_koff[LLB_IMG_KMAX];  // image k's first row
static size_t llb_img_cap;
static llb_dptr llb_img_rows;
static bool llb_img_req;   // SYS1: llb_img_spec is this request's image ("" = none)
static bool llb_img_req_mr;  // SYS1 "images": imrope
static char llb_img_spec[16384];
static bool llb_img_mr;    // LLB_VIS_ROPE=imrope: the first pad sets llb_mr
static bool llb_mr_dirty;  // llb_mr was set (SYS1 zeroes it before the next request)
static u32 llb_mr_h[5];
static float* llb_img_dump;
static u32 llb_img_dn, llb_img_dmax;
static void llb_img_atexit(void) {
  const char* f = getenv("LLB_VIS_DUMPX");
  FILE* o = f ? fopen(f, "wb") : NULL;
  if (o && fwrite(llb_img_dump, 4, (size_t)llb_img_dn * llb_img_d, o) == (size_t)llb_img_dn * llb_img_d) {
    fprintf(stderr, "vision: step rows 0 .. %u (d %u) -> %s\n", llb_img_dn, llb_img_d, f);
  }
  if (o) fclose(o);
}
// CLI: every listed image must have met its run by exit
static void llb_img_count_atexit(void) {
  if (llb_img_state == 2 && !llb_img_req && llb_img_k > 1
    && llb_img_cur + (llb_img_seen == llb_img_n ? 1u : 0u) != llb_img_k) {
    fprintf(stderr, "vision: %u images in LLB_VIS_EMBD but the prompt held %u <|image_pad|> run(s)\n",
      llb_img_k, llb_img_cur + (llb_img_seen == llb_img_n ? 1u : 0u));
    fflush(stdout);
    _exit(1);
  }
}
static void llb_mr_set(const u32* h) {
  CUdeviceptr mp = 0;
  size_t mz = 0;
  if (cuModuleGetGlobal(&mp, &mz, gpu_lib, "llb_mr") != CUDA_SUCCESS || mz != 20
    || !llb_write(mp, h, 20)) err_fail("vision: cannot set the rope's image map");
}
static void llb_img_reset_map(void) {  // SYS1: a request after an imrope one starts from the zero map
  if (!llb_mr_dirty) return;
  static const u32 z[5] = {0};
  if (!llb_sync()) err_fail("vision: device sync failed");
  llb_mr_set(z);
  llb_mr_dirty = false;
}
void llb_vis_request(const char* path, unsigned n) {
  llb_img_reset_map();
  llb_img_req = true, llb_img_req_mr = false, llb_img_state = 0, llb_img_seen = 0, llb_img_spec[0] = 0;
  if (path && snprintf(llb_img_spec, sizeof llb_img_spec, "%s:%u:1", path, n) >= (int)sizeof llb_img_spec)
    err_fail("vision: the rows path is too long");
}
// MIMG: dual.c's "images" (tower rows, imrope), spec = the LLB_VIS_EMBD list form
void llb_vis_request_list(const char* spec) {
  llb_img_reset_map();
  llb_img_req = true, llb_img_req_mr = true, llb_img_state = 0, llb_img_seen = 0, llb_img_spec[0] = 0;
  if (spec && snprintf(llb_img_spec, sizeof llb_img_spec, "%s", spec) >= (int)sizeof llb_img_spec)
    err_fail("vision: the rows list is too long");
}
static void llb_img_load(u32 d) {
  const char* e = llb_img_req ? llb_img_spec : getenv("LLB_VIS_EMBD");
  llb_img_state = 1;
  if (!e || !*e) return;
  const char* rp = getenv("LLB_VIS_ROPE");
  llb_img_mr = false;
  if (!llb_img_req) {
    if (!rp || !strcmp(rp, "imrope")) llb_img_mr = true;
    else if (strcmp(rp, "seq"))
      err_fail("vision: LLB_VIS_ROPE must be imrope (default) or seq");
  } else llb_img_mr = llb_img_req_mr;
  const char* pf = getenv("LLB_PREFILL");
  u32 k = 0;
  size_t total = 0;
  float* r = NULL;
  for (const char* item = e; ; k++) {
    const char* end = strchr(item, ';');
    if (!end) end = item + strlen(item);
    if (k >= LLB_IMG_KMAX) err_fail("vision: LLB_VIS_EMBD lists more than 16 images");
    char path[4096];
    unsigned ax = 0, ay = 0;
    const char* c2 = NULL;
    for (const char* p = item; p < end; p++) if (*p == ':') c2 = p;
    const char* c1 = NULL;
    for (const char* p = item; c2 && p < c2; p++) if (*p == ':') c1 = p;
    char tail[64];
    if (c1 && (size_t)(end - c1) < sizeof tail) memcpy(tail, c1, (size_t)(end - c1)), tail[end - c1] = 0;
    else c1 = NULL;
    if (!c1 || (size_t)(c1 - item) >= sizeof path || sscanf(tail, ":%u:%u", &ax, &ay) != 2 || !ax || !ay)
      err_fail("vision: LLB_VIS_EMBD wants <rows.f32>:<nx>:<ny>[;<rows.f32>:<nx>:<ny>...]");
    if (getenv("LLB_SPEC") || getenv("LLB_MTP") || (pf && !strcmp(pf, "fast")))
      err_fail("vision: exact pass only with an image (no LLB_SPEC, LLB_MTP, LLB_PREFILL=fast)");
    memcpy(path, item, (size_t)(c1 - item));
    path[c1 - item] = 0;
    size_t want = (size_t)ax * ay * d;
    r = realloc(r, (total + want) * 4);
    FILE* f = fopen(path, "rb");
    size_t got = f && r ? fread(r + total, 4, want, f) : 0;
    int extra = f ? fgetc(f) != EOF : 0;
    if (f) fclose(f);
    if (got != want || extra) err_fail("vision: LLB_VIS_EMBD is not nx * ny rows of d floats");
    llb_img_kn[k] = ax * ay, llb_img_knx[k] = ax, llb_img_kny[k] = ay, llb_img_koff[k] = total / d;
    total += want;
    fprintf(stderr, "vision: rows from %s: %u tokens (%u x %u), d %u\n", path, ax * ay, ax, ay, d);
    if (!*end) break;
    item = end + 1;
  }
  if (total > llb_img_cap) {  // grows only (SYS1 reuses it across requests)
    llb_img_rows = llb_alloc_zero(total * 4);
    llb_img_cap = total;
  }
  if (!llb_img_rows || !llb_write(llb_img_rows, r, total * 4)) err_fail("vision: cannot upload the image rows");
  free(r);
  llb_img_k = k + 1, llb_img_cur = 0, llb_img_P = 0;
  llb_img_n = llb_img_kn[0], llb_img_nx = llb_img_knx[0], llb_img_ny = llb_img_kny[0], llb_img_d = d, llb_img_state = 2;
  static bool cnt_reg;
  if (!llb_img_req && llb_img_k > 1 && !cnt_reg) cnt_reg = true, atexit(llb_img_count_atexit);
  if (!llb_img_req && !llb_img_dump && getenv("LLB_VIS_DUMPX")) {
    const char* m = getenv("LLB_VIS_DUMPN");
    llb_img_dmax = m ? (u32)atoi(m) : 0;
    llb_img_dump = calloc((size_t)(llb_img_dmax ? llb_img_dmax : 16384) * d, 4);
    if (!llb_img_dump) err_fail("vision: dump alloc");
    atexit(llb_img_atexit);
  }
}
// a folded step's embedding: an <|image_pad|> slot takes the next image row (true: the
// caller skips the embedding launch)
static bool llb_img_step(u32 tok, u32 pos, llb_dptr x, u32 d) {
  if (llb_img_state == 0) llb_img_load(d);
  if (llb_img_state != 2) return false;
  if (tok != LLB_IMG_PAD) {
    if (llb_img_seen != 0 && llb_img_seen != llb_img_n)
      err_fail("vision: the <|image_pad|> ids must be one run of exactly the image's rows");
    if (llb_img_seen == llb_img_n && llb_img_cur + 1 < llb_img_k) {  // MIMG: image k done, k + 1 next
      llb_img_P += llb_img_n - (llb_img_nx > llb_img_ny ? llb_img_nx : llb_img_ny);
      llb_img_cur += 1, llb_img_seen = 0;
      llb_img_n = llb_img_kn[llb_img_cur], llb_img_nx = llb_img_knx[llb_img_cur], llb_img_ny = llb_img_kny[llb_img_cur];
    }
    return false;
  }
  if (llb_img_seen == 0 && llb_img_mr) {
    llb_img_s0 = pos;
    llb_mr_h[0] = pos, llb_mr_h[1] = llb_img_n, llb_mr_h[2] = llb_img_nx;
    llb_mr_h[3] = llb_img_P + llb_img_n - (llb_img_nx > llb_img_ny ? llb_img_nx : llb_img_ny);
    llb_mr_h[4] = pos - llb_img_P;
    // image k > 0: the queued steps of the text before it read the previous map
    if (llb_img_cur > 0 && !llb_sync()) err_fail("vision: device sync failed");
    llb_mr_set(llb_mr_h);
    llb_mr_dirty = true;
    if (llb_img_k > 1) fprintf(stderr, "vision: image %u of %u: ", llb_img_cur + 1, llb_img_k);
    fprintf(stderr, "vision: image rows at slots %u .. %u, grid %u x %u, rope map {s0 %u, n %u, nx %u, sh %u, p0 %u}\n",
      pos, pos + llb_img_n - 1, llb_img_nx, llb_img_ny, llb_mr_h[0], llb_mr_h[1], llb_mr_h[2], llb_mr_h[3], llb_mr_h[4]);
  } else if (llb_img_seen == 0) {
    llb_img_s0 = pos;
    if (llb_img_k > 1) fprintf(stderr, "vision: image %u of %u: ", llb_img_cur + 1, llb_img_k);
    fprintf(stderr, "vision: image rows at slots %u .. %u, sequential text rope\n", pos, pos + llb_img_n - 1);
  }
  if (pos != llb_img_s0 + llb_img_seen || llb_img_seen >= llb_img_n || d != llb_img_d) {
    if (llb_img_seen >= llb_img_n && llb_img_cur + 1 >= llb_img_k && llb_img_k > 1)
      err_fail("vision: the prompt holds more <|image_pad|> runs than LLB_VIS_EMBD lists images");
    err_fail("vision: the <|image_pad|> ids must be one run of exactly the image's rows");
  }
  if (!llb_copy(x, llb_img_rows + (llb_dptr)(llb_img_koff[llb_img_cur] + llb_img_seen) * d * 4, (size_t)d * 4))
    err_fail("vision: cannot copy an image row");
  llb_img_seen += 1;
  return true;
}
// LLB_VIS_DUMPX: the step input row x of slot pos, read after the caller's embed
static void llb_img_seen_row(u32 pos, llb_dptr x, u32 d) {
  if (!llb_img_dump) return;
  u32 cap = llb_img_dmax ? llb_img_dmax : llb_img_s0 + llb_img_n + 16;
  if (pos >= cap || pos >= 16384) return;
  if (!llb_read(llb_img_dump + (size_t)pos * d, x, (size_t)d * 4)) err_fail("vision: dump read");
  if (pos + 1 > llb_img_dn) llb_img_dn = pos + 1;
}

// BATCH-NATIVE (env LLB_BATCH=N, 2..8; unset, 0 or 1 = off and nothing below runs): N decide
// sequences in one forward a step. dual.c hands a decide_batch's K sequences (K <= BX_K) to
// llb_bx_plan and names the running request's one with llb_bx_select (by its ids). A step is
// one pass over the <= N rows holding an item (row r = its item at that item's own pos). Each
// layer's launches are recorded once per row (llb_graph's recorder: the same chains, args and
// order llb_pass_run would issue, on the row's own state and scratch) and issued interleaved:
// the exact-tier pair GEMVs (Q4_K / Q5_K / Q6_K / Q4_K + residual) as one llb_bx_* launch for
// all rows (one weight read; row t's arithmetic is the T = 1 kernel's: payload-batch.patch),
// every other launch row by row. A row's state (GDN conv ring + S, KV slots by position,
// x ping-pong, 81920-float scratch) is its own, so no row reads another's, and no launch's
// shape depends on its row's pos (pos rides as an argument), so rows at different positions
// record the same chain.
// BATCH-NATIVE-2 stage 3 (continuous batching): the lockstep is resumable. A planned
// request's first step runs it only until that request's item is done: its hidden rows and
// four words go to the host and its row is free. At every step boundary a free row takes the
// plan's next item, on a zeroed state or on the shared prefix's snapshot (stage 2), so the N
// rows stay full while items remain and an item is answered when it is done, not when its
// group is. The request's prompt steps are then served from its host rows (the fast tier's
// cue, pf_mark, turns the deferred steps into one batch op) and its last step gets the four
// words. The live single-sequence state (pass_*) is never touched. Off: llb_bx_select is
// never called and bx_take returns false.
#define BX_R 8
#define BX_K 256
typedef struct {
  u32 kvcap;
  llb_dptr b[64], S[64], kc[64], vc[64], x[3], blk, hid, y, atts;
  u32 r[64];
  CUstream st;  // the row's launches; rows join only at the T-row GEMVs (bx_issue)
  CUevent ev;
} BxSeq;
typedef struct {
  u32 n, cap, *ids;
  u32 st;          // 0 queued, 1 on a row, 2 done (rows and words on the host), 3 served
  float* hh;       // done: its n hidden rows
  size_t hcap;
  float y[4];      // done: its last step's four words
} BxItem;
typedef struct { LlbGNode q[256]; unsigned char b[256 * 128]; u32 n, nb; } BxRec;
static int bx_max = -1;
static BxSeq bx_s[BX_R], bx_pf;  // the rows; the shared prefix's snapshot (state at bx_L)
static BxItem bx_it[BX_K];
static u32 bx_nk, bx_nr, bx_next, bx_L, bx_tok, bx_steps;  // items, rows, next to admit, prefix
static u32 bx_t[BX_R];  // row r's pos
static int bx_ri[BX_R];  // row r's item; -1 free
static bool bx_live;  // the plan's rows are set up (bx_begin ran)
static int bx_cur = -1;
static WarpOp bx_o;  // stage 3b: the step's model words, dT, shape and grid, kept from bx_begin
static u32 bx_dT, bx_grid;
static LlbShape bx_sh;
static CUfunction bx_f[4][3];
static llb_dptr bx_dummy;
static BxRec bx_rec[BX_R];
static double bx_ms;
// LLB_BATCH_PROF=1: a sync around each issued node, its ms into [twin GEMVs, row-by-row launches];
// =2 (BATCH-NATIVE-2, the CUDA-graph lever's gate): per step, the host's issue ms and the step's ms
// to a sync after it, into [issue, step]
static int bx_prof = -1;
static double bx_pt[2];
// BATCH-NATIVE-2 lever 1 (LLB_BATCH_GRAPH=1; unset = off): a step's row launches (embed, every layer
// with its twin joins, the final norm) are captured across the row streams into one CUDA graph and
// launched as one; the next step updates the instantiated graph in place (cuGraphExecUpdate: the same
// kernels, new args), and a changed row set (an item done or admitted) instantiates it again. The
// launches, their args and their order per row are the uncaptured ones, so each row's arithmetic is.
static int bx_graph = -1;
static CUgraphExec bx_ge;
static CUevent bx_fork;
static u32 bx_gstat[2];  // instantiations, in-place updates
static double bx_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// BATCH-NATIVE-2: the corpus bump (H_BUMP, 1 KB pages handed out and never returned to it)
// for dual.c's LLB_MEMPROBE line; a 4-byte device read, so the host page stays put
unsigned llb_heap_bump(void) {
  u32 v = 0;
  return cuMemcpyDtoH(&v, (CUdeviceptr)(uintptr_t)a32_at(CORPUS, H_BUMP), 4) == CUDA_SUCCESS ? v : 0;
}

int llb_bx_max(void) {
  if (bx_max < 0) {
    const char* e = getenv("LLB_BATCH");
    bx_max = e != NULL && atoi(e) >= 2 && atoi(e) <= BX_R ? atoi(e) : 0;
    if (bx_max) {
      fprintf(stderr, "native batch on: up to %d decide sequences a forward (exact per item)\n", bx_max);
    }
  }
  return bx_max;
}

// the next decide_batch's sequences (k <= BX_K; ids copied); k = 0 clears the plan
void llb_bx_plan(const int* const* ids, const unsigned* n, unsigned k) {
  bx_nk = 0, bx_live = false, bx_cur = -1, bx_next = 0;  // FOLD-BEND-ALL: a new plan has not started (stage 3b reads bx_next)
  if (llb_bx_max() < 2 || k < 2 || k > BX_K) {
    return;
  }
  for (u32 j = 0; j < k; j += 1) {
    BxItem* q = &bx_it[j];
    if (n[j] < 2) {
      return;  // a one-token prompt has no deferred steps to serve: no plan
    }
    if (n[j] > q->cap) {
      q->ids = realloc(q->ids, (size_t)n[j] * 4);
      if (!q->ids) err_fail("native batch: cannot allocate the ids");
      q->cap = n[j];
    }
    q->n = n[j], q->st = 0;
    for (u32 t = 0; t < n[j]; t += 1) q->ids[t] = (u32)ids[j][t];
  }
  bx_nk = k;
}

// the running request's sequence (its ids): the first equal item not yet served, else the
// first equal one (a repeat is served again from its rows); NULL = none
void llb_bx_select(const int* ids, unsigned n) {
  bx_cur = -1;
  for (int pass = 0; ids && pass < 2 && bx_cur < 0; pass += 1) {
    for (u32 j = 0; j < bx_nk; j += 1) {
      bool eq = bx_it[j].n == n && (pass || bx_it[j].st != 3);
      for (u32 t = 0; eq && t < n; t += 1) eq = bx_it[j].ids[t] == (u32)ids[t];
      if (eq) {
        bx_cur = (int)j;
        break;
      }
    }
  }
}

// kind of a recorded launch: 1..4 = a pair GEMV with a T-row twin, 0 = row by row
static int bx_kind(CUfunction f) {
  return f == gpu_warpk ? 1 : f == gpu_warp5 ? 2 : f == gpu_warp6 ? 3 : f == gpu_warpke ? 4 : 0;
}

static void bx_load(void) {
  static const char* nm[4] = { "q4k", "q5k", "q6k", "q4kr" };
  static const u32 tv[3] = { 2, 4, 8 };
  char b[32];
  for (u32 k = 0; k < 4; k += 1) {
    for (u32 j = 0; j < 3; j += 1) {
      snprintf(b, sizeof b, "llb_bx_%s_%u", nm[k], tv[j]);
      if (cuModuleGetFunction(&bx_f[k][j], gpu_lib, b) != CUDA_SUCCESS) {
        err_fail("native batch: the llb_bx kernels are missing (payload-batch.patch)");
      }
    }
  }
  bx_dummy = pass_zero(1u << 18);
  for (u32 r = 0; r < BX_R; r += 1) {
    if (cuStreamCreate(&bx_s[r].st, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS
      || cuEventCreate(&bx_s[r].ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
      err_fail("native batch: cannot create the row streams");
    }
  }
}

static void bx_go(CUstream st, CUfunction f, u32 gx, u32 gy, u32 bx, u32 sm, void** a) {
  if (cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, sm, st, a, NULL) != CUDA_SUCCESS) {
    err_fail("device launch failed");
  }
}

static llb_dptr bx_need(llb_dptr p, size_t bytes, bool zero) {
  if (p) {
    return zero && !llb_zero(p, bytes) ? (err_fail("native batch: cannot reset a row"), 0) : p;
  }
  return pass_zero((bytes + 3) / 4);
}

// the recorded args of node q: arg i's bytes
static const unsigned char* bx_arg(const BxRec* rc, const LlbGNode* q, u32 i) {
  const unsigned short* z;
  llb_psz(q->f, q->na, &z);
  u32 at = q->off;
  for (u32 j = 0; j < i; j += 1) at += ((u32)z[j] + 7) & ~7u;
  return rc->b + at;
}

// issue the na rows' recorded launches, node by node: the pair GEMVs as T-row launches
static void bx_issue(u32 na, const u32* act) {
  u32 nn = bx_rec[0].n;
  for (u32 a = 1; a < na; a += 1) {
    if (bx_rec[a].n != nn) err_fail("native batch: rows recorded different chains");
  }
  if (bx_prof < 0) bx_prof = getenv("LLB_BATCH_PROF") != NULL ? atoi(getenv("LLB_BATCH_PROF")) : 0;
  for (u32 k = 0; k < nn; k += 1) {
    const LlbGNode* q0 = &bx_rec[0].q[k];
    double tp = bx_prof == 1 && llb_sync() ? bx_now() : 0;
    int kind = bx_kind(q0->f);
    bool same = true;
    u32 W0[2], cg = 0, n = 0, d = 0;
    u64 W = 0;
    if (kind) {
      memcpy(&W, bx_arg(&bx_rec[0], q0, 0), 8);
      memcpy(&W0[0], bx_arg(&bx_rec[0], q0, 1), 4);
      memcpy(&cg, bx_arg(&bx_rec[0], q0, 2), 4);
      memcpy(&n, bx_arg(&bx_rec[0], q0, 5), 4);
      memcpy(&d, bx_arg(&bx_rec[0], q0, 6), 4);
      // the T = 1 kernel takes the pair body exactly there (llb_warp_q4k / _q6k); q5k / q4kr always
      same = na > 1 && ((kind != 1 && kind != 3) || (cg % 256 == 0 && d >= 9 && d <= 13));
    }
    for (u32 a = 0; a < na; a += 1) {
      const LlbGNode* q = &bx_rec[a].q[k];
      if (q->f != q0->f || q->gx != q0->gx || q->gy != q0->gy || q->bx != q0->bx || q->sm != q0->sm || q->na != q0->na) {
        err_fail("native batch: rows recorded different launches");
      }
      if (kind && same) {
        u64 w;
        u32 v[4];
        memcpy(&w, bx_arg(&bx_rec[a], q, 0), 8);
        memcpy(&v[0], bx_arg(&bx_rec[a], q, 1), 4);
        memcpy(&v[1], bx_arg(&bx_rec[a], q, 2), 4);
        memcpy(&v[2], bx_arg(&bx_rec[a], q, 5), 4);
        memcpy(&v[3], bx_arg(&bx_rec[a], q, 6), 4);
        same = w == W && v[0] == W0[0] && v[1] == cg && v[2] == n && v[3] == d;
      }
    }
    if (kind && same) {
      for (u32 a0 = 0; a0 < na; a0 += 8) {
        u32 m = na - a0 < 8 ? na - a0 : 8, ti = m <= 2 ? 0 : m <= 4 ? 1 : 2, T = 2u << ti;
        struct { u64 x[8], y[8], e[8]; } B;
        for (u32 t = 0; t < T; t += 1) {
          u32 a = a0 + (t < m ? t : 0);
          memcpy(&B.x[t], bx_arg(&bx_rec[a], &bx_rec[a].q[k], 3), 8);
          memcpy(&B.y[t], bx_arg(&bx_rec[a], &bx_rec[a].q[k], 4), 8);
          B.e[t] = 0;
          if (kind == 4) memcpy(&B.e[t], bx_arg(&bx_rec[a], &bx_rec[a].q[k], 7), 8);
          if (t >= m) B.y[t] = bx_dummy;  // a pad row: row a0's x into a scratch y
        }
        u32 rb = W0[0];
        // join: the twin runs on row a0's stream after every row's inputs, then the rows wait on it
        CUstream s0 = bx_s[act[a0]].st;
        for (u32 a = a0 + 1; a < a0 + m; a += 1) {
          if (cuEventRecord(bx_s[act[a]].ev, bx_s[act[a]].st) != CUDA_SUCCESS
            || cuStreamWaitEvent(s0, bx_s[act[a]].ev, 0) != CUDA_SUCCESS) err_fail("native batch: join failed");
        }
        void* ta[] = { &W, &rb, &cg, &B, &n, &d };
        bx_go(s0, bx_f[kind - 1][ti], q0->gx, 1, q0->bx, 0, ta);
        if (cuEventRecord(bx_s[act[a0]].ev, s0) != CUDA_SUCCESS) err_fail("native batch: join failed");
        for (u32 a = a0 + 1; a < a0 + m; a += 1) {
          if (cuStreamWaitEvent(bx_s[act[a]].st, bx_s[act[a0]].ev, 0) != CUDA_SUCCESS) err_fail("native batch: join failed");
        }
      }
      if (bx_prof == 1 && llb_sync()) bx_pt[0] += bx_now() - tp;
      continue;
    }
    for (u32 a = 0; a < na; a += 1) {
      void* args[64];
      CUDA_KERNEL_NODE_PARAMS p;
      llb_gparams(&p, &bx_rec[a].q[k], bx_rec[a].b, args);
      bx_go(bx_s[act[a]].st, p.func, p.gridDimX, p.gridDimY, p.blockDimX, p.sharedMemBytes, args);
    }
    if (bx_prof == 1 && llb_sync()) bx_pt[1] += bx_now() - tp;
  }
}

static void bx_keep(BxRec* rc) {
  if (llb_gn > 256 || llb_gb > sizeof rc->b) err_fail("native batch: a row's layer chain is too long");
  memcpy(rc->q, llb_gq, llb_gn * sizeof *llb_gq);
  memcpy(rc->b, llb_gqb, llb_gb);
  rc->n = llb_gn, rc->nb = llb_gb;
  llb_grec = 0, llb_gn = 0, llb_gb = 0;
}

// a row's (or the snapshot's) state for items of up to mx positions; row: also its x ping-pong,
// scratch, head words and attention scratch
static void bx_alloc(BxSeq* q, u32 mx, u32 nl, LlbShape sh, bool row) {
  u32 d = sh.d, cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd;
  bool grow = mx > q->kvcap;
  for (u32 i = 0; i < nl; i += 1) {
    if ((i + 1) % 4 == 0) {
      if (grow && q->kc[i]) cuMemFree(q->kc[i]), cuMemFree(q->vc[i]), q->kc[i] = q->vc[i] = 0;
      q->kc[i] = bx_need(q->kc[i], (size_t)mx * kv * 4, false);
      q->vc[i] = bx_need(q->vc[i], (size_t)mx * kv * 4, false);
    } else {
      q->b[i] = bx_need(q->b[i], 4 * (size_t)cv * 4, false);
      q->S[i] = bx_need(q->S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4, false);
    }
  }
  if (grow && q->hid) cuMemFree(q->hid), q->hid = 0;
  q->kvcap = grow ? mx : q->kvcap;
  q->hid = bx_need(q->hid, (size_t)q->kvcap * d * 4, false);
  if (row) {
    for (u32 b = 0; b < 3; b += 1) q->x[b] = bx_need(q->x[b], (size_t)d * 4, false);
    q->blk = bx_need(q->blk, 81920 * 4, false);
    q->y = bx_need(q->y, 16, false);
    q->atts = bx_need(q->atts, (size_t)2 * 24 * (1u << 17) * 4, false);  // the row's own attention scratch
  }
}

// row r takes item j (between steps, the device idle): a zeroed state at pos 0, as stage 1's
// rows began, and for j > 0 with a shared prefix the snapshot's (GDN ring + S + ring index,
// KV slots [0, L), hidden rows [0, L)) at pos L: what the item's own L steps would have left
static void bx_admit(u32 r, u32 j, u32 nl, LlbShape sh) {
  BxSeq* q = &bx_s[r];
  u32 d = sh.d, cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd, L = j > 0 ? bx_L : 0;
  bool ok = llb_zero(q->blk, 81920 * 4);
  for (u32 i = 0; ok && i < nl; i += 1) {
    if ((i + 1) % 4 == 0) {
      ok = llb_zero(q->kc[i], (size_t)q->kvcap * kv * 4) && llb_zero(q->vc[i], (size_t)q->kvcap * kv * 4)
        && (!L || (llb_copy(q->kc[i], bx_pf.kc[i], (size_t)L * kv * 4) && llb_copy(q->vc[i], bx_pf.vc[i], (size_t)L * kv * 4)));
    } else {
      ok = L ? llb_copy(q->b[i], bx_pf.b[i], 4 * (size_t)cv * 4) && llb_copy(q->S[i], bx_pf.S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4)
        : llb_zero(q->b[i], 4 * (size_t)cv * 4) && llb_zero(q->S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4);
      q->r[i] = L ? bx_pf.r[i] : 0;
    }
  }
  if (ok && L) ok = llb_copy(q->hid, bx_pf.hid, (size_t)L * d * 4);
  if (!ok) err_fail("native batch: cannot set up a row");
  bx_t[r] = L, bx_ri[r] = (int)j, bx_it[j].st = 1;
}

// at a step boundary: every free row takes the plan's next item
static void bx_fill(u32 nl, LlbShape sh) {
  bool any = false;
  for (u32 r = 0; r < bx_nr && bx_next < bx_nk; r += 1) {
    if (bx_ri[r] >= 0) continue;
    if (!any && !llb_sync()) err_fail("native batch: the device failed before an admission");
    any = true;
    bx_admit(r, bx_next++, nl, sh);
  }
  if (any && !llb_sync()) err_fail("native batch: cannot set up a row");
}

// one lockstep step over the na rows act[], each at its item's pos: hidden rows into the row's
// store; an item at its last pos also takes the head (llb_pass_run's fold: head + amax into its
// four words), then its rows and words go to the host and its row is free
static void bx_step(volatile WarpOp* o, u32 dT, LlbShape sh, u32 grid, u32 na, const u32* act) {
  u32 nl = o->d, d = sh.d, cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  u32 rbe = LLB_FM_RB(o->bb, d);
  llb_dptr blk0 = gpu_blk, atts0 = gpu_atts, cur[BX_R];
  double tp0 = bx_prof == 2 ? bx_now() : 0;
  if (bx_graph < 0) bx_graph = getenv("LLB_BATCH_GRAPH") != NULL && atoi(getenv("LLB_BATCH_GRAPH")) == 1;
  CUstream s0 = bx_s[act[0]].st;
  if (bx_graph) {
    if ((!bx_fork && cuEventCreate(&bx_fork, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
      || cuStreamBeginCapture(s0, CU_STREAM_CAPTURE_MODE_RELAXED) != CUDA_SUCCESS
      || cuEventRecord(bx_fork, s0) != CUDA_SUCCESS) err_fail("native batch: cannot capture a step");
    for (u32 a = 1; a < na; a += 1) {
      if (cuStreamWaitEvent(bx_s[act[a]].st, bx_fork, 0) != CUDA_SUCCESS) err_fail("native batch: cannot capture a step");
    }
  }
  for (u32 a = 0; a < na; a += 1) {
    BxSeq* q = &bx_s[act[a]];
    u64 er = o->r + (u64)bx_it[bx_ri[act[a]]].ids[bx_t[act[a]]] * rbe;
    u32 c4 = d;
    llb_dptr x0 = q->x[0];
    void* ea[] = { &er, &c4, &x0 };
    bx_go(q->st, o->bb == LLB_FM_Q4_K ? gpu_embedk : gpu_embed, (c4 + 255) / 256, 1, 256, 0, ea);
    cur[a] = x0;
  }
  for (u32 i = 0; i < nl; i += 1) {
    bool full = (i + 1) % 4 == 0;
    u64 base = gpu_mail->pw[i];
    u32 p[20];
    for (u32 f = 0; f < 20; f += 1) {
      p[f] = gpu_mail->pt[i][f] & 0x0fffffffu;
    }
    for (u32 a = 0; a < na; a += 1) {
      BxSeq* q = &bx_s[act[a]];
      u32 c = dT | bx_t[act[a]] << 5;
      gpu_blk = q->blk, gpu_atts = q->atts;
      llb_grec = 1, llb_gn = 0, llb_gb = 0;
      WarpOp at = {0};
      at.w = base; at.x = cur[a]; at.brg = o->brg; at.bro = o->bro; at.ban = p[0];
      if (full) {
        at.k = q->kc[i]; at.v = q->vc[i];
        at.bq = p[14]; at.bg = p[15]; at.bb = p[16];
        at.ba = p[17]; at.bsa = p[18]; at.bdt = p[19];
        at.c = c; at.fm = 1;
        llb_fulb_run(&at, q->x[2], grid, gpu_mail->pt[i]);
      } else {
        u32 rr = q->r[i];
        q->r[i] = rr + 1;
        at.k  = q->b[i] + (u64)(rr % 4) * cv * 4;
        at.v  = q->b[i] + (u64)((rr + 1) % 4) * cv * 4;
        at.wn = q->b[i] + (u64)((rr + 2) % 4) * cv * 4;
        at.qk = q->b[i] + (u64)((rr + 3) % 4) * cv * 4;
        at.r  = q->S[i];
        at.c = p[5]; at.bq = p[6]; at.bg = p[7]; at.bb = p[8]; at.ba = p[9];
        at.bsa = p[10]; at.bdt = p[11]; at.bcv = p[12]; at.bsn = p[13];
        llb_linb_run(&at, q->x[2], grid, gpu_mail->pt[i]);
      }
      WarpOp ff = {0};
      ff.w = base + p[1]; ff.k = base + p[2]; ff.v = base + p[3]; ff.wn = base + p[4];
      ff.x = cur[a]; ff.r = q->x[2]; ff.n = sh.d; ff.rb = o->brg; ff.d = o->rb; ff.fm = 1;
      llb_dptr nxt = q->x[cur[a] == q->x[0] ? 1 : 0];
      llb_fbn_run(&ff, nxt, grid, gpu_mail->pt[i]);
      bx_keep(&bx_rec[a]);
      cur[a] = nxt;
    }
    gpu_blk = blk0, gpu_atts = atts0;
    bx_issue(na, act);
  }
  for (u32 a = 0; a < na; a += 1) {
    BxSeq* q = &bx_s[act[a]];
    u64 wn = o->k, op2 = q->hid + (u64)bx_t[act[a]] * d * 4;
    u32 n4 = d;
    void* na4[] = { &cur[a], &wn, &op2, &n4 };
    bx_go(q->st, gpu_rnorm, 1, 1, 1024, 0, na4);
  }
  if (bx_graph) {
    CUgraph g = NULL;
    CUgraphExecUpdateResultInfo ui;
    for (u32 a = 1; a < na; a += 1) {
      if (cuEventRecord(bx_s[act[a]].ev, bx_s[act[a]].st) != CUDA_SUCCESS
        || cuStreamWaitEvent(s0, bx_s[act[a]].ev, 0) != CUDA_SUCCESS) err_fail("native batch: cannot capture a step");
    }
    if (cuStreamEndCapture(s0, &g) != CUDA_SUCCESS) err_fail("native batch: cannot capture a step");
    if (bx_ge && cuGraphExecUpdate(bx_ge, g, &ui) == CUDA_SUCCESS) {
      bx_gstat[1] += 1;
    } else {
      if (bx_ge) cuGraphExecDestroy(bx_ge);
      if (cuGraphInstantiate(&bx_ge, g, 0) != CUDA_SUCCESS) err_fail("native batch: cannot instantiate a step");
      bx_gstat[0] += 1;
    }
    if (cuGraphLaunch(bx_ge, s0) != CUDA_SUCCESS || cuGraphDestroy(g) != CUDA_SUCCESS) err_fail("native batch: graph launch failed");
  }
  bool fin = false;
  for (u32 a = 0; a < na; a += 1) {
    BxSeq* q = &bx_s[act[a]];
    u32 t = bx_t[act[a]];
    u64 wo = o->v, op2 = q->hid + (u64)t * d * 4, lg = gpu_logit, yp = q->y;
    u32 n4 = d, nh = o->gm, rq = o->bq, cg = o->bq / 17 * 4, dh = llb_cdepth(d);
    bool q8 = o->bg == LLB_FM_Q8_0;
    if (t + 1 == bx_it[bx_ri[act[a]]].n) {  // the last position: llb_pass_run's head chain (null stream, after the row)
      if (cuStreamSynchronize(bx_graph ? s0 : q->st) != CUDA_SUCCESS) err_fail("native batch: a row failed on the device");
      if ((q8 && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &wo, &rq, &cg, &op2, &lg, &nh, &dh))) {
        err_fail("device launch failed");
      }
      if (!q8) {
        llb_kgemv(o->bg, wo, n4, op2, lg, nh, grid);
      }
      if (!LLB_LAUNCH(gpu_amaxp, AMAX_G, 1, AMAX_T, &lg, &nh, &yp)
        || !LLB_LAUNCH(gpu_amax, 1, 1, AMAX_G, &lg, &nh, &yp)) {
        err_fail("device launch failed");
      }
      fin = true;
    }
  }
  bx_tok += na, bx_steps += 1;
  if (bx_prof == 2) {
    bx_pt[0] += bx_now() - tp0;
    if (!llb_sync()) err_fail("native batch: a row failed on the device");
    bx_pt[1] += bx_now() - tp0;
  }
  if (fin && !llb_sync()) err_fail("native batch: a row failed on the device");
  for (u32 a = 0; a < na; a += 1) {
    u32 r = act[a];
    BxItem* it = &bx_it[bx_ri[r]];
    bx_t[r] += 1;
    if (bx_t[r] < it->n) continue;
    if ((size_t)it->n * d > it->hcap) {
      it->hcap = (size_t)it->n * d;
      it->hh = realloc(it->hh, it->hcap * 4);
      if (!it->hh) err_fail("native batch: cannot allocate an item's hidden rows");
    }
    if (!llb_read(it->hh, bx_s[r].hid, (size_t)it->n * d * 4) || !llb_read(it->y, bx_s[r].y, 16)) {
      err_fail("native batch: cannot read a done row");
    }
    it->st = 2, bx_ri[r] = -1;
  }
}

static double bx_since(const struct timespec* t0) {
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  return (t1.tv_sec - t0->tv_sec) * 1e3 + (t1.tv_nsec - t0->tv_nsec) * 1e-6;
}

// the plan's first request: rows sized for its longest item, item 0 on row 0; with a shared
// prefix (LLB_BATCH_PREFIX, default on; 0 = off: the longest prefix every item's ids share,
// capped so each keeps its last step) row 0 runs it alone and its state at L is the snapshot
// every other item starts from (equal ids on an equal state give equal bits); then every row
// takes an item
static void bx_begin(volatile WarpOp* o, LlbShape sh) {
  struct timespec ts0;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (!bx_f[0][0]) bx_load();
  if (!llb_sync()) err_fail("native batch: the device failed before the batch");
  u32 nl = o->d, mx = 0;
  const char* pe = getenv("LLB_BATCH_PREFIX");
  u32 L = pe != NULL && atoi(pe) == 0 ? 0 : ~0u;
  for (u32 j = 0; j < bx_nk; j += 1) {
    u32 m = 0, lim = bx_it[j].n < bx_it[0].n ? bx_it[j].n : bx_it[0].n;
    while (m < lim && bx_it[j].ids[m] == bx_it[0].ids[m]) m += 1;
    L = m < L ? m : L;
    L = bx_it[j].n - 1 < L ? bx_it[j].n - 1 : L;
    mx = bx_it[j].n > mx ? bx_it[j].n : mx;
  }
  bx_L = L;
  bx_nr = bx_nk < (u32)bx_max ? bx_nk : (u32)bx_max;
  for (u32 r = 0; r < bx_nr; r += 1) bx_alloc(&bx_s[r], mx, nl, sh, true), bx_ri[r] = -1;
  if (L > 0) bx_alloc(&bx_pf, L, nl, sh, false);
  bx_next = 0, bx_tok = 0, bx_steps = 0, bx_live = true;
  memcpy(&bx_o, (const void*)o, sizeof bx_o);
  bx_sh = sh;
  bx_admit(0, bx_next++, nl, sh);
  if (!llb_sync()) err_fail("native batch: cannot set up a row");
  bx_ms = bx_since(&ts0);
}

// the second half of bx_begin, inside the step loop: at pos L row 0's state becomes the snapshot
static void bx_snap(u32 nl, LlbShape sh) {
  u32 d = sh.d, cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd, L = bx_L;
  BxSeq* q0 = &bx_s[0];
  if (!llb_sync()) err_fail("native batch: the shared prefix failed on the device");
  bool ok = llb_copy(bx_pf.hid, q0->hid, (size_t)L * d * 4);
  for (u32 i = 0; ok && i < nl; i += 1) {
    if ((i + 1) % 4 == 0) {
      ok = llb_copy(bx_pf.kc[i], q0->kc[i], (size_t)L * kv * 4) && llb_copy(bx_pf.vc[i], q0->vc[i], (size_t)L * kv * 4);
    } else {
      ok = llb_copy(bx_pf.b[i], q0->b[i], 4 * (size_t)cv * 4) && llb_copy(bx_pf.S[i], q0->S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4);
      bx_pf.r[i] = q0->r[i];
    }
  }
  if (!ok || !llb_sync()) err_fail("native batch: cannot snapshot the shared prefix");
}

// one lockstep step (the prefix's snapshot and the fill around it); between steps free rows take
// the next items
static void bx_advance(volatile WarpOp* o, u32 dT, LlbShape sh, u32 grid) {
  u32 nl = o->d;
  {
    u32 act[BX_R], na = 0;
    if (bx_next == 1 && bx_t[0] == bx_L) {  // the prefix is done (or empty): snapshot, then fill
      if (bx_L > 0 && bx_nk > 1) bx_snap(nl, sh);
      bx_fill(nl, sh);
    }
    for (u32 r = 0; r < bx_nr; r += 1) {
      if (bx_ri[r] >= 0) act[na++] = r;
    }
    if (na == 0) err_fail("native batch: a planned item never ran");
    bx_step(o, dT, sh, grid, na, act);
    if (bx_next > 1 || bx_nk == 1) bx_fill(nl, sh);
  }
}

// every item on the host: the plan's line, and its rows may be reused by the next plan
static void bx_end(void) {
  bool all = bx_next == bx_nk;
  for (u32 r = 0; all && r < bx_nr; r += 1) all = bx_ri[r] < 0;
  if (all && bx_live) {
    u32 tok = 0;
    for (u32 k = 0; k < bx_nk; k += 1) tok += bx_it[k].n;
    fprintf(stderr, "native batch: %u sequences, %u tokens, %u steps (shared prefix %u, %u rows, %u row-steps), %.1f ms\n",
      bx_nk, tok, bx_steps, bx_L, bx_nr, bx_tok, bx_ms);
    if (bx_prof == 1) fprintf(stderr, "native batch prof: twin GEMVs %.1f ms, row-by-row %.1f ms\n", bx_pt[0], bx_pt[1]);
    if (bx_prof == 2) fprintf(stderr, "native batch prof2: host issue %.1f ms, step to sync %.1f ms (%u steps)\n", bx_pt[0], bx_pt[1], bx_steps);
    if (bx_graph > 0) fprintf(stderr, "native batch graph: %u instantiations, %u in-place updates\n", bx_gstat[0], bx_gstat[1]), bx_gstat[0] = bx_gstat[1] = 0;
    bx_pt[0] = bx_pt[1] = 0;
    bx_live = false;
  }
}

// run the lockstep until item j is done
static void bx_until(volatile WarpOp* o, u32 dT, LlbShape sh, u32 grid, u32 j) {
  struct timespec ts0;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  bx_dT = dT, bx_grid = grid;
  while (bx_it[j].st < 2) bx_advance(o, dT, sh, grid);
  bx_ms += bx_since(&ts0);
  bx_end();
}

// stage 3b (dual.c, LLB_BATCH_OOO=1): the plan item to answer next: one already done and not yet
// served, else the lockstep runs until one is; -1 = the plan has not started (its first request
// starts it) or every item is served
int llb_bx_next_done(void) {
  if (bx_nk == 0 || (!bx_live && bx_next == 0)) return -1;
  struct timespec ts0;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  int got = -1;
  for (;;) {
    for (u32 j = 0; got < 0 && j < bx_nk; j += 1) {
      if (bx_it[j].st == 2) got = (int)j;
    }
    bool run = false;
    for (u32 r = 0; r < bx_nr; r += 1) run = run || bx_ri[r] >= 0;
    if (got >= 0 || (!run && bx_next >= bx_nk)) break;
    bx_advance(&bx_o, bx_dT, bx_sh, bx_grid);
  }
  bx_ms += bx_since(&ts0);
  bx_end();
  return got;
}

// llb_hidden_put for rows pos0 .. pos0 + n - 1 of an item's host rows (one file open; the same bytes)
static void bx_hidden(u32 pos0, u32 n, u32 width, const float* rows) {
  if (!llb_hidden_path) return;
  FILE* f = NULL;
  for (u32 j = 0; j < n; j += 1) {
    u32 pos = pos0 + j;
    if (pos >= llb_hidden_tokens || !llb_hidden_positions[pos]) continue;
    u32 header[2] = { pos, width };
    if (!f) f = fopen(llb_hidden_path, llb_hidden_written ? "ab" : "wb");
    if (!f || fwrite(header, 4, 2, f) != 2 || fwrite(rows + (size_t)j * width, 4, width, f) != width)
      err_fail("LLB_DUMP_HIDDEN: write failed");
    llb_hidden_written = true;
  }
  if (f && fclose(f)) err_fail("LLB_DUMP_HIDDEN: close failed");
}

// llb_pass_run's folded-step hook: true = this step was served from the batch
static bool bx_take(volatile WarpOp* o, u32 pos, u32 dT, LlbShape sh, u64 yp, u32 grid) {
  if (bx_cur < 0) {
    return false;
  }
  BxItem* q = &bx_it[bx_cur];
  if (q->st < 2) {
    // an item runs at its request's first step, on the whole-model folded pass only; one that
    // cannot is not planned (before the plan's rows start: no plan) and runs as a single
    if (pos != 0 || o->n != 0 || o->bsn != 1 || spec_k >= 0 || llb_dump || smp_on() || llb_hidden_path == NULL) {
      if (!bx_live && bx_next == 0) bx_nk = 0;
      bx_cur = -1;
      return false;
    }
    if (!bx_live) bx_begin(o, sh);
    bx_until(o, dT, sh, grid, (u32)bx_cur);
  }
  if (o->bsn >= 2) {
    static float ids[4096];
    u32 n = o->bsn - 2;
    if (n > 4096 || !llb_read(ids, o->x, (size_t)n * 4)) err_fail("native batch: cannot read the batch's ids");
    for (u32 j = 0; j < n; j += 1) {
      if (pos + j >= q->n || (u32)ids[j] != q->ids[pos + j]) err_fail("native batch: a served step differs from its plan");
    }
    bx_hidden(pos, n, sh.d, q->hh + (size_t)pos * sh.d);
    return true;
  }
  u32 rb = LLB_FM_RB(o->bb, sh.d);
  if (pos >= q->n || (u32)((o->w - o->r) / rb) != q->ids[pos]) err_fail("native batch: a served step differs from its plan");
  bx_hidden(pos, 1, sh.d, q->hh + (size_t)pos * sh.d);
  if (o->bsn == 1) {
    // the fast tier's cue: the loop sends the rest but the last as one batch op
    if (!LLB_LAUNCH(gpu_pfmark, 1, 1, 1, &yp)) err_fail("device launch failed");
  } else {
    if (pos + 1 != q->n || cuMemcpyHtoD(yp, q->y, 16) != CUDA_SUCCESS) err_fail("native batch: the last step is not the plan's");
    q->st = 3, bx_cur = -1;  // served
  }
  return true;
}

// tools/payload-anyshape.patch: a device scratch buffer at least n bytes (zeroed when grown:
// the GEMV bodies read x's pad tail to the next power of two, so every pad must be finite)
static void llb_scratch(CUdeviceptr* p, size_t* have, size_t n, const char* what) {
  if (n <= *have) {
    return;
  }
  llb_graph_bail();
  if (*p != 0 && (!llb_sync() || cuMemFree(*p) != CUDA_SUCCESS)) {
    err_fail("warp pass: cannot free a scratch buffer");
  }
  *p = llb_alloc_zero(n);
  if (!*p) {
    fprintf(stderr, "warp pass: cannot allocate %zu bytes of %s scratch\n", n, what);
    err_fail("warp pass: out of device memory for the scratch");
  }
  *have = n;
}

static u64 llb_p2(u64 n) {
  u64 p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

// the shapes LLB_SPEC / LLB_MTP / LLB_BATCH were gated on (the 9B and 27B: d_model 4096 /
// 5120, 4 kv heads, GQA 4 or 6, lvh 2 or 3 lkh, full_attention_interval 4, 2^17 score
// planes): llb_t_attn4's 1024-float kv stride, the spec embed's rows and the bx_* scratch
// are sized for them
static bool llb_legacy(LlbShape sh, u32 dT, u32 nl) {
  u32 qa = sh.nh * sh.hd, lv = sh.lvh * sh.sd;
  bool ok = (sh.d == 4096 || sh.d == 5120) && sh.f > 8192 && sh.f <= 17408 && sh.hd == 256 && sh.sd == 128
    && (sh.nh == 4 * sh.nkv || sh.nh == 6 * sh.nkv) && sh.nkv == 4 && sh.lkh * sh.sd == 2048
    && (sh.lvh == 2 * sh.lkh || sh.lvh == 3 * sh.lkh) && qa == lv && dT <= 17 && nl <= 64;
  for (u32 i = 0; ok && i < nl; i += 1) {
    ok = llb_full(i) == ((i + 1) % 4 == 0) && gpu_mail->pt[i][26] == 0;  // a MoE layer (payload-moeany): not these shapes
  }
  return ok;
}

// the first pass: the config words to the device (llb_cfg: eps, rope base, rope dims), the
// scratch sized from the shape, one stderr line naming both
static void llb_pass_init(LlbShape sh, u32 dT, u32 nl) {
  u32 cf[4] = { gpu_mail->cfg[0], gpu_mail->cfg[1], gpu_mail->cfg[2], gpu_mail->cfg[3] };
  CUdeviceptr cp;
  size_t cz;
  if (cuModuleGetGlobal(&cp, &cz, gpu_lib, "llb_cfg") != CUDA_SUCCESS || cz != 16 || !llb_write(cp, cf, 16)) {
    err_fail("warp pass: cannot set the model config on the device");
  }
  pass_cap = cf[3];
  u64 d = sh.d, f = sh.f, qa = (u64)sh.nh * sh.hd, kv = (u64)sh.nkv * sh.hd, lv = (u64)sh.lvh * sh.sd;
  u64 lvh = sh.lvh, cv = 2 * (u64)sh.lkh * sh.sd + lv, e18 = 4 * lv + 4 * lvh + cv, hb = e18 > 39040 ? e18 : 39040;
  // gpu_blk (floats): fbn x1 h2 g u silu (+ silu's pad), linb's rows to normed and hb18 (+ pads),
  // fulb's rows (+ h22's and gated22's pads)
  u64 nb = 2 * d + 3 * f;
  nb = nb > 2 * d + 2 * f + llb_p2(f) ? nb : 2 * d + 2 * f + llb_p2(f);
  nb = nb > d + llb_p2(d) ? nb : d + llb_p2(d);
  nb = nb > hb + llb_p2(d) ? nb : hb + llb_p2(d);
  nb = nb > e18 + llb_p2(lv) ? nb : e18 + llb_p2(lv);
  nb = nb > e18 + lv ? nb : e18 + lv;
  nb = nb > 6 * qa + d + 3 * kv ? nb : 6 * qa + d + 3 * kv;
  nb = nb > 2 * qa + llb_p2(d) ? nb : 2 * qa + llb_p2(d);
  nb = nb > qa + llb_p2(qa) ? nb : qa + llb_p2(qa);
  u64 mc = d > f ? d : f;
  mc = mc > qa ? mc : qa;
  mc = mc > lv ? mc : lv;
  static size_t hblk = 81920 * 4, hq8b = 32768, hq8c = 40960, hhead = 8192 * 4, hatts = (size_t)2 * 24 * (1u << 17) * 4;
  llb_scratch(&gpu_blk, &hblk, nb * 4, "gpu_blk");
  llb_scratch(&gpu_q8b, &hq8b, mc + mc / 32 * 4 + 256, "gpu_q8b");
  llb_scratch(&gpu_q8c, &hq8c, 2 * mc + mc / 32 * 4 + 256, "gpu_q8c");
  llb_scratch(&gpu_headh, &hhead, llb_p2(d) * 4, "gpu_headh");
  llb_scratch(&gpu_atts, &hatts, (size_t)2 * sh.nh * ((size_t)1 << (dT > 17 ? dT : 17)) * 4, "gpu_atts");
  union { u32 u; float f; } ep = { cf[0] }, rb = { cf[1] };
  fprintf(stderr, "shape d_model %u d_ff %u heads %u kv %u layers %u lin %u/%u sd %u eps %g rope %g/%u kv_cap %u (dT %u)%s\n",
    sh.d, sh.f, sh.nh, sh.nkv, nl, sh.lkh, sh.lvh, sh.sd, (double)ep.f, (double)rb.f, cf[2], cf[3], dT,
    llb_legacy(sh, dT, nl) ? "" : " [generic shape: spec / batch off]");
}

static void pass_fold(WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, bool dev);
// LLB_STEP_US (reports/cuda-moe-speed): `stepus POS US` on stderr, the microseconds since the
// previous folded step began (the whole token period: the Bend loop, the pass and the head);
// timing only, nothing else changes. The decode run's steps print it too (reports/cuda-catchup)
static void step_us(u32 pos) {
  static int on = -1;
  static struct timespec sp;
  if (on < 0) {
    on = getenv("LLB_STEP_US") != NULL;
  }
  if (!on) {
    return;
  }
  struct timespec tn;
  clock_gettime(CLOCK_MONOTONIC, &tn);
  if (sp.tv_sec != 0) {
    fprintf(stderr, "stepus %u %lld\n", pos,
      (long long)(tn.tv_sec - sp.tv_sec) * 1000000 + (tn.tv_nsec - sp.tv_nsec) / 1000);
  }
  sp = tn;
}
static bool gen_ok(volatile WarpOp* o);
static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n);

static void llb_pass_run(volatile WarpOp* o, u64 yp, u32 grid) {
  // reports/cuda-catchup: the decode run (pass_mail: fl word 47 bit 28 with the greedy steps
  // left, o->bsn bit 31 | n); taken off the op first, so every other path sees the plain step
  u32 gen = o->bsn >> 31 ? o->bsn & 0x7fffffffu : 0;
  if (gen != 0) {
    o->bsn = 0;
  }
  u32 c = o->c, dT = c & 31, pos = c >> 5, s = o->n, k = o->d;
  // LLB_STEP_US (reports/cuda-catchup amendment 1): `pfus US` at each prompt's first folded step,
  // the microseconds since the previous prompt's (a multi-prompt run's per-prompt period: its
  // prefill in either tier, the last step and the reset); timing only
  if (o->w != 0 && pos == 0 && getenv("LLB_STEP_US") != NULL) {
    static struct timespec pp;
    struct timespec tn;
    clock_gettime(CLOCK_MONOTONIC, &tn);
    if (pp.tv_sec != 0) {
      fprintf(stderr, "pfus %lld\n", (long long)(tn.tv_sec - pp.tv_sec) * 1000000 + (tn.tv_nsec - pp.tv_nsec) / 1000);
    }
    pp = tn;
  }
  volatile LlbShape* ms = &gpu_mail->sh;
  LlbShape sh = { ms->d, ms->f, ms->nh, ms->nkv, ms->hd, ms->lkh, ms->lvh, ms->sd };
  if (pass_x[0] == 0) {
    llb_sh = sh;
    for (u32 b = 0; b < 3; b += 1) {
      pass_x[b] = pass_zero(sh.d);
    }
    pass_dT = dT;
    llb_pass_init(sh, dT, s + k);
    // reports/cuda-moe-speed: a MoE file records each folded step as a CUDA graph (LLB_GRAPH's
    // replay: the same kernels, args and order) unless LLB_GRAPH is set; dense files unchanged
    for (u32 i = 0; i < s + k && getenv("LLB_GRAPH") == NULL; i += 1) {
      if (llb_moe(i)) {
        setenv("LLB_GRAPH", "1", 0);
      }
    }
  }
  if (dT != pass_dT) {
    err_fail("warp pass: dT changed");
  }
  if (memcmp(&sh, &llb_sh, sizeof sh) != 0) {
    err_fail("warp pass: the shape record changed");
  }
  if (pos >= pass_cap) {
    err_fail("warp pass: the position is past the KV capacity (prompt + steps); raise --steps");
  }
  // LLB_SPEC / LLB_MTP / LLB_BATCH keep the shapes they were gated on (llb_legacy; the 0.8B,
  // d_model 1024, was refused here before tools/payload-anyshape.patch too)
  if ((spec_k >= 0 || spec_mtp || llb_bx_max() >= 2) && !llb_legacy(sh, dT, s + k)) {
    err_fail("LLB_SPEC / LLB_MTP / LLB_BATCH: gated on the 9B / 27B shapes only (d_model 4096 / 5120, 4 kv heads, "
      "full_attention_interval 4, context <= 131072); run without them");
  }
  llb_hidden_init();
  if (llb_hidden_path && !o->w) err_fail("LLB_DUMP_HIDDEN: folded pass required");
  if (spec_k >= 0) {
    if (o->w == 0 || o->n != 0 || llb_dump) {
      err_fail("LLB_SPEC: needs the folded whole-model pass (one op a token) and no LLB_DUMP");
    }
    if (!spec_hist) {
      spec_init(o, sh, dT, o->d);
    }
    if (!pf_take(o, pos, dT, sh, yp)) {
      spec_pass(o, yp, sh, dT, pos);
    }
    return;
  }
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  if (pos == 0 && s == 0) {
    llb_dump_k += 1;
  }
  u64 cur = o->x;
  bool fold = o->w != 0;
  if (fold && o->bsn < 2) {
    smp_note(pos, (u32)((o->w - o->r) / LLB_FM_RB(o->bb, sh.d)));
  }
  if (fold && bx_take(o, pos, dT, sh, yp, grid)) {  // BATCH-NATIVE (LLB_BATCH)
    return;
  }
  if (fold && pf_take(o, pos, dT, sh, yp)) {
    return;
  }
  if (fold) {
    step_us(pos);
  }
  if (fold) {
    if (gen >= 2 && gen_ok(o)) {
      gen_run(o, yp, grid, sh, dT, pos, gen);
    } else {
      WarpOp op = *(WarpOp*)o;
      pass_fold(&op, yp, grid, sh, dT, pos, false);
    }
    return;
  }
  for (u32 j = 0; j < k; j += 1) {
    u32  i = s + j;
    bool full = llb_full(i);
    pass_layer(i, full, pos, dT, sh);
    u64 base = gpu_mail->pw[j];
    u32 p[20];
    for (u32 f = 0; f < 20; f += 1) {
      p[f] = gpu_mail->pt[i][f] & 0x0fffffffu;
    }
    WarpOp at = {0};
    at.w = base; at.x = cur; at.brg = o->brg; at.bro = o->bro; at.ban = p[0];
    if (full) {
      at.k = pass_kc[i]; at.v = pass_vc[i];
      at.bq = p[14]; at.bg = p[15]; at.bb = p[16];
      at.ba = p[17]; at.bsa = p[18]; at.bdt = p[19];
      at.c = c; at.fm = 1;
      llb_fulb_run(&at, pass_x[2], grid, gpu_mail->pt[i]);
    } else {
      u32 r = pass_r[i];
      pass_r[i] = r + 1;
      at.k  = pass_b[i] + (u64)(r % 4) * cv * 4;
      at.v  = pass_b[i] + (u64)((r + 1) % 4) * cv * 4;
      at.wn = pass_b[i] + (u64)((r + 2) % 4) * cv * 4;
      at.qk = pass_b[i] + (u64)((r + 3) % 4) * cv * 4;
      at.r  = pass_S[i];
      at.c = p[5]; at.bq = p[6]; at.bg = p[7]; at.bb = p[8]; at.ba = p[9];
      at.bsa = p[10]; at.bdt = p[11]; at.bcv = p[12]; at.bsn = p[13];
      llb_linb_run(&at, pass_x[2], grid, gpu_mail->pt[i]);
    }
    WarpOp ff = {0};
    ff.w = base + p[1]; ff.k = base + p[2]; ff.v = base + p[3]; ff.wn = base + p[4];
    ff.x = cur; ff.r = pass_x[2]; ff.n = sh.d; ff.rb = o->brg; ff.d = o->rb; ff.fm = 1;
    u64 nxt = j + 1 == k ? yp : pass_x[cur == pass_x[0] ? 1 : 0];
    if (llb_moe(i)) {  // tools/payload-moeany.patch
      ff.qk = base;
      llb_moe_run(&ff, nxt, grid, gpu_mail->pt[i], i);
    } else {
      llb_fbn_run(&ff, nxt, grid, gpu_mail->pt[i]);
    }
    if (llb_dump && llb_dump_pos >= 0) {
      llb_dptr hp[2] = { pass_x[2], nxt };
      u32 hn[2] = { sh.d, sh.d };
      llb_dump_put("h", pos, -1, i == 0 ? "wb" : "ab", hp, hn, 2);
      if ((int)pos == llb_dump_pos) {
        llb_dptr fp[6] = { gpu_blk, gpu_blk + sh.d * 4, gpu_blk + 2 * sh.d * 4,
          gpu_blk + (2 * sh.d + sh.f) * 4, gpu_blk + (2 * sh.d + 2 * sh.f) * 4, nxt };
        u32 fn[6] = { sh.d, sh.d, sh.f, sh.f, sh.f, sh.d };
        llb_dump_put("f", pos, (int)i, "wb", fp, fn, 6);
      }
    }
    cur = nxt;
  }
}

// One folded step at pos (o: the op's copy): the embedding row, every layer, rms_final, the head
// GEMV and llb_amax's four words into yp -- the launches llb_pass_run made inline before
// reports/cuda-catchup. dev: the decode run's later steps, whose token is the device's previous
// pick (yp word 1): llb_embt reads it there, so the host queues the step before it knows the id.
static void pass_fold(WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, bool dev) {
  u32 s = o->n, k = o->d, c = dT | pos << 5;
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  llb_graph_begin();
  // the folded step: the token's embedding row into pass_x[0] (o->bb its
  // format, pass_mail admits Q8_0 and Q4_K; Q6_K the 0.8B's, tools/payload-w1024.patch)
  u64 er = o->w, x0 = pass_x[0], cur;
  u32 c4 = sh.d;
#if CUDA_VERSION >= 13000
  // reports/cuda-catchup: the embedding table lives in the corpus (unified memory) and a table
  // under 16 MiB is read in by a host memcpy, so each token's row first faulted its page over
  // (~40 us a new token): move the whole table to the device once (a hint; no bytes change)
  static u64 epf;
  if (epf != o->r) {
    epf = o->r;
    CUmemLocation lc = { CU_MEM_LOCATION_TYPE_DEVICE, gpu_dev };
    cuMemPrefetchAsync((CUdeviceptr)o->r, (size_t)o->gm * LLB_FM_RB(o->bb, sh.d), lc, 0, NULL);
  }
#endif
  if (dev) {
    u64 eb = o->r, ty = yp;
    u32 rb = LLB_FM_RB(o->bb, sh.d), fm = o->bb;
    if (!LLB_LAUNCH(gpu_embt, (c4 + 255) / 256, 1, 256, &eb, &rb, &fm, &ty, &c4, &x0)) {
      err_fail("device launch failed");
    }
  // (G4 F1, FUS-V46 V5) an <|image_pad|> slot takes its image row instead
  } else if (!llb_img_step((u32)((o->w - o->r) / LLB_FM_RB(o->bb, sh.d)), pos, x0, c4)
    && !LLB_LAUNCH(o->bb == LLB_FM_Q4_K ? gpu_embedk : o->bb == LLB_FM_Q6_K ? gpu_embed6 : gpu_embed, (c4 + 255) / 256, 1, 256, &er, &c4, &x0)) {
    err_fail("device launch failed");
  }
  if (!dev) {
    llb_img_seen_row(pos, x0, c4);
  }
  cur = x0;
  for (u32 j = 0; j < k; j += 1) {
    u32  i = s + j;
    bool full = llb_full(i);
    pass_layer(i, full, pos, dT, sh);
    u64 base = gpu_mail->pw[j];
    // the layer's table row (tools/payload-table.patch): offset | fm << 28
    u32 p[20];
    for (u32 f = 0; f < 20; f += 1) {
      p[f] = gpu_mail->pt[i][f] & 0x0fffffffu;
    }
    WarpOp at = {0};
    at.w = base; at.x = cur; at.brg = o->brg; at.bro = o->bro; at.ban = p[0];
    if (full) {
      at.k = pass_kc[i]; at.v = pass_vc[i];
      at.bq = p[14]; at.bg = p[15]; at.bb = p[16];
      at.ba = p[17]; at.bsa = p[18]; at.bdt = p[19];
      at.c = c; at.fm = 1;
      llb_fulb_run(&at, pass_x[2], grid, gpu_mail->pt[i]);
    } else {
      u32 r = pass_r[i];
      pass_r[i] = r + 1;
      at.k  = pass_b[i] + (u64)(r % 4) * cv * 4;
      at.v  = pass_b[i] + (u64)((r + 1) % 4) * cv * 4;
      at.wn = pass_b[i] + (u64)((r + 2) % 4) * cv * 4;
      at.qk = pass_b[i] + (u64)((r + 3) % 4) * cv * 4;
      at.r  = pass_S[i];
      at.c = p[5]; at.bq = p[6]; at.bg = p[7]; at.bb = p[8]; at.ba = p[9];
      at.bsa = p[10]; at.bdt = p[11]; at.bcv = p[12]; at.bsn = p[13];
      llb_linb_run(&at, pass_x[2], grid, gpu_mail->pt[i]);
    }
    WarpOp ff = {0};
    ff.w = base + p[1]; ff.k = base + p[2]; ff.v = base + p[3]; ff.wn = base + p[4];
    ff.x = cur; ff.r = pass_x[2]; ff.n = sh.d; ff.rb = o->brg; ff.d = o->rb; ff.fm = 1;
    u64 nxt = pass_x[cur == pass_x[0] ? 1 : 0];
    if (llb_moe(i)) {  // tools/payload-moeany.patch
      ff.qk = base;
      llb_moe_run(&ff, nxt, grid, gpu_mail->pt[i], i);
    } else {
      llb_fbn_run(&ff, nxt, grid, gpu_mail->pt[i]);
    }
    // the per-layer h's only with LLB_DUMP_POS set (K3's logit-only runs skip them)
    if (llb_dump && llb_dump_pos >= 0) {
      llb_dptr hp[2] = { pass_x[2], nxt };
      u32 hn[2] = { sh.d, sh.d };
      llb_dump_put("h", pos, -1, i == 0 ? "wb" : "ab", hp, hn, 2);
      if ((int)pos == llb_dump_pos) {
        // fbn's gpu_blk layout: x1, h2 (d each), g, u, silu (f each)
        llb_dptr fp[6] = { gpu_blk, gpu_blk + sh.d * 4, gpu_blk + 2 * sh.d * 4,
          gpu_blk + (2 * sh.d + sh.f) * 4, gpu_blk + (2 * sh.d + 2 * sh.f) * 4, nxt };
        u32 fn[6] = { sh.d, sh.d, sh.f, sh.f, sh.f, sh.d };
        llb_dump_put("f", pos, (int)i, "wb", fp, fn, 6);
      }
    }
    cur = nxt;
  }
  // the head bang's chain (kind 14): rms_final -> gpu_headh, the head GEMV
  // (o->bg its format) into gpu_logit, llb_amax's four words into y
  u64 wn = o->k, wo = o->v, op2 = gpu_headh, lg = gpu_logit;
  u32 n4 = sh.d, nh = o->gm, rq = o->bq, cg = o->bq / 17 * 4, dh = llb_cdepth(sh.d);
  bool q8 = o->bg == LLB_FM_Q8_0 && llb_tuned(LLB_FM_Q8_0, sh.d);
  if (!LLB_LAUNCH(gpu_rnorm, 1, 1, 1024, &cur, &wn, &op2, &n4)
    || (q8 && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &wo, &rq, &cg, &op2, &lg, &nh, &dh))) {
    err_fail("device launch failed");
  }
  llb_hidden_put(pos, n4, op2);
  if (!q8) {
    llb_kgemv(o->bg, wo, n4, op2, lg, nh, grid);
  }
  if (!LLB_LAUNCH(gpu_amaxp, AMAX_G, 1, AMAX_T, &lg, &nh, &yp)
    || !LLB_LAUNCH(gpu_amax, 1, 1, AMAX_G, &lg, &nh, &yp)) {
    err_fail("device launch failed");
  }
  if (smp_on()) {
    smp_pick(yp, lg, nh, pos, -1);
  }
  if (llb_dump) {
    llb_dptr lgp = lg;
    llb_dump_put("l", pos, -1, "wb", &lgp, &nh, 1);
  }
  llb_graph_end();
}

// reports/cuda-catchup: the decode run. gen's bang asks for up to n greedy steps (pos, pos + 1,
// ...); the host serves them back to back -- every step the folded step above, the next one's
// token the device's pick -- so the Bend loop (its host evaluation, unified-memory page traffic
// and device interpreter rounds, ~0.8 ms a token) runs once a run instead of once a token. It
// prints what gen would have printed for every step but the last (stdout: the ids; stderr: the
// logits and step lines, stdout flushed first as io_errs does), and leaves the last step's four
// words in y with the count in word 4; gen prints the rest. Only plain greedy decode takes it:
// any feature that reads or changes the token on the host keeps one step a bang.
// LLB_GEN=0 turns it off (one step a bang, as before).
static bool gen_ok(volatile WarpOp* o) {
  static int env = -1;
  if (env < 0) {
    const char* e = getenv("LLB_GEN");
    env = e == NULL || atoi(e) != 0;
  }
  return env && !smp_on() && !gr_on() && spec_k < 0 && !spec_mtp && llb_bx_max() < 2 && !llb_dump
    && !llb_hidden_path && !llb_img_dump && !llb_img_req && llb_img_state == 1 && !llb_pfx_on()
    && o->n == 0;
}

static void gen_line(FILE* f, const char* t, u64 n) {
  if (fwrite(t, 1, n, f) != n) {
    err_fail("a short write on a standard stream");
  }
}

static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {
  static float* tp;  // pinned: two steps' four words
  static CUevent ev[2];
  // the run's four words on the device, not in y: y is the corpus's (unified memory), which the
  // host's Bend evaluation reads, so each touch there would move its page back (a ~40 us fault)
  static llb_dptr gy;
  if (tp == NULL) {
    gy = llb_alloc_zero(64);
    if (!gy || cuMemHostAlloc((void**)&tp, 8 * sizeof(float), 0) != CUDA_SUCCESS
      || cuEventCreate(&ev[0], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS
      || cuEventCreate(&ev[1], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
      err_fail("decode run: cannot set up");
    }
  }
  WarpOp op = *(WarpOp*)o;
  u32 cnt = n;
  if (pos + n > pass_cap) {
    cnt = pass_cap - pos;
  }
  struct timespec t0;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (u32 s = 0; s <= cnt; s += 1) {
    if (s < cnt) {
      if (s > 0) {
        step_us(pos + s);
      }
      llb_gforce = 1;
      pass_fold(&op, gy, grid, sh, dT, pos + s, s > 0);
      llb_gforce = 0;
      if (cuMemcpyDtoHAsync(tp + 4 * (s & 1), gy, 16, NULL) != CUDA_SUCCESS
        || cuEventRecord(ev[s & 1], NULL) != CUDA_SUCCESS) {
        err_fail("decode run: cannot queue the pick's copy");
      }
    }
    if (s == 0) {
      continue;
    }
    // step s - 1 is done once its copy is: print it (all but the last step of the run)
    u32 q = s - 1;
    if (cuEventSynchronize(ev[q & 1]) != CUDA_SUCCESS) {
      err_fail("device fault");
    }
    const float* t = tp + 4 * (q & 1);
    union { float f; u32 u; } v = { t[0] }, w = { t[2] };
    bool fin = (v.u & 0x7f800000u) != 0x7f800000u && (w.u & 0x7f800000u) != 0x7f800000u;
    if (q + 1 == cnt || !fin) {
      // the last step (or a nonfinite one, which gen's check then reports): its words into y
      // (a nonfinite step q < cnt - 1 has step q + 1 queued behind it: y gets q's words)
      if (q + 1 != cnt) {
        if (cuCtxSynchronize() != CUDA_SUCCESS || cuMemcpyHtoD(yp, t, 16) != CUDA_SUCCESS) {
          err_fail("device fault");
        }
      } else if (cuMemcpyDtoDAsync(yp, gy, 16, NULL) != CUDA_SUCCESS) {
        err_fail("device fault");
      }
      union { float f; u32 u; } cf = { (float)(q + 1) };
      if (cuMemsetD32Async(yp + 16, cf.u, 1, NULL) != CUDA_SUCCESS) {
        err_fail("decode run: cannot set the count");
      }
      return;
    }
    char b[160];
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long long ms = ((long long)t1.tv_sec * 1000 + t1.tv_nsec / 1000000) - ((long long)t0.tv_sec * 1000 + t0.tv_nsec / 1000000);
    t0 = t1;
    u32 id = (u32)t[1], j = (u32)t[3];
    if (fflush(stdout) != 0) {
      err_fail("a short write on a standard stream");
    }
    int m = snprintf(b, sizeof b, "logits %u %u %u %u %u\n", pos + q, id, v.u, j, w.u);
    gen_line(stderr, b, (u64)m);
    m = snprintf(b, sizeof b, "step %u %lld\n", pos + q, ms);
    gen_line(stderr, b, (u64)m);
    m = snprintf(b, sizeof b, "%u\n", id);
    gen_line(stdout, b, (u64)m);
  }
}
#endif

#if defined(LLB_LAUNCH) && defined(LLB_WARP_FID)
// The parked warp GEMVs (warp_park), args (f, +wb, +x, y, +base, +n, +d, +rb)
// as flat.bend's gemv: each runs as one llb_warp launch (LLB_WARP_GRID
// blocks of 128, default 164), then llb_deliver hands y to its continuation.
// The operands come from the mailbox (llb_mail): the host reads no corpus
// word, and the next batch queues behind these launches without a sync.
static bool warp_serve(Corpus H) {
  u32 m = gpu_mail->m;
  if (m == 0) {
    return false;
  }
  if (gpu_mail->bad != 0) {
    err_fail(gpu_mail->bad == 7 ? "warp pass: the model shape is outside the CUDA kernels' envelope (head_dim 256, "
        "ssm_state_size 128, n_heads a multiple of n_kv_heads, ssm_time_step_rank a multiple of ssm_group_count "
        "and <= 256, d_model <= 32768, every width a multiple of 256)"
      : gpu_mail->bad == 6 ? "warp pass: bad config words or a position past the KV capacity (prompt + steps)"
      : gpu_mail->bad == 8 ? "warp pass: a MoE layer's table is outside the CUDA expert kernels (expert_used_count "
        "<= expert_count, expert and shared widths multiples of 256, router and shared-expert gate F32, experts Q8_0 / "
        "Q4_K / Q5_K / Q6_K; rebuild the sidecar with this gguf2pack)"
      : gpu_mail->bad == 2 ? "warp attention: unsupported operands"
      : gpu_mail->bad == 3 ? "warp rms: unsupported operands"
      : gpu_mail->bad == 4 ? "warp GEMV: row format refused (untagged"
        " 17 | rb reads as F32 or Q8_0, or a bad tag)"
      : "warp GEMV: unsupported operands");
  }
  const char* gs   = getenv("LLB_WARP_GRID");
  u32         grid = gs != NULL && atoi(gs) > 0 ? (u32)atoi(gs) : 768;
  const char* qg   = getenv("LLB_Q4GLUE");
  llb_q4glue = qg != NULL ? (u32)atoi(qg) : 3;
  const char* qd   = getenv("LLB_Q4_DP4A");
  llb_q4dp = qd == NULL ? 0 : atoi(qd) == 1 ? 3 : atoi(qd) == 2 ? 1 : atoi(qd) == 3 ? 2
    : atoi(qd) == 4 ? 15 : atoi(qd) == 5 ? 11 : atoi(qd) == 6 ? 7 : atoi(qd) == 7 ? 31 : atoi(qd) == 8 ? 63 : atoi(qd) == 9 ? 127 : 0;
  const char* dp = getenv("LLB_DUMP");
  llb_dump = dp != NULL && *dp ? dp : NULL;
  const char* dpp = getenv("LLB_DUMP_POS");
  llb_dump_pos = dpp != NULL ? atoi(dpp) : -1;
  const char* sps = getenv("LLB_SPEC");
  spec_k = sps != NULL && *sps ? atoi(sps) : -1;
  spec_mtp = getenv("LLB_MTP");
  spec_mtp = spec_mtp != NULL && *spec_mtp ? spec_mtp : NULL;
  spec_adapt = getenv("LLB_SPEC_ADAPT") != NULL && atoi(getenv("LLB_SPEC_ADAPT")) == 1;
  for (u32 j = 0; j < m; j += 1) {
    volatile WarpOp* o = &gpu_mail->op[j];
    u64 w = o->w, xp = o->x, yp = o->y;
    // Q8_0 / F16 / Q4_K / Q6_K (llb_warp_q8 / llb_warp_f16 / llb_warp_q4k /
    // llb_warp_q6k): rq is the row's bytes; cg = c / 4 = rb / 17 * 4 (Q8_0),
    // rb / 8 (F16), rb / 144 * 64 (Q4_K), rb / 210 * 64 (Q6_K)
    u32 n = o->n, d = o->d, fm = o->fm, rq = fm ? o->rb : o->rb / 16;
    u32 cg = fm == 1 ? o->rb / 17 * 4 : fm == 3 ? o->rb / 144 * 64
      : fm == 4 ? o->rb / 210 * 64 : fm == 5 ? o->rb / 110 * 64
      : fm == 6 ? o->rb / 18 * 16 : fm == 7 ? o->rb / 34 * 32
      : fm == 8 ? o->rb / 136 * 64
      : o->rb / (fm ? 8 : 16);
    llb_dptr op = gpu_att + (llb_dptr)j * 2048 * 4;
    // attention (f, +wt, +q, +kc, +vc, +c, y, +wo, +r): its o GEMV reads
    // llb_attn's output, a slot of gpu_att (2048 x 2048 rows of wo, d = 9);
    // then llb_add adds the residual r into y
    if (o->kind == 1) {
      u64 kp = o->k, vp = o->v;
      u32 c  = o->c;
      if (!LLB_LAUNCH(gpu_attn, 16, 1, 128, &xp, &kp, &vp, &op, &c)) {
        err_fail("device launch failed");
      }
      xp = op;
    }
    // rms (f, +wb, +x, y, +base, +n, +d, +rb, +an): the GEMV reads
    // llb_rms(x, the 2048 f32 at byte an of wb), a slot of gpu_att
    if (o->kind == 2 || o->kind == 4) {
      u64 wn = o->wn;
      if (!LLB_LAUNCH(gpu_rms, 1, 1, 1024, &xp, &wn, &op)) {
        err_fail("device launch failed");
      }
      xp = op;
    }
    // FFN (kind 5): llb_rms(x) -> the gate and up GEMVs -> llb_silu, three
    // 6144 slots of gpu_act; the down GEMV below reads the silu slot
    if (o->kind == 5) {
      u64 wn = o->wn, wg = o->k, wu = o->v;
      u32 rg = o->c, gm = o->gm, nf = 6144, df = 9;
      // gate/up carry their own format (gm), which may differ from the down
      // rows' (fm) in a Q4_K_M mixture
      u32 gq = gm ? rg : rg / 16, gc = gm == 1 ? rg / 17 * 4
        : gm == 3 ? rg / 144 * 64 : gm == 4 ? rg / 210 * 64
        : gm == 5 ? rg / 110 * 64 : gm == 6 ? rg / 18 * 16
        : gm == 7 ? rg / 34 * 32 : gm == 8 ? rg / 136 * 64 : rg / (gm ? 8 : 16);
      llb_dptr ga = gpu_act + (llb_dptr)j * 3 * 6144 * 4;
      llb_dptr ua = ga + 6144 * 4, sa = ua + 6144 * 4;
      if (!LLB_LAUNCH(gpu_rms, 1, 1, 1024, &xp, &wn, &op)
        || !LLB_LAUNCH(gm == 8 ? gpu_warp4 : gm == 7 ? gpu_warpq2 : gm == 6 ? gpu_warp2 : gm == 5 ? gpu_warp3 : gm == 4 ? gpu_warp6 : gm == 3 ? gpu_warpk : gm == 2 ? gpu_warph : gm ? gpu_warpq : gpu_warp, grid, 1, 128, &wg, &gq, &gc, &op, &ga, &nf, &df)
        || !LLB_LAUNCH(gm == 8 ? gpu_warp4 : gm == 7 ? gpu_warpq2 : gm == 6 ? gpu_warp2 : gm == 5 ? gpu_warp3 : gm == 4 ? gpu_warp6 : gm == 3 ? gpu_warpk : gm == 2 ? gpu_warph : gm ? gpu_warpq : gpu_warp, grid, 1, 128, &wu, &gq, &gc, &op, &ua, &nf, &df)
        || !LLB_LAUNCH(gpu_silu, 24, 1, 256, &ga, &ua, &sa, &nf)) {
        err_fail("device launch failed");
      }
      xp = sa;
    }
    // the Qwen3.5 hybrid bangs (tools/payload-hybrid.patch): one kernel a task
    // into y; no generic GEMV follows
    if (o->kind >= 6) {
      if (o->kind == 6) {
        union { u32 u; float f; } sc = { o->c };
        u32 hs = o->n;
        if (!LLB_LAUNCH(gpu_l2n, hs, 1, 64, &xp, &yp, &sc.f, &hs)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 7) {
        u64 wp = o->w, xp2 = o->x, h0p = o->k, h1p = o->v, h2p = o->wn;
        u32 nch = o->n;
        if (!LLB_LAUNCH(gpu_conv, (nch + 255) / 256, 1, 256, &wp, &xp2, &h0p, &h1p, &h2p, &yp, &nch)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 8) {
        u64 op8 = o->x, wp8 = o->w, zp = o->k;
        u32 hs = o->n;
        if (!LLB_LAUNCH(gpu_grn, hs, 1, 64, &op8, &wp8, &zp, &yp, &hs)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 9) {
        u64 Sp = o->w, qp = o->x, kkp = o->k, vp = o->v, decp = o->wn, betp = o->r;
        u32 hs = o->n;
        if (!LLB_LAUNCH(gpu_gdl, hs, 32, 128, &Sp, &qp, &kkp, &vp, &decp, &betp, &yp, &hs)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 10) {
        // the standalone rms_norm bang (tools/payload-rmsnorm.patch)
        u64 xp10 = o->x, wp10 = o->w;
        u32 n10 = o->n;
        if (!LLB_LAUNCH(gpu_rnorm, 1, 1, 1024, &xp10, &wp10, &yp, &n10)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 11 || o->kind == 12) {
        // the elementwise vadd / vmul bangs (tools/payload-vadd.patch)
        u64 xp11 = o->x, zp11 = o->k;
        u32 n11 = o->n;
        u32 g11 = (n11 + 1023) / 1024;
        if (!LLB_LAUNCH(o->kind == 11 ? gpu_vadd : gpu_vmul, g11, 1, 1024, &xp11, &zp11, &yp, &n11)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 21) {
        // the KV-cache slot write (tools/payload-slot.patch)
        u64 xp21 = o->x;
        if (!LLB_LAUNCH(gpu_slot, 1, 1, 1024, &xp21, &yp)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 20) {
        // the qk-norm + neox rope bangs (tools/payload-qkr.patch)
        u64 xp20 = o->x, wp20 = o->w;
        u32 n20 = o->n, qk20 = o->c, pos20 = (u32)o->r;
        u32 rows20 = qk20 != 0 ? n20 / 512 : n20 / 256;
        if (!LLB_LAUNCH(gpu_qkr, rows20, 1, 256, &xp20, &wp20, &yp, &n20, &qk20, &pos20)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 14) {
        // the head bang (tools/payload-head.patch): rms_final -> h (gpu_headh),
        // the head GEMV into gpu_logit, then llb_amax into y
        if (smp_on()) {
          err_fail("LLB_SAMPLE: needs the folded pass (the unfolded head op has no position)");
        }
        u64 wn = o->k, op2 = gpu_headh;
        u32 n4 = 4096;
        u32 rq = o->rb, cg = o->rb / 17 * 4, dh = o->d;
        if (!LLB_LAUNCH(gpu_rnorm, 1, 1, 1024, &xp, &wn, &op2, &n4)
          || !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &w, &rq, &cg, &op2, &gpu_logit, &n, &dh)
          || !LLB_LAUNCH(gpu_amaxp, AMAX_G, 1, AMAX_T, &gpu_logit, &n, &yp)
          || !LLB_LAUNCH(gpu_amax, 1, 1, AMAX_G, &gpu_logit, &n, &yp)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 15) {
        // the Qwen3.5 attention block bang (tools/payload-attn4.patch): one
        // block a query head, 256 threads (a dim each)
        u64 qp15 = o->x, kp15 = o->k, vp15 = o->v;
        u32 c15 = o->c;
        u64 s15 = gpu_atts, p15 = gpu_atts + ((u64)16 << 17) * 4;
        u32 gqa4 = 4;
        if (!llb_attn4_run(16, qp15, kp15, vp15, yp, s15, p15, c15, gqa4)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 16) {
        // the SwiGLU silu-mul bang (tools/payload-silumul.patch): y :=
        // silu(g) * u, the existing llb_silu (256 threads, one an element)
        u64 gp16 = o->x, up16 = o->k;
        u32 n16 = o->n;
        u32 g16 = (n16 + 255) / 256;
        if (!LLB_LAUNCH(gpu_silu, g16, 1, 256, &gp16, &up16, &yp, &n16)) {
          err_fail("device launch failed");
        }
      } else if (o->kind == 17) {
        llb_fbn_run(o, yp, grid, NULL);
      } else if (o->kind == 18) {
        llb_linb_run(o, yp, grid, NULL);
      } else if (o->kind == 22) {
        // the full-attention block bang (tools/payload-fulb.patch): the
        // kernels, grids and arguments of the rmsnorm, q GEMV, qkr(q), k
        // GEMV, qkr(k), v GEMV, slot, slot and aob bangs, in that order, on
        // gpu_blk scratch instead of the interpreter's arrays
        if (m != 1) {
          err_fail("warp fulb: gpu_blk scratch needs a one-task batch");
        }
        llb_fulb_run(o, yp, grid, NULL);
      } else if (o->kind == 23) {
        if (m != 1) {
          err_fail("warp pass: gpu_blk scratch needs a one-task batch");
        }
        llb_pass_run(o, yp, grid);
      } else if (o->kind == 19) {
        // the full-attention output tail bang (tools/payload-aob.patch):
        // attn4(q, kc, vc) -> outputgate(attn, gate) -> the o GEMV into y
        u64 qp19 = o->x, gp19 = o->k, kcp19 = o->v, vcp19 = o->wn;
        u64 wao19 = o->w;
        u32 c19 = o->c, rg19 = o->rb, cg19 = rg19 / 17 * 4;
        u64 s19 = gpu_atts, p19 = gpu_atts + ((u64)16 << 17) * 4;
        u32 gqa19 = 4;
        llb_dptr attn19 = gpu_blk, gated19 = attn19 + 4096 * 4;
        u32 n4_19 = 4096, d10_19 = 10, gog19 = (4096 + 1023) / 1024;
        if (!llb_attn4_run(16, qp19, kcp19, vcp19, attn19, s19, p19, c19, gqa19)
          || !LLB_LAUNCH(gpu_og, gog19, 1, 1024, &attn19, &gp19, &gated19, &n4_19)
          || !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &wao19, &rg19, &cg19, &gated19, &yp, &n4_19, &d10_19)) {
          err_fail("device launch failed");
        }
      } else {
        // the output gate bang (tools/payload-og.patch)
        u64 xp13 = o->x, gp13 = o->k;
        u32 n13 = o->n;
        u32 g13 = (n13 + 1023) / 1024;
        if (!LLB_LAUNCH(gpu_og, g13, 1, 1024, &xp13, &gp13, &yp, &n13)) {
          err_fail("device launch failed");
        }
      }
      continue;
    }
    if (!LLB_LAUNCH(fm == 9 ? gpu_warp5 : fm == 8 ? gpu_warp4 : fm == 7 ? gpu_warpq2 : fm == 6 ? gpu_warp2 : fm == 5 ? gpu_warp3 : fm == 4 ? gpu_warp6 : fm == 3 ? gpu_warpk : fm == 2 ? gpu_warph : fm ? gpu_warpq : gpu_warp, grid, 1, 128, &w, &rq, &cg, &xp, &yp, &n, &d)) {
      err_fail("device launch failed");
    }
    // the residual add: attention's (kind 1), the GEMV-add bang's (3), FFN's (5)
    if (o->kind == 1 || o->kind == 3 || o->kind == 5) {
      u64 rp = o->r;
      if (!LLB_LAUNCH(gpu_add, 8, 1, 256, &rp, &yp)) {
        err_fail("device launch failed");
      }
    }
    // rope (kind 4): K.rope over the GEMV's n outputs in place, at o->c
    if (o->kind == 4) {
      float pf = (float)o->c;
      if (!LLB_LAUNCH(gpu_rope, (n / 2 + 127) / 128, 1, 128, &yp, &n, &pf)) {
        err_fail("device launch failed");
      }
    }
  }
  llb_dptr hd = llb_dev(H);
  if (!LLB_LAUNCH(gpu_dlv, 1, 1, 1, &hd, &m)) {
    err_fail("device launch failed");
  }
  gpu_mail->m = 0;
  return true;
}
#else
#define warp_serve(H) false
#endif
