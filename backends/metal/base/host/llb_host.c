#if defined(LLB_LAUNCH) && defined(LLB_WARP_FID)
#include "llb_backend.h"
#include "llb_formats.h"
#include <stdio.h>
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

// the engine's cdepth: a Q8 GEMV's column depth, ceil(log2(cols)) - 2
static u32 llb_cdepth(u32 cols) {
  u32 d = 0;
  while ((1u << d) < cols) {
    d += 1;
  }
  return d - 2;
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
  static float buf[8192];
  u32 header[2] = {pos, width};
  if (width > 8192 || !llb_read(buf, row, (size_t)width * 4))
    err_fail("LLB_DUMP_HIDDEN: read failed");
  FILE* f = fopen(llb_hidden_path, llb_hidden_written ? "ab" : "wb");
  if (!f || fwrite(header, 4, 2, f) != 2 || fwrite(buf, 4, width, f) != width)
    err_fail("LLB_DUMP_HIDDEN: write failed");
  if (fclose(f)) err_fail("LLB_DUMP_HIDDEN: close failed");
  llb_hidden_written = true;
}
static bool llb_kgemv_res(u32 fm, u64 w, u32 cols, u64 x, u64 y, u32 n, u32 grid, u64 x1) {
  u32 rb = LLB_FM_RB(fm, cols), cg = cols / 4, d = llb_cdepth(cols);
  if (!(llb_q4glue & 2) || (fm != LLB_FM_Q4_K && fm != LLB_FM_Q6_K) || cg % 256 != 0 || d < 9 || d > 13) {
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
  if (fg == 1 && fu == 1) {
    if (!LLB_LAUNCH(gpu_warpqc, grid, 1, 128, &rg17, &cgu, &h2b, &c17)) {
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
  if (fd == 1) {
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
  u32 rg18 = o->brg, cg18 = rg18 / 17 * 4, ro18 = o->bro, co18 = ro18 / 17 * 4;
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
  bool q8 = fq == 1 && fz == 1 && fb == 1 && fa == 1;
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
      if (fb == 1 && fa == 1) {
        if (!LLB_LAUNCH(gpu_warpqc, grid, 1, 128, &rg18, &cg18, &hb18, &ba)) {
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
      : fo == 1 && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &sop, &ro18, &co18, &normed, &yp, &n4, &dso))) {
    err_fail("device launch failed");
  }
  if (fo != 1) {
    llb_kgemv(fo, sop, lv, normed, yp, n4, grid);
  }
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
  u64 s22 = gpu_atts, p22 = gpu_atts + ((u64)llb_sh.nh << 17) * 4;
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
  bool q8 = fq == 1 && fk == 1 && fv == 1;
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
    || !LLB_LAUNCH(gpu_slot, 1, 1, 1024, &kn22, &kslot22)
    || !LLB_LAUNCH(gpu_slot, 1, 1, 1024, &vb22, &vslot22)
    || !LLB_LAUNCH(gpu_attn4, ah22, 1, 1024, &qg22, &kcp22, &vcp22, &attn22, &s22, &p22, &c22, &grp22)
    || !LLB_LAUNCH(gpu_og, gog22, 1, 1024, &attn22, &gate22, &gated22, &nqa)
    || (fo == 1 && !LLB_LAUNCH(gpu_warpq, grid, 1, 128, &wop22, &ro22, &co22, &gated22, &yp, &n4, &dqa))) {
    err_fail("device launch failed");
  }
  if (fo != 1) {
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
// ponytail: commits 16 x 2^(dT + 10) f32 of caches up front (8 GB at dT 17);
// allocate by pos if long contexts matter.
static llb_dptr pass_b[64], pass_S[64], pass_kc[64], pass_vc[64], pass_x[3];
static u32  pass_r[64], pass_pos[64], pass_dT;
static bool pass_on[64];

static llb_dptr pass_zero(size_t n) {
  llb_dptr p = llb_alloc_zero(n * 4);
  if (!p) {
    err_fail("warp pass: cannot allocate the layer state");
  }
  return p;
}

// layer i's state at pos: allocated (pos 0), zeroed (pos 0 again: the next prompt of a
// multi-prompt run, engine ids cut at ","), else pos must follow the last
static void pf_dump(u32 i, bool full, u32 pos, LlbShape sh);  // llb_pf.h: G3's LLB_PF_DUMP
static void pass_layer(u32 i, bool full, u32 pos, u32 dT, LlbShape sh) {
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd;
  pf_dump(i, full, pos, sh);
  if (!pass_on[i]) {
    if (pos != 0) {
      err_fail("warp pass: a layer's state must start at pos 0");
    }
    if (full) {
      pass_kc[i] = pass_zero(((size_t)1 << dT) * sh.nkv * sh.hd);
      pass_vc[i] = pass_zero(((size_t)1 << dT) * sh.nkv * sh.hd);
    } else {
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
    bool z = full ? llb_zero(pass_kc[i], ((size_t)1 << dT) * sh.nkv * sh.hd * 4)
        && llb_zero(pass_vc[i], ((size_t)1 << dT) * sh.nkv * sh.hd * 4)
      : llb_zero(pass_b[i], 4 * (size_t)cv * 4) && llb_zero(pass_S[i], (size_t)sh.lvh * sh.sd * sh.sd * 4);
    if (!z) {
      err_fail("warp pass: cannot reset the layer state");
    }
  } else if (pass_pos[i] + 1 != pos) {
    err_fail("warp pass: a layer served out of order");
  }
  pass_pos[i] = pos;
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
  spec_nv += 1;
}

static void llb_pass_run(volatile WarpOp* o, u64 yp, u32 grid) {
  u32 c = o->c, dT = c & 31, pos = c >> 5, s = o->n, k = o->d;
  volatile LlbShape* ms = &gpu_mail->sh;
  LlbShape sh = { ms->d, ms->f, ms->nh, ms->nkv, ms->hd, ms->lkh, ms->lvh, ms->sd };
  if (pass_x[0] == 0) {
    llb_sh = sh;
    for (u32 b = 0; b < 3; b += 1) {
      pass_x[b] = pass_zero(sh.d);
    }
    pass_dT = dT;
  }
  if (dT != pass_dT) {
    err_fail("warp pass: dT changed");
  }
  if (memcmp(&sh, &llb_sh, sizeof sh) != 0) {
    err_fail("warp pass: the shape record changed");
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
  if (fold && pf_take(o, pos, dT, sh, yp)) {
    return;
  }
  if (fold) {
    // the folded step: the token's embedding row into pass_x[0] (o->bb its
    // format, pass_mail admits Q8_0 and Q4_K)
    u64 er = o->w, x0 = pass_x[0];
    u32 c4 = sh.d;
    if (!LLB_LAUNCH(o->bb == LLB_FM_Q4_K ? gpu_embedk : gpu_embed, (c4 + 255) / 256, 1, 256, &er, &c4, &x0)) {
      err_fail("device launch failed");
    }
    cur = x0;
  }
  for (u32 j = 0; j < k; j += 1) {
    u32  i = s + j;
    bool full = (i + 1) % 4 == 0;
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
    u64 nxt = j + 1 == k && !fold ? yp : pass_x[cur == pass_x[0] ? 1 : 0];
    llb_fbn_run(&ff, nxt, grid, gpu_mail->pt[i]);
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
  if (fold) {
    // the head bang's chain (kind 14): rms_final -> gpu_headh, the head GEMV
    // (o->bg its format) into gpu_logit, llb_amax's four words into y
    u64 wn = o->k, wo = o->v, op2 = gpu_headh, lg = gpu_logit;
    u32 n4 = sh.d, nh = o->gm, rq = o->bq, cg = o->bq / 17 * 4, dh = llb_cdepth(sh.d);
    bool q8 = o->bg == LLB_FM_Q8_0;
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
    if (llb_dump) {
      llb_dptr lgp = lg;
      llb_dump_put("l", pos, -1, "wb", &lgp, &nh, 1);
    }
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
    err_fail(gpu_mail->bad == 2 ? "warp attention: unsupported operands"
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
        if (!LLB_LAUNCH(gpu_attn4, 16, 1, 1024, &qp15, &kp15, &vp15, &yp, &s15, &p15, &c15, &gqa4)) {
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
        if (!LLB_LAUNCH(gpu_attn4, 16, 1, 1024, &qp19, &kcp19, &vcp19, &attn19, &s19, &p19, &c19, &gqa19)
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
