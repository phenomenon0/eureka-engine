// llb_pf.h (PF27 stage 2): the fast prefill tier, LLB_PREFILL=fast (default off: every step
// takes the exact path untouched). Inlined into llb_host.c after pass_layer.
//
// The engine's prompt loop sets the defer bit (pass_mail: o->bsn) on every prompt token but
// the last. With the tier on, a deferred folded step only embeds its token into the chunk's
// row pf_x[0][q] and returns (y stays the caller's zeros; the prompt loop reads only the last
// token's). A full chunk (LLB_PF_T tokens, default 512), or the next non-deferred step, runs
// the queued tokens layer-major: T-row GEMMs on int8 tensor cores (tools/pf27/pfmm.cuh) and the
// row-batched small kernels (llb_pf.cuh), on the same host-held layer state the exact path
// reads -- conv ring and index, S, kc / vc slots, pass_pos -- left as T exact steps would leave
// it. The non-deferred step (the prompt's last token) then runs the exact path as before.
// Not byte-exact: x is quantized per 32-block to int8 (q8_1) for the GEMMs.

static u32 pf_mode = 2;  // 2 = LLB_PREFILL unread, 1 = fast, 0 = off
static u32 pf_T, pf_n, pf_pos0;
// chunk rows: x ping-pong, x1, the norm h, the GEMM outputs and the small kernels' rows
static llb_dptr pf_x[2], pf_x1, pf_h, pf_qx, pf_xd, pf_qkv, pf_z, pf_bp, pf_ap, pf_beta, pf_dec;
static llb_dptr pf_cnv, pf_qt, pf_kt, pf_go, pf_nm, pf_g, pf_u, pf_kb, pf_vb, pf_kn;
// the chunk's token ids and embedding rows, the output norm (o->k): LLB_SPEC's handoff reads them
static u32* pf_tk;
static u64* pf_er;
static u64 pf_wn;
static void spec_pf(u32 T, llb_dptr x, LlbShape sh);  // llb_host.c: LLB_SPEC after a fast chunk

static llb_dptr pf_alloc(size_t bytes) {
  llb_dptr p = llb_alloc_zero(bytes);
  if (!p) {
    err_fail("fast prefill: cannot allocate the chunk rows");
  }
  return p;
}

static bool pf_fast(LlbShape sh) {
  if (pf_mode == 2) {
    const char* e = getenv("LLB_PREFILL");
    pf_mode = e != NULL && strcmp(e, "fast") == 0;
    if (pf_mode) {
      const char* t = getenv("LLB_PF_T");
      pf_T = t != NULL && atoi(t) > 0 ? (u32)atoi(t) : 512;
      size_t T = pf_T, d = sh.d, f = sh.f, lk = sh.lkh * sh.sd, lv = sh.lvh * sh.sd;
      size_t cv = 2 * lk + lv, qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd;
      size_t wq = cv > 2 * qa ? cv : 2 * qa;
      pf_x[0] = pf_alloc(T * d * 4); pf_x[1] = pf_alloc(T * d * 4);
      pf_x1 = pf_alloc(T * d * 4); pf_h = pf_alloc(T * d * 4);
      pf_qx = pf_alloc(T * f); pf_xd = pf_alloc(T * f / 32 * 8);
      pf_qkv = pf_alloc(T * wq * 4); pf_cnv = pf_alloc(T * wq * 4);
      pf_z = pf_alloc(T * lv * 4);
      pf_bp = pf_alloc(T * sh.lvh * 4); pf_ap = pf_alloc(T * sh.lvh * 4);
      pf_beta = pf_alloc(T * sh.lvh * 4); pf_dec = pf_alloc(T * sh.lvh * 4);
      pf_qt = pf_alloc(T * lv * 4); pf_kt = pf_alloc(T * lv * 4);
      pf_go = pf_alloc(T * lv * 4); pf_nm = pf_alloc(T * lv * 4);
      pf_g = pf_alloc(T * f * 4); pf_u = pf_alloc(T * f * 4);
      pf_kb = pf_alloc(T * kv * 4); pf_vb = pf_alloc(T * kv * 4); pf_kn = pf_alloc(T * kv * 4);
      pf_tk = malloc(T * 4); pf_er = malloc(T * 8);
      if (!pf_tk || !pf_er) {
        err_fail("fast prefill: cannot allocate the chunk ids");
      }
    }
  }
  return pf_mode == 1;
}

// pfmm.cuh's PfT<TY, 4, 2, EP, NT>::SM: 2 ST + ROWS NJ 8 (ROWS 128); TY 3 Q4_K (TK 64), 1 Q8_0
// (TK 128), 5 Q5_K (TK 48, + the qh plane 128 x 32)
static u32 pf_tile(u32 ty) {
  return ty == 1 ? 128 : ty == 5 ? 48 : 64;
}
static u32 pf_sm(u32 ty) {
  u32 ks = ty == 1 ? 128 : 256, eb = ty == 1 ? 48 : 80, tk = pf_tile(ty), qh = ty == 5 ? 128 * 32 : 0;
  return 2 * (128 * 160 + tk * (ks + 32) + tk * eb + qh) + 128 * (ks / 32) * 8;
}

// x (T rows of c) to q8_1, then y = W x (+ x1 when x1 != 0) in W's format fm
static void pf_q8(llb_dptr x, u32 c, u32 T) {
  u32 g = T * c / 256;
  if (c % 256 != 0 || !LLB_LAUNCH(gpu_pfq8x, g, 1, 256, &x, &pf_qx, &pf_xd)) {
    err_fail("fast prefill: q8x launch failed");
  }
}
static void pf_mm(u32 fm, u64 w, u32 n, u32 c, u32 T, llb_dptr y, llb_dptr x1) {
  u32 tb = (T + 63) / 64, rb = (n + 127) / 128;
  bool ok;
  if (fm == LLB_FM_Q4_K) {
    ok = LLB_LAUNCH_SM(x1 ? gpu_pfq4mr : gpu_pfq4m, tb, rb, 256, pf_sm(3),
      &w, &w, &n, &c, &pf_qx, &pf_xd, &T, &y, &x1);
  } else if (fm == LLB_FM_Q8_0 && c % 128 == 0) {
    tb = (T + pf_tile(1) - 1) / pf_tile(1);
    ok = LLB_LAUNCH_SM(x1 ? gpu_pfq8mr : gpu_pfq8m, tb, rb, 256, pf_sm(1),
      &w, &w, &n, &c, &pf_qx, &pf_xd, &T, &y, &x1);
  } else if (fm == LLB_FM_Q5_K) {
    tb = (T + pf_tile(5) - 1) / pf_tile(5);
    ok = LLB_LAUNCH_SM(x1 ? gpu_pfq5mr : gpu_pfq5m, tb, rb, 256, pf_sm(5),
      &w, &w, &n, &c, &pf_qx, &pf_xd, &T, &y, &x1);
  } else if (fm == LLB_FM_Q6_K) {
    ok = LLB_LAUNCH_SM(x1 ? gpu_pfq6wr : gpu_pfq6w, tb, rb, 256, 90112,  // pfmm.cuh's PF6_SM
      &w, &w, &n, &c, &pf_qx, &pf_xd, &T, &y, &x1);
  } else {
    err_fail("fast prefill: a GEMM format outside Q4_K / Q5_K / Q8_0 / Q6_K");
  }
  if (!ok) {
    err_fail("fast prefill: GEMM launch failed");
  }
}

// pfmm.cuh's PfCat: up to 4 Q8_0 GEMMs on one x in one launch (b: cumulative 128-row blocks)
struct PfCat { u64 W[4]; llb_dptr y[4]; u32 n[4]; u32 b[5]; };
static void pf_catq(u32 m, const u32* fm, const u64* w, const u32* n, const llb_dptr* y, u32 c, u32 T) {
  bool q8 = c % 128 == 0;
  for (u32 i = 0; i < m; i += 1) {
    q8 = q8 && fm[i] == LLB_FM_Q8_0;
  }
  if (!q8) {
    for (u32 i = 0; i < m; i += 1) {
      pf_mm(fm[i], w[i], n[i], c, T, y[i], 0);
    }
    return;
  }
  struct PfCat k = { { 0 }, { 0 }, { 0 }, { 0 } };
  for (u32 i = 0; i < 4; i += 1) {
    u32 j = i < m ? i : m - 1;
    k.W[i] = w[j]; k.y[i] = y[j]; k.n[i] = n[j];
    k.b[i + 1] = k.b[i] + (i < m ? (n[i] + 127) / 128 : 0);
  }
  if (!LLB_LAUNCH_SM(gpu_pfq8mc, (T + pf_tile(1) - 1) / pf_tile(1), k.b[4], 256, pf_sm(1), &k, &c, &pf_qx, &pf_xd, &T)) {
    err_fail("fast prefill: GEMM launch failed");
  }
}

// the queued tokens pf_pos0 .. + T - 1 through all k layers (layer bases and table from the Mail)
static void pf_chunk(u32 T, u32 k, u32 dT, LlbShape sh) {
  u32 d = sh.d, f = sh.f, lk = sh.lkh * sh.sd, lv = sh.lvh * sh.sd, cv = 2 * lk + lv;
  u32 qa = sh.nh * sh.hd, kv = sh.nkv * sh.hd, q2 = 2 * qa, nm = sh.lvh, hk = sh.lkh;
  u32 rep = sh.lvh / sh.lkh, grp = sh.nh / sh.nkv, pos0 = pf_pos0, one = 1, zero = 0;
  union { u32 u; float f; } sq = { 0x413504f3u };
  llb_dptr cur = pf_x[0];
  for (u32 i = 0; i < k; i += 1) {
    bool full = (i + 1) % 4 == 0;
    pass_layer(i, full, pos0, dT, sh);
    pass_pos[i] = pos0 + T - 1;
    u64 base = gpu_mail->pw[i];
    u32 p[20], fm[20];
    for (u32 q = 0; q < 20; q += 1) {
      p[q] = gpu_mail->pt[i][q] & 0x0fffffffu;
      fm[q] = gpu_mail->pt[i][q] >> 28;
    }
    u64 an = base + p[0];
    if (!LLB_LAUNCH(gpu_pfrms, T, 1, 1024, &cur, &an, &pf_h, &d)) {
      err_fail("fast prefill: launch failed");
    }
    pf_q8(pf_h, d, T);
    if (full) {
      llb_dptr kc = pass_kc[i], vc = pass_vc[i], qn = base + p[17], kn = base + p[18];
      llb_dptr gate = pf_cnv + (llb_dptr)qa * 4;
      u32 nh = sh.nh, nkv = sh.nkv, ogx = (qa + 255) / 256;
      u64 cw3[3] = { base + p[14], base + p[15], base + p[16] };
      u32 cn3[3] = { q2, kv, kv };
      llb_dptr cy3[3] = { pf_qkv, pf_kb, pf_vb };
      pf_catq(3, fm + 14, cw3, cn3, cy3, d, T);
      // q rows of 2 qa -> pf_cnv (q heads, then the gates), k -> pf_kn
      if (!LLB_LAUNCH(gpu_pfqkr, nh, T, 256, &pf_qkv, &qn, &pf_cnv, &q2, &one, &pos0)
        || !LLB_LAUNCH(gpu_pfqkr, nkv, T, 256, &pf_kb, &kn, &pf_kn, &kv, &zero, &pos0)
        || !LLB_LAUNCH(gpu_pfslot, T, 2, 1024, &pf_kn, &pf_vb, &kc, &vc, &pos0, &kv)) {
        err_fail("fast prefill: launch failed");
      }
      // the chunk's causal attention, tiled (llb_pf.cuh pf_attn; PF_ATT_SM)
      u32 asm_ = (32 * 264 + 256 * 40) * 2;
      if (!LLB_LAUNCH_SM(gpu_pfattn, nkv, (T * grp + 63) / 64, 128, asm_, &pf_cnv, &q2, &kc, &vc, &kv,
          &pf_go, &qa, &pos0, &T, &grp)) {
        err_fail("fast prefill: launch failed");
      }
      if (!LLB_LAUNCH(gpu_pfog, ogx, T, 256, &pf_go, &gate, &q2, &pf_nm, &qa)) {
        err_fail("fast prefill: launch failed");
      }
      pf_q8(pf_nm, qa, T);
      pf_mm(fm[19], base + p[19], d, qa, T, pf_x1, cur);
    } else {
      llb_dptr ring = pass_b[i], S = pass_S[i];
      llb_dptr dt = base + p[10], sa = base + p[9], cw = base + p[11], sn = base + p[12];
      llb_dptr ck = pf_cnv + (llb_dptr)lk * 4, cvv = pf_cnv + (llb_dptr)2 * lk * 4;
      u32 r0 = pass_r[i], gx = 1 + (cv + 255) / 256, rx = (cv + 255) / 256, rt = T < 4 ? T : 4;
      u32 gdy = sh.sd / 16;  // pf_gdl: 16 rows a block
      pass_r[i] = r0 + T;
      u64 cw4[4] = { base + p[5], base + p[6], base + p[7], base + p[8] };
      u32 cn4[4] = { cv, lv, nm, nm };
      llb_dptr cy4[4] = { pf_qkv, pf_z, pf_bp, pf_ap };
      pf_catq(4, fm + 5, cw4, cn4, cy4, d, T);
      if (!LLB_LAUNCH(gpu_pfmc, gx, T, 256, &pf_bp, &pf_ap, &dt, &sa, &pf_beta, &pf_dec, &nm,
          &cw, &pf_qkv, &ring, &r0, &pf_cnv, &cv)
        || !LLB_LAUNCH(gpu_pfring, rx, rt, 256, &pf_qkv, &ring, &r0, &T, &cv)
        || !LLB_LAUNCH(gpu_pfl2qk, 2 * hk, T, 64, &pf_cnv, &ck, &cv, &pf_qt, &pf_kt, &lv, &sq.f, &hk, &rep)
        || !LLB_LAUNCH(gpu_pfgdl, nm, gdy, 128, &S, &pf_qt, &pf_kt, &lv, &cvv, &cv, &pf_dec, &pf_beta,
          &pf_go, &nm, &T)
        || !LLB_LAUNCH(gpu_pfgrn, nm, T, 64, &pf_go, &sn, &pf_z, &pf_nm, &nm)) {
        err_fail("fast prefill: launch failed");
      }
      pf_q8(pf_nm, lv, T);
      pf_mm(fm[13], base + p[13], d, lv, T, pf_x1, cur);
    }
    // fbn: h = rms(x1, post), gate / up (silu(g) u), down + x1 into the other x
    llb_dptr nxt = pf_x[cur == pf_x[0] ? 1 : 0];
    u64 post = base + p[1];
    if (!LLB_LAUNCH(gpu_pfrms, T, 1, 1024, &pf_x1, &post, &pf_h, &d)) {
      err_fail("fast prefill: launch failed");
    }
    pf_q8(pf_h, d, T);
    if (fm[2] == LLB_FM_Q4_K && fm[3] == LLB_FM_Q4_K) {
      u64 wg = base + p[2], wu = base + p[3];
      if (!LLB_LAUNCH_SM(gpu_pfq4mg, (T + 63) / 64, (f + 63) / 64, 256, pf_sm(3),
        &wg, &wu, &f, &d, &pf_qx, &pf_xd, &T, &pf_g, &zero)) {
        err_fail("fast prefill: GEMM launch failed");
      }
    } else {
      u32 tf = T * f;
      pf_mm(fm[2], base + p[2], f, d, T, pf_g, 0);
      pf_mm(fm[3], base + p[3], f, d, T, pf_u, 0);
      if (!LLB_LAUNCH(gpu_silu, (tf + 255) / 256, 1, 256, &pf_g, &pf_u, &pf_g, &tf)) {
        err_fail("fast prefill: launch failed");
      }
    }
    pf_q8(pf_g, f, T);
    pf_mm(fm[4], base + p[4], d, f, T, nxt, pf_x1);
    cur = nxt;
  }
  spec_pf(T, cur, sh);
}

// token row er (format fm) at pos into the chunk; a full chunk runs
static void pf_push(u64 er, u32 fm, u32 pos, u32 tok, u32 k, u32 dT, LlbShape sh) {
  if (pf_n == 0) {
    pf_pos0 = pos;
  } else if (pos != pf_pos0 + pf_n) {
    err_fail("fast prefill: a deferred token out of order");
  }
  u32 c4 = sh.d;
  llb_dptr row = pf_x[0] + (llb_dptr)pf_n * sh.d * 4;
  if (!LLB_LAUNCH(fm == LLB_FM_Q4_K ? gpu_embedk : gpu_embed, (c4 + 255) / 256, 1, 256, &er, &c4, &row)) {
    err_fail("device launch failed");
  }
  pf_tk[pf_n] = tok;
  pf_er[pf_n] = er;
  pf_n += 1;
  if (pf_n == pf_T) {
    pf_chunk(pf_n, k, dT, sh);
    pf_n = 0;
  }
}

// the folded step's hook: true = deferred (queued, nothing else to run). o->bsn (pass_mail):
// 1 one deferred token (answered with pf_mark's cue), 2 + n a batch of n ids in x (as F32) from
// pos on, o->w then the embedding's row 0
static bool pf_take(volatile WarpOp* o, u32 pos, u32 dT, LlbShape sh, u64 yp) {
  if (!pf_fast(sh)) {
    return false;
  }
  if (o->n != 0) {
    err_fail("fast prefill: a folded step must start at layer 0");
  }
  u32 rb = LLB_FM_RB(o->bb, sh.d);
  pf_wn = o->k;
  if (o->bsn == 1) {
    pf_push(o->w, o->bb, pos, (u32)((o->w - o->r) / rb), o->d, dT, sh);
    if (!LLB_LAUNCH(gpu_pfmark, 1, 1, 1, &yp)) {
      err_fail("device launch failed");
    }
    return true;
  }
  if (o->bsn >= 2) {
    static float ids[4096];
    u32 n = o->bsn - 2;
    if (n > 4096 || !llb_read(ids, o->x, (size_t)n * 4)) {
      err_fail("fast prefill: cannot read the batch's ids");
    }
    for (u32 q = 0; q < n; q += 1) {
      pf_push(o->w + (u64)ids[q] * rb, o->bb, pos + q, (u32)ids[q], o->d, dT, sh);
    }
    return true;
  }
  if (pf_n != 0) {
    if (pos != pf_pos0 + pf_n) {
      err_fail("fast prefill: the step after the queue out of order");
    }
    pf_chunk(pf_n, o->d, dT, sh);
    pf_n = 0;
  }
  return false;
}

// G3's state dump (LLB_PF_DUMP=dir, LLB_PF_DUMP_POS=p0,p1,..: prompt k of a multi-prompt run
// at pos p_k; default off), from pass_layer before layer i's step at pos: its state over the
// tokens 0..pos-1 into dir/p<k>_s<pos>_<i>.bin, f32 -- full: kc then vc rows 0..pos-1; GDN:
// the conv ring's 4 rows oldest first (token t's row in slot (t + 3) & 3), then S. Read only,
// either tier: the exact step of the last token reaches it after the fast chunks
static int pf_dk = -1;
static void pf_dump(u32 i, bool full, u32 pos, LlbShape sh) {
  static const char* dir;
  static u32 dp[64], nd;
  if (nd == 0) {
    dir = getenv("LLB_PF_DUMP");
    const char* e = getenv("LLB_PF_DUMP_POS");
    for (nd = 1; e && *e && nd < 64; nd += 1) {
      dp[nd] = (u32)strtoul(e, (char**)&e, 10);
      e += *e == ',';
    }
  }
  if (!dir) {
    return;
  }
  if (i == 0 && pos == 0) {
    pf_dk += 1;
  }
  if (pf_dk < 0 || (u32)pf_dk + 1 >= nd || pos != dp[pf_dk + 1] || !pass_on[i]) {
    return;
  }
  char path[512];
  snprintf(path, sizeof path, "%s/p%d_s%u_%u.bin", dir, pf_dk, pos, i);
  FILE* f = fopen(path, "wb");
  u32 cv = 2 * sh.lkh * sh.sd + sh.lvh * sh.sd, kv = sh.nkv * sh.hd;
  llb_dptr src[6];
  size_t n[6], m = 0;
  if (full) {
    src[0] = pass_kc[i]; n[0] = (size_t)pos * kv;
    src[1] = pass_vc[i]; n[1] = (size_t)pos * kv;
    m = 2;
  } else {
    for (u32 j = 0; j < 4; j += 1) {
      src[j] = pass_b[i] + (u64)((pass_r[i] + 3 + j) & 3) * cv * 4; n[j] = cv;
    }
    src[4] = pass_S[i]; n[4] = (size_t)sh.lvh * sh.sd * sh.sd;
    m = 5;
  }
  static float buf[1 << 18];
  for (size_t j = 0; f && j < m; j += 1) {
    for (size_t o = 0; o < n[j]; o += 1 << 18) {
      size_t c = n[j] - o < (1 << 18) ? n[j] - o : 1 << 18;
      if (!llb_read(buf, src[j] + o * 4, c * 4) || fwrite(buf, 4, c, f) != c) {
        err_fail("LLB_PF_DUMP: read / write failed");
      }
    }
  }
  if (!f || fclose(f) != 0) {
    err_fail("LLB_PF_DUMP: cannot write a dump file");
  }
}
