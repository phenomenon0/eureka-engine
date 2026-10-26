#!/usr/bin/env python3
"""apply_sampler.py IN OUT (MAC-ENGINE-PENALTY): the emitted C -> + the W1-SAMPLER host pick on the fused step's head.
CUDA's smp_note / smp_pick (host/llb_host.c) on Metal. The fused step notes its input token (pos, F.tok) into the
penalty history (every pos: exact per-token prefill and decode). Its head GEMV lands in M9_LOGIT (2^18 floats in
gpu_scr, shared storage). When the request's spec is not argmax-equivalent, the round that carries a decode head
(pos + 1 >= the prompt length) does NOT fold the next interpreter pass into its command buffer (metal-cbfold): the
pass would read the head's four [v, i, w, j] words on the device. Once the buffer has completed, the host reads the
248320-row vocab row, runs llb_smp_sample (host/llb_sampler.h, byte-identical to CUDA's) over the window and rewrites
words 0 / 1; the run loop then runs the pass on its own (gpu_pass). Per request dualm.c sets the spec
(llb_sampler_configure, the "sampling" object mapped as CUDA's dual.c sample_word) and the prompt length
(llb_sampler_prompt). Greedy (no penalty, no bias): nothing here runs, every round folds as before, and the device
argmax stands, so a greedy stream is byte-identical to the build without this patch."""
import sys
src, out = sys.argv[1:3]
S = open(src).read()

def rep(old, new, n=1):
    global S
    assert S.count(old) == n, (S.count(old), old[:80])
    S = S.replace(old, new)

anchor = 'static void llb_hidden_flush(void) {'
rep(anchor, r'''// (MAC-ENGINE-PENALTY, patches/apply_sampler.py) W1-SAMPLER on Metal: CUDA's smp_note / smp_pick for the fused head
#include "llb_sampler_impl.h"
static LlbSmp*      smp;
static LlbSmpParams smp_par;
static int          smp_live;          // 1: the request's spec is not argmax-equivalent, the host picks
static int*         smp_hist;          // pos -> that pos's input token (the fused step's F.tok)
static u32          smp_cap, smp_hn;   // capacity, ids noted (pos 0 .. smp_hn - 1)
static u32          smp_np;            // the prompt length: heads at pos + 1 < smp_np are prefill rows (discarded)
static bool         smp_pend, smp_nopass;
static u32          smp_pos;
static u64          smp_y;             // the head's four words (corpus byte offset)
static int          smp_picks;
// the per-request C ABI (CUDA's llb_sampler_configure): NULL or "" = back to the device argmax; -1 bad spec,
// 0 argmax-equivalent (the device pick stands), 1 the host picks
int llb_sampler_configure(const char* spec) {
  char err[160];
  LlbSmpParams p;
  llb_smp_defaults(&p);
  if (spec && *spec && llb_smp_parse(&p, spec, err, sizeof err) != 0) {
    fprintf(stderr, "%s\n", err);
    return -1;
  }
  llb_smp_free(smp);
  smp = NULL;
  smp_par = p;
  smp_live = !(llb_smp_is_argmax(&smp_par) && smp_par.n_logprobs == 0);
  smp_pend = false;
  return smp_live;
}
void llb_sampler_prompt(int n_prompt) {
  smp_np = (u32)n_prompt;
}
int llb_sampler_picks(void) {
  return smp_picks;
}
// at the fused step's encode: note pos's input token; a decode head (pos + 1 >= smp_np) is marked for the host pick
static void smp_mark(u32 pos, u32 tok, u64 y) {
  if (!smp_live) {
    return;
  }
  if (pos == 0) {
    smp_hn = 0;
  }
  if (pos != smp_hn) {
    fprintf(stderr, "sampler: pos %u after %u noted ids\n", pos, smp_hn);
    err_fail("sampler: a position skipped the fused step (the penalty window needs every input token)");
  }
  if (pos >= smp_cap) {
    u32 c = smp_cap ? smp_cap : 4096;
    while (c <= pos) c *= 2;
    smp_hist = realloc(smp_hist, (size_t)c * sizeof *smp_hist);
    if (!smp_hist) err_fail("sampler: cannot grow the history");
    smp_cap = c;
  }
  smp_hist[pos] = (int)tok;
  smp_hn = pos + 1;
  if (pos + 1 < smp_np) {
    return;
  }
  if (smp_pend) {
    err_fail("sampler: two heads in one round");
  }
  smp_pend = true;
  smp_pos = pos;
  smp_y = y;
}
// after the round's command buffer completed (its next pass held back): the token for smp_pos + 1 from M9_LOGIT
static void smp_flush(Corpus H) {
  if (!smp_pend) {
    return;
  }
  smp_pend = false;
  if (!smp) {
    smp = llb_smp_new(&smp_par, 248320);
    if (!smp) err_fail("sampler: cannot allocate");
  }
  const f32* lg = (const f32*)((const u8*)[gpu_scr contents] + (M9_LOGIT & ~SA_SCR));
  f32* y = (f32*)((u8*)H + smp_y);
  llb_smp_history(smp, smp_hist, (int)smp_pos + 1);
  int t = llb_smp_sample(smp, lg, (uint64_t)smp_pos + 1);
  y[0] = lg[t];
  y[1] = (f32)t;
  smp_picks += 1;
}
''' + anchor)

# the fused step's head (the decode and exact per-token prefill arm): note + mark it
rep('''      m24_fz_head(en, &F, m24_fin, M9_A(y), grid, true);
      if (llb_hidden_path && pos < llb_hidden_tokens && llb_hidden_positions[pos]) {''',
'''      m24_fz_head(en, &F, m24_fin, M9_A(y), grid, true);
      smp_mark(pos, F.tok, M9_A(y));
      if (llb_hidden_path && pos < llb_hidden_tokens && llb_hidden_positions[pos]) {''')

# the round's end: a marked head holds the folded pass back (it would read the head's words on the device)
rep('''    gpu_enc = en;
    gpu_pass_enc();
    [en endEncoding];''', '''    smp_nopass = smp_pend;
    if (!smp_nopass) {
      gpu_enc = en;
      gpu_pass_enc();
    }
    [en endEncoding];''')

# after the wait: pick
rep('''    llb_hidden_flush();
    xc_flush();''', '''    llb_hidden_flush();
    smp_flush(H);
    xc_flush();''')

# the run loop: a held-back pass runs on its own, after the pick
rep('''      if (!warp_serve(H)) {
        err_fail("frontier drained without a result");
      }
      ran = true;''', '''      if (!warp_serve(H)) {
        err_fail("frontier drained without a result");
      }
      ran = !smp_nopass;
      smp_nopass = false;''')

open(out, 'w').write(S)
