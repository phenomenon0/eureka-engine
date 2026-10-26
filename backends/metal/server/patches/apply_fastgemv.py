#!/usr/bin/env python3
"""apply_fastgemv.py IN OUT KERN (E-METAL-LEAD): the emitted C -> + the fast decode K-quant GEMV tier.
KERN = kern_fastgemv.metal (llb_fq4k / llb_fq5k / llb_fq6k), inserted before the d-11 Q8 GEMV body. Host: m9_kgemv takes
them (every Q4_K / Q5_K / Q6_K decode GEMV, the 248320-row head included) when LLB_FASTGEMV=1; unset/0 = the fixed-tree
bd2/rd2 bodies exactly (the CANON, bit-exact tier). One switch, read once."""
import sys
src, out, kern = sys.argv[1:4]
S = open(src).read()
K = open(kern).read()

def rep(old, new, n=1):
    global S
    assert S.count(old) == n, (S.count(old), old[:80])
    S = S.replace(old, new)

anchor = '''// (tools/metal-m27.patch, lever 1) the d-11 Q8 GEMV (the 27B's layer rows:'''
rep(anchor, K + anchor)
rep('''static void m9_kgemv(id<MTLComputeCommandEncoder> en, u32 fm, u64 w, u32 rb,''', '''static id<MTLComputePipelineState> fk_p[3];
// rows a simdgroup / simdgroups a threadgroup per format (Q4_K, Q5_K, Q6_K); LLB_FG="nr4,ns4,nr5,ns5,nr6,ns6" tunes
static u32 fk_nr[3] = { 2, 2, 2 }, fk_ns[3] = { 2, 2, 2 };
static int fk_enabled(void) {
  static int on = -1;
  if (on < 0) {
    const char* s = getenv("LLB_FASTGEMV");
    on = s != NULL && *s != 0 && atoi(s) != 0;
    const char* g = getenv("LLB_FG");
    if (on && g != NULL && *g != 0) {
      u32 v[6];
      if (sscanf(g, "%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
        for (u32 i = 0; i < 3; i += 1) { fk_nr[i] = v[2 * i]; fk_ns[i] = v[2 * i + 1]; }
      }
    }
  }
  return on;
}
static void m9_kgemv(id<MTLComputeCommandEncoder> en, u32 fm, u64 w, u32 rb,''')
G = '''      " Q6_K d 11, Q4_K 17408-col d 13, whole superblock quads, aligned rows)");
  }
'''
rep(G + '''  u32 f = fm == LLB_FM_Q4_K ? 0 : fm == LLB_FM_Q5_K ? 1 : 2;
  u32 k = d == 12 ? 2 : n == 1024 && !q6 ? 1 : 0;''', G + '''  if (fk_enabled()) {
    // (E-METAL-LEAD) the tolerance tier: simdgroup-per-rows, FMA, simd_sum (llb_fq4k / fq5k / fq6k)
    u32 ff = fm == LLB_FM_Q4_K ? 0 : fm == LLB_FM_Q5_K ? 1 : 2;
    if (fk_p[ff] == nil) {
      char fkn[32];
      snprintf(fkn, sizeof fkn, "llb_fq%uk_r%u_s%u", ff + 4, fk_nr[ff], fk_ns[ff]);
      fk_p[ff] = warp_pipe(fkn);
      if (fk_p[ff].maxTotalThreadsPerThreadgroup < 32 * fk_ns[ff]) {
        err_fail("fast GEMV: the threadgroup does not fit");
      }
    }
    [en setBuffer:((x & SA_SCR) != 0 ? gpu_scr : gpu_buf) offset:(x & ~SA_SCR) atIndex:1];
    [en setBuffer:((y & SA_SCR) != 0 ? gpu_scr : gpu_buf) offset:0 atIndex:3];
    WarpArgs wf = { w, y & ~SA_SCR, rb, cols / 4, n, d, 0 };
    [en setComputePipelineState:fk_p[ff]];
    [en setBytes:&wf length:sizeof wf atIndex:2];
    u32 rows = fk_nr[ff] * fk_ns[ff];
    [en dispatchThreadgroups:MTLSizeMake((n + rows - 1) / rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32 * fk_ns[ff], 1, 1)];
    m28_bar(en);
    return;
  }
  u32 f = fm == LLB_FM_Q4_K ? 0 : fm == LLB_FM_Q5_K ? 1 : 2;
  u32 k = d == 12 ? 2 : n == 1024 && !q6 ? 1 : 0;''')
open(out, 'w').write(S)
