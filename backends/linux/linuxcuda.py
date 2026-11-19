#!/usr/bin/env python3
"""backends/linux/linuxcuda.py IN.c OUT.c - the portable edits of backends/windows/wincuda.ps1, for Linux (LINUX-PARITY).

The served engine needs two of wincuda.ps1's seven edits on every OS; the rest are WDDM / Win32 only:
  2. LLB_GEN_EOG (opt-in, comma-separated ids, e.g. "248046,248044"): gen_run stops computing at the first listed id and
     pads the run's remaining lines with it, so a 30-token answer to a 1024-step request does not keep the GPU busy for
     ~1000 more steps. Unset, the output is byte-identical to the stock run. (Windows' cancel event, 2.b, is left out:
     POSIX llb-serve cancels by killing the engine child.)
  3. gen_ok under SYS1: dual.c's parse() calls llb_vis_request(NULL, 0) for every request, which left llb_img_req set
     with an empty spec, and gen_ok() refused the decode run for every served text request (one host bang a token).
Each anchor must occur exactly once, or this script fails and writes nothing (as wincuda.ps1 does).
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().split("\n")

def one(line, what):
    hits = [i for i, l in enumerate(lines) if l == line]
    if len(hits) != 1:
        sys.exit(f"linuxcuda: expected exactly one line {line!r} in {src}, found {len(hits)}; {what} changed: re-check it")
    return hits[0]

i = one("static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {", "gen_run")
lines[i:i] = """// LINUX-PARITY (backends/linux/linuxcuda.py = wincuda.ps1 edit 2): LLB_GEN_EOG, the ids (comma-separated) that end a reply
static bool gen_eog(u32 id) {
  static int ne = -1;
  static u32 eog[8];
  if (ne < 0) {
    ne = 0;
    const char* e = getenv("LLB_GEN_EOG");
    while (e != NULL && *e != 0 && ne < 8) {
      char* x;
      unsigned long v = strtoul(e, &x, 10);
      if (x == e) break;
      eog[ne++] = (u32)v;
      e = *x != 0 ? x + 1 : x;
    }
  }
  for (int k = 0; k < ne; k += 1) if (eog[k] == id) return true;
  return false;
}
""".split("\n")

i = one("    u32 id = (u32)t[1], j = (u32)t[3];", "gen_run's pick")
lines[i + 1:i + 1] = """    if (gen_eog(id)) {  // LINUX-PARITY (linuxcuda.py): the reply ended at step q
      // pad the run's remaining lines with the end id instead of computing them (dual.c wants n ids), then hand gen
      // step q's words as the run's last, with the whole run's count (step q + 1 may be queued: let it finish first)
      if (fflush(stdout) != 0) {
        err_fail("a short write on a standard stream");
      }
      for (u32 r = q; r + 1 < cnt; r += 1) {
        int pm = snprintf(b, sizeof b, "%u\\n", id);
        gen_line(stdout, b, (u64)pm);
      }
      if (cuCtxSynchronize() != CUDA_SUCCESS || cuMemcpyHtoD(yp, t, 16) != CUDA_SUCCESS) {
        err_fail("device fault");
      }
      union { float f; u32 u; } pc = { (float)cnt };
      if (cuMemsetD32Async(yp + 16, pc.u, 1, NULL) != CUDA_SUCCESS) {
        err_fail("decode run: cannot set the count");
      }
      return;
    }""".split("\n")

i = one("    && !llb_hidden_path && !llb_img_dump && !llb_img_req && llb_img_state == 1 && !llb_pfx_on()", "gen_ok")
lines[i] = ("    && !llb_hidden_path && !llb_img_dump && (!llb_img_req || llb_img_spec[0] == 0) && llb_img_state == 1"
            " && !llb_pfx_on() /* LINUX-PARITY: linuxcuda.py (wincuda.ps1 edit 3) */")

open(dst, "w", encoding="utf-8").write("\n".join(lines))
print(f"linuxcuda: {src} -> {dst} (gen_run stops at LLB_GEN_EOG ids; served text takes the decode run)")
