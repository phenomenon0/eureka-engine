# backends/windows/wincuda.ps1 - the Windows (WDDM) edits to an emitted Bend CUDA runtime (llb-engine.c), XPLAT-WIN.
# Run after winabi.ps1:
#
#   powershell -ExecutionPolicy Bypass -File backends\windows\wincuda.ps1 -In llb-engine.win.c -Out llb-engine.win.c
#
# 1. The device probe. gpu_probe() accepts a device only if CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS is
#    non-zero. Under the WDDM driver model (every consumer GeForce on Windows) that attribute is always 0: the CPU may
#    not touch managed memory while a kernel runs. The engine already obeys that (every host bang runs between
#    synchronized launches), so the corpus still works as managed memory; the probe only has to ask whether managed
#    memory exists at all (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY). Without this the engine says "--gpu on, but this binary
#    found no GPU device" on a working RTX 3090. Measured (RTX 3090, driver 610.88): the 4B and 0.8B pick the same
#    ids as the CPU twin.
#
# 2. The end of a reply (opt-in, LLB_GEN_EOG="248046,248044"). gen_run, the greedy decode run, computes every one of
#    a request's n steps; dual.c needs n ids back, and llb-serve drops everything after the first end-of-generation
#    id. With LLB_GEN_EOG set, the run stops computing at the first listed id and prints that id for the steps left
#    (as engine_cpu does), then hands gen the end step's words with the whole run's count. A 30-token answer to a
#    1024-token request no longer keeps the GPU busy for ~1000 more steps, and a cancelled request drains in
#    milliseconds instead of a restart (llb-serve's Windows drain). Off (unset), the output is byte-identical to the
#    stock run. Safe because llb-serve sends sessionless requests: each starts at position 0, so the uncomputed
#    padding positions are never read.
#
# 3. (below, before the write) the decode run under SYS1.
#
# Each edit's anchor must occur exactly once, or this script fails and writes nothing.
param(
  [Parameter(Mandatory = $true)] [string] $In,
  [Parameter(Mandatory = $true)] [string] $Out
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$text = [IO.File]::ReadAllText((Resolve-Path $In).Path)
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$lines = [System.Collections.Generic.List[string]]::new([regex]::Split($text, "`r?`n"))

function Find-One([string] $line, [string] $what) {
  $hits = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -ceq $line })
  if ($hits.Count -ne 1) {
    Write-Error ("wincuda: expected exactly one line '" + $line + "' in $In, found " + $hits.Count + "; $what changed: re-check it before building")
    exit 1
  }
  return $hits[0]
}

# 1. the probe
$i = Find-One '      CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, gpu_dev);' 'gpu_probe'
$lines[$i] = '      CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, gpu_dev); /* XPLAT-WIN: backends/windows/wincuda.ps1 (WDDM) */'

# 2. the end of a reply
$i = Find-One 'static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {' 'gen_run'
$fn = @(
  '// XPLAT-WIN (backends/windows/wincuda.ps1): LLB_GEN_EOG, the ids (comma-separated) that end a reply',
  'static bool gen_eog(u32 id) {',
  '  static int ne = -1;',
  '  static u32 eog[8];',
  '  if (ne < 0) {',
  '    ne = 0;',
  '    const char* e = getenv("LLB_GEN_EOG");',
  '    while (e != NULL && *e != 0 && ne < 8) {',
  '      char* x;',
  '      unsigned long v = strtoul(e, &x, 10);',
  '      if (x == e) break;',
  '      eog[ne++] = (u32)v;',
  '      e = *x != 0 ? x + 1 : x;',
  '    }',
  '  }',
  '  for (int k = 0; k < ne; k += 1) if (eog[k] == id) return true;',
  '  return false;',
  '}',
  ''
)
$lines.InsertRange($i, [string[]]$fn)

$i = Find-One '    u32 id = (u32)t[1], j = (u32)t[3];' 'gen_run''s pick'
$pad = @(
  '    if (gen_eog(id)) {  // XPLAT-WIN (wincuda.ps1): the reply ended at step q',
  '      // pad the run''s remaining lines with the end id instead of computing them (dual.c wants n ids), then hand gen',
  '      // step q''s words as the run''s last, with the whole run''s count (step q + 1 may be queued: let it finish first)',
  '      if (fflush(stdout) != 0) {',
  '        err_fail("a short write on a standard stream");',
  '      }',
  '      for (u32 r = q; r + 1 < cnt; r += 1) {',
  '        int pm = snprintf(b, sizeof b, "%u\n", id);',
  '        gen_line(stdout, b, (u64)pm);',
  '      }',
  '      if (cuCtxSynchronize() != CUDA_SUCCESS || cuMemcpyHtoD(yp, t, 16) != CUDA_SUCCESS) {',
  '        err_fail("device fault");',
  '      }',
  '      union { float f; u32 u; } pc = { (float)cnt };',
  '      if (cuMemsetD32Async(yp + 16, pc.u, 1, NULL) != CUDA_SUCCESS) {',
  '        err_fail("decode run: cannot set the count");',
  '      }',
  '      return;',
  '    }'
)
$lines.InsertRange($i + 1, [string[]]$pad)

# 3. the decode run under SYS1. dual.c's parse() calls llb_vis_request(NULL, 0) for every request ("no image"), which
#    leaves llb_img_req true with an empty spec, and gen_ok() refused any llb_img_req: served text chat never took the
#    decode run (one host bang a token: 0.8B ~60 tok/s served vs ~126 tok/s for the 4B on the CLI). A request with no
#    image is llb_img_req with an empty llb_img_spec; llb_img_state == 1 (no rows) is still required, so an image
#    request (state 2) keeps one step a bang.
$i = Find-One '    && !llb_hidden_path && !llb_img_dump && !llb_img_req && llb_img_state == 1 && !llb_pfx_on()' 'gen_ok'
$lines[$i] = '    && !llb_hidden_path && !llb_img_dump && (!llb_img_req || llb_img_spec[0] == 0) && llb_img_state == 1 && !llb_pfx_on() /* XPLAT-WIN: wincuda.ps1 */'

[IO.File]::WriteAllText($Out, ($lines -join $nl), (New-Object Text.UTF8Encoding($false)))
Write-Host "wincuda: $In -> $Out (gpu_probe asks for MANAGED_MEMORY; gen_run stops at LLB_GEN_EOG ids)"
exit 0
