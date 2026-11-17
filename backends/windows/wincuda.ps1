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
  '// XPLAT-WIN (wincuda.ps1, 2.b): the cancel request. llb-serve sets Local\llb-engine-cancel-<pid> when a client leaves',
  '// mid-reply; the run then ends at its next step as at an end id, instead of computing the request''s every step (the',
  '// drain took 4.1 s on the 9B). Reset when a run starts, so a late cancel never ends the next request.',
  '__declspec(dllimport) int __stdcall ResetEvent(void* h);',
  'static void* gen_cev;',
  'static bool gen_cancel(void) {',
  '  static int armed;',
  '  if (!armed) {',
  '    char name[64];',
  '    armed = 1;',
  '    snprintf(name, sizeof name, "Local\\llb-engine-cancel-%lu", (unsigned long)GetCurrentProcessId());',
  '    gen_cev = CreateEventA(NULL, 1, 0, name);',
  '  }',
  '  return gen_cev != NULL && WaitForSingleObject(gen_cev, 0) == 0;  /* WAIT_OBJECT_0 */',
  '}',
  ''
)
$lines.InsertRange($i, [string[]]$fn)
$i = Find-One 'static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {' 'gen_run (cancel reset)'
$lines.Insert($i + 1, '  if (gen_cancel()) ResetEvent(gen_cev);  /* XPLAT-WIN (wincuda.ps1, 2.b): a cancel for an earlier request */')

$i = Find-One '    u32 id = (u32)t[1], j = (u32)t[3];' 'gen_run''s pick'
$pad = @(
  '    if (gen_eog(id) || gen_cancel()) {  // XPLAT-WIN (wincuda.ps1): the reply ended (or was cancelled) at step q',
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

# 4. the corpus. Measured (RTX 3090, driver 610.88, Windows 11): a process that ends without cuMemFree on its
#    cuMemAllocManaged block (a plain return, _exit, abort, or a kill) leaves the block's size charged to the system
#    commit with no process left to own it; the charge stays until a reboot (4 GB a run: 14.9 -> 19 -> 23 GB), and the
#    exit itself took 18..117 s. Freed first (cuMemFree) it leaves nothing and the exit takes 0.2 s. So the managed
#    corpus is freed at exit (atexit), in err_fail (_exit), on abort (SIGABRT) and on a crash (an unhandled exception);
#    only a kill (TerminateProcess) still leaks it, so llb-serve ends the engine by closing its stdin. Pinned host memory
#    (cuMemHostAlloc) never leaks, even on a kill, but the device reads it over the bus: 0.8B, a 65-token reply in a
#    200-step request, 2.0 s pinned vs 0.59 s managed. LLB_CORPUS_PINNED=1 picks it.
$i = Find-One 'static void err_fail(const char* msg) {' 'err_fail'
$lines[$i] = 'static void gpu_unmap(void);  // XPLAT-WIN (wincuda.ps1): free a managed corpus before _exit' + $nl + $lines[$i]
$lines.Insert($i + 1, '  gpu_unmap();')

$i = Find-One 'static Corpus gpu_map(u64 bytes) {' 'gpu_map'
$map = @(
  '// XPLAT-WIN (backends/windows/wincuda.ps1): the managed corpus, freed on every exit but a kill (or LLB_CORPUS_PINNED)',
  'static void gpu_unmap(void) {',
  '  CUdeviceptr p = gpu_managed;',
  '  gpu_managed = 0;',
  '  if (p != 0) {',
  '    cuCtxSynchronize();',
  '    cuMemFree(p);',
  '  }',
  '}',
  'typedef struct _EXCEPTION_POINTERS* llbw_xp;',
  '__declspec(dllimport) void* __stdcall SetUnhandledExceptionFilter(void* f);',
  'static long __stdcall gpu_crash(llbw_xp x) {',
  '  (void)x;',
  '  gpu_unmap();',
  '  return 0;  // EXCEPTION_CONTINUE_SEARCH: the crash goes on as before',
  '}',
  'static void gpu_abort(int sig) {',
  '  (void)sig;',
  '  gpu_unmap();',
  '}',
  '// the exit request (edit 6): a kill never frees the managed corpus, so whoever would kill this engine sets the event',
  '// Local\llb-engine-exit-<pid> first; this thread frees the corpus and ends the process, even mid-request',
  '__declspec(dllimport) void* __stdcall CreateEventA(void* sa, int manual, int initial, const char* name);',
  '__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void* h, unsigned long ms);',
  '__declspec(dllimport) void* __stdcall CreateThread(void* sa, size_t stack, unsigned long (__stdcall *fn)(void*), void* arg, unsigned long flags, unsigned long* id);',
  '__declspec(dllimport) unsigned long __stdcall GetCurrentProcessId(void);',
  'static unsigned long __stdcall gpu_exit_watch(void* ev) {',
  '  if (WaitForSingleObject(ev, 0xFFFFFFFFul) == 0) {',
  '    fprintf(stderr, "bend: exit requested: freeing the GPU corpus\n");',
  '    gpu_unmap();',
  '    _exit(3);',
  '  }',
  '  return 0;',
  '}',
  'static void gpu_exit_arm(void) {',
  '  char name[64];',
  '  snprintf(name, sizeof name, "Local\\llb-engine-exit-%lu", GetCurrentProcessId());',
  '  void* ev = CreateEventA(NULL, 1, 0, name);',
  '  if (ev == NULL || CreateThread(NULL, 65536, gpu_exit_watch, ev, 0, NULL) == NULL) {',
  '    fprintf(stderr, "bend: warning: no exit event (%s): a kill would leave the GPU corpus held until a reboot\n", name);',
  '  }',
  '}',
  'static Corpus gpu_map(u64 bytes) {',
  '  const char* m = getenv("LLB_CORPUS_PINNED");',
  '  if (m != NULL && strcmp(m, "1") == 0) {',
  '    void*       h = NULL;',
  '    CUdeviceptr d = 0;',
  '    if (cuMemHostAlloc(&h, bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS) {',
  '      err_fail("corpus reservation failed (pinned host memory: lower --gpu)");',
  '    }',
  '    if (cuMemHostGetDevicePointer(&d, h, 0) != CUDA_SUCCESS || (uintptr_t)d != (uintptr_t)h) {',
  '      err_fail("corpus reservation failed (no unified address for the pinned corpus)");',
  '    }',
  '    return (Corpus)h;',
  '  }',
  '  Corpus c = gpu_map_managed(bytes);',
  '  atexit(gpu_unmap);',
  '  signal(SIGABRT, gpu_abort);',
  '  SetUnhandledExceptionFilter((void*)gpu_crash);',
  '  gpu_exit_arm();',
  '  return c;',
  '}',
  ''
)
$lines[$i] = 'static CUdeviceptr gpu_managed;  // XPLAT-WIN (wincuda.ps1)' + $nl + 'static Corpus gpu_map_managed(u64 bytes) {'
$i = Find-One '  return (Corpus)(uintptr_t)p;' 'gpu_map''s return'
$lines[$i] = '  gpu_managed = p;' + $nl + $lines[$i]
$lines.InsertRange($i + 2, [string[]]$map)

# 6. (inside edit 4's gpu_map) the exit request. A kill is the one exit that still leaks the managed corpus, so the
#    engine names an event, Local\llb-engine-exit-<its pid>; setting it makes a thread free the corpus and end the
#    process (exit code 3), even mid-request. llb-serve sets it before any TerminateProcess and waits for the exit.
#
# 5. the kernel cache. gpu_load maps <exe>.gpu and, when its key is stale, gpu_make rewrites it with fopen("wb") while
#    the old file is still open and mapped. POSIX allows that; Windows refuses to truncate a mapped file, so the write
#    failed silently and every start compiled the kernels again (~80 s on an RTX 3090). Unmap and close it first.
$i = Find-One '    gpu_note(path);' 'gpu_load''s stale branch'
$lines[$i] = '    if (bin != NULL && bin != MAP_FAILED) munmap(bin, st.st_size);  // XPLAT-WIN (wincuda.ps1): Windows keeps a mapped file' + $nl +
  '    if (fd >= 0) close(fd), fd = -1;  // from being rewritten' + $nl + $lines[$i]

# 7. the weight copies. io_put sends a file read into the managed corpus through the copy engine (cuMemcpyHtoD) only
#    from 16 MiB up; smaller ones are a CPU memcpy. Under WDDM a CPU write to managed memory faults page by page:
#    measured ~130 MB/s, 95 ms for each ~13 MB layer of the 0.8B (2.3 s of its 2.5 s load), against 14 ms for a 198 MB
#    copy-engine put. From 64 KiB up every put takes the copy engine.
$i = Find-One '#define io_big(n) (io_gpu && (n) >= 1ull << 24)' 'io_put''s threshold'
$lines[$i] = '#define io_big(n) (io_gpu && (n) >= 1ull << 16)  /* XPLAT-WIN (wincuda.ps1): WDDM managed-memory CPU writes fault */'

[IO.File]::WriteAllText($Out, ($lines -join $nl), (New-Object Text.UTF8Encoding($false)))
Write-Host "wincuda: $In -> $Out (gpu_probe asks for MANAGED_MEMORY; gen_run stops at LLB_GEN_EOG ids; the corpus is freed at exit)"
exit 0
