#!/usr/bin/env python3
"""backends/linux/linuxcuda.py IN.c OUT.c - Linux desktop CUDA runtime edits (LINUX-PARITY + CUDA-V2).

Ports the portable serving edits from backends/windows/wincuda.ps1 plus Linux-native writable NVRTC caching and
instant SIGUSR1 mid-generation cancel:
  1. LLB_GEN_EOG (opt-in, comma-separated ids, e.g. "248046,248044"): gen_run stops computing at the first listed id
     and pads the run's remaining lines with it so short replies do not keep the GPU busy for remaining max_tokens.
  2. SIGUSR1 mid-generation cancel: llb-serve sends SIGUSR1 when a client disconnects or hits Stop; gen_run ends at its
     next step (~15-25 ms) instead of killing and reloading the whole model. Armed with SA_RESTART so stdio never sees EINTR.
  3. gen_ok under SYS1: dual.c's parse() calls llb_vis_request(NULL, 0) for every request, leaving llb_img_req set with
     an empty spec; allow gen_run when llb_img_spec[0] == 0 so served text chat takes the CUDA-graphed decode run.
  4. Writable XDG NVRTC kernel cache: gpu_load checks $BEND_GPU_CACHE_DIR -> $XDG_CACHE_HOME/eureka/nvrtc ->
     $HOME/.cache/eureka/nvrtc (keyed by gpu_hash + sm_XY) first, falls back to bundled <exe>.gpu for instant startup
     on matching GPUs, and writes newly compiled cubins to the writable user cache (never failing on read-only /usr/...).
  5. Copy-engine weight puts from 64 KiB up (wincuda.ps1 edit 7): avoids page-by-page CPU faults into managed memory.
Each anchor must occur exactly once, or this script fails and writes nothing.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().split("\n")


def one(line, what):
    hits = [i for i, l in enumerate(lines) if l == line]
    if len(hits) != 1:
        sys.exit(f"linuxcuda: expected exactly one line {line!r} in {src}, found {len(hits)}; {what} changed: re-check it")
    return hits[0]


# 1 + 2: LLB_GEN_EOG early stop + SIGUSR1 mid-generation cancel
i = one("static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {", "gen_run")
lines[i:i] = """// LINUX-PARITY (backends/linux/linuxcuda.py): LLB_GEN_EOG + SIGUSR1 mid-generation cancel
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
static volatile sig_atomic_t gen_cancelled;
static void gen_cancel_sig(int sig) {
  (void)sig;
  gen_cancelled = 1;
}
static bool gen_cancel(void) {
  static int armed;
  if (!armed) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = gen_cancel_sig;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, NULL);
    armed = 1;
  }
  return gen_cancelled != 0;
}
""".split("\n")

i = one("static void gen_run(volatile WarpOp* o, u64 yp, u32 grid, LlbShape sh, u32 dT, u32 pos, u32 n) {", "gen_run (cancel reset)")
lines.insert(i + 1, "  gen_cancel(); gen_cancelled = 0;  /* LINUX-CUDA-V2: arm SIGUSR1 and clear any stale cancel */")

i = one("    u32 id = (u32)t[1], j = (u32)t[3];", "gen_run's pick")
lines[i + 1:i + 1] = """    if (gen_eog(id) || gen_cancel()) {  // LINUX-CUDA-V2: reply ended or SIGUSR1 cancelled at step q
      gen_cancelled = 0;
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

# 3: gen_ok under SYS1
i = one("    && !llb_hidden_path && !llb_img_dump && !llb_img_req && llb_img_state == 1 && !llb_pfx_on()", "gen_ok")
lines[i] = ("    && !llb_hidden_path && !llb_img_dump && (!llb_img_req || llb_img_spec[0] == 0) && llb_img_state == 1"
            " && !llb_pfx_on() /* LINUX-PARITY: linuxcuda.py (wincuda.ps1 edit 3) */")

# 4: Writable XDG NVRTC kernel cache (reads user cache -> bundled <exe>.gpu -> compiles once to writable user cache)
i = one("  cuDeviceTotalMem(&span, gpu_dev);", "gpu_span body")
# Find end of gpu_span right after this line
assert lines[i + 1].strip() == "return span;" and lines[i + 2].strip() == "}", "unexpected gpu_span layout"
lines[i + 3:i + 3] = """
// LINUX-CUDA-V2: writable per-user NVRTC kernel cache keyed by gpu_hash() and sm_XY
static void gpu_mkdirs(const char* dir) {
  char tmp[4096];
  size_t n = strlen(dir);
  if (n == 0 || n >= sizeof tmp) return;
  memcpy(tmp, dir, n + 1);
  for (size_t k = 1; k < n; k += 1) {
    if (tmp[k] == '/') {
      tmp[k] = 0;
      mkdir(tmp, 0755);
      tmp[k] = '/';
    }
  }
  mkdir(tmp, 0700);
}
static const char* gpu_user_cache_path(void) {
  static char out[4096];
  char base[3800];
  const char* explicit_dir = getenv("BEND_GPU_CACHE_DIR");
  if (explicit_dir != NULL && explicit_dir[0] != 0) {
    if (snprintf(base, sizeof base, "%s", explicit_dir) >= (int)sizeof base) return NULL;
  } else {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg != NULL && xdg[0] == '/') {
      if (snprintf(base, sizeof base, "%s/eureka/nvrtc", xdg) >= (int)sizeof base) return NULL;
    } else {
      const char* home = getenv("HOME");
      if (home == NULL || home[0] != '/') return NULL;
      if (snprintf(base, sizeof base, "%s/.cache/eureka/nvrtc", home) >= (int)sizeof base) return NULL;
    }
  }
  gpu_mkdirs(base);
  int cc[2] = {0, 0};
  cuDeviceGetAttribute(cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gpu_dev);
  cuDeviceGetAttribute(cc + 1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gpu_dev);
  u64 h = gpu_hash();
  if (snprintf(out, sizeof out, "%s/engine_gpu-%016llx-sm%d%d.gpu", base, (unsigned long long)h, cc[0], cc[1]) >= (int)sizeof out) {
    return NULL;
  }
  return out;
}
static bool gpu_try_load_file(const char* p) {
  if (p == NULL) return false;
  int fd = open(p, O_RDONLY);
  struct stat st = { 0 };
  if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 8) {
    if (fd >= 0) close(fd);
    return false;
  }
  char* bin = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (bin == NULL || bin == MAP_FAILED) return false;
  u64 key = 0;
  memcpy(&key, bin, 8);
  bool ok = (key == gpu_hash()) && (cuModuleLoadData(&gpu_lib, bin + 8) == CUDA_SUCCESS);
  munmap(bin, (size_t)st.st_size);
  return ok;
}
""".split("\n")

i_path = one("  const char* path = gpu_path();", "gpu_load path")
i_make = one("    gpu_make(path);", "gpu_load gpu_make")
assert i_make > i_path and (i_make - i_path) < 20, f"unexpected gpu_load span {i_path}..{i_make}"
lines[i_path:i_make + 2] = """  const char* cache = gpu_user_cache_path();
  const char* bundled = gpu_path();
  if (!gpu_try_load_file(cache) && !gpu_try_load_file(bundled)) {
    const char* dest = cache != NULL ? cache : bundled;
    gpu_note(dest);
    gpu_make(dest);
  }""".split("\n")

# 5: Copy-engine weight puts from 64 KiB up (wincuda.ps1 edit 7)
i = one("#define io_big(n) (io_gpu && (n) >= 1ull << 24)", "io_put's threshold")
lines[i] = "#define io_big(n) (io_gpu && (n) >= 1ull << 16)  /* LINUX-CUDA-V2 (wincuda.ps1 edit 7): copy engine from 64 KiB */"

open(dst, "w", encoding="utf-8").write("\n".join(lines))
print(f"linuxcuda: {src} -> {dst} (LLB_GEN_EOG + SIGUSR1 cancel + SYS1 gen_run + XDG NVRTC cache + 64K io_big)")
