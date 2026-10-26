// llb-probe [-q] [--need-mib N]: can this machine run llb-engine? Reads the driver through dlopen("libcuda.so.1")
// (no CUDA toolkit needed) and prints PASS/WARN/FAIL lines in llb doctor's format, each FAIL with a one-line fix.
// -q: FAIL lines only. --need-mib N: FAIL if GPU 0 has less memory than N MiB.
// Exit 0 = runnable (WARN allowed), 1 = a FAIL, 2 = no NVIDIA driver or no CUDA device.
// Build: cc -O2 -o llb-probe llb-probe.c -ldl [-DLLB_FLOOR=12020 -DLLB_FLOOR_R=535] (the floor of the engine it ships with).
// TEST ONLY: LLB_PROBE_FAKE="cc=7.5,managed=0,driver=12020,vram=8,l2=6144" replaces the driver (vram GiB, l2 KiB;
// unset keys keep 8.6 / 1 / 13000 / 24 / 6144) so the refusal paths can be checked on any machine (tools/llb/golden/).
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LLB_FLOOR
#define LLB_FLOOR 13000 // cuDriverGetVersion() the bundled engine needs (its CUDA headers' major.minor)
#endif
#ifndef LLB_FLOOR_R
#define LLB_FLOOR_R 580 // the NVIDIA driver branch that first reports LLB_FLOOR
#endif

static int quiet;
static int fails;
static void pass(const char* m) {
  if (!quiet) printf("PASS %s\n", m);
}
static void fail(const char* m, const char* fix) {
  printf("FAIL %s\n     fix: %s\n", m, fix);
  fails++;
}

int main(int argc, char** argv) {
  long need = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-q")) quiet = 1;
    else if (!strcmp(argv[i], "--need-mib") && i + 1 < argc) need = atol(argv[++i]);
    else {
      fprintf(stderr, "usage: llb-probe [-q] [--need-mib N]\n");
      return 2;
    }
  }
  char   name[256] = "";
  int    drv = 0, n = 0, maj = 0, min = 0, l2 = 0, managed = 0;
  size_t mem = 0;
  const char* fake = getenv("LLB_PROBE_FAKE");
  if (fake) {
    double cc = 8.6, gib = 24;
    drv = 13000, n = 1, managed = 1, l2 = 6144 << 10;
    char  buf[256];
    snprintf(buf, sizeof buf, "%s", fake);
    for (char* kv = strtok(buf, ","); kv; kv = strtok(NULL, ",")) {
      if (!strncmp(kv, "cc=", 3)) cc = atof(kv + 3);
      else if (!strncmp(kv, "managed=", 8)) managed = atoi(kv + 8);
      else if (!strncmp(kv, "driver=", 7)) drv = atoi(kv + 7);
      else if (!strncmp(kv, "vram=", 5)) gib = atof(kv + 5);
      else if (!strncmp(kv, "l2=", 3)) l2 = atoi(kv + 3) << 10;
    }
    maj = (int)cc, min = (int)(cc * 10 + 0.5) % 10, mem = (size_t)(gib * 1073741824.0);
    snprintf(name, sizeof name, "FAKE (LLB_PROBE_FAKE, test only)");
  } else {
    void* h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) {
      printf("FAIL no NVIDIA driver found (libcuda.so.1 cannot be loaded)\n"
             "     fix: install the NVIDIA driver (R%d or newer for this build)\n", LLB_FLOOR_R);
      return 2;
    }
    int (*init)(unsigned)                 = (int (*)(unsigned))dlsym(h, "cuInit");
    int (*version)(int*)                  = (int (*)(int*))dlsym(h, "cuDriverGetVersion");
    int (*count)(int*)                    = (int (*)(int*))dlsym(h, "cuDeviceGetCount");
    int (*dname)(char*, int, int)         = (int (*)(char*, int, int))dlsym(h, "cuDeviceGetName");
    int (*attr)(int*, int, int)           = (int (*)(int*, int, int))dlsym(h, "cuDeviceGetAttribute");
    int (*total)(size_t*, int)            = (int (*)(size_t*, int))dlsym(h, "cuDeviceTotalMem_v2");
    if (!init || !version || !count || !dname || !attr || !total) {
      printf("FAIL the NVIDIA driver's libcuda.so.1 lacks the CUDA driver API\n"
             "     fix: reinstall the NVIDIA driver (R%d or newer for this build)\n", LLB_FLOOR_R);
      return 2;
    }
    version(&drv);
    int e = init(0);
    if (e != 0 || count(&n) != 0 || n == 0) {
      printf("FAIL no CUDA device (cuInit error %d, %d devices; CUDA_VISIBLE_DEVICES=%s)\n"
             "     fix: check nvidia-smi; the driver must see the GPU\n", e, n, getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "unset");
      return 2;
    }
    dname(name, sizeof name, 0);
    attr(&maj, 75, 0); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    attr(&min, 76, 0); // ..._MINOR
    attr(&l2, 38, 0);  // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE (bytes)
    attr(&managed, 89, 0); // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
    total(&mem, 0);
  }
  // the engine's gpu_shape(): a group of 128 lanes per 64 KB of L2, a power of two in 16..128
  int u = l2 >> 16, cube = 0;
  u = u < 16 ? 16 : u > 128 ? 128 : u;
  while (u >> (cube + 1)) cube++;
  long mib = (long)(mem >> 20);
  char m[512], f[256];
  snprintf(m, sizeof m, "GPU 0 of %d: %s, compute %d.%d, %ld MiB, L2 %d KiB (engine bag CUBE_LOG %d)", n, name, maj, min, mib, l2 >> 10, cube);
  pass(m);
  snprintf(m, sizeof m, "driver supports CUDA %d.%d; this engine needs %d.%d or newer", drv / 1000, drv % 1000 / 10, LLB_FLOOR / 1000, LLB_FLOOR % 1000 / 10);
  if (drv >= LLB_FLOOR) pass(m);
  else {
    snprintf(m, sizeof m, "driver supports CUDA %d.%d; this engine needs %d.%d (NVIDIA driver R%d or newer)", drv / 1000, drv % 1000 / 10, LLB_FLOOR / 1000, LLB_FLOOR % 1000 / 10, LLB_FLOOR_R);
    snprintf(f, sizeof f, "update the NVIDIA driver to R%d+%s", LLB_FLOOR_R, LLB_FLOOR >= 13000 ? ", or use the cuda12 bundle (driver R535+)" : "");
    fail(m, f);
  }
  if (maj >= 8) pass("compute capability 8.0 or newer (sm_80+)");
  else {
    snprintf(m, sizeof m, "compute capability %d.%d: needs an Ampere or newer GPU (the kernels use int8 mma.sync and cp.async; NVRTC rejects sm_75)", maj, min);
    fail(m, "run on an RTX 30xx / A100 / A-series or newer GPU");
  }
  if (managed) pass("concurrent managed memory access");
  else fail("the engine needs concurrent managed memory access, which this system does not provide (Windows and WSL2 report none); not supported",
            "run on native Linux");
  if (need > 0) {
    snprintf(m, sizeof m, "GPU memory %ld MiB, the model needs about %ld MiB", mib, need);
    if (mib >= need) pass(m);
    else fail(m, "use a smaller model (llb models) or a GPU with more memory");
  }
  return fails ? 1 : 0;
}
