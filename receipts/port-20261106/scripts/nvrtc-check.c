// nvrtc-check FILE CUBE_LOG ARCH: compile an emitted engine C as its own NVRTC source (gpu_make's options), no GPU.
// Prints the result, the log on failure, and which of the named kernels the PTX defines.
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  char* src = calloc(n + 1, 1); if (fread(src, 1, n, f) != (size_t)n) return 2; fclose(f);
  char arch[40], bag[24];
  snprintf(arch, sizeof arch, "--gpu-architecture=%s", argv[3]); snprintf(bag, sizeof bag, "-DCUBE_LOG=%s", argv[2]);
  const char* opts[] = { arch, bag, "--fmad=false", "-default-device" };
  nvrtcProgram p; if (nvrtcCreateProgram(&p, src, "bend.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return 3;
  nvrtcResult r = nvrtcCompileProgram(p, 4, opts);
  size_t ln = 0; nvrtcGetProgramLogSize(p, &ln); char* log = calloc(ln + 1, 1); nvrtcGetProgramLog(p, log);
  printf("nvrtcCompileProgram %s: %s\n", arch, nvrtcGetErrorString(r)); if (r) { printf("%s\n", log); return 1; }
  size_t pn = 0; nvrtcGetPTXSize(p, &pn); char* ptx = calloc(pn + 1, 1); nvrtcGetPTX(p, ptx);
  printf("ptx bytes %zu, log bytes %zu\n", pn, ln);
  const char* k[] = { "llb_attn4", "llb_attn4_sk8", "llb_attn4_sk8_combine", "llb_embed6", "llb_gemv_anyk" };
  for (int i = 0; i < 5; i++) { char e[64]; snprintf(e, sizeof e, ".entry %s(", k[i]); printf("  %-24s %s\n", k[i], strstr(ptx, e) ? "defined" : "absent"); }
  return 0;
}
