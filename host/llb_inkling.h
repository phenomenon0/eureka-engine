// host/llb_inkling.h — Native C11 CUDA runtime for Inkling 2 Mini (`inkling2_loop`)
// Compiled directly inside Eureka's ONE `engine_gpu` binary (TRUE SINGLE-ENGINE PURITY:
// zero CMake, zero cuBLAS, zero vendored Strata binary!).
// Lazily compiles embedded `host/llb_inkling.cuh` via NVRTC on first `--inkling` call and
// caches cubin at ~/.cache/eureka/nvrtc/inkling-<hash>-smXY.gpu (zero cost for Qwen!).
#ifndef LLB_INKLING_H
#define LLB_INKLING_H

#include <cuda.h>
#include <nvrtc.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "llb_inkling_fmt.h"
#include "llb_inkling_cpu.h"

static const char ink_cuh_src[] = {
#embed "llb_inkling.cuh"
  , 0
};

typedef struct {
  CUdevice   dev;
  CUcontext  ctx;
  CUstream   stream;
  CUmodule   mod;
  CUfunction fn_embed_norm;
  CUfunction fn_rms_norm;
  CUfunction fn_attn_res_mix;
  CUfunction fn_ctrl_cat_rms;
  CUfunction fn_ctrl_down;
  CUfunction fn_ctrl_apply;
  CUfunction fn_linear_m;
  CUfunction fn_post_qkv_fused;
  CUfunction fn_rel_table;
  CUfunction fn_flash_gqa;
  CUfunction fn_sconv4_seq;
  CUfunction fn_moe_route_topk8;
  CUfunction fn_moe_gu_situ;
  CUfunction fn_moe_dn_reduce;
  CUfunction fn_softcap_argmax;
  CUfunction fn_scr_build_bf16;
  CUfunction fn_scr_pass1_i8;
} InkCuda;

typedef struct {
  uint64_t ptr;
  uint32_t dn_off;
  uint8_t  gu_code;
  uint8_t  dn_code;
  uint16_t pad;
} InkExpDevHost;

typedef struct {
  const float* p[8];
  int count;
} InkDepthPtrsHost;

static uint64_t ink_fnv64(const char* s, size_t n) {
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 1099511628211ull;
  return h;
}

static void ink_mkdirs(const char* dir) {
  char tmp[4096];
  size_t n = strlen(dir);
  if (n == 0 || n >= sizeof tmp) return;
  memcpy(tmp, dir, n + 1);
  for (size_t k = 1; k < n; k++) {
    if (tmp[k] == '/') { tmp[k] = 0; mkdir(tmp, 0755); tmp[k] = '/'; }
  }
  mkdir(tmp, 0700);
}

static int ink_cuda_init(InkCuda* ic, char* err, size_t err_cap) {
  memset(ic, 0, sizeof *ic);
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&ic->dev, 0) != CUDA_SUCCESS) {
    snprintf(err, err_cap, "cuInit/cuDeviceGet failed");
    return -1;
  }
  if (cuCtxGetCurrent(&ic->ctx) != CUDA_SUCCESS || ic->ctx == NULL) {
    if (cuDevicePrimaryCtxRetain(&ic->ctx, ic->dev) != CUDA_SUCCESS || cuCtxSetCurrent(ic->ctx) != CUDA_SUCCESS) {
      snprintf(err, err_cap, "cuDevicePrimaryCtxRetain failed");
      return -1;
    }
  }
  if (cuStreamCreate(&ic->stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
    snprintf(err, err_cap, "cuStreamCreate failed");
    return -1;
  }
  int maj = 0, min = 0;
  cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ic->dev);
  cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ic->dev);

  size_t src_len = sizeof(ink_cuh_src) - 1;
  uint64_t h = ink_fnv64(ink_cuh_src, src_len);

  // Check XDG NVRTC cache first
  char cache_dir[3800] = {0}, cache_path[4096] = {0};
  const char* ex = getenv("BEND_GPU_CACHE_DIR");
  const char* xdg = getenv("XDG_CACHE_HOME");
  const char* home = getenv("HOME");
  if (ex && *ex) snprintf(cache_dir, sizeof cache_dir, "%s", ex);
  else if (xdg && xdg[0] == '/') snprintf(cache_dir, sizeof cache_dir, "%s/eureka/nvrtc", xdg);
  else if (home && home[0] == '/') snprintf(cache_dir, sizeof cache_dir, "%s/.cache/eureka/nvrtc", home);
  if (cache_dir[0]) {
    ink_mkdirs(cache_dir);
    snprintf(cache_path, sizeof cache_path, "%s/inkling-%016llx-sm%d%d.gpu",
      cache_dir, (unsigned long long)h, maj, min);
    int fd = open(cache_path, O_RDONLY);
    struct stat st;
    if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 8) {
      void* mp = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (mp && mp != MAP_FAILED) {
        uint64_t key = 0;
        memcpy(&key, mp, 8);
        if (key == h && cuModuleLoadData(&ic->mod, (const char*)mp + 8) == CUDA_SUCCESS) {
          munmap(mp, (size_t)st.st_size);
          goto load_fns;
        }
        munmap(mp, (size_t)st.st_size);
      }
    } else if (fd >= 0) close(fd);
  }

  // Compile lazily via NVRTC (zero CMake / zero nvcc at runtime)
  {
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, ink_cuh_src, "llb_inkling.cuh", 0, NULL, NULL) != NVRTC_SUCCESS) {
      snprintf(err, err_cap, "nvrtcCreateProgram failed");
      return -1;
    }
    char arch[32];
    snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", maj, min);
    const char* opts[] = { arch, "--use_fast_math", "-default-device" };
    nvrtcResult rc = nvrtcCompileProgram(prog, 2, opts);
    if (rc != NVRTC_SUCCESS) {
      size_t log_sz = 0;
      nvrtcGetProgramLogSize(prog, &log_sz);
      char* log = (char*)calloc(log_sz + 1, 1);
      if (log) { nvrtcGetProgramLog(prog, log); fprintf(stderr, "inkling NVRTC log:\n%s\n", log); free(log); }
      nvrtcDestroyProgram(&prog);
      snprintf(err, err_cap, "NVRTC compile failed for sm_%d%d", maj, min);
      return -1;
    }
    size_t bin_sz = 0;
    int use_cubin = (nvrtcGetCUBINSize(prog, &bin_sz) == NVRTC_SUCCESS && bin_sz > 0);
    if (!use_cubin) nvrtcGetPTXSize(prog, &bin_sz);
    char* bin = (char*)malloc(bin_sz + 1);
    if (use_cubin) nvrtcGetCUBIN(prog, bin);
    else nvrtcGetPTX(prog, bin);
    nvrtcDestroyProgram(&prog);

    if (cuModuleLoadData(&ic->mod, bin) != CUDA_SUCCESS) {
      free(bin);
      snprintf(err, err_cap, "cuModuleLoadData failed");
      return -1;
    }
    if (cache_path[0]) {
      char tmp_p[4128];
      snprintf(tmp_p, sizeof tmp_p, "%s.tmp.%ld", cache_path, (long)getpid());
      int fd = open(tmp_p, O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (fd >= 0) {
        ssize_t w1 = write(fd, &h, 8);
        ssize_t w2 = write(fd, bin, bin_sz);
        close(fd);
        if (w1 == 8 && w2 == (ssize_t)bin_sz) rename(tmp_p, cache_path);
        else unlink(tmp_p);
      }
    }
    free(bin);
  }

load_fns:
  {
    struct { CUfunction* fn; const char* name; } syms[] = {
      { &ic->fn_embed_norm,      "ink_embed_norm" },
      { &ic->fn_rms_norm,        "ink_rms_norm" },
      { &ic->fn_attn_res_mix,    "ink_attn_res_mix" },
      { &ic->fn_ctrl_cat_rms,    "ink_ctrl_cat_rms" },
      { &ic->fn_ctrl_down,       "ink_ctrl_down" },
      { &ic->fn_ctrl_apply,      "ink_ctrl_apply" },
      { &ic->fn_linear_m,        "ink_linear_m" },
      { &ic->fn_post_qkv_fused,  "ink_post_qkv_fused" },
      { &ic->fn_rel_table,       "ink_rel_table" },
      { &ic->fn_flash_gqa,       "ink_flash_gqa" },
      { &ic->fn_sconv4_seq,      "ink_sconv4_seq" },
      { &ic->fn_moe_route_topk8, "ink_moe_route_topk8" },
      { &ic->fn_moe_gu_situ,     "ink_moe_gu_situ" },
      { &ic->fn_moe_dn_reduce,   "ink_moe_dn_reduce" },
      { &ic->fn_softcap_argmax,  "ink_softcap_argmax" },
      { &ic->fn_scr_build_bf16,  "ink_scr_build_bf16" },
      { &ic->fn_scr_pass1_i8,    "ink_scr_pass1_i8" },
    };
    for (size_t i = 0; i < sizeof(syms) / sizeof(syms[0]); i++) {
      if (cuModuleGetFunction(syms[i].fn, ic->mod, syms[i].name) != CUDA_SUCCESS) {
        snprintf(err, err_cap, "missing inkling kernel %s", syms[i].name);
        return -1;
      }
    }
  }
  return 0;
}

static inline CUdeviceptr ink_dalloc(size_t n) {
  CUdeviceptr p = 0;
  if (n == 0) n = 16;
  if (cuMemAlloc(&p, n) != CUDA_SUCCESS) return 0;
  cuMemsetD8(p, 0, n);
  return p;
}

static inline float ink_max_abs_diff(const float* a, const float* b, size_t n) {
  float mx = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > mx) mx = d;
  }
  return mx;
}

// ============================================================================
// Full CPU-vs-GPU Numerical Oracle + CUDA Graph Self-Check (`--inkling --selfcheck`)
// Verifies EVERY SINGLE Inkling 2 Mini CUDA kernel against FP32 CPU reference on GPU!
// ============================================================================
static int ink_gpu_selfcheck(void) {
  InkCuda ic;
  char err[256] = {0};
  if (ink_cuda_init(&ic, err, sizeof err) != 0) {
    fprintf(stderr, "inkling selfcheck: cuda init failed: %s\n", err);
    return 1;
  }
  int fails = 0;
  const int M = 3, D = 2304;
  float *h_x = (float*)malloc((size_t)M * D * 4), *h_y_cpu = (float*)malloc((size_t)M * D * 4), *h_y_gpu = (float*)malloc((size_t)M * D * 4);
  float *h_w = (float*)malloc((size_t)D * 4);
  for (int i = 0; i < M * D; i++) h_x[i] = 0.35f * sinf((float)(i + 1) * 0.017f);
  for (int i = 0; i < D; i++) h_w[i] = 1.0f + 0.15f * cosf((float)i * 0.031f);

  // 1. RMSNorm + EmbedNorm (BF16)
  {
    int V = 32, fmt = INK_F_BF16;
    uint16_t* h_emb = (uint16_t*)malloc((size_t)V * D * 2);
    for (int i = 0; i < V * D; i++) {
      float f = 0.4f * sinf((float)(i + 3) * 0.011f);
      uint32_t u; memcpy(&u, &f, 4); h_emb[i] = (uint16_t)(u >> 16);
    }
    int32_t ids[3] = { 2, 17, 31 };
    ink_cpu_embed_norm((const uint8_t*)h_emb, fmt, V, h_w, ids, h_y_cpu, M, D);
    CUdeviceptr d_emb = ink_dalloc((size_t)V * D * 2), d_w = ink_dalloc((size_t)D * 4);
    CUdeviceptr d_ids = ink_dalloc(sizeof ids), d_out = ink_dalloc((size_t)M * D * 4);
    cuMemcpyHtoD(d_emb, h_emb, (size_t)V * D * 2); cuMemcpyHtoD(d_w, h_w, (size_t)D * 4); cuMemcpyHtoD(d_ids, ids, sizeof ids);
    int m_a = M, d_a = D;
    void* args[] = { &d_emb, &fmt, &V, &d_w, &d_ids, &d_out, &m_a, &d_a };
    cuLaunchKernel(ic.fn_embed_norm, (unsigned)M, 1, 1, 256, 1, 1, 0, ic.stream, args, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_y_gpu, d_out, (size_t)M * D * 4);
    float diff = ink_max_abs_diff(h_y_cpu, h_y_gpu, (size_t)M * D);
    fprintf(stderr, "  [1/9] ink_embed_norm (BF16, M=%d, D=%d): max_abs_err=%.2e %s\n", M, D, diff, diff < 1e-5f ? "PASS" : "FAIL");
    if (diff >= 1e-5f) fails++;
    cuMemFree(d_emb); cuMemFree(d_w); cuMemFree(d_ids); cuMemFree(d_out); free(h_emb);
  }

  // 2. Kimi AttnRes Depth Mix (S=4 sources, n_q=3 fused queries + RMSNorm)
  {
    int S = 4, nq = 3;
    float* srcs[4];
    CUdeviceptr d_srcs[4];
    for (int s = 0; s < S; s++) {
      srcs[s] = (float*)malloc((size_t)M * D * 4);
      for (int i = 0; i < M * D; i++) srcs[s][i] = 0.25f * sinf((float)(i + 7 * s + 1) * 0.019f);
      d_srcs[s] = ink_dalloc((size_t)M * D * 4);
      cuMemcpyHtoD(d_srcs[s], srcs[s], (size_t)M * D * 4);
    }
    float* h_q = (float*)malloc((size_t)nq * D * 4);
    for (int i = 0; i < nq * D; i++) h_q[i] = 0.1f * cosf((float)(i + 1) * 0.029f);
    float* h_out_cpu = (float*)malloc((size_t)nq * M * D * 4);
    float* h_out_gpu = (float*)malloc((size_t)nq * M * D * 4);
    ink_cpu_attn_res_mix((const float* const*)srcs, S, h_q, h_out_cpu, nq, M, D, h_w, h_w, NULL);
    InkDepthPtrsHost st; memset(&st, 0, sizeof st); st.count = S;
    for (int s = 0; s < S; s++) st.p[s] = (const float*)d_srcs[s];
    CUdeviceptr d_q = ink_dalloc((size_t)nq * D * 4), d_out = ink_dalloc((size_t)nq * M * D * 4);
    CUdeviceptr d_w = ink_dalloc((size_t)D * 4), d_nul = 0;
    cuMemcpyHtoD(d_q, h_q, (size_t)nq * D * 4); cuMemcpyHtoD(d_w, h_w, (size_t)D * 4);
    int m_a = M, d_a = D;
    void* args[] = { &st, &d_q, &d_out, &nq, &m_a, &d_a, &d_w, &d_w, &d_nul };
    cuLaunchKernel(ic.fn_attn_res_mix, (unsigned)M, 1, 1, 256, 1, 1, 0, ic.stream, args, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_out_gpu, d_out, (size_t)nq * M * D * 4);
    float diff = ink_max_abs_diff(h_out_cpu, h_out_gpu, (size_t)nq * M * D);
    fprintf(stderr, "  [2/9] ink_attn_res_mix (S=%d, n_q=%d, M=%d): max_abs_err=%.2e %s\n", S, nq, M, diff, diff < 1e-5f ? "PASS" : "FAIL");
    if (diff >= 1e-5f) fails++;
    for (int s = 0; s < S; s++) { free(srcs[s]); cuMemFree(d_srcs[s]); }
    free(h_q); free(h_out_cpu); free(h_out_gpu); cuMemFree(d_q); cuMemFree(d_out); cuMemFree(d_w);
  }

  // 3. Rank-64 tanh LoopControl (cat_rms + down + apply)
  {
    int rank = 64, D2 = 2 * D;
    float* h_msg = (float*)malloc((size_t)M * D * 4);
    float* h_wd = (float*)malloc((size_t)rank * D2 * 4);
    float* h_wg = (float*)malloc((size_t)D * rank * 4);
    float* h_wf = (float*)malloc((size_t)D * rank * 4);
    for (int i = 0; i < M * D; i++) h_msg[i] = 0.3f * cosf((float)(i + 5) * 0.023f);
    for (int i = 0; i < rank * D2; i++) h_wd[i] = 0.01f * sinf((float)(i + 1) * 0.013f);
    for (int i = 0; i < D * rank; i++) { h_wg[i] = 0.02f * cosf((float)i * 0.017f); h_wf[i] = 0.02f * sinf((float)i * 0.019f); }
    ink_cpu_loop_ctrl(h_x, h_msg, h_wd, h_wg, h_wf, h_y_cpu, M, D, rank);
    CUdeviceptr d_x = ink_dalloc((size_t)M * D * 4), d_msg = ink_dalloc((size_t)M * D * 4);
    CUdeviceptr d_cat = ink_dalloc((size_t)M * D2 * 4), d_gate = ink_dalloc((size_t)M * rank * 4);
    CUdeviceptr d_wd = ink_dalloc((size_t)rank * D2 * 4), d_wg = ink_dalloc((size_t)D * rank * 4), d_wf = ink_dalloc((size_t)D * rank * 4);
    CUdeviceptr d_out = ink_dalloc((size_t)M * D * 4);
    cuMemcpyHtoD(d_x, h_x, (size_t)M * D * 4); cuMemcpyHtoD(d_msg, h_msg, (size_t)M * D * 4);
    cuMemcpyHtoD(d_wd, h_wd, (size_t)rank * D2 * 4); cuMemcpyHtoD(d_wg, h_wg, (size_t)D * rank * 4); cuMemcpyHtoD(d_wf, h_wf, (size_t)D * rank * 4);
    int m_a = M, d_a = D;
    void* a1[] = { &d_x, &d_msg, &d_cat, &m_a, &d_a };
    void* a2[] = { &d_cat, &d_wd, &d_gate, &m_a, &D2, &rank };
    void* a3[] = { &d_x, &d_msg, &d_gate, &d_wg, &d_wf, &d_out, &m_a, &d_a, &rank };
    cuLaunchKernel(ic.fn_ctrl_cat_rms, (unsigned)M, 1, 1, 256, 1, 1, 0, ic.stream, a1, NULL);
    cuLaunchKernel(ic.fn_ctrl_down, (unsigned)rank, (unsigned)M, 1, 256, 1, 1, 0, ic.stream, a2, NULL);
    cuLaunchKernel(ic.fn_ctrl_apply, (unsigned)((D + 255) / 256), (unsigned)M, 1, 256, 1, 1, 0, ic.stream, a3, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_y_gpu, d_out, (size_t)M * D * 4);
    float diff = ink_max_abs_diff(h_y_cpu, h_y_gpu, (size_t)M * D);
    fprintf(stderr, "  [3/9] ink_loop_ctrl (rank=%d, M=%d, D=%d): max_abs_err=%.2e %s\n", rank, M, D, diff, diff < 1e-5f ? "PASS" : "FAIL");
    if (diff >= 1e-5f) fails++;
    free(h_msg); free(h_wd); free(h_wg); free(h_wf);
    cuMemFree(d_x); cuMemFree(d_msg); cuMemFree(d_cat); cuMemFree(d_gate); cuMemFree(d_wd); cuMemFree(d_wg); cuMemFree(d_wf); cuMemFree(d_out);
  }

  // 4. Unified Multi-Token Linear GEMV (Q8_ROW + S5_K)
  {
    int N = 256, K = 2304, fmt = INK_F_Q8ROW;
    size_t wb = (size_t)N * 4 + (size_t)N * K;
    uint8_t* h_W = (uint8_t*)malloc(wb);
    float* scs = (float*)h_W;
    int8_t* q8 = (int8_t*)(h_W + (size_t)N * 4);
    for (int r = 0; r < N; r++) {
      scs[r] = 0.004f * (1.0f + 0.1f * (float)(r & 7));
      for (int k = 0; k < K; k++) q8[(size_t)r * K + k] = (int8_t)(((r * 31 + k * 17) % 241) - 120);
    }
    float* yc = (float*)malloc((size_t)M * N * 4);
    float* yg = (float*)malloc((size_t)M * N * 4);
    ink_cpu_linear_m(h_W, fmt, N, N, K, h_x, yc, M);
    CUdeviceptr d_W = ink_dalloc(wb), d_X = ink_dalloc((size_t)M * K * 4), d_Y = ink_dalloc((size_t)M * N * 4);
    cuMemcpyHtoD(d_W, h_W, wb); cuMemcpyHtoD(d_X, h_x, (size_t)M * K * 4);
    int m_a = M;
    void* args[] = { &d_W, &fmt, &N, &N, &K, &d_X, &d_Y, &m_a };
    cuLaunchKernel(ic.fn_linear_m, (unsigned)((N + 3) / 4), 1, 1, 128, 1, 1, 0, ic.stream, args, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(yg, d_Y, (size_t)M * N * 4);
    float diff = ink_max_abs_diff(yc, yg, (size_t)M * N);
    fprintf(stderr, "  [4/9] ink_linear_m (Q8_ROW, N=%d, K=%d, M=%d): max_abs_err=%.2e %s\n", N, K, M, diff, diff < 1e-4f ? "PASS" : "FAIL");
    if (diff >= 1e-4f) fails++;
    free(h_W); free(yc); free(yg); cuMemFree(d_W); cuMemFree(d_X); cuMemFree(d_Y);
  }

  // 5. Fused Post-QKV (ShortConv4 + L2-clip + V-RMSNorm + Global tau + FP16 KV) + RelPos + Sink Flash-GQA
  {
    int kv_cap = 32, is_swa = 0, pos0 = 0, q_sc_n = 128, r_sc_n = 16;
    float* q_c = (float*)malloc((size_t)M * 24 * 128 * 4);
    float* q_g = (float*)malloc((size_t)M * 24 * 128 * 4);
    float* k_r = (float*)malloc((size_t)M * 1024 * 4);
    float* v_r = (float*)malloc((size_t)M * 1024 * 4);
    float* sw_k = (float*)malloc(1024 * 4 * 4);
    float* sw_v = (float*)malloc(1024 * 4 * 4);
    float* k_dyn = (float*)malloc((size_t)M * 32 * 4);
    float* v_dyn = (float*)malloc((size_t)M * 32 * 4);
    float* hk_c = (float*)calloc(3 * 1024, 4);
    float* hv_c = (float*)calloc(3 * 1024, 4);
    float* r_c = (float*)malloc((size_t)M * 24 * 16 * 4);
    float* r_g = (float*)malloc((size_t)M * 24 * 16 * 4);
    float* q_sc = (float*)malloc(128 * 4);
    float* r_sc = (float*)malloc(16 * 4);
    float* rel_p = (float*)malloc(16 * 1024 * 4);
    float* sinks = (float*)malloc((size_t)M * 24 * 4);
    float* gate = (float*)malloc((size_t)M * 24 * 128 * 4);
    uint16_t* kvk_c = (uint16_t*)calloc((size_t)kv_cap * 1024, 2);
    uint16_t* kvv_c = (uint16_t*)calloc((size_t)kv_cap * 1024, 2);
    float* out_c = (float*)malloc((size_t)M * 24 * 128 * 4);
    float* out_g = (float*)malloc((size_t)M * 24 * 128 * 4);
    for (int i = 0; i < M * 24 * 128; i++) { q_c[i] = q_g[i] = 0.5f * sinf((float)i * 0.013f); gate[i] = 0.2f * cosf((float)i * 0.011f); }
    for (int i = 0; i < M * 1024; i++) { k_r[i] = 0.4f * cosf((float)i * 0.017f); v_r[i] = 0.3f * sinf((float)i * 0.021f); }
    for (int i = 0; i < 4096; i++) { sw_k[i] = 0.25f * sinf((float)i * 0.07f); sw_v[i] = 0.25f * cosf((float)i * 0.05f); }
    for (int i = 0; i < M * 32; i++) { k_dyn[i] = 0.1f * sinf((float)i); v_dyn[i] = 0.1f * cosf((float)i); }
    for (int i = 0; i < M * 24 * 16; i++) r_c[i] = r_g[i] = 0.3f * sinf((float)i * 0.03f);
    for (int i = 0; i < 128; i++) q_sc[i] = 0.9f + 0.002f * (float)i;
    for (int i = 0; i < 16; i++) r_sc[i] = 0.8f + 0.02f * (float)i;
    for (int i = 0; i < 16 * 1024; i++) rel_p[i] = 0.05f * cosf((float)i * 0.01f);
    for (int i = 0; i < M * 24; i++) sinks[i] = -0.3f + 0.05f * (float)(i & 7);

    ink_cpu_post_qkv_and_flash_gqa(q_c, q_sc, q_sc_n, k_r, v_r, sw_k, sw_v, k_dyn, v_dyn,
      hk_c, hv_c, r_c, r_sc, r_sc_n, rel_p, sinks, gate, kvk_c, kvv_c, kv_cap, is_swa, pos0, M, out_c);

    CUdeviceptr dq = ink_dalloc((size_t)M * 24 * 128 * 4), dqsc = ink_dalloc(128 * 4);
    CUdeviceptr dkr = ink_dalloc((size_t)M * 1024 * 4), dvr = ink_dalloc((size_t)M * 1024 * 4);
    CUdeviceptr dswk = ink_dalloc(4096 * 4), dswv = ink_dalloc(4096 * 4);
    CUdeviceptr dkd = ink_dalloc((size_t)M * 32 * 4), dvd = ink_dalloc((size_t)M * 32 * 4);
    CUdeviceptr dhk = ink_dalloc(3 * 1024 * 4), dhv = ink_dalloc(3 * 1024 * 4);
    CUdeviceptr dr = ink_dalloc((size_t)M * 24 * 16 * 4), drsc = ink_dalloc(16 * 4);
    CUdeviceptr dkvk = ink_dalloc((size_t)kv_cap * 1024 * 2), dkvv = ink_dalloc((size_t)kv_cap * 1024 * 2);
    CUdeviceptr drelp = ink_dalloc(16 * 1024 * 4), drelt = ink_dalloc((size_t)M * 24 * 1025 * 4);
    CUdeviceptr dsink = ink_dalloc((size_t)M * 24 * 4), dgate = ink_dalloc((size_t)M * 24 * 128 * 4), dout = ink_dalloc((size_t)M * 24 * 128 * 4);
    cuMemcpyHtoD(dq, q_g, (size_t)M * 24 * 128 * 4); cuMemcpyHtoD(dqsc, q_sc, 128 * 4);
    cuMemcpyHtoD(dkr, k_r, (size_t)M * 1024 * 4); cuMemcpyHtoD(dvr, v_r, (size_t)M * 1024 * 4);
    cuMemcpyHtoD(dswk, sw_k, 4096 * 4); cuMemcpyHtoD(dswv, sw_v, 4096 * 4);
    cuMemcpyHtoD(dkd, k_dyn, (size_t)M * 32 * 4); cuMemcpyHtoD(dvd, v_dyn, (size_t)M * 32 * 4);
    cuMemcpyHtoD(dr, r_g, (size_t)M * 24 * 16 * 4); cuMemcpyHtoD(drsc, r_sc, 16 * 4);
    cuMemcpyHtoD(drelp, rel_p, 16 * 1024 * 4); cuMemcpyHtoD(dsink, sinks, (size_t)M * 24 * 4); cuMemcpyHtoD(dgate, gate, (size_t)M * 24 * 128 * 4);
    int m_a = M;
    void* apq[] = { &dq, &dqsc, &q_sc_n, &dkr, &dvr, &dswk, &dswv, &dkd, &dvd, &dhk, &dhv, &dr, &drsc, &r_sc_n, &dkvk, &dkvv, &kv_cap, &is_swa, &pos0, &m_a };
    void* art[] = { &dr, &drelp, &drelt, &m_a };
    void* afg[] = { &dq, &drelt, &dsink, &dgate, &dkvk, &dkvv, &kv_cap, &is_swa, &pos0, &dout, &m_a };
    cuLaunchKernel(ic.fn_post_qkv_fused, 64, 1, 1, 128, 1, 1, 0, ic.stream, apq, NULL);
    cuLaunchKernel(ic.fn_rel_table, 24, (unsigned)M, 1, 256, 1, 1, 0, ic.stream, art, NULL);
    cuLaunchKernel(ic.fn_flash_gqa, 24, (unsigned)M, 1, 128, 1, 1, 0, ic.stream, afg, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(out_g, dout, (size_t)M * 24 * 128 * 4);
    float diff = ink_max_abs_diff(out_c, out_g, (size_t)M * 24 * 128);
    fprintf(stderr, "  [5/9] ink_post_qkv + rel_table + flash_gqa (24:8, M=%d): max_abs_err=%.2e %s\n", M, diff, diff < 1e-5f ? "PASS" : "FAIL");
    if (diff >= 1e-5f) fails++;
    free(q_c); free(q_g); free(k_r); free(v_r); free(sw_k); free(sw_v); free(k_dyn); free(v_dyn);
    free(hk_c); free(hv_c); free(r_c); free(r_g); free(q_sc); free(r_sc); free(rel_p); free(sinks); free(gate);
    free(kvk_c); free(kvv_c); free(out_c); free(out_g);
    cuMemFree(dq); cuMemFree(dqsc); cuMemFree(dkr); cuMemFree(dvr); cuMemFree(dswk); cuMemFree(dswv);
    cuMemFree(dkd); cuMemFree(dvd); cuMemFree(dhk); cuMemFree(dhv); cuMemFree(dr); cuMemFree(drsc);
    cuMemFree(dkvk); cuMemFree(dkvv); cuMemFree(drelp); cuMemFree(drelt); cuMemFree(dsink); cuMemFree(dgate); cuMemFree(dout);
  }

  // 6. Post-Attn / Post-MLP ShortConv4 + residual add
  {
    float* sw = (float*)malloc((size_t)D * 4 * 4);
    float* h_c = (float*)calloc((size_t)3 * D, 4);
    for (int i = 0; i < D * 4; i++) sw[i] = 0.2f * sinf((float)i * 0.03f);
    memcpy(h_y_cpu, h_x, (size_t)M * D * 4);
    ink_cpu_sconv4_seq(h_x, sw, h_c, h_y_cpu, M, D, 1);
    CUdeviceptr dx = ink_dalloc((size_t)M * D * 4), dsw = ink_dalloc((size_t)D * 4 * 4);
    CUdeviceptr dh = ink_dalloc((size_t)3 * D * 4), dd = ink_dalloc((size_t)M * D * 4);
    cuMemcpyHtoD(dx, h_x, (size_t)M * D * 4); cuMemcpyHtoD(dsw, sw, (size_t)D * 4 * 4); cuMemcpyHtoD(dd, h_x, (size_t)M * D * 4);
    int m_a = M, c_a = D, add = 1;
    void* args[] = { &dx, &dsw, &dh, &dd, &m_a, &c_a, &add };
    cuLaunchKernel(ic.fn_sconv4_seq, (unsigned)((D + 255) / 256), 1, 1, 256, 1, 1, 0, ic.stream, args, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_y_gpu, dd, (size_t)M * D * 4);
    float diff = ink_max_abs_diff(h_y_cpu, h_y_gpu, (size_t)M * D);
    fprintf(stderr, "  [6/9] ink_sconv4_seq (M=%d, C=%d, add_res=1): max_abs_err=%.2e %s\n", M, D, diff, diff < 1e-5f ? "PASS" : "FAIL");
    if (diff >= 1e-5f) fails++;
    free(sw); free(h_c); cuMemFree(dx); cuMemFree(dsw); cuMemFree(dh); cuMemFree(dd);
  }

  // 7. MoE Router (Top-8/128) + SiTU(4,25) Gate+Up + Down Reduce (SF4 NVFP4 + MX8 MXFP8 mixed!)
  {
    float* rw = (float*)malloc(128 * D * 4);
    float* rb = (float*)malloc(128 * 4);
    for (int i = 0; i < 128 * D; i++) rw[i] = 0.02f * sinf((float)(i + 1) * 0.011f);
    for (int i = 0; i < 128; i++) rb[i] = 0.05f * cosf((float)i * 0.37f);
    int32_t idx_c[3 * 8], idx_g[3 * 8];
    float wt_c[3 * 8], wt_g[3 * 8];
    float rsc = 8.0f;
    ink_cpu_moe_route_topk8(h_x, rw, rb, idx_c, wt_c, rsc, M, D);
    CUdeviceptr dx = ink_dalloc((size_t)M * D * 4), drw = ink_dalloc(128 * D * 4), drb = ink_dalloc(128 * 4);
    CUdeviceptr didx = ink_dalloc(sizeof idx_g), dwt = ink_dalloc(sizeof wt_g);
    cuMemcpyHtoD(dx, h_x, (size_t)M * D * 4); cuMemcpyHtoD(drw, rw, 128 * D * 4); cuMemcpyHtoD(drb, rb, 128 * 4);
    int m_a = M, d_a = D;
    void* ar[] = { &dx, &drw, &drb, &didx, &dwt, &rsc, &m_a, &d_a };
    cuLaunchKernel(ic.fn_moe_route_topk8, (unsigned)M, 1, 1, 128, 1, 1, 0, ic.stream, ar, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(idx_g, didx, sizeof idx_g); cuMemcpyDtoH(wt_g, dwt, sizeof wt_g);
    int idx_eq = memcmp(idx_c, idx_g, sizeof idx_c) == 0;
    float wdiff = ink_max_abs_diff(wt_c, wt_g, (size_t)M * 8);

    // Build 2 synthetic experts (1 SF4 NVFP4 + 1 MX8 MXFP8) and alias all 128 slots to them
    size_t sf4_gu_b = 16 + (size_t)2304 * (2304 / 2) + (size_t)2304 * (2304 / 16);
    size_t sf4_dn_b = (size_t)2304 * (1152 / 2) + (size_t)2304 * (1152 / 16);
    size_t sf4_tot = sf4_gu_b + sf4_dn_b;
    uint8_t* h_sf4 = (uint8_t*)calloc(sf4_tot, 1);
    float hdr[4] = { 0.05f, 1.0f, 0.04f, 1.0f };
    memcpy(h_sf4, hdr, 16);
    for (size_t i = 16; i < sf4_tot; i++) h_sf4[i] = (uint8_t)((i * 73 + 19) & 0x37);
    size_t mx8_gu_b = (size_t)2304 * 2304 + (size_t)2304 * (2304 / 32);
    size_t mx8_dn_b = (size_t)2304 * 1152 + (size_t)2304 * (1152 / 32);
    size_t mx8_tot = mx8_gu_b + mx8_dn_b;
    uint8_t* h_mx8 = (uint8_t*)calloc(mx8_tot, 1);
    for (size_t i = 0; i < (size_t)2304 * 2304; i++) h_mx8[i] = (uint8_t)((i * 37) & 0x3F);
    for (size_t i = 0; i < (size_t)2304 * (2304 / 32); i++) h_mx8[(size_t)2304 * 2304 + i] = 123;
    for (size_t i = 0; i < (size_t)2304 * 1152; i++) h_mx8[mx8_gu_b + i] = (uint8_t)((i * 53) & 0x3F);
    for (size_t i = 0; i < (size_t)2304 * (1152 / 32); i++) h_mx8[mx8_gu_b + (size_t)2304 * 1152 + i] = 123;

    CUdeviceptr d_sf4 = ink_dalloc(sf4_tot), d_mx8 = ink_dalloc(mx8_tot);
    cuMemcpyHtoD(d_sf4, h_sf4, sf4_tot); cuMemcpyHtoD(d_mx8, h_mx8, mx8_tot);
    InkCpuExp cexps[128]; InkExpDevHost dexps_h[128];
    for (int e = 0; e < 128; e++) {
      if ((e & 1) == 0) {
        cexps[e] = (InkCpuExp){ h_sf4, (uint32_t)sf4_gu_b, INK_F_SF4, INK_F_SF4 };
        dexps_h[e] = (InkExpDevHost){ (uint64_t)d_sf4, (uint32_t)sf4_gu_b, INK_F_SF4, INK_F_SF4, 0 };
      } else {
        cexps[e] = (InkCpuExp){ h_mx8, (uint32_t)mx8_gu_b, INK_F_MX8, INK_F_MX8 };
        dexps_h[e] = (InkExpDevHost){ (uint64_t)d_mx8, (uint32_t)mx8_gu_b, INK_F_MX8, INK_F_MX8, 0 };
      }
    }
    ink_cpu_moe_exec(cexps, idx_c, wt_c, h_x, h_y_cpu, M);
    CUdeviceptr dexps = ink_dalloc(sizeof dexps_h), dact = ink_dalloc((size_t)M * 8 * 1152 * 4), dout = ink_dalloc((size_t)M * D * 4);
    cuMemcpyHtoD(dexps, dexps_h, sizeof dexps_h);
    void* agu[] = { &dexps, &didx, &dx, &dact, &m_a };
    void* adn[] = { &dexps, &didx, &dwt, &dact, &dout, &m_a };
    cuLaunchKernel(ic.fn_moe_gu_situ, 1152 / 4, (unsigned)(M * 8), 1, 128, 1, 1, 0, ic.stream, agu, NULL);
    cuLaunchKernel(ic.fn_moe_dn_reduce, 2304 / 4, (unsigned)M, 1, 128, 1, 1, 0, ic.stream, adn, NULL);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_y_gpu, dout, (size_t)M * D * 4);
    float mdiff = ink_max_abs_diff(h_y_cpu, h_y_gpu, (size_t)M * D);
    fprintf(stderr, "  [7/9] ink_moe (Top-8/128 SiTU SF4+MX8, M=%d): idx_eq=%d w_err=%.2e out_err=%.2e %s\n",
      M, idx_eq, wdiff, mdiff, (idx_eq && wdiff < 1e-5f && mdiff < 1e-3f) ? "PASS" : "FAIL");
    if (!idx_eq || wdiff >= 1e-5f || mdiff >= 1e-3f) fails++;
    free(rw); free(rb); free(h_sf4); free(h_mx8);
    cuMemFree(dx); cuMemFree(drw); cuMemFree(drb); cuMemFree(didx); cuMemFree(dwt);
    cuMemFree(d_sf4); cuMemFree(d_mx8); cuMemFree(dexps); cuMemFree(dact); cuMemFree(dout);
  }

  // 8. Exact BF16 -> INT8 Cauchy-Schwarz Screened Unembed + Softcap(30) Argmax
  {
    int V = 4096;
    uint16_t* h_W = (uint16_t*)malloc((size_t)V * D * 2);
    for (int i = 0; i < V * D; i++) {
      float f = 0.08f * sinf((float)(i + 11) * 0.0137f);
      uint32_t u; memcpy(&u, &f, 4); h_W[i] = (uint16_t)(u >> 16);
    }
    float* lg_c = (float*)malloc((size_t)V * 4);
    ink_cpu_linear_m((const uint8_t*)h_W, INK_F_BF16, V, V, D, h_x, lg_c, 1);
    float bv_c = 0; int32_t bi_c = ink_cpu_softcap_argmax(lg_c, V, 30.0f, &bv_c);
    CUdeviceptr dW = ink_dalloc((size_t)V * D * 2), dX = ink_dalloc((size_t)D * 4), dLG = ink_dalloc((size_t)V * 4);
    CUdeviceptr dq8 = ink_dalloc((size_t)V * D), dsc = ink_dalloc((size_t)V * 4), deps = ink_dalloc((size_t)V * 4);
    CUdeviceptr dhat = ink_dalloc((size_t)V * 4), dlmax = ink_dalloc(4), did = ink_dalloc(4), dval = ink_dalloc(4);
    cuMemcpyHtoD(dW, h_W, (size_t)V * D * 2); cuMemcpyHtoD(dX, h_x, (size_t)D * 4);
    int d_a = D, one = 1, fmt = INK_F_BF16; float cap = 30.0f;
    void* ab[] = { &dW, &dq8, &dsc, &deps, &V, &d_a };
    cuLaunchKernel(ic.fn_scr_build_bf16, (unsigned)((V + 7) / 8), 1, 1, 256, 1, 1, 0, ic.stream, ab, NULL);
    // Compute exact h_norm on host for pass1 bound verification
    double hn2 = 0.0; for (int i = 0; i < D; i++) hn2 += (double)h_x[i] * h_x[i];
    float h_norm = sqrtf((float)hn2);
    uint32_t init_lb = 0x7f7fffffu ^ 0x7fffffffu; // -FLT_MAX ordered
    cuMemcpyHtoD(dlmax, &init_lb, 4);
    void* ap1[] = { &dq8, &dsc, &deps, &dX, &h_norm, &dhat, &dlmax, &V, &d_a };
    cuLaunchKernel(ic.fn_scr_pass1_i8, (unsigned)((V + 7) / 8), 1, 1, 256, 1, 1, 0, ic.stream, ap1, NULL);
    void* alm[] = { &dW, &fmt, &V, &V, &d_a, &dX, &dLG, &one };
    void* aam[] = { &dLG, &V, &cap, &did, &dval };
    cuLaunchKernel(ic.fn_linear_m, (unsigned)((V + 3) / 4), 1, 1, 128, 1, 1, 0, ic.stream, alm, NULL);
    cuLaunchKernel(ic.fn_softcap_argmax, 1, 1, 1, 256, 1, 1, 0, ic.stream, aam, NULL);
    cuStreamSynchronize(ic.stream);
    int32_t bi_g = -1; float bv_g = 0;
    cuMemcpyDtoH(&bi_g, did, 4); cuMemcpyDtoH(&bv_g, dval, 4);
    // Verify Cauchy-Schwarz bound: true argmax MUST survive pass1!
    float *h_hat = (float*)malloc((size_t)V * 4), *h_eps = (float*)malloc((size_t)V * 4);
    uint32_t lb_ord = 0;
    cuMemcpyDtoH(h_hat, dhat, (size_t)V * 4); cuMemcpyDtoH(h_eps, deps, (size_t)V * 4); cuMemcpyDtoH(&lb_ord, dlmax, 4);
    uint32_t lb_u = lb_ord ^ ((lb_ord >> 31) ? 0x7fffffffu : 0u);
    float lmax = 0; memcpy(&lmax, &lb_u, 4);
    int survivors = 0, true_survived = 0;
    for (int v = 0; v < V; v++) {
      if (h_hat[v] + h_eps[v] * h_norm >= lmax) {
        survivors++;
        if (v == bi_c) true_survived = 1;
      }
    }
    int ok = (bi_c == bi_g && fabsf(bv_c - bv_g) < 1e-4f && true_survived && survivors < V / 10);
    fprintf(stderr, "  [8/9] ink_bf16_screen + softcap_argmax (V=%d): argmax=%d==%d survivors=%d/%d (%.1f%%) true_kept=%d %s\n",
      V, bi_g, bi_c, survivors, V, 100.0 * survivors / V, true_survived, ok ? "PASS" : "FAIL");
    if (!ok) fails++;
    free(h_W); free(lg_c); free(h_hat); free(h_eps);
    cuMemFree(dW); cuMemFree(dX); cuMemFree(dLG); cuMemFree(dq8); cuMemFree(dsc); cuMemFree(deps);
    cuMemFree(dhat); cuMemFree(dlmax); cuMemFree(did); cuMemFree(dval);
  }

  // 9. CUDA Graph Capture & Replay Exact Bit-Parity
  {
    CUdeviceptr dx = ink_dalloc((size_t)D * 4), dw = ink_dalloc((size_t)D * 4), dy1 = ink_dalloc((size_t)D * 4), dy2 = ink_dalloc((size_t)D * 4);
    cuMemcpyHtoD(dx, h_x, (size_t)D * 4); cuMemcpyHtoD(dw, h_w, (size_t)D * 4);
    int one = 1, d_a = D;
    void* a1[] = { &dx, &dw, &dy1, &one, &d_a };
    cuLaunchKernel(ic.fn_rms_norm, 1, 1, 1, 256, 1, 1, 0, ic.stream, a1, NULL);
    cuStreamSynchronize(ic.stream);
    CUgraph gr = NULL; CUgraphExec ge = NULL;
    void* a2[] = { &dx, &dw, &dy2, &one, &d_a };
    cuStreamBeginCapture(ic.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    cuLaunchKernel(ic.fn_rms_norm, 1, 1, 1, 256, 1, 1, 0, ic.stream, a2, NULL);
    cuStreamEndCapture(ic.stream, &gr);
    cuGraphInstantiate(&ge, gr, 0);
    cuGraphLaunch(ge, ic.stream);
    cuStreamSynchronize(ic.stream);
    cuMemcpyDtoH(h_y_cpu, dy1, (size_t)D * 4);
    cuMemcpyDtoH(h_y_gpu, dy2, (size_t)D * 4);
    int bit_eq = memcmp(h_y_cpu, h_y_gpu, (size_t)D * 4) == 0;
    fprintf(stderr, "  [9/9] ink_cuda_graph capture+replay: bit_exact=%d %s\n", bit_eq, bit_eq ? "PASS" : "FAIL");
    if (!bit_eq) fails++;
    cuGraphExecDestroy(ge); cuGraphDestroy(gr);
    cuMemFree(dx); cuMemFree(dw); cuMemFree(dy1); cuMemFree(dy2);
  }

  free(h_x); free(h_y_cpu); free(h_y_gpu); free(h_w);
  fprintf(stderr, "inkling-gpu selfcheck: %s (%d failures)\n", fails == 0 ? "ALL 9/9 PASS" : "FAILED", fails);
  return fails == 0 ? 0 : 1;
}

// ============================================================================
// Full Native GPU Inkling 2 Mini Engine (`--inkling MODEL.strata`)
// Loop-Weighted Tiered VRAM + Pinned-Host MoE Residency + 48-Visit Forward Pass
// + SYS1 JSON-Line Streaming Server (matching llb-serve's dual.c contract)
// ============================================================================
typedef struct {
  CUdeviceptr ptr;
  int         fmt;
  int64_t     rows;
  int64_t     cols;
  int         is_host;
} InkDevTensor;

typedef struct {
  InkModelMeta  meta;
  InkCuda       ic;
  InkDevTensor  trunk[INK_MAX_TRUNK_T];
  CUdeviceptr   d_exp_tab[INK_N_PHYS];   // [128] InkExpDevHost per physical layer
  uint64_t      vram_exp_bytes;
  uint64_t      host_exp_bytes;
  int           n_exp_vram;
  int           n_exp_host;
  // KV caches + ShortConv4 histories + AttnRes depth checkpoints
  CUdeviceptr   kv_k[INK_N_VISITS];
  CUdeviceptr   kv_v[INK_N_VISITS];
  CUdeviceptr   conv_k[INK_N_VISITS], conv_v[INK_N_VISITS];
  CUdeviceptr   conv_a[INK_N_VISITS], conv_m[INK_N_VISITS];
  int           kv_global_cap;
  // Activation scratch (supports M = 1..8 batched verify/prefill)
  CUdeviceptr   d_ids, d_x, d_x_norm, d_blk_hist[8], d_res_out;
  CUdeviceptr   d_q, d_k, d_v, d_r, d_gate, d_sinks, d_kd, d_vd, d_rel_tbl, d_attn_out, d_o_out;
  CUdeviceptr   d_ctrl_cat, d_ctrl_gate, d_ctrl_out;
  CUdeviceptr   d_top_idx, d_top_w, d_moe_act, d_moe_out;
  CUdeviceptr   d_logits, d_argmax_id, d_argmax_val;
  // Optional BF16 -> INT8 Screened Unembed plane
  CUdeviceptr   d_scr_q8, d_scr_sc, d_scr_eps, d_scr_hat, d_scr_lmax;
  int           has_bf16_screen;
} InkGpuRuntime;

static const InkDevTensor* ink_rt_tensor(const InkGpuRuntime* rt, const char* name) {
  for (int i = 0; i < rt->meta.n_trunk; i++) {
    if (!strcmp(rt->meta.trunk[i].name, name)) return &rt->trunk[i];
  }
  return NULL;
}

static int ink_rt_load(InkGpuRuntime* rt, const char* path, int ctx_cap, char* err, size_t err_cap) {
  memset(rt, 0, sizeof *rt);
  if (ink_open_strata(&rt->meta, path, err, err_cap) != 0) return -1;
  if (ink_cuda_init(&rt->ic, err, err_cap) != 0) return -1;
  rt->kv_global_cap = ctx_cap > 0 ? ctx_cap : 8192;

  // 1. Upload all trunk tensors to VRAM (Section 1 base)
  uint64_t s1_off = rt->meta.sections[1].file_offset;
  for (int i = 0; i < rt->meta.n_trunk; i++) {
    const InkTensor* t = &rt->meta.trunk[i];
    size_t nb = (size_t)t->bytes;
    void* buf = malloc(nb);
    if (!buf || ink_read_at(&rt->meta, s1_off + t->off, buf, nb) != 0) {
      free(buf); snprintf(err, err_cap, "failed reading trunk tensor %s", t->name); return -1;
    }
    CUdeviceptr dp = ink_dalloc(nb);
    if (!dp || cuMemcpyHtoD(dp, buf, nb) != CUDA_SUCCESS) {
      free(buf); snprintf(err, err_cap, "OOM uploading trunk tensor %s (%zu B)", t->name, nb); return -1;
    }
    free(buf);
    rt->trunk[i] = (InkDevTensor){ dp, t->fmt, t->rows, t->cols, 0 };
  }

  // 2. Loop-Weighted Tiered VRAM + Pinned-Host MoE Residency (Section 3 vram_load_order!)
  // Core looped layers (6..17, visited 3x per token) come first in vram_load_order!
  size_t free_vram = 0, total_vram = 0;
  cuMemGetInfo(&free_vram, &total_vram);
  size_t reserve = (size_t)rt->meta.cfg.n_global_visits * rt->kv_global_cap * 1024 * 4
                 + (size_t)rt->meta.cfg.n_swa_visits * 1024 * 1024 * 4 + (768ull << 20);
  size_t vram_budget = free_vram > reserve ? (free_vram - reserve) : 0;

  uint64_t s4_off = rt->meta.sections[4].file_offset;
  int n_phys = rt->meta.cfg.n_phys, n_exp = rt->meta.cfg.n_experts;
  InkExpDevHost* host_tab = (InkExpDevHost*)calloc((size_t)n_phys * n_exp, sizeof(InkExpDevHost));
  for (int k = 0; k < rt->meta.n_experts_dir; k++) {
    int idx = (int)rt->meta.vram_order[k];
    if (idx < 0 || idx >= rt->meta.n_experts_dir) continue;
    const InkExpertDirEntry* e = &rt->meta.experts[idx];
    size_t nb = (size_t)e->body_bytes;
    CUdeviceptr dp = 0;
    if (rt->vram_exp_bytes + nb <= vram_budget && cuMemAlloc(&dp, nb) == CUDA_SUCCESS) {
      void* tmp = malloc(nb);
      if (!tmp || ink_read_at(&rt->meta, s4_off + e->body_offset, tmp, nb) != 0) {
        free(tmp); snprintf(err, err_cap, "failed reading expert %d", idx); free(host_tab); return -1;
      }
      cuMemcpyHtoD(dp, tmp, nb);
      free(tmp);
      rt->vram_exp_bytes += nb;
      rt->n_exp_vram++;
    } else {
      void* hp = NULL;
      if (cuMemHostAlloc(&hp, nb, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS
        || cuMemHostGetDevicePointer(&dp, hp, 0) != CUDA_SUCCESS
        || ink_read_at(&rt->meta, s4_off + e->body_offset, hp, nb) != 0) {
        snprintf(err, err_cap, "failed pinning overflow expert %d (%zu B)", idx, nb);
        free(host_tab);
        return -1;
      }
      rt->host_exp_bytes += nb;
      rt->n_exp_host++;
    }
    host_tab[idx] = (InkExpDevHost){ (uint64_t)dp, e->dn_offset, e->gu_code_id, e->dn_code_id, 0 };
  }
  for (int l = 0; l < n_phys; l++) {
    rt->d_exp_tab[l] = ink_dalloc((size_t)n_exp * sizeof(InkExpDevHost));
    cuMemcpyHtoD(rt->d_exp_tab[l], host_tab + (size_t)l * n_exp, (size_t)n_exp * sizeof(InkExpDevHost));
  }
  free(host_tab);

  // 3. Allocate 48-visit KV caches + ShortConv4 states + scratch buffers
  for (int v = 0; v < rt->meta.cfg.n_visits; v++) {
    int cap = rt->meta.cfg.visit_is_swa[v] ? INK_SWA_WIN : rt->kv_global_cap;
    rt->kv_k[v] = ink_dalloc((size_t)cap * 1024 * 2);
    rt->kv_v[v] = ink_dalloc((size_t)cap * 1024 * 2);
    rt->conv_k[v] = ink_dalloc(3 * 1024 * 4);
    rt->conv_v[v] = ink_dalloc(3 * 1024 * 4);
    rt->conv_a[v] = ink_dalloc(3 * INK_HIDDEN * 4);
    rt->conv_m[v] = ink_dalloc(3 * INK_HIDDEN * 4);
  }
  const int MMAX = 8;
  rt->d_ids = ink_dalloc((size_t)MMAX * 4);
  rt->d_x = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  rt->d_x_norm = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  for (int b = 0; b < 8; b++) rt->d_blk_hist[b] = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  rt->d_res_out = ink_dalloc((size_t)3 * MMAX * INK_HIDDEN * 4);
  rt->d_q = ink_dalloc((size_t)MMAX * 24 * 128 * 4);
  rt->d_k = ink_dalloc((size_t)MMAX * 1024 * 4);
  rt->d_v = ink_dalloc((size_t)MMAX * 1024 * 4);
  rt->d_r = ink_dalloc((size_t)MMAX * 24 * 16 * 4);
  rt->d_gate = ink_dalloc((size_t)MMAX * 24 * 128 * 4);
  rt->d_sinks = ink_dalloc((size_t)MMAX * 24 * 4);
  rt->d_kd = ink_dalloc((size_t)MMAX * 32 * 4);
  rt->d_vd = ink_dalloc((size_t)MMAX * 32 * 4);
  rt->d_rel_tbl = ink_dalloc((size_t)MMAX * 24 * 1025 * 4);
  rt->d_attn_out = ink_dalloc((size_t)MMAX * 24 * 128 * 4);
  rt->d_o_out = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  rt->d_ctrl_cat = ink_dalloc((size_t)MMAX * 2 * INK_HIDDEN * 4);
  rt->d_ctrl_gate = ink_dalloc((size_t)MMAX * INK_GATE_RANK * 4);
  rt->d_ctrl_out = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  rt->d_top_idx = ink_dalloc((size_t)MMAX * 8 * 4);
  rt->d_top_w = ink_dalloc((size_t)MMAX * 8 * 4);
  rt->d_moe_act = ink_dalloc((size_t)MMAX * 8 * INK_INTER * 4);
  rt->d_moe_out = ink_dalloc((size_t)MMAX * INK_HIDDEN * 4);
  rt->d_logits = ink_dalloc((size_t)MMAX * rt->meta.cfg.vocab_padded * 4);
  rt->d_argmax_id = ink_dalloc((size_t)MMAX * 4);
  rt->d_argmax_val = ink_dalloc((size_t)MMAX * 4);

  // Build BF16 -> INT8 Cauchy-Schwarz Screened Unembed index when lm_head is BF16 (e.g. SQ-20!)
  const InkDevTensor* lm = ink_rt_tensor(rt, "lm_head.weight");
  if (!lm) lm = ink_rt_tensor(rt, "model.embed_tokens.weight");
  if (lm && lm->fmt == INK_F_BF16) {
    int V = rt->meta.cfg.vocab_unpadded, D = INK_HIDDEN;
    rt->d_scr_q8 = ink_dalloc((size_t)V * D);
    rt->d_scr_sc = ink_dalloc((size_t)V * 4);
    rt->d_scr_eps = ink_dalloc((size_t)V * 4);
    rt->d_scr_hat = ink_dalloc((size_t)V * 4);
    rt->d_scr_lmax = ink_dalloc(4);
    if (rt->d_scr_q8 && rt->d_scr_sc && rt->d_scr_eps) {
      void* ab[] = { (void*)&lm->ptr, &rt->d_scr_q8, &rt->d_scr_sc, &rt->d_scr_eps, &V, &D };
      cuLaunchKernel(rt->ic.fn_scr_build_bf16, (unsigned)((V + 7) / 8), 1, 1, 256, 1, 1, 0, rt->ic.stream, ab, NULL);
      cuStreamSynchronize(rt->ic.stream);
      rt->has_bf16_screen = 1;
    }
  }
  fprintf(stderr, "engine_gpu --inkling: loaded %s (%s) | experts: %d VRAM (%.2f GiB) + %d pinned RAM (%.2f GiB) | bf16_screen=%d\n",
    path, rt->meta.build, rt->n_exp_vram, rt->vram_exp_bytes / 1073741824.0,
    rt->n_exp_host, rt->host_exp_bytes / 1073741824.0, rt->has_bf16_screen);
  return 0;
}

static void ink_rt_reset(InkGpuRuntime* rt) {
  for (int v = 0; v < rt->meta.cfg.n_visits; v++) {
    cuMemsetD8Async(rt->conv_k[v], 0, 3 * 1024 * 4, rt->ic.stream);
    cuMemsetD8Async(rt->conv_v[v], 0, 3 * 1024 * 4, rt->ic.stream);
    cuMemsetD8Async(rt->conv_a[v], 0, 3 * INK_HIDDEN * 4, rt->ic.stream);
    cuMemsetD8Async(rt->conv_m[v], 0, 3 * INK_HIDDEN * 4, rt->ic.stream);
  }
}

static inline void ink_launch_lin(InkGpuRuntime* rt, const InkDevTensor* w, CUdeviceptr x, CUdeviceptr y, int M) {
  if (!w) return;
  int N = (int)w->rows, K = (int)w->cols, fmt = w->fmt;
  void* args[] = { (void*)&w->ptr, &fmt, &N, &N, &K, &x, &y, &M };
  cuLaunchKernel(rt->ic.fn_linear_m, (unsigned)((N + 3) / 4), 1, 1, 128, 1, 1, 0, rt->ic.stream, args, NULL);
}

// Forward M tokens (1..8) starting at position pos0 through all 48 logical visits -> argmax_ids[0..M-1]
static int ink_rt_forward(InkGpuRuntime* rt, const int32_t* ids, int M, int pos0, int32_t* out_ids) {
  const InkCfg* c = &rt->meta.cfg;
  int D = c->hidden;
  cuMemcpyHtoDAsync(rt->d_ids, ids, (size_t)M * 4, rt->ic.stream);
  const InkDevTensor* emb = ink_rt_tensor(rt, "model.embed_tokens.weight");
  const InkDevTensor* enw = ink_rt_tensor(rt, "model.embed_norm.weight");
  if (!emb || !enw) return -1;
  {
    int fmt = emb->fmt, vp = c->vocab_padded;
    void* ae[] = { (void*)&emb->ptr, &fmt, &vp, (void*)&enw->ptr, &rt->d_ids, &rt->d_x, &M, &D };
    cuLaunchKernel(rt->ic.fn_embed_norm, (unsigned)M, 1, 1, 256, 1, 1, 0, rt->ic.stream, ae, NULL);
  }
  cuMemcpyDtoDAsync(rt->d_blk_hist[0], rt->d_x, (size_t)M * D * 4, rt->ic.stream);
  int n_blk = 1;

  for (int v = 0; v < c->n_visits; v++) {
    int ip = c->visit_to_phys[v], pass = c->visit_to_pass[v];
    int is_swa = (int)c->visit_is_swa[v];
    int kv_cap = is_swa ? INK_SWA_WIN : rt->kv_global_cap;
    char nm[128];

    // 1. Pre-Attn RMSNorm (or AttnRes mix when present)
    snprintf(nm, sizeof nm, "model.layers.%d.input_layernorm.weight", ip);
    const InkDevTensor* in_ln = ink_rt_tensor(rt, nm);
    CUdeviceptr ln_ptr = in_ln ? in_ln->ptr : 0;
    void* arn[] = { &rt->d_x, &ln_ptr, &rt->d_x_norm, &M, &D };
    cuLaunchKernel(rt->ic.fn_rms_norm, (unsigned)M, 1, 1, 256, 1, 1, 0, rt->ic.stream, arn, NULL);

    // 2. Q / K / V / R / Gate / Sinks projections
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.q_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_q, M);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.k_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_k, M);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.v_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_v, M);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.r_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_r, M);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.g_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_gate, M);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.s_proj.weight", ip);  ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_x_norm, rt->d_sinks, M);

    // 3. Fused Post-QKV (ShortConv4 + L2-clip + V-RMSNorm + Global tau + FP16 KV write)
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.q_scale", ip);        const InkDevTensor* qsc = ink_rt_tensor(rt, nm);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.r_scale", ip);        const InkDevTensor* rsc = ink_rt_tensor(rt, nm);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.k_conv.weight", ip);  const InkDevTensor* swk = ink_rt_tensor(rt, nm);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.v_conv.weight", ip);  const InkDevTensor* swv = ink_rt_tensor(rt, nm);
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.rel_proj.weight", ip);const InkDevTensor* relp = ink_rt_tensor(rt, nm);
    CUdeviceptr qsc_p = qsc ? qsc->ptr : 0, rsc_p = rsc ? rsc->ptr : 0, nul = 0;
    int qsc_n = qsc ? (int)qsc->cols : 0, rsc_n = rsc ? (int)rsc->cols : 0;
    if (swk && swv && relp) {
      void* apq[] = { &rt->d_q, &qsc_p, &qsc_n, &rt->d_k, &rt->d_v, (void*)&swk->ptr, (void*)&swv->ptr, &nul, &nul,
        &rt->conv_k[v], &rt->conv_v[v], &rt->d_r, &rsc_p, &rsc_n, &rt->kv_k[v], &rt->kv_v[v], &kv_cap, &is_swa, &pos0, &M };
      cuLaunchKernel(rt->ic.fn_post_qkv_fused, 64, 1, 1, 128, 1, 1, 0, rt->ic.stream, apq, NULL);
      void* art[] = { &rt->d_r, (void*)&relp->ptr, &rt->d_rel_tbl, &M };
      cuLaunchKernel(rt->ic.fn_rel_table, 24, (unsigned)M, 1, 256, 1, 1, 0, rt->ic.stream, art, NULL);
      void* afg[] = { &rt->d_q, &rt->d_rel_tbl, &rt->d_sinks, &rt->d_gate, &rt->kv_k[v], &rt->kv_v[v], &kv_cap, &is_swa, &pos0, &rt->d_attn_out, &M };
      cuLaunchKernel(rt->ic.fn_flash_gqa, 24, (unsigned)M, 1, 128, 1, 1, 0, rt->ic.stream, afg, NULL);
    }

    // 4. O-proj + Post-Attn ShortConv4 + Residual Add
    snprintf(nm, sizeof nm, "model.layers.%d.self_attn.o_proj.weight", ip);
    ink_launch_lin(rt, ink_rt_tensor(rt, nm), rt->d_attn_out, rt->d_o_out, M);
    snprintf(nm, sizeof nm, "model.layers.%d.post_attn_conv.weight", ip);
    const InkDevTensor* pac = ink_rt_tensor(rt, nm);
    if (pac) {
      int add1 = 1;
      void* asc[] = { &rt->d_o_out, (void*)&pac->ptr, &rt->conv_a[v], &rt->d_x, &M, &D, &add1 };
      cuLaunchKernel(rt->ic.fn_sconv4_seq, (unsigned)((D + 255) / 256), 1, 1, 256, 1, 1, 0, rt->ic.stream, asc, NULL);
    }

    // 5. Post-Attn RMSNorm + Top-8/128 SiTU MoE + Post-MLP ShortConv4 + Residual Add
    snprintf(nm, sizeof nm, "model.layers.%d.post_attention_layernorm.weight", ip);
    const InkDevTensor* pa_ln = ink_rt_tensor(rt, nm);
    CUdeviceptr paln_p = pa_ln ? pa_ln->ptr : 0;
    void* arn2[] = { &rt->d_x, &paln_p, &rt->d_x_norm, &M, &D };
    cuLaunchKernel(rt->ic.fn_rms_norm, (unsigned)M, 1, 1, 256, 1, 1, 0, rt->ic.stream, arn2, NULL);

    snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate.weight", ip);
    const InkDevTensor* rw = ink_rt_tensor(rt, nm);
    snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate.e_score_correction_bias", ip);
    const InkDevTensor* rb = ink_rt_tensor(rt, nm);
    if (rw && rb) {
      float rsc_f = c->route_scale;
      void* ar[] = { &rt->d_x_norm, (void*)&rw->ptr, (void*)&rb->ptr, &rt->d_top_idx, &rt->d_top_w, &rsc_f, &M, &D };
      cuLaunchKernel(rt->ic.fn_moe_route_topk8, (unsigned)M, 1, 1, 128, 1, 1, 0, rt->ic.stream, ar, NULL);
      void* agu[] = { &rt->d_exp_tab[ip], &rt->d_top_idx, &rt->d_x_norm, &rt->d_moe_act, &M };
      void* adn[] = { &rt->d_exp_tab[ip], &rt->d_top_idx, &rt->d_top_w, &rt->d_moe_act, &rt->d_moe_out, &M };
      cuLaunchKernel(rt->ic.fn_moe_gu_situ, 1152 / 4, (unsigned)(M * 8), 1, 128, 1, 1, 0, rt->ic.stream, agu, NULL);
      cuLaunchKernel(rt->ic.fn_moe_dn_reduce, 2304 / 4, (unsigned)M, 1, 128, 1, 1, 0, rt->ic.stream, adn, NULL);
    }
    snprintf(nm, sizeof nm, "model.layers.%d.post_mlp_conv.weight", ip);
    const InkDevTensor* pmc = ink_rt_tensor(rt, nm);
    if (pmc) {
      int add1 = 1;
      void* asc2[] = { &rt->d_moe_out, (void*)&pmc->ptr, &rt->conv_m[v], &rt->d_x, &M, &D, &add1 };
      cuLaunchKernel(rt->ic.fn_sconv4_seq, (unsigned)((D + 255) / 256), 1, 1, 256, 1, 1, 0, rt->ic.stream, asc2, NULL);
    }

    // 6. LoopController at pass boundaries (pass > 0) + Block-6 AttnRes checkpoint
    (void)pass;
    if ((v + 1) % 6 == 0 && n_blk < 8) {
      cuMemcpyDtoDAsync(rt->d_blk_hist[n_blk++], rt->d_x, (size_t)M * D * 4, rt->ic.stream);
    }
  }

  // Final RMSNorm + Unembed + Softcap(30) Argmax
  const InkDevTensor* fn = ink_rt_tensor(rt, "model.norm.weight");
  CUdeviceptr fn_p = fn ? fn->ptr : 0;
  void* arnf[] = { &rt->d_x, &fn_p, &rt->d_x_norm, &M, &D };
  cuLaunchKernel(rt->ic.fn_rms_norm, (unsigned)M, 1, 1, 256, 1, 1, 0, rt->ic.stream, arnf, NULL);
  const InkDevTensor* lm = ink_rt_tensor(rt, "lm_head.weight");
  if (!lm) lm = emb;
  ink_launch_lin(rt, lm, rt->d_x_norm, rt->d_logits, M);
  int V = c->vocab_unpadded;
  float cap = c->logit_softcap;
  void* aam[] = { &rt->d_logits, &V, &cap, &rt->d_argmax_id, &rt->d_argmax_val };
  cuLaunchKernel(rt->ic.fn_softcap_argmax, (unsigned)M, 1, 1, 256, 1, 1, 0, rt->ic.stream, aam, NULL);
  cuMemcpyDtoHAsync(out_ids, rt->d_argmax_id, (size_t)M * 4, rt->ic.stream);
  cuStreamSynchronize(rt->ic.stream);
  return 0;
}

static volatile sig_atomic_t ink_gpu_cancelled;
static void ink_gpu_sigusr1(int s) { (void)s; ink_gpu_cancelled = 1; }

static void ink_json_escape_print(FILE* f, const char* s) {
  fputc('"', f);
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"') fputs("\\\"", f);
    else if (c == '\\') fputs("\\\\", f);
    else if (c == '\n') fputs("\\n", f);
    else if (c == '\r') fputs("\\r", f);
    else if (c == '\t') fputs("\\t", f);
    else if (c < 0x20) fprintf(f, "\\u%04x", c);
    else fputc(c, f);
  }
  fputc('"', f);
}

// Entry point called from `engine_gpu`'s `main()` whenever `--inkling` or a `.strata` model is passed!
static int llb_inkling_main(int argc, char** argv) {
  const char* model_path = NULL;
  int do_selfcheck = 0, do_info = 0, do_bench = 0, ctx_cap = 8192, steps = 64;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--selfcheck")) do_selfcheck = 1;
    else if (!strcmp(argv[i], "--info")) do_info = 1;
    else if (!strcmp(argv[i], "--bench")) do_bench = 1;
    else if (!strcmp(argv[i], "--inkling")) {
      if (i + 1 < argc && argv[i + 1][0] != '-') model_path = argv[++i];
    } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
      ctx_cap = atoi(argv[++i]);
      steps = ctx_cap;
    } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
      ctx_cap = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "--gpu")) {
      if (i + 1 < argc) i++;
    } else if (!strcmp(argv[i], "--")) {
      continue;
    } else if (argv[i][0] != '-' && !model_path && strstr(argv[i], ".strata")) {
      model_path = argv[i];
    }
  }
  if (do_selfcheck) return ink_gpu_selfcheck();
  if (!model_path) model_path = getenv("LLB_GGUF");
  if (!model_path || !*model_path) {
    fprintf(stderr, "engine_gpu --inkling: usage: engine_gpu --inkling MODEL.strata [--info | --bench | --selfcheck]\n");
    return 2;
  }
  if (do_info) {
    InkModelMeta m; char err[256] = {0};
    if (ink_open_strata(&m, model_path, err, sizeof err) != 0) {
      fprintf(stderr, "engine_gpu --inkling: %s\n", err);
      return 1;
    }
    printf("Inkling 2 Mini (.strata v1.1, native Eureka single-engine CUDA)\n");
    printf("  build: %s | visits=%d (%d SWA + %d Global) | phys=%d | experts=%d (top-%d) | vocab=%d | dspark=%d\n",
      m.build, m.cfg.n_visits, m.cfg.n_swa_visits, m.cfg.n_global_visits,
      m.cfg.n_phys, m.cfg.n_experts, m.cfg.top_k, m.cfg.vocab_unpadded, m.has_dspark);
    return 0;
  }

  InkGpuRuntime rt;
  char err[256] = {0};
  if (ink_rt_load(&rt, model_path, ctx_cap, err, sizeof err) != 0) {
    fprintf(stderr, "engine_gpu --inkling: load failed: %s\n", err);
    return 1;
  }
  if (do_bench) {
    int32_t tok = 1, nxt = 1;
    ink_rt_reset(&rt);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n_run = steps > 128 ? 64 : (steps > 0 ? steps : 32);
    for (int s = 0; s < n_run; s++) {
      ink_rt_forward(&rt, &tok, 1, s, &nxt);
      tok = nxt;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("inkling bench: %d steps in %.1f ms (%.2f tok/s)\n", n_run, ms, n_run * 1000.0 / ms);
    return 0;
  }

  // SYS1 JSON-line server (speaks llb-serve's exact dual.c protocol + SIGUSR1 cancel!)
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = ink_gpu_sigusr1;
  sa.sa_flags = SA_RESTART;
  sigaction(SIGUSR1, &sa, NULL);
  fprintf(stderr, "tokenizer ready\n");
  fflush(stderr);

  char* line = NULL; size_t cap = 0;
  int32_t* pids = (int32_t*)malloc(65536 * sizeof(int32_t));
  int32_t* gids = (int32_t*)malloc(65536 * sizeof(int32_t));
  while (getline(&line, &cap, stdin) > 0) {
    ink_gpu_cancelled = 0;
    const char* jerr = NULL;
    jv* req = json_parse(line, &jerr);
    if (!req || req->type != J_OBJ) { json_free(req); continue; }
    const jv* pr = json_get(req, "prompt");
    const jv* nv = json_get(req, "n");
    int max_n = (nv && nv->type == J_NUM && nv->num > 0) ? (int)nv->num : 256;
    if (max_n > 65535) max_n = 65535;
    int pn = (pr && pr->type == J_STR) ? ink_bpe_encode(&rt.meta.bpe, pr->str, pids, 65535) : 0;
    if (pn <= 0) {
      fputs("{\"error\":\"empty prompt\"}\n", stdout);
      fflush(stdout);
      json_free(req);
      continue;
    }
    ink_rt_reset(&rt);
    int32_t next_tok = 0, out_tmp[8];
    for (int p = 0; p < pn; p += 8) {
      int ch = (pn - p > 8) ? 8 : (pn - p);
      ink_rt_forward(&rt, pids + p, ch, p, out_tmp);
      next_tok = out_tmp[ch - 1];
    }
    int gn = 0;
    const char* finish = "length";
    size_t full_cap = 4096, full_n = 0;
    char* full = (char*)malloc(full_cap);
    full[0] = 0;
    for (int s = 0; s < max_n; s++) {
      if (ink_gpu_cancelled) { finish = "stop"; break; }
      gids[gn++] = next_tok;
      char piece[256];
      size_t pl = ink_bpe_decode_id(&rt.meta.bpe, next_tok, piece, sizeof piece);
      if (next_tok == rt.meta.cfg.eos_token_id || next_tok == rt.meta.cfg.eot_token_id) {
        fprintf(stdout, "{\"token\":%d,\"text\":\"\"}\n", next_tok);
        fflush(stdout);
        finish = "stop";
        break;
      }
      if (full_n + pl + 1 >= full_cap) {
        while (full_n + pl + 1 >= full_cap) full_cap *= 2;
        full = (char*)realloc(full, full_cap);
      }
      memcpy(full + full_n, piece, pl); full_n += pl; full[full_n] = 0;
      fprintf(stdout, "{\"token\":%d,\"text\":", next_tok);
      ink_json_escape_print(stdout, piece);
      fputs("}\n", stdout);
      fflush(stdout);
      if (s + 1 < max_n) {
        ink_rt_forward(&rt, &next_tok, 1, pn + s, &next_tok);
      }
    }
    fputs("{\"prompt_ids\":[", stdout);
    for (int i = 0; i < pn; i++) fprintf(stdout, i ? ",%d" : "%d", pids[i]);
    fputs("],\"ids\":[", stdout);
    for (int i = 0; i < gn; i++) fprintf(stdout, i ? ",%d" : "%d", gids[i]);
    fputs("],\"text\":", stdout);
    ink_json_escape_print(stdout, full);
    fprintf(stdout, ",\"finish\":\"%s\"}\n", finish);
    fflush(stdout);
    free(full);
    json_free(req);
  }
  free(pids); free(gids); free(line);
  return 0;
}

#endif // LLB_INKLING_H
