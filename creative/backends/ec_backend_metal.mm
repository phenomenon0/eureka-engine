// ec_backend_metal.mm — Universal Apple Silicon (M1-M5) + Intel/AMD Mac Metal Compute Backend
// Features:
//   - Runtime 3-tier GPU-family pipeline selection:
//       Tier 1 (Apple9+ M3/M4/M5): Native hardware `simdgroup_matrix<bfloat, 8, 8>` + 32-wide simdgroup FA2
//       Tier 2 (Apple7/8 M1/M2):   Native hardware `simdgroup_matrix<half, 8, 8>` + FP32 accum + 32-wide FA2
//       Tier 3 (Intel / AMD Mac):  Portable 16x16 threadgroup-tiled FP32 + width-agnostic 64-thread FA2
//   - Deferred async MTLCommandBuffer batching (flushes automatically on sync / memcpy / free / 64 ops)
//   - Disjoint zero-copy page-aligned `newBufferWithBytesNoCopy` Unified Memory safetensors mmap binding
//   - Native Grand Central Dispatch (`dispatch_apply_f`) multi-core CPU glue (zero OpenMP dependency!)

#ifndef EC_BACKEND_METAL
  #define EC_BACKEND_METAL 1
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/vm_page_size.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

#include "../ec_hal.h"
#include "ec_ops_common.h"

typedef struct {
  uint32_t M, N, K;
  uint32_t epilogue;
  uint32_t has_bias, has_res, write_f32, write_16;
} EcMetalGemmArgs;

typedef struct {
  uint32_t Lq, Lkv, Lq_pad, Lkv_pad, Hq, Hkv, D;
  int32_t  window;
  uint32_t block_causal, write_f32, write_16;
  float    scale;
} EcMetalFa2Args;

typedef struct {
  uint32_t H, W, Cin, Cout, has_bias, has_res, write_f32, write_bf16;
} EcMetalConv3x3Args;

typedef struct {
  uintptr_t      user_base; // Disjoint user-visible range [user_base, user_base + user_size)
  size_t         user_size;
  NSUInteger     page_off;  // Offset inside page-aligned MTLBuffer (0 for cudaMalloc, host_ptr - p0 for no_copy)
  id<MTLBuffer>  buf;
  int            no_copy;
} EcMetalAlloc;

#define EC_MTL_MAX_ALLOCS 8192
#define EC_MTL_MAX_BATCH  64

static struct {
  pthread_mutex_t             mu;
  int                         init_done;
  int                         init_ok;
  int                         gpu_tier; // 2=Apple9 bfloat, 1=Apple7 half simdgroup_matrix, 0=Intel/AMD tiled
  id<MTLDevice>               dev;
  id<MTLCommandQueue>         queue;
  id<MTLLibrary>              lib;
  id<MTLComputePipelineState> pso_gemm_bf16;
  id<MTLComputePipelineState> pso_gemm_f16;
  id<MTLComputePipelineState> pso_gemm_q8g64;
  id<MTLComputePipelineState> pso_gemm_i8;
  id<MTLComputePipelineState> pso_quant_q8g64;
  id<MTLComputePipelineState> pso_fa2_bf16;
  id<MTLComputePipelineState> pso_fa2_f16;
  id<MTLComputePipelineState> pso_conv3x3;
  id<MTLBuffer>               dummy_buf;

  // Deferred async command buffer batching (locked by mu)
  id<MTLCommandBuffer>         active_cb;
  id<MTLComputeCommandEncoder> active_enc;
  int                          pending_ops;

  size_t                      total_mem;
  size_t                      used_mem;
  EcMetalAlloc                allocs[EC_MTL_MAX_ALLOCS];
  int                         n_allocs;
} g_mtl = { .mu = PTHREAD_MUTEX_INITIALIZER };

extern "C" const char *ec_hal_backend_name(void) { return "metal"; }

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) return nil;
  NSError *err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso && err) {
    fprintf(stderr, "[ec_metal] pipeline '%s' failed: %s\n", [name UTF8String], [[err localizedDescription] UTF8String]);
  }
  return pso;
}

static void flush_batch_locked(bool wait_done) {
  if (!g_mtl.active_cb) return;
  @autoreleasepool {
    if (g_mtl.active_enc) {
      [g_mtl.active_enc endEncoding];
      g_mtl.active_enc = nil;
    }
    id<MTLCommandBuffer> cb = g_mtl.active_cb;
    g_mtl.active_cb = nil;
    g_mtl.pending_ops = 0;
    [cb commit];
    if (wait_done) [cb waitUntilCompleted];
  }
}

static id<MTLComputeCommandEncoder> get_encoder_locked(void) {
  if (!g_mtl.active_cb) {
    g_mtl.active_cb = [g_mtl.queue commandBuffer];
    g_mtl.active_enc = [g_mtl.active_cb computeCommandEncoder];
    g_mtl.pending_ops = 0;
  }
  return g_mtl.active_enc;
}

static void finish_op_locked(void) {
  g_mtl.pending_ops++;
  if (g_mtl.pending_ops >= EC_MTL_MAX_BATCH) {
    // Commit asynchronously without stalling CPU; next sync waits on latest cb!
    // Keeping wait_done=true at 64-op fence prevents unbounded in-flight command buffers.
    flush_batch_locked(true);
  }
}

extern "C" int ec_hal_init(void) {
  pthread_mutex_lock(&g_mtl.mu);
  if (g_mtl.init_done) {
    int ok = g_mtl.init_ok;
    pthread_mutex_unlock(&g_mtl.mu);
    return ok ? 0 : -1;
  }
  g_mtl.init_done = 1;

  @autoreleasepool {
    g_mtl.dev = MTLCreateSystemDefaultDevice();
    if (!g_mtl.dev) { pthread_mutex_unlock(&g_mtl.mu); return -1; }
    g_mtl.queue = [g_mtl.dev newCommandQueue];
    g_mtl.dummy_buf = [g_mtl.dev newBufferWithLength:256 options:MTLResourceStorageModeShared];

    uint64_t hw_mem = 0;
    size_t sz = sizeof(hw_mem);
    if (sysctlbyname("hw.memsize", &hw_mem, &sz, NULL, 0) == 0 && hw_mem > 0) {
      g_mtl.total_mem = (size_t)hw_mem;
    } else {
      g_mtl.total_mem = (size_t)[g_mtl.dev recommendedMaxWorkingSetSize];
    }

    bool is_apple9 = false;
    bool is_apple7 = false;
    if (@available(macOS 14.0, iOS 17.0, *)) {
      is_apple9 = [g_mtl.dev supportsFamily:MTLGPUFamilyApple9];
    }
    if (@available(macOS 11.0, iOS 14.0, *)) {
      is_apple7 = [g_mtl.dev supportsFamily:MTLGPUFamilyApple7];
    }
    const char *force_tier = getenv("EC_METAL_TIER");
    if (force_tier && !strcmp(force_tier, "tiled")) { is_apple9 = false; is_apple7 = false; }
    else if (force_tier && !strcmp(force_tier, "half")) { is_apple9 = false; }

    NSError *err = nil;
    const char *env_lib = getenv("EC_METALLIB_PATH");
    if (env_lib && env_lib[0]) {
      NSURL *u = [NSURL fileURLWithPath:[NSString stringWithUTF8String:env_lib]];
      g_mtl.lib = [g_mtl.dev newLibraryWithURL:u error:&err];
    }
    if (!g_mtl.lib) {
      NSString *exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
      NSString *cand = [exeDir stringByAppendingPathComponent:@"ec_creative.metallib"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:cand]) {
        g_mtl.lib = [g_mtl.dev newLibraryWithURL:[NSURL fileURLWithPath:cand] error:&err];
      }
    }
    if (!g_mtl.lib) {
      NSString *srcPath = @EC_METAL_SHADER_SRC_PATH;
      NSString *src = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&err];
      if (!src) {
        NSString *exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
        src = [NSString stringWithContentsOfFile:[exeDir stringByAppendingPathComponent:@"ec_creative.metal"]
                                        encoding:NSUTF8StringEncoding error:&err];
      }
      if (src) {
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (is_apple9) {
          if (@available(macOS 14.0, iOS 17.0, *)) opts.languageVersion = MTLLanguageVersion3_1;
        } else if (is_apple7) {
          if (@available(macOS 13.0, iOS 16.0, *)) opts.languageVersion = MTLLanguageVersion3_0;
        }
        g_mtl.lib = [g_mtl.dev newLibraryWithSource:src options:opts error:&err];
      }
    }
    if (!g_mtl.lib) {
      fprintf(stderr, "[ec_metal] failed to load/compile Metal library: %s\n",
              err ? [[err localizedDescription] UTF8String] : "unknown");
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }

    // Select runtime GPU tier cleanly: Apple9 bfloat -> Apple7 half -> Intel/AMD portable tiled
    g_mtl.gpu_tier = 0;
    if (is_apple9) {
      g_mtl.pso_gemm_bf16 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_bf16_bf");
      g_mtl.pso_gemm_q8g64 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_q8g64_bf");
      g_mtl.pso_conv3x3 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_conv2d_3x3_nhwc_bf16_bf");
      if (g_mtl.pso_gemm_bf16 && g_mtl.pso_gemm_q8g64 && g_mtl.pso_conv3x3) {
        g_mtl.gpu_tier = 2;
      }
    }
    if (g_mtl.gpu_tier == 0 && is_apple7) {
      g_mtl.pso_gemm_bf16 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_bf16_f16");
      g_mtl.pso_gemm_q8g64 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_q8g64_f16");
      g_mtl.pso_conv3x3 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_conv2d_3x3_nhwc_bf16_f16");
      if (g_mtl.pso_gemm_bf16 && g_mtl.pso_gemm_q8g64 && g_mtl.pso_conv3x3) {
        g_mtl.gpu_tier = 1;
      }
    }
    if (g_mtl.gpu_tier >= 1) {
      g_mtl.pso_gemm_f16    = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_f16");
      g_mtl.pso_gemm_i8     = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_i8_f16");
      g_mtl.pso_quant_q8g64 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_quantize_bf16_q8g64");
      g_mtl.pso_fa2_bf16    = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_fa2_bf16");
      g_mtl.pso_fa2_f16     = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_fa2_f16");
    } else {
      // Tier 3: Intel UHD/Iris & AMD Radeon Mac GPUs (width-agnostic, zero simdgroup_matrix)
      g_mtl.pso_gemm_bf16   = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_bf16_tiled");
      g_mtl.pso_gemm_f16    = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_f16_tiled");
      g_mtl.pso_gemm_q8g64  = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_q8g64_tiled");
      g_mtl.pso_gemm_i8     = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_gemm_i8_tiled");
      g_mtl.pso_quant_q8g64 = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_quantize_bf16_q8g64_tiled");
      g_mtl.pso_fa2_bf16    = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_fa2_bf16_tiled");
      g_mtl.pso_fa2_f16     = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_fa2_f16_tiled");
      g_mtl.pso_conv3x3     = make_pso(g_mtl.dev, g_mtl.lib, @"k_metal_conv2d_3x3_nhwc_bf16_tiled");
    }

    if (!g_mtl.pso_gemm_bf16 || !g_mtl.pso_gemm_f16 || !g_mtl.pso_gemm_q8g64 || !g_mtl.pso_gemm_i8 ||
        !g_mtl.pso_quant_q8g64 || !g_mtl.pso_fa2_bf16 || !g_mtl.pso_fa2_f16 || !g_mtl.pso_conv3x3) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
  }
  g_mtl.init_ok = 1;
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

static bool lookup_mtl_buf(const void *ptr, id<MTLBuffer> __strong *out_buf, NSUInteger *out_off) {
  if (!ptr) {
    *out_buf = g_mtl.dummy_buf;
    *out_off = 0;
    return true;
  }
  uintptr_t addr = (uintptr_t)ptr;
  int lo = 0, hi = g_mtl.n_allocs - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    const EcMetalAlloc &a = g_mtl.allocs[mid];
    if (addr < a.user_base) hi = mid - 1;
    else if (addr >= a.user_base + a.user_size) lo = mid + 1;
    else {
      *out_buf = a.buf;
      *out_off = a.page_off + (NSUInteger)(addr - a.user_base);
      return true;
    }
  }
  return false;
}

extern "C" cudaError_t ec_hal_get_device_props(cudaDeviceProp *prop, int device) {
  (void)device;
  if (ec_hal_init() != 0 || !prop) return cudaErrorNotSupported;
  memset(prop, 0, sizeof *prop);
  const char *tier_s = (g_mtl.gpu_tier == 2) ? "Metal Apple9 bfloat"
                     : (g_mtl.gpu_tier == 1) ? "Metal Apple7 half"
                                             : "Metal Tiled";
  snprintf(prop->name, sizeof prop->name, "%s (%s)", [[g_mtl.dev name] UTF8String], tier_s);
  prop->totalGlobalMem = g_mtl.total_mem;
  prop->major = (g_mtl.gpu_tier == 2) ? 9 : (g_mtl.gpu_tier == 1 ? 7 : 2);
  prop->minor = 0;
  prop->multiProcessorCount = 32;
  prop->unifiedAddressing = [g_mtl.dev hasUnifiedMemory] ? 1 : 0;
  return cudaSuccess;
}

extern "C" cudaError_t ec_hal_mem_get_info(size_t *free_b, size_t *total_b) {
  if (ec_hal_init() != 0) return cudaErrorNotSupported;
  size_t used = (size_t)[g_mtl.dev currentAllocatedSize];
  size_t tot  = (size_t)[g_mtl.dev recommendedMaxWorkingSetSize];
  if (tot == 0) tot = g_mtl.total_mem;
  if (total_b) *total_b = tot;
  if (free_b)  *free_b  = (tot > used) ? (tot - used) : (1024ULL << 20);
  return cudaSuccess;
}

extern "C" cudaError_t ec_hal_malloc_raw(void **ptr, size_t nbytes) {
  if (!ptr) return cudaErrorMemoryAllocation;
  *ptr = NULL;
  if (nbytes == 0) nbytes = 256;
  if (ec_hal_init() != 0) return cudaErrorMemoryAllocation;
  size_t aligned = (nbytes + 255ULL) & ~255ULL;
  pthread_mutex_lock(&g_mtl.mu);
  if (g_mtl.n_allocs >= EC_MTL_MAX_ALLOCS) { pthread_mutex_unlock(&g_mtl.mu); return cudaErrorMemoryAllocation; }
  id<MTLBuffer> buf = [g_mtl.dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
  if (!buf) { pthread_mutex_unlock(&g_mtl.mu); return cudaErrorMemoryAllocation; }
  void *raw = [buf contents];
  uintptr_t base = (uintptr_t)raw;
  int pos = 0;
  while (pos < g_mtl.n_allocs && g_mtl.allocs[pos].user_base < base) pos++;
  for (int i = g_mtl.n_allocs; i > pos; i--) g_mtl.allocs[i] = g_mtl.allocs[i - 1];
  g_mtl.allocs[pos] = { base, aligned, 0u, buf, 0 };
  g_mtl.n_allocs++;
  g_mtl.used_mem += aligned;
  *ptr = raw;
  pthread_mutex_unlock(&g_mtl.mu);
  return cudaSuccess;
}

// Disjoint zero-copy page-aligned safetensors mmap -> MTLBuffer on Apple Silicon Unified Memory!
extern "C" void *ec_hal_mmap_bind(const void *host_ptr, size_t nbytes) {
  if (!host_ptr || nbytes == 0 || (((uintptr_t)host_ptr) & 3u) != 0 || ec_hal_init() != 0) return NULL;
  if (![g_mtl.dev hasUnifiedMemory]) return NULL;
  uintptr_t p0 = (uintptr_t)host_ptr & ~((uintptr_t)vm_page_size - 1ULL);
  uintptr_t p1 = ((uintptr_t)host_ptr + nbytes + vm_page_size - 1ULL) & ~((uintptr_t)vm_page_size - 1ULL);
  size_t len = (size_t)(p1 - p0);
  if (len > (size_t)[g_mtl.dev maxBufferLength]) return NULL;

  pthread_mutex_lock(&g_mtl.mu);
  if (g_mtl.n_allocs >= EC_MTL_MAX_ALLOCS) { pthread_mutex_unlock(&g_mtl.mu); return NULL; }
  uintptr_t ubase = (uintptr_t)host_ptr;
  int pos = 0;
  while (pos < g_mtl.n_allocs && g_mtl.allocs[pos].user_base < ubase) pos++;
  // Guarantee strict non-overlap in binary-search table
  if (pos > 0 && g_mtl.allocs[pos - 1].user_base + g_mtl.allocs[pos - 1].user_size > ubase) {
    pthread_mutex_unlock(&g_mtl.mu);
    return NULL;
  }
  if (pos < g_mtl.n_allocs && ubase + nbytes > g_mtl.allocs[pos].user_base) {
    pthread_mutex_unlock(&g_mtl.mu);
    return NULL;
  }

  id<MTLBuffer> buf = [g_mtl.dev newBufferWithBytesNoCopy:(void *)p0
                                                   length:len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
  if (!buf) { pthread_mutex_unlock(&g_mtl.mu); return NULL; }
  for (int i = g_mtl.n_allocs; i > pos; i--) g_mtl.allocs[i] = g_mtl.allocs[i - 1];
  g_mtl.allocs[pos] = { ubase, nbytes, (NSUInteger)(ubase - p0), buf, 1 };
  g_mtl.n_allocs++;
  pthread_mutex_unlock(&g_mtl.mu);
  return (void *)host_ptr;
}

extern "C" cudaError_t ec_hal_free_raw(void *ptr) {
  if (!ptr) return cudaSuccess;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true); // Never free a buffer referenced by an in-flight command buffer
  uintptr_t addr = (uintptr_t)ptr;
  for (int i = 0; i < g_mtl.n_allocs; i++) {
    if (g_mtl.allocs[i].user_base == addr) {
      if (!g_mtl.allocs[i].no_copy && g_mtl.used_mem >= g_mtl.allocs[i].user_size)
        g_mtl.used_mem -= g_mtl.allocs[i].user_size;
      g_mtl.allocs[i].buf = nil;
      for (int j = i; j + 1 < g_mtl.n_allocs; j++) g_mtl.allocs[j] = g_mtl.allocs[j + 1];
      g_mtl.n_allocs--;
      break;
    }
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return cudaSuccess;
}

extern "C" cudaError_t ec_hal_host_alloc_raw(void **ptr, size_t nbytes, unsigned int flags) {
  (void)flags;
  return ec_hal_malloc_raw(ptr, nbytes);
}
extern "C" cudaError_t ec_hal_host_free_raw(void *ptr) { return ec_hal_free_raw(ptr); }

extern "C" cudaError_t ec_hal_memcpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  (void)kind; (void)stream;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  if (count > 0 && dst && src && dst != src) memcpy(dst, src, count);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_memset_async(void *dst, int value, size_t count, cudaStream_t stream) {
  (void)stream;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  if (count > 0 && dst) memset(dst, value, count);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_stream_create(cudaStream_t *pStream, unsigned int flags) {
  (void)flags;
  if (ec_hal_init() != 0) return cudaErrorNotSupported;
  if (pStream) *pStream = (cudaStream_t)(uintptr_t)1;
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_stream_destroy(cudaStream_t s) {
  (void)s;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_stream_sync(cudaStream_t s) {
  (void)s;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_device_sync(void) {
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  return cudaSuccess;
}

typedef struct { struct timespec ts; } EcMetalEv;
extern "C" cudaError_t ec_hal_event_create(cudaEvent_t *e, unsigned int f) {
  (void)f;
  EcMetalEv *ev = (EcMetalEv *)calloc(1, sizeof *ev);
  if (!ev) return cudaErrorMemoryAllocation;
  clock_gettime(CLOCK_MONOTONIC, &ev->ts);
  *e = (cudaEvent_t)ev;
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_event_destroy(cudaEvent_t e) { free(e); return cudaSuccess; }
extern "C" cudaError_t ec_hal_event_record(cudaEvent_t e, cudaStream_t s) {
  (void)s;
  pthread_mutex_lock(&g_mtl.mu);
  flush_batch_locked(true);
  pthread_mutex_unlock(&g_mtl.mu);
  if (e) clock_gettime(CLOCK_MONOTONIC, &((EcMetalEv *)e)->ts);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_event_sync(cudaEvent_t e) {
  (void)e;
  return ec_hal_device_sync();
}
extern "C" cudaError_t ec_hal_event_elapsed(float *ms, cudaEvent_t a, cudaEvent_t b) {
  if (!ms || !a || !b) return cudaErrorNotSupported;
  EcMetalEv *ea = (EcMetalEv *)a, *eb = (EcMetalEv *)b;
  *ms = (float)((double)(eb->ts.tv_sec - ea->ts.tv_sec) * 1e3 + (double)(eb->ts.tv_nsec - ea->ts.tv_nsec) * 1e-6);
  return cudaSuccess;
}
extern "C" cudaError_t ec_hal_stream_wait_event(cudaStream_t s, cudaEvent_t e, unsigned int f) {
  (void)s; (void)e; (void)f; return cudaSuccess;
}

// ===========================================================================
// Universal Batched Metal GPU Dispatches (M1-M5 simdgroup_matrix + Intel/AMD Tiled)
// ===========================================================================
static void dispatch_gemm_2d(id<MTLComputeCommandEncoder> enc, uint32_t M, uint32_t N) {
  if (g_mtl.gpu_tier >= 1) {
    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  } else {
    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  }
}

extern "C" int ec_ops_gemm_bf16(
    const uint16_t *X, const uint16_t *W, const float *bias,
    float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X || !W || M <= 0 || N <= 0 || K <= 0 || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bX, bW, bB, bYf, bYb;
    NSUInteger oX, oW, oB, oYf, oYb;
    if (!lookup_mtl_buf(X, &bX, &oX) || !lookup_mtl_buf(W, &bW, &oW) ||
        !lookup_mtl_buf(bias, &bB, &oB) || !lookup_mtl_buf(Y_f32, &bYf, &oYf) ||
        !lookup_mtl_buf(Y_bf16, &bYb, &oYb)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalGemmArgs args = {
      (uint32_t)M, (uint32_t)N, (uint32_t)K, 0u,
      bias ? 1u : 0u, 0u, Y_f32 ? 1u : 0u, Y_bf16 ? 1u : 0u
    };
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_gemm_bf16];
    [enc setBuffer:bX offset:oX atIndex:0];
    [enc setBuffer:bW offset:oW atIndex:1];
    [enc setBuffer:bB offset:oB atIndex:2];
    [enc setBuffer:bYf offset:oYf atIndex:3];
    [enc setBuffer:bYb offset:oYb atIndex:4];
    [enc setBytes:&args length:sizeof(args) atIndex:5];
    dispatch_gemm_2d(enc, (uint32_t)M, (uint32_t)N);
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_launch_gemm_f16(
    const __half *X, const __half *W, const float *bias, const float *res,
    float *Y, __half *Y_f16, int M, int N, int K, int epilogue, cudaStream_t stream) {
  (void)stream;
  if (!X || !W || M <= 0 || N <= 0 || K <= 0 || (!Y && !Y_f16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bX, bW, bB, bR, bYf, bYh;
    NSUInteger oX, oW, oB, oR, oYf, oYh;
    if (!lookup_mtl_buf(X, &bX, &oX) || !lookup_mtl_buf(W, &bW, &oW) ||
        !lookup_mtl_buf(bias, &bB, &oB) || !lookup_mtl_buf(res, &bR, &oR) ||
        !lookup_mtl_buf(Y, &bYf, &oYf) || !lookup_mtl_buf(Y_f16, &bYh, &oYh)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalGemmArgs args = {
      (uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)epilogue,
      bias ? 1u : 0u, res ? 1u : 0u, Y ? 1u : 0u, Y_f16 ? 1u : 0u
    };
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_gemm_f16];
    [enc setBuffer:bX offset:oX atIndex:0];
    [enc setBuffer:bW offset:oW atIndex:1];
    [enc setBuffer:bB offset:oB atIndex:2];
    [enc setBuffer:bR offset:oR atIndex:3];
    [enc setBuffer:bYf offset:oYf atIndex:4];
    [enc setBuffer:bYh offset:oYh atIndex:5];
    [enc setBytes:&args length:sizeof(args) atIndex:6];
    dispatch_gemm_2d(enc, (uint32_t)M, (uint32_t)N);
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_ops_quantize_bf16_q8g64(
    const uint16_t *src_bf16, int8_t *dst_q8, float *dst_sc, int rows, int cols, cudaStream_t stream) {
  (void)stream;
  if (!src_bf16 || !dst_q8 || !dst_sc || rows <= 0 || (cols & 63)) return -1;
  if (ec_hal_init() != 0) return -1;
  uint32_t ng = (uint32_t)rows * (uint32_t)(cols >> 6);
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bS, bQ, bSc;
    NSUInteger oS, oQ, oSc;
    if (!lookup_mtl_buf(src_bf16, &bS, &oS) || !lookup_mtl_buf(dst_q8, &bQ, &oQ) || !lookup_mtl_buf(dst_sc, &bSc, &oSc)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_quant_q8g64];
    [enc setBuffer:bS offset:oS atIndex:0];
    [enc setBuffer:bQ offset:oQ atIndex:1];
    [enc setBuffer:bSc offset:oSc atIndex:2];
    [enc setBytes:&ng length:sizeof(ng) atIndex:3];
    if (g_mtl.gpu_tier >= 1) {
      [enc dispatchThreadgroups:MTLSizeMake((ng + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake((ng + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_ops_gemm_q8g64(
    const int8_t *X_q8, const float *x_sc, const int8_t *W_q8, const float *w_sc,
    const float *bias, float *Y_f32, uint16_t *Y_bf16, int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X_q8 || !x_sc || !W_q8 || !w_sc || M <= 0 || N <= 0 || (K & 63) || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bX, bXs, bW, bWs, bB, bYf, bYb;
    NSUInteger oX, oXs, oW, oWs, oB, oYf, oYb;
    if (!lookup_mtl_buf(X_q8, &bX, &oX) || !lookup_mtl_buf(x_sc, &bXs, &oXs) ||
        !lookup_mtl_buf(W_q8, &bW, &oW) || !lookup_mtl_buf(w_sc, &bWs, &oWs) ||
        !lookup_mtl_buf(bias, &bB, &oB) || !lookup_mtl_buf(Y_f32, &bYf, &oYf) ||
        !lookup_mtl_buf(Y_bf16, &bYb, &oYb)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalGemmArgs args = {
      (uint32_t)M, (uint32_t)N, (uint32_t)K, 0u,
      bias ? 1u : 0u, 0u, Y_f32 ? 1u : 0u, Y_bf16 ? 1u : 0u
    };
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_gemm_q8g64];
    [enc setBuffer:bX offset:oX atIndex:0];
    [enc setBuffer:bXs offset:oXs atIndex:1];
    [enc setBuffer:bW offset:oW atIndex:2];
    [enc setBuffer:bWs offset:oWs atIndex:3];
    [enc setBuffer:bB offset:oB atIndex:4];
    [enc setBuffer:bYf offset:oYf atIndex:5];
    [enc setBuffer:bYb offset:oYb atIndex:6];
    [enc setBytes:&args length:sizeof(args) atIndex:7];
    dispatch_gemm_2d(enc, (uint32_t)M, (uint32_t)N);
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_launch_gemm_i8(
    const int8_t *X_i8, const float *x_sc, const int8_t *W_i8, const float *w_sc,
    const float *bias, float *Y, int M, int N, int K, cudaStream_t stream) {
  (void)stream;
  if (!X_i8 || !x_sc || !W_i8 || !w_sc || !Y || M <= 0 || N <= 0 || K <= 0) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bX, bXs, bW, bWs, bB, bY;
    NSUInteger oX, oXs, oW, oWs, oB, oY;
    if (!lookup_mtl_buf(X_i8, &bX, &oX) || !lookup_mtl_buf(x_sc, &bXs, &oXs) ||
        !lookup_mtl_buf(W_i8, &bW, &oW) || !lookup_mtl_buf(w_sc, &bWs, &oWs) ||
        !lookup_mtl_buf(bias, &bB, &oB) || !lookup_mtl_buf(Y, &bY, &oY)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalGemmArgs args = {
      (uint32_t)M, (uint32_t)N, (uint32_t)K, 0u,
      bias ? 1u : 0u, 0u, 1u, 0u
    };
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_gemm_i8];
    [enc setBuffer:bX offset:oX atIndex:0];
    [enc setBuffer:bXs offset:oXs atIndex:1];
    [enc setBuffer:bW offset:oW atIndex:2];
    [enc setBuffer:bWs offset:oWs atIndex:3];
    [enc setBuffer:bB offset:oB atIndex:4];
    [enc setBuffer:bY offset:oY atIndex:5];
    [enc setBytes:&args length:sizeof(args) atIndex:6];
    dispatch_gemm_2d(enc, (uint32_t)M, (uint32_t)N);
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_ops_fa2_bf16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, float *O_f32, uint16_t *O_bf16,
    int Lq, int Lkv, int Lq_pad, int Lkv_pad, int Hq, int Hkv, int D, float scale, int window, cudaStream_t stream) {
  (void)stream;
  if (!Q || !K || !V || Lq <= 0 || Lkv <= 0 || Hq <= 0 || Hkv <= 0 || (Hq % Hkv) || (D != 64 && D != 128)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bQ, bK, bV, bOf, bOb;
    NSUInteger oQ, oK, oV, oOf, oOb;
    if (!lookup_mtl_buf(Q, &bQ, &oQ) || !lookup_mtl_buf(K, &bK, &oK) || !lookup_mtl_buf(V, &bV, &oV) ||
        !lookup_mtl_buf(O_f32, &bOf, &oOf) || !lookup_mtl_buf(O_bf16, &bOb, &oOb)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalFa2Args args = {
      (uint32_t)Lq, (uint32_t)Lkv, (uint32_t)Lq_pad, (uint32_t)Lkv_pad,
      (uint32_t)Hq, (uint32_t)Hkv, (uint32_t)D, window, 0u,
      O_f32 ? 1u : 0u, O_bf16 ? 1u : 0u, scale
    };
    uint32_t pairs = (uint32_t)Lq * (uint32_t)Hq;
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_fa2_bf16];
    [enc setBuffer:bQ offset:oQ atIndex:0];
    [enc setBuffer:bK offset:oK atIndex:1];
    [enc setBuffer:bV offset:oV atIndex:2];
    [enc setBuffer:bOf offset:oOf atIndex:3];
    [enc setBuffer:bOb offset:oOb atIndex:4];
    [enc setBytes:&args length:sizeof(args) atIndex:5];
    if (g_mtl.gpu_tier >= 1) {
      [enc dispatchThreadgroups:MTLSizeMake((pairs + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake(pairs, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_launch_fa2_f16(
    const __half *Q, const __half *K, const __half *V, float *O, __half *O_f16,
    int n_q, int n_kv, int n_q_pad, int n_kv_pad, int n_heads, int head_dim, float scale, int block_causal_size, cudaStream_t stream) {
  (void)stream;
  if (!Q || !K || !V || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || (head_dim != 64 && head_dim != 128)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bQ, bK, bV, bOf, bOh;
    NSUInteger oQ, oK, oV, oOf, oOh;
    if (!lookup_mtl_buf(Q, &bQ, &oQ) || !lookup_mtl_buf(K, &bK, &oK) || !lookup_mtl_buf(V, &bV, &oV) ||
        !lookup_mtl_buf(O, &bOf, &oOf) || !lookup_mtl_buf(O_f16, &bOh, &oOh)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalFa2Args args = {
      (uint32_t)n_q, (uint32_t)n_kv, (uint32_t)n_q_pad, (uint32_t)n_kv_pad,
      (uint32_t)n_heads, (uint32_t)n_heads, (uint32_t)head_dim, 0,
      (uint32_t)(block_causal_size > 0 ? block_causal_size : 0),
      O ? 1u : 0u, O_f16 ? 1u : 0u, scale
    };
    uint32_t pairs = (uint32_t)n_q * (uint32_t)n_heads;
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_fa2_f16];
    [enc setBuffer:bQ offset:oQ atIndex:0];
    [enc setBuffer:bK offset:oK atIndex:1];
    [enc setBuffer:bV offset:oV atIndex:2];
    [enc setBuffer:bOf offset:oOf atIndex:3];
    [enc setBuffer:bOh offset:oOh atIndex:4];
    [enc setBytes:&args length:sizeof(args) atIndex:5];
    if (g_mtl.gpu_tier >= 1) {
      [enc dispatchThreadgroups:MTLSizeMake((pairs + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      [enc dispatchThreadgroups:MTLSizeMake(pairs, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}

extern "C" int ec_ops_conv2d_3x3_nhwc_bf16(
    const uint16_t *X_nhwc, const uint16_t *W_ohwi, const float *bias, const float *res_f32,
    float *Y_f32, uint16_t *Y_bf16, int H, int W, int Cin, int Cout, cudaStream_t stream) {
  (void)stream;
  if (!X_nhwc || !W_ohwi || H <= 0 || W <= 0 || (Cin & 31) != 0 || (Cout & 63) != 0 || (!Y_f32 && !Y_bf16)) return -1;
  if (ec_hal_init() != 0) return -1;
  pthread_mutex_lock(&g_mtl.mu);
  @autoreleasepool {
    id<MTLBuffer> bX, bW, bB, bR, bYf, bYb;
    NSUInteger oX, oW, oB, oR, oYf, oYb;
    if (!lookup_mtl_buf(X_nhwc, &bX, &oX) || !lookup_mtl_buf(W_ohwi, &bW, &oW) ||
        !lookup_mtl_buf(bias, &bB, &oB) || !lookup_mtl_buf(res_f32, &bR, &oR) ||
        !lookup_mtl_buf(Y_f32, &bYf, &oYf) || !lookup_mtl_buf(Y_bf16, &bYb, &oYb)) {
      pthread_mutex_unlock(&g_mtl.mu);
      return -1;
    }
    EcMetalConv3x3Args args = {
      (uint32_t)H, (uint32_t)W, (uint32_t)Cin, (uint32_t)Cout,
      bias ? 1u : 0u, res_f32 ? 1u : 0u, Y_f32 ? 1u : 0u, Y_bf16 ? 1u : 0u
    };
    uint32_t M = (uint32_t)H * (uint32_t)W;
    id<MTLComputeCommandEncoder> enc = get_encoder_locked();
    [enc setComputePipelineState:g_mtl.pso_conv3x3];
    [enc setBuffer:bX offset:oX atIndex:0];
    [enc setBuffer:bW offset:oW atIndex:1];
    [enc setBuffer:bB offset:oB atIndex:2];
    [enc setBuffer:bR offset:oR atIndex:3];
    [enc setBuffer:bYf offset:oYf atIndex:4];
    [enc setBuffer:bYb offset:oYb atIndex:5];
    [enc setBytes:&args length:sizeof(args) atIndex:6];
    dispatch_gemm_2d(enc, M, (uint32_t)Cout);
    finish_op_locked();
  }
  pthread_mutex_unlock(&g_mtl.mu);
  return 0;
}
