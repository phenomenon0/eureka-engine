// ec_hal.h — Portable Hardware Abstraction Layer for Eureka Creative Engine
// Supports 3 native GPU backends behind one C11/C++17 codebase:
//   1. EC_BACKEND_CUDA   — NVIDIA PTX TensorCore (Linux & Windows)
//   2. EC_BACKEND_METAL  — Apple Silicon Metal 3 simdgroup_matrix + Unified Memory (macOS & iOS)
//   3. EC_BACKEND_VULKAN — Cross-Vendor Vulkan 1.2+ SPIR-V Compute (Linux & Windows AMD/Intel/NVIDIA)

#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__) || defined(EC_BACKEND_CUDA)
  #ifndef EC_BACKEND_CUDA
    #define EC_BACKEND_CUDA 1
  #endif
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
  #ifdef __CUDACC__
    #include <cuda_bf16.h>
  #endif

  #ifdef __cplusplus
  extern "C" {
  #endif
  static inline const char *ec_hal_backend_name(void) { return "cuda"; }
  static inline void *ec_hal_mmap_bind(const void *host_ptr, size_t nbytes) {
    (void)host_ptr; (void)nbytes;
    return NULL; // Discrete PCIe VRAM uses explicit staging/upload
  }
  #ifdef __cplusplus
  }
  #endif

  #ifdef __CUDACC__
    #define EC_LAUNCH_1D(kern, grid_x, block_x, stream, ...) \
      kern<<<(grid_x), (block_x), 0, (stream)>>>(__VA_ARGS__)
  #endif

#elif defined(EC_BACKEND_METAL) || defined(EC_BACKEND_VULKAN)

  #ifdef __cplusplus
  extern "C" {
  #endif

  typedef int   cudaError_t;
  typedef void *cudaStream_t;
  typedef void *cudaEvent_t;
  typedef void *cudaGraph_t;
  typedef void *cudaGraphExec_t;

  #define cudaSuccess                   0
  #define cudaErrorMemoryAllocation     2
  #define cudaErrorNotSupported         801
  #define cudaStreamNonBlocking         0x01
  #define cudaStreamCaptureModeGlobal   0x01
  #define cudaEventDisableTiming        0x02
  #define cudaHostAllocDefault          0x00
  #define cudaFuncAttributePreferredSharedMemoryCarveout 9
  #define cudaSharedmemCarveoutMaxShared 100

  typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
  } cudaMemcpyKind;

  typedef struct {
    char   name[256];
    size_t totalGlobalMem;
    int    major;
    int    minor;
    int    multiProcessorCount;
    int    unifiedAddressing;
  } cudaDeviceProp;

  typedef struct {
    uint16_t __x;
  } __half;

  typedef struct {
    uint32_t __x;
  } __half2;

  static inline uint32_t __float_as_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
  }
  static inline float __uint_as_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
  }
  static inline float ec_hal_rsqrtf(float x) {
    return 1.0f / sqrtf(x);
  }
  #undef rsqrtf
  #define rsqrtf(x) ec_hal_rsqrtf(x)

  // Exact IEEE-754 float32 <-> binary16 conversions (round-to-nearest-even, subnormals/Inf/NaN preserved)
  static inline __half __float2half(float f) {
    uint32_t x = __float_as_uint(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    __half h;
    if (exp <= 0) {
      if (exp < -10) {
        h.__x = (uint16_t)sign;
        return h;
      }
      mant = (mant | 0x800000u) >> (1 - exp);
      if ((mant & 0x1000u) && ((mant & 0x2fffu) != 0x1000u)) mant += 0x1000u;
      h.__x = (uint16_t)(sign | (mant >> 13));
      return h;
    } else if (exp == 0xff - 127 + 15) {
      h.__x = (uint16_t)(sign | 0x7c00u | (mant ? ((mant >> 13) | 1u) : 0u));
      return h;
    } else if (exp > 30) {
      h.__x = (uint16_t)(sign | 0x7c00u);
      return h;
    }
    if ((mant & 0x1000u) && ((mant & 0x2fffu) != 0x1000u)) {
      mant += 0x1000u;
      if (mant & 0x800000u) { mant = 0; exp++; if (exp > 30) { h.__x = (uint16_t)(sign | 0x7c00u); return h; } }
    }
    h.__x = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    return h;
  }
  static inline __half __float2half_rn(float f) { return __float2half(f); }

  static inline float __half2float(__half h) {
    uint32_t sign = ((uint32_t)(h.__x & 0x8000u)) << 16;
    uint32_t exp  = (h.__x >> 10) & 0x1fu;
    uint32_t mant = h.__x & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
      if (mant == 0) {
        out = sign;
      } else {
        exp = 1;
        while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
        mant &= 0x3ffu;
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
      }
    } else if (exp == 31) {
      out = sign | 0x7f800000u | (mant << 13);
    } else {
      out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return __uint_as_float(out);
  }

  // Backend-implemented HAL entrypoints (creative/backends/ec_backend_{metal.mm,vulkan.c})
  const char *ec_hal_backend_name(void);
  int         ec_hal_init(void);
  cudaError_t ec_hal_get_device_props(cudaDeviceProp *prop, int device);
  cudaError_t ec_hal_mem_get_info(size_t *free_b, size_t *total_b);
  cudaError_t ec_hal_malloc_raw(void **ptr, size_t nbytes);
  cudaError_t ec_hal_free_raw(void *ptr);
  void       *ec_hal_mmap_bind(const void *host_ptr, size_t nbytes);
  cudaError_t ec_hal_host_alloc_raw(void **ptr, size_t nbytes, unsigned int flags);
  cudaError_t ec_hal_host_free_raw(void *ptr);
  cudaError_t ec_hal_memcpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t ec_hal_memset_async(void *dst, int value, size_t count, cudaStream_t stream);
  cudaError_t ec_hal_stream_create(cudaStream_t *pStream, unsigned int flags);
  cudaError_t ec_hal_stream_destroy(cudaStream_t stream);
  cudaError_t ec_hal_stream_sync(cudaStream_t stream);
  cudaError_t ec_hal_device_sync(void);
  cudaError_t ec_hal_event_create(cudaEvent_t *event, unsigned int flags);
  cudaError_t ec_hal_event_destroy(cudaEvent_t event);
  cudaError_t ec_hal_event_record(cudaEvent_t event, cudaStream_t stream);
  cudaError_t ec_hal_event_sync(cudaEvent_t event);
  cudaError_t ec_hal_event_elapsed(float *ms, cudaEvent_t start, cudaEvent_t end);
  cudaError_t ec_hal_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

  #ifdef __cplusplus
  } // extern "C"

  template <typename T>
  static inline cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return ec_hal_malloc_raw((void **)devPtr, size);
  }
  template <typename T>
  static inline cudaError_t cudaHostAlloc(T **pHost, size_t size, unsigned int flags) {
    return ec_hal_host_alloc_raw((void **)pHost, size, flags);
  }
  #else
  #define cudaMalloc(p, s)          ec_hal_malloc_raw((void **)(p), (s))
  #define cudaHostAlloc(p, s, f)    ec_hal_host_alloc_raw((void **)(p), (s), (f))
  #endif

  static inline cudaError_t cudaFree(void *p) { return ec_hal_free_raw(p); }
  static inline cudaError_t cudaFreeHost(void *p) { return ec_hal_host_free_raw(p); }
  static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind k) {
    cudaError_t rc = ec_hal_memcpy_async(dst, src, n, k, NULL);
    if (rc == cudaSuccess) rc = ec_hal_device_sync();
    return rc;
  }
  static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind k, cudaStream_t s) {
    return ec_hal_memcpy_async(dst, src, n, k, s);
  }
  static inline cudaError_t cudaMemset(void *dst, int val, size_t n) {
    cudaError_t rc = ec_hal_memset_async(dst, val, n, NULL);
    if (rc == cudaSuccess) rc = ec_hal_device_sync();
    return rc;
  }
  static inline cudaError_t cudaMemsetAsync(void *dst, int val, size_t n, cudaStream_t s) {
    return ec_hal_memset_async(dst, val, n, s);
  }
  static inline cudaError_t cudaStreamCreate(cudaStream_t *s) { return ec_hal_stream_create(s, 0); }
  static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int f) { return ec_hal_stream_create(s, f); }
  static inline cudaError_t cudaStreamDestroy(cudaStream_t s) { return ec_hal_stream_destroy(s); }
  static inline cudaError_t cudaStreamSynchronize(cudaStream_t s) { return ec_hal_stream_sync(s); }
  static inline cudaError_t cudaDeviceSynchronize(void) { return ec_hal_device_sync(); }
  static inline cudaError_t cudaEventCreate(cudaEvent_t *e) { return ec_hal_event_create(e, 0); }
  static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int f) { return ec_hal_event_create(e, f); }
  static inline cudaError_t cudaEventDestroy(cudaEvent_t e) { return ec_hal_event_destroy(e); }
  #ifdef __cplusplus
  static inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = NULL) { return ec_hal_event_record(e, s); }
  #else
  static inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) { return ec_hal_event_record(e, s); }
  #endif
  static inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { return ec_hal_event_sync(e); }
  static inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { return ec_hal_event_elapsed(ms, a, b); }
  static inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int f) { return ec_hal_stream_wait_event(s, e, f); }
  static inline cudaError_t cudaMemGetInfo(size_t *f, size_t *t) { return ec_hal_mem_get_info(f, t); }
  static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int d) { return ec_hal_get_device_props(p, d); }
  static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }
  static inline cudaError_t cudaFuncSetAttribute(const void *fn, int attr, int val) {
    (void)fn; (void)attr; (void)val;
    return cudaSuccess;
  }
  static inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, int m) { (void)s; (void)m; return cudaErrorNotSupported; }
  static inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g) { (void)s; (void)g; return cudaErrorNotSupported; }
  static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *ge, cudaGraph_t g, void *a, char *b, size_t c) {
    (void)ge; (void)g; (void)a; (void)b; (void)c; return cudaErrorNotSupported;
  }
  static inline cudaError_t cudaGraphLaunch(cudaGraphExec_t ge, cudaStream_t s) { (void)ge; (void)s; return cudaErrorNotSupported; }
  static inline cudaError_t cudaGraphDestroy(cudaGraph_t g) { (void)g; return cudaSuccess; }
  static inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t ge) { (void)ge; return cudaSuccess; }

  // Portable 1D embarrassingly-parallel grid executor for non-barrier glue kernels on Unified/Coherent memory.
  // On macOS/iOS uses native Grand Central Dispatch (`dispatch_apply_f`) so Apple Clang runs across all P+E cores
  // with zero external OpenMP dependency; on Linux/Windows uses OpenMP.
  // NOTE: Intentionally does NOT define __syncthreads() or __shfl_xor_sync() so any barrier/warp-shuffle kernel
  // MUST route to a GPU shader (ec_ops_*) or explicit row/head implementation — enforced at compile time!
  #ifdef __cplusplus
  #ifdef __APPLE__
  #include <dispatch/dispatch.h>
  #endif
  struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  };
  inline thread_local dim3 ec_tls_blockIdx;
  inline thread_local dim3 ec_tls_blockDim;
  inline thread_local dim3 ec_tls_threadIdx;
  #define blockIdx  ec_tls_blockIdx
  #define blockDim  ec_tls_blockDim
  #define threadIdx ec_tls_threadIdx
  #define __global__
  #define __device__
  #define __forceinline__ inline
  #define __launch_bounds__(...)

  template <typename Fn>
  static inline void ec_hal_par_for(long n, Fn &&fn) {
    if (n <= 0) return;
  #if defined(__APPLE__)
    if (n >= 16) {
      struct _EcGcdCtx { Fn *f; };
      _EcGcdCtx ctx{ &fn };
      dispatch_apply_f((size_t)n, DISPATCH_APPLY_AUTO, &ctx, [](void *p, size_t i) {
        (*static_cast<_EcGcdCtx *>(p)->f)((long)i);
      });
      return;
    }
    for (long i = 0; i < n; i++) fn(i);
  #else
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) fn(i);
  #endif
  }

  #if (defined(__APPLE__) || defined(EC_TEST_PAR_LAMBDA)) && defined(__cplusplus)
    #define EC_PAR_FOR_BEGIN(var, count) ec_hal_par_for((long)(count), [&](long var) {
    #define EC_PAR_FOR_END               });
  #endif

  #define EC_LAUNCH_1D(kern, grid_val, block_val, stream, ...) do { \
    ec_hal_stream_sync(stream); \
    const dim3 _g(grid_val); \
    const dim3 _b(block_val); \
    const long _total_blocks = (long)_g.x * (long)_g.y * (long)_g.z; \
    ec_hal_par_for(_total_blocks, [&](long _bi) { \
      const unsigned int _bx = (unsigned int)(_bi % _g.x); \
      const unsigned int _by = (unsigned int)((_bi / _g.x) % _g.y); \
      const unsigned int _bz = (unsigned int)(_bi / ((long)_g.x * _g.y)); \
      ec_tls_blockDim = _b; \
      ec_tls_blockIdx = dim3(_bx, _by, _bz); \
      for (unsigned int _tz = 0; _tz < _b.z; _tz++) { \
        for (unsigned int _ty = 0; _ty < _b.y; _ty++) { \
          for (unsigned int _tx = 0; _tx < _b.x; _tx++) { \
            ec_tls_threadIdx = dim3(_tx, _ty, _tz); \
            kern(__VA_ARGS__); \
          } \
        } \
      } \
    }); \
  } while (0)
  #endif

  #ifndef EC_PAR_FOR_BEGIN
    #define EC_PAR_FOR_BEGIN(var, count) _Pragma("omp parallel for schedule(static)") for (long var = 0; var < (long)(count); var++) {
    #define EC_PAR_FOR_END               }
  #endif

#else
  #error "Define one of EC_BACKEND_CUDA, EC_BACKEND_METAL, or EC_BACKEND_VULKAN"
#endif
