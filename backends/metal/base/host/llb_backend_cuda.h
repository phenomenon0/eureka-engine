// llb_backend_cuda.h -- llb_backend.h on the CUDA driver API (the emitted C's
// runtime already holds the context; launches go to the null stream).
#ifndef LLB_BACKEND_CUDA_H
#define LLB_BACKEND_CUDA_H
typedef CUdeviceptr llb_dptr;

#define LLB_LAUNCH(k, gx, gy, bx, ...) \
  (cuLaunchKernel((k), (gx), (gy), 1, (bx), 1, 1, 0, NULL, \
    (void*[]){ __VA_ARGS__ }, NULL) == CUDA_SUCCESS)
// LLB_LAUNCH with sm bytes of dynamic shared memory (the fast prefill tier's wide leaves)
#define LLB_LAUNCH_SM(k, gx, gy, bx, sm, ...) \
  (cuLaunchKernel((k), (gx), (gy), 1, (bx), 1, 1, (sm), NULL, \
    (void*[]){ __VA_ARGS__ }, NULL) == CUDA_SUCCESS)

static inline llb_dptr llb_alloc_zero(size_t bytes) {
  CUdeviceptr p = 0;
  return cuMemAlloc(&p, bytes) == CUDA_SUCCESS
    && cuMemsetD32(p, 0, bytes / 4) == CUDA_SUCCESS ? p : 0;
}

static inline bool llb_zero(llb_dptr p, size_t bytes) {
  return cuMemsetD32(p, 0, bytes / 4) == CUDA_SUCCESS;
}

static inline bool llb_read(void* h, llb_dptr p, size_t bytes) {
  return cuMemcpyDtoH(h, p, bytes) == CUDA_SUCCESS;
}

// LLB_SPEC: a device copy in launch order (the null stream; no host sync)
static inline bool llb_copy(llb_dptr d, llb_dptr s, size_t bytes) {
  return cuMemcpyDtoDAsync(d, s, bytes, NULL) == CUDA_SUCCESS;
}

// LLB_MTP: host bytes onto the device (load time)
static inline bool llb_write(llb_dptr d, const void* h, size_t bytes) {
  return cuMemcpyHtoD(d, h, bytes) == CUDA_SUCCESS;
}

#define llb_dev(p) ((llb_dptr)(uintptr_t)(p))
#endif
