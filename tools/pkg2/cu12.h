/* cu12.h: pre-included (-include) for the CUDA 12 build. The engine's CUDA_VERSION >= 13000 branches
 * (cuMemAdvise with a CUmemLocation, the seam_pf prefetch, io_put's cuMemcpyHtoD for big reads) use
 * calls CUDA 12.2+ already has as cuMemAdvise_v2 / cuMemPrefetchAsync_v2 (CUDA 13 made them the
 * default names). Without this the 12.x headers take the old branches, where big reads memcpy into
 * managed memory: the 27B load then charges ~12+ GB of host RAM (OOM under a 12G cap, pkg2 receipts). */
#include <cuda.h>
#undef CUDA_VERSION
#define CUDA_VERSION 13000
#define cuMemAdvise cuMemAdvise_v2
#define cuMemPrefetchAsync cuMemPrefetchAsync_v2
