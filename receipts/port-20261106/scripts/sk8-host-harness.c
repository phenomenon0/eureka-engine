// SK8 host dispatch, isolated: blk.c is the `#if defined(BEND_CUDA) && defined(LLB_LAUNCH)` block at the top of
// host/llb_host.c (llb_sk8_enabled / llb_sk8_fits / llb_sk8_launch), with stub CUDA types and a counting LLB_LAUNCH.
//   sed -n "1,/^#endif/p" host/llb_host.c > blk.c && clang -std=c11 -Wall -Wextra -Werror sk8-host-harness.c -o t && ./t  # exit 0
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned u32; typedef unsigned long long CUdeviceptr; typedef int CUresult; typedef void* CUfunction;
#define CUDA_SUCCESS 0
#define BEND_CUDA 1
static CUresult cuMemAlloc(CUdeviceptr* p, size_t n) { (void)n; *p = 1; return 0; }
static CUfunction gpu_sk8, gpu_sk8_combine;
static int launches;
static bool launch_stub(int n, ...) { launches++; return n > 0; }
#define LLB_LAUNCH(k, gx, gy, bx, ...) ((void)(k), launch_stub((int)((gx) * (gy) * (bx)), __VA_ARGS__))
#include "blk.c"
int main(void) {
  setenv("LLB_ATTN_SK8", "1", 1);
  int ok = llb_sk8_enabled() && llb_sk8_fits(17u | (5u << 5), 16) && !llb_sk8_fits(18u, 16) && !llb_sk8_fits(17u, 129)
    && llb_sk8_launch(1, 2, 3, 4, 17u | (5u << 5), 4, 16) && launches == 2;
  setenv("LLB_ATTN_SK8", "0", 1); ok = ok && !llb_sk8_enabled();
  return ok ? 0 : 1;
}
