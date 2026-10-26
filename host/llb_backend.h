// llb_backend.h -- what the host chains (llb_host.c) need from a GPU backend.
// Neutral: no CUDA or Metal type appears here or in llb_host.c. A backend
// header (llb_backend_cuda.h; later an Objective-C llb_backend_metal.h) comes
// first and defines, macro- or static-inline-thin (the pass bang queues ~392
// launches a step, so a wrapper must cost nothing):
//
//   llb_dptr                  a device address, an unsigned 64-bit integer:
//                             byte offsets add to it (gpu_blk + 4096 * 4)
//   LLB_LAUNCH(k, gx, gy, bx, &a0, &a1, ...)  -> true if queued
//                             gx * gy groups of bx threads, no dynamic shared
//                             memory. Each argument is the address of a host
//                             variable holding the value (a u32, a float, an
//                             llb_dptr or a by-value struct such as Q8Cat);
//                             the value is copied at the call. Launches run in
//                             call order and each sees every earlier one's
//                             writes (CUDA: the null stream; Metal: one
//                             serial encoder with a buffer barrier after each).
//                             The backend owns encoder lifetime: a launch may
//                             come after a sync, so it opens a new one lazily.
//                             "Queued" may not mean "valid": Metal reports a
//                             failure only on the command buffer, so the
//                             runtime's sync must check it
//   LLB_LAUNCH_SM(k, gx, gy, bx, sm, &a0, ...)  -> LLB_LAUNCH with sm bytes of
//                             dynamic shared memory (the fast prefill tier)
//   llb_alloc_zero(bytes)     -> llb_dptr of zeroed device memory, 0 on failure
//                             (bytes a multiple of 4)
//   llb_zero(p, bytes)        -> true if queued: p's bytes set to 0, in launch order
//   llb_copy(d, s, bytes)    -> true if queued: s's bytes into d, in launch order
//   llb_write(d, h, bytes)   -> true if host h's bytes were copied into d
//   llb_read(h, p, bytes)     -> true once p's bytes are in host h (after every
//                             earlier launch; diagnostics only: it syncs)
//   llb_dev(p)                -> llb_dptr of a corpus host pointer (CUDA UVM:
//                             the same bits; Metal: the buffer's gpuAddress
//                             plus the offset)
//
// The runtime side (comp.ts gpu_load / gpu_pass / seam_pf) still calls the
// driver directly; its contract, for a backend that also owns it:
//   kernel-by-name (cuModuleGetFunction), a host-mapped allocation with its
//   device address (cuMemHostAlloc + cuMemHostGetDevicePointer: the Mail),
//   sync (cuCtxSynchronize: the host reads the Mail only after it), and a
//   prefetch hint (cuMemPrefetchAsync: a no-op on unified memory).
// The chains also use runtime names: Corpus, WarpOp, Mail, LINE, WARP_MAX,
// AMAX_G / AMAX_T, err_fail and the gpu_* kernel and buffer handles; the
// list is tools/step2a_stub.h, which tools/step2a_neutral_check.sh compiles
// this file's user against with no CUDA header.
#ifndef LLB_BACKEND_H
#define LLB_BACKEND_H
#ifndef LLB_LAUNCH
#error "include a backend (llb_backend_cuda.h) before llb_backend.h"
#endif
#endif
