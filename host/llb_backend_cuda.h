// llb_backend_cuda.h -- llb_backend.h on the CUDA driver API (the emitted C's
// runtime already holds the context; launches go to the null stream).
#ifndef LLB_BACKEND_CUDA_H
#define LLB_BACKEND_CUDA_H
typedef CUdeviceptr llb_dptr;

#define LLB_LAUNCH(k, gx, gy, bx, ...) LLB_LAUNCH_SM(k, gx, gy, bx, 0, __VA_ARGS__)
// LLB_LAUNCH with sm bytes of dynamic shared memory (the fast prefill tier's wide leaves)
#define LLB_LAUNCH_SM(k, gx, gy, bx, sm, ...) ({ \
  void* a_[] = { __VA_ARGS__ }; \
  llb_launch((k), (gx), (gy), (bx), (sm), a_, sizeof a_ / sizeof *a_); })

// LLB_GRAPH=1 (FLOORPORT step 3; ported onto the P1+S6-b tree by W1-PREFILL): between llb_graph_begin and llb_graph_end the
// launches are recorded (kernel, dims, arg bytes), not issued, then run as one
// CUDA graph on the null stream: the same kernels, args and order, so the same
// bytes. The graph is kept; a step with the same kernel sequence only updates
// the nodes whose dims or args changed. Any other device call while recording
// (copy/zero/read/write) first issues the recorded launches eagerly and ends the
// recording (the step stays eager: "bail"). Off by default (plain launches).
// Q27MTP: LLB_GSLOT cached graphs, one a kernel sequence (a spec cycle's shape
// varies with its catch-up rows and drafts); a new sequence takes the slots in turn.
#define LLB_GMAX 2048
#define LLB_GBYTES (LLB_GMAX * 128)
#define LLB_GSLOT 16
typedef struct { CUfunction f; u32 gx, gy, bx, sm, na, off, len; } LlbGNode;
static int llb_grec;  // recording
static u32 llb_gn, llb_gb, llb_cn[LLB_GSLOT], llb_gv;  // recorded nodes / bytes; each slot's nodes; next victim
static LlbGNode llb_gq[LLB_GMAX], llb_gc[LLB_GSLOT][LLB_GMAX];
static unsigned char llb_gqb[LLB_GBYTES], llb_gcb[LLB_GSLOT][LLB_GBYTES];
static CUgraphNode llb_gnode[LLB_GSLOT][LLB_GMAX];
static CUgraph llb_gg[LLB_GSLOT];
static CUgraphExec llb_ge[LLB_GSLOT];
static u64 llb_gstat[4];  // graphs built, steps replayed, nodes updated, bails

static void llb_graph_fail(const char* m) {
  fprintf(stderr, "LLB_GRAPH: %s\n", m);
  exit(1);
}

static bool llb_graph_on(void) {
  static int v = -1;
  if (v < 0) {
    const char* s = getenv("LLB_GRAPH");
    v = s != NULL && s[0] == '1';
  }
  return v;
}

// a kernel's param sizes, cached (cuFuncGetParamInfo per arg per launch was ~1 ms a spec cycle);
// returns the param count (at most na)
#define LLB_PSLOT 1024
static struct { CUfunction f; u32 n; unsigned short z[64]; } llb_ps[LLB_PSLOT];
static u32 llb_psz(CUfunction f, u32 na, const unsigned short** z) {
  u32 h = (u32)(((uintptr_t)f >> 4) % LLB_PSLOT);
  while (llb_ps[h].f != NULL && llb_ps[h].f != f) h = (h + 1) % LLB_PSLOT;
  if (llb_ps[h].f == NULL) {
    if (na > 64) llb_graph_fail("too many args");
    u32 i = 0;
    for (size_t o, sz; i < na && cuFuncGetParamInfo(f, i, &o, &sz) == CUDA_SUCCESS; i += 1) llb_ps[h].z[i] = (unsigned short)sz;
    llb_ps[h].f = f; llb_ps[h].n = i;  // past the kernel's params (cuLaunchKernel ignores extra args: gpu_amaxp)
  }
  *z = llb_ps[h].z;
  return llb_ps[h].n < na ? llb_ps[h].n : na;
}

static void llb_gparams(CUDA_KERNEL_NODE_PARAMS* p, LlbGNode* n, unsigned char* b, void** a) {
  const unsigned short* z;
  llb_psz(n->f, n->na, &z);
  for (u32 i = 0, at = n->off; i < n->na; i += 1) {
    a[i] = b + at;
    at += ((u32)z[i] + 7) & ~7u;
  }
  memset(p, 0, sizeof *p);
  p->func = n->f; p->gridDimX = n->gx; p->gridDimY = n->gy; p->gridDimZ = 1;
  p->blockDimX = n->bx; p->blockDimY = 1; p->blockDimZ = 1;
  p->sharedMemBytes = n->sm; p->kernelParams = a;
}

static bool llb_launch(CUfunction k, u32 gx, u32 gy, u32 bx, u32 sm, void** a, u32 na) {
  if (!llb_grec) {
    return cuLaunchKernel(k, gx, gy, 1, bx, 1, 1, sm, NULL, a, NULL) == CUDA_SUCCESS;
  }
  if (llb_gn == LLB_GMAX) llb_graph_fail("too many launches in one step");
  LlbGNode* n = &llb_gq[llb_gn];
  *n = (LlbGNode){ k, gx, gy, bx, sm, na, llb_gb, 0 };
  const unsigned short* z;
  n->na = llb_psz(k, na, &z);
  for (u32 i = 0; i < n->na; i += 1) {
    if (llb_gb + z[i] + 8 > LLB_GBYTES) llb_graph_fail("arg bytes");
    memcpy(llb_gqb + llb_gb, a[i], z[i]);
    llb_gb += ((u32)z[i] + 7) & ~7u;
  }
  n->len = llb_gb - n->off;
  llb_gn += 1;
  return true;
}

// issue what was recorded, eagerly and in order, and stop recording
static void llb_graph_bail(void) {
  if (!llb_grec) return;
  llb_grec = 0;
  llb_gstat[3] += 1;
  for (u32 i = 0; i < llb_gn; i += 1) {
    void* a[64];
    CUDA_KERNEL_NODE_PARAMS p;
    if (llb_gq[i].na > 64) llb_graph_fail("too many args");
    llb_gparams(&p, &llb_gq[i], llb_gqb, a);
    if (cuLaunchKernel(p.func, p.gridDimX, p.gridDimY, 1, p.blockDimX, 1, 1, p.sharedMemBytes, NULL, a, NULL)
      != CUDA_SUCCESS) {
      llb_graph_fail("device launch failed");
    }
  }
  llb_gn = 0;
}

static void llb_graph_stat(void) {
  fprintf(stderr, "LLB_GRAPH: built %llu, steps %llu, node updates %llu, bails %llu\n",
    (unsigned long long)llb_gstat[0], (unsigned long long)llb_gstat[1],
    (unsigned long long)llb_gstat[2], (unsigned long long)llb_gstat[3]);
}

static void llb_graph_begin(void) {
  if (llb_graph_on()) {
    if (llb_gstat[1] + llb_gstat[3] == 0) atexit(llb_graph_stat);
    llb_grec = 1; llb_gn = 0; llb_gb = 0;
  }
}

static void llb_graph_end(void) {
  if (!llb_grec) return;
  llb_grec = 0;
  bool same = false;
  u32 sl = 0;
  for (; !same && sl < LLB_GSLOT; sl += 1) {
    same = llb_ge[sl] != NULL && llb_gn == llb_cn[sl];
    for (u32 i = 0; same && i < llb_gn; i += 1) {
      same = llb_gq[i].f == llb_gc[sl][i].f && llb_gq[i].na == llb_gc[sl][i].na && llb_gq[i].len == llb_gc[sl][i].len;
    }
  }
  sl = same ? sl - 1 : llb_gv;
  llb_gv = same ? llb_gv : (llb_gv + 1) % LLB_GSLOT;
  LlbGNode* gc = llb_gc[sl];
  unsigned char* gcb = llb_gcb[sl];
  void* a[64];
  CUDA_KERNEL_NODE_PARAMS p;
  if (same) {
    for (u32 i = 0; i < llb_gn; i += 1) {
      LlbGNode *q = &llb_gq[i], *c = &gc[i];
      if (q->gx != c->gx || q->gy != c->gy || q->bx != c->bx || q->sm != c->sm
        || memcmp(llb_gqb + q->off, gcb + c->off, q->len) != 0) {
        llb_gparams(&p, q, llb_gqb, a);
        if (cuGraphExecKernelNodeSetParams(llb_ge[sl], llb_gnode[sl][i], &p) != CUDA_SUCCESS) {
          llb_graph_fail("node update failed");
        }
        llb_gstat[2] += 1;
      }
    }
  } else {
    if (llb_ge[sl] != NULL) {
      cuGraphExecDestroy(llb_ge[sl]);
      cuGraphDestroy(llb_gg[sl]);
    }
    if (cuGraphCreate(&llb_gg[sl], 0) != CUDA_SUCCESS) llb_graph_fail("create failed");
    for (u32 i = 0; i < llb_gn; i += 1) {
      if (llb_gq[i].na > 64) llb_graph_fail("too many args");
      llb_gparams(&p, &llb_gq[i], llb_gqb, a);
      CUgraphNode* gn = llb_gnode[sl];
      if (cuGraphAddKernelNode(&gn[i], llb_gg[sl], i ? &gn[i - 1] : NULL, i ? 1 : 0, &p) != CUDA_SUCCESS) {
        llb_graph_fail("add node failed");
      }
    }
    if (cuGraphInstantiate(&llb_ge[sl], llb_gg[sl], 0) != CUDA_SUCCESS) llb_graph_fail("instantiate failed");
    llb_gstat[0] += 1;
  }
  // the recorded step becomes the cached one (its offsets index llb_gcb)
  memcpy(gc, llb_gq, llb_gn * sizeof *llb_gq);
  memcpy(gcb, llb_gqb, llb_gb);
  llb_cn[sl] = llb_gn;
  llb_gstat[1] += 1;
  if (cuGraphLaunch(llb_ge[sl], NULL) != CUDA_SUCCESS) llb_graph_fail("graph launch failed");
}

static inline llb_dptr llb_alloc_zero(size_t bytes) {
  llb_graph_bail();
  CUdeviceptr p = 0;
  return cuMemAlloc(&p, bytes) == CUDA_SUCCESS
    && cuMemsetD32(p, 0, bytes / 4) == CUDA_SUCCESS ? p : 0;
}

static inline bool llb_zero(llb_dptr p, size_t bytes) {
  llb_graph_bail();
  return cuMemsetD32(p, 0, bytes / 4) == CUDA_SUCCESS;
}

static inline bool llb_read(void* h, llb_dptr p, size_t bytes) {
  llb_graph_bail();
  return cuMemcpyDtoH(h, p, bytes) == CUDA_SUCCESS;
}

// LLB_SPEC: a device copy in launch order (the null stream; no host sync)
static inline bool llb_copy(llb_dptr d, llb_dptr s, size_t bytes) {
  llb_graph_bail();
  return cuMemcpyDtoDAsync(d, s, bytes, NULL) == CUDA_SUCCESS;
}

// LLB_MTP: host bytes onto the device (load time)
static inline bool llb_write(llb_dptr d, const void* h, size_t bytes) {
  llb_graph_bail();
  return cuMemcpyHtoD(d, h, bytes) == CUDA_SUCCESS;
}

// W1-PREFILL: wait for the queued device work (LLB_PF_MS times a fast prefill chunk)
static inline bool llb_sync(void) {
  llb_graph_bail();
  return cuCtxSynchronize() == CUDA_SUCCESS;
}

#define llb_dev(p) ((llb_dptr)(uintptr_t)(p))
#endif
