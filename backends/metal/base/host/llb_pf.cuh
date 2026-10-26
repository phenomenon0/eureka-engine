// llb_pf.cuh (PF27 stage 2): the fast prefill tier's row-batched small kernels, device code
// inlined into the module after tools/pf27/pfmm.cuh (tools/payload-pf.patch). LLB_PREFILL=fast
// only. Each is the exact path's kernel with a token index (blockIdx.y unless noted) and the
// same per-token ops, so the tier's numerics differ from the exact path only in the GEMMs.
// Rows are [T][n]; pos0 / r0 the chunk's first position / conv ring index.

// llb_rmsnorm (any n <= 8192), token blockIdx.x
extern "C" __global__ void __launch_bounds__(1024) pf_rmsnorm(const float* __restrict__ x0,
  const float* __restrict__ w, float* __restrict__ y0, u32 n) {
  __shared__ float a[33];
  const u32 t = threadIdx.x;
  const float* x = x0 + (u64)blockIdx.x * n;
  float* y = y0 + (u64)blockIdx.x * n;
  float acc;
  if (n == 4096) {
    const float* xp = x + (u64)t * 4;
    acc = __fadd_rn(__fadd_rn(__fmul_rn(xp[0], xp[0]), __fmul_rn(xp[1], xp[1])),
      __fadd_rn(__fmul_rn(xp[2], xp[2]), __fmul_rn(xp[3], xp[3])));
  } else {
    float b[8];
    #pragma unroll
    for (u32 k = 0; k < 8; k++) {
      const u64 i = (u64)t * 8 + k;
      b[k] = i < n ? __fmul_rn(x[i], x[i]) : 0.0f;
    }
    acc = __fadd_rn(__fadd_rn(__fadd_rn(b[0], b[1]), __fadd_rn(b[2], b[3])),
      __fadd_rn(__fadd_rn(b[4], b[5]), __fadd_rn(b[6], b[7])));
  }
  const float sum = rms_tree(acc, a, t, 32);
  const float s = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, (float)n),
    __uint_as_float(0x358637BDu))));
  for (u32 i = t; i < n; i += 1024) {
    y[i] = __fmul_rn(__fmul_rn(x[i], s), w[i]);
  }
}

// llb_midconv: block 0 the nm-element middleware, blocks 1.. conv1d over nch channels. Token
// t's taps x[t - 3 + j] come from QKV (the chunk's raw qkv rows) or, before the chunk, from
// the ring's slot (r0 + t + j) % 4 (the exact path's h0 / h1 / h2 at ring index r0 + t)
extern "C" __global__ void __launch_bounds__(256) pf_midconv(
  const float* __restrict__ bpre, const float* __restrict__ apre,
  const float* __restrict__ dtp, const float* __restrict__ sap,
  float* __restrict__ beta, float* __restrict__ dec, u32 nm,
  const float* __restrict__ w, const float* __restrict__ qkv, const float* __restrict__ ring,
  u32 r0, float* __restrict__ y, u32 nch) {
  const u32 tk = blockIdx.y;
  if (blockIdx.x == 0) {
    const u32 i = threadIdx.x, o = tk * nm + i;
    if (i < nm) {
      const float b = bpre[o];
      beta[o] = __fdiv_rn(1.0f, __fadd_rn(1.0f, attn_exp(-b)));
      const float gp = __fmul_rn(mid_sp(__fadd_rn(apre[o], dtp[i])), sap[i]);
      dec[o] = attn_exp(gp);
    }
    return;
  }
  const u32 c = (blockIdx.x - 1) * 256 + threadIdx.x;
  if (c >= nch) {
    return;
  }
  float h[4];
  #pragma unroll
  for (u32 j = 0; j < 4; j++) {
    h[j] = tk + j >= 3 ? qkv[(u64)(tk + j - 3) * nch + c] : ring[(u64)((r0 + tk + j) & 3) * nch + c];
  }
  const float a = __fadd_rn(__fmul_rn(h[0], w[4 * c]), __fmul_rn(h[1], w[4 * c + 1]));
  const float b = __fadd_rn(__fmul_rn(h[2], w[4 * c + 2]), __fmul_rn(h[3], w[4 * c + 3]));
  const float u = __fadd_rn(a, b);
  y[(u64)tk * nch + c] = __fdiv_rn(u, __fadd_rn(1.0f, expf(-u)));
}

// after the chunk: the last min(T, 4) raw qkv rows into the ring as the exact path leaves it
// (step r writes slot (r + 3) % 4); grid (ceil(nch / 256), min(T, 4))
extern "C" __global__ void __launch_bounds__(256) pf_ringw(const float* __restrict__ qkv,
  float* __restrict__ ring, u32 r0, u32 T, u32 nch) {
  const u32 tk = T - 1 - blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
  if (c < nch) {
    ring[(u64)((r0 + tk + 3) & 3) * nch + c] = qkv[(u64)tk * nch + c];
  }
}

// llb_l2norm_qk: grid (2 heads, T); vq / vk rows of stride vs, yq / yk rows of stride ys
extern "C" __global__ void __launch_bounds__(64) pf_l2qk(
  const float* __restrict__ vq0, const float* __restrict__ vk0, u32 vs,
  float* __restrict__ yq0, float* __restrict__ yk0, u32 ys, float sq, u32 heads, u32 rep) {
  __shared__ float a[33];
  const u32 kq = blockIdx.x >= heads, h = blockIdx.x - (kq ? heads : 0);
  const u32 t = threadIdx.x;
  const u64 vo = (u64)blockIdx.y * vs, yo = (u64)blockIdx.y * ys;
  const float sc = kq ? 1.0f : sq;
  const float2 x = ((const float2*)((kq ? vk0 : vq0) + vo + ((u64)h << 7)))[t];
  const float sum = rms_tree(__fadd_rn(__fmul_rn(x.x, x.x), __fmul_rn(x.y, x.y)), a, t, 2);
  const float rt = __fsqrt_rn(sum);
  const float ep = __uint_as_float(0x358637BDu);
  const float s = __fdiv_rn(__fdiv_rn(1.0f, (rt < ep ? ep : rt)), sc);
  const float2 r = make_float2(__fmul_rn(x.x, s), __fmul_rn(x.y, s));
  float* y = (kq ? yk0 : yq0) + yo + ((u64)h << 7);
  for (u32 c = 0; c < rep; c++) {
    ((float2*)(y + (u64)c * heads * 128))[t] = r;
  }
}

// PF28: pf_gdl with 8 lanes a row (16 values each, float4 s + 8 i), 4 rows a warp. S_old.k, S_old.q
// and k.q in one 3-level butterfly; y = decay S.q + delta k.q is S_new.q (fast-tier rounding).
// The next token's q / k / v / dec / bet are loaded ahead. grid (heads, 8), 128 threads
extern "C" __global__ void __launch_bounds__(128) pf_gdl(
  float* __restrict__ S, const float* __restrict__ q, const float* __restrict__ kk, u32 qs,
  const float* __restrict__ v, u32 vs, const float* __restrict__ dec,
  const float* __restrict__ bet, float* __restrict__ y, u32 heads, u32 T) {
  const u32 h = blockIdx.x, s = threadIdx.x & 7, j = blockIdx.y * 16 + (threadIdx.x >> 3);
  float4* Srow = (float4*)(S + (((u64)h * 128 + j) << 7));
  float4 s4[4], k4[4], q4[4];
  #pragma unroll
  for (u32 i = 0; i < 4; i++) {
    s4[i] = Srow[s + 8 * i];
    k4[i] = ((const float4*)(kk + ((u64)h << 7)))[s + 8 * i];
    q4[i] = ((const float4*)(q + ((u64)h << 7)))[s + 8 * i];
  }
  float vn = v[((u64)h << 7) + j], dn = dec[h], bn = bet[h];
  for (u32 tk = 0; tk < T; tk++) {
    float4 kc[4], qc[4];
    #pragma unroll
    for (u32 i = 0; i < 4; i++) {
      kc[i] = k4[i];
      qc[i] = q4[i];
    }
    const float vc = vn, decay = dn, beta = bn;
    if (tk + 1 < T) {
      const u64 rn = (u64)(tk + 1) * qs + ((u64)h << 7);
      #pragma unroll
      for (u32 i = 0; i < 4; i++) {
        k4[i] = ((const float4*)(kk + rn))[s + 8 * i];
        q4[i] = ((const float4*)(q + rn))[s + 8 * i];
      }
      vn = v[(u64)(tk + 1) * vs + ((u64)h << 7) + j];
      dn = dec[(tk + 1) * heads + h];
      bn = bet[(tk + 1) * heads + h];
    }
    float sk = 0.0f, sq = 0.0f, kq = 0.0f;
    #pragma unroll
    for (u32 i = 0; i < 4; i++) {
      sk = fmaf(s4[i].x, kc[i].x, sk); sk = fmaf(s4[i].y, kc[i].y, sk);
      sk = fmaf(s4[i].z, kc[i].z, sk); sk = fmaf(s4[i].w, kc[i].w, sk);
      sq = fmaf(s4[i].x, qc[i].x, sq); sq = fmaf(s4[i].y, qc[i].y, sq);
      sq = fmaf(s4[i].z, qc[i].z, sq); sq = fmaf(s4[i].w, qc[i].w, sq);
      kq = fmaf(kc[i].x, qc[i].x, kq); kq = fmaf(kc[i].y, qc[i].y, kq);
      kq = fmaf(kc[i].z, qc[i].z, kq); kq = fmaf(kc[i].w, qc[i].w, kq);
    }
    #pragma unroll
    for (u32 m = 4; m > 0; m >>= 1) {
      sk += __shfl_xor_sync(0xffffffffu, sk, m);
      sq += __shfl_xor_sync(0xffffffffu, sq, m);
      kq += __shfl_xor_sync(0xffffffffu, kq, m);
    }
    const float delta = (vc - decay * sk) * beta;
    #pragma unroll
    for (u32 i = 0; i < 4; i++) {
      s4[i].x = fmaf(delta, kc[i].x, decay * s4[i].x);
      s4[i].y = fmaf(delta, kc[i].y, decay * s4[i].y);
      s4[i].z = fmaf(delta, kc[i].z, decay * s4[i].z);
      s4[i].w = fmaf(delta, kc[i].w, decay * s4[i].w);
    }
    if (s == 0) {
      y[(u64)tk * qs + ((u64)h << 7) + j] = fmaf(delta, kq, decay * sq);
    }
  }
  #pragma unroll
  for (u32 i = 0; i < 4; i++) {
    Srow[s + 8 * i] = s4[i];
  }
}

// llb_gated_rmsnorm: grid (heads, T), rows of stride heads * 128
extern "C" __global__ void __launch_bounds__(64) pf_grn(
  const float* __restrict__ o0, const float* __restrict__ w,
  const float* __restrict__ z0, float* __restrict__ y0, u32 heads) {
  __shared__ float a[64];
  __shared__ float sv;
  const u32 h = blockIdx.x, t = threadIdx.x;
  const u64 ro = (u64)blockIdx.y * heads * 128 + ((u64)h << 7);
  const float2 x = ((const float2*)(o0 + ro))[t];
  a[t] = __fadd_rn(__fmul_rn(x.x, x.x), __fmul_rn(x.y, x.y));
  __syncthreads();
  #pragma unroll
  for (u32 st = 1; st < 64; st <<= 1) {
    if ((t & (2 * st - 1)) == 0) {
      a[t] = __fadd_rn(a[t], a[t + st]);
    }
    __syncthreads();
  }
  if (t == 0) {
    sv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a[0], 128.0f),
      __uint_as_float(0x358637BDu))));
  }
  __syncthreads();
  const float s = sv;
  const float2 wv = ((const float2*)w)[t];
  const float2 zz = ((const float2*)(z0 + ro))[t];
  const float zx = __fdiv_rn(zz.x, __fadd_rn(1.0f, expf(-zz.x)));
  const float zy = __fdiv_rn(zz.y, __fadd_rn(1.0f, expf(-zz.y)));
  ((float2*)(y0 + ro))[t] = make_float2(
    __fmul_rn(__fmul_rn(__fmul_rn(x.x, s), wv.x), zx),
    __fmul_rn(__fmul_rn(__fmul_rn(x.y, s), wv.y), zy));
}

// llb_qkr: grid (heads, T), token t at pos0 + t; x rows of stride n (the GEMM's), y rows of
// stride n too (q: heads at h * 256, gates after all heads; k: n = heads * 256)
extern "C" __global__ void __launch_bounds__(256) pf_qkr(
  const float* __restrict__ x0, const float* __restrict__ w,
  float* __restrict__ y0, u32 n, u32 qk, u32 pos0) {
  __shared__ float a[256];
  __shared__ float sv;
  const u32 t = threadIdx.x, pos = pos0 + blockIdx.y;
  const float* xr = x0 + (u64)blockIdx.y * n + (u64)blockIdx.x * (qk != 0 ? 512 : 256);
  float* y = y0 + (u64)blockIdx.y * n;
  float* yr = y + (u64)blockIdx.x * 256;
  const float xv = xr[t];
  const float xb = xr[t + 32];
  const float xg = qk != 0 ? xr[256 + t] : 0.0f;
  a[t] = __fmul_rn(xv, xv);
  __syncthreads();
  #pragma unroll
  for (u32 st = 1; st < 256; st <<= 1) {
    if ((t & (2 * st - 1)) == 0) {
      a[t] = __fadd_rn(a[t], a[t + st]);
    }
    __syncthreads();
  }
  if (t == 0) {
    sv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a[0], 256.0f),
      __uint_as_float(0x358637BDu))));
  }
  __syncthreads();
  const float s = sv;
  const float v = __fmul_rn(__fmul_rn(xv, s), w[t]);
  if (t < 32) {
    const float th = __fmul_rn((float)pos, (float)pow((double)10000000.0f,
      (double)__fdiv_rn(-(float)(2 * t), 64.0f)));
    const float co = (float)cos((double)th), si = (float)sin((double)th);
    const float b = __fmul_rn(__fmul_rn(xb, s), w[t + 32]);
    yr[2 * t]     = __fsub_rn(__fmul_rn(v, co), __fmul_rn(b, si));
    yr[2 * t + 1] = __fadd_rn(__fmul_rn(v, si), __fmul_rn(b, co));
  } else if (t >= 64) {
    yr[t] = v;
  }
  if (qk != 0) {
    y[(u64)gridDim.x * 256 + (u64)blockIdx.x * 256 + t] = xg;
  }
}

// the chunk's k / v rows (kv floats each) into the caches' slots pos0 + t: grid (T, 2), 1024
extern "C" __global__ void __launch_bounds__(1024) pf_slot(const float* __restrict__ k,
  const float* __restrict__ v, float* __restrict__ kc, float* __restrict__ vc, u32 pos0, u32 kv) {
  const u32 t = blockIdx.x;
  for (u32 i = threadIdx.x; i < kv; i += 1024) {
    (blockIdx.y ? vc : kc)[(u64)(pos0 + t) * kv + i] = (blockIdx.y ? v : k)[(u64)t * kv + i];
  }
}

// llb_outputgate: grid (ceil(n / 256), T); x rows of stride n, the gate at g + t gs
extern "C" __global__ void __launch_bounds__(256) pf_og(const float* __restrict__ x,
  const float* __restrict__ g, u32 gs, float* __restrict__ y, u32 n) {
  const u32 i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) {
    const u64 o = (u64)blockIdx.y * n + i;
    y[o] = __fmul_rn(x[o], __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g[(u64)blockIdx.y * gs + i]))));
  }
}

// causal attention over the chunk, flash style on f16 tensor cores (mma m16n8k16, f32 accumulate;
// q, k, v, p rounded to f16 = llama's -fa 0 numerics, not llb_attn4's bits). The grp q heads of
// a kv head share one tile: row i of kv head blockIdx.x is token i / grp, q head
// blockIdx.x grp + i % grp; grid (nkv, ceil(T grp / 64)), 128 threads (a warp per 16 rows),
// dynamic smem PF_ATT_SM: a 32-key K tile [key][264] and its V transposed [dim][40], f16.
// Query t (pos pos0 + t) of head h at q + t qs + h 256; keys / values slots 0 .. pos0 + t of
// kc / vc (rows of kv, head h / grp at (h / grp) 256); out rows of stride os, head h at h 256.
#define PF_ATT_SM ((32 * 264 + 256 * 40) * 2)
__device__ __forceinline__ u32 pf_h2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ void pf_hmma(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
    "{%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
extern "C" __global__ void __launch_bounds__(128) pf_attn(const float* __restrict__ q, u32 qs,
  const float* __restrict__ kc, const float* __restrict__ vc, u32 kv, float* __restrict__ out,
  u32 os, u32 pos0, u32 T, u32 grp) {
  extern __shared__ u32 pfa_sm[];
  u32* Ks = pfa_sm;               // 32 keys x 132 (264 halves)
  u32* Vt = pfa_sm + 32 * 132;    // 256 dims x 20 (40 halves)
  const u32 tid = threadIdx.x, w = tid >> 5, g = (tid >> 2) & 7, c = tid & 3;
  const u32 R = T * grp, rb = blockIdx.y * 64, gk = blockIdx.x * 256;
  const u32 re = R < rb + 64 ? R : rb + 64, kend = pos0 + (re - 1) / grp + 1;
  // this thread's two rows (g, g + 8 of the warp's 16): token, q head, last visible key
  u32 tr[2], qp[2];
  const float* qr[2];
#pragma unroll
  for (u32 j = 0; j < 2; j++) {
    const u32 i = rb + w * 16 + g + 8 * j, ii = i < R ? i : R - 1;
    tr[j] = ii / grp;
    qp[j] = pos0 + tr[j];
    qr[j] = q + (u64)tr[j] * qs + (blockIdx.x * grp + ii % grp) * 256;
  }
  // q (scaled by 1 / 16, exact) as A fragments: 16 k-steps of 16 dims
  u32 qf[16][4];
#pragma unroll
  for (u32 s = 0; s < 16; s++) {
    const u32 d = s * 16 + 2 * c;
    qf[s][0] = pf_h2(qr[0][d] * 0.0625f, qr[0][d + 1] * 0.0625f);
    qf[s][1] = pf_h2(qr[1][d] * 0.0625f, qr[1][d + 1] * 0.0625f);
    qf[s][2] = pf_h2(qr[0][d + 8] * 0.0625f, qr[0][d + 9] * 0.0625f);
    qf[s][3] = pf_h2(qr[1][d + 8] * 0.0625f, qr[1][d + 9] * 0.0625f);
  }
  float o[32][4];
#pragma unroll
  for (u32 n = 0; n < 32; n++) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  }
  float mx[2] = {__uint_as_float(0xFF800000u), __uint_as_float(0xFF800000u)}, l[2] = {0.0f, 0.0f};
  for (u32 k0 = 0; k0 < kend; k0 += 32) {
    __syncthreads();
    for (u32 i = tid; i < 32 * 64; i += 128) {  // K: key i / 64, dims 4 (i % 64)
      const u32 r = i >> 6, d = (i & 63) * 4;
      float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (k0 + r < kend) x = *(const float4*)(kc + (u64)(k0 + r) * kv + gk + d);
      Ks[r * 132 + d / 2] = pf_h2(x.x, x.y);
      Ks[r * 132 + d / 2 + 1] = pf_h2(x.z, x.w);
    }
    for (u32 i = tid; i < 16 * 64; i += 128) {  // V: keys 2 (i / 64) + 0/1, dims 4 (i % 64)
      const u32 r = (i >> 6) * 2, d = (i & 63) * 4;
      float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = a;
      if (k0 + r < kend) a = *(const float4*)(vc + (u64)(k0 + r) * kv + gk + d);
      if (k0 + r + 1 < kend) b = *(const float4*)(vc + (u64)(k0 + r + 1) * kv + gk + d);
      Vt[d * 20 + r / 2] = pf_h2(a.x, b.x);
      Vt[(d + 1) * 20 + r / 2] = pf_h2(a.y, b.y);
      Vt[(d + 2) * 20 + r / 2] = pf_h2(a.z, b.z);
      Vt[(d + 3) * 20 + r / 2] = pf_h2(a.w, b.w);
    }
    __syncthreads();
    float sc[4][4];
#pragma unroll
    for (u32 n = 0; n < 4; n++) {
      sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
      const u32* kb = Ks + (n * 8 + g) * 132 + c;
#pragma unroll
      for (u32 s = 0; s < 16; s++) {
        pf_hmma(sc[n], qf[s][0], qf[s][1], qf[s][2], qf[s][3], kb[s * 8], kb[s * 8 + 4]);
      }
    }
    // causal mask, online softmax per row (a row's 32 scores sit on the 4 lanes of a quad)
    float al[2];
#pragma unroll
    for (u32 j = 0; j < 2; j++) {
      float m = mx[j];
#pragma unroll
      for (u32 n = 0; n < 4; n++) {
#pragma unroll
        for (u32 e = 0; e < 2; e++) {
          const u32 key = k0 + n * 8 + 2 * c + e;
          if (key > qp[j]) sc[n][2 * j + e] = __uint_as_float(0xFF800000u);
          m = fmaxf(m, sc[n][2 * j + e]);
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
      al[j] = expf(mx[j] - m);
      mx[j] = m;
      float sum = 0.0f;
#pragma unroll
      for (u32 n = 0; n < 4; n++) {
#pragma unroll
        for (u32 e = 0; e < 2; e++) {
          sc[n][2 * j + e] = expf(sc[n][2 * j + e] - m);
          sum += sc[n][2 * j + e];
        }
      }
      l[j] = l[j] * al[j] + sum;
    }
#pragma unroll
    for (u32 n = 0; n < 32; n++) {
      o[n][0] *= al[0];
      o[n][1] *= al[0];
      o[n][2] *= al[1];
      o[n][3] *= al[1];
    }
    // P (C layout of two key n-tiles = the A layout of one k-step) x V
#pragma unroll
    for (u32 s = 0; s < 2; s++) {
      const u32 a0 = pf_h2(sc[2 * s][0], sc[2 * s][1]), a1 = pf_h2(sc[2 * s][2], sc[2 * s][3]);
      const u32 a2 = pf_h2(sc[2 * s + 1][0], sc[2 * s + 1][1]);
      const u32 a3 = pf_h2(sc[2 * s + 1][2], sc[2 * s + 1][3]);
#pragma unroll
      for (u32 n = 0; n < 32; n++) {
        const u32* vb = Vt + (n * 8 + g) * 20 + s * 8 + c;
        pf_hmma(o[n], a0, a1, a2, a3, vb[0], vb[4]);
      }
    }
  }
#pragma unroll
  for (u32 j = 0; j < 2; j++) {
    l[j] += __shfl_xor_sync(0xffffffffu, l[j], 1);
    l[j] += __shfl_xor_sync(0xffffffffu, l[j], 2);
  }
#pragma unroll
  for (u32 j = 0; j < 2; j++) {
    const u32 i = rb + w * 16 + g + 8 * j;
    if (i < R) {
      float* y = out + (u64)tr[j] * os + (blockIdx.x * grp + i % grp) * 256 + 2 * c;
#pragma unroll
      for (u32 n = 0; n < 32; n++) {
        *(float2*)(y + n * 8) = make_float2(o[n][2 * j] / l[j], o[n][2 * j + 1] / l[j]);
      }
    }
  }
}

// the deferred step's answer with the tier on: top-1 id 1e9 (no token's), the prompt loop's
// cue to send the rest of the prompt as one batch bang
extern "C" __global__ void pf_mark(float* y) {
  y[1] = 1.0e9f;
}
