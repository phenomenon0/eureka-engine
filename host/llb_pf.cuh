// llb_pf.cuh (PF27 stage 2): the fast prefill tier's row-batched small kernels, device code
// inlined into the module after tools/pf27/pfmm.cuh (tools/payload-pf.patch). LLB_PREFILL=fast
// only. Each is the exact path's kernel with a token index (blockIdx.y unless noted) and the
// same per-token ops, so the tier's numerics differ from the exact path only in the GEMMs.
// Rows are [T][n]; pos0 / r0 the chunk's first position / conv ring index.

// llb_rmsnorm (any n <= 8192; past it rms_wide, tools/payload-anyshape.patch), token blockIdx.x
extern "C" __global__ void __launch_bounds__(1024) pf_rmsnorm(const float* __restrict__ x0,
  const float* __restrict__ w, float* __restrict__ y0, u32 n) {
  __shared__ float a[33];
  const u32 t = threadIdx.x;
  const float* x = x0 + (u64)blockIdx.x * n;
  float* y = y0 + (u64)blockIdx.x * n;
  float acc;
  if (n > 8192) {
    acc = rms_wide(x, nullptr, nullptr, n, t);
  } else if (n == 4096) {
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
    LLB_EPS)));
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
  const float ep = LLB_EPS;
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
      LLB_EPS)));
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

#ifndef LLB_MR_DEF
#define LLB_MR_DEF
// (FUS-V46 V4, tools/payload-imrope.patch) IMRoPE: the image map llb_mr = {s0, n, nx, sh, p0}, set by the host
// (llb_host.c llb_img_step) at each image's first <|image_pad|> slot, zero otherwise. The rope kernels'
// pos stays the KV SLOT; the rope ANGLE of rotary pair t is llb_mrp(slot, t): mtmd's MROPE (t, h, w) =
// (p0, p0 + i / nx, p0 + i % nx) for image row i = slot - s0, by sector t % 3 (ggml imrope, sections
// [11, 11, 10], every t < 32 inside them); slots after the image resume at slot - sh (MIMG: p0 and sh carry
// the earlier images' n - max(nx, ny), mtmd's n_pos); n == 0 (every text-only run) is the slot: the old expression
__device__ u32 llb_mr[5];
__device__ __forceinline__ u32 llb_mrp(u32 s, u32 t) {
  const u32 s0 = llb_mr[0], n = llb_mr[1];
  if (n == 0 || s < s0) return s;
  if (s - s0 >= n) return s - llb_mr[3];
  const u32 i = s - s0, c = t % 3, p0 = llb_mr[4];
  return c == 0 ? p0 : c == 1 ? p0 + i / llb_mr[2] : p0 + i % llb_mr[2];
}
#endif

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
  const u32 rh = llb_cfg[2] >> 1;  // rope pairs (tools/payload-anyshape.patch)
  const float xb = xr[t + rh];
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
      LLB_EPS)));
  }
  __syncthreads();
  const float s = sv;
  const float v = __fmul_rn(__fmul_rn(xv, s), w[t]);
  if (t < rh) {
    const float th = __fmul_rn((float)llb_mrp(pos, t), (float)pow((double)__uint_as_float(llb_cfg[1]),
      (double)__fdiv_rn(-(float)(2 * t), (float)(2 * rh))));
    const float co = (float)cos((double)th), si = (float)sin((double)th);
    const float b = __fmul_rn(__fmul_rn(xb, s), w[t + rh]);
    yr[2 * t]     = __fsub_rn(__fmul_rn(v, co), __fmul_rn(b, si));
    yr[2 * t + 1] = __fadd_rn(__fmul_rn(v, si), __fmul_rn(b, co));
  } else if (t >= 2 * rh) {
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

// ---- reports/cuda-catchup: the routed experts on int8 activations (not exact) ----
// One family for both new MoE tiers: the dp4a decode tier (LLB_MOE_DP4A=1, one token, the k chosen
// experts) and the fast MoE prefill (LLB_PREFILL=fast, T tokens grouped by expert). x is pf_q8x's
// q8_1 (per 32 values d = amax / 127, q = round(x / d), s = sum x), llama.cpp's mmvq numerics:
// a 32-block's dot is (weight scale) d (sum of int8 products) [- (weight min) s for Q4_K / Q5_K].
// The weights are the GGUF's own blocks (the expert stacks load from the GGUF): Q8_0 34 B / 32,
// Q4_K 144 B / 256, Q5_K 176 B / 256, Q6_K 210 B / 256 (format codes 1, 3, 9, 4).
// A group g is one expert E (gid[g], or g) over the slots off[g] .. off[g + 1] (or g .. g + 1):
// slot s reads x row xmap[s] (or 0) and writes output row s.
INLINE u32 mx_u32(const u8* p) {  // 2-aligned
  const unsigned short* h = (const unsigned short*)p;
  return (u32)h[0] | (u32)h[1] << 16;
}
INLINE void mx_scmin(const u8* q, u32 j, u32* sc, u32* m) {  // ggml get_scale_min_k4
  if (j < 4) {
    *sc = q[j] & 63; *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}
// block b (32 values) of a row in format f against x block (qx 32 int8, e = {d, s})
INLINE float mx_blk(const u8* __restrict__ row, u32 f, u32 b, const int* __restrict__ qx, float2 e) {
  if (f == 3 || f == 9) {
    const u32 sb = b >> 3, j = b & 7, sh = 4 * (j & 1);
    const u8* B = row + (u64)sb * (f == 3 ? 144 : 176);
    const uint4* qs = (const uint4*)(B + (f == 3 ? 16 : 48) + 32 * (j >> 1));
    u32 sc, m;
    mx_scmin(B + 4, j, &sc, &m);
    const uint4 a0 = qs[0], a1 = qs[1];
    const u32 w[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
    uint4 h0 = make_uint4(0, 0, 0, 0), h1 = h0;
    if (f == 9) {
      h0 = ((const uint4*)(B + 16))[0]; h1 = ((const uint4*)(B + 16))[1];
    }
    const u32 hw[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
    int acc = 0;
    #pragma unroll
    for (u32 i = 0; i < 8; i++) {
      u32 v = (w[i] >> sh) & 0x0F0F0F0Fu;
      if (f == 9) v |= ((hw[i] >> j) & 0x01010101u) << 4;
      acc = __dp4a((int)v, qx[i], acc);
    }
    const float d = llb_f16(B[0] | B[1] << 8), dm = llb_f16(B[2] | B[3] << 8);
    return d * (float)sc * e.x * (float)acc - dm * (float)m * e.y;
  }
  if (f == 4) {
    const u32 sb = b >> 3, bb = b & 7, n = bb >> 2, quad = bb & 3;
    const u8* B = row + (u64)sb * 210;
    const u8* ql = B + 64 * n + 32 * (quad & 1);
    const u8* qh = B + 128 + 32 * n;
    const u32 nsh = quad >= 2 ? 4 : 0, qsh = 2 * quad;
    int a0 = 0, a1 = 0;
    #pragma unroll
    for (u32 i = 0; i < 8; i++) {
      const u32 lo = (mx_u32(ql + 4 * i) >> nsh) & 0x0F0F0F0Fu, hi = ((mx_u32(qh + 4 * i) >> qsh) & 0x03030303u) << 4;
      const int v = (int)__vsub4(lo | hi, 0x20202020u);
      if (i < 4) a0 = __dp4a(v, qx[i], a0); else a1 = __dp4a(v, qx[i], a1);
    }
    const signed char* sc = (const signed char*)(B + 192 + 8 * n + 2 * quad);
    const float d = llb_f16(B[208] | B[209] << 8);
    return d * e.x * ((float)sc[0] * (float)a0 + (float)sc[1] * (float)a1);
  }
  // Q8_0
  const u8* B = row + (u64)b * 34;
  int acc = 0;
  #pragma unroll
  for (u32 i = 0; i < 8; i++) acc = __dp4a((int)mx_u32(B + 2 + 4 * i), qx[i], acc);
  return llb_f16(B[0] | B[1] << 8) * e.x * (float)acc;
}
// a row's dot over nb blocks, the warp's lanes each a block in turn, then the xor tree
INLINE float mx_dot(const u8* __restrict__ row, u32 f, u32 nb, const signed char* __restrict__ qx,
  const float2* __restrict__ xd) {
  const u32 l = threadIdx.x & 31;
  float s = 0.0f;
  for (u32 b = l; b < nb; b += 32) {
    const int4 q0 = ((const int4*)(qx + 32 * b))[0], q1 = ((const int4*)(qx + 32 * b))[1];
    const int q[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
    s += mx_blk(row, f, b, q, xd[b]);
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
  return s;
}
// gate + up + silu: act[s ef + r] = silu(G_E row r . x) (U_E row r . x); a warp a row, the group's
// slots in turn (the row stays in L1). G / U the stacks (U = G + ef rows when fused), es the stride
extern "C" __global__ void __launch_bounds__(128) llb_mx_gu(const u8* __restrict__ G, const u8* __restrict__ U,
  u64 es, u32 rbg, u32 rbu, u32 fg, u32 fu, u32 c, const signed char* __restrict__ qx,
  const float2* __restrict__ xd, const u32* __restrict__ gid, const u32* __restrict__ off,
  const u32* __restrict__ xmap, float* __restrict__ act, u32 ef) {
  const u32 g = blockIdx.y, s0 = off ? off[g] : g, s1 = off ? off[g + 1] : g + 1;
  if (s0 == s1) {
    return;
  }
  const u64 e = gid ? gid[g] : g;
  const u32 nb = c / 32;
  for (u32 r = blockIdx.x * 4 + (threadIdx.x >> 5); r < ef; r += gridDim.x * 4) {
    const u8* gr = G + e * es + (u64)r * rbg;
    const u8* ur = U + e * es + (u64)r * rbu;
    for (u32 s = s0; s < s1; s++) {
      const u64 t = xmap ? xmap[s] : 0;
      const float gv = mx_dot(gr, fg, nb, qx + t * c, xd + t * nb);
      const float uv = mx_dot(ur, fu, nb, qx + t * c, xd + t * nb);
      if ((threadIdx.x & 31) == 0) {
        act[(u64)s * ef + r] = gv / (1.0f + expf(-gv)) * uv;
      }
    }
  }
}
// a row's dot on a half warp (nb <= 16 blocks): lanes 0..15 and 16..31 each take their own row
INLINE float mx_dot16(const u8* __restrict__ row, u32 f, u32 nb, const signed char* __restrict__ qx,
  const float2* __restrict__ xd) {
  const u32 l = threadIdx.x & 15;
  float s = 0.0f;
  if (l < nb) {
    const int4 q0 = ((const int4*)(qx + 32 * l))[0], q1 = ((const int4*)(qx + 32 * l))[1];
    const int q[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
    s = mx_blk(row, f, l, q, xd[l]);
  }
  #pragma unroll
  for (u32 m = 8; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
  return s;
}
// down: out[s dm + r] = D_E row r . act_s (act q8_1: aq [slots][ef], ad)
extern "C" __global__ void __launch_bounds__(128) llb_mx_dn(const u8* __restrict__ D, u64 es, u32 rbd, u32 fd,
  u32 ef, const signed char* __restrict__ aq, const float2* __restrict__ ad, const u32* __restrict__ gid,
  const u32* __restrict__ off, float* __restrict__ out, u32 dm) {
  const u32 g = blockIdx.y, s0 = off ? off[g] : g, s1 = off ? off[g + 1] : g + 1;
  if (s0 == s1) {
    return;
  }
  const u64 e = gid ? gid[g] : g;
  const u32 nb = ef / 32;
  if (nb <= 16) {
    // two rows a warp, a half warp each (the grid covers dm / 8 blocks of 8 rows)
    const u32 h = (threadIdx.x >> 4) & 1;
    for (u32 r0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 2; r0 < dm; r0 += gridDim.x * 8) {
      const u32 r = r0 + h < dm ? r0 + h : dm - 1;
      const u8* dr = D + e * es + (u64)r * rbd;
      for (u32 s = s0; s < s1; s++) {
        const float v = mx_dot16(dr, fd, nb, aq + (u64)s * ef, ad + (u64)s * nb);
        if ((threadIdx.x & 15) == 0 && r0 + h < dm) {
          out[(u64)s * dm + r] = v;
        }
      }
    }
    return;
  }
  for (u32 r = blockIdx.x * 4 + (threadIdx.x >> 5); r < dm; r += gridDim.x * 4) {
    const u8* dr = D + e * es + (u64)r * rbd;
    for (u32 s = s0; s < s1; s++) {
      const float v = mx_dot(dr, fd, nb, aq + (u64)s * ef, ad + (u64)s * nb);
      if ((threadIdx.x & 31) == 0) {
        out[(u64)s * dm + r] = v;
      }
    }
  }
}
// y[t][r] = (sum over ranks j (in order) wt[t nk + j] out[slot(t, j)][r] + shd[t][r] sgo[t]) + x1[t][r];
// slot(t, j) = inv[t nk + j] (or j: one token). Token blockIdx.y
extern "C" __global__ void __launch_bounds__(256) llb_mx_comb(const float* __restrict__ out,
  const float* __restrict__ wt, const u32* __restrict__ inv, u32 nk, const float* __restrict__ shd,
  const float* __restrict__ sgo, const float* __restrict__ x1, float* __restrict__ y, u32 dm) {
  const u32 t = blockIdx.y, r = blockIdx.x * 256 + threadIdx.x;
  if (r >= dm) {
    return;
  }
  float mo = 0.0f;
  for (u32 j = 0; j < nk; j++) {
    const u64 s = inv ? inv[(u64)t * nk + j] : j;
    mo = j == 0 ? out[s * dm + r] * wt[(u64)t * nk] : mo + out[s * dm + r] * wt[(u64)t * nk + j];
  }
  const u64 o = (u64)t * dm + r;
  y[o] = (mo + shd[o] * sgo[t]) + x1[o];
}
// the router (F32 rows: ne experts, + the shared gate row when nr = ne + 1) for T tokens:
// lg[t nr + r] = W row r . h_t; a warp a (row, token)
extern "C" __global__ void __launch_bounds__(128) llb_mx_route(const float* __restrict__ W, u32 nr, u32 c,
  const float* __restrict__ h, float* __restrict__ lg) {
  const u32 t = blockIdx.y, r = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (r >= nr) {
    return;
  }
  const float4* w = (const float4*)(W + (u64)r * c);
  const float4* x = (const float4*)(h + (u64)t * c);
  float s = 0.0f;
  for (u32 i = l; i < c / 4; i += 32) {
    const float4 a = w[i], b = x[i];
    s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
  if (l == 0) {
    lg[(u64)t * nr + r] = s;
  }
}
// the router for T tokens as a tiled F32 GEMM (the fast prefill): lg[t nr + r] = W row r . h_t;
// a block 64 rows x 64 tokens, k steps of 32 through shared memory, a thread 4 x 4
extern "C" __global__ void __launch_bounds__(256) llb_mx_rgemm(const float* __restrict__ W, u32 nr, u32 c,
  const float* __restrict__ h, u32 T, float* __restrict__ lg) {
  __shared__ float ws[32][65], xs[32][65];
  const u32 tx = threadIdx.x & 15, ty = threadIdx.x >> 4, r0 = blockIdx.y * 64, t0 = blockIdx.x * 64;
  float acc[4][4];
  #pragma unroll
  for (u32 i = 0; i < 4; i++)
    #pragma unroll
    for (u32 j = 0; j < 4; j++) acc[i][j] = 0.0f;
  for (u32 k0 = 0; k0 < c; k0 += 32) {
    for (u32 i = threadIdx.x; i < 64 * 32; i += 256) {
      const u32 rr = i >> 5, kk = i & 31;
      ws[kk][rr] = r0 + rr < nr ? W[(u64)(r0 + rr) * c + k0 + kk] : 0.0f;
      xs[kk][rr] = t0 + rr < T ? h[(u64)(t0 + rr) * c + k0 + kk] : 0.0f;
    }
    __syncthreads();
    #pragma unroll 8
    for (u32 kk = 0; kk < 32; kk++) {
      float a[4], b[4];
      #pragma unroll
      for (u32 i = 0; i < 4; i++) {
        a[i] = ws[kk][ty * 4 + i];
        b[i] = xs[kk][tx * 4 + i];
      }
      #pragma unroll
      for (u32 i = 0; i < 4; i++)
        #pragma unroll
        for (u32 j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (u32 i = 0; i < 4; i++)
    #pragma unroll
    for (u32 j = 0; j < 4; j++) {
      const u32 r = r0 + ty * 4 + i, t = t0 + tx * 4 + j;
      if (r < nr && t < T) lg[(u64)t * nr + r] = acc[i][j];
    }
}
// the top k of token blockIdx.y (llb_moe_topk's math: softmax with the exp through F64, the k
// largest p, ties to the lower index, renormalized with the 6.1e-5 clamp); sgo[t] = sigmoid of
// the shared gate logit (lg[t nr + ne] when nr > ne, else sg[t]). pv: T ne scratch
extern "C" __global__ void __launch_bounds__(256) llb_mx_topk(const float* __restrict__ lg, u32 nr, u32 ne,
  u32 nk, const float* __restrict__ sg, float* __restrict__ pv0, u32* __restrict__ ids0, float* __restrict__ wt0,
  float* __restrict__ sgo) {
  __shared__ float rv[8];
  __shared__ u32 ri[8];
  __shared__ float bc;
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5, tk = blockIdx.y;
  const float* x = lg + (u64)tk * nr;
  float* pv = pv0 + (u64)tk * ne;
  u32* ids = ids0 + (u64)tk * nk;
  float* wt = wt0 + (u64)tk * nk;
  float mx = __uint_as_float(0xFF800000u);
  for (u32 e = t; e < ne; e += 256) mx = fmaxf(mx, x[e]);
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
  if (l == 0) rv[w] = mx;
  __syncthreads();
  if (t == 0) {
    float v = rv[0];
    for (u32 k = 1; k < 8; k++) v = fmaxf(v, rv[k]);
    bc = v;
  }
  __syncthreads();
  mx = bc;
  float sm = 0.0f;
  for (u32 e = t; e < ne; e += 256) {
    const float v = attn_exp(x[e] - mx);
    pv[e] = v;
    sm += v;
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, m);
  __syncthreads();
  if (l == 0) rv[w] = sm;
  __syncthreads();
  if (t == 0) {
    float v = 0.0f;
    for (u32 k = 0; k < 8; k++) v += rv[k];
    bc = v;
  }
  __syncthreads();
  const float inv = 1.0f / bc;
  for (u32 e = t; e < ne; e += 256) pv[e] *= inv;
  __syncthreads();
  for (u32 j = 0; j < nk; j++) {
    float bv = -1.0f;
    u32 bix = 0xffffffffu;
    for (u32 e = t; e < ne; e += 256) {
      const float v = pv[e];
      if (v > bv) {
        bv = v; bix = e;
      }
    }
    #pragma unroll
    for (u32 m = 16; m > 0; m >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
      const u32 oi = __shfl_xor_sync(0xffffffffu, bix, m);
      if (ov > bv || (ov == bv && oi < bix)) {
        bv = ov; bix = oi;
      }
    }
    if (l == 0) {
      rv[w] = bv; ri[w] = bix;
    }
    __syncthreads();
    if (t == 0) {
      float v = rv[0];
      u32 q = ri[0];
      for (u32 k = 1; k < 8; k++) {
        if (rv[k] > v || (rv[k] == v && ri[k] < q)) {
          v = rv[k]; q = ri[k];
        }
      }
      ids[j] = q;
      wt[j] = v;
      pv[q] = -2.0f;
    }
    __syncthreads();
  }
  if (t == 0) {
    float s = 0.0f;
    for (u32 j = 0; j < nk; j++) s += wt[j];
    s = s < 6.103515625e-5f ? 6.103515625e-5f : s;
    for (u32 j = 0; j < nk; j++) wt[j] = wt[j] / s;
    const float g = nr > ne ? x[ne] : sg[tk];
    sgo[tk] = 1.0f / (1.0f + attn_exp(-g));
  }
}
// group the T nk (token, rank) slots by expert, in (token, rank) order within an expert: block e
// counts the slots of experts < e (its offset) and writes its own; off[e] (off[ne] = T nk), xmap
// [slot] = token, inv[t nk + j] = slot
extern "C" __global__ void __launch_bounds__(256) llb_mx_group(const u32* __restrict__ ids, u32 n, u32 nk,
  u32 ne, u32* __restrict__ off, u32* __restrict__ xmap, u32* __restrict__ inv) {
  __shared__ u32 wc[8];
  __shared__ u32 base;
  const u32 e = blockIdx.x, t = threadIdx.x, l = t & 31, w = t >> 5;
  u32 lt = 0;
  for (u32 i = t; i < n; i += 256) lt += ids[i] < e;
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) lt += __shfl_xor_sync(0xffffffffu, lt, m);
  if (l == 0) wc[w] = lt;
  __syncthreads();
  if (t == 0) {
    u32 v = 0;
    for (u32 k = 0; k < 8; k++) v += wc[k];
    base = v;
    off[e] = v;
    if (e + 1 == ne) off[ne] = n;
  }
  __syncthreads();
  for (u32 i0 = 0; i0 < n; i0 += 256) {
    const u32 i = i0 + t;
    const bool hit = i < n && ids[i] == e;
    const u32 bal = __ballot_sync(0xffffffffu, hit);
    __syncthreads();
    if (l == 0) wc[w] = __popc(bal);
    __syncthreads();
    u32 before = base;
    for (u32 k = 0; k < w; k++) before += wc[k];
    const u32 at = before + __popc(bal & ((1u << l) - 1));
    if (hit) {
      xmap[at] = i / nk;
      inv[i] = at;
    }
    __syncthreads();
    if (t == 0) {
      u32 v = 0;
      for (u32 k = 0; k < 8; k++) v += wc[k];
      base += v;
    }
    __syncthreads();
  }
}
// the fast MoE prefill's grouped GEMMs: expert blockIdx.z over its slots off[z] .. off[z + 1]
// (rows of the slot-ordered q8_1 x, qx / xd, and of y), one pfmm leaf call per expert on its
// own weights; blocks past an expert's slots return. Q4_K / Q5_K / Q6_K experts (the GGUF's
// blocks are the pack's); gate / up pairs on EP 2 (y = silu(g) u), down on EP 0
#define PFX_HEAD(c) \
  const u32 z = blockIdx.z, o = off[z], m = off[z + 1] - o; \
  if (blockIdx.x * TK >= m) return; \
  const signed char* q = qx + (u64)o * (c); const float2* d = xd + (u64)o * ((c) / 32)
template <int TY, int NT, int EP>
INLINE void pfx_wide(const u8* G, const u8* U, u64 es, u32 n, u32 c, const signed char* __restrict__ qx,
  const float2* __restrict__ xd, const u32* __restrict__ off, float* __restrict__ y) {
  constexpr u32 TK = 8 * NT * 2;
  PFX_HEAD(c);
  pf_wide<TY, 4, 2, NT, EP>(G + z * es, U + z * es, n, c, q, d, m, y + (u64)o * n, nullptr, blockIdx.y);
}
#define PFX_K(nm, TY, NT, EP) \
  extern "C" __global__ void __launch_bounds__(256, 1) nm(const u8* __restrict__ G, const u8* __restrict__ U, \
    u64 es, u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd, \
    const u32* __restrict__ off, float* __restrict__ y) { pfx_wide<TY, NT, EP>(G, U, es, n, c, qx, xd, off, y); }
PFX_K(pfx_q4k_g, 3, 1, 2)
PFX_K(pfx_q4k_m, 3, 1, 0)
PFX_K(pfx_q5k_g, 5, 1, 2)
PFX_K(pfx_q5k_m, 5, 1, 0)
extern "C" __global__ void __launch_bounds__(256, 1) pfx_q6k_m(const u8* __restrict__ G, const u8* __restrict__ U,
  u64 es, u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd,
  const u32* __restrict__ off, float* __restrict__ y) {
  constexpr u32 TK = 64;
  PFX_HEAD(c);
  pf_wide6<0>(G + z * es, n, c, q, d, m, y + (u64)o * n, nullptr);
}
// the slots' x rows in slot order: q8_1 row xmap[s] of qx / xd into row s of qg / dg (c columns)
extern "C" __global__ void __launch_bounds__(256) llb_mx_gather(const signed char* __restrict__ qx,
  const float2* __restrict__ xd, const u32* __restrict__ xmap, u32 c, signed char* __restrict__ qg,
  float2* __restrict__ dg) {
  const u64 s = blockIdx.x, t = xmap[s];
  for (u32 i = threadIdx.x; i < c / 16; i += 256) {
    ((int4*)(qg + s * c))[i] = ((const int4*)(qx + t * c))[i];
  }
  for (u32 i = threadIdx.x; i < c / 32; i += 256) {
    dg[s * (c / 32) + i] = xd[t * (c / 32) + i];
  }
}

// ATTN4S: llb_attn4 split over slot chunks (LONGDEC: one block a head left 16 blocks on the
// board, 24.8 ms/token at 14,426 ids). Four launches, grid (NC chunks, nkv kv heads) for the
// first three, one block a kv head's grp q heads so K / V are read once, not grp times. Every
// reduction of llb_attn4 is a perfect pairwise tree over T2 = the power of two >= n (lower
// index on the left); a tree over T2 is the tree over its NC = T2 / C chunks of the chunks'
// perfect subtrees, so the max, the softmax sum and the weighted V keep llb_attn4's bits.
// C = T2 up to 256, else max(256, T2 / 256): NC <= 256. Per head h, sc + (h << 17) holds the
// scores then the exps; pr + (h << 17) holds the chunks' V partials (NC x 256 <= 2^16 floats),
// the chunk maxes at 2^16 and the chunk sums at 2^16 + 512.
#define A4S_GMAX 8
#define A4S_ST 0x10000u
__device__ __forceinline__ void a4s_shape(u32 c, u32* n, u32* T2, u32* C, u32* T) {
  const u32 dT = c & 31, pos = c >> 5;
  *T = 1u << dT;
  *n = pos < *T ? pos + 1 : *T;
  u32 t2 = 1;
  while (t2 < *n) {
    t2 <<= 1;
  }
  *T2 = t2;
  *C = t2 <= 256 ? t2 : (t2 >> 8 > 256 ? t2 >> 8 : 256);
}

__device__ __forceinline__ float a4s_max(float a, float v) {
  return a < v ? v : a;
}

// scores of chunk blockIdx.x for the grp q heads of kv head blockIdx.y, then each head's
// chunk max (the subtree of llb_attn4's max tree)
extern "C" __global__ void __launch_bounds__(256) llb_a4s_score(
  const float* __restrict__ q, const float* __restrict__ kc,
  float* __restrict__ sc, float* __restrict__ pr, u32 c, u32 grp) {
  __shared__ float qs[A4S_GMAX][8 * 33];
  __shared__ float bs[A4S_GMAX * 512];
  u32 n, T2, C, T;
  a4s_shape(c, &n, &T2, &C, &T);
  const u32 kvh = blockIdx.y, h0 = kvh * grp, t0 = blockIdx.x * C;
  const u32 g = kvh * 256, ks = gridDim.y * 256;
  const float scale = __fdiv_rn(1.0f, __fsqrt_rn(256.0f));
  const float ninf = __uint_as_float(0xFF800000u);
  for (u32 i = threadIdx.x; i < grp * 256; i += blockDim.x) {
    const u32 gi = i >> 8, d = i & 255;
    qs[gi][(d >> 5) * 33 + (d & 31)] = q[(u64)(h0 + gi) * 256 + d];
  }
  __syncthreads();
  const u32 sub = threadIdx.x & 7;
  for (u32 tb = 0; tb < C; tb += blockDim.x >> 3) {
    const u32 tl = tb + (threadIdx.x >> 3), t = t0 + tl;
    float v[A4S_GMAX];
#pragma unroll
    for (u32 gi = 0; gi < A4S_GMAX; gi++) {
      v[gi] = 0.0f;
    }
    if (tl < C && t < n) {
      const float4* k4 = (const float4*)(kc + (u64)t * ks + g + sub * 32);
      float kk[32];
#pragma unroll
      for (u32 u = 0; u < 8; u++) {
        const float4 w = k4[u];
        kk[4 * u] = w.x, kk[4 * u + 1] = w.y, kk[4 * u + 2] = w.z, kk[4 * u + 3] = w.w;
      }
      for (u32 gi = 0; gi < grp; gi++) {
        const float* qq = qs[gi] + sub * 33;
        float st[6];
        u32   sp = 0;
#pragma unroll
        for (u32 i = 0; i < 32; i++) {
          attn_push(st, &sp, i, __fmul_rn(kk[i], qq[i]));
        }
        v[gi] = st[0];
      }
    }
    for (u32 gi = 0; gi < grp; gi++) {
      float x = v[gi];
#pragma unroll
      for (u32 m = 1; m < 8; m <<= 1) {
        x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, m));
      }
      if (sub == 0 && tl < C) {
        const float s = t < n ? __fmul_rn(x, scale) : ninf;
        sc[((u64)(h0 + gi) << 17) + t] = s;
        bs[gi * C + tl] = s;
      }
    }
  }
  __syncthreads();
  for (u32 s = 1; s < C; s <<= 1) {
    for (u32 i = threadIdx.x * 2 * s; i < grp * C; i += blockDim.x * 2 * s) {
      bs[i] = a4s_max(bs[i], bs[i + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x < grp) {
    pr[((u64)(h0 + threadIdx.x) << 17) + A4S_ST + blockIdx.x] = bs[threadIdx.x * C];
  }
}

// the head's max (the tree over the NC chunk maxes), the chunk's exps and its sum subtree
extern "C" __global__ void __launch_bounds__(256) llb_a4s_exp(
  float* __restrict__ sc, float* __restrict__ pr, u32 c, u32 grp) {
  __shared__ float bs[512];
  __shared__ float ms;
  u32 n, T2, C, T;
  a4s_shape(c, &n, &T2, &C, &T);
  const u32 NC = T2 / C, t0 = blockIdx.x * C;
  for (u32 gi = 0; gi < grp; gi++) {
    const u32 h = blockIdx.y * grp + gi;
    float* sh = sc + ((u64)h << 17);
    float* st = pr + ((u64)h << 17) + A4S_ST;
    for (u32 i = threadIdx.x; i < NC; i += blockDim.x) {
      bs[i] = st[i];
    }
    __syncthreads();
    for (u32 s = 1; s < NC; s <<= 1) {
      for (u32 i = threadIdx.x * 2 * s; i < NC; i += blockDim.x * 2 * s) {
        bs[i] = a4s_max(bs[i], bs[i + s]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      ms = bs[0];
    }
    __syncthreads();
    const float m = ms;
    for (u32 i = threadIdx.x; i < C; i += blockDim.x) {
      bs[i] = sh[t0 + i] = attn_exp(__fsub_rn(sh[t0 + i], m));
    }
    __syncthreads();
    for (u32 s = 1; s < C; s <<= 1) {
      for (u32 i = threadIdx.x * 2 * s; i < C; i += blockDim.x * 2 * s) {
        bs[i] = __fadd_rn(bs[i], bs[i + s]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      st[512 + blockIdx.x] = bs[0];
    }
    __syncthreads();
  }
}

// z (the tree over the NC chunk sums, + 0 when T2 < T as llb_attn4), the chunk's probs and
// its weighted-V subtree per q head and dim; NC == 1 writes att directly
extern "C" __global__ void __launch_bounds__(1024) llb_a4s_wv(
  const float* __restrict__ vc, float* __restrict__ att,
  float* __restrict__ sc, float* __restrict__ pr, u32 c, u32 grp) {
  __shared__ float ps[A4S_GMAX][512];
  __shared__ float bs[A4S_GMAX][256];
  u32 n, T2, C, T;
  a4s_shape(c, &n, &T2, &C, &T);
  const u32 NC = T2 / C, t0 = blockIdx.x * C, h0 = blockIdx.y * grp;
  const u32 g = blockIdx.y * 256, ks = gridDim.y * 256;
  for (u32 i = threadIdx.x; i < grp * NC; i += blockDim.x) {
    const u32 gi = i / NC, k = i % NC;
    bs[gi][k] = pr[((u64)(h0 + gi) << 17) + A4S_ST + 512 + k];
  }
  __syncthreads();
  for (u32 s = 1; s < NC; s <<= 1) {
    for (u32 i = threadIdx.x; i < grp * NC; i += blockDim.x) {
      const u32 gi = i / NC, k = i % NC;
      if ((k & (2 * s - 1)) == 0) {
        bs[gi][k] = __fadd_rn(bs[gi][k], bs[gi][k + s]);
      }
    }
    __syncthreads();
  }
  for (u32 i = threadIdx.x; i < grp * C; i += blockDim.x) {
    const u32 gi = i / C, k = i % C;
    const float z = T2 < T ? __fadd_rn(bs[gi][0], 0.0f) : bs[gi][0];
    ps[gi][k] = __fdiv_rn(sc[((u64)(h0 + gi) << 17) + t0 + k], z);
  }
  __syncthreads();
  const u32 j = threadIdx.x & 255;
  for (u32 gi = threadIdx.x >> 8; gi < grp; gi += blockDim.x >> 8) {
    float st[11];
    u32   sp = 0;
    const float* pp = ps[gi];
    u32 t = 0;
    for (; t + 8 <= C; t += 8) {
      float vv[8];
#pragma unroll
      for (u32 u = 0; u < 8; u++) {
        vv[u] = vc[(u64)(t0 + t + u) * ks + g + j];
      }
#pragma unroll
      for (u32 u = 0; u < 8; u++) {
        attn_push(st, &sp, t + u, __fmul_rn(vv[u], pp[t + u]));
      }
    }
    for (; t < C; t++) {
      attn_push(st, &sp, t, __fmul_rn(vc[(u64)(t0 + t) * ks + g + j], pp[t]));
    }
    if (NC == 1) {
      att[(u64)(h0 + gi) * 256 + j] = T2 < T ? __fadd_rn(st[0], 0.0f) : st[0];
    } else {
      pr[((u64)(h0 + gi) << 17) + (u64)blockIdx.x * 256 + j] = st[0];
    }
  }
}

// att: the tree over the NC chunk partials (NC > 1), + 0 when T2 < T; a block a q head
extern "C" __global__ void __launch_bounds__(256) llb_a4s_out(
  float* __restrict__ att, const float* __restrict__ pr, u32 c) {
  u32 n, T2, C, T;
  a4s_shape(c, &n, &T2, &C, &T);
  const u32 NC = T2 / C, h = blockIdx.x, j = threadIdx.x;
  const float* p = pr + ((u64)h << 17) + j;
  float st[9];
  u32   sp = 0;
  for (u32 k = 0; k < NC; k++) {
    attn_push(st, &sp, k, p[(u64)k * 256]);
  }
  att[(u64)h * 256 + j] = T2 < T ? __fadd_rn(st[0], 0.0f) : st[0];
}
