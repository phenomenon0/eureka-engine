# edit_speed.py COMP.TS : the cuda-moe-speed edits to the MoE kernels, applied to comp.ts with
# tools/payload-moeany.patch already applied (the output then diffs against the pre-moeany comp.ts
# into the new tools/payload-moeany.patch)
import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:100], s.count(old))
    s = s.replace(old, new)

# 1. the format pieces (load / decode split) and the multi-row dot, after moe_dot
rep('''  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, m));
  }
  return acc;
}

// one block of 1024''', '''  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, m));
  }
  return acc;
}

// reports/cuda-moe-speed: moe_dot's groups with the loads split from the decode (MQ<F>::ld reads the
// bytes of group g, MQ<F>::dec turns them into exactly q8g_w / q4k_w / q6k_w / q5k_w's four weights:
// the same f32 operations on the same bytes), so moe_dotn can issue every load of U groups of NR rows
// before the first decode. Each row's sum keeps moe_dot's order (lane l: groups l, l + 32, .. in
// turn, then the xor butterfly), so the bits are moe_dot's.
INLINE u32 moe_b(uint4 h, u32 k) {  // byte k of h.y h.z h.w (a K-quant's scales[12])
  u32 w = k < 4 ? h.y : k < 8 ? h.z : h.w;
  return (w >> (8 * (k & 3))) & 255u;
}
template <u32 F> struct MQ;
template <> struct MQ<1> {  // Q8_0, GGUF blocks ([f16 d][32 q], 34 bytes: 2-aligned)
  struct Raw { u32 s, q; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 g) {
    const unsigned short* h = (const unsigned short*)(row + (u64)(g >> 3) * 34);
    Raw r;
    r.s = h[0];
    r.q = (u32)h[1 + 2 * (g & 7)] | (u32)h[2 + 2 * (g & 7)] << 16;
    return r;
  }
  static __device__ __forceinline__ float4 dec(const Raw& r, u32 g) {
    float s = llb_f16(r.s);
    return make_float4(__fmul_rn((float)(signed char)r.q, s), __fmul_rn((float)(signed char)(r.q >> 8), s),
      __fmul_rn((float)(signed char)(r.q >> 16), s), __fmul_rn((float)(signed char)(r.q >> 24), s));
  }
};
template <> struct MQ<3> {  // Q4_K (144 bytes: f16 d, dmin, scales[12], qs[128]; 16-aligned)
  struct Raw { uint4 h; u32 q; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 g) {
    u32  c4 = 4 * g, rr = c4 & 255;
    const u8* B = row + (u64)(c4 >> 8) * 144;
    Raw r;
    r.h = *(const uint4*)B;
    r.q = *(const u32*)(B + 16 + (rr >> 6) * 32 + (rr & 31));
    return r;
  }
  static __device__ __forceinline__ float4 dec(const Raw& x, u32 g) {
    u32  rr = (4 * g) & 255, b = rr >> 6, hi = (rr & 63) >> 5;
    float d  = llb_f16(x.h.x & 65535u);
    float dm = llb_f16(x.h.x >> 16);
    u32  sb = 2 * b + hi, dsc, msc;
    if (sb < 4) {
      dsc = moe_b(x.h, sb) & 63; msc = moe_b(x.h, sb + 4) & 63;
    } else {
      dsc = (moe_b(x.h, sb + 4) & 15) | ((moe_b(x.h, sb - 4) >> 6) << 4);
      msc = (moe_b(x.h, sb + 4) >> 4) | ((moe_b(x.h, sb) >> 6) << 4);
    }
    float d1 = __fmul_rn(d, (float)dsc), m1 = __fmul_rn(dm, (float)msc);
    u32  v = x.q >> (hi * 4);
    return make_float4(__fsub_rn(__fmul_rn(d1, (float)(v & 15)), m1),
      __fsub_rn(__fmul_rn(d1, (float)((v >> 8) & 15)), m1),
      __fsub_rn(__fmul_rn(d1, (float)((v >> 16) & 15)), m1),
      __fsub_rn(__fmul_rn(d1, (float)((v >> 24) & 15)), m1));
  }
};
template <> struct MQ<9> {  // Q5_K (176 bytes: f16 d, dmin, scales[12], qh[32], qs[128]; 16-aligned)
  struct Raw { uint4 h; u32 q, qh; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 g) {
    u32  c4 = 4 * g, rr = c4 & 255;
    const u8* B = row + (u64)(c4 >> 8) * 176;
    Raw r;
    r.h = *(const uint4*)B;
    r.q = *(const u32*)(B + 48 + (rr >> 6) * 32 + (rr & 31));
    r.qh = *(const u32*)(B + 16 + (rr & 31));
    return r;
  }
  static __device__ __forceinline__ float4 dec(const Raw& x, u32 g) {
    u32  rr = (4 * g) & 255, b = rr >> 6, hi = (rr & 63) >> 5;
    float d  = llb_f16(x.h.x & 65535u);
    float dm = llb_f16(x.h.x >> 16);
    u32  sb = 2 * b + hi, dsc, msc;
    if (sb < 4) {
      dsc = moe_b(x.h, sb) & 63; msc = moe_b(x.h, sb + 4) & 63;
    } else {
      dsc = (moe_b(x.h, sb + 4) & 15) | ((moe_b(x.h, sb - 4) >> 6) << 4);
      msc = (moe_b(x.h, sb + 4) >> 4) | ((moe_b(x.h, sb) >> 6) << 4);
    }
    float d1 = __fmul_rn(d, (float)dsc), m1 = __fmul_rn(dm, (float)msc);
    float w[4];
    #pragma unroll
    for (u32 k = 0; k < 4; k++) {
      u32 qs = (x.q >> (8 * k)) & 255u, qh = (x.qh >> (8 * k)) & 255u;
      u32 q = (hi ? qs >> 4 : qs & 15) | ((qh >> sb) & 1) << 4;
      w[k] = __fsub_rn(__fmul_rn(d1, (float)q), m1);
    }
    return make_float4(w[0], w[1], w[2], w[3]);
  }
};
template <> struct MQ<4> {  // Q6_K (210 bytes: ql[128], qh[64], scales[16], f16 d; 2-aligned)
  struct Raw { u32 ql, qh, sd; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 g) {
    u32  c4 = 4 * g, r = c4 & 255;
    u32  n = r >> 7, rr = r & 127, quad = rr >> 5, l = rr & 31;
    const u8* B = row + (u64)(c4 >> 8) * 210;
    const unsigned short* ql = (const unsigned short*)(B + n * 64 + (quad & 1) * 32 + l);
    const unsigned short* qh = (const unsigned short*)(B + 128 + n * 32 + l);
    Raw x;
    x.ql = (u32)ql[0] | (u32)ql[1] << 16;
    x.qh = (u32)qh[0] | (u32)qh[1] << 16;
    x.sd = (u32)B[192 + n * 8 + quad * 2 + (l >> 4)] | (u32)*(const unsigned short*)(B + 208) << 16;
    return x;
  }
  static __device__ __forceinline__ float4 dec(const Raw& x, u32 g) {
    u32  quad = ((4 * g) & 127) >> 5;
    float d = llb_f16(x.sd >> 16);
    u32  nsh = (quad >= 2) ? 4u : 0u, qsh = 2 * quad;
    int  scv = (int)(signed char)(x.sd & 255u);
    float ds = __fmul_rn(d, (float)scv);
    float w[4];
    #pragma unroll
    for (u32 k = 0; k < 4; k++) {
      u32 a = (x.ql >> (8 * k)) & 255u, h = (x.qh >> (8 * k)) & 255u;
      w[k] = __fmul_rn(ds, (float)(int)(((a >> nsh) & 15) | (((h >> qsh) & 3) << 4)) - 32.0f);
    }
    return make_float4(w[0], w[1], w[2], w[3]);
  }
};

// NR rows' dots at once (row j against x[j]; SX: one x for all), U groups a lane per round: every
// load of the round first, then the decodes and the adds in moe_dot's order; cg / 32 groups a lane
// (cg = cols / 4, a multiple of 64 for whole superblocks)
template <u32 F, u32 NR, bool SX, u32 U>
__device__ __forceinline__ void moe_dotn(const u8* const* rw, const float* const* xs, u32 cg, float* out) {
  const u32 l = threadIdx.x & 31, ni = cg >> 5;
  float acc[NR];
  #pragma unroll
  for (u32 j = 0; j < NR; j++) {
    acc[j] = 0.0f;
  }
  u32 i = 0;
  #pragma unroll 1
  for (; i + U <= ni; i += U) {
    typename MQ<F>::Raw raw[NR][U];
    float4 xv[SX ? 1 : NR][U];
    #pragma unroll
    for (u32 u = 0; u < U; u++) {
      const u32 g = l + 32 * (i + u);
      #pragma unroll
      for (u32 j = 0; j < NR; j++) {
        raw[j][u] = MQ<F>::ld(rw[j], g);
      }
      #pragma unroll
      for (u32 j = 0; j < (SX ? 1 : NR); j++) {
        xv[j][u] = *(const float4*)(xs[j] + 4 * (u64)g);
      }
    }
    #pragma unroll
    for (u32 u = 0; u < U; u++) {
      const u32 g = l + 32 * (i + u);
      #pragma unroll
      for (u32 j = 0; j < NR; j++) {
        const float4 w = MQ<F>::dec(raw[j][u], g), xx = xv[SX ? 0 : j][u];
        acc[j] = __fadd_rn(acc[j], __fadd_rn(__fadd_rn(__fmul_rn(w.x, xx.x), __fmul_rn(w.y, xx.y)),
          __fadd_rn(__fmul_rn(w.z, xx.z), __fmul_rn(w.w, xx.w))));
      }
    }
  }
  #pragma unroll 1
  for (; i < ni; i++) {
    const u32 g = l + 32 * i;
    #pragma unroll
    for (u32 j = 0; j < NR; j++) {
      const float4 w = MQ<F>::dec(MQ<F>::ld(rw[j], g), g), xx = *(const float4*)(xs[SX ? 0 : j] + 4 * (u64)g);
      acc[j] = __fadd_rn(acc[j], __fadd_rn(__fadd_rn(__fmul_rn(w.x, xx.x), __fmul_rn(w.y, xx.y)),
        __fadd_rn(__fmul_rn(w.z, xx.z), __fmul_rn(w.w, xx.w))));
    }
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    #pragma unroll
    for (u32 j = 0; j < NR; j++) {
      acc[j] = __fadd_rn(acc[j], __shfl_xor_sync(0xffffffffu, acc[j], m));
    }
  }
  #pragma unroll
  for (u32 j = 0; j < NR; j++) {
    out[j] = acc[j];
  }
}

// one block of 1024''')

# 2. top-k: the k rounds as one warp-level top-k a warp, then one merge in warp 0 (no block barrier a
#    round); the same total order (probability descending, then index ascending) as the block argmax,
#    so the same ids and weights. Kept for k > 32 or more than 8 experts a thread (ne > 8192).
rep('''  // k rounds of a block argmax over the probabilities (first index on equal values); a chosen
  // expert's slot is set to -2 (p >= 0), the chosen p kept in wt[j] until the renormalization
  for (u32 j = 0; j < nk; j++) {''', '''  // reports/cuda-moe-speed: k <= 32 and at most 8 experts a thread: each warp takes the k best of its
  // own experts (shuffles only), warp 0 the k best of those 32 k; the order (p descending, then the
  // index ascending) is the block argmax's below, so the same experts in the same ranks
  __shared__ float cv[1024];
  __shared__ u32 ci[1024];
  if (nk <= 32 && ne <= 8192) {
    float pl[8];
    u32 own = 0;
    #pragma unroll
    for (u32 q = 0; q < 8; q++) {
      const u32 e = t + 1024 * q;
      pl[q] = e < ne ? pv[e] : -1.0f;
      own |= (e < ne ? 1u : 0u) << q;
    }
    for (u32 j = 0; j < nk; j++) {
      float bv = -1.0f;
      u32 bix = 0xffffffffu;
      #pragma unroll
      for (u32 q = 0; q < 8; q++) {
        if ((own >> q & 1) && pl[q] > bv) {
          bv = pl[q]; bix = t + 1024 * q;
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
      #pragma unroll
      for (u32 q = 0; q < 8; q++) {
        if (bix == t + 1024 * q) {
          own &= ~(1u << q);
        }
      }
      if (l == 0) {
        cv[w * nk + j] = bv; ci[w * nk + j] = bix;
      }
    }
    __syncthreads();
    if (w == 0) {
      const u32 nc = 32 * nk;
      for (u32 j = 0; j < nk; j++) {
        float bv = -1.0f;
        u32 bix = 0xffffffffu, bc = 0xffffffffu;
        for (u32 c = l; c < nc; c += 32) {
          const float v = cv[c];
          const u32 x = ci[c];
          if (x != 0xffffffffu && (v > bv || (v == bv && x < bix))) {
            bv = v; bix = x; bc = c;
          }
        }
        #pragma unroll
        for (u32 m = 16; m > 0; m >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
          const u32 oi = __shfl_xor_sync(0xffffffffu, bix, m);
          const u32 oc = __shfl_xor_sync(0xffffffffu, bc, m);
          if (ov > bv || (ov == bv && oi < bix)) {
            bv = ov; bix = oi; bc = oc;
          }
        }
        if (l == 0) {
          ids[j] = bix;
          wt[j] = bv;
          ci[bc] = 0xffffffffu;
        }
        __syncwarp();
      }
    }
  } else
  // k rounds of a block argmax over the probabilities (first index on equal values); a chosen
  // expert's slot is set to -2 (p >= 0), the chosen p kept in wt[j] until the renormalization
  for (u32 j = 0; j < nk; j++) {''')

# 3. gate + up: both rows of a warp's row in one moe_dotn (MQ bodies) where gate and up share a format
rep('''  const u64 e = ids[j];
  const float g = moe_dot(G + e * es + (u64)r * rbg, fg, cg, x);
  const float u = moe_dot(U + e * es + (u64)r * rbu, fu, cg, x);
  if ((threadIdx.x & 31) == 0) {''', '''  const u64 e = ids[j];
  float g, u;
  if (fg == fu) {
    // reports/cuda-moe-speed: gate and up rows together, loads first (moe_dotn: moe_dot's bits)
    const u8* rw[2] = { G + e * es + (u64)r * rbg, U + e * es + (u64)r * rbu };
    const float* xs[1] = { x };
    float v[2];
    if (fg == 1) moe_dotn<1, 2, true, 4>(rw, xs, cg, v);
    else if (fg == 3) moe_dotn<3, 2, true, 4>(rw, xs, cg, v);
    else if (fg == 4) moe_dotn<4, 2, true, 4>(rw, xs, cg, v);
    else moe_dotn<9, 2, true, 4>(rw, xs, cg, v);
    g = v[0]; u = v[1];
  } else {
    g = moe_dot(G + e * es + (u64)r * rbg, fg, cg, x);
    u = moe_dot(U + e * es + (u64)r * rbu, fu, cg, x);
  }
  if ((threadIdx.x & 31) == 0) {''')

# 4. down: two experts' rows at once (MQ bodies), summed in rank order as before
rep('''  float mo = 0.0f;
  for (u32 j = 0; j < nk; j++) {
    const float v = moe_dot(D + (u64)ids[j] * es + (u64)r * rbd, fd, cg, act + (u64)j * ef);
    mo = j == 0 ? __fmul_rn(v, wt[0]) : __fadd_rn(mo, __fmul_rn(v, wt[j]));
  }''', '''  float mo = 0.0f;
  // reports/cuda-moe-speed: experts j and j + 1 in one moe_dotn (loads first; moe_dot's bits), the
  // weighted sum still in rank order
  u32 j = 0;
  for (; j + 2 <= nk; j += 2) {
    const u8* rw[2] = { D + (u64)ids[j] * es + (u64)r * rbd, D + (u64)ids[j + 1] * es + (u64)r * rbd };
    const float* xs[2] = { act + (u64)j * ef, act + (u64)(j + 1) * ef };
    float v[2];
    if (fd == 1) moe_dotn<1, 2, false, 4>(rw, xs, cg, v);
    else if (fd == 3) moe_dotn<3, 2, false, 4>(rw, xs, cg, v);
    else if (fd == 4) moe_dotn<4, 2, false, 4>(rw, xs, cg, v);
    else moe_dotn<9, 2, false, 4>(rw, xs, cg, v);
    mo = j == 0 ? __fmul_rn(v[0], wt[0]) : __fadd_rn(mo, __fmul_rn(v[0], wt[j]));
    mo = __fadd_rn(mo, __fmul_rn(v[1], wt[j + 1]));
  }
  for (; j < nk; j++) {
    const float v = moe_dot(D + (u64)ids[j] * es + (u64)r * rbd, fd, cg, act + (u64)j * ef);
    mo = j == 0 ? __fmul_rn(v, wt[0]) : __fadd_rn(mo, __fmul_rn(v, wt[j]));
  }''')
open(p, "w").write(s)
print("ok")

# ---- v2: unit (256-column) bodies replace the per-group MQ pieces in moe_dotn; single-warp top-k ----
s = open(p).read()
a = s.index("// reports/cuda-moe-speed: moe_dot's groups with the loads split")
b = s.index("// one block of 1024")
s = s[:a] + r'''// reports/cuda-moe-speed: moe_dot's sum, unit by unit. A 256-column unit is lane l's two groups
// g0 = l + 64 s and g1 = l + 32 + 64 s (moe_dot's groups in its order: lane l, then l + 32 in the
// unit), so MS<F>::ld reads both groups' bytes of unit s and MS<F>::dec makes exactly q8g_w /
// q4k_w / q6k_w / q5k_w's weights for them: the same values (each a small integer, d * scale, then
// ggml's f32 multiply / subtract in ggml's order; an integer's float through the 2^23 magic, exact),
// with the lane's scale bytes and f16s decoded once a unit, not once a group.
INLINE float moe_i(u32 v, u32 k) {  // byte k of v as a float, exact (0x4B0000xx - 2^23)
  return __fsub_rn(__uint_as_float(__byte_perm(v, 0x4B000000u, 0x7540 + k)), 8388608.0f);
}
template <u32 F> struct MS;
template <> struct MS<1> {  // Q8_0, GGUF blocks ([f16 d][32 q], 34 bytes, 2-aligned); unit = 8 blocks
  struct Raw { u32 s0, s1, q0, q1; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 s, u32 l) {
    const unsigned short* a = (const unsigned short*)(row + (u64)(8 * s + (l >> 3)) * 34);
    const unsigned short* b = (const unsigned short*)(row + (u64)(8 * s + 4 + (l >> 3)) * 34);
    const u32 o = 1 + 2 * (l & 7);
    Raw r;
    r.s0 = a[0]; r.q0 = (u32)a[o] | (u32)a[o + 1] << 16;
    r.s1 = b[0]; r.q1 = (u32)b[o] | (u32)b[o + 1] << 16;
    return r;
  }
  static __device__ __forceinline__ float4 one(u32 sh, u32 q) {
    const float d = llb_f16(sh);
    const u32 v = q ^ 0x80808080u;  // signed byte + 128
    float w[4];
    #pragma unroll
    for (u32 k = 0; k < 4; k++) {
      w[k] = __fmul_rn(__fsub_rn(__uint_as_float(__byte_perm(v, 0x4B000000u, 0x7540 + k)), 8388736.0f), d);
    }
    return make_float4(w[0], w[1], w[2], w[3]);
  }
  static __device__ __forceinline__ void dec(const Raw& r, u32 l, float4& w0, float4& w1) {
    w0 = one(r.s0, r.q0); w1 = one(r.s1, r.q1);
  }
};
// Q4_K / Q5_K's lane scales: sub-block l / 8 (unit half 0) and 4 + l / 8 (half 1), ggml's get_scale_min_k4
INLINE void moe_k4(uint4 h, u32 l, float* d1, float* m1) {
  const float d = llb_f16(h.x & 65535u), dm = llb_f16(h.x >> 16);
  const u32 sh = 8 * (l >> 3), A = (h.y >> sh) & 255u, Bz = (h.z >> sh) & 255u, C = (h.w >> sh) & 255u;
  const u32 sc[2] = { A & 63, (C & 15) | (A >> 6) << 4 }, mn[2] = { Bz & 63, (C >> 4) | (Bz >> 6) << 4 };
  #pragma unroll
  for (u32 t = 0; t < 2; t++) {
    d1[t] = __fmul_rn(d, moe_i(sc[t], 0)); m1[t] = __fmul_rn(dm, moe_i(mn[t], 0));
  }
}
template <> struct MS<3> {  // Q4_K (144 bytes: f16 d, dmin, scales[12], qs[128]; 16-aligned)
  struct Raw { uint4 h; u32 q0, q1; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 s, u32 l) {
    const u8* B = row + (u64)s * 144;
    const u32 o = 16 + 32 * (l >> 4) + 4 * (l & 7);
    Raw r;
    r.h = *(const uint4*)B; r.q0 = *(const u32*)(B + o); r.q1 = *(const u32*)(B + o + 64);
    return r;
  }
  static __device__ __forceinline__ void dec(const Raw& r, u32 l, float4& w0, float4& w1) {
    float d1[2], m1[2];
    moe_k4(r.h, l, d1, m1);
    const u32 hs = 4 * ((l >> 3) & 1), v[2] = { (r.q0 >> hs) & 0x0F0F0F0Fu, (r.q1 >> hs) & 0x0F0F0F0Fu };
    float w[2][4];
    #pragma unroll
    for (u32 t = 0; t < 2; t++) {
      #pragma unroll
      for (u32 k = 0; k < 4; k++) {
        w[t][k] = __fsub_rn(__fmul_rn(d1[t], moe_i(v[t], k)), m1[t]);
      }
    }
    w0 = make_float4(w[0][0], w[0][1], w[0][2], w[0][3]); w1 = make_float4(w[1][0], w[1][1], w[1][2], w[1][3]);
  }
};
template <> struct MS<9> {  // Q5_K (176 bytes: f16 d, dmin, scales[12], qh[32], qs[128]; 16-aligned)
  struct Raw { uint4 h; u32 q0, q1, qh; };
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 s, u32 l) {
    const u8* B = row + (u64)s * 176;
    const u32 o = 48 + 32 * (l >> 4) + 4 * (l & 7);
    Raw r;
    r.h = *(const uint4*)B; r.q0 = *(const u32*)(B + o); r.q1 = *(const u32*)(B + o + 64);
    r.qh = *(const u32*)(B + 16 + 4 * (l & 7));
    return r;
  }
  static __device__ __forceinline__ void dec(const Raw& r, u32 l, float4& w0, float4& w1) {
    float d1[2], m1[2];
    moe_k4(r.h, l, d1, m1);
    const u32 hs = 4 * ((l >> 3) & 1), sb0 = l >> 3;
    const u32 v[2] = { ((r.q0 >> hs) & 0x0F0F0F0Fu) | ((r.qh >> sb0) & 0x01010101u) << 4,
      ((r.q1 >> hs) & 0x0F0F0F0Fu) | ((r.qh >> (sb0 + 4)) & 0x01010101u) << 4 };
    float w[2][4];
    #pragma unroll
    for (u32 t = 0; t < 2; t++) {
      #pragma unroll
      for (u32 k = 0; k < 4; k++) {
        w[t][k] = __fsub_rn(__fmul_rn(d1[t], moe_i(v[t], k)), m1[t]);
      }
    }
    w0 = make_float4(w[0][0], w[0][1], w[0][2], w[0][3]); w1 = make_float4(w[1][0], w[1][1], w[1][2], w[1][3]);
  }
};
template <> struct MS<4> {  // Q6_K (210 bytes: ql[128], qh[64], scales[16], f16 d; 2-aligned)
  struct Raw { u32 l0, l1, h0, h1, sc; };
  static __device__ __forceinline__ u32 u2(const u8* p) {
    const unsigned short* h = (const unsigned short*)p;
    return (u32)h[0] | (u32)h[1] << 16;
  }
  static __device__ __forceinline__ Raw ld(const u8* __restrict__ row, u32 s, u32 l) {
    const u8* B = row + (u64)s * 210;
    const u32 q = l >> 3, o = (q & 1) * 32 + 4 * (l & 7), so = 192 + 2 * q + ((l & 7) >> 2);
    Raw r;
    r.l0 = u2(B + o); r.l1 = u2(B + 64 + o);
    r.h0 = u2(B + 128 + 4 * (l & 7)); r.h1 = u2(B + 160 + 4 * (l & 7));
    r.sc = (u32)B[so] | (u32)B[so + 8] << 8 | (u32)*(const unsigned short*)(B + 208) << 16;
    return r;
  }
  static __device__ __forceinline__ void dec(const Raw& r, u32 l, float4& w0, float4& w1) {
    const u32 q = l >> 3, nsh = q >= 2 ? 4u : 0u, qsh = 2 * q;
    const float d = llb_f16(r.sc >> 16);
    const float ds[2] = { __fmul_rn(d, (float)(int)(signed char)(r.sc & 255u)),
      __fmul_rn(d, (float)(int)(signed char)((r.sc >> 8) & 255u)) };
    const u32 v[2] = { ((r.l0 >> nsh) & 0x0F0F0F0Fu) | ((r.h0 >> qsh) & 0x03030303u) << 4,
      ((r.l1 >> nsh) & 0x0F0F0F0Fu) | ((r.h1 >> qsh) & 0x03030303u) << 4 };
    float w[2][4];
    #pragma unroll
    for (u32 t = 0; t < 2; t++) {
      #pragma unroll
      for (u32 k = 0; k < 4; k++) {  // (d * sc) * (q - 32), q - 32 exact (2^23 + 32)
        w[t][k] = __fmul_rn(ds[t], __fsub_rn(__uint_as_float(__byte_perm(v[t], 0x4B000000u, 0x7540 + k)), 8388640.0f));
      }
    }
    w0 = make_float4(w[0][0], w[0][1], w[0][2], w[0][3]); w1 = make_float4(w[1][0], w[1][1], w[1][2], w[1][3]);
  }
};

// NR rows' dots at once (row j against x[j]; SX: one x for all), US units a round: every load of the
// round first, then the decodes and the adds in moe_dot's order (unit by unit, group l before l + 32)
INLINE float moe_g(float acc, float4 w, float4 x) {
  return __fadd_rn(acc, __fadd_rn(__fadd_rn(__fmul_rn(w.x, x.x), __fmul_rn(w.y, x.y)),
    __fadd_rn(__fmul_rn(w.z, x.z), __fmul_rn(w.w, x.w))));
}
template <u32 F, u32 NR, bool SX, u32 US>
__device__ __forceinline__ void moe_dotn(const u8* const* rw, const float* const* xs, u32 cg, float* out) {
  const u32 l = threadIdx.x & 31, nu = cg >> 6;
  float acc[NR];
  #pragma unroll
  for (u32 j = 0; j < NR; j++) {
    acc[j] = 0.0f;
  }
  u32 s = 0;
  #pragma unroll 1
  for (; s + US <= nu; s += US) {
    typename MS<F>::Raw raw[NR][US];
    float4 xv[SX ? 1 : NR][US][2];
    #pragma unroll
    for (u32 u = 0; u < US; u++) {
      #pragma unroll
      for (u32 j = 0; j < NR; j++) {
        raw[j][u] = MS<F>::ld(rw[j], s + u, l);
      }
      #pragma unroll
      for (u32 j = 0; j < (SX ? 1 : NR); j++) {
        const float* xp = xs[j] + 256 * (u64)(s + u) + 4 * l;
        xv[j][u][0] = *(const float4*)xp; xv[j][u][1] = *(const float4*)(xp + 128);
      }
    }
    #pragma unroll
    for (u32 u = 0; u < US; u++) {
      #pragma unroll
      for (u32 j = 0; j < NR; j++) {
        float4 w0, w1;
        MS<F>::dec(raw[j][u], l, w0, w1);
        acc[j] = moe_g(moe_g(acc[j], w0, xv[SX ? 0 : j][u][0]), w1, xv[SX ? 0 : j][u][1]);
      }
    }
  }
  #pragma unroll 1
  for (; s < nu; s++) {
    #pragma unroll
    for (u32 j = 0; j < NR; j++) {
      float4 w0, w1;
      MS<F>::dec(MS<F>::ld(rw[j], s, l), l, w0, w1);
      const float* xp = xs[SX ? 0 : j] + 256 * (u64)s + 4 * l;
      acc[j] = moe_g(moe_g(acc[j], w0, *(const float4*)xp), w1, *(const float4*)(xp + 128));
    }
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    #pragma unroll
    for (u32 j = 0; j < NR; j++) {
      acc[j] = __fadd_rn(acc[j], __shfl_xor_sync(0xffffffffu, acc[j], m));
    }
  }
  #pragma unroll
  for (u32 j = 0; j < NR; j++) {
    out[j] = acc[j];
  }
}

''' + s[b:]

# top-k: one warp, a lane's best kept and rescanned only when it is taken (ne <= 1024)
a = s.index("  // reports/cuda-moe-speed: k <= 32 and at most 8 experts a thread")
b = s.index("  } else\n  // k rounds of a block argmax")
s = s[:a] + r'''  // reports/cuda-moe-speed: up to 1024 experts, warp 0 picks all k: lane l holds experts l, l + 32, ..
  // and its best untaken one; a round is a warp argmax of the lanes' bests, and only the lane that
  // lost its best rescans. The order (p descending, then the index ascending) is the block argmax's
  // below, so the same experts in the same ranks
  if (ne <= 1024) {
    if (w == 0) {
      float pl[32];
      u32 own = 0;
      #pragma unroll
      for (u32 q = 0; q < 32; q++) {
        const u32 e = l + 32 * q;
        pl[q] = e < ne ? pv[e] : -1.0f;
        own |= (e < ne ? 1u : 0u) << q;
      }
      float mv = -1.0f;
      u32 mi = 0xffffffffu;
      #pragma unroll
      for (u32 q = 0; q < 32; q++) {
        if ((own >> q & 1) && pl[q] > mv) {
          mv = pl[q]; mi = l + 32 * q;
        }
      }
      for (u32 j = 0; j < nk; j++) {
        float bv = mv;
        u32 bix = mi;
        #pragma unroll
        for (u32 m = 16; m > 0; m >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
          const u32 oi = __shfl_xor_sync(0xffffffffu, bix, m);
          if (ov > bv || (ov == bv && oi < bix)) {
            bv = ov; bix = oi;
          }
        }
        if (l == 0) {
          ids[j] = bix;
          wt[j] = bv;
        }
        if (bix == mi) {
          own &= ~(1u << (mi >> 5));
          mv = -1.0f; mi = 0xffffffffu;
          #pragma unroll
          for (u32 q = 0; q < 32; q++) {
            if ((own >> q & 1) && pl[q] > mv) {
              mv = pl[q]; mi = l + 32 * q;
            }
          }
        }
      }
    }
''' + s[b:]
s = s.replace('''  __shared__ float cv[1024];
  __shared__ u32 ci[1024];
''', '')
# gate + up and down call the unit bodies (US 2 / 2)
s = s.replace("moe_dotn<1, 2, true, 4>", "moe_dotn<1, 2, true, 2>").replace("moe_dotn<3, 2, true, 4>", "moe_dotn<3, 2, true, 2>")
s = s.replace("moe_dotn<4, 2, true, 4>", "moe_dotn<4, 2, true, 2>").replace("moe_dotn<9, 2, true, 4>", "moe_dotn<9, 2, true, 2>")
s = s.replace("moe_dotn<1, 2, false, 4>", "moe_dotn<1, 2, false, 2>").replace("moe_dotn<3, 2, false, 4>", "moe_dotn<3, 2, false, 2>")
s = s.replace("moe_dotn<4, 2, false, 4>", "moe_dotn<4, 2, false, 2>").replace("moe_dotn<9, 2, false, 4>", "moe_dotn<9, 2, false, 2>")
open(p, "w").write(s)
print("ok v2")

# ---- v3: top-k + the shared expert's silu in one launch; down takes 4 experts a round and the shared
# expert's down (where llb_kgemv would run llb_gemv_any: the same leaf and order) ----
s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:100], s.count(old))
    s = s.replace(old, new)
rep('''extern "C" __global__ void __launch_bounds__(1024) llb_moe_topk(const float* __restrict__ lg, u32 ne, u32 nk,
  const float* __restrict__ sg, float* __restrict__ pv, u32* __restrict__ ids, float* __restrict__ wt,
  float* __restrict__ sgo) {
  __shared__ float rv[32];
  __shared__ u32 ri[32];
  __shared__ float bc;
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5;
''', '''// reports/cuda-moe-speed: blocks 1.. are the shared expert's silu (llb_silu's formula): sa[i] =
// silu(sgg[i]) * sgu[i], i < nsf, 1024 a block
extern "C" __global__ void __launch_bounds__(1024) llb_moe_topk(const float* __restrict__ lg, u32 ne, u32 nk,
  const float* __restrict__ sg, float* __restrict__ pv, u32* __restrict__ ids, float* __restrict__ wt,
  float* __restrict__ sgo, const float* __restrict__ sgg, const float* __restrict__ sgu, float* __restrict__ sa,
  u32 nsf) {
  __shared__ float rv[32];
  __shared__ u32 ri[32];
  __shared__ float bc;
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5;
  if (blockIdx.x > 0) {
    const u32 i = (blockIdx.x - 1) * 1024 + t;
    if (i < nsf) {
      const float v = sgg[i];
      sa[i] = __fmul_rn(__fdiv_rn(v, __fadd_rn(1.0f, attn_exp(-v))), sgu[i]);
    }
    return;
  }
''')
a = s.index('''extern "C" __global__ void __launch_bounds__(128) llb_moe_dn(''')
b = s.index("// flat.bend's k.silu: a[i] = K.silu1(g[i]) * u[i]")
s = s[:a] + r'''template <u32 NR> __device__ __forceinline__ void moe_dnv(u32 f, const u8* const* rw, const float* const* xs,
  u32 cg, float* v) {
  if (f == 1) moe_dotn<1, NR, false, 1>(rw, xs, cg, v);
  else if (f == 3) moe_dotn<3, NR, false, 1>(rw, xs, cg, v);
  else if (f == 4) moe_dotn<4, NR, false, 1>(rw, xs, cg, v);
  else moe_dotn<9, NR, false, 1>(rw, xs, cg, v);
}

// down for the k experts, weighted and summed in rank order, then + the gated shared expert, then
// + the residual: y[r] = (sum_j wt[j] (down_ids[j] row r . act_j) + shd[r] sgo[0]) + x1[r]
// (llama.cpp's order: moe_out + ffn_shexp, then + ffn_inp). A warp a row of dm.
// reports/cuda-moe-speed: four experts' rows a round (moe_dotn, moe_dot's bits), the weighted sum
// in rank order; SD != 0: the shared expert's down row here (format fs, rbs bytes, cgs groups, x sa:
// llb_gemv_any's leaf and order, which llb_kgemv runs for it where no tuned body applies), else shd
extern "C" __global__ void __launch_bounds__(128) llb_moe_dn(const u8* __restrict__ D, u64 es, u32 rbd, u32 fd,
  u32 cg, const float* __restrict__ act, const u32* __restrict__ ids, const float* __restrict__ wt, u32 nk, u32 ef,
  const float* __restrict__ x1, const float* __restrict__ shd, const float* __restrict__ sgo,
  float* __restrict__ y, u32 dm, const u8* __restrict__ SD, u32 rbs, u32 fs, u32 cgs, const float* __restrict__ sa) {
  const u32 r = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (r >= dm) {
    return;
  }
  float hv;
  if (SD != nullptr) {
    const u8* rw[1] = { SD + (u64)r * rbs };
    const float* xs[1] = { sa };
    moe_dnv<1>(fs, rw, xs, cgs, &hv);
  } else {
    hv = shd[r];
  }
  float mo = 0.0f;
  u32 j = 0;
  for (; j + 4 <= nk; j += 4) {
    const u8* rw[4];
    const float* xs[4];
    #pragma unroll
    for (u32 q = 0; q < 4; q++) {
      rw[q] = D + (u64)ids[j + q] * es + (u64)r * rbd; xs[q] = act + (u64)(j + q) * ef;
    }
    float v[4];
    moe_dnv<4>(fd, rw, xs, cg, v);
    #pragma unroll
    for (u32 q = 0; q < 4; q++) {
      mo = j + q == 0 ? __fmul_rn(v[0], wt[0]) : __fadd_rn(mo, __fmul_rn(v[q], wt[j + q]));
    }
  }
  if (j + 2 <= nk) {
    const u8* rw[2] = { D + (u64)ids[j] * es + (u64)r * rbd, D + (u64)ids[j + 1] * es + (u64)r * rbd };
    const float* xs[2] = { act + (u64)j * ef, act + (u64)(j + 1) * ef };
    float v[2];
    moe_dnv<2>(fd, rw, xs, cg, v);
    mo = j == 0 ? __fmul_rn(v[0], wt[0]) : __fadd_rn(mo, __fmul_rn(v[0], wt[j]));
    mo = __fadd_rn(mo, __fmul_rn(v[1], wt[j + 1]));
    j += 2;
  }
  if (j < nk) {
    const u8* rw[1] = { D + (u64)ids[j] * es + (u64)r * rbd };
    const float* xs[1] = { act + (u64)j * ef };
    float v;
    moe_dnv<1>(fd, rw, xs, cg, &v);
    mo = j == 0 ? __fmul_rn(v, wt[0]) : __fadd_rn(mo, __fmul_rn(v, wt[j]));
  }
  if ((threadIdx.x & 31) == 0) {
    y[r] = __fadd_rn(__fadd_rn(mo, __fmul_rn(hv, sgo[0])), x1[r]);
  }
}

''' + s[b:]
open(p, "w").write(s)
print("ok v3")

# ---- v4: top-k rounds on __reduce_max_sync / __reduce_min_sync (the same order: p's bits are
# monotone for p >= 0); gate + up a warp takes MOE_GU_RPW rows (grid-stride, any grid) ----
s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:100], s.count(old))
    s = s.replace(old, new)
rep('''      for (u32 j = 0; j < nk; j++) {
        float bv = mv;
        u32 bix = mi;
        #pragma unroll
        for (u32 m = 16; m > 0; m >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
          const u32 oi = __shfl_xor_sync(0xffffffffu, bix, m);
          if (ov > bv || (ov == bv && oi < bix)) {
            bv = ov; bix = oi;
          }
        }
        if (l == 0) {''', '''      for (u32 j = 0; j < nk; j++) {
        // the largest p (its bits + 1: p in [0, 1], 0 = none), then the lowest index holding it
        const u32 kb = __reduce_max_sync(0xffffffffu, mi == 0xffffffffu ? 0u : __float_as_uint(mv) + 1);
        const u32 bix = __reduce_min_sync(0xffffffffu, mi != 0xffffffffu && __float_as_uint(mv) + 1 == kb ? mi : 0xffffffffu);
        const float bv = __uint_as_float(kb - 1);
        if (l == 0) {''')
a = s.index('''extern "C" __global__ void __launch_bounds__(128) llb_moe_gu(''')
b = s.index('''template <u32 NR> __device__ __forceinline__ void moe_dnv(''')
s = s[:a] + r'''// reports/cuda-moe-speed: a warp takes MOE_GU_RPW rows at once (gate and up of each in one moe_dotn,
// loads first: moe_dot's bits), grid-stride over the rows, so any grid covers them
#define MOE_GU_RPW 1
#define MOE_GU_US 2
template <u32 F> __device__ __forceinline__ void moe_gu_rows(const u8* __restrict__ G, const u8* __restrict__ U,
  u64 eo, u32 rbg, u32 rbu, u32 cg, const float* __restrict__ x, float* __restrict__ act, u32 ef) {
  const u32 nw = gridDim.x * 4;
  for (u32 r0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * MOE_GU_RPW; r0 < ef; r0 += nw * MOE_GU_RPW) {
    const u8* rw[2 * MOE_GU_RPW];
    #pragma unroll
    for (u32 q = 0; q < MOE_GU_RPW; q++) {
      const u32 r = r0 + q < ef ? r0 + q : ef - 1;
      rw[2 * q] = G + eo + (u64)r * rbg; rw[2 * q + 1] = U + eo + (u64)r * rbu;
    }
    const float* xs[1] = { x };
    float v[2 * MOE_GU_RPW];
    moe_dotn<F, 2 * MOE_GU_RPW, true, MOE_GU_US>(rw, xs, cg, v);
    #pragma unroll
    for (u32 q = 0; q < MOE_GU_RPW; q++) {
      if ((threadIdx.x & 31) == q && r0 + q < ef) {
        act[r0 + q] = __fmul_rn(__fdiv_rn(v[2 * q], __fadd_rn(1.0f, attn_exp(-v[2 * q]))), v[2 * q + 1]);
      }
    }
  }
}

// gate + up + silu for the k chosen experts: blocks (bx, j) take rows of expert ids[j], act[j ef + r] =
// silu(gate . x) * (up . x) (llb_silu's formula). G / U the stacks' bases, es the expert stride
// (bytes), rbg / rbu the row bytes (fused: U = G + ef rows, the same es)
extern "C" __global__ void __launch_bounds__(128) llb_moe_gu(const u8* __restrict__ G, const u8* __restrict__ U,
  u64 es, u32 rbg, u32 rbu, u32 fg, u32 fu, u32 cg, const float* __restrict__ x, const u32* __restrict__ ids,
  float* __restrict__ act, u32 ef) {
  const u32 j = blockIdx.y;
  const u64 e = ids[j];
  float* aj = act + (u64)j * ef;
  if (fg == fu) {
    if (fg == 1) moe_gu_rows<1>(G, U, e * es, rbg, rbu, cg, x, aj, ef);
    else if (fg == 3) moe_gu_rows<3>(G, U, e * es, rbg, rbu, cg, x, aj, ef);
    else if (fg == 4) moe_gu_rows<4>(G, U, e * es, rbg, rbu, cg, x, aj, ef);
    else moe_gu_rows<9>(G, U, e * es, rbg, rbu, cg, x, aj, ef);
    return;
  }
  const u32 nw = gridDim.x * 4;
  for (u32 r = blockIdx.x * 4 + (threadIdx.x >> 5); r < ef; r += nw) {
    const float g = moe_dot(G + e * es + (u64)r * rbg, fg, cg, x);
    const float u = moe_dot(U + e * es + (u64)r * rbu, fu, cg, x);
    if ((threadIdx.x & 31) == 0) {
      aj[r] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, attn_exp(-g))), u);
    }
  }
}

''' + s[b:]
open(p, "w").write(s)
print("ok v4")

# ---- v5: the down's experts a round and units a round as defines ----
s = open(p).read()
a = s.index("template <u32 NR> __device__ __forceinline__ void moe_dnv(")
b = s.index("// flat.bend's k.silu: a[i] = K.silu1(g[i]) * u[i]")
d = s[a:b]
d = d.replace("template <u32 NR> __device__ __forceinline__ void moe_dnv(", "#define MOE_DN_NR 4\n#define MOE_DN_US 1\ntemplate <u32 NR> __device__ __forceinline__ void moe_dnv(", 1)
for f in "1349":
    d = d.replace(f"moe_dotn<{f}, NR, false, 1>", f"moe_dotn<{f}, NR, false, MOE_DN_US>")
d = d.replace('''  for (; j + 4 <= nk; j += 4) {
    const u8* rw[4];
    const float* xs[4];
    #pragma unroll
    for (u32 q = 0; q < 4; q++) {''', '''  for (; j + MOE_DN_NR <= nk; j += MOE_DN_NR) {
    const u8* rw[MOE_DN_NR];
    const float* xs[MOE_DN_NR];
    #pragma unroll
    for (u32 q = 0; q < MOE_DN_NR; q++) {''')
d = d.replace('''    float v[4];
    moe_dnv<4>(fd, rw, xs, cg, v);
    #pragma unroll
    for (u32 q = 0; q < 4; q++) {''', '''    float v[MOE_DN_NR];
    moe_dnv<MOE_DN_NR>(fd, rw, xs, cg, v);
    #pragma unroll
    for (u32 q = 0; q < MOE_DN_NR; q++) {''')
assert "MOE_DN_NR <= nk" in d  # the remainder (< MOE_DN_NR = 4): a pair, then a single
s = s[:a] + d + s[b:]
open(p, "w").write(s)
print("ok v5")

# ---- v6: the down kernel held to 64 registers (8 blocks an SM: 17.7 us vs 26 at 80 registers, S35 Q4_K_M) ----
s = open(p).read()
old = 'extern "C" __global__ void __launch_bounds__(128) llb_moe_dn('
assert s.count(old) == 1
s = s.replace(old, 'extern "C" __global__ void __launch_bounds__(128, 8) llb_moe_dn(')
open(p, "w").write(s)
print("ok v6")
