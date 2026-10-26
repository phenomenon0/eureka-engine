// pfmm.cuh (PF27 stage 1): the fast prefill tier's T-row GEMMs on int8 tensor cores,
// y[t][r] = (W x_t)[r] for T tokens at once. Not exact (LLB_PREFILL=fast only).
// Needs the engine module's u8 / u32 / u64, INLINE and llb_f16 (nvrtc-clean, no includes).
//
// x: pf_q8x's per-token q8_1 numerics (llama.cpp quantize_q8_1): per 32-block d = amax / 127,
// q = round(x / d), s = sum(x) unrounded; qx [T][c] int8, xd [T][c / 32] float2 {d, s}.
// W: the pack's rows -- Q4_K (144 B / 256), the repacked Q8_0 (c int8 then c / 32 f16),
// Q6_K (210 B / 256). One mma per 32-block (Q4_K, Q8_0: m16n8k32) or 16-group (Q6_K:
// m16n8k16), s32 out, then scaled into f32: Q4_K acc += (d sc) e.d (W.q) - (dmin m) e.s,
// Q8_0 acc += (w.d e.d) (W.q), Q6_K acc += (d sc) e.d (W.q).
// k order: the mma's k is a free relabelling while A and B agree, so lane tig (lane & 3) takes
// elements 8 tig .. 8 tig + 7 of a 32-block (a0 / b0 the first 4, a2 / b1 the next 4): a
// Q4_K nibble run and q8 x load as one u64 each, no repack (Q6_K: 4 tig .. 4 tig + 3 of 16).
// Tile: a block of 4 warps x MT m-tiles (16 rows) x NT n-tiles (8 tokens); x's 256-column
// slab and its scales staged in shared memory, W straight from global into fragments.
// Epilogue EP: 0 y = acc, 1 y = x1 + acc (residual), 2 gate / up pair: m-tile 0 Wg, 1 Wu,
// the same 16 rows; y = silu(g) u. y and x1 are [T][n].
#define PF_XS 288  // x slab row stride: 256 + 32 keeps a half-warp's u64 reads conflict-free

INLINE void pf_mma32(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    "{%10,%10,%10,%10};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
#define PF_MAGIC 0x4B400000  // C seed: int_as_float(c) - 12582912 = the exact sum, |sum| < 2^22
INLINE float pf_f(int c) { return __int_as_float(c) - 12582912.0f; }
INLINE void pf_mma32m(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    "{%10,%10,%10,%10};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(PF_MAGIC));
}
INLINE void pf_mma16(int* c, u32 a0, u32 a1, u32 b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
    "{%7,%7,%7,%7};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
    : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// one warp a 32-block: T * c elements (c % 256 == 0), grid T * c / 256
extern "C" __global__ void __launch_bounds__(256) pf_q8x(const float* __restrict__ x,
  signed char* __restrict__ q, float2* __restrict__ xd) {
  const u64 i = (u64)blockIdx.x * 256 + threadIdx.x;
  const float v = x[i];
  float a = fabsf(v), s = v;
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    s += __shfl_xor_sync(0xffffffffu, s, m);
  }
  const float d = a / 127.0f;
  q[i] = a == 0.0f ? 0 : (signed char)roundf(v / d);
  if ((threadIdx.x & 31) == 0) {
    xd[i >> 5] = make_float2(d, s);
  }
}

// x's 256-column slab kb of tokens t0 .. t0 + 8 NT - 1 (zero past T) into xs / xe
template <int NT>
INLINE void pf_slab(signed char (*xs)[PF_XS], float2 (*xe)[8], const signed char* __restrict__ qx,
  const float2* __restrict__ xd, u32 c, u32 T, u32 t0, u32 kb) {
  for (u32 i = threadIdx.x; i < NT * 8 * 16; i += 128) {
    const u32 tk = i >> 4, o = (i & 15) * 16;
    int4 v = make_int4(0, 0, 0, 0);
    if (t0 + tk < T) v = *(const int4*)(qx + (u64)(t0 + tk) * c + kb * 256 + o);
    *(int4*)&xs[tk][o] = v;
  }
  for (u32 i = threadIdx.x; i < NT * 8 * 8; i += 128) {
    const u32 tk = i >> 3;
    xe[tk][i & 7] = t0 + tk < T ? xd[(u64)(t0 + tk) * (c / 32) + kb * 8 + (i & 7)] : make_float2(0.0f, 0.0f);
  }
}

template <int MT, int NT, int EP>
INLINE void pf_store(float (*acc)[NT][4], u32 rw, u32 n, u32 T, u32 t0, float* __restrict__ y,
  const float* __restrict__ x1) {
  const u32 l = threadIdx.x & 31, gid = l >> 2, tig = l & 3;
  #pragma unroll
  for (u32 mt = 0; mt < (EP == 2 ? 1 : MT); mt++) {
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++) {
      #pragma unroll
      for (u32 e = 0; e < 4; e++) {
        const u32 r = rw + 16 * mt + gid + 8 * (e >> 1), tk = t0 + 8 * nt + 2 * tig + (e & 1);
        if (r < n && tk < T) {
          const u64 o = (u64)tk * n + r;
          float v = acc[mt][nt][e];
          if (EP == 1) v = x1[o] + v;
          if (EP == 2) v = v / (1.0f + expf(-v)) * acc[1][nt][e];
          y[o] = v;
        }
      }
    }
  }
}

INLINE u32 pf_b12(uint4 h, u32 k) {  // byte k of the Q4_K 12-byte scales (h.y .. h.w)
  const u32 w = k < 4 ? h.y : k < 8 ? h.z : h.w;
  return (w >> (8 * (k & 3))) & 255;
}

// Q4_K: rb = c / 256 * 144; grid (n / (64 MT) or n / 64 for EP 2, T / (8 NT)), 128 threads
template <int MT, int NT, int EP>
INLINE void pf_q4k(const u8* __restrict__ Wa, const u8* __restrict__ Wb, u32 n, u32 c,
  const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T, float* __restrict__ y,
  const float* __restrict__ x1) {
  __shared__ __align__(16) signed char xs[NT * 8][PF_XS];
  __shared__ float2 xe[NT * 8][8];
  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5, gid = l >> 2, tig = l & 3;
  const u32 rw = (blockIdx.x * 4 + w) * (EP == 2 ? 16 : 16 * MT), t0 = blockIdx.y * NT * 8;
  const u32 rb = c / 256 * 144;
  float acc[MT][NT][4];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;
  const u8* rp[MT][2];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++) {
    #pragma unroll
    for (u32 h = 0; h < 2; h++) {
      u32 r = rw + (EP == 2 ? 0 : 16 * mt) + gid + 8 * h;
      r = r < n ? r : n - 1;
      rp[mt][h] = (EP == 2 && mt == 1 ? Wb : Wa) + (u64)r * rb;
    }
  }
  for (u32 kb = 0; kb < c / 256; kb++) {
    __syncthreads();
    pf_slab<NT>(xs, xe, qx, xd, c, T, t0, kb);
    __syncthreads();
    uint4 hd[MT][2];
    #pragma unroll
    for (u32 mt = 0; mt < MT; mt++)
      #pragma unroll
      for (u32 h = 0; h < 2; h++) hd[mt][h] = *(const uint4*)(rp[mt][h] + kb * 144);
    #pragma unroll 1
    for (u32 g = 0; g < 4; g++) {
      #pragma unroll
      for (u32 mt = 0; mt < MT; mt++) {
        unsigned long long qs[2];
        #pragma unroll
        for (u32 h = 0; h < 2; h++) qs[h] = *(const unsigned long long*)(rp[mt][h] + kb * 144 + 16 + 32 * g + 8 * tig);
        #pragma unroll
        for (u32 jj = 0; jj < 2; jj++) {
          const u32 j = 2 * g + jj;
          float ds[2], dm[2];
          #pragma unroll
          for (u32 h = 0; h < 2; h++) {
            const uint4 q = hd[mt][h];
            const u32 sc = j < 4 ? pf_b12(q, j) & 63 : (pf_b12(q, j + 4) & 15) | (pf_b12(q, j - 4) >> 6) << 4;
            const u32 m = j < 4 ? pf_b12(q, j + 4) & 63 : (pf_b12(q, j + 4) >> 4) | (pf_b12(q, j) >> 6) << 4;
            ds[h] = llb_f16(q.x & 0xffff) * (float)sc;
            dm[h] = llb_f16(q.x >> 16) * (float)m;
          }
          const u32 sh = 4 * jj;
          const u32 a0 = ((u32)qs[0] >> sh) & 0x0F0F0F0Fu, a2 = ((u32)(qs[0] >> 32) >> sh) & 0x0F0F0F0Fu;
          const u32 a1 = ((u32)qs[1] >> sh) & 0x0F0F0F0Fu, a3 = ((u32)(qs[1] >> 32) >> sh) & 0x0F0F0F0Fu;
          #pragma unroll
          for (u32 nt = 0; nt < NT; nt++) {
            const uint2 b = *(const uint2*)&xs[8 * nt + gid][32 * j + 8 * tig];
            int ci[4];
            pf_mma32(ci, a0, a1, a2, a3, b.x, b.y);
            const float2 e0 = xe[8 * nt + 2 * tig][j], e1 = xe[8 * nt + 2 * tig + 1][j];
            acc[mt][nt][0] = fmaf(-dm[0], e0.y, fmaf(ds[0] * e0.x, (float)ci[0], acc[mt][nt][0]));
            acc[mt][nt][1] = fmaf(-dm[0], e1.y, fmaf(ds[0] * e1.x, (float)ci[1], acc[mt][nt][1]));
            acc[mt][nt][2] = fmaf(-dm[1], e0.y, fmaf(ds[1] * e0.x, (float)ci[2], acc[mt][nt][2]));
            acc[mt][nt][3] = fmaf(-dm[1], e1.y, fmaf(ds[1] * e1.x, (float)ci[3], acc[mt][nt][3]));
          }
        }
      }
    }
  }
  pf_store<MT, NT, EP>(acc, rw, n, T, t0, y, x1);
}

// the repacked Q8_0: rb = c + c / 16; grid (n / (64 MT), T / (8 NT))
template <int MT, int NT, int EP>
INLINE void pf_q8(const u8* __restrict__ W, u32 n, u32 c, const signed char* __restrict__ qx,
  const float2* __restrict__ xd, u32 T, float* __restrict__ y, const float* __restrict__ x1) {
  __shared__ __align__(16) signed char xs[NT * 8][PF_XS];
  __shared__ float2 xe[NT * 8][8];
  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5, gid = l >> 2, tig = l & 3;
  const u32 rw = (blockIdx.x * 4 + w) * 16 * MT, t0 = blockIdx.y * NT * 8;
  const u32 rb = c + c / 16;
  float acc[MT][NT][4];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;
  const u8* rp[MT][2];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++) {
    #pragma unroll
    for (u32 h = 0; h < 2; h++) {
      u32 r = rw + 16 * mt + gid + 8 * h;
      r = r < n ? r : n - 1;
      rp[mt][h] = W + (u64)r * rb;
    }
  }
  for (u32 kb = 0; kb < c / 256; kb++) {
    __syncthreads();
    pf_slab<NT>(xs, xe, qx, xd, c, T, t0, kb);
    __syncthreads();
    #pragma unroll 1
    for (u32 j = 0; j < 8; j++) {
      #pragma unroll
      for (u32 mt = 0; mt < MT; mt++) {
        uint2 wq[2];
        float wd[2];
        #pragma unroll
        for (u32 h = 0; h < 2; h++) {
          wq[h] = *(const uint2*)(rp[mt][h] + 256 * kb + 32 * j + 8 * tig);
          wd[h] = llb_f16(*(const unsigned short*)(rp[mt][h] + c + 16 * kb + 2 * j));
        }
        #pragma unroll
        for (u32 nt = 0; nt < NT; nt++) {
          const uint2 b = *(const uint2*)&xs[8 * nt + gid][32 * j + 8 * tig];
          int ci[4];
          pf_mma32(ci, wq[0].x, wq[1].x, wq[0].y, wq[1].y, b.x, b.y);
          const float e0 = xe[8 * nt + 2 * tig][j].x, e1 = xe[8 * nt + 2 * tig + 1][j].x;
          acc[mt][nt][0] = fmaf(wd[0] * e0, (float)ci[0], acc[mt][nt][0]);
          acc[mt][nt][1] = fmaf(wd[0] * e1, (float)ci[1], acc[mt][nt][1]);
          acc[mt][nt][2] = fmaf(wd[1] * e0, (float)ci[2], acc[mt][nt][2]);
          acc[mt][nt][3] = fmaf(wd[1] * e1, (float)ci[3], acc[mt][nt][3]);
        }
      }
    }
  }
  pf_store<MT, NT, EP>(acc, rw, n, T, t0, y, x1);
}

INLINE u32 pf_w4(const u8* p) {  // 4 bytes at a 2-aligned address
  return (u32)*(const unsigned short*)p | (u32)*(const unsigned short*)(p + 2) << 16;
}

// Q6_K: rb = c / 256 * 210; 16 groups of 16 a superblock (ggml dequantize_row_q6_K: group gi
// = 8 hn + 2 q + l0 / 16 holds cols 128 hn + 32 q + l0 .. +15, ql nibble q >> 1 of byte 64 hn
// + 32 (q & 1) + l, qh bits 2q of byte 32 hn + l, value - 32, scale sc[gi]); grid as Q8_0
template <int MT, int NT, int EP>
INLINE void pf_q6k(const u8* __restrict__ W, u32 n, u32 c, const signed char* __restrict__ qx,
  const float2* __restrict__ xd, u32 T, float* __restrict__ y, const float* __restrict__ x1) {
  __shared__ __align__(16) signed char xs[NT * 8][PF_XS];
  __shared__ float2 xe[NT * 8][8];
  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5, gid = l >> 2, tig = l & 3;
  const u32 rw = (blockIdx.x * 4 + w) * 16 * MT, t0 = blockIdx.y * NT * 8;
  const u32 rb = c / 256 * 210;
  float acc[MT][NT][4];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;
  const u8* rp[MT][2];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++) {
    #pragma unroll
    for (u32 h = 0; h < 2; h++) {
      u32 r = rw + 16 * mt + gid + 8 * h;
      r = r < n ? r : n - 1;
      rp[mt][h] = W + (u64)r * rb;
    }
  }
  for (u32 kb = 0; kb < c / 256; kb++) {
    __syncthreads();
    pf_slab<NT>(xs, xe, qx, xd, c, T, t0, kb);
    __syncthreads();
    #pragma unroll 1
    for (u32 gi = 0; gi < 16; gi++) {
      const u32 hn = gi >> 3, q = (gi >> 1) & 3, l0 = 16 * (gi & 1), o = l0 + 4 * tig;
      const u32 qo = 64 * hn + 32 * (q & 1) + o, ho = 128 + 32 * hn + o, sh = 4 * (q >> 1);
      const u32 xo = 128 * hn + 32 * q + o, xb = 4 * hn + q;
      #pragma unroll
      for (u32 mt = 0; mt < MT; mt++) {
        const u8* B0 = rp[mt][0] + kb * 210;
        const u8* B1 = rp[mt][1] + kb * 210;
        const u32 a0 = __vsub4(((pf_w4(B0 + qo) >> sh) & 0x0F0F0F0Fu) | ((pf_w4(B0 + ho) >> (2 * q)) & 0x03030303u) << 4, 0x20202020u);
        const u32 a1 = __vsub4(((pf_w4(B1 + qo) >> sh) & 0x0F0F0F0Fu) | ((pf_w4(B1 + ho) >> (2 * q)) & 0x03030303u) << 4, 0x20202020u);
        const float s0 = llb_f16(*(const unsigned short*)(B0 + 208)) * (float)(signed char)B0[192 + gi];
        const float s1 = llb_f16(*(const unsigned short*)(B1 + 208)) * (float)(signed char)B1[192 + gi];
        #pragma unroll
        for (u32 nt = 0; nt < NT; nt++) {
          const u32 b = *(const u32*)&xs[8 * nt + gid][xo];
          int ci[4];
          pf_mma16(ci, a0, a1, b);
          const float e0 = xe[8 * nt + 2 * tig][xb].x, e1 = xe[8 * nt + 2 * tig + 1][xb].x;
          acc[mt][nt][0] = fmaf(s0 * e0, (float)ci[0], acc[mt][nt][0]);
          acc[mt][nt][1] = fmaf(s0 * e1, (float)ci[1], acc[mt][nt][1]);
          acc[mt][nt][2] = fmaf(s1 * e0, (float)ci[2], acc[mt][nt][2]);
          acc[mt][nt][3] = fmaf(s1 * e1, (float)ci[3], acc[mt][nt][3]);
        }
      }
    }
  }
  pf_store<MT, NT, EP>(acc, rw, n, T, t0, y, x1);
}

// v1 wide tile (T >= 16): 256 threads = WR x WC warps, a warp MT m-tiles x NT n-tiles; the block
// ROWS = 16 WR MT weight rows (EP 2: 64 gate rows then the same 64 up rows) x 64 tokens. Each 256-
// column step's raw W rows, x slab and x scales land in shared memory by cp.async, double-
// buffered; the W scales are decoded once a step into ws {d sc, dmin m} (Q8_0: {d, 0}).
// grid (T / 64 tokens, n / rows): the token blocks of a row block run together (W read ~once).
INLINE u32 pf_sa(const void* p) {
  u32 a;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
  return a;
}
INLINE void pf_cp16(void* s, const void* g, u32 nb) {  // nb 0 = zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(pf_sa(s)), "l"(g), "r"(nb));
}
INLINE void pf_cp8(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(pf_sa(s)), "l"(g));
}
// TY 3 Q4_K: a step = 256 columns (one superblock: 144 B a row); TY 1 Q8_0: 128 columns (128 qs
// + 8 scale B a row). Strides = 8 (mod 32) words: a half-warp's 8-byte reads of 4 rows (tokens)
// hit distinct banks; EB = 2 tokens' scale rows apart in distinct banks.
// NT: a warp's n-tiles; the block's tokens TK = 8 NT (8 / WR) (PF28: Q8_0 runs NT 8 = 128 tokens)
// PF28 TY 5 Q5_K (176 B: Q4_K's d, dmin, sc[12], then qh[32], qs[128]): the header and qs land as
// Q4_K's, qh in its own plane Q0 (32 B a row); sub-block j's fifth bit is bit j of qh[l]. NT 3 (48
// tokens) keeps the plane inside 99 KB.
template <int TY, int WR, int MT, int EP, int NT = 4>
struct PfT {
  static constexpr u32 KS = TY != 1 ? 256 : 128, NJ = KS / 32, XB = KS + 32;
  static constexpr u32 ROWS = 16 * WR * MT, WB = 160, EB = TY != 1 ? 80 : 48, TK = 8 * NT * (8 / WR);
  static constexpr u32 W0 = 0, X0 = ROWS * WB, E0 = X0 + TK * XB, Q0 = E0 + TK * EB;
  static constexpr u32 ST = Q0 + (TY == 5 ? ROWS * 32 : 0);
  static constexpr u32 SM = 2 * ST + ROWS * NJ * 8;  // + ws [NJ][ROWS] float2
};
template <int TY, int WR, int MT, int EP, int NT>
INLINE void pf_stage(u8* sm, const u8* __restrict__ Wa, const u8* __restrict__ Wb, u32 n, u32 c,
  const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T, u32 t0, u32 r0, u32 kb) {
  typedef PfT<TY, WR, MT, EP, NT> P;
  constexpr u32 CH = TY == 5 ? 11 : 9;  // 16-byte chunks a W row (Q8_0: 8 qs + the 8 scale bytes)
  for (u32 i = threadIdx.x; i < P::ROWS * CH; i += 256) {
    const u32 tr = i / CH, ch = i % CH;
    u32 r = r0 + (EP == 2 ? (tr & 63) : tr);
    r = r < n ? r : n - 1;
    const u8* row = (EP == 2 && tr >= 64 ? Wb : Wa) + (u64)r * (TY == 1 ? c + c / 16 : c / 256 * (TY == 5 ? 176 : 144));
    u8* d = sm + P::W0 + tr * P::WB + 16 * ch;
    if (TY == 5) pf_cp16(ch == 0 ? d : ch < 3 ? sm + P::Q0 + tr * 32 + 16 * (ch - 1) : d - 32, row + kb * 176 + 16 * ch, 16);
    else if (TY == 3) pf_cp16(d, row + kb * 144 + 16 * ch, 16);
    else if (ch < 8) pf_cp16(d, row + 128 * kb + 16 * ch, 16);
    else pf_cp8(d, row + c + 8 * kb);
  }
  constexpr u32 XC = P::KS / 16, EC = P::NJ / 2;
  for (u32 i = threadIdx.x; i < P::TK * XC; i += 256) {
    const u32 tk = i / XC, ch = i % XC, ok = t0 + tk < T;
    pf_cp16(sm + P::X0 + tk * P::XB + 16 * ch, qx + (u64)(ok ? t0 + tk : 0) * c + P::KS * kb + 16 * ch, ok ? 16 : 0);
  }
  for (u32 i = threadIdx.x; i < P::TK * EC; i += 256) {
    const u32 tk = i / EC, ch = i % EC, ok = t0 + tk < T;
    pf_cp16(sm + P::E0 + tk * P::EB + 16 * ch, xd + (u64)(ok ? t0 + tk : 0) * (c / 32) + P::NJ * kb + 2 * ch, ok ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;");
}
template <int TY, int WR, int MT, int NT, int EP>
INLINE void pf_wide(const u8* __restrict__ Wa, const u8* __restrict__ Wb, u32 n, u32 c,
  const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T, float* __restrict__ y,
  const float* __restrict__ x1, u32 rbk) {
  typedef PfT<TY, WR, MT, EP, NT> P;
  extern __shared__ __align__(16) u8 sm[];
  float2* ws = (float2*)(sm + 2 * P::ST);  // [NJ][ROWS]
  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5, gid = l >> 2, tig = l & 3;
  const u32 wr = w % WR, wc = w / WR, t0 = blockIdx.x * P::TK, r0 = rbk * (EP == 2 ? 64 : P::ROWS);
  u32 tro[MT];  // the warp's tile row of m-tile mt
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++) tro[mt] = EP == 2 ? 64 * mt + 16 * wr : 16 * (wr * MT + mt);
  float acc[MT][NT][4];
  #pragma unroll
  for (u32 mt = 0; mt < MT; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;
  const u32 nk = c / P::KS;
  pf_stage<TY, WR, MT, EP, NT>(sm, Wa, Wb, n, c, qx, xd, T, t0, r0, 0);
  for (u32 kb = 0; kb < nk; kb++) {
    if (kb + 1 < nk) {
      pf_stage<TY, WR, MT, EP, NT>(sm + ((kb + 1) & 1) * P::ST, Wa, Wb, n, c, qx, xd, T, t0, r0, kb + 1);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    const u8* S = sm + (kb & 1) * P::ST;
    for (u32 i = threadIdx.x; i < P::ROWS * P::NJ; i += 256) {
      const u32 tr = i % P::ROWS, j = i / P::ROWS;
      const u8* B = S + P::W0 + tr * P::WB;
      if (TY != 1) {
        const uint4 q = *(const uint4*)B;
        const u32 sc = j < 4 ? pf_b12(q, j) & 63 : (pf_b12(q, j + 4) & 15) | (pf_b12(q, j - 4) >> 6) << 4;
        const u32 m = j < 4 ? pf_b12(q, j + 4) & 63 : (pf_b12(q, j + 4) >> 4) | (pf_b12(q, j) >> 6) << 4;
        ws[i] = make_float2(llb_f16(q.x & 0xffff) * (float)sc, llb_f16(q.x >> 16) * (float)m);
      } else {
        ws[i] = make_float2(llb_f16(*(const unsigned short*)(B + 128 + 2 * j)), 0.0f);
      }
    }
    __syncthreads();
    const signed char* xs = (const signed char*)(S + P::X0);
    const u8* xe = S + P::E0;
    #pragma unroll 1
    for (u32 g = 0; g < P::NJ / 2; g++) {
      unsigned long long qa[MT][2], qh[MT][2];
      #pragma unroll
      for (u32 mt = 0; mt < MT; mt++)
        #pragma unroll
        for (u32 h = 0; h < 2; h++) {
          const u8* B = S + P::W0 + (tro[mt] + gid + 8 * h) * P::WB;
          qa[mt][h] = TY != 1 ? *(const unsigned long long*)(B + 16 + 32 * g + 8 * tig) : 0;
          qh[mt][h] = TY == 5 ? *(const unsigned long long*)(S + P::Q0 + (tro[mt] + gid + 8 * h) * 32 + 8 * tig) : 0;
        }
      #pragma unroll
      for (u32 jj = 0; jj < 2; jj++) {
        const u32 j = 2 * g + jj;
        u32 a[MT][4];
        float2 sc[MT][2];
        #pragma unroll
        for (u32 mt = 0; mt < MT; mt++) {
          #pragma unroll
          for (u32 h = 0; h < 2; h++) {
            const u32 tr = tro[mt] + gid + 8 * h;
            unsigned long long v = qa[mt][h];
            if (TY != 1) {
              const u32 sh = 4 * jj;
              v = (u64)(((u32)v >> sh) & 0x0F0F0F0Fu) | (u64)(((u32)(v >> 32) >> sh) & 0x0F0F0F0Fu) << 32;
              if (TY == 5) v |= (qh[mt][h] >> (2 * g + jj) & 0x0101010101010101ull) << 4;
            } else {
              v = *(const unsigned long long*)(S + P::W0 + tr * P::WB + 32 * j + 8 * tig);
            }
            a[mt][h] = (u32)v;
            a[mt][2 + h] = (u32)(v >> 32);
            sc[mt][h] = ws[j * P::ROWS + tr];
          }
        }
        #pragma unroll
        for (u32 nt = 0; nt < NT; nt++) {
          const u32 tk = 8 * (wc * NT + nt);
          const uint2 b = *(const uint2*)&xs[(tk + gid) * P::XB + 32 * j + 8 * tig];
          const float2 e0 = *(const float2*)(xe + (tk + 2 * tig) * P::EB + 8 * j);
          const float2 e1 = *(const float2*)(xe + (tk + 2 * tig + 1) * P::EB + 8 * j);
          #pragma unroll
          for (u32 mt = 0; mt < MT; mt++) {
            int ci[4];
            pf_mma32m(ci, a[mt][0], a[mt][1], a[mt][2], a[mt][3], b.x, b.y);
            acc[mt][nt][0] = fmaf(-sc[mt][0].y, e0.y, fmaf(sc[mt][0].x * e0.x, pf_f(ci[0]), acc[mt][nt][0]));
            acc[mt][nt][1] = fmaf(-sc[mt][0].y, e1.y, fmaf(sc[mt][0].x * e1.x, pf_f(ci[1]), acc[mt][nt][1]));
            acc[mt][nt][2] = fmaf(-sc[mt][1].y, e0.y, fmaf(sc[mt][1].x * e0.x, pf_f(ci[2]), acc[mt][nt][2]));
            acc[mt][nt][3] = fmaf(-sc[mt][1].y, e1.y, fmaf(sc[mt][1].x * e1.x, pf_f(ci[3]), acc[mt][nt][3]));
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (u32 mt = 0; mt < (EP == 2 ? 1 : MT); mt++)
    #pragma unroll
    for (u32 nt = 0; nt < NT; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) {
        const u32 r = r0 + tro[mt] + gid + 8 * (e >> 1), tk = t0 + 8 * (wc * NT + nt) + 2 * tig + (e & 1);
        if (r < n && tk < T) {
          const u64 o = (u64)tk * n + r;
          float v = acc[mt][nt][e];
          if (EP == 1) v = x1[o] + v;
          if (EP == 2) v = v / (1.0f + expf(-v)) * acc[1][nt][e];
          y[o] = v;
        }
      }
}
#define PF_WIDE(nm, TY, WR, MT, NT, EP) \
  extern "C" __global__ void __launch_bounds__(256, 1) nm(const u8* __restrict__ W, const u8* __restrict__ Wb, \
    u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T, \
    float* __restrict__ y, const float* __restrict__ x1) { pf_wide<TY, WR, MT, NT, EP>(W, Wb, n, c, qx, xd, T, y, x1, blockIdx.y); }
PF_WIDE(pf_q4k_m, 3, 4, 2, 4, 0)
PF_WIDE(pf_q5k_m, 5, 4, 2, 3, 0)  // PF28
PF_WIDE(pf_q5k_mr, 5, 4, 2, 3, 1)
PF_WIDE(pf_q4k_mr, 3, 4, 2, 4, 1)
PF_WIDE(pf_q4k_mg, 3, 4, 2, 4, 2)
PF_WIDE(pf_q8_m, 1, 4, 2, 8, 0)
PF_WIDE(pf_q8_mr, 1, 4, 2, 8, 1)
// PF28: up to 4 Q8_0 tensors on the same x in one launch (GDN qkv / z / beta / alpha; attn q / k / v):
// row blocks b[i] .. b[i + 1] - 1 run tensor i (n[i] rows into y[i]); grid (T / 128, b[4])
struct PfCat { const u8* W[4]; float* y[4]; u32 n[4]; u32 b[5]; };
extern "C" __global__ void __launch_bounds__(256, 1) pf_q8_mc(PfCat k, u32 c,
  const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T) {
  const u32 rb = blockIdx.y, i = rb < k.b[1] ? 0 : rb < k.b[2] ? 1 : rb < k.b[3] ? 2 : 3;
  pf_wide<1, 4, 2, 8, 0>(k.W[i], k.W[i], k.n[i], c, qx, xd, T, k.y[i], nullptr, rb - k.b[i]);
}

// PF28 Q6_K wide tile: 256 threads = 4 x 2 warps, a warp 2 m-tiles x 4 n-tiles, 128 rows x 64
// tokens a block; grid (T / 64, n / 128). The 210-byte superblock is only 2-aligned (no cp.async):
// a step's raw rows load into registers one step ahead (aligned u32 + funnel shift), then unpack
// to int8 (q - 32) in shared memory, a 32-block's two 16-groups interleaved per lane (bytes 8 tig
// .. +3 = group 0's 4 tig .. +3, +4 .. +7 = group 1's) so one u64 feeds both m16n8k16; the scales
// d sc go to ws [16][128]. x: cp.async double-buffered, row stride 272 (conflict-free u32 reads).
// acc += e.d (d sc0 W0.x0 + d sc1 W1.x1), the int sums by the magic seed (|sum| <= 16 32 127).
#define PF6_WB 288
#define PF6_XB 272
#define PF6_EB 80
#define PF6_X0 (128 * PF6_WB + 16 * 128 * 4)
#define PF6_ST (64 * PF6_XB + 64 * PF6_EB)
#define PF6_SM (PF6_X0 + 2 * PF6_ST)
INLINE void pf_mma16m(int* c, u32 a0, u32 a1, u32 b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
    "{%7,%7,%7,%7};" : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
    : "r"(a0), "r"(a1), "r"(b0), "r"(PF_MAGIC));
}
INLINE void pf6_x(u8* S, const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 c, u32 T,
  u32 t0, u32 kb) {
  for (u32 i = threadIdx.x; i < 64 * 16; i += 256) {
    const u32 tk = i >> 4, ch = i & 15, ok = t0 + tk < T;
    pf_cp16(S + tk * PF6_XB + 16 * ch, qx + (u64)(ok ? t0 + tk : 0) * c + 256 * kb + 16 * ch, ok ? 16 : 0);
  }
  for (u32 i = threadIdx.x; i < 64 * 4; i += 256) {
    const u32 tk = i >> 2, ch = i & 3, ok = t0 + tk < T;
    pf_cp16(S + 64 * PF6_XB + tk * PF6_EB + 16 * ch, xd + (u64)(ok ? t0 + tk : 0) * (c / 32) + 8 * kb + 2 * ch, ok ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;");
}
// a unit = (row, hn, half): ql 16 B at 64 hn + 16 half and + 32, qh 16 B at 128 + 32 hn + 16 half
struct Pf6R { u32 a[5], b[5], h[5]; u32 sc, d; };
INLINE void pf6_ld(Pf6R& R, const u8* __restrict__ W, u32 n, u32 c, u32 r0, u32 u, u32 kb) {
  u32 r = r0 + (u >> 2);
  r = r < n ? r : n - 1;
  const u32 hn = (u >> 1) & 1, hf = u & 1;
  const u8* B = W + (u64)r * (c / 256 * 210) + kb * 210;
  const u32 off = (u32)(u64)B & 3;
  const u32* A = (const u32*)(B - off);
  #pragma unroll
  for (u32 k = 0; k < 5; k++) {
    R.a[k] = A[(64 * hn + 16 * hf) / 4 + k];
    R.b[k] = A[(64 * hn + 16 * hf + 32) / 4 + k];
    R.h[k] = A[(128 + 32 * hn + 16 * hf) / 4 + k];
  }
  // scales of groups 8 hn + 2 q + hf, q = 0..3: bytes 192 + 8 hn + hf + 2 q
  const u8* sp = B + 192 + 8 * hn + hf;
  R.sc = (u32)sp[0] | (u32)sp[2] << 8 | (u32)sp[4] << 16 | (u32)sp[6] << 24;
  R.d = *(const unsigned short*)(B + 208);
  if (off) {
    #pragma unroll
    for (u32 k = 0; k < 4; k++) {
      R.a[k] = __funnelshift_r(R.a[k], R.a[k + 1], 16);
      R.b[k] = __funnelshift_r(R.b[k], R.b[k + 1], 16);
      R.h[k] = __funnelshift_r(R.h[k], R.h[k + 1], 16);
    }
  }
}
INLINE void pf6_st(const Pf6R& R, u8* sm, u32 u) {
  const u32 tr = u >> 2, hn = (u >> 1) & 1, hf = u & 1;
  float* ws = (float*)(sm + 128 * PF6_WB);
  const float d = llb_f16(R.d);
  #pragma unroll
  for (u32 q = 0; q < 4; q++) {
    ws[(8 * hn + 2 * q + hf) * 128 + tr] = d * (float)(signed char)(R.sc >> (8 * q));
    u8* D = sm + tr * PF6_WB + 32 * (4 * hn + q) + 4 * hf;
    #pragma unroll
    for (u32 k = 0; k < 4; k++) {  // cols 16 hf + 4 k .. +3 of 32-block 4 hn + q = lane tig k
      const u32 lo = ((q & 1 ? R.b[k] : R.a[k]) >> (4 * (q >> 1))) & 0x0F0F0F0Fu;
      const u32 hi = (R.h[k] >> (2 * q)) & 0x03030303u;
      *(u32*)(D + 8 * k) = __vsub4(lo | hi << 4, 0x20202020u);
    }
  }
}
template <int EP>
INLINE void pf_wide6(const u8* __restrict__ W, u32 n, u32 c, const signed char* __restrict__ qx,
  const float2* __restrict__ xd, u32 T, float* __restrict__ y, const float* __restrict__ x1) {
  extern __shared__ __align__(16) u8 sm[];
  const float* ws = (const float*)(sm + 128 * PF6_WB);
  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5, gid = l >> 2, tig = l & 3;
  const u32 wr = w % 4, wc = w / 4, t0 = blockIdx.x * 64, r0 = blockIdx.y * 128;
  float acc[2][4][4];
  #pragma unroll
  for (u32 mt = 0; mt < 2; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < 4; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) acc[mt][nt][e] = 0.0f;
  const u32 nk = c / 256;
  Pf6R R0, R1;
  pf6_x(sm + PF6_X0, qx, xd, c, T, t0, 0);
  pf6_ld(R0, W, n, c, r0, threadIdx.x, 0);
  pf6_ld(R1, W, n, c, r0, threadIdx.x + 256, 0);
  for (u32 kb = 0; kb < nk; kb++) {
    __syncthreads();  // step kb - 1's reads of the W tile, ws and x buffer (kb + 1) & 1 are done
    if (kb + 1 < nk) pf6_x(sm + PF6_X0 + ((kb + 1) & 1) * PF6_ST, qx, xd, c, T, t0, kb + 1);
    pf6_st(R0, sm, threadIdx.x);
    pf6_st(R1, sm, threadIdx.x + 256);
    if (kb + 1 < nk) {
      pf6_ld(R0, W, n, c, r0, threadIdx.x, kb + 1);
      pf6_ld(R1, W, n, c, r0, threadIdx.x + 256, kb + 1);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    const u8* X = sm + PF6_X0 + (kb & 1) * PF6_ST;
    const u8* xe = X + 64 * PF6_XB;
    #pragma unroll 1
    for (u32 j = 0; j < 8; j++) {
      u32 a[2][2][2];  // [mt][group][row half]
      float s[2][2][2];
      #pragma unroll
      for (u32 mt = 0; mt < 2; mt++)
        #pragma unroll
        for (u32 h = 0; h < 2; h++) {
          const u32 tr = 16 * (wr * 2 + mt) + gid + 8 * h;
          const uint2 v = *(const uint2*)(sm + tr * PF6_WB + 32 * j + 8 * tig);
          a[mt][0][h] = v.x;
          a[mt][1][h] = v.y;
          s[mt][0][h] = ws[(2 * j) * 128 + tr];
          s[mt][1][h] = ws[(2 * j + 1) * 128 + tr];
        }
      #pragma unroll
      for (u32 nt = 0; nt < 4; nt++) {
        const u32 tk = 8 * (wc * 4 + nt);
        const u32 b0 = *(const u32*)(X + (tk + gid) * PF6_XB + 32 * j + 4 * tig);
        const u32 b1 = *(const u32*)(X + (tk + gid) * PF6_XB + 32 * j + 16 + 4 * tig);
        const float e0 = *(const float*)(xe + (tk + 2 * tig) * PF6_EB + 8 * j);
        const float e1 = *(const float*)(xe + (tk + 2 * tig + 1) * PF6_EB + 8 * j);
        #pragma unroll
        for (u32 mt = 0; mt < 2; mt++) {
          int c0[4], c1[4];
          pf_mma16m(c0, a[mt][0][0], a[mt][0][1], b0);
          pf_mma16m(c1, a[mt][1][0], a[mt][1][1], b1);
          #pragma unroll
          for (u32 e = 0; e < 4; e++) {
            const u32 h = e >> 1;
            const float t = fmaf(s[mt][1][h], pf_f(c1[e]), s[mt][0][h] * pf_f(c0[e]));
            acc[mt][nt][e] = fmaf(e & 1 ? e1 : e0, t, acc[mt][nt][e]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (u32 mt = 0; mt < 2; mt++)
    #pragma unroll
    for (u32 nt = 0; nt < 4; nt++)
      #pragma unroll
      for (u32 e = 0; e < 4; e++) {
        const u32 r = r0 + 16 * (wr * 2 + mt) + gid + 8 * (e >> 1), tk = t0 + 8 * (wc * 4 + nt) + 2 * tig + (e & 1);
        if (r < n && tk < T) {
          const u64 o = (u64)tk * n + r;
          y[o] = EP == 1 ? x1[o] + acc[mt][nt][e] : acc[mt][nt][e];
        }
      }
}
extern "C" __global__ void __launch_bounds__(256, 1) pf_q6k_m(const u8* __restrict__ W, const u8* __restrict__ Wb,
  u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T,
  float* __restrict__ y, const float* __restrict__ x1) { pf_wide6<0>(W, n, c, qx, xd, T, y, x1); }
extern "C" __global__ void __launch_bounds__(256, 1) pf_q6k_mr(const u8* __restrict__ W, const u8* __restrict__ Wb,
  u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T,
  float* __restrict__ y, const float* __restrict__ x1) { pf_wide6<1>(W, n, c, qx, xd, T, y, x1); }

// the leaves: _w (wide, 64 tokens a block) and _t (thin, 8); the registry is pf_pick below
#define PF_LEAF(nm, MT, NT, EP, body) \
  extern "C" __global__ void __launch_bounds__(128) nm(const u8* __restrict__ W, const u8* __restrict__ Wb, \
    u32 n, u32 c, const signed char* __restrict__ qx, const float2* __restrict__ xd, u32 T, \
    float* __restrict__ y, const float* __restrict__ x1) { body; }
PF_LEAF(pf_q4k_w, 2, 8, 0, (pf_q4k<2, 8, 0>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q4k_wr, 2, 8, 1, (pf_q4k<2, 8, 1>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q4k_wg, 2, 8, 2, (pf_q4k<2, 8, 2>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q4k_t, 1, 1, 0, (pf_q4k<1, 1, 0>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q4k_tr, 1, 1, 1, (pf_q4k<1, 1, 1>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q4k_tg, 2, 1, 2, (pf_q4k<2, 1, 2>(W, Wb, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q8_w, 2, 8, 0, (pf_q8<2, 8, 0>(W, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q8_wr, 2, 8, 1, (pf_q8<2, 8, 1>(W, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q8_t, 1, 1, 0, (pf_q8<1, 1, 0>(W, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q6k_w, 2, 8, 0, (pf_q6k<2, 8, 0>(W, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q6k_wr, 2, 8, 1, (pf_q6k<2, 8, 1>(W, n, c, qx, xd, T, y, x1)))
PF_LEAF(pf_q6k_t, 1, 1, 0, (pf_q6k<1, 1, 0>(W, n, c, qx, xd, T, y, x1)))
