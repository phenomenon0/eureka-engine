// The Q4_K dp4a tier (tools/payload-dp4a.patch, LLB_Q4_DP4A=1): llama.cpp's
// mmvq method (ggml-cuda vec_dot_q4_K_q8_1, quantize_q8_1) with every float op
// explicit, so it has a bit-exact CPU twin (tools/q4dp/q4dp_ref.c).
// x is quantized once per vector in blocks of 32: d = amax / 127, q = round(x
// / d) clamped to +-127 (0 if d is 0), d f32 (llama rounds it to f16); a block
// with an inf / NaN gets d = NaN, q = 0, so its rows come out NaN.
// A row of nsb Q4_K superblocks: thread t = 16 s + 4 g + k takes superblocks
// s, s + 8, ..; in each the int pair at qs 32 g + 4 k and + 16 (lo nibbles =
// sub-block 2 g, hi = 2 g + 1) against x's ints k and k + 4 of those blocks:
//   A = d8[2g] * (dot0 * sc0), A = fma(d8[2g + 1], dot1 * sc1, A)
//   M = d8[2g] * (sum0 * m0),  M = fma(d8[2g + 1], sum1 * m1, M)
//   acc += fma(d, A, -(dmin * M))      (sum = the q8 ints' sum, dp4a by 1s)
// then warps 1..3 into warp 0 in order and the xor butterfly 16 .. 1.
__device__ __forceinline__ float q4dp_f16(u32 h) { return llb_f16(h & 0xffffu); }

// x[n] (n % 256 == 0) -> qx[n] int8, dx[n / 32]: a warp a block of 32
extern "C" __global__ void __launch_bounds__(256) llb_q8x(const float* __restrict__ x,
  signed char* __restrict__ qx, float* __restrict__ dx, u32 n) {
  const u32 i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  float a = fabsf(v);
  int nf = (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u;
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
  nf = __any_sync(0xffffffffu, nf);
  const float d = __fdiv_rn(a, 127.0f);
  int q = 0;
  if (!nf && d != 0.0f) q = max(-127, min(127, (int)roundf(__fdiv_rn(v, d))));
  qx[i] = (signed char)q;
  if ((threadIdx.x & 31) == 0) dx[i >> 5] = nf ? __uint_as_float(0x7fffffffu) : d;
}

// one superblock's share for thread (g, k)
__device__ __forceinline__ float q4dp_sb(const u8* __restrict__ B, const signed char* __restrict__ qx,
  const float* __restrict__ dx, u32 sb, u32 g, u32 k) {
  const u32 dm = *(const u32*)B;
  const unsigned short* s16 = (const unsigned short*)(B + 4);
  u32 a0, a1;  // scales then mins of sub-blocks 2g, 2g + 1 (llama's aux pair)
  if (g < 2) {
    a0 = s16[g] & 0x3f3fu;
    a1 = s16[g + 2] & 0x3f3fu;
  } else {
    a0 = ((s16[g + 2] >> 0) & 0x0f0fu) | ((s16[g - 2] & 0xc0c0u) >> 2);
    a1 = ((s16[g + 2] >> 4) & 0x0f0fu) | ((s16[g] & 0xc0c0u) >> 2);
  }
  const int* q4 = (const int*)(B + 16 + 32 * g + 4 * k);
  const int v0 = q4[0], v1 = q4[4];
  const u32 j0 = 8 * sb + 2 * g;
  const int* u = (const int*)(qx + 32 * j0 + 4 * k);
  const int u00 = u[0], u01 = u[4], u10 = u[8], u11 = u[12];
  const float e0 = dx[j0], e1 = dx[j0 + 1];
  const int t0 = __dp4a(v1 & 0x0F0F0F0F, u01, __dp4a(v0 & 0x0F0F0F0F, u00, 0));
  const int t1 = __dp4a((v1 >> 4) & 0x0F0F0F0F, u11, __dp4a((v0 >> 4) & 0x0F0F0F0F, u10, 0));
  const int s0 = __dp4a(0x01010101, u01, __dp4a(0x01010101, u00, 0));
  const int s1 = __dp4a(0x01010101, u11, __dp4a(0x01010101, u10, 0));
  float A = __fmul_rn(e0, (float)(t0 * (int)(a0 & 0xff)));
  A = __fmaf_rn(e1, (float)(t1 * (int)(a0 >> 8)), A);
  float M = __fmul_rn(e0, (float)(s0 * (int)(a1 & 0xff)));
  M = __fmaf_rn(e1, (float)(s1 * (int)(a1 >> 8)), M);
  return __fmaf_rn(q4dp_f16(dm), A, -__fmul_rn(q4dp_f16(dm >> 16), M));
}

struct DpPlain {
  static __device__ __forceinline__ float st(const float*, u32, float v) { return v; }
};
struct DpRes {
  static __device__ __forceinline__ float st(const float* x1, u32 r, float v) { return __fadd_rn(x1[r], v); }
};

// row r of W (rb bytes, nsb superblocks) into y[r]; a 128-thread block a row
template <class EP>
__device__ __forceinline__ void q4dp_row(const u8* __restrict__ W, u32 rb, u32 nsb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 r,
  const float* __restrict__ e) {
  __shared__ float part[3][32];
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5, g = (t >> 2) & 3, k = t & 3;
  const u8* row = W + (u64)r * rb;
  float acc = 0.0f;
  for (u32 sb = t >> 4; sb < nsb; sb += 8) acc = __fadd_rn(acc, q4dp_sb(row + 144 * sb, qx, dx, sb, g, k));
  if (w > 0) part[w - 1][l] = acc;
  __syncthreads();
  if (w > 0) return;
  acc = __fadd_rn(acc, part[0][l]);
  acc = __fadd_rn(acc, part[1][l]);
  acc = __fadd_rn(acc, part[2][l]);
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, m));
  if (l == 0) y[r] = EP::st(e, r, acc);
}

// grid n: y = W x
extern "C" __global__ void __launch_bounds__(128) llb_q4dp(const u8* __restrict__ W, u32 rb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 n) {
  q4dp_row<DpPlain>(W, rb, rb / 144, qx, dx, y, blockIdx.x, nullptr);
}
// grid n: gate and up row r in one block, both dots per superblock on the same
// x (llama's glu loop; stage 1: 852 GB/s vs 795 for 2 n blocks of one row),
// each in q4dp_row's order
extern "C" __global__ void __launch_bounds__(128) llb_q4dp_gu(const u8* __restrict__ Wg,
  const u8* __restrict__ Wu, u32 rb, const signed char* __restrict__ qx, const float* __restrict__ dx,
  float* __restrict__ yg, float* __restrict__ yu, u32 n) {
  __shared__ float part[2][3][32];
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5, g = (t >> 2) & 3, k = t & 3, nsb = rb / 144, r = blockIdx.x;
  const u8* rg = Wg + (u64)r * rb;
  const u8* ru = Wu + (u64)r * rb;
  float ag = 0.0f, au = 0.0f;
  for (u32 sb = t >> 4; sb < nsb; sb += 8) {
    ag = __fadd_rn(ag, q4dp_sb(rg + 144 * sb, qx, dx, sb, g, k));
    au = __fadd_rn(au, q4dp_sb(ru + 144 * sb, qx, dx, sb, g, k));
  }
  if (w > 0) {
    part[0][w - 1][l] = ag;
    part[1][w - 1][l] = au;
  }
  __syncthreads();
  if (w > 0) return;
  #pragma unroll
  for (u32 i = 0; i < 3; i++) {
    ag = __fadd_rn(ag, part[0][i][l]);
    au = __fadd_rn(au, part[1][i][l]);
  }
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) {
    ag = __fadd_rn(ag, __shfl_xor_sync(0xffffffffu, ag, m));
    au = __fadd_rn(au, __shfl_xor_sync(0xffffffffu, au, m));
  }
  if (l == 0) {
    yg[r] = ag;
    yu[r] = au;
  }
}
extern "C" __global__ void __launch_bounds__(128) llb_q4dp_res(const u8* __restrict__ W, u32 rb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 n,
  const float* __restrict__ x1) {
  q4dp_row<DpRes>(W, rb, rb / 144, qx, dx, y, blockIdx.x, x1);
}
