// The Q4_K dp4a tier, two int8 planes (tools/payload-dp4b.patch, LLB_Q4_DP4A=4):
// llb_q8x / q4dp_sb's numerics with x's rounding error cut ~127x. Per block of
// 32: d = amax / 127, q = round(x / d) clamped +-127, e = d / 127, q2 =
// round(fma(-d, q, x) / e) clamped +-127 (|q2| <= 64 but for e's rounding), so
// x = e (127 q + q2) to e / 2. qx[0, n) holds q, qx[n, 2n) q2, dx[n / 32] e (0
// where d is; NaN for a block with an inf / NaN, q = q2 = 0). The dots merge
// the planes as integers before the float combine (127 t + t2: |.| < 2^21, x sc
// < 2^27), so the float ops are q4dp_sb's. CPU twin: tools/q27c/q4dp2_ref.h.
extern "C" __global__ void __launch_bounds__(256) llb_q8x2(const float* __restrict__ x,
  signed char* __restrict__ qx, float* __restrict__ dx, u32 n) {
  const u32 i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  float a = fabsf(v);
  int nf = (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u;
  #pragma unroll
  for (u32 m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
  nf = __any_sync(0xffffffffu, nf);
  const float d = __fdiv_rn(a, 127.0f), e = __fdiv_rn(d, 127.0f);
  int q = 0, q2 = 0;
  if (!nf && d != 0.0f) q = max(-127, min(127, (int)roundf(__fdiv_rn(v, d))));
  if (!nf && e != 0.0f) q2 = max(-127, min(127, (int)roundf(__fdiv_rn(__fmaf_rn(-d, (float)q, v), e))));
  qx[i] = (signed char)q;
  qx[n + i] = (signed char)q2;
  if ((threadIdx.x & 31) == 0) dx[i >> 5] = nf ? __uint_as_float(0x7fffffffu) : e;
}

// q4dp_sb on the two planes (n = x's length: plane 2 at qx + n)
__device__ __forceinline__ float q4dp2_sb(const u8* __restrict__ B, const signed char* __restrict__ qx,
  const float* __restrict__ dx, u32 n, u32 sb, u32 g, u32 k) {
  const u32 dm = *(const u32*)B;
  const unsigned short* s16 = (const unsigned short*)(B + 4);
  u32 a0, a1;
  if (g < 2) {
    a0 = s16[g] & 0x3f3fu;
    a1 = s16[g + 2] & 0x3f3fu;
  } else {
    a0 = ((s16[g + 2] >> 0) & 0x0f0fu) | ((s16[g - 2] & 0xc0c0u) >> 2);
    a1 = ((s16[g + 2] >> 4) & 0x0f0fu) | ((s16[g] & 0xc0c0u) >> 2);
  }
  const int* q4 = (const int*)(B + 16 + 32 * g + 4 * k);
  const int v0 = q4[0], v1 = q4[4];
  const int l0 = v0 & 0x0F0F0F0F, l1 = v1 & 0x0F0F0F0F, h0 = (v0 >> 4) & 0x0F0F0F0F, h1 = (v1 >> 4) & 0x0F0F0F0F;
  const u32 j0 = 8 * sb + 2 * g;
  const int* u = (const int*)(qx + 32 * j0 + 4 * k);
  const int* w = (const int*)(qx + n + 32 * j0 + 4 * k);
  const int u00 = u[0], u01 = u[4], u10 = u[8], u11 = u[12];
  const int w00 = w[0], w01 = w[4], w10 = w[8], w11 = w[12];
  const float e0 = dx[j0], e1 = dx[j0 + 1];
  const int t0 = 127 * __dp4a(l1, u01, __dp4a(l0, u00, 0)) + __dp4a(l1, w01, __dp4a(l0, w00, 0));
  const int t1 = 127 * __dp4a(h1, u11, __dp4a(h0, u10, 0)) + __dp4a(h1, w11, __dp4a(h0, w10, 0));
  const int s0 = 127 * __dp4a(0x01010101, u01, __dp4a(0x01010101, u00, 0)) + __dp4a(0x01010101, w01, __dp4a(0x01010101, w00, 0));
  const int s1 = 127 * __dp4a(0x01010101, u11, __dp4a(0x01010101, u10, 0)) + __dp4a(0x01010101, w11, __dp4a(0x01010101, w10, 0));
  float A = __fmul_rn(e0, (float)(t0 * (int)(a0 & 0xff)));
  A = __fmaf_rn(e1, (float)(t1 * (int)(a0 >> 8)), A);
  float M = __fmul_rn(e0, (float)(s0 * (int)(a1 & 0xff)));
  M = __fmaf_rn(e1, (float)(s1 * (int)(a1 >> 8)), M);
  return __fmaf_rn(q4dp_f16(dm), A, -__fmul_rn(q4dp_f16(dm >> 16), M));
}

// q4dp_row / llb_q4dp_gu / llb_q4dp_res on q4dp2_sb, the same thread map and order
template <class EP>
__device__ __forceinline__ void q4dp2_row(const u8* __restrict__ W, u32 rb, u32 nsb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 r,
  const float* __restrict__ e) {
  __shared__ float part[3][32];
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5, g = (t >> 2) & 3, k = t & 3, n = 256 * nsb;
  const u8* row = W + (u64)r * rb;
  float acc = 0.0f;
  for (u32 sb = t >> 4; sb < nsb; sb += 8) acc = __fadd_rn(acc, q4dp2_sb(row + 144 * sb, qx, dx, n, sb, g, k));
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
extern "C" __global__ void __launch_bounds__(128) llb_q4dp2(const u8* __restrict__ W, u32 rb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 n) {
  q4dp2_row<DpPlain>(W, rb, rb / 144, qx, dx, y, blockIdx.x, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) llb_q4dp2_gu(const u8* __restrict__ Wg,
  const u8* __restrict__ Wu, u32 rb, const signed char* __restrict__ qx, const float* __restrict__ dx,
  float* __restrict__ yg, float* __restrict__ yu, u32 n) {
  __shared__ float part[2][3][32];
  const u32 t = threadIdx.x, l = t & 31, w = t >> 5, g = (t >> 2) & 3, k = t & 3, nsb = rb / 144, r = blockIdx.x;
  const u32 nx = 256 * nsb;
  const u8* rg = Wg + (u64)r * rb;
  const u8* ru = Wu + (u64)r * rb;
  float ag = 0.0f, au = 0.0f;
  for (u32 sb = t >> 4; sb < nsb; sb += 8) {
    ag = __fadd_rn(ag, q4dp2_sb(rg + 144 * sb, qx, dx, nx, sb, g, k));
    au = __fadd_rn(au, q4dp2_sb(ru + 144 * sb, qx, dx, nx, sb, g, k));
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
extern "C" __global__ void __launch_bounds__(128) llb_q4dp2_res(const u8* __restrict__ W, u32 rb,
  const signed char* __restrict__ qx, const float* __restrict__ dx, float* __restrict__ y, u32 n,
  const float* __restrict__ x1) {
  q4dp2_row<DpRes>(W, rb, rb / 144, qx, dx, y, blockIdx.x, x1);
}
