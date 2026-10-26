// q4dp_ref.h: the CPU twin of tools/q4dp/q4dp.cuh (K1). Host code only, built
// with -ffp-contract=off; fmaf where the device has __fmaf_rn. The scale /
// min unpack is ggml's get_scale_min_k4 and the dot is plain integer sums
// over bytes (not the device's packed masks), so the twin is an independent
// read of the format. mut: 0 none, 1 sc + 1, 2 lo / hi nibble swap, 3 x
// block off by one, 4 the dmin term's sign flipped (each must FAIL K1).
#include <math.h>
#include <string.h>
typedef unsigned int u32;
typedef unsigned char u8;
static float ref_f16(u32 h) {
  u32 e = (h >> 10) & 31, m = h & 1023, s = (h >> 15) << 31;
  float mag = ldexpf((float)(e ? m + 1024 : m), (int)(e ? e : 1) - 25);
  u32 b; memcpy(&b, &mag, 4);
  b = e == 31 ? s | 0x7f800000u | m << 13 : b | s;
  float f; memcpy(&f, &b, 4); return f;
}
static void ref_q8x(const float* x, signed char* qx, float* dx, u32 n) {
  for (u32 j = 0; j < n / 32; j++) {
    float a = 0.0f; int nf = 0;
    for (u32 i = 0; i < 32; i++) { a = fmaxf(a, fabsf(x[32 * j + i])); nf |= !isfinite(x[32 * j + i]); }
    const float d = a / 127.0f;
    for (u32 i = 0; i < 32; i++) {
      int q = 0;
      if (!nf && d != 0.0f) { q = (int)roundf(x[32 * j + i] / d); q = q < -127 ? -127 : q > 127 ? 127 : q; }
      qx[32 * j + i] = (signed char)q;
    }
    dx[j] = nf ? NAN : d;
  }
}
static void ref_sm(const u8* q, int j, int* sc, int* m) {
  if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
  else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// thread (g, k)'s share of superblock sb
static float ref_sb(const u8* B, const signed char* qx, const float* dx, u32 n, u32 sb, u32 g, u32 k, int mut) {
  u32 dmw = B[0] | B[1] << 8 | B[2] << 16 | (u32)B[3] << 24;
  int sc0, m0, sc1, m1;
  ref_sm(B + 4, 2 * g, &sc0, &m0); ref_sm(B + 4, 2 * g + 1, &sc1, &m1);
  if (mut == 1) sc0 += 1;
  u32 j0 = 8 * sb + 2 * g;
  u32 c0 = 32 * j0 + (mut == 3 ? 32 : 0);
  int t0 = 0, t1 = 0, s0 = 0, s1 = 0;
  for (u32 h = 0; h < 2; h++) for (u32 b = 0; b < 4; b++) {
    u32 i = 16 * h + 4 * k + b;
    int lo = B[16 + 32 * g + i] & 15, hi = B[16 + 32 * g + i] >> 4;
    if (mut == 2) { int t = lo; lo = hi; hi = t; }
    int u0 = qx[(c0 + i) % n], u1 = qx[(c0 + 32 + i) % n];
    t0 += lo * u0; t1 += hi * u1; s0 += u0; s1 += u1;
  }
  const float e0 = dx[j0], e1 = dx[j0 + 1];
  float A = e0 * (float)(t0 * sc0);
  A = fmaf(e1, (float)(t1 * sc1), A);
  float M = e0 * (float)(s0 * m0);
  M = fmaf(e1, (float)(s1 * m1), M);
  float dmin = ref_f16(dmw >> 16) * M;
  return fmaf(ref_f16(dmw & 0xffff), A, mut == 4 ? dmin : -dmin);
}
// one row: 128 threads' accumulators, warps 1..3 into warp 0, xor butterfly
static float ref_row(const u8* row, u32 nsb, const signed char* qx, const float* dx, u32 n, int mut) {
  float acc[128];
  for (u32 t = 0; t < 128; t++) {
    acc[t] = 0.0f;
    for (u32 sb = t >> 4; sb < nsb; sb += 8) acc[t] = acc[t] + ref_sb(row + 144 * sb, qx, dx, n, sb, (t >> 2) & 3, t & 3, mut);
  }
  float v[32];
  for (u32 l = 0; l < 32; l++) v[l] = ((acc[l] + acc[32 + l]) + acc[64 + l]) + acc[96 + l];
  for (u32 m = 16; m > 0; m >>= 1) {
    float w[32];
    for (u32 l = 0; l < 32; l++) w[l] = v[l] + v[l ^ m];
    memcpy(v, w, sizeof v);
  }
  return v[0];
}
