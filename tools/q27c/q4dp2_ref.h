// q4dp2_ref.h: the CPU twin of tools/q27c/q4dp2.cuh (K1), on q4dp_ref.h's
// f16 / scale-min reads and integer byte sums. mut 1-4 as ref_sb's; 5 plane 2
// dropped (q2 read as 0) -- each must FAIL K1.
#include "q4dp_ref.h"
static void ref_q8x2(const float* x, signed char* qx, float* dx, u32 n) {
  for (u32 j = 0; j < n / 32; j++) {
    float a = 0.0f; int nf = 0;
    for (u32 i = 0; i < 32; i++) { a = fmaxf(a, fabsf(x[32 * j + i])); nf |= !isfinite(x[32 * j + i]); }
    const float d = a / 127.0f, e = d / 127.0f;
    for (u32 i = 0; i < 32; i++) {
      int q = 0, q2 = 0; float v = x[32 * j + i];
      if (!nf && d != 0.0f) { q = (int)roundf(v / d); q = q < -127 ? -127 : q > 127 ? 127 : q; }
      if (!nf && e != 0.0f) { q2 = (int)roundf(fmaf(-d, (float)q, v) / e); q2 = q2 < -127 ? -127 : q2 > 127 ? 127 : q2; }
      qx[32 * j + i] = (signed char)q; qx[n + 32 * j + i] = (signed char)q2;
    }
    dx[j] = nf ? NAN : e;
  }
}
static float ref_sb2(const u8* B, const signed char* qx, const float* dx, u32 n, u32 sb, u32 g, u32 k, int mut) {
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
    int u0 = 127 * qx[(c0 + i) % n] + (mut == 5 ? 0 : qx[n + (c0 + i) % n]);
    int u1 = 127 * qx[(c0 + 32 + i) % n] + (mut == 5 ? 0 : qx[n + (c0 + 32 + i) % n]);
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
static float ref_row2(const u8* row, u32 nsb, const signed char* qx, const float* dx, u32 n, int mut) {
  float acc[128];
  for (u32 t = 0; t < 128; t++) {
    acc[t] = 0.0f;
    for (u32 sb = t >> 4; sb < nsb; sb += 8) acc[t] = acc[t] + ref_sb2(row + 144 * sb, qx, dx, n, sb, (t >> 2) & 3, t & 3, mut);
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
// q8dp2.cuh's row (c cols, rb = c / 16 * 17); mut 1 scale + 1 ulp, 2 a block's e
// from the next block, 3 plane 2 dropped, 4 int8 weight sign read as u8
static float ref_q8row2(const u8* row, u32 c, const signed char* qx, const float* dx, int mut) {
  float acc[128];
  for (u32 t = 0; t < 128; t++) {
    acc[t] = 0.0f;
    for (u32 o = 16 * t; o < c; o += 2048) {
      int a = 0, b = 0;
      for (u32 i = 0; i < 16; i++) {
        int wv = mut == 4 ? (int)row[o + i] : (int)(signed char)row[o + i];
        a += wv * qx[o + i]; b += mut == 3 ? 0 : wv * qx[c + o + i];
      }
      float s = ref_f16(row[c + 2 * (o >> 5)] | row[c + 2 * (o >> 5) + 1] << 8);
      if (mut == 1) s = nextafterf(s, INFINITY);
      float e = dx[((o >> 5) + (mut == 2)) % (c / 32)];
      acc[t] = fmaf(s * e, (float)(127 * a + b), acc[t]);
    }
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
// q6dp2.cuh's row (c cols, c / 256 Q6_K blocks); mut 1 sc + 1, 2 quads 1 / 2's qh
// shifts swapped, 3 plane 2 dropped, 4 the - 32 dropped
static float ref_q6row2(const u8* row, u32 c, const signed char* qx, const float* dx, int mut) {
  float v[32];
  for (u32 j = 0; j < 32; j++) {
    float acc = 0.0f;
    for (u32 u = j; u < c / 32; u += 32) {
      u32 b = u >> 3, hn = (u >> 2) & 1, l = 8 * (u & 3);
      const u8* B = row + 210 * b;
      float d = ref_f16(B[208] | B[209] << 8);
      for (u32 q = 0; q < 4; q++) {
        u32 xb = 8 * b + 4 * hn + q; int a = 0, s2 = 0;
        for (u32 i = 0; i < 8; i++) {
          u8 L = B[64 * hn + (q & 1) * 32 + l + i], H = B[128 + 32 * hn + l + i];
          u32 qs = mut == 2 && (q == 1 || q == 2) ? 3 - q : q;
          int w = ((q >= 2 ? L >> 4 : L) & 15 | ((H >> (2 * qs)) & 3) << 4) - (mut == 4 ? 0 : 32);
          a += w * qx[32 * xb + l + i]; s2 += mut == 3 ? 0 : w * qx[c + 32 * xb + l + i];
        }
        float sc = (float)((signed char)B[192 + 8 * hn + 2 * q + (l >> 4)] + (mut == 1));
        acc = fmaf((d * sc) * dx[xb], (float)(127 * a + s2), acc);
      }
    }
    v[j] = acc;
  }
  for (u32 m = 16; m > 0; m >>= 1) {
    float w[32];
    for (u32 l = 0; l < 32; l++) w[l] = v[l] + v[l ^ m];
    memcpy(v, w, sizeof v);
  }
  return v[0];
}
