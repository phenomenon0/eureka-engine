// vitm.metal: ENGINE-VISION3 V7, the Qwen3-VL vision tower's kernels on Metal (host side: vitm.m; op-by-op spec:
// vit.c). One kernel per op. The precision staging and tiling mirror llama.cpp's Metal kernels (fn-run 2690873, MIT),
// i.e. the backend the goldens came from (R1 finding 4: copy the staging, do not "upgrade" it to F32):
//   vt_mm    = kernel_mul_mm_f16_{f32,f16}: tile 64 (weight rows) x 32 (activation rows), NK 32, 4 simdgroups;
//              A = F16 weights, B staged as half, F32 accumulation; + an epilogue (bias, GELU, residual) that
//              evaluates the same float expressions as ggml's separate add / unary / add kernels
//   vt_fa    = kernel_flash_attn_ext_f16_dk{72,64}_dv{72,64} (Q 8, C 64, NSG 4, no mask): Q, K, V half; scores, softmax, O F32.
//              Bidirectional: every query sees every key; a partial last key block is masked with -VT_MAXHALF, as
//              ggml's kv-pad path does
//   vt_ln    = kernel_norm_mul_add_f32_4 (288 threads per 1152-wide row, 256 per 1024-wide row)
//   vt_rope  = kernel_rope_vision (n_dims D/2, sections D/4 + D/4, theta computed in the kernel) + the F16 casts of K and V
// (ENGINE-IPHONEVIS) vt_fa and vt_rope are templates on the head dim D: 72 (the 9B's tower, 1152 / 16) and 64 (the
// 4B's, 1024 / 16), instantiated as vt_fa_dk72 / vt_fa_dk64 and vt_rope_d72 / vt_rope_d64. D = 72 is the same code as
// before (the constants are the template parameter's values), so the 9B tower's bytes do not move.
//   vt_gelu  = kernel_unary op 103 (precise::tanh); vt_add, vt_embed = kernel_bin add; vt_im2col = kernel_im2col_f16
// Compiled at run time with default MTLCompileOptions, as ggml compiles its library.
#include <metal_stdlib>
using namespace metal;

#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define VT_MAXHALF        65504.0f

static inline float vt_gelu1(float x) {
  return 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

// ---------------------------------------------------------------- GEMM: D[n][o] (=|+=) epi(B[n][k] . A[o][k]^T)
struct VtMmArgs { int k, o, n, ldb, ldd, flags; };  // flags: 1 = + bias[o], 2 = GELU, 4 = + D (residual)

template <typename T1, typename T1_2x4>
kernel void vt_mm(constant VtMmArgs & a [[buffer(0)]],
                  device const half * A [[buffer(1)]],
                  device const T1 * B [[buffer(2)]],
                  device float * D [[buffer(3)]],
                  device const float * bias [[buffer(4)]],
                  threadgroup char * shmem [[threadgroup(0)]],
                  uint3 tgpig [[threadgroup_position_in_grid]],
                  ushort tiitg [[thread_index_in_threadgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half * sa = (threadgroup half *)(shmem);
  threadgroup half * sb = (threadgroup half *)(shmem + 4096);
  constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK/16, NL1 = NK/8;
  const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
  const short nr0 = (a.o - r0 < NR0) ? (a.o - r0) : NR0;
  const short nr1 = (a.n - r1 < NR1) ? (a.n - r1) : NR1;
  // a thread does not load outside the matrix (clamped rows are loaded twice, their results dropped)
  const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
  const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;
  const short il0 = tiitg % NL0;
  const bool bc = (a.k % NK) != 0;
  device const half * x = A + (ulong)(r0 + lr0)*a.k + 16*il0;
  const short iy = 8*(tiitg % NL1);
  device const T1 * y = B + (ulong)(r1 + lr1)*a.ldb + iy;

  simdgroup_half8x8 ma[4];
  simdgroup_half8x8 mb[2];
  simdgroup_float8x8 mc[8];
  for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

  for (int loop_k = 0; loop_k < a.k; loop_k += NK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A: 16 halves of weight row lr0 (k 16*il0..), stored as transposed 8x8 blocks
    for (short i = 0; i < 16; i++) {
      const short sx = 2*il0 + i/8, sy = (tiitg/NL0)/8, lx = (tiitg/NL0)%8, ly = i%8;
      const short ib = 8*sx + sy;
      *(sa + 64*ib + 8*ly + lx) = (!bc || loop_k + 16*il0 + i < a.k) ? x[i] : (half)0;
    }
    // B: 8 values of activation row lr1 (k iy..), rounded to half
    {
      const short sx = tiitg % NL1, sy = (tiitg/NL1)/8, ly = (tiitg/NL1)%8;
      const short ib = 4*sx + sy;
      if (bc) {
        for (short i = 0; i < 8; i++) *(sb + 64*ib + 8*ly + i) = loop_k + iy + i < a.k ? (half)y[i] : (half)0;
      } else {
        *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device const T1_2x4 *)y));
      }
    }
    x += NK;
    y += NK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const half * lsma = sa + 4*64*(sgitg%2);
    threadgroup const half * lsmb = sb + 2*64*(sgitg/2);
    for (short ik = 0; ik < NK/8; ik++) {
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
      for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
      lsma += 8*64;
      lsmb += 4*64;
    }
  }

  // epilogue through threadgroup memory: [32 activation rows][64 weight rows] F32
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float * ts = (threadgroup float *)shmem;
  threadgroup float * tsg = ts + 32*(sgitg & 1) + (16*(sgitg >> 1))*NR0;
  for (short i = 0; i < 8; i++) simdgroup_store(mc[i], tsg + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = tiitg; t < NR1*NR0; t += 128) {
    const int j = t / NR0, i = t % NR0;
    if (j < nr1 && i < nr0) {
      device float * d = D + (ulong)(r1 + j)*a.ldd + r0 + i;
      float v = ts[j*NR0 + i];
      if (a.flags & 1) v = v + bias[r0 + i];
      if (a.flags & 2) v = vt_gelu1(v);
      if (a.flags & 4) v = v + *d;
      *d = v;
    }
  }
}

template [[host_name("vt_mm_f32")]] kernel decltype(vt_mm<float, float2x4>) vt_mm<float, float2x4>;
template [[host_name("vt_mm_f16")]] kernel decltype(vt_mm<half, half2x4>) vt_mm<half, half2x4>;

// ---------------------------------------------------------------- bidirectional attention (flash, one head per tg.y)
struct VtFaArgs { int n, npad, ldq, ldo; float scale; };

// q: F32 rows [n][ldq], head h at h*DK; k, v: half [H][npad][DK] (rows >= n zero); dst: F32 [n][ldo]
// PV = DV padded to 64 (ggml's PAD2(DV, 64)): 128 at D 72, 64 at D 64
template <short DK>
kernel void vt_fa(constant VtFaArgs & a [[buffer(0)]],
                  device const float * q [[buffer(1)]],
                  device const half * k [[buffer(2)]],
                  device const half * v [[buffer(3)]],
                  device float * dst [[buffer(4)]],
                  threadgroup half * shmem_f16 [[threadgroup(0)]],
                  uint3 tgpig [[threadgroup_position_in_grid]],
                  ushort tiisg [[thread_index_in_simdgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  constexpr short DV = DK, Q = 8, C = 64, NSG = 4, NW = 32;
  constexpr short DK4 = DK/4, DK8 = DK/8, DV4 = DV/4, PV = (DV + 63)/64*64, PV4 = PV/4, PV8 = PV/8;
  constexpr short NQ = Q/NSG, SH = 2*C, T = DK + 2*PV;
  const int h = tgpig.y;
  const int iq1 = tgpig.x*Q;

  threadgroup half   * sq  = (threadgroup half   *)(shmem_f16);
  threadgroup half4  * sq4 = (threadgroup half4  *)(shmem_f16);
  threadgroup float  * so  = (threadgroup float  *)(shmem_f16 + Q*DK);
  threadgroup float4 * so4 = (threadgroup float4 *)(shmem_f16 + Q*DK);
  threadgroup float  * ss  = (threadgroup float  *)(shmem_f16 + Q*T);
  threadgroup float2 * ss2 = (threadgroup float2 *)(shmem_f16 + Q*T);

  for (short jj = 0; jj < NQ; ++jj) {
    const short j = jj*NSG + sgitg;
    device const float4 * q4 = (device const float4 *)(q + (ulong)(iq1 + j)*a.ldq + h*DK);
    for (short i = tiisg; i < DK4; i += NW) sq4[j*DK4 + i] = (iq1 + j < a.n) ? (half4)q4[i] : (half4)0;
  }
  for (short jj = 0; jj < NQ; ++jj) {
    const short j = jj*NSG + sgitg;
    for (short i = tiisg; i < PV4; i += NW) so4[j*PV4 + i] = 0;  // (ggml zeroes DV4; the pad columns never reach dst)
    for (short i = tiisg; i < SH; i += NW) ss[j*SH + i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float S[NQ] = { [0 ... NQ-1] = 0.0f };
  float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };
  device const half * kh = k + (ulong)h*a.npad*DK;
  device const half * vh = v + (ulong)h*a.npad*DV;

  for (int ic = 0; ic < a.n; ic += C) {
    // S = Q K^T for this block of C keys; simdgroup sg takes keys 8sg.. and 8sg+32..
    {
      device const half * pk = kh + (ulong)ic*DK + sgitg*(8*DK);
      threadgroup const half * pq = sq;
      threadgroup float * ps = ss + sgitg*8;
      for (short cc = 0; cc < (C/8)/NSG; ++cc) {
        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_half8x8 mk;
        simdgroup_half8x8 mq;
        for (short i = 0; i < DK8; ++i) {
          simdgroup_barrier(mem_flags::mem_none);
          simdgroup_load(mk, pk + 8*i, DK, 0, true);
          simdgroup_load(mq, pq + 8*i, DK);
          simdgroup_barrier(mem_flags::mem_none);
          simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, ps, SH, 0, false);
        pk += 8*(NSG*DK);
        ps += 8*NSG;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // online softmax, F32; simdgroup sg owns queries sg and sg+4, 2 keys per lane
    for (short jj = 0; jj < NQ; ++jj) {
      const short j = jj*NSG + sgitg;
      const float m = M[jj];
      float2 s2 = ss2[j*SH/2 + tiisg]*a.scale;
      if (ic + C > a.n) {  // the partial last block: keys >= n get -VT_MAXHALF (ggml's kv-pad mask)
        const int c0 = ic + 2*tiisg;
        s2 += float2(c0 < a.n ? 0.0f : -VT_MAXHALF, c0 + 1 < a.n ? 0.0f : -VT_MAXHALF);
      }
      M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
      const float  ms  = exp(m  - M[jj]);
      const float2 vs2 = exp(s2 - M[jj]);
      S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);
      ss2[j*SH/2 + tiisg] = vs2;
      for (short i = tiisg; i < DV4; i += NW) so4[j*PV4 + i] *= ms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P V; simdgroup sg owns output columns 8sg + 32i
    {
      constexpr short NO = PV8/NSG;
      simdgroup_float8x8 lo[NO];
      threadgroup float * sot = so + 8*sgitg;
      for (short ii = 0; ii < NO; ++ii) { simdgroup_load(lo[ii], sot, PV, 0, false); sot += 8*NSG; }
      device const half * pv = vh + (ulong)ic*DV + 8*sgitg;
      for (short cc = 0; cc < (C/8)/2; ++cc) {
        simdgroup_float8x8 vs[2];
        simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
        simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);
        for (short ii = 0; ii < NO/2; ++ii) {
          simdgroup_half8x8 mv[4];
          simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*DV, DV, 0, false);
          simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*DV, DV, 0, false);
          simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*DV, DV, 0, false);
          simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*DV, DV, 0, false);
          simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
          simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
          simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
          simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
        }
        pv += 2*8*DV;
      }
      sot = so + 8*sgitg;
      for (short ii = 0; ii < NO; ++ii) { simdgroup_store(lo[ii], sot, PV, 0, false); sot += 8*NSG; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (short jj = 0; jj < NQ; ++jj) {
    const short j = jj*NSG + sgitg;
    if (iq1 + j >= a.n) break;
    device float4 * dst4 = (device float4 *)(dst + (ulong)(iq1 + j)*a.ldo + h*DV);
    const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];
    for (short i = tiisg; i < DV4; i += NW) dst4[i] = (float4)so4[j*PV4 + i]*scale;
  }
}

template [[host_name("vt_fa_dk72")]] kernel decltype(vt_fa<72>) vt_fa<72>;
template [[host_name("vt_fa_dk64")]] kernel decltype(vt_fa<64>) vt_fa<64>;

// ---------------------------------------------------------------- LayerNorm with weight and bias (one row per tg)
struct VtLnArgs { int ne00, ne00_t; float eps; };

kernel void vt_ln(constant VtLnArgs & a [[buffer(0)]],
                  device const float4 * src [[buffer(1)]],
                  device const float4 * f0 [[buffer(2)]],
                  device const float4 * f1 [[buffer(3)]],
                  device float4 * dst [[buffer(4)]],
                  threadgroup float * shmem_f32 [[threadgroup(0)]],
                  uint3 tgpig [[threadgroup_position_in_grid]],
                  ushort3 tpitg [[thread_position_in_threadgroup]],
                  ushort sgitg [[simdgroup_index_in_threadgroup]],
                  ushort tiisg [[thread_index_in_simdgroup]],
                  ushort3 ntg [[threads_per_threadgroup]]) {
  if (sgitg == 0) shmem_f32[tiisg] = 0.0f;
  device const float4 * x = src + (ulong)tgpig.x*a.ne00_t;
  float4 sumft(0.0f);
  float sumf = 0.0f;
  for (int i00 = tpitg.x; i00 < a.ne00_t; i00 += ntg.x) sumft += x[i00];
  sumf = dot(sumft, float4(1.0f));
  sumf = simd_sum(sumf);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) shmem_f32[sgitg] = sumf;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sumf = shmem_f32[tiisg];
  sumf = simd_sum(sumf);
  const float mean = sumf/a.ne00;
  device float4 * y = dst + (ulong)tgpig.x*a.ne00_t;
  sumf = 0.0f;
  for (int i00 = tpitg.x; i00 < a.ne00_t; i00 += ntg.x) {
    y[i00] = x[i00] - mean;
    sumf += dot(y[i00], y[i00]);
  }
  sumf = simd_sum(sumf);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) shmem_f32[sgitg] = sumf;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sumf = shmem_f32[tiisg];
  sumf = simd_sum(sumf);
  const float variance = sumf/a.ne00;
  const float scale = 1.0f/sqrt(variance + a.eps);
  for (int i00 = tpitg.x; i00 < a.ne00_t; i00 += ntg.x) y[i00] = (y[i00]*scale)*f0[i00] + f1[i00];
}

// ---------------------------------------------------------------- 2D vision rope on Q and K, + the F16 casts of K, V
struct VtRopeArgs { int n, E, H, npad, dbg; };

// one thread per (patch r, head h, pair ic < D/2): rotates (ic, ic+D/2) of Q and K; Q -> qr F32 [n][E];
// K -> half kh [H][npad][D] (+ F32 kd [n][E] if dbg); V -> half vh [H][npad][D]
template <int D>
kernel void vt_rope(constant VtRopeArgs & a [[buffer(0)]],
                    device const float * qkv [[buffer(1)]],
                    device const int * pos [[buffer(2)]],  // [2][n]: row positions, then column positions
                    device float * qr [[buffer(3)]],
                    device half * kh [[buffer(4)]],
                    device half * vh [[buffer(5)]],
                    device float * kd [[buffer(6)]],
                    uint t [[thread_position_in_grid]]) {
  const int ND = D/2, S0 = D/4;
  if ((int)t >= a.n*a.H*ND) return;
  const int ic = t % ND, h = (t / ND) % a.H, r = t / (ND*a.H);
  const float inv_ndims = -1.f/ND;
  const int sector = ic % ND;
  float p, theta_base;
  if (sector < S0) {
    p = (float)sector;
    theta_base = (float)pos[r];
  } else {
    p = (float)sector - S0;
    theta_base = (float)pos[r + a.n];
  }
  const float theta = theta_base * pow(10000.0f, 2.0f * inv_ndims * p);
  const float cos_theta = cos(theta) * 1.0f, sin_theta = sin(theta) * 1.0f;
  device const float * qs = qkv + (ulong)r*3*a.E + h*D;
  device const float * ks = qs + a.E;
  device const float * vs = qs + 2*a.E;
  device float * qd = qr + (ulong)r*a.E + h*D;
  float x0 = qs[ic], x1 = qs[ic + ND];
  qd[ic]      = x0*cos_theta - x1*sin_theta;
  qd[ic + ND] = x0*sin_theta + x1*cos_theta;
  x0 = ks[ic], x1 = ks[ic + ND];
  const float k0 = x0*cos_theta - x1*sin_theta;
  const float k1 = x0*sin_theta + x1*cos_theta;
  device half * kp = kh + ((ulong)h*a.npad + r)*D;
  device half * vp = vh + ((ulong)h*a.npad + r)*D;
  kp[ic] = (half)k0, kp[ic + ND] = (half)k1;
  vp[ic] = (half)vs[ic], vp[ic + ND] = (half)vs[ic + ND];
  if (a.dbg) {
    device float * kq = kd + (ulong)r*a.E + h*D;
    kq[ic] = k0, kq[ic + ND] = k1;
  }
}

template [[host_name("vt_rope_d72")]] kernel decltype(vt_rope<72>) vt_rope<72>;
template [[host_name("vt_rope_d64")]] kernel decltype(vt_rope<64>) vt_rope<64>;

// ---------------------------------------------------------------- elementwise
kernel void vt_gelu(device float4 * x [[buffer(0)]], constant uint & n4 [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= n4) return;
  const float4 v = x[i];
  x[i] = (float4)(0.5*v*(1 + precise::tanh(SQRT_2_OVER_PI*v*(1 + GELU_COEF_A*v*v))));
}

// y += a
kernel void vt_add(device float4 * y [[buffer(0)]], device const float4 * a [[buffer(1)]], constant uint & n4 [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= n4) return;
  y[i] = y[i] + a[i];
}

// x = (c0 + c1) + bias[e]: the two temporal convs and the patch bias
kernel void vt_embed(device float4 * x [[buffer(0)]], device const float4 * c0 [[buffer(1)]], device const float4 * c1 [[buffer(2)]],
                     device const float4 * b [[buffer(3)]], constant uint2 & ne [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= ne.x) return;
  const float4 s = c0[i] + c1[i];
  x[i] = s + b[i % ne.y];
}

// rows [n][3*16*16] half in 2x2-merge order from planar [3][ny][nx] F32 (ggml's im2col output type = the kernel's F16)
struct VtI2cArgs { int nx, ny, px, n; };
kernel void vt_im2col(constant VtI2cArgs & a [[buffer(0)]], device const float * planar [[buffer(1)]], device half * rows [[buffer(2)]],
                      uint t [[thread_position_in_grid]]) {
  const int K = 768;
  if ((int)t >= a.n*K) return;
  const int r = t / K, kk = t % K;
  const int b = r / 4, dy = (r % 4) / 2, dx = r % 2;
  const int y = 2*(b / (a.px/2)) + dy, x = 2*(b % (a.px/2)) + dx;
  const int ch = kk / 256, ky = (kk / 16) % 16, kx = kk % 16;
  rows[t] = (half)planar[(ulong)ch*a.nx*a.ny + (ulong)(y*16 + ky)*a.nx + x*16 + kx];
}
