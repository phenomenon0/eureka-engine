// ec_creative.metal — Universal Apple Silicon (M1-M5) + Intel/AMD Mac Metal Compute Library
// Runtime GPU-family dispatch in ec_backend_metal.mm selects the optimal tier automatically:
//   Tier 1 (MTLGPUFamilyApple9 — M3/M4/M5): Native `simdgroup_matrix<bfloat, 8, 8>` -> `float8x8`
//   Tier 2 (MTLGPUFamilyApple7 — M1/M2):    Native `simdgroup_matrix<half, 8, 8>` -> `float8x8`
//   Tier 3 (Intel UHD/Iris & AMD Mac):      Portable 16x16 threadgroup-tiled FP32 + width-agnostic 64T FA2
// Zero external ML frameworks. Zero simdgroup_matrix assumptions on Intel/AMD Macs.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 230)
  #define EC_HAS_SIMDGROUP_MATRIX 1
  #include <metal_simdgroup_matrix>
#else
  #define EC_HAS_SIMDGROUP_MATRIX 0
#endif

inline float bf16_to_f32(ushort b) {
  return as_type<float>(uint(b) << 16);
}

inline ushort f32_to_bf16(float f) {
  uint u = as_type<uint>(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return ushort(u >> 16);
}

inline half bf16_to_f16_safe(ushort b) {
  return half(clamp(bf16_to_f32(b), -65504.0f, 65504.0f));
}

inline half f32_to_f16_safe(float f) {
  return half(clamp(f, -65504.0f, 65504.0f));
}

inline float ec_silu(float x) {
  return x / (1.0f + exp(-x));
}

inline float ec_gelu(float x) {
  float c = 0.7978845608f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + precise::tanh(clamp(c, -12.0f, 12.0f)));
}

struct EcGemmArgs {
  uint M, N, K;
  uint epilogue; // 0=none, 1=bias, 2=bias+SiLU, 3=bias+GELU, 4=bias+res
  uint has_bias, has_res, write_f32, write_16;
};

struct EcFa2Args {
  uint  Lq, Lkv, Lq_pad, Lkv_pad, Hq, Hkv, D;
  int   window;
  uint  block_causal, write_f32, write_16;
  float scale;
};

struct EcConv3x3Args {
  uint H, W, Cin, Cout, has_bias, has_res, write_f32, write_bf16;
};

// ===========================================================================
// TIER 1 & TIER 2: Apple Silicon Hardware `simdgroup_matrix` Kernels
// ===========================================================================
#if EC_HAS_SIMDGROUP_MATRIX

#if defined(__HAVE_BFLOAT__) && (__METAL_VERSION__ >= 310)
// Tier 1 (M3/M4/M5 Apple9): Native hardware bfloat8x8 -> float8x8 BF16 GEMM
kernel void k_metal_gemm_bf16_bf(
    device const ushort *X       [[buffer(0)]],
    device const ushort *W       [[buffer(1)]],
    device const float  *bias    [[buffer(2)]],
    device float        *Y_f32   [[buffer(3)]],
    device ushort       *Y_bf16  [[buffer(4)]],
    constant EcGemmArgs &args    [[buffer(5)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat sA[32 * 16];
  threadgroup bfloat sB[16 * 32];
  threadgroup float  sOut[32 * 32];

  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;

  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);

  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K)
                            ? as_type<bfloat>(X[(ulong)(m0 + r) * args.K + (k0 + c)]) : bfloat(0.0f);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K)
                            ? as_type<bfloat>(W[(ulong)(n0 + r) * args.K + (k0 + c)]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_matrix<bfloat, 8, 8> ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i];
      if (args.has_bias) v += bias[c];
      ulong off = ulong(r) * args.N + c;
      if (args.write_f32) Y_f32[off] = v;
      if (args.write_16)  Y_bf16[off] = f32_to_bf16(v);
    }
  }
}

// Tier 1 (M3/M4/M5 Apple9): W8A16 fused threadgroup-dequant bfloat8x8 Q8_G64 GEMM
kernel void k_metal_gemm_q8g64_bf(
    device const char   *X_q8    [[buffer(0)]],
    device const float  *x_sc    [[buffer(1)]],
    device const char   *W_q8    [[buffer(2)]],
    device const float  *w_sc    [[buffer(3)]],
    device const float  *bias    [[buffer(4)]],
    device float        *Y_f32   [[buffer(5)]],
    device ushort       *Y_bf16  [[buffer(6)]],
    constant EcGemmArgs &args    [[buffer(7)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat sA[32 * 16];
  threadgroup bfloat sB[16 * 32];
  threadgroup float  sOut[32 * 32];
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  const uint groups = args.K >> 6;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    const uint g = k0 >> 6;
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K)
          ? bfloat(float(X_q8[(ulong)(m0 + r) * args.K + (k0 + c)]) * x_sc[(ulong)(m0 + r) * groups + g]) : bfloat(0.0f);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K)
          ? bfloat(float(W_q8[(ulong)(n0 + r) * args.K + (k0 + c)]) * w_sc[(ulong)(n0 + r) * groups + g]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_matrix<bfloat, 8, 8> ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i];
      if (args.has_bias) v += bias[c];
      ulong off = ulong(r) * args.N + c;
      if (args.write_f32) Y_f32[off] = v;
      if (args.write_16)  Y_bf16[off] = f32_to_bf16(v);
    }
  }
}

// Tier 1 (M3/M4/M5 Apple9): Implicit-GEMM NHWC 3x3 Conv2D via bfloat8x8
kernel void k_metal_conv2d_3x3_nhwc_bf16_bf(
    device const ushort      *X_nhwc [[buffer(0)]],
    device const ushort      *W_ohwi [[buffer(1)]],
    device const float       *bias   [[buffer(2)]],
    device const float       *res    [[buffer(3)]],
    device float             *Y_f32  [[buffer(4)]],
    device ushort            *Y_bf16 [[buffer(5)]],
    constant EcConv3x3Args   &args   [[buffer(6)]],
    uint3 tg_id                      [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg_id                      [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat sA[32 * 16];
  threadgroup bfloat sB[16 * 32];
  threadgroup float  sOut[32 * 32];
  const uint M = args.H * args.W, K = 9u * args.Cin;
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u, m = m0 + r, kk = k0 + c;
      bfloat xv = bfloat(0.0f);
      if (m < M && kk < K) {
        uint tap = kk / args.Cin, ic = kk % args.Cin;
        int oy = int(m / args.W), ox = int(m % args.W);
        int iy = oy + int(tap / 3u) - 1, ix = ox + int(tap % 3u) - 1;
        if (iy >= 0 && iy < int(args.H) && ix >= 0 && ix < int(args.W))
          xv = as_type<bfloat>(X_nhwc[(ulong(iy * int(args.W) + ix) * args.Cin) + ic]);
      }
      sA[r * 16u + c] = xv;
      sB[c * 32u + r] = (n0 + r < args.Cout && kk < K) ? as_type<bfloat>(W_ohwi[(ulong)(n0 + r) * K + kk]) : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_matrix<bfloat, 8, 8> ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint m = m0 + (i >> 5), oc = n0 + (i & 31u);
    if (m < M && oc < args.Cout) {
      ulong off = ulong(m) * args.Cout + oc;
      float v = sOut[i];
      if (args.has_bias) v += bias[oc];
      if (args.has_res)  v += res[off];
      if (args.write_f32)  Y_f32[off]  = v;
      if (args.write_bf16) Y_bf16[off] = f32_to_bf16(v);
    }
  }
}
#endif // __HAVE_BFLOAT__

// Tier 2 (M1/M2 Apple7/8 — also safe fallback on M3/M4): half8x8 -> float8x8 BF16 GEMM
kernel void k_metal_gemm_bf16_f16(
    device const ushort *X       [[buffer(0)]],
    device const ushort *W       [[buffer(1)]],
    device const float  *bias    [[buffer(2)]],
    device float        *Y_f32   [[buffer(3)]],
    device ushort       *Y_bf16  [[buffer(4)]],
    constant EcGemmArgs &args    [[buffer(5)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup half  sA[32 * 16];
  threadgroup half  sB[16 * 32];
  threadgroup float sOut[32 * 32];
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K) ? bf16_to_f16_safe(X[(ulong)(m0 + r) * args.K + (k0 + c)]) : half(0.0h);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K) ? bf16_to_f16_safe(W[(ulong)(n0 + r) * args.K + (k0 + c)]) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i];
      if (args.has_bias) v += bias[c];
      ulong off = ulong(r) * args.N + c;
      if (args.write_f32) Y_f32[off] = v;
      if (args.write_16)  Y_bf16[off] = f32_to_bf16(v);
    }
  }
}

// Tier 1/2 (M1-M5): Hardware simdgroup_half8x8 FP16 GEMM + Epilogue
kernel void k_metal_gemm_f16(
    device const half   *X       [[buffer(0)]],
    device const half   *W       [[buffer(1)]],
    device const float  *bias    [[buffer(2)]],
    device const float  *res     [[buffer(3)]],
    device float        *Y_f32   [[buffer(4)]],
    device half         *Y_f16   [[buffer(5)]],
    constant EcGemmArgs &args    [[buffer(6)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup half  sA[32 * 16];
  threadgroup half  sB[16 * 32];
  threadgroup float sOut[32 * 32];
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K) ? X[(ulong)(m0 + r) * args.K + (k0 + c)] : half(0.0h);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K) ? W[(ulong)(n0 + r) * args.K + (k0 + c)] : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i];
      ulong off = ulong(r) * args.N + c;
      if (args.has_bias && args.epilogue >= 1u) v += bias[c];
      if (args.epilogue == 2u) v = ec_silu(v);
      else if (args.epilogue == 3u) v = ec_gelu(v);
      else if (args.epilogue == 4u && args.has_res) v += res[off];
      if (args.write_f32) Y_f32[off] = v;
      if (args.write_16)  Y_f16[off] = f32_to_f16_safe(v);
    }
  }
}

// Tier 2 (M1/M2): half8x8 Q8_G64 GEMM
kernel void k_metal_gemm_q8g64_f16(
    device const char   *X_q8    [[buffer(0)]],
    device const float  *x_sc    [[buffer(1)]],
    device const char   *W_q8    [[buffer(2)]],
    device const float  *w_sc    [[buffer(3)]],
    device const float  *bias    [[buffer(4)]],
    device float        *Y_f32   [[buffer(5)]],
    device ushort       *Y_bf16  [[buffer(6)]],
    constant EcGemmArgs &args    [[buffer(7)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup half  sA[32 * 16];
  threadgroup half  sB[16 * 32];
  threadgroup float sOut[32 * 32];
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  const uint groups = args.K >> 6;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    const uint g = k0 >> 6;
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K)
          ? f32_to_f16_safe(float(X_q8[(ulong)(m0 + r) * args.K + (k0 + c)]) * x_sc[(ulong)(m0 + r) * groups + g]) : half(0.0h);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K)
          ? f32_to_f16_safe(float(W_q8[(ulong)(n0 + r) * args.K + (k0 + c)]) * w_sc[(ulong)(n0 + r) * groups + g]) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i];
      if (args.has_bias) v += bias[c];
      ulong off = ulong(r) * args.N + c;
      if (args.write_f32) Y_f32[off] = v;
      if (args.write_16)  Y_bf16[off] = f32_to_bf16(v);
    }
  }
}

// Tier 1/2 (M1-M5): Hardware simdgroup_half8x8 INT8 W8A8 Per-Row/Col Scaled GEMM (exact in half since |int8|<=127!)
kernel void k_metal_gemm_i8_f16(
    device const char   *X_i8    [[buffer(0)]],
    device const float  *x_sc    [[buffer(1)]],
    device const char   *W_i8    [[buffer(2)]],
    device const float  *w_sc    [[buffer(3)]],
    device const float  *bias    [[buffer(4)]],
    device float        *Y_f32   [[buffer(5)]],
    constant EcGemmArgs &args    [[buffer(6)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint   tid                   [[thread_index_in_threadgroup]],
    uint   sg_id                 [[simdgroup_index_in_threadgroup]]) {
  threadgroup half  sA[32 * 16];
  threadgroup half  sB[16 * 32];
  threadgroup float sOut[32 * 32];
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < args.K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u;
      sA[r * 16u + c] = (m0 + r < args.M && k0 + c < args.K) ? half(int(X_i8[(ulong)(m0 + r) * args.K + (k0 + c)])) : half(0.0h);
      sB[c * 32u + r] = (n0 + r < args.N && k0 + c < args.K) ? half(int(W_i8[(ulong)(n0 + r) * args.K + (k0 + c)])) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint r = m0 + (i >> 5), c = n0 + (i & 31u);
    if (r < args.M && c < args.N) {
      float v = sOut[i] * (x_sc[r] * w_sc[c]);
      if (args.has_bias) v += bias[c];
      Y_f32[ulong(r) * args.N + c] = v;
    }
  }
}

// Tier 2 (M1/M2): Implicit-GEMM NHWC 3x3 Conv2D via half8x8
kernel void k_metal_conv2d_3x3_nhwc_bf16_f16(
    device const ushort      *X_nhwc [[buffer(0)]],
    device const ushort      *W_ohwi [[buffer(1)]],
    device const float       *bias   [[buffer(2)]],
    device const float       *res    [[buffer(3)]],
    device float             *Y_f32  [[buffer(4)]],
    device ushort            *Y_bf16 [[buffer(5)]],
    constant EcConv3x3Args   &args   [[buffer(6)]],
    uint3 tg_id                      [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg_id                      [[simdgroup_index_in_threadgroup]]) {
  threadgroup half  sA[32 * 16];
  threadgroup half  sB[16 * 32];
  threadgroup float sOut[32 * 32];
  const uint M = args.H * args.W, K = 9u * args.Cin;
  const uint m0 = tg_id.y * 32u, n0 = tg_id.x * 32u;
  const uint sg_m = (sg_id >> 1) * 16u, sg_n = (sg_id & 1u) * 16u;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < K; k0 += 16u) {
    for (uint i = tid; i < 512u; i += 128u) {
      uint r = i >> 4, c = i & 15u, m = m0 + r, kk = k0 + c;
      half xv = half(0.0h);
      if (m < M && kk < K) {
        uint tap = kk / args.Cin, ic = kk % args.Cin;
        int oy = int(m / args.W), ox = int(m % args.W);
        int iy = oy + int(tap / 3u) - 1, ix = ox + int(tap % 3u) - 1;
        if (iy >= 0 && iy < int(args.H) && ix >= 0 && ix < int(args.W))
          xv = bf16_to_f16_safe(X_nhwc[(ulong(iy * int(args.W) + ix) * args.Cin) + ic]);
      }
      sA[r * 16u + c] = xv;
      sB[c * 32u + r] = (n0 + r < args.Cout && kk < K) ? bf16_to_f16_safe(W_ohwi[(ulong)(n0 + r) * K + kk]) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, sA + (sg_m + 0u) * 16u + kk, 16u);
      simdgroup_load(ma1, sA + (sg_m + 8u) * 16u + kk, 16u);
      simdgroup_load(mb0, sB + kk * 32u + sg_n + 0u, 32u);
      simdgroup_load(mb1, sB + kk * 32u + sg_n + 8u, 32u);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++)
      simdgroup_store(acc[i][j], sOut + (sg_m + i * 8u) * 32u + (sg_n + j * 8u), 32u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 1024u; i += 128u) {
    uint m = m0 + (i >> 5), oc = n0 + (i & 31u);
    if (m < M && oc < args.Cout) {
      ulong off = ulong(m) * args.Cout + oc;
      float v = sOut[i];
      if (args.has_bias) v += bias[oc];
      if (args.has_res)  v += res[off];
      if (args.write_f32)  Y_f32[off]  = v;
      if (args.write_bf16) Y_bf16[off] = f32_to_bf16(v);
    }
  }
}

// Tier 1/2 (Apple Silicon 32-wide simdgroup): Quantize BF16 -> Q8_G64
kernel void k_metal_quantize_bf16_q8g64(
    device const ushort *src_bf16 [[buffer(0)]],
    device char         *dst_q8   [[buffer(1)]],
    device float        *dst_sc   [[buffer(2)]],
    constant uint       &n_groups [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]]) {
  uint g = gid >> 5;
  if (g >= n_groups) return;
  ulong base = ulong(g) * 64ul;
  float v0 = bf16_to_f32(src_bf16[base + lane * 2u]);
  float v1 = bf16_to_f32(src_bf16[base + lane * 2u + 1u]);
  float amax = simd_max(max(abs(v0), abs(v1)));
  float sc  = amax / 127.0f;
  float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
  dst_q8[base + lane * 2u]      = char(clamp(int(rint(v0 * inv)), -127, 127));
  dst_q8[base + lane * 2u + 1u] = char(clamp(int(rint(v1 * inv)), -127, 127));
  if (lane == 0u) dst_sc[g] = sc;
}

// Tier 1/2 (Apple Silicon 32-wide simdgroup): FlashAttention-2 BF16
kernel void k_metal_fa2_bf16(
    device const ushort *Q        [[buffer(0)]],
    device const ushort *K        [[buffer(1)]],
    device const ushort *V        [[buffer(2)]],
    device float        *O_f32    [[buffer(3)]],
    device ushort       *O_bf16   [[buffer(4)]],
    constant EcFa2Args  &args     [[buffer(5)]],
    uint gid                      [[thread_position_in_grid]],
    uint lane                     [[thread_index_in_simdgroup]]) {
  uint pair = gid >> 5;
  if (pair >= args.Lq * args.Hq) return;
  uint qpos = pair / args.Hq, hq = pair % args.Hq, hkv = hq / (args.Hq / args.Hkv);
  device const ushort *qp = Q + (ulong(hq) * args.Lq_pad + qpos) * args.D;
  float qr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint iters = args.D >> 5;
  for (uint i = 0; i < iters; i++) qr[i] = bf16_to_f32(qp[lane + i * 32u]);
  uint k_start = 0u, k_end = args.Lkv;
  if (args.window > 0) {
    k_start = uint(max(0, int(qpos) - args.window));
    k_end   = uint(min(int(args.Lkv), int(qpos) + args.window + 1));
  }
  float m_max = -1e30f, l_sum = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  device const ushort *kb = K + ulong(hkv) * args.Lkv_pad * args.D;
  device const ushort *vb = V + ulong(hkv) * args.Lkv_pad * args.D;
  for (uint kpos = k_start; kpos < k_end; kpos++) {
    device const ushort *kp = kb + ulong(kpos) * args.D;
    float dot = 0.0f;
    for (uint i = 0; i < iters; i++) dot = fma(qr[i], bf16_to_f32(kp[lane + i * 32u]), dot);
    float sc = simd_sum(dot) * args.scale;
    float nm = max(m_max, sc);
    float al = exp(m_max - nm), p = exp(sc - nm);
    l_sum = l_sum * al + p; m_max = nm;
    device const ushort *vp = vb + ulong(kpos) * args.D;
    for (uint i = 0; i < iters; i++) acc[i] = acc[i] * al + p * bf16_to_f32(vp[lane + i * 32u]);
  }
  float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  ulong o_off = (ulong(qpos) * args.Hq + hq) * args.D;
  for (uint i = 0; i < iters; i++) {
    float v = acc[i] * inv;
    if (args.write_f32) O_f32[o_off + lane + i * 32u]  = v;
    if (args.write_16)  O_bf16[o_off + lane + i * 32u] = f32_to_bf16(v);
  }
}

// Tier 1/2 (Apple Silicon 32-wide simdgroup): FlashAttention-2 FP16
kernel void k_metal_fa2_f16(
    device const half  *Q        [[buffer(0)]],
    device const half  *K        [[buffer(1)]],
    device const half  *V        [[buffer(2)]],
    device float       *O_f32    [[buffer(3)]],
    device half        *O_f16    [[buffer(4)]],
    constant EcFa2Args &args     [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]]) {
  uint pair = gid >> 5;
  if (pair >= args.Lq * args.Hq) return;
  uint qpos = pair / args.Hq, h = pair % args.Hq;
  device const half *qp = Q + (ulong(h) * args.Lq_pad + qpos) * args.D;
  float qr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint iters = args.D >> 5;
  for (uint i = 0; i < iters; i++) qr[i] = float(qp[lane + i * 32u]);
  uint k_end = args.Lkv;
  if (args.block_causal > 0u) k_end = min(args.Lkv, ((qpos / args.block_causal) + 1u) * args.block_causal);
  float m_max = -1e30f, l_sum = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  device const half *kb = K + ulong(h) * args.Lkv_pad * args.D;
  device const half *vb = V + ulong(h) * args.Lkv_pad * args.D;
  for (uint kpos = 0; kpos < k_end; kpos++) {
    device const half *kp = kb + ulong(kpos) * args.D;
    float dot = 0.0f;
    for (uint i = 0; i < iters; i++) dot = fma(qr[i], float(kp[lane + i * 32u]), dot);
    float sc = simd_sum(dot) * args.scale;
    float nm = max(m_max, sc);
    float al = exp(m_max - nm), p = exp(sc - nm);
    l_sum = l_sum * al + p; m_max = nm;
    device const half *vp = vb + ulong(kpos) * args.D;
    for (uint i = 0; i < iters; i++) acc[i] = acc[i] * al + p * float(vp[lane + i * 32u]);
  }
  float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  ulong o_off = (ulong(qpos) * args.Hq + h) * args.D;
  for (uint i = 0; i < iters; i++) {
    float v = acc[i] * inv;
    if (args.write_f32) O_f32[o_off + lane + i * 32u] = v;
    if (args.write_16)  O_f16[o_off + lane + i * 32u] = f32_to_f16_safe(v);
  }
}
#endif // EC_HAS_SIMDGROUP_MATRIX

// ===========================================================================
// TIER 3: Universal Intel Mac / AMD Mac Fallback (ZERO simdgroup_matrix, ZERO simd_width=32 assumptions)
// ===========================================================================
kernel void k_metal_gemm_bf16_tiled(
    device const ushort *X       [[buffer(0)]],
    device const ushort *W       [[buffer(1)]],
    device const float  *bias    [[buffer(2)]],
    device float        *Y_f32   [[buffer(3)]],
    device ushort       *Y_bf16  [[buffer(4)]],
    constant EcGemmArgs &args    [[buffer(5)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint3  lid                   [[thread_position_in_threadgroup]]) {
  threadgroup float sA[16 * 32];
  threadgroup float sB[16 * 32];
  uint row = tg_id.y * 16u + lid.y, col = tg_id.x * 16u + lid.x;
  float acc = 0.0f;
  for (uint k0 = 0; k0 < args.K; k0 += 32u) {
    uint kk = k0 + lid.x * 2u;
    sA[lid.y * 32u + lid.x * 2u]      = (row < args.M && kk < args.K) ? bf16_to_f32(X[(ulong)row * args.K + kk]) : 0.0f;
    sA[lid.y * 32u + lid.x * 2u + 1u] = (row < args.M && kk + 1u < args.K) ? bf16_to_f32(X[(ulong)row * args.K + kk + 1u]) : 0.0f;
    uint brow = tg_id.x * 16u + lid.y;
    sB[lid.y * 32u + lid.x * 2u]      = (brow < args.N && kk < args.K) ? bf16_to_f32(W[(ulong)brow * args.K + kk]) : 0.0f;
    sB[lid.y * 32u + lid.x * 2u + 1u] = (brow < args.N && kk + 1u < args.K) ? bf16_to_f32(W[(ulong)brow * args.K + kk + 1u]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32u; k++) acc = fma(sA[lid.y * 32u + k], sB[lid.x * 32u + k], acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < args.M && col < args.N) {
    if (args.has_bias) acc += bias[col];
    ulong off = ulong(row) * args.N + col;
    if (args.write_f32) Y_f32[off] = acc;
    if (args.write_16)  Y_bf16[off] = f32_to_bf16(acc);
  }
}

kernel void k_metal_gemm_f16_tiled(
    device const half   *X       [[buffer(0)]],
    device const half   *W       [[buffer(1)]],
    device const float  *bias    [[buffer(2)]],
    device const float  *res     [[buffer(3)]],
    device float        *Y_f32   [[buffer(4)]],
    device half         *Y_f16   [[buffer(5)]],
    constant EcGemmArgs &args    [[buffer(6)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint3  lid                   [[thread_position_in_threadgroup]]) {
  threadgroup float sA[16 * 32];
  threadgroup float sB[16 * 32];
  uint row = tg_id.y * 16u + lid.y, col = tg_id.x * 16u + lid.x;
  float acc = 0.0f;
  for (uint k0 = 0; k0 < args.K; k0 += 32u) {
    uint kk = k0 + lid.x * 2u;
    sA[lid.y * 32u + lid.x * 2u]      = (row < args.M && kk < args.K) ? float(X[(ulong)row * args.K + kk]) : 0.0f;
    sA[lid.y * 32u + lid.x * 2u + 1u] = (row < args.M && kk + 1u < args.K) ? float(X[(ulong)row * args.K + kk + 1u]) : 0.0f;
    uint brow = tg_id.x * 16u + lid.y;
    sB[lid.y * 32u + lid.x * 2u]      = (brow < args.N && kk < args.K) ? float(W[(ulong)brow * args.K + kk]) : 0.0f;
    sB[lid.y * 32u + lid.x * 2u + 1u] = (brow < args.N && kk + 1u < args.K) ? float(W[(ulong)brow * args.K + kk + 1u]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32u; k++) acc = fma(sA[lid.y * 32u + k], sB[lid.x * 32u + k], acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < args.M && col < args.N) {
    ulong off = ulong(row) * args.N + col;
    if (args.has_bias && args.epilogue >= 1u) acc += bias[col];
    if (args.epilogue == 2u) acc = ec_silu(acc);
    else if (args.epilogue == 3u) acc = ec_gelu(acc);
    else if (args.epilogue == 4u && args.has_res) acc += res[off];
    if (args.write_f32) Y_f32[off] = acc;
    if (args.write_16)  Y_f16[off] = f32_to_f16_safe(acc);
  }
}

kernel void k_metal_gemm_q8g64_tiled(
    device const char   *X_q8    [[buffer(0)]],
    device const float  *x_sc    [[buffer(1)]],
    device const char   *W_q8    [[buffer(2)]],
    device const float  *w_sc    [[buffer(3)]],
    device const float  *bias    [[buffer(4)]],
    device float        *Y_f32   [[buffer(5)]],
    device ushort       *Y_bf16  [[buffer(6)]],
    constant EcGemmArgs &args    [[buffer(7)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint3  lid                   [[thread_position_in_threadgroup]]) {
  threadgroup float sA[16 * 32];
  threadgroup float sB[16 * 32];
  uint row = tg_id.y * 16u + lid.y, col = tg_id.x * 16u + lid.x;
  uint groups = args.K >> 6;
  float acc = 0.0f;
  for (uint g = 0; g < groups; g++) {
    for (uint h = 0; h < 2u; h++) {
      uint kk = g * 64u + h * 32u + lid.x * 2u;
      if (row < args.M) {
        float xs = x_sc[(ulong)row * groups + g];
        sA[lid.y * 32u + lid.x * 2u]      = float(X_q8[(ulong)row * args.K + kk]) * xs;
        sA[lid.y * 32u + lid.x * 2u + 1u] = float(X_q8[(ulong)row * args.K + kk + 1u]) * xs;
      } else { sA[lid.y * 32u + lid.x * 2u] = 0.0f; sA[lid.y * 32u + lid.x * 2u + 1u] = 0.0f; }
      uint brow = tg_id.x * 16u + lid.y;
      if (brow < args.N) {
        float ws = w_sc[(ulong)brow * groups + g];
        sB[lid.y * 32u + lid.x * 2u]      = float(W_q8[(ulong)brow * args.K + kk]) * ws;
        sB[lid.y * 32u + lid.x * 2u + 1u] = float(W_q8[(ulong)brow * args.K + kk + 1u]) * ws;
      } else { sB[lid.y * 32u + lid.x * 2u] = 0.0f; sB[lid.y * 32u + lid.x * 2u + 1u] = 0.0f; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 32u; k++) acc = fma(sA[lid.y * 32u + k], sB[lid.x * 32u + k], acc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (row < args.M && col < args.N) {
    if (args.has_bias) acc += bias[col];
    ulong off = ulong(row) * args.N + col;
    if (args.write_f32) Y_f32[off] = acc;
    if (args.write_16)  Y_bf16[off] = f32_to_bf16(acc);
  }
}

kernel void k_metal_gemm_i8_tiled(
    device const char   *X_i8    [[buffer(0)]],
    device const float  *x_sc    [[buffer(1)]],
    device const char   *W_i8    [[buffer(2)]],
    device const float  *w_sc    [[buffer(3)]],
    device const float  *bias    [[buffer(4)]],
    device float        *Y_f32   [[buffer(5)]],
    constant EcGemmArgs &args    [[buffer(6)]],
    uint3  tg_id                 [[threadgroup_position_in_grid]],
    uint3  lid                   [[thread_position_in_threadgroup]]) {
  threadgroup float sA[16 * 32];
  threadgroup float sB[16 * 32];
  uint row = tg_id.y * 16u + lid.y, col = tg_id.x * 16u + lid.x;
  float acc = 0.0f;
  for (uint k0 = 0; k0 < args.K; k0 += 32u) {
    uint kk = k0 + lid.x * 2u;
    sA[lid.y * 32u + lid.x * 2u]      = (row < args.M && kk < args.K) ? float(X_i8[(ulong)row * args.K + kk]) : 0.0f;
    sA[lid.y * 32u + lid.x * 2u + 1u] = (row < args.M && kk + 1u < args.K) ? float(X_i8[(ulong)row * args.K + kk + 1u]) : 0.0f;
    uint brow = tg_id.x * 16u + lid.y;
    sB[lid.y * 32u + lid.x * 2u]      = (brow < args.N && kk < args.K) ? float(W_i8[(ulong)brow * args.K + kk]) : 0.0f;
    sB[lid.y * 32u + lid.x * 2u + 1u] = (brow < args.N && kk + 1u < args.K) ? float(W_i8[(ulong)brow * args.K + kk + 1u]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32u; k++) acc = fma(sA[lid.y * 32u + k], sB[lid.x * 32u + k], acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < args.M && col < args.N) {
    float v = acc * (x_sc[row] * w_sc[col]);
    if (args.has_bias) v += bias[col];
    Y_f32[ulong(row) * args.N + col] = v;
  }
}

kernel void k_metal_conv2d_3x3_nhwc_bf16_tiled(
    device const ushort      *X_nhwc [[buffer(0)]],
    device const ushort      *W_ohwi [[buffer(1)]],
    device const float       *bias   [[buffer(2)]],
    device const float       *res    [[buffer(3)]],
    device float             *Y_f32  [[buffer(4)]],
    device ushort            *Y_bf16 [[buffer(5)]],
    constant EcConv3x3Args   &args   [[buffer(6)]],
    uint3 tg_id                      [[threadgroup_position_in_grid]],
    uint3 lid                        [[thread_position_in_threadgroup]]) {
  threadgroup float sA[16 * 32];
  threadgroup float sB[16 * 32];
  uint M = args.H * args.W, K = 9u * args.Cin;
  uint m = tg_id.y * 16u + lid.y, oc = tg_id.x * 16u + lid.x;
  int oy = int(m / args.W), ox = int(m % args.W);
  float acc = 0.0f;
  for (uint k0 = 0; k0 < K; k0 += 32u) {
    uint kk0 = k0 + lid.x * 2u;
    for (uint s = 0; s < 2u; s++) {
      uint kk = kk0 + s;
      float xv = 0.0f;
      if (m < M && kk < K) {
        uint tap = kk / args.Cin, ic = kk % args.Cin;
        int iy = oy + int(tap / 3u) - 1, ix = ox + int(tap % 3u) - 1;
        if (iy >= 0 && iy < int(args.H) && ix >= 0 && ix < int(args.W))
          xv = bf16_to_f32(X_nhwc[(ulong(iy * int(args.W) + ix) * args.Cin) + ic]);
      }
      sA[lid.y * 32u + lid.x * 2u + s] = xv;
    }
    uint boc = tg_id.x * 16u + lid.y;
    sB[lid.y * 32u + lid.x * 2u]      = (boc < args.Cout && kk0 < K) ? bf16_to_f32(W_ohwi[(ulong)boc * K + kk0]) : 0.0f;
    sB[lid.y * 32u + lid.x * 2u + 1u] = (boc < args.Cout && kk0 + 1u < K) ? bf16_to_f32(W_ohwi[(ulong)boc * K + kk0 + 1u]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32u; k++) acc = fma(sA[lid.y * 32u + k], sB[lid.x * 32u + k], acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (m < M && oc < args.Cout) {
    ulong off = ulong(m) * args.Cout + oc;
    if (args.has_bias) acc += bias[oc];
    if (args.has_res)  acc += res[off];
    if (args.write_f32)  Y_f32[off]  = acc;
    if (args.write_bf16) Y_bf16[off] = f32_to_bf16(acc);
  }
}

kernel void k_metal_quantize_bf16_q8g64_tiled(
    device const ushort *src_bf16 [[buffer(0)]],
    device char         *dst_q8   [[buffer(1)]],
    device float        *dst_sc   [[buffer(2)]],
    constant uint       &n_groups [[buffer(3)]],
    uint g                        [[thread_position_in_grid]]) {
  if (g >= n_groups) return;
  ulong base = ulong(g) * 64ul;
  float amax = 0.0f;
  for (uint i = 0; i < 64u; i++) amax = max(amax, abs(bf16_to_f32(src_bf16[base + i])));
  float sc = amax / 127.0f;
  float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
  dst_sc[g] = sc;
  for (uint i = 0; i < 64u; i++) dst_q8[base + i] = char(clamp(int(rint(bf16_to_f32(src_bf16[base + i]) * inv)), -127, 127));
}

// Width-agnostic 64-thread threadgroup-reduction FA2 (works on Intel simd16, AMD wave64, Apple simd32!)
kernel void k_metal_fa2_bf16_tiled(
    device const ushort *Q        [[buffer(0)]],
    device const ushort *K        [[buffer(1)]],
    device const ushort *V        [[buffer(2)]],
    device float        *O_f32    [[buffer(3)]],
    device ushort       *O_bf16   [[buffer(4)]],
    constant EcFa2Args  &args     [[buffer(5)]],
    uint pair                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]]) {
  if (pair >= args.Lq * args.Hq) return;
  threadgroup float s_red[64];
  uint qpos = pair / args.Hq, hq = pair % args.Hq, hkv = hq / (args.Hq / args.Hkv);
  uint d0 = (args.D == 128u) ? (tid * 2u) : tid;
  uint d1 = d0 + 1u;
  device const ushort *qp = Q + (ulong(hq) * args.Lq_pad + qpos) * args.D;
  float q0 = bf16_to_f32(qp[d0]);
  float q1 = (args.D == 128u) ? bf16_to_f32(qp[d1]) : 0.0f;
  float acc0 = 0.0f, acc1 = 0.0f, m_max = -1e30f, l_sum = 0.0f;
  uint k_start = 0u, k_end = args.Lkv;
  if (args.window > 0) {
    k_start = uint(max(0, int(qpos) - args.window));
    k_end   = uint(min(int(args.Lkv), int(qpos) + args.window + 1));
  }
  device const ushort *kb = K + ulong(hkv) * args.Lkv_pad * args.D;
  device const ushort *vb = V + ulong(hkv) * args.Lkv_pad * args.D;
  for (uint kpos = k_start; kpos < k_end; kpos++) {
    device const ushort *kp = kb + ulong(kpos) * args.D;
    float part = q0 * bf16_to_f32(kp[d0]);
    if (args.D == 128u) part = fma(q1, bf16_to_f32(kp[d1]), part);
    s_red[tid] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
      s_red[tid] = s_red[tid] + s_red[tid + 8u] + s_red[tid + 16u] + s_red[tid + 24u] +
                   s_red[tid + 32u] + s_red[tid + 40u] + s_red[tid + 48u] + s_red[tid + 56u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      s_red[0] = (s_red[0] + s_red[1] + s_red[2] + s_red[3] +
                  s_red[4] + s_red[5] + s_red[6] + s_red[7]) * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc = s_red[0];
    float nm = max(m_max, sc);
    float al = exp(m_max - nm), p = exp(sc - nm);
    l_sum = l_sum * al + p; m_max = nm;
    device const ushort *vp = vb + ulong(kpos) * args.D;
    acc0 = acc0 * al + p * bf16_to_f32(vp[d0]);
    if (args.D == 128u) acc1 = acc1 * al + p * bf16_to_f32(vp[d1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  ulong o_off = (ulong(qpos) * args.Hq + hq) * args.D;
  if (args.write_f32) O_f32[o_off + d0] = acc0 * inv;
  if (args.write_16)  O_bf16[o_off + d0] = f32_to_bf16(acc0 * inv);
  if (args.D == 128u) {
    if (args.write_f32) O_f32[o_off + d1] = acc1 * inv;
    if (args.write_16)  O_bf16[o_off + d1] = f32_to_bf16(acc1 * inv);
  }
}

kernel void k_metal_fa2_f16_tiled(
    device const half  *Q        [[buffer(0)]],
    device const half  *K        [[buffer(1)]],
    device const half  *V        [[buffer(2)]],
    device float       *O_f32    [[buffer(3)]],
    device half        *O_f16    [[buffer(4)]],
    constant EcFa2Args &args     [[buffer(5)]],
    uint pair                    [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]) {
  if (pair >= args.Lq * args.Hq) return;
  threadgroup float s_red[64];
  uint qpos = pair / args.Hq, h = pair % args.Hq;
  uint d0 = (args.D == 128u) ? (tid * 2u) : tid;
  uint d1 = d0 + 1u;
  device const half *qp = Q + (ulong(h) * args.Lq_pad + qpos) * args.D;
  float q0 = float(qp[d0]);
  float q1 = (args.D == 128u) ? float(qp[d1]) : 0.0f;
  float acc0 = 0.0f, acc1 = 0.0f, m_max = -1e30f, l_sum = 0.0f;
  uint k_end = args.Lkv;
  if (args.block_causal > 0u) k_end = min(args.Lkv, ((qpos / args.block_causal) + 1u) * args.block_causal);
  device const half *kb = K + ulong(h) * args.Lkv_pad * args.D;
  device const half *vb = V + ulong(h) * args.Lkv_pad * args.D;
  for (uint kpos = 0; kpos < k_end; kpos++) {
    device const half *kp = kb + ulong(kpos) * args.D;
    float part = q0 * float(kp[d0]);
    if (args.D == 128u) part = fma(q1, float(kp[d1]), part);
    s_red[tid] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
      s_red[tid] = s_red[tid] + s_red[tid + 8u] + s_red[tid + 16u] + s_red[tid + 24u] +
                   s_red[tid + 32u] + s_red[tid + 40u] + s_red[tid + 48u] + s_red[tid + 56u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      s_red[0] = (s_red[0] + s_red[1] + s_red[2] + s_red[3] +
                  s_red[4] + s_red[5] + s_red[6] + s_red[7]) * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc = s_red[0];
    float nm = max(m_max, sc);
    float al = exp(m_max - nm), p = exp(sc - nm);
    l_sum = l_sum * al + p; m_max = nm;
    device const half *vp = vb + ulong(kpos) * args.D;
    acc0 = acc0 * al + p * float(vp[d0]);
    if (args.D == 128u) acc1 = acc1 * al + p * float(vp[d1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  ulong o_off = (ulong(qpos) * args.Hq + h) * args.D;
  if (args.write_f32) O_f32[o_off + d0] = acc0 * inv;
  if (args.write_16)  O_f16[o_off + d0] = f32_to_f16_safe(acc0 * inv);
  if (args.D == 128u) {
    if (args.write_f32) O_f32[o_off + d1] = acc1 * inv;
    if (args.write_16)  O_f16[o_off + d1] = f32_to_f16_safe(acc1 * inv);
  }
}
