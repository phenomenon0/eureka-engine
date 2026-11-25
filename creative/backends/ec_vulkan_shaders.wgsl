// ec_vulkan_shaders.wgsl — High-Throughput Cross-Vendor Vulkan Compute Shaders (SPIR-V 1.3 via Naga)
// Features:
//   - Zero-bank-conflict 4x4 register-tiled BM=64, BN=64, BK=16 GEMM & Implicit-GEMM Conv3x3:
//       * Transposed padded shared tiles [16 x 65] (8.32 KiB total -> high SM/CU occupancy & fast RADV ACO compile!)
//       * 16-strided thread ownership (lid + {0,16,32,48}) -> 100% coalesced global writes & bias reads
//       * Coalesced packed u32 BF16x2 / FP16x2 global loads
//   - 4-key-unrolled zero-spill coalesced 64-thread workgroup FlashAttention-2 (4x fewer barriers)
//   - Explicit sub-256B element offsets (o0..o5) + TDR-safe workgroup offsets (pad0, pad1).

struct Params {
  p0: u32, p1: u32, p2: u32, p3: u32,
  p4: u32, p5: u32, p6: u32, p7: u32,
  o0: u32, o1: u32, o2: u32, o3: u32,
  o4: u32, o5: u32, pad0: u32, pad1: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> b_in0: array<u32>;
@group(0) @binding(2) var<storage, read> b_in1: array<u32>;
@group(0) @binding(3) var<storage, read> b_in2: array<f32>;
@group(0) @binding(4) var<storage, read> b_in3: array<f32>;
@group(0) @binding(5) var<storage, read_write> b_out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> b_out1: array<u32>;

fn bf16_at_in0(elem_idx: u32) -> f32 {
  let g = u.o0 + elem_idx;
  let w = b_in0[g >> 1u];
  return select(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u), (g & 1u) != 0u);
}

fn bf16_at_in1(elem_idx: u32) -> f32 {
  let g = u.o1 + elem_idx;
  let w = b_in1[g >> 1u];
  return select(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u), (g & 1u) != 0u);
}

fn bf16_at_out1(elem_idx: u32) -> f32 {
  let g = u.o5 + elem_idx;
  let w = b_out1[g >> 1u];
  return select(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u), (g & 1u) != 0u);
}

fn bf16x2_in0(elem_idx: u32) -> vec2<f32> {
  let g = u.o0 + elem_idx;
  if ((g & 1u) == 0u) {
    let w = b_in0[g >> 1u];
    return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
  }
  return vec2<f32>(bf16_at_in0(elem_idx), bf16_at_in0(elem_idx + 1u));
}

fn bf16x2_in1(elem_idx: u32) -> vec2<f32> {
  let g = u.o1 + elem_idx;
  if ((g & 1u) == 0u) {
    let w = b_in1[g >> 1u];
    return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
  }
  return vec2<f32>(bf16_at_in1(elem_idx), bf16_at_in1(elem_idx + 1u));
}

fn f16_at_in0(elem_idx: u32) -> f32 {
  let g = u.o0 + elem_idx;
  let p = unpack2x16float(b_in0[g >> 1u]);
  return select(p.x, p.y, (g & 1u) != 0u);
}

fn f16_at_in1(elem_idx: u32) -> f32 {
  let g = u.o1 + elem_idx;
  let p = unpack2x16float(b_in1[g >> 1u]);
  return select(p.x, p.y, (g & 1u) != 0u);
}

fn f16_at_out1(elem_idx: u32) -> f32 {
  let g = u.o5 + elem_idx;
  let p = unpack2x16float(b_out1[g >> 1u]);
  return select(p.x, p.y, (g & 1u) != 0u);
}

fn f16x2_in0(elem_idx: u32) -> vec2<f32> {
  let g = u.o0 + elem_idx;
  if ((g & 1u) == 0u) {
    return unpack2x16float(b_in0[g >> 1u]);
  }
  return vec2<f32>(f16_at_in0(elem_idx), f16_at_in0(elem_idx + 1u));
}

fn f16x2_in1(elem_idx: u32) -> vec2<f32> {
  let g = u.o1 + elem_idx;
  if ((g & 1u) == 0u) {
    return unpack2x16float(b_in1[g >> 1u]);
  }
  return vec2<f32>(f16_at_in1(elem_idx), f16_at_in1(elem_idx + 1u));
}

fn f32_to_bf16_u16(v: f32) -> u32 {
  var bits = bitcast<u32>(v);
  let lsb = (bits >> 16u) & 1u;
  bits = bits + 0x7fffu + lsb;
  return (bits >> 16u) & 0xffffu;
}

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

fn gelu_erf(x: f32) -> f32 {
  let c = 0.7978845608 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(clamp(c, -12.0, 12.0)));
}

fn s8_at_in0(byte_off: u32) -> i32 {
  let g = u.o0 + byte_off;
  let w = b_in0[g >> 2u];
  let sh = (3u - (g & 3u)) * 8u;
  return (i32(w << sh)) >> 24u;
}

fn s8_at_in1(byte_off: u32) -> i32 {
  let g = u.o1 + byte_off;
  let w = b_in1[g >> 2u];
  let sh = (3u - (g & 3u)) * 8u;
  return (i32(w << sh)) >> 24u;
}

// Transposed padded [16 x 65] shared tiles (4.16 KiB each, 8.32 KiB total -> 0 bank conflicts + fast RADV compile!)
var<workgroup> tileA: array<f32, 1040>;
var<workgroup> tileB: array<f32, 1040>;

// 1. Register-Tiled 64x64x16 (4x4 per thread) BF16 GEMM
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_bf16_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let m0 = (wg.y + u.pad1) * 64u;
  let n0 = (wg.x + u.pad0) * 64u;
  let tid = lid.y * 16u + lid.x;
  let load_r0 = tid >> 3u;       // 0..31
  let load_c  = (tid & 7u) * 2u; // 0,2..14

  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);

  let n_tiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 16u + load_c;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let r = load_r0 + s * 32u;
      let so0 = load_c * 65u + r;
      let so1 = (load_c + 1u) * 65u + r;
      let gm = m0 + r;
      if (gm < M && k0 + 1u < K) {
        let p = bf16x2_in0(gm * K + k0);
        tileA[so0] = p.x; tileA[so1] = p.y;
      } else if (gm < M && k0 < K) {
        tileA[so0] = bf16_at_in0(gm * K + k0); tileA[so1] = 0.0;
      } else {
        tileA[so0] = 0.0; tileA[so1] = 0.0;
      }
      let gn = n0 + r;
      if (gn < N && k0 + 1u < K) {
        let p = bf16x2_in1(gn * K + k0);
        tileB[so0] = p.x; tileB[so1] = p.y;
      } else if (gn < N && k0 < K) {
        tileB[so0] = bf16_at_in1(gn * K + k0); tileB[so1] = 0.0;
      } else {
        tileB[so0] = 0.0; tileB[so1] = 0.0;
      }
    }
    workgroupBarrier();

    let ly = lid.y;
    let lx = lid.x;
    let k_lim = min(16u, K - t * 16u);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      let kb = kk * 65u;
      let bv = vec4<f32>(tileB[kb + lx], tileB[kb + lx + 16u], tileB[kb + lx + 32u], tileB[kb + lx + 48u]);
      c0 = fma(vec4<f32>(tileA[kb + ly]),       bv, c0);
      c1 = fma(vec4<f32>(tileA[kb + ly + 16u]), bv, c1);
      c2 = fma(vec4<f32>(tileA[kb + ly + 32u]), bv, c2);
      c3 = fma(vec4<f32>(tileA[kb + ly + 48u]), bv, c3);
    }
    workgroupBarrier();
  }

  let r0 = m0 + lid.y;
  let col = n0 + lid.x;
  var bs = vec4<f32>(0.0);
  if (has_bias != 0u) {
    if (col + 0u  < N) { bs.x = b_in2[u.o2 + col + 0u]; }
    if (col + 16u < N) { bs.y = b_in2[u.o2 + col + 16u]; }
    if (col + 32u < N) { bs.z = b_in2[u.o2 + col + 32u]; }
    if (col + 48u < N) { bs.w = b_in2[u.o2 + col + 48u]; }
  }
  c0 = c0 + bs; c1 = c1 + bs; c2 = c2 + bs; c3 = c3 + bs;
  if (r0 + 0u < M) {
    let ro = u.o4 + (r0 + 0u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c0.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c0.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c0.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c0.w; }
  }
  if (r0 + 16u < M) {
    let ro = u.o4 + (r0 + 16u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c1.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c1.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c1.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c1.w; }
  }
  if (r0 + 32u < M) {
    let ro = u.o4 + (r0 + 32u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c2.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c2.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c2.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c2.w; }
  }
  if (r0 + 48u < M) {
    let ro = u.o4 + (r0 + 48u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c3.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c3.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c3.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c3.w; }
  }
}

fn apply_epi(val: f32, epilogue: u32, has_bias: u32, has_res: u32, bv: f32, off: u32) -> f32 {
  var acc = val;
  if (has_bias != 0u && epilogue >= 1u) { acc = acc + bv; }
  if (epilogue == 2u) { acc = silu(acc); }
  else if (epilogue == 3u) { acc = gelu_erf(acc); }
  else if (epilogue == 4u && has_res != 0u) { acc = acc + b_in3[u.o3 + off]; }
  return acc;
}

// 2. Register-Tiled 64x64x16 (4x4 per thread) FP16 GEMM + Epilogue
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_f16_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let epilogue = u.p3;
  let has_bias = u.p4;
  let has_res = u.p5;
  let m0 = (wg.y + u.pad1) * 64u;
  let n0 = (wg.x + u.pad0) * 64u;
  let tid = lid.y * 16u + lid.x;
  let load_r0 = tid >> 3u;
  let load_c  = (tid & 7u) * 2u;

  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);

  let n_tiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 16u + load_c;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let r = load_r0 + s * 32u;
      let so0 = load_c * 65u + r;
      let so1 = (load_c + 1u) * 65u + r;
      let gm = m0 + r;
      if (gm < M && k0 + 1u < K) {
        let p = f16x2_in0(gm * K + k0);
        tileA[so0] = p.x; tileA[so1] = p.y;
      } else if (gm < M && k0 < K) {
        tileA[so0] = f16_at_in0(gm * K + k0); tileA[so1] = 0.0;
      } else {
        tileA[so0] = 0.0; tileA[so1] = 0.0;
      }
      let gn = n0 + r;
      if (gn < N && k0 + 1u < K) {
        let p = f16x2_in1(gn * K + k0);
        tileB[so0] = p.x; tileB[so1] = p.y;
      } else if (gn < N && k0 < K) {
        tileB[so0] = f16_at_in1(gn * K + k0); tileB[so1] = 0.0;
      } else {
        tileB[so0] = 0.0; tileB[so1] = 0.0;
      }
    }
    workgroupBarrier();

    let ly = lid.y;
    let lx = lid.x;
    let k_lim = min(16u, K - t * 16u);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      let kb = kk * 65u;
      let bv = vec4<f32>(tileB[kb + lx], tileB[kb + lx + 16u], tileB[kb + lx + 32u], tileB[kb + lx + 48u]);
      c0 = fma(vec4<f32>(tileA[kb + ly]),       bv, c0);
      c1 = fma(vec4<f32>(tileA[kb + ly + 16u]), bv, c1);
      c2 = fma(vec4<f32>(tileA[kb + ly + 32u]), bv, c2);
      c3 = fma(vec4<f32>(tileA[kb + ly + 48u]), bv, c3);
    }
    workgroupBarrier();
  }

  let r0 = m0 + lid.y;
  let col = n0 + lid.x;
  var bs = vec4<f32>(0.0);
  if (has_bias != 0u && epilogue >= 1u) {
    if (col + 0u  < N) { bs.x = b_in2[u.o2 + col + 0u]; }
    if (col + 16u < N) { bs.y = b_in2[u.o2 + col + 16u]; }
    if (col + 32u < N) { bs.z = b_in2[u.o2 + col + 32u]; }
    if (col + 48u < N) { bs.w = b_in2[u.o2 + col + 48u]; }
  }
  if (r0 + 0u < M) {
    let ro = (r0 + 0u) * N + col;
    if (col + 0u  < N) { b_out0[u.o4 + ro + 0u]  = apply_epi(c0.x, epilogue, has_bias, has_res, bs.x, ro + 0u); }
    if (col + 16u < N) { b_out0[u.o4 + ro + 16u] = apply_epi(c0.y, epilogue, has_bias, has_res, bs.y, ro + 16u); }
    if (col + 32u < N) { b_out0[u.o4 + ro + 32u] = apply_epi(c0.z, epilogue, has_bias, has_res, bs.z, ro + 32u); }
    if (col + 48u < N) { b_out0[u.o4 + ro + 48u] = apply_epi(c0.w, epilogue, has_bias, has_res, bs.w, ro + 48u); }
  }
  if (r0 + 16u < M) {
    let ro = (r0 + 16u) * N + col;
    if (col + 0u  < N) { b_out0[u.o4 + ro + 0u]  = apply_epi(c1.x, epilogue, has_bias, has_res, bs.x, ro + 0u); }
    if (col + 16u < N) { b_out0[u.o4 + ro + 16u] = apply_epi(c1.y, epilogue, has_bias, has_res, bs.y, ro + 16u); }
    if (col + 32u < N) { b_out0[u.o4 + ro + 32u] = apply_epi(c1.z, epilogue, has_bias, has_res, bs.z, ro + 32u); }
    if (col + 48u < N) { b_out0[u.o4 + ro + 48u] = apply_epi(c1.w, epilogue, has_bias, has_res, bs.w, ro + 48u); }
  }
  if (r0 + 32u < M) {
    let ro = (r0 + 32u) * N + col;
    if (col + 0u  < N) { b_out0[u.o4 + ro + 0u]  = apply_epi(c2.x, epilogue, has_bias, has_res, bs.x, ro + 0u); }
    if (col + 16u < N) { b_out0[u.o4 + ro + 16u] = apply_epi(c2.y, epilogue, has_bias, has_res, bs.y, ro + 16u); }
    if (col + 32u < N) { b_out0[u.o4 + ro + 32u] = apply_epi(c2.z, epilogue, has_bias, has_res, bs.z, ro + 32u); }
    if (col + 48u < N) { b_out0[u.o4 + ro + 48u] = apply_epi(c2.w, epilogue, has_bias, has_res, bs.w, ro + 48u); }
  }
  if (r0 + 48u < M) {
    let ro = (r0 + 48u) * N + col;
    if (col + 0u  < N) { b_out0[u.o4 + ro + 0u]  = apply_epi(c3.x, epilogue, has_bias, has_res, bs.x, ro + 0u); }
    if (col + 16u < N) { b_out0[u.o4 + ro + 16u] = apply_epi(c3.y, epilogue, has_bias, has_res, bs.y, ro + 16u); }
    if (col + 32u < N) { b_out0[u.o4 + ro + 32u] = apply_epi(c3.z, epilogue, has_bias, has_res, bs.z, ro + 32u); }
    if (col + 48u < N) { b_out0[u.o4 + ro + 48u] = apply_epi(c3.w, epilogue, has_bias, has_res, bs.w, ro + 48u); }
  }
}

// 3. Quantize BF16 [rows, cols] -> Q8_G64
@compute @workgroup_size(64, 1, 1)
fn vk_quantize_bf16_q8g64(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = (wg.x + u.pad0) * 64u + lid.x;
  if (g >= u.p0) { return; }
  let elem_base = g * 64u;
  var amax: f32 = 0.0;
  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    amax = max(amax, abs(bf16_at_in0(elem_base + i)));
  }
  let sc = amax / 127.0;
  let inv = select(0.0, 127.0 / amax, amax > 1e-30);
  b_out0[u.o4 + g] = sc;
  let out_u32 = u.o5 + g * 16u;
  for (var j: u32 = 0u; j < 16u; j = j + 1u) {
    let q0 = u32(clamp(i32(round(bf16_at_in0(elem_base + j * 4u + 0u) * inv)), -127, 127)) & 0xffu;
    let q1 = u32(clamp(i32(round(bf16_at_in0(elem_base + j * 4u + 1u) * inv)), -127, 127)) & 0xffu;
    let q2 = u32(clamp(i32(round(bf16_at_in0(elem_base + j * 4u + 2u) * inv)), -127, 127)) & 0xffu;
    let q3 = u32(clamp(i32(round(bf16_at_in0(elem_base + j * 4u + 3u) * inv)), -127, 127)) & 0xffu;
    b_out1[out_u32 + j] = q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u);
  }
}

// 4. Register-Tiled 64x64x16 (4x4 per thread) Q8_G64 GEMM
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_q8g64_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let m0 = (wg.y + u.pad1) * 64u;
  let n0 = (wg.x + u.pad0) * 64u;
  let groups = K >> 6u;
  let tid = lid.y * 16u + lid.x;
  let load_r0 = tid >> 3u;
  let load_c  = (tid & 7u) * 2u;

  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);

  let n_tiles = K >> 4u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let g = t >> 2u;
    let k0 = t * 16u + load_c;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let r = load_r0 + s * 32u;
      let so0 = load_c * 65u + r;
      let so1 = (load_c + 1u) * 65u + r;
      let gm = m0 + r;
      if (gm < M) {
        let xsc = b_in2[u.o2 + gm * groups + g];
        tileA[so0] = f32(s8_at_in0(gm * K + k0)) * xsc;
        tileA[so1] = f32(s8_at_in0(gm * K + k0 + 1u)) * xsc;
      } else {
        tileA[so0] = 0.0; tileA[so1] = 0.0;
      }
      let gn = n0 + r;
      if (gn < N) {
        let wsc = b_in3[u.o3 + gn * groups + g];
        tileB[so0] = f32(s8_at_in1(gn * K + k0)) * wsc;
        tileB[so1] = f32(s8_at_in1(gn * K + k0 + 1u)) * wsc;
      } else {
        tileB[so0] = 0.0; tileB[so1] = 0.0;
      }
    }
    workgroupBarrier();

    let ly = lid.y;
    let lx = lid.x;
    let k_lim = min(16u, K - t * 16u);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      let kb = kk * 65u;
      let bv = vec4<f32>(tileB[kb + lx], tileB[kb + lx + 16u], tileB[kb + lx + 32u], tileB[kb + lx + 48u]);
      c0 = fma(vec4<f32>(tileA[kb + ly]),       bv, c0);
      c1 = fma(vec4<f32>(tileA[kb + ly + 16u]), bv, c1);
      c2 = fma(vec4<f32>(tileA[kb + ly + 32u]), bv, c2);
      c3 = fma(vec4<f32>(tileA[kb + ly + 48u]), bv, c3);
    }
    workgroupBarrier();
  }

  let r0 = m0 + lid.y;
  let col = n0 + lid.x;
  var bs = vec4<f32>(0.0);
  if (has_bias != 0u) {
    if (col + 0u  < N) { bs.x = bitcast<f32>(b_out1[u.o5 + col + 0u]); }
    if (col + 16u < N) { bs.y = bitcast<f32>(b_out1[u.o5 + col + 16u]); }
    if (col + 32u < N) { bs.z = bitcast<f32>(b_out1[u.o5 + col + 32u]); }
    if (col + 48u < N) { bs.w = bitcast<f32>(b_out1[u.o5 + col + 48u]); }
  }
  c0 = c0 + bs; c1 = c1 + bs; c2 = c2 + bs; c3 = c3 + bs;
  if (r0 + 0u < M) {
    let ro = u.o4 + (r0 + 0u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c0.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c0.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c0.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c0.w; }
  }
  if (r0 + 16u < M) {
    let ro = u.o4 + (r0 + 16u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c1.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c1.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c1.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c1.w; }
  }
  if (r0 + 32u < M) {
    let ro = u.o4 + (r0 + 32u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c2.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c2.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c2.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c2.w; }
  }
  if (r0 + 48u < M) {
    let ro = u.o4 + (r0 + 48u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = c3.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = c3.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = c3.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = c3.w; }
  }
}

// 5. 4-Key-Unrolled Coalesced 64-Thread Workgroup FlashAttention-2 BF16 (4x fewer barriers!)
@compute @workgroup_size(64, 1, 1)
fn vk_fa2_bf16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Lq = u.p0;
  let Lkv = u.p1;
  let Lq_pad = u.p2;
  let Lkv_pad = u.p3;
  let Hq = u.p4;
  let Hkv = u.p5;
  let D = u.p6;
  let win = i32(u.p7);
  let scale = u.f0;

  let pair = wg.x + u.pad0;
  if (pair >= Lq * Hq) { return; }
  let tid = lid.x;
  let qpos = pair / Hq;
  let hq = pair % Hq;
  let hkv = hq / (Hq / Hkv);

  let q_off = (hq * Lq_pad + qpos) * D;
  let d0 = select(tid, tid * 2u, D == 128u);
  let d1 = d0 + 1u;
  let q0 = bf16_at_in0(q_off + d0);
  let q1 = select(0.0, bf16_at_in0(q_off + d1), D == 128u);

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var k_start: u32 = 0u;
  var k_end: u32 = Lkv;
  if (win > 0) {
    k_start = u32(max(0, i32(qpos) - win));
    k_end = u32(min(i32(Lkv), i32(qpos) + win + 1));
  }

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  let kv_head_off = hkv * Lkv_pad * D;

  var kpos = k_start;
  loop {
    if (kpos >= k_end) { break; }
    let chunk = min(4u, k_end - kpos);
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      var part: f32 = 0.0;
      if (j < chunk) {
        let k_off = kv_head_off + (kpos + j) * D;
        part = q0 * bf16_at_in1(k_off + d0);
        if (D == 128u) { part = fma(q1, bf16_at_in1(k_off + d1), part); }
      }
      tileA[j * 64u + tid] = part;
    }
    workgroupBarrier();
    if (tid < 32u) {
      let j = tid >> 3u;
      let s = tid & 7u;
      let base = j * 64u + s;
      tileA[base] = tileA[base] + tileA[base + 8u] + tileA[base + 16u] + tileA[base + 24u] +
                    tileA[base + 32u] + tileA[base + 40u] + tileA[base + 48u] + tileA[base + 56u];
    }
    workgroupBarrier();
    if (tid < 4u) {
      let base = tid * 64u;
      tileB[tid] = (tileA[base] + tileA[base + 1u] + tileA[base + 2u] + tileA[base + 3u] +
                    tileA[base + 4u] + tileA[base + 5u] + tileA[base + 6u] + tileA[base + 7u]) * scale;
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < chunk; j = j + 1u) {
      let sc = tileB[j];
      let nm = max(m_max, sc);
      let alpha = exp(m_max - nm);
      let p = exp(sc - nm);
      l_sum = l_sum * alpha + p;
      m_max = nm;
      let k_off = kv_head_off + (kpos + j) * D;
      acc0 = acc0 * alpha + p * bf16_at_out1(k_off + d0);
      if (D == 128u) { acc1 = acc1 * alpha + p * bf16_at_out1(k_off + d1); }
    }
    workgroupBarrier();
    kpos = kpos + 4u;
  }

  let inv = select(0.0, 1.0 / l_sum, l_sum > 0.0);
  let o_off = u.o4 + (qpos * Hq + hq) * D;
  b_out0[o_off + d0] = acc0 * inv;
  if (D == 128u) { b_out0[o_off + d1] = acc1 * inv; }
}

// 6. 4-Key-Unrolled Coalesced 64-Thread Workgroup FlashAttention-2 FP16
@compute @workgroup_size(64, 1, 1)
fn vk_fa2_f16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n_q = u.p0;
  let n_kv = u.p1;
  let n_q_pad = u.p2;
  let n_kv_pad = u.p3;
  let H = u.p4;
  let D = u.p5;
  let bcs = u.p6;
  let scale = u.f0;

  let pair = wg.x + u.pad0;
  if (pair >= n_q * H) { return; }
  let tid = lid.x;
  let qpos = pair / H;
  let h = pair % H;

  let q_off = (h * n_q_pad + qpos) * D;
  let d0 = select(tid, tid * 2u, D == 128u);
  let d1 = d0 + 1u;
  let q0 = f16_at_in0(q_off + d0);
  let q1 = select(0.0, f16_at_in0(q_off + d1), D == 128u);

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var k_limit = n_kv;
  if (bcs > 0u) { k_limit = min(n_kv, ((qpos / bcs) + 1u) * bcs); }

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  let kv_head_off = h * n_kv_pad * D;

  var kpos: u32 = 0u;
  loop {
    if (kpos >= k_limit) { break; }
    let chunk = min(4u, k_limit - kpos);
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      var part: f32 = 0.0;
      if (j < chunk) {
        let k_off = kv_head_off + (kpos + j) * D;
        part = q0 * f16_at_in1(k_off + d0);
        if (D == 128u) { part = fma(q1, f16_at_in1(k_off + d1), part); }
      }
      tileA[j * 64u + tid] = part;
    }
    workgroupBarrier();
    if (tid < 32u) {
      let j = tid >> 3u;
      let s = tid & 7u;
      let base = j * 64u + s;
      tileA[base] = tileA[base] + tileA[base + 8u] + tileA[base + 16u] + tileA[base + 24u] +
                    tileA[base + 32u] + tileA[base + 40u] + tileA[base + 48u] + tileA[base + 56u];
    }
    workgroupBarrier();
    if (tid < 4u) {
      let base = tid * 64u;
      tileB[tid] = (tileA[base] + tileA[base + 1u] + tileA[base + 2u] + tileA[base + 3u] +
                    tileA[base + 4u] + tileA[base + 5u] + tileA[base + 6u] + tileA[base + 7u]) * scale;
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < chunk; j = j + 1u) {
      let sc = tileB[j];
      let nm = max(m_max, sc);
      let alpha = exp(m_max - nm);
      let p = exp(sc - nm);
      l_sum = l_sum * alpha + p;
      m_max = nm;
      let k_off = kv_head_off + (kpos + j) * D;
      acc0 = acc0 * alpha + p * f16_at_out1(k_off + d0);
      if (D == 128u) { acc1 = acc1 * alpha + p * f16_at_out1(k_off + d1); }
    }
    workgroupBarrier();
    kpos = kpos + 4u;
  }

  let inv = select(0.0, 1.0 / l_sum, l_sum > 0.0);
  let o_off = u.o4 + (qpos * H + h) * D;
  b_out0[o_off + d0] = acc0 * inv;
  if (D == 128u) { b_out0[o_off + d1] = acc1 * inv; }
}

// 7. Register-Tiled 64x64x16 (4x4 per thread) Implicit-GEMM NHWC 3x3 Conv2D BF16
@compute @workgroup_size(16, 16, 1)
fn vk_conv2d_3x3_nhwc_bf16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let H = i32(u.p0);
  let W = i32(u.p1);
  let Cin = u.p2;
  let Cout = u.p3;
  let has_bias = u.p4;
  let has_res = u.p5;
  let K = 9u * Cin;
  let M = u.p0 * u.p1;
  let m0 = (wg.y + u.pad1) * 64u;
  let n0 = (wg.x + u.pad0) * 64u;
  let tid = lid.y * 16u + lid.x;
  let load_r0 = tid >> 3u;
  let load_c  = (tid & 7u) * 2u;

  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);

  let n_tiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 16u + load_c;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let r = load_r0 + s * 32u;
      let so0 = load_c * 65u + r;
      let so1 = (load_c + 1u) * 65u + r;
      let gm = m0 + r;
      if (gm < M) {
        let oy = i32(gm) / W;
        let ox = i32(gm) - oy * W;
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
          let kk = k0 + j;
          var xv: f32 = 0.0;
          if (kk < K) {
            let tap = kk / Cin;
            let ic = kk % Cin;
            let iy = oy + i32(tap / 3u) - 1;
            let ix = ox + i32(tap % 3u) - 1;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
              xv = bf16_at_in0((u32(iy * W + ix) * Cin) + ic);
            }
          }
          if (j == 0u) { tileA[so0] = xv; } else { tileA[so1] = xv; }
        }
      } else {
        tileA[so0] = 0.0; tileA[so1] = 0.0;
      }
      let goc = n0 + r;
      if (goc < Cout && k0 + 1u < K) {
        let p = bf16x2_in1(goc * K + k0);
        tileB[so0] = p.x; tileB[so1] = p.y;
      } else if (goc < Cout && k0 < K) {
        tileB[so0] = bf16_at_in1(goc * K + k0); tileB[so1] = 0.0;
      } else {
        tileB[so0] = 0.0; tileB[so1] = 0.0;
      }
    }
    workgroupBarrier();

    let ly = lid.y;
    let lx = lid.x;
    let k_lim = min(16u, K - t * 16u);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      let kb = kk * 65u;
      let bv = vec4<f32>(tileB[kb + lx], tileB[kb + lx + 16u], tileB[kb + lx + 32u], tileB[kb + lx + 48u]);
      c0 = fma(vec4<f32>(tileA[kb + ly]),       bv, c0);
      c1 = fma(vec4<f32>(tileA[kb + ly + 16u]), bv, c1);
      c2 = fma(vec4<f32>(tileA[kb + ly + 32u]), bv, c2);
      c3 = fma(vec4<f32>(tileA[kb + ly + 48u]), bv, c3);
    }
    workgroupBarrier();
  }

  let r0 = m0 + lid.y;
  let col = n0 + lid.x;
  var bs = vec4<f32>(0.0);
  if (has_bias != 0u) {
    if (col + 0u  < Cout) { bs.x = b_in2[u.o2 + col + 0u]; }
    if (col + 16u < Cout) { bs.y = b_in2[u.o2 + col + 16u]; }
    if (col + 32u < Cout) { bs.z = b_in2[u.o2 + col + 32u]; }
    if (col + 48u < Cout) { bs.w = b_in2[u.o2 + col + 48u]; }
  }
  c0 = c0 + bs; c1 = c1 + bs; c2 = c2 + bs; c3 = c3 + bs;
  if (r0 + 0u < M) {
    let ro = (r0 + 0u) * Cout + col;
    if (col + 0u  < Cout) { b_out0[u.o4 + ro + 0u]  = c0.x + select(0.0, b_in3[u.o3 + ro + 0u],  has_res != 0u); }
    if (col + 16u < Cout) { b_out0[u.o4 + ro + 16u] = c0.y + select(0.0, b_in3[u.o3 + ro + 16u], has_res != 0u); }
    if (col + 32u < Cout) { b_out0[u.o4 + ro + 32u] = c0.z + select(0.0, b_in3[u.o3 + ro + 32u], has_res != 0u); }
    if (col + 48u < Cout) { b_out0[u.o4 + ro + 48u] = c0.w + select(0.0, b_in3[u.o3 + ro + 48u], has_res != 0u); }
  }
  if (r0 + 16u < M) {
    let ro = (r0 + 16u) * Cout + col;
    if (col + 0u  < Cout) { b_out0[u.o4 + ro + 0u]  = c1.x + select(0.0, b_in3[u.o3 + ro + 0u],  has_res != 0u); }
    if (col + 16u < Cout) { b_out0[u.o4 + ro + 16u] = c1.y + select(0.0, b_in3[u.o3 + ro + 16u], has_res != 0u); }
    if (col + 32u < Cout) { b_out0[u.o4 + ro + 32u] = c1.z + select(0.0, b_in3[u.o3 + ro + 32u], has_res != 0u); }
    if (col + 48u < Cout) { b_out0[u.o4 + ro + 48u] = c1.w + select(0.0, b_in3[u.o3 + ro + 48u], has_res != 0u); }
  }
  if (r0 + 32u < M) {
    let ro = (r0 + 32u) * Cout + col;
    if (col + 0u  < Cout) { b_out0[u.o4 + ro + 0u]  = c2.x + select(0.0, b_in3[u.o3 + ro + 0u],  has_res != 0u); }
    if (col + 16u < Cout) { b_out0[u.o4 + ro + 16u] = c2.y + select(0.0, b_in3[u.o3 + ro + 16u], has_res != 0u); }
    if (col + 32u < Cout) { b_out0[u.o4 + ro + 32u] = c2.z + select(0.0, b_in3[u.o3 + ro + 32u], has_res != 0u); }
    if (col + 48u < Cout) { b_out0[u.o4 + ro + 48u] = c2.w + select(0.0, b_in3[u.o3 + ro + 48u], has_res != 0u); }
  }
  if (r0 + 48u < M) {
    let ro = (r0 + 48u) * Cout + col;
    if (col + 0u  < Cout) { b_out0[u.o4 + ro + 0u]  = c3.x + select(0.0, b_in3[u.o3 + ro + 0u],  has_res != 0u); }
    if (col + 16u < Cout) { b_out0[u.o4 + ro + 16u] = c3.y + select(0.0, b_in3[u.o3 + ro + 16u], has_res != 0u); }
    if (col + 32u < Cout) { b_out0[u.o4 + ro + 32u] = c3.z + select(0.0, b_in3[u.o3 + ro + 32u], has_res != 0u); }
    if (col + 48u < Cout) { b_out0[u.o4 + ro + 48u] = c3.w + select(0.0, b_in3[u.o3 + ro + 48u], has_res != 0u); }
  }
}

// 8. Fast Vector Pack: F32 [n_elems] -> BF16x2 u32 (mode=0), SiLU->BF16x2 (mode=1), or FP16x2 (mode=2)
@compute @workgroup_size(256, 1, 1)
fn vk_pack_f32_to_16x2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let pair = (wg.x + u.pad0) * 256u + lid.x;
  let n_elems = u.p0;
  let mode = u.p1;
  let idx0 = pair * 2u;
  if (idx0 + 1u >= n_elems) { return; }
  var v0 = b_in2[u.o2 + idx0];
  var v1 = b_in2[u.o2 + idx0 + 1u];
  if (mode == 1u) { v0 = silu(v0); v1 = silu(v1); }
  if (mode == 2u) {
    b_out1[u.o5 + pair] = pack2x16float(vec2<f32>(v0, v1));
  } else {
    b_out1[u.o5 + pair] = f32_to_bf16_u16(v0) | (f32_to_bf16_u16(v1) << 16u);
  }
}

// 9. Register-Tiled 64x64x16 (4x4 per thread) INT8 W8A8 Per-Row/Col Scaled GEMM
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_i8_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let m0 = (wg.y + u.pad1) * 64u;
  let n0 = (wg.x + u.pad0) * 64u;
  let tid = lid.y * 16u + lid.x;
  let load_r0 = tid >> 3u;
  let load_c  = (tid & 7u) * 2u;

  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);

  let n_tiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 16u + load_c;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let r = load_r0 + s * 32u;
      let so0 = load_c * 65u + r;
      let so1 = (load_c + 1u) * 65u + r;
      let gm = m0 + r;
      if (gm < M && k0 < K) {
        tileA[so0] = f32(s8_at_in0(gm * K + k0));
        tileA[so1] = select(0.0, f32(s8_at_in0(gm * K + k0 + 1u)), k0 + 1u < K);
      } else {
        tileA[so0] = 0.0; tileA[so1] = 0.0;
      }
      let gn = n0 + r;
      if (gn < N && k0 < K) {
        tileB[so0] = f32(s8_at_in1(gn * K + k0));
        tileB[so1] = select(0.0, f32(s8_at_in1(gn * K + k0 + 1u)), k0 + 1u < K);
      } else {
        tileB[so0] = 0.0; tileB[so1] = 0.0;
      }
    }
    workgroupBarrier();

    let ly = lid.y;
    let lx = lid.x;
    let k_lim = min(16u, K - t * 16u);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      let kb = kk * 65u;
      let bv = vec4<f32>(tileB[kb + lx], tileB[kb + lx + 16u], tileB[kb + lx + 32u], tileB[kb + lx + 48u]);
      c0 = fma(vec4<f32>(tileA[kb + ly]),       bv, c0);
      c1 = fma(vec4<f32>(tileA[kb + ly + 16u]), bv, c1);
      c2 = fma(vec4<f32>(tileA[kb + ly + 32u]), bv, c2);
      c3 = fma(vec4<f32>(tileA[kb + ly + 48u]), bv, c3);
    }
    workgroupBarrier();
  }

  let r0 = m0 + lid.y;
  let col = n0 + lid.x;
  var ws = vec4<f32>(0.0);
  var bs = vec4<f32>(0.0);
  if (col + 0u  < N) { ws.x = b_in3[u.o3 + col + 0u];  if (has_bias != 0u) { bs.x = bitcast<f32>(b_out1[u.o5 + col + 0u]); } }
  if (col + 16u < N) { ws.y = b_in3[u.o3 + col + 16u]; if (has_bias != 0u) { bs.y = bitcast<f32>(b_out1[u.o5 + col + 16u]); } }
  if (col + 32u < N) { ws.z = b_in3[u.o3 + col + 32u]; if (has_bias != 0u) { bs.z = bitcast<f32>(b_out1[u.o5 + col + 32u]); } }
  if (col + 48u < N) { ws.w = b_in3[u.o3 + col + 48u]; if (has_bias != 0u) { bs.w = bitcast<f32>(b_out1[u.o5 + col + 48u]); } }

  if (r0 + 0u < M) {
    let v = c0 * (b_in2[u.o2 + r0 + 0u] * ws) + bs; let ro = u.o4 + (r0 + 0u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = v.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = v.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = v.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = v.w; }
  }
  if (r0 + 16u < M) {
    let v = c1 * (b_in2[u.o2 + r0 + 16u] * ws) + bs; let ro = u.o4 + (r0 + 16u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = v.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = v.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = v.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = v.w; }
  }
  if (r0 + 32u < M) {
    let v = c2 * (b_in2[u.o2 + r0 + 32u] * ws) + bs; let ro = u.o4 + (r0 + 32u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = v.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = v.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = v.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = v.w; }
  }
  if (r0 + 48u < M) {
    let v = c3 * (b_in2[u.o2 + r0 + 48u] * ws) + bs; let ro = u.o4 + (r0 + 48u) * N + col;
    if (col + 0u  < N) { b_out0[ro + 0u]  = v.x; }
    if (col + 16u < N) { b_out0[ro + 16u] = v.y; }
    if (col + 32u < N) { b_out0[ro + 32u] = v.z; }
    if (col + 48u < N) { b_out0[ro + 48u] = v.w; }
  }
}
