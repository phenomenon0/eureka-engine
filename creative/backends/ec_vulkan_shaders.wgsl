// ec_vulkan_shaders.wgsl — Cross-Vendor Vulkan Compute Shaders for Eureka Creative Engine
// Uses standard 32-bit packed BF16x2 / FP16x2 (unpack2x16float) / INT8x4 storage + explicit sub-256B
// element/word offsets (o0..o5) + TDR-safe workgroup offsets (pad0=wg_off_x, pad1=wg_off_y).

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

var<workgroup> tileA: array<f32, 512>; // 16 * 32 (also first 64 used as s_red in FA2)
var<workgroup> tileB: array<f32, 512>; // 16 * 32

// 1. Tiled 16x16 BF16 GEMM: Y_f32[M, N] = X_bf16[M, K] @ W_bf16[N, K]^T + optional bias
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_bf16_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let wgx = wg.x + u.pad0;
  let wgy = wg.y + u.pad1;
  let row = wgy * 16u + lid.y;
  let col = wgx * 16u + lid.x;

  var acc: f32 = 0.0;
  let n_tiles = (K + 31u) / 32u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 32u + lid.x * 2u;
    if (row < M && k0 < K) {
      let idx0 = row * K + k0;
      tileA[lid.y * 32u + lid.x * 2u] = bf16_at_in0(idx0);
      tileA[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, bf16_at_in0(idx0 + 1u), k0 + 1u < K);
    } else {
      tileA[lid.y * 32u + lid.x * 2u] = 0.0;
      tileA[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }

    let b_row = wgx * 16u + lid.y;
    if (b_row < N && k0 < K) {
      let bidx0 = b_row * K + k0;
      tileB[lid.y * 32u + lid.x * 2u] = bf16_at_in1(bidx0);
      tileB[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, bf16_at_in1(bidx0 + 1u), k0 + 1u < K);
    } else {
      tileB[lid.y * 32u + lid.x * 2u] = 0.0;
      tileB[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
      acc = fma(tileA[lid.y * 32u + kk], tileB[lid.x * 32u + kk], acc);
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    if (has_bias != 0u) {
      acc = acc + b_in2[u.o2 + col];
    }
    b_out0[u.o4 + row * N + col] = acc;
  }
}

// 2. Tiled 16x16 FP16 GEMM + Epilogue: Y[M, N] = X_f16[M, K] @ W_f16[N, K]^T
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_f16_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let epilogue = u.p3;
  let has_bias = u.p4;
  let has_res = u.p5;
  let wgx = wg.x + u.pad0;
  let wgy = wg.y + u.pad1;
  let row = wgy * 16u + lid.y;
  let col = wgx * 16u + lid.x;

  var acc: f32 = 0.0;
  let n_tiles = (K + 31u) / 32u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 32u + lid.x * 2u;
    if (row < M && k0 < K) {
      let idx0 = row * K + k0;
      tileA[lid.y * 32u + lid.x * 2u] = f16_at_in0(idx0);
      tileA[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, f16_at_in0(idx0 + 1u), k0 + 1u < K);
    } else {
      tileA[lid.y * 32u + lid.x * 2u] = 0.0;
      tileA[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }

    let b_row = wgx * 16u + lid.y;
    if (b_row < N && k0 < K) {
      let bidx0 = b_row * K + k0;
      tileB[lid.y * 32u + lid.x * 2u] = f16_at_in1(bidx0);
      tileB[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, f16_at_in1(bidx0 + 1u), k0 + 1u < K);
    } else {
      tileB[lid.y * 32u + lid.x * 2u] = 0.0;
      tileB[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
      acc = fma(tileA[lid.y * 32u + kk], tileB[lid.x * 32u + kk], acc);
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    if (has_bias != 0u && epilogue >= 1u) {
      acc = acc + b_in2[u.o2 + col];
    }
    if (epilogue == 2u) {
      acc = silu(acc);
    } else if (epilogue == 3u) {
      acc = gelu_erf(acc);
    } else if (epilogue == 4u && has_res != 0u) {
      acc = acc + b_in3[u.o3 + row * N + col];
    }
    b_out0[u.o4 + row * N + col] = acc;
  }
}

// 3. Quantize BF16 [rows, cols] (cols % 64 == 0) -> Q8_G64 (b_out1=dst_q8 u32, b_out0=dst_sc f32)
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

// 4. Tiled 16x16 Q8_G64 GEMM
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_q8g64_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let wgx = wg.x + u.pad0;
  let wgy = wg.y + u.pad1;
  let row = wgy * 16u + lid.y;
  let col = wgx * 16u + lid.x;
  let groups = K >> 6u;

  var acc: f32 = 0.0;
  for (var g: u32 = 0u; g < groups; g = g + 1u) {
    for (var half_g: u32 = 0u; half_g < 2u; half_g = half_g + 1u) {
      let c_base = g * 64u + half_g * 32u + lid.x * 2u;
      if (row < M) {
        let xsc = b_in2[u.o2 + row * groups + g];
        tileA[lid.y * 32u + lid.x * 2u] = f32(s8_at_in0(row * K + c_base)) * xsc;
        tileA[lid.y * 32u + lid.x * 2u + 1u] = f32(s8_at_in0(row * K + c_base + 1u)) * xsc;
      } else {
        tileA[lid.y * 32u + lid.x * 2u] = 0.0;
        tileA[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
      }

      let b_row = wgx * 16u + lid.y;
      if (b_row < N) {
        let wsc = b_in3[u.o3 + b_row * groups + g];
        tileB[lid.y * 32u + lid.x * 2u] = f32(s8_at_in1(b_row * K + c_base)) * wsc;
        tileB[lid.y * 32u + lid.x * 2u + 1u] = f32(s8_at_in1(b_row * K + c_base + 1u)) * wsc;
      } else {
        tileB[lid.y * 32u + lid.x * 2u] = 0.0;
        tileB[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
      }
      workgroupBarrier();

      for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
        acc = fma(tileA[lid.y * 32u + kk], tileB[lid.x * 32u + kk], acc);
      }
      workgroupBarrier();
    }
  }

  if (row < M && col < N) {
    if (has_bias != 0u) {
      acc = acc + bitcast<f32>(b_out1[u.o5 + col]);
    }
    b_out0[u.o4 + row * N + col] = acc;
  }
}

// 5. Zero-Spill Coalesced 64-Thread Workgroup FlashAttention-2 BF16 (1 workgroup per (qpos, hq) row!)
// o0=Q_u16_off, o1=K_u16_off, o5=V_u16_off, o4=O_f32_off
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

  for (var kpos: u32 = k_start; kpos < k_end; kpos = kpos + 1u) {
    let k_off = kv_head_off + kpos * D;
    var part: f32 = q0 * bf16_at_in1(k_off + d0);
    if (D == 128u) {
      part = fma(q1, bf16_at_in1(k_off + d1), part);
    }
    tileA[tid] = part;
    workgroupBarrier();
    if (tid < 8u) {
      tileA[tid] = tileA[tid] + tileA[tid + 8u] + tileA[tid + 16u] + tileA[tid + 24u] +
                   tileA[tid + 32u] + tileA[tid + 40u] + tileA[tid + 48u] + tileA[tid + 56u];
    }
    workgroupBarrier();
    if (tid == 0u) {
      tileA[0] = (tileA[0] + tileA[1] + tileA[2] + tileA[3] +
                  tileA[4] + tileA[5] + tileA[6] + tileA[7]) * scale;
    }
    workgroupBarrier();
    let sc = tileA[0];
    let nm = max(m_max, sc);
    let alpha = exp(m_max - nm);
    let p = exp(sc - nm);
    l_sum = l_sum * alpha + p;
    m_max = nm;
    acc0 = acc0 * alpha + p * bf16_at_out1(k_off + d0);
    if (D == 128u) {
      acc1 = acc1 * alpha + p * bf16_at_out1(k_off + d1);
    }
    workgroupBarrier();
  }

  let inv = select(0.0, 1.0 / l_sum, l_sum > 0.0);
  let o_off = u.o4 + (qpos * Hq + hq) * D;
  b_out0[o_off + d0] = acc0 * inv;
  if (D == 128u) {
    b_out0[o_off + d1] = acc1 * inv;
  }
}

// 6. Zero-Spill Coalesced 64-Thread Workgroup FlashAttention-2 FP16 (1 workgroup per (qpos, h) row!)
// o0=Q_u16_off, o1=K_u16_off, o5=V_u16_off, o4=O_f32_off
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
  if (bcs > 0u) {
    k_limit = min(n_kv, ((qpos / bcs) + 1u) * bcs);
  }

  var m_max: f32 = -1e30;
  var l_sum: f32 = 0.0;
  let kv_head_off = h * n_kv_pad * D;

  for (var kpos: u32 = 0u; kpos < k_limit; kpos = kpos + 1u) {
    let k_off = kv_head_off + kpos * D;
    var part: f32 = q0 * f16_at_in1(k_off + d0);
    if (D == 128u) {
      part = fma(q1, f16_at_in1(k_off + d1), part);
    }
    tileA[tid] = part;
    workgroupBarrier();
    if (tid < 8u) {
      tileA[tid] = tileA[tid] + tileA[tid + 8u] + tileA[tid + 16u] + tileA[tid + 24u] +
                   tileA[tid + 32u] + tileA[tid + 40u] + tileA[tid + 48u] + tileA[tid + 56u];
    }
    workgroupBarrier();
    if (tid == 0u) {
      tileA[0] = (tileA[0] + tileA[1] + tileA[2] + tileA[3] +
                  tileA[4] + tileA[5] + tileA[6] + tileA[7]) * scale;
    }
    workgroupBarrier();
    let sc = tileA[0];
    let nm = max(m_max, sc);
    let alpha = exp(m_max - nm);
    let p = exp(sc - nm);
    l_sum = l_sum * alpha + p;
    m_max = nm;
    acc0 = acc0 * alpha + p * f16_at_out1(k_off + d0);
    if (D == 128u) {
      acc1 = acc1 * alpha + p * f16_at_out1(k_off + d1);
    }
    workgroupBarrier();
  }

  let inv = select(0.0, 1.0 / l_sum, l_sum > 0.0);
  let o_off = u.o4 + (qpos * H + h) * D;
  b_out0[o_off + d0] = acc0 * inv;
  if (D == 128u) {
    b_out0[o_off + d1] = acc1 * inv;
  }
}

// 7. Implicit-GEMM NHWC 3x3 Conv2D BF16
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
  let wgx = wg.x + u.pad0;
  let wgy = wg.y + u.pad1;

  let m = wgy * 16u + lid.y;
  let oc = wgx * 16u + lid.x;
  let oy = i32(m) / W;
  let ox = i32(m) - oy * W;

  var acc: f32 = 0.0;
  let n_tiles = (K + 31u) / 32u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 32u + lid.x * 2u;
    for (var s: u32 = 0u; s < 2u; s = s + 1u) {
      let kk = k0 + s;
      var xv: f32 = 0.0;
      if (m < M && kk < K) {
        let tap = kk / Cin;
        let ic = kk % Cin;
        let iy = oy + i32(tap / 3u) - 1;
        let ix = ox + i32(tap % 3u) - 1;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
          xv = bf16_at_in0((u32(iy * W + ix) * Cin) + ic);
        }
      }
      tileA[lid.y * 32u + lid.x * 2u + s] = xv;
    }

    let b_oc = wgx * 16u + lid.y;
    if (b_oc < Cout && k0 < K) {
      let widx0 = b_oc * K + k0;
      tileB[lid.y * 32u + lid.x * 2u] = bf16_at_in1(widx0);
      tileB[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, bf16_at_in1(widx0 + 1u), k0 + 1u < K);
    } else {
      tileB[lid.y * 32u + lid.x * 2u] = 0.0;
      tileB[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
      acc = fma(tileA[lid.y * 32u + kk], tileB[lid.x * 32u + kk], acc);
    }
    workgroupBarrier();
  }

  if (m < M && oc < Cout) {
    let off = m * Cout + oc;
    if (has_bias != 0u) { acc = acc + b_in2[u.o2 + oc]; }
    if (has_res != 0u)  { acc = acc + b_in3[u.o3 + off]; }
    b_out0[u.o4 + off] = acc;
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
  if (mode == 1u) {
    v0 = silu(v0);
    v1 = silu(v1);
  }
  if (mode == 2u) {
    b_out1[u.o5 + pair] = pack2x16float(vec2<f32>(v0, v1));
  } else {
    b_out1[u.o5 + pair] = f32_to_bf16_u16(v0) | (f32_to_bf16_u16(v1) << 16u);
  }
}

// 9. Tiled 16x16x32 INT8 W8A8 Per-Row/Col Scaled GEMM
@compute @workgroup_size(16, 16, 1)
fn vk_gemm_i8_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let M = u.p0;
  let N = u.p1;
  let K = u.p2;
  let has_bias = u.p3;
  let wgx = wg.x + u.pad0;
  let wgy = wg.y + u.pad1;
  let row = wgy * 16u + lid.y;
  let col = wgx * 16u + lid.x;

  var acc: f32 = 0.0;
  let n_tiles = (K + 31u) / 32u;
  for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
    let k0 = t * 32u + lid.x * 2u;
    if (row < M && k0 < K) {
      tileA[lid.y * 32u + lid.x * 2u] = f32(s8_at_in0(row * K + k0));
      tileA[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, f32(s8_at_in0(row * K + k0 + 1u)), k0 + 1u < K);
    } else {
      tileA[lid.y * 32u + lid.x * 2u] = 0.0;
      tileA[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }

    let b_row = wgx * 16u + lid.y;
    if (b_row < N && k0 < K) {
      tileB[lid.y * 32u + lid.x * 2u] = f32(s8_at_in1(b_row * K + k0));
      tileB[lid.y * 32u + lid.x * 2u + 1u] = select(0.0, f32(s8_at_in1(b_row * K + k0 + 1u)), k0 + 1u < K);
    } else {
      tileB[lid.y * 32u + lid.x * 2u] = 0.0;
      tileB[lid.y * 32u + lid.x * 2u + 1u] = 0.0;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
      acc = fma(tileA[lid.y * 32u + kk], tileB[lid.x * 32u + kk], acc);
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    var v = acc * (b_in2[u.o2 + row] * b_in3[u.o3 + col]);
    if (has_bias != 0u) {
      v = v + bitcast<f32>(b_out1[u.o5 + col]);
    }
    b_out0[u.o4 + row * N + col] = v;
  }
}
