// llb_vulkan_shaders.wgsl — Cross-Vendor Vulkan 1.2+ SPIR-V 1.3 LLM GEMV Kernels (via Naga)
// Runs on AMD Radeon (RADV/Windows), Intel Arc (ANV/Windows), NVIDIA Vulkan, Qualcomm, and Lavapipe.
// Coalesced 32-lanes-per-row (2 rows per 64-thread workgroup), zero subgroup extensions required.
// Exact bit-layout match to tools/qwen35_engine.c dot_fmt():
//   fmt 0 = F32 | fmt 1 = Q8_0 (row-split planar) | fmt 3 = Q4_K | fmt 4 = Q6_K | fmt 9 = Q5_K

struct Params {
  rows: u32,
  cols: u32,
  rb: u32,
  w_off: u32,     // 0..255 byte sub-offset from 256B-aligned VkDescriptorBufferInfo offset
  row_base: u32,  // row chunk offset for large vocabularies (e.g. V=248320 > 65535*2)
  p5: u32,
  p6: u32,
  p7: u32,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> b_w: array<u32>;
@group(0) @binding(2) var<storage, read> b_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_y: array<f32>;

var<workgroup> sm: array<f32, 64>;

fn u8_at(bo: u32) -> u32 {
  return (b_w[bo >> 2u] >> ((bo & 3u) * 8u)) & 0xffu;
}

fn s8_at(bo: u32) -> i32 {
  let sh = (3u - (bo & 3u)) * 8u;
  return (i32(b_w[bo >> 2u] << sh)) >> 24u;
}

fn u16_at(bo: u32) -> u32 {
  return u8_at(bo) | (u8_at(bo + 1u) << 8u);
}

fn h2f_at(bo: u32) -> f32 {
  return unpack2x16float(u16_at(bo)).x;
}

// ggml get_scale_min_k4: 6-bit scale/min from 12 packed bytes of Q4_K / Q5_K
fn gsm4(j: u32, sc_bo: u32) -> vec2<f32> {
  if (j < 4u) {
    let sd = u8_at(sc_bo + j) & 63u;
    let sm_v = u8_at(sc_bo + j + 4u) & 63u;
    return vec2<f32>(f32(sd), f32(sm_v));
  }
  let b4 = u8_at(sc_bo + j + 4u);
  let sd = (b4 & 0xfu) | ((u8_at(sc_bo + j - 4u) >> 6u) << 4u);
  let sm_v = (b4 >> 4u) | ((u8_at(sc_bo + j) >> 6u) << 4u);
  return vec2<f32>(f32(sd), f32(sm_v));
}

// 1. Q4_K (fmt 3): 144 bytes per 256-element super-block
@compute @workgroup_size(64, 1, 1)
fn vk_gemv_q4k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sub = lid.x >> 5u;
  let lane = lid.x & 31u;
  let row = u.row_base + wg.x * 2u + sub;

  var acc: f32 = 0.0;
  if (row < u.rows) {
    let row_bo = u.w_off + row * u.rb;
    let nblk = u.cols / 256u;
    for (var b: u32 = 0u; b < nblk; b = b + 1u) {
      let blk = row_bo + b * 144u;
      let d = h2f_at(blk);
      let dmin = h2f_at(blk + 2u);
      let sc_bo = blk + 4u;
      let qs_bo = blk + 16u;
      let xb = b * 256u;
      for (var g: u32 = 0u; g < 4u; g = g + 1u) {
        let p0 = gsm4(2u * g, sc_bo);
        let p1 = gsm4(2u * g + 1u, sc_bo);
        let d1 = d * p0.x;
        let m1 = dmin * p0.y;
        let d2 = d * p1.x;
        let m2 = dmin * p1.y;
        let qb = u8_at(qs_bo + g * 32u + lane);
        let xg = xb + g * 64u;
        acc = fma(d1 * f32(qb & 0xfu) - m1, b_x[xg + lane], acc);
        acc = fma(d2 * f32(qb >> 4u) - m2, b_x[xg + 32u + lane], acc);
      }
    }
  }

  let base = sub * 32u;
  sm[base + lane] = acc;
  workgroupBarrier();
  if (lane < 16u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 4u]; }
  workgroupBarrier();
  if (lane == 0u && row < u.rows) {
    b_y[row] = sm[base] + sm[base + 1u] + sm[base + 2u] + sm[base + 3u];
  }
}

// 2. Q5_K (fmt 9): 176 bytes per 256-element super-block
@compute @workgroup_size(64, 1, 1)
fn vk_gemv_q5k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sub = lid.x >> 5u;
  let lane = lid.x & 31u;
  let row = u.row_base + wg.x * 2u + sub;

  var acc: f32 = 0.0;
  if (row < u.rows) {
    let row_bo = u.w_off + row * u.rb;
    let nblk = u.cols / 256u;
    for (var b: u32 = 0u; b < nblk; b = b + 1u) {
      let blk = row_bo + b * 176u;
      let d = h2f_at(blk);
      let dmin = h2f_at(blk + 2u);
      let sc_bo = blk + 4u;
      let qh = u8_at(blk + 16u + lane);
      let qs_bo = blk + 48u;
      let xb = b * 256u;
      for (var g: u32 = 0u; g < 4u; g = g + 1u) {
        let p0 = gsm4(2u * g, sc_bo);
        let p1 = gsm4(2u * g + 1u, sc_bo);
        let d1 = d * p0.x;
        let m1 = dmin * p0.y;
        let d2 = d * p1.x;
        let m2 = dmin * p1.y;
        let qb = u8_at(qs_bo + g * 32u + lane);
        let q0 = (qb & 0xfu) + (((qh >> (2u * g)) & 1u) << 4u);
        let q1 = (qb >> 4u) + (((qh >> (2u * g + 1u)) & 1u) << 4u);
        let xg = xb + g * 64u;
        acc = fma(d1 * f32(q0) - m1, b_x[xg + lane], acc);
        acc = fma(d2 * f32(q1) - m2, b_x[xg + 32u + lane], acc);
      }
    }
  }

  let base = sub * 32u;
  sm[base + lane] = acc;
  workgroupBarrier();
  if (lane < 16u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 4u]; }
  workgroupBarrier();
  if (lane == 0u && row < u.rows) {
    b_y[row] = sm[base] + sm[base + 1u] + sm[base + 2u] + sm[base + 3u];
  }
}

// 3. Q6_K (fmt 4): 210 bytes per 256-element super-block
@compute @workgroup_size(64, 1, 1)
fn vk_gemv_q6k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sub = lid.x >> 5u;
  let lane = lid.x & 31u;
  let row = u.row_base + wg.x * 2u + sub;

  var acc: f32 = 0.0;
  if (row < u.rows) {
    let row_bo = u.w_off + row * u.rb;
    let nblk = u.cols / 256u;
    let l_hi = lane >> 4u;
    for (var b: u32 = 0u; b < nblk; b = b + 1u) {
      let blk = row_bo + b * 210u;
      let d = h2f_at(blk + 208u);
      let xb = b * 256u;
      for (var half: u32 = 0u; half < 2u; half = half + 1u) {
        let qlh = blk + half * 64u;
        let qhh = blk + 128u + half * 32u;
        let sch = blk + 192u + half * 8u;
        let ql0 = u8_at(qlh + lane);
        let ql1 = u8_at(qlh + 32u + lane);
        let qhv = u8_at(qhh + lane);
        let xh = xb + half * 128u + lane;

        let c0 = i32((ql0 & 0xfu) | ((qhv & 3u) << 4u)) - 32;
        let c1 = i32((ql1 & 0xfu) | (((qhv >> 2u) & 3u) << 4u)) - 32;
        let c2 = i32((ql0 >> 4u)  | (((qhv >> 4u) & 3u) << 4u)) - 32;
        let c3 = i32((ql1 >> 4u)  | (((qhv >> 6u) & 3u) << 4u)) - 32;

        let s0 = d * f32(s8_at(sch + 0u + l_hi));
        let s1 = d * f32(s8_at(sch + 2u + l_hi));
        let s2 = d * f32(s8_at(sch + 4u + l_hi));
        let s3 = d * f32(s8_at(sch + 6u + l_hi));

        acc = fma(s0 * f32(c0), b_x[xh + 0u],  acc);
        acc = fma(s1 * f32(c1), b_x[xh + 32u], acc);
        acc = fma(s2 * f32(c2), b_x[xh + 64u], acc);
        acc = fma(s3 * f32(c3), b_x[xh + 96u], acc);
      }
    }
  }

  let base = sub * 32u;
  sm[base + lane] = acc;
  workgroupBarrier();
  if (lane < 16u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 4u]; }
  workgroupBarrier();
  if (lane == 0u && row < u.rows) {
    b_y[row] = sm[base] + sm[base + 1u] + sm[base + 2u] + sm[base + 3u];
  }
}

// 4. Q8_0 (fmt 1): row-split planar layout — q[0..cols-1] followed by f16 d[0..cols/32-1]
@compute @workgroup_size(64, 1, 1)
fn vk_gemv_q8_0(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sub = lid.x >> 5u;
  let lane = lid.x & 31u;
  let row = u.row_base + wg.x * 2u + sub;

  var acc: f32 = 0.0;
  if (row < u.rows) {
    let row_bo = u.w_off + row * u.rb;
    let d_bo = row_bo + u.cols;
    let nblk = u.cols / 32u;
    for (var b: u32 = 0u; b < nblk; b = b + 1u) {
      let dd = h2f_at(d_bo + 2u * b);
      let qv = f32(s8_at(row_bo + b * 32u + lane));
      acc = fma(qv * dd, b_x[b * 32u + lane], acc);
    }
  }

  let base = sub * 32u;
  sm[base + lane] = acc;
  workgroupBarrier();
  if (lane < 16u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 4u]; }
  workgroupBarrier();
  if (lane == 0u && row < u.rows) {
    b_y[row] = sm[base] + sm[base + 1u] + sm[base + 2u] + sm[base + 3u];
  }
}

// 5. F32 (fmt 0): contiguous 4 bytes/elem
@compute @workgroup_size(64, 1, 1)
fn vk_gemv_f32(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sub = lid.x >> 5u;
  let lane = lid.x & 31u;
  let row = u.row_base + wg.x * 2u + sub;

  var acc: f32 = 0.0;
  if (row < u.rows) {
    let w0 = (u.w_off + row * u.rb) >> 2u;
    var c = lane;
    loop {
      if (c >= u.cols) { break; }
      acc = fma(bitcast<f32>(b_w[w0 + c]), b_x[c], acc);
      c = c + 32u;
    }
  }

  let base = sub * 32u;
  sm[base + lane] = acc;
  workgroupBarrier();
  if (lane < 16u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { sm[base + lane] = sm[base + lane] + sm[base + lane + 4u]; }
  workgroupBarrier();
  if (lane == 0u && row < u.rows) {
    b_y[row] = sm[base] + sm[base + 1u] + sm[base + 2u] + sm[base + 3u];
  }
}
