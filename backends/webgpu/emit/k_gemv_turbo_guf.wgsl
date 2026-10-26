// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"gu_r1t8","entry":"cube_gu_r1t8_main","n":160,"len":160,"lanes":160,"node":"f32","per":"row","tile":1,"ins":[{"name":"wg","ty":"bytes","cls":"bytes","binding":0},{"name":"wu","ty":"bytes","cls":"bytes","binding":1},{"name":"xb","ty":"bytes","cls":"bytes","binding":2}],"out":{"ty":"f32","binding":3},"sc":{"binding":4,"scalars":[{"name":"bg","ty":"u32","at":1},{"name":"bu","ty":"u32","at":2},{"name":"xs","ty":"u32","at":3},{"name":"os","ty":"u32","at":4},{"name":"nt","ty":"u32","at":5}],"lens":[]},"rows_at":0,"gurt":[1,8]}],"turbo":{"rows_per_wg":4,"subgroups":true},"gurt":["1:8"]}
// webprefill2 gu R x T twins (bin/gen-gurt.mjs from k_gemv_turbo_sg.wgsl): leaf fns verbatim; per (token,row) the shipped gu sequence
enable subgroups;

fn sc12(h: vec4<u32>, k: u32) -> u32 {
  let w = select(select(h.y, h.z, k >= 4u), h.w, k >= 8u);
  return (w >> ((k & 3u) * 8u)) & 255u;
}

// Cube gu_r1t8 (webprefill2): gu for 1 rows x 8 tokens a workgroup, 160 threads (thread = leaf, as shipped)
@group(0) @binding(0) var<storage, read> cube_gu_r1t80_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_gu_r1t81_wb: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cube_gu_r1t8_xb: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cube_gu_r1t8__out: array<f32>;
@group(0) @binding(4) var<storage, read> cube_gu_r1t8__sc: array<u32>;
var<workgroup> cube_gu_r1t8__p: array<f32, 256>;
fn leaf_gu_r1t80(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 9u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_gu_r1t80_wb[bv];
  let qsv = cube_gu_r1t80_wb[bv + 1u + b * 2u + (l >> 4u)];
  let dd = unpack2x16float(hdr.x);
  let sb = b * 2u + hi;
  var scv: u32; var mnv: u32;
  if (sb < 4u) {
    scv = (sc12(hdr, sb)) & 63u; mnv = (sc12(hdr, sb + 4u)) & 63u;
  } else {
    scv = (sc12(hdr, sb + 4u) & 15u) | ((sc12(hdr, sb - 4u) >> 6u) << 4u);
    mnv = (sc12(hdr, sb + 4u) >> 4u) | ((sc12(hdr, sb) >> 6u) << 4u);
  }
  let sh = hi * 4u;
  let n = (qsv >> vec4<u32>(sh)) & vec4<u32>(0x0F0F0F0Fu);
  let a = vec4<f32>(f32(n.x & 255u), f32((n.x >> 8u) & 255u), f32((n.x >> 16u) & 255u), f32(n.x >> 24u));
  let bq = vec4<f32>(f32(n.y & 255u), f32((n.y >> 8u) & 255u), f32((n.y >> 16u) & 255u), f32(n.y >> 24u));
  let cq = vec4<f32>(f32(n.z & 255u), f32((n.z >> 8u) & 255u), f32((n.z >> 16u) & 255u), f32(n.z >> 24u));
  let dq = vec4<f32>(f32(n.w & 255u), f32((n.w >> 8u) & 255u), f32((n.w >> 16u) & 255u), f32(n.w >> 24u));
  let dot = dot(a, x0) + dot(bq, x1) + dot(cq, x2) + dot(dq, x3);
  return fma(dd.x * f32(scv), dot, -(dd.y * f32(mnv)) * sx);
}
fn leaf_gu_r1t81(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 9u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_gu_r1t81_wb[bv];
  let qsv = cube_gu_r1t81_wb[bv + 1u + b * 2u + (l >> 4u)];
  let dd = unpack2x16float(hdr.x);
  let sb = b * 2u + hi;
  var scv: u32; var mnv: u32;
  if (sb < 4u) {
    scv = (sc12(hdr, sb)) & 63u; mnv = (sc12(hdr, sb + 4u)) & 63u;
  } else {
    scv = (sc12(hdr, sb + 4u) & 15u) | ((sc12(hdr, sb - 4u) >> 6u) << 4u);
    mnv = (sc12(hdr, sb + 4u) >> 4u) | ((sc12(hdr, sb) >> 6u) << 4u);
  }
  let sh = hi * 4u;
  let n = (qsv >> vec4<u32>(sh)) & vec4<u32>(0x0F0F0F0Fu);
  let a = vec4<f32>(f32(n.x & 255u), f32((n.x >> 8u) & 255u), f32((n.x >> 16u) & 255u), f32(n.x >> 24u));
  let bq = vec4<f32>(f32(n.y & 255u), f32((n.y >> 8u) & 255u), f32((n.y >> 16u) & 255u), f32(n.y >> 24u));
  let cq = vec4<f32>(f32(n.z & 255u), f32((n.z >> 8u) & 255u), f32((n.z >> 16u) & 255u), f32(n.z >> 24u));
  let dq = vec4<f32>(f32(n.w & 255u), f32((n.w >> 8u) & 255u), f32((n.w >> 16u) & 255u), f32(n.w >> 24u));
  let dot = dot(a, x0) + dot(bq, x1) + dot(cq, x2) + dot(dq, x3);
  return fma(dd.x * f32(scv), dot, -(dd.y * f32(mnv)) * sx);
}

@compute @workgroup_size(160)
fn cube_gu_r1t8_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 1u; let tg = g.y;
  if (row0 >= cube_gu_r1t8__sc[0u]) {
    return;
  }
  let bg = cube_gu_r1t8__sc[1u]; let bu = cube_gu_r1t8__sc[2u]; let xs = cube_gu_r1t8__sc[3u]; let os = cube_gu_r1t8__sc[4u]; let nt = cube_gu_r1t8__sc[5u];
  var a0_0 = 0.0;
  var a0_1 = 0.0;
  var a1_0 = 0.0;
  var a1_1 = 0.0;
  var a2_0 = 0.0;
  var a2_1 = 0.0;
  var a3_0 = 0.0;
  var a3_1 = 0.0;
  var a4_0 = 0.0;
  var a4_1 = 0.0;
  var a5_0 = 0.0;
  var a5_1 = 0.0;
  var a6_0 = 0.0;
  var a6_1 = 0.0;
  var a7_0 = 0.0;
  var a7_1 = 0.0;
  let i = l;
  {
    let xo = (tg * 8u + 0u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a0_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a0_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 1u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a1_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a1_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 2u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a2_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a2_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 3u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a3_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a3_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 4u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a4_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a4_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 5u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a5_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a5_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 6u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a6_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a6_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  {
    let xo = (tg * 8u + 7u) * xs;
    let x0 = cube_gu_r1t8_xb[xo + i * 4u]; let x1 = cube_gu_r1t8_xb[xo + i * 4u + 1u]; let x2 = cube_gu_r1t8_xb[xo + i * 4u + 2u]; let x3 = cube_gu_r1t8_xb[xo + i * 4u + 3u];
    let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
    a7_0 += leaf_gu_r1t80((bg + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
    a7_1 += leaf_gu_r1t81((bu + (row0 + 0u) * 1440u) >> 4u, i, x0, x1, x2, x3, sx);
  }
  let s0_0 = subgroupAdd(a0_0);
  let s0_1 = subgroupAdd(a0_1);
  let s1_0 = subgroupAdd(a1_0);
  let s1_1 = subgroupAdd(a1_1);
  let s2_0 = subgroupAdd(a2_0);
  let s2_1 = subgroupAdd(a2_1);
  let s3_0 = subgroupAdd(a3_0);
  let s3_1 = subgroupAdd(a3_1);
  let s4_0 = subgroupAdd(a4_0);
  let s4_1 = subgroupAdd(a4_1);
  let s5_0 = subgroupAdd(a5_0);
  let s5_1 = subgroupAdd(a5_1);
  let s6_0 = subgroupAdd(a6_0);
  let s6_1 = subgroupAdd(a6_1);
  let s7_0 = subgroupAdd(a7_0);
  let s7_1 = subgroupAdd(a7_1);
  if (sgi == 0u) {
    cube_gu_r1t8__p[0u + sgid] = s0_0;
    cube_gu_r1t8__p[16u + sgid] = s0_1;
    cube_gu_r1t8__p[32u + sgid] = s1_0;
    cube_gu_r1t8__p[48u + sgid] = s1_1;
    cube_gu_r1t8__p[64u + sgid] = s2_0;
    cube_gu_r1t8__p[80u + sgid] = s2_1;
    cube_gu_r1t8__p[96u + sgid] = s3_0;
    cube_gu_r1t8__p[112u + sgid] = s3_1;
    cube_gu_r1t8__p[128u + sgid] = s4_0;
    cube_gu_r1t8__p[144u + sgid] = s4_1;
    cube_gu_r1t8__p[160u + sgid] = s5_0;
    cube_gu_r1t8__p[176u + sgid] = s5_1;
    cube_gu_r1t8__p[192u + sgid] = s6_0;
    cube_gu_r1t8__p[208u + sgid] = s6_1;
    cube_gu_r1t8__p[224u + sgid] = s7_0;
    cube_gu_r1t8__p[240u + sgid] = s7_1;
  }
  workgroupBarrier();
  if (l < 8u) {
    let t = l / 1u; let r = l % 1u; let tk = tg * 8u + t;
    var sg_ = 0.0; var su = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { sg_ += cube_gu_r1t8__p[(t * 2u + r) * 16u + k]; su += cube_gu_r1t8__p[(t * 2u + 1u + r) * 16u + k]; }
    if (tk < nt) { cube_gu_r1t8__out[tk * os + row0 + r] = (sg_ / (1.0 + exp(-sg_))) * su; }
  }
}
