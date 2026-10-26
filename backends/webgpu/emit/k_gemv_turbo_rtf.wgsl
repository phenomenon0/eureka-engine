// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"q4k2560_r2t4","entry":"cube_q4k2560_r2t4_main","n":256,"len":160,"lanes":160,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"base","ty":"u32","at":1},{"name":"xs","ty":"u32","at":2},{"name":"os","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4}],"lens":[]},"rows_at":0,"tile":2,"gurt":[2,4]},{"kind":"cube","name":"q4k4096_r2t4","entry":"cube_q4k4096_r2t4_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"base","ty":"u32","at":1},{"name":"xs","ty":"u32","at":2},{"name":"os","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4}],"lens":[]},"rows_at":0,"tile":2,"gurt":[2,4]},{"kind":"cube","name":"q4k9216_r4t2","entry":"cube_q4k9216_r4t2_main","n":1024,"len":576,"lanes":144,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"base","ty":"u32","at":1},{"name":"xs","ty":"u32","at":2},{"name":"os","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4}],"lens":[]},"rows_at":0,"tile":4,"gurt":[4,2]},{"kind":"cube","name":"q5k2560_r2t8","entry":"cube_q5k2560_r2t8_main","n":256,"len":160,"lanes":160,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"base","ty":"u32","at":1},{"name":"xs","ty":"u32","at":2},{"name":"os","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4}],"lens":[]},"rows_at":0,"tile":2,"gurt":[2,8]},{"kind":"cube","name":"q5k4096_r2t4","entry":"cube_q5k4096_r2t4_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"base","ty":"u32","at":1},{"name":"xs","ty":"u32","at":2},{"name":"os","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4}],"lens":[]},"rows_at":0,"tile":2,"gurt":[2,4]}],"turbo":{"rows_per_wg":4,"subgroups":true},"rt":["q4k2560@2:4","q4k4096@2:4","q4k9216@4:2","q5k2560@2:8","q5k4096@2:4"]}
// webprefill2 R x T GEMV twins (bin/gen-rt.mjs from k_gemv_turbo_sg.wgsl): leaf/helper fns verbatim; per (token,row) the shipped sequence
enable subgroups;

fn sc12(h: vec4<u32>, k: u32) -> u32 {
  let w = select(select(h.y, h.z, k >= 4u), h.w, k >= 8u);
  return (w >> ((k & 3u) * 8u)) & 255u;
}

// Cube q4k2560_r2t4 (webprefill2): q4k2560 for 2 rows x 4 tokens a workgroup, 160 threads, 1 leaves/thread (as shipped)
@group(0) @binding(0) var<storage, read> cube_q4k2560_r2t4_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_q4k2560_r2t4_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cube_q4k2560_r2t4__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_q4k2560_r2t4__sc: array<u32>;
var<workgroup> cube_q4k2560_r2t4__p: array<f32, 128>;
fn leaf_q4k2560_r2t4(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 9u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_q4k2560_r2t4_wb[bv];
  let qsv = cube_q4k2560_r2t4_wb[bv + 1u + b * 2u + (l >> 4u)];
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
fn cube_q4k2560_r2t4_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 2u; let tg = g.y;
  if (row0 >= cube_q4k2560_r2t4__sc[0u]) {
    return;
  }
  let base = cube_q4k2560_r2t4__sc[1u]; let xs = cube_q4k2560_r2t4__sc[2u]; let os = cube_q4k2560_r2t4__sc[3u]; let nt = cube_q4k2560_r2t4__sc[4u];
  let rv0 = (base + (row0 + 0u) * 1440u) >> 4u;
  let rv1 = (base + (row0 + 1u) * 1440u) >> 4u;
  let xo0 = (tg * 4u + 0u) * xs;
  let xo1 = (tg * 4u + 1u) * xs;
  let xo2 = (tg * 4u + 2u) * xs;
  let xo3 = (tg * 4u + 3u) * xs;
  var a0_0 = 0.0;
  var a0_1 = 0.0;
  var a1_0 = 0.0;
  var a1_1 = 0.0;
  var a2_0 = 0.0;
  var a2_1 = 0.0;
  var a3_0 = 0.0;
  var a3_1 = 0.0;
  for (var j = 0u; j < 1u; j += 1u) {
    let i = l * 1u + j;
    {
      let x0 = cube_q4k2560_r2t4_xb[xo0 + i * 4u]; let x1 = cube_q4k2560_r2t4_xb[xo0 + i * 4u + 1u]; let x2 = cube_q4k2560_r2t4_xb[xo0 + i * 4u + 2u]; let x3 = cube_q4k2560_r2t4_xb[xo0 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a0_0 += leaf_q4k2560_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a0_1 += leaf_q4k2560_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k2560_r2t4_xb[xo1 + i * 4u]; let x1 = cube_q4k2560_r2t4_xb[xo1 + i * 4u + 1u]; let x2 = cube_q4k2560_r2t4_xb[xo1 + i * 4u + 2u]; let x3 = cube_q4k2560_r2t4_xb[xo1 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a1_0 += leaf_q4k2560_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a1_1 += leaf_q4k2560_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k2560_r2t4_xb[xo2 + i * 4u]; let x1 = cube_q4k2560_r2t4_xb[xo2 + i * 4u + 1u]; let x2 = cube_q4k2560_r2t4_xb[xo2 + i * 4u + 2u]; let x3 = cube_q4k2560_r2t4_xb[xo2 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a2_0 += leaf_q4k2560_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a2_1 += leaf_q4k2560_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k2560_r2t4_xb[xo3 + i * 4u]; let x1 = cube_q4k2560_r2t4_xb[xo3 + i * 4u + 1u]; let x2 = cube_q4k2560_r2t4_xb[xo3 + i * 4u + 2u]; let x3 = cube_q4k2560_r2t4_xb[xo3 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a3_0 += leaf_q4k2560_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a3_1 += leaf_q4k2560_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
  }
  let s0_0 = subgroupAdd(a0_0);
  let s0_1 = subgroupAdd(a0_1);
  let s1_0 = subgroupAdd(a1_0);
  let s1_1 = subgroupAdd(a1_1);
  let s2_0 = subgroupAdd(a2_0);
  let s2_1 = subgroupAdd(a2_1);
  let s3_0 = subgroupAdd(a3_0);
  let s3_1 = subgroupAdd(a3_1);
  if (sgi == 0u) {
    cube_q4k2560_r2t4__p[0u + sgid] = s0_0;
    cube_q4k2560_r2t4__p[16u + sgid] = s0_1;
    cube_q4k2560_r2t4__p[32u + sgid] = s1_0;
    cube_q4k2560_r2t4__p[48u + sgid] = s1_1;
    cube_q4k2560_r2t4__p[64u + sgid] = s2_0;
    cube_q4k2560_r2t4__p[80u + sgid] = s2_1;
    cube_q4k2560_r2t4__p[96u + sgid] = s3_0;
    cube_q4k2560_r2t4__p[112u + sgid] = s3_1;
  }
  workgroupBarrier();
  if (l < 8u) {
    let t = l / 2u; let r = l % 2u; let tk = tg * 4u + t;
    var s = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { s += cube_q4k2560_r2t4__p[l * 16u + k]; }
    if (tk < nt) { cube_q4k2560_r2t4__out[tk * os + row0 + r] = s; }
  }
}

// Cube q4k4096_r2t4 (webprefill2): q4k4096 for 2 rows x 4 tokens a workgroup, 256 threads, 1 leaves/thread (as shipped)
@group(0) @binding(0) var<storage, read> cube_q4k4096_r2t4_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_q4k4096_r2t4_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cube_q4k4096_r2t4__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_q4k4096_r2t4__sc: array<u32>;
var<workgroup> cube_q4k4096_r2t4__p: array<f32, 128>;
fn leaf_q4k4096_r2t4(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 9u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_q4k4096_r2t4_wb[bv];
  let qsv = cube_q4k4096_r2t4_wb[bv + 1u + b * 2u + (l >> 4u)];
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

@compute @workgroup_size(256)
fn cube_q4k4096_r2t4_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 2u; let tg = g.y;
  if (row0 >= cube_q4k4096_r2t4__sc[0u]) {
    return;
  }
  let base = cube_q4k4096_r2t4__sc[1u]; let xs = cube_q4k4096_r2t4__sc[2u]; let os = cube_q4k4096_r2t4__sc[3u]; let nt = cube_q4k4096_r2t4__sc[4u];
  let rv0 = (base + (row0 + 0u) * 2304u) >> 4u;
  let rv1 = (base + (row0 + 1u) * 2304u) >> 4u;
  let xo0 = (tg * 4u + 0u) * xs;
  let xo1 = (tg * 4u + 1u) * xs;
  let xo2 = (tg * 4u + 2u) * xs;
  let xo3 = (tg * 4u + 3u) * xs;
  var a0_0 = 0.0;
  var a0_1 = 0.0;
  var a1_0 = 0.0;
  var a1_1 = 0.0;
  var a2_0 = 0.0;
  var a2_1 = 0.0;
  var a3_0 = 0.0;
  var a3_1 = 0.0;
  for (var j = 0u; j < 1u; j += 1u) {
    let i = l * 1u + j;
    {
      let x0 = cube_q4k4096_r2t4_xb[xo0 + i * 4u]; let x1 = cube_q4k4096_r2t4_xb[xo0 + i * 4u + 1u]; let x2 = cube_q4k4096_r2t4_xb[xo0 + i * 4u + 2u]; let x3 = cube_q4k4096_r2t4_xb[xo0 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a0_0 += leaf_q4k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a0_1 += leaf_q4k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k4096_r2t4_xb[xo1 + i * 4u]; let x1 = cube_q4k4096_r2t4_xb[xo1 + i * 4u + 1u]; let x2 = cube_q4k4096_r2t4_xb[xo1 + i * 4u + 2u]; let x3 = cube_q4k4096_r2t4_xb[xo1 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a1_0 += leaf_q4k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a1_1 += leaf_q4k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k4096_r2t4_xb[xo2 + i * 4u]; let x1 = cube_q4k4096_r2t4_xb[xo2 + i * 4u + 1u]; let x2 = cube_q4k4096_r2t4_xb[xo2 + i * 4u + 2u]; let x3 = cube_q4k4096_r2t4_xb[xo2 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a2_0 += leaf_q4k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a2_1 += leaf_q4k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k4096_r2t4_xb[xo3 + i * 4u]; let x1 = cube_q4k4096_r2t4_xb[xo3 + i * 4u + 1u]; let x2 = cube_q4k4096_r2t4_xb[xo3 + i * 4u + 2u]; let x3 = cube_q4k4096_r2t4_xb[xo3 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a3_0 += leaf_q4k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a3_1 += leaf_q4k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
  }
  let s0_0 = subgroupAdd(a0_0);
  let s0_1 = subgroupAdd(a0_1);
  let s1_0 = subgroupAdd(a1_0);
  let s1_1 = subgroupAdd(a1_1);
  let s2_0 = subgroupAdd(a2_0);
  let s2_1 = subgroupAdd(a2_1);
  let s3_0 = subgroupAdd(a3_0);
  let s3_1 = subgroupAdd(a3_1);
  if (sgi == 0u) {
    cube_q4k4096_r2t4__p[0u + sgid] = s0_0;
    cube_q4k4096_r2t4__p[16u + sgid] = s0_1;
    cube_q4k4096_r2t4__p[32u + sgid] = s1_0;
    cube_q4k4096_r2t4__p[48u + sgid] = s1_1;
    cube_q4k4096_r2t4__p[64u + sgid] = s2_0;
    cube_q4k4096_r2t4__p[80u + sgid] = s2_1;
    cube_q4k4096_r2t4__p[96u + sgid] = s3_0;
    cube_q4k4096_r2t4__p[112u + sgid] = s3_1;
  }
  workgroupBarrier();
  if (l < 8u) {
    let t = l / 2u; let r = l % 2u; let tk = tg * 4u + t;
    var s = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { s += cube_q4k4096_r2t4__p[l * 16u + k]; }
    if (tk < nt) { cube_q4k4096_r2t4__out[tk * os + row0 + r] = s; }
  }
}

// Cube q4k9216_r4t2 (webprefill2): q4k9216 for 4 rows x 2 tokens a workgroup, 144 threads, 4 leaves/thread (as shipped)
@group(0) @binding(0) var<storage, read> cube_q4k9216_r4t2_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_q4k9216_r4t2_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cube_q4k9216_r4t2__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_q4k9216_r4t2__sc: array<u32>;
var<workgroup> cube_q4k9216_r4t2__p: array<f32, 128>;
fn leaf_q4k9216_r4t2(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 9u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_q4k9216_r4t2_wb[bv];
  let qsv = cube_q4k9216_r4t2_wb[bv + 1u + b * 2u + (l >> 4u)];
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

@compute @workgroup_size(144)
fn cube_q4k9216_r4t2_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 4u; let tg = g.y;
  if (row0 >= cube_q4k9216_r4t2__sc[0u]) {
    return;
  }
  let base = cube_q4k9216_r4t2__sc[1u]; let xs = cube_q4k9216_r4t2__sc[2u]; let os = cube_q4k9216_r4t2__sc[3u]; let nt = cube_q4k9216_r4t2__sc[4u];
  let rv0 = (base + (row0 + 0u) * 5184u) >> 4u;
  let rv1 = (base + (row0 + 1u) * 5184u) >> 4u;
  let rv2 = (base + (row0 + 2u) * 5184u) >> 4u;
  let rv3 = (base + (row0 + 3u) * 5184u) >> 4u;
  let xo0 = (tg * 2u + 0u) * xs;
  let xo1 = (tg * 2u + 1u) * xs;
  var a0_0 = 0.0;
  var a0_1 = 0.0;
  var a0_2 = 0.0;
  var a0_3 = 0.0;
  var a1_0 = 0.0;
  var a1_1 = 0.0;
  var a1_2 = 0.0;
  var a1_3 = 0.0;
  for (var j = 0u; j < 4u; j += 1u) {
    let i = l * 4u + j;
    {
      let x0 = cube_q4k9216_r4t2_xb[xo0 + i * 4u]; let x1 = cube_q4k9216_r4t2_xb[xo0 + i * 4u + 1u]; let x2 = cube_q4k9216_r4t2_xb[xo0 + i * 4u + 2u]; let x3 = cube_q4k9216_r4t2_xb[xo0 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a0_0 += leaf_q4k9216_r4t2(rv0, i, x0, x1, x2, x3, sx);
      a0_1 += leaf_q4k9216_r4t2(rv1, i, x0, x1, x2, x3, sx);
      a0_2 += leaf_q4k9216_r4t2(rv2, i, x0, x1, x2, x3, sx);
      a0_3 += leaf_q4k9216_r4t2(rv3, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q4k9216_r4t2_xb[xo1 + i * 4u]; let x1 = cube_q4k9216_r4t2_xb[xo1 + i * 4u + 1u]; let x2 = cube_q4k9216_r4t2_xb[xo1 + i * 4u + 2u]; let x3 = cube_q4k9216_r4t2_xb[xo1 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a1_0 += leaf_q4k9216_r4t2(rv0, i, x0, x1, x2, x3, sx);
      a1_1 += leaf_q4k9216_r4t2(rv1, i, x0, x1, x2, x3, sx);
      a1_2 += leaf_q4k9216_r4t2(rv2, i, x0, x1, x2, x3, sx);
      a1_3 += leaf_q4k9216_r4t2(rv3, i, x0, x1, x2, x3, sx);
    }
  }
  let s0_0 = subgroupAdd(a0_0);
  let s0_1 = subgroupAdd(a0_1);
  let s0_2 = subgroupAdd(a0_2);
  let s0_3 = subgroupAdd(a0_3);
  let s1_0 = subgroupAdd(a1_0);
  let s1_1 = subgroupAdd(a1_1);
  let s1_2 = subgroupAdd(a1_2);
  let s1_3 = subgroupAdd(a1_3);
  if (sgi == 0u) {
    cube_q4k9216_r4t2__p[0u + sgid] = s0_0;
    cube_q4k9216_r4t2__p[16u + sgid] = s0_1;
    cube_q4k9216_r4t2__p[32u + sgid] = s0_2;
    cube_q4k9216_r4t2__p[48u + sgid] = s0_3;
    cube_q4k9216_r4t2__p[64u + sgid] = s1_0;
    cube_q4k9216_r4t2__p[80u + sgid] = s1_1;
    cube_q4k9216_r4t2__p[96u + sgid] = s1_2;
    cube_q4k9216_r4t2__p[112u + sgid] = s1_3;
  }
  workgroupBarrier();
  if (l < 8u) {
    let t = l / 4u; let r = l % 4u; let tk = tg * 2u + t;
    var s = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { s += cube_q4k9216_r4t2__p[l * 16u + k]; }
    if (tk < nt) { cube_q4k9216_r4t2__out[tk * os + row0 + r] = s; }
  }
}

// Cube q5k2560_r2t8 (webprefill2): q5k2560 for 2 rows x 8 tokens a workgroup, 160 threads, 1 leaves/thread (as shipped)
@group(0) @binding(0) var<storage, read> cube_q5k2560_r2t8_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_q5k2560_r2t8_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cube_q5k2560_r2t8__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_q5k2560_r2t8__sc: array<u32>;
var<workgroup> cube_q5k2560_r2t8__p: array<f32, 256>;
fn leaf_q5k2560_r2t8(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 11u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_q5k2560_r2t8_wb[bv];
  let qhv = cube_q5k2560_r2t8_wb[bv + 1u + (l >> 4u)];
  let hb = b * 2u + hi;
let qsv = cube_q5k2560_r2t8_wb[bv + 3u + b * 2u + (l >> 4u)];
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
  let n = ((qsv >> vec4<u32>(sh)) & vec4<u32>(0x0F0F0F0Fu)) | (((qhv >> vec4<u32>(hb)) & vec4<u32>(0x01010101u)) << vec4<u32>(4u));
  let a = vec4<f32>(f32(n.x & 255u), f32((n.x >> 8u) & 255u), f32((n.x >> 16u) & 255u), f32(n.x >> 24u));
  let bq = vec4<f32>(f32(n.y & 255u), f32((n.y >> 8u) & 255u), f32((n.y >> 16u) & 255u), f32(n.y >> 24u));
  let cq = vec4<f32>(f32(n.z & 255u), f32((n.z >> 8u) & 255u), f32((n.z >> 16u) & 255u), f32(n.z >> 24u));
  let dq = vec4<f32>(f32(n.w & 255u), f32((n.w >> 8u) & 255u), f32((n.w >> 16u) & 255u), f32(n.w >> 24u));
  let dot = dot(a, x0) + dot(bq, x1) + dot(cq, x2) + dot(dq, x3);
  return fma(dd.x * f32(scv), dot, -(dd.y * f32(mnv)) * sx);
}

@compute @workgroup_size(160)
fn cube_q5k2560_r2t8_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 2u; let tg = g.y;
  if (row0 >= cube_q5k2560_r2t8__sc[0u]) {
    return;
  }
  let base = cube_q5k2560_r2t8__sc[1u]; let xs = cube_q5k2560_r2t8__sc[2u]; let os = cube_q5k2560_r2t8__sc[3u]; let nt = cube_q5k2560_r2t8__sc[4u];
  let rv0 = (base + (row0 + 0u) * 1760u) >> 4u;
  let rv1 = (base + (row0 + 1u) * 1760u) >> 4u;
  let xo0 = (tg * 8u + 0u) * xs;
  let xo1 = (tg * 8u + 1u) * xs;
  let xo2 = (tg * 8u + 2u) * xs;
  let xo3 = (tg * 8u + 3u) * xs;
  let xo4 = (tg * 8u + 4u) * xs;
  let xo5 = (tg * 8u + 5u) * xs;
  let xo6 = (tg * 8u + 6u) * xs;
  let xo7 = (tg * 8u + 7u) * xs;
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
  for (var j = 0u; j < 1u; j += 1u) {
    let i = l * 1u + j;
    {
      let x0 = cube_q5k2560_r2t8_xb[xo0 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo0 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo0 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo0 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a0_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a0_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo1 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo1 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo1 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo1 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a1_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a1_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo2 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo2 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo2 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo2 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a2_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a2_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo3 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo3 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo3 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo3 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a3_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a3_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo4 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo4 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo4 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo4 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a4_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a4_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo5 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo5 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo5 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo5 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a5_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a5_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo6 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo6 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo6 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo6 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a6_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a6_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k2560_r2t8_xb[xo7 + i * 4u]; let x1 = cube_q5k2560_r2t8_xb[xo7 + i * 4u + 1u]; let x2 = cube_q5k2560_r2t8_xb[xo7 + i * 4u + 2u]; let x3 = cube_q5k2560_r2t8_xb[xo7 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a7_0 += leaf_q5k2560_r2t8(rv0, i, x0, x1, x2, x3, sx);
      a7_1 += leaf_q5k2560_r2t8(rv1, i, x0, x1, x2, x3, sx);
    }
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
    cube_q5k2560_r2t8__p[0u + sgid] = s0_0;
    cube_q5k2560_r2t8__p[16u + sgid] = s0_1;
    cube_q5k2560_r2t8__p[32u + sgid] = s1_0;
    cube_q5k2560_r2t8__p[48u + sgid] = s1_1;
    cube_q5k2560_r2t8__p[64u + sgid] = s2_0;
    cube_q5k2560_r2t8__p[80u + sgid] = s2_1;
    cube_q5k2560_r2t8__p[96u + sgid] = s3_0;
    cube_q5k2560_r2t8__p[112u + sgid] = s3_1;
    cube_q5k2560_r2t8__p[128u + sgid] = s4_0;
    cube_q5k2560_r2t8__p[144u + sgid] = s4_1;
    cube_q5k2560_r2t8__p[160u + sgid] = s5_0;
    cube_q5k2560_r2t8__p[176u + sgid] = s5_1;
    cube_q5k2560_r2t8__p[192u + sgid] = s6_0;
    cube_q5k2560_r2t8__p[208u + sgid] = s6_1;
    cube_q5k2560_r2t8__p[224u + sgid] = s7_0;
    cube_q5k2560_r2t8__p[240u + sgid] = s7_1;
  }
  workgroupBarrier();
  if (l < 16u) {
    let t = l / 2u; let r = l % 2u; let tk = tg * 8u + t;
    var s = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { s += cube_q5k2560_r2t8__p[l * 16u + k]; }
    if (tk < nt) { cube_q5k2560_r2t8__out[tk * os + row0 + r] = s; }
  }
}

// Cube q5k4096_r2t4 (webprefill2): q5k4096 for 2 rows x 4 tokens a workgroup, 256 threads, 1 leaves/thread (as shipped)
@group(0) @binding(0) var<storage, read> cube_q5k4096_r2t4_wb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> cube_q5k4096_r2t4_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cube_q5k4096_r2t4__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_q5k4096_r2t4__sc: array<u32>;
var<workgroup> cube_q5k4096_r2t4__p: array<f32, 128>;
fn leaf_q5k4096_r2t4(rv: u32, i: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>, sx: f32) -> f32 {
  let cc = i * 16u;
  let bv = rv + (cc >> 8u) * 11u;
  let rr = cc & 255u;
  let b = rr >> 6u;
  let hi = (rr >> 5u) & 1u;
  let l = rr & 16u;
  let hdr = cube_q5k4096_r2t4_wb[bv];
  let qhv = cube_q5k4096_r2t4_wb[bv + 1u + (l >> 4u)];
  let hb = b * 2u + hi;
let qsv = cube_q5k4096_r2t4_wb[bv + 3u + b * 2u + (l >> 4u)];
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
  let n = ((qsv >> vec4<u32>(sh)) & vec4<u32>(0x0F0F0F0Fu)) | (((qhv >> vec4<u32>(hb)) & vec4<u32>(0x01010101u)) << vec4<u32>(4u));
  let a = vec4<f32>(f32(n.x & 255u), f32((n.x >> 8u) & 255u), f32((n.x >> 16u) & 255u), f32(n.x >> 24u));
  let bq = vec4<f32>(f32(n.y & 255u), f32((n.y >> 8u) & 255u), f32((n.y >> 16u) & 255u), f32(n.y >> 24u));
  let cq = vec4<f32>(f32(n.z & 255u), f32((n.z >> 8u) & 255u), f32((n.z >> 16u) & 255u), f32(n.z >> 24u));
  let dq = vec4<f32>(f32(n.w & 255u), f32((n.w >> 8u) & 255u), f32((n.w >> 16u) & 255u), f32(n.w >> 24u));
  let dot = dot(a, x0) + dot(bq, x1) + dot(cq, x2) + dot(dq, x3);
  return fma(dd.x * f32(scv), dot, -(dd.y * f32(mnv)) * sx);
}

@compute @workgroup_size(256)
fn cube_q5k4096_r2t4_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>,
  @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_id) sgid: u32, @builtin(num_subgroups) nsg: u32) {
  let row0 = g.x * 2u; let tg = g.y;
  if (row0 >= cube_q5k4096_r2t4__sc[0u]) {
    return;
  }
  let base = cube_q5k4096_r2t4__sc[1u]; let xs = cube_q5k4096_r2t4__sc[2u]; let os = cube_q5k4096_r2t4__sc[3u]; let nt = cube_q5k4096_r2t4__sc[4u];
  let rv0 = (base + (row0 + 0u) * 2816u) >> 4u;
  let rv1 = (base + (row0 + 1u) * 2816u) >> 4u;
  let xo0 = (tg * 4u + 0u) * xs;
  let xo1 = (tg * 4u + 1u) * xs;
  let xo2 = (tg * 4u + 2u) * xs;
  let xo3 = (tg * 4u + 3u) * xs;
  var a0_0 = 0.0;
  var a0_1 = 0.0;
  var a1_0 = 0.0;
  var a1_1 = 0.0;
  var a2_0 = 0.0;
  var a2_1 = 0.0;
  var a3_0 = 0.0;
  var a3_1 = 0.0;
  for (var j = 0u; j < 1u; j += 1u) {
    let i = l * 1u + j;
    {
      let x0 = cube_q5k4096_r2t4_xb[xo0 + i * 4u]; let x1 = cube_q5k4096_r2t4_xb[xo0 + i * 4u + 1u]; let x2 = cube_q5k4096_r2t4_xb[xo0 + i * 4u + 2u]; let x3 = cube_q5k4096_r2t4_xb[xo0 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a0_0 += leaf_q5k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a0_1 += leaf_q5k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k4096_r2t4_xb[xo1 + i * 4u]; let x1 = cube_q5k4096_r2t4_xb[xo1 + i * 4u + 1u]; let x2 = cube_q5k4096_r2t4_xb[xo1 + i * 4u + 2u]; let x3 = cube_q5k4096_r2t4_xb[xo1 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a1_0 += leaf_q5k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a1_1 += leaf_q5k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k4096_r2t4_xb[xo2 + i * 4u]; let x1 = cube_q5k4096_r2t4_xb[xo2 + i * 4u + 1u]; let x2 = cube_q5k4096_r2t4_xb[xo2 + i * 4u + 2u]; let x3 = cube_q5k4096_r2t4_xb[xo2 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a2_0 += leaf_q5k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a2_1 += leaf_q5k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
    {
      let x0 = cube_q5k4096_r2t4_xb[xo3 + i * 4u]; let x1 = cube_q5k4096_r2t4_xb[xo3 + i * 4u + 1u]; let x2 = cube_q5k4096_r2t4_xb[xo3 + i * 4u + 2u]; let x3 = cube_q5k4096_r2t4_xb[xo3 + i * 4u + 3u];
      let sx = dot(x0 + x1 + x2 + x3, vec4<f32>(1.0));
      a3_0 += leaf_q5k4096_r2t4(rv0, i, x0, x1, x2, x3, sx);
      a3_1 += leaf_q5k4096_r2t4(rv1, i, x0, x1, x2, x3, sx);
    }
  }
  let s0_0 = subgroupAdd(a0_0);
  let s0_1 = subgroupAdd(a0_1);
  let s1_0 = subgroupAdd(a1_0);
  let s1_1 = subgroupAdd(a1_1);
  let s2_0 = subgroupAdd(a2_0);
  let s2_1 = subgroupAdd(a2_1);
  let s3_0 = subgroupAdd(a3_0);
  let s3_1 = subgroupAdd(a3_1);
  if (sgi == 0u) {
    cube_q5k4096_r2t4__p[0u + sgid] = s0_0;
    cube_q5k4096_r2t4__p[16u + sgid] = s0_1;
    cube_q5k4096_r2t4__p[32u + sgid] = s1_0;
    cube_q5k4096_r2t4__p[48u + sgid] = s1_1;
    cube_q5k4096_r2t4__p[64u + sgid] = s2_0;
    cube_q5k4096_r2t4__p[80u + sgid] = s2_1;
    cube_q5k4096_r2t4__p[96u + sgid] = s3_0;
    cube_q5k4096_r2t4__p[112u + sgid] = s3_1;
  }
  workgroupBarrier();
  if (l < 8u) {
    let t = l / 2u; let r = l % 2u; let tk = tg * 4u + t;
    var s = 0.0;
    for (var k = 0u; k < nsg; k += 1u) { s += cube_q5k4096_r2t4__p[l * 16u + k]; }
    if (tk < nt) { cube_q5k4096_r2t4__out[tk * os + row0 + r] = s; }
  }
}
