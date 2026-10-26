// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"convh_c","entry":"cube_convh_c_main","n":256,"len":8192,"lanes":256,"node":"f32","per":"row","ins":[{"name":"xb","ty":"f32","cls":"row","binding":0},{"name":"hb","ty":"f32","cls":"row","binding":1},{"name":"wb","ty":"f32","cls":"shared","binding":2}],"out":{"ty":"f32","binding":3},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":4}},{"kind":"cube","name":"gdn_c","entry":"cube_gdn_c_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"sb","ty":"f32","cls":"row","binding":0},{"name":"kb","ty":"f32","cls":"row","binding":1},{"name":"db","ty":"f32","cls":"row","binding":2},{"name":"vb","ty":"f32","cls":"row","binding":3},{"name":"bb","ty":"f32","cls":"row","binding":4},{"name":"qb","ty":"f32","cls":"row","binding":5}],"out":{"ty":"f32","binding":6},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":7}},{"kind":"cube","name":"convh_ci","entry":"cube_convh_ci_main","n":256,"len":8192,"lanes":256,"node":"f32","per":"row","ins":[{"name":"xb","ty":"f32","cls":"row","binding":0},{"name":"wb","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":3},"inplace":"out2"},{"kind":"cube","name":"gdn_ci","entry":"cube_gdn_ci_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"kb","ty":"f32","cls":"row","binding":0},{"name":"db","ty":"f32","cls":"row","binding":1},{"name":"vb","ty":"f32","cls":"row","binding":2},{"name":"bb","ty":"f32","cls":"row","binding":3},{"name":"qb","ty":"f32","cls":"row","binding":4}],"out":{"ty":"f32","binding":5},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":6},"inplace":"out"}]}
// webprefill2 recurrent chunk twins (bin/gen-recur.mjs SG from k_fuse_t.wgsl convh/gdn)
enable subgroups;
const EPS: f32 = 1.0e-6;

// convh_c (webprefill2): convh for the chunk's nt tokens in order; history in registers (h0 = oldest)
@group(0) @binding(0) var<storage, read> cube_convh_c_xb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_convh_c_hb: array<f32>;
@group(0) @binding(2) var<storage, read> cube_convh_c_wb: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_convh_c__out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_convh_c__out2: array<f32>;
@compute @workgroup_size(256)
fn cube_convh_c_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let i = g.x * 256u + l;
  let w = i * 4u;
  let nt = arrayLength(&cube_convh_c_xb) / 8192u;   // the x binding holds exactly the chunk's nt token slices
  var h0 = cube_convh_c_hb[i]; var h1 = cube_convh_c_hb[8192u + i]; var h2 = cube_convh_c_hb[16384u + i];
  for (var t = 0u; t < nt; t += 1u) {
    let xt = cube_convh_c_xb[t * 8192u + i];
    let v = (h0 * cube_convh_c_wb[w] + h1 * cube_convh_c_wb[w + 1u])
          + (h2 * cube_convh_c_wb[w + 2u] + xt * cube_convh_c_wb[w + 3u]);
    cube_convh_c__out[t * 8192u + i] = v / (1.0 + exp(-v));
    h0 = h1; h1 = h2; h2 = xt;
  }
  cube_convh_c__out2[i] = h0;
  cube_convh_c__out2[8192u + i] = h1;
  cube_convh_c__out2[16384u + i] = h2;
}
// gdn_c (webprefill2): gdn for the chunk's nt tokens in order; S[row0+r][l] in registers
@group(0) @binding(0) var<storage, read> cube_gdn_c_sb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_gdn_c_kb: array<f32>;
@group(0) @binding(2) var<storage, read> cube_gdn_c_db: array<f32>;
@group(0) @binding(3) var<storage, read> cube_gdn_c_vb: array<f32>;
@group(0) @binding(4) var<storage, read> cube_gdn_c_bb: array<f32>;
@group(0) @binding(5) var<storage, read> cube_gdn_c_qb: array<f32>;
@group(0) @binding(6) var<storage, read_write> cube_gdn_c__out: array<f32>;
@group(0) @binding(7) var<storage, read_write> cube_gdn_c__out2: array<f32>;
var<workgroup> cube_gdn_c_gt1: array<f32, 512>;
var<workgroup> cube_gdn_c_gt2: array<f32, 512>;
@compute @workgroup_size(128)
fn cube_gdn_c_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>, @builtin(subgroup_size) sgs: u32) {
  let row0 = (g.y * ng.x + g.x) * 4u;
  let head = row0 >> 7u;
  let nt = arrayLength(&cube_gdn_c_qb) / 4096u;   // the q binding holds exactly the chunk's nt token slices
  var st: array<f32, 4>;
  for (var r = 0u; r < 4u; r += 1u) { st[r] = cube_gdn_c_sb[(row0 + r) * 128u + l]; }
  for (var t = 0u; t < nt; t += 1u) {
    let dec = cube_gdn_c_db[t * 64u + head]; let beta = cube_gdn_c_bb[t * 64u + head];
    let kk = cube_gdn_c_kb[t * 4096u + head * 128u + l]; let qq = cube_gdn_c_qb[t * 4096u + head * 128u + l];
    var s: array<f32, 4>;
    for (var r = 0u; r < 4u; r += 1u) { s[r] = st[r] * dec; cube_gdn_c_gt1[r * 128u + l] = s[r] * kk; }
    workgroupBarrier();
    for (var h = 64u; h > 16u; h >>= 1u) {
      if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_c_gt1[r * 128u + l] += cube_gdn_c_gt1[r * 128u + l + h]; } }
      workgroupBarrier();
    }
    if (sgs == 32u) {   // h = 16..1 inside subgroup 0 (l 0..31): lane l + h's partial by shuffle, the same add a + b; WGs of 128 = 4 subgroups
      for (var r = 0u; r < 4u; r += 1u) {   // every subgroup shuffles (uniform control flow); only subgroup 0's lane 0 result is kept
        var v = cube_gdn_c_gt1[r * 128u + l];
        for (var h = 16u; h > 0u; h >>= 1u) { v = v + subgroupShuffleDown(v, h); }
        if (l == 0u) { cube_gdn_c_gt1[r * 128u] = v; }
      }
    } else {
      for (var h = 16u; h > 0u; h >>= 1u) {
        if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_c_gt1[r * 128u + l] += cube_gdn_c_gt1[r * 128u + l + h]; } }
        workgroupBarrier();
      }
    }
    workgroupBarrier();
    for (var r = 0u; r < 4u; r += 1u) {
      let delta = (cube_gdn_c_vb[t * 8192u + row0 + r] - cube_gdn_c_gt1[r * 128u]) * beta;
      let ns = s[r] + kk * delta;
      st[r] = ns;
      cube_gdn_c_gt2[r * 128u + l] = ns * qq;
    }
    workgroupBarrier();
    for (var h = 64u; h > 16u; h >>= 1u) {
      if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_c_gt2[r * 128u + l] += cube_gdn_c_gt2[r * 128u + l + h]; } }
      workgroupBarrier();
    }
    if (sgs == 32u) {   // h = 16..1 inside subgroup 0 (l 0..31): lane l + h's partial by shuffle, the same add a + b; WGs of 128 = 4 subgroups
      for (var r = 0u; r < 4u; r += 1u) {   // every subgroup shuffles (uniform control flow); only subgroup 0's lane 0 result is kept
        var v = cube_gdn_c_gt2[r * 128u + l];
        for (var h = 16u; h > 0u; h >>= 1u) { v = v + subgroupShuffleDown(v, h); }
        if (l == 0u) { cube_gdn_c_gt2[r * 128u] = v; }
      }
    } else {
      for (var h = 16u; h > 0u; h >>= 1u) {
        if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_c_gt2[r * 128u + l] += cube_gdn_c_gt2[r * 128u + l + h]; } }
        workgroupBarrier();
      }
    }
    workgroupBarrier();
    if (l < 4u) { cube_gdn_c__out2[t * 4096u + row0 + l] = cube_gdn_c_gt2[l * 128u]; }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r += 1u) { cube_gdn_c__out[(row0 + r) * 128u + l] = st[r]; }
}

// convh_ci (webprefill2): convh for the chunk's nt tokens in order; history in registers (h0 = oldest); history in/out = out2 (in place)
@group(0) @binding(0) var<storage, read> cube_convh_ci_xb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_convh_ci_wb: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_convh_ci__out: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_convh_ci__out2: array<f32>;
@compute @workgroup_size(256)
fn cube_convh_ci_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let i = g.x * 256u + l;
  let w = i * 4u;
  let nt = arrayLength(&cube_convh_ci_xb) / 8192u;   // the x binding holds exactly the chunk's nt token slices
  var h0 = cube_convh_ci__out2[i]; var h1 = cube_convh_ci__out2[8192u + i]; var h2 = cube_convh_ci__out2[16384u + i];
  for (var t = 0u; t < nt; t += 1u) {
    let xt = cube_convh_ci_xb[t * 8192u + i];
    let v = (h0 * cube_convh_ci_wb[w] + h1 * cube_convh_ci_wb[w + 1u])
          + (h2 * cube_convh_ci_wb[w + 2u] + xt * cube_convh_ci_wb[w + 3u]);
    cube_convh_ci__out[t * 8192u + i] = v / (1.0 + exp(-v));
    h0 = h1; h1 = h2; h2 = xt;
  }
  cube_convh_ci__out2[i] = h0;
  cube_convh_ci__out2[8192u + i] = h1;
  cube_convh_ci__out2[16384u + i] = h2;
}
// gdn_ci (webprefill2): gdn for the chunk's nt tokens in order; S[row0+r][l] in registers; state in/out = out (in place)
@group(0) @binding(0) var<storage, read> cube_gdn_ci_kb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_gdn_ci_db: array<f32>;
@group(0) @binding(2) var<storage, read> cube_gdn_ci_vb: array<f32>;
@group(0) @binding(3) var<storage, read> cube_gdn_ci_bb: array<f32>;
@group(0) @binding(4) var<storage, read> cube_gdn_ci_qb: array<f32>;
@group(0) @binding(5) var<storage, read_write> cube_gdn_ci__out: array<f32>;
@group(0) @binding(6) var<storage, read_write> cube_gdn_ci__out2: array<f32>;
var<workgroup> cube_gdn_ci_gt1: array<f32, 512>;
var<workgroup> cube_gdn_ci_gt2: array<f32, 512>;
@compute @workgroup_size(128)
fn cube_gdn_ci_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>, @builtin(subgroup_size) sgs: u32) {
  let row0 = (g.y * ng.x + g.x) * 4u;
  let head = row0 >> 7u;
  let nt = arrayLength(&cube_gdn_ci_qb) / 4096u;   // the q binding holds exactly the chunk's nt token slices
  var st: array<f32, 4>;
  for (var r = 0u; r < 4u; r += 1u) { st[r] = cube_gdn_ci__out[(row0 + r) * 128u + l]; }
  for (var t = 0u; t < nt; t += 1u) {
    let dec = cube_gdn_ci_db[t * 64u + head]; let beta = cube_gdn_ci_bb[t * 64u + head];
    let kk = cube_gdn_ci_kb[t * 4096u + head * 128u + l]; let qq = cube_gdn_ci_qb[t * 4096u + head * 128u + l];
    var s: array<f32, 4>;
    for (var r = 0u; r < 4u; r += 1u) { s[r] = st[r] * dec; cube_gdn_ci_gt1[r * 128u + l] = s[r] * kk; }
    workgroupBarrier();
    for (var h = 64u; h > 16u; h >>= 1u) {
      if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_ci_gt1[r * 128u + l] += cube_gdn_ci_gt1[r * 128u + l + h]; } }
      workgroupBarrier();
    }
    if (sgs == 32u) {   // h = 16..1 inside subgroup 0 (l 0..31): lane l + h's partial by shuffle, the same add a + b; WGs of 128 = 4 subgroups
      for (var r = 0u; r < 4u; r += 1u) {   // every subgroup shuffles (uniform control flow); only subgroup 0's lane 0 result is kept
        var v = cube_gdn_ci_gt1[r * 128u + l];
        for (var h = 16u; h > 0u; h >>= 1u) { v = v + subgroupShuffleDown(v, h); }
        if (l == 0u) { cube_gdn_ci_gt1[r * 128u] = v; }
      }
    } else {
      for (var h = 16u; h > 0u; h >>= 1u) {
        if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_ci_gt1[r * 128u + l] += cube_gdn_ci_gt1[r * 128u + l + h]; } }
        workgroupBarrier();
      }
    }
    workgroupBarrier();
    for (var r = 0u; r < 4u; r += 1u) {
      let delta = (cube_gdn_ci_vb[t * 8192u + row0 + r] - cube_gdn_ci_gt1[r * 128u]) * beta;
      let ns = s[r] + kk * delta;
      st[r] = ns;
      cube_gdn_ci_gt2[r * 128u + l] = ns * qq;
    }
    workgroupBarrier();
    for (var h = 64u; h > 16u; h >>= 1u) {
      if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_ci_gt2[r * 128u + l] += cube_gdn_ci_gt2[r * 128u + l + h]; } }
      workgroupBarrier();
    }
    if (sgs == 32u) {   // h = 16..1 inside subgroup 0 (l 0..31): lane l + h's partial by shuffle, the same add a + b; WGs of 128 = 4 subgroups
      for (var r = 0u; r < 4u; r += 1u) {   // every subgroup shuffles (uniform control flow); only subgroup 0's lane 0 result is kept
        var v = cube_gdn_ci_gt2[r * 128u + l];
        for (var h = 16u; h > 0u; h >>= 1u) { v = v + subgroupShuffleDown(v, h); }
        if (l == 0u) { cube_gdn_ci_gt2[r * 128u] = v; }
      }
    } else {
      for (var h = 16u; h > 0u; h >>= 1u) {
        if (l < h) { for (var r = 0u; r < 4u; r += 1u) { cube_gdn_ci_gt2[r * 128u + l] += cube_gdn_ci_gt2[r * 128u + l + h]; } }
        workgroupBarrier();
      }
    }
    workgroupBarrier();
    if (l < 4u) { cube_gdn_ci__out2[t * 4096u + row0 + l] = cube_gdn_ci_gt2[l * 128u]; }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r += 1u) { cube_gdn_ci__out[(row0 + r) * 128u + l] = st[r]; }
}
