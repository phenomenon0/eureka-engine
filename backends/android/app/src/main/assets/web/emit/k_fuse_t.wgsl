// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"gatesg","entry":"cube_gatesg_main","n":256,"len":32,"lanes":256,"node":"f32","per":"row","ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"f32","cls":"row","binding":1},{"name":"dt","ty":"f32","cls":"shared","binding":2},{"name":"sa","ty":"f32","cls":"shared","binding":3}],"out":{"ty":"f32","binding":4},"sc":{"binding":6,"scalars":[{"name":"bb","ty":"u32","at":1},{"name":"ba","ty":"u32","at":2}],"lens":[]},"rows_at":0,"out2":{"ty":"f32","binding":5}},{"kind":"cube","name":"qprep","entry":"cube_qprep_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"qx","ty":"f32","cls":"row","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"cube","name":"kvput","entry":"cube_kvput_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"cb","ty":"f32","cls":"row","binding":0},{"name":"vb","ty":"f32","cls":"row","binding":1},{"name":"kx","ty":"f32","cls":"row","binding":2},{"name":"vx","ty":"f32","cls":"row","binding":3},{"name":"w","ty":"f32","cls":"shared","binding":4}],"out":{"ty":"f32","binding":5},"sc":{"binding":7,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0,"out2":{"ty":"f32","binding":6}},{"kind":"cube","name":"attn","entry":"cube_attn_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"q","ty":"f32","cls":"row","binding":0},{"name":"cb","ty":"f32","cls":"row","binding":1},{"name":"vb","ty":"f32","cls":"row","binding":2},{"name":"qx","ty":"f32","cls":"row","binding":3}],"out":{"ty":"f32","binding":4},"sc":{"binding":5,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"cube","name":"convh","entry":"cube_convh_main","n":256,"len":8192,"lanes":256,"node":"f32","per":"row","ins":[{"name":"xb","ty":"f32","cls":"row","binding":0},{"name":"hb","ty":"f32","cls":"row","binding":1},{"name":"wb","ty":"f32","cls":"shared","binding":2}],"out":{"ty":"f32","binding":3},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":4}},{"kind":"cube","name":"gdn","entry":"cube_gdn_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"sb","ty":"f32","cls":"row","binding":0},{"name":"kb","ty":"f32","cls":"row","binding":1},{"name":"db","ty":"f32","cls":"row","binding":2},{"name":"vb","ty":"f32","cls":"row","binding":3},{"name":"bb","ty":"f32","cls":"row","binding":4},{"name":"qb","ty":"f32","cls":"row","binding":5}],"out":{"ty":"f32","binding":6},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":7}},{"kind":"cube","name":"addrms","entry":"cube_addrms_main","n":256,"len":2560,"lanes":256,"node":"f32","per":"row","ins":[{"name":"a","ty":"f32","cls":"row","binding":0},{"name":"b","ty":"f32","cls":"row","binding":1},{"name":"w","ty":"f32","cls":"shared","binding":2}],"out":{"ty":"f32","binding":3},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":4}},{"kind":"cube","name":"gates","entry":"cube_gates_main","n":256,"len":32,"lanes":256,"node":"f32","per":"row","ins":[{"name":"b0","ty":"f32","cls":"row","binding":0},{"name":"a0","ty":"f32","cls":"row","binding":1},{"name":"dt","ty":"f32","cls":"shared","binding":2},{"name":"sa","ty":"f32","cls":"shared","binding":3}],"out":{"ty":"f32","binding":4},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":5}},{"kind":"cube","name":"l2qk","entry":"cube_l2qk_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"c","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null,"out2":{"ty":"f32","binding":2}}]}
// TURBO fused small ops (bin/turbo-build.py): hardware math, two outputs each (manifest out2); NOT bit-exact
const EPS: f32 = 1.0e-6;
@group(0) @binding(0) var<storage, read> cube_addrms_a: array<f32>;
@group(0) @binding(1) var<storage, read> cube_addrms_b: array<f32>;
@group(0) @binding(2) var<storage, read> cube_addrms_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_addrms__out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_addrms__out2: array<f32>;
var<workgroup> red256: array<f32, 256>;
@compute @workgroup_size(256)
fn cube_addrms_main(@builtin(local_invocation_index) l: u32) {
  var sv: array<f32, 10>;
  var ss = 0.0;
  for (var k = 0u; k < 10u; k += 1u) {
    let i = l + k * 256u;
    let s = cube_addrms_a[i] + cube_addrms_b[i];
    sv[k] = s; cube_addrms__out[i] = s; ss += s * s;
  }
  red256[l] = ss;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (l < h) { red256[l] += red256[l + h]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(red256[0] / 2560.0 + EPS);
  for (var k = 0u; k < 10u; k += 1u) {
    let i = l + k * 256u;
    cube_addrms__out2[i] = (sv[k] * inv) * cube_addrms_w[i];
  }
}
// ---- full-attention block: qprep (qnorm + partial NEOX rope), kvput (knorm + rope + K/V cache put), attn (scores..weighted + output gate; online softmax over positions <= pos)
fn rope_head(l: u32, pos: f32) -> f32 {
  if (l < 64u) {
    let j = l >> 1u;
    let a = nh[j]; let b = nh[j + 32u];
    let theta = pos * pow(10000000.0, -f32(2u * j) / 64.0);
    let c = cos(theta); let s = sin(theta);
    return select(a * s + b * c, a * c - b * s, (l & 1u) == 0u);
  }
  return nh[l];
}
var<workgroup> nh: array<f32, 256>;
@group(0) @binding(0) var<storage, read> cube_qprep_qx: array<f32>;
@group(0) @binding(1) var<storage, read> cube_qprep_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_qprep__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_qprep__sc: array<u32>;
@compute @workgroup_size(256)
fn cube_qprep_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let hd = g.x;
  let x = cube_qprep_qx[hd * 512u + l];
  red256[l] = x * x;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (l < h) { red256[l] += red256[l + h]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(red256[0] / 256.0 + EPS);
  nh[l] = (x * inv) * cube_qprep_w[l];
  workgroupBarrier();
  cube_qprep__out[hd * 256u + l] = rope_head(l, f32(cube_qprep__sc[1u]));
}
@group(0) @binding(0) var<storage, read> cube_kvput_cb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_kvput_vb: array<f32>;
@group(0) @binding(2) var<storage, read> cube_kvput_kx: array<f32>;
@group(0) @binding(3) var<storage, read> cube_kvput_vx: array<f32>;
@group(0) @binding(4) var<storage, read> cube_kvput_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> cube_kvput__out: array<f32>;
@group(0) @binding(6) var<storage, read_write> cube_kvput__out2: array<f32>;
@group(0) @binding(7) var<storage, read> cube_kvput__sc: array<u32>;
@compute @workgroup_size(256)
fn cube_kvput_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let wg = g.y * ng.x + g.x;
  let i = wg * 256u + l;
  let pos = cube_kvput__sc[1u];
  if ((i >> 10u) != pos) {   // a workgroup covers 256 words of one slot: this branch is workgroup-uniform
    cube_kvput__out[i] = cube_kvput_cb[i];
    cube_kvput__out2[i] = cube_kvput_vb[i];
    return;
  }
  let hk = wg & 3u;   // no barriers (the slot test is not provably uniform): each thread recomputes the head's rms from global memory
  var ss = 0.0;
  for (var k = 0u; k < 256u; k += 1u) { let v = cube_kvput_kx[hk * 256u + k]; ss += v * v; }
  let inv = 1.0 / sqrt(ss / 256.0 + EPS);
  var o: f32;
  if (l < 64u) {
    let j = l >> 1u;
    let a = (cube_kvput_kx[hk * 256u + j] * inv) * cube_kvput_w[j];
    let b = (cube_kvput_kx[hk * 256u + j + 32u] * inv) * cube_kvput_w[j + 32u];
    let theta = f32(pos) * pow(10000000.0, -f32(2u * j) / 64.0);
    let c = cos(theta); let s = sin(theta);
    o = select(a * s + b * c, a * c - b * s, (l & 1u) == 0u);
  } else {
    o = (cube_kvput_kx[hk * 256u + l] * inv) * cube_kvput_w[l];
  }
  cube_kvput__out[i] = o;
  cube_kvput__out2[i] = cube_kvput_vx[i & 1023u];
}
@group(0) @binding(0) var<storage, read> cube_attn_q: array<f32>;
@group(0) @binding(1) var<storage, read> cube_attn_cb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cube_attn_vb: array<f32>;
@group(0) @binding(3) var<storage, read> cube_attn_qx: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_attn__out: array<f32>;
@group(0) @binding(5) var<storage, read> cube_attn__sc: array<u32>;
var<workgroup> aq4: array<vec4<f32>, 64>;
var<workgroup> apr: array<f32, 256>;
@compute @workgroup_size(256)
fn cube_attn_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let hd = g.x; let kvh = hd >> 2u; let pos = cube_attn__sc[1u];
  if (l < 64u) { let o = hd * 256u + l * 4u; aq4[l] = vec4<f32>(cube_attn_q[o], cube_attn_q[o + 1u], cube_attn_q[o + 2u], cube_attn_q[o + 3u]); }
  workgroupBarrier();
  var m = -3.0e38; var lsum = 0.0; var acc = 0.0;
  let nchunk = (pos + 256u) >> 8u;
  for (var c = 0u; c < nchunk; c += 1u) {
    let t = c * 256u + l;
    var s = -3.0e38;
    if (t <= pos) {
      let base = (t * 1024u + kvh * 256u) >> 2u;
      var a = 0.0;
      for (var d = 0u; d < 64u; d += 1u) { a += dot(cube_attn_cb[base + d], aq4[d]); }
      s = a * 0.0625;
    }
    apr[l] = s;
    workgroupBarrier();
    var cm = -3.0e38;
    let cnt = min(256u, pos + 1u - c * 256u);
    for (var i = 0u; i < cnt; i += 1u) { cm = max(cm, apr[i]); }
    workgroupBarrier();
    let mn = max(m, cm);
    let sc = exp(m - mn);
    apr[l] = select(0.0, exp(s - mn), t <= pos);
    workgroupBarrier();
    acc = acc * sc; lsum = lsum * sc;
    for (var i = 0u; i < cnt; i += 1u) {
      let p = apr[i];
      lsum += p;
      acc += p * cube_attn_vb[((c * 256u + i) * 1024u) + kvh * 256u + l];
    }
    workgroupBarrier();
    m = mn;
  }
  let gt = cube_attn_qx[hd * 512u + 256u + l];
  cube_attn__out[hd * 256u + l] = (acc / lsum) * (1.0 / (1.0 + exp(-gt)));
}
@group(0) @binding(0) var<storage, read> cube_convh_xb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_convh_hb: array<f32>;
@group(0) @binding(2) var<storage, read> cube_convh_wb: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_convh__out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_convh__out2: array<f32>;
@compute @workgroup_size(256)
fn cube_convh_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let i = g.x * 256u + l;
  let w = i * 4u;
  let v = (cube_convh_hb[i] * cube_convh_wb[w] + cube_convh_hb[8192u + i] * cube_convh_wb[w + 1u])
        + (cube_convh_hb[16384u + i] * cube_convh_wb[w + 2u] + cube_convh_xb[i] * cube_convh_wb[w + 3u]);
  cube_convh__out[i] = v / (1.0 + exp(-v));
  cube_convh__out2[i] = cube_convh_hb[8192u + i];
  cube_convh__out2[8192u + i] = cube_convh_hb[16384u + i];
  cube_convh__out2[16384u + i] = cube_convh_xb[i];
}
@group(0) @binding(0) var<storage, read> cube_gdn_sb: array<f32>;
@group(0) @binding(1) var<storage, read> cube_gdn_kb: array<f32>;
@group(0) @binding(2) var<storage, read> cube_gdn_db: array<f32>;
@group(0) @binding(3) var<storage, read> cube_gdn_vb: array<f32>;
@group(0) @binding(4) var<storage, read> cube_gdn_bb: array<f32>;
@group(0) @binding(5) var<storage, read> cube_gdn_qb: array<f32>;
@group(0) @binding(6) var<storage, read_write> cube_gdn__out: array<f32>;
@group(0) @binding(7) var<storage, read_write> cube_gdn__out2: array<f32>;
var<workgroup> gt1: array<f32, 512>;
var<workgroup> gt2: array<f32, 512>;
@compute @workgroup_size(128)
fn cube_gdn_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  // one workgroup = 4 consecutive value rows j of one head; thread l = key index i. S'[j,i] = S*dec + k*delta_j; delta_j = (v_j - sum_i S*dec*k) * beta; out2[j] = sum_i S'*q
  let row0 = (g.y * ng.x + g.x) * 4u;
  let head = row0 >> 7u;
  let dec = cube_gdn_db[head]; let beta = cube_gdn_bb[head];
  let kk = cube_gdn_kb[head * 128u + l]; let qq = cube_gdn_qb[head * 128u + l];
  var s: array<f32, 4>;
  for (var r = 0u; r < 4u; r += 1u) { s[r] = cube_gdn_sb[(row0 + r) * 128u + l] * dec; gt1[r * 128u + l] = s[r] * kk; }
  workgroupBarrier();
  for (var h = 64u; h > 0u; h >>= 1u) {
    if (l < h) { for (var r = 0u; r < 4u; r += 1u) { gt1[r * 128u + l] += gt1[r * 128u + l + h]; } }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r += 1u) {
    let delta = (cube_gdn_vb[row0 + r] - gt1[r * 128u]) * beta;
    let ns = s[r] + kk * delta;
    cube_gdn__out[(row0 + r) * 128u + l] = ns;
    gt2[r * 128u + l] = ns * qq;
  }
  workgroupBarrier();
  for (var h = 64u; h > 0u; h >>= 1u) {
    if (l < h) { for (var r = 0u; r < 4u; r += 1u) { gt2[r * 128u + l] += gt2[r * 128u + l + h]; } }
    workgroupBarrier();
  }
  if (l < 4u) { cube_gdn__out2[row0 + l] = gt2[l * 128u]; }
}
// gatesg: the two Q8_0 projections (beta0 = sb.h, alpha0 = sa.h; 32 rows each, K=2560, planar rows: 2560 int8 then 80 f16 block scales) + the gate math
@group(0) @binding(0) var<storage, read> cube_gatesg_wb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_gatesg_xb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cube_gatesg_dt: array<f32>;
@group(0) @binding(3) var<storage, read> cube_gatesg_sa: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_gatesg__out: array<f32>;
@group(0) @binding(5) var<storage, read_write> cube_gatesg__out2: array<f32>;
@group(0) @binding(6) var<storage, read> cube_gatesg__sc: array<u32>;
var<workgroup> gg_part: array<f32, 256>;
var<workgroup> gg_row: array<f32, 64>;
@compute @workgroup_size(256)
fn cube_gatesg_main(@builtin(local_invocation_index) l: u32) {
  let row = l >> 2u; let part = l & 3u;                 // 64 rows x 4 parts of 640 weights
  let base = select(cube_gatesg__sc[1u], cube_gatesg__sc[2u], row >= 32u) + (row & 31u) * 2720u;
  var acc = 0.0;
  for (var bb = 0u; bb < 20u; bb += 1u) {
    let blk = part * 20u + bb;                          // 32-element block
    let sw = cube_gatesg_wb[(base + 2560u + blk * 2u) >> 2u];
    let d = unpack2x16float(sw).x;
    let dd = select(d, unpack2x16float(sw >> 16u).x, ((blk & 1u) == 1u));
    var dot_ = 0.0;
    for (var w = 0u; w < 8u; w += 1u) {
      let q = cube_gatesg_wb[(base >> 2u) + blk * 8u + w];
      let qv = vec4<f32>(f32(i32(q << 24u) >> 24u), f32(i32(q << 16u) >> 24u), f32(i32(q << 8u) >> 24u), f32(i32(q) >> 24u));
      dot_ += dot(qv, cube_gatesg_xb[blk * 8u + w]);
    }
    acc += dd * dot_;
  }
  gg_part[l] = acc;
  workgroupBarrier();
  if (part == 0u) { gg_row[row] = gg_part[l] + gg_part[l + 1u] + gg_part[l + 2u] + gg_part[l + 3u]; }
  workgroupBarrier();
  if (l < 32u) {
    cube_gatesg__out[l] = 1.0 / (1.0 + exp(-gg_row[l]));
    let adt = gg_row[32u + l] + cube_gatesg_dt[l];
    let sp = select(log(1.0 + exp(adt)), adt, adt > 20.0);
    cube_gatesg__out2[l] = exp(sp * cube_gatesg_sa[l]);
  }
}
@group(0) @binding(0) var<storage, read> cube_gates_b0: array<f32>;
@group(0) @binding(1) var<storage, read> cube_gates_a0: array<f32>;
@group(0) @binding(2) var<storage, read> cube_gates_dt: array<f32>;
@group(0) @binding(3) var<storage, read> cube_gates_sa: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_gates__out: array<f32>;
@group(0) @binding(5) var<storage, read_write> cube_gates__out2: array<f32>;
@compute @workgroup_size(256)
fn cube_gates_main(@builtin(local_invocation_index) l: u32) {
  if (l < 32u) {
    cube_gates__out[l] = 1.0 / (1.0 + exp(-cube_gates_b0[l]));
    let adt = cube_gates_a0[l] + cube_gates_dt[l];
    let sp = select(log(1.0 + exp(adt)), adt, adt > 20.0);
    cube_gates__out2[l] = exp(sp * cube_gates_sa[l]);
  }
}
@group(0) @binding(0) var<storage, read> cube_l2qk_c: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_l2qk__out: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_l2qk__out2: array<f32>;
var<workgroup> red128: array<f32, 128>;
@compute @workgroup_size(128)
fn cube_l2qk_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let hd = g.x;                       // 0..15 q heads, 16..31 k heads
  let isq = hd < 16u;
  let src = hd * 128u + l;            // conv layout: q 0..2047, k 2048..4095
  let x = cube_l2qk_c[src];
  red128[l] = x * x;
  workgroupBarrier();
  for (var h = 64u; h > 0u; h >>= 1u) {
    if (l < h) { red128[l] += red128[l + h]; }
    workgroupBarrier();
  }
  var s = 1.0 / max(sqrt(red128[0]), EPS);
  if (isq) { s = s / 11.313708498984761; }
  let y = x * s;
  let o = (hd & 15u) * 128u + l;
  if (isq) { cube_l2qk__out[o] = y; cube_l2qk__out[2048u + o] = y; }
  else { cube_l2qk__out2[o] = y; cube_l2qk__out2[2048u + o] = y; }
}
