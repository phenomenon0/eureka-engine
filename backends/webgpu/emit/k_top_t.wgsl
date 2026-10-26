// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"tp1","entry":"cube_tp1_main","n":1024,"len":1024,"lanes":256,"node":"u32","per":"row","ins":[{"name":"z","ty":"f32","cls":"row","binding":0}],"out":{"ty":"u32","binding":1},"sc":null,"rows_at":null},{"kind":"cube","name":"tp2","entry":"cube_tp2_main","n":243,"len":243,"lanes":256,"node":"u32","per":"row","ins":[{"name":"p","ty":"u32","cls":"row","binding":0}],"out":{"ty":"u32","binding":1},"sc":null,"rows_at":null}]}
// TURBO fused top-2 (bin/turbo-build.py): value-descending, index-ascending order (ties -> lowest index); any NaN logit -> id 0xFFFFFFFF (the engine throws)
const NV: u32 = 248320u;
const NP: u32 = 243u;
struct T2 { v1: f32, i1: u32, v2: f32, i2: u32 }
fn better(av: f32, ai: u32, bv: f32, bi: u32) -> bool { return av > bv || (av == bv && ai < bi); }
fn ins(t: T2, v: f32, i: u32) -> T2 {
  var r = t;
  if (better(v, i, r.v1, r.i1)) { r.v2 = r.v1; r.i2 = r.i1; r.v1 = v; r.i1 = i; }
  else if (better(v, i, r.v2, r.i2)) { r.v2 = v; r.i2 = i; }
  return r;
}
fn merge(a: T2, b: T2) -> T2 { return ins(ins(a, b.v1, b.i1), b.v2, b.i2); }
var<workgroup> wv1: array<f32, 256>; var<workgroup> wi1: array<u32, 256>; var<workgroup> wv2: array<f32, 256>; var<workgroup> wi2: array<u32, 256>; var<workgroup> wn: array<u32, 256>;
fn tree(l: u32, t: T2, nan: u32) -> T2 {
  wv1[l] = t.v1; wi1[l] = t.i1; wv2[l] = t.v2; wi2[l] = t.i2; wn[l] = nan;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (l < h) {
      let m = merge(T2(wv1[l], wi1[l], wv2[l], wi2[l]), T2(wv1[l + h], wi1[l + h], wv2[l + h], wi2[l + h]));
      wv1[l] = m.v1; wi1[l] = m.i1; wv2[l] = m.v2; wi2[l] = m.i2; wn[l] = wn[l] | wn[l + h];
    }
    workgroupBarrier();
  }
  return T2(wv1[0], wi1[0], wv2[0], wi2[0]);
}
@group(0) @binding(0) var<storage, read> cube_tp1_z: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_tp1__out: array<u32>;
@compute @workgroup_size(256)
fn cube_tp1_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let wg = g.y * ng.x + g.x;
  let ninf = -3.0e38;
  var t = T2(ninf, 0xFFFFFFFFu, ninf, 0xFFFFFFFFu);
  var nan = 0u;
  for (var k = 0u; k < 4u; k += 1u) {
    let i = wg * 1024u + k * 256u + l;
    if (i < NV) {
      let v = cube_tp1_z[i];
      if (v != v) { nan = 1u; } else { t = ins(t, v, i); }
    }
  }
  let r = tree(l, t, nan);
  if (l == 0u) {
    let o = wg * 8u;
    cube_tp1__out[o] = bitcast<u32>(r.v1); cube_tp1__out[o + 1u] = r.i1; cube_tp1__out[o + 2u] = bitcast<u32>(r.v2); cube_tp1__out[o + 3u] = r.i2; cube_tp1__out[o + 4u] = wn[0];
  }
}
@group(0) @binding(0) var<storage, read> cube_tp2_p: array<u32>;
@group(0) @binding(1) var<storage, read_write> cube_tp2__out: array<u32>;
@compute @workgroup_size(256)
fn cube_tp2_main(@builtin(local_invocation_index) l: u32) {
  let ninf = -3.0e38;
  var t = T2(ninf, 0xFFFFFFFFu, ninf, 0xFFFFFFFFu);
  var nan = 0u;
  if (l < NP) {
    let o = l * 8u;
    t = T2(bitcast<f32>(cube_tp2_p[o]), cube_tp2_p[o + 1u], bitcast<f32>(cube_tp2_p[o + 2u]), cube_tp2_p[o + 3u]);
    nan = cube_tp2_p[o + 4u];
  }
  let r = tree(l, t, nan);
  if (l == 0u) {
    let bad = wn[0] != 0u;
    cube_tp2__out[0] = select(r.i1, 0xFFFFFFFFu, bad); cube_tp2__out[1] = bitcast<u32>(r.v1);
    cube_tp2__out[2] = select(r.i2, 0xFFFFFFFFu, bad); cube_tp2__out[3] = bitcast<u32>(r.v2);
  }
}
