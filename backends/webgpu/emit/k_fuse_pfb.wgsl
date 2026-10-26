// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"kvput1","entry":"cube_kvput1_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"kx","ty":"f32","cls":"row","binding":0},{"name":"vx","ty":"f32","cls":"row","binding":1},{"name":"w","ty":"f32","cls":"shared","binding":2}],"out":{"ty":"f32","binding":3},"sc":{"binding":5,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0,"out2":{"ty":"f32","binding":4},"tokens":true,"strides":{"kx":1024,"vx":1024}}]}
// BATCHPREFILL token-batched twins (webprefill/bin/gen-tokbatch.mjs from k_fuse_pf.wgsl): kvput1; tk = workgroup_id.y
// BATCHPREFILL (webprefill/bin/gen-kvput1.mjs from k_fuse_t.wgsl): kvput, one row, in place
const EPS: f32 = 1.0e-6;
@group(0) @binding(0) var<storage, read> cube_kvput1_kx: array<f32>;
@group(0) @binding(1) var<storage, read> cube_kvput1_vx: array<f32>;
@group(0) @binding(2) var<storage, read> cube_kvput1_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_kvput1__out: array<f32>;
@group(0) @binding(4) var<storage, read_write> cube_kvput1__out2: array<f32>;
@group(0) @binding(5) var<storage, read> cube_kvput1__sc: array<u32>;
@compute @workgroup_size(256)
fn cube_kvput1_main(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let tk = g.y;   // BATCHPREFILL: the token slot
  let pos = (cube_kvput1__sc[1u] + tk);
  let wg = pos * 4u + g.x;   // BATCHPREFILL: the row's 4 workgroups only
  let i = wg * 256u + l;
  let hk = wg & 3u;   // no barriers (the slot test is not provably uniform): each thread recomputes the head's rms from global memory
  var ss = 0.0;
  for (var k = 0u; k < 256u; k += 1u) { let v = cube_kvput1_kx[tk * 1024u + (hk * 256u + k)]; ss += v * v; }
  let inv = 1.0 / sqrt(ss / 256.0 + EPS);
  var o: f32;
  if (l < 64u) {
    let j = l >> 1u;
    let a = (cube_kvput1_kx[tk * 1024u + (hk * 256u + j)] * inv) * cube_kvput1_w[j];
    let b = (cube_kvput1_kx[tk * 1024u + (hk * 256u + j + 32u)] * inv) * cube_kvput1_w[j + 32u];
    let theta = f32(pos) * pow(10000000.0, -f32(2u * j) / 64.0);
    let c = cos(theta); let s = sin(theta);
    o = select(a * s + b * c, a * c - b * s, (l & 1u) == 0u);
  } else {
    o = (cube_kvput1_kx[tk * 1024u + (hk * 256u + l)] * inv) * cube_kvput1_w[l];
  }
  cube_kvput1__out[i] = o;
  cube_kvput1__out2[i] = cube_kvput1_vx[tk * 1024u + (i & 1023u)];
}
