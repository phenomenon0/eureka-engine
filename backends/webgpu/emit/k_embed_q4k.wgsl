// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"map","name":"embed4","entry":"map_embed4_main","lanes":256,"ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"id","ty":"u32","at":2},{"name":"chunk_row0","ty":"u32","at":3}],"lens":[{"name":"wb","at":1}]},"len_at":0}]}
// BENDWEB-SHOWCASE (hand-written, NOT bend-emitted, NOT bit-checked): the embedding gather for a Q4_K token_embd (Kev-4B), the
// same ABI as k_embed.wgsl's `embed` (wb = one head chunk of 62080 rows, out[i] = row (id - chunk_row0) element i, i < 2560).
// llama.cpp dequantize_row_q4_K: block = d f16, dmin f16, scales[12], qs[128]; element j: chunk c = j/64, half h = (j%64)/32,
// sub-block is = 2c+h, q = qs[32c + j%32] (low nibble h=0, high h=1); y = (d*sc) * q - (dmin*m).
@group(0) @binding(0) var<storage, read> map_embed4_wb: array<u32>;
@group(0) @binding(1) var<storage, read_write> map_embed4__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_embed4__sc: array<u32>;
fn byte_at(o: u32) -> u32 { return (map_embed4_wb[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
@compute @workgroup_size(256)
fn map_embed4_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_embed4__sc[0u]) { return; }
  let row = map_embed4__sc[2u] - map_embed4__sc[3u];
  let blk = row * 1440u + (i >> 8u) * 144u;
  let j = i & 255u;
  let dd = unpack2x16float(map_embed4_wb[blk >> 2u]);   // blocks are 144 bytes: 4-byte aligned
  let c = j >> 6u; let h = (j >> 5u) & 1u; let is = 2u * c + h;
  let sb = blk + 4u;
  var sc: u32; var m: u32;
  if (is < 4u) { sc = byte_at(sb + is) & 63u; m = byte_at(sb + is + 4u) & 63u; }
  else { sc = (byte_at(sb + is + 4u) & 0xFu) | ((byte_at(sb + is - 4u) >> 6u) << 4u); m = (byte_at(sb + is + 4u) >> 4u) | ((byte_at(sb + is) >> 6u) << 4u); }
  let qb = byte_at(blk + 16u + 32u * c + (j & 31u));
  let q = select(qb & 0xFu, qb >> 4u, h == 1u);
  map_embed4__out[i] = (dd.x * f32(sc)) * f32(q) - (dd.y * f32(m));
}
