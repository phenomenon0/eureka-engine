// trunk_kernels.js (WEB-TRUNK-PORT): the trunk engine's WGSL, hand-written and shape-generic. Every width comes from a uniform;
// the head size (D), the delta-net state size (S), and the conv kernel (KC) are baked per model when its pipelines are built,
// so one set of kernels runs any Qwen3.5-family shape: 0.8B (d1024), 2B (d2048), 4B (d2560), 9B / clef-flash (d4096), 27B (d5120),
// and the MoE models (35B-A3B, 122B-A10B, 397B-A17B).
// Plain f32 arithmetic (not the shipped engine's bit-exact x32 canon); weights stay in their GGUF block layout, byte for byte.
// The KV cache is f16 (two halves a u32 word), the same storage type llama.cpp uses by default.
// The shipped 4B engine (q35_*.js, emit/*.wgsl) is not touched or imported.
import { IQ_GRID_OFF } from "./trunk_iq_grids.js";

// ---- shared helpers: byte reads over a u32 view of the raw GGUF blocks (most blocks are not 4-byte aligned)
const BYTES = /* wgsl */`
fn b8(a: u32) -> u32 { return (W[a >> 2u] >> ((a & 3u) * 8u)) & 0xffu; }
fn w32(a: u32) -> u32 {                      // 4 bytes at any byte address
  let s = (a & 3u) * 8u; let lo = W[a >> 2u];
  if (s == 0u) { return lo; }
  return (lo >> s) | (W[(a >> 2u) + 1u] << (32u - s));
}
fn u16at(a: u32) -> u32 { return b8(a) | (b8(a + 1u) << 8u); }
fn f16at(a: u32) -> f32 { return unpack2x16float(u16at(a)).x; }
fn s8(v: u32) -> f32 { return f32(i32(v << 24u) >> 24u); }
`;
const SCALE_K4 = /* wgsl */`
fn scmin(bw: u32, j: u32) -> vec2f {         // get_scale_min_k4 over the 12 scale bytes at words bw+1..bw+3
  let q = array<u32, 3>(W[bw + 1u], W[bw + 2u], W[bw + 3u]);
  var sc: u32; var mn: u32;
  if (j < 4u) {
    sc = (q[j >> 2u] >> ((j & 3u) * 8u)) & 63u;
    mn = (q[(j + 4u) >> 2u] >> (((j + 4u) & 3u) * 8u)) & 63u;
  } else {
    let a = (q[(j + 4u) >> 2u] >> (((j + 4u) & 3u) * 8u)) & 0xffu;
    let lo = (q[(j - 4u) >> 2u] >> (((j - 4u) & 3u) * 8u)) & 0xffu;
    let hi = (q[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu;
    sc = (a & 0xfu) | ((lo >> 6u) << 4u);
    mn = (a >> 4u) | ((hi >> 6u) << 4u);
  }
  return vec2f(f32(sc), f32(mn));
}
`;
// IQ codebooks live in one storage table G (trunk_iq_grids.js); a grid entry is 8 bytes (2 words) or 4 bytes (1 word)
const IQ4 = BYTES + /* wgsl */`
const KV_IQ4 = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
`;
const IQ = BYTES + /* wgsl */`
fn ksign(i: u32) -> u32 { return i | ((countOneBits(i) & 1u) << 7u); }   // ksigns_iq2xs
fn g8(off: u32, idx: u32, j: u32) -> u32 { return (G[off + idx * 2u + (j >> 2u)] >> ((j & 3u) * 8u)) & 0xffu; }
fn g4(off: u32, idx: u32, j: u32) -> u32 { return (G[off + idx] >> (j * 8u)) & 0xffu; }
fn sgn(signs: u32, j: u32) -> f32 { return select(1.0, -1.0, ((signs >> j) & 1u) == 1u); }
`;
const OFF = (k) => `${IQ_GRID_OFF[k]}u`;

// ---- one 32-value unit of a row against x: unit(row, u, K, xo), xo = the x index of the row's value 0.
// The first five are hand-tuned; the rest go through grp(), 8 dequantized values at a time (generic unit below).
const UNIT = {
  F32: /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  let a = row * K + u * 32u; var s = 0.0;
  for (var l = 0u; l < 32u; l++) { s += bitcast<f32>(W[a + l]) * X[xo + u * 32u + l]; }
  return s;
}`,
  Q8_0: BYTES + /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  let a = (row * (K / 32u) + u) * 34u; let d = f16at(a); var s = 0.0; let xb = xo + u * 32u;
  for (var l = 0u; l < 32u; l += 4u) {
    let w = w32(a + 2u + l);
    s += s8(w) * X[xb + l] + s8(w >> 8u) * X[xb + l + 1u] + s8(w >> 16u) * X[xb + l + 2u] + s8(w >> 24u) * X[xb + l + 3u];
  }
  return d * s;
}`,
  Q4_K: SCALE_K4 + /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  let b = u >> 3u; let sub = u & 7u; let j = sub >> 1u; let hi = (sub & 1u) * 4u;
  let bw = (row * (K / 256u) + b) * 36u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let xb = xo + b * 256u + sub * 32u; var sq = 0.0; var sx = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let w = W[bw + 4u + j * 8u + k] >> hi;
    let x = vec4f(X[xb + 4u * k], X[xb + 4u * k + 1u], X[xb + 4u * k + 2u], X[xb + 4u * k + 3u]);
    let q = vec4f(f32(w & 0xfu), f32((w >> 8u) & 0xfu), f32((w >> 16u) & 0xfu), f32((w >> 24u) & 0xfu));
    sq += dot(q, x); sx += x.x + x.y + x.z + x.w;
  }
  return dm.x * sm.x * sq - dm.y * sm.y * sx;
}`,
  Q5_K: SCALE_K4 + /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  let b = u >> 3u; let sub = u & 7u; let j = sub >> 1u; let hi = (sub & 1u) * 4u;
  let bw = (row * (K / 256u) + b) * 44u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let xb = xo + b * 256u + sub * 32u; var sq = 0.0; var sx = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let w = W[bw + 12u + j * 8u + k] >> hi;
    let h = W[bw + 4u + k] >> sub;            // bit (2j + hi/4) of each qh byte
    let x = vec4f(X[xb + 4u * k], X[xb + 4u * k + 1u], X[xb + 4u * k + 2u], X[xb + 4u * k + 3u]);
    let q = vec4f(f32((w & 0xfu) | ((h & 1u) << 4u)), f32(((w >> 8u) & 0xfu) | (((h >> 8u) & 1u) << 4u)),
                  f32(((w >> 16u) & 0xfu) | (((h >> 16u) & 1u) << 4u)), f32(((w >> 24u) & 0xfu) | (((h >> 24u) & 1u) << 4u)));
    sq += dot(q, x); sx += x.x + x.y + x.z + x.w;
  }
  return dm.x * sm.x * sq - dm.y * sm.y * sx;
}`,
  Q6_K: BYTES + /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  let b = u >> 3u; let n = (u >> 2u) & 1u; let sub = u & 3u;
  let a = (row * (K / 256u) + b) * 210u;
  let d = f16at(a + 208u);
  let qla = a + 64u * n + (sub & 1u) * 32u; let nib = (sub >> 1u) * 4u;
  let qha = a + 128u + 32u * n; let sh = 2u * sub;
  let sca = a + 192u + 8u * n + 2u * sub;
  let xb = xo + b * 256u + 128u * n + 32u * sub;
  var s0 = 0.0; var s1 = 0.0;
  for (var l = 0u; l < 32u; l += 4u) {
    let ql = w32(qla + l) >> nib; let qh = w32(qha + l) >> sh;
    let q = vec4f(f32((ql & 0xfu) | ((qh & 3u) << 4u)), f32(((ql >> 8u) & 0xfu) | (((qh >> 8u) & 3u) << 4u)),
                  f32(((ql >> 16u) & 0xfu) | (((qh >> 16u) & 3u) << 4u)), f32(((ql >> 24u) & 0xfu) | (((qh >> 24u) & 3u) << 4u))) - 32.0;
    let x = vec4f(X[xb + l], X[xb + l + 1u], X[xb + l + 2u], X[xb + l + 3u]);
    if (l < 16u) { s0 += dot(q, x); } else { s1 += dot(q, x); }
  }
  return d * (s8(b8(sca)) * s0 + s8(b8(sca + 1u)) * s1);
}`,
};

// grp(row, u, g, K): values 8g .. 8g+7 of 32-value unit u of a row, dequantized (ggml's dequantize_row_* order)
const GRP = {
  F16: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * K + u * 32u + g * 8u) * 2u; var v: array<f32, 8>;
  for (var m = 0u; m < 4u; m++) { let p = unpack2x16float(w32(a + 4u * m)); v[2u * m] = p.x; v[2u * m + 1u] = p.y; }
  return v;
}`,
  BF16: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * K + u * 32u + g * 8u) * 2u; var v: array<f32, 8>;
  for (var m = 0u; m < 4u; m++) { let w = w32(a + 4u * m); v[2u * m] = bitcast<f32>(w << 16u); v[2u * m + 1u] = bitcast<f32>(w & 0xffff0000u); }
  return v;
}`,
  Q4_0: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * (K / 32u) + u) * 18u; let d = f16at(a); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { let e = g * 8u + j; let q = (b8(a + 2u + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu; v[j] = d * (f32(q) - 8.0); }
  return v;
}`,
  Q4_1: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * (K / 32u) + u) * 20u; let d = f16at(a); let m = f16at(a + 2u); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { let e = g * 8u + j; let q = (b8(a + 4u + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu; v[j] = d * f32(q) + m; }
  return v;
}`,
  Q5_0: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * (K / 32u) + u) * 22u; let d = f16at(a); let qh = w32(a + 2u); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) {
    let e = g * 8u + j; let q = ((b8(a + 6u + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu) | (((qh >> e) & 1u) << 4u);
    v[j] = d * (f32(q) - 16.0);
  }
  return v;
}`,
  Q5_1: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * (K / 32u) + u) * 24u; let d = f16at(a); let m = f16at(a + 2u); let qh = w32(a + 4u); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) {
    let e = g * 8u + j; let q = ((b8(a + 8u + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu) | (((qh >> e) & 1u) << 4u);
    v[j] = d * f32(q) + m;
  }
  return v;
}`,
  Q2_K: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // scales[16] @0, qs[64] @16, d @80, dmin @82
  let a = (row * (K / 256u) + (u >> 3u)) * 84u; let n = (u & 7u) >> 2u; let j = u & 3u; let half = g >> 1u;
  let sc = b8(a + n * 8u + j * 2u + half); let dl = f16at(a + 80u) * f32(sc & 0xfu); let ml = f16at(a + 82u) * f32(sc >> 4u);
  let qb = a + 16u + 32u * n + 16u * half + (g & 1u) * 8u; var v: array<f32, 8>;
  for (var l = 0u; l < 8u; l++) { v[l] = dl * f32((b8(qb + l) >> (2u * j)) & 3u) - ml; }
  return v;
}`,
  Q3_K: BYTES + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // hmask[32] @0, qs[64] @32, scales[12] @96, d @108
  let a = (row * (K / 256u) + (u >> 3u)) * 110u; let n = (u & 7u) >> 2u; let j = u & 3u; let half = g >> 1u;
  let is = n * 8u + j * 2u + half;
  let lo = (b8(a + 96u + (is & 7u)) >> (4u * (is >> 3u))) & 0xfu; let hi = (b8(a + 96u + 8u + (is & 3u)) >> (2u * (is >> 2u))) & 3u;
  let dl = f16at(a + 108u) * (f32(lo | (hi << 4u)) - 32.0);
  let l0 = 16u * half + (g & 1u) * 8u; let qb = a + 32u + 32u * n + l0; let mb = n * 4u + j; var v: array<f32, 8>;
  for (var l = 0u; l < 8u; l++) {
    let q = (b8(qb + l) >> (2u * j)) & 3u; let h = (b8(a + l0 + l) >> mb) & 1u;
    v[l] = dl * (f32(q) - select(4.0, 0.0, h == 1u));
  }
  return v;
}`,
  IQ4_NL: IQ4 + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {
  let a = (row * (K / 32u) + u) * 18u; let d = f16at(a); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { let e = g * 8u + j; v[j] = d * KV_IQ4[(b8(a + 2u + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu]; }
  return v;
}`,
  IQ4_XS: IQ4 + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8
  let a = (row * (K / 256u) + (u >> 3u)) * 136u; let ib = u & 7u;
  let ls = ((b8(a + 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 0xfu) | (((u16at(a + 2u) >> (2u * ib)) & 3u) << 4u);
  let dl = f16at(a) * (f32(ls) - 32.0); var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { let e = g * 8u + j; v[j] = dl * KV_IQ4[(b8(a + 8u + 16u * ib + (e & 15u)) >> ((e >> 4u) * 4u)) & 0xfu]; }
  return v;
}`,
  IQ2_XXS: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs u16[32] @2
  let a = (row * (K / 256u) + (u >> 3u)) * 66u; let ib = u & 7u;
  let a0 = w32(a + 2u + 8u * ib); let a1 = w32(a + 6u + 8u * ib);
  let db = f16at(a) * (0.5 + f32(a1 >> 28u)) * 0.25; let idx = (a0 >> (8u * g)) & 0xffu; let s = ksign((a1 >> (7u * g)) & 127u);
  var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { v[j] = db * f32(g8(${OFF("iq2xxs")}, idx, j)) * sgn(s, j); }
  return v;
}`,
  IQ2_XS: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs u16[32] @2, scales[8] @66
  let a = (row * (K / 256u) + (u >> 3u)) * 74u; let ib = u & 7u;
  let q = u16at(a + 2u + 2u * (4u * ib + g)); let s = ksign(q >> 9u);
  let db = f16at(a) * (0.5 + f32((b8(a + 66u + ib) >> (4u * (g >> 1u))) & 0xfu)) * 0.25;
  var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { v[j] = db * f32(g8(${OFF("iq2xs")}, q & 511u, j)) * sgn(s, j); }
  return v;
}`,
  IQ2_S: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs[64] @2 (idx 32, signs 32), qh[8] @66, scales[8] @74
  let a = (row * (K / 256u) + (u >> 3u)) * 82u; let ib = u & 7u;
  let idx = b8(a + 2u + 4u * ib + g) | ((b8(a + 66u + ib) << (8u - 2u * g)) & 0x300u);
  let s = b8(a + 34u + 4u * ib + g);
  let db = f16at(a) * (0.5 + f32((b8(a + 74u + ib) >> (4u * (g >> 1u))) & 0xfu)) * 0.25;
  var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { v[j] = db * f32(g8(${OFF("iq2s")}, idx, j)) * sgn(s, j); }
  return v;
}`,
  IQ3_XXS: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs[64] @2, scales_and_signs u32[8] @66
  let a = (row * (K / 256u) + (u >> 3u)) * 98u; let ib = u & 7u;
  let aux = w32(a + 66u + 4u * ib); let db = f16at(a) * (0.5 + f32(aux >> 28u)) * 0.5; let s = ksign((aux >> (7u * g)) & 127u);
  let i1 = b8(a + 2u + 8u * ib + 2u * g); let i2 = b8(a + 3u + 8u * ib + 2u * g);
  var v: array<f32, 8>;
  for (var j = 0u; j < 4u; j++) {
    v[j] = db * f32(g4(${OFF("iq3xxs")}, i1, j)) * sgn(s, j); v[j + 4u] = db * f32(g4(${OFF("iq3xxs")}, i2, j)) * sgn(s, j + 4u);
  }
  return v;
}`,
  IQ3_S: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
  let a = (row * (K / 256u) + (u >> 3u)) * 110u; let ib = u & 7u;
  let db = f16at(a) * f32(1u + 2u * ((b8(a + 106u + (ib >> 1u)) >> (4u * (ib & 1u))) & 0xfu));
  let qh = b8(a + 66u + ib);
  let i1 = b8(a + 2u + 8u * ib + 2u * g) | ((qh << (8u - 2u * g)) & 256u);
  let i2 = b8(a + 3u + 8u * ib + 2u * g) | ((qh << (7u - 2u * g)) & 256u);
  let s = b8(a + 74u + 4u * ib + g);
  var v: array<f32, 8>;
  for (var j = 0u; j < 4u; j++) {
    v[j] = db * f32(g4(${OFF("iq3s")}, i1, j)) * sgn(s, j); v[j + 4u] = db * f32(g4(${OFF("iq3s")}, i2, j)) * sgn(s, j + 4u);
  }
  return v;
}`,
  IQ1_S: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // d @0, qs[32] @2, qh u16[8] @34
  let a = (row * (K / 256u) + (u >> 3u)) * 50u; let ib = u & 7u;
  let qh = u16at(a + 34u + 2u * ib);
  let dl = f16at(a) * f32(2u * ((qh >> 12u) & 7u) + 1u); let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
  let idx = b8(a + 2u + 4u * ib + g) | (((qh >> (3u * g)) & 7u) << 8u);
  var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { v[j] = dl * (s8(g8(${OFF("iq1s")}, idx, j)) + delta); }
  return v;
}`,
  IQ1_M: IQ + /* wgsl */`
fn grp(row: u32, u: u32, g: u32, K: u32) -> array<f32, 8> {   // qs[32] @0, qh[16] @32, scales u16[4] @48 (the f16 scale is spread over their top nibbles)
  let a = (row * (K / 256u) + (u >> 3u)) * 56u; let ib = u & 7u;
  let sc = array<u32, 4>(u16at(a + 48u), u16at(a + 50u), u16at(a + 52u), u16at(a + 54u));
  let d = unpack2x16float((sc[0] >> 12u) | ((sc[1] >> 8u) & 0xf0u) | ((sc[2] >> 4u) & 0xf00u) | (sc[3] & 0xf000u)).x;
  let dl = d * f32(2u * ((sc[ib >> 1u] >> (6u * (ib & 1u) + 3u * (g >> 1u))) & 7u) + 1u);
  let qh = b8(a + 32u + 2u * ib + (g >> 1u)); let sh = select(8u, 4u, (g & 1u) == 1u);
  let idx = b8(a + 4u * ib + g) | ((qh << sh) & 0x700u);
  let delta = select(0.125, -0.125, (qh & select(0x08u, 0x80u, (g & 1u) == 1u)) != 0u);
  var v: array<f32, 8>;
  for (var j = 0u; j < 8u; j++) { v[j] = dl * (s8(g8(${OFF("iq1s")}, idx, j)) + delta); }
  return v;
}`,
};
const GRP_UNIT = /* wgsl */`
fn unit(row: u32, u: u32, K: u32, xo: u32) -> f32 {
  var s = 0.0;
  for (var g = 0u; g < 4u; g++) {
    let v = grp(row, u, g, K); let xb = xo + u * 32u + g * 8u;
    s += dot(vec4f(v[0], v[1], v[2], v[3]), vec4f(X[xb], X[xb + 1u], X[xb + 2u], X[xb + 3u]))
       + dot(vec4f(v[4], v[5], v[6], v[7]), vec4f(X[xb + 4u], X[xb + 5u], X[xb + 6u], X[xb + 7u]));
  }
  return s;
}`;
for (const [q, code] of Object.entries(GRP)) UNIT[q] = code + GRP_UNIT;

export const TRUNK_QUANTS = Object.keys(UNIT);
/** Quants whose kernels read the IQ codebook table (an extra binding). */
export const NEEDS_GRID = new Set(Object.keys(GRP).filter(q => q.startsWith("IQ") && q !== "IQ4_NL" && q !== "IQ4_XS"));
/** Values a block (the row width must be a multiple). */
export const QUANT_BLOCK = Object.fromEntries(TRUNK_QUANTS.map(q => [q, /^(F32|F16|BF16)$/.test(q) ? 32 : /^(Q4_0|Q4_1|Q5_0|Q5_1|Q8_0|IQ4_NL)$/.test(q) ? 32 : 256]));
const GRID_DECL = (q, at) => NEEDS_GRID.has(q) ? `@group(0) @binding(${at}) var<storage, read> G: array<u32>;` : "";

// y[yoff + r0 + row] (+= if p.acc) = W[row] . x[xoff ..]; 8 rows a workgroup, 32 lanes a row, rows over a 2-D grid
export const gemv = (q) => /* wgsl */`
struct P { rows: u32, K: u32, xoff: u32, yoff: u32, acc: u32, gx: u32, r0: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${GRID_DECL(q, 4)}
var<workgroup> red: array<f32, 256>;
${UNIT[q]}
@compute @workgroup_size(32, 8)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let row = (g.y * p.gx + g.x) * 8u + l.y;
  var acc = 0.0;
  if (row < p.rows) { let nu = p.K / 32u; for (var u = l.x; u < nu; u += 32u) { acc += unit(row, u, p.K, p.xoff); } }
  let i = l.y * 32u + l.x; red[i] = acc; workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) { if (l.x < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  if (l.x == 0u && row < p.rows) { let o = p.yoff + p.r0 + row; if (p.acc == 1u) { Y[o] += red[i]; } else { Y[o] = red[i]; } }
}`;

// MoE expert GEMV: slot z (of the k routed experts) runs expert sel[z] over its rows, when that expert lives in this part
// [e0, e0 + ne) of the stacked tensor: y[yoff + z*ys + row] = W[expert][row] . x[z*xs ..]
export const gemv_id = (q) => /* wgsl */`
struct P { rows: u32, K: u32, xs: u32, ys: u32, yoff: u32, gx: u32, e0: u32, ne: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> sel: array<u32>;
${GRID_DECL(q, 5)}
var<workgroup> red: array<f32, 256>;
${UNIT[q]}
@compute @workgroup_size(32, 8)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let e = sel[g.z]; let mine = e >= p.e0 && e < p.e0 + p.ne;   // no early return: the barriers below need uniform control flow
  let row = (g.y * p.gx + g.x) * 8u + l.y; let wrow = (e - p.e0) * p.rows + row;
  var acc = 0.0;
  if (mine && row < p.rows) { let nu = p.K / 32u; for (var u = l.x; u < nu; u += 32u) { acc += unit(wrow, u, p.K, g.z * p.xs); } }
  let i = l.y * 32u + l.x; red[i] = acc; workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) { if (l.x < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  if (mine && l.x == 0u && row < p.rows) { Y[p.yoff + g.z * p.ys + row] = red[i]; }
}`;

// embedding gather: y[k] = dequant(W row tok)[k]; the token id is ids[0]
const DQ = {
  F32: `fn dq(row: u32, k: u32, K: u32) -> f32 { return bitcast<f32>(W[row * K + k]); }`,
  Q8_0: BYTES + `fn dq(row: u32, k: u32, K: u32) -> f32 { let a = (row * (K / 32u) + k / 32u) * 34u; return f16at(a) * s8(b8(a + 2u + (k & 31u))); }`,
  Q4_K: SCALE_K4 + `fn dq(row: u32, k: u32, K: u32) -> f32 {
  let b = k / 256u; let r = k & 255u; let sub = r / 32u; let l = r & 31u; let bw = (row * (K / 256u) + b) * 36u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let q = (W[bw + 4u + (sub >> 1u) * 8u + (l >> 2u)] >> ((l & 3u) * 8u + (sub & 1u) * 4u)) & 0xfu;
  return dm.x * sm.x * f32(q) - dm.y * sm.y; }`,
  Q5_K: SCALE_K4 + `fn dq(row: u32, k: u32, K: u32) -> f32 {
  let b = k / 256u; let r = k & 255u; let sub = r / 32u; let l = r & 31u; let bw = (row * (K / 256u) + b) * 44u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let q = (W[bw + 12u + (sub >> 1u) * 8u + (l >> 2u)] >> ((l & 3u) * 8u + (sub & 1u) * 4u)) & 0xfu;
  let h = (W[bw + 4u + (l >> 2u)] >> ((l & 3u) * 8u + sub)) & 1u;
  return dm.x * sm.x * f32(q | (h << 4u)) - dm.y * sm.y; }`,
  Q6_K: BYTES + `fn dq(row: u32, k: u32, K: u32) -> f32 {
  let b = k / 256u; let r = k & 255u; let n = r / 128u; let sub = (r & 127u) / 32u; let l = r & 31u;
  let a = (row * (K / 256u) + b) * 210u;
  let ql = (b8(a + 64u * n + (sub & 1u) * 32u + l) >> ((sub >> 1u) * 4u)) & 0xfu;
  let qh = (b8(a + 128u + 32u * n + l) >> (2u * sub)) & 3u;
  return f16at(a + 208u) * s8(b8(a + 192u + 8u * n + 2u * sub + l / 16u)) * (f32(ql | (qh << 4u)) - 32.0); }`,
};
for (const [q, code] of Object.entries(GRP)) DQ[q] = code + `
fn dq(row: u32, k: u32, K: u32) -> f32 { let r = k & 31u; let v = grp(row, k / 32u, r >> 3u, K); return v[r & 7u]; }`;
export const embed = (q) => /* wgsl */`
struct P { K: u32, rows0: u32, nrows: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${GRID_DECL(q, 4)}
${DQ[q]}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let t = ids[0];
  if (gi.x < p.K && t >= p.rows0 && t - p.rows0 < p.nrows) { Y[gi.x] = dq(t - p.rows0, gi.x, p.K); }
}`;

// RMS norm per head: y[yoff + h*ys + i] = x[xoff + h*xs + i] * rsqrt(mean + eps) * w[i] (mode 1: no weight, mode 2: L2: x / sqrt(sum + eps))
export const rmsnorm = /* wgsl */`
struct P { dim: u32, xoff: u32, xs: u32, yoff: u32, ys: u32, mode: u32, eps: f32, _a: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wn: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let xb = p.xoff + g.x * p.xs; let yb = p.yoff + g.x * p.ys;
  var s = 0.0; for (var i = l.x; i < p.dim; i += 256u) { let v = X[xb + i]; s += v * v; }
  red[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  var r: f32;
  if (p.mode == 2u) { r = inverseSqrt(red[0] + p.eps); } else { r = inverseSqrt(red[0] / f32(p.dim) + p.eps); }
  for (var i = l.x; i < p.dim; i += 256u) { var v = X[xb + i] * r; if (p.mode == 0u) { v *= Wn[i]; } Y[yb + i] = v; }
}`;

// swiglu: y[i] = silu(g) * u, with g = GU[(i / blk) * gs + i % blk] and u = GU[that + uo]
//   dense: blk = n, gs = 0, uo = n (gate then up); fused MoE gate_up: blk = ffe, gs = 2 ffe, uo = ffe (per slot gate, up)
export const swiglu = /* wgsl */`
struct P { n: u32, blk: u32, gs: u32, uo: u32 }
@group(0) @binding(0) var<storage, read> GU: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let i = gi.x + gi.y * 65535u * 256u;
  if (i < p.n) { let a = (i / p.blk) * p.gs + i % p.blk; let g = GU[a]; Y[i] = g / (1.0 + exp(-g)) * GU[a + p.uo]; }
}`;

// causal conv1d step (ggml_ssm_conv, kernel KC) + silu; the state keeps the last KC-1 inputs per channel, oldest first
export const conv = (KC) => /* wgsl */`
const KC: u32 = ${KC}u;
struct P { C: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wc: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let c = gi.x; if (c >= p.C) { return; }
  let sb = c * (KC - 1u); let x = X[c];
  var v = Wc[c * KC + KC - 1u] * x;
  for (var k = 0u; k + 1u < KC; k++) { v += Wc[c * KC + k] * S[sb + k]; }
  for (var k = 0u; k + 2u < KC; k++) { S[sb + k] = S[sb + k + 1u]; }
  S[sb + KC - 2u] = x;
  Y[c] = v / (1.0 + exp(-v));
}`;

// one gated-delta-net step for v-head h (workgroup), lane j = a value column (state S x S, S a power of two <= 256):
//   g = softplus(alpha + dt) * a; beta = sigmoid(b); s *= exp(g); d = (v - s^T k) beta; s += k d^T; o = s^T q / sqrt(S)
// then the gated norm: o * rsqrt(mean(o^2) + eps) * ssm_norm * silu(z). q, k = the (L2-normed) k-head h % HK (ggml_repeat tiling).
// The state is s[h][i][j] (i = key dim) so lanes read adjacent words.
export const delta = (S) => /* wgsl */`
const S: u32 = ${S}u;
struct P { HK: u32, HV: u32, eps: f32, _a: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<f32>;    // conv output: q [HK*S], k [HK*S], v [HV*S]
@group(0) @binding(1) var<storage, read> AB: array<f32>;     // alpha [HV] at 0, beta [HV] at HV (raw GEMV outputs)
@group(0) @binding(2) var<storage, read> Pr: array<f32>;     // dt bias [HV] at 0, a [HV] at HV, ssm_norm [S] at 2 HV
@group(0) @binding(3) var<storage, read_write> St: array<f32>;
@group(0) @binding(4) var<storage, read> Z: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> qs: array<f32, S>;
var<workgroup> ks: array<f32, S>;
var<workgroup> red: array<f32, S>;
fn rsum(j: u32) -> f32 { for (var k = S / 2u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); } return red[0]; }
@compute @workgroup_size(S)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let h = g.x; let j = l.x; let kh = h % p.HK;
  let q0 = QKV[kh * S + j]; let k0 = QKV[p.HK * S + kh * S + j];
  red[j] = q0 * q0; workgroupBarrier();                // the L2 norms of q and k (x / sqrt(sum x^2 + eps), build_gdn_l2_norm)
  let nq = rsum(j); workgroupBarrier();
  qs[j] = q0 * inverseSqrt(nq + p.eps);
  red[j] = k0 * k0; workgroupBarrier();
  let nk = rsum(j); workgroupBarrier();
  ks[j] = k0 * inverseSqrt(nk + p.eps); workgroupBarrier();
  let a = AB[h] + Pr[h]; let sp = select(log(1.0 + exp(a)), a, a > 20.0);
  let eg = exp(sp * Pr[p.HV + h]); let beta = 1.0 / (1.0 + exp(-AB[p.HV + h]));
  let base = h * S * S + j;
  var sk = 0.0;
  for (var i = 0u; i < S; i++) { sk += St[base + i * S] * eg * ks[i]; }
  let d = (QKV[2u * p.HK * S + h * S + j] - sk) * beta;
  var o = 0.0;
  for (var i = 0u; i < S; i++) { let s = St[base + i * S] * eg + ks[i] * d; St[base + i * S] = s; o += s * qs[i]; }
  o *= inverseSqrt(f32(S));
  red[j] = o * o; workgroupBarrier();
  let no = rsum(j);
  let z = Z[h * S + j];
  O[h * S + j] = o * inverseSqrt(no / f32(S) + p.eps) * Pr[2u * p.HV + j] * (z / (1.0 + exp(-z)));
}`;

// full-attention prep, one workgroup a head (NH query heads, then NKV key heads), head size D (even, <= 512):
//   q = rms(qg[h*2D .. +D]) * qn, roped; k = rms(k[h*D ..]) * kn, roped, stored at Kc[pos]; v stored at Vc[pos] (both f16).
//   K is stored transposed, Kc[(kh*H + m)*CAP + pos] (one word = dims 2m, 2m+1), so the score pass, a lane a token, reads
//   adjacent words; V is row-major, Vc[(pos*NKV + kh)*H + m], for the weighted sum, a lane a dim pair.
//   rope = NEOX over the first R dims (MRoPE with one position for every section), theta = pos * base^(-2i/R)
export const attnprep = (D) => /* wgsl */`
const D: u32 = ${D}u;
const H: u32 = ${D / 2}u;
struct P { NH: u32, NKV: u32, R: u32, eps: f32, base: f32, CAP: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> QG: array<f32>;
@group(0) @binding(1) var<storage, read> KV: array<f32>;     // k [NKV*D] at 0, v [NKV*D] at NKV*D
@group(0) @binding(2) var<storage, read> Nw: array<f32>;     // q_norm [D] at 0, k_norm [D] at D
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> Kc: array<u32>;
@group(0) @binding(5) var<storage, read_write> Vc: array<u32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<storage, read> stp: array<u32>;   // [0] = pos
var<workgroup> v: array<f32, D>;
var<workgroup> y: array<f32, D>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let isq = g.x < p.NH; let pos = stp[0];
  var s = 0.0;
  for (var i = l.x; i < D; i += 256u) {
    var x: f32; if (isq) { x = QG[g.x * 2u * D + i]; } else { x = KV[(g.x - p.NH) * D + i]; }
    v[i] = x; s += x * x;
  }
  red[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(D) + p.eps);
  for (var i = l.x; i < D; i += 256u) { var nw: f32; if (isq) { nw = Nw[i]; } else { nw = Nw[D + i]; } v[i] = v[i] * r * nw; }
  workgroupBarrier();
  let hf = p.R / 2u;
  for (var i = l.x; i < D; i += 256u) {
    var o = v[i];
    if (i < p.R) {
      let fi = i % hf; let th = f32(pos) * pow(p.base, -2.0 * f32(fi) / f32(p.R));
      let c = cos(th); let sn = sin(th);
      if (i < hf) { o = v[i] * c - v[i + hf] * sn; } else { o = v[i - hf] * sn + v[i] * c; }
    }
    y[i] = o;
  }
  workgroupBarrier();
  if (isq) { for (var i = l.x; i < D; i += 256u) { Q[g.x * D + i] = y[i]; } }
  else {
    let kh = g.x - p.NH; let at = (pos * p.NKV + kh) * H; let vb = p.NKV * D + kh * D;
    for (var m = l.x; m < H; m += 256u) {
      Kc[(kh * H + m) * p.CAP + pos] = pack2x16float(vec2f(y[2u * m], y[2u * m + 1u]));
      Vc[at + m] = pack2x16float(vec2f(KV[vb + 2u * m], KV[vb + 2u * m + 1u]));
    }
  }
}`;

// split-KV attention (flash-decoding), pass 1: workgroup (chunk c, head h) over tokens [c*CH, c*CH + CH) of [0, pos]:
// lane t scores one token, then the chunk's max m, sum l, and the p-weighted V sum go to Pm/Pa. Dispatched indirectly
// (the host writes [ceil((pos+1)/CH), NH, 1] each step), so the work grows with the context actually used, not its capacity.
// At small capacities the host dispatches every chunk directly instead (cheaper than an indirect dispatch); unused chunks idle.
export const ATTN_CH = 256;
export const attn_part = (D) => /* wgsl */`
const D: u32 = ${D}u;
const H: u32 = ${D / 2}u;
const CH: u32 = ${ATTN_CH}u;
struct P { NH: u32, NKV: u32, MC: u32, scale: f32, CAP: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> Kc: array<u32>;
@group(0) @binding(2) var<storage, read> Vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> Pm: array<f32>;   // [NH][MC][2] = (max, sum)
@group(0) @binding(4) var<storage, read_write> Pa: array<f32>;   // [NH][MC][D]
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<storage, read> stp: array<u32>;
var<workgroup> q: array<f32, D>;
var<workgroup> pr: array<f32, CH>;
var<workgroup> red: array<f32, CH>;
@compute @workgroup_size(CH)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let c = g.x; let h = g.y; let t = l.x; let n = stp[0] + 1u; let kh = h / (p.NH / p.NKV);
  for (var i = t; i < D; i += CH) { q[i] = Q[h * D + i]; }
  workgroupBarrier();
  let tt = c * CH + t; var s = -3.0e38;
  if (tt < n) {
    let kb = kh * H * p.CAP + tt; var acc = 0.0;
    for (var m = 0u; m < H; m++) { let k2 = unpack2x16float(Kc[kb + m * p.CAP]); acc += q[2u * m] * k2.x + q[2u * m + 1u] * k2.y; }
    s = acc * p.scale;
  }
  red[t] = s; workgroupBarrier();
  for (var k = CH / 2u; k > 0u; k >>= 1u) { if (t < k) { red[t] = max(red[t], red[t + k]); } workgroupBarrier(); }
  let mx = red[0]; workgroupBarrier();
  var e = 0.0; if (tt < n) { e = exp(s - mx); }
  pr[t] = e; red[t] = e; workgroupBarrier();
  for (var k = CH / 2u; k > 0u; k >>= 1u) { if (t < k) { red[t] += red[t + k]; } workgroupBarrier(); }
  let pc = h * p.MC + c;
  if (t == 0u) { Pm[pc * 2u] = mx; Pm[pc * 2u + 1u] = red[0]; }
  var nt = 0u; if (c * CH < n) { nt = min(CH, n - c * CH); }   // chunks past the context (direct dispatch at small capacities) do nothing
  for (var m = t; m < H; m += CH) {
    var a = vec2f(0.0);
    for (var u = 0u; u < nt; u++) { a += pr[u] * unpack2x16float(Vc[((c * CH + u) * p.NKV + kh) * H + m]); }
    Pa[pc * D + 2u * m] = a.x; Pa[pc * D + 2u * m + 1u] = a.y;
  }
}`;
// pass 2: one workgroup a head merges the chunks (log-sum-exp), times sigmoid(gate) (gate = qg[h*2D + D ..])
export const attn_merge = (D) => /* wgsl */`
const D: u32 = ${D}u;
const CH: u32 = ${ATTN_CH}u;
struct P { NH: u32, MC: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> Pm: array<f32>;
@group(0) @binding(1) var<storage, read> Pa: array<f32>;
@group(0) @binding(2) var<storage, read> QG: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> stp: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let h = g.x; let nc = (stp[0] + CH) / CH;
  var M = -3.0e38; for (var c = 0u; c < nc; c++) { M = max(M, Pm[(h * p.MC + c) * 2u]); }
  var den = 0.0; for (var c = 0u; c < nc; c++) { den += exp(Pm[(h * p.MC + c) * 2u] - M) * Pm[(h * p.MC + c) * 2u + 1u]; }
  for (var i = l.x; i < D; i += 256u) {
    var num = 0.0;
    for (var c = 0u; c < nc; c++) { num += exp(Pm[(h * p.MC + c) * 2u] - M) * Pa[(h * p.MC + c) * D + i]; }
    let gt = QG[h * 2u * D + D + i];
    O[h * D + i] = num / den / (1.0 + exp(-gt));
  }
}`;

// MoE router: softmax over the expert logits, the top k (ties to the lower index), renormalized (norm_w);
// writes sel[k] (expert ids) and wt[k]. One workgroup.
export const topk = /* wgsl */`
struct P { NE: u32, K: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> L: array<f32>;
@group(0) @binding(1) var<storage, read_write> sel: array<u32>;
@group(0) @binding(2) var<storage, read_write> wt: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> taken: array<u32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u) {
  for (var e = l.x; e < p.NE; e += 256u) { taken[e] = 0u; }
  workgroupBarrier();
  var sum = 0.0;
  for (var k = 0u; k < p.K; k++) {
    var bv = -3.4e38; var bi = 0xffffffffu;
    for (var e = l.x; e < p.NE; e += 256u) { let v = L[e]; if (taken[e] == 0u && v > bv) { bv = v; bi = e; } }
    rv[l.x] = bv; ri[l.x] = bi; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
      if (l.x < s) { let v = rv[l.x + s]; let j = ri[l.x + s]; if (v > rv[l.x] || (v == rv[l.x] && j < ri[l.x])) { rv[l.x] = v; ri[l.x] = j; } }
      workgroupBarrier();
    }
    let e = ri[0]; let v = rv[0];
    if (l.x == 0u) { taken[e] = 1u; sel[k] = e; wt[k] = v; }
    workgroupBarrier();
  }
  if (l.x == 0u) {   // softmax(L)[sel] / sum over the selected = exp(L[sel] - max) / sum exp(...) (the full-softmax denominator cancels)
    let m = wt[0]; var z = 0.0;
    for (var k = 0u; k < p.K; k++) { let x = exp(wt[k] - m); wt[k] = x; z += x; }
    for (var k = 0u; k < p.K; k++) { wt[k] = wt[k] / z; }
  }
}`;
// MoE combine: x[i] += sum_k wt[k] * E[k*d + i] (+ sigmoid(sg[0]) * SH[i] when there is a shared expert)
export const moe_add = /* wgsl */`
struct P { d: u32, K: u32, sh: u32, _a: u32 }   // (not "shared": a WGSL reserved word)
@group(0) @binding(0) var<storage, read> E: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> SH: array<f32>;   // shared-expert output [d], then its gate logit at [d]
@group(0) @binding(3) var<storage, read_write> Xr: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let i = gi.x; if (i >= p.d) { return; }
  var s = 0.0; for (var k = 0u; k < p.K; k++) { s += wt[k] * E[k * p.d + i]; }
  if (p.sh == 1u) { s += SH[i] / (1.0 + exp(-SH[p.d])); }
  Xr[i] += s;
}`;

// argmax over the logits: stage 1 (one workgroup per 4096 words) -> (max, index) pairs; stage 2 one workgroup over the pairs
export const argmax = /* wgsl */`
struct P { n: u32, stage: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> L: array<f32>;
@group(0) @binding(1) var<storage, read_write> Pm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pi: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  var bv = -3.4e38; var bi = 0xffffffffu;
  if (p.stage == 0u) {
    let s = g.x * 4096u;
    for (var k = l.x; k < 4096u; k += 256u) { let t = s + k; if (t < p.n) { let v = L[t]; if (v > bv) { bv = v; bi = t; } } }
  } else {
    for (var k = l.x; k < p.n; k += 256u) { let v = Pm[k]; if (v > bv || (v == bv && Pi[k] < bi)) { bv = v; bi = Pi[k]; } }
  }
  rv[l.x] = bv; ri[l.x] = bi; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (l.x < k) { let v = rv[l.x + k]; let j = ri[l.x + k]; if (v > rv[l.x] || (v == rv[l.x] && j < ri[l.x])) { rv[l.x] = v; ri[l.x] = j; } }
    workgroupBarrier();
  }
  if (l.x == 0u) {
    if (p.stage == 0u) { Pm[g.x] = rv[0]; Pi[g.x] = ri[0]; } else { Pm[1048575u] = rv[0]; Pi[1048575u] = ri[0]; }
  }
}`;
