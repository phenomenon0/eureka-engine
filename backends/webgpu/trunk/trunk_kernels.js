// trunk_kernels.js (WEB-TRUNK-PORT): the trunk engine's WGSL, hand-written and shape-generic (every size comes from a uniform),
// so one set of kernels runs any Qwen3.5-hybrid shape: 0.8B (d1024), 4B (d2560), 9B / clef-flash (d4096), 27B (d5120).
// Plain f32 arithmetic (not the shipped engine's bit-exact x32 canon); weights stay in their GGUF block layout, byte for byte.
// The shipped 4B engine (q35_*.js, emit/*.wgsl) is not touched or imported.

// ---- shared helpers: byte reads over a u32 view of the raw GGUF blocks (Q6_K / Q8_0 blocks are not 4-byte aligned)
const BYTES = /* wgsl */`
fn b8(a: u32) -> u32 { return (W[a >> 2u] >> ((a & 3u) * 8u)) & 0xffu; }
fn w32(a: u32) -> u32 {                      // 4 bytes at any byte address
  let s = (a & 3u) * 8u; let lo = W[a >> 2u];
  if (s == 0u) { return lo; }
  return (lo >> s) | (W[(a >> 2u) + 1u] << (32u - s));
}
fn f16at(a: u32) -> f32 { return unpack2x16float(b8(a) | (b8(a + 1u) << 8u)).x; }
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

// ---- one 32-value unit of a row against x: unit(rowbase, u, xbase) for each quant. xbase = the x index of value 0 of the unit.
const UNIT = {
  F32: /* wgsl */`
fn unit(row: u32, u: u32, K: u32) -> f32 {
  let a = row * K + u * 32u; var s = 0.0;
  for (var l = 0u; l < 32u; l++) { s += bitcast<f32>(W[a + l]) * X[p.xoff + u * 32u + l]; }
  return s;
}`,
  Q8_0: BYTES + /* wgsl */`
fn unit(row: u32, u: u32, K: u32) -> f32 {
  let a = (row * (K / 32u) + u) * 34u; let d = f16at(a); var s = 0.0; let xb = p.xoff + u * 32u;
  for (var l = 0u; l < 32u; l += 4u) {
    let w = w32(a + 2u + l);
    s += s8(w) * X[xb + l] + s8(w >> 8u) * X[xb + l + 1u] + s8(w >> 16u) * X[xb + l + 2u] + s8(w >> 24u) * X[xb + l + 3u];
  }
  return d * s;
}`,
  Q4_K: SCALE_K4 + /* wgsl */`
fn unit(row: u32, u: u32, K: u32) -> f32 {
  let b = u >> 3u; let sub = u & 7u; let j = sub >> 1u; let hi = (sub & 1u) * 4u;
  let bw = (row * (K / 256u) + b) * 36u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let xb = p.xoff + b * 256u + sub * 32u; var sq = 0.0; var sx = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let w = W[bw + 4u + j * 8u + k] >> hi;
    let x = vec4f(X[xb + 4u * k], X[xb + 4u * k + 1u], X[xb + 4u * k + 2u], X[xb + 4u * k + 3u]);
    let q = vec4f(f32(w & 0xfu), f32((w >> 8u) & 0xfu), f32((w >> 16u) & 0xfu), f32((w >> 24u) & 0xfu));
    sq += dot(q, x); sx += x.x + x.y + x.z + x.w;
  }
  return dm.x * sm.x * sq - dm.y * sm.y * sx;
}`,
  Q5_K: SCALE_K4 + /* wgsl */`
fn unit(row: u32, u: u32, K: u32) -> f32 {
  let b = u >> 3u; let sub = u & 7u; let j = sub >> 1u; let hi = (sub & 1u) * 4u;
  let bw = (row * (K / 256u) + b) * 44u;
  let dm = unpack2x16float(W[bw]); let sm = scmin(bw, sub);
  let xb = p.xoff + b * 256u + sub * 32u; var sq = 0.0; var sx = 0.0;
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
fn unit(row: u32, u: u32, K: u32) -> f32 {
  let b = u >> 3u; let n = (u >> 2u) & 1u; let sub = u & 3u;
  let a = (row * (K / 256u) + b) * 210u;
  let d = f16at(a + 208u);
  let qla = a + 64u * n + (sub & 1u) * 32u; let nib = (sub >> 1u) * 4u;
  let qha = a + 128u + 32u * n; let sh = 2u * sub;
  let sca = a + 192u + 8u * n + 2u * sub;
  let xb = p.xoff + b * 256u + 128u * n + 32u * sub;
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
export const TRUNK_QUANTS = Object.keys(UNIT);

// y[yoff + row] (+= if p.acc) = W[row] . x[xoff ..]; 8 rows a workgroup, 32 lanes a row, rows over a 2-D grid
export const gemv = (q) => /* wgsl */`
struct P { rows: u32, K: u32, xoff: u32, yoff: u32, acc: u32, gx: u32, r0: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
${UNIT[q]}
@compute @workgroup_size(32, 8)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let row = (g.y * p.gx + g.x) * 8u + l.y;
  var acc = 0.0;
  if (row < p.rows) { let nu = p.K / 32u; for (var u = l.x; u < nu; u += 32u) { acc += unit(row, u, p.K); } }
  let i = l.y * 32u + l.x; red[i] = acc; workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) { if (l.x < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  if (l.x == 0u && row < p.rows) { let o = p.yoff + p.r0 + row; if (p.acc == 1u) { Y[o] += red[i]; } else { Y[o] = red[i]; } }
}`;

// embedding gather: y[k] = dequant(W row tok)[k]; the token id is ids[0]. One thread a 32-value unit, via a ones-free dequant.
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
export const embed = (q) => /* wgsl */`
struct P { K: u32, rows0: u32, nrows: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${DQ[q]}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let t = ids[0];
  if (gi.x < p.K && t >= p.rows0 && t - p.rows0 < p.nrows) { Y[gi.x] = dq(t - p.rows0, gi.x, p.K); }
}`;

// RMS norm per head: y[yoff + h*dim + i] = x[xoff + h*xs + i] * rsqrt(mean + eps) * w[i] (mode 1: no weight, L2: x / sqrt(sum + eps))
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

// y = a + b (n words)
export const vadd = /* wgsl */`
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) { if (gi.x < p.n) { Y[gi.x] = A[gi.x] + B[gi.x]; } }`;

// swiglu: y[i] = silu(g[i]) * u[i]; gate and up stacked in one buffer (g at 0, u at n)
export const swiglu = /* wgsl */`
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> GU: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  if (gi.x < p.n) { let g = GU[gi.x]; Y[gi.x] = g / (1.0 + exp(-g)) * GU[p.n + gi.x]; }
}`;

// causal conv1d step (ggml_ssm_conv, kernel 4) + silu; the state keeps the last 3 inputs per channel, oldest first
export const conv = /* wgsl */`
struct P { C: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wc: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3u) {
  let c = gi.x; if (c >= p.C) { return; }
  let s0 = S[c * 3u]; let s1 = S[c * 3u + 1u]; let s2 = S[c * 3u + 2u]; let x = X[c];
  let v = Wc[c * 4u] * s0 + Wc[c * 4u + 1u] * s1 + Wc[c * 4u + 2u] * s2 + Wc[c * 4u + 3u] * x;
  S[c * 3u] = s1; S[c * 3u + 1u] = s2; S[c * 3u + 2u] = x;
  Y[c] = v / (1.0 + exp(-v));
}`;

// one gated-delta-net step for v-head h (workgroup), lane j = a value column (dim S = 128):
//   g = softplus(alpha + dt) * a; beta = sigmoid(b); s *= exp(g); d = (v - s^T k) beta; s += k d^T; o = s^T q / sqrt(S)
// then the gated norm: o * rsqrt(mean(o^2) + eps) * ssm_norm * silu(z). q, k = the (L2-normed) k-head h % HK (ggml_repeat tiling).
// The state is s[h][i][j] (i = key dim) so lanes read adjacent words.
export const delta = /* wgsl */`
struct P { HK: u32, HV: u32, eps: f32, _a: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<f32>;    // conv output: q [HK*S], k [HK*S], v [HV*S]
@group(0) @binding(1) var<storage, read> AB: array<f32>;     // alpha [HV] at 0, beta [HV] at HV (raw GEMV outputs)
@group(0) @binding(2) var<storage, read> Pr: array<f32>;     // dt bias [HV] at 0, a [HV] at HV, ssm_norm [S] at 2 HV
@group(0) @binding(3) var<storage, read_write> St: array<f32>;
@group(0) @binding(4) var<storage, read> Z: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let h = g.x; let j = l.x; let kh = h % p.HK;
  let q0 = QKV[kh * 128u + j]; let k0 = QKV[p.HK * 128u + kh * 128u + j];
  red[j] = q0 * q0; workgroupBarrier();                // the L2 norms of q and k (x / sqrt(sum x^2 + eps), build_gdn_l2_norm)
  for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
  qs[j] = q0 * inverseSqrt(red[0] + p.eps); workgroupBarrier();
  red[j] = k0 * k0; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
  ks[j] = k0 * inverseSqrt(red[0] + p.eps); workgroupBarrier();
  let a = AB[h] + Pr[h]; let sp = select(log(1.0 + exp(a)), a, a > 20.0);
  let eg = exp(sp * Pr[p.HV + h]); let beta = 1.0 / (1.0 + exp(-AB[p.HV + h]));
  let base = h * 16384u + j;
  var sk = 0.0;
  for (var i = 0u; i < 128u; i++) { sk += St[base + i * 128u] * eg * ks[i]; }
  let d = (QKV[2u * p.HK * 128u + h * 128u + j] - sk) * beta;
  var o = 0.0;
  for (var i = 0u; i < 128u; i++) { let s = St[base + i * 128u] * eg + ks[i] * d; St[base + i * 128u] = s; o += s * qs[i]; }
  o *= 0.08838834764831845;                   // 1 / sqrt(128)
  red[j] = o * o; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (j < k) { red[j] += red[j + k]; } workgroupBarrier(); }
  let z = Z[h * 128u + j];
  O[h * 128u + j] = o * inverseSqrt(red[0] / 128.0 + p.eps) * Pr[2u * p.HV + j] * (z / (1.0 + exp(-z)));
}`;

// full-attention prep, one workgroup a head (NH query heads, then NKV key heads), lane i = a head dim (D = 256):
//   q = rms(qg[h*2D .. +D]) * qn, roped; k = rms(k[h*D ..]) * kn, roped, stored at Kc[pos]; v stored at Vc[pos]
//   rope = NEOX over the first R dims (MRoPE with one position for every section), theta = pos * base^(-2i/R)
export const attnprep = /* wgsl */`
struct P { NH: u32, NKV: u32, R: u32, eps: f32, base: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> QG: array<f32>;
@group(0) @binding(1) var<storage, read> KV: array<f32>;     // k [NKV*D] at 0, v [NKV*D] at NKV*D
@group(0) @binding(2) var<storage, read> Nw: array<f32>;     // q_norm [D] at 0, k_norm [D] at D
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> Vc: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<storage, read> stp: array<u32>;   // [0] = pos
var<workgroup> v: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let i = l.x; let isq = g.x < p.NH; let pos = stp[0];
  var x: f32;
  if (isq) { x = QG[g.x * 512u + i]; } else { x = KV[(g.x - p.NH) * 256u + i]; }
  red[i] = x * x; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (i < k) { red[i] += red[i + k]; } workgroupBarrier(); }
  var nw: f32; if (isq) { nw = Nw[i]; } else { nw = Nw[256u + i]; }
  v[i] = x * inverseSqrt(red[0] / 256.0 + p.eps) * nw;
  workgroupBarrier();
  var y = v[i]; let hf = p.R / 2u;
  if (i < p.R) {
    let fi = i % hf; let th = f32(pos) * pow(p.base, -2.0 * f32(fi) / f32(p.R));
    let c = cos(th); let s = sin(th);
    if (i < hf) { y = v[i] * c - v[i + hf] * s; } else { y = v[i - hf] * s + v[i] * c; }
  }
  if (isq) { Q[g.x * 256u + i] = y; }
  else {
    let kh = g.x - p.NH; let at = (pos * p.NKV + kh) * 256u + i;
    Kc[at] = y; Vc[at] = KV[p.NKV * 256u + kh * 256u + i];
  }
}`;

// attention for query head h (workgroup): softmax(q . K[0..pos] / 16) . V, times sigmoid(gate) (gate = qg[h*2D + D ..])
export const attn = /* wgsl */`
struct P { NH: u32, NKV: u32, ctx: u32, _a: u32 }
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> Kc: array<f32>;
@group(0) @binding(2) var<storage, read> Vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Sc: array<f32>;
@group(0) @binding(4) var<storage, read> QG: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<storage, read> stp: array<u32>;
var<workgroup> q: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let h = g.x; let i = l.x; let n = stp[0] + 1u; let kh = h / (p.NH / p.NKV); let sb = h * p.ctx;
  q[i] = Q[h * 256u + i]; workgroupBarrier();
  var mx = -3.0e38;
  for (var t = i; t < n; t += 256u) {
    let kb = (t * p.NKV + kh) * 256u; var s = 0.0;
    for (var d = 0u; d < 256u; d += 4u) {
      s += dot(vec4f(q[d], q[d + 1u], q[d + 2u], q[d + 3u]), vec4f(Kc[kb + d], Kc[kb + d + 1u], Kc[kb + d + 2u], Kc[kb + d + 3u]));
    }
    s *= 0.0625; Sc[sb + t] = s; mx = max(mx, s);
  }
  red[i] = mx; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (i < k) { red[i] = max(red[i], red[i + k]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var t = i; t < n; t += 256u) { let e = exp(Sc[sb + t] - m); Sc[sb + t] = e; sum += e; }
  red[i] = sum; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (i < k) { red[i] += red[i + k]; } workgroupBarrier(); }
  let tot = red[0];
  storageBarrier(); workgroupBarrier();
  var acc = 0.0;
  for (var t = 0u; t < n; t++) { acc += Sc[sb + t] * Vc[(t * p.NKV + kh) * 256u + i]; }
  let gt = QG[h * 512u + 256u + i];
  O[h * 256u + i] = acc / tot / (1.0 + exp(-gt));
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
