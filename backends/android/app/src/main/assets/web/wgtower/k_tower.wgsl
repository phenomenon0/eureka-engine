// k_tower.wgsl: the Qwen3.5 vision tower (clip qwen3vl_merger mmproj) as WGSL. Hand-written, hardware f32 arithmetic
// (like k_fuse_t.wgsl: NOT bit-exact; gated per kernel vs llama.cpp's CPU dumps and end to end by tokens).
// Precision mirrors the registered CPU reference (cudavis/mac/vision/host/vit.c, f16act): weights stay F16 on the GPU
// (two halves a u32, unpack2x16float: no shader-f16 needed), GEMM inputs and im2col are rounded to F16 and accumulated in
// F32, attention K/V are rounded to F16, softmax / LayerNorm / GELU / RoPE are F32. Entries:
//   gemm    Y[n][o] = X[n][k] . W[o][k]^T, epilogue flags (bias, residual accumulate, GELU-tanh); 64x64 tiles, 4x4 a thread
//   lnorm   LayerNorm (+w, +b), a workgroup a row
//   im2col  planar [3][ny][nx] -> patch rows [N][3*16*16] in 2x2-merge order (the conv3d's frame; frame 1 = frame 0)
//   posadd  + the learned 48x48 position table, bilinear (align corners) to the patch grid (host-computed taps)
//   addb    y[r][e] += b[e]
//   rope    2D axial vision RoPE (pairs (i, i+D/2); i < D/4 row pos, else column pos; host-computed cos/sin tables)
//   attn    bidirectional (full, unmasked) MHA over all patches, online softmax, 16 queries x 32 keys a tile
struct P { n: u32, k: u32, o: u32, f: u32, a: u32, b: u32, c: u32, d: u32 };
const GELU_A: f32 = 0.044715;
const S2PI: f32 = 0.79788456080286535587989211986876;
const NEG: f32 = -1.0e30;
fn r16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }
fn gelu(v: f32) -> f32 { let t = clamp(S2PI * v * (1.0 + GELU_A * v * v), -15.0, 15.0); return 0.5 * v * (1.0 + tanh(t)); }
fn gid(g: vec3<u32>, ng: vec3<u32>, l: u32) -> u32 { return (g.y * ng.x + g.x) * 256u + l; }

// ---- gemm. f: 1 = round X to F16, 2 = + bias, 4 = GELU after, 8 = accumulate (y = y_old + v), 16 = bias after accumulate
@group(0) @binding(0) var<uniform> gp: P;
@group(0) @binding(1) var<storage, read> gx: array<f32>;
@group(0) @binding(2) var<storage, read> gw: array<u32>;
@group(0) @binding(3) var<storage, read> gb: array<f32>;
@group(0) @binding(4) var<storage, read_write> gy: array<f32>;
var<workgroup> ta: array<array<f32, 64>, 16>;
var<workgroup> tb: array<array<f32, 64>, 16>;
@compute @workgroup_size(256)
fn gemm(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let n = gp.n; let K = gp.k; let O = gp.o; let f = gp.f;
  let r0 = g.y * 64u; let c0 = g.x * 64u;
  let tr = (l / 16u) * 4u; let tc = (l % 16u) * 4u;
  var acc: array<vec4<f32>, 4>;
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    for (var i = 0u; i < 4u; i += 1u) {
      let idx = l + i * 256u; let row = idx / 16u; let kk = idx % 16u;
      var v = 0.0;
      if (r0 + row < n) { v = gx[(r0 + row) * K + k0 + kk]; if ((f & 1u) != 0u) { v = r16(v); } }
      ta[kk][row] = v;
    }
    for (var i = 0u; i < 2u; i += 1u) {
      let idx = l + i * 256u; let col = idx / 8u; let kp = idx % 8u;
      let w = unpack2x16float(gw[((c0 + col) * K + k0) / 2u + kp]);
      tb[kp * 2u][col] = w.x; tb[kp * 2u + 1u][col] = w.y;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk += 1u) {
      let b = vec4<f32>(tb[kk][tc], tb[kk][tc + 1u], tb[kk][tc + 2u], tb[kk][tc + 3u]);
      for (var i = 0u; i < 4u; i += 1u) { acc[i] += ta[kk][tr + i] * b; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i += 1u) {
    let r = r0 + tr + i;
    if (r >= n) { continue; }
    for (var j = 0u; j < 4u; j += 1u) {
      let c = c0 + tc + j; let at = r * O + c;
      var v = acc[i][j];
      if ((f & 2u) != 0u && (f & 16u) == 0u) { v = v + gb[c]; }
      if ((f & 8u) != 0u) { v = gy[at] + v; }
      if ((f & 16u) != 0u) { v = v + gb[c]; }
      if ((f & 4u) != 0u) { v = gelu(v); }
      gy[at] = v;
    }
  }
}

// ---- lnorm: n rows of k; eps = bitcast a
@group(0) @binding(0) var<uniform> lp: P;
@group(0) @binding(1) var<storage, read> lx: array<f32>;
@group(0) @binding(2) var<storage, read> lw: array<f32>;
@group(0) @binding(3) var<storage, read> lb: array<f32>;
@group(0) @binding(4) var<storage, read_write> ly: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn lnorm(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let E = lp.k; let base = g.x * E; let eps = bitcast<f32>(lp.a);
  var s = 0.0;
  for (var i = l; i < E; i += 256u) { s += lx[base + i]; }
  red[l] = s; workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) { if (l < h) { red[l] += red[l + h]; } workgroupBarrier(); }
  let mean = red[0] / f32(E);
  workgroupBarrier();
  var v = 0.0;
  for (var i = l; i < E; i += 256u) { let d = lx[base + i] - mean; v += d * d; }
  red[l] = v; workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) { if (l < h) { red[l] += red[l + h]; } workgroupBarrier(); }
  let scale = 1.0 / sqrt(red[0] / f32(E) + eps);
  for (var i = l; i < E; i += 256u) { ly[base + i] = ((lx[base + i] - mean) * scale) * lw[i] + lb[i]; }
}

// patch r (2x2-merge order) -> (row y, column x) on the patch grid of px columns
fn merge_yx(r: u32, px: u32) -> vec2<u32> {
  let b = r / 4u; let q = r % 4u; let half = px / 2u;
  return vec2<u32>((b / half) * 2u + q / 2u, (b % half) * 2u + q % 2u);
}

// ---- im2col: n = N patches, a = nx, b = ny, c = px, f & 1 = round F16
@group(0) @binding(0) var<uniform> ip: P;
@group(0) @binding(1) var<storage, read> iplanar: array<f32>;
@group(0) @binding(4) var<storage, read_write> irows: array<f32>;
@compute @workgroup_size(256)
fn im2col(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid(g, ng, l);
  if (i >= ip.n * 768u) { return; }
  let r = i / 768u; let c = i % 768u; let ch = c / 256u; let ky = (c / 16u) % 16u; let kx = c % 16u;
  let yx = merge_yx(r, ip.c);
  var v = iplanar[ch * ip.a * ip.b + (yx.x * 16u + ky) * ip.a + yx.y * 16u + kx];
  if ((ip.f & 1u) != 0u) { v = r16(v); }
  irows[i] = v;
}

// ---- posadd: n = N, k = E, c = px, a = G; taps = [px x (i0, i1, fd bits)] then [py x (i0, i1, fd bits)]
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read> ptab: array<f32>;
@group(0) @binding(2) var<storage, read> ptaps: array<u32>;
@group(0) @binding(4) var<storage, read_write> px_: array<f32>;
@compute @workgroup_size(256)
fn posadd(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid(g, ng, l);
  let E = pp.k;
  if (i >= pp.n * E) { return; }
  let r = i / E; let e = i % E; let G = pp.a;
  let yx = merge_yx(r, pp.c);
  let tx = yx.y * 3u; let ty = pp.c * 3u + yx.x * 3u;
  let i00 = ptaps[tx]; let i00p = ptaps[tx + 1u]; let fd0 = bitcast<f32>(ptaps[tx + 2u]);
  let i01 = ptaps[ty]; let i01p = ptaps[ty + 1u]; let fd1 = bitcast<f32>(ptaps[ty + 2u]);
  let s00 = ptab[(i01 * G + i00) * E + e]; let s10 = ptab[(i01 * G + i00p) * E + e];
  let s01 = ptab[(i01p * G + i00) * E + e]; let s11 = ptab[(i01p * G + i00p) * E + e];
  let o = s00 * (1.0 - fd0) * (1.0 - fd1) + s10 * fd0 * (1.0 - fd1) + s01 * (1.0 - fd0) * fd1 + s11 * fd0 * fd1;
  px_[i] = px_[i] + o;
}

// ---- addb: n rows of k
@group(0) @binding(0) var<uniform> bp: P;
@group(0) @binding(3) var<storage, read> bb_: array<f32>;
@group(0) @binding(4) var<storage, read_write> by_: array<f32>;
@compute @workgroup_size(256)
fn addb(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid(g, ng, l);
  if (i >= bp.n * bp.k) { return; }
  by_[i] = by_[i] + bb_[i % bp.k];
}

// ---- rope: in place on qkv [n][3E]; k = E, a = H, b = D, c = offset (0 = Q, E = K); cs/sn [n][D/2]
@group(0) @binding(0) var<uniform> rp: P;
@group(0) @binding(1) var<storage, read> rcs: array<f32>;
@group(0) @binding(2) var<storage, read> rsn: array<f32>;
@group(0) @binding(4) var<storage, read_write> rq: array<f32>;
@compute @workgroup_size(256)
fn rope(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid(g, ng, l);
  let nd = rp.b / 2u;
  if (i >= rp.n * rp.a * nd) { return; }
  let ic = i % nd; let h = (i / nd) % rp.a; let r = i / (nd * rp.a);
  let at = r * 3u * rp.k + rp.c + h * rp.b;
  let c = rcs[r * nd + ic]; let s = rsn[r * nd + ic];
  let x0 = rq[at + ic]; let x1 = rq[at + ic + nd];
  rq[at + ic] = x0 * c - x1 * s;
  rq[at + ic + nd] = x0 * s + x1 * c;
}

// ---- attn: n patches, k = E, D = 64; grid (ceil(n/16), H); f & 1 = round K/V to F16. out [n][E]
@group(0) @binding(0) var<uniform> ap: P;
@group(0) @binding(1) var<storage, read> aqkv: array<f32>;
@group(0) @binding(4) var<storage, read_write> aout: array<f32>;
var<workgroup> qs: array<array<f32, 64>, 16>;
var<workgroup> ks: array<array<f32, 64>, 32>;
var<workgroup> vs: array<array<f32, 64>, 32>;
var<workgroup> ss: array<array<f32, 32>, 16>;
@compute @workgroup_size(256)
fn attn(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let n = ap.n; let E = ap.k; let ld = 3u * E; let h = g.y; let q0 = g.x * 16u; let qi = l / 16u; let t = l % 16u;
  let rnd = (ap.f & 1u) != 0u;
  for (var i = 0u; i < 4u; i += 1u) {
    let idx = l + i * 256u; let row = idx / 64u; let d = idx % 64u;
    var v = 0.0;
    if (q0 + row < n) { v = aqkv[(q0 + row) * ld + h * 64u + d]; }
    qs[row][d] = v;
  }
  var m = NEG; var lsum = 0.0; var o = vec4<f32>(0.0);
  for (var j0 = 0u; j0 < n; j0 += 32u) {
    workgroupBarrier();
    for (var i = 0u; i < 8u; i += 1u) {
      let idx = l + i * 256u; let row = idx / 64u; let d = idx % 64u; let key = j0 + row;
      var kv = 0.0; var vv = 0.0;
      if (key < n) {
        kv = aqkv[key * ld + E + h * 64u + d]; vv = aqkv[key * ld + 2u * E + h * 64u + d];
        if (rnd) { kv = r16(kv); vv = r16(vv); }
      }
      ks[row][d] = kv; vs[row][d] = vv;
    }
    workgroupBarrier();
    for (var u = 0u; u < 2u; u += 1u) {
      let jj = t + u * 16u;
      var s = NEG;
      if (j0 + jj < n) {
        var dot = 0.0;
        for (var d = 0u; d < 64u; d += 1u) { dot += qs[qi][d] * ks[jj][d]; }
        s = dot * 0.125;
      }
      ss[qi][jj] = s;
    }
    workgroupBarrier();
    var mx = m;
    for (var jj = 0u; jj < 32u; jj += 1u) { mx = max(mx, ss[qi][jj]); }
    let corr = exp(m - mx);
    lsum = lsum * corr; o = o * corr;
    for (var jj = 0u; jj < 32u; jj += 1u) {
      let p = exp(ss[qi][jj] - mx);
      lsum += p;
      o += p * vec4<f32>(vs[jj][4u * t], vs[jj][4u * t + 1u], vs[jj][4u * t + 2u], vs[jj][4u * t + 3u]);
    }
    m = mx;
  }
  if (q0 + qi < n) {
    let inv = 1.0 / lsum;
    for (var d = 0u; d < 4u; d += 1u) { aout[(q0 + qi) * E + h * 64u + 4u * t + d] = o[d] * inv; }
  }
}
