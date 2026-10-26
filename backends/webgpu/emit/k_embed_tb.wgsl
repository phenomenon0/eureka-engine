// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"map","name":"embed_tb","entry":"map_embed_tb_main","lanes":256,"ins":[{"name":"wb","ty":"bytes","cls":"bytes","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"chunk_row0","ty":"u32","at":2},{"name":"c","ty":"u32","at":3},{"name":"nt","ty":"u32","at":4},{"name":"id0","ty":"u32","at":5},{"name":"id1","ty":"u32","at":6},{"name":"id2","ty":"u32","at":7},{"name":"id3","ty":"u32","at":8},{"name":"id4","ty":"u32","at":9},{"name":"id5","ty":"u32","at":10},{"name":"id6","ty":"u32","at":11},{"name":"id7","ty":"u32","at":12},{"name":"id8","ty":"u32","at":13},{"name":"id9","ty":"u32","at":14},{"name":"id10","ty":"u32","at":15},{"name":"id11","ty":"u32","at":16},{"name":"id12","ty":"u32","at":17},{"name":"id13","ty":"u32","at":18},{"name":"id14","ty":"u32","at":19},{"name":"id15","ty":"u32","at":20},{"name":"id16","ty":"u32","at":21},{"name":"id17","ty":"u32","at":22},{"name":"id18","ty":"u32","at":23},{"name":"id19","ty":"u32","at":24},{"name":"id20","ty":"u32","at":25},{"name":"id21","ty":"u32","at":26},{"name":"id22","ty":"u32","at":27},{"name":"id23","ty":"u32","at":28},{"name":"id24","ty":"u32","at":29},{"name":"id25","ty":"u32","at":30},{"name":"id26","ty":"u32","at":31},{"name":"id27","ty":"u32","at":32},{"name":"id28","ty":"u32","at":33},{"name":"id29","ty":"u32","at":34},{"name":"id30","ty":"u32","at":35},{"name":"id31","ty":"u32","at":36}],"lens":[{"name":"wb","at":1}]},"len_at":0,"tokens":true}]}
// webprefill2 token-batched embed (bin/gen-embed.mjs from k_embed.wgsl): every shipped fn byte-identical + embed_tb
// emitted by bend (wgsl.ts): the program's lane defs and runtime_wgsl

// X32
// ===

// x32_div / x32_sqrt: IEEE-754 binary32 a / b and sqrt(v), round to nearest even, the full domain
// (subnormal operands and results, zeros, infinities, NaN -> 0x7fc00000), as the C lane's f32 ops.
// Integer-only: restoring long division / digit-by-digit square root; the result cannot depend on
// the GPU's f32 unit (flush-to-zero, fma contraction, 2.5-ulp division). Replaces wgsl.ts's
// hardware-candidate version (x32_mul/x32_shl/x32_lt/x32_normal/x32_man/x32_exp/x32_side).

// q * 2^e (+ a sticky tail below q's last bit), q in [2^25, 2^26): f32 bits with sign sg
fn x32d_pack(sg: u32, q: u32, sticky: bool, e: i32) -> u32 {
  let eb = e + 25 + 127;
  if (eb >= 255) {
    return 0x7f800000u | sg;
  }
  let sh = select(3 - eb, 2, eb >= 1);
  if (sh > 26) {
    return sg;
  }
  let s = u32(sh);
  let kept = q >> s;
  let rem = q & ((1u << s) - 1u);
  let hf = 1u << (s - 1u);
  let up = rem > hf || (rem == hf && (sticky || (kept & 1u) == 1u));
  var b = kept + select(0u, 1u, up);
  if (eb >= 1) {
    b += u32(eb - 1) << 23u;
  }
  return b | sg;
}

// |x| = m 2^e, m in [2^23, 2^24), for finite nonzero magnitude bits a; vec2(m, bitcast e)
fn x32d_norm(a: u32) -> vec2<u32> {
  if (a >= 0x800000u) {
    return vec2<u32>((a & 0x7fffffu) | 0x800000u, bitcast<u32>(i32(a >> 23u) - 150));
  }
  let z = countLeadingZeros(a) - 8u;
  return vec2<u32>(a << z, bitcast<u32>(-149 - i32(z)));
}

fn x32d_div_bits(ua: u32, ub: u32) -> u32 {
  let sg = (ua ^ ub) & 0x80000000u;
  let aa = ua & 0x7fffffffu;
  let ab = ub & 0x7fffffffu;
  if (aa > 0x7f800000u || ab > 0x7f800000u) {
    return 0x7fc00000u;
  }
  if (aa == 0x7f800000u) {
    return select(0x7f800000u | sg, 0x7fc00000u, ab == 0x7f800000u);
  }
  if (ab == 0x7f800000u) {
    return sg;
  }
  if (ab == 0u) {
    return select(0x7f800000u | sg, 0x7fc00000u, aa == 0u);
  }
  if (aa == 0u) {
    return sg;
  }
  let na = x32d_norm(aa);
  let nb = x32d_norm(ab);
  var ma = na.x;
  let mb = nb.x;
  var e = bitcast<i32>(na.y) - bitcast<i32>(nb.y) - 25;
  if (ma < mb) {
    ma = ma << 1u;
    e -= 1;
  }
  // q = floor(ma 2^25 / mb) in [2^25, 2^26)
  var r = ma;
  var q = 0u;
  for (var i = 0; i < 26; i++) {
    q = q << 1u;
    if (r >= mb) {
      r -= mb;
      q |= 1u;
    }
    r = r << 1u;
  }
  return x32d_pack(sg, q, r != 0u, e);
}

fn x32d_sqrt_bits(uv: u32) -> u32 {
  if (uv == 0x80000000u || uv == 0u || uv == 0x7f800000u) {
    return uv;
  }
  if (uv > 0x7f800000u) {
    return 0x7fc00000u;                                  // NaN or negative
  }
  let n = x32d_norm(uv);
  var m = n.x << 1u;
  var e = bitcast<i32>(n.y) - 1;
  if ((e & 1) != 0) {
    m = m << 1u;
    e -= 1;
  }
  // m in [2^24, 2^26), e even; root of N = m 2^24 in [2^24, 2^25), 25 pairs of N's bits
  var r = 0u;
  var q = 0u;
  for (var i = 0u; i < 25u; i++) {
    var pair = 0u;
    if (i < 13u) {
      pair = (m >> (24u - 2u * i)) & 3u;
    }
    r = (r << 2u) | pair;
    let t = (q << 2u) | 1u;
    if (r >= t) {
      r -= t;
      q = (q << 1u) | 1u;
    } else {
      q = q << 1u;
    }
  }
  return x32d_pack(0u, q << 1u, r != 0u, (e - 24) / 2 - 1);
}

fn x32_div(a: f32, b: f32) -> f32 {
  return bitcast<f32>(x32d_div_bits(bitcast<u32>(a), bitcast<u32>(b)));
}

fn x32_sqrt(v: f32) -> f32 {
  return bitcast<f32>(x32d_sqrt_bits(bitcast<u32>(v)));
}

// x32w u64: a u64 is vec2<u32>(hi, lo); WGSL has no u64 and no mulhi.
// Integer-only (no f32 arithmetic anywhere in the x32 helpers).

// a * b, full 64 bits, from 16-bit limbs
fn x32w_m32(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu;
  let a1 = a >> 16u;
  let b0 = b & 0xFFFFu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = p01 + (p00 >> 16u) + (p10 & 0xFFFFu);
  return vec2<u32>(p11 + (mid >> 16u) + (p10 >> 16u), (mid << 16u) | (p00 & 0xFFFFu));
}

fn x32w_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.y + b.y;
  return vec2<u32>(a.x + b.x + select(0u, 1u, lo < a.y), lo);
}

fn x32w_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x - b.x - select(0u, 1u, a.y < b.y), a.y - b.y);
}

fn x32w_neg(a: vec2<u32>) -> vec2<u32> {
  return x32w_sub(vec2<u32>(0u, 0u), a);
}

fn x32w_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

fn x32w_eq(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.x == b.x && a.y == b.y;
}

fn x32w_isz(a: vec2<u32>) -> bool {
  return (a.x | a.y) == 0u;
}

// a << s, 0 for s >= 64 (as the C code's guarded shifts)
fn x32w_shl(a: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) {
    return a;
  }
  if (s >= 64u) {
    return vec2<u32>(0u, 0u);
  }
  if (s >= 32u) {
    return vec2<u32>(a.y << (s - 32u), 0u);
  }
  return vec2<u32>((a.x << s) | (a.y >> (32u - s)), a.y << s);
}

fn x32w_shr(a: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) {
    return a;
  }
  if (s >= 64u) {
    return vec2<u32>(0u, 0u);
  }
  if (s >= 32u) {
    return vec2<u32>(0u, a.x >> (s - 32u));
  }
  return vec2<u32>(a.x >> s, (a.y >> s) | (a.x << (32u - s)));
}

// 2^s - 1, s < 64
fn x32w_mask(s: u32) -> vec2<u32> {
  return x32w_sub(x32w_shl(vec2<u32>(0u, 1u), s), vec2<u32>(0u, 1u));
}

fn x32w_and(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x & b.x, a.y & b.y);
}

fn x32w_or(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x | b.x, a.y | b.y);
}

// leading zeros, 64 for 0
fn x32w_clz(a: vec2<u32>) -> u32 {
  if (a.x != 0u) {
    return countLeadingZeros(a.x);
  }
  return 32u + countLeadingZeros(a.y);
}

fn x32w_isneg(a: vec2<u32>) -> bool {
  return (a.x >> 31u) != 0u;
}

// a * b mod 2^64
fn x32w_mullo(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let p = x32w_m32(a.y, b.y);
  return vec2<u32>(p.x + a.x * b.y + a.y * b.x, p.y);
}

// (a * b) >> 64, unsigned
fn x32w_mulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let p00 = x32w_m32(a.y, b.y);
  let p01 = x32w_m32(a.y, b.x);
  let p10 = x32w_m32(a.x, b.y);
  let p11 = x32w_m32(a.x, b.x);
  let s1 = p00.x + p01.y;
  let s2 = s1 + p10.y;
  let c = select(0u, 1u, s1 < p00.x) + select(0u, 1u, s2 < s1);
  var r = x32w_add(p11, vec2<u32>(0u, p01.x));
  r = x32w_add(r, vec2<u32>(0u, p10.x));
  return x32w_add(r, vec2<u32>(0u, c));
}

// (a * b) >> 64, signed (two's complement)
fn x32w_mulhis(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  var r = x32w_mulhi(a, b);
  if (x32w_isneg(a)) {
    r = x32w_sub(r, b);
  }
  if (x32w_isneg(b)) {
    r = x32w_sub(r, a);
  }
  return r;
}

// Q63 product of signed fractions (x32.h x32_mul63)
fn x32w_mul63(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return x32w_shl(x32w_mulhis(a, b), 1u);
}

// round M/2^63 * 2^E (M's top bit set) to a double, then to f32 (x32.h x32_round); f32 bits
fn x32w_round(M: vec2<u32>, E: i32, sgn: u32) -> u32 {
  var D = x32w_add(x32w_shr(M, 11u), vec2<u32>(0u, (M.y >> 10u) & 1u));
  var e = E;
  if ((D.x >> 21u) != 0u) {
    D = x32w_shr(D, 1u);
    e += 1;
  }
  let eb = e + 127;
  if (eb >= 255) {
    return 0x7f800000u | sgn;
  }
  let sh = select(30 - eb, 29, eb >= 1);
  if (sh >= 64) {
    return sgn;
  }
  let s = u32(sh);
  let q = x32w_shr(D, s);
  let rem = x32w_and(D, x32w_mask(s));
  let hf = x32w_shl(vec2<u32>(0u, 1u), s - 1u);
  let up = x32w_lt(hf, rem) || (x32w_eq(rem, hf) && (q.y & 1u) == 1u);
  var b = q.y + select(0u, 1u, up);
  if (eb >= 1) {
    b += u32(eb - 1) << 23u;
  }
  return b | sgn;
}

// (float)v for v != 0, round to nearest even: v ~ m * 2^e, m in [2^23, 2^24); vec2(m, bitcast e)
fn x32w_u64f(v: vec2<u32>) -> vec2<u32> {
  let z = x32w_clz(v);
  let top = x32w_shl(v, z);
  var m = top.x >> 8u;
  let rest = vec2<u32>(top.x & 0xFFu, top.y);
  let hf = vec2<u32>(0x80u, 0u);
  if (x32w_lt(hf, rest) || (x32w_eq(rest, hf) && (m & 1u) == 1u)) {
    m += 1u;
  }
  var e = 40 - i32(z);
  if (m == 0x1000000u) {
    m = 0x800000u;
    e += 1;
  }
  return vec2<u32>(m, bitcast<u32>(e));
}

// f32 product (ma 2^ea) (mb 2^eb), both mantissas in [2^23, 2^24), rounded to nearest
// even (no overflow / underflow handling: callers stay in range); vec2(m, bitcast e)
fn x32w_fmul(ma: u32, ea: i32, mb: u32, eb: i32) -> vec2<u32> {
  let P = x32w_m32(ma, mb);
  var sh = 23u;
  if ((P.x >> 15u) != 0u) {
    sh = 24u;
  }
  var m = x32w_shr(P, sh).y;
  let rem = x32w_and(P, x32w_mask(sh));
  let hf = x32w_shl(vec2<u32>(0u, 1u), sh - 1u);
  if (x32w_lt(hf, rem) || (x32w_eq(rem, hf) && (m & 1u) == 1u)) {
    m += 1u;
  }
  var e = ea + eb + i32(sh);
  if (m == 0x1000000u) {
    m = 0x800000u;
    e += 1;
  }
  return vec2<u32>(m, bitcast<u32>(e));
}

fn x32a_jam(x: u32, d: u32) -> u32 {
  if (d == 0u) { return x; }
  if (d >= 32u) { return select(0u, 1u, x != 0u); }
  return (x >> d) | select(0u, 1u, (x & ((1u << d) - 1u)) != 0u);
}

// ARM's scalar arithmetic quiets the first signaling NaN before considering
// quiet NaNs. Payload/sign are retained; invalid finite/inf ops use +qNaN.
fn x32a_nan(a: u32, b: u32) -> u32 {
  let aa = a & 0x7fffffffu;
  let ab = b & 0x7fffffffu;
  if (aa > 0x7f800000u && (a & 0x400000u) == 0u) { return a | 0x400000u; }
  if (ab > 0x7f800000u && (b & 0x400000u) == 0u) { return b | 0x400000u; }
  if (aa > 0x7f800000u) { return a | 0x400000u; }
  return b | 0x400000u;
}

fn x32a_add_bits(ua: u32, ub: u32) -> u32 {
  let aa = ua & 0x7fffffffu;
  let ab = ub & 0x7fffffffu;
  if (aa > 0x7f800000u || ab > 0x7f800000u) { return x32a_nan(ua, ub); }
  if (aa == 0x7f800000u) {
    if (ab == aa && ua != ub) { return 0x7fc00000u; }
    return ua;
  }
  if (ab == 0x7f800000u) { return ub; }
  if (aa == 0u && ab == 0u) { return (ua & ub) & 0x80000000u; }
  if (aa == 0u) { return ub; }
  if (ab == 0u) { return ua; }
  // Keep the greater magnitude on the left, so cancellation has its sign.
  let a = select(ub, ua, aa >= ab);
  let b = select(ua, ub, aa >= ab);
  let na = x32d_norm(a & 0x7fffffffu);
  let nb = x32d_norm(b & 0x7fffffffu);
  var e = bitcast<i32>(na.y) - 3;
  let delta = u32(bitcast<i32>(na.y) - bitcast<i32>(nb.y));
  let ma = na.x << 3u;
  let mb = x32a_jam(nb.x << 3u, delta);
  var m = 0u;
  if (((a ^ b) & 0x80000000u) == 0u) {
    m = ma + mb;
    if ((m & 0x8000000u) != 0u) { m = x32a_jam(m, 1u); e += 1; }
  } else {
    m = ma - mb;
    if (m == 0u) { return 0u; }
    let shift = countLeadingZeros(m) - 5u;
    m = m << shift;
    e -= i32(shift);
  }
  return x32d_pack(a & 0x80000000u, m >> 1u, (m & 1u) != 0u, e + 1);
}

fn x32a_mul_bits(ua: u32, ub: u32) -> u32 {
  let aa = ua & 0x7fffffffu;
  let ab = ub & 0x7fffffffu;
  let sign = (ua ^ ub) & 0x80000000u;
  if (aa > 0x7f800000u || ab > 0x7f800000u) { return x32a_nan(ua, ub); }
  if (aa == 0x7f800000u || ab == 0x7f800000u) {
    if (aa == 0u || ab == 0u) { return 0x7fc00000u; }
    return sign | 0x7f800000u;
  }
  if (aa == 0u || ab == 0u) { return sign; }
  let na = x32d_norm(aa);
  let nb = x32d_norm(ab);
  let p = x32w_m32(na.x, nb.x);
  // A 48-bit product, bit 46 or 47 set. Keep 26 bits and a sticky tail.
  let shift = select(21u, 22u, (p.x & 0x8000u) != 0u);
  let q = (p.x << (32u - shift)) | (p.y >> shift);
  let tail = (p.y & ((1u << shift) - 1u)) != 0u;
  return x32d_pack(sign, q, tail, bitcast<i32>(na.y) + bitcast<i32>(nb.y) + i32(shift));
}

// The hardware result h is taken when its exponent field alone shows it is
// exact (P2.6 U9a.4): for a product, h in 2 .. 254 (a subnormal operand is
// flushed, giving a zero h, or used exactly; NaN, infinity and overflow have
// field 255); for a sum or difference, h in 27 .. 254, |h| >= 2^-100, where a
// subnormal operand is below half an ulp of h, so flushing it or not rounds
// alike. Normal operands with such a result are rounded once, correctly. The
// fma forms are p2's barriers (no fusion or reassociation across them). Else,
// the exact answers the integer path gives without rounding (x32a_zok): x + 0
// = x, 0 + 0 by the sign rule, x + (-x) = +0, 0 * finite = a signed zero;
// else the integer path (cancellation and underflow into the subnormals,
// overflow, NaN payloads).
fn x32a_okadd(uh: u32) -> bool {
  return (((uh >> 23u) & 0xffu) - 27u) < 228u;
}
fn x32a_okmul(uh: u32) -> bool {
  return (((uh >> 23u) & 0xffu) - 2u) < 253u;
}

fn x32a_zok(ua: u32, ub: u32, mul: bool) -> bool {
  let fin = ((ua & 0x7f800000u) != 0x7f800000u) & ((ub & 0x7f800000u) != 0x7f800000u);
  let z = ((ua & 0x7fffffffu) == 0u) | ((ub & 0x7fffffffu) == 0u);
  return fin & (z | (!mul & (ua == (ub ^ 0x80000000u))));
}

// a + b when x32a_zok: the exact zero-operand / cancellation answer
fn x32a_zadd(ua: u32, ub: u32) -> u32 {
  let za = (ua & 0x7fffffffu) == 0u;
  let zb = (ub & 0x7fffffffu) == 0u;
  return select(select(select(0u, ub, za), ua, zb), ua & ub & 0x80000000u, za & zb);
}

fn x32_add(a: f32, b: f32) -> f32 {
  let h = fma(1.0, a, b);
  if (x32a_okadd(bitcast<u32>(h))) { return h; }
  let ua = bitcast<u32>(a);
  let ub = bitcast<u32>(b);
  if (x32a_zok(ua, ub, false)) { return bitcast<f32>(x32a_zadd(ua, ub)); }
  return bitcast<f32>(x32a_add_bits(ua, ub));
}
fn x32_sub(a: f32, b: f32) -> f32 {
  let h = fma(-1.0, b, a);
  if (x32a_okadd(bitcast<u32>(h))) { return h; }
  let ua = bitcast<u32>(a);
  let ub = bitcast<u32>(b);
  let nb = ub ^ 0x80000000u;
  if (x32a_zok(ua, nb, false)) { return bitcast<f32>(x32a_zadd(ua, nb)); }
  if ((ua & 0x7fffffffu) > 0x7f800000u || (ub & 0x7fffffffu) > 0x7f800000u) {
    return bitcast<f32>(x32a_nan(ua, ub));
  }
  return bitcast<f32>(x32a_add_bits(ua, nb));
}
fn x32_mul(a: f32, b: f32) -> f32 {
  let h = fma(a, b, -0.0);
  if (x32a_okmul(bitcast<u32>(h))) { return h; }
  let ua = bitcast<u32>(a);
  let ub = bitcast<u32>(b);
  if (x32a_zok(ua, ub, true)) { return bitcast<f32>((ua ^ ub) & 0x80000000u); }
  return bitcast<f32>(x32a_mul_bits(ua, ub));
}

// F32
// ===

fn fisnan(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn fne(a: f32, b: f32) -> bool {
  return fisnan(a) || fisnan(b) || a != b;
}

// a * b rounded once, never fused into a sum, a zero keeping its sign (fma(a, b,
// -0.0) alone gives +0 for a zero product of opposite signs on Apple's GPUs)
fn bmul(a: f32, b: f32) -> f32 {
  let q = fma(a, b, -0.0);
  return select(q, bitcast<f32>((bitcast<u32>(a) ^ bitcast<u32>(b)) & 0x80000000u),
    q == 0.0);
}

// Nat
// ===

// A Nat is a u32 here; past 2^32 - 1 the lane sets nat_ovf and the run
// answers "past the envelope" (the C and JS lanes hold any Nat).
var<private> nat_ovf: u32;

fn nat_add(a: u32, b: u32) -> u32 {
  let s = a + b;
  if (s < a) {
    nat_ovf = 1u;
  }
  return s;
}

// a b = a1 b1 2^32 + (a1 b0 + a0 b1) 2^16 + a0 b0, a = a1 2^16 + a0
fn nat_mul(a: u32, b: u32) -> u32 {
  let a1 = a >> 16u;
  let b1 = b >> 16u;
  let mid = a1 * (b & 0xFFFFu) + b1 * (a & 0xFFFFu);
  let lo = (a & 0xFFFFu) * (b & 0xFFFFu);
  if ((a1 != 0u && b1 != 0u) || mid > 0xFFFFu || lo + (mid << 16u) < lo) {
    nat_ovf = 1u;
  }
  return a * b;
}

// Program
// =======

fn b_map_embed_f_map_embed_wb(_i_0: u32, _id_0: u32, _chunk_row0_0: u32) -> f32 {
  let _row_0 = (_id_0 - _chunk_row0_0);
  return b____tree_engine_kernels_kq_q6k_w1_map_embed_wb((_row_0 * 2100u), _i_0);
}

fn b____tree_engine_kernels_kq_q6k_w1_map_embed_wb(_r_0: u32, _c_0: u32) -> f32 {
  let _blk_0 = select(_c_0 >> 8u, 0u, 8u >= 32u);
  let _rr_0 = (_c_0 & 255u);
  let _n_0 = select(_rr_0 >> 7u, 0u, 7u >= 32u);
  let _rd_0 = (_rr_0 & 127u);
  let _quad_0 = select(_rd_0 >> 5u, 0u, 5u >= 32u);
  let _l_0 = (_rd_0 & 31u);
  let _at_0 = (_r_0 + (_blk_0 * 210u));
  let _d_0 = b____tree_engine_kernels_flat_q8_f16((bw_map_embed_wb((_at_0 + 208u)) & 65535u));
  let _sc_0 = b____tree_engine_kernels_flat_q6k_sc8_map_embed_wb(((_at_0 + 192u) + ((_n_0 * 8u) + ((_quad_0 * 2u) + select(_l_0 >> 4u, 0u, 4u >= 32u)))));
  let _ds_0 = x32_mul(_d_0, _sc_0);
  let _nsh_0 = select(0u, 4u, (2u <= _quad_0));
  let _qsh_0 = (_quad_0 * 2u);
  let _qlb_0 = (_at_0 + ((_n_0 * 64u) + (((_quad_0 & 1u) * 32u) + _l_0)));
  let _qhb_0 = ((_at_0 + 128u) + ((_n_0 * 32u) + _l_0));
  return x32_mul(_ds_0, b____tree_engine_kernels_flat_q6k_code(b____tree_engine_kernels_flat_q6k_by_map_embed_wb(_qlb_0), b____tree_engine_kernels_flat_q6k_by_map_embed_wb(_qhb_0), _nsh_0, _qsh_0));
}

fn b____tree_engine_kernels_flat_q8_f16(_h_0: u32) -> f32 {
  let _e_0 = (select(_h_0 >> 10u, 0u, 10u >= 32u) & 31u);
  let _m_0 = (_h_0 & 1023u);
  let _s_0 = select(select(_h_0 >> 15u, 0u, 15u >= 32u) << 31u, 0u, 31u >= 32u);
  let _p_0 = select((select(_e_0, 1u, (_e_0 == 0u)) + 102u) << 23u, 0u, 23u >= 32u);
  let _mag_0 = x32_mul(f32(select((_m_0 + 1024u), _m_0, (_e_0 == 0u))), b____tree_engine_kernels_ops_f32_from_bits(_p_0));
  return b____tree_engine_kernels_ops_f32_from_bits(select((bitcast<u32>(_mag_0) | _s_0), ((_s_0 | 2139095040u) | select(_m_0 << 13u, 0u, 13u >= 32u)), (_e_0 == 31u)));
}

fn b____tree_engine_kernels_flat_q6k_sc8_map_embed_wb(_at_0: u32) -> f32 {
  let _b_0 = b____tree_engine_kernels_flat_q6k_by_map_embed_wb(_at_0);
  return select(x32_sub(f32(_b_0), bitcast<f32>(1132462080u)), f32(_b_0), (_b_0 < 128u));
}

fn b____tree_engine_kernels_flat_q6k_by_map_embed_wb(_at_0: u32) -> u32 {
  return (bw_map_embed_wb(_at_0) & 255u);
}

fn b____tree_engine_kernels_flat_q6k_code(_q_0: u32, _h_0: u32, _nsh_0: u32, _sh_0: u32) -> f32 {
  let _v_0 = ((select(_q_0 >> _nsh_0, 0u, _nsh_0 >= 32u) & 15u) | select((select(_h_0 >> _sh_0, 0u, _sh_0 >= 32u) & 3u) << 4u, 0u, 4u >= 32u));
  return x32_sub(f32(_v_0), bitcast<f32>(1107296256u));
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}

fn bg_map_embed_wb(i: u32) -> u32 {
  if (i >= map_embed__sc[1u]) {
    return 0u;
  }
  return (map_embed_wb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_embed_wb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_embed__sc[1u]) {
    return map_embed_wb[i >> 2u];
  }
  return bg_map_embed_wb(i) | (bg_map_embed_wb(i + 1u) << 8u) | (bg_map_embed_wb(i + 2u) << 16u)
    | (bg_map_embed_wb(i + 3u) << 24u);
}


// Map embed
// ====

@group(0) @binding(0) var<storage, read> map_embed_wb: array<u32>;
@group(0) @binding(1) var<storage, read_write> map_embed__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_embed__sc: array<u32>;

@compute @workgroup_size(256)
fn map_embed_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_embed__sc[0u]) {
    return;
  }
  map_embed__out[i] = b_map_embed_f_map_embed_wb(i, map_embed__sc[2u], map_embed__sc[3u]);
}

// embed_tb (webprefill2): the embed gather for the nt tokens of one head chunk; tk = workgroup_id.y
@compute @workgroup_size(256)
fn map_embed_tb_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let i = gid.x; let tk = wg.y;
  if (i >= map_embed__sc[0u] || tk >= map_embed__sc[4u]) {
    return;
  }
  let id = map_embed__sc[5u + tk];
  if (id / 62080u != map_embed__sc[3u]) {
    return;
  }
  map_embed__out[tk * 4096u + i] = b_map_embed_f_map_embed_wb(i, id, map_embed__sc[2u]);
}
