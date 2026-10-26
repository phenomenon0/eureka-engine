// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"map","name":"conv","entry":"map_conv_main","lanes":256,"ins":[{"name":"xb","ty":"bytes","cls":"bytes","binding":0},{"name":"hb","ty":"bytes","cls":"bytes","binding":1},{"name":"wb","ty":"bytes","cls":"bytes","binding":2}],"out":{"ty":"f32","binding":3},"sc":{"binding":4,"scalars":[],"lens":[{"name":"xb","at":1},{"name":"hb","at":2},{"name":"wb","at":3}]},"len_at":0},{"kind":"cube","name":"delta","entry":"cube_delta_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"sb","ty":"bytes","cls":"bytes","binding":0},{"name":"kb","ty":"bytes","cls":"bytes","binding":1},{"name":"db","ty":"bytes","cls":"bytes","binding":2},{"name":"vb","ty":"bytes","cls":"bytes","binding":3},{"name":"bb","ty":"bytes","cls":"bytes","binding":4}],"out":{"ty":"f32","binding":5},"sc":{"binding":6,"scalars":[],"lens":[{"name":"sb","at":1},{"name":"kb","at":2},{"name":"db","at":3},{"name":"vb","at":4},{"name":"bb","at":5}]},"rows_at":0},{"kind":"map","name":"state","entry":"map_state_main","lanes":256,"ins":[{"name":"sb","ty":"bytes","cls":"bytes","binding":0},{"name":"kb","ty":"bytes","cls":"bytes","binding":1},{"name":"db","ty":"bytes","cls":"bytes","binding":2},{"name":"eb","ty":"bytes","cls":"bytes","binding":3}],"out":{"ty":"f32","binding":4},"sc":{"binding":5,"scalars":[],"lens":[{"name":"sb","at":1},{"name":"kb","at":2},{"name":"db","at":3},{"name":"eb","at":4}]},"len_at":0},{"kind":"cube","name":"readout","entry":"cube_readout_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"sb","ty":"bytes","cls":"bytes","binding":0},{"name":"qb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"sb","at":1},{"name":"qb","at":2}]},"rows_at":0},{"kind":"map","name":"history","entry":"map_history_main","lanes":256,"ins":[{"name":"hb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"hb","at":1},{"name":"xb","at":2}]},"len_at":0}]}
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

// x32_exp: (f32)exp((double)x), x32.h x32_exp_core / x32_exp_parts / x32_exp

struct x32w_me {
  m: vec2<u32>,
  e: i32,
}

// kk = (int)rint((float)hi * 0x1p-56f * 92.332482616893656f), in integers
fn x32w_kk(hi: vec2<u32>) -> i32 {
  if (x32w_isz(hi)) {
    return 0;
  }
  let f = x32w_u64f(hi);
  // 92.332482616893656f = 0xb8aa3b * 2^-17
  let p = x32w_fmul(f.x, bitcast<i32>(f.y) - 56, 0xb8aa3bu, -17);
  let ep = bitcast<i32>(p.y);
  if (ep >= 0) {
    return i32(p.x << u32(ep));
  }
  let s = u32(-ep);
  if (s >= 25u) {
    return 0;
  }
  let q = p.x >> s;
  let rem = p.x & ((1u << s) - 1u);
  let hf = 1u << (s - 1u);
  return i32(q + select(0u, 1u, rem > hf || (rem == hf && (q & 1u) == 1u)));
}

// e^t for t = +-(hi:lo)/2^120, |t| < 2^7, as M/2^63 * 2^E (M's top bit set)
fn x32w_exp_parts(neg: bool, hi: vec2<u32>, lo: vec2<u32>) -> x32w_me {
  let kk = x32w_kk(hi);
  let ukk = vec2<u32>(0u, u32(kk));
  let blo = x32w_mullo(ukk, vec2<u32>(0xe6af278eu, 0xce600fccu));
  let bhi = x32w_add(x32w_mullo(ukk, vec2<u32>(0x0002c5c8u, 0x5fdf473du)),
                     x32w_mulhi(ukk, vec2<u32>(0xe6af278eu, 0xce600fccu)));
  let dlo = x32w_sub(lo, blo);
  let dhi = x32w_sub(x32w_sub(hi, bhi), vec2<u32>(0u, select(0u, 1u, x32w_lt(lo, blo))));
  var r = x32w_or(x32w_shl(dhi, 7u), x32w_shr(dlo, 57u));
  var k = kk;
  if (neg) {
    r = x32w_neg(r);
    k = -k;
  }
  var q = x32w_expc[6];
  for (var n = 5; n >= 2; n--) {
    q = x32w_add(x32w_expc[n], x32w_mul63(q, r));
  }
  let p = x32w_add(r, x32w_mul63(x32w_mul63(r, r), q));
  let T = x32w_exp2j[u32(k & 63)];
  let pn = x32w_isneg(p);
  let d = x32w_mulhi(T, select(p, x32w_neg(p), pn));
  let h = x32w_shr(T, 1u);
  let V = select(x32w_add(h, d), x32w_sub(h, d), pn);
  let z = x32w_clz(V);
  return x32w_me(x32w_shl(V, z), (k >> 6u) + 1 - i32(z));
}

fn x32w_exp_core(neg: bool, hi: vec2<u32>, lo: vec2<u32>) -> u32 {
  let me = x32w_exp_parts(neg, hi, lo);
  return x32w_round(me.m, me.e, 0u);
}

fn x32w_exp_bits(u: u32) -> u32 {
  let a = u & 0x7fffffffu;
  let sg = (u >> 31u) != 0u;
  if (a >= 0x7f800000u) {
    if (a > 0x7f800000u) {
      return 0x7fc00000u;
    }
    return select(u, 0u, sg);
  }
  if (a < 0x32800000u) {
    return 0x3f800000u;
  }
  if (a >= 0x42d00000u) {
    return select(0x7f800000u, 0u, sg);
  }
  let m = (a & 0x7fffffu) | 0x800000u;
  return x32w_exp_core(sg, x32w_shl(vec2<u32>(0u, m), (a >> 23u) - 94u), vec2<u32>(0u, 0u));
}

fn x32_exp(x: f32) -> f32 {
  return bitcast<f32>(x32w_exp_bits(bitcast<u32>(x)));
}

const x32w_exp2j = array<vec2<u32>, 64>(
  vec2<u32>(0x80000000u, 0x00000000u),
  vec2<u32>(0x8164d1f3u, 0xbc030773u),
  vec2<u32>(0x82cd8698u, 0xac2ba1d7u),
  vec2<u32>(0x843a28c3u, 0xacde4046u),
  vec2<u32>(0x85aac367u, 0xcc487b15u),
  vec2<u32>(0x871f6196u, 0x9e8d1010u),
  vec2<u32>(0x88980e80u, 0x92da8527u),
  vec2<u32>(0x8a14d575u, 0x496efd9au),
  vec2<u32>(0x8b95c1e3u, 0xea8bd6e7u),
  vec2<u32>(0x8d1adf5bu, 0x7e5ba9e6u),
  vec2<u32>(0x8ea4398bu, 0x45cd53c0u),
  vec2<u32>(0x9031dc43u, 0x1466b1dcu),
  vec2<u32>(0x91c3d373u, 0xab11c336u),
  vec2<u32>(0x935a2b2fu, 0x13e6e92cu),
  vec2<u32>(0x94f4efa8u, 0xfef70961u),
  vec2<u32>(0x96942d37u, 0x20185a00u),
  vec2<u32>(0x9837f051u, 0x8db8a96fu),
  vec2<u32>(0x99e04593u, 0x20b7fa65u),
  vec2<u32>(0x9b8d39b9u, 0xd54e5539u),
  vec2<u32>(0x9d3ed9a7u, 0x2cffb751u),
  vec2<u32>(0x9ef53260u, 0x91a111aeu),
  vec2<u32>(0xa0b0510fu, 0xb9714fc2u),
  vec2<u32>(0xa2704303u, 0x0c496819u),
  vec2<u32>(0xa43515aeu, 0x09e6809eu),
  vec2<u32>(0xa5fed6a9u, 0xb15138eau),
  vec2<u32>(0xa7cd93b4u, 0xe965356au),
  vec2<u32>(0xa9a15ab4u, 0xea7c0ef8u),
  vec2<u32>(0xab7a39b5u, 0xa93ed337u),
  vec2<u32>(0xad583eeau, 0x42a14ac6u),
  vec2<u32>(0xaf3b78adu, 0x690a4375u),
  vec2<u32>(0xb123f581u, 0xd2ac2590u),
  vec2<u32>(0xb311c412u, 0xa9112489u),
  vec2<u32>(0xb504f333u, 0xf9de6484u),
  vec2<u32>(0xb6fd91e3u, 0x28d17791u),
  vec2<u32>(0xb8fbaf47u, 0x62fb9ee9u),
  vec2<u32>(0xbaff5ab2u, 0x133e45fbu),
  vec2<u32>(0xbd08a39fu, 0x580c36bfu),
  vec2<u32>(0xbf1799b6u, 0x7a731083u),
  vec2<u32>(0xc12c4ccau, 0x66709456u),
  vec2<u32>(0xc346ccdau, 0x24976407u),
  vec2<u32>(0xc5672a11u, 0x5506daddu),
  vec2<u32>(0xc78d74c8u, 0xabb9b15du),
  vec2<u32>(0xc9b9bd86u, 0x6e2f27a3u),
  vec2<u32>(0xcbec14feu, 0xf2727c5du),
  vec2<u32>(0xce248c15u, 0x1f8480e4u),
  vec2<u32>(0xd06333dau, 0xef2b2595u),
  vec2<u32>(0xd2a81d91u, 0xf12ae45au),
  vec2<u32>(0xd4f35aabu, 0xcfedfa1fu),
  vec2<u32>(0xd744fccau, 0xd69d6af4u),
  vec2<u32>(0xd99d15c2u, 0x78afd7b6u),
  vec2<u32>(0xdbfbb797u, 0xdaf23755u),
  vec2<u32>(0xde60f482u, 0x5e0e9124u),
  vec2<u32>(0xe0ccdeecu, 0x2a94e111u),
  vec2<u32>(0xe33f8972u, 0xbe8a5a51u),
  vec2<u32>(0xe5b906e7u, 0x7c8348a8u),
  vec2<u32>(0xe8396a50u, 0x3c4bdc68u),
  vec2<u32>(0xeac0c6e7u, 0xdd24392fu),
  vec2<u32>(0xed4f301eu, 0xd9942b84u),
  vec2<u32>(0xefe4b99bu, 0xdcdaf5cbu),
  vec2<u32>(0xf281773cu, 0x59ffb13au),
  vec2<u32>(0xf5257d15u, 0x2486cc2cu),
  vec2<u32>(0xf7d0df73u, 0x0ad13bb9u),
  vec2<u32>(0xfa83b2dbu, 0x722a033au),
  vec2<u32>(0xfd3e0c0cu, 0xf486c175u));

const x32w_expc = array<vec2<u32>, 7>(
  vec2<u32>(0x00000000u, 0x00000000u),
  vec2<u32>(0x00000000u, 0x00000000u),
  vec2<u32>(0x40000000u, 0x00000000u),
  vec2<u32>(0x15555555u, 0x55555555u),
  vec2<u32>(0x05555555u, 0x55555555u),
  vec2<u32>(0x01111111u, 0x11111111u),
  vec2<u32>(0x002d82d8u, 0x2d82d82eu));

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

fn b_map_conv_f_map_conv_xb_map_conv_hb_map_conv_wb(_i_0: u32) -> f32 {
  let _w_0 = (4u * _i_0);
  let _a_0 = x32_add(x32_mul(b_word_map_conv_hb(_i_0), b_word_map_conv_wb(_w_0)), x32_mul(b_word_map_conv_hb((8192u + _i_0)), b_word_map_conv_wb((_w_0 + 1u))));
  let _b_0 = x32_add(x32_mul(b_word_map_conv_hb((16384u + _i_0)), b_word_map_conv_wb((_w_0 + 2u))), x32_mul(b_word_map_conv_xb(_i_0), b_word_map_conv_wb((_w_0 + 3u))));
  return b____tree_engine_kernels_ops_silu1(x32_add(_a_0, _b_0));
}

fn b_cube_delta_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_delta_leaf_cube_delta_sb_cube_delta_kb_cube_delta_db(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(x32_mul(b_word_cube_delta_sb(((128u * _row_0) + _i_0)), b_word_cube_delta_db(select(_row_0 >> 7u, 0u, 7u >= 32u))), b_word_cube_delta_kb(((select(_row_0 >> 7u, 0u, 7u >= 32u) * 128u) + _i_0)));
}

fn b_cube_delta_fin_cube_delta_vb_cube_delta_bb(_t_0: f32, _row_0: u32) -> f32 {
  return x32_mul(x32_sub(b_word_cube_delta_vb(_row_0), _t_0), b_word_cube_delta_bb(select(_row_0 >> 7u, 0u, 7u >= 32u)));
}

fn b_map_state_f_map_state_sb_map_state_kb_map_state_db_map_state_eb(_i_0: u32) -> f32 {
  let _row_0 = select(_i_0 >> 7u, 0u, 7u >= 32u);
  let _head_0 = select(_row_0 >> 7u, 0u, 7u >= 32u);
  return x32_add(x32_mul(b_word_map_state_sb(_i_0), b_word_map_state_db(_head_0)), x32_mul(b_word_map_state_kb(((_head_0 * 128u) + (_i_0 & 127u))), b_word_map_state_eb(_row_0)));
}

fn b_cube_readout_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_readout_leaf_cube_readout_sb_cube_readout_qb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_readout_sb(((_row_0 * 128u) + _i_0)), b_word_cube_readout_qb(((select(_row_0 >> 7u, 0u, 7u >= 32u) * 128u) + _i_0)));
}

fn b_cube_readout_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_history_f_map_history_hb_map_history_xb(_i_0: u32) -> f32 {
  return b_map_history_f_pick0_map_history_hb_map_history_xb((_i_0 < 16384u), _i_0);
}

fn b_word_map_conv_hb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_conv_hb((4u * _i_0)));
}

fn b_word_map_conv_wb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_conv_wb((4u * _i_0)));
}

fn b_word_map_conv_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_conv_xb((4u * _i_0)));
}

fn b____tree_engine_kernels_ops_silu1(_x_0: f32) -> f32 {
  let _v_0 = _x_0;
  return x32_div(_v_0, x32_add(bitcast<f32>(1065353216u), b____tree_engine_kernels_ops_exp32(bitcast<f32>(bitcast<u32>(_v_0) ^ 0x80000000u))));
}

fn b_word_cube_delta_sb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_delta_sb((4u * _i_0)));
}

fn b_word_cube_delta_db(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_delta_db((4u * _i_0)));
}

fn b_word_cube_delta_kb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_delta_kb((4u * _i_0)));
}

fn b_word_cube_delta_vb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_delta_vb((4u * _i_0)));
}

fn b_word_cube_delta_bb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_delta_bb((4u * _i_0)));
}

fn b_word_map_state_sb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_state_sb((4u * _i_0)));
}

fn b_word_map_state_db(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_state_db((4u * _i_0)));
}

fn b_word_map_state_kb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_state_kb((4u * _i_0)));
}

fn b_word_map_state_eb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_state_eb((4u * _i_0)));
}

fn b_word_cube_readout_sb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_readout_sb((4u * _i_0)));
}

fn b_word_cube_readout_qb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_readout_qb((4u * _i_0)));
}

fn b_map_history_f_pick0_map_history_hb_map_history_xb(_c_0: bool, _i_0: u32) -> f32 {
  if (!_c_0) {
    return b_word_map_history_xb((_i_0 - 16384u));
  } else {
    return b_word_map_history_hb((_i_0 + 8192u));
  }
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}

fn b____tree_engine_kernels_ops_exp32(_x_0: f32) -> f32 {
  return x32_exp(_x_0);
}

fn b_word_map_history_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_history_xb((4u * _i_0)));
}

fn b_word_map_history_hb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_history_hb((4u * _i_0)));
}

fn bg_map_conv_xb(i: u32) -> u32 {
  if (i >= map_conv__sc[1u]) {
    return 0u;
  }
  return (map_conv_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_conv_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_conv__sc[1u]) {
    return map_conv_xb[i >> 2u];
  }
  return bg_map_conv_xb(i) | (bg_map_conv_xb(i + 1u) << 8u) | (bg_map_conv_xb(i + 2u) << 16u)
    | (bg_map_conv_xb(i + 3u) << 24u);
}

fn bg_map_conv_hb(i: u32) -> u32 {
  if (i >= map_conv__sc[2u]) {
    return 0u;
  }
  return (map_conv_hb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_conv_hb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_conv__sc[2u]) {
    return map_conv_hb[i >> 2u];
  }
  return bg_map_conv_hb(i) | (bg_map_conv_hb(i + 1u) << 8u) | (bg_map_conv_hb(i + 2u) << 16u)
    | (bg_map_conv_hb(i + 3u) << 24u);
}

fn bg_map_conv_wb(i: u32) -> u32 {
  if (i >= map_conv__sc[3u]) {
    return 0u;
  }
  return (map_conv_wb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_conv_wb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_conv__sc[3u]) {
    return map_conv_wb[i >> 2u];
  }
  return bg_map_conv_wb(i) | (bg_map_conv_wb(i + 1u) << 8u) | (bg_map_conv_wb(i + 2u) << 16u)
    | (bg_map_conv_wb(i + 3u) << 24u);
}

fn bg_cube_delta_sb(i: u32) -> u32 {
  if (i >= cube_delta__sc[1u]) {
    return 0u;
  }
  return (cube_delta_sb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_delta_sb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_delta__sc[1u]) {
    return cube_delta_sb[i >> 2u];
  }
  return bg_cube_delta_sb(i) | (bg_cube_delta_sb(i + 1u) << 8u) | (bg_cube_delta_sb(i + 2u) << 16u)
    | (bg_cube_delta_sb(i + 3u) << 24u);
}

fn bg_cube_delta_kb(i: u32) -> u32 {
  if (i >= cube_delta__sc[2u]) {
    return 0u;
  }
  return (cube_delta_kb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_delta_kb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_delta__sc[2u]) {
    return cube_delta_kb[i >> 2u];
  }
  return bg_cube_delta_kb(i) | (bg_cube_delta_kb(i + 1u) << 8u) | (bg_cube_delta_kb(i + 2u) << 16u)
    | (bg_cube_delta_kb(i + 3u) << 24u);
}

fn bg_cube_delta_db(i: u32) -> u32 {
  if (i >= cube_delta__sc[3u]) {
    return 0u;
  }
  return (cube_delta_db[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_delta_db(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_delta__sc[3u]) {
    return cube_delta_db[i >> 2u];
  }
  return bg_cube_delta_db(i) | (bg_cube_delta_db(i + 1u) << 8u) | (bg_cube_delta_db(i + 2u) << 16u)
    | (bg_cube_delta_db(i + 3u) << 24u);
}

fn bg_cube_delta_vb(i: u32) -> u32 {
  if (i >= cube_delta__sc[4u]) {
    return 0u;
  }
  return (cube_delta_vb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_delta_vb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_delta__sc[4u]) {
    return cube_delta_vb[i >> 2u];
  }
  return bg_cube_delta_vb(i) | (bg_cube_delta_vb(i + 1u) << 8u) | (bg_cube_delta_vb(i + 2u) << 16u)
    | (bg_cube_delta_vb(i + 3u) << 24u);
}

fn bg_cube_delta_bb(i: u32) -> u32 {
  if (i >= cube_delta__sc[5u]) {
    return 0u;
  }
  return (cube_delta_bb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_delta_bb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_delta__sc[5u]) {
    return cube_delta_bb[i >> 2u];
  }
  return bg_cube_delta_bb(i) | (bg_cube_delta_bb(i + 1u) << 8u) | (bg_cube_delta_bb(i + 2u) << 16u)
    | (bg_cube_delta_bb(i + 3u) << 24u);
}

fn bg_map_state_sb(i: u32) -> u32 {
  if (i >= map_state__sc[1u]) {
    return 0u;
  }
  return (map_state_sb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_state_sb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_state__sc[1u]) {
    return map_state_sb[i >> 2u];
  }
  return bg_map_state_sb(i) | (bg_map_state_sb(i + 1u) << 8u) | (bg_map_state_sb(i + 2u) << 16u)
    | (bg_map_state_sb(i + 3u) << 24u);
}

fn bg_map_state_kb(i: u32) -> u32 {
  if (i >= map_state__sc[2u]) {
    return 0u;
  }
  return (map_state_kb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_state_kb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_state__sc[2u]) {
    return map_state_kb[i >> 2u];
  }
  return bg_map_state_kb(i) | (bg_map_state_kb(i + 1u) << 8u) | (bg_map_state_kb(i + 2u) << 16u)
    | (bg_map_state_kb(i + 3u) << 24u);
}

fn bg_map_state_db(i: u32) -> u32 {
  if (i >= map_state__sc[3u]) {
    return 0u;
  }
  return (map_state_db[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_state_db(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_state__sc[3u]) {
    return map_state_db[i >> 2u];
  }
  return bg_map_state_db(i) | (bg_map_state_db(i + 1u) << 8u) | (bg_map_state_db(i + 2u) << 16u)
    | (bg_map_state_db(i + 3u) << 24u);
}

fn bg_map_state_eb(i: u32) -> u32 {
  if (i >= map_state__sc[4u]) {
    return 0u;
  }
  return (map_state_eb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_state_eb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_state__sc[4u]) {
    return map_state_eb[i >> 2u];
  }
  return bg_map_state_eb(i) | (bg_map_state_eb(i + 1u) << 8u) | (bg_map_state_eb(i + 2u) << 16u)
    | (bg_map_state_eb(i + 3u) << 24u);
}

fn bg_cube_readout_sb(i: u32) -> u32 {
  if (i >= cube_readout__sc[1u]) {
    return 0u;
  }
  return (cube_readout_sb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_readout_sb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_readout__sc[1u]) {
    return cube_readout_sb[i >> 2u];
  }
  return bg_cube_readout_sb(i) | (bg_cube_readout_sb(i + 1u) << 8u) | (bg_cube_readout_sb(i + 2u) << 16u)
    | (bg_cube_readout_sb(i + 3u) << 24u);
}

fn bg_cube_readout_qb(i: u32) -> u32 {
  if (i >= cube_readout__sc[2u]) {
    return 0u;
  }
  return (cube_readout_qb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_readout_qb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_readout__sc[2u]) {
    return cube_readout_qb[i >> 2u];
  }
  return bg_cube_readout_qb(i) | (bg_cube_readout_qb(i + 1u) << 8u) | (bg_cube_readout_qb(i + 2u) << 16u)
    | (bg_cube_readout_qb(i + 3u) << 24u);
}

fn bg_map_history_hb(i: u32) -> u32 {
  if (i >= map_history__sc[1u]) {
    return 0u;
  }
  return (map_history_hb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_history_hb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_history__sc[1u]) {
    return map_history_hb[i >> 2u];
  }
  return bg_map_history_hb(i) | (bg_map_history_hb(i + 1u) << 8u) | (bg_map_history_hb(i + 2u) << 16u)
    | (bg_map_history_hb(i + 3u) << 24u);
}

fn bg_map_history_xb(i: u32) -> u32 {
  if (i >= map_history__sc[2u]) {
    return 0u;
  }
  return (map_history_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_history_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_history__sc[2u]) {
    return map_history_xb[i >> 2u];
  }
  return bg_map_history_xb(i) | (bg_map_history_xb(i + 1u) << 8u) | (bg_map_history_xb(i + 2u) << 16u)
    | (bg_map_history_xb(i + 3u) << 24u);
}


// Map conv
// ====

@group(0) @binding(0) var<storage, read> map_conv_xb: array<u32>;
@group(0) @binding(1) var<storage, read> map_conv_hb: array<u32>;
@group(0) @binding(2) var<storage, read> map_conv_wb: array<u32>;
@group(0) @binding(3) var<storage, read_write> map_conv__out: array<f32>;
@group(0) @binding(4) var<storage, read> map_conv__sc: array<u32>;

@compute @workgroup_size(256)
fn map_conv_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_conv__sc[0u]) {
    return;
  }
  map_conv__out[i] = b_map_conv_f_map_conv_xb_map_conv_hb_map_conv_wb(i);
}

// Cube delta
// ====

const cube_delta__N: u32 = 128u;
const cube_delta__LEN: u32 = 128u;
const cube_delta__T: u32 = 128u;
const cube_delta__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_delta_sb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_delta_kb: array<u32>;
@group(0) @binding(2) var<storage, read> cube_delta_db: array<u32>;
@group(0) @binding(3) var<storage, read> cube_delta_vb: array<u32>;
@group(0) @binding(4) var<storage, read> cube_delta_bb: array<u32>;
@group(0) @binding(5) var<storage, read_write> cube_delta__out: array<f32>;
@group(0) @binding(6) var<storage, read> cube_delta__sc: array<u32>;

var<workgroup> cube_delta__t: array<f32, cube_delta__T>;


@compute @workgroup_size(cube_delta__T)
fn cube_delta_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_delta__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_delta__Q; j += 1u) {
    let i = l * cube_delta__Q + j;
    var v: f32;
    if (i < cube_delta__LEN) {
      v = b_cube_delta_leaf_cube_delta_sb_cube_delta_kb_cube_delta_db(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_delta_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_delta__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_delta__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_delta__T; p += cube_delta__T * 2u * h) {
      cube_delta__t[p] = b_cube_delta_join(cube_delta__t[p], cube_delta__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_delta__out[row] = b_cube_delta_fin_cube_delta_vb_cube_delta_bb(cube_delta__t[0], row);
  }
}

// Map state
// ====

@group(0) @binding(0) var<storage, read> map_state_sb: array<u32>;
@group(0) @binding(1) var<storage, read> map_state_kb: array<u32>;
@group(0) @binding(2) var<storage, read> map_state_db: array<u32>;
@group(0) @binding(3) var<storage, read> map_state_eb: array<u32>;
@group(0) @binding(4) var<storage, read_write> map_state__out: array<f32>;
@group(0) @binding(5) var<storage, read> map_state__sc: array<u32>;

@compute @workgroup_size(256)
fn map_state_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_state__sc[0u]) {
    return;
  }
  map_state__out[i] = b_map_state_f_map_state_sb_map_state_kb_map_state_db_map_state_eb(i);
}

// Cube readout
// ====

const cube_readout__N: u32 = 128u;
const cube_readout__LEN: u32 = 128u;
const cube_readout__T: u32 = 128u;
const cube_readout__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_readout_sb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_readout_qb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_readout__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_readout__sc: array<u32>;

var<workgroup> cube_readout__t: array<f32, cube_readout__T>;


@compute @workgroup_size(cube_readout__T)
fn cube_readout_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_readout__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_readout__Q; j += 1u) {
    let i = l * cube_readout__Q + j;
    var v: f32;
    if (i < cube_readout__LEN) {
      v = b_cube_readout_leaf_cube_readout_sb_cube_readout_qb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_readout_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_readout__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_readout__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_readout__T; p += cube_readout__T * 2u * h) {
      cube_readout__t[p] = b_cube_readout_join(cube_readout__t[p], cube_readout__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_readout__out[row] = b_cube_readout_fin(cube_readout__t[0]);
  }
}

// Map history
// ====

@group(0) @binding(0) var<storage, read> map_history_hb: array<u32>;
@group(0) @binding(1) var<storage, read> map_history_xb: array<u32>;
@group(0) @binding(2) var<storage, read_write> map_history__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_history__sc: array<u32>;

@compute @workgroup_size(256)
fn map_history_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_history__sc[0u]) {
    return;
  }
  map_history__out[i] = b_map_history_f_map_history_hb_map_history_xb(i);
}
