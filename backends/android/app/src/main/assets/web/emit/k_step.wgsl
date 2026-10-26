// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"map","name":"softplus","entry":"map_softplus_main","lanes":256,"ins":[{"name":"x","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"exp","entry":"map_exp_main","lanes":256,"ins":[{"name":"x","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"without","entry":"map_without_main","lanes":256,"ins":[{"name":"z","ty":"f32","cls":"row","binding":0},{"name":"ib","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"ib","at":1}]},"len_at":0},{"kind":"map","name":"top2","entry":"map_top2_main","lanes":256,"ins":[{"name":"zb","ty":"bytes","cls":"bytes","binding":0},{"name":"ab","ty":"bytes","cls":"bytes","binding":1},{"name":"bb","ty":"bytes","cls":"bytes","binding":2}],"out":{"ty":"u32","binding":3},"sc":{"binding":4,"scalars":[],"lens":[{"name":"zb","at":1},{"name":"ab","at":2},{"name":"bb","at":3}]},"len_at":0}]}
// emitted by bend (wgsl.ts): the program's lane defs and runtime_wgsl

// X32
// ===

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

// x32_softplus: kernels/ops.bend softplus1, as the C lane emits it (comp.ts):
//   20.0f < x ? x : x32_softplus(x < 88.0f ? x : 88.0f)
// where x32.h's x32_softplus(x) = (f32)log(1.0 + exp((double)x)) for x <= 88.
// Note softplus1(NaN) = 88.0 (NaN < 88 is false). x32w_softplus_core_bits = x32.h's x32_softplus.

fn x32w_softplus_core_bits(u: u32) -> u32 {
  let a = u & 0x7fffffffu;
  let sg = (u >> 31u) != 0u;
  if (a > 0x7f800000u) {
    return 0x7fc00000u;
  }
  if (a == 0x7f800000u) {
    return select(u, 0u, sg);
  }
  if (sg && a >= 0x42200000u) {
    return 0u;
  }
  if (a < 0x32800000u) {
    return 0x3f317218u;
  }
  let m = (a & 0x7fffffu) | 0x800000u;
  let me = x32w_exp_parts(sg, x32w_shl(vec2<u32>(0u, m), (a >> 23u) - 94u), vec2<u32>(0u, 0u));
  let M = me.m;
  var E = me.e;
  var D = x32w_add(x32w_shr(M, 11u), vec2<u32>(0u, (M.y >> 10u) & 1u));
  if ((D.x >> 21u) != 0u) {
    D = x32w_shr(D, 1u);
    E += 1;
  }
  var Sm: vec2<u32>;
  var Se: i32;
  if (E >= 0) {
    Se = E;
    if (E <= 52) {
      Sm = x32w_add(D, x32w_shl(vec2<u32>(0u, 1u), u32(52 - E)));
    } else if (E == 53) {
      Sm = x32w_add(D, vec2<u32>(0u, D.y & 1u));
    } else {
      Sm = D;
    }
  } else {
    let sh = u32(-E);
    var q = vec2<u32>(0u, 0u);
    if (sh <= 53u) {
      let rem = x32w_and(D, x32w_mask(sh));
      let hf = x32w_shl(vec2<u32>(0u, 1u), sh - 1u);
      q = x32w_shr(D, sh);
      if (x32w_lt(hf, rem) || (x32w_eq(rem, hf) && (q.y & 1u) == 1u)) {
        q = x32w_add(q, vec2<u32>(0u, 1u));
      }
    }
    Sm = x32w_add(vec2<u32>(0x100000u, 0u), q);
    Se = 0;
  }
  for (var g = 0; g < 4 && (Sm.x >> 21u) != 0u; g++) {
    Sm = x32w_add(x32w_shr(Sm, 1u), vec2<u32>(0u, Sm.y & (Sm.y >> 1u) & 1u));
    Se += 1;
  }
  let k = x32w_sub(Sm, vec2<u32>(0x100000u, 0u));
  let den = x32w_add(Sm, vec2<u32>(0x100000u, 0u));
  var ph = vec2<u32>(0u, 0u);
  var pl = vec2<u32>(0u, 0u);
  var i = 0;
  if (!x32w_isz(k)) {
    var W = vec2<u32>(0u, 0u);
    var r = k;
    for (var g = 0; g < 200 && (W.x >> 31u) == 0u; g++) {
      r = x32w_shl(r, 1u);
      W = x32w_shl(W, 1u);
      if (!x32w_lt(r, den)) {
        r = x32w_sub(r, den);
        W.y |= 1u;
      }
      i += 1;
    }
    let ts = 2 * i - 128;
    var t = vec2<u32>(0u, 0u);
    if (ts < 64) {
      if (ts > 0) {
        t = x32w_shr(x32w_mulhi(W, W), u32(ts));
      } else {
        t = x32w_shl(x32w_mulhi(W, W), u32(-ts));
      }
    }
    var F = vec2<u32>(0u, 0u);
    for (var n = 24; n >= 0; n--) {
      F = x32w_add(x32w_inv_odd[n], x32w_mulhi(F, t));
    }
    ph = x32w_mulhi(W, F);
    pl = x32w_mullo(W, F);
  }
  if (Se == 0) {
    if (x32w_isz(k)) {
      return 0u;
    }
    let z = x32w_clz(ph);
    var Mo = ph;
    if (z != 0u) {
      Mo = x32w_or(x32w_shl(ph, z), x32w_shr(pl, 64u - z));
    }
    return x32w_round(Mo, 65 - i32(z) - i, 0u);
  }
  // Q120: ln u = (ph:pl) >> (i - 58), plus Se ln2
  var lh = vec2<u32>(0u, 0u);
  var ll = vec2<u32>(0u, 0u);
  if (!x32w_isz(k)) {
    let sh = u32(i - 58);
    lh = x32w_shr(ph, sh);
    ll = x32w_or(x32w_shr(pl, sh), x32w_shl(ph, 64u - sh));
  }
  let un = vec2<u32>(0u, u32(Se));
  let nlo = x32w_mullo(un, vec2<u32>(0xabc9e3b3u, 0x9803f2f7u));
  let nhi = x32w_add(x32w_mullo(un, vec2<u32>(0x00b17217u, 0xf7d1cf79u)), x32w_mulhi(un, vec2<u32>(0xabc9e3b3u, 0x9803f2f7u)));
  let l = x32w_add(ll, nlo);
  let h = x32w_add(x32w_add(lh, nhi), vec2<u32>(0u, select(0u, 1u, x32w_lt(l, ll))));
  var z: u32;
  if (!x32w_isz(h)) {
    z = x32w_clz(h);
  } else {
    z = 64u + x32w_clz(l);
  }
  var Mo: vec2<u32>;
  if (z == 0u) {
    Mo = h;
  } else if (z < 64u) {
    Mo = x32w_or(x32w_shl(h, z), x32w_shr(l, 64u - z));
  } else {
    Mo = x32w_shl(l, z - 64u);
  }
  return x32w_round(Mo, 7 - i32(z), 0u);
}

// softplus1: 20 < x ? x : core(x < 88 ? x : 88), compares done on the bits
fn x32w_softplus_bits(u: u32) -> u32 {
  let a = u & 0x7fffffffu;
  let pos = (u >> 31u) == 0u;
  if (pos && a > 0x41a00000u && a <= 0x7f800000u) {
    return u;                                            // 20 < x (incl. +inf)
  }
  var v = u;
  if (a > 0x7f800000u || (pos && a >= 0x42b00000u)) {
    v = 0x42b00000u;                                     // !(x < 88): NaN (x > 20 already left)
  }
  return x32w_softplus_core_bits(v);
}

fn x32_softplus(x: f32) -> f32 {
  return bitcast<f32>(x32w_softplus_bits(bitcast<u32>(x)));
}

const x32w_inv_odd = array<vec2<u32>, 25>(
  vec2<u32>(0x80000000u, 0x00000000u),
  vec2<u32>(0x2aaaaaaau, 0xaaaaaaaau),
  vec2<u32>(0x19999999u, 0x99999999u),
  vec2<u32>(0x12492492u, 0x49249249u),
  vec2<u32>(0x0e38e38eu, 0x38e38e38u),
  vec2<u32>(0x0ba2e8bau, 0x2e8ba2e8u),
  vec2<u32>(0x09d89d89u, 0xd89d89d8u),
  vec2<u32>(0x08888888u, 0x88888888u),
  vec2<u32>(0x07878787u, 0x87878787u),
  vec2<u32>(0x06bca1afu, 0x286bca1au),
  vec2<u32>(0x06186186u, 0x18618618u),
  vec2<u32>(0x0590b216u, 0x42c8590bu),
  vec2<u32>(0x051eb851u, 0xeb851eb8u),
  vec2<u32>(0x04bda12fu, 0x684bda12u),
  vec2<u32>(0x0469ee58u, 0x469ee584u),
  vec2<u32>(0x04210842u, 0x10842108u),
  vec2<u32>(0x03e0f83eu, 0x0f83e0f8u),
  vec2<u32>(0x03a83a83u, 0xa83a83a8u),
  vec2<u32>(0x03759f22u, 0x983759f2u),
  vec2<u32>(0x03483483u, 0x48348348u),
  vec2<u32>(0x031f3831u, 0xf3831f38u),
  vec2<u32>(0x02fa0be8u, 0x2fa0be82u),
  vec2<u32>(0x02d82d82u, 0xd82d82d8u),
  vec2<u32>(0x02b93105u, 0x72620ae4u),
  vec2<u32>(0x029cbc14u, 0xe5e0a72fu));

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

fn b_map_softplus_f(_x_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_softplus1(_x_0);
}

fn b_map_exp_f(_x_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(_x_0);
}

fn b_map_without_f_map_without_ib(_z_0: f32, _i_0: u32) -> f32 {
  return b_map_without_f_pick0((_i_0 == bw_map_without_ib(0u)), _z_0);
}

fn b_map_top2_f_map_top2_zb_map_top2_ab_map_top2_bb(_i_0: u32) -> u32 {
  let _id_0 = b_map_top2_f_pick0_map_top2_ab_map_top2_bb((_i_0 < 2u));
  return b_map_top2_f_pick1_map_top2_zb(((_i_0 & 1u) == 0u), _id_0);
}

fn b____tree_engine_kernels_ops_softplus1(_x_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_softplus_hi((bitcast<f32>(1101004800u) < _x_0), _x_0);
}

fn b____tree_engine_kernels_ops_exp32(_x_0: f32) -> f32 {
  return x32_exp(_x_0);
}

fn b_map_without_f_pick0(_c_0: bool, _z_0: f32) -> f32 {
  if (!_c_0) {
    return _z_0;
  } else {
    return b____tree_engine_kernels_ops_neg_inf();
  }
}

fn b_map_top2_f_pick0_map_top2_ab_map_top2_bb(_c_0: bool) -> u32 {
  if (!_c_0) {
    return bw_map_top2_bb(0u);
  } else {
    return bw_map_top2_ab(0u);
  }
}

fn b_map_top2_f_pick1_map_top2_zb(_c_0: bool, _id_0: u32) -> u32 {
  if (!_c_0) {
    return bw_map_top2_zb((_id_0 * 4u));
  } else {
    return _id_0;
  }
}

fn b____tree_engine_kernels_ops_softplus_hi(_hi_0: bool, _x_0: f32) -> f32 {
  if (_hi_0) {
    return _x_0;
  } else {
    return x32_softplus(select(bitcast<f32>(1118830592u), _x_0, (_x_0 < bitcast<f32>(1118830592u))));
  }
}

fn b____tree_engine_kernels_ops_neg_inf() -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(4286578688u);
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}

fn bg_map_without_ib(i: u32) -> u32 {
  if (i >= map_without__sc[1u]) {
    return 0u;
  }
  return (map_without_ib[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_without_ib(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_without__sc[1u]) {
    return map_without_ib[i >> 2u];
  }
  return bg_map_without_ib(i) | (bg_map_without_ib(i + 1u) << 8u) | (bg_map_without_ib(i + 2u) << 16u)
    | (bg_map_without_ib(i + 3u) << 24u);
}

fn bg_map_top2_zb(i: u32) -> u32 {
  if (i >= map_top2__sc[1u]) {
    return 0u;
  }
  return (map_top2_zb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_top2_zb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_top2__sc[1u]) {
    return map_top2_zb[i >> 2u];
  }
  return bg_map_top2_zb(i) | (bg_map_top2_zb(i + 1u) << 8u) | (bg_map_top2_zb(i + 2u) << 16u)
    | (bg_map_top2_zb(i + 3u) << 24u);
}

fn bg_map_top2_ab(i: u32) -> u32 {
  if (i >= map_top2__sc[2u]) {
    return 0u;
  }
  return (map_top2_ab[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_top2_ab(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_top2__sc[2u]) {
    return map_top2_ab[i >> 2u];
  }
  return bg_map_top2_ab(i) | (bg_map_top2_ab(i + 1u) << 8u) | (bg_map_top2_ab(i + 2u) << 16u)
    | (bg_map_top2_ab(i + 3u) << 24u);
}

fn bg_map_top2_bb(i: u32) -> u32 {
  if (i >= map_top2__sc[3u]) {
    return 0u;
  }
  return (map_top2_bb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_top2_bb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_top2__sc[3u]) {
    return map_top2_bb[i >> 2u];
  }
  return bg_map_top2_bb(i) | (bg_map_top2_bb(i + 1u) << 8u) | (bg_map_top2_bb(i + 2u) << 16u)
    | (bg_map_top2_bb(i + 3u) << 24u);
}


// Map softplus
// ====

@group(0) @binding(0) var<storage, read> map_softplus_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_softplus__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_softplus__sc: array<u32>;

@compute @workgroup_size(256)
fn map_softplus_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_softplus__sc[0u]) {
    return;
  }
  map_softplus__out[i] = b_map_softplus_f(map_softplus_x[i]);
}

// Map exp
// ====

@group(0) @binding(0) var<storage, read> map_exp_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_exp__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_exp__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp__sc[0u]) {
    return;
  }
  map_exp__out[i] = b_map_exp_f(map_exp_x[i]);
}

// Map without
// ====

@group(0) @binding(0) var<storage, read> map_without_z: array<f32>;
@group(0) @binding(1) var<storage, read> map_without_ib: array<u32>;
@group(0) @binding(2) var<storage, read_write> map_without__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_without__sc: array<u32>;

@compute @workgroup_size(256)
fn map_without_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_without__sc[0u]) {
    return;
  }
  map_without__out[i] = b_map_without_f_map_without_ib(map_without_z[i], i);
}

// Map top2
// ====

@group(0) @binding(0) var<storage, read> map_top2_zb: array<u32>;
@group(0) @binding(1) var<storage, read> map_top2_ab: array<u32>;
@group(0) @binding(2) var<storage, read> map_top2_bb: array<u32>;
@group(0) @binding(3) var<storage, read_write> map_top2__out: array<u32>;
@group(0) @binding(4) var<storage, read> map_top2__sc: array<u32>;

@compute @workgroup_size(256)
fn map_top2_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_top2__sc[0u]) {
    return;
  }
  map_top2__out[i] = b_map_top2_f_map_top2_zb_map_top2_ab_map_top2_bb(i);
}
