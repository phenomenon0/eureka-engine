// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"rms2048","entry":"cube_rms2048_main","n":2048,"len":2048,"lanes":256,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null},{"kind":"cube","name":"rms2560","entry":"cube_rms2560_main","n":8192,"len":2560,"lanes":256,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null},{"kind":"cube","name":"qk256","entry":"cube_qk256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"g","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null},{"kind":"cube","name":"qk128","entry":"cube_qk128_main","n":128,"len":128,"lanes":128,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"g","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null},{"kind":"cube","name":"l2","entry":"cube_l2_main","n":128,"len":128,"lanes":128,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"sc","ty":"f32","at":1}],"lens":[]},"rows_at":0},{"kind":"cube","name":"gated","entry":"cube_gated_main","n":128,"len":128,"lanes":128,"node":"f32","per":"elem","ins":[{"name":"o","ty":"f32","cls":"row","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1},{"name":"z","ty":"f32","cls":"row","binding":2}],"out":{"ty":"f32","binding":3},"sc":null,"rows_at":null},{"kind":"map","name":"silu","entry":"map_silu_main","lanes":256,"ins":[{"name":"x","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"swiglu","entry":"map_swiglu_main","lanes":256,"ins":[{"name":"g","ty":"f32","cls":"row","binding":0},{"name":"u","ty":"f32","cls":"row","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"ogate","entry":"map_ogate_main","lanes":256,"ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"g","ty":"f32","cls":"row","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"sigmoid","entry":"map_sigmoid_main","lanes":256,"ins":[{"name":"x","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"vadd","entry":"map_vadd_main","lanes":256,"ins":[{"name":"a","ty":"f32","cls":"row","binding":0},{"name":"b","ty":"f32","cls":"row","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"map","name":"vmul","entry":"map_vmul_main","lanes":256,"ins":[{"name":"a","ty":"f32","cls":"row","binding":0},{"name":"b","ty":"f32","cls":"row","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0}]}
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

fn b_cube_rms2048_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_rms2048_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_rms2048_fin(_ss_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_ss_0, bitcast<f32>(1157627904u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_rms2048_out(_x_0: f32, _s_0: f32, _w_0: f32) -> f32 {
  return x32_mul(x32_mul(_x_0, _s_0), _w_0);
}

fn b_cube_rms2560_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_rms2560_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_rms2560_fin(_ss_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_ss_0, bitcast<f32>(1159725056u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_rms2560_out(_x_0: f32, _s_0: f32, _w_0: f32) -> f32 {
  return x32_mul(x32_mul(_x_0, _s_0), _w_0);
}

fn b_cube_qk256_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_qk256_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_qk256_fin(_ss_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_ss_0, bitcast<f32>(1132462080u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_qk256_out(_x_0: f32, _s_0: f32, _g_0: f32) -> f32 {
  return x32_mul(x32_mul(_x_0, _s_0), _g_0);
}

fn b_cube_qk128_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_qk128_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_qk128_fin(_ss_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_ss_0, bitcast<f32>(1124073472u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_qk128_out(_x_0: f32, _s_0: f32, _g_0: f32) -> f32 {
  return x32_mul(x32_mul(_x_0, _s_0), _g_0);
}

fn b_cube_l2_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_l2_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_l2_fin(_ss_0: f32, _sc_0: f32) -> f32 {
  return x32_div(x32_div(bitcast<f32>(1065353216u), b_F32_max(x32_sqrt(_ss_0), b____tree_engine_kernels_ops_eps())), _sc_0);
}

fn b_cube_l2_out(_x_0: f32, _s_0: f32) -> f32 {
  return x32_mul(_x_0, _s_0);
}

fn b_cube_gated_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_gated_leaf(_o_0: f32) -> f32 {
  return x32_mul(_o_0, _o_0);
}

fn b_cube_gated_fin(_ss_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_ss_0, bitcast<f32>(1124073472u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_gated_out(_o_0: f32, _s_0: f32, _w_0: f32, _z_0: f32) -> f32 {
  return x32_mul(x32_mul(x32_mul(_o_0, _s_0), _w_0), b____tree_engine_kernels_ops_silu1(_z_0));
}

fn b_map_silu_f(_x_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_silu1(_x_0);
}

fn b_map_swiglu_f(_g_0: f32, _u_0: f32) -> f32 {
  return x32_mul(b____tree_engine_kernels_ops_silu1(_g_0), _u_0);
}

fn b_map_ogate_f(_x_0: f32, _g_0: f32) -> f32 {
  return x32_mul(_x_0, x32_div(bitcast<f32>(1065353216u), x32_add(bitcast<f32>(1065353216u), b____tree_engine_kernels_ops_exp32(bitcast<f32>(bitcast<u32>(_g_0) ^ 0x80000000u)))));
}

fn b_map_sigmoid_f(_x_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_sigmoid1(_x_0);
}

fn b_map_vadd_f(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_map_vmul_f(_a_0: f32, _b_0: f32) -> f32 {
  return x32_mul(_a_0, _b_0);
}

fn b____tree_engine_kernels_ops_eps() -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(897988541u);
}

fn b_F32_max(_a_0: f32, _b_0: f32) -> f32 {
  return select(_a_0, _b_0, (_a_0 < _b_0));
}

fn b____tree_engine_kernels_ops_silu1(_x_0: f32) -> f32 {
  let _v_0 = _x_0;
  return x32_div(_v_0, x32_add(bitcast<f32>(1065353216u), b____tree_engine_kernels_ops_exp32(bitcast<f32>(bitcast<u32>(_v_0) ^ 0x80000000u))));
}

fn b____tree_engine_kernels_ops_exp32(_x_0: f32) -> f32 {
  return x32_exp(_x_0);
}

fn b____tree_engine_kernels_ops_sigmoid1(_x_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_add(bitcast<f32>(1065353216u), b____tree_engine_kernels_ops_exp32(bitcast<f32>(bitcast<u32>(_x_0) ^ 0x80000000u))));
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}



// Cube rms2048
// ====

const cube_rms2048__N: u32 = 2048u;
const cube_rms2048__LEN: u32 = 2048u;
const cube_rms2048__T: u32 = 256u;
const cube_rms2048__Q: u32 = 8u;

@group(0) @binding(0) var<storage, read> cube_rms2048_x: array<f32>;
@group(0) @binding(1) var<storage, read> cube_rms2048_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_rms2048__out: array<f32>;

var<workgroup> cube_rms2048__t: array<f32, cube_rms2048__T>;
var<workgroup> cube_rms2048__s: f32;

@compute @workgroup_size(cube_rms2048__T)
fn cube_rms2048_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 4>;
  for (var j = 0u; j < cube_rms2048__Q; j += 1u) {
    let i = l * cube_rms2048__Q + j;
    var v: f32;
    if (i < cube_rms2048__LEN) {
      v = b_cube_rms2048_leaf(cube_rms2048_x[row * cube_rms2048__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_rms2048_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_rms2048__t[l] = st[3];
  workgroupBarrier();
  for (var h = 1u; h < cube_rms2048__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_rms2048__T; p += cube_rms2048__T * 2u * h) {
      cube_rms2048__t[p] = b_cube_rms2048_join(cube_rms2048__t[p], cube_rms2048__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_rms2048__s = b_cube_rms2048_fin(cube_rms2048__t[0]);
  }
  workgroupBarrier();
  let s = cube_rms2048__s;
  for (var i = l; i < cube_rms2048__LEN; i += cube_rms2048__T) {
    cube_rms2048__out[row * cube_rms2048__LEN + i] = b_cube_rms2048_out(cube_rms2048_x[row * cube_rms2048__LEN + i], s, cube_rms2048_w[i]);
  }
}

// Cube rms2560
// ====

const cube_rms2560__N: u32 = 8192u;
const cube_rms2560__LEN: u32 = 2560u;
const cube_rms2560__T: u32 = 256u;
const cube_rms2560__Q: u32 = 32u;

@group(0) @binding(0) var<storage, read> cube_rms2560_x: array<f32>;
@group(0) @binding(1) var<storage, read> cube_rms2560_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_rms2560__out: array<f32>;

var<workgroup> cube_rms2560__t: array<f32, cube_rms2560__T>;
var<workgroup> cube_rms2560__s: f32;

@compute @workgroup_size(cube_rms2560__T)
fn cube_rms2560_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 6>;
  for (var j = 0u; j < cube_rms2560__Q; j += 1u) {
    let i = l * cube_rms2560__Q + j;
    var v: f32;
    if (i < cube_rms2560__LEN) {
      v = b_cube_rms2560_leaf(cube_rms2560_x[row * cube_rms2560__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_rms2560_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_rms2560__t[l] = st[5];
  workgroupBarrier();
  for (var h = 1u; h < cube_rms2560__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_rms2560__T; p += cube_rms2560__T * 2u * h) {
      cube_rms2560__t[p] = b_cube_rms2560_join(cube_rms2560__t[p], cube_rms2560__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_rms2560__s = b_cube_rms2560_fin(cube_rms2560__t[0]);
  }
  workgroupBarrier();
  let s = cube_rms2560__s;
  for (var i = l; i < cube_rms2560__LEN; i += cube_rms2560__T) {
    cube_rms2560__out[row * cube_rms2560__LEN + i] = b_cube_rms2560_out(cube_rms2560_x[row * cube_rms2560__LEN + i], s, cube_rms2560_w[i]);
  }
}

// Cube qk256
// ====

const cube_qk256__N: u32 = 256u;
const cube_qk256__LEN: u32 = 256u;
const cube_qk256__T: u32 = 256u;
const cube_qk256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_qk256_x: array<f32>;
@group(0) @binding(1) var<storage, read> cube_qk256_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_qk256__out: array<f32>;

var<workgroup> cube_qk256__t: array<f32, cube_qk256__T>;
var<workgroup> cube_qk256__s: f32;

@compute @workgroup_size(cube_qk256__T)
fn cube_qk256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_qk256__Q; j += 1u) {
    let i = l * cube_qk256__Q + j;
    var v: f32;
    if (i < cube_qk256__LEN) {
      v = b_cube_qk256_leaf(cube_qk256_x[row * cube_qk256__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_qk256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_qk256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_qk256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_qk256__T; p += cube_qk256__T * 2u * h) {
      cube_qk256__t[p] = b_cube_qk256_join(cube_qk256__t[p], cube_qk256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_qk256__s = b_cube_qk256_fin(cube_qk256__t[0]);
  }
  workgroupBarrier();
  let s = cube_qk256__s;
  for (var i = l; i < cube_qk256__LEN; i += cube_qk256__T) {
    cube_qk256__out[row * cube_qk256__LEN + i] = b_cube_qk256_out(cube_qk256_x[row * cube_qk256__LEN + i], s, cube_qk256_g[i]);
  }
}

// Cube qk128
// ====

const cube_qk128__N: u32 = 128u;
const cube_qk128__LEN: u32 = 128u;
const cube_qk128__T: u32 = 128u;
const cube_qk128__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_qk128_x: array<f32>;
@group(0) @binding(1) var<storage, read> cube_qk128_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_qk128__out: array<f32>;

var<workgroup> cube_qk128__t: array<f32, cube_qk128__T>;
var<workgroup> cube_qk128__s: f32;

@compute @workgroup_size(cube_qk128__T)
fn cube_qk128_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_qk128__Q; j += 1u) {
    let i = l * cube_qk128__Q + j;
    var v: f32;
    if (i < cube_qk128__LEN) {
      v = b_cube_qk128_leaf(cube_qk128_x[row * cube_qk128__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_qk128_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_qk128__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_qk128__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_qk128__T; p += cube_qk128__T * 2u * h) {
      cube_qk128__t[p] = b_cube_qk128_join(cube_qk128__t[p], cube_qk128__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_qk128__s = b_cube_qk128_fin(cube_qk128__t[0]);
  }
  workgroupBarrier();
  let s = cube_qk128__s;
  for (var i = l; i < cube_qk128__LEN; i += cube_qk128__T) {
    cube_qk128__out[row * cube_qk128__LEN + i] = b_cube_qk128_out(cube_qk128_x[row * cube_qk128__LEN + i], s, cube_qk128_g[i]);
  }
}

// Cube l2
// ====

const cube_l2__N: u32 = 128u;
const cube_l2__LEN: u32 = 128u;
const cube_l2__T: u32 = 128u;
const cube_l2__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_l2_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_l2__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_l2__sc: array<u32>;

var<workgroup> cube_l2__t: array<f32, cube_l2__T>;
var<workgroup> cube_l2__s: f32;

@compute @workgroup_size(cube_l2__T)
fn cube_l2_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_l2__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_l2__Q; j += 1u) {
    let i = l * cube_l2__Q + j;
    var v: f32;
    if (i < cube_l2__LEN) {
      v = b_cube_l2_leaf(cube_l2_x[row * cube_l2__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_l2_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_l2__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_l2__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_l2__T; p += cube_l2__T * 2u * h) {
      cube_l2__t[p] = b_cube_l2_join(cube_l2__t[p], cube_l2__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_l2__s = b_cube_l2_fin(cube_l2__t[0], bitcast<f32>(cube_l2__sc[1u]));
  }
  workgroupBarrier();
  let s = cube_l2__s;
  for (var i = l; i < cube_l2__LEN; i += cube_l2__T) {
    cube_l2__out[row * cube_l2__LEN + i] = b_cube_l2_out(cube_l2_x[row * cube_l2__LEN + i], s);
  }
}

// Cube gated
// ====

const cube_gated__N: u32 = 128u;
const cube_gated__LEN: u32 = 128u;
const cube_gated__T: u32 = 128u;
const cube_gated__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_gated_o: array<f32>;
@group(0) @binding(1) var<storage, read> cube_gated_w: array<f32>;
@group(0) @binding(2) var<storage, read> cube_gated_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> cube_gated__out: array<f32>;

var<workgroup> cube_gated__t: array<f32, cube_gated__T>;
var<workgroup> cube_gated__s: f32;

@compute @workgroup_size(cube_gated__T)
fn cube_gated_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_gated__Q; j += 1u) {
    let i = l * cube_gated__Q + j;
    var v: f32;
    if (i < cube_gated__LEN) {
      v = b_cube_gated_leaf(cube_gated_o[row * cube_gated__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_gated_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_gated__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_gated__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_gated__T; p += cube_gated__T * 2u * h) {
      cube_gated__t[p] = b_cube_gated_join(cube_gated__t[p], cube_gated__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_gated__s = b_cube_gated_fin(cube_gated__t[0]);
  }
  workgroupBarrier();
  let s = cube_gated__s;
  for (var i = l; i < cube_gated__LEN; i += cube_gated__T) {
    cube_gated__out[row * cube_gated__LEN + i] = b_cube_gated_out(cube_gated_o[row * cube_gated__LEN + i], s, cube_gated_w[i], cube_gated_z[row * cube_gated__LEN + i]);
  }
}

// Map silu
// ====

@group(0) @binding(0) var<storage, read> map_silu_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_silu__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_silu__sc: array<u32>;

@compute @workgroup_size(256)
fn map_silu_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_silu__sc[0u]) {
    return;
  }
  map_silu__out[i] = b_map_silu_f(map_silu_x[i]);
}

// Map swiglu
// ====

@group(0) @binding(0) var<storage, read> map_swiglu_g: array<f32>;
@group(0) @binding(1) var<storage, read> map_swiglu_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_swiglu__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_swiglu__sc: array<u32>;

@compute @workgroup_size(256)
fn map_swiglu_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_swiglu__sc[0u]) {
    return;
  }
  map_swiglu__out[i] = b_map_swiglu_f(map_swiglu_g[i], map_swiglu_u[i]);
}

// Map ogate
// ====

@group(0) @binding(0) var<storage, read> map_ogate_x: array<f32>;
@group(0) @binding(1) var<storage, read> map_ogate_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_ogate__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_ogate__sc: array<u32>;

@compute @workgroup_size(256)
fn map_ogate_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_ogate__sc[0u]) {
    return;
  }
  map_ogate__out[i] = b_map_ogate_f(map_ogate_x[i], map_ogate_g[i]);
}

// Map sigmoid
// ====

@group(0) @binding(0) var<storage, read> map_sigmoid_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_sigmoid__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_sigmoid__sc: array<u32>;

@compute @workgroup_size(256)
fn map_sigmoid_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_sigmoid__sc[0u]) {
    return;
  }
  map_sigmoid__out[i] = b_map_sigmoid_f(map_sigmoid_x[i]);
}

// Map vadd
// ====

@group(0) @binding(0) var<storage, read> map_vadd_a: array<f32>;
@group(0) @binding(1) var<storage, read> map_vadd_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_vadd__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_vadd__sc: array<u32>;

@compute @workgroup_size(256)
fn map_vadd_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_vadd__sc[0u]) {
    return;
  }
  map_vadd__out[i] = b_map_vadd_f(map_vadd_a[i], map_vadd_b[i]);
}

// Map vmul
// ====

@group(0) @binding(0) var<storage, read> map_vmul_a: array<f32>;
@group(0) @binding(1) var<storage, read> map_vmul_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_vmul__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_vmul__sc: array<u32>;

@compute @workgroup_size(256)
fn map_vmul_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_vmul__sc[0u]) {
    return;
  }
  map_vmul__out[i] = b_map_vmul_f(map_vmul_a[i], map_vmul_b[i]);
}
