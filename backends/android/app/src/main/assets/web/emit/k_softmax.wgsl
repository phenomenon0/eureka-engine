// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"smx256","entry":"cube_smx256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"cube","name":"ssm256","entry":"cube_ssm256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"sout256","entry":"map_sout256_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1},{"name":"z","ty":"f32","cls":"rowv","binding":2}],"out":{"ty":"f32","binding":3},"sc":{"binding":4,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"cube","name":"smx4096","entry":"cube_smx4096_main","n":4096,"len":4096,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"cube","name":"ssm4096","entry":"cube_ssm4096_main","n":4096,"len":4096,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"sout4096","entry":"map_sout4096_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1},{"name":"z","ty":"f32","cls":"rowv","binding":2}],"out":{"ty":"f32","binding":3},"sc":{"binding":4,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"cube","name":"anan","entry":"cube_anan_main","n":262144,"len":248320,"lanes":256,"node":"u32","per":"row","ins":[{"name":"z","ty":"f32","cls":"row","binding":0}],"out":{"ty":"u32","binding":1},"sc":null,"rows_at":null},{"kind":"cube","name":"amx","entry":"cube_amx_main","n":262144,"len":248320,"lanes":256,"node":"f32","per":"row","ins":[{"name":"z","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"cube","name":"aix","entry":"cube_aix_main","n":262144,"len":248320,"lanes":256,"node":"u32","per":"row","ins":[{"name":"z","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"u32","binding":2},"sc":null,"rows_at":null},{"kind":"map","name":"asel","entry":"map_asel_main","lanes":256,"ins":[{"name":"a","ty":"u32","cls":"row","binding":0},{"name":"c","ty":"u32","cls":"row","binding":1}],"out":{"ty":"u32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0}]}
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

fn b_cube_smx256_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_smx256_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_smx256_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_smx256_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_ssm256_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_ssm256_leaf(_s_0: f32, _i_0: u32, _pos_0: u32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(b_cube_ssm256_leaf_pick0((_i_0 <= _pos_0), _s_0), _m_0));
}

fn b_cube_ssm256_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_sout256_f(_s_0: f32, _col_0: u32, _pos_0: u32, _m_0: f32, _z_0: f32) -> f32 {
  return x32_div(b____tree_engine_kernels_ops_exp32(x32_sub(b_map_sout256_f_pick0((_col_0 <= _pos_0), _s_0), _m_0)), _z_0);
}

fn b_cube_smx4096_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_smx4096_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_smx4096_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_smx4096_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_ssm4096_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_ssm4096_leaf(_s_0: f32, _i_0: u32, _pos_0: u32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(b_cube_ssm4096_leaf_pick0((_i_0 <= _pos_0), _s_0), _m_0));
}

fn b_cube_ssm4096_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_sout4096_f(_s_0: f32, _col_0: u32, _pos_0: u32, _m_0: f32, _z_0: f32) -> f32 {
  return x32_div(b____tree_engine_kernels_ops_exp32(x32_sub(b_map_sout4096_f_pick0((_col_0 <= _pos_0), _s_0), _m_0)), _z_0);
}

fn b_cube_anan_pad() -> u32 {
  return b_big();
}

fn b_cube_anan_join(_a_0: u32, _b_0: u32) -> u32 {
  return b_U32_min(_a_0, _b_0);
}

fn b_cube_anan_leaf(_z_0: f32, _i_0: u32) -> u32 {
  return b_cube_anan_leaf_pick0(b____tree_engine_kernels_ops_is_nan(_z_0), _i_0);
}

fn b_cube_anan_fin(_t_0: u32) -> u32 {
  return _t_0;
}

fn b_cube_amx_pad() -> f32 {
  return b____tree_engine_kernels_ops_neg_inf();
}

fn b_cube_amx_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_amx_leaf(_z_0: f32) -> f32 {
  return _z_0;
}

fn b_cube_amx_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_aix_pad() -> u32 {
  return b_big();
}

fn b_cube_aix_join(_a_0: u32, _b_0: u32) -> u32 {
  return b_U32_min(_a_0, _b_0);
}

fn b_cube_aix_leaf(_z_0: f32, _i_0: u32, _m_0: f32) -> u32 {
  return b_cube_aix_leaf_pick0((_z_0 == _m_0), _i_0);
}

fn b_cube_aix_fin(_t_0: u32) -> u32 {
  return _t_0;
}

fn b_map_asel_f(_a_0: u32, _c_0: u32) -> u32 {
  return select(_c_0, _a_0, (_a_0 < b_big()));
}

fn b_F32_max(_a_0: f32, _b_0: f32) -> f32 {
  return select(_a_0, _b_0, (_a_0 < _b_0));
}

fn b_cube_smx256_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_cube_ssm256_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b____tree_engine_kernels_ops_exp32(_x_0: f32) -> f32 {
  return x32_exp(_x_0);
}

fn b_map_sout256_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_cube_smx4096_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_cube_ssm4096_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_sout4096_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_big() -> u32 {
  return 4294967295u;
}

fn b_U32_min(_a_0: u32, _b_0: u32) -> u32 {
  return select(_b_0, _a_0, (_a_0 < _b_0));
}

fn b____tree_engine_kernels_ops_is_nan(_x_0: f32) -> bool {
  return fne(_x_0, _x_0);
}

fn b_cube_anan_leaf_pick0(_c_0: bool, _i_0: u32) -> u32 {
  if (!_c_0) {
    return b_big();
  } else {
    return _i_0;
  }
}

fn b____tree_engine_kernels_ops_neg_inf() -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(4286578688u);
}

fn b_cube_aix_leaf_pick0(_c_0: bool, _i_0: u32) -> u32 {
  if (!_c_0) {
    return b_big();
  } else {
    return _i_0;
  }
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}



// Cube smx256
// ====

const cube_smx256__N: u32 = 256u;
const cube_smx256__LEN: u32 = 256u;
const cube_smx256__T: u32 = 256u;
const cube_smx256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_smx256_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_smx256__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_smx256__sc: array<u32>;

var<workgroup> cube_smx256__t: array<f32, cube_smx256__T>;


@compute @workgroup_size(cube_smx256__T)
fn cube_smx256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_smx256__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_smx256__Q; j += 1u) {
    let i = l * cube_smx256__Q + j;
    var v: f32;
    if (i < cube_smx256__LEN) {
      v = b_cube_smx256_leaf(cube_smx256_s[row * cube_smx256__LEN + i], i, cube_smx256__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_smx256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_smx256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_smx256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_smx256__T; p += cube_smx256__T * 2u * h) {
      cube_smx256__t[p] = b_cube_smx256_join(cube_smx256__t[p], cube_smx256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_smx256__out[row] = b_cube_smx256_fin(cube_smx256__t[0]);
  }
}

// Cube ssm256
// ====

const cube_ssm256__N: u32 = 256u;
const cube_ssm256__LEN: u32 = 256u;
const cube_ssm256__T: u32 = 256u;
const cube_ssm256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_ssm256_s: array<f32>;
@group(0) @binding(1) var<storage, read> cube_ssm256_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_ssm256__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_ssm256__sc: array<u32>;

var<workgroup> cube_ssm256__t: array<f32, cube_ssm256__T>;


@compute @workgroup_size(cube_ssm256__T)
fn cube_ssm256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_ssm256__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_ssm256__Q; j += 1u) {
    let i = l * cube_ssm256__Q + j;
    var v: f32;
    if (i < cube_ssm256__LEN) {
      v = b_cube_ssm256_leaf(cube_ssm256_s[row * cube_ssm256__LEN + i], i, cube_ssm256__sc[1u], cube_ssm256_m[row]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_ssm256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_ssm256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_ssm256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_ssm256__T; p += cube_ssm256__T * 2u * h) {
      cube_ssm256__t[p] = b_cube_ssm256_join(cube_ssm256__t[p], cube_ssm256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_ssm256__out[row] = b_cube_ssm256_fin(cube_ssm256__t[0]);
  }
}

// Map sout256
// ====

@group(0) @binding(0) var<storage, read> map_sout256_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_sout256_m: array<f32>;
@group(0) @binding(2) var<storage, read> map_sout256_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> map_sout256__out: array<f32>;
@group(0) @binding(4) var<storage, read> map_sout256__sc: array<u32>;

@compute @workgroup_size(256)
fn map_sout256_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_sout256__sc[0u]) {
    return;
  }
  let row = i / 256u;
  let col = i % 256u;
  map_sout256__out[i] = b_map_sout256_f(map_sout256_s[i], col, map_sout256__sc[1u], map_sout256_m[row], map_sout256_z[row]);
}

// Cube smx4096
// ====

const cube_smx4096__N: u32 = 4096u;
const cube_smx4096__LEN: u32 = 4096u;
const cube_smx4096__T: u32 = 256u;
const cube_smx4096__Q: u32 = 16u;

@group(0) @binding(0) var<storage, read> cube_smx4096_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_smx4096__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_smx4096__sc: array<u32>;

var<workgroup> cube_smx4096__t: array<f32, cube_smx4096__T>;


@compute @workgroup_size(cube_smx4096__T)
fn cube_smx4096_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_smx4096__sc[0u]) {
    return;
  }
var st: array<f32, 5>;
  for (var j = 0u; j < cube_smx4096__Q; j += 1u) {
    let i = l * cube_smx4096__Q + j;
    var v: f32;
    if (i < cube_smx4096__LEN) {
      v = b_cube_smx4096_leaf(cube_smx4096_s[row * cube_smx4096__LEN + i], i, cube_smx4096__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_smx4096_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_smx4096__t[l] = st[4];
  workgroupBarrier();
  for (var h = 1u; h < cube_smx4096__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_smx4096__T; p += cube_smx4096__T * 2u * h) {
      cube_smx4096__t[p] = b_cube_smx4096_join(cube_smx4096__t[p], cube_smx4096__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_smx4096__out[row] = b_cube_smx4096_fin(cube_smx4096__t[0]);
  }
}

// Cube ssm4096
// ====

const cube_ssm4096__N: u32 = 4096u;
const cube_ssm4096__LEN: u32 = 4096u;
const cube_ssm4096__T: u32 = 256u;
const cube_ssm4096__Q: u32 = 16u;

@group(0) @binding(0) var<storage, read> cube_ssm4096_s: array<f32>;
@group(0) @binding(1) var<storage, read> cube_ssm4096_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_ssm4096__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_ssm4096__sc: array<u32>;

var<workgroup> cube_ssm4096__t: array<f32, cube_ssm4096__T>;


@compute @workgroup_size(cube_ssm4096__T)
fn cube_ssm4096_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_ssm4096__sc[0u]) {
    return;
  }
var st: array<f32, 5>;
  for (var j = 0u; j < cube_ssm4096__Q; j += 1u) {
    let i = l * cube_ssm4096__Q + j;
    var v: f32;
    if (i < cube_ssm4096__LEN) {
      v = b_cube_ssm4096_leaf(cube_ssm4096_s[row * cube_ssm4096__LEN + i], i, cube_ssm4096__sc[1u], cube_ssm4096_m[row]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_ssm4096_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_ssm4096__t[l] = st[4];
  workgroupBarrier();
  for (var h = 1u; h < cube_ssm4096__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_ssm4096__T; p += cube_ssm4096__T * 2u * h) {
      cube_ssm4096__t[p] = b_cube_ssm4096_join(cube_ssm4096__t[p], cube_ssm4096__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_ssm4096__out[row] = b_cube_ssm4096_fin(cube_ssm4096__t[0]);
  }
}

// Map sout4096
// ====

@group(0) @binding(0) var<storage, read> map_sout4096_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_sout4096_m: array<f32>;
@group(0) @binding(2) var<storage, read> map_sout4096_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> map_sout4096__out: array<f32>;
@group(0) @binding(4) var<storage, read> map_sout4096__sc: array<u32>;

@compute @workgroup_size(256)
fn map_sout4096_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_sout4096__sc[0u]) {
    return;
  }
  let row = i / 4096u;
  let col = i % 4096u;
  map_sout4096__out[i] = b_map_sout4096_f(map_sout4096_s[i], col, map_sout4096__sc[1u], map_sout4096_m[row], map_sout4096_z[row]);
}

// Cube anan
// ====

const cube_anan__N: u32 = 262144u;
const cube_anan__LEN: u32 = 248320u;
const cube_anan__T: u32 = 256u;
const cube_anan__Q: u32 = 1024u;

@group(0) @binding(0) var<storage, read> cube_anan_z: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_anan__out: array<u32>;

var<workgroup> cube_anan__t: array<u32, cube_anan__T>;


@compute @workgroup_size(cube_anan__T)
fn cube_anan_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<u32, 11>;
  for (var j = 0u; j < cube_anan__Q; j += 1u) {
    let i = l * cube_anan__Q + j;
    var v: u32;
    if (i < cube_anan__LEN) {
      v = b_cube_anan_leaf(cube_anan_z[row * cube_anan__LEN + i], i);
    } else {
      v = b_cube_anan_pad();
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_anan_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_anan__t[l] = st[10];
  workgroupBarrier();
  for (var h = 1u; h < cube_anan__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_anan__T; p += cube_anan__T * 2u * h) {
      cube_anan__t[p] = b_cube_anan_join(cube_anan__t[p], cube_anan__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_anan__out[row] = b_cube_anan_fin(cube_anan__t[0]);
  }
}

// Cube amx
// ====

const cube_amx__N: u32 = 262144u;
const cube_amx__LEN: u32 = 248320u;
const cube_amx__T: u32 = 256u;
const cube_amx__Q: u32 = 1024u;

@group(0) @binding(0) var<storage, read> cube_amx_z: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_amx__out: array<f32>;

var<workgroup> cube_amx__t: array<f32, cube_amx__T>;


@compute @workgroup_size(cube_amx__T)
fn cube_amx_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 11>;
  for (var j = 0u; j < cube_amx__Q; j += 1u) {
    let i = l * cube_amx__Q + j;
    var v: f32;
    if (i < cube_amx__LEN) {
      v = b_cube_amx_leaf(cube_amx_z[row * cube_amx__LEN + i]);
    } else {
      v = b_cube_amx_pad();
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_amx_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_amx__t[l] = st[10];
  workgroupBarrier();
  for (var h = 1u; h < cube_amx__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_amx__T; p += cube_amx__T * 2u * h) {
      cube_amx__t[p] = b_cube_amx_join(cube_amx__t[p], cube_amx__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_amx__out[row] = b_cube_amx_fin(cube_amx__t[0]);
  }
}

// Cube aix
// ====

const cube_aix__N: u32 = 262144u;
const cube_aix__LEN: u32 = 248320u;
const cube_aix__T: u32 = 256u;
const cube_aix__Q: u32 = 1024u;

@group(0) @binding(0) var<storage, read> cube_aix_z: array<f32>;
@group(0) @binding(1) var<storage, read> cube_aix_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_aix__out: array<u32>;

var<workgroup> cube_aix__t: array<u32, cube_aix__T>;


@compute @workgroup_size(cube_aix__T)
fn cube_aix_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<u32, 11>;
  for (var j = 0u; j < cube_aix__Q; j += 1u) {
    let i = l * cube_aix__Q + j;
    var v: u32;
    if (i < cube_aix__LEN) {
      v = b_cube_aix_leaf(cube_aix_z[row * cube_aix__LEN + i], i, cube_aix_m[row]);
    } else {
      v = b_cube_aix_pad();
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_aix_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_aix__t[l] = st[10];
  workgroupBarrier();
  for (var h = 1u; h < cube_aix__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_aix__T; p += cube_aix__T * 2u * h) {
      cube_aix__t[p] = b_cube_aix_join(cube_aix__t[p], cube_aix__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_aix__out[row] = b_cube_aix_fin(cube_aix__t[0]);
  }
}

// Map asel
// ====

@group(0) @binding(0) var<storage, read> map_asel_a: array<u32>;
@group(0) @binding(1) var<storage, read> map_asel_c: array<u32>;
@group(0) @binding(2) var<storage, read_write> map_asel__out: array<u32>;
@group(0) @binding(3) var<storage, read> map_asel__sc: array<u32>;

@compute @workgroup_size(256)
fn map_asel_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_asel__sc[0u]) {
    return;
  }
  map_asel__out[i] = b_map_asel_f(map_asel_a[i], map_asel_c[i]);
}
