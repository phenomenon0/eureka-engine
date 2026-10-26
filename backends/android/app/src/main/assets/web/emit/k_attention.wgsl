// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"qnorm","entry":"cube_qnorm_main","n":256,"len":256,"lanes":256,"node":"f32","per":"elem","ins":[{"name":"xb","ty":"bytes","cls":"bytes","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"xb","at":1}]},"rows_at":0},{"kind":"cube","name":"knorm","entry":"cube_knorm_main","n":256,"len":256,"lanes":256,"node":"f32","per":"elem","ins":[{"name":"x","ty":"f32","cls":"row","binding":0},{"name":"w","ty":"f32","cls":"shared","binding":1}],"out":{"ty":"f32","binding":2},"sc":null,"rows_at":null},{"kind":"map","name":"rope","entry":"map_rope_main","lanes":256,"ins":[{"name":"xb","ty":"bytes","cls":"bytes","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"f32","at":2}],"lens":[{"name":"xb","at":1}]},"len_at":0},{"kind":"map","name":"qgrope","entry":"map_qgrope_main","lanes":256,"ins":[{"name":"qb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"pos","ty":"f32","at":3}],"lens":[{"name":"qb","at":1},{"name":"xb","at":2}]},"len_at":0},{"kind":"map","name":"query","entry":"map_query_main","lanes":256,"ins":[{"name":"qgb","ty":"bytes","cls":"bytes","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[{"name":"qgb","at":1}]},"len_at":0},{"kind":"map","name":"gate","entry":"map_gate_main","lanes":256,"ins":[{"name":"qgb","ty":"bytes","cls":"bytes","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[],"lens":[{"name":"qgb","at":1}]},"len_at":0},{"kind":"map","name":"put","entry":"map_put_main","lanes":256,"ins":[{"name":"cb","ty":"bytes","cls":"bytes","binding":0},{"name":"xb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"pos","ty":"u32","at":3}],"lens":[{"name":"cb","at":1},{"name":"xb","at":2}]},"len_at":0},{"kind":"cube","name":"scores","entry":"cube_scores_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"qb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[{"name":"ctx","ty":"u32","at":1}],"lens":[{"name":"qb","at":2},{"name":"cb","at":3}]},"rows_at":0},{"kind":"cube","name":"max16","entry":"cube_max16_main","n":16,"len":16,"lanes":16,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"mask16","entry":"map_mask16_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"map","name":"exp16","entry":"map_exp16_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"sum16","entry":"cube_sum16_main","n":16,"len":16,"lanes":16,"node":"f32","per":"row","ins":[{"name":"e","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"map","name":"prob16","entry":"map_prob16_main","lanes":256,"ins":[{"name":"e","ty":"f32","cls":"row","binding":0},{"name":"z","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"weighted16","entry":"cube_weighted16_main","n":16,"len":16,"lanes":16,"node":"f32","per":"row","ins":[{"name":"pb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"pb","at":1},{"name":"cb","at":2}]},"rows_at":0},{"kind":"cube","name":"max32","entry":"cube_max32_main","n":32,"len":32,"lanes":32,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"mask32","entry":"map_mask32_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"map","name":"exp32","entry":"map_exp32_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"sum32","entry":"cube_sum32_main","n":32,"len":32,"lanes":32,"node":"f32","per":"row","ins":[{"name":"e","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"map","name":"prob32","entry":"map_prob32_main","lanes":256,"ins":[{"name":"e","ty":"f32","cls":"row","binding":0},{"name":"z","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"weighted32","entry":"cube_weighted32_main","n":32,"len":32,"lanes":32,"node":"f32","per":"row","ins":[{"name":"pb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"pb","at":1},{"name":"cb","at":2}]},"rows_at":0},{"kind":"cube","name":"max128","entry":"cube_max128_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"mask128","entry":"map_mask128_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"map","name":"exp128","entry":"map_exp128_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"sum128","entry":"cube_sum128_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"e","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"map","name":"prob128","entry":"map_prob128_main","lanes":256,"ins":[{"name":"e","ty":"f32","cls":"row","binding":0},{"name":"z","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"weighted128","entry":"cube_weighted128_main","n":128,"len":128,"lanes":128,"node":"f32","per":"row","ins":[{"name":"pb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"pb","at":1},{"name":"cb","at":2}]},"rows_at":0},{"kind":"cube","name":"max256","entry":"cube_max256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"mask256","entry":"map_mask256_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"map","name":"exp256","entry":"map_exp256_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"sum256","entry":"cube_sum256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"e","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"map","name":"prob256","entry":"map_prob256_main","lanes":256,"ins":[{"name":"e","ty":"f32","cls":"row","binding":0},{"name":"z","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"weighted256","entry":"cube_weighted256_main","n":256,"len":256,"lanes":256,"node":"f32","per":"row","ins":[{"name":"pb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"pb","at":1},{"name":"cb","at":2}]},"rows_at":0},{"kind":"cube","name":"max4096","entry":"cube_max4096_main","n":4096,"len":4096,"lanes":256,"node":"f32","per":"row","ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"rows_at":0},{"kind":"map","name":"mask4096","entry":"map_mask4096_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":{"binding":2,"scalars":[{"name":"pos","ty":"u32","at":1}],"lens":[]},"len_at":0},{"kind":"map","name":"exp4096","entry":"map_exp4096_main","lanes":256,"ins":[{"name":"s","ty":"f32","cls":"row","binding":0},{"name":"m","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"sum4096","entry":"cube_sum4096_main","n":4096,"len":4096,"lanes":256,"node":"f32","per":"row","ins":[{"name":"e","ty":"f32","cls":"row","binding":0}],"out":{"ty":"f32","binding":1},"sc":null,"rows_at":null},{"kind":"map","name":"prob4096","entry":"map_prob4096_main","lanes":256,"ins":[{"name":"e","ty":"f32","cls":"row","binding":0},{"name":"z","ty":"f32","cls":"rowv","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[]},"len_at":0},{"kind":"cube","name":"weighted4096","entry":"cube_weighted4096_main","n":4096,"len":4096,"lanes":256,"node":"f32","per":"row","ins":[{"name":"pb","ty":"bytes","cls":"bytes","binding":0},{"name":"cb","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"f32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"pb","at":1},{"name":"cb","at":2}]},"rows_at":0}]}
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

// x32_sin / x32_cos: (f32)sin/cos((double)x), x32.h x32_sincos (Payne-Hanek, Q63 polynomials)

fn x32w_sincos_bits(u: u32, want_cos: u32) -> u32 {
  let a = u & 0x7fffffffu;
  if (a >= 0x7f800000u) {
    return 0x7fc00000u;
  }
  if (a < 0x39800000u) {
    return select(u, 0x3f800000u, want_cos == 1u);
  }
  let m = (a & 0x7fffffu) | 0x800000u;
  let s = i32(a >> 23u) - 150;
  var R: vec2<u32>;
  var er: i32;
  var N = 0u;
  var rneg = 0u;
  if (a <= 0x3f490fdau) {
    R = x32w_shl(vec2<u32>(0u, m), 40u);
    er = s + 24;
  } else {
    let p0 = s + 62;
    let w = u32(p0 >> 5u);
    let o = u32(p0 & 31);
    var W: array<u32, 6>;
    for (var i = 0u; i < 6u; i++) {
      if (o != 0u) {
        W[i] = (x32w_2opi[w + i] << o) | (x32w_2opi[w + i + 1u] >> (32u - o));
      } else {
        W[i] = x32w_2opi[w + i];
      }
    }
    var out: array<u32, 6>;
    var acc = vec2<u32>(0u, 0u);
    let m4 = m << 2u;
    for (var i = 5; i >= 0; i--) {
      acc = x32w_add(vec2<u32>(0u, acc.x), x32w_m32(m4, W[i]));
      out[i] = acc.y;
    }
    N = acc.x & 3u;
    var fh = vec2<u32>(out[0], out[1]);
    var fl = vec2<u32>(out[2], out[3]);
    if ((fh.x >> 31u) != 0u) {
      N += 1u;
      rneg = 1u;
      fh = x32w_add(vec2<u32>(~fh.x, ~fh.y), vec2<u32>(0u, select(0u, 1u, x32w_isz(fl))));
      fl = x32w_neg(fl);
    }
    var z: u32;
    if (!x32w_isz(fh)) {
      z = x32w_clz(fh);
    } else {
      z = 64u + x32w_clz(fl);
    }
    var F = vec2<u32>(0u, 0u);
    if (z == 0u) {
      F = fh;
    } else if (z < 64u) {
      F = x32w_or(x32w_shl(fh, z), x32w_shr(fl, 64u - z));
    } else if (z < 128u) {
      F = x32w_shl(fl, z - 64u);
    }
    let Rp = x32w_mulhi(F, vec2<u32>(0xc90fdaa2u, 0x2168c235u));
    let z2 = x32w_clz(Rp);
    R = x32w_shl(Rp, z2);
    er = 1 - i32(z2) - i32(z);
  }
  // r = R/2^64 * 2^er, |r| <= pi/4; zz = r^2 in Q63
  let zs = 1 - 2 * er;
  var zz = vec2<u32>(0u, 0u);
  if (zs < 64) {
    zz = x32w_shr(x32w_mulhi(R, R), u32(max(zs, 0)));
  }
  let nw = N + want_cos;
  let use_cos = (nw & 1u) == 1u;
  let neg = ((nw >> 1u) ^ select(0u, rneg, !use_cos) ^ select(0u, u >> 31u, want_cos == 0u)) & 1u;
  var M: vec2<u32>;
  var E: i32;
  if (use_cos) {
    var c = x32w_cosc[9];
    for (var k = 8; k >= 0; k--) {
      c = x32w_add(x32w_cosc[k], x32w_mul63(c, zz));
    }
    let cq = x32w_sub(vec2<u32>(0x40000000u, 0u), x32w_mulhis(zz, c));
    let l = x32w_clz(cq);
    M = x32w_shl(cq, l);
    E = 1 - i32(l);
  } else {
    var c = x32w_sinc[9];
    for (var k = 8; k >= 0; k--) {
      c = x32w_add(x32w_sinc[k], x32w_mul63(c, zz));
    }
    let sm = x32w_sub(R, x32w_shl(x32w_mulhi(R, x32w_mul63(zz, c)), 1u));
    let l = x32w_clz(sm);
    M = x32w_shl(sm, l);
    E = er - 1 - i32(l);
  }
  return x32w_round(M, E, neg << 31u);
}

fn x32_sin(x: f32) -> f32 {
  return bitcast<f32>(x32w_sincos_bits(bitcast<u32>(x), 0u));
}

fn x32_cos(x: f32) -> f32 {
  return bitcast<f32>(x32w_sincos_bits(bitcast<u32>(x), 1u));
}

const x32w_sinc = array<vec2<u32>, 10>(
  vec2<u32>(0x15555555u, 0x55555555u),
  vec2<u32>(0xfeeeeeeeu, 0xeeeeeeefu),
  vec2<u32>(0x00068068u, 0x06806807u),
  vec2<u32>(0xffffe8e2u, 0x1c5aa939u),
  vec2<u32>(0x00000035u, 0xcc8acfebu),
  vec2<u32>(0xffffffffu, 0xa7b6e7b1u),
  vec2<u32>(0x00000000u, 0x006b9fd0u),
  vec2<u32>(0xffffffffu, 0xffff9ab5u),
  vec2<u32>(0x00000000u, 0x0000004cu),
  vec2<u32>(0x00000000u, 0x00000000u));

const x32w_cosc = array<vec2<u32>, 10>(
  vec2<u32>(0x40000000u, 0x00000000u),
  vec2<u32>(0xfaaaaaaau, 0xaaaaaaabu),
  vec2<u32>(0x002d82d8u, 0x2d82d82eu),
  vec2<u32>(0xffff2ff2u, 0xff2ff2ffu),
  vec2<u32>(0x0000024fu, 0xc9f6ef14u),
  vec2<u32>(0xfffffffbu, 0x8449c402u),
  vec2<u32>(0x00000000u, 0x064e5d2au),
  vec2<u32>(0xffffffffu, 0xfff94603u),
  vec2<u32>(0x00000000u, 0x000005a1u),
  vec2<u32>(0xffffffffu, 0xfffffffcu));

const x32w_2opi = array<u32, 12>(
  0x00000000u,
  0x00000000u,
  0xa2f9836eu,
  0x4e441529u,
  0xfc2757d1u,
  0xf534ddc0u,
  0xdb629599u,
  0x3c439041u,
  0xfe5163abu,
  0xdebbc561u,
  0xb7246e3au,
  0x424dd2e0u);

// x32_pow: (f32)pow((double)x, (double)y), x32.h x32_yint / x32_ln / x32_pow

// 0: not an integer, 1: odd integer, 2: even integer (y finite)
fn x32w_yint(a: u32) -> u32 {
  let e = i32(a >> 23u) - 150;
  if (a == 0u || e >= 1) {
    return 2u;
  }
  if (e < -23) {
    return 0u;
  }
  let m = (a & 0x7fffffu) | 0x800000u;
  let s = u32(-e);
  if ((m & ((1u << s) - 1u)) != 0u) {
    return 0u;
  }
  return select(2u, 1u, ((m >> s) & 1u) == 1u);
}

// ln x for finite x > 0 (bits a), a signed 128-bit Q120: vec4(hi.hi, hi.lo, lo.hi, lo.lo)
fn x32w_ln(a: u32) -> vec4<u32> {
  var s: i32;
  var m: u32;
  if (a >= 0x800000u) {
    m = (a & 0x7fffffu) | 0x800000u;
    s = i32(a >> 23u) - 150;
  } else {
    let z = countLeadingZeros(a) - 8u;
    m = a << z;
    s = -149 - i32(z);
  }
  let j = ((m + 0x10000u) >> 17u) - 64u;
  let cm = (64u + j) << 17u;
  let den = m + cm;
  let wneg = m < cm;
  let num = select(m - cm, cm - m, wneg);
  // W = floor(num 2^71 / den) = x32.h's q1 << 32 | (r1 << 32) / den (q1 < 2^32), |w| Q71
  var r = num;
  var W = vec2<u32>(0u, 0u);
  for (var i = 0; i < 71; i++) {
    r = r << 1u;
    W = vec2<u32>((W.x << 1u) | (W.y >> 31u), W.y << 1u);
    if (r >= den) {
      r -= den;
      W.y |= 1u;
    }
  }
  let w2 = x32w_shr(x32w_mulhi(W, W), 14u);
  let S = x32w_add(vec2<u32>(0x55555555u, 0x55555555u),
    x32w_mulhi(w2, x32w_add(vec2<u32>(0x33333333u, 0x33333333u), x32w_mulhi(w2, vec2<u32>(0x24924924u, 0x92492492u)))));
  let at = x32w_add(W, x32w_mulhi(W, x32w_mulhi(w2, S)));
  var ahi = x32w_shr(at, 14u);
  var alo = x32w_shl(at, 50u);
  if (wneg) {
    ahi = x32w_add(vec2<u32>(~ahi.x, ~ahi.y), vec2<u32>(0u, select(0u, 1u, x32w_isz(alo))));
    alo = x32w_neg(alo);
  }
  let n = s + 23;
  let un = vec2<u32>(0u, u32(abs(n)));
  var nlo = x32w_mullo(un, vec2<u32>(0xabc9e3b3u, 0x9803f2f7u));
  var nhi = x32w_add(x32w_mullo(un, vec2<u32>(0x00b17217u, 0xf7d1cf79u)), x32w_mulhi(un, vec2<u32>(0xabc9e3b3u, 0x9803f2f7u)));
  if (n < 0) {
    nhi = x32w_add(vec2<u32>(~nhi.x, ~nhi.y), vec2<u32>(0u, select(0u, 1u, x32w_isz(nlo))));
    nlo = x32w_neg(nlo);
  }
  let c = x32w_lnc[j];
  let l = x32w_add(nlo, c.zw);
  let h = x32w_add(x32w_add(nhi, c.xy), vec2<u32>(0u, select(0u, 1u, x32w_lt(l, nlo))));
  let l2 = x32w_add(l, alo);
  let hi = x32w_add(x32w_add(h, ahi), vec2<u32>(0u, select(0u, 1u, x32w_lt(l2, l))));
  return vec4<u32>(hi, l2);
}

// est = (float)lh * 0x1p-56f * |y| > 120.0f, in integers (lh != 0 unsigned, y bits ay)
fn x32w_est_big(lh: vec2<u32>, ay: u32) -> bool {
  if (x32w_isz(lh) || ay < 0x800000u) {
    return false;   // est < 2^7 * 2^-126
  }
  if (ay >= 0x7f800000u) {
    return true;
  }
  let f = x32w_u64f(lh);
  let p = x32w_fmul(f.x, bitcast<i32>(f.y) - 56, (ay & 0x7fffffu) | 0x800000u, i32(ay >> 23u) - 150);
  let ep = bitcast<i32>(p.y);
  return ep > -17 || (ep == -17 && p.x > 0xf00000u);
}

fn x32w_pow_bits(ux: u32, uy: u32) -> u32 {
  let ax = ux & 0x7fffffffu;
  let ay = uy & 0x7fffffffu;
  if (ay == 0u || ux == 0x3f800000u) {
    return 0x3f800000u;
  }
  if (ax > 0x7f800000u || ay > 0x7f800000u) {
    return 0x7fc00000u;
  }
  var yi = 2u;
  if (ay != 0x7f800000u) {
    yi = x32w_yint(ay);
  }
  let sgn = select(0u, 0x80000000u, (ux >> 31u) != 0u && yi == 1u);
  let yneg = (uy >> 31u) != 0u;
  if (ay == 0x7f800000u) {
    if (ax == 0x3f800000u) {
      return 0x3f800000u;
    }
    return select(0u, 0x7f800000u, (ax > 0x3f800000u) == !yneg);
  }
  if (ax == 0u) {
    return select(0u, 0x7f800000u, yneg) | sgn;
  }
  if (ax == 0x7f800000u) {
    return select(0x7f800000u, 0u, yneg) | sgn;
  }
  if ((ux >> 31u) != 0u && yi == 0u) {
    return 0x7fc00000u;
  }
  let L = x32w_ln(ax);
  var lh = L.xy;
  var ll = L.zw;
  let lneg = x32w_isneg(lh);
  if (lneg) {
    lh = x32w_add(vec2<u32>(~lh.x, ~lh.y), vec2<u32>(0u, select(0u, 1u, x32w_isz(ll))));
    ll = x32w_neg(ll);
  }
  let tneg = lneg != yneg;
  if (x32w_est_big(lh, ay)) {
    return select(0x7f800000u, 0u, tneg) | sgn;
  }
  // t = my * |L| * 2^ey, Q120
  let ynorm = ay >= 0x800000u;
  let my = vec2<u32>(0u, (ay & 0x7fffffu) | select(0u, 0x800000u, ynorm));
  let ey = select(1, i32(ay >> 23u), ynorm) - 150;
  let p0 = x32w_mullo(my, ll);
  let c0 = x32w_mulhi(my, ll);
  let p1 = x32w_add(x32w_mullo(my, lh), c0);
  let p2 = x32w_add(x32w_mulhi(my, lh), vec2<u32>(0u, select(0u, 1u, x32w_lt(p1, c0))));
  var th: vec2<u32>;
  var tl: vec2<u32>;
  if (ey >= 0) {
    let e = u32(ey);
    th = p1;
    if (e != 0u) {
      th = x32w_or(x32w_shl(p1, e), x32w_shr(p0, 64u - e));
    }
    tl = x32w_shl(p0, e);
  } else {
    let sh = u32(-ey);
    if (sh >= 128u) {
      th = vec2<u32>(0u, 0u);
      tl = x32w_shr(p2, sh - 128u);
      if (sh >= 192u) {
        tl = vec2<u32>(0u, 0u);
      }
    } else if (sh >= 64u) {
      if (sh == 64u) {
        th = p2;
        tl = p1;
      } else {
        th = x32w_shr(p2, sh - 64u);
        tl = x32w_or(x32w_shr(p1, sh - 64u), x32w_shl(p2, 128u - sh));
      }
    } else {
      th = x32w_or(x32w_shr(p1, sh), x32w_shl(p2, 64u - sh));
      tl = x32w_or(x32w_shr(p0, sh), x32w_shl(p1, 64u - sh));
    }
  }
  return x32w_exp_core(tneg, th, tl) ^ sgn;
}

fn x32_pow(x: f32, y: f32) -> f32 {
  return bitcast<f32>(x32w_pow_bits(bitcast<u32>(x), bitcast<u32>(y)));
}

const x32w_lnc = array<vec4<u32>, 65>(
  vec4<u32>(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u),
  vec4<u32>(0x0003f815u, 0x161f807cu, 0x79f3db4eu, 0x9a6f57abu),
  vec4<u32>(0x0007e0a6u, 0xc39e0cc0u, 0x133e3f04u, 0xf1ef22a0u),
  vec4<u32>(0x000bba2cu, 0x7b196e7eu, 0x231a7950u, 0xf7252c16u),
  vec4<u32>(0x000f8518u, 0x6008b153u, 0x30be64b8u, 0xb7759979u),
  vec4<u32>(0x001341d7u, 0x961bd1d0u, 0x92998376u, 0x104d1375u),
  vec4<u32>(0x0016f0d2u, 0x8ae56b4bu, 0x9be499b9u, 0xed19b641u),
  vec4<u32>(0x001a926du, 0x3a4ad563u, 0x650bd22au, 0x9c3aa4c8u),
  vec4<u32>(0x001e2707u, 0x6e2af2e5u, 0xe9ea87ffu, 0xe1fe9e15u),
  vec4<u32>(0x0021aefcu, 0xf9a11cb2u, 0xcd2ee2f4u, 0x81855d1cu),
  vec4<u32>(0x00252aa5u, 0xf03fea46u, 0x980bb8e2u, 0x03edf4d1u),
  vec4<u32>(0x00289a56u, 0xd996fa3cu, 0xcfa7b2a1u, 0xf0fc3c19u),
  vec4<u32>(0x002bfe60u, 0xe14f27a7u, 0x90e7c414u, 0x0e424776u),
  vec4<u32>(0x002f5712u, 0x0421b212u, 0x37c6d65au, 0xd40c100du),
  vec4<u32>(0x0032a4b5u, 0x39e8ad68u, 0xec8260eau, 0x71712cecu),
  vec4<u32>(0x0035e792u, 0x9d017fe5u, 0xb19cc032u, 0x6f99eb97u),
  vec4<u32>(0x00391fefu, 0x8f353443u, 0x584bb03du, 0xe5ff7345u),
  vec4<u32>(0x003c4e0eu, 0xdc55e5cbu, 0xd3d50fffu, 0xc3fd3c2bu),
  vec4<u32>(0x003f7230u, 0xdabc7c55u, 0x1aaa8cd8u, 0x6f29a594u),
  vec4<u32>(0x00428c93u, 0x89ce438du, 0x7dcfde80u, 0x61c030e3u),
  vec4<u32>(0x00459d72u, 0xaeae9838u, 0x0e731f55u, 0xc41b8b82u),
  vec4<u32>(0x0048a507u, 0xef3de596u, 0x890a14f6u, 0x9d750cbdu),
  vec4<u32>(0x004ba38au, 0xeb8474c2u, 0x70b3246au, 0x14206cf3u),
  vec4<u32>(0x004e9931u, 0x55a517a7u, 0x1cbcd735u, 0xd034237du),
  vec4<u32>(0x0051862fu, 0x08717b09u, 0xf42decdeu, 0xccf1cd10u),
  vec4<u32>(0x00546ab6u, 0x1cb7e0b4u, 0x2724f583u, 0x3eabc624u),
  vec4<u32>(0x005746f6u, 0xfd602729u, 0x4236383du, 0xc7fe115au),
  vec4<u32>(0x005a1b20u, 0x7a6c52bbu, 0x110af840u, 0x538e1a59u),
  vec4<u32>(0x005ce75fu, 0xdaef401au, 0x7389314fu, 0xeb4fbde6u),
  vec4<u32>(0x005fabe0u, 0xee0abf0du, 0x92ce979eu, 0xd2950437u),
  vec4<u32>(0x006268ceu, 0x1b05096au, 0xd69c6204u, 0x40f055b4u),
  vec4<u32>(0x00651e50u, 0x70845beau, 0xe9337451u, 0xf441babbu),
  vec4<u32>(0x0067cc8fu, 0xb2fe612fu, 0xcada35d9u, 0xbd014886u),
  vec4<u32>(0x006a73b2u, 0x6a682126u, 0x35213fd4u, 0xbc950d7cu),
  vec4<u32>(0x006d13ddu, 0xef323d8au, 0x32fbb6abu, 0xa63878efu),
  vec4<u32>(0x006fad36u, 0x769c6defu, 0xde1874deu, 0xaef06b26u),
  vec4<u32>(0x00723fdfu, 0x1e6a6886u, 0xb097607bu, 0xcbfee689u),
  vec4<u32>(0x0074cbf9u, 0xf803af55u, 0x877b232fu, 0xafa36fd2u),
  vec4<u32>(0x007751a8u, 0x13071282u, 0xfb989a92u, 0x7476e1ffu),
  vec4<u32>(0x0079d109u, 0x875a1e1fu, 0x8df68dbcu, 0xf2ed1bb4u),
  vec4<u32>(0x007c4a3du, 0x7ebc1bb2u, 0xcd720ec4u, 0x4c73d75du),
  vec4<u32>(0x007ebd62u, 0x3de3cc7bu, 0x66becf93u, 0xaa1afec7u),
  vec4<u32>(0x00812a95u, 0x2d2e87f6u, 0x34e34aebu, 0xf73ffe34u),
  vec4<u32>(0x008391f2u, 0xe0e6fa02u, 0x72bcb1c4u, 0x88b755b3u),
  vec4<u32>(0x0085f397u, 0x21295415u, 0xb4c4bdd9u, 0x9effe69bu),
  vec4<u32>(0x00884f9cu, 0xf16a64b7u, 0xef1f64d8u, 0x5bc8c5f2u),
  vec4<u32>(0x008aa61eu, 0x97a6af4du, 0x4c799d1cu, 0xb2f14055u),
  vec4<u32>(0x008cf735u, 0xa33e4b76u, 0x62e5eebbu, 0xc0ef3d57u),
  vec4<u32>(0x008f42fau, 0xf3820681u, 0xef62cd2fu, 0x9f1e35f3u),
  vec4<u32>(0x00918986u, 0xbdf5fa14u, 0x16f1b439u, 0x165240a4u),
  vec4<u32>(0x0093caf0u, 0x944d88d7u, 0x5bc1f9edu, 0xcb438ffcu),
  vec4<u32>(0x0096074fu, 0x6a24745du, 0xcbd4e18du, 0xd14f312au),
  vec4<u32>(0x00983eb9u, 0x9a7885f0u, 0xfdac850fu, 0xab36cdeeu),
  vec4<u32>(0x009a7144u, 0xece70e98u, 0xb75c96c4u, 0x2e727572u),
  vec4<u32>(0x009c9f06u, 0x9ab150cdu, 0x4e221301u, 0xb6f8c38fu),
  vec4<u32>(0x009ec813u, 0x538ab7d5u, 0x202131e8u, 0x5693cf6cu),
  vec4<u32>(0x00a0ec7fu, 0x42339573u, 0x2325e617u, 0xa300bbcbu),
  vec4<u32>(0x00a30c5eu, 0x10e2f613u, 0xe85bd9bdu, 0x99e39a21u),
  vec4<u32>(0x00a527c2u, 0xed81f5d8u, 0x113dfa3du, 0x3761b631u),
  vec4<u32>(0x00a73ec0u, 0x8dbadd84u, 0xe584c2b2u, 0x2c2aee1au),
  vec4<u32>(0x00a95169u, 0x32de2d57u, 0x73be4578u, 0xad97aea8u),
  vec4<u32>(0x00ab5fceu, 0xad9f9ccau, 0x08e310b9u, 0xb1fe59ceu),
  vec4<u32>(0x00ad6a02u, 0x61acf967u, 0xd94d552fu, 0x811cd408u),
  vec4<u32>(0x00af7015u, 0x4920b3abu, 0x86b04afeu, 0x92103ef5u),
  vec4<u32>(0x00b17217u, 0xf7d1cf79u, 0xabc9e3b3u, 0x9803f2f7u));

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

fn b_cube_qnorm_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_qnorm_leaf_cube_qnorm_xb(_row_0: u32, _i_0: u32) -> f32 {
  let _x_0 = b_word_cube_qnorm_xb(((_row_0 * 512u) + _i_0));
  return x32_mul(_x_0, _x_0);
}

fn b_cube_qnorm_fin(_t_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_t_0, bitcast<f32>(1132462080u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_qnorm_out_cube_qnorm_xb(_s_0: f32, _row_0: u32, _i_0: u32, _w_0: f32) -> f32 {
  return x32_mul(x32_mul(b_word_cube_qnorm_xb(((_row_0 * 512u) + _i_0)), _s_0), _w_0);
}

fn b_cube_knorm_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_knorm_leaf(_x_0: f32) -> f32 {
  return x32_mul(_x_0, _x_0);
}

fn b_cube_knorm_fin(_t_0: f32) -> f32 {
  return x32_div(bitcast<f32>(1065353216u), x32_sqrt(x32_add(x32_div(_t_0, bitcast<f32>(1132462080u)), b____tree_engine_kernels_ops_eps())));
}

fn b_cube_knorm_out(_x_0: f32, _s_0: f32, _w_0: f32) -> f32 {
  return x32_mul(x32_mul(_x_0, _s_0), _w_0);
}

fn b_map_rope_f_map_rope_xb(_i_0: u32, _pos_0: f32) -> f32 {
  return b_rot_map_rope_xb(select(_i_0 >> 8u, 0u, 8u >= 32u), (_i_0 & 255u), _pos_0);
}

fn b_map_qgrope_f_map_qgrope_qb_map_qgrope_xb(_i_0: u32, _pos_0: f32) -> f32 {
  return b_map_qgrope_f_pick0_map_qgrope_qb_map_qgrope_xb(((_i_0 & 511u) < 256u), _i_0, _pos_0);
}

fn b_map_query_f_map_query_qgb(_i_0: u32) -> f32 {
  return b_word_map_query_qgb(((select(_i_0 >> 8u, 0u, 8u >= 32u) * 512u) + (_i_0 & 255u)));
}

fn b_map_gate_f_map_gate_qgb(_i_0: u32) -> f32 {
  return b_word_map_gate_qgb((((select(_i_0 >> 8u, 0u, 8u >= 32u) * 512u) + 256u) + (_i_0 & 255u)));
}

fn b_map_put_f_map_put_cb_map_put_xb(_i_0: u32, _pos_0: u32) -> f32 {
  return b_map_put_f_pick0_map_put_cb_map_put_xb((select(_i_0 >> 10u, 0u, 10u >= 32u) == _pos_0), _i_0);
}

fn b_cube_scores_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_scores_leaf_cube_scores_qb_cube_scores_cb(_row_0: u32, _i_0: u32, _ctx_0: u32) -> f32 {
  let _head_0 = select(_row_0 / _ctx_0, 0u, _ctx_0 == 0u);
  let _time_0 = select(_row_0 % _ctx_0, _row_0, _ctx_0 == 0u);
  return x32_mul(b_word_cube_scores_qb(((_head_0 * 256u) + _i_0)), b_word_cube_scores_cb((((_time_0 * 1024u) + (select(_head_0 >> 2u, 0u, 2u >= 32u) * 256u)) + _i_0)));
}

fn b_cube_scores_fin(_t_0: f32) -> f32 {
  return x32_mul(_t_0, bitcast<f32>(1031798784u));
}

fn b_cube_max16_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_max16_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_max16_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_max16_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_mask16_f(_s_0: f32, _col_0: u32, _pos_0: u32) -> f32 {
  return b_map_mask16_f_pick0((_col_0 <= _pos_0), _s_0);
}

fn b_map_exp16_f(_s_0: f32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(_s_0, _m_0));
}

fn b_cube_sum16_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_sum16_leaf(_e_0: f32) -> f32 {
  return _e_0;
}

fn b_cube_sum16_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_prob16_f(_e_0: f32, _z_0: f32) -> f32 {
  return x32_div(_e_0, _z_0);
}

fn b_cube_weighted16_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_weighted16_leaf_cube_weighted16_pb_cube_weighted16_cb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_weighted16_cb((((_i_0 * 1024u) + (select(_row_0 >> 10u, 0u, 10u >= 32u) * 256u)) + (_row_0 & 255u))), b_word_cube_weighted16_pb(((select(_row_0 >> 8u, 0u, 8u >= 32u) * 16u) + _i_0)));
}

fn b_cube_weighted16_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_max32_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_max32_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_max32_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_max32_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_mask32_f(_s_0: f32, _col_0: u32, _pos_0: u32) -> f32 {
  return b_map_mask32_f_pick0((_col_0 <= _pos_0), _s_0);
}

fn b_map_exp32_f(_s_0: f32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(_s_0, _m_0));
}

fn b_cube_sum32_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_sum32_leaf(_e_0: f32) -> f32 {
  return _e_0;
}

fn b_cube_sum32_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_prob32_f(_e_0: f32, _z_0: f32) -> f32 {
  return x32_div(_e_0, _z_0);
}

fn b_cube_weighted32_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_weighted32_leaf_cube_weighted32_pb_cube_weighted32_cb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_weighted32_cb((((_i_0 * 1024u) + (select(_row_0 >> 10u, 0u, 10u >= 32u) * 256u)) + (_row_0 & 255u))), b_word_cube_weighted32_pb(((select(_row_0 >> 8u, 0u, 8u >= 32u) * 32u) + _i_0)));
}

fn b_cube_weighted32_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_max128_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_max128_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_max128_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_max128_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_mask128_f(_s_0: f32, _col_0: u32, _pos_0: u32) -> f32 {
  return b_map_mask128_f_pick0((_col_0 <= _pos_0), _s_0);
}

fn b_map_exp128_f(_s_0: f32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(_s_0, _m_0));
}

fn b_cube_sum128_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_sum128_leaf(_e_0: f32) -> f32 {
  return _e_0;
}

fn b_cube_sum128_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_prob128_f(_e_0: f32, _z_0: f32) -> f32 {
  return x32_div(_e_0, _z_0);
}

fn b_cube_weighted128_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_weighted128_leaf_cube_weighted128_pb_cube_weighted128_cb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_weighted128_cb((((_i_0 * 1024u) + (select(_row_0 >> 10u, 0u, 10u >= 32u) * 256u)) + (_row_0 & 255u))), b_word_cube_weighted128_pb(((select(_row_0 >> 8u, 0u, 8u >= 32u) * 128u) + _i_0)));
}

fn b_cube_weighted128_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_max256_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_max256_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_max256_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_max256_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_mask256_f(_s_0: f32, _col_0: u32, _pos_0: u32) -> f32 {
  return b_map_mask256_f_pick0((_col_0 <= _pos_0), _s_0);
}

fn b_map_exp256_f(_s_0: f32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(_s_0, _m_0));
}

fn b_cube_sum256_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_sum256_leaf(_e_0: f32) -> f32 {
  return _e_0;
}

fn b_cube_sum256_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_prob256_f(_e_0: f32, _z_0: f32) -> f32 {
  return x32_div(_e_0, _z_0);
}

fn b_cube_weighted256_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_weighted256_leaf_cube_weighted256_pb_cube_weighted256_cb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_weighted256_cb((((_i_0 * 1024u) + (select(_row_0 >> 10u, 0u, 10u >= 32u) * 256u)) + (_row_0 & 255u))), b_word_cube_weighted256_pb(((select(_row_0 >> 8u, 0u, 8u >= 32u) * 256u) + _i_0)));
}

fn b_cube_weighted256_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_cube_max4096_join(_a_0: f32, _b_0: f32) -> f32 {
  return b_F32_max(_a_0, _b_0);
}

fn b_cube_max4096_leaf(_s_0: f32, _i_0: u32, _pos_0: u32) -> f32 {
  return b_cube_max4096_leaf_pick0((_i_0 <= _pos_0), _s_0);
}

fn b_cube_max4096_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_mask4096_f(_s_0: f32, _col_0: u32, _pos_0: u32) -> f32 {
  return b_map_mask4096_f_pick0((_col_0 <= _pos_0), _s_0);
}

fn b_map_exp4096_f(_s_0: f32, _m_0: f32) -> f32 {
  return b____tree_engine_kernels_ops_exp32(x32_sub(_s_0, _m_0));
}

fn b_cube_sum4096_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_sum4096_leaf(_e_0: f32) -> f32 {
  return _e_0;
}

fn b_cube_sum4096_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_map_prob4096_f(_e_0: f32, _z_0: f32) -> f32 {
  return x32_div(_e_0, _z_0);
}

fn b_cube_weighted4096_join(_a_0: f32, _b_0: f32) -> f32 {
  return x32_add(_a_0, _b_0);
}

fn b_cube_weighted4096_leaf_cube_weighted4096_pb_cube_weighted4096_cb(_row_0: u32, _i_0: u32) -> f32 {
  return x32_mul(b_word_cube_weighted4096_cb((((_i_0 * 1024u) + (select(_row_0 >> 10u, 0u, 10u >= 32u) * 256u)) + (_row_0 & 255u))), b_word_cube_weighted4096_pb(((select(_row_0 >> 8u, 0u, 8u >= 32u) * 4096u) + _i_0)));
}

fn b_cube_weighted4096_fin(_t_0: f32) -> f32 {
  return _t_0;
}

fn b_word_cube_qnorm_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_qnorm_xb((4u * _i_0)));
}

fn b____tree_engine_kernels_ops_eps() -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(897988541u);
}

fn b_rot_map_rope_xb(_head_0: u32, _col_0: u32, _pos_0: f32) -> f32 {
  return b_rot_go_map_rope_xb((_col_0 < 64u), _head_0, _col_0, _pos_0);
}

fn b_map_qgrope_f_pick0_map_qgrope_qb_map_qgrope_xb(_c_0: bool, _i_0: u32, _pos_0: f32) -> f32 {
  if (!_c_0) {
    return b_word_map_qgrope_xb(_i_0);
  } else {
    return b_rot_map_qgrope_qb(select(_i_0 >> 9u, 0u, 9u >= 32u), (_i_0 & 511u), _pos_0);
  }
}

fn b_word_map_query_qgb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_query_qgb((4u * _i_0)));
}

fn b_word_map_gate_qgb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_gate_qgb((4u * _i_0)));
}

fn b_map_put_f_pick0_map_put_cb_map_put_xb(_c_0: bool, _i_0: u32) -> f32 {
  if (!_c_0) {
    return b_word_map_put_cb(_i_0);
  } else {
    return b_word_map_put_xb((_i_0 & 1023u));
  }
}

fn b_word_cube_scores_qb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_scores_qb((4u * _i_0)));
}

fn b_word_cube_scores_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_scores_cb((4u * _i_0)));
}

fn b_F32_max(_a_0: f32, _b_0: f32) -> f32 {
  return select(_a_0, _b_0, (_a_0 < _b_0));
}

fn b_cube_max16_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_mask16_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b____tree_engine_kernels_ops_exp32(_x_0: f32) -> f32 {
  return x32_exp(_x_0);
}

fn b_word_cube_weighted16_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted16_cb((4u * _i_0)));
}

fn b_word_cube_weighted16_pb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted16_pb((4u * _i_0)));
}

fn b_cube_max32_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_mask32_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_word_cube_weighted32_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted32_cb((4u * _i_0)));
}

fn b_word_cube_weighted32_pb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted32_pb((4u * _i_0)));
}

fn b_cube_max128_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_mask128_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_word_cube_weighted128_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted128_cb((4u * _i_0)));
}

fn b_word_cube_weighted128_pb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted128_pb((4u * _i_0)));
}

fn b_cube_max256_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_mask256_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_word_cube_weighted256_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted256_cb((4u * _i_0)));
}

fn b_word_cube_weighted256_pb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted256_pb((4u * _i_0)));
}

fn b_cube_max4096_leaf_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_map_mask4096_f_pick0(_c_0: bool, _s_0: f32) -> f32 {
  if (!_c_0) {
    return b____tree_engine_kernels_ops_neg_inf();
  } else {
    return _s_0;
  }
}

fn b_word_cube_weighted4096_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted4096_cb((4u * _i_0)));
}

fn b_word_cube_weighted4096_pb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_cube_weighted4096_pb((4u * _i_0)));
}

fn b____tree_engine_kernels_ops_f32_from_bits(_w_0: u32) -> f32 {
  return bitcast<f32>(_w_0);
}

fn b_rot_go_map_rope_xb(_active_0: bool, _head_0: u32, _col_0: u32, _pos_0: f32) -> f32 {
  if (_active_0) {
    let _j_0 = select(_col_0 >> 1u, 0u, 1u >= 32u);
    let _at_0 = (_head_0 * 256u);
    let _a_0 = b_word_map_rope_xb((_at_0 + _j_0));
    let _b_0 = b_word_map_rope_xb(((_at_0 + _j_0) + 32u));
    let _theta_0 = x32_mul(_pos_0, b____tree_engine_kernels_ops_pow32(bitcast<f32>(1259902592u), x32_div(bitcast<f32>(bitcast<u32>(f32((2u * _j_0))) ^ 0x80000000u), bitcast<f32>(1115684864u))));
    let _c_0 = b____tree_engine_kernels_ops_cos32(_theta_0);
    let _s_0 = b____tree_engine_kernels_ops_sin32(_theta_0);
    return select(x32_add(x32_mul(_a_0, _s_0), x32_mul(_b_0, _c_0)), x32_sub(x32_mul(_a_0, _c_0), x32_mul(_b_0, _s_0)), ((_col_0 & 1u) == 0u));
  } else {
    return b_word_map_rope_xb(((_head_0 * 256u) + _col_0));
  }
}

fn b_word_map_qgrope_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_qgrope_xb((4u * _i_0)));
}

fn b_rot_map_qgrope_qb(_head_0: u32, _col_0: u32, _pos_0: f32) -> f32 {
  return b_rot_go_map_qgrope_qb((_col_0 < 64u), _head_0, _col_0, _pos_0);
}

fn b_word_map_put_cb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_put_cb((4u * _i_0)));
}

fn b_word_map_put_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_put_xb((4u * _i_0)));
}

fn b____tree_engine_kernels_ops_neg_inf() -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(4286578688u);
}

fn b_word_map_rope_xb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_rope_xb((4u * _i_0)));
}

fn b____tree_engine_kernels_ops_pow32(_x_0: f32, _y_0: f32) -> f32 {
  return x32_pow(_x_0, _y_0);
}

fn b____tree_engine_kernels_ops_cos32(_x_0: f32) -> f32 {
  return x32_cos(_x_0);
}

fn b____tree_engine_kernels_ops_sin32(_x_0: f32) -> f32 {
  return x32_sin(_x_0);
}

fn b_rot_go_map_qgrope_qb(_active_0: bool, _head_0: u32, _col_0: u32, _pos_0: f32) -> f32 {
  if (_active_0) {
    let _j_0 = select(_col_0 >> 1u, 0u, 1u >= 32u);
    let _at_0 = (_head_0 * 256u);
    let _a_0 = b_word_map_qgrope_qb((_at_0 + _j_0));
    let _b_0 = b_word_map_qgrope_qb(((_at_0 + _j_0) + 32u));
    let _theta_0 = x32_mul(_pos_0, b____tree_engine_kernels_ops_pow32(bitcast<f32>(1259902592u), x32_div(bitcast<f32>(bitcast<u32>(f32((2u * _j_0))) ^ 0x80000000u), bitcast<f32>(1115684864u))));
    let _c_0 = b____tree_engine_kernels_ops_cos32(_theta_0);
    let _s_0 = b____tree_engine_kernels_ops_sin32(_theta_0);
    return select(x32_add(x32_mul(_a_0, _s_0), x32_mul(_b_0, _c_0)), x32_sub(x32_mul(_a_0, _c_0), x32_mul(_b_0, _s_0)), ((_col_0 & 1u) == 0u));
  } else {
    return b_word_map_qgrope_qb(((_head_0 * 256u) + _col_0));
  }
}

fn b_word_map_qgrope_qb(_i_0: u32) -> f32 {
  return b____tree_engine_kernels_ops_f32_from_bits(bw_map_qgrope_qb((4u * _i_0)));
}

fn bg_cube_qnorm_xb(i: u32) -> u32 {
  if (i >= cube_qnorm__sc[1u]) {
    return 0u;
  }
  return (cube_qnorm_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_qnorm_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_qnorm__sc[1u]) {
    return cube_qnorm_xb[i >> 2u];
  }
  return bg_cube_qnorm_xb(i) | (bg_cube_qnorm_xb(i + 1u) << 8u) | (bg_cube_qnorm_xb(i + 2u) << 16u)
    | (bg_cube_qnorm_xb(i + 3u) << 24u);
}

fn bg_map_rope_xb(i: u32) -> u32 {
  if (i >= map_rope__sc[1u]) {
    return 0u;
  }
  return (map_rope_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_rope_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_rope__sc[1u]) {
    return map_rope_xb[i >> 2u];
  }
  return bg_map_rope_xb(i) | (bg_map_rope_xb(i + 1u) << 8u) | (bg_map_rope_xb(i + 2u) << 16u)
    | (bg_map_rope_xb(i + 3u) << 24u);
}

fn bg_map_qgrope_qb(i: u32) -> u32 {
  if (i >= map_qgrope__sc[1u]) {
    return 0u;
  }
  return (map_qgrope_qb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_qgrope_qb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_qgrope__sc[1u]) {
    return map_qgrope_qb[i >> 2u];
  }
  return bg_map_qgrope_qb(i) | (bg_map_qgrope_qb(i + 1u) << 8u) | (bg_map_qgrope_qb(i + 2u) << 16u)
    | (bg_map_qgrope_qb(i + 3u) << 24u);
}

fn bg_map_qgrope_xb(i: u32) -> u32 {
  if (i >= map_qgrope__sc[2u]) {
    return 0u;
  }
  return (map_qgrope_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_qgrope_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_qgrope__sc[2u]) {
    return map_qgrope_xb[i >> 2u];
  }
  return bg_map_qgrope_xb(i) | (bg_map_qgrope_xb(i + 1u) << 8u) | (bg_map_qgrope_xb(i + 2u) << 16u)
    | (bg_map_qgrope_xb(i + 3u) << 24u);
}

fn bg_map_query_qgb(i: u32) -> u32 {
  if (i >= map_query__sc[1u]) {
    return 0u;
  }
  return (map_query_qgb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_query_qgb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_query__sc[1u]) {
    return map_query_qgb[i >> 2u];
  }
  return bg_map_query_qgb(i) | (bg_map_query_qgb(i + 1u) << 8u) | (bg_map_query_qgb(i + 2u) << 16u)
    | (bg_map_query_qgb(i + 3u) << 24u);
}

fn bg_map_gate_qgb(i: u32) -> u32 {
  if (i >= map_gate__sc[1u]) {
    return 0u;
  }
  return (map_gate_qgb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_gate_qgb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_gate__sc[1u]) {
    return map_gate_qgb[i >> 2u];
  }
  return bg_map_gate_qgb(i) | (bg_map_gate_qgb(i + 1u) << 8u) | (bg_map_gate_qgb(i + 2u) << 16u)
    | (bg_map_gate_qgb(i + 3u) << 24u);
}

fn bg_map_put_cb(i: u32) -> u32 {
  if (i >= map_put__sc[1u]) {
    return 0u;
  }
  return (map_put_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_put_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_put__sc[1u]) {
    return map_put_cb[i >> 2u];
  }
  return bg_map_put_cb(i) | (bg_map_put_cb(i + 1u) << 8u) | (bg_map_put_cb(i + 2u) << 16u)
    | (bg_map_put_cb(i + 3u) << 24u);
}

fn bg_map_put_xb(i: u32) -> u32 {
  if (i >= map_put__sc[2u]) {
    return 0u;
  }
  return (map_put_xb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_map_put_xb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= map_put__sc[2u]) {
    return map_put_xb[i >> 2u];
  }
  return bg_map_put_xb(i) | (bg_map_put_xb(i + 1u) << 8u) | (bg_map_put_xb(i + 2u) << 16u)
    | (bg_map_put_xb(i + 3u) << 24u);
}

fn bg_cube_scores_qb(i: u32) -> u32 {
  if (i >= cube_scores__sc[2u]) {
    return 0u;
  }
  return (cube_scores_qb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_scores_qb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_scores__sc[2u]) {
    return cube_scores_qb[i >> 2u];
  }
  return bg_cube_scores_qb(i) | (bg_cube_scores_qb(i + 1u) << 8u) | (bg_cube_scores_qb(i + 2u) << 16u)
    | (bg_cube_scores_qb(i + 3u) << 24u);
}

fn bg_cube_scores_cb(i: u32) -> u32 {
  if (i >= cube_scores__sc[3u]) {
    return 0u;
  }
  return (cube_scores_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_scores_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_scores__sc[3u]) {
    return cube_scores_cb[i >> 2u];
  }
  return bg_cube_scores_cb(i) | (bg_cube_scores_cb(i + 1u) << 8u) | (bg_cube_scores_cb(i + 2u) << 16u)
    | (bg_cube_scores_cb(i + 3u) << 24u);
}

fn bg_cube_weighted16_pb(i: u32) -> u32 {
  if (i >= cube_weighted16__sc[1u]) {
    return 0u;
  }
  return (cube_weighted16_pb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted16_pb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted16__sc[1u]) {
    return cube_weighted16_pb[i >> 2u];
  }
  return bg_cube_weighted16_pb(i) | (bg_cube_weighted16_pb(i + 1u) << 8u) | (bg_cube_weighted16_pb(i + 2u) << 16u)
    | (bg_cube_weighted16_pb(i + 3u) << 24u);
}

fn bg_cube_weighted16_cb(i: u32) -> u32 {
  if (i >= cube_weighted16__sc[2u]) {
    return 0u;
  }
  return (cube_weighted16_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted16_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted16__sc[2u]) {
    return cube_weighted16_cb[i >> 2u];
  }
  return bg_cube_weighted16_cb(i) | (bg_cube_weighted16_cb(i + 1u) << 8u) | (bg_cube_weighted16_cb(i + 2u) << 16u)
    | (bg_cube_weighted16_cb(i + 3u) << 24u);
}

fn bg_cube_weighted32_pb(i: u32) -> u32 {
  if (i >= cube_weighted32__sc[1u]) {
    return 0u;
  }
  return (cube_weighted32_pb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted32_pb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted32__sc[1u]) {
    return cube_weighted32_pb[i >> 2u];
  }
  return bg_cube_weighted32_pb(i) | (bg_cube_weighted32_pb(i + 1u) << 8u) | (bg_cube_weighted32_pb(i + 2u) << 16u)
    | (bg_cube_weighted32_pb(i + 3u) << 24u);
}

fn bg_cube_weighted32_cb(i: u32) -> u32 {
  if (i >= cube_weighted32__sc[2u]) {
    return 0u;
  }
  return (cube_weighted32_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted32_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted32__sc[2u]) {
    return cube_weighted32_cb[i >> 2u];
  }
  return bg_cube_weighted32_cb(i) | (bg_cube_weighted32_cb(i + 1u) << 8u) | (bg_cube_weighted32_cb(i + 2u) << 16u)
    | (bg_cube_weighted32_cb(i + 3u) << 24u);
}

fn bg_cube_weighted128_pb(i: u32) -> u32 {
  if (i >= cube_weighted128__sc[1u]) {
    return 0u;
  }
  return (cube_weighted128_pb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted128_pb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted128__sc[1u]) {
    return cube_weighted128_pb[i >> 2u];
  }
  return bg_cube_weighted128_pb(i) | (bg_cube_weighted128_pb(i + 1u) << 8u) | (bg_cube_weighted128_pb(i + 2u) << 16u)
    | (bg_cube_weighted128_pb(i + 3u) << 24u);
}

fn bg_cube_weighted128_cb(i: u32) -> u32 {
  if (i >= cube_weighted128__sc[2u]) {
    return 0u;
  }
  return (cube_weighted128_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted128_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted128__sc[2u]) {
    return cube_weighted128_cb[i >> 2u];
  }
  return bg_cube_weighted128_cb(i) | (bg_cube_weighted128_cb(i + 1u) << 8u) | (bg_cube_weighted128_cb(i + 2u) << 16u)
    | (bg_cube_weighted128_cb(i + 3u) << 24u);
}

fn bg_cube_weighted256_pb(i: u32) -> u32 {
  if (i >= cube_weighted256__sc[1u]) {
    return 0u;
  }
  return (cube_weighted256_pb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted256_pb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted256__sc[1u]) {
    return cube_weighted256_pb[i >> 2u];
  }
  return bg_cube_weighted256_pb(i) | (bg_cube_weighted256_pb(i + 1u) << 8u) | (bg_cube_weighted256_pb(i + 2u) << 16u)
    | (bg_cube_weighted256_pb(i + 3u) << 24u);
}

fn bg_cube_weighted256_cb(i: u32) -> u32 {
  if (i >= cube_weighted256__sc[2u]) {
    return 0u;
  }
  return (cube_weighted256_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted256_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted256__sc[2u]) {
    return cube_weighted256_cb[i >> 2u];
  }
  return bg_cube_weighted256_cb(i) | (bg_cube_weighted256_cb(i + 1u) << 8u) | (bg_cube_weighted256_cb(i + 2u) << 16u)
    | (bg_cube_weighted256_cb(i + 3u) << 24u);
}

fn bg_cube_weighted4096_pb(i: u32) -> u32 {
  if (i >= cube_weighted4096__sc[1u]) {
    return 0u;
  }
  return (cube_weighted4096_pb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted4096_pb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted4096__sc[1u]) {
    return cube_weighted4096_pb[i >> 2u];
  }
  return bg_cube_weighted4096_pb(i) | (bg_cube_weighted4096_pb(i + 1u) << 8u) | (bg_cube_weighted4096_pb(i + 2u) << 16u)
    | (bg_cube_weighted4096_pb(i + 3u) << 24u);
}

fn bg_cube_weighted4096_cb(i: u32) -> u32 {
  if (i >= cube_weighted4096__sc[2u]) {
    return 0u;
  }
  return (cube_weighted4096_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_weighted4096_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_weighted4096__sc[2u]) {
    return cube_weighted4096_cb[i >> 2u];
  }
  return bg_cube_weighted4096_cb(i) | (bg_cube_weighted4096_cb(i + 1u) << 8u) | (bg_cube_weighted4096_cb(i + 2u) << 16u)
    | (bg_cube_weighted4096_cb(i + 3u) << 24u);
}


// Cube qnorm
// ====

const cube_qnorm__N: u32 = 256u;
const cube_qnorm__LEN: u32 = 256u;
const cube_qnorm__T: u32 = 256u;
const cube_qnorm__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_qnorm_xb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_qnorm_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_qnorm__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_qnorm__sc: array<u32>;

var<workgroup> cube_qnorm__t: array<f32, cube_qnorm__T>;
var<workgroup> cube_qnorm__s: f32;

@compute @workgroup_size(cube_qnorm__T)
fn cube_qnorm_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_qnorm__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_qnorm__Q; j += 1u) {
    let i = l * cube_qnorm__Q + j;
    var v: f32;
    if (i < cube_qnorm__LEN) {
      v = b_cube_qnorm_leaf_cube_qnorm_xb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_qnorm_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_qnorm__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_qnorm__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_qnorm__T; p += cube_qnorm__T * 2u * h) {
      cube_qnorm__t[p] = b_cube_qnorm_join(cube_qnorm__t[p], cube_qnorm__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_qnorm__s = b_cube_qnorm_fin(cube_qnorm__t[0]);
  }
  workgroupBarrier();
  let s = cube_qnorm__s;
  for (var i = l; i < cube_qnorm__LEN; i += cube_qnorm__T) {
    cube_qnorm__out[row * cube_qnorm__LEN + i] = b_cube_qnorm_out_cube_qnorm_xb(s, row, i, cube_qnorm_w[i]);
  }
}

// Cube knorm
// ====

const cube_knorm__N: u32 = 256u;
const cube_knorm__LEN: u32 = 256u;
const cube_knorm__T: u32 = 256u;
const cube_knorm__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_knorm_x: array<f32>;
@group(0) @binding(1) var<storage, read> cube_knorm_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cube_knorm__out: array<f32>;

var<workgroup> cube_knorm__t: array<f32, cube_knorm__T>;
var<workgroup> cube_knorm__s: f32;

@compute @workgroup_size(cube_knorm__T)
fn cube_knorm_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_knorm__Q; j += 1u) {
    let i = l * cube_knorm__Q + j;
    var v: f32;
    if (i < cube_knorm__LEN) {
      v = b_cube_knorm_leaf(cube_knorm_x[row * cube_knorm__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_knorm_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_knorm__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_knorm__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_knorm__T; p += cube_knorm__T * 2u * h) {
      cube_knorm__t[p] = b_cube_knorm_join(cube_knorm__t[p], cube_knorm__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_knorm__s = b_cube_knorm_fin(cube_knorm__t[0]);
  }
  workgroupBarrier();
  let s = cube_knorm__s;
  for (var i = l; i < cube_knorm__LEN; i += cube_knorm__T) {
    cube_knorm__out[row * cube_knorm__LEN + i] = b_cube_knorm_out(cube_knorm_x[row * cube_knorm__LEN + i], s, cube_knorm_w[i]);
  }
}

// Map rope
// ====

@group(0) @binding(0) var<storage, read> map_rope_xb: array<u32>;
@group(0) @binding(1) var<storage, read_write> map_rope__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_rope__sc: array<u32>;

@compute @workgroup_size(256)
fn map_rope_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_rope__sc[0u]) {
    return;
  }
  map_rope__out[i] = b_map_rope_f_map_rope_xb(i, bitcast<f32>(map_rope__sc[2u]));
}

// Map qgrope
// ====

@group(0) @binding(0) var<storage, read> map_qgrope_qb: array<u32>;
@group(0) @binding(1) var<storage, read> map_qgrope_xb: array<u32>;
@group(0) @binding(2) var<storage, read_write> map_qgrope__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_qgrope__sc: array<u32>;

@compute @workgroup_size(256)
fn map_qgrope_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_qgrope__sc[0u]) {
    return;
  }
  map_qgrope__out[i] = b_map_qgrope_f_map_qgrope_qb_map_qgrope_xb(i, bitcast<f32>(map_qgrope__sc[3u]));
}

// Map query
// ====

@group(0) @binding(0) var<storage, read> map_query_qgb: array<u32>;
@group(0) @binding(1) var<storage, read_write> map_query__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_query__sc: array<u32>;

@compute @workgroup_size(256)
fn map_query_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_query__sc[0u]) {
    return;
  }
  map_query__out[i] = b_map_query_f_map_query_qgb(i);
}

// Map gate
// ====

@group(0) @binding(0) var<storage, read> map_gate_qgb: array<u32>;
@group(0) @binding(1) var<storage, read_write> map_gate__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_gate__sc: array<u32>;

@compute @workgroup_size(256)
fn map_gate_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_gate__sc[0u]) {
    return;
  }
  map_gate__out[i] = b_map_gate_f_map_gate_qgb(i);
}

// Map put
// ====

@group(0) @binding(0) var<storage, read> map_put_cb: array<u32>;
@group(0) @binding(1) var<storage, read> map_put_xb: array<u32>;
@group(0) @binding(2) var<storage, read_write> map_put__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_put__sc: array<u32>;

@compute @workgroup_size(256)
fn map_put_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_put__sc[0u]) {
    return;
  }
  map_put__out[i] = b_map_put_f_map_put_cb_map_put_xb(i, map_put__sc[3u]);
}

// Cube scores
// ====

const cube_scores__N: u32 = 256u;
const cube_scores__LEN: u32 = 256u;
const cube_scores__T: u32 = 256u;
const cube_scores__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_scores_qb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_scores_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_scores__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_scores__sc: array<u32>;

var<workgroup> cube_scores__t: array<f32, cube_scores__T>;


@compute @workgroup_size(cube_scores__T)
fn cube_scores_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_scores__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_scores__Q; j += 1u) {
    let i = l * cube_scores__Q + j;
    var v: f32;
    if (i < cube_scores__LEN) {
      v = b_cube_scores_leaf_cube_scores_qb_cube_scores_cb(row, i, cube_scores__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_scores_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_scores__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_scores__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_scores__T; p += cube_scores__T * 2u * h) {
      cube_scores__t[p] = b_cube_scores_join(cube_scores__t[p], cube_scores__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_scores__out[row] = b_cube_scores_fin(cube_scores__t[0]);
  }
}

// Cube max16
// ====

const cube_max16__N: u32 = 16u;
const cube_max16__LEN: u32 = 16u;
const cube_max16__T: u32 = 16u;
const cube_max16__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_max16_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_max16__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_max16__sc: array<u32>;

var<workgroup> cube_max16__t: array<f32, cube_max16__T>;


@compute @workgroup_size(cube_max16__T)
fn cube_max16_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_max16__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_max16__Q; j += 1u) {
    let i = l * cube_max16__Q + j;
    var v: f32;
    if (i < cube_max16__LEN) {
      v = b_cube_max16_leaf(cube_max16_s[row * cube_max16__LEN + i], i, cube_max16__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_max16_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_max16__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_max16__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_max16__T; p += cube_max16__T * 2u * h) {
      cube_max16__t[p] = b_cube_max16_join(cube_max16__t[p], cube_max16__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_max16__out[row] = b_cube_max16_fin(cube_max16__t[0]);
  }
}

// Map mask16
// ====

@group(0) @binding(0) var<storage, read> map_mask16_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_mask16__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_mask16__sc: array<u32>;

@compute @workgroup_size(256)
fn map_mask16_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_mask16__sc[0u]) {
    return;
  }
  let row = i / 16u;
  let col = i % 16u;
  map_mask16__out[i] = b_map_mask16_f(map_mask16_s[i], col, map_mask16__sc[1u]);
}

// Map exp16
// ====

@group(0) @binding(0) var<storage, read> map_exp16_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_exp16_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_exp16__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_exp16__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp16_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp16__sc[0u]) {
    return;
  }
  let row = i / 16u;
  let col = i % 16u;
  map_exp16__out[i] = b_map_exp16_f(map_exp16_s[i], map_exp16_m[row]);
}

// Cube sum16
// ====

const cube_sum16__N: u32 = 16u;
const cube_sum16__LEN: u32 = 16u;
const cube_sum16__T: u32 = 16u;
const cube_sum16__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_sum16_e: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_sum16__out: array<f32>;

var<workgroup> cube_sum16__t: array<f32, cube_sum16__T>;


@compute @workgroup_size(cube_sum16__T)
fn cube_sum16_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_sum16__Q; j += 1u) {
    let i = l * cube_sum16__Q + j;
    var v: f32;
    if (i < cube_sum16__LEN) {
      v = b_cube_sum16_leaf(cube_sum16_e[row * cube_sum16__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_sum16_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_sum16__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_sum16__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_sum16__T; p += cube_sum16__T * 2u * h) {
      cube_sum16__t[p] = b_cube_sum16_join(cube_sum16__t[p], cube_sum16__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_sum16__out[row] = b_cube_sum16_fin(cube_sum16__t[0]);
  }
}

// Map prob16
// ====

@group(0) @binding(0) var<storage, read> map_prob16_e: array<f32>;
@group(0) @binding(1) var<storage, read> map_prob16_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_prob16__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_prob16__sc: array<u32>;

@compute @workgroup_size(256)
fn map_prob16_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_prob16__sc[0u]) {
    return;
  }
  let row = i / 16u;
  let col = i % 16u;
  map_prob16__out[i] = b_map_prob16_f(map_prob16_e[i], map_prob16_z[row]);
}

// Cube weighted16
// ====

const cube_weighted16__N: u32 = 16u;
const cube_weighted16__LEN: u32 = 16u;
const cube_weighted16__T: u32 = 16u;
const cube_weighted16__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_weighted16_pb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_weighted16_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_weighted16__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_weighted16__sc: array<u32>;

var<workgroup> cube_weighted16__t: array<f32, cube_weighted16__T>;


@compute @workgroup_size(cube_weighted16__T)
fn cube_weighted16_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_weighted16__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_weighted16__Q; j += 1u) {
    let i = l * cube_weighted16__Q + j;
    var v: f32;
    if (i < cube_weighted16__LEN) {
      v = b_cube_weighted16_leaf_cube_weighted16_pb_cube_weighted16_cb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_weighted16_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_weighted16__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_weighted16__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_weighted16__T; p += cube_weighted16__T * 2u * h) {
      cube_weighted16__t[p] = b_cube_weighted16_join(cube_weighted16__t[p], cube_weighted16__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_weighted16__out[row] = b_cube_weighted16_fin(cube_weighted16__t[0]);
  }
}

// Cube max32
// ====

const cube_max32__N: u32 = 32u;
const cube_max32__LEN: u32 = 32u;
const cube_max32__T: u32 = 32u;
const cube_max32__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_max32_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_max32__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_max32__sc: array<u32>;

var<workgroup> cube_max32__t: array<f32, cube_max32__T>;


@compute @workgroup_size(cube_max32__T)
fn cube_max32_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_max32__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_max32__Q; j += 1u) {
    let i = l * cube_max32__Q + j;
    var v: f32;
    if (i < cube_max32__LEN) {
      v = b_cube_max32_leaf(cube_max32_s[row * cube_max32__LEN + i], i, cube_max32__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_max32_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_max32__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_max32__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_max32__T; p += cube_max32__T * 2u * h) {
      cube_max32__t[p] = b_cube_max32_join(cube_max32__t[p], cube_max32__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_max32__out[row] = b_cube_max32_fin(cube_max32__t[0]);
  }
}

// Map mask32
// ====

@group(0) @binding(0) var<storage, read> map_mask32_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_mask32__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_mask32__sc: array<u32>;

@compute @workgroup_size(256)
fn map_mask32_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_mask32__sc[0u]) {
    return;
  }
  let row = i / 32u;
  let col = i % 32u;
  map_mask32__out[i] = b_map_mask32_f(map_mask32_s[i], col, map_mask32__sc[1u]);
}

// Map exp32
// ====

@group(0) @binding(0) var<storage, read> map_exp32_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_exp32_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_exp32__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_exp32__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp32_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp32__sc[0u]) {
    return;
  }
  let row = i / 32u;
  let col = i % 32u;
  map_exp32__out[i] = b_map_exp32_f(map_exp32_s[i], map_exp32_m[row]);
}

// Cube sum32
// ====

const cube_sum32__N: u32 = 32u;
const cube_sum32__LEN: u32 = 32u;
const cube_sum32__T: u32 = 32u;
const cube_sum32__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_sum32_e: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_sum32__out: array<f32>;

var<workgroup> cube_sum32__t: array<f32, cube_sum32__T>;


@compute @workgroup_size(cube_sum32__T)
fn cube_sum32_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_sum32__Q; j += 1u) {
    let i = l * cube_sum32__Q + j;
    var v: f32;
    if (i < cube_sum32__LEN) {
      v = b_cube_sum32_leaf(cube_sum32_e[row * cube_sum32__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_sum32_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_sum32__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_sum32__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_sum32__T; p += cube_sum32__T * 2u * h) {
      cube_sum32__t[p] = b_cube_sum32_join(cube_sum32__t[p], cube_sum32__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_sum32__out[row] = b_cube_sum32_fin(cube_sum32__t[0]);
  }
}

// Map prob32
// ====

@group(0) @binding(0) var<storage, read> map_prob32_e: array<f32>;
@group(0) @binding(1) var<storage, read> map_prob32_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_prob32__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_prob32__sc: array<u32>;

@compute @workgroup_size(256)
fn map_prob32_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_prob32__sc[0u]) {
    return;
  }
  let row = i / 32u;
  let col = i % 32u;
  map_prob32__out[i] = b_map_prob32_f(map_prob32_e[i], map_prob32_z[row]);
}

// Cube weighted32
// ====

const cube_weighted32__N: u32 = 32u;
const cube_weighted32__LEN: u32 = 32u;
const cube_weighted32__T: u32 = 32u;
const cube_weighted32__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_weighted32_pb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_weighted32_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_weighted32__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_weighted32__sc: array<u32>;

var<workgroup> cube_weighted32__t: array<f32, cube_weighted32__T>;


@compute @workgroup_size(cube_weighted32__T)
fn cube_weighted32_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_weighted32__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_weighted32__Q; j += 1u) {
    let i = l * cube_weighted32__Q + j;
    var v: f32;
    if (i < cube_weighted32__LEN) {
      v = b_cube_weighted32_leaf_cube_weighted32_pb_cube_weighted32_cb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_weighted32_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_weighted32__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_weighted32__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_weighted32__T; p += cube_weighted32__T * 2u * h) {
      cube_weighted32__t[p] = b_cube_weighted32_join(cube_weighted32__t[p], cube_weighted32__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_weighted32__out[row] = b_cube_weighted32_fin(cube_weighted32__t[0]);
  }
}

// Cube max128
// ====

const cube_max128__N: u32 = 128u;
const cube_max128__LEN: u32 = 128u;
const cube_max128__T: u32 = 128u;
const cube_max128__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_max128_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_max128__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_max128__sc: array<u32>;

var<workgroup> cube_max128__t: array<f32, cube_max128__T>;


@compute @workgroup_size(cube_max128__T)
fn cube_max128_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_max128__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_max128__Q; j += 1u) {
    let i = l * cube_max128__Q + j;
    var v: f32;
    if (i < cube_max128__LEN) {
      v = b_cube_max128_leaf(cube_max128_s[row * cube_max128__LEN + i], i, cube_max128__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_max128_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_max128__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_max128__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_max128__T; p += cube_max128__T * 2u * h) {
      cube_max128__t[p] = b_cube_max128_join(cube_max128__t[p], cube_max128__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_max128__out[row] = b_cube_max128_fin(cube_max128__t[0]);
  }
}

// Map mask128
// ====

@group(0) @binding(0) var<storage, read> map_mask128_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_mask128__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_mask128__sc: array<u32>;

@compute @workgroup_size(256)
fn map_mask128_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_mask128__sc[0u]) {
    return;
  }
  let row = i / 128u;
  let col = i % 128u;
  map_mask128__out[i] = b_map_mask128_f(map_mask128_s[i], col, map_mask128__sc[1u]);
}

// Map exp128
// ====

@group(0) @binding(0) var<storage, read> map_exp128_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_exp128_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_exp128__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_exp128__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp128_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp128__sc[0u]) {
    return;
  }
  let row = i / 128u;
  let col = i % 128u;
  map_exp128__out[i] = b_map_exp128_f(map_exp128_s[i], map_exp128_m[row]);
}

// Cube sum128
// ====

const cube_sum128__N: u32 = 128u;
const cube_sum128__LEN: u32 = 128u;
const cube_sum128__T: u32 = 128u;
const cube_sum128__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_sum128_e: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_sum128__out: array<f32>;

var<workgroup> cube_sum128__t: array<f32, cube_sum128__T>;


@compute @workgroup_size(cube_sum128__T)
fn cube_sum128_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_sum128__Q; j += 1u) {
    let i = l * cube_sum128__Q + j;
    var v: f32;
    if (i < cube_sum128__LEN) {
      v = b_cube_sum128_leaf(cube_sum128_e[row * cube_sum128__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_sum128_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_sum128__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_sum128__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_sum128__T; p += cube_sum128__T * 2u * h) {
      cube_sum128__t[p] = b_cube_sum128_join(cube_sum128__t[p], cube_sum128__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_sum128__out[row] = b_cube_sum128_fin(cube_sum128__t[0]);
  }
}

// Map prob128
// ====

@group(0) @binding(0) var<storage, read> map_prob128_e: array<f32>;
@group(0) @binding(1) var<storage, read> map_prob128_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_prob128__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_prob128__sc: array<u32>;

@compute @workgroup_size(256)
fn map_prob128_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_prob128__sc[0u]) {
    return;
  }
  let row = i / 128u;
  let col = i % 128u;
  map_prob128__out[i] = b_map_prob128_f(map_prob128_e[i], map_prob128_z[row]);
}

// Cube weighted128
// ====

const cube_weighted128__N: u32 = 128u;
const cube_weighted128__LEN: u32 = 128u;
const cube_weighted128__T: u32 = 128u;
const cube_weighted128__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_weighted128_pb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_weighted128_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_weighted128__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_weighted128__sc: array<u32>;

var<workgroup> cube_weighted128__t: array<f32, cube_weighted128__T>;


@compute @workgroup_size(cube_weighted128__T)
fn cube_weighted128_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_weighted128__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_weighted128__Q; j += 1u) {
    let i = l * cube_weighted128__Q + j;
    var v: f32;
    if (i < cube_weighted128__LEN) {
      v = b_cube_weighted128_leaf_cube_weighted128_pb_cube_weighted128_cb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_weighted128_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_weighted128__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_weighted128__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_weighted128__T; p += cube_weighted128__T * 2u * h) {
      cube_weighted128__t[p] = b_cube_weighted128_join(cube_weighted128__t[p], cube_weighted128__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_weighted128__out[row] = b_cube_weighted128_fin(cube_weighted128__t[0]);
  }
}

// Cube max256
// ====

const cube_max256__N: u32 = 256u;
const cube_max256__LEN: u32 = 256u;
const cube_max256__T: u32 = 256u;
const cube_max256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_max256_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_max256__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_max256__sc: array<u32>;

var<workgroup> cube_max256__t: array<f32, cube_max256__T>;


@compute @workgroup_size(cube_max256__T)
fn cube_max256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_max256__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_max256__Q; j += 1u) {
    let i = l * cube_max256__Q + j;
    var v: f32;
    if (i < cube_max256__LEN) {
      v = b_cube_max256_leaf(cube_max256_s[row * cube_max256__LEN + i], i, cube_max256__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_max256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_max256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_max256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_max256__T; p += cube_max256__T * 2u * h) {
      cube_max256__t[p] = b_cube_max256_join(cube_max256__t[p], cube_max256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_max256__out[row] = b_cube_max256_fin(cube_max256__t[0]);
  }
}

// Map mask256
// ====

@group(0) @binding(0) var<storage, read> map_mask256_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_mask256__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_mask256__sc: array<u32>;

@compute @workgroup_size(256)
fn map_mask256_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_mask256__sc[0u]) {
    return;
  }
  let row = i / 256u;
  let col = i % 256u;
  map_mask256__out[i] = b_map_mask256_f(map_mask256_s[i], col, map_mask256__sc[1u]);
}

// Map exp256
// ====

@group(0) @binding(0) var<storage, read> map_exp256_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_exp256_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_exp256__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_exp256__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp256_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp256__sc[0u]) {
    return;
  }
  let row = i / 256u;
  let col = i % 256u;
  map_exp256__out[i] = b_map_exp256_f(map_exp256_s[i], map_exp256_m[row]);
}

// Cube sum256
// ====

const cube_sum256__N: u32 = 256u;
const cube_sum256__LEN: u32 = 256u;
const cube_sum256__T: u32 = 256u;
const cube_sum256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_sum256_e: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_sum256__out: array<f32>;

var<workgroup> cube_sum256__t: array<f32, cube_sum256__T>;


@compute @workgroup_size(cube_sum256__T)
fn cube_sum256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 1>;
  for (var j = 0u; j < cube_sum256__Q; j += 1u) {
    let i = l * cube_sum256__Q + j;
    var v: f32;
    if (i < cube_sum256__LEN) {
      v = b_cube_sum256_leaf(cube_sum256_e[row * cube_sum256__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_sum256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_sum256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_sum256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_sum256__T; p += cube_sum256__T * 2u * h) {
      cube_sum256__t[p] = b_cube_sum256_join(cube_sum256__t[p], cube_sum256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_sum256__out[row] = b_cube_sum256_fin(cube_sum256__t[0]);
  }
}

// Map prob256
// ====

@group(0) @binding(0) var<storage, read> map_prob256_e: array<f32>;
@group(0) @binding(1) var<storage, read> map_prob256_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_prob256__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_prob256__sc: array<u32>;

@compute @workgroup_size(256)
fn map_prob256_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_prob256__sc[0u]) {
    return;
  }
  let row = i / 256u;
  let col = i % 256u;
  map_prob256__out[i] = b_map_prob256_f(map_prob256_e[i], map_prob256_z[row]);
}

// Cube weighted256
// ====

const cube_weighted256__N: u32 = 256u;
const cube_weighted256__LEN: u32 = 256u;
const cube_weighted256__T: u32 = 256u;
const cube_weighted256__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_weighted256_pb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_weighted256_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_weighted256__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_weighted256__sc: array<u32>;

var<workgroup> cube_weighted256__t: array<f32, cube_weighted256__T>;


@compute @workgroup_size(cube_weighted256__T)
fn cube_weighted256_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_weighted256__sc[0u]) {
    return;
  }
var st: array<f32, 1>;
  for (var j = 0u; j < cube_weighted256__Q; j += 1u) {
    let i = l * cube_weighted256__Q + j;
    var v: f32;
    if (i < cube_weighted256__LEN) {
      v = b_cube_weighted256_leaf_cube_weighted256_pb_cube_weighted256_cb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_weighted256_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_weighted256__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_weighted256__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_weighted256__T; p += cube_weighted256__T * 2u * h) {
      cube_weighted256__t[p] = b_cube_weighted256_join(cube_weighted256__t[p], cube_weighted256__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_weighted256__out[row] = b_cube_weighted256_fin(cube_weighted256__t[0]);
  }
}

// Cube max4096
// ====

const cube_max4096__N: u32 = 4096u;
const cube_max4096__LEN: u32 = 4096u;
const cube_max4096__T: u32 = 256u;
const cube_max4096__Q: u32 = 16u;

@group(0) @binding(0) var<storage, read> cube_max4096_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_max4096__out: array<f32>;
@group(0) @binding(2) var<storage, read> cube_max4096__sc: array<u32>;

var<workgroup> cube_max4096__t: array<f32, cube_max4096__T>;


@compute @workgroup_size(cube_max4096__T)
fn cube_max4096_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_max4096__sc[0u]) {
    return;
  }
var st: array<f32, 5>;
  for (var j = 0u; j < cube_max4096__Q; j += 1u) {
    let i = l * cube_max4096__Q + j;
    var v: f32;
    if (i < cube_max4096__LEN) {
      v = b_cube_max4096_leaf(cube_max4096_s[row * cube_max4096__LEN + i], i, cube_max4096__sc[1u]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_max4096_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_max4096__t[l] = st[4];
  workgroupBarrier();
  for (var h = 1u; h < cube_max4096__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_max4096__T; p += cube_max4096__T * 2u * h) {
      cube_max4096__t[p] = b_cube_max4096_join(cube_max4096__t[p], cube_max4096__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_max4096__out[row] = b_cube_max4096_fin(cube_max4096__t[0]);
  }
}

// Map mask4096
// ====

@group(0) @binding(0) var<storage, read> map_mask4096_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> map_mask4096__out: array<f32>;
@group(0) @binding(2) var<storage, read> map_mask4096__sc: array<u32>;

@compute @workgroup_size(256)
fn map_mask4096_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_mask4096__sc[0u]) {
    return;
  }
  let row = i / 4096u;
  let col = i % 4096u;
  map_mask4096__out[i] = b_map_mask4096_f(map_mask4096_s[i], col, map_mask4096__sc[1u]);
}

// Map exp4096
// ====

@group(0) @binding(0) var<storage, read> map_exp4096_s: array<f32>;
@group(0) @binding(1) var<storage, read> map_exp4096_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_exp4096__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_exp4096__sc: array<u32>;

@compute @workgroup_size(256)
fn map_exp4096_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_exp4096__sc[0u]) {
    return;
  }
  let row = i / 4096u;
  let col = i % 4096u;
  map_exp4096__out[i] = b_map_exp4096_f(map_exp4096_s[i], map_exp4096_m[row]);
}

// Cube sum4096
// ====

const cube_sum4096__N: u32 = 4096u;
const cube_sum4096__LEN: u32 = 4096u;
const cube_sum4096__T: u32 = 256u;
const cube_sum4096__Q: u32 = 16u;

@group(0) @binding(0) var<storage, read> cube_sum4096_e: array<f32>;
@group(0) @binding(1) var<storage, read_write> cube_sum4096__out: array<f32>;

var<workgroup> cube_sum4096__t: array<f32, cube_sum4096__T>;


@compute @workgroup_size(cube_sum4096__T)
fn cube_sum4096_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  var st: array<f32, 5>;
  for (var j = 0u; j < cube_sum4096__Q; j += 1u) {
    let i = l * cube_sum4096__Q + j;
    var v: f32;
    if (i < cube_sum4096__LEN) {
      v = b_cube_sum4096_leaf(cube_sum4096_e[row * cube_sum4096__LEN + i]);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_sum4096_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_sum4096__t[l] = st[4];
  workgroupBarrier();
  for (var h = 1u; h < cube_sum4096__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_sum4096__T; p += cube_sum4096__T * 2u * h) {
      cube_sum4096__t[p] = b_cube_sum4096_join(cube_sum4096__t[p], cube_sum4096__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_sum4096__out[row] = b_cube_sum4096_fin(cube_sum4096__t[0]);
  }
}

// Map prob4096
// ====

@group(0) @binding(0) var<storage, read> map_prob4096_e: array<f32>;
@group(0) @binding(1) var<storage, read> map_prob4096_z: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_prob4096__out: array<f32>;
@group(0) @binding(3) var<storage, read> map_prob4096__sc: array<u32>;

@compute @workgroup_size(256)
fn map_prob4096_main(@builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 256u + gid.x;
  if (i >= map_prob4096__sc[0u]) {
    return;
  }
  let row = i / 4096u;
  let col = i % 4096u;
  map_prob4096__out[i] = b_map_prob4096_f(map_prob4096_e[i], map_prob4096_z[row]);
}

// Cube weighted4096
// ====

const cube_weighted4096__N: u32 = 4096u;
const cube_weighted4096__LEN: u32 = 4096u;
const cube_weighted4096__T: u32 = 256u;
const cube_weighted4096__Q: u32 = 16u;

@group(0) @binding(0) var<storage, read> cube_weighted4096_pb: array<u32>;
@group(0) @binding(1) var<storage, read> cube_weighted4096_cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_weighted4096__out: array<f32>;
@group(0) @binding(3) var<storage, read> cube_weighted4096__sc: array<u32>;

var<workgroup> cube_weighted4096__t: array<f32, cube_weighted4096__T>;


@compute @workgroup_size(cube_weighted4096__T)
fn cube_weighted4096_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_weighted4096__sc[0u]) {
    return;
  }
var st: array<f32, 5>;
  for (var j = 0u; j < cube_weighted4096__Q; j += 1u) {
    let i = l * cube_weighted4096__Q + j;
    var v: f32;
    if (i < cube_weighted4096__LEN) {
      v = b_cube_weighted4096_leaf_cube_weighted4096_pb_cube_weighted4096_cb(row, i);
    } else {
      v = 0.0;
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_weighted4096_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_weighted4096__t[l] = st[4];
  workgroupBarrier();
  for (var h = 1u; h < cube_weighted4096__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_weighted4096__T; p += cube_weighted4096__T * 2u * h) {
      cube_weighted4096__t[p] = b_cube_weighted4096_join(cube_weighted4096__t[p], cube_weighted4096__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_weighted4096__out[row] = b_cube_weighted4096_fin(cube_weighted4096__t[0]);
  }
}
