// bend-wgsl {"lane":"bend-wgsl/2","entries":[{"kind":"cube","name":"xchk4","entry":"cube_xchk4_main","n":2048,"len":1153,"lanes":256,"node":"u32","per":"row","ins":[{"name":"f","ty":"u32","cls":"rowv","binding":0},{"name":"ab","ty":"bytes","cls":"bytes","binding":1},{"name":"bb","ty":"bytes","cls":"bytes","binding":2},{"name":"cb","ty":"bytes","cls":"bytes","binding":3},{"name":"db","ty":"bytes","cls":"bytes","binding":4}],"out":{"ty":"u32","binding":5},"sc":{"binding":6,"scalars":[],"lens":[{"name":"ab","at":1},{"name":"bb","at":2},{"name":"cb","at":3},{"name":"db","at":4}]},"rows_at":0},{"kind":"cube","name":"xchk1","entry":"cube_xchk1_main","n":256,"len":161,"lanes":256,"node":"u32","per":"row","ins":[{"name":"f","ty":"u32","cls":"rowv","binding":0},{"name":"ab","ty":"bytes","cls":"bytes","binding":1}],"out":{"ty":"u32","binding":2},"sc":{"binding":3,"scalars":[],"lens":[{"name":"ab","at":1}]},"rows_at":0}]}
// emitted by bend (wgsl.ts): the program's lane defs and runtime_wgsl

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

fn b_cube_xchk4_pad() -> u32 {
  return 0u;
}

fn b_cube_xchk4_join(_a_0: u32, _b_0: u32) -> u32 {
  return b_U32_max(_a_0, _b_0);
}

fn b_cube_xchk4_leaf_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db(_row_0: u32, _i_0: u32, _f_0: u32) -> u32 {
  return b_xchk_l4_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db((_i_0 == 0u), _f_0, _i_0);
}

fn b_cube_xchk4_fin(_t_0: u32) -> u32 {
  return _t_0;
}

fn b_cube_xchk1_pad() -> u32 {
  return 0u;
}

fn b_cube_xchk1_join(_a_0: u32, _b_0: u32) -> u32 {
  return b_U32_max(_a_0, _b_0);
}

fn b_cube_xchk1_leaf_cube_xchk1_ab(_row_0: u32, _i_0: u32, _f_0: u32) -> u32 {
  return b_xchk_l1_cube_xchk1_ab((_i_0 == 0u), _f_0, _i_0);
}

fn b_cube_xchk1_fin(_t_0: u32) -> u32 {
  return _t_0;
}

fn b_U32_max(_a_0: u32, _b_0: u32) -> u32 {
  return select(_a_0, _b_0, (_a_0 < _b_0));
}

fn b_xchk_l4_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db(_first_0: bool, _f_0: u32, _i_0: u32) -> u32 {
  if (_first_0) {
    return _f_0;
  } else {
    return b_xchk_abcd_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db(((_i_0 - 1u) < 416u), (_i_0 - 1u));
  }
}

fn b_xchk_l1_cube_xchk1_ab(_first_0: bool, _f_0: u32, _i_0: u32) -> u32 {
  if (_first_0) {
    return _f_0;
  } else {
    return b_xmax16_cube_xchk1_ab(((_i_0 - 1u) * 16u));
  }
}

fn b_xchk_abcd_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db(_lo_0: bool, _j_0: u32) -> u32 {
  if (_lo_0) {
    return b_xchk_ab_cube_xchk4_ab_cube_xchk4_bb((_j_0 < 160u), _j_0);
  } else {
    return b_xchk_cd_cube_xchk4_cb_cube_xchk4_db(((_j_0 - 416u) < 160u), (_j_0 - 416u));
  }
}

fn b_xmax16_cube_xchk1_ab(_c_0: u32) -> u32 {
  let _m0_0 = b_U32_max(b_U32_max(b_xt_cube_xchk1_ab(_c_0), b_xt_cube_xchk1_ab((_c_0 + 1u))), b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 2u)), b_xt_cube_xchk1_ab((_c_0 + 3u))));
  let _m1_0 = b_U32_max(b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 4u)), b_xt_cube_xchk1_ab((_c_0 + 5u))), b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 6u)), b_xt_cube_xchk1_ab((_c_0 + 7u))));
  let _m2_0 = b_U32_max(b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 8u)), b_xt_cube_xchk1_ab((_c_0 + 9u))), b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 10u)), b_xt_cube_xchk1_ab((_c_0 + 11u))));
  let _m3_0 = b_U32_max(b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 12u)), b_xt_cube_xchk1_ab((_c_0 + 13u))), b_U32_max(b_xt_cube_xchk1_ab((_c_0 + 14u)), b_xt_cube_xchk1_ab((_c_0 + 15u))));
  return b_U32_max(b_U32_max(_m0_0, _m1_0), b_U32_max(_m2_0, _m3_0));
}

fn b_xchk_ab_cube_xchk4_ab_cube_xchk4_bb(_lo_0: bool, _j_0: u32) -> u32 {
  if (_lo_0) {
    return b_xmax16_cube_xchk4_ab((_j_0 * 16u));
  } else {
    return b_xmax16_cube_xchk4_bb(((_j_0 - 160u) * 16u));
  }
}

fn b_xchk_cd_cube_xchk4_cb_cube_xchk4_db(_lo_0: bool, _j_0: u32) -> u32 {
  if (_lo_0) {
    return b_xmax16_cube_xchk4_cb((_j_0 * 16u));
  } else {
    return b_xmax16_cube_xchk4_db(((_j_0 - 160u) * 16u));
  }
}

fn b_xt_cube_xchk1_ab(_k_0: u32) -> u32 {
  return ((bw_cube_xchk1_ab((_k_0 * 4u)) & 2147483647u) - 562036736u);
}

fn b_xmax16_cube_xchk4_ab(_c_0: u32) -> u32 {
  let _m0_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_ab(_c_0), b_xt_cube_xchk4_ab((_c_0 + 1u))), b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 2u)), b_xt_cube_xchk4_ab((_c_0 + 3u))));
  let _m1_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 4u)), b_xt_cube_xchk4_ab((_c_0 + 5u))), b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 6u)), b_xt_cube_xchk4_ab((_c_0 + 7u))));
  let _m2_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 8u)), b_xt_cube_xchk4_ab((_c_0 + 9u))), b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 10u)), b_xt_cube_xchk4_ab((_c_0 + 11u))));
  let _m3_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 12u)), b_xt_cube_xchk4_ab((_c_0 + 13u))), b_U32_max(b_xt_cube_xchk4_ab((_c_0 + 14u)), b_xt_cube_xchk4_ab((_c_0 + 15u))));
  return b_U32_max(b_U32_max(_m0_0, _m1_0), b_U32_max(_m2_0, _m3_0));
}

fn b_xmax16_cube_xchk4_bb(_c_0: u32) -> u32 {
  let _m0_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_bb(_c_0), b_xt_cube_xchk4_bb((_c_0 + 1u))), b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 2u)), b_xt_cube_xchk4_bb((_c_0 + 3u))));
  let _m1_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 4u)), b_xt_cube_xchk4_bb((_c_0 + 5u))), b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 6u)), b_xt_cube_xchk4_bb((_c_0 + 7u))));
  let _m2_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 8u)), b_xt_cube_xchk4_bb((_c_0 + 9u))), b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 10u)), b_xt_cube_xchk4_bb((_c_0 + 11u))));
  let _m3_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 12u)), b_xt_cube_xchk4_bb((_c_0 + 13u))), b_U32_max(b_xt_cube_xchk4_bb((_c_0 + 14u)), b_xt_cube_xchk4_bb((_c_0 + 15u))));
  return b_U32_max(b_U32_max(_m0_0, _m1_0), b_U32_max(_m2_0, _m3_0));
}

fn b_xmax16_cube_xchk4_cb(_c_0: u32) -> u32 {
  let _m0_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_cb(_c_0), b_xt_cube_xchk4_cb((_c_0 + 1u))), b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 2u)), b_xt_cube_xchk4_cb((_c_0 + 3u))));
  let _m1_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 4u)), b_xt_cube_xchk4_cb((_c_0 + 5u))), b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 6u)), b_xt_cube_xchk4_cb((_c_0 + 7u))));
  let _m2_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 8u)), b_xt_cube_xchk4_cb((_c_0 + 9u))), b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 10u)), b_xt_cube_xchk4_cb((_c_0 + 11u))));
  let _m3_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 12u)), b_xt_cube_xchk4_cb((_c_0 + 13u))), b_U32_max(b_xt_cube_xchk4_cb((_c_0 + 14u)), b_xt_cube_xchk4_cb((_c_0 + 15u))));
  return b_U32_max(b_U32_max(_m0_0, _m1_0), b_U32_max(_m2_0, _m3_0));
}

fn b_xmax16_cube_xchk4_db(_c_0: u32) -> u32 {
  let _m0_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_db(_c_0), b_xt_cube_xchk4_db((_c_0 + 1u))), b_U32_max(b_xt_cube_xchk4_db((_c_0 + 2u)), b_xt_cube_xchk4_db((_c_0 + 3u))));
  let _m1_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_db((_c_0 + 4u)), b_xt_cube_xchk4_db((_c_0 + 5u))), b_U32_max(b_xt_cube_xchk4_db((_c_0 + 6u)), b_xt_cube_xchk4_db((_c_0 + 7u))));
  let _m2_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_db((_c_0 + 8u)), b_xt_cube_xchk4_db((_c_0 + 9u))), b_U32_max(b_xt_cube_xchk4_db((_c_0 + 10u)), b_xt_cube_xchk4_db((_c_0 + 11u))));
  let _m3_0 = b_U32_max(b_U32_max(b_xt_cube_xchk4_db((_c_0 + 12u)), b_xt_cube_xchk4_db((_c_0 + 13u))), b_U32_max(b_xt_cube_xchk4_db((_c_0 + 14u)), b_xt_cube_xchk4_db((_c_0 + 15u))));
  return b_U32_max(b_U32_max(_m0_0, _m1_0), b_U32_max(_m2_0, _m3_0));
}

fn b_xt_cube_xchk4_ab(_k_0: u32) -> u32 {
  return ((bw_cube_xchk4_ab((_k_0 * 4u)) & 2147483647u) - 562036736u);
}

fn b_xt_cube_xchk4_bb(_k_0: u32) -> u32 {
  return ((bw_cube_xchk4_bb((_k_0 * 4u)) & 2147483647u) - 562036736u);
}

fn b_xt_cube_xchk4_cb(_k_0: u32) -> u32 {
  return ((bw_cube_xchk4_cb((_k_0 * 4u)) & 2147483647u) - 562036736u);
}

fn b_xt_cube_xchk4_db(_k_0: u32) -> u32 {
  return ((bw_cube_xchk4_db((_k_0 * 4u)) & 2147483647u) - 562036736u);
}

fn bg_cube_xchk4_ab(i: u32) -> u32 {
  if (i >= cube_xchk4__sc[1u]) {
    return 0u;
  }
  return (cube_xchk4_ab[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_xchk4_ab(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_xchk4__sc[1u]) {
    return cube_xchk4_ab[i >> 2u];
  }
  return bg_cube_xchk4_ab(i) | (bg_cube_xchk4_ab(i + 1u) << 8u) | (bg_cube_xchk4_ab(i + 2u) << 16u)
    | (bg_cube_xchk4_ab(i + 3u) << 24u);
}

fn bg_cube_xchk4_bb(i: u32) -> u32 {
  if (i >= cube_xchk4__sc[2u]) {
    return 0u;
  }
  return (cube_xchk4_bb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_xchk4_bb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_xchk4__sc[2u]) {
    return cube_xchk4_bb[i >> 2u];
  }
  return bg_cube_xchk4_bb(i) | (bg_cube_xchk4_bb(i + 1u) << 8u) | (bg_cube_xchk4_bb(i + 2u) << 16u)
    | (bg_cube_xchk4_bb(i + 3u) << 24u);
}

fn bg_cube_xchk4_cb(i: u32) -> u32 {
  if (i >= cube_xchk4__sc[3u]) {
    return 0u;
  }
  return (cube_xchk4_cb[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_xchk4_cb(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_xchk4__sc[3u]) {
    return cube_xchk4_cb[i >> 2u];
  }
  return bg_cube_xchk4_cb(i) | (bg_cube_xchk4_cb(i + 1u) << 8u) | (bg_cube_xchk4_cb(i + 2u) << 16u)
    | (bg_cube_xchk4_cb(i + 3u) << 24u);
}

fn bg_cube_xchk4_db(i: u32) -> u32 {
  if (i >= cube_xchk4__sc[4u]) {
    return 0u;
  }
  return (cube_xchk4_db[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_xchk4_db(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_xchk4__sc[4u]) {
    return cube_xchk4_db[i >> 2u];
  }
  return bg_cube_xchk4_db(i) | (bg_cube_xchk4_db(i + 1u) << 8u) | (bg_cube_xchk4_db(i + 2u) << 16u)
    | (bg_cube_xchk4_db(i + 3u) << 24u);
}

fn bg_cube_xchk1_ab(i: u32) -> u32 {
  if (i >= cube_xchk1__sc[1u]) {
    return 0u;
  }
  return (cube_xchk1_ab[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn bw_cube_xchk1_ab(i: u32) -> u32 {
  if ((i & 3u) == 0u && i + 4u <= cube_xchk1__sc[1u]) {
    return cube_xchk1_ab[i >> 2u];
  }
  return bg_cube_xchk1_ab(i) | (bg_cube_xchk1_ab(i + 1u) << 8u) | (bg_cube_xchk1_ab(i + 2u) << 16u)
    | (bg_cube_xchk1_ab(i + 3u) << 24u);
}


// Cube xchk4
// ====

const cube_xchk4__N: u32 = 2048u;
const cube_xchk4__LEN: u32 = 1153u;
const cube_xchk4__T: u32 = 256u;
const cube_xchk4__Q: u32 = 8u;

@group(0) @binding(0) var<storage, read> cube_xchk4_f: array<u32>;
@group(0) @binding(1) var<storage, read> cube_xchk4_ab: array<u32>;
@group(0) @binding(2) var<storage, read> cube_xchk4_bb: array<u32>;
@group(0) @binding(3) var<storage, read> cube_xchk4_cb: array<u32>;
@group(0) @binding(4) var<storage, read> cube_xchk4_db: array<u32>;
@group(0) @binding(5) var<storage, read_write> cube_xchk4__out: array<u32>;
@group(0) @binding(6) var<storage, read> cube_xchk4__sc: array<u32>;

var<workgroup> cube_xchk4__t: array<u32, cube_xchk4__T>;


@compute @workgroup_size(cube_xchk4__T)
fn cube_xchk4_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_xchk4__sc[0u]) {
    return;
  }
var st: array<u32, 4>;
  for (var j = 0u; j < cube_xchk4__Q; j += 1u) {
    let i = l * cube_xchk4__Q + j;
    var v: u32;
    if (i < cube_xchk4__LEN) {
      v = b_cube_xchk4_leaf_cube_xchk4_ab_cube_xchk4_bb_cube_xchk4_cb_cube_xchk4_db(row, i, cube_xchk4_f[row]);
    } else {
      v = b_cube_xchk4_pad();
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_xchk4_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_xchk4__t[l] = st[3];
  workgroupBarrier();
  for (var h = 1u; h < cube_xchk4__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_xchk4__T; p += cube_xchk4__T * 2u * h) {
      cube_xchk4__t[p] = b_cube_xchk4_join(cube_xchk4__t[p], cube_xchk4__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_xchk4__out[row] = b_cube_xchk4_fin(cube_xchk4__t[0]);
  }
}

// Cube xchk1
// ====

const cube_xchk1__N: u32 = 256u;
const cube_xchk1__LEN: u32 = 161u;
const cube_xchk1__T: u32 = 256u;
const cube_xchk1__Q: u32 = 1u;

@group(0) @binding(0) var<storage, read> cube_xchk1_f: array<u32>;
@group(0) @binding(1) var<storage, read> cube_xchk1_ab: array<u32>;
@group(0) @binding(2) var<storage, read_write> cube_xchk1__out: array<u32>;
@group(0) @binding(3) var<storage, read> cube_xchk1__sc: array<u32>;

var<workgroup> cube_xchk1__t: array<u32, cube_xchk1__T>;


@compute @workgroup_size(cube_xchk1__T)
fn cube_xchk1_main(@builtin(local_invocation_index) l: u32,
  @builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let row = g.y * ng.x + g.x;
  if (row >= cube_xchk1__sc[0u]) {
    return;
  }
var st: array<u32, 1>;
  for (var j = 0u; j < cube_xchk1__Q; j += 1u) {
    let i = l * cube_xchk1__Q + j;
    var v: u32;
    if (i < cube_xchk1__LEN) {
      v = b_cube_xchk1_leaf_cube_xchk1_ab(row, i, cube_xchk1_f[row]);
    } else {
      v = b_cube_xchk1_pad();
    }
    var lv = 0u;
    for (var t = j; (t & 1u) != 0u; t >>= 1u) {
      v = b_cube_xchk1_join(st[lv], v);
      lv += 1u;
    }
    st[lv] = v;
  }
  cube_xchk1__t[l] = st[0];
  workgroupBarrier();
  for (var h = 1u; h < cube_xchk1__T; h *= 2u) {
    for (var p = l * 2u * h; p < cube_xchk1__T; p += cube_xchk1__T * 2u * h) {
      cube_xchk1__t[p] = b_cube_xchk1_join(cube_xchk1__t[p], cube_xchk1__t[p + h]);
    }
    workgroupBarrier();
  }
  if (l == 0u) {
    cube_xchk1__out[row] = b_cube_xchk1_fin(cube_xchk1__t[0]);
  }
}
