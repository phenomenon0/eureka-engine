// NOTE: Bend's runtime was designed by humans, but this file was mostly written
// by AI's, as it includes a ton of optimizations. It works and tests pass, yet,
// bugs ARE expected. It will take some time for the compiler to be stable.

import * as fs from "node:fs";

import * as Bend from "./bend.ts";

// Comp
// ====

// Types
// =====

type Kind = "w32" | "w64" | "box";

type Lay = { ks: Kind[]; arms: Arm[] | null };

type Arm = { k: Name; fs: Field[] };

type Field = { at: number; lay: Lay };

type Val = { ws: string[]; lay: Lay; stat: boolean };

type Bind = { val: Val; n: number; A: HTerm };

type Dst = Val | null;

type Seg = {
  fid: string;
  def: Name;
  ret: Lay;
  lines: string[];
  params: string[];
  ks: Kind[];
  frame: { pop: number; at: number[] } | null;
  refs: Set<string>;
  host?: boolean;
  spin?: boolean;
  fork?: boolean;
};

type Spine = {
  h: HTerm;
  t: HTerm;
  all: HTerm[];
  args: HTerm[];
  tld: TLD | undefined;
  call: Call | null;
};

type HTerm = Bend.HTerm;

type Name = Bend.Name;

type Def  = Bend.Def & { h?: HTerm };

type TLD  = Bend.ADT | Def;

type Src = { refs: Set<Name>; deps: Set<Name>; flat: boolean };

type Carb = {
  // a copy of the book whose defs carry their raised body (def_body)
  book: Bend.Book & { tlds: Record<Name, TLD> };
  bangs: Set<Name>;
  sites: Map<Name, number>;
  hot: Set<Name>;
  stat: Set<Name>;
  own: Set<string>;
  lend: Set<string>;
};

type File = Carb & {
  spares: { words: number; name: string; z: boolean }[];
  fresh: Map<string, number>;
  uses: Map<Probe, Bind>;
  brwl: Map<string, string>;
  rest: HTerm[];
  def: Name;
  segs: Seg[];
  seg: Seg;
  tab: number;
  js: boolean;
  cids: Map<string, number>;
  tabs: Map<string, number>;
  spins: { fid: string; text: string; refs: Set<string> }[];
  spun: Map<string, string>;
  clos: Set<string>;
  img: string[];
  lits: Map<string, number>;
  consts: Map<string, Map<HTerm, Val>>;
  reqs: string;
  fuel: number;
};

type Gen = string | ((xs: string[]) => string);

type Native = Record<Name, {
  intr: Gen;
  elim?: string[];
  cond?: string;
}>;

type Of<K> = Extract<HTerm, { $: K }>;

type HAll = Of<"All">;

type Probe = Of<"Var">;

type HAdt = Of<"ADT">;

type HLet = Of<"Let">;

type UMap = Bend.PMap<number>;

type Chain = [HTerm, number | null][];

type Leaf = [HTerm, number, number, number];

type Call = {
  k: Name;
  args: HTerm[];
  all: HTerm[];
  bang?: boolean;
};

type Intr = {
  C?: string | string[];
  call?: boolean;
  peek?: number[];
  JS: Gen;
};

type Dom = [Bend.Quant, Name, HTerm];

type Sig = { live: Dom[]; lays: Lay[]; ret: Lay };

type Show = { cells: (number | Name)[]; names: string[] };

// Constants
// =========

const CLO_APPLY = "Clo.apply";

const ATOM   = /^(?:[A-Za-z_$][A-Za-z0-9_$]*|\d+n?|\d+\.\d+)$/;
const STRLIT = new RegExp("^\"(?:[^\"\\\\]|\\\\.)*\"$");

// The term nodes a segment may gain by compile-time folding.
const FOLD_FUEL = 8192;

// A native this long is a call on both lanes: the device inlines every
// native (slow Metal compiles); at 128 raytrace lost 31% on PAR-CPU.
const SPIN_FAR = 256;

const USE0 = Bend.Emp<number>();

const W32: Lay = { ks: ["w32"], arms: null };

const BOX: Lay = { ks: ["box"], arms: null };

const W64: Lay = { ks: ["w64"], arms: null };

// A U64, I64 or F64: a w64 like Nat's (lay_eq), but any bit pattern, where a
// Nat stays below 2^48 and so always reads as a trivial Term. It is told
// apart by identity: val_box boxes it (x64_box), lay_pack keeps it out of a
// slot another arm drops as a Term.
const X64: Lay = { ks: ["w64"], arms: null };

const WORDS: Record<string, Lay> = Object.setPrototypeOf(
  { U32: W32, F32: W32, F64: X64, Nat: W64, U64: X64, I64: X64 }, null);

// The widest flat layout (u8 arity tables); the shader's Tri is 24 words.
const WIDE = 255;

const ERRS = ("|*|*|out of memory: run again with a bigger span, as in"
  + " --gpu 8GB|a function the device does not hold|a Nat past the"
  + " largest immediate 2^48-1|*|memory fault (machine stack overflow?)|an"
  + " array past the deepest block class 31|a string past the maximum length 2^31").split("|")
  .map((e) => e === "*" ? "runtime fail-stop" : e);

// Operations
// ----------

const CMPS = "is_eq:==:=== is_ne:!=:!== is_lt:< is_le:<= is_gt:> is_ge:>=";

const OPERATIONS: Record<string, Intr> = Object.setPrototypeOf({
  ...tpl_ops("u32_", "add:+ sub:- and:& or:| xor:^",
    "U32_BIN($0, $o, $1)", "(($0 $o $1) >>> 0)"),
  ...tpl_ops("u32_", CMPS, "U32_BIN($0, $o, $1)", "($0 $o $1)"),
  u32_mul: {
    C:  "U32_BIN($0, *, $1)",
    JS: "(Math.imul($0, $1) >>> 0)",
  },
  u32_div: {
    C:  "((u32)($1) == 0 ? 0 : (u64)U32_QUO((u32)($0), (u32)($1)))",
    JS: "($1 === 0 ? 0 : ($0 / $1) >>> 0)",
  },
  u32_mod: {
    C:  "((u32)($1) == 0 ? $0 : U32_BIN($0, -,"
      + " U32_QUO((u32)($0), (u32)($1)) * $1))",
    JS: "($1 === 0 ? $0 : $0 % $1)",
  },
  ...tpl_ops("u32_", "inc:+ shl:<< shr:>>:>>>", "U32_BIN($0, $o, 1)",
    "(($0 $o 1) >>> 0)"),
  ...tpl_ops("u32_", "shln:<< shrn:>>:>>>", "($1 >= 32 ? 0 : U32_BIN($0, $o, $1))",
    "($1 >= 32n ? 0 : ($0 $o Number($1)) >>> 0)"),
  u32_not: {
    C:  "((u64)~(u32)($0))",
    JS: "(~$0 >>> 0)",
  },
  u32_is_zero: {
    C:  "U32_BIN($0, ==, 0)",
    JS: "($0 === 0)",
  },
  u32_cmp: {
    C:  "(U32_BIN($0, >, $1) + U32_BIN($0, >=, $1))",
    JS: "cmp_new($0, $1)",
  },
  u32_to_f32: {
    C:  "f32_rewrap((f32)(u32)($0))",
    JS: "Math.fround($0)",
  },
  u32_to_nat: {
    C:  "$0",
    JS: "BigInt($0)",
  },
  u32_from_nat: {
    C:  "((u64)(u32)($0))",
    JS: "Number($0 & 0xFFFFFFFFn)",
  },
  ...tpl_w64("u64_"),
  ...tpl_ops("u64_", CMPS, "((u64)((u64)($0) $o (u64)($1)))", "($0 $o $1)"),
  u64_cmp: {
    C:  "(((u64)($0) > (u64)($1)) + ((u64)($0) >= (u64)($1)))",
    JS: "cmp_new($0, $1)",
  },
  ...tpl_w64("i64_"),
  ...tpl_ops("i64_", CMPS, "((u64)((int64_t)($0) $o (int64_t)($1)))",
    "(BigInt.asIntN(64, $0) $o BigInt.asIntN(64, $1))"),
  i64_cmp: {
    C:  "(((int64_t)($0) > (int64_t)($1))"
      + " + ((int64_t)($0) >= (int64_t)($1)))",
    JS: "cmp_new(BigInt.asIntN(64, $0), BigInt.asIntN(64, $1))",
  },
  i64_neg: {
    C:  "((u64)0 - (u64)($0))", // -(int64_t)min is UB
    JS: "((-$0) & 0xFFFFFFFFFFFFFFFFn)",
  },
  i64_shr_s: {
    C:  "((u64)((int64_t)($0) >> 1))",
    JS: "BigInt.asUintN(64, BigInt.asIntN(64, $0) >> 1n)",
  },
  i64_shr_s_n: {
    C:  "((u64)((int64_t)($0) >> ($1 >= 64 ? 63 : $1)))",
    JS: "BigInt.asUintN(64,"
      + " BigInt.asIntN(64, $0) >> ($1 >= 64n ? 63n : $1))",
  },
  i64_is_neg: {
    C:  "((u64)((int64_t)($0) < 0))",
    JS: "(BigInt.asIntN(64, $0) < 0n)",
  },
  ...tpl_ops("f32_", "add:+ sub:- mul:* div:/",
    "f32_rewrap(f32_unbox($0) $o f32_unbox($1))", "Math.fround($0 $o $1)"),
  f32_neg: {
    C:  "f32_rewrap(-f32_unbox($0))",
    JS: "(-$0)",
  },
  ...tpl_ops("f32_", CMPS, "((u64)(f32_unbox($0) $o f32_unbox($1)))",
    "($0 $o $1)"),
  ...tpl_ops("f32_", "sqrt exp log log2 log10 sin cos tan asin acos atan"
    + " sinh cosh tanh floor ceil trunc abs:fabs:abs",
    "f32_rewrap((f32)$o(f32_unbox($0)))", "Math.fround(Math.$o($0))"),
  ...tpl_ops("f32_", "pow atan2",
    "f32_rewrap((f32)$o(f32_unbox($0), f32_unbox($1)))",
    "Math.fround(Math.$o($0, $1))"),
  f32_mod: {
    C:  "f32_rewrap((f32)fmod(f32_unbox($0), f32_unbox($1)))",
    JS: "Math.fround($0 % $1)",
  },
  f32_to_u32: {
    C:  "f32_to_u32($0)",
    JS: "($0 >= 1 && $0 < 4294967296 ? Math.floor($0) : 0)",
  },
  f32_bits: {
    C:  "$0",
    JS: "f32_bits($0)",
  },
  ...tpl_ops("f32_", "show read", "f32_$o(e, $0)", "f32_$o($0)", true),
  ...tpl_ops("f64_", "add:+ sub:- mul:* div:/",
    "f64_rewrap(f64_unbox($0) $o f64_unbox($1))", "($0 $o $1)"),
  f64_neg: {
    C:  "f64_rewrap(-f64_unbox($0))",
    JS: "(-$0)",
  },
  ...tpl_ops("f64_", CMPS, "((u64)(f64_unbox($0) $o f64_unbox($1)))",
    "($0 $o $1)"),
  ...tpl_ops("f64_", "sqrt exp log log2 log10 sin cos tan asin acos atan"
    + " sinh cosh tanh floor ceil trunc abs:fabs:abs",
    "f64_rewrap((double)$o(f64_unbox($0)))", "Math.$o($0)"),
  ...tpl_ops("f64_", "pow atan2",
    "f64_rewrap((double)$o(f64_unbox($0), f64_unbox($1)))",
    "Math.$o($0, $1)"),
  f64_mod: {
    C:  "f64_rewrap(fmod(f64_unbox($0), f64_unbox($1)))",
    JS: "($0 % $1)",
  },
  f64_to_u32: {
    C:  "f64_to_u32($0)",
    JS: "($0 >= 1 && $0 < 4294967296 ? Math.floor($0) : 0)",
  },
  f64_bits: {
    C:  "$0",
    JS: "f64_bits($0)",
  },
  ...tpl_ops("f64_", "show read", "f64_$o(e, $0)", "f64_$o($0)", true),
  u32_to_f64: {
    C:  "f64_rewrap((double)(u32)($0))",
    JS: "($0)",
  },
  f32_to_f64: {
    C:  "f64_rewrap((double)f32_unbox($0))",
    JS: "($0)",
  },
  f64_to_f32: {
    C:  "f32_rewrap((f32)f64_unbox($0))",
    JS: "Math.fround($0)",
  },
  nat_add: {
    C:  "nat_chk(e, $0 + $1)",
    JS: "nat_chk($0 + $1)",
  },
  nat_sub: {
    C:  "($0 < $1 ? 0 : $0 - $1)",
    JS: "($0 < $1 ? 0n : $0 - $1)",
  },
  nat_mul: {
    C:  "nat_mul(e, $0, $1)",
    JS: "nat_chk($0 * $1)",
  },
  nat_double: {
    C:  "nat_chk(e, $0 + $0)",
    JS: "nat_chk($0 << 1n)",
  },
  nat_cmp: {
    C:  "(($0 > $1) + ($0 >= $1))",
    JS: "cmp_new($0, $1)",
  },
  nat_is_lt: {
    C:  "($0 < $1)",
    JS: "($0 < $1)",
  },
  nat_min: {
    C:  "($0 < $1 ? $0 : $1)",
    JS: "($0 < $1 ? $0 : $1)",
  },
  nat_max: {
    C:  "($0 < $1 ? $1 : $0)",
    JS: "($0 < $1 ? $1 : $0)",
  },
  nat_show: { C: "str_show_u64(e, (u64)($0))", JS: "String($0)" },
  u32_show: { C: "str_show_u64(e, (u64)(u32)($0))", JS: "String($0 >>> 0)" },
  nat_divmod: {
    C:    ["($1 == 0 ? 0 : $0 / $1)", "($1 == 0 ? $0 : $0 % $1)"],
    call: true,
    JS:   "nat_divmod($0, $1)",
  },
  ...tpl_ops("bool_", "or:|:|| xor:^:!==", "(($0) $o ($1))", "($0 $o $1)"),
  // String adapters consume inputs, except cmp which returns both original
  // owned handles. Array expression results are flattened; string expressions
  // returning aggregates use the generic boxed-result adapter.
  ...tpl_ops("string_", "length hash from_list splitlines",
    "str_$o_take(e, $0)", "str_$o($0)"),
  string_is_empty: { C: "(str_length_take(e, $0) == 0)", JS: '($0 === "")' },
  ...tpl_ops("string_", "get:false:str_get get_end:true:str_get_end",
    "str_get_take(e, $0, $1, $o)", "$o($0, $1)"),
  string_take: { C: "str_slice_take(e, $0, 0, $1)", JS: "$0.slice(0, str_offset($0, $1))" },
  string_drop: { C: "str_slice_take(e, $0, $1, STR_LIMIT)", JS: "$0.slice(str_offset($0, $1))" },
  ...tpl_ops("string_", "slice replace",
    "str_$o_take(e, $0, $1, $2)", "str_$o($0, $1, $2)"),
  ...tpl_ops("string_", "take_end:true drop_end:false",
    "str_end_take(e, $0, $1, $o)", "str_end($0, $1, $o)"),
  string_cut: {
    C: ["str_slice_take(e, term_keep(e, $0), 0, $1)", "str_slice_take(e, $0, $1, STR_LIMIT)"],
    JS: '{$: "Tuple", fst: $0.slice(0, str_offset($0, $1)), snd: $0.slice(str_offset($0, $1))}',
  },
  string_copy: { C: "str_copy_take(e, $0)", JS: '($0.split("").join(""))' },
  string_append: { C: "str_append_take(e, $0, $1)", JS: "($0 + $1)" },
  string_reverse: { C: "str_transform_take(e, $0, 0)", JS: '[...$0].reverse().join("")' },
  string_cmp: { C: ["$0", "$1", "str_order_peek(e, $0, $1)"], JS: "str_cmp($0, $1)" },
  ...tpl_ops("string_", "order join repeat partition",
    "str_$o_take(e, $0, $1)", "str_$o($0, $1)"),
  string_eq: { C: "str_eq_take(e, $0, $1)", JS: "($0 === $1)" },
  ...tpl_ops("string_", "is_lt:< is_le:<= is_gt:> is_ge:>=",
    "(str_order_take(e, $0, $1) $o 1)",
    '(({LT: 0, EQ: 1, GT: 2})[str_order($0, $1).$] $o 1)'),
  ...tpl_ops("string_", "starts_with:false:startsWith ends_with:true:endsWith",
    "str_edge_take(e, $0, $1, $o)", "$0.$o($1)"),
  string_split: { C: "str_split_take(e, $0, $1, false)", JS: "str_split($0, $1)" },
  string_lines: { C: "str_split_take(e, $0, 10, false)", JS: 'str_list($0.split("\\n"))' },
  string_trim_start: { C: "str_trim_take(e, $0, 1)", JS: '$0.replace(/^[ \\t-\\r]+/, "")' },
  string_trim_end: { C: "str_trim_take(e, $0, 2)", JS: '$0.replace(/[ \\t-\\r]+$/, "")' },
  string_trim: { C: "str_trim_take(e, $0, 3)", JS: '$0.replace(/^[ \\t-\\r]+|[ \\t-\\r]+$/g, "")' },
  string_to_upper: { C: "str_transform_take(e, $0, 1)", JS: '$0.replace(/[a-z]/g, c => String.fromCharCode(c.charCodeAt(0) - 32))' },
  string_to_lower: { C: "str_transform_take(e, $0, 2)", JS: '$0.replace(/[A-Z]/g, c => String.fromCharCode(c.charCodeAt(0) + 32))' },
  string_to_list: { C: "str_to_list_take(e, $0)", JS: "str_list([...$0])" },
  string_concat: { C: "str_join_take(e, $0, term_pak(CID_SNIL, 0))", JS: 'str_join($0, "")' },
  string_words: { C: "str_split_take(e, $0, 0, true)", JS: 'str_list($0.split(/[ \\t-\\r]+/).filter(s => s !== ""))' },
  ...tpl_ops("string_", "find:false find_last:true",
    "str_find_take(e, $0, $1, $o)", "str_find($0, $1, $o)"),
  string_contains: { C: "(str_search_take(e, $0, $1, 0) != STR_ABSENT)", JS: "$0.includes($1)" },
  string_count: { C: "str_search_take(e, $0, $1, 2)", JS: "str_count($0, $1)" },
  string_split_on: { C: "str_split_on_take(e, $0, $1)", JS: 'str_list($1 === "" ? [$0] : $0.split($1))' },
  string_capitalize: { C: "str_transform_take(e, $0, 3)", JS: "str_capitalize($0)" },
  string_zfill: { C: "str_pad_take(e, $0, $1, 48, 2)", JS: 'str_pad($0, $1, "0", 2)' },
  ...tpl_ops("string_", "pad_start:0 pad_end:1",
    "str_pad_take(e, $0, $1, $2, $o)", "str_pad($0, $1, $2, $o)"),
  // The Pike VM of base, natively: a C Regex arrives flat (prog, ngroups).
  ...tpl_ops("regex_", "exec:false match_at:true",
    "re_exec_take(e, $0, $1, $2, $3, $o)", "re_exec($0, $1, $2, $o)"),
  // Map.bit borrows the key and returns it: the tree order and the 33-bit key
  // protocol of the reference definition, O(1) on C, an endpoint scan on JS.
  map_bit: { C: ["$0", "str_bit_peek(e, $0, $1)"], JS: "map_bit($0, $1)" },
  // Bytes: len, get, span and the finds borrow the buffer (peek): the call site
  // neither shares nor drops it, so a scan loop over one buffer touches no
  // count; push appends in place into an unshared buffer's spare room.
  bytes_get: { C: "str_byte_peek(e, $0, $1)", peek: [0], JS: "bytes_get($0, $1)" },
  bytes_len: { C: "str_len_peek(e, $0)", peek: [0], JS: "str_length($0)" },
  bytes_word_le: { C: "str_word_le_peek(e, $0, $1)", peek: [0], JS: "bytes_word_le($0, $1)" },
  bytes_push: { C: "str_push_take(e, $0, $1)", JS: "bytes_push($0, $1)" },
  bytes_span: { C: "str_span_peek(e, $0, $1, $2, $3)", peek: [0],
    JS: "bytes_span($0, $1, $2, $3)" },
  bytes_find_byte: { C: "str_find_byte_peek(e, $0, $1, $2)", peek: [0],
    JS: "bytes_find_byte($0, $1, $2)" },
  bytes_find_any: { C: "str_find_any_peek(e, $0, $1, $2, $3, $4, $5)", peek: [0],
    JS: "bytes_find_any($0, $1, $2, $3, $4, $5)" },
  // the C lane lays these out by the element (arr_op)
  ...Object.fromEntries(Object.entries({
    new: "array_new($0, $1)", set: "($0[$1 % $0.length] = $2, $0)",
    get: "{$: \"Tuple\", fst: $0, snd: $0[$1 % $0.length]}",
    swap: "array_rmw($0, $1, () => $2)", get4: "array_get4($0, $1)",
    size: "{$: \"Tuple\", fst: $0, snd: $0.length}",
  }).map(([k, JS]) => ["array_" + k, { call: true, JS }])),
  array_clone: {
    C:    ["$0", "blk_copy(e, $0)"],
    call: true,
    JS:   "{$: \"Tuple\", fst: $0, snd: $0.slice()}",
  },
  ...Object.fromEntries(Object.entries({
    add: "(o + $2) >>> 0", min: "Math.min(o, $2)", max: "Math.max(o, $2)",
    and: "(o & $2) >>> 0", or: "(o | $2) >>> 0", xor: "(o ^ $2) >>> 0",
    exch: "$2", cmpx: "o === $2 ? $3 : o", fadd: "Math.fround(o + $2)",
  }).map(([k, js]) => ["array_atomic_" + k.replace("cmpx", "cas"), {
    C:    ["$0", "a32_" + k + "(blk_ptr(e.mem, blk_loc(e.mem, $0),"
      + " blk_at($0, $1, 0)), (u32)$2" + (k === "cmpx" ? ", (u32)$3)" : ")")],
    call: true,
    JS:   "array_rmw($0, $1, (o) => " + js + ")",
  }])),
}, null);

// Optimized
// ---------

// Per native constructor: its builder, field readers and optional test.
const OPTIMIZED: Record<Name, Native> = Object.setPrototypeOf({
  Nat: {
    Zero: { intr: "0n" },
    Succ: { intr: tpl_nat("n", "nat_chk($0 + 1n)") },
  },
  Bool: {
    False: { intr: "false", cond: "!$0" },
    True: { intr: "true", cond: "$0" },
  },
  U32: {
    U32: { intr: "word_to_u32($0)" },
  },
  F32: {
    F32: { intr: "f32_from_bits(word_to_u32($0))" },
  },
  F64: {
    F64: { intr: "f64_from_bits(word_to_u64($0))", elim: ["u64_to_word(f64_bits($0))"] },
  },
  U64: {
    U64: { intr: "word_to_u64($0)", elim: ["u64_to_word($0)"] },
  },
  I64: {
    I64: { intr: "word_to_u64($0)", elim: ["u64_to_word($0)"] },
  },
  Char: {
    Chr: {
      intr: ([c]: string[]) => {
        const n = Number(c);
        return /^\d+$/.test(c)
          && (n < 0xd800 || n >= 0xe000 && n <= 0x10ffff)
          ? JSON.stringify(String.fromCodePoint(n))
          : "char_new(" + c + ")";
      },
      elim: ["char_code($0)"],
    },
  },
  Array: {
    ALeaf: { intr: "[$0]", elim: ["$0[0]"], cond: "$0.length === 1" },
    ANode: {
      intr: "array_node($0, $1)",
      elim: ["$0.slice(0, $0.length >> 1)", "$0.slice($0.length >> 1)"],
      cond: "$0.length !== 1",
    },
  },
  String: {
    SNil: { intr: "\"\"", cond: "$0 === \"\"" },
    SCon: {
      intr: ([h, t]: string[]) => STRLIT.test(h) && STRLIT.test(t)
        ? JSON.stringify(JSON.parse(h) + JSON.parse(t))
        : "str_prepend(" + h + ", " + t + ")",
      elim: ["($0.codePointAt(0) > 0xFFFF ? $0.slice(0, 2) : $0[0])",
        "($0.codePointAt(0) > 0xFFFF ? $0.slice(2) : $0.slice(1))"],
      cond: "$0 !== \"\"",
    },
  },
} satisfies Record<Name, Native>, null);

// Native
// ------

// sin, cos and tan are fast:: (cheap, the same pixels); the rest precise::
const SHIMS = "sqrt exp log log2 log10 sin cos tan pow fmod".split(" ")
  .map((n) => "#define " + n.padEnd(5) + (["sin", "cos", "tan"].includes(n)
    ? " fast::" : " precise::") + n).join("\n")
  + "\n#define atan2 atan2_c99";

const NATIVE = {
  C: String.raw`
#ifdef __METAL_VERSION__
// Metal's atan2 is NaN at the origin; libm answers +-0 or +-pi there
INLINE f32 atan2_c99(f32 y, f32 x) {
  return y == 0.0f && x == x
    ? copysign(signbit(x) ? M_PI_F : 0.0f, y) : atan2(y, x);
}
${SHIMS}
#endif

#define U32_BIN(a, o, b) ((u64)((u32)(a) o (u32)(b)))

// Metal folds a constant dividend within 128 of 2^32 through an f32: divide
// its half, then fix the odd bit.
#define U32_QUO(a, b) \
  ((a) / 2 / (b) * 2 + ((a) - (a) / 2 / (b) * 2 * (b) >= (b)))

INLINE f32 f32_unbox(u64 x) {
  union { u32 u; f32 f; } p = { (u32)x };
  return p.f;
}

INLINE u64 f32_rewrap(f32 x) {
  union { f32 f; u32 u; } p = { x };
  return p.u;
}

INLINE U32 f32_to_u32(U32 a) {
  f32 v = f32_unbox(a);
  return v >= 0.0f && v < 4294967296.0f ? (u32)v : 0;
}

#ifndef __METAL_VERSION__

INLINE f64 f64_unbox(u64 x) {
  union { u64 u; f64 f; } p = { x };
  return p.f;
}

INLINE u64 f64_rewrap(f64 x) {
  union { f64 f; u64 u; } p = { x };
  return p.u;
}

INLINE U32 f64_to_u32(U32 a) {
  f64 v = f64_unbox(a);
  return v >= 0.0 && v < 4294967296.0 ? (u32)v : 0;
}

#endif

INLINE Nat nat_chk(Env e, Nat n) {
  if (n > NAT_IMM) {
    err_post(e.mem, ERR_NATS);
    return NAT_IMM;
  }
  return n;
}

INLINE Nat nat_mul(Env e, Nat a, Nat b) {
  return nat_chk(e, b != 0 && a > NAT_IMM / b ? NAT_IMM + 1 : a * b);
}

#if DEVICE

#define f32_show(e, x) (err_post(e.mem, ERR_FIDS), 0)
#define f32_read(e, s) (err_post(e.mem, ERR_FIDS), 0)

#define f64_show(e, x) (err_post(e.mem, ERR_FIDS), 0)
#define f64_read(e, s) (err_post(e.mem, ERR_FIDS), 0)

#else

static Term f32_show(Env e, Term x);
static Term f32_read(Env e, Term s);
static Term f64_show(Env e, Term x);
static Term f64_read(Env e, Term s);

#endif
`.slice(1),
  IO: String.raw`
static int f32_text(char* buf, f32 v) {
  int n = 0;
  int p = 0;
  if (v != v) {
    return sprintf(buf, "nan");
  }
  for (; p < 9; p += 1) {
    n = snprintf(buf, 40, "%.*e", p, (double)v);
    if (strtof(buf, NULL) == v) {
      break;
    }
  }
  char* ep = strchr(buf, 'e');
  if (ep == NULL) {
    return n;
  }
  int ex = atoi(ep + 1);
  if (ex >= 21 || ex <= -7) {
    n = (int)(ep - buf) + sprintf(ep, "e%c%d", ex < 0 ? '-' : '+', abs(ex));
  } else if (ex <= p) {
    n = snprintf(buf, 40, "%.*f", p - ex, (double)v);
  } else {
    int s = *buf == '-';
    memmove(buf + s + 1, buf + s + 2, p);
    memset(buf + s + 1 + p, '0', ex - p);
    n = s + 1 + ex;
  }
  return n;
}

static Term f32_show(Env e, Term x) {
  char buf[40];
  return io_str(e, buf, f32_text(buf, f32_unbox(x)));
}

// F32.read/F64.read accept one grammar on both lanes, over the whole
// extent: ASCII space, a sign, digits with a point and an exponent, or
// inf/infinity/nan, any case. A NUL is a character, not an end, and the
// strtod extras (hex floats, nan(...)) are not spellings.
static bool io_word(const char* p, u64 n, const char* w) {
  for (u64 i = 0; i < n; i++) {
    if ((p[i] | 32) != w[i]) { return false; }
  }
  return w[n] == 0;
}

static bool io_num(const char* p, u64 n) {
  u64 i = 0, d = 0;
  while (i < n && (p[i] == ' ' || (p[i] >= 9 && p[i] <= 13))) { i++; }
  if (i < n && (p[i] == '+' || p[i] == '-')) { i++; }
  if (io_word(p + i, n - i, "inf") || io_word(p + i, n - i, "infinity")
    || io_word(p + i, n - i, "nan")) {
    return true;
  }
  while (i < n && p[i] >= '0' && p[i] <= '9') { i++; d++; }
  if (i < n && p[i] == '.') {
    i++;
    while (i < n && p[i] >= '0' && p[i] <= '9') { i++; d++; }
  }
  if (d == 0) { return false; }
  if (i < n && (p[i] == 'e' || p[i] == 'E')) {
    i++;
    if (i < n && (p[i] == '+' || p[i] == '-')) { i++; }
    d = 0;
    while (i < n && p[i] >= '0' && p[i] <= '9') { i++; d++; }
    if (d == 0) { return false; }
  }
  return i == n;
}

static Term f32_read(Env e, Term s) {
  u64 n = 0;
  char* text = io_cstr(e, s, &n);
  char* end;
  f32 v = strtof(text, &end);
  Term out = n > 0 && (u64)(end - text) == n && strpbrk(text, "xX(") == NULL
    ? io_box(e, CID_SOME, f32_rewrap(v)) : term_pak(CID_NONE, 0);
  free(text);
  return out;
}

static int f64_text(char* buf, f64 v) {
  int n = 0;
  int p = 0;
  if (v != v) {
    return sprintf(buf, "nan");
  }
  for (; p < 17; p += 1) {
    n = snprintf(buf, 40, "%.*e", p, (double)v);
    if (strtod(buf, NULL) == v) {
      break;
    }
  }
  char* ep = strchr(buf, 'e');
  if (ep == NULL) {
    return n;
  }
  int ex = atoi(ep + 1);
  if (ex >= 21 || ex <= -7) {
    n = (int)(ep - buf) + sprintf(ep, "e%c%d", ex < 0 ? '-' : '+', abs(ex));
  } else if (ex <= p) {
    n = snprintf(buf, 40, "%.*f", p - ex, (double)v);
  } else {
    int s = *buf == '-';
    memmove(buf + s + 1, buf + s + 2, p);
    memset(buf + s + 1 + p, '0', ex - p);
    n = s + 1 + ex;
  }
  return n;
}

static Term f64_show(Env e, Term x) {
  char buf[48];
  return io_str(e, buf, f64_text(buf, f64_unbox(x)));
}

static Term f64_read(Env e, Term s) {
  u64 n = 0;
  char* text = io_cstr(e, s, &n);
  Term out = io_num(text, n) ? io_box(e, CID_SOME, x64_box(e, f64_rewrap(strtod(text, NULL))))
    : term_pak(CID_NONE, 0);
  free(text);
  return out;
}
`.slice(1),
  JS: String.raw`
function word_to_u32(w) {
  let x = 0;
  for (let i = 0; w.$ === "WCon"; i++) {
    x |= Number(w.head) << i;
    w = w.tail;
  }
  return x >>> 0;
}

function u32_to_word(x) {
  let w = {$: "WNil"};
  for (let i = 31; i >= 0; i--) {
    w = {$: "WCon", head: ((x >>> i) & 1) === 1, tail: w};
  }
  return w;
}

function cmp_new(a, b) {
  return {$: a < b ? "LT"
    : a === b ? "EQ" : "GT"};
}

function nat_divmod(a, b) {
  return b === 0n ? {$: "Tuple", fst: 0n, snd: a}
    : {$: "Tuple", fst: a / b, snd: a % b};
}

function nat_chk(n) {
  if (n > 281474976710655n) {
    throw "bend: ${ERRS[5]}";
  }
  return n;
}

function f32_show(x) {
  if (x !== x) {
    return "nan";
  }
  if (!Number.isFinite(x) || Object.is(x, -0)) {
    return x < 0 ? "-inf"
      : x === 0 ? "-0" : "inf";
  }
  let s = "x";
  for (let p = 1; p <= 9 && Math.fround(Number(s)) !== x; p += 1) {
    s = String(Number(x.toExponential(p - 1)));
  }
  return s;
}

function f32_bits(x) {
  return new Uint32Array(new Float32Array([x]).buffer)[0];
}

function f32_from_bits(u) {
  return new Float32Array(new Uint32Array([u]).buffer)[0];
}

function f32_read(s) {
  const re = /^\s*[+-]?((\d+\.?\d*|\.\d+)(e[+-]?\d+)?|inf(inity)?|nan)$/i;
  const v = Number(s.replace(/inf\w*/i, "Infinity"));
  return re.test(s) ? {$: "Some", value: Math.fround(v)} : {$: "None"};
}

function word_to_u64(w) {
  let x = 0n;
  for (let i = 0; w.$ === "WCon"; i++) {
    if (w.head) { x |= 1n << BigInt(i); }
    w = w.tail;
  }
  return x;
}

function u64_to_word(x) {
  let w = {$: "WNil"};
  for (let i = 63; i >= 0; i--) {
    w = {$: "WCon", head: ((x >> BigInt(i)) & 1n) === 1n, tail: w};
  }
  return w;
}

function f64_bits(x) {
  const d = new DataView(new ArrayBuffer(8));
  d.setFloat64(0, x);
  return d.getBigUint64(0);
}

function f64_from_bits(u) {
  const d = new DataView(new ArrayBuffer(8));
  d.setBigUint64(0, u);
  return d.getFloat64(0);
}

function f64_show(x) {
  if (x !== x) {
    return "nan";
  }
  if (!Number.isFinite(x) || Object.is(x, -0)) {
    return x < 0 ? "-inf"
      : x === 0 ? "-0" : "inf";
  }
  let s = "x";
  for (let p = 1; p <= 17 && Number(s) !== x; p += 1) {
    s = String(Number(x.toExponential(p - 1)));
  }
  return s;
}

function f64_read(s) {
  const re = /^\s*[+-]?((\d+\.?\d*|\.\d+)(e[+-]?\d+)?|inf(inity)?|nan)$/i;
  const v = Number(s.replace(/inf\w*/i, "Infinity"));
  return re.test(s) ? {$: "Some", value: v} : {$: "None"};
}

// Scalar strings stay primitive; raw U32 Char values dispatch through char_code.
// The string boundary rejects raw values consistently, including SCon/from_list.
function char_code(c) { return typeof c === "string" ? c.codePointAt(0) : c.code; }
function str_prepend(c, s) {
  if (typeof c !== "string") { throw "bend: JS strings cannot contain non-scalar Char values"; }
  return c + s;
}
// A string with no surrogate is its own code points: a position is an
// index. The last string asked about is remembered, so a scanner reading
// one buffer at many positions tests it once.
let STR_FLAT_S = "", STR_FLAT = true;
function str_flat(s) {
  if (s !== STR_FLAT_S) { STR_FLAT_S = s; STR_FLAT = !/[\uD800-\uDFFF]/.test(s); }
  return STR_FLAT;
}
function str_length(s) {
  if (str_flat(s)) { return BigInt(s.length); }
  let n = 0n; for (const c of s) { n++; } return n;
}
function str_offset(s, n) {
  if (n >= 32n && str_flat(s)) { return n < BigInt(s.length) ? Number(n) : s.length; }
  let i = 0;
  while (n > 0n && i < s.length) { i += s.codePointAt(i) > 0xffff ? 2 : 1; n--; }
  return i;
}
function bytes_get(s, i) {
  const j = str_offset(s, i);
  return j < s.length ? s.codePointAt(j) : 256;
}
function bytes_word_le(s, i) {
  let j = str_offset(s, i), x = 0;
  for (let k = 0; k < 4 && j < s.length; k++) {
    const c = s.codePointAt(j);
    x |= (c & 255) << (8 * k);
    j += c > 0xffff ? 2 : 1;
  }
  return x >>> 0;
}
function bytes_push(s, c) { return s + str_prepend(char_new(c), ""); }
function bytes_span(s, i, lo, hi) {
  let j = str_offset(s, i), n = 0;
  while (j < s.length) {
    const c = s.codePointAt(j);
    if (c < lo || c >= hi) { break; }
    j += c > 0xffff ? 2 : 1; n++;
  }
  return BigInt(n);
}
function bytes_find_any(s, i, w, x, y, z) {
  let j = str_offset(s, i), n = 0;
  while (j < s.length) {
    const c = s.codePointAt(j);
    if (c === w || c === x || c === y || c === z) { break; }
    j += c > 0xffff ? 2 : 1; n++;
  }
  return i + BigInt(n);
}
function bytes_find_byte(s, i, x) {
  let j = str_offset(s, i), n = 0;
  if (str_flat(s) && x < 0xd800) {
    const k = s.indexOf(String.fromCharCode(x), j);
    return i + BigInt((k < 0 ? s.length : k) - j);
  }
  while (j < s.length) {
    const c = s.codePointAt(j);
    if (c === x) { break; }
    j += c > 0xffff ? 2 : 1; n++;
  }
  return i + BigInt(n);
}
function str_slice(s, lo, hi) {
  if (hi <= lo) { return ""; }
  const a = str_offset(s, lo);
  return s.slice(a, a + str_offset(s.slice(a), hi - lo));
}
function str_get(s, n) {
  const i = str_offset(s, n);
  return i === s.length ? {$: "None"} : {$: "Some", value: String.fromCodePoint(s.codePointAt(i))};
}
function str_end(s, n, take) {
  const len = str_length(s), at = len > n ? len - n : 0n;
  return take ? s.slice(str_offset(s, at)) : s.slice(0, str_offset(s, at));
}
function str_get_end(s, n) {
  const len = str_length(s);
  return n === 0n || n > len ? {$: "None"} : str_get(s, len - n);
}
function map_bit(s, pos) {
  const i = str_offset(s, pos / 33n), off = Number(pos % 33n);
  const b = i < s.length
    && (off === 0 || ((s.codePointAt(i) >>> (32 - off)) & 1) === 1);
  return {$: "Tuple", fst: s, snd: b};
}
function str_order(a, b) {
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    const x = a.codePointAt(i), y = b.codePointAt(j);
    if (x !== y) { return {$: x < y ? "LT" : "GT"}; }
    i += x > 0xffff ? 2 : 1; j += y > 0xffff ? 2 : 1;
  }
  return {$: i < a.length ? "GT" : j < b.length ? "LT" : "EQ"};
}
function str_cmp(a, b) { return {$: "Tuple", fst: {$: "Tuple", fst: a, snd: b}, snd: str_order(a, b)}; }
function str_list(xs) {
  let out = {$: "Nil"};
  for (let i = xs.length - 1; i >= 0; i--) { out = {$: "Con", head: xs[i], tail: out}; }
  return out;
}
function str_from_list(xs) {
  const out = [];
  for (; xs.$ === "Con"; xs = xs.tail) { out.push(str_prepend(xs.head, "")); }
  return out.join("");
}
function str_join(xs, sep) {
  const out = [];
  for (; xs.$ === "Con"; xs = xs.tail) { out.push(xs.head); }
  return out.join(sep);
}
function str_repeat(s, n) {
  const len = str_length(s);
  if (len === 0n) { return ""; }
  if (n > 2147483648n / len) { throw "bend: a string past the maximum length 2^31"; }
  return s.repeat(Number(n));
}
function str_split(s, c) {
  // A raw separator cannot equal any scalar string element.
  return str_list(typeof c === "string" ? s.split(c) : [s]);
}

// Strings contain scalars only, so a nonempty scalar needle cannot match
// inside a surrogate pair. Native UTF-16 search is exact; convert only the
// final position to code points. No repeated positional get/offset scans.
function str_find(s, p, last) {
  const i = last ? s.lastIndexOf(p) : s.indexOf(p);
  return i < 0 ? {$: "None"} : {$: "Some", value: str_length(s.slice(0, i))};
}
function str_count(s, p) {
  if (p === "") { return str_length(s) + 1n; }
  let n = 0n, at = 0;
  for (;;) {
    const i = s.indexOf(p, at);
    if (i < 0) { return n; }
    n++; at = i + p.length;
  }
}
function str_replace(s, old, value) {
  // A string separator and array join treat $&, $1, etc. literally.
  if (old !== "") { return s.split(old).join(value); }
  return s === "" ? value : value + [...s].join(value) + value;
}
function str_partition(s, sep) {
  const i = sep === "" ? -1 : s.indexOf(sep);
  return {$: "Tuple", fst: i < 0 ? s : s.slice(0, i),
    snd: {$: "Tuple", fst: i < 0 ? "" : sep, snd: i < 0 ? "" : s.slice(i + sep.length)}};
}
function str_splitlines(s) {
  const lines = s.split(/\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/);
  if (lines[lines.length - 1] === "") { lines.pop(); }
  return str_list(lines);
}
function str_capitalize(s) {
  return s.replace(/[a-zA-Z]/g, (c, i) => i === 0 ? c.toUpperCase() : c.toLowerCase());
}
function str_pad(s, width, c, mode) {
  const n = str_length(s);
  if (width <= n) { return s; }
  if (width > 2147483648n) { throw "bend: a string past the maximum length 2^31"; }
  const pad = str_prepend(c, "").repeat(Number(width - n));
  if (mode === 1) { return s + pad; }
  if (mode === 2 && (s[0] === "+" || s[0] === "-")) { return s[0] + pad + s.slice(1); }
  return pad + s;
}
// The C VM over Int32Array: three ints per inst, the set ranges in pairs, a
// slot -1 unset, a dead pc or slot -1. A JS string indexes UTF-16 units, so
// at costs one prefix scan; the walk itself goes by codePointAt.
function re_exec(re, s, at, anchored) {
  const insts = [];
  for (let t = re.prog; t.$ === "Con"; t = t.tail) { insts.push(t.head); }
  const m = insts.length, ns = 2 * (1 + Number(re.ngroups)), sets = [];
  if (!(m < 16777215 && ns <= 8388608)) { throw "bend: a regex past the maximum size"; }
  const arg = (n, lim) => n < lim ? Number(n) : -1;
  const P = new Int32Array(3 * m);
  insts.forEach((h, i) => {
    let op = 10, a = 0, b = 0;
    switch (h.$) {
      case "IChr": op = 0; a = h.c | 0; break;
      case "IAny": op = 1; break;
      case "ISet":
        op = h.neg ? 3 : 2; a = sets.length >> 1;
        for (let r = h.rs; r.$ === "Con"; r = r.tail) { sets.push(r.head.fst, r.head.snd); }
        b = (sets.length >> 1) - a; break;
      case "ISplit": op = 4; a = arg(h.x, m); b = arg(h.y, m); break;
      case "IJmp": op = 5; a = arg(h.x, m); break;
      case "ISave": op = 6; a = arg(h.slot, ns); break;
      case "IBol": op = 7; a = h.multi ? 1 : 0; break;
      case "IEol": op = 8; a = h.multi ? 1 : 0; break;
      case "IWordB": op = 9; a = h.neg ? 1 : 0; break;
    }
    P[3 * i] = op; P[3 * i + 1] = a; P[3 * i + 2] = b;
  });
  const S = Int32Array.from(sets), V = new Int32Array(m);
  const RP = new Int32Array(m), RS = new Int32Array(m * ns);
  const CP = new Int32Array(m), CS = new Int32Array(m * ns);
  const CUR = new Int32Array(ns), BEST = new Int32Array(ns), STK = new Int32Array(2 * m + 2);
  const word = (c) => (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || c === 95 || (c >= 97 && c <= 122);
  let i = 0, pos = 0, prev = -1;
  for (; at > 0n && i < s.length; at--, pos++) { prev = s.codePointAt(i); i += prev > 0xffff ? 2 : 1; }
  let nraw = 0, best = false;
  for (const at0 = pos; ; pos++) {
    const fresh = !best && (pos === at0 || !anchored) ? 1 : 0, gen = pos - at0 + 1;
    if (nraw + fresh === 0) { break; }
    const c = i < s.length ? s.codePointAt(i) : -1;
    i += c > 0xffff ? 2 : 1;
    let ncl = 0;
    for (let r = 0; r < nraw + fresh; r++) {
      if (r < nraw) { CUR.set(RS.subarray(r * ns, (r + 1) * ns)); } else { CUR.fill(-1); }
      let sp = 1;
      STK[0] = r < nraw ? RP[r] : 0;
      while (sp) {
        const x = STK[--sp];
        if (x < -1) { CUR[x & 0x7fffffff] = STK[--sp]; continue; }
        if (x < 0 || x >= m || V[x] === gen) { continue; }
        V[x] = gen;
        const op = P[3 * x], a = P[3 * x + 1], b = P[3 * x + 2];
        let go = false;
        if (op === 4) { STK[sp++] = b; STK[sp++] = a; }
        else if (op === 5) { STK[sp++] = a; }
        else if (op === 6) {
          if (a >= 0) { STK[sp++] = CUR[a]; STK[sp++] = a | -0x80000000; CUR[a] = pos; go = true; }
        }
        else if (op === 7) { go = pos === 0 || (a === 1 && prev === 10); }
        else if (op === 8) { go = c < 0 || (c === 10 && (a === 1 || i >= s.length)); }
        else if (op === 9) { go = (pos > 0 || s.length > 0) && (a === 1) !== (word(prev) !== word(c)); }
        else { CP[ncl] = x; CS.set(CUR, ncl++ * ns); }
        if (go) { STK[sp++] = x + 1; }
      }
    }
    nraw = 0;
    for (let t = 0; t < ncl; t++) {
      const x = CP[t], op = P[3 * x], a = P[3 * x + 1], b = P[3 * x + 2];
      if (op === 10) { BEST.set(CS.subarray(t * ns, (t + 1) * ns)); best = true; break; }
      let eat = op === 0 ? a === c : op === 1 ? c !== 10 : false;
      if (op === 2 || op === 3) {
        for (let j = a; j < a + b && !eat; j++) { eat = (S[2 * j] >>> 0) <= c && c <= (S[2 * j + 1] >>> 0); }
        eat = eat !== (op === 3);
      }
      if (eat && c >= 0) { RP[nraw] = x + 1; RS.set(CS.subarray(t * ns, (t + 1) * ns), nraw++ * ns); }
    }
    prev = c;
    if (c < 0) { break; }
  }
  if (!best || BEST[0] < 0 || BEST[1] < 0) { return {$: "None"}; }
  let gs = {$: "Nil"};
  for (let g = ns - 2; g >= 2; g -= 2) {
    const f = BEST[g] < 0 || BEST[g + 1] < 0 ? {$: "None"}
      : {$: "Some", value: {$: "Tuple", fst: BigInt(BEST[g]), snd: BigInt(BEST[g + 1])}};
    gs = {$: "Con", head: f, tail: gs};
  }
  return {$: "Some", value: {$: "Match", start: BigInt(BEST[0]), end: BigInt(BEST[1]), groups: gs}};
}
function str_hash(s) {
  let h = 2166136261;
  for (const c of s) {
    const x = c.codePointAt(0);
    for (let shift = 0; shift < 32; shift += 8) { h = Math.imul(h ^ ((x >>> shift) & 255), 16777619) >>> 0; }
  }
  return h;
}

function char_new(code) {
  if (code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) {
    return {$: "RawChar", code: code};
  }
  return String.fromCodePoint(code);
}
`.slice(1),
};

// Caches
// ------

const PROBES: Probe[] = [];

const DUMMY = probe("~");

const OPENS: Map<Of<"Lam"> | HLet, { ps: Probe[]; b: HTerm }> = new Map();

const USES: Map<HTerm, UMap> = new Map();

const TELES: Map<HTerm, ReturnType<typeof Bend.tele_unbind>> = new Map();

const SRCS: Map<Name, Src> = new Map();

const LOCAL: Map<Name, string> = new Map();

const FOLDS: Map<HTerm, HTerm | null> = new Map();

const FLATS: Map<Name, boolean> = new Map();

const SIGS: Map<Name, Sig> = new Map();

const BRWS: Map<Name, boolean[]> = new Map();

const SPINES: Map<HTerm, Spine> = new Map();

const NODES: Map<Name, Lay> = new Map();

const LAYS: Map<string, Lay> = new Map();

const CYCLES: Map<Name, boolean> = new Map();

const CONSTS: Map<HTerm, boolean> = new Map();

const LITS = new Map<HTerm, HTerm>();

// Name
// ====

function name_own(k: Name, tld: { T: HTerm }, head: string): string {
  const segs = k.split(".");
  return segs.map((_, i) => segs.slice(i).join(".")).find((own) =>
    new RegExp("^" + head + own.replace(/\./g, "\\.") + "[(:<{\\s]", "m")
      .test(tld.T.s?.src ?? "")) ?? k;
}

function name_clean(k: string): string {
  return (LOCAL.get(k) ?? k).replace(/[^A-Za-z0-9_]/g, "_");
}

function name_local(fl: File, k: Name): string {
  const base = name_clean(k).replace(/^_+/, "");
  const n = fl.fresh.get(base) ?? 0;
  fl.fresh.set(base, n + 1);
  return "_" + base + "_" + n;
}

// Die
// ===

function die(m: string): never {
  throw new Error(m);
}

// Tpl
// ===

function tpl_ops(pre: string, names: string, C: string, JS: string, call = false):
  Record<string, Intr> {
  const out: Record<string, Intr> = {};
  for (const p of names.split(" ")) {
    const [k, o = k, jo = o] = p.split(":");
    out[pre + k] = { C: C.replaceAll("$o", o), JS: JS.replaceAll("$o", jo), call };
  }
  return out;
}

// The bit-level 64-bit integer ops, shared by U64 and I64: two's complement
// makes add, sub, mul, the bitwise ops and the logical shifts sign-agnostic,
// so only the comparisons, the negation and the arithmetic shift differ.
function tpl_w64(p: string): Record<string, Intr> {
  return {
    ...tpl_ops(p, "add:+ sub:- mul:* and:& or:| xor:^",
      "((u64)($0) $o (u64)($1))",
      "(($0 $o $1) & 0xFFFFFFFFFFFFFFFFn)"),
    ...tpl_ops(p, "inc:+ shl:<< shr:>>",
      "((u64)((u64)($0) $o 1))",
      "(($0 $o 1n) & 0xFFFFFFFFFFFFFFFFn)"),
    ...tpl_ops(p, "shln:<< shrn:>>",
      "($1 >= 64 ? 0 : ((u64)($0) $o $1))",
      "($1 >= 64n ? 0n : (($0 $o $1) & 0xFFFFFFFFFFFFFFFFn))"),
    [p + "not"]: {
      C:  "((u64)~(u64)($0))",
      JS: "(~$0 & 0xFFFFFFFFFFFFFFFFn)",
    },
    [p + "is_zero"]: {
      C:  "((u64)((u64)($0) == 0))",
      JS: "($0 === 0n)",
    },
  };
}

function tpl(t: Gen, xs: string[]): string {
  return typeof t !== "string" ? t(xs)
    : t.split(/\$(\d)/).map((p, i) => (i % 2 === 1 ? xs[+p] : p)).join("");
}

// Succ over a literal is the next literal; over a checked sum, one more.
function tpl_nat(u: string, f: string): Gen {
  return ([p]) => {
    if (/^\d/.test(p)) {
      return (BigInt(parseInt(p)) + 1n) + u;
    }
    if (/^nat_chk\(.* \+ \d+n?\)$/.test(p)) {
      return p.replace(/\d+(?=n?\)$)/, (k) => String(+k + 1));
    }
    return tpl(f, [p]);
  };
}

// Memo
// ====

function memo<K, V>(m: Map<K, V>, k: K, f: () => V): V {
  const got = m.get(k);
  if (got !== undefined) {
    return got;
  }
  const out = f();
  m.set(k, out);
  return out;
}

function memo_gc(): void {
  [OPENS, USES, FOLDS, SPINES, CONSTS, LITS].forEach((m) => m.clear());
}

// Probe
// =====

function probe(k: Name): Probe {
  const p = Bend.Var(k, PROBES.length) as Probe;
  PROBES.push(p);
  return p;
}

function probe_of(t: HTerm): Probe {
  return PROBES[(term_force(t) as Probe).i];
}

// Term
// ====

function term_force(t: HTerm): HTerm {
  const s = Bend.term_force(t);
  if (s.$ !== "Lit") {
    return s;
  }
  // a Nat literal past the cap is U32.to_nat of its word
  return memo(LITS, s, () => Bend.term_higher(
    typeof s.v === "number" && s.v > Bend.NAT_LITERAL_MAX
      ? Bend.App(Bend.Ref("U32.to_nat"), Bend.u32_to_term(s.v))
      : Bend.lit_full(s)));
}

function term_strip(t: HTerm): HTerm {
  return term_force(Bend.term_strip(t));
}

function term_open(t: Of<"Lam"> | HLet): { ps: Probe[]; b: HTerm } {
  return memo(OPENS, t, () => {
    const ps = (t.$ === "Lam" ? [t.k] : t.k).map(probe);
    return { ps, b: t.$ === "Lam" ? t.f(ps[0]) : t.f(ps) };
  });
}

// A let opened ahead: its body `b` over `ps` is never rebuilt.
function let_open(ps: Probe[], vs: HTerm[], b: HTerm): HLet {
  const l = Bend.Let(ps.map((p) => p.k), ps.map(() => 0), vs,
    () => die("a pre-opened let")) as HLet;
  OPENS.set(l, { ps, b });
  return l;
}

// A let's live binders: an unused (or erased) one dies with its value.
function let_live(cb: Carb, t: HLet): boolean[] {
  const o = term_open(t);
  const u = term_uses(cb, o.b);
  return t.q.map((_, j) => term_use(u, o.ps[j]) > 0);
}

// A term's application view and its call: the direct call when the live
// arguments meet the def's, else Clo.apply over the outermost live one.
function term_spine(cf: Carb, tm: HTerm): Spine {
  return memo(SPINES, tm, () => {
    const apps: Of<"App">[] = [];
    let h = tm;
    let c = term_force(tm);
    while (c.$ === "Ann" || c.$ === "App") {
      if (c.$ === "App") {
        apps.push(c);
        h = c.f;
      }
      c = term_force(c.$ === "App" ? c.f : c.x);
    }
    apps.reverse();
    const tld = c.$ === "Ref" ? def_body(cf, c.k) : undefined;
    const T = tld?.$ === "Def" ? tld.T : ty_ann(h);
    const qs = T === null ? [] : tele_unbind(cf.book, T).doms;
    const live = apps.map((_, i) => i >= qs.length || quant_live(qs[i][0]));
    const all = apps.map((a) => a.x);
    const args = all.filter((_, i) => live[i]);
    const def = c.$ === "Ref" && intr_of(cf, c.k) === undefined
      && (done_live(tld) || def_foreign(tld)) ? c.k : null;
    const need = def === null ? 0 : sig_def(cf, def).lays.length;
    const a = apps[live.lastIndexOf(true)];
    let call: Call | null = null;
    if (def !== null && args.length === need) {
      call = { k: def, args, all, bang: (c as Of<"Ref">).b };
    } else if (args.length > need && (def !== null || c.$ !== "Ref")) {
      call = { k: CLO_APPLY, args: [a.f, a.x], all: [a.f, a.x] };
    }
    return { h, t: c, all, args, tld, call };
  });
}

function term_eta(book: Bend.Book, t: HTerm, T: HTerm, n: number): HTerm {
  if (n === 0) {
    return t;
  }
  const all = ty_all(book, T);
  return Bend.Ann(Bend.Lam("x", 0, (y: HTerm) =>
    term_eta(book, Bend.App(t, y), all.B(y), n - 1)), T);
}

function term_kids(cf: Carb, tm: HTerm): HTerm[] {
  const t = term_force(tm);
  switch (t.$) {
    case "Ann": return [t.x];
    case "Lam": return [term_open(t).b];
    case "Let": {
      const on = let_live(cf, t);
      return [...t.v.filter((_, j) => on[j]), term_open(t).b];
    }
    case "App": {
      const m = term_spine(cf, t);
      return [m.h, ...m.args];
    }
    case "Ctr": return term_const(t) ? [] : ctr_flds(cf.book, t.k, t.x);
    case "Mat": return [t.h, t.m];
    case "Rwt": return [t.f];
    default: return [];
  }
}

// Does `p` hold at a node of `t`? Each node once, told if it is in tail
// position (under annotations, binders, arms and let bodies).
function term_any(cf: Carb, t: HTerm, p: (s: HTerm, tail: boolean) => boolean,
  tail = true, seen: Set<HTerm> = new Set()): boolean {
  const s = term_force(t);
  if (seen.has(s)) {
    return false;
  }
  seen.add(s);
  const kids = term_kids(cf, s);
  return p(s, tail) || kids.some((x, i) => term_any(cf, x, p, tail
    && (s.$ === "Let" ? i === kids.length - 1 : "Ann Lam Mat Rwt".includes(s.$)),
  seen));
}

// A term's node count, shared parts once: fold fuel is the size an unfold
// adds, so a literal-bounded loop does not unroll into its caller.
function term_nodes(cf: Carb, t: HTerm): number {
  let n = 0;
  term_any(cf, t, () => (n += 1) < 0);
  return n;
}

function term_const(t: HTerm): boolean {
  const s = term_strip(t);
  return s.$ === "Ctr" && memo(CONSTS, s, () => s.x.every(term_const));
}

function term_use(u: UMap, p: Probe): number {
  return Bend.pmap_get(u, p.i) ?? 0;
}

// The uses of each probe in a term.
function term_uses(cb: Carb, tm: HTerm): UMap {
  return memo(USES, tm, () => {
    const t = term_force(tm);
    switch (t.$) {
      case "Var": {
        const p = probe_of(t);
        return p === DUMMY ? USE0 : Bend.pmap_set(USE0, p.i, 1);
      }
      case "Mat": return Bend.pmap_union(term_uses(cb, t.h),
        term_uses(cb, t.m), Math.max);
      default: return term_kids(cb, t).reduce((u, x) =>
        Bend.pmap_union(u, term_uses(cb, x), (a, b) => a + b), USE0);
    }
  });
}

function rest_use(cb: Carb, rest: HTerm[], p: Probe): number {
  return rest.reduce((n, r) => n + term_use(term_uses(cb, r), p), 0);
}

// Live
// ====

function live_dom([q]: Dom): boolean {
  return quant_live(q);
}

// Intr
// ====

function intr_of(c: Carb, k: Name, js = false): Intr | undefined {
  const tld = c.book.tlds[k];
  const it = tld?.$ === "Def" && tld.i === undefined && (tld.b || tld.v === null)
    ? OPERATIONS[eff_name(k)] : undefined;
  return it !== undefined && (js || it.C !== undefined || it.call === true)
    ? it : undefined;
}

// Call
// ====

function call_kind(c: Carb, t: HTerm): Call | null {
  return term_spine(c, t).call;
}

// A partial application of a def is its eta-expansion: a closure.
function call_eta(cb: Carb, t: HTerm): HTerm | null {
  const m = term_spine(cb, t);
  const d = m.tld;
  if (m.t.$ !== "Ref" || d?.$ !== "Def"
    || m.args.length >= sig_def(cb, m.t.k).live.length) {
    return null;
  }
  return term_eta(cb.book, t, Bend.tele_fill(cb.book, d.T, m.all,
    Bend.ctx_nil()), d.n - m.all.length);
}

// Tele
// ====

function tele_unbind(book: Bend.Book,
  T: HTerm): ReturnType<typeof Bend.tele_unbind> {
  return memo(TELES, T, () => Bend.tele_unbind(book, T));
}

// Ty
// ==

function ty_ann(t: HTerm): HTerm | null {
  const v = term_force(t);
  return v.$ === "Ann" ? v.T : null;
}

function ty_wnf(book: Bend.Book, ty: HTerm | null): HTerm | null {
  return ty && Bend.term_wnf(book, ty);
}

function ty_all(book: Bend.Book, ty: HTerm | null): HAll {
  return Bend.tele_open(book, ty!)!;
}

function ty_peel(tm: HTerm,
  ty: HTerm | null): [HTerm, HTerm | null] {
  let x = term_force(tm);
  while (x.$ === "Ann" || x.$ === "Rwt") {
    ty = x.$ === "Ann" ? x.T : ty;
    x = term_force(x.$ === "Ann" ? x.x : x.f);
  }
  return [x, ty];
}

function ty_adt(book: Bend.Book, A: HTerm | null): HAdt | null {
  const t = ty_wnf(book, A);
  return t?.$ === "ADT" ? t : null;
}

// A type may hold a closure: a function, variable or stuck type, or a
// datatype with such a live field; a word, quantity or kind never does.
function ty_clo(book: Bend.Book, A: HTerm | null,
  seen = new Set<Name>()): boolean {
  const t = ty_wnf(book, A);
  switch (t?.$) {
    case "ADT": {
      if (WORDS[t.k] !== undefined) {
        return false;
      }
      if (t.x.some((x) => ty_clo(book, x, seen))) {
        return true;
      }
      const tld = book.tlds[t.k];
      if (tld?.$ !== "ADT" || seen.has(t.k)) {
        return false;
      }
      seen.add(t.k);
      return tld.c.some((c) =>
        ctr_doms(book, c, t.x).some((f) => ty_clo(book, f, seen)));
    }
    case "Typ": case "Qua": case "Min": case "Eql": return false;
    default: return true;
  }
}

// Lay
// ===

// Boxes: an Array, IO.OP, a recursive datatype, and a field re-entering the
// datatype under layout (a family hid the cycle).
function lay_of(book: Bend.Book, A: HTerm | null): Lay {
  const t = ty_adt(book, A);
  if (t === null) {
    return BOX;
  }
  const key = Bend.term_key(Bend.term_lower(t));
  return WORDS[t.k] ?? memo(LAYS, key, () => {
    const tld = book.tlds[t.k];
    if (t.k === "Array" || t.k === "IO.OP" || tld?.$ !== "ADT"
      || lay_cyclic(book, t.k)) {
      return BOX;
    }
    LAYS.set(key, BOX);
    return lay_pack(tld.c.map((c) =>
      [c.k, lay_wide(ctr_doms(book, c, t.x).map((A) => lay_of(book, A)))]));
  });
}

// An Array cell is its element datatype's open layout (its constructors'
// return type), so every caller agrees.
function lay_el(book: Bend.Book, A: HTerm | null): Lay {
  const t = ty_adt(book, A) ?? die("an open Array element type");
  const tld = book.tlds[t.k];
  return lay_of(book, tld?.$ === "ADT" && tld.c[0]
    ? tele_unbind(book, tld.c[0].T).ret : A);
}

// Past WIDE words in total, the multi-word layouts go boxed.
function lay_wide(lays: Lay[]): Lay[] {
  return lays.flatMap((l) => l.ks).length > WIDE
    ? lays.map((l) => l.ks.length > 1 ? BOX : l) : lays;
}

// Fields start after the tag; the packer owns their final offsets. A
// full word (X64) sharing a slot with another arm's box would be dropped as
// a Term when that slot is: its field is boxed instead, until none does.
function lay_pack(arms: [Bend.Name, Lay[]][]): Lay {
  for (;;) {
    const lay = lay_pack_at(arms);
    const full = lay_full(lay);
    const bad = full.map((f, j) => f && lay.ks[j] === "box");
    if (!bad.includes(true)) {
      return lay;
    }
    arms = lay.arms!.map((a) => [a.k, a.fs.map((f) => lay_full(f.lay)
      .some((b, j) => b && bad[f.at + j]) ? BOX : f.lay)]);
  }
}

// The words of a layout that may hold a full 64-bit word.
function lay_full(lay: Lay): boolean[] {
  if (lay === X64) {
    return [true];
  }
  const out = lay.ks.map(() => false);
  lay.arms?.forEach((a) => a.fs.forEach((f) => lay_full(f.lay)
    .forEach((b, j) => b && (out[f.at + j] = true))));
  return out;
}

function lay_pack_at(arms: [Bend.Name, Lay[]][]): Lay {
  const tag = arms.length > 1 ? 1 : 0;
  const ks: Kind[] = tag === 1 ? ["w32"] : [];
  return { ks, arms: arms.map(([k, lays]) => {
    let at = tag;
    const fs = lays.map((lay) => {
      const f = { at, lay };
      for (const k of lay.ks) {
        const old = ks[at] ?? "w32";
        ks[at++] = old === "box" || k === "box" ? "box"
          : old === "w64" || k === "w64" ? "w64" : "w32";
      }
      return f;
    });
    return { k, fs };
  }) };
}

function lay_cyclic(book: Bend.Book, k: Name): boolean {
  const seen = new Set<Name>();
  const hits = (A: HTerm): boolean => {
    const a = ty_adt(book, A);
    if (a === null) {
      return false;
    }
    if (a.k === k || a.x.some(hits)) {
      return true;
    }
    if (seen.has(a.k)) {
      return false;
    }
    seen.add(a.k);
    return walk(a.k);
  };
  const walk = (d: Name): boolean => {
    const tld = book.tlds[d];
    return tld?.$ === "ADT" && WORDS[d] === undefined && d !== "Array"
      && tld.c.some((c) => ctr_doms(book, c).some(hits));
  };
  return memo(CYCLES, k, () => walk(k));
}

function lay_node(book: Bend.Book, k: Name): Lay {
  return memo(NODES, k, () => lay_pack([[k, lay_wide((book.ctrs[k]
    ? ctr_doms(book, book.ctrs[k]) : []).map((A) => lay_of(book, A)))]]));
}

function lay_eq(a: Lay, b: Lay): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

function lay_c(k: Kind): string {
  return k === "w32" ? "u32" : "Term";
}

function lay_box(lay: Lay): boolean {
  return lay.arms === null && lay.ks[0] === "box";
}

function lay_arm(lay: Lay, k: Name): Arm {
  return lay.arms!.find((a) => a.k === k)!;
}

function lay_arr(lay: Lay): { arr: boolean; lgs: number } {
  return { arr: lay.ks.some((k) => k !== "w32"),
    lgs: cls_fit(Math.max(1, lay.ks.length)) };
}

// Ctr
// ===

function ctr_adt(fl: File, x: Of<"Ctr">,
  ty: HTerm | null): [HAdt, number | null] {
  const ctr = fl.book.ctrs[x.k];
  const adt = adt_of(fl.book, ty ?? (ctr ? tele_unbind(fl.book, ctr.T).ret
    : null));
  if (ty === null && adt.x.length > 0) {
    die("a constructor outside a datatype");
  }
  return [adt, WORDS[adt.k] === W32 ? Bend.u32_from_term(x, adt.k) : null];
}

// A constructor's fields: the last n domains of its type, over the
// datatype's arguments `xs` when given.
function ctr_tail(book: Bend.Book, ctr: Bend.Ctr, xs?: HTerm[]): Dom[] {
  const doms = tele_unbind(book, xs === undefined ? ctr.T
    : Bend.tele_fill(book, ctr.T, xs, Bend.ctx_nil())).doms;
  return doms.slice(doms.length - ctr.n);
}

function ctr_doms(book: Bend.Book, ctr: Bend.Ctr, xs?: HTerm[]): HTerm[] {
  return ctr_tail(book, ctr, xs).filter(live_dom).map(([, , A]) => A);
}

function ctr_flds(book: Bend.Book, k: Name,
  xs: HTerm[]): HTerm[] {
  const ds = book.ctrs[k] ? ctr_tail(book, book.ctrs[k]) : [];
  return xs.filter((_, j) => ds[j] === undefined || live_dom(ds[j]));
}

function ctr_build(fl: File, k: Name, exprs: string[],
  stat = false): string {
  if (k === "SCon") {
    return `str_prepend_take(e, ${exprs[0]}, ${exprs[1]})`;
  }
  const cid = cid_mac(k);
  const node = lay_node(fl.book, k);
  if (exprs.length === 0 || (node.ks.length === 1 && node.ks[0] === "w32")) {
    return `term_pak(${cid}, ${exprs[0] ?? 0})`;
  }
  if (stat) {
    fl.stat.add(k);
    const at = memo(fl.lits, exprs.join(", "), () =>
      fl.img.push(...exprs) - exprs.length);
    return `term_ctr(${cid}, STAT_OFF + ${at})`;
  }
  const alloc = `heap_alloc(e, cls_fit(${exprs.length}))`;
  const at = fl.spares.findIndex((s) =>
    cls_fit(s.words) === cls_fit(exprs.length));
  const s = at < 0 ? null : fl.spares.splice(at, 1)[0];
  const got = s === null ? alloc
    : s.z ? `${s.name} >= HEAP_OFF ? ${s.name} : ${alloc}` : s.name;
  return `term_ctr(${cid}, ${node_fill(fl, "nd", got, exprs,
    fl.hot.has(k))})`;
}

// Mat
// ===

function adt_of(book: Bend.Book, A: HTerm | null): HAdt {
  const adt = ty_adt(book, A)!;
  // an Array's element layout, which refuses an open element type
  if (adt.k === "Array") {
    lay_el(book, adt.x[0]);
  }
  return adt;
}

function mat_head(t: HTerm): boolean {
  return t.$ === "Mat" || t.$ === "Efq";
}

// A function value: a match, or a lambda over a live binder.
function fun_live(book: Bend.Book, x: HTerm, ty: HTerm | null): boolean {
  return mat_head(x) || (x.$ === "Lam"
    && quant_live(ty_all(book, ty).q));
}

function mat_arms(t: HTerm): { arms: [Name, HTerm][]; end: HTerm } {
  const arms: [Name, HTerm][] = [];
  let cur = t;
  for (let m = term_strip(cur); m.$ === "Mat"; m = term_strip(cur)) {
    arms.push([m.k, m.h]);
    cur = m.m;
  }
  return { arms, end: cur };
}

// Quant
// =====

function quant_live(q: Bend.Quant): boolean {
  return q.$ !== "None";
}

// Def
// ===

// A def's live parameters, the layouts a call passes and its return layout
// (boxes for a foreign def and Clo.apply).
function sig_def(cb: Carb, k: Name): Sig {
  return memo(SIGS, k, () => {
    const tld = def_body(cb, k);
    if (tld?.$ !== "Def") {
      return { live: [], lays: [BOX, BOX], ret: BOX };
    }
    const doms = tele_unbind(cb.book, tld.T).doms;
    const live = doms.slice(0, tld.n).filter(live_dom);
    const lays = live.map(([, , A]) => lay_of(cb.book, A));
    if (def_foreign(tld)) {
      return { live, lays: [...lays.map(() => BOX), BOX], ret: BOX };
    }
    const ret = lay_of(cb.book, Bend.tele_fill(cb.book, tld.T,
      Array(tld.n).fill(DUMMY), Bend.ctx_nil()));
    return { live, lays: lay_wide(lays), ret: ret.ks.length === 0 ? BOX : ret };
  });
}

// A def's borrowed parameters (a box, not an Array, not owned), fixed a pass.
function brw_of(cb: Carb, k: Name): boolean[] {
  return memo(BRWS, k, () => {
    const { live, lays } = sig_def(cb, k);
    return lays.map((l, i) => done_live(def_body(cb, k)) && l.ks.includes("box")
      && ty_adt(cb.book, live[i][2])?.k !== "Array" && !cb.own.has(k + "~" + i));
  });
}

// Lambdas a body opens past its `left` parameters, under every arm (an arm
// binds its fields first).
function def_raise(book: Bend.Book, t: HTerm, left: number): number {
  const s = term_strip(t);
  if (s.$ === "Lam") {
    const b = term_open(s).b;
    return left > 0 ? def_raise(book, b, left - 1) : 1 + def_raise(book, b, 0);
  }
  if (s.$ === "Mat") {
    return Math.min(def_raise(book, s.h, left - 1 + book.ctrs[s.k].n),
      def_raise(book, s.m, left));
  }
  return s.$ === "Efq" ? 99 : 0;
}

function def_foreign(tld: Bend.TLD | undefined):
  tld is Bend.Def & { i: string[] } {
  return tld?.$ === "Def" && tld.i !== undefined;
}

// Eff
// ===

function eff_name(k: Name): string {
  return (LOCAL.get(k) ?? k).toLowerCase().replace(/[./]/g, "_");
}

// Io
// ==

export function io_base(book: Bend.Book, t: HTerm): HTerm[] | null {
  const io = book.tlds["IO"];
  if (io?.$ !== "Def" || io.b !== true) {
    return null;
  }
  const tlds = { ...book.tlds, IO: { ...io, v: null } };
  const [h, xs] = Bend.term_unapply(Bend.term_wnf({ ...book, tlds }, t));
  return h.$ === "Ref" && h.k === "IO" ? xs : null;
}

export function io_type(book: Bend.Book): HTerm | null {
  const main = book.tlds["main"];
  const xs = main?.$ === "Def" ? io_base(book, main.T) : null;
  if (xs !== null && def_foreign(main as Bend.Def)) {
    die("main must be a filled def: a foreign main cannot anchor IO");
  }
  return xs?.length === 1 ? xs[0] : null;
}

// A pure main prints through a descriptor of its type, a node per (type,
// boxed?): 0 U32, 1 F32, 2 Nat, 3 Char, 4 String, 5 Eql, 6 Array (element,
// lgs), 7 Data (boxed?, arms; per arm name, cid, fields, bracket, then an
// (offset, node) per field). Null for an IO main; an unprintable type (a
// function, a Type, an erased or dependent field) refuses the build.
function show_main(book: Bend.Book): Show | null {
  const main = book.tlds["main"];
  if (main?.$ !== "Def" || (main.v === null && main.i === undefined)
    || book.tlds["IO"] === undefined) {
    die(book.tlds["IO"] === undefined ? "a build needs import Base"
      : "no main to run");
  }
  if (io_type(book) !== null) {
    return null;
  }
  const show: Show = { cells: [], names: [] };
  const ids = new Map<string, number>();
  const refuse = (): never => die("main's type " + Bend.term_show(
    Bend.term_lower(main.T)) + " cannot be printed (a function, a Type, an"
    + " erased or dependent field)");
  const node = (T: HTerm, lay: Lay): number => {
    const t = ty_wnf(book, T) as HTerm;
    const box = lay_box(lay);
    const key = String(box) + Bend.term_key(Bend.term_lower(t));
    const adt = ty_adt(book, t);
    const tld = adt && book.tlds[adt.k];
    const kind = t.$ === "Eql" ? 5 : "U32 F32 Nat Char String . Array"
      .split(" ").indexOf(adt?.k ?? "") & 7;
    if (ids.has(key)) {
      return ids.get(key)!;
    }
    if (kind !== 5 && (adt === null || adt.k === "IO.OP" || tld?.$ !== "ADT")) {
      return refuse();
    }
    const id = show.cells.push(kind) - 1;
    ids.set(key, id);
    const refs: [number, HTerm, Lay][] = [];
    if (kind === 3) {
      show.cells.push(Number(box));
    } else if (kind === 6) {
      const el = lay_el(book, adt!.x[0]);
      refs.push([show.cells.push(0, lay_arr(el).lgs) - 2, adt!.x[0], el]);
    } else if (kind === 7 && tld?.$ === "ADT") {
      show.cells.push(Number(box), tld.c.length);
      for (const [j, c] of tld.c.entries()) {
        const fs = box ? lay_node(book, c.k).arms![0].fs : lay.arms![j].fs;
        const doms = ctr_tail(book, c, adt!.x);
        show.cells.push(show.names.push(c.k) - 1, c.k, doms.length,
          c.k === "Tuple" ? 2 : Number(c.k === "Con" || c.k === "Nil"));
        for (const [f, d] of doms.entries()) {
          if (!live_dom(d)) {
            refuse();
          }
          refs.push([show.cells.push(fs[f].at, 0) - 1, d[2], fs[f].lay]);
        }
      }
    }
    for (const [at, T2, l] of refs) {
      show.cells[at] = node(T2, l);
    }
    return id;
  };
  const lay = lay_of(book, main.T);
  node(main.T, lay.ks.length === 0 ? BOX : lay);
  return show;
}

export function io_run(book: Bend.Book, args: string[] = []): number {
  const src = js_lib(book, ["main"], null) + "\n" + RUNTIME_MAIN
    + "\ncli_args = " + JSON.stringify(args) + ";\nreturn io_run("
    + js_sat("main") + ");";
  return new Function("require", src)(import.meta.require) as number;
}

// Anf
// ===
// A statement in normal form: a fork, a cut, a let of a value, or a tail.

function anf(cb: Carb, t: HTerm, ty: HTerm | null = null): HTerm {
  const binds: [Probe, HTerm][] = [];
  const cut = (r: HTerm, T: HTerm | null): HTerm => {
    if (call_kind(cb, r) === null || flat_call(cb, r)) {
      return r;
    }
    const p = probe("h");
    binds.push([p, Bend.Ann(r, T ?? die("an untyped cut"))]);
    return Bend.Ann(p, T!);
  };
  const go = (u: HTerm, top: boolean, T: HTerm | null): HTerm => {
    const s = term_force(u);
    if (term_const(s)) {
      return s;
    }
    switch (s.$) {
      case "Ann": {
        const x = go(s.x, top, s.T);
        return x === s.x ? s : Bend.Ann(x, s.T, s.s);
      }
      case "Rwt": return go(s.f, top, T);
      case "Ctr": {
        const on = ctr_flds(cb.book, s.k, s.x);
        const xs = s.x.map((x) => on.includes(x) ? go(x, false, null) : x);
        return xs.every((x, j) => x === s.x[j]) ? s : Bend.Ctr(s.k, xs, s.s);
      }
      case "Ref":
      case "App": {
        const m = term_spine(cb, s);
        // A call prefix of the spine (an over-application) cuts; a variable
        // applied to erased arguments is the variable.
        const spine = (v: HTerm): HTerm => {
          const f = term_force(v);
          if (f.$ === "Ann") {
            const x = spine(f.x);
            return x === f.x ? f : Bend.Ann(x, f.T, f.s);
          }
          if (f.$ !== "App") {
            return f;
          }
          if (m.t.$ === "Var" && !m.args.includes(f.x)) {
            return spine(f.f);
          }
          const g = cut(spine(f.f), ty_ann(f.f));
          const x = m.args.includes(f.x) ? go(f.x, false, null) : f.x;
          return g === f.f && x === f.x ? f : Bend.App(g, x, f.s);
        };
        const r = spine(s);
        return top ? r : cut(r, T);
      }
      case "Let": {
        const o = term_open(s);
        const on = let_live(cb, s);
        for (const [j, v] of s.v.entries()) {
          if (on[j]) {
            binds.push([o.ps[j], go(v, true, null)]);
          }
        }
        return go(o.b, top, T);
      }
      case "Lam": {
        const all = T && Bend.tele_open(cb.book, T);
        if (all === null || quant_live(all.q)) {
          return s;
        }
        return Bend.Ann(go(s.f(DUMMY), top, all.B(DUMMY)), all.B(DUMMY));
      }
      default: return s;
    }
  };
  const wrap = (b: HTerm): HTerm =>
    binds.reduceRight((b2, [p, v]) => let_open([p], [v], b2), b);
  const x = term_force(t);
  if (x.$ !== "Let") {
    const b = go(x, true, ty);
    return wrap(binds.length === 0 || ty === null ? b : Bend.Ann(b, ty));
  }
  const o = term_open(x);
  const on = let_live(cb, x);
  const ps = o.ps.filter((_, j) => on[j]);
  const vs = x.v.filter((_, j) => on[j]);
  if (ps.length === 0) {
    return o.b;
  }
  if (ps.length >= 2 && !vs.every((v) => call_kind(cb, v) !== null)) {
    return anf(cb, ps.reduceRight((b, p, j) => let_open([p], [vs[j]], b), o.b));
  }
  const ws = vs.map((v) => go(v, true, null));
  return wrap(on.every(Boolean) && ws.every((w, j) => w === x.v[j]) ? x
    : let_open(ps, ws, o.b));
}

// Carb
// ====

function def_body(cb: Carb, k: Name): TLD | undefined {
  const tld = cb.book.tlds[k];
  if (tld?.$ === "Def" && tld.e !== undefined && tld.h === undefined) {
    const h0 = Bend.term_higher(tld.e);
    const h = pick_any(tld.e) ? pick_lift(cb, k, h0) : h0;
    const n = tld.n + Math.min(def_raise(cb.book, h, tld.n),
      tele_unbind(cb.book, tld.T).doms.length - tld.n);
    cb.book.tlds[k] = { ...tld, n, h };
  }
  return cb.book.tlds[k];
}

// Pick
// ----
// Bool.pick(A, c, a, b) is a def, so a call evaluates both arms. A site
// whose arms are not both cheap (a variable, a literal, a constructor or a
// word op of those) becomes a call to a def of its own that matches on c,
// as a hand-written helper would: c and the arms' free variables are its
// parameters, so only the chosen arm runs and the other's variables drop
// with the match's arm. The checker never sees it: this is the emitters'
// view of the body alone. The def's body is lowered to levels (its type
// cells kept as they are), each site is lifted bottom-up, and the result
// is raised again; a def with no such site keeps its body.

type PickBind = { k: Bend.Name; q: Bend.Quant; A: HTerm | null };

const PICK_WORD = /^(U32|I32|F32|U64|I64|F64|Char|Nat)\./;

function pick_cheap(cb: Carb, t: Bend.LTerm): boolean {
  const s = Bend.term_strip(t);
  switch (s.$) {
    case "Var": case "Lit": return true;
    case "Ctr": return s.x.every((x) => pick_cheap(cb, x));
    case "App": {
      const [g, xs] = pick_spine(s);
      const it = g.$ === "Ref" ? intr_of(cb, g.k) : undefined;
      return g.$ === "Ref" && it !== undefined && it.call !== true
        && PICK_WORD.test(g.k) && xs.every((x) => pick_cheap(cb, x)
          || "Ref ADT Typ Qua".includes(Bend.term_strip(x).$));
    }
    default: return false;
  }
}

function pick_lift(cb: Carb, def: Bend.Name, h: HTerm): HTerm {
  let n = 0;
  const cell = (T: HTerm): Bend.LTerm => Bend.term_cell(T) as Bend.LTerm;
  const lift = (as: Bend.LTerm[], R: HTerm | null, d: number,
    bs: PickBind[]): Bend.LTerm | null => {
    const [, c, a, b] = as;
    if (R === null || (pick_cheap(cb, a) && pick_cheap(cb, b))) {
      return null;
    }
    const lv = new Set<number>();
    const scan = (t: Bend.LTerm): void => {
      switch (t.$) {
        case "Var": if (t.i >= 0 && t.i < d) lv.add(t.i); return;
        case "Sub": return scan(t.f);
        case "Let": t.v.forEach(scan); return scan(t.f);
        case "All": scan(t.A); return scan(t.B);
        case "Lam": return scan(t.f);
        case "App": scan(t.f); return scan(t.x);
        case "ADT": case "Ctr": return t.x.forEach(scan);
        case "Mat": scan(t.h); return scan(t.m);
        case "Eql": scan(t.a); scan(t.b); return scan(t.T);
        case "Rwt": scan(t.e); scan(t.p); return scan(t.f);
        case "Min": scan(t.a); return scan(t.b);
        case "Typ": return scan(t.g);
        case "Ann": scan(t.x); return scan(t.T);
        default: return;
      }
    };
    scan(a);
    scan(b);
    const fv = [...lv].sort((x, y) => x - y);
    if (fv.some((i) => bs[i]?.A == null)) {
      return null;
    }
    const ps = fv.map((i) => bs[i]);
    const k = def + "$pick" + String(n++);
    const tel = (j: number): HTerm => j === ps.length ? R
      : Bend.All(ps[j].q, ps[j].k, 0, ps[j].A as HTerm, () => tel(j + 1));
    const T = Bend.All(Bend.Lone(), "c", 0, Bend.Ref("Bool"), () => tel(0));
    const arm = (x: Bend.LTerm, j: number, env: Bend.Env): HTerm =>
      j === ps.length ? Bend.term_higher(x, env)
      : Bend.Ann(Bend.Lam(ps[j].k, 0, (v: HTerm) =>
        arm(x, j + 1, Bend.list_set(env, fv[j], v)), undefined, ps[j].q),
      tel(j));
    const m = Bend.Ann(Bend.Mat("False", arm(b, 0, null),
      Bend.Ann(Bend.Mat("True", arm(a, 0, null), Bend.Ann(Bend.Efq(), T)), T)),
    T);
    cb.book.tlds[k] = { $: "Def", n: 1 + ps.length, x: 0, T, v: m, h: m };
    return Bend.Ann(fv.reduce((f: Bend.LTerm, i, j) => Bend.App(f,
      Bend.Ann(Bend.Var(ps[j].k, i), cell(ps[j].A as HTerm))),
    Bend.App(Bend.Ref(k), c)), cell(R));
  };
  const low = (t0: HTerm, d: number, bs: PickBind[],
    ty: HTerm | null): Bend.LTerm => {
    const t = t0.$ === "Var" && t0.i < 0 ? t0 : Bend.term_force(t0);
    switch (t.$) {
      case "Var": return t.i < 0 ? t as Bend.LTerm : Bend.Var(t.k, t.i, t.s);
      case "Ref": return Bend.Ref(t.k, t.s, t.b);
      case "Sub": return Bend.Sub(t.i, t.v.$ === "PVar" || t.v.$ === "PCtr"
        ? t.v : low(t.v, d, bs, null), low(t.f, d, bs, null), t.s);
      case "Let": {
        const xs = t.k.map((k, j): HTerm => Bend.Var(k, d + j));
        const vs = t.v.map((v) => low(v, d, bs, null));
        const inner = [...bs, ...t.k.map((k, j): PickBind =>
          ({ k, q: t.q[j] ?? Bend.Lone(), A: ty_ann(t.v[j]) }))];
        return Bend.Let(t.k, xs.map((_, j) => d + j), vs,
          low(t.f(xs), d + t.k.length, inner, null), t.s, t.q);
      }
      case "Typ": return Bend.Typ(low(t.g, d, bs, null), t.s);
      case "Qnt": case "Qua": case "Lit": return t as Bend.LTerm;
      case "Min": return Bend.Min(low(t.a, d, bs, null), low(t.b, d, bs, null),
        t.s);
      case "All": return Bend.All(t.q, t.k, d, low(t.A, d, bs, null),
        low(t.B(Bend.Var(t.k, d)), d + 1, [...bs, { k: t.k, q: t.q, A: null }],
          null), t.s);
      case "Lam": {
        const all = ty && Bend.tele_open(cb.book, ty);
        const b: PickBind = { k: t.k, q: all?.q ?? t.q ?? Bend.Lone(),
          A: all?.A ?? null };
        return Bend.Lam(t.k, d, low(t.f(Bend.Var(t.k, d)), d + 1, [...bs, b],
          null), t.s, t.q);
      }
      case "App": {
        const [g, xs] = pick_spine(t);
        if (g.$ === "Ref" && g.k === "Bool.pick" && xs.length === 4) {
          const ls = xs.map((x) => low(x, d, bs, null));
          const r = lift(ls, ty ?? ty_ann(xs[2]), d, bs);
          if (r !== null) {
            return r;
          }
          // kept: the spine again, over the arguments lowered once
          let at = 0;
          const re = (u: HTerm): Bend.LTerm => {
            const v = Bend.term_force(u);
            return v.$ === "Ann" ? Bend.Ann(re(v.x), low(v.T, d, bs, null), v.s)
              : v.$ === "App" ? Bend.App(re(v.f), ls[at++], v.s)
              : low(v, d, bs, null);
          };
          return re(t);
        }
        return Bend.App(low(t.f, d, bs, null), low(t.x, d, bs, null), t.s);
      }
      case "ADT": return Bend.ADT(t.k, t.x.map((x) => low(x, d, bs, null)),
        t.s, t.r);
      case "Ctr": return Bend.Ctr(t.k, t.x.map((x) => low(x, d, bs, null)),
        t.s);
      case "Mat": return Bend.Mat(t.k, low(t.h, d, bs, null),
        low(t.m, d, bs, null), t.s);
      case "Efq": return Bend.Efq(t.s);
      case "Eql": return Bend.Eql(low(t.a, d, bs, null), low(t.b, d, bs, null),
        low(t.T, d, bs, null), t.s);
      case "Rfl": return Bend.Rfl(t.s);
      case "Rwt": return Bend.Rwt(low(t.e, d, bs, null), low(t.p, d, bs, null),
        low(t.f, d, bs, null), t.s);
      case "Hol": return Bend.Hol(t.k, t.s);
      case "Ann": return Bend.Ann(low(t.x, d, bs, t.T), low(t.T, d, bs, null),
        t.s);
    }
  };
  const out = low(h, 0, [], null);
  return n === 0 ? h : Bend.term_higher(out);
}

// Does an elaborated body call Bool.pick? (its type cells aside)
function pick_any(t: Bend.LTerm): boolean {
  switch (t.$) {
    case "Ref": return t.k === "Bool.pick";
    case "Sub": return (t.v.$ !== "PVar" && t.v.$ !== "PCtr"
      && pick_any(t.v as Bend.LTerm)) || pick_any(t.f);
    case "Let": return t.v.some(pick_any) || pick_any(t.f);
    case "Lam": return pick_any(t.f);
    case "App": return pick_any(t.f) || pick_any(t.x);
    case "Ctr": return t.x.some(pick_any);
    case "Mat": return pick_any(t.h) || pick_any(t.m);
    case "Rwt": return pick_any(t.f);
    case "Ann": return pick_any(t.x);
    default: return false;
  }
}

// A term's head and arguments, through its annotations
function pick_spine<X>(t: Bend.TermOf<X>): [Bend.TermOf<X>, Bend.TermOf<X>[]] {
  const xs: Bend.TermOf<X>[] = [];
  let c = Bend.term_force(t);
  while (c.$ === "Ann" || c.$ === "App") {
    if (c.$ === "App") {
      xs.push(c.x);
    }
    c = Bend.term_force(c.$ === "App" ? c.f : c.x);
  }
  return [c, xs.reverse()];
}

// The reachable defs, raised, with bang and call-site counts and each one's
// refs, calls and flatness (no fork, no bang call, only tail self-calls).
function carb_book(src: Bend.Book, roots: Name[]): Carb {
  book_owned(src);
  [TELES, SRCS, NODES, LAYS, CYCLES, FLATS, SIGS, BRWS].forEach((m) =>
    m.clear());
  LOCAL.clear();
  PROBES.length = 1;
  for (const [k, tld] of Object.entries(src.tlds)) {
    if (def_foreign(tld)) {
      LOCAL.set(k, name_own(k, tld, "(def|law) "));
    }
  }
  const cb: Carb = {
    book: { ...src, tlds: { ...src.tlds } },
    bangs: new Set(),
    sites: new Map(),
    hot: new Set(),
    stat: new Set(),
    own: new Set(),
    lend: new Set(),
  };
  for (const queue = roots.slice(); queue.length > 0;) {
    const d = queue.shift() as Name;
    if (SRCS.has(d)) {
      continue;
    }
    memo_gc();
    const tld = def_body(cb, d);
    const own: Src = { refs: new Set(), deps: new Set(), flat: done_live(tld) };
    SRCS.set(d, own);
    for (const x of tld?.$ === "ADT" ? tld.c : tld ? [tld] : []) {
      queue.push(...type_adts(cb, x.T));
    }
    if (!done_live(tld)) {
      continue;
    }
    term_any(cb, tld.h as HTerm, (s, tail) => {
      if (s.$ === "Ann") {
        queue.push(...type_adts(cb, s.T));
      }
      if (s.$ === "Ref") {
        if (s.b) {
          cb.bangs.add(s.k);
        }
        if (intr_of(cb, s.k) === undefined) {
          own.refs.add(s.k);
          cb.sites.set(s.k, (cb.sites.get(s.k) ?? 0) + 1);
        }
      }
      const ck = call_kind(cb, s);
      if (ck !== null && ck.k !== d) {
        own.deps.add(ck.k);
      }
      if ((s.$ === "Let" && s.k.length >= 2)
        || (ck !== null && (ck.bang === true || (ck.k === d && !tail)))) {
        own.flat = false;
      }
      return false;
    });
    queue.push(...own.refs);
  }
  return cb;
}

// The datatypes a type mentions
function type_adts(cb: Carb, T: HTerm): Name[] {
  const t = ty_wnf(cb.book, T);
  switch (t?.$) {
    case "All": return [...type_adts(cb, t.A), ...type_adts(cb, t.B(DUMMY))];
    case "Lam": return type_adts(cb, t.f(DUMMY));
    case "ADT": return [...WORDS[t.k] === undefined && t.k !== "Array"
      ? [t.k] : [], ...t.x.flatMap((x) => type_adts(cb, x))];
    default: return [];
  }
}

// Flat
// ====

function flat_call(c: Carb, t: HTerm): boolean {
  const ck = call_kind(c, t);
  return ck !== null && ck.bang !== true && flat_of(ck.k);
}

// A def is flat when its source is and every def it calls is.
function flat_of(k: Name): boolean {
  return memo(FLATS, k, () => {
    const own = SRCS.get(k);
    FLATS.set(k, false);
    return own !== undefined && own.flat && [...own.deps].every(flat_of);
  });
}

// Done
// ====

function done_live(tld: Bend.TLD | undefined): tld is Bend.Def {
  return tld?.$ === "Def" && tld.v !== null;
}

function done_defs(cb: Carb, live = done_live): [Name, Def][] {
  return [...SRCS.keys()].map((k) => [k, cb.book.tlds[k]] as [Name, Def])
    .filter((p) => live(p[1]));
}

// Cid
// ===

function cid_mac(k: string): string {
  return "CID_" + name_clean(k).toUpperCase();
}

// File
// ====

function file_new(cb: Carb, js: boolean): File {
  return { ...cb, js, segs: [], seg: seg_new("", BOX, []), tab: 2,
    cids: new Map(), tabs: new Map(), spins: [], spun: new Map(), clos: new Set(),
    img: [], lits: new Map(), consts: new Map(), reqs: "", fuel: 0,
    fresh: new Map(), spares: [],
    uses: new Map(), brwl: new Map(), rest: [], def: "" };
}

function file_push(fl: File, line: string): void {
  fl.seg.lines.push("  ".repeat(fl.tab) + line);
}

// Block
// =====

function block(fl: File, open: string, go: () => void): void {
  file_push(fl, open);
  fl.tab += 1;
  go();
  fl.tab -= 1;
  file_push(fl, "}");
}

// Cls
// ===

function cls_fit(words: number): number {
  return 32 - Math.clz32(words - 1);
}

// Spare
// =====

function spare_free(fl: File, words: number, name: string,
  z: boolean): void {
  file_push(fl,
    `${z ? "spare_free" : "heap_free"}(e, cls_fit(${words}), ${name});`);
}

function spare_flush(fl: File): void {
  for (const s of fl.spares.reverse()) {
    spare_free(fl, s.words, s.name, s.z);
  }
  fl.spares = [];
}

// Seg
// ===

function seg_new(name: string, ret: Lay, params: string[],
  ks: Kind[] = params.map(() => "w64"), frame: Seg["frame"] = null): Seg {
  return { fid: seg_fid(name), def: name, ret, lines: [], params, ks, frame,
    refs: new Set() };
}

function seg_fid(k: Name): string {
  return "FID_" + name_clean(k).toUpperCase();
}

// A segment's entry: its frame popped, its parameters read from the frame
// slots, then the bank.
function seg_take(seg: Seg): string[] {
  const { pop, at } = seg.frame ?? { pop: 0, at: [] };
  return [...pop > 0 ? [`WL_POPN(${pop});`] : [], ...seg.params.map((p, i) =>
    `${lay_c(seg.ks[i])} ${p} = ${i < at.length ? `STK(${at[i]})`
      : `r${i - at.length}`};`)];
}

function seg_ref(fl: File, fid: string): string {
  fl.seg.refs.add(fid);
  return fid;
}

// A closure over `fid` holding `words` (so the device holds `fid`).
function seg_clo(fl: File, fid: string, words: string[]): string {
  fl.clos.add(fid);
  return `term_clo(${seg_ref(fl, fid)}, ${words.length === 0 ? 0 : node_fill(
    fl, "nd", `heap_alloc(e, cls_fit(${words.length}))`, words)})`;
}

function seg_name(fl: File, stem: string): string {
  return fl.seg.def.split("$")[0] + "$" + stem + fl.segs.length;
}

// Opens `name`: takes `live` (per `frame`, else in r0..; a `boxed` one
// unboxed), then `ks` words.
function seg_open(fl: File, name: string, ret: Lay, frame: Seg["frame"],
  live: [Probe, Bind][], k: string, ks: Kind[], rest: HTerm[],
  boxed: boolean[] = []): string[] {
  const olds = live.flatMap(([, b]) => b.val.ws);
  const news = olds.map((w) => name_local(fl, w.replace(/_\d+$/, "")));
  const ts = ks.map(() => name_local(fl, k));
  const seg = seg_new(name, ret, [...news, ...ts],
    [...live.flatMap(([, b]) => b.val.lay.ks), ...ks], frame);
  fl.segs.push(seg);
  Object.assign(fl, { seg, spares: [], tab: 2, uses: new Map() });
  olds.forEach((w, i) => {
    if (fl.brwl.has(w)) {
      fl.brwl.set(news[i], fl.brwl.get(w)!);
    }
  });
  let i = 0;
  live.forEach(([p, b], j) => {
    const v = val_new(news.slice(i, i += b.val.ws.length), b.val.lay);
    bind_uses(fl, p, boxed[j] ? val_unbox(fl, v, X64) : v, rest, b.A, false);
  });
  return ts;
}

// Node
// ====

function node_fill(fl: File, k: string, alloc: string,
  exprs: string[], shr = false): string {
  const nd = name_local(fl, k);
  file_push(fl, `u64 ${nd} = ${alloc};`);
  exprs.forEach((w, j) => {
    file_push(fl, `e.mem[${nd} + ${j}] = ${shr ? `rfc_seal(e, ${w})` : w};`);
  });
  return nd;
}

function node_fields(fl: File, t: string, node: Lay,
  tail = false): Val[] {
  const n = node.ks.length;
  const fs = node.arms![0].fs;
  if (node.arms![0].k === "SCon") {
    // A tail is new owned metadata; consuming this root participates in
    // the fixed-point borrow analysis, including polymorphic containers.
    val_own(fl, val_new([t], BOX));
    const out = name_local(fl, "str");
    file_push(fl, `Term ${out}[2];`);
    file_push(fl, `str_uncons(e, ${t}, ${out});`);
    const ws = emit_hold(fl, [`${out}[0]`, `${out}[1]`], "f", node.ks);
    return fs.map((f) => val_field(val_new(ws, node), f));
  }
  if (n === 0 || (n === 1 && node.ks[0] === "w32")) {
    return fs.map((f) => val_new(f.lay.ks.map(() => `term_loc(${t})`), f.lay));
  }
  const r = fl.brwl.get(t);
  const k = node.arms![0].k;
  const z = r === undefined && (fl.hot.has(k) || fl.stat.has(k));
  const sp = name_local(fl, "sp");
  let fb = `e.mem[${sp} + `;
  if (z) {
    fb = name_local(fl, "fb") + "[";
    file_push(fl, `Term ${fb}${n}];`);
    file_push(fl, `u64 ${sp} = ctr_take(e, ${t}, ${n}, ${fb.slice(0, -1)});`);
  } else {
    file_push(fl, `u64 ${sp} = ${r === undefined ? "term_loc(" : "term_peek(e, "
    }${t});`);
  }
  const ws = emit_hold(fl, node.ks.map((_, j) => `${fb}${j}]`), "f", node.ks);
  if (r !== undefined) {
    ws.forEach((w, j) => {
      if (node.ks[j] === "box") {
        fl.brwl.set(w, r);
      }
    });
  } else if (tail) {
    fl.spares.push({ words: n, name: sp, z });
  } else {
    spare_free(fl, n, sp, z);
  }
  return fs.map((f) => val_field(val_new(ws, node), f));
}

// Facts
// =====
// The emitter is the analysis: boxed parameters start borrowed (brwl) and
// become owned (own) when owned or unlent; a shared value heats its type
// (hot). compile_book emits until a pass changes nothing.

function facts_hot(fl: File, B: HTerm | null, force: boolean,
  local = false): void {
  const w = ty_wnf(fl.book, B);
  if (w?.$ === "Lam") {
    return facts_hot(fl, w.f(DUMMY), force, local);
  }
  if (w?.$ !== "ADT") {
    if (!force || (w?.$ === "App" && facts_fam(fl, w, local))) {
      return;
    }
    if (w?.$ === "Mat") {
      return term_kids(fl, w).forEach((h) => facts_hot(fl, h, true, local));
    }
    const dom = w?.$ === "Var" && !local && tele_unbind(fl.book,
      (fl.book.tlds[fl.def] as Def).T).doms[w.i];
    if (dom && dom[1] === w.k && !live_dom(dom)) {
      fl.hot.add(fl.def + "~" + w.i);
    } else if ("All Var App".includes(w?.$!)) {
      fl.hot.add("*");
    }
    return;
  }
  const tk = "t:" + w.k;
  const hot = force || fl.hot.has(tk);
  w.x.forEach((x) => facts_hot(fl, x, hot, local));
  if (!hot || fl.hot.has(tk)) {
    return;
  }
  fl.hot.add(tk);
  const tld = fl.book.tlds[w.k];
  if (tld?.$ === "ADT") {
    for (const c of tld.c) {
      fl.hot.add(c.k);
      facts_ctr(fl, c, w.x);
    }
  }
}

// A family stuck on an open index is its arms' types, walked once per
// family; its arguments, like an ADT's, at every instantiation.
function facts_fam(fl: File, w: HTerm, local: boolean): boolean {
  const m = term_spine(fl, w);
  const fam = m.tld?.$ === "Def" && m.tld.v !== null && term_strip(
    Bend.term_unapply(m.all.reduce((b, x) => Bend.term_apply(b, x),
      m.tld.v))[0]);
  if (!fam || fam.$ !== "Mat") {
    return false;
  }
  m.all.forEach((x) => facts_hot(fl, x, true, local));
  const key = "m:" + (m.t as Of<"Ref">).k;
  if (!fl.hot.has(key)) {
    fl.hot.add(key);
    facts_hot(fl, fam, true, local);
  }
  return true;
}

// A hot constructor's fields are hot at this instantiation; a field typed
// by its own erased binder is unknown, never poly.
function facts_ctr(fl: File, c: Bend.Ctr, xs: HTerm[]): void {
  const own = ctr_tail(fl.book, c, xs).some((d) => !live_dom(d));
  ctr_doms(fl.book, c, xs).forEach((A) => facts_hot(fl, A, true, own));
}

// A lend is asked by a holder or passed on from a lent root (k~i<j~q); a
// parameter nobody asks to lend is owned.
function facts_lend(cb: Carb): void {
  for (let n = -1; n !== cb.lend.size;) {
    n = cb.lend.size;
    cb.lend.forEach((l) => {
      const [a, r] = l.split("<");
      if (r !== undefined && cb.lend.has(r)) {
        cb.lend.add(a);
      }
    });
  }
  BRWS.forEach((bs, k) => bs.forEach((b, i) => {
    if (b && !cb.lend.has(k + "~" + i)) {
      cb.own.add(k + "~" + i);
    }
  }));
}

// A value with no heap: a constructor packed into its word.
function facts_packed(cb: Carb, t: HTerm): boolean {
  const s = term_strip(t);
  return s.$ === "Ctr"
    && ["", "w32"].includes(lay_node(cb.book, s.k).ks.join());
}

// Val
// ===

function val_new(ws: string[], lay: Lay, stat = false): Val {
  return { ws, lay, stat };
}

function val_field(v: Val, f: Field): Val {
  return val_new(v.ws.slice(f.at, f.at + f.lay.ks.length), f.lay);
}

function val_word(v: Val): string {
  if (v.ws.length !== 1) {
    die(`a ${v.ws.length}-word value where one word was expected`);
  }
  return v.ws[0];
}

function val_hold(fl: File, v: Val, k: string): Val {
  return val_new(v.ws.map((w, j) => emit_alias(fl, w, k, v.lay.ks[j])),
    v.lay);
}

// The one gate: at an owned position a rooted word owns its root; lent at
// `at`, a rooted word passes its lend on, an unheld owned one owns at.
function val_own(fl: File, v: Val, at: string | null = null,
  held = false): string[] {
  v.ws.forEach((w, j) => {
    const r = fl.brwl.get(w);
    if (r !== undefined && at !== null) {
      fl.lend.add(at + "<" + r);
    } else if (r !== undefined
      || (at !== null && !held && v.lay.ks[j] === "box")) {
      fl.own.add(r ?? at!);
    }
  });
  return v.ws;
}

// The boxes a value owns (its words not rooted in a borrowed parameter).
function val_owned(fl: File, v: Val): string[] {
  return v.ws.filter((w, j) => v.lay.ks[j] === "box" && !fl.brwl.has(w));
}

function val_brw(fl: File, v: Val): boolean {
  return val_owned(fl, v).length === 0;
}

function val_sink(fl: File, v: Val): void {
  val_owned(fl, v).forEach((w) => file_push(fl, `term_sink(e, ${w});`));
}

function val_to(fl: File, v: Val, lay: Lay): Val {
  if (lay_eq(v.lay, lay)) {
    return v;
  }
  if (lay_box(lay)) {
    return val_new([val_box(fl, v)], BOX);
  }
  if (lay_box(v.lay)) {
    return val_unbox(fl, v, lay);
  }
  return val_arms(fl, lay, v.ws[0], (t, i) => `${t} == ${i}`, (arm) => {
    const from = lay_arm(v.lay, arm.k);
    return arm.fs.map((f, j) => val_to(fl, val_field(v, from.fs[j]), f.lay));
  });
}

// A destination every arm fills from one root is rooted; else its rooted
// sources own their roots.
function val_arms(fl: File, lay: Lay, sel: string,
  cond: (t: string, i: number) => string, read: (arm: Arm) => Val[]): Val {
  const arms = lay.arms!;
  if (arms.length <= 1) {
    return val_new(arms.flatMap(read).flatMap((g) => g.ws), lay);
  }
  const out = emit_dst(fl, lay, "o").ws;
  const t = emit_alias(fl, sel, "t");
  const rs: string[][] = out.map(() => []);
  const bodies = arms.map((arm, i) => () => {
    file_push(fl, `${out[0]} = ${i};`);
    read(arm).forEach((g, j) => g.ws.forEach((w, n) => {
      rs[arm.fs[j].at + n].push(fl.brwl.get(w) ?? "");
      file_push(fl, `${out[arm.fs[j].at + n]} = ${w};`);
    }));
  });
  emit_chain(fl, (i) => cond(t, i), bodies);
  rs.forEach((r, k) => {
    if (r[0] && r.every((x) => x === r[0])) {
      fl.brwl.set(out[k], r[0]);
    } else {
      r.filter((x) => x).forEach((x) => fl.own.add(x));
    }
  });
  return val_new(out, lay);
}

function val_box(fl: File, v: Val): string {
  if (v.lay.arms === null) {
    const w = val_own(fl, v)[0];
    return v.lay === X64 ? emit_alias(fl, `x64_box(e, ${w})`, "b") : w;
  }
  const arms = v.lay.arms!;
  const build = (arm: Arm): string => {
    const fs = lay_node(fl.book, arm.k).arms![0].fs;
    return ctr_build(fl, arm.k, arm.fs.flatMap((f, j) =>
      val_own(fl, val_to(fl, val_field(v, f), fs[j].lay))));
  };
  if (arms.length <= 1) {
    return arms.map(build)[0] ?? "0";
  }
  const out = emit_hold(fl, ["0"], "b")[0];
  const tag = emit_alias(fl, v.ws[0], "t");
  emit_chain(fl, (i) => `${tag} == ${i}`, arms.map((arm) => () => {
    file_push(fl, `${out} = ${build(arm)};`);
  }));
  return out;
}

function val_unbox(fl: File, v: Val, lay: Lay): Val {
  if (lay.arms === null) {
    const w = v.ws[0];
    return lay !== X64 ? val_new(v.ws, lay) : val_new([emit_alias(fl,
      `x64_${fl.brwl.has(w) ? "peek" : "take"}(e, ${w})`, "u", "w64")], lay);
  }
  const t = emit_alias(fl, v.ws[0], "u");
  return val_arms(fl, lay, t, (_, i) =>
    `term_aux(${t}) == ${cid_mac(lay.arms![i].k)}`, (arm) => {
    const fs = node_fields(fl, t, lay_node(fl.book, arm.k));
    return arm.fs.map((f, j) => val_to(fl, fs[j], f.lay));
  });
}

// Arr
// ===

function arr_lay(el: Lay): Lay {
  return lay_pack([["Tuple", [BOX, el]]]);
}

function arr_words(l: string, at: string, el: Lay, box: string): string[] {
  const { arr } = lay_arr(el);
  return el.ks.map((k, j) => k === "box" ? box.replaceAll("$", `${l} + ${at} + ${j}`)
    : `blk_read(e.mem, ${Number(arr)}, ${l}, ${at} + ${j})`);
}

function arr_cells(fl: File, l: string, at: string, el: Lay,
  box: string): Val {
  return val_new(emit_hold(fl, arr_words(l, at, el, box), "c", el.ks), el);
}

function arr_new(fl: File, d: string, v: Val, el: Lay): string {
  const { arr, lgs } = lay_arr(el);
  const ws = val_own(fl, val_to(fl, v, el));
  const fv = name_local(fl, "fv");
  file_push(fl, `Term ${fv}[${Math.max(1, ws.length)}];`);
  ws.forEach((w, j) => file_push(fl, `${fv}[${j}] = ${w};`));
  return `blk_new(e, ${Number(arr)}, ${d}, ${lgs}, ${ws.length}, ${fv})`;
}

function arr_op(fl: File, k: string, el: Lay, args: Val[]): Val {
  const { arr, lgs } = lay_arr(el);
  if (k === "array_new") {
    return val_new([arr_new(fl, val_word(args[0]), args[1], el)], BOX);
  }
  const a = emit_alias(fl, val_own(fl, args[0])[0], "a");
  if (k === "array_size") {
    return val_new([a, `(1ull << (blk_cls(${a}) - ${lgs}))`], arr_lay(W32));
  }
  // One range check for the four: in range, adjacent reads clang can widen
  if (k === "array_get4") {
    const i = emit_alias(fl, val_word(args[1]), "i", "w32");
    const [l, m] = emit_hold(fl, [`blk_loc(e.mem, ${a})`,
      `(u32)((1ull << (blk_cls(${a}) - ${lgs})) - 1)`], "at", ["w64", "w32"]);
    const j = emit_hold(fl, [`${i} & ${m}`], "at", ["w32"])[0];
    const xs = [0, 1, 2, 3].map(() => emit_dst(fl, el, "x"));
    const read = (at: (k: number) => string) => xs.forEach((x, k) => arr_words(l,
      at(k), el, "blk_keep(e, $)").forEach((c, n) => file_push(fl, `${x.ws[n]} = ${c};`)));
    block(fl, `if (${m} - ${j} >= 3) {`, () => read((k) => `((${j} + ${k}) << ${lgs})`));
    block(fl, "else {", () => read((k) => `blk_at(${a}, ${i} + ${k}, ${lgs})`));
    const lay = [el, el, el, el, BOX].reduce((r, l) => lay_pack([["Tuple", [l, r]]]));
    if (lay.ks.length > WIDE) {
      die("an Array.get4 of elements too wide");
    }
    return val_new([a, ...xs.flatMap((x) => x.ws)], lay);
  }
  const get = k === "array_get";
  const [l, at] = emit_hold(fl, [`blk_loc(e.mem, ${a})`,
    `blk_${get ? "get" : "at"}(${a}, ${val_word(args[1])}, ${lgs})`], "at");
  const old = arr_cells(fl, l, at, el, get ? "blk_keep(e, $)" : "e.mem[$]");
  if (!get) {
    val_own(fl, val_to(fl, args[2], el)).forEach((w, j) => {
      file_push(fl, `blk_write(e.mem, ${Number(arr)}, ${l}, `
        + `${at} + ${j}, ${w});`);
    });
    if (k !== "array_swap") {
      val_sink(fl, old);
      return val_new([a], BOX);
    }
  }
  return val_new([a, ...old.ws], arr_lay(el));
}

function arr_leaf(fl: File, s: string, el: Lay): Val {
  const got = arr_cells(fl, `blk_loc(e.mem, ${s})`, "0", el,
    `blk_shr(${s}) ? blk_keep(e, $) : e.mem[$]`);
  file_push(fl, `blk_free(e, ${s});`);
  return got;
}

// Bind
// ====

function bind_of(fl: File, p: Probe): Bind {
  return fl.uses.get(p)!;
}

// A use: the last takes the value, an earlier one shares it.
function bind_pop(fl: File, x: HTerm): Val {
  const p = probe_of(x);
  const b = bind_of(fl, p);
  if (b.n <= 1) {
    fl.uses.delete(p);
    return b.val;
  }
  fl.uses.set(p, { ...b, n: b.n - 1 });
  val_owned(fl, b.val).forEach((w) => {
    file_push(fl, `${w} = term_keep(e, ${w});`);
    facts_hot(fl, b.A, true);
  });
  return b.val;
}

// A shared box of a flat type unboxes before its first share (its words
// copy, not its node); only a fresh binding decides this.
function bind_uses(fl: File, p: Probe, v: Val, rest: HTerm[], A: HTerm,
  fresh = true): void {
  const n = rest_use(fl, rest, p);
  const lay = lay_of(fl.book, A);
  if (fresh && n > 1 && lay_box(v.lay) && !lay_box(lay) && !val_brw(fl, v)) {
    v = val_unbox(fl, v, lay);
  }
  facts_hot(fl, A, fl.hot.has("*"));
  if (n > 0) {
    fl.uses.set(p, { val: v, n, A });
  } else {
    val_sink(fl, v);
  }
}

// A binding with no use in `rest` dies here: its value is sunk.
function bind_dead(fl: File, rest: HTerm[]): void {
  for (const [p, b] of [...fl.uses]) {
    const n = rest_use(fl, rest, p);
    if (n === 0) {
      fl.uses.delete(p);
      val_sink(fl, b.val);
    } else if (n < b.n) {
      fl.uses.set(p, { ...b, n });
    }
  }
}

// Emit
// ====

function emit_hold(fl: File, exprs: string[], k: string,
  ks?: Kind[]): string[] {
  return exprs.map((ex, i) => {
    const al = name_local(fl, k);
    const ty = fl.js ? "const" : lay_c(ks?.[i] ?? "w64");
    file_push(fl, `${ty} ${al} = ${ex};`);
    return al;
  });
}

function emit_alias(fl: File, e: string, k: string, kd?: Kind): string {
  return /^\w*_\d+$/.test(e) ? e : emit_hold(fl, [e], k, kd && [kd])[0];
}

function emit_task(fl: File, fid: string, rem: number, words: string[],
  cont = "WL_CONT", idx: string | number = "WL_IDX"): string {
  return node_fill(fl, "t",
    `task_node(e, ${seg_ref(fl, fid)}, ${cont}, ${idx}, ${rem})`, words);
}

function emit_frame(fl: File, words: string[], next: string): void {
  const ws = [...words, seg_ref(fl, next)];
  file_push(fl, `WL_ROOM(${ws.length});`);
  ws.forEach((w, i) => file_push(fl, `STK(${i}) = ${w};`));
  file_push(fl, `WL_PUSHN(${ws.length});`);
}

// A self-jump reads its parameters back: the device's loop carries them
// typed, not as words (raytrace GPU 1.72x).
function emit_jump(fl: File, args: string[], k: Name,
  bang?: boolean): void {
  const fid = seg_fid(k);
  fl.seg.fork ||= bang;
  if (bang || fl.seg.def !== k) {
    block(fl, `if (${bang ? "!seq" : `!DEVICE && !seq && fid_nofk(${fid})`}) {`, () =>
      file_push(fl, `return term_tsk(${fid}, ${
        emit_task(fl, fid, 0, args)});`));
  }
  args.forEach((a, i) => file_push(fl, `r${i} = ${a};`));
  if (fl.seg.def !== k) {
    return file_push(fl, `WL_JMP(${seg_ref(fl, fid)});`);
  }
  fl.seg.spin = true;
  fl.seg.params.forEach((p, i) => file_push(fl, `${p} = r${i};`));
  file_push(fl, `WL_AGAIN(${fl.seg.fid});`);
}

// A call's arguments, laid out as the def takes them: nested ones first,
// owned ones popped before borrowed ones are read (a read asks a lend).
function emit_args(fl: File, ck: Call, jump = false, fork = false): string[] {
  const brw = brw_of(fl, ck.k);
  ck.all.forEach((a, q) => {
    if (fl.hot.has(ck.k + "~" + q)) {
      facts_hot(fl, a, true);
    }
  });
  const xs = ck.args.map((a) => term_strip(a));
  const vars = xs.filter((x) => x.$ === "Var");
  const rest = fl.rest;
  const lays = sig_def(fl, ck.k).lays;
  const vs = ck.args.map((a, i): Val | null => {
    if (xs[i].$ === "Var") {
      return null;
    }
    fl.rest = [...xs.slice(i + 1).filter((x) => x.$ !== "Var"), ...vars,
      ...rest];
    return emit_expr(fl, a, null, lays[i]);
  });
  fl.rest = rest;
  // an owned Var pops its binding now; a borrowed one is read below
  xs.forEach((x, i) => {
    if (!brw[i]) {
      vs[i] ??= bind_pop(fl, x);
    }
  });
  return xs.flatMap((x, i) => {
    const at = ck.k + "~" + i;
    let b = vs[i];
    if (b === null) {
      const p = probe_of(x);
      const bd = bind_of(fl, p);
      const twin = vars.filter((y) => probe_of(y) === p).length > 1;
      const dead = rest_use(fl, rest, p) === 0;
      if (!dead || (!jump && twin)) {
        fl.lend.add(at);
      } else {
        val_own(fl, bd.val, at, !jump);
      }
      if (dead && !twin && val_brw(fl, bd.val)) {
        fl.uses.delete(p);
      } else if (!fork) {
        fl.uses.set(p, { ...bd, n: Math.max(bd.n - 1, 1) });
      }
      b = bd.val;
    }
    const v = val_to(fl, b, lays[i]);
    if (!brw[i]) {
      return val_own(fl, v);
    }
    // a borrowed parameter takes an unconverted read or a packed value as is
    return (vs[i] === null && v === b) || facts_packed(fl, x) ? v.ws
      : val_own(fl, v, at);
  });
}

// Expressions in order, each seeing the later ones as its rest.
function emit_each(fl: File, xs: HTerm[], ats: Lay[] | null): Val[] {
  const rest = fl.rest;
  const vs = xs.map((x, i) => {
    fl.rest = [...xs.slice(i + 1), ...rest];
    return emit_expr(fl, x, null, ats && ats[i]);
  });
  fl.rest = rest;
  return vs;
}

function emit_put(fl: File, dst: Dst, v: Val): void {
  if (dst === null) {
    spare_flush(fl);
  }
  const ws = val_own(fl, val_to(fl, v, dst?.lay ?? fl.seg.ret));
  ws.forEach((w, j) => file_push(fl, `${dst?.ws[j] ?? "r" + j} = ${w};`));
  if (dst === null) {
    file_push(fl, `WL_RETN(${ws.length});`);
  }
}

function emit_fuse(fl: File, ck: Call, dst: Dst, tail = false): void {
  const tld = fl.book.tlds[ck.k] as Def;
  const doms = tele_unbind(fl.book, tld.T).doms;
  const ers = ck.all.filter((_, i) => i < tld.n && !quant_live(doms[i][0]));
  const { lays, ret } = sig_def(fl, ck.k);
  const flat = flat_of(ck.k);
  const ws = emit_args(fl, ck, tail && !flat);
  if (!flat) {
    const outer = fl.def;
    fl.def = ck.k;
    emit_body(fl, tld.h as HTerm, tld.T, ers, lays.map((lay) =>
      val_new(ws.splice(0, lay.ks.length), lay)), dst);
    fl.def = outer;
    return;
  }
  const out = emit_dst(fl, ret);
  const name = emit_native(fl, ck, ers);
  const o = name_local(fl, "o");
  file_push(fl, `Term ${o}[${out.ws.length}];`);
  block(fl, `if (${name}(${["e", o, ...ws].join(", ")}) == 0) {`, () => {
    file_push(fl, "return 0;");
  });
  out.ws.forEach((v, j) => file_push(fl, `${v} = ${o}[${j}];`));
  if (tail) {
    bind_dead(fl, []);
  }
  emit_put(fl, dst, out);
}

// Opens a unit of `k`: fresh state, parameters bound, segment made.
function emit_open(fl: File, k: Name): Val[] {
  Object.assign(fl, { spares: [], tab: 2, uses: new Map(),
    fuel: FOLD_FUEL, def: k });
  const { live, lays, ret } = sig_def(fl, k);
  const vals = lays.map((l, i) =>
    val_new(l.ks.map(() => name_local(fl, live[i][1])), l));
  // a borrowed parameter's boxes are rooted in it
  brw_of(fl, k).forEach((b, i) => vals[i].ws.forEach((w, j) => {
    if (b && lays[i].ks[j] === "box") {
      fl.brwl.set(w, k + "~" + i);
    }
  }));
  fl.seg = seg_new(k, ret, vals.flatMap((v) => v.ws),
    vals.flatMap((v) => v.lay.ks));
  return vals;
}

function emit_native(fl: File, ck: Call, ers: HTerm[]): string {
  const key = [ck.k, ...ers.map((e) => JSON.stringify(lay_of(fl.book, e)))]
    .join("|");
  const got = fl.spun.get(key);
  if (got !== undefined) {
    return seg_ref(fl, got);
  }
  const name = seg_ref(fl, `spin_${fl.spun.size}`);
  fl.spun.set(key, name);
  const tld = fl.book.tlds[ck.k] as Def;
  const outer = { ...fl };
  const vals = emit_open(fl, ck.k);
  const seg = fl.seg;
  seg.fid = name;
  const dst = val_new(seg.ret.ks.map(() => name_local(fl, "v")), seg.ret);
  emit_body(fl, tld.h as HTerm, tld.T, ers, vals, dst);
  fl.spins.push({ fid: name, refs: seg.refs, text: [`${seg.lines.length < SPIN_FAR ? "INLINE" : "FAR"} Term ${name}(Env e, THR Term* o${
    seg.ks.map((k, i) => `, ${lay_c(k)} r${i}`).join("")}) {`,
  "  u32 wpoll = 0;",
  ...dst.ws.map((v, j) => `  ${lay_c(seg.ret.ks[j])} ${v} = 0;`),
  ...seg_take(seg).map((l) => "  " + l),
  "  WL_SPIN", ...seg.lines, "  break;", "  }",
  ...dst.ws.map((v, j) => `  o[${j}] = ${v};`),
  "  return 1;", "}"].join("\n") });
  Object.assign(fl, outer);
  return name;
}

function emit_dst(fl: File, lay: Lay, k = "v"): Val {
  return val_new(emit_hold(fl, lay.ks.map(() => "0"), k, lay.ks), lay);
}

function emit_intr(fl: File, it: Intr, x: HTerm,
  ty: HTerm | null): Val {
  const m = term_spine(fl, x);
  const k = (m.t as Of<"Ref">).k;
  // A value a polymorphic call handed back boxed is read out of its box.
  const lays = sig_def(fl, k).lays;
  const peek = it.peek ?? [];
  const sinks: Val[] = [];
  const args = emit_peek(fl, m.args, peek, sinks).map((v, i) =>
    lays[i] !== undefined && !lay_box(lays[i]) && lay_box(v.lay)
      ? val_to(fl, v, lays[i]) : v);
  const op = eff_name(k);
  // Native aggregate builders publish sealed fields. Teach field extraction and
  // the transitive borrow analysis about those counts on every pass.
  if (["string_split", "string_lines", "string_words", "string_to_list",
    "string_split_on", "string_splitlines", "string_partition", "regex_exec",
    "regex_match_at"].includes(op)) {
    facts_hot(fl, ty ?? tele_unbind(fl.book, (fl.book.tlds[k] as Def).T).ret, true);
  }
  // An intrinsic that installs count cells (blk_new, blk_keep; clone's C
  // too) heats its element type.
  if ("array_get array_get4 array_new array_clone".includes(op)
    && lay_el(fl.book, m.all[0]).ks.includes("box")
    && !(op === "array_new" && facts_packed(fl, m.all[2]))) {
    facts_hot(fl, m.all[0], true);
  }
  if (it.call === true && it.C === undefined) {
    return arr_op(fl, op, lay_el(fl.book, m.all[0]), args);
  }
  const ws = op.startsWith("regex_") ? args.flatMap((v) => val_own(fl, v))
    : args.map((v, i) => (peek.includes(i) || val_own(fl, v), val_word(v)));
  if (Array.isArray(it.C)) {
    const as = ws.map((z) => emit_alias(fl, z, "a"));
    const vs: string[] = [];
    for (const p of it.C) {
      vs.push(emit_alias(fl, tpl(p, [...as, ...vs]), "a"));
    }
    return val_new(vs, lay_of(fl.book, ty ?? tele_unbind(fl.book,
      (fl.book.tlds[k] as Bend.Def).T).ret));
  }
  const dup = typeof it.C === "string" && /\$(\d)[^]*\$\1/.test(it.C);
  let out = tpl(it.C as Gen, dup ? ws.map((a) => emit_alias(fl, a, "a")) : ws);
  // A peek reads now: a later argument or statement may take what it lent.
  if (peek.length > 0) {
    out = emit_hold(fl, [out], "a", [lay_of(fl.book, ty ?? tele_unbind(fl.book,
      (fl.book.tlds[k] as Bend.Def).T).ret).ks[0] ?? "w64"])[0];
    sinks.forEach((v) => val_sink(fl, v));
  }
  const lay = ty === null ? sig_def(fl, k).ret : lay_of(fl.book, ty);
  // a full word is raw, whatever the site knows of its type
  return val_new([out], sig_def(fl, k).ret === X64 ? X64
    : lay.ks.length === 1 ? lay : BOX);
}

// A native's arguments in order, each seeing the later ones as its rest. A
// peeked one is only read: a variable's use is counted but not taken (no
// share, no ownership asked of a borrowed root), and one that dies here, or
// a computed one, is sunk after the call (sinks).
function emit_peek(fl: File, xs: HTerm[], peek: number[], sinks: Val[]): Val[] {
  const rest = fl.rest;
  const vs = xs.map((x, i) => {
    fl.rest = [...xs.slice(i + 1), ...rest];
    const s = term_strip(x);
    if (!peek.includes(i)) {
      return emit_expr(fl, x, null, null);
    }
    const later = s.$ === "Var" && xs.slice(i + 1).some((y) =>
      term_use(term_uses(fl, y), probe_of(s)) > 0);
    if (s.$ !== "Var" || later) {
      // held: the call reads it and the sink drops it, one evaluation
      // (a computed SCon twice would prepend twice onto one tail)
      const v = val_hold(fl, emit_expr(fl, x, null, null), "p");
      sinks.push(v);
      return v;
    }
    const p = probe_of(s);
    const b = bind_of(fl, p);
    if (b.n <= 1) {
      fl.uses.delete(p);
      sinks.push(b.val);
    } else {
      fl.uses.set(p, { ...b, n: b.n - 1 });
    }
    return b.val;
  });
  fl.rest = rest;
  return vs;
}

// A closure moves its captures into a node (each one use of its binding,
// whatever the closure does); its segment takes them, then x. A dropped
// closure drops its node as Terms: a w64 rides boxed.
function emit_clo(fl: File, x: HTerm, ty: HTerm | null): Val {
  const u = term_uses(fl, x);
  const boxed: boolean[] = [];
  const live = [...fl.uses].filter(([p]) => term_use(u, p) > 0)
    .map(([p, b]): [Probe, Bind] => {
      fl.uses.set(p, { ...b, n: b.n - term_use(u, p) + 1 });
      const v = bind_pop(fl, p);
      boxed.push(v.lay === X64);
      return [p, { ...b, val: v.lay === X64 ? val_new([val_box(fl, v)], BOX) : v }];
    });
  const words = live.flatMap(([, b]) => val_own(fl, b.val));
  const name = seg_name(fl, "c");
  const clo = seg_clo(fl, seg_fid(name), words);
  const outer = { seg: fl.seg, uses: fl.uses, spares: fl.spares,
    tab: fl.tab, rest: fl.rest };
  const [arg] = seg_open(fl, name, BOX, null, live, "x", ["w64"], [x], boxed);
  emit_body(fl, x, ty, [], [val_new([arg], BOX)], null);
  Object.assign(fl, outer);
  return val_new([clo], BOX);
}

// A closed SCon chain, including manually written chains. No references
// are evaluated here; open tails stay on the ordinary prepend path.
function str_constant(t: HTerm): number[] | null {
  const cells: number[] = [];
  for (let s = term_strip(t); s.$ === "Ctr";) {
    if (s.k === "SNil") { return cells; }
    if (s.k !== "SCon") { return null; }
    const h = term_strip(s.x[0]);
    const n = h.$ === "Ctr" && h.k === "Chr"
      ? Bend.u32_from_term(h.x[0], "U32") : null;
    if (n === null) { return null; }
    cells.push(n);
    s = term_strip(s.x[1]);
  }
  return null;
}

function str_static(fl: File, cells: number[]): string {
  if (!cells.length) { return "term_pak(CID_SNIL, 0)"; }
  const key = "string:" + cells.join(",");
  const at = memo(fl.lits, key, () => {
    // The narrowest cells the content allows (nar of the C runtime's String).
    const nar = cells.some((c) => c > 65535) ? 0 : cells.some((c) => c > 255) ? 1 : 2;
    const per = 2 << nar, bits = BigInt(64 / per);
    const cls = cls_fit(Math.ceil(cells.length / 2 ** nar));
    const data = fl.img.length;
    for (let i = 0; i < Math.max(2, 2 ** cls) << nar; i += per) {
      let w = 0n;
      for (let j = per - 1; j >= 0; j--) { w = w << bits | BigInt(cells[i + j] ?? 0); }
      fl.img.push(w + "ull");
    }
    const desc = fl.img.length;
    fl.img.push(`term_buf(${cls | nar << 5}, STAT_OFF + ${data})`, `${cells.length}ull`);
    return desc;
  });
  fl.stat.add("SCon");
  return `term_make(TAG_STR, CID_SCON, STAT_OFF + ${at})`;
}

function emit_ctr(fl: File, x: Of<"Ctr">, ty: HTerm | null,
  at: Lay | null): Val {
  const [adt, u] = ctr_adt(fl, x, ty);
  if (u !== null) {
    return val_new([`${u}ull`], W32, true);
  }
  if (adt.k === "String") {
    const cells = str_constant(x);
    if (cells !== null) { return val_new([str_static(fl, cells)], BOX, true); }
  }
  const flds = ctr_flds(fl.book, x.k, x.x);
  // A word type's constructor is its word: a Word's bits packed, a box
  // read, Succ one more (checked), Zero 0.
  if (WORDS[adt.k] !== undefined) {
    const vs = emit_each(fl, flds, null);
    const lay = WORDS[adt.k];
    if (vs.length === 1 && vs[0].ws.length > 1) {
      return val_new([`(${vs[0].ws.map((w, i) => `((u64)${w} << ${i})`)
        .join(" | ")})`], lay);
    }
    if (vs.length === 0) {
      return val_new(["0"], lay, true);
    }
    const w = adt.k === "Nat" ? tpl(tpl_nat("ull", "nat_chk(e, $0 + 1)"),
      [val_word(vs[0])])
      : `term_word(e, ${val_word(vs[0])}, ${lay.ks[0] === "w64" ? 64 : 32})`;
    return val_new([w], lay, /^\d/.test(w));
  }
  if (adt.k === "Array") {
    const vs = emit_each(fl, flds, null);
    return val_new([x.k === "ALeaf"
      ? arr_new(fl, "0", vs[0], lay_el(fl.book, adt.x[0]))
      : `blk_node(e, ${val_own(fl, vs[0])[0]}, ${val_own(fl, vs[1])[0]})`],
    BOX);
  }
  if (fl.hot.has(x.k)) {
    facts_ctr(fl, fl.book.ctrs[x.k], adt.x);
  }
  const pos = at ?? lay_of(fl.book, adt);
  // A folded call is a DAG: a static term emits once per layout.
  const seen = memo(fl.consts, JSON.stringify(pos), () => new Map());
  const got = seen.get(x);
  if (got !== undefined) {
    return got;
  }
  const lay = lay_box(pos) ? lay_node(fl.book, x.k) : pos;
  const arm = lay_arm(lay, x.k);
  const vs = emit_each(fl, flds, arm.fs.map((f) => f.lay));
  const ws = lay.ks.map((_, j) => j === 0 && lay.arms!.length > 1
    ? String(lay.arms!.indexOf(arm)) : "0");
  arm.fs.forEach((f, j) => val_to(fl, vs[j], f.lay).ws.forEach((w, n) => {
    ws[f.at + n] = w;
  }));
  const v = val_new(ws, lay, adt.k !== "String" && vs.every((f) => f.stat));
  const out = lay === pos ? v
    : val_new([ctr_build(fl, x.k, val_own(fl, v), v.stat)], BOX, v.stat);
  if (out.stat) {
    seen.set(x, out);
  }
  return out;
}

function emit_fold(fl: File, t: HTerm): HTerm | null {
  const s = term_strip(t);
  const r = memo(FOLDS, s, () => {
    if (term_const(s)) {
      return s;
    }
    const m = term_spine(fl, s);
    const it = m.t.$ === "Ref" ? intr_of(fl, m.t.k) : undefined;
    if (it === undefined) {
      const b = emit_unfold(fl, s);
      fl.fuel -= b === null ? 0 : term_nodes(fl, b);
      return b === null || term_any(fl, b, (y) => {
        if (y.$ === "App" || y.$ === "Ref") {
          emit_fold(fl, y);
        }
        return fl.fuel < 0;
      }) ? null : b;
    }
    const as = m.all.map((a) =>
      m.args.includes(a) ? emit_fold(fl, a) ?? a : a);
    return it.call === true ? null : as.every((a, i) => a === m.all[i]) ? s
      : as.reduce((f, x) => Bend.App(f, x), m.t as HTerm);
  });
  return r === s ? t : r;
}

function emit_unfold(fl: File, s: HTerm): HTerm | null {
  const m = term_spine(fl, s);
  const d = m.tld;
  if (m.t.$ !== "Ref" || d?.$ !== "Def" || d.h === undefined
    || m.all.length !== d.n
    || intr_of(fl, m.t.k) !== undefined || !flat_of(m.t.k)) {
    return null;
  }
  const fs = m.all.map((a) => m.args.includes(a) ? emit_fold(fl, a) ?? a : a);
  const walk = (ys: HTerm[]): HTerm | null => {
    let b = d.h as HTerm;
    let xs = ys;
    let hit = m.args.every((a) => term_const(fs[m.all.indexOf(a)]));
    for (let w = term_strip(b); xs.length > 0; w = term_strip(b)) {
      if (w.$ === "Lam") {
        b = w.f(xs[0]);
        xs = xs.slice(1);
        continue;
      }
      const c = w.$ === "Mat" ? term_strip(xs[0]) : null;
      if (c === null || c.$ !== "Ctr" || !term_const(c)) {
        return null;
      }
      const { arms, end } = mat_arms(w);
      const arm = arms.find(([k]) => k === c.k);
      b = arm === undefined ? end : arm[1];
      xs = arm === undefined ? xs
        : [...ctr_flds(fl.book, c.k, c.x), ...xs.slice(1)];
      hit = true;
    }
    return !hit || term_any(fl, b, (y) => y.$ === "Lam" || mat_head(y))
      ? null : b;
  };
  const doms = tele_unbind(fl.book, d.T).doms;
  const bind = (i: number, ys: HTerm[]): HTerm => {
    const a = fs[i];
    if (i === fs.length) {
      return walk(ys) as HTerm;
    }
    if (!m.args.includes(m.all[i]) || term_const(a)
      || term_strip(a).$ === "Var") {
      return bind(i + 1, [...ys, a]);
    }
    return Bend.Let(["a"], [0], [Bend.Ann(a, doms[i][2])], (xs: HTerm[]) =>
      bind(i + 1, [...ys, xs[0]]), undefined, [Bend.Many()]);
  };
  return walk(fs) === null ? null : bind(0, []);
}

// Bool.pick(A, c, a, b) over an A that is one raw word (a U32, an F64, a
// Nat): a C select. The generic def would take a and b boxed (an F64 is
// x64_box'd for the call and unboxed after); the arms are converted to A's
// word instead and both evaluated, as the call evaluates them (pick_lift
// has turned the costly ones into a match). An A that holds a box keeps
// the call, which drops the arm it does not choose.
function emit_pick(fl: File, A: HTerm, xs: HTerm[]): Val | null {
  const lay = lay_of(fl.book, A);
  const cl = sig_def(fl, "Bool.pick").lays[0];
  const yes = cl?.arms?.findIndex((a) => a.k === "True") ?? -1;
  if (lay.arms !== null || lay.ks.length !== 1 || lay.ks[0] === "box"
    || yes < 0) {
    return null;
  }
  const [c, a, b] = emit_each(fl, xs, [cl, lay, lay]);
  const cw = val_word(val_to(fl, c, cl));
  const aw = val_word(val_to(fl, a, lay));
  const bw = val_word(val_to(fl, b, lay));
  return val_new([emit_alias(fl, `(${cw} == ${yes} ? ${aw} : ${bw})`, "p",
    lay.ks[0])], lay);
}

function emit_expr(fl: File, tm: HTerm, ty0: HTerm | null,
  at: Lay | null): Val {
  const [x, ty] = ty_peel(tm, ty0);
  switch (x.$) {
    case "Var": return bind_pop(fl, x);
    case "Ref":
    case "App": {
      const got = emit_fold(fl, x);
      if (got !== null && got !== x) {
        const a = term_uses(fl, x);
        const b = term_uses(fl, got);
        fl.uses.forEach((bd, p) => {
          const n = bd.n - term_use(a, p) + term_use(b, p);
          n > 0 ? fl.uses.set(p, { ...bd, n })
            : (fl.uses.delete(p), val_sink(fl, bd.val));
        });
        return emit_expr(fl, got, ty, at);
      }
      const m = term_spine(fl, x);
      if (m.t.$ === "Ref" && m.t.k === "Bool.pick" && m.args.length === 3) {
        const v = emit_pick(fl, m.all[0], m.args);
        if (v !== null) {
          return v;
        }
      }
      if (flat_call(fl, x)) {
        const dst = emit_dst(fl, sig_def(fl, m.call!.k).ret);
        emit_fuse(fl, m.call!, dst);
        return dst;
      }
      // An eta-expansion, or a head applied to erased arguments alone.
      const y = call_eta(fl, x)
        ?? (m.t.$ !== "Ref" && m.args.length === 0 ? m.h : null);
      if (y !== null) {
        return emit_expr(fl, y, ty, at);
      }
      const g = m.t as Of<"Ref">;
      const intr = intr_of(fl, g.k);
      if (intr !== undefined) {
        return emit_intr(fl, intr, x, ty);
      }
      if (m.tld?.$ === "ADT") {
        return emit_zero(fl, ty);
      }
      if (!def_foreign(m.tld)) {
        die(`a live call into the law ${g.k}`);
      }
      // A foreign def short of its continuation: an IO action awaiting it.
      return val_new([seg_clo(fl, seg_fid(g.k),
        emit_each(fl, m.args, m.args.map(() => BOX))
          .map((v) => val_box(fl, v)))], BOX);
    }
    case "Ctr": return emit_ctr(fl, x, ty, at);
    case "Let": {
      const o = term_open(x);
      if (let_live(fl, x)[0]) {
        const rest = fl.rest;
        fl.rest = [o.b, ...rest];
        emit_let(fl, x, o);
        fl.rest = rest;
      }
      return emit_expr(fl, o.b, null, at);
    }
    case "Lam": case "Mat": case "Efq": return fun_live(fl.book, x, ty)
      ? emit_clo(fl, x, ty) : emit_expr(fl, (x as Of<"Lam">).f(DUMMY),
        ty_all(fl.book, ty).B(DUMMY), at);
    case "Hol": die("a hole value");
    default: return emit_zero(fl, ty);
  }
}

function emit_zero(fl: File, ty: HTerm | null): Val {
  const lay = lay_of(fl.book, ty);
  return val_new(lay.ks.map(() => "0ull"), lay);
}

// A let's one value, emitted and bound over its body.
function emit_let(fl: File, x: HLet, o: { ps: Probe[]; b: HTerm }): void {
  const v = val_hold(fl, emit_expr(fl, x.v[0], null, null), x.k[0]);
  bind_uses(fl, o.ps[0], v, [o.b], ty_ann(x.v[0]) ?? die("an unannotated let"));
}

function emit_body(fl: File, tm: HTerm, ty0: HTerm | null,
  ers: HTerm[], args: Val[], dst: Dst): void {
  const [x, ty] = ty_peel(tm, ty0);
  if (args.length === 0 && fun_live(fl.book, x, ty)) {
    return emit_put(fl, dst, emit_clo(fl, x, ty));
  }
  const l = x.$ === "Let" || (args.length === 0 && x.$ !== "Lam")
    ? anf(fl, x, ty) : x;
  if (l !== x) {
    return emit_body(fl, l, ty, ers, args, dst);
  }
  switch (x.$) {
    case "Lam": {
      const all = ty_all(fl.book, ty);
      if (!quant_live(all.q)) {
        const t = ers[0] ?? Bend.Var(x.k, x.i);
        return emit_body(fl, x.f(t), all.B(t), ers.slice(1), args, dst);
      }
      const o = term_open(x);
      const v = val_hold(fl, val_to(fl, args[0], lay_of(fl.book, all.A)), x.k);
      bind_uses(fl, o.ps[0], v, [o.b], all.A);
      return emit_body(fl, o.b, all.B(DUMMY), ers, args.slice(1), dst);
    }
    case "Mat":
    case "Efq": return emit_match(fl, x, ty, ers, args, dst);
    case "Let": {
      if (x.k.length >= 2
        || (call_kind(fl, x.v[0]) !== null && !flat_call(fl, x.v[0]))) {
        return emit_fork(fl, x, ers);
      }
      const o = term_open(x);
      fl.rest = [o.b];
      emit_let(fl, x, o);
      bind_dead(fl, [o.b]);
      return emit_body(fl, o.b, null, ers, [], dst);
    }
    default: {
      if (args.length > 0) {
        return emit_body(fl, term_eta(fl.book, x,
          ty!, 1), ty, ers, args, dst);
      }
      fl.rest = [];
      const ck = call_kind(fl, x);
      if (ck === null) {
        const v = emit_expr(fl, x, ty, dst?.lay ?? fl.seg.ret);
        bind_dead(fl, []);
        return emit_put(fl, dst, v);
      }
      const ret = sig_def(fl, ck.k).ret;
      const once = fl.sites.get(ck.k) === 1 && !ck.bang
        && !def_foreign(fl.book.tlds[ck.k])
        && (!lay_box(ret) || lay_box(fl.seg.ret));
      if (fl.seg.def !== ck.k && (flat_call(fl, x) || (dst === null && once))) {
        return emit_fuse(fl, ck, dst, true);
      }
      // A jump's returns must agree or both be one word (a box holds a word
      // as is); a call whose return disagrees becomes a cut.
      if (!lay_eq(fl.seg.ret, ret)
        && (fl.seg.ret.arms !== null || ret.arms !== null)) {
        return emit_body(fl, Bend.Let(["r"], [0], [Bend.Ann(x,
          ty ?? die("an untyped cut"))], (xs) => xs[0]), ty,
          ers, args, dst);
      }
      const cargs = emit_args(fl, ck, true);
      spare_flush(fl);
      emit_jump(fl, cargs, ck.k, ck.bang);
    }
  }
}

// A fork: in parallel a join task and a kid per call (a cut's continuation
// is a task); in sequence one frame read by every step, the last jumping
// into the joiner. Both paths hold the same values, so open one joiner.
function emit_fork(fl: File, x: HLet, ers: HTerm[]): void {
  const o = term_open(x);
  const calls = x.v.map((v) => call_kind(fl, v) as Call);
  const fork = calls.length > 1;
  const name = seg_name(fl, "j");
  let hold: Probe[] = [];
  if (fork) {
    spare_flush(fl);
    fl.seg.fork = true;
    const uses = new Map(fl.uses);
    block(fl, "if (!seq) {", () => {
      const margs = calls.map((c, j) => {
        fl.rest = [...x.v.filter((_, i) => i !== j), o.b];
        return emit_args(fl, c, false, true);
      });
      const live = [...fl.uses].filter(([p, b]) =>
        !val_brw(fl, b.val) || rest_use(fl, [o.b], p) > 0);
      hold = live.map(([p]) => p);
      const caps = live.flatMap(([, b]) => b.val.ws);
      spare_flush(fl);
      const jn = emit_task(fl, seg_fid(name), calls.length, caps);
      const jt = `term_tsk(${seg_fid(name)}, ${jn})`;
      let idx = caps.length;
      calls.forEach((c, j) => {
        const fj = seg_fid(c.k);
        file_push(fl, `e.mem[${jn} + ${idx}] = term_tsk(${fj}, ${
          emit_task(fl, fj, 0, margs[j], jt, idx)});`);
        idx += sig_def(fl, c.k).ret.ks.length;
      });
      file_push(fl, `return ${jt};`);
    });
    fl.uses = uses;
  }
  // What follows call i: the later calls' lets over the body.
  const chain = calls.map(() => o.b);
  for (let j = calls.length - 2; j >= 0; j -= 1) {
    chain[j] = let_open([o.ps[j + 1]], [x.v[j + 1]], chain[j + 1]);
  }
  const pos = new Map<Probe, number>();
  let depth = 0;
  calls.forEach((c, i) => {
    fl.rest = [chain[i]];
    const cargs = emit_args(fl, c);
    const vs = i === 0 ? [...fl.uses]
      : [[o.ps[i - 1], fl.uses.get(o.ps[i - 1]) as Bind] as [Probe, Bind]];
    const kn = seg_name(fl, "k");
    spare_flush(fl);
    const ws = vs.flatMap(([p, b]) =>
      (pos.set(p, depth), depth += b.val.ws.length, b.val.ws));
    emit_chain(fl, () => "seq", [() => emit_frame(fl, ws, seg_fid(kn)),
      ...fork ? [] : [() => {
        file_push(fl, `WL_CONT = term_tsk(${seg_fid(kn)}, ${
          emit_task(fl, seg_fid(kn), 1, ws)});`);
        file_push(fl, `WL_IDX = ${ws.length};`);
      }]]);
    emit_jump(fl, cargs, c.k, !fork && c.bang);
    const last = i === calls.length - 1;
    const held = [...fl.uses].filter(([p]) => pos.has(p));
    const at = held.flatMap(([p, b]) => b.val.ws.map((_, j) =>
      (pos.get(p) as number) + j - (last ? 0 : depth)));
    const ret = sig_def(fl, c.k).ret;
    const rest = [...hold, chain[i]];
    const rs = seg_open(fl, kn, fl.seg.ret, { pop: last ? depth : 0, at },
      held, o.ps[i].k, ret.ks, rest);
    bind_uses(fl, o.ps[i], val_new(rs, ret), rest,
      ty_ann(x.v[i]) ?? die("an unannotated let"));
  });
  if (fork) {
    const live = [...fl.uses];
    if (live.map(([p]) => p.i).join() !== [...hold, ...o.ps].map((p) => p.i)
      .join()) {
      die("a fork's paths hold different values");
    }
    emit_jump(fl, live.flatMap(([, b]) => b.val.ws), name);
    seg_open(fl, name, fl.seg.ret, null, live, "", [], [o.b]);
  }
  emit_body(fl, o.b, null, ers, [], null);
}

// A row is JS text; a C-lane F32 row is its bits (a NaN payload has no JS
// number): a constant's own, an intrinsic's through f32_bits.
function emit_row(fl: File, t: HTerm, ty: HTerm | null): string | null {
  const k = ty_adt(fl.book, ty)?.k ?? "";
  if (ty !== null && WORDS[k] === undefined) {
    return null;
  }
  let s = term_strip(t);
  while (s.$ === "Lam") {
    s = term_strip(term_open(s).b);
  }
  s = emit_fold(fl, s) ?? s;
  const bits = k === "F32" && !fl.js;
  if (term_const(s)) {
    return bits ? String(Bend.u32_from_term(s, "F32")) : js_expr(fl, s, ty);
  }
  const m = term_spine(fl, s);
  const it = m.t.$ === "Ref" ? intr_of(fl, m.t.k) : undefined;
  if (it === undefined || typeof it.JS !== "string" || TAB_BAD.test(it.JS)) {
    return null;
  }
  const xs = m.args.map((a) => emit_row(fl, a, null));
  const r = xs.includes(null) ? null : tpl(it.JS, xs as string[]);
  return r !== null && bits ? `f32_bits(${r})` : r;
}

// The table a match reads at `s`, when it has rows and every row is a
// constant: the read clamps past the last row.
function emit_tab(fl: File, rows: Chain | null, ty: HTerm,
  s: string): string | null {
  const ls = (rows ?? []).map(([t]) => emit_row(fl, t, ty));
  if (rows === null || ls.includes(null)) {
    return null;
  }
  const key = fl.js ? ls.join(", ") : Function("f32_bits",
    "return [" + ls + "]")(Bend.f32_to_bits).map((v: number) => BigInt(v)
    + "ull").join(", ");
  const tab = "TAB_" + memo(fl.tabs, key, () => fl.tabs.size);
  return fl.js ? `${tab}[Math.min(Number(${s}), ${rows.length - 1})]`
    : `TAB_AT(${tab}, ${s}, ${rows.length - 1})`;
}

// A Nat match's rows: Succ counted down its chain, each row a case or
// the level's default with its residual.
function emit_nats(adt: HAdt, x: HTerm): Chain | null {
  if (adt.k !== "Nat") {
    return null;
  }
  const ls: Chain = [];
  for (let m = x, n = 0; ; n++) {
    const { arms, end } = mat_arms(m);
    const { Zero, Succ } = Object.fromEntries(arms);
    ls.push([Zero ?? end, Zero ? null : n]);
    m = term_strip(Succ ?? end);
    if (Succ === undefined || m.$ !== "Mat") {
      return [...ls, [Succ ?? end, Succ ? n + 1 : n]];
    }
  }
}

// A word match's leaves: the bits each knows (32: a hit), their value, its
// arm and the words it binds. A default covers the deeper ones that are its
// instance (the flattener's substitution replayed).
function emit_lits(adt: HAdt, x: HTerm): Leaf[] | null {
  if (WORDS[adt.k] !== W32) {
    return null;
  }
  const ws: Leaf[] = [];
  const key = (t: HTerm): string => JSON.stringify(Bend.term_lower(t),
    (k, v) => k === "s" ? undefined : v?.$ === "Ann" ? Bend.term_strip(v) : v);
  const walk = (t: HTerm, j: number, n: number,
    cov: ((w: Of<"Ctr">) => HTerm) | null): void => {
    const h = mat_arms(t).arms[0]?.[1];
    if (h !== undefined && j === 32) {
      ws.push([h, j, n, 0]);
      return;
    }
    const { arms, end } = mat_arms(h ?? t);
    const e = h === undefined ? 1 : 2;
    const inst = (w: Of<"Ctr">): HTerm => (e === 1 ? [w] : w.x)
      .reduce((f, a) => Bend.term_apply(f, a), end);
    const w = Bend.Ctr("WCon", [probe("b"), probe("t")]) as Of<"Ctr">;
    const own = arms.length < 2 && (cov === null
      || key(cov(w)) !== key(inst(w)));
    const sub = own ? inst : cov;
    for (const [k, a] of arms) {
      walk(a, j + 1, n + (k === "True" ? 2 ** j : 0), sub && ((v) =>
        sub(Bend.Ctr("WCon", [Bend.Ctr(k, []), v]) as Of<"Ctr">)));
    }
    if (own) {
      ws.push([end, j, n, e]);
    }
  };
  walk(mat_arms(x).arms[0][1], 0, 0, null);
  return ws;
}

// A U32 table: the hits over half of 0..max, and one same row from every
// other leaf (a gap and the clamp read it).
function lits_rows(fl: File, adt: HAdt, ws: Leaf[] | null,
  ty: HTerm): Chain | null {
  if (adt.k !== "U32" || ws === null) {
    return null;
  }
  const hit = new Map(ws.flatMap(([h, j, n]) => j === 32 ? [[n, h]] : []));
  const out = ws.filter(([, j]) => j < 32);
  const rs = new Set(out.map(([o]) => emit_row(fl, o, ty)));
  const len = Math.max(-1, ...hit.keys()) + 1;
  return hit.size * 2 > len && rs.size === 1 ? [...Array(len + 1)]
    .map((_, i): Chain[number] => [hit.get(i) ?? out[0][0], null]) : null;
}

function lits_cond(w: string, j: number, n: number): string {
  return j === 32 ? `${w} == ${n}` : `(${w} & ${2 ** j - 1}) == ${n}`;
}

// A match's constructor arms, and its default as a "" arm when some
// constructor has none, or always when `keep`.
function mat_ctrs(fl: File, x: HTerm, adt: HAdt,
  keep = false): [Name, HTerm][] {
  const { arms, end } = mat_arms(x);
  return keep || arms.length < Bend.book_adt(fl.book, adt, Bend.Emp()).c.length
    ? [...arms, ["", end]] : arms;
}

function emit_match(fl: File, x: Of<"Mat"> | Of<"Efq">,
  ty: HTerm | null, ers: HTerm[], args: Val[], dst: Dst): void {
  if (x.$ === "Efq") {
    return emit_stuck(fl);
  }
  const all = ty_all(fl.book, ty);
  const adt = adt_of(fl.book, all.A);
  const ret = all.B(DUMMY);
  const ls = emit_nats(adt, x);
  const ws = emit_lits(adt, x);
  // A word is matched on its bits, a 64-bit word (U64, I64, F64) on its one
  // constructor's node at its own width, anything else in its own layout.
  const w64 = adt.k === "U64" || adt.k === "I64" || adt.k === "F64";
  const lay = ws || w64 ? lay_node(fl.book, adt.k) : lay_of(fl.book, all.A);
  const u = val_hold(fl, val_to(fl, args[0],
    ws ? W32 : w64 ? WORDS[adt.k] : lay), "s");
  const sw = u.ws[0];
  const tab = emit_tab(fl, ls ?? lits_rows(fl, adt, ws, ret), ret, sw);
  if (tab !== null) {
    bind_dead(fl, []);
    return emit_put(fl, dst, val_new([tab], lay_of(fl.book, ret)));
  }
  // Each level: its condition, its arm, and the fields the arm binds (read
  // inside the arm).
  let lv: [string, HTerm, () => Val[]][];
  if (ls !== null) {
    lv = ls.map(([h, n], i) => [`${sw} == ${i}`, h,
      () => n === null ? [] : [val_new([`(${sw} - ${n})`], lay)]]);
  } else if (ws !== null) {
    // the word at depth j, or its bit there and the tail
    lv = ws.map(([h, j, n, e]) => [lits_cond(sw, j, n), h, () => {
      let v = val_new(lay.ks.map((_, i) => `((${sw} >> ${i}) & 1)`),
        lay.arms![0].fs[0].lay);
      for (let i = 0; i < j; i++) {
        v = val_field(v, v.lay.arms![0].fs[1]);
      }
      return e === 1 ? [v]
        : v.lay.arms![0].fs.slice(0, e).map((f) => val_field(v, f));
    }]);
  } else if (w64) {
    // a 64-bit word's one constructor: its bits, exploded into the node
    const b = val_hold(fl, val_new(lay.ks.map((_, i) => `((${sw} >> ${i}) & 1)`),
      lay), "s");
    lv = mat_ctrs(fl, x, adt).map(([k, h]) => ["", h, () => k === ""
      ? [u] : lay.arms![0].fs.map((f) => val_field(b, f))]);
  } else {
    lv = mat_ctrs(fl, x, adt, adt.k === "IO.OP").map(([k, h]) => {
      if (k === "") {
        return ["", h, () => [u]];
      }
      if (adt.k === "Array") {
        const el = lay_el(fl.book, adt.x[0]);
        const leaf = k === "ALeaf";
        return [`blk_cls(${sw}) ${leaf ? "==" : "!="} ${lay_arr(el).lgs}`, h,
          () => {
            val_own(fl, u);
            return leaf ? [arr_leaf(fl, sw, el)] : emit_hold(fl,
              [0, 1].map((hi) => `blk_half(e, ${sw}, ${hi})`), "h")
              .map((w) => val_new([w], BOX));
          }];
      }
      if (lay_box(lay)) {
        // TAG_STR carries the logical SCon constructor id; extraction
        // below routes through str_uncons instead of reading cons fields.
        return [`term_aux(${sw}) == ${cid_mac(k)}`, h,
          () => node_fields(fl, sw, lay_node(fl.book, k), true)];
      }
      const arm = lay_arm(lay, k);
      return [`${sw} == ${lay.arms!.indexOf(arm)}`, h,
        () => arm.fs.map((f) => val_field(u, f))];
    });
  }
  // Each arm starts from the match's state; a tail arm keeps its spares.
  const spares = fl.spares;
  emit_chain(fl, (i) => lv[i][0], lv.map(([, h, fs]) => () => {
    const outer = { seg: fl.seg, tab: fl.tab, uses: new Map(fl.uses),
      spares: dst === null ? spares : [] };
    fl.spares = spares.slice();
    bind_dead(fl, [h]);
    emit_body(fl, h, null, ers, [...fs(), ...args.slice(1)], dst);
    if (dst !== null) {
      spare_flush(fl);
    }
    Object.assign(fl, outer);
  }));
}

function emit_stuck(fl: File): void {
  file_push(fl, "err_post(e.mem, ERR_TAGS);");
  file_push(fl, "return 0;");
}

function emit_chain(fl: File, cond: (i: number) => string,
  bodies: (() => void)[]): void {
  if (bodies.length === 1) {
    return bodies[0]();
  }
  bodies.forEach((body, i) => {
    if (i === bodies.length - 1) {
      file_push(fl, "} else {");
    } else {
      file_push(fl, `${i === 0 ? "if" : "} else if"} (${cond(i)}) {`);
    }
    fl.tab += 1;
    body();
    fl.tab -= 1;
  });
  file_push(fl, "}");
}

// Compile
// =======

function compile_def(fl: File, k: Name, tld: Def): void {
  Object.assign(fl, { fresh: new Map(), brwl: new Map(), rest: [] });
  memo_gc();
  const vals = emit_open(fl, k);
  fl.segs.push(fl.seg);
  emit_body(fl, tld.h as HTerm, tld.T, [], vals, null);
}

function compile_reqs(fl: File): void {
  const seen = new Set<string>();
  fl.spares = [];
  for (const [k, tld] of done_defs(fl, def_foreign)) {
    const ns = k.slice(0, k.length - LOCAL.get(k)!.length);
    const own = ns === "" ? []
      : Object.keys(fl.book.ctrs).filter((c) => c.startsWith(ns));
    const macs = own.map((c) => cid_mac(c.slice(ns.length)));
    // A shared source is spliced once, its namespace's names in scope.
    const path = fs.realpathSync(tld.i!.find((x) => x.endsWith(".c"))
      ?? die("no .c import: " + k));
    fl.reqs += macs.map((m, j) => `#pragma push_macro("${m}")\n#define ${m} ${
      cid_mac(own[j])}\n`).join("") + (seen.has(path) ? ""
      : (seen.add(path), fs.readFileSync(path, "utf8")))
      + macs.map((m) => `#pragma pop_macro("${m}")\n`).join("");
    const qp = [...sig_def(fl, k).live.map(([, n]) => n), "k"].map((n) =>
      name_local(fl, n));
    fl.seg = seg_new(k, BOX, qp);
    fl.segs.push(fl.seg);
    fl.cids.set(k, qp.length);
    file_push(fl, `r0 = ${ctr_build(fl, k, qp)};`);
    file_push(fl, "WL_RETN(1);");
  }
}

// The datatypes whose constructors the runtime or the elaborator lays itself.
const RUNTIME_ADTS = ["Sigma", "String", "Word.Con", "Word.Nil", "IO.OP",
  "Result", "Maybe", "Bool", "Unit", "List", "Char", "Cmp", "Inst", "Match"];

// Names the compiler encodes itself: SYNTH no file may declare; OWNED adds
// Base's types, which a file without `import Base` may declare but not
// compile.
export const SYNTH = [CLO_APPLY];
const OWNED = [...SYNTH, "IO", ...RUNTIME_ADTS, ...Object.keys(OPTIMIZED)];

export function book_owned(src: Bend.Book, ks = OWNED): void {
  for (const k of ks) {
    if (src.tlds[k] !== undefined && src.tlds[k].b !== true) {
      die(k + " is a name the compiler encodes itself: name yours apart");
    }
  }
}

function compile_tables(fl: File, entries: Seg[]): string[] {
  // A segment may fork (or bang) if it or one it reaches does; a closure
  // apply reaches every closure.
  const forky = new Set(fl.segs.filter((s) => s.fork).map((s) => s.fid));
  for (let n = -1; n !== forky.size;) {
    n = forky.size;
    for (const s of [...fl.segs, { fid: "FID_CLO_APPLY", refs: fl.clos }]) {
      if (!forky.has(s.fid) && [...s.refs].some((r) => forky.has(r))) {
        forky.add(s.fid);
      }
    }
  }
  const tabs: [string, number[]][] = [
    ["FID_ARITY_T", entries.map((s) => s.params.length)],
    ["FID_FLAG_T", entries.map((s) => Number(fl.bangs.has(s.def))
      | Number(!forky.has(s.fid)) << 1)],
    ["FID_RESW_T", entries.map((s) =>
      s.frame === null ? 0 : s.params.length - s.frame.at.length)],
    ["CID_ARITY_T", [...fl.cids.values()]],
    ["CID_HOT_T", [...fl.cids.keys()].map((k) => Number(fl.hot.has(k)))],
  ];
  const defs: string[] = [];
  for (const ms of [[...fl.cids.keys()].map(cid_mac),
    [...entries.map((s) => s.fid), "FID_EXIT", "FID_ENTER"]]) {
    const dup = [...ms, ...tabs.map((t) => t[0])].find((m, i, a) =>
      a.indexOf(m) < i);
    if (ms.length > 65536 || dup !== undefined) {
      die(dup === undefined ? "an id over 65535"
        : "two names mangle to " + dup);
    }
    defs.push(...ms.map((m, i) => `#define ${m} ${i}`));
  }
  for (const [nm, vals] of tabs) {
    if (vals.some((v) => v > 255)) {
      die("an arity over 255");
    }
    defs.push(`CONSTV u8 ${nm}[] = { ${vals.join(", ")} };`);
  }
  defs.push(`#define STAT_LEN ${fl.img.length}`, "");
  // One bank for both lanes, as wide as the widest segment or return; rp
  // pads the host's twelfth slot, keeping rax free for the tail call.
  const resw = Math.max(...entries.map((s) => s.ret.ks.length));
  const n = Math.max(resw, ...entries.filter((s) => s.frame === null)
    .map((s) => s.params.length));
  const rs = [...Array(n).keys()].map((i) => "r" + i);
  const ws = n > 6 ? [...rs.slice(0, 6), "rp", ...rs.slice(6)] : rs;
  // a ladder: a fallthrough switch's phi cascade costs clang O(n^2) to build
  const load = rs.map((r, i) =>
    `    if ((N) <= ${i}) break; ${r} = e.mem[(A) + ${i}]; \\\n`).join("");
  const last = rs.map((r, i) =>
    `    case ${i}: ${r} = (X); \\\n      break; \\\n`).join("");
  defs.push(`#define WL_RESW ${resw}`, `#define BANGS   ${fl.bangs.size}`, "",
  `#define WL_BANK Term ${ws.join(", ")};`, "",
  `#define WL_LOAD(A, N) \\\n  do { \\\n${load}  } while (0);`, "",
  `#define WL_LAST(X) \\\n  switch (war) { \\\n${last}  }`, "",
  `#define WL_SAVE(V) ${rs.slice(0, resw).map((r, j) =>
    `(V)[${j}] = ${r};`).join(" ")}`, "",
  `#define WL_TAKE(V) ${rs.slice(0, resw).map((r, j) =>
    `${r} = (V)[${j}];`).join(" ")}`, "",
  `#define WL_SIG Env e, Stk sp, u32 seq, u32 rn, ${ws.map((w) =>
    "Term " + w).join(", ")}`, "", `#define WL_ALL e, sp, seq, rn, ${ws
    .join(", ")}`, "",
  `#define WL_TABLE ${entries.map((s) => `WL_X(${s.fid})`).join(" ")}`
    + " WL_X(FID_EXIT)");
  return defs;
}

function compile_segs(fl: File): string {
  return fl.segs.map((seg) => {
    const out = [`  WL_CASE(${seg.fid})`, "  {",
      ...seg_take(seg).map((l) => "    " + l),
      "    WL_OPEN", ...seg.spin ? ["    WL_SPIN"] : [],
      ...seg.lines, ...seg.spin ? ["    WL_SPUN"] : [], "  }}"];
    return (seg.host ? ["#if !DEVICE", ...out, "#endif"] : out).join("\n");
  }).join("\n\n");
}

export function compile_book(book: Bend.Book): string {
  const cb = carb_book(book, ["main", ...RUNTIME_ADTS]);
  const show = show_main(book);
  const facts = () => cb.own.size + cb.hot.size + cb.stat.size;
  const pass = (defs: [Name, Def][]): File => {
    [cb.lend, BRWS].forEach((m) => m.clear());
    const fl = file_new(cb, false);
    for (const k of SRCS.keys()) {
      for (const c of (cb.book.tlds[k] as Bend.ADT).c ?? []) {
        fl.cids.set(c.k, lay_node(fl.book, c.k).ks.length);
      }
    }
    for (const [k, tld] of defs) {
      compile_def(fl, k, tld);
    }
    compile_reqs(fl);
    facts_lend(cb);
    return fl;
  };
  // Callees emit first, pass after pass until nothing changes (own, hot
  // and stat only grow); that pass is kept.
  let fl: File;
  let was: number;
  do {
    was = facts();
    fl = pass(done_defs(cb).reverse());
  } while (was !== facts());
  const reach = (from: string[], set = new Set<string>()): Set<string> => {
    const grab = (fid: string) => set.has(fid) || (set.add(fid)
      && [...fl.segs, ...fl.spins].find((s) => s.fid === fid)?.refs
        .forEach(grab));
    from.forEach(grab);
    return set;
  };
  const live = reach([seg_fid("main")]);
  // The device holds what the bangs reach, and every closure when a bang's
  // parameter may hold one (a jump through its fid).
  const wide = [...fl.bangs].some((k) =>
    sig_def(fl, k).live.some(([, , A]) => ty_clo(fl.book, A)));
  const dev = reach([...[...fl.bangs].map(seg_fid), ...wide ? fl.clos : []]);
  fl.segs = fl.segs.filter((s) => live.has(s.fid));
  for (const s of fl.segs) {
    s.host = !dev.has(s.fid);
  }
  fl.spins = fl.spins.filter((s) => live.has(s.fid));
  const entries = [...fl.segs, seg_new("io_emit", BOX, [""]),
    seg_new("clo_apply", BOX, ["", ""])];
  const desc = show === null ? [] : ["#if !DEVICE",
    `static const u32 SHOW_DESC[] = { ${show.cells.map((c) =>
      typeof c === "string" ? cid_mac(c) : c).join(", ")} };`,
    `static const char* SHOW_NAMES[] = { ${show.names.map((n) =>
      JSON.stringify(n)).join(", ")} };`, "#endif"];
  const defs = compile_tables(fl, entries);
  defs.push(`#define MAIN_FID ${seg_fid("main")}`, `#define MAIN_PURE ${
    Number(show !== null)}`,
    `#define BLK_SHR ${Number(cb.hot.has("t:Array"))}`);
  const tabs = [defs.join("\n"), ...[...fl.tabs].map(([r, i]) =>
    `CONSTV u64 TAB_${i}[] = { ${r} };`)].join("\n\n");
  const spins = [`CONSTV u64 STAT_IMG[] = { ${fl.img.join(", ") || 0} };`,
    ...fl.spins.map((s) => s.text)].join("\n\n");
  const segs = compile_segs(fl);
  if (/\bundefined\b/.test([tabs, spins, segs, fl.reqs].join("\n"))) {
    die("an unbound name in the emitted C");
  }
  return runtime_c([tabs, ...desc].join("\n\n"), spins, segs, fl.reqs);
}

// Js
// ==

function js_sat(k: Name): string {
  return "$" + k.replace(/\W/g, "$") + "$";
}

function js_call(fl: File, k: Name, args: HTerm[],
  tail: boolean): string {
  let exprs = args.map((x) => js_expr(fl, x, null));
  if (k === CLO_APPLY) {
    const [f, x] = exprs;
    return tail ? "run_tail(" + f + ", " + x + ")" : f + "(" + x + ")";
  }
  const tld = fl.book.tlds[k];
  if (tld.$ === "ADT") {
    return "null";
  }
  const intr = intr_of(fl, k, true)?.JS ?? null;
  if (intr === null && tld.v === null && tld.i === undefined) {
    die("a live call into the law " + k);
  }
  // A foreign def short of its continuation: an IO action awaiting it.
  const live = sig_def(fl, k).lays.length;
  const v = def_foreign(tld) && exprs.length === live - 1
    ? name_local(fl, "x") : "";
  if (v !== "") {
    exprs = [...exprs, v];
  } else if (exprs.length !== live) {
    die("an under-applied def value: " + k);
  }
  if (intr !== null) {
    const xs = exprs.map((e) => ATOM.test(e) || STRLIT.test(e)
      ? e : emit_hold(fl, [e], "x")[0]);
    return tpl(intr, xs);
  }
  const call = js_sat(k) + "(" + exprs.join(", ") + ")";
  if (v !== "") {
    return "(" + v + ") => " + call;
  }
  if (def_foreign(tld)) {
    return call;
  }
  return tail ? "run_jump(" + js_sat(k) + ", [" + exprs.join(", ") + "])"
    : "run_loop(" + call + ")";
}

function js_open(fl: File, x: HLet): HTerm {
  const on = let_live(fl, x);
  return x.f(x.v.map((v, j): HTerm => !on[j] ? v
    : Bend.Var(emit_hold(fl, [js_expr(fl, v, null)], x.k[j])[0], 0)));
}

function js_ctr(fl: File, k: Name): Name[] {
  return ctr_tail(fl.book, fl.book.ctrs[k]).filter(live_dom).map(([, n]) => n);
}

function js_expr(fl: File, tm: HTerm,
  ty0: HTerm | null): string {
  const [x, ty] = ty_peel(tm, ty0);
  switch (x.$) {
    case "Var": return x.k;
    case "Ref":
    case "App": {
      const ck = call_kind(fl, x);
      if (ck !== null) {
        return js_call(fl, ck.k, ck.args, false);
      }
      const m = term_spine(fl, x);
      const y = call_eta(fl, x) ?? (m.t.$ !== "Ref" ? m.h : null);
      if (y !== null) {
        return js_expr(fl, y, ty);
      }
      const k = (m.t as Of<"Ref">).k;
      const it = intr_of(fl, k, true);
      // an array intrinsic's element layout, which refuses an open type
      if (it?.call === true && it.C === undefined) {
        lay_el(fl.book, m.all[0]);
      }
      return js_call(fl, k, m.args, false);
    }
    case "Ctr": {
      const [adt, u] = ctr_adt(fl, x, ty);
      if (u !== null) {
        const v = adt.k === "F32" ? Bend.f32_from_bits(u) : u;
        return Object.is(v, -0) ? "-0" : String(v);
      }
      if (adt.k === "String") {
        const cells = str_constant(x);
        if (cells !== null && cells.every((c) => c < 0xd800 || c > 0xdfff && c <= 0x10ffff)) {
          return JSON.stringify(cells.map((c) => String.fromCodePoint(c)).join(""));
        }
      }
      const exprs = ctr_flds(fl.book, x.k, x.x)
        .map((f) => js_expr(fl, f, null));
      const native = OPTIMIZED[adt.k];
      if (native !== undefined) {
        return tpl(native[x.k].intr, exprs);
      }
      const keys = js_ctr(fl, x.k);
      return exprs.reduce((e, z, j) => e + ", [\"" + keys[j] + "\"]: " + z,
        "{$: \"" + name_own(x.k, fl.book.tlds[adt.k], " +") + "\"") + "}";
    }
    case "Let": return js_expr(fl, js_open(fl, x), ty);
    case "Lam": case "Mat": case "Efq": {
      if (!fun_live(fl.book, x, ty)) {
        return js_expr(fl, (x as Of<"Lam">).f(Bend.Var("null", 0)),
          ty_all(fl.book, ty).B(DUMMY));
      }
      const arg = name_local(fl, "x");
      const seg = fl.seg;
      fl.seg = seg_new("", BOX, []);
      js_func(fl, x, ty, [arg]);
      const lines = fl.seg.lines;
      fl.seg = seg;
      return `run_clo((${arg}) => {\n${lines.join("\n")}\n})`;
    }
    case "Hol": die("cannot compile a hole");
    default: return "null";
  }
}

function js_func(fl: File, tm: HTerm, ty0: HTerm | null,
  args: string[]): void {
  const [x, ty] = ty_peel(tm, ty0);
  if (args.length === 0 && fun_live(fl.book, x, ty)) {
    return file_push(fl, "return " + js_expr(fl, x, ty) + ";");
  }
  if (x.$ === "Lam") {
    const all = ty_all(fl.book, ty);
    const v: HTerm = Bend.Var(!quant_live(all.q) ? "null"
      : emit_alias(fl, args[0], x.k), 0);
    return js_func(fl, x.f(v), all.B(v),
      quant_live(all.q) ? args.slice(1) : args);
  }
  if (mat_head(x)) {
    return js_match(fl, x, ty, args);
  }
  if (x.$ === "Let") {
    return js_func(fl, js_open(fl, x), ty, args);
  }
  if (args.length > 0) {
    return js_func(fl, term_eta(fl.book, x, ty!, 1), ty, args);
  }
  const ck = call_kind(fl, x);
  file_push(fl, "return " + (ck === null ? js_expr(fl, x, ty)
    : js_call(fl, ck.k, ck.args, true)) + ";");
}

function js_match(fl: File, x: HTerm, ty: HTerm | null,
  args: string[]): void {
  if (x.$ === "Efq") {
    return file_push(fl, `throw "bend: ${ERRS[2]}";`);
  }
  const all = ty_all(fl.book, ty);
  const adt = adt_of(fl.book, all.A);
  const ret = all.B(DUMMY);
  const ls = emit_nats(adt, x);
  const ws = emit_lits(adt, x);
  const s = emit_alias(fl, args[0], "$t");
  // A foreign request reaching a match over IO.OP is refused, unless a
  // default arm takes it.
  if (adt.k === "IO.OP" && mat_arms(x).arms.length
    >= Bend.book_adt(fl.book, adt, Bend.Emp()).c.length) {
    block(fl, `if (${s}.$ === "$FFI") {`, () => file_push(fl, `throw ${s};`));
  }
  const tab = emit_tab(fl, ls ?? lits_rows(fl, adt, ws, ret), ret, s);
  if (tab !== null) {
    return file_push(fl, `return ${tab};`);
  }
  // Each level: its condition, its arm, and the fields the arm binds.
  let lv: [string, HTerm, string[]][];
  if (ls !== null) {
    lv = ls.map(([h, n], i) => [`${s} === ${i}n`, h,
      n === null ? [] : [`(${s} - ${n}n)`]]);
  } else if (ws !== null) {
    // the word at depth j, or its bit there and the tail
    const bits = adt.k === "F32" ? `f32_bits(${s})` : s;
    const wd = (j: number): string =>
      `u32_to_word(${bits})` + "[\"tail\"]".repeat(j);
    lv = ws.map(([h, j, n, e]) => [lits_cond(bits, j, n), h, e === 1
      ? [wd(j)] : [wd(j) + "[\"head\"]", wd(j + 1)].slice(0, e)]);
  } else {
    const native = OPTIMIZED[adt.k];
    lv = mat_ctrs(fl, x, adt).map(([k, h]): [string, HTerm, string[]] => {
      if (k === "") {
        return ["", h, [s]];
      }
      if (native === undefined) {
        const tag = name_own(k, fl.book.tlds[adt.k], " +");
        return [`${s}.$ === "${tag}"`, h,
          js_ctr(fl, k).map((f) => `${s}["${f}"]`)];
      }
      return [tpl(native[k].cond ?? "", [s]), h,
        (native[k].elim ?? []).map((e) => tpl(e, [s]))];
    });
  }
  emit_chain(fl, (i) => lv[i][0], lv.map(([, h, fs]) => () =>
    js_func(fl, h, null, [...fs, ...args.slice(1)])));
}

function js_def(fl: File, k: Name, def: Def): void {
  fl.fresh = new Map();
  fl.fuel = FOLD_FUEL;
  if (intr_of(fl, k, true) !== undefined) {
    return;
  }
  const params = sig_def(fl, k).live.map(([, n]) => name_local(fl, n));
  const kont = def.i ? [name_local(fl, "k")] : [];
  block(fl, `function ${js_sat(k)}(${[...params, ...kont].join(", ")}) {`,
    () => {
      if (def.i === undefined) {
        js_func(fl, def.h!, def.T, params);
      } else {
        const n = eff_name(k);
        file_push(fl, `return { $: "$FFI", run: $0eff.${n}.run, need: $0eff.${n
          }.need, args: [${params.join(", ")}], kont: ${kont[0]} };`);
      }
    });
  file_push(fl, "");
}

export function js_lib(book: Bend.Book, roots: Name[],
  outs: Name[] | null): string {
  const cb = carb_book(book, roots.slice());
  const fl = file_new(cb, true);
  fl.tab = 0;
  const ms = done_defs(cb).map(([k]) => js_sat(k));
  for (const [k, def] of done_defs(cb)) {
    memo_gc();
    js_def(fl, k, def);
  }
  const grps = new Map<string, string[]>();
  for (const [k, tld] of done_defs(cb, def_foreign)) {
    const path = fs.realpathSync(tld.i!.find((x) => x.endsWith(".js"))
      ?? die("a foreign def without a .js import: " + k));
    js_def(fl, k, tld);
    const n = eff_name(k);
    ms.push(n);
    memo(grps, path, () => []).push(`  ["${n}"]: { run: typeof ${n} === "function" ? ${n} : undefined,`
      + ` need: typeof ${n}_need === "function" ? ${n}_need : undefined },`);
  }
  const dup = ms.find((m, i) => ms.indexOf(m) < i);
  if (dup !== undefined) die("two names mangle to " + dup);
  const effs = grps.size === 0 ? "" : "const $0eff = {\n" + [...grps]
    .map(([p, rows]) => "...(() => {\n" + fs.readFileSync(p, "utf8")
      + "\nreturn {\n" + rows.join("\n") + "\n};\n})(),").join("\n")
    + "\n};\n\n";
  const tabs = [...fl.tabs].map(([r, i]) => `const TAB_${i} = [${r}];`);
  const lib = outs === null ? "" : "export default {\n" + outs.map((k) =>
    `  "${k}": run_lib(${js_sat(k)}, ${sig_def(cb, k).lays.length}),`)
    .join("\n") + "\n};\n";
  return RUNTIME + effs + "// Program\n// =======\n\n"
    + [...fl.seg.lines, ...tabs].join("\n") + lib;
}

export function js_book(book: Bend.Book): string {
  const lib = js_lib(book, ["main"], null);
  const show = show_main(book);
  return lib + "\n" + RUNTIME_MAIN
    + "\ncli(process.argv.slice(2));\nio_exit(" + js_sat("main") + ", "
    + JSON.stringify(show && [show.cells.map((c) => typeof c === "string"
      ? 0 : c), show.names]) + ");";
}

// RuntimeC
// ========

function a32_ops(f: (k: string) => string): string {
  return ["add", "sub", "and", "or", "xor", "min", "max"].map((k) =>
    `#define a32_${k}(p, v) ${f(k)}`).join("\n");
}

// The runtime around the program's tables, spins, segments and requests.
const runtime_c = (tabs: string, spins: string, segs: string,
  reqs: string): string => String.raw`

// Imports
// =======

#pragma clang fp contract(off)

// the device dialect: a second RTC lane adds its macro here
#if defined(__CUDACC_RTC__)
#define BEND_RTC 1
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#elif !defined(BEND_RTC)
#ifdef __APPLE__
#define _DARWIN_UNLIMITED_SELECT
#else
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <poll.h>
#include <sys/select.h>
#include <sys/wait.h>
#ifdef __linux__
#include <sys/epoll.h>
#include <sys/prctl.h>
#endif
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef __OBJC__
// #include, not #import: bend -o reads an #import as an effect's framework
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#elif BEND_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif
#endif

// Dialect
// =======

#ifdef __METAL_VERSION__
// coherent(device) (MSL 3.2): M1-class parts else lose stores across a
// dispatch's threadgroups
#if __METAL_VERSION__ >= 320
#define DEV     coherent(device) device
#else
#define DEV     device
#endif
#define GA32    threadgroup atomic_uint
#define THR     thread
#define INLINE  inline
#define OUTLINE static
#define CONSTV  constant
#define DEVICE  1
#define CLZ(x)  clz(x)
#define A32(p)  ((DEV atomic_uint*)(p))
#define RLX     memory_order_relaxed
#define FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)
#define BAR()   threadgroup_barrier(mem_flags::mem_threadgroup)
#define BARD()  threadgroup_barrier(mem_flags::mem_device \
  | mem_flags::mem_threadgroup)

#define g32_ini(p)    atomic_store_explicit(p, 0, RLX)
#define g32_add(p, v) atomic_fetch_add_explicit(p, v, RLX)
#define g32_get(p)    atomic_load_explicit(p, RLX)
#else
// CUDA keeps plain data L1-cacheable: lanes hand off through a32 + FENCE
#define DEV
#define THR
#define INLINE  static inline
#define CONSTV  static const

#define g32_ini(p)    a32_store(p, 0)
#define g32_add(p, v) a32_add(p, v)
#define g32_get(p)    a32_load(p)
#ifdef BEND_RTC
#define GA32    __shared__ u32
#define OUTLINE static __attribute__((noinline))
#define DEVICE  1
#define CLZ(x)  (u32)__clz((int)(x))
#define FENCE() __threadfence()
#define BAR()   __syncthreads()
#define BARD()  \
  { __threadfence(); __syncthreads(); }
#else
// only clang 19+ has both, and only it compiles preserve_most soundly
#if __has_attribute(preserve_none) && __has_attribute(preserve_most)
#define PRESERVE(A) __attribute__((A))
#else
#define PRESERVE(A)
#endif
#define OUTLINE static __attribute__((noinline, cold)) PRESERVE(preserve_most)
#define DEVICE  0
#define CLZ(x)  (u32)__builtin_clz(x)
#endif
#endif
#define FAR static __attribute__((noinline))

// A segment: a case of the device's switch; on the host, a preserve_none
// function (WL_SIG) entered by musttail, its words fresh at WL_OPEN.
#if DEVICE
#define LOCK(l)
#define UNLOCK(l)
#define WL_CASE(F) case F:
#define WL_OPEN    {
#define WL_JMP(F)  { fid = (F); break; }
#define WL_DYN     WL_JMP
#else
#define LOCK(l)    while (__atomic_exchange_n(&(l), 1, __ATOMIC_ACQUIRE)) {}
#define UNLOCK(l)  __atomic_store_n(&(l), 0, __ATOMIC_RELEASE)
#define WL_FN      static PRESERVE(preserve_none) __attribute__((noinline)) Reply
#define WL_CASE(F) WL_FN WL_##F(WL_SIG)
#define WL_OPEN    { WL_BANK u32 rn;
#define WL_JMP(F)  __attribute__((musttail)) return WL_##F(WL_ALL)
#define WL_DYN(F)  __attribute__((musttail)) return wl_tab[F](WL_ALL)
#endif
#define WL_SPIN     for (;;) { if (err_spun(e.mem, &wpoll)) { return 0; }
#define WL_SPUN     } break;
#define WL_AGAIN(F) continue

#define LANE_STEP (DEVICE ? (int64_t)CUBE : 1)
#define STK(I)    sp[(int64_t)(I) * LANE_STEP]

#define WL_RETN(N)  { rn = (N); sp -= LANE_STEP; WL_DYN((Fid)STK(0)); }
#define WL_CONT     STK(-3)
#define WL_IDX      STK(-2)
#define WL_POPN(N)  sp -= N * LANE_STEP
#define WL_PUSHN(N) sp += N * LANE_STEP
#define WL_FRAME(T) \
  Loc wtl = task_tail(T); \
  u64 wtw = e.mem[wtl + 1]; \
  STK(0) = e.mem[wtl]; \
  STK(1) = (wtw >> 32) & 0xFFFF; \
  STK(2) = FID_EXIT; \
  sp += 3 * LANE_STEP;
#define WL_ARGS(A, N) \
  for (u32 wi = 0; wi + 1 < N; wi += 1) { \
    STK(wi) = e.mem[A + wi]; \
  } \
  sp += (N - 1) * LANE_STEP;
#define WL_ROOM(N) \
  if (DEVICE && sp + (N) * CUBE >= e.mem + STAT_OFF + CUBE) { \
    err_post(e.mem, ERR_DEEP); \
    return 0; \
  }

// Types
// =====

#ifdef __METAL_VERSION__
typedef ulong u64;
typedef long  int64_t;
typedef uint  u32;
typedef uchar u8;
typedef float f32;
// Metal Shading Language has no fp64: F64 is a host and CUDA type, and the
// helpers below are guarded so an F32 program compiles for Metal unchanged
#elif defined(BEND_RTC)
typedef unsigned long long u64;
typedef long long          int64_t;
typedef unsigned int       u32;
typedef unsigned char      u8;
typedef float              f32;
typedef double             f64;
#else
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
typedef float    f32;
typedef double   f64;
#endif

typedef u64 Loc;
#define LOC_MASK ((1ull << 40) - 1)

typedef u32 Cls;
typedef u32 Fid;

typedef u64 Term;
#define TAG_PAK 1ull
#define TAG_CTR 2ull
#define TAG_CLO 3ull
#define TAG_BUF 4ull
#define TAG_TSK 5ull
#define TAG_ARR 6ull
#define TAG_STR 7ull

#define TERM_HOLE (~0ull)

#define RFC_BIT  (1ull << 63)
#define RFC_CNT  ((1u << 24) - 1)

typedef Term Reply;

typedef u32 Err;
#define ERR_RING 1
#define ERR_TAGS 2
#define ERR_HEAP 3
#define ERR_FIDS 4
#define ERR_NATS 5
#define ERR_RFCS 6
#define ERR_DEEP 7
#define ERR_ARRS 8
#define ERR_STRS 9

typedef u32 Ring;

typedef DEV u64* Corpus;

typedef struct {
  Corpus   mem;
  DEV u64* alc;
} Env;

typedef struct {
  u64 off;
  u32 rd;
  u32 wr;
  u32 top;
} Bank;

typedef DEV Term* Stk;

typedef Term Nat;
#define NAT_IMM ((1ull << 48) - 1)

typedef Term U32;

#if DEVICE
typedef u32 u32a;
#else
typedef u32 __attribute__((may_alias)) u32a;
#endif

#ifdef __METAL_VERSION__
typedef threadgroup atomic_uint* Cur;
#else
typedef u32* Cur;
#endif

// Constants
// =========

#define LINE      16
#define PAGE_BITS 7
#define PAGE_LEN  (1ull << PAGE_BITS)
#define CUBE_T    128
#define CUBE      ((u64)CUBE_T * CUBE_T)
#define CUBE_G    (1u << CUBE_LOG)
#define LANES     ((u64)CUBE_T << CUBE_LOG)
#define RING_LOG  (17 - CUBE_LOG)
#define RING_LEN  (1ull << RING_LOG)
#define STAK_LEN  (1ull << 11)
#define NCLS      8
#define NCLS_ALL  32
#define IO_HELP   64

#define ALC_WORDS NCLS_ALL
#define TG_HOLD   2304
#define CHUNK     256
#define CAP_WORDS 32768
#define QUANTUM   (DEVICE ? PAGE_LEN \
  : KEEP_WORDS < 32 * PAGE_LEN ? KEEP_WORDS : 32 * PAGE_LEN)
#if DEVICE
#define KEEP_WORDS CHUNK
#endif
#define RING_WORDS ((1ull << 10) + 2)

#define H_BUMP       0
#define H_CAP        1
#define H_CURSOR     LINE
#define H_ROOT_DONE  (2 * LINE)
#define H_ERROR_CODE (3 * LINE)
#define H_ROOT_WORD  (4 * LINE)
#define H_BANK       (H_ROOT_WORD + WL_RESW)

#define PAGE_UP(n) (((n) + PAGE_LEN - 1) & ~(PAGE_LEN - 1))
#define ALC_OFF  PAGE_UP(H_BANK + 3 * NCLS_ALL)
#define RING_OFF (ALC_OFF + CUBE * 2 * ALC_WORDS)
#define STAK_OFF (RING_OFF + CUBE * RING_WORDS)
#define STAT_OFF (STAK_OFF + CUBE * STAK_LEN)
#define HEAP_OFF (STAT_OFF + PAGE_UP(STAT_LEN))

// Globals
// =======

#if !DEVICE

typedef pthread_mutex_t lock;

static Corpus CORPUS;
static u64    ALC[CUBE_T + 1][3 * ALC_WORDS] __attribute__((aligned(128)));
static u32    KEEP_WORDS;
// the bag: 2^CUBE_LOG groups of CUBE_T lanes (a -D constant on the device)
static u32    CUBE_LOG = 7;
static u32    bank_lock;

static u32            pool_size;
static _Atomic u32    pool_row;
static bool           pool_grow;
static _Atomic u64    pool_tick;
static _Atomic u32    pool_done;
static lock           pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_wake = PTHREAD_COND_INITIALIZER;

// The device program compiles from the binary's own text.
#if BEND_METAL || BEND_CUDA
#pragma clang diagnostic ignored "-Wc23-extensions"
static const char BEND_SRC[] = {
#embed __FILE__
, 0 };
#endif

#ifdef __OBJC__
static id<MTLDevice>               gpu_dev;
static id<MTLCommandQueue>         gpu_que;
static id<MTLComputePipelineState> gpu_pso;
static id<MTLBuffer>               gpu_buf;
static id<MTLComputeCommandEncoder> gpu_enc;
#elif BEND_CUDA
static CUdevice   gpu_dev;
static CUmodule   gpu_lib;
static CUfunction gpu_pso;
#endif
static bool io_gpu;
static Stk  io_stk;

static const char* CLI_HELP =
  "usage: %s [options] [arguments]\n"
  "  --threads N       worker threads, 1 to 128 (default: the CPU count)\n"
  "  --workers N       run N copies of the program as processes, as nginx\n"
  "                    runs its workers; a server shares its port\n"
  "                    (TCP.listen_shared) (default: 1)\n"
  "  --gpu on|off|4GB  run ! calls on the GPU, over this much of its memory\n"
  "                    (default: on if present, over 2GB on Metal)\n"
  "  --gpu-build       write the GPU program and exit\n"
  "  --help            show this text\n"
  "  --                the rest are the program's arguments (IO.args)\n";

#endif

// Tables
// ======

${tabs}

#define TAB_AT(T, S, I) T[S < I ? S : I]

#define fid_arity(x) ((u32)FID_ARITY_T[x])

#define fid_bangs(x) ((bool)(FID_FLAG_T[x] & 1))

#define fid_nofk(x) ((bool)(FID_FLAG_T[x] & 2))

#define fid_seqk(x) (fid_resw(x) != 0)

#define fid_resw(x) ((u32)FID_RESW_T[x])

#define cid_arity(x) ((u32)CID_ARITY_T[x])
#define cid_hot(x) ((bool)CID_HOT_T[x])

// A32
// ===

#define A32_LOOP(k, x) \
  INLINE u32 a32_##k(DEV u32* p, u32 v) { \
    u32 o = a32_load(p); \
    while (!a32_cas(p, &o, x)) { \
    } \
    return o; \
  }

#ifdef __METAL_VERSION__

// through a volatile local, or the M1 pipeline build dies
#define a32_load(p)      \
  ({ volatile thread u32 _a32v = atomic_load_explicit(A32(p), RLX); _a32v; })
#define a32_store(p, v)  atomic_store_explicit(A32(p), v, RLX)
${a32_ops((k) => `atomic_fetch_${k}_explicit(A32(p), v, RLX)`)}
#define a32_swp(p, e, v) \
  atomic_compare_exchange_weak_explicit(A32(p), e, v, RLX, RLX)

#elif defined(BEND_RTC)

#define a32_load(p)     (*(volatile u32*)(p))
#define a32_store(p, v) (*(volatile u32*)(p) = (v))
${a32_ops((k) => `atomic${k[0].toUpperCase()}${k.slice(1)}((u32*)(p), v)`)}

INLINE bool a32_swp(DEV u32* p, u32* e, u32 v) {
  u32 x = *e;
  *e = atomicCAS((u32*)p, x, v);
  return *e == x;
}

#endif

#if DEVICE

INLINE u32 a32_sub_rel(DEV u32* p, u32 v) {
  FENCE();
  return a32_sub(p, v);
}

INLINE void a32_store_rel(DEV u32* p, u32 v) {
  FENCE();
  a32_store(p, v);
}

INLINE u32 a32_load_acq(DEV u32* p) {
  u32 v = a32_load(p);
  FENCE();
  return v;
}

#define a32_acq(p) FENCE()

INLINE bool a32_cas(DEV u32* p, THR u32* e, u32 v) {
  FENCE();
  bool ok = a32_swp(p, e, v);
  FENCE();
  return ok;
}

#else

#define a32_load(p)         __atomic_load_n(p, __ATOMIC_RELAXED)
#define a32_store(p, v)     __atomic_store_n(p, v, __ATOMIC_RELAXED)
${a32_ops((k) => `__atomic_fetch_${k}(p, v, __ATOMIC_RELAXED)`)}
#define a32_sub_rel(p, v)   __atomic_fetch_sub(p, v, __ATOMIC_RELEASE)
#define a32_store_rel(p, v) __atomic_store_n(p, v, __ATOMIC_RELEASE)
#define a32_load_acq(p)     __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define a32_acq(p)          ((void)a32_load_acq(p))

INLINE bool a32_cas(u32* p, u32* e, u32 v) {
  return __atomic_compare_exchange_n(
    p, e, v, 1, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

#endif

A32_LOOP(exch, v)

// a weak CAS may fail with the cell still x
INLINE u32 a32_cmpx(DEV u32* p, u32 x, u32 v) {
  u32 o = x;
  while (!a32_cas(p, &o, v) && o == x) {
  }
  return o;
}

#define a32_at(H, word) ((DEV u32*)&(H)[word])

// Err
// ===

#if DEVICE

INLINE void err_post(Corpus H, Err code) {
  u32 seen = 0;
  while (seen == 0 && !a32_cas(a32_at(H, H_ERROR_CODE), &seen, code)) {}
}

#else

static const char* ERR_TEXT[] = { ${ERRS.map((s) => JSON.stringify(s))
  .join(",\n  ")} };

static void err_fail(const char* msg) {
  fflush(stdout);
  fprintf(stderr, "bend: %s\n", msg);
  _exit(1);
}

static void err_post(Corpus H, Err code) {
  err_fail(ERR_TEXT[code]);
}

static void err_trap(int sig) {
  err_post(NULL, ERR_DEEP);
}

#endif

#define err_seen(H)    (DEVICE && a32_load(a32_at(H, H_ERROR_CODE)) != 0)
#define err_spun(H, n) ((++*(n) & 4095) == 0 && err_seen(H))

${NATIVE.C}
A32_LOOP(fadd, f32_rewrap(f32_unbox(o) + f32_unbox(v)))

// Bank
// ====

// A stack of exact generations per class. The host pops and pushes at rd;
// a device pass pops below rd and pushes above top, compacted after it.

#define bank_at(H, c) ((DEV Bank*)((H) + H_BANK) + (c))

INLINE Loc bank_pop(Corpus H, Cls c) {
  DEV Bank* b = bank_at(H, c);
  Loc got = 0;
  LOCK(bank_lock);
  u32 t = a32_sub(&b->rd, 1);
  if ((int)t > 0) {
    got = H[b->off + t - 1];
  } else {
    a32_add(&b->rd, 1);
  }
  if (!DEVICE) {
    b->wr = b->top = b->rd;
  }
  UNLOCK(bank_lock);
  return got;
}

INLINE void bank_push(Corpus H, Cls c, Loc head) {
  DEV Bank* b = bank_at(H, c);
  LOCK(bank_lock);
  H[b->off + a32_add(&b->wr, 1)] = head;
  if (!DEVICE) {
    b->rd = b->top = b->wr;
  }
  UNLOCK(bank_lock);
}

// Heap
// ====

// Per lane and class: HOT, a LIFO free chain; LEN, its length in words;
// on the host COLD, one parked generation. A host free reaching KEEP_WORDS
// parks HOT as COLD and banks the old COLD. A miss takes COLD, a bank entry
// or a fresh quantum. A device lane banks its complete generations at the
// kernel end (dev_cut). The bump grows only when all of these are empty.

#define ALC_AT(e, i)   (e).alc[(i) * LANE_STEP]
#define ALC_LEN(e, c)  ALC_AT(e, ALC_WORDS + (c))
#define ALC_COLD(e, c) ALC_AT(e, 2 * ALC_WORDS + (c))
#define KEEP(c)        (KEEP_WORDS >> (c) ? KEEP_WORDS >> (c) : 1)

INLINE Cls cls_fit(u32 words) {
  return words > 1 ? 32 - CLZ(words - 1) : 0;
}

OUTLINE void heap_hand(Env e, Cls cls) {
  Loc cold = ALC_COLD(e, cls);
  if (cold) {
    bank_push(e.mem, cls, cold);
  }
  ALC_COLD(e, cls) = ALC_AT(e, cls);
  ALC_AT(e, cls)   = 0;
  ALC_LEN(e, cls)  = 0;
}

#if DEVICE
#define corpus_grow(H, n) false
#else
static bool corpus_grow(Corpus H, u64 need);
#endif

OUTLINE Loc heap_alloc_miss(Env e, Cls cls) {
  Corpus H = e.mem;
  Loc  got = 0;
  if (!DEVICE) {
    got = ALC_COLD(e, cls);
    ALC_COLD(e, cls) = 0;
  }
  if (!got) {
    got = bank_pop(H, cls);
  }
  u32 n = got ? KEEP(cls) : cls < NCLS ? QUANTUM >> cls : 1;
  if (!got) {
    u32 pages = (n << cls) >> PAGE_BITS;
    u32 p     = a32_add(a32_at(H, H_BUMP), pages);
    if ((u64)p + pages > a32_load_acq(a32_at(H, H_CAP))
      && !corpus_grow(H, (u64)p + pages)) {
      err_post(H, ERR_HEAP);
      return HEAP_OFF;
    }
    got = HEAP_OFF + ((u64)p << PAGE_BITS);
    for (u32 i = 1; i <= n; i += 1) {
      H[got + ((u64)(i - 1) << cls)] = i < n ? got + ((u64)i << cls) : 0;
    }
  }
  ALC_AT(e, cls)  = H[got];
  ALC_LEN(e, cls) = (u64)(n - 1) << cls;
  return got;
}

INLINE Loc heap_alloc(Env e, Cls cls) {
  Loc h = ALC_AT(e, cls);
  if (h) {
    ALC_AT(e, cls)   = e.mem[h];
    ALC_LEN(e, cls) -= 1ull << cls;
    return h;
  }
  return heap_alloc_miss(e, cls);
}

INLINE void heap_free(Env e, Cls cls, Loc loc) {
  if (err_seen(e.mem)) {
    return;
  }
  e.mem[loc]       = ALC_AT(e, cls);
  ALC_AT(e, cls)   = loc;
  ALC_LEN(e, cls) += 1ull << cls;
  if (!DEVICE && ALC_LEN(e, cls) >= KEEP_WORDS) {
    heap_hand(e, cls);
  }
}

INLINE void spare_free(Env e, Cls cls, Loc loc) {
  if (loc >= HEAP_OFF) {
    heap_free(e, cls, loc);
  }
}

// Term
// ====

#define term_make(tag, aux, loc) \
  (((u64)(tag) << 56) | ((u64)(aux) << 40) | (u64)(loc))

#define term_ctr(cid, loc) term_make(TAG_CTR, cid, loc)
#define term_pak(cid, loc) term_make(TAG_PAK, cid, loc)
#define term_clo(fid, loc) term_make(TAG_CLO, fid, loc)
#define term_buf(cls, loc) term_make(TAG_BUF, cls, loc)
#define term_tsk(fid, loc) term_make(TAG_TSK, fid, loc)

INLINE Term term_blk(bool arr, Cls cls, Loc loc) {
  return term_buf(cls, loc) | ((u64)arr << 57);
}

INLINE u64 term_tag(Term t) {
  return (t >> 56) & 0x7f;
}

INLINE bool term_rfc(Term t) {
  return (t & RFC_BIT) != 0;
}

INLINE u64 term_aux(Term t) {
  return (t >> 40) & 0xFFFF;
}

INLINE Loc term_loc(Term t) {
  return t & LOC_MASK;
}

// A static node (below the heap) is trivial, as is a captureless closure.
INLINE bool term_triv(Term t) {
  return term_tag(t) <= TAG_PAK || t == TERM_HOLE || term_loc(t) < HEAP_OFF;
}

OUTLINE Term rfc_wrap(Env e, Term t, u32 cnt) {
  if (term_tag(t) == TAG_CLO || term_tag(t) == TAG_TSK) {
    err_post(e.mem, ERR_RFCS);
    return t;
  }
  Loc r = heap_alloc(e, 0);
  if (err_seen(e.mem)) { return t; }
  e.mem[r] = ((u64)term_loc(t) << 24) | cnt;
  return (t & ~LOC_MASK) | RFC_BIT | r;
}

INLINE Term rfc_seal(Env e, Term t) {
  if ((term_tag(t) != TAG_CTR && term_tag(t) != TAG_STR)
    || term_rfc(t) || term_triv(t)) {
    return t;
  }
  return rfc_wrap(e, t, 1);
}

INLINE u64 rfc_view(Env e, Loc r) {
  DEV u32* w = a32_at(e.mem, r);
  u64 cell = ((u64)a32_load(w + 1) << 32) | a32_load(w);
  if ((cell & RFC_CNT) == 1) {
    a32_acq(w);
  }
  return cell;
}

INLINE void rfc_bump(Env e, Loc r, u32 k) {
  u32 c = a32_add(a32_at(e.mem, r), k);
  if ((c & RFC_CNT) >= RFC_CNT - k) {
    err_post(e.mem, ERR_RFCS);
  }
}

INLINE Term term_keep(Env e, Term t) {
  if (term_rfc(t)) {
    rfc_bump(e, term_loc(t), 1);
    return t;
  }
  if (term_triv(t)) {
    return t;
  }
  return rfc_wrap(e, t, 2);
}

INLINE Loc term_peek(Env e, Term t) {
  if (term_rfc(t)) {
    return rfc_view(e, term_loc(t)) >> 24;
  }
  return term_loc(t);
}

// A fork's handle (BLK_SHR: an Array binder is hot) is a redirect: loaded
// plainly, copied and dropped by a match.
#define blk_shr(t) (BLK_SHR && term_rfc(t))

INLINE Loc blk_loc(Corpus H, Term a) {
  return blk_shr(a) ? H[term_loc(a)] >> 24 : term_loc(a);
}

INLINE Cls blk_cls(Term t) {
  return (u32)term_aux(t) & 31;
}

#define buf_wcls(c) ((c) == 0 ? 0 : (c) - 1)

INLINE Cls blk_span(Term t) {
  Cls c = blk_cls(t);
  return term_tag(t) == TAG_ARR ? c : buf_wcls(c);
}

FAR void term_drop(Env e, Term t) {
  Corpus H = e.mem;
  u64  cur = 0;
  Term c0  = 0;
  u32  step = 0;
  for (;;) {
    if (!term_triv(t) && term_rfc(t)) {
      Loc      r = term_loc(t);
      DEV u32* p = a32_at(H, r);
      if ((a32_sub_rel(p, 1) & RFC_CNT) != 1) {
        t = 0;
      } else {
        a32_acq(p);
        t = (t & ~(RFC_BIT | LOC_MASK)) | (H[r] >> 24);
        heap_free(e, 0, r);
      }
    }
    if (!term_triv(t)) {
      u64 tag = term_tag(t);
      if (tag == TAG_BUF) {
        heap_free(e, blk_span(t), term_loc(t));
      } else {
        u32 aux = (u32)term_aux(t);
        Loc loc = term_loc(t);
        // a string is one cell of class 1; an array's cells count by its
        // class; a task holds two more words
        u32 n   = tag == TAG_STR ? 1 : tag == TAG_ARR ? 0
          : tag == TAG_CTR ? cid_arity(aux) : fid_arity(aux) - (tag == TAG_CLO);
        Cls cls = tag == TAG_STR ? 1 : tag == TAG_ARR ? 64 | blk_cls(t)
          : cls_fit(tag == TAG_TSK ? n + 2 : n);
        c0 = H[loc];
        H[loc] = cur;
        cur = loc | ((u64)n << 48) | ((u64)cls << 56);
      }
    }
    for (;;) {
      if (err_spun(H, &step)) {
        return;
      }
      if (cur == 0) {
        return;
      }
      Loc  loc = cur & LOC_MASK;
      u32  i   = (u8)(cur >> 40);
      u32  n   = (u8)(cur >> 48);
      Cls  cls = (u32)(cur >> 56);
      bool arr = cls > 63;
      u32  j   = i;
      if (arr) {
        cls &= 63;
        n   = 1u << cls;
        if (i == 2) {
          j = (u32)H[loc + 1];
        }
      }
      if (j < n) {
        Term c = j == 0 ? c0 : H[loc + j];
        if (arr && j > 0) {
          H[loc + 1] = j + 1;
        }
        if (!arr || i < 2) {
          cur += 1ull << 40;
        }
        if (!term_triv(c)) {
          t = c;
          break;
        }
      } else {
        u64 up = H[loc];
        heap_free(e, cls, loc);
        cur = up;
      }
    }
  }
}

INLINE void term_sink(Env e, Term t) {
  if (!term_triv(t)) {
    term_drop(e, t);
  }
}

// A U64, I64 or F64 in a slot the runtime drops or shares as a Term (a
// polymorphic parameter, a generic constructor's field): a word that would
// read as a reference (a heap location, or the count bit) rides in a
// one-word block, the rest ride as they are. A boxed one is never trivial.
INLINE bool x64_raw(u64 w) {
  return term_triv(w) && !term_rfc(w);
}

INLINE Term x64_box(Env e, u64 w) {
  if (x64_raw(w)) {
    return w;
  }
  Loc l = heap_alloc(e, 0);
  if (err_seen(e.mem)) {
    return 0;
  }
  e.mem[l] = w;
  return term_buf(0, l);
}

INLINE u64 x64_peek(Env e, Term t) {
  return x64_raw(t) ? t : e.mem[term_peek(e, t)];
}

INLINE u64 x64_take(Env e, Term t) {
  if (x64_raw(t)) {
    return t;
  }
  u64 w = e.mem[term_peek(e, t)];
  term_drop(e, t);
  return w;
}

OUTLINE void span_fade(Env e, Term t, Loc src, u32 n) {
  for (u32 j = 0; j < n; j += 1) {
    Term f = e.mem[src + j];
    if (term_rfc(f)) {
      rfc_bump(e, term_loc(f), 1);
    } else if (!term_triv(f)) {
      err_post(e.mem, ERR_RFCS);
    }
  }
  term_drop(e, t);
}

INLINE Loc ctr_take(Env e, Term t, u32 n, THR Term* out) {
  Corpus H = e.mem;
  if (!term_rfc(t)) {
    for (u32 j = 0; j < n; j += 1) {
      out[j] = H[term_loc(t) + j];
    }
    return term_loc(t);
  }
  Loc r    = term_loc(t);
  u64 cell = rfc_view(e, r);
  Loc src  = cell >> 24;
  for (u32 j = 0; j < n; j += 1) {
    out[j] = H[src + j];
  }
  if ((cell & RFC_CNT) == 1) {
    heap_free(e, 0, r);
    return src;
  }
  span_fade(e, t, src, n);
  return 0;
}

INLINE Term term_word(Env e, Term w, u32 n) {
  u64 x = 0;
  Term t = w;
  for (u32 i = 0; i < n && term_aux(t) == CID_WCON; i += 1) {
    Loc l = term_peek(e, t);
    x |= (u64)(e.mem[l] & 1) << i;
    t = e.mem[l + 1];
  }
  term_sink(e, w);
  return x;
}

// Blk
// ===

// A block owns one allocation in its class (an ARR 2^c Terms, a BUF 2^c
// u32). Matching ANode is blk_half twice (the high call frees the source);
// ANode{l, r} is blk_node; Array.clone is blk_copy.

#define BLK_ALLOC(n, w) \
  Loc n = heap_alloc(e, w); \
  if (err_seen(e.mem)) { \
    return term_buf(0, n); \
  }

INLINE DEV u32a* blk_ptr(Corpus H, Loc loc, u32 i) {
  return (DEV u32a*)(H + loc) + i;
}

INLINE Term blk_read(Corpus H, bool arr, Loc loc, u32 i) {
  if (arr) {
    return H[loc + i];
  }
  return (u64)*blk_ptr(H, loc, i);
}

INLINE void blk_write(Corpus H, bool arr, Loc loc, u32 i, Term v) {
  if (arr) {
    H[loc + i] = v;
  } else {
    *blk_ptr(H, loc, i) = (u32)v;
  }
}

INLINE u32 blk_at(Term a, U32 i, u32 lgs) {
  return ((u32)i & (u32)((1ull << (blk_cls(a) - lgs)) - 1)) << lgs;
}

// in range, a host read skips the mask, so its loop can vectorize
INLINE u32 blk_get(Term a, U32 i, u32 lgs) {
  return DEVICE || (u32)i > blk_at(a, -1, lgs) >> lgs ? blk_at(a, i, lgs)
    : (u32)i << lgs;
}

INLINE Term blk_keep(Env e, Loc at) {
  Term w = e.mem[at];
  Term v = term_keep(e, w);
  if (v != w) {
    e.mem[at] = v;
  }
  return v;
}

INLINE void blk_fill(Env e, Loc dst, Loc src, u64 n, bool keep) {
#if !DEVICE
  // Fresh blocks cannot overlap; keep leaf-sized copies inline.
  if (!keep && n >= 16) {
    memcpy(e.mem + dst, e.mem + src, n * sizeof(Term));
    return;
  }
#endif
  for (u64 j = 0; j < n; j += 1) {
    e.mem[dst + j] = keep ? blk_keep(e, src + j) : e.mem[src + j];
  }
}

INLINE void blk_free(Env e, Term t) {
  blk_shr(t) ? term_drop(e, t) : heap_free(e, blk_span(t), term_loc(t));
}

OUTLINE Term blk_copy(Env e, Term a) {
  bool arr = term_tag(a) == TAG_ARR;
  Cls cls = blk_span(a);
  BLK_ALLOC(dst, cls)
  blk_fill(e, dst, blk_loc(e.mem, a), 1ull << cls, arr);
  return term_blk(arr, blk_cls(a), dst);
}

INLINE Term blk_node(Env e, Term l, Term r) {
  Corpus H = e.mem;
  bool arr = term_tag(l) == TAG_ARR;
  Cls c = blk_cls(l);
  if (c != blk_cls(r) || c + 1 >= NCLS_ALL) {
    err_post(H, ERR_TAGS);
    return l;
  }
  Loc pl = blk_loc(H, l);
  Loc pr = blk_loc(H, r);
  BLK_ALLOC(n, arr ? c + 1 : c)
  if (!arr && c == 0) {
    H[n] = (u64)*blk_ptr(H, pl, 0) | ((u64)*blk_ptr(H, pr, 0) << 32);
  } else {
    u64 cw = 1ull << blk_span(l);
    blk_fill(e, n, pl, cw, arr && blk_shr(l));
    blk_fill(e, n + cw, pr, cw, arr && blk_shr(r));
  }
  blk_free(e, l);
  blk_free(e, r);
  return term_blk(arr, c + 1, n);
}

INLINE Term blk_half(Env e, Term a, u32 hi) {
  Corpus H = e.mem;
  bool arr = term_tag(a) == TAG_ARR;
  Cls c = blk_cls(a);
  if (c == 0) {
    err_post(H, ERR_TAGS);
    return a;
  }
  c -= 1;
  Cls cw = arr ? c : buf_wcls(c);
  Loc src = blk_loc(H, a);
  BLK_ALLOC(n, cw)
  if (!arr && c == 0) {
    H[n] = (u64)*blk_ptr(H, src, hi);
  } else {
    blk_fill(e, n, src + ((u64)hi << cw), 1ull << cw, arr && blk_shr(a));
  }
  if (hi) {
    blk_free(e, a);
  }
  return term_blk(arr, c, n);
}

INLINE Term blk_new(Env e, bool arr, Nat d, u32 lgs, u32 n, THR Term* v) {
  Corpus H = e.mem;
  if (d + lgs > 31) {
    err_post(H, ERR_ARRS);
    d = 0;
  }
  Cls c = (u32)d + lgs;
  BLK_ALLOC(l, arr ? c : buf_wcls(c))
  for (u32 j = 0; j < n; j += 1) {
    Term w = v[j];
    if (arr && d > 0 && !term_triv(w)) {
      if (d >= 24) {
        err_post(H, ERR_RFCS);
      } else if (term_rfc(w)) {
        rfc_bump(e, term_loc(w), (1u << d) - 1);
      } else {
        w = rfc_wrap(e, w, 1u << d);
      }
    }
    v[j] = w;
  }
  for (u64 i = 0; i < (1ull << c); i += 1) {
    blk_write(H, arr, l, (u32)i, i % (1u << lgs) < n ? v[i % (1u << lgs)] : 0);
  }
  return term_blk(arr, c, l);
}

// String: each descriptor owns one already-counted packed payload, its cells
// 4, 2 or 1 bytes wide: the payload Term's aux holds nar = 0, 1, 2 above the
// class, the narrowest width its content allowed when it was built. A view
// never changes nar; a write of a wider cell reallocates in str_reserve.
// Peek borrows. Take consumes metadata and moves/retains the payload BEFORE
// releasing a shared descriptor. Only taken parts may enter str_writable.
// Taken from its one owner, a heap descriptor is not released but kept as
// home, and str_view_owned writes the new view back into it: a cut, a push
// or an append on an unshared string frees and allocates no descriptor.
#define STR_LIMIT (1ull << 31)
#define str_nar(p) ((p).data ? (u32)(term_aux((p).data) >> 5) & 3 : 2)
#define str_cap(d) (1ull << (blk_cls(d) + ((term_aux(d) >> 5) & 3)))
typedef struct { Term data; u32 off; u32 len; Term home; } StrParts;

INLINE u32 str_fit(u32 c) { return c < 256 ? 2 : c < 65536 ? 1 : 0; }

INLINE StrParts str_peek(Env e, Term s) {
  StrParts p = {0, 0, 0};
  if (term_tag(s) == TAG_STR) {
    Loc l = term_peek(e, s);
    p.data = e.mem[l];
    p.off = (u32)(e.mem[l + 1] >> 32);
    p.len = (u32)e.mem[l + 1];
  }
  return p;
}

INLINE StrParts str_take(Env e, Term s) {
  StrParts p = str_peek(e, s);
  if (p.len && term_rfc(s)
      && (rfc_view(e, term_loc(s)) & RFC_CNT) == 1) {
    p.home = s;
  } else if (p.len) {
    Term data[1];
    Loc l = ctr_take(e, s, 1, data);
    p.data = data[0];
    spare_free(e, 1, l);
  }
  return p;
}

// A home no view goes back into: its count cell and node are freed, the
// payload it held having been taken with the parts.
INLINE void str_home_free(Env e, Term h) {
  if (h) {
    Loc r = term_loc(h);
    Loc l = rfc_view(e, r) >> 24;
    heap_free(e, 0, r);
    spare_free(e, 1, l);
  }
}

INLINE Term str_view_owned(Env e, StrParts p) {
  if (!p.len || err_seen(e.mem)) {
    str_home_free(e, p.home);
    term_sink(e, p.data);
    return term_pak(CID_SNIL, 0);
  }
  u64 cap = str_cap(p.data);
  if (cap > STR_LIMIT || p.len > cap || p.off > cap - p.len) {
    err_post(e.mem, ERR_STRS);
    str_home_free(e, p.home);
    term_sink(e, p.data);
    return term_pak(CID_SNIL, 0);
  }
  if (p.home) {
    Loc l = rfc_view(e, term_loc(p.home)) >> 24;
    e.mem[l] = p.data;
    e.mem[l + 1] = ((u64)p.off << 32) | p.len;
    return p.home;
  }
  Loc l = heap_alloc(e, 1);
  if (err_seen(e.mem)) { term_sink(e, p.data); return term_pak(CID_SNIL, 0); }
  e.mem[l] = p.data;
  e.mem[l + 1] = ((u64)p.off << 32) | p.len;
  return rfc_seal(e, term_make(TAG_STR, CID_SCON, l));
}

INLINE StrParts str_alloc(Env e, u64 n, u32 nar) {
  StrParts p = {0, 0, 0};
  if (!n || err_seen(e.mem)) { return p; }
  if (n > STR_LIMIT) { err_post(e.mem, ERR_STRS); return p; }
  Cls c = cls_fit((u32)((n + (1u << nar) - 1) >> nar));
  Loc l = heap_alloc(e, buf_wcls(c));
  if (err_seen(e.mem)) { return p; }
  // Install the count cell before this payload can escape into metadata.
  p.data = rfc_wrap(e, term_buf(c | nar << 5, l), 1);
  p.len = (u32)n;
  return p;
}

INLINE u32 str_cell(Corpus H, Loc l, u32 nar, u32 i) {
  DEV u8* b = (DEV u8*)(H + l);
  return nar == 2 ? b[i] : nar == 1 ? b[2 * i] | (u32)b[2 * i + 1] << 8
    : *blk_ptr(H, l, i);
}

INLINE void str_cell_put(Corpus H, Loc l, u32 nar, u32 i, u32 c) {
  DEV u8* b = (DEV u8*)(H + l);
  if (nar == 2) { b[i] = (u8)c; }
  else if (nar == 1) { b[2 * i] = (u8)c; b[2 * i + 1] = (u8)(c >> 8); }
  else { *blk_ptr(H, l, i) = c; }
}

INLINE u32 str_at_peek(Env e, StrParts p, u32 i) {
  return str_cell(e.mem, term_peek(e, p.data), str_nar(p), p.off + i);
}

INLINE void str_put(Env e, StrParts p, u32 i, u32 c) {
  str_cell_put(e.mem, term_peek(e, p.data), str_nar(p), p.off + i, c);
}

// Three-part write test: owned private metadata (str_take), dynamic origin,
// and a count of one, observed with the runtime's acquire discipline.
INLINE bool str_writable(Env e, StrParts p) {
  return p.data && term_rfc(p.data) && term_peek(e, p.data) >= HEAP_OFF
    && (rfc_view(e, term_loc(p.data)) & RFC_CNT) == 1;
}

INLINE void str_copy_cells(Env e, StrParts dst, u32 at, StrParts src) {
  if (err_seen(e.mem)) { return; }
  u32 poll = 0;
  Loc from = term_peek(e, src.data), to = term_peek(e, dst.data);
  u32 sn = str_nar(src), dn = str_nar(dst);
#if !DEVICE
  // cells of one width are laid out alike (1, 2 or 4 bytes, the low
  // byte first): the copy is one block move
  if (sn == dn) {
    u32 w = 4u >> sn;
    memmove((u8*)(e.mem + to) + (u64)(dst.off + at) * w,
      (const u8*)(e.mem + from) + (u64)src.off * w, (u64)src.len * w);
    return;
  }
#endif
  for (u32 i = 0; i < src.len; i++) {
    if (err_spun(e.mem, &poll)) { return; }
    str_cell_put(e.mem, to, dn, dst.off + at + i,
      str_cell(e.mem, from, sn, src.off + i));
  }
}

// Consumes owned parts, copying only their visible range when necessary:
// no room, not writable, or cells narrower than nar, what the write needs.
INLINE StrParts str_reserve(Env e, StrParts p, u64 need, bool front, u32 nar) {
  u64 n = (u64)p.len + need;
  if (n > STR_LIMIT) {
    err_post(e.mem, ERR_STRS);
    term_sink(e, p.data);
    StrParts z = {0, 0, 0, p.home}; return z;
  }
  u64 cap = p.data ? str_cap(p.data) : 0;
  if (str_nar(p) < nar) { nar = str_nar(p); }
  if (str_writable(e, p) && str_nar(p) == nar && (front ? p.off >= need
      : cap - p.off - p.len >= need)) { return p; }
  u64 target = (u64)p.len * (need ? 2 : 1);
  if (target < n) { target = n; }
  if (target > STR_LIMIT) { target = STR_LIMIT; }
  StrParts q = str_alloc(e, target, nar);
  if (!err_seen(e.mem) && q.data) {
    q.len = p.len;
    q.off = front ? (u32)(str_cap(q.data) - p.len) : 0;
    str_copy_cells(e, q, 0, p);
  }
  term_sink(e, p.data);
  q.home = p.home;
  return q;
}

INLINE Term str_prepend_take(Env e, u32 c, Term s) {
  StrParts p = str_take(e, s);
  // Putting back the cell an uncons just stepped past is a view, not a
  // write: nothing is stored, so a shared or static payload qualifies.
  if (p.data && p.off > 0) {
    StrParts b = {p.data, p.off - 1, p.len + 1, p.home};
    if (str_at_peek(e, b, 0) == c) { return str_view_owned(e, b); }
  }
  p = str_reserve(e, p, 1, true, str_fit(c));
  if (err_seen(e.mem)) { return str_view_owned(e, p); }
  p.off--; p.len++;
  str_put(e, p, 0, c);
  return str_view_owned(e, p);
}

INLINE Term str_slice_take(Env e, Term s, u64 lo, u64 hi) {
  StrParts p = str_take(e, s);
  if (lo > p.len) { lo = p.len; }
  if (hi > p.len) { hi = p.len; }
  if (hi < lo) { hi = lo; }
  p.off += (u32)lo;
  p.len = (u32)(hi - lo);
  return str_view_owned(e, p);
}

INLINE void str_uncons(Env e, Term s, THR Term* out) {
  // A heap descriptor with a count of one advances in place: the walk
  // over a token then frees and allocates no descriptor/count pair.
  if (!term_triv(s) && term_rfc(s)) {
    u64 cell = rfc_view(e, term_loc(s));
    Loc l = cell >> 24;
    if ((cell & RFC_CNT) == 1 && (u32)e.mem[l + 1] > 1) {
      StrParts p = {e.mem[l], (u32)(e.mem[l + 1] >> 32), (u32)e.mem[l + 1]};
      out[0] = str_at_peek(e, p, 0);
      e.mem[l + 1] = ((u64)(p.off + 1) << 32) | (p.len - 1);
      out[1] = s;
      return;
    }
  }
  StrParts p = str_take(e, s);
  if (err_seen(e.mem)) { out[0] = 0; out[1] = term_pak(CID_SNIL, 0); return; }
  out[0] = str_at_peek(e, p, 0);
  p.off++; p.len--;
  out[1] = str_view_owned(e, p);
}

// Bytes.push: one cell on the end, in place when the string is its
// payload's one owner with room (append's doubling amortizes the rest).
FAR Term str_push_grow(Env e, Term s, u32 c) {
  StrParts p = str_reserve(e, str_take(e, s), 1, false, str_fit(c));
  if (!err_seen(e.mem) && p.data) {
    str_put(e, p, p.len, c);
    p.len++;
  }
  return str_view_owned(e, p);
}

INLINE Term str_push_take(Env e, Term s, u32 c) {
  // The filling loop's case, read once: the one owner of a descriptor
  // holding the one count of a heap payload with room and wide enough
  // cells stores the cell and bumps the length.
  // Both counts are read whole and plainly; the one acquire, once both are
  // one, orders the store after whatever the last other owners did.
  if (!term_triv(s) && term_rfc(s)) {
    u64 cell = e.mem[term_loc(s)];
    Loc l = cell >> 24;
    Term d = e.mem[l];
    u64 ol = e.mem[l + 1];
    if ((cell & RFC_CNT) == 1 && (u32)ol && term_rfc(d)) {
      u64 dc = e.mem[term_loc(d)];
      Loc dl = dc >> 24;
      u32 nar = (u32)(term_aux(d) >> 5) & 3, at = (u32)(ol >> 32) + (u32)ol;
      if ((dc & RFC_CNT) == 1 && dl >= HEAP_OFF && str_fit(c) >= nar
          && at < str_cap(d)) {
        a32_acq(a32_at(e.mem, term_loc(d)));
        str_cell_put(e.mem, dl, nar, at, c);
        e.mem[l + 1] = ol + 1;
        return s;
      }
    }
  }
  return str_push_grow(e, s, c);
}

// Bytes.len, get, span and the finds borrow the string (the call
// site neither shares nor drops it): no view, no count, no Maybe.
// A borrowed read holds a count on what it reads, so no location it follows
// can move and nothing it reads was written after it got the count: its
// loads are plain ones, which a loop over one buffer hoists.
INLINE Loc term_peek_ro(Env e, Term t) {
  return term_rfc(t) ? (Loc)(e.mem[term_loc(t)] >> 24) : term_loc(t);
}

INLINE StrParts str_peek_ro(Env e, Term s) {
  StrParts p = {0, 0, 0, 0};
  if (term_tag(s) == TAG_STR) {
    Loc l = term_peek_ro(e, s);
    p.data = e.mem[l];
    p.off = (u32)(e.mem[l + 1] >> 32);
    p.len = (u32)e.mem[l + 1];
  }
  return p;
}

INLINE Nat str_len_peek(Env e, Term s) {
  return str_peek_ro(e, s).len;
}

INLINE u32 str_byte_peek(Env e, Term s, Nat i) {
  StrParts p = str_peek_ro(e, s);
  return i < p.len
    ? str_cell(e.mem, term_peek_ro(e, p.data), str_nar(p), p.off + (u32)i) : 256;
}

// the four cells from i, each's low byte, little end first, 0 past the end
INLINE u32 str_word_le_peek(Env e, Term s, Nat i) {
  StrParts p = str_peek_ro(e, s);
  if (i >= p.len) { return 0; }
  Loc l = term_peek_ro(e, p.data);
  u32 nar = str_nar(p), j = p.off + (u32)i, n = p.len - (u32)i, x = 0;
  if (nar == 2 && n >= 4) {
    DEV u8* b = (DEV u8*)(e.mem + l) + j;
    return (u32)b[0] | (u32)b[1] << 8 | (u32)b[2] << 16 | (u32)b[3] << 24;
  }
  for (u32 k = 0; k < 4 && k < n; k++) {
    x |= (str_cell(e.mem, l, nar, j + k) & 255) << (8 * k);
  }
  return x;
}

// how many cells from i are in [lo, hi)
INLINE Nat str_span_peek(Env e, Term s, Nat i, u32 lo, u32 hi) {
  StrParts p = str_peek_ro(e, s);
  if (i >= p.len || hi <= lo) { return 0; }
  Loc l = term_peek_ro(e, p.data);
  u32 nar = str_nar(p), w = hi - lo, j = p.off + (u32)i, end = p.off + p.len;
  if (nar == 2) {
    DEV u8* b = (DEV u8*)(e.mem + l);
    while (j < end && (u32)b[j] - lo < w) { j++; }
  } else {
    while (j < end && str_cell(e.mem, l, nar, j) - lo < w) { j++; }
  }
  return j - p.off - (u32)i;
}

// i plus how many cells from i come before the first x
INLINE Nat str_find_byte_peek(Env e, Term s, Nat i, u32 x) {
  StrParts p = str_peek_ro(e, s);
  if (i >= p.len) { return i; }
  Loc l = term_peek_ro(e, p.data);
  u32 nar = str_nar(p), j = p.off + (u32)i, end = p.off + p.len;
  if (nar == 2) {
    if (x > 255) { return p.len; }
    DEV u8* b = (DEV u8*)(e.mem + l);
#if DEVICE
    while (j < end && b[j] != x) { j++; }
#else
    const u8* at = (const u8*)memchr((const u8*)b + j, (int)x, end - j);
    j = at ? (u32)(at - (const u8*)b) : end;
#endif
  } else {
    while (j < end && str_cell(e.mem, l, nar, j) != x) { j++; }
  }
  return j - p.off;
}

// i plus how many cells from i come before the first of w, x, y, z
INLINE Nat str_find_any_peek(Env e, Term s, Nat i, u32 w, u32 x, u32 y, u32 z) {
  StrParts p = str_peek_ro(e, s);
  if (i >= p.len) { return i; }
  Loc l = term_peek_ro(e, p.data);
  u32 nar = str_nar(p), j = p.off + (u32)i, end = p.off + p.len;
  if (nar == 2) {
    DEV u8* b = (DEV u8*)(e.mem + l);
    while (j < end && b[j] != w && b[j] != x && b[j] != y && b[j] != z) { j++; }
  } else {
    for (; j < end; j++) {
      u32 c = str_cell(e.mem, l, nar, j);
      if (c == w || c == x || c == y || c == z) { break; }
    }
  }
  return j - p.off;
}

INLINE Nat str_length_take(Env e, Term s) {
  Nat n = str_peek(e, s).len;
  term_sink(e, s);
  return n;
}

INLINE Term str_append_take(Env e, Term a, Term b) {
  StrParts q = str_peek(e, b);
  if (!q.len) { term_sink(e, b); return a; }
  // The building loop's case, as str_push_take reads it: the one owner of
  // a descriptor holding the one count of a heap payload with room for b,
  // in cells as wide as b's or wider, copies b's cells in and bumps the
  // length; no descriptor is taken, reserved or viewed again.
  if (!term_triv(a) && term_rfc(a)) {
    u64 cell = e.mem[term_loc(a)];
    Loc l = cell >> 24;
    Term d = e.mem[l];
    u64 ol = e.mem[l + 1];
    if ((cell & RFC_CNT) == 1 && (u32)ol && term_rfc(d)) {
      u64 dc = e.mem[term_loc(d)];
      Loc dl = dc >> 24;
      u32 nar = (u32)(term_aux(d) >> 5) & 3, at = (u32)(ol >> 32) + (u32)ol;
      if ((dc & RFC_CNT) == 1 && dl >= HEAP_OFF && str_nar(q) >= nar
          && (u64)at + q.len <= str_cap(d)) {
        a32_acq(a32_at(e.mem, term_loc(d)));
        StrParts p = {d, 0, at, 0};
        str_copy_cells(e, p, at, q);
        e.mem[l + 1] = ol + q.len;
        term_sink(e, b);
        return a;
      }
    }
  }
  if (!str_peek(e, a).len) { term_sink(e, a); return b; }
  StrParts p = str_reserve(e, str_take(e, a), q.len, false, str_nar(q));
  if (!err_seen(e.mem)) {
    str_copy_cells(e, p, p.len, q);
    p.len += q.len;
  }
  term_sink(e, b);
  return str_view_owned(e, p);
}

// Nat.show and U32.show: the decimal digits, most significant first,
// "0" for zero, in one payload of byte cells (the Bend definitions'
// digit-at-a-time prepends, each a view and a count, made at once)
INLINE Term str_show_u64(Env e, u64 v) {
  u32 d[20];
  u32 n = 0;
  do { d[n++] = 48 + (u32)(v % 10); v /= 10; } while (v != 0);
  StrParts p = str_alloc(e, n, 2);
  if (!err_seen(e.mem) && p.data) {
    for (u32 i = 0; i < n; i++) { str_put(e, p, i, d[n - 1 - i]); }
  }
  return str_view_owned(e, p);
}

INLINE Term str_copy_take(Env e, Term s) {
  StrParts p = str_peek(e, s);
  u32 nar = 2, poll = 0;
  for (u32 i = 0; i < p.len && nar > str_nar(p); i++) {
    if (err_spun(e.mem, &poll)) { break; }
    u32 fit = str_fit(str_at_peek(e, p, i));
    if (fit < nar) { nar = fit; }
  }
  StrParts q = str_alloc(e, p.len, nar);
  if (!err_seen(e.mem)) { str_copy_cells(e, q, 0, p); }
  term_sink(e, s);
  return str_view_owned(e, q);
}

// A fresh node of n <= 3 words, its boxed fields already sealed.
INLINE Term str_node(Env e, u32 cid, u32 n, u64 a, u64 b, u64 c) {
  Loc l = heap_alloc(e, cls_fit(n));
  if (err_seen(e.mem)) { return term_pak(CID_NONE, 0); }
  e.mem[l] = a;
  if (n > 1) { e.mem[l + 1] = b; }
  if (n > 2) { e.mem[l + 2] = c; }
  return term_ctr(cid, l);
}

INLINE Term str_get_take(Env e, Term s, Nat n, bool end) {
  StrParts p = str_peek(e, s);
  bool ok = end ? n > 0 && n <= p.len : n < p.len;
  Term c = ok ? term_pak(CID_CHR, str_at_peek(e, p,
    end ? p.len - (u32)n : (u32)n)) : 0;
  term_sink(e, s);
  return ok ? str_node(e, CID_SOME, 1, c, 0, 0) : term_pak(CID_NONE, 0);
}

// Map.bit's 33-bit key protocol: position 33*i says whether cell i exists;
// positions 33*i+1..33*i+32 are its U32 bits, high first. Borrows the key;
// the row hands the original owned key back beside the Bool.
INLINE bool str_bit_peek(Env e, Term s, Nat pos) {
  StrParts p = str_peek(e, s);
  u64 ci = pos / 33, off = pos % 33;
  if (ci >= p.len) { return false; }
  return off == 0 || ((str_at_peek(e, p, (u32)ci) >> (32 - off)) & 1) != 0;
}

INLINE Term str_end_take(Env e, Term s, Nat n, bool take) {
  u64 len = str_peek(e, s).len;
  u64 at = n > len ? 0 : len - n;
  return str_slice_take(e, s, take ? at : 0, take ? len : at);
}

// Borrow both; Cmp is flattened as LT=0, EQ=1, GT=2.
INLINE u32 str_order_peek(Env e, Term a, Term b) {
  StrParts p = str_peek(e, a), q = str_peek(e, b);
  u32 poll = 0, n = p.len < q.len ? p.len : q.len;
  Loc pl = term_peek(e, p.data), ql = term_peek(e, q.data);
#if !DEVICE
  // byte cells on both sides order as their bytes do, unsigned
  if (n && str_nar(p) == 2 && str_nar(q) == 2) {
    int c = memcmp((const u8*)(e.mem + pl) + p.off, (const u8*)(e.mem + ql) + q.off, n);
    if (c != 0) { return c < 0 ? 0 : 2; }
    n = 0;
  }
#endif
  for (u32 i = 0; i < n; i++) {
    if (err_spun(e.mem, &poll)) { return 1; }
    u32 x = str_cell(e.mem, pl, str_nar(p), p.off + i);
    u32 y = str_cell(e.mem, ql, str_nar(q), q.off + i);
    if (x != y) { return x < y ? 0 : 2; }
  }
  return p.len == q.len ? 1 : p.len < q.len ? 0 : 2;
}

INLINE u32 str_order_take(Env e, Term a, Term b) {
  u32 c = str_order_peek(e, a, b);
  term_sink(e, a); term_sink(e, b);
  return c;
}

// String.eq: two lengths that differ answer before a cell is read
INLINE bool str_eq_take(Env e, Term a, Term b) {
  bool eq = str_peek(e, a).len == str_peek(e, b).len && str_order_peek(e, a, b) == 1;
  term_sink(e, a); term_sink(e, b);
  return eq;
}

INLINE bool str_edge_take(Env e, Term s, Term sub, bool end) {
  StrParts p = str_peek(e, s), q = str_peek(e, sub);
  bool ok = q.len <= p.len;
  u32 off = ok && end ? p.len - q.len : 0, poll = 0;
  Loc pl = term_peek(e, p.data), ql = term_peek(e, q.data);
#if !DEVICE
  if (ok && q.len && str_nar(p) == 2 && str_nar(q) == 2) {
    ok = memcmp((const u8*)(e.mem + pl) + p.off + off, (const u8*)(e.mem + ql) + q.off, q.len) == 0;
    term_sink(e, s); term_sink(e, sub);
    return ok;
  }
#endif
  for (u32 i = 0; ok && i < q.len; i++) {
    if (err_spun(e.mem, &poll)) { ok = false; break; }
    ok = str_cell(e.mem, pl, str_nar(p), p.off + off + i)
      == str_cell(e.mem, ql, str_nar(q), q.off + i);
  }
  term_sink(e, s); term_sink(e, sub);
  return ok;
}

INLINE bool str_space(u32 c) { return c == 32 || (c >= 9 && c <= 13); }

INLINE Term str_trim_take(Env e, Term s, u32 ends) {
  StrParts p = str_peek(e, s);
  u32 lo = 0, hi = p.len, poll = 0;
  Loc l = term_peek(e, p.data);
  while ((ends & 1) && lo < hi && str_space(str_cell(e.mem, l, str_nar(p), p.off + lo))) {
    if (err_spun(e.mem, &poll)) { break; } lo++;
  }
  while ((ends & 2) && hi > lo && str_space(str_cell(e.mem, l, str_nar(p), p.off + hi - 1))) {
    if (err_spun(e.mem, &poll)) { break; } hi--;
  }
  return str_slice_take(e, s, lo, hi);
}

// mode: reverse=0, ASCII upper=1, ASCII lower=2, capitalize=3. Private visible copy
// on shared/static input; bounds changes alone never require a payload copy.
INLINE Term str_transform_take(Env e, Term s, u32 mode) {
  StrParts p = str_take(e, s);
  if (!p.len) { return str_view_owned(e, p); }
  p = str_reserve(e, p, 0, false, 2);
  if (err_seen(e.mem)) { return str_view_owned(e, p); }
  u32 poll = 0;
  for (u32 i = 0; i < (mode ? p.len : p.len / 2); i++) {
    if (err_spun(e.mem, &poll)) { break; }
    u32 c = str_at_peek(e, p, i);
    if (!mode) {
      str_put(e, p, i, str_at_peek(e, p, p.len - 1 - i));
      str_put(e, p, p.len - 1 - i, c);
    } else {
      bool upper = mode == 1 || (mode == 3 && i == 0);
      if (upper && c >= 97 && c <= 122) { c -= 32; }
      else if (!upper && c >= 65 && c <= 90) { c += 32; }
      str_put(e, p, i, c);
    }
  }
  return str_view_owned(e, p);
}

// Generic List fields are boxed. Seal before publishing to a container.
INLINE Term str_cons(Env e, Term h, Term t) {
  Loc l = heap_alloc(e, 1);
  if (err_seen(e.mem)) { term_sink(e, h); term_sink(e, t); return term_pak(CID_NIL, 0); }
  e.mem[l] = rfc_seal(e, h);
  e.mem[l + 1] = rfc_seal(e, t);
  return term_ctr(CID_CON, l);
}

INLINE Term str_to_list_take(Env e, Term s) {
  StrParts p = str_peek(e, s);
  Term out = term_pak(CID_NIL, 0);
  u32 poll = 0;
  for (u32 i = p.len; i > 0; i--) {
    if (err_spun(e.mem, &poll)) { break; }
    out = str_cons(e, term_pak(CID_CHR, str_at_peek(e, p, i - 1)), out);
  }
  term_sink(e, s);
  return out;
}

INLINE Term str_from_list_take(Env e, Term xs) {
  StrParts p = {0, 0, 0};
  u32 poll = 0;
  while (term_aux(xs) == CID_CON) {
    if (err_spun(e.mem, &poll)) { break; }
    Term f[2]; spare_free(e, 1, ctr_take(e, xs, 2, f));
    xs = f[1];
    p = str_reserve(e, p, 1, false, str_fit((u32)term_loc(f[0])));
    if (err_seen(e.mem)) { break; }
    str_put(e, p, p.len, (u32)term_loc(f[0])); p.len++;
  }
  term_sink(e, xs);
  return str_view_owned(e, p);
}

// Split emits views in reverse scan order directly into a forward list.
// words=true skips empty runs; split preserves every empty field.
INLINE Term str_split_take(Env e, Term s, u32 sep, bool words) {
  StrParts p = str_peek(e, s);
  Term out = term_pak(CID_NIL, 0);
  u32 hi = p.len, poll = 0;
  Loc l = term_peek(e, p.data);
  for (u64 j = (u64)p.len + 1; j > 0; j--) {
    if (err_spun(e.mem, &poll)) { break; }
    u32 i = (u32)(j - 1);
    bool cut = i == 0;
    if (!cut) {
      u32 c = str_cell(e.mem, l, str_nar(p), p.off + i - 1);
      cut = words ? str_space(c) : c == sep;
    }
    if (cut) {
      if (!words || hi > i) {
        StrParts q = {term_keep(e, p.data), p.off + i, hi - i};
        out = str_cons(e, str_view_owned(e, q), out);
      }
      hi = i ? i - 1 : 0;
    }
  }
  term_sink(e, s);
  return out;
}

// Bulk construction uses a single destination, including repeat's wide
// pre-multiply check. These also keep deep specimen construction bounded.
INLINE Term str_repeat_take(Env e, Term s, Nat n) {
  StrParts p = str_peek(e, s);
  if (p.len && n > STR_LIMIT / p.len) {
    err_post(e.mem, ERR_STRS); term_sink(e, s); return term_pak(CID_SNIL, 0);
  }
  StrParts q = str_alloc(e, (u64)p.len * n, str_nar(p));
  if (err_seen(e.mem)) { term_sink(e, s); return str_view_owned(e, q); }
  u32 poll = 0;
  for (u64 i = 0; p.len && i < n; i++) {
    if (err_spun(e.mem, &poll)) { break; }
    str_copy_cells(e, q, (u32)(i * p.len), p);
  }
  term_sink(e, s);
  return str_view_owned(e, q);
}

INLINE Term str_join_take(Env e, Term xs, Term sep) {
  StrParts p = {0, 0, 0}, sp = str_peek(e, sep);
  bool first = true;
  u32 poll = 0;
  while (term_aux(xs) == CID_CON) {
    if (err_spun(e.mem, &poll)) { break; }
    Term f[2]; spare_free(e, 1, ctr_take(e, xs, 2, f)); xs = f[1];
    StrParts q = str_peek(e, f[0]);
    u64 add = (u64)q.len + (first ? 0 : sp.len);
    p = str_reserve(e, p, add, false, str_nar(q) < str_nar(sp) || first ? str_nar(q) : str_nar(sp));
    if (!err_seen(e.mem)) {
      if (!first) { str_copy_cells(e, p, p.len, sp); p.len += sp.len; }
      str_copy_cells(e, p, p.len, q); p.len += q.len;
    }
    term_sink(e, f[0]); first = false;
  }
  term_sink(e, xs); term_sink(e, sep);
  return str_view_owned(e, p);
}

// A search cursor borrows text/needle and owns exactly one packed prefix
// table. All consumers close it, including early results and device errors.
// Empty needles are handled by each public contract; one-cell needles and
// needles longer than the text do not allocate a table.
#define STR_ABSENT (~0ull)
typedef struct {
  StrParts text, needle;
  Loc table;
  Cls cls;
  u32 pos, matched, poll;
} StrSearch;

INLINE void str_scratch_free(Env e, Cls cls, Loc l) {
  if (err_seen(e.mem)) {
    // heap_free intentionally stops on a sticky error. This private scratch
    // block has no children: recycle it locally without clearing the error
    // or publishing to the shared bank after a device failure.
    e.mem[l] = ALC_AT(e, cls);
    ALC_AT(e, cls) = l;
    ALC_LEN(e, cls) += 1ull << cls;
  } else {
    heap_free(e, cls, l);
  }
}

INLINE void str_search_close(Env e, THR StrSearch* k) {
  if (k->table) { str_scratch_free(e, k->cls, k->table); k->table = 0; }
}

INLINE StrSearch str_search_open(Env e, StrParts text, StrParts needle) {
  StrSearch k = {text, needle, 0, 0, 0, 0, 0};
  if (needle.len <= 1 || needle.len > text.len || err_seen(e.mem)) { return k; }
  k.cls = buf_wcls(cls_fit(needle.len));
  Loc table = heap_alloc(e, k.cls);
  if (err_seen(e.mem)) { return k; }
  k.table = table;
  blk_write(e.mem, false, table, 0, 0);
  for (u32 i = 1, j = 0; i < needle.len; i++) {
    if (err_spun(e.mem, &k.poll)) { break; }
    u32 c = str_at_peek(e, needle, i);
    while (j && c != str_at_peek(e, needle, j)) {
      if (err_spun(e.mem, &k.poll)) { break; }
      j = (u32)blk_read(e.mem, false, table, j - 1);
    }
    if (err_seen(e.mem)) { break; }
    if (c == str_at_peek(e, needle, j)) { j++; }
    blk_write(e.mem, false, table, i, j);
  }
  return k;
}

INLINE bool str_search_next(Env e, THR StrSearch* k, bool overlap, THR u32* at) {
  if (!k->needle.len || k->needle.len > k->text.len || err_seen(e.mem)) { return false; }
  while (k->pos < k->text.len) {
    if (err_spun(e.mem, &k->poll)) { return false; }
    u32 c = str_at_peek(e, k->text, k->pos++);
    while (k->matched && c != str_at_peek(e, k->needle, k->matched)) {
      if (err_spun(e.mem, &k->poll)) { return false; }
      k->matched = (u32)blk_read(e.mem, false, k->table, k->matched - 1);
    }
    if (c == str_at_peek(e, k->needle, k->matched)) { k->matched++; }
    if (k->matched == k->needle.len) {
      *at = k->pos - k->needle.len;
      k->matched = overlap && k->needle.len > 1
        ? (u32)blk_read(e.mem, false, k->table, k->matched - 1) : 0;
      return true;
    }
  }
  return false;
}

// Consume both inputs. mode: first=0, last overlapping start=1, count=2.
INLINE u64 str_search_take(Env e, Term s, Term needle, u32 mode) {
  StrParts p = str_peek(e, s), q = str_peek(e, needle);
  u64 out = mode == 2 ? 0 : STR_ABSENT;
  if (!q.len) { out = mode == 0 ? 0 : (u64)p.len + (mode == 2); }
#if !DEVICE
  // The first hit of a short needle in byte cells, as a head is searched
  // for "\r\n\r\n" or a field's name: memchr to each candidate for the
  // needle's first byte, then one compare. A needle this short bounds
  // the compares at 32 a byte; a longer one keeps the prefix table's
  // linear walk.
  else if (mode == 0 && q.len <= 32 && str_nar(p) == 2 && str_nar(q) == 2) {
    const u8* h = (const u8*)(e.mem + term_peek(e, p.data)) + p.off;
    const u8* n = (const u8*)(e.mem + term_peek(e, q.data)) + q.off;
    if (q.len <= p.len) {
      const u8* at = h;
      const u8* end = h + (p.len - q.len) + 1;
      while (at < end && (at = (const u8*)memchr(at, n[0], (size_t)(end - at))) != NULL) {
        if (memcmp(at, n, q.len) == 0) { out = (u64)(at - h); break; }
        at++;
      }
    }
  }
#endif
  else {
    StrSearch k = str_search_open(e, p, q);
    u32 at;
    while (str_search_next(e, &k, mode == 1, &at)) {
      out = mode == 2 ? out + 1 : at;
      if (mode == 0) { break; }
    }
    str_search_close(e, &k);
  }
  term_sink(e, s); term_sink(e, needle);
  return out;
}

INLINE Term str_find_take(Env e, Term s, Term needle, bool last) {
  u64 at = str_search_take(e, s, needle, last ? 1 : 0);
  if (at == STR_ABSENT || err_seen(e.mem)) { return term_pak(CID_NONE, 0); }
  // Nat is a raw word even in a generic Some payload.
  return str_node(e, CID_SOME, 1, at, 0, 0);
}

// Borrow a range, return an owned zero-copy window (empty never pins).
INLINE Term str_window(Env e, StrParts p, u32 lo, u32 hi) {
  StrParts q = {lo < hi ? term_keep(e, p.data) : 0, p.off + lo, hi - lo};
  return str_view_owned(e, q);
}

// Consumes the destination parts, borrows the source; one growing output.
INLINE void str_push_range(Env e, THR StrParts* out, StrParts p, u32 lo, u32 hi) {
  if (lo == hi || err_seen(e.mem)) { return; }
  p.off += lo; p.len = hi - lo;
  *out = str_reserve(e, *out, p.len, false, str_nar(p));
  if (!err_seen(e.mem)) { str_copy_cells(e, *out, out->len, p); out->len += p.len; }
}

INLINE Term str_replace_take(Env e, Term s, Term old, Term value) {
  StrParts p = str_peek(e, s), q = str_peek(e, old), r = str_peek(e, value);
  StrParts out = {0, 0, 0};
  StrSearch k = str_search_open(e, p, q);
  u32 at = 0, lo = 0, poll = 0;
  // A counting pass sizes the output once: no doubling, no recopying.
  u64 hits = q.len ? 0 : (u64)p.len + 1;
  while (q.len && str_search_next(e, &k, false, &at)) { hits++; }
  k.pos = k.matched = 0;
  out = str_reserve(e, out, p.len + hits * r.len - (q.len ? hits * q.len : 0), false,
    hits && str_nar(r) < str_nar(p) ? str_nar(r) : str_nar(p));
  if (!q.len) {
    for (u64 i = 0; i <= p.len; i++) {
      if (err_spun(e.mem, &poll)) { break; }
      str_push_range(e, &out, r, 0, r.len);
      if (i < p.len) { str_push_range(e, &out, p, (u32)i, (u32)i + 1); }
    }
  } else {
    while (str_search_next(e, &k, false, &at)) {
      str_push_range(e, &out, p, lo, at);
      str_push_range(e, &out, r, 0, r.len);
      lo = at + q.len;
    }
    str_push_range(e, &out, p, lo, p.len);
  }
  str_search_close(e, &k);
  term_sink(e, s); term_sink(e, old); term_sink(e, value);
  return str_view_owned(e, out);
}

// Owns a private forward list during construction; seal every link before
// publication. Consumes field, mutates only fresh unaliased list metadata.
INLINE void str_list_push(Env e, THR Term* out, THR Loc* tail, Term field) {
  if (err_seen(e.mem)) { term_sink(e, field); return; }
  Term node = rfc_seal(e, str_cons(e, field, term_pak(CID_NIL, 0)));
  if (err_seen(e.mem)) { term_sink(e, node); return; }
  if (*tail) { e.mem[*tail] = node; } else { *out = node; }
  *tail = term_peek(e, node) + 1;
}

INLINE Term str_split_on_take(Env e, Term s, Term sep) {
  StrParts p = str_peek(e, s), q = str_peek(e, sep);
  Term out = term_pak(CID_NIL, 0);
  Loc tail = 0;
  StrSearch k = str_search_open(e, p, q);
  u32 at, lo = 0;
  while (str_search_next(e, &k, false, &at)) {
    str_list_push(e, &out, &tail, str_window(e, p, lo, at));
    lo = at + q.len;
  }
  if (!err_seen(e.mem)) { str_list_push(e, &out, &tail, str_window(e, p, lo, p.len)); }
  str_search_close(e, &k);
  term_sink(e, s); term_sink(e, sep);
  return out;
}

// Own both fields; nested tuple payloads obey the same sealing protocol.
INLINE Term str_pair(Env e, Term a, Term b) {
  if (err_seen(e.mem)) { term_sink(e, a); term_sink(e, b); return 0; }
  Loc l = heap_alloc(e, 1);
  if (err_seen(e.mem)) { term_sink(e, a); term_sink(e, b); return 0; }
  e.mem[l] = rfc_seal(e, a); e.mem[l + 1] = rfc_seal(e, b);
  return term_ctr(CID_TUPLE, l);
}

INLINE Term str_partition_take(Env e, Term s, Term sep) {
  StrParts p = str_peek(e, s), q = str_peek(e, sep);
  StrSearch k = str_search_open(e, p, q);
  u32 at;
  bool found = str_search_next(e, &k, false, &at);
  str_search_close(e, &k);
  Term a = s, b = term_pak(CID_SNIL, 0), c = b;
  if (found) {
    a = str_window(e, p, 0, at); b = sep;
    c = str_window(e, p, at + q.len, p.len);
    term_sink(e, s);
  } else { term_sink(e, sep); }
  return str_pair(e, a, str_pair(e, b, c));
}

INLINE bool str_line_break(u32 c) {
  return (c >= 10 && c <= 13) || (c >= 28 && c <= 30)
    || c == 133 || c == 8232 || c == 8233;
}

INLINE Term str_splitlines_take(Env e, Term s) {
  StrParts p = str_peek(e, s);
  Term out = term_pak(CID_NIL, 0);
  Loc tail = 0;
  u32 lo = 0, poll = 0;
  for (u32 i = 0; i < p.len; i++) {
    if (err_spun(e.mem, &poll)) { break; }
    u32 c = str_at_peek(e, p, i);
    if (str_line_break(c)) {
      str_list_push(e, &out, &tail, str_window(e, p, lo, i));
      if (c == 13 && i + 1 < p.len && str_at_peek(e, p, i + 1) == 10) { i++; }
      lo = i + 1;
    }
  }
  if (lo < p.len && !err_seen(e.mem)) { str_list_push(e, &out, &tail, str_window(e, p, lo, p.len)); }
  term_sink(e, s);
  return out;
}

// Consume s; raw U32 fill values are ordinary cells. mode: start=0, end=1,
// sign-aware zero fill=2. Widen/check before narrowing or subtracting.
INLINE Term str_pad_take(Env e, Term s, Nat width, u32 c, u32 mode) {
  u32 len = str_peek(e, s).len;
  if (width <= len) { return s; }
  if (width > STR_LIMIT) { err_post(e.mem, ERR_STRS); term_sink(e, s); return term_pak(CID_SNIL, 0); }
  u32 n = (u32)(width - len);
  StrParts p = str_reserve(e, str_take(e, s), n, mode != 1, str_fit(c));
  if (err_seen(e.mem)) { return str_view_owned(e, p); }
  u32 sign = 0, first = len ? str_at_peek(e, p, 0) : 0;
  if (mode == 2 && (first == '+' || first == '-')) { sign = 1; }
  if (mode != 1) { p.off -= n; }
  p.len = (u32)width;
  u32 start = mode == 1 ? len : sign, poll = 0;
  if (sign) { str_put(e, p, 0, first); }
  for (u32 i = 0; i < n; i++) {
    if (err_spun(e.mem, &poll)) { break; }
    str_put(e, p, start + i, c);
  }
  return str_view_owned(e, p);
}

INLINE u32 str_hash_take(Env e, Term s) {
  StrParts p = str_peek(e, s);
  u32 h = 2166136261u, poll = 0;
  for (u32 i = 0; i < p.len; i++) {
    if (err_spun(e.mem, &poll)) { break; }
    u32 c = str_at_peek(e, p, i);
    for (u32 shift = 0; shift < 32; shift += 8) { h = (h ^ ((c >> shift) & 255)) * 16777619u; }
  }
  term_sink(e, s);
  return h;
}

// Regex
// =====

// The Pike VM of base (Regex.exec.go, full and ne off): its priorities, its
// dead threads (a pc or a slot out of range), so a forged Regex never
// faults. One scratch block per call: the program packed op:8|a:24|b:24 (an
// IChr keeps its 32 bits in a:b), the set ranges lo|hi<<32, a visited row
// stamped by generation (never cleared per step), the raw and the closed
// thread banks (a pc row and a slot row each), the current and the best
// slots, the closure stack (a visited pc nets <= 2 words: 2m + 1 deep). A slot is a
// position, RE_NONE unset; RE_NONE is also the char before 0 and at the end.
#define RE_NONE (~0ull)
#define RE_DEAD 0xFFFFFFu
#define RE_CHR   0
#define RE_ANY   1
#define RE_SET   2
#define RE_NSET  3
#define RE_SPLIT 4
#define RE_JMP   5
#define RE_SAVE  6
#define RE_BOL   7
#define RE_EOL   8
#define RE_WORDB 9
#define RE_MATCH 10
#define re_ins(op, a, b) ((u64)(op) << 48 | (u64)(a) << 24 | (u64)(b))
#define re_arg(n, m) ((n) < (m) ? (u64)(n) : (u64)RE_DEAD)

INLINE bool re_word(u64 c) {
  return (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || c == 95
    || (c >= 97 && c <= 122);
}

INLINE void re_copy(Env e, Loc dst, Loc src, u64 n) {
  for (u64 j = 0; j < n; j++) { e.mem[dst + j] = e.mem[src + j]; }
}

// Consumes prog and s; returns a boxed Maybe<Match>.
INLINE Term re_exec_take(Env e, Term prog, u64 ng, Term s, u64 at, bool anchored) {
  StrParts p = str_peek(e, s);
  Term out = term_pak(CID_NONE, 0);
  u64 m = 0, sets = 0, len = p.len, ns = 2 * (1 + ng);
  u32 poll = 0;
  for (Term t = prog; term_aux(t) == CID_CON; m++) {
    if (err_spun(e.mem, &poll)) { break; }
    Loc l = term_peek(e, t);
    Term h = e.mem[l];
    t = e.mem[l + 1];
    if (term_aux(h) != CID_ISET) { continue; }
    for (Term r = e.mem[term_peek(e, h) + 1]; term_aux(r) == CID_CON; sets++) {
      r = e.mem[term_peek(e, r) + 1];
    }
  }
  u64 words = 6 * m + sets + 2 * m * (1 + ns) + 2 * ns + 2;
  if (m >= RE_DEAD || sets >= RE_DEAD || ng >= 1u << 22 || words > STR_LIMIT) {
    err_post(e.mem, ERR_HEAP);
  }
  Cls cls = cls_fit((u32)words);
  Loc P = err_seen(e.mem) ? 0 : heap_alloc(e, cls);
  if (err_seen(e.mem)) { term_sink(e, prog); term_sink(e, s); return out; }
  Loc S = P + m, V = S + sets, RP = V + m, RS = RP + m, CP = RS + m * ns;
  Loc CS = CP + m, CUR = CS + m * ns, BEST = CUR + ns, STK = BEST + ns;
  u64 i = 0, k = 0;
  for (Term t = prog; term_aux(t) == CID_CON; i++) {
    if (err_spun(e.mem, &poll)) { break; }
    Loc l = term_peek(e, t);
    Term h = e.mem[l];
    t = e.mem[l + 1];
    u64 c = term_aux(h), w = re_ins(RE_MATCH, 0, 0);
    Loc f = c == CID_ISET || c == CID_ISPLIT || c == CID_IJMP || c == CID_ISAVE
      ? term_peek(e, h) : 0;
    if (c == CID_ICHR) { w = re_ins(RE_CHR, 0, 0) | (u32)term_loc(h); }
    else if (c == CID_IANY) { w = re_ins(RE_ANY, 0, 0); }
    else if (c == CID_ISPLIT) { w = re_ins(RE_SPLIT, re_arg(e.mem[f], m), re_arg(e.mem[f + 1], m)); }
    else if (c == CID_IJMP) { w = re_ins(RE_JMP, re_arg(e.mem[f], m), 0); }
    else if (c == CID_ISAVE) { w = re_ins(RE_SAVE, re_arg(e.mem[f], ns), 0); }
    else if (c == CID_IBOL) { w = re_ins(RE_BOL, term_loc(h) & 1, 0); }
    else if (c == CID_IEOL) { w = re_ins(RE_EOL, term_loc(h) & 1, 0); }
    else if (c == CID_IWORDB) { w = re_ins(RE_WORDB, term_loc(h) & 1, 0); }
    else if (c == CID_ISET) {
      u64 k0 = k;
      for (Term r = e.mem[f + 1]; term_aux(r) == CID_CON; k++) {
        Loc q = term_peek(e, r), u = term_peek(e, e.mem[q]);
        e.mem[S + k] = (u64)(u32)e.mem[u] | (u64)(u32)e.mem[u + 1] << 32;
        r = e.mem[q + 1];
      }
      w = re_ins(e.mem[f] & 1 ? RE_NSET : RE_SET, k0, k - k0);
    }
    e.mem[P + i] = w;
    e.mem[V + i] = 0;
  }
  if (at > len) { at = len; }
  u64 nraw = 0, best = 0;
  u64 prev = at ? str_at_peek(e, p, (u32)at - 1) : RE_NONE;
  for (u64 pos = at; !err_seen(e.mem); pos++) {
    u64 fresh = !best && (pos == at || !anchored), gen = pos - at + 1, ncl = 0;
    if (nraw + fresh == 0) { break; }
    u64 c = pos < len ? str_at_peek(e, p, (u32)pos) : RE_NONE;
    for (u64 r = 0; r < nraw + fresh; r++) {
      if (r < nraw) { re_copy(e, CUR, RS + r * ns, ns); }
      else { for (u64 j = 0; j < ns; j++) { e.mem[CUR + j] = RE_NONE; } }
      u64 sp = 1;
      e.mem[STK] = r < nraw ? e.mem[RP + r] : 0;
      while (sp) {
        if (err_spun(e.mem, &poll)) { break; }
        u64 x = e.mem[STK + --sp];
        if (x >> 63) { e.mem[CUR + (u32)x] = e.mem[STK + --sp]; continue; }
        if (x >= m || e.mem[V + x] == gen) { continue; }
        e.mem[V + x] = gen;
        u64 w = e.mem[P + x];
        u32 op = (u32)(w >> 48), a = (u32)(w >> 24) & RE_DEAD, b = (u32)w & RE_DEAD;
        bool go = false;
        if (op == RE_SPLIT) { e.mem[STK + sp++] = b; e.mem[STK + sp++] = a; }
        else if (op == RE_JMP) { e.mem[STK + sp++] = a; }
        else if (op == RE_SAVE) {
          if (a < ns) {
            e.mem[STK + sp++] = e.mem[CUR + a];
            e.mem[STK + sp++] = 1ull << 63 | a;
            e.mem[CUR + a] = pos;
            go = true;
          }
        }
        else if (op == RE_BOL) { go = pos == 0 || (a && prev == 10); }
        else if (op == RE_EOL) { go = c == RE_NONE || (c == 10 && (a || pos + 1 == len)); }
        else if (op == RE_WORDB) { go = (pos || len) && (a != 0) != (re_word(prev) != re_word(c)); }
        else { e.mem[CP + ncl] = x; re_copy(e, CS + ncl++ * ns, CUR, ns); }
        if (go) { e.mem[STK + sp++] = x + 1; }
      }
    }
    nraw = 0;
    for (u64 t = 0; t < ncl; t++) {
      u64 x = e.mem[CP + t], w = e.mem[P + x];
      u32 op = (u32)(w >> 48), a = (u32)(w >> 24) & RE_DEAD, b = (u32)w & RE_DEAD;
      if (op == RE_MATCH) { re_copy(e, BEST, CS + t * ns, ns); best = 1; break; }
      bool eat = op == RE_CHR ? (u32)w == c : op == RE_ANY ? c != 10 : false;
      if (op == RE_SET || op == RE_NSET) {
        for (u64 j = 0; j < b && !eat; j++) {
          u64 r = e.mem[S + a + j];
          eat = (u32)r <= c && c <= r >> 32;
        }
        eat = eat != (op == RE_NSET);
      }
      if (eat && c != RE_NONE) {
        e.mem[RP + nraw] = x + 1;
        re_copy(e, RS + nraw++ * ns, CS + t * ns, ns);
      }
    }
    prev = c;
    if (c == RE_NONE) { break; }
  }
  u64 lo = e.mem[BEST], hi = e.mem[BEST + 1];
  if (best && lo != RE_NONE && hi != RE_NONE && !err_seen(e.mem)) {
    Term gs = term_pak(CID_NIL, 0);
    for (u64 g = ng; g > 0; g--) {
      u64 a = e.mem[BEST + 2 * g], b = e.mem[BEST + 2 * g + 1];
      Term f = term_pak(CID_NONE, 0);
      if (a != RE_NONE && b != RE_NONE) {
        f = str_node(e, CID_SOME, 1, rfc_seal(e, str_node(e, CID_TUPLE, 2, a, b, 0)), 0, 0);
      }
      gs = str_cons(e, f, gs);
    }
    out = str_node(e, CID_SOME, 1,
      rfc_seal(e, str_node(e, CID_MATCH, 3, lo, hi, rfc_seal(e, gs))), 0, 0);
  }
  str_scratch_free(e, cls, P);
  term_sink(e, prog); term_sink(e, s);
  return out;
}

// Ring
// ====

// planes LANES wide: a smaller bag has deeper rings in the same region
#define ring_word(H, r, w) ((H) + RING_OFF + (w) * LANES + (r))
#define ring_slot(H, r, p) ring_word(H, r, (p) & (RING_LEN - 1))
#define ring_get(H, r)     ((DEV u32*)ring_word(H, r, RING_LEN))
#define ring_put(H, r)     ((DEV u32*)ring_word(H, r, RING_LEN + 1))

INLINE u32 ring_lap(u32 pos) {
  return ~(u32)(pos / RING_LEN) & 1;
}

INLINE void ring_push(Corpus H, Ring r, Term tsk) {
  u32 pos = a32_add(ring_put(H, r), 1);
  if (pos - a32_load(ring_get(H, r)) >= RING_LEN) {
    err_post(H, ERR_RING);
    return;
  }
  DEV u32* lo = (DEV u32*)ring_slot(H, r, pos);
  a32_store(lo, (u32)tsk);
  a32_store_rel(lo + 1, (u32)(tsk >> 32) | (ring_lap(pos) << 31));
}

INLINE Ring ring_flip(u32 i) {
  return (i % CUBE_T << CUBE_LOG) + i / CUBE_T;
}

#define ring_pick(b, s, c) ((b) + (s) * (g32_add(c, 1) & (CUBE_T - 1)))

// Task
// ====

INLINE Loc task_node(Env e, Fid fid, Term cont, u32 idx, u32 rem) {
  u32 ar  = fid_arity(fid);
  Loc loc = heap_alloc(e, cls_fit(ar + 2));
  for (u32 i = 0; rem && i < ar; i += 1) {
    e.mem[loc + i] = TERM_HOLE;
  }
  e.mem[loc + ar]     = cont;
  e.mem[loc + ar + 1] = ((u64)idx << 32) | rem;
  return loc;
}

INLINE Loc task_tail(Term t) {
  return term_loc(t) + fid_arity((u32)term_aux(t));
}

INLINE Term task_deliver(Corpus H, Term cont, u32 idx, THR Term* v, u32 n) {
  Loc at = cont == TERM_HOLE ? H_ROOT_WORD : term_loc(cont) + idx;
  for (u32 j = 0; j < WL_RESW; j += 1) {
    if (j < n) {
      H[at + j] = v[j];
    }
  }
  if (cont == TERM_HOLE) {
    a32_store_rel(a32_at(H, H_ROOT_DONE), n + 1);
    return 0;
  }
  Loc tl = task_tail(cont);
  if (a32_sub_rel(a32_at(H, tl + 1), 1) == 1) {
    a32_acq(a32_at(H, tl + 1));
    return cont;
  }
  return 0;
}

INLINE void task_deal(Corpus H, Term join, u32 base, u32 stride, Cur cur) {
  Loc loc = term_loc(join);
  u32 ar  = fid_arity((u32)term_aux(join));
  u32 g   = 0;
  if (stride == 0) {
    u32 rem = (u32)H[loc + ar + 1];
    g = a32_add(a32_at(H, H_CURSOR), rem);
  }
  for (u32 i = 0; i < ar; i += 1) {
    Term k = H[loc + i];
    if (term_tag(k) == TAG_TSK) {
      H[loc + i] = TERM_HOLE;
      Ring to;
      if (stride != 0) {
        to = ring_pick(base, stride, cur);
      } else {
        to = ring_flip(g & (u32)(LANES - 1));
        g += 1;
      }
      ring_push(H, to, k);
    }
  }
}

// Root
// ====

INLINE bool root_done(Corpus H) {
  return a32_load_acq(a32_at(H, H_ROOT_DONE)) != 0;
}

static u32 root_take(Corpus H, THR Term* v) {
  u32 n = a32_load_acq(a32_at(H, H_ROOT_DONE)) - 1;
  for (u32 j = 0; j < n; j += 1) {
    v[j] = H[H_ROOT_WORD + j];
  }
  a32_store(a32_at(H, H_ROOT_DONE), 0);
  return n;
}

// Spins
// =====

${spins}

// Work
// ====

// A host self-jump is a tail call: as a loop, clang hoisted constants into
// symreg's entry (3.05 s against 2.51 s).
#if !DEVICE
#undef  WL_SPIN
#undef  WL_SPUN
#undef  WL_AGAIN
#define WL_SPIN
#define WL_SPUN
#define WL_AGAIN(F) __attribute__((musttail)) return WL_##F(WL_ALL)

typedef Reply (PRESERVE(preserve_none) *WlFn)(WL_SIG);
#define WL_X(F) WL_FN WL_##F(WL_SIG);
WL_TABLE WL_X(FID_ENTER)
#undef WL_X
#define WL_X(F) WL_##F,
static const WlFn wl_tab[] = { WL_TABLE };
#undef WL_X
#endif

static Reply work_loop(Env e, Stk sp, Term t, u32 seq) {
  WL_BANK
  u32 rn = 0;
  r0 = t;
#if DEVICE
  Fid fid   = FID_ENTER;
  u32 wpoll = 0;
  for (;;) {
  if (err_spun(e.mem, &wpoll)) {
    return 0;
  }
  switch (fid) {
#else
  return WL_FID_ENTER(WL_ALL);
}
#endif

// Segments
// ========

${segs}

// A task enters through its words: a continuation's results ride r0.. and
// its parameters the stack; any other segment's parameters ride r0..
  WL_CASE(FID_ENTER)
  {
    Term t = r0;
    WL_OPEN
    Fid f   = (u32)term_aux(t);
    Loc a   = term_loc(t);
    u32 war = fid_arity(f);
    WL_FRAME(t)
    seq |= fid_nofk(f) << 1;
    if (fid_seqk(f)) {
      u32 rw = fid_resw(f);
      WL_LOAD(a + war - rw, rw)
      WL_ARGS(a, war - rw + 1)
    } else {
      WL_LOAD(a, war)
    }
    heap_free(e, cls_fit(war + 2), a);
    WL_DYN(f);
  }}

  WL_CASE(FID_IO_EMIT)
  {
    Term x = r0;
    WL_OPEN
    Loc l = heap_alloc(e, 0);
    e.mem[l] = x;
    r0 = term_ctr(CID_EMIT, l);
    WL_RETN(1);
  }}

  WL_CASE(FID_CLO_APPLY)
  {
    Term fun = r0;
    Term arg = r1;
    WL_OPEN
    Fid f    = (Fid)term_aux(fun);
    u32 war  = fid_arity(f) - 1;
    Loc a    = term_loc(fun);
    WL_LOAD(a, war)
    spare_free(e, cls_fit(war), a);
    WL_LAST(arg)
    WL_DYN(f);
  }}

  WL_CASE(FID_EXIT)
  {
    u32  n = rn;
    Term rv[WL_RESW];
    WL_SAVE(rv)
    WL_OPEN
    if (err_seen(e.mem)) {
      return 0;
    }
    sp -= 2 * LANE_STEP;
    Term cont = STK(0);
    u32  idx  = (u32)STK(1);
    if (cont != TERM_HOLE && fid_seqk((u32)term_aux(cont))) {
      Fid wf = (u32)term_aux(cont);
      Loc wa = term_loc(cont);
      u32 wn = fid_arity(wf);
      WL_FRAME(cont)
      seq = (seq & 1) | fid_nofk(wf) << 1;
      WL_ARGS(wa, wn - n + 1)
      heap_free(e, cls_fit(wn + 2), wa);
      WL_TAKE(rv)
      WL_DYN(wf);
    }
    return task_deliver(e.mem, cont, idx, rv, n);
  }}

#if DEVICE
  default: {
    err_post(e.mem, ERR_FIDS);
    return 0;
  }
  }
  }
}
#endif

// Monk
// ====

// One turn on a ring: its head task below put0 runs (a growing lane skips a
// fork-free one). The host grows a row ring by ring and drains a ring; a
// device lane does both.
INLINE u32 monk_step(Env e, Stk stk, Ring rg, u32 put0, bool seq, u32 base,
  u32 stride, Cur cur) {
  Corpus   H   = e.mem;
  DEV u32* get = ring_get(H, rg);
  if (*get == put0) {
    return 0;
  }
  DEV u32* lo = (DEV u32*)ring_slot(H, rg, *get);
  u32      hi = a32_load_acq(lo + 1);
  Term     t  = (((u64)hi << 32) | a32_load(lo)) & ~RFC_BIT;
  if ((hi >> 31) != ring_lap(*get) || (!seq && fid_nofk((u32)term_aux(t)))) {
    return 0;
  }
  a32_store(get, *get + 1);
  u32 spin = 0;
  for (;;) {
    Reply r = work_loop(e, stk, t, seq);
    if (r == 0) {
      return 2;
    }
    if ((u32)H[task_tail(r) + 1] == 0) {
      if (err_spun(H, &spin)) {
        return 2;
      }
      if (stride != 0 && fid_nofk((u32)term_aux(r))) {
        ring_push(H, ring_pick(base, stride, cur), r);
        return 2;
      }
      t      = r;
      seq    = false;
      stride = 0;
      continue;
    }
    task_deal(H, r, base, stride, cur);
    return 1;
  }
}

// Dev
// ===

// TG_HOLD words of threadgroup memory hold one group per Apple core
// (bitonic 1.35x without). A grow pass ends when its group is full or
// nothing grew, so a spine of forks unrolls whole.

#if DEVICE

INLINE void dev_cut(Env e) {
  if (err_seen(e.mem)) {
    return;
  }
  for (Cls c = 0; c < NCLS_ALL; c += 1) {
    u64 gen = (u64)KEEP(c) << c;
    while (ALC_LEN(e, c) >= gen) {
      Loc head = ALC_AT(e, c);
      Loc tail = head;
      for (u32 i = KEEP(c); --i;) {
        tail = e.mem[tail];
      }
      ALC_AT(e, c)    = e.mem[tail];
      ALC_LEN(e, c)  -= gen;
      e.mem[tail]     = 0;
      bank_push(e.mem, c, head);
    }
  }
}

// Pass 2, one group: each bank's [top, wr) slides onto rd, CUBE_T entries a
// step (loads, barrier, stores: rd <= top), off the host's pages.
INLINE void bank_pack(Corpus H, u32 lane) {
  for (Cls c = 0; c < NCLS_ALL; c += 1) {
    DEV Bank* b  = bank_at(H, c);
    u32       rd = b->rd;
    u32       n  = b->wr - b->top;
    for (u32 i = 0; i < n; i += CUBE_T) {
      Term v = i + lane < n ? H[b->off + b->top + i + lane] : 0;
      BAR();
      if (i + lane < n) {
        H[b->off + rd + i + lane] = v;
      }
    }
    BAR();
    if (lane == 0) {
      b->rd = b->wr = b->top = rd + n;
    }
  }
}

// One kernel: pass 0 grows the frontier, pass 1 drains each lane's ring,
// pass 2 packs the banks.
#ifdef __METAL_VERSION__
kernel void bend_dev(Corpus H [[buffer(0)]], constant u32& pass [[buffer(1)]],
  threadgroup volatile u64* hold [[threadgroup(0)]],
  u32 grids [[threadgroups_per_grid]],
  u32 row [[threadgroup_position_in_grid]],
  u32 lane [[thread_position_in_threadgroup]]) {
#else
extern "C" __global__ void bend_dev(Corpus H, u32 pass) {
  extern __shared__ volatile u64 hold[];
  u32 grids = gridDim.x;
  u32 row   = blockIdx.x;
  u32 lane  = threadIdx.x;
#endif
  if (pass == 2) {
    bank_pack(H, lane);
    return;
  }
  u32  stride = grids == 1 ? CUBE_G : 1;
  u32  me     = row * CUBE_T + stride * lane;
  Ring rg     = pass ? ring_flip(me) : me;
  Env  e      = { H, H + ALC_OFF + me };
  Stk  stk    = (Stk)(H + STAK_OFF + me);
  if (lane == 0) {
    hold[0] = 0;
  }
  GA32 tg_cur, tg_grew, tg_has;
  g32_ini(&tg_cur);
  g32_ini(&tg_grew);
  g32_ini(&tg_has);
  BAR();
  u32 put0      = a32_load(ring_put(H, rg));
  u32 seen_has  = 0;
  u32 seen_grew = 0;
  for (;;) {
    if (pass) {
      if (*ring_get(H, rg) == put0 || err_seen(H)) {
        break;
      }
    } else {
      put0 = a32_load(ring_put(H, rg));
      u32 vote = put0 != a32_load(ring_get(H, rg));
      if (lane == 0 && (err_seen(H) || root_done(H))) {
        vote = CUBE_T;
      }
      g32_add(&tg_has, vote);
      BAR();
      u32 has = g32_get(&tg_has);
      if (has - seen_has >= CUBE_T) {
        break;
      }
      seen_has = has;
    }
    u32 ran = monk_step(e, stk, rg, put0, pass, pass ? rg : row * CUBE_T,
      pass ? 0 : stride, &tg_cur);
    if (!pass) {
      if (ran == 1) {
        g32_add(&tg_grew, 1);
      }
      BARD();
      u32 grew = g32_get(&tg_grew);
      if (grew == seen_grew) {
        break;
      }
      seen_grew = grew;
    }
  }
  dev_cut(e);
}

#endif

// Window
// ======

// Linux's window fill (the Mac's is window_msl): an Image is a quadtree over
// 2^k x 2^k (Qua splits tl, tr, bl, br; Pix is 0xRRGGBB).
#if defined(__linux__) || defined(BEND_RTC)

INLINE u32 window_pix(Corpus H, Term t, u32 k, u32 x, u32 y) {
  for (u32 i = k; term_tag(t) == TAG_CTR;) {
    u32 j = 0;
    if (i > 0) {
      i -= 1;
      j = ((y >> i) & 1) * 2 + ((x >> i) & 1);
    }
    Loc l = term_rfc(t) ? H[term_loc(t)] >> 24 : term_loc(t);
    t = H[l + j];
  }
  return (u32)term_loc(t) & 0xFFFFFF;
}

#ifdef BEND_RTC
extern "C" __global__ void window_dev(Corpus H, Term root, u32 w, u32 h,
  u32 k, u32* out) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    out[y * w + x] = window_pix(H, root, k, x, y);
  }
}
#endif

#endif

#if !DEVICE

// Row
// ===

static void row_grow(Env e, Stk stk, u32 base, u32 stride, u32 want) {
  Corpus H = e.mem;
  u32 cur = 0;
  for (;;) {
    u32 put0[CUBE_T];
    u32 has = 0;
    for (u32 i = 0; i < CUBE_T; i += 1) {
      put0[i] = *ring_put(H, base + i * stride);
      has += put0[i] != *ring_get(H, base + i * stride);
    }
    if (root_done(H) || has >= want) {
      return;
    }
    u32 grew = 0;
    u32 ran  = 0;
    for (u32 i = 0; i < CUBE_T && ran != 2; i += 1) {
      ran   = monk_step(e, stk, base + i * stride, put0[i], false, base,
        stride, &cur);
      grew += ran == 1;
    }
    if (grew == 0) {
      return;
    }
  }
}

// Pool
// ====

static void* pool_try(void* at, u64 bytes) {
  return mmap(at, bytes, PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
}

static void* pool_mmap(u64 bytes) {
  void* p = pool_try(NULL, bytes);
  if (p == MAP_FAILED) {
    err_fail("reservation failed");
  }
  return p;
}

static Term* pool_stack(void) {
  u64   len = 1ull << 31;
  char* p   = pool_mmap(len + 16384 + SIGSTKSZ);
  if (mprotect(p + len, 16384, PROT_NONE) != 0) {
    err_fail("stack guard failed");
  }
  stack_t ss = { .ss_sp = p + len + 16384, .ss_size = SIGSTKSZ };
  sigaltstack(&ss, NULL);
  struct sigaction sa = { .sa_handler = err_trap, .sa_flags = SA_ONSTACK };
  sigaction(SIGSEGV, &sa, NULL);
  sigaction(SIGBUS, &sa, NULL);
  return (Term*)p;
}

static void* pool_work(void* arg) {
  Term* stk  = pool_stack();
  u64   seen = 0;
  for (;;) {
    pthread_mutex_lock(&pool_lock);
    while (atomic_load_explicit(&pool_tick, memory_order_acquire) == seen) {
      pthread_cond_wait(&pool_wake, &pool_lock);
    }
    pthread_mutex_unlock(&pool_lock);
    seen = atomic_load_explicit(&pool_tick, memory_order_acquire);
    Env e = { CORPUS, ALC[1 + (u32)(uintptr_t)arg] };
    for (;;) {
      u32 r = atomic_fetch_add_explicit(&pool_row, 1, memory_order_relaxed);
      if (r >= (pool_grow ? CUBE_G : LANES / LINE)) {
        break;
      }
      if (pool_grow) {
        row_grow(e, stk, r * CUBE_T, 1, CUBE_T);
      } else {
        u32  step = CUBE_T / LINE;
        Ring row  = r / step * CUBE_T;
        for (Ring rg = row + r % step; rg < row + CUBE_T; rg += step) {
          u32 put0 = a32_load(ring_put(e.mem, rg));
          while (*ring_get(e.mem, rg) != put0 && !err_seen(e.mem)) {
            monk_step(e, stk, rg, put0, true, rg, 0, NULL);
          }
        }
      }
    }
    u32 done = atomic_fetch_add_explicit(&pool_done, 1, memory_order_release);
    if (done + 1 == pool_size) {
      pthread_mutex_lock(&pool_lock);
      pthread_cond_broadcast(&pool_wake);
      pthread_mutex_unlock(&pool_lock);
    }
  }
}

OUTLINE void pool_open(void) {
  static bool up;
  if (up) {
    return;
  }
  up = true;
  for (u32 w = 0; w < pool_size; w += 1) {
    pthread_t tid;
    if (pthread_create(&tid, NULL, pool_work, (void*)(uintptr_t)w)) {
      err_fail("pthread_create");
    }
  }
}

// The CPUs this process may use: affinity mask under the cgroup quota
static int cpu_read(const char* path, long* a, long* b) {
  FILE* f = fopen(path, "r");
  int   n = f == NULL ? 0 : fscanf(f, "%ld %ld", a, b);
  if (f != NULL) {
    fclose(f);
  }
  return n;
}

static long cpu_count(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) {
    n = CPU_COUNT(&set);
  }
  long q = 0;
  long p = 0;
  if (cpu_read("/sys/fs/cgroup/cpu.max", &q, &p) != 2) {
    cpu_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &q, &p);
    cpu_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us", &p, &p);
  }
  if (q > 0 && p > 0 && (q + p - 1) / p < n) {
    n = (q + p - 1) / p;
  }
#endif
  return n;
}

OUTLINE void pool_turn(bool grow) {
  pool_grow = grow;
  atomic_store_explicit(&pool_row, 0, memory_order_relaxed);
  atomic_store_explicit(&pool_done, 0, memory_order_relaxed);
  pthread_mutex_lock(&pool_lock);
  atomic_fetch_add_explicit(&pool_tick, 1, memory_order_release);
  pthread_cond_broadcast(&pool_wake);
  while (atomic_load_explicit(&pool_done, memory_order_acquire) < pool_size) {
    pthread_cond_wait(&pool_wake, &pool_lock);
  }
  pthread_mutex_unlock(&pool_lock);
}

// Gpu
// ===

// gpu_make compiles the device program into <binary>.gpu (--gpu-build):
// Metal's binary archive, or CUDA's cubin behind a hash of the text. A
// launch loads it, else notes and compiles.

static const char* gpu_path(void) {
  static char path[4096];
  u32 n = sizeof path - 8;
#ifdef __APPLE__
  _NSGetExecutablePath(path, &n);
#else
  path[readlink("/proc/self/exe", path, n)] = 0;
#endif
  return strcat(path, ".gpu");
}

static void gpu_note(const char* path) {
  fprintf(stderr, "bend: compiling the GPU program (%s is missing or"
    " stale)\n", path);
}

#if !BEND_CUDA
#define gpu_map pool_mmap
#endif

#if BEND_METAL || BEND_CUDA

static void gpu_kernel(u32 pass, u32 groups);

static void gpu_run(u32 f) {
  if (f < CUBE_T) {
    gpu_kernel(0, 1);
  }
  if (f < LANES) {
    gpu_kernel(0, CUBE_G);
  }
  gpu_kernel(1, CUBE_G);
  gpu_kernel(2, 1);
}

#endif

#if BEND_CUDA

static u64 gpu_hash(void) {
  u64 key = 14695981039346656037ull ^ CUBE_LOG;
  for (const char* p = BEND_SRC; *p != 0; p += 1) {
    key = (key ^ (u8)*p) * 1099511628211ull;
  }
  return key;
}

#endif

#if BEND_METAL

static void gpu_fail(NSError* err) {
  err_fail([[err localizedDescription] UTF8String]);
}

static bool gpu_probe(void) {
  return (gpu_dev = MTLCreateSystemDefaultDevice()) != nil;
}

static MTLComputePipelineDescriptor* gpu_desc(void) {
  NSError* err = nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.mathMode = MTLMathModeSafe;
  opts.preprocessorMacros = @{ @"CUBE_LOG": @(CUBE_LOG) };
  id<MTLLibrary> lib = [gpu_dev newLibraryWithSource:@(BEND_SRC) options:opts
    error:&err];
  if (!lib) {
    gpu_fail(err);
  }
  MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
  d.computeFunction = [lib newFunctionWithName:@"bend_dev"];
  return d;
}

static bool gpu_make(const char* path) {
  NSError* err = nil;
  id<MTLBinaryArchive> ar = [gpu_dev
    newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
  if (![ar addComputePipelineFunctionsWithDescriptor:gpu_desc() error:&err]) {
    gpu_fail(err);
  }
  return [ar serializeToURL:[NSURL fileURLWithPath:@(path)] error:&err];
}

static id<MTLComputePipelineState> gpu_pipe(MTLComputePipelineDescriptor* d,
  id<MTLBinaryArchive> ar) {
  NSError* err = nil;
  d.binaryArchives = ar ? @[ar] : @[];
  id<MTLComputePipelineState> pso = [gpu_dev
    newComputePipelineStateWithDescriptor:d
    options:ar ? MTLPipelineOptionFailOnBinaryArchiveMiss : 0 reflection:nil
    error:&err];
  if (!pso && !ar) {
    gpu_fail(err);
  }
  return pso;
}

static u64 gpu_span(void) {
  u64 span = [gpu_dev recommendedMaxWorkingSetSize];
  u64 most = [gpu_dev maxBufferLength];
  span = span < most ? span : most;
  return span < (2ull << 30) ? span : 2ull << 30;
}

static void gpu_load(u64 bytes) {
  gpu_buf = [gpu_dev newBufferWithBytesNoCopy:CORPUS length:bytes
    options:MTLResourceStorageModeShared
      | MTLResourceHazardTrackingModeUntracked deallocator:nil];
  if (!gpu_buf) {
    err_fail("the GPU span is more than the device has");
  }
  @autoreleasepool {
    gpu_que = [gpu_dev newCommandQueue];
    const char* path = gpu_path();
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:@(path)];
    MTLComputePipelineDescriptor* d = gpu_desc();
    id<MTLBinaryArchive> ar = [gpu_dev newBinaryArchiveWithDescriptor:ad
      error:nil];
    gpu_pso = ar ? gpu_pipe(d, ar) : nil;
    if (!gpu_pso) {
      gpu_note(path);
      gpu_pso = gpu_pipe(d, nil);
    }
  }
}

static void gpu_kernel(u32 pass, u32 groups) {
  [gpu_enc setComputePipelineState:gpu_pso];
  [gpu_enc setBuffer:gpu_buf offset:0 atIndex:0];
  [gpu_enc setBytes:&pass length:sizeof pass atIndex:1];
  [gpu_enc setThreadgroupMemoryLength:TG_HOLD * 8 atIndex:0];
  [gpu_enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(CUBE_T, 1, 1)];
  [gpu_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void gpu_pass(u32 f) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [gpu_que commandBuffer];
    gpu_enc = [cb computeCommandEncoder];
    gpu_run(f);
    [gpu_enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb error]) {
      gpu_fail([cb error]);
    }
  }
}

#elif BEND_CUDA

// The bag from the device: a group of 128 lanes per 64 KB of L2, a power of
// two in 16..128 (Apple keeps the tuned 128).
static void gpu_shape(int units) {
  CUBE_LOG = 31 - CLZ(units < 16 ? 16 : units > 128 ? 128 : units);
}

static bool gpu_probe(void) {
  int       managed = 0;
  CUcontext ctx;
  // one stream: the default 8 cost about half of the startup
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
  if (cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&gpu_dev, 0) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&managed,
      CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, gpu_dev);
  }
  int l2 = 1 << 23;
  cuDeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, gpu_dev);
  gpu_shape(l2 >> 16);
  return managed != 0
    && cuDevicePrimaryCtxRetain(&ctx, gpu_dev) == CUDA_SUCCESS
    && cuCtxSetCurrent(ctx) == CUDA_SUCCESS;
}

static Corpus gpu_map(u64 bytes) {
  CUdeviceptr p = 0;
  if (cuMemAllocManaged(&p, bytes, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
    err_fail("corpus reservation failed");
  }
#if CUDA_VERSION >= 13000
  cuMemAdvise(p, bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
    (CUmemLocation){ CU_MEM_LOCATION_TYPE_DEVICE, gpu_dev });
#else
  cuMemAdvise(p, bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, gpu_dev);
#endif
  return (Corpus)(uintptr_t)p;
}

static bool gpu_make(const char* path) {
  int cc[2] = {0, 0};
  cuDeviceGetAttribute(cc,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gpu_dev);
  cuDeviceGetAttribute(cc + 1,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gpu_dev);
  char arch[40];
  char bag[24];
  snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", cc[0], cc[1]);
  snprintf(bag, sizeof bag, "-DCUBE_LOG=%u", CUBE_LOG);
  const char* opts[] = { arch, bag, "--fmad=false", "-default-device" };
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, BEND_SRC, "bend.cu", 0, NULL, NULL)
    != NVRTC_SUCCESS) {
    err_fail("cannot compile the CUDA library");
  }
  if (nvrtcCompileProgram(prog, 4, opts) != NVRTC_SUCCESS) {
    size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    char* log = calloc(n + 1, 1);
    if (log != NULL && nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
      fprintf(stderr, "%s\n", log);
    }
    err_fail("cannot compile the CUDA library");
  }
  size_t len = 0;
  nvrtcGetCUBINSize(prog, &len);
  char* bin = malloc(len);
  if (bin == NULL || nvrtcGetCUBIN(prog, bin) != NVRTC_SUCCESS) {
    err_fail("cannot load the CUDA library");
  }
  nvrtcDestroyProgram(&prog);
  u64   key = gpu_hash();
  FILE* out = path == NULL ? NULL : fopen(path, "wb");
  bool  ok  = out != NULL && fwrite(&key, 8, 1, out) == 1
    && fwrite(bin, 1, len, out) == len && fclose(out) == 0;
  if (cuModuleLoadData(&gpu_lib, bin) != CUDA_SUCCESS) {
    err_fail("cannot load the CUDA library");
  }
  free(bin);
  return path == NULL || ok;
}

static u64 gpu_span(void) {
  size_t span = 0;
  cuDeviceTotalMem(&span, gpu_dev);
  return span;
}

static void gpu_load(u64 bytes) {
  const char* path = gpu_path();
  int         fd   = open(path, O_RDONLY);
  struct stat st   = { 0 };
  u64         key  = 0;
  char*       bin  = fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 8 ? NULL
    : mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (bin != NULL && bin != MAP_FAILED) {
    memcpy(&key, bin, 8);
  }
  if (key != gpu_hash()
    || cuModuleLoadData(&gpu_lib, bin + 8) != CUDA_SUCCESS) {
    gpu_note(path);
    gpu_make(path);
  }
  if (cuModuleGetFunction(&gpu_pso, gpu_lib, "bend_dev") != CUDA_SUCCESS) {
    err_fail("cannot load the GPU program");
  }
}

static void gpu_kernel(u32 pass, u32 groups) {
  void* args[] = { &CORPUS, &pass };
  if (cuLaunchKernel(gpu_pso, groups, 1, 1, CUBE_T, 1, 1, TG_HOLD * 8, NULL,
    args, NULL) != CUDA_SUCCESS) {
    err_fail("device launch failed");
  }
}

static void gpu_pass(u32 f) {
  gpu_run(f);
  if (cuCtxSynchronize() != CUDA_SUCCESS) {
    err_fail("device fault");
  }
}

#else

#define gpu_probe() false
#define gpu_make(p) true
#define gpu_span()  0
#define gpu_load(b)
#define gpu_pass(f)

#endif

// Cube
// ====

static void cube_run(Corpus H, bool gpu) {
  for (;;) {
    u32 f = a32_load(a32_at(H, H_CURSOR));
    a32_store(a32_at(H, H_CURSOR), 0);
    if (root_done(H)) {
      return;
    }
    if (f == 0) {
      err_fail("frontier drained without a result");
    }
    if (gpu) {
      gpu_pass(f);
    } else {
      // Under a unit (CUBE_T / LINE a row) per thread, the column grows to
      // the rows that give one, no more: each touches a page of every plane.
      if (f * (CUBE_T / LINE) < pool_size) {
        row_grow((Env){ H, ALC[0] }, io_stk, 0, CUBE_G,
          (pool_size + CUBE_T / LINE - 1) / (CUBE_T / LINE));
      }
      if (f < CUBE) {
        pool_turn(true);
      }
      pool_turn(false);
    }
    u32 ec = a32_load(a32_at(H, H_ERROR_CODE));
    if (ec != 0) {
      err_post(H, ec);
    }
  }
}

// Corpus
// ======

// The cores map 8 GiB at a high base and double it in place, so one base
// holds every Loc; the banks move up past the pages. The GPU maps its span
// once.

static u64 corpus_size;

static void* corpus_map(u64 size) {
  u64   hint = 1ull << 45;
  void* p    = pool_try((void*)hint, size);
  while (p != (void*)hint && hint > size) {
    if (p != MAP_FAILED) {
      munmap(p, size);
    }
    hint /= 2;
    p     = pool_try((void*)hint, size);
  }
  if (p == MAP_FAILED) {
    err_fail("reservation failed");
  }
  return p;
}

static void corpus_lay(Corpus H, u64 size) {
  u64 span = size / 8;
  u64 cap  = span > HEAP_OFF ? (span - HEAP_OFF) / (PAGE_LEN + 10) : 0;
  if (cap <= CUBE) {
    err_fail("the GPU span is under the rings, stacks and a page per lane");
  }
  cap = cap < ~0u ? cap : ~0u - 1;
  u64 at = HEAP_OFF + (cap << PAGE_BITS);
  for (u32 c = 0; c < NCLS_ALL; c += 1) {
    Bank* b = bank_at(H, c);
    memcpy(H + at, H + b->off, b->wr * sizeof(u64));
    b->off  = at;
    at     += 2 * (cap >> ((c < NCLS ? NCLS : c) - PAGE_BITS));
  }
  corpus_size = size;
  a32_store_rel(a32_at(H, H_CAP), (u32)cap);
}

static bool corpus_grow(Corpus H, u64 need) {
  bool ok = true;
  LOCK(bank_lock);
  while (ok && need > a32_load(a32_at(H, H_CAP))) {
    u64   more = corpus_size;
    char* at   = (char*)H + more;
    void* got  = io_gpu || more >= 1ull << 43 ? MAP_FAILED
      : pool_try(at, more);
    ok = got == at;
    if (ok) {
      corpus_lay(H, more * 2);
    } else if (got != MAP_FAILED) {
      munmap(got, more);
    }
  }
  UNLOCK(bank_lock);
  return ok;
}

static Corpus corpus_setup(bool gpu, long threads, u64 bytes) {
  io_gpu     = gpu;
  KEEP_WORDS = gpu ? CHUNK : CAP_WORDS;
  u64 dflt   = gpu ? gpu_span() : 1ull << 33;
  u64 size   = (gpu && bytes != 0 ? bytes : dflt) & ~16383ull;
  CORPUS     = gpu ? gpu_map(size) : corpus_map(size);
  Corpus H   = CORPUS;
#if BEND_CUDA
  if (gpu) {
    cuMemsetD8((CUdeviceptr)(uintptr_t)H, 0, STAK_OFF * 8);
    cuCtxSynchronize();
  }
#endif
  corpus_lay(H, size);
  memcpy(H + STAT_OFF, STAT_IMG, STAT_LEN * sizeof(u64));
  a32_store(a32_at(H, H_BUMP), 1);
  if (gpu) {
    gpu_load(size);
  }
  pool_size = threads < 1 ? 1 : threads < CUBE_T ? threads : CUBE_T;
  return H;
}

OUTLINE Term corpus_eval(Corpus H, Term t) {
  Env  e = { H, ALC[0] };
  Term rv[WL_RESW];
  for (;;) {
    Reply r = work_loop(e, io_stk, t, !BANGS && pool_size == 1);
    if (r == 0) {
      if (root_done(H)) {
        break;
      }
      err_fail("solo delivery lost");
    }
    if ((u32)H[task_tail(r) + 1] == 0) {
      t = r;
      if (io_gpu && fid_bangs((u32)term_aux(t))) {
        Loc  tl   = task_tail(t);
        Term cont = H[tl];
        u32  idx  = (u32)(H[tl + 1] >> 32) & 0xFFFF;
        H[tl]     = TERM_HOLE;
        a32_store(a32_at(H, H_CURSOR), 1);
        ring_push(H, 0, t);
        cube_run(H, true);
        Term p = task_deliver(H, cont, idx, rv, root_take(H, rv));
        if (root_done(H)) {
          break;
        }
        if (p == 0) {
          err_fail("seam delivery lost");
        }
        t = p;
      }
      continue;
    }
    task_deal(H, r, 0, 0, (Cur)0);
    pool_open();
    cube_run(H, false);
    break;
  }
  root_take(H, rv);
  return rv[0];
}

// Io
// ==

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define IO_READ 1
#define IO_TIME 2
#define IO_PARK TERM_HOLE

// Base's opaque, linear handles pack host fds/pointers into aux/loc: no
// forging, copying, reuse or host wrapper.
#define io_hand(v)   term_make(TAG_PAK, (u64)(v) >> 40, (u64)(v) & LOC_MASK)
#define io_hand_v(t) (((u64)term_aux(t) << 40) | term_loc(t))

struct IoWork;
typedef void (*IoCall)(struct IoWork* w);
typedef Term (*IoPack)(Env e, struct IoWork* w);

typedef struct IoWork {
  intptr_t hand;
  intptr_t made;
  u32      word;
  u64      size;
  char*    data;
  char*    text;
  u32      code;
  IoCall   call;
  IoPack   pack;
} IoWork;

typedef Term (*Effect)(Env e, Term* f, IoWork* w);

typedef struct {
  Effect run;
  u32    ask;
} IoEff;

static IoEff io_eff_rows[1 << 16];
static u32   io_live;

static u64 io_tick(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

OUTLINE void* io_mem(void* mem) {
  if (mem == NULL) {
    err_fail("host allocation failed");
  }
  return mem;
}

static int io_sys_addr(const char* host, u32 port, struct sockaddr_in* at) {
  memset(at, 0, sizeof(*at));
  at->sin_family = AF_INET;
  at->sin_port   = htons((uint16_t)port);
  // a leading zero is octal to inet_aton: refused
  for (const char* p = host; *p != 0; p += 1) {
    if ((p == host || p[-1] == '.') && *p == '0' && p[1] >= '0' && p[1] <= '9') {
      return -1;
    }
  }
  return port > 65535 || inet_pton(AF_INET, host, &at->sin_addr) != 1
    ? -1 : 0;
}

// An address of either family: a dotted IPv4 one, read as io_sys_addr
// reads it, or an IPv6 one in its text form ("::1", "::ffff:1.2.3.4";
// no brackets, no zone). *len is the length the address takes; the
// answer is its family, or -1.
static int io_sys_sa(const char* host, u32 port, struct sockaddr_storage* at, socklen_t* len) {
  memset(at, 0, sizeof(*at));
  if (io_sys_addr(host, port, (struct sockaddr_in*)at) == 0) {
    *len = sizeof(struct sockaddr_in);
    return AF_INET;
  }
  struct sockaddr_in6* a6 = (struct sockaddr_in6*)at;
  memset(at, 0, sizeof(*at));
  a6->sin6_family = AF_INET6;
  a6->sin6_port   = htons((uint16_t)port);
  *len            = sizeof(struct sockaddr_in6);
  return port > 65535 || inet_pton(AF_INET6, host, &a6->sin6_addr) != 1 ? -1 : AF_INET6;
}

// An IPv6 listener takes IPv4 connections too (IPV6_V6ONLY off) where
// the system lets it, so "::" is every interface of both families, as
// 0.0.0.0 is every one of IPv4's; where it does not, "::" is IPv6's.
static void io_sys_dual(int fd, int fam) {
  if (fam == AF_INET6) {
    int zero = 0;
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
  }
}

// The program's arguments (IO.args).
static int    io_argc;
static char** io_argv;

static void io_eff(u32 cid, Effect run, u32 need) {
  io_eff_rows[cid] = (IoEff){ run, need };
}

static u64 io_sys_end(IoWork* w, ssize_t n) {
  w->code = n < 0 ? (u32)errno : 0;
  return n < 0 ? 0 : (u64)n;
}

// cont(item) is the next request (run by io_exec). Parked, word/time/evts
// hold fd/deadline/readiness; pack resumes. Leading work permits IoWork*
// to IoAct* casts. race: IO.within's, whose answer io_race takes.
// IoAct ::=
//   | IoAct(work, cont, item, time, evts, next, race)
typedef struct IoAct {
  IoWork        work;
  Term          cont;
  Term          item;
  u64           time;
  short         evts;
  u32           heap;
  struct IoAct* next;
  void*         race;
} IoAct;

typedef struct {
  IoAct* head;
  IoAct* last;
} IoQue;

static IoQue io_runs;
static IoQue io_park;
static IoQue io_jobs;

// Where a parked activation waits. On Linux a descriptor waiter is
// registered with epoll once and a deadline waiter sits in a binary
// min-heap, so a pass costs what is ready rather than what is waiting.
// An activation can be in both at once -- that is what TCP.poll is --
// so it carries its heap slot (heap - 1, 0 when it is in no heap) and
// whichever fires first takes it out of the other. Everywhere else
// io_park holds them all and poll reads the whole set every pass, as
// it always has.
#ifdef __linux__
static int     io_ep = -1;
static u32     io_fds;
static IoAct** io_time;
static u32     io_time_n;
static u32     io_time_cap;

// A descriptor's place in the poller, and the activations parked on it
// each way (rd, wr). reg is whether it is believed to be in the set.
//
// A socket the runtime made itself (own: accepted or connected, and
// closed through Socket.close) is registered once, for both ways and
// edge-triggered, and stays until it is closed: a park costs no syscall
// and a wake none either. An edge says the socket became ready since the
// poller last reported it, so what it needs besides is what was learnt
// in between: rdy holds, a bit a way (1 read, 2 write), whether it may
// be ready -- set by an edge, cleared by a call that found it not
// (EAGAIN) and by a plain read that came back short, which on Linux
// means the socket was drained. A read that finds its bit clear parks
// without trying, as nginx does.
//
// Any other descriptor (a listener, a foreign effect's, one whose close
// the runtime may not see) is registered with EPOLLONESHOT and left
// there: the kernel disarms it as it fires, so a wake costs no syscall
// and the next park is one MOD rather than an ADD and a DEL. That belief
// can be wrong in one way -- a descriptor closed and its number handed
// out again -- so every call takes the other operation when the first
// is refused, which is what makes reg a hint rather than bookkeeping.
typedef struct {
  IoAct* rd;
  IoAct* wr;
  u8     reg;
  u8     rdy;
  u8     own;
} IoFd;

static IoFd* io_fdt;
static u32   io_fdt_cap;

static IoFd* io_fd_at(int fd) {
  if ((u32)fd >= io_fdt_cap) {
    u32 was = io_fdt_cap;
    io_fdt_cap = io_fdt_cap != 0 ? io_fdt_cap * 2 : 1024;
    while ((u32)fd >= io_fdt_cap) {
      io_fdt_cap *= 2;
    }
    io_fdt = io_mem(realloc(io_fdt, io_fdt_cap * sizeof *io_fdt));
    for (u32 i = was; i < io_fdt_cap; i += 1) {
      io_fdt[i] = (IoFd){ NULL, NULL, 0, 3, 0 };
    }
  }
  return &io_fdt[fd];
}

#endif

// Whether a read would find nothing: the descriptor is registered and
// nothing has arrived since it was found empty. Only a plain socket's
// reads keep this; a TLS session buffers what it decrypted.
static bool io_fd_quiet(int fd, u8 way) {
#ifdef __linux__
  return fd >= 0 && (u32)fd < io_fdt_cap && io_fdt[fd].own && io_fdt[fd].reg
    && !(io_fdt[fd].rdy & way);
#else
  (void)fd; (void)way;
  return false;
#endif
}

// what a call learnt: that a way may be ready, or that it is not
static void io_fd_seen(int fd, u8 way, bool ready) {
#ifdef __linux__
  if (fd >= 0) {
    IoFd* d = io_fd_at(fd);
    d->rdy = ready ? d->rdy | way : d->rdy & (u8)~way;
  }
#else
  (void)fd; (void)way; (void)ready;
#endif
}

// A socket an effect has just made (own), or one it is about to close.
static void io_fd_made(int fd, bool own) {
#ifdef __linux__
  if (fd >= 0 && (own || (u32)fd < io_fdt_cap)) {
    *io_fd_at(fd) = (IoFd){ NULL, NULL, 0, 3, own };
  }
#else
  (void)fd; (void)own;
#endif
}

#define io_fd_fresh(fd) io_fd_made((fd), true)
#define io_fd_gone(fd)  io_fd_made((fd), false)

#ifdef __linux__
static void io_time_put(IoAct* a, u32 i) {
  io_time[i] = a;
  a->heap    = i + 1;
}

static void io_time_up(u32 i) {
  while (i != 0) {
    u32 p = (i - 1) / 2;
    if (io_time[p]->time <= io_time[i]->time) {
      break;
    }
    IoAct* t = io_time[p];
    io_time_put(io_time[i], p);
    io_time_put(t, i);
    i = p;
  }
}

static void io_time_down(u32 i) {
  for (;;) {
    u32 l = 2 * i + 1;
    u32 r = l + 1;
    u32 m = i;
    if (l < io_time_n && io_time[l]->time < io_time[m]->time) {
      m = l;
    }
    if (r < io_time_n && io_time[r]->time < io_time[m]->time) {
      m = r;
    }
    if (m == i) {
      break;
    }
    IoAct* t = io_time[m];
    io_time_put(io_time[i], m);
    io_time_put(t, i);
    i = m;
  }
}

static void io_time_push(IoAct* a) {
  if (io_time_n == io_time_cap) {
    io_time_cap = io_time_cap != 0 ? io_time_cap * 2 : 64;
    io_time = io_mem(realloc(io_time, io_time_cap * sizeof *io_time));
  }
  io_time_put(a, io_time_n);
  io_time_n += 1;
  io_time_up(io_time_n - 1);
}

// Takes it out of the heap wherever it sits, which is how the other
// half of a descriptor-and-deadline park is cancelled.
static void io_time_drop(IoAct* a) {
  if (a->heap == 0) {
    return;
  }
  u32 i = a->heap - 1;
  a->heap = 0;
  io_time_n -= 1;
  if (i != io_time_n) {
    io_time_put(io_time[io_time_n], i);
    io_time_up(i);
    io_time_down(i);
  }
}
#endif

static void io_push(IoQue* q, IoAct* a) {
  a->next = NULL;
  *(q->head == NULL ? &q->head : &q->last->next) = a;
  q->last = a;
}

static IoAct* io_pop(IoQue* q) {
  IoAct* a = q->head;
  q->head  = a->next;
  return a;
}

static void io_spawn(Term m) {
  IoAct* a = io_mem(calloc(1, sizeof(IoAct)));
  a->cont  = m;
  a->item  = term_clo(FID_IO_EMIT, 0);
  io_push(&io_runs, a);
  io_live += 1;
}

// Park until evts (POLLIN/POLLOUT; 0 ignores fd) or time (0: none); the
// loop then calls more: a value resumes, IO_PARK re-parks.
static Term io_wait_on(IoWork* w, int fd, short evts, u64 time, IoPack more) {
  IoAct* a     = (IoAct*)w;
  a->work.word = (u32)fd;
  a->work.pack = more;
  a->time      = time;
  a->evts      = evts;
#ifdef __linux__
  if (evts != 0) {
    IoFd* d   = io_fd_at(fd);
    u8    way = evts == POLLOUT ? 2 : 1;
    // the caller has just found it not ready that way
    d->rdy &= (u8)~way;
    if (!d->own || !d->reg) {
      struct epoll_event ev;
      ev.events   = d->own ? EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET
        : (way == 2 ? EPOLLOUT : EPOLLIN) | EPOLLONESHOT;
      ev.data.u64 = (u64)fd + 1;
      bool known  = d->reg;
      if (epoll_ctl(io_ep, known ? EPOLL_CTL_MOD : EPOLL_CTL_ADD, fd, &ev) != 0
          && epoll_ctl(io_ep, known ? EPOLL_CTL_ADD : EPOLL_CTL_MOD, fd, &ev) != 0) {
        err_fail("the poller refused a descriptor");
      }
      d->reg = 1;
    }
    *(way == 2 ? &d->wr : &d->rd) = a;
    io_fds += 1;
  }
  if (time != 0) {
    io_time_push(a);
  }
  if (evts == 0 && time == 0) {
    io_push(&io_park, a);
  }
#else
  io_push(&io_park, a);
#endif
  return IO_PARK;
}

// Parked deadline (0: none).
static u64 io_wait_time(IoWork* w) {
  return ((IoAct*)w)->time;
}

// Wakes a parked activation now, as its deadline would: the next pass
// fires it (a descriptor wait taken out as a deadline's is) and its
// continuation looks again. A channel's bell rings this way (chan.c).
static void io_ring(IoAct* a) {
  a->time = 1;
#ifdef __linux__
  if (a->heap != 0) {
    io_time_up(a->heap - 1);
  } else {
    io_time_push(a);
  }
#endif
}

OUTLINE void io_out(FILE* h, const char* data, u64 len) {
  if (fwrite(data, 1, len, h) != len) {
    err_fail("a short write on a standard stream");
  }
}

OUTLINE void io_sync(void) {
  if (fflush(stdout) != 0) {
    err_fail("a short write on a standard stream");
  }
}

// the edge is UTF-8
static u64 io_utf8(char* buf, u64 c) {
  u64 k = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  for (u64 i = k; i > 1; i -= 1) {
    buf[i - 1] = (char)(0x80 | (c & 0x3F));
    c >>= 6;
  }
  buf[0] = (char)(k == 1 ? c : (0xF00 >> k) | c);
  return k;
}

OUTLINE char* io_cstr(Env e, Term s, u64* len) {
  StrParts p = str_peek(e, s);
  u64 n = 0;
  char* buf = io_mem(malloc((u64)p.len * 4 + 1));
  for (u32 i = 0; i < p.len; i++) {
    u32 c = str_at_peek(e, p, i);
    if (c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)) {
      free(buf); term_sink(e, s);
      err_fail("cannot encode a non-scalar Char as UTF-8");
    }
    n += io_utf8(buf + n, c);
  }
  term_sink(e, s);
  buf[n] = 0;
  *len = n;
  return buf;
}

OUTLINE void io_errs(Env e, Term s) {
  u64   n    = 0;
  char* text = io_cstr(e, s, &n);
  io_sync();
  io_out(stderr, text, n);
  io_out(stderr, "\n", 1);
  free(text);
}

#define io_nul(s, n) (strlen(s) != (n))

#define io_seal(e, t, cid) (cid_hot(cid) ? rfc_seal(e, t) : (t))

static Term io_node(Env e, u64 cid, Term a, Term b) {
  Loc l = heap_alloc(e, 1);
  e.mem[l]     = io_seal(e, a, cid);
  e.mem[l + 1] = io_seal(e, b, cid);
  return term_ctr(cid, l);
}

// 1-byte cells until a wider scalar arrives; then the decoded cells move to
// that width (at most twice).
static Term io_str(Env e, const char* p, u64 n) {
  StrParts out = str_alloc(e, n, 2);
  if (err_seen(e.mem)) { return str_view_owned(e, out); }
  u32 len = 0;
  for (u64 i = 0; i < n;) {
    u32 b = (u8)p[i], c = b, k = b < 0x80 ? 1
      : b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3
      : b >= 0xf0 && b <= 0xf4 ? 4 : 0;
    bool ok = k && k <= n - i;
    if (k > 1) {
      c = b & (0x7f >> k);
      for (u32 j = 1; ok && j < k; j++) {
        u32 t = (u8)p[i + j]; ok = (t & 0xc0) == 0x80;
        c = (c << 6) | (t & 63);
      }
      ok = ok && c >= (k == 2 ? 0x80u : k == 3 ? 0x800u : 0x10000u)
        && c <= 0x10ffff && !(c >= 0xd800 && c <= 0xdfff);
    }
    c = ok ? c : 0xfffd;
    i += ok ? k : 1;
    if (str_fit(c) < str_nar(out)) {
      StrParts q = str_alloc(e, len + 1 + (n - i), str_fit(c));
      out.len = len;
      if (q.data) { str_copy_cells(e, q, 0, out); }
      term_sink(e, out.data);
      out = q;
      if (err_seen(e.mem)) { break; }
    }
    str_put(e, out, len++, c);
  }
  out.len = len;
  return str_view_owned(e, out);
}

// Bytes as they are: a byte a 1-byte cell, one copy, no decoder.
static inline Term io_buf(Env e, const char* p, u64 n) {
  StrParts out = str_alloc(e, n, 2);
  if (out.data && !err_seen(e.mem)) {
    memcpy((u8*)(e.mem + term_peek(e, out.data)), p, n);
  }
  return str_view_owned(e, out);
}

// A Bytes' cells as a byte run: the block itself when its cells are one
// byte (*own = NULL), else a copy (*own to free); a cell past 255 is NULL.
static inline const char* io_buf_ptr(Env e, Term s, u32* n, char** own) {
  StrParts p = str_peek(e, s);
  *n = p.len;
  *own = NULL;
  if (!p.len || str_nar(p) == 2) {
    return p.len ? (const char*)(e.mem + term_peek(e, p.data)) + p.off : "";
  }
  *own = io_mem(malloc(p.len));
  for (u32 i = 0; i < p.len; i++) {
    u32 c = str_at_peek(e, p, i);
    if (c > 255) { free(*own); *own = NULL; return NULL; }
    (*own)[i] = (char)c;
  }
  return *own;
}

#define io_tup(e, a, b) io_node(e, CID_TUPLE, a, b)
#define io_done(e, v)   io_box(e, CID_DONE, v)

static Term io_box(Env e, u64 cid, Term v) {
  Loc l = heap_alloc(e, 0);
  e.mem[l] = io_seal(e, v, cid);
  return term_ctr(cid, l);
}

static Term io_fail(Env e, u32 code, const char* text) {
  const char* s = text != NULL ? text : strerror((int)code);
  Term t = io_tup(e, code, io_str(e, s, strlen(s)));
  return io_box(e, CID_FAIL, t);
}

static lock           io_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t io_bell = PTHREAD_COND_INITIALIZER;
static u32            io_busy;
static u32            io_size;
static int            io_wake_fd[2];

static void io_take(Env e) {
  IoAct*  acts[64];
  ssize_t n;
  while ((n = read(io_wake_fd[0], acts, sizeof acts)) > 0) {
    for (u32 i = 0; i < (u32)n / sizeof(IoAct*); i += 1) {
      IoAct* a = acts[i];
      a->item  = a->work.pack(e, &a->work);
      io_push(&io_runs, a);
      io_busy -= 1;
    }
  }
}

static void* io_help(void* arg) {
  for (;;) {
    pthread_mutex_lock(&io_gate);
    while (io_jobs.head == NULL) {
      pthread_cond_wait(&io_bell, &io_gate);
    }
    IoAct* a = io_pop(&io_jobs);
    pthread_mutex_unlock(&io_gate);
    a->work.call(&a->work);
    while (write(io_wake_fd[1], &a, sizeof a) != sizeof a) {
    }
  }
}

// Run call on a helper thread, then pack on the loop to resume the effect.
static Term io_work(IoWork* w, IoCall call, IoPack pack) {
  w->call  = call;
  w->pack  = pack;
  io_busy += 1;
  if (io_busy > io_size && io_size < IO_HELP) {
    pthread_t tid;
    if (pthread_create(&tid, NULL, io_help, NULL)) {
      err_fail("pthread_create");
    }
    pthread_detach(tid);
    io_size += 1;
  }
  pthread_mutex_lock(&io_gate);
  io_push(&io_jobs, (IoAct*)w);
  pthread_cond_signal(&io_bell);
  pthread_mutex_unlock(&io_gate);
  return IO_PARK;
}

// Run call on the loop itself and resume at once: for work that does not
// block on a warm system (an open, an fstat, a read the page cache holds),
// where a helper's round trip -- two wakes and the eventfd -- costs more
// than the call.
static Term io_now(Env e, IoWork* w, IoCall call, IoPack pack) {
  call(w);
  return pack(e, w);
}

// Consume cont's request node; the effect returns a value or IO_PARK.
static Term io_exec(Env e, IoWork* w) {
  IoAct* a = (IoAct*)w;
  Term   fs[256];
  u32    c = (u32)term_aux(a->cont);
  u32    n = cid_arity(c);
  spare_free(e, cls_fit(n), ctr_take(e, a->cont, n, fs));
  a->cont = fs[n - 1];
  return io_eff_rows[c].run(e, fs, w);
}

// Whether nothing waits on a descriptor or a deadline. On Linux those
// live in the epoll set and the heap rather than on io_park, so the
// deadlock test has to ask.
static bool io_idle(void) {
#ifdef __linux__
  return io_fds == 0 && io_time_n == 0;
#else
  return true;
#endif
}

#ifdef __linux__

// Takes the activation out of both sets and runs its continuation. A
// re-park puts it back through io_wait_on, so nothing is left behind
// when the effect changes what it waits for.
// ready says the descriptor is what woke this activation, and the poller
// has already taken it from its slot. A deadline that fires first leaves
// it in the slot, so it is taken out there; the registration stays.
static void io_fire(Env e, IoAct* a, bool ready) {
  io_time_drop(a);
  if (a->evts != 0) {
    if (!ready) {
      IoFd* d = io_fd_at((int)a->work.word);
      if (d->rd == a) { d->rd = NULL; }
      if (d->wr == a) { d->wr = NULL; }
      // armed and pointing at an activation that has moved on
      if (!d->own) {
        epoll_ctl(io_ep, EPOLL_CTL_DEL, (int)a->work.word, NULL);
        d->reg = 0;
      }
    }
    io_fds -= 1;
    a->evts = 0;
  }
  Term x = a->work.pack(e, &a->work);
  if (x != IO_PARK) {
    a->item = x;
    io_push(&io_runs, a);
  }
}

// One pass: wait for a descriptor to be ready or for the soonest
// deadline, then fire what fired. Only what fired is touched.
static void io_wait(Env e) {
  struct epoll_event es[64];
  int ms = -1;
  if (io_time_n != 0) {
    u64 soon = io_time[0]->time;
    u64 tick = io_tick();
    u64 gap  = soon > tick ? (soon - tick) / 1000000 + 1 : 0;
    ms = gap > 0x7fffffff ? 0x7fffffff : (int)gap;
  }
  int m;
  io_sync();
  while ((m = epoll_wait(io_ep, es, 64, ms)) < 0) {
    if (errno != EINTR) {
      err_fail("the poller failed");
    }
  }
  for (int i = 0; i < m; i += 1) {
    u64 k = es[i].data.u64;
    if (k == 0) {
      io_take(e);
      continue;
    }
    IoFd* d = io_fd_at((int)(k - 1));
    u32   v = es[i].events;
    if (v & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
      d->rdy |= 1;
      if (d->rd != NULL) {
        IoAct* a = d->rd;
        d->rd = NULL;
        io_fire(e, a, true);
      }
    }
    if (v & (EPOLLOUT | EPOLLHUP | EPOLLERR)) {
      d->rdy |= 2;
      if (d->wr != NULL) {
        IoAct* a = d->wr;
        d->wr = NULL;
        io_fire(e, a, true);
      }
    }
  }
  u64 now = io_tick();
  while (io_time_n != 0 && io_time[0]->time <= now) {
    io_fire(e, io_time[0], false);
  }
}

#else

// macOS poll misses FIFO EOF. Size select sets to the highest fd;
// _DARWIN_UNLIMITED_SELECT allows fds past FD_SETSIZE.
static bool io_bit(u8* set, int fd, bool put) {
  u8* at = set + fd / 8;
  *at |= put << fd % 8;
  return *at >> fd % 8 & 1;
}

static void io_wait(Env e) {
  int top  = io_wake_fd[0];
  u64 soon = 0;
  for (IoAct* a = io_park.head; a != NULL; a = a->next) {
    if (a->time != 0 && (soon == 0 || a->time < soon)) {
      soon = a->time;
    }
    if (a->evts != 0 && (int)a->work.word > top) {
      top = (int)a->work.word;
    }
  }
  u64 len = (u64)top / 64 * 8 + 8;
  u8* set[2] = { io_mem(calloc(2, len)), NULL };
  set[1] = set[0] + len;
  io_bit(set[0], io_wake_fd[0], true);
  for (IoAct* a = io_park.head; a != NULL; a = a->next) {
    if (a->evts != 0) {
      io_bit(set[a->evts == POLLOUT], (int)a->work.word, true);
    }
  }
  u64 tick = io_tick();
  u64 ms = soon > tick ? (soon - tick) / 1000000 + 1 : 0;
  struct timeval tv = { ms / 1000, ms % 1000 * 1000 };
  io_sync();
  while (select(top + 1, (fd_set*)set[0], (fd_set*)set[1], NULL,
    soon == 0 ? NULL : &tv) < 0) {
    if (errno != EINTR) {
      err_fail("the poller failed");
    }
  }
  if (io_bit(set[0], io_wake_fd[0], false)) {
    io_take(e);
  }
  u64   now  = io_tick();
  IoQue todo = io_park;
  io_park = (IoQue){0};
  while (todo.head != NULL) {
    IoAct* a   = io_pop(&todo);
    bool   due = (a->evts != 0
        && io_bit(set[a->evts == POLLOUT], (int)a->work.word, false))
      || (a->time != 0 && a->time <= now);
    if (!due) {
      io_push(&io_park, a);
      continue;
    }
    Term x = a->work.pack(e, &a->work);
    if (x != IO_PARK) {
      a->item = x;
      io_push(&io_runs, a);
    }
  }
  free(set[0]);
}

#endif

// The wire under a socket
// =======================

// A plain socket reads and writes itself. Something that sits between
// the socket and the bytes -- a TLS session is the only one so far --
// installs this, and the effects that carry bytes go through it without
// knowing what it is. NULL is the plain path, which costs one branch
// the processor predicts.
//
// read and write answer as recv and send do, and set dir when they
// could not finish: POLLIN or POLLOUT says which way the socket has to
// become ready before trying again, since a TLS read may be waiting to
// write and the other way about. join is told about a socket accepted
// from a listener, and shut about one that is closing.
typedef struct IoWire {
  ssize_t (*read)(int fd, void* buf, size_t len, short* dir);
  ssize_t (*write)(int fd, const void* buf, size_t len, short* dir);
  void    (*join)(int lfd, int fd);
  void    (*shut)(int fd);
} IoWire;

static IoWire* io_wire;

static ssize_t io_wire_read(int fd, void* buf, size_t len, short* dir) {
  *dir = POLLIN;
  return io_wire != NULL ? io_wire->read(fd, buf, len, dir)
    : recv(fd, buf, len, 0);
}

static ssize_t io_wire_write(int fd, const void* buf, size_t len, short* dir) {
  *dir = POLLOUT;
  return io_wire != NULL ? io_wire->write(fd, buf, len, dir)
    : send(fd, buf, len, 0);
}

static void io_wire_join(int lfd, int fd) {
  if (io_wire != NULL) {
    io_wire->join(lfd, fd);
  }
}

static void io_wire_shut(int fd) {
  if (io_wire != NULL) {
    io_wire->shut(fd);
  }
}

${NATIVE.IO}
// Show
// ====

#if MAIN_PURE

// A pure main's value as term_show spells it: d a SHOW_DESC node (see
// show_main), w its words.
static void show_val(Env e, u32 d, const Term* w, char chain);

// char_show: an escape, a \u{hex}, else the code point in UTF-8
static void show_chr(u64 c, char q) {
  char b[4];
  int  k = c == 10 ? 'n' : c == 9 ? 't' : c == 13 ? 'r' : c == 0 ? '0'
    : c == 92 || c == (u64)q ? (int)c : 0;
  if (k != 0) {
    printf("\\%c", k);
  } else if (c < 32 || c == 127 || (c >= 0xD800 && c <= 0xDFFF)
    || c > 0x10FFFF) {
    printf("\\u{%llx}", (unsigned long long)c);
  } else {
    fwrite(b, 1, io_utf8(b, c), stdout);
  }
}

// The shortest text that reads back, as a literal: a point before an e
static void show_f32(u32 x) {
  char  buf[40];
  int   n  = f32_text(buf, f32_unbox(x));
  char* ep = memchr(buf, 'e', n);
  int   m  = ep == NULL ? n : (int)(ep - buf);
  buf[n] = 0;
  if (strpbrk(buf, ".ni") == NULL) {
    printf("%.*s.0%s", m, buf, buf + m);
  } else {
    fputs(buf, stdout);
  }
}

// chain is the bracket of the [a, b] or (a, b) this value continues, or 0:
// Con or Nil spell a list, Tuple a tuple, and their tails continue it
static void show_val(Env e, u32 d, const Term* w, char chain) {
  const u32* D = SHOW_DESC;
  Term one;
  char zs[4];
  u32  zn = 0;
  for (bool tail = true; tail;) switch (tail = false, D[d]) {
    case 0: printf("%u", (u32)w[0]); break;
    case 1: show_f32((u32)w[0]); break;
    case 2: printf("%llun", (unsigned long long)w[0]); break;
    case 3:
      putchar('\'');
      show_chr(D[d + 1] != 0 ? term_loc(w[0]) : w[0], '\'');
      putchar('\'');
      break;
    case 4:
      putchar('"');
      {
        StrParts p = str_peek(e, w[0]);
        for (u32 i = 0; i < p.len; i++) { show_chr(str_at_peek(e, p, i), '"'); }
      }
      putchar('"');
      break;
    case 5: fputs("{==}", stdout); break;
    case 6:
      putchar('[');
      // an element is 2^lgs cells of the block
      for (u32 i = 0, g = D[d + 2]; i < 1u << (blk_cls(w[0]) - g); i += 1) {
        Term v[1u << g];
        for (u32 j = 0; j < 1u << g; j += 1) {
          v[j] = blk_read(e.mem, term_tag(w[0]) == TAG_ARR,
            term_peek(e, w[0]), (i << g) + j);
        }
        fputs(i > 0 ? ", " : "", stdout);
        show_val(e, D[d + 1], v, 0);
      }
      putchar(']');
      break;
    default: {
      Term t   = w[0];
      bool box = D[d + 1] != 0;
      u32  key = box ? (u32)term_aux(t) : D[d + 2] > 1 ? (u32)t : 0;
      u32  a   = d + 3;
      for (u32 i = 0; box ? D[a + 1] != key : i != key; i += 1) {
        a += 4 + 2 * D[a + 2];
      }
      if (box) {
        one = term_loc(t);
        w   = term_tag(t) == TAG_PAK ? &one : e.mem + term_peek(e, t);
      }
      char o = "{[("[D[a + 3]];
      if (o == '{') {
        printf("%s{", SHOW_NAMES[D[a]]);
      } else if (chain != o) {
        putchar(o);
      }
      if (o == '{' || chain != o) {
        zs[zn++] = "}])"[D[a + 3]];
      }
      for (u32 j = 0; j < D[a + 2]; j += 1) {
        if (o == '[' ? j == 0 && chain == o : j > 0) {
          fputs(", ", stdout);
        }
        if (j == 1 && o != '{') {
          tail  = true;
          chain = o;
          d     = D[a + 5 + 2 * j];
          w     = w + D[a + 4 + 2 * j];
        } else {
          show_val(e, D[a + 5 + 2 * j], w + D[a + 4 + 2 * j], 0);
        }
      }
    }
  }
  while (zn > 0) {
    putchar(zs[--zn]);
  }
}

#endif

// IO.within's (effs/within.c): an answer handed to whoever waits on it;
// the activation is its to free.
static void (*io_race)(Env e, IoAct* a, Term v);

// The continuation applied to the item is the next request (or req, the
// first, when it is not TERM_HOLE).
static int io_step(Env e, IoAct* a, Term req) {
  for (;; req = TERM_HOLE) {
    if (req == TERM_HOLE) {
      Loc ap = task_node(e, FID_CLO_APPLY, TERM_HOLE, 0, 0);
      e.mem[ap]     = a->cont;
      e.mem[ap + 1] = a->item;
      req = corpus_eval(e.mem, term_tsk(FID_CLO_APPLY, ap));
    }
    u32  c   = (u32)term_aux(req);
    Loc  at  = term_peek(e, req);
    if (c == CID_EMIT) {
      io_live -= 1;
      if (a->race != NULL) {
        Term v[1];
        spare_free(e, cls_fit(1), ctr_take(e, req, 1, v));
        io_race(e, a, v[0]);
        return -1;
      }
      term_drop(e, req);
      free(a);
      return -1;
    }
    if (c == CID_HALT) {
      io_errs(e, e.mem[at + 1]);
      return (int)(u32)e.mem[at];
    }
    if (io_eff_rows[c].run == NULL) {
      err_fail("an alien request");
    }
    u32 need = io_eff_rows[c].ask;
    u32 word = (u32)(need & IO_READ ? io_hand_v(e.mem[at]) : e.mem[at]);
    a->cont  = req;
#ifdef __linux__
    // Parked before any call has found the descriptor empty, an edge
    // already reported (and a read that left bytes behind) would never
    // come again, so a registered one is asked as it stands.
    if (need & IO_READ && io_fdt_cap > word && io_fdt[word].own && io_fdt[word].reg) {
      struct pollfd q = { (int)word, POLLIN, 0 };
      if (poll(&q, 1, 0) > 0) {
        need = 0;
      }
    }
#endif
    if (need != 0) {
      io_wait_on(&a->work, (int)word, need & IO_READ ? POLLIN : 0,
        need & IO_TIME ? io_tick() + (u64)word * 1000000ull : 0, io_exec);
      return -1;
    }
    Term x = io_exec(e, &a->work);
    if (x == IO_PARK) {
      return -1;
    }
    a->item = x;
  }
}

OUTLINE int io_loop(Corpus H) {
  Env e = { H, ALC[0] };
  io_stk = pool_stack();
  signal(SIGPIPE, SIG_IGN);
#ifdef __linux__
  io_ep = epoll_create1(0);
  if (io_ep < 0) {
    err_fail("the poller could not be opened");
  }
#endif
  if (pipe(io_wake_fd) | fcntl(io_wake_fd[0], F_SETFL, O_NONBLOCK)) {
    err_fail("the event loop failed to open");
  }
#ifdef __linux__
  struct epoll_event wake = { .events = EPOLLIN, .data.u64 = 0 };
  if (epoll_ctl(io_ep, EPOLL_CTL_ADD, io_wake_fd[0], &wake)) {
    err_fail("the event loop failed to open");
  }
#endif
  Term m = corpus_eval(H, term_tsk(MAIN_FID, task_node(e, MAIN_FID,
    TERM_HOLE, 0, 0)));
#if MAIN_PURE
  show_val(e, 0, H + H_ROOT_WORD, 0);
  putchar('\n');
  return 0;
#endif
  io_spawn(m);
  for (u32 n = 0;; n += 1) {
    if (io_runs.head == NULL) {
      if (io_live == 0) {
        return 0;
      }
      if (io_park.head == NULL && io_idle() && io_busy == 0) {
        io_sync();
        fprintf(stderr, "bend: deadlock: every computation waits on a"
          " channel\n");
        return 1;
      }
      io_wait(e);
      continue;
    }
    if ((n & 63) == 0 && io_busy != 0) {
      io_take(e);
    }
    int code = io_step(e, io_pop(&io_runs), TERM_HOLE);
    if (code >= 0) {
      return code;
    }
  }
}

// Requests
// ========

${reqs}

// Cli
// ===

static void cli_fail(const char* msg, const char* arg) {
  fprintf(stderr, "bend: %s%s\n", msg, arg != NULL ? arg : "");
  exit(1);
}

// Workers
// =======

// --workers N: the process forks N copies of the program before it
// starts a thread, and stays as their master, as nginx's does: it runs
// no program itself, passes SIGTERM, SIGINT and SIGHUP on to every
// copy, and ends when they have all ended, with the first failure's
// code. A copy dies with its master. Each is its own heap, its own loop
// and its own threads; a server's copies share its port through
// SO_REUSEPORT, and the kernel spreads the connections between them.
static pid_t* cli_kids;
static int    cli_kids_n;

static void cli_pass(int sig) {
  for (int i = 0; i < cli_kids_n; i += 1) {
    if (cli_kids[i] > 0) {
      kill(cli_kids[i], sig);
    }
  }
}

static int cli_workers(long n) {
  cli_kids   = io_mem(calloc((size_t)n, sizeof *cli_kids));
  pid_t boss = getpid();
  for (long i = 0; i < n; i += 1) {
    pid_t k = fork();
    if (k < 0) {
      cli_pass(SIGTERM);
      cli_fail("could not fork a worker", NULL);
    }
    if (k == 0) {
#ifdef __linux__
      prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
      if (getppid() != boss) {
        _exit(1);
      }
      return -1;
    }
    cli_kids[cli_kids_n++] = k;
  }
  struct sigaction sa = { .sa_handler = cli_pass };
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, NULL);
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGHUP, &sa, NULL);
  int code = 0, left = cli_kids_n, st = 0;
  while (left > 0) {
    pid_t k = waitpid(-1, &st, 0);
    if (k < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    left -= 1;
    int c = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    code  = code != 0 ? code : c;
  }
  return code;
}

// Main
// ====

int main(int argc, char** argv) {
  long thr = 0;
  long wks = 1;
  int  gpu = -1;
  u64  mem = 0;
  io_argv = argv + 1;
  for (int i = 1; i < argc; i += 1) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : NULL;
    if (strcmp(a, "--") == 0) {
      while (i + 1 < argc) {
        io_argv[io_argc++] = argv[++i];
      }
    } else if (strcmp(a, "--help") == 0) {
      printf(CLI_HELP, argv[0]);
      return 0;
    } else if (strcmp(a, "--gpu-build") == 0) {
      if (gpu_probe() && !gpu_make(gpu_path())) {
        cli_fail("cannot write ", gpu_path());
      }
      return 0;
    } else if (strcmp(a, "--threads") == 0) {
      char* end = NULL;
      thr = v != NULL ? strtol(v, &end, 10) : 0;
      if (thr < 1 || end == NULL || *end != '\0') {
        cli_fail("expected a thread count of 1 or more after --threads", NULL);
      }
      i += 1;
    } else if (strcmp(a, "--workers") == 0) {
      char* end = NULL;
      wks = v != NULL ? strtol(v, &end, 10) : 0;
      if (wks < 1 || wks > 1024 || end == NULL || *end != '\0') {
        cli_fail("expected a worker count of 1 to 1024 after --workers", NULL);
      }
      i += 1;
    } else if (strcmp(a, "--gpu") == 0) {
      char*  end = NULL;
      double n   = v != NULL ? strtod(v, &end) : 0;
      u64    mul = end == NULL ? 0 : strcmp(end, "GB") == 0 ? 1ull << 30
        : strcmp(end, "MB") == 0 ? 1ull << 20 : 0;
      if (v != NULL && strcmp(v, "off") == 0) {
        gpu = 0;
      } else if (v != NULL && (strcmp(v, "on") == 0 || (mul != 0 && n > 0))) {
        gpu = 1;
        mem = (u64)(n * (double)mul);
      } else {
        cli_fail("expected on, off or a size like 4GB after --gpu", NULL);
      }
      i += 1;
    } else {
      io_argv[io_argc++] = argv[i];
    }
  }
  if (wks > 1) {
    int code = cli_workers(wks);
    if (code >= 0) {
      return code;
    }
  }
  bool dev = gpu != 0 && BANGS != 0 && gpu_probe();
  if (gpu == 1 && BANGS != 0 && !dev) {
    cli_fail("--gpu on, but this binary found no GPU device", NULL);
  }
  Corpus H  = corpus_setup(dev, thr > 0 ? thr : cpu_count(), mem);
  int code  = io_loop(H);
  io_sync();
  return code;
}

#endif
`.slice(1);

// RuntimeJs
// =========

const RUNTIME: string = String.raw`
${NATIVE.JS}
// Array
// =====

function array_new(d, v) {
  if (d > 31n) {
    throw "bend: ${ERRS[8]}";
  }
  return Array(2 ** Number(d)).fill(v);
}

// An unbalanced tree fails, as in C.
function array_node(a, b) {
  if (a.length !== b.length) {
    throw "bend: ${ERRS[2]}";
  }
  return a.concat(b);
}

function array_rmw(a, i, f) {
  const at = i % a.length;
  const old = a[at];
  a[at] = f(old);
  return {$: "Tuple", fst: a, snd: old};
}

function array_get4(a, i) {
  const x = (k) => a[(i + k) % a.length];
  return {$: "Tuple", fst: a, snd: {$: "Tuple", fst: x(0), snd: {$: "Tuple",
    fst: x(1), snd: {$: "Tuple", fst: x(2), snd: x(3)}}}};
}

// Run
// ===

function run_jump(f, x) {
  return {$: "$JMP", f: f, x: x};
}

function run_tail(f, x) {
  return {$: "$JMP", f: f.j?.f === f ? f.j : f, x: [x]};
}

function run_clo(j) {
  const f = (x) => run_loop(j(x));
  f.j = j;
  j.f = f;
  return f;
}

function run_loop(r) {
  while (r !== null && typeof r === "object" && r.$ === "$JMP") {
    r = r.f(...r.x);
  }
  return r;
}

function run_lib(f, n) {
  return (...a) => a.length < n ? run_lib((...b) => f(...a, ...b), n - a.length)
    : run_loop(f(...a));
}
`.slice(1);

const RUNTIME_MAIN: string = String.raw`
// Cli
// ===

// A JS program runs one thread, one process and no GPU: --threads,
// --workers and --gpu do nothing.
let cli_args = [];

function cli(argv) {
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--") {
      cli_args.push(...argv.slice(i + 1));
      break;
    } else if (argv[i] === "--help") {
      io_out(1, io_bytes("usage: " + process.argv[1] + "\n"));
      process.exit(0);
    } else if (argv[i] === "--threads" || argv[i] === "--workers" || argv[i] === "--gpu") {
      i += 1;
    } else {
      cli_args.push(argv[i]);
    }
  }
}

// Show
// ====

// char_show: an escape, a \u{hex}, else the code point
function show_chr(c, q) {
  const k = { 10: "n", 9: "t", 13: "r", 0: "0", 92: "\\" }[c]
    ?? (c === q.codePointAt(0) ? q : null);
  return k !== null ? "\\" + k : c < 32 || c === 127
    || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)
    ? "\\u{" + c.toString(16) + "}" : String.fromCodePoint(c);
}

// A pure main's value as term_show spells it (see show_main); chain is the
// bracket it continues, or 0.
function show_val(D, N, d, v, chain) {
  if (D[d] === 7) {
    const fs = Object.values(typeof v === "boolean"
      ? { $: v ? "True" : "False" } : v);
    let a = d + 3;
    for (; N[D[a]] !== fs[0]; a += 4 + 2 * D[a + 2]) {}
    const o = "{[("[D[a + 3]];
    let s = o === "{" ? fs[0] + "{" : chain === o ? "" : o;
    for (const [j, f] of fs.slice(1).entries()) {
      if (o === "[" ? j === 0 && chain === o : j > 0) {
        s += ", ";
      }
      s += show_val(D, N, D[a + 5 + 2 * j], f, j === 1 && o !== "{" ? o : 0);
    }
    return o === "{" || chain !== o ? s + "}])"[D[a + 3]] : s;
  }
  return D[d] === 0 ? String(v)
    : D[d] === 1 ? f32_show(v).replace(/^-?\d+(?=e|$)/, "$&.0")
    : D[d] === 2 ? v + "n"
    : D[d] === 3 ? "'" + show_chr(char_code(v), "'") + "'"
    : D[d] === 4 ? "\"" + [...v].map((c) =>
      show_chr(c.codePointAt(0), "\"")).join("") + "\""
    : D[d] === 5 ? "{==}"
    : "[" + v.map((x) => show_val(D, N, D[d + 1], x, 0)).join(", ") + "]";
}

// Io
// ==

function io_exit(main, show) {
  try {
    if (show !== null) {
      io_out(1, io_bytes(show_val(...show, 0, run_loop(main()), 0) + "\n"));
      process.exit(0);
    }
    process.exit(io_run(main));
  } catch (e) {
    io_errs(String(e));
    process.exit(1);
  }
}

function io_out(fd, data) {
  const fs = require("fs");
  let at = 0;
  while (at < data.length) {
    try {
      at += fs.writeSync(fd, data, at, data.length - at);
    } catch (e) {
      if (e.code === "EAGAIN" || e.code === "EINTR") {
        continue;
      }
      try {
        fs.writeSync(2, "bend: a short write on a standard stream\n");
      } catch (o) {
      }
      process.exit(1);
    }
  }
}

function io_errs(message) {
  io_out(2, io_bytes(message + "\n"));
}

function io_sys() {
  if (globalThis.BEND_SYS === undefined) {
    const ffi = require("bun:ffi");
    const mac = process.platform === "darwin";
    const err = mac ? "__error" : "__errno_location";
    // Darwin's extended select supports high fds.
    const sel = mac ? "select$DARWIN_EXTSN" : "select";
    const T = { i: "i32", u: "u32", U: "u64", I: "i64", p: "ptr",
      c: "cstring" };
    // Apple arm64 passes variadic fcntl flags on the stack: use the ninth
    // fixed argument (the third elsewhere).
    const vari = mac && process.arch === "arm64";
    const lib = ffi.dlopen(mac ? "libSystem.dylib" : "libc.so.6",
      Object.fromEntries(("socket:iii>i bind:ipu>i listen:ii>i connect:ipu>i"
        + " accept:ipp>i send:ipUi>I recv:ipUi>I read:ipU>I pread:ipUI>I"
        + " sendto:ipUipu>I recvfrom:ipUipp>I close:i>i setsockopt:iiipu>i"
        + " " + sel + ":ipppp>i"
        + (vari ? " fcntl:iiiiiiiii>i" : " fcntl:iii>i") + " getsockopt:iiipp>i"
        + " strerror:i>c " + err + ":>p").split(" ").map((s) => {
        const [name, args, ret] = s.split(/[:>]/);
        return [name, { args: [...args].map((a) => T[a]), returns: T[ret] }];
      }))).symbols;
    const fcntl = (fd, cmd, arg) => vari
      ? lib.fcntl(fd, cmd, 0, 0, 0, 0, 0, 0, arg)
      : lib.fcntl(fd, cmd, arg);
    globalThis.BEND_SYS = { ...lib, fcntl, select: lib[sel],
      ptr: ffi.ptr, mac,
      errno: () => ffi.read.i32(lib[err](), 0) };
  }
  return globalThis.BEND_SYS;
}

function io_fail(code) {
  return { $: "Fail",
    error: io_tup(code >>> 0, String(io_sys().strerror(code))) };
}

function io_done(value) {
  return { $: "Done", value };
}

function io_tup(...xs) {
  return xs.reduceRight((snd, fst) => ({ $: "Tuple", fst, snd }));
}

function io_bytes(text) {
  if (typeof text !== "string") { throw "bend: cannot encode a non-scalar Char as UTF-8"; }
  for (const c of text) {
    const code = c.codePointAt(0);
    if (code >= 0xd800 && code <= 0xdfff) { throw "bend: cannot encode a non-scalar Char as UTF-8"; }
  }
  return new TextEncoder().encode(text);
}

// One replacement per ill-formed byte, including truncated sequences; BOM
// is a normal U+FEFF element. Each IO chunk is decoded independently.
// Well-formed input takes the native decoder (fatal rejects exactly the
// ill-formed chunks; the byte walk below then keeps the per-byte contract).
function io_text(b, n) {
  try {
    return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(b.subarray(0, n));
  } catch (_) {}
  const out = [];
  for (let i = 0; i < n;) {
    const h = b[i];
    const k = h < 0x80 ? 1 : h >= 0xc2 && h <= 0xdf ? 2
      : h >= 0xe0 && h <= 0xef ? 3 : h >= 0xf0 && h <= 0xf4 ? 4 : 0;
    let c = k === 1 ? h : h & (0x7f >> k), ok = k > 0 && k <= n - i;
    for (let j = 1; ok && j < k; j++) {
      const t = b[i + j]; ok = (t & 0xc0) === 0x80; c = (c << 6) | (t & 63);
    }
    ok = ok && (k === 1 || c >= (k === 2 ? 0x80 : k === 3 ? 0x800 : 0x10000))
      && c <= 0x10ffff && !(c >= 0xd800 && c <= 0xdfff);
    out.push(String.fromCodePoint(ok ? c : 0xfffd));
    i += ok ? k : 1;
  }
  return out.join("");
}

function io_addr(host, port) {
  const part = host.split(".");
  const deci = (p) => /^(0|[1-9]\d{0,2})$/.test(p) && Number(p) < 256;
  if (port > 65535 || part.length !== 4 || !part.every(deci)) {
    return null;
  }
  const b = new Uint8Array(16);
  const head = io_sys().mac ? [16, 2] : [2, 0];
  b.set([...head, port >> 8, port & 255, ...part.map(Number)]);
  return b;
}

// The 16 bytes of an IPv6 address in its text form ("::1",
// "::ffff:1.2.3.4"; no brackets, no zone), as inet_pton reads one, or null
function io_ip6(host) {
  const two = typeof host === "string" ? host.indexOf("::") : -2;
  if (two === -2 || !/^[0-9A-Fa-f:.]+$/.test(host) || two !== host.lastIndexOf("::")
    || (two >= 0 && host.slice(two).startsWith(":::"))) {
    return null;
  }
  const side = (s, tail) => {
    const out = [];
    const ps = s === "" ? [] : s.split(":");
    for (let i = 0; i < ps.length; i += 1) {
      const p = ps[i];
      const v4 = tail && i === ps.length - 1 && p.includes(".") ? io_addr(p, 0) : null;
      if (v4 !== null) {
        out.push(v4[4] << 8 | v4[5], v4[6] << 8 | v4[7]);
      } else if (/^[0-9A-Fa-f]{1,4}$/.test(p)) {
        out.push(parseInt(p, 16));
      } else {
        return null;
      }
    }
    return out;
  };
  const l = side(two < 0 ? host : host.slice(0, two), two < 0);
  const r = two < 0 ? [] : side(host.slice(two + 2), true);
  if (l === null || r === null || (two < 0 ? l.length !== 8 : l.length + r.length > 7)) {
    return null;
  }
  const gs = [...l, ...new Array(8 - l.length - r.length).fill(0), ...r];
  const b = new Uint8Array(16);
  gs.forEach((g, i) => b.set([g >> 8, g & 255], 2 * i));
  return b;
}

// An IPv6 address's text as RFC 5952 writes it: lowercase, no leading
// zeros, the longest run of two or more zero groups (the first of equals)
// as "::", and an IPv4-mapped one as ::ffff: and the dotted address
function io_ip6_show(b) {
  const g = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => b[2 * i] << 8 | b[2 * i + 1]);
  if (g.slice(0, 5).every((x) => x === 0) && g[5] === 0xffff) {
    return "::ffff:" + [b[12], b[13], b[14], b[15]].join(".");
  }
  let at = -1, len = 0;
  for (let i = 0; i < 8;) {
    let j = i;
    while (j < 8 && g[j] === 0) {
      j += 1;
    }
    if (j - i > len) {
      at = i;
      len = j - i;
    }
    i = j === i ? i + 1 : j;
  }
  const hex = (xs) => xs.map((x) => x.toString(16)).join(":");
  return len < 2 ? hex(g) : hex(g.slice(0, at)) + "::" + hex(g.slice(at + len));
}

// A socket address of either family, the host read as io_addr or io_ip6
// reads it: { fam, b } (b the address's bytes), or null
function io_sa(host, port) {
  const v4 = io_addr(host, port);
  if (v4 !== null) {
    return { fam: 2, b: v4 };
  }
  const ip = port > 65535 ? null : io_ip6(host);
  if (ip === null) {
    return null;
  }
  const mac = io_sys().mac;
  const b = new Uint8Array(28);
  b.set(mac ? [28, 30, port >> 8, port & 255] : [10, 0, port >> 8, port & 255]);
  b.set(ip, 8);
  return { fam: mac ? 30 : 10, b };
}

// As io_sys_dual: an IPv6 listener takes IPv4 too, where it may
function io_dual(fd, fam) {
  if (fam !== 2) {
    const sys = io_sys();
    sys.setsockopt(fd, 41, sys.mac ? 27 : 26, sys.ptr(new Int32Array([0])), 4);
  }
}

function io_push(fun, arg, fresh) {
  const io = globalThis.BEND_IO;
  io.runs.push({ fun, arg });
  io.live += fresh ? 1 : 0;
}

// A waiter on the clock alone sits in io.heap, a min-heap on its deadline
// (the earlier park breaks a tie), so a sleeper costs a pass nothing.
function io_heap_lt(a, b) {
  return a.at < b.at || a.at === b.at && a.n < b.n;
}

function io_heap_pop(h) {
  const top = h[0], w = h.pop();
  let i = 0;
  for (let c = 1; c < h.length; c = 2 * i + 1) {
    c += c + 1 < h.length && io_heap_lt(h[c + 1], h[c]) ? 1 : 0;
    if (!io_heap_lt(h[c], w)) {
      break;
    }
    h[i] = h[c];
    i = c;
  }
  if (h.length > 0) {
    h[i] = w;
  }
  return top;
}

function io_wait(io) {
  const h = io.heap;
  const soon = io.waits.reduce((m, w) => Math.min(m, w.at ?? m),
    h.length > 0 ? h[0].at : Infinity);
  const ms = soon === Infinity ? -1
    : Math.max(0, Math.ceil(soon - performance.now()));
  const fds = io.waits.filter((w) => w.fd !== undefined);
  const top = fds.reduce((m, w) => Math.max(m, w.fd), 0);
  const len = (top >> 6 << 3) + 8;
  const set = new Uint8Array(2 * len);
  const at = (w) => (w.out ? len : 0) + (w.fd >> 3);
  for (const w of fds) {
    set[at(w)] |= 1 << (w.fd & 7);
  }
  const tv = new BigInt64Array([BigInt(ms / 1000 | 0),
    BigInt(ms % 1000 * 1000)]);
  const sys = io_sys();
  sys.select(top + 1, sys.ptr(set), sys.ptr(set, len), null,
    ms < 0 ? null : sys.ptr(tv));
  const now = performance.now();
  io.waits = io.waits.filter((w) => {
    const ready = w.at <= now || w.fd !== undefined
      && set[at(w)] & 1 << (w.fd & 7);
    if (ready) {
      io_push(io_wake, w, false);
    }
    return !ready;
  });
  while (h.length > 0 && h[0].at <= now) {
    io_push(io_wake, io_heap_pop(h), false);
  }
}

// Resume k with more's value; undefined means re-parked.
function io_wake(w) {
  const x = w.more();
  return x === undefined ? undefined : w.k(x);
}

// Park for read/write (out) or until at (performance.now()); an undefined
// fd or at disables that source. A descriptor's waiter is answered, so a
// channel's bell can ring it (at = 0: due at the next pass).
function io_park_on(fd, out, k, more, at) {
  const io = globalThis.BEND_IO;
  const w = { fd, out, k, more, at, n: io.n++ };
  if (fd !== undefined) {
    io.waits.push(w);
    return w;
  }
  let i = io.heap.push(w) - 1;
  for (; i > 0 && io_heap_lt(w, io.heap[i - 1 >> 1]); i = i - 1 >> 1) {
    io.heap[i] = io.heap[i - 1 >> 1];
  }
  io.heap[i] = w;
}

function io_run(m) {
  const io = { runs: [], live: 0, waits: [], heap: [], n: 0 };
  globalThis.BEND_IO = io;
  try {
    io_push(run_loop(m()), (x) => ({ $: "Emit", value: x }), true);
    for (;;) {
      if (io.runs.length === 0) {
        if (io.live === 0) {
          return 0;
        }
        if (io.waits.length + io.heap.length === 0) {
          io_errs("bend: deadlock: every computation waits on a channel");
          return 1;
        }
        io_wait(io);
        continue;
      }
      const s = io.runs.shift();
      let op = s.fun(s.arg);
      while (op !== undefined) {
        if (op.$ === "Emit") {
          io.live -= 1;
          break;
        }
        if (op.$ === "Halt") {
          io_errs(op.message);
          return op.code;
        }
        const need = op.need?.() ?? {};
        if (need.time || need.read) {
          const more = () => op.run(...op.args, op.kont);
          io_park_on(need.read ? op.args[0] : undefined, false, op.kont, more,
            need.read ? undefined : performance.now() + Number(op.args[0]));
          break;
        }
        const x = op.run(...op.args, op.kont);
        if (x === undefined) {
          break;
        }
        op = op.kont(x);
      }
    }
  } catch (req) {
    if (req instanceof RangeError) {
      throw "bend: ${ERRS[7]}";
    }
    if (req?.$ !== "$FFI") {
      throw req;
    }
    io_errs("bend: ${ERRS[2]}");
    return 1;
  }
}
`.slice(1);

const TAB_BAD = /\b(?!(?:fround|imul|Number|BigInt)\()\w+\(/;
