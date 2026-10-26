# engine_gpu <packdir> <tok>... [--steps N] [--layers L] [--fork F]
#
# v0: F32, one token per forward with a KV cache, greedy argmax, N steps on raw
# token ids (no EOT/chat). stdout: the N generated ids, one per line, then
# `wall_ms <ms>`. stderr: `op <tag> <ms>` per op, `step <pos> <ms>` per
# forward, and `logits <pos> <id1> <bits1> <id2> <bits2>` (top two, F32 U32 bits).
# --layers L runs the first L layers only (a test knob; default all).
# All durations use IO.now's monotonic millisecond clock. argmax includes the
# diagnostic second pick; small ops may measure 0 ms, not zero compute cost.
#
# Weights: token_embd, output_norm and output from pack.bin; layer i from
# layers/blk.i.bin (tools/split_pack.py: File.read_at takes a U32 offset and
# pack.bin is 10 GB). Offsets come from manifest.json; the dims below (2048,
# 16x128 q, 2x128 kv, 6144 ffn, 130560 vocab) are this pack's (ops.md). The op
# order is kernels/ops.bend's; tools/verify_forward.py mirrors it.
#
# GPU lane (engine_gpu): the same forward, one `!` per layer (`lay`) and one
# for the head (`head`: rms_final, output gemv, top two), over weights resident
# in the heap (kernels/flat.bend's fork-trees, bit-exact with ops.bend). The
# layer files, token_embd, output_norm and output.weight are read once at
# start into Bytes; a step does no file IO. Intermediates stay on the device:
# with managed memory, every host read of a bang's output faults its pages
# back across PCIe, which cost more than the bangs themselves. --fork F forks
# each gemv F levels (default 14); the tree, so every bit, is the same for any F.
# stderr per step: `op embed|layer|head <ms>`. --gpu off runs the same bangs
# on the CPU pool.
#
# Q8_0 (quant-q8): a pack with format llama-bend-pack-q8/v0 (tools/q8_pack.py)
# runs the same forward on the GGUF's Q8_0 bytes, dequantized on use (w =
# f32(q) * f32(d), exact: kernels/flat.bend's q8.gemv); F32 packs run as before.
# F16 (quant-f16): format llama-bend-pack-f16/v0, f16le matrices upconverted
# exactly on use (kernels/flat.bend's h16.gemv, tools/merge3-f16.patch).
# Q4_K (w0-quants): format llama-bend-pack-q4k/v0 (tools/q4k_pack.py), GGUF
# block layout kept whole (144 bytes a 256-weight super-block, tools/q4k.py);
# each weight w = d * sc * nib - dmin * m dequantized on use, ggml's
# dequantize_row_q4_K (kernels/flat.bend's q4k.gemv, tools/q4k.patch).
import Base
import ./kernels/ops.bend as K
import ./kernels/io.bend as I
import ./kernels/flat.bend as F

# a layer's tensors: byte offsets within its layer file
type Offs is Data:
  Offs{an: U32, q: U32, k: U32, v: U32, o: U32, fn: U32, g: U32, u: U32, dn: U32}

# layers to run, a layer's offsets, the resident layer files, output.weight,
# token_embd and output_norm
type Model is Data:
  Model{nl: Nat, lay: Offs, ws: List<&2, Bytes()>, ow: Bytes(), eb: Bytes(), nb: Bytes(), q8: Bool, hf: Bool, q4k: Bool, k6: Bool, x4: Bool, rvs: List<&2, U32>, rds: List<&2, U32>}

def m.offs(m: Model) -> Offs:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      lay

def m.nl(m: Model) -> Nat:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      nl

def m.ow(m: Model) -> Bytes():
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      ow

def m.eb(m: Model) -> Bytes():
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      eb

def m.nb(m: Model) -> Bytes():
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      nb

def m.q8(m: Model) -> Bool:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      q8

def m.hf(m: Model) -> Bool:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      hf

def m.q4k(m: Model) -> Bool:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      q4k

def m.x4(m: Model) -> Bool:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      x4

# a matrix's bytes: F32, Q8_0 row-split (tools/q8_pack.py: 34 bytes per 32
# weights), F16 (half), or Q4_K block (tools/q4k_pack.py: 144 bytes per 256
# weights; f32b is 4 elems, so 4/64 * 9 = 9/16 bytes an element)
def mb(+q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +f32b: U32) -> U32:
  Bool.pick(U32, x4, ((f32b / 1024 : U32) * 136 : U32),
    Bool.pick(U32, hf, (f32b / 2 : U32), Bool.pick(U32, q8, ((f32b / 64 : U32) * 17 : U32),
      Bool.pick(U32, q4k, ((f32b / 64 : U32) * 9 : U32), f32b))))

# a row's word as F.mv and llb_mail read it: the row's bytes, tagged with the
# pack's format (tools/quant-fix.patch: bit 30, then 0 F32, 1 Q8_0, 2 F16,
# 3 Q4_K in bits 28-29; mf.validate and mf.relative check the pack against it)
def rw(+q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +f32b: U32) -> U32:
  (mb(q4k, q8, hf, x4, f32b) .|. Bool.pick(U32, x4, 1207959552,
    Bool.pick(U32, q4k, 1879048192, Bool.pick(U32, hf, 1610612736,
      Bool.pick(U32, q8, 1342177280, 1073741824)))) : U32)

# Q6_K matrix bytes (tools/q6k.py): 210 bytes a 256-weight super-block, so a
# row's bytes for f32b bytes of f32 (4c) is (f32b/1024)*210
def mb6(+f32b: U32) -> U32:
  ((f32b / 1024 : U32) * 210 : U32)

# the Q6_K row word: tag bit 30 | bit 31 (format 4, tools/q6k.patch), row bytes below
def rw6(+f32b: U32) -> U32:
  (mb6(f32b) .|. 3221225472 : U32)

# one U32 in a List<&2, U32> by index (like ws.at)
def row.at(n: Nat, xs: List<&2, U32>) -> U32:
  match n xs:
    case 0n Con{x, t}:
      x
    case 1n+p Con{x, t}:
      row.at(p, t)
    case _ _:
      0

# mixed Q4_K_M pack: attn_v / ffn_down / output.weight may be Q6_K. rvs and
# rds are the per-layer row words read from the manifest (Q4_K or Q6_K); for
# every other pack they are empty and the uniform rw/q4k path is used.
def m.k6(m: Model) -> Bool:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      k6

def m.rv(m: Model, i: U32) -> U32:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      Bool.pick(U32, k6, row.at(U32.to_nat(i), rvs), rw(q4k, q8, hf, x4, 8192))

def m.rd(m: Model, i: U32) -> U32:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      Bool.pick(U32, k6, row.at(U32.to_nat(i), rds), rw(q4k, q8, hf, x4, 24576))

# output.weight's row word (Q6_K in a Q4_K_M mixture)
def m.orw(m: Model) -> U32:
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      Bool.pick(U32, k6, rw6(8192), rw(q4k, q8, hf, x4, 8192))

def ws.at(n: Nat, ws: List<&2, Bytes()>) -> Bytes():
  match n ws:
    case 0n Con{w, t}:
      w
    case 1n+p Con{w, t}:
      ws.at(p, t)
    case _ _:
      SNil{}

def m.w(m: Model, i: U32) -> Bytes():
  match m:
    case Model{nl, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}:
      ws.at(U32.to_nat(i), ws)

# Resident weights
# ----------------

# The layer files (tools/split_pack.py: 188760064 bytes each; Q8: tools/q8_pack.py,
# 50151424) and output.weight.
def ws.close(got: File & Bytes()) -> IO(Bytes()):
  (f, b) = got
  do IO<Bytes()>:
    File.close(f)
    return b

def ws.load(+path: String, +off: U32, +n: U32) -> IO(Bytes()):
  do IO<Bytes()>:
    f : File <- IO.try(File, File.open(path, "r"))
    h : File <- I.io.buf.skip(U32.to_nat(off), U32.to_nat(off), f)
    got : File & Bytes() <- I.io.buf.read(h, n)
    ws.close(got)

# layers i..i+n-1, in order
def ws.layers(n: Nat, +dir: String, +lb: U32, +i: U32) -> IO(List<&2, Bytes()>):
  match n:
    case 0n:
      IO.pure(List<&2, Bytes()>, Nil{})
    case 1n+p:
      do IO<List<&2, Bytes()>>:
        w : Bytes() <- ws.load(dir ++ "/layers/blk." ++ U32.show(i) ++ ".bin", 0, lb)
        rest : List<&2, Bytes()> <- ws.layers(p, dir, lb, (i + 1 : U32))
        return w <> rest

# manifest.json
# -------------

def mf.after(+s: String, n: Nat, m: Maybe<&2, Nat>) -> String:
  match m:
    case Some{i}:
      String.drop(s, Nat.add(i, n))
    case None{}:
      ""

def mf.upto(+s: String, m: Maybe<&2, Nat>) -> String:
  match m:
    case Some{k}:
      String.take(s, k)
    case None{}:
      ""

# The converter's schema has flat config/tensor objects (shape is an array).
# Bound the lookup to that object: a missing key must not match a later tensor.
# This is a schema-specific reader, not a general-purpose JSON parser.
def mf.colon(s: String) -> String:
  match s:
    case SCon{Chr{58}, t}:
      String.trim_start(t)
    case _:
      ""

def mf.field(+s: String, +key: String) -> String:
  +k = "\"" ++ key ++ "\""
  mf.colon(String.trim_start(mf.after(s, String.length(k), String.find(s, k))))

def mf.object.body(s: String) -> String:
  match s:
    case SCon{Chr{123}, t0}:
      +t = t0
      mf.upto(t, String.find(t, "}"))
    case _:
      ""

def mf.object(s: String, name: String) -> String:
  mf.object.body(mf.field(s, name))

def mf.scalar.end(+s: String, at: Maybe<&2, Nat>) -> String:
  match at:
    case Some{n}:
      String.trim(String.take(s, n))
    case None{}:
      String.trim(s)

def mf.scalar(s: String, key: String) -> String:
  +u = mf.field(s, key)
  +v = mf.scalar.end(u, String.find(u, ","))
  mf.scalar.end(v, String.find(v, "}"))

def mf.shape.body(s: String) -> String:
  match s:
    case SCon{Chr{91}, t0}:
      +t = t0
      +v = mf.upto(t, String.find(t, "]"))
      "[" ++ String.replace(String.replace(String.replace(String.replace(v,
        " ", ""), "\n", ""), "\r", ""), "\t", "") ++ "]"
    case _:
      ""

def mf.shape(s: String) -> String:
  mf.shape.body(mf.field(s, "shape"))

def mf.value(s: String, name: String, key: String) -> String:
  mf.scalar(mf.object(s, name), key)

def mf.num(s: String, name: String, key: String) -> Maybe<&2, U32>:
  U32.read(mf.value(s, name, key))

def mf.check(ok: Bool, what: String) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, "unsupported/invalid v0 pack: " ++ what)

def mf.config(+s: String, +key: String, want: String) -> IO(Unit):
  mf.check(String.eq(mf.value(s, "config", key), want), key)

def mf.validate(+s: String, +q4k: Bool, +q8: Bool, +hf: Bool, +k6: Bool, +x4: Bool, ly: U32) -> IO(Unit):
  do IO<Unit>:
    mf.check(Bool.or(x4, Bool.or(Bool.or(q4k, Bool.or(q8, hf)), String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-f32/v0\""))), "format")
    mf.check(String.eq(mf.scalar(s, "dtype"), Bool.pick(String, x4, "\"iq4_xs-block\"",
      Bool.pick(String, k6, "\"q4_k_m-mixed\"",
        Bool.pick(String, q4k, "\"q4_k-block\"",
          Bool.pick(String, hf, "\"f16le\"", Bool.pick(String, q8, "\"q8_0-rowsplit\"", "\"f32le\"")))))), "dtype")
    mf.config(s, "architecture", "\"llama\"")
    mf.config(s, "n_layers", "42")
    mf.config(s, "n_heads", "16")
    mf.config(s, "n_kv_heads", "2")
    mf.config(s, "d_model", "2048")
    mf.config(s, "head_dim", "128")
    mf.config(s, "d_ff", "6144")
    mf.config(s, "vocab_size", "130560")
    mf.config(s, "context_length", "131072")
    mf.config(s, "rope_freq_base", "5000000.0")
    mf.config(s, "rope_dim", "128")
    mf.config(s, "rope_type", "\"norm\"")
    mf.config(s, "rope_scaling", "null")
    mf.config(s, "rms_norm_eps", "9.999999974752427e-07")
    mf.config(s, "tied_embeddings", "false")
    mf.check((ly <= 42 : U32), "--layers must be 0..42")

def mf.need(what: String, m: Maybe<&2, U32>) -> IO(U32):
  match m:
    case Some{v}:
      IO.pure(U32, v)
    case None{}:
      IO.die(U32, 1, "manifest.json: no " ++ what)

def mf.off(+s: String, +name: String) -> IO(U32):
  mf.need(name ++ " offset", mf.num(s, name, "offset"))

def mf.need_nat(what: String, m: Maybe<&2, Nat>) -> IO(Nat):
  match m:
    case Some{v}:
      IO.pure(Nat, v)
    case None{}:
      IO.die(Nat, 1, "manifest.json: no " ++ what)

def mf.off_nat(+s: String, +name: String) -> IO(Nat):
  mf.need_nat(name ++ " offset", Nat.read(mf.value(s, name, "offset")))

def mf.tensor(+s: String, +name: String, shape: String, +bytes: U32, +size: Nat) -> IO(Unit):
  +obj = mf.object(s, name)
  do IO<Unit>:
    mf.check(String.eq(mf.shape(obj), shape), name ++ " shape")
    b : U32 <- mf.need(name ++ " bytes", U32.read(mf.scalar(obj, "bytes")))
    mf.check(U32.is_eq(b, bytes), name ++ " bytes")
    +off : Nat <- mf.off_nat(s, name)
    mf.check(Bool.and(U32.is_eq(U32.and(U32.from_nat(off), 63), 0),
      Nat.is_le(Nat.add(off, U32.to_nat(bytes)), size)), name ++ " bounds/alignment")

# a Q4_K_M tensor whose format varies per layer (attn_v, ffn_down): the row word
# comes from the manifest's "rw" (Q4_K 0x70000000|rb or Q6_K 0xC0000000|rb), so
# the whole-tensor bytes must be rows * row_bytes, not a fixed size
def mf.tensor.rw(+s: String, +name: String, shape: String, +rows: U32, +size: Nat) -> IO(Unit):
  +obj = mf.object(s, name)
  do IO<Unit>:
    mf.check(String.eq(mf.shape(obj), shape), name ++ " shape")
    +rwm : U32 <- mf.need(name ++ " rw", U32.read(mf.scalar(obj, "rw")))
    +b : U32 <- mf.need(name ++ " bytes", U32.read(mf.scalar(obj, "bytes")))
    mf.check(U32.is_eq(b, U32.mul(rows, (rwm .&. 268435455 : U32))), name ++ " bytes")
    +off : Nat <- mf.off_nat(s, name)
    mf.check(Bool.and(U32.is_eq(U32.and(U32.from_nat(off), 63), 0),
      Nat.is_le(Nat.add(off, U32.to_nat(b)), size)), name ++ " bounds/alignment")

# a manifest field as U32, 0 if absent/unreadable (pure)
def mf.u32.m(m: Maybe<&2, U32>) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      0

def mf.u32(+s: String) -> U32:
  mf.u32.m(U32.read(s))

# the per-layer row word of tensor `name` (attn_v.weight / ffn_down.weight)
def mf.rwlist(n: Nat, +i: U32, +s: String, +name: String, acc: List<&2, U32>) -> List<&2, U32>:
  match n:
    case 0n:
      List.reverse(&2, U32, acc)
    case 1n+p:
      +t = "blk." ++ U32.show(i) ++ "." ++ name
      mf.rwlist(p, (i + 1 : U32), s, name, mf.u32(mf.value(s, t, "rw")) <> acc)

def mf.text(r: File & Result<&1, &1, U32 & String, String>) -> IO(String):
  (f, t) = r
  do IO<String>:
    File.close(f)
    IO.pass(String, t)

# Absolute offsets may exceed U32; narrow only after rebasing into a layer file.
def mf.relative(name: String, off: Nat, base: Nat, bytes: U32) -> IO(U32):
  +r = Nat.sub(off, base)
  do IO<U32>:
    mf.check(Nat.is_le(r, Nat.sub(Nat.add(4294967295n, 1n), U32.to_nat(bytes))), name ++ " layer exceeds U32 addressing")
    return U32.from_nat(r)

def mf.offs(+s: String, +q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, i: U32) -> IO(Offs):
  +p = "blk." ++ U32.show(i) ++ "."
  do IO<Offs>:
    +an : Nat <- mf.off_nat(s, p ++ "attn_norm.weight")
    +q : Nat <- mf.off_nat(s, p ++ "attn_q.weight")
    +k : Nat <- mf.off_nat(s, p ++ "attn_k.weight")
    +v : Nat <- mf.off_nat(s, p ++ "attn_v.weight")
    +o : Nat <- mf.off_nat(s, p ++ "attn_output.weight")
    +fn : Nat <- mf.off_nat(s, p ++ "ffn_norm.weight")
    +g : Nat <- mf.off_nat(s, p ++ "ffn_gate.weight")
    +u : Nat <- mf.off_nat(s, p ++ "ffn_up.weight")
    +dn : Nat <- mf.off_nat(s, p ++ "ffn_down.weight")
    +b : Nat = Nat.min(Nat.min(Nat.min(an, q), Nat.min(k, v)),
      Nat.min(Nat.min(o, fn), Nat.min(g, Nat.min(u, dn))))
    ar : U32 <- mf.relative(p ++ "attn_norm", an, b, 8192)
    qr : U32 <- mf.relative(p ++ "attn_q", q, b, mb(q4k, q8, hf, x4, 16777216))
    kr : U32 <- mf.relative(p ++ "attn_k", k, b, mb(q4k, q8, hf, x4, 2097152))
    vr : U32 <- mf.relative(p ++ "attn_v", v, b, mb(q4k, q8, hf, x4, 2097152))
    owr : U32 <- mf.relative(p ++ "attn_output", o, b, mb(q4k, q8, hf, x4, 16777216))
    fr : U32 <- mf.relative(p ++ "ffn_norm", fn, b, 8192)
    gr : U32 <- mf.relative(p ++ "ffn_gate", g, b, mb(q4k, q8, hf, x4, 50331648))
    ur : U32 <- mf.relative(p ++ "ffn_up", u, b, mb(q4k, q8, hf, x4, 50331648))
    dr : U32 <- mf.relative(p ++ "ffn_down", dn, b, mb(q4k, q8, hf, x4, 50331648))
    return Offs{ar, qr, kr, vr, owr, fr, gr, ur, dr}

def mf.same(a: Offs, b: Offs) -> Bool:
  Offs{an, q, k, v, o, fn, g, u, dn} = a
  Offs{bn, bq, bk, bv, bo, bf, bg, bu, bd} = b
  Bool.and(U32.is_eq(an, bn), Bool.and(U32.is_eq(q, bq), Bool.and(U32.is_eq(k, bk),
    Bool.and(U32.is_eq(v, bv), Bool.and(U32.is_eq(o, bo), Bool.and(U32.is_eq(fn, bf),
      Bool.and(U32.is_eq(g, bg), Bool.and(U32.is_eq(u, bu), U32.is_eq(dn, bd)))))))))

def mf.tensors(n: Nat, +q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +i: U32, +s: String, +size: Nat, +layout: Offs) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      +b = "blk." ++ U32.show(i) ++ "."
      do IO<Unit>:
        mf.tensor(s, b ++ "attn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "attn_q.weight", "[2048,2048]", mb(q4k, q8, hf, x4, 16777216), size)
        mf.tensor(s, b ++ "attn_k.weight", "[256,2048]", mb(q4k, q8, hf, x4, 2097152), size)
        mf.tensor(s, b ++ "attn_v.weight", "[256,2048]", mb(q4k, q8, hf, x4, 2097152), size)
        mf.tensor(s, b ++ "attn_output.weight", "[2048,2048]", mb(q4k, q8, hf, x4, 16777216), size)
        mf.tensor(s, b ++ "ffn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "ffn_gate.weight", "[6144,2048]", mb(q4k, q8, hf, x4, 50331648), size)
        mf.tensor(s, b ++ "ffn_up.weight", "[6144,2048]", mb(q4k, q8, hf, x4, 50331648), size)
        mf.tensor(s, b ++ "ffn_down.weight", "[2048,6144]", mb(q4k, q8, hf, x4, 50331648), size)
        offsets : Offs <- mf.offs(s, q4k, q8, hf, x4, i)
        mf.check(mf.same(offsets, layout), b ++ "relative layout differs from blk.0")
        mf.tensors(p, q4k, q8, hf, x4, (i + 1 : U32), s, size, layout)

# a Q4_K_M mixture: attn_v and ffn_down carry their own row word and may be
# Q4_K or Q6_K; the fixed-slot pack keeps every layer's relative layout equal
def mf.tensors.k6(n: Nat, +i: U32, +s: String, +size: Nat, +layout: Offs) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      +b = "blk." ++ U32.show(i) ++ "."
      do IO<Unit>:
        mf.tensor(s, b ++ "attn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "attn_q.weight", "[2048,2048]", mb(True{}, False{}, False{}, False{}, 16777216), size)
        mf.tensor(s, b ++ "attn_k.weight", "[256,2048]", mb(True{}, False{}, False{}, False{}, 2097152), size)
        mf.tensor.rw(s, b ++ "attn_v.weight", "[256,2048]", 256, size)
        mf.tensor(s, b ++ "attn_output.weight", "[2048,2048]", mb(True{}, False{}, False{}, False{}, 16777216), size)
        mf.tensor(s, b ++ "ffn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "ffn_gate.weight", "[6144,2048]", mb(True{}, False{}, False{}, False{}, 50331648), size)
        mf.tensor(s, b ++ "ffn_up.weight", "[6144,2048]", mb(True{}, False{}, False{}, False{}, 50331648), size)
        mf.tensor.rw(s, b ++ "ffn_down.weight", "[2048,6144]", 2048, size)
        offsets : Offs <- mf.offs(s, True{}, False{}, False{}, False{}, i)
        mf.check(mf.same(offsets, layout), b ++ "relative layout differs from blk.0")
        mf.tensors.k6(p, (i + 1 : U32), s, size, layout)

def mf.tensors.any(+k6: Bool, +q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +s: String, +size: Nat, +layout: Offs) -> IO(Unit):
  match k6:
    case True{}:
      mf.tensors.k6(42n, 0, s, size, layout)
    case False{}:
      mf.tensors(42n, q4k, q8, hf, x4, 0, s, size, layout)

def mf.model(+dir: String, +ly: U32) -> IO(Model):
  do IO<Model>:
    f : File <- IO.try(File, File.open(dir ++ "/manifest.json", "r"))
    r : File & Result<&1, &1, U32 & String, String> <- File.read(f, 16777216)
    +s : String <- mf.text(r)
    +q8 : Bool = String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q8/v0\"")
    +hf : Bool = String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-f16/v0\"")
    +k6 : Bool = String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q4km/v0\"")
    +q4k : Bool = Bool.or(String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q4k/v0\""), k6)
    +x4 : Bool = String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-iq4xs/v0\"")
    mf.validate(s, q4k, q8, hf, k6, x4, ly)
    +nl : U32 <- mf.need("n_layers", mf.num(s, "config", "n_layers"))
    +emb : U32 <- mf.off(s, "token_embd.weight")
    +onorm : U32 <- mf.off(s, "output_norm.weight")
    +out : U32 <- mf.off(s, "output.weight")
    mf.check(Bool.and((emb <= 3225419776 : U32), Bool.and((out <= 3225419776 : U32),
      (onorm <= 4294959104 : U32))), "head tensor offset exceeds U32 addressing")
    +size : Nat <- mf.need_nat("pack_bytes", Nat.read(mf.scalar(s, "pack_bytes")))
    mf.check(String.eq(mf.scalar(s, "alignment"), "64"), "alignment")
    +hb : U32 = mb(q4k, q8, hf, x4, 1069547520)
    # Q4_K_M: output.weight is Q6_K (hbo); token_embd stays Q4_K (hb)
    +hbo : U32 = Bool.pick(U32, k6, mb6(1069547520), hb)
    +lb : U32 = Bool.pick(U32, x4, mf.u32(mf.scalar(s, "layer_bytes")),
      Bool.pick(U32, k6, mf.u32(mf.scalar(s, "layer_bytes")),
        Bool.pick(U32, q4k, 26558464, Bool.pick(U32, hf, 94388224, Bool.pick(U32, q8, 50151424, 188760064)))))
    mf.tensor(s, "token_embd.weight", "[130560,2048]", hb, size)
    mf.tensor(s, "output.weight", "[130560,2048]", hbo, size)
    mf.tensor(s, "output_norm.weight", "[2048]", 8192, size)
    +lay : Offs <- mf.offs(s, q4k, q8, hf, x4, 0)
    mf.tensors.any(k6, q4k, q8, hf, x4, s, size, lay)
    +rvs : List<&2, U32> = mf.rwlist(42n, 0, s, "attn_v.weight", Nil{})
    +rds : List<&2, U32> = mf.rwlist(42n, 0, s, "ffn_down.weight", Nil{})
    +n : Nat = U32.to_nat(Bool.pick(U32, U32.is_eq(ly, 0), nl, ly))
    t0 : Nat <- IO.now()
    ws : List<&2, Bytes()> <- ws.layers(n, dir, lb, 0)
    ow : Bytes() <- ws.load(dir ++ "/pack.bin", out, hbo)
    eb : Bytes() <- ws.load(dir ++ "/pack.bin", emb, hb)
    nb : Bytes() <- ws.load(dir ++ "/pack.bin", onorm, 8192)
    t1 : Nat <- IO.now()
    IO.print_err("resident " ++ Nat.show(Nat.add(Nat.mul(n, U32.to_nat(lb)), Nat.add(Nat.add(U32.to_nat(hb), U32.to_nat(hbo)), 8192n)))
      ++ " bytes in " ++ Nat.show(Nat.sub(t1, t0)) ++ " ms" ++ Bool.pick(String, k6, " (q4_k_m)", Bool.pick(String, q4k, " (q4_k)", Bool.pick(String, hf, " (f16)", Bool.pick(String, q8, " (q8_0)", Bool.pick(String, x4, " (iq4_xs)", " (f32)"))))))
    return Model{n, lay, ws, ow, eb, nb, q8, hf, q4k, k6, x4, rvs, rds}

# One layer
# ---------
# x -> x + Wo attn(rope(Wq h), rope(Wk h), Wv h), h = rms(x) -> x + Wd (silu(Wg h) * Wu h),
# h = rms(x): the op order and every conversion of the IO engine, run inside one
# bang; q/k/v and gate/up are forked side by side. Norms come from the layer's
# resident bytes.

# every stage is a flat Array<F32> written by fork-trees (kernels/flat.bend),
# each value computed as ops.bend computes it
@unsafe
def lay(+q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +rv: U32, +rd: U32, +f: Nat, +wt: Bytes(), +o: Offs, +dT: Nat, +nT: U32, +pos: U32, +x: Array<F32>,
  kv: Array<F32> & Array<F32>) -> Array<F32> & Array<F32> & Array<F32>:
  Offs{an, wq, wk, wv, wo, fn, g, u, dn} = o
  (kc, vc) = kv
  +at = (pos * 256 : U32)
  +pf = U32.to_f32(pos)
  +c = (U32.from_nat(dT) .|. (pos << 5n) : U32)
  # the row bytes pick the format (F.mv): F32, Q8_0, F16, Q4_K or Q6_K; rv and
  # rd are attn_v's and ffn_down's own words (a Q4_K_M mixture; equal to rb and
  # the 24576 word for every uniform pack)
  +rb = rw(q4k, q8, hf, x4, 8192)
  qs ks vs = F.llb.rope_warp!(f, wt, x, Array.new(F32, 11n, 0.0), wq, 2048, 9n, rb, an, 0, pos) F.llb.rope_warp!(f, wt, x, kc, wk, 256, 9n, rb, an, at, pos) F.llb.put_warp!(f, wt, x, vc, wv, 256, 9n, rv, an, at)
  +q = qs
  +kc2 = ks
  +vc2 = vs
  +x2 = F.llb.attn_warp!(f, wt, q, kc2, vc2, c, Array.new(F32, 11n, 0.0), wo, x, rb)
  (F.llb.ffn_warp!(f, wt, x2, Array.new(F32, 11n, 0.0), dn, 2048, 11n, rd, g, u, fn, rb), (kc2, vc2))

def fwd.cons(kv: Array<F32> & Array<F32>, r: Array<F32> & List<(Array<F32> & Array<F32>)>) ->
  Array<F32> & List<(Array<F32> & Array<F32>)>:
  (x, cs) = r
  (x, kv <> cs)

def fwd.with(k: Array<F32> -> Array<F32> & Array<F32> -> Array<F32> & List<(Array<F32> & Array<F32>)>,
  r: Array<F32> & Array<F32> & Array<F32>) -> Array<F32> & List<(Array<F32> & Array<F32>)>:
  (x, kv) = r
  k(x, kv)

# skip s caches, then run the next k layers (i counts from the list's head),
# each with its own (K, V) cache; the list comes back whole and in order
@unsafe
def fwd(s: Nat, k: Nat, +i: U32, +m: Model, +f: Nat, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  cs: List<(Array<F32> & Array<F32>)>) -> Array<F32> & List<(Array<F32> & Array<F32>)>:
  match s k cs:
    case 1n+p _ Con{kv, t}:
      fwd.cons(kv, fwd(p, k, (i + 1 : U32), m, f, dT, nT, pos, x, t))
    case 0n 1n+p Con{kv, t}:
      fwd.with(x2 => kv2 => fwd.cons(kv2, fwd(0n, p, (i + 1 : U32), m, f, dT, nT, pos, x2, t)),
        lay(m.q4k(m), m.q8(m), m.hf(m), m.x4(m), m.rv(m, i), m.rd(m, i), f, m.w(m, i), m.offs(m), dT, nT, pos, x, kv))
    case _ _ _:
      (x, cs)

def layers.fin(+t0: Nat, r: Array<F32> & List<(Array<F32> & Array<F32>)>) ->
  IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  do IO<Array<F32> & List<(Array<F32> & Array<F32>)>>:
    t1 : Nat <- IO.now()
    IO.print_err("op layer " ++ Nat.show(Nat.sub(t1, t0)))
    return r

def layers.with(k: Array<F32> -> List<(Array<F32> & Array<F32>)> ->
  IO(Array<F32> & List<(Array<F32> & Array<F32>)>),
  r: Array<F32> & List<(Array<F32> & Array<F32>)>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  (x, cs) = r
  k(x, cs)

# k layers from s in one bang (entered after an IO step: the solo loop, the
# GPU's entry)
def chunk(+s: Nat, +k: Nat, +m: Model, +f: Nat, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  do IO<Array<F32> & List<(Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    layers.fin(t0, fwd!(s, k, 0, m, f, dT, nT, pos, x, cs))

# c bangs of up to b layers each, from layer s (n remain)
def layers.go(c: Nat, +n: Nat, +s: Nat, +b: Nat, +m: Model, +f: Nat, +dT: Nat, +nT: U32, +pos: U32,
  x: Array<F32>, cs: List<(Array<F32> & Array<F32>)>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  match c:
    case 1n+p:
      +k = Nat.min(n, b)
      IO.bind(Array<F32> & List<(Array<F32> & Array<F32>)>, Array<F32> & List<(Array<F32> & Array<F32>)>,
        chunk(s, k, m, f, dT, nT, pos, x, cs),
        layers.with(x2 => cs2 => layers.go(p, Nat.sub(n, k), Nat.add(s, k), b, m, f, dT, nT, pos, x2, cs2)))
    case 0n:
      IO.pure(Array<F32> & List<(Array<F32> & Array<F32>)>, (x, cs))

# the n layers, b per bang
def layers(+n: Nat, +b: Nat, +m: Model, +f: Nat, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  layers.go(Nat.div(Nat.sub(Nat.add(n, b), 1n), b), n, 0n, b, m, f, dT, nT, pos, x, cs)

type Tops is Data:
  Tops{v: F32, i: U32, w: F32, j: U32}

def step.ranked(v: F32, i: U32, second: F32 & U32) -> Tops:
  (w, j) = second
  Tops{v, i, w, j}

# Padding above 130560 is -inf. The second pick (slot i read as -inf) is
# diagnostics only; the generated id remains the first-index argmax.
@unsafe
def step.second(first: F32 & U32, +z: Array<F32>) -> Tops:
  (v, +i) = first
  step.ranked(v, i, F.argmax(10n, z, i, 131072))

def step.diag(+pos: U32, tops: Tops) -> IO(U32):
  Tops{+v, +i, +w, j} = tops
  do IO<U32>:
    mf.check(Bool.and((U32.and(F32.bits(v), 2139095040) < 2139095040 : U32),
      (U32.and(F32.bits(w), 2139095040) < 2139095040 : U32)), "nonfinite logits at pos " ++ U32.show(pos))
    IO.print_err("logits " ++ U32.show(pos) ++ " " ++ U32.show(i) ++ " " ++ U32.show(F32.bits(v))
      ++ " " ++ U32.show(j) ++ " " ++ U32.show(F32.bits(w)))
    return i

# rms_final, the output gemv and the top two, on the device
@unsafe
def head(+orw: U32, +f: Nat, +ow: Bytes(), +nb: Bytes(), +x: Array<F32>) -> Tops:
  +z = F.llb.gemv_warp!(f, ow, F.rms(nb, 0, x), Array.new(F32, 17n, K.neg_inf()), 0, 130560, 9n, orw)
  step.second(F.argmax(10n, z, 4294967295, 131072), z)

@unsafe
def embed.q8(+q8: Bool, +eb: Bytes(), +tok: U32) -> Array<F32>:
  match q8:
    case True{}:
      F.q8.load(6n, eb, (tok * 2176 : U32), 11n)
    case False{}:
      F.load(6n, eb, (tok * 8192 : U32), 11n)

# the F32 / Q8_0 / F16 / Q4_K embedding row (the pre-IQ4_XS formats)
@unsafe
def embed.fmt(+q4k: Bool, +q8: Bool, hf: Bool, +eb: Bytes(), +tok: U32) -> Array<F32>:
  match q4k:
    case True{}:
      F.q4k.load(6n, eb, (tok * 1152 : U32), 11n)
    case False{}:
      match hf:
        case True{}:
          F.h16.load(6n, eb, (tok * 4096 : U32), 11n)
        case False{}:
          embed.q8(q8, eb, tok)

@unsafe
def embed(+q4k: Bool, +q8: Bool, +hf: Bool, +x4: Bool, +eb: Bytes(), +tok: U32) -> Array<F32>:
  match x4:
    case True{}:
      F.iq4xs.load(6n, eb, (tok * 1088 : U32), 11n)
    case False{}:
      embed.fmt(q4k, q8, hf, eb, tok)

def step.fin(+pos: U32, +t0: Nat, +t1: Nat, cs: List<(Array<F32> & Array<F32>)>, tops: Tops) ->
  IO(U32 & List<(Array<F32> & Array<F32>)>):
  do IO<U32 & List<(Array<F32> & Array<F32>)>>:
    +t2 : Nat <- IO.now()
    IO.print_err("op head " ++ Nat.show(Nat.sub(t2, t1)))
    tok : U32 <- step.diag(pos, tops)
    IO.print_err("step " ++ U32.show(pos) ++ " " ++ Nat.show(Nat.sub(t2, t0)))
    return (tok, cs)

def step.out(+m: Model, +f: Nat, +pos: U32, +t0: Nat, r: Array<F32> & List<(Array<F32> & Array<F32>)>) ->
  IO(U32 & List<(Array<F32> & Array<F32>)>):
  (x, cs) = r
  do IO<U32 & List<(Array<F32> & Array<F32>)>>:
    t1 : Nat <- IO.now()
    step.fin(pos, t0, t1, cs, head!(m.orw(m), f, m.ow(m), m.nb(m), x))

# the least d <= f with 2^d >= need
def lg(f: Nat, +d: Nat, +need: U32) -> Nat:
  match f:
    case 0n:
      d
    case 1n+p:
      lg(p, Bool.pick(Nat, (need <= (1 << d : U32) : U32), d, 1n+d), need)

# one forward: token tok at position pos -> (argmax, caches). kv (LLB_KV_PREFIX=1):
# attention reads the least 2^d >= pos + 1 slots, not all 2^dT. The slots past pos
# add -inf to the max and +0 to every sum, so the capacity tree's value is its
# leftmost 2^d subtree's: the same bits in O(pos) instead of O(capacity).
def step(+m: Model, +f: Nat, +b: Nat, +kv: Bool, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(U32 & List<(Array<F32> & Array<F32>)>):
  +d = Bool.pick(Nat, kv, lg(17n, 0n, (pos + 1 : U32)), dT)
  do IO<U32 & List<(Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    x : Array<F32> <- I.io.time(Array<F32>, "embed", _ => embed(m.q4k(m), m.q8(m), m.hf(m), m.x4(m), m.eb(m), tok))
    r : Array<F32> & List<(Array<F32> & Array<F32>)> <- layers(m.nl(m), b, m, f, d, nT, pos, x, cs)
    step.out(m, f, pos, t0, r)

def run.with(k: U32 -> List<(Array<F32> & Array<F32>)> -> IO(Unit),
  r: U32 & List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  (t, cs) = r
  k(t, cs)

# print tok; while more are wanted, feed it at pos for the next
def gen(n: Nat, +m: Model, +f: Nat, +b: Nat, +kv: Bool, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  match n:
    case 0n:
      IO.print(U32.show(tok))
    case 1n+p:
      do IO<Unit>:
        IO.print(U32.show(tok))
        IO.bind(U32 & List<(Array<F32> & Array<F32>)>, Unit, step(m, f, b, kv, dT, nT, pos, tok, cs),
          run.with(t => cs2 => gen(p, m, f, b, kv, dT, nT, (pos + 1 : U32), t, cs2)))

# feed the prompt from pos; the argmax after its last token is the first generated
def prompt(ts: List<&2, U32>, n: Nat, +m: Model, +f: Nat, +b: Nat, +kv: Bool, +dT: Nat, +nT: U32, +pos: U32, +prev: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  match ts:
    case Con{t, rest}:
      IO.bind(U32 & List<(Array<F32> & Array<F32>)>, Unit, step(m, f, b, kv, dT, nT, pos, t, cs),
        run.with(nt => cs2 => prompt(rest, n, m, f, b, kv, dT, nT, (pos + 1 : U32), nt, cs2)))
    case Nil{}:
      gen(n, m, f, b, kv, dT, nT, pos, prev, cs)

# Setup
# -----

# ponytail: fixed N reserves the whole run rounded to a power of two; an
# open-ended decoder would need capacity growth. C stores each K/V array in
# one contiguous F32 arena, updated in place and retained across tokens.
def caches(n: Nat, +d: Nat) -> List<(Array<F32> & Array<F32>)>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      (Array.new(F32, d, 0.0), Array.new(F32, d, 0.0)) <> caches(p, d)

def num.of(m: Maybe<&2, U32>) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      4294967295

# (prompt ids, (steps, (layers, (fork, lpb)))); mode 1/2/3/4: the previous arg was --steps/--layers/--fork/--lpb
def cfg(xs: List<String>, +ts: List<&2, U32>, +st: U32, +ly: U32, +fk: U32, +lb: U32, +mode: U32, +ok: Bool) ->
  List<&2, U32> & U32 & U32 & U32 & U32:
  match xs:
    case Con{a0, t}:
      +a = a0
      +fs = String.eq(a, "--steps")
      +fl = String.eq(a, "--layers")
      +ff = String.eq(a, "--fork")
      +fb = String.eq(a, "--lpb")
      +flag = Bool.or(Bool.or(fs, fb), Bool.or(fl, ff))
      +v = num.of(U32.read(a))
      cfg(t,
        Bool.pick(List<&2, U32>, Bool.and(U32.is_eq(mode, 0), Bool.not(flag)), v <> ts, ts),
        Bool.pick(U32, U32.is_eq(mode, 1), v, st), Bool.pick(U32, U32.is_eq(mode, 2), v, ly),
        Bool.pick(U32, U32.is_eq(mode, 3), v, fk), Bool.pick(U32, U32.is_eq(mode, 4), v, lb),
        Bool.pick(U32, fs, 1, Bool.pick(U32, fl, 2, Bool.pick(U32, ff, 3, Bool.pick(U32, fb, 4, 0)))),
        Bool.and(ok, Bool.pick(Bool, flag, U32.is_eq(mode, 0), Bool.not(U32.is_eq(v, 4294967295)))))
    case Nil{}:
      (List.reverse(&2, U32, ts), (Bool.pick(U32, Bool.and(ok, U32.is_eq(mode, 0)), st, 0), (ly, (fk, lb))))

# every id below the vocab (a bad number parses as 2^32 - 1)
def ok_ids.go(ts: List<&2, U32>, ok: Bool) -> Bool:
  match ts:
    case Con{t, rest}:
      ok_ids.go(rest, Bool.and((t < 130560 : U32), ok))
    case Nil{}:
      ok

def ok_ids(ts: List<&2, U32>) -> Bool:
  ok_ids.go(ts, True{})

# Base.List.length uses non-tail recursion in JS; count long prompts in a loop.
def count_ids(ts: List<&2, U32>, n: Nat) -> Nat:
  match ts:
    case Con{t, rest}:
      count_ids(rest, 1n+n)
    case Nil{}:
      n

def usage() -> String:
  "usage: engine_gpu <packdir> <tok>... [--steps N] [--layers L] [--fork F] [--lpb K]  (N >= 1, ids < 130560, prompt+N-1 <= 131072, L=0..42, F=0..20, K=1..42)"

def kv.on(e: Result<&1, &1, U32 & String, String>) -> Bool:
  match e:
    case Done{v}:
      String.eq(v, "1")
    case Fail{_}:
      False{}

def run.go(ok: Bool, +dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +fk: U32, +lb: U32) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +m : Model <- mf.model(dir, ly)
        +dT : Nat = lg(17n, 0n, (U32.from_nat(count_ids(ts, 0n)) + st - 1 : U32))
        e : Result<&1, &1, U32 & String, String> <- IO.get_env("LLB_KV_PREFIX")
        +kv : Bool = kv.on(e)
        IO.print_err("kv_prefix " ++ Bool.pick(String, kv, "1", "0"))
        prompt(ts, U32.to_nat((st - 1 : U32)), m, U32.to_nat(fk), U32.to_nat(lb), kv, dT, (1 << dT : U32), 0, 0,
          caches(m.nl(m), Nat.add(dT, 8n)))
        t1 : Nat <- IO.now()
        IO.print("wall_ms " ++ Nat.show(Nat.sub(t1, t0)))
    case False{}:
      IO.die(Unit, 2, usage())

def run.chk(dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +fk: U32, +lb: U32) -> IO(Unit):
  +nt = U32.from_nat(count_ids(ts, 0n))
  run.go(Bool.and(Bool.and(ok_ids(ts), (nt >= 1 : U32)),
    Bool.and(Bool.and((st >= 1 : U32), (st <= 131072 : U32)),
      Bool.and(Bool.and(Bool.and((ly <= 42 : U32), (fk <= 20 : U32)), Bool.and((lb >= 1 : U32), (lb <= 42 : U32))),
        (nt <= (131073 - st : U32) : U32)))), dir, ts, st, ly, fk, lb)

def run.sl(dir: String, ts: List<&2, U32>, sl: U32 & U32 & U32 & U32) -> IO(Unit):
  (st, lf) = sl
  (ly, fb) = lf
  (fk, lb) = fb
  run.chk(dir, ts, st, ly, fk, lb)

def run(dir: String, c: List<&2, U32> & U32 & U32 & U32 & U32) -> IO(Unit):
  (ts, sl) = c
  run.sl(dir, ts, sl)

def main.go(args: List<String>) -> IO(Unit):
  match args:
    case Con{dir, rest}:
      run(dir, cfg(rest, Nil{}, 8, 0, 14, 1, 0, True{}))
    case Nil{}:
      IO.die(Unit, 2, usage())

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
