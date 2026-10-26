# engine <packdir> <tok>... [--steps N] [--layers L]
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
import Base
import ./kernels/ops.bend as K
import ./kernels/io.bend as I
import ./kernels/order.bend as O

# a layer's tensors: byte offsets within its layer file
type Offs is Data:
  Offs{an: U32, q: U32, k: U32, v: U32, o: U32, fn: U32, g: U32, u: U32, dn: U32}

# pack dir, layers to run, head offsets in pack.bin, a layer's offsets
type Model is Data:
  Model{dir: String, nl: Nat, emb: U32, onorm: U32, out: U32, lay: Offs}

def m.pack(m: Model) -> String:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      dir ++ "/pack.bin"

def m.layer(m: Model, i: U32) -> String:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      dir ++ "/layers/blk." ++ U32.show(i) ++ ".bin"

def m.offs(m: Model) -> Offs:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      lay

def m.nl(m: Model) -> Nat:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      nl

def m.emb(m: Model) -> U32:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      emb

def m.onorm(m: Model) -> U32:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      onorm

def m.out(m: Model) -> U32:
  match m:
    case Model{dir, nl, emb, onorm, out, lay}:
      out

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

def mf.validate(+s: String, ly: U32) -> IO(Unit):
  do IO<Unit>:
    mf.check(String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-f32/v0\""), "format")
    mf.check(String.eq(mf.scalar(s, "dtype"), "\"f32le\""), "dtype")
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

def mf.offs(+s: String, i: U32) -> IO(Offs):
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
    qr : U32 <- mf.relative(p ++ "attn_q", q, b, 16777216)
    kr : U32 <- mf.relative(p ++ "attn_k", k, b, 2097152)
    vr : U32 <- mf.relative(p ++ "attn_v", v, b, 2097152)
    owr : U32 <- mf.relative(p ++ "attn_output", o, b, 16777216)
    fr : U32 <- mf.relative(p ++ "ffn_norm", fn, b, 8192)
    gr : U32 <- mf.relative(p ++ "ffn_gate", g, b, 50331648)
    ur : U32 <- mf.relative(p ++ "ffn_up", u, b, 50331648)
    dr : U32 <- mf.relative(p ++ "ffn_down", dn, b, 50331648)
    return Offs{ar, qr, kr, vr, owr, fr, gr, ur, dr}

def mf.same(a: Offs, b: Offs) -> Bool:
  Offs{an, q, k, v, o, fn, g, u, dn} = a
  Offs{bn, bq, bk, bv, bo, bf, bg, bu, bd} = b
  Bool.and(U32.is_eq(an, bn), Bool.and(U32.is_eq(q, bq), Bool.and(U32.is_eq(k, bk),
    Bool.and(U32.is_eq(v, bv), Bool.and(U32.is_eq(o, bo), Bool.and(U32.is_eq(fn, bf),
      Bool.and(U32.is_eq(g, bg), Bool.and(U32.is_eq(u, bu), U32.is_eq(dn, bd)))))))))

def mf.tensors(n: Nat, +i: U32, +s: String, +size: Nat, +layout: Offs) -> IO(Unit):
  match n:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      +b = "blk." ++ U32.show(i) ++ "."
      do IO<Unit>:
        mf.tensor(s, b ++ "attn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "attn_q.weight", "[2048,2048]", 16777216, size)
        mf.tensor(s, b ++ "attn_k.weight", "[256,2048]", 2097152, size)
        mf.tensor(s, b ++ "attn_v.weight", "[256,2048]", 2097152, size)
        mf.tensor(s, b ++ "attn_output.weight", "[2048,2048]", 16777216, size)
        mf.tensor(s, b ++ "ffn_norm.weight", "[2048]", 8192, size)
        mf.tensor(s, b ++ "ffn_gate.weight", "[6144,2048]", 50331648, size)
        mf.tensor(s, b ++ "ffn_up.weight", "[6144,2048]", 50331648, size)
        mf.tensor(s, b ++ "ffn_down.weight", "[2048,6144]", 50331648, size)
        offsets : Offs <- mf.offs(s, i)
        mf.check(mf.same(offsets, layout), b ++ "relative layout differs from blk.0")
        mf.tensors(p, (i + 1 : U32), s, size, layout)

def mf.model(+dir: String, +ly: U32) -> IO(Model):
  do IO<Model>:
    f : File <- IO.try(File, File.open(dir ++ "/manifest.json", "r"))
    r : File & Result<&1, &1, U32 & String, String> <- File.read(f, 16777216)
    +s : String <- mf.text(r)
    mf.validate(s, ly)
    +nl : U32 <- mf.need("n_layers", mf.num(s, "config", "n_layers"))
    +emb : U32 <- mf.off(s, "token_embd.weight")
    +onorm : U32 <- mf.off(s, "output_norm.weight")
    +out : U32 <- mf.off(s, "output.weight")
    mf.check(Bool.and((emb <= 3225419776 : U32), Bool.and((out <= 3225419776 : U32),
      (onorm <= 4294959104 : U32))), "head tensor offset exceeds U32 addressing")
    +size : Nat <- mf.need_nat("pack_bytes", Nat.read(mf.scalar(s, "pack_bytes")))
    mf.check(String.eq(mf.scalar(s, "alignment"), "64"), "alignment")
    mf.tensor(s, "token_embd.weight", "[130560,2048]", 1069547520, size)
    mf.tensor(s, "output.weight", "[130560,2048]", 1069547520, size)
    mf.tensor(s, "output_norm.weight", "[2048]", 8192, size)
    +lay : Offs <- mf.offs(s, 0)
    mf.tensors(42n, 0, s, size, lay)
    return Model{dir, U32.to_nat(Bool.pick(U32, U32.is_eq(ly, 0), nl, ly)), emb, onorm, out, lay}

# One layer
# ---------
# x -> x + Wo attn(rope(Wq h), rope(Wk h), Wv h), h = rms(x) -> x + Wd (silu(Wg h) * Wu h),
# h = rms(x). Each gemv hands its input back as the first half of its result.

# (x, rms_norm(x, w))
def rmsc(w: Array<F32>, xx: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (xa, xb) = xx
  (xa, K.rms_norm(xb, w, 2048.0))

def l.fin(x: Array<F32>, kv: Array<F32> & Array<F32>, r: Array<F32> & List<Array<F32>>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  (a, ys) = r
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    y : Array<F32> <- I.io.time(Array<F32>, "residual_ffn", _ => K.vadd(x, K.from_rev(11n, 0n, 0.0, ys)))
    return (y, kv)

def l.down(+p: String, +o: Offs, x: Array<F32>, gs: List<Array<F32>>,
  ur: Array<F32> & List<Array<F32>>, kv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (h, us) = ur
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    a : Array<F32> <- I.io.time(Array<F32>, "silu_up", _ =>
      K.vmul(K.silu(K.from_rev(13n, 2048n, 0.0, gs)), K.from_rev(13n, 2048n, 0.0, us)))
    dr : Array<F32> & List<Array<F32>> <- I.io.gemv("ffn_down", p, dn, 2048n, 13n, 24576, a)
    l.fin(x, kv, dr)

def l.up(+p: String, +o: Offs, x: Array<F32>, gr: Array<F32> & List<Array<F32>>,
  kv: Array<F32> & Array<F32>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (h, gs) = gr
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    ur : Array<F32> & List<Array<F32>> <- I.io.gemv("ffn_up", p, u, 6144n, 11n, 8192, h)
    l.down(p, o, x, gs, ur, kv)

def l.gate(+p: String, +o: Offs, xh: Array<F32> & Array<F32>, kv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (x, h) = xh
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    gr : Array<F32> & List<Array<F32>> <- I.io.gemv("ffn_gate", p, g, 6144n, 11n, 8192, h)
    l.up(p, o, x, gr, kv)

def l.ffn(+p: String, +o: Offs, x: Array<F32>, kv: Array<F32> & Array<F32>,
  r: Array<F32> & List<Array<F32>>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (a, ys) = r
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    y : Array<F32> <- I.io.time(Array<F32>, "residual_attn", _ => K.vadd(x, K.from_rev(11n, 0n, 0.0, ys)))
    w : Array<F32> <- I.io.load_exact(p, fn, 11n)
    xh : Array<F32> & Array<F32> <- I.io.time(Array<F32> & Array<F32>, "rms_ffn", _ => rmsc(w, Array.clone(F32, y)))
    l.gate(p, o, xh, kv)

# attention over copies of the updated caches; the originals go on to the next token
def l.att(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>, q: Array<F32>,
  kk: Array<F32> & Array<F32>, vv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, wq, wk, wv, wo, fn, g, u, dn} = o
  (kb, ka) = kk
  (vb, va) = vv
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    s : Array<F32> <- I.io.time(Array<F32>, "scores", _ => K.scores(4n, 16, 0, dT, q, ka))
    ps : Array<F32> <- I.io.time(Array<F32>, "softmax", _ => K.probs(4n, nT, pos, s))
    a : Array<F32> <- I.io.time(Array<F32>, "weighted_sum", _ => K.weighted(4n, 16, 0, dT, ps, va))
    r : Array<F32> & List<Array<F32>> <- I.io.gemv("attn_output", p, wo, 2048n, 11n, 8192, a)
    l.ffn(p, o, x, (kb, vb), r)

def l.rope(+pos: U32, +n: U32, +d: Nat, ys: List<Array<F32>>) -> Array<F32>:
  K.rope(K.from_rev(d, 0n, 0.0, ys), n, 0, U32.to_f32(pos))

def l.cached(p: String, o: Offs, dT: Nat, nT: U32, pos: U32, x: Array<F32>, q: Array<F32>,
  kv: Array<F32> & Array<F32>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  (kc, vc) = kv
  l.att(p, o, dT, nT, pos, x, q, Array.clone(F32, kc), Array.clone(F32, vc))

# Array.set writes the existing native F32 arena; no cache tree reconstruction.
def kv.write(vs: List<F32>, c: Array<F32>, +i: U32) -> Array<F32>:
  match vs:
    case Nil{}:
      c
    case Con{v, t}:
      kv.write(t, Array.set(F32, c, i, v), (i + 1 : U32))

def l.store(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  qk: Array<F32> & Array<F32>, vs: List<Array<F32>>, kv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  (q, k) = qk
  (kc, vc) = kv
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    updated : Array<F32> & Array<F32> <- I.io.time(Array<F32> & Array<F32>, "kv_append", _ =>
      (kv.write(Array.to_list(~F32, k), kc, (pos * 256 : U32)),
        kv.write(Array.to_list(~F32, K.from_rev(8n, 0n, 0.0, vs)), vc, (pos * 256 : U32))))
    l.cached(p, o, dT, nT, pos, x, q, updated)

def l.attn(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  qs: List<Array<F32>>, ks: List<Array<F32>>, vr: Array<F32> & List<Array<F32>>,
  kv: Array<F32> & Array<F32>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  (h, vs) = vr
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    qk : Array<F32> & Array<F32> <- I.io.time(Array<F32> & Array<F32>, "rope", _ =>
      (l.rope(pos, 2048, 11n, qs), l.rope(pos, 256, 8n, ks)))
    l.store(p, o, dT, nT, pos, x, qk, vs, kv)

def l.v(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  qs: List<Array<F32>>, kr: Array<F32> & List<Array<F32>>, kv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (h, ks) = kr
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    vr : Array<F32> & List<Array<F32>> <- I.io.gemv("attn_v", p, v, 256n, 11n, 8192, h)
    l.attn(p, o, dT, nT, pos, x, qs, ks, vr, kv)

def l.k(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  qr: Array<F32> & List<Array<F32>>, kv: Array<F32> & Array<F32>) ->
  IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (h, qs) = qr
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    kr : Array<F32> & List<Array<F32>> <- I.io.gemv("attn_k", p, k, 256n, 11n, 8192, h)
    l.v(p, o, dT, nT, pos, x, qs, kr, kv)

def l.q(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, xh: Array<F32> & Array<F32>,
  kv: Array<F32> & Array<F32>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  (x, h) = xh
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    qr : Array<F32> & List<Array<F32>> <- I.io.gemv("attn_q", p, q, 2048n, 11n, 8192, h)
    l.k(p, o, dT, nT, pos, x, qr, kv)

# (x, (K cache, V cache)) after the layer in file p at position pos
def layer(+p: String, +o: Offs, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  kv: Array<F32> & Array<F32>) -> IO(Array<F32> & Array<F32> & Array<F32>):
  Offs{an, q, k, v, wo, fn, g, u, dn} = o
  do IO<Array<F32> & Array<F32> & Array<F32>>:
    w : Array<F32> <- I.io.load_exact(p, an, 11n)
    xh : Array<F32> & Array<F32> <- I.io.time(Array<F32> & Array<F32>, "rms_attn", _ => rmsc(w, Array.clone(F32, x)))
    l.q(p, o, dT, nT, pos, xh, kv)

# The forward
# -----------

def layers.with(k: Array<F32> -> Array<F32> & Array<F32> ->
  IO(Array<F32> & List<(Array<F32> & Array<F32>)>),
  r: Array<F32> & Array<F32> & Array<F32>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  (x, kv) = r
  k(x, kv)

def layers.push(kv: Array<F32> & Array<F32>, r: Array<F32> & List<(Array<F32> & Array<F32>)>) ->
  IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  (x, cs) = r
  IO.pure(Array<F32> & List<(Array<F32> & Array<F32>)>, (x, kv <> cs))

# layers i.. over x, each with its own (K, V) cache, returned updated in order
def layers(n: Nat, +i: U32, +m: Model, +dT: Nat, +nT: U32, +pos: U32, x: Array<F32>,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Array<F32> & List<(Array<F32> & Array<F32>)>):
  match n cs:
    case 1n+p Con{kv, t}:
      IO.bind(Array<F32> & Array<F32> & Array<F32>, Array<F32> & List<(Array<F32> & Array<F32>)>,
        layer(m.layer(m, i), m.offs(m), dT, nT, pos, x, kv),
        layers.with(x2 => kv2 =>
          IO.bind(Array<F32> & List<(Array<F32> & Array<F32>)>,
            Array<F32> & List<(Array<F32> & Array<F32>)>,
            layers(p, (i + 1 : U32), m, dT, nT, pos, x2, t), layers.push(kv2))))
    case _ _:
      IO.pure(Array<F32> & List<(Array<F32> & Array<F32>)>, (x, cs))

type Tops is Data:
  Tops{v: F32, i: U32, w: F32, j: U32}

def step.ranked(v: F32, i: U32, second: F32 & U32) -> Tops:
  (w, j) = second
  Tops{v, i, w, j}

def step.second(first: F32 & U32, z: Array<F32>) -> Tops:
  (v, +i) = first
  step.ranked(v, i, K.argmax(K.put(17n, z, 131072, 0, i, ALeaf{K.neg_inf()}), 131072, 0))

def step.top2(zz: Array<F32> & Array<F32>) -> Tops:
  (a, b) = zz
  step.second(K.argmax(a, 131072, 0), b)

# Padding above 130560 is -inf. The second pick is diagnostics only; the
# generated id remains the original kernel argmax with first-index ties.
def step.tops(r: Array<F32> & List<Array<F32>>) -> Tops:
  (h, zs) = r
  step.top2(Array.clone(F32, K.from_rev(17n, 512n, K.neg_inf(), zs)))

def step.diag(+pos: U32, tops: Tops) -> IO(U32):
  Tops{+v, +i, +w, j} = tops
  do IO<U32>:
    mf.check(Bool.and((U32.and(F32.bits(v), 2139095040) < 2139095040 : U32),
      (U32.and(F32.bits(w), 2139095040) < 2139095040 : U32)), "nonfinite logits at pos " ++ U32.show(pos))
    IO.print_err("logits " ++ U32.show(pos) ++ " " ++ U32.show(i) ++ " " ++ U32.show(F32.bits(v))
      ++ " " ++ U32.show(j) ++ " " ++ U32.show(F32.bits(w)))
    return i

def step.out(+m: Model, +pos: U32, t0: Nat, r: Array<F32> & List<(Array<F32> & Array<F32>)>) ->
  IO(U32 & List<(Array<F32> & Array<F32>)>):
  (x, cs) = r
  do IO<U32 & List<(Array<F32> & Array<F32>)>>:
    w : Array<F32> <- I.io.load_exact(m.pack(m), m.onorm(m), 11n)
    h : Array<F32> <- I.io.time(Array<F32>, "rms_final", _ => K.rms_norm(x, w, 2048.0))
    zr : Array<F32> & List<Array<F32>> <- I.io.gemv("output", m.pack(m), m.out(m), 130560n, 11n, 8192, h)
    tops : Tops <- I.io.time(Tops, "argmax", _ => step.tops(zr))
    tok : U32 <- step.diag(pos, tops)
    t1 : Nat <- IO.now()
    IO.print_err("step " ++ U32.show(pos) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return (tok, cs)

# one forward: token tok at position pos -> (argmax, caches)
def step(+m: Model, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(U32 & List<(Array<F32> & Array<F32>)>):
  do IO<U32 & List<(Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    x : Array<F32> <- I.io.time_io(Array<F32>, "embed",
      I.io.load_exact(m.pack(m), (m.emb(m) + (tok * 8192 : U32) : U32), 11n))
    r : Array<F32> & List<(Array<F32> & Array<F32>)> <- layers(m.nl(m), 0, m, dT, nT, pos, x, cs)
    step.out(m, pos, t0, r)

def run.with(k: U32 -> List<(Array<F32> & Array<F32>)> -> IO(Unit),
  r: U32 & List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  (t, cs) = r
  k(t, cs)

# print tok; while more are wanted, feed it at pos for the next
def gen(n: Nat, +m: Model, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  match n:
    case 0n:
      IO.print(U32.show(tok))
    case 1n+p:
      do IO<Unit>:
        IO.print(U32.show(tok))
        IO.bind(U32 & List<(Array<F32> & Array<F32>)>, Unit, step(m, dT, nT, pos, tok, cs),
          run.with(t => cs2 => gen(p, m, dT, nT, (pos + 1 : U32), t, cs2)))

# feed the prompt from pos; the argmax after its last token is the first generated
def prompt(ts: List<&2, U32>, n: Nat, +m: Model, +dT: Nat, +nT: U32, +pos: U32, +prev: U32,
  cs: List<(Array<F32> & Array<F32>)>) -> IO(Unit):
  match ts:
    case Con{t, rest}:
      IO.bind(U32 & List<(Array<F32> & Array<F32>)>, Unit, step(m, dT, nT, pos, t, cs),
        run.with(nt => cs2 => prompt(rest, n, m, dT, nT, (pos + 1 : U32), nt, cs2)))
    case Nil{}:
      gen(n, m, dT, nT, pos, prev, cs)

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

# the least d <= f with 2^d >= need
def lg(f: Nat, +d: Nat, +need: U32) -> Nat:
  match f:
    case 0n:
      d
    case 1n+p:
      lg(p, Bool.pick(Nat, (need <= (1 << d : U32) : U32), d, 1n+d), need)

def num.of(m: Maybe<&2, U32>) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      4294967295

# (prompt ids, (steps, layers)); mode 1/2: the previous arg was --steps/--layers
def cfg(xs: List<String>, +ts: List<&2, U32>, +st: U32, +ly: U32, +mode: U32, +ok: Bool) ->
  List<&2, U32> & U32 & U32:
  match xs:
    case Con{a0, t}:
      +a = a0
      +fs = String.eq(a, "--steps")
      +fl = String.eq(a, "--layers")
      +v = num.of(U32.read(a))
      cfg(t,
        Bool.pick(List<&2, U32>, Bool.and(U32.is_eq(mode, 0), Bool.not(Bool.or(fs, fl))), v <> ts, ts),
        Bool.pick(U32, U32.is_eq(mode, 1), v, st), Bool.pick(U32, U32.is_eq(mode, 2), v, ly),
        Bool.pick(U32, fs, 1, Bool.pick(U32, fl, 2, 0)),
        Bool.and(ok, Bool.pick(Bool, Bool.or(fs, fl), U32.is_eq(mode, 0), Bool.not(U32.is_eq(v, 4294967295)))))
    case Nil{}:
      (List.reverse(&2, U32, ts), (Bool.pick(U32, Bool.and(ok, U32.is_eq(mode, 0)), st, 0), ly))

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
  "usage: engine <packdir> <tok>... [--steps N] [--layers L]  (N >= 1, ids < 130560, prompt+N-1 <= 131072, L=0..42)"

def run.go(ok: Bool, +dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        IO.print_err(Bool.pick(String, O.order.v2(), "order v2", "order v1"))
        +m : Model <- mf.model(dir, ly)
        +dT : Nat = lg(17n, 0n, (U32.from_nat(count_ids(ts, 0n)) + st - 1 : U32))
        prompt(ts, U32.to_nat((st - 1 : U32)), m, dT, (1 << dT : U32), 0, 0,
          caches(m.nl(m), Nat.add(dT, 8n)))
        t1 : Nat <- IO.now()
        IO.print("wall_ms " ++ Nat.show(Nat.sub(t1, t0)))
    case False{}:
      IO.die(Unit, 2, usage())

def run.chk(dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32) -> IO(Unit):
  +nt = U32.from_nat(count_ids(ts, 0n))
  run.go(Bool.and(Bool.and(ok_ids(ts), (nt >= 1 : U32)),
    Bool.and(Bool.and((st >= 1 : U32), (st <= 131072 : U32)),
      Bool.and((ly <= 42 : U32), (nt <= (131073 - st : U32) : U32)))), dir, ts, st, ly)

def run.sl(dir: String, ts: List<&2, U32>, sl: U32 & U32) -> IO(Unit):
  (st, ly) = sl
  run.chk(dir, ts, st, ly)

def run(dir: String, c: List<&2, U32> & U32 & U32) -> IO(Unit):
  (ts, sl) = c
  run.sl(dir, ts, sl)

def main.go(args: List<String>) -> IO(Unit):
  match args:
    case Con{dir, rest}:
      run(dir, cfg(rest, Nil{}, 8, 0, 0, True{}))
    case Nil{}:
      IO.die(Unit, 2, usage())

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
