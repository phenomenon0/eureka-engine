# engine_qwen35_gpu — Qwen3.5-9B hybrid on the emitted GPU lane.
#
# STATUS (w2speed1): the four hybrid ops are host-served warp bangs
# (F.llb.{l2norm,conv1d,gated_rmsnorm,gated_delta}_warp!; tools/payload-hybrid
# .patch adds the FIDs, the mail decode and the llb_l2norm/llb_conv1d/
# llb_gated_rmsnorm/llb_gated_delta kernels). Byte-exact with the emitted CPU
# lane on all 12 oracle positions + 8/8 tokens; locked hot median ~1504 ms/fwd
# (was 168968 ms in the w2h device-interpreter lane). --lpb K groups the layer
# bangs per serve boundary (default 32 = the whole pass in one bang, the
# optimum). The residual cost is the device interpreter (~90%), not the hybrid
# kernels (<0.2%).
#
# engine_qwen35 <packdir> <tok>... [--steps N] [--layers L] [--emit <layer>] [--lpb K]
#
# Emitted CPU path for Qwen3.5-9B (hybrid: 24 gated-delta-net linear layers +
# 8 full-attention layers, (i+1)%4==0). Pure Bend: every op is kernels/ops.bend's
# balanced tree; Q8_0 matrices decode through kernels/flat.bend's q8 bodies.
# No bangs: this is the interpreter-free native CPU lane.
#
# stdout: generated ids, one per line, then `wall_ms <ms>`.
# stderr: `logits <pos> <id1> <bits1> <id2> <bits2>`, `step <pos> <ms>`.
# --layers L runs the first L layers only (test knob). --emit L prints the
# hidden state after layer L-1 (or the embedding for L=0) as U32 bit lines.
import Base
import ./kernels/ops.bend as K
import ./kernels/io.bend as I
import ./kernels/flat.bend as F
import ./gguf.bend as G
import ./tools/sys1dual/sys1.bend as S

# Array.map's callback is a closure fid the device program does not hold, so
# the scalar activations run as direct structural recursion (same order).
def sig.a(a: Array<F32>) -> Array<F32>:
  match a:
    case ALeaf{x}:
      ALeaf{K.sigmoid1(x)}
    case ANode{l, r}:
      ANode{sig.a(l), sig.a(r)}

def sp.a(a: Array<F32>) -> Array<F32>:
  match a:
    case ALeaf{x}:
      ALeaf{K.softplus1(x)}
    case ANode{l, r}:
      ANode{sp.a(l), sp.a(r)}

def ex.a(a: Array<F32>) -> Array<F32>:
  match a:
    case ALeaf{x}:
      ALeaf{K.exp32(x)}
    case ANode{l, r}:
      ANode{ex.a(l), ex.a(r)}

# a layer's tensor offsets, relative to its layer file. Linear layers fill the
# first half; full layers the second. `an` (attn_norm) and the FFN/post fields
# are shared across both shapes.
type Offs is Data:
  Offs{an: U32, post: U32, fg: U32, fu: U32, fd: U32,
    qkv: U32, gate: U32, sb: U32, sa: U32, ssa: U32, dt: U32, cv: U32, sn: U32, so: U32,
    aq: U32, ak: U32, av: U32, aqn: U32, akn: U32, ao: U32}

# The shape record (step2a phase 3): the manifest's d_model, d_ff, n_heads,
# n_kv_heads, head_dim, ssm_group_count, ssm_time_step_rank, ssm_state_size;
# the pass bang carries it to the host chains (comp.ts LlbShape)
type Shape is Data:
  Shape{d: U32, f: U32, nh: U32, nkv: U32, hd: U32, lkh: U32, lvh: U32, sd: U32}

# The model config the kernels read (tools/payload-anyshape.patch): the rms eps and rope base as
# F32 bits, the rope dims, the vocabulary (the head's rows), the KV capacity in positions (the
# run's prompt + steps; 0 until run.go sets it). The pass bang carries all but voc as its config
# words 66..69 (comp.ts pass_mail; llb_host.c llb_pass_init puts eps / rope on the device)
type Cfg is Data:
  Cfg{eps: U32, rb: U32, rd: U32, voc: U32, cap: U32}

type Model is Data:
  Model{nl: Nat, lin: Offs, ful: Offs, ws: List<&2, Bytes()>, pk: Bytes(), ep: Bytes(), ooff: U32, noff: U32,
    sh: Shape, tb: Bytes(), orw: U32, erw: U32, cf: Cfg}

def ws.at(n: Nat, ws: List<&2, Bytes()>) -> Bytes():
  match n ws:
    case 0n Con{w, t}:
      w
    case 1n+p Con{w, t}:
      ws.at(p, t)
    case _ _:
      SNil{}

def m.nl(m: Model) -> Nat:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      nl

def m.lino(m: Model) -> Offs:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      lin

def m.fulo(m: Model) -> Offs:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      ful

def m.ws(m: Model) -> List<&2, Bytes()>:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      ws

# pack.bin output+output_norm resident region (pk, ooff/noff inside it) and the
# token_embd region (ep, its rows at tok*4352)
def m.pk(m: Model) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      pk

def m.ep(m: Model) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      ep

def m.ow(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      ooff

def m.nb(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      noff

def m.sh(m: Model) -> Shape:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      sh

# table.bin (tools/pack_table.py): a 32-word row per layer, [0] the layer
# file's bytes, [1..20] each tensor's offset | fm << 28 in Offs order
def m.tb(m: Model) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      tb

# the head (output.weight) and embedding (token_embd) rows as tagged row words
# (kw): the pass bang's words 55 and 65, the row stride tok * (erw & 0x07ffffff)
def m.orw(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      orw

def m.erw(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      erw

def m.cf(m: Model) -> Cfg:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      cf

def cf.eps(c: Cfg) -> U32:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      eps

def cf.rb(c: Cfg) -> U32:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      rb

def cf.rd(c: Cfg) -> U32:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      rd

def cf.voc(c: Cfg) -> U32:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      voc

def cf.cap(c: Cfg) -> U32:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      cap

def cf.with_cap(c: Cfg, +k: U32) -> Cfg:
  match c:
    case Cfg{eps, rb, rd, voc, cap}:
      Cfg{eps, rb, rd, voc, k}

def m.voc(+m: Model) -> U32:
  cf.voc(m.cf(m))

# the model with its KV capacity set (run.go, once dT is known)
def m.with_cap(m: Model, +c: U32) -> Model:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf}:
      Model{nl, lin, ful, ws, pk, ep, ooff, noff, sh, tb, orw, erw, cf.with_cap(cf, c)}

# Resident weight files
# ---------------------

def ws.close(got: File & Bytes()) -> IO(Bytes()):
  (f, b) = got
  do IO<Bytes()>:
    File.close(f)
    return b

def ws.load(+path: String, +n: U32) -> IO(Bytes()):
  do IO<Bytes()>:
    f : File <- IO.try(File, File.open(path, "r"))
    got : File & Bytes() <- I.io.buf.read(f, n)
    ws.close(got)

# n bytes from skip: Bytes are capped at 2^31, so a >2 GB head region must be
# read in parts; there is no seek, so discard the prefix in 1 MiB chunks.
def ws.load_skip(+path: String, +skip: Nat, +n: U32) -> IO(Bytes()):
  do IO<Bytes()>:
    f : File <- IO.try(File, File.open(path, "r"))
    f2 : File <- I.io.buf.skip(skip, skip, f)
    got : File & Bytes() <- I.io.buf.read(f2, n)
    ws.close(got)

# the u32 LE word at byte k of b
def le32(+b: Bytes(), +k: Nat) -> U32:
  (Bytes.get(b, k) .|. (Bytes.get(b, Nat.add(k, 1n)) << 8n) .|. (Bytes.get(b, Nat.add(k, 2n)) << 16n)
    .|. (Bytes.get(b, Nat.add(k, 3n)) << 24n) : U32)

# one layer file, its size from the table (formats differ per layer)
def ws.layer(+dir: String, +i: U32, +sz: U32, +g: Bool) -> IO(Bytes()):
  match g:
    case True{}:
      G.gguf.get(i, 0, sz)
    case False{}:
      ws.load(dir ++ "/layers/blk." ++ U32.show(i) ++ ".bin", sz)

def ws.layers(n: Nat, +dir: String, +i: U32, +tb: Bytes(), +g: Bool) -> IO(List<&2, Bytes()>):
  match n:
    case 0n:
      IO.pure(List<&2, Bytes()>, Nil{})
    case 1n+p:
      do IO<List<&2, Bytes()>>:
        w : Bytes() <- ws.layer(dir, i, le32(tb, Nat.mul(U32.to_nat(i), 128n)), g)
        rest : List<&2, Bytes()> <- ws.layers(p, dir, (i + 1 : U32), tb, g)
        return w <> rest

# manifest.json — a schema-specific reader, not a general JSON parser
# ------------------------------------------------------------------

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

def mf.value(s: String, name: String, key: String) -> String:
  mf.scalar(mf.object(s, name), key)

def mf.num(s: String, name: String, key: String) -> Maybe<&2, U32>:
  U32.read(mf.value(s, name, key))

def mf.check(ok: Bool, what: String) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, "unsupported/invalid qwen3.5 pack: " ++ what)

def mf.cv(+s: String, +key: String, want: String) -> Bool:
  String.eq(mf.value(s, "config", key), want)

def mf.config(+s: String, +key: String, want: String) -> IO(Unit):
  mf.check(String.eq(mf.value(s, "config", key), want), key)

# the Q8_0 pack, or the quantized pack (tools/pack_qwen35_quant: each tensor's
# src_type kept, the GEMVs Q8_0 / Q4_K / Q5_K / Q6_K by table.bin)
def mf.q8(+s: String) -> Bool:
  String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q8/v0\"")

# a routed-expert model (tools/payload-moeany.patch): the shared expert takes the dense FFN's
# table words (ffn_gate / ffn_up / ffn_down = ffn_*_shexp); the router and the experts are
# table.bin words 21..29, read on the device (comp.ts pass_mail) and by llb_host.c llb_moe_run
def mf.moe(+s: String) -> Bool:
  String.eq(mf.value(s, "config", "architecture"), "\"qwen3_5_moe\"")

def mf.validate(+s: String) -> IO(Unit):
  do IO<Unit>:
    mf.check(Bool.or(Bool.and(mf.q8(s), String.eq(mf.scalar(s, "dtype"), "\"q8_0-rowsplit\"")),
      Bool.and(String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q35-quant/v0\""),
        String.eq(mf.scalar(s, "dtype"), "\"mixed-quant\""))), "format/dtype")
    # qwen3_5_moe: the sidecar of a qwen35moe GGUF (gguf2pack --sidecar-only; tools/payload-moeany.patch)
    mf.check(Bool.or(mf.cv(s, "architecture", "\"qwen3_5\""), mf.moe(s)), "architecture (qwen3_5 or qwen3_5_moe)")
    # tools/payload-anyshape.patch: any dense Qwen3.5 shape the CUDA kernels instantiate (was a
    # whitelist of the 9B, 27B and 0.8B); the widths are checked in mf.shape once read. These
    # two are kernel instantiations: llb_attn4 / llb_qkr run a 256-thread head, l2norm_qk /
    # gated_delta / gated_rmsnorm a 128-wide state
    mf.check(mf.cv(s, "head_dim", "256"), "head_dim (the CUDA kernels need 256)")
    mf.check(mf.cv(s, "ssm_state_size", "128"), "ssm_state_size (the CUDA kernels need 128)")
    mf.check(Bool.or(mf.cv(s, "ssm_conv_kernel", "4"), String.eq(mf.value(s, "config", "ssm_conv_kernel"), "")),
      "ssm_conv_kernel (the CUDA conv1d needs 4)")

# a head / embedding tensor's registry format (host/llb_formats.h): the Q8_0
# pack is all Q8_0; else its src_type, 0 (refused) outside Q8_0 / Q4_K / Q6_K / Q5_K
def mf.fm(+s: String, +name: String) -> U32:
  +t = mf.value(s, name, "src_type")
  Bool.pick(U32, Bool.or(mf.q8(s), String.eq(t, "\"Q8_0\"")), 1,
    Bool.pick(U32, String.eq(t, "\"Q4_K\""), 3, Bool.pick(U32, String.eq(t, "\"Q6_K\""), 4,
      Bool.pick(U32, String.eq(t, "\"Q5_K\""), 9, 0))))

# the tagged row word for format fm (as q8rw: bit 30, fm in bits 28-29, 31, 27)
def kw(+fm: U32, +rb: U32) -> U32:
  (rb .|. 1073741824 .|. Bool.pick(U32, U32.is_eq(fm, 1), 268435456, Bool.pick(U32, U32.is_eq(fm, 3), 805306368,
    Bool.pick(U32, U32.is_eq(fm, 4), 2147483648, 402653184))) : U32)

def mf.need(what: String, m: Maybe<&2, U32>) -> IO(U32):
  match m:
    case Some{v}:
      IO.pure(U32, v)
    case None{}:
      IO.die(U32, 1, "manifest.json: no " ++ what)

def mf.need_nat(what: String, m: Maybe<&2, Nat>) -> IO(Nat):
  match m:
    case Some{v}:
      IO.pure(Nat, v)
    case None{}:
      IO.die(Nat, 1, "manifest.json: no " ++ what)

def mf.off_nat(+s: String, +name: String) -> IO(Nat):
  mf.need_nat(name ++ " offset", Nat.read(mf.value(s, name, "offset")))

def mf.rel(name: String, off: Nat, base: Nat, +bytes: U32) -> IO(U32):
  +r = Nat.sub(off, base)
  do IO<U32>:
    mf.check(Nat.is_le(r, Nat.sub(Nat.add(4294967295n, 1n), U32.to_nat(bytes))), name ++ " layer exceeds U32")
    return U32.from_nat(r)

def mf.text(r: File & Result<&1, &1, U32 & String, String>) -> IO(String):
  (f, t) = r
  do IO<String>:
    File.close(f)
    IO.pass(String, t)

# the linear (gated-delta-net) layer layout
def mf.offl.lin(+s: String, +p: String, +x: String) -> IO(Offs):
  do IO<Offs>:
    +an : Nat <- mf.off_nat(s, p ++ "attn_norm.weight")
    +post : Nat <- mf.off_nat(s, p ++ "post_attention_norm.weight")
    +fg : Nat <- mf.off_nat(s, p ++ "ffn_gate" ++ x ++ ".weight")
    +fu : Nat <- mf.off_nat(s, p ++ "ffn_up" ++ x ++ ".weight")
    +fd : Nat <- mf.off_nat(s, p ++ "ffn_down" ++ x ++ ".weight")
    +qkv : Nat <- mf.off_nat(s, p ++ "attn_qkv.weight")
    +gate : Nat <- mf.off_nat(s, p ++ "attn_gate.weight")
    +sb : Nat <- mf.off_nat(s, p ++ "ssm_beta.weight")
    +sa : Nat <- mf.off_nat(s, p ++ "ssm_alpha.weight")
    +ssa : Nat <- mf.off_nat(s, p ++ "ssm_a")
    +dt : Nat <- mf.off_nat(s, p ++ "ssm_dt.bias")
    +cv : Nat <- mf.off_nat(s, p ++ "ssm_conv1d.weight")
    +sn : Nat <- mf.off_nat(s, p ++ "ssm_norm.weight")
    +so : Nat <- mf.off_nat(s, p ++ "ssm_out.weight")
    +base : Nat = Nat.min(Nat.min(Nat.min(an, post), Nat.min(fg, Nat.min(fu, fd))),
      Nat.min(Nat.min(qkv, gate), Nat.min(sb, Nat.min(sa, Nat.min(ssa, Nat.min(dt, Nat.min(cv, Nat.min(sn, so))))))))
    anr : U32 <- mf.rel("attn_norm", an, base, 16384)
    postr : U32 <- mf.rel("post_attention_norm", post, base, 16384)
    fgr : U32 <- mf.rel("ffn_gate", fg, base, 53477376)
    fur : U32 <- mf.rel("ffn_up", fu, base, 53477376)
    fdr : U32 <- mf.rel("ffn_down", fd, base, 53477376)
    qkvr : U32 <- mf.rel("attn_qkv", qkv, base, 35651584)
    gater : U32 <- mf.rel("attn_gate", gate, base, 17825792)
    sbr : U32 <- mf.rel("ssm_beta", sb, base, 139264)
    sar : U32 <- mf.rel("ssm_alpha", sa, base, 139264)
    ssar : U32 <- mf.rel("ssm_a", ssa, base, 128)
    dtr : U32 <- mf.rel("ssm_dt", dt, base, 128)
    cvr : U32 <- mf.rel("ssm_conv1d", cv, base, 131072)
    snr : U32 <- mf.rel("ssm_norm", sn, base, 512)
    sor : U32 <- mf.rel("ssm_out", so, base, 17825792)
    return Offs{anr, postr, fgr, fur, fdr, qkvr, gater, sbr, sar, ssar, dtr, cvr, snr, sor,
      0, 0, 0, 0, 0, 0}

# the full-attention layer layout
def mf.offl.ful(+s: String, +p: String, +x: String) -> IO(Offs):
  do IO<Offs>:
    +an : Nat <- mf.off_nat(s, p ++ "attn_norm.weight")
    +post : Nat <- mf.off_nat(s, p ++ "post_attention_norm.weight")
    +fg : Nat <- mf.off_nat(s, p ++ "ffn_gate" ++ x ++ ".weight")
    +fu : Nat <- mf.off_nat(s, p ++ "ffn_up" ++ x ++ ".weight")
    +fd : Nat <- mf.off_nat(s, p ++ "ffn_down" ++ x ++ ".weight")
    +aq : Nat <- mf.off_nat(s, p ++ "attn_q.weight")
    +ak : Nat <- mf.off_nat(s, p ++ "attn_k.weight")
    +av : Nat <- mf.off_nat(s, p ++ "attn_v.weight")
    +aqn : Nat <- mf.off_nat(s, p ++ "attn_q_norm.weight")
    +akn : Nat <- mf.off_nat(s, p ++ "attn_k_norm.weight")
    +ao : Nat <- mf.off_nat(s, p ++ "attn_output.weight")
    +base : Nat = Nat.min(Nat.min(Nat.min(an, post), Nat.min(fg, Nat.min(fu, fd))),
      Nat.min(Nat.min(aq, ak), Nat.min(av, Nat.min(aqn, Nat.min(akn, ao)))))
    anr : U32 <- mf.rel("attn_norm", an, base, 16384)
    postr : U32 <- mf.rel("post_attention_norm", post, base, 16384)
    fgr : U32 <- mf.rel("ffn_gate", fg, base, 53477376)
    fur : U32 <- mf.rel("ffn_up", fu, base, 53477376)
    fdr : U32 <- mf.rel("ffn_down", fd, base, 53477376)
    aqr : U32 <- mf.rel("attn_q", aq, base, 35651584)
    akr : U32 <- mf.rel("attn_k", ak, base, 4456448)
    avr : U32 <- mf.rel("attn_v", av, base, 4456448)
    aqnr : U32 <- mf.rel("attn_q_norm", aqn, base, 1024)
    aknr : U32 <- mf.rel("attn_k_norm", akn, base, 1024)
    aor : U32 <- mf.rel("attn_output", ao, base, 17825792)
    return Offs{anr, postr, fgr, fur, fdr, 0, 0, 0, 0, 0, 0, 0, 0, 0, aqr, akr, avr, aqnr, aknr, aor}

# a manifest integer, dflt when absent (tools/payload-anyshape.patch: rope_dim)
def mf.u32or(m: Maybe<&2, U32>, +dflt: U32) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      dflt

# a manifest float (decimal text, as json.dumps wrote it) as F32 bits; dflt when absent or
# unreadable (tools/payload-anyshape.patch: rms_norm_eps, rope_freq_base)
def mf.f32bits.of(m: Maybe<&2, F64>, +dflt: U32) -> U32:
  match m:
    case Some{v}:
      F32.bits(F64.to_f32(v))
    case None{}:
      dflt

def mf.f32bits(s: String, +dflt: U32) -> U32:
  mf.f32bits.of(F64.read(s), dflt)

# the routed-expert shape (tools/payload-moeany.patch): what llb_moe_topk / llb_moe_gu / llb_moe_dn
# take -- any expert count and top-k (k <= the count), expert and shared widths in whole 256-col
# superblocks (the K-quant rows); the per-layer formats are checked by gguf2pack's sidecar and
# pass_mail. The stacks load from the GGUF (llb_host.c llb_moe_load), so LLB_GGUF is needed
def mf.moechk(moe: Bool, +s: String, +g: Bool) -> IO(Unit):
  match moe:
    case False{}:
      IO.pure(Unit, Unit{})
    case True{}:
      do IO<Unit>:
        +ne : U32 <- mf.need("n_experts", mf.num(s, "config", "n_experts"))
        +nk : U32 <- mf.need("n_experts_used", mf.num(s, "config", "n_experts_used"))
        +ef : U32 <- mf.need("expert_ff", mf.num(s, "config", "expert_ff"))
        +sf : U32 <- mf.need("shared_ff", mf.num(s, "config", "shared_ff"))
        mf.check(g, "a MoE model loads its experts from the GGUF: set LLB_GGUF=MODEL.gguf (the dir is its gguf2pack --sidecar-only)")
        mf.check((ne >= 1 : U32), "expert_count (>= 1)")
        mf.check(Bool.and((nk >= 1 : U32), (nk <= ne : U32)), "expert_used_count (1..expert_count)")
        mf.check(Bool.and(Bool.and((ef >= 256 : U32), (ef <= 65536 : U32)), U32.is_eq(U32.mod(ef, 256), 0)),
          "expert_feed_forward_length (a multiple of 256: the expert rows in whole superblocks)")
        mf.check(Bool.and(Bool.and((sf >= 256 : U32), (sf <= 65536 : U32)), U32.is_eq(U32.mod(sf, 256), 0)),
          "expert_shared_feed_forward_length (a multiple of 256)")

# the structural shape check (tools/payload-anyshape.patch; comp.ts pass_mail's envelope
# backs it on the device): what the CUDA kernels instantiate, the rest any value
def mf.shape(+nl: U32, +dm: U32, +dff: U32, +nh: U32, +nkv: U32, +lkh: U32, +lvh: U32, +iv: U32) -> IO(Unit):
  do IO<Unit>:
    mf.check(Bool.and((nl >= 1 : U32), (nl <= 256 : U32)), "n_layers (1..256: the CUDA pass's layer tables)")
    mf.check(Bool.and(Bool.and((dm >= 256 : U32), (dm <= 32768 : U32)), U32.is_eq(U32.mod(dm, 256), 0)),
      "d_model (a multiple of 256, 256..32768: the rms trees and K-quant rows)")
    mf.check(Bool.and(Bool.and((dff >= 256 : U32), (dff <= 65536 : U32)), U32.is_eq(U32.mod(dff, 256), 0)),
      "d_ff (a multiple of 256, 256..65536)")
    mf.check(Bool.and(Bool.and((nkv >= 1 : U32), (nh >= nkv : U32)), U32.is_eq(U32.mod(nh, nkv), 0)),
      "n_heads / n_kv_heads (a whole number of q heads per kv head)")
    mf.check(Bool.and(Bool.and((lkh >= 1 : U32), (lvh >= lkh : U32)), Bool.and(U32.is_eq(U32.mod(lvh, lkh), 0), (lvh <= 256 : U32))),
      "ssm_time_step_rank / ssm_group_count (a whole number of value heads per key head, <= 256)")
    mf.check(Bool.and((nh <= 128 : U32), (lkh <= 128 : U32)), "n_heads / ssm_group_count (<= 128: q / k widths <= 32768)")
    mf.check(Bool.and((iv >= 2 : U32), (iv <= nl : U32)),
      "full_attention_interval (2..n_layers: a hybrid with a linear layer first)")

def mf.model(+dir: String, +ly: U32, +em: U32, +lb: U32) -> IO(Model):
  do IO<Model>:
    f : File <- IO.try(File, File.open(dir ++ "/manifest.json", "r"))
    r : File & Result<&1, &1, U32 & String, String> <- File.read(f, 16777216)
    +s : String <- mf.text(r)
    mf.validate(s)
    +nl : U32 <- mf.need("n_layers", mf.num(s, "config", "n_layers"))
    +emb : U32 <- mf.need("token_embd offset", mf.num(s, "token_embd.weight", "offset"))
    +onorm : U32 <- mf.need("output_norm offset", mf.num(s, "output_norm.weight", "offset"))
    +out : U32 <- mf.need("output offset", mf.num(s, "output.weight", "offset"))
    +ob : U32 <- mf.need("output bytes", mf.num(s, "output.weight", "bytes"))
    +eb : U32 <- mf.need("token_embd bytes", mf.num(s, "token_embd.weight", "bytes"))
    +orb : U32 <- mf.need("output row_bytes", mf.num(s, "output.weight", "row_bytes"))
    +erb : U32 <- mf.need("token_embd row_bytes", mf.num(s, "token_embd.weight", "row_bytes"))
    +ofm : U32 = mf.fm(s, "output.weight")
    +efm : U32 = mf.fm(s, "token_embd.weight")
    # embeddings Q8_0 / Q4_K / Q6_K (llb_embed / _embedk / _embed6; Q6_K any width since
    # tools/payload-anyshape.patch, was the 0.8B's only); heads Q8_0 / Q4_K / Q6_K / Q5_K
    mf.check(Bool.and(Bool.not(U32.is_eq(ofm, 0)), Bool.or(Bool.or(U32.is_eq(efm, 1), U32.is_eq(efm, 3)),
      U32.is_eq(efm, 4))),
      "head/embedding format (head Q8_0 / Q4_K / Q6_K / Q5_K, embedding Q8_0 / Q4_K / Q6_K)")
    +dm : U32 <- mf.need("d_model", mf.num(s, "config", "d_model"))
    +dff : U32 <- mf.need("d_ff", mf.num(s, "config", "d_ff"))
    +nhd : U32 <- mf.need("n_heads", mf.num(s, "config", "n_heads"))
    +nkv : U32 <- mf.need("n_kv_heads", mf.num(s, "config", "n_kv_heads"))
    +hd : U32 <- mf.need("head_dim", mf.num(s, "config", "head_dim"))
    +lkh : U32 <- mf.need("ssm_group_count", mf.num(s, "config", "ssm_group_count"))
    +lvh : U32 <- mf.need("ssm_time_step_rank", mf.num(s, "config", "ssm_time_step_rank"))
    +sd : U32 <- mf.need("ssm_state_size", mf.num(s, "config", "ssm_state_size"))
    +inner : U32 <- mf.need("ssm_inner_size", mf.num(s, "config", "ssm_inner_size"))
    # the value heads are sd wide (gdl / grn): the record has no separate key
    mf.check(U32.is_eq(inner, (lvh * sd : U32)), "ssm_inner_size != ssm_time_step_rank * ssm_state_size")
    +voc : U32 <- mf.need("vocab_size", mf.num(s, "config", "vocab_size"))
    +iv : U32 <- mf.need("full_attention_interval", mf.num(s, "config", "full_attention_interval"))
    +rd : U32 = mf.u32or(mf.num(s, "config", "rope_dim"), 64)
    +eps : U32 = mf.f32bits(mf.value(s, "config", "rms_norm_eps"), 897988541)
    +rbase : U32 = mf.f32bits(mf.value(s, "config", "rope_freq_base"), 1259902592)
    +moe : Bool = mf.moe(s)
    gm : Result<&1, &1, U32 & String, String> <- IO.get_env("LLB_GGUF")
    mf.moechk(moe, s, G.gguf.on(gm))
    mf.shape(nl, dm, dff, nhd, nkv, lkh, lvh, iv)
    mf.check(Bool.and(Bool.and((rd >= 2 : U32), (rd <= 256 : U32)), U32.is_eq(U32.and(rd, 1), 0)),
      "rope_dim (even, 2..256)")
    mf.check(Bool.and(Bool.not(U32.is_eq(eps, 0)), (eps < 2139095040 : U32)), "rms_norm_eps (finite, > 0)")
    mf.check(Bool.and(Bool.not(U32.is_eq(rbase, 0)), (rbase < 2139095040 : U32)), "rope_freq_base (finite, > 0)")
    # the head's rows: gpu_logit holds 2^18 floats less llb_amax_part's 4 x 128-word tail
    mf.check(Bool.and((voc >= 1 : U32), (voc <= 261632 : U32)), "vocab_size (1..261632: the CUDA head's logit buffer)")
    mf.check(U32.is_eq((ob / orb : U32), voc), "output.weight rows != vocab_size")
    mf.check(Bool.and((emb <= 3225419776 : U32), Bool.and((out <= 3225419776 : U32),
      (onorm <= 4294959104 : U32))), "head offset exceeds U32")
    +fx : String = Bool.pick(String, moe, "_shexp", "")
    lin : Offs <- mf.offl.lin(s, "blk.0.", fx)
    # the first full-attention layer: blk.(interval - 1) (was blk.3)
    ful : Offs <- mf.offl.ful(s, "blk." ++ U32.show((iv - 1 : U32)) ++ ".", fx)
    +n : Nat = U32.to_nat(Bool.pick(U32, U32.is_eq(ly, 0), nl, ly))
    # the experts run in the folded pass only (llb_host.c llb_moe_run): the split path's
    # ffn_block bang is the dense FFN
    mf.check(Bool.or(Bool.not(moe), Bool.and(U32.is_eq(em, 4294967295), (lb >= U32.from_nat(n) : U32))),
      "a MoE model needs the folded step (no --emit, --lpb >= layers)")
    # output.weight and output_norm sit contiguously at the front of pack.bin;
    # token_embd follows. Bytes are capped at 2^31, so read the head region in
    # two parts (pk: output+norm from 0, ep: token_embd after a skip).
    +out0 : Nat = U32.to_nat(out)
    +norm0 : Nat = U32.to_nat(onorm)
    +emb0 : Nat = U32.to_nat(emb)
    +headn : Nat = Nat.max(Nat.add(out0, U32.to_nat(ob)), Nat.add(norm0, Nat.mul(U32.to_nat(dm), 4n)))
    mf.check(Nat.is_le(headn, 2147483648n), "output/norm region exceeds 2^31")
    ge : Result<&1, &1, U32 & String, String> <- IO.get_env("LLB_GGUF")
    +g : Bool = G.gguf.on(ge)
    t0 : Nat <- IO.now()
    +tb : Bytes() <- ws.load(dir ++ "/table.bin", (nl * 128 : U32))
    ws : List<&2, Bytes()> <- ws.layers(n, dir, 0, tb, g)
    pk : Bytes() <- Bool.pick(IO(Bytes()), g, G.gguf.get(4294967295, 0, U32.from_nat(headn)),
      ws.load(dir ++ "/pack.bin", U32.from_nat(headn)))
    ep : Bytes() <- Bool.pick(IO(Bytes()), g, G.gguf.get(4294967295, emb, eb),
      ws.load_skip(dir ++ "/pack.bin", emb0, eb))
    t1 : Nat <- IO.now()
    IO.print_err("resident head " ++ Nat.show(Nat.add(headn, U32.to_nat(eb)))
      ++ " bytes + layers in " ++ Nat.show(Nat.sub(t1, t0)) ++ " ms")
    return Model{n, lin, ful, ws, pk, ep, U32.from_nat(out0), U32.from_nat(norm0),
      Shape{dm, dff, nhd, nkv, hd, lkh, lvh, sd}, tb, kw(ofm, orb), kw(efm, erb), Cfg{eps, rbase, rd, voc, 0}}

# the loader needs ws.at; define after Model so the field order matches
def ws.at2(n: Nat, ws: List<&2, Bytes()>) -> Bytes():
  ws.at(n, ws)

# Ops
# ----
# f = fork depth for the Q8 gemvs / loads
# D=4096 (d12), CONVD=8192 (d13), d_ff=12288 -> 16384 (d14),
# vocab 248320 -> 262144 (d18), S 32*128*128 (d19)

# the least d <= f with 2^d >= need
def lg(f: Nat, +d: Nat, +need: U32) -> Nat:
  match f:
    case 0n:
      d
    case 1n+p:
      lg(p, Bool.pick(Nat, (need <= (1 << d : U32) : U32), d, 1n+d), need)

# F.q8.gemv's depth is the row's COLUMN depth: it reduces 2^d groups of four
# columns, so d = (x's depth) - 2 (kernels/dev.bend's gemv). The `d` argument
# of q8mv/q8mv.inf below is the ROW depth (it sizes y, Array.new's depth).
def cdepth(+cols: U32) -> Nat:
  Nat.sub(lg(31n, 0n, cols), 2n)

# the warp GEMV row word: Q8_0 row bytes (cols * 17 / 16) with the pack tag
# (tools/quant-fix.patch bit 30, format 1 in bits 28-29); kernels/flat.bend's
# llb.gemv_warp reads it, the emitted CPU q8.gemv read the same raw bytes.
def q8rw(+cols: U32) -> U32:
  ((((cols / 32 : U32) * 34 : U32) .|. 1342177280 : U32))

# the pass's Q8 row words (rg d_model, rd d_ff columns) from the Shape
def sh.rg(sh: Shape) -> U32:
  match sh:
    case Shape{d, f, nh, nkv, hd, lkh, lvh, sd}:
      q8rw(d)

def sh.rd(sh: Shape) -> U32:
  match sh:
    case Shape{d, f, nh, nkv, hd, lkh, lvh, sd}:
      q8rw(f)

@unsafe
def q8mv(+f: Nat, +wt: Bytes(), +x: Array<F32>, +off: U32, +n: U32, +d: Nat, +cols: U32) -> Array<F32>:
  F.llb.gemv_warp!(f, wt, x, Array.new(F32, d, 0.0), off, n, cdepth(cols), q8rw(cols))

@unsafe
def q8mv.inf(+f: Nat, +wt: Bytes(), +x: Array<F32>, +off: U32, +n: U32, +d: Nat, +cols: U32) -> Array<F32>:
  F.llb.gemv_warp!(f, wt, x, Array.new(F32, d, K.neg_inf()), off, n, cdepth(cols), q8rw(cols))

@unsafe
def vec(+wt: Bytes(), +off: U32, +d: Nat) -> Array<F32>:
  F.load(6n, wt, off, d)

# State (one 6-tuple of arrays per layer: conv h0,h1,h2, S, K, V)
# ----------------------------------------------------------------
# A Data record cannot hold Array fields in this compiler, so a layer's state
# is a right-nested tuple; the six are tied in one type expression.

def st.new(+cd: Nat) -> Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>:
  (Array.new(F32, 13n, 0.0), (Array.new(F32, 13n, 0.0), (Array.new(F32, 13n, 0.0),
    (Array.new(F32, 19n, 0.0), (Array.new(F32, cd, 0.0), Array.new(F32, cd, 0.0))))))

# cd: the K / V arrays' depth, 2^cd floats each (tools/payload-anyshape.patch: st.kd)
def states(n: Nat, +cd: Nat) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      st.new(cd) <> states(p, cd)

# the Bend-side K / V depth (tools/payload-anyshape.patch): the split path (--emit, --lpb < layers)
# reads kc / vc, 2^(dT + 10) floats a layer; the folded step keeps its caches on the host
# (llb_host.c pass_kv) and never reads these, so one leaf each -- they took 2^(dT + 13) bytes of
# host memory a layer (6.4 GB at dT 15 on the 0.8B; 48 GB at dT 18)
def st.kd(+m: Model, +b: Nat, +em: U32, +dT: Nat) -> Nat:
  Bool.pick(Nat, Bool.and(U32.is_eq(em, 4294967295), (U32.from_nat(b) >= U32.from_nat(m.nl(m)) : U32)),
    0n, Nat.add(dT, 10n))

# One linear (gated delta net) layer: returns the updated state and the
# attention block's contribution (Wo y), to be added to x by the caller.
@unsafe
def lin.gd(gd: Array<F32> & Array<F32>, z: Array<F32>, +wt: Bytes(), +wsn: U32, +wso: U32, h1: Array<F32>, h2: Array<F32>,
  qkva: Array<F32>, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (gout, s2) = gd
  +normed = F.llb.gated_rmsnorm_warp!(5n, gout, wt, Array.new(F32, 12n, 0.0), wsn, z, 5n)
  +out0 = q8mv(14n, wt, normed, wso, 4096, 12n, 4096)
  ((h1, (h2, (qkva, (s2, (kc, vc))))), out0)

@unsafe
def lin.y(y: Array<F32>, +s: Array<F32>, +z: Array<F32>, +gp: Array<F32>, +beta: Array<F32>, +wt: Bytes(), +wsn: U32,
  +wso: U32, h1: Array<F32>, h2: Array<F32>, qkva: Array<F32>, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  match y:
    case ANode{ANode{yq0, yk0}, yv}:
      +qt16 = F.llb.l2norm_warp!(4n, yq0, F32.bits(F32.sqrt(128.0)), Array.new(F32, 11n, 0.0), 4n)
      +kt16 = F.llb.l2norm_warp!(4n, yk0, F32.bits(1.0), Array.new(F32, 11n, 0.0), 4n)
      +qt = K.both(Array.clone(F32, qt16), x => x, w => w)
      +kt = K.both(Array.clone(F32, kt16), x => x, w => w)
      lin.gd((F.llb.gated_delta_warp!(5n, qt, s, Array.new(F32, 12n, 0.0), kt, yv, ex.a(gp), beta, 5n), s),
        z, wt, wsn, wso, h1, h2, qkva, kc, vc)
    case _:
      ((h1, (h2, (qkva, (s, (kc, vc))))), ALeaf{0.0})

@unsafe
def lin.conv(qkc: Array<F32> & Array<F32>, h1c: Array<F32> & Array<F32>, h2c: Array<F32> & Array<F32>,
  h0: Array<F32>, +wcv: U32, s: Array<F32>, z: Array<F32>, gp: Array<F32>, beta: Array<F32>,
  +wt: Bytes(), +wsn: U32, +wso: U32, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (qkcx, qkcs) = qkc
  (h1a, h1b) = h1c
  (h2a, h2b) = h2c
  +y = F.llb.conv1d_warp!(13n, wt, wcv, Array.new(F32, 13n, 0.0), qkcx, h0, h1a, h2a, 13n)
  lin.y(y, s, z, gp, beta, wt, wsn, wso, h1b, h2b, qkcs, kc, vc)

@unsafe
def lin.go(+h0: Array<F32>, +h1: Array<F32>, +h2: Array<F32>, +s: Array<F32>, +kc: Array<F32>, +vc: Array<F32>,
  +wt: Bytes(), +x: Array<F32>, +qkva: Array<F32>, +an: U32, +wqkv: U32, +wgate: U32, +wb: U32, +wa: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32, +rg: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  ((h1, (h2, (qkva, (s, (kc, vc))))), F.llb.lin_block_warp!(14n, wt, x, Array.new(F32, 12n, 0.0), qkva,
    an, wqkv, wgate, wb, wa, wsa, wdt, wcv, wsn, wso, h0, h1, h2, s, rg))

@unsafe
def lin(+wt: Bytes(), +wqkv: U32, +wgate: U32, +wb: U32, +wa: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +x: Array<F32>, +an: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (h0, (h1, (h2, (s, (kc, vc))))) = e
  +qkva = Array.new(F32, 13n, 0.0)
  lin.go(h0, h1, h2, s, kc, vc, wt, x, qkva, an, wqkv, wgate, wb, wa, wsa, wdt, wcv, wsn, wso, q8rw(4096))

# One full-attention layer as one host-served bang (tools/payload-fulb.patch):
# attn_norm, q/k/v GEMVs, the q/k norm-ropes, the two KV-cache slot writes (kc / vc
# in place), attn4, the output gate and the o GEMV; the caches come back as lent
@unsafe
def ful.go(+h0: Array<F32>, +h1: Array<F32>, +h2: Array<F32>, +s: Array<F32>, +kc: Array<F32>, +vc: Array<F32>,
  +wt: Bytes(), x: Array<F32>, +an: U32, +waq: U32, +wak: U32, +wav: U32, +waqn: U32, +wakn: U32, +wao: U32,
  +c: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  ((h0, (h1, (h2, (s, (kc, vc))))), F.llb.ful_block_warp!(14n, wt, x, Array.new(F32, 12n, 0.0), kc, vc,
    an, waq, wak, wav, waqn, wakn, wao, c, q8rw(4096)))

@unsafe
def ful(+wt: Bytes(), +an: U32, +waq: U32, +wak: U32, +wav: U32, +waqn: U32, +wakn: U32, +wao: U32,
  e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, x: Array<F32>, +pos: U32, +dT: Nat) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (h0, (h1, (h2, (s, (kc, vc))))) = e
  ful.go(h0, h1, h2, s, kc, vc, wt, x, an, waq, wak, wav, waqn, wakn, wao, (U32.from_nat(dT) .|. (pos << 5n) : U32))

# the attention block (attn_norm + the right layer type): (state, contribution)
@unsafe
def attn.go(full: Bool, +wt: Bytes(), x: Array<F32>, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +an: U32, +waq: U32, +wak: U32, +wav: U32, +waqn: U32,
  +wakn: U32, +wao: U32, +wqkv: U32, +wgate: U32, +wb: U32, +wa: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  match full:
    case True{}:
      ful(wt, an, waq, wak, wav, waqn, wakn, wao, e, x, pos, dT)
    case False{}:
      lin(wt, wqkv, wgate, wb, wa, wsa, wdt, wcv, wsn, wso, e, x, an)

# one layer: attention + residual + post norm + SwiGLU FFN + residual
@unsafe
def lay.res(ar: (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>, x: Array<F32>, +wt: Bytes(), +post: U32, +fg: U32, +fu: U32, +fd: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (e2, a) = ar
  (e2, F.llb.ffn_block_warp!(14n, wt, x, Array.new(F32, 12n, 0.0), a, post, fg, fu, fd,
    q8rw(4096), q8rw(12288)))

@unsafe
def lay.offs(o: Offs, +wt: Bytes(), +i: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +x: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  Offs{an, post, fg, fu, fd, qkv, gate, sb, sa, ssa, dt, cv, sn, so, aq, ak, av, aqn, akn, ao} = o
  +full = U32.is_eq(U32.and((i + 1 : U32), 3), 0)
  lay.res(attn.go(full, wt, x, e, an, aq, ak, av, aqn, akn, ao, qkv, gate, sb, sa, ssa, dt, cv, sn, so, pos, dT, nT),
    x, wt, post, fg, fu, fd)
@unsafe
def lay(+m: Model, +wt: Bytes(), +i: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, x: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  +full = U32.is_eq(U32.and((i + 1 : U32), 3), 0)
  lay.offs(Bool.pick(Offs, full, m.fulo(m), m.lino(m)), wt, i, e, x, pos, dT, nT)

# helpers first (a recursive def cannot be forward referenced)
@unsafe
def layers.push(rec: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
  e2: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  (st2, x3) = rec
  (e2 <> st2, x3)

@unsafe
def layers.bind(got: (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>,
  k: (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) -> Array<F32> -> (List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>)) ->
  List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  (e2, x2) = got
  k(e2, x2)

@unsafe
def layers.run(n: Nat, +i: U32, +m: Model, wl: List<&2, Bytes()>, +pos: U32, +dT: Nat, +nT: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  match n wl st:
    case 1n+p0 Con{w, wr} Con{s0, t}:
      layers.bind(lay(m, w, i, s0, x, pos, dT, nT),
        e2 => x2 => layers.push(layers.run(p0, (i + 1 : U32), m, wr, pos, dT, nT, t, x2), e2))
    case _ _ _:
      (st, x)

@unsafe
def layers.pre(rec: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
  s0: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  (st2, x3) = rec
  (s0 <> st2, x3)

# skip sk states (re-prepending them), then run the next k layers: one serve group
@unsafe
def layers.skip(sk: Nat, +k: Nat, +i: U32, +m: Model, wl: List<&2, Bytes()>, +pos: U32, +dT: Nat, +nT: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  match sk wl st:
    case 1n+p0 Con{w, wr} Con{s0, t}:
      layers.pre(layers.skip(p0, k, i, m, wr, pos, dT, nT, t, x), s0)
    case _ _ _:
      layers.run(k, i, m, wl, pos, dT, nT, st, x)



# Head
# ----

type Tops is Data:
  Tops{v: F32, i: U32, w: F32, j: U32}

# t = llb.head_warp's [v, i, w, j]. The second pick (slot i read as -inf) is
# diagnostics only; the generated id remains the first-index argmax.
@unsafe
def step.tops(+t: Array<F32>) -> Tops:
  Tops{F.got(Array.get(F32, t, 0)), F32.to_u32(F.got(Array.get(F32, t, 1))),
    F.got(Array.get(F32, t, 2)), F32.to_u32(F.got(Array.get(F32, t, 3)))}

# rms_final, the head GEMV (Q8_0) and the top two as one bang (F.llb.head_warp):
# only the four words of the top two leave the device. wt = m.pk (output.weight
# then output_norm), m.ow the output weight's offset, m.nb the norm's.
@unsafe
def head(+m: Model, x: Array<F32>) -> Tops:
  step.tops(F.llb.head_warp!(2n, m.pk(m), x, Array.new(F32, 2n, 0.0), m.nb(m),
    m.ow(m), m.voc(m), q8rw(4096)))

# the head as a device bang (its q8mv is a warp task): a bang in a sequential
# lane faults, so it is entered directly, as engine_gpu's head! is.
def head.fin(+t0: Nat, top: Tops) -> IO(Tops):
  do IO<Tops>:
    t1 : Nat <- IO.now()
    IO.print_err("op head " ++ Nat.show(Nat.sub(t1, t0)))
    return top

@unsafe
def head.timed(+m: Model, x: Array<F32>) -> IO(Tops):
  do IO<Tops>:
    t0 : Nat <- IO.now()
    head.fin(t0, head!(m, x))

@unsafe
def embed(+m: Model, +tok: U32) -> Array<F32>:
  F.q8.load(6n, m.ep(m), (tok * 4352 : U32), 12n)

# Steps
# -----

def step.diag(+pos: U32, +top: Tops) -> IO(U32):
  Tops{+v, +i, +w, j} = top
  do IO<U32>:
    mf.check(Bool.and((U32.and(F32.bits(v), 2139095040) < 2139095040 : U32),
      (U32.and(F32.bits(w), 2139095040) < 2139095040 : U32)), "nonfinite logits at pos " ++ U32.show(pos))
    IO.print_err("logits " ++ U32.show(pos) ++ " " ++ U32.show(i) ++ " " ++ U32.show(F32.bits(v))
      ++ " " ++ U32.show(j) ++ " " ++ U32.show(F32.bits(w)))
    return i

# --emit L: print the hidden state after L layers (L=0: the embedding)
def emit_hidden.go(ok: Bool, x: Array<F32>) -> IO(Unit):
  match ok:
    case True{}:
      I.io.emit_vec(4096n, x)
    case False{}:
      IO.pure(Unit, Unit{})

def emit_hidden(+want: U32, +em: U32, x: Array<F32>) -> IO(Unit):
  emit_hidden.go(U32.is_eq(em, want), x)

@unsafe
def step.emit(xx: Array<F32> & Array<F32>, st2: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>,
  +m: Model, +em: U32, +pos: U32, +t0: Nat) -> IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  (xa, xb) = xx
  do IO<U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    emit_hidden(em, U32.from_nat(m.nl(m)), xa)
    top : Tops <- head.timed(m, xb)
    tok2 : U32 <- step.diag(pos, top)
    t1 : Nat <- IO.now()
    IO.print_err("step " ++ U32.show(pos) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return (tok2, st2)

# one serve group: k layers from s as a single device bang (lay's q8mv and the
# hybrid ops are host-served tasks parked inside it)
@unsafe
def fwd.grp(+s: Nat, +k: Nat, +m: Model, +pos: U32, +dT: Nat, +nT: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) ->
  List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  layers.skip(s, k, U32.from_nat(s), m, m.ws(m), pos, dT, nT, st, x)

# The whole pass (layers s..s+k-1) as one host-served bang (tools/payload-pass.patch):
# the host runs each layer's linb / fulb chain and the fbn chain on host-held layer
# state; the Bend state list is not read. No CPU body: NaN, so step.diag's
# nonfinite check fires if the bang is ever not served.
@unsafe
def llb.pass_warp(+f: Nat, +ws: List<&2, Bytes()>, x: Array<F32>, y: Array<F32>, +lo: Offs, +fo: Offs,
  +c: U32, +rg: U32, +rd: U32, +s: U32, +k: U32, +eb: Bytes(), +eat: U32, +hb: Bytes(),
  +noff: U32, +ooff: U32, +nh: U32, +rwh: U32, +sh: Shape, +tb: Bytes(), +ew: U32,
  +eps: U32, +rb: U32, +rd: U32, +cap: U32) -> Array<F32>:
  Array.new(F32, 12n, K.f32_from_bits(2143289344))

@unsafe
def fwd.pass(+s: Nat, +k: Nat, +m: Model, +pos: U32, +dT: Nat, x: Array<F32>) -> Array<F32>:
  llb.pass_warp!(14n, m.ws(m), x, Array.new(F32, 12n, 0.0), m.lino(m), m.fulo(m),
    (U32.from_nat(dT) .|. (pos << 5n) : U32), sh.rg(m.sh(m)), sh.rd(m.sh(m)), U32.from_nat(s), U32.from_nat(k),
    m.ep(m), 0, m.pk(m), 0, 0, 0, 0, m.sh(m), m.tb(m), 0,
    cf.eps(m.cf(m)), cf.rb(m.cf(m)), cf.rd(m.cf(m)), cf.cap(m.cf(m)))

# The folded step (tools/payload-fold.patch): embed, the whole pass and the head
# as one bang -- the host dequantizes the token's row, runs the layers, then
# rms_final, the head GEMV and the top two into the four words.
@unsafe
def fwd.step(+m: Model, +pos: U32, +dT: Nat, +tok: U32, +fl: U32) -> Tops:
  step.tops(llb.pass_warp!(14n, m.ws(m), Array.new(F32, 12n, 0.0), Array.new(F32, 12n, 0.0),
    m.lino(m), m.fulo(m), (U32.from_nat(dT) .|. (pos << 5n) : U32), sh.rg(m.sh(m)), sh.rd(m.sh(m)),
    fl, U32.from_nat(m.nl(m)), m.ep(m), (tok * U32.and(m.erw(m), 134217727) : U32), m.pk(m), m.nb(m),
    m.ow(m), m.voc(m), m.orw(m), m.sh(m), m.tb(m), m.erw(m),
    cf.eps(m.cf(m)), cf.rb(m.cf(m)), cf.rd(m.cf(m)), cf.cap(m.cf(m))))

# the fast prefill tier's batch (tools/payload-pf.patch): cnt prompt ids in ids (as F32) from
# pos on, word 47 bits 30 and 29; the host queues them all (row 0 the embedding's base)
@unsafe
def fwd.batch(+m: Model, +pos: U32, +dT: Nat, ids: Array<F32>, +cnt: U32) -> Tops:
  step.tops(llb.pass_warp!(14n, m.ws(m), ids, Array.new(F32, 12n, 0.0),
    m.lino(m), m.fulo(m), (U32.from_nat(dT) .|. (pos << 5n) : U32), sh.rg(m.sh(m)), sh.rd(m.sh(m)),
    (1610612736 .|. cnt : U32), U32.from_nat(m.nl(m)), m.ep(m), 0, m.pk(m), m.nb(m),
    m.ow(m), m.voc(m), m.orw(m), m.sh(m), m.tb(m), m.erw(m),
    cf.eps(m.cf(m)), cf.rb(m.cf(m)), cf.rd(m.cf(m)), cf.cap(m.cf(m))))

def layers.fin(+t0: Nat, r: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>) ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>):
  do IO<List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>>:
    t1 : Nat <- IO.now()
    IO.print_err("op layers " ++ Nat.show(Nat.sub(t1, t0)))
    return r

def layers.with(k: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> Array<F32> ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>),
  r: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>) ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>):
  (st, x) = r
  k(st, x)

@unsafe
def layers.chunk(s: Nat, +k: Nat, +m: Model, +pos: U32, +dT: Nat, +nT: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>):
  do IO<List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>>:
    t0 : Nat <- IO.now()
    layers.fin(t0, (st, fwd.pass!(s, k, m, pos, dT, x)))

# c bangs of up to b layers each, from layer s (n remain)
def layers.go(c: Nat, +n: Nat, +s: Nat, +b: Nat, +m: Model, +pos: U32, +dT: Nat, +nT: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>):
  match c:
    case 1n+p:
      +k = Nat.min(n, b)
      IO.bind(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
        List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
        layers.chunk(s, k, m, pos, dT, nT, st, x),
        layers.with(st2 => x2 => layers.go(p, Nat.sub(n, k), Nat.add(s, k), b, m, pos, dT, nT, st2, x2)))
    case 0n:
      IO.pure(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>, (st, x))

# the n layers, b per bang (--lpb; b = n is the whole pass in one bang)
def layers(+n: Nat, +b: Nat, +m: Model, +pos: U32, +dT: Nat, +nT: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, x: Array<F32>) ->
  IO(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>):
  layers.go(Nat.div(Nat.sub(Nat.add(n, b), 1n), b), n, 0n, b, m, pos, dT, nT, st, x)

# the host-side clone walks x2 over device pages; only --emit nl reads the copy
def step.xx(ok: Bool, x: Array<F32>) -> Array<F32> & Array<F32>:
  match ok:
    case True{}:
      Array.clone(F32, x)
    case False{}:
      (Array.new(F32, 0n, 0.0), x)

@unsafe
def step.after(r: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
  +m: Model, +em: U32, +pos: U32, +t0: Nat) -> IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  (st2, x2) = r
  step.emit(step.xx(U32.is_eq(em, U32.from_nat(m.nl(m))), x2), st2, m, em, pos, t0)

@unsafe
def step.fold(+m: Model, +dT: Nat, +pos: U32, +tok: U32, +fl: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) ->
  IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  do IO<U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    tok2 : U32 <- step.diag(pos, fwd.step!(m, pos, dT, tok, fl))
    t1 : Nat <- IO.now()
    IO.print_err("step " ++ U32.show(pos) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return (tok2, st)

@unsafe
def step.split(+m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) ->
  IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  do IO<U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    x : Array<F32> <- I.io.time(Array<F32>, "embed", _ => embed!(m, tok))
    tb : Nat <- IO.now()
    r : List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32> <- layers(m.nl(m), b, m, pos, dT, nT, st, x)
    step.after(r, m, em, pos, t0)

@unsafe
def step.pick(fold: Bool, +m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, +fl: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) ->
  IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  match fold:
    case True{}:
      step.fold(m, dT, pos, tok, fl, st)
    case False{}:
      step.split(m, b, em, dT, nT, pos, tok, st)

# no --emit and the whole pass in one bang (--lpb >= layers): the folded step
@unsafe
def step(+m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, +steps: U32, +fl: U32) ->
  IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  step.pick(Bool.and(U32.is_eq(em, 4294967295), (U32.from_nat(b) >= U32.from_nat(m.nl(m)) : U32)),
    m, b, em, dT, nT, pos, tok, fl, st)

def run.with(k: U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit), r: U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) -> IO(Unit):
  (t, cs) = r
  k(t, cs)

# The decode run (tools/payload-catchup.patch, reports/cuda-catchup): gen's folded step asks the
# host for up to n greedy steps in one bang (word 47 bit 28, n in bits 0..27). The host serves c
# of them back to back (1 <= c <= n), printing the ids and the logits and step lines of all but
# the last exactly as this loop would; y's word 4 holds c (0, never written: one step). gen then
# prints the last step's lines and goes on from pos + c. The split path (--emit, --lpb) is one
# step a bang, as before.
@unsafe
def gen.cnt(+t: Array<F32>) -> U32:
  +c = F32.to_u32(F.got(Array.get(F32, t, 4)))
  Bool.pick(U32, U32.is_eq(c, 0), 1, c)

@unsafe
def gen.pair(+t: Array<F32>) -> Tops & U32:
  (step.tops(t), gen.cnt(t))

@unsafe
def fwd.gen(+m: Model, +pos: U32, +dT: Nat, +tok: U32, +n: U32) -> Tops & U32:
  gen.pair(llb.pass_warp!(14n, m.ws(m), Array.new(F32, 12n, 0.0), Array.new(F32, 12n, 0.0),
    m.lino(m), m.fulo(m), (U32.from_nat(dT) .|. (pos << 5n) : U32), sh.rg(m.sh(m)), sh.rd(m.sh(m)),
    (268435456 .|. n : U32), U32.from_nat(m.nl(m)), m.ep(m), (tok * U32.and(m.erw(m), 134217727) : U32), m.pk(m), m.nb(m),
    m.ow(m), m.voc(m), m.orw(m), m.sh(m), m.tb(m), m.erw(m),
    cf.eps(m.cf(m)), cf.rb(m.cf(m)), cf.rd(m.cf(m)), cf.cap(m.cf(m))))

@unsafe
def gen.after(+t0: Nat, +pos: U32, r: Tops & U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) ->
  IO(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  (top, +c) = r
  +last = ((pos + c : U32) - 1 : U32)
  do IO<U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    tok2 : U32 <- step.diag(last, top)
    t1 : Nat <- IO.now()
    IO.print_err("step " ++ U32.show(last) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return (tok2, (c, st))

@unsafe
def step.gfold(+m: Model, +dT: Nat, +pos: U32, +tok: U32, +n: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) ->
  IO(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  do IO<U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    gen.after(t0, pos, fwd.gen!(m, pos, dT, tok, n), st)

def gen.one(r: U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) -> IO(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  (t, st) = r
  IO.pure(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, (t, (1, st)))

@unsafe
def step.gpick(fold: Bool, +m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, +n: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) -> IO(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  match fold:
    case True{}:
      step.gfold(m, dT, pos, tok, n, st)
    case False{}:
      IO.bind(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, step.split(m, b, em, dT, nT, pos, tok, st), gen.one)

@unsafe
def step.g(+m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, +n: U32) ->
  IO(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  step.gpick(Bool.and(U32.is_eq(em, 4294967295), (U32.from_nat(b) >= U32.from_nat(m.nl(m)) : U32)),
    m, b, em, dT, nT, pos, tok, n, st)

# (the count twice: the steps left and the next position)
def run.with3(k: U32 -> U32 -> U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit), r: U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) -> IO(Unit):
  (t, r2) = r
  (+c, cs) = r2
  k(t, c, c, cs)

@unsafe
def gen(n: Nat, +m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>,
  +steps: U32) -> IO(Unit):
  match n:
    case 0n:
      IO.print(U32.show(tok))
    case 1n+p:
      +l = U32.from_nat(p)
      do IO<Unit>:
        IO.print(U32.show(tok))
        IO.bind(U32 & U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, Unit, step.g(m, b, em, dT, nT, pos, tok, st, (l + 1 : U32)),
          run.with3(t => c1 => c2 => st2 => gen(U32.to_nat(((l + 1 : U32) - c1 : U32)), m, b, em, dT, nT, (pos + c2 : U32), t, st2, steps)))

# the fast prefill tier's defer bit (word 47 bit 30, tools/payload-pf.patch): set on every
# prompt token with another after it. With LLB_PREFILL=fast the host queues those and answers
# 1e9 (pf_mark), the cue to send the rest but the last (up to 4096) as one batch bang
def pf.more(ts: List<&2, U32>) -> U32:
  match ts:
    case Con{t, r}:
      1073741824
    case Nil{}:
      0

def pf.len(ts: List<&2, U32>, +c: U32) -> U32:
  match ts:
    case Con{t, r}:
      pf.len(r, (c + 1 : U32))
    case Nil{}:
      c

def pf.take(+ts: List<&2, U32>) -> Nat:
  +c = pf.len(ts, 0)
  U32.to_nat(Bool.pick(U32, (c > 4096 : U32), 4096, (c - 1 : U32)))

# the first n of ts into a from slot i: (a, count) and the rest
def pf.fill(n: Nat, ts: List<&2, U32>, a: Array<F32>, +i: U32) -> (Array<F32> & U32) & List<&2, U32>:
  match n:
    case 0n:
      ((a, i), ts)
    case 1n+p:
      match ts:
        case Con{t, r}:
          pf.fill(p, r, Array.set(F32, a, i, U32.to_f32(t)), (i + 1 : U32))
        case Nil{}:
          ((a, i), Nil{})

@unsafe
def prompt.batch2(ac: Array<F32> & U32, rest: List<&2, U32>, +m: Model, +dT: Nat, +pos: U32, +nt: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, k: List<&2, U32> -> U32 -> U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit)) -> IO(Unit):
  (a, +cnt) = ac
  do IO<Unit>:
    t0 : Nat <- IO.now()
    tb : U32 <- step.diag(pos, fwd.batch!(m, pos, dT, a, cnt))
    t1 : Nat <- IO.now()
    IO.print_err("batch " ++ U32.show(pos) ++ " " ++ U32.show(cnt) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    k(rest, (pos + cnt : U32), nt, st)

@unsafe
def prompt.batch(r: (Array<F32> & U32) & List<&2, U32>, +m: Model, +dT: Nat, +pos: U32, +nt: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, k: List<&2, U32> -> U32 -> U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit)) -> IO(Unit):
  (ac, rest) = r
  prompt.batch2(ac, rest, m, dT, pos, nt, st, k)

# after a prompt step: the cue (cu) sends a batch first, then k (the prompt loop) goes on
@unsafe
def prompt.next(cu: Bool, +rest: List<&2, U32>, +m: Model, +dT: Nat, +pos: U32, +nt: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, k: List<&2, U32> -> U32 -> U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit)) -> IO(Unit):
  match cu:
    case True{}:
      prompt.batch(pf.fill(pf.take(rest), rest, Array.new(F32, 12n, 0.0), 0), m, dT, pos, nt, st, k)
    case False{}:
      k(rest, pos, nt, st)

@unsafe
def prompt.cue(+nt: U32, +rest: List<&2, U32>, +m: Model, +dT: Nat, +pos: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, k: List<&2, U32> -> U32 -> U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit)) -> IO(Unit):
  prompt.next(U32.is_eq(nt, 1000000000), rest, m, dT, pos, nt, st, k)

@unsafe
def prompt(ts: List<&2, U32>, n: Nat, +m: Model, +b: Nat, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +prev: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, +steps: U32) -> IO(Unit):
  match ts:
    case Con{t0, rest0}:
      +t = t0
      +rest = rest0
      IO.bind(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, Unit, step(m, b, em, dT, nT, pos, t, st, steps, pf.more(rest)),
        run.with(nt => st2 => prompt.cue(nt, rest, m, dT, (pos + 1 : U32), st2,
          r2 => p2 => t2 => st3 => prompt(r2, n, m, b, em, dT, nT, p2, t2, st3, steps))))
    case Nil{}:
      gen(n, m, b, em, dT, nT, pos, prev, st, steps)

# Setup
# -----

def num.of(m: Maybe<&2, U32>) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      4294967295

def cfg(xs: List<String>, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32, +lb: U32, +sp: U32, +mode: U32, +ok: Bool) ->
  List<&2, U32> & U32 & U32 & U32 & U32 & U32:
  match xs:
    case Con{a0, t}:
      +a = a0
      +fs = String.eq(a, "--steps")
      +fl = String.eq(a, "--layers")
      +fe = String.eq(a, "--emit")
      +fb = String.eq(a, "--lpb")
      +fp = String.eq(a, "--pos")
      +flag = Bool.or(fs, Bool.or(fl, Bool.or(fe, Bool.or(fb, fp))))
      +v = Bool.pick(U32, String.eq(a, ","), 4294967294, num.of(U32.read(a)))
      cfg(t,
        Bool.pick(List<&2, U32>, Bool.and(U32.is_eq(mode, 0), Bool.not(flag)), v <> ts, ts),
        Bool.pick(U32, U32.is_eq(mode, 1), v, st), Bool.pick(U32, U32.is_eq(mode, 2), v, ly),
        Bool.pick(U32, U32.is_eq(mode, 3), v, em), Bool.pick(U32, U32.is_eq(mode, 4), v, lb),
        Bool.pick(U32, U32.is_eq(mode, 5), v, sp),
        Bool.pick(U32, fs, 1, Bool.pick(U32, fl, 2, Bool.pick(U32, fe, 3, Bool.pick(U32, fb, 4, Bool.pick(U32, fp, 5, 0))))),
        Bool.and(ok, Bool.pick(Bool, flag, U32.is_eq(mode, 0), Bool.not(U32.is_eq(v, 4294967295)))))
    case Nil{}:
      (List.reverse(&2, U32, ts),
        (Bool.pick(U32, Bool.and(ok, U32.is_eq(mode, 0)), st, 0), (ly, (em, (lb, sp)))))

# every id below the vocabulary v (or the "," cut); run.chk passes 4294967294 (any id, the
# model not yet read), run.go the model's vocab_size (tools/payload-anyshape.patch)
def ok_ids.go(ts: List<&2, U32>, +v: U32, ok: Bool) -> Bool:
  match ts:
    case Con{t0, rest}:
      +t = t0
      ok_ids.go(rest, v, Bool.and(Bool.or((t < v : U32), U32.is_eq(t, 4294967294)), ok))
    case Nil{}:
      ok

def ok_ids(ts: List<&2, U32>, +v: U32) -> Bool:
  ok_ids.go(ts, v, True{})

def count_ids(ts: List<&2, U32>, n: Nat) -> Nat:
  match ts:
    case Con{t, rest}:
      count_ids(rest, 1n+n)
    case Nil{}:
      n

def usage() -> String:
  "usage: engine_qwen35 <packdir> <tok>... [--steps N] [--layers L] [--emit L] [--lpb K] [--pos K]  (N>=1, ids<vocab_size, prompt+N-1 <= 2^26, L/K <= 256)"

# ids "a b , c d": prompts cut at "," (4294967294), each from pos 0 on fresh
# states after one load; stdout "end" between them (tools/q4dp K2/K3 batches)
def cut(ts: List<&2, U32>, +acc: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match ts:
    case Con{t0, rest0}:
      +t = t0
      +rest = rest0
      Bool.pick(List<&2, U32> & List<&2, U32>, U32.is_eq(t, 4294967294),
        (List.reverse(&2, U32, acc), rest), cut(rest, t <> acc))
    case Nil{}:
      (List.reverse(&2, U32, acc), Nil{})

def multi.with(k: List<&2, U32> -> List<&2, U32> -> IO(Unit), r: List<&2, U32> & List<&2, U32>) -> IO(Unit):
  (a, z) = r
  k(a, z)

def multi.next(rest: List<&2, U32>, k: List<&2, U32> -> List<&2, U32> -> IO(Unit)) -> IO(Unit):
  match rest:
    case Con{t, r}:
      do IO<Unit>:
        IO.print("end")
        multi.with(k, cut(t <> r, Nil{}))
    case Nil{}:
      IO.pure(Unit, Unit{})

# the longest prompt (the host keeps one dT for the run)
def maxlen(ts: List<&2, U32>, +cur: U32, +best: U32) -> U32:
  match ts:
    case Con{t0, rest}:
      +t = t0
      +c = Bool.pick(U32, U32.is_eq(t, 4294967294), 0, (cur + 1 : U32))
      maxlen(rest, c, Bool.pick(U32, (c >= best : U32), c, best))
    case Nil{}:
      best

@unsafe
def multi(ps: List<&2, U32>, rest: List<&2, U32>, +m: Model, +b: Nat, +em: U32, +dT: Nat, +st: U32) -> IO(Unit):
  do IO<Unit>:
    prompt(ps, U32.to_nat((st - 1 : U32)), m, b, em, dT, (1 << dT : U32), 0, 0, states(m.nl(m), st.kd(m, b, em, dT)), st)
    multi.next(rest, a => z => multi(a, z, m, b, em, dT, st))

# SYS1-DUAL (LLB_SYS1 set, docs/sys1dual-registration.md): one load, then
# requests from S.sys1.next until EOF; each from pos 0 on fresh states (the
# multi path's reset), one dT for the run (argv "0 --steps C" sizes it)
#
# === S2 twoheads SOURCE-SPEC — P1 APPLIED (S6-P1 pass): cfg parses --pos; the
# session start pos threads srv.go -> srv.run -> srv.chk -> srv.one -> prompt's
# 8th arg (`sp`); the sys1.c latch stays as the compiled-side belt ===
# (was: UNVERIFIED-BY-REEMIT — bend toolchain down: M1 bun + M2 checkout missing) ===
# Retention across requests (same conversation = no re-prefill) is served
# request-side in tools/sys1dual/dual.c this session (retained session record:
# identical repeats are answered without an engine pass; a differing request in
# a live session is refused). The engine-side continuation delta to apply when
# re-emit works, mirroring the S2 LANE-STATE plan:
#   * srv.one's prompt call below (the 8th argument): WAS the literal 0; now
#     the threaded session start pos `sp` (dual.c emits `--pos` K on every
#     request; cfg parses it; K=0 fresh keeps the old path byte-identical).
#   * srv.go's run cfg (`:1083`) keeps dT; the request past dT is refused in
#     dual.c (KV capacity = --steps, one live session).
#   * pass_layer (host/llb_host.c S2 note): the pos==0 zero branch becomes
#     reset-only-on-explicit-reset; the `pass_pos[i] + 1 != pos` guard already
#     admits a request that starts exactly at the retained pos + 1.
# === end S2 SOURCE-SPEC ===
@unsafe
def srv.one(ok: Bool, ts: List<&2, U32>, +st: U32, +sp: U32, +m: Model, +b: Nat, +em: U32, +dT: Nat, k: Unit -> IO(Unit)) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        prompt(ts, U32.to_nat((st - 1 : U32)), m, b, em, dT, (1 << dT : U32), sp, 0, states(m.nl(m), st.kd(m, b, em, dT)), st)
        S.sys1.done()
        k(Unit{})
    case False{}:
      IO.die(Unit, 2, "sys1: bad request command")

@unsafe
def srv.chk(+ts: List<&2, U32>, +st: U32, +sp: U32, +m: Model, +b: Nat, +em: U32, +dT: Nat, k: Unit -> IO(Unit)) -> IO(Unit):
  # the request inside the run's KV capacity (was 131072; tools/payload-anyshape.patch)
  +c = cf.cap(m.cf(m))
  srv.one(Bool.and(Bool.and(ok_ids(ts, m.voc(m)), (st >= 1 : U32)), Bool.and(Bool.and((st <= (c + 1 : U32) : U32), (sp <= c : U32)), ((sp + st) <= (c + 1 : U32) : U32))), ts, st, sp, m, b, em, dT, k)

@unsafe
def srv.run(c: List<&2, U32> & U32 & U32 & U32 & U32 & U32, +m: Model, +b: Nat, +em: U32, +dT: Nat, k: Unit -> IO(Unit)) -> IO(Unit):
  (ts, sl) = c
  (st, s2) = sl
  (s3, s4) = s2
  (s5, s6) = s4
  (s7, sp) = s6
  srv.chk(ts, st, sp, m, b, em, dT, k)

@unsafe
def srv.go(q: List<String>, +m: Model, +b: Nat, +em: U32, +dT: Nat, k: Unit -> IO(Unit)) -> IO(Unit):
  match q:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{w, ws}:
      srv.run(cfg(w <> ws, Nil{}, 0, 0, 4294967295, 64, 0, 0, True{}), m, b, em, dT, k)

@unsafe
def srv(+dir: String, +m: Model, +b: Nat, +em: U32, +dT: Nat) -> IO(Unit):
  do IO<Unit>:
    # the KV capacity in positions (prompt + steps; was 2^dT, tools/payload-anyshape.patch)
    q : List<String> <- S.sys1.next(dir, cf.cap(m.cf(m)))
    srv.go(q, m, b, em, dT, u => srv(dir, m, b, em, dT))

@unsafe
def srv.or(on: Bool, +dir: String, ts: List<&2, U32>, +m: Model, +b: Nat, +em: U32, +dT: Nat, +st: U32) -> IO(Unit):
  match on:
    case True{}:
      srv(dir, m, b, em, dT)
    case False{}:
      multi.with(a => z => multi(a, z, m, b, em, dT, st), cut(ts, Nil{}))

@unsafe
def run.go(ok: Bool, +dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32, +lb: U32) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +m0 : Model <- mf.model(dir, ly, em, lb)
        mf.check(ok_ids(ts, m.voc(m0)), "a prompt id >= vocab_size")
        # the KV capacity: the longest prompt + steps - 1 positions (2^dT rounds it up; the
        # host allocates the slots by position up to it, tools/payload-anyshape.patch)
        +m : Model = m.with_cap(m0, (maxlen(ts, 0, 0) + st - 1 : U32))
        # the split path (--emit / --lpb < layers) has Q8_0-only embed and head
        mf.check(Bool.or(Bool.and(U32.is_eq(m.orw(m), q8rw(4096)), U32.is_eq(m.erw(m), q8rw(4096))),
          Bool.and(U32.is_eq(em, 4294967295), (lb >= U32.from_nat(m.nl(m)) : U32))),
          "quantized head/embedding needs the folded step (no --emit, --lpb >= layers)")
        +dT : Nat = lg(26n, 0n, (maxlen(ts, 0, 0) + st - 1 : U32))
        se : Result<&1, &1, U32 & String, String> <- IO.get_env("LLB_SYS1")
        srv.or(G.gguf.on(se), dir, ts, m, U32.to_nat(lb), em, dT, st)
        t1 : Nat <- IO.now()
        IO.print("wall_ms " ++ Nat.show(Nat.sub(t1, t0)))
    case False{}:
      IO.die(Unit, 2, usage())

@unsafe
def run.chk(dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32, +lb: U32) -> IO(Unit):
  +nt = U32.from_nat(count_ids(ts, 0n))
  # tools/payload-anyshape.patch: prompt + steps - 1 <= 2^26 positions (was 2^17), layers 256
  +ok = Bool.and(ok_ids(ts, 4294967294), (nt >= 1 : U32))
  +ok1 = Bool.and(ok, (st >= 1 : U32))
  +ok2 = Bool.and(ok1, (st <= 67108864 : U32))
  +ok3 = Bool.and(ok2, (ly <= 256 : U32))
  +ok4 = Bool.and(ok3, Bool.or(U32.is_eq(em, 4294967295), (em <= 256 : U32)))
  +ok5 = Bool.and(ok4, (lb >= 1 : U32))
  +ok6 = Bool.and(ok5, (lb <= 256 : U32))
  run.go(Bool.and(ok6, (nt <= (67108865 - st : U32) : U32)), dir, ts, st, ly, em, lb)

@unsafe
def run.sl(dir: String, ts: List<&2, U32>, sl: U32 & U32 & U32 & U32 & U32) -> IO(Unit):
  (st, lf) = sl
  (ly, ef) = lf
  (em, ff) = ef
  (lb, rest) = ff
  run.chk(dir, ts, st, ly, em, lb)

@unsafe
def run(dir: String, c: List<&2, U32> & U32 & U32 & U32 & U32 & U32) -> IO(Unit):
  (ts, sl) = c
  run.sl(dir, ts, sl)

@unsafe
def main.go(args: List<String>) -> IO(Unit):
  match args:
    case Con{dir, rest}:
      run(dir, cfg(rest, Nil{}, 8, 0, 4294967295, 64, 0, 0, True{}))
    case Nil{}:
      IO.die(Unit, 2, usage())

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
