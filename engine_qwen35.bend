# engine_qwen35 <packdir> <tok>... [--steps N] [--layers L] [--emit <layer>]
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

# a layer's tensor offsets, relative to its layer file. Linear layers fill the
# first half; full layers the second. `an` (attn_norm) and the FFN/post fields
# are shared across both shapes.
type Offs is Data:
  Offs{an: U32, post: U32, fg: U32, fu: U32, fd: U32,
    qkv: U32, gate: U32, sb: U32, sa: U32, ssa: U32, dt: U32, cv: U32, sn: U32, so: U32,
    aq: U32, ak: U32, av: U32, aqn: U32, akn: U32, ao: U32}

type Model is Data:
  Model{nl: Nat, lin: Offs, ful: Offs, ws: List<&2, Bytes()>, pk: Bytes(), ep: Bytes(), ooff: U32, noff: U32}

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
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      nl

def m.lino(m: Model) -> Offs:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      lin

def m.fulo(m: Model) -> Offs:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      ful

def m.w(m: Model, i: U32) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      ws.at(U32.to_nat(i), ws)

# pack.bin output+output_norm resident region (pk, ooff/noff inside it) and the
# token_embd region (ep, its rows at tok*4352)
def m.pk(m: Model) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      pk

def m.ep(m: Model) -> Bytes():
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      ep

def m.ow(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      ooff

def m.nb(m: Model) -> U32:
  match m:
    case Model{nl, lin, ful, ws, pk, ep, ooff, noff}:
      noff

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

# one layer file; linear (232178432 B) and full (222857216 B) differ
def ws.layer(+dir: String, +i: U32, +full: Bool, +sz: U32) -> IO(Bytes()):
  ws.load(dir ++ "/layers/blk." ++ U32.show(i) ++ ".bin", sz)

def ws.layers(n: Nat, +dir: String, +i: U32, +sz: U32, +full: Bool, +szf: U32) -> IO(List<&2, Bytes()>):
  match n:
    case 0n:
      IO.pure(List<&2, Bytes()>, Nil{})
    case 1n+p:
      do IO<List<&2, Bytes()>>:
        w : Bytes() <- ws.layer(dir, i, U32.is_eq(U32.and((i + 1 : U32), 3), 0),
          Bool.pick(U32, U32.is_eq(U32.and((i + 1 : U32), 3), 0), szf, sz))
        rest : List<&2, Bytes()> <- ws.layers(p, dir, (i + 1 : U32), sz, full, szf)
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

def mf.config(+s: String, +key: String, want: String) -> IO(Unit):
  mf.check(String.eq(mf.value(s, "config", key), want), key)

def mf.validate(+s: String) -> IO(Unit):
  do IO<Unit>:
    mf.check(String.eq(mf.scalar(s, "format"), "\"llama-bend-pack-q8/v0\""), "format")
    mf.check(String.eq(mf.scalar(s, "dtype"), "\"q8_0-rowsplit\""), "dtype")
    mf.config(s, "architecture", "\"qwen3_5\"")
    mf.config(s, "n_layers", "32")
    mf.config(s, "n_heads", "16")
    mf.config(s, "n_kv_heads", "4")
    mf.config(s, "d_model", "4096")
    mf.config(s, "head_dim", "256")
    mf.config(s, "d_ff", "12288")
    mf.config(s, "vocab_size", "248320")
    mf.config(s, "full_attention_interval", "4")

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
def mf.offl.lin(+s: String, +p: String) -> IO(Offs):
  do IO<Offs>:
    +an : Nat <- mf.off_nat(s, p ++ "attn_norm.weight")
    +post : Nat <- mf.off_nat(s, p ++ "post_attention_norm.weight")
    +fg : Nat <- mf.off_nat(s, p ++ "ffn_gate.weight")
    +fu : Nat <- mf.off_nat(s, p ++ "ffn_up.weight")
    +fd : Nat <- mf.off_nat(s, p ++ "ffn_down.weight")
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
def mf.offl.ful(+s: String, +p: String) -> IO(Offs):
  do IO<Offs>:
    +an : Nat <- mf.off_nat(s, p ++ "attn_norm.weight")
    +post : Nat <- mf.off_nat(s, p ++ "post_attention_norm.weight")
    +fg : Nat <- mf.off_nat(s, p ++ "ffn_gate.weight")
    +fu : Nat <- mf.off_nat(s, p ++ "ffn_up.weight")
    +fd : Nat <- mf.off_nat(s, p ++ "ffn_down.weight")
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

def mf.model(+dir: String, +ly: U32) -> IO(Model):
  do IO<Model>:
    f : File <- IO.try(File, File.open(dir ++ "/manifest.json", "r"))
    r : File & Result<&1, &1, U32 & String, String> <- File.read(f, 16777216)
    +s : String <- mf.text(r)
    mf.validate(s)
    +nl : U32 <- mf.need("n_layers", mf.num(s, "config", "n_layers"))
    +emb : U32 <- mf.need("token_embd offset", mf.num(s, "token_embd.weight", "offset"))
    +onorm : U32 <- mf.need("output_norm offset", mf.num(s, "output_norm.weight", "offset"))
    +out : U32 <- mf.need("output offset", mf.num(s, "output.weight", "offset"))
    mf.check(Bool.and((emb <= 3225419776 : U32), Bool.and((out <= 3225419776 : U32),
      (onorm <= 4294959104 : U32))), "head offset exceeds U32")
    lin : Offs <- mf.offl.lin(s, "blk.0.")
    ful : Offs <- mf.offl.ful(s, "blk.3.")
    +n : Nat = U32.to_nat(Bool.pick(U32, U32.is_eq(ly, 0), nl, ly))
    # output.weight and output_norm sit contiguously at the front of pack.bin;
    # token_embd follows. Bytes are capped at 2^31, so read the head region in
    # two parts (pk: output+norm from 0, ep: token_embd after a skip).
    +out0 : Nat = U32.to_nat(out)
    +norm0 : Nat = U32.to_nat(onorm)
    +emb0 : Nat = U32.to_nat(emb)
    +headn : Nat = Nat.max(Nat.add(out0, 1080688640n), Nat.add(norm0, 16384n))
    mf.check(Nat.is_le(headn, 2147483648n), "output/norm region exceeds 2^31")
    t0 : Nat <- IO.now()
    ws : List<&2, Bytes()> <- ws.layers(n, dir, 0, 232178432, False{}, 222857216)
    pk : Bytes() <- ws.load(dir ++ "/pack.bin", U32.from_nat(headn))
    ep : Bytes() <- ws.load_skip(dir ++ "/pack.bin", emb0, 1080688640)
    t1 : Nat <- IO.now()
    IO.print_err("resident " ++ Nat.show(Nat.add(Nat.add(Nat.mul(n, 232178432n), headn), 1080688640n))
      ++ " bytes in " ++ Nat.show(Nat.sub(t1, t0)) ++ " ms (q8_0, emitted cpu)")
    return Model{n, lin, ful, ws, pk, ep, U32.from_nat(out0), U32.from_nat(norm0)}

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

@unsafe
def q8mv(+f: Nat, +wt: Bytes(), +x: Array<F32>, +off: U32, +n: U32, +d: Nat, +cols: U32) -> Array<F32>:
  F.q8.gemv(f, wt, x, Array.new(F32, d, 0.0), off, n, cdepth(cols), ((cols / 32 : U32) * 34 : U32))

@unsafe
def q8mv.inf(+f: Nat, +wt: Bytes(), +x: Array<F32>, +off: U32, +n: U32, +d: Nat, +cols: U32) -> Array<F32>:
  F.q8.gemv(f, wt, x, Array.new(F32, d, K.neg_inf()), off, n, cdepth(cols), ((cols / 32 : U32) * 34 : U32))

@unsafe
def vec(+wt: Bytes(), +off: U32, +d: Nat) -> Array<F32>:
  F.load(6n, wt, off, d)

# State (one 6-tuple of arrays per layer: conv h0,h1,h2, S, K, V)
# ----------------------------------------------------------------
# A Data record cannot hold Array fields in this compiler, so a layer's state
# is a right-nested tuple; the six are tied in one type expression.

def st.new(+dT: Nat) -> Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>:
  +cd = Nat.add(dT, 10n)
  (Array.new(F32, 13n, 0.0), (Array.new(F32, 13n, 0.0), (Array.new(F32, 13n, 0.0),
    (Array.new(F32, 19n, 0.0), (Array.new(F32, cd, 0.0), Array.new(F32, cd, 0.0))))))

def states(n: Nat, +dT: Nat) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      st.new(dT) <> states(p, dT)

# One linear (gated delta net) layer: returns the updated state and the
# attention block's contribution (Wo y), to be added to x by the caller.
@unsafe
def lin.gd(gd: Array<F32> & Array<F32>, z: Array<F32>, +wt: Bytes(), +wsn: U32, +wso: U32, h1: Array<F32>, h2: Array<F32>,
  qkva: Array<F32>, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (gout, s2) = gd
  +normed = K.gated_rmsnorm(5n, gout, vec(wt, wsn, 7n), z, 128.0)
  +out0 = q8mv(14n, wt, normed, wso, 4096, 12n, 4096)
  ((h1, (h2, (qkva, (s2, (kc, vc))))), out0)

@unsafe
def lin.y(y: Array<F32>, s: Array<F32>, +z: Array<F32>, +gp: Array<F32>, +beta: Array<F32>, +wt: Bytes(), +wsn: U32,
  +wso: U32, h1: Array<F32>, h2: Array<F32>, qkva: Array<F32>, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  match y:
    case ANode{ANode{yq0, yk0}, yv}:
      +qt16 = K.l2norm(4n, yq0, F32.sqrt(128.0))
      +kt16 = K.l2norm(4n, yk0, 1.0)
      +qt = K.both(Array.clone(F32, qt16), x => x, w => w)
      +kt = K.both(Array.clone(F32, kt16), x => x, w => w)
      lin.gd(K.gated_delta(5n, s, qt, kt, yv, K.expmap(gp), beta), z, wt, wsn, wso, h1, h2, qkva, kc, vc)
    case _:
      ((h1, (h2, (qkva, (s, (kc, vc))))), ALeaf{0.0})

@unsafe
def lin.conv(qkc: Array<F32> & Array<F32>, h1c: Array<F32> & Array<F32>, h2c: Array<F32> & Array<F32>,
  h0: Array<F32>, cw: Array<F32>, s: Array<F32>, z: Array<F32>, gp: Array<F32>, beta: Array<F32>,
  +wt: Bytes(), +wsn: U32, +wso: U32, kc: Array<F32>, vc: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (qkcx, qkcs) = qkc
  (h1a, h1b) = h1c
  (h2a, h2b) = h2c
  +y = K.conv1d(13n, cw, qkcx, h0, h1a, h2a)
  lin.y(y, s, z, gp, beta, wt, wsn, wso, h1b, h2b, qkcs, kc, vc)

@unsafe
def lin(+wt: Bytes(), +wqkv: U32, +wgate: U32, +wb: U32, +wa: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +h: Array<F32>) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (h0, (h1, (h2, (s, (kc, vc))))) = e
  +qkva = q8mv(14n, wt, h, wqkv, 8192, 13n, 4096)
  +z = q8mv(14n, wt, h, wgate, 4096, 12n, 4096)
  +bpre = q8mv(14n, wt, h, wb, 32, 5n, 4096)
  +apre = q8mv(14n, wt, h, wa, 32, 5n, 4096)
  +beta = K.sigmoid(bpre)
  +gp = K.vmul(K.softplus(K.vadd(apre, vec(wt, wdt, 5n))), vec(wt, wsa, 5n))
  +cw = vec(wt, wcv, 15n)
  lin.conv(Array.clone(F32, qkva), Array.clone(F32, h1), Array.clone(F32, h2), h0, cw, s, z, gp, beta, wt, wsn, wso, kc, vc)

# One full-attention layer
@unsafe
def ful.qg(qgs: Array<F32> & Array<F32>, +wt: Bytes(), +wak: U32, +wav: U32, +wakn: U32, +wao: U32,
  +h: Array<F32>, h0: Array<F32>, h1: Array<F32>, h2: Array<F32>, s: Array<F32>, kc: Array<F32>, vc: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (q, gate) = qgs
  +kb = q8mv(14n, wt, h, wak, 1024, 10n, 4096)
  +kn = K.kr.go(2n, kb, vec(wt, wakn, 8n), 256, 64, U32.to_f32(pos), 10000000.0)
  +vb = q8mv(14n, wt, h, wav, 1024, 10n, 4096)
  +kc2 = K.put(dT, kc, nT, 0, pos, kn)
  +vc2 = K.put(dT, vc, nT, 0, pos, vb)
  +sc = (1.0 / F32.sqrt(256.0) : F32)
  +scores = K.scores4(4n, 16, 0, dT, sc, q, kc2)
  +ps = K.probs(4n, nT, pos, scores)
  +attn = K.weighted4(4n, 16, 0, dT, ps, vc2)
  +gated = K.output_gate(attn, gate)
  +out0 = q8mv(14n, wt, gated, wao, 4096, 12n, 4096)
  ((h0, (h1, (h2, (s, (kc2, vc2))))), out0)

@unsafe
def ful(+wt: Bytes(), +waq: U32, +wak: U32, +wav: U32, +waqn: U32, +wakn: U32, +wao: U32,
  e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +h: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (h0, (h1, (h2, (s, (kc, vc))))) = e
  +qgn = q8mv(14n, wt, h, waq, 8192, 13n, 4096)
  +qg2 = K.qg.go(4n, qgn, vec(wt, waqn, 8n), 256, 64, U32.to_f32(pos), 10000000.0)
  ful.qg(K.split_qg.go(4n, qg2), wt, wak, wav, wakn, wao, h, h0, h1, h2, s, kc, vc, pos, dT, nT)

# the attention block (attn_norm + the right layer type): (state, contribution)
@unsafe
def attn.go(full: Bool, +wt: Bytes(), h: Array<F32>, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +waq: U32, +wak: U32, +wav: U32, +waqn: U32,
  +wakn: U32, +wao: U32, +wqkv: U32, +wgate: U32, +wb: U32, +wa: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  match full:
    case True{}:
      ful(wt, waq, wak, wav, waqn, wakn, wao, e, h, pos, dT, nT)
    case False{}:
      lin(wt, wqkv, wgate, wb, wa, wsa, wdt, wcv, wsn, wso, e, h)

# one layer: attention + residual + post norm + SwiGLU FFN + residual
@unsafe
def lay.res(ar: (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>, x: Array<F32>, +wt: Bytes(), +post: U32, +fg: U32, +fu: U32, +fd: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  (e2, a) = ar
  +x1 = K.vadd(x, a)
  +h2 = K.rms_norm(x1, vec(wt, post, 12n), 4096.0)
  +g = q8mv(14n, wt, h2, fg, 12288, 14n, 4096)
  +u = q8mv(14n, wt, h2, fu, 12288, 14n, 4096)
  +a2 = K.vmul(K.silu(g), u)
  +dn = q8mv(14n, wt, a2, fd, 4096, 12n, 12288)
  (e2, K.vadd(x1, dn))

@unsafe
def lay.offs(o: Offs, +m: Model, +i: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, +x: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  Offs{an, post, fg, fu, fd, qkv, gate, sb, sa, ssa, dt, cv, sn, so, aq, ak, av, aqn, akn, ao} = o
  +full = U32.is_eq(U32.and((i + 1 : U32), 3), 0)
  +wt = m.w(m, i)
  +h = K.rms_norm(x, vec(wt, an, 12n), 4096.0)
  lay.res(attn.go(full, wt, h, e, aq, ak, av, aqn, akn, ao, qkv, gate, sb, sa, ssa, dt, cv, sn, so, pos, dT, nT),
    x, wt, post, fg, fu, fd)

@unsafe
def lay(+m: Model, +i: U32, e: Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>, x: Array<F32>, +pos: U32, +dT: Nat, +nT: U32) -> (Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>) & Array<F32>:
  +full = U32.is_eq(U32.and((i + 1 : U32), 3), 0)
  lay.offs(Bool.pick(Offs, full, m.fulo(m), m.lino(m)), m, i, e, x, pos, dT, nT)

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
def layers(n: Nat, +i: U32, +m: Model, +st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, +pos: U32, +dT: Nat, +nT: U32, x: Array<F32>) -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>:
  match n st:
    case 1n+p0 Con{s0, t}:
      layers.bind(lay(m, i, s0, x, pos, dT, nT),
        e2 => x2 => layers.push(layers(p0, (i + 1 : U32), m, t, pos, dT, nT, x2), e2))
    case _ _:
      (st, x)



# Head
# ----

type Tops is Data:
  Tops{v: F32, i: U32, w: F32, j: U32}

def step.ranked(v: F32, i: U32, second: F32 & U32) -> Tops:
  (w, j) = second
  Tops{v, i, w, j}

@unsafe
def step.second(first: F32 & U32, z: Array<F32>) -> Tops:
  (v, +i) = first
  step.ranked(v, i, K.argmax(K.put(18n, z, 262144, 0, i, ALeaf{K.neg_inf()}), 262144, 0))

@unsafe
def tops(+z: Array<F32>) -> Tops:
  step.second(K.argmax(z, 262144, 0), z)

@unsafe
def head(+m: Model, x: Array<F32>) -> Tops:
  +h = K.rms_norm(x, vec(m.pk(m), m.nb(m), 12n), 4096.0)
  +z = q8mv.inf(14n, m.pk(m), h, m.ow(m), 248320, 18n, 4096)
  tops(z)

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
    top : Tops <- I.io.time(Tops, "head", _ => head(m, xb))
    tok2 : U32 <- step.diag(pos, top)
    t1 : Nat <- IO.now()
    IO.print_err("step " ++ U32.show(pos) ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return (tok2, st2)

@unsafe
def step.after(r: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>,
  +m: Model, +em: U32, +pos: U32, +t0: Nat) -> IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  (st2, x2) = r
  step.emit(Array.clone(F32, x2), st2, m, em, pos, t0)

@unsafe
def step(+m: Model, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>,
  +steps: U32) -> IO(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>):
  do IO<U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>>:
    t0 : Nat <- IO.now()
    x : Array<F32> <- I.io.time(Array<F32>, "embed", _ => embed(m, tok))
    r : List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32> <- I.io.time(List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> & Array<F32>, "layers", _ =>
      layers(m.nl(m), 0, m, st, pos, dT, nT, x))
    step.after(r, m, em, pos, t0)

def run.with(k: U32 -> List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)> -> IO(Unit), r: U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>) -> IO(Unit):
  (t, cs) = r
  k(t, cs)

@unsafe
def gen(n: Nat, +m: Model, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +tok: U32, st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>,
  +steps: U32) -> IO(Unit):
  match n:
    case 0n:
      IO.print(U32.show(tok))
    case 1n+p:
      do IO<Unit>:
        IO.print(U32.show(tok))
        IO.bind(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, Unit, step(m, em, dT, nT, pos, tok, st, steps),
          run.with(t => st2 => gen(p, m, em, dT, nT, (pos + 1 : U32), t, st2, steps)))

@unsafe
def prompt(ts: List<&2, U32>, n: Nat, +m: Model, +em: U32, +dT: Nat, +nT: U32, +pos: U32, +prev: U32,
  st: List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, +steps: U32) -> IO(Unit):
  match ts:
    case Con{t, rest}:
      IO.bind(U32 & List<(Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32> & Array<F32>)>, Unit, step(m, em, dT, nT, pos, t, st, steps),
        run.with(nt => st2 => prompt(rest, n, m, em, dT, nT, (pos + 1 : U32), nt, st2, steps)))
    case Nil{}:
      gen(n, m, em, dT, nT, pos, prev, st, steps)

# Setup
# -----

def num.of(m: Maybe<&2, U32>) -> U32:
  match m:
    case Some{v}:
      v
    case None{}:
      4294967295

def cfg(xs: List<String>, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32, +mode: U32, +ok: Bool) ->
  List<&2, U32> & U32 & U32 & U32:
  match xs:
    case Con{a0, t}:
      +a = a0
      +fs = String.eq(a, "--steps")
      +fl = String.eq(a, "--layers")
      +fe = String.eq(a, "--emit")
      +flag = Bool.or(fs, Bool.or(fl, fe))
      +v = num.of(U32.read(a))
      cfg(t,
        Bool.pick(List<&2, U32>, Bool.and(U32.is_eq(mode, 0), Bool.not(flag)), v <> ts, ts),
        Bool.pick(U32, U32.is_eq(mode, 1), v, st), Bool.pick(U32, U32.is_eq(mode, 2), v, ly),
        Bool.pick(U32, U32.is_eq(mode, 3), v, em),
        Bool.pick(U32, fs, 1, Bool.pick(U32, fl, 2, Bool.pick(U32, fe, 3, 0))),
        Bool.and(ok, Bool.pick(Bool, flag, U32.is_eq(mode, 0), Bool.not(U32.is_eq(v, 4294967295)))))
    case Nil{}:
      (List.reverse(&2, U32, ts), (Bool.pick(U32, Bool.and(ok, U32.is_eq(mode, 0)), st, 0), (ly, em)))

def ok_ids.go(ts: List<&2, U32>, ok: Bool) -> Bool:
  match ts:
    case Con{t, rest}:
      ok_ids.go(rest, Bool.and((t < 248320 : U32), ok))
    case Nil{}:
      ok

def ok_ids(ts: List<&2, U32>) -> Bool:
  ok_ids.go(ts, True{})

def count_ids(ts: List<&2, U32>, n: Nat) -> Nat:
  match ts:
    case Con{t, rest}:
      count_ids(rest, 1n+n)
    case Nil{}:
      n

def usage() -> String:
  "usage: engine_qwen35 <packdir> <tok>... [--steps N] [--layers L] [--emit L]  (N>=1, ids<248320)"

@unsafe
def run.go(ok: Bool, +dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +m : Model <- mf.model(dir, ly)
        +dT : Nat = lg(17n, 0n, (U32.from_nat(count_ids(ts, 0n)) + st - 1 : U32))
        prompt(ts, U32.to_nat((st - 1 : U32)), m, em, dT, (1 << dT : U32), 0, 0,
          states(m.nl(m), dT), st)
        t1 : Nat <- IO.now()
        IO.print("wall_ms " ++ Nat.show(Nat.sub(t1, t0)))
    case False{}:
      IO.die(Unit, 2, usage())

@unsafe
def run.chk(dir: String, +ts: List<&2, U32>, +st: U32, +ly: U32, +em: U32) -> IO(Unit):
  +nt = U32.from_nat(count_ids(ts, 0n))
  run.go(Bool.and(Bool.and(ok_ids(ts), (nt >= 1 : U32)),
    Bool.and(Bool.and((st >= 1 : U32), (st <= 131072 : U32)),
      Bool.and((ly <= 32 : U32), Bool.and(Bool.or(U32.is_eq(em, 4294967295), (em <= 32 : U32)),
        (nt <= (131073 - st : U32) : U32))))),
    dir, ts, st, ly, em)

@unsafe
def run.sl(dir: String, ts: List<&2, U32>, sl: U32 & U32 & U32) -> IO(Unit):
  (st, lf) = sl
  (ly, em) = lf
  run.chk(dir, ts, st, ly, em)

@unsafe
def run(dir: String, c: List<&2, U32> & U32 & U32 & U32) -> IO(Unit):
  (ts, sl) = c
  run.sl(dir, ts, sl)

@unsafe
def main.go(args: List<String>) -> IO(Unit):
  match args:
    case Con{dir, rest}:
      run(dir, cfg(rest, Nil{}, 8, 0, 4294967295, 0, True{}))
    case Nil{}:
      IO.die(Unit, 2, usage())

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
