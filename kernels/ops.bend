# llama-bend v0 kernels: pure F32 ops over Array<F32>, no IO.
#
# A vector is an Array<F32> of 2^d leaves; a node's left child holds the lower
# half of the indices, so ANode{l, r} splits [lo, lo+n) at lo+n/2.
#
# Op order is fixed here, not by a scheduler: every reduction is the balanced
# binary tree over that split (leaf pairs (2i, 2i+1) first, then pairs of
# pairs, ...). The two halves of a node are independent parallel calls, so a
# perf pass may run them anywhere and on any number of cores without changing
# one bit of output. Transcendentals are Base's exp/cos/sin/pow, evaluated in F64 (exp32..).
import Base

# Bits
# ----

def f32_from_bits(w: U32) -> F32:
  match w:
    case U32{d}:
      F32{d}

def f32_of(b0: U32, b1: U32, b2: U32, b3: U32) -> F32:
  f32_from_bits((b0 .|. (b1 << 8n) .|. (b2 << 16n) .|. (b3 << 24n) : U32))

def neg_inf() -> F32:
  f32_from_bits(4286578688)

# f32(1e-6) = 0x358637BD, the GGUF's rms_norm_eps, taken bit-exact
def eps() -> F32:
  f32_from_bits(897988541)

def is_nan(+x: F32) -> Bool:
  F32.is_ne(x, x)

# Transcendentals through F64, by name: host C evaluates F32.exp(x) as
# (f32)exp((double)x), but NVRTC compiles the same line as C++, where exp(float)
# is CUDA's expf (up to 2 ulp). Through F64 both lanes call the double function
# and round once; host bits are unchanged.
def exp32(x: F32) -> F32:
  F64.to_f32(F64.exp(F32.to_f64(x)))

def cos32(x: F32) -> F32:
  F64.to_f32(F64.cos(F32.to_f64(x)))

def sin32(x: F32) -> F32:
  F64.to_f32(F64.sin(F32.to_f64(x)))

def pow32(x: F32, y: F32) -> F32:
  F64.to_f32(F64.pow(F32.to_f64(x), F32.to_f64(y)))

# Building a vector from little-endian bytes
# ------------------------------------------

# Everything here is tail-recursive (the JS lane has a machine stack): each
# pass reverses the list, so the orientation flips level by level.

# exactly n leaves, 4 bytes each, then pad once the bytes run out; reversed
def leaves(n: Nat, +pad: F32, bs: List<&2, U32>, acc: List<Array<F32>>) ->
  List<Array<F32>>:
  match n:
    case 0n:
      acc
    case 1n+k:
      match bs:
        case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
          leaves(k, pad, t, ALeaf{f32_of(b0, b1, b2, b3)} <> acc)
        case _:
          leaves(k, pad, Nil{}, ALeaf{pad} <> acc)

# rev: the list runs from high index to low, so b comes before a
def node(rev: Bool, a: Array<F32>, b: Array<F32>) -> Array<F32>:
  match rev:
    case True{}:
      ANode{b, a}
    case False{}:
      ANode{a, b}

def pair(+rev: Bool, xs: List<Array<F32>>, acc: List<Array<F32>>) ->
  List<Array<F32>>:
  match xs:
    case Con{a, Con{b, t}}:
      pair(rev, t, node(rev, a, b) <> acc)
    case _:
      acc

def pairs(d: Nat, +rev: Bool, xs: List<Array<F32>>) -> List<Array<F32>>:
  match d:
    case 0n:
      xs
    case 1n+p:
      pairs(p, Bool.not(rev), pair(rev, xs, Nil{}))

def root(xs: List<Array<F32>>) -> Array<F32>:
  match xs:
    case Con{a, _}:
      a
    case Nil{}:
      ALeaf{0.0}

def vec(+d: Nat, pad: F32, bs: List<&2, U32>) -> Array<F32>:
  root(pairs(d, True{}, leaves(U32.to_nat((1 << d : U32)), pad, bs, Nil{})))

# Reductions
# ----------

def tsum(a: Array<F32>) -> F32:
  match a:
    case ALeaf{x}:
      x
    case ANode{l, r}:
      x y = tsum(l) tsum(r)
      (x + y : F32)

# rms_norm
# --------
# h[i] = (x[i] * scale) * w[i],  scale = 1 / sqrt(tsum(x*x) / n + eps)
# (llama.cpp's shape: rms_norm scales, then a separate mul by the weight.)

def sumsq.join(a: F32 & Array<F32>, b: F32 & Array<F32>) -> F32 & Array<F32>:
  (sa, xa) = a
  (sb, xb) = b
  ((sa + sb : F32), ANode{xa, xb})

# the sum of squares, handing x back
def sumsq(a: Array<F32>) -> F32 & Array<F32>:
  match a:
    case ALeaf{+v}:
      ((v * v : F32), ALeaf{v})
    case ANode{l, r}:
      x y = sumsq(l) sumsq(r)
      sumsq.join(x, y)

def scale_mul(x: Array<F32>, w: Array<F32>, +s: F32) -> Array<F32>:
  match x w:
    case ALeaf{v} ALeaf{g}:
      ALeaf{((v * s : F32) * g : F32)}
    case ANode{xl, xr} ANode{wl, wr}:
      l r = scale_mul(xl, wl, s) scale_mul(xr, wr, s)
      ANode{l, r}
    case _ _:
      ALeaf{0.0}

def rms_norm.go(n: F32, w: Array<F32>, sx: F32 & Array<F32>) -> Array<F32>:
  (ss, x) = sx
  scale_mul(x, w, (1.0 / F32.sqrt(((ss / n : F32) + eps() : F32)) : F32))

def rms_norm(x: Array<F32>, w: Array<F32>, n: F32) -> Array<F32>:
  rms_norm.go(n, w, sumsq(x))

# rope (llama.cpp NORM: adjacent pairs within a head)
# ---------------------------------------------------
# pair i of a head (elements 2i, 2i+1), head_dim 128, base 5e6:
#   theta = pos * pow(5e6, -(2i)/128)
#   (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)

def rope.theta(i: U32, pos: F32) -> F32:
  (pos * pow32(5000000.0, (F32.neg(U32.to_f32((2 * i : U32))) / 128.0 : F32)) : F32)

def rope.rot(+a: F32, +b: F32, +c: F32, +s: F32) -> Array<F32>:
  ANode{ALeaf{((a * c : F32) - (b * s : F32) : F32)},
        ALeaf{((a * s : F32) + (b * c : F32) : F32)}}

def rope.pair(a: F32, b: F32, +t: F32) -> Array<F32>:
  rope.rot(a, b, cos32(t), sin32(t))

# n: leaves under this node; lo: index of its first leaf
def rope(x: Array<F32>, +n: U32, +lo: U32, +pos: F32) -> Array<F32>:
  match x:
    case ANode{ALeaf{a}, ALeaf{b}}:
      rope.pair(a, b, rope.theta(((lo >> 1n) .&. 63 : U32), pos))
    case ANode{l, r}:
      +h = (n >> 1n : U32)
      xl xr = rope(l, h, lo, pos) rope(r, h, (lo + h : U32), pos)
      ANode{xl, xr}
    case ALeaf{v}:
      ALeaf{v}

# softmax (causal, stable)
# ------------------------
# slots t > pos are masked to -inf; m = tree max; e = exp(s - m) (masked -> 0);
# p = e / tsum(e)  (a true division, not a reciprocal multiply)

def mmax.join(a: F32 & Array<F32>, b: F32 & Array<F32>) -> F32 & Array<F32>:
  (ma, xa) = a
  (mb, xb) = b
  (F32.max(ma, mb), ANode{xa, xb})

def mmax.leaf(+v: F32) -> F32 & Array<F32>:
  (v, ALeaf{v})

# the masked max, handing back the masked scores
def mmax(a: Array<F32>, +n: U32, +lo: U32, +pos: U32) -> F32 & Array<F32>:
  match a:
    case ALeaf{v}:
      mmax.leaf(Bool.pick(F32, (lo <= pos : U32), v, neg_inf()))
    case ANode{l, r}:
      +h = (n >> 1n : U32)
      x y = mmax(l, h, lo, pos) mmax(r, h, (lo + h : U32), pos)
      mmax.join(x, y)

def exps(a: Array<F32>, +m: F32) -> F32 & Array<F32>:
  match a:
    case ALeaf{v}:
      mmax.leaf(exp32((v - m : F32)))
    case ANode{l, r}:
      x y = exps(l, m) exps(r, m)
      sumsq.join(x, y)

def divs(a: Array<F32>, +z: F32) -> Array<F32>:
  match a:
    case ALeaf{v}:
      ALeaf{(v / z : F32)}
    case ANode{l, r}:
      x y = divs(l, z) divs(r, z)
      ANode{x, y}

def softmax.norm(ez: F32 & Array<F32>) -> Array<F32>:
  (z, e) = ez
  divs(e, z)

def softmax.exp(mx: F32 & Array<F32>) -> Array<F32>:
  (m, x) = mx
  softmax.norm(exps(x, m))

def softmax(s: Array<F32>, n: U32, pos: U32) -> Array<F32>:
  softmax.exp(mmax(s, n, 0, pos))

# silu (ggml_silu_f32): x / (1 + exp(-x))
# ---------------------------------------

def silu1(x: F32) -> F32:
  +v = x
  (v / (1.0 + exp32(F32.neg(v)) : F32) : F32)

def silu(a: Array<F32>) -> Array<F32>:
  Array.map(~F32, ~F32, ~silu1, a)

# argmax: numpy's rule
# --------------------
# the first NaN if any, else the first index holding the max; a tree of
# left-biased picks (the left half has the lower indices), so it is exact
# for any split. -0.0 and +0.0 compare equal (first wins).

def amax.pick(l: F32 & U32, r: F32 & U32) -> F32 & U32:
  (+lv, +li) = l
  (+rv, +ri) = r
  Bool.pick(F32 & U32, is_nan(lv), (lv, li),
    Bool.pick(F32 & U32, is_nan(rv), (rv, ri),
      Bool.pick(F32 & U32, (rv > lv : F32), (rv, ri), (lv, li))))

def argmax(a: Array<F32>, +n: U32, +lo: U32) -> F32 & U32:
  match a:
    case ALeaf{v}:
      (v, lo)
    case ANode{l, r}:
      +h = (n >> 1n : U32)
      x y = argmax(l, h, lo) argmax(r, h, (lo + h : U32))
      amax.pick(x, y)

# gemv row: y_i = tsum(W[i,:] * x), handing x back for the next row
# -----------------------------------------------------------------

def dot(r: Array<F32>, x: Array<F32>) -> F32 & Array<F32>:
  match r x:
    case ALeaf{w} ALeaf{+v}:
      ((w * v : F32), ALeaf{v})
    case ANode{rl, rr} ANode{xl, xr}:
      a b = dot(rl, xl) dot(rr, xr)
      sumsq.join(a, b)
    case _ _:
      (0.0, ALeaf{0.0})

# Decode one row while reducing it: no temporary weight arena. Exact-read
# validation stays in IO. Missing padding bytes still multiply +0 by x.
def dot_bytes.fin(bs: List<&2, U32>, a: F32, b: F32, c: F32, d: F32,
  got: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32 & List<&2, U32> & Array<F32>:
  (x, (w, (v, (u, t)))) = got
  ((((a * w : F32) + (b * v : F32) : F32) +
    ((c * u : F32) + (d * t : F32) : F32) : F32), (bs, x))

def dot_bytes.tail(x: Array<F32>, i: U32,
  got: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32 & List<&2, U32> & Array<F32>:
  (r, (a, (b, (c, d)))) = got
  dot_bytes.fin(Nil{}, a, b, c, d, Array.get4(F32, x, i))

def dot_bytes.leaf(bs: List<&2, U32>, x: Array<F32>, i: U32) -> F32 & List<&2, U32> & Array<F32>:
  match bs:
    case Con{a0, Con{a1, Con{a2, Con{a3,
      Con{b0, Con{b1, Con{b2, Con{b3,
      Con{c0, Con{c1, Con{c2, Con{c3,
      Con{d0, Con{d1, Con{d2, Con{d3, t}}}}}}}}}}}}}}}}:
      dot_bytes.fin(t, f32_of(a0, a1, a2, a3), f32_of(b0, b1, b2, b3),
        f32_of(c0, c1, c2, c3), f32_of(d0, d1, d2, d3), Array.get4(F32, x, i))
    case Nil{}:
      dot_bytes.fin(Nil{}, 0.0, 0.0, 0.0, 0.0, Array.get4(F32, x, i))
    case _:
      dot_bytes.tail(x, i, Array.get4(F32, vec(2n, 0.0, bs), 0))

def dot_bytes.join(a: F32, got: F32 & List<&2, U32> & Array<F32>) -> F32 & List<&2, U32> & Array<F32>:
  (b, rest) = got
  ((a + b : F32), rest)

def dot_bytes.right(k: List<&2, U32> -> Array<F32> -> F32 & List<&2, U32> & Array<F32>,
  got: F32 & List<&2, U32> & Array<F32>) -> F32 & List<&2, U32> & Array<F32>:
  (a, (bs, x)) = got
  dot_bytes.join(a, k(bs, x))

# cols is a multiple of four. Padding must NOT consume the next row's bytes.
def dot_bytes.at(in_row: Bool, bs: List<&2, U32>, x: Array<F32>, i: U32) -> F32 & List<&2, U32> & Array<F32>:
  match in_row:
    case True{}:
      dot_bytes.leaf(bs, x, i)
    case False{}:
      dot_bytes.fin(bs, 0.0, 0.0, 0.0, 0.0, Array.get4(F32, x, i))

def dot_bytes.go(d: Nat, +i: U32, +cols: U32, bs: List<&2, U32>, x: Array<F32>) -> F32 & List<&2, U32> & Array<F32>:
  match d:
    case 0n:
      dot_bytes.at((i < cols : U32), bs, x, i)
    case 1n+p0:
      +p = p0
      dot_bytes.right(t => x2 => dot_bytes.go(p, (i + (4 << p : U32) : U32), cols, t, x2),
        dot_bytes.go(p, i, cols, bs, x))

def dot_bytes.done(got: F32 & List<&2, U32> & Array<F32>) -> F32 & Array<F32>:
  (y, (bs, x)) = got
  (y, x)

def dot_bytes(d: Nat, bs: List<&2, U32>, x: Array<F32>) -> F32 & Array<F32>:
  match d:
    case 1n+1n+p0:
      +p = p0
      dot_bytes.done(dot_bytes.go(p, 0, (4 << p : U32), bs, x))
    case _:
      dot(vec(d, 0.0, bs), x)

# Packed-byte rows use the same four-leaf subtrees, without byte-list cells.
# cols is divisible by four; at is the row's byte offset within this chunk.
def dot_buf.fin(a: F32, b: F32, c: F32, d: F32,
  got: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32 & Array<F32>:
  (x, (w, (v, (u, t)))) = got
  ((((a * w : F32) + (b * v : F32) : F32) +
    ((c * u : F32) + (d * t : F32) : F32) : F32), x)

def dot_buf.leaf(in_row: Bool, +bs: Bytes(), +at: Nat, x: Array<F32>, i: U32) -> F32 & Array<F32>:
  match in_row:
    case True{}:
      dot_buf.fin(f32_from_bits(Bytes.word_le(bs, at)), f32_from_bits(Bytes.word_le(bs, Nat.add(at, 4n))),
        f32_from_bits(Bytes.word_le(bs, Nat.add(at, 8n))), f32_from_bits(Bytes.word_le(bs, Nat.add(at, 12n))),
        Array.get4(F32, x, i))
    case False{}:
      dot_buf.fin(0.0, 0.0, 0.0, 0.0, Array.get4(F32, x, i))

def dot_buf.join(a: F32, got: F32 & Array<F32>) -> F32 & Array<F32>:
  (b, x) = got
  ((a + b : F32), x)

def dot_buf.right(k: Array<F32> -> F32 & Array<F32>, got: F32 & Array<F32>) -> F32 & Array<F32>:
  (a, x) = got
  dot_buf.join(a, k(x))

def dot_buf(d: Nat, +i: U32, +cols: U32, +at: U32, +bs: Bytes(), x: Array<F32>) -> F32 & Array<F32>:
  match d:
    case 0n:
      dot_buf.leaf((i < cols : U32), bs, U32.to_nat((at + 4 * i : U32)), x, i)
    case 1n+p0:
      +p = p0
      dot_buf.right(x2 => dot_buf(p, (i + (4 << p : U32) : U32), cols, at, bs, x2),
        dot_buf(p, i, cols, at, bs, x))

# a vector from leaves listed highest index first (as a gemv pushes its rows),
# with k pad leaves on top, so 2^d leaves in all
def padl(k: Nat, +pad: F32, acc: List<Array<F32>>) -> List<Array<F32>>:
  match k:
    case 0n:
      acc
    case 1n+p:
      padl(p, pad, ALeaf{pad} <> acc)

def from_rev(+d: Nat, k: Nat, +pad: F32, acc: List<Array<F32>>) -> Array<F32>:
  root(pairs(d, True{}, padl(k, pad, acc)))

# Elementwise
# -----------

def vadd(a: Array<F32>, b: Array<F32>) -> Array<F32>:
  match a b:
    case ALeaf{x} ALeaf{y}:
      ALeaf{(x + y : F32)}
    case ANode{al, ar} ANode{bl, br}:
      l r = vadd(al, bl) vadd(ar, br)
      ANode{l, r}
    case _ _:
      ALeaf{0.0}

def vmul(a: Array<F32>, b: Array<F32>) -> Array<F32>:
  match a b:
    case ALeaf{x} ALeaf{y}:
      ALeaf{(x * y : F32)}
    case ANode{al, ar} ANode{bl, br}:
      l r = vmul(al, bl) vmul(ar, br)
      ANode{l, r}
    case _ _:
      ALeaf{0.0}

def vscale(a: Array<F32>, +s: F32) -> Array<F32>:
  match a:
    case ALeaf{x}:
      ALeaf{(x * s : F32)}
    case ANode{l, r}:
      x y = vscale(l, s) vscale(r, s)
      ANode{x, y}

# Attention (GQA 8:1, one query position)
# ---------------------------------------
# A layer's K (and V) cache is flat [T,2,128], index t*256+g*128+j.
# The C lane packs Array<F32> into a contiguous buffer; these kernels use its
# tree interface. Slot t is the subtree at depth dT; kv head g is that slot's
# left (0) or right (1) half. Query head h reads kv head h >> 3. Unwritten
# slots hold zeros and are masked by the softmax (t > pos).

# 1 / sqrt(head_dim), in f32 as llama.cpp's kq_scale
def kq_scale() -> F32:
  (1.0 / F32.sqrt(128.0) : F32)

def half(+g: U32, a: Array<F32>) -> Array<F32>:
  match a:
    case ANode{l, r}:
      Bool.pick(Array<F32>, U32.is_eq(g, 0), l, r)
    case ALeaf{v}:
      ALeaf{v}

# k into the half that holds pos; f recurses into a half
def put.if(in_l: Bool, f: Array<F32> -> Array<F32> -> U32 -> Array<F32>,
  l: Array<F32>, r: Array<F32>, +lo: U32, +h: U32, k: Array<F32>) -> Array<F32>:
  match in_l:
    case True{}:
      ANode{f(l, k, lo), r}
    case False{}:
      ANode{l, f(r, k, (lo + h : U32))}

# slot pos := k (n slots under this node, the first at lo)
def put(d: Nat, c: Array<F32>, +n: U32, +lo: U32, +pos: U32, k: Array<F32>) ->
  Array<F32>:
  match d c:
    case 0n _:
      k
    case 1n+p ANode{l, r}:
      +h = (n >> 1n : U32)
      put.if((pos < (lo + h : U32) : U32), sub => x => at => put(p, sub, h, at, pos, x),
        l, r, lo, h, k)
    case _ _:
      c

def fst(r: F32 & Array<F32>) -> F32:
  (s, x) = r
  s

# the two halves' results, each from its own copy of a shared vector
def both(qq: Array<F32> & Array<F32>, fa: Array<F32> -> Array<F32>,
  fb: Array<F32> -> Array<F32>) -> Array<F32>:
  (qa, qb) = qq
  x y = fa(qa) fb(qb)
  ANode{x, y}

# qknorm
# ------
# Qwen2.5 per-head RMSNorm on v with gain g:
# for each row: v_i *= rsqrt(mean_j(v_j^2) + eps) * g_i
# k: log2 of head count (e.g. 4n for 16 heads, 1n for 2 heads, 0n for 1 head)
# d: head dimension as float (e.g. 128.0)

def qknorm.go(k: Nat, v: Array<F32>, g: Array<F32>, +d: F32) -> Array<F32>:
  match k v:
    case 0n _:
      rms_norm(v, g, d)
    case 1n+p0 ANode{l, r}:
      +p = p0
      both(Array.clone(F32, g), x => qknorm.go(p, l, x, d), y => qknorm.go(p, r, y, d))
    case _ _:
      v

def qknorm(k: Nat, v: Array<F32>, g: Array<F32>, +d: F32) -> Array<F32>:
  qknorm.go(k, v, g, d)

# output_gate: x * sigmoid(gate)
# ------------------------------
# Qwen3.5 attention output gate: elementwise product with sigmoid(gate)
def output_gate(x: Array<F32>, g: Array<F32>) -> Array<F32>:
  match x g:
    case ALeaf{xv} ALeaf{gv}:
      +sig = (1.0 / (1.0 + exp32(F32.neg(gv)) : F32) : F32)
      ALeaf{(xv * sig : F32)}
    case ANode{xl, xr} ANode{gl, gr}:
      l r = output_gate(xl, gl) output_gate(xr, gr)
      ANode{l, r}
    case _ _:
      ALeaf{0.0}

# one query head's scores over the d-deep slots of c: s_t = tsum(q * K_t[g]) * sc
def kscores(d: Nat, c: Array<F32>, q: Array<F32>, +g: U32, +sc: F32) -> Array<F32>:
  match d c:
    case 0n _:
      ALeaf{(fst(dot(half(g, c), q)) * sc : F32)}
    case 1n+p0 ANode{l, r}:
      +p = p0
      both(Array.clone(F32, q), x => kscores(p, l, x, g, sc), y => kscores(p, r, y, g, sc))
    case _ _:
      ALeaf{0.0}

# o = the tree sum over slots of p_t * V_t[g] (the same balanced split over t)
def vsum(d: Nat, c: Array<F32>, p: Array<F32>, +g: U32) -> Array<F32>:
  match d c p:
    case 0n _ ALeaf{w}:
      vscale(half(g, c), w)
    case 1n+k0 ANode{cl, cr} ANode{pl, pr}:
      +k = k0
      x y = vsum(k, cl, pl, g) vsum(k, cr, pr, g)
      vadd(x, y)
    case _ _ _:
      ALeaf{0.0}

# The top d levels are query heads; each reads kv head lo >> 3. Keep the
# three attention stages separate so the runner can time each, without
# changing any reduction order.
def scores(d: Nat, +n: U32, +lo: U32, +dT: Nat, q: Array<F32>, kc: Array<F32>) -> Array<F32>:
  match d q:
    case 0n _:
      kscores(dT, kc, q, (lo >> 3n : U32), kq_scale())
    case 1n+p0 ANode{ql, qr}:
      +p = p0
      +h = (n >> 1n : U32)
      both(Array.clone(F32, kc), k => scores(p, h, lo, dT, ql, k),
        k => scores(p, h, (lo + h : U32), dT, qr, k))
    case _ _:
      ALeaf{0.0}

def probs(d: Nat, +nT: U32, +pos: U32, s: Array<F32>) -> Array<F32>:
  match d s:
    case 0n _:
      softmax(s, nT, pos)
    case 1n+p0 ANode{l, r}:
      +p = p0
      a b = probs(p, nT, pos, l) probs(p, nT, pos, r)
      ANode{a, b}
    case _ _:
      ALeaf{0.0}

def weighted(d: Nat, +n: U32, +lo: U32, +dT: Nat, ps: Array<F32>, vc: Array<F32>) -> Array<F32>:
  match d ps:
    case 0n _:
      vsum(dT, vc, ps, (lo >> 3n : U32))
    case 1n+p0 ANode{pl, pr}:
      +p = p0
      +h = (n >> 1n : U32)
      both(Array.clone(F32, vc), v => weighted(p, h, lo, dT, pl, v),
        v => weighted(p, h, (lo + h : U32), dT, pr, v))
    case _ _:
      ALeaf{0.0}

# Blocked dot: order v2 (reports/llb-blocked.md)
# ----------------------------------------------
# 16 lanes: lane j sums the products of columns c = j mod 16, left to right
# from +0.0, one F32 multiply then one F32 add each; the 16 lane sums then
# merge by the balanced tree, as tsum. Only a row's real columns take part.
# A step feeds one quad into lanes 0..3, then rotates the lanes by four, so
# all are home again once cols % 16 == 0 (io.gemv refuses anything else).

def bdot.fin(s0: F32, s1: F32, s2: F32, s3: F32, s4: F32, s5: F32, s6: F32, s7: F32,
  s8: F32, s9: F32, s10: F32, s11: F32, s12: F32, s13: F32, s14: F32, s15: F32) -> F32:
  ((((s0 + s1 : F32) + (s2 + s3 : F32) : F32) + ((s4 + s5 : F32) + (s6 + s7 : F32) : F32) : F32) +
    (((s8 + s9 : F32) + (s10 + s11 : F32) : F32) + ((s12 + s13 : F32) + (s14 + s15 : F32) : F32) : F32) : F32)

# got = Array.get4(x, i) carries the quad for the step at i (after the last
# step it has read x[cols..cols+3] mod the size, unused).
def bdot_buf.go(fuel: Nat, +at: Nat, +i: U32, +bs: Bytes(),
  s0: F32, s1: F32, s2: F32, s3: F32, s4: F32, s5: F32, s6: F32, s7: F32,
  s8: F32, s9: F32, s10: F32, s11: F32, s12: F32, s13: F32, s14: F32, s15: F32,
  got: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32 & Array<F32>:
  match fuel:
    case 0n:
      (x, q) = got
      (bdot.fin(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15), x)
    case 1n+k:
      (x, (w, (v, (u, t)))) = got
      bdot_buf.go(k, Nat.add(at, 16n), (i + 4 : U32), bs,
        s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        (s0 + (f32_from_bits(Bytes.word_le(bs, at)) * w : F32) : F32),
        (s1 + (f32_from_bits(Bytes.word_le(bs, Nat.add(at, 4n))) * v : F32) : F32),
        (s2 + (f32_from_bits(Bytes.word_le(bs, Nat.add(at, 8n))) * u : F32) : F32),
        (s3 + (f32_from_bits(Bytes.word_le(bs, Nat.add(at, 12n))) * t : F32) : F32),
        Array.get4(F32, x, (i + 4 : U32)))

# the row of cols F32 at byte at of bs, against x[0..cols)
def bdot_buf(+cols: U32, +at: U32, +bs: Bytes(), x: Array<F32>) -> F32 & Array<F32>:
  bdot_buf.go(U32.to_nat((cols >> 2n : U32)), U32.to_nat(at), 0, bs,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    Array.get4(F32, x, 0))

# The byte-list path (large offsets): the same order, handing the rest back.
# io.expect makes every row exact, so the short-list case is unreachable; it
# keeps v1's rule (missing bytes are +0 weights) rather than inventing one.
def bdot_list.go(fuel: Nat, +i: U32, bs: List<&2, U32>,
  s0: F32, s1: F32, s2: F32, s3: F32, s4: F32, s5: F32, s6: F32, s7: F32,
  s8: F32, s9: F32, s10: F32, s11: F32, s12: F32, s13: F32, s14: F32, s15: F32,
  got: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32 & List<&2, U32> & Array<F32>:
  match fuel bs:
    case 0n _:
      (x, q) = got
      (bdot.fin(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15), (bs, x))
    case 1n+k Con{a0, Con{a1, Con{a2, Con{a3,
      Con{b0, Con{b1, Con{b2, Con{b3,
      Con{c0, Con{c1, Con{c2, Con{c3,
      Con{d0, Con{d1, Con{d2, Con{d3, r}}}}}}}}}}}}}}}}:
      (x, (w, (v, (u, t)))) = got
      bdot_list.go(k, (i + 4 : U32), r,
        s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        (s0 + (f32_of(a0, a1, a2, a3) * w : F32) : F32), (s1 + (f32_of(b0, b1, b2, b3) * v : F32) : F32),
        (s2 + (f32_of(c0, c1, c2, c3) * u : F32) : F32), (s3 + (f32_of(d0, d1, d2, d3) * t : F32) : F32),
        Array.get4(F32, x, (i + 4 : U32)))
    case 1n+k _:
      (x, (w, (v, (u, t)))) = got
      bdot_list.go(k, (i + 4 : U32), Nil{},
        s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        (s0 + (0.0 * w : F32) : F32), (s1 + (0.0 * v : F32) : F32),
        (s2 + (0.0 * u : F32) : F32), (s3 + (0.0 * t : F32) : F32),
        Array.get4(F32, x, (i + 4 : U32)))

def bdot_list(+cols: U32, bs: List<&2, U32>, x: Array<F32>) -> F32 & List<&2, U32> & Array<F32>:
  bdot_list.go(U32.to_nat((cols >> 2n : U32)), 0, bs,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    Array.get4(F32, x, 0))

# Qwen3.5 hybrid ops (w2/w2c semantics)
# =====================================
# The gated delta net, 4-tap causal conv1d and gated RMSNorm of the Qwen3.5
# linear-attention layers, and the per-head L2 norm. All per-head reductions are
# this file's balanced trees (so every split, and the CPU/CUDA/MSL lanes, agree
# with the same order); transcendentals go through F64 exactly as exp32 does.

# L2 norm (per head) + scale
# --------------------------
# ggml's l2_norm has no +eps inside the sqrt: scale = 1 / max(sqrt(ss), eps).
# q is then divided by sqrt(head_dim); k is not. `sc` is the divisor after the
# l2 scale (sqrt(d) for q, 1.0 for k), passed as F32 so the two divisions keep
# the w2c engine's order. k: log2 of the head count.

def l2norm.leaf(+sc: F32, sx: F32 & Array<F32>) -> Array<F32>:
  (ss, x) = sx
  vscale(x, ((1.0 / F32.max(F32.sqrt(ss), eps()) : F32) / sc : F32))

def l2norm.go(k: Nat, v: Array<F32>, +sc: F32) -> Array<F32>:
  match k v:
    case 0n _:
      l2norm.leaf(sc, sumsq(v))
    case 1n+p0 ANode{l, r}:
      +p = p0
      x y = l2norm.go(p, l, sc) l2norm.go(p, r, sc)
      ANode{x, y}
    case _ _:
      v

def l2norm(k: Nat, v: Array<F32>, +sc: F32) -> Array<F32>:
  l2norm.go(k, v, sc)

# causal depthwise conv1d (kernel 4, + SiLU) with history
# -------------------------------------------------------
# x is the conv-dim input at t (2^d leaves); h0,h1,h2 are the three previous
# inputs, oldest first; w is [2^d,4] row-major (2^(d+2) leaves, the four taps
# of a channel its depth-2 subtree in index order). The recursion descends x
# and w in lockstep; a leaf has one channel and its four taps. Returns y and
# the next history (h1, h2, x). y_c = silu(h0_c*w[c,0] + h1_c*w[c,1] +
# h2_c*w[c,2] + x_c*w[c,3]).

def conv1d.leaf(x: Array<F32>, h0: Array<F32>, h1: Array<F32>, h2: Array<F32>, w: Array<F32>) -> F32:
  match x h0 h1 h2 w:
    case ALeaf{xv} ALeaf{v0} ALeaf{v1} ALeaf{v2}
      ANode{ANode{ALeaf{w0}, ALeaf{w1}}, ANode{ALeaf{w2}, ALeaf{w3}}}:
      +a = ((v0 * w0 : F32) + (v1 * w1 : F32) : F32)
      +b = ((v2 * w2 : F32) + (xv * w3 : F32) : F32)
      silu1((a + b : F32))
    case _ _ _ _ _:
      0.0

def conv1d.go(d: Nat, x: Array<F32>, h0: Array<F32>, h1: Array<F32>, h2: Array<F32>, w: Array<F32>) ->
  Array<F32>:
  match d x h0 h1 h2 w:
    case 0n _ _ _ _ _:
      ALeaf{conv1d.leaf(x, h0, h1, h2, w)}
    case 1n+p0 ANode{xl, xr} ANode{al, ar} ANode{bl, br} ANode{cl, cr} ANode{wl, wr}:
      +p = p0
      ANode{conv1d.go(p, xl, al, bl, cl, wl), conv1d.go(p, xr, ar, br, cr, wr)}
    case _ _ _ _ _ _:
      ALeaf{0.0}

# Returns only y; the next history is (h1, h2, x) by reference (arrays are
# immutable), which the caller shifts.
def conv1d(+d: Nat, w: Array<F32>, x: Array<F32>, h0: Array<F32>, h1: Array<F32>, h2: Array<F32>) ->
  Array<F32>:
  conv1d.go(d, x, h0, h1, h2, w)

# gated RMSNorm (per value head)
# ------------------------------
# out = rms(o, w, d) * silu(z), head by head; w (ssm_norm.weight) is shared.
# k: log2 of the head count; d: head dimension as F32.

def gated_rmsnorm.go(k: Nat, o: Array<F32>, w: Array<F32>, z: Array<F32>, +d: F32) -> Array<F32>:
  match k o z:
    case 0n _ _:
      vmul(rms_norm(o, w, d), silu(z))
    case 1n+p0 ANode{ol, or} ANode{zl, zr}:
      +p = p0
      both(Array.clone(F32, w),
        x => gated_rmsnorm.go(p, ol, x, zl, d),
        y => gated_rmsnorm.go(p, or, y, zr, d))
    case _ _ _:
      o

def gated_rmsnorm(k: Nat, o: Array<F32>, w: Array<F32>, z: Array<F32>, +d: F32) -> Array<F32>:
  gated_rmsnorm.go(k, o, w, z, d)

# gated delta recurrence (per value head)
# ---------------------------------------
# For each value head: S is [SZ,SZ], q/k/v are [SZ]; decay = exp(g) and beta =
# sigmoid(b). Per row j (the v dimension):
#   sdec[i] = S[j][i] * decay; kv = sum_i sdec[i]*k[i];
#   delta = (v[j] - kv) * beta; S'[j][i] = sdec[i] + k[i]*delta;
#   o[j] = sum_i S'[j][i] * q[i].
# q must already carry the 1/sqrt(SZ) scale (l2norm's q leg). S is stored with
# the key dimension innermost (S[j][i], i = key), as the w2c engine's S row.

def gd.row.fin(sn: Array<F32>, orr: F32 & Array<F32>) -> F32 & Array<F32>:
  (o, _b) = orr
  (o, sn)

@unsafe
def gd.row.go(sd: Array<F32>, +vj: F32, q: Array<F32>, k: Array<F32>, +beta: F32,
  kvr: F32 & Array<F32>) -> F32 & Array<F32>:
  (kv, _a) = kvr
  +delta = ((vj - kv : F32) * beta : F32)
  +sn = vadd(sd, vscale(k, delta))
  gd.row.fin(sn, dot(sn, q))

@unsafe
def gd.row(srow: Array<F32>, +vj: F32, +q: Array<F32>, +k: Array<F32>, +decay: F32, +beta: F32) ->
  F32 & Array<F32>:
  +sd = vscale(srow, decay)
  gd.row.go(sd, vj, q, k, beta, dot(sd, k))

# the rows are independent (row j's update and readout use only row j and
# v[j]), so the two halves join structurally, with no ordered accumulation.
def gd.join(l: Array<F32> & Array<F32>, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (ol, nl) = l
  (orr, nr) = r
  (ANode{ol, orr}, ANode{nl, nr})

def gd.leaf(os: F32 & Array<F32>) -> Array<F32> & Array<F32>:
  (o, sn) = os
  (ALeaf{o}, sn)

@unsafe
def gd.rows(d: Nat, S: Array<F32>, v: Array<F32>, +q: Array<F32>, +k: Array<F32>,
  +decay: F32, +beta: F32) -> Array<F32> & Array<F32>:
  match d S v:
    case 0n _ ALeaf{vj}:
      gd.leaf(gd.row(S, vj, q, k, decay, beta))
    case 1n+p0 ANode{Sl, Sr} ANode{vl, vr}:
      +p = p0
      gd.join(gd.rows(p, Sl, vl, q, k, decay, beta), gd.rows(p, Sr, vr, q, k, decay, beta))
    case _ _ _:
      (v, S)

def gd.head(S: Array<F32>, q: Array<F32>, k: Array<F32>, v: Array<F32>, +decay: F32, +beta: F32) ->
  Array<F32> & Array<F32>:
  gd.rows(7n, S, v, q, k, decay, beta)

@unsafe
def gd.go(k: Nat, S: Array<F32>, q: Array<F32>, kk: Array<F32>, v: Array<F32>,
  dec: Array<F32>, bet: Array<F32>) -> Array<F32> & Array<F32>:
  match k S q kk v dec bet:
    case 0n _ _ _ _ ALeaf{dv} ALeaf{bv}:
      gd.head(S, q, kk, v, dv, bv)
    case 1n+p0 ANode{Sl, Sr} ANode{ql, qr} ANode{kl, kr} ANode{vl, vr} ANode{dl, dr} ANode{bl, br}:
      +p = p0
      l = gd.go(p, Sl, ql, kl, vl, dl, bl)
      r = gd.go(p, Sr, qr, kr, vr, dr, br)
      gd.join(l, r)
    case _ _ _ _ _ _ _:
      (q, S)

# out and the updated state, head by head; k: log2 of the value-head count
@unsafe
def gated_delta(k: Nat, S: Array<F32>, q: Array<F32>, kk: Array<F32>, v: Array<F32>,
  dec: Array<F32>, bet: Array<F32>) -> Array<F32> & Array<F32>:
  gd.go(k, S, q, kk, v, dec, bet)

# Qwen3.5 full-attention ops (w2f)
# ================================
# GQA 16/4, head_dim 256, partial NEOX rope (nd of hd, base 1e7), per-head
# q/k RMSNorm, and the sigmoid output gate. Same balanced-tree order as the
# linear-attention ops above.

# scalar activations (through F64, as exp32)
def sigmoid1(x: F32) -> F32:
  (1.0 / (1.0 + exp32(F32.neg(x)) : F32) : F32)

def softplus.hi(hi: Bool, +x: F32) -> F32:
  match hi:
    case True{}:
      x
    case False{}:
      F64.to_f32(F64.log(F64.add(F64.from_nat(1n), F64.exp(F32.to_f64(Bool.pick(F32, F32.is_lt(x, 88.0), x, 88.0))))))

def softplus1(+x: F32) -> F32:
  softplus.hi(F32.is_lt(20.0, x), x)

def sigmoid(a: Array<F32>) -> Array<F32>:
  Array.map(~F32, ~F32, ~sigmoid1, a)

def softplus(a: Array<F32>) -> Array<F32>:
  Array.map(~F32, ~F32, x => softplus1(x), a)

def expmap(a: Array<F32>) -> Array<F32>:
  Array.map(~F32, ~F32, ~exp32, a)

# select kv head g of four from a [4, hd] slot
def head4(+g: U32, a: Array<F32>) -> Array<F32>:
  match a:
    case ANode{ANode{a0, a1}, ANode{a2, a3}}:
      Bool.pick(Array<F32>, U32.is_eq(g, 0), a0,
        Bool.pick(Array<F32>, U32.is_eq(g, 1), a1,
          Bool.pick(Array<F32>, U32.is_eq(g, 2), a2, a3)))
    case _:
      a

def kscores4(d: Nat, c: Array<F32>, q: Array<F32>, +g: U32, +sc: F32) -> Array<F32>:
  match d c:
    case 0n _:
      ALeaf{(fst(dot(head4(g, c), q)) * sc : F32)}
    case 1n+p0 ANode{l, r}:
      +p = p0
      both(Array.clone(F32, q), x => kscores4(p, l, x, g, sc), y => kscores4(p, r, y, g, sc))
    case _ _:
      ALeaf{0.0}

def vsum4(d: Nat, c: Array<F32>, p: Array<F32>, +g: U32) -> Array<F32>:
  match d c p:
    case 0n _ ALeaf{w}:
      vscale(head4(g, c), w)
    case 1n+k0 ANode{cl, cr} ANode{pl, pr}:
      +k = k0
      x y = vsum4(k, cl, pl, g) vsum4(k, cr, pr, g)
      vadd(x, y)
    case _ _ _:
      ALeaf{0.0}

# query heads read kv head lo >> 2 (GQA 16/4)
def scores4(d: Nat, +n: U32, +lo: U32, +dT: Nat, +sc: F32, q: Array<F32>, kc: Array<F32>) -> Array<F32>:
  match d q:
    case 0n _:
      kscores4(dT, kc, q, (lo >> 2n : U32), sc)
    case 1n+p0 ANode{ql, qr}:
      +p = p0
      +h = (n >> 1n : U32)
      both(Array.clone(F32, kc), k => scores4(p, h, lo, dT, sc, ql, k),
        k => scores4(p, h, (lo + h : U32), dT, sc, qr, k))
    case _ _:
      ALeaf{0.0}

def weighted4(d: Nat, +n: U32, +lo: U32, +dT: Nat, ps: Array<F32>, vc: Array<F32>) -> Array<F32>:
  match d ps:
    case 0n _:
      vsum4(dT, vc, ps, (lo >> 2n : U32))
    case 1n+p0 ANode{pl, pr}:
      +p = p0
      +h = (n >> 1n : U32)
      both(Array.clone(F32, vc), v => weighted4(p, h, lo, dT, pl, v),
        v => weighted4(p, h, (lo + h : U32), dT, pr, v))
    case _ _:
      ALeaf{0.0}

# partial NEOX rope: within a head of n leaves, rotate (i, i+nd/2) for
# i < nd/2; dims >= nd are copied. theta_i = pos * base^(-2i/nd).
def neox.pair(+i: U32, +nd: F32, +pos: F32, +base: F32, a: F32, b: F32) -> Array<F32>:
  +t = (pos * pow32(base, (F32.neg(U32.to_f32((2 * i : U32))) / nd : F32)) : F32)
  rope.rot(a, b, cos32(t), sin32(t))

def neox.pairs(l: Array<F32>, r: Array<F32>, +i: U32, +m: U32, +nd: F32, +pos: F32, +base: F32) -> Array<F32>:
  match l r:
    case ALeaf{a} ALeaf{b}:
      neox.pair(i, nd, pos, base, a, b)
    case ANode{ll, lr} ANode{rl, rr}:
      +h = (m >> 1n : U32)
      ANode{neox.pairs(ll, rl, i, h, nd, pos, base), neox.pairs(lr, rr, (i + h : U32), h, nd, pos, base)}
    case _ _:
      ALeaf{0.0}

# sink k levels into the head (k = log2(head_dim / rope_dim)), rotating the
# nd-leaf node reached (its two halves paired elementwise); siblings are copied
def neox.sink(k: Nat, x: Array<F32>, +nd: U32, +pos: F32, +base: F32) -> Array<F32>:
  match k x:
    case 0n ANode{l, r}:
      neox.pairs(l, r, 0, (nd >> 1n : U32), U32.to_f32(nd), pos, base)
    case 1n+p0 ANode{l, r}:
      +p = p0
      ANode{neox.sink(p, l, nd, pos, base), r}
    case _ _:
      x

def neox(x: Array<F32>, +k: Nat, +nd: U32, +pos: F32, +base: F32) -> Array<F32>:
  neox.sink(k, x, nd, pos, base)

# per head (2*hd leaves: q then gate): rms(q, w) then rope; gate copied
def qg.go(k: Nat, v: Array<F32>, w: Array<F32>, +hd: U32, +nd: U32, +pos: F32, +base: F32) -> Array<F32>:
  match k v:
    case 0n ANode{q, g}:
      ANode{neox(rms_norm(q, w, U32.to_f32(hd)), 2n, nd, pos, base), g}
    case 1n+p0 ANode{l, r}:
      +p = p0
      both(Array.clone(F32, w),
        x => qg.go(p, l, x, hd, nd, pos, base),
        y => qg.go(p, r, y, hd, nd, pos, base))
    case _ _:
      v

# per head k: rms(v, w) then rope
def kr.go(k: Nat, v: Array<F32>, w: Array<F32>, +hd: U32, +nd: U32, +pos: F32, +base: F32) -> Array<F32>:
  match k v:
    case 0n _:
      neox(rms_norm(v, w, U32.to_f32(hd)), 2n, nd, pos, base)
    case 1n+p0 ANode{l, r}:
      +p = p0
      both(Array.clone(F32, w),
        x => kr.go(p, l, x, hd, nd, pos, base),
        y => kr.go(p, r, y, hd, nd, pos, base))
    case _ _:
      v

def split_qg.join(a: Array<F32> & Array<F32>, b: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (ql, gl) = a
  (qr, gr) = b
  (ANode{ql, qr}, ANode{gl, gr})

# split each head's (q, gate) into two parallel trees
def split_qg.go(k: Nat, v: Array<F32>) -> Array<F32> & Array<F32>:
  match k v:
    case 0n ANode{q, g}:
      (q, g)
    case 1n+p0 ANode{l, r}:
      +p = p0
      a b = split_qg.go(p, l) split_qg.go(p, r)
      split_qg.join(a, b)
    case _ _:
      (v, ALeaf{0.0})
