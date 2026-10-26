# Device GEMV: y = W x, W resident as Bytes (f32le rows of rb bytes from base)
# and x as Bytes (its 2^d leaves, f32le), for one `!` call.
#
# The arithmetic is K.dot_buf's, bit for bit: four-leaf groups
# ((w0*x0 + w1*x1) + (w2*x2 + w3*x3)), +0 weights times the real x past cols,
# then the balanced binary tree over groups, left + right at every node. The
# bang forks rows down to 2^14 leaves; a leaf that is one row forks that row's
# top subtrees instead, and joins them with the same left + right, so the tree
# is unchanged wherever it runs. Leaves are flat loops (t0..t11: one def a
# level, no self call). Build with --fmad=false (Bend's NVRTC default).
import Base
import ./ops.bend as K

def rd(+b: Bytes(), +at: U32) -> F32:
  K.f32_from_bits(Bytes.word_le(b, U32.to_nat(at)))

def grp.go(in_row: Bool, +wb: Bytes(), +xb: Bytes(), +w: U32, +o: U32) -> F32:
  match in_row:
    case True{}:
      (((rd(wb, w) * rd(xb, o) : F32) + (rd(wb, (w + 4 : U32)) * rd(xb, (o + 4 : U32)) : F32) : F32) +
        ((rd(wb, (w + 8 : U32)) * rd(xb, (o + 8 : U32)) : F32) +
          (rd(wb, (w + 12 : U32)) * rd(xb, (o + 12 : U32)) : F32) : F32) : F32)
    case False{}:
      (((0.0 * rd(xb, o) : F32) + (0.0 * rd(xb, (o + 4 : U32)) : F32) : F32) +
        ((0.0 * rd(xb, (o + 8 : U32)) : F32) + (0.0 * rd(xb, (o + 12 : U32)) : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
def t0(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  grp.go(((g * 4 : U32) < c : U32), wb, xb, (r + (g * 16 : U32) : U32), (g * 16 : U32))

def t1(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t0(wb, xb, r, c, g) + t0(wb, xb, r, c, (g + 1 : U32)) : F32)

def t2(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t1(wb, xb, r, c, g) + t1(wb, xb, r, c, (g + 2 : U32)) : F32)

def t3(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t2(wb, xb, r, c, g) + t2(wb, xb, r, c, (g + 4 : U32)) : F32)

def t4(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t3(wb, xb, r, c, g) + t3(wb, xb, r, c, (g + 8 : U32)) : F32)

def t5(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t4(wb, xb, r, c, g) + t4(wb, xb, r, c, (g + 16 : U32)) : F32)

def t6(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t5(wb, xb, r, c, g) + t5(wb, xb, r, c, (g + 32 : U32)) : F32)

def t7(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t6(wb, xb, r, c, g) + t6(wb, xb, r, c, (g + 64 : U32)) : F32)

def t8(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t7(wb, xb, r, c, g) + t7(wb, xb, r, c, (g + 128 : U32)) : F32)

def t9(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t8(wb, xb, r, c, g) + t8(wb, xb, r, c, (g + 256 : U32)) : F32)

def t10(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t9(wb, xb, r, c, g) + t9(wb, xb, r, c, (g + 512 : U32)) : F32)

def t11(+wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  (t10(wb, xb, r, c, g) + t10(wb, xb, r, c, (g + 1024 : U32)) : F32)

# the subtree of 2^d groups from g (d <= 11)
def tree(+d: Nat, +wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      t0(wb, xb, r, c, g)
    case 1n:
      t1(wb, xb, r, c, g)
    case 2n:
      t2(wb, xb, r, c, g)
    case 3n:
      t3(wb, xb, r, c, g)
    case 4n:
      t4(wb, xb, r, c, g)
    case 5n:
      t5(wb, xb, r, c, g)
    case 6n:
      t6(wb, xb, r, c, g)
    case 7n:
      t7(wb, xb, r, c, g)
    case 8n:
      t8(wb, xb, r, c, g)
    case 9n:
      t9(wb, xb, r, c, g)
    case 10n:
      t10(wb, xb, r, c, g)
    case _:
      t11(wb, xb, r, c, g)

# one row's subtree of 2^d groups from g, its top f levels forked
def cols(+f: Nat, +d: Nat, +wb: Bytes(), +xb: Bytes(), +r: U32, +c: U32, +g: U32) -> F32:
  match f d:
    case 1n+f1 1n+d1:
      a b = cols(f1, d1, wb, xb, r, c, g) cols(f1, d1, wb, xb, r, c, (g + (1 << d1 : U32) : U32))
      (a + b : F32)
    case _ _:
      tree(d, wb, xb, r, c, g)

# the k rows below r, lowest first, onto acc
def rows(k: Nat, +d: Nat, +wb: Bytes(), +xb: Bytes(), +base: U32, +rb: U32, +c: U32, r: U32,
  acc: List<F32>) -> List<F32>:
  match k:
    case 1n+p:
      +i = (r - 1 : U32)
      rows(p, d, wb, xb, base, rb, c, i, tree(d, wb, xb, (base + (i * rb : U32) : U32), c, 0) <> acc)
    case 0n:
      acc

# rows of a leaf, lowest first; a tree of leaves in row order
type Ys is Type:
  YRows{vs: List<F32>}
  YCat{a: Ys, b: Ys}

def node(+f: Nat, one: Bool, +k: Nat, +d: Nat, +wb: Bytes(), +xb: Bytes(), +base: U32, +rb: U32, +c: U32,
  +lo: U32, +n: U32) -> Ys:
  match f one:
    case 0n _:
      YRows{rows(U32.to_nat(n), d, wb, xb, base, rb, c, (lo + n : U32), Nil{})}
    case 1n+f1 True{}:
      YRows{cols(k, d, wb, xb, (base + (lo * rb : U32) : U32), c, 0) <> Nil{}}
    case 1n+f1 False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = node(f1, U32.is_eq(h, 1), f1, d, wb, xb, base, rb, c, lo, h) node(f1, U32.is_eq(m, 1), f1, d, wb, xb, base, rb, c, (lo + h : U32), m)
      YCat{a, b}

# n rows of rb bytes from base, 2^d groups a row (d = x's depth - 2), rb/4 real
# columns; the bang's tree beside the weights, which the host keeps
def gemv(+wb: Bytes(), +xb: Bytes(), +base: U32, +n: U32, +d: Nat, +rb: U32) -> Ys & Bytes():
  (node!(14n, U32.is_eq(n, 1), 14n, d, wb, xb, base, rb, (rb / 4 : U32), 0, n), wb)

# the same tree forked f levels, no bang: for a caller already on the device
def gemv.tree(+f: Nat, +wb: Bytes(), +xb: Bytes(), +base: U32, +n: U32, +d: Nat, +rb: U32) -> Ys:
  node(f, U32.is_eq(n, 1), f, d, wb, xb, base, rb, (rb / 4 : U32), 0, n)

# Host side
# ---------

def push4(b: Bytes(), +w: U32) -> Bytes():
  Bytes.push(Bytes.push(Bytes.push(Bytes.push(b, (w .&. 255 : U32)), ((w >> 8n) .&. 255 : U32)),
    ((w >> 16n) .&. 255 : U32)), (w >> 24n : U32))

def xbytes(xs: List<F32>, acc: Bytes()) -> Bytes():
  match xs:
    case Con{v, t}:
      xbytes(t, push4(acc, F32.bits(v)))
    case Nil{}:
      acc

def flat.rows(vs: List<F32>, acc: List<Array<F32>>) -> List<Array<F32>>:
  match vs:
    case Con{v, t}:
      flat.rows(t, ALeaf{v} <> acc)
    case Nil{}:
      acc

# the rows as the CPU path returns them: leaves, highest row first
def flat(t: Ys, acc: List<Array<F32>>) -> List<Array<F32>>:
  match t:
    case YRows{vs}:
      flat.rows(vs, acc)
    case YCat{a, b}:
      flat(b, flat(a, acc))
