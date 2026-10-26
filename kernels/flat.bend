# Device GEMV over flat arrays: y = W x with x an Array<F32> borrowed in place
# (Array.peek_f32; tools/build_gpu.sh supplies the compiler patch) and each
# row's sum written into its slot of y (the rows are disjoint). The arithmetic is
# dev.bend's (and K.dot_buf's), bit for bit: the same four-leaf groups, the
# same balanced tree over groups, the same fork split of rows and columns;
# only the loads and the output's layout change. Build with --fmad=false.
import Base
import ./ops.bend as K
import ./dev.bend as D

# group at w of W against x's four slots, as dev.bend's grp.go
def grp.go(in_row: Bool, +wb: Bytes(), +w: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      (((D.rd(wb, w) * x0 : F32) + (D.rd(wb, (w + 4 : U32)) * x1 : F32) : F32) +
        ((D.rd(wb, (w + 8 : U32)) * x2 : F32) + (D.rd(wb, (w + 12 : U32)) * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  grp.go((i < c : U32), wb, (r + (g * 16 : U32) : U32),
    Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (t0(wb, x, r, c, g) + t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (t1(wb, x, r, c, g) + t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (t2(wb, x, r, c, g) + t2(wb, x, r, c, (g + 4 : U32)) : F32)

# the subtree of 2^d groups from g: a call a level down to t3 (a bounded
# device function; the tree, so every sum, is dev.bend's)
@unsafe
def tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      t0(wb, x, r, c, g)
    case 1n:
      t1(wb, x, r, c, g)
    case 2n:
      t2(wb, x, r, c, g)
    case _:
      t3(wb, x, r, c, g)

@unsafe
def tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (tree(p, wb, x, r, c, g) + tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      tree.lo(d, wb, x, r, c, g)

# one row's subtree of 2^d groups from g, its top f levels forked
@unsafe
def cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match f d:
    case 1n+f1 1n+d1:
      a b = cols(f1, d1, wb, x, r, c, g) cols(f1, d1, wb, x, r, c, (g + (1 << d1 : U32) : U32))
      (a + b : F32)
    case _ _:
      tree(d, wb, x, r, c, g)

# rows i.. (k of them) into y
@unsafe
def rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      rows(p, d, wb, x, Array.set(F32, y, i, tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

# rows lo..lo+n-1, split as dev.bend's node splits them
@unsafe
def node(+f: Nat, one: Bool, +k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>, +base: U32,
  +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f one:
    case 0n _:
      rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+f1 True{}:
      Array.set(F32, y, lo, cols(k, d, wb, x, (base + (lo * rb : U32) : U32), c, 0))
    case 1n+f1 False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = node(f1, U32.is_eq(h, 1), f1, d, wb, x, y, base, rb, c, lo, h) node(f1, U32.is_eq(m, 1), f1, d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

@unsafe
def tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      tree(d, wb, x, r, c, g)
    case False{}:
      0.0

# Native CUDA leaf (tools/gemv-tile.patch); this body is the CPU/JS fallback.
# Four consecutive rows of the SAME column subtree; no row sum is reordered.
@unsafe
def llb.gemv_tile(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

def tile.add(l: F32 & F32 & F32 & F32, r: F32 & F32 & F32 & F32) -> F32 & F32 & F32 & F32:
  (a, (b, (c, d))) = l
  (e, (f, (g, h))) = r
  ((a + e : F32), ((b + f : F32), ((c + g : F32), (d + h : F32))))

@unsafe
def tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = tile.cols(p, q, wb, x, base, rb, c, lo, n, g) tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      llb.gemv_tile(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def tile.put(valid: Bool, y: Array<F32>, i: U32, v: F32) -> Array<F32>:
  match valid:
    case True{}:
      Array.set(F32, y, i, v)
    case False{}:
      y

@unsafe
def tile.save(y: Array<F32>, +lo: U32, +n: U32, v: F32 & F32 & F32 & F32) -> Array<F32>:
  (a, (b, (c, d))) = v
  tile.put((3 < n : U32), tile.put((2 < n : U32), tile.put((1 < n : U32),
    tile.put((0 < n : U32), y, lo, a), (lo + 1 : U32), b), (lo + 2 : U32), c), (lo + 3 : U32), d)

@unsafe
def tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# y with rows 0..n-1 := W x; y's other slots are untouched. Each leaf owns
# up to four rows, reusing each x group; f still bounds the fork depth.
@unsafe
def gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, (rb / 4 : U32), 0, n)

# Q8_0 GEMV (tools/q8_pack.py): a row of c columns is [c int8 q][c/32 f16le d],
# rb = c * 17 / 16 bytes. Weight w = f32(q) * f32(d), ggml's dequantize_row_q8_0:
# one F32 product that never rounds (<= 19 significant bits), so each w is the
# F32 pack's and every product, group, tree and sum below is gemv's, bit for bit.

# f16 bits -> F32, exact for finite h (the pack rejects nonfinite d):
# (implicit bit | m) * 2^(max(e, 1) - 25), both factors and the product exact;
# e = 31 is inf/NaN, the payload shifted up, as numpy and ggml
# (tools/quant-fix.patch: the product gave 65536 / 98304)
def q8.f16(+h: U32) -> F32:
  +e = ((h >> 10n) .&. 31 : U32)
  +m = (h .&. 1023 : U32)
  +s = (((h >> 15n) : U32) << 31n : U32)
  +p = ((Bool.pick(U32, U32.is_eq(e, 0), 1, e) + 102 : U32) << 23n : U32)
  +mag = (U32.to_f32(Bool.pick(U32, U32.is_eq(e, 0), m, (m + 1024 : U32))) * K.f32_from_bits(p) : F32)
  K.f32_from_bits(Bool.pick(U32, U32.is_eq(e, 31), ((s .|. 2139095040 : U32) .|. (m << 13n : U32) : U32),
    (F32.bits(mag) .|. s : U32)))

# byte k of word w as int8, in F32
def q8.i8(+w: U32, +k: Nat) -> F32:
  +b = ((w >> k) .&. 255 : U32)
  Bool.pick(F32, (b < 128 : U32), U32.to_f32(b), (U32.to_f32(b) - 256.0 : F32))

# the f16 scale at byte at of wb
def q8.d(+wb: Bytes(), +at: U32) -> F32:
  q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))

# as grp.go: the four quants at byte w, scale s
def q8.grp(in_row: Bool, +wb: Bytes(), +w: U32, +s: F32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +q = Bytes.word_le(wb, U32.to_nat(w))
      ((((q8.i8(q, 0n) * s : F32) * x0 : F32) + ((q8.i8(q, 8n) * s : F32) * x1 : F32) : F32) +
        (((q8.i8(q, 16n) * s : F32) * x2 : F32) + ((q8.i8(q, 24n) * s : F32) * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: quants at r + 4g, its block's d at r + c + 2 (g / 8)
@unsafe
def q8.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  q8.grp((i < c : U32), wb, (r + i : U32), q8.d(wb, ((r + c : U32) + ((g >> 3n) * 2 : U32) : U32)),
    Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def q8.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q8.t0(wb, x, r, c, g) + q8.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def q8.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q8.t1(wb, x, r, c, g) + q8.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def q8.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q8.t2(wb, x, r, c, g) + q8.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def q8.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      q8.t0(wb, x, r, c, g)
    case 1n:
      q8.t1(wb, x, r, c, g)
    case 2n:
      q8.t2(wb, x, r, c, g)
    case _:
      q8.t3(wb, x, r, c, g)

@unsafe
def q8.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (q8.tree(p, wb, x, r, c, g) + q8.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      q8.tree.lo(d, wb, x, r, c, g)

@unsafe
def q8.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q8.rows(p, d, wb, x, Array.set(F32, y, i, q8.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def q8.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      q8.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

# the tile leaf, rows as q8.tile.row (the q8 lane's CUDA leaf, gemv-q8.patch,
# is not carried: under remap every GEMV is a warp bang)
@unsafe
def q8.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (q8.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (q8.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (q8.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        q8.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def q8.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = q8.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) q8.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      q8.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def q8.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, q8.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      q8.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = q8.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) q8.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a Q8_0 matrix of row bytes rb
@unsafe
def q8.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  q8.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 17 : U32) * 16 : U32), 0, n)

# y[i] := row element i of the Q8_0 row at byte at (c columns), i in lo..lo+k-1
@unsafe
def q8.load.run(k: Nat, +wb: Bytes(), +at: U32, +c: U32, y: Array<F32>, +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q8.load.run(p, wb, at, c, Array.set(F32, y, i, (q8.i8(Bytes.word_le(wb, U32.to_nat((at + i : U32))), 0n) *
        q8.d(wb, ((at + c : U32) + ((i >> 5n) * 2 : U32) : U32)) : F32)), (i + 1 : U32))

@unsafe
def q8.load.node(+f: Nat, +wb: Bytes(), +at: U32, +c: U32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      q8.load.run(U32.to_nat(n), wb, at, c, y, lo)
    case 1n+f1:
      +h = (n / 2 : U32)
      a b = q8.load.node(f1, wb, at, c, y, lo, h) q8.load.node(f1, wb, at, c, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, a, b)

# the 2^d columns of the Q8_0 row at byte at, dequantized (as load reads F32 rows)
@unsafe
def q8.load(+f: Nat, +wb: Bytes(), +at: U32, +d: Nat) -> Array<F32>:
  q8.load.node(f, wb, at, (1 << d : U32), Array.new(F32, d, 0.0), 0, (1 << d : U32))

# Q4_K GEMV (tools/q4k.py, tools/q4k.patch): a row of c columns is c/256
# super-blocks of 144 bytes kept whole in the GGUF layout
# [f16 d][f16 dmin][12 scales][128 qs]. Weight w = d * sc * nib - dmin * m,
# ggml's dequantize_row_q4_K; d*sc, dmin*m and d1*nib are exact, only
# d1*nib - m1 rounds, in F32, so each w is ggml's dequant word.

# byte at of wb, as U32
def q4k.by(+wb: Bytes(), +at: U32) -> U32:
  (Bytes.word_le(wb, U32.to_nat(at)) .&. 255 : U32)

# get_scale_min_k4: sub-block sb's scale (isd True) or min of the block at base
def q4k.sc(+wb: Bytes(), +base: U32, +sb: U32, +isd: Bool) -> U32:
  Bool.pick(U32, (sb < 4 : U32),
    Bool.pick(U32, isd, (q4k.by(wb, ((base + 4 : U32) + sb : U32)) .&. 63 : U32),
      (q4k.by(wb, ((base + 8 : U32) + sb : U32)) .&. 63 : U32)),
    Bool.pick(U32, isd,
      ((q4k.by(wb, ((base + 8 : U32) + sb : U32)) .&. 15 : U32) .|.
        ((q4k.by(wb, (base + sb : U32)) >> 6n : U32) << 4n : U32) : U32),
      ((q4k.by(wb, ((base + 8 : U32) + sb : U32)) >> 4n : U32) .|.
        ((q4k.by(wb, ((base + 4 : U32) + sb : U32)) >> 6n : U32) << 4n : U32) : U32)))

# nibble k of the 4-byte qs word, the high half selected by sh
def q4k.nib(+w: U32, +sh: U32, +k: Nat) -> U32:
  ((w >> (U32.to_nat(sh) + (8n * k) : Nat)) .&. 15 : U32)

# as q8.grp: the group's four dequantized weights, against x's four slots
def q4k.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c = (g * 4 : U32)
      +blk = (c >> 8n : U32)
      +rr = (c .&. 255 : U32)
      +b = (rr >> 6n : U32)
      +l = ((rr .&. 63 : U32) .&. 31 : U32)
      +hi = ((rr .&. 63 : U32) >> 5n : U32)
      +sb = ((b * 2 : U32) + hi : U32)
      +at = (r + (blk * 144 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
      +dm = q8.f16((Bytes.word_le(wb, U32.to_nat((at + 2 : U32))) .&. 65535 : U32))
      +d1 = (d * U32.to_f32(q4k.sc(wb, at, sb, True{})) : F32)
      +m1 = (dm * U32.to_f32(q4k.sc(wb, at, sb, False{})) : F32)
      +word = Bytes.word_le(wb, U32.to_nat(((at + 16 : U32) + (b * 32 : U32) + l : U32)))
      +sh = (hi * 4 : U32)
      +w0 = ((d1 * U32.to_f32(q4k.nib(word, sh, 0n)) : F32) - m1 : F32)
      +w1 = ((d1 * U32.to_f32(q4k.nib(word, sh, 1n)) : F32) - m1 : F32)
      +w2 = ((d1 * U32.to_f32(q4k.nib(word, sh, 2n)) : F32) - m1 : F32)
      +w3 = ((d1 * U32.to_f32(q4k.nib(word, sh, 3n)) : F32) - m1 : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def q4k.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  q4k.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def q4k.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q4k.t0(wb, x, r, c, g) + q4k.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def q4k.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q4k.t1(wb, x, r, c, g) + q4k.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def q4k.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q4k.t2(wb, x, r, c, g) + q4k.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def q4k.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      q4k.t0(wb, x, r, c, g)
    case 1n:
      q4k.t1(wb, x, r, c, g)
    case 2n:
      q4k.t2(wb, x, r, c, g)
    case _:
      q4k.t3(wb, x, r, c, g)

@unsafe
def q4k.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (q4k.tree(p, wb, x, r, c, g) + q4k.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      q4k.tree.lo(d, wb, x, r, c, g)

@unsafe
def q4k.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q4k.rows(p, d, wb, x, Array.set(F32, y, i, q4k.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def q4k.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      q4k.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def q4k.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (q4k.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (q4k.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (q4k.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        q4k.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def q4k.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = q4k.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) q4k.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      q4k.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def q4k.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, q4k.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      q4k.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = q4k.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) q4k.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a Q4_K matrix of row bytes rb
@unsafe
def q4k.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  q4k.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 144 : U32) * 256 : U32), 0, n)

# the weight of column c of the row at byte r (as q4k.grp's four, one column)
@unsafe
def q4k.w1(+wb: Bytes(), +r: U32, +c: U32) -> F32:
  +blk = (c >> 8n : U32)
  +rr = (c .&. 255 : U32)
  +b = (rr >> 6n : U32)
  +l = ((rr .&. 63 : U32) .&. 31 : U32)
  +hi = ((rr .&. 63 : U32) >> 5n : U32)
  +sb = ((b * 2 : U32) + hi : U32)
  +at = (r + (blk * 144 : U32) : U32)
  +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
  +dm = q8.f16((Bytes.word_le(wb, U32.to_nat((at + 2 : U32))) .&. 65535 : U32))
  +d1 = (d * U32.to_f32(q4k.sc(wb, at, sb, True{})) : F32)
  +m1 = (dm * U32.to_f32(q4k.sc(wb, at, sb, False{})) : F32)
  +word = Bytes.word_le(wb, U32.to_nat(((at + 16 : U32) + (b * 32 : U32) + l : U32)))
  +sh = (hi * 4 : U32)
  +nib = q4k.nib(word, sh, U32.to_nat((c .&. 3 : U32)))
  ((d1 * U32.to_f32(nib) : F32) - m1 : F32)

# y[i] := row element i of the Q4_K row at byte at, i in lo..lo+k-1
@unsafe
def q4k.load.run(k: Nat, +wb: Bytes(), +at: U32, y: Array<F32>, +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q4k.load.run(p, wb, at, Array.set(F32, y, i, q4k.w1(wb, at, i)), (i + 1 : U32))

@unsafe
def q4k.load.node(+f: Nat, +wb: Bytes(), +at: U32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      q4k.load.run(U32.to_nat(n), wb, at, y, lo)
    case 1n+f1:
      +h = (n / 2 : U32)
      a b = q4k.load.node(f1, wb, at, y, lo, h) q4k.load.node(f1, wb, at, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, a, b)

# 2^d Q4_K weights from byte at of wb, dequantized
@unsafe
def q4k.load(+f: Nat, +wb: Bytes(), +at: U32, +d: Nat) -> Array<F32>:
  q4k.load.node(f, wb, at, Array.new(F32, d, 0.0), 0, (1 << d : U32))

# IQ4_XS GEMV (tools/iq4xs.py, tools/iq4xs.patch): a row of c columns is c/256
# super-blocks of 136 bytes kept whole in the GGUF layout
# [f16 d][u16 scales_h][4 scales_l][128 qs]. Weight w = dl * kvalues_iq4nl[code],
# dl = d * (ls - 32), ggml's dequantize_row_iq4_xs. Every product is exact (f16
# significand * 5-bit (ls-32) * 7-bit codebook), so each w is ggml's dequant word.

# kvalues_iq4nl[k], ggml-common.h:1120, as F32 (the 16 signed codebook entries)
def iq4xs.kv(+k: Nat) -> F32:
  match k:
    case 0n:
      F32.neg(127.0)
    case 1n:
      F32.neg(104.0)
    case 2n:
      F32.neg(83.0)
    case 3n:
      F32.neg(65.0)
    case 4n:
      F32.neg(49.0)
    case 5n:
      F32.neg(35.0)
    case 6n:
      F32.neg(22.0)
    case 7n:
      F32.neg(10.0)
    case 8n:
      1.0
    case 9n:
      13.0
    case 10n:
      25.0
    case 11n:
      38.0
    case 12n:
      53.0
    case 13n:
      69.0
    case 14n:
      89.0
    case _:
      113.0

# as q8.grp: the group's four dequantized weights, against x's four slots
def iq4xs.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c4 = (g * 4 : U32)
      +blk = (c4 >> 8n : U32)
      +rr = (c4 .&. 255 : U32)
      +ib = (rr >> 5n : U32)
      +l = (rr .&. 15 : U32)
      +hi = (((rr .&. 31 : U32) >> 4n) : U32)
      +at = (r + (blk * 136 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
      +sl = (((q4k.by(wb, ((at + 4 : U32) + (ib >> 1n : U32) : U32)) >>
        U32.to_nat(((4 : U32) * (ib .&. 1 : U32) : U32)) : U32) .&. 15 : U32) .|.
        (((Bytes.word_le(wb, U32.to_nat((at + 2 : U32))) >>
          U32.to_nat(((2 : U32) * ib : U32)) : U32) .&. 3 : U32) << 4n : U32) : U32)
      +dl = (d * (U32.to_f32(sl) - 32.0 : F32) : F32)
      +word = (Bytes.word_le(wb, U32.to_nat(((at + 8 : U32) + ((ib * 16 : U32) + l : U32) : U32))) >>
        U32.to_nat((hi * 4 : U32)) : U32)
      +w0 = (dl * iq4xs.kv(U32.to_nat((word .&. 15 : U32))) : F32)
      +w1 = (dl * iq4xs.kv(U32.to_nat(((word >> 8n : U32) .&. 15 : U32))) : F32)
      +w2 = (dl * iq4xs.kv(U32.to_nat(((word >> 16n : U32) .&. 15 : U32))) : F32)
      +w3 = (dl * iq4xs.kv(U32.to_nat(((word >> 24n : U32) .&. 15 : U32))) : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def iq4xs.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  iq4xs.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def iq4xs.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (iq4xs.t0(wb, x, r, c, g) + iq4xs.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def iq4xs.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (iq4xs.t1(wb, x, r, c, g) + iq4xs.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def iq4xs.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (iq4xs.t2(wb, x, r, c, g) + iq4xs.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def iq4xs.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      iq4xs.t0(wb, x, r, c, g)
    case 1n:
      iq4xs.t1(wb, x, r, c, g)
    case 2n:
      iq4xs.t2(wb, x, r, c, g)
    case _:
      iq4xs.t3(wb, x, r, c, g)

@unsafe
def iq4xs.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (iq4xs.tree(p, wb, x, r, c, g) + iq4xs.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      iq4xs.tree.lo(d, wb, x, r, c, g)

@unsafe
def iq4xs.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      iq4xs.rows(p, d, wb, x, Array.set(F32, y, i, iq4xs.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def iq4xs.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      iq4xs.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def iq4xs.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (iq4xs.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (iq4xs.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (iq4xs.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        iq4xs.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def iq4xs.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = iq4xs.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) iq4xs.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      iq4xs.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def iq4xs.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, iq4xs.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      iq4xs.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = iq4xs.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) iq4xs.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over an IQ4_XS matrix of row bytes rb
@unsafe
def iq4xs.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  iq4xs.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 136 : U32) * 256 : U32), 0, n)

# the weight of column c of the row at byte r (as iq4xs.grp's four, one column)
@unsafe
def iq4xs.one(+wb: Bytes(), +r: U32, +c: U32) -> F32:
  +blk = (c >> 8n : U32)
  +rr = (c .&. 255 : U32)
  +ib = (rr >> 5n : U32)
  +l = (rr .&. 15 : U32)
  +hi = (((rr .&. 31 : U32) >> 4n) : U32)
  +at = (r + (blk * 136 : U32) : U32)
  +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
  +sl = (((q4k.by(wb, ((at + 4 : U32) + (ib >> 1n : U32) : U32)) >>
    U32.to_nat(((4 : U32) * (ib .&. 1 : U32) : U32)) : U32) .&. 15 : U32) .|.
    (((Bytes.word_le(wb, U32.to_nat((at + 2 : U32))) >>
      U32.to_nat(((2 : U32) * ib : U32)) : U32) .&. 3 : U32) << 4n : U32) : U32)
  +dl = (d * (U32.to_f32(sl) - 32.0 : F32) : F32)
  +word = (Bytes.word_le(wb, U32.to_nat(((at + 8 : U32) + ((ib * 16 : U32) + l : U32) : U32))) >>
    U32.to_nat((hi * 4 : U32)) : U32)
  (dl * iq4xs.kv(U32.to_nat((word .&. 15 : U32))) : F32)

# y[i] := row element i of the IQ4_XS row at byte at, i in lo..lo+k-1
@unsafe
def iq4xs.load.run(k: Nat, +wb: Bytes(), +at: U32, y: Array<F32>, +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      iq4xs.load.run(p, wb, at, Array.set(F32, y, i, iq4xs.one(wb, at, i)), (i + 1 : U32))

@unsafe
def iq4xs.load.node(+f: Nat, +wb: Bytes(), +at: U32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      iq4xs.load.run(U32.to_nat(n), wb, at, y, lo)
    case 1n+f1:
      +h = (n / 2 : U32)
      a b = iq4xs.load.node(f1, wb, at, y, lo, h) iq4xs.load.node(f1, wb, at, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, a, b)

# 2^d IQ4_XS weights from byte at of wb, dequantized
@unsafe
def iq4xs.load(+f: Nat, +wb: Bytes(), +at: U32, +d: Nat) -> Array<F32>:
  iq4xs.load.node(f, wb, at, Array.new(F32, d, 0.0), 0, (1 << d : U32))

# Q6_K GEMV (tools/q6k.py, tools/q6k.patch): a row of c columns is c/256
# super-blocks of 210 bytes kept whole in the GGUF layout
# [ql 128][qh 64][scales 16][d f16]. Weight w = (d * sc) * q, ggml's
# dequantize_row_q6_K; d*sc is exact (f16 significand * 8-bit int), (d*sc)*q is
# exact too (|q| <= 32), both in ggml's order. The bytes are read one at a time
# because 210 is not 4-aligned.

# byte at of wb, as U32
def q6k.by(+wb: Bytes(), +at: U32) -> U32:
  (Bytes.word_le(wb, U32.to_nat(at)) .&. 255 : U32)

# the signed 8-bit scale at byte at of wb, as F32
def q6k.sc8(+wb: Bytes(), +at: U32) -> F32:
  +b = q6k.by(wb, at)
  Bool.pick(F32, (b < 128 : U32), U32.to_f32(b), (U32.to_f32(b) - 256.0 : F32))

# the 6-bit code of ql byte q and qh byte h, ql shift nsh and qh shift sh,
# minus 32 (ggml's y = (d*sc)*((ql|qh<<4) - 32))
def q6k.code(+q: U32, +h: U32, +nsh: U32, +sh: U32) -> F32:
  +v = (((q >> U32.to_nat(nsh) : U32) .&. 15 : U32) .|.
    (((h >> U32.to_nat(sh) : U32) .&. 3 : U32) << 4n : U32) : U32)
  (U32.to_f32(v) - 32.0 : F32)

# as q8.grp: the group's four dequantized weights, against x's four slots
def q6k.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c4 = (g * 4 : U32)
      +blk = (c4 >> 8n : U32)
      +rr = (c4 .&. 255 : U32)
      +n = (rr >> 7n : U32)
      +rd = (rr .&. 127 : U32)
      +quad = (rd >> 5n : U32)
      +l = (rd .&. 31 : U32)
      +at = (r + (blk * 210 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat((at + 208 : U32))) .&. 65535 : U32))
      +sc = q6k.sc8(wb, ((at + 192 : U32) + ((n * 8 : U32) + ((quad * 2 : U32) + (l >> 4n : U32)) : U32) : U32))
      +ds = (d * sc : F32)
      +nsh = Bool.pick(U32, (2 <= quad : U32), 4, 0)
      +qsh = (quad * 2 : U32)
      +qlb = (at + ((n * 64 : U32) + (((quad .&. 1 : U32) * 32 : U32) + l : U32) : U32) : U32)
      +qhb = ((at + 128 : U32) + ((n * 32 : U32) + l : U32) : U32)
      +w0 = (ds * q6k.code(q6k.by(wb, qlb), q6k.by(wb, qhb), nsh, qsh) : F32)
      +w1 = (ds * q6k.code(q6k.by(wb, (qlb + 1 : U32)), q6k.by(wb, (qhb + 1 : U32)), nsh, qsh) : F32)
      +w2 = (ds * q6k.code(q6k.by(wb, (qlb + 2 : U32)), q6k.by(wb, (qhb + 2 : U32)), nsh, qsh) : F32)
      +w3 = (ds * q6k.code(q6k.by(wb, (qlb + 3 : U32)), q6k.by(wb, (qhb + 3 : U32)), nsh, qsh) : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def q6k.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  q6k.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def q6k.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q6k.t0(wb, x, r, c, g) + q6k.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def q6k.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q6k.t1(wb, x, r, c, g) + q6k.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def q6k.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q6k.t2(wb, x, r, c, g) + q6k.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def q6k.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      q6k.t0(wb, x, r, c, g)
    case 1n:
      q6k.t1(wb, x, r, c, g)
    case 2n:
      q6k.t2(wb, x, r, c, g)
    case _:
      q6k.t3(wb, x, r, c, g)

@unsafe
def q6k.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (q6k.tree(p, wb, x, r, c, g) + q6k.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      q6k.tree.lo(d, wb, x, r, c, g)

@unsafe
def q6k.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q6k.rows(p, d, wb, x, Array.set(F32, y, i, q6k.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def q6k.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      q6k.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def q6k.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (q6k.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (q6k.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (q6k.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        q6k.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def q6k.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = q6k.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) q6k.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      q6k.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def q6k.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, q6k.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      q6k.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = q6k.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) q6k.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a Q6_K matrix of row bytes rb
@unsafe
def q6k.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  q6k.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 210 : U32) * 256 : U32), 0, n)

# Q3_K GEMV (tools/q3k.py, tools/q3k.patch): a row of c columns is c/256
# super-blocks of 110 bytes [hmask 32][qs 64][scales 12][d f16]. Weight
# w = d * (scales[is] - 32) * code, ggml's dequantize_row_q3_K; both products
# are exact, in ggml's order. 110 is not 4-aligned, so bytes are read one at a
# time.

# byte at of wb, as U32
def q3k.by(+wb: Bytes(), +at: U32) -> U32:
  (Bytes.word_le(wb, U32.to_nat(at)) .&. 255 : U32)

# the unpacked scale byte `is` of the block at byte at, as F32 of (int8 - 32)
def q3k.scv(+wb: Bytes(), +at: U32, +si: U32) -> F32:
  +a0 = Bytes.word_le(wb, U32.to_nat((at + 96 : U32)))
  +a1 = Bytes.word_le(wb, U32.to_nat((at + 100 : U32)))
  +a2 = Bytes.word_le(wb, U32.to_nat((at + 104 : U32)))
  +tmp = a2
  +a2d = (((a0 >> 4n : U32) .&. 252645135 : U32) .|. (((tmp >> 4n : U32) .&. 50529027 : U32) << 4n : U32) : U32)
  +a3d = (((a1 >> 4n : U32) .&. 252645135 : U32) .|. (((tmp >> 6n : U32) .&. 50529027 : U32) << 4n : U32) : U32)
  +a0b = ((a0 .&. 252645135 : U32) .|. (((tmp .&. 50529027 : U32) << 4n : U32)) : U32)
  +a1b = ((a1 .&. 252645135 : U32) .|. ((((tmp >> 2n : U32) .&. 50529027 : U32) << 4n : U32)) : U32)
  +word = Bool.pick(U32, (si < 4 : U32), a0b,
    Bool.pick(U32, (si < 8 : U32), a1b,
      Bool.pick(U32, (si < 12 : U32), a2d, a3d)))
  +byte = ((word >> U32.to_nat(((si .&. 3 : U32) * 8 : U32)) : U32) .&. 255 : U32)
  +scf = Bool.pick(F32, (byte < 128 : U32), U32.to_f32(byte), (U32.to_f32(byte) - 256.0 : F32))
  (scf - 32.0 : F32)

# as q8.grp: the group's four dequantized weights, against x's four slots
def q3k.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c4 = (g * 4 : U32)
      +blk = (c4 >> 8n : U32)
      +rr = (c4 .&. 255 : U32)
      +n = (rr >> 7n : U32)
      +r2 = (rr .&. 127 : U32)
      +j = (r2 >> 5n : U32)
      +l = (r2 .&. 31 : U32)
      +w16 = (l >> 4n : U32)
      +lb = (l .&. 15 : U32)
      +at = (r + (blk * 110 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat((at + 108 : U32))) .&. 65535 : U32))
      +si = ((n * 8 : U32) + ((j * 2 : U32) + w16 : U32) : U32)
      +dl = (d * q3k.scv(wb, at, si) : F32)
      +qb = ((at + 32 : U32) + ((n * 32 : U32) + ((lb + (16 * w16 : U32) : U32)) : U32) : U32)
      +hb = (at + ((lb + (16 * w16 : U32) : U32)) : U32)
      +sh = (j * 2 : U32)
      +bit = (j + (n * 4 : U32) : U32)
      +c0 = (U32.to_f32(((q3k.by(wb, qb) >> U32.to_nat(sh) : U32) .&. 3 : U32)) -
        U32.to_f32(Bool.pick(U32, U32.is_eq(((q3k.by(wb, hb) >> U32.to_nat(bit) : U32) .&. 1 : U32), 0), 4, 0)) : F32)
      +c1 = (U32.to_f32(((q3k.by(wb, (qb + 1 : U32)) >> U32.to_nat(sh) : U32) .&. 3 : U32)) -
        U32.to_f32(Bool.pick(U32, U32.is_eq(((q3k.by(wb, (hb + 1 : U32)) >> U32.to_nat(bit) : U32) .&. 1 : U32), 0), 4, 0)) : F32)
      +c2 = (U32.to_f32(((q3k.by(wb, (qb + 2 : U32)) >> U32.to_nat(sh) : U32) .&. 3 : U32)) -
        U32.to_f32(Bool.pick(U32, U32.is_eq(((q3k.by(wb, (hb + 2 : U32)) >> U32.to_nat(bit) : U32) .&. 1 : U32), 0), 4, 0)) : F32)
      +c3 = (U32.to_f32(((q3k.by(wb, (qb + 3 : U32)) >> U32.to_nat(sh) : U32) .&. 3 : U32)) -
        U32.to_f32(Bool.pick(U32, U32.is_eq(((q3k.by(wb, (hb + 3 : U32)) >> U32.to_nat(bit) : U32) .&. 1 : U32), 0), 4, 0)) : F32)
      +w0 = (dl * c0 : F32)
      +w1 = (dl * c1 : F32)
      +w2 = (dl * c2 : F32)
      +w3 = (dl * c3 : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def q3k.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  q3k.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def q3k.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q3k.t0(wb, x, r, c, g) + q3k.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def q3k.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q3k.t1(wb, x, r, c, g) + q3k.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def q3k.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q3k.t2(wb, x, r, c, g) + q3k.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def q3k.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      q3k.t0(wb, x, r, c, g)
    case 1n:
      q3k.t1(wb, x, r, c, g)
    case 2n:
      q3k.t2(wb, x, r, c, g)
    case _:
      q3k.t3(wb, x, r, c, g)

@unsafe
def q3k.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (q3k.tree(p, wb, x, r, c, g) + q3k.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      q3k.tree.lo(d, wb, x, r, c, g)

@unsafe
def q3k.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q3k.rows(p, d, wb, x, Array.set(F32, y, i, q3k.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def q3k.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      q3k.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def q3k.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (q3k.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (q3k.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (q3k.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        q3k.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def q3k.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = q3k.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) q3k.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      q3k.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def q3k.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, q3k.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      q3k.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = q3k.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) q3k.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a Q3_K matrix of row bytes rb
@unsafe
def q3k.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  q3k.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 110 : U32) * 256 : U32), 0, n)

# byte at of wb, as U32 (shared by the ternary modules)
def q20.by(+wb: Bytes(), +at: U32) -> U32:
  (Bytes.word_le(wb, U32.to_nat(at)) .&. 255 : U32)

# Q2_0 ternary GEMV (tools/q20.py, tools/q20.patch): a row of c columns is
# c/64 super-blocks of 18 bytes [f16 d][qs 16]. Weight
# w = ((q - 1) * d), ggml's dequantize_row_q2_0; (q-1)*d is exact. A group of
# four columns is exactly one qs byte.
def q2.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c = (g * 4 : U32)
      +blk = (c / 64 : U32)
      +rr = (c .&. 63 : U32)
      +at = (r + (blk * 18 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
      +v = q20.by(wb, (at + (2 : U32) + (rr >> 2n : U32) : U32))
      +w0 = (d * (U32.to_f32((v .&. 3 : U32)) - 1.0 : F32) : F32)
      +w1 = (d * (U32.to_f32(((v >> 2n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      +w2 = (d * (U32.to_f32(((v >> 4n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      +w3 = (d * (U32.to_f32(((v >> 6n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

@unsafe
def q2.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  q2.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def q2.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q2.t0(wb, x, r, c, g) + q2.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def q2.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q2.t1(wb, x, r, c, g) + q2.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def q2.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (q2.t2(wb, x, r, c, g) + q2.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def q2.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      q2.t0(wb, x, r, c, g)
    case 1n:
      q2.t1(wb, x, r, c, g)
    case 2n:
      q2.t2(wb, x, r, c, g)
    case _:
      q2.t3(wb, x, r, c, g)

@unsafe
def q2.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (q2.tree(p, wb, x, r, c, g) + q2.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      q2.tree.lo(d, wb, x, r, c, g)

@unsafe
def q2.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      q2.rows(p, d, wb, x, Array.set(F32, y, i, q2.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def q2.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      q2.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def q2.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (q2.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (q2.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (q2.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        q2.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def q2.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = q2.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) q2.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      q2.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def q2.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, q2.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      q2.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = q2.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) q2.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a Q2_0 matrix of row bytes rb
@unsafe
def q2.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  q2.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 18 : U32) * 64 : U32), 0, n)

# PQ2_0 ternary GEMV (tools/q20.py, tools/q20.patch): a row of c columns is
# c/128 super-blocks of 34 bytes [f16 d][qs 32]. Weight
# w = ((q - 1) * d), ggml's dequantize_row_pq2_0; (q-1)*d is exact. A group of
# four columns is exactly one qs byte.
def pq.grp(in_row: Bool, +wb: Bytes(), +r: U32, +g: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +c = (g * 4 : U32)
      +blk = (c / 128 : U32)
      +rr = (c .&. 127 : U32)
      +at = (r + (blk * 34 : U32) : U32)
      +d = q8.f16((Bytes.word_le(wb, U32.to_nat(at)) .&. 65535 : U32))
      +v = q20.by(wb, (at + (2 : U32) + (rr >> 2n : U32) : U32))
      +w0 = (d * (U32.to_f32((v .&. 3 : U32)) - 1.0 : F32) : F32)
      +w1 = (d * (U32.to_f32(((v >> 2n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      +w2 = (d * (U32.to_f32(((v >> 4n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      +w3 = (d * (U32.to_f32(((v >> 6n : U32) .&. 3 : U32)) - 1.0 : F32) : F32)
      (((w0 * x0 : F32) + (w1 * x1 : F32) : F32) +
        ((w2 * x2 : F32) + (w3 * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

@unsafe
def pq.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  pq.grp((i < c : U32), wb, r, g, Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def pq.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (pq.t0(wb, x, r, c, g) + pq.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def pq.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (pq.t1(wb, x, r, c, g) + pq.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def pq.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (pq.t2(wb, x, r, c, g) + pq.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def pq.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      pq.t0(wb, x, r, c, g)
    case 1n:
      pq.t1(wb, x, r, c, g)
    case 2n:
      pq.t2(wb, x, r, c, g)
    case _:
      pq.t3(wb, x, r, c, g)

@unsafe
def pq.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (pq.tree(p, wb, x, r, c, g) + pq.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      pq.tree.lo(d, wb, x, r, c, g)

@unsafe
def pq.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      pq.rows(p, d, wb, x, Array.set(F32, y, i, pq.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def pq.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      pq.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

@unsafe
def pq.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (pq.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (pq.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (pq.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        pq.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def pq.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = pq.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) pq.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      pq.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def pq.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, pq.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      pq.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = pq.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) pq.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# gemv over a PQ2_0 matrix of row bytes rb
@unsafe
def pq.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  pq.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, ((rb / 34 : U32) * 128 : U32), 0, n)

# F16 GEMV (the quant-f16 lane's kernels/flat16.bend, tools/merge3-f16.patch):
# rows are f16le, rb = 2 * columns; each weight upconverted exactly on load
# (q8.f16: subnormals as m * 2^-24, inf/NaN kept; the pack has none),
# then every product, group, tree and sum is gemv's, bit for bit.
def h16.word(+b: Bytes(), +at: U32) -> U32:
  Bytes.word_le(b, U32.to_nat(at))

# the f16 at byte at (even) of b, as F32; reads the aligned word holding it
def h16.rh(+b: Bytes(), +at: U32) -> F32:
  q8.f16(((h16.word(b, (at .&. 4294967292 : U32)) >> Bool.pick(Nat, U32.is_eq((at .&. 2 : U32), 0), 0n, 16n)) .&. 65535 : U32))

# group at w of W against x's four slots, as flat.bend's grp.go
def h16.grp.go(in_row: Bool, +wb: Bytes(), +w: U32, x0: F32, x1: F32, x2: F32, x3: F32) -> F32:
  match in_row:
    case True{}:
      +a = h16.word(wb, w)
      +b = h16.word(wb, (w + 4 : U32))
      (((q8.f16((a .&. 65535 : U32)) * x0 : F32) + (q8.f16((a >> 16n : U32)) * x1 : F32) : F32) +
        ((q8.f16((b .&. 65535 : U32)) * x2 : F32) + (q8.f16((b >> 16n : U32)) * x3 : F32) : F32) : F32)
    case False{}:
      (((0.0 * x0 : F32) + (0.0 * x1 : F32) : F32) +
        ((0.0 * x2 : F32) + (0.0 * x3 : F32) : F32) : F32)

# group g of the row at byte r: columns 4g..4g+3
@unsafe
def h16.t0(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  +i = (g * 4 : U32)
  h16.grp.go((i < c : U32), wb, (r + (g * 8 : U32) : U32),
    Array.peek_f32(x, i), Array.peek_f32(x, (i + 1 : U32)),
    Array.peek_f32(x, (i + 2 : U32)), Array.peek_f32(x, (i + 3 : U32)))

@unsafe
def h16.t1(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (h16.t0(wb, x, r, c, g) + h16.t0(wb, x, r, c, (g + 1 : U32)) : F32)

@unsafe
def h16.t2(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (h16.t1(wb, x, r, c, g) + h16.t1(wb, x, r, c, (g + 2 : U32)) : F32)

@unsafe
def h16.t3(+wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  (h16.t2(wb, x, r, c, g) + h16.t2(wb, x, r, c, (g + 4 : U32)) : F32)

@unsafe
def h16.tree.lo(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 0n:
      h16.t0(wb, x, r, c, g)
    case 1n:
      h16.t1(wb, x, r, c, g)
    case 2n:
      h16.t2(wb, x, r, c, g)
    case _:
      h16.t3(wb, x, r, c, g)

@unsafe
def h16.tree(+d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match d:
    case 1n+1n+1n+1n+q:
      +p = Nat.add(q, 3n)
      (h16.tree(p, wb, x, r, c, g) + h16.tree(p, wb, x, r, c, (g + (1 << p : U32) : U32)) : F32)
    case _:
      h16.tree.lo(d, wb, x, r, c, g)

# rows i.. (k of them) into y
@unsafe
def h16.rows(k: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +rb: U32, +c: U32,
  +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      h16.rows(p, d, wb, x, Array.set(F32, y, i, h16.tree(d, wb, x, (base + (i * rb : U32) : U32), c, 0)), base, rb, c,
        (i + 1 : U32))

@unsafe
def h16.tile.row(valid: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +r: U32, +c: U32, +g: U32) -> F32:
  match valid:
    case True{}:
      h16.tree(d, wb, x, r, c, g)
    case False{}:
      0.0

# the tile leaf, rows as h16.tile.row (the f16 lane's CUDA leaf,
# gemv-tile16.patch, is not carried: under remap every GEMV is a warp bang)
@unsafe
def h16.leaf(+d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  (h16.tile.row((0 < n : U32), d, wb, x, (base + (lo * rb : U32) : U32), c, g),
    (h16.tile.row((1 < n : U32), d, wb, x, (base + ((lo + 1 : U32) * rb : U32) : U32), c, g),
      (h16.tile.row((2 < n : U32), d, wb, x, (base + ((lo + 2 : U32) * rb : U32) : U32), c, g),
        h16.tile.row((3 < n : U32), d, wb, x, (base + ((lo + 3 : U32) * rb : U32) : U32), c, g))))

@unsafe
def h16.tile.cols(+f: Nat, +d: Nat, +wb: Bytes(), +x: Array<F32>, +base: U32, +rb: U32,
  +c: U32, +lo: U32, +n: U32, +g: U32) -> F32 & F32 & F32 & F32:
  match f d:
    case 1n+p 1n+q:
      a b = h16.tile.cols(p, q, wb, x, base, rb, c, lo, n, g) h16.tile.cols(p, q, wb, x, base, rb, c, lo, n, (g + (1 << q : U32) : U32))
      tile.add(a, b)
    case _ _:
      h16.leaf(d, wb, x, base, rb, c, lo, n, g)

@unsafe
def h16.tile.node(+f: Nat, small: Bool, +d: Nat, +wb: Bytes(), +x: Array<F32>, +y: Array<F32>,
  +base: U32, +rb: U32, +c: U32, +lo: U32, +n: U32) -> Array<F32>:
  match f small:
    case _ True{}:
      tile.save(y, lo, n, h16.tile.cols(f, d, wb, x, base, rb, c, lo, n, 0))
    case 0n _:
      h16.rows(U32.to_nat(n), d, wb, x, y, base, rb, c, lo)
    case 1n+p False{}:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      a b = h16.tile.node(p, (h <= 4 : U32), d, wb, x, y, base, rb, c, lo, h) h16.tile.node(p, (m <= 4 : U32), d, wb, x, y, base, rb, c, (lo + h : U32), m)
      Array.join(F32, a, b)

# y with rows 0..n-1 := W x, W f16le [n, rb/2] from byte base; as F.gemv
@unsafe
def h16.gemv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  h16.tile.node(f, (n <= 4 : U32), d, wb, x, y, base, rb, (rb / 2 : U32), 0, n)

# y[i] := the f16le at byte at + 2 i, as F32, for i in lo..lo+k-1
@unsafe
def h16.load.run(k: Nat, +wb: Bytes(), +at: U32, y: Array<F32>, +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      h16.load.run(p, wb, at, Array.set(F32, y, i, h16.rh(wb, (at + (i * 2 : U32) : U32))), (i + 1 : U32))

@unsafe
def h16.load.node(+f: Nat, +wb: Bytes(), +at: U32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      h16.load.run(U32.to_nat(n), wb, at, y, lo)
    case 1n+f1:
      +h = (n / 2 : U32)
      a b = h16.load.node(f1, wb, at, y, lo, h) h16.load.node(f1, wb, at, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, a, b)

# 2^d f16le values from byte at of wb, as F32
@unsafe
def h16.load(+f: Nat, +wb: Bytes(), +at: U32, +d: Nat) -> Array<F32>:
  h16.load.node(f, wb, at, Array.new(F32, d, 0.0), 0, (1 << d : U32))

# gemv over F32 rows, or Q8_0 rows
@unsafe
def mv.q8(q8: Bool, +f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32) -> Array<F32>:
  match q8:
    case True{}:
      q8.gemv(f, wb, x, y, base, n, d, rb)
    case False{}:
      gemv(f, wb, x, y, base, n, d, rb)

# rw, the row word, as llb_mail reads it (tools/quant-fix.patch; format 4 Q6_K
# added by tools/q6k.patch). Tagged (the engine's rw, from the manifest): bit 30
# set, the format in bits 28-29 (0 F32, 1 Q8_0, 2 F16, 3 Q4_K), bit 31 adds 4
# (Q6_K, Q3_K, Q2_0, PQ2_0) and bit 27 adds 8 (IQ4_XS, tools/iq4xs.patch).
# Untagged, the old sniff: bit 31 F16, else 17 | rb Q8_0 (F32 rows are 4c),
# else F32.
def mv.tg(+rw: U32) -> Bool:
  U32.is_eq((rw .&. 1073741824 : U32), 1073741824)

def mv.by(+a: U32, +k: U32) -> Bool:
  U32.is_eq(((a / k : U32) * k : U32), a)

def mv.rb(+rw: U32) -> U32:
  (rw .&. Bool.pick(U32, mv.tg(rw), 134217727, 2147483647) : U32)

def mv.fm(+rw: U32) -> U32:
  Bool.pick(U32, mv.tg(rw),
    ((((rw >> 28n) .&. 3 : U32) + (((rw >> 31n) .&. 1 : U32) << 2n : U32) : U32) +
      (((rw >> 27n) .&. 1 : U32) << 3n : U32) : U32),
    Bool.pick(U32, (2147483647 < rw : U32), 2, Bool.pick(U32, mv.by(mv.rb(rw), 17), 1, 0)))

# refused: a tag with bit 31 (a format past PQ2_0), a Q8_0 tag without 17 | rb, a
# Q4_K tag without 144 | rb, a Q6_K tag without 210 | rb, an IQ4_XS tag without
# 136 | rb, and an untagged 17 | rb that is also 16 | rb, read as F32 (c = rb/4)
# or Q8_0 (c = rb * 16 / 17) alike: the F32 c = 4352 the old sniff took for Q8_0
def mv.ok(+rw: U32) -> Bool:
  U32.is_eq(Bool.pick(U32, mv.tg(rw),
    Bool.pick(U32, U32.is_eq(mv.fm(rw), 1), Bool.pick(U32, mv.by(mv.rb(rw), 17), 0, 1),
      Bool.pick(U32, U32.is_eq(mv.fm(rw), 3), Bool.pick(U32, mv.by(mv.rb(rw), 144), 0, 1),
        Bool.pick(U32, U32.is_eq(mv.fm(rw), 4), Bool.pick(U32, mv.by(mv.rb(rw), 210), 0, 1),
          Bool.pick(U32, U32.is_eq(mv.fm(rw), 5), Bool.pick(U32, mv.by(mv.rb(rw), 110), 0, 1),
            Bool.pick(U32, U32.is_eq(mv.fm(rw), 6), Bool.pick(U32, mv.by(mv.rb(rw), 18), 0, 1),
              Bool.pick(U32, U32.is_eq(mv.fm(rw), 7), Bool.pick(U32, mv.by(mv.rb(rw), 34), 0, 1),
                Bool.pick(U32, U32.is_eq(mv.fm(rw), 8), Bool.pick(U32, mv.by(mv.rb(rw), 136), 0, 1), 0))))))),
    Bool.pick(U32, U32.is_eq(mv.fm(rw), 1), Bool.pick(U32, mv.by(mv.rb(rw), 16), 1, 0), 0)), 0)

# pure Bend has no abort: a Nat past 2^48 - 1 stops both runtimes ("bend: a
# Nat past the largest immediate 2^48-1", exit 1), as bad = 4 fails the device
def mv.stop(z: Bool, y: Array<F32>) -> Array<F32>:
  match z:
    case True{}:
      y
    case False{}:
      y

def mv.refuse(+rw: U32, y: Array<F32>) -> Array<F32>:
  mv.stop(Nat.is_le(Nat.mul(Nat.mul(4294967295n, 65536n), U32.to_nat((rw .|. 2 : U32))), 0n), y)

@unsafe
def mv.h16(hf: Bool, +f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rw: U32) -> Array<F32>:
  match hf:
    case True{}:
      h16.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case False{}:
      mv.q8(U32.is_eq(mv.fm(rw), 1), f, wb, x, y, base, n, d, mv.rb(rw))

# one GEMV per format: 0 F32, 1 Q8_0, 2 F16, 3 Q4_K, 4 Q6_K, 5 Q3_K, 6 Q2_0, 7 PQ2_0, 8 IQ4_XS
@unsafe
def mv.byfm(+fm: Nat, +f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rw: U32) -> Array<F32>:
  match fm:
    case 0n:
      gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 1n:
      q8.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 2n:
      h16.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 3n:
      q4k.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 4n:
      q6k.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 5n:
      q3k.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 6n:
      q2.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 7n:
      pq.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case 8n:
      iq4xs.gemv(f, wb, x, y, base, n, d, mv.rb(rw))
    case _:
      mv.refuse(rw, y)

@unsafe
def mv.go(ok: Bool, +f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rw: U32) -> Array<F32>:
  match ok:
    case True{}:
      mv.byfm(U32.to_nat(mv.fm(rw)), f, wb, x, y, base, n, d, rw)
    case False{}:
      mv.refuse(rw, y)

@unsafe
def mv(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat, +rb: U32) -> Array<F32>:
  mv.go(mv.ok(rb), f, wb, x, y, base, n, d, rb)

# The same gemv as one native launch (tools/remap.patch): call it as a bang;
# the device parks the task and the host runs it as the contract kernel, a
# warp per row over this tree (9 <= d <= 11). This body is the CPU/JS path.
@unsafe
def llb.gemv_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32) -> Array<F32>:
  mv(f, wb, x, y, base, n, d, rb)

# Fork writers
# ------------
# Each splits an index range as node splits rows (top f levels forked) and
# writes disjoint slots of one shared block; no element's value depends on
# the split.

# y[i] := the f32le word of wb at byte at + 4 i, for i in lo..lo+k-1
@unsafe
def load.run(k: Nat, +wb: Bytes(), +at: U32, y: Array<F32>, +i: U32) -> Array<F32>:
  match k:
    case 0n:
      y
    case 1n+p:
      load.run(p, wb, at, Array.set(F32, y, i, D.rd(wb, (at + (i * 4 : U32) : U32))), (i + 1 : U32))

@unsafe
def load.node(+f: Nat, +wb: Bytes(), +at: U32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      load.run(U32.to_nat(n), wb, at, y, lo)
    case 1n+f1:
      +h = (n / 2 : U32)
      a b = load.node(f1, wb, at, y, lo, h) load.node(f1, wb, at, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, a, b)

# 2^d F32 from byte at of wb (as K.vec reads the same bytes)
@unsafe
def load(+f: Nat, +wb: Bytes(), +at: U32, +d: Nat) -> Array<F32>:
  load.node(f, wb, at, Array.new(F32, d, 0.0), 0, (1 << d : U32))

def got(r: Array<F32> & F32) -> F32:
  (a, v) = r
  v

# argmax: K.argmax's pick (the first NaN, else the first max) over a's
# slots lo..lo+k-1, slot skip read as -inf (K.put's second-pick mask). The
# pick is the same for any split, so the leaves run it in index order.
@unsafe
def am.run(k: Nat, +a: Array<F32>, +skip: U32, +i: U32, best: F32 & U32) -> F32 & U32:
  match k:
    case 0n:
      best
    case 1n+p:
      am.run(p, a, skip, (i + 1 : U32), K.amax.pick(best,
        (Bool.pick(F32, U32.is_eq(i, skip), K.neg_inf(), got(Array.get(F32, a, i))), i)))

@unsafe
def am.node(+f: Nat, +a: Array<F32>, +skip: U32, +lo: U32, +n: U32) -> F32 & U32:
  match f:
    case 0n:
      am.run(U32.to_nat((n - 1 : U32)), a, skip, (lo + 1 : U32),
        (Bool.pick(F32, U32.is_eq(lo, skip), K.neg_inf(), got(Array.get(F32, a, lo))), lo))
    case 1n+f1:
      +h = (n / 2 : U32)
      x y = am.node(f1, a, skip, lo, h) am.node(f1, a, skip, (lo + h : U32), (n - h : U32))
      K.amax.pick(x, y)

# (value, index) of a's first max over n slots (n >= 2^f), slot skip as -inf
@unsafe
def argmax(+f: Nat, +a: Array<F32>, +skip: U32, +n: U32) -> F32 & U32:
  am.node(f, a, skip, 0, n)

# Fill: y[i] := k(a, b, c, s, i) for i in lo..lo+n-1
# -------------------------------------------------
# k reads a and b (shared, possibly y itself: a lane reads slot i before it
# writes it, and no lane reads another's slot of y) and returns slot i's value.

@unsafe
def fill.run(~k: Array<F32> -> Array<F32> -> U32 -> F32 -> U32 -> F32, n: Nat, +a: Array<F32>,
  +b: Array<F32>, +c: U32, +s: F32, y: Array<F32>, +i: U32) -> Array<F32>:
  match n:
    case 0n:
      y
    case 1n+p:
      fill.run(~k, p, a, b, c, s, Array.set(F32, y, i, k(a, b, c, s, i)), (i + 1 : U32))

@unsafe
def fill(~k: Array<F32> -> Array<F32> -> U32 -> F32 -> U32 -> F32, +f: Nat, +a: Array<F32>,
  +b: Array<F32>, +c: U32, +s: F32, +y: Array<F32>, +lo: U32, +n: U32) -> Array<F32>:
  match f:
    case 0n:
      fill.run(~k, U32.to_nat(n), a, b, c, s, y, lo)
    # two levels' quarters in one fork: the same leaves, half the grow rounds
    case 1n+1n+f2:
      +h = (n / 2 : U32)
      +m = (n - h : U32)
      +h1 = (h / 2 : U32)
      +m1 = (m / 2 : U32)
      +mid = (lo + h : U32)
      l0 l1 r0 r1 = fill(~k, f2, a, b, c, s, y, lo, h1) fill(~k, f2, a, b, c, s, y, (lo + h1 : U32), (h - h1 : U32)) fill(~k, f2, a, b, c, s, y, mid, m1) fill(~k, f2, a, b, c, s, y, (mid + m1 : U32), (m - m1 : U32))
      Array.join(F32, Array.join(F32, l0, l1), Array.join(F32, r0, r1))
    case 1n+f1:
      +h = (n / 2 : U32)
      l r = fill(~k, f1, a, b, c, s, y, lo, h) fill(~k, f1, a, b, c, s, y, (lo + h : U32), (n - h : U32))
      Array.join(F32, l, r)

@unsafe
def at(+a: Array<F32>, +i: U32) -> F32:
  Array.peek_f32(a, i)

# Reductions over index ranges: the balanced tree of ops.bend's vectors (a
# node over [lo, lo+2^d) splits at lo+2^(d-1), left + right). All but sumsq (rms's
# top-level sum) are serial: they run inside a fill's lane, where forking
# only buys spawn overhead.

# K.sumsq's sum over a[lo..lo+2^d)
@unsafe
def sumsq(+d: Nat, +a: Array<F32>, +lo: U32) -> F32:
  match d:
    case 0n:
      +v = at(a, lo)
      (v * v : F32)
    case 1n+p:
      x y = sumsq(p, a, lo) sumsq(p, a, (lo + (1 << p : U32) : U32))
      (x + y : F32)

# K.dot's sum of c[co+i] * q[qo+i] over 2^d
@unsafe
def dot(+d: Nat, +c: Array<F32>, +co: U32, +q: Array<F32>, +qo: U32) -> F32:
  match d:
    case 0n:
      (at(c, co) * at(q, qo) : F32)
    case 1n+p:
      +h = (1 << p : U32)
      (dot(p, c, co, q, qo) + dot(p, c, (co + h : U32), q, (qo + h : U32)) : F32)

# score t of a head at base, K.mmax's mask (t > pos -> -inf)
@unsafe
def masked(+s: Array<F32>, +base: U32, +t: U32, +pos: U32) -> F32:
  Bool.pick(F32, (t <= pos : U32), at(s, (base + t : U32)), K.neg_inf())

# K.mmax's max over slots lo..lo+2^d
@unsafe
def mmax(+d: Nat, +s: Array<F32>, +base: U32, +lo: U32, +pos: U32) -> F32:
  match d:
    case 0n:
      masked(s, base, lo, pos)
    case 1n+p:
      F32.max(mmax(p, s, base, lo, pos), mmax(p, s, base, (lo + (1 << p : U32) : U32), pos))

# K.exps's sum
@unsafe
def esum(+d: Nat, +s: Array<F32>, +base: U32, +lo: U32, +pos: U32, +m: F32) -> F32:
  match d:
    case 0n:
      K.exp32((masked(s, base, lo, pos) - m : F32))
    case 1n+p:
      (esum(p, s, base, lo, pos, m) + esum(p, s, base, (lo + (1 << p : U32) : U32), pos, m) : F32)

# K.vsum's element j: the sum over slots t of V_t[g][j] * p_t
@unsafe
def vsum(+d: Nat, +p: Array<F32>, +pb: U32, +v: Array<F32>, +vo: U32, +lo: U32) -> F32:
  match d:
    case 0n:
      (at(v, (vo + (lo * 256 : U32) : U32)) * at(p, (pb + lo : U32)) : F32)
    case 1n+q:
      (vsum(q, p, pb, v, vo, lo) + vsum(q, p, pb, v, vo, (lo + (1 << q : U32) : U32)) : F32)

# Slot kernels for fill
# ---------------------

# a[i - c] (a copy into y from offset c)
@unsafe
def k.at.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  at(a, (i - c : U32))

@unsafe
def k.at(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.at.go(a, b, c, s, i)

# K.vadd: a + b
@unsafe
def k.add.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  (at(a, i) + at(b, i) : F32)

@unsafe
def k.add(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.add.go(a, b, c, s, i)

# K.scale_mul: (a * s) * w, w = b
@unsafe
def k.scale.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  ((at(a, i) * s : F32) * at(b, i) : F32)

@unsafe
def k.scale(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.scale.go(a, b, c, s, i)

# K.vmul(K.silu(a), b)
@unsafe
def k.silu.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  (K.silu1(at(a, i)) * at(b, i) : F32)

@unsafe
def k.silu(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.silu.go(a, b, c, s, i)

def rope.even(e: Bool, +x0: F32, +x1: F32, +co: F32, +si: F32) -> F32:
  match e:
    case True{}:
      ((x0 * co : F32) - (x1 * si : F32) : F32)
    case False{}:
      ((x0 * si : F32) + (x1 * co : F32) : F32)

# K.rope's slot i - c of a at position s: pair (x0, x1) = a[2p], a[2p+1]
@unsafe
def k.rope.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  +j = (i - c : U32)
  +e = (j .&. 4294967294 : U32)
  +t = K.rope.theta(((j >> 1n) .&. 63 : U32), s)
  rope.even(U32.is_eq(j, e), at(a, e), at(a, (e + 1 : U32)), K.cos32(t), K.sin32(t))

@unsafe
def k.rope(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.rope.go(a, b, c, s, i)

# The attention slots use c = dT | pos << 5: 2^dT cache slots, query at pos.

# K.scores: slot h * 2^dT + t = (q_h . K_t[h >> 3]) * kq_scale, a = q, b = K cache
@unsafe
def k.score.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  +dT = U32.to_nat((c .&. 31 : U32))
  +h = (i >> dT : U32)
  +t = (i - (h << dT : U32) : U32)
  (dot(7n, b, ((t * 256 : U32) + ((h >> 3n : U32) * 128 : U32) : U32), a, (h * 128 : U32)) * K.kq_scale() : F32)

@unsafe
def k.score(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.score.go(a, b, c, s, i)

# K.softmax over a head's slots, a = scores
@unsafe
def k.prob.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  +d = U32.to_nat((c .&. 31 : U32))
  +pos = (c >> 5n : U32)
  +h = (i >> d : U32)
  +base = (h << d : U32)
  +m = mmax(d, a, base, 0, pos)
  (K.exp32((masked(a, base, (i - base : U32), pos) - m : F32)) / esum(d, a, base, 0, pos, m) : F32)

@unsafe
def k.prob(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.prob.go(a, b, c, s, i)

# K.weighted: slot h * 128 + j, a = probs, b = V cache
@unsafe
def k.wsum.go(+a: Array<F32>, +b: Array<F32>, +c: U32, +s: F32, +i: U32) -> F32:
  +dT = U32.to_nat((c .&. 31 : U32))
  +h = (i >> 7n : U32)
  vsum(dT, a, (h << dT : U32), b, (((h >> 3n : U32) * 128 : U32) + (i .&. 127 : U32) : U32), 0)

@unsafe
def k.wsum(a: Array<F32>, b: Array<F32>, c: U32, s: F32, i: U32) -> F32:
  k.wsum.go(a, b, c, s, i)

# K.rms_norm(x, the 2048 F32 at byte at of wb, 2048.0)
@unsafe
def rms(+wb: Bytes(), +at: U32, +x: Array<F32>) -> Array<F32>:
  +w = load(6n, wb, at, 11n)
  +s = (1.0 / F32.sqrt(((sumsq(11n, x, 0) / 2048.0 : F32) + K.eps() : F32)) : F32)
  fill(~k.scale, 6n, x, w, 0, s, Array.new(F32, 11n, 0.0), 0, 2048)

# K.qknorm(k, v, g, d)
@unsafe
def qknorm(+k: Nat, +v: Array<F32>, +g: Array<F32>, +d: F32) -> Array<F32>:
  K.qknorm(k, v, g, d)

# K.output_gate(x, g)
@unsafe
def output_gate(+x: Array<F32>, +g: Array<F32>) -> Array<F32>:
  K.output_gate(x, g)

# Qwen3.5 hybrid device bodies
# ----------------------------
# The gated linear attention ops of kernels/ops.bend, as device bodies: the
# CPU/JS body is ops.bend's tree exactly (bit-exact with the emitted CPU lane);
# a CUDA patch may replace each with its warp kernel. Weight bytes are read
# from wb at at in F32 (the pack stores conv1d/ssm_norm as F32), as vec/rms do.
# `d`/`k` are ops.bend's: conv1d's channel depth, l2norm/gated_* head depth.

# causal depthwise conv1d (kernel 4) + SiLU over 2^d channels; w is [2^d,4]
# F32 at byte at of wb, x the 2^d inputs, h0..h2 the three previous inputs.
@unsafe
def llb.conv1d(+d: Nat, +wb: Bytes(), +at: U32, +x: Array<F32>, h0: Array<F32>, h1: Array<F32>,
  h2: Array<F32>) -> Array<F32>:
  K.conv1d(d, load(6n, wb, at, Nat.add(d, 2n)), x, h0, h1, h2)

# per-head L2 norm + scale over 2^k heads of 128 (sc = sqrt(128) for q)
@unsafe
def llb.l2norm(+k: Nat, +v: Array<F32>, +sc: F32) -> Array<F32>:
  K.l2norm(k, v, sc)

# per-head gated RMSNorm over 2^k heads of 128; w [128] F32 at byte at of wb
@unsafe
def llb.gated_rmsnorm(+k: Nat, +o: Array<F32>, +wb: Bytes(), +at: U32, +z: Array<F32>) -> Array<F32>:
  K.gated_rmsnorm(k, o, load(6n, wb, at, 7n), z, 128.0)

# the recurrent gated delta update over 2^k value heads; returns (out, S')
@unsafe
def llb.gated_delta(+k: Nat, +S: Array<F32>, +q: Array<F32>, +kk: Array<F32>, +v: Array<F32>,
  +dec: Array<F32>, +bet: Array<F32>) -> Array<F32> & Array<F32>:
  K.gated_delta(k, S, q, kk, v, dec, bet)

# Hybrid warp bangs (tools/payload-hybrid.patch)
# ---------------------------------------------
# The four Qwen3.5 hybrid ops as host-served warp bangs, the same FID/warp
# discipline as llb.rms_warp: call each as a bang; the device parks the task and
# the host runs the matching llb_* kernel into the preallocated output y (field
# index 3, so the generic delivery hands y back). The body here is the CPU/JS
# tree contract, never run on device. `scb` is the l2 scale's F32 bits.

# l2norm over 2^k heads of 128: y := v * (1 / max(sqrt(sumsq), eps)) / sc
@unsafe
def llb.l2norm_warp(+f: Nat, +v: Array<F32>, +scb: U32, y: Array<F32>, +k: Nat) -> Array<F32>:
  K.l2norm(k, v, K.f32_from_bits(scb))

# causal depthwise conv1d (kernel 4 + SiLU) over 2^d channels; w [2^d,4] F32 at
# byte at of wb, x the 2^d inputs, h0..h2 the three previous inputs
@unsafe
def llb.conv1d_warp(+f: Nat, +wb: Bytes(), +at: U32, y: Array<F32>, x: Array<F32>,
  h0: Array<F32>, h1: Array<F32>, h2: Array<F32>, +d: Nat) -> Array<F32>:
  K.conv1d(d, load(6n, wb, at, Nat.add(d, 2n)), x, h0, h1, h2)

# gated RMSNorm over 2^k heads of 128; w [128] F32 at byte at of wb
@unsafe
def llb.gated_rmsnorm_warp(+f: Nat, +o: Array<F32>, +wb: Bytes(), y: Array<F32>, +at: U32,
  z: Array<F32>, +k: Nat) -> Array<F32>:
  K.gated_rmsnorm(k, o, load(6n, wb, at, 7n), z, 128.0)

# gated delta recurrence over 2^k value heads; y := out, S updated in place
# (the host kernel writes S'), so the caller reuses its own S as the next state
def gd.pair.out(r: Array<F32> & Array<F32>) -> Array<F32>:
  (o, _s) = r
  o

@unsafe
def llb.gated_delta_warp(+f: Nat, +q: Array<F32>, +S: Array<F32>, y: Array<F32>, +kk: Array<F32>,
  +v: Array<F32>, +dec: Array<F32>, +bet: Array<F32>, +k: Nat) -> Array<F32>:
  gd.pair.out(K.gated_delta(k, S, q, kk, v, dec, bet))

# The standalone rms_norm as a host-served warp bang (tools/payload-rmsnorm.patch,
# the hybrid bangs' shape): the device parks the task and the host runs
# llb_rmsnorm into the preallocated output y (field index 3). This body is the
# CPU/JS tree contract: K.rms_norm(x, the 2^d F32 at byte at of wb, 2^d).
@unsafe
def llb.rmsnorm_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +at: U32, +d: Nat) -> Array<F32>:
  K.rms_norm(x, load(6n, wb, at, d), U32.to_f32((1 << d : U32)))

# The elementwise add / multiply as host-served warp bangs
# (tools/payload-vadd.patch): y := x + z / x * z over 2^d, the output y at
# field index 3. ops.bend's K.vadd / K.vmul are pure maps, so a flat index
# kernel is the same tree leaf for leaf.
@unsafe
def llb.vadd_warp(+f: Nat, x: Array<F32>, z: Array<F32>, y: Array<F32>, +d: Nat) -> Array<F32>:
  K.vadd(x, z)

@unsafe
def llb.vmul_warp(+f: Nat, x: Array<F32>, z: Array<F32>, y: Array<F32>, +d: Nat) -> Array<F32>:
  K.vmul(x, z)

# The SwiGLU silu-mul as one host-served warp bang (tools/payload-silumul.patch):
# y := silu(g) * u over 2^d, output y at field index 3. The host runs the
# existing llb_silu kernel (interiors-ffn.patch), so this body is the CPU/JS
# contract, K.vmul(K.silu(g), u), leaf for leaf: K.silu is Array.map(silu1),
# K.vmul is __fmul_rn; llb_silu is (float)(g/(1+exp(-g))) * u through attn_exp.
@unsafe
def llb.silu_mul_warp(+f: Nat, +g: Array<F32>, +u: Array<F32>, y: Array<F32>, +d: Nat) -> Array<F32>:
  K.vmul(K.silu(g), u)

# The FFN as one host-served block bang (tools/payload-fbn.patch): the whole
# lay.res tail -- y := x + Wd (silu(Wg h) * Wu h), h = rms(post, x + a) -- as a
# single serve boundary. The CPU/JS body is exactly the sequence
# llb.rmsnorm_warp + two llb.gemv_warp + llb.silu_mul_warp + llb.gemv_warp +
# llb.vadd_warp as lay.res ran them, leaf for leaf: K.rms_norm(x1, post, 4096),
# the gate and up GEMVs over 4096 columns (rg = q8rw(4096), d = 10), the SwiGLU
# mul, the down GEMV over 12288 (rd = q8rw(12288), d = 12), the residual add.
# y is field index 3 (the generic delivery hands it back); x1 = x + a.
@unsafe
def llb.ffn_block_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, a: Array<F32>,
  +post: U32, +fg: U32, +fu: U32, +fd: U32, +rg: U32, +rd: U32) -> Array<F32>:
  +x1 = K.vadd(x, a)
  +h2 = K.rms_norm(x1, load(6n, wb, post, 12n), 4096.0)
  +g = mv(14n, wb, h2, Array.new(F32, 14n, 0.0), fg, 12288, 10n, rg)
  +u = mv(14n, wb, h2, Array.new(F32, 14n, 0.0), fu, 12288, 10n, rg)
  +a2 = K.vmul(K.silu(g), u)
  +dn = mv(14n, wb, a2, Array.new(F32, 12n, 0.0), fd, 4096, 12n, rd)
  K.vadd(x1, dn)

# The linear (gated-delta-net) attention block as one host-served block bang
# (tools/payload-linb.patch): attn_norm, the qkv / z / beta / alpha GEMVs, the
# 32-element sig / softplus / exp middleware, conv1d, the two l2norms,
# gated_delta, gated_rmsnorm and the ssm_out GEMV -- one serve boundary, y (the
# attn contribution a) is field index 3. The CPU/JS body is that sequence as the
# engine's lin / lin.conv / lin.y / lin.gd ran it: K.rms_norm(x, an, 4096), the
# four mv GEMVs (cols 4096, rg = q8rw(4096)), K.sigmoid / K.softplus / K.vmul /
# K.expmap, llb.conv1d, llb.l2norm twice (sc = sqrt(128) / 1), the in-place
# llb.gated_delta (S updated, so the caller keeps its own S) and llb.gated_rmsnorm
# and the ssm_out GEMV.
# The linear layer's 16 q/k heads share 32 v heads (ssm_group_count 16,
# ssm_time_step_rank 32): the shipped lin.y broadcasts each l2normed q/k head
# with K.both(Array.clone(...), x=>x, w=>w) -- a real 16->32 duplication, not
# identity -- and gated_delta (k=5) then reads the 4096-element q/kk.
# the conv output splits ((q(2048), k(2048)), v(4096)): the two l2norms are each
# duplicated, then the in-place gated_delta, gated_rmsnorm and the ssm_out GEMV
# (a parameter scrutinee, so this is its own def; helpers precede their use)
@unsafe
def llb.linb.split(cnv: Array<F32>, +S: Array<F32>, z: Array<F32>, gp: Array<F32>, beta: Array<F32>,
  +wb: Bytes(), y: Array<F32>, +wsn: U32, +wso: U32, +rg: U32) -> Array<F32>:
  match cnv:
    case ANode{ANode{yq, yk}, yv}:
      +qt16 = llb.l2norm(4n, yq, F32.sqrt(128.0))
      +kt16 = llb.l2norm(4n, yk, 1.0)
      +qt = K.both(Array.clone(F32, qt16), x => x, w => w)
      +kt = K.both(Array.clone(F32, kt16), x => x, w => w)
      +gd = llb.gated_delta(5n, S, qt, kt, yv, K.expmap(gp), beta)
      mv(14n, wb, llb.gated_rmsnorm(5n, gd.pair.out(gd), wb, wsn, z), y, wso, 4096, 10n, rg)
    case _:
      y

@unsafe
def llb.lin_block_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +qkva: Array<F32>,
  +an: U32, +wqkv: U32,
  +wgate: U32, +wbo: U32, +wao: U32, +wsa: U32, +wdt: U32,
  +wcv: U32, +wsn: U32, +wso: U32,
  +h0: Array<F32>, +h1: Array<F32>, +h2: Array<F32>, +S: Array<F32>, +rg: U32) -> Array<F32>:
  +h = K.rms_norm(x, load(6n, wb, an, 12n), 4096.0)
  +qkv = mv(14n, wb, h, qkva, wqkv, 8192, 13n, rg)
  +z = mv(14n, wb, h, Array.new(F32, 12n, 0.0), wgate, 4096, 10n, rg)
  +bpre = mv(14n, wb, h, Array.new(F32, 5n, 0.0), wbo, 32, 10n, rg)
  +apre = mv(14n, wb, h, Array.new(F32, 5n, 0.0), wao, 32, 10n, rg)
  +beta = K.sigmoid(bpre)
  +gp = K.vmul(K.softplus(K.vadd(apre, load(6n, wb, wdt, 5n))), load(6n, wb, wsa, 5n))
  llb.linb.split(llb.conv1d(13n, wb, wcv, qkv, h0, h1, h2), S, z, gp, beta, wb, y, wsn, wso, rg)

# The Qwen3.5 attention output gate as a host-served warp bang
# (tools/payload-og.patch): y := x * sigmoid(g) over 2^d, output y at field
# index 3, the body ops.bend's K.output_gate.
@unsafe
def llb.outputgate_warp(+f: Nat, x: Array<F32>, g: Array<F32>, y: Array<F32>, +d: Nat) -> Array<F32>:
  K.output_gate(x, g)

# The Qwen3.5 full-attention qk-norm + partial NEOX rope as host-served warp
# bangs (tools/payload-qkr.patch): y at field index 3, the body ops.bend's
# K.qg.go / K.kr.go exactly. The 256 gains at byte at of wb; hd 256, nd 64,
# base 1e7, pos the slot. q: 2^d = 16 heads of [q(256), gate(256)], qk-norm +
# rope on q, gate copied; k: 2^d = 4 heads of 256, qk-norm + rope on each.
# q returns ANode{q heads, gates} (tools/payload-sqg.patch): ful's split is one
# top-level node, not the per-head split_qg rebuild.
def llb.qg_join(r: Array<F32> & Array<F32>) -> Array<F32>:
  (q, g) = r
  ANode{q, g}

@unsafe
def llb.qknorm_rope_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +at: U32,
  +pos: U32) -> Array<F32>:
  llb.qg_join(K.split_qg.go(4n, K.qg.go(4n, x, load(6n, wb, at, 8n), 256, 64, U32.to_f32(pos), 10000000.0)))

@unsafe
def llb.knorm_rope_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +at: U32,
  +pos: U32) -> Array<F32>:
  K.kr.go(2n, x, load(6n, wb, at, 8n), 256, 64, U32.to_f32(pos), 10000000.0)

# K.put of the 1024-f32 row x into the KV cache y at slot pos, c = dT | pos << 5
# (tools/payload-slot.patch): call it as a bang; the host copies x into y's
# slot in place. This body is the CPU/JS path: K.put as ful.qg ran it. wb only
# keeps y at field 3.
@unsafe
def llb.slot_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +c: U32) -> Array<F32>:
  +dT = U32.to_nat((c .&. 31 : U32))
  K.put(dT, y, (1 << dT : U32), 0, (c >> 5n : U32), x)

# The head (rms_final, the output GEMV and the top two) as a host-served warp
# bang (tools/payload-head.patch): call it as a bang; the device parks the task
# and the host runs llb_rmsnorm (n = 4096, the norm at byte noff of wb), the
# GEMV (the n rows of Q8_0 at byte ooff, row word rw) into a 2^18 scratch and
# llb_amax, delivering the four words [v, i, w, j] to y. This body is the
# CPU/JS path and the tree contract: K.rms_norm, then the q8 GEMV (2^18,
# -inf padded) and K.argmax twice (the second with slot i read as -inf).
@unsafe
def head.pair(+z: Array<F32>, first: F32 & U32) -> F32 & U32:
  (v, +i) = first
  K.argmax(K.put(18n, z, 262144, 0, i, ALeaf{K.neg_inf()}), 262144, 0)

@unsafe
def head.top(+z: Array<F32>, first: F32 & U32, second: F32 & U32) -> Array<F32>:
  (v, +i) = first
  (w, j) = second
  ANode{ANode{ALeaf{v}, ALeaf{U32.to_f32(i)}}, ANode{ALeaf{w}, ALeaf{U32.to_f32(j)}}}

@unsafe
def llb.head_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +noff: U32, +ooff: U32,
  +n: U32, +rw: U32) -> Array<F32>:
  +z = mv(f, wb, K.rms_norm(x, load(6n, wb, noff, 12n), 4096.0),
    Array.new(F32, 18n, K.neg_inf()), ooff, n, 10n, rw)
  head.top(z, K.argmax(z, 262144, 0), head.pair(z, K.argmax(z, 262144, 0)))

# The Qwen3.5 full-attention block: the query scores, the causal softmax and the
# weighted V of all 16 heads as one host-served warp bang
# (tools/payload-attn4.patch): call it as a bang; the device parks the task and
# the host runs llb_attn4 into y. q is 16 x 256, kc / vc 2^dT slots of 4 kv
# heads x 256 (head h reads kv head h >> 2), c = dT | pos << 5, sc =
# 1 / sqrt(256). This body is the CPU/JS path and the tree contract: K.scores4,
# K.probs and K.weighted4 as the engine ran them.
@unsafe
def llb.attn4_warp(+f: Nat, +q: Array<F32>, +kc: Array<F32>, +vc: Array<F32>, y: Array<F32>,
  +c: U32) -> Array<F32>:
  +dT = U32.to_nat((c .&. 31 : U32))
  +pos = (c >> 5n : U32)
  +nT = (1 << dT : U32)
  +sc = (1.0 / F32.sqrt(256.0) : F32)
  +scores = K.scores4(4n, 16, 0, dT, sc, q, kc)
  +ps = K.probs(4n, nT, pos, scores)
  K.weighted4(4n, 16, 0, dT, ps, vc)

# The Qwen3.5 full-attention output tail as one host-served block bang
# (tools/payload-aob.patch): attn4 (scores4 + probs + weighted4), the output
# gate and the o GEMV as one serve boundary, y (the attn contribution) at field
# index 5. The CPU/JS body is exactly the sequence llb.attn4_warp +
# llb.outputgate_warp + the o q8mv as ful.qg ran them. q and the gate come as
# one buffer, ANode{q heads, gates} (tools/payload-aqg.patch): the host reads
# the gate at q + 4096, the body splits it at the top node.
@unsafe
def llb.attn_out_go(+f: Nat, +wb: Bytes(), qg: Array<F32> & Array<F32>, +kc: Array<F32>,
  +vc: Array<F32>, y: Array<F32>, +wo: U32, +c: U32, +rb: U32) -> Array<F32>:
  (q, g) = qg
  +dT = U32.to_nat((c .&. 31 : U32))
  +pos = (c >> 5n : U32)
  +nT = (1 << dT : U32)
  +sc = (1.0 / F32.sqrt(256.0) : F32)
  +scores = K.scores4(4n, 16, 0, dT, sc, q, kc)
  +ps = K.probs(4n, nT, pos, scores)
  +attn = K.weighted4(4n, 16, 0, dT, ps, vc)
  +gated = K.output_gate(attn, g)
  mv(f, wb, gated, y, wo, 4096, 12n, rb)

@unsafe
def llb.attn_out_warp(+f: Nat, +wb: Bytes(), qg: Array<F32>, +kc: Array<F32>,
  +vc: Array<F32>, y: Array<F32>, +wo: U32, +c: U32, +rb: U32) -> Array<F32>:
  llb.attn_out_go(f, wb, K.split_qg.go(0n, qg), kc, vc, y, wo, c, rb)

# The Qwen3.5 full-attention block as one host-served bang
# (tools/payload-fulb.patch): the host runs exactly the kernels of
# llb.rmsnorm_warp, the q / k / v gemv_warps, llb.qknorm_rope_warp,
# llb.knorm_rope_warp, the two llb.slot_warps and llb.attn_out_warp, in that
# order, into y (field index 3); kc / vc (4, 5) come lent and the host writes
# row pos of each IN PLACE. This body is y only: K.put is functional, so on the
# CPU/JS path the caller's caches are not updated -- the engine must not run it
# off the device. c = dT | pos << 5; rb the Q8 row word of the q/k/v/o rows.
@unsafe
def llb.ful_block_warp(+f: Nat, +wb: Bytes(), x: Array<F32>, y: Array<F32>, +kc: Array<F32>,
  +vc: Array<F32>, +an: U32, +wq: U32, +wk: U32, +wv: U32, +wqn: U32, +wkn: U32, +wo: U32,
  +c: U32, +rb: U32) -> Array<F32>:
  +pos = (c >> 5n : U32)
  +h = llb.rmsnorm_warp(f, wb, x, Array.new(F32, 12n, 0.0), an, 12n)
  +qg = llb.qknorm_rope_warp(f, wb, mv(f, wb, h, Array.new(F32, 13n, 0.0), wq, 8192, 10n, rb),
    Array.new(F32, 13n, 0.0), wqn, pos)
  +kn = llb.knorm_rope_warp(f, wb, mv(f, wb, h, Array.new(F32, 10n, 0.0), wk, 1024, 10n, rb),
    Array.new(F32, 10n, 0.0), wkn, pos)
  +vb = mv(f, wb, h, Array.new(F32, 10n, 0.0), wv, 1024, 10n, rb)
  llb.attn_out_warp(f, wb, qg, llb.slot_warp(f, wb, kn, kc, c), llb.slot_warp(f, wb, vb, vc, c),
    y, wo, c, rb)

# Attention's score, softmax and weighted V over the 16 heads, then the o gemv
# (2048 rows, d = 9), as one native pass (tools/payload-attn.patch): call it as
# a bang; the device parks the task and the host runs llb_attn, then llb_warp
# on its output. c = dT | pos << 5; rb: wo's row bytes (F32 8192, Q8_0 2176,
# tools/merge3-q8.patch). This body is the CPU/JS path and the tree
# contract: k.score, k.prob, k.wsum and gemv as lay ran them. Then the
# residual x + o (k.add; llb_add on the device, tools/payload-add.patch).
@unsafe
def llb.attn_warp(+f: Nat, +wb: Bytes(), +q: Array<F32>, +kc: Array<F32>, +vc: Array<F32>, +c: U32,
  y: Array<F32>, +wo: U32, +x: Array<F32>, +rb: U32) -> Array<F32>:
  +dT = U32.to_nat((c .&. 31 : U32))
  +pf = U32.to_f32((c >> 5n : U32))
  # Enough score/probability tasks to leave the runtime's one-block grow pass.
  +af = Nat.min(8n, Nat.add(dT, 4n))
  +sc = fill(~k.score, af, q, kc, c, pf, Array.new(F32, Nat.add(dT, 4n), 0.0), 0, (16 << dT : U32))
  +ps = fill(~k.prob, af, sc, sc, c, pf, Array.new(F32, Nat.add(dT, 4n), 0.0), 0, (16 << dT : U32))
  +att = fill(~k.wsum, 11n, ps, vc, c, pf, Array.new(F32, 11n, 0.0), 0, 2048)
  +o = mv(f, wb, att, y, wo, 2048, 9n, rb)
  fill(~k.add, 6n, x, o, 0, pf, Array.new(F32, 11n, 0.0), 0, 2048)

# rms then gemv as one native pass (tools/payload-rms.patch): call it as a
# bang; the device parks the task and the host runs llb_rms (x, the norm at
# byte an of wb), then llb_warp on its output. This body is the CPU/JS path.
@unsafe
def llb.rms_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32, +an: U32) -> Array<F32>:
  mv(f, wb, rms(wb, an, x), y, base, n, d, rb)

# gemv then the residual add as one native pass (tools/interiors-add.patch):
# call it as a bang; the device parks the task and the host runs llb_warp,
# then llb_add (r + y, 2048). This body is the CPU/JS path: k.add of r and
# the GEMV's output, as lay ran them.
@unsafe
def llb.add_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32, +r: Array<F32>) -> Array<F32>:
  fill(~k.add, 6n, r, mv(f, wb, x, y, base, n, d, rb), 0, 0.0, Array.new(F32, 11n, 0.0), 0, 2048)

# rms, the gemv, then K.rope of its n outputs into y's slots at .. at + n at
# position pos, as one native pass (tools/interiors-rope.patch): call it as a
# bang; the device parks the task and the host runs llb_rms, llb_warp into y
# from slot at, then llb_rope in place. This body is the CPU/JS path: the
# rms-GEMV, then k.rope into y as lay ran it.
@unsafe
def llb.rope_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32, +an: U32, +at: U32, +pos: U32) -> Array<F32>:
  fill(~k.rope, 3n, mv(f, wb, rms(wb, an, x), Array.new(F32, 11n, 0.0), base, n, d, rb), x, at,
    U32.to_f32(pos), y, at, n)

# the same without the rope: the rms-GEMV into y's slots at .. at + n (k.at)
@unsafe
def llb.put_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32, +an: U32, +at: U32) -> Array<F32>:
  fill(~k.at, 3n, mv(f, wb, rms(wb, an, x), Array.new(F32, 11n, 0.0), base, n, d, rb), x, at, 0.0,
    y, at, n)

# The FFN as one native pass (tools/interiors-ffn.patch): x + Wd (silu(Wg h) *
# Wu h), h = rms(x) with the norm at byte an; gate/up rows at bytes g, u (rg
# bytes a row), the down rows at base (rb, d). Call it as a bang; the host
# runs llb_rms, the gate and up GEMVs, llb_silu, the down GEMV and llb_add.
# This body is the CPU/JS path: the rms-GEMVs, k.silu, the GEMV-add as lay ran them.
@unsafe
def llb.ffn_warp(+f: Nat, +wb: Bytes(), +x: Array<F32>, y: Array<F32>, +base: U32, +n: U32, +d: Nat,
  +rb: U32, +g: U32, +u: U32, +an: U32, +rg: U32) -> Array<F32>:
  gs us = mv(f, wb, rms(wb, an, x), Array.new(F32, 13n, 0.0), g, 6144, 9n, rg) mv(f, wb, rms(wb, an, x), Array.new(F32, 13n, 0.0), u, 6144, 9n, rg)
  fill(~k.add, 6n, x, mv(f, wb, fill(~k.silu, 6n, gs, us, 0, 0.0, Array.new(F32, 13n, 0.0), 0, 6144), y,
    base, n, d, rb), 0, 0.0, Array.new(F32, 11n, 0.0), 0, 2048)
