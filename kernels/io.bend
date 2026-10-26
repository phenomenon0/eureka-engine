# IO for the kernel drivers: load a vector file, parse args, print F32 bits.
import Base
import ./ops.bend as K
import ./order.bend as O

def io.bytes(pair: File & Result<&1, &1, U32 & String, List<&2, U32>>) ->
  IO(List<&2, U32>):
  (f, r) = pair
  do IO<List<&2, U32>>:
    File.close(f)
    IO.pass(List<&2, U32>, r)

def io.load_bytes(path: String, off: U32, n: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    p : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, off, n)
    io.bytes(p)

# the file's f32le values from byte off as a vector of 2^d slots; slots past
# the file's end hold pad (kernel fixtures may intentionally be unpadded)
def io.load_at(path: String, off: U32, +d: Nat, pad: F32) -> IO(Array<F32>):
  do IO<Array<F32>>:
    bs : List<&2, U32> <- io.load_bytes(path, off, (4 << d : U32))
    return K.vec(d, pad, bs)

# Weight reads must never silently turn a truncated file into zero weights.
def io.expect.if(ok: Bool, bs: List<&2, U32>) -> IO(List<&2, U32>):
  match ok:
    case True{}:
      IO.pure(List<&2, U32>, bs)
    case False{}:
      IO.die(List<&2, U32>, 1, "short weight read (run tools/split_pack.py --check)")

# Tail recursion also supports multi-row reads on the JS lane.
def io.byte_count(bs: List<&2, U32>, n: U32) -> U32:
  match bs:
    case Nil{}:
      n
    case Con{b, t}:
      io.byte_count(t, (n + 1 : U32))

def io.expect(n: U32, +bs: List<&2, U32>) -> IO(List<&2, U32>):
  io.expect.if(U32.is_eq(n, io.byte_count(bs, 0)), bs)

def io.load_exact(path: String, off: U32, +d: Nat) -> IO(Array<F32>):
  do IO<Array<F32>>:
    bs : List<&2, U32> <- io.load_bytes(path, off, (4 << d : U32))
    exact : List<&2, U32> <- io.expect((4 << d : U32), bs)
    return K.vec(d, 0.0, exact)

def io.load(path: String, +d: Nat, pad: F32) -> IO(Array<F32>):
  io.load_at(path, 0, d, pad)

def io.u32.got(s: String, m: Maybe<&2, U32>) -> IO(U32):
  match m:
    case Some{v}:
      IO.pure(U32, v)
    case None{}:
      IO.die(U32, 2, "not a U32: " ++ s)

def io.u32(s: String) -> IO(U32):
  +t = s
  io.u32.got(t, U32.read(t))

# one line per value: the decimal U32 bit pattern
def io.emit(n: Nat, xs: List<F32>) -> IO(Unit):
  match n xs:
    case 1n+k Con{h, t}:
      do IO<Unit>:
        IO.print(U32.show(F32.bits(h)))
        io.emit(k, t)
    case _ _:
      IO.pure(Unit, Unit{})

def io.emit_vec(n: Nat, a: Array<F32>) -> IO(Unit):
  io.emit(n, Array.to_list(~F32, a))

# Time a thunk, not an already-evaluated argument. IO.now is millisecond wall
# time, so small ops can honestly measure 0 ms. IO/load costs stay in their tag.
def io.time(-A: Type, tag: String, f: Unit -> A) -> IO(A):
  do IO<A>:
    t0 : Nat <- IO.now()
    a : A = f(Unit{})
    t1 : Nat <- IO.now()
    IO.print_err("op " ++ tag ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return a

def io.time_io(-A: Type, tag: String, action: IO(A)) -> IO(A):
  do IO<A>:
    t0 : Nat <- IO.now()
    a : A <- action
    t1 : Nat <- IO.now()
    IO.print_err("op " ++ tag ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    return a

# gemv over a file
# ---------------

def io.rows.got(rb: U32, k: File -> List<&2, U32> -> IO(File & Array<F32> & List<Array<F32>>),
  got: File & Result<&1, &1, U32 & String, List<&2, U32>>) ->
  IO(File & Array<F32> & List<Array<F32>>):
  (h, r) = got
  do IO<File & Array<F32> & List<Array<F32>>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    exact : List<&2, U32> <- io.expect(rb, bs)
    k(h, exact)

def io.rows.dot(k: F32 -> Array<F32> -> IO(File & Array<F32> & List<Array<F32>>),
  yx: F32 & Array<F32>) -> IO(File & Array<F32> & List<Array<F32>>):
  (y, x) = yx
  k(y, x)

def io.chunk.next(k: List<&2, U32> -> Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>),
  acc: List<Array<F32>>, got: F32 & List<&2, U32> & Array<F32>) -> IO(File & Array<F32> & List<Array<F32>>):
  (y, (bs, x)) = got
  k(bs, x, ALeaf{y} <> acc)

# Multi-row chunks have d >= 2 and rb divisible by 16; other shapes use one row.
def io.chunk(n: Nat, +d: Nat, +rb: U32, bs: List<&2, U32>, x: Array<F32>, acc: List<Array<F32>>,
  k: Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>)) -> IO(File & Array<F32> & List<Array<F32>>):
  match n:
    case 0n:
      k(x, acc)
    case 1n+0n:
      io.rows.dot(y => x2 => k(x2, ALeaf{y} <> acc), K.dot_bytes(d, bs, x))
    case 1n+p:
      io.chunk.next(t => x2 => ys => io.chunk(p, d, rb, t, x2, ys, k), acc,
        K.dot_bytes.go(Nat.sub(d, 2n), 0, (rb / 4 : U32), bs, x))

# order v2: every row by K.bdot_list, whatever the chunk size
def io.chunk.v2(n: Nat, +rb: U32, bs: List<&2, U32>, x: Array<F32>, acc: List<Array<F32>>,
  k: Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>)) -> IO(File & Array<F32> & List<Array<F32>>):
  match n:
    case 0n:
      k(x, acc)
    case 1n+p:
      io.chunk.next(t => x2 => ys => io.chunk.v2(p, rb, t, x2, ys, k), acc,
        K.bdot_list((rb / 4 : U32), bs, x))

def io.chunk.pick(v2: Bool, n: Nat, +d: Nat, +rb: U32, bs: List<&2, U32>, x: Array<F32>, acc: List<Array<F32>>,
  k: Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>)) -> IO(File & Array<F32> & List<Array<F32>>):
  match v2:
    case True{}:
      io.chunk.v2(n, rb, bs, x, acc, k)
    case False{}:
      io.chunk(n, d, rb, bs, x, acc, k)

# n rows of W, rb bytes each, still reduced independently in the fixed tree.
# Batch at most 16 aligned rows to amortize pread/IO scheduling; keep exact reads.
def io.rows.go(fuel: Nat, n: Nat, +d: Nat, +rb: U32, +off: U32, f: File, x: Array<F32>,
  acc: List<Array<F32>>) -> IO(File & Array<F32> & List<Array<F32>>):
  match fuel n:
    case 1n+left 1n+p0:
      +p = p0
      +rows = Nat.min(1n+p, Bool.pick(Nat,
        Bool.and(Nat.is_le(2n, d), Bool.and(U32.is_eq((rb .&. 15 : U32), 0),
          Bool.and((rb <= (4 << d : U32) : U32), (rb <= 268435455 : U32)))), 16n, 1n))
      +bytes = (U32.from_nat(rows) * rb : U32)
      IO.bind(File & Result<&1, &1, U32 & String, List<&2, U32>>,
        File & Array<F32> & List<Array<F32>>,
        File.read_at(f, off, bytes), io.rows.got(bytes, h => bs =>
          io.chunk.pick(O.order.v2(), rows, d, rb, bs, x, acc, x2 => ys =>
            io.rows.go(left, Nat.sub(1n+p, rows), d, rb, (off + bytes : U32), h, x2, ys))))
    case _ _:
      IO.pure(File & Array<F32> & List<Array<F32>>, (f, (x, acc)))

def io.rows(+n: Nat, d: Nat, rb: U32, off: U32, f: File, x: Array<F32>,
  acc: List<Array<F32>>) -> IO(File & Array<F32> & List<Array<F32>>):
  io.rows.go(n, n, d, rb, off, f, x, acc)

def io.gemv.fin(tag: String, +t0: Nat, t1: Nat, r: File & Array<F32> & List<Array<F32>>) ->
  IO(Array<F32> & List<Array<F32>>):
  (f, xa) = r
  do IO<Array<F32> & List<Array<F32>>>:
    File.close(f)
    IO.print_err("op " ++ tag ++ " " ++ Nat.show(Nat.sub(t1, t0)))
    IO.pure(Array<F32> & List<Array<F32>>, xa)

# y = W x for W [n, rb/4] f32le at byte off of path: (x, y's leaves highest row
# first); the wall ms goes to stderr as `op <tag> <ms>`
def io.gemv.list(tag: String, path: String, off: U32, n: Nat, +d: Nat, rb: U32, x: Array<F32>) ->
  IO(Array<F32> & List<Array<F32>>):
  do IO<Array<F32> & List<Array<F32>>>:
    t0 : Nat <- IO.now()
    f : File <- IO.try(File, File.open(path, "r"))
    r : File & Array<F32> & List<Array<F32>> <- io.rows(n, d, rb, off, f, x, Nil{})
    t1 : Nat <- IO.now()
    io.gemv.fin(tag, t0, t1, r)

def io.buf.expect.if(ok: Bool, bs: Bytes(), f: File) -> IO(File & Bytes()):
  match ok:
    case True{}:
      IO.pure(File & Bytes(), (f, bs))
    case False{}:
      IO.die(File & Bytes(), 1, "short weight read (run tools/split_pack.py --check)")

def io.buf.part.next(k: File -> Bytes() -> IO(File & Bytes()), got: File & Bytes()) -> IO(File & Bytes()):
  (f, bs) = got
  k(f, bs)

def io.buf.part(+n: Nat, +bs: Bytes(), f: File, k: Nat -> File -> Bytes() -> IO(File & Bytes())) -> IO(File & Bytes()):
  +got = Bytes.len(bs)
  IO.bind(File & Bytes(), File & Bytes(),
    io.buf.expect.if(Bool.and(Nat.is_lt(0n, got), Nat.is_le(got, n)), bs, f),
    io.buf.part.next(k(Nat.sub(n, got))))

def io.buf.got(n: Nat, k: Nat -> File -> Bytes() -> IO(File & Bytes()),
  got: File & Result<&1, &1, U32 & String, Bytes()>) -> IO(File & Bytes()):
  (f, r) = got
  do IO<File & Bytes()>:
    bs : Bytes() <- IO.pass(Bytes(), r)
    io.buf.part(n, bs, f, k)

# read_buf may return a positive partial transfer (including RWF_NOWAIT).
# Accumulate until complete; only zero bytes before completion mean EOF.
# At least one byte arrives per iteration, so byte-count fuel is sufficient.
def io.buf.read.go(fuel: Nat, n: Nat, f: File, acc: Bytes()) -> IO(File & Bytes()):
  match fuel n:
    case _ 0n:
      IO.pure(File & Bytes(), (f, acc))
    case 1n+p 1n+x:
      +want = Nat.add(1n, x)
      IO.bind(File & Result<&1, &1, U32 & String, Bytes()>, File & Bytes(),
        File.read_buf(f, U32.from_nat(want)),
        io.buf.got(want, left => h => bs => io.buf.read.go(p, left, h, Bytes.append(acc, bs))))
    case _ _:
      IO.die(File & Bytes(), 1, "short weight read (run tools/split_pack.py --check)")

def io.buf.read(f: File, +n: U32) -> IO(File & Bytes()):
  io.buf.read.go(U32.to_nat(n), U32.to_nat(n), f, SNil{})

def io.buf.skip.got(k: File -> IO(File), got: File & Bytes()) -> IO(File):
  (f, bs) = got
  k(f)

# No seek/read_buf_at exists in Base. Discard bounded prefixes in 1 MiB chunks.
def io.buf.skip(fuel: Nat, left: Nat, f: File) -> IO(File):
  match fuel left:
    case 1n+p 1n+x:
      +remain = Nat.add(1n, x)
      +n = Nat.min(remain, 1048576n)
      IO.bind(File & Bytes(), File, io.buf.read(f, U32.from_nat(n)),
        io.buf.skip.got(h => io.buf.skip(p, Nat.sub(remain, n), h)))
    case _ _:
      IO.pure(File, f)

def io.buf.chunk.next(k: Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>),
  acc: List<Array<F32>>, got: F32 & Array<F32>) -> IO(File & Array<F32> & List<Array<F32>>):
  (y, x) = got
  k(x, ALeaf{y} <> acc)

def io.buf.dot(v2: Bool, +d: Nat, +cols: U32, +at: U32, +bs: Bytes(), x: Array<F32>) -> F32 & Array<F32>:
  match v2:
    case True{}:
      K.bdot_buf(cols, at, bs, x)
    case False{}:
      K.dot_buf(Nat.sub(d, 2n), 0, cols, at, bs, x)

def io.buf.chunk(n: Nat, +d: Nat, +rb: U32, +at: U32, +bs: Bytes(), x: Array<F32>, acc: List<Array<F32>>,
  k: Array<F32> -> List<Array<F32>> -> IO(File & Array<F32> & List<Array<F32>>)) -> IO(File & Array<F32> & List<Array<F32>>):
  match n:
    case 0n:
      k(x, acc)
    case 1n+p:
      io.buf.chunk.next(x2 => ys => io.buf.chunk(p, d, rb, (at + rb : U32), bs, x2, ys, k), acc,
        io.buf.dot(O.order.v2(), d, (rb / 4 : U32), at, bs, x))

def io.buf.rows.got(k: File -> Bytes() -> IO(File & Array<F32> & List<Array<F32>>),
  got: File & Bytes()) -> IO(File & Array<F32> & List<Array<F32>>):
  (f, bs) = got
  k(f, bs)

def io.buf.rows(fuel: Nat, n: Nat, +d: Nat, +rb: U32, f: File, x: Array<F32>, acc: List<Array<F32>>) -> IO(File & Array<F32> & List<Array<F32>>):
  match fuel n:
    case 1n+left 1n+p0:
      +p = p0
      +rows = Nat.min(1n+p, 16n)
      IO.bind(File & Bytes(), File & Array<F32> & List<Array<F32>>, io.buf.read(f, (U32.from_nat(rows) * rb : U32)),
        io.buf.rows.got(h => bs => io.buf.chunk(rows, d, rb, 0, bs, x, acc, x2 => ys =>
          io.buf.rows(left, Nat.sub(1n+p, rows), d, rb, h, x2, ys))))
    case _ _:
      IO.pure(File & Array<F32> & List<Array<F32>>, (f, (x, acc)))

def io.gemv.buf(tag: String, path: String, +off: U32, +n: Nat, d: Nat, rb: U32, x: Array<F32>) -> IO(Array<F32> & List<Array<F32>>):
  do IO<Array<F32> & List<Array<F32>>>:
    t0 : Nat <- IO.now()
    f : File <- IO.try(File, File.open(path, "r"))
    h : File <- io.buf.skip(U32.to_nat(off), U32.to_nat(off), f)
    r : File & Array<F32> & List<Array<F32>> <- io.buf.rows(n, n, d, rb, h, x, Nil{})
    t1 : Nat <- IO.now()
    io.gemv.fin(tag, t0, t1, r)

def io.gemv.pick(packed: Bool, tag: String, path: String, off: U32, n: Nat, d: Nat, rb: U32, x: Array<F32>) -> IO(Array<F32> & List<Array<F32>>):
  match packed:
    case True{}:
      io.gemv.buf(tag, path, off, n, d, rb, x)
    case False{}:
      io.gemv.list(tag, path, off, n, d, rb, x)

# ponytail: bounded prefix replay until Base has read_buf_at. Split layers fit
# below 256 MiB; large pack offsets retain pread instead of rereading gigabytes.
# Packed chunks stay <= 1 MiB. Small/unaligned/empty shapes retain the old path.
def io.gemv.at(tag: String, path: String, +off: U32, +n: Nat, +d: Nat, +rb: U32, x: Array<F32>) -> IO(Array<F32> & List<Array<F32>>):
  io.gemv.pick(Bool.and((off <= 268435456 : U32), Bool.and(Nat.is_le(1n, n),
    Bool.and(Nat.is_le(2n, d), Bool.and(U32.is_eq((rb .&. 15 : U32), 0),
      Bool.and(Bool.and((rb > 0 : U32), (rb <= 65536 : U32)), (rb <= (4 << d : U32) : U32)))))),
    tag, path, off, n, d, rb, x)

# v2 lanes need whole 16-column blocks inside x; anything else fails loudly
def io.gemv.order(ok: Bool, tag: String, path: String, +off: U32, +n: Nat, +d: Nat, +rb: U32, x: Array<F32>) -> IO(Array<F32> & List<Array<F32>>):
  match ok:
    case True{}:
      io.gemv.at(tag, path, off, n, d, rb, x)
    case False{}:
      IO.die(Array<F32> & List<Array<F32>>, 1, "order v2: gemv " ++ tag ++ " needs row bytes a positive multiple of 64 within x")

def io.gemv(tag: String, path: String, +off: U32, +n: Nat, +d: Nat, +rb: U32, x: Array<F32>) -> IO(Array<F32> & List<Array<F32>>):
  io.gemv.order(Bool.or(Bool.not(O.order.v2()), Bool.and(U32.is_eq((rb .&. 63 : U32), 0),
    Bool.and((rb > 0 : U32), (rb <= (4 << d : U32) : U32)))), tag, path, off, n, d, rb, x)

def io.emit_all(xs: List<F32>) -> IO(Unit):
  match xs:
    case Con{h, t}:
      do IO<Unit>:
        IO.print(U32.show(F32.bits(h)))
        io.emit_all(t)
    case Nil{}:
      IO.pure(Unit, Unit{})

# leaves listed highest index first, printed lowest first as io.emit does
def io.emit_rev(xs: List<Array<F32>>, acc: List<F32>) -> IO(Unit):
  match xs:
    case Con{ALeaf{v}, t}:
      io.emit_rev(t, v <> acc)
    case _:
      io.emit_all(acc)
