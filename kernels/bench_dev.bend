# bench_dev <blk.bin> <mode> <reps>: device timings of a layer's pieces, each
# its own bang over resident blk.0 bytes; stderr `<mode> <ms>` per rep. mode
# cmp prints the slots where the old and flat gate gemvs differ (want 0).
import Base
import ./ops.bend as K
import ./dev.bend as D
import ./flat.bend as F
import ./io.bend as I

def vec.go(k: Nat, +wb: Bytes(), +at: U32, acc: List<Array<F32>>) -> List<Array<F32>>:
  match k:
    case 0n:
      acc
    case 1n+p:
      vec.go(p, wb, (at + 4 : U32), ALeaf{D.rd(wb, at)} <> acc)

def vec(+wb: Bytes(), +at: U32) -> Array<F32>:
  K.from_rev(11n, 0n, 0.0, vec.go(2048n, wb, at, Nil{}))

def xb(x: Array<F32>) -> Bytes():
  D.xbytes(Array.to_list(~F32, x), SNil{})

def first.v(r: Array<F32> & F32) -> U32:
  (b, v) = r
  F32.bits(v)

def first(a: Array<F32>) -> U32:
  first.v(Array.get(F32, a, 5))

def old(+wb: Bytes(), x: Array<F32>) -> Array<F32>:
  K.from_rev(13n, 2048n, 0.0, D.flat(D.gemv.tree(14n, wb, xb(x), 88088576, 6144, 9n, 8192), Nil{}))

@unsafe
def new(+wb: Bytes(), x: Array<F32>) -> Array<F32>:
  F.gemv(14n, wb, x, Array.new(F32, 13n, 0.0), 88088576, 6144, 9n, 8192)

def big(+wb: Bytes()) -> Array<F32>:
  K.from_rev(13n, 2048n, 0.0, vec.go(6144n, wb, 0, Nil{}))

# one piece: its tag and the bang

def piece2.a(a: Bool, b: Bool, c: Bool, d: Bool, +wb: Bytes(), x: Array<F32>) -> U32:
  match a b c d:
    case True{} _ _ _:
      first(K.rms_norm(x, vec(wb, 8192), 2048.0))
    case _ True{} _ _:
      first(big(wb))
    case _ _ True{} _:
      first(K.silu(big(wb)))
    case _ _ _ True{}:
      first(Array.new(F32, 17n, 0.0))
    case _ _ _ _:
      first(K.vadd(x, vec(wb, 8192)))

def piece2(+m: String, +wb: Bytes(), x: Array<F32>) -> U32:
  piece2.a(String.eq(m, "rms"), String.eq(m, "big"), String.eq(m, "silu"), String.eq(m, "new8k"), wb, x)

def piece.a(a: Bool, b: Bool, c: Bool, d: Bool, +m: String, +wb: Bytes(), x: Array<F32>) -> U32:
  match a b c d:
    case True{} _ _ _:
      first(old(wb, x))
    case _ True{} _ _:
      first(new(wb, x))
    case _ _ True{} _:
      Bytes.get(xb(x), 0n)
    case _ _ _ True{}:
      first(vec(wb, 0))
    case _ _ _ _:
      piece2(m, wb, x)

def piece(+m: String, +wb: Bytes(), x: Array<F32>) -> U32:
  piece.a(String.eq(m, "old"), String.eq(m, "new"), String.eq(m, "xb"), String.eq(m, "vec"), m, wb, x)

def done(+m: String, +t0: Nat, v: U32) -> IO(Unit):
  do IO<Unit>:
    t1 : Nat <- IO.now()
    IO.print_err(m ++ " " ++ Nat.show(Nat.sub(t1, t0)) ++ " " ++ U32.show(v))

def reps(k: Nat, +m: String, +wb: Bytes()) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        done(m, t0, piece!(m, wb, vec(wb, 0)))
        reps(p, m, wb)

def diff(a: List<F32>, b: List<F32>, +i: U32, n: U32) -> U32:
  match a b:
    case Con{x, s} Con{y, t}:
      diff(s, t, (i + 1 : U32), (n + Bool.to_u32(U32.is_ne(F32.bits(x), F32.bits(y))) : U32))
    case _ _:
      n

def loaded(k: Nat, +m: String, got: File & Bytes()) -> IO(Unit):
  (h, +wb) = got
  do IO<Unit>:
    File.close(h)
    reps(k, m, wb)
    IO.print("mismatch " ++ U32.show(diff(Array.to_list(~F32, old!(wb, vec(wb, 0))),
          Array.to_list(~F32, new!(wb, vec(wb, 0))), 0, 0)))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{m, Con{ks, Nil{}}}}:
      do IO<Unit>:
        f : File <- IO.try(File, File.open(wp, "r"))
        got : File & Bytes() <- I.io.buf.read(f, 188760064)
        k : U32 <- I.io.u32(ks)
        loaded(U32.to_nat(k), m, got)
    case _:
      IO.die(Unit, 2, "usage: bench_dev <blk.bin> <mode> <reps>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
