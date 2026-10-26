# bench_flat <blk.bin> <mode> <reps>: device timings of the flat layer's
# pieces over resident blk.0 bytes, one bang per rep; stderr `<mode> <ms> <bits>`.
import Base
import ./ops.bend as K
import ./flat.bend as F
import ./io.bend as I

@unsafe
def at(+a: Array<F32>, +i: U32) -> U32:
  F32.bits(F.at(a, i))

@unsafe
def attn(+x: Array<F32>) -> Array<F32>:
  +kc = F.fill(~F.k.at, 6n, x, x, 0, 0.0, Array.new(F32, 12n, 0.0), 0, 4096)
  +c = (4 .|. (9 << 5n) : U32)
  +sc = F.fill(~F.k.score, 8n, x, kc, c, 0.0, Array.new(F32, 8n, 0.0), 0, 256)
  +ps = F.fill(~F.k.prob, 8n, sc, sc, c, 0.0, Array.new(F32, 8n, 0.0), 0, 256)
  F.fill(~F.k.wsum, 11n, ps, kc, c, 0.0, Array.new(F32, 11n, 0.0), 0, 2048)

@unsafe
def piece2(a: Bool, b: Bool, c: Bool, d: Bool, +wb: Bytes(), +x: Array<F32>) -> U32:
  match a b c d:
    case True{} _ _ _:
      at(F.rms(wb, 8192, x), 5)
    case _ True{} _ _:
      at(F.fill(~F.k.rope, 6n, x, x, 0, 7.0, Array.new(F32, 11n, 0.0), 0, 2048), 5)
    case _ _ True{} _:
      at(attn(x), 5)
    case _ _ _ True{}:
      at(F.fill(~F.k.add, 6n, x, x, 0, 0.0, Array.new(F32, 11n, 0.0), 0, 2048), 5)
    case _ _ _ _:
      0

@unsafe
def piece(a: Bool, b: Bool, c: Bool, +m: String, +wb: Bytes(), +x: Array<F32>) -> U32:
  match a b c:
    case True{} _ _:
      at(F.gemv(14n, wb, x, Array.new(F32, 11n, 0.0), 18882560, 2048, 9n, 8192), 5)
    case _ True{} _:
      at(F.gemv(14n, wb, x, Array.new(F32, 13n, 0.0), 88088576, 6144, 9n, 8192), 5)
    case _ _ True{}:
      at(F.gemv(14n, wb, F.load(6n, wb, 0, 13n), Array.new(F32, 11n, 0.0), 37756928, 2048, 11n, 24576), 5)
    case _ _ _:
      piece2(String.eq(m, "rms"), String.eq(m, "rope"), String.eq(m, "attn"), String.eq(m, "add"), wb, x)

@unsafe
def bang(+m: String, +wb: Bytes(), +x: Array<F32>) -> U32:
  piece(String.eq(m, "q"), String.eq(m, "gate"), String.eq(m, "down"), m, wb, x)

def done(+m: String, +t0: Nat, v: U32) -> IO(Unit):
  do IO<Unit>:
    t1 : Nat <- IO.now()
    IO.print_err(m ++ " " ++ Nat.show(Nat.sub(t1, t0)) ++ " " ++ U32.show(v))

@unsafe
def reps(k: Nat, +m: String, +wb: Bytes(), +x: Array<F32>) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        done(m, t0, bang!(m, wb, x))
        reps(p, m, wb, x)

@unsafe
def loaded(k: Nat, +m: String, got: File & Bytes()) -> IO(Unit):
  (h, +wb) = got
  do IO<Unit>:
    File.close(h)
    reps(k, m, wb, F.load(6n, wb, 0, 11n))

@unsafe
def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{m, Con{ks, Nil{}}}}:
      do IO<Unit>:
        f : File <- IO.try(File, File.open(wp, "r"))
        got : File & Bytes() <- I.io.buf.read(f, 188760064)
        k : U32 <- I.io.u32(ks)
        loaded(U32.to_nat(k), m, got)
    case _:
      IO.die(Unit, 2, "usage: bench_flat <blk.bin> <mode> <reps>")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
