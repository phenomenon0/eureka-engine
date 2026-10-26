# bench_q8q <blk.bin> <f> <reps>: pure-Bend q8.gemv timing on blk.0.attn_qkv
# (8192x4096 Q8_0) over resident layer bytes. stderr `gemv <ms> <bits0>` per rep.
import Base
import ./flat.bend as F
import ./io.bend as I

@unsafe
def runs(k: Nat, +wb: Bytes(), +x: Array<F32>, +f: Nat) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        y : Array<F32> = F.q8.gemv(f, wb, x, Array.new(F32, 13n, 0.0), 17842176, 8192, 13n, 4352)
        t1 : Nat <- IO.now()
        IO.print_err("gemv " ++ Nat.show(Nat.sub(t1, t0)) ++ " ms " ++ U32.show(F32.bits(Array.peek_f32(y, 0))))
        runs(p, wb, x, f)

@unsafe
def run_loaded(wp: String, got: File & Bytes(), +f: Nat, +k: Nat) -> IO(Unit):
  (h, +wb) = got
  do IO<Unit>:
    File.close(h)
    runs(k, wb, F.load(6n, wb, 0, 12n), f)

@unsafe
def pick(fh: File, args: List<String>) -> IO(Unit):
  match args:
    case Con{fs, Con{ks, Nil{}}}:
      do IO<Unit>:
        fv : U32 <- I.io.u32(fs)
        kv : U32 <- I.io.u32(ks)
        got : File & Bytes() <- I.io.buf.read(fh, 232178432)
        run_loaded("", got, U32.to_nat(fv), U32.to_nat(kv))
    case _:
      IO.die(Unit, 2, "usage: bench_q8q <blk.bin> <f> <reps>")

@unsafe
def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, rest}:
      do IO<Unit>:
        fh : File <- IO.try(File, File.open(wp, "r"))
        pick(fh, rest)
    case Nil{}:
      IO.die(Unit, 2, "usage: bench_q8q <blk.bin> <f> <reps>")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
