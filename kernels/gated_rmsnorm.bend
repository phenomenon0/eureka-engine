# gated_rmsnorm <o.bin> <w.bin> <z.bin> <k>: per-head gated RMSNorm over 2^k heads of 128.
import Base
import ./ops.bend as K
import ./io.bend as I

def run(op: String, wp: String, zp: String, k: U32) -> IO(Unit):
  do IO<Unit>:
    +kn : Nat = U32.to_nat(k)
    o : Array<F32> <- I.io.load(op, Nat.add(kn, 7n), 0.0)
    w : Array<F32> <- I.io.load(wp, 7n, 0.0)
    z : Array<F32> <- I.io.load(zp, Nat.add(kn, 7n), 0.0)
    I.io.emit_vec(U32.to_nat((1 << Nat.add(kn, 7n) : U32)), K.gated_rmsnorm(kn, o, w, z, 128.0))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{op, Con{wp, Con{zp, Con{ks, Nil{}}}}}:
      do IO<Unit>:
        k : U32 <- I.io.u32(ks)
        run(op, wp, zp, k)
    case _:
      IO.die(Unit, 2, "usage: gated_rmsnorm <o.bin> <w.bin> <z.bin> <k>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
