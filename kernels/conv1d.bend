# conv1d <x.bin> <h0.bin> <h1.bin> <h2.bin> <w.bin> <d>: causal depthwise conv1d (kernel 4) + SiLU over 2^d channels.
import Base
import ./ops.bend as K
import ./io.bend as I

def run(xp: String, p0: String, p1: String, p2: String, wp: String, k: U32) -> IO(Unit):
  do IO<Unit>:
    +d : Nat = U32.to_nat(k)
    x : Array<F32> <- I.io.load(xp, d, 0.0)
    h0 : Array<F32> <- I.io.load(p0, d, 0.0)
    h1 : Array<F32> <- I.io.load(p1, d, 0.0)
    h2 : Array<F32> <- I.io.load(p2, d, 0.0)
    w : Array<F32> <- I.io.load(wp, Nat.add(d, 2n), 0.0)
    I.io.emit_vec(U32.to_nat((1 << d : U32)), K.conv1d(d, w, x, h0, h1, h2))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{xp, Con{p0, Con{p1, Con{p2, Con{wp, Con{ds, Nil{}}}}}}}:
      do IO<Unit>:
        k : U32 <- I.io.u32(ds)
        run(xp, p0, p1, p2, wp, k)
    case _:
      IO.die(Unit, 2, "usage: conv1d <x.bin> <h0.bin> <h1.bin> <h2.bin> <w.bin> <d>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
