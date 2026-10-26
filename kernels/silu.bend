# silu <x.bin> <d> <n>: silu over the file's n values (2^d slots, zero-padded)
import Base
import ./ops.bend as K
import ./io.bend as I

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{xp, Con{ds, Con{ns, Nil{}}}}:
      do IO<Unit>:
        d : U32 <- I.io.u32(ds)
        n : U32 <- I.io.u32(ns)
        x : Array<F32> <- I.io.load(xp, U32.to_nat(d), 0.0)
        I.io.emit_vec(U32.to_nat(n), K.silu(x))
    case _:
      IO.die(Unit, 2, "usage: silu <x.bin> <d> <n>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
