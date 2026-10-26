# rope <x.bin> <d> <pos>: 2^d values = heads x 128, rotated at position pos
import Base
import ./ops.bend as K
import ./io.bend as I

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{xp, Con{ds, Con{ps, Nil{}}}}:
      do IO<Unit>:
        d : U32 <- I.io.u32(ds)
        pos : U32 <- I.io.u32(ps)
        +dn : Nat = U32.to_nat(d)
        x : Array<F32> <- I.io.load(xp, dn, 0.0)
        I.io.emit_vec(U32.to_nat((1 << dn : U32)), K.rope(x, (1 << dn : U32), 0, U32.to_f32(pos)))
    case _:
      IO.die(Unit, 2, "usage: rope <x.bin> <d> <pos>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
