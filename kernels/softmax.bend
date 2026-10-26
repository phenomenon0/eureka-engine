# softmax <s.bin> <d> <pos>: causal softmax over 2^d score slots, slots t > pos masked
import Base
import ./ops.bend as K
import ./io.bend as I

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{sp, Con{ds, Con{ps, Nil{}}}}:
      do IO<Unit>:
        d : U32 <- I.io.u32(ds)
        pos : U32 <- I.io.u32(ps)
        +dn : Nat = U32.to_nat(d)
        s : Array<F32> <- I.io.load(sp, dn, 0.0)
        I.io.emit_vec(U32.to_nat((1 << dn : U32)), K.softmax(s, (1 << dn : U32), pos))
    case _:
      IO.die(Unit, 2, "usage: softmax <s.bin> <d> <pos>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
