# argmax <z.bin> <d>: numpy-rule argmax over the file's values (2^d slots, -inf padded)
import Base
import ./ops.bend as K
import ./io.bend as I

def show(r: F32 & U32) -> String:
  (v, i) = r
  U32.show(i)

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{zp, Con{ds, Nil{}}}:
      do IO<Unit>:
        d : U32 <- I.io.u32(ds)
        +dn : Nat = U32.to_nat(d)
        z : Array<F32> <- I.io.load(zp, dn, K.neg_inf())
        IO.print(show(K.argmax(z, (1 << dn : U32), 0)))
    case _:
      IO.die(Unit, 2, "usage: argmax <z.bin> <d>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
