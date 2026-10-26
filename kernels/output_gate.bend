# output_gate <x.bin> <g.bin> [dx]: attention output gate x * sigmoid(g).
# dx: log2 of total elements in x (default 11 for 2048, 8 for 256).
import Base
import ./ops.bend as K
import ./io.bend as I

def run(xp: String, gp: String, dx: U32) -> IO(Unit):
  do IO<Unit>:
    +dxn : Nat = U32.to_nat(dx)
    x : Array<F32> <- I.io.load(xp, dxn, 0.0)
    g : Array<F32> <- I.io.load(gp, dxn, 0.0)
    I.io.emit_vec(U32.to_nat((1 << dxn : U32)), K.output_gate(x, g))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{xp, Con{gp, Nil{}}}:
      run(xp, gp, 11)
    case Con{xp, Con{gp, Con{dxs, Nil{}}}}:
      do IO<Unit>:
        dx : U32 <- I.io.u32(dxs)
        run(xp, gp, dx)
    case _:
      IO.die(Unit, 2, "usage: output_gate <x.bin> <g.bin> [dx]")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
