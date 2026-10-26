# qknorm <v.bin> <g.bin> [dv] [dg]: per-head RMSNorm on v with gain g.
# dv: log2 of total elements in v (default 11 for 2048, 8 for 256).
# dg: log2 of head dimension d (default 7 for 128).
import Base
import ./ops.bend as K
import ./io.bend as I

def run(vp: String, gp: String, dv: U32, dg: U32) -> IO(Unit):
  do IO<Unit>:
    +dvn : Nat = U32.to_nat(dv)
    +dgn : Nat = U32.to_nat(dg)
    v : Array<F32> <- I.io.load(vp, dvn, 0.0)
    g : Array<F32> <- I.io.load(gp, dgn, 0.0)
    +k : Nat = Nat.sub(dvn, dgn)
    I.io.emit_vec(U32.to_nat((1 << dvn : U32)), K.qknorm(k, v, g, U32.to_f32((1 << dgn : U32))))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{vp, Con{gp, Nil{}}}:
      run(vp, gp, 11, 7)
    case Con{vp, Con{gp, Con{dvs, Nil{}}}}:
      do IO<Unit>:
        dv : U32 <- I.io.u32(dvs)
        run(vp, gp, dv, 7)
    case Con{vp, Con{gp, Con{dvs, Con{dgs, Nil{}}}}}:
      do IO<Unit>:
        dv : U32 <- I.io.u32(dvs)
        dg : U32 <- I.io.u32(dgs)
        run(vp, gp, dv, dg)
    case _:
      IO.die(Unit, 2, "usage: qknorm <v.bin> <g.bin> [dv] [dg]")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
