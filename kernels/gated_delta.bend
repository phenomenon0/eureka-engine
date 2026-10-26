# gated_delta <S.bin> <q.bin> <k.bin> <v.bin> <dec.bin> <bet.bin> <k>: the gated delta recurrence.
import Base
import ./ops.bend as K
import ./io.bend as I

def fst(r: Array<F32> & Array<F32>) -> Array<F32>:
  (a, b) = r
  a

def run(sp: String, qp: String, kp: String, vp: String, dp: String, bp: String, ku: U32) -> IO(Unit):
  do IO<Unit>:
    +k : Nat = U32.to_nat(ku)
    S : Array<F32> <- I.io.load(sp, Nat.add(k, 14n), 0.0)
    q : Array<F32> <- I.io.load(qp, Nat.add(k, 7n), 0.0)
    kk : Array<F32> <- I.io.load(kp, Nat.add(k, 7n), 0.0)
    v : Array<F32> <- I.io.load(vp, Nat.add(k, 7n), 0.0)
    dec : Array<F32> <- I.io.load(dp, k, 0.0)
    bet : Array<F32> <- I.io.load(bp, k, 0.0)
    I.io.emit_vec(U32.to_nat((1 << Nat.add(k, 7n) : U32)), fst(K.gated_delta(k, S, q, kk, v, dec, bet)))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{sp, Con{qp, Con{kp, Con{vp, Con{dp, Con{bp, Con{ks, Nil{}}}}}}}}:
      do IO<Unit>:
        k : U32 <- I.io.u32(ks)
        run(sp, qp, kp, vp, dp, bp, k)
    case _:
      IO.die(Unit, 2, "usage: gated_delta <S.bin> <q.bin> <k.bin> <v.bin> <dec.bin> <bet.bin> <k>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
