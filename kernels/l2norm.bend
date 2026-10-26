# l2norm <v.bin> <k> [q]: per-head L2 norm + scale over 2^k heads of 128.
# q=1 divides q by sqrt(128) after the l2 scale (the w2c engine's q leg);
# q=0 (default) is the k leg. v.bin holds 2^k * 128 f32.
import Base
import ./ops.bend as K
import ./io.bend as I

def run(vp: String, k: U32, q: U32) -> IO(Unit):
  do IO<Unit>:
    +kn : Nat = U32.to_nat(k)
    v : Array<F32> <- I.io.load(vp, Nat.add(kn, 7n), 0.0)
    +sc : F32 = Bool.pick(F32, U32.is_eq(q, 0), 1.0, F32.sqrt(128.0))
    I.io.emit_vec(U32.to_nat((1 << Nat.add(kn, 7n) : U32)), K.l2norm(kn, v, sc))

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{vp, Con{ks, Nil{}}}:
      do IO<Unit>:
        k : U32 <- I.io.u32(ks)
        run(vp, k, 0)
    case Con{vp, Con{ks, Con{qs, Nil{}}}}:
      do IO<Unit>:
        k : U32 <- I.io.u32(ks)
        q : U32 <- I.io.u32(qs)
        run(vp, k, q)
    case _:
      IO.die(Unit, 2, "usage: l2norm <v.bin> <k> [q]")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
