# k35_dev — flat/device-body runner for the w2g Qwen3.5 hybrid ops.
#
# The same cases as kernels/{l2norm,conv1d,gated_rmsnorm,gated_delta}.bend,
# but exercising kernels/flat.bend's llb.* device bodies (which a CUDA patch
# can replace). Output is emitted as u32 bit lines, so it compares byte-for-
# byte with the emitted-CPU op runners. Op is dispatched by arity.
#
#   k35_dev <v.bin> <k> <q>                                          l2norm
#   k35_dev <o.bin> <w.bin> <z.bin> <k>                              gated_rmsnorm
#   k35_dev <x.bin> <h0.bin> <h1.bin> <h2.bin> <w.bin> <d>           conv1d
#   k35_dev <S.bin> <q.bin> <k.bin> <v.bin> <dec.bin> <bet.bin> <k>  gated_delta
import Base
import ./ops.bend as K
import ./flat.bend as F
import ./io.bend as I

def close_b(got: File & Bytes()) -> IO(Bytes()):
  (f, b) = got
  do IO<Bytes()>:
    File.close(f)
    return b

def file_bytes(path: String, n: U32) -> IO(Bytes()):
  do IO<Bytes()>:
    f : File <- IO.try(File, File.open(path, "r"))
    got : File & Bytes() <- I.io.buf.read(f, n)
    close_b(got)

def fst(r: Array<F32> & Array<F32>) -> Array<F32>:
  (a, b) = r
  a

def l2(vp: String, ks: String, qs: String) -> IO(Unit):
  do IO<Unit>:
    k : U32 <- I.io.u32(ks)
    q : U32 <- I.io.u32(qs)
    +kn : Nat = U32.to_nat(k)
    v : Array<F32> <- I.io.load(vp, Nat.add(kn, 7n), 0.0)
    +sc : F32 = Bool.pick(F32, U32.is_eq(q, 0), 1.0, F32.sqrt(128.0))
    I.io.emit_vec(U32.to_nat((1 << Nat.add(kn, 7n) : U32)), F.llb.l2norm(kn, v, sc))

def gnorm(op: String, wp: String, zp: String, ks: String) -> IO(Unit):
  do IO<Unit>:
    k : U32 <- I.io.u32(ks)
    +kn : Nat = U32.to_nat(k)
    o : Array<F32> <- I.io.load(op, Nat.add(kn, 7n), 0.0)
    z : Array<F32> <- I.io.load(zp, Nat.add(kn, 7n), 0.0)
    wb : Bytes() <- file_bytes(wp, 512)
    I.io.emit_vec(U32.to_nat((1 << Nat.add(kn, 7n) : U32)), F.llb.gated_rmsnorm(kn, o, wb, 0, z))

def conv(xp: String, p0: String, p1: String, p2: String, wp: String, ds: String) -> IO(Unit):
  do IO<Unit>:
    d : U32 <- I.io.u32(ds)
    +dn : Nat = U32.to_nat(d)
    x : Array<F32> <- I.io.load(xp, dn, 0.0)
    h0 : Array<F32> <- I.io.load(p0, dn, 0.0)
    h1 : Array<F32> <- I.io.load(p1, dn, 0.0)
    h2 : Array<F32> <- I.io.load(p2, dn, 0.0)
    wb : Bytes() <- file_bytes(wp, ((1 << Nat.add(dn, 2n) : U32) * 4 : U32))
    I.io.emit_vec(U32.to_nat((1 << dn : U32)), F.llb.conv1d(dn, wb, 0, x, h0, h1, h2))

def gdelta(sp: String, qp: String, kp: String, vp: String, dp: String, bp: String, ks: String) -> IO(Unit):
  do IO<Unit>:
    k : U32 <- I.io.u32(ks)
    +kn : Nat = U32.to_nat(k)
    S : Array<F32> <- I.io.load(sp, Nat.add(kn, 14n), 0.0)
    q : Array<F32> <- I.io.load(qp, Nat.add(kn, 7n), 0.0)
    kk : Array<F32> <- I.io.load(kp, Nat.add(kn, 7n), 0.0)
    v : Array<F32> <- I.io.load(vp, Nat.add(kn, 7n), 0.0)
    dec : Array<F32> <- I.io.load(dp, kn, 0.0)
    bet : Array<F32> <- I.io.load(bp, kn, 0.0)
    I.io.emit_vec(U32.to_nat((1 << Nat.add(kn, 7n) : U32)),
      fst(F.llb.gated_delta(kn, S, q, kk, v, dec, bet)))

def go(rest: List<String>) -> IO(Unit):
  match rest:
    case Con{a, Con{b, Con{c, Nil{}}}}:
      l2(a, b, c)
    case Con{a, Con{b, Con{c, Con{d, Nil{}}}}}:
      gnorm(a, b, c, d)
    case Con{a, Con{b, Con{c, Con{d, Con{e, Con{f0, Nil{}}}}}}}:
      conv(a, b, c, d, e, f0)
    case Con{a, Con{b, Con{c, Con{d, Con{e, Con{f0, Con{g, Nil{}}}}}}}}:
      gdelta(a, b, c, d, e, f0, g)
    case _:
      IO.die(Unit, 2, "usage: k35_dev <files...> (op by arity)")

def main.go(args: List<String>) -> IO(Unit):
  go(args)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
