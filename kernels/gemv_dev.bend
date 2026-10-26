# gemv_dev <w.bin> <byte_offset> <rows> <x.bin> <d> <row_bytes> <reps>: y = W x
# through dev.bend's bang, W read once into resident Bytes; x has 2^d leaves.
# stdout: y's F32 bits as gemv prints them; stderr: `bang <ms>` per rep.
import Base
import ./ops.bend as K
import ./io.bend as I
import ./dev.bend as D

def xb.of(a: Array<F32>) -> Bytes():
  D.xbytes(Array.to_list(~F32, a), SNil{})

def keep(r: D.Ys & Bytes()) -> Bytes():
  (ys, wb) = r
  wb

def emit(r: D.Ys & Bytes()) -> IO(Unit):
  (ys, wb) = r
  I.io.emit_rev(D.flat(ys, Nil{}), Nil{})

def rep.fin(t0: Nat, r: D.Ys & Bytes()) -> IO(D.Ys & Bytes()):
  do IO<D.Ys & Bytes()>:
    t1 : Nat <- IO.now()
    IO.print_err("bang " ++ Nat.show(Nat.sub(t1, t0)))
    return r

# one timed bang, entered after an IO step (the solo loop: the GPU's entry)
def once(+n: U32, +d: Nat, +rb: U32, +xb: Bytes(), wb: Bytes()) -> IO(D.Ys & Bytes()):
  do IO<D.Ys & Bytes()>:
    t0 : Nat <- IO.now()
    rep.fin(t0, D.gemv(wb, xb, 0, n, Nat.sub(d, 2n), rb))

def rep(k: Nat, +n: U32, +d: Nat, +rb: U32, +xb: Bytes(), wb: Bytes()) -> IO(D.Ys & Bytes()):
  match k:
    case 0n:
      once(n, d, rb, xb, wb)
    case 1n+p:
      do IO<D.Ys & Bytes()>:
        r : D.Ys & Bytes() <- once(n, d, rb, xb, wb)
        rep(p, n, d, rb, xb, keep(r))

def loaded(k: Nat, +n: U32, +d: Nat, +rb: U32, +xb: Bytes(), got: File & Bytes()) -> IO(Unit):
  (f, wb) = got
  do IO<Unit>:
    File.close(f)
    r : D.Ys & Bytes() <- rep(k, n, d, rb, xb, wb)
    emit(r)

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{os, Con{ns, Con{xp, Con{ds, Con{rbs, Con{ks, Nil{}}}}}}}}:
      do IO<Unit>:
        +off : U32 <- I.io.u32(os)
        +n : U32 <- I.io.u32(ns)
        dd : U32 <- I.io.u32(ds)
        +rb : U32 <- I.io.u32(rbs)
        k : U32 <- I.io.u32(ks)
        +d : Nat = U32.to_nat(dd)
        x : Array<F32> <- I.io.load(xp, d, 0.0)
        +xb : Bytes() = xb.of(x)
        f : File <- IO.try(File, File.open(wp, "r"))
        h : File <- I.io.buf.skip(U32.to_nat(off), U32.to_nat(off), f)
        got : File & Bytes() <- I.io.buf.read(h, (n * rb : U32))
        loaded(Nat.sub(U32.to_nat(k), 1n), n, d, rb, xb, got)
    case _:
      IO.die(Unit, 2, "usage: gemv_dev <w.bin> <byte_offset> <rows> <x.bin> <d> <row_bytes> <reps>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
