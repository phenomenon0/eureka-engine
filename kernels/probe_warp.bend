# probe_warp <output.bin> <x.bin> <reps>: probe_head (llb-headprobe) with the head GEMV
# as the warp bang (F.llb.gemv_warp, tools/remap.patch) over resident output.weight bytes, on a given
# normed x. reps timed bangs (stderr `hg <ms> <bits of z[483]>`), then one more
# whose 130,560 logits go to stdout as decimal U32 bits, one per line.
import Base
import ./ops.bend as K
import ./flat.bend as F
import ./io.bend as I

@unsafe
def hg(+ow: Bytes(), +x: Array<F32>) -> Array<F32>:
  F.llb.gemv_warp!(14n, ow, x, Array.new(F32, 17n, K.neg_inf()), 0, 130560, 9n, 8192)

@unsafe
def at(+a: Array<F32>, +i: U32) -> U32:
  F32.bits(F.at(a, i))

def done(+t0: Nat, v: U32) -> IO(Unit):
  do IO<Unit>:
    t1 : Nat <- IO.now()
    IO.print_err("hg " ++ Nat.show(Nat.sub(t1, t0)) ++ " " ++ U32.show(v))

@unsafe
def reps(k: Nat, +ow: Bytes(), +x: Array<F32>) -> IO(Unit):
  match k:
    case 0n:
      I.io.emit_vec(130560n, hg!(ow, x))
    case 1n+p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        done(t0, at(hg!(ow, x), 483))
        reps(p, ow, x)

@unsafe
def loaded(k: Nat, +x: Array<F32>, got: File & Bytes()) -> IO(Unit):
  (h, +ow) = got
  do IO<Unit>:
    File.close(h)
    reps(k, ow, x)

@unsafe
def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{xp, Con{ks, Nil{}}}}:
      do IO<Unit>:
        x : Array<F32> <- I.io.load(xp, 11n, 0.0)
        f : File <- IO.try(File, File.open(wp, "r"))
        got : File & Bytes() <- I.io.buf.read(f, 1069547520)
        k : U32 <- I.io.u32(ks)
        loaded(U32.to_nat(k), x, got)
    case _:
      IO.die(Unit, 2, "usage: probe_warp <output.bin> <x.bin> <reps>")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
