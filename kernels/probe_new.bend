# probe_new <reps>: the device cost of Array.new(F32, 2^d, 0.0) inside a bang,
# for d = 8, 11, 13, 15, reps bangs each (stderr `new <d> <ms> <bits>`). Read
# the per-bang grow launches under nsys: they scale with 2^d if the zero-fill
# runs on one lane.
import Base
import ./flat.bend as F
import ./io.bend as I

@unsafe
def mk(+d: Nat) -> U32:
  F32.bits(F.at(Array.new(F32, d, 0.0), 5))

def done(+d: Nat, +t0: Nat, v: U32) -> IO(Unit):
  do IO<Unit>:
    t1 : Nat <- IO.now()
    IO.print_err("new " ++ Nat.show(d) ++ " " ++ Nat.show(Nat.sub(t1, t0)) ++ " " ++ U32.show(v))

@unsafe
def reps(k: Nat, +d: Nat) -> IO(Unit):
  match k:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        done(d, t0, mk!(d))
        reps(p, d)

@unsafe
def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{ks, Nil{}}:
      do IO<Unit>:
        +k : U32 <- I.io.u32(ks)
        reps(U32.to_nat(k), 8n)
        reps(U32.to_nat(k), 11n)
        reps(U32.to_nat(k), 13n)
        reps(U32.to_nat(k), 15n)
    case _:
      IO.die(Unit, 2, "usage: probe_new <reps>")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
