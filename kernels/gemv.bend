# gemv <w.bin> <byte_offset> <rows> <x.bin>: y = W x, W [rows, 2048] f32le read row by row at the offset
import Base
import ./ops.bend as K
import ./io.bend as I

def emit(r: Array<F32> & List<Array<F32>>) -> IO(Unit):
  (x, ys) = r
  I.io.emit_rev(ys, Nil{})

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{os, Con{ns, Con{xp, Nil{}}}}}:
      do IO<Unit>:
        off : U32 <- I.io.u32(os)
        n : U32 <- I.io.u32(ns)
        x : Array<F32> <- I.io.load(xp, 11n, 0.0)
        r : Array<F32> & List<Array<F32>> <- I.io.gemv("gemv", wp, off, U32.to_nat(n), 11n, 8192, x)
        emit(r)
    case _:
      IO.die(Unit, 2, "usage: gemv <w.bin> <byte_offset> <rows> <x.bin>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
