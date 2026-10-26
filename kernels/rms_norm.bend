# rms_norm <x.bin> <w.bin>: d_model 2048
import Base
import ./ops.bend as K
import ./io.bend as I

def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{xp, Con{wp, Nil{}}}:
      do IO<Unit>:
        x : Array<F32> <- I.io.load(xp, 11n, 0.0)
        w : Array<F32> <- I.io.load(wp, 11n, 0.0)
        I.io.emit_vec(2048n, K.rms_norm(x, w, 2048.0))
    case _:
      IO.die(Unit, 2, "usage: rms_norm <x.bin> <w.bin>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
