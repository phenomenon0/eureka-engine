# probe_rb (quant-audit test 5) <w.bin> <x.bin> <wbytes> <n> <d> <rb>: one F.llb.gemv_warp! over the
# n rows in w.bin, the row-bytes word rb passed exactly as engine_gpu.bend's rw passes it (F16: top
# bit set; tagged words, tools/quant-fix.patch, carry the format), on x (2^13 f32 slots,
# zero-padded). The n outputs go to stdout as decimal U32 bits. On
# the GPU the bang takes llb_mail's format pick (tools/merge3-f16.patch); with --gpu off, F.mv's.
import Base
import ./ops.bend as K
import ./flat.bend as F
import ./io.bend as I

@unsafe
def loaded(+x: Array<F32>, +n: U32, +d: U32, +rb: U32, got: File & Bytes()) -> IO(Unit):
  (h, +w) = got
  do IO<Unit>:
    File.close(h)
    I.io.emit_vec(U32.to_nat(n), F.llb.gemv_warp!(14n, w, x, Array.new(F32, 13n, 0.0), 0, n, U32.to_nat(d), rb))

@unsafe
def go(args: List<String>) -> IO(Unit):
  match args:
    case Con{wp, Con{xp, Con{bs, Con{ns, Con{ds, Con{rs, Nil{}}}}}}}:
      do IO<Unit>:
        x : Array<F32> <- I.io.load(xp, 13n, 0.0)
        b : U32 <- I.io.u32(bs)
        n : U32 <- I.io.u32(ns)
        d : U32 <- I.io.u32(ds)
        rb : U32 <- I.io.u32(rs)
        f : File <- IO.try(File, File.open(wp, "r"))
        got : File & Bytes() <- I.io.buf.read(f, b)
        loaded(x, n, d, rb, got)
    case _:
      IO.die(Unit, 2, "usage: probe_rb <w.bin> <x.bin> <wbytes> <n> <d> <rb>")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    go(args)
