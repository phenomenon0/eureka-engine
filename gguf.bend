# gguf.bend: GGUF2, the engine's weights straight from a GGUF (LLB_GGUF=path).
# The C side (host/llb_gguf.c) derives gguf2pack's pack layout from the GGUF
# header and fills the asked region; the pack dir is then only the sidecar
# (manifest.json + table.bin, gguf2pack --sidecar-only).
import Base

# which = layer N (its whole region; n must be its size), or 0xFFFFFFFF =
# the virtual pack.bin [off, off + n)
def Gguf.read(which: U32, off: U32, n: U32) -> IO(Result<&1, &1, U32 & String, Bytes()>):
  import "./host/llb_gguf.c"
  import "./host/llb_gguf.js"

def gguf.get(+which: U32, +off: U32, +n: U32) -> IO(Bytes()):
  IO.try(Bytes(), Gguf.read(which, off, n))

# LLB_GGUF set and non-empty
def gguf.on(e: Result<&1, &1, U32 & String, String>) -> Bool:
  match e:
    case Done{v}:
      Bool.not(String.eq(v, ""))
    case Fail{_}:
      False{}
