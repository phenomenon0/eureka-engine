# The gemv row-dot order. v1 (False) = the strict balanced tree, THE ground
# truth. v2 (True) = the blocked dot, ops.bend's bdot (reports/llb-blocked.md).
# Chosen at build time only: tools/build.sh v2 compiles a copy with this
# flipped. The engine prints `order v1|v2`; verify_forward --order checks it.
import Base

def order.v2() -> Bool:
  False{}
