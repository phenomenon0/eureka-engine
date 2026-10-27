# sys1.bend: SYS1-DUAL request effects (docs/sys1dual-registration.md).
# The C side: sys1.c (spliced into the engine) + dual.c (linked object).
import Base

# the next request as engine argv words "id" ... "--steps" "N" ([] at EOF);
# ctx = the KV capacity fixed at launch (prompt + steps positions; was 2^dT)
def Sys1.next(dir: String, ctx: U32) -> IO(List<String>):
  import "./sys1.c"
  import "./sys1.js"

# the request's forward ran: answer it (one JSON line on stdout)
def Sys1.done(ok: U32) -> IO(Unit):
  import "./sys1.c"
  import "./sys1.js"

def sys1.next(+dir: String, +ctx: U32) -> IO(List<String>):
  Sys1.next(dir, ctx)

def sys1.done() -> IO(Unit):
  Sys1.done(1)
