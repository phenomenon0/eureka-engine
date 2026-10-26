#!/usr/bin/env python3
"""apply_chunkarea.py IN OUT (DECIDE-CHAT-FIX): the emitted C -> pass_reserve reserves the fast chunk area at d_model 4096 / 2560
whatever the process's first pass was. pass_reserve runs once (the first m9_pass_init); it took the chunk area at 4096 / 2560 only
when that first pass was a chunk (f9_want, set by f27_chunk / xc_chunk). A process whose first pass was exact (a decide, or a
prompt under the fast tier's threshold) had f27_base 0 for good, and its first fast-tier generate exited in f27_chunk
("fast chunk: no chunk area ..."). The area sits past the layer state, so the state's addresses do not move; cost F27_BYTES
(152 MiB) of scratch in a process that never runs a chunk."""
import sys
src, out = sys.argv[1:3]
S = open(src).read()
old = '''  u64 fch = R->d == 5120 || ((R->d == 4096 || R->d == 2560) && f9_want)
    ? F27_BYTES : 0;
'''
new = '''  // (DECIDE-CHAT-FIX) at 4096 / 2560 too whatever the first pass was (an exact first pass = no area, then a chunk exits)
  u64 fch = R->d == 5120 || R->d == 4096 || R->d == 2560 ? F27_BYTES : 0;
'''
assert S.count(old) == 1, S.count(old)
open(out, 'w').write(S.replace(old, new))
