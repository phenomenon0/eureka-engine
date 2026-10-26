#!/usr/bin/env bash
# tools/sys1dual/build.sh: the SYS1-DUAL engine = the plain engine build (build.sh)
# + the request objects (dual.c and the unmodified CLEF-3 head/spans/json/tokenizer
# sources, clef-decide's flags) linked in through LLB_CC_EXTRA. CPU only; the .gpu
# program is made on the first run (under the bench lock).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tools/sys1dual/env.sh"
o=$W/build/obj
mkdir -p "$o" "$W/sidecar"
c=${CLEF_OUT:-$root/build/clef3}/sidecar
[ -d "$c" ] && cp -n "$c"/{chat.flags,manifest.json,table.bin,tokenizer.bin} "$W/sidecar/" || echo "note: no sidecar at $c (build one with tools/clef3/build.sh)" >&2
for f in sys1dual/dual.c metrics/metrics.c metrics/tokstream.c glue/glue_out.c clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
  clang -D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function \
    -c "$root/tools/$f" -o "$o/$(basename "$f" .c).o"
done
LLB_CC_EXTRA="$(echo "$o"/*.o) -lopenblas" LLB_OUT=$W/build bash "$root/build.sh"
