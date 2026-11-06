#!/usr/bin/env bash
# G5 CPU suite: build + unit + template parity + HTTP e2e + text identity (+ encode parity if PY has torch).
#   tests/mm/run.sh OUTDIR [OLD_SERVE]      env: CC (clang-21), PY (python with PIL/jinja2/transformers),
#   MM_COCO (COCO val2017 000000142472.jpg: mm_test.c and http_e2e.py pin its 640x480 size and sha), MM_TEMPLATE
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd); O=${1:?outdir}; OLD=${2:-}
CC=${CC:-clang}; PY=${PY:-python3}
mkdir -p "$O/bin"; t=$root/tools
"$CC" -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -Werror -pthread $t/sys1dual/serve.c $t/sys1dual/mm.c $t/sys1dual/serve_structured.c $root/host/llb_structured.c $t/clef3/json.c -lm -o "$O/bin/llb-serve"
"$CC" -D_GNU_SOURCE -std=c11 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -Wall -Wextra -Werror \
  $t/sys1dual/mm_test.c $t/sys1dual/mm.c $t/clef3/json.c -o "$O/bin/mm_test"
"$CC" -D_GNU_SOURCE -std=c11 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -Wall -Wextra -Wno-unused-function \
  -pthread $root/tests/mm/render_driver.c $t/sys1dual/mm.c $t/sys1dual/serve_structured.c $root/host/llb_structured.c $t/clef3/json.c -lm -o "$O/bin/render_driver"
"$PY" -I $root/tests/mm/mkfix.py "$O/fix" "${MM_COCO:?set MM_COCO to COCO val2017 000000142472.jpg}"
echo "== unit";     "$O/bin/mm_test" "$O/fix" > "$O/unit.log" 2>&1; tail -1 "$O/unit.log"
echo "== parity";   "$PY" -I $root/tests/mm/parity.py --driver "$O/bin/render_driver" \
                      --template "${MM_TEMPLATE:?set MM_TEMPLATE to the Clef chat template}" --fix "$O/fix" --out "$O/parity" | tail -1
echo "== http e2e"; python3 -I $root/tests/mm/http_e2e.py --serve "$O/bin/llb-serve" --stub $root/tests/mm/stub_engine.py \
                      --fix "$O/fix" --out "$O/e2e" > "$O/e2e.log" 2>&1; tail -1 "$O/e2e.log"
if [ -n "$OLD" ]; then echo "== text identity"; python3 -I $root/tests/mm/text_identity.py --old "$OLD" --new "$O/bin/llb-serve" \
                      --stub $root/tests/mm/stub_engine.py --out "$O/ident" | tail -1; fi
sha256sum "$O/bin/llb-serve" $t/sys1dual/serve.c $t/sys1dual/mm.c $t/sys1dual/mm.h > "$O/sha256.txt"
