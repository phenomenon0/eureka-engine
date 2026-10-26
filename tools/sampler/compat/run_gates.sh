#!/usr/bin/env bash
# Full gate run (CPU only, nice 19, <=2 jobs). Gate 2 takes ~5 min; set SKIP_G2=1 to skip. usage: run_gates.sh
set -eu
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); export OUT=${OUT:-/tmp/llb-sampler}
export CUDA_VISIBLE_DEVICES=
"$here/build.sh"
python3 "$here/gen_cases.py" "$OUT/cases"
mkdir -p "$OUT/refs"
ls "$OUT/cases"/*.args | sed 's/.*\///;s/\.args//' | xargs -P2 -I{} sh -c "[ -s $OUT/refs/{}.ref ] || nice -n 19 $OUT/ref_chain $OUT/cases/{}.f32 $OUT/refs/{}.ref \$(cat $OUT/cases/{}.args)"
nice -n 19 "$OUT/gate1" "$OUT/cases" "$OUT/refs"
[ "${SKIP_G2:-0}" = 1 ] || nice -n 19 "$OUT/gate2" "$OUT/cases" "$OUT/refs" 15 18 20
"$OUT/gate3"
python3 "$here/check_misc.py" "$OUT"
