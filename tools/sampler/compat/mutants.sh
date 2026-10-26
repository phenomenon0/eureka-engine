#!/usr/bin/env bash
# Each mutant must make its gate FAIL (exit != 0).  usage: mutants.sh  (needs cases/refs under $OUT)
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=${OUT:-/tmp/llb-sampler}; export OUT
check() { # name gate-cmd...
  local name=$1; shift
  if "$@" >"$OUT/mut_$name.log" 2>&1; then echo "MUTANT $name: NOT CAUGHT (gate passed) <-- gate is worthless"; return 1
  else echo "MUTANT $name: caught ($(grep -m1 -E 'FAIL|failing|gate[0-9]' "$OUT/mut_$name.log" | head -c 110))"; fi
}
rc=0
for m in TOPP_CUTOFF DROP_MINKEEP PEN_SIGN DIST_NOTEMP; do
  MUTSUF=_$m "$here/build.sh" -DMUT_$m >/dev/null 2>&1
  check $m "$OUT/gate1_$m" "$OUT/cases" "$OUT/refs" || rc=1
done
MUTSUF=_STOP_TRIM "$here/build.sh" -DMUT_STOP_TRIM >/dev/null 2>&1
check STOP_TRIM "$OUT/gate3_STOP_TRIM" || rc=1
MUTSUF=_RNG_BIAS "$here/build.sh" -DMUT_RNG_BIAS >/dev/null 2>&1
check RNG_BIAS nice -n 19 "$OUT/gate2_RNG_BIAS" "$OUT/cases" "$OUT/refs" 15 || rc=1
exit $rc
