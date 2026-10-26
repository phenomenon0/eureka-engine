#!/usr/bin/env bash
# gate S (spec decode lossless under sampling) + its mutant. CPU only. usage: run_spec_stat.sh [N]
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=${OUT:-/tmp/llb-sampler}; mkdir -p "$OUT"
gcc -O2 -std=c11 -ffp-contract=off -Wall -Wextra "$here/spec_stat.c" -o "$OUT/spec_stat" -lm || exit 2
gcc -O2 -std=c11 -ffp-contract=off -DMUT_SPEC_NORESID "$here/spec_stat.c" -o "$OUT/spec_stat_MUT" -lm || exit 2
"$OUT/spec_stat" "${1:-400000}"; rc=$?
if "$OUT/spec_stat_MUT" "${1:-400000}" > "$OUT/spec_stat_MUT.log"; then echo "MUTANT SPEC_NORESID: NOT CAUGHT"; rc=1
else echo "MUTANT SPEC_NORESID: caught ($(grep -c FAIL "$OUT/spec_stat_MUT.log") failing checks)"; fi
exit $rc
