#!/usr/bin/env bash
# probe.sh PROBE PROBE12 [--update]: llb-probe's refusal paths (LLB_PROBE_FAKE, test only) vs tools/llb/golden/probe-*.txt
# (output + "exit N"). PROBE = the cu13 build (floor 13000), PROBE12 = the cu12 build (floor 12020). P6 (docs/records/pkg2-registration.md).
set -u
p=$1 p12=$2 gd=$(dirname "$0")/../golden upd=${3-} bad=0 n=0
g() {  # g NAME FAKE BIN ARGS...
  local name=$1 fake=$2; shift 2; n=$((n + 1))
  local out; out=$(LLB_PROBE_FAKE=$fake "$@" 2>&1; echo "exit $?")
  if [ "$upd" = --update ]; then printf '%s\n' "$out" > $gd/probe-$name.txt; echo "wrote probe-$name.txt"
  elif [ "$out" = "$(cat $gd/probe-$name.txt)" ]; then echo "PASS probe-$name"
  else echo "FAIL probe-$name"; diff $gd/probe-$name.txt <(printf '%s\n' "$out"); bad=$((bad + 1)); fi
}
g ok "" $p --need-mib 20000
g cc75 cc=7.5 $p
g managed0 managed=0 $p
g driver12020 driver=12020 $p
g vram8 vram=8 $p --need-mib 20000
g quiet-ok "" $p -q --need-mib 20000
g cu12-driver12020 driver=12020 $p12
g cu12-driver12000 driver=12000 $p12 -q
g l2-24mb l2=24576 $p
[ "$upd" = --update ] || echo "P6 $((n - bad))/$n PASS"
