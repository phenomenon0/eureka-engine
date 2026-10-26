#!/usr/bin/env bash
# run27.sh ENG OUT STEPS ids...: one 27B engine run (--gpu 40GB) under the bench lock,
# idle GPU only; env LLB_Q4_DP4A passes through. Receipts OUT.{out,err,state}
set -uo pipefail
cd "$(dirname "$0")/../.."
e=$1 o=$2 s=$3; shift 3
P=${PACK27:?set PACK27 to the 27B pack directory}
{ date -u +%FT%TZ; sha256sum $e $e.gpu; echo "LLB_Q4_DP4A=${LLB_Q4_DP4A-unset}"; echo "cmd $e --gpu 40GB --threads 16 -- $P $* --steps $s"; } > $o.state
flock /tmp/llb-gpu-bench.lock bash -c '[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy" >&2; exit 3; }; exec timeout 580 "$@"' _ \
  $e --gpu 40GB --threads 16 -- $P "$@" --steps $s > $o.out 2> $o.err
echo "exit $?" >> $o.state
