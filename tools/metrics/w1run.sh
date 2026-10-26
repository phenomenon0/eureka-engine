#!/usr/bin/env bash
# w1run.sh <out>: W1-METRICS GPU gate run under the shared bench lock (guard.sh rules):
# flock /tmp/llb-gpu-bench.lock, refuse if the card has compute users or memory is short,
# then golden(base) + golden(metrics) + load(metrics) inside a memory-capped user scope.
set -uo pipefail
out=$1; here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
exec 9>>/tmp/llb-gpu-bench.lock
flock -w 5400 9 || { echo "REFUSED: lock timeout" | tee -a "$out/guard.log"; exit 75; }
# any other compute user on the card refuses the run
users=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z $users ]] || { echo "REFUSED: GPU compute users $users" | tee -a "$out/guard.log"; exit 75; }
read -r avail swap < <(awk '/MemAvailable:/{m=$2}/SwapFree:/{s=$2}END{print m,s}' /proc/meminfo)
(( avail > 8388608 )) || { echo "REFUSED: MemAvailable ${avail} kB" | tee -a "$out/guard.log"; exit 76; }
echo "locked $(date -Ins) avail_kb=$avail swap_kb=$swap" >> "$out/guard.log"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv >> "$out/guard.log"
run() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 "$@"; }
export W1_ROOT=${W1_ROOT:-$(cd "$here/../.." && pwd)}
run python3 -I "$here/w1load.py" golden "$W1_BASE_BIN" "$out/golden-base"; echo "golden-base rc=$?" >> "$out/guard.log"
run python3 -I "$here/w1load.py" golden "$W1_MET_BIN" "$out/golden-metrics"; echo "golden-metrics rc=$?" >> "$out/guard.log"
run python3 -I "$here/w1load.py" load "$W1_MET_BIN" "$out/load"; echo "load rc=$?" >> "$out/guard.log"
echo "released $(date -Ins)" >> "$out/guard.log"
