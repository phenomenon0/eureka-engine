#!/usr/bin/env bash
# gpu.sh OUT CMD...: one GPU job (CMD, usually an llb command) in its OWN transient unit (systemd-run --user,
# MemoryMax=12G, nice 10, CUDA_VISIBLE_DEVICES=0) under /tmp/llb-gpu-bench.lock with an empty card
# (tools/gguf2/run.sh's wrapper: up to 60 s for the last holder; exit 3 busy / 75 lock wait over -> wait 60 s, retry <= 10).
# LLB_* env passes through. After CMD, pgrep -x llb-engine = orphan check.
# Receipts OUT.{out,err,state,unit}; OUT.state ends with "exit N", wall seconds, and the orphan count.
set -uo pipefail
o=$1; shift
mkdir -p "$(dirname $o)"
{ date -u +%FT%TZ; env | grep '^LLB_' | sort; printf 'cmd'; printf ' %q' "$@"; echo; } > $o.state
envs=(-E CUDA_VISIBLE_DEVICES=0 -E HOME="$HOME" -E PATH="$PATH" -E TO="${TO:-1800}" -E IN="${IN:-/dev/null}"); for v in $(env | grep -o '^LLB_[A-Z0-9_]*'); do envs+=(-E "$v"); done
for try in 1 2 3 4 5 6 7 8 9 10; do
  systemd-run --user --wait --collect -p MemoryMax=12G -p MemorySwapMax=0 -p Nice=10 --working-directory=$PWD "${envs[@]}" \
    flock -E 75 -w 1800 /tmp/llb-gpu-bench.lock bash -c 'w=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do [ $w -ge 60 ] && { echo "GPU busy" >&2; exit 3; }; sleep 1; w=$((w + 1)); done
    t0=$(date +%s.%N); timeout $TO "$@" < "${IN:-/dev/null}" > '"$o.out"' 2> '"$o.err"'; r=$?
    echo "wall_s $(awk -v a=$t0 -v b=$(date +%s.%N) "BEGIN { printf \"%.2f\", b - a }")" >> '"$o.state"'
    echo "orphans $(pgrep -xc llb-engine || true)" >> '"$o.state"'
    exit $r' _ "$@" 2> $o.unit
  r=$?
  [ "$r" = 3 ] || [ "$r" = 75 ] || break
  echo "busy, try $try" >> $o.state; sleep 60
done
echo "exit $r" >> $o.state
tail -3 $o.state
