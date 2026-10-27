#!/usr/bin/env bash
# arms.sh TAG GGUF [ENGDIR] : sidecar + R (llama.cpp CPU greedy) + F (llama.cpp CUDA TF) + C (engine greedy + TF), 3 prompts
set -u
S=$HOME/worktrees/.scratch-cuda-moe; tag=$1 g=$2 eng=${3:-new-out}
P1="760 511 314 938 369"; P2="774 264 2716 3170 883 279 1741 13"; P3="727 731 1393 1590"
export TMPDIR=$S/tmp
o=$S/receipts/gate/$tag; mkdir -p $o
side=$S/side2/$tag
[ -f $side/manifest.json ] || $S/new-out/gguf2pack $g --sidecar-only $side > $o/sidecar.txt 2>&1 || { echo "sidecar failed"; cat $o/sidecar.txt; exit 1; }
cap() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout 1800 "$@"; }
for i in 1 2 3; do eval p=\$P$i; pc=$(echo $p | tr ' ' ',')
  [ -s $o/R-p$i.json ] || cap $S/ref/ref_greedy $g $pc 64 > $o/R-p$i.json 2> $o/R-p$i.err
  gen=$(python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['gen'][:63])))" $o/R-p$i.json)
  tf="$p $gen"
  [ -s $o/F-p$i.json ] || cap $S/ref/ref_tf $g $(echo $tf | tr ' ' ',') 99 > $o/F-p$i.json 2> $o/F-p$i.err
  [ "$eng" = none ] && continue
  ce=${CTAG:-C}
  $S/scripts/run.sh $S/$eng $g $side $o/$ce-greedy-p$i.txt 64 $p
  $S/scripts/run.sh $S/$eng $g $side $o/$ce-tf-p$i.txt 1 $tf
done
