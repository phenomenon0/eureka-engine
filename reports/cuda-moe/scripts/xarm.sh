#!/usr/bin/env bash
# xarm.sh TAG GGUF : X = llama.cpp CPU (op offload off) TF on the F32-dequantized copy (deleted after)
S=$HOME/worktrees/.scratch-cuda-moe; tag=$1 g=$2; o=$S/receipts/gate/$tag
P1="760 511 314 938 369"; P2="774 264 2716 3170 883 279 1741 13"; P3="727 731 1393 1590"
export TMPDIR=$S/tmp PYTHONPATH=$HOME/llama.cpp/gguf-py
x=${XDIR:-$S/models}/$tag-f32.gguf
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 python3 $S/scripts/deq_stream.py $g $x > $o/X-deq.txt 2>&1
for i in 1 2 3; do eval p=\$P$i
  gen=$(python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['gen'][:63])))" $o/R-p$i.json)
  systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock timeout 3600 $S/ref/ref_tf $x $(echo "$p $gen" | tr ' ' ',') 0 > $o/X-p$i.json 2> $o/X-p$i.err
done
rm -f $x
