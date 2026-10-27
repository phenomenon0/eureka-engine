#!/usr/bin/env bash
# arms.sh TAG GGUF ENG CTAG : greedy 64 steps + TF (prompt + R's first 63 ids) on 3 prompts (env passes through)
S=$HOME/worktrees/.scratch-cuda-catchup; tag=$1 g=$2 eng=$3 ce=$4
P1="760 511 314 938 369"; P2="774 264 2716 3170 883 279 1741 13"; P3="727 731 1393 1590"
o=$S/receipts/${GD:-gate}/$tag; side=$S/side/$tag
for i in 1 2 3; do eval p=\$P$i
  gen=$(python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['gen'][:63])))" $o/R-p$i.json)
  [ -n "$NOGREEDY" ] || $S/scripts/run.sh $S/$eng $g $side $o/$ce-greedy-p$i.txt 64 $p
  $S/scripts/run.sh $S/$eng $g $side $o/$ce-tf-p$i.txt 1 $p $gen
done
