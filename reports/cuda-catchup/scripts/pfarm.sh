#!/usr/bin/env bash
# pfarm.sh TAG GGUF ENG CTAG : G-PF prefix probes: for each prompt i, one multi-prompt run of the 64 prefixes
# seq[:n0 + k] (k = 0..63; seq = prompt + R's first 63 ids), --steps 1 (env passes through, e.g. LLB_PREFILL=fast)
S=$HOME/worktrees/.scratch-cuda-catchup; tag=$1 g=$2 eng=$3 ce=$4
P1="760 511 314 938 369"; P2="774 264 2716 3170 883 279 1741 13"; P3="727 731 1393 1590"
o=$S/receipts/${GD:-gate}/$tag; side=$S/side/$tag
for i in 1 2 3; do eval p=\$P$i
  ids=$(python3 -c "
import json,sys
p=sys.argv[2].split(); g=json.load(open(sys.argv[1]))['gen'][:63]; s=p+[str(x) for x in g]; n0=len(p)
print(' , '.join(' '.join(s[:n0+k]) for k in range(64)))" $o/R-p$i.json "$p")
  $S/scripts/run.sh $S/$eng $g $side $o/$ce-pf-p$i.txt 1 $ids
done
