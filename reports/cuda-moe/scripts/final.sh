#!/usr/bin/env bash
# final.sh TAG GGUF : the final candidate on one gate file: determinism check, C (device), G-MODE arms, scores
S=$HOME/worktrees/.scratch-cuda-moe; tag=$1 g=$2; o=$S/receipts/gate/$tag
export TMPDIR=$S/tmp
pc=760,511,314,938,369
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock timeout 1800 $S/ref/ref_greedy $g $pc 64 > $o/R-p1.rerun.json 2>/dev/null
cmp -s $o/R-p1.json $o/R-p1.rerun.json && echo "determinism $tag: R p1 identical to the first run" > $o/DETERMINISM.txt || echo "determinism $tag: R p1 DIFFERS" > $o/DETERMINISM.txt
sha256sum $g >> $o/DETERMINISM.txt
LLB_MOE_EXPERTS=device CTAG=C $S/scripts/arms.sh $tag $g > /dev/null 2>&1
LLB_MOE_EXPERTS=host CTAG=Chost $S/scripts/arms.sh $tag $g > /dev/null 2>&1
LLB_MOE_EXPERTS=stream CTAG=Cstream $S/scripts/arms.sh $tag $g > /dev/null 2>&1
LLB_MOE_VRAM=1000000000 LLB_MOE_RAM=1000000000 CTAG=Csmall $S/scripts/arms.sh $tag $g > /dev/null 2>&1
python3 $S/scripts/gate.py $o > $o/GATE.txt
[ -f $o/X-p1.json ] && python3 $S/scripts/gx.py $o > $o/GX.txt
$S/scripts/gmode.sh $tag > /dev/null
cat $o/DETERMINISM.txt $o/GMODE.txt; grep "VERDICT\|TF agree\|G-X" $o/GATE.txt $o/GX.txt 2>/dev/null
