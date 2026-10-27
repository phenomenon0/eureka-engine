#!/usr/bin/env bash
# devcmp.sh GGUF SIDE [STEPS] : dev check, base vs new device logits on prompt 1 (env passes through)
S=$HOME/worktrees/.scratch-cuda-moe-speed; g=$1 side=$2 n=${3:-32}; o=$S/receipts/dev; mkdir -p $o
[ -s $o/base-$(basename $side)-$n.txt.stderr ] || LLB_MOE_EXPERTS=device $S/scripts/run.sh $S/base-out $g $side $o/base-$(basename $side)-$n.txt $n 760 511 314 938 369
$S/scripts/run.sh $S/new-out $g $side $o/new-$(basename $side)-$n.txt $n 760 511 314 938 369
grep -q '^rc=0' $o/new-$(basename $side)-$n.txt || { echo "RUN FAILED"; tail -5 $o/new-$(basename $side)-$n.txt; }
cmp -s <(grep '^logits' $o/base-$(basename $side)-$n.txt.stderr) <(grep '^logits' $o/new-$(basename $side)-$n.txt.stderr) && echo "IDENTICAL $(grep -c '^logits' $o/new-$(basename $side)-$n.txt.stderr) logits lines" || { echo DIFFERS; diff <(grep '^logits' $o/base-$(basename $side)-$n.txt.stderr) <(grep '^logits' $o/new-$(basename $side)-$n.txt.stderr) | head -4; }
