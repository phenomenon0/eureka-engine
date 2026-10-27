#!/usr/bin/env bash
# period.sh ENGDIR GGUF SIDE TAG [STEPS] : median LLB_STEP_US period over pos >= 12 (decode), prompt P1 (env passes through)
S=$HOME/worktrees/.scratch-cuda-catchup; o=${OUTDIR:-$S/receipts/dev}; mkdir -p $o
LLB_STEP_US=1 $S/scripts/run.sh $1 $2 $3 $o/$4.txt ${5:-128} 760 511 314 938 369
grep -q '^rc=0' $o/$4.txt || { echo "$4 FAILED"; exit 1; }
grep '^stepus' $o/$4.txt.stderr | awk '$2>=12{print $3}' | sort -n | awk -v t=$4 '{a[NR]=$1} END{m=a[int((NR+1)/2)]; printf "%s median %d us -> %.1f tok/s (n %d)\n", t, m, 1e6/m, NR}'
