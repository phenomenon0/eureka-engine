#!/bin/bash
# needle.sh: G5, the metal-moe needle prompt (153,352 ids, PELICAN-7342 at 45%) through the final build's default
# (fast) prefill, 24 greedy tokens; the answer's ids against llama-tokenize's ids of "PELICAN-7342"
cd "${LANE:-$HOME/scratch/metal-catchup}"; . ./paths.sh; R=runs/g5; mkdir -p $R
s=$(date +%s); ./run.sh final Qwen3.5-35B-A3B-Q4_K_S 35b @runs/needle140k.ids --steps 24 > $R/needle.out 2> $R/needle.err; rc=$?
echo "rc=$rc secs=$(( $(date +%s) - s ))" > $R/needle-summary.txt
want=$($HOME/llama.cpp/build/bin/llama-tokenize -m $M08 -p "PELICAN-7342" --ids --log-disable 2>/dev/null | tail -1 | tr -d "[],")
got=$(grep -v wall $R/needle.out | tr "\n" " ")
echo "want ids: $want" >> $R/needle-summary.txt; echo "got ids: $got" >> $R/needle-summary.txt
case " $got " in *" $want "*) echo "PELICAN-7342 found: YES" ;; *) echo "PELICAN-7342 found: NO" ;; esac >> $R/needle-summary.txt
awk '$1=="chunkf" {c+=$4; n++} END {print "chunks " n ", chunk ms " c}' $R/needle.err >> $R/needle-summary.txt
