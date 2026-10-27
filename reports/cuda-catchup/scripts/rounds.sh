#!/usr/bin/env bash
S=$HOME/worktrees/.scratch-cuda-catchup
for r in 1 2 3; do $S/scripts/speed.sh $r; done
python3 $S/scripts/speedsum.py $S/receipts/speed > $S/receipts/speed/SPEED.txt 2>&1
touch $S/tmp/speed.done
