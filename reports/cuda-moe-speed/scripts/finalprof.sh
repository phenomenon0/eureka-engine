#!/usr/bin/env bash
S=$HOME/worktrees/.scratch-cuda-moe-speed; M=${BIG:?a scratch dir for the big files}
until [ -f $S/tmp/speed.done ]; do sleep 30; done
mkdir -p /dev/shm/cuda-moe-speed; cp $M/s35-q4km.gguf /dev/shm/cuda-moe-speed/p.gguf
for b in baset new; do
  $S/scripts/prof.sh $S/$b-out /dev/shm/cuda-moe-speed/p.gguf $S/side/s35-q4km $S/receipts/prof/final-$b 128 760 511 314 938 369
  { python3 $S/scripts/split.py $S/receipts/prof/final-$b.sqlite; python3 $S/scripts/steps.py $S/receipts/prof/final-$b.sqlite 10; } > $S/receipts/prof/final-$b.txt 2>&1
done
rm -f /dev/shm/cuda-moe-speed/p.gguf
touch $S/tmp/prof.done
