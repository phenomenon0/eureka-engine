#!/usr/bin/env bash
S=$HOME/worktrees/.scratch-cuda-moe-speed
until [ -f $S/tmp/reg.done ]; do sleep 20; done
mkdir -p /dev/shm/cuda-moe-speed; for f in s35-q4km s128f-q4km; do cp $BIG/$f.gguf /dev/shm/cuda-moe-speed/; done
for r in 1 2 3; do $S/scripts/speed.sh $r; done
python3 $S/scripts/speedsum.py $S/receipts/speed > $S/receipts/speed/SPEED.txt 2>&1
touch $S/tmp/speed.done
rm -f /dev/shm/cuda-moe-speed/s35-q4km.gguf /dev/shm/cuda-moe-speed/s128f-q4km.gguf
