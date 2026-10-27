#!/bin/bash
cd ~/scratch/metal-catchup
./g1.sh > runs/g1.txt 2>&1
./g89.sh Qwen3.5-35B-A3B-Q4_K_S 35b runs/p2.ids 64 > runs/g89-35b.txt 2>&1
./g89.sh Qwen3.5-35B-A3B-Q4_K_S 35b runs/pp1000.ids 16 >> runs/g89-35b.txt 2>&1
./g3.sh > runs/g3.log 2>&1
./g4b.sh > runs/g4b.log 2>&1
./g10.sh > runs/g10.log 2>&1
./speed.sh 3 > runs/speed.txt 2>&1
echo ALLDONE > runs/drive1.done
