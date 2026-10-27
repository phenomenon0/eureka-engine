#!/bin/bash
cd ~/scratch/metal-catchup
./run.sh base Qwen3.5-35B-A3B-Q4_K_S 35b @runs/pp4096.ids --steps 1 > runs/basepf.out 2> runs/basepf.err
awk '$1=="step" {s+=$3; n++} END {print "base exact prefill: " n " steps, " s " ms, " n*1000/s " tok/s"}' runs/basepf.err > runs/basepf.txt
./needle.sh
./mk_synth.sh > runs/mk_synth.log 2>&1
./g4c.sh s128x8b-Q4_K_M s60x6-Q4_K_M s40x5f-Q4_K_M s40x5f-Q8_0 > runs/g4c.log 2>&1
echo ALLDONE > runs/drive2.done
