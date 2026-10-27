#!/bin/bash
# g4.sh MODEL...: the synthetic shapes. Per model: two prompts x 32 tokens, the engine and llama.cpp Metal (the
# floor) each scored against llama.cpp CPU (tf.py), and the engine against llama.cpp Metal (G4c); the MoE FFN oracle (LLB_MX_DUMP, 2 steps, mx_ref.py float64);
# the forced split (LLB_MX_SPLIT=3) and streamed (LLB_MX_STREAM=1) paths: ids and logits equal to resident
cd ~/scratch/metal-moe; R=runs/g4; mkdir -p $R
for M in "$@"; do S=side-$M
  for p in p1 p2; do
    python3 tf.py moe $M $S runs/$p.ids 32 > $R/$M-moe-$p.json
    python3 tf.py llamagpu $M x runs/$p.ids 32 > $R/$M-floor-$p.json
    TFREF=gpu python3 tf.py moe $M $S runs/$p.ids 32 > $R/$M-moevsgpu-$p.json
  done
  : > "$R/$M.dump"
  LLB_MX_DUMP=$R/$M.dump ./run.sh moe $M $S @runs/p1.ids --steps 2 > /dev/null 2> $R/$M.dump.err
  PYTHONPATH=~/llama.cpp/gguf-py:pystub python3 mx_ref.py models/$M.gguf $R/$M.dump | tail -1 > $R/$M.oracle.json
  for mode in "LLB_MX_RES=1" "LLB_MX_SPLIT=3" "LLB_MX_STREAM=1"; do
    env $mode ./run.sh moe $M $S @runs/p2.ids --steps 32 > $R/f.out 2> $R/f.err
    echo "$M $mode ids $(grep -v wall $R/f.out | tr "\n" " " | shasum | cut -c1-16) logits $(grep "^logits" $R/f.err | shasum | cut -c1-16) | $(grep "mx pool" $R/f.err | sed "s/.*mapped from [^;]*; //")"
  done >> $R/forced.txt
done
echo done >> $R/DONE
