#!/bin/bash
# g3.sh: the real 35B-A3B through the general MoE path vs llama.cpp (CPU, greedy) on the same GGUF: three prompts x
# 64 tokens; plus the controls (llama.cpp vs itself; llama.cpp Metal vs CPU, the floor) and the forced paths
# (LLB_MX_SPLIT, LLB_MX_STREAM: same ids and logits as resident)
cd ~/scratch/metal-moe; M=Qwen3.5-35B-A3B-Q4_K_S; S=side-35b; R=runs/g3; mkdir -p $R
export GPU=${GPU:-8GB}
for p in p1 p2 p3; do
  python3 tf.py moe $M $S runs/$p.ids 64 > $R/moe-$p.json
  python3 tf.py llamagpu $M x runs/$p.ids 64 > $R/floor-$p.json
done
python3 tf.py llamacpu $M x runs/p1.ids 64 > $R/perfect-p1.json
LLB_MX_MUT=no_renorm python3 tf.py moe $M $S runs/p2.ids 64 > $R/mutant-p2.json
for mode in "LLB_MX_SPLIT=16" "LLB_MX_STREAM=1"; do
  env $mode ./run.sh moe $M $S @runs/p2.ids --steps 64 > $R/forced.out 2> $R/forced.err
  a=$(grep -v wall runs/tf/$M-p2.ids-64.moe.out | tr "\n" " "); b=$(grep -v wall $R/forced.out | tr "\n" " ")
  la=$(grep "^logits" runs/tf/$M-p2.ids-64.moe.err | shasum | cut -c1-16); lb=$(grep "^logits" $R/forced.err | shasum | cut -c1-16)
  echo "$mode: ids same=$([ "$a" = "$b" ] && echo YES || echo NO) logits $la vs $lb same=$([ $la = $lb ] && echo YES || echo NO); $(grep "mx pool" $R/forced.err | sed "s/.*; //") $(grep wall_ms $R/forced.out)"
done > $R/forced.txt
echo done > $R/DONE
