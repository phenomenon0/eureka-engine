#!/bin/bash
# reg.sh: G1, dense no-regression: the any-shape build (out-any, 00979448) vs the MoE build (out-moe) on the 4B and
# 9B Q4_K_M, three prompts x 64 greedy ids, exact and default prefill; and the logits lines on P2 (exact)
cd ~/scratch/metal-moe; . ./prompts.sh
G4=${G4:-MODELS/Qwen3.5-4B-Q4_K_M.gguf}; G9=${G9:-MODELS/Qwen3.5-9B-Q4_K_M.gguf}
run() { GPU=16GB ./run.sh $1 $2 $3 ${@:4} > runs/reg.out 2> runs/reg.err; echo "$(grep -v wall_ms runs/reg.out | tr "\n" " ")"; }
for m in 4b 9b; do G=$([ $m = 4b ] && echo $G4 || echo $G9); for p in P1 P2 P3; do for tier in --exactpf ""; do
  a=$(run any $G $m ${!p} --steps 64 $tier); b=$(run moe $G $m ${!p} --steps 64 $tier)
  echo "$m $p ${tier:-default} same=$([ "$a" = "$b" ] && [ -n "$a" ] && echo YES || echo NO) ids: $b"
done; done
  run any $G $m $P2 --steps 64 --exactpf > /dev/null; grep "^logits" runs/reg.err | shasum > runs/la.sha
  run moe $G $m $P2 --steps 64 --exactpf > /dev/null; grep "^logits" runs/reg.err | shasum > runs/lb.sha
  echo "$m P2 logits lines: any $(cut -c1-16 runs/la.sha) moe $(cut -c1-16 runs/lb.sha) same=$(cmp -s runs/la.sha runs/lb.sha && echo YES || echo NO)"
done
