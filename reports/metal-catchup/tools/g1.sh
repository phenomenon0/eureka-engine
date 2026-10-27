#!/bin/bash
# g1.sh: G1, dense no-regression (bit-exact): base (a20c8ee2) vs final on the 4B and 9B Q4_K_M (paths in paths.sh:
# M4, M9), three prompts x 64 greedy ids, exact and default prefill; the P2 logits lines (exact)
cd "${LANE:-$HOME/scratch/metal-catchup}"; . ./prompts.sh; . ./paths.sh; mkdir -p runs/g1
run() { GPU=16GB ./run.sh $1 $2 $3 ${@:4} > runs/g1/r.out 2> runs/g1/r.err; grep -v wall_ms runs/g1/r.out | tr "\n" " "; }
for m in 4b 9b; do G=$([ $m = 4b ] && echo $M4 || echo $M9); for p in P1 P2 P3; do for tier in --exactpf ""; do
  a=$(run base $G $m ${!p} --steps 64 $tier); b=$(run final $G $m ${!p} --steps 64 $tier)
  echo "$m $p ${tier:-default} same=$([ "$a" = "$b" ] && [ -n "$a" ] && echo YES || echo NO) ids: $b"
done; done
  run base $G $m $P2 --steps 64 --exactpf > /dev/null; la=$(grep "^logits" runs/g1/r.err | shasum | cut -c1-16)
  run final $G $m $P2 --steps 64 --exactpf > /dev/null; lb=$(grep "^logits" runs/g1/r.err | shasum | cut -c1-16)
  echo "$m P2 logits lines: base $la final $lb same=$([ "$la" = "$lb" ] && echo YES || echo NO)"
done
