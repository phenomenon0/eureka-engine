#!/usr/bin/env bash
# reg.sh : G-REG, base vs candidate on the dense models
S=$HOME/worktrees/.scratch-cuda-moe; R=$S/receipts/reg
declare -A G=( [08b]=${M08B:?Qwen3.5-0.8B-Q4_K_M.gguf} [2b]=${M2B:?Qwen3.5-2B-Q4_K_M.gguf} [9b]=${M9B:?Qwen3.5-9B-Q4_K_M.gguf} [flash]=${MFLASH:?clef-flash-Q4_K_S.gguf} )
P1="760 6511 314 9338 369"; P2="7734 264 2716 31708 883 279 17415 13"; P3="727 73111 1393 1590"
for m in 08b 2b 9b flash; do for b in base new; do
  for i in 1 2 3; do eval p=\$P$i
    $S/scripts/run.sh $S/$b-out ${G[$m]} $S/side3/$m-base $R/$m-$b-p$i-exact.txt 64 $p
    LLB_Q4_DP4A=9 $S/scripts/run.sh $S/$b-out ${G[$m]} $S/side3/$m-base $R/$m-$b-p$i-dp9.txt 64 $p
  done
  LLB_PREFILL=fast $S/scripts/run.sh $S/$b-out ${G[$m]} $S/side3/$m-base $R/$m-$b-p2-fast.txt 64 $P2
done; done
n=0; bad=0
for f in $R/*-base-*.txt; do g=${f/-base-/-new-}; n=$((n+1))
  if cmp -s <(grep -v wall_ms $f.stdout) <(grep -v wall_ms $g.stdout) && cmp -s <(grep '^logits' $f.stderr) <(grep '^logits' $g.stderr) && grep -q '^rc=0' $g && [ -s $g.stdout ]; then :; else bad=$((bad+1)); echo "DIFF $(basename $g)"; fi
done
echo "G-REG: $((n-bad))/$n runs byte-identical (stdout ids + every logits line), base 61a673aa3 vs candidate"
