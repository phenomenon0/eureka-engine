#!/usr/bin/env bash
# reg.sh : G-REG (dense 0.8B / 2B / 9B / clef-flash, base 89dba9dde vs candidate: 3 prompts x 64 steps exact and
# LLB_Q4_DP4A=9, + LLB_PREFILL=fast on prompt 2), G-REG4 (4B exact 3 prompts), G-MULTI (0.8B, "P1 , P2 , P3"),
# G-OFF (0.8B LLB_GEN=0 vs default)
S=$HOME/worktrees/.scratch-cuda-catchup; R=$S/receipts/regC; mkdir -p $R
declare -A G=( [08b]=$HOME/.local/share/models/Qwen3.5-0.8B-Q4_K_M.gguf [2b]=$HOME/scratch/android-perf-models/dl-2b/Qwen3.5-2B-Q4_K_M.gguf [9b]=$HOME/.local/share/models/Qwen3.5-9B-Q4_K_M.gguf [flash]=$HOME/clef-dl/clef-flash-Q4_K_S.gguf [4b]=$HOME/lanes/echo-4b/models/Qwen3.5-4B-Q4_K_M.gguf )
P1="760 6511 314 9338 369"; P2="7734 264 2716 31708 883 279 17415 13"; P3="727 73111 1393 1590"
for m in 08b 2b 9b flash; do for b in base new; do
  for i in 1 2 3; do eval p=\$P$i
    $S/scripts/run.sh $S/${b/new/newC}-out ${G[$m]} $S/side/$m $R/$m-$b-p$i-exact.txt 64 $p
    LLB_Q4_DP4A=9 $S/scripts/run.sh $S/${b/new/newC}-out ${G[$m]} $S/side/$m $R/$m-$b-p$i-dp9.txt 64 $p
  done
  LLB_PREFILL=fast $S/scripts/run.sh $S/${b/new/newC}-out ${G[$m]} $S/side/$m $R/$m-$b-p2-fast.txt 64 $P2
done; done
for b in base new; do for i in 1 2 3; do eval p=\$P$i
  $S/scripts/run.sh $S/${b/new/newC}-out ${G[4b]} $S/side/4b $R/4b-$b-p$i-exact.txt 64 $p
done
  $S/scripts/run.sh $S/${b/new/newC}-out ${G[08b]} $S/side/08b $R/multi-$b-08b.txt 32 $P1 , $P2 , $P3
done
mkdir -p /dev/shm/cuda-catchup/regC; cp $BIG/s35-q4km.gguf /dev/shm/cuda-catchup/regC/s35-q4km.gguf
for b in base new; do
  $S/scripts/run.sh $S/${b/new/newC}-out /dev/shm/cuda-catchup/regC/s35-q4km.gguf $S/side/s35-q4km $R/multi-$b-s35.txt 32 760 511 314 938 369 , 774 264 2716 3170 883 279 1741 13 , 727 731 1393 1590
done
rm -f /dev/shm/cuda-catchup/regC/s35-q4km.gguf
LLB_GEN=0 $S/scripts/run.sh $S/newC-out ${G[08b]} $S/side/08b $R/off-gen0-08b.txt 64 $P1
cp $R/08b-new-p1-exact.txt.stdout $R/off-new-08b.txt.stdout; cp $R/08b-new-p1-exact.txt.stderr $R/off-new-08b.txt.stderr
same() { cmp -s <(grep -v wall_ms $1.stdout) <(grep -v wall_ms $2.stdout) && cmp -s <(grep '^logits' $1.stderr) <(grep '^logits' $2.stderr) && [ -s $2.stdout ]; }
{
n=0; bad=0
for f in $R/{08b,2b,9b,flash}-base-*.txt; do g=${f/-base-/-new-}; n=$((n+1))
  if same $f $g && grep -q '^rc=0' $g; then :; else bad=$((bad+1)); echo "DIFF $(basename $g)"; fi
done
echo "G-REG: $((n-bad))/$n runs byte-identical (stdout ids + every logits line), base 89dba9dde vs candidate (7f84d5a82)"
n=0; bad=0
for f in $R/4b-base-*.txt; do g=${f/-base-/-new-}; n=$((n+1)); same $f $g || { bad=$((bad+1)); echo "DIFF $(basename $g)"; }; done
echo "G-REG4: $((n-bad))/$n identical"
same $R/multi-base-08b.txt $R/multi-new-08b.txt && echo "G-MULTI 0.8B: identical" || echo "G-MULTI 0.8B: DIFF"
same $R/multi-base-s35.txt $R/multi-new-s35.txt && echo "G-MULTI S35 Q4_K_M: identical" || echo "G-MULTI S35 Q4_K_M: DIFF"
same $R/off-new-08b.txt $R/off-gen0-08b.txt && echo "G-OFF 0.8B: identical" || echo "G-OFF 0.8B: DIFF"
} > $S/receipts/GREG-C.txt
touch $S/tmp/regC.done
