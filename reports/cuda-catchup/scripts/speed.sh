#!/usr/bin/env bash
# speed.sh ROUND : one interleaved round of the prereg's speed arms (decode: P1 x 128 steps, median stepus pos >= 12;
# prefill: amendment 1, P512 x 4 copies, median pfus; llama.cpp: llama-bench 9d57ce45 tg128 -r 5 / pp512 -r 3)
S=$HOME/worktrees/.scratch-cuda-catchup; M=${BIG:?}; r=$1; o=$S/receipts/speed/r$r; mkdir -p $o
P512=$(cat $HOME/worktrees/cuda-catchup/reports/cuda-catchup/scripts/p512.txt)
declare -A G=( [08b]=$HOME/.local/share/models/Qwen3.5-0.8B-Q4_K_M.gguf [2b]=$HOME/scratch/android-perf-models/dl-2b/Qwen3.5-2B-Q4_K_M.gguf [4b]=$HOME/lanes/echo-4b/models/Qwen3.5-4B-Q4_K_M.gguf [9b]=$HOME/.local/share/models/Qwen3.5-9B-Q4_K_M.gguf )
dec() {  # TAG ENG GGUF SIDE (env passes through)
  case $4 in *08b|*2b|*4b|*9b) p="760 6511 314 9338 369";; *) p="760 511 314 938 369";; esac
  LLB_STEP_US=1 $S/scripts/run.sh $S/$2 $3 $4 $o/$1-dec.txt 128 $p
}
pf() {  # TAG ENG GGUF SIDE
  LLB_STEP_US=1 $S/scripts/run.sh $S/$2 $3 $4 $o/$1-pf.txt 1 $P512 , $P512 , $P512 , $P512
}
export TMPDIR=$S/tmp
for m in 08b 2b 4b 9b; do
  dec $m-base base-out ${G[$m]} $S/side/$m
  dec $m-new new-out ${G[$m]} $S/side/$m
  NOPF=1 $S/scripts/lcpp.sh ${G[$m]} $o/$m-llama > /dev/null
done
mkdir -p /dev/shm/cuda-catchup/speed
for f in s35-q4km s128f-q4km s35x8-q4km; do
  cp $M/$f.gguf /dev/shm/cuda-catchup/speed/$f.gguf; g=/dev/shm/cuda-catchup/speed/$f.gguf; sd=$S/side/$f
  LLB_MOE_EXPERTS=device dec $f-base-device base-out $g $sd
  LLB_MOE_EXPERTS=device dec $f-new-device new-out $g $sd
  LLB_MOE_EXPERTS=device LLB_MOE_DP4A=1 dec $f-new-dp4a new-out $g $sd
  if [ $f = s35-q4km ]; then
    LLB_MOE_EXPERTS=host dec $f-base-host base-out $g $sd
    LLB_MOE_EXPERTS=host dec $f-new-host new-out $g $sd
    LLB_MOE_EXPERTS=stream dec $f-base-stream base-out $g $sd
    LLB_MOE_EXPERTS=stream dec $f-new-stream new-out $g $sd
  fi
  if [ $f != s35x8-q4km ]; then
    LLB_MOE_EXPERTS=device pf $f-base-exact base-out $g $sd
    LLB_MOE_EXPERTS=device pf $f-new-exact new-out $g $sd
    LLB_MOE_EXPERTS=device LLB_PREFILL=fast pf $f-new-fast new-out $g $sd
    $S/scripts/lcpp.sh $g $o/$f-llama > /dev/null
  else
    NOPF=1 $S/scripts/lcpp.sh $g $o/$f-llama > /dev/null
  fi
  rm -f $g
done
echo "round $r done $(date -Is)"
