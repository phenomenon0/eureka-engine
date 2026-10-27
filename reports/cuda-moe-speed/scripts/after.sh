#!/usr/bin/env bash
# after.sh : after gates.sh: controls, G-REG (GPU) while the S128F Q4_K_M F32 copy is made (CPU), then its X arm,
# then the speed rounds, then the before / after profiles
S=$HOME/worktrees/.scratch-cuda-moe-speed; M=${BIG:?a scratch dir for the big files}
mkdir -p /dev/shm/cuda-moe-speed
( LLB_MOE_EXPERTS=device $S/scripts/arms.sh s35-q4km $M/s35-q4km.gguf mutrev-out Mrev
  LLB_MOE_EXPERTS=device $S/scripts/arms.sh s512-q8 $M/s512-q8.gguf mutlow-out Mlow
  touch $S/tmp/ctl.done
  $S/scripts/reg.sh > $S/receipts/GREG.txt 2>&1; touch $S/tmp/reg.done ) &
$S/scripts/xarm.sh s128f-q4km $M/s128f-q4km.gguf
touch $S/tmp/x.done
wait
$S/scripts/rounds.sh
$S/scripts/finalprof.sh
