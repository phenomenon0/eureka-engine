#!/usr/bin/env bash
# gates.sh : per gate file (staged in /dev/shm, sha256 checked, removed after): B (base) and C (candidate) device;
export GD=gate2
# candidate host / stream / forced-small; D (LLB_MOE_DP4A=1) greedy + TF; PF (LLB_PREFILL=fast) prefix probes;
# on s35-q4km also G (LLB_GEN=0) and the exact-tier probe E (the PF instrument without the fast tier)
S=$HOME/worktrees/.scratch-cuda-catchup; M=${BIG:?}; T=/dev/shm/cuda-catchup/gate2; mkdir -p $T
for t in s35-q4km s35-q8 s60-q8 s60-q4km s128f-q8 s128f-q4km s512-q8 s512-q5km; do
  [ -f $S/receipts/gate2/$t/DONE ] && continue
  cp $M/$t.gguf $T/$t.gguf; sha256sum $T/$t.gguf | sed "s|$T|/dev/shm|" > $S/receipts/gate2/$t/STAGED-SHA256.txt
  LLB_MOE_EXPERTS=device $S/scripts/arms.sh $t $T/$t.gguf base-out B
  LLB_MOE_EXPERTS=device $S/scripts/arms.sh $t $T/$t.gguf new-out C
  LLB_MOE_EXPERTS=host $S/scripts/arms.sh $t $T/$t.gguf new-out Chost
  LLB_MOE_EXPERTS=stream $S/scripts/arms.sh $t $T/$t.gguf new-out Cstream
  LLB_MOE_VRAM=1000000000 LLB_MOE_RAM=1000000000 $S/scripts/arms.sh $t $T/$t.gguf new-out Csmall
  LLB_MOE_EXPERTS=device LLB_MOE_DP4A=1 $S/scripts/arms.sh $t $T/$t.gguf new-out D
  LLB_MOE_EXPERTS=device LLB_PREFILL=fast $S/scripts/pfarm.sh $t $T/$t.gguf new-out PF
  if [ $t = s35-q4km ]; then
    LLB_MOE_EXPERTS=device LLB_GEN=0 $S/scripts/arms.sh $t $T/$t.gguf new-out G
    LLB_MOE_EXPERTS=device $S/scripts/pfarm.sh $t $T/$t.gguf new-out E
  fi
  { $S/scripts/cmpc.sh $t B C; for m in Chost Cstream Csmall; do $S/scripts/cmpc.sh $t C $m; done; $S/scripts/cmpc.sh $t C D; } > $S/receipts/gate2/$t/BITS-MODE.txt
  rm -f $T/$t.gguf; touch $S/receipts/gate2/$t/DONE
done
touch $S/tmp/gates2.done
