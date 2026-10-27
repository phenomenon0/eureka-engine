#!/usr/bin/env bash
# gates.sh : G-BITS (base B vs candidate C, device) and G-MODE (candidate host / stream / forced-small auto) on the 8 gate files;
# each file staged in /dev/shm while its arms run (a page-cacheable copy; same bytes, sha256 checked), then removed
S=$HOME/worktrees/.scratch-cuda-moe-speed; M=${BIG:?a scratch dir for the big files}; T=/dev/shm/cuda-moe-speed/gate
mkdir -p $T
for t in s35-q4km s35-q8 s60-q8 s60-q4km s128f-q8 s128f-q4km s512-q8 s512-q5km; do
  [ -f $S/receipts/gate/$t/BITS-MODE.txt ] && continue
  cp $M/$t.gguf $T/$t.gguf; sha256sum $T/$t.gguf > $S/receipts/gate/$t/STAGED-SHA256.txt
  LLB_MOE_EXPERTS=device $S/scripts/arms.sh $t $T/$t.gguf base-out B
  LLB_MOE_EXPERTS=device $S/scripts/arms.sh $t $T/$t.gguf new-out C
  LLB_MOE_EXPERTS=host $S/scripts/arms.sh $t $T/$t.gguf new-out Chost
  LLB_MOE_EXPERTS=stream $S/scripts/arms.sh $t $T/$t.gguf new-out Cstream
  LLB_MOE_VRAM=1000000000 LLB_MOE_RAM=1000000000 $S/scripts/arms.sh $t $T/$t.gguf new-out Csmall
  { $S/scripts/cmpc.sh $t B C; for m in Chost Cstream Csmall; do $S/scripts/cmpc.sh $t C $m; done; } > $S/receipts/gate/$t/BITS-MODE.txt
  rm -f $T/$t.gguf
done
touch $S/tmp/gates.done
