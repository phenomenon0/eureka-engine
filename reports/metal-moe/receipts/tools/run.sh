#!/bin/bash
# run.sh VARIANT(moe|any) MODEL(models/X.gguf basename w/o .gguf, or a path) SIDE [engine args...]
cd ~/scratch/metal-moe
B=$HOME/scratch/metal-moe/out-$1/sys1mac; m=$2; s=$3; shift 3
G=$m; [ -f "$G" ] || G=$HOME/scratch/metal-moe/models/$m.gguf
exec env LLB_GGUF=$G $B --gpu ${GPU:-8GB} -- side/$s "$@"
