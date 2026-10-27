#!/bin/bash
# run.sh BUILD MODEL SIDE [engine args...]: the engine of out-BUILD (base = the first MoE build a20c8ee2, final = this
# lane's default build) on MODEL (a .gguf path, or models/MODEL.gguf) with the sidecar side/SIDE
cd "${LANE:-$HOME/scratch/metal-catchup}"
B=$PWD/out-$1/sys1mac; m=$2; s=$3; shift 3
G=$m; [ -f "$G" ] || G=$PWD/models/$m.gguf
exec env LLB_GGUF=$G $B --gpu ${GPU:-8GB} -- side/$s "$@"
