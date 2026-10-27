#!/bin/bash
# g89.sh MODEL SIDE IDS N: G8 (forced split / streamed = resident, the final build, default prefill: the decode
# kernels and, for a prompt of >= 13 ids, the MoE fast chunk) and G9 (the final build with the v1 kernels, LLB_MX_V1=1,
# = the base build; exact prefill, resident and streamed): ids and logits-line hashes
cd "${LANE:-$HOME/scratch/metal-catchup}"; M=$1; S=$2; I=$3; N=$4; mkdir -p runs/g89
h() { env $1 ./run.sh $2 $M $S @$I --steps $N ${@:3} > runs/g89/r.out 2> runs/g89/r.err
  echo "ids $(grep -v wall runs/g89/r.out | tr "\n" " " | shasum | cut -c1-16) logits $(grep "^logits" runs/g89/r.err | shasum | cut -c1-16)"; }
r=$(h LLB_X=0 final); echo "G8 $M $(basename $I) resident:          $r"
for mode in LLB_MX_SPLIT=3 LLB_MX_STREAM=1; do x=$(h $mode final); echo "G8 $M $(basename $I) $mode: $x same=$([ "$x" = "$r" ] && echo YES || echo NO)"; done
for st in LLB_X=0 LLB_MX_STREAM=1; do
  a=$(h $st base --exactpf); b=$(h "$st LLB_MX_V1=1" final --exactpf)
  echo "G9 $M $(basename $I) $st: base $a | final+V1 $b same=$([ "$a" = "$b" ] && echo YES || echo NO)"
done
