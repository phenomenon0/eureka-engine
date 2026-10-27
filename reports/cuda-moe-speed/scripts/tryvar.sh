#!/usr/bin/env bash
# tryvar.sh NAME PYFILE [HOSTSED] : build a dev variant (kernel edit PYFILE; optional sed on llb_host.c) and nsys its MoE kernels
S=$HOME/worktrees/.scratch-cuda-moe-speed; v=$1
$S/scripts/mkvar.sh $v $2 || exit 1
PATCH_OVERRIDE=$S/var/$v.patch $S/scripts/build.sh $v > $S/var/build-$v.log 2>&1 || { tail -5 $S/var/build-$v.log; exit 1; }
[ -n "$3" ] && { sed -i "$3" $S/$v-src/host/llb_host.c; cd $S/$v-src && LLB_BEND=$HOME/worktrees/cuda-moe-speed LLB_OUT=$S/$v-out TMPDIR=$S/tmp PATH=$HOME/.bun/bin:$PATH CC=gcc ./build.sh > $S/var/build-$v.log 2>&1; }
$S/scripts/prof.sh $S/$v-out $BIG/s35-q4km.gguf $S/side/s35-q4km $S/receipts/prof/var-$v 64 760 511 314 938 369
echo "$v: $(python3 $S/scripts/steps.py $S/receipts/prof/var-$v.sqlite 10 | grep -E 'moe' | awk '{printf "%s %.1f  ", $1, $2/4}')"
