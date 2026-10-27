#!/usr/bin/env bash
# build.sh SRCREV-or-WORKTREE OUTNAME : copy the worktree's files (tracked + new) to scratch and build there
# usage: build.sh new   (from ~/worktrees/cuda-moe working tree)   | build.sh base (git archive 61a673aa3)
set -e
S=$HOME/worktrees/.scratch-cuda-moe; W=$HOME/worktrees/cuda-moe; tag=$1
rm -rf $S/$tag-src $S/$tag-out/bend-joint; mkdir -p $S/$tag-src $S/$tag-out
if [ "$tag" = base ]; then git -C $W archive 61a673aa3 | tar -xf - -C $S/$tag-src
else (cd $W && git ls-files -co --exclude-standard | tar -cf - -T -) | tar -xf - -C $S/$tag-src; fi
export TMPDIR=$S/tmp PATH=$HOME/.bun/bin:$PATH CC=${CC:-gcc}
cd $S/$tag-src && LLB_BEND=$W LLB_OUT=$S/$tag-out ./build.sh
