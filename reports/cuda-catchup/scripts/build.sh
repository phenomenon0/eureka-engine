#!/usr/bin/env bash
# build.sh TAG [REV] : REV given -> git archive REV; else the working tree (tracked + new) of ~/worktrees/cuda-catchup. Builds in scratch.
set -e
S=$HOME/worktrees/.scratch-cuda-catchup; W=$HOME/worktrees/cuda-catchup; tag=$1
rm -rf $S/$tag-src $S/$tag-out/bend-joint; mkdir -p $S/$tag-src $S/$tag-out
if [ -n "$2" ]; then git -C $W archive $2 | tar -xf - -C $S/$tag-src
else (cd $W && git ls-files -co --exclude-standard | tar -cf - -T -) | tar -xf - -C $S/$tag-src; fi
export TMPDIR=$S/tmp PATH=$HOME/.bun/bin:$PATH CC=${CC:-gcc}
cd $S/$tag-src && LLB_BEND=$W LLB_OUT=$S/$tag-out ./build.sh
