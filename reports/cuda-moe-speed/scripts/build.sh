#!/usr/bin/env bash
# build.sh TAG [REV] : REV given -> git archive REV; else the working tree (tracked + new) of ~/worktrees/cuda-moe-speed. Builds in scratch.
set -e
S=$HOME/worktrees/.scratch-cuda-moe-speed; W=$HOME/worktrees/cuda-moe-speed; tag=$1
rm -rf $S/$tag-src $S/$tag-out/bend-joint; mkdir -p $S/$tag-src $S/$tag-out
if [ -n "$2" ]; then git -C $W archive $2 | tar -xf - -C $S/$tag-src
else (cd $W && git ls-files -co --exclude-standard | tar -cf - -T -) | tar -xf - -C $S/$tag-src; fi
[ -n "$HOSTPATCH" ] && patch -d $S/$tag-src -p1 < $HOSTPATCH
[ -n "$PATCH_OVERRIDE" ] && cp $PATCH_OVERRIDE $S/$tag-src/tools/payload-moeany.patch
export TMPDIR=$S/tmp PATH=$HOME/.bun/bin:$PATH CC=${CC:-gcc}
cd $S/$tag-src && LLB_BEND=$W LLB_OUT=$S/$tag-out ./build.sh
