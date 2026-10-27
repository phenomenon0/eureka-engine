#!/usr/bin/env bash
# build_mut.sh NAME 'SED-EXPR' : a scratch-only mutant of new-src (the sed applies to tools/payload-moeany.patch)
set -e
S=$HOME/worktrees/.scratch-cuda-moe; n=$1
rm -rf $S/$n-src $S/$n-out; cp -r $S/new-src $S/$n-src; mkdir -p $S/$n-out
sed -i "$2" $S/$n-src/tools/payload-moeany.patch
diff <(cat $S/new-src/tools/payload-moeany.patch) $S/$n-src/tools/payload-moeany.patch > $S/receipts/$n.diff || true
export TMPDIR=$S/tmp PATH=$HOME/.bun/bin:$PATH CC=gcc
cd $S/$n-src && LLB_BEND=$HOME/worktrees/cuda-moe LLB_OUT=$S/$n-out ./build.sh > $S/receipts/build-$n.log 2>&1
grep -A6 "== built" $S/receipts/build-$n.log
