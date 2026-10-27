#!/usr/bin/env bash
# mkpatch.sh : b/ = the fully patched comp.ts (a/, every patch before payload-catchup) + edit_catchup.py;
# writes the worktree's tools/payload-catchup.patch
set -e
S=$HOME/worktrees/.scratch-cuda-catchup; W=$HOME/worktrees/cuda-catchup
cd $S/patch; cp a/bend2/comp.ts b/bend2/comp.ts
python3 $W/reports/cuda-catchup/scripts/edit_catchup.py b/bend2/comp.ts
diff -u a/bend2/comp.ts b/bend2/comp.ts | sed -e '1s|^--- a/bend2/comp.ts.*|--- a/bend2/comp.ts|' -e '2s|^+++ b/bend2/comp.ts.*|+++ b/bend2/comp.ts|' > $W/tools/payload-catchup.patch || true
wc -l $W/tools/payload-catchup.patch
