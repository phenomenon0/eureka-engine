#!/usr/bin/env bash
# mkpatch.sh : b/ = pre-moeany comp.ts + the old moeany patch + edit_speed.py; writes the worktree's tools/payload-moeany.patch
set -e
S=$HOME/worktrees/.scratch-cuda-moe-speed; W=$HOME/worktrees/cuda-moe-speed
cd $S/patch; cp a/bend2/comp.ts b/bend2/comp.ts
git -C $W show 4c7c24c10:tools/payload-moeany.patch | patch --batch --silent --fuzz=0 -d b -p1
python3 $S/scripts/edit_speed.py b/bend2/comp.ts
diff -u a/bend2/comp.ts b/bend2/comp.ts | sed -e '1s|^--- a/bend2/comp.ts.*|--- a/bend2/comp.ts|' -e '2s|^+++ b/bend2/comp.ts.*|+++ b/bend2/comp.ts|' > $W/tools/payload-moeany.patch || true
wc -l $W/tools/payload-moeany.patch
