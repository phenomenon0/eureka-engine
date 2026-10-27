#!/usr/bin/env bash
# mkvar.sh NAME PYFILE : a dev / mutant variant: the current b/ comp.ts edited by PYFILE (s -> s), its moeany patch at $S/var/NAME.patch
set -e
S=$HOME/worktrees/.scratch-cuda-moe-speed; mkdir -p $S/var/$1/b/bend2
cp $S/patch/b/bend2/comp.ts $S/var/$1/b/bend2/comp.ts
python3 - $S/var/$1/b/bend2/comp.ts $2 <<'PY'
import sys
p, f = sys.argv[1:3]; s = open(p).read(); exec(open(f).read()); open(p, "w").write(s)
PY
cd $S; diff -u patch/a/bend2/comp.ts var/$1/b/bend2/comp.ts | sed -e '1s|^--- .*|--- a/bend2/comp.ts|' -e '2s|^+++ .*|+++ b/bend2/comp.ts|' > $S/var/$1.patch || true
