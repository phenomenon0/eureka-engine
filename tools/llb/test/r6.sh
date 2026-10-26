#!/usr/bin/env bash
# r6.sh LLB [--update]: llb output vs tools/llb/golden/. help: exact. --dry-run: LLB_HOME, model dir and thread
# count normalized. doctor: paths and numbers normalized (the card state changes PASS/WARN on "GPU idle").
# Needs LLB_HOME (prepared for 27b), LLB_MODELS, LLB_ENGINE.
set -u
L=$1 gd=$(dirname "$0")/../golden upd=${2-}; mkdir -p $gd; bad=0 n=0
g() {  # g NAME CMD...
  local name=$1; shift; n=$((n + 1))
  "$@" > /tmp/r6.$$ 2>&1; echo "exit $?" >> /tmp/r6.$$
  if [ "$upd" = --update ]; then cp /tmp/r6.$$ $gd/$name; echo "wrote $name"
  elif cmp -s /tmp/r6.$$ $gd/$name; then echo "PASS $name"
  else echo "FAIL $name"; diff $gd/$name /tmp/r6.$$ | head -20; bad=$((bad + 1)); fi
  rm -f /tmp/r6.$$
}
g help.txt $L help
for c in run chat models prep doctor bench tok pull version; do g help-$c.txt $L help $c; done
g dryrun-27b.txt bash -c "set -o pipefail; $L run 27b 'The capital of France is' -n 16 --dry-run | sed -e \"s#\$LLB_HOME#<LLB_HOME>#g\" -e \"s#\$LLB_MODELS#<MODELS>#g\" -e \"s#\$LLB_ENGINE#<ENGINE>#g\" -e \"s#'--threads' '[0-9]*'#'--threads' '<T>'#\""
g dryrun-27b-fast-spec.txt bash -c "set -o pipefail; $L run 27b 'x' --fast --spec --dry-run -- --foo | sed -e \"s#\$LLB_HOME#<LLB_HOME>#g\" -e \"s#\$LLB_MODELS#<MODELS>#g\" -e \"s#\$LLB_ENGINE#<ENGINE>#g\" -e \"s#'--threads' '[0-9]*'#'--threads' '<T>'#\""
g doctor.txt bash -c "set -o pipefail; NO_COLOR=1 $L doctor | sed -e \"s#\$LLB_HOME#<LLB_HOME>#g\" -e \"s#\$LLB_MODELS#<MODELS>#g\" -e \"s#\$LLB_ENGINE#<ENGINE>#g\" -e 's/[0-9][0-9.]*/N/g'"
[ "$upd" = --update ] || echo "R6 $((n - bad))/$n PASS"
