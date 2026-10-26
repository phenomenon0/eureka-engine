#!/usr/bin/env bash
# summary.sh TOK DIR: llb-tok run's summary line against a stub engine (no GPU). Under LLB_SPEC, step 0 also loads the
# NextN drafter (unify1 receipts: step 0 = 6-11 s with spec, ~10 ms without), so its time must count as load, not prompt.
set -u
tok=$1 dir=$2 e=$(mktemp -p "${TMPDIR:-/tmp}"); bad=0
printf '#!/bin/sh\nsleep 0.5; sleep 2; echo "step 0 2000" >&2\nfor p in 1 2 3 4; do echo "step $p 10" >&2; done\nfor p in 5 6 7; do echo "step $p 13" >&2; echo 760; done\necho "wall_ms 1"\n' > $e; chmod +x $e
s=$($tok run $dir -m t -P 5 -- env LLB_SPEC=2 $e 2>&1 >/dev/null | tail -1); echo "spec: $s"
case $s in *"prompt 5 tok, 125.0 tok/s"*"load 2.5 s"*) echo PASS spec ;; *) echo FAIL spec; bad=1 ;; esac
s=$($tok run $dir -m t -P 5 -- $e 2>&1 >/dev/null | tail -1); echo "plain: $s"
case $s in *"prompt 5 tok, 2.5 tok/s"*"load 0.5 s"*) echo PASS plain ;; *) echo FAIL plain; bad=1 ;; esac
rm -f $e; exit $bad
