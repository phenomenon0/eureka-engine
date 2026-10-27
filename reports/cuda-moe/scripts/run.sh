#!/usr/bin/env bash
# run.sh ENGDIR GGUF SIDE OUT STEPS IDS... (env passes through): one engine run under the house GPU locks and a 14G cap
S=$HOME/worktrees/.scratch-cuda-moe
eng=$1 g=$2 side=$3 out=$4 steps=$5; shift 5
export TMPDIR=$S/tmp LLB_GGUF=$g
{ echo "# eng=$eng gguf=$g side=$side steps=$steps env: $(env | grep '^LLB_' | grep -v '^LLB_GGUF=' | tr '\n' ' ')"; date -Is; } > $out
systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-14G} -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout ${RUN_TIMEOUT:-900} $eng/llb-engine $side "$@" --steps $steps > $out.stdout 2> $out.stderr
echo "rc=$?" >> $out
cat $out.stdout >> $out
grep -v '^\(step\|op\|logits\|batch\) ' $out.stderr | tail -20 >> $out
