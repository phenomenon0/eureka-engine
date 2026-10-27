#!/usr/bin/env bash
# prof.sh ENGDIR GGUF SIDE OUTBASE STEPS IDS... : nsys (cuda trace) of one engine run, under the locks and the 14G cap
S=$HOME/worktrees/.scratch-cuda-catchup
eng=$1 g=$2 side=$3 out=$4 steps=$5; shift 5
export TMPDIR=$S/tmp LLB_GGUF=$g
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout 900 \
  nsys profile -t cuda --cuda-graph-trace=node -o $out -f true $eng/llb-engine $side "$@" --steps $steps > $out.stdout 2> $out.stderr
nsys stats -q -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $out $out.nsys-rep > /dev/null 2>&1
