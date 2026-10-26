#!/usr/bin/env bash
# r4.sh LLB OUT: each error path must print ONE stderr line and exit nonzero. W = scratch dir.
# Empty stdin / -f cases run with a busy stub GPU: the prompt error must come before the GPU check.
# Needs LLB_MODELS (with the 27b) and LLB_ENGINE set by the caller; GPU cases use a stub nvidia-smi on PATH (no engine is started).
set -u
L=$1 out=$2 W=${W:-${TMPDIR:-/tmp}/llb-r4}; rm -rf $W; mkdir -p $W/ro $W/fake $W/empty; : > $out; bad=0
echo 'not a model' > $W/x.gguf; : > $W/x.empty; chmod 500 $W/ro
printf '#!/bin/sh\ncase "$*" in *compute-apps*) echo 4242 ;; *) echo "$FAKE_MEM" ;; esac\n' > $W/fake/nvidia-smi; chmod +x $W/fake/nvidia-smi
t() {  # t NAME CMD...
  local n=$1; shift
  "$@" > $W/o 2> $W/e < /dev/null; local rc=$? lines=$(wc -l < $W/e)
  if [ $rc -ne 0 ] && [ $lines -eq 1 ]; then r=PASS; else r=FAIL; bad=$((bad + 1)); fi
  printf '%s %-14s rc=%s lines=%s | %s\n' $r "$n" $rc $lines "$(head -c 300 $W/e | tr '\n' '|')" | tee -a $out
}
H=$W/h; env LLB_HOME=$H $L prep 27b 2> /dev/null || echo "setup: prep failed"
t missing-model   env LLB_HOME=$W/h0 LLB_MODELS=$W/empty HOME=$W/empty $L run 9b hi
t missing-file    env LLB_HOME=$H $L run $W/nope.gguf hi
t not-a-gguf      env LLB_HOME=$H $L run $W/x.gguf hi
t unknown-alias   env LLB_HOME=$H $L run 70b hi
t unwritable-home env LLB_HOME=$W/ro/h FAKE_MEM="24576, 100" PATH=$W/fake:$PATH $L run 27b hi
t gpu-too-small   env LLB_HOME=$H FAKE_MEM="8192, 100" PATH=$W/fake:$PATH $L run 27b hi
t gpu-busy        env LLB_HOME=$H FAKE_MEM="24576, 19000" PATH=$W/fake:$PATH $L run 27b hi
t engine-missing  env LLB_HOME=$H FAKE_MEM="24576, 100" PATH=$W/fake:$PATH LLB_ENGINE=$W/no-engine LLB_OUT=$W/empty $L run 27b hi
t empty-prompt    env LLB_HOME=$H $L run 27b ""
t empty-stdin     env LLB_HOME=$H FAKE_MEM="24576, 19000" PATH=$W/fake:$PATH $L run 27b
t empty-file      env LLB_HOME=$H FAKE_MEM="24576, 19000" PATH=$W/fake:$PATH $L run 27b -f $W/x.empty
t unreadable-file env LLB_HOME=$H FAKE_MEM="24576, 19000" PATH=$W/fake:$PATH $L run 27b -f $W/nope.txt
t bad-n           env LLB_HOME=$H $L run 27b hi -n x
t unknown-option  env LLB_HOME=$H $L run 27b hi --temp 0.7
t unknown-command env LLB_HOME=$H $L frob
chmod 700 $W/ro
echo "R4 $((15 - bad))/15 PASS" | tee -a $out
