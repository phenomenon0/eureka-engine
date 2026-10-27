#!/usr/bin/env bash
# speed.sh ROUND : one interleaved round of the prereg's speed arms (decode P1 x 128 steps; prefill p512),
# order: base device, candidate device / host / stream, base host / stream, llama.cpp CUDA, llama.cpp -ot exps=CPU
S=$HOME/worktrees/.scratch-cuda-moe-speed; M=${BIG:?a scratch dir for the big files}; r=$1; o=$S/receipts/speed/r$r; mkdir -p $o
P512=$(cat $HOME/worktrees/cuda-moe-speed/reports/cuda-moe-speed/scripts/p512.txt)
LB=$HOME/llama.cpp/build/bin/llama-bench
eng() {  # TAG ENG FILE MODE (the file from /dev/shm when copied there: the stream residency needs a page-cacheable file)
  g=$M/$3.gguf; [ -f /dev/shm/cuda-moe-speed/$3.gguf ] && g=/dev/shm/cuda-moe-speed/$3.gguf
  LLB_STEP_US=1 LLB_MOE_EXPERTS=$4 $S/scripts/run.sh $S/$2 $g $S/side/$3 $o/$1-dec.txt 128 760 511 314 938 369
  LLB_STEP_US=1 LLB_MOE_EXPERTS=$4 $S/scripts/run.sh $S/$2 $g $S/side/$3 $o/$1-pf.txt 1 $P512
}
lcpp() {  # TAG FILE EXTRA...
  t=$1 f=$2; shift 2
  g=$M/$f.gguf; [ -f /dev/shm/cuda-moe-speed/$f.gguf ] && g=/dev/shm/cuda-moe-speed/$f.gguf
  systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout 1800 \
    $LB -m $g -ngl 99 "$@" -p 0 -n 128 -r 5 -o json > $o/$t-dec.json 2> $o/$t-dec.err
  systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0 flock -w 3600 /tmp/llb-gpu-bench.lock flock -w 3600 /tmp/bend-bench.lock timeout 1800 \
    $LB -m $g -ngl 99 "$@" -p 512 -n 0 -r 3 -o json > $o/$t-pf.json 2> $o/$t-pf.err
}
export TMPDIR=$S/tmp
for f in s35-q4km s128f-q4km; do
  eng $f-base-device baset-out $f device
  eng $f-new-device new-out $f device
  eng $f-new-host new-out $f host
  eng $f-new-stream new-out $f stream
  eng $f-base-host baset-out $f host
  eng $f-base-stream baset-out $f stream
  lcpp $f-llama-cuda $f
  lcpp $f-llama-cpumoe $f -ot exps=CPU
done
eng s35x8-q4km-base-device baset-out s35x8-q4km device
eng s35x8-q4km-new-device new-out s35x8-q4km device
lcpp s35x8-q4km-llama-cuda s35x8-q4km
echo "round $r done $(date -Is)"
