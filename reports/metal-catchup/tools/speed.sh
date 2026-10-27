#!/bin/bash
# speed.sh ROUNDS: interleaved rounds (engine then llama.cpp): decode (p2, 128 steps; llama-bench tg128) and prefill
# (pp4096: the 4,096-id prompt; llama-bench pp4096); streamed decode (report); base decode (before)
cd "${LANE:-$HOME/scratch/metal-catchup}"; M=Qwen3.5-35B-A3B-Q4_K_S; R=runs/speed; mkdir -p $R; LB=$HOME/llama.cpp/build/bin/llama-bench
dec() { # BUILD ENV.. : mean ms of the decode steps (pos >= prompt length), skipping the first 8
  env ${@:2} ./run.sh $1 $M 35b @runs/p2.ids --steps 128 > $R/d.out 2> $R/d.err
  awk -v n=$(wc -w < runs/p2.ids | tr -d " ") '$1=="step" && $2>=n {k++; if (k>8 && k<=128) {s+=$3; c++}} END {printf "%.3f ms (%d steps)", s/c, c}' $R/d.err; }
pf() { # prefill ms: the chunks + the last prompt id's exact step - the expert pool's load (its mx pool seconds)
  ./run.sh final $M 35b @runs/pp4096.ids --steps 1 > $R/p.out 2> $R/p.err
  awk -v n=$(wc -w < runs/pp4096.ids | tr -d " ") '$1=="chunkf" {c+=$4} $1=="step" && $2==n-1 {s=$3} /^mx pool/ {match($0, /, [0-9.]+ s$/); l=substr($0, RSTART+2, RLENGTH-4)*1000}
    END {printf "%.0f ms raw %.0f (chunks %.0f + last step %.0f - pool %.0f) = %.1f tok/s", c+s-l, c+s, c, s, l, n*1000/(c+s-l)}' $R/p.err; }
for r in $(seq 1 ${1:-3}); do
  echo "round $r engine decode: $(dec final LLB_X=0)"
  echo "round $r llama.cpp tg128: $($LB -m models/$M.gguf -p 0 -n 128 -r 3 2>/dev/null | grep tg128 | awk -F'|' '{print $(NF-1)}')"
  echo "round $r engine prefill pp4096: $(pf)"
  echo "round $r llama.cpp pp4096: $($LB -m models/$M.gguf -p 4096 -n 0 -r 3 2>/dev/null | grep pp4096 | awk -F'|' '{print $(NF-1)}')"
  echo "round $r engine streamed decode: $(dec final LLB_MX_STREAM=1)"
  echo "round $r base decode: $(dec base LLB_X=0)"
  echo "round $r base streamed decode: $(dec base LLB_MX_STREAM=1)"
done
echo done > $R/DONE
