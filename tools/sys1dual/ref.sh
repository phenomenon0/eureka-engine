#!/usr/bin/env bash
# G3 references from the retained unmodified CLEF-3 engine (a scratch copy of
# llama-bend-clef3/build/clef3/{llb-engine,.gpu}), one fresh process per prompt.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tools/sys1dual/env.sh"
unset LLB_PREFILL LLB_PF_SSMQ8 LLB_SPEC LLB_MTP LLB_DUMP LLB_DUMP_POS LLB_DUMP_HIDDEN LLB_DUMP_HIDDEN_POS LLB_DUMP_HIDDEN_TOKENS LLB_Q4_DP4A LLB_Q4GLUE LLB_WARP_GRID LLB_SYS1
for p in "p1-ref:The capital of France is" "p2-ref:Water boils at a temperature of"; do
  d=$root/receipts/sys1dual/${p%%:*}; mkdir -p "$d"
  ids=$("$W/ref/llb-tok" enc "$W/sidecar" "${p#*:}")
  echo "$ids" > "$d/prompt_ids.txt"
  sha256sum "$W/ref/llb-engine" "$W/ref/llb-engine.gpu" > "$d/engine.sha256"
  # shellcheck disable=SC2086
  LLB_GGUF=$SYS1_GGUF "$W/ref/llb-engine" --threads 16 -- "$W/sidecar" $ids --steps 16 > "$d/stdout.txt" 2> "$d/stderr.txt"
  grep -E '^[0-9]+$' "$d/stdout.txt" | tr '\n' ' ' | sed 's/ $/\n/' > "$d/ids.txt"
  cat "$d/ids.txt"
done
