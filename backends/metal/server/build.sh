#!/usr/bin/env bash
# METAL-LEAD: build the Metal engine provider for the desk supervisor (BEND_PROVIDER=metal) into desktop/engine/metal/bin:
#   llb-serve   OpenAI-compatible HTTP front (src/serve.c: per-launch --api-key, SSE, Qwen3.5 chat template, token-id prompts)
#   sys1mac     the Bend-emitted Metal engine + src/dualm.c (headless generate, stream events, stop ids) via build-engine.sh
#   gguf2pack, llb-tok   sidecar makers (manifest/table from the GGUF; tokenizer.bin + chat.flags)
# The engine tree (Bend compiler, patch stack) is the Mac engine line: EUREKA_ENGINE_SRC (required).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${EUREKA_ENGINE_SRC:?set EUREKA_ENGINE_SRC to the Mac engine tree}
BIN=$HERE/bin; mkdir -p "$BIN"
clang -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread "$HERE/src/serve.c" "$HERE/src/json.c" "$HERE/src/speak_route.c" -o "$BIN/llb-serve"
clang -O2 -std=c11 -D_GNU_SOURCE "$SRC/S1D/tools/gguf2pack.c" -o "$BIN/gguf2pack"
clang -O2 "$SRC/S1D/tools/llb/llb-tok.c" -o "$BIN/llb-tok" 2>/dev/null || clang -O2 "$HOME/engine-4b/S1D/tools/llb/llb-tok.c" -o "$BIN/llb-tok"
VISION=1 W="$HERE/work" O="$HERE/work/build" bash "$HERE/build-engine.sh"
cp "$HERE/work/build/sys1mac" "$BIN/sys1mac"
