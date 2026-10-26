#!/usr/bin/env bash
# tools/sys1dual/build-serve.sh: W1-SERVER. Two artifacts, no Bend re-emit, no NVRTC re-JIT:
#   1. llb-serve  = serve.c + clef3/json.c (the HTTP front; plain C, pthreads), + mm.c (G5 multimodal content),
#      and for G16's --structured: serve_structured.c + host/llb_structured.c (G1's grammar + tool calls);
#      G41's --tts runs tools/tts/kokoro_tts.py, copied beside it; /v1/see and /v1/hear are
#      see_route.c / hear_route.c, #include'd by serve.c (tools/see/see_worker.py is copied beside it)
#   2. llb-engine = RELINK of an already-emitted llb-engine.c (LLB_ENGINE_C) with
#      the request objects, dual.c rebuilt (LLB_SYS1_STREAM events + "special").
#      llb-engine.c is unchanged, so its .gpu program (LLB_ENGINE_GPU) is reused as is.
# Env: OUT (output dir), LLB_ENGINE_C, LLB_ENGINE_GPU, CC (clang-21: llb-engine.c uses #embed).
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${OUT:?set OUT}" "${LLB_ENGINE_C:?emitted llb-engine.c}" "${LLB_ENGINE_GPU:?its .gpu program}"
CC=${CC:-clang}
o=$OUT/obj
mkdir -p "$o"
f=(-D_POSIX_C_SOURCE=200809L -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wno-unused-function)
git=$(git -C "$root" rev-parse --short=8 HEAD 2>/dev/null || echo unknown) # /version + --version
"$CC" -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread -DLLB_GIT="\"$git\"" "$root/tools/sys1dual/serve.c" \
  "$root/tools/sys1dual/serve_structured.c" "$root/host/llb_structured.c" "$root/tools/sys1dual/mm.c" "$root/tools/clef3/json.c" -lm -o "$OUT/llb-serve"
# G41: llb-serve --tts runs the Kokoro worker that sits beside it (tools/tts/README.md)
cp "$root/tools/tts/kokoro_tts.py" "$root/tools/tts/pins.json" "$OUT/"
# POST /v1/see: the detector child (LLB_SEE_SCRIPT=$OUT/see_worker.py; tools/see/README.md)
cp "$root/tools/see/see_worker.py" "$root/tools/see/coco80.json" "$OUT/"
for s in sys1dual/dual.c metrics/metrics.c metrics/tokstream.c glue/glue_out.c clef3/head.c clef3/weights.c clef3/spans.c clef3/json.c qwen35_tokenizer.c; do
  "$CC" "${f[@]}" -c "$root/tools/$s" -o "$o/$(basename "$s" .c).o.new" && mv "$o/$(basename "$s" .c).o.new" "$o/$(basename "$s" .c).o"
done
cp "$LLB_ENGINE_C" "$OUT/llb-engine.c"
cp "$LLB_ENGINE_GPU" "$OUT/llb-engine.gpu"
CC=$CC LLB_CC_EXTRA="$(echo "$o"/*.o) -lopenblas" bash "$root/tools/remap_cc.sh" "$OUT/llb-engine.c" "$OUT/llb-engine.new"
mv "$OUT/llb-engine.new" "$OUT/llb-engine"
sha256sum "$OUT/llb-serve" "$OUT/llb-engine" "$OUT/llb-engine.c" "$OUT/llb-engine.gpu" "$o"/*.o
