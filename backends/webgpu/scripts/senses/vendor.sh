#!/bin/bash
# vendor.sh: rebuild public/senses/vendor/ (gitignored): transformers.js 4.3.1 + its onnxruntime-web, kokoro-js 1.2.1 +
# the onnxruntime-web build its bundle was made against, and two patches:
#  1. kokoro.web.js: export transformers' env as `tfenv` (so the page can pin models to /models/hf/) and fetch voices from
#     /models/hf/… instead of huggingface.co;
#  2. both bundles: ORT session log severity defaults to 3 (error), so ORT's warning lines don't land as console errors.
set -euo pipefail
V="$(cd "$(dirname "$0")/../.." && pwd)/senses/vendor"; T="$(mktemp -d)"
(cd "$T" && npm init -y >/dev/null && npm i --silent kokoro-js@1.2.1 @huggingface/transformers@4.3.1 && mkdir k && cd k && npm init -y >/dev/null && npm i --silent onnxruntime-web@1.22.0-dev.20250409-89f8206ba4)
rm -rf "$V"; mkdir -p "$V/tjs" "$V/ort" "$V/kokoro"
O="$T/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist"; [ -d "$O" ] || O="$T/node_modules/onnxruntime-web/dist"
cp "$T/node_modules/@huggingface/transformers/dist/transformers.min.js" "$V/tjs/"
cp "$O/ort.webgpu.min.mjs" "$O"/*.wasm "$O"/ort-wasm-simd-threaded*.mjs "$V/ort/"
cp "$T/k/node_modules/onnxruntime-web/dist"/ort-wasm-simd-threaded{,.jsep}.{mjs,wasm} "$V/kokoro/"
sed -e 's/export{Ef as KokoroTTS,kf as TextSplitterStream,Mf as env};$/export{Ef as KokoroTTS,kf as TextSplitterStream,Mf as env,Wg as tfenv};/' \
    -e 's|https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/|/models/hf/onnx-community/Kokoro-82M-v1.0-ONNX/voices/|' \
    "$T/node_modules/kokoro-js/dist/kokoro.web.js" > "$V/kokoro/kokoro.web.js"
for f in "$V/kokoro/kokoro.web.js" "$V/tjs/transformers.min.js"; do sed -i '' -e 's/logSeverityLevel??2/logSeverityLevel??3/g' -e 's/logSeverityLevel=2;/logSeverityLevel=3;/g' "$f"; done
grep -q 'Wg as tfenv' "$V/kokoro/kokoro.web.js" && grep -q '/models/hf/onnx-community/Kokoro' "$V/kokoro/kokoro.web.js" && echo "vendor.sh: ok → $V"
