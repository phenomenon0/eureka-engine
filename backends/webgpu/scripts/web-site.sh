#!/bin/zsh
# web-site.sh (APP-FOLD): assemble the served web site from THIS tree's dist: one build serves the desk (≥900 px) and the
# phone (<720 px) grades. dist already carries the app, engine/, emit/, trunk/ and senses/ (public/); this adds what git
# doesn't carry: senses/vendor/ (scripts/senses/vendor.sh's output, + the Kokoro voice-URL patch) and models/ (symlinks).
# Usage: scripts/web-site.sh <out-dir> <senses-vendor-dir> <models-dir> [hf]
#   hf  → a box build: VITE_MODEL_HOST=hf + senses/hosts.js MODELS="hf"; only the tokenizer is copied into models/.
set -euo pipefail
OUT="${1:?out dir}" VENDOR="${2:?senses vendor dir}" MODELS="${3:?models dir}" MODE="${4:-local}"
cd "$(dirname "$0")/.."
if [ "$MODE" = hf ]; then VITE_BEND_ENGINE=tab VITE_MODEL_HOST=hf npm run build; else VITE_BEND_ENGINE=tab npm run build; fi
rm -rf "$OUT" && mkdir -p "$OUT/models"
rsync -a dist/ "$OUT/"
rsync -a "$VENDOR/" "$OUT/senses/vendor/"
if [ "$MODE" = hf ]; then
  cp -L "$MODELS/qwen35-tokenizer.bin" "$OUT/models/"
  sed -i '' 's/^export const MODELS = "local";/export const MODELS = "hf";/' "$OUT/senses/hosts.js"
  grep -q '^export const MODELS = "hf";' "$OUT/senses/hosts.js"
  sed -i '' 's# · <a href="../engine/smoke.html?kind=gguf">smoke</a>##' "$OUT/senses/index.html"
else
  for m in "$MODELS"/*(N); do ln -s "$(readlink "$m" 2>/dev/null || echo "$m")" "$OUT/models/${m:t}"; done
fi
echo "web-site: $OUT ($MODE) $(du -sh "$OUT" | cut -f1)"
