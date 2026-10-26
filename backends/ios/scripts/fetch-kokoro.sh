#!/bin/bash
# KOKOROWIRE: the phone's house voice (Kokoro-82M v1.0, Apache-2.0), every input pinned by SHA-256. Idempotent.
# Builds the pack KokoroVoice.swift reads: kokoro.json + model .onnx + lexicon-us.tsv (misaki 0.7.4, no espeak-ng) +
# voices/<v>.f32 (af_heart af_bella af_nicole am_michael bf_emma bm_george; 512 KB each).
#
#   fetch-kokoro.sh [DEST]   DEST defaults to <app>/ios/build/kokoro-kit (IOSHIP fold: see below; the kit's own default was the bundle)
#   KOKORO_MODEL=fp16w (default): fp32 graph with fp16-stored weights, 156 MB; sim: ~0.66 s first clause at 4 threads.
#                                 Built here from the pinned fp32 export (needs a throwaway venv with onnx + numpy).
#   KOKORO_MODEL=int8:            the pinned int8 export as-is, 88 MB, no conversion; sim: ~1.1 s first clause, ~180 MB
#                                 less memory.
# NOT the onnx-community fp16 graph: it overflows to NaN/silence on ~13% of lines (kokorowire kit/eval).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS="$HERE/kokoro"; [ -d "$TOOLS" ] || TOOLS="$HERE/tools"
# IOSHIP fold (foldin-r4): VOICESHOP owns ios/Vendor/voices/kokoro (scripts/fetch-voice.sh: the int8 model, vocab.json,
# misaki json lexicons, packs/*.bin: the bundled Bella). This kit's pack is a different layout and keeps one model per
# dir, so here it builds beside the app, not into the bundle: default DEST = ios/build/kokoro-kit (gitignored, not
# bundled), and it refuses a DEST that holds VOICESHOP's runtime. Not called by build-ios.sh. docs/VOICES.md, "KOKOROWIRE".
DEST="${1:-$(cd "$HERE/.." && pwd)/ios/build/kokoro-kit}"
if [ -f "$DEST/vocab.json" ] || [ -d "$DEST/packs" ]; then
  echo "fetch-kokoro: $DEST is VOICESHOP's Kokoro dir (vocab.json/packs); this kit would delete its model. Pick another DEST." >&2; exit 5
fi
MODEL="${KOKORO_MODEL:-fp16w}"
TMP="${TMPDIR:-/tmp}/bend-fetch-kokoro"
mkdir -p "$TMP" "$DEST"

fetch() { # url sha256 dest
  if [ -f "$3" ] && [ "$(shasum -a 256 "$3" | cut -d' ' -f1)" = "$2" ]; then return; fi
  curl -fsSL --retry 3 -o "$3.part" "$1"
  got="$(shasum -a 256 "$3.part" | cut -d' ' -f1)"
  [ "$got" = "$2" ] || { echo "fetch-kokoro: $1: sha256 $got, expected $2" >&2; rm -f "$3.part"; exit 3; }
  mv "$3.part" "$3"
}
check() { [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$2" ] || { echo "fetch-kokoro: $1 does not match $2" >&2; exit 4; }; }

GH=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
fetch "$GH/voices-v1.0.bin" bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d "$TMP/voices-v1.0.bin"
fetch https://files.pythonhosted.org/packages/06/e9/f092172a37c0994bf1e57e0d2f3b16bd18f63b98744d6039f18a055c2ad5/misaki-0.7.4-py3-none-any.whl \
  f9cf1afc0a2c0e77dadf02ae8203d3a7c312ee67b235d3b658c08367919b17c1 "$TMP/misaki-0.7.4.whl"
unzip -o -q "$TMP/misaki-0.7.4.whl" 'misaki/data/us_gold.json' 'misaki/data/us_silver.json' -d "$TMP/misaki"
check "$TMP/misaki/misaki/data/us_gold.json" 363673890e892285fb12c66fc10591a520db4c427943ec14f18df5e5976a242d
check "$TMP/misaki/misaki/data/us_silver.json" 01e066a4af73c60b96178cd1c81862b812b811d1dac72d26a37b420597aa8859

case "$MODEL" in
  int8)
    NAME=kokoro-v1.0.int8.onnx
    fetch "$GH/kokoro-v1.0.int8.onnx" 6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb "$DEST/$NAME" ;;
  fp16w)
    NAME=kokoro-v1.0.fp16w.onnx
    SHA=13bbfcc26d7693267a521a0a033632649fe00f22eff73611e0b74b527accd360
    if ! { [ -f "$DEST/$NAME" ] && [ "$(shasum -a 256 "$DEST/$NAME" | cut -d' ' -f1)" = "$SHA" ]; }; then
      fetch "$GH/kokoro-v1.0.onnx" 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 "$TMP/kokoro-v1.0.onnx"
      [ -x "$TMP/venv/bin/python" ] || { python3 -m venv "$TMP/venv" && "$TMP/venv/bin/pip" -q install onnx==1.23.2 numpy==2.5.3; }
      "$TMP/venv/bin/python" "$TOOLS/fp16_storage.py" "$TMP/kokoro-v1.0.onnx" "$DEST/$NAME.part"
      mv "$DEST/$NAME.part" "$DEST/$NAME"
      check "$DEST/$NAME" "$SHA"
    fi ;;
  *) echo "fetch-kokoro: KOKORO_MODEL=$MODEL (fp16w|int8)" >&2; exit 2 ;;
esac
# one model per pack
find "$DEST" -maxdepth 1 -name '*.onnx' ! -name "$NAME" -delete
python3 "$TOOLS/build_assets.py" "$DEST" "$TMP/voices-v1.0.bin" "$NAME" "$TMP/misaki/misaki/data"
echo "fetch-kokoro: Kokoro-82M v1.0 ($MODEL, $(du -sh "$DEST" | cut -f1)) in $DEST"
