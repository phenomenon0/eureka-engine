#!/bin/bash
# VOICEREAL: fetches what the phone's neural voice needs, pinned by SHA-256, into ios/Vendor (not checked in: ~300 MB of
# static libraries + ~96 MB of weights; the hashes below are the record). Idempotent: a verified file is not fetched again.
#
#   ios/Vendor/onnxruntime.xcframework   onnxruntime 1.24.2 (MIT), Microsoft's pod archive: ios-arm64, ios-arm64_x86_64-simulator,
#                                        macos-arm64_x86_64; static. A module map is added so Swift can `import onnxruntime`.
#   ios/Vendor/voices/supertonic2/       Supertonic 2 (Supertone; weights OpenRAIL-M, see LICENSE there), voice F1:
#                                        duration predictor + text encoder + tts.json + unicode indexer + F1 style from
#                                        huggingface.co/Supertone/supertonic-2 @ 75e6727; the int8 vector estimator + vocoder
#                                        from k2-fsa's sherpa-onnx tts-models release (the same graphs, weights quantized).
#
#   ios/Vendor/voices/kokoro/            VOICESHOP: Kokoro-82M v1.0 (hexgrad, Apache-2.0): kokoro-onnx's int8 export (MIT tooling,
#                                        92 MB), the model's phoneme vocab (hexgrad config.json), misaki's US + GB lexicons
#                                        (Apache-2.0; the G2P, no espeak-ng), and packs/: the bundled voices of voices/bundled.json
#                                        as 510 x 256 float32 tables (<id>.bin, 522,240 B) + <id>.json + index.json (their order).
#
# The app bundles voices/ (project.yml); a missing folder builds and runs on Apple's voice (the fallback).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/ios/Vendor"
TMP="${TMPDIR:-/tmp}/bend-fetch-voice"
mkdir -p "$TMP" "$VENDOR/voices/supertonic2"

fetch() { # url sha256 dest
  if [ -f "$3" ] && [ "$(shasum -a 256 "$3" | cut -d' ' -f1)" = "$2" ]; then return; fi
  curl -fsSL --retry 3 -o "$3.part" "$1"
  got="$(shasum -a 256 "$3.part" | cut -d' ' -f1)"
  [ "$got" = "$2" ] || { echo "fetch-voice: $1: sha256 $got, expected $2" >&2; rm -f "$3.part"; exit 3; }
  mv "$3.part" "$3"
}

# onnxruntime
ORT_ZIP="$TMP/pod-archive-onnxruntime-c-1.24.2.zip"
fetch https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip f7100a992d2a8135168c8afd831e6a58b465349101982aa58b3e11d36e600b54 "$ORT_ZIP"
if [ ! -f "$VENDOR/onnxruntime.xcframework/.verified-1.24.2" ]; then
  rm -rf "$VENDOR/onnxruntime.xcframework" "$TMP/ort"
  unzip -q "$ORT_ZIP" -d "$TMP/ort"
  mv "$TMP/ort/onnxruntime.xcframework" "$VENDOR/onnxruntime.xcframework"
  cp "$TMP/ort/LICENSE" "$VENDOR/onnxruntime.xcframework/LICENSE"
  for fw in "$VENDOR"/onnxruntime.xcframework/*/onnxruntime.framework; do
    mkdir -p "$fw/Modules"
    printf 'framework module onnxruntime {\n  header "onnxruntime_c_api.h"\n  export *\n}\n' > "$fw/Modules/module.modulemap"
  done
  touch "$VENDOR/onnxruntime.xcframework/.verified-1.24.2"
fi

# Supertonic 2, voice F1
V="$VENDOR/voices/supertonic2"
HF=https://huggingface.co/Supertone/supertonic-2/resolve/75e6727618a02f323c720cba9478152d4bc16ca4
fetch "$HF/onnx/duration_predictor.onnx" 6d556b3691165c364be91dc0bd894656b5949f5acd2750d8ec2f954010845011 "$V/duration_predictor.onnx"
fetch "$HF/onnx/text_encoder.onnx" dd5f535ed629f7df86071043e15f541ce1b2ab7f1bdbce4c7892b307bca79fa3 "$V/text_encoder.onnx"
fetch "$HF/onnx/tts.json" ee531d9af9b80438a2ed703e22155ee6c83b12595ab22fd3bb6de94c7502fe96 "$V/tts.json"
fetch "$HF/onnx/unicode_indexer.json" b7662a73a0703f43b97c0f2e089f8e8325e26f5d841aca393b5a54c509c92df1 "$V/unicode_indexer.json"
fetch "$HF/voice_styles/F1.json" 6106950ebeb8a5da29ea22075f605db659cd07dbc288a68292543d9129aa250f "$V/F1.json"
fetch "$HF/LICENSE" 0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f "$V/LICENSE"
if [ ! -f "$V/vector_estimator.int8.onnx" ] || [ ! -f "$V/vocoder.int8.onnx" ] || \
   ! (cd "$V" && echo "9cce89801068dc998719fbc91777f99f3f4d16ee36592281f8703f32e305daa2  vector_estimator.int8.onnx
db99c36fb871fbfb128b903662995f2e22213e875f2434f319f8d10ee0f10e90  vocoder.int8.onnx" | shasum -a 256 -c --quiet 2>/dev/null); then
  TAR="$TMP/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2"
  fetch https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2 \
    8c74359f63edd5045d47747f65331f0f6dbcbc91d7e898dd756d631295fe3259 "$TAR"
  tar xjf "$TAR" -C "$TMP" sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx
  mv "$TMP"/sherpa-onnx-supertonic-tts-int8-2026-03-06/*.onnx "$V/"
  (cd "$V" && echo "9cce89801068dc998719fbc91777f99f3f4d16ee36592281f8703f32e305daa2  vector_estimator.int8.onnx
db99c36fb871fbfb128b903662995f2e22213e875f2434f319f8d10ee0f10e90  vocoder.int8.onnx" | shasum -a 256 -c --quiet)
fi
# VOICESHOP: Kokoro-82M v1.0 + the bundled voice set
K="$VENDOR/voices/kokoro"
mkdir -p "$K/packs" "$TMP/kokoro-pt"
fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx \
  6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb "$K/kokoro-v1.0.int8.onnx"
KREV=$(python3 -c "import json;print(json.load(open('$ROOT/voices/bundled.json'))['revision'])")
KHF="https://huggingface.co/hexgrad/Kokoro-82M/resolve/$KREV"
fetch "$KHF/config.json" 5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f "$TMP/kokoro-config.json"
MISAKI=https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3
fetch "$MISAKI/misaki/data/us_gold.json" dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f "$K/us_gold.json"
fetch "$MISAKI/misaki/data/us_silver.json" de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475 "$K/us_silver.json"
fetch "$MISAKI/misaki/data/gb_gold.json" 29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71 "$K/gb_gold.json"
fetch "$MISAKI/misaki/data/gb_silver.json" 48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce "$K/gb_silver.json"
fetch "$MISAKI/LICENSE" c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 "$K/LICENSE-misaki"
python3 - "$ROOT/voices/bundled.json" "$TMP/kokoro-config.json" "$K" "$TMP/kokoro-pt" "$KHF" <<'PY'
import hashlib, json, os, subprocess, sys, zipfile
spec, config, out, tmp, hf = sys.argv[1:]
vocab = json.load(open(config))["vocab"]
json.dump(vocab, open(f"{out}/vocab.json", "w"), ensure_ascii=False, sort_keys=True)
open(f"{out}/LICENSE", "w").write("Kokoro-82M (hexgrad/Kokoro-82M): Apache-2.0. Voices: hexgrad/Kokoro-82M voices/*.pt, Apache-2.0.\n"
    "kokoro-v1.0.int8.onnx: thewh1teagle/kokoro-onnx model-files-v1.0 (MIT tooling; Apache-2.0 weights).\n"
    "us_/gb_ gold/silver lexicons: hexgrad/misaki, Apache-2.0 (LICENSE-misaki).\n")
order = []
for v in json.load(open(spec))["voices"]:
    pt = f"{tmp}/{v['id']}.pt"
    if not os.path.exists(pt) or hashlib.sha256(open(pt, "rb").read()).hexdigest() != v["pt_sha256"]:
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", pt, f"{hf}/voices/{v['id']}.pt"], check=True)
    got = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    if got != v["pt_sha256"]: sys.exit(f"fetch-voice: {v['id']}.pt sha256 {got}, expected {v['pt_sha256']}")
    z = zipfile.ZipFile(pt)
    entry = next(i for i in z.infolist() if i.filename.endswith("/data/0"))
    raw = z.read(entry)
    assert len(raw) == 510 * 256 * 4, (v["id"], len(raw))
    open(f"{out}/packs/{v['id']}.bin", "wb").write(raw)
    meta = {k: v[k] for k in ("id", "name", "lang", "gender", "grade", "detail")}
    meta |= {"engine": "kokoro", "license": "Apache-2.0", "credit": "hexgrad/Kokoro-82M", "source_sha256": v["pt_sha256"]}
    json.dump(meta, open(f"{out}/packs/{v['id']}.json", "w"), ensure_ascii=False, indent=1)
    order.append(v["id"])
json.dump(order, open(f"{out}/packs/index.json", "w"))
PY
# VOICESHOP2: Kokoro's Core ML chain (FluidInference/kokoro-82m-coreml ANE/, Apache-2.0, pinned revision; 7 compiled
# stages, ~79 MB): the phone's fast Kokoro runtime (KokoroANE.swift); onnx above stays as its fallback.
# scripts/kokoro-ane.sha256 pins every file. A local copy (FluidAudio's cache) is used when it matches.
KA="$VENDOR/voices/kokoro-ane"
KAHF="https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/d93c5f3eed11847fb81bd6d6a76665e133d7ca77/ANE"
KACACHE="$HOME/.cache/fluidaudio/Models/kokoro-82m-coreml/ANE"
while read -r sha rel; do
  mkdir -p "$(dirname "$KA/$rel")"
  if [ ! -f "$KA/$rel" ] && [ -f "$KACACHE/$rel" ] && [ "$(shasum -a 256 "$KACACHE/$rel" | cut -d' ' -f1)" = "$sha" ]; then cp "$KACACHE/$rel" "$KA/$rel"; fi
  fetch "$KAHF/$rel" "$sha" "$KA/$rel"
done < "$ROOT/scripts/kokoro-ane.sha256"
printf 'Kokoro-82M Core ML chain: huggingface.co/FluidInference/kokoro-82m-coreml @ d93c5f3 (ANE/), Apache-2.0;\nconverted from hexgrad/Kokoro-82M (Apache-2.0). Glue in KokoroANE.swift after FluidAudio (Apache-2.0).\n' > "$KA/LICENSE-NOTE.txt"

echo "fetch-voice: onnxruntime 1.24.2 + supertonic2 F1 ($(du -sh "$V" | cut -f1)) + kokoro ($(du -sh "$K" | cut -f1), $(ls "$K"/packs/*.bin | wc -l | tr -d ' ') voices) + kokoro-ane ($(du -sh "$KA" | cut -f1)) in ios/Vendor"
