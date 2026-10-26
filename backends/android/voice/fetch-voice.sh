#!/usr/bin/env bash
# KOKORODROID: fetches what the app's Kokoro voice needs, pinned by SHA-256, into app/src/main/assets/voices/kokoro
# (not checked in: ~100 MB; the hashes below are the record). Idempotent. The Android mirror of iOS voiceshop's
# scripts/fetch-voice.sh (same files, same pins), minus Supertonic and the xcframework (onnxruntime-android is an AAR in app/libs).
#
#   assets/voices/kokoro/   Kokoro-82M v1.0 (hexgrad, Apache-2.0): kokoro-onnx's int8 export (92 MB), vocab.json (hexgrad
#                           config.json), misaki US + GB lexicons (Apache-2.0; the G2P, no espeak-ng), packs/ = voice/bundled.json.
#   app/libs/onnxruntime-android-1.24.2.aar   Microsoft onnxruntime (MIT), Maven Central.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
K="$ROOT/app/src/main/assets/voices/kokoro"; TMP=${TMPDIR:-/tmp}/kokorodroid-fetch
mkdir -p "$K/packs" "$TMP/pt" "$ROOT/app/libs"

fetch() { # url sha256 dest
  if [ -f "$3" ] && [ "$(sha256sum "$3" | cut -d' ' -f1)" = "$2" ]; then return; fi
  curl -fsSL --retry 3 -o "$3.part" "$1"
  got="$(sha256sum "$3.part" | cut -d' ' -f1)"
  [ "$got" = "$2" ] || { echo "fetch-voice: $1: sha256 $got, expected $2" >&2; rm -f "$3.part"; exit 3; }
  mv "$3.part" "$3"
}

fetch https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar \
  bc461499a735653dff285a6a3477d28b9cfd119a09c7753eaf003426b577f223 "$ROOT/app/libs/onnxruntime-android-1.24.2.aar"
fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx \
  6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb "$K/kokoro-v1.0.int8.onnx"
KREV=$(python3 -c "import json;print(json.load(open('$HERE/bundled.json'))['revision'])")
KHF="https://huggingface.co/hexgrad/Kokoro-82M/resolve/$KREV"
fetch "$KHF/config.json" 5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f "$TMP/kokoro-config.json"
MISAKI=https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3
fetch "$MISAKI/misaki/data/us_gold.json" dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f "$K/us_gold.json"
fetch "$MISAKI/misaki/data/us_silver.json" de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475 "$K/us_silver.json"
fetch "$MISAKI/misaki/data/gb_gold.json" 29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71 "$K/gb_gold.json"
fetch "$MISAKI/misaki/data/gb_silver.json" 48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce "$K/gb_silver.json"
fetch "$MISAKI/LICENSE" c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 "$K/LICENSE-misaki"
python3 -I - "$HERE/bundled.json" "$TMP/kokoro-config.json" "$K" "$TMP/pt" "$KHF" <<'PY'
import hashlib, json, os, subprocess, sys, zipfile
spec, config, out, tmp, hf = sys.argv[1:]
vocab = json.load(open(config))["vocab"]
json.dump(vocab, open(f"{out}/vocab.json", "w"), ensure_ascii=False, sort_keys=True)
open(f"{out}/LICENSE", "w").write("Kokoro-82M (hexgrad/Kokoro-82M): Apache-2.0. Voices: hexgrad/Kokoro-82M voices/*.pt, Apache-2.0.\n"
    "kokoro-v1.0.int8.onnx: thewh1teagle/kokoro-onnx model-files-v1.0 (MIT tooling; Apache-2.0 weights).\n"
    "us_/gb_ gold/silver lexicons: hexgrad/misaki, Apache-2.0 (LICENSE-misaki).\n"
    "onnxruntime-android: Microsoft, MIT.\n")
order = []
for v in json.load(open(spec))["voices"]:
    pt = f"{tmp}/{v['id']}.pt"
    if not os.path.exists(pt) or hashlib.sha256(open(pt, "rb").read()).hexdigest() != v["pt_sha256"]:
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", pt, f"{hf}/voices/{v['id']}.pt"], check=True)
    got = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    if got != v["pt_sha256"]: sys.exit(f"fetch-voice: {v['id']}.pt sha256 {got}, expected {v['pt_sha256']}")
    raw = zipfile.ZipFile(pt).read(next(i for i in zipfile.ZipFile(pt).infolist() if i.filename.endswith("/data/0")))
    assert len(raw) == 510 * 256 * 4, (v["id"], len(raw))
    open(f"{out}/packs/{v['id']}.bin", "wb").write(raw)
    meta = {k: v[k] for k in ("id", "name", "lang", "gender", "grade", "detail")}
    meta |= {"engine": "kokoro", "license": "Apache-2.0", "credit": "hexgrad/Kokoro-82M", "source_sha256": v["pt_sha256"]}
    json.dump(meta, open(f"{out}/packs/{v['id']}.json", "w"), ensure_ascii=False, indent=1)
    order.append(v["id"])
json.dump(order, open(f"{out}/packs/index.json", "w"))
PY
echo "fetch-voice: kokoro $(du -sh "$K" | cut -f1) ($(ls "$K"/packs/*.bin | wc -l) voices: $(cat "$K/packs/index.json")) + onnxruntime-android 1.24.2"
