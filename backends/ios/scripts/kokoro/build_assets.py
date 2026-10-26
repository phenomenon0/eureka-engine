#!/usr/bin/env python3
"""KOKOROWIRE phone kit: build the on-device Kokoro pack around a model file. Python 3 standard library only (the stock
/usr/bin/python3 works), so the app's fetch script needs nothing installed.

  lexicon-us.tsv   word<TAB>phonemes, from misaki 0.7.4's us_gold + us_silver (Apache-2.0): misaki's grow_dictionary
                   (capitalised variants), POS dicts resolved to DEFAULT, gold wins over silver. No espeak-ng anywhere.
  voices/<v>.f32   the voice pack: raw little-endian Float32 [510][256] (row n = the style for an n-phoneme input),
                   from kokoro-onnx's voices-v1.0.bin (an .npz of the Kokoro-82M v1.0 voice tensors the Mac uses).
  kokoro.json      the manifest KokoroVoice reads: model file, input/output names, sample rate, default voice.

Usage: build_assets.py OUT_DIR VOICES_BIN MODEL_FILE MISAKI_DATA_DIR [voice ...]
  (MODEL_FILE is the .onnx's name inside OUT_DIR; put the model there yourself.)
"""
import ast, json, os, sys, zipfile

out, voices_bin, model_file, misaki_dir = sys.argv[1:5]
voices = sys.argv[5:] or ["af_heart", "af_bella", "af_nicole", "am_michael", "bf_emma", "bm_george"]
os.makedirs(f"{out}/voices", exist_ok=True)


def grow(d):  # misaki Lexicon.grow_dictionary
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


lex = {}
for name in ("us_silver.json", "us_gold.json"):  # gold last, so it wins
    for k, v in grow(json.load(open(os.path.join(misaki_dir, name), encoding="utf-8"))).items():
        if isinstance(v, dict):
            v = v.get("DEFAULT")
        if v and "\t" not in k and "\n" not in k:
            lex[k] = v
with open(f"{out}/lexicon-us.tsv", "w", encoding="utf-8") as f:
    for k in sorted(lex):
        f.write(f"{k}\t{lex[k]}\n")

# voices-v1.0.bin is an .npz: one .npy per voice, float32 [510, 1, 256], C order
with zipfile.ZipFile(voices_bin) as z:
    for v in voices:
        raw = z.read(f"{v}.npy")
        assert raw[:6] == b"\x93NUMPY", v
        hlen = int.from_bytes(raw[8:10], "little") if raw[6] == 1 else int.from_bytes(raw[8:12], "little")
        start = (10 if raw[6] == 1 else 12) + hlen
        hdr = ast.literal_eval(raw[(10 if raw[6] == 1 else 12):start].decode("latin1"))
        assert hdr["descr"] == "<f4" and not hdr["fortran_order"] and hdr["shape"] == (510, 1, 256), (v, hdr)
        data = raw[start:]
        assert len(data) == 510 * 256 * 4, v
        open(f"{out}/voices/{v}.f32", "wb").write(data)

json.dump({"model": model_file, "input": "tokens", "output": "audio", "sampleRate": 24000,
           "defaultVoice": "af_heart", "voices": voices, "lexicon": "lexicon-us.tsv",
           "licence": "Kokoro-82M v1.0 Apache-2.0 (hexgrad; ONNX export thewh1teagle/kokoro-onnx, MIT); "
                      "lexicon: misaki 0.7.4 Apache-2.0; no espeak-ng"},
          open(f"{out}/kokoro.json", "w"), indent=1)
print(f"kokoro pack: {len(lex)} lexicon entries, {len(voices)} voices, model {model_file} -> {out}")
