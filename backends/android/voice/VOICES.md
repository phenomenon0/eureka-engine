# Voices on Android: the bundled set, the picker, the drop-in folder, the shop (KOKORODROID)

The Android half of iOS VOICESHOP (the iOS app's `docs/VOICES.md`). It uses the same voices, the same
catalog file, the same pins and the same error wording. It has the same sections, so the two platforms can be read side by side.

## The bundled set (5 voices, `voice/bundled.json`)

| Order | Voice | Grade | Note |
|---|---|---|---|
| 1 | **af_bella**, house voice | A- | Owner pick: the bundle default and the playback default (`Voices.DEFAULT_ID`). |
| 2 | af_heart | A | |
| 3 | am_michael | C+ | The male voice. |
| 4 | af_nicole | B- | |
| 5 | bf_emma | B- | British. The G2P switches to the GB lexicon. |

`voice/fetch-voice.sh` fetches everything at the pinned hexgrad revision `f3ff3571`. It checks the sha256 of every file
and writes `app/src/main/assets/voices/kokoro/`, which is gitignored because it is about 100 MB; the hashes in the script are the record.

## Sizes (measured, bytes)

| File | Bytes |
|---|---|
| `kokoro-v1.0.int8.onnx` (kokoro-onnx int8 export, stored uncompressed and mmapped) | 92,361,271 |
| misaki lexicons (us/gb, gold + silver) | 12,602,436 |
| `packs/` (5 × 522,240 `.bin` + json + index) | 2,612,813 |
| `onnxruntime-android-1.24.2.aar` (`app/libs/`, gitignored) | 40,822,946 |
| Debug APK, whole app (sha256 `0ba09bc1…`) | 113,701,159 |
| One shop pack (hexgrad `.pt`) | about 523,400 |

## Licenses

Every shipped piece is permissive:

- Kokoro-82M weights and voices: Apache-2.0 (hexgrad).
- int8 ONNX export: kokoro-onnx (MIT tooling, Apache-2.0 weights).
- misaki lexicons: Apache-2.0.
- onnxruntime: MIT.

These are recorded in `assets/voices/kokoro/LICENSE` and `LICENSE-misaki`. Every catalog entry carries its `license`, and the shop shows it.

## Picking a voice (`voices.html`)

Open it from the Voice screen: tap the "speech on device · Kokoro Bella" line. `Server.kt` injects a click hook into the
phoneone page. The page stays inside the app's WebView: `MainActivity` keeps 127.0.0.1 URLs in-app, and hardware Back returns.

- **List:** `GET /voices` returns the bundled voices in order, then shop installs, then drop-ins.
- **Select:** `POST /voices/select` persists the choice in the SharedPreferences file `voices`, key `choice`.
- **Hot-swap:** `Voice.speak` reads the current pack on every call, so the next reply uses the new voice. Nothing restarts. The
  ONNX session is shared, and only the 522 KB style table changes.
- **Preview:** `POST /voices/preview {id, text?, play}` speaks the demo line. It returns timings: `first_clause_ms`, `rtf` and `first_audio_ms`.

## The drop-in convention (`voices/`)

Put a Kokoro voice file in `/sdcard/Android/data/dev.bend.app/files/voices/`. The app creates this folder, and you can reach it with `adb push` or a
file manager. On the next list it appears as "your file". Accepted formats:

- `<name>.bin`: raw float32 [510, 256] (522,240 bytes). This is the same format as the bundled packs.
- `<name>.pt`: hexgrad's torch zip, as downloaded. The app reads the stored `…/data/0` tensor; no conversion is needed.

The id is `kokoro:<name>`. Files below 522,240 bytes or above 4× that are ignored. A file with a `<name>.json` sidecar
carrying `installed` is a shop install, not a drop-in. Receipt: `receipts/dropin.txt`, where a `kore.bin` cut from `af_kore.pt` lists
and speaks (RTF 0.93).

## The shop (More voices)

The flow is: catalog → **Get** (progress bar, Pause/Resume) → `.part` file + HTTP Range resume → streaming sha256 and byte-count check
→ rename → style-table validation → install. The new voice appears in the picker. **▶** plays the catalog's preview clip.

Failure behavior:

- **Transient network failures:** retried twice (after 1 s, then 3 s), then **Retry** is the person's. Every retry resumes from the `.part`.
- **Wrong checksum or wrong host size:** the file is deleted and there is no automatic retry.
- **Offline:** the bundled voices keep working. The shop says "Can’t reach the voice shop (host)" and shows the last cached list,
  if there is one (`cacheDir/voice-catalog.json`).

### The catalog (`voice/catalog.json`, one file)

This is the same file as iOS. Each entry has `id, engine, name, lang, gender, grade, license, credit, preview_url, pack_url, sha256, bytes, file`.
`pack_url` and `preview_url` are absolute, or relative to the catalog. The 5 entries (aoede, kore, sarah, isabella, fable) point at
hexgrad's pinned revision on Hugging Face. The previews are the iOS voiceshop `voices/previews/*.m4a`.

### Decision gate: where the catalog lives

`Voices.CATALOG_URL` is one constant in `app/src/main/java/dev/bend/app/Voices.kt`. It is
**`https://models.bend.invalid/v0/voices/catalog.json`**. This mirrors iOS `VoiceCatalog.CATALOG_URL`: the `.invalid` host means
nothing is live until the owner picks a host, and meanwhile the app shows the clean offline state.

**To go live:** host `catalog.json`, `previews/` and optionally `packs/` anywhere static over HTTPS with Range support, then change that line
on both platforms. Range support is what resume needs; `tools/serve-models.py` is a LAN stand-in.

**Developer override:** voices.html → "Your own voices & shop source" → Catalog URL. It is kept in prefs, and empty resets it.

## Adding another engine's voices

Catalog entries carry `engine`, and `Voices.packs` skips anything that is not `kokoro`. A second engine needs:

- an `engine` tag;
- a branch in `Voices.voice()` that returns a synthesizer with the same `synthesize(text) -> FloatArray` shape;
- its own pack validation in `install`.

The shop, the picker and the drop-in folder stay as they are.

## The G2P

`KokoroG2P` (in `Kokoro.kt`) is a line-for-line Kotlin port of the iOS `KokoroG2P.swift`. It uses misaki lexicons, has no espeak, and handles
numbers, currency, times and ordinals. The iOS output on 16 sentences (us + gb) is the golden: `voice/g2p-golden-ios.tsv`. Android matches it byte
for byte, and `voice/test-voices.py` checks this. Out-of-lexicon words fall back to letter spelling, the same as iOS. That fallback is an iOS limitation, not an Android one.

## Runtime and speed (Pixel 6a, measured)

The runtime is ORT 1.24.2 CPU on 4 intra-op threads, with the model mmapped. Kotlin synthesizes clause by clause on one thread and plays on another,
so the first clause plays while the next one renders.

| Measure | Value |
|---|---|
| Model load, cold | 0.9–1.06 s |
| Bella, demo line (7.08 s of audio) | RTF 0.88–0.91 when cool, 1.05–1.13 when warm |
| Heart / Michael / Emma | RTF 0.91 / 0.96 / 1.00 |
| First clause of 3.85 s | 3.4–3.5 s to render |
| First clause "Sure." | 1.07–1.14 s to first audio (`receipts/short-first-clause.json`) |
| Threads 2 / 4 / 6 / 8 | RTF ≈ 1.0 / 0.9 / 1.15 / 1.23 (`bench-threads.json`) |
| XNNPACK EP | Slower: 1.22–1.31 vs 0.98–1.08 (`bench-xnnpack.json`). Not used. |

At an RTF near 0.9, the phone renders barely faster than real time. A long clause that follows a short one leaves a gap. Untested ideas
for later: an fp16/fp32 export (dynamic int8 can lose on ARM), and capping clause length.

## Tests

`voice/test-voices.py` runs on the phone through `voice/page.py`. The trunk has no JVM test setup, and JUnit is not in the offline gradle cache. It checks:

- bella is the house voice and leads the order;
- G2P parity with iOS;
- with `--packs DIR --host IP`: a dropped download resumes from byte 200000, a wrong-sha pack is refused and kept out of the
  picker, the honest pack installs and speaks, and remove takes it out again.

Each check states why it matters.
