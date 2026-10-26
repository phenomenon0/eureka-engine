# Android backend

This directory holds the source for the Bend Android app (`dev.bend.app`, debug
build `dev.bend.app.gpu`) and the native engine it runs. The app serves the
Eureka web UI in a WebView and answers the five verbs (chat, decide, see, hear,
speak) on the phone, with no network calls.

The app starts a small HTTP server (`Server.kt`) on `127.0.0.1` with an
ephemeral port. The WebView, and any client forwarded over `adb forward`, calls
that server. Every `POST` needs the per-launch `X-Api-Key`, which the server
injects into the `index.html` it serves.

```
WebView / adb forward ──http──▶ Server.kt (127.0.0.1:<port>)
                                  ├─ chat, decide ─▶ libbendengine.so (child process, JSON lines)
                                  ├─ see          ─▶ Detector.kt  → ONNX Runtime (D-FINE-N)
                                  ├─ hear         ─▶ Voice.hearWav → whisper.cpp (libbendwhisper.so)
                                  └─ speak        ─▶ Voice.speakWav → Kokoro-82M int8 (ONNX Runtime)
```

## Measured result

| | |
|---|---|
| Device | Pixel 6a (`bluejay`, Tensor G1) |
| Model | Kev-4B Q4_K_M (fixture GGUF sha256 `33ae6b18…`, loaded as the Q4_K_M pack `pack-kev33ae-q4`) |
| Chat, end to end | **3,176.4 ms median** for a 19-token prompt and a 2-token reply |
| Route | CPU. The Vulkan path is not built and was not measured. |

The number comes from the graded receipt `android-vulkan-chat.json` (the cell
is named for the planned Vulkan target; the receipt records the route as CPU
and `vulkan_verdict: NO`). Method: HTTP `POST /v1/chat/completions` to the
in-app server over `adb forward`, wall clock per request, median over the
session's repetitions. The other four receipts are `android-vulkan-decide.json`,
`-see.json`, `-hear.json`, and `-speak.json` in the same mission-5x5 receipt set.

## How each verb runs

| Verb | Route | Runtime | Model |
|---|---|---|---|
| chat | `POST /v1/chat/completions` (SSE or JSON) | Bend CPU engine, `libbendengine.so` (`kev.c generate`) | Kev-4B Q4_K_M pack |
| decide | `POST /v1/systemone` | Bend CPU engine, `libbendengine.so` (`kev.c --decide 2048`) | Kev-4B Q4_K_M pack + `kev-head-q8.gguf` |
| see | `POST /v1/see` | `Detector.kt` on ONNX Runtime Android 1.24.2, CPU execution provider | D-FINE-N, `dfine_n_coco.onnx` |
| hear | `POST /v1/hear` | `Voice.hearWav` → whisper.cpp v1.9.2 in `libbendwhisper.so`, 4 threads | `ggml-tiny.en-q5_1.bin` (measured) |
| speak | `POST /v1/speak` | `Voice.speakWav` → `Voices` / `Kokoro.kt` on ONNX Runtime | Kokoro-82M int8 ONNX, voice `af_heart` |

- **chat and decide.** `Engine.kt` runs one engine process for the life of the
  app: `taskset f0 libbendengine.so <files>/<pack> --decide 2048`, with
  `OMP_NUM_THREADS=4 LLB_RESIDENT=1 KEV_EOS=1`. Requests are serialized. The
  engine is a static aarch64 executable named `libbendengine.so` so that
  Android extracts it to the native library directory, the only place an app
  may execute a binary from. `Models.kt` picks the pack from
  `<files>/models.json`.
- **see.** `Detector.kt` decodes the image, resizes it to 640×640 with a
  PIL-equivalent bilinear filter, and runs the D-FINE-N export, which has
  sigmoid, top-300, and box rescale built into the graph.
- **hear.** `Voice.hearWav` parses the WAV, downmixes to mono, resamples to
  16 kHz, and calls whisper.cpp through `asr/whisper_jni.c`. The APK bundles
  `ggml-base.en-q8_0.bin`; a `.bin` placed in `<files>/asr/` replaces it. The
  measured route used `ggml-tiny.en-q5_1.bin` this way.
- **speak.** `Voice.speakWav` synthesizes with Kokoro-82M (misaki G2P lexicons,
  no espeak) and returns 24 kHz PCM16 WAV. An optional fallback TTS for phones
  without a system TTS engine uses espeak-ng; it is not in this repository
  (see Known gaps).

## Layout

| Path | Contents |
|---|---|
| `app/` | Kotlin app (`dev.bend.app`), manifest, resources, and the prebuilt web UI in `app/src/main/assets/web/` |
| `engine/` | The engine that becomes `libbendengine.so`: vendored C sources (`vendor/`, see `vendor/PROVENANCE.txt`), ordered patches (`patches/`), the vision tower (`vision/`), and `build.sh` |
| `asr/` | `whisper_jni.c`, `CMakeLists.txt`, and `build.sh` for `libbendwhisper.so` |
| `voice/` | `fetch-voice.sh` (Kokoro model, voices, lexicons, ONNX Runtime AAR, all pinned by SHA-256), voice catalog, and device test scripts |
| `docs/` | `APP.md` (original app notes) and `RELEASE.md` (release signing) |
| `tools/` | `serve-models.py`, a Range-capable file server for model downloads |

## Requirements

- Linux x86-64 host
- Android SDK with platform 35 and NDK **27.2.12479018**
- JDK 17
- Gradle 8.11.1 (through the wrapper), Android Gradle Plugin 8.7.2, Kotlin 2.0.21
- CMake 3.22 or later and Ninja (for whisper.cpp)
- An arm64 phone running Android 14 (API 34) or later

## Build

Set the SDK location first:

```sh
export ANDROID_HOME=$HOME/Android/Sdk
export JAVA_HOME=/path/to/jdk-17
cd backends/android
```

1. Build the two native libraries. Each script writes into
   `app/src/main/jniLibs/arm64-v8a/`. Pass a scratch directory outside the
   source tree to the ASR script.

   ```sh
   engine/build.sh android /path/to/scratch/engine
   mkdir -p app/src/main/jniLibs/arm64-v8a
   cp /path/to/scratch/engine/kev-engine-android app/src/main/jniLibs/arm64-v8a/libbendengine.so
   asr/build.sh /path/to/scratch/whisper      # clones whisper.cpp v1.9.2, downloads ggml-base.en-q8_0.bin
   ```

   `engine/build.sh` copies `engine/vendor/` to the output directory and applies
   `engine/patches/*.patch` with `patch --fuzz=0`, so any drift fails the build.

2. Fetch the voice model and ONNX Runtime:

   ```sh
   voice/fetch-voice.sh
   ```

3. Build the debug APK:

   ```sh
   ./gradlew assembleDebug
   ```

   The APK is `app/build/outputs/apk/debug/app-debug.apk`. The debug build
   installs as `dev.bend.app.gpu`, next to a release `dev.bend.app`.

`build-and-install.sh` runs the engine build, the APK build, install, the first
model push, and launch in one step. It needs `SERIAL`, `PACK`, and `HEAD` set,
and it uses `./gradlew --offline`, so the Gradle cache must already be warm.

For a signed release build, see `docs/RELEASE.md`. Release signing reads a
properties file outside the repository; no key or password is stored here.

## Download separately

These files are not in the repository. Push the model files to the app's
external files directory, `/sdcard/Android/data/dev.bend.app.gpu/files/` for
the debug build. Launch the app once before you push, so that the app owns the
directory.

| File | Size | Where it goes | How to get it |
|---|---|---|---|
| Kev-4B Q4_K_M pack: `pack.bin`, `tokenizer.bin`, `manifest.json`, `index.tsv`, `chat.flags` | 2.7 GB | `files/<pack>/` | Build from the Kev-4B Q4_K_M GGUF with the llama-bend `gguf2pack --format quant` tool |
| `Kev-4B-Q4_K_M.gguf` | 3.0 GB | `files/` | Kev model release |
| `kev-head-q8.gguf` (decide head) | 1.4 MB | `files/` | llama-bend `mkhead.py` on Kev-4B Q8_0 |
| `mmproj-F16.gguf` (optional, photos in chat) | >600 MB | `files/` | Qwen3-VL vision projector for the Kev-4B model; not used by the five verbs above |
| `models.json` (optional) | small | `files/` | Selects the active pack; format in `Models.kt` |
| `dfine_n_coco.onnx` (15,534,295 B, sha256 `4daf426f…`) | 15.5 MB | `files/see/` | D-FINE-N COCO export with post-processing in the graph |
| `ggml-tiny.en-q5_1.bin` (measured) or `ggml-base.en-q8_0.bin` (bundled) | 32 MB / 82 MB | `files/asr/` or `app/src/main/assets/asr/` | `huggingface.co/ggerganov/whisper.cpp`; `asr/build.sh` fetches the bundled one |
| `kokoro-v1.0.int8.onnx`, voices, lexicons | ~100 MB | `app/src/main/assets/voices/kokoro/` | `voice/fetch-voice.sh` |
| `onnxruntime-android-1.24.2.aar` | — | `app/libs/` | `voice/fetch-voice.sh` (Maven Central) |
| `libbendengine.so`, `libbendwhisper.so` | — | `app/src/main/jniLibs/arm64-v8a/` | Build steps above |

## Source

Imported from the private llama-bend repository, branch
`ship/android/voice/ship-wip/app`, commit
`87d32614af07c4c1f298d6eb47561a7a22d1bda3` (2026-10-23), path `android/`. That
commit is a work-in-progress snapshot of the app tree that built the measured
`dev.bend.app.gpu` (versionName `0.8-models`, versionCode 8). The exact
measured APK is identified in the receipts by its installed sha256
(`4d0a6d9e…`), not by a commit.

Changes on import: the espeak-ng build and data were removed (GPL-3.0); device serials, LAN addresses, and absolute host paths in
the build and test scripts were replaced with environment variables
(`ANDROID_HOME`, `NDK`, `ADB`, `SERIAL`, `PACK`, `HEAD`), and the original
`android/README.md` moved to `docs/APP.md`.

## Known gaps

- The route is CPU only. No Vulkan build of the engine is wired into the app.
- The fallback TTS (`libespeak.so` and `espeak-ng-data`) is not included,
  because espeak-ng is GPL-3.0-or-later and this repository is Apache-2.0.
  Kokoro speech does not need it. To add the fallback, build espeak-ng from
  upstream yourself; an APK that includes it must be distributed under the
  GPL-3.0 terms. See `docs/RELEASE.md`.
- `Voices.CATALOG_URL` points to the placeholder host `models.bend.invalid`.
