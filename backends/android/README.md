# Android

The Android engine: the Bend CPU engine built as a static aarch64 executable,
`libbendengine.so`. The Android app runs it as a child process and talks to it
in JSON lines. The app itself (Kotlin, WebView, and the `see`, `hear`, and
`speak` glue) is in the app repository; see [The app](#the-app).

## Measured result

| | |
|---|---|
| Device | Pixel 6a (`bluejay`, Tensor G1) |
| Model | Kev-4B Q4_K_M (pack `pack-kev33ae-q4`) |
| Chat, end to end | 3,176.4 ms median for a 19-token prompt and a 2-token reply |
| Route | CPU. The Vulkan path is not built and was not measured. |

Method: HTTP `POST /v1/chat/completions` to the in-app server over
`adb forward`, wall clock per request, median over the session's repetitions.
All five verbs passed in the backend-parity battery (2026-10-21).

## How the engine is used

| Verb | Runs on |
|---|---|
| `chat` | `libbendengine.so`, `kev.c generate` |
| `decide` | `libbendengine.so`, `kev.c --decide 2048`, with `kev-head-q8.gguf` |
| `see`, `hear`, `speak` | The app: D-FINE-N on ONNX Runtime, whisper.cpp, and Kokoro-82M on ONNX Runtime |

The app runs one engine process for its lifetime:

```sh
taskset f0 libbendengine.so <files>/<pack> --decide 2048
# environment: OMP_NUM_THREADS=4 LLB_RESIDENT=1 KEV_EOS=1
```

The executable is named `libbendengine.so` so that Android extracts it to the
native library directory, the only place an app may execute a binary from.

## Layout

| Path | Contents |
|---|---|
| `engine/vendor/` | Vendored qwen35 CPU engine sources (`vendor/PROVENANCE.txt`). Never edited. |
| `engine/patches/` | Ordered patches: speed (01–12), Kev decide loop (20), end-of-turn stop (21), vision (30–31), shape-general state (40), width-general decide head (41). |
| `engine/vision/` | The vision tower used by the vision patches. |
| `engine/build.sh` | Copies `vendor/` to the output directory and applies the patches with `patch --fuzz=0`. |

## Models

The engine reads the model shape (layers, width, heads, KV heads, FFN size,
vocabulary, linear-attention sizes) from the pack's `index.tsv`, so any dense
Qwen3.5 pack runs. Weights may be F32, Q8_0, Q4_K, Q5_K, Q6_K, or Q3_K; the
engine refuses F16 and IQ4_XS tensors at load.

| Model | Tested | Result |
|---|---|---|
| Kev-4B Q4_K_M | host build, `--decide 2048` | chat and decide output byte-identical to the build without patches 40–41 |
| Qwen3.5-2B Q4_K_M (d 2048, 2 KV heads, 24 layers) | host build, `--decide 2048`, no `LLB_GGUF` | chat works; decide is refused (no head) |

`decide` needs a Kev decide head (`LLB_GGUF`) whose width matches the model.
Without `LLB_GGUF` the engine serves `generate` only.

Patch 40 allocates the attention KV cache only for full-attention layers and
the recurrent state only for linear layers. On Kev-4B at context 2048 this
reserves 384 MiB less KV; measured peak RSS fell by 33.7 MB (2,800,752 →
2,767,060 KB, `/usr/bin/time -f %M`), because untouched KV pages were never
resident.

## Build

Requirements: Linux x86-64, Android NDK 27.2.12479018.

```sh
export NDK=/path/to/android-ndk-r27c
engine/build.sh android /path/to/scratch/engine
```

The output is `/path/to/scratch/engine/kev-engine-android`. The app ships it as
`app/src/main/jniLibs/arm64-v8a/libbendengine.so`. `engine/build.sh host DIR`
builds the same engine for the host, for testing.

Any drift between `vendor/` and the patches fails the build.

## Download separately

| File | Size | How to get it |
|---|---|---|
| Kev-4B Q4_K_M pack (`pack.bin`, `tokenizer.bin`, `manifest.json`, `index.tsv`, `chat.flags`) | 2.7 GB | Build from the Kev-4B Q4_K_M GGUF with `gguf2pack --format quant` |
| `kev-head-q8.gguf` (decide head) | 1.4 MB | Built from Kev-4B Q8_0 |

## The app

The Android app that hosts this engine is in `github.com/phenomenon0/bend-app`,
directory `android/`. Its `build-and-install.sh` builds this engine from a
checkout of this repository (`EUREKA_ENGINE`). The measured build is
`dev.bend.app.gpu` 0.8 (versionCode 8).

## Source

`engine/` is imported from that commit, path `android/engine/`. Host paths in
`engine/build.sh` were replaced with environment variables (`NDK`).
