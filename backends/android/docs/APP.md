# Bend Android app (v0)

The phoneone web UI (`app/src/main/assets/web/`, the phoneone web app's `dist` build at app commit aa5a7c1) in one WebView. It talks to **our** engine running on the phone. llama.cpp is not in the loop.

```
WebView ──http──▶ Server.kt (127.0.0.1:<ephemeral>, same origin) ──JSON lines──▶ libbendengine.so (one resident child process)
           dist from assets + the routes in src/engines/transport.ts       qwen35 CPU engine + kev.c --decide 2048, Kev-4B Q4_K_M pack
```

- `MainActivity.kt` starts the server and the WebView, and warms the engine at launch. `onDestroy` kills the engine.
- `Server.kt` serves the dist and injects `window.__BEND_NATIVE__ = {control, engine, key, autorun}` into `index.html`. It handles:
  - `GET /engines/status`
  - `POST /engines/ensure`
  - `POST /engines/visibility`
  - `POST /v1/chat/completions`: SSE or not. The ChatML no-think prompt goes to kev.c `generate`, and tokens stream from the engine's stderr `{"piece":hex}` lines.
  - `POST /v1/systemone`: the body goes to kev.c's decide verbatim.

  POSTs need the per-launch `X-Api-Key`.
- `Engine.kt` runs ONE process for the app's life:
  - command: `taskset f0 libbendengine.so <files>/pack --decide 2048`
  - env: `OMP_NUM_THREADS=4 OMP_SCHEDULE=dynamic,8 LLB_RESIDENT=1 KEV_EOS=1 LLB_GGUF=<files>/kev-head-q8.gguf`

  Requests are serialized. Everything goes to `<files>/engine.log`.
- `engine/build.sh` builds the engine from source:
  - It starts from feddecide's vendored engine (`engine/vendor`, untouched) and applies `engine/patches`, using `patch --fuzz=0`.
  - Patches 01-12 are the cpspeed4 stack (g12d8 flags, dot8 order).
  - Patch 20 is feddecide's `--decide` loop.
  - Patch 21 (`KEV_EOS`): stop at `<|im_end|>`, stream pieces, and parse ChatML markers in generate prompts. Without `KEV_EOS`, kevserve's contract is unchanged.
  - The output is a static aarch64 executable, 16 KB page-aligned. It ships as `jniLibs/arm64-v8a/libbendengine.so` because the native lib dir is the place an app may exec from.

## Build + install

```
./build-and-install.sh              # engine, APK, install, models (first time), launch
AUTORUN=1 ./build-and-install.sh    # + the web app's autorun: chat turns and decisions, marks in window.__bendTimings
```

Needs:
- NDK 27.2;
- the Gradle/AGP cache (the build runs offline);
- JDK 17 (`~/.gradle/jdks/eclipse_adoptium-17-amd64-linux.2`).

Every adb op runs under `flock /tmp/pixel6a.lock`.

## Model files (not in the APK)

The files go in `/sdcard/Android/data/dev.bend.app/files/`:

| file | bytes |
|---|---|
| `pack/pack.bin` | 2,698,575,872 |
| `pack/tokenizer.bin` | 7,460,363 |
| `pack/manifest.json` | 105,355 |
| `pack/index.tsv` | 26,519 |
| `pack/chat.flags` | 8 |
| `kev-head-q8.gguf` | 1,395,008 |

The pack is the Kev-4B Q4_K_M pack (`$PACK` in `build-and-install.sh`), and the head from `mkhead.py` on Kev-4B-Q8_0.

Launch the app once **before** pushing. A directory that adb shell creates under `Android/data` is not readable by the app, and the app shows "no models".

`adb uninstall dev.bend.app` removes the app and the files.

Release (signed, minified) build and the signing key: `docs/RELEASE.md`.

## Debug

- `adb shell cat /sdcard/Android/data/dev.bend.app/files/engine.log` shows the spawn, live, chat and decide lines with ms and RSS.
- WebView devtools are enabled (debug build): `chrome://inspect`, then read `window.__bendTimings`.
