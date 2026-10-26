# iOS

The Eureka iPhone app. It runs the Bend Metal engine in the app process and
serves all five verbs on a loopback HTTP listener (`127.0.0.1`). The web UI
runs in a `WKWebView` and calls that listener.

Tested on an iPhone 18 Pro Max (12 GB).

## Result

| Verb | Model | Measured |
|---|---|---|
| `chat` | Kev-4B Q4_K_M | 20.6 tok/s end to end, 24.1 tok/s decode (warm median, 10 reps) |

The graded build was a Debug build with `SWIFT_OPTIMIZATION_LEVEL=-O`. Debug
is needed for the `BEND_API_KEY` environment seam that the receipts use.

## How each verb runs

| Verb | Runtime | Code |
|---|---|---|
| `chat` | Bend Metal engine (`LLBEngine.xcframework`), resident session | `Bend/Engines/LLB/` |
| `decide` | Bend Metal engine, same model and session as `chat` | `Bend/Engines/LLB/`, `Bend/Engines/Provider/` |
| `see` | D-FINE-N detector on ONNX Runtime 1.24, CPU execution provider | `Bend/Engines/LLB/DetectORT.swift`, `VisionPocket.swift` |
| `hear` | Apple SpeechAnalyzer (Speech framework, on device) | `Bend/Voice/` |
| `speak` | Kokoro-82M on ONNX Runtime | `Bend/Voice/KokoroVoice.swift` |

`Bend/Engines/LLB/LLBEngine.swift` and `Bend-LLB-Bridging-Header.h` are the
bridge between Swift and the engine's C API (`include/llb_engine.h`).
`EngineServer.swift` serves the OpenAI-style routes; `ProviderRoutes.swift`
serves `/v1/see`, `/v1/hear`, `/v1/speak`, and the decision route.

## Layout

| Path | Contents |
|---|---|
| `project.yml` | XcodeGen project. Targets: `Bend` (Bend engine), `BendLlama` (llama.cpp fallback), `BendUITests`. |
| `Bend/` | App source: engine bridge, HTTP servers, camera, voice, model download. |
| `Bend/Engines/LLB/engine-check/`, `Bend/Engines/Provider/provider-check/`, `Bend/Voice/voice-check/` | Fixtures for the in-app `-engineCheck` and `-voiceCheck` self-tests. Bundled as resources. |
| `EngineMac/`, `VoiceMac/`, `CamMac/`, `ModelsMac/` | macOS command-line harnesses for the same code. |
| `Tests/`, `UITests/` | Unit and UI tests. |
| `Vendor/SHA256SUMS` | Hashes of the engine libraries the app was built against. |
| `Vendor/ears-src/` | Build script for the optional `EARS.xcframework` speech tower. |
| `scripts/` | Build, deploy, and fetch scripts from `bend-app/scripts/`. |

## Requirements

- macOS with Xcode and the iOS 27 SDK.
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
- Node.js, to build the web UI.
- An Apple developer team, for device builds. The app uses the
  `increased-memory-limit` and `extended-virtual-addressing` entitlements.

## Download or build separately

These files are not in this repository.

| Item | Where it goes | How to get it |
|---|---|---|
| `LLBEngine.xcframework` (static, about 3.6 MB per slice) | `Vendor/LLBEngine.xcframework` | Build the Metal engine in [`../metal`](../metal/README.md) for `ios-arm64` and `ios-arm64-simulator`, and package it with `xcodebuild -create-xcframework`. Check it against `Vendor/SHA256SUMS`. |
| `EARS.xcframework` (static, about 11.5 MB per slice) | `Vendor/EARS.xcframework` | `Vendor/ears-src/build-ears-v3.sh Vendor`. Set `EARS_CARRY` and `EARS_LLAMA` to the source trees first. Needed because `project.yml` links it. `hear` does not use it by default. |
| `onnxruntime.xcframework` 1.24.2 | `Vendor/onnxruntime.xcframework` | `scripts/fetch-voice.sh` (pinned SHA-256). |
| Kokoro-82M int8 model, voice packs, lexicons | `Vendor/voices/` | `scripts/fetch-voice.sh`. |
| Kev-4B Q4_K_M (`Kev-4B-Q4_K_M.gguf`, 3,033,489,824 bytes) and its sidecar files | `Documents/models/kev-4b-q4km/` on the phone | The in-app first-run download, or copy with `xcrun devicectl` or Finder file sharing. The file list is in `Bend/Models/ModelManifest.swift`. |
| D-FINE-N (`dfine-n.onnx`, 15,534,295 bytes) | `Documents/ml/` on the phone | Same as above. |
| Web UI bundle | `web/` | Build the `bend-app` web app (`npm run build`) and copy `dist/` into `web/`. |
| Host-side test fixtures (`ios/fixtures/` in `bend-app`) | not needed | Only the Mac harnesses use them. |

## Build

1. Put the items above in place.
2. Set your team ID. `project.yml` reads it from the environment:

   ```sh
   export BEND_TEAM_ID=<your Apple team ID>
   ```

3. Replace the placeholder bundle prefix `com.example` in `project.yml`,
   `Bend/Info.plist`, `EngineMac/Info.plist`, and `Bend/OSBridge.swift` with
   your own.
4. Generate and open the project:

   ```sh
   xcodegen generate
   open Bend.xcodeproj
   ```

5. Build the `Bend` scheme for a device.

The scripts in `scripts/` are copied from `bend-app` and expect the
`bend-app` layout (`ios/` next to `scripts/`, the web app at the root).
`scripts/build-ios.sh` runs the web build, `fetch-voice.sh`, the
`SHA256SUMS` check, XcodeGen, and a simulator build. `scripts/deploy-phone.sh`
needs `UDID` and `BEND_TEAM_ID` set.

## Run the self-tests

Launch arguments select the graded checks. For example, the `chat` check:

```text
-kevChat -engineCheck chatcell -chatIters 5 -engineCheckExit
```

The loopback routes require the per-launch key in the `X-Api-Key` header.
Set `BEND_API_KEY` in the launch environment to fix it (Debug builds only).

## Source

Imported from `github.com/phenomenon0/bend-app`, commit
`41e85a686e79c14611ae091384bd29082b79e531` (2026-10-19), directory `ios/`,
plus the iOS scripts from `scripts/`. This is the `repo_commit` in the
`metal-ios-*` receipts.

Changes from the source:

- Bundle ID prefix replaced with `com.example`.
- Team ID and device UDID removed; read from the environment.
- Developer machine paths removed from comments and scripts.
