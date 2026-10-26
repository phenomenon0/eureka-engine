# iOS

The iOS engine: `LLBEngine.xcframework`, the Bend Metal engine as a static
library for iPhone, the iOS Simulator, and macOS. The Eureka iPhone app links
it and runs it in the app process. The app itself is in
`github.com/phenomenon0/bend-app`, directory `ios/`.

Tested on an iPhone 18 Pro Max (12 GB).

## Result

| Verb | Model | Measured |
|---|---|---|
| `chat` | Kev-4B Q4_K_M | 20.6 tok/s end to end, 24.1 tok/s decode (warm median, 10 reps) |

All five verbs passed in the backend-parity battery (2026-10-21). In the app,
`chat` and `decide` run on this engine. `see` (D-FINE-N on ONNX Runtime),
`hear` (Apple SpeechAnalyzer), and `speak` (Kokoro-82M) run in the app.

## Layout

| Path | Contents |
|---|---|
| `engine/` | Build recipe for `LLBEngine.xcframework` (the 4B and Kev phone engine): `isildur.sh`, ordered patches, the library wrapper `llb_engine.{h,m}`, the decide head, and the 4B vision tower host. See [`engine/README.md`](engine/README.md). |
| `EngineMac/` | macOS command-line harness that runs the same engine library. |
| `Vendor/SHA256SUMS` | Hashes of the engine libraries the graded app was built against. |
| `Vendor/ears-src/` | Build script for the optional `EARS.xcframework` speech tower ([`tools/ears`](../../tools/ears/README.md)). |

## Build

On a Mac with Xcode:

```sh
cd engine
./isildur.sh
```

[`engine/README.md`](engine/README.md) gives the environment variables, the
emit, library, and xcframework steps, and the pinned hashes. Copy the
resulting `LLBEngine.xcframework` into the app's `ios/Vendor/`.

## Gaps

- The graded app's `Vendor/SHA256SUMS` pins an engine header that does not
  match `engine/` at its imported commit, so the graded library was built from
  a nearby revision.
- `engine/isildur.sh verify` compares against the app's Swift bridge,
  `ios/Bend/Engines/LLB/LLBEngine.swift`, which is in the app repository.
