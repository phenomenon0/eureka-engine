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

## Any shape and long context

The Metal any-shape patches (`../metal/cc/metal-anyshape-cc.patch`,
`../metal/engine/metal-anyshape-engine.patch`, see
[`../metal/README.md`](../metal/README.md#any-shape)) also apply, with
`patch --fuzz=0`, on top of the full isildur stack (`VISION=1`, the resident
session and `metal-camera.patch`): append them after `metal-camera.patch` in
`engine/mac/build.sh`. That emit (`3fe84a23…` on 2026-10-26) compiles on
macOS as Objective-C. The recipe here does not apply them yet, so
`patches/CAM_SRC_SHA` is unchanged.

The wrapper (`engine/engine/llb_engine.m`) changed in two places:

- `llb_open` rounds `ctx` up to a power of two in 64 .. 2^27 (was 2^17). The
  engine keeps the KV, attention planes and fast-chunk area in its own Metal
  buffer, sized from that capacity, and refuses with the bytes it needs when
  they exceed the device's `maxBufferLength`.
- `llb_span` defaults to GGUF + GGUF/2 + 1 GiB (was GGUF + 3 GiB).
  `LLB_GPU_EXTRA_MB` and `LLB_GPU_SPAN_MB` still override it. The span holds
  weights and tables only, so it does not grow with `ctx`. Floors measured
  on the M3 Ultra at ctx 4096: 0.8B 1.6 GB, 2B 2.6 GB, 4B 4.5 GB, 9B 8.8 GB.

None of this ran on an iPhone or in the simulator: the library, the span
default, the Metal kernels of the iOS stack and the 0.8B / 2B models are
untested there.

## Gaps

- The graded app's `Vendor/SHA256SUMS` pins an engine header that does not
  match `engine/` at its imported commit, so the graded library was built from
  a nearby revision.
- `engine/isildur.sh verify` compares against the app's Swift bridge,
  `ios/Bend/Engines/LLB/LLBEngine.swift`, which is in the app repository.
- The any-shape changes are untested on a device (see
  [Any shape and long context](#any-shape-and-long-context)).
