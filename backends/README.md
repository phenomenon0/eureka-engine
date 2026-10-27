# Backends

This repository holds the engine for each backend. The apps that host it
(iOS, Android, macOS desktop, and the web UI) are in
`github.com/phenomenon0/bend-app`. Model weights and prebuilt libraries are
downloaded or built separately; each page lists them.

| Backend | Devices | Source | Result | Page |
|---|---|---|---|---|
| CUDA | NVIDIA, Linux | Repository root (`build.sh`) and `tools/sys1dual/` (server) | 38.8 tok/s, 27B, RTX 3090 | [cuda](cuda/README.md) |
| Metal | macOS | `metal/` (engine and server) | 70.2 tok/s, 9B, M3 Ultra | [metal](metal/README.md) |
| Metal | iOS | `ios/` (engine library) | 20.6 tok/s, 4B, iPhone 18 Pro Max | [ios](ios/README.md) |
| CPU | Android | `android/` (engine executable) | 3.2 s per short reply, 4B, Pixel 6a | [android](android/README.md) |
| WebGPU | Browser | `webgpu/` (in-tab engine) | 35.8 tok/s, 9B, M3 Ultra | [webgpu](webgpu/README.md) |

All five passed `chat`, `decide`, `see`, `hear`, and `speak` in the
backend-parity battery (2026-10-21). Conditions and gaps:
[docs/backends.md](../docs/backends.md).

Every backend reads the model shape from the file, so it is not tied to the
model in the table above. Models, formats, and context per backend:
[docs/limits.md](../docs/limits.md).
