# Backends

| Backend | Devices | Source | Result | Page |
|---|---|---|---|---|
| CUDA | NVIDIA, Linux | This repository | 38.8 tok/s, 27B, RTX 3090 | [cuda](cuda/README.md) |
| Metal | macOS | This repository + `bend-app` | 70.2 tok/s, 9B, M3 Ultra | [metal](metal/README.md) |
| Metal | iOS | `bend-app` | 20.6 tok/s, 4B, iPhone 18 Pro Max | [ios](ios/README.md) |
| CPU | Android | `ship/android/voice/ship-wip/app` branch | 3.2 s per short reply, 4B, Pixel 6a | [android](android/README.md) |
| WebGPU | Browser | `bend-app` | 35.8 tok/s, 9B, M3 Ultra | [webgpu](webgpu/README.md) |

All five pass `chat`, `decide`, `see`, `hear`, and `speak` (2026-10-21).
Conditions and receipts: [docs/backends.md](../docs/backends.md).

`bend-app` is `github.com/phenomenon0/bend-app`.
