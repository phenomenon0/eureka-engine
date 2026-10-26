# Backends

Eureka Engine runs on five backends. Each backend serves the same five verbs
through the same HTTP contract: `chat`, `decide`, `see`, `hear`, and `speak`.
This page lists what runs on each backend, what was measured, and the known
gaps.

## Status

The backend-parity battery grades every backend on every verb: a real HTTP
route, the model that ships for that backend, a declared quality tolerance, a
performance floor, a stability run, and a clean teardown. On 2026-10-21, all
25 cells passed. 24 cells are at `final` maturity. `metal-macos` `chat` is at
`v1`.

| Backend | Tier | Device tested | Engine | chat | decide | see | hear | speak |
|---|---|---|---|---|---|---|---|---|
| CUDA | Native | RTX 3090 (24 GB), Linux | Bend, emitted as C + CUDA | Pass | Pass | Pass | Pass | Pass |
| Metal, macOS | Native | Mac Studio, M3 Ultra, 96 GB | Bend, emitted as C + Metal (`sys1mac`) | Pass (v1) | Pass | Pass | Pass | Pass |
| Metal, iOS | Native | iPhone 18 Pro Max, 12 GB | Bend Metal engine in the iOS app | Pass | Pass | Pass | Pass | Pass |
| Android | Native | Pixel 6a, Tensor G1, 5.46 GB | Bend CPU engine (`libbendengine.so`) in the Android app | Pass | Pass | Pass | Pass | Pass |
| WebGPU | Preview | Browser tab on the M3 Ultra; cross-check on the RTX 3090 | Hand-written WGSL port | Pass | Pass | Pass | Pass | Pass |

Native backends are product targets. The WebGPU backend is a test bed and a
zero-install preview. It is not the reference implementation.

## Measured chat performance

Each row uses the model that ships for that backend. The rows are not
comparable to each other: model size, quantization, prompt, and metric differ.

| Backend | Model | Result | Metric | Comparison |
|---|---|---|---|---|
| CUDA | Qwen3.8-27B Q4_K_M | 38.8 tok/s | Decode, `LLB_Q4_DP4A=9`, 3 interleaved rounds | 1.011x [1.009..1.012] of llama.cpp `tg128` in the same rounds |
| CUDA | clef-flash Q4_K_S (9B) | 95.1 tok/s | Decode median, answers up to 64 tokens | No same-model llama.cpp result in the receipt |
| Metal, macOS | clef-flash Q4_K_S (9B) | 70.2 tok/s | Server decode, 128 tokens, 1 warm-up and 10 runs | llama.cpp `e70802a01` cannot load this GGUF file |
| Metal, iOS | Kev-4B Q4_K_M | 20.6 tok/s | End to end, over loopback HTTP | None |
| Android | Kev-4B Q4_K_M | 3,176.4 ms | End-to-end wall time, 19-token prompt, 2-token answer, median of 10 | The earlier Pixel 6a baseline was about 3.3 s |
| WebGPU | clef-flash Q4_K_S (9B) | 35.8 tok/s | Decode median, in-tab, M3 Ultra | 16.4 tok/s on the RTX 3090 through Vulkan/ANGLE |

The CUDA 27B row is from the `unify1` release
([`reports/llb-unify1-final.md`](../reports/llb-unify1-final.md)). The other
rows are from the backend-parity receipts.

## Cross-backend agreement

For `decide`, the CUDA, Metal macOS, and WebGPU backends return the same
answer on 30 of 30 questions.

## What runs on the Bend engine

The Bend engine runs the language model: `chat` and `decide`. The other verbs
use established models and runtimes, which the server routes to:

| Verb | Model | Runtime examples |
|---|---|---|
| `see` | D-FINE-N | ONNX Runtime; native CUDA on the CUDA backend |
| `hear` | Whisper base.en or tiny.en | whisper.cpp with ggml-Metal or GGML_CUDA; Apple SpeechAnalyzer on iOS |
| `speak` | Kokoro-82M | ONNX Runtime |

The runtime for each cell is recorded in its receipt.

## Where each backend is built

See [`backends/`](../backends/README.md). Each backend has a page with its
source location and build commands.

## Known gaps

- Android runs on the CPU. The Vulkan path for Android is not built yet. The
  battery labels the cells `android-vulkan`, but every Android receipt
  declares the CPU route.
- Vulkan on Linux is parked. As of 2026-10-19, SPIR-V output is proven, coverage
  is 26 %, and there is no runtime.
- There is no CPU/WASM build of the engine.
- Windows: a native CPU build (`qwen35-cli.exe`, llvm-mingw, x86-64) answers
  correctly in a Windows Server 2025 VM, at 2 tokens in 128 s on 4 vCPUs. The CUDA build does not
  support Windows or WSL2. The WebGPU backend is not yet tested on Windows.
- The Metal macOS `see` cell runs D-FINE-N on the ONNX Runtime CPU execution
  provider, not on Metal.
- The iOS receipts come from a debug build with Swift optimization `-O`.
- The Android result is on one mid-tier phone. The iOS result is on one
  flagship phone.

## Receipts

The battery, its gate definitions, and the 25 cell receipts are kept in the
project's lab archive, not in this repository. Each receipt records the route,
the model file and its sha256, the device, and the measured result.
