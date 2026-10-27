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

## Models and context

Every backend reads the model's shape from the file instead of matching a
list of known models. Each one was checked against llama.cpp on sizes it did
not run before, and against its own earlier build on the sizes it did.

| Backend | Qwen3.5 models | Weight formats | Longest prompt tested | Context bound |
|---|---|---|---|---|
| CUDA | Any dense size (0.8B, 2B, 4B, 9B, 27B, clef, Kev); any MoE (`qwen35moe`: any expert count, top-k and expert widths), checked on synthetic files only | GGUF K-quants (Q4_K, Q5_K, Q6_K) and Q8_0; MoE router F32 | 145,115 tokens, 0.8B, RTX 3090 | VRAM (f32 KV allocated by position, up to 2^26) |
| Metal, macOS | Any dense size from a quant pack; any MoE (`qwen35moe`: 35B-A3B run, any expert count / top-k / widths) | Quant pack (K-quants); MoE experts F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, mapped from the GGUF | 153,031 tokens, 0.8B and 2B, M3 Ultra; 153,352 tokens, 35B-A3B MoE (exact prefill, 4.1 h) | Largest Metal buffer (f32 KV, up to 2^27 slots); MoE experts streamed past the GPU working set |
| Metal, iOS | Same engine as macOS | Quant pack | Compiled, not run at new sizes | Device's largest buffer |
| Android | Any dense pack | Pack (F32, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K) | Launch-time capacity; prompts up to 16,384 | `--decide CTX` |
| WebGPU | Any size, dense or MoE (35B-A3B, 122B-A10B, 397B-A17B), any head, state, and conv size | 21 GGUF formats, F32 to IQ1_S | 153,352 tokens, 0.8B, on an RTX 3090 and an M3 Ultra | GPU binding limit for one layer's f16 KV (about 2 million tokens for the 0.8B) |

A model must still fit in memory: no backend offloads dense weights. The CUDA
MoE path keeps the routed experts in VRAM, in pinned host RAM, or streams the
chosen experts from the mmapped GGUF, picked from the free memory
([reports/cuda-moe](../reports/cuda-moe/REPORT.md)). Details,
remaining refusals, and the cheapest next fixes:
[Model and hardware limits](limits.md). WebGPU receipts:
[reports/webgpu-any-shape](../reports/webgpu-any-shape/REPORT.md).

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
