# Receipts: the llama-bend port, 2026-11-06

These receipts back [`PORT.md`](../../PORT.md) and the build and run commands in
the top-level README. Each file holds the exact command and its exit code.
"base" is this branch's parent `f14c1f3`, before the port. "port" is the
branch after it.

Machine: Mac Studio, Apple silicon (arm64), macOS 27.0, Apple clang, bun from
Homebrew. The Linux rows ran in Docker on the same Mac (Ubuntu 24.04 arm64, clang
18 / 19); the images are `scripts/Dockerfile.*`. No NVIDIA GPU was used for
anything here.

Local paths in these files were replaced by `$REPO` (this checkout) and
`$MODELS` (a local model directory).

## Build

| What | Command | Result | Receipt |
|---|---|---|---|
| Metal engine (`sys1mac`), base | `LLB_BACKEND=metal LLB_OUT=… ./build.sh` | exit 0; emitted C `5a766ce4`, the pinned product hash | `build-metal-baseline.txt` |
| Metal engine (`sys1mac`), port | same | exit 0; emitted C `5a766ce4` and binary `8809576f`, both identical to base | `build-metal-port.txt` |
| Mac served-path tools | Metal `llb-serve`, `gguf2pack`, `llb-tok` (clang) | exit 0 ×3 | `build-mac-served-path.txt` |
| CUDA engine C emit, base / port | `BEND_ROOT=… BEND_BUILD=… tools/build_gpu.sh engine_qwen35_gpu.bend -o llb-engine.c` | exit 0 / exit 0. The port applies 66 patches at `--fuzz=0`, including `payload-sk8add.patch` | `emit-cuda-baseline.txt`, `emit-cuda-port.txt` |
| CUDA engine C, compile only | clang-19, `tools/remap_cc.sh`'s flags, `-c`, CUDA 12.8 headers (NVIDIA pip wheels). No link: there is no libcuda here | base exit 0, port exit 0. The port adds `llb_sk8_launch` | `cuda-engine-compile-{base,port}.txt`, `scripts/cuda-compile.sh` |
| CUDA kernels through NVRTC | the emitted C as NVRTC source with `gpu_make`'s options, `--gpu-architecture=sm_86` | base and port `NVRTC_SUCCESS`. The port's PTX defines `llb_attn4_sk8` and `llb_attn4_sk8_combine`. The NVRTC log is the same length (169,461 bytes) for both | `nvrtc-sm86-{base,port}.txt`, `scripts/nvrtc-check.c` |
| Request objects, `gguf2pack`, `llb-serve` (CUDA line, Linux) | `build.sh` step 4 objects, `-Wall -Wextra`; `build-serve.sh`'s `llb-serve` line | base exit 0, port exit 0, no warnings | `linux-host-{base,port}.txt`, `scripts/linux-host-build.sh` |
| SK8 host dispatch, isolated | `scripts/sk8-host-harness.c` with stub CUDA types, `-Wall -Wextra -Werror` (the command is in its header) | exit 0. Guard: enabled only for `LLB_ATTN_SK8=1`; fits only when dT ≤ 17 and ≤ 128 heads; 2 launches | `scripts/sk8-host-harness.c` |

The CUDA engine was not linked or run: this Mac has no NVIDIA driver.

## Run

| What | Result | Receipt |
|---|---|---|
| `sys1mac` CLI on stock Qwen3.5-0.8B Q4_K_M (sha256 `bd258782…`), 5 prompt ids, 16 steps | exit 0, 16 ids, decodes to text | `run-metal-cli-08b.txt` |
| Metal `llb-serve` + `sys1mac` on the same stock GGUF | The engine exits at load: `clef: missing/corrupt head tensors`. The served (SYS1) engine needs a GGUF with the CLEF decision head | `run-metal-served-stock-gguf.txt`, `scripts/metal-serve-smoke.sh` |
| `gguf2pack --sidecar-only` on Qwen3.5-0.8B IQ4_XS and on Unsloth UD-IQ2_XXS | exit 1 for both: `type IQ4_XS unsupported for packing`, `ggml type 18 unsupported` | `quant-refusal.txt` |

## Tests

| Suite | Base | Port | Receipt |
|---|---|---|---|
| `tests/inference/run-host.sh` (macOS): JSON parser sanitizer ×2; Metal and CUDA server regressions ×4; cancellation ×2; Android JSON hardening ×1; WebGPU worker tests ×6 | exit 0, 15 / 15 | exit 0, 15 / 15 | `tests-baseline.txt`, `tests-port.txt` |
| `tests/mm/run.sh`, Linux, 2 runs each | unit 92/94; e2e 41/42 in run 1, 13 then timeout in run 2 | unit 92/94; e2e 41/42 in both runs | `tests-mm-linux-{base,port}.txt`, `tests-mm-linux-2x.txt` |
| `tests/mm/run.sh`, macOS (Darwin compat header), 3 runs each | unit 92/94; e2e 13 then timeout | unit 92/94; e2e 13 then timeout | `tests-mm-mac-3x.txt`, `tests-mm-{baseline,port}.txt` |

Why `tests/mm` is not all green, on either side:

- The template-parity step was skipped. It needs `transformers` and the model's
  chat template (`MM_TEMPLATE`).
- `MM_COCO` was a 640×480 stand-in, because the real COCO image is not on this
  machine. Unit checks 42 and 43 pin the real file's byte size and sha256, so
  they fail with the stand-in.
- One e2e check, "9.3 MiB base64 body over the 8 MiB body cap → 413", fails on
  base and port alike. The server answers 408, or the client times out first.
  This was not investigated; the port does not touch that code.
