# Eureka Engine

Eureka Engine is the inference engine behind Eureka. It is written in
[Bend](https://github.com/bendlang/bend), reads GGUF model files, and has a
Metal backend (macOS) and a CUDA backend (Linux). llama.cpp is not part of the
serving path. The reports in this repository use llama.cpp only as a
reference to compare against.

## 1. What this is

The engine is one Bend program, [`engine_qwen35_gpu.bend`](engine_qwen35_gpu.bend).
It has kernels in [`kernels/`](kernels/) and a C host in [`host/`](host/). The
host holds the GGUF reader, the sampler, grammars and tool calls. A pinned,
patched Bend compiler turns the program into C:

- **CUDA:** a host binary. Its GPU kernels are compiled by NVRTC on first run.
  The patch stack is listed in [`tools/build_gpu.sh`](tools/build_gpu.sh).
- **Metal:** the same program with the Mac patches applied, built by
  [`backends/metal/build.sh`](backends/metal/build.sh).

The engine reads weights from the GGUF file directly (`LLB_GGUF`). Before that,
two tools write a small sidecar from the same file:

- [`tools/gguf2pack.c`](tools/gguf2pack.c) writes the tensor table.
- [`tools/llb/llb-tok.c`](tools/llb/llb-tok.c) writes the tokenizer.

`llb-serve` puts an OpenAI-compatible HTTP API in front of the engine. There is
one server for the CUDA line ([`tools/sys1dual/`](tools/sys1dual/README.md)) and
one for Metal ([`backends/metal/server/`](backends/metal/README.md)).

[`backends/`](backends/README.md) holds three more backends. Each has its own
README, and each has limits:

- **Android:** a CPU build of the engine.
- **WebGPU:** an engine in JavaScript and WGSL.
- **iOS:** only the build recipe and patches. The iOS engine source is not in
  this repository.

## 2. What it runs

### Architectures

| GGUF `general.architecture` | What it is | CUDA | Metal |
|---|---|---|---|
| `qwen35` | Qwen3.5 dense: hybrid gated-delta and full attention | Yes | Yes |
| `qwen35moe` | Qwen3.5 mixture of experts | Yes | Yes |
| `clef` | A Qwen3.5 trunk plus Eureka's CLEF decision head | Yes | Yes |

The sidecar step refuses any other architecture with
`architecture X: only qwen35 or clef (Qwen3.5 hybrid backbone) is packed`.

The shape is checked by structure, not by a list of model sizes. A dense model
needs:

- `head_dim` 256 and SSM state 128;
- query heads a whole multiple of the KV heads;
- `d_model` and `d_ff` multiples of 256, with `d_model` ≤ 8192.

The full rules are in [docs/limits.md](docs/limits.md).

Model tiers this repository records as run:

| Tier | CUDA | Metal |
|---|---|---|
| 0.8B | Yes | Yes |
| 4B | Yes | Yes |
| 9B | Yes | Yes |
| 27B | Yes (Qwen3.8-27B Q4_K_M) | Not recorded |

Also run: the 2B on both backends, and MoE Qwen3.5-35B-A3B Q4_K_S on Metal.

### Quantization formats (CUDA, Metal and the CPU engine)

Every model goes through `gguf2pack`. It refuses any tensor type that the
engines have no kernel for, and the error names the tensor.

| ggml type | Status |
|---|---|
| Q8_0, Q4_K, Q5_K, Q6_K | Kernels on CUDA and Metal. These are the formats in every tested file (Q4_K_M, Q4_K_S, Q8_0 GGUFs). |
| F32 | Norms and small tensors |
| Q3_K | `gguf2pack` packs it and there is a CUDA GEMV kernel. Not verified end to end in this repository. |
| Q2_0, PQ2_0 | This project's own 2-bit formats ([`tools/q20.py`](tools/q20.py)), not llama.cpp types |
| Output head | Q8_0, Q4_K, Q5_K or Q6_K |
| Token embedding | Q8_0, Q4_K or Q6_K |
| F16, BF16 | Refused for dense-model weights. Allowed for MoE experts. |
| **IQ4_XS, IQ4_NL, IQ3_S** | **No kernel. Refused.** |
| Every other IQ type (IQ1_*, IQ2_*, IQ3_XXS) | No kernel. Refused. |
| Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, TQ1_0, TQ2_0, MXFP4 | No kernel. Refused. |

**Consequence:** an Unsloth Dynamic quant (`UD-*`) is refused, because it mixes
IQ types into some layers. For example, the Qwen3.5-0.8B `UD-IQ2_XXS` file
stops at `blk.0.attn_gate.weight: ggml type 18 unsupported`, and the plain
`IQ4_XS` file at `type IQ4_XS unsupported for packing`
([receipt](receipts/port-20261106/quant-refusal.txt)). Use a llama.cpp K-quant
or Q8_0 file instead.

The WebGPU backend is a separate engine with different formats:

- Its trunk engine has kernels for 22 ggml types, including IQ4_XS, IQ4_NL and
  IQ3_S ([reports/webgpu-any-shape](reports/webgpu-any-shape/REPORT.md)).
- Its shipped 4B engine reads only F32, Q8_0, Q4_K, Q5_K and Q6_K.

## 3. Install a release (Linux, NVIDIA)

Release [v0.2.0](https://github.com/phenomenon0/eureka-engine/releases/tag/v0.2.0)
has one bundle: Linux x86-64 with an NVIDIA GPU (Ampere or newer, driver R580+).
NVRTC is bundled, so no CUDA toolkit is needed. The bundle has the engine, the
`llb` front end, `gguf2pack` and `llb-tok`. `llb-serve` is not in it: build that from
source (below).

```sh
curl -fsSL https://github.com/phenomenon0/eureka-engine/releases/latest/download/install.sh -o install.sh
sh install.sh            # read it first; it checks the sha256 before unpacking
llb doctor
llb run /path/to/Qwen3.5-4B-Q8_0.gguf "The capital of France is"
```

The installer, `llb doctor` and `llb run` were run from the published manifest's
contents on an RTX 3090 ([receipts](receipts/release-0.2.0/README.md)). There is no
macOS, CUDA 12 or CPU bundle yet. On those, the installer says so and exits; build
from source instead.

## 4. How to build

### The pinned Bend compiler (all backends)

The build uses a Bend compiler snapshot kept on the branch `pins/bend-240ecef4`.
Fetch it after cloning:

```sh
git clone https://github.com/phenomenon0/eureka-engine && cd eureka-engine
git fetch origin pins/bend-240ecef4:pins/bend-240ecef4
```

When `LLB_BEND` does not point at a Bend checkout, `build.sh` uses this branch.
It checks the branch's tree id against `BEND_TREE`. See
[Pinned compiler](docs/guides/run-on-cuda.md#pinned-compiler).

Common tools: `git`, `patch`, [`bun`](https://bun.sh), `clang`, and `sha256sum`
or `shasum`.

### macOS (Apple silicon, Metal)

Also needed: Xcode command-line tools and Homebrew OpenBLAS
(`brew install openblas`).

```sh
LLB_BACKEND=metal ./build.sh               # -> build/metal/sys1mac (engine)
mkdir -p build/bin
clang -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread \
  backends/metal/server/src/serve.c backends/metal/server/src/json.c backends/metal/server/src/speak_route.c \
  -o build/bin/llb-serve                   # HTTP server
clang -std=c11 -O2 -Wall -o build/bin/gguf2pack tools/gguf2pack.c
clang -O2 -o build/bin/llb-tok tools/llb/llb-tok.c
```

The build checks the emitted C against a pinned hash and stops if they differ.
These exact commands exit 0 on a Mac Studio with macOS 27, arm64
([build](receipts/port-20261106/build-metal-port.txt),
[tools](receipts/port-20261106/build-mac-served-path.txt)).

### Linux (x86-64, NVIDIA, CUDA)

Also needed:

- CUDA 12 or 13: headers in `CUDA_HOME` (default `/usr/local/cuda`), plus
  `libcuda` and `libnvrtc`;
- clang 19 or newer, because the emitted C uses `#embed`;
- OpenBLAS (`libopenblas-dev`).

```sh
./build.sh                                 # -> build/llb-engine, build/gguf2pack, build/llb-probe
OUT=build/serve LLB_ENGINE_C=build/llb-engine.c LLB_ENGINE_GPU=build/llb-engine.gpu \
  tools/sys1dual/build-serve.sh            # -> build/serve/llb-serve, build/serve/llb-engine
```

The first run of the engine compiles its kernels and writes `llb-engine.gpu`
beside it. Run the engine once before `build-serve.sh`.

The full guide is [Run on CUDA](docs/guides/run-on-cuda.md). The CUDA numbers
below are from an RTX 3090.

These commands were run from a fresh clone on an RTX 3090 (Fedora 43, CUDA 13.1,
clang 21). The engine ran the Qwen3.5-2B, 4B and 9B, and `llb-serve` served the 4B
([record](docs/records/cuda-proof-2026-11-06.md),
[receipts](receipts/cuda-proof-20261106/README.md)).

### CPU only (Linux)

The Android engine also builds for the host (gcc with OpenMP):

```sh
bash backends/android/engine/build.sh host build/cpu    # -> build/cpu/kev-engine-host
```

It reads a full pack, not a sidecar:

```sh
gguf2pack MODEL.gguf PACK
llb-tok extract MODEL.gguf PACK
```

This build exits 0 in an Ubuntu 24.04 container, and the binary runs the
Qwen3.5-0.8B
([build](receipts/port-20261106/build-cpu-host-linux.txt),
[run](receipts/port-20261106/run-cpu-host-08b.txt)). The CPU engine is slow.
[docs/backends.md](docs/backends.md) has the numbers it records.

## 5. How to run

### Make the sidecar (once per GGUF)

```sh
build/bin/llb-tok extract MODEL.gguf SIDE           # tokenizer.bin, chat.flags
build/bin/gguf2pack MODEL.gguf --sidecar-only SIDE  # manifest.json, table.bin
```

### Run the engine directly (macOS)

```sh
export LLB_GGUF=MODEL.gguf
build/metal/sys1mac --gpu 4GB --threads 8 -- SIDE $(build/bin/llb-tok enc SIDE "The capital of France is") --steps 16 > out.txt
build/bin/llb-tok dec SIDE $(grep -E '^[0-9]+$' out.txt)
```

The engine prints the generated token ids. The last line decodes them. This
works on a stock Qwen3.5 GGUF
([receipt](receipts/port-20261106/run-metal-cli-08b.txt)).

### The served path

`llb-serve` runs the engine as a child process (the `SYS1` mode) and serves an
OpenAI-compatible API. The default port is **8080** (`--port`).

**Required environment:** `LLB_GGUF`, the path to the model's GGUF, and `TMPDIR`, a
scratch directory the engine owns.

Everything after `--` goes to the engine, in this order:

1. the engine's own flags;
2. a second `--`;
3. the sidecar directory, `0`, and `--steps N`. `--steps` sets the KV capacity.

```sh
export LLB_GGUF=MODEL.gguf
export TMPDIR=${TMPDIR:-$(mktemp -d)}   # the engine needs TMPDIR set; macOS sets it, most Linux systems do not
# macOS
build/bin/llb-serve --port 8080 --engine build/metal/sys1mac -- --gpu 12GB --threads 8 -- SIDE 0 --steps 8192
# Linux / CUDA
build/serve/llb-serve --port 8080 --engine build/serve/llb-engine -- --threads 16 -- SIDE 0 --steps 4096

curl -s localhost:8080/health      # 503 while loading, then 200
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Routes:

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/decide`;
- the senses: `/v1/see`, `/v1/hear` and `/v1/speak` (CUDA also has
  `/v1/audio/*`). These need their sidecar workers.

A reply ends at the model's end-of-turn token with `finish_reason` `"stop"`.
The CUDA engine has no early stop, though: it runs all `max_tokens` steps and the
server drops the extra text, so set `max_tokens` to what you need. `stop` strings
and `logprobs` are not supported.

Details are in [HTTP API](docs/reference/http-api.md) and the
[server quickstart](tools/sys1dual/README.md).

**The served engine expects the CLEF decision head:**

- **CUDA:** a stock Qwen3.5 GGUF serves with `LLB_SYS1_NOHEAD=1`. Chat works;
  `decide` is refused.
- **macOS:** the Metal engine has no such switch. A stock GGUF makes it exit at
  load with `clef: missing/corrupt head tensors`
  ([receipt](receipts/port-20261106/run-metal-served-stock-gguf.txt)). On a Mac,
  serve a `clef` GGUF, or run a stock model with the engine directly as shown
  above.

**Opt-in switches.** Each one changes numerics or memory. All are off by default.

| Switch | Where | What it does |
|---|---|---|
| `--prefill fast` | CUDA `llb-serve` | Fast prefill |
| `--prefix-cache` | CUDA `llb-serve` | Prefix cache |
| `LLB_ATTN_SK8=1` | CUDA engine | Split-key decode attention. `--canon` forces it off. |
| `LLB_MOE_DP4A=1` | CUDA engine | int8 MoE experts |

## 6. What is not in this repository

- **The Eureka app.** The desktop, iOS and Android apps, and the Android app's
  Kotlin, WebView and sense glue, live in the app repository.
- **The iOS engine source.** [`backends/ios`](backends/ios/README.md) holds the
  recipe and patches. The engine tree it builds from is private.
- **The Metal server's engine rebuild.** `backends/metal/server/build.sh` needs
  a separate Mac engine tree (`EUREKA_ENGINE_SRC`). The `llb-serve` build above
  does not.
- **Development history.** This repository is an export of a private working
  repository. Its lane logs, most per-experiment receipts, and its experiment
  branches are not here. That includes the receipts behind some figures in the
  backend READMEs and `docs/`. [`PORT.md`](PORT.md) lists what was carried over
  on 2026-11-06 and what was left behind.
- **Model files.** No GGUF is distributed here. A `clef` GGUF is not published
  in this repository either.

## 7. Measured numbers

Only numbers with a receipt in this repository are listed here.

**CUDA decode vs llama.cpp.** RTX 3090, CUDA 13, llama.cpp `9d57ce45`. Median of
3 interleaved rounds, exact tier, greedy decode. Sources:
[reports/cuda-catchup](reports/cuda-catchup/REPORT.md),
[`receipts/speed/SPEED.txt`](reports/cuda-catchup/receipts/speed/SPEED.txt).

| Model | This engine | llama.cpp CUDA | Ratio |
|---|---|---|---|
| Qwen3.5-0.8B Q4_K_M | 483.4 tok/s | 463.1 tok/s | 1.04 |
| Qwen3.5-2B Q4_K_M | 299.5 tok/s | 329.4 tok/s | 0.91 |
| Qwen3.5-4B Q4_K_M | 135.3 tok/s | 174.5 tok/s | 0.78 |
| Qwen3.5-9B Q4_K_M | 91.7 tok/s | 114.9 tok/s | 0.80 |

**Metal MoE vs llama.cpp Metal.** Qwen3.5-35B-A3B Q4_K_S (Unsloth, not a
Dynamic quant). Mac Studio M3 Ultra, llama.cpp `e70802a01`. Median of 3 rounds.
Sources: [reports/metal-catchup](reports/metal-catchup/REPORT.md),
[`receipts/speed.txt`](reports/metal-catchup/receipts/speed.txt).

| | This engine | llama.cpp Metal | Ratio |
|---|---|---|---|
| Decode | 98.5 tok/s (10.151 ms a token) | 86.10 tok/s (tg128) | 1.14 |
| Prefill, 4,096 tokens | 1,250.7 tok/s | 1,706.95 tok/s (pp4096) | 0.73 |

Both measurements predate the 2026-11-06 port. The port's new features
(split-key attention, fast-prefill and prefix-cache flags) are off by default,
so these defaults are unchanged. The port's own figures were measured in the
private repository, so they are not listed here.

## License

Apache License 2.0. See [LICENSE](LICENSE).
