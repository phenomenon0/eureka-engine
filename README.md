![Eureka Engine](docs/img/header.svg)

<p align="center">
  <img src="https://img.shields.io/badge/Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20iOS%20%7C%20Android%20%7C%20Web-supported-success?style=flat&colorA=2d2d2d&colorB=4a4a4a" alt="Platforms" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Metal-000000?style=flat&logo=apple&logoColor=white" alt="Metal" />
  <img src="https://img.shields.io/badge/Android-3DDC84?style=flat&logo=android&logoColor=white" alt="Android" />
  <img src="https://img.shields.io/badge/WebGPU-8A2BE2?style=flat&logo=googlechrome&logoColor=white" alt="WebGPU" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/API-OpenAI--compatible-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI-compatible API" />
  <img src="https://img.shields.io/badge/Models-GGUF-blue?style=flat" alt="GGUF" />
  <img src="https://img.shields.io/badge/Written%20in-Bend-orange?style=flat" alt="Written in Bend" />
</p>

<p align="center">
<em>Your models, your device, no cloud. One API for chat, vision, speech, and transcription.</em>
</p>

<br>

## Backends

<div align="center">

| Backend | Runs on | Chat speed | Guide |
| :---: | :---: | :---: | :---: |
| **CUDA** | NVIDIA, Linux | `38.8 tok/s` 27B | [cuda](backends/cuda/README.md) |
| **Metal** | macOS | `70.2 tok/s` 9B | [metal](backends/metal/README.md) |
| **iOS** | iPhone | `20.6 tok/s` 4B | [ios](backends/ios/README.md) |
| **Android** | Android phones | `3.2 s` per reply, 4B | [android](backends/android/README.md) |
| **WebGPU** | WebGPU browsers | `35.8 tok/s` 9B | [webgpu](backends/webgpu/README.md) |

</div>

All five backends pass all five verbs in the backend-parity battery (2026-10-21).
Windows runs a CPU build of the engine today. Devices, conditions, and gaps:
[docs/backends.md](docs/backends.md).

## Models

<div align="center">

| Backend | Qwen3.5 models | Context |
| :---: | :--- | :--- |
| **CUDA** | Any dense size, 0.8B to 27B, and any MoE (experts in VRAM, host RAM, or streamed from disk) | VRAM-bound; 145k-token prompt tested |
| **Metal** | Any dense size, and any MoE (experts resident or streamed from the file) | Buffer-bound; 153k-token prompt tested (0.8B, 2B, 35B-A3B) |
| **iOS** | Same engine as Metal (new sizes compiled, not yet run on a phone) | Buffer-bound |
| **Android** | Any dense size | Set at launch |
| **WebGPU** | Any size, dense or MoE, in 21 GGUF formats (F16 to IQ1_S) | About 2M tokens on the 0.8B; 153k-token prompt tested |

</div>

Each engine reads the shape from the model file and refuses, with the reason,
anything it can't run. MoE models bigger than the GPU run on CUDA by keeping
the experts in host RAM or reading only the chosen ones from the file each
token, and on Metal by mapping the experts from the file or copying only the
chosen ones each token. Sizes, formats, and limits per engine:
[docs/limits.md](docs/limits.md).

## What it does

<p align="center">
  <img src="docs/img/fig-five-verbs.svg" width="900" alt="Five panels, one per verb. chat: POST /v1/chat/completions, What is the capital of France? in, Paris. out. decide: POST /v1/decide, three choices (refund, escalate, ignore), refund out. see: POST /v1/see, a photo in, a box labeled cup out. hear: POST /v1/audio/transcriptions, audio in, turn on the lights out. speak: POST /v1/audio/speech, hello in, speech audio out." />
</p>

<p align="center"><em>What each verb takes in and gives back, and the model behind it.</em></p>

<div align="center">

| Verb | What you get | Model |
| :---: | :--- | :--- |
| **chat** | Text generation, streaming, tool calls, JSON output | Qwen3.5 family, on the Bend engine |
| **decide** | A labeled choice with probabilities | Decision head on the same model |
| **see** | Object detection on an image | D-FINE-N |
| **hear** | Speech to text | Whisper; Apple SpeechAnalyzer on iOS |
| **speak** | Text to speech | Kokoro-82M |

</div>

## Results

<div align="center">

| Metric | Score |
| :---: | :---: |
| **Decode speed vs llama.cpp (↑)**, 27B Q4_K_M | `1.011x` |
| **Prefill speed vs llama.cpp (↑)**, 3,011-token prompt | `0.953x` |
| **MoE decode vs llama.cpp Metal (↑)**, 35B-A3B Q4_K_S, M3 Ultra | `1.14x` |
| **Speculative decoding (↑)**, same tokens as greedy | `65.4 tok/s` |
| **Cold load from GGUF (↓)**, 27B | `17.9 s` |
| **Same `decide` answer on CUDA, Metal, WebGPU (↑)** | `30 / 30` |

</div>

## Install

Linux (x86-64, NVIDIA) or macOS (Apple silicon):

```sh
curl -fsSL https://github.com/phenomenon0/eureka-engine/releases/latest/download/install.sh | sh
```

The installer detects your platform, checks the download's sha256 before it
unpacks anything, and installs `llb` to `~/.local/bin`. To see what it would
do without installing, end the command with `sh -s -- --dry-run` instead of
`sh`.

> **Note:** The first release is not published yet. Until then, build from
> source with the quick start below.

## Quick start from source

### Step 1: Build the engine

```sh
LLB_BEND=BEND_CHECKOUT LLB_OUT=$PWD/build ./build.sh
```

`BEND_CHECKOUT` is the [pinned Bend compiler](docs/guides/run-on-cuda.md#pinned-compiler).
On a Mac, add `LLB_BACKEND=metal`.

### Step 2: Package it

```sh
tools/unify/bundle.sh 0.1.0 build && cd release/llb-linux-x86_64-cuda-0.1.0
```

### Step 3: Run a model

```sh
LLB_MODELS=MODELS_DIR ./llb run 27b "The capital of France is" -n 16
```

`MODELS_DIR` holds `Qwen3.8-27B-Q4_K_M.gguf`.

### Step 4: Call the API

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

Start the server with the [server quickstart](tools/sys1dual/README.md).

## Directory structure

```text
eureka-engine/
├── engine_qwen35_gpu.bend   # The engine, in Bend
├── kernels/                 # Bend kernels
├── host/                    # C host: GGUF reader, sampler, grammar, tool calls
├── backends/                # One folder per backend
├── tools/                   # Build, server (sys1dual/), CLI (llb/), packaging
├── vision/                  # Vision encoder
├── release/                 # Release templates
├── docs/                    # Guides and reference
├── tests/                   # Tests and fixtures
└── reports/                 # Reports behind the published numbers
```

## Documentation

<div align="center">

| Page | Description |
| :---: | :--- |
| [Backends](docs/backends.md) | Status, devices, and gaps for each backend |
| [Model and hardware limits](docs/limits.md) | Model sizes, formats, and context per engine |
| [Run on CUDA](docs/guides/run-on-cuda.md) | Full build and run guide |
| [HTTP API](docs/reference/http-api.md) | Endpoints, fields, streaming, errors |
| [Multimodal content](docs/reference/multimodal-content.md) | Images, audio, and video in requests |
| [Structured output](docs/guides/structured-output.md) | JSON schema, grammars, tool calls |
| [All docs](docs/README.md) | Full index |

</div>

## License

Apache License 2.0. See [LICENSE](LICENSE).

<sub>Figures drawn in the iso-glow style (MIT). Sources: <a href="docs/img/src/">docs/img/src/</a>.</sub>
