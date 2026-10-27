# Documentation

This directory holds the Eureka Engine documentation. Start with the
[repository README](../README.md) for build and run steps.

## Guides

| Page | Description |
|---|---|
| [Run on CUDA](guides/run-on-cuda.md) | Build and run the engine on Linux with an NVIDIA GPU. |
| [Build a MoE pack](guides/build-moe-pack.md) | Convert the Qwen 35B-A3B GGUF file to a pack with `gguf2pack`. |
| [Load a GGUF file directly](guides/load-gguf-directly.md) | Run the GPU engine from a GGUF file and a sidecar, without a pack. |
| [Structured output](guides/structured-output.md) | JSON schema, GBNF grammar, and tool calls on the server. |
| [Decision CLI](guides/decision-cli.md) | Run `clef-decide`, the local decision-head command. |
| [macOS CLI adapter](guides/macos-cli-adapter.md) | Run `llb` through the Mac adapter. Not tested. |

## Reference

| Page | Description |
|---|---|
| [HTTP API](reference/http-api.md) | Wire format of `llb-serve`, the OpenAI-compatible server. |
| [Multimodal content](reference/multimodal-content.md) | Image, audio, and video content in chat requests. |
| [Decide configuration](reference/decide-configuration.md) | Served configuration of the decide path and its signed bound. |
| [Module contract](reference/module-contract.md) | Module classes, attach points, and pack containers. |
| [Dual-mode residency](reference/dual-mode-residency.md) | Memory residency and queue policy of the dual-mode engine. |
| [Clean machine requirements](reference/clean-machine-requirements.md) | What a clean machine needs to reproduce the build. |

## Tool documentation

| Page | Description |
|---|---|
| [`tools/sys1dual/README.md`](../tools/sys1dual/README.md) | Server quickstart. |
| [`tools/metrics/README.md`](../tools/metrics/README.md) | Prometheus metrics and per-request timings. |
| [`tools/grammar/README.md`](../tools/grammar/README.md) | Grammar-constrained sampling. |
| [`tools/sampler/README.md`](../tools/sampler/README.md) | Sampler. |
| [`tools/ears/README.md`](../tools/ears/README.md) | Speech-to-text worker. |
| [`tools/tts/README.md`](../tools/tts/README.md) | Text-to-speech. |
| [`tools/glue/README.md`](../tools/glue/README.md) | Vision connector rows. |
| [`tools/mtp27/README.md`](../tools/mtp27/README.md) | MTP drafter pack for speculative decoding. |
| [`backends/README.md`](../backends/README.md) | One page per backend: CUDA, Metal, iOS, Android, WebGPU. |
| [`backends/metal/build-internals.md`](../backends/metal/build-internals.md) | How the Metal build reproduces the Mac product. |
| [`release/README-llb.md`](../release/README-llb.md) | The `llb` command. |

## Project

| Page | Description |
|---|---|
| [Backends](backends.md) | Status, measurements, and gaps for each of the five backends. |
| [Model and hardware limits](limits.md) | What each engine accepts, what blocks other sizes and smaller hardware, and the cheapest fixes. |
| [Branches](branches.md) | Branch naming convention. |
| [Documentation style](documentation-style.md) | Rules for file names and writing. |

## Evidence

The reports in [`reports/`](../reports/) back the numbers in these docs. They
cite raw logs that are kept in the project's lab archive, not in this
repository.

| Report | Covers |
|---|---|
| [`llb-unify1-final.md`](../reports/llb-unify1-final.md) | CUDA 27B release: decode, prefill, and cold-load numbers. |
| [`unify1-gates.md`](../reports/unify1-gates.md) | Gates for that release. |
| [`gguf2-final.md`](../reports/gguf2-final.md) | Direct GGUF load. |
| [`llb-mtp27-final.md`](../reports/llb-mtp27-final.md) | Speculative decoding on the 27B. |
| [`clef3-final.md`](../reports/clef3-final.md) | Decision CLI. |
| [`cli1-final.md`](../reports/cli1-final.md) | The `llb` command. |
| [`pack35-final.md`](../reports/pack35-final.md) | 35B-A3B pack conversion. |
| [`e5-08b/REPORT.md`](../reports/e5-08b/REPORT.md) | Server batching on the 0.8B. |
| [`webgpu-any-shape/REPORT.md`](../reports/webgpu-any-shape/REPORT.md) | WebGPU trunk on any Qwen3.5 model, format, and context length. |
| [`cuda-moe/REPORT.md`](../reports/cuda-moe/REPORT.md) | CUDA on any Qwen3.5 MoE, with experts in VRAM, host RAM, or streamed. |
