# Vulkan backend

The Vulkan backend runs both Eureka Engine stacks on any Vulkan 1.2+ GPU
without CUDA or Metal: AMD Radeon (`RADV` / Windows), Intel Arc (`ANV` /
Windows), and NVIDIA Vulkan.

- **LLM stack (`engine_vulkan`)**: accelerates `tools/qwen35_engine.c`
  `LLB_GEMV_ROWS` using `DEVICE_LOCAL` VRAM weight slabs (`<= 1 GiB` per slab)
  and SPIR-V 1.3 shaders (`F32`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`). Source:
  `backends/vulkan/llb_vulkan_gemv.c`, `backends/vulkan/llb_vulkan_shaders.wgsl`,
  and `backends/vulkan/llb_vulkan_spv.h`.
- **Creative Studio stack (`eureka-creative-serve`, `ec-bench`)**: accelerates
  BF16/FP16/INT8/Q8_G64 GEMM, implicit-GEMM 3x3 Conv2D, and FlashAttention-2
  through a lazy-coherent `DEVICE_LOCAL` VRAM mirror and a 256-slot batched
  command buffer. Source: `creative/backends/ec_backend_vulkan.c`,
  `creative/backends/ec_vulkan_shaders.wgsl`, and
  `creative/backends/ec_vulkan_spv.h`.

For the optimization roadmap to close the remaining throughput gap against CUDA,
see [Vulkan turbo path](turbo-path.md).

## Before you begin

Install the following build and runtime dependencies:

- **Linux**: C11 compiler (`cc` or `clang`), Vulkan headers and loader
  (`vulkan-headers`, `vulkan-loader`), `libopenblas.so.0`, and a Vulkan 1.2+
  ICD (`mesa-vulkan-drivers` for AMD/Intel or the NVIDIA Vulkan ICD).
- **Windows** (build script integrated; Windows GPU execution not yet tested on
  Windows hardware): LLVM `clang` for Windows and the LunarG Vulkan SDK
  (`VULKAN_SDK` or `C:\VulkanSDK\*`).

## Build

### Linux

1. Build the Vulkan LLM engine:

```sh
OUT=/tmp/eureka-vulkan-llm bash backends/vulkan/build.sh
```

2. Build the Vulkan Creative Studio engine:

```sh
EC_BACKEND=vulkan OUT=/tmp/eureka-creative-vulkan bash creative/build.sh
```

3. Or build and stage all Linux backends (`cuda/`, `vulkan/`, `cpu/`) together:

```sh
bash backends/linux/build.sh
```

### Windows

Run the unified Windows build script (untested on Windows Vulkan hardware):

```sh
pwsh -File backends/windows/build.ps1
```

Pass `-VulkanSdk PATH` if the Vulkan SDK is outside `VULKAN_SDK` and
`C:\VulkanSDK`, or `-NoVulkan` to skip the Vulkan lane.

## Run

Run the LLM engine against a GGUF model directly:

```sh
LLB_GGUF=MODEL_PATH /tmp/eureka-vulkan-llm/engine_vulkan --prompt "Hello" --steps 32
```

Run the Creative verification suite on a specific Vulkan GPU:

```sh
EC_VULKAN_DEVICE=amd /tmp/eureka-creative-vulkan/ec-bench
```

## Environment variables

| Variable | Applies to | Description |
|---|---|---|
| `LLB_VULKAN_DEVICE` | `engine_vulkan` | Case-insensitive substring matched against `VkPhysicalDeviceProperties.deviceName` (for example `amd`, `intel`, `nvidia`). Falls back to `EC_VULKAN_DEVICE` if unset. |
| `LLB_VULKAN_OFF` | `engine_vulkan` | Set to `1` to disable Vulkan initialization and run on CPU threads as an oracle. |
| `EC_VULKAN_DEVICE` | Creative Vulkan | Case-insensitive substring matched against `deviceName` to select the Vulkan physical device. |
| `EC_VULKAN_NO_MIRROR` | Creative Vulkan | Set to `1` to disable the `DEVICE_LOCAL` VRAM mirror on discrete GPUs and use host-visible buffers directly. |

## Measured baseline

Recorded at commit `a4ca66248cd7638055c4e4ef12ee16b02eeb6102` on Linux
(Ryzen 7 7700X with 2-CU Radeon RDNA2 iGPU `RADV RAPHAEL_MENDOCINO`, and
NVIDIA GeForce RTX 3090 24 GB):

- **LLM (`Qwen3.5-4B-Q4_K_M.gguf`, 426/426 tensors, 2.54 GiB VRAM)**:
  - RTX 3090 Vulkan HTTP (`llb-serve`, 23 prompt / 12 decode tokens):
    `21.337 tok/s` prefill (`1077.932 ms`), `18.409 tok/s` decode (`597.535 ms`).
  - RTX 3090 Vulkan CLI (6 prompt + 16 decode steps): `1034 ms` wall time
    (`11.72x` vs 16-thread CPU oracle at `12115 ms`; `16/16` exact token match).
  - AMD RDNA2 2-CU iGPU Vulkan CLI: `14206 ms` wall time, `16/16` exact tokens,
    and `32/32` identical top-2 float logit bits compared to RTX 3090 Vulkan.
- **Creative Studio (`ec-bench` 46/46 PASS on RTX 3090 Vulkan v2)**:
  - BF16 GEMM (`G-C5`): `7.280 TFLOPS` (`18.88 ms`, `rel=6.75e-08`).
  - FP16 GEMM (`4096x3072x3072`): `7.228 TFLOPS` (`10.70 ms`).
  - INT8 W8A8 GEMM (`4096x3072x3072`): `7.375 TOPS` (`10.48 ms`).
  - FlashAttention-2 BF16 (`H=12, L=4096, D=128`): `0.497 TFLOPS` (`207.31 ms`).
  - `FLUX.2-klein-4B` (`256x256`, 4 steps, seed 42): `28265.44 ms` cold
    (`dit_ms=12747.3`, `vae_ms=1731.2`, `text_enc_ms=6047.8`).

## Limitations

- LLM decode submits one command buffer and fence wait per GEMV and runs
  RMSNorm, RoPE, Gated Delta-Net, attention, and SwiGLU on the host CPU.
- LLM prefill runs token-by-token GEMV rather than batched GEMM.
- Creative Studio runs portable FP32-SIMT SPIR-V 1.3 compute shaders without
  `VK_KHR_cooperative_matrix`, and per-layer glue ops synchronize activations
  back to the host CPU.
- Windows Vulkan execution is wired in `backends/windows/build.ps1` but has not
  been run on Windows hardware yet.
- Android Vulkan is not built yet; Android still ships the CPU engine.

See [Vulkan turbo path](turbo-path.md) for the staged plan to remove each of
these bottlenecks.
