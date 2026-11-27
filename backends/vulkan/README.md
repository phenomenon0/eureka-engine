# Vulkan backend

The Vulkan backend runs both Eureka Engine stacks on any Vulkan 1.2+ GPU
without CUDA or Metal: AMD Radeon (`RADV` / Windows), Intel Arc (`ANV` /
Windows), and NVIDIA Vulkan.

- **LLM stack (`engine_vulkan`)**: runs a single-submit GPU-resident 32-layer
  forward graph (`DEVICE_LOCAL` `<= 1 GiB` weight slabs, static per-slab
  descriptor sets, 64-byte push constants, disk `VkPipelineCache`, dedicated
  transfer queue) with runtime-probed Vulkan 1.3 wave32 subgroups,
  `VK_KHR_shader_integer_dot_product` (`dp4a`), and `BM=8` batched prefill
  (`F32`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`), falling back to portable SPIR-V 1.3
  on Vulkan 1.2 drivers. Source: `backends/vulkan/llb_vulkan_gemv.c`,
  `backends/vulkan/shaders/`, and `backends/vulkan/llb_vulkan_turbo*_spv.h`.
- **Creative Studio stack (`eureka-creative-serve`, `ec-bench`)**: runs
  GPU-resident SPIR-V 1.3 glue shaders (`ec_glue.comp`) and runtime-probed
  `VK_KHR_cooperative_matrix` Tensor Core / WMMA / XMX kernels
  (`ec_coopmat.comp`) for BF16/FP16/INT8/Q8_G64 GEMM, implicit-GEMM 3x3 Conv2D,
  and 4-warp register-accumulator FlashAttention-2, backed by a disk
  `VkPipelineCache`, caching VRAM allocator, and automatic Tier-0 SPIR-V 1.3
  fallback on GPUs without cooperative matrices. Source:
  `creative/backends/ec_backend_vulkan.c`, `creative/backends/shaders/`, and
  `creative/backends/ec_vulkan_*_spv.h`.

For the full five-phase architecture (`T1` through `T5`), driver invariants, and
paired v1/v2/Turbo/CUDA benchmark tables, see [Vulkan turbo path](turbo-path.md).

## Before you begin

Install the following build and runtime dependencies:

- **Linux**: C11/C++17 compiler (`cc`/`c++` or `clang`/`clang++`), Vulkan
  headers and loader (`vulkan-headers`, `vulkan-loader`), `libopenblas.so.0`,
  and a Vulkan 1.2+ ICD (`mesa-vulkan-drivers` for AMD/Intel or the NVIDIA
  Vulkan ICD).
- **Windows** (build script integrated; Windows GPU execution not yet tested on
  Windows hardware): LLVM `clang` for Windows and the LunarG Vulkan SDK
  (`VULKAN_SDK` or `C:\VulkanSDK\*`).

Precompiled SPIR-V headers are committed in the repository. Recompiling shader
sources via `backends/vulkan/compile_spv.py` requires `libshaderc_shared.so.1`
only when editing `.comp` shaders.

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

Run the LLM engine against a GGUF model and sidecar directory:

```sh
LLB_GGUF=MODEL_PATH /tmp/eureka-vulkan-llm/engine_vulkan --threads 16 -- SIDECAR_DIR 9707 11 1245 525 498 30 --steps 16
```

Run the Creative verification suite on a specific Vulkan GPU:

```sh
EC_VULKAN_DEVICE=nvidia /tmp/eureka-creative-vulkan/ec-bench
```

## Environment variables

| Variable | Applies to | Description |
|---|---|---|
| `LLB_VULKAN_DEVICE` | `engine_vulkan` | Case-insensitive substring matched against `VkPhysicalDeviceProperties.deviceName` (for example `amd`, `intel`, `nvidia`). Falls back to `EC_VULKAN_DEVICE` if unset. |
| `LLB_VULKAN_OFF` | `engine_vulkan` | Set to `1` to disable Vulkan initialization and run on CPU threads as an oracle. |
| `LLB_VULKAN_EXACT` | `engine_vulkan` | Set to `1` to force exact FP32 subgroup dequantization instead of `dp4a` Q8_1 dot products. |
| `LLB_VULKAN_NO_GRAPH` | `engine_vulkan` | Set to `1` to disable the single-submit GPU graph and run per-GEMV fallback mode. |
| `EC_VULKAN_DEVICE` | Creative Vulkan | Case-insensitive substring matched against `deviceName` to select the Vulkan physical device. |
| `EC_VULKAN_NO_COOPMAT` | Creative Vulkan | Set to `1` to disable `VK_KHR_cooperative_matrix` and force Tier-0 SPIR-V 1.3 compute shaders. |
| `EC_VULKAN_NO_MIRROR` | Creative Vulkan | Set to `1` to disable the `DEVICE_LOCAL` VRAM mirror on discrete GPUs and use host-visible buffers directly. |

## Measured performance

Recorded on Linux (Ryzen 7 7700X with 2-CU Radeon RDNA2 iGPU
`RADV RAPHAEL_MENDOCINO`, and NVIDIA GeForce RTX 3090 24 GB) on 2026-11-27:

- **LLM (`Qwen3.5-4B-Q4_K_M.gguf`, 426/426 tensors, 2.54 GiB VRAM)**:
  - RTX 3090 Vulkan Turbo HTTP (`llb-serve --reasoning off`, 25 prompt / 12
    decode tokens, identical primary-colors output to CUDA): `393.065 tok/s`
    prefill (`63.603 ms`, `18.42x` vs v1), `102.124 tok/s` decode
    (`107.712 ms`, `5.55x` vs v1).
  - RTX 3090 Vulkan Turbo HTTP (58 prompt / 64 decode tokens): `416.122 tok/s`
    prefill (`139.382 ms`), `105.343 tok/s` decode (`598.046 ms`).
  - RTX 3090 Vulkan Turbo CLI (6 prompt + 16 decode steps): `168 ms` wall time
    (`72.82x` vs 16-thread CPU oracle at `12233 ms`, `16/16` exact token match);
    `LLB_VULKAN_EXACT=1` runs in `310 ms` with top-2 float logits within 0 to 36
    ULP of the CPU oracle.
  - AMD RDNA2 2-CU iGPU Vulkan Turbo CLI (`RADV`, wave32 `dp4a+subgroup`):
    `6814 ms` wall time (`2.08x` vs v1 `14206 ms`), `16/16` exact token match.
- **Creative Studio (`ec-bench` 46/46 PASS on RTX 3090 Vulkan Turbo)**:
  - BF16 GEMM (`G-C5`): `57.643 TFLOPS` (`2.38 ms`, `rel=1.04e-07`, `7.92x` vs v2).
  - FP16 GEMM (`4096x3072x3072`): `53.990 TFLOPS` (`1.43 ms`, `1.14x` faster than CUDA `47.480 TFLOPS`).
  - INT8 W8A8 GEMM (`4096x3072x3072`): `148.066 TOPS` (`0.52 ms`, `1.12x` faster than CUDA `131.807 TOPS`).
  - FlashAttention-2 BF16 (`H=12, L=4096, D=128`): `29.872 TFLOPS` (`3.45 ms`, `GQA_cos=0.999999`, `60.10x` vs v2).
  - FlashAttention-2 F16 (`N=4096, H=24, D=128`): `26.5 TFLOPS` (`7.77 ms`, `cos=1.00000`, `54.55x` vs v2).
  - `FLUX.2-klein-4B` (`256x256`, 4 steps, warm seed 43): `601.26 ms` total
    `gpu_ms` (`dit_ms=533.8`, `vae_ms=59.3`, `text_enc_ms=7.4` Cowrie hit;
    `47.01x` faster than v2 `28265.44 ms`).
  - Cross-vendor supervisor E2E (`creative.test.mjs`): `3/3` PASS on AMD RDNA2
    Vulkan, NVIDIA RTX 3090 Vulkan, and NVIDIA RTX 3090 CUDA.

## Limitations

- Windows Vulkan execution is wired in `backends/windows/build.ps1` and has not
  been run on Windows hardware yet.
- Android Vulkan is not built yet; Android still ships the CPU engine.
- `VK_KHR_cooperative_matrix` requires Vulkan 1.3 and driver matrix support
  (NVIDIA RTX 20/30/40/50, AMD RDNA3/4, Intel Arc); older GPUs (including RDNA2
  iGPUs) run the Tier-0 SPIR-V 1.3 shaders automatically.
- `ec_launch_conv2d_tc` (legacy NCHW im2col path in `ec_ops_common.h`) and the
  final VAE RGB8 conversion still run on CPU threads; FLUX.2 2D VAE decode uses
  the GPU NHWC `coop_conv3x3_bf16` path (`59.3 ms` warm).
