# Vulkan turbo path

This page documents the five-phase Vulkan optimization architecture (`T1` through
`T5`) across both Eureka Engine stacks: the LLM stack (`engine_vulkan`) and the
Creative Studio stack (`eureka-creative-serve`, `ec-bench`).

All measurements in this document were recorded on the same Linux host (AMD
Ryzen 7 7700X 8-Core CPU with 2-CU Radeon RDNA2 iGPU `RADV RAPHAEL_MENDOCINO`,
and NVIDIA GeForce RTX 3090 24 GB) so Vulkan v1/v2, Vulkan Turbo, and CUDA run
on identical silicon. Baseline numbers were recorded at commit
`a4ca66248cd7638055c4e4ef12ee16b02eeb6102` (2026-11-25). All Phase `T1` through
`T5` optimizations are implemented and hardware-verified on Linux (2026-11-27).
Windows Vulkan build wiring is included in `backends/windows/build.ps1` and
remains untested on Windows hardware.

## Measured results: Vulkan v1/v2 vs Vulkan turbo vs CUDA

### LLM stack (`Qwen3.5-4B-Q4_K_M.gguf`)

Model: `Qwen3.5-4B-Q4_K_M.gguf` (`427` tensors, `2,729,969,664` weight bytes;
`L=32`, `D=2560`, `H=16`, `KV=4`, `hd=256`, `dff=9216`, `V=248320`). On Vulkan,
`426/426` weight tensors stay resident in `2.54 GiB` of `DEVICE_LOCAL` VRAM
across 3 slabs.

#### Paired HTTP serving (`llb-serve` `/v1/chat/completions`, `--reasoning off`, RTX 3090)

Greedy temperature-0 completion returning identical output (`"The three primary
colors are red, blue, and yellow."`). CUDA ran with `LLB_PREFILL=fast` and
`LLB_Q4_DP4A=10`. Vulkan Turbo ran with the single-submit GPU graph, wave32
subgroups, `VK_KHR_shader_integer_dot_product` (`dotPacked4x8EXT`), and `BM=8`
batched prefill.

| Metric | Vulkan v1 (`engine_vulkan`) | Vulkan turbo (`engine_vulkan`) | CUDA (`engine_gpu`) | Turbo vs v1 | CUDA / turbo |
|---|---:|---:|---:|---:|---:|
| Prefill throughput (short prompt) | `21.337 tok/s` (`1077.932 ms`, `23` tok) | `393.065 tok/s` (`63.603 ms`, `25` tok) | `722.291 tok/s` (`31.843 ms`, `23` tok) | `18.42x` | `1.84x` |
| Prefill throughput (58-tok prompt) | — | `416.122 tok/s` (`139.382 ms`, `2.403 ms/tok`) | — | `19.50x` (vs v1) | — |
| Decode throughput (12-tok reply) | `18.409 tok/s` (`597.535 ms`, `54.321 ms/tok`) | `102.124 tok/s` (`107.712 ms`, `9.792 ms/tok`) | `155.457 tok/s` (`70.759 ms`, `6.433 ms/tok`) | `5.55x` | `1.52x` |
| Decode throughput (64-tok reply) | — | `105.343 tok/s` (`598.046 ms`, `9.493 ms/tok`) | — | `5.72x` (vs v1) | `1.48x` |

#### Cross-vendor CLI parity (6 prompt tokens `9707 11 1245 525 498 30` + 16 greedy decode steps)

| Route | Device | Vulkan init + upload | Wall time | Parity |
|---|---|---:|---:|---|
| CPU oracle (`LLB_VULKAN_OFF=1`) | Ryzen 7 7700X (16 threads) | — (`192 ms` GGUF load) | `12233 ms` | Reference |
| Vulkan v1 (`LLB_VULKAN_DEVICE=nvidia`) | RTX 3090 (Vulkan 1.4) | `598 ms` | `1034 ms` | `16/16` exact tokens |
| Vulkan turbo (`dp4a+subgroup`, `w32+dma`) | RTX 3090 (Vulkan 1.4) | `573 ms` | `168 ms` | `16/16` exact tokens (`72.82x` vs CPU, `6.15x` vs v1) |
| Vulkan turbo exact (`LLB_VULKAN_EXACT=1`) | RTX 3090 (Vulkan 1.4) | `520 ms` | `310 ms` | `16/16` exact tokens; top-2 float logits within 0 to 36 ULP of CPU oracle |
| Vulkan v1 (`LLB_VULKAN_DEVICE=amd`) | Radeon RDNA2 2-CU (`RADV`) | `2369 ms` | `14206 ms` | `16/16` exact tokens |
| Vulkan turbo (`dp4a+subgroup`, `w32`) | Radeon RDNA2 2-CU (`RADV`) | `770 ms` | `6814 ms` | `16/16` exact tokens (`2.08x` vs v1) |

### Creative Studio stack (`ec-bench` and `FLUX.2-klein-4B`, RTX 3090)

`ec-bench` passed `46/46` gates on Vulkan Turbo and CUDA on the RTX 3090. On the
AMD RDNA2 2-CU iGPU (`RADV RAPHAEL_MENDOCINO`, automatic Tier-0 SPIR-V 1.3
fallback), `ec-bench` passed `19/19` `G-C0.1` numerical oracle gates against CPU
F64 (`cos=1.000000`, `rel=3.49e-08`). Supervisor E2E (`creative.test.mjs`)
passed `3/3` on AMD RDNA2 Vulkan, NVIDIA RTX 3090 Vulkan, and NVIDIA RTX 3090
CUDA. `desktop/engine/tests/backend.test.mjs` passed `17/17`.

#### `ec-bench` kernel throughput (RTX 3090)

| Benchmark | Vulkan v1 | Vulkan v2 | Vulkan turbo | CUDA | Turbo vs v2 | Turbo vs CUDA |
|---|---:|---:|---:|---:|---:|---:|
| BF16 GEMM (`G-C5`, `BM=128` shape) | `0.3 TF` (`422.37 ms`) | `7.280 TF` (`18.88 ms`) | `57.643 TF` (`2.38 ms`, `rel=1.04e-07`) | `63.285 TF` (`2.17 ms`) | `7.92x` | `0.91x` |
| FP16 GEMM (`4096x3072x3072`) | `0.5 TF` (`161.57 ms`) | `7.228 TF` (`10.70 ms`) | `53.990 TF` (`1.43 ms`) | `47.480 TF` (`1.63 ms`) | `7.47x` | `1.14x` (faster) |
| INT8 W8A8 GEMM (`4096x3072x3072`) | `0.5 TOPS` (`149.62 ms`) | `7.375 TOPS` (`10.48 ms`) | `148.066 TOPS` (`0.52 ms`) | `131.807 TOPS` (`0.59 ms`) | `20.08x` | `1.12x` (faster) |
| FA2 BF16 (`H=12, L=4096, D=128`) | `0.2 TF` (`616.60 ms`) | `0.497 TF` (`207.31 ms`) | `29.872 TF` (`3.45 ms`, `GQA_cos=0.999999`) | `31.105 TF` (`3.31 ms`) | `60.10x` | `0.96x` |
| FA2 F16 (`N=4096, H=24, D=128`) | `0.2 TF` (`1263.64 ms`) | `0.5 TF` (`423.86 ms`) | `26.5 TF` (`7.77 ms`, `cos=1.00000`) | `31.5 TF` (`6.54 ms`) | `54.55x` | `0.84x` |
| Im2col Conv2D (`64x64x64->64`) | `5.07 ms` | `3.86 ms` | `5.00 ms` (`2.78x` vs scalar) | `0.06 ms` | `0.77x` | `83.33x` |
| FLUX.2 2D VAE roundtrip (`256x256`, warm) | `2706.1 ms` | `1863.2 ms` (`PSNR=35.36 dB`) | `912.1 ms` (`PSNR=35.33 dB`, `cos=0.99953`) | `44.9 ms` (`PSNR=35.34 dB`) | `2.04x` | `20.31x` |
| LoRA delta hot-swap | `2.10 ms` | `7.14 ms` | `5.05 ms` swap / `6.78 ms` unswap (`cold_dit=2907 ms`) | `0.95 ms` | `1.41x` | `5.32x` |
| Async VRAM DMA swapper (`32 MiB` slabs) | `3.5 GB/s` | `3.3 GB/s` | `3.9 GB/s` (first-alloc pinned DMA; caching allocator reuses freed slabs) | `18.5 GB/s` | `1.18x` | `4.74x` |

#### End-to-end `FLUX.2-klein-4B` (`flux2_klein_4b_bf16.safetensors`, `256x256`, 4 steps, `tier=fast`, RTX 3090)

| Stage | Vulkan v1 (cold, seed 2026) | Vulkan v2 (cold, seed 42) | Vulkan turbo (cold, seed 42) | Vulkan turbo (warm, seed 43) | CUDA (warm TE, cold DiT, seed 42) | Turbo warm vs v2 |
|---|---:|---:|---:|---:|---:|---:|
| Text encoder | `16773.0 ms` | `6047.8 ms` | `6482.6 ms` | `7.4 ms` (Cowrie cache hit) | `6.1 ms` (Cowrie cache hit) | — |
| DiT 4 steps | `90360.5 ms` | `12747.3 ms` | `756.0 ms` | `533.8 ms` | `258.6 ms` | `23.88x` (`169.3x` vs v1) |
| 2D VAE decode | `1955.5 ms` | `1731.2 ms` | `1341.8 ms` | `59.3 ms` | `43.1 ms` | `29.19x` (`32.98x` vs v1) |
| Total `gpu_ms` | `119512.33 ms` | `28265.44 ms` | `16815.99 ms` | `601.26 ms` | `2062.32 ms` | `47.01x` (`198.77x` vs v1) |

## Architecture of phases T1 through T5

The five phases preserve full Tier-0 portable Vulkan 1.2 compatibility while
enabling Vulkan 1.3+ subgroup, `dp4a`, and `VK_KHR_cooperative_matrix` kernels
behind runtime feature probes.

```text
T1 (LLM single-submit GPU graph, portable Vk 1.2)
  -> T2 (LLM batched prefill GEMM BM=8 + wave32 subgroups/dp4a, probed Vk 1.3+)
T3 (Creative GPU-resident glue shaders, portable Vk 1.2)
  -> T4 (Creative vectorized uvec4 VK_KHR_cooperative_matrix GEMM/Conv3x3/FA2, probed Vk 1.3+)
T5 (Persistent VkPipelineCache + dedicated transfer queue + zero-redundant-D2H sync + VRAM caching allocator)
```

### Phase T1: LLM single-submit batched layer graph (implemented)

**Source files:**
- `backends/vulkan/shaders/llm_ops.comp` compiled to
  `backends/vulkan/llb_vulkan_turbo_spv.h` (portable SPIR-V 1.3).
- `backends/vulkan/llb_vulkan_gemv.h`, `backends/vulkan/llb_vulkan_gemv.c`, and
  `backends/windows/engine_cpu.c`.

**Implemented design:**
1. Allocates persistent `DEVICE_LOCAL` buffers (`act_buf`, `state_buf`,
   `kv_buf`, `xq8_buf`) at `llb_vk_graph_init()` time with coalesced transposed
   Gated Delta-Net state `gS` (`[L, TR, SZ, SZ]`).
2. Executes all Qwen3.5 mid-layer operations on GPU in portable SPIR-V 1.3:
   `OP_DEQ_EMBED`, fused residual+`OP_RMS_NORM`, `OP_QUANT_X_Q8_1`,
   `OP_FULL_QK_ROPE_KV`, `OP_GQA_ATTN_GATE`, `OP_LIN_CONV_SILU`,
   `OP_LIN_DELTA_NET`, `OP_SWIGLU_MUL`, `OP_RES_ADD`, and `OP_ARGMAX_TOP2`.
3. Records all 32 layers and the LM head into a single `VkCommandBuffer` using
   static per-slab descriptor sets (0 `vkUpdateDescriptorSets` calls per token)
   and 64-byte push constants. Greedy decode transfers 16 bytes (`top-2` index
   and float bits) per token.
4. Packs all small tensors (`<= 1 MiB`, including norms, biases, `ssm_dt.bias`,
   `ssm_a`, and `ssm_conv1d.weight`) into slab 0 and verifies at init time that
   every multi-weight layer op resides in a single slab.

### Phase T2: LLM batched prefill GEMM and subgroup / `dp4a` compute (implemented)

**Source files:**
- `backends/vulkan/shaders/llm_gemv_dp4a.comp` and
  `backends/vulkan/shaders/llm_gemv_fp32.comp` compiled to
  `backends/vulkan/llb_vulkan_turbo_ext_spv.h` (SPIR-V 1.6) and
  `backends/vulkan/llb_vulkan_turbo_spv.h` (Tier-0 SPIR-V 1.3 fallback).

**Implemented design:**
1. Runtime probe checks Vulkan 1.3, `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`,
   `subgroupSizeControl` (pinning `requiredSubgroupSize = 32` on RADV wave64
   devices), and `shaderIntegerDotProduct`.
2. Implements three automatic precision/capability tiers:
   - Tier 1A (`dp4a+subgroup`, default when supported): on-the-fly Q8_1
     activation quantization + `dotPacked4x8EXT` (`OpSDot`) + `subgroupAdd()`.
   - Tier 1B (`fp32-subgroup`, selected via `LLB_VULKAN_EXACT=1` or when `dp4a`
     is absent): exact FP32 dequantize + `subgroupAdd()`.
   - Tier 0 (`portable-spv13`): Vulkan 1.2 shared-memory tree reduction.
3. Implements `BM=8` batched prefill (`LVK_PREFILL_CHUNK=32`) across `Q4_K`,
   `Q5_K`, `Q6_K`, `Q8_0`, and `F32`, amortizing weight fetches across 8 prompt
   tokens per pass.

### Phase T3: Creative Studio GPU-resident glue shaders (implemented)

**Source files:**
- `creative/backends/shaders/ec_glue.comp` compiled to
  `creative/backends/ec_vulkan_glue_spv.h` (12 portable SPIR-V 1.3 kernels).
- `creative/backends/ec_backend_vulkan.c`, `creative/backends/ec_ops_common.h`,
  `creative/backends/ec_model_kernels_portable.h`, and `creative/ec_flux2.cu`.

**Implemented GPU glue kernels:**
- `norm_adaln` (RMSNorm and LayerNorm with optional weight and AdaLN
  shift/scale, outputting F32, BF16, or F16).
- `gated_res`, `elemwise` (7 fused modes: `add_inplace`, `silu_bf16`,
  `swiglu_mul`, `swiglu_pair_bf16`, `flux2_swiglu_bf16`, `flow_cfg_euler`,
  `dequant_q8g64_bf16`).
- `flux2_pack_qkv_rope`, `flux2_single_cat_q8g64`.
- `nhwc_gn_stats`, `nhwc_gn_apply_bf16`, `upsample2x_nhwc_bf16`,
  `vae_mid_attn_d512`, `groupnorm_silu`.
- `qwen3_qk_rope`, `qwen3_gqa_attn`.

Combined with separating `ec_hal_stream_sync()` (GPU fence wait only, matching
CUDA `cudaStreamSynchronize` semantics) from `ec_hal_sync_all_for_cpu()` (used
only by remaining CPU fallback paths), FLUX.2 DiT and 2D VAE decode eliminate
mid-step VRAM-to-host readbacks.

### Phase T4: Creative Studio `VK_KHR_cooperative_matrix` (implemented)

**Source files:**
- `creative/backends/shaders/ec_coopmat.comp` compiled to
  `creative/backends/ec_vulkan_coop_spv.h` (7 SPIR-V 1.6 kernels).
- `creative/backends/ec_backend_vulkan.c`.

**Implemented kernels and runtime probes:**
1. Probes `vkEnumerateDeviceExtensionProperties` before querying
   `VK_KHR_cooperative_matrix` and `VK_KHR_shader_bfloat16` features and
   `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR()` tile shapes
   (`16x16x16` F16/BF16 -> F32 and `16x16x32` S8 -> S32).
2. Vectorized `uvec4` (128-bit) global-to-shared loads with 24.0 KiB dual-CTA
   tiles:
   - `coop_gemm_bf16` and `coop_gemm_f16`: `128x128x32` tile, 8 warps, `uvec4`
     loads (`57.643 TF` BF16, `53.990 TF` FP16 on RTX 3090).
   - `coop_gemm_i8`: `128x128x64` tile, 8 warps, `16x16x32` S8 -> S32
     (`148.066 TOPS` on RTX 3090).
   - `coop_gemm_q8g64`: `128x64x64` tile, 8 warps, per-64-group rescaling
     accelerating all FLUX.2 DiT quantized linear layers.
   - `coop_conv3x3_bf16`: `64x64x32` implicit-GEMM 3x3 NHWC Conv2D (`29.19x`
     warm 2D VAE decode speedup).
3. 4-warp register-accumulator FlashAttention-2 (`coop_fa2_f16`,
   `coop_fa2_bf16`): `Br=64 x Bc=32`, 128 threads, vectorized `uvec4` Q/K/V
   loads, 38.75 KiB SMEM (`< 48 KiB`), keeping all 8 `16x16` FP32 `mO` tiles in
   registers across KV blocks via a 1-time portable `mRowId` layout probe
   (`29.872 TF` BF16 FA2, `60.10x` vs v2).
4. Automatic fallback: integrated GPUs without BF16/S8x32 matrix units (such as
   RDNA2 iGPUs) and unaligned `K` shapes fall back transparently to the Tier-0
   SPIR-V 1.3 shaders (`ec_vulkan_spv.h`).

### Phase T5: Persistent `VkPipelineCache`, transfer queue, and VRAM allocator (implemented)

1. **Disk-backed `VkPipelineCache`**: Both `llb_vulkan_gemv.c`
   (`vk_llm_<vendor>_<device>_<driver>.bin`) and `ec_backend_vulkan.c`
   (`vk_ec_<vendor>_<device>_<driver>.bin`) persist driver ISA under
   `${XDG_CACHE_HOME:-$HOME/.cache}/eureka/` (or `%LOCALAPPDATA%\Eureka\cache\`
   on Windows), validating `vendorID`, `deviceID`, and `pipelineCacheUUID`.
2. **Dedicated transfer queue (`VK_QUEUE_TRANSFER_BIT`)**: `llb_vulkan_gemv.c`
   uploads weight slabs over a dedicated transfer queue family when present,
   using `VK_SHARING_MODE_CONCURRENT` across compute and transfer families.
3. **Zero-copy pinned DMA and caching allocator**: `ec_backend_vulkan.c` backs
   `ec_hal_host_alloc_raw()` with `HOST_CACHED` Vulkan staging buffers for
   direct `vkCmdCopyBuffer` PCIe transfers, keeps `ec_hal_stream_sync()` free of
   spurious D2H readbacks, and caches up to 16 freed VRAM buffers (`<= 256 MiB`)
   for exact-size reuse.
4. **Integrated-GPU TDR chunking**: On non-discrete GPUs, `ec_backend_vulkan.c`
   caps per-submit workgroup chunks (`4096` for SIMT FA2, `8192` otherwise) so
   long dispatches on 2-CU iGPUs stay below Linux `amdgpu` and Windows WDDM TDR
   timeouts.

## Shader toolchain rule

All GLSL 4.60 Turbo shaders compile through `backends/vulkan/compile_spv.py`
using system `libshaderc_shared.so.1` with `SHADERC_OPT_PERFORMANCE`:

| Tier | Language and compiler | Target SPIR-V | Runtime guard |
|---|---|---|---|
| Tier 0 (universal baseline, shipped) | WGSL -> `naga` (`ZeroInitializeWorkgroupMemoryMode::None`) and GLSL 4.60 -> `compile_spv.py --vulkan 1.2` | SPIR-V 1.3 | Always available on any Vulkan 1.2+ ICD |
| Tier 1 (Turbo extensions, T2 and T4) | GLSL 4.60 -> `compile_spv.py --vulkan 1.3` | SPIR-V 1.6 | Enabled only after runtime `vkGetPhysicalDeviceFeatures2` and `vkEnumerateDeviceExtensionProperties` checks pass |

Precompiled C headers (`llb_vulkan_spv.h`, `llb_vulkan_turbo_spv.h`,
`llb_vulkan_turbo_ext_spv.h`, `ec_vulkan_spv.h`, `ec_vulkan_glue_spv.h`,
`ec_vulkan_coop_spv.h`) are committed alongside the shader sources so C11 builds
require no shader compiler at build time.

## Vulkan driver invariants

Preserve these invariants across future Vulkan changes:

1. **Never emit SPIR-V workgroup zero-initialization polyfills.** Initialize
   shared memory explicitly inside shader loops.
2. **Chunk X workgroup dispatches below `65,535`.** Keep per-dispatch X groups
   at `<= 32,768` (`LVK_MAX_WG_X`) using a row/tile base offset, and cap chunks
   to `4096..8192` on integrated GPUs to avoid driver TDR resets.
3. **Compile pipelines lazily or cache them on disk.** Load and save
   UUID-validated `VkPipelineCache` blobs so warm startups skip driver shader
   recompilation.
4. **Distinguish discrete `DEVICE_LOCAL` VRAM from APU unified memory.** Use
   pure `DEVICE_LOCAL` slabs/mirrors on discrete GPUs and unified buffers on
   APUs advertising `DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED`.
5. **Respect `maxStorageBufferRange` with `<= 1 GiB` weight slabs and pack small
   tensors (`<= 1 MiB`) into slab 0.** Multi-weight layer shaders (`qnw+knw`,
   `cw+dt+al`) require their weights in the same descriptor slab.
6. **Guard every extension behind runtime feature probes.** Never query or
   enable `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`, or
   `VK_KHR_shader_integer_dot_product` without checking
   `vkEnumerateDeviceExtensionProperties` and `vkGetPhysicalDeviceFeatures2`
   first.
