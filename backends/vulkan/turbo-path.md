# Vulkan turbo path

This page documents the engineering roadmap to close the performance gap between
the cross-vendor Vulkan backend and the native CUDA backend across both Eureka
Engine stacks: the LLM stack (`engine_vulkan`) and the Creative Studio stack
(`eureka-creative-serve`).

All baseline measurements in this document were recorded at commit
`a4ca66248cd7638055c4e4ef12ee16b02eeb6102` (2026-11-25) on the same Linux
machine (AMD Ryzen 7 7700X 8-Core CPU with 2-CU Radeon RDNA2 iGPU
`RADV RAPHAEL_MENDOCINO`, and NVIDIA GeForce RTX 3090 24 GB) so CUDA and
Vulkan run on identical silicon. All Phase T1 through T5 speedup numbers are
engineering targets and are untested until implemented and verified against the
gates below.

## Measured baseline: Vulkan vs CUDA

### LLM stack (`Qwen3.5-4B-Q4_K_M.gguf`)

Model: `Qwen3.5-4B-Q4_K_M.gguf` (`427` tensors, `2,729,969,664` weight bytes;
`L=32`, `D=2560`, `H=16`, `KV=4`, `hd=256`, `dff=9216`, `V=248320`). On Vulkan,
`426/426` weight tensors stay resident in `2.54 GiB` of `DEVICE_LOCAL` VRAM
across 3 slabs.

#### Paired HTTP serving (`llb-serve` `/v1/chat/completions`, RTX 3090)

Exact same 23-token prompt and 12-token completion (`"The three primary colors
are red, blue, and yellow."`). CUDA ran with `LLB_PREFILL=fast` and
`LLB_Q4_DP4A=10`.

| Metric | Vulkan v1 (`engine_vulkan`) | CUDA (`engine_gpu`) | CUDA / Vulkan ratio |
|---|---:|---:|---:|
| Prefill throughput | `21.337 tok/s` (`1077.932 ms`, `46.867 ms/tok`) | `722.291 tok/s` (`31.843 ms`, `1.384 ms/tok`) | `33.85x` |
| Decode throughput | `18.409 tok/s` (`597.535 ms`, `54.321 ms/tok`) | `155.457 tok/s` (`70.759 ms`, `6.433 ms/tok`) | `8.44x` |

#### Cross-vendor CLI parity (6 prompt tokens + 16 greedy decode steps)

| Route | Device | Load + VRAM upload | Wall time | Parity |
|---|---|---:|---:|---|
| CPU oracle (`LLB_VULKAN_OFF=1`) | Ryzen 7 7700X (16 threads) | `1025 ms` | `12115 ms` | Reference |
| Vulkan v1 (`LLB_VULKAN_DEVICE=nvidia`) | RTX 3090 (Vulkan 1.4) | `1171 ms` (`598 ms` upload) | `1034 ms` | `16/16` exact tokens (`11.72x` vs CPU) |
| Vulkan v1 (`LLB_VULKAN_DEVICE=amd`) | Radeon RDNA2 2-CU (`RADV`) | `2564 ms` (`2369 ms` upload) | `14206 ms` | `16/16` exact tokens; `32/32` identical top-2 float logit bits vs RTX 3090 Vulkan |

### Creative Studio stack (`ec-bench` and `FLUX.2-klein-4B`, RTX 3090)

`ec-bench` passed `46/46` gates on Vulkan v1, Vulkan v2, and CUDA on the RTX
3090. Supervisor E2E (`creative.test.mjs`) passed `3/3` on AMD RDNA2 Vulkan,
NVIDIA RTX 3090 Vulkan, and NVIDIA RTX 3090 CUDA.

#### `ec-bench` kernel throughput (RTX 3090)

| Benchmark | Vulkan v1 | Vulkan v2 | CUDA | CUDA / Vulkan v2 |
|---|---:|---:|---:|---:|
| BF16 GEMM (`G-C5`, `BM=128` shape) | `0.3 TF` (`422.37 ms`) | `7.280 TF` (`18.88 ms`) | `63.285 TF` (`2.17 ms`) | `8.69x` |
| FP16 GEMM (`4096x3072x3072`) | `0.5 TF` (`161.57 ms`) | `7.228 TF` (`10.70 ms`) | `47.480 TF` (`1.63 ms`) | `6.57x` |
| INT8 W8A8 GEMM (`4096x3072x3072`) | `0.5 TOPS` (`149.62 ms`) | `7.375 TOPS` (`10.48 ms`) | `131.807 TOPS` (`0.59 ms`) | `17.87x` |
| FA2 BF16 (`H=12, L=4096, D=128`) | `0.2 TF` (`616.60 ms`) | `0.497 TF` (`207.31 ms`) | `31.105 TF` (`3.31 ms`) | `62.63x` |
| FA2 F16 (`N=4096, H=24, D=128`) | `0.2 TF` (`1263.64 ms`) | `0.5 TF` (`423.86 ms`) | `31.5 TF` (`6.54 ms`); Sage `49.9 TF` (`4.13 ms`) | `64.81x` (FA2) |
| Im2col Conv2D (`64x64x64->64`) | `5.07 ms` | `3.86 ms` | `0.06 ms` | `64.33x` |
| FLUX.2 2D VAE (`256x256`, warm) | `2706.1 ms` | `1863.2 ms` (`PSNR=35.36 dB`) | `44.9 ms` (`PSNR=35.34 dB`) | `41.50x` |
| Wan 2.2 Causal 3D VAE encode | `2166.0 ms` | `1868.6 ms` | `468.0 ms` | `3.99x` |
| ACE-Step 1D Audio VAE encode (`2.0 s`) | `1497.7 ms` | `1342.7 ms` | `270.3 ms` | `4.97x` |
| LoRA delta hot-swap | `2.10 ms` | `7.14 ms` | `0.95 ms` | `7.52x` |
| Async VRAM DMA swapper | `3.5 GB/s` | `3.3 GB/s` | `18.5 GB/s` | `5.61x` |

#### End-to-end `FLUX.2-klein-4B` (`flux2_klein_4b_bf16.safetensors`, `256x256`, 4 steps, `tier=fast`, RTX 3090)

| Stage | Vulkan v1 (seed 2026) | Vulkan v2 (seed 42) | CUDA (seed 42) | CUDA / Vulkan v2 |
|---|---:|---:|---:|---:|
| Text encoder | `16773.0 ms` | `6047.8 ms` | `6.1 ms` (Cowrie cache hit) | N/A (cache hit vs cold) |
| DiT 4 steps | `90360.5 ms` | `12747.3 ms` | `258.6 ms` | `49.29x` |
| 2D VAE decode | `1955.5 ms` | `1731.2 ms` | `43.1 ms` | `40.17x` |
| Total `gpu_ms` | `119512.33 ms` | `28265.44 ms` | `2062.32 ms` | `13.71x` |

## Root-cause breakdown of the remaining gap

None of the gaps above come from Vulkan API limits or hardware memory bandwidth
limits. Each gap maps to a specific missing execution path in the current
portable v1/v2 implementation:

### LLM bottlenecks (`backends/vulkan/llb_vulkan_gemv.c` and `llb_vulkan_shaders.wgsl`)

1. **Per-GEMV submit and fence stall (~30 ms wasted per token).**
   `llb_vk_gemv()` copies the activation vector `x` to `x_dev_buf`, records one
   dispatch, copies `y_dev_buf` back to `y_host_buf`, calls `vkQueueSubmit()`,
   and blocks in `vkWaitForFences()`. A 32-layer Qwen3.5 forward pass executes
   roughly 210 GEMV calls per token. At ~140 to 160 us of round-trip submit,
   PCIe copy, and fence wakeup per call, ~31 ms of each 54.3 ms decode token is
   host/driver synchronization overhead rather than memory bandwidth.
2. **Host CPU execution of mid-layer ops.**
   RMSNorm, L2 QK-norm, partial NEOX RoPE, causal depthwise Conv1D + SiLU,
   Gated Delta-Net state recurrence, grouped-query attention over the KV cache,
   SwiGLU, and residual adds run on host CPU threads between Vulkan GEMVs in
   `backends/windows/engine_cpu.c` and `tools/qwen35_engine.c`.
3. **Token-by-token prefill instead of batched GEMM.**
   `engine_vulkan` calls `llb_vk_gemv()` one prompt token at a time (`M = 1`),
   loading every quantized weight matrix from VRAM once per prompt token instead
   of amortizing weight loads across a prompt tile (`M = 32..128`). This causes
   the `33.85x` prefill gap (`21.337` vs `722.291 tok/s`).
4. **No subgroup reduction or integer dot product (`dp4a`).**
   Naga 26.0 does not parse WGSL `enable subgroups;`. The five shipped GEMV
   shaders (`vk_gemv_q4k`, `vk_gemv_q5k`, `vk_gemv_q6k`, `vk_gemv_q8_0`,
   `vk_gemv_f32`) reduce across `var<workgroup>` shared memory and dequantize
   weights to FP32 instead of using `OpGroupNonUniformFAdd` and
   `VK_KHR_shader_integer_dot_product` (`OpSDot`).

### Creative Studio bottlenecks (`creative/backends/ec_backend_vulkan.c` and `ec_vulkan_shaders.wgsl`)

1. **Per-layer CPU glue ops force VRAM-to-RAM round-trips.**
   The portable glue layers (`creative/backends/ec_ops_common.h` and
   `creative/backends/ec_model_kernels_portable.h`) call `ec_hal_stream_sync()`
   before every norm, RoPE, concat, GroupNorm, upsample, and residual op. Each
   sync flushes the 256-slot Vulkan command buffer, downloads `dev_dirty`
   activation buffers over PCIe, executes OpenMP loops on CPU, marks outputs
   `host_dirty`, and re-uploads them on the next Vulkan GEMM. On a 4-step
   FLUX.2 DiT + 2D VAE run, this triggers hundreds of full pipeline stalls and
   accounts for the gap between `8.7x` raw GEMM and `49.3x` DiT wall time.
2. **FP32 SIMT compute instead of matrix hardware (`VK_KHR_cooperative_matrix`).**
   The v2 `64x64x16` zero-bank-conflict shaders (`vk_gemm_bf16_f32`,
   `vk_gemm_f16_f32`, `vk_gemm_i8_f32`, `vk_conv2d_3x3_nhwc_bf16`) run on FP32
   SIMT ALUs (`7.28 TFLOPS` on RTX 3090). Hardware matrix units (NVIDIA Tensor
   Cores, AMD RDNA3/4 WMMA, Intel Arc XMX) are exposed in Vulkan through
   `VK_KHR_cooperative_matrix` and remain unused.
3. **Single-query FlashAttention-2 tile geometry.**
   `vk_fa2_bf16` and `vk_fa2_f16` process one query row per 64-thread workgroup
   with 4-key unrolling (`0.497 TF` vs `31.105 TF` in CUDA). They perform no
   query-tile reuse of `K`/`V` across multiple queries (`Br = 1` instead of
   `Br = 32..64`).
4. **Cold pipeline compile and synchronous host staging.**
   Pipelines compile fresh on each process start (no disk `VkPipelineCache`),
   and DMA uploads use the compute queue without asynchronous transfer-queue
   overlap.

## Staged implementation roadmap

Implement the five phases in order. Phases T1 and T3 require only portable
Vulkan 1.2 (no vendor extensions) and ship across every Vulkan GPU, including
older RDNA2 iGPUs and Lavapipe. Phases T2 and T4 add runtime-probed Vulkan 1.3+
and extension shader paths while retaining the shipped SPIR-V 1.3 shaders as an
automatic fallback.

```text
T1 (LLM single-submit GPU graph, portable Vk 1.2)
  -> T2 (LLM batched prefill GEMM + subgroups/OpSDot, probed Vk 1.3+)
T3 (Creative GPU-resident glue shaders, portable Vk 1.2)
  -> T4 (Creative VK_KHR_cooperative_matrix GEMM/Conv3x3/FA2, probed Vk 1.3+)
T5 (Persistent VkPipelineCache + async VK_QUEUE_TRANSFER_BIT + timeline semaphores)
```

### Phase T1: LLM single-submit batched layer graph (portable Vulkan 1.2)

**Goal:** Eliminate per-GEMV `vkQueueSubmit` / `vkWaitForFences` round-trips and
move all per-token Qwen3.5 mid-layer ops into `DEVICE_LOCAL` VRAM.

**Target files:**
- `backends/vulkan/llb_vulkan_layer_shaders.wgsl` (new) and regenerated
  `backends/vulkan/llb_vulkan_spv.h`.
- `backends/vulkan/llb_vulkan_graph.h` / `backends/vulkan/llb_vulkan_graph.c`
  (or extension of `llb_vulkan_gemv.c`).
- `backends/windows/engine_cpu.c` (route `LLB_VULKAN=1` full-step decode into
  the resident Vulkan graph when all model tensors fit in VRAM).

**Work items:**
1. Allocate persistent `DEVICE_LOCAL` activation and state buffers at
   `llb_vk_init()` time: hidden vector `x`, scratch `qkv`/`mlp`, FP16/FP32 KV
   cache, and Gated Delta-Net recurrent state `gS` and `gConv`.
2. Add portable SPIR-V 1.3 compute shaders for every Qwen3.5 mid-layer op:
   - `vk_rms_norm` (with optional `+1` weight offset for Qwen3.5 RMSNorm)
   - `vk_l2_qk_norm_rope_neox` (fused Q/K norm + partial NEOX RoPE)
   - `vk_conv1d_silu_state` (depthwise causal Conv1D state update + SiLU)
   - `vk_gated_delta_net_step` (per-head recurrent state update)
   - `vk_gqa_attn_decode` (split-KV online-softmax decode attention over VRAM KV
     cache)
   - `vk_swiglu_mul` and `vk_residual_add`
   - `vk_argmax_f32` (device-side greedy argmax so greedy decode transfers 4
     bytes instead of `248,320 * 4` bytes of logits)
3. Record all 32 layers and the LM head into a single `VkCommandBuffer` with
   lightweight `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` memory barriers between
   dependent dispatches, and submit once per token (`1` `vkQueueSubmit`, `1`
   `vkWaitForFences`).
4. Replace uniform-buffer per-dispatch slots with `vkCmdPushConstants` (128
   bytes guaranteed by Vulkan 1.0) for dispatch parameters (`rows`, `cols`,
   `rb`, `w_off`, `row_base`, `pos`), removing descriptor-set rebinding between
   layers that share the same weight slab.

**Projected performance (untested target):**
- RTX 3090 `Qwen3.5-4B-Q4_K_M` decode: `18.4` -> `85 to 115 tok/s` (`~4.6x to
  6.2x` speedup; within `~1.4x to 1.8x` of CUDA).
- AMD RX 7900 XTX / RDNA3 discrete GPUs: `~80 to 110 tok/s` (bandwidth-bound on
  `2.54 GiB` weights).
- AMD RDNA2 2-CU iGPU: `~1.5` -> `~5 to 8 tok/s` (eliminating CPU-GPU ping-pong
  and redundant copies).

### Phase T2: LLM batched prefill GEMM + integer dot product and subgroups

**Goal:** Close the `33.85x` prefill gap and recover the final ~25% decode
efficiency on GPUs supporting Vulkan 1.2/1.3 subgroup and `dp4a` instructions.

**Target files:**
- `backends/vulkan/shaders_glsl/` (new GLSL 4.60 / Slang compute sources
  compiled via `glslangValidator --target-env vulkan1.3` to an optional
  `llb_vulkan_spv_ext.h`).
- `backends/vulkan/llb_vulkan_gemv.c` (runtime capability probe and batched
  prefill entry point `llb_vk_prefill_chunk()`).

**Work items:**
1. **Runtime capability probe:** Query `VkPhysicalDeviceVulkan11Features`,
   `VkPhysicalDeviceVulkan12Features`,
   `VkPhysicalDeviceSubgroupProperties` (`VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`),
   and `VkPhysicalDeviceShaderIntegerDotProductFeatures`
   (`shaderIntegerDotProduct`). Enable extended SPIR-V 1.5/1.6 pipelines only
   when the driver advertises the required bits; fall back transparently to
   Phase T1 SPIR-V 1.3 shaders otherwise.
2. **Subgroup-reduced and `dp4a` decode GEMV:**
   - Replace `var<workgroup>` tree reductions with `subgroupAdd()`.
   - Add on-the-fly Q8_1 quantization of activation vector `x` in VRAM and use
     `dotPacked4x8EXT` (`OpSDot`) in `Q4_K` and `Q8_0` decode GEMV shaders.
3. **Batched prefill (`M = 32` or `M = 64` tokens per chunk):**
   - Implement tiled dequant-GEMM shaders (`vk_gemm_q4k_prefill`,
     `vk_gemm_q5k_prefill`, `vk_gemm_q6k_prefill`, `vk_gemm_q8_0_prefill`) that
     dequantize weight tiles once into shared memory and multiply against `M`
     prompt token activations.
   - Add batched causal attention and chunked Gated Delta-Net prefill in VRAM.

**Projected performance (untested target):**
- RTX 3090 `Qwen3.5-4B-Q4_K_M` prefill: `21.3` -> `350 to 550+ tok/s` (`~16x to
  26x` speedup).
- RTX 3090 `Qwen3.5-4B-Q4_K_M` decode: `110 to 135 tok/s` (within `~1.15x to
  1.4x` of CUDA).

### Phase T3: Creative Studio GPU-resident glue shaders (portable Vulkan 1.2)

**Goal:** Eliminate every mid-step `ec_hal_stream_sync()` PCIe readback in the
DiT and VAE forward passes so activations never leave `DEVICE_LOCAL` VRAM during
a generation step.

**Target files:**
- `creative/backends/ec_vulkan_shaders.wgsl` and `creative/backends/ec_vulkan_spv.h`.
- `creative/backends/ec_backend_vulkan.c` (implement native Vulkan HAL hooks
  instead of delegating to `ec_ops_common.h` and `ec_model_kernels_portable.h`
  CPU fallbacks).
- `creative/ec_hal.h` (expose optional backend-native hook table or weak
  overrides for the hot glue kernels).

**Hot glue kernels to move from CPU OpenMP (`ec_model_kernels_portable.h` / `ec_ops_common.h`) to SPIR-V:**

| Kernel | Current CPU symbol | Impact |
|---|---|---|
| RMSNorm + AdaLN-Zero (`shift`, `scale`) | `ec_port_k_rmsnorm_bf16`, `ec_port_k_adaln_bf16`, `ec_port_k_as_rmsnorm_adaln`, `ec_port_k_wn_rmsnorm` | Called twice per DiT block |
| LayerNorm + AdaLN-Zero | `ec_port_k_tr_layernorm`, `ec_port_k_wn_layernorm`, `ec_ops_adaln` | Called in FLUX.2, TRELLIS.2, Wan 2.2 |
| QKV pack + QK-Norm + ND RoPE | `ec_port_k_flux2_pack_qkv_rope`, `ec_port_k_qwen3_qk_rope`, `ec_port_k_as_qk_norm_rope_bf16`, `ec_port_k_ss_self_qkv_rope` | Called once per attention block |
| Single-stream concat + Q8_G64 quant | `ec_port_k_flux2_single_cat_q8g64` | Called in every FLUX.2 single block |
| Gated residual add + SwiGLU / GeGLU | `ec_ops_gated_res`, `ec_ops_swiglu`, `ec_ops_silu_mul` | Called twice per DiT block |
| NHWC GroupNorm + SiLU (two-pass stats + apply) | `ec_port_k_nhwc_gn_stats`, `ec_port_k_nhwc_gn_apply_bf16` | Called twice per 2D VAE ResBlock |
| NHWC 2x nearest upsample + RGB8 pack | `ec_port_k_upsample2x_nhwc_bf16`, `ec_port_k_vae_conv_out_rgb8_nhwc` | Called in VAE decoder stages |
| VAE mid-block self-attention (`D=512`) | `ec_port_k_vae_mid_attn`, `ec_port_k_vae_mid_attn_tc_d512` | Removes the last CPU sync in 2D VAE |

Once these eight shader groups run on Vulkan, `ec_backend_vulkan.c` keeps the
entire DiT step and the entire 2D VAE decode inside its batched command buffer
with zero `dev_dirty -> host_buf` downloads until the final PNG pixel buffer is
read out.

**Projected performance (untested target):**
- `FLUX.2-klein-4B` (`256x256`, 4 steps, RTX 3090):
  - DiT 4 steps: `12747.3 ms` -> `~3000 to 4000 ms` (`~3.2x to 4.2x` speedup
    before adding matrix cores).
  - 2D VAE decode: `1731.2 ms` -> `~300 to 450 ms` (`~4x to 5.5x` speedup).
  - Cold total `gpu_ms`: `28265.44 ms` -> `~8000 to 10500 ms`.

### Phase T4: Creative Studio `VK_KHR_cooperative_matrix` (Tensor Core / WMMA / XMX)

**Goal:** Unlock hardware matrix multiply-accumulate units across NVIDIA RTX
20/30/40/50, AMD RDNA3/4 (RX 7000/9000), and Intel Arc (Alchemist/Battlemage)
through the Khronos standard `VK_KHR_cooperative_matrix` extension.

**Target files:**
- `creative/backends/vulkan_coop/` (new GLSL 4.60 shaders using
  `GL_KHR_cooperative_matrix`, `GL_EXT_shader_explicit_arithmetic_types_float16`,
  `GL_EXT_bfloat16`, and `GL_EXT_shader_explicit_arithmetic_types_int8`,
  compiled via `glslangValidator --target-env vulkan1.3` to
  `ec_vulkan_spv_coop.h`).
- `creative/backends/ec_backend_vulkan.c` (runtime extension and tile-shape
  probe, dual pipeline table with automatic fallback to SPIR-V 1.3 v2 shaders).

**Work items:**
1. **Runtime extension and property probe:**
   - Check `vkEnumerateDeviceExtensionProperties` for
     `VK_KHR_cooperative_matrix`, `VK_KHR_shader_float16_int8`, and (for BF16)
     `VK_KHR_shader_bfloat16`.
   - Call `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR()` and select a
     supported `M x N x K` hardware tile (`16x16x16` for FP16/BF16 -> F32,
     `16x16x32` for INT8 -> I32; pass tile sizes via SPIR-V specialization
     constants so one shader binary adapts to NVIDIA 32-lane warps, AMD RDNA3
     32-lane wave32, and Intel 16-lane subgroups).
   - If BF16 cooperative matrix is unsupported on a device that supports FP16
     cooperative matrix (for example older RDNA3 drivers), convert BF16 weight
     tiles in shared memory or route FP16-compatible paths to FP16 coopmat and
     fall back to the v2 SPIR-V 1.3 BF16 shader when exact BF16 range is
     required.
2. **Cooperative-matrix GEMM (`vk_coop_gemm_{bf16,f16,i8,q8g64}`):**
   - Use `BM=128, BN=128, BK=32` (or `64x64x32` for smaller matrices) with
     `2x2` or `2x4` subgroups per workgroup and double-buffered shared memory
     tiles.
3. **Cooperative-matrix implicit-GEMM 3x3 Conv2D (`vk_coop_conv3x3_bf16`):**
   - Feed `16x16x16` cooperative matrix loads from the NHWC shared tile directly
     to replace the `3.86 ms` SIMT Conv2D path.
4. **Block-tiled FlashAttention-2 (`vk_coop_fa2_{bf16,f16}`):**
   - Upgrade from `Br=1 x Bc=4` SIMT to `Br=32 x Bc=32` (or `Br=64 x Bc=64`)
     cooperative-matrix `S = Q @ K^T` and `O += P @ V` with online softmax in
     registers, plus a `4x4` register-tiled `Br=16 x Bc=16` SIMT fallback for
     non-coopmat GPUs (RDNA2, integrated GPUs).

**Projected performance (untested target, RTX 3090 / RX 7900 XTX class):**
- BF16 / FP16 GEMM: `7.28 TF` -> `40 to 55 TF` (`~6x to 7.5x` vs Vulkan v2;
  within `~1.15x to 1.5x` of CUDA).
- INT8 W8A8 GEMM: `7.38 TOPS` -> `85 to 110 TOPS` (`~11.5x to 15x` vs Vulkan v2;
  within `~1.2x to 1.5x` of CUDA).
- FlashAttention-2 BF16 (`H=12, L=4096, D=128`): `0.50 TF` -> `18 to 25 TF`
  (`~36x to 50x` vs Vulkan v2; within `~1.3x to 1.7x` of CUDA).
- `FLUX.2-klein-4B` (`256x256`, 4 steps, warm DiT + VAE): `14.48 s` ->
  `~0.45 to 0.70 s` (within `~1.5x to 2.3x` of CUDA `0.30 s`).

### Phase T5: Persistent `VkPipelineCache` and async transfer queue

**Goal:** Reduce cold process startup latency and raise DMA upload bandwidth
from `3.3 GB/s` toward the `18.5 GB/s` CUDA pinned-DMA ceiling.

**Target files:**
- `backends/vulkan/llb_vulkan_gemv.c` and `creative/backends/ec_backend_vulkan.c`.

**Work items:**
1. **Disk-backed `VkPipelineCache`:** Serialize driver-compiled ISA to
   `${XDG_CACHE_HOME:-$HOME/.cache}/eureka/vk_cache_<vendorId>_<deviceId>_<driverVersion>_<shaderHash>.bin`
   (or `%LOCALAPPDATA%\Eureka\cache\` on Windows). Validate header `vendorID`,
   `deviceID`, and `pipelineCacheUUID` before loading. Target: `< 2 ms` warm
   pipeline creation across both NVIDIA and Mesa RADV.
2. **Dedicated transfer queue and pinned staging ring:** Probe for a distinct
   queue family with `VK_QUEUE_TRANSFER_BIT` (without `VK_QUEUE_GRAPHICS_BIT`)
   and overlap weight slab uploads and LoRA swaps with compute using
   `VK_KHR_timeline_semaphore` (core in Vulkan 1.2) and a persistent ring of
   `HOST_VISIBLE | HOST_COHERENT` pinned staging buffers.
3. **Zero-copy Cowrie text-encoder cache parity:** Ensure Vulkan Creative
   invocations share the same Cowrie StatePage prompt-embedding cache path that
   gives CUDA its `6.1 ms` cached `text_enc_ms`.

## Shader toolchain rule

Naga 26.0 compiles our portable SPIR-V 1.3 baseline cleanly (`llb_vulkan_spv.h`
and `ec_vulkan_spv.h`), but does not support `enable subgroups;`, `OpSDot`, or
`VK_KHR_cooperative_matrix`. Keep a two-tier shader build:

| Tier | Language and compiler | Target SPIR-V | Runtime guard |
|---|---|---|---|
| Tier 0 (universal baseline, shipped) | WGSL -> `naga` (with `ZeroInitializeWorkgroupMemoryMode::None`) | SPIR-V 1.3 | Always available on any Vulkan 1.2+ ICD |
| Tier 1 (Turbo extensions, T2 and T4) | GLSL 4.60 or Slang -> `glslangValidator --target-env vulkan1.3` / `slangc` | SPIR-V 1.5 / 1.6 | Enabled only after runtime `vkGetPhysicalDeviceFeatures2` and `vkEnumerateDeviceExtensionProperties` checks pass |

Commit precompiled C header blobs (`*_spv.h`) alongside shader sources so end
users and CI build machines do not need `naga`, `glslangValidator`, or `slangc`
installed to compile the C11 engine binaries.

## Verification and acceptance gates

Every Turbo phase must pass the paired acceptance matrix before merging:

1. **LLM numerical parity (T1, T2):**
   - `Qwen3.5-4B-Q4_K_M.gguf` CLI 16-step greedy decode matches the CPU oracle
     (`LLB_VULKAN_OFF=1`) on `16/16` tokens on both NVIDIA RTX 3090 Vulkan and
     AMD RDNA2 Vulkan.
   - `llb-serve` `/v1/chat/completions` returns identical greedy output to
     `engine_gpu` on the paired prompt.
   - Cross-vendor check: AMD RDNA2 (`RADV`) must pass with zero driver hangs,
     validation errors, or regressions on the SPIR-V 1.3 fallback path.
2. **Creative numerical and E2E parity (T3, T4, T5):**
   - `ec-bench` passes `46/46` gates on RTX 3090 Vulkan, with all `G-C0.1`
     numerical gates within declared tolerances (`cos >= 0.998` for attention,
     `PSNR >= 35 dB` for 2D VAE, `rel_L2` matching or beating v2).
   - `creative.test.mjs` supervisor E2E passes `3/3` on AMD RDNA2 Vulkan,
     NVIDIA RTX 3090 Vulkan, and NVIDIA RTX 3090 CUDA.
   - Real `FLUX.2-klein-4B` `256x256` 4-step output matches CUDA visual and
     PSNR parity with zero NaN or Inf values.
3. **Paired performance receipts:**
   - Record before/after measurements on the same RTX 3090 and AMD RDNA2 host
     in paired rounds, with exact command lines, model hashes, and commit SHAs.

## Vulkan driver invariants

Preserve the following invariants across all Turbo phases. Each invariant fixes
a real driver or hardware failure observed during v1/v2 bring-up:

1. **Never emit SPIR-V workgroup zero-initialization polyfills.**
   When compiling WGSL via Naga, always set
   `ZeroInitializeWorkgroupMemoryMode::None`. Naga's default `Polyfill` mode
   stalled Mesa RADV ACO shader compilation to `29,729 ms` across the 9
   Creative modules (`~4.2 s` per shader). Disabling it dropped total RADV
   compile time `739x` to `40.2 ms` and accelerated FA2 execution `1.78x`.
   Initialize shared memory explicitly inside shader loops instead.
2. **Chunk X workgroup dispatches below `65,535`.**
   Vulkan's guaranteed minimum `maxComputeWorkGroupCount[0]` is `65,535`. Large
   LLM vocabularies (`V = 248,320`) exceed this limit if mapped 1 or 2 rows per
   workgroup. Always chunk dispatches using a `row_base` offset with
   `<= 32,768` workgroups per dispatch (`LVK_MAX_WG_X`).
3. **Compile pipelines lazily on first use.**
   Create `VkShaderModule` and `VkPipeline` objects on first dispatch of each
   pipeline ID rather than during `init()`. Short runs and health checks then
   compile zero unused shaders.
4. **Distinguish discrete `DEVICE_LOCAL` VRAM from APU unified memory.**
   On discrete PCIe GPUs, allocate pure `DEVICE_LOCAL` slabs (avoiding
   `HOST_VISIBLE`) for weights and mirrors so reads run at GDDR6/GDDR6X
   bandwidth. On APUs advertising `DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED`,
   use a single unified buffer with zero mirror copies.
5. **Respect `maxStorageBufferRange` with `<= 1 GiB` weight slabs.**
   Split multi-gigabyte models (4B to 27B+) into `<= 1 GiB` `VkBuffer` slabs
   aligned to `minStorageBufferOffsetAlignment` so models load cleanly across
   AMD, Intel, and NVIDIA drivers with 1 GiB to 2 GiB per-binding limits.
6. **Guard every extension behind runtime feature probes.**
   Never require `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`, or
   `VK_KHR_shader_integer_dot_product` at `vkCreateDevice()` time without first
   verifying device support. A missing extension must downgrade cleanly to the
   Tier 0 SPIR-V 1.3 pipeline rather than failing device creation.
