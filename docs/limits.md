# Model and hardware limits

What each engine accepts today, what stops it from running other model sizes
or smaller hardware, and how hard each limit is to lift. Every engine runs the
Qwen3.5 dense hybrid architecture only (linear-attention layers plus a
full-attention layer every fourth layer).

Classes:

- **Cap**: a constant, array size, or allowlist. It can be read from the
  model or sized at run time without new kernels.
- **Kernel**: a GPU or SIMD kernel is written for this value. Lifting it
  needs a new kernel instantiation.
- **Hardware**: a real device or format requirement.

Line references are to files in this repository; for patch files they are
lines inside the patch.

## Summary

| Engine | Model shapes accepted | Max context | Memory model |
|---|---|---|---|
| CPU (Android, `backends/android/engine`) | Any dense Qwen3.5 pack (shape read from `index.tsv`) | KV capacity set at launch (`--decide CTX`) | Pack mmapped; f32 KV for full-attention layers only |
| CUDA (root) | 9B (d 4096), 27B (d 5120), 0.8B (d 1024) | 131,072 (2^17) | Whole model in VRAM; f32 KV for 2^dT slots |
| Metal, macOS (`backends/metal`) | 4B (d 2560), 9B, 27B K-quant pack, 35B MoE | 131,072 | One heap span (`--gpu`); f32 KV |
| Metal, iOS (`backends/ios`) | Same engine as macOS; shipped with 4B and Kev-4B | 131,072 (`llb_open` cap) | One span: GGUF + 3 GiB |
| WebGPU, shipped (`backends/webgpu/engine`) | 4B only | 4,096 | Whole model on the GPU, split into ≤128 MiB buffers |
| WebGPU, trunk (`backends/webgpu/trunk`) | Any Qwen3.5 shape, dense or MoE (head, state, conv read from the file); 21 GGUF formats | Bounded by the GPU's binding limit for one layer's f16 KV (about 2M tokens for the 0.8B on an RTX 3090) | Weights split by rows (or experts) to fit the device |

## What blocks each case

### Smaller models (0.8B, 2B)

- **CPU:** runs them. Qwen3.5-2B Q4_K_M was tested (see
  [backends/android](../backends/android/README.md#models)).
- **CUDA:** `mf.validate` in `engine_qwen35_gpu.bend` (lines 277–289) allows
  only three shapes. Behind it, the device envelope in
  `tools/payload-27b.patch` (lines 169–175) accepts d_model 4096 or 5120
  (1024 by a special case), and the rmsnorm reduction trees
  (`payload-27b.patch` 199–223) handle only 1024, 2048, 4096, and ≥8-leaf
  widths, so d 2560 or 3072 would read out of bounds. Below d 2048, spec,
  batch, and fast prefill are refused (`host/llb_host.c` 2244).
- **Metal:** `mf.validate` (`backends/metal/engine/metal-engine.patch`
  258–299) and `m9_shape` (`cc/metal-cc.patch` 5682–5696) whitelist four
  shapes. Attention kernels assume 4 KV heads × 256 (a 1024-float KV row,
  `cc/metal-cc.patch` 4300–4865); 0.8B and 2B have 2 KV heads. The rmsnorm
  kernels compute d 1024 wrongly (`cc/metal-cc.patch` 2109–2160). K-quant
  GEMVs refuse 1024 and 2048 columns (5495–5520).
- **WebGPU shipped:** `q35_tensors` asserts the 4B geometry
  (`engine/q35_plan.js` 30–34), and the emitted kernels bake in its widths,
  head counts, and vocabulary.
- **WebGPU trunk:** runs them (0.8B and 2B tested against llama.cpp; see
  [reports/webgpu-any-shape](../reports/webgpu-any-shape/REPORT.md)).

### Bigger models (above 27B)

- **CUDA and Metal:** layer arrays sized 64; d_model ≤ 8192 (head scratch);
  d_ff ≤ 17,408 and GQA 4 or 6; ≤ 24 query heads (attention score planes);
  fixed scratch sized for the 27B. Head region ≤ 2 GiB and 32-bit offsets.
  On CUDA, `gpu_q8b`/`gpu_q8c` (`payload-dp4a.patch` 182,
  `payload-dp4b.patch` 386) and `gpu_atts` overflow silently past those
  sizes.
- **CPU:** no shape cap. Memory is the limit: the pack is mmapped and must be
  mostly resident for speed.

### Less memory (8–12 GB GPUs, 8 GB Macs, 6–8 GB phones)

- **CUDA:** the whole model must fit in VRAM (no offload). KV is f32 and
  allocated for the full power-of-two capacity at launch. Turing and older
  cards are out (sm_80 floor, a hardware class).
- **Metal and iOS:** one no-copy `MTLBuffer` holds weights, KV, and scratch.
  The iOS default span is GGUF + 3 GiB (`backends/ios/engine/engine/llb_engine.m`
  192–206). Weight blobs round up to power-of-two size classes, up to 2× on
  large layers.
- **CPU (Android):** the binary needs ARMv8.2 dot-product
  (`-march=armv8.2-a+dotprod`); ARMv8.0 phones get SIGILL. The app pins
  CPUs 4–7 with 4 threads (`taskset f0`), which suits the Pixel 6a's
  Tensor G1 layout only.
- **WebGPU:** no preflight against device memory before uploading the model;
  `kvput` and `gdn` use exactly 8 storage buffers, the default per-stage
  limit.

### Long context

- **CUDA and Metal:** hard ceiling 2^17 = 131,072 tokens. The served prompt is
  capped at 16,384 tokens (`CLEF_MAX_T`) even when the KV holds more. f32 KV
  is fixed at launch.
- **CPU:** KV capacity is fixed at launch; prompts are capped at 16,384
  tokens in `kev.c`.
- **WebGPU shipped:** 4,096 (canonical kernels are emitted per context
  length); the turbo decode path rewrites the whole KV cache each token.

## Cheapest wins, by engine

| Engine | Change | Class |
|---|---|---|
| CUDA, Metal | Read vocab size, full-attention interval, KV heads, rope base and dims, and eps from the model, not literals | Cap |
| CUDA, Metal | Size scratch (`gpu_blk`, `gpu_atts`, `gpu_q8b`, `M9_*`, `F27_TB`) from the shape; turn silent overflows into refusals | Cap |
| CUDA, Metal | Heap-allocate the 64-entry layer arrays | Cap |
| CUDA, Metal | Make the padded 8-leaf rmsnorm tree the default for any width | Kernel |
| Metal | Use the stride-parameter attention (`llb_m_attn5x`) for every KV-head count | Kernel |
| CUDA, Metal | Instantiate head_dim 128 and SSM state 64 | Kernel |
| CUDA, Metal | f16 KV, allocated by position | Kernel |
| WebGPU shipped | Use the in-place KV write for decode | Cap |
| All servers | Replace the 16,384-token prompt arrays with buffers sized by the KV capacity | Cap |
| Android app | Choose the thread count and CPU mask from `/sys/devices/system/cpu/*/cpu_capacity` | Cap |

## Done

- **CPU engine, patch 40:** per-layer arrays sized from the model; KV only for
  full-attention layers; recurrent state only for linear layers; F16 and
  IQ4_XS refused at load instead of computing zeros; tensor cap 1,024 → 4,096.
- **CPU engine, patch 41:** `kev.c` reads the head width from the decide
  head; the head is optional, so chat runs on any Qwen3.5 pack.
- **WebGPU trunk:** any Qwen3.5 shape including MoE, 21 GGUF formats, f16 KV
  with split-KV attention (context bounded by the binding limit), batched
  prompt submissions; the old 16,384-token tokenizer cap is gone
  ([report](../reports/webgpu-any-shape/REPORT.md)).
