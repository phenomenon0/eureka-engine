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
| Metal, macOS (`backends/metal`) | Any dense Qwen3.5 quant pack (shape read from the manifest; tested 0.8B, 2B, 4B, 9B), 35B MoE | 2^27 slots, bounded by the largest Metal buffer (35B MoE: 131,072) | Weights in one heap span (`--gpu`); f32 KV and attention planes in the engine's own buffer, sized from the context |
| Metal, iOS (`backends/ios`) | Same engine as macOS; shipped with 4B and Kev-4B | 2^27 (`llb_open` cap), bounded by the device's largest buffer | One span: GGUF + GGUF/2 + 1 GiB; KV outside it |
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
- **Metal:** runs them since `cc/metal-anyshape-cc.patch` and
  `engine/metal-anyshape-engine.patch` (applied by default; `LLB_METAL_BASE=1`
  builds the old product). `mf.dense` checks the quant pack's manifest
  structurally and passes the head counts, vocabulary and layer count to the
  host (`m9_shape_w`), which sizes its scratch from them. Attention takes any
  KV-head count (`llb_m_attn6` and its split; the fast chunk has GQA 4 or 6
  over 2 or 4 KV heads), rmsnorm uses the padded tree at every width but 2048
  and 4096, and K-quant GEMVs take any multiple of 256 columns at depth 8–13.
  0.8B and 2B Q4_K_M were checked against llama.cpp (see
  [backends/metal](../backends/metal/README.md#any-shape)). Still refused, by
  name: head_dim other than 256, SSM state other than 128, a conv other than
  4 taps, a full-attention interval other than 4, rope other than 1e7 over 64
  dims, eps other than 1e-6 (all compiled into the kernels), d_model above
  8192, d_ff above 17,408, vocabularies above 250,048, and the Q8 pack
  (`pack.bin` format) at a new shape.
- **WebGPU shipped:** `q35_tensors` asserts the 4B geometry
  (`engine/q35_plan.js` 30–34), and the emitted kernels bake in its widths,
  head counts, and vocabulary.
- **WebGPU trunk:** runs them (0.8B and 2B tested against llama.cpp; see
  [reports/webgpu-any-shape](../reports/webgpu-any-shape/REPORT.md)).

### Bigger models (above 27B)

- **Metal:** layer arrays sized 1,024; d_model ≤ 8192; d_ff ≤ 17,408; at
  most 64 query heads (the fixed 81,920-float block scratch is checked
  against the shape, with a message). Head region ≤ 2 GiB and 32-bit offsets
  remain.
- **CUDA:** layer arrays sized 64; d_model ≤ 8192 (head scratch);
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
- **Metal and iOS:** one no-copy `MTLBuffer` holds weights and tables; the
  KV, attention planes and fast-chunk area are a second buffer sized from the
  context, checked against `maxBufferLength` (58.3 GiB on the 96 GB M3
  Ultra). Its pages are committed as the slots fill. The iOS default span is
  GGUF + GGUF/2 + 1 GiB (`backends/ios/engine/engine/llb_engine.m`
  `llb_span`); measured floors at ctx 4096: 0.8B 1.6 GB, 2B 2.6 GB, 4B
  4.5 GB, 9B 8.8 GB. Weight blobs round up to power-of-two size classes, up
  to 2× on large layers. KV stays f32.
- **CPU (Android):** the binary needs ARMv8.2 dot-product
  (`-march=armv8.2-a+dotprod`); ARMv8.0 phones get SIGILL. The app pins
  CPUs 4–7 with 4 threads (`taskset f0`), which suits the Pixel 6a's
  Tensor G1 layout only.
- **WebGPU:** no preflight against device memory before uploading the model;
  `kvput` and `gdn` use exactly 8 storage buffers, the default per-stage
  limit.

### Long context

- **CUDA:** hard ceiling 2^17 = 131,072 tokens. The served prompt is
  capped at 16,384 tokens (`CLEF_MAX_T`) even when the KV holds more. f32 KV
  is fixed at launch.
- **Metal:** up to 2^27 slots (the step word's 27 position bits), bounded by
  the largest Metal buffer: f32 KV is 2 × n_kv_heads × 256 × 4 bytes a slot
  per full-attention layer (0.8B: 24 KiB a slot, so 2^21 slots fit 58 GiB).
  `--ctx C` asks for C slots; past the buffer limit the engine refuses with
  the bytes it needs. A 153,031-token prompt ran on the 0.8B (needle found).
  The CLI takes `@FILE` for prompts past the OS argument limit; the
  `generate` request of `link/dual.c` and `server/src/dualm.c` takes prompts
  up to the KV capacity. `decide` keeps CLEF's 16,384 budget. The 35B MoE
  arm stays at 2^17.
- **CPU:** KV capacity is fixed at launch; prompts are capped at 16,384
  tokens in `kev.c`.
- **WebGPU shipped:** 4,096 (canonical kernels are emitted per context
  length); the turbo decode path rewrites the whole KV cache each token.

## Cheapest wins, by engine

| Engine | Change | Class |
|---|---|---|
| CUDA | Read vocab size, full-attention interval, KV heads, rope base and dims, and eps from the model, not literals | Cap |
| CUDA | Size scratch (`gpu_blk`, `gpu_atts`, `gpu_q8b`) from the shape; turn silent overflows into refusals | Cap |
| CUDA | Heap-allocate the 64-entry layer arrays | Cap |
| CUDA | Make the padded 8-leaf rmsnorm tree the default for any width | Kernel |
| Metal | Fast chunk attention for GQA other than 4 and 6, or more than 4 KV heads | Kernel |
| CUDA, Metal | Instantiate head_dim 128 and SSM state 64; rope base, eps as kernel arguments | Kernel |
| CUDA, Metal | f16 KV, allocated by position | Kernel |
| WebGPU shipped | Use the in-place KV write for decode | Cap |
| CUDA servers | Replace the 16,384-token prompt arrays with buffers sized by the KV capacity | Cap |
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
- **Metal any-shape** (`backends/metal/cc/metal-anyshape-cc.patch`,
  `engine/metal-anyshape-engine.patch`, `link/dual-anyshape.patch`): any
  dense Qwen3.5 quant-pack shape (structural manifest check), any KV-head
  count, rmsnorm at any width, K-quant GEMVs at depth 8–13, scratch and KV
  sized from the shape and context, 1,024-entry layer arrays, context up to
  2^27 slots with a buffer-limit refusal, `--ctx C`, `@FILE` prompts, served
  prompts up to the KV capacity, and an iOS span sized from the model.
  4B and 9B greedy ids and logits are unchanged against the base build.
