# Model and hardware limits

What each engine accepts today, what stops it from running other model sizes
or smaller hardware, and how hard each limit is to lift. Every engine runs the
Qwen3.5 hybrid architecture (linear-attention layers plus a full-attention
layer every fourth layer). The dense sizes run on every engine; the MoE
models (`qwen35moe`) run on the WebGPU trunk, on CUDA (any shape; synthetic
files only so far) and on Metal (any shape; the 35B-A3B and synthetic files).

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
| CUDA (root) | Any dense Qwen3.5 shape with head_dim 256 and SSM state 128 (structural check; tested 0.8B, 2B, 4B, 9B); any `qwen35moe` with those rules (synthetic files) | prompt + steps, up to 2^26 positions; in practice VRAM | Dense: whole model in VRAM. MoE: trunk in VRAM; routed experts in VRAM, pinned host RAM, or streamed from the mmapped GGUF. f32 KV for full-attention layers, allocated by position up to prompt + steps |
| Metal, macOS (`backends/metal`) | Any dense Qwen3.5 quant pack (shape read from the manifest; tested 0.8B, 2B, 4B, 9B); any `qwen35moe` GGUF (expert count, top-k, widths from the file; tested 35B-A3B and four synthetic shapes) | 2^27 slots, bounded by the largest Metal buffer (the registered 35B pack's own arm: 131,072) | Trunk weights in one heap span (`--gpu`); routed experts mapped from the GGUF (no-copy chunks, or streamed when they exceed the GPU working set); f32 KV and attention planes in the engine's own buffer, sized from the context |
| Metal, iOS (`backends/ios`) | Same engine as macOS; shipped with 4B and Kev-4B | 2^27 (`llb_open` cap), bounded by the device's largest buffer | One span: GGUF + GGUF/2 + 1 GiB; KV outside it |
| WebGPU, shipped (`backends/webgpu/engine`) | 4B only | 4,096 | Whole model on the GPU, split into ≤128 MiB buffers |
| WebGPU, trunk (`backends/webgpu/trunk`) | Any Qwen3.5 shape, dense or MoE (head, state, conv read from the file); 21 GGUF formats | Bounded by the GPU's binding limit for one layer's f16 KV (about 2M tokens for the 0.8B on an RTX 3090) | Weights split by rows (or experts) to fit the device |

## What blocks each case

### Smaller models (0.8B, 2B)

- **CPU:** runs them. Qwen3.5-2B Q4_K_M was tested (see
  [backends/android](../backends/android/README.md#models)).
- **CUDA:** runs them (`tools/payload-anyshape.patch`). The whitelist is a
  structural check (`mf.validate` / `mf.shape` in `engine_qwen35_gpu.bend`,
  backed by `pass_mail`'s envelope): head_dim 256 and ssm_state_size 128
  (kernel instantiations), n_heads a multiple of n_kv_heads, value heads a
  multiple of key heads (at most 256), every width a multiple of 256,
  d_model at most 32,768, at most 256 layers, vocab at most 261,632. A
  failed check names the field. Rmsnorm handles any width (8-leaf tree to
  8,192, `rms_wide` past it). Rows that no tuned GEMV body covers (for
  example the 4B's 2,560-column Q5_K `attn_qkv`) run `llb_gemv_any`. Fast
  prefill runs on any shape. Speculative decoding and native batch stay
  refused outside the 9B and 27B shapes: `llb_t_attn4` has a 1,024-float
  KV stride and the spec embed and `bx_*` scratch are sized for those
  shapes.
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

### MoE (`qwen35moe`) on CUDA

`tools/payload-moeany.patch` + `host/llb_host.c` (`llb_moe_run`,
`llb_moe_load`), with `gguf2pack --sidecar-only` writing the MoE sidecar.
Any expert count and top-k, any expert and shared width, MTP blocks
skipped; checked on synthetic random-weight files against llama.cpp
([reports/cuda-moe](../reports/cuda-moe/REPORT.md)). What still bounds it:

- **Kernel:** expert and shared widths are whole 256-column superblocks;
  experts Q8_0 / Q4_K / Q5_K / Q6_K; router and shared-expert gate F32;
  head_dim 256, SSM state 128 and the other dense rules above.
- **Cap:** the trunk of one layer (everything but its expert stacks) is one
  Bytes of at most 2 GiB with tensor offsets below 2^28 (table.bin's 28-bit
  field); the head region at most 2 GiB; 256 layers. A 397B-A17B trunk layer
  is far below these. The expert stacks are outside them: loaded by name
  with 64-bit offsets and sizes, one allocation a stack.
- **Hardware:** the trunk, head, KV and scratch must fit in VRAM. The experts
  go to VRAM when they fit the free VRAM less 1 GiB, else to pinned
  device-mapped host RAM when they fit MemAvailable less 4 GiB, else they
  stream from the mmapped GGUF (the k chosen experts of each layer copied to
  a k-slot VRAM staging area each token; disk and page-cache speed).
  `LLB_MOE_EXPERTS` forces a residency; `LLB_MOE_VRAM` / `LLB_MOE_RAM` stand
  in for the free amounts. One GPU only.
- **Refused:** fast prefill (the MoE file runs the exact tier, with a stderr
  note), `LLB_SPEC`, `LLB_MTP`, `LLB_BATCH`, the split path (`--emit`,
  `--lpb` < layers), and the pack path (the experts load from `LLB_GGUF`).

### Bigger models (above 27B)

- **Metal:** layer arrays sized 1,024; d_model ≤ 8192; d_ff ≤ 17,408; at
  most 64 query heads (the fixed 81,920-float block scratch is checked
  against the shape, with a message). Head region ≤ 2 GiB and 32-bit offsets
  remain for the trunk.
- **Metal MoE** (`cc/metal-moe-cc.patch`, `engine/metal-moe-engine.patch`):
  no cap on the expert count, top-k or the expert / shared widths. The MoE
  scratch is its own buffer sized from them; the routing kernel loops over
  the experts. The routed experts never enter the heap: each expert tensor
  is cut into no-copy chunks of the mapped GGUF, each within
  `maxBufferLength`, reached through a table of 64-bit GPU addresses; when the
  experts plus the heap exceed `recommendedMaxWorkingSetSize`, the engine
  streams them (only the k chosen experts of a layer are copied, each token).
  What still bounds a MoE: the trunk (every non-expert weight and the heads)
  must fit the heap span, one Metal buffer; one expert must fit one buffer
  (k of them when streamed); the dense trunk limits above (d_model ≤ 8192,
  64 q / 16 kv heads, 1,024 layers, vocab ≤ 250,048, head region ≤ 2 GiB,
  a layer's trunk < 2^28 bytes: the table's offset field); expert and shared
  widths whole blocks of their format (32 columns; 256 for a K-quant);
  formats: experts F32 / F16 / BF16 / Q8_0 / Q4_K / Q5_K / Q6_K, trunk Q8_0
  and K-quants (BF16 or IQ trunk tensors are refused by name, requantize).
  A 122B-A10B or 397B-A17B file was not run (none on the test Mac); the split
  and streamed paths were forced on smaller models (`LLB_MX_SPLIT`,
  `LLB_MX_STREAM`) and gave the resident path's bits.
- **CUDA:** layer tables hold 256 layers (`LLB_MAXL`); scratch (`gpu_blk`,
  `gpu_q8b`, `gpu_q8c`, `gpu_headh`, `gpu_atts`) is sized from the shape at
  the first pass. Remaining limits: d_model ≤ 32,768, the head region
  (output + norm) ≤ 2 GiB, 32-bit layer offsets, vocab ≤ 261,632 (the
  logit buffer). The 27B was not re-run for this change (it needs 19 GB of
  VRAM).
- **CPU:** no shape cap. Memory is the limit: the pack is mmapped and must be
  mostly resident for speed.

### Less memory (8–12 GB GPUs, 8 GB Macs, 6–8 GB phones)

- **CUDA:** the whole model must fit in VRAM (no offload). KV is f32,
  allocated by position (doubling, up to prompt + steps); a failed
  allocation stops with the slot count and MiB. With LLB_SPEC, LLB_MTP, or
  LLB_BATCH the full capacity is allocated at the first step. Turing and
  older cards are out (sm_80 floor, a hardware class).
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
- **WebGPU trunk:** the whole model is resident (no offload);
  `trunk_resident(check, ctx)` estimates it before loading. KV is f16.
- **WebGPU shipped:** no preflight against device memory before uploading the model;
  `kvput` and `gdn` use exactly 8 storage buffers, the default per-stage
  limit.

### Long context

- **CUDA:** prompt + steps ≤ 2^26 positions (`c` carries the position in
  27 bits); the practical bound is VRAM for the f32 KV. Attention score
  planes are 2^max(dT, 17) floats a head. The served generate prompt is
  bounded by the KV capacity fixed at launch (`--steps`), not `CLEF_MAX_T`;
  `decide` keeps its 16,384-token budget. The engine no longer allocates
  unused Bend-side K/V arrays on the folded step (2^(dT + 13) bytes of host
  memory a layer before; the split path, `--emit` / `--lpb`, still does). Decode attention is one block a
  head, so a token at 100k+ positions is slow; use `LLB_PREFILL=fast` for
  long prompts.
- **Metal:** up to 2^27 slots (the step word's 27 position bits), bounded by
  the largest Metal buffer: f32 KV is 2 × n_kv_heads × 256 × 4 bytes a slot
  per full-attention layer (0.8B: 24 KiB a slot, so 2^21 slots fit 58 GiB).
  `--ctx C` asks for C slots; past the buffer limit the engine refuses with
  the bytes it needs. A 153,031-token prompt ran on the 0.8B (needle found).
  The CLI takes `@FILE` for prompts past the OS argument limit; the
  `generate` request of `link/dual.c` and `server/src/dualm.c` takes prompts
  up to the KV capacity. `decide` keeps CLEF's 16,384 budget. MoE models on
  the general path have the same range (the 35B-A3B found a needle at 153,352 tokens; `--ctx 1048576` runs, `--ctx 16777216` is refused with the 642 GiB it needs); their prefill is exact,
  one token at a time (no MoE fast chunk). The registered 35B pack's own arm
  stays at 2^17.
- **CPU:** KV capacity is fixed at launch; prompts are capped at 16,384
  tokens in `kev.c`.
- **WebGPU shipped:** 4,096 (canonical kernels are emitted per context
  length); the turbo decode path rewrites the whole KV cache each token.
- **WebGPU trunk:** f16 KV, split-KV attention over 256-token chunks; the
  bound is the binding limit for one full-attention layer's KV
  (`e.max_ctx`: 2,096,896 for the 0.8B and 2B, 1,048,320 for the 4B and 9B
  on an M3 Ultra). Needles were found at 153,352 tokens on the 0.8B on an
  RTX 3090 and an M3 Ultra. Prefill is one decode pass a token, so a 153k
  prompt takes 22 to 26 minutes.

## Cheapest wins, by engine

| Engine | Change | Class |
|---|---|---|
| Metal | Fast chunk attention for GQA other than 4 and 6, or more than 4 KV heads | Kernel |
| Metal | A fast prefill chunk for the general MoE path (grouped expert GEMM); faster MoE decode kernels | Kernel |
| CUDA, Metal | Instantiate head_dim 128 and SSM state 64; rope base, eps as kernel arguments | Kernel |
| CUDA, Metal | f16 KV (CUDA now allocates f32 KV by position) | Kernel |
| WebGPU shipped | Use the in-place KV write for decode | Cap |
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
- **CUDA MoE, `tools/payload-moeany.patch`:** any `qwen35moe` shape; experts
  in VRAM, pinned host RAM, or streamed from the GGUF; dense outputs
  byte-identical to the base build ([report](../reports/cuda-moe/REPORT.md)).
- **Metal any MoE** (`backends/metal/cc/metal-moe-cc.patch`,
  `engine/metal-moe-engine.patch`, `gguf2pack --sidecar-only --target metal` for
  `qwen35moe`): any expert count, top-k, expert and shared width, fused or
  separate gate/up, NextN blocks dropped; experts mapped from the GGUF in
  buffer-sized chunks or streamed past the GPU working set; context to 2^27
  slots. The 35B-A3B Q4_K_S matches llama.cpp's own Metal-vs-CPU agreement
  (teacher-forced top-1 187/192 for both); the MoE FFN matches a float64
  oracle to 1e-6. Dense 4B / 9B ids and logits unchanged
  ([report](../reports/metal-moe/REPORT.md)).
- **Metal any-shape** (`backends/metal/cc/metal-anyshape-cc.patch`,
  `engine/metal-anyshape-engine.patch`, `link/dual-anyshape.patch`): any
  dense Qwen3.5 quant-pack shape (structural manifest check), any KV-head
  count, rmsnorm at any width, K-quant GEMVs at depth 8–13, scratch and KV
  sized from the shape and context, 1,024-entry layer arrays, context up to
  2^27 slots with a buffer-limit refusal, `--ctx C`, `@FILE` prompts, served
  prompts up to the KV capacity, and an iOS span sized from the model.
  4B and 9B greedy ids and logits are unchanged against the base build.
- **CUDA engine, `tools/payload-anyshape.patch`:** structural shape check
  instead of the 9B / 27B / 0.8B whitelist; vocab, full-attention layers
  (from the table), rope base and dims, and eps read from the model
  (`llb_cfg` on the device; the defaults are the old literals, so the
  supported models keep their bits); rmsnorm for any width; scratch sized
  from the shape; `llb_gemv_any` for rows outside the tuned GEMV bodies;
  KV allocated by position; context up to 2^26 positions; fast prefill on
  any shape; the SYS1 generate prompt bounded by the KV capacity.
