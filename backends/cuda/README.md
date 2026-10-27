# CUDA

Linux, NVIDIA GPU, CUDA 13.x. Tested on an RTX 3090.

## Source

`engine_qwen35_gpu.bend`, `kernels/`, `host/`, and the compiler patches in
`tools/build_gpu.sh`. All in this repository.

## Build and run

```sh
LLB_BEND=BEND_CHECKOUT ./build.sh
tools/run27b.sh --gguf MODEL.gguf --gpu 40GB --threads 16 -- SIDECAR 760 6511 314 --steps 64
```

## Models and context

- **Shapes:** any dense Qwen3.5 GGUF with head_dim 256 and ssm_state_size
  128, n_heads a multiple of n_kv_heads, value heads a multiple of key heads
  (at most 256), widths a multiple of 256, d_model at most 32,768, at most
  256 layers, vocab at most 261,632. Tested: 0.8B, 2B, 4B, 9B. A model
  outside the check is refused at load with the field named.
- **Context:** prompt + `--steps` up to 2^26 positions. The f32 KV of the
  full-attention layers is allocated by position, up to prompt + steps, so
  VRAM is the bound. Use `LLB_PREFILL=fast` for long prompts.
- **MoE (`qwen35moe`):** any expert count, top-k (at most the expert count),
  expert and shared-expert width (multiples of 256), with the dense shape
  rules above for the rest; MTP blocks (`nextn_predict_layers`) are skipped.
  Experts in Q8_0, Q4_K, Q5_K or Q6_K, router and shared-expert gate F32.
  Load it from the GGUF: `gguf2pack MODEL.gguf --sidecar-only SIDE`, then
  `LLB_GGUF=MODEL.gguf llb-engine SIDE ...`. The expert stacks are held in
  VRAM, in pinned device-mapped host RAM, or streamed from the mmapped GGUF
  (the chosen experts copied each token), whichever the free VRAM and
  MemAvailable allow; `LLB_MOE_EXPERTS=device|host|stream` forces one. All
  three give the same bits. Checked against llama.cpp on synthetic files only
  (no real MoE checkpoint was loaded): [reports/cuda-moe](../../reports/cuda-moe/REPORT.md).
  MoE runs the exact tier by default (spec, batch, MTP are dense only) and
  the folded step (no `--emit`). Two opt-in tiers change the numerics and are
  gated against llama.cpp ([reports/cuda-catchup](../../reports/cuda-catchup/REPORT.md)):
  `LLB_MOE_DP4A=1` (the routed experts on int8 activations, dp4a) and
  `LLB_PREFILL=fast` (the prompt's tokens grouped by expert, each expert's rows
  a GEMM on the fast prefill's int8 tensor-core leaves; not with streamed
  experts).
- **Decode loop:** gen hands the host up to n greedy steps in one bang; the
  host runs them back to back as CUDA graphs (the next step's token read from
  the device's pick), so the Bend loop runs once a run, not once a token. Same
  bits as before; off for the sampler, grammar, spec, MTP, batch, images,
  dumps and the prefix cache, or with `LLB_GEN=0`.
- **Speed** (RTX 3090, decode tok/s, medians of 3 interleaved rounds against
  llama.cpp CUDA, build 9d57ce45; exact tier unless noted):

  | model | engine | llama.cpp | ratio |
  |---|---|---|---|
  | Qwen3.5-0.8B Q4_K_M | 483 | 463 | 1.04x |
  | Qwen3.5-2B Q4_K_M | 300 | 329 | 0.91x |
  | Qwen3.5-4B Q4_K_M | 135 | 175 | 0.78x |
  | Qwen3.5-9B Q4_K_M | 92 | 115 | 0.80x |
  | MoE 35B-A3B-like, 4 layers (synthetic) | 1733 (dp4a 1787) | 1687 | 1.03x (1.06x) |
  | MoE 128 experts, fused gate/up, 4 layers (synthetic) | 2410 (dp4a 2564) | 2096 | 1.15x (1.22x) |
  | MoE 35B-A3B-like, 8 layers (synthetic) | 904 (dp4a 933) | 912 | 0.99x (1.02x) |

  - Each added MoE layer costs about 132 us (dp4a 128), against llama.cpp's 126.
  - MoE prefill, 512 tokens on the 35B-like file: `LLB_PREFILL=fast` 31,900
    tok/s against llama.cpp's 25,000 (1.28x); the exact tier still goes one
    token at a time (704 tok/s).
  - Experts in host RAM: 305 tok/s; streamed from the GGUF in the page cache: 271.
  - Details: [reports/cuda-catchup](../../reports/cuda-catchup/REPORT.md),
    [reports/cuda-moe-speed](../../reports/cuda-moe-speed/REPORT.md).
- **Not on every shape:** `LLB_SPEC`, `LLB_MTP`, and `LLB_BATCH` run on the
  9B and 27B shapes only (d_model 4096 / 5120, 4 KV heads, context up to
  131,072); elsewhere they are refused.

Details and remaining limits: [docs/limits.md](../../docs/limits.md).

## Result

Qwen3.8-27B Q4_K_M: 38.8 tok/s decode, 1.011x llama.cpp.

Full guide: [Run on CUDA](../../docs/guides/run-on-cuda.md).
