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
- **Not on every shape:** `LLB_SPEC`, `LLB_MTP`, and `LLB_BATCH` run on the
  9B and 27B shapes only (d_model 4096 / 5120, 4 KV heads, context up to
  131,072); elsewhere they are refused.

Details and remaining limits: [docs/limits.md](../../docs/limits.md).

## Result

Qwen3.8-27B Q4_K_M: 38.8 tok/s decode, 1.011x llama.cpp.

Full guide: [Run on CUDA](../../docs/guides/run-on-cuda.md).
