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

## Result

Qwen3.8-27B Q4_K_M: 38.8 tok/s decode, 1.011x llama.cpp.

Full guide: [Run on CUDA](../../docs/guides/run-on-cuda.md).
