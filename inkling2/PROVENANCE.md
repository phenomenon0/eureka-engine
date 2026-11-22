# Inkling 2 Mini Native Engine (`inkling2/`)

This directory contains Eureka's native Inkling 2 Mini inference engine, ported from the MIT-licensed Strata `inkling2` shape-engine component.

## Upstream Provenance

- **Repository:** `https://github.com/Niko1221/Strata`
- **License:** MIT (`LICENSE` preserved verbatim in this directory)
- **Upstream Commit:** `c6f6427d9de9016fd92d13e29740f65625261df7`
- **Engine Version:** `strata-inkling 0.1.65`
- **Model Family:** Inkling 2 Mini (`inkling2_loop`: 30B total / 3.4B–5.4B active routed MoE, 24 physical layers × 48 visits with middle loop-core layers 6–17 executed 3×, 256k context, hybrid sliding-window + global GQA + ShortConv, optional lossless BF16 ViT vision shard)

## Key Speed & Memory Optimizations

1. **Loop-weighted 3× VRAM expert placement (`vram_load_order`)**: prioritizes middle loop-core experts (visited 3× per token) inside GPU VRAM before outer layers.
2. **CUDA-Graph mapped-memory doorbell hybrid offload (`hybrid_doorbell.cu`)**: executes cold RAM-resident experts on a lock-free CPU thread pool concurrently with GPU compute, enabling 4–12 GiB GPUs to run `SQ-10` without OOM.
3. **Fused on-GPU DSpark speculative decoding (`ink_dspark.cuh` / `ink_metal_dspark.metal`)**: W=4 adaptive beam tree / Block-8 Markov drafter with zero-logits-D2H verification (`alpha ~ 2.58–2.70`).
4. **Native `.strata` v1.1 packs (`SQ-10` 10.47 GB, `SQ-15` 14.35 GB, `SQ-20` 19.35 GB lossless) + direct GGUF & HF Safetensors loaders**, plus built-in ViT vision (`ink_vision.h` / `ink_metal_vision.metal`).

## Cross-Platform Build Commands (Scratch Builds Only)

- **Linux (CUDA + CPU):** handled by `backends/linux/build.sh` → stages `cuda/engine_inkling_gpu` and `cpu/engine_inkling_cpu`.
- **Windows (CUDA + CPU):** handled by `backends/windows/build.ps1` → stages `cuda/engine_inkling_gpu.exe` and `cpu/engine_inkling_cpu.exe`.
- **macOS Apple Silicon (Metal):** handled by `backends/metal/build.sh` and `bend-app/desktop/engine/metal/build.sh` → stages `metal/engine_inkling_metal`.
