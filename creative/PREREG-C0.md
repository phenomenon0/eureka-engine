# PREREG-C0 — 100% Native Eureka Creative Engine (`creative/`) Foundation Gates

**Date:** 2026-11-23  
**Base SHA:** `7695e8d` (`eureka-engine` `main`)  
**Branch:** `creative/c0-foundation`  
**Doctrine:** Governed by `Verdict Gates v2` and `TOLERANCE-DOCTRINE.md` (`--exact` = oracle, tolerance-gated fast tiers ship labeled).  
**Ownership Contract:** **100% pure Eureka Engine code** (C11 + CUDA PTX Tensor Cores, zero `ggml`, zero `stable-diffusion.cpp`, zero cuBLAS, zero Python runtime).

---

## 1. Frozen Acceptance Gates (Set Before Any Run)

### G-C0.1 — Numerical Oracle Correctness (Exact Tier vs CPU F64 Reference)
Every creative primitive MUST pass against a double-precision (`double`) CPU oracle on non-trivial seeded inputs:
- **Fused `adaLN-Zero` + RMSNorm/LayerNorm (`ec_adaln_zero`)**: `max_abs_err <= 2e-5` (F32 out).
- **Fused Gated Residual (`ec_gated_res`)**: `max_abs_err <= 1e-6`.
- **N-D RoPE (1D Audio, 2D Image, 3D Video/Voxel) (`ec_rope_nd`)**: `max_abs_err <= 2e-5`.
- **Tensor-Core FP16 GEMM (`ec_gemm_f16`, `mma.sync.aligned.m16n8k16`)**: `max_abs_err <= 2e-3`, `rel_L2 <= 5e-4` (matching F16 input rounding).
- **Hot-Fused LoRA (`ec_lora_add`)**: `max_abs_err <= 2e-3` vs `(X @ A^T) @ B^T * (alpha/r)`.
- **Bidirectional & Block-Causal FlashAttention-2 (`ec_fa2_f16`, `D=64, 128`)**: `max_abs_err <= 3e-3`, `rel_L2 <= 1e-3`, `cosine_sim >= 0.99999`.
- **GroupNorm + SiLU (`ec_groupnorm_silu`)**: `max_abs_err <= 2e-5`.
- **Causal Conv1D (Audio DCAE), Conv2D (Image VAE), Causal Conv3D (Video VAE)**: `max_abs_err <= 3e-3`, `rel_L2 <= 5e-4`.
- **Sparse 3D O-Voxel Hash + Submanifold 3x3x3 Conv (`ec_sparse_conv3d`, TRELLIS.2)**: `max_abs_err <= 2e-4`, `100%` active-coordinate lookup parity.
- **Rectified-Flow CFG Euler Step (`ec_flow_cfg_euler`)**: `max_abs_err <= 1e-5`.
- **Realtime Video Self-Forcing Rolling KV-Cache (`ec_kv_ring_write` + causal chunk attention)**: `max_abs_err <= 3e-3` vs full dense sliding-window reference.

### G-C0.2 — Tolerance-Gated Fast Tier (`EC_TIER_FAST`: SageAttention INT8 Q/K + CUDA Graph + MagCache)
- **SageAttention INT8 Q/K (`ec_sage_attn_i8`, PTX `m16n8k32` s8×s8→s32 + `m16n8k16` P·V)** vs `ec_fa2_f16` oracle on `N=4096, H=24, D=128` (Wan/FLUX/LTX shape):
  - `cosine_sim >= 0.998`
  - `rel_L2 <= 0.030`
  - Paired GPU speedup on RTX 3090 (`sm_86`): **`>= 1.20x`** over `ec_fa2_f16` (paired runs under house locks).
- **CUDA Graph Replay (`ec_dit_run_graph`)** vs eager `ec_dit_run_eager`:
  - `max_abs_err <= 1e-5` (bit-exact/near-exact numerical identity).
  - Launch overhead reduction measurable on short/realtime video chunk shapes.
- **MagCache Dynamic Timestep Skip (`ec_magcache`)** on 20-step DiT trajectory:
  - Deterministic skip schedule under calibrated `delta_thresh`; speedup **`>= 1.40x`** with trajectory `rel_L2 <= 0.05` vs full 20-step oracle.

### G-C0.3 — Engine Infrastructure & End-to-End Closed Cycle Smoke
- **Safetensors + `AUX1` Loader (`ec_safetensors.c`)**: Round-trip valid multi-dtype (`F32, F16, BF16, F8_E4M3, I8`) file + reject truncated, out-of-bounds, overlapping-header, and integer-overflow malformed inputs cleanly (0 crashes).
- **Compiled DAG Engine (`ec_flow.c`, ComfyUI JSON compatible)**:
  - Parses ComfyUI prompt JSON graph (`class_type` + `["node_id", slot]` links), computes topological order + static VRAM liveness free-points.
  - Content-addressed cache: second identical run achieves `100%` cache hits (`0` node executions); mutating one leaf node re-executes *only* downstream dependents.
- **Native Exporters & Zero-Copy SHM Ring (`ec_media.c`)**:
  - Exports valid binary **GLB 2.0** (glTF magic `0x46546C67`, JSON+BIN chunks 4-byte aligned, parseable header).
  - Exports valid **3D Gaussian Splat `.ply`** and **16-bit Stereo `.wav`** (RIFF/WAVE header validated) and **PPM** image.
  - **Zero-copy POSIX SHM Ring (`ec_shm_ring`)**: 60 frames of 1280×720 RGBA pushed & consumed with `0` drops, `100%` checksum match, and measured throughput `>= 60 FPS`.

---

## 2. Build & Execution Hygiene
- All builds occur strictly in `/tmp/eureka-creative-build` (never inside `/home/omen/worktrees/eureka-engine`).
- All GPU verification/benchmarks execute under both house locks (`/tmp/bend-bench.lock` and `/tmp/llb-gpu-bench.lock`, `<= 30s` bounded slices).
- Receipts saved under `/tmp/eureka-creative-build/receipts/c0-receipt.txt` and summarized in `creative/REPORT-C0.md`.
