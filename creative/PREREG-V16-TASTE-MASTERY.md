# PREREG-V16-TASTE-MASTERY — Eureka Creative Engine v16 ("Taste & Mastery")

**Status:** FROZEN BEFORE IMPLEMENTATION & BENCHMARK RUNS
**Date:** 2026-11-24
**Target Hardware:** NVIDIA GeForce RTX 3090 (24 GB VRAM, Ampere `sm_86`), `/home` NVMe (`69 GB` free budget -> keep `>= 35 GB` free after curated weights).
**Governance:** Verdict Gates v2 (`~/agent-lib/entries/verdict-gates-v2.md`) & Tolerance Doctrine (`~/agent-lib/TOLERANCE-DOCTRINE.md`).

## 1. Objective
Transform all 4 creative modalities (Images, Cinema Video, Studio Music, 3D Assets) in Eureka Creative Engine + Creative Studio from underwhelming demo defaults to curated, museum/studio-grade outputs by fixing every verified numerical, color-science, audio-DSP, geometry, tokenizer, and UX guardrail defect uncovered by our 9-agent audit, and integrating high-ROI Apache-2.0 / open-weights upgrades (`FLUX.2-klein-base-4B`, `Z-Image-Turbo 6.15B S3-DiT`, `ACE-Step 1.5 SFT`, and curated aesthetic LoRAs).

## 2. Frozen Acceptance Gates (ALL Must PASS)
| Gate | Metric | Frozen Threshold | Method / Receipt |
| :--- | :--- | :--- | :--- |
| **G1 — Image Engine Parity & Speed** | `FLUX.2-klein-4B` warm `1024×1024` (Fast tier `Q8_G64 + K-smoothed SageAttn-I8`) AND `Exact BF16 + BF16 FA2` + Qwen3 pad-mask fix (`flux2_qwen3_v2`) + `ec_bpe.c` Qwen3 regex | Fast tier `<= 2,250 ms` warm `1024×1024` (zero regression vs ComfyUI `3,200 ms`); Exact BF16 `<= 2,600 ms`; Qwen3 pad-token cosine parity vs HF `>= 0.999`; zero filename collisions across parameter sweeps | `ec-bench` + `bench-v16-taste.py` |
| **G2 — Image Taste & SOTA Engines** | `Z-Image-Turbo 6.15B` native CUDA BF16 (`1024×1024`, 8 steps) + `FLUX.2-klein-base-4B` (28-step CFG-Zero*) + 2-Pass Lanczos Pixel HiRes (`1024 -> 1536/2048`) | `Z-Image-Turbo` warm `1024×1024 <= 5,500 ms` BF16 resident on 3090; visually inspected PNGs on `"200 dollar bill"`, `"kazakh girl"`, Magnum portrait, typography poster, and fine-art painting show crisp engraving/eyes/skin pores/text with zero plastic blowout | Visual PNG inspection + `bench-v16-taste.py` |
| **G3 — Cinema Video Color & Motion** | `ec_nvenc.c` BT.709 matrix (`scale=out_color_matrix=bt709:out_range=tv`) + `-b:v 0 -cq 16` + `ec_wan22.cu` official bilingual neg prompt (`色调艳丽`), fixed `AvgDown3D` `factor_t`, `RES-Multistep` + CFG-Zero*, `Hero-Keyframe -> I2V` at `1280×704 @ 24 fps` | `ffprobe` confirms `color_space=bt709`, `color_range=tv`, bitrate `> 4.5 Mbps` on `1280×704` (no 2 Mbps cap); visually inspected keyframes free of neon-magenta shift & waxy plastic sheen | `ffprobe` + keyframe visual inspection |
| **G4 — Studio Music Precision & Mastering** | `AutoencoderOobleck` FP32 1D VAE + `ec_audio_master_studio` DSP (28 Hz HPF, anti-mud EQ, air shelf, `<120 Hz` bass mono, `-13.5 LUFS` norm, `-1.0 dBTP` true-peak limiter, TPDF dither) + `acestep_v15_sft_bf16` support | Integrated loudness `-15.0 <= LUFS <= -11.5`; True Peak `<= -0.8 dBTP`; DC offset `|mean| < 0.001`; non-silent RMS (`>= -20 dBFS`) on both Turbo (8-step) and SFT (32-step CFG) tracks | Python DSP measurement script + WAV receipts |
| **G5 — 3D Asset & Three.js PBR Stage** | `1024×1024` Telephoto 3/4 Softbox ref + alpha-preserving PNG crop + 2-pass Taubin smoothing + per-vertex `COLOR_0` linear albedo GLB + Three.js r187 `RoomEnvironment` PMREM ACES Turntable in `creativeStudio.tsx` | GLB contains `POSITION`, `NORMAL`, `TEXCOORD_0`, `COLOR_0`, valid bounds & non-shrunk geometry; Playwright headless Chrome renders Three.js PBR viewport with `0` console/page errors & visible PBR lighting/colors | GLB validator + Playwright visual inspection |
| **G6 — Zero Regression & Co-Residency** | Existing test suite across Chat+Image Co-Residency, Desktop Playwright, Linux Runtime, and Engine Bench | `ec-bench` `0` failures; Anti-OOM Chat+Images `7/7 PASS`; Playwright `8/8 PASS`; Linux Runtime `6/6 PASS`; `/home` free disk `>= 35 GB` | Saved receipts in `~/agent-lib/` |

## 3. Non-Negotiable Constraints
- Never modify or stage `creative/ec_cowrie.{c,h}` (other lane's work).
- Scratch builds only in `/tmp`; FF-only branch pushes recording before/after SHAs.
- Acquire `/tmp/bend-bench.lock` (`flock -w 30`) for all GPU benchmark runs; never `pkill`.
- Document all curated findings, before/after wipe comparators, audio/video/3D media, and creator recipes in one unified interactive Paged HTML in `~/agent-lib/` served on `:8795`.
