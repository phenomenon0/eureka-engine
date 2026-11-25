# PREREG-V17-FRONTIER-PEAK — Eureka Creative v17 Frontier Peak (RTX 3090 24 GB)

**Status:** FROZEN BEFORE IMPLEMENTATION & BENCHMARKS (2026-11-24)  
**Governing Doctrine:**
- Verdict Gates v2 (`~/agent-lib/entries/verdict-gates-v2.md`) — class-shaped verdicts; paired forms decide; near-tie noise documented; drops carry numbers.
- Tolerance Doctrine (`~/agent-lib/TOLERANCE-DOCTRINE.md`) — byte-parity retired as universal gate; declared affordable tolerances + quality gates per modality.
- Elpis Margins (`lab/fabric/runs/wave-best/RULES-BETTER/ELPIS-MARGINS.md`) — stability-over-speed, explicit margins, receipts with method + path.

## 1. Frontier Upgrade Scope (Zero Python / Zero cuBLAS / Zero ggml — 100% Native Eureka CUDA)

1. **v17.1 Music Frontier — ACE-Step 1.5-XL 4B SFT (`ACE-Step/acestep-v15-xl-sft`, MIT)**
   - Architecture: Dynamic-shape DiT in `ec_acestep.cu` supporting BOTH 2B (`dim=2048, 24L, 16Q/8KV, ffn=6144`) AND 4B XL (`dim=2560, 32L, 32Q/8KV, ffn=9728`) alongside existing `Qwen3-0.6B` text encoder, `2048`-dim condition encoder (`8L` lyric + `4L` timbre), pure IEEE FP32 Oobleck 1D Stereo VAE, and studio mastering (`-13.5 LUFS / -1.0 dBTP`).
   - Storage format: Single-file BF16 weights (`acestep_v15_xl_sft_bf16.safetensors`, ~9.98 GB; 1D norms/biases/tables preserved in FP32).
   - **Pre-declared Gates (Paired A/B vs v16 ACE-Step 1.5 2B SFT on identical prompts & seeds):**
     - Zero NaN/Inf, zero CUDA errors, zero OOM on RTX 3090 24 GB.
     - True Peak `<= -0.90 dBTP`, Integrated Loudness within `[-15.0, -12.0] LUFS`, DC offset `|dc| < 1e-3`, non-silent RMS `[-20, -8] dBFS`.
     - Spectral & stereo width health: stereo correlation `< 0.98` (real stereo field, not collapsed mono), upper-band (`>6 kHz`) energy present without aliasing/clipping (`0` clipped samples).
     - Wall-clock budget: 20 s 48 kHz stereo master (36 steps SFT CFG) `<= 45 s` on RTX 3090.

2. **v17.2 3D Frontier — Full 3-Stage TRELLIS.2-4B O-Voxel PBR (`microsoft/TRELLIS.2-4B`, MIT)**
   - Architecture: Full 3-stage pure-CUDA pipeline in `ec_trellis2.cu`:
     - Stage 1: DINOv3-ViT-L/16 + `ss_flow` (1.3B) + `ss_dec` -> `64³` occupancy -> `32³` active sparse coordinates.
     - Stage 2: Sparse `SLatFlowModel` (`slat_flow_img2shape_dit_1_3B_512_bf16`, 30L, dim=1536, 12H, 32-ch shape latent, 3D sparse RoPE) -> `FlexiDualGridVaeDecoder` (`trellis2_shape_dec_fp16`, 4-stage sparse 3D ConvNeXt + octree `2×` subdivision `32³ -> 512³`, 7-ch Flexible Dual Contouring) -> sharp sub-voxel mesh.
     - Stage 3: Sparse `SLatFlowModel` (`slat_flow_imgshape2tex_dit_1_3B_512_bf16`, 64-ch in `[noise_32, shape_slat_32]` -> 32-ch tex latent) -> `SparseUnetVaeDecoder` (`trellis2_tex_dec_fp16`, matching octree coords -> 6-ch native 3D PBR `[base_color RGB, metallic, roughness, alpha]`).
     - Export: glTF 2.0 `.glb` with embedded PBR texture atlas (`baseColorTexture` + `metallicRoughnessTexture` PNGs or true per-vertex/UV PBR) + 3DGS `.ply`.
   - **Pre-declared Gates (Paired A/B vs v16 Stage-1 64³ Surface Nets on `d3_01_espresso_machine` & `d3_02_celadon_teapot`):**
     - Zero NaN/Inf, zero CUDA errors, zero OOM on RTX 3090 24 GB (auto-evict Stage-1 weights before Stage-2/3).
     - Genuine 3-stage mesh & material output: `n_verts >= 10,000`, `n_tris >= 20,000`, non-trivial spatially varying PBR albedo AND non-constant metallic/roughness across materials (e.g., brushed metal vs matte body / glossy ceramic glaze).
     - Visual verification in real Three.js r187 `RoomEnvironment` PMREM ACES PBR + Clay renders (`0` WebGL/console errors, zero cyclorama box, crisp sharp edges & 360° back-face texture fidelity — no 2D front-projection smear on the back!).
     - Wall-clock budget: Full 3-stage 512³ PBR asset `<= 120 s` on RTX 3090.

3. **v17.3 Video+Audio Frontier — Lightricks LTX-2.3 22B Distilled FP8 Joint Audio+Video (`Lightricks/LTX-2.3-fp8` + `Comfy-Org/ltx-2` Gemma-3-12B FP8)**
   - Architecture: 100% pure-CUDA `creative/ec_ltx2.{h,cu}` (zero Python, zero cuBLAS, zero ggml):
     - Text encoder: Streamed `Gemma-3-12B-IT` FP8-scaled extracting all 49 hidden states (`49 × 3840 = 188,160`) -> LTX-2.3 `video_aggregate_embed` (`4096`) & `audio_aggregate_embed` (`2048`) + 8-layer gated `EmbeddingsConnector` (with 128 learnable registers).
     - Joint Dual-Stream DiT: 48-layer `AVTransformer3DModel` (Video `32H×128=4096`, Audio `32H×64=2048`, bidirectional A/V cross-attention + Cross-AdaLN, split RoPE, gated attention, FP8 E4M3 weight-streamed/resident hybrid for 24 GB VRAM) running official 8-step Distilled schedule (`cfg=1.0`).
     - Decoders: Non-causal 3D Video VAE (`128` ch, spatial 32×, temporal 8×) + Audio 2D Causal Mel VAE (`8×16=128` -> stereo `64`-bin 16 kHz mel) + AMP1 Stereo Vocoder + 16 kHz -> 48 kHz BWE neural upsampler -> BT.709/tv NVENC CQ16 MP4 muxed with native frame-synchronized 48 kHz AAC audio.
   - **Pre-declared Gates:**
     - Zero NaN/Inf, zero CUDA errors, zero OOM on RTX 3090 24 GB (<= 22.5 GB peak VRAM).
     - `ffprobe` contract: `color_space=bt709`, `color_range=tv`, valid H.264 video stream + valid 48 kHz stereo AAC audio stream of matching duration (`|dur_a - dur_v| <= 0.25 s`).
     - Visual inspection of `start`, `mid`, `end` frames: zero black/static frames, clear temporal motion & prompt adherence.
     - Audio inspection: non-silent RMS `[-35, -6] dBFS`, peak `<= -0.5 dBTP`, synchronized energy envelope with visual events.
     - Wall-clock budget: 3–5 s clip (`768×512` or `832×480` @ 24 fps, 8 steps) `<= 120 s` on RTX 3090.

## 2. Non-Regression Gates (All Existing v16 Contracts Must Stay 100% GREEN)
- `ec-bench`: 0 failures.
- Anti-OOM Chat + Images stress test: 7/7 rounds PASS, 0 OOMs.
- Creative Studio Playwright (`test/creative-elite-v9.playwright.mjs`): 8/8 PASS, 0 console/page errors.
- Linux Runtime Verify (`scripts/verify-linux-runtime.mjs`): 6/6 PASS, 0 missing libs.
- Never modify or stage `creative/ec_cowrie.{c,h}`.
- Scratch builds only (`/tmp/eureka-creative-build`, `CARGO_TARGET_DIR=/home/omen/scratch/lp-rpm-target`).
- Unified Paged HTML Showcase served on `http://100.108.6.13:8795/` updated with verified v17 media & A/B receipts (0 headless Chrome errors).
