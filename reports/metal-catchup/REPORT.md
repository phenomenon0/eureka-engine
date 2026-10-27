# Metal: MoE catch-up to llama.cpp Metal (decode + prefill) — report

Lane: METAL-CATCHUP, 2026-10-27. Branch `x/metal-catchup` off `feature/all-backends` `89dba9dde` (tree = published
eureka-engine main). Test machine: Mac Studio, M3 Ultra, 96 GB. Model: unsloth `Qwen3.5-35B-A3B-Q4_K_S.gguf`
(20,673,845,888 B, sha256 `ee93ceff…`, `receipts/35b.sha`). Reference: llama.cpp `e70802a01` on the same Mac.
Gates were frozen in [PREREG.md](PREREG.md) (commit `55a4e8bd9`, sha256 `d9c19db5…`) before any gated or timed
candidate run; no amendment.

## Profile (before)

Method: Metal System Trace (`xctrace`, command-buffer intervals, `tools/gpuint.py`, `tools/gp2.py`) and a dev build
that cuts the command buffer after every MoE stage and sums each piece's GPU time (kept as the `LLB_MX_PROF=1` knob).
Receipts: `receipts/xctrace-cb-timeline.txt`, `receipts/prof-mxprof.txt` (v1 kernels), `receipts/prof3*-mxprof.txt`.

- Resident decode has no per-layer CPU wait: 3 command buffers a token (0.16 ms; 1.0 ms = layers 0-1; 19.4 ms =
  layers 2-39 + head), the GPU busy about 97% of the 21.6 ms. The cost was GPU time, not scheduling.
- Per layer (GPU us, about 5 us of cut overhead each): trunk 139 (SSM 136, attention 149); MoE 383: routing 38,
  routed gate/up 168, routed down 131, shared gate/up 20, router 9, two addrms 8 each. The routed GEMVs ran at
  45-56 GB/s (32 KB of threadgroup memory a group: one group a core), and the routing and SiLU paid the engine's
  float64-emulated `exp` (`attn_exp`).
- A dependent dispatch costs about 3.7 us (10 extra empty dispatches a layer: +1.49 ms a token).

## What changed

`backends/metal/cc/metal-catchup-cc.patch` + `engine/metal-catchup-engine.patch`, applied after the MoE patches
(default emitted C `5a766ce4…`; `LLB_METAL_MOE1=1` reproduces the first MoE build `a20c8ee2`).

- **Decode kernels (v2).** Per layer `[router | shared gate/up]` -> `[routing | shared down]` -> routed gate/up ->
  routed down + combine (bracketed pairs side by side; was 7 serial dispatches). Routing: one threadgroup; max and sum
  by simd + one threadgroup step; top-k by rank counting (float4 compares, no reduction rounds); renormalized; sigmoid
  shared gate. Row dots on 256-column units of 8 lanes with the dense path's fused K-quant leaves (`kq_fused45` /
  `kq_fused6`), 2 rows a simdgroup sharing every x load; Q8_0 a block a lane; F32/F16/BF16 the generic chunk. `exp`
  is the hardware `precise::exp`. `LLB_MX_V1=1` keeps the v1 kernels and schedule.
- **Trunk GEMVs of a MoE model** take the same unit bodies (`llb_mx_gv`; dense models never do; `LLB_MX_TRUNK=0`).
- **MoE fast prefill chunk.** The engine sends a MoE prompt through the generic chunk (512 tokens; prompts of >= 13
  ids). Per layer the dense chunk's trunk, then: the router for all tokens (`llb_mx_crdot`), routing per token, a GPU
  grouping of the (token, slot) entries by expert (`llb_mx_cgroup`: counts, offsets, tiles of 32), gathered
  simdgroup-matrix GEMMs (f27_mm's 64 x 32 tile, f32 operands) for gate|up and down (`llb_mx_cmm{8,4,5,6}`: Q8_0,
  Q4_K, Q5_K, Q6_K), SiLU, the weighted combine; the shared expert on the dense chunk's GEMMs. Chunk attention for GQA
  8 (`llb_f8_attn*`; `m.gfast` admits it). Streamed experts: each chunk layer reads its grouping back and copies the
  chosen experts into a whole-layer staging. `LLB_MX_DUMP` writes the chunk's records too.
- **Streamed decode:** copies on all cores (`dispatch_apply`, 256 KiB pieces); the routed part committed as soon as
  it is encoded.
- Dropped (dev builds, step means): more command-buffer splits a token (at 2, 6, 14, 26 layers) 10.17 -> 10.24-10.29
  ms; routing with 4 threads an expert 12.8 -> 13.9 us; the scalar rank loop 21 us (vectorized: 12.8).

## Gates

| Gate | Class | Result | Numbers | Receipt |
|---|---|---|---|---|
| G1 dense no-regression | bit-exact | **PASS** | 4B, 9B x 3 prompts x exact/default: 12/12 ids equal to `a20c8ee2`; P2 logits hashes equal (4B `f6ee75c6`, 9B `60598769`) | `receipts/g1.txt` |
| G1h build hashes | bit-exact | **PASS** | Linux and Mac: default `5a766ce4`, `LLB_METAL_MOE1=1` `a20c8ee2`, `LLB_METAL_ANY=1` `00979448`, `LLB_METAL_BASE=1` `c13b9e28` | `receipts/build/` |
| G8 forced split / streamed | bit-exact | **PASS** | resident = `LLB_MX_SPLIT=3` = `LLB_MX_STREAM=1` (ids + logits) on the 35B (p2 x 64; a 1,000-id prompt through the chunk x 16) and the four synthetic files | `receipts/g89-35b.txt`, `receipts/g4c/g89.txt` |
| G9 scheduling-only | bit-exact | **PASS** | new build + `LLB_MX_V1=1` = base build (ids + logits), resident and streamed, 35B (p2, pp1000) and all four synthetic files | same |
| G3 35B vs llama.cpp CPU | numerics-changing | **PASS** | first token 3/3; teacher-forced 187/192 = floor (llama.cpp Metal) 187/192 at the same 5 positions (margins 0.028-0.103) | `receipts/g3/` |
| G3 controls | — | perfect PASS, garbage FAIL | llama.cpp vs itself 64/64; `no_renorm` 55/64, first token differs, margins to 3.86 | `receipts/g3/perfect-p1.json`, `mutant-p2.json` |
| G4b FFN float64 oracle | numerics-changing | **PASS** | p2 (16-token chunk + decode): 960 records, ids 960/960, rel max 2.1e-6; 1,000-id prompt: 240 sampled chunk records, ids 240/240, rel max 2.0e-6 (bounds 1e-5 / 1e-4); mutant rel 0.85 (FAIL, as it must) | `receipts/g4b/` |
| G4c synthetic vs llama.cpp Metal (>= 0.90) | numerics-changing | **PASS** | s128x8b-Q4_K_M 64/64, s60x6-Q4_K_M 58/64 = 0.906, s40x5f-Q4_K_M 59/64 = 0.922, s40x5f-Q8_0 64/64; max margin 0.24 | `receipts/g4c/` |
| G10 fast prefill quality | numerics-changing | G3b/G3c **PASS**; G3a **FAIL as preregistered — the floor fails identically** | pf2k (2,000 ids): 64/64, first token equal (floor 64/64); pf4k (4,096): 63/64, first token differs at llama.cpp margin 0.242, and llama.cpp Metal gives the same 63/64 with the same flip; `--exactpf` dispatched no chunk | `receipts/g10/` |
| G5 needle 153,352 ids | — | **PASS** | `PELICAN-7342` (ids equal to llama-tokenize's); 599 s through 300 fast chunks (was 14,908 s exact) | `receipts/g5/` |
| S1 decode >= 0.80 x llama.cpp | speed | **PASS** | 10.151 ms = 98.5 tok/s vs tg128 86.10 tok/s: **1.14x** | `receipts/speed.txt` |
| S2 prefill >= 0.50 x llama.cpp | speed | **PASS** | pp4096 1,250.7 vs 1,706.95 tok/s: **0.73x** | `receipts/speed.txt` |

G10's G3a: the preregistered rule makes the pf4k first token hard; llama.cpp's own Metal path misses it the same way,
so under Verdict Gates v2 the instrument fails the floor there. Reported as preregistered, not re-thresholded. The
synthetic files were regenerated (`tools/mk_synth.sh`, gen_moe.py, seeds 11-13); s40x5f has GQA 3 and prefills
exactly, the others take the chunk.

## Speed (medians of 3 interleaved rounds; `receipts/speed.txt`)

| 35B-A3B Q4_K_S | Before (`a20c8ee2`) | After (`5a766ce4`) | llama.cpp Metal |
|---|---|---|---|
| Decode, resident (p2, steps 9-128) | 21.798 ms (45.9 tok/s) | 10.151 ms (98.5 tok/s) | 86.10 tok/s (tg128) |
| Decode, streamed (`LLB_MX_STREAM=1`) | 44.235 ms | 27.227 ms | — |
| Prefill, 4,096 ids | 41.7 tok/s (exact; one run, `receipts/basepf.txt`) | 1,250.7 tok/s | 1,706.95 tok/s (pp4096) |
| Needle, 153,352 ids | 14,908 s | 599 s | — |

Prefill method: the `chunkf` ms + the last prompt id's exact step - the expert pool's load (its `mx pool` seconds);
raw with the load 4,685-4,883 ms. Steady chunks run 1,500-1,600 tok/s; the first chunk carries the remaining cold
costs. Profile after (v2, cut overhead included, us a layer; `receipts/prof3-mxprof.txt`): SSM trunk 126, attention
147, router 9, shared gate/up 14, routing 21, shared down 10, routed gate/up 33, routed down 22.

Correction: metal-moe's "streamed about 15% slower" compared walls with the load in them; by step time the first MoE
build streamed at 44.2 ms against 21.8 resident. Streamed decode still idles the GPU during the copy (dev timing: wait
450 us + copy 200 us a layer); the next step is a prefetch of the next layer's likely experts.

## Safety and hygiene

- The Mac was used read-only outside `~/scratch/metal-catchup`; only this lane's processes ran; none was killed. The
  35B GGUF (re-downloaded, 20.7 GB) and the synthetic files were deleted at the end (41 GB free).
- One `speed.sh` run printed no engine figures (macOS `wc -w` padding broke its awk); the parser was fixed and the
  whole 3-round run repeated (`receipts/speed-broken-awk.txt` keeps the first run's llama.cpp lines).
- The branch changes only `backends/metal`, docs and reports: the CUDA sidecar bytes, `tools/gguf2pack.c` and root
  `host/` are untouched. `python3 tools/check_no_gpl_tts.py`: PASS.

## SUMMARY

- **Commits** on `x/metal-catchup` (not pushed): `55a4e8bd9` prereg; `3ed00420c` kernels + chunk; `3e4c2ce68`
  streamed chunk + copies (emitted C `5a766ce4`); the docs/receipts commit carrying this report.
- **Before -> after vs llama.cpp Metal (35B-A3B Q4_K_S, M3 Ultra):** decode 21.8 -> 10.15 ms (45.9 -> 98.5 tok/s;
  llama.cpp 86.1: 0.53x -> **1.14x**); prefill 41.7 -> 1,251 tok/s on 4,096 ids (llama.cpp 1,707: 0.02x ->
  **0.73x**); streamed decode 44.2 -> 27.2 ms; the 153k needle 4.1 h -> 10 min, found.
- **Gates:** G1, G1h, G8, G9 bit-exact PASS; G3 PASS (187/192 = llama.cpp Metal, same positions); G4b PASS (rel <=
  2.1e-6, ids all equal; mutant fails); G4c PASS (0.906-1.000); G10 G3b/c PASS, G3a fails on one prompt exactly as
  the floor does (reported as preregistered); G5 PASS; S1 PASS (1.14x); S2 PASS (0.73x).
- **Knobs / defaults changed:** MoE GGUFs prefill through the fast chunk by default (`--exactpf` keeps exact); generic
  shapes with GQA 8 take the chunk; new `LLB_METAL_MOE1=1`, `LLB_MX_V1=1`, `LLB_MX_TRUNK=0`, `LLB_MX_PROF=1`;
  `LLB_MX_DUMP` covers the chunk.
- **Not done / limits:** streamed decode idles the GPU during expert copies (no prefetch); the chunk refuses
  F32/F16/BF16 experts by name (`--exactpf`); no 122B / 397B run; dense GQA-8 shapes through the chunk untested.
