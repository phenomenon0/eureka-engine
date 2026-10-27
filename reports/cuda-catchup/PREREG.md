# CUDA catch-up — preregistration

Frozen 2026-10-27, before any measured run of a candidate (gate or speed round). Amendments are appended below,
dated, and the file is re-hashed (`receipts/prereg*.sha256`). Rules followed: Verdict Gates v2 and the tolerance
doctrine. Base: `89dba9dde` (`feature/all-backends`, tree = published main `9f792f8c5`).

## What happened before this freeze (disclosed)

Development only, single runs, no interleaved rounds, no gate scoring:
- base `89dba9dde` built (`git archive`; engine sha256 `d98b5758…`, the same binary as reports/cuda-moe-speed's
  candidate); nsys of the dense 0.8B with and without `LLB_GRAPH=1`;
- the 8 gate files and S35x8 regenerated from the seeded commands; every sha256 matches reports/cuda-moe-speed;
- llama-bench (build 9d57ce45) once on the dense 0.8B, 2B, 4B and 9B Q4_K_M;
- development builds of the decode run and the partial-quad pair body (below), each checked against the base on one
  prompt (128 steps) of the 0.8B, 2B, 4B, 9B and S35 Q4_K_M: stdout and every `logits` line identical.

What the profile shows (base, 0.8B, nsys, one steady step of 3.48 ms):
- 0–175 us: the Bend interpreter's first device round (`bend_dev`, one 114 us launch);
- 175–515 us: the host evaluating the IO continuation (print the id, `IO.now`, the logits and step lines), with
  about a dozen unified-memory page faults (DtoH 4 KiB, about 30 us apart);
- 515–850 us: prefetch, a second device round (one 159 us launch), a sync;
- 850–1155 us: the graph's node updates and `cuGraphLaunch` (199 us of host time);
- 1155–3482 us: the pass itself (embed, 24 layers, head).
- The Q4_K / Q6_K GEMVs whose width is not whole 1024-column quads (the 0.8B's 3584-column down, the 4B's
  2560-column gate / up / ...) run the tile body at 90–140 GB/s: 24 us for the 0.8B's down (24 a token), 96 us for
  the 4B's gate (115 such launches a token, 11 ms of the 4B's 20 ms).

## Candidate scope and numeric classes

Branch `x/cuda-catchup` off `89dba9dde`.

**Exact tier (the default; class bit-exact).** Must be proven output-neutral by differential (G-BITS, G-REG):
1. **The decode run.** gen's folded step asks the host for up to n greedy steps in one pass bang (word 47 bit 28).
   The host serves them back to back (the same folded-step launches; the next step's embedding row is read from
   the device pick by `llb_embt`, the same element ops as `llb_embed*`), prints what gen would have printed for all
   but the last, and returns the count. Off for the sampler, grammar, spec, MTP, batch, images, dumps and the
   prefix cache (one step a bang, as before); `LLB_GEN=0` turns it off. Decode-run steps record CUDA graphs unless
   `LLB_GRAPH` is set.
2. **The partial-quad pair body.** `kp_pair` takes a last quad of 1–3 superblocks: lanes past the row take zero
   weights times the real x, as the tile body does, so the leaves, the quad tree and the carry are the tile body's.
   Rows the tile body ran (whole superblocks, not whole quads, depth 9–11) move to it; the fused residual store
   takes them too (the same `__fadd_rn(x1, v)` as `llb_vadd`).
3. Any later exact change (an exact prompt run, kernel work keeping the float order) is in this class and is held
   to the same gates.

**New numeric tiers (class real numerics-changing; opt-in, never the default):**
- **dp4a MoE tier** (`LLB_MOE_DP4A=1`): the routed experts' gate / up and down dot products on int8 activations
  (q8_1 per 32 values, llama.cpp's mmvq method), reusing the dense `LLB_Q4_DP4A` bodies where they apply.
- **fast MoE prefill** (`LLB_PREFILL=fast` on a MoE file): the prompt's tokens grouped by expert, each expert's rows
  a GEMM, the trunk batched as the dense fast prefill is.

The exact tier stays selectable (it is the default) and bit-identical with either tier built.

## Test files

The 8 gate files of reports/cuda-moe (S35, S60, S128F, S512; Q8_0 and Q4_K_M / Q5_K_M), regenerated, sha256
checked (`receipts/gen-sha256.txt`). S35x8 Q4_K_M (8 layers) for speed only. Dense: Qwen3.5 0.8B, 2B, 4B, 9B
Q4_K_M and clef-flash Q4_K_S. No real MoE checkpoint, no file over 10 GB, no 27B.

## Correctness gates — exact tier

- **G-BITS.** All 8 gate files, candidate default vs base `89dba9dde`, device residency, greedy 64 steps and TF,
  3 prompts (the prompts and TF ids of reports/cuda-moe): stdout ids and every stderr `logits` line byte-identical.
  PASS iff 48/48.
- **G-MODE.** All 8 files, candidate `LLB_MOE_EXPERTS=host`, `=stream`, and auto with
  `LLB_MOE_VRAM=LLB_MOE_RAM=1000000000`, vs the candidate's device residency, greedy + TF, 3 prompts. PASS iff
  24/24 (file x mode, each 6 comparisons) identical.
- **G-REG.** Dense 0.8B, 2B, 9B Q4_K_M and clef-flash, base vs candidate as reg.sh of reports/cuda-moe-speed:
  3 prompts x 64 steps, exact and `LLB_Q4_DP4A=9`, plus `LLB_PREFILL=fast` on prompt 2. PASS iff 28/28
  byte-identical. Also the 4B, exact, 3 prompts (G-REG4: 3/3) — it carries the partial-quad body's widest use.
- **G-MULTI.** A multi-prompt run (`P1 , P2 , P3`, 32 steps; three gen runs, fresh state each) on the 0.8B and S35
  Q4_K_M, base vs candidate: identical. PASS iff 2/2.
- **G-OFF.** `LLB_GEN=0` candidate vs candidate default on the 0.8B and S35 Q4_K_M, prompt P1, 64 steps:
  identical. PASS iff 2/2.
- **G-SIDE.** `gguf2pack --sidecar-only` output (manifest.json, table.bin) for the 5 dense and 8 gate files, base vs
  candidate: identical; `--target metal` untouched (no change under `backends/metal`, no change to
  `tools/gguf2pack.c`).
- **Control.** The perfect control (base scored against itself) must PASS G-BITS; the dp4a tier, a real numeric
  change, must FAIL G-BITS on every Q4_K_M file (the instrument sees it).

## Correctness gates — new numeric tiers (tolerance declared now)

Reference **X**: llama.cpp CPU on the F32-dequantized file (reports/cuda-moe amendment 1), reused as made there
(7 files). **S35 Q4_K_M has no X** (its F32 copy is 13.5 GB, over the house limit): its reference is the exact
tier's TF top-1 (proxy; on the 7 files with X the exact tier is 192/192 = X, rms ≤ 0.0027). Floor **F**: llama.cpp
CUDA teacher-forced on the same file, reused as made there.

- **G-TF (dp4a MoE tier).** TF runs (192 positions a file, the R ids). Per file: a_C = positions where the
  candidate's top-1 = the reference's top-1; a_F the same for F; b = C wrong & F right, c = C right & F wrong.
  **PASS iff a_C ≥ a_F, or a_C < a_F with the exact one-sided McNemar p (C worse) > 0.10.** FAIL otherwise. Each
  C ≠ reference position is listed with the reference's top-1 margin and whether C's pick is in the reference's
  top 3 with margin < 0.05 (near-tie). Information: rms of (C's logit of the reference's top-1 − the reference's).
- **G-PF (fast MoE prefill).** The prefix probe: for each of the 3 TF sequences and each of its 64 positions p, the
  prefix up to p is prefilled through the fast tier as one prompt of a multi-prompt run (its last token runs the
  exact step) and the logits line at p is scored as G-TF scores TF lines, paired with the same F positions. Same
  PASS rule, per file.
- **Verdict per tier:** PASS iff every file passes. Controls (calibration, must behave or no verdict is called):
  F scored as the candidate must PASS (floor-identical); the exact tier scored as the candidate must PASS (perfect
  proxy); a garbage arm (the lowest-probability top-k mutant's TF logits, S512 Q8_0) must FAIL.

## Speed (method frozen)

- **Engine decode.** `LLB_STEP_US=1`; prompt P1 (`760 511 314 938 369`; the dense models' P1 is
  `760 6511 314 9338 369`), `--steps 128`; the figure is the median `stepus` over pos ≥ 12; tok/s = 1e6 / median.
- **Engine prefill.** One fixed 512-id prompt (`scripts/p512.txt`, reports/cuda-moe-speed's), `--steps 1`, run as
  1 copy and as 4 copies (`P , P , P , P`, fresh state each, one load). Figure: tok/s = 3 x 512 / (wall_ms(4) −
  wall_ms(1)) x 1000. The load, the graph builds and the one-time setup cancel.
- **llama.cpp.** `llama-bench` build 9d57ce45, `-ngl 99`, decode `-p 0 -n 128 -r 5`, prefill `-p 512 -n 0 -r 3`;
  its mean tok/s.
- **Rounds.** 3 interleaved rounds; each round runs every arm once; the figure is the median over rounds.
- **Arms.** MoE (S35 Q4_K_M, S128F Q4_K_M; device): base, candidate exact, candidate dp4a (if built), llama.cpp
  CUDA; prefill: base, candidate exact, candidate fast (if built), llama.cpp. S35 host and stream: base and candidate
  exact. S35x8 Q4_K_M (device decode): base, candidate exact, candidate dp4a, llama.cpp. Dense 0.8B, 2B, 4B, 9B
  decode: base, candidate, llama.cpp.
- **Marginal per MoE layer** = (period(S35x8) − period(S35)) / 4, engine and llama.cpp.
- Every run: `systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=0` under both house GPU locks, one
  heavy process at a time; MoE files read from `/dev/shm` while their arms run.

## Speed targets (frozen; a miss is reported as a miss)

- **T1:** S35 Q4_K_M device decode, exact tier ≥ 0.8x llama.cpp CUDA.
- **T2:** dense 0.8B decode, exact tier ≥ 0.9x llama.cpp CUDA.
- **T3:** S35 Q4_K_M prefill, the best gated tier (fast if it passes G-PF, else exact) ≥ 0.5x llama.cpp pp512.
- **T4 (no regression):** every candidate exact arm (dense decode x4, MoE device / host / stream decode on S35, MoE
  device on S128F and S35x8, exact prefill on S35) ≥ 0.98x its base.
- Information: 2B, 4B, 9B ratios to llama.cpp; the marginal per-layer cost; dp4a vs exact; S128F.

## Amendment 1 — 2026-10-27, before any measured candidate run (gate or speed round)

Disclosed: written after development runs of the new tiers, before any gate scoring or speed round. It fixes the
prefill instrument; no threshold changes.
- **Why.** In development, `wall_ms(4 copies) − wall_ms(1 copy)` of the fast tier's 512-id prompt gave −67k and
  +43k tok/s on two back-to-back pairs: the load's run-to-run spread (tens of ms, the GPU shared with resident
  llama-servers) is larger than three fast prefills (about 50 ms). The frozen figure cannot resolve the fast tier.
- **Amended prefill figure.** The 4-copy run only (`P , P , P , P`, `--steps 1`, `LLB_STEP_US=1`). The host prints
  `pfus US` at each prompt's first folded step: the microseconds since the previous prompt's first step, that is one
  whole prompt's period (its prefill in either tier, the exact last-token step, the prompt's end and the next
  prompt's state reset). Three lines a run (prompts 1–3, timed from prompts 2–4 starting). Figure: tok/s =
  512e6 / median(the 3 `pfus` lines). The load, the graph builds and the GPU program compile fall in prompt 1's
  first step, before the first line. Medians over 3 interleaved rounds as before.
- **Arm added (information).** The fast tier on S128F Q4_K_M prefill. T3 stays on S35 Q4_K_M.
