# UNIFY1 gate registry

Registered before the runs each gate governs. The commit time is the registration time. Gates are never loosened after a result. A failed gate means the feature ships OFF or is excluded.

## R0: the regression byte set, after every carry step (`tools/unify/byteset.sh STAGE ENG` + `tools/unify/bytecmp.py STAGE`)
Flags off, pack path. Each check must be EQUAL to its reference:
- 27B, mode 9:
  - K2: 20 × 64, vs `receipts/q27c/k2/fz`.
  - K3: teacher-forced, vs `q27c/final-k/fz-tf`.
  - K4: 122 lines, vs `final-k/k4-fz`.
  - K5: p1 × 64, vs `final-ab/1-fz`.
- 27B default: K5 p1 × 64, vs `q27c/s3/k5/27b-p1-off`.
- 9B Q8: stream `5f07a4ec32f1`, with 12 and 68 logits lines.
- 9B Q4_K_M: logits 12 / 68 / 83 / 523, vs `receipts/q1/p3b`, `p3c`.

If the emitted C changes, the report says exactly what changed.

## R1: GGUF path == pack path (after s1, repeated after s3)
`LLB_Q4_DP4A=9 LLB_PREFILL=fast LLB_SPEC=2 LLB_MTP=blk64-wip.bin` on the 14 prompts of `tools/integ/s1.ids` × 64 steps.
- Run once via `LLB_GGUF=Qwen3.8-27B-Q4_K_M.gguf` with sidecar `/data/gguf2/sidecar/q27`.
- Run once from `pack-q27a`.
- PASS: the token streams are equal, and so are the `logits` lines.

## R2: PF28 KILL gate under `LLB_Q4_DP4A=9` (step 2). Exact references are regenerated under mode 9 with the s2 build.
The thresholds are PF27/PF28's, unchanged:
- G1v2 (i): 158 handoffs (`sigma.ids`). Top-2 margin Δ rms of fast-m9 vs exact-m9 ≤ 0.142 (1.2 × llama's 0.1187).
- G1v2 (ii): each handoff top-1 flip has exact margin < 0.237.
- G1v2 (iii): 20 prompts × 64, fast-m9 vs exact-m9. At most 11 prompts diverge, and each first divergence has exact margin < 0.237.
- G2: sets g1 + g1v2, 2,560 positions, llama `build/llg` teacher-forced on each tier's own tokens. Both ratios must be ≤ 1.2× the exact-m9 tier's: KL(llama‖fast-m9) / KL(llama‖exact-m9), and the same for |Δlogprob|.

FAIL → the PF28 levers ship OFF (the patch is reverted from the default build) and the 0.91x claim does not ship.

## R3: memory (nvidia-smi `card_used_peak_mib`, 1 s sampler, card otherwise empty inside the lock)
- Both-on = `LLB_Q4_DP4A=9 LLB_PREFILL=fast LLB_SPEC=2 LLB_MTP`: one synthetic prompt of N ids, + 64 steps.
- The base is integ1's `i1-eng` at N = 8192, measured first. unify1 is measured at N = 3011 and N = 8192.
- Kill line: 21,458 MiB (22.5 GB). Any combination above it is NOT a default, and the report says so in bold.

## R4: both-on identity (after s2 and after s3)
- 14 prompts (`tools/integ/s1.ids`) × 64 steps. Arms: fast (mode 9 + `LLB_PREFILL=fast`) vs both (+ `LLB_SPEC=2` + `LLB_MTP`).
- PASS: the token streams are equal, and the 924 `logits` lines are equal.

## R5: spec identity (after s3)
- k=2 and k=3 with mode 9, each vs plain greedy of mode 9 on the same prompts.
- PASS: the tokens are equal.

## R6: combined gate (after s3)
- ≥ 5 prompts × 64 steps at N = 200 / 1009 / 3011.
- (fast prefill, then spec k=2) must equal (fast prefill, spec off) of the same binary: tokens equal.
- Spec off must stay byte-equal to plain: R0 covers this.

## R7: G5 decode speed (CARRY-PLAN)
- Interleaved A/B vs `i1-eng`, median of ≥ 3 runs, flags off, mode 9, p1 × 64.
- Reported as a ratio. Below 0.97 is flagged as a regression.
