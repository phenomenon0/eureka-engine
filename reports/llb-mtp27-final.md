# MTP27 — speculative decoding on the Bend→CUDA engine, Qwen3.8-27B Q4_K_M (RTX 3090)

Branch `mtp27` (worktree `llama-bend-mtp27`). HyperQwen study copy @ e371b42c, ideas cited file:line in
`receipts/mtp27/hyperqwen-cites.md` (no code copied). Base = c5 mode 9 (`LLB_Q4_DP4A=9`, q27c final: 25.690 ms/step).

## Stage 0 — measurements before building

### llama.cpp plain vs MTP (9d57ce45, 8 HyperQwen-kind prompts × 256, `--no-mmap`, one slot)

`tools/mtp27/summ.py receipts/mtp27/s0/off-hq-{greedy,default}.json`

| arm (greedy) | tok/s pooled | median | × | tok/verify | acc by pos % |
|---|---:|---:|---:|---:|---|
| plain | 39.35 | 39.22 | 1.00 | 1.00 | |
| sidecar k=1 | 60.38 | 61.70 | 1.53 | 1.84 | 85 |
| sidecar k=2 | 67.62 | 68.31 | 1.72 | 2.46 | 84 65 |
| sidecar k=3 | 65.32 | 69.40 | 1.66 | 2.84 | 82 61 45 |
| sidecar k=4 | 64.63 | 69.49 | 1.64 | 3.19 | 84 61 44 35 |
| sidecar k=5 | 58.67 | 63.91 | 1.49 | 3.34 | 82 59 42 32 23 |
| sidecar k=6 | 54.05 | 59.57 | 1.37 | 3.47 | 81 57 41 31 24 19 |
| graft k=2 | 68.34 | 69.41 | 1.74 | 2.49 | 85 66 |

| arm (default sampling: T 1, top-k 20, top-p 0.95) | tok/s pooled | median | × | tok/verify | acc by pos % |
|---|---:|---:|---:|---:|---|
| plain | 39.22 | 39.04 | 1.00 | 1.00 | |
| graft k=2 | 64.46 | 66.84 | 1.64 | 2.35 | 79 58 |
| graft k=3 | 62.91 | 66.13 | 1.60 | 2.75 | 78 57 43 |

Hauhau's spec2 (plain 41.10, k=2 1.88×): NOT reproduced. On this card/build: plain 39.35 (−4%), k=2 1.72–1.74×.
Greedy identity vs plain 2/8 prompts (llama.cpp's batched verify is not bit-equal to its own T=1 path; the
diverging indices are the same across arms, i.e. the verify's numerics, not the drafts). KILL rule (<1.4×
everywhere): not hit.

### Graft (plain Q4_K_M trunk + official blk.64, `tools/mtp27/graft.py`)

Acceptance at position 0: 85% greedy, 79% default (gate ≥ 40%: PASS). Graft ≈ sidecar (k=2 1.74 vs 1.72×).

### m(T): T-row dp4a GEMV (`receipts/mtp27/s0/mT-v1.txt`, bit check row tt of T == q27c leaf: PASS)

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GEMV ms/step (best variant) | 21.79 | 25.15 | 29.21 | 32.80 | 40.54 | 42.79 | 49.01 | 58.04 | 61.25 |
| m(T) | 1.000 | 1.154 | 1.340 | 1.506 | 1.861 | 1.964 | 2.249 | 2.664 | 2.811 |

### Prediction (written before Stage 1; `tools/mtp27/predict.py` → `receipts/mtp27/s0/predict.txt`)

step(k) = GEMV(T) + other(1) + (OTHER_ROW 0.50 + SNAP 0.19)·(T−1) + k·DRAFT 0.65, T = k+1, E(k) from llama.cpp.

| samp | k | E | step ms | tok/s | × |
|---|---:|---:|---:|---:|---:|
| greedy | 1 | 1.84 | 30.39 | 60.5 | 1.56 |
| greedy | 2 | 2.49 | 35.79 | 69.6 | 1.79 |
| greedy | 3 | 2.84 | 40.72 | 69.7 | 1.79 |
| greedy | 4 | 3.19 | 49.80 | 64.1 | 1.65 |
| default | 2 | 2.35 | 35.79 | 65.7 | 1.69 |
| default | 3 | 2.75 | 40.72 | 67.5 | 1.73 |

KILL rule (<1.25×): not hit.

## Stage 1 — T-row verify (`LLB_SPEC=k`, default off)

Commits ffe35e57 (patch + host), 826d7dda (K2 receipts).

- `tools/payload-spec.patch` (new LAST patch; no earlier patch edited): T-row twins (T = 1..8) of the
  mode-9 dp4a leaves (Q4_K gate/up and down+res, Q8_0 cat with the conv-ring member, Q6_K), `llb_spec_q4s`,
  out-of-place gated delta `llb_spec_gdl`; the pass op's embedding base in `o->r`.
- host: one cycle = T rows in one pass; gated-delta state ring of k+2 states per linear layer (a rejected
  row costs no copy back), conv as a 16-slot ring by position, KV slots by position (rewritten). Small
  kernels loop per row.

Gates:

| gate | result | receipt |
|---|---|---|
| 27B spec OFF vs q27c fz (mode 9) K2 20×64 | ids + logits identical (only wall_ms differs) | `receipts/q27c/k2/spec-off.*` |
| 27B `LLB_SPEC=0` (T = 1 path) | 20/20 logits identical | `receipts/mtp27/s1/smoke0.*` |
| 27B `LLB_SPEC=3` (lookup drafter) K2 20×64 | 20/20 IDENTICAL, all ids in order; 1.51 tok/cycle on smoke, 1.17 on K2 | `receipts/q27c/k2/spec-k3.*` |

VRAM: the S ring is (k+2) states × 48 linear layers × 48 heads × 128 × 128 × 4 B = 151 MB per state set
(fp32), i.e. 0.75 GB at k = 3; fits beside the 18.25 GB pack (no fp16 state variant was built: no bit change).
The T-row attention keeps the existing per-row attn kernel (one query row per launch, no split-KV kernel
written): at ≤ 300 positions it is not the cost (nsys below).

## Stage 2 — MTP drafter (`LLB_MTP=file`, with `LLB_SPEC=k`)

Commits fa7a4105 (code), b8d31462 (gates).

- `tools/mtp27/mtp_pack.py` ("packer row 64"): the official sidecar's `blk.64` (15 tensors, raw GGUF Q4_K /
  Q6_K / F32 blocks, the pack's own raw layout) + the draft-vocab slice of the trunk head (40,960 Q6_K rows of
  `output.weight`, HyperQwen `prepare/build_draft_vocab.py:91-103`) + the ids → one 438 MB file.
- host: the NextN block as a 65th layer on its own KV (llama.cpp `src/models/qwen35.cpp` graph_mtp semantics:
  concat[rms(e)·enorm, rms(h)·hnorm] → eh_proj → attention → FFN → shared_head_norm; h = the trunk's
  output-normed hidden row). Catch-up rows over the positions the last verify served (h from the verify's own
  rows), then chained single-row drafts on the drafter's own h. The draft head = top-1 over the 40,960-row
  slice, mapped back to vocab ids. Same launches as the trunk's rows (`spec_attn` / `spec_ffn` helpers shared
  with the verify). `LLB_SPEC_ADAPT=1`: k_eff ±1 per cycle in 1..k (all accepted → +1, else −1).
- Draft vocab: OUR count over 25.6k tokens of the 27B's own default-sampling outputs (32 corpus prompts,
  `tools/mtp27/vocab.py`; only 2,812 distinct ids, padded to 40,960 by the trunk's id order). Coverage of a
  40,960 slice: 98.67% corpus held-out, 94.09% / 93.90% on the HyperQwen-prompt plain greedy / default
  outputs (HyperQwen's own 5.4M-token list on the same outputs: 97.31 / 97.75%). The ≥ 200k-token corpus was
  NOT reached (GPU time contended by the sibling lanes): the slice is under-counted, ~3.5 points of
  coverage below HyperQwen's list (their ladder: 92 → 97.5% coverage was worth ~10%).

Gates:

| gate | result | receipt |
|---|---|---|
| 27B `LLB_SPEC=3 LLB_MTP` K2 20×64 vs q27c fz | 20/20 IDENTICAL (1532 logits lines, ids in order) | `receipts/q27c/k2/spec-mtp3.*` |
| smoke p1 × 64 vs Stage 1 smoke | out + 68 logits lines identical | `receipts/mtp27/s2/smoke-mtp3.*` |
| spec OFF (27B) | = Stage 1 (spec OFF path unchanged by the drafter: no `LLB_MTP` → no load) | `receipts/q27c/k2/spec-off.*` |

Acceptance (MTP k = 3, by draft position, incl. prompt positions): smoke p1 95/59/55%, 3.09 tok/cycle;
K2 62/38/25%, 2.256 tok/cycle (K2 mixes CJK / code / random-token prompts). Lookup drafter on K2: 1.17.
Draft cost: a T = 4 MTP cycle 46–47 ms vs a lookup T = 4 cycle 42–43 ms → ~1.2–1.5 ms per draft incl. the
host sync per draft (predicted 0.65).

## Stage 3 — interleaved A/B (5 rounds)

Commits 1f668b13 (harness), 4e07a363 / 2ac23141 / 2006b2c2 / 04de1a0d / 4c3dde48 (rounds 1–5), 89675025 (nsys),
86ae3f4e (spec-OFF gates). Receipts `receipts/mtp27/s3/` (`summary.txt` = every round's numbers, `chain.log` =
order, clocks, load).

Protocol: each round runs every arm once, one engine process at a time, each under `/tmp/llb-gpu-bench.lock`
held for the arm (`tools/mtp27/s3ab.sh`), with an idle GPU checked inside the lock. Engine arms: mode 9,
`tools/mtp27/s3.ids` = 3 anchors (K2 p1–p3) + the 8 HyperQwen bench prompts (`bench/prompts_real.jsonl`) in the
chat turn with thinking off, greedy × 64, one load (704 tokens). ms/token comes from the engine's step lines
(`tools/mtp27/s3ms.py`; spec windows start at the first generated cycle). llama.cpp arms: 9d57ce45 server on the
graft GGUF, `-mmp 0`, the 8 hq prompts × 64 greedy (`tools/mtp27/probe.py`).

**Deviations:**
- Within a round, the llama arms ran back to back, and so did the engine arms (warm page cache on the 17 MB/s
  NTFS disk), so the order is not one fixed interleave.
- 10 runs were quarantined in `receipts/contaminated/mtp27-s3-*` and re-run: 9 load timeouts (slow disk) and
  one c5 that slowed mid-run (3-c5: its last 3 prompts ran at 30–31 ms).

| arm | tok/s r1..r5 | mean | × c5 | tok/cycle | accepted at draft pos j % | cycle ms (= tok/cycle × ms/tok) |
|---|---|---:|---:|---:|---|---:|
| c5 plain (q27c `c5-eng`) | 38.37 38.30 38.40 38.35 38.38 | **38.36** (26.07 ms) | 1.00 | 1 | | 26.07 |
| MTP k=2 | 57.40 57.09 57.17 57.66 57.80 | **57.42** | **1.497** | 2.346 | 80 55 | 40.9 |
| MTP k=3 | 54.42 54.85 55.02 54.88 54.59 | 54.75 | 1.427 | 2.717 | 80 56 36 | 49.6 |
| MTP k=4 | 49.70 49.76 50.06 50.09 50.01 | 49.92 | 1.301 | 2.923 | 77 52 37 26 | 58.4 |
| MTP k=3 adaptive (1 round) | 56.05 | — | 1.46 | 2.394 | 79 45 16 | 42.7 |
| llama plain (pooled / median) | 39.74 39.74 39.75 39.64 39.78 | 39.73 / 39.16 | | 1 | | |
| llama MTP k=2 (pooled / median) | 65.79 66.14 66.18 65.44 66.09 | 65.93 / 68.20 | 1.66 (own plain) | 2.34 | 83 61 | |

Best arm: k=2, **1.497× c5** (KILL < 1.15×: not hit). That is 87% of llama.cpp's MTP throughput on the same
card (57.4 vs 65.9), from a plain base at 97% of llama's (38.36 vs 39.73).

**Correctness:**
- Every spec arm in every round is token-identical to c5 on all 11 prompts.
- Round 1 m2 / m3 / m4 / m3a logits are byte-identical to c5 (1532 lines).
- The c5 rounds are identical to each other.

The spec output IS the plain engine's output (lossless, not just "within tolerance"). llama.cpp's MTP matches
its own plain output on only 4/8 prompts (batched-verify numerics).

### nsys of one k=2 cycle (`receipts/mtp27/s3/nsys-spec-m2.txt`; `tools/mtp27/s3nsys.sh` + `tools/mtp27/cyc.py`)

p1 × 40 steps, 14 hot cycles. Plain c5 for comparison: `receipts/q27c/s3/nsys-c5-fz.txt`.

| class | c5 plain step ms | k=2 cycle ms | predicted (Stage 0) |
|---|---:|---:|---:|
| GEMV (gu/cat/res/q8/q6 leaves) | 21.73 | 26.70 (1.229×; bench m(3) 1.340) | 29.21 |
| small kernels (rmsnorm, silu, gdl, midconv, attn, qkr, …) | 3.08 | 8.05 (2.61×: looped per row, ~2.5 ms/extra row) | 3.90 + 0.50·2 |
| gaps / snapshots | 1.16 | 1.56 | 0.19·2 |
| drafts (2 MTP calls) | — | 3.87 (busy 2.62; `bend_dev` 40 launches 0.93, head slice `q61` 0.72) | 0.65·2 |
| host accept / reject | — | 0.03 | — |
| **span** | **25.98** | **40.22** | **35.79** |

The A/B cycle at k=2 (40.9 ms over all 11 prompts) agrees with the nsys span (40.2 ms on p1).

### Cost model: predicted vs measured

| k | E tok/cycle pred / meas | cycle ms pred / meas | × pred / meas |
|---|---|---|---|
| 2 | 2.49 / 2.346 | 35.79 / 40.9 | 1.79 / **1.497** |
| 3 | 2.84 / 2.717 | 40.72 / 49.6 | 1.79 / 1.427 |
| 4 | 3.19 / 2.923 | 49.80 / 58.4 | 1.65 / 1.301 |

Two misses:
- **Cycle cost (most of the miss).** The GEMV came in better than predicted (−2.5 ms at k=2). The per-row small
  kernels (+3.4 ms) and the draft calls (+2.6 ms: ~1.9 ms per draft, not 0.65) more than ate that saving.
- **Acceptance (5–8% low).** The prediction used llama.cpp's graft E from 256-token outputs; ours is measured
  over 64-token outputs, and the draft vocab is under-counted (Stage 2).

### Against HyperQwen's ladder (`docs/optimizations.md:381-389`)

| | base tok/s | MTP tok/s | × | tok/step | spec step ms | step / plain step |
|---|---:|---:|---:|---:|---:|---:|
| HyperQwen fast variant, k=4 | 46 | ~114 / 118–124 | 2.48–2.70 | 2.8 / 2.9–3.0 | 24.2–24.6 | 1.12–1.13 |
| ours k=4 | 38.36 | 49.92 | 1.30 | 2.92 | 58.4 | 2.24 |
| ours k=2 (best) | 38.36 | 57.42 | 1.50 | 2.35 | 40.9 | 1.57 |

**Fraction of their gain reached:** (1.497 − 1) / (2.48..2.70 − 1) = **29–34%**.

**The term that differs is the step cost, not the acceptance.**
- **Acceptance matches.** At k=4 we get 2.92 tok/step against their 2.8–3.0.
- **Our step costs about twice as much relative to plain.** Their T=5 verify step costs 1.12× a plain step
  (int4 Marlin on tensor cores: the T rows ride the same weight read). Ours costs 2.24×, for three reasons:
  1. The dp4a GEMV is not flat in T: m(5) = 1.861.
  2. Every small kernel loops once per row.
  3. Each draft costs ~1.9 ms, against their 0.5–1 ms (`docs/optimizations.md:78`).

**Upgrade paths, in order of yield:**
1. Batch the small kernels over T rows: one launch per T instead of per row, ≈ −5 ms per cycle at T=3.
2. Fold the draft's per-call `bend_dev` launches and host sync.
3. A flatter T-row GEMV (tensor-core int8 MMA at T ≥ 4).
4. A ≥ 200k-token draft-vocab corpus.

With paths 1 and 2, k=2 would sit near 35 ms per cycle, i.e. ~1.75×.

## Spec-OFF exactness gates on the final build (`build/spec-eng` f8e9c73e339d, commit 86ae3f4e)

| gate | result | receipt |
|---|---|---|
| 9B Q8 s8 / s64 | 8/8 tokens + 12/12 logits, 64/64 + 68/68 IDENTICAL; **s64 stream 5f07a4ec32f1** | `receipts/mtp27/s3/k5/spec-off.gate.log` |
| 9B Q4_K_M vs Q1 engine | logits **12 / 68 / 83 / 523 BYTE-IDENTICAL**; twin + p1 / p3 anchors identical; 523/523 finite | `receipts/mtp27/s3/k5-q4/gate-{a,b}.txt` |
| 27B spec OFF vs q27c fz | ids + logits identical (Stage 1) | `receipts/q27c/k2/spec-off.*` |

The p2 anchor's "32/64 DIFF at step 32" is the pre-existing Q1 state: the same line appears in 9 older receipts,
and the logits are byte-identical to the Q1 engine. The gates ran one run per tool call under a blocking lock,
because the GPU was contended. They use the same commands as `tools/r3090_gate.sh` / `tools/q4b_gate.sh`.

## The 46 tok/s question

| | ms/step | tok/s | weight bytes/step | effective GB/s |
|---|---:|---:|---:|---:|
| HyperQwen no speculation (`docs/optimizations.md:381`, vLLM W4A16 AutoRound, int8 head/embed, fp16 state) | 21.7 | 46 | ~19.5 GB (BELIEVED, from their docs) | ~900 |
| ours, c5 mode 9 (q27c final; this lane's s3 set: 26.07) | 25.69 | 38.9 | 18.25 GB | ~710 |
| llama.cpp 9d57ce45 (q27c `llama-bench` tg; s3 pooled 39.73 tok/s) | 26.04 | 38.4 | ≈ ours (same Q4_K_M tensors; 18.97 GB file incl. the embedding table, read one row per step) | ~700 |

Their base is ~18% faster per step at a similar byte count. The gap is bandwidth efficiency (Marlin int4 at ~96%
of the 3090's 936 GB/s vs our dp4a at ~76%), not fewer bytes. Their 114–127 tok/s headline is ~2.5–2.7× of
speculation on top of that base.

## Lookup drafting (HyperQwen's 259–381 tok/s mode) — estimate, not built beyond the Stage 1 drafter

HyperQwen's figures:
- Their mode is a token-match proposer over the context plus a 16-token verify block
  (`patches/dflash2-lookup-drafting.patch:283-314`, `docs/optimizations.md:309-316`).
- It pays only where the model reproduces its context: 14.97 tok/step, 381 tok/s on "reproduce verbatim".
- On free-form text it gives 2.13 tok/step, 71 tok/s (below their MTP).

Ours:
- Our Stage 1 lookup drafter on K2 (no context-copy text): 1.206 tok/cycle, 27.58 ms/tok. That is **slower
  than plain** (25.80); MTP k=3 on the same set is 20.17 ms/tok.
- Estimated ceiling on pure copy text: a T=8 cycle ≈ GEMV(8) 58.0 + 4.2 + 7 × 2.5 small ops ≈ 80 ms for ≤ 8
  tokens, i.e. ≤ ~100 tok/s. A 16-row block would need m(T) beyond the measured T ≤ 9 and is not estimated.

To approach their 259–381, the step must be flat in T (the batched small kernels + a tensor-core T-row GEMV
above). Until then, lookup is a niche win at best on this engine.

## Commits and gate counts

| stage | commits | gates |
|---|---|---|
| 0 | 6c34ddcb, cfe59e1e, 88e6df88 | m(T) bit check row-of-T == q27c leaf, T ≤ 9: PASS; graft acc pos0 85% (≥ 40%) |
| 1 | ffe35e57, 826d7dda | 27B spec-off K2 20×64 identical; LLB_SPEC=0 20/20; LLB_SPEC=3 lookup 20/20 |
| 2 | fa7a4105, b8d31462 | K2 LLB_SPEC=3 + MTP 20/20 (1532 logits lines); smoke 68/68 |
| 3 | 1f668b13, 4e07a363, 2ac23141, 2006b2c2, 04de1a0d, 4c3dde48, 89675025, 86ae3f4e | 5 rounds: all spec arms token-identical to c5 (11 prompts each), r1 logits 1532 identical; 9B Q8 5f07a4ec32f1; 9B Q4_K_M 12/68/83/523 |

No earlier patch was edited (no CENSUS MISS); new code lives only in `tools/payload-spec.patch`,
`engine_qwen35_gpu.bend` and the host files. Fable consults used: 0 of 3.

## NOT verified / known gaps

- **Host neutrality check.** `tools/step2a_neutral_check.sh` fails at HEAD, as it did at the lane's base
  (20 errors: the stub lacks the dp4b names; pre-existing, flagged, not fixed). host/llb_host.c CUDA-type-free
  was checked by grep only: 0 CUDA names or includes.
- **Hauhau's spec2 (1.88×) was NOT reproduced.** This card gives 1.72–1.74×.
- **Draft vocab under-counted.** The corpus has 25.6k tokens, not ≥ 200k; coverage is 94% vs HyperQwen's 97.3%.
- **Features not built.**
  - No fp16 recurrent state.
  - No split-KV verify attention kernel.
  - The lookup drafter has no context-length cap.
  - No batched small kernels.
- **Measurement limits.**
  - Timing is integer ms per step line (truncation < 1 ms per cycle line, on both arms).
  - The adaptive k=3 arm ran one round only.
  - Default-sampling spec was not A/B'd on the engine (greedy only); the engine has no sampler-side spec.
  - The ~19.5 GB / ~900 GB/s for HyperQwen is from their docs, not measured here.
  - The nsys trace covers one prompt (p1) at short context.
