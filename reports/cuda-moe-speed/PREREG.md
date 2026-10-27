# CUDA MoE speed — preregistration

Frozen 2026-10-27, before any measured run of a candidate. Amendments are appended below, dated, and the
file is re-hashed (`receipts/prereg*.sha256`). Rules followed: Verdict Gates v2 and the tolerance doctrine.

## What happened before this freeze (disclosed)

Development and profiling only, no candidate speed measurement:
- the base build (`4c7c24c10`, `git archive`) was profiled with nsys on S35 Q4_K_M (128 steps) and on the dense
  0.8B (64 steps) (`receipts/prof/`);
- one base run each with and without `LLB_GRAPH=1`, using the new `LLB_STEP_US` timer, to check the timer;
- the S35 Q8_0 and Q4_K_M files were regenerated; their sha256 match reports/cuda-moe.

What the profile shows (base, S35 Q4_K_M, nsys, steady-state step):
- The token period is about 1.7 ms. About 0.75–0.9 ms of it is per-token work outside the layers: the Bend
  program (`bend_dev`, 16 launches, about 340 us), unified-memory page traffic and host gaps. The dense 0.8B has
  the same amount (`bend_dev` 337 us, 918 us of gaps).
- The MoE FFN of one layer takes about 150 us of GPU time:
  - `llb_moe_gu` 61 us;
  - `llb_moe_dn` 35 us;
  - `llb_moe_topk` 19 us;
  - the router and shared-gate F32 GEMVs 16 us;
  - the shared expert (4 launches) 19 us.
- Expectation, written down before any candidate run: the fixed per-token part alone limits T1 below to about
  0.8x llama.cpp, even with infinitely fast layers. T1 is expected to miss; T2 is the figure that carries to a
  40-layer model.

## Candidate

Branch `x/cuda-moe-speed`, based on `4c7c24c10` (tree = published main `c8f55083c`). Scope:
- the routed-expert kernels (`tools/payload-moeany.patch`);
- `llb_moe_run` / `llb_moe_load` (`host/llb_host.c`), including the host and stream residencies;
- the env-only `LLB_STEP_US` timer;
- CUDA graphs for MoE files.

**Numeric class: bit-exact.** Every dot product keeps its float operation order:
- lane l sums groups l, l + 32, ... in order, then the xor butterfly;
- the router softmax trees and top-k selection are unchanged;
- the down sum runs in rank order.

So the candidate must be proven output-neutral by differential (G-BITS). No new tolerance is declared for the
default path. A faster numeric tier (int8 activations, dp4a) is out of scope for this prereg. If one is built, it
gets a dated amendment with its own tolerance before it is measured.

## Test files

These are the 8 gate files of reports/cuda-moe, regenerated with the same seeded commands, with sha256 checked
against that report: S35, S60, S128F and S512, in Q8_0 and in Q4_K_M or Q5_K_M.

For speed only there is also **S35x8**: the S35 shape with 8 layers (`L=8`, the same seed and generator),
Q4_K_M. Its Q8_0 file is 6.9 GB, under the 10 GB house limit.

## Correctness gates

- **G-BITS (primary).** On all 8 gate files, the candidate (device residency) gives stdout ids and every stderr
  `logits` line byte-identical to the base build (`4c7c24c10`, device residency). That covers greedy 64 steps and
  teacher-forced (TF) runs, 3 prompts each (the prompts and TF ids of reports/cuda-moe): 48 comparisons. PASS iff
  48/48.
- **G-X (re-scored).** The candidate's TF logits are scored by `gx.py` (reports/cuda-moe amendment 1) against the
  X and R arms of reports/cuda-moe, which are reused as they are. PASS per file iff top-1 = X at 192/192 and
  rms(C − X) ≤ rms(R − X).
  - Files with X: S512 Q8_0, S512 Q5_K_M, S60 Q8_0, S60 Q4_K_M, S35 Q8_0 and S128F Q8_0.
  - S128F Q4_K_M: X is made now. Its F32 copy is about 6 GB, on `/mnt/storage`, deleted after.
  - S35 Q4_K_M: its F32 copy is 13.5 GB, over the 10 GB house limit, so X is **not run**. G-BITS identity to the
    base and the base's frozen-gate PASS stand for it.
- **G-MODE.** On all 8 gate files, `LLB_MOE_EXPERTS=host`, `=stream`, and auto with
  `LLB_MOE_VRAM=LLB_MOE_RAM=1000000000` give stdout and every `logits` line byte-identical to the candidate's
  device residency, greedy and TF, 3 prompts. PASS iff 24/24. Tolerance 0: the kernels and bytes are the same in
  every residency.
- **G-REG.** The dense 0.8B, 2B and 9B Q4_K_M and clef-flash Q4_K_S, base vs candidate, run as reg.sh of
  reports/cuda-moe does: 3 prompts x 64 steps, exact and `LLB_Q4_DP4A=9`, plus `LLB_PREFILL=fast` on prompt 2.
  PASS iff 28/28 byte-identical.
- **G-SIDE.** `gguf2pack --sidecar-only` output for the 4 dense files and the 8 gate files is identical between
  base and candidate.
- **Controls** (the instrument must discriminate, or no verdict is called):
  - Perfect: the base build scored as the candidate must PASS G-BITS.
  - Subtle mutant: a scratch-only build whose routed gate/up lane sum adds the groups in reverse order (a pure
    reassociation, same weights) must FAIL G-BITS on S35 Q4_K_M. That shows G-BITS sees an order-level change.
  - Garbage mutant: top-k takes the lowest-probability experts. It must FAIL G-BITS and G-X on S512 Q8_0.
- **Verdict.** Correctness PASS iff G-BITS, G-X on every file where X exists, G-MODE, G-REG and G-SIDE all pass
  and the controls behave.

## Speed (method frozen)

- **Engine figure.** `LLB_STEP_US=1` prints the microseconds between consecutive folded steps: the whole token
  period, including the Bend loop and the head.
  - Decode: prompt P1 (`760 511 314 938 369`), `--steps 128`. The figure is the median of the `stepus` lines with
    pos ≥ 12. tok/s = 1e6 / median.
  - Prefill: a fixed 512-id prompt (`scripts/p512.txt`), `--steps 1`. tok/s = 511e6 / (the sum of the `stepus`
    lines with 1 ≤ pos ≤ 511).
- **llama.cpp figure.** `llama-bench` (build 9d57ce45) with `-ngl 99 -p 0 -n 128 -r 5` (decode) and
  `-p 512 -n 0 -r 3` (prefill). The figure is its mean tok/s.
  - Host-RAM comparison: the same commands with `-ot exps=CPU` (experts on the CPU, default threads).
- **Arms, interleaved.** There are 3 rounds. Each round runs, in this order: base device, candidate device,
  candidate host, candidate stream, base host, base stream, llama.cpp CUDA, llama.cpp `-ot exps=CPU`. The reported
  figure is the median over the 3 rounds.
- **Files.** S35 Q4_K_M (4 layers), S128F Q4_K_M (4 layers), and S35x8 Q4_K_M (8 layers, device and llama.cpp
  CUDA only).
- **Marginal per-layer cost** = (period(S35x8) − period(S35)) / 4, for the engine (device) and for llama.cpp
  CUDA (1e6 / tok/s). It removes the per-token fixed costs (the Bend loop, embed, head) and is what scales with
  depth.
- Every engine and llama.cpp run is wrapped in `systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0`
  under the house GPU locks. One process at a time.

## Speed targets (frozen; a miss is reported as a miss)

- **T1:** device tok/s on S35 Q4_K_M ≥ 0.8x llama.cpp CUDA.
- **T2:** device marginal per-layer cost ≤ 1.25x llama.cpp CUDA's, that is ≥ 0.8x llama.cpp speed per layer.
- **T3:** on S35 Q4_K_M, candidate host ≥ 1.2x base host, and candidate stream ≥ 1.5x base stream.
- **T4 (no regression):** on both 4-layer files, the candidate device, host and stream are each ≥ the base's.
- Prefill is information only: engine vs llama.cpp pp512, base vs candidate.

## Amendment 1 — 2026-10-27, after speed round 1, before rounds 2 and 3 were scored

Disclosed as post-hoc. It fixes a defect in the measuring instrument; no threshold changes. In round 1, the
`stepus` line at pos 1 holds the one-time expert load: `llb_moe_load` runs inside the first MoE step.
- On S35 Q4_K_M that line is 0.42–0.43 s, for base and candidate alike.
- On S35x8, read from the second disk, it is 80 s.

The candidate's pos 2 also holds its one-time CUDA-graph instantiation (5.3 ms against a steady 1.4 ms). The
frozen prefill figure, the sum over 1 ≤ pos ≤ 511, therefore measured load time, not prefill.

Amended prefill figure: tok/s = 509e6 / (the sum of the `stepus` lines with 3 ≤ pos ≤ 511). The decode figure
(pos ≥ 12) and every other definition are unchanged. Prefill was information only, so no verdict depends on it.
