# Metal MoE catch-up (decode + prefill) — preregistration

Frozen 2026-10-27, before any gated or timed candidate run of this lane (the only runs before this file were the
profile of the base build and development smoke runs of the new kernels; none is a receipt below). Amendments are
appended at the end with a date and the new SHA-256 of this file; nothing above the amendment line changes after the
first gated run.

Rules: Verdict Gates v2 (class-shaped verdicts; near-tie flips documented, not auto-kills; paired forms decide; a
perfect candidate must pass, garbage must fail) and the tolerance doctrine (each tolerance declared here, before the
run). Token slack follows the elpis practice (`ELPIS-MARGINS.md`: +-2 tokens where content is measured).

Machine: the test Mac (M3 Ultra, 96 GB). Model: unsloth `Qwen3.5-35B-A3B-Q4_K_S.gguf` (20,673,845,888 B). Reference:
llama.cpp `e70802a01` on the same Mac (CPU greedy = the reference; its Metal path = the floor). Base build = the
default build before this lane (emitted C `a20c8ee2`).

## Classes

- **Bit-exact** (must not move): the dense path (4B, 9B); the forced split / streamed MoE paths against resident;
  every change that only reschedules the v1 MoE kernels (command buffers, waits, copies).
- **Numerics-changing**: the new MoE decode kernels (routing, gate/up, down; association of sums and the exp change)
  and the MoE fast prefill chunk (batched expert GEMMs, the dense chunk's trunk).

## Gates

G1 — dense no-regression (bit-exact). 4B and 9B Q4_K_M, three prompts x 64 greedy tokens, exact and default prefill:
ids equal to the base build on 12/12 runs, and the P2 per-step `logits` lines byte-identical (4B, 9B). Any difference
= FAIL.

G1h — build hashes. `LLB_METAL_ANY=1` emits `00979448…`, `LLB_METAL_BASE=1` `c13b9e28…`, and a knob that stops before
this lane's patches emits the base `a20c8ee2…`; the new default emit hash is recorded. Any drift = FAIL.

G8 — forced paths (bit-exact). With the new default kernels: resident = `LLB_MX_SPLIT=3` = `LLB_MX_STREAM=1` (same ids
and the same `logits` hashes) on the 35B and on every synthetic model run.

G9 — scheduling-only changes (bit-exact). The new build with the v1 kernels selected (`LLB_MX_V1=1`) gives the base
build's ids and `logits` hashes on the 35B (resident and streamed) and on one synthetic model. This is the form that
proves every non-kernel change (command buffers, streamed copy/overlap) output-neutral.

G3 — 35B-A3B decode vs llama.cpp CPU (numerics-changing; paired against the floor). Three prompts (the metal-moe
lane's p1, p2, p3) x 64 greedy tokens; teacher-forced top-1 = llama.cpp CPU's argmax at every position of the
candidate's own continuation. The floor F = llama.cpp Metal's pooled teacher-forced agreement on the same three
prompts, measured in this lane with the same scorer (metal-moe measured 187/192).
- G3a: first generated token equal on 3/3 prompts (hard).
- G3b: pooled agreement C >= F -> PASS. If C < F: PASS only if F - C <= 2 and each of the candidate's disagreements
  beyond the floor's count is at a llama.cpp CPU margin < 0.25 logit (a documented near-tie); else FAIL.
- G3c: FAIL if any disagreement has a llama.cpp CPU margin > 1.0 logit.
- Free-running divergences are documented, not gated.

G4b — MoE FFN float64 oracle (numerics-changing, component). `LLB_MX_DUMP` records (every MoE layer and position)
recomputed from the GGUF bytes in float64 (`mx_ref.py`, metal-moe's tool). PASS: the chosen expert ids equal on every
record (or the record is a documented near-tie: the float64 gap between the k-th and (k+1)-th probability < 1e-6
relative), and relative error |out - ref| / |ref| <= 1e-5 on every record for the decode kernels. Fast prefill chunk
(if built): the same records for its tokens, bound <= 1e-4 (float32 GEMM operands, other association). The
`no_renorm` mutant must FAIL.

G4c — synthetic shapes vs llama.cpp Metal (numerics-changing). The metal-moe synthetic files (s128x8b-Q4_K_M,
s60x6-Q4_K_M, s40x5f-Q4_K_M, s40x5f-Q8_0, regenerated with `gen_moe.py` if absent), two prompts x 32 greedy tokens:
teacher-forced top-1 against llama.cpp Metal >= 0.90 pooled per file; disagreements documented with margins.

G5 — long context with the fast prefill (only if a MoE fast chunk lands and a 153k prompt then fits in under 1 hour):
the metal-moe needle prompt (153,352 tokens, `PELICAN-7342` at 45%) must output `PELICAN-7342`. Not run otherwise
(reported as not run, with the projected time).

G10 — MoE fast prefill quality (numerics-changing; only if it lands). Two prompts of 2,000-4,100 tokens (cut from the
needle haystack, chat template) + 64 greedy tokens: G3's scorer and rules (G3a-c) against llama.cpp CPU on the same
prompts, floor = llama.cpp Metal on the same prompts. `--exactpf` keeps the per-token prefill (checked: no chunk is dispatched under it).

## Speed targets (gated; interleaved rounds, medians, receipts)

Method: >= 3 interleaved rounds (engine, then llama.cpp, then the next round); the median of the rounds is the figure;
nothing else runs on the GPU from this lane meanwhile (the Mac's resident servers are idle and not touched).
- Decode: engine = mean of the `step` lines of steps 9..128 of a 128-token greedy run from p2 (resident experts);
  llama.cpp = `llama-bench -m MODEL -p 0 -n 128 -r 3` (tg128, Metal, all layers offloaded). S1 PASS: engine tok/s >=
  0.80 x llama.cpp tok/s.
- Prefill: engine = prompt tokens / the time from the first prompt step to the first generated token on a 4,096-token
  prompt (the engine's stderr timing; model load excluded); llama.cpp = `llama-bench -p 4096 -n 0 -r 3` (pp4096). S2
  PASS: engine >= 0.50 x llama.cpp.
- Streamed mode (`LLB_MX_STREAM=1`) decode: reported against resident, not gated.

## Controls

- Perfect: the G3 scorer on llama.cpp CPU against itself = 64/64 (PASS).
- Garbage: `LLB_MX_MUT=no_renorm` with the new kernels must FAIL G4b (and is reported on G3).

## Not gated (reported)

Load time, peak memory, the per-stage GPU profile (method: command-buffer cuts, `LLB_MX_PROF`, dev builds only), the
dense 9B decode before / after.
