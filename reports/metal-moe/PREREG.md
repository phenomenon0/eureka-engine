# Metal MoE (any qwen35moe shape) — preregistration

Frozen 2026-10-26, before any MoE build or run of this lane. Amendments are appended below with a date and
the new SHA-256 of this file; nothing above the amendment line changes after the first run.

Rules applied: Verdict Gates v2 (class-shaped verdicts, near-tie flips documented not auto-killed, paired forms
decide, a perfect candidate must pass) and the tolerance doctrine (per-case tolerance declared before the run;
`--exact`-style bit identity is the oracle for paths that must not change).

## Classes

- **Bit-exact class** (paths this lane must not change): the dense any-shape path and the legacy 35B-A3B arm.
- **Numerics-changing class** (new): the general MoE FFN. It is a new arithmetic path, compared with llama.cpp
  (CPU, greedy, same GGUF file) under the tolerances below.

## Gates

G1 — dense no-regression (bit-exact). 4B and 9B Q4_K_M, three prompts x 64 greedy tokens, exact and default
prefill: the new build's ids equal the base any-shape build's (emitted C `00979448`) on 12/12 runs; on one
prompt the per-step `logits` lines are byte-identical. Any difference = FAIL.

G2 — legacy 35B arm unchanged (bit-exact). Run form: greedy ids + logits of the old arm, base build vs new build,
on the registered 35B-A3B pack. If the registered GGUF/pack is not available on the test Mac, the run form is
NOT RUNNABLE and is replaced by the structural form: the emitted-C diff (base any-shape vs new) shows no changed
line inside the 35B arm (its kernels `moe_*` / `f35_*` and host functions `m24_*` / `f35_*`), and the arm's
admission differs only by a test that is false for every legacy pack (MoE words = 0). Reported as structural,
not as a run.

G3 — real model through the general path vs llama.cpp (numerics-changing). A qwen35moe 35B-A3B GGUF, three
prompts x 64 greedy tokens:
- first generated token equal on 3/3 prompts (hard);
- teacher-forced top-1 agreement (llama.cpp argmax at every position of the engine's own 64-token
  continuation, compared with the engine's next token) >= 0.95 pooled over the 192 positions;
- every teacher-forced disagreement documented with llama.cpp's top-1/top-2 margin; FAIL if any disagreement
  has a llama.cpp margin > 1.0 logit (a real error, not a near-tie);
- free-running divergences are documented (position, margins), not gated.

G4 — synthetic shapes (numerics-changing). Random-weight qwen35moe GGUFs at >= 3 shapes (including 128 experts
/ 8 used and 60 / 6 with expert ff 768, and one with a fused ffn_gate_up_exps), each in Q4_K_M and Q8_0;
two prompts x 32 greedy tokens each:
- teacher-forced top-1 agreement >= 0.90 pooled per shape+quant;
- FAIL if any teacher-forced disagreement has a llama.cpp margin > 1.0 logit.
(Random weights give flat distributions; 0.90 is set below G3's 0.95 for that reason.)

G5 — long context. One MoE needle prompt > 100,000 tokens on 35B-A3B (hay140k, needle PELICAN-7342): the
generated text contains `PELICAN-7342`. Plus: `--ctx` above 2^17 is accepted for MoE (one run at
`--ctx 1048576` gives the same first 16 ids as the default capacity), and an over-large `--ctx` is refused
with a message naming the buffer limit.

G6 — refusals by name. Each unsupported MoE case exits non-zero with a message naming the field: an expert
format outside the GEMV set, a missing shared expert, top-k or expert count past the compiled limits.

## Controls (v2: perfect must pass, garbage must fail)

- Perfect: G3/G4's scorer applied to llama.cpp vs itself (same run twice) must give 1.0 / PASS.
- Garbage: the engine with a broken router (`LLB_MX_MUT=no_renorm`, weights not renormalized) must FAIL G4 on
  at least one shape, or be reported as a scorer blind spot.

## Not gated (reported)

Decode ms/token (engine and llama.cpp), load time, peak memory.
