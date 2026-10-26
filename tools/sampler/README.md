# Sampler

`tools/sampler/` holds the gates for the sampler module (W1-SAMPLER). The
module adds temperature, top-k, top-p, min-p, penalties, logit bias, seeds,
logprobs, and stop strings to the engine. It keeps speculative decoding
lossless under sampling.

## Files

| File | Description |
|---|---|
| `host/llb_sampler.h` | The API. |
| `host/llb_sampler_impl.h` | The implementation. |
| `host/llb_sampler.c` | Standalone object for users outside the engine, such as the server and tests. |
| [`compat/run_gates.sh`](compat/run_gates.sh) | Re-runs the sampler2 gates on this module through `q35compat.c`. |
| [`compat/mutants.sh`](compat/mutants.sh) | Checks that every mutant is caught. |
| [`run_spec_stat.sh`](run_spec_stat.sh) | Gate S: speculative sampling statistics. |

The engine inlines both headers into the emitted C. The Bend `comp.ts`
inlines every `#include "llb_*.h"`. The module is plain C11 and has no
dependencies.

## Features

| Feature | API | Behavior |
|---|---|---|
| Temperature, top-k, top-p, min-p | `LlbSmpParams`, `llb_smp_sample` | llama.cpp default chain order: bias → penalties → top_k → top_p → min_p → temp → dist. The stage code is from sampler2, gated stage by stage against libllama (177 cases, `compat/`). |
| Repetition, presence, and frequency penalties | Same | llama.cpp windowed form (`repeat_last_n`, prompt tokens included). |
| Logit bias | `bias=ID:VAL;ID:-inf` | Added first. `-inf` bans a token. |
| Seed | `seed=N` | Counter-based RNG: draw = splitmix64(seed, pos, k). The same seed gives the same stream however the prefill ran. It is not mt19937, so the same seed does not reproduce the tokens of llama.cpp. |
| Logprobs (top-n) | `logprobs=N` (≤20), `llb_smp_logprobs` | log_softmax of the raw logits: the model distribution, as in OpenAI. The engine prints `logprobs <pos> <tok> <lp_tok> id:lp …` on stderr. |
| Stop strings | `llb_stop_*` | Safe for streaming. Holds back any suffix that is a proper prefix of a stop string, and any incomplete UTF-8 tail. Trims at the leftmost match. Runs on detokenized bytes, on the server or front-end side. |
| Max tokens | `max_tokens=N` field | A policy field for the caller. The engine's own cap is `--steps`. |
| Per-request selection | `llb_sampler_configure(spec)` (engine C ABI); `--sample SPEC` request word (sys1 server); `LLB_SAMPLE=SPEC` environment default | See [Options](#options). |

## Set sampling parameters

1. Write a sampling spec as `key=val,...`. For example:

   ```text
   temp=0.7,top_p=0.9,seed=42
   ```

2. Pass the spec in one of three ways:

   - As the environment default:

     ```sh
     LLB_SAMPLE=SPEC ENGINE_COMMAND
     ```

   - As the `--sample SPEC` request word on the sys1 server.
   - From C, with `llb_sampler_configure(spec)`.

   Replace `SPEC` with the spec and `ENGINE_COMMAND` with the engine command
   line.

## Options

The spec keys are:

| Key | Description |
|---|---|
| `temp` | Temperature. |
| `top_p` | Top-p. |
| `top_k` | Top-k. |
| `min_p` | Min-p. |
| `repeat_penalty` | Repetition penalty. |
| `repeat_last_n` | Penalty window. |
| `presence` | Presence penalty. |
| `frequency` | Frequency penalty. |
| `seed` | RNG seed. |
| `min_keep` | Minimum tokens kept. |
| `logprobs` | Number of top logprobs, ≤20. |
| `max_tokens` | Policy field for the caller. |
| `bias` | Logit bias, `ID:VAL;ID:-inf`. |
| `preset` | `greedy`, `qwen_thinking`, or `qwen_non_thinking`. |

## Greedy output

Greedy output stays byte-identical by construction.

- When the parameters are equivalent to argmax (temp ≤ 0, no bias, penalties
  off) and logprobs=0, the host never touches the head's four words.
- With the sampler active at temp=0 (for example, `logprobs=5`), the host
  argmax uses the same first-index tie rule as `gpu_amax`. The GPU receipts
  verify this.

## Speculative decoding under sampling

`llb_smp_verify` implements the acceptance rule of speculative sampling.
Accept the draft x with probability min(1, p(x)/q(x)). On reject, draw from
norm(max(0, p − q)). The emitted token is then exactly p-distributed, for any
drafter.

- Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via
  Speculative Decoding*, ICML 2023 (arXiv:2211.17192), Algorithm 1 and
  Appendix A.1 (proof of correctness).
- C. Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling*, 2023 (arXiv:2302.01318). Same rule, derived independently.

Both drafters in this repository are deterministic: prompt lookup
(`LLB_SPEC`) and the argmax of the MTP head (`LLB_MTP`). So q is one-hot at
the draft. The engine accepts with probability p(x), or else draws from p
with x removed. Under greedy decoding, this reduces to "accept if and only if
x == argmax", the earlier served-row check.

The engine hook is in `spec_pass`. The token of row r is
`verify(row r logits, spec_tin[r+1])`. The last row has no draft and is a
plain draw. A rejected draft yields a different token, so the engine's next
request starts a new cycle there, the same as a greedy mismatch.

## Run the gates

The gates run on the CPU. They need no GPU.

1. Build `~/llama.cpp` (libllama). The compat gates and mutants need it.
2. Run the compat gates.

   ```sh
   tools/sampler/compat/run_gates.sh
   ```

   | Gate | Check |
   |---|---|
   | Gate 1 | Stage by stage against libllama, 177 cases. |
   | Gate 2 | 200k-draw χ² against the llama dist, plus determinism. |
   | Gate 3 | 15 stop-string cases. |
   | Misc | Presets and logprobs against numpy. |

3. Check that every mutant is caught.

   ```sh
   tools/sampler/compat/mutants.sh
   ```

4. Run gate S. `N` is optional.

   ```sh
   tools/sampler/run_spec_stat.sh [N]
   ```

Gate S decodes a toy LM (V=8) directly and through the engine's spec cycle:
verify rows in order, residual on reject, bonus row on full accept. It checks
both empirical joints over the first 3 tokens against the exact joint
enumerated from `llb_smp_dist`.

- Thresholds, fixed before the first run: χ² ≤ df + 5√(2df), zero hits on
  zero-probability sequences, and acceptance inside (0.05, 0.95) so that the
  reject branch is exercised.
- 5 configs: plain temp; the k/p/min_p chain; the chain plus penalties; a
  stochastic drafter q; and bias with k=1.
- Also checked: greedy spec output is byte-identical to direct output (2000
  toy models), and the determinism matrix.
- The mutant `MUT_SPEC_NORESID`, which resamples p instead of the residual,
  must fail.

## Limitations

- The same seed does not reproduce the tokens of llama.cpp, because the RNG
  is not mt19937. Compare distributions instead.
- `max_tokens` is a policy field for the caller. The engine stops at
  `--steps`.
- Stop strings run on detokenized bytes, on the server or front-end side,
  not in the engine.
- `logprobs` is limited to 20.
