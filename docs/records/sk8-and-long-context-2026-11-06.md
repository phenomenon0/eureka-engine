# Split-key attention and long-context decode, 2026-11-06

Two measurements made in the private working repository on an RTX 3090 with the
Qwen3.5-4B. They used the engine sources that the 2026-11-06 port carried into
this repository ([`PORT.md`](../../PORT.md)). Their receipts were not copied
here, so this page records the measured numbers and the method only.

## Split-key decode attention (`LLB_ATTN_SK8=1`) has no measurable NLL cost

Method, frozen before any run:

- 17 prompts from 8 to 14,426 tokens long, including a 3,769-token needle prompt;
- greedy reference tokens, then teacher-forced replay of each prompt;
- two repetitions per arm, 2,112 paired positions in all, every one finite;
- NLL is the mean of −log softmax(logits)[reference token], in nats;
- 90% bootstrap interval over prompts, then positions (seed 20261106, 2,000 draws).

"No measurable NLL cost" was defined in advance as follows: the 90% interval
of the pooled mean delta includes 0, and the mean's magnitude is under 1e-3 nat.

| Contrast | Mean ΔNLL (nat) | 90% interval | Verdict |
|---|---:|---|---|
| SK8 vs default | 6.1e-09 | [−1.4e-08, 2.8e-08] | no measurable cost |
| SK8 + `LLB_Q4_DP4A=9` vs default | 2.5e-06 | [−2.0e-06, 7.3e-06] | no measurable cost |
| SK8 + `LLB_Q4_DP4A=9` vs SK8 | 2.4e-06 | [−1.9e-06, 7.4e-06] | no measurable cost |

SK8 stays off by default. This result does not change any default.

## Long-context decode time is attention-bound

The 4B was served at prompt depths of 512, 2,048, 8,192 and 14,426 tokens, with
3 kept repetitions per depth.

| Depth | Decode ms/token (default) | tok/s |
|---:|---:|---:|
| 512 | 8.97 | 111.5 |
| 2,042 | 12.16 | 82.2 |
| 8,198 | 24.65 | 40.6 |
| 14,426 | 32.86 | 30.4 |

- Decode time per token grows linearly with depth: 1.74 ms per 1,000 tokens,
  R² 0.99.
- At 14,426 tokens, the attention kernel `llb_attn4` takes 24.8 ms of each
  token, against 1.1 ms at short depth. That is 77% of the GPU time.
- Attention accounts for all of the growth. Between short depth and 14,426
  tokens, the attention kernel's time grows by 23.7 ms, and the whole token
  grows by 23.0 ms.
- No engine change came out of this. It marks where decode time goes at long
  context. It is not a fix.
