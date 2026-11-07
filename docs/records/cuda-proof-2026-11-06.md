# CUDA clean-clone proof, 2026-11-06

The 2026-11-06 port was checked only up to compiling: no NVIDIA GPU was available
([`PORT.md`](../../PORT.md)). This record closes that gap. A fresh clone of this
repository was built and run on an RTX 3090. Receipts:
[`receipts/cuda-proof-20261106/`](../../receipts/cuda-proof-20261106/README.md).

| Step | Result |
|---|---|
| `./build.sh` from a fresh clone and the `pins/bend-240ecef4` branch | Exit 0. |
| Sidecars for Qwen3.5-2B Q4_K_M, 4B Q8_0 and 9B Q8_0 | Exit 0. |
| `llb-engine` on each model, `LLB_ATTN_SK8` off and on | All six runs exit 0 and decode to " Paris.". SK8 gives identical token ids on every model. |
| `llb-serve` on the 4B with `LLB_SYS1_NOHEAD=1` | `/health` 200, `/v1/models`, `/v1/chat/completions` and `/v1/completions` answer. `/v1/decide` is refused, as documented. |

## Found and fixed

- **`TMPDIR` must be set for `llb-serve`.** The engine stops at load with
  `clef: set TMPDIR to owned scratch` if `TMPDIR` is unset. macOS sets it.
  Many Linux distributions do not. The README's Linux serve command now sets it.
- **The CUDA server did not stop at the end of a turn.** For a request without a
  grammar or tools, the reply ran past `<|im_end|>` to `max_tokens` and made up
  further turns. The Metal server already stopped there. `llb-serve` now ends the
  reply at the first end-of-generation token (`<|im_end|>` or `<|endoftext|>`):
  `finish_reason` is `stop`, and `usage.completion_tokens` counts the tokens
  before it. The server's test suite passes (58 / 0).

## Still open

- The CUDA engine has no early stop. It runs all `max_tokens` steps, and the
  server drops the text after the end-of-generation token. So a reply takes as long
  as `max_tokens` steps, however short the text is. Set `max_tokens` to what you
  need.
- For a chat without tools, the model's thinking comes back inside `content`. It
  is not split into a separate reasoning field.
