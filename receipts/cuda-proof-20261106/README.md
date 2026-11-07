# Receipts: CUDA clean-clone proof, 2026-11-06

These receipts back [the CUDA proof record](../../docs/records/cuda-proof-2026-11-06.md).

Machine: Fedora 43 (Linux 6.17), x86-64, 16 threads, RTX 3090 (driver 580.119.02),
CUDA 13.1, clang 21.1.7, bun. The source was a fresh `git clone` of
`windows-test-20261106` at `fbea5f9`, plus `git fetch origin pins/bend-240ecef4:pins/bend-240ecef4`.
The `*-eogfix*` files come from the same tree with the `tools/sys1dual/serve.c` end-of-generation
fix applied.

Local paths were replaced by `$PROOF` (the proof directory), `~` and `$MODELS`. Each
`run-cli-*` file starts with the model and the first 16 hex digits of its sha256.

| File | What it shows |
|---|---|
| `env.txt` | Machine, compiler, CUDA and commit. |
| `build-cuda.txt` | `./build.sh`, exit 0. |
| `sidecar-{2b,4b,9b}.txt` | `llb-tok extract` + `gguf2pack --sidecar-only`, exit 0. |
| `run-cli-{2b,4b,9b}-sk8{0,1}.txt` | `llb-engine` on "The capital of France is", 32 steps, `LLB_ATTN_SK8` off and on. All exit 0 and decode to " Paris.". The token ids are identical with SK8 off and on for each model. |
| `serve-4b.log`, `serve-4b-requests.txt` | `llb-serve` on the 4B before the fix. Chat answers, but the text runs past `<\|im_end\|>` to `max_tokens`. `/v1/decide` is refused with 400 under `LLB_SYS1_NOHEAD=1`. |
| `build-structured.txt`, `structured-tests.txt` | The server's CPU test suite with the fix: 58 PASS / 0 FAIL. |
| `build-serve-eogfix.txt`, `serve-4b-eogfix*.{log,txt}` | `llb-serve` with the fix: chat, streamed chat and completions stop at the end-of-generation token with `finish_reason` `stop`. |

`wall_ms` values are not speed measurements. Other jobs shared the GPU and the
model files were read from a slow disk.
