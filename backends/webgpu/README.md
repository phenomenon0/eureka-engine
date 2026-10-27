# WebGPU

The in-browser engine: `chat` and `decide` in a browser tab. There is no
model server; the page reads the model files over HTTP byte ranges and runs
them on the GPU. The browser `see`, `hear`, and `speak` pages are part of the
app, in `github.com/phenomenon0/bend-app` (`public/senses/`).

This backend is preview tier. The `chat` and `decide` engine is a hand-written
WGSL port of the Bend trunk (`bend web-trunk-port`), not code emitted by the
Bend compiler. It is a test bed, not the reference implementation.

## Result

| Verb | Model | Machine | Measured |
|---|---|---|---|
| `chat` | clef-flash Q4_K_S (9B) | Mac Studio, M3 Ultra, Chrome 155 (Metal) | 35.8 tok/s decode (median, 10 reps) |
| `chat` | clef-flash Q4_K_S (9B) | RTX 3090, Chrome 146 (Vulkan/ANGLE) | 16.4 tok/s decode (median, 10 reps) |

Both machines return the same 104-token reply, byte for byte. The engine
issues 413 GPU dispatches per token.

## How each verb runs

| Verb | Runtime | Page |
|---|---|---|
| `chat` | WGSL trunk engine (`trunk/trunk_engine.js`, `trunk/trunk_kernels.js`) | `trunk/smoke.html` |
| `decide` | WGSL trunk engine plus the Clef decision head in plain JavaScript (`trunk/clef_head.js`, `trunk/clef_session.js`) | `trunk/clef.html` |
| `see`, `hear`, `speak` | onnxruntime-web (D-FINE-n, whisper-base.en, Kokoro-82M), WebGPU execution provider | App repository, `public/senses/` |

`see`, `hear`, and `speak` use onnxruntime-web, not Bend WGSL, so they live
with the app.

## Models, formats, and context

The trunk engine reads the model's shape from the GGUF and builds its kernels
for it. `trunk_check(g)` says in sentences why a file can't run.

| What | Accepted |
|---|---|
| Architectures | `qwen35` (every dense size: 0.8B, 2B, 4B, 9B, 27B, and Kev), `qwen35moe` (35B-A3B, 122B-A10B, 397B-A17B: softmax top-k routing, renormalized, plus the gated shared expert; separate or fused `gate_up` experts), `clef` |
| Shapes | Head size any multiple of 32 up to 512; delta-net state 16 to 256 (a power of two); conv kernel 2 to 16; any head counts that divide; up to 1024 experts. NextN/MTP blocks are skipped. |
| Weight formats | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, mixed freely |
| Context | Any length whose f16 KV buffer for one full-attention layer fits the GPU's binding limit (`e.max_ctx`; about 2 million tokens for the 0.8B on an RTX 3090). Attention is split-KV over 256-token chunks, dispatched for the tokens in use. |
| Memory | The whole trunk is resident on the GPU; `trunk_resident(check, ctx)` estimates it. There is no offload. |

`e.prefill(tokens)` runs a prompt in submissions of 64 tokens with no
per-token readback. Each token is still one pass of the decode graph, so
prompt speed is about decode speed.

### Verification

Receipts are in [`reports/webgpu-any-shape/`](../../reports/webgpu-any-shape/REPORT.md).

| Check | Result |
|---|---|
| Kernels: `trunk/qcheck.html` against gguf-py's reference dequantizers (`tools/qref.py`) | 30/30 tensors over all 21 quantized and float formats: dequantized values equal, GEMV relative error ≤ 2.6e-6 |
| Real models, greedy, 4 prompts x 64 tokens, against llama.cpp on the CPU | 0.8B, 2B, 4B, 9B: same first token every time; every divergence is a near tie in llama.cpp itself (top-2 gap ≤ 0.17 logit; the trunk's pick is its #2 or #3) |
| Every format on the 0.8B | All 21 run. IQ1_S, IQ1_M, IQ2_XXS differ from llama.cpp's quantized-activation CPU path, but equal llama.cpp on the same weights dequantized to F32 (32/32 tokens) |
| Random-weight files in shapes no release has: head 128, state 64, conv 3, 2 KV heads; MoE with 16 experts / 4 used and a shared expert, separate and fused | 24/24 greedy tokens equal to llama.cpp |
| Old engine vs this one on the shapes the old one ran (0.8B, 2B, 4B, 9B) | Same ids 16/16; decode speed within 1.3% (interleaved medians) |
| Needle in a long prompt, 0.8B, RTX 3090 | Found at 20,973 tokens (129 s prefill) and 153,352 tokens (1,586 s prefill, 49 tok/s decode after) |
| Chrome on an M3 Ultra (Metal) | 0.8B, 2B, 4B, 9B, five formats, and the synthetic dense and MoE files run; 0.8B 169 tok/s, 9B 36 tok/s. Needle found at 20,973 tokens on the 2B and 153,352 tokens on the 0.8B (59 tok/s decode after) |

## Layout

| Path | Contents |
|---|---|
| `trunk/` | WGSL trunk engine, Clef decision head, and test pages. |
| `engine/` | GGUF reader, tokenizer (`tok.js`, `tok.wasm`), and the in-tab 4B engine (`q35_*.js`, `bend-worker.js`, `app-worker.js`). |
| `emit/` | WGSL kernels for the 4B engine in `engine/`. |
| `tools/serve.mjs` | Static server with byte ranges and COOP/COEP headers. |
| `tools/run_chat.mjs` | Playwright driver that loads a page N times and print the results as JSON. |
| `trunk/qcheck.html`, `tools/qref.py` | Kernel check: runs the GEMV and embedding kernels on GGUF tensors; `qref.py` compares them with gguf-py's reference dequantizers. |
| `trunk/trunk_iq_grids.js` | The IQ quant codebooks from ggml (MIT), as one table. |

## Requirements

- A browser with WebGPU (Chrome 146 or later was tested).
- Node.js 20 or later.
- Playwright and Google Chrome, for the drivers in `tools/`.

## Download separately

These files are not in this repository.

| Item | Path under the models directory |
|---|---|
| `clef-flash-Q4_K_S.gguf` (9B, SHA-256 `ec1472cd7c72cf933ca2da528c466e0932d7814aca503de53965417eec5b49c2`) | `clef-flash-Q4_K_S.gguf` |
| Qwen3.5 tokenizer (`qwen35-tokenizer.bin`, SHA-256 `de71a648a71960a539d4d660ebff804a63a691ba74aea89e3e09dc7bf4553ad9`) | `qwen35-tokenizer.bin` |

## Build and run

1. Put `clef-flash-Q4_K_S.gguf` and `qwen35-tokenizer.bin` in `models/`.
2. Start the server:

   ```sh
   node tools/serve.mjs 8802
   ```

   Set `MODELS_DIR` to use another models directory.

3. Open a page:

   | Verb | URL |
   |---|---|
   | `chat` | `http://127.0.0.1:8802/trunk/smoke.html?m=clef-flash-Q4_K_S.gguf&ctx=4096&n=128&p=Hello` |
   | `decide` | `http://127.0.0.1:8802/trunk/clef.html` (one demo request), or `?cases=/cases/<file>.json` |

To repeat the `chat` measurement:

```sh
node tools/run_chat.mjs "<chat URL>" 10 chat.json
```

`run_chat.mjs` launches Chrome with the Vulkan/ANGLE flags
used on the RTX 3090. Change the flags for other GPUs.

## Batched prefill and the Safari/WebKit twins

The in-tab 4B engine (`engine/q35_engine.js`) prefills the prompt in chunks
instead of token by token. All prompt tokens but the last run layer-major in
chunks of 32 through the turbo kernels, with no head and no readback. The last
token is then stepped as before, for the first token's logits. This is on by
default in `bend-worker.js`. Send `{prefill: false}` to get the
token-by-token prompt back. Kev's decide steps still run token by token.

Safari/WebKit has no WebGPU subgroups, so the engine selects a different
kernel set there. On the subgroup path (Chrome), the plan is unchanged.

| Device | GEMV entries | gate+up | Modules |
|---|---|---|---|
| Subgroups (Chrome) | R x T twins for q4k/q5k (`prefillRt`), grid-token twins for q6k (`prefillGrid`), batched entries (`prefillBatched`) | `r1t8` | `k_gemv_turbo_rtf`, `_grf`, `_guf`, `_b`, `k_fuse_rcs` |
| No subgroups (Safari/WebKit) | Grid-token twins of all 7 GEMV entries of the shipped non-subgroup `k_gemv_turbo.wgsl` (q4k2560/4096/9216, q5k2560/4096, q6k2560/9216) | `g` (`gu_g`) | `k_gemv_turbo_ngrf`, `k_fuse_rc` |

Both paths also load `k_embed_tb`, `k_fuse_tb`, `k_norms_tb` and
`k_fuse_pfb`. `k_fuse_pf` is used only when `prefillTokens` is false. A
grid-token twin keeps every function of the shipped entry byte-identical.
Only `main` changes: it reads the token slot from `workgroup_id.y` and offsets
x and out by it.

Engine options (`bend-worker.js` passes them through):

| Option | Default | Effect |
|---|---|---|
| `prefill` | `true` | `false` steps the prompt token by token, as the shipped engine did. |
| `prefillNsg` | `true` | No subgroups only. `false` uses the per-token GEMVs inside the batched prefill. |
| `prefillGrid`, `prefillGu` | See the table above | Overrides the twin selection. `prefillGu: "g"` is valid only without subgroups. |
| `prefillChunk`, `prefillBatchWidth` | 32, 4 | Chunk size and batch width. The chunk must be a multiple of the width. |
| `prefillKV` | `"inplace"` | `"pingpong"` writes the whole cache every token, like the decode step. |
| `prefillTokens`, `prefillRecur`, `prefillEmbed`, `prefillRt`, `prefillBatched` | On | Per-stage switches. |

### Results

These were measured on an M3 Ultra with the race harness (86- and
413-token prompts). Each value is the median of 3 interleaved rounds in
WebKit and 2 in Chrome. "Before" is the engine with batched prefill but with
per-token GEMVs on WebKit. "After" is this engine.

| Browser | Prompt | TTFT before | TTFT after | Prefill tok/s before | Prefill tok/s after |
|---|---|---|---|---|---|
| Safari/WebKit | 86 tokens | 730 ms | 495 ms | 117.8 | 173.7 |
| Safari/WebKit | 413 tokens | 3249 ms | 2324 ms | 127.1 | 177.7 |
| Chrome (control) | 86 tokens | 285 ms | 285.5 ms | 301.5 | 301.2 |
| Chrome (control) | 413 tokens | 1301.5 ms | 1304.5 ms | 317.3 | 316.6 |

Decode did not change beyond noise: WebKit went from 49.7 to 50.8 tok/s
(86 tokens) and from 46.8 to 47.1 tok/s (413 tokens).

The output is bit-identical to the shipped token-by-token engine:

- On the registered parity set, both WebKit and Chrome matched 6/6 prompts
  x 128 generated ids, with 0 differing logit bits. Follow-up turns matched
  4/4.
- Each of the 8 WebKit twins is bit-identical in WebKit to 32 dispatches of
  the shipped non-subgroup entry. A 1-ulp sabotage of the accumulator is
  caught on all 8.
- The R x T twins are not bit-identical without subgroups. WebKit compiles
  the shared weight decode differently for token slots 1 and up, so they are
  not used there.

To recheck the kernels, open this page in Safari:

```text
http://127.0.0.1:8802/engine/bcheck7.html?gr=k_gemv_turbo_ngrf.wgsl&mod=k_gemv_turbo_ngrf.wgsl
```

Expect "ALL BIT-IDENTICAL; sabotage CAUGHT". `k_gemv_turbo_ngrfx.wgsl` is
the sabotage twin that the page loads. The app never loads it.

### Source

From the webprefill3 lane, commit `3a66615`, adopted tree `race/n3`. The
engine files that this backend had before were byte-equal to the lane's
`race/shipped-src`, so the lane's runtime files were copied unchanged. This
is the same result as applying its `engine-prefill3-cumulative.diff`
(`c6948b93…`), which includes webprefill1 and webprefill2. The check-only
modules and pages of the earlier lanes are left out.

| File | Before | After |
|---|---|---|
| `engine/bend-worker.js` | `a47f970d…` | `967c807e…` |
| `engine/q35_engine.js` | `7e8caf15…` | `cb854877…` |
| `engine/q35_plan.js` | `a8a7864f…` | `b3b6aa53…` |
| `emit/k_gemv_turbo_ngrf.wgsl` | new | `79f9a3da…` |

The other 12 new `emit/` modules and `engine/bcheck7.html` match the lane's
`n3-tree.sha256`. The full list is in `engine/VENDOR.md`.

## Gaps

- The graded `decide` battery (15 cases, 30 questions) is not included.
  `clef.html` without `?cases=` runs one demo request.

## Source

- `trunk/`, `engine/` (except `app-worker.js` and `VENDOR.md`),
  `tools/`: the files that produced the graded receipts (WebGPU battery,
  2026-10-20). They were copied from the Mac test runner, which was not under
  version control. `trunk/trunk_kernels.js` and every file in `engine/` that
  also exists in `bend-app` are byte-identical to `bend-app` at the commit
  below. `trunk/trunk_engine.js` adds the hidden-state readout that the Clef
  head needs (`wantHidden`, `noHead`). `trunk/trunk_session.js` uses absolute
  `/engine/` and `/models/` paths. `clef_head.js`, `clef_session.js`, and
  `clef_prompt.js` are not in `bend-app` at that commit.
- `emit/`, `engine/app-worker.js`, `engine/VENDOR.md`:
  `github.com/phenomenon0/bend-app`, commit
  `41e85a686e79c14611ae091384bd29082b79e531` (2026-10-19), directory `public/`.

Changes from the source: `tools/serve.mjs` serves this layout instead of
fixed paths on the test machine; the drivers import `playwright` by package
name; developer machine paths removed from comments.
