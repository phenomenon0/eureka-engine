# HTTP API

`llb-serve` is the OpenAI-compatible HTTP server for the dual-mode engine
(SYS1-DUAL). It owns one engine child, `LLB_SYS1=1 llb-engine ...`. The child
speaks the stdin/stdout JSON-line protocol in `tools/sys1dual/dual.c`. Each HTTP
request becomes one `{"mode":"generate",...}` line to that engine. The server
contains no engine code. Tokenize, prefill, decode, and detokenize all run inside
the engine process, exactly as they do on the stdin path. This page describes the
wire format (W1-SERVER) and the opt-in serving features added after it.

The source is `tools/sys1dual/serve.c` (new, about 750 lines), next to `dual.c`.
The build script is `tools/sys1dual/build-serve.sh`. For a quickstart, see
[llb-serve quickstart](../../tools/sys1dual/README.md). For image and audio
content, see [Multimodal content](multimodal-content.md).

## Run the server

```sh
LLB_GGUF=CLEF_GGUF TMPDIR=SCRATCH_DIR \
  llb-serve --host 127.0.0.1 --port 8080 --alias clef-flash-Q4_K_S \
            --engine ENGINE_DIR/llb-engine -- --threads 16 -- SIDECAR 0 --steps 4096
```

Replace the following:

- `SCRATCH_DIR`: a scratch directory that you own.
- `ENGINE_DIR`: the directory that holds the engine binary.
- `SIDECAR`: the SYS1 sidecar.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` or `::` for all addresses. |
| `--port` | `8080` | TCP port. |
| `--alias` | Basename of `$LLB_GGUF` | Model ID reported in `/v1/models` and in every response. |
| `--engine` | Required | The engine binary. It must be the W1-SERVER relink. See [Build](#build). |
| `--chat-max-tokens` | `256` | Chat default when the request has no `max_tokens`. |
| `--drain-seconds` | `120` | Graceful-shutdown wait for in-flight requests. |
| `--max-body-mb` | `8` | Request body limit (G5). |
| `--mm` | `none` | What the engine takes through the media hook: `none`, `image`, `audio`, or `image,audio` (G5). See [Multimodal content](multimodal-content.md). |
| `--mm-remote` | `off` | Handling of http(s) image URLs: `off`, `on` (public addresses only), or `private` (any address). |
| `--mm-max-image-mb` / `--mm-max-images` | `6` / `8` | Per-image decoded cap / images per request. |
| `--ear DIR` | Off | Runs the speech-recognition worker (EARS-CANON). `llb-ear` sits beside `llb-serve` (`LLB_EAR_BIN`). `DIR` holds `Qwen3-ASR-0.6B-Q8_0.gguf` and `mmproj-Qwen3-ASR-0.6B-bf16.gguf`. Serves `POST /v1/audio/transcriptions` and turns a chat `input_audio` part into the user turn's text. See [Multimodal content §8](multimodal-content.md#8-audio-transcription). |
| `--vis-tower CLI --vis-mmproj MMPROJ` | Off | Sends images through the Qwen3.5 vision tower (MIMG). Each request's images run `CLI --batch MMPROJ` (cudavis-cli, outside the engine lock). The rows are stamped and sent as the dual.c `"images"` key (IMRoPE). Needs `--exact` (refused with `--decide-path`) and an mmproj whose rows are the trunk width (9B 4096, 27B 5120, 0.8B 1024; a mismatch is refused per request). Implies `--mm image`. See [Multi-image requests](#multi-image-requests). |
| `--ear-llm-cpu` | Off | Runs the ear's decoder on the CPU. The tower stays on the GPU. |
| `-- ARGS` | `--threads 16 -- $SYS1_SIDECAR 0 --steps 4096` | Engine argv. This is the usual SYS1 launch. `--steps` fixes the KV capacity. |

### Engine environment

The server sets `LLB_SYS1=1` and `LLB_SYS1_STREAM=1` in the engine's environment.
It unsets `LLB_PREFILL`, `LLB_SPEC`, `LLB_MTP`, and `LLB_DUMP_HIDDEN`, because the
SYS1 hook refuses all four (exact pass only).

## Endpoints

| Method | Path | Aliases | Result |
|---|---|---|---|
| GET | `/health` | `/v1/health` | `200 {"status":"ok"}`. Returns `503` (error shape, `"Loading model"`) until the engine's init line appears. Returns `503 "engine is not running"` after the engine dies. |
| GET | `/v1/models` | `/models` | `{"object":"list","data":[{"id":ALIAS,"object":"model","created":T,"owned_by":"llama-bend"}]}` |
| POST | `/v1/completions` | `/completions` | Text completion. |
| POST | `/v1/chat/completions` | `/chat/completions` | Chat completion. |
| POST | `/v1/decide` | `/decide` | One clef decide task. Every question of the task is answered from one engine pass (SERVE-BATCH). |
| POST | `/v1/see` | – | Object detection (D-FINE-N, COCO-80). Returns `503` until configured. See [Senses](#senses). |
| POST | `/v1/hear` | – | Speech to text through a resident whisper.cpp `whisper-server`. Returns `503` until configured. See [Senses](#senses). |
| POST | `/v1/speak` | – | Text to WAV with the `--tts` voice. Returns `503` without `--tts`. See [Senses](#senses). |

Any other path returns `404`. A wrong method returns `405`. Every connection is
HTTP/1.1 with `Connection: close` (no keep-alive). Request bodies must carry
`Content-Length`. Chunked request bodies get `411`. The body limit is 8 MiB.

## Request fields

| Field | Completions | Chat | Notes |
|---|---|---|---|
| `prompt` | String, or a 1-element array of strings | – | Token-array prompts are not accepted. |
| `messages` | – | Array of `{role, content}` | Roles are `system`, `user`, and `assistant`. `content` is a string or an array of parts: `text`, `image_url` (data: URI or http(s)), `input_audio`. Media is always validated, and served only with `--mm` (G5). See [Multimodal content](multimodal-content.md). With `--ear`, an `input_audio` part becomes the ear's transcript ([§8](multimodal-content.md#8-audio-transcription)). |
| `max_tokens` / `max_completion_tokens` / `n_predict` | Default 16 | Default `--chat-max-tokens` | Range 1..65536. The prompt plus `max_tokens` must fit in the engine's KV, otherwise 400. |
| `stream` | bool | bool | SSE. |
| `stream_options.include_usage` | Yes | Yes | Adds the OpenAI usage-only last chunk. |
| `n` | Must be 1 | Must be 1 | Otherwise 400. |
| `chat_template_kwargs.enable_thinking` | – | Default true | The model template's own switch (`<think>\n` versus `<think>\n\n</think>\n\n`). |
| `add_generation_prompt` | – | Default true | |
| `return_tokens` | bool | bool | Non-standard. The non-stream response gains `"llb":{"prompt_ids":[...],"ids":[...]}`. |
| `tools`, `functions`, role `tool` | – | 400 | Not supported in the wire-format release. |
| `temperature top_p top_k min_p seed stop presence_penalty frequency_penalty repeat_penalty repetition_penalty logit_bias logprobs top_logprobs n_probs typical_p` | Accepted | Accepted | Passed through verbatim to the engine line as `"sampling":{...}`. The engine applies `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty` and `repeat_penalty` / `repetition_penalty` through its sampler (`--sample`). The rest are accepted and ignored: `stop` strings, `logit_bias`, `logprobs`, `top_logprobs`, `n_probs`, `typical_p`. Without sampling fields, decoding is greedy. Responses carry the header `X-LLB-Sampler: w1-sampler`. |
| `model` | Ignored | Ignored | One model per server. |

### Chat template

The template is `flash-chat_template.jinja` (Qwen3.5 ChatML). The server
implements it in C for the no-tools subset:

| Message | Rendered as |
|---|---|
| System message (first only), trimmed | `<\|im_start\|>system\n{c}<\|im_end\|>\n` |
| User message | `<\|im_start\|>user\n{c}<\|im_end\|>\n` |
| Assistant message after the last user turn | `<think>\n{r}\n</think>\n\n{c}`, followed by `<\|im_end\|>\n`. The think block is split out exactly as the template does it. |
| Assistant message before the last user turn | `{c}` only, followed by `<\|im_end\|>\n`. |
| Generation prompt | `<\|im_start\|>assistant\n` plus `<think>\n` (thinking on) or `<think>\n\n</think>\n\n` (thinking off). |

The server sends the rendered chat text to the engine with `"special":true`, so
the engine parses the control tokens (`q35_tok_encode(..., parse_special=1, ...)`).
Completions omit `"special"`, which keeps the old encode. A `<|im_start|>` in a
completion prompt therefore stays plain text.

## Responses

A non-stream completion looks like this:

```json
{"id":"cmpl-…","object":"text_completion","created":T,"model":ALIAS,"system_fingerprint":"llb-sys1",
 "choices":[{"index":0,"text":"…","logprobs":null,"finish_reason":"length"}],
 "usage":{"prompt_tokens":5,"completion_tokens":16,"total_tokens":21},
 "timings":{"prompt_n":5,"prompt_ms":…,"prompt_per_token_ms":…,"prompt_per_second":…,
            "predicted_n":16,"predicted_ms":…,"predicted_per_token_ms":…,"predicted_per_second":…}}
```

A chat response is the same, except that `"object"` is `"chat.completion"` and
the choice carries `"message":{"role":"assistant","content":"…"}`.

`finish_reason` is `"stop"` when the model emits an end-of-generation token
(`<|im_end|>` or `<|endoftext|>`), and `"length"` when it reaches `max_tokens`
first. The reply ends before that token, and `usage.completion_tokens` counts the
tokens before it. The engine itself has no early stop. It still runs all
`max_tokens` steps and the server drops their text, so the reply takes as long as
`max_tokens` steps. `stop` strings are not applied. `logprobs` is always `null`.

### Timings

The timing fields use llama.cpp's names, but the server measures them.

| Field | Measures |
|---|---|
| `prompt_ms` | From the request line written to the engine until the first token event. This covers tokenize, prefill, and the first argmax. |
| `predicted_ms` | From the first to the last token event. This is the `n-1` decode steps. |
| `predicted_per_token_ms`, `predicted_per_second` | Computed from `n-1`. |

## Streaming (SSE)

The response headers are `Content-Type: text/event-stream`,
`Cache-Control: no-cache`, and `Connection: close`. Each event is
`data: <json>\n\n`. The stream ends with `data: [DONE]`.

| Event | Shape |
|---|---|
| Completions chunk | `{"id","object":"text_completion","created","model","system_fingerprint","choices":[{"index":0,"text":PIECE,"logprobs":null,"finish_reason":null}]}` |
| Chat chunk | `"object":"chat.completion.chunk"` with `"delta":{"content":PIECE}`. The first chunk also carries `"role":"assistant"`. |
| Final chunk | `"finish_reason":"length"`, plus an empty `text` or `{}` delta. It carries `usage` and `timings` in the llama.cpp style. |
| Usage chunk, with `stream_options.include_usage` | One more chunk: `"choices":[]` plus `usage` (the OpenAI style). |
| Engine refusal after the stream opened | `data: {"error":{...}}`, then `data: [DONE]`. |

Each piece is a real per-token event. Pieces are not buffered and replayed at the
end. Each event's text is that token's decoded bytes. The server holds back an
incomplete UTF-8 tail for the next event, so every chunk is valid UTF-8. A
held-back tail left at the end is sent before the final chunk. The pieces
concatenate to exactly the non-stream text. A gate checks this (G3), including a
multi-byte case (`café`/emoji).

## Errors

Errors use the OpenAI shape:

```json
{"error":{"message":"…","type":"invalid_request_error|unavailable_error|not_found_error","param":null,"code":<http>}}
```

| Status | When |
|---|---|
| 400 | Bad JSON, a bad field, `n≠1`, tools, a prompt plus `max_tokens` over the KV capacity, or an empty prompt. |
| 404 / 405 | Unknown path / wrong method. |
| 411 / 413 | Chunked body / headers over 64 KiB or a body over 8 MiB. Media over a size, side, or pixel cap also gives 413 (G5). |
| 415 | Media whose bytes are not an accepted format, or a data: URI with a non-image type (G5). |
| 503 | The engine is loading, or the engine exited. |

## Batch requests

This section describes batch requests (SERVE-BATCH).

Disclosure: batch requests are N items, one pass where supported. The questions
of one decide task are answered from one engine pass (one prefill). A
`{"batch":[...]}` of N tasks or prompts runs N passes back to back on one slot.
That means one HTTP request and one queue ticket, with no reload and no per-item
HTTP. The batch is not fused into one pass. The engine runs one sequence per
pass, and fusing would change the prompt and the answers.

- `{"batch":[item,...]}` works on `/v1/completions`, `/v1/chat/completions`, and
  `/v1/decide`. `"prompt":[s1,s2,...]` works on `/v1/completions`. That is the
  OpenAI array form, answered as `choices[i]`.
- The reply is
  `{"object":"batch","data":[{"index","custom_id","response":{"status_code","body"}}...],"usage","llb"}`,
  in item order.
- Each item's `body` is byte-identical to the single-item response for the same
  request (masked: `id`, `created`, `timings`). Gate: SERVE-BATCH G1, 3 × 24 items
  plus 2 × 200-item decide batches, all identical.
- `"stream":true` sends one SSE event per finished item.
- A bad item gets its own error status. The other items still run.
- A decide task that the engine cannot encode is answered with 400, and the engine
  stays up. Before SERVE-BATCH, such a task made the engine exit.
- Speed: a batch
  saves only the per-request HTTP time, so it runs at the speed of N single
  requests. To get the one-pass gain, put the questions of one state in one task.
- For field rules and examples, see
  [llb-serve quickstart §6](../../tools/sys1dual/README.md#6-batch-requests-and-v1decide).

### Fixed: per-decide device memory growth

This fix is BATCH-NATIVE-2. Every engine step leaked ~17 KB of Bend heap, about
6 MiB per decide. The cause: the host-served pass bang never ran its body, so its
x array, its task words, and its lends were never freed. One `--gpu 22GB` engine
life died after ~620 decides. `tools/payload-passdrop.patch` (compiler patch #60)
frees them in `llb_deliver`.

Receipt: 330 decides at `--native-batch 8`. The Bend heap bump is +2 KiB in total.
The engine's device share is flat after warm-up. It was +6.3 MiB per decide
before the fix. `LLB_MEMPROBE=1` prints the device MiB and the heap bump after
every request.

## Native batch

This section describes `--native-batch N` (BATCH-NATIVE).

Disclosure: native batch is N sequences, exact per item. With
`llb-serve --native-batch N` (2..8, default off), a `/v1/decide`
`{"batch":[...]}` goes to the engine N tasks at a time. The engine runs their
sequences in one forward per step. Each exact-tier pair GEMV (Q4_K / Q5_K / Q6_K,
Q4_K + residual) reads its weights once for all rows (`llb_bx_*`,
`tools/payload-batch.patch`). Every other kernel runs on each row's own state, on
its own stream. Each item's response is byte-identical to the same item sent
alone (gate S1-G1: B3 24/24 at N = 2/4/8, reversed order, error items). A shared
ID prefix is computed once and forked (exact copy).

Measured on 9B clef-flash, one RTX 3090: ≈1.5× decide items/s at N = 8 (24 items
113.4 s → 75.0 s). Exact arithmetic is ALU-bound per row, so the batch shares only
the weight reads and the decode. It is not a 5× tier.

- When `--native-batch` is absent, `LLB_BATCH` is unset and the engine never
  batches. Single requests and batches take the SERVE-BATCH path byte for byte.
- A non-stream batch's first item arrives after its group of N has run. Use
  `stream:true` to get each item as it is served.
- Generate and chat requests are not batched (decide only, stage 1).
- `--native-batch` refuses `--banks`.
- KV per batch row is sized by the row's own length, because attention reads
  slots `t <= pos` only. A 512-token 9B decide needs ≈ 83 MiB (32 MiB KV + 51 MiB
  GDN state), plus 25 MiB attention scratch.

## Continuous batch

This section describes `--native-batch N --native-cont` (BATCH-NATIVE-2).

Disclosure: native-cont: identity exact; in-order TTFT not improved; OOO measured
first-item TTFT 10.2 s vs 18.7 s group / 21.8 s in-order (B3 N=8), after the
bx_next reset fix; measured 2026-10-18 N-o8 B3s/stream first_ms 10169 ms, p50
40.2 s, p95 68.9 s; in-order path unchanged.

With `llb-serve --native-batch N --native-cont` (default off), the server sends a
`/v1/decide` batch to the engine in lines of up to 256 tasks. `--native-cont`
needs `--native-batch`. A task joins a free row at a step boundary instead of
waiting for its group of N. This is continuous batching (`LLB_BATCH_OOO=1`). The
engine answers each item when it is done (`{"bx_item":k}` routing), so a
`stream:true` batch gets its items out of order, each tagged by `index`.

Gate: each item is byte-identical to the same item sent alone. E1 4/4; B3 stream,
batch, and reversed 24/24; B1 + B2 single requests 48/48; B4 200/200. With
`--heads`, MH-G1, MH-G2, and MH-G3 pass. The engine exits with rc 0 in every life.
The in-order path (the non-stream reply, or the group path) is unchanged. The gain
is in the first items, not in the wall time.

- When `--native-cont` is absent, nothing changes. The core suite (B1+B2 48/48,
  B3 24/24, E1 4/4, N=8 B3 24/24) equals canonical byte for byte.
- Fixed before landing: stage 3b as first written crashed the engine (rc 1,
  `native batch: a planned item never ran`) on any plan larger than the plan
  before it. `llb_bx_plan` did not reset `bx_next`. The fix is that reset.
- `LLB_BATCH_GRAPH=1` (one CUDA graph per step) and `LLB_BATCH_PROF=2` are also
  available, default off. The identity gates cover them. No speed result is
  claimed.

## Multi-head batch

This section describes `--heads name=gguf[,...]` (BATCH-NATIVE-3).

Disclosure: multi-head batch is N rows, a per-row head, exact per row.
`llb-serve --heads probe=/path/head.gguf` (engine `LLB_HEADS`, up to 6 names
`[a-z0-9_-]`, default off) loads more clef decide heads beside head0. A decide's
`"head":"<name>"` runs only that head on the request's hidden rows. The response
ends with `"head":"<name>"`, like the S4 `head0`/`head2` keys, which keep their
meaning. The trunk pass never sees the head, so inside a `--native-batch` group
every row picks its own head: one forward, per-row readout.

Gate MH-G1: B3 24/24 equal to the same item's single run with the same head, for
each of these cases:

- All head0.
- All probe.
- Cyclic head0 / probe / no key.
- The same, reversed.
- An 8-item mix with an unknown head. The unknown head is refused with 400, and the
  other 7 are equal.

The probe head differs from head0 on 24/24 items, so a mis-route would show. With
no `head` key, results equal canonical single requests on 24/24.

Measured on 9B clef-flash, RTX 3090, n = 24, N = 8: a mixed-head batch costs the
same as a single-head batch (74.9 s versus 75.2 s), and is 1.55× faster than
serial single requests (115.9 s). Each extra head costs ≈ 0.46 GB host RAM and no
device memory.

- Heads are decide-class only.
- The loader takes the head's shape from its GGUF (ENGINE-HEAD27).
  `output.weight` cols must equal `<arch>.embedding_length`, and rows must equal
  the tokenizer's count.
- Every head must read the trunk's width. A 27B head on a 9B trunk is refused at
  start, and the message names both widths.
- Generate and readout requests refuse `head`.
- An unknown name is a 400, and the engine stays up.
- The readout runs on the CPU per row after the batch's lockstep. It takes
  ≈ 155–175 ms per decide, ≈ 5% of batch wall time.

## Modes

This section describes `--discipline strict|normal` (CATALOG-STRICT).

Disclosure: head loader: metadata-driven shape, 27B heads served.
(ENGINE-HEAD27; receipts omen `~/eh27/`.)

Disclosure: discipline=strict abstains far more and costs accuracy. On B1, omni
non_halluc goes 9.94 → 75.32, and when2call cannot_answer goes 40.15 → 81.39
(decision index Δ +10.61 [9.67, 11.55]). Omni accuracy goes 44.67 → 21.67
(Δ −23.0 [−26.5, −19.5]), and omni RMS-CE goes 6.1 → 40.1. It passes G1, G2, G4,
G5, and G6. It fails G3 (accuracy) and G7 (calibration). It is opt-in and is
never the default.

| Mode | GGUF | Default | How it is served |
|---|---|---|---|
| `normal` | The crown, `clef27-q4ks-head-c27-selected.gguf`, sha256 `ec724b223a722f7d185cc05e2fb54ffe13abc2d7e0cfbbbe96c6146c4de2e1cb` | Yes | `LLB_GGUF` = the crown plus its 27B sidecar (`gguf2pack --sidecar-only`), engine `--gpu 40GB`. |
| `strict` | `clef27-q4ks-head-dd-h-w2-cap.gguf`, sha256 `23df859c64f2649e46f97aca1111e749a493761c6b0860324a14caf91aee65dc`: the same trunk with the Phase-H discipline head `h-w2:cap` | No | `--heads strict=<gguf> --discipline strict` on the crown trunk. |

The flag is `llb-serve --discipline strict|normal` (engine `LLB_DISCIPLINE`). The
default is `normal`.

- With `strict`, a decide with no `"head"` key runs the registry head `strict`.
  This needs `--heads strict=<gguf>`. Otherwise the engine refuses at start.
- The response then ends with `"head":"strict"`, the same as asking for it.
- An explicit `"head"` still wins, so `"head":"head0"` gets the crown.
- With `normal`, nothing changes.
- The trunk is the crown (`LLB_GGUF`). Only the strict GGUF's head tensors are
  read, not its trunk.
- Pointing `LLB_GGUF` at the strict GGUF has had no B3 engine-parity check on
  llb-serve.

Measured on RTX 3090, B3 single requests, `--decide-path` with `LLB_PF_SSMQ8=1`:
the crown takes ≈ 15 s per decide, against ≈ 5 s for the 9B. Its head is F32,
≈ 0.5 GB host RAM per head.

Gates:

- The 9B head is byte-identical to before (head-test logits and every
  checkpoint; B3 24/24 on llb-serve).
- The crown and strict heads match `head-reference.py` within 1e-5
  stage-relative.
- On the crown trunk (B3, engine rc 0), `normal` with the strict head registered
  equals no flags (24/24).
- Under `strict`, no key equals `"head":"strict"` (8/8), and `"head":"head0"`
  equals `normal` (8/8).
- Strict differs from normal on 24/24, so a mis-route would show.

The B1 numbers above come from the llama-server eval harness.

Dev row, for the model card: acc 33.24, non_halluc 90.08, when2call 94.45,
cannot_answer 96.58, request_for_info 90.86.

The signal does not fully transfer. The head's p(dont_know) separates right
answers from wrong ones at AUROC 0.84 on dev, but at 0.61 on B1.

No blend fixes the default. 1,526 crown↔head blend points were swept, and 0
reached non_halluc ≥ 60 with acc ≥ 43. The best result was 56.5 non_halluc at
acc 43.1. For non_halluc ≥ 60, the best accuracy was 42.16. Strict is therefore a
mode, not a change to `normal`.

Catalogue: the app repository's `docs/MODES.md` and `catalogue/modes.json`.

## Concurrency, logging, and shutdown

- Each connection runs on its own thread. With the default single slot,
  requests reach the engine one at a time under one lock. `--parallel N` adds
  slots, and `--native-batch N` (with `--native-cont`) batches decide tasks
  inside a slot. See [Native batch](#native-batch) and
  [Continuous batch](#continuous-batch).
- A batch request is one request that holds the slot for all its items. See
  [Batch requests](#batch-requests).
- The engine-side session and continuation machinery is not used. The server
  sends no `session`, so every request starts at pos 0.
- Request log: the server writes one line per request to stderr:
  `TIME serve: #ID IP "METHOD PATH" STATUS BYTES MS prompt=N gen=N [stream] [client-gone]`.
- The server forwards the engine's stderr with an `engine: ` prefix. That
  includes the engine's per-request timing JSON lines.
- Client gone mid-stream: the server keeps reading the engine's events until that
  request's final line, so the protocol stays in sync. The log line marks the
  request `client-gone`.

On SIGINT or SIGTERM, the server shuts down gracefully:

1. It closes the listen socket, so new connections are refused.
2. It waits up to `--drain-seconds` for in-flight requests, including running
   streams, to finish.
3. It closes the engine's stdin. dual.c reads EOF, and the engine exits by itself.
4. It reaps the engine, sending SIGTERM after 30 s if needed, and logs
   `shutdown complete … engine exit rc N`.

## Engine-side change (dual.c)

The engine-side change is opt-in only.

| Env var or key | Effect | When absent |
|---|---|---|
| `LLB_SYS1_STREAM=1` | During a generate request, every ID the engine prints is also emitted immediately as a `{"token":ID,"text":PIECE}` line. The final response line is unchanged. | The protocol is byte-identical (gate G2). |
| `"special":true` (generate) | Encodes the prompt with special tokens parsed. | The old encode. |

The mechanism: `stdout` is pointed at a `fopencookie` stream for the request. The
stream writes every byte through to the same memstream as before, so the final
`ids` and `text` are computed unchanged. The stream answers each complete ID line
as an event. Nothing changed in `llb-engine.c`, `.bend`, the host, or the kernels.
The engine is a relink of the emitted C (`596ccd7a…`), so the `.gpu` program is
reused and nothing is re-JIT'd.

## Multi-image requests

This section describes the opt-in multi-image path (FOLD-CONTINUE MIMG).

A request can carry K >= 1 images. Each image's tower rows sit at its own
`<|image_pad|>` run. IMRoPE positions follow llama.cpp mtmd's multi-image rule:
image k's position base is p0_k = S_k - P_k, where P_k = the sum over earlier
images of n_j - max(nx_j, ny_j). Text after image k is at slot - P_k+1. The path
is off by default. Text-only requests and the G4 `"image"` form are unchanged.

| Interface | Usage |
|---|---|
| CLI | `LLB_VIS_EMBD='a.f32:nx:ny;b.f32:nx:ny;...'` (`;`-separated, 1..16 entries, paths without `;`), `LLB_VIS_ROPE=imrope`, exact tier. The prompt holds one run of nx*ny pads per entry, in order. One entry is the old path. |
| SYS1 (dual.c) | `"images":[{"rows":"X.f32","nx":NX,"ny":NY},...]` (1..16, total <= 16384 rows). |
| HTTP | `llb-serve --exact --vis-tower .../cudavis-cli --vis-mmproj .../mmproj-F16.gguf`. Chat `image_url` parts go to the tower, then to `"images"`. `usage.prompt_tokens` counts the expanded pads. |

SYS1 rules:

- Each `X.f32` needs the `llb-tower-rows-v1` stamp `X.json`, written by
  [`tools/vision/tower_stamp.py`](../../tools/vision/tower_stamp.py). The stamp
  holds mmproj_sha256, nx, ny, d (the tower rows width), and rows_sha256. `d` must equal
  the trunk width (4096 on the 9B, 5120 on the 27B, 1024 on the 0.8B), or `"images"` is refused.
- The engine's `LLB_VIS_MMPROJ_SHA256` must equal the stamp's. If it is unset,
  `"images"` is refused.
- Tower rows carry no trunk table. The binding from mmproj to trunk is set by the
  operator's environment. It is not cryptographic.
- The engine refuses the following with a one-line `{"error"}` and keeps serving:
  a run count, run length, or order mismatch; `image` together with `images`;
  unstamped rows; another mmproj; `session`; the fast tier, `LLB_SPEC`, or
  `LLB_MTP`.

Gate (`FOLD-CONTINUE/mimg/REPORT-MIMG.md`):

- 2-, 3-, and 4-image rows versus llama.cpp mtmd multi-image (CPU, 2ca15f5): 5/9
  are token-exact at 32 greedy tokens.
- The 4 others diverge at near-ties. The mtmd top-2 margins are 0.020-0.312. The
  engine's token is mtmd's top-2 each time, and the results are identical with
  mtmd's own projector rows.
- Every image's rope base equals mtmd's pos_0.
- Single-image rows are unchanged on 6/6. Text-only output is byte-identical
  (CLI plus served fix1, 76/76).

Disclosure: multi-image = industry-class, not exact-class. Strict mtmd
token-exactness is 5/9 (above), which is a FAIL as registered. The Verdict Gates
v2 paired gate M1-v2 (`mimg/V2-GATE.md`, frozen amendment A3) is PASS-EQUIV
against an industry-class floor:

- n = 1606 teacher-forced positions. R = mtmd CPU. The floor is the same
  llama.cpp built with CUDA.
- V-T1 D = +0.75 pt [+0.25, +1.27]. V-KL ratio 0.557 [0.498, 0.623]. V-FLIP
  high-margin flips: 6, versus the floor's 14.
- Controls: a perfect candidate is PASS-EQUIV. Wrong-rope garbage is FAIL
  (KL 1.55x).

No exact-class floor was measured. This path is therefore opt-in. It runs only
for 2 or more images, `"images"`, or `--vis-tower`. It is never a silent default.

Limitations of the multi-image path:

- The tower is a fresh process per request. It loads the 918 MB mmproj each time
  (~2-4 s), and it also computes its batch rows.
- Images are refused on the decide path (fast prefill).
- The Metal backend is still single-image.
- The engine runs past EOG. The server drops the text after it (see
  [Responses](#responses)), but the request still takes `max_tokens` steps.

## Pocket 0.8B tier

This section describes the 0.8B tier with d 1024 (E5-08B). Source:
[`reports/e5-08b/REPORT.md`](../../reports/e5-08b/REPORT.md).

Supported: Qwen3.5-0.8B Q4_K_M (24 layers / 8 heads / d 1024 / d_ff 3584) loads
and runs on the CUDA engine. It supports text, decide, and image-to-text through
`LLB_VIS_EMBD` rows through the existing imrope / `llb_img_*` block.
[`tools/llb/models.conf`](../../tools/llb/models.conf) has an `08b` row, fast 0.

- Decode: ~306 tok/s (bridges, 128 steps, warm: 309.8 / 305.3 tok/s) on the RTX
  3090.
- Gates: G0, G1a/b/c, G3, G4, G5, and G6 pass on the frozen bar. The 9B golden is
  64/64 + 68/68 on both binaries. 9B Q8_0 and Q4_K_M logits are byte-identical,
  old versus new.

Refused or not covered:

| Item | Status |
|---|---|
| Fast prefill, `LLB_SPEC` / `LLB_MTP`, `LLB_BATCH` | Refused at d < 2048 (rc 1, one-line message). `LLB_PREFILL=fast`, `LLB_SPEC=2`, and `LLB_BATCH=4` fail on the 0.8B. |
| 4B | Not measurable on canonical. There is no 4B CUDA shape, and both binaries refuse it. |
| Metal | Deferred (Mac disk floor). The Metal patches still refuse d 1024. |
| EOG in the raw engine | `llb-engine … --steps N` runs to N past EOG. Only callers stop at it (`llb-tok` measured: stops at 248046 `<\|im_end\|>`). |
| Build recipe | At E5-08B, a plain `bash build.sh` did not link, because the emitted C needs the sys1dual objects. Fixed by CANON-SERVE. See [Plain engine build](#plain-engine-build). |
| `capacity.tsv` | At E5-08B, it had no measured rows, so `llb run` and `llb chat` refused every model. Filled by CANON-SERVE. See [Capacity table](#capacity-table). |

## Plain engine build and capacity table

This section describes the CANON-SERVE changes of 2026-10-18.

### Plain engine build

`build.sh` links the request objects that the emitted C calls (`dual_next` /
`dual_done`: `tools/sys1dual/dual.c` plus the CLEF-3 sources, `-lopenblas`),
unless `LLB_CC_EXTRA` is set. It also writes `llb-probe` to `LLB_OUT`, and
[`tools/llb/install.sh`](../../tools/llb/install.sh) installs it.

- Before the fix: on `9319739c`, clang 21.1.7 failed with
  `undefined reference to dual_done / dual_next`
  (lab archive).
- After the fix: a from-scratch build is rc 0 with 0 warnings in about 9 s.
- The engine is `b97ce1ba…` and the `.c` is `89c84baa…`. Both are byte-identical
  to E5-08B's sys1dual-recipe engine.
- `tools/sys1dual/build.sh` is unchanged. Its `LLB_CC_EXTRA` recipe gives the same
  `b97ce1ba…`.

The build environment on the test machine:

```sh
export PATH=CLANG21_DIR/bin:$HOME/.bun/bin:$PATH   # clang 21 (#embed), bun
CC=clang LLB_BEND=BEND_CHECKOUT \
  LLB_OUT=SCRATCH_DIR bash build.sh   # BEND_CHECKOUT carries pins/bend-240ecef4 (bend2 tree aada4cbf)
LLB_OUT=SCRATCH_DIR tools/llb/install.sh DEST   # then DEST/llb doctor
```

Replace `CLANG21_DIR` with a clang 21 install, `BEND_CHECKOUT` with the
pinned compiler checkout, `SCRATCH_DIR` with a scratch directory, and `DEST`
with the install directory. The build needs CUDA 13.1 headers and libraries, and OpenBLAS.
`/usr/bin` has only clang-18, which lacks `#embed`.

### Capacity table

The table is [`tools/llb/capacity.tsv`](../../tools/llb/capacity.tsv). The policy
is unchanged: the smallest ballast target with RUN at prompts 16 / 1009 / 3011, SLOW excluded.
The threshold is the maximum achieved free MiB. RTX 3090 sm_86 only.

| Alias | Free MiB needed | Evidence |
|---|---|---|
| `9b` | 8192 | 8192 target: achieved 8191 / 8192 / 8191 MiB free. Original 0.1.2 engine (`6f20325a`), same as the preserved cells. |
| `08b` | 8192 | 8192 target: achieved 8192 MiB at all three prompts. Canonical `llb` + `b97ce1ba`. Decode 3.4 / 4.6 / 7.4 ms/step. |
| `27b` | - | Not re-run. No target is all-RUN. `llb` refuses it. |
| `9b-q8` | - | Not re-run. No target is all-RUN. `llb` refuses it. |

- Not established: 8192 MiB is the lowest level tested. Lower levels were not
  measured. Other cards were not run.
- FAIL cells: 9b-16384-p3011 and 08b-16384-p16 failed the harness's 64 MiB
  GPU-return check, because desktop apps were on the card. They stay FAIL.
  Neither is on the selected 8192 rows.
- The prereg, its amendments, and the cell logs are in the lab archive.
- End to end: `llb doctor` passes, and `llb chat 08b` completes at 294–313 tok/s
.
- `llb-serve` serves the stock 0.8B with `LLB_SYS1_NOHEAD=1`. Without it, the
  SYS1/CLEF engine exits at load with `clef: missing/corrupt head tensors` on a GGUF
  without the CLEF decision head. NOHEAD reads the width from the GGUF and, on a
  tied-embedding trunk, the readout rows from `token_embd.weight`. Decide is refused
  (no head).

## Encoder batch

This section describes the opt-in encoder batch (CHAIN-2-1-3 Stage 2).

Disclosure: encoder batch is N inputs, exact per input. The vision tower
(`vitc_encode_batch`, `cudavis-cli --batch`) and the ear tower
(`aut_encode_batch`, `LLB_EAR_BATCH`) take N images or N mel windows in one pass.
Each input's rows are byte-identical to that input encoded alone.

Gate (RTX 3090):

- 24/24 images (9B tower) and 12/12 audio windows are identical in every mode
  below.
- Single paths are unchanged (24/24 rows, 6/6 ear transcripts = canonical).
- `LLB_EAR_BATCH` worker IDs and rows equal flag-off on 6/6 clips.

How it works:

- Row-local ops (im2col, embed, LN, epilogues, rope, merger) run once over the
  concatenated rows.
- Attention is block-diagonal. Each input attends to its own rows, with the same
  padded planes as a single encode.
- GEMMs stay per input, at the single encode's shape. One GEMM over all rows
  (`--gemm cat`) is not exact: cuBLAS picks other kernels at the larger row count
  (0/24, 0/12). It is kept only as a measured contrast.

| Mode | Behavior |
|---|---|
| `seg` (default) | Per-input GEMMs on the default stream. |
| `streams` (recommended) | The same GEMMs on up to 8 side streams, one cuBLAS handle each, so small inputs' GEMMs overlap. |

Measured as the warm median of 5. 24 images = 13,574 tokens. 12 windows = 73.8 s
of audio.

| Tower | Serial | `seg` | `streams` |
|---|---|---|---|
| Vision, 24 in one pass | 10.31 img/s | 10.40 img/s (1.006x) | 10.74 img/s (1.041x) |
| Vision, batches of 8 | 10.25 img/s | — | 10.54 img/s (1.028x) |
| Ear, 12 windows | Tower RTF 0.00072 | 0.00064 (1.119x) | 0.00053 (1.362x) |

The vision gain is small because the two large images (page1472 at 2,116 tokens,
big3000 at 4,015) dominate GEMM time.

Memory: vision weights are 865.5 MiB. Activations scale with the batch's rows:
1,662.6 MiB for big3000 alone, 3,226 MiB for batches of 8, and 5,683 MiB for all
24 in one pass. Use `--max-batch B` to cap it. Ear weights are 361 MiB.

Usage:

- `cudavis-cli --batch MMPROJ OUTDIR [--gemm seg|streams|cat] [--repeat R] [--max-batch B] IMG...`
  writes `single-K.f32` and `batch-K.f32`, and prints BYTE-IDENTICAL per image. It
  returns rc 0 if and only if all images are byte-identical.
- `LLB_EAR_BATCH=1|seg|streams` makes `llb-ear`, and so `llb-serve --ear`, encode
  a request's windows in one pass.
- `llb-ear encb MMPROJ [...] WAV...` is the gate.
- The feature is off by default. Without these flags, nothing changes.

## Senses

Three routes serve the senses next to the engine: `/v1/see`, `/v1/hear`, and
`/v1/speak`. They do not use the engine or its slots. Each one is always routed.
A sense that is not configured returns `503` with the error shape, never `404`.
Each sense runs one request at a time (FIFO). Errors use the [error shape](#errors).

### `POST /v1/see`

Detects objects in one image with D-FINE-N (COCO-80 classes). The detector is a
child process, `tools/see/see_worker.py` (ONNX Runtime), forked on the first
request. To get the model and set up Python, see
[`tools/see/README.md`](../../tools/see/README.md).

To turn it on, set these environment variables before you start `llb-serve`:

| Variable | Default | Description |
|---|---|---|
| `LLB_SEE_SCRIPT` | Unset | Path to `see_worker.py`. `build-serve.sh` copies it to `OUT_DIR`. Unset means `503 vision is not enabled on this backend`. |
| `LLB_SEE_MODEL` | Unset | Path to `dfine_n_coco.onnx` (sha256 `4daf426f09e5e1496dd879f2818a65e8706137bd1d9d9f785173df86cee38604`). |
| `LLB_SEE_PYTHON` | `python3` | An interpreter with `onnxruntime` (or `onnxruntime-gpu`), `numpy`, and `Pillow`. |
| `LLB_SEE_EP` | `auto` | `auto`, `cuda` (fails instead of falling back to CPU), or `cpu`. |
| `LLB_SEE_LABELS`, `LLB_SEE_THREADS`, `LLB_SEE_TIMEOUT_MS`, `LLB_SEE_MAX_IMAGE_BYTES` | `coco80.json` beside the script, `4`, `30000`, `4194304` | Label table, ORT threads, per-request budget, decoded image cap. |

Request:

| Field | Type | Description |
|---|---|---|
| `image` (or `image_b64`) | string | Required. A `data:image/...;base64,` URL or bare base64 (no whitespace). JPEG, PNG, or any format Pillow reads. |
| `task` | string | `detect` (default). `describe` returns `503`: this build is detect-only. |
| `threshold` | number | Score cutoff in [0, 1]. Default `0.4`. |
| `max_detections` | integer | 1..300. Default `100`. |

Response (`200`):

```json
{"model":"dfine-n","task":"detect","image_wh":[810,1080],
 "detections":[{"box":[48.5,396.0,242.9,906.1],"label":"person","class_id":0,"score":0.9397}],
 "usage":{"images":1},
 "llb":{"images":1,"encode_ms":39.0,"preprocess_ms":9.27,"infer_ms":29.73,
        "ep":"CPUExecutionProvider","runtime":"onnxruntime","model_sha256":"4daf426f..."}}
```

`box` is `[x0, y0, x1, y1]` in original-image pixels. Detections are sorted by
score, highest first. `image_wh` is `[width, height]`. `llb.ep` names the
execution provider that actually ran.

Errors: `400` for a bad field, bad base64, or an image that does not decode;
`413` for an image over `LLB_SEE_MAX_IMAGE_BYTES`; `503` when the route is not
configured, the worker does not start, dies, or times out, or for
`task: describe`. A worker that dies or times out is restarted on the next
request.

### `POST /v1/hear`

Transcribes one WAV clip. `llb-serve` forwards the clip over loopback HTTP to a
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) `whisper-server` that you
start. The sidecar picks its own ggml backend: CUDA (`-DGGML_CUDA=ON`), Metal (the
default on Apple silicon), or CPU. This route is separate from the `--ear`
transcription route (`/v1/audio/transcriptions`).

1. Build `whisper-server` from whisper.cpp (graded with v1.9.2):

   ```sh
   cmake -B build -DGGML_CUDA=ON      # omit -DGGML_CUDA=ON on macOS (Metal) or for CPU
   cmake --build build -j --config Release --target whisper-server
   ```

2. Download the model
   [`ggml-base.en-q5_1.bin`](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin)
   (59,721,011 bytes, sha256
   `4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f`). You can
   also run `models/download-ggml-model.sh base.en-q5_1` in the whisper.cpp
   checkout.

3. Start the sidecar on loopback, then start `llb-serve` with `LLB_HEAR_URL`:

   ```sh
   build/bin/whisper-server --host 127.0.0.1 --port 8137 -m MODEL_DIR/ggml-base.en-q5_1.bin -t 8 --no-timestamps &
   export LLB_HEAR_URL=http://127.0.0.1:8137/inference
   export LLB_HEAR_MODEL_PATH=MODEL_DIR/ggml-base.en-q5_1.bin
   export LLB_HEAR_MODEL_SHA256=$(sha256sum "$LLB_HEAR_MODEL_PATH" | cut -d' ' -f1)
   ```

| Variable | Default | Description |
|---|---|---|
| `LLB_HEAR_URL` | Unset | `http://HOST:PORT/PATH` of `whisper-server`'s `/inference`. Plain HTTP only. Unset means `503 hearing is not enabled on this backend`. |
| `LLB_HEAR_MODEL_ID` | `whisper-base.en-q5_1` | Reported as `model` when the request has none. |
| `LLB_HEAR_MODEL_PATH`, `LLB_HEAR_MODEL_SHA256` | Empty | Reported in `llb`. The server does not open or hash the file. |
| `LLB_HEAR_ENGINE` | `whisper.cpp` | Reported as `llb.engine`, for example `whisper.cpp (GGML_CUDA)`. |
| `LLB_HEAR_TIMEOUT_MS` | `30000` | Upstream connect, send, and receive budget. |
| `LLB_HEAR_MAX_AUDIO_BYTES` | `16777216` | Decoded WAV cap (`413` above). The request body limit (`--max-body-mb`) also applies. |

Request:

| Field | Type | Description |
|---|---|---|
| `audio` (or `audio_b64`) | string | Required. A RIFF/WAVE file as a `data:audio/wav;base64,` URL or bare base64. |
| `task` | string | `transcribe` (default). Any other value returns `503`: this build is transcribe-only. |
| `language` | string | Echoed back. The sidecar is always asked for English (`base.en`). |
| `model` | string | Echoed back as `model`. |

Response (`200`):

```json
{"model":"whisper-base.en-q5_1","task":"transcribe","text":"What advice would you give to a young entrepreneur starting out today?",
 "language":"en","audio_seconds":3.3810,"usage":{"audio_seconds":3.3810},
 "llb":{"engine":"whisper.cpp","route":"/v1/hear","ms":…,"rtf":…,"audio_bytes":…,
        "model_path":"…","model_sha256":"4baf70dd…","upstream":"127.0.0.1:8137/inference"}}
```

`llb.ms` is the wall time of the upstream call. `llb.rtf` is that time divided by
the clip length.

Errors: `400` for a bad field, bad base64, or audio that is not RIFF/WAVE;
`413` for audio over the cap; `503` when the route is not configured, the
sidecar is unreachable, returns a non-200 status, or returns no `text`.

### `POST /v1/speak`

The `--tts` speech route under the senses name. It takes `{"text": "...",
"voice"?: "...", "speed"?: N}` and answers like `POST /v1/audio/speech` with
`response_format: wav`: a 24 kHz WAV body with the `X-LLB-TTS-*` headers. Other
fields, such as `style`, are ignored. Without `--tts`, it returns
`503 speech is not enabled on this backend`.

## Build

```sh
OUT=OUT_DIR LLB_ENGINE_C=$W/build/llb-engine.c LLB_ENGINE_GPU=$W/build/llb-engine.gpu \
  bash tools/sys1dual/build-serve.sh
```

Replace `OUT_DIR` with the output directory. `$W` is the engine worktree that
holds the emitted C and `.gpu` program.

This builds `OUT_DIR/llb-serve` (serve.c + clef3/json.c; serve.c `#include`s
`see_route.c` and `hear_route.c`) and copies `see_worker.py` and `coco80.json`
next to it. It also builds
`OUT_DIR/llb-engine`: the relink of the given emitted C with the rebuilt request
objects. Compile with clang-21, because `llb-engine.c` uses `#embed`.

## Limitations

The server does not support the following:

- Token-array prompts, `echo`, and `suffix`.
- Authentication and API keys.

The first wire-format release (W1-SERVER) also did not include the features in
the following table. Later releases added them.

| Feature | Where it is now |
|---|---|
| Sampling, stops, EOS, and logprobs | Request fields (W1-SAMPLER). |
| Metrics | Each engine serves them on `LLB_METRICS_PORT`, one port per engine. `llb-serve` does not serve `/metrics`. See [`tools/metrics/README.md`](../../tools/metrics/README.md). |
| Prefix-cache reuse | `LLB_PREFIX_CACHE=1` (W1-KVPFX). |
| Keep-alive | `--keepalive-seconds`. Streaming responses and the busy `503` response still send `Connection: close`. |
| CORS | `--cors`. |
| N-slot serving | `--parallel N` (`-np`), and `GET /slots` when more than one slot is configured. |
| Tool calls and reasoning content | See [Structured output](../guides/structured-output.md). |
| Multimodal content | See [Multimodal content](multimodal-content.md). |

The [llb-serve quickstart](../../tools/sys1dual/README.md#5-behavior-to-know)
describes keep-alive, CORS, and `--parallel`.
