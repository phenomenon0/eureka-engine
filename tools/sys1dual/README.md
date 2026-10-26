# llb-serve quickstart

`llb-serve` (`serve.c`) is the OpenAI-compatible HTTP server for the dual-mode
engine (SYS1-DUAL). It owns one engine child per slot. The child speaks the
stdin/stdout JSON-line protocol of `dual.c`. The server handles the HTTP framing,
the chat template, the OpenAI JSON shapes, and SSE. This page shows you how to
build the server, run it, and send requests to it. For the full wire format, see
[HTTP API](../../docs/reference/http-api.md).

## Before you begin

- Get the emitted engine C file and the engine `.gpu` program.
- Get a GGUF model file and its SYS1 sidecar.
- Install curl. To use an SDK client, install Python with `pip`, or Node with
  `npm`.

## 1. Build

1. Build `llb-serve` and `llb-engine` into a scratch directory:

   ```sh
   OUT=/tmp/llb LLB_ENGINE_C=ENGINE_C LLB_ENGINE_GPU=ENGINE_GPU \
     tools/sys1dual/build-serve.sh          # -> $OUT/llb-serve, $OUT/llb-engine
   ```

   Replace the following:

   - `ENGINE_C`: the path to `llb-engine.c`.
   - `ENGINE_GPU`: the path to `llb-engine.gpu`.

   The script compiles with clang 21 by default, because `llb-engine.c` uses
   `#embed`. To use another compiler, set `CC`. clang 18 does not support
   `#embed`.

2. Check the version:

   ```sh
   $OUT/llb-serve --version                 # llb-serve 0.9.0 (git <sha>)
   ```

## 2. Run

1. Start the server:

   ```sh
   export LLB_GGUF=~/clef-dl/clef-flash-Q4_K_S.gguf
   $OUT/llb-serve --port 8080 --alias clef-flash --engine $OUT/llb-engine \
     -- --threads 16 -- ~/sys1dual-work/sidecar 0 --steps 4096
   ```

2. Wait for the engine to load. `/health` returns 503 `Loading model` until the
   engine has loaded, and then returns 200. A script can wait for it like this:

   ```sh
   until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/health)" = 200 ]; do sleep 0.5; done
   ```

## 3. Send requests with curl

1. Get the version and the model list:

   ```sh
   curl -s localhost:8080/version
   curl -s localhost:8080/v1/models
   ```

2. Send a chat completion:

   ```sh
   curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "messages": [{"role": "user", "content": "What is the capital of France?"}],
     "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": false}}'
   ```

3. Send a streamed (SSE) chat completion, with a final usage chunk:

   ```sh
   curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "messages": [{"role": "user", "content": "Count to five."}], "max_tokens": 32, "stream": true,
     "stream_options": {"include_usage": true}, "chat_template_kwargs": {"enable_thinking": false}}'
   ```

4. Send a text completion:

   ```sh
   curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 16}'
   ```

## 4. Use an OpenAI client

Set the client's base URL to `/v1`. The server does not check the API key, but
most clients require you to set one.

Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
r = client.chat.completions.create(model="clef-flash", max_tokens=32,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print(r.choices[0].message.content)
for chunk in client.chat.completions.create(model="clef-flash", max_tokens=32, stream=True,
        messages=[{"role": "user", "content": "Count to five."}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}):
    if chunk.choices: print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

Node (`npm install openai`):

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8080/v1", apiKey: "none" });
const r = await client.chat.completions.create({ model: "clef-flash", max_tokens: 32,
  messages: [{ role: "user", content: "What is the capital of France?" }],
  chat_template_kwargs: { enable_thinking: false } });
console.log(r.choices[0].message.content);
```

The server accepts and ignores the `model` field. It serves one model, under the
name given by `--alias`. Both SDKs keep their connection alive between calls. See
[§5](#5-behavior-to-know).

## Options

This table lists the options that this page uses. For all flags, see
[HTTP API](../../docs/reference/http-api.md#flags).

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. The server has no authentication. |
| `--port` | `8080` | TCP port. |
| `--alias` | Basename of `$LLB_GGUF` | The model name that the server reports. |
| `--engine` | Required | The engine binary. |
| `--keepalive-seconds` | `30` | Idle time for a reused HTTP/1.1 connection. `0` = one request per connection. |
| `--cors` | Local origins | `ORIGIN,…` allows more origins. `'*'` allows all. `none` turns off CORS handling. |
| `--drain-seconds` | `120` | Shutdown wait for in-flight requests. |
| `--parallel` | – | Number of engine slots. `GET /slots` is served when it is greater than 1. |
| `--native-batch N` | Off | Native batching of decide tasks. See [HTTP API](../../docs/reference/http-api.md#native-batch). |
| `-- ARGS` | – | Engine arguments, such as `--threads 16 -- SIDECAR 0 --steps 4096`. |

## 5. Behavior to know

| Topic | Behavior |
|---|---|
| Endpoints | `GET /health`, `GET /v1/models`, `GET /version`, `POST /v1/completions`, `POST /v1/chat/completions`, `POST /v1/decide`, and unversioned aliases. `GET /slots` with `--parallel` > 1. |
| Batch requests | N items, one pass where supported. The questions of one decide task are answered from one engine pass. A `{"batch":[...]}` of N tasks or prompts runs N passes back to back on one slot: one HTTP request, one queue ticket, no reload. It is not fused into one pass. See [§6](#6-batch-requests-and-v1decide). |
| Keep-alive | HTTP/1.1 connections are reused for non-streamed responses, up to `--keepalive-seconds` idle (default 30; 0 = one request per connection). An SSE stream always ends with the connection closed. HTTP/1.0 or `Connection: close` closes the connection after the response. Pipelined requests are answered in order. `Expect: 100-continue` is honored. |
| Errors | Every error is `{"error":{"message":…,"type":…,"param":null,"code":<HTTP status>}}`. An error event inside an SSE stream has the same shape, followed by `data: [DONE]`. Types: `invalid_request_error` (400/405/408/411/413), `permission_error` (403), `not_found_error` (404), `unavailable_error` (503: loading, engine gone, shutting down). |
| CORS | Browser pages from `http(s)://localhost`, `127.0.0.1`, or `[::1]` (any port) are allowed. The server answers the preflight, and `Access-Control-Allow-Origin` echoes the origin. A request that carries any other `Origin` gets 403. Use `--cors https://app.example,…` to allow more origins, `--cors '*'` to allow all, or `--cors none` for no CORS handling. Requests without an `Origin` (curl, SDKs) are unaffected. |
| Shutdown | SIGTERM or SIGINT stops accepting new connections and closes idle kept-alive ones. The server drains in-flight requests for up to `--drain-seconds` (default 120). A second signal skips the drain. Then the server sends SIGTERM to any engine still busy (and SIGKILL 10 s later), and closes each engine's stdin. A request still queued gets 503 `server is shutting down`. |
| Version | `GET /version` returns `{"name":"llb-serve","version":"0.9.0","git":…,"model":…,"slots":…,"keepalive_s":…,"cors":…,"api":"openai-v1"}`. You can also run `llb-serve --version`. |
| Sampling | `temperature`, `top_p`, `seed`, `stop`, and the other sampling fields are accepted and passed through to the engine. This build's engine is greedy (header `X-LLB-Sampler: greedy`). |
| Limits | `n` = 1 only. Request bodies must use `Content-Length` (no chunked bodies), with a maximum of 8 MiB. Text content parts only. No tools without the structured-output build (G16; see [Structured output](../../docs/guides/structured-output.md)). The server binds 127.0.0.1 unless `--host` says otherwise, and has no authentication. |

## 6. Batch requests and `/v1/decide`

Disclosure: batch requests = N items, one pass where supported. The engine runs
one sequence per pass. The questions of one decide task (1..128) are answered
from one pass. A batch of N tasks or prompts is N passes, run back to back on one
slot. Each batch item gets exactly the bytes that the single-item request would
get (masked: `id`, `created`, `timings`).

### Send a decide task

`POST /v1/decide` takes one clef decide task and returns the engine's answer
object:

```sh
curl -s localhost:8080/v1/decide -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Storage holds 270 GB of a 300 GB quota; grew 15 GB last week.",
  "questions": {
    "over_quota": {"type": "noul", "instructions": "Is used storage above its quota?"},
    "capacity":   {"type": "score", "instructions": "How much headroom is left?", "criteria": ["Plenty", "Watch", "Critical"]}
  }}'
```

```text
{"model":"clef","answers":{"over_quota":{"type":"noul","noul":0.03...},"capacity":{"type":"score","score":1.2...,...}}}
```

Ask every question of one state in one task. That is the one-pass form: one
prefill for the state, and the head scores every question from the same pass. N
one-question tasks over the same state cost N prefills. The answers also differ,
because the task is rendered and decided jointly ("Decide every field jointly").

### Send a batch

`{"batch": [...]}` works on `/v1/completions`, `/v1/chat/completions`, and
`/v1/decide`:

```sh
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "max_tokens": 16,
  "batch": [ {"prompt": "Summarize: ...", "custom_id": "a"},
             {"prompt": "Summarize: ...", "custom_id": "b", "max_tokens": 32} ]}'
```

```text
{"object":"batch","model":...,
 "data":[{"index":0,"custom_id":"a","response":{"status_code":200,"body":{...the single-item response...}}}, ...],
 "usage":{...summed...},"llb":{"items":2,"ok":2,"failed":0,"wall_ms":...}}
```

| Rule | Behavior |
|---|---|
| Item fields | Item i has its own fields, plus the batch object's other fields that it does not set (shared defaults). `batch`, `stream`, and `custom_id` are not shared. `custom_id` is echoed back and never sent to the engine. |
| Per-item errors | A bad item gets its own `status_code` and the usual error body. The other items still run. If the engine dies mid-batch, the remaining items get 503 `engine exited`. |
| Whole-request errors | These are answered as a normal error: `batch` is not an array of 1..4096 items, the server is loading, or the engine is down. |
| Streaming | `"stream": true` on the batch sends one SSE event per item as it finishes (the `data[i]` object). Then it sends a `{"object":"batch.done",...}` event, then `[DONE]`. Per-item `stream` is refused. |
| Scheduling | One queue ticket, one slot held for the whole batch, items in order. With `--parallel` > 1, the batch does not fan out across slots. |
| OpenAI array prompt | `/v1/completions` also takes `"prompt": ["...", "..."]`. It runs as a batch, and is answered as one `text_completion` with `choices[i]` and summed `usage`. One failing prompt fails the request with its error. `stream` is refused here. Use the batch form instead. |

Decide errors: a decide task that the engine cannot encode returns 400
`invalid_request_error`, and the engine stays up. Examples: an unknown question
type, empty `questions`, or a schema or state over the 16,384-token budget.

## Related pages

- [HTTP API](../../docs/reference/http-api.md): the full wire format, flags, and
  opt-in serving features.
- [Multimodal content](../../docs/reference/multimodal-content.md): image and
  audio content in chat requests.
- [Decide configuration](../../docs/reference/decide-configuration.md): decide
  path tiers.
- [Dual-mode residency](../../docs/reference/dual-mode-residency.md).
