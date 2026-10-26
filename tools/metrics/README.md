# Metrics and timings

`tools/metrics/` provides production telemetry for the Eureka Engine
(W1-METRICS). It has three parts:

- A Prometheus text endpoint, `/metrics`.
- A llama.cpp-style `timings` object for each request.
- A small C interface, `metrics.h`, that other components call.

The code is plain C11 with pthreads. It uses no client library. The
Prometheus text is written by hand.

## Files

| File | Description |
|---|---|
| `metrics.h` | The interface. This page is its contract. |
| `metrics.c` | Aggregates, histograms, sliding-window percentiles, the Prometheus renderer, and the standalone listener. |
| `tokstream.c` | `llb_token_stream()`: a `FILE*` that keeps the engine's id prints and timestamps each line as one token. Also `llb_stdin_queued()`. |
| `test_metrics.c` | Unit tests: percentile math (checked against numpy), bucket edges, window wrap, timings JSON, and exposition shape. |
| `numpy_ref.py` | Prints the numpy reference values that the tests hard-code. |
| `metrics_demo.c` | CPU-only demo. It sends 20 timed requests through the API, then serves `/metrics`. |
| `w1load.py`, `w1run.sh` | GPU gate driver. It checks golden byte identity (base build against metrics build) and runs a 20-request load with scrapes, under the bench lock. |

## Use metrics with the stdin engine

The `LLB_SYS1` stdin engine ([`tools/sys1dual/dual.c`](../sys1dual/dual.c))
records metrics today.

1. Set `LLB_METRICS_PORT` to start the standalone listener thread.

   ```sh
   LLB_METRICS_PORT=PORT ENGINE_COMMAND
   ```

   Replace `PORT` with a TCP port and `ENGINE_COMMAND` with the engine
   command line.

2. Scrape the listener.

   ```sh
   curl -s localhost:PORT/metrics
   ```

   The listener answers `GET /metrics` and `GET /health`. It returns 404 for
   anything else.

3. Optional: add `timings` to every response with `LLB_TIMINGS=1`, or to one
   generate request with `"timings":true` in that request.

### How the stdin engine records a request

- Every request line gets a timer. `llb_req_begin` runs when the line is
  read. `llb_req_start` runs when the engine pass starts. `llb_req_token` runs
  once per generated id. `llb_req_end` runs when the response line is
  written.
- Per-token clocks need no engine change. The engine prints each id as
  `fwrite(id) + "\n"` before it starts the next decode step. The memstream
  that captured those prints is now `llb_token_stream()`. It is line-buffered
  and timestamps each newline. It keeps the same bytes, so responses do not
  change.
- Cache accounting:
  - A retained-session repeat, served with no engine pass, is a cache `hit`
    with `cache_n = prompt_n`.
  - A continuation (`--pos K`) is a `partial` with `cache_n = K`.
  - Everything else is a `miss`.
  - The prefix cache (W1-KVPFX) adds its own numbers through
    `llb_req_start(r, prompt_n, cached_n)`.
- Queue depth: in stdin mode, the queue depth counts the complete request
  lines that are already buffered in `stdin` and not yet read. It is a lower
  bound, because bytes still in the pipe are not visible. A server sets the
  value with `llb_metrics_queue_depth()`.
- Defaults are byte-identical. Without `timings`, response lines are exactly
  what the old build wrote (gate G1, run by `w1run.sh`). The per-request JSON
  line on stderr is unchanged. A retained record stores the untimed line, so a
  repeat stays byte-identical to the original.

### Metrics under llb-serve with several engines

- Under `llb-serve --parallel N` with N > 1, each engine child gets its own
  port. engine[i] listens on `LLB_METRICS_PORT + i`, so the ports are
  P..P+N-1.
- `/slots` lists each port as `metrics_port`.
- `llb-serve` exits with status 2 at startup if that range passes 65535 or
  contains its own `--port`.
- With N = 1, the variable reaches the one engine unchanged.
- Before this fix (R5-FIX), every child bound the same port and all but one
  died on bind. The interim rule was "metrics only with `--parallel 1`". That
  rule no longer applies.

## Options

| Variable or field | Default | Description |
|---|---|---|
| `LLB_METRICS_PORT=N` | Unset | Starts the standalone listener on port N. Under `llb-serve --parallel N`, engine[i] uses `LLB_METRICS_PORT + i`. |
| `LLB_METRICS_HOST` | `127.0.0.1` | Listener address. |
| `LLB_TIMINGS=1` | Off | Adds `timings` to every response. |
| `"timings":true` | Off | Request field. Adds `timings` to that response only. |

## Use metrics from a server

A server (W1-SERVER) uses this contract:

```c
#include "tools/metrics/metrics.h"      // link metrics.o (+ tokstream.o if you capture engine prints)
llb_req r;
llb_req_begin(&r);                       // request accepted (queue time starts)
llb_metrics_queue_depth(queued);         // whenever your queue length changes
llb_req_start(&r, prompt_n, cached_n);   // the engine pass begins
llb_req_token(&r);                       // each generated token as it is produced (SSE: when you emit the chunk)
llb_req_spec(&r, drafted, accepted);     // if speculative decoding ran
llb_req_end(&r, "chat", "ok");           // response done; outcome: ok | retained | refused | error
char t[1024]; llb_req_timings_json(&r, t, sizeof t);   // the "timings" object for the response
// GET /metrics:
size_t n; char *body = llb_metrics_render(&n); /* send, Content-Type: text/plain; version=0.0.4 */ free(body);
```

- Thread safety: every aggregate update and every render takes one mutex. One
  `llb_req` belongs to one thread.
- If the server runs the stdin engine as a child process, the engine's own
  listener (`LLB_METRICS_PORT`) already has the engine-side numbers. The
  server can proxy that listener. It can also link `metrics.o` and record its
  own HTTP-side view: TTFT as the client sees it, including tokenization and
  HTTP.

## Metric catalog

All names start with `llb_`.

| Metric | Type | Meaning |
|---|---|---|
| `requests_total{mode,outcome}` | counter | Finished requests. Mode is `generate`, `decide`, or `invalid`. Outcome is `ok`, `retained`, `refused`, or `error`. |
| `requests_in_flight` | gauge | Accepted, not yet finished. |
| `queue_depth` | gauge | Waiting for the engine. See "How the stdin engine records a request". |
| `prompt_tokens_total` | counter | Prompt tokens, cached plus prefilled. |
| `prefill_tokens_total`, `prefill_seconds_total` | counter | Tokens actually prefilled, and the time from pass start to first token. `rate(tokens)/rate(seconds)` is the prefill rate in tok/s. |
| `generated_tokens_total` | counter | Ids emitted. |
| `decode_steps_total`, `decode_seconds_total` | counter | Steps after each request's first token, and the time from first to last token. The ratio is the decode rate in tok/s. |
| `prefill_tokens_per_second`, `decode_tokens_per_second` | gauge | The last request's value. |
| `cache_requests_total{result=hit\|partial\|miss}`, `cache_tokens_total` | counter | Prompt-cache outcome per request, and tokens not prefilled. |
| `spec_drafted_tokens_total`, `spec_accepted_tokens_total`, `spec_acceptance_ratio` | counter/gauge | Speculative decoding. The ratio is `NaN` until something is drafted. |
| `time_to_first_token_seconds` | histogram | Arrival to first generated token. |
| `inter_token_latency_seconds` | histogram | Gap between consecutive tokens of one request. Recorded live, per token. |
| `request_duration_seconds` | histogram | Arrival to response written. |
| `queue_time_seconds` | histogram | Arrival to engine pass start. |
| `prefill_seconds` | histogram | Pass start to first token. |
| `latency_window{metric,quantile}` | gauge | Exact p50/p90/p95/p99 (plus `count`) over the last 1024 observations of ttft, itl, e2e, prefill tok/s, and decode tok/s. |
| `uptime_seconds` | gauge | Time since the module started. |

Histograms use Prometheus `le` semantics. A bucket counts values `<=` its
bound, and the `le` buckets are cumulative. The unit tests pin the edge cases.
Use the histograms for `histogram_quantile()` across scrapes. The
`latency_window` gauges are exact, with no bucket interpolation, and cover a
recent window.

The percentile method is linear interpolation between the closest ranks. This
is the same as the numpy default (`numpy.percentile(..., method="linear")`):
`h = q·(n−1)`, `v = s[⌊h⌋] + (h−⌊h⌋)·(s[⌊h⌋+1] − s[⌊h⌋])`.

## Timings object

The `timings` object has the llama.cpp shape plus three fields:

`{"cache_n","prompt_n","prompt_ms","prompt_per_token_ms","prompt_per_second","predicted_n","predicted_ms",
"predicted_per_token_ms","predicted_per_second","ttft_ms","queue_ms","total_ms","draft_n","draft_n_accepted"}`

| Field | Meaning |
|---|---|
| `prompt_n` | Tokens prefilled (prompt − cache_n), as in llama-server. `cache_n` is reported separately. |
| `prompt_ms` | Engine pass start to first token. This covers prefill plus the sampling of the first token, because this engine samples the first id from the prefill logits. |
| `predicted_n` | All generated tokens, as in llama.cpp. |
| `predicted_ms` | First to last token. This contains `predicted_n − 1` decode steps. |
| `predicted_per_second` | `(predicted_n−1)/predicted_ms`, the true decode rate. It excludes the first token, which is sampled from prefill. |

- llama.cpp divides `n_decoded` by a window that starts at the end of the
  prompt. The two values differ by that one-token convention.
- A decide request generates no tokens. Its `prompt_ms` is the whole pass.
  `"timings":true` works for decide requests too, because `clef_encode`
  ignores keys that it does not read. The load run exercises this case.
- Retained (cache hit) responses report `cache_n = prompt_n`, `prompt_n = 0`,
  and near-zero times.

## Limitations

- On the `LLB_SYS1` path, speculative decoding is forbidden (the `sys1.c`
  guard). The spec series exist and are unit-tested, but they read 0 or NaN on
  that path until the plain-spec path or the sampler (W1-SAMPLER) calls
  `llb_req_spec`.
- The host's `spec:` summary in `llb_host.c` is not wired. Wiring it needs a
  re-emit.
- Token clocks are taken when the id reaches the host's stdout. That is after
  the device step plus the host print. It is the latency that a streaming
  client sees from this process.
- The stdin engine serves one request at a time, so `requests_in_flight` is 0
  or 1 there.
- Stdin mode cannot see arrival time. A request's clock starts when the
  engine reads its line. Lines that a client pipelines behind a running
  request wait unseen in the pipe or the `stdin` buffer. Their `queue_time`
  reads about 0, and their TTFT excludes that wait. In the GPU load, the 4th
  burst request reported a TTFT of 342 ms after a client-side wait of about
  1.4 s. `queue_depth` still shows the backlog (3→2→1→0). A server calls
  `llb_req_begin` at accept time, which avoids this problem.
