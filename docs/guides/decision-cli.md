# Run the decision CLI

This guide shows you how to build and run `clef-decide`, a local decision CLI.
The CLI uses a plain C CPU head that reads the Clef GGUF. The CUDA engine
supplies each final output-normalized prompt row. `clef-decide` uses no Python
runtime.

## Before you begin

- Install these dependencies:
  - clang
  - The pinned Bend compiler and Bun, for the engine build. See
    [Run on CUDA](run-on-cuda.md#pinned-compiler).
  - The CUDA driver and NVRTC
  - OpenBLAS, for CPU SGEMM
- Download the Clef GGUF, for example `clef-flash-Q4_K_S.gguf`.
- Run every command from the repository root.

## Build the CLI

```sh
LLB_BEND=BEND_CHECKOUT bash tools/clef3/build.sh CLEF_GGUF
```

The script writes `clef-decide`, `clef-spans`, `llb-engine`, and the GGUF's
sidecar to `build/clef3/`. Set `CLEF_OUT` to use another directory.

## Run a decision

1. Point the CLI at the engine:

   ```sh
   export CLEF_ENGINE="$PWD/build/clef3/llb-engine"
   ```

2. Run one request:

   ```sh
   build/clef3/clef-decide \
     --gguf CLEF_GGUF \
     --sidecar build/clef3/sidecar \
     --request tests/fixtures/goldens/01-outage.request.json
   ```

The CLI writes one SystemOne response to stdout. It writes the render,
backbone, head-load, and head latencies to stderr.

`tests/fixtures/goldens/` holds 24 example requests with their expected
answers.

## Options

| Option or variable | Description |
|---|---|
| `--gguf` | Path to the Clef GGUF file. |
| `--sidecar` | Path to the sidecar directory. |
| `--request` | Path to the request JSON file. |
| `CLEF_ENGINE` | Path to the `llb-engine` binary. |

## Limits

| Item | Limit |
|---|---|
| Tokens | 16384 maximum |
| Questions | 128 maximum |
| Options | 4096 maximum |

## Limitations

- The CLI supports text and JSON only, on this GGUF and its shape.
- The CLI rejects oversized states. It does not truncate them silently.
- The CLI rejects images and videos.
- The CLI rejects missing instructions. The span helper keeps the reference's
  ID fallback.
- Strings that contain NUL are not supported.
- There is no HTTP route in this CLI. `llb-serve` serves decisions over HTTP;
  see the [HTTP API](../reference/http-api.md).
- This version uses exact token-by-token prefill. It still runs the vocabulary
  head before it discards the result.
- Each CLI process dequantizes the head weights to FP32 on the CPU, which costs
  seconds per request.
- Each request starts a new engine process, so KV and recurrent state cannot
  leak between requests.

Results: [reports/clef3-final.md](../../reports/clef3-final.md).
