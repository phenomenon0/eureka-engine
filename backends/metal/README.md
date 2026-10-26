# Metal

macOS, Apple silicon. Tested on a Mac Studio (M3 Ultra, 96 GB). The same
emitted C (`c13b9e28`) is the engine in the [iOS app](../ios/README.md).

## Source

- Engine (`sys1mac`): this directory. The Metal compiler and engine deltas
  are `cc/metal-cc.patch` and `engine/metal-engine.patch`.
- Server provider (`llb-serve`) and its sense sidecars: [`server/`](server/).

## Result

clef-flash Q4_K_S (9B): 70.2 tok/s decode.

## How each verb runs

`llb-serve` is one HTTP process. It starts `sys1mac` as a child process and
talks to it over stdin and stdout, one JSON line per request. It starts
`whisper-server` as a loopback sidecar, and it runs the `see` script once
per request.

| Verb | Route | Runtime |
|---|---|---|
| `chat` | `llb-serve` `POST /v1/chat/completions` | `sys1mac`, the Bend Metal engine |
| `decide` | `llb-serve` `POST /v1/decide` | `sys1mac`, same model as `chat`; the clef head, or the Kev head (`server/src/kevhead.c`) for a GGUF that carries one |
| `see` | `llb-serve` `POST /v1/see` | D-FINE-N on ONNX Runtime, CPU execution provider (`server/see/detect_cli.py`) |
| `hear` | `llb-serve` `POST /v1/hear` | whisper.cpp `whisper-server` on Metal, model `ggml-base.en-q5_1.bin` |
| `speak` | `server/speak/serve.py` `POST /v1/speak` | Kokoro-82M on ONNX Runtime, CPU execution provider, with Misaki G2P |

`speak` is a separate server. `llb-serve` also has a `/v1/speak` route
(`server/src/speak_route.c`), but it calls a launcher script, set with
`--speak-cmd`, that is not included here. The graded `speak` route was
`serve.py`, called directly.

## Build

On a Mac:

```sh
LLB_BACKEND=metal ./build.sh
```

This builds `sys1mac` (emitted C `c13b9e28`, the Mac product build).
`LLB_BEND` must point at the pinned Bend compiler: export the
`pins/bend-240ecef4` branch's `bend2/` (git tree `aada4cbf`) into a
directory with a `.commit` file. A plain clone does not fetch that branch;
run `git fetch origin pins/bend-240ecef4:pins/bend-240ecef4` first. How the
build reproduces the Mac product: [build-internals.md](build-internals.md).

### Server

Build `llb-serve`. It needs only clang. Build outside the source tree:

```sh
mkdir -p /path/to/scratch/bin
clang -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -pthread \
  server/src/serve.c server/src/json.c server/src/speak_route.c \
  -o /path/to/scratch/bin/llb-serve
```

`server/build.sh` builds `llb-serve` and also rebuilds `sys1mac` with the
server's patches (`server/patches/`, `server/build-engine.sh`). It needs the
Mac engine source tree in `EUREKA_ENGINE_SRC`, which is not in this
repository. That tree must have the five-argument `clef_encode` (with
`const char **err`, commit `27bec70`; this repository carries it as
`vision/clef-encode-err.patch`). It writes `bin/` and `work/` next to itself, so copy `server/` to
a scratch directory before you run it.

### Sense sidecars

`see`: a Python environment with `onnxruntime`, `numpy`, and `pillow`, and
the single-input D-FINE-N export (`dfine_n_hf.onnx`, 15,434,564 bytes,
SHA-256 `9c649a78457939665ea13189bdbfe8d1fa42711ab3e332376fafc3579dd468a9`;
`pixel_values` in, `logits` and `boxes` out). The two-input
`dfine_n_coco.onnx` that the CUDA server uses does not work here. Set
the paths for `llb-serve`:

```sh
export LLB_SEE_PY=/path/to/see-venv/bin/python
export LLB_SEE_CLI=$PWD/server/see/detect_cli.py
export LLB_SEE_MODEL=/path/to/dfine_n_hf.onnx
```

`hear`: build `whisper-server` from
[whisper.cpp](https://github.com/ggml-org/whisper.cpp). Metal is its default
backend on Apple silicon. Download `ggml-base.en-q5_1.bin` (SHA-256
`4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f`).

`speak`: Python 3.12 and the pinned packages:

```sh
cd server/speak
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download these files from `onnx-community/Kokoro-82M-v1.0-ONNX` (revision
`1939ad2a8e416c0acfeecc08a694d14ef25f2231`) into `server/speak/dl/`, or set
`TTS_ASSETS` to their directory: `onnx/model.onnx`, `voices/af_heart.bin`,
`tokenizer.json`, `config.json`. The expected hashes are in
`server/speak/ASSETS.sha256`.

## Run

Start `llb-serve` with the model and the `hear` sidecar:

```sh
export LLB_GGUF=/path/to/clef-flash-Q4_K_S.gguf
export LLB_DUAL_FASTPF=0
llb-serve --port 19021 --api-key "$BEND_API_KEY" \
  --engine /path/to/sys1mac \
  --whisper /path/to/whisper-server --whisper-model /path/to/ggml-base.en-q5_1.bin \
  --whisper-port 18137 \
  -- --gpu 12GB --threads 8 -- /path/to/sidecar 0 --steps 8192
```

The arguments after the first `--` go to `sys1mac`. `/path/to/sidecar` is
the sidecar directory. Make it from the GGUF with both tools:

```sh
llb-tok extract MODEL.gguf /path/to/sidecar        # tokenizer.bin, chat.flags
gguf2pack MODEL.gguf --sidecar-only /path/to/sidecar  # manifest.json, table.bin
```

The engine reads the weights from `LLB_GGUF` directly.
`LLB_DUAL_FASTPF=0` selects exact per-token prefill, because the fast prefill
tier cannot serve this model. To list all options, run `llb-serve` with no
arguments.

Start the `speak` server:

```sh
cd server/speak
TTS_EP=cpu TTS_MODEL=model.onnx TTS_THREADS=16 .venv/bin/python -I serve.py
```

It listens on `127.0.0.1:19022` (`TTS_PORT`). `GET /health` reports the model
and the execution provider. `run.sh` starts it, runs `bench.py`, and stops it.

## Server source

- `server/src/`, `server/build.sh`, `server/build-engine.sh`,
  `server/patches/`: `github.com/phenomenon0/bend-app`, branch `main`,
  commit `b47d73d419c4b91e4524a52bd3bb13b9b91d5a15` (2026-10-26), directory
  `desktop/engine/metal/`. The last code change there is `2d3ffe0e` (KEVGATE3).
  Compared with the earlier import, this source adds:
  - `src/kevhead.c`, `src/kevhead.h`: the Kev-4B decide head. `dualm.c`
    probes the GGUF for the clef head first, then the Kev head
    (`<arch>.decision.type` `"kev"` and `cls.output.{weight,bias}`), and
    otherwise launches headless (generate only).
  - clef-pick: a rejected clef `decide` (for example an unknown question type)
    is an HTTP 400 error line, not an engine exit.
  - `patches/apply_f27mm32.py` (CHUNKF-F32): an F32 arm for the fast chunk's
    GEMM, so clef 9B's F32 `ssm_beta`/`ssm_alpha` run on the fast prefill
    tier. `NOF32MM=1` skips it.
  - `patches/apply_chunkarea.py`: the fast chunk area is reserved whatever the
    first pass was. `NOCHUNKAREA=1` skips it.

  It drops the engine sampler of the earlier import
  (`patches/apply_sampler.py`, `src/llb_sampler*.h`) and `serve --canon`.
  Those came from `04ca5f0fc550023e2f303c9aa4d14cf77ed807dd` (tag
  `ship/mac/xo-web/mac-engine-penalty`), which is not an ancestor of `main`.
  On `main`, the fast GEMV tier is off unless `LLB_FASTGEMV=1`, and the fast
  prefill tier is on unless `LLB_DUAL_FASTPF=0`.
- Receipts: the `chat` receipt names `04ca5f0f`. The `see`, `hear`, and
  `decide` receipts name `d2b2a5818b7c5876848148909f59fecd30c6f7be` (tag
  `ship/mac/xo-web/metal-lead`). None was rerun on `b47d73d4`.
- `server/see/detect_cli.py`: the copy deployed on the test Mac. It is not
  in `bend-app`.
- `server/speak/`: the `speak` runner used for the receipt (`serve.py`,
  `run.sh`, `bench.py`). `requirements.txt` is the `pip freeze` of the graded
  environment. It is not in `bend-app`.

Changes from the source: default paths in `serve.c`, `speak_route.h`,
`serve.py`, `build.sh`, and `build-engine.sh` no longer point at the
developer's machine. The defaults are now `$HOME/.eureka/see/` for `see`
and `server/speak/dl/` for `speak`, and `EUREKA_ENGINE_SRC` is required.

## Verified

On 2026-10-26, on a Mac Studio (M3 Ultra, macOS 27), from this repository's
sources plus the fixes above: `./build.sh` emitted `c13b9e28` and linked
`sys1mac`; `llb-serve` built with no warnings; with clef-flash Q4_K_S, `chat`
returned 200 at 84.8 tok/s decode, `decide` 200, `see` 200, `hear` 200
(whisper-server, `ggml-base.en-q5_1.bin`), and `speak` 200 (24 kHz WAV).
This check used the `server/` sources before the `b47d73d4` refresh.

## Gaps

- The engine is a vendored fork. It is not rebased onto the CUDA stack.
- `see` runs on the ONNX Runtime CPU provider, not on Metal.
- The graded `sys1mac` was built from emitted C `9f59cd2c` plus the
  `04ca5f0f` engine sampler, not from `./build.sh` in this directory. That
  sampler is not in this source, so request `sampling` knobs (for example
  `frequency_penalty`) do not act. Generation is greedy.
- `server/` sources are from `b47d73d4`; no receipt has been rerun on them.
- `server/build.sh` builds `gguf2pack` and `llb-tok` from the Mac engine
  tree. The same tools are in this repository as `tools/gguf2pack.c` and
  `tools/llb/llb-tok.c`.
- The `detect_cli.py` SHA-256 (`a45ac8bd…`) does not match the `cli_sha256`
  in the `see` receipt (`62be4a57…`). It does match the copy deployed on the
  Mac.
