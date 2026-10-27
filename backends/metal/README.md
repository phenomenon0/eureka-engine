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

This builds `sys1mac` with the [any-shape](#any-shape) patches (emitted C
`00979448`). `LLB_METAL_BASE=1` builds the Mac product (`c13b9e28`).
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

## Any shape

The default build applies `cc/metal-anyshape-cc.patch`,
`engine/metal-anyshape-engine.patch` and `link/dual-anyshape.patch` after the
product patches (emitted C `00979448…`, `VISION=1` `85c8d3c4…`).
`LLB_METAL_BASE=1 ./build.sh` leaves them out and still reproduces
`c13b9e28` / `fbdefec7`. What they change:

- **Shapes.** The quant pack's manifest is checked structurally
  (`mf.dense`): any layer count up to 1,024, heads with a whole number of q
  heads per kv head, d_model and d_ff multiples of 256 (at most 8192 and
  17,408), SSM value heads a multiple of the key heads, vocabulary up to
  250,048. head_dim 256, SSM state 128, a 4-tap conv, a full-attention layer
  every fourth, rope 1e7 over 64 dims and eps 1e-6 are compiled into the
  kernels and are refused by name otherwise. The engine passes the head
  counts, vocabulary and layer count to the host (pass words 57–59, chunk
  words 52–54). The 9B, 27B and 4B keep every legacy dispatch; another shape
  takes the generic arms: attention over any kv-head count
  (`llb_m_attn6`, six-dispatch split), the slot copy for kv rows other than
  1024 floats, K-quant GEMV bodies at depth 8–13 in Q4_K / Q5_K / Q6_K, the
  padded rmsnorm tree at any width but 2048 / 4096, the fused step's head
  and embedding at its width and vocabulary. The fast prefill chunk runs a
  generic shape when its GQA is 4 or 6 over 2 or 4 kv heads (the 0.8B and 2B:
  4 over 2). The plain (non-fused) path stays the 9B's; a generic shape runs
  the fused step only. The Q8 pack and the 35B MoE keep the whitelist.
- **Context.** Up to 2^27 slots. The KV (kv rows of n_kv_heads × 256
  floats), the attention planes and the chunk area live in the engine's own
  Metal buffer, sized from the context and checked against
  `maxBufferLength`; its pages are committed as the slots fill. The fused
  step no longer allocates a Bend state of 2^(dT + 10) floats per layer.
  `--ctx C` sizes the KV for at least C slots. An argument `@FILE` expands to
  the file's words, so a prompt of ids can pass the OS argument limit. The
  `generate` request (`link/dual.c` via `link/dual-anyshape.patch`, and
  `server/src/dualm.c`) takes prompts up to the KV capacity. The 35B MoE arm
  stays at 2^17.

Run a new shape exactly as the 9B: make the sidecar with `gguf2pack
MODEL.gguf --sidecar-only DIR` and set `LLB_GGUF`.

Checked on the M3 Ultra, 2026-10-26 (receipts are not in this repository):

- 4B and 9B Q4_K_M against the base build: the same 64 greedy ids on three
  prompts, exact and default prefill (12/12), and byte-identical logits lines
  on a 20,027-token prompt.
- 0.8B and 2B Q4_K_M against llama.cpp `9d57ce45` (CPU, greedy, no
  template): exact and default prefill agree with each other on every
  prompt. Against llama.cpp, three of six 64-token runs match all 64 ids; the
  others split at near-ties (llama.cpp margins 0.435, 0.067, 0.073 logits).
  The 4B base build splits from llama.cpp the same way on the same prompts
  (margins 0.018, 0.055). Mean top-1 logit difference over the matching
  prefixes: 0.09–0.18 (0.8B, 2B), 0.05–0.09 (4B).
- Needle in a haystack (`violet-harbor-42` at 40% depth): found by the 0.8B,
  2B, 4B and 9B at 20,027 tokens, and by the 0.8B and 2B at 153,031 tokens
  (KV 2^18 slots; 189 s and 208 s with the default fast prefill).
- `--ctx 300000` on the 4B gives the same ids as the default capacity;
  `--ctx 2097152` (2^21) runs on the 0.8B; `--ctx 16777216` on the 0.8B is
  refused (385 GiB > the 58.3 GiB buffer limit).
- Decode: 0.8B about 5 ms a token, 2B 6, 4B 10, 9B 14 (both builds).

Not done: f16 KV; head_dim 128 or SSM state 64; rope base and eps as kernel
arguments; the fast chunk at other GQA ratios. The
`VISION=1` image rows are placed by the legacy reserve only, so a vision run
needs a legacy shape at 2^17 slots or fewer. The registered 35B pack's own arm
stays at 2^17; any MoE GGUF takes the general MoE path below, which has the
dense range.

## Any MoE

The default build also applies `cc/metal-moe-cc.patch` and
`engine/metal-moe-engine.patch` (emitted C `a20c8ee2…`). `LLB_METAL_ANY=1`
stops before them (the any-shape build, `00979448`); `VISION=1` does not take
them. They run any `qwen35moe` GGUF on the generic pass:

- **Sidecar.** `gguf2pack MODEL.gguf --sidecar-only DIR` takes a `qwen35moe`
  GGUF: `manifest.json` (config `"experts": "gguf"`, the expert and shared
  widths) and `table.bin`, whose rows also carry the router and shared gate
  (words 21, 22) and the MoE words (24–29: a magic, `expert_count`,
  `expert_used_count`, `expert_feed_forward_length`,
  `expert_shared_feed_forward_length`, a fused `ffn_gate_up_exps`). NextN
  (MTP) blocks are dropped. Run it as a dense model: `LLB_GGUF=MODEL.gguf`.
- **Shapes.** Any expert count, top-k, expert and shared width (multiples of
  32; of 256 for a K-quant row), separate or fused gate/up experts, and the
  dense trunk's range (layers, d_model, heads). The routing kernel loops over
  the experts and the top-k; the MoE scratch is its own buffer, sized from
  the four words. Experts: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. Trunk:
  Q8_0 and K-quant matrices, F32 vectors, an F32 router.
- **Experts.** They never enter the heap. The engine maps the GGUF read-only
  and gives each expert tensor as chunks of no-copy buffers, each within the
  device's largest buffer; the kernels reach a chunk through a table of GPU
  addresses (offsets are 64-bit). When the experts and the heap do not fit
  the GPU's working set (`recommendedMaxWorkingSetSize`), the engine streams
  instead: per layer it runs the routing, reads the k chosen ids, copies only
  those experts from the map into a staging buffer, and goes on. Both modes
  give the same bits. `LLB_MX_SPLIT=N` forces N experts a chunk and
  `LLB_MX_STREAM=1` forces streaming (tests). `LLB_MX_DUMP=FILE` writes each
  layer's FFN input, output and routing for the float64 oracle in
  `reports/metal-moe/`.
- **Arithmetic** (llama.cpp's order): router `ffn_gate_inp · x`, softmax over
  every expert, top-k (ties to the lowest id), the k weights renormalized,
  no weight scale; expert `down(silu(gate · x) * (up · x))`, summed in rank
  order; the shared expert's SwiGLU times `sigmoid(ffn_gate_inp_shexp · x)`,
  added last. Not byte-exact to anything: a numerics-changing path.
- **Context.** The dense path's: up to 2^27 slots, bounded by the largest
  buffer (`--ctx C`, `@FILE`). Prefill is exact, one token at a time: there
  is no MoE fast chunk, and `--fastpf` is ignored for these models.

Checked on the M3 Ultra, 2026-10-26 (receipts: `reports/metal-moe/`):

- **35B-A3B** (unsloth `Qwen3.5-35B-A3B-Q4_K_S.gguf`: 256 experts, top 8,
  expert and shared ff 512, Q4_K experts, Q8_0 trunk) against llama.cpp
  `e70802a01` on the CPU (greedy, no template, three prompts x 64 tokens):
  first token equal on 3/3; teacher-forced top-1 187/192. The five
  disagreements are at llama.cpp margins 0.03–0.10 logit, and llama.cpp's own
  Metal path disagrees with its CPU at the same five positions (187/192).
  Mean top-1 logit difference 0.13–0.21 (llama.cpp Metal vs CPU: 0.12–0.22).
- **MoE FFN oracle.** Every layer's FFN output against a float64 recomputation
  from the GGUF's bytes (`LLB_MX_DUMP`, `mx_ref.py`): the chosen experts equal
  on every record, relative error at most 7.1e-7 (35B, 120 records) and
  1.6e-6 (synthetic shapes). A broken router (`LLB_MX_MUT=no_renorm`)
  gives 0.88.
- **Synthetic shapes** (random weights, `gen_moe.py`): 128 experts / 8 used
  with expert ff 768 and shared ff 1024; 60 / 6 with 768 / 512; 40 / 5 with a
  fused `ffn_gate_up_exps`, 512 / 1536; Q4_K_M and Q8_0. Against llama.cpp on
  Metal: teacher-forced top-1 0.92–1.00. Against llama.cpp on the CPU the
  random networks are too chaotic to score (llama.cpp's own Metal path gets
  0.13–0.67 there, as does this engine).
- **Forced paths.** `LLB_MX_SPLIT` (3 or 16 experts a buffer, up to 1,920
  buffers) and `LLB_MX_STREAM=1` give the resident path's ids and logits on
  every model.
- **Dense regression.** 4B and 9B Q4_K_M: the same 64 greedy ids as the
  any-shape build on three prompts, exact and default prefill (12/12), and
  byte-identical logits lines.
- **Long context.** A needle (`PELICAN-7342` at 45% of `hay140k`, chat
  template) found by the 35B-A3B at 153,352 tokens (KV 2^18 slots; 14,908 s,
  exact prefill: about 22 ms a token at the start, 110 ms past 65k, 195 ms past
  131k, the generic attention's cost doubling with each power of two). With
  `--ctx 1048576` the first 16 ids equal the default capacity's; `--ctx
  16777216` is refused (642.19 GiB needed, the largest buffer is 58.32 GiB).
- **Speed.** 35B decode 21.6 ms a token at short context (resident experts;
  streamed about 15% slower); llama.cpp `e70802a01` on Metal: 86.3 tok/s
  (11.6 ms, `llama-bench -n 64`). Prefill is one token at a time.

Not done: a MoE fast prefill chunk; a 122B-A10B or 397B-A17B run (no file on
the test Mac; the split and streamed paths that such a model takes were
forced on small ones); BF16 or IQ-format trunk tensors; the served
`generate` route (`llb-serve`) on a MoE model.

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
