# Run on CUDA

This guide builds and runs the Eureka Engine on Linux with an NVIDIA GPU.
It covers the 27B Q4_K_M model, which is the CUDA release target (`unify1`).
The release report is
[`reports/llb-unify1-final.md`](../../reports/llb-unify1-final.md).

For the other backends, see [Backends](../backends.md).

## Features and defaults

The launcher defaults come from the `unify1` measurements. Conditions: one
RTX 3090, three rounds interleaved with llama.cpp in the same rounds. Ratios
are the median and the range [min..max] against llama.cpp.

| Feature | Flag | Result | Status |
|---|---|---|---|
| Decode | `LLB_Q4_DP4A=9` | 38.8 tok/s. 1.011x [1.009..1.012] of llama.cpp `tg128`. | Default |
| Fast prefill | `--prefill fast` | See [Fast prefill results](#fast-prefill-results). | Default. `--prefill exact` turns it off. |
| Direct GGUF load | `--gguf` | Same tokens and logits as the pack path. Cold load 17.9 s, against 44.7 s from a pack. | Recommended |
| Speculative decoding | `--spec 2` | 65.4 tok/s. Same tokens as plain greedy decoding. | Opt-in. Needs a drafter file that is not distributed. Uses the most memory of any flag combination. |

### Fast prefill results

| Prompt tokens | Throughput | Ratio to llama.cpp |
|---|---|---|
| 1,009 | 1,295 tok/s | 0.92x of the steady llama.cpp result (1,410.6 tok/s, `llama-bench -mmp 1 -r 5`). 0.988x [0.986..0.991] of the in-round `-r 3` means, each of which includes one slow repetition. |
| 3,011 | 1,307 tok/s | 0.953x [0.953..0.955] |

### 9B model

The 9B model (Q8_0 and Q4_K_M) builds and decodes from the same binary with
all flags off. It was not re-timed in `unify1`. Fast prefill does not support
the 9B model.

## Before you begin

You need the following:

| Requirement | Details |
|---|---|
| Operating system | Linux x86-64. |
| GPU | NVIDIA GPU with CUDA 13.x. Tested: RTX 3090, sm_86, CUDA 13.1.80. The build needs only the CUDA headers and libraries, not a GPU. |
| Tools | `bun` 1.3.4, `clang` 21, `patch`, `git`, `tar`, `sha256sum`. |
| Python | Not required. The `tools/*.py` pack tools are kept only as a reference oracle. |
| OpenBLAS | Required (`-lopenblas`). The emitted C calls `dual_next` and `dual_done`, which are in `tools/sys1dual/dual.c` and the CLEF-3 sources. `build.sh` builds and links these objects unless `LLB_CC_EXTRA` is set. See `tools/sys1dual/build.sh`. |
| Bend compiler | A git checkout of the Bend compiler that contains commit `240ecef4b52a1efcc2151df689cca9a2a769aac6`. See [Pinned compiler](#pinned-compiler). |
| Model | `Qwen3.8-27B-Q4_K_M.gguf`, 18.97 GB, sha256 `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747`. The download source is not recorded in this repository. |

### Pinned compiler

Commit `240ecef4` is not on any public branch of `bendlang/bend`. You need the
owner's fork or a bundle.

The commit object is lost. Its `bend2/` directory is kept in this repository on
the branch `pins/bend-240ecef4`. See `PIN.md` on that branch. If `LLB_BEND`
does not point to a checkout that contains `240ecef4`, `build.sh` uses this
branch, local or `origin/`, and checks its `bend2` tree ID against `BEND_TREE`.
To fetch the branch:

```sh
git fetch origin pins/bend-240ecef4:pins/bend-240ecef4
```

## Build the engine

1. Clone the repository:

   ```sh
   git clone https://github.com/phenomenon0/eureka-engine
   cd eureka-engine
   ```

1. Set the build variables:

   ```sh
   export LLB_BEND=BEND_CHECKOUT
   export LLB_OUT=$PWD/build
   export CUDA_HOME=/usr/local/cuda
   ```

   Replace `BEND_CHECKOUT` with the path to a Bend checkout that contains
   `240ecef4`. `LLB_OUT` and `CUDA_HOME` are optional. `LLB_OUT` must be on a
   POSIX file system, because `patch` needs `chmod`. Do not use NTFS or exFAT.

1. Run the build:

   ```sh
   ./build.sh
   ```

`build.sh` does the following:

1. Exports the pinned compiler.
1. Applies the compiler patch stack listed in `tools/build_gpu.sh`, with
   `--fuzz=0`. It stops if a patch does not apply exactly.
1. Emits `llb-engine.c` from `engine_qwen35_gpu.bend`.
1. Compiles `llb-engine`.
1. Prints the sha256 of the commit, the patch stack, the patched compiler, the
   C source, and the binary.
1. Writes `gguf2pack` and `llb-probe` to `LLB_OUT`. `llb doctor` and the
   capacity table use `llb-probe` to check the GPU.

### Build variables

| Variable | Description |
|---|---|
| `LLB_ROOT` | Path to this repository. |
| `LLB_BEND` | Path to the Bend checkout. |
| `LLB_OUT` | Output directory. Must be a POSIX file system. |
| `CUDA_HOME` | CUDA installation directory. |
| `CC` | C compiler. |
| `LLB_CC_EXTRA` | Extra objects to link. If set, `build.sh` does not build the default request objects. |

## Prepare the model

The engine reads weights from a pack directory or directly from the GGUF
file. Direct GGUF load is recommended for the GPU engine. The CPU twin and the
KAT tools need a pack.

### Option 1: Load the GGUF file directly

1. Write the sidecar files. This takes about 1 second and writes two small
   files:

   ```sh
   export LLB_MODELS=MODELS_DIR
   $LLB_OUT/gguf2pack $LLB_MODELS/Qwen3.8-27B-Q4_K_M.gguf --sidecar-only $LLB_MODELS/side-q27
   ```

1. Run the engine with `LLB_GGUF` set:

   ```sh
   LLB_GGUF=$LLB_MODELS/Qwen3.8-27B-Q4_K_M.gguf tools/run27b.sh --gpu 40GB --threads 16 -- $LLB_MODELS/side-q27 760 6511 314 --steps 8
   ```

With `LLB_GGUF=PATH`, the GPU engine reads weights from the GGUF file. The
reader is `host/llb_gguf.c`, a Bend user effect declared in `gguf.bend`. It
uses `pread` with 8 threads, and converts Q8_0 tensors to the pack row-split
layout as it reads them.

The positional directory argument is then a sidecar directory. It contains
`manifest.json` and `table.bin`, which `gguf2pack --sidecar-only` writes from
the GGUF header. Add `--sha` to include per-tensor sha256 values; this reads the
file once.

Results:

- Device bytes, token streams, and logits are identical to the pack path for
  the 9B Q8_0, 9B Q4_K_M, and 27B models
  ([`docs/guides/load-gguf-directly.md`](load-gguf-directly.md)).
- The 27B model loads cold in 17.9 s [16.7..18.0], against 44.7 s
  [44.2..45.2] from a pack (`unify1`, three rounds). No 38 GB pack is needed on
  disk.

To check every loaded tensor, set `LLB_GGUF_VERIFY=PACK/manifest.json`. The
engine reads each tensor back from the device and checks its sha256. If
`LLB_GGUF` is not set, the engine uses the pack path.

### Option 2: Make a pack

1. Convert the GGUF file to a pack:

   ```sh
   export LLB_MODELS=MODELS_DIR
   $LLB_OUT/gguf2pack $LLB_MODELS/Qwen3.8-27B-Q4_K_M.gguf $LLB_MODELS/pack-q27a
   ```

`gguf2pack` is a C program, built by `build.sh` from `tools/gguf2pack.c`. It
reads the GGUF file once and writes `pack.bin`, `layers/blk.N.bin`,
`manifest.json`, `table.bin`, and `index.tsv`. The output is byte-identical to
the output of the Python tools: 851 tensors, `pack_sha256
90f5f92cac810765e2a05d5009e4c231f94f56be920621ccafef7eaffa7a0888`.

For the 27B model, the conversion took 12.6 minutes at about 25 MB/s with
36 MB peak RSS. The Python tools took 1 hour 47 minutes and 7.5 GB. The pack
needs about 38 GB of disk.

| Option | Description |
|---|---|
| `--verify-against PACK_DIR` | Writes nothing. Checks an existing pack against the GGUF file: per-tensor sha256, and the bytes of `manifest.json`, `index.tsv`, and `table.bin`. |
| `--probe` | Prints the tensor table. |
| `--head-only` | Leaves the block ranges of `pack.bin` as holes. Uses about half the disk. The GPU engine does not read these ranges. The CPU twin and the KAT tools do. |

## Run the 27B model

1. Run the engine from a GGUF file or from a pack:

   ```sh
   tools/run27b.sh --gguf $LLB_MODELS/Qwen3.8-27B-Q4_K_M.gguf --gpu 40GB --threads 16 -- $LLB_MODELS/side-q27 760 6511 314 9338 369 --steps 64
   tools/run27b.sh --gpu 40GB --threads 16 -- $LLB_MODELS/pack-q27a 760 6511 314 9338 369 --steps 64
   ```

The prompt is a list of token IDs. `run27b.sh` runs `$LLB_OUT/llb-engine`. To
use another binary, set `ENG`. The script sets `LLB_Q4_DP4A=9` and
`LLB_PREFILL=fast` unless they are already set.

The first run builds the NVRTC kernel cache `llb-engine.gpu` next to the
binary. This takes about a minute.

Output on stdout: the generated token IDs, then `wall_ms`.

### Run options

| Option | Variable | Description |
|---|---|---|
| `--gguf FILE` | `LLB_GGUF` | Load weights from the GGUF file. See [Option 1](#option-1-load-the-gguf-file-directly). |
| `--prefill fast\|exact` | `LLB_PREFILL` | Prefill mode. Any value other than `fast` turns fast prefill off. See [Fast prefill](#fast-prefill). |
| `--spec K` | `LLB_SPEC=K`, `LLB_MTP=FILE` | Greedy speculative decoding. See [Speculative decoding](#speculative-decoding). |
| None | `LLB_Q4_DP4A=0` | Exact tier with fixed bytes. 31.36 ms/step in `q27c`. |

With all flags off, the engine reproduces the `q27c` and `integ1` tokens and
logits. This is an internal regression check (R0).

### Fast prefill

Fast prefill is a chunked, layer-major prefill on int8 tensor cores. Its output
is not bit-identical to the token-by-token prefill. It is gated against
llama.cpp under mode 9 ([`reports/unify1-gates.md`](../../reports/unify1-gates.md),
R2):

- KL(llama‖fast) / KL(llama‖exact) = 1.150. The limit is 1.2.
- 3 of 20 prompts diverge within 64 tokens. llama.cpp diverges on 9 of 20
  prompts between its own prefill and decode paths.
- Peak memory is 19.8 GB, against 19.0 GB for exact prefill.

### Speculative decoding

`--spec K` runs greedy speculative decoding with the NextN (MTP) drafter. It
produces the same tokens as plain greedy decoding after the same prefill (R5,
R6).

- The drafter file is set by `LLB_MTP` (438 MB). Build it with
  [`tools/mtp27/make_drafter.sh`](../../tools/mtp27/README.md).
- `tools/mtp27/mtp_pack.py` makes the drafter file from an MTP GGUF file. The
  source of that GGUF file is not recorded.
- If `LLB_MTP` is not set, `LLB_SPEC` uses a prompt-lookup drafter.
- With fast prefill, peak memory is 21.4 GB. This is 25 to 65 MiB below the
  22.5 GB limit set for this project. For this reason the flag is opt-in.

## Convert text to token IDs

For text input and output, use the `llb` front end. For example,
`llb run 27b "..."` or `llb chat 27b`. See
[`release/README-llb.md`](../../release/README-llb.md).

To convert text by hand:

1. Build the tokenizer tool:

   ```sh
   cc -O2 -o $LLB_OUT/llb-tok tools/llb/llb-tok.c -lm
   ```

1. Extract the tokenizer from the GGUF file. This writes
   `tok27/tokenizer.bin`:

   ```sh
   $LLB_OUT/llb-tok extract $LLB_MODELS/Qwen3.8-27B-Q4_K_M.gguf $LLB_OUT/tok27
   ```

1. Encode and decode:

   ```sh
   $LLB_OUT/llb-tok enc $LLB_OUT/tok27 "The capital of France is"
   $LLB_OUT/llb-tok dec $LLB_OUT/tok27 760 6511 314 9338 369
   ```

   The `enc` command prints `760 6511 314 9338 369`.

| Command | Behavior |
|---|---|
| `llb-tok enc` | Treats the text as plain text. Control tokens such as `<\|im_end\|>` are not parsed. This matches llama.cpp with `parse_special=false`. |
| `llb-tok chat` | Applies the ChatML template from the GGUF file. Only the template's own markers become special tokens. |

Test results:

- `llb-tok` matched llama.cpp on 37 of 37 corpus strings and 116 of 116
  held-out snippets. It matched on 9 of 9 special-token injection chats and 11
  of 11 plain chats.
- The older tool `tools/qwen35_toktest.c`, built as `llb-toktest`, matched on
  313 of 313 recorded prompts and 1,580 of 1,580 gate strings
.

## Create a release bundle

1. Run the bundle script:

   ```sh
   tools/unify/bundle.sh VERSION $LLB_OUT
   ```

The script writes `release/llb-linux-x86_64-cuda-VERSION/`, a `.tar.gz` of that
directory, and `release/SHA256SUMS`. The bundle contains:

- `llb`, `defaults.conf`, `models.conf`, and `llb-tok`.
- The engine and its C source.
- `build.sh` and `run.sh`.
- `gguf2pack` and its C source.
- `llb-toktest`.
- `README.md`, made from [`release/README-cuda.md`](../../release/README-cuda.md).

The bundle contains no Python. After you unpack it, `llb` finds every file in
its own directory.

## Run the license check

```sh
make gpl-tts-check
```

The check fails if the shipping path links or imports GPL `espeak-ng`, the
`phonemizer` package, or Piper. Piper wheels embed `espeak-ng`. Put dev-only
exemptions in `tools/gpl_tts_allowlist.txt`. The rules and the scanned
directories are documented in `tools/check_no_gpl_tts.py`.

## Limitations

- Only sm_86 on an RTX 3090 (24 GB) is tested. The 27B model needs a 24 GB
  card. The minimum driver, CUDA version, and compute capability are not
  known. Tested: driver 580.119.02, CUDA 13.1.80.
- Prompts up to 3,011 tokens, plus 64 generated tokens, are tested. An
  8,192-token prompt does not run with `--gpu 40GB`, with any flag set: the
  Bend heap span runs out of memory. `integ1` has the same limit.
- The fast tier (`LLB_Q4_DP4A=9`) is not bit-identical to the exact tier.
  - It is gated on the 27B model.
  - It is gated on the 9B clef-flash Q4_K_S model on the sys1 path, which is
    the default in `tools/sys1dual/env.sh`. Quantization audit of 2026-10-16:
    speed +11.66 %, KL half PASS, decide battery G4 PASS.
  - It is not gated on the stock 9B model (`9b` in `models.conf`), the 4B
    model, or the 0.8B model.
  - Class: LABEL-EXACT under the tolerance amendment, signed 2026-10-17.
- Speculative decoding supports greedy decoding and the 27B model only.
- The Mac (Metal) build is not scripted in this repository.
  See [`backends/metal/README.md`](../../backends/metal/README.md).
- The pinned compiler is a fork plus the patch stack in `tools/build_gpu.sh`.
  The upstream Bend is about 113 commits ahead, and the patches do not
  apply to it.
