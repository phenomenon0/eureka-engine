# @@NAME@@

Eureka Engine is an inference engine for the Qwen3.5 model family. The engine is written in Bend and compiled to C and CUDA. This bundle runs Qwen3.8-27B Q4_K_M on one NVIDIA GPU through `llb`, a front end that runs a model with one command. The bundle is built from engine commit `@@COMMIT@@` (branch `pkg2`) with Bend compiler `240ecef4b52a1efcc2151df689cca9a2a769aac6` plus 57 patches. The bundle needs no Python.

## System requirements

The following sentence describes the static instruction classification of the bundled binaries (Q1b):

our binaries run on any x86-64 CPU; NVIDIA's NVRTC library is bundled as shipped and was not audited for old CPUs

The Q1b classification is a static scan. It is not a test on old CPU hardware. The original strict gate, which scans every ELF file (Q1), still fails. The PKG3 receipts count compatible hint encodings and third-party instructions separately.

| Requirement | Value | Evidence |
|---|---|---|
| GPU memory | A 24 GB card. Peak use is 19.8 GB with default settings, 19.0 GB with `--exact`, and 21.4 GB with default settings plus `--spec` (N ≤ 3,011). | Measured on an RTX 3090. |
| Driver, `cuda` tarball | A driver that supports CUDA 13.0: R580 or newer. NVIDIA, [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), Table 3 (minor version compatibility, CUDA 13.x: >= 580) and Table 4 (CUDA 13.0 GA: >= 580). The engine uses CUDA 13 driver entry points (`cuMemAdvise_v2`, `cuMemPrefetchAsync_v2`). | Believed. Not run on drivers older than 580.119.02. |
| Driver, `cuda12` tarball | A driver that supports CUDA 12.2: R535 or newer (Table 4: CUDA 12.2 GA >= 535.54.03). The engine uses the same entry points, which first appear in CUDA 12.2. The minor-version floor for CUDA 12.x (>= 525, Table 3) is not enough for them. | Believed. Only run on driver 580.119.02. |
| Operating system | x86-64 Linux with glibc 2.34 or newer. The newest symbol version that any binary uses is `GLIBC_2.34`. You also need a POSIX `sh` and coreutils `stdbuf`. | Measured on Fedora 43 (glibc 2.42). `llb version`, `llb doctor`, and `llb-probe` (no GPU) were run in `ubuntu:22.04` and `debian:12` containers. |
| Host RAM | Runs inside a 12 GB cgroup. | Measured. |

You do not need a CUDA toolkit. You need only the NVIDIA driver (`libcuda.so.1`). NVRTC is bundled as `libnvrtc.so.N` and `libnvrtc-builtins.so.N.M`. The engine finds these files through its DT_RPATH `$ORIGIN`.

NVIDIA's CUDA EULA, Attachment A, lists the NVRTC files as distributable with applications: "Component / NVIDIA Runtime Compilation Library and Header / ... / Linux / libnvrtc.so, libnvrtc-builtins.so, libnvrtc_static.a, libnvrtx-builtins_static.a" (EULA 2.6). Section 2.3 states that they "may be copied and redistributed for use in accordance with this Agreement, provided that the object code files are not modified in any way". This bundle ships them unmodified.

### Supported GPUs

| Compute capability | Status | Evidence |
|---|---|---|
| sm_86 (RTX 30xx, A-series Ampere) | Runs on an RTX 3090. | Both tarballs give the same tokens and logits. |
| sm_80, sm_89, sm_90, sm_100, sm_120 | Compiles only. The kernels compile with NVRTC 13.1 and NVRTC 12.9 (the NVRTC of each tarball). They have never been run. | Compile-only check. No GPU of these kinds was available. |
| sm_75 and older | Refused. | NVRTC reports `Feature 'cp.async' requires .target sm_80 or higher`. `llb-probe` fails the check. |

The engine sizes one kernel parameter, `CUBE_LOG`, from the GPU's L2 cache. The value is 6 on the 6 MB L2 cache of an RTX 3090, and 5 to 7 on other cards. On the RTX 3090, kernels built with `CUBE_LOG` forced to 5, 6, and 7 gave the same tokens and the same logits on the standard prompts. This was checked on one card, not across cards. A different GPU can still give different results.

### Choose a tarball

Two builds of the same engine source are available. The file `llb-engine.c` is identical in both.

| Tarball | Bundled NVRTC | Driver requirement | Notes |
|---|---|---|---|
| `llb-linux-x86_64-cuda-@@VERSION@@` | NVRTC 13.1 | Supports CUDA 13.0 (R580+). | Use this tarball if your driver supports it. |
| `llb-linux-x86_64-cuda12-@@VERSION@@` | NVRTC 12.9 | Supports CUDA 12.2 (R535+). | Fast prefill measured 3.1% slower: 1,270.8 vs 1,312.1 tok/s at 1,009 prompt tokens, median of 3 interleaved rounds. Decode is the same: 25.98 vs 25.97 ms/step. Use this tarball only if you cannot update the driver. |

`llb doctor` reports which tarball the machine can run. `llb run` calls `llb-probe` first, which makes the same check.

### Windows and WSL2

Windows and WSL2 are not supported. The engine needs concurrent managed memory access, where the CPU and the GPU use the same allocation. NVIDIA's [WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications) (5.1 Known Limitations) states: "Concurrent CPU/GPU access is not supported", "Full Managed Memory Support is not available on Windows native and therefore WSL 2 will not support it for the foreseeable future." `llb-probe` checks that device attribute and fails with that reason.

## Install

1. Unpack the tarball and change to its directory.

   ```sh
   tar xzf @@NAME@@.tar.gz && cd @@NAME@@
   ```

2. Verify the file checksums.

   ```sh
   sha256sum -c SHA256SUMS.files
   ```

3. Optional: link `llb` into a directory on your `PATH`. `llb` finds every other file next to itself.

   ```sh
   ln -s "$PWD/llb" ~/.local/bin/llb
   ```

4. Check the machine. `llb doctor` prints PASS, WARN, or FAIL for each check, each with a fix. The exit code is the number of FAIL results.

   ```sh
   llb doctor
   ```

Keep the directory writable. The first run compiles the GPU kernels into `llb-engine.gpu` next to the engine. This takes about 70 s and happens once.

## Run a model

1. Get the model file `Qwen3.8-27B-Q4_K_M.gguf`. The file is 18.97 GB, with sha256 `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747`. Its download source is not recorded.

2. Put the file in one of these directories: `$LLB_MODELS`, `~/models`, the current directory, or `~/.cache/huggingface/hub`. Alternatively, give the file path instead of `27b` in the commands that follow.

3. Run a prompt.

   ```sh
   llb run 27b "The capital of France is" -n 16
   ```

   To read the prompt from standard input:

   ```sh
   echo "Explain RAID 5 in two lines." | llb run 27b
   ```

4. Optional: start a chat. `llb chat` uses ChatML and keeps the history. It reloads the model each turn, so each turn takes about 17 s to the first token.

   ```sh
   llb chat 27b
   ```

`llb` writes only the generated text to stdout, as it streams. It writes a loading line and a summary to stderr. The summary gives the speeds, the load time, and the reason that generation stopped.

The first run of a model builds its cache in `${LLB_HOME:-~/.cache/llb}`. The cache holds the tokenizer and a small sidecar, and takes seconds to build.

Decoding is greedy. The same prompt gives the same text.

Prompt text is plain text. If a prompt or chat message contains `<|im_end|>` or `<|im_start|>`, the CLI tokenizes it as those characters, not as control tokens. A message therefore cannot fake a chat turn. This matches llama.cpp with `parse_special=false`. Only the chat template's own markers become special tokens.

The tokenizer has the following measured agreement with llama.cpp:

- `llb tok enc` matched llama.cpp `/tokenize` (`parse_special:false`) on 37/37 corpus strings and 116/116 held-out snippets.
- Chat ids matched llama.cpp's template and tokenizer on 9/9 specials-injection conversations and 11/11 plain conversations.

## Commands and options

`llb help` and `llb help run` list all commands.

| Command | Description |
|---|---|
| `llb run` | Generate text from a prompt. |
| `llb chat` | Chat with ChatML. Reloads the model each turn. |
| `llb bench` | Benchmark a model. |
| `llb models` | List known models. |
| `llb prep` | Build the model cache. |
| `llb tok enc\|dec` | Encode text to token ids, or decode token ids to text. |
| `llb doctor` | Check the machine. |
| `llb version` | Print the version. |
| `llb help` | List commands. |

`llb run`, `llb chat`, and `llb bench` accept these options:

| Option | Description |
|---|---|
| `-n N` | Number of tokens to generate. Default: 256. |
| `-f FILE` | Read the prompt from `FILE`. |
| `--exact` | Exact prefill: feed the prompt one token at a time. |
| `--fast` | Fast prefill. |
| `--spec` | Speculative decoding. |
| `--ids` | The prompt and the output are token ids. |
| `--dry-run` | Print the commands. |
| `-v` | Show the engine output. |
| `-- ARGS` | Pass `ARGS` to the engine. |

### Settings and measurements

The numbers in this table were re-timed for this release on one RTX 3090 (driver 580.119.02) with the engine from the `cuda` tarball. Each number is the median of 3 rounds, with [min..max]. No llama.cpp comparison was re-timed for this release, so no comparison is claimed.

| Setting | Default | Description | Measurement |
|---|---|---|---|
| int8 dp4a decode (`LLB_Q4_DP4A=9`) | On for `27b`. | Decode tier. | 25.97 [25.94..26.00] ms per decoded token (38.5 tok/s), mean over three 64-token prompts. |
| Fast prefill (`--fast`, `LLB_PREFILL=fast`) | On for `27b`. `--exact` turns it off. | Chunked layer-major prefill on int8 tensor cores. | 1,312 [1,305..1,316] tok/s at 1,009 prompt tokens, computed as prompt tokens divided by the time to the first generated token. The result is not bit-equal to token-by-token prefill. It is gated against llama.cpp's own logits (KL ratio 1.15, limit 1.2). `--exact` feeds the prompt one token at a time, at decode speed. |
| Speculative decoding (`--spec`, `LLB_SPEC`, `LLB_MTP=FILE`) | Off. | Greedy speculative decoding with a NextN (MTP) drafter. | Not re-timed for this release. No speed is claimed. It gives the same tokens as plain greedy decoding (0.1.1 check). It needs a 438 MB drafter file that this bundle does not include. The drafter is made in the source repository from an MTP GGUF whose source is not recorded. Peak memory is 21.4 GB, which is only 25-65 MiB under the 22.5 GB line used as the memory limit. |

### Run the engine with token ids

You can run the engine and the tokenizer directly, without `llb`.

1. Build the sidecar for the GGUF file. This takes about 1 s and writes `manifest.json` and `table.bin`.

   ```sh
   ./gguf2pack Qwen3.8-27B-Q4_K_M.gguf --sidecar-only side-q27
   ```

2. Run the engine on a list of token ids.

   ```sh
   ./run.sh --gguf Qwen3.8-27B-Q4_K_M.gguf --gpu 40GB --threads 16 -- side-q27 760 6511 314 9338 369 --steps 64
   ```

3. Optional: extract `tokenizer.bin` from the GGUF file, then encode and decode text.

   ```sh
   ./llb-tok extract Qwen3.8-27B-Q4_K_M.gguf tok27
   ./llb-tok enc tok27 "The capital of France is"
   ./llb-tok dec tok27 760 6511 314 9338 369
   ```

   The `enc` command prints `760 6511 314 9338 369`. The `dec` command prints `The capital of France is`.

`run.sh` prints the generated ids, then `wall_ms`. It sets `LLB_Q4_DP4A=9` and `LLB_PREFILL=fast`. Use `--prefill exact` for exact prefill, and `--spec K` for speculative decoding.

## Files in this bundle

| File | Description |
|---|---|
| `llb` | The front end (POSIX sh). |
| `defaults.conf`, `models.conf` | Settings for the front end. |
| `llb-tok` | The C helper for the front end: GGUF info, tokenizer, and chat template. |
| `llb-engine` | The engine. Its NVRTC kernel source is `llb-engine.c` itself, embedded with `#embed __FILE__`. |
| `libnvrtc.so.N`, `libnvrtc-builtins.so.N.M` | NVIDIA's runtime compiler. The engine loads it from this directory. |
| `llb-probe` | The GPU and driver check. `llb doctor` runs it, and `llb run` runs it before every run. |
| `gguf2pack` | Makes the 2-file sidecar that the GGUF path needs, or a full pack. |
| `run.sh` | Launcher for the raw ids workflow. |
| `llb-toktest` | The tokenizer test tool for the ids workflow. In 0.1.0 it was named `llb-tok`. |
| `llb-engine.c`, `gguf2pack.c`, `tools/`, `build.sh` | Source. `./build.sh` rebuilds the five binaries. It needs clang ≥ 19, CUDA headers, and libnvrtc. CUDA 12 headers build the `cuda12` variant. |
| `SHA256SUMS.files` | The sha256 of each file. |

## Limitations

- Only sm_86 (RTX 3090) has been run. See [Supported GPUs](#supported-gpus). The 27B model needs a 24 GB card.
- Prompts of up to 3,011 tokens, plus 64 generated tokens, have been tested.
- An 8,192-token prompt does not run with `--gpu 40GB` with any flag set. The error is Bend heap span out of memory. With `--gpu 80GB`, it needs more than 12 GB of host RAM. With fast prefill and `--spec`, the card reached 24,109 MiB.
- Decoding is greedy only.
- A process runs one model. `chat` reloads the model every turn.

## License

License: Apache License 2.0.

The bundle has no LICENSE file. The Bend compiler commit `240ecef4` is on no public ref, and the source repository has no public remote.
