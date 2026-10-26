# llb

`llb` runs the Eureka Engine on a GGUF model with one command. It is a POSIX sh script with two small C helpers. It needs no Python. It runs on Linux with an NVIDIA GPU. A macOS adapter is written but has not been run. For the adapter, see `docs/guides/macos-cli-adapter.md` in this repository.

## System requirements

| Requirement | Notes |
|---|---|
| Operating system and GPU | Linux with an NVIDIA GPU. |
| C compiler | `cc`. |
| NVIDIA driver | `nvidia-smi`. |
| Libraries | `libnvrtc.so.13` and `libcuda.so.1`. `llb doctor` checks them. |
| coreutils | `stdbuf`. |
| curl | Needed for `llb pull` only. |
| Writable install directory | The engine writes its kernel cache to the install directory on first use. |
| GPU memory | The GPU must have room for the model. `llb` refuses with one line when it does not. |

## Install

1. Build the engine. This builds `build/llb-engine` and is needed once.

   ```sh
   ./build.sh
   ```

2. Install `llb`. The install script copies `llb`, its `.conf` files, `llb-tok`, `gguf2pack`, and the engine to `DEST`. This example uses `~/.local/llb` as `DEST`.

   ```sh
   tools/llb/install.sh ~/.local/llb
   ```

3. Link `llb` into a directory on your `PATH`.

   ```sh
   ln -s ~/.local/llb/llb ~/.local/bin/llb
   ```

4. Check the machine. `llb doctor` prints PASS, WARN, or FAIL for each check, each with a fix. The exit code is the number of FAIL results.

   ```sh
   llb doctor
   ```

## Run a model

1. Optional: download a model. `llb pull` shows the size first, can resume, and asks before it downloads anything over 5 GB. Use `-y` to skip the question.

   ```sh
   llb pull 9b
   ```

2. Run a prompt.

   ```sh
   llb run 27b "The capital of France is" -n 16
   ```

   To read the prompt from standard input:

   ```sh
   echo "Explain RAID 5 in two lines." | llb run 9b
   ```

   To read the prompt from a file:

   ```sh
   llb run 27b -f prompt.txt
   ```

3. Optional: start a chat. `llb chat` uses ChatML and keeps the history. It reloads the model each turn. For `27b`, each turn takes about 17 s.

   ```sh
   llb chat 27b
   ```

`llb` writes only the generated text to stdout, as it streams. It writes status to stderr:

- A loading line.
- A one-time note while the GPU program compiles. Compilation takes about 70 s, once per engine build.
- A summary line with the prompt and generation speed, the load time, the first-token time, and the reason that generation stopped.

Generation stops at the end-of-text token or after `-n` tokens. The engine exits when `llb` exits.

Decoding is greedy only. The same prompt gives the same text. There is no `--temp` option.

### Models

| Alias | File | Notes |
|---|---|---|
| `27b` | `Qwen3.8-27B-Q4_K_M.gguf` | Fast prefill by default. Use `--exact` for token-by-token prefill. Fast prefill (27B, industry-class) passed Verdict Gates v2 on 2026-10-16: top-1 agreement with exact prefill is 98.8 %, compared with llama.cpp's own 98.9 %. The KL ratio is 0.84. `--spec` is opt-in. |
| `9b` | `Qwen3.5-9B-Q4_K_M.gguf` | Exact prefill only. Download with `llb pull 9b`. |
| `9b-q8` | `Qwen3.5-9B-Q8_0.gguf` | Exact prefill only. Download with `llb pull 9b-q8`. |

`llb` matches an alias by the GGUF header (architecture, layers, file type, and size), not by the file name. It searches these locations in order:

1. `$LLB_MODELS`
2. `~/models`
3. The current directory
4. `~/.cache/huggingface/hub`

Any other qwen35 `.gguf` path runs with plain settings. The engine runs the qwen35 architecture only.

## Commands and options

| Command | Description |
|---|---|
| `llb run` | Generate text from a prompt. |
| `llb chat` | Chat with ChatML. Reloads the model each turn. |
| `llb models` | List known models and where they were found. |
| `llb bench 27b` | Benchmark a model. |
| `llb tok enc 27b "hello"` | Encode text to token ids. |
| `llb tok dec 27b 14556` | Decode token ids to text. |
| `llb pull 9b` | Download a model. |
| `llb prep` | Build the model cache. |
| `llb doctor` | Check the machine. |

`llb run`, `llb chat`, and `llb bench` accept these options:

| Option | Description |
|---|---|
| `-n N` | Number of tokens to generate. Default: 256. |
| `-f FILE` | Read the prompt from `FILE`. |
| `--exact` | Exact prefill, token by token. |
| `--fast` | Fast prefill. `27b` only. This is the `27b` default. |
| `--spec` | Speculative decoding with the NextN drafter. `27b` only. Gives the same tokens. Never the default. |
| `--ids` | The prompt is token ids. Prints ids. |
| `--dry-run` | Print the commands and run nothing. |
| `-v` | Show the engine's own output. |
| `-- ARGS` | Pass `ARGS` to the engine. |

## Files

| File | Description |
|---|---|
| `models.conf` | Per-model settings: alias, header facts, GPU span, engine environment, gated tiers, and Hugging Face repository. |
| `defaults.conf` | Defaults: `-n`, the prefill tier, and the search directories. |
| `${LLB_HOME:-~/.cache/llb}/<size>-<mtime>-<hash>/` | The model cache. It holds the tokenizer, chat flags, and the engine's sidecar (`manifest.json`, `table.bin`). `llb prep` or the first `run` builds it. |
| `last.log` | The ids and engine output of the last run. |

## Limitations

- A process runs one model. `chat` reloads the model every turn. In a 5-turn test, the first token came after 16.8-18.0 s in each turn. For the cost of a persistent mode, see `reports/cli1-final.md` in this repository.
- The GPU must have room for the model. `llb` refuses with one line when it does not.
- `llb-tok` is the `llb` helper (`tools/llb/llb-tok.c`). The `llb-tok` of the 0.1.0 bundle (`tools/qwen35_toktest.c`) is now named `llb-toktest`.
- Prompt and chat text is plain text. If you type `<|im_end|>` in a message, it stays text. Only the template's markers are special tokens.
- Windows is not supported. For the cost of Windows support, see `reports/cli1-final.md` in this repository.
- The macOS adapter has not been run.

## License

The repository has no LICENSE file.
