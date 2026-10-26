# macOS adapter for the llb CLI

This page describes how `llb` runs a model on macOS through the Metal runner `run.sh` of the M36 Mac release. The adapter is designed and written, but it has not been run. Nobody has run it on a Mac.

The adapter code is the `mac_cmd` function in [`tools/llb/llb`](../../tools/llb/llb). The Linux dry run of that path was checked with `LLB_OS=Darwin` and a stub runner. `--dry-run` prints the command.

## How the adapter works

On Darwin, `llb` keeps its front half unchanged and replaces only the engine command. `llb` detects Darwin with `uname -s`. For tests, `LLB_OS` overrides the result.

`llb` keeps these parts on macOS:

- Model lookup.
- `prep`, which builds the sidecar and `tokenizer.bin`.
- The C tokenizer and chat template (`llb-tok`).
- The stop and summary wrapper (`llb-tok run`).

`llb` skips these parts on macOS:

- The CUDA engine.
- `stdbuf`.
- The `nvidia-smi` check.

The Metal runner reads token ids from a file that `llb-tok` writes, so `llama-tokenize` is not needed.

## Mapping of llb options to run.sh

| llb | `run.sh` | Notes |
|---|---|---|
| `27b` | `--model 27b` | Dense. Needs a pack. |
| `9b` (Q4_K_M) | `--model 9b-q4km` | Dense. Needs a pack. |
| `9b-q8` | `--model 9b-q8` | Dense. Needs a pack. |
| `35b` | `--model 35b` | Not in [`models.conf`](../../tools/llb/models.conf), because the UNIFY1 CUDA build has no 35b. To use it, add a row. |
| Model GGUF path | `--gguf PATH` | Always passed. |
| `$LLB_HOME/<key>/pack` | `--pack DIR` | Built once by `gguf2pack GGUF DIR`. |
| Prompt or chat history | `--ids $LLB_HOME/<key>/ids` | The ids come from `llb-tok enc` or `llb-tok chat`. This replaces `--ids-text` and `--prompt-file F --tokenizer llama-tokenize --tokenizer-gguf GGUF`. |
| `-n N` | `--steps N` | |
| `--exact` | `--exactpf` | Passed only when you give it explicitly. |
| `--fast` | `--fastpf 512` | `27b` only, the same gate as on Linux. K=512 comes from the Mac release's `CHAIN.md`. |
| Neither `--exact` nor `--fast` | Nothing is passed. | The default of `run.sh` applies. According to `CHAIN.md`, that default is `--fastpf 512`. See item 3 in [Known issues](#known-issues). |
| `--spec` | Refused. | The message is "--spec is not available on the Mac". |
| `-- ARGS` | Appended. | Passed through. |
| `LLB_MAC_RUN` | Path to `run.sh`. | Default: `<llb dir>/run.sh`. |

## Disk space check

Metal has no GGUF loader, so dense models need a pack. A pack is about 2 times the size of the GGUF file. The Mac release's README gives the size of the 27B pack as about 38 GB.

Before `llb` builds a pack, it checks the free space in `$LLB_HOME`. It refuses to build the pack when the free space is below 2 times the pack size plus 6 GB. It takes the pack size as 2 times the GGUF size:

```text
need_MB = 4 * gguf_bytes / 2^20 + 6144
```

For `27b`, this is about 77 GB free. The refusal message states how much space is needed and how much is free, and suggests `LLB_HOME`.

`llb` builds the pack into `pack.tmp` and moves it into place only when `gguf2pack` succeeds. The build log is `<key>/pack.log`.

## Known issues

The following questions cannot be answered without a Mac.

1. The stdout format of `run.sh`. `llb-tok run` expects one decimal id per line and stops at `<|im_end|>` (248046). If `run.sh` prints text or prefixed lines, the wrapper needs a small filter, or `run.sh` needs a flag that prints one id per line.
2. Whether the `--ids FILE` option of `run.sh` accepts ids separated by spaces on one line, which is what `llb` writes, or needs one id per line.
3. The default prefill of `run.sh` when neither flag is given. On Linux, `llb` follows [`defaults.conf`](../../tools/llb/defaults.conf), which now selects fast prefill for `27b` and exact prefill for the other models. On the Mac, when you give no flag, the runner's own default applies instead. That default can be `--fastpf 512`, even for the 9b models.
4. Whether `gguf2pack`, built on Linux, makes the pack layout that the M36 runner expects. It is the same tool from the same repository, so it probably does.
5. Whether `run.sh` builds its Metal kernels on first use. This would be a one-time cost, like NVRTC on Linux. `llb` prints no message for it on the Mac.
6. Metal memory. `llb` makes no memory check on the Mac, because there is no `nvidia-smi`. A model that is too big fails inside `run.sh`.
