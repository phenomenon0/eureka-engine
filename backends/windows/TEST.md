# Testing the engine CLI on Windows (NVIDIA GPU)

**Status: untested on Windows.** This is the procedure for the first run. Build the
binaries with [`BUILD.md`](BUILD.md) first. All commands run in PowerShell from the
repo root. The commands that redirect output go through `cmd /c` because Windows
PowerShell 5.1 wraps a native program's stderr in error records.

## 1. Get the model (Qwen3.5-4B Q4_K_M)

This is the 4B row pinned in [`tools/llb/models.conf`](../../tools/llb/models.conf):

| | |
|---|---|
| URL | `https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf` |
| SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Size | 2,740,937,888 bytes (2.74 GB, 2.55 GiB) |

On 2026-11-06, Hugging Face reported the same size and SHA-256 for this URL.

Disk: the GGUF is 2.7 GB and the pack built from it is 4.8 GB, so plan for about 7.5 GB.

```powershell
mkdir models -Force
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest many times slower
Invoke-WebRequest -Uri https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf -OutFile models\Qwen3.5-4B-Q4_K_M.gguf
(Get-FileHash -Algorithm SHA256 models\Qwen3.5-4B-Q4_K_M.gguf).Hash.ToLower()
# must print 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
```

`curl.exe -L -o models\Qwen3.5-4B-Q4_K_M.gguf <URL>` also works, and it can resume
an interrupted download with `-C -`.

## 2. Pack it

The usage line in `tools/gguf2pack.c` is
`gguf2pack MODEL.gguf OUT_DIR [--format q8|quant] [--head-only]`. `OUT_DIR` must not
already hold a pack, and its parent directory must exist.

```powershell
cmd /c "build\gguf2pack.exe models\Qwen3.5-4B-Q4_K_M.gguf pack4b --format quant"
echo "exit=$LASTEXITCODE"
```

**Pass:** exit 0, and the last line reads:

```
wrote pack4b (pack.bin, 32 layer files, manifest.json, index.tsv, table.bin)
```

The line before it should report
`426 tensors (+ output.weight alias), 2729969664 bytes, pack_sha256 86be153fc89f33ea1125d7d6c07e193b2ef73043eafbb1be851511a3c0d9ba9f`.
That is the `pack_sha256` macOS produced for this GGUF. For the 0.8B, macOS and Linux
produced the same `pack.bin` hash. The pack is a byte copy of the GGUF tensors, so a
different hash on Windows is a converter bug. Report it.

## 3. Run it

The CLI's usage line is `qwen35-cli <packdir> [tok...] [options]`. The pack
directory comes first, then the prompt token ids, then the options. The prompt
`760 6511 314 9338 369` is "The capital of France is". It is the prompt the in-tree
CPU receipt uses (`receipts/port-20261106/run-cpu-host-08b.txt`), and it is the same
token ids for the 0.8B and the 4B.

**GPU** (the whole 2.73 GB `pack.bin` is copied to the card, so it needs about 3 GB of free VRAM):

```powershell
cmd /c "build\qwen35-cli-gpu.exe pack4b 760 6511 314 9338 369 --steps 8 --gpu > run-gpu.out 2> run-gpu.err"
echo "exit=$LASTEXITCODE"
type run-gpu.out
type run-gpu.err
nvidia-smi
```

**GPU self-test** (optional, and worth running). It runs every tensor's GEMV on the
CPU and the GPU and counts the output floats whose bits differ:

```powershell
cmd /c "build\qwen35-cli-gpu.exe pack4b --selftest 2> selftest.err"
echo "exit=$LASTEXITCODE"
type selftest.err
```

The design target is `selftest TOTAL tensors=... bits_diff=0`, with exit 0. A non-zero
`bits_diff` is worth reporting, but it does not fail the test.

**CPU** (slow; see section 5):

```powershell
cmd /c "build\qwen35-cli.exe pack4b 760 6511 314 9338 369 --steps 8 > run-cpu.out 2> run-cpu.err"
echo "exit=$LASTEXITCODE"
type run-cpu.out
```

## 4. Report and the pass criterion

**Pass:** the exe exits 0, and `run-*.out` holds 8 token ids, one per line, then
`wall_ms <ms>` and `tok_s <x>`.

The ids do **not** have to match the reference below. A different OS, C runtime math
library (UCRT vs glibc vs Apple libm), and compiler can change the last bit of a
logit. Once a near-tie flips, the rest of the sequence goes a different way. Report a
mismatch; it is not a failure.

Paste back:

1. The last 3 lines that `build.ps1` printed (the `built:` / `FAILED:` summary).
2. The output of `run-gpu.out` and `run-gpu.err` (the token ids, `wall_ms`, and the
   `loaded`, `gpu:`, `logits`, and `step` lines).
3. The `nvidia-smi` output.
4. `selftest.err`, if you ran it.
5. `run-cpu.out` and its `wall_ms`, if you ran it.
6. `nvcc --version`, `clang --version` or `cl` (whichever built the CPU exe), and
   `[Environment]::OSVersion`.

### Reference: the same pack and command on macOS (arm64, CPU twin, this commit)

```
$ qwen35-cli pack4b 760 6511 314 9338 369 --steps 8
11751
13
198
32
13
2912
198
33
wall_ms 42336
tok_s 0.1890
stderr:
loaded 427 tensors, pack 2729969664 bytes; L=32 D=2560 H=16 KV=4 hd=256 dff=9216 V=248320
logits 4 11751 1100047528 7172 1098922251
logits 5 13 1101343111 11 1100849196
logits 6 198 1099743449 271 1099474450
logits 7 32 1097709062 760 1097591250
logits 8 13 1099243022 3177 1095299429
logits 9 2912 1096092421 220 1094342170
logits 10 198 1103114782 318 1099815782
logits 11 33 1103993842 51 1100464919
```

The ids decode to ` Paris.\nA. True\nB`. The integer after each id is the logit's
float32 bits. Comparing those bits shows exactly where Windows first drifts, if it
does. No CUDA reference exists for this 4B command. The CUDA build is meant to match
the CPU twin bit for bit, and `--selftest` checks that on your card.

## 5. CPU is slow; that is normal

The CPU twin is a correctness reference, not a fast path. It runs on one thread,
in plain C, with no SIMD intrinsics. The only earlier Windows datapoint is 64 s per
token: 2 tokens in 128 s on a 4-vCPU Windows Server 2025 VM. That build had no
recipe in the tree, and the doc does not say which model it ran. On an M-series Mac,
this 4B takes about 3.7 s per token, or 42 s for the command above.

Expect minutes, not seconds. The console may show **nothing** until the run ends:
ids go to stdout, and when stdout is redirected the C runtime buffers it. To see
progress, watch the `step <pos> <ms>` lines grow in `run-cpu.err` with
`Get-Content run-cpu.err -Wait`. A silent console is not a hang.
