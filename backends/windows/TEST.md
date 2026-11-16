# Testing the engine on Windows

Build first with [`BUILD.md`](BUILD.md) (`build.ps1 -NoCuda` for the CPU lane).
The commands run from the repo root. Where output is redirected they go through
`cmd /c`, because Windows PowerShell 5.1 wraps a native program's stderr in error
records.

**Status, 2026-11-16:**

- Sections 1–4 (CPU) were run on Windows 11 (Ryzen 7 7700X, clang 23.1.5); the
  observed results are below each step.
- Section 5 (GPU) has not run: the CUDA toolkit was still installing.

The models on the test machine are in `D:\models\eureka\` and the packs in
`D:\models\eureka\packs\<model>`; substitute your own paths.

## 1. Get the model (Qwen3.5-4B Q4_K_M)

This is the 4B row pinned in [`tools/llb/models.conf`](../../tools/llb/models.conf):

| | |
|---|---|
| URL | `https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf` |
| SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Size | 2,740,937,888 bytes |

```powershell
curl.exe -L -C - -o models\Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
(Get-FileHash -Algorithm SHA256 models\Qwen3.5-4B-Q4_K_M.gguf).Hash.ToLower()
```

Disk: the GGUF is 2.7 GB, and a full pack built from it is another 2.7 GB.
`engine_cpu` needs only the sidecar (about 7.5 MB).

## 2. Pack it, and build the engine's sidecar

The usage line in `tools/gguf2pack.c` is
`gguf2pack MODEL.gguf OUT_DIR [--format q8|quant] [--head-only]`. `OUT_DIR` must
not already hold a pack.

```powershell
cmd /c "build\win\cpu\gguf2pack.exe D:\models\eureka\Qwen3.5-4B-Q4_K_M.gguf D:\models\eureka\packs\Qwen3.5-4B-Q4_K_M --format quant"
echo "exit=$LASTEXITCODE"
```

**Pass:** exit 0, and the last two lines read:

```
426 tensors (+ output.weight alias), 2729969664 bytes, pack_sha256 86be153fc89f33ea1125d7d6c07e193b2ef73043eafbb1be851511a3c0d9ba9f
wrote <OUT_DIR> (pack.bin, 32 layer files, manifest.json, index.tsv, table.bin)
```

**Observed on Windows:** exit 0 in 43.8 s (63.6 MB/s), and `pack_sha256 86be153f…`,
equal to macOS. The 0.8B gave
`320 tensors (+ output.weight alias), 521555200 bytes, pack_sha256 246c84850555c948c2de41e63acc5acec16cda83cb856dfd72150d8bbabd74fd`.

The desktop's sidecar is what the supervisor makes on first use (no pack.bin):

```powershell
build\win\cpu\llb-tok.exe extract D:\models\eureka\Qwen3.5-4B-Q4_K_M.gguf SIDECAR
build\win\cpu\gguf2pack.exe D:\models\eureka\Qwen3.5-4B-Q4_K_M.gguf --sidecar-only SIDECAR
```

**Observed:** both exit 0. `SIDECAR\tokenizer.bin` is 7,460,363 bytes with
sha256 `de71a648a71960a539d4d660ebff804a63a691ba74aea89e3e09dc7bf4553ad9`. That is
the hash `docs/models.md` records for the Qwen3.5 tokenizer, for both the 0.8B and
the 4B.

## 3. The CPU CLI

The usage line is `qwen35-cli <packdir> [tok...] [options]`. The prompt
`760 6511 314 9338 369` is "The capital of France is".

```powershell
cmd /c "build\win\cpu\qwen35-cli.exe D:\models\eureka\packs\Qwen3.5-4B-Q4_K_M 760 6511 314 9338 369 --steps 8 > run-cpu.out 2> run-cpu.err"
echo "exit=$LASTEXITCODE"
type run-cpu.out
```

**Pass:** exit 0, 8 ids then `wall_ms` and `tok_s`. The ids do not have to match
macOS; a mismatch is a finding to report.

**Observed on Windows: exit 0.** The ids `11751 13 198 32 13 2912 198 33` (" Paris.\nA. True\nB")
and all 16 logit values of the 8 `logits` lines are **bit-identical to the macOS
reference below**. `wall_ms 43585`, `tok_s 0.1835` (single-threaded twin; macOS
42336). The 0.8B pack gave `279 6511 314 279 3046 13 198 760` in
`wall_ms 30464`, the same ids as the Linux host receipt
`receipts/port-20261106/run-cpu-host-08b.txt`.

### Reference: the same pack and command on macOS (arm64, CPU twin)

```
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

The integer after each id is the logit's float32 bits.

## 4. engine_cpu, the desktop CPU engine

### 4a. One run, no server

`LLB_SYS1` unset: the given ids for `--steps`, as the Bend engine prints them. The
weights come from the sidecar + GGUF.

```powershell
$env:LLB_GGUF = 'D:\models\eureka\Qwen3.5-4B-Q4_K_M.gguf'
cmd /c "build\win\cpu\engine_cpu.exe --threads 16 -- SIDECAR 760 6511 314 9338 369 --steps 8 > e.out 2> e.err"
```

**Observed:** exit 0. The same 8 ids, and the 8 `logits` lines **bit-identical
to the macOS reference** above. It matched both with the 4-row GEMV
(`wall_ms 4333`) and with `LLB_CPU_ROWS4=0` (`wall_ms 5840`). The 0.8B gave logit
bits identical to `qwen35-cli`'s, from the sidecar and from the full pack.

### 4b. Driven the way llb-serve drives it (JSON lines over stdin/stdout)

```powershell
$env:LLB_SYS1 = '1'; $env:LLB_SYS1_STREAM = '1'; $env:TMPDIR = "$env:TEMP"
$env:LLB_GGUF = 'D:\models\eureka\Qwen3.5-4B-Q4_K_M.gguf'
'{"mode":"generate","prompt":"The capital of France is","n":8}' | build\win\cpu\engine_cpu.exe --threads 16 -- SIDECAR 0 --steps 4096
```

**Pass:** on stdout, one `{"token":..,"text":..}` line per id, then
`{"prompt_ids":[..],"ids":[..],"text":".."}`. On stderr, `sys1: ... ready`, then
one `{"engine_cpu":{...}}` line per request.

**Observed.** These ran from Git Bash with the same environment and a file of
request lines on stdin:

- The 4B answered
  `{"prompt_ids":[760,6511,314,9338,369],"ids":[11751,13,198,32,13,2912,198,33],"text":" Paris.
A. True
B"}`.
- A ChatML request with `"special":true` got a real reply from both models. The
  4B said "The sea is a vast, ever-shifting expanse of blue that breathes
  rhythmically against the shore, ...". The 0.8B answered "What is 2+2?" with
  `4<|im_end|>`, followed by 22 padded `<|im_end|>` ids (`padded_after_eog`).
- On the 0.8B, a `sampling` request (temperature 0.8, top_p 0.9, seed 42) got
  sampled text.
- On the 0.8B, a decide request on this head-less GGUF got
  `{"error":"decide needs the clef head (LLB_SYS1_NOHEAD is set)"}`.

Speeds are in WINDOWS-TEST-STATUS.md.

### 4c. The shim self-test

`build.ps1` builds and runs `build\win\cpu\test_shim.exe`.

**Observed:** 27/27 `ok`, `PASS: 0 failed`.

## 5. GPU (not yet run on Windows)

**GPU** (the whole 2.73 GB `pack.bin` is copied to the card; it needs about 3 GB
of free VRAM):

```powershell
cmd /c "build\qwen35-cli-gpu.exe D:\models\eureka\packs\Qwen3.5-4B-Q4_K_M 760 6511 314 9338 369 --steps 8 --gpu > run-gpu.out 2> run-gpu.err"
cmd /c "build\qwen35-cli-gpu.exe D:\models\eureka\packs\Qwen3.5-4B-Q4_K_M --selftest 2> selftest.err"
nvidia-smi
```

**Pass:** exit 0 and 8 ids. The design target for `--selftest` is
`selftest TOTAL tensors=... bits_diff=0`. The CPU run above is the reference to
compare the GPU ids and logit bits against.
