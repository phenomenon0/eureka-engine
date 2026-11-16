# Windows test status

As of 2026-11-16, on a real Windows machine:

- **Machine:** Windows 11 Pro N for Workstations 26200, x64, AMD Ryzen 7 7700X
  (8C/16T), 31 GB RAM, RTX 3090.
- **Toolchain:** clang 23.1.5 (`x86_64-pc-windows-msvc`), MSVC 14.44, Windows SDK
  10.0.26100, PowerShell 7.6.10 and 5.1.

| | |
|---|---|
| CPU lane builds on Windows? | **Yes.** `qwen35-cli`, `gguf2pack`, `llb-tok`, `engine_cpu`, and the POSIX shim, all with clang. |
| CPU lane runs on Windows? | **Yes, measured below.** |
| CUDA lane | **Not built, not run.** The CUDA toolkit was still installing. |
| llb-serve.exe | Not part of this lane (another lane ports `tools/sys1dual/serve.c`). |
| Windows support claimed? | CPU engine: works on this machine. Nothing else yet. |

## Proven (commands in `backends/windows/TEST.md`)

### 1. Build

`powershell -ExecutionPolicy Bypass -File backends\windows\build.ps1 -NoCuda` exits
0 (about 5 s) in both PowerShell 7.6 and 5.1, ending with:

```
built: qwen35-cli.exe (CPU) | gguf2pack.exe | llbw_posix.o (POSIX shim) | test_shim.exe (shim self-test, run) | llb-tok.exe | engine_cpu.exe (tools/qwen35_engine.c + dual.c)
```

`tools/llb_platform.h`'s Windows branch, written blind on macOS, compiled
unchanged on its first build. So did `qwen35_engine.c` and `gguf2pack.c`: no
errors, no warnings.

### 2. Pack

`gguf2pack Qwen3.5-4B-Q4_K_M.gguf <dir> --format quant` exits 0 in 43.8 s, with
`pack_sha256 86be153fc89f33ea1125d7d6c07e193b2ef73043eafbb1be851511a3c0d9ba9f`,
equal to macOS. The 0.8B gives
`246c84850555c948c2de41e63acc5acec16cda83cb856dfd72150d8bbabd74fd`, 521,555,200
bytes. The packs are in `D:\models\eureka\packs\Qwen3.5-{4B,0.8B}-Q4_K_M`.

### 3. CPU CLI, 4B

`qwen35-cli <4B pack> 760 6511 314 9338 369 --steps 8` exits 0 with ids
`11751 13 198 32 13 2912 198 33`. **All 8 `logits` lines are bit-identical to the
macOS reference** in TEST.md: ids and float32 bits, top two at every position.
The UCRT's `exp`/`log1p`/`sin`/`cos`/`pow` did not move one bit here.

- 4B: `wall_ms 43585`, `tok_s 0.1835` (single thread; macOS 42336).
- 0.8B: `279 6511 314 279 3046 13 198 760`, the same ids as
  `receipts/port-20261106/run-cpu-host-08b.txt` (Linux host).

### 4. llb-tok

`llb-tok extract GGUF DIR` writes `tokenizer.bin`, 7,460,363 bytes, sha256
`de71a648…53ad9`, equal to the hash in `docs/models.md`. `gguf2pack --sidecar-only`
writes `manifest.json` + `table.bin` + `chat.flags`. These are the supervisor's
sidecar steps.

### 5. engine_cpu

This is the desktop contract's CPU engine (`backends/windows/BUILD.md` says what
it is). It runs `tools/qwen35_engine.c`'s forward under `tools/sys1dual/dual.c`,
the JSON-line protocol llb-serve speaks.

**One run, no server** (sidecar + `LLB_GGUF`, `--threads 16`):

- 4B: the same ids and the same 8 logit lines, bit for bit, as the macOS
  reference, with the 4-row GEMV (`wall_ms 4333`) and without it (`5840`).
- 0.8B: logit bits identical to `qwen35-cli`'s.

**Driven as llb-serve drives it** (`LLB_SYS1=1 LLB_SYS1_STREAM=1`, JSON lines on
stdin, `SIDECAR 0 --steps 4096`):

- Streamed `{"token","text"}` events, then the response line.
- 4B: `" Paris.\nA. True\nB"` for the reference prompt, and coherent ChatML replies.
- Sampling (`temperature`/`top_p`/`seed` through `host/llb_sampler.c`) works.
- A decide request on a head-less GGUF gets `dual.c`'s refusal. The engine sets
  `LLB_SYS1_NOHEAD=1` itself and says so on stderr.

Speeds: one 19-token ChatML prompt, 48 steps, 16 threads unless noted. The
machine was about 45% loaded by other work (another agent's `find`, the CUDA
installer).

| Model | Prefill | Decode, 16 threads | Decode, 8 threads |
|---|---|---|---|
| Qwen3.5-0.8B Q4_K_M | 19 tokens in 985 ms (19.3 tok/s) | **10.3 tok/s** | 8.7 tok/s |
| Qwen3.5-4B Q4_K_M | 19 tokens in 5,679 ms (3.3 tok/s) | **2.5 tok/s** | 2.1 tok/s |

Load: weights in 83 ms (0.8B) and 433 ms (4B) from a warm page cache. Then
`dual.c`'s init takes about 3.4–3.6 s: with NOHEAD it opens `glue_out` over
`output.weight`.

Why it is no faster: the twin sums each row in one serial float chain. That fixed
order is the bit-exactness contract, and it is not vectorized. The 4-row
interleave keeps every row's sum order and fills the add latency with other rows
(1.35x on the 4B).

### 6. POSIX shim

`backends/windows/compat`, an `-I` dir plus `llbw_posix.c`:

- The unchanged `dual.c`, `metrics`, `glue_out`, `clef3`, `llb_gguf.c`,
  `llb_sampler.c`, and `llb-tok.c` compile and run against it.
- `test_shim.exe`: 27/27 checks pass.

### 7. Emitted Bend C on Windows

- **Byte-identical emission.** The compiler runs under bun on Windows. With LF
  sources and the working directory set to the compiler's `bend2`, the CUDA patch
  stack plus `engine_qwen35_gpu.bend` emits the same `llb-engine.c` as macOS and
  Linux: sha256 `657f29bf…` at commit 135100a, as
  `receipts/port-20261106/emit-cuda-port.txt` records. At HEAD `af953ef` it is
  `2838ed404a6c…`, because `host/llb_host.c` has moved on since.
- **Found: the emitted runtime crashes on Win64 as emitted.** Its `musttail`
  segment chain passes the 16-byte `Env` by value. Win64 passes it by a pointer
  into a frame the tail call frees. A program died with `bend: memory fault` or
  `bend: solo delivery lost`, `Env.mem == NULL`.
- **The fix.** `backends/windows/winabi.ps1` marks two lines `sysv_abi`. After it,
  `kernels/argmax.bend` runs through the shim and returns the right index on 1 and
  16 threads. This was checked with the pinned compiler and with the CUDA-stack
  compiler. The CUDA lane must run `winabi.ps1` on `llb-engine.c`.

## Not proven

- **CUDA:** `qwen35-cli-gpu.exe`, `--selftest`, and the emitted CUDA engine
  (`engine_gpu.exe`) are not built.
- **llb-serve.exe on Windows,** and so the desktop app's end-to-end chat on
  `engine_cpu`, is not run. `engine_cpu` was driven with the same environment and
  JSON lines llb-serve sends; the server itself was not in the loop.
- **Not in `engine_cpu`:**
  - `json_schema` / `grammar` requests (refused: "not in the CPU engine");
  - the prefix cache, banks, native batch, and vision rows (off);
  - a CLEF decide head: a head GGUF was not tried. Kev-4B was not run.
- **Decide on a head GGUF** would sum with the shim's plain-C `cblas_sgemm`, not
  OpenBLAS (last-bit differences possible).
- **MSVC `cl`:** it builds only the two plain targets and was not run.

## Earlier datapoint

`docs/backends.md` recorded 2 tokens in 128 s on CPU in a Windows Server 2025 VM,
with no recipe in the tree. This status replaces it.
