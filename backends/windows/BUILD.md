# Building the engine on Windows

**Status (2026-11-16): the CPU part is built and run on Windows.** That covers
`qwen35-cli`, `gguf2pack`, `llb-tok`, the desktop CPU engine `engine_cpu`, and the
POSIX shim. The CUDA part (`qwen35-cli-gpu`, and the emitted CUDA engine) has not
been built here. Measurements are in
[`WINDOWS-TEST-STATUS.md`](../../WINDOWS-TEST-STATUS.md).

| File | What it is |
|---|---|
| `build.ps1` | Builds everything below; prints each command. |
| `compat/` | The POSIX shim (an `-I` dir) and its self-test. See [compat/README.md](compat/README.md). |
| `engine_cpu.c`, `engine_cpu_gguf.c`, `cblas_sgemm.c` | The desktop CPU engine. |
| `winabi.ps1` | The Win64 fix for emitted Bend C (see [Emitting Bend C on Windows](#emitting-bend-c-on-windows)). |

## Outputs

`build.ps1` writes the CPU lane to `build\win\cpu\`:

| File | From | Compiler |
|---|---|---|
| `qwen35-cli.exe` | `tools/qwen35_engine.c` (the CPU twin CLI) | clang or MSVC `cl` |
| `gguf2pack.exe` | `tools/gguf2pack.c` | clang or `cl` |
| `obj\llbw_posix.o`, `test_shim.exe` | `compat/llbw_posix.c`, `compat/test_shim.c` (built and run) | clang |
| `llb-tok.exe` | `tools/llb/llb-tok.c` + the shim | clang |
| `engine_cpu.exe` | `engine_cpu.c` + `tools/sys1dual/dual.c` and its request objects + the shim | clang |
| `engine\cpu\` | The desktop contract dir: `engine_cpu.exe`, `gguf2pack.exe`, `llb-tok.exe`, and `llb-serve.exe` when `-LlbServe PATH` is given | |

`build\qwen35-cli-gpu.exe` is the CUDA CLI, built when `nvcc` is on PATH (not yet
run here).

## The desktop engine contract, and what `engine_cpu` is

The app (`bend-app/desktop/engine/backend.mjs`) expects each engine dir to hold
four binaries: `llb-serve`, the engine (`sys1mac` on the Mac, `engine_gpu` for
CUDA, `engine_cpu` for CPU), `gguf2pack`, and `llb-tok`. Each has a `.exe` suffix
on Windows. **Nothing in this repo was named `engine_cpu` before this kit.** The
app's Linux plan says "the omen lane builds it", and no recipe for it exists here.

### What the supervisor runs

`bend-app/desktop/engine/supervisor.mjs` runs these three commands.

1. Make the sidecar (once per GGUF):

   ```
   llb-tok extract MODEL.gguf SIDECAR                 -> SIDECAR\tokenizer.bin
   gguf2pack MODEL.gguf --sidecar-only SIDECAR        -> SIDECAR\manifest.json, table.bin, chat.flags
   ```

   It passes no `--target` off the Mac.

2. Start the server with the engine's argv after `--`. The engine argv is the
   Bend runtime's form, and `0 --steps CTX` sizes the KV capacity:

   ```
   llb-serve --host 127.0.0.1 --port P --api-key K --alias NAME --reasoning off --parallel N
             --engine-grammar --parent-fd 3 [--canon] --engine <dir>\engine_cpu.exe
             -- [--gpu <n>GB, cuda only] --threads T -- SIDECAR 0 --steps CTX
   ```

   Environment: `LLB_GGUF=MODEL.gguf`, `TMPDIR=<state>\engine-tmp`, and
   `LLB_DUAL_FASTPF` / `LLB_DECIDE_FASTPF`.

3. `llb-serve` starts each engine slot. It sets `LLB_SYS1=1` and
   `LLB_SYS1_STREAM=1`, then talks to the engine over **stdin/stdout JSON lines**,
   the protocol of `tools/sys1dual/dual.c`.

### The protocol, per request

- **In:** one line on stdin, for example
  `{"mode":"generate","prompt":TEXT,"n":N,...}`. Optional fields:
  - `special`
  - `sampling` or `sample`
  - `session`, `bank`
  - `json_schema` or `grammar`
  - `image`, `images`, `readout`

  Anything else is a decide request.
- **Out, while generating:** with `LLB_SYS1_STREAM=1`, one
  `{"token":ID,"text":PIECE}` line on stdout per id.
- **Out, at the end:** one response line,
  `{"prompt_ids":[..],"ids":[..],"text":".."}`, or `{"error":MSG}` if the request
  is refused.
- **stderr:** after the model loads, the engine writes `sys1: ... ready` lines;
  llb-serve treats the engine as up from then on. It also writes one timing line
  per request.

### How the engine side works

On Linux the engine is the emitted Bend program (`engine_qwen35_gpu.bend`, CUDA
host) linked with `dual.c`. Its `Sys1.next` effect calls `dual_next()`, which:

- reads the next line;
- tokenizes it;
- returns a command `ID ID ... --steps N --pos K [--sample SPEC]`.

The engine then runs that, printing one generated id per line on stdout (`dual.c`
captures those lines), and calls `dual_done()` to answer.

### Why the CPU engine is not emitted Bend

There is no emitted CPU engine in this repo, for any platform:

- `engine_qwen35_gpu.bend`'s per-token pass is one host-served bang, `llb.pass_warp`.
  Its Bend body is a NaN placeholder ("No CPU body").
- The host chain that serves it (`host/llb_host.c`) is compiled only with a GPU
  backend (`LLB_LAUNCH`).
- Emitted without `BEND_CUDA`, `llb-engine.c` does not compile: `sys1.c` needs
  `llb_hidden_*` from that chain. Even if it compiled, there would be nothing to
  run.
- `engine_qwen35.bend` (the pure-Bend CPU lane) is a 9B Q8_0 research path with
  no `Sys1` loop.
- Android's CPU engine is `tools/qwen35_engine.c` (the C twin) with patches, plus
  a JSON-lines loop (`backends/android`).

### What `engine_cpu` is on Windows

**`engine_cpu` = the C twin's forward + the same `dual.c`.** `engine_cpu.c` runs
the loop the Bend engine's `srv` / `Sys1.next` / `Sys1.done` run, around
`forward_token` / `final_logits`. So llb-serve drives it exactly as it drives
`llb-engine` / `sys1mac`. In detail:

- **argv:** `[--threads N] [--gpu X (ignored, one stderr line)] [--workers 1] [--] SIDECAR TOK... --steps CTX`.
  The KV capacity is the longest prompt plus steps minus 1, so `0 --steps CTX`
  gives CTX positions.
- **Weights:** `SIDECAR\manifest.json` gives the shape and each tensor's pack
  offset. The bytes come straight from `LLB_GGUF` through `host/llb_gguf.c`'s
  loader, with each tensor checked against the manifest. If `SIDECAR` is a full
  pack dir (`pack.bin` + `index.tsv`), that is read instead.
- **`LLB_SYS1` unset:** runs `TOK...` for CTX steps once. It prints ids and
  `wall_ms`, plus `logits` lines on stderr, as the Bend engine does.
- **Per request:** handles `--pos K`. A continuation replays the retained ids;
  the recurrent state cannot step back, so a rewind recomputes from 0. It takes
  `--sample SPEC` (else `LLB_SAMPLE`) through `host/llb_sampler.c`, the Bend
  host's own sampler. Greedy is the first max. For a decide request it writes the
  CLEF hidden rows (`{pos, width}` + floats after the output norm), as
  `llb_hidden_put` does.
- **No decide head:** if the GGUF has no CLEF decide head and `LLB_SYS1_NOHEAD`
  is unset, it sets `LLB_SYS1_NOHEAD=1` and says so on stderr. Without that,
  `dual.c` exits at start on a stock Qwen GGUF, as `sys1mac` did in
  `receipts/port-20261106/run-metal-served-stock-gguf.txt`. A decide request then
  gets `dual.c`'s refusal.
- **Host hooks:** prefix cache, banks, native batch, and vision rows are off, as
  `llb_host.c` answers when their env switch is unset. `json_schema` / `grammar`
  are refused, with one stderr line.
- **After an end-of-generation id:** llb-serve drops every id after the first
  one; the Bend engine still runs all N steps. `engine_cpu` hands back copies of
  that id for the remaining steps instead of computing them, and reports
  `padded_after_eog` on stderr. Set `LLB_CPU_EOG_PAD=0` to compute every step.
- **Speed:** GEMV rows are split over `--threads` workers, four rows interleaved
  per worker. Every row is summed term for term as the twin does, so the logit
  bits are the twin's; `LLB_CPU_ROWS4=0` is the A/B switch. The only edit to a
  shared source is a 6-line `#ifdef LLB_GEMV_ROWS` hook in `tools/qwen35_engine.c`.
  Without the define, the preprocessed file is identical to HEAD's.

## Emitting Bend C on Windows

This is the CUDA lane's first step, run and checked here.

1. **Use LF sources.** This checkout has `core.autocrlf=true`, so the working tree
   is CRLF. The patch stack needs LF, and the emitted hash depends on it. Export
   with `git -c core.autocrlf=false archive`:
   - `origin/pins/bend-240ecef4 bend2` gives the pinned compiler (tree `aada4cbf`,
     as `build.sh` checks);
   - `HEAD <the .bend files> kernels host tools` gives the engine tree.
2. **Patch the compiler copy as Linux does.** Run
   `BEND_ROOT=<pinned> BEND_BUILD=<copy> bash tools/build_gpu.sh ...` from Git Bash.
   It applies the 67-patch stack (`--fuzz=0`) and then fails at the emit, which
   step 3 redoes.
3. **Emit with the compiler copy's `bend2` dir as the working directory.** Pass
   the entry as an absolute path with forward slashes:

   ```
   cd <copy>\bend2
   set LLB_HOST_DIR=<tree>/host
   bun main.ts <tree>/engine_qwen35_gpu.bend -o <out>/llb-engine.c
   ```

   `bend.ts` takes a file's directory from its last `/`. `base.bend`'s path comes
   from `fileURLToPath` with backslashes, so its directory resolves to "" (the
   working directory), and `effs/` is looked for there. `LLB_HOST_DIR` is needed
   for `llb_formats.h`.
4. **Check the result.** At commit 135100a this produced
   `657f29bf8eeb2bc6c2fed35133801fab5ab82fe005ca9c6be46578949663cdf8`. That is the
   hash macOS and Linux recorded in
   `receipts/port-20261106/emit-cuda-port.txt`, so Windows emission is
   byte-identical. At HEAD (`af953ef`) it is `2838ed40…`; HEAD's `host/llb_host.c`
   has moved on since then.
5. **Run `winabi.ps1` on the output before compiling it.** On Win64 the
   runtime's `musttail` segment chain reads a dead `Env` copy. The script changes
   two lines to `sysv_abi`. Details: [compat/README.md](compat/README.md).
6. **Compile against `compat/`** with `-D_CRT_DECLARE_NONSTDC_NAMES=0`, and link
   `llbw_posix.o`, `cblas_sgemm.c` (for `head.c`), `clang_rt.builtins`,
   `-lws2_32 -lwinmm`, and `/STACK:8388608`.

## Prerequisites

- Windows 10/11 x64.
- **LLVM for Windows** (clang; 23.1.5 used). `build.ps1` finds it on PATH or in
  `Program Files\LLVM`.
- **Visual Studio 2022** with "Desktop development with C++" (MSVC + Windows SDK).
  clang uses its headers and libraries; 14.44 / SDK 10.0.26100 were used.
  `cl /std:c11` with `windows.h` needs SDK 10.0.20348 or newer.
- For the GPU build only: an NVIDIA driver (`nvidia-smi` works) and the **CUDA
  Toolkit** (`nvcc` on PATH, in "Developer PowerShell for VS 2022"). Cards with
  compute capability 12.x need CUDA 12.8 or newer.
- For emitting Bend C: bun and Git Bash (patch).

## One command

From the repo root, in PowerShell 7 or Windows PowerShell 5.1 (both were run):

```powershell
powershell -ExecutionPolicy Bypass -File backends\windows\build.ps1 -NoCuda [-LlbServe <path to llb-serve.exe>]
```

Options:

- `-Cc clang|cl` forces the CPU compiler. `cl` builds only `qwen35-cli` and
  `gguf2pack`; the shim targets need clang.
- `-Arch sm_89` overrides GPU detection.
- `-NoCuda` skips the GPU build.

The script rebuilds every target on every run. It ends with a
`built: ... | engine_cpu.exe (...)` line, or a `FAILED:` line naming each target
that failed and its exit code, and then exits 1. Run on this machine: exit 0 in
about 5 s.

## The commands it runs (CPU)

From the repo root. `$C` is the compat dir, `$O` is `build\win\cpu\obj`, and
`$RT` is the clang resource dir's `lib\windows\clang_rt.builtins-x86_64.lib`.

```powershell
clang -O2 -std=c11 -D_CRT_SECURE_NO_WARNINGS -ffp-contract=off -fno-fast-math tools\qwen35_engine.c -o build\win\cpu\qwen35-cli.exe
clang -O2 -std=c11 -D_CRT_SECURE_NO_WARNINGS tools\gguf2pack.c -o build\win\cpu\gguf2pack.exe
clang -std=c11 -O2 -D_CRT_DECLARE_NONSTDC_NAMES=0 -c backends\windows\compat\llbw_posix.c -o $O\llbw_posix.o
clang -std=c11 -O2 -D_CRT_SECURE_NO_WARNINGS -D_CRT_DECLARE_NONSTDC_NAMES=0 -I $C tools\llb\llb-tok.c $O\llbw_posix.o $RT -lws2_32 -lwinmm -o build\win\cpu\llb-tok.exe
# each of: tools\sys1dual\dual.c tools\metrics\metrics.c tools\metrics\tokstream.c tools\glue\glue_out.c
#          tools\clef3\{head,weights,spans,json}.c host\llb_sampler.c backends\windows\{cblas_sgemm,engine_cpu_gguf,engine_cpu}.c
clang -std=c11 -O2 -ffp-contract=off -fno-fast-math -D_POSIX_C_SOURCE=200809L -D_CRT_SECURE_NO_WARNINGS -D_CRT_DECLARE_NONSTDC_NAMES=0 -I $C -c X.c -o $O\X.o
clang -O2 -o build\win\cpu\engine_cpu.exe $O\*.o $RT -lws2_32 -lwinmm -Wl,/STACK:8388608
```

`qwen35_tokenizer.c` is not linked separately: `qwen35_engine.c` includes it.

With `clang-cl`, write the twin's flags as `/clang:-ffp-contract=off /clang:-fno-fast-math`.
MSVC for the two plain targets:

```powershell
cl /nologo /O2 /std:c11 /D_CRT_SECURE_NO_WARNINGS /fp:precise tools\qwen35_engine.c /Fe:build\win\cpu\qwen35-cli.exe /Fo:build\win\cpu\obj\
cl /nologo /O2 /std:c11 /D_CRT_SECURE_NO_WARNINGS tools\gguf2pack.c /Fe:build\win\cpu\gguf2pack.exe /Fo:build\win\cpu\obj\
```

**CUDA CLI** (not yet run here):

```powershell
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # 8.6 -> sm_86
nvcc -O3 -fmad=false -std=c++14 -arch=sm_XX -Xcompiler /fp:precise -D_CRT_SECURE_NO_WARNINGS -o build\qwen35-cli-gpu.exe tools\qwen35_engine.cu
```

## What `tools/llb_platform.h` changed

Before this kit, `tools/qwen35_engine.c` and `tools/gguf2pack.c` called POSIX
directly. Both now include [`tools/llb_platform.h`](../../tools/llb_platform.h).
Its Windows branch compiled with no change on its first build here (clang 23,
2026-11-16).

| Call | POSIX (unchanged) | Windows |
|---|---|---|
| open for read / write | `open` | `_open` with `_O_BINARY` |
| pread / pwrite | `pread` / `pwrite` | `ReadFile` / `WriteFile` with an `OVERLAPPED` offset, at most 1 GiB per call |
| file size | `fstat` | `_fstat64` |
| map pack.bin | `mmap(PROT_READ, MAP_PRIVATE)` | `CreateFileMappingW` + `MapViewOfFile(FILE_MAP_READ)` |
| map fails | error, exit 1 | read the file into RAM, print one line, skip the per-layer page drop |
| per-layer page drop | `madvise(MADV_DONTNEED)` | `VirtualUnlock` on the unlocked view |
| monotonic clock | `clock_gettime` | `QueryPerformanceCounter` |
| mkdir, exists, truncate | `mkdir`, `access`, `ftruncate` | `_mkdir`, `_access`, `_chsize_s` |

These differences on Windows are known. None is hidden by a stub:

- `gguf2pack --head-only` writes a full-size, zero-filled `pack.bin`, because
  NTFS files are not sparse unless asked.
- `gguf2pack` prints `VmHWM -1 kB`; there is no `/proc/self/status`.
- `posix_fadvise` is compiled out, as on macOS.

## The numerical flags

The CPU twin and the CUDA build are meant to give bit-identical results for the
same pack. That holds only if no compiler fuses `a*b + c` into an FMA and nothing
reorders the math.

| Toolchain | Flags | Notes |
|---|---|---|
| gcc / clang (Linux, macOS) | `-ffp-contract=off -fno-fast-math` | As in `tools/build_qwen35_cli.sh`. |
| clang on Windows | `-ffp-contract=off -fno-fast-math` | The same flags. The 4B logit bits equal macOS's (WINDOWS-TEST-STATUS.md). |
| clang-cl | `/clang:-ffp-contract=off /clang:-fno-fast-math` | |
| MSVC `cl` | `/fp:precise`, **without** `/fp:contract` and without `/arch:AVX2` | MSVC has no "contract off" switch; leaving out `/fp:contract` is how you get it. Use VS 2022. |
| nvcc device code | `-fmad=false` | As on Linux. |
| nvcc host code (cl) | `-Xcompiler /fp:precise` | |

`exp`, `log1p`, `sin`, `cos`, and `pow` all go through `double` and are rounded
once. The UCRT's versions could differ from glibc's or Apple's in the last bit.
On this machine they did not move a single logit bit of the 4B reference run.
