# Building the engine CLI on Windows

**Status: untested on Windows.** The files here were written on macOS. No one has
built or run them on Windows, and PowerShell has not even parsed `build.ps1`. The
first Windows run is the test. See [`WINDOWS-TEST-STATUS.md`](../../WINDOWS-TEST-STATUS.md).

This page covers the text CLI only: the CPU twin `qwen35-cli`, its CUDA build
`qwen35-cli-gpu`, and the converter `gguf2pack`. It does not cover the CUDA engine
(`build.sh`), the server, or WebGPU.

## What changed so it can build

Before this kit, `tools/qwen35_engine.c` and `tools/gguf2pack.c` called POSIX directly:
`mmap`, `madvise`, `open`, `pread`, `pwrite`, `fstat`, `mkdir`, `access`, `ftruncate`,
and `clock_gettime`. They would not compile with MSVC or plain clang on Windows.
Both now include [`tools/llb_platform.h`](../../tools/llb_platform.h):

| Call | POSIX (unchanged) | Windows |
|---|---|---|
| open for read / write | `open` | `_open` with `_O_BINARY` (without `_O_BINARY` the CRT opens in text mode and mangles the bytes) |
| pread / pwrite | `pread` / `pwrite` | `ReadFile` / `WriteFile` with an `OVERLAPPED` offset, at most 1 GiB per call (callers already loop) |
| file size | `fstat` | `_fstat64` |
| map pack.bin | `mmap(PROT_READ, MAP_PRIVATE)` | `CreateFileMappingW` + `MapViewOfFile(FILE_MAP_READ)` |
| map fails | error, exit 1 | read the file into RAM, print one line, skip the per-layer page drop |
| per-layer page drop | `madvise(MADV_DONTNEED)` | `VirtualUnlock` on the unlocked view. MSDN documents that this releases the pages from the working set. |
| perror | `perror` | `strerror(errno)` and `GetLastError()` |
| monotonic clock | `clock_gettime` | `QueryPerformanceCounter` |
| mkdir, exists, truncate | `mkdir`, `access`, `ftruncate` | `_mkdir`, `_access`, `_chsize_s` |

Neither file uses `pthread`, so there is nothing to port there.

These are known differences on Windows. None of them is hidden by a stub:

- `gguf2pack --head-only` writes a full-size `pack.bin` with zero-filled holes.
  NTFS files are not sparse unless the tool asks for it, and this one does not.
  The default (full) pack is unaffected.
- `gguf2pack` prints `VmHWM -1 kB`. It reads peak RSS from `/proc/self/status`,
  which does not exist on Windows. macOS prints the same thing.
- `posix_fadvise` is compiled out, as it already was on macOS.

On POSIX the change is byte-identical, and this was checked on macOS and Linux
(receipts in [`WINDOWS-TEST-STATUS.md`](../../WINDOWS-TEST-STATUS.md)).

## Prerequisites

- Windows 10/11 x64.
- **Visual Studio 2022 Build Tools** with the "Desktop development with C++" workload
  (MSVC + Windows SDK). Both `nvcc` and `clang` use its headers and libraries.
- For the CPU build, either **LLVM for Windows** (`clang` on PATH; recommended) or MSVC
  `cl` alone.
  `cl /std:c11` with `windows.h` needs Windows SDK 10.0.20348 or newer; older SDK
  headers fail to compile in C11 mode.
- For the GPU build: an NVIDIA driver (`nvidia-smi` works) and the **CUDA Toolkit**
  (`nvcc` on PATH). Its version must know your card's `sm_XX`. Cards with compute
  capability 12.x need CUDA 12.8 or newer.
- Open **"Developer PowerShell for VS 2022"** so `cl.exe` is on PATH. `nvcc` needs it,
  even when the CPU build uses clang.

## One command

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File backends\windows\build.ps1
```

Options: `-Cc clang|cl` forces the CPU compiler (default: clang if present, else cl).
`-Arch sm_89` overrides GPU detection. `-NoCuda` skips the GPU build.

The script rebuilds everything into `build\` on every run and prints each command
before running it. It ends with one of these:

```
built: qwen35-cli.exe (CPU) | qwen35-cli-gpu.exe (CUDA sm_XX) | gguf2pack.exe
```

or a `FAILED:` line naming each target that failed and its exit code. The script
exits 1 if any target failed.

## The commands it runs

Here they are by hand, from the repo root, so you can run them without the script.

**CPU, clang** (LLVM for Windows, GNU-style driver):

```powershell
mkdir build -Force
clang -O2 -std=c11 -D_CRT_SECURE_NO_WARNINGS -ffp-contract=off -fno-fast-math tools\qwen35_engine.c -o build\qwen35-cli.exe
clang -O2 -std=c11 -D_CRT_SECURE_NO_WARNINGS tools\gguf2pack.c -o build\gguf2pack.exe
```

With `clang-cl`, write the flags as `/clang:-ffp-contract=off /clang:-fno-fast-math`.

**CPU, MSVC:**

```powershell
cl /nologo /O2 /std:c11 /D_CRT_SECURE_NO_WARNINGS /fp:precise tools\qwen35_engine.c /Fe:build\qwen35-cli.exe /Fo:build\
cl /nologo /O2 /std:c11 /D_CRT_SECURE_NO_WARNINGS tools\gguf2pack.c /Fe:build\gguf2pack.exe /Fo:build\
```

**CUDA:**

```powershell
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.6 -> sm_86, 8.9 -> sm_89, 12.0 -> sm_120
nvcc -O3 -fmad=false -std=c++14 -arch=sm_XX -Xcompiler /fp:precise -D_CRT_SECURE_NO_WARNINGS -o build\qwen35-cli-gpu.exe tools\qwen35_engine.cu
```

The Linux script (`tools/build_qwen35_cli.sh`) hardcodes `sm_86`. This kit reads the
arch from the card and prints it.

## The numerical flags

The CPU twin and the CUDA build are meant to give bit-identical results for the same
pack. That holds only if no compiler fuses `a*b + c` into an FMA and nothing
reorders the math.

| Toolchain | Flags | Notes |
|---|---|---|
| gcc / clang (Linux, macOS) | `-ffp-contract=off -fno-fast-math` | As in `tools/build_qwen35_cli.sh`. |
| clang on Windows | `-ffp-contract=off -fno-fast-math` | The same flags. |
| clang-cl | `/clang:-ffp-contract=off /clang:-fno-fast-math` | |
| MSVC `cl` | `/fp:precise`, **without** `/fp:contract` and without `/arch:AVX2` | VS 2022 does not contract under `/fp:precise` unless `/fp:contract` is given. VS 2019 may contract once `/arch:AVX2` is set. MSVC has no "contract off" switch; leaving out `/fp:contract` is how you get it. Use VS 2022. |
| nvcc device code | `-fmad=false` | As on Linux. |
| nvcc host code (cl) | `-Xcompiler /fp:precise` | |

`exp`, `log1p`, `sin`, `cos`, and `pow` all go through `double` and are rounded once.
The Microsoft UCRT's versions can still differ from glibc's or Apple's in the last
bit. So on Windows, the token ids can drift from the Linux and macOS ids late in a run.
That is a finding to report, not a build failure.
