# Windows test status: engine CLI

As of the commit that adds this file (2026-11-06):

| | |
|---|---|
| Builds on Windows? | **Unknown: not built.** No MSVC, clang-for-Windows, or nvcc build has been run. PowerShell has not parsed `backends/windows/build.ps1`. |
| Runs on Windows? | **Unknown: not run.** |
| Windows support claimed? | **No.** |

## What exists

- [`tools/llb_platform.h`](tools/llb_platform.h): a portability layer for the file,
  mapping, and clock calls in `tools/qwen35_engine.c` and `tools/gguf2pack.c`.
  Both files now go through it. On POSIX the behaviour is byte-identical. That was
  checked by A/B against the previous commit on macOS and on Linux (glibc), covering
  pack bytes, token ids, and logit bits:
  [`receipts/windows-kit-20261106/`](receipts/windows-kit-20261106/README.md).
  The Windows branch was written to the Win32 and UCRT documentation and has
  **never been compiled**.
- [`backends/windows/BUILD.md`](backends/windows/BUILD.md) and
  [`build.ps1`](backends/windows/build.ps1): the CPU build (clang or MSVC) and the
  CUDA build (nvcc, with the arch read from `nvidia-smi`).
- [`backends/windows/TEST.md`](backends/windows/TEST.md): the model download (the
  pinned 4B), the pack step, the run, and a macOS reference output to compare against.

The only earlier Windows datapoint is in `docs/backends.md`: 2 tokens in 128 s on
CPU, in a Windows Server 2025 VM. That build had no recipe in the tree, and this
kit does not reproduce it.

## What the owner must report back

From the steps in `backends/windows/TEST.md`:

1. The `built:` / `FAILED:` summary from `build.ps1`, and the full compiler output
   for any target that failed.
2. The `gguf2pack` exit code and its last two lines (`pack_sha256` should be
   `86be153f…`).
3. `qwen35-cli-gpu.exe ... --gpu`: the exit code, stdout (token ids, `wall_ms`,
   `tok_s`), stderr, and `nvidia-smi`.
4. Optionally, `--selftest` (`bits_diff`) and the CPU run (`wall_ms`).
5. Versions: `nvcc --version`, the CPU compiler's version, and the Windows version.

**Pass:** exit 0, and token ids printed. The ids do not have to match the
macOS reference. When they differ, report it as a finding, not a failure.

Until those results come back, the status above stays as it is.
