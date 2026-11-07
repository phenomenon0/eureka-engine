# Receipts: Windows kit for the engine CLI, 2026-11-06

These back [`WINDOWS-TEST-STATUS.md`](../../WINDOWS-TEST-STATUS.md) and
[`backends/windows/`](../../backends/windows/BUILD.md). **Nothing here ran on Windows.**
They show three things. The POSIX path is unchanged by `tools/llb_platform.h`. The
pinned 4B model URL is live. The macOS reference run matches what
`backends/windows/TEST.md` asks the owner to run.

Machine: Mac Studio, Apple silicon (arm64), macOS 27.0, Apple clang 21. The Linux
rows ran in Docker on the same Mac (`eepub-linux:1`, Ubuntu 24.04 arm64, glibc
2.39, clang 18, `--network none`). That image has no gcc, so `gcc` was a symlink to
clang there. Nothing here used gcc, MSVC, nvcc, or an NVIDIA GPU. "base" is
`dd824b9`, the parent of this change.

| What | Result | Receipt |
|---|---|---|
| `tools/build_qwen35_cli.sh`, macOS | exit 0 | `posix-ab-mac.txt` |
| `tools/build_qwen35_cli.sh`, Linux | exit 0 | `posix-ab-linux.txt` |
| A/B base vs this change, Qwen3.5-0.8B Q4_K_M (sha256 `bd258782…`), macOS: pack dir, `--head-only` pack, verify-against, ids and logit bits (mmap and `--preload`), error text | 9/9 PASS | `posix-ab-mac.txt` |
| The same A/B on Linux | 9/9 PASS; ids `279 6511 314 279 3046 13 198 760` | `posix-ab-linux.txt` |
| 4B Q4_K_M (sha256 `00fe7986…`): pack and CLI on macOS, the TEST.md reference | exit 0; ids `11751 13 198 32 13 2912 198 33` (" Paris.\nA. True\nB") | `run-cli-4b-mac.txt` |
| HF URL for the 4B pin | 200, 2740937888 bytes, LFS sha256 = pin | `hf-head-4b.txt` |
| Engine compiled as C++14 (host side of the `.cu`); no VLAs or GNU extensions (MSVC C) | clean | `portability-checks-mac.txt` |

Scripts: `scripts/posix-ab.sh GGUF WORKDIR [BASE]` and `scripts/linux-docker.sh`.
