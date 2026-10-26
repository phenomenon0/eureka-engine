# Metal build internals

This directory builds `sys1mac`, the Mac product engine for Metal, from this tree. It is step 1 of the unification plan (G7 step 1), which is described in `UNIFICATION.md` in fabric (`runs/gap-chase/G7/`).

The Metal stack is vendored as-is. The tree that it reproduces is the tree that the Mac ships. The Metal stack has not been rebased onto the CUDA stack yet. That is step 2 of the plan.

## Build the Metal engine

To build the Metal engine, run one of these commands:

```sh
LLB_BACKEND=metal ./build.sh
```

```sh
bash backends/metal/build.sh
```

If `LLB_BACKEND` is unset or set to `cuda`, `./build.sh` runs the CUDA build exactly as before. Nothing on the CUDA path reads `backends/`.

## What the build reproduces

The following hashes were checked on 2026-10-14.

| Build | Emitted C | Binary | Checked on |
|---|---|---|---|
| Default | `c13b9e28…` | `1cdd8854…`. The 6 objects are also equal (6/6). | Linux (emit) and the M3 Ultra (emit and link). |
| `VISION=1` | `fbdefec7…`. This is the ENGINE-IPHONEVIS emit `e15c60e0` plus 8 host lines for the 4B model's d 2560. | Not carried. The tower sources (D 64 + 72) live in `~/engine-iphonevis/vision/host`. | Linux (emit) and the M3 Ultra (emit and link, SENSES-MATRIX). |

These are the hashes that the `REPORT.md` of engine-vision3 records for the product. `build-ios.sh` pins `SRC_SHA = c13b9e28`. The script refuses any other emitted C.

## Build steps

The build script performs these steps:

1. Export the pinned Bend `bend2`. The commit is `BEND_COMMIT` in [`build.sh`](build.sh).
2. Apply the patch list of [`tools/build_gpu.sh`](../../tools/build_gpu.sh) up to and including `tools/gemv-q8-rw.patch`. These are entries 1–36. This point is the fork point. These 36 files are byte-identical to the files of the Mac integrate stack (sha256 compared 36/36).
3. Apply [`cc/metal-cc.patch`](cc/metal-cc.patch) (+9494/−20, essentially additive). This patch is the Mac compiler delta from the fork point, collapsed into one patch. It contains:
   - integrate `metal-port`, `metal-9b`, and `metal-batch`.
   - M36 release `cc/01–14`.
   - M38, M41, and `metal-sys1mac`.
   - The 4B stack: `metal-4b`, `kev`, `4bpf`, `dchunk`, and `reopen`.
4. Make a scratch copy of the engine sources and apply [`engine/metal-engine.patch`](engine/metal-engine.patch) with `patch -E`. The patch does three things:
   - It turns the engine of this tree into the Mac engine: M36 engine `01–11`, M41, `engine-sys1mac`, and the 4B engine patches through `engine-stop`.
   - It removes the host seam files (`host/llb_host.c`, `llb_backend*.h`, `llb_pf*`), because the Mac fork predates `payload-host.patch`.
   - It pins `tools/sys1dual/sys1.c` to the S1D copy (`llama-bend-sys1dual @ 7e61bda`).
5. If `VISION=1` is set, also apply `vision/{engine-mmproj,metal-imrope,metal-vision}.patch`. These patches are verbatim copies of the patches of engine-iphonevis, with the same sha256.
   - `engine-mmproj` and `metal-imrope` are equal to those of engine-vision3.
   - `metal-vision` is the 4B port of ENGINE-IPHONEVIS (`b40ab16`). The image splice launches <= 1024 columns and accepts any d % 256 == 0. For d 4096, this is 4 x 1024 launches, the same dispatches as before. For d 2560, this is 1024 + 1024 + 512.
   - Gates (SENSES-MATRIX, this emit linked on the M3 Ultra): 4B image-in is 32/32 exact on small277, invoice, and photo640 compared with llama.cpp mtmd. The default emit stays `c13b9e28`.
6. Emit C. This step is platform-independent. On Darwin, the script then links exactly as the `mac/build.sh` of engine-vision3 does. `link/dual.c` is the S1D pin, because the `dual.c` of this tree has changed since then, with twoheads S3.

7. Unless `LLB_METAL_BASE=1` is set, apply `cc/metal-anyshape-cc.patch` and `engine/metal-anyshape-engine.patch` (after the vision patches when `VISION=1`), and patch a scratch copy of `link/dual.c` with `link/dual-anyshape.patch` before the link. The emitted C is then `00979448…` (`VISION=1`: `85c8d3c4…`). See [README.md](README.md#any-shape).

## Why the build uses collapsed patches

The original Mac recipe read a patch directory that no longer exists, so it cannot rebuild from scratch. Only a materialized source tree remains. The two patches in this directory are the diff of that tree against this tree. With them, the Mac product is reproducible again, from git.

## Pinned files

[`SHA256SUMS`](SHA256SUMS) pins every vendored file.

`base/` also pins the engine inputs that the merge train has moved since the G7 base:

- `host/llb_host.c`
- `host/llb_pf.h`
- `host/llb_backend_cuda.h`
- `tools/sys1dual/sys1.c`
- `host/llb_gguf.c`, at `14d69135`. This is before the mmproj section of FUS-A. `vision/engine-mmproj.patch` adds that section itself.
- `engine_qwen35_gpu.bend`, at `768bff35`. This is before the d 1024 guards of E5-08B `5de86bd1`. Without this pin, `metal-engine.patch` fails 2 of 21 hunks on every canonical from `5de86bd1` (FOLD-CONTINUE, 2026-10-18).
