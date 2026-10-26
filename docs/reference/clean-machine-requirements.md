# Clean machine requirements

This page lists what a clean machine needs to build and run the engine: the
compiler, the toolchain, the build chain, the models and packs, the launch
flags, and the machine paths that are still hard-coded.

The Evidence column uses these values:

| Value | Meaning |
|---|---|
| Checked | Read or run for this inventory on 2026-10-04. |
| From a report | Taken from a report. Not re-run. |
| Open | Not resolved. |

## Compiler

The Bend compiler emits the engine C.

| Item | Value | Evidence |
|---|---|---|
| Repository | `bendlang/bend` (origin) and `phenomenon0/bend` (the fork) | Checked (`git remote -v` of the shared checkout) |
| Pinned commit | `240ecef4b52a1efcc2151df689cca9a2a769aac6` ("foldshape: host Array.get skips the wrap mask in range (blk_get)") | Checked |
| Where the commit is reachable | Only local branches of the shared checkout: `lane-foldshape`, `lane-rt-extract`, `lane-sync-2.0.32`, `omen`, `sync-cert`, `upstream-sched-resume`. `git branch -r --contains` returns nothing. | Checked. An outside user cannot fetch the commit from `bendlang/bend`. It must be pushed to a public ref first. Open, owner action. |
| Files used from it | `bend2/{bend.ts,comp.ts,main.ts,hand.ts,export.ts,base.bend,effs/}`, copied by [`tools/build_gpu.sh`](../../tools/build_gpu.sh) | Checked |
| Patch stack | 54 patches applied by `tools/build_gpu.sh` with `patch --batch --fuzz=0 --forward -p1`. The list and order are the `patches=(...)` array in that file. 64 `tools/*.patch` files exist. The 10 not in the stack are Metal, older-quant, and experiment patches: `grow-cube7 iq4xs metal-qknorm payload-hand-pf payload-qnrms payload-silu q20 q3k q4k q6k`. | Checked |
| Patch check | `build_gpu.sh` unwinds all 54 patches on a scratch copy and refuses a compiler copy that does not match. | Checked |
| Canonical Bend | About 113 commits ahead. Merging fails 42/44 patches. A merge was not attempted. | From a report |
| Runtime | `bun` 1.3.4. No npm dependencies: only `node:*` imports. | Checked |

## Toolchain

These are the versions on the record machine.

| Tool | Version | Used for |
|---|---|---|
| bun | 1.3.4 | Runs the patched Bend compiler. |
| clang | 21.1.7 | Compiles the emitted C and the CUDA host ([`tools/remap_cc.sh`](../../tools/remap_cc.sh), `-O3 -ffp-contract=off -std=c11`). |
| CUDA toolkit | 13.1.80 (`cuda.h`, `libcuda`, `libnvrtc`). GPU kernels are compiled by NVRTC at run time for the `sm_86` target. | Link and run. |
| NVIDIA driver | See `/proc/driver/nvidia/version`. | Run. |
| gcc | 15.2.1 | Only the `qwen35-cli` CPU twin ([`tools/build_qwen35_cli.sh`](../../tools/build_qwen35_cli.sh)). Not the engine. |
| GNU patch, git, tar, sha256sum | Any | Build. |
| python3 and numpy | 3.14.2 and 2.4.3 | Pack tools ([`tools/quant_pack_qwen35.py`](../../tools/quant_pack_qwen35.py), `pack_index.py`, `pack_table.py`). |
| cmake | Not needed by the engine. Checked. | Not applicable. |
| GPU | RTX 3090, sm_86 | The only validated device. |

## Build chain

Checked.

1. [`tools/build_gpu.sh`](../../tools/build_gpu.sh) runs the patched Bend
   compiler on `engine_qwen35_gpu.bend` and writes `llb-engine.c`.
2. [`tools/remap_cc.sh`](../../tools/remap_cc.sh) compiles `llb-engine.c`
   with clang into `llb-engine`. The C includes `host/llb_host.c`.

The patches (`payload-dp4a.patch`, `payload-dp4b.patch`) embed the GPU
kernels in the C, including the q27c dp4a kernels (`tools/q27c/q4dp2.cuh` and
others). There is no separate `nvcc` step. The first run writes `<engine>.gpu`,
the NVRTC cubin cache.

The output of `./build.sh` from this tree, compiled under the output name
`c5-eng.c`, is byte-identical to the q27c record binary `build/c5-eng`
(sha256 `de5e0490...f20`). The C file name is embedded in the binary, so
`llb-engine` has a different hash (`ab1dfd4d...`) with the same code.

## Models and packs

| Item | Value | Evidence |
|---|---|---|
| 27B GGUF | `Qwen3.8-27B-Q4_K_M.gguf`, 18,973,870,528 B; sha256 `c600de03...a747`. The download source is not recorded in the repository. | Checked (size). Open (source URL). |
| 27B pack | `pack-q27a`: `pack.bin` 18,962,876,416 B, `layers/blk.N.bin` (64), `manifest.json`, `index.tsv` (877 lines), `table.bin` (64 rows x 32 words), `PACK_DONE`. 851 tensors, `pack_sha256 90f5f92c...`. | Checked |
| How a pack is built | `python3 tools/quant_pack_qwen35.py MODEL.gguf OUT_DIR` writes `pack.bin`, `layers/`, and `manifest.json`, verifies itself, and calls `pack_table.write`. `python3 tools/pack_index.py OUT_DIR` writes `index.tsv`. `Q35_PACK_LAYERS=0` skips `layers/`. `tools/gguf_convert.py` and `q4km_pack.py` are the older MiniCPM/F32 paths. They are not used for the 27B. | Checked (read). See the report for the run. |
| 9B | `Qwen3.5-9B-Q4_K_M.gguf`, sha256 `03b74727...b7e8` (`SHA256SUMS-w2e.txt`) | Checked |
| `pack_table` fix | Commit `77275e35` (offset < 2^28 check). The 27B pack needs it. | Checked (in this tree) |
| Shapes the engine accepts | Exactly the 9B (32 layers, d 4096, ffn 12288) and the 27B (64, 24 attn, d 5120, ffn 17408), per `mf.validate`. | Checked |
| RAM and VRAM | The 27B run uses `--gpu 40GB`. The argument is a host-side budget. 24 GB card. | From a report |

## Launch and flags

- The launcher [`tools/run27b.sh`](../../tools/run27b.sh) was untracked in the
  q27c worktree. It is copied into this branch.
- `tools/run27b.sh` sets `LLB_Q4_DP4A=9` (the default). Set `LLB_Q4_DP4A=0`
  for the exact tier.
- `tools/run27b.sh` execs `${ENG:-$LLB_ROOT/build/llb-engine}`. Its default
  engine path was changed from `build/c5-eng` to `build/llb-engine`, the
  output of `build.sh`.
- The 27B run form, from [`tools/q4dp/run27.sh`](../../tools/q4dp/run27.sh):

  ```sh
  ENGINE --gpu 40GB --threads 16 -- PACK_DIR TOKEN_IDS --steps N
  ```

  `ENGINE` is the engine binary, `PACK_DIR` is the pack directory,
  `TOKEN_IDS` is the list of prompt token ids, and `N` is the number of steps.
- `LLB_Q4_DP4A` mode 9 is for the 27B only. It is never gated on the 9B.
- The host also reads `LLB_WARP_GRID`, `LLB_Q4GLUE`, `LLB_DUMP`, and
  `LLB_DUMP_POS`. At build time it reads `LLB_HOST_DIR` (default `host/`).
- The prompt is token ids, not text. The Qwen3.5 tokenizer and chat CLI is a
  separate tool (`tools/qwen35_engine.c`, `build_qwen35_cli.sh`). From a
  report: it is not wired to the emitted engine.


## Limitations

- The pinned Bend compiler is published as the content record
  `pins/bend-240ecef4`. See [Run on CUDA](../guides/run-on-cuda.md#pinned-compiler).
- The download source of the 27B GGUF is not recorded. Open.
- The RTX 3090 (sm_86) is the only validated GPU.
