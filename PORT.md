# Port from llama-bend, 2026-11-06

This branch (`port/eureka-pub-20261106`) starts at eureka-engine
`fix/inference-source-recovered-20261105` (`f14c1f3`). It carries over the engine
work that landed on llama-bend `integration/canonical` (`e0b7d6f4`) after
eureka-engine was exported from it, wherever eureka-engine did not already have
that work in another form. Build and test receipts are in
[`receipts/port-20261106/`](receipts/port-20261106/README.md).

## Method

The two repositories share no commits, but eureka-engine's first commit
(`69193ec`) is an export of llama-bend. For each file that differs, the merge base
is the llama-bend commit whose copy of the file equals eureka-engine's first copy
byte for byte (or, for the 34 sanitized files, differs only in sanitized lines).
From that base:

- **llama-bend Δ**: the lines llama-bend changed after the export
  (`diff base canonical`).
- **eureka Δ**: the lines eureka-engine changed (`diff base f14c1f3`), including
  the export's sanitizing (private paths, repo names).

A file with llama-bend Δ = 0 is **eureka-newer**: llama-bend has nothing that
eureka-engine lacks. For the rest, every llama-bend commit was read and either
ported, merged by hand, or left out with the reason given.

The brief counted 43 differing files against `main`. This branch also differs
in `tools/clef3/json.c`, which `f14c1f3` hardened, so the table lists 44.

## Per-file decisions

| File | Base (llama-bend) | llama-bend Δ | eureka Δ | Decision | Evidence |
|---|---|---|---|---|---|
| `host/llb_host.c` | `0a0b40cb` | 23 | 992 | **merged** | Ported SK8 (`a36fad93`) by hand onto eureka's any-shape slot launches. Changed one thing: past SK8's 2^17-slot planes, the canonical `llb_attn4` runs instead of failing. llama-bend's guard could not trigger there, because its context stops at 2^17; eureka goes past 2^17. Not ported: `1f666a40`'s warpqr guard at d 2560, because eureka's `dfd == 12 && df <= 12288` (anyshape) already excludes the 4B's 9216. |
| `host/llb_pf.h` | `5de86bd1` | 8 | 166 | **merged** | `68741291` has two hunks. eureka already had the Q6_K embedding dispatch (anyshape fixed the same bug). Ported the other hunk, the `LLB_PREFILL=fast` admission guard that refuses before defer. |
| `tools/build_gpu.sh` | `5de86bd1` | 2 | 2 | **merged** | Appended `payload-sk8add.patch` after eureka's `anyshape` / `moeany` / `catchup`. Not added: `payload-w2560.patch` (superseded, below). |
| `tools/payload-sk8add.patch` (new) | — | new | — | **ported (rebased)** | Its added and removed lines are byte-identical to llama-bend's. Hunk offsets moved, and the `gpu_sk8` declaration now sits after catchup's `gpu_ganyk`. Applies at `--fuzz=0` as entry 66 of the stack. |
| `tools/sys1dual/dual.c` | `b47434d1` | 14 | 41 | **merged** | Ported `91d0cb72` (SENSES-MATRIX): NOHEAD readout and `"images"` rows at the trunk width, checked against the stamped `d`. Kept eureka's soft behavior in the one conflict: a trunk glue_out cannot read still serves generate and refuses readout instead of exiting. Kept eureka's anyshape `ids_fit()` prompt buffers. |
| `tools/sys1dual/serve.c` | `fcc2b414` | 45 | 145 | **merged** | Ported `91d0cb72` (tower rows stamped at their own width) and `d7ed24d4`: `--prefill exact\|fast` and `--prefix-cache`, opt-in, defaults unchanged, closed by llama-bend's SERVEFAST3 gates. Kept eureka's senses routes and `f14c1f3` cancellation work. The brief says eureka "replaced" a `prefill_fast` block. It did not: llama-bend added that block on 11-06, after the export, and eureka never had it. |
| `tools/glue/glue_out.c` | `8e6c03fd` | 28 | 0 | **llama-bend-newer** | Taken whole (`91d0cb72`): readout width from the GGUF, tied `token_embd.weight`. |
| `tools/glue/glue_out.h` | `8e6c03fd` | 6 | 0 | **llama-bend-newer** | Taken whole (`91d0cb72`): `glue_out_width()`. |
| `tools/vision/tower_stamp.py` | `0a0b40cb` | 10 | 0 | **llama-bend-newer** | Taken whole (`91d0cb72`): stamps the rows' own `d`. |
| `engine_qwen35_gpu.bend` | `5de86bd1` | 22 | 367 | **eureka-newer; llama-bend change not ported** | `1f666a40` / `5246fead` whitelist the 4B shape (32/16/2560/9216) and Q6_K embeddings at d 2560. eureka's any-shape admission is structural (any dense Qwen3.5 with head_dim 256, SSM state 128) and already admits Q6_K embeddings at any width. Also not ported: the tied-output fallback for "the original 4B Q8 pack", which only matters for a pack with no `output.weight` entry. This repo's `gguf2pack` writes that alias for every tied model (`tools/gguf2pack.c:1336`, `:1773`). |
| `backends/metal/build.sh` | `768bff35` | 7 | 50 | **eureka-newer; llama-bend change not ported** | `38c75c57` / `fe87228b` (08BMETAL) admit the 0.8B on the Metal fork by rewriting the base `metal-cc.patch` / `metal-engine.patch` / `metal-vision.patch` and the pinned hashes. eureka's Metal any-shape patches stack on those base patches at `--fuzz=0` and already run the 0.8B, 2B, 4B and 9B (`docs/limits.md`). Porting would break eureka's stack to re-add a subset of what it has. |
| `host/llb_backend_cuda.h` | `44bea56f` | 0 | 5 | eureka-newer | cuda-catchup decode-run graphs (`4822549`). |
| `host/llb_gguf.c` | `f4eb2375` | 0 | 63 | eureka-newer | Any qwen35moe GGUF (`abbc3f4`). |
| `tools/gguf2pack.c` | `543caec4` | 0 | 443 | eureka-newer | qwen35moe sidecars for CUDA and Metal (`abbc3f4`, `dfe1f9f`). |
| `tools/clef3/json.c` | `020c6c31` | 0 | 21 | eureka-newer | JSON hardening (`f14c1f3`). Differs only from this branch, not from `main`. |
| `tools/sys1dual/sys1.bend` | `543caec4` | 0 | 2 | eureka-newer | KV capacity is prompt + steps (`bb99c22`). |
| `release/install.sh` | `4f13d4fe` | 0 | 4 | eureka-newer | Release URLs point at eureka-engine. |
| `tools/release/publish-github.sh` | `4f13d4fe` | 0 | 6 | eureka-newer | Same. |
| `tools/check_no_gpl_tts.py` | `e163c21b` | 0 | 4 | eureka-newer | Private memo path removed. |
| `tools/verify_kernels.py` | `543caec4` | 0 | 2 | eureka-newer | Hard-coded Bend path became `$LLB_BEND`. |
| `tests/mm/mkfix.py` | `c8b28f45` | 0 | 2 | eureka-newer | Private fixture path removed (`/home/omen/lab/...`). |
| `tests/mm/run.sh` | `43978a22` | 0 | 4 | eureka-newer | Private clang and template paths removed. See "Build fixes". |
| `tools/glue/glue_rows.py` | `8e6c03fd` | 0 | 4 | eureka-newer | Private model path removed (`/mnt/storage/...`). |
| `tools/iq4xs.py`, `tools/q20.py`, `tools/q3k.py`, `tools/q4k.py`, `tools/q5k.py`, `tools/q6k.py` | `543caec4` | 0 | 2–5 each | eureka-newer | Private `libggml-base.so` path became `$GGML_BASE_LIB`. |
| `tools/llb/test/b7.sh`, `r4.sh`, `snips.sh`, `summary.sh`, `t3.sh` | `acda3d63` / `bc3897d5` | 0 | 2 each | eureka-newer | Private scratch paths became `$TMPDIR`. |
| `tools/metrics/w1load.py`, `tools/metrics/w1run.sh` | `6ee965e2` | 0 | 8 / 5 | eureka-newer | Private sidecar and venv paths removed. |
| `tools/mtp27/make_drafter.sh`, `tools/mtp27/mtp_pack.py` | `fac48167` | 0 | 4 each | eureka-newer | Private model paths removed. |
| `tools/q4dp/run27.sh`, `tools/run27b.sh` | `a169c561` / `5bdfd706` | 0 | 2 each | eureka-newer | Private model paths now required as env vars. |
| `tools/sys1dual/build-serve.sh`, `build-structured.sh`, `build.sh`, `env.sh` | `85a308ff` / `43978a22` / `7069e2c7` / `b47434d1` | 0 | 2–8 each | eureka-newer | Private clang, sidecar and work paths became env vars. |
| `tools/toolcall/run_tests.py` | `6d046852` | 0 | 4 | eureka-newer | Private template path removed. |

Totals: 3 llama-bend-newer and taken whole, 5 merged by hand (plus 1 new patch
file), 2 where llama-bend's change is superseded, and 34 eureka-newer.

## The three named priorities

| Item | Result |
|---|---|
| `a36fad93` split-key decode attention (SK8, default off) | **Ported.** `LLB_ATTN_SK8=1` opts in; `--canon` forces it off. Rebased patch, host dispatch with a fall-back past 2^17 slots. |
| `metal/08b-d1024-20261106` (`23ea5bf6`, `38c75c57`, `fe87228b`) | **Not ported: superseded.** eureka's Metal any-shape build already runs the 0.8B. See `backends/metal/build.sh` above. |
| `perf/4b-kquant-speed-20261106` (`5246fead`, `ad2d2220`, both inside `payload-w2560.patch`) | **Not ported: superseded.** eureka's `payload-catchup.patch` makes the same partial-quad change to `kp_pair`, generalized from llama-bend's `cg == 640` special case to any `cg % 64 == 0`. eureka's `payload-anyshape.patch` already has the 2560 rms tree (any n ≤ 8192 but 2048 / 4096) and the structural envelope. Its catch-up report times that change alone at 20.7 → 7.6 ms per 4B token in development runs, and measures 4B decode at 47.8 → 135.3 tok/s across all the catch-up changes (`reports/cuda-catchup/REPORT.md`, `receipts/speed/SPEED.txt`). |

Both branches are fully merged into `integration/canonical`. Neither adds commits
of its own.

## Deliberately not ported

- `receipts/`, `reports/`, `lanes/`, lane logs, `BRIEF.md` / `PLAN.md`, and the
  per-experiment branches, as the brief requires. The SK8, SENSES-MATRIX and
  SERVEFAST receipts stay in llama-bend.
- `tools/payload-w2560.patch`: superseded, see above.
- llama-bend-only compiler patches that neither build list uses:
  `grow-cube7`, `metal-qknorm`, `payload-hand-pf`, `payload-qnrms`,
  `payload-silu` (all from the `543caec4` snapshot of 10-13).
- About 330 other llama-bend-only sources (`tools/pf27`, `tools/pkg3`,
  `tools/clef1`/`clef2`, `tools/q27*`, `tools/integ`, and so on). None was
  touched after 2026-10-25, which is before or at the export. They are lane
  tooling and earlier generations that eureka-engine's "Keep the engine only"
  commit (`6897fe5`) dropped on purpose.

## Where llama-bend is ahead and this port does not follow

- **Metal vision at d 1024** (`fe87228b`). eureka's `VISION=1` Metal build stops
  at the any-shape step, and its native link is "not carried yet"
  (`backends/metal/build.sh`). Porting the 0.8B image-row fix means rebasing
  `metal-vision.patch` onto any-shape, and nothing on this branch can gate that.
- **The SK8 and 4B CUDA measurements.** llama-bend measured SK8 and the 4B
  CUDA work on an RTX 3090. This port is compile-checked only: the kernel compiles
  under NVRTC for sm_86 (`receipts/port-20261106/nvrtc-sm86-port.txt`), but no
  GPU ran it on top of eureka's stack. SK8 is off by default, so the default
  numerics are unchanged.

## Build fixes

None of the port's files needed a fix to build. One unrelated test-harness gap
was found: `tests/mm/run.sh` calls `mkfix.py` without the COCO image that
`mkfix.py` requires. The export dropped llama-bend's private default path and put
nothing in its place, so the suite stopped at its fixture step with a usage
error. `run.sh` now takes the image as `MM_COCO`.
