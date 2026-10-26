# Metal vision patches

The `VISION=1` patches of the Mac Metal engine (`sys1mac`, 9B lineage), and
two opt-in patches that are vendored here but not applied by `build.sh`.

| File | Applied by `../build.sh` | What it is |
|---|---|---|
| `engine-mmproj.patch`, `metal-imrope.patch`, `metal-vision.patch` | Yes, with `VISION=1` | The vision splice. See [build-internals.md](../build-internals.md). The iOS 4B engine carries byte-identical copies in [`../../ios/engine/patches/`](../../ios/engine/README.md). |
| `tower-speed.patch` | No | Two faster attention and GEMM kernels for the Metal vision tower, plus a serve-mode cap on image tokens. The tower's output is bitwise unchanged. |
| `clef-encode-err.patch` | No | `clef_encode` returns its rejections to the caller instead of exiting. |

All five files are pinned in [`../SHA256SUMS`](../SHA256SUMS).

## tower-speed.patch

The patch changes `vision/host/llbv.c`, `vision/host/vitm.m` and
`vision/host/vitm.metal`. It is 28,992 bytes, SHA-256 `8f76fa19…`.

### Base

It applies to the Metal tower sources of the fus-m-v46 / ENGINE-VISION3
line, with these blob hashes:

| File | Before the patch | After the patch |
|---|---|---|
| `vision/host/llbv.c` | `6f7c3477…` | `816df550…` |
| `vision/host/vitm.m` | `8929e671…` | `1659ebf7…` |
| `vision/host/vitm.metal` | `16f2e20c…` | `ee66d21f…` |

These tower sources are not in this repository. `build.sh` does not build the
tower (see "What the build reproduces" in
[build-internals.md](../build-internals.md)).
The 4B tower in [`../../ios/engine/vision/host/`](../../ios/engine/vision/host)
is a different revision, and the patch does not apply to it.

### Source

- `receipts/tower-speed.patch` of the TOWER-SPEED lane, commit `cb871bf`.
  It is the diff of `cb871bf~1..cb871bf -- vision/host`, byte for byte.
- It was folded into fus-m-v46 on branch `engfold`, commit `f6ddbdf`
  ("ENGFOLD stage 2"). There it applied cleanly, and the result is the same
  three file hashes.

### Kernels and flags

| Change | Default | Control flag | Bitwise |
|---|---|---|---|
| `vt_fa3`, an attention twin of `vt_fa`. Each simdgroup owns 8 whole query rows, so there are no threadgroup barriers. P·V covers only the 72 real columns, and O stays in registers. The block size, the chunk order, the softmax and the `simd_sum` lanes are unchanged. | On when N ≥ 512 patches. Below that, `vt_fa` runs as before. | `VITM_FA3=0` uses `vt_fa` at every N. `VITM_FA3=<nsg>` uses `vt_fa3` at every N. | Yes, it equals `vt_fa`. |
| `vt_mm_f32_kl`, a template twin of `vt_mm_f32` for k % 32 ≠ 0 (only `ffn_down`, k = 4304). It bounds-checks only the last k block. | On | `VITM_MMLAST=0` | Yes |
| `LLB_VIS_MAXTOK=<n>`, a cap on tokens per image. It applies to `LLB_VIS_IMAGE`, where it is checked before the tower runs, and to `LLB_VIS_EMBD`. An image over the cap is refused through the existing `vision: no image rows` exit. A value of `0` or a non-number is refused. | Unset, so there is no cap. The lane recommends 2048, but the value is left to the owner. | | Not applicable |
| `VITM_PROF=1`, a per-kernel-class GPU profiler. Each dispatch gets its own command buffer. | Off | | Yes |
| `VITM_FA2=<qt>,<nsg>`, development variants of `vt_fa2`. The `VT_SAB`/`VT_SAB2` sabotage ifdefs are inert. | Off | | |

The adopted kernels change the speed, not the numerics. `VITM_FA3=0
VITM_MMLAST=0` is the pre-patch kernel set, and the lane used it as the A arm
of its timing.

### Measured

These numbers are on an M3 Ultra (60-core GPU). Each is the median of 5 warm
runs, interleaved ABBA with R ≥ 5. A is `VITM_FA3=0 VITM_MMLAST=0` on the
same binary, and B is the patch's defaults. An A/A control differed by
0.14 %.

| Image | Image tokens | Before (ms) | After (ms) | Change |
|---|---|---|---|---|
| small277 | 63 | 26.9 | 25.5 | −5.0 % |
| photo640 | 300 | 102.8 | 95.2 | −7.3 % |
| tall333 | 470 | 172.3 | 154.9 | −10.1 % |
| invoice | 640 | 249.8 | 219.9 | −12.0 % |
| page1472 | 2116 | 1411.9 | 1114.8 | −21.0 % |
| big3000 | 4015 | 4188.9 | 3137.4 | −25.1 % |

On its own, the attention stage was 28 % faster at 640 tokens (82.9 to
59.7 ms) and 31 % faster at 4015 tokens (3189 to 2191 ms).

The gates were registered before any GPU run, and all of them passed:

- Tower embeddings are bitwise equal to the reference tower on 6/6 images,
  with the flags on and off.
- Captions are 32/32 token-exact against llama.cpp mtmd on 3/3 images, in
  both the fast tier and the exact tier.
- Text-only output is byte-identical to `c13b9e28` on 9/9 runs.
- The `LLB_VIS_MAXTOK` refusal tests pass 5/5.
- A sabotaged `vt_fa3` is caught on 5/6 images. On small277 the flip is
  rounded away, and `vt_fa3` does not run there by default. A sabotaged
  `vt_mm_f32_kl` is caught on 6/6.

After the fold into `engfold` (`f6ddbdf`), the emitted C was unchanged
(`1ae0f14f`). The tower gates passed 6/6, captions passed 3/3 in each tier,
text was identical 9/9, and the cap tests passed 5/5. `LLB_VIS_MAXTOK` was
left unset.

The tower is still about 3.1× slower than CUDA (219.9 ms against 71 ms at
640 tokens). The lane attributes the rest of the gap to MMA throughput,
because both terms run at about 14 TFLOPS. Closing more of it needs a
precision change.

## clef-encode-err.patch

This is commit `27bec70` of fus-m-v46 (clef-upstream). The signature becomes
`clef_encode(tok, req, r, max_length, err) -> int`. Every input that it
rejects is returned through `err`, and the partial record is freed.

- With `err == NULL`, which every CLI passes, the behavior is the same as
  before: `clef_fail` is called, it prints `clef: MSG`, and the process
  exits 1.
- With `err != NULL`, which a serving caller such as
  `server/src/dualm.c` passes, `*err` is set to the message and the function
  returns -1. It does not exit.

The patch replaces the Mac server's post-copy `apply_clefjmp.py` patcher.

It was not in this repository: `tools/clef3/spans.c` still has the 4-argument
form. The paths in the patch are `S1D/tools/...`, as upstream. Checked with
`git apply --check` against this tree, with the `S1D/` prefix removed:

| File | Target here | Result |
|---|---|---|
| `clef3/spans.c`, `clef3/spans.h`, `clef3/decide.c` | `tools/clef3/` | Applies, with line offsets |
| `sys1dual/dual.c` | `link/dual.c`, the S1D pin that `build.sh` links | Applies cleanly |
| `clef3/head-test.c` | `tools/clef3/head-test.c` | Does not apply. The call there differs. Add `,NULL` by hand. |

The patch is not applied, because `tools/clef3/` and `link/` are outside
this directory.
