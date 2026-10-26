# UNIFY1 final (CUDA, RTX 3090), 2026-10-05

Branch `unify1`, from integ1 `2f9d7ac4`. One CUDA build of the 27B Q4_K_M engine. Every feature sits behind a flag and is gated (`reports/unify1-gates.md`, registered in `a40f6f9e` before any run). It is packaged as `release/llb-linux-x86_64-cuda-0.1.0.tar.gz`.

## Carried / excluded

| item | status | reason / receipt |
|---|---|---|
| s1 gguf2 `542130c2` (direct GGUF load, `LLB_GGUF`, `gguf2pack --sidecar-only`) | CARRIED (`d162f6f2`) | R0, R1 PASS |
| s2 PF28 `pf28-set.patch` (Q5_K wide leaf + PF28 fast-prefill levers) | CARRIED (`bf88a2f5`) | R2 KILL gate PASS under mode 9 |
| s3 mtp28 `fe20f039` (L1: spec small kernels, T rows a launch) | CARRIED (`ae42888c`) | R0, R1, R4, R5, R6 PASS |
| tokenizer (`tools/qwen35_toktest.c` → `llb-tok`) | SHIPPED as a separate tool | 313/313 recorded prompts and 1,580/1,580 gate strings vs llama.cpp (`receipts/unify1/tok/summary.txt`). The engine still takes ids. |
| s4 35B MoE (`moe2b` `feb75f6c`) | **NOT CARRIED: STOP rule** | `git merge-tree` conflicts in `engine_qwen35_gpu.bend`, `host/llb_host.c`, `tools/build_gpu.sh`, `tools/step2a_stub.h`. The 480-line `payload-moe.patch` needs hand-rebasing over payload-27b, pf and spec2, and its gate families (llama-coded decode gate, moe2 G1–G5 twins, ≥1k timing, long-context peak) are not registered. That is well over ~2 h. No 35B number is quoted; the 1.88x was never timed here. Priced plan below. |
| 9B Q8 / Q4_K_M | flag-off byte gates only (R0) | Owner STOP: no 9B timing. 9B fast prefill is OFF and not supported. |
| sys1b, MTP28 L2, Mac MTP, nine-q4km2 | not carried | per the brief |

## Gates (receipts under `receipts/unify1/`)

| gate | result | receipt |
|---|---|---|
| R0 byte set (flag-off regression, internal) | 12/12 EQUAL at s1, s2 and s3 | `s{1,2,3}/bytes.log` (s3: `s3/bytes`) |
| R1 GGUF == pack, both-on, 14 prompts × 64 | PASS at s1 (tokens `5b6d8f25f570`, 924 logits `7cc8efe0b5c4`) and at s3 (`3eb336baac75` / `080841c0c62e`) | `s1/gate/r1-*`, `s3/gate/r1-*` |
| R2 PF28 KILL gate, mode 9 | PASS. G1v2 margin Δ rms 0.0594 ≤ 0.142; 1 flip, at exact margin 0.0096 < 0.237; 3/20 prompts diverge (margins 0.0070 / 0.0098 / 0.0165) ≤ 11. G2: KL ratio 1.150, \|Δlogprob\| ratio 1.020 (≤ 1.2), over 2,560 positions | `s2/r2.txt`, `s2/g1v2`, `s2/g2` |
| R3 memory | see below | `s3/mem.log`, `s3/gate/*.mem`, `base/gate/*.mem`, `time/*-off.mem` |
| R4 fast vs both-on, 14 × 64 | PASS at s2 and s3, 924/924 lines | `s{2,3}/gate/r4-*` |
| R5 spec k=2 / k=3 vs plain | PASS: tokens `23e48df736f4`, 1,532 logits lines `507830fe44e4` | `s3/gate/r5-*` |
| R6 combined (fast + spec k=2 vs fast), 5 seeds × N 200 / 1009 / 3011 × 64 | PASS: tokens `0fe78f23642e`, 990 logits lines `298f9cf37c19` | `s3/gate/r6-*` |
| R7 G5 decode vs i1-eng | 0.999 [0.996..1.001], ≥ 0.97 PASS | `time/table.txt` |

Changes to the emitted C, per step:
- s0 → s1 `4216b6aa`: equal to gguf2's.
- s1 → s2 `2d6d3114`: only the PF28 fast-prefill code (pf_* kernels, PF_WIDE, pf_chunk/pf_mm/pf_sm/pf_fast, the gpu_load pf allocations, the gpu_pf* CUfunctions).
- s2 → s3 `c1f2b74e`: adds the mtp28 L1 spec T-row code (spec_attn/spec_ffn/spec_rows/spec_init/mtp_rows/spec_load, the sp_* buffers, pf_mark, pf_dump).
- Final engine: `f9b96578` (C `c1f2b74e`), equal to s3. Only scripts and docs changed after s3.

### Token agreement with llama.cpp (teacher-forced, 2,560 positions, sets g1 + g1v2; `s2/agree.txt`)

| tier | llama top-1 == engine token |
|---|---|
| exact (mode 9) | 2,530 / 2,560 (98.83%) |
| fast prefill (mode 9) | 2,528 / 2,560 (98.75%) |

The engine's own dump top-1 matched its emitted token 2,560/2,560 (parser check).

For comparison, llama.cpp against itself (its own prefill path vs its decode path, pf27): 9/20 prompts diverge within 64 tokens, with handoff margin rms 0.1187. Our fast vs exact: 3/20 diverge, margin rms 0.0594.

## Timed check (one check, 3 interleaved rounds, llama.cpp `llama-bench` in the same rounds; median [min..max]; `time/table.txt`)

| row | value | ratio vs llama.cpp |
|---|---|---|
| decode, mode 9, p1 × 64 | 38.77 tok/s | 1.011x [1.009..1.012] vs tg128 38.34 |
| fast prefill, N = 1009 | 1,296.9 tok/s | 0.986x [0.960..0.996] vs pp1009 1,315.3 |
| fast prefill, N = 3011 | 1,301.2 tok/s | 0.953x [0.950..0.953] vs pp3011 1,366.2 |
| spec k=2, s3.ids × 64 | 65.39 tok/s | 1.705x [1.700..1.707] vs plain tg128 (different prompts and contexts: indicative only) |
| load, GGUF cold | 17,879 ms [16,681..18,000] | |
| load, pack cold | 44,655 ms [44,233..45,245] | |
| load, pack with head pre-read | 47,707 ms [45,635..64,036] | the pre-read does not help |

No row is under 0.9x. PF28's earlier 0.911x is superseded by the re-timed 0.953–0.986x.

## Peak memory (card_used_peak MiB, card otherwise empty; limit 21,458)

| combination | peak |
|---|---|
| off (mode 9) | 19,011–19,069 |
| fast prefill (the new default) | 19,729–19,819 |
| spec k=2 / k=3 | 19,863 / 20,021 |
| **both-on (fast + spec k=2), N ≤ 3,011** | **21,393–21,433: only 25–65 MiB under the limit** |
| **both-on, N = 8,192** | **DOES NOT FIT.** With `--gpu 40GB` the Bend heap span runs out; with 80GB the card reached 24,109 MiB and the host cgroup OOM-killed it at 12 G. integ1's base fails the same way. |
| fast-only / off, N = 8,192 | span out of memory at `--gpu 40GB` (`memf-8192`, `memo-8192`) |

Both-on is not a default. The tested context ceiling is 3,011 + 64 for every combination.

## Defaults (by evidence)

- **mode 9 (`LLB_Q4_DP4A=9`): ON.**
- **fast prefill: ON** in `tools/run27b.sh` and in the bundle's `run.sh`; `--prefill exact` turns it off. Reasons: R2 PASS, about 33x faster time to first token, 19.8 GB.
- **`--gguf`: the recommended path.** R1 gives identical tokens and logits, cold load is 2.5x faster, and no 38 GB pack is needed.
- **spec: opt-in.** Its drafter file has no recorded provenance and is not distributable, and both-on sits within 25–65 MiB of the limit.

## Release

- Fresh clone built with the README commands only, in `~/unify1-build/llama-bend` (`release/fresh.sh`, `release/fresh.log`).
  - Result: engine `f9b96578…`, C `c1f2b74e…` (equal to s3), gguf2pack `1413e086…`, llb-tok `60645971…`.
  - Size: 168,416,458 bytes (210 MB on disk). Deleted afterwards.
- Tarball: `release/llb-linux-x86_64-cuda-0.1.0.tar.gz`.
  - sha256 `8e70a4aa6968f73a2a12bbb252f64ce6a18a6a1ea243109eb3b42a0d1ab361f1`, 1,040,782 bytes.
  - Contents: binary, `llb-engine.c` (its NVRTC kernel source), `build.sh`, `run.sh`, `gguf2pack` (+ .c), `llb-tok` + tokenizer source, README, `SHA256SUMS.files`.
  - Built from the fresh-clone binaries. The engine commit is stamped in the README (`8e55a055`).
- Unpack test: unpacked in a fresh dir, `SHA256SUMS.files` all OK. The sidecar was made by the bundled `gguf2pack` and is byte-equal to `/data/gguf2/sidecar/q27`. Both runs went through the bundle's `run.sh` with its defaults on the GGUF path; the first run compiled `llb-engine.gpu`.
  - 64-token decode, s1.ids prompt 1: tokens equal the s3 `r4-fast` stream 1 (`release/unpack-dec.*`).
  - 1,009-token prefill (time.sh seed r=1): first token 220, equal to `time/1-fast` (`release/unpack-pre.*`).
  - The unpack dir was deleted afterwards. The `.mem` samplers read 0 because they match the engine path, not `run.sh`; memory is covered by R3.
- `publish-artifact.sh`: tested only on a copy of `llama-bend-review/dist` in a temp dir with `LLB_SITE` pointing at the copy.
  - It published, the manifest entry was rewritten (`release/publish-test.manifest.json`), and it rendered 3 cards.
  - The real `site/manifest.json` sha256 was unchanged (`release/publish-test.diff` lists only the copy's changes). The copy was deleted.
- License: none was picked at the time of this report; the repository is now Apache-2.0.

## UNKNOWNs

- Minimum driver, CUDA and compute capability. MEASURED: driver 580.119.02, CUDA 13.1.80, cc 8.6, clang 21.1.7, Fedora 43 / Linux 6.17. Only sm_86 has been run.
- Chat-template fidelity: `llb-tok` applies none.
- Provenance of the MTP drafter (`mtp-Qwen3.8-27B-Q4_0.gguf` → `blk64-wip.bin`).
- Download source of the 27B GGUF (its sha256 is recorded).
- Host RAM minimum: the runs used a 12 G cgroup.
- `/dev/shm` was at 397 MB at close, all `u1000-Shm_*` segments that are not from these runs (not touched).

## Deviations (said plainly)

- llama-bench ran with `-mmp 1`; integ1 used `-mmp 0`, which was OOM-killed under the 12 G unit cap. Because of the restart, round 1's llama arm ran after round 2's spec arm (`time.notes`).
- The GGUF-warm load arm was dropped: the 12 G cap charges the page cache.
- The s1 9B q4/q8 `.mem` values are polluted, because same-binary runs overlapped the poller. Bytes are unaffected.
- The first R2 exact run timed out at 240 s: `TO` was not passed into the unit. Fixed in `tools/gguf2/run.sh` and re-run.
- The pf_tk/pf_cat rename is carried as in pf28-set.patch.
- "Byte-identical" is used only as the internal flag-off regression statement (R0), not as a headline.
- The 35B MoE was not attempted beyond the merge-tree probe.

## MoE priced plan (not done)

Steps (CARRY-PLAN row 81):
1. Hand-rebase the 6 `payload-moe.patch` hunks over `payload-27b.patch`.
2. Resolve ~10 conflict blocks (~100 lines) plus the gguf2 engine/host overlap.
3. Register, then run: byte set, a llama-coded decode gate (control: llama.cpp CUDA vs CPU), moe2 G1–G5 twins, ≥1k-context timing, long-context peak under 21,458 MiB.

Estimate (not measured): about 1 working day, of which ~4–6 h of card time under the lock.

## Next 3 steps

1. Owner: pick the license, and publish compiler commit `240ecef4` and the repo to a public remote. Until then the source path cannot be reproduced outside this machine.
2. Fix the N = 8,192 limit: size the Bend heap span and host buffers so long prompts fit under 22.5 GB, or document a hard cap in the engine.
3. The 35B MoE carry, per the priced plan above, as its own branch with registered gates.
