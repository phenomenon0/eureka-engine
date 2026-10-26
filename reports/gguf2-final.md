# GGUF2 final: the GPU engine loads a GGUF directly (`LLB_GGUF`)

Branch `gguf2` (worktree `llama-bend-gguf2`), base INTEG1 `630d79b2` + gguf1 carried (`2933eab7`). RTX 3090, CUDA. Every engine and GPU run went through the bench lock `/tmp/llb-gpu-bench.lock` with an empty card, each in its own transient `systemd-run --user` unit (MemoryMax=12G, MemorySwapMax=0, Nice=10). There was one run per lock hold.

| stage | commit | status |
|---|---|---|
| S0 plan + PREDICTION | `ff36340e` (+ carry `2933eab7`) | done |
| S1 loader + unit/mutation test | `d08908aa` | done, tested |
| S2 sidecar `--sidecar-only [--sha]` | `85d0cd33` | done, tested |
| S3 engine wiring | `1dc8c3d7` | done, tested (by S4) |
| S4 gates G1-G4 | `b30dccb2` | done, **all PASS** |
| S5 README + `docs/GGUF2.md` + patch | `9d5b3119`, `d53300f5` | done, patch checked |
| S6 this report | this commit | done |

## S1: loader (`host/llb_gguf.c`, `tools/gguf2/unit.c`)

The CPU test, against the existing packs:

| model | tensors matching the pack sha | mutations caught | notes |
|---|---|---|---|
| 9B Q4_K_M | 427/427 | 3/3 | 48 Q8_0 tensors; throughput sha-bound, ~300 MB/s |
| Kev-4B Q8_0 | 428/428 | 3/3 | HDD, ~70 MB/s |

- Mid-row virtual ranges in every layer equal the layer slice.
- The mutations were: a flipped output byte, Q8_0 copied raw without the row-split, and a tensor read 1 byte off.
- Receipts: `reports/gguf2/unit-9b-q4km.txt`, `unit-kev4b.txt`.

## S2: sidecar

`gguf2pack --sidecar-only` was run on all four GGUFs (27B, 9B Q4_K_M, 9B Q8_0, Kev-4B):
- Each took 0.8-1.3 s and read 11,538,279 bytes (strace).
- `table.bin` is byte-identical to each pack's.
- `manifest.json` is identical once `source.file` and the sha256 fields are normalised.
- With `--sha` (Kev-4B), the manifest is byte-identical without normalising.
- Receipts: `reports/gguf2/sidecar.txt` and `sidecar-kev-sha.txt`. `sidecar.txt` predates the fix that suppressed a bogus "MB/s" line in header-only mode; that line is cosmetic.

## S3: engine wiring

| file | change |
|---|---|
| `engine_qwen35_gpu.bend` | +17/-8 |
| `gguf.bend` | new, 22 lines |
| `host/llb_gguf.c` | new, 564 lines |
| `host/llb_gguf.js` | new, 4 lines (stub) |
| `llb_host.c` | 0 |
| build scripts | 0 |
| compiler | 0 |
| kernels | 0 |

The emitted C gains two CIDs and four FIDs, and `WS_LOAD_SKIP` is now inlined (`reports/gguf2/emitted-c.txt`).

## S4: gates (`tools/gguf2/s4.sh`, `s4cmp.py`, `g4.sh`; verdicts `reports/gguf2/s4cmp.txt`; receipts `receipts/gguf2/s4/`)

### G1: device bytes

`LLB_GGUF_VERIFY` reads each region back with `cuMemcpyDtoH` after `io_put` and takes a sha256 per tensor.
- **9B Q4_K_M: 427/427 = pack manifest. PASS.**
- **27B: 851/851 = pack manifest. PASS.**
- Every tensor in the manifest was read by the engine.

### G2: streams, tokens and logits lines

With the flag on:

| check | logits lines | result |
|---|---|---|
| 9B Q8_0 s8 vs the base receipt | 12 | IDENTICAL |
| 9B Q8_0 s64 vs the base receipt | 68 | IDENTICAL |
| 9B Q8_0 stream sha256 | – | `5f07a4ec32f1`, as wanted |
| 9B Q4_K_M p1 | 12 / 68 | IDENTICAL |
| 9B Q4_K_M p2 | 83 | IDENTICAL |
| 9B Q4_K_M long | 523 | IDENTICAL |
| 27B mode 9 p1 x 64 vs `receipts/q27c/final-ab/1-fz` | 68 | IDENTICAL |

The three 9B Q4_K_M prompts ran in one load.

With the flag off (pack path, same `build/g2` binary):
- 9B Q4_K_M s64, 9B Q8_0 s64 and 27B mode 9 are all IDENTICAL to the base receipts.
- The Q8 stream sha is `5f07a4ec32f1`.

**PASS.**

### G4: Kev-4B

The engine's `mf.validate` refuses Kev's shapes (it accepts only the 9B and the 27B), so G4 runs at the loader level. `build/g2/unit-dev` reads every region from `Kev-4B-Q8_0.gguf`, puts it on the device as `io_put` does (`cuMemcpyHtoD` into managed memory), poisons the host copy, and hashes the `cuMemcpyDtoH` readback. **428/428 match `~/sys1b-tmp/pack-kev4b-q8`'s manifest (read only). PASS.** It is not an engine run.

### G3: load time vs the prediction

"Load" is the engine's `resident head ... + layers in N ms`. "Pack (receipt)" is INTEG1 s2 or the plan's receipt. "Pack (fresh)" was measured here.

| | predicted GGUF | **measured GGUF** | pack (receipt) | pack (fresh) |
|---|---|---|---|---|
| 9B Q4_K_M cold | 4-6 s | **5.1 s** | 248 s | not run (> 4 min of lock time) |
| 9B Q4_K_M warm | 2.5-3.5 s | **1.2 s** (GGUF 100% resident) | 2.9 s | 20.5 s, 39.6 s (head never cached, see below) |
| 9B Q8_0 cold | 7-10 s | **8.5 s** | – | **5.7 s: the pack is faster cold** |
| 9B Q8_0 warm | 4-6 s | **2.5 s** (100% resident) | 5.8 s (INTEG1 q8, cache state unknown) | 4.5 s |
| 27B cold | 13-20 s | **15.4 s** | 317 s | not run (> 4 min) |
| 27B warm | 5-9 s | **not reachable**; second run 18.7 s, 0.35 of 19 GB resident | 4.4-8.5 s | 77.8 s with the HDD head pre-read |

Notes on the table:
- **Q8_0 cold:** the GGUF copy is on NVMe2 (ntfs-3g FUSE), and `pack-q8` is on `/home` (btrfs, NVMe0). These are different disks, so this is not a like-for-like loader comparison. Both numbers are reported as measured.
- **9B Q4_K_M warm, pack path:** both packs keep `pack.bin` on the HDD, which is also ntfs-3g (FUSE). Pre-reading it inside the lock left 0 B resident at load time, so the HDD head cannot be warmed on this box. The two fresh pack numbers are HDD-bound. The 2.9 s receipt is the fair warm reference.
- **Warm in general:** the page cache did not survive a queue wait. Under 16 GB of swap use and other lanes, a pre-read done before the wait was gone by load time (the first try is kept as `try1-g3-q4-gguf-warm`: 7.6 s). So `run.sh` now runs the evict/warm step inside the lock right before the engine and records `fincore` residency in `OUT.pre`.
- **Two extra GGUF findings:**
  - `g3-q8-gguf-cold` had a wall time of 29.3 s for a load of 8.5 s plus one step. The extra ~20 s after the load is not explained. Same-model pack runs took 5.4-5.7 s wall. This was not investigated (unknown below).
  - The G1/G2 runs carry the verify readback, so their load times (19.6 s for the 9B, 67.9 s for the 27B) are not load numbers.

Memory:

| | engine process VmHWM | GPU per-process | card used |
|---|---|---|---|
| GGUF path | 0.5-1.3 GB | 292-508 MiB (managed memory is not counted per process) | 4.7-19.0 GB |
| pack path | 0.9-1.2 GB | same range | equal within ~0.1 GB |

- **The loader's host RAM:** a transient malloc of one region (at most 1.04 GB, the 27B `output.weight`) plus 8 x 16 MiB of scratch. That stays under the 1.5 GB rule but **misses the plan's < 200 MB prediction**: the bytes go malloc → `io_put`, not straight into the Corpus block (risk 1 in the plan was taken).
- The unit memory peaks (1.7-10.1 GB) include page cache.

## S5: handoff

- `patches-gguf2/gguf2.patch` = `git diff 2933eab7..9d5b3119`, 9574 lines including receipts.
- `git apply --check` results:

| applied on | result |
|---|---|
| `2933eab7` (gguf2's parent) | CLEAN |
| `630d79b2` alone | does NOT apply: `tools/gguf2pack.c` comes from gguf1 |
| `630d79b2` + `patches-gguf1/gguf1.patch` | CLEAN, and the resulting tree equals the gguf2 tree (`94d9ff7b`) |

- One whitespace warning: a receipt line `pre ` (an empty PRE).
- The README has a "load the GGUF directly" paragraph, and `docs/GGUF2.md` covers what ships, what doesn't, and the next three steps.
- **The Mac / Metal follow-up is NOT done.**

## Deviations from the plan, named

- The G1 hook is `LLB_GGUF_VERIFY=<manifest>` (it checks and exits 1 on a mismatch), not `LLB_GGUF_G1=1` + `tools/gguf2/g1.sh`. `s4cmp.py` counts its `g1` lines.
- `host/llb_gguf.c` is 564 lines, against ~350 predicted (sha256, threads, partial Q8 rows, verify).
- The engine edit is +17/-8 (25 lines), against ~20 predicted. The ~150-line stop rule was not hit.
- Loader RAM: see G3.
- The README's `tools/run27b.sh` + `LLB_GGUF` command line was not run literally. The gates ran `build/g2/llb-engine` with the same env (`LLB_Q4_DP4A=9`), and `run27b.sh` only exports env and execs the engine.

## OOM / lock / queue

- **OOM:** none. Every unit exited 0 with swap 0 B.
- **Queue:** the lock was contended all session by other lanes' bench and pf-q5 / pf-q8ab runs (about 6 waiters at times). Of my runs:
  - none got exit 3;
  - each waited in `flock -w 1800`;
  - nothing of mine was killed;
  - I touched no other lane's process.
- **The poller:** `run.sh`'s `.mem` poller matches any `build/g2/llb-engine` process. For one stretch, two of my runs were queued at once (an orphaned q8-cold run whose parent script I had stopped, plus the new batch), so `g3-q4-gguf-warm.mem` holds `g3-q8-gguf-cold`'s peaks (8737 MiB card / 1.29 GB). **That `.mem` is wrong.** Its load time is right. All later runs were serial.
- **The Q8_0 GGUF copy:** copied to NVMe2. The sha256 matches the original (`80962657…`), the original is untouched, and NVMe2 has 25 GB free.

## Unknowns

- The extra ~20 s of wall time after the load in `g3-q8-gguf-cold` (first Q8 GGUF load after an evict). It did not recur in `g2-q8` (3.4 s wall).
- Warm 27B GGUF load time: not measurable here.
- A same-disk Q8_0 cold comparison (GGUF and pack on one filesystem): not run.
- Why ntfs-3g keeps the NVMe2 GGUF cached but not the HDD `pack.bin`: not investigated.
