# PACK35 close-out — 2026-10-06

VERIFIED: `tools/gguf2pack.c` now packs the registered qwen35moe 35B-A3B GGUF
using C only. G1, G2, G3, G4 and G5 PASS. Both full builds are byte-identical
to the read-only moe2b reference across all 44 files. No pack is shipped.
Task 5 is PARTIAL: local build documentation and a proposal are complete, but
the exact current Mac release README/llb wording was unavailable here.

Labels throughout: VERIFIED = read/run in this session; BELIEVED = another
source's unrerun claim; PREDICTION = committed before the run; UNKNOWN and
NOT RUN explicitly bound the evidence. No GPU commands or downloads occurred.

## Tasks and commits

| Task | Status and verification | Commit |
|---|---|---|
| 1 Read + predict | DONE; complete MoE builder and dense C reader, registry/read/row-split helpers, table diff and reference inspected; plan and G1–G5 committed before compilation | 7c9a260a |
| 2 Implement C branch | DONE, TESTED; +240/-6 C lines including comments; separate dispatch, existing row-split helper reused, dense source flow retained | d6af190c |
| 3 G1/G2 | DONE, TESTED; both builds match every reference size/hash/byte, and each other | a4cdea8e |
| 4 G3 | DONE, TESTED; 27B and 9B Q4_K_M stored sidecars match with no normalization; five refusal controls PASS | 950c4acb |
| 5 Docs + Mac proposal | PARTIAL; docs/PACK35.md and dist-patch proposal written; exact current Mac source unavailable, application NOT RUN | 1ac1a3b7 |
| 6 Close-out | DONE; this report, receipt hashes, scratch size/deletion and terminal-job checks committed | close-out commit containing this file |

Implementation stays within the predicted 230–300 added C lines. `pack.bin`
contains only heads; layer data uses virtual manifest offsets, dense shared
aliases, table words 21/22 for router/shared gate, and experts.idx references
absolute GGUF offsets and per-expert sizes. MoE JSON preserves the oracle's
field/tensor order, omission of fmt/source.types, numeric spelling and newline.
No Python file was added or changed. The existing dense behavior is preserved.

## Inputs and measured builds

- Source: `/data/models/Qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`,
  VERIFIED 22,360,456,160 bytes. Its complete-file SHA-256 was NOT RUN; the
  converter reads headers and stored tensor ranges, not the routed-expert pool.
- Ground truth: `/data/moe2/pack-35b`, read-only throughout. The pre-build
  reference hash list equals the freshly read reference list after G2.
- Python oracle: `~/Documents/Project/llama-bend-moe2b/tools/moe2/pack_moe.py`.
  It was read, not executed. Source hashes: `pack35/oracle-sources.sha256`;
  immutable source recheck: `pack35/G5-oracles-unchanged.txt`.
- Base C tool: `~/Documents/Project/llama-bend-pkg3/tools/gguf2pack.c`,
  read-only; its initial hash equals the initial worktree tool hash.
- Compiler: GCC 15.2.1, Linux x86_64 (`pack35/environment.txt`).

| Run | Wall time including strace | /usr/bin/time max RSS (kB) | Output file bytes | Exit |
|---|---:|---:|---:|---:|
| build1 | 28.80 s | 35,188 | 2,678,327,538 | 0 |
| build2 | 25.51 s | 34,748 | 2,678,327,538 | 0 |
| 27B dense sidecar | 0.67 s | see time receipt | 162,778 | 0 |
| 9B dense sidecar | 0.68 s | see time receipt | 81,902 | 0 |

These are two observed build times, not a benchmark distribution or Mac
prediction. Hashing/cmp time is excluded. Per-file sizes are in
`pack35/reference-sizes.tsv`: heads 1,080,696,832; 30 layers of 41,771,776;
10 layers of 34,433,024; table 5,120; index 2,644; manifest 139,422 bytes.

## Gates and receipts

G1 PASS: `pack35/build1.result.txt`, `build1.candidate.sha256`,
`build1.candidate.sizes.tsv`, `build1.time.txt`, stdout/stderr/trace/exit/command
receipts. All 44 files matched reference SHA-256 and `cmp`; no differing byte.

G2 PASS: corresponding `pack35/build2.*` receipts plus
`pack35/determinism.result.txt` and its checksum/size lists. All 44 matched the
reference again, and build1. Both builds ran in `env -i PATH=/usr/bin:/bin`.

The full 44-file SHA-256 list is `pack35/reference.sha256`. Its SHA-256 is
`bcdd13bd791f72a6d0641fe449eb7da5880ba1d2555d86e013899f9e60dae2a4`.
The build1, build2 and determinism candidate checksum-list files have exactly
that same hash. Every individual layer hash is recorded in those lists.

| Output | SHA-256 (both builds and reference) |
|---|---|
| pack.bin | c87aa42f825c6e514bc42fdaefbf01ab052bd8f4f7635a0240dd244474b567d1 |
| manifest.json | 6269e0b5533d2304440c94da8e34fe2388cd23dee1b7602ee69a9257065f1ab3 |
| experts.idx | 0aa81d4bbae25dded635eb9a127f33e9371b9972f5aafa9b6165b89e66c93341 |
| table.bin | 26f482e468076d8b8ff17a82a092f078ddde07d943531f49f0157baf4cb5e769 |

G3 PASS: exact stored references were available, so no fallback or normalization
was needed. Commands use the exact source.file paths in those references:

| Model | Reference directory | GGUF path |
|---|---|---|
| 27B | /data/gguf2/sidecar/q27 | /data/models/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf |
| 9B | /data/gguf2/sidecar/q4km | /data/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf |

| Reference and candidate file | SHA-256 |
|---|---|
| q27/manifest.json | 65939b6952c6ac2d002d5f9e849264df4a68e970eea1d4d8149949c27c760b26 |
| q27/table.bin | 9284ee92a1970bb4ef4213d20bf7c4f503c0c63273f6b249fcfdd393b888be31 |
| q4km/manifest.json | 9ccda27a1d8a1761976ad83e46adb98dcddd44b553f7f9b2b786a4d2fdc660f1 |
| q4km/table.bin | a108eabbd8c43570b6e52381053ab543e6a6e8b51dbba302eb38f8daddfc3be7 |

Receipts: `pack35/G3-q27.*` and `pack35/G3-q4km.*`. SHA-256 of each candidate
checksum-list file: q27
`381f4dad49c8e445388a699d662410e7e08a457a72bc45d7313ae291d42aaa2d`;
q4km `f55495f0d25aaee6031213286378a36d61713735499266075fb0d3a59046b4aa`.

G4 PASS: tested release-candidate binary size 61,120 bytes; SHA-256
`752a82a01c86463c1eb190422556725336d8cf048ed58026c91ab7e1e9dc0c25`.
`strings` plus case-insensitive python/pypy grep found no match;
`G4-imports.txt` has no process-launch imports. `G4-execs.txt` records only env
and the converter; no child interpreter. Both real 35B builds passed with one
environment variable, PATH. No release archive or published binary was changed.
The candidate was deleted with scratch, as requested; source and binary hashes
are retained (`binary.sha256`, `implementation-sources.sha256`).

G5 PASS for this lane's commands and traced compiler/converter operations.
All write operations in `G5-write-operations.txt` are inside scratch; the
outside-match file is empty. Other explicit writes were the source edits,
documentation, scripts and receipts in this worktree, plus the allowed shared
git metadata `~/Documents/Project/llama-bend/.git`. portpanel2 was not
used. No other worktree, GGUF, reference pack, Mac file, dist/, CHAIN.md or
~/.codex was written. This is not a machine-wide audit of unrelated processes.

Additional controls: overwrite refusal and each unsupported MoE mode
(head-only, sidecar-only, q8 and verify-against) returned 1 with no writable
file opens. `pack35/refusals.result.txt` and per-case traces preserve evidence.

## Findings and limits

- Initial strict `-Werror` compile FAIL: GCC reports existing format-truncation
  warnings in dense check_shapes' W macro. Unmodified pkg3 reproduces them.
  Saved logs: `compile-werror-failed.log`, `base-werror.log`. Final build keeps
  -Werror but disables that one existing warning. Dense code was not changed
  to silence it. No G1–G5 gate was loosened, and none failed.
- Read-time validation correction before the first pack run: shared gate is
  a rank-one [2048] F32 vector; router is [256,2048]. This did not cause a failed
  runtime gate. No byte-mismatch finding occurred in either build.
- Mac docs finding: current release README/llb source was not present in the
  inspected local paths. The owner's exact quoted prebuilt sentence is BELIEVED;
  the exact current llb models sentence is UNKNOWN. See
  `dist-patch/pack35-mac-proposal.md` and `pack35/mac-source-discovery.txt`.
  The proposal patch is based on the owner's quote and is NOT an applied patch.
- BELIEVED, UNVERIFIED-by-me: Mac run.sh's overlay makes experts.idx name the
  user's GGUF. Mac-native compilation, rebuild, overlay and llb flow are NOT RUN.
- Engine-level load is NOT RUN here: the unified Linux build has no MoE carry
  yet, so equality to the reference pack transfers the moe2b gates' evidence.
  The source report states 6.800 ms/step versus 12.782 ms; those historical
  timings are BELIEVED here, not remeasured performance or a Linux release claim.
- Other MoE GGUFs, path relocation and dense full-pack conversion were NOT RUN.
  Dense G3 covers the requested stored sidecars; the shared row-split loop is
  exercised by both complete MoE equality runs.

## Exact commands and reproduction

Run from `~/Documents/Project/llama-bend-pack35`:

```sh
mkdir -p ~/pack35-work
TMPDIR=~/pack35-work cc -std=c11 -O2 -Wall -Wextra -Werror -Wno-format-truncation tools/gguf2pack.c -o ~/pack35-work/gguf2pack
tools/pack35/run-moe-gate.sh build1
tools/pack35/run-moe-gate.sh build2
tools/pack35/run-dense-gate.sh
tools/pack35/check-refusals.sh
```

The committed harnesses contain the exact `/usr/bin/time -v`, `strace -f -qq
-e trace=%file`, `env -i PATH=/usr/bin:/bin`, converter, SHA-256 and cmp commands.
The build command was itself traced with TMPDIR set to scratch; the trace is
`pack35/compile.trace`. Core conversion command, used identically for both
builds with the output directory changed:

```sh
env -i PATH=/usr/bin:/bin ~/pack35-work/gguf2pack /data/models/Qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ~/pack35-work/build1
```

Exact G4 scans:

```sh
strings ~/pack35-work/gguf2pack > ~/pack35-work/binary.strings
rg -n -i 'python|pypy' ~/pack35-work/binary.strings
nm -u ~/pack35-work/gguf2pack
rg 'execve\(' reports/pack35/build1.trace
```

The string search returned 1 (no matches), as required. G1/G2 check commands
and generated hashes, including first-difference reporting on any future FAIL,
are implemented in `tools/pack35/check-pack.sh`.

## Cleanup and next steps

VERIFIED scratch before deletion: 5,357,329,625 apparent bytes, 5,357,604,864
allocated bytes, 99 regular files. At most two full 35B builds existed at once.
The initial forced deletion form was rejected by automatic command review
before execution; after inventory inspection, ordinary `rm -r --
~/pack35-work` succeeded. The directory is absent. Receipts:
`pack35/cleanup.txt`, `pack35/jobs-final.txt`. Every foreground session returned;
no systemd unit, queued job or background job was created or left running.

1. Orchestrator: carry the C change and rebuild on the Mac; repeat the 44-file
   equality gate against its reference using the same Mac GGUF path.
2. Inspect the current Mac llb/README, finish the proposed wording/prep/disk
   changes, and validate the GGUF-path overlay plus end-to-end 35B launch.
3. Integrate MoE into unified Linux separately with its own engine gates before
   making any unified-Linux inference or performance claim.

All committed gate receipt files have hashes in `pack35/receipts.sha256`.

SHA-256 of that receipt index: `ddf617246cb6fc6107376728f77b42a25bed3caa4929af89c8a12a5f95a1193f`.
