# CLI1 final: `llb`, one front end for the llama-bend engine

Branch `cli1`, fork `95eaa146` (unify1). Gates were registered in `docs/cli1-registration.md` (commit `1160b075`) before
the runs, and none were loosened. Engine under test: a copy of unify1's `build/s3/llb-engine` (not rebuilt). GPU runs
went through `tools/llb/test/gpu.sh`: one `flock -w 1800 /tmp/llb-gpu-bench.lock` hold per run, an empty card
(waited behind UNIFY1), and their own `systemd-run --user` unit with MemoryMax=12G, nice 10 and CUDA_VISIBLE_DEVICES=0.
GPU receipts were copied from the work dir to `receipts/cli1/gpu/` (`NAME.{state,out,err,unit}`; `.state` ends with
the wall time, the orphan count `pgrep -xc llb-engine`, and the exit code).

## What shipped (all new files; nothing else in the tree is edited)

| Item | File | Status |
|---|---|---|
| Front end (POSIX sh) | `tools/llb/llb` | done, tested |
| C helper: GGUF info, vocab extract, enc/dec, ChatML, run wrapper (stop at EOS, kill engine, summary) | `tools/llb/llb-tok.c` | done, tested |
| Config | `tools/llb/defaults.conf`, `tools/llb/models.conf` | done |
| Installer | `tools/llb/install.sh DEST` | done, tested (`inst3` run) |
| Tests (bash; jq/curl/expect in tests only) | `tools/llb/test/` | done |
| Goldens (R6) | `tools/llb/golden/` | done |
| Mac adapter | `mac_cmd` in `tools/llb/llb`, `docs/CLI-mac-adapter.md` | design + code, NOT RUN on a Mac (Linux dry-run with `LLB_OS=Darwin` + stub run.sh only) |
| User README | `release/README-llb.md` | done |
| Windows static C `llb` | priced below | NOT BUILT (as asked) |
| Persistent chat mode | priced below | NOT BUILT |

No Python ships. `qwen35_toktest.c`, `tools/unify/bundle.sh`, `README.md` and `run27b.sh` are untouched.
`gguf2pack` is built by `install.sh` from the existing `tools/gguf2pack.c`; that file is not modified.

Commands: `run`, `chat`, `models`, `prep`, `doctor` (exit = number of FAILs; checks GPU, compute capability, VRAM, idle card, engine + libs, kernel cache, llb-tok, stdbuf, cache disk, RAM, a 1 MB `/tmp` write for quota, `/dev/shm`, models; minimum driver UNKNOWN, tested on 580.119.02), `bench`, `tok enc|dec`, `pull` (size
first, resumable, asks above 5 GB, refuses without a tty unless `-y`), `version`, `help [CMD]`. Flags: `-n` (256),
`-f`, `--fast|--exact`, `--spec` (opt-in), `--ids`, `--dry-run`, `-v`, `--`. Greedy only; help says so; no `--temp`
(rejected with one line, R4 `unknown-option`).

## Gates

| Gate | Result | Receipt |
|---|---|---|
| T1 tokenizer.bin byte-identical (27B, 9B Q8_0, 9B Q4_K_M) | PASS 3/3 (`cmp`) | `receipts/cli1/T1.txt` |
| T2 ids == llama.cpp `/tokenize` | PASS 437/437 (37 corpus + 400 fuzz) | `receipts/cli1/T2.txt`, `T2-fuzz.txt`, `T2.diff` (empty) |
| T3 template == llama.cpp `/apply-template` | PASS 27B 12/12; 9B 12/12 (think on/off, both 9B forms) | `receipts/cli1/T3-*.txt`, `.diff` |
| R1 `llb run 27b "The capital of France is" -n 16`, empty LLB_HOME == hand pipeline, same flags | PASS, exact (`r1`) and again after the fast flip (`r1f`), cold and warm. Ids `11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369` | `receipts/cli1/r1/`, `gpu/r1f-llb.last.log` vs `gpu/r1f-hand.out` |
| R2 chat stops at `<\|im_end\|>`, no orphan, summary | PASS ("4", `stop: end of text`, orphans 0) | `gpu/r2-chat.*`, `gpu/r2f-chat.*` |
| R3 input modes | PASS: argument (R1), stdin (EOS), `-f` (9 tok), unicode (15 tok), 20 KB (3311 tok prompt, 1307 tok/s, EOS), empty (one-line error, exit 1) | `gpu/r3-*.*`, `receipts/cli1/R4.txt` (`empty-stdin`, `empty-file`) |
| R4 error paths: one line + nonzero exit | PASS 15/15 | `receipts/cli1/R4.txt` |
| R5 wrapper overhead < 150 ms warm | PASS: full run path (stub engine, real nvidia-smi, idle card, under the lock) 91-95 ms warm, 99 ms first; `--dry-run` median 70 ms | `receipts/cli1/R5.txt`, `gpu/r5-real.*` |
| R6 goldens (help x10, 2 dry-runs, doctor) | PASS 13/13 (`gpu/r6-verify2`; goldens rewritten after the last code change, the doctor checks for `/tmp`, `/dev/shm` and the driver note, then verified; doctor numbers and paths normalized; the dry-run goldens hold this machine's cache key, so they must be regenerated with `--update` on another machine) | `tools/llb/golden/`, `gpu/r6-verify.*` |
| R7 `--ids ... --exact` stdout == `run27b.sh` stdout | PASS, 33 lines identical (`wall_ms` excluded) | `gpu/r7-llb.out` vs `gpu/r7-hand.out` |

Extra checks (not gates): `llb pull` 7/7 (`receipts/cli1/pull.txt`: file:// fixture for resume and size, HF HEAD only, no
download over 100 MB; the >5 GB question is answered `n` on a pty through `expect`). The `--spec` summary attribution is
covered by `tools/llb/test/summary.sh` (stub engine). Every GPU receipt shows `orphans 0`.

## Before / after (27b, "The capital of France is", -n 16, fast prefill)

The hand way is 5 manual steps: find the GGUF, build the sidecar (`gguf2pack`), extract the tokenizer, encode the
prompt, run the engine with the right env (`LLB_GGUF`, `LLB_Q4_DP4A=9`, `LLB_PREFILL`, span, threads) and decode the ids
(`tools/llb/test/hand.sh`). The llb way is one command.

| | wall (cold) | first token (cold) | wall (warm) | first token (warm) |
|---|---|---|---|---|
| hand, 5 steps | 16.87 s | 15.87 s* | 20.71 s | 19.70 s* |
| `llb run` | 26.74 s (prep included) | 22.9 s | 17.44 s | 16.2 s |

\* The hand pipeline's engine stdout is block-buffered, so the first id reaches the pipe almost at the end; llb uses
`stdbuf -oL` so text streams as it is made. "Cold" means an empty `LLB_HOME` (or hand dir) with the NVRTC kernel cache
already built. The very first run on a new engine build adds the one-time NVRTC compile: 91.98 s total wall
(`gpu/r1-llb.*`; llb prints a one-time line with the cost). Load time is dominated by reading the 16 GB GGUF from
`/data` (ntfs-3g on an HDD) and swings 14-23 s between identical runs, so the cold/warm gaps above are mostly
disk noise, not llb.

## Numbers

| Run | Result | Receipt |
|---|---|---|
| 27b decode | 38.7-39.1 tok/s (25.6-25.8 ms/tok) | `gpu/spec-off`, `r1f-*` |
| 27b `--spec` (NextN, gated `blk64-wip.bin`) | 74.5 / 73.9 tok/s (13.4 / 13.5 ms/tok); ids == spec-off (64 tok), both runs; summary now counts the drafter setup as load (prompt 46.7 tok/s) | `gpu/spec-on`, `gpu/spec3` |
| 27b bench | prompt 45 tok at 369 tok/s; 128 tok at 37.8 tok/s | `gpu/bench-27b` |
| 27b 20 KB prompt | 3311 tok at 1307 tok/s (fast prefill), first token 21.5 s | `gpu/r3-20k` |
| 9b | load 4.5 s, 100 tok/s | `gpu/r9b` |
| chat 27b, 5 turns | 92.03 s wall; first token per turn 17.4 / 17.6 / 18.0 / 16.8 / 17.8 s | `gpu/chat5` |
| installed copy (`install.sh`) | runs 27b, stdout identical to the repo llb (`cmp`); reinstalled after the run-path changes (only `doctor` changed after that) | `gpu/inst`, `gpu/inst3` |

Chat reloads the model every turn. One turn costs about 17-18 s before the first token, which is under the ~30 s bar, so
chat ships as a normal command and not as "experimental". Help and the banner say a reload happens every turn.

## Decisions to know about

- **Default prefill = fast (27b).** UNIFY1's PF28 kill gate R2 passed (`reports/unify1-gates.md`, `e9e66ffe`), so
  `defaults.conf` has `LLB_DEFAULT_PREFILL=fast`. Flipping back is one line (`exact`). 9b and 9b-q8 are always exact
  (`fast` column = 0 in `models.conf`, and `--fast` on them is refused with one line).
- **27b gets `LLB_Q4_DP4A=9`; 9b does not** (`models.conf` env column).
- **`--spec` uses the gated drafter.** It sets `LLB_SPEC=2` plus `LLB_MTP=<gguf dir>/mtp/blk64-wip.bin` (`models.conf`
  spec column). Without `LLB_MTP` the engine would fall back to the ungated lookup drafter; an earlier llb draft did
  exactly that, and it was fixed before the spec runs. Under spec the engine's step 0 also builds the drafter (6-11 s),
  and `llb-tok` counts that as load, not prompt time (otherwise the summary showed 0.8 tok/s prompt).
- **stdbuf.** The engine's stdout is fully buffered when piped. llb runs it under `stdbuf -oL` (coreutils) when stdbuf
  is present (doctor checks for it), so ids stream line by line. Without stdbuf llb still works, but text arrives in
  blocks. Cost: none measured (decode is GPU-bound at ~26 ms/tok). On the Mac path no stdbuf is used (UNKNOWN 1 in the
  Mac doc).
- **Name clash.** UNIFY1's `tools/unify/bundle.sh` ships a different binary also named `llb-tok`. llb only ever runs the
  `llb-tok` next to itself (or `$LLB_OUT`). Whoever merges both should rename one; not done here because `bundle.sh` is
  off-limits.
- **R5 fixes.** The first measure was ~230 ms. Three causes were fixed: `q()` forked sed once per argument (31 forks;
  now pure sh), the reap poll slept 100 ms (now 10 ms, with a 5 s cap before SIGKILL), and the GGUF header was read again
  on every run (now cached in `where.ALIAS`, invalidated by size and mtime).
- **Alias cache.** `where.ALIAS` is trusted while the file has the same size and is not newer than the cache entry. A
  different file with the same size and an older mtime moved into the same path would not be noticed (marked in the
  code). `llb prep` after swapping files, or deleting `where.*`, fixes it.

## Priced, not built

**Persistent chat (keep the model loaded between turns).** The engine needs to read more ids from stdin after it
finishes a turn, keep its KV cache, and prefill only the new turn. llb then keeps one engine child per chat session.
Work: engine stdin turn loop + KV keep (~2 h), llb-tok run in multi-turn mode (~1 h), identity gates (multi-turn ids
== single-shot ids on the same history, orphan checks, ~1-2 h). **About 4-6 agent-hours.** Saving: the ~17 s reload per
turn plus re-prefilling the history (small with fast prefill: 124 tok at 925 tok/s = 0.13 s). This needs an engine
change, which was out of scope for CLI1 (new files only).

**Windows: a single static C `llb`.** It would port the sh front end to C (arg parsing, conf files, model search,
cache dirs, `CreateProcess` + pipes instead of fork/exec and `stdbuf`, `TerminateProcess` for stop, WinHTTP or
`curl.exe` for pull, `nvidia-smi.exe` for doctor), reusing `llb-tok.c` as a library. **About 6-10 agent-hours for the
front end.** NOT included: the engine itself on Windows (CUDA/NVRTC build with MSVC, its POSIX calls); that cost is
UNKNOWN until someone builds the engine there. The front end is useless without it.

## UNKNOWNs

1. Mac: all six items in `docs/CLI-mac-adapter.md` (run.sh stdout format, `--ids` file format, run.sh's default
   prefill, pack layout from Linux `gguf2pack`, Metal compile cost, no memory check). Nothing ran on a Mac.
2. Windows engine port cost (above).
3. Cold-load spread: 14-23 s comes from the HDD/ntfs-3g model disk; on an NVMe disk the load numbers would be
   different, not measured.

## Rule notes (disclosed)

- **Rule breach, once:** an early R4 "GPU busy" case started the real engine directly in my session (not under the lock,
  not in its own unit). It exited at once with "no GPU device" (the session has `CUDA_VISIBLE_DEVICES` empty), so no
  GPU work happened, but it broke the rule. After that, R4's GPU cases use a stub `nvidia-smi` and never start an engine.
- `/dev/shm` was at ~406 MB during the work, above the 300 MB line. The files are `u1000-Shm_*` from Sep 28/30, not
  mine. I did not touch them.
- llama-server (for T2/T3) ran in its own unit with MemoryMax=10G only while those tests ran, and it was stopped after.
- Work dir `~/cli1-work`: 91M at close-out (engine copy, caches, stubs, scratch), then deleted. Its GPU receipts were copied to `receipts/cli1/gpu/` first.

## Handoff

- `patches-cli1/cli1.patch` = `git diff 95eaa146..cli1 -- . ':!patches-cli1'` (everything except the patch file itself, which is committed after it). `git apply --check` against unify1's tip `5bdfd706` (it moved from a6126c97 during close-out; re-checked against the new tip, into a scratch index, no worktree touched): PASS. All paths are new files, so it cannot conflict with unify1 edits.
- `tools/llb/install.sh DEST`, then `DEST/llb doctor`.
