# E5-08B — the 0.8B (d 1024) trunk on canonical (PC / RTX 3090) — REPORT (2026-10-18)

## Headline
- **Qwen3.5-0.8B Q4_K_M now loads and runs on canonical's CUDA engine** (branch `e5-08b-canon`, off `768bff35`; not pushed).
- **Every gate passes on its frozen bar**, including the G6 stretch: G0, G1a/b/c, G3, G4, G5 and G6.
- It runs text, decide and image→text on the exact tier at ~306 tok/s decode.
- The 9B golden is 64/64 + 68/68 on both binaries. 9B Q8_0 and Q4_K_M logits are byte-identical old vs new at every
  position.

**What is blocked or limited:**
- The 0.8B fast tier, LLB_SPEC/MTP and LLB_BATCH are refused at d < 2048 (by design).
- The 4B byte gate is not measurable on canonical (no 4B CUDA shape).
- **Metal is DEFERRED:disk-floor.** The Mac is at 3.9 GiB free (100 %), so no Metal build or gate was attempted.

Prereg: `PREREG.md`, frozen at commit `5de86bd1` before the first GPU run. Amendments 1–4 are procedural or method-only,
each committed before the run it governs. No bar was retuned.

## G0 — build (PASS)
- **Command:** `build.sh` with:
  - `CC=clang` (21.1.7, `~/opt/clang21`);
  - `LLB_CC_EXTRA` = the sys1dual request objects + `-lopenblas`;
  - `LLB_BEND=~/restored-repos/llama-bend` (the `pins/bend-240ecef4` tree, `aada4cbf`);
  - `LLB_OUT=~/scratch/e5-08b/new`.
- **Result:** rc 0. The log has 0 warnings and 0 errors (`receipts/build/new-build.log`).
- **Patch stack:** `feb1836d1f425276e462cdf112041a5cd63245bba6055c868c4eea193b41b20c` (62 patches; canonical's 61 +
  `payload-w1024.patch` `1a358d53…`).
- **Compiler:** `2b40faf7…`.

| binary | `llb-engine` | `.c` |
|---|---|---|
| new | `b97ce1ba…` | `89c84baa…` |
| old (canonical 768bff35) | `a6005eb6…` | `fdd12e35…` |

Full sums are in `receipts/build/SHA256SUMS`. It was green on the first build→gate iteration.

**Pre-existing canonical defect (not from this port):**
- A plain `bash build.sh` does not link on canonical. The emitted C references `dual_next` / `dual_done` (sys1dual)
  unconditionally, so the link fails with gcc 15 and clang 21 alike (`receipts/build/base-build-gcc-linkfail.log`).
- Only `tools/sys1dual/build.sh`'s `LLB_CC_EXTRA` recipe produces an engine. Both binaries here use it.
- Two smaller environment drifts:
  - `bun` is not on `PATH` (it is at `~/.bun/bin`);
  - `/usr/bin` has no plain `clang`, only clang-18, which lacks `#embed`.

## The port: G33's blocks mapped onto canonical
G33's tree was diffed against canonical's patched compiler: 360 differing lines. Of those, only the 0.8B's needs were
carried, by intent. Two dependencies G33 inherited from the 4B lineage are missing on canonical, so they were carried
too: the rms 8192-tree generalization and the Q6_K embedding.

| block (G33) | canonical site | change | guard (old path unchanged) |
|---|---|---|---|
| validate 24/8/1024/3584, kv 4 or (2 & d 1024) | `engine_qwen35_gpu.bend` `mf.validate` | 3rd tuple; `n_kv_heads` check | exact tuple |
| Q6_K tied embedding (from G33's q6k2560) | `.bend` head/embedding check; `pass_mail` `fme`; host fold embed; new `llb_embed6` | Q6_K admitted | `d_model 1024` / `env08` only (G33 admitted it for every shape) |
| envelope `env08`, `env \|\| env08` | `pass_mail` | G33's clause verbatim | exact 0.8B record |
| down K-quant admit | `pass_mail` | `\|\| (env08 && df % 256 == 0)` | env08 |
| K-quant GEMV d 8 | `kp_warp`, `llb_warp_q4k/q6k` (q5k is pair-only on canonical; it gets the d 8 arm via `kp_warp`) | `d == 8` → `kp_pair<FMT,1>`; floor 9 → 8 | d 8 was a broken path before (ch underflow / NP 32), so no working model used it |
| attn4 kv stride | `llb_attn4` (exact path) | `ks = (gridDim.x/grp)·256` | = 1024 at every 4-kv-head launch (16/4, 24/6, legacy 16) |
| rms tree at n 1024 (from G33's w2560) | `llb_rmsnorm_add`, `llb_rmsnorm` | n 1024 → the 8192 tree | every n ≠ 1024 keeps its branch (narrower than w2560's `n == 4096`) |
| `llb_slot` width | `llb_fulb_run` | `kv` | 1024 for every existing model |
| Q8_0 cats depth ≥ 10 | `llb_fbn_run`, `llb_linb_run` (2 sites), `llb_fulb_run` | `&& d >= 10` | existing models are d 10/11 |
| fast-prefill refusal | `pf_fast` (`llb_pf.h`) | `err_fail` at d < 2048 with `LLB_PREFILL=fast` | env opt-in only |
| **new:** multi-row tiers | `pass_run` (host) | `LLB_SPEC` / `LLB_MTP` / `LLB_BATCH` refused at d < 2048 | canonical-only paths G33 lacked (`llb_t_attn4` keeps the 1024 stride; spec embeds Q4_K/Q8_0 only; `bx_*` bodies d ≥ 9) |
| models.conf | `tools/llb/models.conf` | `08b` row, fast 0 | new row |

Not needed:
- The image path. Canonical already has `payload-imrope` and the `llb_img_*` host block (FUS-V46 / G4 F1).
- The dp4a and q4glue paths. `llb_q4dp_ok` and `llb_kgemv_res` already require d ≥ 9.
- The rms `_q8x2` variants, which are reachable only via dp4a.

## G1 — text vs llama.cpp (PASS; receipts `receipts/ANALYSIS.txt`, `receipts/gpu/08b-*`)
- **Reference:** llama.cpp capfix `2ca15f54`, CPU (`CUDA_VISIBLE_DEVICES=`, `-ngl 0`), greedy, same GGUF, its
  special-aware ids fed raw. The fresh refs equal G33's banked refs exactly.
- **G1a:**
  - labels match 2/2 on all three: decide `billing`, decide2 `shipping`, decide3 `shipping`;
  - decide3's gold is `returns`; both engines say `shipping`, a model-tier limit.
- **G1b:** ledger **34/34** token-exact.
- **G1c (teacher-forced, top-1 agreement):** every flip is a near-tie.

| prompt | engine↔llama.cpp CUDA | llama.cpp CPU↔CUDA (bar) | engine↔CPU | flips (engine margin) |
|---|---|---|---|---|
| sky (64) | **63/64**, median \|Δ\| 0.017 | 62/64 | 63/64 | 1, 0.051 |
| bridges (128) | **127/128**, median \|Δ\| 0.018 | 122/128 | 122/128 | 1, 0.022 |

- **Free-running (informational, same as G33):** sky splits at gen 4 (4/64) and bridges at gen 16. These are the
  documented near-ties where llama.cpp disagrees with itself.

## G3 — EOG stop (PASS; `receipts/gpu/g3-*`)
- **Surface:** `llb chat 08b -n 512`. That is `llb-tok run`, with the chat template and the EOS / `<|endoftext|>` cut.
  It is canonical's completion surface.

| prompt | generated | stop | stop id |
|---|---|---|---|
| ledger | 33 | end of text | **248046** `<\|im_end\|>` |
| capital of France | 1 ("Paris") | end of text | **248046** `<\|im_end\|>` |
| primary colors | 79 | end of text | **248046** `<\|im_end\|>` |

- The stop set is {248046 `<|im_end|>` = GGUF eos, 248044 `<|endoftext|>`}. In each case the token count equals the
  EOG's index, and no engine id follows the EOG in the stream log.
- **Correction to the brief's wording:** the chat model stops at `<|im_end|>` 248046, not `<|endoftext|>`.
  248044 is in the stop set but was not hit.
- **Attempt 1 failed before reaching the engine** (`receipts/gpu/g3-attempt1/`):
  - Canonical's `capacity.tsv` has no measured row for **any** model, so `llb run` / `llb chat` refuse every model on
    canonical today.
  - AMEND-3 ran G3 with `LLB_CONF` set to a scratch copy without that table.
  - **Owner item:** canonical's `llb` refuses all models until `capacity.tsv` gets measured rows.
- **Not changed:** the raw engine (`llb-engine … --steps N`) still runs to N past EOG, as OMNI-TRY saw.
  - Changing that would alter the flag-off decode loop for every model.
  - Callers that cut at EOG: `llb-tok` (measured here) and, believed but not measured here, `llb-serve` / dual.c.

## G4 — do-no-harm (PASS; old `a6005eb6` vs new `b97ce1ba`, LLB_DUMP at every position)

| leg | result |
|---|---|
| 9B Q8_0 (GGUF sidecar), golden ids, 64 steps vs `receipts/integ-base188-x64` | old **64/64 + 68/68**; new **64/64 + 68/68** |
| 9B Q8_0 golden, old vs new | 68/68 positions byte-identical, tokens identical |
| 9B Q8_0 sky | 81/81 byte-identical |
| 9B Q4_K_M sky (K-quant coverage) | 81/81 byte-identical |
| 4B Q8_0, 4B Q4_K_M | **N/A on canonical.** Both binaries refuse with the same `unsupported/invalid qwen3.5 pack: n_layers/n_heads/d_model/d_ff` (rc 1). Canonical has no 4B CUDA shape; this port adds only the 0.8B tuple |

- Concatenated dump hashes old = new on every 9B leg (`receipts/dumps.sha256`).
- 27B: not run, no file here (as in G33). It is unaffected by construction:
  - its GEMVs are d ≥ 10;
  - the attn4 stride is 24/6·256 = 1024 and the slot width is 1024;
  - rms n 5120 takes its old branch;
  - the cat guards are d ≥ 10, and env08 is false.

## G5 — determinism (PASS)
- Sky, 64 steps. Compared against the main run at all 81 positions, byte-identical and with tokens identical:
  - a second and a third prompt in one process (`,` resets): 81/81 each;
  - a fresh process: 81/81.

## G6 — stretch: 0.8B vision on CUDA (PASS; `receipts/G6.txt`)
- `LLB_VIS_EMBD` rows at trunk width 1024 go through canonical's existing imrope / `llb_img_*` block. No rebuild was
  needed.
- Tower rows (pocket-omni) vs mtmd CPU `caption.tsv`:

| image | tower rows | note |
|---|---|---|
| invoice | **32/32** | |
| photo640 | **32/32** | |
| small277 | **31/32** | splits at step 14, the documented 0.116 near-tie ("around 1960") |

- The CPU-mtmd-row control gives the same counts.
- Rope maps match G33, e.g. {s0 4, n 63, nx 9, sh 54}.

## Informational
- **Speed:** bridges, 128 steps, warm: **309.8 / 305.3 tok/s** decode (ms-granular step lines), 0.89 s wall.
  `llb chat` reports 292–333 tok/s. G33's lineage engine did ~260.
- **Refusal probes on the 0.8B** (rc 1, one-line message each): `LLB_PREFILL=fast`, `LLB_SPEC=2`, `LLB_BATCH=4`.
- **The old binary on the 0.8B:** `unsupported/invalid qwen3.5 pack: n_layers/n_heads/d_model/d_ff`. This is the
  pre-port wall.

## Open / owner items
1. **Metal: DEFERRED:disk-floor (Mac 3.9 GiB free, 100 %).** No Metal build or gate was attempted. The Metal patches
   (`backends/metal/*`) still refuse d 1024.
2. **0.8B fast tier on canonical.** The lineage's PREFILL-MATRIX lift was not carried, so `08b` has fast 0. Enabling
   it would need the `llb_t_*` kernels audited at d 1024 (the `t_attn4` stride, the `t_rn*` trees) and the
   PREFILL-MATRIX B1 re-run here.
3. **Raw-engine EOG.** It stops only through callers. An engine-side opt-in stop would need `cfg` / `gen` plumbing.
4. **The canonical build-recipe defect** (G0 above) and the **empty `capacity.tsv`** (G3 above).
5. **The 4B on canonical CUDA** would need its own port (G33's w2560 / q6k2560).

## Artifacts
- **Branch `e5-08b-canon`:**
  - `5de86bd1`: port + prereg;
  - `aef06832`, `16cf3e77`, `1d28e55f`, `74e9db9e`: AMEND-1..4;
  - the close-out commit with this report.
- **Code:** `tools/payload-w1024.patch`, `host/llb_host.c`, `host/llb_pf.h`, `engine_qwen35_gpu.bend`,
  `tools/build_gpu.sh`, `tools/llb/models.conf`.
- **Lane dir `reports/e5-08b/`:** `PREREG.md`, `REPORT.md`, `scripts/`, and `receipts/`:
  - `lcpp/`: llama.cpp CPU refs plus the CUDA TF ref;
  - `gpu/`: every run's cmd / out / err / rc / time, plus `lock.txt`;
  - `build/`, `ANALYSIS.txt`, `analysis.json`, `G6.txt`, `g6.json`, `dumps.sha256`.
- **Scratch (not committed):** `~/scratch/e5-08b/` holds the binaries, sidecars, and the logit dumps
  (shas in `dumps.sha256`).
- **GPU:** every run ran under both house locks (`receipts/gpu/lock.txt`).

## SUMMARY
- **Runs:** the 0.8B pocket model runs on canonical CUDA. Text, decide and image→text pass on the exact tier at
  ~306 tok/s.
  - G1: decide 3/3, ledger 34/34; teacher-forced 63/64 and 127/128 vs llama.cpp CUDA (bars 62/64, 122/128).
  - G3: EOG stop 3/3 at 248046 `<|im_end|>`.
  - G5: determinism 81/81 ×4.
  - G6: vision 32/32, 32/32, 31/32.
- **Do-no-harm:** the 9B golden is 64/64 + 68/68 on both binaries. 9B Q8_0 and Q4_K_M are byte-identical old vs new
  at every position. The 4B is N/A: canonical refuses its shape before and after.
- **Blocked or limited:**
  - fast / spec / batch are refused at d < 2048;
  - the raw engine still runs past EOG (callers cut);
  - canonical's plain `build.sh` doesn't link without the sys1dual objects;
  - `llb` refuses all models on canonical's empty `capacity.tsv`;
  - **Metal is deferred at the Mac disk floor (3.9 GiB free).**
- **Branch tip:** `e5-08b-canon`, the close-out commit (see `git log`); not pushed.

## STATUS
DONE
