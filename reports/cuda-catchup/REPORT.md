# CUDA catch-up to llama.cpp

Branch `x/cuda-catchup`, based on `89dba9dde` (`feature/all-backends`, tree = published main `9f792f8c5`).
- Gates and method: [PREREG.md](PREREG.md), frozen before any measured candidate run, with two dated amendments
  (the prefill instrument), both written before the runs they govern.
- Receipts: `receipts/`; the scripts that made them: `scripts/`. Big files were in `$BIG` (a second disk) and staged
  in `/dev/shm` while their arms ran.
- RTX 3090, CUDA 13; llama.cpp build 9d57ce45. Two llama-servers of other lanes stayed resident on the GPU
  throughout (about 14 GB of VRAM, light intermittent load). Every MoE file is synthetic; no real MoE checkpoint and
  nothing over 10 GB was loaded; no 27B.

## What changed

### Exact tier (the default; bit-identical to the base)

1. **The decode run** (`engine_qwen35_gpu.bend`, `host/llb_host.c` `gen_run` / `pass_fold`,
   `tools/payload-catchup.patch`).
   - Before: every token went through the Bend program's IO loop: the host evaluated the continuation (print the
     id, `IO.now`, the logits and step lines) over unified memory, with about a dozen page faults; the device
     interpreter ran two rounds (`bend_dev`, 16 launches, about 330 us); then the pass. That was about 0.85 ms of
     every token on the 0.8B and on the MoE files alike.
   - Now gen's folded step asks the host for up to n greedy steps in one pass bang (word 47 bit 28 carries n). The
     host serves them back to back:
     - each step is the same folded-step launches, recorded as a CUDA graph;
     - the next step's embedding row is read by `llb_embt` from the device's own pick (`llb_amax`'s word 1), with
       `llb_embed*`'s element ops, so the host queues step s + 1 before it reads step s's ids;
     - it prints what gen would have printed for every step but the last (stdout ids; stderr logits and step lines,
       stdout flushed first as `io_errs` does), and returns the count in y's word 4. gen prints the last step's lines.
   - Off (one step a bang, as before) for the sampler, grammar, spec, MTP, batch, images, dumps and the prefix
     cache, and with `LLB_GEN=0`. A nonfinite step stops the run so gen's own check reports it.
   - The embedding table is prefetched to the device once (a table under 16 MiB is read in by a host memcpy, so
     each new token's row used to fault its page over: about 40 us a token on the MoE files).
2. **The partial-quad pair body** (`kp_pair`). Rows of whole superblocks that are not whole 1024-column quads (the
   0.8B's 3584-column down, the 4B's 2560-column gate / up / z) ran the tile body at 90–140 GB/s. The pair body now
   takes a partial last quad: lanes past the row take zero weights times the real x, as the tile body does, so the
   leaves, the quad tree and the carry are the tile body's. The fused residual store then takes those rows too
   (the same `__fadd_rn(x1, v)` as `llb_vadd`), except a Q6_K row under `LLB_Q4_DP4A` bit 32, which keeps its
   two-plane dp4a GEMV as before.
3. **`llb_gemv_anyk`.** `llb_gemv_any`'s K-quant rows (Q4_K, Q5_K, Q6_K) on `moe_dotn`: the same leaf and order
   (lane l sums groups l, l + 32, ... then the xor butterfly), two rows a warp sharing x, loads first.

### New tiers (opt-in, numerics-changing, gated)

- **`LLB_MOE_DP4A=1`** (decode): the routed experts' gate / up and down on int8 activations (q8_1 per 32 values,
  llama.cpp's mmvq method; `llb_mx_gu` / `llb_mx_dn` / `llb_mx_comb` in `host/llb_pf.cuh`). Q8_0, Q4_K, Q5_K and
  Q6_K experts. The trunk, router, top-k and shared expert stay exact.
- **`LLB_PREFILL=fast` on MoE files** (`host/llb_pf.h` `pf_moe`). Was refused for MoE (exact tier with a note).
  Now, per routed-expert layer of a chunk:
  - the router as a tiled F32 GEMM;
  - the top k per token (`llb_moe_topk`'s math);
  - the T k (token, rank) slots grouped by expert, in (token, rank) order;
  - each expert's gate / up / silu and down over its slots on the fast prefill's int8 tensor-core leaves (`pf_wide`
    / `pf_wide6`, one grid over every expert; Q4_K, Q5_K, Q6_K); Q8_0 experts on the int8 GEMV kernels;
  - the shared expert on the pfmm GEMMs;
  - then the rank-order weighted sum, the gated shared expert and the residual.

  The trunk is the dense fast tier's. Not with streamed experts (exact tier, with a note).

Development ledger (single dev runs, nsys or `stepus`):

| change | measured | kept? |
|---|---|---|
| decode run, plain launches | 0.8B 2680 us a token (base 3459) | replaced by graphs |
| decode run, graphs | 0.8B 2342 us | **kept** |
| run's four words moved off unified memory | `llb_embt` 41 us a step, unchanged | kept (harmless); not the cause |
| embedding table prefetched | `llb_embt` 41 → 2 us; S35 nsys period 667 → 632 us | **kept** |
| partial-quad pair body | 0.8B 2394 → 2068 us; 4B 20.7 → 7.6 ms | **kept** |
| `llb_gemv_anyk` | 4B 7.60 → 6.91 ms | **kept** |
| fast MoE prefill on the int8 GEMV kernels | S35 chunk 45.9 ms (gate/up 4.4, down 5.2 ms a layer) | replaced |
| grouped pfmm leaves, 64-token tiles, grid over T | chunk span 18.2 ms; router 630 us a layer | improved below |
| router as a tiled GEMM | 630 → 147 us a layer | **kept** |
| 16-token tiles | gate/up 925 → 864 us a layer | **kept** |
| grid x from the largest expert's slots (one read a layer) | gate/up 864 → 671 us a layer; span 13.9 ms | **kept** |
| dp4a down, a half warp a row (16 blocks) | 15 → 11 us a layer | **kept** |

## Profile, before and after (nsys, one steady decode step; `receipts/prof/`)

| per token | 0.8B base | 0.8B now | S35 Q4_K_M base | S35 Q4_K_M now |
|---|---|---|---|---|
| period (under nsys) | 3555 us | 2203 us | 1502 us | 631 us |
| outside the pass (Bend loop, page faults, host gaps) | 828 us | ~40 us | 868 us | ~16 us |
| `bend_dev` launches | 16 (325 us) | 0 | 16 (327 us) | 0 |
| launches | 438 | 371 | 113 | 70 |
| largest GEMV class | `llb_warp_q4k` 794 us x98 | 518 us x86 | `llb_warp_q4k` 201 us x21 | 211 us x21 |

- The 0.8B's 17 Q6_K launches (566 us, the 3584-column down on the tile body at about 24 us each) are now 5 plain
  (268 us, mostly the head) and 12 with the residual folded (164 us).
- What is left on the MoE step is the layers themselves: the trunk's exact GEMVs (about 52 us a layer), the routed
  experts (25 + 18 us), the top-k (9 us) and the F32 router (6 us).

## Correctness gates

| gate | result | receipt |
|---|---|---|
| **G-BITS:** candidate vs base `89dba9dde`, device, greedy 64 + TF, 3 prompts, 8 files | **48/48 byte-identical** (stdout ids + every `logits` line) | `receipts/BITS-MODE-ALL.txt`, `gate/*/BITS-MODE.txt` |
| **G-MODE:** host, stream, forced-small auto vs the candidate's device | **24/24** file x mode identical (6/6 each) | same |
| **G-REG:** dense 0.8B / 2B / 9B / clef-flash, exact + `LLB_Q4_DP4A=9` + `LLB_PREFILL=fast` | **28/28 byte-identical** | `receipts/GREG.txt` |
| **G-REG4:** 4B exact, 3 prompts | **3/3** | same |
| **G-MULTI:** `P1 , P2 , P3` (three gen runs) on the 0.8B and S35 Q4_K_M | **2/2** identical | same |
| **G-OFF:** `LLB_GEN=0` vs default (0.8B; S35 Q4_K_M greedy + TF) | **identical** (1/1, 6/6) | same, `BITS-MODE-ALL.txt` |
| **G-SIDE:** sidecars, 5 dense + 8 MoE, base vs candidate `gguf2pack` | **13/13 identical** (the same binary) | `receipts/GSIDE.txt` |

- No change under `backends/metal` or to `tools/gguf2pack.c`; the `--sidecar-only` and `--target metal` outputs are
  the base's.
- **Control:** the dp4a tier, a real numeric change, is 0/6 identical on every file: G-BITS sees it.

New tiers (G-TF: TF, 192 positions a file; G-PF: the prefix probe, each prefix prefilled as one prompt of a
multi-prompt run; paired against llama.cpp CUDA's TF on the same positions; reference X = llama.cpp CPU on the
F32-dequantized file, or for S35 Q4_K_M the exact tier's TF top-1, as the prereg declared):

| file | ref | llama.cpp CUDA (F) = ref | **dp4a** = ref | rms dp4a | **fast prefill** = ref | rms fast | verdicts |
|---|---|---|---|---|---|---|---|
| S35 Q4_K_M | exact tier | 114 | **167** | — | **134** | — | PASS / PASS |
| S35 Q8_0 | X | 131 | **181** | 0.121 | **142** | 0.307 | PASS / PASS |
| S60 Q8_0 | X | 158 | **177** | 0.102 | 157 (b 9, c 8, p 0.50) | 0.275 | PASS / PASS |
| S60 Q4_K_M | X | 114 | **167** | 0.155 | **116** | 0.517 | PASS / PASS |
| S128F Q8_0 | X | 154 | **181** | 0.096 | 154 | 0.273 | PASS / PASS |
| S128F Q4_K_M | X | 123 | **176** | 0.134 | **135** | 0.437 | PASS / PASS |
| S512 Q8_0 | X | 151 | **179** | 0.103 | **159** | 0.310 | PASS / PASS |
| S512 Q5_K_M | X | 149 | **176** | 0.130 | 146 (b 16, c 13, p 0.36) | 0.431 | PASS / PASS |

- Counts are of 192 positions. "rms" is the rms, over the 192 positions, of the candidate's logit for the
  reference's top-1 minus the reference's. llama.cpp's own CPU run (R) is 0.29–0.55 from X on these files
  (reports/cuda-moe).
- **The dp4a tier beats llama.cpp CUDA's own score on every file.** Its disagreements with the reference are 11–25
  a file, 3–10 of them near-ties (`gate/*/GTF-D.txt`).
- **The fast prefill** is at or above llama.cpp CUDA's score on 6 files. On S60 Q8_0 and S512 Q5_K_M it is 1 and 3
  positions below. The paired test does not support "worse" there (p 0.50 and 0.36), so both pass under the frozen
  rule. Every disagreement is listed in `gate/*/GPF.txt`.
- **Controls** (`gate/*/CTRL-*.txt`):
  - F scored as the candidate (floor-identical): PASS on 8/8.
  - The exact tier scored as the candidate (perfect): 192/192, PASS on 8/8.
  - The garbage arm (the lowest-probability top-k mutant's TF logits, S512 Q8_0): 11/192, b 142, c 2, p < 1e-4:
    **FAIL**.
  - The prefix-probe instrument run on the exact tier gives 192/192 = the TF reference (S35 Q4_K_M,
    `gate/s35-q4km/CTRL-PROBE-E.txt`), so the probe itself adds no error.

**Correctness verdict: PASS** (exact tier bit-identical everywhere; both new tiers pass on all 8 files).

### Which build each receipt is from (disclosed)

- **A** `f88fba108`: the first frozen candidate. Its gate run was stopped after 2 files (both PASS) when B was
  made; superseded.
- **B** `4d4a1a991` (A + `llb_gemv_anyk`): every MoE gate (G-BITS, G-MODE, G-OFF, G-TF, G-PF) and all 3 speed
  rounds. Its G-REG was 25/28: `LLB_Q4_DP4A=9` on the 0.8B differed, because the residual store had taken the
  partial-quad Q6_K down from the two-plane dp4a GEMV.
- **C** `7f84d5a82` (B + that routing fix, a branch taken only under `LLB_Q4_DP4A` bit 32): the final engine
  commit. G-REG, G-REG4, G-MULTI and G-OFF were rerun on C: all pass. C vs B on S35 Q4_K_M and S128F Q4_K_M,
  greedy + TF: 12/12 identical (`receipts/SPOT-C-vs-B.txt`). The MoE gates and speed arms never set
  `LLB_Q4_DP4A`, so they run the same code on B and C. Engine sha256s: `receipts/build-shas.txt`.

## Speed

Method (prereg, amendments 1 and 2):
- **Decode:** median `stepus` over pos ≥ 12, 128 steps of P1.
- **Prefill:** 512e6 / the median per-prompt period of a 4-copy run of the 512-id prompt (`pfus`; the base's from
  its `stepus` lines, amendment 2; the candidate's exact arm read both ways agrees to 0.1%).
- **llama.cpp:** `llama-bench` tg128 `-r 5` / pp512 `-r 3`.
- 3 interleaved rounds; medians. `receipts/speed/SPEED.txt`, raw lines and json in `receipts/speed/r1..r3`.

Decode tok/s:

| model | base | candidate | x base | llama.cpp CUDA | candidate / llama.cpp |
|---|---|---|---|---|---|
| Qwen3.5-0.8B Q4_K_M | 288.6 | **483.4** | 1.67 | 463.1 | **1.044** |
| Qwen3.5-2B Q4_K_M | 238.2 | **299.5** | 1.26 | 329.4 | 0.909 |
| Qwen3.5-4B Q4_K_M | 47.8 | **135.3** | 2.83 | 174.5 | 0.775 |
| Qwen3.5-9B Q4_K_M | 90.3 | **91.7** | 1.02 | 114.9 | 0.798 |
| S35 Q4_K_M, device | 773.7 | **1733.1** | 2.24 | 1686.9 | **1.027** |
| S35 Q4_K_M, device, dp4a tier | — | **1787.3** | — | 1686.9 | 1.060 |
| S35 Q4_K_M, host | 241.4 | **304.6** | 1.26 | — | — |
| S35 Q4_K_M, stream | 207.0 | **271.2** | 1.31 | — | — |
| S128F Q4_K_M, device | 847.8 | **2409.6** | 2.84 | 2095.7 | 1.150 |
| S128F Q4_K_M, device, dp4a tier | — | **2564.1** | — | 2095.7 | 1.224 |
| S35x8 Q4_K_M, device | 561.2 | **904.2** | 1.61 | 911.7 | 0.992 |
| S35x8 Q4_K_M, device, dp4a tier | — | **932.8** | — | 911.7 | 1.023 |

Marginal cost per MoE layer, (period(S35x8) − period(S35)) / 4:

| | from the median figures | rounds 1 / 2 / 3 |
|---|---|---|
| candidate, exact | **132 us** | 133 / 133 / 131 |
| candidate, dp4a | 128 us | — |
| llama.cpp CUDA | **126 us** | 126 / 128 / 115 |
| base | 122 us | 87 / 122 / 126 |

- The candidate's figure is stable across rounds. The base's swings with its per-token loop, so its
  difference-of-periods is not a clean per-layer figure.
- reports/cuda-moe-speed measured the same exact kernels at 154 us a layer.

Prefill tok/s (512 tokens):

| file | base (exact) | candidate exact | candidate `LLB_PREFILL=fast` | llama.cpp pp512 | fast / llama.cpp |
|---|---|---|---|---|---|
| S35 Q4_K_M | 643.6 | 704.4 (1.09x) | **31,896** | 25,005 | **1.276** |
| S128F Q4_K_M | 688.9 | 783.1 (1.14x) | **48,152** | 49,460 | 0.974 |

### Targets (frozen)

| target | result | verdict |
|---|---|---|
| T1: S35 Q4_K_M device decode, exact ≥ 0.8x llama.cpp | 1.027x | **PASS** |
| T2: dense 0.8B decode, exact ≥ 0.9x llama.cpp | 1.044x | **PASS** |
| T3: S35 Q4_K_M prefill, best gated tier ≥ 0.5x llama.cpp | fast (G-PF PASS): 1.276x | **PASS** |
| T4: every candidate exact arm ≥ 0.98x base | 1.016x (9B) to 2.84x | **PASS** |

Not caught up (information; no target was set on these):
- **2B (0.91x), 4B (0.78x), 9B (0.80x).** With the per-token loop gone, these are bound by the exact tier's GEMVs.
  - On the 9B, one token is 10.2 ms of GPU time, almost all GEMVs at about 75% of the memory bandwidth (the FFN
    gate / up: 42 us against a 31 us floor).
  - llama.cpp runs int8 activations (dp4a). The engine's existing `LLB_Q4_DP4A=9` tier is the dense answer there
    (gated in earlier lanes); it was not re-measured here.
- **The exact tier's prompt** still goes one token at a time through the Bend loop: 704 tok/s on S35. An exact
  prompt run (the decode run's mechanism, fed the prompt's ids) would roughly double that; not built.
- **Fast prefill with Q8_0 experts** uses the int8 GEMV kernels, not the tensor-core leaves (the GGUF's Q8_0
  blocks are not the pack's row-split layout). It is correct (G-PF passes on the 4 Q8_0 files) but slower; its
  speed was not measured.

## Env knobs

| knob | default | effect |
|---|---|---|
| `LLB_GEN` | on | `LLB_GEN=0`: one step a bang (the old decode loop) |
| `LLB_GRAPH` | unset: decode-run steps (and MoE steps) as graphs | `1` graphs everywhere, `0` plain launches in the decode run too |
| `LLB_MOE_DP4A` | off | `1`: the dp4a MoE tier |
| `LLB_PREFILL=fast` | off | now also on MoE files (not with streamed experts) |
| `LLB_STEP_US` | off | adds `pfus` (per-prompt period) to `stepus` |

## Other checks

- `python3 tools/check_no_gpl_tts.py`: PASS.
- A scan of the committed files finds no private paths, IPs, emails or hostnames. llama-bench's model paths in the
  json receipts are cut to file names.
- Every engine and llama.cpp run was wrapped in `systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0`
  under both house GPU locks. Builds ran in scratch only.
- The 8 gate files and S35x8 were regenerated from the seeded commands; every sha256 matches
  reports/cuda-moe-speed (`receipts/gen-sha256.txt`, staged copies in `gate/*/STAGED-SHA256.txt`).

## SUMMARY

- **Built:**
  - **Exact (default, bit-identical):**
    - the decode run: greedy decode no longer pays the Bend loop per token, and its steps run as CUDA graphs with
      the next token read on the device;
    - the embedding table prefetched to the device;
    - the partial-quad pair body;
    - two-rows-a-warp `llb_gemv_any` for K-quants.
  - **New tiers (opt-in, gated):**
    - `LLB_MOE_DP4A=1`: int8 routed experts;
    - `LLB_PREFILL=fast` on MoE files: tokens grouped by expert onto the int8 tensor-core GEMM leaves.
- **Correctness: PASS.**
  - G-BITS 48/48, G-MODE 24/24, G-REG 28/28, G-REG4 3/3, G-MULTI 2/2, G-OFF, G-SIDE 13/13.
  - G-TF (dp4a) 8/8 files: it beats llama.cpp CUDA's own top-1 score on each.
  - G-PF (fast prefill) 8/8 files: two files 1 and 3 positions below llama.cpp's score, not significant
    (p 0.50 / 0.36).
  - Controls: floor-identical PASS, perfect PASS, garbage FAIL.
- **Speed against llama.cpp CUDA** (decode unless noted; medians of 3 interleaved rounds):

  | | before | after |
  |---|---|---|
  | MoE 35B-like | 0.46x (773.7 tok/s) | **1.03x** (1733.1); dp4a 1.06x |
  | MoE 128-expert | 0.40x (847.8) | **1.15x** (2409.6) |
  | 0.8B | 0.62x (288.6) | **1.04x** (483.4) |
  | 2B | 0.72x | 0.91x |
  | 4B | 0.27x | 0.78x |
  | 9B | 0.79x | 0.80x |
  | MoE prefill (35B-like) | 0.026x (643.6 tok/s) | **1.28x** (31,896, fast tier); exact 704 |
  | marginal MoE layer | 154 us (earlier lane) | 132 us, against llama.cpp's 126 |

- **Targets T1–T4: all PASS.**
- **Not caught up:** the 2B / 4B / 9B decode (exact GEMVs, bandwidth-bound), the exact tier's per-token prefill,
  and fast prefill speed with Q8_0 experts.
