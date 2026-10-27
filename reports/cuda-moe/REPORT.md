# CUDA MoE: any `qwen35moe` GGUF on the CUDA engine

Branch `x/cuda-moe`, based on `61a673aa3`. The gates are in [PREREG.md](PREREG.md). It was frozen before the
gate runs and then amended twice: amendment 1 is post-hoc and says so; amendment 2 records the scope change.
Receipts are in `receipts/`. The scripts that made them are in `scripts/`. Paths are written as `~`.

## What runs

- **Shapes:** any `qwen35moe`: any expert count, any top-k up to the expert count, and any expert or
  shared-expert width that is a multiple of 256. The rest of the shape follows the dense CUDA rules
  (head_dim 256, SSM state 128, d_model up to 32,768, and so on). MTP blocks (`nextn_predict_layers`) are skipped.
  Experts can be separate `ffn_gate_exps` / `ffn_up_exps` or fused `ffn_gate_up_exps`.
- **Formats:** experts in Q8_0, Q4_K, Q5_K or Q6_K, in any mix per tensor (as Q4_K_M and Q5_K_M files
  have). The router and the shared-expert gate must be F32. The trunk follows the dense rules.
- **Semantics:** the same as llama.cpp's `build_moe_ffn` for qwen35moe:
  - softmax over every expert's router logit;
  - the top k;
  - weights renormalized to sum to 1 (the sum is clamped at 6.1e-5, as ggml does);
  - each expert computes down(silu(gate x) * (up x)), and the results are summed in rank order;
  - plus the shared expert times sigmoid(its gate);
  - plus the residual.
- **Loading:** `gguf2pack MODEL.gguf --sidecar-only SIDE`, then run `LLB_GGUF=MODEL.gguf llb-engine SIDE ids
  --steps N`. The sidecar's virtual layout puts every `*_exps` stack after all the layers. A layer's Bytes on
  the device are therefore its trunk only. The host loads the expert stacks itself, by name, with 64-bit
  offsets and sizes.
- **Expert residency** (`host/llb_host.c` `llb_moe_load`), chosen from what the file needs and what the
  device has:

  | residency | when (auto) | how |
  |---|---|---|
  | device | stacks + 1 GiB fit the free VRAM | `cuMemAlloc` per stack |
  | host | stacks + 4 GiB fit MemAvailable | pinned, device-mapped RAM; the kernels read only the k chosen experts' rows over PCIe |
  | stream | otherwise | the GGUF is mmapped; each token, the k chosen experts of each layer are copied into a k-slot VRAM staging area, so a model larger than RAM runs at page-cache or disk speed |

  `LLB_MOE_EXPERTS=device|host|stream` forces a residency. `LLB_MOE_VRAM` and `LLB_MOE_RAM` (in bytes) stand
  in for the free amounts. That is how the host and stream paths were run on small synthetic files. All three
  residencies give the same bits (G-MODE below).

Code:
- `tools/payload-moeany.patch`: `llb_moe_topk`, `llb_moe_gu`, `llb_moe_dn`, and pass_mail's MoE table checks
  (`Mail.pt` holds 31 words).
- `host/llb_host.c`: `llb_moe_run` and `llb_moe_load`. A MoE layer is never treated as a legacy shape, and
  error code 8 is named.
- `host/llb_pf.h`: fast prefill is off for MoE files, with a note.
- `host/llb_gguf.c`: the MoE virtual order, 3-D Q8_0, and `llb_gg_tensor`.
- `tools/gguf2pack.c`: `moe_sidecar`.
- `engine_qwen35_gpu.bend`: the `qwen3_5_moe` manifest and named checks.

## Gates (final candidate, commit `2cdb87c5d`; build sha in `receipts/build-new.sha.txt`)

All test files are synthetic: random weights from a seeded generator. Regenerating each file gave
byte-identical llama.cpp output (`*/DETERMINISM.txt`).

Arms:
- **R** is llama.cpp CPU greedy (the oracle).
- **F** is llama.cpp CUDA, teacher-forced (the floor).
- **C** is this engine.
- **X** is llama.cpp CPU on the same file dequantized to F32. llama.cpp then quantizes no activations
  (amendment 1). The F32 copy only fit on disk for S512 and S60.

There are 192 teacher-forced positions per file.

| file | shape | C = R | F = R | frozen gate | G-X: C = X, rms(C-X) vs rms(R-X) | file verdict |
|---|---|---|---|---|---|---|
| S512 Q8_0 | d 512, 512 exp / 10, ff 256 | 154 | 168 | FAIL | 192/192, 0.0002 vs 0.343 | **PASS** (G-X) |
| S512 Q5_K_M | same | 139 | 136 | FAIL | 192/192, 0.0001 vs 0.550 | **PASS** (G-X) |
| S60 Q8_0 | d 1024, 60 exp / 6, ff 768 / shared 512, vocab 248,320 | 143 | 167 | FAIL | 192/192, 0.0027 vs 0.332 | **PASS** (G-X) |
| S60 Q4_K_M | same | 122 | 96 | PASS (v2) | 192/192, 0.0002 vs 0.522 | **PASS** |
| S128F Q8_0 | d 1024, 128 exp / 8, fused gate_up, ff 768, + MTP block | 150 | 160 | FAIL (V-T1 D -5.21 pt, CI90 [-9.90, -0.52]) | not run (6 GB F32) | **FAIL** as frozen |
| S128F Q4_K_M | same | 136 | 115 | PASS (v2) | not run | **PASS** |
| S35 Q8_0 | d 2048, 256 exp / 8, ff 512 / 512, 16/2 heads, 16/32 SSM heads | 141 | 158 | FAIL (V-T1 D -8.85 pt, CI90 [-14.58, -3.12]) | not run (13.5 GB F32) | **FAIL** as frozen |
| S35 Q4_K_M | same | 128 | 106 | PASS (v2) (D +11.46 pt, CI90 [+5.21, +17.71]; V-FLIP b 1 c 1) | not run | **PASS** |

**MoE verdict as preregistered: FAIL, on 2 of 8 files (S35 Q8_0, S128F Q8_0).** Those are the two files where
the F32 oracle could not be built on this disk. They fail the frozen gate the same way S512 Q8_0 and S60 Q8_0
did before X showed C matched fp32 on every position. Read the failure with these facts:

- On the Q8_0 files, llama.cpp CPU and llama.cpp CUDA both round activations to 8 bits before the Q8_0 dot.
  This makes F track R more closely than any exact path can. On every file where X exists, C matches X at
  192 of 192 positions with logit rms 0.0001 to 0.0027. R and F are 0.27 to 0.55 away from X.
- On S35 Q8_0 and S128F Q8_0, C's top-1 logit spread from R (rms 0.33 / 0.24) is the same size as llama.cpp
  CUDA's (0.18 / 0.22). The rows are in `receipts/LDIFF.txt`, scored with `scripts/ldiff.py`.
- Information only: smaller files with the same features fit the X arm.
  - `minif` (fused gate_up, MTP block, Q4_K_M): C = X 192/192, rms 0.0002.
  - `mini35` (d 2048 with the tuned F32 router GEMV, 32 experts / 8): C = X 192/192, rms 0.0001.
  - `k72` (512 experts, top-72, beyond the old 64 cap): C = X 192/192, rms 0.0004. Its frozen gate fails
    the same way.

So I call these two files **UNRESOLVED by the instrument**, not passed. A machine with about 15 GB of free disk
can settle them with `scripts/xarm.sh`.

### Controls

All controls are on S35 Q4_K_M, and also under G-X on S512 Q8_0.

| arm | result |
|---|---|
| Perfect (F scored as the candidate) | PASS (v2) |
| Garbage mutant (top-k takes the k lowest-probability experts, scratch build) | C = R 5/192, V-T1 D -52.6 pt, V-FLIP p < 1e-4: FAIL. G-X: 2/192: FAIL |
| No-renormalization mutant (information only) | 42/192: FAIL. G-X: 34/192, rms 1.11: FAIL |

### G-MODE: the residencies give the same bits

On all 8 gate files, `LLB_MOE_EXPERTS=host`, `=stream`, and auto with `LLB_MOE_VRAM=LLB_MOE_RAM=1e9` all
give stdout ids and every `logits` line byte-identical to the device residency. That covers greedy and
teacher-forced runs on 3 prompts: **24/24** comparisons identical (`*/GMODE.txt`). The forced-small auto
case picks stream.

### No regressions

- **G-REG:** the 0.8B, 2B and 9B Q4_K_M and clef-flash Q4_K_S, base `61a673aa3` vs the candidate. 3 prompts
  x 64 steps in the exact tier and `LLB_Q4_DP4A=9`, plus `LLB_PREFILL=fast`: **28/28 runs byte-identical**
  in stdout and every `logits` line (`receipts/GREG.txt`). The 27B was not run.
- **G-SIDE:** dense `--sidecar-only` output, base vs candidate `gguf2pack`: manifest.json and table.bin
  **identical for all 4** (`receipts/GSIDE.txt`).

### Refusals and fallbacks (`receipts/refuse/`)

| input | result |
|---|---|
| F32 experts | gguf2pack: "blk.0.ffn_gate_exps.weight: type F32: the CUDA expert kernels take Q8_0, Q4_K, Q5_K, Q6_K" |
| expert ff 128 | engine: "expert_feed_forward_length (a multiple of 256 ...)" |
| top-k 17 > 16 experts | gguf2pack: "expert_used_count 17 (1..expert_count)" |
| `--emit` | "a MoE model needs the folded step" |
| no `LLB_GGUF` | "a MoE model loads its experts from the GGUF: set LLB_GGUF ..." |
| `LLB_PREFILL=fast` | runs the exact tier with a stderr note; ids and logits identical to the exact run |

The prereg listed "expert_used_count 65" and "expert_count 5000" as refusals. Amendment 2 removed those caps.
A 4,200-expert, top-72 file now runs, and device and stream are byte-identical. llama.cpp itself refuses it
(`LLAMA_MAX_EXPERTS` 512), so it has no oracle (`receipts/bigk4200/NOTE.txt`).

### Speed (information)

S35 Q4_K_M has 4 layers. The engine figure is the mean of the `step` lines over 127 decode steps, at 1 ms
resolution (`receipts/speed/SPEED.txt`).

| residency | ms/step | tok/s |
|---|---|---|
| device | 1.71 | 585 |
| host | 4.41 | 227 |
| stream | 8.18 | 122 |

llama.cpp CUDA `llama-bench tg64` on the same file runs at **1641 tok/s**, so the device residency is 0.36x
llama.cpp on this shape. The expert kernels are a warp-per-row float-dequant GEMV, not a tuned dp4a body.

## Limits that remain, and what bounds them

- **Kernel instantiations:** head_dim 256, SSM state 128 and conv 4, as for dense. Expert and shared widths
  must be multiples of 256 (whole K-quant superblocks).
- **Formats:**
  - experts Q8_0, Q4_K, Q5_K or Q6_K;
  - router and shared-expert gate F32;
  - embedding Q8_0, Q4_K or Q6_K. A Q5_K_M file's Q5_K embedding is refused, as for dense; S512 Q5_K_M was
    quantized with `--token-embedding-type q6_k`.
- **Caps from the file format, not this machine:**
  - each layer's trunk Bytes is at most 2 GiB, with tensor offsets below 2^28 (table.bin's 28-bit field);
  - the head region is at most 2 GiB;
  - 256 layers (`LLB_MAXL`);
  - d_model at most 32,768;
  - vocab at most 261,632;
  - expert count up to 2^32 (a table word); the router scratch holds 2 x count floats, and top-k runs k
    block-argmax rounds.

  Expert stacks have no size cap: 64-bit offsets, one allocation per stack. For 397B-A17B, the trunk per layer
  is about 130M parameters, far below these caps.
- **Memory:**
  - The trunk, head, KV and scratch must fit in VRAM; this is the dense rule.
  - Experts follow the residency table above. The host residency pins its RAM.
  - Stream needs VRAM only for k x the largest expert, and is bounded by disk and page-cache bandwidth.
  - One GPU only: multi-GPU was not simple in this engine and was not done.
- **Dense-only features:** fast prefill (MoE runs every prompt token through the exact tier), `LLB_SPEC`,
  `LLB_MTP`, `LLB_BATCH`, and the split path.
- **Not tested:** any real MoE checkpoint. None was loaded on this machine, by rule.

Other checks: the GPL gate (`python3 tools/check_no_gpl_tts.py`) passes, and a scan of the committed files
finds no private data.

## SUMMARY

- **Built:** the CUDA engine now runs any `qwen35moe` GGUF: any expert count, any top-k, any 256-multiple
  expert or shared width, separate or fused gate/up, and MTP blocks skipped. Experts can be Q8_0, Q4_K, Q5_K
  or Q6_K, held in VRAM or pinned host RAM, or streamed from the mmapped GGUF. The residency is chosen from
  free VRAM and RAM, or forced with `LLB_MOE_EXPERTS`.
- **Exactness:** on every file where the F32 oracle fit on disk (S512 x2, S60 x2, and the minis), C matches
  llama.cpp-on-fp32 at 192/192 positions with logit rms <= 0.0027. llama.cpp's own quantized CPU and CUDA
  runs are 0.27 to 0.55 away.
- **Preregistered verdict: FAIL on 2 of 8 files** (S35 Q8_0, S128F Q8_0). The frozen gate counts llama.cpp's
  8-bit activation noise against the candidate, and the F32 oracle that settled the same failure on the other
  Q8_0 files does not fit on this disk. These two are unresolved, not passed. The other 6 files PASS.
- **Controls:** the perfect candidate PASSes, and the garbage mutant FAILs, both under the frozen gate and
  under G-X.
- **Residencies:** device, host and stream are byte-identical (24/24).
- **Dense regressions:** 28/28 runs byte-identical to the base, and all 4 sidecars identical.
- **Speed:** device 585 tok/s on the 4-layer 35B-like Q4_K_M file, 0.36x llama.cpp CUDA. Host 227 tok/s,
  stream 122 tok/s.
