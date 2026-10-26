# Decide configuration

This page describes the configuration that the server uses for decide
requests (G15, 2026-10-14). It lists the settings, the models where fast
prefill is the default, the fast-path format check, the measured tiers, and
the benchmark numbers.

`exact` is the golden oracle. Use it for receipt runs only.

## Signed bound

This bound was signed on 2026-10-14. Do not change these values.

| Item | Value |
|---|---|
| Labels | 100 % vs llama.cpp CPU |
| Median \|Δp\| | ≤ 0.02 |
| Max \|Δp\| | ≤ 0.2 |

Measured against the bound on the 4B Qwen3.5-4B-Q4_K_M-openjev-wide model:

| Measure | Result |
|---|---|
| Labels | 22/22 |
| Median \|Δp\| | 0.003 |
| Max \|Δp\| | 0.197 |

## Settings

| Setting | Decide path | Oracle (`--exact`) |
|---|---|---|
| `LLB_PREFILL` | `fast`: q8_1 activations, int8 tensor-core GEMMs, 512-token chunks. | Unset: exact, token-serial. |
| `LLB_PREFIX_CACHE` | `1`: reuses the shared system and state prefix (W1-KVPFX). | Unset. |
| `LLB_SAMPLE` | `preset=greedy`. Request `sampling` fields override it per request (W1-SAMPLER). | Engine default (greedy). |
| Render | Request order: the options and state in the order the request gives them. Never sorted. | Same. |

## Server and engine behavior

- `llb-serve --decide-path` sets the three environment variables above on the
  engine child process.
- `llb-serve --exact` is the explicit oracle. It is the default.
- Every response carries `X-LLB-Prefill: fast|exact`.
- `sys1.c` admits `LLB_PREFILL=fast`. Requests that read hidden rows at every
  position, such as the CLEF decide head, take the exact pass regardless. The
  fast chunks never call `llb_hidden_put`, so they emit no rows.
- `dual.c` maps the request's `sampling` object onto the `--sample` word of
  the sampler: temperature, top_p, top_k, min_p, seed, and presence,
  frequency, and repeat penalty.
- The engine with no flags produces the same bytes as the base server
  (W1-SERVER). Gate A1: 62/62 byte-identical.

## Fast prefill by model

| Served model | Fast default | Reason |
|---|---|---|
| 4B openjev-wide Q4_K_M (native 4bq4 engine, `llb run 4b --ids -n 1`) | Yes. This is the signed decide path. Run it with `LLB_PREFILL=fast`. | The bound was measured on this model. All GEMMs are Q4_K, Q5_K, Q6_K, or Q8_0. |
| 9B clef-flash Q4_K_S (sys1 engine behind `llb-serve`) | Labeled fast tier with `LLB_PF_SSMQ8=1` (SHIP45, 2026-10-16). The `llb-serve` default stays `--exact`. The 9B serving paths that ask for fast (`--decide-path` or `LLB_PREFILL=fast`) set `LLB_PF_SSMQ8=1`. Without it, fast on this model falls back to exact. See [Fast-path format check](#fast-path-format-check). | `ssm_alpha` and `ssm_beta` are F32 GEMMs. In the 4B they are Q8_0. The fast GEMM covers only Q4_K, Q5_K, Q6_K, and Q8_0, so the fast tier reads a Q8_0 shadow of them. See [9B fast tier](#9b-fast-tier). Before the format check existed, the engine exited on the first fast chunk with `fast prefill: a GEMM format outside Q4_K / Q5_K / Q8_0 / Q6_K`, and a fast-default `llb-serve` answered 503 to every request. The default was flipped in `291cba19` and reverted in `b6689c37`. |
| 27B Qwen3.8 Q4_K_M (q27c engine, `llb run 27b`; pack-q27a or GGUF plus sidecar) | Yes. This is the `llb` CLI default: `defaults.conf` sets `LLB_DEFAULT_PREFILL=fast` and `models.conf` sets fast=1. It was re-judged under Verdict Gates v2 (PREFILL-27B, 2026-10-16): PASS, industry-class. Use `--exact` per run. `llb-serve` cannot serve the plain 27B yet (glue reports `output.weight wants 4096 cols`; a d5120 plus NOHEAD build is still owed), so there is no 27B `--decide-path`. | All GEMMs are Q4_K, Q6_K, or Q8_0 (PF27). The 27B has no CLEF head on this path, so no hidden-row decide is affected. |

### Mode disclosures

Declare the matching disclosure wherever you serve one of these tiers. Use
the text as written.

27B fast tier:

> fast (27B Qwen3.8 Q4_K_M): prefill on int8 tensor-core GEMMs (PF27 tier). Verdict Gates v2 vs the industry-class
> floor (llama.cpp CUDA teacher-forced on the exact path's tokens), 2,123 matched positions at 1K and 3K prompts:
> top-1 vs exact 98.82 % (floor 98.92 %; paired D −0.09 pt [−0.52, +0.33]); KL(exact‖fast) / KL(exact‖llama) 0.84
> [0.75, 0.95]; 1K set: all 13 free-run divergences from exact start at near-ties (exact margin ≤ 0.023), 6 of 13 on llama.cpp's side. PASS. 1,247 / 1,284 / 1,259
> tok/s at 1,009 / 3,011 / 8,192 (35–52× exact; 0.91–0.95× llama.cpp default `-fa 1`)..

9B fast tier:

> fast (9B clef-flash): prefill on int8 GEMMs with Q8_0 ssm_alpha/beta shadow; mean KL to exact 0.0169 ≈ the
> clef-flash exact-route floor; top-1 90% at word starts (flips are near-ties; 42/77 shared with llama.cpp’s own
> CUDA route); decides remain exact. Kept under Verdict Gates v2 (industry-class PASS).

### Options to enable fast prefill on the 9B

Each option is a separate lane.

| Option | Status |
|---|---|
| Add an F32 path, or an F32-to-Q8_0-on-load path, to the `pf` GEMM dispatch. | Done: `LLB_PF_SSMQ8=1`. See [9B fast tier](#9b-fast-tier). |
| Make a 9B pack with Q8_0 `ssm_alpha` and `ssm_beta`. | Not done. This changes the numerics, so the bound must be measured and signed again. |
| Make `pf_fast()` check every layer's format first and fall back to exact instead of `err_fail`. | Done. See [Fast-path format check](#fast-path-format-check). It is safe and gives no speed gain on this model. |

With any of these options, the CLEF hidden-row decide stays exact until
`pf_chunk` can emit per-position rows.

## Fast-path format check

The engine uses fast prefill only when every GEMM it would run has a fast
format (S6-GUARD, 2026-10-15). Otherwise it serves the request exact and
writes one line to stderr. The server never exits and never answers 503 for
this reason.

The rule is the same one that `pf_mm` enforces:

- The format is Q4_K, Q5_K, or Q6_K, or Q8_0 with an input width that is a
  multiple of 128.
- Every input width is a multiple of 256 (`pf_q8`).

There are two checks.

| Check | Behavior |
|---|---|
| `llb-serve --decide-path` | Reads the GGUF tensor table at start. It reads no weights and takes well under 0.2 s on a 9.5 GB file. It checks the `blk.*` GEMMs: `ffn_gate/up/down`, `attn_qkv`, `attn_gate`, `ssm_alpha/beta`, `ssm_out`, `attn_q/k/v`, and `attn_output`. If any check fails, it logs the line below and serves exactly like `--exact`: no fast environment, no prefix cache, and `X-LLB-Prefill: exact`, so the header stays accurate. An unreadable tensor table also falls back to exact. |
| The engine (`host/llb_pf.h`, `pf_formats`) | The backstop for `LLB_PREFILL=fast` set by hand. Once per process, before it allocates the chunk buffers, it walks the layer table: slots 2–8 and 13 for GDN layers, and 2–4, 14–16, and 19 for full layers. On a miss it logs the line below and the tier stays off. The output is the same bytes as with `LLB_PREFILL` unset. This replaces the mid-request `err_fail`. |

Log line from `llb-serve`:

```text
serve: --decide-path refused for this model (blk.0.ssm_alpha.weight is ggml type 0 (width 4096) = F32; ...): serving --exact
```

Log line from the engine:

```text
fast prefill: off for this model (layer 0 GEMM slot 7 is F32, width 4096; ...): every step exact
```

A model whose GEMMs all pass keeps fast prefill. The `llb-serve` check
passes the 9B Q8_0 on CPU. The engine check on such a model is not tested on
GPU. When fast prefill is not requested, no new code runs.

Tested on 9B clef-flash with the G15 battery from R1:

| Arm | Result |
|---|---|
| Flag off | A1 62/62 == REF, decide noul `0.95878480462004589` ×3, A4 6/6, serve default and `--exact` == REF g0. |
| `llb-serve --decide-path` | All 200, `X-LLB-Prefill: exact`, ids == REF g0, one refusal line. |
| The `--decide-path` engine environment | rc 0, 62/62 == REF, one "off" line. |
| The prefix-cache (KVPFX) fast arms | 16/16 == LIVE. |

## Mode-9 decode on the 9B

Mode 9 (`LLB_Q4_DP4A=9`) is llama.cpp-class int8 dp4a two-plane decode GEMVs,
the 27B's mode 9, on the 9B clef-flash sys1 engine (QUANTS-AUDIT, prepared
2026-10-16, owner-gated).

| Measure | Result |
|---|---|
| Speed | +11.66 % [+10.76, +12.58] vs exact (BEST-AR-VERDICTS Q1 r2, bar +8 %, KEPT). Free-running prefix 1.000. |
| KL half (prereg 52a7363d) | PASS. K2 24/24, top-1 vs exact 100 % (floor 99.02 %), KL(llama‖m9)/KL(llama‖exact) 1.0001× [0.9994, 1.0009], 2,048-position drift 0.00115 vs budget 0.696, 2,048/2,048 free-run identical. |
| Decide battery G4 (prereg `QUANTS-AUDIT/m9-g4/PREREG.md`) | Labels 33/33 and \|Δp\| ≤ 0.005 in the `--exact` and `--decide-path` shapes. It also classes decides as BIT-EXACT or LABEL-EXACT. Only a BIT-EXACT PASS keeps the "decides stay exact" scope. LABEL-EXACT needs the decide scope signed again, or m9 limited to generation. |

The 9B decide scope was re-signed under the tolerance amendment on
2026-10-17. The sys1 9B path runs mode 9 by
default at class LABEL-EXACT. To opt out, set `CLEF_CANON=1` or
`LLB_Q4_DP4A=0`.

Declare this disclosure wherever you serve the 9B m9 tier. Use the text as
written:

> m9 decode (9B clef-flash): int8 dp4a GEMVs; not bit-identical to exact; KL to exact ≈ 3e-8 (1.0001× the llama.cpp
> route ratio); top-1 identical on 1,536 scored + 2,048 free-run positions. Kept under Verdict Gates v2.

## 9B fast tier

The 9B fast tier is set by `LLB_PF_SSMQ8=1` (SI-FASTDECIDE, shipped
2026-10-16). The flag is off by default.

| Item | Description |
|---|---|
| Mechanism | With `LLB_PREFILL=fast` and `LLB_PF_SSMQ8=1`, `host/llb_pf.h` builds a Q8_0 shadow of the 48 F32 `ssm_beta` and `ssm_alpha` GEMMs (GDN slots 7 and 8). It uses ggml `q8_0_ref` rounding in the pack's row-split layout (`llb_gguf.c:307`), once per process, in about 6.7 MB of VRAM. The fast chunks read the shadow. The exact path keeps reading F32. |
| Format check | `pf_formats` and the `llb-serve` check admit slots 7 and 8 only when the flag is on. With the flag off, the format check falls back to exact, as before. |
| Accuracy | Not byte-exact. Tolerance-gated on 757 word-start probes. Mean KLD 0.0169, at clef-flash's llama.cpp R2 floor of 0.0165. Same-top 89.8 %. The flips are near-ties. |
| Verdict | Verdict Gates v2: industry-class PASS, exact-class FAIL. For this reason it is a labeled tier and never the oracle. |
| Decides | Decides stay exact by construction: hidden-row requests take the exact pass. S2: 24/24 == REF. |
| Flag off | 62/62 byte-identical. |
| Speed | Prefill 74 → 3,356 tok/s at 3.5K (45×). |

## Benchmarks

These are the corrected numbers from the G15 session. The raw logs are in the
project's lab archive.

### 4B decide

The 4B decide runs on the native 4bq4 engine, with fast prefill and
request-order (server-faithful) ids. The llama.cpp column is decide-speed's
GPU `/v1/systemone`, warm, median of 5.

| Item | Tokens | llama.cpp warm | Fast warm (median of 3) | Fast / llama | Cold wall s (load s, decide ms) |
|---|---|---|---|---|---|
| finer-62008 | 756 | 121.0 | 152 | 1.26× | 1.89 (1.5, 155) |
| finer-51069 | 717 | 117.9 | 145 | 1.23× | 2.35 (2.0, 155) |
| bfcl-25 | 562 | 95.7 | 125 | 1.31× | 2.25 (1.9, 135) |
| bfcl-7 | 502 | 80.8 | 102 | 1.26× | 1.41 (1.1, 105) |
| trialgpt-866 | 315 | 56.9 | 78 | 1.37× | 1.58 (1.3, 82) |
| trialgpt-412 | 351 | 63.1 | 84 | 1.33× | 1.53 (1.3, 86) |
| tos-2350 | 291 | 55.4 | 77 | 1.39× | 1.65 (1.4, 81) |
| tos-3008 | 279 | 51.2 | 76 | 1.48× | 1.67 (1.4, 77) |
| cfpb-aff57b375c38 | 441 | 73.0 | 94 | 1.29× | 1.83 (1.5, 99) |
| cfpb-680c39496370 | 328 | 58.2 | 83 | 1.43× | 1.84 (1.6, 86) |
| Total | 4,542 | 773 | 1,016 (runs 1,023 / 1,013 / 1,011) | 1.31× | 1.4–2.4 s |

- Exact in the same session took 98,892 ms for the same 10 items, 97× slower
  than fast. This reproduces decide-speed2 (1,020 ms / 98,255 ms).
- Cold wall time here is 1.4–2.4 s, not decide-speed2's 4.6–6.8 s, because
  the load took 1.1–2.0 s with the pack in page cache. A cold disk is slower.
- The ratio is worst on short prompts, because the last token is one exact
  step (about 22 ms).

### 9B serving engine, exact

REF is the base server (W1-SERVER) at `19f765a2`. OFF is the G15 engine with
no flags.

| Workload | REF | OFF |
|---|---|---|
| 24 CLEF decide goldens | 96.3 s | 95.0 s |
| openjev-22 as generate n=1 | 111.1 s | 110.6 s |

1.1k-token agent turns took ≈12.6–12.8 s each.

There are no 9B timings for FAST and DP in this session. Before the format
check existed, the engine exited on the first fast chunk. For the 9B with
`LLB_PF_SSMQ8=1`, see [9B fast tier](#9b-fast-tier).


## Limitations

- `llb-serve` cannot serve the plain 27B, so there is no 27B
  `--decide-path`.
- On the 9B, fast prefill needs `LLB_PF_SSMQ8=1`. Without it, fast prefill
  falls back to exact.
- The CLEF hidden-row decide stays exact until `pf_chunk` can emit
  per-position rows.
- The engine format check on a model whose GEMMs all pass is not tested on
  GPU.
- The 9B fast tier and mode 9 are not byte-exact. They are labeled tiers,
  not the oracle.
- The raw logs behind this page are in the project's lab archive, not in
  this repository.
