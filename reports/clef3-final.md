# CLEF-3 decision head closeout

VERIFIED: our rebuilt CUDA backbone plus plain C CPU head answers all 24
SystemOne goldens with identical top answers (32/32 questions).
Largest absolute probability delta: **0.014678789**. The response is generated
by clef-decide from our engine's hidden rows and the original GGUF's head
weights; no oracle inference is used by the CLI. No download occurred.

## Gates and executed evidence

| Gate | Verdict | Evidence / boundary |
|---|---|---|
| L1 | PASS | 24/24 golden counts; c1-0 145, long state 3020. All IDs and half-open spans match exact reference encode_record with shared C-tokenizer adapter. Piece and whole-text round trips pass. Receipt. |
| L2 | PASS | Clef and four dense stream/logit comparisons identical to CLEF-2. Two requested-row captures byte-identical, finite F32; all 145 real rows captured. Receipt. |
| L3 | PASS (registered metric) | 29 checkpoints each on synthetic multi-question input and real c1-0 dump; max stage-relative errors 1.60097e-6 / 8.59879e-7 <=1e-5. Receipt. |
| L4a | PASS | p(true)=0.95878480462004589 vs 0.9597779556352316; signed delta -0.0009931510151857. Repeat identical, fixed tolerance 0.05. Receipt. |
| L4b | PASS | All 24 requests / 32 questions top-identical; all probability deltas and latencies below. Receipt. |
| L5 | PASS (bounded audit) | Owned paths and bounded source/foreign-state checks; mandatory lock/systemd bookkeeping exceptions registered. |

Gates, metrics, scope and tolerance were committed before their runs in
registration, hook,
head, and CLI. The 0.05 L4a
allowance came from max(0.05, twice the zero captured repeat spread), with the
SYS1A 0.0299 bf16-vs-Q8 result as context from a different head, not a Clef
bound. The older paper plan's proposed 0.002 gate was superseded explicitly
before runs, not loosened after seeing differences. L4b requires exact winners
and reports deltas; it does not assert that every delta is <=0.002.

L3's preregistered relative metric is max(abs(C-Python))/max(abs(Python))
for each stage. Strict elementwise relative 1e-5 is **FAIL** on these checkpoints:
maximum pointwise relative errors are 0.407155 synthetic / 0.024831 real,
near zero. Absolute errors, shapes and pointwise errors for every stage are
in the receipt. Do not promote this stage-scaled pass into elementwise parity
or llama.cpp head/backbone parity. Python NumPy/SciPy expands the reference
independently; tokenizer is shared when checking rendering, so L1 alone is
not an independent tokenizer implementation proof.

## Implementation and stop-rule accounting

Host hook: **45 added lines, zero removed**, one file host/llb_host.c.
Whole emitted-C comparison also has only those 45 additions; no kernel or
compiler-patch changes. Build diff.
LLB_DUMP_HIDDEN, LLB_DUMP_HIDDEN_TOKENS and optional LLB_DUMP_HIDDEN_POS
select prompt-only final output-RMS-normalized rows. Records contain u32
position, u32 width and width little-endian F32 values. Fast prefill and
speculation are refused for capture. Normal mode is unchanged in executed
Clef/Q8/Q4KM/27-default/27-mode9 regressions.

The new C head/math, loader, CLI and header total
**669 lines**:
head.c 298, weights.c 148,
decide.c 204, head.h 19.
The separate L1 renderer/header are 165
lines; adapted SYS1A JSON support/header are 172
lines. Existing tokenizer/GGUF reader primitives are reused without edits.
Python tests/build/receipt scripts are separate. No stop condition triggered:
head port remained below approximately 900 lines; hook did not touch kernels.

The loader reads 160 head tensors directly from GGUF (140 dec.*, 19 decision.*,
and token_types.weight), checks requested shapes/types/bounds and dequantizes
Q8_0/F32/BF16 weights. Lexical means use selected Q6_K output.weight rows,
not input embeddings. CPU FP32 OpenBLAS SGEMM plus plain C affine LayerNorm,
erf GELU, 16-head attention, pooling, routing, four decoder blocks, cosine,
stored transformed scales and scorer produce the logits. A fresh engine
child per CLI request isolates recurrent/KV state. Response probabilities use
double softmax and the oracle's confidence/score formulas.

The CLI rejects missing instructions before GPU work (rc 1, error).
Full precision answers, usage counts, option IDs, choice/score consistency and
legends passed for every golden. Build/use instructions.

## Every request: probability deltas and latency

Delta is ours minus captured golden. Noul false is derived as 1-p(true).
All option deltas, including both noul sides, are in the table and the
full precision CSV.
Our latency is cold CLI wall time including tokenizer/model loading, child
startup, exact token-by-token prefill, CPU head load and execution. Oracle
columns are archived CLEF-1 HTTP timings, not reruns or paired trials. The
45.8 ms warm c1 reference is a different execution lifecycle. These are
single per-case measurements, not serving throughput or a performance win.

| Request | Tops | Max abs delta | Cold CLI ms | Archived oracle ms | Signed deltas by question.option |
|---|---|---:|---:|---:|---|
| 01-outage | PASS | 0.000993151 | 4382.67 | 65.48 | outage: true -0.0009932, false +0.0009932 |
| 02-healthy | PASS | 0.000328516 | 4396.86 | 118.03 | outage: true -0.0003285, false +0.0003285 |
| 03-negation | PASS | 0.000261359 | 4287.58 | 119.29 | outage: true -0.0002614, false +0.0002614 |
| 04-invoice | PASS | 0.002177131 | 5486.56 | 189.24 | status: draft +0.0007350, overdue -0.0021771, paid +0.0014422; large: true +0.0003814, false -0.0003814 |
| 05-route3 | PASS | 0.001035241 | 4340.15 | 47.57 | route: billing -0.0001179, shipping -0.0009174, technical +0.0010352 |
| 06-route4 | PASS | 0.000090281 | 4463.68 | 52.17 | route: billing -0.0000415, privacy +0.0000903, shipping -0.0000283, technical -0.0000205 |
| 07-route5 | PASS | 0.001255102 | 4769.01 | 52.61 | route: finance -0.0001595, hr -0.0010439, legal +0.0000346, sales -0.0000863, support +0.0012551 |
| 08-score2 | PASS | 0.000292464 | 4162.81 | 44.56 | urgency: 0 -0.0002925, 1 +0.0002925 |
| 09-score3 | PASS | 0.001414863 | 4306.62 | 46.42 | urgency: 0 -0.0013971, 1 +0.0014149, 2 -0.0000177 |
| 10-score5 | PASS | 0.009404661 | 4700.25 | 50.94 | severity: 0 +0.0070297, 1 -0.0094047, 2 +0.0011739, 3 +0.0006732, 4 +0.0005279 |
| 11-mixed | PASS | 0.009228895 | 6250.68 | 256.52 | outage: true -0.0092289, false +0.0092289; department: billing -0.0012418, sales -0.0012980, technical +0.0025398; urgency: 0 -0.0009916, 1 -0.0011496, 2 -0.0014962, 3 +0.0036374 |
| 12-nested-json | PASS | 0.003135434 | 5601.95 | 72.21 | delivered: true +0.0000080, false -0.0000080; status: failed -0.0017887, paid +0.0031354, pending -0.0013468 |
| 13-array-state | PASS | 0.000176477 | 4476.35 | 51.03 | healthy: true -0.0001765, false +0.0001765 |
| 14-unicode | PASS | 0.000065310 | 4301.28 | 49.63 | route: account +0.0000317, billing -0.0000653, delivery +0.0000336 |
| 15-null-descriptions | PASS | 0.002010572 | 3977.88 | 44.03 | color: blue -0.0020106, green +0.0012384, red +0.0007722 |
| 17-custom-noul | PASS | 0.000030728 | 4028.28 | 45.07 | ready: true -0.0000307, false +0.0000307 |
| 18-reversed-choices | PASS | 0.000286435 | 4162.13 | 45.24 | route: billing -0.0002771, shipping +0.0002864, technical -0.0000093 |
| 19-empty-state | PASS | 0.001654753 | 4066.05 | 44.18 | evidence: true +0.0016548, false -0.0016548 |
| 20-question-json | PASS | 0.014678789 | 4441.05 | 49.55 | safe: true +0.0146788, false -0.0146788 |
| 21-multi-noul | PASS | 0.000350003 | 6111.98 | 82.01 | database: true +0.0000860, false -0.0000860; cache: true +0.0000950, false -0.0000950; api: true +0.0003500, false -0.0003500 |
| 22-score4-json | PASS | 0.002701118 | 4596.37 | 52.25 | satisfaction: 0 -0.0016530, 1 -0.0013061, 2 +0.0027011, 3 +0.0002581 |
| 23-escaped-state | PASS | 0.000070269 | 4408.40 | 51.08 | incident: true +0.0000703, false -0.0000703 |
| 24-long-state | PASS | 0.000929594 | 43747.38 | 2102.22 | outage: true -0.0005890, false +0.0005890; route: billing -0.0004480, sales -0.0004816, technical +0.0009296 |
| 25-explicit-instructions | PASS | 0.002797623 | 5240.61 | 63.12 | outage: true +0.0003415, false -0.0003415; urgency: 0 -0.0005370, 1 -0.0022606, 2 +0.0027976 |

Observed cold CLI median 4424.73 ms, range
3977.88–43747.38 ms across these unequal request shapes.
Per-stage render/backbone/head timings are in each request's stderr receipt.
For c1-0's first L4a run: render 25.779 ms, backbone 4046.890 ms,
head load 225.172 ms, CPU head 97.958 ms (cold external wall 4432.11 ms).
The exact backbone still computes vocabulary logits on each prompt token and
one discarded next-token argmax; output_tokens=0 describes the decision API,
not absence of that internal work. This overhead is an explicit v1 limitation.

## Failed attempts retained

- L1 first harness run: all 24 goldens and c1-0 passed, then the extra fixture
  was mistaken for its own response, KeyError usage. Fixed only the test-file
  lookup; original log, rerun passed.
- First engine build succeeded but its check expected the runtime .gpu cache
  before first execution. Replaced with complete emitted-C scope diff;
  original log.
- First L2 unit **FAIL / rc 76** on memory rule during 27 mode9: full memory
  PSI avg10 10.16% >=10%. Completed passes retained; owned cgroup stopped.
  Resumed only unfinished checks after PSI fell to 1.14% and GPU was empty;
  no memory threshold changed. Monitor,
  resumed unit.
- First L4a prose transcribed the absolute delta with the wrong sign. Raw
  response and comparison were always correct; STATE and user update were
  corrected immediately. Final tables use the raw receipt.

## Scope, retained artifacts and cleanup

L5 **PASS (bounded audit)**: no observed write outside the authorized worktree
and owned scratch, apart from registered shared Git metadata and mandatory
lock/systemd bookkeeping. All compiler, Bun, Python and CUDA cache/temp paths
were scoped from their first build. No downloads, credentials, Codex config
edits, foreign process kills, or foreign worktree edits were performed.
This is not a complete historical filesystem syscall trace.

The final protected model SHA256 was freshly measured as
`ec1472cd7c72cf933ca2da528c466e0932d7814aca503de53965417eec5b49c2`;
the prior engine remains `7a4c524b...`. CLEF-1 and CLEF-2 worktree statuses
are clean, protected inherited source files match their foreign references.
Hash/status audit,
source audit.
TRAIN0 was not edited or interfered with. Its live status is UNKNOWN here.

Early source-writing shells inherited no TMPDIR. A scoped strace probe of a
16 KiB here-document (larger than the largest source heredoc, 9608 bytes)
observed no temporary-file creation. Thus no
shell staging scope violation was observed. Historical syscall-complete
coverage is UNKNOWN; probe.

All 11 owned units are terminal, with failed attempts preserved.
No clef3 jobs are queued and the final compute-card query is empty.
Units,
memory, GPU.
There were 119 memory samples: minimum MemAvailable
17032120 KiB, minimum SwapFree 15508984 KiB,
maximum load 7.71, maximum full PSI avg10 10.16%
(the registered stop was enforced). No memory threshold was relaxed.

Retained local, ignored build/clef3 contains 76 files,
**39,614,190 bytes**: runnable clef-decide and engine, emitted C/cache,
tokenizer/converter, sidecar, raw selected/all-prompt hidden dumps and all CPU
stage checkpoints. Every retained hash.
No binary was installed, committed as source, or published.

Scratch was inventoried after every owned unit became terminal and removed:
**126,251,008 allocated bytes / 125,070,573 apparent bytes**, 439 entries.
Cleanup,
inventory. No owned job remains.


## Unknowns and next steps

- UNKNOWN: numerical differences from llama.cpp by backbone/head layer;
  attention/BLAS reduction association and activation rounding can contribute.
  L3 uses equal hidden inputs and dequantized weights, not oracle checkpoints.
  No fresh oracle server was started. No unquantized checkpoint was downloaded.
- UNKNOWN: quality on inputs beyond these 24 requests. Exact winner agreement
  here is a battery result, not general decision accuracy.
- NOT RUN / out of scope: multimodal, server and llb integration, fast prefill,
  GPU head, batched serving, physical/other-device validation.
- Explicit v1 limits: no silent oversized-state truncation, no NUL strings;
  literal <<clef: sanitization and exhaustive JSON float-edge equivalence to
  the server are unverified. The span helper's instruction fallback follows
  the Python reference; the decision CLI rejects missing instructions as the
  oracle does. This is a local CLI, not a hardened HTTP service.
- Next: add oracle hidden/head checkpoint capture to localize probability
  deltas; remove vocabulary GEMV in decision-only prefill; implement and gate
  batched final-row export; keep weights resident and profile CPU head versus
  GPU SGEMM; then add server/llb integration with reset/interleaving tests.
  PREDICTION: 4–8 agent-hours for checkpoint localization and readout cleanup,
  8–16 for batched/GPU head and paired timing, 4–8 for service integration.
  These are estimates, not run results or performance promises.
