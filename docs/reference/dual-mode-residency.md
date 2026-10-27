# Dual-mode residency and queue policy

This page describes what the dual-mode engine keeps in memory and how it
queues requests. One OS process, `llb-engine` with `LLB_SYS1=1`, holds one
model for both generate and decide modes.

## Residency

Sizes were measured on 2026-10-07.

| Item | Location | When loaded | Size |
|---|---|---|---|
| Backbone weights (32 layers, embedding, output) | Device, `cuMemAllocManaged`. `host/llb_gguf.c` fills them from the GGUF with pread. | Once, at start. | Card `memory.used` is 6491 MiB above the empty card for the whole process. |
| KV and recurrent state | Device. Sized once from the arguments `0 --steps C` (dT = lg(C)). The default C is 4096. | Reset at position 0 for every request (the multi-prompt path). | Included in the figure above. |
| CLEF-3 FP32 head (dequantized head tensors) | Host RAM of the same process (`tools/clef3/head.c`). | Once, at the first request. | VmRSS 1.82 GB after the first request. VmHWM 2.01 GB for the session. |
| `output.weight` rows of the answer-option ids | Read from the GGUF with pread for each decide request (`clef_output_row`). | Per request. | A few rows. |
| Tokenizer | Host RAM (`tokenizer.bin` from the sidecar). | Once. | Small. |
| Hidden rows (decide only) | `$TMPDIR/sys1dual-<pid>.hidden`. Rewritten for each decide and unlinked at EOF. | Per request. | n × 4096 × 4 B. 3020 tokens use 49 MB. |

The engine opens the GGUF by file descriptor and reads it with pread. It
never maps the file into memory: `/proc/PID/maps` has no `.gguf` line in any
sample.

One GGUF and one sidecar serve both modes. The sidecar holds `chat.flags`,
`manifest.json`, `table.bin`, and `tokenizer.bin`.

### Measure device memory

The per-PID `used_memory` column in nvidia-smi (598 MiB) does not count
managed memory. Use the card-level `memory.used` value for the real device
footprint. It is 6491 MiB for one process, and 12597 MiB for two resident
processes.

## Queue policy

Version 1 of this policy (2026-10-07) served one request at a time in FIFO
order, with no batching. The server has since gained the features below. Each
item is byte-identical to the same request served alone unless a row says
otherwise.

| Feature | Flag | What it does | Measured |
|---|---|---|---|
| One task, many questions | (default) | Every question about one state is answered from one prefill. | One prefill instead of N. |
| Batch requests | `{"batch":[...]}` | N items in one HTTP request, one queue ticket, back to back on one slot. | Saves the per-request HTTP time. |
| Native batch | `--native-batch N` (2..8) | N decide sequences run in one forward per step. Each weight is read once for all rows. | 9B clef-flash, RTX 3090: 24 items 113.4 s → 75.0 s (≈1.5×). |
| Continuous batch | `--native-batch N --native-cont` | A task joins a free row at a step boundary instead of waiting for its group. | First item at 10.2 s, against 18.7 s grouped and 21.8 s in order (N = 8). |
| Swappable heads | `--heads name=GGUF[,...]` | Up to 6 more decide heads beside head0. A decide picks one with `"head"`. The trunk pass never sees the head, so rows in one batch can use different heads. | Mixed-head batch 74.9 s against 75.2 s for one head. ≈0.46 GB host RAM a head, no device memory. |
| Engine slots | `--parallel N` | N engine slots; `GET /slots` lists them. A batch stays on one slot. | — |
| Prefix cache | `LLB_PREFIX_CACHE=1` | Reuses the shared system and state prefix across requests. | — |
| Fast decide prefill | `--decide-path` / `LLB_PREFILL=fast` | int8 prefill for decide where the head allows it (the 4B signed path). Heads that read every position, such as Clef on the 9B, take the exact pass. | 4B: labels 22/22, median \|Δp\| 0.003 against llama.cpp CPU. |

Details and gates: [HTTP API](http-api.md#batch-requests),
[Native batch](http-api.md#native-batch),
[Continuous batch](http-api.md#continuous-batch),
[Multi-head batch](http-api.md#multi-head-batch), and
[Decide configuration](decide-configuration.md).

Still true of every slot:

- The KV need of a request must fit the capacity fixed at launch: prompt + n - 1
  tokens, or n for decide. An oversized request gets an error line. The engine
  does not truncate it.
- Malformed JSON, missing fields, and over-capacity requests get an error line,
  and the request loop continues. A decide the engine cannot encode returns 400,
  and the engine stays up.

## Limitations

- Within one slot, a long exact-tier decide still holds that slot until it
  ends. Native and continuous batching share the slot's forward among decides,
  and `--parallel` adds slots, but generate requests are not batched.
- The residency figures above were measured on 2026-10-07 with one slot.
