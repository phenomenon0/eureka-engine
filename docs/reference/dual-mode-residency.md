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

This is version 1 of the queue policy.

- The engine serves requests one at a time, in FIFO order. Requests wait in
  the stdin pipe. There is no batching and no preemption.
- The KV need of a request must fit the capacity fixed at launch. The need is
  prompt + n - 1 tokens, or n for decide. An oversized request gets an error
  line. The engine does not truncate it.
- Malformed JSON, missing fields, and over-capacity requests get an error
  line, and the request loop continues.

## Limitations

- Requests block each other at the head of the line. Decide uses the exact
  prefill, at about 12.8 ms per prompt token. A 3020-token decide holds the
  engine for about 47 s, and a generate request queued behind it waits that
  long.
- If mixed traffic matters, the planned next step is a priority lane, with
  generate first, or splitting long decides. Both need the request loop to
  see more than one pending request at a time.
- Inputs that `clef_encode` rejects still call `clef_fail` and end the
  process. To return error lines for them, the encoder must not exit.
