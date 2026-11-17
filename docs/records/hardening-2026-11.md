# Hardening review, 2026-11-17 (engine items)

A review of the app and this engine ran on 2026-11-17 at engine commit `f10a5fa`. It used one reviewer per surface,
and a second reviewer re-read every critical or high finding. These are the engine-side findings. The parts marked
"all" are shared code, so a fix must hold on Linux, macOS and Windows.

| # | Item | Where | Platforms | Severity | Status |
|---|---|---|---|---|---|
| E1 | The tokenizer loader copies token and merge bytes using lengths read from the file, and never checks them against the buffer it allocated. A crafted or mismatched tokenizer can overflow the heap at model load. The 32-bit size multiplies can also wrap. | `tools/qwen35_tokenizer.c` (`rd()`, the vocab and merges loops) | all | high, confirmed | fixed: counts bounded, blob buffers sized by the bytes left in the file, every length checked against them, special ids checked, mallocs checked. The old loader overflowed the heap when only `max_len` was set to 0; the new one matches the old output on the 0.8B and 35B tokenizers and is clean under AddressSanitizer over 315 damaged files. |
| E2 | Backbone tensor size and offset arithmetic can wrap, which defeats the truncation check. A downloaded GGUF can then point a read past the mapping. | `host/llb_gguf.c` (tensor info, about line 303) | all | medium | fixed: every dimension bounded and the product checked before it is used; offsets compared without sums that can wrap. A crafted header that wrapped `src + bytes` passed the old check and is refused now. |
| E3 | The mmproj F16 widening uses unchecked multiplies. | `host/llb_gguf.c` (about line 773) | all | medium | fixed: projector dimensions bounded; every typed tensor (BF16 too, which the old check skipped) checked against the file size without wrapping sums. |
| E4 | The layer index from the file is not bounded, and the alignment is not validated (the rounding can wrap). | `host/llb_gguf.c` (about lines 263 and 270) | all | low | fixed: layer numbers above 4096 and alignments that are not a power of two up to 1 MiB are refused (both parsers). |
| E5 | `vpre_decode` uses a 32-bit `ftell`, which overflows the heap for files of 2 GB or more. This affects the CLI only. | `vision/vpre.c` (about line 50) | all | low | fixed: a photo file over 256 MB, a PPM side over 16,384 or an allocation failure is refused before any copy. |
| E6 | A Stop pressed during prefill is lost. The cancel event is reset at `gen_run` entry, so a cancel set while the prompt is still being read counts as stale. The step-per-bang paths (photos, grammar, sampling) never check cancel or end-of-generation, so a photo reply runs every step. | `backends/windows/wincuda.ps1` edit 2 | Windows CUDA; check the Linux CUDA and Metal cancel paths | medium | partly fixed in the app: llb-serve resends the cancel every 250 ms while it drains (a Stop during a long prompt now ends in 0.26 s, was 7.9 s). Open: the step-per-bang paths (photos, grammar, sampling) still run every step; the fix is a gen_eog/gen_cancel check in that path (wincuda.ps1), or llb-serve finishing the HTTP reply at the end id and draining behind it. |
| E7 | The cancel and exit events use the default DACL, so any process in the same session can cancel or stop the engine. | `backends/windows/wincuda.ps1`, the app's `serve_win32.c` | Windows | low | open |

Checks: the old and new parsers give identical tables for the 9B, 27B and 35B-A3B GGUFs and a projector; four
crafted headers (wrapped dimensions, a wrapping offset, alignment 3, layer 4000000000) were all accepted before and are
all refused now. The fix for E1, E2 and E4 is the same: check every length and count read from a model file against the bytes left,
using `size_t` and overflow checks before multiplying, and refuse the file with the field named.

Not reviewed: `tools/gguf2pack.c`, the rest of `host/llb_host.c`, the tokenizer encode and decode paths, the vision
tower's destination sizing (`vmm.c`), and the Windows compat shims. Malformed inputs were not constructed.
