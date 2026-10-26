# Module contract

This page defines version 0 of the module and pack contract (twoheads S6-a,
2026-10-11). It fixes the vocabulary for modules and the pack containers that
hold them.

The planned end state, from the fusion plan, is a single process with one
backbone and N modules. In that design, the engine loads, enables, and mixes
modules.

Each statement on this page has one of three statuses:

- Tested: run or read in the S6-a session.
- Not tested: believed, but not run.
- Planned: a prediction or a later increment.

## Module classes

A module class describes what a module consumes and produces.

| Class | Consumes and produces | Current instance |
|---|---|---|
| Row consumer | Reads trunk hidden rows and produces output. Runs on the same engine pass. | The clef decision heads, head0 and head2. Tested. |
| Front end | Produces tokens or embeddings that are fed into the trunk. | The STT sidecar (audio-crew, port 8090). Planned: the Qwen3-VL ViT (fusion F1). |
| Generator | Produces sequence output other than text, such as codec or audio heads. | None. This is a design class. |
| Sidecar | An endpoint in a separate process. | The audio-crew STT and TTS services (recipe tested), and LM Studio embedders. |

## Seams

A seam is a point where a module connects to a request.

| Seam | Contract | Status |
|---|---|---|
| Attach | By path. Today this is an environment variable, `LLB_HEAD2_GGUF`. Later, a container as described in [Pack containers](#pack-containers). | `LLB_HEAD2_GGUF` tested. |
| Select | Per request: `"head":"head0"\|"head2"`. This generalizes to a NAME key in a module registry. The registry today holds head0 and the resident head2, in two slots. An N-slot registry is a later increment. | Per-request select tested. N-slot registry planned. |
| Splice | Into the response tail: `"head":"<name>"` and `"head2":{...}`. Removing the spliced fields gives exactly the plain golden response. | Tested. |
| Persist | Sessions, retention, and continuation (S2, S4, S5). The selection is part of the record identity, so a change to the head only is a different request. When the key is absent, behavior is byte-identical to the default. | Tested. |

## Pack containers

| Container | Purpose | Status |
|---|---|---|
| GGUF | Heads that use full GGUF features. | Current. |
| AUX1 | A table of names, dtypes, and shapes for auxiliary packs. Introduced in S6-a, in `tools/auxpack/`. | Current. |
| MTP1 | The legacy positional table. | Legacy. Still readable. |

### GGUF heads

The clef3 head loader contract stays exactly as it is today:

| Item | Value |
|---|---|
| Tensor count | 161 |
| Keep filter | `dec.*`, `decision.*`, `token_types.weight`, `output.weight` (`tools/clef3/weights.c:74`) |
| `output.weight` | Q6_K, 248320×4096 (`tools/clef3/weights.c:95-97`) |
| Other tensors | F32, Q8_0, or BF16 |

### AUX1

AUX1 generalizes the MTP1 positional table, so that a pack describes each
tensor by name instead of by position. The MTP1 producer is
`tools/mtp27/mtp_pack.py` (its ORDER list). The MTP1 consumer is `mtp_load`
in `host/llb_host.c` (its `want[17]` array).

The layout is little-endian:

1. The magic `AUX1`.
2. `u32 version`, equal to 1.
3. `u32 n`, the number of entries.
4. `u32 flags`.
5. n entries of 72 bytes each.
6. The payloads, each 256-aligned, in entry order.

Each entry has these fields:

| Field | Type | Meaning |
|---|---|---|
| `name` | `char[32]` | Tensor name. |
| `dtype` | `u32` | 0 F32, 1 BF16, 2 Q8_0, 3 Q4_K, 4 Q6_K, 5 U32, 6 BLOB. |
| `rank` | `u32` | Number of dimensions. 0 means opaque. |
| `ne` | `u32[4]` | Dimension sizes. |
| `off` | `u64` | Payload offset, 256-aligned. |
| `bytes` | `u64` | Payload size. |

Tools:

| Tool | Functions |
|---|---|
| `auxpack.py` | `convert`: MTP1 to AUX1. Names come from the mtp ORDER list plus `mtp.head_slice` and `mtp.ids`. Also `mkfixture`, `mkbads`, and `show`. |
| `auxpack.c` | Reader with `aux_open`, `aux_find`, and `aux_expect`, plus `--selftest`. |

Tested in S6-a:

- A real 266 MB MTP1 pack, the 27B sidecar `blk.64.*` plus a 64-id head
  slice, converted to AUX1. All 17 named entries are present, and their
  sizes match the producer.
- The fixture read and validate test passes.
- The negative tests for a missing name, a tampered shape, and a truncated
  file are all rejected with clean errors.

### MTP1

MTP1 is the legacy format and is still readable. Engine-side AUX1 reading in
`mtp_load` is the next increment (S6-b). In it, names replace position as
the identity, and byte sizes are validated against the same `want[]`
calculation.

## Required rules

These rules carry over from earlier lanes.

1. When a key is absent, behavior is byte-identical. The selection is part of
   the record identity.
2. Gates are byte-exact against cold references. Equivalence is established
   by measurement.
3. Heads are trained against the resident trunk. An attach to a different
   trunk never counts.
4. Any host-only hook compiled into shared entry blocks needs a device-lane
   stub (precedent: W5-a).
5. Every GPU run takes both house locks, applies the VRAM gate of at least 8 GiB,
   and labels co-tenants.

## Planned consumers

These are the next consumers of AUX1 and this contract.

| Consumer | Plan |
|---|---|
| `mtp_load` reads AUX1 (S6-b) | An engine edit on the P2 route. The gate is identical speculative-decoding statistics on the converted pack. |
| head2 (W6-B) | A clef3-layout head trained on clef-flash. It loads through the existing GGUF head loader, unchanged. |
| Embedding head | A pool-and-project pack, in GGUF or AUX1, trained on trunk rows. It fills the embedder slot that the engine owns. |
| Audio and vision packs | The first consumers in the fusion plan milestones: F1 (vision ViT port) and F2 (audio sidecar). |

## Limitations

- The module registry has two slots: head0 and the resident head2. An N-slot
  registry is not built.
- The engine does not read AUX1 yet. `mtp_load` reads MTP1 by position until
  S6-b lands.
- No generator module exists.
