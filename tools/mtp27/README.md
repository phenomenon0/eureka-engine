# MTP drafter pack

`tools/mtp27/` builds the drafter pack for `LLB_MTP` from a clean checkout.
With `LLB_MTP=<pack>`, the engine runs the Qwen3.8-27B NextN head as the
speculative drafter. It is used with `LLB_SPEC=k` and `LLB_Q4_DP4A=9`, the 27B
Q4_K_M mode-9 trunk. See
[`reports/llb-mtp27-final.md`](../../reports/llb-mtp27-final.md) (mtp27 and
mtp28 lanes).

## Files and inputs

| Input | Location |
|---|---|
| NextN sidecar `blk.64.*` (15 tensors) | `MTP_SIDE` (required): `mtp-Qwen3.8-27B-Q4_0.gguf`. |
| Trunk `output.weight` (Q6_K rows of the draft vocabulary) | `MTP_TRUNK` (required): `Qwen3.8-27B-Q4_K_M.gguf`. |
| Draft vocabulary | In the repository: [`draft_ids_40960.json`](draft_ids_40960.json) (40,960 distinct ids, max 248076). Extracted from the `mtp.ids` entry of the production pack. u32 sha256 `e3a69c2d…7801600`, checked by the script. |
| Build script | [`make_drafter.sh`](make_drafter.sh). |

## Before you begin

- Install python3 with `gguf` and `numpy`.
- Have the NextN sidecar GGUF and the 27B Q4_K_M trunk GGUF. The trunk is a
  19 GB read with mmap. The script touches only the 40,960 rows.

## Build and run the drafter

1. Build the drafter pack.

   ```sh
   tools/mtp27/make_drafter.sh build/mtp-drafter.mtp1 [--aux1 build/mtp-drafter.aux1]
   ```

   You can also build it as part of the main build. This writes
   `$LLB_OUT/mtp-drafter.mtp1`.

   ```sh
   LLB_MTP_PACK=1 ./build.sh
   ```

2. Run the engine with the drafter.

   ```sh
   LLB_Q4_DP4A=9 LLB_SPEC=3 LLB_MTP=build/mtp-drafter.mtp1 ./build/llb-engine --gpu 40GB --threads 16 \
     -- PACK_DIR PROMPT_IDS --steps N
   ```

   Replace `PACK_DIR` with the 27B pack directory, `PROMPT_IDS` with the prompt token ids and `N` with the number of
   steps.

The output is pinned. `make_drafter.sh` compares it with the sha256 of the
production pack,
`c00130c43aa4bdcff3c8bc628f2682b93e77aa31357dbf73eb51ccb063e3c6cd`
(438,213,120 bytes). It was reproduced byte-identically in 30 s.

## Options

| Variable or flag | Default | Description |
|---|---|---|
| `MTP_SIDE` | Required | NextN sidecar GGUF. |
| `MTP_TRUNK` | Required | Trunk GGUF that holds `output.weight`. |
| `--aux1 PATH` | Not written | Also writes the pack as an AUX1 container at `PATH` (`tools/auxpack`). |
| `LLB_MTP_PACK=1` | Off | `./build.sh` also builds the drafter pack. |
| `LLB_MTP=PATH` | Unset | Engine: use the drafter pack at `PATH`. |
| `LLB_SPEC=k` | Unset | Engine: speculative decoding. The example uses `LLB_SPEC=3`. |
| `LLB_Q4_DP4A=9` | Not applicable | Engine: the 27B Q4_K_M mode-9 trunk. The configuration tested with `LLB_MTP`. |

## Sampling

With `LLB_SAMPLE=...`, the engine verifies the MTP drafts by Leviathan
rejection sampling ([`tools/sampler/README.md`](../sampler/README.md)), so
sampled output keeps the target distribution. Under greedy decoding, output
is byte-identical to the non-spec path, as before.

## Limitations

- The drafter is tested only with the 27B Q4_K_M mode-9 trunk (`LLB_Q4_DP4A=9`).
