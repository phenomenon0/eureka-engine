# Glue connector rows

`tools/glue/` feeds the rows of a trained glue connector into the fused trunk
on CUDA (G4 F1).

A trained glue connector is an input projector. An example is the trainer's
Perceiver `Resampler` in glue-9b-v4, which maps SigLIP-base 196×768 features
to 64 rows at trunk width 4096. Its rows enter the trunk as `inputs_embeds`
at ordinary text positions, with no BOS, followed by the question. The answer
is read as σ(logit 9542 " yes" − logit 874 " no") at the last prompt
position.

In this tree, the engine splices the rows in place of the embedding launch at
a run of `<|image_pad|>` (248056) ids. The rows keep the text rope. This is
the `vis-rope-seq` wire of glue-attach, ported from fus-a-wip `fc7af11e`
(FUS-V46 V5) with only the sequential-rope branch. This tree has no IMRoPE
image map yet. That is the encoder step listed under "Limitations".

## Files

| File | Description |
|---|---|
| `host/llb_host.c` `llb_img_*` | The splice: `llb_img_step` in the folded step of `llb_pass_run`. `llb_vis_request` is the per-request entry that SYS1 uses. |
| [`glue_out.c`](glue_out.c), [`glue_out.h`](glue_out.h) | `output.weight` rows (Q8_0/Q6_K/F32, the GGUF width, at most 8192; `token_embd.weight` on a tied-embedding trunk) for the served readout when no clef head is resident. |
| [`glue_rows.py`](glue_rows.py) | Run-ahead step. It verifies the pack sha, refuses a GGUF that is not the bound trunk or is below the quant floor, and runs the trainer's own `Resampler`. It writes `rows.f32` (K×4096) and `ids.txt`. |
| [`train_connector.py`](train_connector.py) | The trainer, verbatim (sha `f9aa291d…`, asserted by the pack). |
| [`readout.py`](readout.py) | CLI readout. It takes the last row of an `LLB_DUMP_HIDDEN` dump and the dequantized `output.weight`, and prints the diff and p_yes. `--full` adds the vocabulary argmax. |
| [`relayout_gguf.py`](relayout_gguf.py) | Re-lays the output of a stock `convert_hf_to_gguf.py` into per-layer-contiguous order with identical tensor bytes (the per-layer u32 of gguf2pack), and sets the bound trunk name. |
| [`served_smoke.py`](served_smoke.py) | The HTTP smoke test that the G4 gate uses. |

## Run the CLI path

1. Run the engine with the rows and dump the hidden states.

   ```sh
   LLB_GGUF=Qwen3.5-9B-Base-Q8_0.gguf LLB_VIS_EMBD=rows.f32:64:1 LLB_VIS_ROPE=seq \
   LLB_DUMP_HIDDEN=out.hid LLB_DUMP_HIDDEN_TOKENS=16384 \
     llb-engine --threads 16 -- SIDECAR_DIR $(cat ids.txt) --steps 1
   ```

   Replace `SIDECAR_DIR` with the sidecar directory of the GGUF.

2. Read out the answer.

   ```sh
   python3 tools/glue/readout.py Qwen3.5-9B-Base-Q8_0.gguf out.hid --full
   ```

Rules for the CLI path:

- `LLB_VIS_EMBD` without `LLB_VIS_ROPE` is refused. `seq` is the path of this
  connector. `imrope` is the path of the Qwen3.5 tower (IMRoPE image map,
  ),
  so the two cannot be confused.
- An image is refused with `LLB_SPEC`, `LLB_MTP`, or `LLB_PREFILL=fast`. The
  splice lives in the exact folded step.
- With both variables unset, the engine makes no device call, and the text
  path is unchanged (gated byte-identical).

## Serve rows through llb-serve

`llb-serve` passes requests to the SYS1 engine. `LLB_SYS1_NOHEAD=1` lets the
SYS1 engine run a trunk GGUF without the clef joint head, for example the
Base model. Decide requests are then refused.

The server adds two non-standard request fields. They are sessionless and
are validated before any engine work.

```text
POST /v1/completions
{"prompt": "<|image_pad|>×64 Question: Is there a car in the image? Answer:",
 "max_tokens": 1, "return_tokens": true,
 "llb_image": {"rows": "/path/rows.f32", "n": 64},
 "llb_readout": [9542, 874]}
→ … "choices":[{"text":" yes",…}], "llb":{"prompt_ids":[…],"ids":[9542],
       "readout":{"ids":[9542,874],"logits":[…],"p":[p_yes,p_no]}}
```

| Field | Description |
|---|---|
| `llb_image` | The file must be exactly n×4096 F32. With an image, the prompt is encoded with special tokens parsed, and it must hold one run of exactly n `<|image_pad|>`. Every request resets the engine's image, so nothing persists into the next request. |
| `llb_readout` | 1..8 ids. The last prompt row (final-normed; the rows of the decide hook) is dotted with the `output.weight` rows in float64. `p` is the softmax over the listed ids, so for [yes, no] it is the glue p_yes. |

The readout object is returned on non-stream responses only.

## Serve images with the live encoder

This path adds the live encoder, the cache, and trunk binding (G24 F2, F3,
and F4).

1. Bind the pack once. This writes the v1 pack.

   ```sh
   glue_rows.py bind --pack glue-9b-v4 --gguf Qwen3.5-9B-Base-Q8_0.gguf --encoder siglip-base-patch16-224 \
                     --out glue-9b-v4-bound --expect-table 58357be7…644c --cache CACHE
   ```

2. Start the server with the bound pack. Set `LLB_GLUE_CACHE=CACHE` and
   `LLB_SYS1_NOHEAD=1` in the environment.

   ```sh
   llb-serve --glue-pack glue-9b-v4-bound --glue-rows tools/glue/glue_rows.py --glue-cache CACHE \
             [--glue-python PY] --engine llb-engine -- …
   ```

3. Send a request with an image path or base64 image data.

   ```text
   POST /v1/completions {"prompt": "<|image_pad|>×64 Question: …", "llb_image": {"image": "/srv/path.jpg"}, …}
                                                      or  {"image_b64": "<base64>"}
   → … "llb":{…, "readout":{…}, "glue":{"image_sha256","rows_sha256","feats":"hit|miss","rows":"hit|miss","ms",…}}
   ```

### Live encoder (F2)

- `llb-serve` starts one resident `glue_rows.py worker` next to the engine.
  The worker loads SigLIP and the connector once.
- For each request with `llb_image.image` or `image_b64`, the server sends
  the worker one JSON line, outside the engine lock. The worker returns
  stamped rows. The server rewrites the request to `{"rows","n"}` for the
  engine.
- The encoder step is the one in `prep_coco.py`: `AutoImageProcessor` (slow,
  the closer match to the trainer's features), `SiglipVisionModel` in bf16,
  and `last_hidden_state` to f16.
- `glue_rows.py rows --image IMG` runs the same step on the CLI.

### Cached-features mode (F3)

- `rows --feats FEATS.npy` still takes the trainer's cached features.
- `--cache DIR` keeps two maps:
  - sha(image) → `feats/<img>-<encoder>.npy`
  - → `rows/<img>-<connector16>-<table16>/rows.{f32,json}`
- Rows do not depend on the question, so one image serves every question.
- Writes use atomic renames. The cache is LRU-trimmed to `--cache-max-mb`
  (default 2048).

### Trunk binding (F4)

- The v1 pack carries `trunk.tensor_table_sha256`: the sha over the bytes of
  every tensor, as `relayout_gguf.py` defines it. A GGUF with other content is
  refused, whatever its label says. The pack also carries the file shas of the
  encoder.
- Every stamp (`rows.json`) carries `trunk_tensor_table_sha256`,
  `connector_sha256`, and `rows_sha256`.
- `dual.c` refuses served rows in these cases:
  - The stamp is missing. Rows from before F2 are refused.
  - The stamp is for another trunk. It is compared with the table sha of the
    loaded GGUF.
  - `rows_sha256` or K differs from the request.
- The CLI `LLB_VIS_EMBD` path is unchanged. It is for development and gates.

### Table-sha cache

Hashing reads the whole GGUF: 4–6 min for the 9B on the FUSE store. So
`glue_rows.py` and `glue_out.c` share records under
`CACHE/tabsha/<dev>-<ino>-<size>-<mtime_ns>`.

- A hit re-hashes the header (KV and tensor infos), so a relabel that keeps
  size and mtime still misses.
- A content edit that keeps size, mtime, and header is not detected without a
  cold re-hash (`rm CACHE/tabsha/*`).

## Options

| Option | Default | Description |
|---|---|---|
| `LLB_VIS_EMBD` | Unset | CLI: the rows to splice, as in the example `rows.f32:64:1`. Requires `LLB_VIS_ROPE`. |
| `LLB_VIS_ROPE` | Unset | `seq` for this connector, `imrope` for the Qwen3.5 tower. |
| `LLB_DUMP_HIDDEN`, `LLB_DUMP_HIDDEN_TOKENS` | Unset | Dump hidden states for `readout.py`. |
| `LLB_SYS1_NOHEAD=1` | Off | Run a trunk GGUF without the clef joint head. Decide requests are refused. |
| `LLB_GLUE_CACHE` | Unset | Cache directory for the served path. |
| `--glue-pack`, `--glue-rows`, `--glue-cache`, `--glue-python` | Unset | `llb-serve` flags for the live encoder path. |
| `--cache-max-mb` | 2048 | LRU limit of the cache. |

## Limitations

- There is no IMRoPE image map in this tree yet.
- There is no in-engine SigLIP. Python stays on the served path.
- The connector rows (`llb_image`) work only at width 4096. The readout reads the
  trunk width from the GGUF.
- The readout is not returned on streams.
- An image is refused with `LLB_SPEC`, `LLB_MTP`, or `LLB_PREFILL=fast`.

