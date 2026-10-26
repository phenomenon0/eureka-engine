# Multimodal content

This page describes image and audio content in `POST /v1/chat/completions`
(G5). It extends the [HTTP API](http-api.md) wire format (W1-SERVER). The parsing,
validation, and template rendering live in `tools/sys1dual/mm.c`. `serve.c` wires
them into the chat path. The server still contains no engine code. It hands
validated media to the engine through the media hook ([§6](#6-engine-media-hook)),
and the engine encodes it.

## Status

| Part | Status |
|---|---|
| Server side | Done and tested. |
| Engine side | Specified ([§6](#6-engine-media-hook)). Implemented for the stub engine only. The SYS1 engine that `llb-serve` drives today has no vision path, so `--mm` defaults to `none`. With `none`, image and audio requests are fully validated and then refused with a 400. |
| Fused-trunk encode path | A reference implementation exists (`mm_encode_glue.py`, [§7.1](#71-fused-trunk-siglip-to-glue-connector-to-rows-at-text-positions)). |
| Audio | The ear is implemented ([§8](#8-audio-transcription), EARS-CANON). `--ear DIR` runs `llb-ear`, which holds the CUDA AuT tower and the Qwen3-ASR-0.6B decoder. It serves `POST /v1/audio/transcriptions`. In chat, it turns an `input_audio` part into the user turn's text. |

## 1. Content parts

`messages[i].content` can be a string, `null`, or an array of parts. The server
handles each part in the order that `flash-chat_template.jinja` handles it: image
keys first, then video, then text.

| Part | Shape | Rendered as |
|---|---|---|
| Text | `{"type":"text","text":STR}` | `STR`, verbatim. |
| Image (OpenAI) | `{"type":"image_url","image_url":{"url":URL,"detail":"auto\|low\|high"}}` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
| Image (string form) | `{"type":"image_url","image_url":URL}` | Same. |
| Image (HF / llama.cpp form) | `{"type":"image","image":URL}` | Same. |
| Audio (OpenAI) | `{"type":"input_audio","input_audio":{"data":BASE64,"format":"wav\|mp3\|flac\|ogg"}}` | `<\|audio_start\|><\|audio_pad\|><\|audio_end\|>`. With `--ear`, the ear's transcript ([§8](#8-audio-transcription)). |
| Video | `video` / `video_url` | 400 `video content is not supported`. |
| Any other type | | 400 `Unexpected item type in content.` (the template's own message). |

Placement rules:

- The placeholder sits exactly where the part sits. Text before and after it is
  kept.
- The whole message is then trimmed, as the template does.
- System messages cannot carry media. They get a 400 with
  `System message cannot contain images.`, as the template raises.
- `detail` is passed to the engine. It defaults to `auto`.

Parity: the C renderer was checked against the model's own template, compiled by
the transformers chat-template code, on 17 cases. 17/17 are byte-identical, and
both template-raise cases map to 400. The cases include text-only, 1 image, 2
images, adjacent images, multi-turn with images, thinking off,
`add_generation_prompt:false`, assistant parts with `reasoning_content`,
whitespace trim, and unicode. Receipt: `parity.json`.

## 2. Image sources

| URL form | Handling |
|---|---|
| `data:image/<type>;base64,<b64>` | Decoded in the server. The base64 must use the RFC 4648 alphabet. `=` padding is optional, and ASCII whitespace is skipped. The URL-safe alphabet, data after the padding, and non-canonical trailing bits all give 400. |
| `data:` without `;base64` | 400. |
| `http://` and `https://` | Off by default. The server returns 400 and asks for a data: URI. With `--mm-remote on`, the server fetches the URL itself ([§2.1](#21-remote-fetch)). |
| `file:`, bare paths, other schemes | 400. |

### 2.1 Remote fetch

This section describes `--mm-remote on|private`.

1. The server resolves the host once.
   - With `on`, a target in a loopback, private, CGNAT, link-local, multicast, or
     unspecified range (v4, v6, or v4-mapped v6) is refused with 400.
   - `private` allows those ranges, for a LAN image store.
2. The server pins `curl` to that address with `--resolve`, so a DNS rebind
   between the check and the connect cannot redirect it.

Fetch limits:

| Limit | Value |
|---|---|
| Protocols | http and https only. |
| Redirects | Not followed. A 3xx gives 400 `redirects are not followed`. |
| Time | `--max-time 10`. |
| Size | The per-image cap is enforced by `--max-filesize`, and again on the bytes actually read. |
| Credentials | URLs with `user:pw@` are refused. |
| URL length | URLs over 2048 bytes are refused. |

Fetches run after the whole request has parsed, and outside the engine lock. A
fetched image goes through the same sniffing and caps as an inline one.

## 3. Content-type handling

The bytes decide the type.

Images:

- The magic bytes decide the type: PNG, JPEG (baseline or progressive), WebP
  (VP8, VP8L, or VP8X), or GIF.
- A data: URI must declare `image/*`, otherwise 415.
- A wrong subtype is accepted, and the sniffed type wins. For example,
  `image/jpeg` holding a PNG is accepted as `image/png`. The declared type is kept
  for logs.
- Bytes that are not one of the four formats give 415.
- A recognized format with no readable dimensions in its header, such as a PNG
  with no IHDR, gives 400 `corrupt`.

Audio:

- `format` must be wav, mp3, flac, or ogg, otherwise 415.
- `format` must also match the sniffed bytes. A WAV sent with `format:"mp3"` gives
  415.

## 4. Size caps

These are the defaults. The flags are listed in [HTTP API](http-api.md#flags).

| Cap | Default | Over the cap |
|---|---|---|
| Request body | 8 MiB (`--max-body-mb`) | 413, before the body is read. |
| One image, decoded | 6 MiB (`--mm-max-image-mb`) | 413. The base64 length is checked before decoding, so nothing is allocated. |
| One audio clip, decoded | 6 MiB | 413 |
| All media in one request | 24 MiB. Only reachable with URL images, because inline images are bounded by the body cap. | 413 |
| Images per request | 8 (`--mm-max-images`) | 400 |
| Audio clips per request | 1 | 400 |
| Image side (from the header) | 8192 px | 413. This guards against decompression bombs, and is checked before any decoder runs. |
| Image pixels w×h | 40 MP | 413 |

## 5. Errors

All errors use the OpenAI shape described in [HTTP API](http-api.md#errors).
Media errors also set `param` to the offending part. For example:

```json
{"error":{"message":"image is 9000x8; the limit is 8192 per side and 40000000 pixels",
          "type":"invalid_request_error","param":"messages[0].content[0]","code":413}}
```

The server runs its checks in this order:

1. The template rules (role and part shapes).
2. Decode, sniff, and caps.
3. The engine gate. If `--mm` does not include the kind, the server returns 400
   `this model has no vision path…` or `audio input needs the ear…`.
4. The remote fetch.
5. Staging.

A bad image is therefore reported as bad even on a text-only engine.

Text-only traffic is unchanged. On 14 text requests, the W1-SERVER binary and the
G5 binary were compared over the same stub engine:

- The engine request lines are byte-identical (8/8). Text-only requests carry no
  `media` key.
- 13/14 HTTP results (status + body) are identical. The one intended difference
  is the error for `content: 5`. It is now the template's
  `Unexpected content type.`, with a `param`.

Receipt: `text_identity.json`.

## 6. Engine media hook

This section is the engine's half of the contract. When media is present, the
generate line gains a `media` array, with one entry per placeholder in prompt
order:

```json
{"mode":"generate","prompt":"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Is there a car?<|im_end|>\n…",
 "n":64,"special":true,
 "media":[{"kind":"image","path":"$TMPDIR/llb-serve-mm/<sha256>-<seq>.jpeg","mime":"image/jpeg",
           "bytes":191947,"sha256":"db7173813c23…","w":640,"h":480,"detail":"auto"}]}
```

The server guarantees the following:

- `prompt` contains exactly one `<|vision_start|><|image_pad|><|vision_end|>` per
  image entry, and one audio tag per audio entry, in the same order.
- `path` is a file that the server wrote, with mode 0600, under the server's
  `TMPDIR`. Its content has exactly `bytes` bytes and the given `sha256`.
- The file exists until the engine's final line for this request. The server
  unlinks it after that line: on success, on error, and when the client has gone.
- Requests without media produce the old line, byte for byte.

The engine must do the following:

1. Refuse the entry if the file's sha256 does not match. The refusal is
   `{"error":...}`, which the server forwards as 400.
2. Encode each entry with the encoder bound to its trunk ([§7](#7-encode-paths)).
   The result is `n_i` rows of width `d` = the trunk hidden size.
3. Expand the i-th `<|image_pad|>` into `n_i` pads. `mm_expand_pads` is the
   reference rule. It refuses on a count mismatch or zero rows. Inject row `j` at
   the `j`-th pad position. The encoder's `layout` ([§7](#7-encode-paths)) says
   whether the `<|vision_start|>` and `<|vision_end|>` tokens stay or go.
4. Report `prompt_ids` including the expanded pads, so `usage.prompt_tokens`
   counts the image rows.
5. Reset the image state per request. Today, `LLB_VIS_EMBD` is read once per
   process (`llb_img_state`), so this is a per-request load from the media entry.
6. Refuse a kind that it does not take, instead of decoding the bare tag as text.
   This is defense in depth, because the server's `--mm` gate already refuses it.

The stub engine ([`tests/mm/stub_engine.py`](../../tests/mm/stub_engine.py))
implements steps 1, 3, and 4 against the real `llb-serve`. HTTP receipt:
`http_e2e.json`.

## 7. Encode paths

Two encode paths fit the same hook. They differ only in the encoder, `rope`, and
`layout`. The engine takes these from the encoder bound to the trunk it serves.
The server is unchanged.

### 7.1 Fused trunk: SigLIP to glue connector to rows at text positions

This is the G4 wire.

| Component | Description |
|---|---|
| Encoder | SigLIP-base-patch16-224 `last_hidden_state` (196×768). Preprocessing is `AutoImageProcessor` on the RGB image. The training features were bf16 and stored fp16. |
| Connector | `glue-9b-v4`, the trainer's own Resampler (`connector.safetensors d8718b43…`). It produces K = 64 rows × d 4096. The rows are cast to bf16 and stored as F32. |
| Engine | `LLB_VIS_EMBD=rows.f32:64:1 LLB_VIS_ROPE=seq` (the vis-rope-seq patch, fus-a-wip `fc7af11e` + glue-attach). The rows take ordinary sequential text positions. |
| Reference run-ahead | [`tools/sys1dual/mm_encode_glue.py`](../../tools/sys1dual/mm_encode_glue.py). It takes the media entry's path and sha256, and writes `rows.f32` and `rows.json` (`n, d, rope:"seq", layout:"replace-tag"`, the env line, and shas). |

**Caution:** Read this layout note before you serve this path.

- glue-9b-v4 was trained on `[64 rows | " Question: {q} Answer:"]`, on
  Qwen3.5-9B-Base, with no chat template and no `<|vision_start|>` or
  `<|vision_end|>`.
- Its layout is therefore `replace-tag`: the rows replace the whole tag.
- Served under the chat template, the rows sit inside ChatML on a base trunk.
  That is off the training distribution.
- v4 can therefore be served faithfully only as the probe that it is. A
  chat-served fused path needs a connector trained on the chat layout, or a
  server-side "raw layout" mode for the binding. This is recorded as an open item
  for G4 (F2–F4).

Encode receipt (CPU): the COCO image (val2017 142472) was sent through
`llb-serve` as base64. The bytes that the server staged have sha `db717381…`
(= the file). Encoding them live gave these results:

| Live encode vs glue-attach2's rows (cached training features; engine-scored at \|Δp\| 0.0005) | Feats cos mean / min | Rows cos mean / min | Rows rel‑Frobenius |
|---|---|---|---|
| fp32 CPU | 0.99889 / 0.958 | 0.99979 / 0.9964 | 2.0% |
| bf16 CPU | 0.99808 / 0.886 | 0.99945 / 0.982 | 3.3% |

The live encoder reproduces the training features up to the reference's own
bf16-on-ROCm noise.

Engine receipt (GPU, under both house locks): the live rows were run through
glue-attach2's patched engine (`42dfb9aa`) on the 9B-Base Q8_0, with the same IDs
and one step, plus a control run on the cached rows.

| Rows | p_yes | \|Δ\| vs cached rows | \|Δ\| vs offline trainer 0.99683 | Greedy |
|---|---|---|---|---|
| Cached (control) | 0.99732 | 0 | 0.00049 | 9542 " yes" |
| Live fp32 | 0.99706 | 0.00026 | 0.00023 | 9542 " yes" |
| Live bf16 | 0.99675 | 0.00057 | 0.00008 | 9542 " yes" |

- The control's hidden dump is byte-identical to glue-attach2's `s-row0`, so the
  engine is deterministic across days.
- Every live run is well inside the registered ≤ 0.033 bar and gives the same
  answer.
- The chain from base64 request to server staging, live encoder, connector, and
  engine is closed for the smoke image.

### 7.2 Two-model look path: native vision tower to rows on the image grid

| Component | Description |
|---|---|
| Encoder | The model's own Qwen3.5 vision tower (the "look" tower; the cudavis goldens are the parity discipline). It runs ahead and emits `nx × ny` merged-patch rows. |
| Engine | `LLB_VIS_EMBD=rows.f32:nx:ny LLB_VIS_ROPE=imrope` (IMRoPE grid positions; CLI, exact pass). Without `LLB_VIS_ROPE`, the engine refuses, so the tower path and the connector path (`seq`) cannot be confused. |
| Layout `keep-tag` | `<\|vision_start\|>` and `<\|vision_end\|>` stay, and `<\|image_pad\|>` expands to `nx·ny`. This is exactly what the HF processor does after the template, so the server's rendering is already correct for this path. |
| Pad count | Comes from the processor's smart resize: patch 16, merge 2, within min/max pixels. `detail` can choose the pixel budget (`low` = the min budget). That choice belongs to the engine. |
| K images (MIMG) | `LLB_VIS_EMBD` takes a `;`-separated list. dual.c takes `"images":[{rows,nx,ny},...]` (tower rows, stamped `llb-tower-rows-v1`). `llb-serve --vis-tower CLI --vis-mmproj MMPROJ` wires chat images to it. See [Multi-image requests](http-api.md#multi-image-requests). The real engine does not read the `media` hook. With `--vis-tower`, the image items leave `media` and travel as `images`. |

## 8. Audio transcription

This section describes the ear (`--ear DIR`, EARS-CANON). The ear is a resident
child of `llb-serve`, like the `--tts` worker: `llb-ear worker`. Its source is in
`tools/ears/`. See [the ears README](../../tools/ears/README.md).

The ear holds the following components:

- Host log-mel (`aud_mel.cpp`): llama.cpp 2ca15f5's qwen3a preprocessor, op for
  op, gated byte-equal.
- The Qwen3 AuT audio tower on CUDA (`aut.cu`, the qwen3a bf16 mmproj).
- The Qwen3-ASR-0.6B Q8_0 decoder, through llama.cpp 2ca15f5 (the goldens'
  commit).

The tower and the decoder are each loaded once, at start. The slim qwen3a chunker
builds the decoder prompt. mtmd and clip are not linked, so only the tower opens
the mmproj, once. On the Mac and on iOS, the same `ears.cpp` runs over the Metal
tower (`aut_metal.mm`, EARS.xcframework).

The audio tag and its row expansion: `MM_AUDIO_TAG` is
`<|audio_start|><|audio_pad|><|audio_end|>`, the Qwen3-ASR template's audio part.
It lives in `mm.h`.

- Inside the ear, `<|audio_pad|>` expands to the tower's rows: one chunk per 8 s
  mel window, `n/100·13` rows each (104 for a full window).
- The rows take sequential positions, `n_past + i`. The decoder's rope is NEOX, so
  there is no M-RoPE map. This is the audio analog of the vision IMRoPE map in
  [§7.2](#72-two-model-look-path-native-vision-tower-to-rows-on-the-image-grid).
- The chunk layout is mtmd's for qwen3a. It was checked identical on the 5 EARS
  clips (ios-ears2 `chunkdiff`).

Qwen3.5 has no audio tokens, so the audio rows never reach the trunk. The ear
turns the audio into text, and the trunk gets that text.

### 8.1 `POST /v1/audio/transcriptions`

The endpoint is served with `--ear`. Otherwise it returns 404.

```json
{"input_audio": {"data": "<base64 WAV>", "format": "wav"},
 "response_format": "json",           // json (default) | text | verbose_json
 "max_tokens": 1024}                  // optional cap on decoder steps (default 1024, max 2048)
```

| Field | Description |
|---|---|
| `input_audio` | The chat part's inner object, with the same checks: format, sniff, 6 MiB cap, sha256. |
| `response_format` | `json` (default), `text`, or `verbose_json`. |
| `max_tokens` | Optional cap on decoder steps. Default 1024, maximum 2048. |

Input rules:

- The ear takes 16 kHz mono WAV, PCM16 or float32.
- A valid mp3, flac, or ogg passes the server's checks, but the ear refuses it
  with 400 `the ear takes 16 kHz mono WAV…`. Convert the audio first. The goldens'
  mel is defined on 16 kHz PCM, and the ear does not resample.
- Multipart (OpenAI's `file=` upload) is not accepted. Bodies are JSON, as
  everywhere in `llb-serve`.

Replies:

| `response_format` | Reply |
|---|---|
| `json` | `{"text":"Nor is Mister Quilter's manner less interesting than matter."}` |
| `text` | The transcript plus `\n`, as `text/plain`. |
| `verbose_json` | `{"text","task":"transcribe","language","duration","llb":{…}}`. `llb` is the ear's receipt (see the next table). |

| `llb` field | Content |
|---|---|
| `raw` | The decoder's full answer, for example `language English<asr_text>…`, rendered as the goldens' `transcript.txt`. |
| `ids` | The greedy IDs, EOG step included. |
| `top1` / `top2` | The logits, as `%a` hex strings. |
| `rows_sha256` | The sha256 of the tower's rows. |
| `n_windows`, `n_audio_tokens`, `audio_s` | Window count, audio token count, and audio length. |
| `timings` | `mel_ms`, `tower_ms`, `tower_gpu_ms`, `ctx_ms`, `prefill_ms`, `decode_ms`, `first_token_ms`, `total_ms`. |

Every 200 response carries the headers `X-LLB-Ear-Ms`, `X-LLB-Audio-Seconds`,
and `X-LLB-Audio-Sha256`.

Errors use the usual shape:

| Status | When |
|---|---|
| 415 | A format/bytes mismatch, with `param` `input_audio`. |
| 400 | A bad clip, or an ear refusal. |
| 503 | `Loading ear` while the ear loads, or `the ear is not running`. |

### 8.2 Chat: the audio is the user turn

With `--ear`, the server handles a chat `input_audio` part as follows:

1. It validates and stages the part like any media item ([§3](#3-content-type-handling)–[§6](#6-engine-media-hook)).
2. It transcribes the part outside the engine lock. The ear checks the staged
   file's sha256 first ([§6](#6-engine-media-hook) step 1).
3. It replaces that part's tag in the rendered prompt with the transcript
   (`mm_replace_audio`).
4. It removes the audio item from the media list.

So `[{"type":"input_audio",…},{"type":"text","text":" summarize"}]` reaches the
engine as the text request `"<transcript> summarize"`. The engine line is
byte-identical to that text request's line, as checked by `http_e2e` config D.
The trunk is unchanged: no `media` key and no audio rows.

Other rules:

- An ear refusal gives 400 `ear: …`, and the server removes the staged file.
- Batch chat items take the same path.
- `--ear` implies the audio half of `--mm`.
- Without `--ear`, `--mm audio` still hands audio to the engine's media hook
  ([§6](#6-engine-media-hook)), unchanged.

Text traffic is unchanged with the ear attached.
`tests/mm/text_identity.py --new-args "--ear DIR"` compares canonical against the
candidate with `--ear` on. The engine lines are identical (8/8), and the HTTP
results are identical (14/14).

Gates (EARS-CANON-PREP.md): transcripts are ID-identical 5/5 to the EARS-G0
goldens on the served path, mel is byte-equal 11/11, determinism holds, and the
mmproj is loaded once. The per-tensor rule is pending the owner's signature
(`ears-bittune/TOLERANCE-PROPOSAL.md`). It is not claimed.

## 9. Not covered

This section lists the limitations of multimodal content.

- Tools with images. Tools still return 400 (W1-SERVER). The combined tests wait
  for the tools work. The harness has a placeholder check that asserts the 400.
- Video.
- Image decoding in the server. By design, the server only reads headers.
- The `detail` to pixel-budget policy. That belongs to the engine.
- Caching encoded rows by sha256. The sha is already in the hook (G4 F3).

## 10. Tests

| Test | What it covers | Result |
|---|---|---|
| [`tools/sys1dual/mm_test.c`](../../tools/sys1dual/mm_test.c) (ASan + UBSan) | base64 (RFC 4648 vectors and strictness), SHA-256 (FIPS vectors, 1 MB), sniff + dims on real encoder output (png/jpeg/progressive/gif/webp×3), render, content-type, caps, malformed parts, system rule, audio, engine gate, SSRF refusals, pad expansion, audio replacement + the transcription audio object | 94/94 |
| [`tests/mm/parity.py`](../../tests/mm/parity.py) + [`render_driver.c`](../../tests/mm/render_driver.c) | serve.c's chat renderer vs the jinja template | 17/17 byte-identical |
| [`tests/mm/http_e2e.py`](../../tests/mm/http_e2e.py) + [`stub_engine.py`](../../tests/mm/stub_engine.py) (+ [`stub_ear.py`](../../tests/mm/stub_ear.py)) | The real `llb-serve` binary in 5 configs (`--mm none`, `image`+remote private, remote on, `image,audio`, `--ear`): 1 image, 2 images, stream, URL fetch, redirect, 404, every cap and status, staging cleanup, tool+image (400). With `--ear`: transcription formats and errors, chat audio = the text request with its transcript (engine line byte-identical), ear refusal cleanup, batch. | 42/42 |
| [`tests/mm/text_identity.py`](../../tests/mm/text_identity.py) | W1-SERVER vs G5 binary, text-only | Engine lines 8/8 identical; HTTP 13/14 (1 intended) |
| `tests/mm/text_identity.py --new-args "--ear DIR"` | Canonical (f4eb2375) vs EARS-CANON with the ear attached, text-only | Engine lines 8/8 identical; HTTP 14/14 |
| [`tests/mm/encode_parity.py`](../../tests/mm/encode_parity.py) + `mm_encode_glue.py` | The server-staged bytes to live SigLIP + connector vs the engine-proven rows | See [§7.1](#71-fused-trunk-siglip-to-glue-connector-to-rows-at-text-positions) |
