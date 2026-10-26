# Speech recognition (ear)

`tools/ears/` turns speech into text with the engine's own audio tower
(EARS-CANON). The served path is `llb-serve --ear DIR`
([docs/reference/multimodal-content.md](../../docs/reference/multimodal-content.md) §8).

## Files

| File | Description |
|---|---|
| `aud_mel.cpp`, `aud_mel.h`, `mel_tables_glibc.h` | Host log-mel. It is the qwen3a preprocessor of llama.cpp 2ca15f5, op for op. The n_fft=400 sin/cos tables are pinned to glibc's values, so one source is byte-equal on Linux and on Apple libm. |
| `aut.cu`, `aut.h` | The Qwen3 AuT audio tower (qwen3a bf16 mmproj) on CUDA, sm_86, `-fmad=false`. This is the ears-fix tower. Attention staging is Q F32, K/V F16, P F32. The Metal twin with the same `aut.h` (`aut_metal.mm`) is in the iOS lane (EARS.xcframework). |
| `ears.cpp`, `ears.h` | The request: mel, then the tower, then the Qwen3-ASR-0.6B decoder (llama.cpp 2ca15f5), fed the rows at the audio chunks, then greedy ids. There are two forms: `ears_transcribe`, the one-shot contract (IOS-EARS), and `ears_open` / `ears_run`, the resident form. |
| `ear_worker.cpp` | `llb-ear`: the JSON-line child of `llb-serve --ear`. The protocol is in the file header. Also provides `llb-ear mel WAV GOLDEN_DIR`. |
| `aut_encode_batch`, `LLB_EAR_BATCH=1\|seg\|streams`, `llb-ear encb` | Encoder batch (CHAIN-2-1-3): the mel windows of one request in one tower pass, with rows byte-identical per window. See the Encoder batch section of [docs/reference/http-api.md](../../docs/reference/http-api.md). Off by default. |
| `build.sh` | `OUT=… LC=<llama.cpp 2ca15f5> LCB=<its CUDA build> tools/ears/build.sh` writes `$OUT/llb-ear`. |

## How it works

### Chunker

`ears.cpp` is the slim qwen3a chunker. It does not link mtmd. It rebuilds the
mtmd chunk layout from the decoder's vocabulary and the tower's per-window
token counts:

- `<|audio_start|>`, then one chunk of `n/100·13` rows per 8 s window, then
  `<|audio_end|>`.
- Sequential positions, causal.

As a result, only the tower opens the mmproj, once.

### Resident form

The tower and the decoder are loaded once. Each request runs on a fresh llama
context, so its ids and logits are the same as those of `ears_transcribe`.

## Before you begin

- Get llama.cpp at commit 2ca15f5 and its CUDA build.
- Get the models `Qwen3-ASR-0.6B-Q8_0.gguf` (`bca25981…`) and
  `mmproj-Qwen3-ASR-0.6B-bf16.gguf` (`dae36c85…`), and put them in one
  directory.
- Prepare audio as 16 kHz mono WAV, PCM16 or float32. The ear does not
  resample.

## Build and serve

1. Build `llb-ear`.

   ```sh
   OUT=/tmp/llb LC=LLAMA_CPP_DIR LCB=LLAMA_CPP_CUDA_BUILD tools/ears/build.sh
   ```

   Replace `LLAMA_CPP_DIR` with a llama.cpp checkout at 2ca15f5 and
   `LLAMA_CPP_CUDA_BUILD` with the CUDA build directory of llama.cpp
   2ca15f5.

2. Build `llb-serve`. `llb-serve` finds `llb-ear` in its own directory, or at
   `LLB_EAR_BIN`.

   ```sh
   OUT=/tmp/llb … tools/sys1dual/build-serve.sh
   ```

3. Start the server with the ear.

   ```sh
   /tmp/llb/llb-serve --ear ~/.local/share/models/ears --engine /tmp/llb/llb-engine -- …
   ```

4. Send a transcription request.

   ```sh
   curl -s localhost:8080/v1/audio/transcriptions -H 'Content-Type: application/json' \
     -d "{\"input_audio\":{\"data\":\"$(base64 -w0 clip.wav)\",\"format\":\"wav\"}}"
   ```

## Options

| Option | Default | Description |
|---|---|---|
| `llb-serve --ear DIR` | Off | Serves the ear. `DIR` holds the two model files. |
| `--ear-llm-cpu` | Off | Runs the decoder on the CPU, which is the backend of the goldens. The tower stays on the GPU. |
| `LLB_EAR_BIN` | `llb-ear` next to `llb-serve` | Path to `llb-ear`. |
| `LLB_EAR_BATCH=1\|seg\|streams` | Off | Encoder batch mode. |
| `OUT`, `LC`, `LCB` | None | `build.sh` inputs: output directory, llama.cpp 2ca15f5 source, and its CUDA build. |

## Gates

These results are from the canon preparation gates (EARS-CANON-PREP):

- Transcripts are id- and text-identical to the goldens (EARS-G0), 5/5, on
  the served HTTP path, with both the GPU and the CPU decoder.
- Mel is byte-equal, 11/11 windows.
- Repeat runs, including across a restart, are bit-identical: ids, logits,
  and tower rows.
- The mmproj is opened once, for load plus 5 requests. No mtmd is linked.

## Limitations

- Per-tensor equality is not claimed. It is pending the owner's signature on
  the tolerance proposal (`ears-bittune/TOLERANCE-PROPOSAL.md`).
- The ear accepts only 16 kHz mono WAV and does not resample.
- The CUDA tower targets sm_86.
