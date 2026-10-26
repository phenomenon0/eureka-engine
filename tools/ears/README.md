# Speech recognition (ear)

`tools/ears/` turns speech into text with the engine's own audio tower
(EARS-CANON). The served path is `llb-serve --ear DIR`
([docs/reference/multimodal-content.md](../../docs/reference/multimodal-content.md) §8).

## Files

| File | Description |
|---|---|
| `aud_mel.cpp`, `aud_mel.h`, `mel_tables_glibc.h` | Host log-mel. It is the qwen3a preprocessor of llama.cpp 2ca15f5, op for op. The n_fft=400 sin/cos tables are pinned to glibc's values, so one source is byte-equal on Linux and on Apple libm. |
| `aut.cu`, `aut.h` | The Qwen3 AuT audio tower (qwen3a bf16 mmproj) on CUDA, sm_86, `-fmad=false`. This is the ears-fix tower. Attention staging is Q F32, K/V F16, P F32. |
| `aut_metal.mm` | The Metal twin of `aut.cu`, same `aut.h` API, for macOS and iOS (EARS.xcframework). See [Metal tower](#metal-tower). |
| `build-metal.sh` | `OUT=… LC=<llama.cpp 2ca15f5> tools/ears/build-metal.sh` compiles `aut_metal.mm` and `aud_mel.cpp` for the macOS, iOS and simulator slices with the EARS.xcframework flags. |
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

## Metal tower

`aut_metal.mm` is the Qwen3 AuT tower on Metal: a port of `aut.cu`, op for
op, with the same `aut.h` API and graph (llama.cpp 2ca15f5 `qwen3a.cpp` plus
`build_vit`). Staging is the CPU golden's: conv operands F16, block and
projector GEMMs BF16 x BF16 with F32 accumulate, attention Q F32, K/V F16,
P F32. There is one command buffer per window, all buffers are shared
(unified memory), and the MSL is compiled at load with fast math off. Metal
has no `erf`, so GELU-erf uses a positive-term series (about 1-2 ulp).

It implements `aut_load`, `aut_free`, `aut_proj_dim`, `aut_n_tokens`,
`aut_encode`, `aut_gpu_ms` and `aut_weight_bytes`. It does not implement the
CUDA-only `aut_encode_batch` and `aut_mem_info`, so it links with the
slim-chunker `ears.cpp` of the iOS lane, not with this directory's
`ears.cpp`, which calls them.

### Source

From the EARS-BITTUNE lane, commit `ce9c338` (branch `bittune-diag`),
`src/aut_metal.mm` (28,985 bytes, SHA-256 `08e07e4c…`), unmodified. It was
cross-checked with the copy in the EARS-CARRY lane's `work/src/` (27,012
bytes, `8bfd150d…`), which is the one the shipped EARS.xcframework was built
from. The only differences are compiled-out diagnostics:

- `#ifdef AUT_DIAG`: teacher forcing (`AUT_TF_DIR`, `AUT_TF_CALLS_PER_WIN`)
  and an MSL prefix (`AUT_MSL_PREFIX`).
- `GEMM_SCALAR` and `GEMM_KAHAN` GEMM bodies, selectable only through that
  MSL prefix.

Without `-DAUT_DIAG`, the kernels and the host code are the carry copy's.
The lane checked this: the production tower is byte-identical to the
registered one on 324/324 tensor dumps and on the tower rows of all 5 clips.
The BITTUNE copy was taken because it is the newer commit, it carries the
instruments that produced the tolerance numbers below, and it costs nothing
in a production build.

`aut.h`, `aud_mel.*` and `mel_tables_glibc.h` are shared with the CUDA
tower. This directory's `aut.h` is a superset of the lane's (it adds the
two CUDA-only declarations), so `aut_metal.mm` compiles against it
unchanged. `aud_mel.cpp`, `aud_mel.h` and `mel_tables_glibc.h` are
byte-identical to the lane's.

### Build into EARS.xcframework

`backends/ios/Vendor/ears-src/build-ears-v3.sh` builds EARS.xcframework v3:

1. For each slice (iOS `arm64-apple-ios18.0`, simulator, and the macOS
   harness slice `arm64-apple-macos14.0`), it compiles `aud_mel.cpp`
   (`-O3 -ffp-contract=off`), `aut_metal.mm` (`-O2 -fobjc-arc`), the
   slim-chunker `ears.cpp`, and `ears_text.cpp`.
2. It prelinks them with llama.cpp 2ca15f5's static libraries (`libllama`,
   `libggml`, `libggml-cpu`, `libggml-metal`, `libggml-base`) into one object
   with `ld -r`, and exports only `ears_*`. mtmd is not linked.
3. It wraps the iOS and simulator `libears.a` in `EARS.xcframework` and
   writes `macos/libears.a`.

That script reads the tower from the EARS-CARRY tree (`EARS_CARRY`). To build
from this directory instead, point it at a tree whose `src/aut_metal.mm` is
this file. Without `AUT_DIAG`, both copies compile to the same kernels.
`build-metal.sh` compiles just the tower and mel objects with the same flags.

### Tolerance against CUDA and the CPU golden

The Metal tower is not bitwise equal to the CPU golden or to the CUDA tower,
and no fix is possible. The EARS-BITTUNE lane report and its tolerance
proposal found the following:

- Every tensor differs from the CPU golden by more than 1e-6, starting at the
  first conv (302/302). The cause is fp32 reduction order. The error is about
  1e-3 per layer and is amplified onto channel 770, the model's
  massive-activation channel (|x| up to 93), to at most 0.31 at
  layer_out-12..17. The embedding differs by at most 2.2e-3.
- There is no defective op. Teacher-forced from the golden's input, the
  Metal error per layer matches an fp64 oracle's (L14 ch770 rms 5.0e-4 vs
  4.3e-4, with no signed bias). An exact fp64 tower misses the golden by
  0.21-0.26 at L14-16. The same llama.cpp source on CUDA, ARM CPU or Metal
  misses it by 0.14-1.05.
- On the registered gate (2x the CPU-CUDA spread), Metal fails 10/302
  tensors, and the CUDA port `aut.cu` (EARS-G1) fails 23/302. The strict
  per-tensor gate (1e-6 abs / 1e-5 rel) fails for every implementation,
  fp64 included.
- Scale-relative (max|d| / max|ref|), Metal is closer to the reference than
  every accepted reference sample. For example, at layer_out-12..17 Metal is
  6.2e-3, against 5.4e-2 for llama.cpp CUDA. For embd it is 1.3e-2 against
  τ 3.0e-2.
- The output gates pass. Transcripts are id-identical 5/5 with the Metal LLM
  and 5/5 with the CPU LLM, mel is byte-equal 11/11, repeat runs are
  byte-identical 10/10 (ids, `%a` logits and tower rows), the mmproj is
  opened once per request, and peak footprint is 874-883 MB.

The proposed rule, P-SR, passes a tensor when
`max|ours - CPU| <= τ_fam · max|CPU|`. Here τ_fam is taken only from the
accepted reference samples. It is 9.2e-3 for after_conv_blocks, 3.5e-2 for
layer_out-0..11, 5.4e-2 for layer_out-12..17, 2.0e-2 for after_transformer,
and 3.9e-2 / 3.0e-2 for projected / embd. P-SR is a proposal. It is not
signed, so no per-tensor tolerance is claimed for either tower.

## Limitations

- Per-tensor equality is not claimed, on CUDA or on Metal. It is pending the
  owner's signature on the P-SR tolerance proposal (EARS-BITTUNE
  `TOLERANCE-PROPOSAL.md`, summarized under
  [Metal tower](#tolerance-against-cuda-and-the-cpu-golden)).
- The Metal tower links only with the slim-chunker `ears.cpp` of the iOS
  lane. This directory's `ears.cpp` needs `aut_encode_batch`, which is
  CUDA-only.
- The ear accepts only 16 kHz mono WAV and does not resample.
- The CUDA tower targets sm_86.
