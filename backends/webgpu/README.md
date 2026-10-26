# WebGPU

Runs the five verbs in a browser tab. There is no model server; the page
reads the model files over HTTP byte ranges and runs them on the GPU.

This backend is preview tier. The `chat` and `decide` engine is a hand-written
WGSL port of the Bend trunk (`bend web-trunk-port`), not code emitted by the
Bend compiler. It is a test bed, not the reference implementation.

## Result

| Verb | Model | Machine | Measured |
|---|---|---|---|
| `chat` | clef-flash Q4_K_S (9B) | Mac Studio, M3 Ultra, Chrome 155 (Metal) | 35.8 tok/s decode (median, 10 reps) |
| `chat` | clef-flash Q4_K_S (9B) | RTX 3090, Chrome 146 (Vulkan/ANGLE) | 16.4 tok/s decode (median, 10 reps) |

Both machines return the same 104-token reply, byte for byte. The engine
issues 413 GPU dispatches per token.

## How each verb runs

| Verb | Runtime | Page |
|---|---|---|
| `chat` | WGSL trunk engine (`trunk/trunk_engine.js`, `trunk/trunk_kernels.js`) | `trunk/smoke.html` |
| `decide` | WGSL trunk engine plus the Clef decision head in plain JavaScript (`trunk/clef_head.js`, `trunk/clef_session.js`) | `trunk/clef.html` |
| `see` | D-FINE-n on onnxruntime-web, WebGPU execution provider | `senses/detect.html` |
| `hear` | whisper-base.en on onnxruntime-web (through transformers.js), WebGPU execution provider | `senses/ears.html`, `senses/battery.html` |
| `speak` | Kokoro-82M on kokoro-js (onnxruntime-web), WebGPU execution provider | `senses/speak.html`, `senses/battery.html` |

`see`, `hear`, and `speak` use onnxruntime-web, not Bend WGSL.

## Layout

| Path | Contents |
|---|---|
| `trunk/` | WGSL trunk engine, Clef decision head, and test pages. |
| `engine/` | GGUF reader, tokenizer (`tok.js`, `tok.wasm`), and the in-tab 4B engine (`q35_*.js`, `bend-worker.js`, `app-worker.js`). |
| `emit/` | WGSL kernels for the 4B engine in `engine/`. |
| `senses/` | `see`, `hear`, and `speak` pages, and the reference assets they check against. |
| `tools/serve.mjs` | Static server with byte ranges and the COOP/COEP headers that `senses/` needs. |
| `tools/run_chat.mjs`, `tools/run_see.mjs` | Playwright drivers that load a page N times and print the results as JSON. |
| `scripts/senses/vendor.sh` | Builds `senses/vendor/` (transformers.js, onnxruntime-web, kokoro-js). |
| `scripts/senses/models.sh` | Downloads the `hear`, `speak`, and `see` models into a models directory. |

## Requirements

- A browser with WebGPU (Chrome 146 or later was tested).
- Node.js 20 or later.
- Playwright and Google Chrome, for the drivers in `tools/`.

## Download separately

These files are not in this repository.

| Item | Path under the models directory |
|---|---|
| `clef-flash-Q4_K_S.gguf` (9B, SHA-256 `ec1472cd7c72cf933ca2da528c466e0932d7814aca503de53965417eec5b49c2`) | `clef-flash-Q4_K_S.gguf` |
| Qwen3.5 tokenizer (`qwen35-tokenizer.bin`, SHA-256 `de71a648a71960a539d4d660ebff804a63a691ba74aea89e3e09dc7bf4553ad9`) | `qwen35-tokenizer.bin` |
| D-FINE-n, folded for the browser (SHA-256 `9c649a78457939665ea13189bdbfe8d1fa42711ab3e332376fafc3579dd468a9`) | `det-web/dfine_n.onnx` |
| whisper-base.en, Kokoro-82M (Hugging Face layout) | `hf/<repo>/` |
| JavaScript runtimes (transformers.js 4.3.1, onnxruntime-web, kokoro-js 1.2.1) | `senses/vendor/` (not under models) |

## Build and run

1. Build the browser runtimes. The script patches kokoro-js to load voices
   from `/models/hf/`. It uses BSD `sed -i ''`; on Linux, change it to
   `sed -i`.

   ```sh
   scripts/senses/vendor.sh
   ```

2. Download the sense models. To fold D-FINE yourself, set `DFINE_ONNX_DIR`
   to a directory with the exported `dfine_n.onnx`.

   ```sh
   scripts/senses/models.sh models
   ```

3. Put `clef-flash-Q4_K_S.gguf` and `qwen35-tokenizer.bin` in `models/`.
4. Start the server:

   ```sh
   node tools/serve.mjs 8802
   ```

   Set `MODELS_DIR` to use another models directory.

5. Open a page:

   | Verb | URL |
   |---|---|
   | `chat` | `http://127.0.0.1:8802/trunk/smoke.html?m=clef-flash-Q4_K_S.gguf&ctx=4096&n=128&p=Hello` |
   | `decide` | `http://127.0.0.1:8802/trunk/clef.html` (one demo request), or `?cases=/cases/<file>.json` |
   | `see` | `http://127.0.0.1:8802/senses/detect.html?bench=60&ep=webgpu` |
   | `hear`, `speak` | `http://127.0.0.1:8802/senses/battery.html` |

To repeat the `chat` measurement:

```sh
node tools/run_chat.mjs "<chat URL>" 10 chat.json
```

`run_chat.mjs` and `run_see.mjs` launch Chrome with the Vulkan/ANGLE flags
used on the RTX 3090. Change the flags for other GPUs.

## Gaps

- The graded `decide` battery (15 cases, 30 questions) is not included.
  `clef.html` without `?cases=` runs one demo request.
- `senses/battery-hear.json` keeps 3 of the 10 graded clips. The other 7 are
  private voice recordings and are not included.
- Kokoro uses eSpeak NG for G2P in kokoro-js. Its licence is not cleared.

## Source

- `trunk/`, `engine/` (except `app-worker.js` and `VENDOR.md`),
  `tools/`: the files that produced the graded receipts (WebGPU battery,
  2026-10-20). They were copied from the Mac test runner, which was not under
  version control. `trunk/trunk_kernels.js` and every file in `engine/` that
  also exists in `bend-app` are byte-identical to `bend-app` at the commit
  below. `trunk/trunk_engine.js` adds the hidden-state readout that the Clef
  head needs (`wantHidden`, `noHead`). `trunk/trunk_session.js` uses absolute
  `/engine/` and `/models/` paths. `clef_head.js`, `clef_session.js`, and
  `clef_prompt.js` are not in `bend-app` at that commit.
- `emit/`, `senses/`, `engine/app-worker.js`, `engine/VENDOR.md`,
  `scripts/`: `github.com/phenomenon0/bend-app`, commit
  `41e85a686e79c14611ae091384bd29082b79e531` (2026-10-19), directories
  `public/` and `scripts/`.
- `senses/battery.html`, `senses/battery-hear.json`,
  `senses/battery-speak.json`: the graded `hear` and `speak` battery.

Changes from the source: `tools/serve.mjs` serves this layout instead of
fixed paths on the test machine; the drivers import `playwright` by package
name; `scripts/senses/vendor.sh` writes to `senses/vendor/`; developer machine
paths removed from comments.
