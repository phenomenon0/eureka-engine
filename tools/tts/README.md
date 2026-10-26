# Text to speech

`tools/tts/` serves speech on the Linux engine path (G41). It runs Kokoro-82M
v1.0 (hexgrad, Apache-2.0) behind `llb-serve`. It is the Linux version of the
phone app's voice layer (VOICESHOP). In the bend app repository, see
`ios/Bend/Voice/KokoroVoice.swift` and `docs/VOICES.md`.

## Files

| File | Description |
|---|---|
| [`kokoro_tts.py`](kokoro_tts.py) | Fetch, worker, and CLI. `KokoroG2P` and `KokoroLTS` are line-for-line ports of the Swift code. |
| [`pins.json`](pins.json) | The sha256 pin of every input. |
| [`test_serve_tts.py`](test_serve_tts.py) | The served-path gate. |

## How it works

- Model and voices: one model, many voices. A voice is its 510 × 256 float32
  style table (522,240 B). Switching the voice swaps a table and does not
  reload the model.
- House voice and voice set: the house voice is `af_bella`. The voice set is
  the eight voices bundled with the app (hexgrad/Kokoro-82M @ f3ff357), plus
  the catalog of the voice shop (HF `jethrovic/bend-models`
  `voices/catalog.json`, pinned revision). That is 15 voices today, including
  the shop's own `af_bella_sarah` and `af_nova` packs.
- G2P: misaki 0.7.4 gold and silver lexicons, with no espeak-ng and no spaCy.
  On a 25-line battery × US/GB, the phonemes are byte-equal to the output of
  the Mac `voice-mac phonemes` command, 50/50.
- The audio pipeline is the same as the app's:
  - Phonemes are split at ≤ 400 symbols.
  - Each piece runs `[0, ids…, 0]` with the style row for `len(ids)`.
  - Each render is trimmed to 50 ms of leading and 200 ms of trailing silence
    (VOICE-LATENCY).
  - Each render then gets a fixed +4 dB lift held under 0.98 peak
    (VOICESHOP2).
- The model file differs from the phone's on purpose:
  - Linux defaults to the fp32 export of kokoro-onnx
    (`kokoro-v1.0.onnx`, 325 MB). The phone's int8 export is about 9× slower
    on x86 onnxruntime CPU (G41 bench: ~1.2 s vs ~11 s for 6.6 s of audio at
    4 threads, same machine).
  - `BEND_KOKORO_MODEL=int8` runs the phone's file.

## Before you begin

- Install python3 with numpy and onnxruntime. The CPU provider is enough.

## Install and serve

1. Fetch the fp32 model and the 15 voices, about 340 MB.

   ```sh
   python3 tools/tts/kokoro_tts.py fetch ~/.local/share/models/voice/kokoro
   ```

   To also fetch the phone's int8 model (92 MB), add `--model both`:

   ```sh
   python3 tools/tts/kokoro_tts.py fetch DIR --model both
   ```

   `pins.json` pins every input by sha256. These are the same pins as the
   app's `scripts/fetch-voice.sh` and `fetch-kokoro.sh`, plus the HF catalog
   at a fixed revision. The fetch is idempotent.

2. Build `llb-serve`. `build-serve.sh` copies `kokoro_tts.py` and
   `pins.json` next to `llb-serve`.

3. Start the server with `--tts`.

   ```sh
   llb-serve --port 8080 --tts ~/.local/share/models/voice/kokoro --engine … -- …
   ```

   `--tts DIR` starts the worker as a resident child process
   (`python3 kokoro_tts.py worker --dir DIR`). Without `--tts`, nothing
   changes: the responses are byte-identical to the base server.

## Routes

| Route | Description |
|---|---|
| `POST /v1/audio/speech` | OpenAI shape: `{"input", "voice"?, "speed"? (0.25–4), "response_format"? ("wav" default \| "pcm" s16le 24 kHz mono)}` → `audio/wav`. Headers: `X-LLB-TTS-Voice`, `X-LLB-TTS-Synth-Ms`, `X-LLB-TTS-Ms`, `X-LLB-Audio-Seconds`, `X-LLB-Audio-Sha256`. |
| `GET /v1/audio/voices`, `GET /voice/voices` | `{"voices":[{id, engine, name, lang, gender, grade, origin, house}], "selected", "house_voice"}`. This is the app VoiceServer's `/voice/voices`. |
| `POST /voice/select {"id"}` | Sets the default voice from the next request on. This is the hot swap of VoiceServer. Nothing reloads. |

- Voice ids: `kokoro:af_bella` or the bare `af_bella`. An unknown voice
  returns 400 with a message that names `/v1/audio/voices`. OpenAI's `alloy`
  and its other voice names are not mapped.
- Errors use the OpenAI error shape of llb-serve. The worker returns 400 for
  empty input, input over 4096 characters, an unknown voice, mp3, opus, or
  aac, and speed out of range. A worker that is loading or gone returns 503.
- Concurrency: one render at a time, FIFO on the worker's pipe. Chat
  completions keep running on the engine meanwhile.
- Startup: on start, the worker renders "Ready." once to warm the graph.
  Before that, the speech routes return 503 "Loading voice".
- Shutdown: SIGTERM drains as before, then closes the worker's stdin. The
  worker exits 0. If it has not exited after 10 s, it gets SIGTERM.

## Options

| Variable | Default | Description |
|---|---|---|
| `BEND_KOKORO_THREADS` | 4 | Worker threads. |
| `BEND_KOKORO_MODEL` | The `model` value in `kokoro.json`. If that is not set, `fp32`. | `fp32`, `int8`, or a model file path. |
| `BEND_VOICE` | `af_bella` (the house voice) | The starting voice. |
| `LLB_TTS_PYTHON` | `python3` | Python interpreter for the worker. |
| `LLB_TTS_SCRIPT` | `kokoro_tts.py` next to the binary | Worker script. |

## Use the CLI

- Render a WAV file. The command prints a receipt line.

  ```sh
  python3 tools/tts/kokoro_tts.py say --text "Hello there." --voice af_bella --out hello.wav
  ```

- Run the G2P only.

  ```sh
  python3 tools/tts/kokoro_tts.py phonemes --text "Dr. Okafor paid \$3.50."
  ```

## Run the gate

Run the gate script.

```sh
tools/tts/test_serve_tts.py --serve BIN --engine ENGINE --tts DIR --out OUT [-- ENGINE_ARGS]
```

Replace `BIN` with the `llb-serve` binary, `ENGINE` with the engine binary,
`DIR` with the voice pack directory, and `OUT` with the output directory.

The script runs 26 checks:

- Routes and response shapes.
- The sha in the header equals the sha of the body.
- WAV framing.
- Determinism.
- Voice per request and through `/voice/select`.
- The PCM format equals the WAV samples.
- Speed.
- 9 error paths.
- 4 concurrent requests plus a chat request.
- A clean SIGTERM.

It writes `receipts.jsonl` with these fields for each render: text, voice,
bytes, sha256, end-to-end ms, synth ms, audio seconds, and load average.

## Limitations

- No streaming. A reply is rendered whole. For lower first-audio latency, the
  client splits the text as the app does: the first piece is ≤ 6 words.
- English only: US and GB lexicons.
- Names that are not in the lexicon go through the letter rules. These are a
  best guess, as on the phone.
- CPU only on this machine. onnxruntime-gpu is not installed, and the 3090 is
  reserved for the engines under the house lock.
