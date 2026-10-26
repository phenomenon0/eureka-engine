# On-device ASR (ANDROIDVOICE)

whisper.cpp v1.9.2 (306c88f), ggml CPU, NDK 27.2, arm64 `armv8.2-a+dotprod+fp16`, no OpenMP, static → `libbendwhisper.so`
(2.9 MB) + `assets/asr/ggml-base.en-q8_0.bin` (81.8 MB, stored). `build.sh <scratch>` rebuilds both; sums in `SHA256SUMS`.
No platform recognizer (the Pixel 6a's SODA pack is missing: LANGUAGE_PACK_ERROR 12), no network.

## Numbers (Pixel 6a, Tensor G1, `taskset f0`, 4 threads, greedy, 2026-10-09)

whisper-cli on jfk.wav (11 s), 3 runs, cool phone:

| model | total ms | encode ms |
|---|---|---|
| tiny.en f16 | 1001–1149 | 598–706 |
| **tiny.en q8_0** | 837–914 | 447–471 |
| tiny.en q5_1 | 1458–1467 | 917–923 |
| **base.en q8_0** | 1626–1662 | 929–986 |
| base.en q5_1 | 2928–3001 | 2164–2233 |
| base.en f16 | 2365–2481 | 1581–1675 |

q8_0 is the fastest on this CPU (dotprod, no i8mm); q5_1 is slower than f16. All transcripts were exact.

Accuracy: 20 piper phrases (2 voices × 10 assistant asks) played from a second app on the same phone's speaker while the
app held the mic; the app kept each heard turn as a wav (`--ez keep_audio true`). Both models were then run on those same
19 wavs (one botched hold dropped) on the 6a (thermal status 1):

| model | WER | ms p50 / max |
|---|---|---|
| tiny.en q8_0 | 12.2 % (17/139) | 621 / 691 |
| **base.en q8_0** | 10.8 % (15/139) | 1282 / 1347 |
| tiny / base with `audio_ctx` fit to the clip | 33 % / 20 % (host run) | ~3× faster |

**base.en q8_0 is bundled.** The errors it fixes are times, the words a reminder or alarm turns on: tiny heard "call my
mother at six" as "at sex" and "seven thirty" as "7th Thursday". It costs about 0.66 s per turn, and the turn's first word is
15–22 s (the chat model). `audio_ctx` loops ("What is on Thursday?" ×4), so it was rejected.

A whole `.bin` dropped into `<external files>/asr/` replaces the bundled model (how the two were compared in-app).

Ceilings: a hot phone (thermal status 2) roughly doubles ASR (tiny jfk 0.85 → 1.6 s). Whisper competing with a still-running
engine decode on the big cores pushed in-app tiny to 4–4.8 s.
