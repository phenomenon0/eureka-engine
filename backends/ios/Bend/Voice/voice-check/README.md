# Voice fixtures (PHONEVOICE)

Fed by `Voice.FixtureSource` at real-time pace in 100 ms buffers, as the mic tap would deliver them.

| clip | source | expected words |
|---|---|---|
| `book.wav` | `~/stt-bench/q_book.wav` (48 kHz int16, 2.23 s) | Can you recommend a good book about leadership? |
| `advice.wav` | `~/stt-bench/q_advice.wav` (48 kHz int16, 3.38 s) | What advice would you give to a young entrepreneur starting out today? |
| `lesson-mic.wav` | **live-mic-shaped** from `q_lesson.wav`: 48 kHz float32 (the iPhone tap's format), 0.7 s room tone before and 0.5 s after, −9 dB, high-pass (phone-mic roll-off), pink room noise at ~22 dB SNR, 60 Hz hum. Seeded, samples reproducible (`scripts/voice-fixtures.py`; the float WAV's PEAK chunk carries a timestamp, so the file hash varies). | Richie, what is the best lesson you ever learned? |

The WAV clips are not in this repository. Record or synthesize clips with the
names and formats in the table and place them in this directory before you
run `-voiceCheck`.
