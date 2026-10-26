#!/usr/bin/env python3
"""EARS-CANON stub ear: speaks llb-ear's worker protocol (tools/ears/ear_worker.cpp) without a model, so the server's
--ear wiring is testable on CPU. It checks the staged file's sha256 (docs/reference/multimodal-content.md §6 step 1), refuses anything but a
16 kHz mono WAV like the real ear, and transcribes a clip as "stub transcript <first 8 hex of its sha256>".
Every request is appended to $STUB_EAR_LOG (one JSON line)."""
import hashlib, json, os, struct, sys

log = open(os.environ["STUB_EAR_LOG"], "a") if os.environ.get("STUB_EAR_LOG") else None
print("ear: ready (stub)", file=sys.stderr, flush=True)
def out(o): print(json.dumps(o, separators=(",", ":")), flush=True)
for line in sys.stdin:
    q = json.loads(line)
    if log: log.write(json.dumps(q) + "\n"); log.flush()
    if q.get("op") == "ping": out({"ok": True, "version": "stub"}); continue
    try: b = open(q["path"], "rb").read()
    except OSError: out({"ok": False, "status": 400, "error": "ear: cannot read the staged audio file"}); continue
    sha = hashlib.sha256(b).hexdigest()
    if q.get("sha256") and q["sha256"] != sha: out({"ok": False, "status": 400, "error": "ear: the staged audio file does not match its sha256"}); continue
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE": out({"ok": False, "status": 400, "error": "the ear takes 16 kHz mono WAV"}); continue
    ch, sr = struct.unpack("<HI", b[22:28])
    if ch != 1 or sr != 16000: out({"ok": False, "status": 400, "error": f"ears: need 16 kHz mono (got {ch} ch, {sr} Hz)"}); continue
    t = f"stub transcript {sha[:8]}"
    out({"ok": True, "text": t, "language": "English", "raw": f"language English<asr_text>{t}", "ids": [11528, 6364, 151645],
         "top1": ["0x1p+4"] * 3, "top2": ["0x1p+3"] * 3, "rows_sha256": "0" * 64, "n_windows": 1, "n_audio_tokens": 13,
         "audio_s": (len(b) - 44) / 32000, "timings": {"total_ms": 0.0}})
