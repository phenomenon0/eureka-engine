#!/usr/bin/env python3
"""tools/tts/test_serve_tts.py: G41 gate for llb-serve --tts (speech on the served path).

  test_serve_tts.py --serve BIN --engine ENGINE --tts DIR --out OUT [--port P] [--engine-args ...]

Starts `BIN --port P --tts DIR --engine ENGINE -- ENGINE_ARGS`, waits for /health and the voice worker, then checks:
routes + shapes, header sha == body sha, WAV framing, voice selection (per request + /voice/select hot swap), PCM,
the error paths (400/405/404), concurrent requests (serialised, all complete), that chat completions still work with
the TTS child resident, and a clean SIGTERM shutdown (engine + worker exit). Writes OUT/receipts.jsonl (one line per
render: text, voice, bytes, sha256, client e2e ms, worker synth ms, audio seconds, load average) + OUT/checks.json.
Exit 0 iff every check passed. Works with the G6 mock engine (sys1dual/g6-slots/mock-engine.py) or a real engine.
"""
import argparse, hashlib, http.client, io, json, os, signal, subprocess, sys, threading, time, wave

ap = argparse.ArgumentParser()
ap.add_argument("--serve", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--tts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=18641)
ap.add_argument("--chat-n", type=int, default=8)
ap.add_argument("--ready-timeout", type=float, default=600)
ap.add_argument("engine_args", nargs="*")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
checks, receipts = [], []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def req(method, path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=timeout)
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    t0 = time.monotonic()
    c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data is not None else {})
    r = c.getresponse()
    raw = r.read()
    ms = (time.monotonic() - t0) * 1000
    c.close()
    return r.status, dict(r.getheaders()), raw, ms


def speak(text, voice=None, fmt=None, speed=None, tag=""):
    body = {"model": "kokoro", "input": text}
    if voice is not None:
        body["voice"] = voice
    if fmt is not None:
        body["response_format"] = fmt
    if speed is not None:
        body["speed"] = speed
    st, h, raw, ms = req("POST", "/v1/audio/speech", body)
    rec = {"tag": tag, "status": st, "text": text, "voice_req": voice, "format": fmt or "wav", "speed": speed,
           "content_type": h.get("Content-Type"), "voice": h.get("X-LLB-TTS-Voice"), "bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "header_sha256": h.get("X-LLB-Audio-Sha256"),
           "e2e_ms": round(ms, 1), "server_tts_ms": float(h.get("X-LLB-TTS-Ms", "nan")),
           "synth_ms": float(h.get("X-LLB-TTS-Synth-Ms", "nan")), "audio_s": float(h.get("X-LLB-Audio-Seconds", "nan")),
           "load1": round(os.getloadavg()[0], 2)}
    if st == 200 and (fmt or "wav") == "wav":
        w = wave.open(io.BytesIO(raw))
        rec["wav"] = {"channels": w.getnchannels(), "rate": w.getframerate(), "width": w.getsampwidth(), "frames": w.getnframes()}
        with open(os.path.join(a.out, f"{tag or 'speech'}.wav"), "wb") as f:
            f.write(raw)
    if st == 200:
        receipts.append(rec)
    else:
        rec["error"] = raw.decode(errors="replace")
    return rec


errf = open(os.path.join(a.out, "server.stderr"), "w")
env = dict(os.environ)
env.setdefault("LLB_GGUF", "/mock/model.gguf")
cmd = [a.serve, "--port", str(a.port), "--alias", "g41", "--tts", a.tts, "--engine", a.engine, "--"] + a.engine_args
srv = subprocess.Popen(cmd, stderr=errf, env=env)
t0 = time.monotonic()
health = voices_up = None
while time.monotonic() - t0 < a.ready_timeout:
    try:
        if health is None and req("GET", "/health")[0] == 200:
            health = time.monotonic() - t0
        if voices_up is None and req("GET", "/v1/audio/voices")[0] == 200:
            voices_up = time.monotonic() - t0
        if health is not None and voices_up is not None:
            break
    except OSError:
        pass
    if srv.poll() is not None:
        break
    time.sleep(0.1)
check("server up (engine /health 200 + voice worker ready)", health is not None and voices_up is not None,
      f"health_s={health and round(health, 2)} voices_s={voices_up and round(voices_up, 2)}")
if health is None or voices_up is None:
    srv.kill()
    json.dump(checks, open(os.path.join(a.out, "checks.json"), "w"), indent=1)
    sys.exit(1)

st, h, raw, _ = req("GET", "/v1/audio/voices")
v = json.loads(raw)
ids = [x["id"] for x in v["voices"]]
check("GET /v1/audio/voices", st == 200 and v["house_voice"] == "kokoro:af_bella" and v["selected"] == "kokoro:af_bella"
      and "kokoro:af_nova" in ids and "kokoro:bm_george" in ids and "ok" not in v and "bytes" not in v,
      f"{len(ids)} voices, house={v.get('house_voice')}, selected={v.get('selected')}")
st2, _, raw2, _ = req("GET", "/voice/voices")
check("GET /voice/voices (VoiceServer alias) == /v1/audio/voices", st2 == 200 and raw2 == raw)

TEXTS = [
    ("demo", "Hi, I'm the voice Bend will use to read its replies. Shall we plan your Saturday, or keep it simple?"),
    ("short", "Ready when you are."),
    ("numbers", "The meeting moved to 3:30 p.m. on Oct. 14th, so bring the 2026 budget."),
    ("names", "Dr. Okafor paid $3.50 for coffee at Kolawole's café."),
]
for tag, t in TEXTS:
    r = speak(t, tag=tag)
    check(f"speech {tag}: 200 audio/wav, 24 kHz mono s16, header sha == body sha",
          r["status"] == 200 and r["content_type"] == "audio/wav" and r.get("wav") == {"channels": 1, "rate": 24000, "width": 2, "frames": (r["bytes"] - 44) // 2}
          and r["sha256"] == r["header_sha256"] and r["voice"] == "kokoro:af_bella",
          f"bytes={r['bytes']} e2e_ms={r['e2e_ms']} synth_ms={r['synth_ms']} audio_s={r['audio_s']}")
again = speak(TEXTS[1][1], tag="short-again")
check("deterministic: same text + voice -> same bytes", again["sha256"] == receipts[1]["sha256"], again["sha256"][:16])

g = speak(TEXTS[1][1], voice="kokoro:bm_george", tag="short-george")
check("per-request voice (kokoro:bm_george, British lexicon)", g["status"] == 200 and g["voice"] == "kokoro:bm_george"
      and g["sha256"] != receipts[1]["sha256"], f"sha={g['sha256'][:16]}")
n = speak(TEXTS[1][1], voice="af_nova", tag="short-nova")
check("bare id from the HF shop catalog (af_nova)", n["status"] == 200 and n["voice"] == "kokoro:af_nova")
p = speak(TEXTS[1][1], fmt="pcm", tag="short-pcm")
check("response_format pcm = the WAV's samples", p["status"] == 200 and p["content_type"].startswith("audio/pcm")
      and p["bytes"] == receipts[1]["bytes"] - 44
      and hashlib.sha256(open(os.path.join(a.out, "short.wav"), "rb").read()[44:]).hexdigest() == p["sha256"])
fast = speak(TEXTS[1][1], speed=1.5, tag="short-fast")
check("speed 1.5 is shorter", fast["status"] == 200 and fast["audio_s"] < receipts[1]["audio_s"], f"{fast['audio_s']} < {receipts[1]['audio_s']}")

st, _, raw, _ = req("POST", "/voice/select", {"id": "kokoro:af_heart"})
sel = speak(TEXTS[1][1], tag="short-after-select")
st3, _, raw3, _ = req("GET", "/v1/audio/voices")
check("POST /voice/select hot-swaps the default voice", st == 200 and json.loads(raw)["selected"] == "kokoro:af_heart"
      and sel["voice"] == "kokoro:af_heart" and json.loads(raw3)["selected"] == "kokoro:af_heart")
req("POST", "/voice/select", {"id": "kokoro:af_bella"})

for name, method, path, body, want in [
    ("missing input", "POST", "/v1/audio/speech", {"voice": "af_bella"}, 400),
    ("empty input", "POST", "/v1/audio/speech", {"input": "  "}, 400),
    ("unknown voice", "POST", "/v1/audio/speech", {"input": "hi", "voice": "alloy"}, 400),
    ("mp3 refused", "POST", "/v1/audio/speech", {"input": "hi", "response_format": "mp3"}, 400),
    ("speed out of range", "POST", "/v1/audio/speech", {"input": "hi", "speed": 9}, 400),
    ("bad JSON", "POST", "/v1/audio/speech", b"{nope", 400),
    ("GET on speech", "GET", "/v1/audio/speech", None, 405),
    ("select unknown", "POST", "/voice/select", {"id": "kokoro:nobody"}, 400),
    ("select no id", "POST", "/voice/select", {}, 400),
]:
    st, h, raw, _ = req(method, path, body)
    try:
        e = json.loads(raw)["error"]
        shape = isinstance(e["message"], str) and e["code"] == want
    except Exception:
        shape = False
    check(f"error: {name} -> {want} (OpenAI error shape)", st == want and shape, raw.decode(errors="replace")[:140])
st, _, raw, _ = req("GET", "/health")
check("worker survives bad requests (/health 200, speech still 200)", st == 200 and speak("Still here.", tag="still")["status"] == 200)

# concurrency: 4 speech + 1 chat at once; speech serialises on the worker, chat runs on the engine meanwhile
res = {}


def worker(i):
    res[i] = speak(f"Request number {i + 1} is speaking now.", tag=f"conc{i}")


def chat():
    st, h, raw, ms = req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": a.chat_n})
    res["chat"] = {"status": st, "ms": round(ms, 1), "body": raw.decode(errors="replace")[:300]}


th = [threading.Thread(target=worker, args=(i,)) for i in range(4)] + [threading.Thread(target=chat)]
tc = time.monotonic()
[t.start() for t in th]
[t.join() for t in th]
wall = (time.monotonic() - tc) * 1000
conc_ok = all(res[i]["status"] == 200 and res[i]["sha256"] == res[i]["header_sha256"] for i in range(4))
check("4 concurrent speech requests all complete, bytes intact", conc_ok,
      f"wall_ms={wall:.0f} e2e_ms={[res[i]['e2e_ms'] for i in range(4)]}")
check("chat completion alongside speech (engine path unaffected)", res["chat"]["status"] == 200 and '"choices"' in res["chat"]["body"],
      f"chat_ms={res['chat']['ms']}")

srv.send_signal(signal.SIGTERM)
try:
    rc = srv.wait(timeout=60)
except subprocess.TimeoutExpired:
    srv.kill()
    rc = "timeout"
errf.close()
log = open(os.path.join(a.out, "server.stderr")).read()
check("SIGTERM: clean shutdown, worker exits rc 0", rc == 0 and "tts exit rc 0" in log and "shutdown complete" in log, f"rc={rc}")

with open(os.path.join(a.out, "receipts.jsonl"), "w") as f:
    for r in receipts:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"pass": all(c["ok"] for c in checks), "n": len(checks), "failed": [c["check"] for c in checks if not c["ok"]], "checks": checks},
          open(os.path.join(a.out, "checks.json"), "w"), indent=1)
print(f"{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
sys.exit(0 if all(c["ok"] for c in checks) else 1)
