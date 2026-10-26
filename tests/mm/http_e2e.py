#!/usr/bin/env python3
"""G5 HTTP end-to-end: the real llb-serve binary over a stub engine (stub_engine.py).
  http_e2e.py --serve llb-serve --stub stub_engine.py --fix FIXDIR --out OUTDIR [--port 18531]
Server configs: --mm none (default), --mm image (+ remote private), --mm image,audio, and --ear (EARS-CANON, stub_ear.py).
Writes OUTDIR/http_e2e.json + OUTDIR/serve-*.log; exit 1 on any failed check."""
import argparse, base64, hashlib, http.server, io, json, os, signal, socket, subprocess, sys, threading, time, urllib.request, wave

ap = argparse.ArgumentParser()
for k in ("serve", "stub", "fix", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--port", type=int, default=18531)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FIX = lambda f: open(os.path.join(a.fix, f), "rb").read()
uri = lambda f, mime: f"data:{mime};base64," + base64.b64encode(FIX(f)).decode()
img = lambda u, **kw: {"type": "image_url", "image_url": {"url": u, **kw}}
txt = lambda t: {"type": "text", "text": t}
sha = lambda b: hashlib.sha256(b).hexdigest()

# a loopback HTTP server for image_url fetches
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.lstrip("/")
        if p == "redirect":
            self.send_response(302); self.send_header("Location", "/a.png"); self.end_headers(); return
        f = os.path.join(a.fix, p)
        if not os.path.isfile(f): self.send_response(404); self.end_headers(); return
        b = open(f, "rb").read(); self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *x): pass
    def handle(self):
        try: super().handle()
        except (ConnectionResetError, BrokenPipeError): pass  # curl hung up at --max-filesize / our cap
hs = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); IMGPORT = hs.server_address[1]
threading.Thread(target=hs.serve_forever, daemon=True).start()

def post(port, body, path="/v1/chat/completions", raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=60) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()

def start(tag, port, extra):
    tmp = f"{a.out}/tmp-{tag}"; os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, LLB_GGUF="/nonexistent/stub.gguf", TMPDIR=tmp, STUB_LOG=f"{a.out}/stub-{tag}.jsonl", STUB_KEEP=f"{a.out}/kept")
    os.makedirs(f"{a.out}/kept", exist_ok=True)
    lf = open(f"{a.out}/serve-{tag}.log", "w")
    p = subprocess.Popen([a.serve, "--port", str(port), "--alias", "stub", "--engine", a.stub, *extra, "--", "stub"],
                         env=env, stderr=lf, stdout=lf)
    procs.append(p)
    for _ in range(200):
        if p.poll() is not None: sys.exit(f"llb-serve [{tag}] exited rc {p.returncode}: see serve-{tag}.log")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200: break
        except Exception: time.sleep(0.1)
    return p, tmp

R, bad, procs = [], 0, []
import atexit
atexit.register(lambda: [q.kill() for q in procs if q.poll() is None])
def check(cfg, name, ok, **info):
    global bad
    bad += not ok; R.append({"config": cfg, "check": name, "pass": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + f"[{cfg}] {name}" + (f"  {info}" if not ok else ""), flush=True)

def verdict(body):  # the stub's reply text = what the engine saw
    return json.loads(json.loads(body)["choices"][0]["message"]["content"])

def err(body): return json.loads(body)["error"]

PNG, JPG, WEBP = uri("a.png", "image/png"), uri("coco142472.jpg", "image/jpeg"), uri("a-lossy.webp", "image/webp")
one = {"messages": [{"role": "user", "content": [img(JPG), txt("Is there a car in the image?")]}], "max_tokens": 64}

# ---------------- config A: default --mm none (the shipped SYS1 text engine) ----------------
p, tmp = start("none", a.port, [])
s, b = post(a.port, {"messages": [{"role": "user", "content": [txt("hello")]}], "max_tokens": 8})
check("none", "text parts still served (200)", s == 200 and verdict(b)["media"] == [], status=s)
s, b = post(a.port, one)
check("none", "1 image -> 400 no vision path, param points at the part", s == 400 and "no vision path" in err(b)["message"]
      and err(b)["param"] == "messages[0].content[0]", status=s, body=b[:200])
s, b = post(a.port, {"messages": [{"role": "user", "content": [img("data:text/plain;base64,aGk=")]}]})
check("none", "content-type is validated before the engine gate (415)", s == 415, status=s)
p.send_signal(signal.SIGTERM); p.wait(30)

# ---------------- config B: --mm image, remote fetch allowed for private hosts ----------------
port = a.port + 1
p, tmp = start("image", port, ["--mm", "image", "--mm-remote", "private"])
s, b = post(port, one)
v = verdict(b) if s == 200 else {}
check("image", "1 image (base64 jpeg) -> 200, engine got 1 media entry, sha/bytes verified", s == 200 and v.get("ok")
      and len(v["media"]) == 1 and v["media"][0]["sha256"] == sha(FIX("coco142472.jpg")) and v["media"][0]["w"] == 640
      and v["media"][0]["h"] == 480 and v["media"][0]["mime"] == "image/jpeg" and v["media"][0]["mode"] == "0o600", verdict=v)
usage = json.loads(b)["usage"] if s == 200 else {}
check("image", "usage.prompt_tokens counts the expanded rows (>= 64)", usage.get("prompt_tokens", 0) >= 64, usage=usage)
check("image", "staged media file removed after the request", os.listdir(f"{tmp}/llb-serve-mm") == [], left=os.listdir(f"{tmp}/llb-serve-mm"))
two = {"messages": [{"role": "user", "content": [txt("Compare "), img(PNG, detail="low"), txt(" and "), img(WEBP), txt(".")]}], "max_tokens": 64}
s, b = post(port, two); v = verdict(b) if s == 200 else {}
check("image", "2 images (png + webp) -> 200, order + detail kept", s == 200 and v.get("ok") and
      [m["mime"] for m in v["media"]] == ["image/png", "image/webp"] and v["media"][0]["detail"] == "low"
      and v["media"][1]["detail"] == "auto", verdict=v)
st = {**one, "stream": True, "stream_options": {"include_usage": True}}
rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(st).encode(), headers={"Content-Type": "application/json"})
ev = [l[6:] for l in urllib.request.urlopen(rq, timeout=60).read().decode().split("\n") if l.startswith("data: ")]
txt_s = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in ev[:-1] if json.loads(e)["choices"])
check("image", "1 image streamed: pieces concatenate to the engine verdict, [DONE] last", ev[-1] == "[DONE]" and json.loads(txt_s).get("ok"), n_events=len(ev))
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/coco142472.jpg"), txt("car?")]}], "max_tokens": 64})
v = verdict(b) if s == 200 else {}
check("image", "image_url http (loopback, --mm-remote private) fetched, sha = file", s == 200 and v.get("ok") and v["media"][0]["sha256"] == sha(FIX("coco142472.jpg")), status=s, body=b[:300])
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/redirect")]}]})
check("image", "redirects are not followed (400)", s == 400 and "redirect" in err(b)["message"], status=s, body=b[:200])
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/a.txt")]}]})
check("image", "fetched non-image bytes -> 415", s == 415, status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/big.bin")]}]})
check("image", "fetched 7 MiB over the 6 MiB image cap -> 413", s == 413, status=s, body=b[:200])
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/missing.png")]}]})
check("image", "fetch 404 -> 400", s == 400, status=s)
big = json.dumps({"messages": [{"role": "user", "content": [img(uri("big.bin", "image/png"))]}]}).encode()
sk = socket.create_connection(("127.0.0.1", port), timeout=30)  # headers only: the server answers before the body
sk.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(big)}\r\n\r\n".encode())
resp = b"".join(iter(lambda: sk.recv(65536), b"")).decode(); sk.close()
check("image", f"{len(big) / 2**20:.1f} MiB base64 body over the 8 MiB body cap -> 413", resp.startswith("HTTP/1.1 413"), resp=resp[:80])
mid = {"messages": [{"role": "user", "content": [img("data:image/png;base64," + base64.b64encode(FIX("big.bin")[:5 * 2**20]).decode())]}]}
s, b = post(port, mid)
check("image", "5 MiB png-magic body without IHDR -> 400 corrupt header (under every byte cap)", s == 400 and "corrupt" in err(b)["message"], status=s, body=b[:200])
s, b = post(port, {"messages": [{"role": "user", "content": [img(uri("wide.png", "image/png"))]}]})
check("image", "9000x8 image over the side cap -> 413", s == 413 and err(b)["param"] == "messages[0].content[0]", status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img("data:image/png;base64,iVBOR***")]}]})
check("image", "bad base64 -> 400", s == 400, status=s)
s, b = post(port, {"messages": [{"role": "system", "content": [img(PNG)]}, {"role": "user", "content": "x"}]})
check("image", "image in system message -> 400 (template rule)", s == 400 and "System message" in err(b)["message"], status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img(PNG)] * 9}]})
check("image", "9 images over max 8 -> 400", s == 400 and "too many images" in err(b)["message"], status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img(PNG)] * 8}], "max_tokens": 64})
check("image", "8 images (the cap) -> 200, 8 entries", s == 200 and len(verdict(b)["media"]) == 8, status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(FIX("a.wav")).decode(), "format": "wav"}}]}]})
check("image", "audio with no ear attached -> 400", s == 400 and "ear" in err(b)["message"], status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img(PNG), txt("weather?")]}], "tools": [{"type": "function", "function": {"name": "f"}}]})
check("image", "tool+image -> 400 tools not supported (combo deferred to the tools lane)", s == 400 and "tools" in err(b)["message"], status=s)
s, b = post(port, {"prompt": "plain completion", "max_tokens": 4}, path="/v1/completions")
check("image", "/v1/completions unaffected (no media key)", s == 200, status=s)
check("image", "no staged files left after all requests", os.listdir(f"{tmp}/llb-serve-mm") == [], left=os.listdir(f"{tmp}/llb-serve-mm"))
lines = [json.loads(l) for l in open(f"{a.out}/stub-image.jsonl")]
check("image", "text-only requests carry no media key (engine line unchanged)", all("media" not in l["req"] for l in lines if "<|image_pad|>" not in l["prompt"]))
p.send_signal(signal.SIGTERM); p.wait(30)

# ---------------- config B2: --mm-remote on (public only): loopback refused ----------------
port = a.port + 2
p, tmp = start("remote-on", port, ["--mm", "image", "--mm-remote", "on"])
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://127.0.0.1:{IMGPORT}/a.png")]}]})
check("remote-on", "--mm-remote on refuses a loopback target (SSRF guard)", s == 400 and "private address" in err(b)["message"], status=s)
s, b = post(port, {"messages": [{"role": "user", "content": [img(f"http://localhost:{IMGPORT}/a.png")]}]})
check("remote-on", "hostname resolving to loopback refused too", s == 400 and "private address" in err(b)["message"], status=s)
p.send_signal(signal.SIGTERM); p.wait(30)

# ---------------- config C: --mm image,audio (the ear attached) ----------------
port = a.port + 3
p, tmp = start("audio", port, ["--mm", "image,audio"])
s, b = post(port, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(FIX("a.wav")).decode(), "format": "wav"}}, txt("transcribe")]}], "max_tokens": 64})
v = verdict(b) if s == 200 else {}
check("audio", "wav input_audio -> 200, engine got an audio/wav entry", s == 200 and v.get("ok") and v["media"][0]["mime"] == "audio/wav", verdict=v)
s, b = post(port, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(FIX("a.wav")).decode(), "format": "mp3"}}]}]})
check("audio", "format mp3 on wav bytes -> 415", s == 415, status=s)
s, b = post(port, {}, "/v1/audio/transcriptions")
check("audio", "no --ear: /v1/audio/transcriptions is not served (404)", s == 404, status=s)
p.send_signal(signal.SIGTERM); p.wait(30)

# ---------------- config D: --ear (EARS-CANON) with the stub ear: transcription route + chat audio as the user turn ----------------
port = a.port + 4
edir = f"{a.out}/ear-models"; os.makedirs(edir, exist_ok=True)
for f in ("Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-bf16.gguf"): open(f"{edir}/{f}", "wb").close()
os.environ["LLB_EAR_BIN"] = os.path.join(os.path.dirname(os.path.abspath(a.stub)), "stub_ear.py")
os.environ["STUB_EAR_LOG"] = f"{a.out}/ear-requests.jsonl"
p, tmp = start("ear", port, ["--ear", edir])
W = FIX("a.wav"); WSHA = hashlib.sha256(W).hexdigest(); WT = f"stub transcript {WSHA[:8]}"
IA = lambda data=base64.b64encode(W).decode(), fmt="wav": {"data": data, "format": fmt}
for _ in range(100):  # the ear loads beside the engine (like --tts): 503 "Loading ear" until it is ready
    s, b = post(port, {"input_audio": IA()}, "/v1/audio/transcriptions")
    if s != 503: break
    time.sleep(0.1)
check("ear", "POST /v1/audio/transcriptions wav -> 200 {text} (OpenAI json shape)", s == 200 and json.loads(b) == {"text": WT}, status=s, body=b[:200])
s, b = post(port, {"input_audio": IA(), "response_format": "verbose_json"}, "/v1/audio/transcriptions")
v = json.loads(b) if s == 200 else {}
check("ear", "verbose_json: task/language/duration + llb.ids/top1/timings", s == 200 and v.get("task") == "transcribe" and v.get("language") == "English"
      and v["llb"]["ids"] == [11528, 6364, 151645] and "timings" in v["llb"] and "ok" not in v["llb"], body=b[:300])
rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/transcriptions", headers={"Content-Type": "application/json"},
                            data=json.dumps({"input_audio": IA(), "response_format": "text"}).encode())
with urllib.request.urlopen(rq, timeout=60) as r: ct, tb, xs = r.headers["Content-Type"], r.read().decode(), r.headers["X-LLB-Audio-Sha256"]
check("ear", "response_format text -> text/plain transcript + X-LLB-Audio-Sha256", ct.startswith("text/plain") and tb == WT + "\n" and xs == WSHA, ct=ct, body=tb)
s, b = post(port, {"input_audio": IA(fmt="mp3")}, "/v1/audio/transcriptions")
check("ear", "transcription: format mp3 on wav bytes -> 415 (param input_audio)", s == 415 and err(b)["param"] == "input_audio", status=s)
s, b = post(port, {"input_audio": IA(base64.b64encode(FIX("a.mp3")).decode(), "mp3")}, "/v1/audio/transcriptions")
check("ear", "transcription: a real mp3 -> 400 (the ear takes 16 kHz mono WAV)", s == 400 and "WAV" in err(b)["message"], status=s, body=b[:200])
s, b = post(port, {}, "/v1/audio/transcriptions")
check("ear", "transcription: no input_audio -> 400", s == 400, status=s)
s, b = post(port, {"input_audio": IA(), "response_format": "srt"}, "/v1/audio/transcriptions")
check("ear", "transcription: response_format srt -> 400", s == 400, status=s)
# chat: the audio becomes the user turn; the engine line must equal the one for a TEXT request carrying the transcript
s, b = post(port, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": IA()}, txt(" summarize")]}], "max_tokens": 64})
va = verdict(b) if s == 200 else {}
s2, b2 = post(port, {"messages": [{"role": "user", "content": WT + " summarize"}], "max_tokens": 64})
vt = verdict(b2) if s2 == 200 else {}
check("ear", "chat input_audio -> 200; the engine saw no media, the transcript is the user turn", s == 200 and va.get("ok") and va["media"] == []
      and va["prompt_sha256"] == vt.get("prompt_sha256"), audio=va, text=vt)
ln = [json.loads(l)["line_sha256"] for l in open(f"{a.out}/stub-ear.jsonl")]
check("ear", "chat audio request == the text request with its transcript, byte for byte on the engine pipe", len(ln) >= 2 and ln[-1] == ln[-2], lines=ln[-2:])
eq = [json.loads(l) for l in open(f"{a.out}/ear-requests.jsonl")]
check("ear", "the ear got the staged file + its sha256 (contract step 1)", any(q.get("sha256") == WSHA for q in eq))
bio = io.BytesIO()
with wave.open(bio, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(bytes(1600))
s, b = post(port, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": IA(base64.b64encode(bio.getvalue()).decode())}]}]})
left = os.listdir(f"{tmp}/llb-serve-mm") if os.path.isdir(f"{tmp}/llb-serve-mm") else []
check("ear", "chat: an ear refusal (8 kHz wav) -> 400 'ear: ...', staged file removed", s == 400 and err(b)["message"].startswith("ear: ") and not left,
      status=s, body=b[:200], left=left)
s, b = post(port, {"batch": [{"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": IA()}]}], "max_tokens": 8},
                             {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}]})
check("ear", "batch: an audio chat item is transcribed too (200, no media at the engine)", s == 200, status=s, body=b[:300])
p.send_signal(signal.SIGTERM); p.wait(30)
hs.shutdown()
json.dump({"passed": len(R) - bad, "total": len(R), "checks": R}, open(f"{a.out}/http_e2e.json", "w"), indent=1)
print(f"{len(R) - bad}/{len(R)} checks passed"); sys.exit(1 if bad else 0)
