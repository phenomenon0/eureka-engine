#!/usr/bin/env python3
"""G5 no-regression gate: text-only traffic through the W1-SERVER llb-serve (before G5) and the G5 llb-serve
must put byte-identical request lines on the engine pipe and return identical HTTP results (status + body,
with id/created/timings masked). Both run over stub_engine.py.
  text_identity.py --old OLD_SERVE --new NEW_SERVE --stub stub_engine.py --out OUTDIR [--port 18541] [--new-args "ARGS"]
--new-args: extra flags for the NEW server only (EARS-CANON: "--ear DIR", with LLB_EAR_BIN set), to prove that
attaching a worker leaves text traffic byte-identical."""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request
ap = argparse.ArgumentParser()
for k in ("old", "new", "stub", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--port", type=int, default=18541)
ap.add_argument("--new-args", default="")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
REQ = [
    ("chat", {"messages": [{"role": "user", "content": "The capital of France is"}], "max_tokens": 16}),
    ("chat", {"messages": [{"role": "system", "content": " sys "}, {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}], "max_tokens": 8}),
    ("chat", {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nr\n</think>\n\nans"}, {"role": "user", "content": "q2"}],
              "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7, "seed": 3, "stop": ["x"], "max_tokens": 8}),
    ("chat", {"messages": [{"role": "user", "content": "café 🚗"}], "add_generation_prompt": False, "return_tokens": True, "max_tokens": 4}),
    ("chat", {"messages": [{"role": "user", "content": "s"}], "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 4}),
    ("chat", {"messages": [{"role": "user", "content": None}, {"role": "user", "content": "x"}], "max_tokens": 4}),
    ("chat", {"messages": [{"role": "tool", "content": "x"}]}),
    ("chat", {"messages": [{"role": "user", "content": "x"}], "tools": []}),
    ("chat", {"messages": [{"role": "user", "content": "x"}], "n": 2}),
    ("chat", {"messages": []}),
    ("chat", {"messages": [{"role": "user", "content": 5}]}),
    ("comp", {"prompt": "Water boils at a temperature of", "max_tokens": 16}),
    ("comp", {"prompt": ["<|im_start|>x"], "max_tokens": 4, "top_p": 0.9}),
    ("comp", {"prompt": "", "max_tokens": 4}),
]
def post(port, kind, body):
    path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
    rq = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=60) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
mask = lambda s: re.sub(r'"(id|created)":("[^"]*"|\d+)', r'"\1":X', re.sub(r'"timings":\{[^}]*\}', '"timings":X', s))
res = {}
for tag, binary, port in (("old", a.old, a.port), ("new", a.new, a.port + 1)):
    log = f"{a.out}/stub-{tag}.jsonl"
    if os.path.exists(log): os.remove(log)
    env = dict(os.environ, LLB_GGUF="/nonexistent/stub.gguf", TMPDIR=a.out, STUB_LOG=log)
    extra = a.new_args.split() if tag == "new" else []
    p = subprocess.Popen([binary, "--port", str(port), "--engine", a.stub, *extra, "--", "stub"], env=env, stderr=open(f"{a.out}/serve-{tag}.log", "w"))
    for _ in range(200):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200: break
        except Exception: time.sleep(0.1)
    res[tag] = [(s, mask(b)) for s, b in (post(port, k, b) for k, b in REQ)]
    p.send_signal(signal.SIGTERM); p.wait(30)
    res[tag + "_lines"] = [json.loads(l)["line_sha256"] for l in open(log)]
same_lines = res["old_lines"] == res["new_lines"]
diff = [i for i, (o, n) in enumerate(zip(res["old"], res["new"])) if o != n]
out = {"requests": len(REQ), "engine_lines": len(res["new_lines"]), "engine_lines_identical": same_lines,
       "http_identical": len(REQ) - len(diff), "http_diffs": [{"i": i, "old": res["old"][i], "new": res["new"][i]} for i in diff],
       "statuses": [s for s, _ in res["new"]]}
json.dump(out, open(f"{a.out}/text_identity.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("requests", "engine_lines", "engine_lines_identical", "http_identical", "statuses")}))
for d in out["http_diffs"]: print("DIFF", d)
# intended: a non-string/array content now gets the template's own error + a param (docs/reference/multimodal-content.md §errors)
INTENDED = {10: (400, "Unexpected content type.")}
unexpected = [i for i in diff if not (i in INTENDED and res["new"][i][0] == INTENDED[i][0] and INTENDED[i][1] in res["new"][i][1])]
out["intended_diffs"], out["unexpected_diffs"] = sorted(set(diff) - set(unexpected)), unexpected
json.dump(out, open(f"{a.out}/text_identity.json", "w"), indent=1)
print(f"engine lines identical: {same_lines}; http identical {len(REQ) - len(diff)}/{len(REQ)}, intended diffs {out['intended_diffs']}, unexpected {unexpected}")
sys.exit(0 if same_lines and not unexpected else 1)
