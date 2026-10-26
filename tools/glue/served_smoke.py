#!/usr/bin/env python3
"""served.py URL GA2_WORK OUTDIR: G4 F1 served-path requests (llb-serve -> SYS1 engine, Base trunk, NOHEAD).
Writes every response to OUTDIR/<tag>.json; one summary line per request on stdout."""
import json, sys, urllib.request, urllib.error
U, W, O = sys.argv[1:4]
PAD = "<|image_pad|>" * 64
Q = {0: " Question: Is there a car in the image? Answer:", 1: " Question: Is there a tv in the image? Answer:"}
def post(tag, body, path="/v1/completions"):
    req = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r: st, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e: st, raw = e.code, e.read().decode()
    open(f"{O}/{tag}.json", "w").write(raw)
    print(f"{tag} status={st} {raw[:400] if not body.get('stream') else raw[-300:]}"); sys.stdout.flush()
    return st, raw
img = lambda n, k=64: {"rows": f"{W}/rows{n}/rows.f32", "n": k}
post("text-P0-before", {"prompt": "The capital of France is", "max_tokens": 16, "return_tokens": True})
for n in (0, 1):
    post(f"img-row{n}", {"prompt": PAD + Q[n], "max_tokens": 1, "return_tokens": True,
                          "llb_image": img(n), "llb_readout": [9542, 874]})
post("img-row0-gen8", {"prompt": PAD + Q[0], "max_tokens": 8, "return_tokens": True, "llb_image": img(0)})
post("img-row0-stream", {"prompt": PAD + Q[0], "max_tokens": 1, "stream": True, "llb_image": img(0)})
post("text-P0-after", {"prompt": "The capital of France is", "max_tokens": 16, "return_tokens": True})
post("text-row0-noimg-readout", {"prompt": Q[0], "max_tokens": 1, "return_tokens": True, "llb_readout": [9542, 874]})
# refusals (each must be a 400 from the engine's validation; the engine must survive)
post("ref-n-mismatch", {"prompt": PAD + Q[0], "max_tokens": 1, "llb_image": img(0, 63)})
post("ref-no-pads", {"prompt": Q[0], "max_tokens": 1, "llb_image": img(0)})
post("ref-split-run", {"prompt": PAD[:13*32] + " x" + PAD[13*32:] + Q[0], "max_tokens": 1, "llb_image": img(0)})
post("ref-bad-readout", {"prompt": Q[0], "max_tokens": 1, "llb_readout": [248320]})
post("ref-missing-file", {"prompt": PAD + Q[0], "max_tokens": 1, "llb_image": {"rows": "/nonexistent.f32", "n": 64}})
post("text-P0-final", {"prompt": "The capital of France is", "max_tokens": 16, "return_tokens": True})
