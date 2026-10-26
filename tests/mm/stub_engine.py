#!/usr/bin/env python3
"""G5 stub engine: speaks dual.c's stdin/stdout JSON-line protocol (generate only) and checks the
engine media hook from docs/reference/multimodal-content.md instead of running a model. Per request it verifies every
media entry (file exists, size and sha256 match, kind/mime/dims present) and that the prompt holds
exactly one <|vision_start|><|image_pad|><|vision_end|> per image, in order; then it expands each
pad to K rows (the fused-trunk connector's K=64) to report a realistic prompt length.
Its reply text is a JSON verdict, so the HTTP harness can read what the engine saw.
Every request is also appended to $STUB_LOG (one JSON line)."""
import hashlib, json, os, sys

K = int(os.environ.get("STUB_ROWS", "64"))
TAG, PAD = "<|vision_start|><|image_pad|><|vision_end|>", "<|image_pad|>"
stream = os.environ.get("LLB_SYS1_STREAM") == "1"
log = open(os.environ["STUB_LOG"], "a") if os.environ.get("STUB_LOG") else None
print("sys1: FP32 clef head + tokenizer ready (stub)", file=sys.stderr, flush=True)
for line in sys.stdin:
    req = json.loads(line)
    media, prompt = req.get("media", []), req["prompt"]
    v = {"ok": True, "media": [], "problems": []}
    for i, m in enumerate(media):
        p = m.get("path", "")
        e = {"kind": m.get("kind"), "mime": m.get("mime"), "w": m.get("w"), "h": m.get("h"), "detail": m.get("detail")}
        try:
            b = open(p, "rb").read()
            e["sha_ok"] = hashlib.sha256(b).hexdigest() == m.get("sha256") and len(b) == m.get("bytes")
            e["mode"] = oct(os.stat(p).st_mode & 0o777)
        except OSError as ex:
            e["sha_ok"] = False; v["problems"].append(f"media[{i}] unreadable: {ex}")
        if not e["sha_ok"]: v["ok"] = False
        elif os.environ.get("STUB_KEEP"):  # keep a copy of the staged bytes (the server unlinks its file)
            open(os.path.join(os.environ["STUB_KEEP"], m["sha256"] + "." + m["mime"].split("/")[1]), "wb").write(b)
        e["sha256"] = m.get("sha256"); v["media"].append(e)
    n_img = sum(1 for m in media if m.get("kind") == "image")
    if prompt.count(TAG) != n_img or prompt.count(PAD) != n_img:
        v["ok"] = False; v["problems"].append(f"{prompt.count(TAG)} image tags for {n_img} images")
    expanded = prompt.replace(PAD, PAD * K)
    v["special"] = req.get("special", False)
    v["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    text = json.dumps(v, separators=(",", ":"))
    pieces = [text[i:i + 40] for i in range(0, len(text), 40)][: max(1, req.get("n", 16))]
    if len(pieces) < len(text) / 40: pieces[-1] = text[40 * (len(pieces) - 1):]  # keep the verdict whole
    if log: log.write(json.dumps({"line_sha256": hashlib.sha256(line.encode()).hexdigest(), "line": line if len(line) < 4096 else None,
                                  "req": {k: req[k] for k in req if k != "prompt"}, "prompt": prompt, "verdict": v}) + "\n"); log.flush()
    if stream:
        for i, pc in enumerate(pieces): print(json.dumps({"token": i, "text": pc}), flush=True)
    n_prompt = len(expanded.split(PAD)) - 1 + len(expanded.replace(PAD, "")) // 4  # rows + ~chars/4
    print(json.dumps({"prompt_ids": list(range(n_prompt)), "ids": list(range(len(pieces))), "text": "".join(pieces)}), flush=True)
