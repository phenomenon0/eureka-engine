#!/usr/bin/python3 -I
# G6 mock engine: the dual.c stdin/stdout protocol (LLB_SYS1_STREAM events) with
# deterministic ids = f(prompt bytes, step). Per-token sleep makes overlap visible.
# Not the model: it exercises the server's slot/queue/stream plumbing only.
import json, os, sys, time
tok_s = float(os.environ.get("MOCK_TOKEN_S", "0.02"))
time.sleep(float(os.environ.get("MOCK_LOAD_S", "0.3")))
print("sys1: FP32 clef head + tokenizer ready 0.0 ms", file=sys.stderr, flush=True)
seq = 0
for line in sys.stdin:
    seq += 1
    req = json.loads(line)
    p = req["prompt"].encode()
    n = int(req["n"])
    if n > 4000:
        print(json.dumps({"error": "prompt + n exceeds the KV capacity fixed at launch"}), flush=True); continue
    h = 2166136261
    for b in p: h = ((h ^ b) * 16777619) & 0xffffffff
    ids, text = [], ""
    for i in range(n):
        h = (h * 1103515245 + 12345) & 0x7fffffff
        t = h % 50000
        piece = ["a", "b ", "é", "🙂", "x\n"][t % 5] + str(t % 97)
        ids.append(t); text += piece
        time.sleep(tok_s)
        print(json.dumps({"token": t, "text": piece}, ensure_ascii=False), flush=True)
    print(json.dumps({"prompt_ids": list(p), "ids": ids, "text": text}, ensure_ascii=False), flush=True)
    print(json.dumps({"seq": seq, "mode": "generate", "pid": os.getpid(), "n": n}), file=sys.stderr, flush=True)
