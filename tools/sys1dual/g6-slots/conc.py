#!/usr/bin/python3 -I
"""G6 receipt driver: parallel (N-slot server) vs serial outputs, byte-identical.

  conc.py pairs  --serial URL --par URL --seeds 1-5 --out DIR [--long]
  conc.py fifo   --url URL --k 6 --out DIR
Each seed builds a workload of request pairs (prompt, chat/completion, stream,
max_tokens, start offset, and the request's own "seed" field). Serial: every
request alone, one after another. Parallel: the two of a pair at once. Compared:
generated ids, prompt ids, text; streamed pieces concatenate to the non-stream
text. Concurrency evidence: per pair, B's first token arrives before A's last.
"""
import argparse, json, random, sys, threading, time, urllib.request

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "Once upon a time",
    "List three prime numbers:",
    "def fibonacci(n):",
    "The quick brown fox",
    "In 1969, astronauts",
    "Translate to German: good morning",
]
CHATS = [
    "What is the capital of France?",
    "Write the word café and the emoji 🙂 three times.",
    "Name two rivers in Africa.",
    "Explain recursion in one sentence.",
]


def call(url, body):
    """POST; returns dict(text, ids, pids, t0, t_first, t_end, status)."""
    stream = body.get("stream", False)
    data = json.dumps(body).encode()
    rq = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    r = {"t0": t0, "t_first": None}
    with urllib.request.urlopen(rq, timeout=900) as f:
        r["status"] = f.status
        h = f.headers
        r["slot"] = int(h["X-LLB-Slot"]) if h.get("X-LLB-Slot") else None
        r["ticket"] = int(h["X-LLB-Ticket"]) if h.get("X-LLB-Ticket") else None
        r["run_start"] = float(h["X-LLB-Run-Start"]) if h.get("X-LLB-Run-Start") else None
        if not stream:
            j = json.loads(f.read())
            ch = j["choices"][0]
            r["text"] = ch["message"]["content"] if "message" in ch else ch["text"]
            r["ids"] = j.get("llb", {}).get("ids")
            r["pids"] = j.get("llb", {}).get("prompt_ids")
            r["usage"] = j["usage"]
            r["timings"] = j["timings"]
            r["t_first"] = time.monotonic()
        else:
            parts = []
            for raw in f:
                line = raw.decode().rstrip("\n")
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if "error" in j:
                    raise RuntimeError(d)
                for ch in j.get("choices", []):
                    piece = ch.get("delta", {}).get("content") if "delta" in ch else ch.get("text")
                    if piece:
                        if r["t_first"] is None:
                            r["t_first"] = time.monotonic()
                        parts.append(piece)
                if "usage" in j and j.get("choices"):
                    r["usage"] = j["usage"]
                if "timings" in j:
                    r["timings"] = j["timings"]
            r["text"] = "".join(parts)
            r["ids"] = r["pids"] = None
    r["t_end"] = time.monotonic()
    # the server-measured engine interval (same CLOCK_MONOTONIC as the server):
    # run start (slot acquired) .. + prompt_ms + predicted_ms
    if r["run_start"] is not None:
        tm = r["timings"]
        r["run_end"] = r["run_start"] + (tm["prompt_ms"] + tm["predicted_ms"]) / 1000
    return r


def workload(seed, long):
    rng = random.Random(seed)
    pairs = []
    for _ in range(4):
        pair = []
        for _ in range(2):
            chat = rng.random() < 0.4
            n = rng.choice([48, 64, 96] if long else [16, 24, 32])
            b = {"max_tokens": n, "seed": rng.randrange(1 << 31), "return_tokens": True}
            if chat:
                b["messages"] = [{"role": "user", "content": rng.choice(CHATS)}]
                b["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                b["prompt"] = rng.choice(PROMPTS)
            pair.append({"chat": chat, "body": b, "stream": rng.random() < 0.5})
        pairs.append({"pair": pair, "offset": rng.choice([0.0, 0.0, 0.05, 0.2])})
    return pairs


def url_for(base, chat):
    return base + ("/v1/chat/completions" if chat else "/v1/completions")


def one(base, item, stream):
    b = dict(item["body"])
    if stream:
        b["stream"] = True
        b.pop("return_tokens", None)
    return call(url_for(base, item["chat"]), b)


def pairs_cmd(a):
    lo, hi = map(int, a.seeds.split("-"))
    out = open(f"{a.out}/pairs.jsonl", "w")
    fails = overl = npairs = 0
    ser_wall = par_wall = 0.0
    for seed in range(lo, hi + 1):
        for pi, P in enumerate(workload(seed, a.long)):
            A, B = P["pair"]
            # serial reference: non-stream (ids) for both, then stream for the streamed ones
            ref = [one(a.serial, x, False) for x in (A, B)]
            ref_s = [one(a.serial, x, True) if x["stream"] else None for x in (A, B)]
            ser_wall += sum(r["t_end"] - r["t0"] for r in ref)
            # parallel: both at once (B after the offset), each in its own mode
            res = [None, None]
            def go(i, x, delay):
                time.sleep(delay)
                res[i] = one(a.par, x, x["stream"])
            th = [threading.Thread(target=go, args=(0, A, 0.0)), threading.Thread(target=go, args=(1, B, P["offset"]))]
            t = time.monotonic()
            [x.start() for x in th]
            [x.join() for x in th]
            par_wall += time.monotonic() - t
            ok = True
            why = []
            for i, x in enumerate((A, B)):
                r, rf = res[i], ref[i]
                if r["text"] != rf["text"]:
                    ok = False; why.append(f"{'AB'[i]} text")
                if not x["stream"] and (r["ids"] != rf["ids"] or r["pids"] != rf["pids"]):
                    ok = False; why.append(f"{'AB'[i]} ids")
                if x["stream"] and ref_s[i]["text"] != rf["text"]:
                    ok = False; why.append(f"{'AB'[i]} serial stream concat")
            # concurrency (server truth): two different slots, engine intervals intersect
            overlap = (res[0]["slot"] is not None and res[0]["slot"] != res[1]["slot"]
                       and max(r["run_start"] for r in res) < min(r["run_end"] for r in res))
            overl += overlap
            npairs += 1
            fails += not ok
            out.write(json.dumps({"seed": seed, "pair": pi, "ok": ok, "why": why, "overlap": overlap,
                "offset": P["offset"], "req": [x["body"] | {"stream": x["stream"]} for x in (A, B)],
                "par": [{k: r[k] for k in ("text", "ids", "slot", "ticket")} | {"ms": round((r["t_end"] - r["t0"]) * 1000, 1),
                         "ttft_ms": round((r["t_first"] - r["t0"]) * 1000, 1)} for r in res],
                "serial_ms": [round((r["t_end"] - r["t0"]) * 1000, 1) for r in ref]}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"seed {seed} pair {pi}: {'OK' if ok else 'FAIL ' + ','.join(why)} overlap={overlap}", flush=True)
    v = {"pairs": npairs, "identical": npairs - fails, "overlapping": overl,
         "serial_wall_s": round(ser_wall, 3), "parallel_wall_s": round(par_wall, 3),
         "expect": "serial" if a.expect_serial else "overlap"}
    v["verdict"] = "PASS" if fails == 0 and overl == (0 if a.expect_serial else npairs) else "FAIL"
    json.dump(v, open(f"{a.out}/pairs-verdict.json", "w"), indent=1)
    print(json.dumps(v))
    return v["verdict"] == "PASS"


def serial_cmd(a):
    """Every request of pairs.jsonl alone (non-stream) on another server (the W1
    single-slot binary): text + ids must equal the parallel results there."""
    n = bad = 0
    out = open(f"{a.out}/serial-{a.tag}.jsonl", "w")
    for l in open(f"{a.out}/pairs.jsonl"):
        d = json.loads(l)
        for i, b in enumerate(d["req"]):
            b = {k: v for k, v in b.items() if k != "stream"}
            r = call(url_for(a.url, "messages" in b), b)
            par = d["par"][i]
            ok = r["text"] == par["text"] and (par["ids"] is None or r["ids"] == par["ids"])
            n += 1; bad += not ok
            out.write(json.dumps({"seed": d["seed"], "pair": d["pair"], "i": i, "ok": ok, "ids": r["ids"]}) + "\n")
    v = {"requests": n, "identical": n - bad, "verdict": "PASS" if bad == 0 else "FAIL"}
    json.dump(v, open(f"{a.out}/serial-{a.tag}-verdict.json", "w"))
    print(json.dumps(v))
    return bad == 0


def fifo_cmd(a):
    """K stream requests staggered 40 ms apart: they must START (first token)
    in arrival order, and with N slots at most N run at once."""
    res = [None] * a.k
    def go(i):
        time.sleep(0.15 * i)
        res[i] = call(a.url + "/v1/completions",
                      {"prompt": PROMPTS[i % len(PROMPTS)], "max_tokens": a.n, "stream": True})
    th = [threading.Thread(target=go, args=(i,)) for i in range(a.k)]
    [x.start() for x in th]
    [x.join() for x in th]
    # service order = engine start order (server clock); FIFO = it follows the ticket order
    z = min(r["run_start"] for r in res)
    rows = [{"i": i, "ticket": res[i]["ticket"], "slot": res[i]["slot"], "t0_client": round(res[i]["t0"] - res[0]["t0"], 3),
             "first": round(res[i]["run_start"] - z, 3), "end": round(res[i]["run_end"] - z, 3)} for i in range(a.k)]
    by_ticket = sorted(range(a.k), key=lambda i: res[i]["ticket"])
    order = sorted(range(a.k), key=lambda i: res[i]["run_start"])
    # max concurrently running (first token .. end)
    ev = sorted([(r["first"], 1) for r in rows] + [(r["end"], -1) for r in rows], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d; peak = max(peak, cur)
    v = {"k": a.k, "ticket_order": by_ticket, "start_order": order, "fifo": order == by_ticket, "peak_running": peak,
         "slots": a.slots, "rows": rows}
    v["verdict"] = "PASS" if v["fifo"] and peak == a.slots else "FAIL"
    json.dump(v, open(f"{a.out}/fifo-{a.slots}.json", "w"), indent=1)
    print(json.dumps({k: v[k] for k in ("ticket_order", "start_order", "fifo", "peak_running", "verdict")}))
    return v["verdict"] == "PASS"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--serial"); ap.add_argument("--par"); ap.add_argument("--url")
    ap.add_argument("--seeds", default="1-5"); ap.add_argument("--out", required=True)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--tag", default="w1")
    ap.add_argument("--expect-serial", action="store_true", help="N=1 server: identity only, no overlap expected")
    a = ap.parse_args()
    sys.exit(0 if {"pairs": pairs_cmd, "fifo": fifo_cmd, "serial": serial_cmd}[a.cmd](a) else 1)
