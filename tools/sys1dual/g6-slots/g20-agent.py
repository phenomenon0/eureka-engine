#!/usr/bin/python3 -I
"""G20 agent-loop receipt: 2 clients, each its own long system prompt, T turns each.
  agent.py conc  --url U --out DIR --tag TAG   (both clients at once; each sends its next turn when the last returns)
  agent.py alone --url U --out DIR --tag TAG   (every request alone, A1 B1 A2 B2 ...)
Writes DIR/agent-TAG.jsonl: client, turn, ids, pids, text, prompt_ms, predicted_ms, slot, t_end (engine order)."""
import argparse, json, sys, threading, time
sys.path.insert(0, "/tmp/g20/tools")
from conc import call

TOPIC = {"A": ("a ship's navigator", "tides, stars, charts, knots, the compass, the sextant, dead reckoning"),
         "B": ("a pastry chef", "butter, flour, lamination, proofing, ovens, sugar work, tempering chocolate")}
QS = {"A": ["What do you check first at dawn?", "How do you fix a position without GPS?", "Name one danger near reefs."],
      "B": ["What makes a croissant flaky?", "Why rest the dough?", "Name one mistake with caramel."]}

def system(c):
    who, words = TOPIC[c]
    rule = (f"You are {who}. Answer briefly and stay in character. Your craft covers {words}. ")
    body = "".join(f"Rule {i}: when asked about {words.split(', ')[i % 7]}, give one concrete fact, never more than two sentences, "
                   f"and mention the number {i * 7 + (3 if c == 'A' else 5)} only if it is relevant. " for i in range(40))
    return rule + body

def req(c, t, n):
    return {"messages": [{"role": "system", "content": system(c)}, {"role": "user", "content": QS[c][t]}],
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": n, "return_tokens": True}

def go(url, c, t, n):
    r = call(url + "/v1/chat/completions", req(c, t, n))
    return {"client": c, "turn": t, "ids": r["ids"], "pids": r["pids"], "text": r["text"], "slot": r["slot"],
            "prompt_ms": r["timings"]["prompt_ms"], "predicted_ms": r["timings"]["predicted_ms"],
            "prompt_n": r["usage"]["prompt_tokens"], "t0": r["t0"], "t_end": r["t_end"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--turns", type=int, default=3); ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    res = []
    if a.cmd == "alone":
        for t in range(a.turns):
            for c in "AB":
                res.append(go(a.url, c, t, a.n))
    else:
        lock = threading.Lock()
        def client(c, delay):
            time.sleep(delay)
            for t in range(a.turns):
                r = go(a.url, c, t, a.n)
                with lock:
                    res.append(r)
        th = [threading.Thread(target=client, args=(c, d)) for c, d in (("A", 0.0), ("B", 0.05))]
        [x.start() for x in th]; [x.join() for x in th]
    res.sort(key=lambda r: r["t_end"])
    with open(f"{a.out}/agent-{a.tag}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    for r in res:
        print(f"{a.tag} {r['client']}{r['turn']} slot={r['slot']} prompt_n={r['prompt_n']} prompt_ms={r['prompt_ms']:.0f} text={r['text']!r}", flush=True)

main()
