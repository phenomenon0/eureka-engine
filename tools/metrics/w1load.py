#!/usr/bin/env python3
# w1load.py: W1-METRICS gate driver (client side only; nothing here is on the served path).
#   golden <bin> <out>  : the fixed golden sequence -> out/responses.jsonl (byte-exact lines)
#   load   <bin> <out>  : 20 requests incl. retained hit, continuation, decides, a refusal and a
#                         pipelined burst; scrapes /metrics mid-run and at the end
# Env: W1_ROOT (repository root, for tests/fixtures/goldens), W1_SIDECAR, W1_GGUF, W1_PORT (load only).
import json, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("W1_ROOT", "."))
GOLD = ROOT / "tests/fixtures/goldens"
SIDECAR = os.environ["W1_SIDECAR"]
GGUF = os.environ["W1_GGUF"]
CLEAR = ["LLB_PREFILL", "LLB_SPEC", "LLB_MTP", "LLB_DUMP", "LLB_DUMP_POS", "LLB_DUMP_HIDDEN", "LLB_DUMP_HIDDEN_POS",
         "LLB_DUMP_HIDDEN_TOKENS", "LLB_Q4_DP4A", "LLB_Q4GLUE", "LLB_WARP_GRID", "LLB_SESSION_STRICT", "LLB_HEAD2_GGUF",
         "LLB_POS_PROBE", "LLB_POS_TARGET", "LLB_TIMINGS", "LLB_METRICS_PORT", "LLB_METRICS_HOST"]
P1 = {"mode": "generate", "prompt": "The capital of France is", "n": 16}
P2 = {"mode": "generate", "prompt": "Water boils at a temperature of", "n": 16}
P1_IDS = [11751, 13, 561, 6511, 314, 6124, 369, 25358, 13, 561, 6511, 314]  # prefix (session.py P1_IDS[:12])


def gold(name):
    return json.loads((GOLD / f"{name}.request.json").read_text())


def first_gold(prefix):
    return next(p.name[: -len(".request.json")] for p in sorted(GOLD.glob(f"{prefix}*.request.json")))


class Engine:
    def __init__(self, binary, out, metrics_port=None):
        env = {k: v for k, v in os.environ.items() if k not in CLEAR}
        tmp = out / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update(LLB_SYS1="1", LLB_GGUF=GGUF, TMPDIR=str(tmp))
        if metrics_port:
            env["LLB_METRICS_PORT"] = str(metrics_port)
        self.err = open(out / "engine.stderr", "wb")
        argv = [binary, "--threads", "16", "--", SIDECAR, "0", "--steps", "4096"]
        (out / "argv.json").write_text(json.dumps({"argv": argv, "metrics_port": metrics_port}, indent=1))
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, env=env)

    def send(self, reqs):
        data = b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in reqs)
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("engine closed stdout")
        return line

    def ask(self, req):
        t = time.monotonic()
        self.send([req])
        line = self.recv()
        return line, (time.monotonic() - t) * 1000

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait(timeout=120)
        self.err.close()
        return rc


def golden_seq():
    d1 = first_gold("01-")
    return [("P1", P1), ("P2", P2), (d1, gold(d1)), ("P1", P1),
            ("P1-s", dict(P1, session="g1")), ("P1-s-repeat", dict(P1, session="g1")),
            ("P2", P2)]


def run_golden(binary, out):
    e = Engine(binary, out)
    rows = []
    with open(out / "responses.jsonl", "wb") as f:
        for name, req in golden_seq():
            line, ms = e.ask(req)
            f.write(line)
            rows.append({"name": name, "client_ms": round(ms, 3)})
    rc = e.close()
    (out / "rows.json").write_text(json.dumps({"rc": rc, "rows": rows}, indent=1))
    print("golden rc", rc, "rows", len(rows))


def scrape(port, path="/metrics"):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def run_load(binary, out):
    port = int(os.environ.get("W1_PORT", "9431"))
    e = Engine(binary, out, port)
    prompts = ["The capital of France is", "Water boils at a temperature of", "The three primary colors are",
               "In 1969, the first person to walk on the moon was", "A haiku about autumn:\n",
               "def fibonacci(n):", "The speed of light in vacuum is approximately",
               "Photosynthesis converts sunlight into", "The largest planet in the solar system is",
               "Explain RAID 5 in one sentence:"]
    d1, d2 = first_gold("01-"), first_gold("02-")
    reqs = []
    for i, p in enumerate(prompts):                              # 10 fresh generates, n 8..32
        reqs.append((f"gen{i}", {"mode": "generate", "prompt": p, "n": 8 + (i % 4) * 8, **({"timings": True} if i % 2 == 0 else {})}))
    reqs.append(("decide-" + d1, gold(d1)))                      # 11: decide
    reqs.append(("decide-" + d2, dict(gold(d2), timings=True)))  # 12: decide + timings
    s = {"mode": "generate", "prompt": "The capital of France is", "n": 8, "session": "agent", "timings": True}
    reqs.append(("sess-fresh", s))                               # 13: session miss
    reqs.append(("sess-repeat", s))                              # 14: retained full hit
    reqs.append(("refused", {"mode": "generate", "prompt": "x"}))  # 15: refused (no n)
    rows, scrapes = [], {}
    t0 = time.monotonic()
    for name, req in reqs:
        line, ms = e.ask(req)
        rows.append({"name": name, "client_ms": round(ms, 3), "response": json.loads(line)})
        if name == "decide-" + d1:
            scrapes["mid"] = scrape(port)
    # 16: continuation of the "agent" session: previous prompt + its generated text + more
    prev = rows[-3]["response"]
    cont = {"mode": "generate", "prompt": s["prompt"] + prev["text"] + " And the capital of Italy is", "n": 8,
            "session": "agent", "timings": True}
    line, ms = e.ask(cont)
    rows.append({"name": "sess-continue", "client_ms": round(ms, 3), "response": json.loads(line)})
    # 17-20: pipelined burst (4 requests written at once) -> queue depth > 0 while it drains
    burst = [{"mode": "generate", "prompt": p, "n": 24, "timings": True} for p in prompts[:4]]
    tb = time.monotonic()
    e.send(burst)
    got = []
    burst_scrape = []

    def poll():
        while len(got) < len(burst):
            try:
                txt = scrape(port)[2]
                q = [l for l in txt.splitlines() if l.startswith(("llb_queue_depth ", "llb_requests_in_flight "))]
                burst_scrape.append({"t_ms": round((time.monotonic() - tb) * 1000, 1), "lines": q})
            except Exception as x:  # noqa: BLE001
                burst_scrape.append({"error": str(x)})
            time.sleep(0.05)

    th = threading.Thread(target=poll)
    th.start()
    for i in range(len(burst)):
        got.append(e.recv())
        rows.append({"name": f"burst{i}", "client_ms_since_burst": round((time.monotonic() - tb) * 1000, 3),
                     "response": json.loads(got[-1])})
    th.join()
    wall = time.monotonic() - t0
    scrapes["final"] = scrape(port)
    health = scrape(port, "/health")
    try:
        scrape(port, "/nope")
        nf = "200?"
    except urllib.error.HTTPError as x:
        nf = x.code
    rc = e.close()
    (out / "metrics-mid.txt").write_text(scrapes["mid"][2])
    (out / "metrics-final.txt").write_text(scrapes["final"][2])
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"rc": rc, "requests": len(rows), "wall_s": round(wall, 3),
               "content_type": scrapes["final"][1], "health": health[:3:2], "not_found_status": nf,
               "burst_queue_samples": burst_scrape}
    (out / "load-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "burst_queue_samples"}))


if __name__ == "__main__":
    mode, binary, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    {"golden": run_golden, "load": run_load}[mode](binary, out)
