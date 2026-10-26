#!/usr/bin/env python3
"""presets.json == compiled defaults; top-n logprobs == numpy log_softmax (independent)."""
import json, subprocess, sys, numpy as np
out = sys.argv[1]; here = __file__.rsplit('/', 1)[0]
J = json.load(open(f"{here}/presets.json"))["defaults"]
rows = [l.split() for l in subprocess.check_output([f"{out}/dump_misc", "presets"], text=True).splitlines()]
bad = 0
for (m, t, tp, tk, mp, pres, rep), key in zip(rows, ["greedy", "qwen_thinking", "qwen_non_thinking"]):
    j = J[key]; got = (float(t), float(tp), int(tk), float(mp), float(pres), float(rep))
    exp = (j["temperature"], j["top_p"], j["top_k"], j["min_p"], j["presence_penalty"], j["repetition_penalty"])
    ok = all(abs(a - b) < 1e-6 for a, b in zip(got, exp)); bad += not ok
    print(f"preset mode {m} ({key}): {'ok' if ok else 'MISMATCH'} {got}")
worst = 0
for c in ["000", "004", "012", "030", "060"]:
    x = np.fromfile(f"{out}/cases/{c}.f32", dtype=np.float32).astype(np.float64)
    if len(x) < 10: continue
    ls = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
    r = [l.split() for l in subprocess.check_output([f"{out}/dump_misc", "lp", f"{out}/cases/{c}.f32", "5"], text=True).splitlines()]
    top = np.argsort(-x, kind="stable")[:5]
    ok = [int(a) for a, _ in r] == top.tolist() and all(abs(float(b) - ls[int(a)]) < 1e-5 for a, b in r)
    worst = max(worst, max(abs(float(b) - ls[int(a)]) for a, b in r)); bad += not ok
    print(f"logprobs case {c}: {'ok' if ok else 'MISMATCH'}")
print(f"worst |lp - numpy| = {worst:.3g}; misc: {'FAIL' if bad else 'PASS'}")
sys.exit(1 if bad else 0)
