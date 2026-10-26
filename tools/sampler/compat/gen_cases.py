#!/usr/bin/env python3
"""Generate gate-1 cases: NNN.f32 (logits), NNN.args (ref_chain / gate1 key=val line), NNN.name.
Real engine logits were NOT used: random + structure + pathologicals."""
import numpy as np, sys, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
V = 248320
rs = np.random.RandomState(20261002)

def gauss(scale):  return (rs.randn(V) * scale).astype(np.float32)
def neartie(scale):
    x = (rs.randn(V) * scale).astype(np.float32)
    top = rs.choice(V, 60, replace=False)
    x[top] = np.float32(scale * 6) + (rs.randn(60) * 1e-4).astype(np.float32)  # near-ties, distinct values
    return x
def flat(scale):
    x = (-10 + rs.randn(V) * 1e-3).astype(np.float32)               # long flat tail
    hi = rs.choice(V, 40, replace=False)
    x[hi] = (rs.randn(40) * scale + 4).astype(np.float32)
    return x
inputs = []   # (name, V, array, ties_flag)
for sc in (0.5, 1.5, 4.0, 10.0):
    inputs += [(f"gauss{sc}", V, gauss(sc), 0), (f"neartie{sc}", V, neartie(sc), 0), (f"flat{sc}", V, flat(sc), 0)]
for i in range(8): inputs.append((f"gaussR{i}", V, gauss(float(rs.choice([1, 2, 3, 6]))), 0))
# pathologicals
inputs.append(("allequal", V, np.full(V, 1.25, np.float32), 1))
huge = gauss(1.0); huge[777] = 1e4; inputs.append(("onehuge", V, huge, 0))
ninf = gauss(2.0); ninf[rs.rand(V) < 0.3] = -np.inf; inputs.append(("neginf", V, ninf, 0))
inputs.append(("V1", 1, np.array([0.5], np.float32), 0))
two = np.full(V, -20, np.float32); two[10] = 3; two[20] = 3; inputs.append(("twoequal", V, two, 1))
half = gauss(1.0); half[:] = -30; half[5] = 0; half[9] = 0; inputs.append(("twoequal2", V, half, 1))

def hist_for(x, n):
    top = np.argsort(-x)[:100]
    h = np.where(rs.rand(n) < 0.7, rs.choice(top, n), rs.randint(0, len(x), n))
    return ",".join(map(str, h))
def bias_for(x, k):
    top = np.argsort(-x)[:30]
    ids = list(rs.choice(top, k // 2)) + list(rs.randint(0, len(x), k - k // 2)) + [int(top[0])]  # duplicate id on purpose
    return ",".join(f"{i}:{b:.3f}" for i, b in zip(ids, rs.choice([-3.0, 2.5, -100.0, 1.0, 5.0], len(ids))))

PS = [  # stage-alone and combined sets; {H} = history, {B} = bias placeholders
 "top_k=20", "top_k=400", "top_p=0.9", "top_p=0.95 min_keep=3", "min_p=0.05", "min_p=0.3 min_keep=2",
 "temp=0.7", "temp=0", "temp=2.5",
 "pen=64,1.3,0,0 hist={H}", "pen=512,1.0,0.4,0.9 hist={H}", "pen=64,1.1,0.2,0 hist={H}",
 "bias={B}",
 "top_k=20 top_p=0.95 temp=1.0 pen=64,1.0,0,1.5 hist={H}",                      # qwen thinking
 "top_k=20 top_p=0.8 temp=0.7 pen=64,1.0,0,1.5 hist={H}",                       # qwen non-thinking
 "top_k=40 top_p=0.95 min_p=0.05 temp=0.8",                                     # llama.cpp defaults
 "top_k=0 top_p=0.9 min_p=0.02 temp=1.3 pen=128,1.15,0.1,0.3 hist={H} bias={B}",  # everything on, top_k off
 "top_p=0.0001 min_keep=4",
]
cases = []
for ii, (nm, v, x, ties) in enumerate(inputs):
    if nm.startswith("gauss") or nm.startswith("neartie") or nm.startswith("flat"):
        picks = [(ii * 3 + j) % len(PS) for j in range(3)] + [13 + (ii % 3)]     # 3 rotating + a qwen/llama chain
    else:
        picks = list(range(len(PS))) if nm != "V1" else [0, 2, 4, 6, 7, 13, 15]
    for pi in sorted(set(picks)): cases.append((nm, v, x, ties, PS[pi]))
for n, (nm, v, x, ties, ps) in enumerate(cases):
    if x is None: continue
    x.astype(np.float32).tofile(f"{out}/{n:03d}.f32")
    h = hist_for(x, 300 if v > 1 else 5) if v > 1 else "0,0,0"
    b = bias_for(x, 6) if v > 1 else "0:1.0"
    line = ps.replace("{H}", h).replace("{B}", b)
    open(f"{out}/{n:03d}.args", "w").write(line + "\n")
    open(f"{out}/{n:03d}.name", "w").write(f"{nm}|{ps.split(' hist')[0].split(' bias')[0]}|ties={ties}\n")
# exact-cutoff cases: equal logits on tiny V, so cumulative p hits top_p exactly (0.25 steps are exact in float)
for nm, v, ps in [("exact4", 4, "top_p=0.5"), ("exact4", 4, "top_p=0.75"), ("exact4", 4, "top_p=0.5 min_keep=3"),
                  ("exact4", 4, "top_k=3 top_p=0.5"), ("exact8", 8, "top_p=0.5"), ("exact8", 8, "top_p=0.875 min_p=0.5")]:
    n = len(cases)
    np.zeros(v, np.float32).tofile(f"{out}/{n:03d}.f32")
    open(f"{out}/{n:03d}.args", "w").write(ps + "\n")
    open(f"{out}/{n:03d}.name", "w").write(f"{nm}|{ps}|ties=1\n")
    cases.append((nm, v, None, 1, ps))
print(len(cases), "cases")
