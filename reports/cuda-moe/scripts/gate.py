# gate.py DIR [CTAG] : the PREREG gates for one file (R, F, C arms in DIR); CTAG F scores the floor as the candidate
import json, sys, math, random, re
d = sys.argv[1]; ct = sys.argv[2] if len(sys.argv) > 2 else "C"
plen = {1: 5, 2: 8, 3: 4}
rows = []  # per position: dict
greedy = []
for i in (1, 2, 3):
    R = json.load(open(f"{d}/R-p{i}.json")); F = json.load(open(f"{d}/F-p{i}.json"))["top"]
    n0 = plen[i]
    if ct == "F":
        C = {}
        for k in range(64):
            t = F[n0 - 1 + k]; C[k] = (t[0][0], t[0][1], t[1][0], t[1][1])
        cg = None
    else:
        C = {}
        for ln in open(f"{d}/{ct}-tf-p{i}.txt.stderr"):
            m = re.match(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
            if m:
                pos, a, va, b, vb = map(int, m.groups())
                f = lambda u: __import__("struct").unpack("<f", __import__("struct").pack("<I", u))[0]
                if n0 - 1 <= pos < n0 - 1 + 64: C[pos - n0 + 1] = (a, f(va), b, f(vb))
        cg = [int(x) for x in open(f"{d}/{ct}-greedy-p{i}.txt.stdout").read().split() if x.isdigit()][:64]
    assert len(C) == 64, (i, len(C))
    rg = R["gen"]
    if cg is not None:
        k = next((j for j in range(min(len(cg), 64)) if cg[j] != rg[j]), 64)
        greedy.append((i, k, rg, cg))
    for k in range(64):
        rt = R["top"][k]; ft = F[n0 - 1 + k]
        r1, r2 = rt[0][0], rt[1][0]; mr = rt[0][1] - rt[1][1]
        fd = {a: v for a, v in ft}
        mf = fd[r1] - fd[r2] if r1 in fd and r2 in fd else None
        rows.append(dict(p=i, k=k, r1=r1, r3=[x[0] for x in rt[:3]], rtop=rt, mr=mr, mf=mf, c1=C[k][0], cm=C[k][1] - C[k][3], f1=ft[0][0]))
diffs = [r["mf"] - r["mr"] for r in rows if r["mf"] is not None]
m = max(0.01, 2 * math.sqrt(sum(x * x for x in diffs) / len(diffs)))
dis = [r for r in rows if r["c1"] != r["r1"]]
near = lambda r: r["mr"] < m and r["c1"] in r["r3"]
ga = all(near(r) for r in dis)
aC = [int(r["c1"] == r["r1"]) for r in rows]; aF = [int(r["f1"] == r["r1"]) for r in rows]
D = 100 * (sum(aC) - sum(aF)) / len(rows)
rnd = random.Random(0); bs = []
n = len(rows)
for _ in range(10000):
    idx = [rnd.randrange(n) for _ in range(n)]
    bs.append(100 * sum(aC[j] - aF[j] for j in idx) / n)
bs.sort(); lo, hi = bs[int(0.05 * 10000)], bs[int(0.95 * 10000) - 1]
hC = [r["c1"] != r["r1"] and r["mr"] >= m for r in rows]; hF = [r["f1"] != r["r1"] and r["mr"] >= m for r in rows]
b = sum(1 for x, y in zip(hC, hF) if x and not y); c = sum(1 for x, y in zip(hC, hF) if y and not x)
pv = sum(math.comb(b + c, j) for j in range(b, b + c + 1)) / 2 ** (b + c) if b + c else 1.0
vt1 = lo >= -2.0; vflip_fail = b + c > 0 and pv < 0.0125
verdict = "PASS" if ga else ("PASS (v2)" if vt1 and not vflip_fail else "FAIL")
print(f"file {d.split('/')[-1]} candidate {ct}")
print(f"positions {n}  m {m:.4f} (2 x rms(marg_F - marg_R) over {len(diffs)})  TF agree C {sum(aC)}/{n}  F {sum(aF)}/{n}")
for g in greedy:
    i, k, rg, cg = g
    s = f"  greedy p{i}: {k}/64 equal"
    if k < 64:
        r = rows[(i - 1) * 64 + k]
        s += f"; first divergence step {k}: R margin {r['mr']:.4f}, C picked {cg[k]} (R rank {[x[0] for x in r['rtop']].index(cg[k]) + 1 if cg[k] in [x[0] for x in r['rtop']] else '>5'}) -> {'near-tie' if r['mr'] < m and cg[k] in r['r3'] else 'NOT near-tie'}"
    print(s)
for r in dis:
    rk = [x[0] for x in r["rtop"]]
    print(f"  TF disagreement p{r['p']} pos {r['k']}: R margin {r['mr']:.4f} (m {m:.4f}), C top1 {r['c1']} = R rank {rk.index(r['c1']) + 1 if r['c1'] in rk else '>5'}, C margin {r['cm']:.4f}, F top1 {'= R' if r['f1'] == r['r1'] else r['f1']} -> {'near-tie' if near(r) else 'HIGH-MARGIN'}")
print(f"G-A (every disagreement a near-tie): {'PASS' if ga else 'FAIL'} ({len(dis)} disagreements, {sum(1 for r in dis if near(r))} near-ties)")
print(f"G-B V-T1: D = {D:+.2f} pt, CI90 [{lo:+.2f}, {hi:+.2f}] -> {'PASS' if vt1 else 'FAIL'}")
print(f"G-B V-FLIP: b {b} c {c} p {pv:.4f} -> {'FAIL' if vflip_fail else 'not FAIL'}")
print(f"VERDICT {verdict}")
