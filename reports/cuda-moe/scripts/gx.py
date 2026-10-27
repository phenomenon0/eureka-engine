# gx.py DIR [CTAG] : PREREG amendment 1's G-X: C against X (llama.cpp CPU on the F32-dequantized GGUF)
import json, re, struct, sys, math
d = sys.argv[1]; ct = sys.argv[2] if len(sys.argv) > 2 else "C"; plen = {1: 5, 2: 8, 3: 4}
f32 = lambda u: struct.unpack("<f", struct.pack("<I", u))[0]
eC, eR, dis, n, agree = [], [], [], 0, 0
for i in (1, 2, 3):
    R = json.load(open(f"{d}/R-p{i}.json"))["top"]; X = json.load(open(f"{d}/X-p{i}.json"))["top"]
    n0 = plen[i]; C = {}
    for ln in open(f"{d}/{ct}-tf-p{i}.txt.stderr"):
        m = re.match(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m:
            pos, a, va, b, vb = map(int, m.groups()); C[pos] = [[a, f32(va)], [b, f32(vb)]]
    for k in range(64):
        x = X[n0 - 1 + k]; x1 = x[0][0]; mx = x[0][1] - x[1][1]; c = C[n0 - 1 + k]; n += 1
        dc = {u: v for u, v in c}; dr = {u: v for u, v in R[k]}
        if x1 in dc: eC.append(dc[x1] - x[0][1])
        else: eC.append(c[0][1] - x[0][1])  # X's top-1 not in C's top-2: C's top-1 logit stands in (an underestimate of the error)
        if x1 in dr: eR.append(dr[x1] - x[0][1])
        if c[0][0] == x1: agree += 1
        else:
            xr = [t[0] for t in x]
            dis.append((i, k, mx, xr.index(c[0][0]) + 1 if c[0][0] in xr else ">5", mx < 0.05 and c[0][0] in xr[:3]))
rms = lambda v: math.sqrt(sum(z * z for z in v) / len(v))
ok1 = rms(eC) <= rms(eR); ok2 = all(t[4] for t in dis)
print(f"G-X {d.split('/')[-1]} candidate {ct}: top-1 = X {agree}/{n}; rms(C - X) {rms(eC):.4f} vs rms(R - X) {rms(eR):.4f} -> (i) {'PASS' if ok1 else 'FAIL'}")
for t in dis:
    print(f"  C != X p{t[0]} pos {t[1]}: X margin {t[2]:.4f}, C pick = X rank {t[3]} -> {'fp32 near-tie' if t[4] else 'NOT near-tie'}")
print(f"(ii) every C != X an fp32 near-tie (margin < 0.05, X top-3): {'PASS' if ok2 else 'FAIL'} ({len(dis)} disagreements)")
print(f"G-X VERDICT {'PASS' if ok1 and ok2 else 'FAIL'}")
