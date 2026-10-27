# ldiff.py DIR : logit-value spread of C and F against R on positions where all three top-1 agree
import json, re, struct, sys, math
d = sys.argv[1]; plen = {1: 5, 2: 8, 3: 4}
f32 = lambda u: struct.unpack("<f", struct.pack("<I", u))[0]
dc, df = [], []
for i in (1, 2, 3):
    R = json.load(open(f"{d}/R-p{i}.json"))["top"]; F = json.load(open(f"{d}/F-p{i}.json"))["top"]; n0 = plen[i]
    C = {}
    for ln in open(f"{d}/C-tf-p{i}.txt.stderr"):
        m = re.match(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m:
            pos, a, va, b, vb = map(int, m.groups()); C[pos] = (a, f32(va))
    for k in range(64):
        r = R[k][0]; f = F[n0 - 1 + k][0]; c = C[n0 - 1 + k]
        if r[0] == f[0] == c[0]:
            dc.append(c[1] - r[1]); df.append(f[1] - r[1])
rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
print(f"{d}: n {len(dc)}  rms(C-R) top1 logit {rms(dc):.4f}  rms(F-R) {rms(df):.4f}  mean(C-R) {sum(dc)/len(dc):+.4f}")
