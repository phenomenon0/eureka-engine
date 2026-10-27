# vs32.py DIR : each arm (R llama CPU quant, F llama CUDA, C engine) against X (llama CPU on the F32-dequantized
# weights, no activation quantization): top-1 agreement with X and rms of (arm logit - X logit) on X's top-1 token
import json, re, struct, sys, math
d = sys.argv[1]; plen = {1: 5, 2: 8, 3: 4}
f32 = lambda u: struct.unpack("<f", struct.pack("<I", u))[0]
st = {a: [0, []] for a in "RFC"}; n = 0
for i in (1, 2, 3):
    R = json.load(open(f"{d}/R-p{i}.json"))["top"]; F = json.load(open(f"{d}/F-p{i}.json"))["top"]; X = json.load(open(f"{d}/X-p{i}.json"))["top"]
    n0 = plen[i]; C = {}
    for ln in open(f"{d}/C-tf-p{i}.txt.stderr"):
        m = re.match(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m:
            pos, a, va, b, vb = map(int, m.groups()); C[pos] = [[a, f32(va)], [b, f32(vb)]]
    for k in range(64):
        x = X[n0 - 1 + k]; x1 = x[0][0]; n += 1
        for a, t in (("R", R[k]), ("F", F[n0 - 1 + k]), ("C", C[n0 - 1 + k])):
            st[a][0] += t[0][0] == x1
            dd = {u: v for u, v in t}
            if x1 in dd: st[a][1].append(dd[x1] - x[0][1])
for a in "RFC":
    v = st[a][1]
    print(f"{d.split('/')[-1]} {a}: top-1 = X {st[a][0]}/{n}, rms(logit - X) on X top-1 {math.sqrt(sum(z*z for z in v)/len(v)):.4f} (n {len(v)})")
