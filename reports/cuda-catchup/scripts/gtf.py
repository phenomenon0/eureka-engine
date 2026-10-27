# gtf.py DIR CTAG KIND [REF] : the prereg's G-TF (KIND tf: CTAG-tf-p*.txt.stderr) / G-PF (KIND pf: CTAG-pf-p*.txt.stderr,
# the prefix probes) per file. REF X (llama.cpp CPU on the F32 file) or B (the exact tier's TF top-1, the S35 Q4_K_M
# proxy). CTAG F scores the floor itself (control). Paired against F (llama.cpp CUDA TF) on the same positions:
# PASS iff a_C >= a_F, or the exact one-sided McNemar p (C worse) > 0.10.
import json, re, struct, sys, math
d, ct, kind = sys.argv[1], sys.argv[2], sys.argv[3]; ref = sys.argv[4] if len(sys.argv) > 4 else "X"
plen = {1: 5, 2: 8, 3: 4}
f32 = lambda u: struct.unpack("<f", struct.pack("<I", u))[0]
def lines(path):
    for ln in open(path):
        m = re.match(r"logits (\d+) (\d+) (\d+) (\d+) (\d+)", ln)
        if m: yield tuple(map(int, m.groups()))
def tf_top(path):
    C = {}
    for pos, a, va, b, vb in lines(path): C[pos] = [[a, f32(va)], [b, f32(vb)]]
    return C
def pf_top(path, n0):
    C, k = {}, 0
    for pos, a, va, b, vb in lines(path):
        if k < 64 and pos == n0 - 1 + k and a != 1000000000:
            C[pos] = [[a, f32(va)], [b, f32(vb)]]; k += 1
    return C
n = aC = aF = b = c = 0; dis = []; eC = []
for i in (1, 2, 3):
    n0 = plen[i]
    F = json.load(open(f"{d}/F-p{i}.json"))["top"]
    if ref == "X": R = json.load(open(f"{d}/X-p{i}.json"))["top"]
    else: Rb = tf_top(f"{d}/B-tf-p{i}.txt.stderr")
    if ct == "F": C = {n0 - 1 + k: F[n0 - 1 + k] for k in range(64)}
    elif kind == "tf": C = tf_top(f"{d}/{ct}-tf-p{i}.txt.stderr")
    else: C = pf_top(f"{d}/{ct}-pf-p{i}.txt.stderr", n0)
    for k in range(64):
        pos = n0 - 1 + k; n += 1
        rt = R[pos] if ref == "X" else Rb[pos]
        r1 = rt[0][0]; mr = rt[0][1] - rt[1][1]
        cc = C.get(pos)
        c1 = cc[0][0] if cc else -1
        f1 = F[pos][0][0]
        okc, okf = c1 == r1, f1 == r1
        aC += okc; aF += okf; b += (not okc) and okf; c += okc and (not okf)
        if cc and ref == "X":
            dc = {u: v for u, v in cc}
            eC.append((dc[r1] if r1 in dc else cc[0][1]) - rt[0][1])
        if not okc:
            rr = [t[0] for t in rt]
            rk = rr.index(c1) + 1 if c1 in rr else ">%d" % len(rr)
            dis.append((i, k, mr, rk, mr < 0.05 and c1 in rr[:3]))
def mcnemar(b, c):
    if b <= c: return 1.0
    m = b + c
    return sum(math.comb(m, j) for j in range(b, m + 1)) / 2 ** m
p = mcnemar(b, c)
ok = aC >= aF or p > 0.10
name = d.rstrip("/").split("/")[-1]
print(f"{kind.upper()} {name} {ct} vs ref {ref}: a_C {aC}/{n}, a_F {aF}/{n}, b (C wrong, F right) {b}, c (C right, F wrong) {c}, "
      f"McNemar one-sided p {p:.4f} -> {'PASS' if ok else 'FAIL'}")
if eC: print(f"  rms(C - ref at ref's top-1) {math.sqrt(sum(z * z for z in eC) / len(eC)):.4f} over {len(eC)}")
nt = sum(1 for t in dis if t[4])
print(f"  disagreements with ref: {len(dis)} ({nt} near-tie: ref margin < 0.05, C's pick in ref top 3)")
for t in dis: print(f"    p{t[0]} pos {t[1]}: ref margin {t[2]:.4f}, C pick = ref rank {t[3]}{' near-tie' if t[4] else ''}")
