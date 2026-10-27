# speedsum.py DIR : the prereg's figures from the rounds under DIR (r1 r2 r3 ..): medians over rounds
import glob, json, os, re, statistics as st, sys
d = sys.argv[1]; R = sorted(glob.glob(d + "/r*"))
def eng(f, kind):
    if not os.path.exists(f + ".stderr"): return None
    v = [(int(a), int(b)) for a, b in re.findall(r"^stepus (\d+) (\d+)$", open(f + ".stderr").read(), re.M)]
    if kind == "dec":
        x = sorted(u for p, u in v if p >= 12)
        return 1e6 / st.median(x) if x else None
    x = [u for p, u in v if 3 <= p <= 511]  # PREREG amendment 1: pos 1 holds the expert load, pos 2 the first graph
    return 509e6 / sum(x) if len(x) == 509 else None
def lc(f):
    try: return json.load(open(f))[0]["avg_ts"]
    except Exception: return None
rows = {}
for r in R:
    for f in glob.glob(r + "/*-dec.txt") + glob.glob(r + "/*-pf.txt"):
        b = os.path.basename(f)[:-4]; tag, kind = b.rsplit("-", 1)
        rows.setdefault((tag, kind), []).append(eng(f, kind))
    for f in glob.glob(r + "/*.json"):
        b = os.path.basename(f)[:-5]; tag, kind = b.rsplit("-", 1)
        rows.setdefault((tag, kind), []).append(lc(f))
out = {}
for (tag, kind), v in sorted(rows.items()):
    ok = [x for x in v if x]
    m = st.median(ok) if ok else None; out[(tag, kind)] = m
    print(f"{tag:32s} {kind:3s} median {m and round(m, 1)} tok/s  rounds {[x and round(x, 1) for x in v]}")
json.dump({f"{t}|{k}": v for (t, k), v in out.items()}, open(d + "/SUMMARY.json", "w"), indent=1)
