# g4sum.py: the synthetic shapes' table (G4 as preregistered, the floor, G4c, G4b) from runs/g4
import json, glob, collections, os
os.chdir(os.path.expanduser("~/scratch/metal-moe/runs/g4"))
rows = collections.defaultdict(dict)
for f in sorted(glob.glob("*-p?.json")):
    m, kind, p = f[:-5].rsplit("-", 2)
    rows[m][kind + "-" + p] = json.load(open(f))
print("model | engine vs CPU (G4) | llama.cpp Metal vs CPU (floor) | engine vs llama.cpp Metal (G4c) | FFN oracle (G4b)")
for m, d in rows.items():
    out = [m]
    for kind in ("moe", "floor", "moevsgpu"):
        ks = [k for k in d if k.startswith(kind + "-")]
        a = sum(d[k]["tf_agree"] for k in ks); n = sum(d[k]["n"] for k in ks)
        mm = max(d[k]["max_dis_margin"] for k in ks)
        out.append("%d/%d = %.3f (max disagreement margin %.3f)" % (a, n, a / n, mm))
    o = json.load(open(m + ".oracle.json"))
    out.append("ids %d/%d, rel max %.2e" % (o["ids_equal"], o["records"], o["rel_max"]))
    print(" | ".join(out))
