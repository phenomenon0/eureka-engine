# speedsum.py DIR : per arm, the round figures and their median (decode: 1e6 / median stepus pos >= 12; prefill:
# 512e6 / median pfus; llama.cpp: llama-bench avg_ts)
import json, os, re, statistics as st, sys, glob
d = sys.argv[1]; rounds = sorted(glob.glob(f"{d}/r*")); res = {}
def ok(path): return os.path.exists(path) and 'rc=0' in open(path).read()
for r in rounds:
    for f in sorted(os.listdir(r)):
        p = f"{r}/{f}"
        if f.endswith("-dec.txt.stderr") and ok(p[:-7]):
            v = sorted(int(m.group(2)) for m in (re.match(r"stepus (\d+) (\d+)", l) for l in open(p)) if m and int(m.group(1)) >= 12)
            if v: res.setdefault(f[:-15] + " decode", []).append(1e6 / st.median(v))
        elif f.endswith("-pf.txt.stderr") and ok(p[:-7]):
            v = [int(l.split()[1]) for l in open(p) if l.startswith("pfus ")]
            # amendment 2: the per-prompt period from the stepus lines (each prompt's lines after its pos 0
            # line up to and including the next prompt's pos 0 line; prompt 1 from the first line)
            su = [tuple(map(int, l.split()[1:3])) for l in open(p) if l.startswith("stepus ")]
            per, acc = [], 0
            for pos, us in su:
                acc += us
                if pos == 0:
                    per.append(acc); acc = 0
            if len(v) == 3: res.setdefault(f[:-14] + " prefill", []).append(512e6 / st.median(v))
            elif len(per) == 3: res.setdefault(f[:-14] + " prefill", []).append(512e6 / st.median(per))
            if len(v) == 3 and len(per) == 3: res.setdefault(f[:-14] + " prefill (stepus check)", []).append(512e6 / st.median(per))
        elif f.endswith("-dec.json") or f.endswith("-pf.json"):
            try:
                for x in json.load(open(p)):
                    k = f[:-9] + " decode" if f.endswith("-dec.json") else f[:-8] + " prefill"
                    res.setdefault(k, []).append(x["avg_ts"])
            except Exception: pass
for k in sorted(res):
    v = res[k]; print(f"{k:34s} median {st.median(v):9.1f} tok/s  rounds {' '.join('%.1f' % x for x in v)}")
