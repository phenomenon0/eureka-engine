# steps.py NSYS.sqlite [SKIP] : per-step GPU busy vs wall from an nsys trace; a step = the span between llb_deliver ends
import sqlite3, sys, collections, statistics as st
db = sqlite3.connect(sys.argv[1]); skip = int(sys.argv[2]) if len(sys.argv) > 2 else 10
names = dict(db.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
M = [(s, e, 'memcpy') for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY order by start")] if db.execute("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMCPY'").fetchone() else []
ev = sorted(K + M)
cut = [i for i, x in enumerate(ev) if x[2] == 'llb_deliver']
steps = []
for a, b in zip(cut[skip:-1], cut[skip + 1:]):
    seg = ev[a + 1:b + 1]
    wall = seg[-1][1] - ev[a][1]
    busy = sum(e - s for s, e, _ in seg)
    per = collections.Counter(); cnt = collections.Counter()
    for s, e, n in seg: per[n] += e - s; cnt[n] += 1
    steps.append((wall, busy, per, cnt))
w = st.median(x[0] for x in steps); bz = st.median(x[1] for x in steps)
print(f"steps {len(steps)} (skip {skip}): median wall {w/1e3:.1f} us, median GPU busy {bz/1e3:.1f} us, gaps {(w-bz)/1e3:.1f} us, launches/step {sum(steps[0][3].values())}")
agg = collections.Counter(); c0 = steps[len(steps)//2][3]
for x in steps: agg.update(x[2])
for n, t in agg.most_common(30):
    print(f"  {n:28s} {t/len(steps)/1e3:8.1f} us/step  x{c0[n]}")
