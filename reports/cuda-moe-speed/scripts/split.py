# split.py NSYS.sqlite : per step, the pass span (llb_embed* start .. llb_deliver end) vs the rest of the period
import sqlite3, sys, statistics as st
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
dl = [i for i, x in enumerate(K) if x[2] == 'llb_deliver']
P, R = [], []
for a, b in zip(dl[10:-1], dl[11:]):
    em = next(i for i in range(a + 1, b) if K[i][2].startswith('llb_embed'))
    span = K[b][1] - K[em][0]; per = K[b][1] - K[a][1]
    P.append(span); R.append(per - span)
print(f"median period {st.median([p + r for p, r in zip(P, R)])/1e3:.1f} us: pass span {st.median(P)/1e3:.1f} us, outside the pass {st.median(R)/1e3:.1f} us")
