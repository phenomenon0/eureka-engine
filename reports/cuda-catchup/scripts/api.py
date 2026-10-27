import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); si = int(sys.argv[2])
names = dict(db.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
cut = [x for x in K if x[2] == 'llb_deliver']
t0, t1 = cut[si][1], cut[si + 1][1]
A = [(s, e, names[n]) for s, e, n in db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
G = [(s, e, 'GPU:' + n) for s, e, n in K if s >= t0 and s <= t1 and not n.startswith('llb_') or n in ('llb_embed6','llb_deliver','llb_mail')]
U = []
try:
  U = [(s, e, 'UM:%s' % k) for s, e, k in db.execute("select start, end, migrationCause from CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER order by start")]
except Exception: pass
ev = sorted([x for x in A if x[0] >= t0 and x[0] <= t1] + [x for x in G if x[0] >= t0 and x[0] <= t1] + [x for x in U if x[0] >= t0 and x[0] <= t1])
last = None
for s, e, n in ev:
    if n == last and n == 'cuLaunchKernel': continue
    last = n
    print(f"{(s - t0)/1e3:9.1f} {(e - s)/1e3:8.1f} {n}")
