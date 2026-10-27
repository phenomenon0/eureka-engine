# tl.py NSYS.sqlite STEPIDX : one step's timeline (kernels, memcpys, and CUDA API calls on host) between llb_deliver ends
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); si = int(sys.argv[2])
names = dict(db.execute("select id, value from StringIds"))
K = [(s, e, names[n], gx, bx) for s, e, n, gx, bx in db.execute("select start, end, shortName, gridX, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
M = [(s, e, 'memcpy%d:%d' % (k, b), 0, 0) for s, e, k, b in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
A = [(s, e, names[n]) for s, e, n in db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
ev = sorted(K + M)
cut = [i for i, x in enumerate(ev) if x[2] == 'llb_deliver']
a, b = cut[si], cut[si + 1]
t0 = ev[a][1]
for s, e, n, gx, bx in ev[a + 1:b + 1]:
    if (e - s) > float(sys.argv[3] if len(sys.argv) > 3 else 0) * 1e3 or not n.startswith('llb_'):
        print(f"{(s - t0)/1e3:9.1f} {(e - s)/1e3:8.1f} {n} g{gx} b{bx}")
print("--- API calls >5us")
t1 = ev[b][1]
for s, e, n in A:
    if s >= t0 - 0 and s <= t1 and e - s > 5000: print(f"{(s - t0)/1e3:9.1f} {(e - s)/1e3:8.1f} {n}")
