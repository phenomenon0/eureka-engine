# stepk.py NSYS.sqlite [KEY] : per-step kernel table for decode-run steps (a step = llb_embt .. next llb_embt)
import sqlite3, sys, collections, statistics as st
db=sqlite3.connect(sys.argv[1]); key=sys.argv[2] if len(sys.argv)>2 else 'llb_embt'
names=dict(db.execute("select id,value from StringIds"))
K=[(s,e,names[n],gx,bx) for s,e,n,gx,bx in db.execute("select start,end,shortName,gridX,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
idx=[i for i,k in enumerate(K) if k[2]==key]
spans=[]; agg=collections.Counter(); cnt=collections.Counter(); busy=[]
use=list(zip(idx[5:-1],idx[6:]))
for a,b in use:
    spans.append(K[b][0]-K[a][0]); busy.append(sum(K[i][1]-K[i][0] for i in range(a,b)))
    for i in range(a,b): agg[K[i][2]]+=K[i][1]-K[i][0]; cnt[K[i][2]]+=1
n=len(use)
print(f"steps {n}: median period {st.median(spans)/1e3:.1f} us, GPU busy {st.median(busy)/1e3:.1f} us, launches/step {sum(cnt.values())/n:.0f}")
for k,v in agg.most_common(25): print(f"  {k:28s} {v/n/1e3:8.1f} us/step x{cnt[k]/n:.0f}")
