import sqlite3, collections, sys
db=sqlite3.connect(sys.argv[1])
names=dict(db.execute("select id,value from StringIds"))
K=[(s,e,names[n]) for s,e,n in db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
sel=[k for k in K if k[2].startswith('pf') or k[2].startswith('llb_mx') or k[2]=='llb_silu']
half=len(sel)//2; sel=sel[half:]
agg=collections.Counter(); cnt=collections.Counter()
for s,e,n in sel: agg[n]+=e-s; cnt[n]+=1
for k,v in agg.most_common(14): print(f"{k:24s} {v/1e3:9.1f} us x{cnt[k]}")
print('busy', sum(agg.values())/1e3, 'span', (sel[-1][1]-sel[0][0])/1e3)
