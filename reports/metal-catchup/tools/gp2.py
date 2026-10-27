import sys
sys.argv=["x", sys.argv[1] if len(sys.argv) > 1 else "prof/trace-gpu.xml"]
src=open("gpuint.py").read().split("out.sort()")[0]
exec(src)
out=[o for o in out if "sys1mac" in o[4] and o[2]=="Compute"]
out.sort()
print(len(out))
t0=out[0][0]
for o in out[:60]: print(round((o[0]-t0)/1e6,3), round(o[1]/1e3,1), o[3], o[4][:50], o[5], o[6])
