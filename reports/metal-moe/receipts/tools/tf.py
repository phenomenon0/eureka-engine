# tf.py VARIANT MODEL SIDE IDS_FILE N [engine flags..]: the engine's greedy run vs llama.cpp (CPU) on the same GGUF:
# free-running prefix, teacher-forced top-1 agreement (llama.cpp's argmax at each position of the engine's own
# continuation), every disagreement with llama.cpp's margin, and |d top-1 logit| where the top-1 ids agree.
# TFREF=gpu: the reference is llama.cpp on Metal instead of the CPU.
import sys, subprocess, struct, json, os
v, m, side, idf, n = sys.argv[1:6]; extra = sys.argv[6:]
os.chdir(os.path.expanduser("~/scratch/metal-moe"))
pid = open(idf).read().split()
tag = f"{os.path.basename(m)}-{os.path.basename(idf)}-{n}{''.join(extra).replace('-', '')}"
eo, ee = f"runs/tf/{tag}.{v}.out", f"runs/tf/{tag}.{v}.err"
os.makedirs("runs/tf", exist_ok=True)
G = m if os.path.isfile(m) else f"models/{m}.gguf"
f = lambda b: struct.unpack("f", struct.pack("I", int(b)))[0]
eng = {}
if v in ("llamagpu", "llamacpu"):
    # the floor / perfect controls: llama.cpp's own Metal path (or the CPU reference itself) as the candidate
    a = ["./bin/greedy2", G, idf, n] + (["--gpu"] if v == "llamagpu" else [])
    out = subprocess.run(a, capture_output=True, text=True).stdout
    open(eo, "w").write(out); rc = 0
    gen = next(l for l in out.splitlines() if l.startswith("gen_ids")).split()[2:]
    for l in out.splitlines():
        if l.startswith("top "):
            q = l.split(); eng[len(pid) - 1 + int(q[1])] = (q[2], float(q[3]), q[4], float(q[5]))
else:
    with open(eo, "w") as fo, open(ee, "w") as fe:
        rc = subprocess.run(["./run.sh", v, m, side, "@" + idf, "--steps", n] + extra, stdout=fo, stderr=fe).returncode
    gen = [l.strip() for l in open(eo) if l.strip() and not l.startswith("wall_ms")]
    for l in open(ee):
        if l.startswith("logits "):
            q = l.split(); eng[int(q[1])] = (q[2], f(q[3]), q[4], f(q[5]))
open(f"runs/tf/{tag}.force", "w").write(" ".join(gen))
def ref(force):
    a = ["./bin/greedy2", G, idf, n] + (["--force", f"runs/tf/{tag}.force"] if force else []) \
        + (["--gpu"] if os.environ.get("TFREF") == "gpu" else [])
    out = subprocess.run(a, capture_output=True, text=True).stdout.splitlines()
    g = next(l for l in out if l.startswith("gen_ids")).split()[2:]
    tops = {int(l.split()[1]): l.split()[2:] for l in out if l.startswith("top ")}
    return g, tops
rg, rtops = ref(False)
fg, ftops = ref(True)
N = len(gen)
pre = 0
while pre < min(N, len(rg)) and gen[pre] == rg[pre]: pre += 1
dis = []; agree = 0; dl = []
for k in range(N):
    t = ftops[k]; e = eng.get(len(pid) - 1 + k)
    if t[0] == gen[k]:
        agree += 1
    else:
        dis.append(dict(pos=k, ref=t[0], ref_logit=float(t[1]), ref_2nd=t[2], ref_margin=round(float(t[1]) - float(t[3]), 4),
                        engine=gen[k], engine_2nd=e[2] if e else None,
                        engine_margin=round(e[1] - e[3], 4) if e else None))
    if e and e[0] == t[0]: dl.append(abs(e[1] - float(t[1])))
r = dict(variant=v, ref="llama.cpp " + ("Metal" if os.environ.get("TFREF") == "gpu" else "CPU"), model=m, prompt=idf, n=N, rc=rc, first_equal=bool(N and rg and gen[0] == rg[0]), free_prefix=pre,
         tf_agree=agree, tf_rate=round(agree / N, 4) if N else 0, max_dis_margin=max([d["ref_margin"] for d in dis], default=0),
         dlogit_mean=round(sum(dl) / len(dl), 4) if dl else None, dlogit_max=round(max(dl), 4) if dl else None,
         disagreements=dis, wall=[l.strip() for l in open(eo) if l.startswith("wall_ms")])
print(json.dumps(r))
