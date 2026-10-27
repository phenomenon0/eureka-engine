# mx_ref.py MODEL.gguf DUMP [MAXREC]: the MoE FFN's own oracle. Each LLB_MX_DUMP record (the engine's FFN input h2, its
# output, its chosen ids / weights / shared gate) is recomputed from the GGUF's bytes in float64 (gguf-py's dequantize;
# llama.cpp's routing: softmax over every expert, top-k, renormalized; SwiGLU experts; sigmoid-gated shared expert).
# Prints per record the relative error |out - ref| / |ref|, whether the ids match, the max weight error; then a summary.
import sys, struct, json
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

path, dump = sys.argv[1:3]
maxrec = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 30
r = GGUFReader(path)
T = {t.name: t for t in r.tensors}
def deq(t, sl=None):
    a = np.asarray(t.data)
    if sl is not None:
        a = a[sl]
    return dequantize(a, t.tensor_type).astype(np.float64)
cache = {}
def layer(i):
    if i in cache:
        return cache[i]
    p = f"blk.{i}."
    c = dict(wr=deq(T[p + "ffn_gate_inp.weight"]), ws=deq(T[p + "ffn_gate_inp_shexp.weight"]).reshape(-1),
             sg=deq(T[p + "ffn_gate_shexp.weight"]), su=deq(T[p + "ffn_up_shexp.weight"]), sd=deq(T[p + "ffn_down_shexp.weight"]),
             fused=(p + "ffn_gate_up_exps.weight") in T)
    cache.clear(); cache[i] = c
    return c
def expert(i, e, c):
    p = f"blk.{i}."
    dn = deq(T[p + "ffn_down_exps.weight"], e)
    if c["fused"]:
        gu = deq(T[p + "ffn_gate_up_exps.weight"], e); h = gu.shape[0] // 2
        return gu[:h], gu[h:], dn
    return deq(T[p + "ffn_gate_exps.weight"], e), deq(T[p + "ffn_up_exps.weight"], e), dn
silu = lambda v: v / (1.0 + np.exp(-v))
b = open(dump, "rb").read(); o = 0; res = []
while o < len(b) and len(res) < maxrec:
    i, d, k = struct.unpack_from("<3I", b, o); o += 12
    h2 = np.frombuffer(b, np.float32, d, o).astype(np.float64); o += 4 * d
    out = np.frombuffer(b, np.float32, d, o).astype(np.float64); o += 4 * d
    ids = list(struct.unpack_from(f"<{k}I", b, o)); o += 4 * k
    w = np.frombuffer(b, np.float32, k + 1, o).astype(np.float64); o += 4 * (k + 1)
    c = layer(i)
    lg = c["wr"] @ h2
    pr = np.exp(lg - lg.max()); pr /= pr.sum()
    rid = list(np.argsort(-pr, kind="stable")[:k])
    rw = pr[rid]; rw = rw / max(rw.sum(), 6.103515625e-5)
    gs = 1.0 / (1.0 + np.exp(-(c["ws"] @ h2)))
    ref = np.zeros(d)
    for s, e in enumerate(rid):
        g, u, dn = expert(i, e, c)
        ref += rw[s] * (dn @ (silu(g @ h2) * (u @ h2)))
    ref += gs * (c["sd"] @ (silu(c["sg"] @ h2) * (c["su"] @ h2)))
    rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    res.append(dict(layer=i, rel=rel, ids_equal=[int(x) for x in ids] == [int(x) for x in rid],
                    w_err=float(np.abs(w[:k] - rw).max()) if ids == rid else None, gs_err=float(abs(w[k] - gs))))
for x in res[:8]:
    print(json.dumps(x))
rels = [x["rel"] for x in res]
print(json.dumps(dict(model=path, records=len(res), ids_equal=sum(x["ids_equal"] for x in res),
                      rel_max=max(rels), rel_mean=sum(rels) / len(rels),
                      w_err_max=max([x["w_err"] for x in res if x["w_err"] is not None], default=None),
                      gs_err_max=max(x["gs_err"] for x in res))))
