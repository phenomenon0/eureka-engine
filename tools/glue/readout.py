#!/usr/bin/env python3
"""readout.py GGUF HIDDEN_DUMP [--full]: last prompt row of an LLB_DUMP_HIDDEN dump (u32 pos, u32 W, f32[W] LE, final-normed)
. dequantized output.weight rows -> logit_yes(9542) - logit_no(874), p_yes; --full also the argmax over the whole vocab."""
import json, sys
import numpy as np
from gguf import GGUFReader, quants

YES, NO = 9542, 874
gguf, hid = sys.argv[1:3]; full = "--full" in sys.argv
b = open(hid, "rb").read(); o = 0; rows = {}
while o < len(b):
    pos, w = np.frombuffer(b, "<u4", 2, o); o += 8
    rows[int(pos)] = np.frombuffer(b, "<f4", int(w), o).astype(np.float64); o += 4 * int(w)
last = max(rows); h = rows[last]
W = next(t for t in GGUFReader(gguf).tensors if t.name == "output.weight"); data = np.asarray(W.data)
lg = lambda ids: quants.dequantize(data[ids], W.tensor_type).astype(np.float64) @ h
ly, ln = lg([YES, NO]); out = {"rows": len(rows), "last_pos": last, "logit_yes": ly, "logit_no": ln,
                              "diff": ly - ln, "p_yes": float(1 / (1 + np.exp(-(ly - ln))))}
if full:
    best, arg = -np.inf, -1
    for s in range(0, data.shape[0], 16384):
        v = lg(list(range(s, min(s + 16384, data.shape[0])))); i = int(v.argmax())
        if v[i] > best: best, arg = float(v[i]), s + i
    out["argmax"] = arg; out["argmax_logit"] = best
print(json.dumps(out))
