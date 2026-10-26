#!/usr/bin/env python3
"""qref.py GGUF QCHECK.json: check trunk/qcheck.html's GPU results against ggml's reference dequantizers (gguf-py).

Needs gguf-py (pip install gguf, or PYTHONPATH=<llama.cpp>/gguf-py). For each tensor: the dequantized first rows must match
the GPU embedding gather, and W[:rows] . x must match the GPU GEMV, to f32 rounding. Prints one line a tensor and a JSON
summary; exits 1 on any mismatch.
"""
import json
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

model, results = sys.argv[1], json.load(open(sys.argv[2]))
if results.get("errors"):
    print("page errors:", results["errors"])
tensors = {t.name: t for t in GGUFReader(model).tensors}
bad, out = 0, []
for name, r in results["results"].items():
    t = tensors[name]
    K, R = r["K"], r["rows"]
    raw = np.asarray(t.data).reshape(-1, np.asarray(t.data).shape[-1])[:R]  # quantized rows (bytes) or float rows
    W = dequantize(raw, t.tensor_type).astype(np.float64).reshape(R, K)
    i = np.arange(K)
    x = (np.sin(np.float32(0.731) * i.astype(np.float32) + np.float32(0.3)) * (1 + 0.1 * (i % 7)).astype(np.float32)).astype(np.float32).astype(np.float64)
    ref = W @ x
    scale = np.abs(W) @ np.abs(x) + 1e-30          # the dot's magnitude: f32 accumulation error is relative to this
    gemv_err = float(np.max(np.abs(np.array(r["y"]) - ref) / scale))
    deq_err = float(max(np.max(np.abs(np.array(r["row0"]) - W[0])), np.max(np.abs(np.array(r["rowN"]) - W[R - 1]))))
    deq_rel = deq_err / (float(np.max(np.abs(W[[0, R - 1]]))) + 1e-30)
    ok = gemv_err < 1e-4 and deq_rel < 1e-5
    bad += not ok
    out.append({"tensor": name, "type": r["type"], "K": K, "rows": R, "gemv_rel_err": gemv_err, "dequant_rel_err": deq_rel, "ok": ok})
    print(f"{'ok ' if ok else 'BAD'} {r['type']:8s} {name:40s} K={K:6d} gemv_rel={gemv_err:.2e} deq_rel={deq_rel:.2e}")
print(json.dumps({"model": model.split("/")[-1], "tensors": len(out), "bad": bad, "results": out}))
sys.exit(1 if bad else 0)
