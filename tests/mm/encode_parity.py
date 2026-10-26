#!/usr/bin/env python3
"""G5 encode-path receipt: the image bytes as staged by llb-serve (via the stub engine's STUB_KEEP copy)
-> mm_encode_glue.py (live SigLIP + connector) vs glue-attach2's rows (cached training SigLIP features,
the rows the engine smoke scored at |dp| 0.0005). Reports per-row cosine and max |delta| for feats and rows.
  encode_parity.py --staged FILE --sha256 HEX --ref-feats feats_142472.npy --ref-rows rows0/rows.f32 --enc mm_encode_glue.py
                   --pack PACK --siglip DIR --out OUTDIR"""
import argparse, json, os, subprocess, sys
import numpy as np
ap = argparse.ArgumentParser()
for k in ("staged", "sha256", "ref_feats", "ref_rows", "enc", "pack", "siglip", "out"): ap.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
def cos(x, y): return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-30)
res = {}
rf, rr = np.load(a.ref_feats).astype(np.float32), np.fromfile(a.ref_rows, np.float32)
for dt in ("bf16", "fp32"):
    o = f"{a.out}/{dt}"
    st = json.loads(subprocess.run([sys.executable, "-I", a.enc, "--pack", a.pack, "--siglip", a.siglip, "--image", a.staged,
                                    "--sha256", a.sha256, "--out", o, "--dtype", dt, "--feats-out", f"{o}/feats.npy"],
                                   capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    f, r = np.load(f"{o}/feats.npy"), np.fromfile(f"{o}/rows.f32", np.float32).reshape(st["n"], st["d"])
    R = rr.reshape(r.shape)
    cf, cr = cos(f, rf), cos(r, R)
    res[dt] = {"stamp": st, "feats": {"shape": list(f.shape), "cos_min": float(cf.min()), "cos_mean": float(cf.mean()),
               "max_abs": float(np.abs(f - rf).max()), "ref_absmax": float(np.abs(rf).max())},
               "rows": {"shape": list(r.shape), "cos_min": float(cr.min()), "cos_mean": float(cr.mean()),
                        "max_abs": float(np.abs(r - R).max()), "ref_absmax": float(np.abs(R).max()),
                        "rel_fro": float(np.linalg.norm(r - R) / np.linalg.norm(R)), "byte_identical": bool((r == R).all())}}
    print(dt, json.dumps({k: res[dt][k] for k in ("feats", "rows")}))
json.dump(res, open(f"{a.out}/encode_parity.json", "w"), indent=1)
