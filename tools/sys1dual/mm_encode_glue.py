#!/usr/bin/env python3
"""mm_encode_glue.py: G5 reference for the ENGINE side of the media hook, fused-trunk path.
One staged image (the "path" of a media entry) -> SigLIP-base last_hidden_state (196 x 768)
-> the glue connector (the trainer's own Resampler, from the pack) -> K rows (K x d, F32),
written as rows.f32 + rows.json for LLB_VIS_EMBD=rows.f32:K:1 with LLB_VIS_ROPE=seq.
  mm_encode_glue.py --pack PACKDIR --siglip SIGLIPDIR --image PATH --sha256 HEX --out DIR [--dtype bf16|fp32]
Preprocessing = prep_coco.py's (AutoImageProcessor, RGB convert); the encoder ran bf16 there.
This is the run-ahead step only; the trunk binding (identity, quant floor) is glue_rows.py's check."""
import argparse, hashlib, io, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
for k in ("pack", "siglip", "image", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--sha256", help="the media entry's sha256: refuse bytes that do not match")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--feats-out", help="also save the SigLIP features (.npy) for parity checks")
a = ap.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor, SiglipVisionModel

raw = open(a.image, "rb").read()
if a.sha256 and sha(raw) != a.sha256: sys.exit("mm_encode_glue: image bytes do not match the media entry sha256")
man = json.load(open(f"{a.pack}/connector.json"))
if sha(open(f"{a.pack}/connector.safetensors", "rb").read()) != man["connector_sha256"]: sys.exit("pack: connector sha mismatch")
dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
proc = AutoImageProcessor.from_pretrained(a.siglip)
enc = SiglipVisionModel.from_pretrained(a.siglip, torch_dtype=dt).eval()
im = Image.open(io.BytesIO(raw)).convert("RGB")
with torch.no_grad():
    px = proc(images=[im], return_tensors="pt")["pixel_values"].to(dt)
    feats = enc(pixel_values=px).last_hidden_state.half().float()[0]  # prep_coco stored .half()
    sys.path.insert(0, a.pack); from train_connector import Resampler
    sd = load_file(f"{a.pack}/connector.safetensors")
    c = Resampler(sd["inp.0.weight"].shape[0], sd["out.weight"].shape[0], K=sd["q"].shape[0]); c.load_state_dict(sd); c = c.float().eval()
    rows = c(feats[None])[0].to(torch.bfloat16).float().numpy()  # = glue_rows.py / batch_forward
os.makedirs(a.out, exist_ok=True)
rows.astype(np.float32).tofile(f"{a.out}/rows.f32")
if a.feats_out: np.save(a.feats_out, feats.numpy())
st = {"n": int(rows.shape[0]), "d": int(rows.shape[1]), "rope": "seq", "layout": "replace-tag",
      "env": f"LLB_VIS_EMBD={a.out}/rows.f32:{rows.shape[0]}:1 LLB_VIS_ROPE=seq",
      "rows_sha256": sha(open(f"{a.out}/rows.f32", "rb").read()), "image_sha256": sha(raw),
      "connector_sha256": man["connector_sha256"], "bound_trunk": man["trunk"]["id"], "encoder": a.siglip, "dtype": a.dtype}
json.dump(st, open(f"{a.out}/rows.json", "w"), indent=1); print(json.dumps(st))
