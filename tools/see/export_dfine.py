"""Export D-FINE-N (official .pth) -> ONNX (deploy mode, NMS-free, postprocessor baked in).
usage: python export_dfine.py <dfine_repo> <config.yml> <ckpt.pth> <out.onnx>
See tools/see/README.md for the pinned D-FINE commit, checkpoint and output sha256."""
import sys, os, json, time
repo, cfgp, ckpt, out = sys.argv[1:5]
sys.path.insert(0, repo); os.chdir(repo)
import torch, torch.nn as nn
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_category2name

cfg = YAMLConfig(cfgp, resume=ckpt)
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
st = torch.load(ckpt, map_location="cpu", weights_only=True)
st = st["ema"]["module"] if "ema" in st else st["model"]
cfg.model.load_state_dict(st)

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.model = cfg.model.deploy(); s.post = cfg.postprocessor.deploy()
    def forward(s, images, orig_target_sizes):
        return s.post(s.model(images), orig_target_sizes)

m = M().eval()
nparam = sum(p.numel() for p in m.parameters())
x = torch.rand(1, 3, 640, 640); sz = torch.tensor([[640, 640]])
with torch.no_grad(): m(x, sz)
t = time.time()
torch.onnx.export(m, (x, sz), out, input_names=["images", "orig_target_sizes"],
    output_names=["labels", "boxes", "scores"],
    dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
    opset_version=17, do_constant_folding=True, dynamo=False)
import onnx; onnx.checker.check_model(onnx.load(out))
torch.save(m.state_dict(), out + ".deploy.pt")  # deploy-mode torch reference for parity
json.dump(list(mscoco_category2name.values()), open(os.path.join(os.path.dirname(out), "coco80.json"), "w"))
print(json.dumps({"params": nparam, "export_s": round(time.time() - t, 2), "onnx_bytes": os.path.getsize(out)}))
