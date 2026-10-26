#!/usr/bin/env python3
"""glue_rows.py — the connector's run-ahead step (house pattern: tower/module runs ahead, rows spliced via LLB_VIS_EMBD).

  pack  --conn CONNECTOR.safetensors --ledger LEDGER.json --out PACKDIR        build the connector pack (binds to its trunk)
  rows  --pack PACKDIR --gguf TRUNK.gguf (--feats FEATS.npy | --image IMG) [--question Q] --out OUTDIR [--cache DIR] [--plumbing-only]
        refuses a trunk that is not the pack's bound trunk; writes rows.f32 (K x d F32) + rows.json (the stamp)
        [+ ids.txt (K pads + question)]
  bind  --pack PACKDIR --gguf TRUNK.gguf --encoder SIGLIPDIR --out PACKDIR [--expect-table SHA] [--cache DIR]   (G24 F4)
        the v1 pack: trunk.tensor_table_sha256 (content of every tensor) + the encoder's file shas
  worker --pack PACKDIR --gguf TRUNK.gguf --cache DIR    (G24 F2, llb-serve --glue-pack)
        one JSON line in ({"image":PATH} | {"image_b64":B64}), one line out ({"image":{"rows","n"},...} | {"error"})

G24 F2/F3/F4: --image runs SigLIP live (prep_coco.py's encoder step: AutoImageProcessor + SiglipVisionModel bf16,
last_hidden_state -> f16); --cache keeps sha(image) -> feats -> rows (keyed by connector + trunk table sha), atomic
renames, LRU-trimmed to --cache-max-mb; a v1 pack refuses a GGUF whose tensor_table_sha256 differs (not a label check)
and the stamp carries trunk_tensor_table_sha256, which the engine (dual.c) checks against the loaded GGUF.

The connector math is the trainer's own Resampler class (tools/train_connector.py, copied verbatim, sha in the pack)."""
import argparse, base64, hashlib, io, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAD = 248056


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def table_sha(path, cache=None):
    """tensor_table_sha256 (relayout_gguf.py's definition: sha256 of "name:sha256(bytes)\\n", names sorted) and whether it
    came from the stat-keyed cache DIR/tabsha/<dev>-<ino>-<size>-<mtime_ns> ("v1 HEADER_SHA TABLE_SHA", shared with
    glue_out.c). A hit re-hashes the header (bytes before the first tensor), so a relabel keeping size+mtime misses."""
    from gguf import GGUFReader
    st = os.stat(path); r = GGUFReader(path)
    lo = min(t.data_offset for t in r.tensors)
    with open(path, "rb") as f: hs = sha_bytes(f.read(lo))
    key = f"{cache}/tabsha/{st.st_dev}-{st.st_ino}-{st.st_size}-{st.st_mtime_ns}" if cache else None
    if key and os.path.exists(key):
        v = open(key).read().split()
        if len(v) == 3 and v[0] == "v1" and v[1] == hs: return v[2], True
    with ThreadPoolExecutor(16) as ex:  # hashlib drops the GIL on big buffers
        hx = dict(ex.map(lambda t: (t.name, hashlib.sha256(memoryview(np.ascontiguousarray(t.data)).cast("B")).hexdigest()), r.tensors))
    h = sha_bytes("".join(f"{k}:{hx[k]}\n" for k in sorted(hx)).encode())
    if key:
        os.makedirs(os.path.dirname(key), exist_ok=True); tmp = f"{key}.{os.getpid()}"
        open(tmp, "w").write(f"v1 {hs} {h}\n"); os.replace(tmp, key)
    return h, False


def gguf_identity(path):
    from gguf import GGUFReader
    r = GGUFReader(path); f = {}
    for k in ("general.name", "general.basename", "general.finetune", "general.size_label"):
        if k in r.fields:
            v = r.fields[k]; f[k] = bytes(v.parts[v.data[0]]).decode(errors="replace")
    ty = {t.name: t.tensor_type.name for t in r.tensors}
    emb = next(t for t in r.tensors if t.name == "token_embd.weight")
    return f, ty, int(emb.shape[0])


def cmd_pack(a):
    os.makedirs(a.out, exist_ok=True)
    led = json.load(open(a.ledger))
    shutil.copyfile(a.conn, f"{a.out}/connector.safetensors")
    shutil.copyfile(f"{HERE}/train_connector.py", f"{a.out}/train_connector.py")
    base = os.path.basename(led["args"]["base"].rstrip("/"))  # e.g. Qwen3.5-9B-Base
    man = {"format": "glue-connector-pack/v0", "connector_sha256": sha(f"{a.out}/connector.safetensors"),
           "ledger_artifact_sha256": led["artifact_sha256"], "trainer_sha256": sha(f"{a.out}/train_connector.py"),
           "arch": "Resampler(d_in=768, d_out=trunk_hidden, K, d=1024, L=2, heads=16)", "K": led["args"]["K"],
           "trunk": {"id": base, "hidden": led["trunk_hidden"], "fingerprint_bf16": led["trunk_fingerprint_pre"],
                     "require_gguf_basename": base, "quant_floor": "Q6_K", "allowed_types": ["Q6_K", "Q8_0", "BF16", "F16", "F32"]},
           "encoder": "google/siglip-base-patch16-224 (last_hidden_state, 196 x 768; prep_coco.py)",
           "prompt": {"layout": "[K connector rows | ' Question: {q} Answer:']", "bos": None, "positions": "sequential text (LLB_VIS_ROPE=seq)",
                      "readout": "sigmoid(logit[9542 ' yes'] - logit[874 ' no']) at the last prompt position"},
           "eval": {"acc": led["eval_bf16"]["connector"]["acc"], "auroc": led["eval_bf16"]["connector"]["auroc"],
                    "q8_agree": led["deploy_precision"]["Q8_0"]["top_answer_agreement_vs_bf16"]}}
    assert man["connector_sha256"] == man["ledger_artifact_sha256"], "connector does not match its ledger"
    json.dump(man, open(f"{a.out}/connector.json", "w"), indent=1); print(json.dumps(man, indent=1))


def load_conn(pack, man):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, pack); from train_connector import Resampler  # the trainer's class, verbatim
    sd = load_file(f"{pack}/connector.safetensors")
    d_in = sd["inp.0.weight"].shape[0]; d_out = sd["out.weight"].shape[0]; K = sd["q"].shape[0]
    assert K == man["K"] and d_out == man["trunk"]["hidden"], (K, d_out)
    c = Resampler(d_in, d_out, K=K); c.load_state_dict(sd); return c.float().eval()


def check_trunk(pack, gguf, cache):
    """the pack's refusals for this GGUF -> (manifest, identity, table sha, [reasons]); [] = bound trunk"""
    man = json.load(open(f"{pack}/connector.json"))
    assert sha(f"{pack}/connector.safetensors") == man["connector_sha256"], "pack: connector sha mismatch"
    ident, ty, d = gguf_identity(gguf); tr = man["trunk"]; why = []
    tab = None
    if tr.get("tensor_table_sha256"):  # v1: content binding (authoritative); the label checks below stay as extra reasons
        tab, _ = table_sha(gguf, cache)
        if tab != tr["tensor_table_sha256"]:
            why.append(f"trunk content: pack is bound to tensor_table_sha256 {tr['tensor_table_sha256'][:16]}..., GGUF is {tab[:16]}...")
    if ident.get("general.basename") != tr["require_gguf_basename"] and ident.get("general.name") != tr["require_gguf_basename"]:
        why.append(f"trunk identity: pack is bound to {tr['require_gguf_basename']!r}, GGUF is {ident}")
    if d != tr["hidden"]:
        why.append(f"trunk hidden: pack {tr['hidden']}, GGUF {d}")
    bad = sorted({v for k, v in ty.items() if k.endswith(".weight") and v not in tr["allowed_types"]})
    if bad:
        why.append(f"quant floor {tr['quant_floor']}: GGUF has {bad}")
    return man, ident, tab, why


class Encoder:
    """prep_coco.py's encoder step for one image: AutoImageProcessor + SiglipVisionModel (bf16), last_hidden_state -> f16"""
    def __init__(self, path, device, fast=False):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(path, use_fast=fast)
        self.enc = SiglipVisionModel.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.sha = sha(f"{path}/model.safetensors"); self.pre_sha = sha(f"{path}/preprocessor_config.json")
        self.kind = type(self.proc).__name__

    def __call__(self, data):
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        px = self.proc(images=[im], return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
        with self.torch.no_grad():
            return self.enc(pixel_values=px).last_hidden_state.half().cpu()[0].numpy()  # 196 x 768 f16


class Glue:
    """pack + trunk check + (lazy) encoder + connector + the feats/rows cache"""
    def __init__(self, pack, gguf, cache=None, plumbing_only=False, device=None, cache_max_mb=2048):
        import torch
        self.torch, self.pack, self.gguf, self.cache, self.cap = torch, pack, gguf, cache, cache_max_mb << 20
        self.man, self.ident, self.tab, self.why = check_trunk(pack, gguf, cache)
        if self.why and not plumbing_only:
            raise SystemExit("glue_rows: REFUSED\n  " + "\n  ".join(self.why))
        self.conn = load_conn(pack, self.man); self.K = self.man["K"]; self.d = self.man["trunk"]["hidden"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.enc = None

    def encoder(self):
        if self.enc is None:
            eb = self.man.get("encoder_binding") or {}
            path = eb.get("path", "google/siglip-base-patch16-224")
            self.enc = Encoder(path, self.device, fast=eb.get("processor_fast", False))
            if eb and (self.enc.sha, self.enc.pre_sha) != (eb["model_sha256"], eb["preprocessor_config_sha256"]):
                raise SystemExit(f"glue_rows: REFUSED encoder files differ from the pack's binding ({path})")
        return self.enc

    def _atomic_dir(self, final, write):
        tmp = f"{final}.tmp{os.getpid()}"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        write(tmp)
        try: os.rename(tmp, final)
        except OSError: shutil.rmtree(tmp, ignore_errors=True)  # a concurrent writer won: same bytes

    def trim(self):
        ents = []
        for sub in ("rows", "feats"):
            d = f"{self.cache}/{sub}"
            for n in os.listdir(d) if os.path.isdir(d) else []:
                p = f"{d}/{n}"; fs = [p] if os.path.isfile(p) else [f"{p}/{x}" for x in os.listdir(p)]
                ents.append((os.stat(p).st_mtime, sum(os.path.getsize(x) for x in fs), p))
        tot = sum(e[1] for e in ents)
        for _, b, p in sorted(ents):
            if tot <= self.cap: break
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p); tot -= b

    def feats(self, data):
        """image bytes -> (feats f16 196x768, info); cache key sha(image) + encoder sha"""
        enc = self.encoder(); ish = sha_bytes(data); t0 = time.time()
        fp = f"{self.cache}/feats/{ish}-{enc.sha[:16]}.npy" if self.cache else None
        if fp and os.path.exists(fp):
            os.utime(fp); return np.load(fp), {"image_sha256": ish, "feats": "hit", "encode_ms": 0.0}
        f = enc(data); ms = (time.time() - t0) * 1e3
        if fp:
            os.makedirs(os.path.dirname(fp), exist_ok=True); tmp = f"{fp}.tmp{os.getpid()}.npy"; np.save(tmp, f); os.replace(tmp, fp)
        return f, {"image_sha256": ish, "feats": "miss", "encode_ms": round(ms, 1)}

    def write_rows(self, out, f, src):
        torch = self.torch
        x = torch.from_numpy(np.asarray(f)).float()
        with torch.no_grad():
            rows = self.conn(x[None] if x.ndim == 2 else x)[0].to(torch.bfloat16).float().numpy()  # = batch_forward: conn(f).to(bf16)
        rows.astype(np.float32).tofile(f"{out}/rows.f32")
        stamp = {"rows_sha256": sha(f"{out}/rows.f32"), "K": self.K, "d": self.d,
                 "gguf": self.gguf, "gguf_identity": self.ident, "bound_trunk": self.man["trunk"]["id"],
                 "connector_sha256": self.man["connector_sha256"], "pack_format": self.man["format"], "source": src,
                 "PLUMBING_ONLY_NOT_A_GLUE_RESULT": bool(self.why), "refusal_overridden": self.why}
        if self.tab and not self.why: stamp["trunk_tensor_table_sha256"] = self.tab  # the engine refuses rows without it
        json.dump(stamp, open(f"{out}/rows.json", "w"), indent=1)
        return stamp

    def rows_for_image(self, data):
        """image bytes -> (rows dir, stamp, info) through the cache (rows are question-independent)"""
        t0 = time.time(); ish = sha_bytes(data)
        key = f"{ish}-{self.man['connector_sha256'][:16]}-{(self.tab or 'unbound')[:16]}"
        rd = f"{self.cache}/rows/{key}"
        if os.path.exists(f"{rd}/rows.json"):
            st = json.load(open(f"{rd}/rows.json"))
            if sha(f"{rd}/rows.f32") == st["rows_sha256"]:
                os.utime(rd); return rd, st, {"image_sha256": ish, "rows": "hit", "ms": round((time.time() - t0) * 1e3, 1)}
            shutil.rmtree(rd)
        f, info = self.feats(data)
        src = {"image_sha256": ish, "encoder_model_sha256": self.enc.sha, "processor": self.enc.kind, "device": self.device}
        os.makedirs(f"{self.cache}/rows", exist_ok=True)
        self._atomic_dir(rd, lambda d: self.write_rows(d, f, src)); self.trim()
        info.update(rows="miss", ms=round((time.time() - t0) * 1e3, 1))
        return rd, json.load(open(f"{rd}/rows.json")), info


def cmd_rows(a):
    g = Glue(a.pack, a.gguf, a.cache, a.plumbing_only, a.device)
    os.makedirs(a.out, exist_ok=True)
    if a.image:
        data = open(a.image, "rb").read()
        if a.cache:
            rd, st, info = g.rows_for_image(data)
            for n in ("rows.f32", "rows.json"): shutil.copyfile(f"{rd}/{n}", f"{a.out}/{n}")
        else:
            f, info = g.feats(data)
            st = g.write_rows(a.out, f, {"image_sha256": info["image_sha256"], "encoder_model_sha256": g.enc.sha,
                                         "processor": g.enc.kind, "device": g.device})
        if a.save_feats: np.save(a.save_feats, g.feats(data)[0])
        print(json.dumps({"cache": info}), file=sys.stderr)
    else:
        st = g.write_rows(a.out, np.load(a.feats), {"feats_sha256": sha(a.feats)})
    if a.question:
        from transformers import AutoTokenizer
        q = AutoTokenizer.from_pretrained(a.tokenizer)(f" Question: {a.question} Answer:", add_special_tokens=False).input_ids
        open(f"{a.out}/ids.txt", "w").write(" ".join(map(str, [PAD] * g.K + q)) + "\n")
    print(json.dumps(st, indent=1))


def cmd_bind(a):
    man, ident, _, why = check_trunk(a.pack, a.gguf, None)
    if why: sys.exit("glue_rows: REFUSED (bind)\n  " + "\n  ".join(why))
    tab, hit = table_sha(a.gguf, a.cache)
    if a.expect_table and tab != a.expect_table: sys.exit(f"glue_rows: REFUSED (bind) tensor_table_sha256 {tab} != expected {a.expect_table}")
    os.makedirs(a.out, exist_ok=True)
    for n in ("connector.safetensors", "train_connector.py"): shutil.copyfile(f"{a.pack}/{n}", f"{a.out}/{n}")
    man["format"] = "glue-connector-pack/v1"; man["trunk"]["tensor_table_sha256"] = tab
    man["encoder_binding"] = {"path": a.encoder, "model_sha256": sha(f"{a.encoder}/model.safetensors"),
                              "preprocessor_config_sha256": sha(f"{a.encoder}/preprocessor_config.json"),
                              "dtype": "bfloat16", "processor_fast": a.fast, "output": "last_hidden_state.half()"}
    man["bound_from"] = {"pack": a.pack, "gguf": a.gguf, "gguf_identity": ident, "table_sha_cache_hit": hit}
    json.dump(man, open(f"{a.out}/connector.json", "w"), indent=1); print(json.dumps(man, indent=1))


def cmd_worker(a):
    try:
        g = Glue(a.pack, a.gguf, a.cache, False, a.device, a.cache_max_mb); g.encoder()
    except SystemExit as e:
        print(f"glue worker refused: {e}", file=sys.stderr); sys.exit(2)
    print(f"glue worker ready: pack {g.man['connector_sha256'][:16]} trunk {g.tab[:16]} K {g.K} device {g.device} "
          f"processor {g.enc.kind}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            q = json.loads(line)
            if "image_b64" in q: data = base64.b64decode(q["image_b64"], validate=True)
            elif isinstance(q.get("image"), str): data = open(q["image"], "rb").read()
            else: raise ValueError("llb_image needs image (server path) or image_b64")
            if not data: raise ValueError("empty image")
            rd, st, info = g.rows_for_image(data)
            rep = {"image": {"rows": f"{rd}/rows.f32", "n": g.K}, "rows_sha256": st["rows_sha256"],
                   "trunk_tensor_table_sha256": st.get("trunk_tensor_table_sha256"), "connector_sha256": st["connector_sha256"], **info}
        except Exception as e:  # one bad request must not kill the worker
            rep = {"error": f"{type(e).__name__}: {e}"[:400]}
        sys.stdout.write(json.dumps(rep) + "\n"); sys.stdout.flush()


ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("pack"); p.add_argument("--conn", required=True); p.add_argument("--ledger", required=True); p.add_argument("--out", required=True)
r = sub.add_parser("rows"); r.add_argument("--pack", required=True); r.add_argument("--gguf", required=True)
src = r.add_mutually_exclusive_group(required=True); src.add_argument("--feats"); src.add_argument("--image")
r.add_argument("--question"); r.add_argument("--out", required=True); r.add_argument("--plumbing-only", action="store_true")
r.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B"); r.add_argument("--cache"); r.add_argument("--device")
r.add_argument("--save-feats")
b = sub.add_parser("bind"); b.add_argument("--pack", required=True); b.add_argument("--gguf", required=True)
b.add_argument("--encoder", required=True); b.add_argument("--out", required=True); b.add_argument("--expect-table"); b.add_argument("--cache")
b.add_argument("--fast", action="store_true")
w = sub.add_parser("worker"); w.add_argument("--pack", required=True); w.add_argument("--gguf", required=True); w.add_argument("--cache", required=True)
w.add_argument("--device"); w.add_argument("--cache-max-mb", type=int, default=2048)
a = ap.parse_args(); {"pack": cmd_pack, "rows": cmd_rows, "bind": cmd_bind, "worker": cmd_worker}[a.cmd](a)
