#!/usr/bin/env python3
"""eureka-shift — first real W2 glue bead: sensory connector trained THROUGH a frozen Qwen3.5 trunk.

Re-implements the glue-kit spec (glue-training-readiness.md §7; review §7 W2) because the PC-side skeleton was not reachable:
  frozen SigLIP features (cached; encoder is upstream) -> Perceiver resampler (trainable, K latents at trunk width)
  -> inputs_embeds = [latents | question | answer | caption] -> ONE frozen-trunk forward WITH autograd (rule a)
  loss = L_task(answer span) + lambda_ground * L_ground(caption span)  (rule b)
Guards: grad-flow canary, base-drift (bitwise) check, trunk params never get .grad. Eval: connector vs shuffled-sensor vs
no-sensor (+ independent linear-probe baseline), then the SAME connector against Q8_0/Q4_0 round-tripped trunk weights (rule c).
"""
import argparse, hashlib, json, math, os, random, time
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Resampler(nn.Module):
    def __init__(self, d_in, d_out, K=64, d=1024, L=2, heads=16, target_norm=1.0):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.q = nn.Parameter(torch.randn(K, d) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            ln_q=nn.LayerNorm(d), ln_kv=nn.LayerNorm(d), xa=nn.MultiheadAttention(d, heads, batch_first=True),
            ln_m=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))) for _ in range(L)])
        self.ln_out = nn.LayerNorm(d); self.out = nn.Linear(d, d_out)
        self.scale = nn.Parameter(torch.tensor(float(target_norm)))  # latents start at the trunk's token-embedding norm

    def forward(self, x):
        kv = self.inp(x); q = self.q.expand(x.size(0), -1, -1)
        for l in self.layers:
            k = l.ln_kv(kv)
            q = q + l.xa(l.ln_q(q), k, k, need_weights=False)[0]
            q = q + l.mlp(l.ln_m(q))
        y = self.out(self.ln_out(q))
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.scale


def load_trunk(path):
    m, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    m.requires_grad_(False)  # frozen = requires_grad False; NEVER no_grad on the receiving pass (rule a)
    return m.cuda().eval(), {k: len(v) for k, v in info.items() if isinstance(v, (list, set))}


def trunk_fingerprint(m, n=12):
    names = [k for k, _ in m.named_parameters()]; pick = names[:: max(1, len(names) // n)]
    p = dict(m.named_parameters()); h = hashlib.sha256()
    for k in pick:
        h.update(k.encode()); h.update(p[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class Data:
    def __init__(self, d, tok, K):
        self.rows = [json.loads(l) for l in open(f"{d}/rows.jsonl")]
        fz = torch.load(f"{d}/siglip_feats.pt"); self.idx = {i: k for k, i in enumerate(fz["image_ids"])}; self.feats = fz["feats"]
        self.tok, self.K = tok, K
        ids = sorted(self.idx); self.other = {i: ids[(k + 1) % len(ids)] for k, i in enumerate(ids)}
        self.yes = tok(" yes", add_special_tokens=False).input_ids; self.no = tok(" no", add_special_tokens=False).input_ids
        assert len(self.yes) == 1 and len(self.no) == 1, (self.yes, self.no)
        self.eos = tok.eos_token_id
        self.cap_pre = tok("\nCaption:", add_special_tokens=False).input_ids

    def split(self, s):
        return [r for r in self.rows if r["split"] == s]

    def encode(self, r):
        q = self.tok(f" Question: {r['question']} Answer:", add_special_tokens=False).input_ids
        a = self.yes if r["answer"] == "yes" else self.no
        c = self.tok(" " + r["caption"], add_special_tokens=False).input_ids[:48] + [self.eos]
        ids = q + a + self.cap_pre + c
        lab_t = [-100] * len(q) + a + [-100] * (len(self.cap_pre) + len(c))
        lab_g = [-100] * (len(q) + len(a) + len(self.cap_pre)) + c
        return ids, lab_t, lab_g, len(q)  # answer token sits at text position len(q); predicted from position len(q)-1


def batch_forward(model, conn, data, rows, mode, emb):
    """mode: connector | shuffle | nosensor. Returns task CE, ground CE, p_yes per row."""
    enc = [data.encode(r) for r in rows]
    f = data.feats[[data.idx[r["doc_id"]] for r in rows]].cuda().float()
    if mode == "shuffle":  # each question sees a DIFFERENT image's features (fixed image->next-image map; never same doc_id)
        f = data.feats[[data.idx[data.other[r["doc_id"]]] for r in rows]].cuda().float()
    K = 0 if mode == "nosensor" else data.K
    lat = conn(f).to(torch.bfloat16) if K else None
    T = max(len(e[0]) for e in enc) + K
    D = emb.weight.shape[1]
    x = torch.zeros(len(rows), T, D, dtype=torch.bfloat16, device="cuda"); am = torch.zeros(len(rows), T, dtype=torch.long, device="cuda")
    lt = torch.full((len(rows), T), -100, device="cuda"); lg = torch.full((len(rows), T), -100, device="cuda"); apos = []
    for b, (ids, a, g, nq) in enumerate(enc):
        n = len(ids); te = emb(torch.tensor(ids, device="cuda"))
        if K: x[b, :K] = lat[b]
        x[b, K:K + n] = te; am[b, :K + n] = 1
        lt[b, K:K + n] = torch.tensor(a, device="cuda"); lg[b, K:K + n] = torch.tensor(g, device="cuda"); apos.append(K + nq - 1)
    logits = model(inputs_embeds=x, attention_mask=am).logits.float()
    lp = logits[:, :-1]
    task = F.cross_entropy(lp.reshape(-1, lp.size(-1)), lt[:, 1:].reshape(-1), ignore_index=-100)
    ground = F.cross_entropy(lp.reshape(-1, lp.size(-1)), lg[:, 1:].reshape(-1), ignore_index=-100)
    al = logits[torch.arange(len(rows)), torch.tensor(apos)]
    p_yes = torch.sigmoid(al[:, data.yes[0]] - al[:, data.no[0]])
    return task, ground, p_yes


def auroc(y, s):
    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]
    import bisect; neg = sorted(neg); return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def evaluate(model, conn, data, rows, mode, emb, bs=32):
    conn.eval(); P, Tl, Gl = [], [], []
    with torch.no_grad():  # eval only — training path never uses no_grad
        for i in range(0, len(rows), bs):
            t, g, p = batch_forward(model, conn, data, rows[i:i + bs], mode, emb); P += p.tolist(); Tl.append(t.item()); Gl.append(g.item())
    conn.train(); y = [r["answer"] == "yes" for r in rows]
    acc = sum((p > 0.5) == t for p, t in zip(P, y)) / len(y)
    ece = 0.0
    for lo in [i / 10 for i in range(10)]:
        bucket = [(p, t) for p, t in zip(P, y) if lo <= p < lo + 0.1 or (lo == 0.9 and p == 1.0)]
        if bucket: ece += len(bucket) / len(y) * abs(sum(p for p, _ in bucket) / len(bucket) - sum(t for _, t in bucket) / len(bucket))
    return {"mode": mode, "n": len(y), "acc": round(acc, 4), "auroc": round(auroc(y, P), 4), "ece": round(ece, 4),
            "task_ce": round(sum(Tl) / len(Tl), 4), "ground_ce": round(sum(Gl) / len(Gl), 4)}, P


def boot_diff(y, pa, pb, reps=2000, seed=1):
    r = random.Random(seed); n = len(y); ca = [(p > 0.5) == t for p, t in zip(pa, y)]; cb = [(p > 0.5) == t for p, t in zip(pb, y)]
    ds = sorted(sum(ca[i] - cb[i] for i in idx) / n for idx in ([r.randrange(n) for _ in range(n)] for _ in range(reps)))
    return [round(sum(ca) / n - sum(cb) / n, 4), round(ds[int(.025 * reps)], 4), round(ds[int(.975 * reps)], 4)]


def quant_roundtrip(model, qname, gguf_py):
    """Replace every 2-D trunk weight by its ggml-quantized->dequantized value (exact ggml block formats via gguf-py)."""
    import sys; sys.path.insert(0, gguf_py)
    import numpy as np
    from gguf import quants, GGMLQuantizationType as Q
    qt = getattr(Q, qname); n = 0
    for k, p in model.named_parameters():
        if p.ndim == 2 and p.shape[-1] % 32 == 0 and "norm" not in k:
            a = p.detach().float().cpu().numpy()
            p.data.copy_(torch.from_numpy(quants.dequantize(quants.quantize(a, qt), qt).reshape(a.shape).astype(np.float32)).to(p.dtype)); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-ground", type=float, default=0.5); ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--gguf-py", default="/root/llama.cpp/gguf-py"); ap.add_argument("--quants", default="Q8_0,Q4_0")
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed)
    led = {"args": vars(a), "torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0)}
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
        led["fast_path"] = {k: bool(getattr(MQ, k)) for k in dir(MQ) if k.startswith("is_fast_path") or k in (
            "causal_conv1d_fn", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")}
    except Exception as e:
        led["fast_path"] = f"introspection failed: {e}"
    import transformers; led["transformers"] = transformers.__version__
    tok = AutoTokenizer.from_pretrained(a.base); t0 = time.time()
    model, led["load_info"] = load_trunk(a.base); led["trunk_load_s"] = round(time.time() - t0, 1)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    emb = model.get_input_embeddings(); D = emb.weight.shape[1]
    led["trunk_hidden"] = D; led["trunk_params"] = sum(p.numel() for p in model.parameters())
    tn = emb.weight.float().norm(dim=-1).mean().item()
    data = Data(a.data, tok, a.K)
    conn = Resampler(data.feats.shape[-1], D, K=a.K, target_norm=tn).cuda()
    led["connector_params"] = sum(p.numel() for p in conn.parameters())
    fp0 = trunk_fingerprint(model); led["trunk_fingerprint_pre"] = fp0
    train, val = data.split("train"), data.split("val")
    if a.smoke: train, val = train[:a.micro_batch * 3], val[:64]
    steps = 3 if a.smoke else int(a.epochs * len(train) / a.micro_batch)
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    ck = os.path.join(a.out, "ckpt.pt"); start = 0
    if os.path.exists(ck) and not a.smoke:
        st = torch.load(ck); conn.load_state_dict(st["conn"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]
        led["resumed_from"] = start
    zero = {k: v for k, v in evaluate(model, conn, data, val, "nosensor", emb)[0].items()}
    led["eval_nosensor_zero_shot"] = zero
    order = []; ep = 0
    while len(order) < steps * a.micro_batch:
        r = random.Random(a.seed + ep); idx = list(range(len(train))); r.shuffle(idx); order += idx; ep += 1
    torch.cuda.reset_peak_memory_stats(); tt = time.time(); log = open(os.path.join(a.out, "train-log.jsonl"), "a"); ntok = 0
    for s in range(start, steps):
        rows = [train[i] for i in order[s * a.micro_batch:(s + 1) * a.micro_batch]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            task, ground, _ = batch_forward(model, conn, data, rows, "connector", emb)
        loss = task + a.lambda_ground * ground
        opt.zero_grad(set_to_none=True); loss.backward()
        if s == start:  # gradient-flow canary: the #1 silent no-op is a connector that receives no gradient
            gn = math.sqrt(sum(float(p.grad.float().norm() ** 2) for p in conn.parameters() if p.grad is not None))
            trunk_grads = sum(p.grad is not None for p in model.parameters())
            led["canary"] = {"connector_grad_norm": gn, "connector_params_with_grad": sum(p.grad is not None for p in conn.parameters()),
                             "trunk_params_with_grad": trunk_grads}
            assert math.isfinite(gn) and gn > 0, "CANARY FAILED: connector got no gradient through the trunk"
            assert trunk_grads == 0, "CANARY FAILED: trunk received gradients"
        torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0); opt.step(); sched.step()
        ntok += sum(len(data.encode(r)[0]) + a.K for r in rows)
        if s % 25 == 0 or s == steps - 1:
            rec = {"step": s, "task": round(task.item(), 4), "ground": round(ground.item(), 4), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - tt, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if (s + 1) % 200 == 0 and not a.smoke:
            torch.save({"conn": conn.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s + 1}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if (time.time() - tt) / 60 > a.max_minutes:
            led["stopped_early_at_step"] = s + 1; torch.save({"conn": conn.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s + 1}, ck); break
    led["train_s"] = round(time.time() - tt, 1); led["train_tokens"] = ntok; led["tok_per_s"] = round(ntok / max(1e-9, led["train_s"]), 1)
    led["peak_vram_gb_train"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); led["steps"] = steps
    fp1 = trunk_fingerprint(model); led["trunk_fingerprint_post"] = fp1; led["base_drift_zero"] = fp0 == fp1
    from safetensors.torch import save_file
    art = os.path.join(a.out, "connector.safetensors"); save_file({k: v.detach().contiguous() for k, v in conn.state_dict().items()}, art)
    led["artifact"] = art; led["artifact_sha256"] = hashlib.sha256(open(art, "rb").read()).hexdigest()
    y = [r["answer"] == "yes" for r in val]; ev = {}; P = {}
    for mode in ("connector", "shuffle", "nosensor"):
        ev[mode], P[mode] = evaluate(model, conn, data, val, mode, emb)
    led["eval_bf16"] = ev
    led["paired_acc_diff"] = {"connector_minus_nosensor": boot_diff(y, P["connector"], P["nosensor"]),
                              "connector_minus_shuffle": boot_diff(y, P["connector"], P["shuffle"])}
    # independent small-model baseline (review §7): 80-way multi-label logistic probe on mean-pooled frozen SigLIP features
    names = json.load(open(f"{a.data}/meta.json"))["names"]; ci = {n: i for i, n in enumerate(names)}
    tr_imgs = sorted({r["doc_id"] for r in data.split("train")}); Xtr = data.feats[[data.idx[i] for i in tr_imgs]].float().mean(1).cuda()
    present = {r["doc_id"]: r["present"] for r in data.rows}; Ytr = torch.zeros(len(tr_imgs), len(names), device="cuda")
    for j, i in enumerate(tr_imgs):
        for n in present[i]: Ytr[j, ci[n]] = 1
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; lin = nn.Linear(Xtr.shape[1], len(names)).cuda(); o2 = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(500):
        o2.zero_grad(); F.binary_cross_entropy_with_logits(lin((Xtr - mu) / sd), Ytr).backward(); o2.step()
    with torch.no_grad():
        Xv = data.feats[[data.idx[r["doc_id"]] for r in val]].float().mean(1).cuda()
        pl = torch.sigmoid(lin((Xv - mu) / sd))[torch.arange(len(val)), torch.tensor([ci[r["category"]] for r in val])].tolist()
    led["eval_linear_probe"] = {"acc": round(sum((p > 0.5) == t for p, t in zip(pl, y)) / len(y), 4), "auroc": round(auroc(y, pl), 4)}
    led["paired_acc_diff"]["connector_minus_linear_probe"] = boot_diff(y, P["connector"], pl)
    # question-only prior baseline: P(yes | category) from train rows (detects label leakage through the question text)
    from collections import defaultdict
    cy = defaultdict(lambda: [1, 2])
    for r in data.split("train"): cy[r["category"]][0] += r["answer"] == "yes"; cy[r["category"]][1] += 1
    pp = [cy[r["category"]][0] / cy[r["category"]][1] for r in val]
    led["eval_category_prior"] = {"acc": round(sum((p > 0.5) == t for p, t in zip(pp, y)) / len(y), 4), "auroc": round(auroc(y, pp), 4)}
    led["paired_acc_diff"]["connector_minus_category_prior"] = boot_diff(y, P["connector"], pp)
    # rule (c): same connector, trunk weights at deployment precision (ggml block quant round-trip of all 2-D trunk weights)
    led["deploy_precision"] = {}
    if not a.smoke or a.quants:
        sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for qn in [q for q in a.quants.split(",") if q]:
            tq = time.time(); nq = quant_roundtrip(model, qn, a.gguf_py)
            e, pq = evaluate(model, conn, data, val, "connector", emb)
            agree = sum((p > 0.5) == (b > 0.5) for p, b in zip(pq, P["connector"])) / len(pq)
            tv = sum(abs(p - b) for p, b in zip(pq, P["connector"])) / len(pq)
            led["deploy_precision"][qn] = {**e, "tensors_quantized": nq, "top_answer_agreement_vs_bf16": round(agree, 4),
                                           "mean_tv_vs_bf16": round(tv, 4), "gate_pass": agree >= 0.961 and tv <= 0.033,
                                           "acc_diff_vs_bf16": boot_diff(y, pq, P["connector"]), "s": round(time.time() - tq, 1)}
            model.load_state_dict(sd0)
            print(json.dumps({qn: led["deploy_precision"][qn]}), flush=True)
    led["peak_vram_gb_total"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(led, open(os.path.join(a.out, "ledger.json"), "w"), indent=1); print(json.dumps(led, indent=1))


if __name__ == "__main__":
    main()
