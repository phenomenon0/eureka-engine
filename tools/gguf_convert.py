#!/usr/bin/env python3
"""GGUF -> llama-bend F32 pack (pack.bin + manifest.json + ops.md).

--f16 keeps F16 source tensors as raw f16le bytes (format llama-bend-pack-f16/v0;
each tensor records its dtype); every other type is still dequantized to F32.

stdlib + numpy only. Raw GGUF reader per the spec in ggml/docs/gguf.md.

usage: gguf_convert.py MODEL.gguf OUT_DIR [--f16]
"""

import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

ALIGN = 64  # pack alignment for every tensor

# GGUF metadata value types -> struct format (8 = string, 9 = array handled separately)
KV_FMT = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

# ggml tensor types this reader can dequantize: id -> (name, block elems, block bytes)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    23: ("IQ4_XS", 256, 136),
    30: ("BF16", 1, 2),
    42: ("Q2_0", 64, 18),
    142: ("PQ2_0", 128, 34),
}


class Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def unpack(self, fmt):
        vals = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return vals[0] if len(vals) == 1 else vals

    def string(self):
        n = self.unpack("Q")
        s = bytes(self.buf[self.pos : self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.unpack("IQ")
            if et in KV_FMT:  # numeric arrays: one bulk unpack
                arr = list(struct.unpack_from(f"<{n}{KV_FMT[et]}", self.buf, self.pos))
                self.pos += n * struct.calcsize(KV_FMT[et])
                return arr
            return [self.value(et) for _ in range(n)]
        return self.unpack(KV_FMT[t])


def read_gguf(path):
    """Returns (kv dict, tensor list, mmap). Each tensor: name, type, shape (row-major), offset (absolute)."""
    f = open(path, "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    r = Reader(buf)
    magic, version, n_tensors, n_kv = r.unpack("4sIQQ")
    if magic != b"GGUF" or version not in (2, 3):
        raise ValueError(f"not a GGUF v2/v3 file: magic={magic!r} version={version}")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.unpack("I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.unpack("I")
        ne = [r.unpack("Q") for _ in range(nd)]
        ttype, off = r.unpack("IQ")
        # GGUF dims are ggml ne[] (fastest first); row-major shape is the reverse.
        tensors.append(
            {"name": name, "type": ttype, "shape": ne[::-1], "rel_offset": off}
        )
    align = kv.get("general.alignment", 32)
    data_start = (r.pos + align - 1) // align * align
    for t in tensors:
        t["offset"] = data_start + t["rel_offset"]
    return kv, tensors, buf


def dequant(buf, t):
    """Dequantize one tensor to a row-major float32 ndarray. Unknown type -> hard error."""
    if t["type"] not in GGML_TYPES:
        raise NotImplementedError(
            f"unknown/unsupported ggml type {t['type']} for tensor {t['name']}"
        )
    tname, blk, bsz = GGML_TYPES[t["type"]]
    n = int(np.prod(t["shape"]))
    assert n % blk == 0, (t["name"], n, blk)
    raw = np.frombuffer(buf, dtype=np.uint8, count=n // blk * bsz, offset=t["offset"])
    if tname == "F32":
        out = raw.view("<f4").copy()
    elif tname == "F16":
        out = raw.view("<f2").astype(np.float32)
    elif tname == "BF16":
        out = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    elif tname == "Q4_K":  # 256-weight super-blocks, see tools/q4k.py
        import q4k

        out = q4k.dequant_blocks(raw)
    elif tname == "IQ4_XS":  # 256-weight super-blocks, see tools/iq4xs.py
        import iq4xs

        out = iq4xs.dequant_blocks(raw)
    else:  # Q8_0: per 32-elem block, f16 scale d then 32 int8; x = d * q (same as ggml's dequantize_row_q8_0)
        b = raw.reshape(-1, 34)
        d = b[:, :2].copy().view("<f2").astype(np.float32)
        q = b[:, 2:].copy().view(np.int8).astype(np.float32)
        out = (q * d).reshape(-1)
    return out.reshape(t["shape"])


def build_config(kv, tensors):
    a = kv["general.architecture"]
    if a == "llama":
        return build_config_llama(kv, tensors)
    if a == "qwen35":
        return build_config_qwen35(kv, tensors)
    raise NotImplementedError(
        f"architecture {a!r}: only 'llama' and 'qwen35' are specified for v0"
    )


def build_config_qwen35(kv, tensors):
    """Qwen3.5 hybrid (gated delta net linear attention + gated full attention).

    Rope is the MRoPE/NEOX variant over the first `rope_dim` (=64) of each 256-wide
    full-attention head; for the text-only path all three position ids are equal, so
    this reduces to NEOX rotation of pairs (i, i+32), i in [0,32), base 1e7, with the
    remaining 192 dims copied through.
    """
    a = "qwen35"
    k = lambda s: kv[f"{a}.{s}"]
    names = {t["name"] for t in tensors}
    D = k("embedding_length")
    n_heads = k("attention.head_count")
    head_dim = k("attention.key_length")
    tok = "tokenizer.ggml."
    cfg = {
        "architecture": "qwen3_5",
        "n_layers": k("block_count"),
        "n_heads": n_heads,
        "n_kv_heads": k("attention.head_count_kv"),
        "d_model": D,
        "head_dim": head_dim,
        "d_ff": k("feed_forward_length"),
        "vocab_size": kv.get(f"{a}.vocab_size", len(kv[tok + "tokens"])),
        "context_length": k("context_length"),
        "rope_freq_base": kv.get(f"{a}.rope.freq_base", 10000.0),
        "rope_dim": kv.get(f"{a}.rope.dimension_count", head_dim),
        "rope_sections": kv.get(f"{a}.rope.dimension_sections", [0, 0, 0, 0]),
        "rope_type": "neox_mrope_text",  # pairs (i, i+rope_dim/2), dims >= rope_dim copied
        "rms_norm_eps": k("attention.layer_norm_rms_epsilon"),
        "full_attention_interval": k("full_attention_interval"),
        # gated delta net (linear attention) dimensions
        "ssm_conv_kernel": k("ssm.conv_kernel"),
        "ssm_state_size": k("ssm.state_size"),
        "ssm_group_count": k("ssm.group_count"),
        "ssm_time_step_rank": k("ssm.time_step_rank"),
        "ssm_inner_size": k("ssm.inner_size"),
        "tied_embeddings": "output.weight" not in names,
        "bos_token_id": kv.get(tok + "bos_token_id"),
        "eos_token_id": kv.get(tok + "eos_token_id"),
        "eot_token_id": kv.get(tok + "eot_token_id"),
        "pad_token_id": kv.get(tok + "padding_token_id"),
        "add_bos_token": kv.get(tok + "add_bos_token"),
        "tokenizer_model": kv.get(tok + "model"),
        "tokenizer_pre": kv.get(tok + "pre"),
        "source_file_type": kv.get("general.file_type"),
    }
    # derived dims, so an engine can read them without recomputing
    S = cfg["ssm_state_size"]
    cfg["linear_key_dim"] = S * cfg["ssm_group_count"]              # 128*16 = 2048
    cfg["linear_value_dim"] = cfg["ssm_inner_size"]                  # 4096
    cfg["linear_conv_dim"] = 2 * cfg["linear_key_dim"] + cfg["linear_value_dim"]  # 8192
    cfg["linear_value_head_dim"] = cfg["ssm_inner_size"] // cfg["ssm_time_step_rank"]  # 128
    return cfg


def build_config_llama(kv, tensors):
    a = kv["general.architecture"]
    k = lambda s: kv[f"{a}.{s}"]
    names = {t["name"] for t in tensors}
    n_heads = k("attention.head_count")
    d_model = k("embedding_length")
    head_dim = kv.get(f"{a}.attention.key_length", d_model // n_heads)
    tok = "tokenizer.ggml."
    return {
        "architecture": a,
        "n_layers": k("block_count"),
        "n_heads": n_heads,
        "n_kv_heads": kv.get(f"{a}.attention.head_count_kv", n_heads),
        "d_model": d_model,
        "head_dim": head_dim,
        "d_ff": k("feed_forward_length"),
        "vocab_size": kv.get(f"{a}.vocab_size", len(kv[tok + "tokens"])),
        "context_length": k("context_length"),
        "rope_freq_base": kv.get(f"{a}.rope.freq_base", 10000.0),
        "rope_dim": kv.get(f"{a}.rope.dimension_count", head_dim),
        "rope_type": "norm",  # llama.cpp LLAMA_ROPE_TYPE_NORM for arch=llama: rotate adjacent pairs (x[2i], x[2i+1])
        "rope_scaling": kv.get(f"{a}.rope.scaling.type", None),
        "rms_norm_eps": k("attention.layer_norm_rms_epsilon"),
        "tied_embeddings": "output.weight" not in names,
        "bos_token_id": kv.get(tok + "bos_token_id"),
        "eos_token_id": kv.get(tok + "eos_token_id"),
        "eot_token_id": kv.get(tok + "eot_token_id"),
        "pad_token_id": kv.get(tok + "padding_token_id"),
        "add_bos_token": kv.get(tok + "add_bos_token"),
        "tokenizer_model": kv.get(tok + "model"),
        "tokenizer_pre": kv.get(tok + "pre"),
        "source_file_type": kv.get("general.file_type"),
    }


def check_shapes(cfg, tensors):
    """Every tensor shape must be derivable from the config. Returns list of mismatches."""
    if cfg["architecture"] == "qwen3_5":
        return check_shapes_qwen35(cfg, tensors)
    return check_shapes_llama(cfg, tensors)


def check_shapes_qwen35(cfg, tensors):
    D, F, V = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]
    H, KV, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    S, GK, TR = cfg["ssm_state_size"], cfg["ssm_group_count"], cfg["ssm_time_step_rank"]
    kd, vd = cfg["linear_key_dim"], cfg["linear_value_dim"]
    conv_dim, kc = cfg["linear_conv_dim"], cfg["ssm_conv_kernel"]
    vhd = cfg["linear_value_head_dim"]
    qdim = 2 * hd * H  # query + gate, interleaved per head (256 q then 256 gate)
    want = {
        "token_embd.weight": [V, D],
        "output_norm.weight": [D],
        "output.weight": [V, D],
    }
    interval = cfg["full_attention_interval"]
    for i in range(cfg["n_layers"]):
        p = f"blk.{i}."
        per = {
            "attn_norm.weight": [D],
            "post_attention_norm.weight": [D],
            "ffn_gate.weight": [F, D],
            "ffn_up.weight": [F, D],
            "ffn_down.weight": [D, F],
        }
        if (i + 1) % interval == 0:  # full attention layer
            per.update(
                {
                    "attn_q.weight": [qdim, D],
                    "attn_k.weight": [hd * KV, D],
                    "attn_v.weight": [hd * KV, D],
                    "attn_output.weight": [D, hd * H],
                    "attn_q_norm.weight": [hd],
                    "attn_k_norm.weight": [hd],
                }
            )
        else:  # linear attention (gated delta net) layer
            per.update(
                {
                    "attn_qkv.weight": [2 * kd + vd, D],
                    "attn_gate.weight": [vd, D],
                    "ssm_conv1d.weight": [conv_dim, kc],
                    "ssm_dt.bias": [TR],
                    "ssm_a": [TR],
                    "ssm_beta.weight": [TR, D],
                    "ssm_alpha.weight": [TR, D],
                    "ssm_norm.weight": [vhd],
                    "ssm_out.weight": [D, vd],
                }
            )
        want.update({p + s: sh for s, sh in per.items()})
    if cfg["tied_embeddings"]:
        del want["output.weight"]
    got = {t["name"]: t["shape"] for t in tensors}
    bad = [
        f"{n}: want {w} got {got.get(n)}" for n, w in want.items() if got.get(n) != w
    ]
    bad += [
        f"{n}: unexpected tensor {s}"
        for n, s in got.items()
        if n not in want and not n.startswith("rope_")
    ]
    if cfg["rope_dim"] > hd or cfg["rope_dim"] % 2:
        bad.append(f"rope_dim {cfg['rope_dim']} incompatible with head_dim {hd}")
    return bad


def check_shapes_llama(cfg, tensors):
    D, F, V, H, KV, hd = (
        cfg[x]
        for x in ("d_model", "d_ff", "vocab_size", "n_heads", "n_kv_heads", "head_dim")
    )
    want = {
        "token_embd.weight": [V, D],
        "output_norm.weight": [D],
        "output.weight": [V, D],
    }
    per = {
        "attn_norm.weight": [D],
        "attn_q.weight": [H * hd, D],
        "attn_k.weight": [KV * hd, D],
        "attn_v.weight": [KV * hd, D],
        "attn_output.weight": [D, H * hd],
        "ffn_norm.weight": [D],
        "ffn_gate.weight": [F, D],
        "ffn_up.weight": [F, D],
        "ffn_down.weight": [D, F],
    }
    for i in range(cfg["n_layers"]):
        want.update({f"blk.{i}.{s}": v for s, v in per.items()})
    if cfg["tied_embeddings"]:
        del want["output.weight"]
    got = {t["name"]: t["shape"] for t in tensors}
    bad = [
        f"{n}: want {w} got {got.get(n)}" for n, w in want.items() if got.get(n) != w
    ]
    bad += [
        f"{n}: unexpected tensor {s}"
        for n, s in got.items()
        if n not in want and not n.startswith("rope_")
    ]
    if cfg["rope_dim"] > hd or cfg["rope_dim"] % 2:
        bad.append(f"rope_dim {cfg['rope_dim']} incompatible with head_dim {hd}")
    return bad


def ops_md(cfg):
    if cfg["architecture"] == "qwen3_5":
        return ops_md_qwen35(cfg)
    return ops_md_llama(cfg)


def ops_md_qwen35(cfg):
    D, F, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"], cfg["n_layers"]
    H, KV, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    S, GK, TR = cfg["ssm_state_size"], cfg["ssm_group_count"], cfg["ssm_time_step_rank"]
    kd, vd, conv_dim, kc = (
        cfg["linear_key_dim"],
        cfg["linear_value_dim"],
        cfg["linear_conv_dim"],
        cfg["ssm_conv_kernel"],
    )
    vhd = cfg["linear_value_head_dim"]
    n_lin = sum(1 for i in range(L) if (i + 1) % cfg["full_attention_interval"] != 0)
    n_full = L - n_lin
    lines = [
        "# ops.md — Qwen3.5 hybrid op inventory (single-token decode step)",
        "",
        f"Config: L={L} ({n_lin} linear + {n_full} full) D={D} H={H} KV={KV} head_dim={hd} "
        f"d_ff={F} vocab={V} rope_dim={cfg['rope_dim']} base={cfg['rope_freq_base']} "
        f"eps={cfg['rms_norm_eps']} tied={cfg['tied_embeddings']}",
        f"Linear (gated delta net): state={S}x{S} heads={TR} groups={GK} key_dim={kd} "
        f"value_dim={vd} conv_dim={conv_dim} conv_kernel={kc}",
        "",
        "All weights row-major [out, in]; gemv y[i] = sum_j W[i,j] x[j]. All math F32.",
        "Layer i is full attention iff (i+1) % interval == 0, else linear.",
        "",
        "| # | op | math | shapes | count/token |",
        "|---|---|---|---|---|",
    ]
    rows = [
        ("embed lookup", "x = token_embd[tok]", f"token_embd [{V},{D}]", "once"),
        ("attn_norm", "h = rms(x)*w", f"x [{D}]", f"{L}x"),
        (
            "linear: qkvz",
            "qkv=Wqkv h; z=Wgate h",
            f"Wqkv [{2 * kd + vd},{D}] -> [{conv_dim}]; Wgate [{vd},{D}] -> [{vd}]",
            f"{n_lin}x",
        ),
        (
            "linear: beta/alpha",
            "b=sig(Wb h); a=softplus(Wa h + dt); g=-exp(A)*a",
            f"Wb,Wa [{TR},{D}]; dt,A [{TR}]",
            f"{n_lin}x",
        ),
        (
            "linear: conv1d+silu",
            "y=silu(sum_k conv[k,c]*x[t-3+k,c])",
            f"conv [{conv_dim},{kc}] over window {kc}",
            f"{n_lin}x",
        ),
        (
            "linear: q/k l2norm + repeat",
            "q,k /= sqrt(sum sq + eps); heads {GK}->{TR}",
            f"q,k [{S},{GK}] -> [{S},{TR}]; scale 1/sqrt({S})",
            f"{n_lin}x",
        ),
        (
            "linear: delta recurrence",
            "S=S*exp(g)+(S^T k - v)*b outer k; o=S^T q",
            f"S [{S},{S}] per value head ({TR} heads)",
            f"{n_lin}x",
        ),
        (
            "linear: gated rmsnorm",
            "y=rms(o)*w*silu(z)",
            f"o,z [{vhd},{TR}]; w [{vhd}]",
            f"{n_lin}x",
        ),
        ("linear: ssm_out", "x = Wout y", f"Wout [{D},{vd}]", f"{n_lin}x"),
        (
            "full: qg",
            "qg=Wq h; q=qg[0:{hd}], gate=qg[{hd}:{2 * hd}] per head",
            f"Wq [{2 * hd * H},{D}]",
            f"{n_full}x",
        ),
        ("full: k,v", "k=Wk h; v=Wv h", f"Wk,Wv [{hd * KV},{D}]", f"{n_full}x"),
        (
            "full: q/k norm",
            "rms per head over {hd}",
            f"q_norm,k_norm [{hd}]",
            f"{n_full}x",
        ),
        (
            "full: rope (neox, partial)",
            "rotate (i,i+{rope_dim}/2) for i<{rope_dim}/2; copy rest",
            f"rope_dim={cfg['rope_dim']} of {hd}",
            f"{n_full}x",
        ),
        (
            "full: attention",
            "softmax(q.K^T/sqrt(hd)) V; GQA",
            f"q [{H},{hd}] reads kv head h//{H // KV}",
            f"{n_full}x",
        ),
        ("full: out gate", "x = (attn) * sigmoid(gate)", f"gate [{hd * H}]", f"{n_full}x"),
        ("full: wo", "x = Wo x", f"Wo [{D},{hd * H}]", f"{n_full}x"),
        ("attn residual", "x += attn_out", f"[{D}]", f"{L}x"),
        ("attn_post_norm", "h = rms(x)*w", f"[{D}]", f"{L}x"),
        (
            "ffn",
            "x += Wd (silu(Wg h) * Wu h)",
            f"Wg,Wu [{F},{D}]; Wd [{D},{F}]",
            f"{L}x",
        ),
        ("final rms_norm + logits", "z = Wout rms(x)", f"[{V},{D}]", "once"),
    ]
    lines += [
        f"| {i} | {a} | `{b}` | {c} | {d} |" for i, (a, b, c, d) in enumerate(rows, 1)
    ]
    tot = 4 * (
        L * (D * D * 0 + F * D * 3)
        + n_lin * ((2 * kd + vd) * D + vd * D + 2 * TR * D + conv_dim * kc + 2 * TR + D * vd)
        + n_full * (2 * hd * H * D + 2 * hd * KV * D + D * hd * H)
        + 2 * V * D
    )
    lines += ["", f"- Rough per-token weight traffic at F32: {tot / 2**30:.2f} GiB."]
    return "\n".join(lines) + "\n"


def ops_md_llama(cfg):
    D, F, V, H, KV, hd, L = (
        cfg[x]
        for x in (
            "d_model",
            "d_ff",
            "vocab_size",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "n_layers",
        )
    )
    Q, K = H * hd, KV * hd
    rows = [
        (
            "embed lookup",
            "x = token_embd[tok]",
            f"token_embd [{V},{D}] -> x [{D}]",
            "once",
        ),
        (
            "rms_norm (attn)",
            "h = x / sqrt(mean(x^2) + eps) * attn_norm",
            f"x [{D}], w [{D}], eps={cfg['rms_norm_eps']}",
            f"{L}x",
        ),
        (
            "q gemv",
            "q = Wq h",
            f"Wq [{Q},{D}] . h [{D}] -> q [{Q}] = {H} heads x {hd}",
            f"{L}x",
        ),
        (
            "k gemv",
            "k = Wk h",
            f"Wk [{K},{D}] . h [{D}] -> k [{K}] = {KV} heads x {hd}",
            f"{L}x",
        ),
        (
            "v gemv",
            "v = Wv h",
            f"Wv [{K},{D}] . h [{D}] -> v [{K}] = {KV} heads x {hd}",
            f"{L}x",
        ),
        (
            "rope (q, k)",
            "rotate pairs (x[2i],x[2i+1]) by pos*base^(-2i/rope_dim)",
            f"q [{H},{hd}], k [{KV},{hd}], rope_dim={cfg['rope_dim']}, base={cfg['rope_freq_base']}, type={cfg['rope_type']}",
            f"{L}x",
        ),
        (
            "kv-cache append",
            "K[pos]=k, V[pos]=v",
            f"per layer K,V [ctx,{KV},{hd}]",
            f"{L}x",
        ),
        (
            "attention scores",
            "s = q_h . K_g^T / sqrt(hd), g = h // (H/KV)",
            f"q [{H},{hd}], K [T,{KV},{hd}] -> s [{H},T]; GQA group={H // KV}; scale=1/sqrt({hd})",
            f"{L}x",
        ),
        ("causal softmax", "p = softmax(s) over t <= pos", f"s [{H},T]", f"{L}x"),
        (
            "attention weighted sum",
            "o_h = p_h . V_g",
            f"p [{H},T], V [T,{KV},{hd}] -> o [{Q}]",
            f"{L}x",
        ),
        ("o gemv + residual", "x += Wo o", f"Wo [{D},{Q}] . o [{Q}] -> [{D}]", f"{L}x"),
        ("rms_norm (ffn)", "h = rmsnorm(x) * ffn_norm", f"x [{D}], w [{D}]", f"{L}x"),
        ("gate gemv", "g = Wg h", f"Wg [{F},{D}] . h [{D}] -> [{F}]", f"{L}x"),
        ("up gemv", "u = Wu h", f"Wu [{F},{D}] . h [{D}] -> [{F}]", f"{L}x"),
        ("silu * up", "a = g * sigmoid(g) * u", f"[{F}]", f"{L}x"),
        (
            "down gemv + residual",
            "x += Wd a",
            f"Wd [{D},{F}] . a [{F}] -> [{D}]",
            f"{L}x",
        ),
        ("final rms_norm", "h = rmsnorm(x) * output_norm", f"x [{D}], w [{D}]", "once"),
        (
            "logits gemv",
            "z = Wout h",
            f"{'token_embd' if cfg['tied_embeddings'] else 'output'} [{V},{D}] . h [{D}] -> z [{V}]",
            "once",
        ),
        ("argmax", "tok = argmax(z)", f"z [{V}]", "once"),
    ]
    out = [
        "# ops.md — op inventory for this model (single-token decode step)",
        "",
        f"Config: L={L} D={D} H={H} KV={KV} head_dim={hd} d_ff={F} vocab={V} "
        f"rope_base={cfg['rope_freq_base']} rope_dim={cfg['rope_dim']} eps={cfg['rms_norm_eps']} "
        f"tied={cfg['tied_embeddings']}",
        "",
        "All weights row-major [out, in]; gemv is y[i] = sum_j W[i,j] x[j]. All math F32.",
        "",
        "| # | op | math | shapes | count/token |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {i} | {a} | `{b}` | {c} | {d} |" for i, (a, b, c, d) in enumerate(rows, 1)
    ]
    out += [
        "",
        "Notes:",
        "- RoPE is the llama.cpp NORM variant: GGUF `llama` checkpoints have Q/K rows permuted at convert time so",
        "  rotation pairs are adjacent elements (x[2i], x[2i+1]) within each head, theta_i = pos * base^(-2i/rope_dim).",
        "  Do NOT use the HF/NeoX half-split rotation on these weights.",
        f"- GQA: query head h reads kv head h // {H // KV}.",
        f"- Per-token weight traffic at F32: {4 * (L * (Q * D + 2 * K * D + D * Q + 3 * F * D) + V * D) / 2**30:.2f} GiB (excl. embedding row).",
    ]
    return "\n".join(out) + "\n"


def convert(src, out_dir, f16=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kv, tensors, buf = read_gguf(src)
    cfg = build_config(kv, tensors)
    bad = check_shapes(cfg, tensors)
    if bad:
        raise SystemExit("shape check failed:\n  " + "\n  ".join(bad))
    unknown = sorted({t["type"] for t in tensors} - GGML_TYPES.keys())
    if unknown:  # stop before writing anything
        raise SystemExit(f"STOP: unknown ggml tensor types {unknown}")
    manifest = {
        "format": "llama-bend-pack-f16/v0" if f16 else "llama-bend-pack-f32/v0",
        "alignment": ALIGN,
        "dtype": "f16le" if f16 else "f32le",
        "source": {"file": str(src), "types": {}},
        "config": cfg,
        "tensors": {},
    }
    if f16:  # head tensors first (the F16 GGUF puts output_norm last, past U32 offsets)
        tensors = sorted(tensors, key=lambda t: t["name"].startswith("blk."))
    pack_hash, pos = hashlib.sha256(), 0
    # ponytail: tensors in GGUF file order, one at a time -> peak RAM ~ largest tensor x2
    with open(out_dir / "pack.bin", "wb") as f:
        for t in tensors:
            pad = (-pos) % ALIGN
            f.write(b"\0" * pad)
            pack_hash.update(b"\0" * pad)
            pos += pad
            keep = f16 and GGML_TYPES[t["type"]][0] == "F16"
            if keep:  # the source bits, untouched
                data = bytes(buf[t["offset"] : t["offset"] + 2 * int(np.prod(t["shape"]))])
                # as q8_pack.py's scales: the engine's F16 contract is finite halves
                assert np.isfinite(np.frombuffer(data, "<f2")).all(), f"{t['name']}: nonfinite f16 weight"
            else:
                data = np.ascontiguousarray(dequant(buf, t), dtype="<f4").tobytes()
            f.write(data)
            pack_hash.update(data)
            manifest["tensors"][t["name"]] = {
                "shape": t["shape"],
                "offset": pos,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "src_type": GGML_TYPES[t["type"]][0],
            }
            if f16:
                manifest["tensors"][t["name"]]["dtype"] = "f16le" if keep else "f32le"
            tn = GGML_TYPES[t["type"]][0]
            manifest["source"]["types"][tn] = manifest["source"]["types"].get(tn, 0) + 1
            pos += len(data)
    manifest["pack_bytes"] = pos
    manifest["pack_sha256"] = pack_hash.hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out_dir / "ops.md").write_text(ops_md(cfg))
    return manifest


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[3:] not in ([], ["--f16"]):
        raise SystemExit(__doc__)
    m = convert(sys.argv[1], sys.argv[2], sys.argv[3:] == ["--f16"])
    print(json.dumps(m["config"], indent=1))
    print(
        f"{len(m['tensors'])} tensors, {m['pack_bytes']} bytes, types {m['source']['types']}, pack_sha256 {m['pack_sha256']}"
    )
