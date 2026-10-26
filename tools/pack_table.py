#!/usr/bin/env python3
"""Write table.bin, the per-layer per-tensor format table, beside a pack.

usage: pack_table.py PACK_DIR

From manifest.json alone (the Q8 pack and the quant pack alike). u32 LE, one
32-word row per layer: [0] the layer file's bytes (layers/blk.N.bin), [1..20]
each tensor's offset in that file (< 2^28; word 0 is untagged) | fm << 28, in the engine's Offs order
(below; 0 where the layer shape has no such tensor). fm is the registry's
(tools/formats.tsv) for the tensor's src_type; a manifest's own "fmt" is
ignored. The engine reads word 0 to load each layer; pass_mail the rest.
"""

import json
import struct
import sys
from pathlib import Path

from formats import FMT

OFFS = [
    "attn_norm.weight",
    "post_attention_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_a",
    "ssm_dt.bias",
    "ssm_conv1d.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
]
ROW = 32


def rows(m):
    out = []
    for i in range(m["config"]["n_layers"]):
        p = f"blk.{i}."
        ts = {k[len(p) :]: v for k, v in m["tensors"].items() if k.startswith(p)}
        if set(ts) - set(OFFS):
            raise SystemExit(
                f"{p}: tensors outside the table: {sorted(set(ts) - set(OFFS))}"
            )
        lo = min(v["offset"] for v in ts.values())
        n = max(v["offset"] + v["bytes"] for v in ts.values()) - lo
        w = [n] + [0] * (ROW - 1)
        for j, k in enumerate(OFFS):
            if k in ts:
                if ts[k]["offset"] - lo >= 1 << 28:
                    raise SystemExit(f"{p}{k}: offset {ts[k]['offset'] - lo} >= 2^28")
                w[1 + j] = ts[k]["offset"] - lo | FMT[ts[k]["src_type"]] << 28
        out.append(w)
    return out


def write(pack):
    pack = Path(pack)
    r = rows(json.load(open(pack / "manifest.json")))
    (pack / "table.bin").write_bytes(b"".join(struct.pack(f"<{ROW}I", *w) for w in r))
    return r


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    r = write(sys.argv[1])
    print(
        f"{len(r)} layers -> {Path(sys.argv[1]) / 'table.bin'}; layer bytes "
        f"{sorted({w[0] for w in r})}"
    )
