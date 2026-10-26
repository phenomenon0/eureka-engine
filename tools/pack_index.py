#!/usr/bin/env python3
"""Write a C-friendly tensor index beside a llama-bend pack.

usage: pack_index.py PACK_DIR

Reads manifest.json and writes index.tsv: one line per tensor,
    name<TAB>offset<TAB>bytes<TAB>rows<TAB>cols<TAB>src_type<TAB>row_bytes<TAB>fmt
rows/cols are 1 for vectors; row_bytes is the stored bytes per matrix row
(0 for F32 vectors); fmt is the registry fm of src_type (tools/formats.tsv;
a manifest's own "fmt" is ignored: pre-registry packs wrote Q5_K = 8). This lets a small C/CUDA engine
load a pack without a JSON parser.
"""

import json
import sys
from pathlib import Path

from formats import FMT



def main(pack):
    pack = Path(pack)
    m = json.load(open(pack / "manifest.json"))
    out = []
    for name, t in m["tensors"].items():
        sh = t["shape"]
        rows = sh[0] if len(sh) == 2 else 1
        cols = sh[1] if len(sh) == 2 else sh[0]
        rb = t.get("row_bytes", 0 if t["src_type"] == "F32" else 0)
        fmt = FMT.get(t["src_type"], -1)
        out.append(f"{name}\t{t['offset']}\t{t['bytes']}\t{rows}\t{cols}\t{t['src_type']}\t{rb}\t{fmt}")
    cfg = m["config"]
    keys = [
        "architecture", "n_layers", "n_heads", "n_kv_heads", "d_model", "head_dim",
        "d_ff", "vocab_size", "context_length", "rope_freq_base", "rope_dim",
        "rms_norm_eps", "full_attention_interval", "ssm_conv_kernel", "ssm_state_size",
        "ssm_group_count", "ssm_time_step_rank", "ssm_inner_size", "linear_key_dim",
        "linear_value_dim", "linear_conv_dim", "linear_value_head_dim",
    ]
    with open(pack / "index.tsv", "w") as f:
        for k in keys:
            f.write(f"#cfg\t{k}\t{cfg.get(k)}\n")
        f.write("# config\t" + json.dumps({k: cfg.get(k) for k in keys}, separators=(",", ":")) + "\n")
        f.write("# pack_bytes\t" + str(m["pack_bytes"]) + "\n")
        f.write("# pack_sha256\t" + m["pack_sha256"] + "\n")
        f.write("# format\t" + m["format"] + "\n")
        f.write("# name\toffset\tbytes\trows\tcols\tsrc_type\trow_bytes\n")
        f.write("\n".join(out) + "\n")
    print(f"{len(out)} tensors -> {pack / 'index.tsv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
