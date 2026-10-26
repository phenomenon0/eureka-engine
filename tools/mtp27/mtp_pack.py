#!/usr/bin/env python3
"""mtp_pack.py VOCAB.json OUT.bin: the MTP drafter's weights for LLB_MTP (Stage 2, "packer row 64"): the
official sidecar's blk.64 (the NextN block, llama.cpp src/models/qwen35.cpp graph_mtp) and the draft-vocab
slice of the trunk head (output.weight rows of the ids, HyperQwen's idea: prepare/build_draft_vocab.py:91-103),
raw GGUF blocks (the pack keeps Q4_K / Q6_K raw too). Layout: "MTP1", u32 N, N x (u64 off, u64 bytes), the
entries 256-aligned in ORDER, the last two the head slice (Q6_K rows, vocab order) and the ids (u32)."""

import json, os, struct, sys
import numpy as np
import gguf

SIDE = os.environ["MTP_SIDE"]
TRUNK = os.environ["MTP_TRUNK"]
ORDER = [  # name, type (host spec_mtp_load checks the bytes)
    ("nextn.enorm", "F32"),
    ("nextn.hnorm", "F32"),
    ("nextn.eh_proj", "Q4_K"),
    ("attn_norm", "F32"),
    ("attn_q", "Q6_K"),
    ("attn_k", "Q6_K"),
    ("attn_v", "Q6_K"),
    ("attn_q_norm", "F32"),
    ("attn_k_norm", "F32"),
    ("attn_output", "Q6_K"),
    ("post_attention_norm", "F32"),
    ("ffn_gate", "Q4_K"),
    ("ffn_up", "Q4_K"),
    ("ffn_down", "Q4_K"),
    ("nextn.shared_head_norm", "F32"),
]


def tensors(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def main():
    ids = json.load(open(sys.argv[1]))
    side, trunk = tensors(SIDE), tensors(TRUNK)
    blobs = []
    for n, ty in ORDER:
        t = side[f"blk.64.{n}.weight"]
        assert t.tensor_type.name == ty, (n, t.tensor_type.name)
        blobs.append(bytes(t.data))
    h = trunk["output.weight"]
    assert h.tensor_type.name == "Q6_K"
    rows = np.asarray(h.data).reshape(248320, -1)
    assert rows.shape[0] == 248320 and rows.shape[1] == 4200, rows.shape
    blobs.append(rows[np.asarray(ids)].tobytes())
    blobs.append(np.asarray(ids, dtype=np.uint32).tobytes())
    off = (8 + 16 * len(blobs) + 255) // 256 * 256
    tab = []
    for b in blobs:
        tab.append((off, len(b)))
        off = (off + len(b) + 255) // 256 * 256
    with open(sys.argv[2], "wb") as f:
        f.write(b"MTP1" + struct.pack("<I", len(blobs)))
        for o, n in tab:
            f.write(struct.pack("<QQ", o, n))
        for (o, n), b in zip(tab, blobs):
            f.seek(o)
            f.write(b)
    for (n, _), (o, b) in zip(ORDER + [("head", 0), ("ids", 0)], tab):
        print(f"{n:24s} {o:12d} {b:12d}")


if __name__ == "__main__":
    main()
