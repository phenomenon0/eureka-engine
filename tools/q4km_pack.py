#!/usr/bin/env python3
"""GGUF Q4_K_M (mixed Q4_K + Q6_K) -> llama-bend Q4_K_M pack.

usage: q4km_pack.py MODEL-Q4_K_M.gguf OUT_DIR

Same bytes as the GGUF, kept in the GGUF block layout (no row-split): a Q4_K
matrix row of `cols` columns is cols/256 super-blocks of 144 bytes, a Q6_K row
cols/256 super-blocks of 210 bytes. F32 tensors (the norms) are copied as
f32le.

The Q4_K_M mixture stores attn_v and ffn_down as Q6_K on some layers, so a layer
file's size varies layer to layer. To keep the engine's one-layout-per-model
loader simple (engine_gpu.bend), every layer is written at fixed 64-aligned
slots sized for the LARGER (Q6_K) of the two formats, and the smaller Q4_K
tensors are zero-padded to their slot. Offsets within every layer are therefore
identical; only the row word (format + row bytes) varies per tensor, and the
engine reads that word from the manifest (`"rw"` per matrix tensor; Q4_K =
0x70000000 | rb, Q6_K = 0xC0000000 | rb). The head is written tightly: token_embd
is Q4_K, output.weight is Q6_K, both read from pack.bin.

The contract is llama.cpp's bytes. For every Q4_K/Q6_K tensor this pack asserts:
  * the stored bytes are the GGUF's bytes, byte for byte;
  * dequant(stored) == llama.cpp dequantize_row_q4_K / dequantize_row_q6_K;
  * d (and dmin) finite.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q4k
import q6k
from gguf_convert import GGML_TYPES, build_config, check_shapes, read_gguf

ALIGN = 64
K = 256


def rowb(tn, cols):
    return q6k.row_bytes(cols) if tn == "Q6_K" else q4k.row_bytes(cols)


def dequant(data, tn, rows, cols):
    return (q6k if tn == "Q6_K" else q4k).dequant_rows(data, rows, cols)


def ggml_dequant(data, tn):
    return (q6k if tn == "Q6_K" else q4k).ggml_dequant(data)


# per-layer tensor order and the formats each slot may hold
LAYER = [
    ("attn_norm.weight", ("F32",)),
    ("attn_q.weight", ("Q4_K",)),
    ("attn_k.weight", ("Q4_K",)),
    ("attn_v.weight", ("Q4_K", "Q6_K")),
    ("attn_output.weight", ("Q4_K",)),
    ("ffn_norm.weight", ("F32",)),
    ("ffn_gate.weight", ("Q4_K",)),
    ("ffn_up.weight", ("Q4_K",)),
    ("ffn_down.weight", ("Q4_K", "Q6_K")),
]

RW = {"Q4_K": 0x70000000, "Q6_K": 0xC0000000}


def convert(src, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "layers").mkdir(parents=True, exist_ok=False)
    kv, tensors, buf = read_gguf(src)
    cfg = build_config(kv, tensors)
    bad = check_shapes(cfg, tensors)
    if bad:
        raise SystemExit("shape check failed:\n  " + "\n  ".join(bad))
    types = {GGML_TYPES[t["type"]][0] for t in tensors}
    if not types <= {"Q4_K", "Q6_K", "F32"}:
        raise SystemExit(f"STOP: Q4_K_M pack takes Q4_K + Q6_K + F32, got {sorted(types)}")
    byname = {t["name"]: t for t in tensors}

    def tn_of(name):
        return GGML_TYPES[byname[name]["type"]][0]

    def raw_of(name):
        t = byname[name]
        if tn_of(name) == "F32":
            return np.frombuffer(buf, np.uint8, 4 * int(np.prod(t["shape"])), t["offset"])
        rows, cols = t["shape"]
        return np.frombuffer(buf, np.uint8, rows * rowb(tn_of(name), cols), t["offset"])

    # fixed 64-aligned slot offsets for one layer
    slots, pos = {}, 0
    for name, allowed in LAYER:
        full = f"blk.0.{name}"
        t = byname[full]
        if tn_of(full) == "F32":
            size = 4 * int(np.prod(t["shape"]))
        else:
            rows, cols = t["shape"]
            size = rows * max(rowb(a, cols) for a in allowed)
        pos += (-pos) % ALIGN
        slots[name] = pos
        pos += size
    layer_bytes = pos

    manifest = {
        "format": "llama-bend-pack-q4km/v0",
        "alignment": ALIGN,
        "dtype": "q4_k_m-mixed",
        "layer_bytes": layer_bytes,
        "source": {"file": str(src), "types": {}},
        "config": cfg,
        "tensors": {},
    }

    def check(name, data):
        t = byname[name]
        tn = tn_of(name)
        if tn == "F32":
            return
        rows, cols = t["shape"]
        block = q6k.BLOCK_BYTES if tn == "Q6_K" else q4k.BLOCK_BYTES
        di = slice(208, 210) if tn == "Q6_K" else slice(0, 4)
        d = data.reshape(-1, block)[:, di].copy().view(np.uint16)
        assert np.isfinite(d.view(np.float16)).all(), f"{name}: nonfinite d/dmin"
        got = dequant(data, tn, rows, cols).ravel().view(np.uint32)
        want = ggml_dequant(data, tn).ravel().view(np.uint32)
        assert np.array_equal(got, want), f"{name}: decode ({tn}) differs from ggml"

    pack_hash, pos = hashlib.sha256(), 0
    with open(out_dir / "pack.bin", "wb") as f:

        def emit(name, data, shape, tn, rw=None):
            nonlocal pos
            pad = b"\0" * ((-pos) % ALIGN)
            f.write(pad), pack_hash.update(pad)
            pos += len(pad)
            e = {"shape": shape, "offset": pos, "bytes": len(data), "src_type": tn,
                 "sha256": hashlib.sha256(data).hexdigest()}
            if rw is not None:
                e["row_bytes"] = rw & 0x0FFFFFFF
                e["rw"] = rw
            f.write(data.tobytes()), pack_hash.update(data)
            manifest["tensors"][name] = e
            manifest["source"]["types"][tn] = manifest["source"]["types"].get(tn, 0) + 1
            pos += len(data)

        # head, in GGUF order: token_embd (Q4_K), output_norm (F32), output (Q6_K)
        for name in ("token_embd.weight", "output_norm.weight", "output.weight"):
            tn = tn_of(name)
            data = np.ascontiguousarray(raw_of(name))
            check(name, data)
            t = byname[name]
            if tn == "F32":
                emit(name, data, t["shape"], "F32")
            else:
                rows, cols = t["shape"]
                emit(name, data, t["shape"], tn, RW[tn] | rowb(tn, cols))
        head_end = pos

        # layers, each at its fixed slot layout
        lb = None
        for i in range(cfg["n_layers"]):
            base = pos
            for name, allowed in LAYER:
                full = f"blk.{i}.{name}"
                tn = tn_of(full)
                assert tn in allowed, (full, tn, allowed)
                data = np.ascontiguousarray(raw_of(full))
                check(full, data)
                t = byname[full]
                # pad to the slot end
                want_off = base + slots[name]
                pad = b"\0" * (want_off - pos)
                f.write(pad), pack_hash.update(pad)
                pos += len(pad)
                if tn == "F32":
                    emit(full, data, t["shape"], "F32")
                else:
                    rows, cols = t["shape"]
                    emit(full, data, t["shape"], tn, RW[tn] | rowb(tn, cols))
            # pad the layer tail to layer_bytes
            tail = b"\0" * (base + layer_bytes - pos)
            f.write(tail), pack_hash.update(tail)
            pos += len(tail)
            lb = layer_bytes

    manifest["pack_bytes"] = pos
    manifest["pack_sha256"] = pack_hash.hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # layer slices from the written pack.bin
    raw = np.memmap(out_dir / "pack.bin", dtype=np.uint8, mode="r")
    for i in range(cfg["n_layers"]):
        b = f"blk.{i}."
        ts = [v for kname, v in manifest["tensors"].items() if kname.startswith(b)]
        lo = min(v["offset"] for v in ts)
        assert lo == head_end + i * layer_bytes, (i, lo)
        (out_dir / "layers" / f"blk.{i}.bin").write_bytes(raw[lo:lo + layer_bytes].tobytes())
    return manifest, layer_bytes


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    m, lb = convert(*sys.argv[1:])
    print(f"{len(m['tensors'])} tensors, {m['pack_bytes']} bytes, types {m['source']['types']}, "
          f"layer_bytes {lb}, pack_sha256 {m['pack_sha256']}")
    print("checks: GGUF bytes verbatim, finite d, decode == ggml (Q4_K and Q6_K): all tensors PASS")
