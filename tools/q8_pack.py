#!/usr/bin/env python3
"""GGUF Q8_0 -> llama-bend Q8 pack (pack.bin + manifest.json + layers/blk.N.bin).

usage: q8_pack.py MODEL-Q8_0.gguf OUT_DIR PACK_F32_DIR

Same bytes as the GGUF, re-sliced per row. A Q8_0 matrix row of `cols` columns
(cols/32 blocks of [f16 d][32 x int8 q]) is stored as
    [cols x int8 q][cols/32 x f16le d]        row_bytes = cols * 34 / 32
so the quants stay 16-byte aligned (2048 cols -> 2176 B/row, 6144 -> 6528).
F32 tensors (the norms) are copied as f32le. Tensors are in GGUF order, each at
a multiple of 64. layers/blk.N.bin is layer N's contiguous byte range.

Dequant (the one arithmetic, ggml's dequantize_row_q8_0): w = f32(q) * f32(d),
one IEEE f32 multiply. With |q| <= 128 (8 bits) and d an f16 (11-bit
significand) the product has <= 19 significant bits, so it never rounds: the
dequantized matrix is exactly pack-f32's. Checked here for all 381 tensors,
along with the lossless re-slice (row-split -> GGUF blocks, byte-equal) and
finite scales (the engine's f16 decode covers finite values only).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gguf_convert import GGML_TYPES, build_config, check_shapes, read_gguf
from read_pack import load

ALIGN = 64


def rowsplit(raw, rows, cols):
    """GGUF Q8_0 block bytes [rows * cols/32 * 34] -> row-split bytes."""
    b = raw.reshape(rows, cols // 32, 34)
    return np.concatenate(
        [b[:, :, 2:].reshape(rows, cols), b[:, :, :2].reshape(rows, cols // 16)], axis=1
    )


def dequant(data, t):
    """The pack's stored bytes of tensor t -> f32 [rows, cols] (or the F32 vector)."""
    if t["src_type"] == "F32":
        return data.view("<f4").reshape(t["shape"])
    rows, cols = t["shape"]
    r = data.reshape(rows, t["row_bytes"])
    q = r[:, :cols].view(np.int8).astype(np.float32)
    d = np.ascontiguousarray(r[:, cols:]).view("<f2").astype(np.float32)
    return q * np.repeat(d, 32, axis=1)  # f32(q) * f32(d), exact


def load_q8(d):
    """(manifest, raw, get) with get(name) = the dequantized f32 tensor."""
    m = json.load(open(f"{d}/manifest.json"))
    raw = np.memmap(f"{d}/pack.bin", dtype=np.uint8, mode="r")
    get = lambda n: dequant(
        raw[m["tensors"][n]["offset"] :][: m["tensors"][n]["bytes"]], m["tensors"][n]
    )
    return m, raw, get


def convert(src, out_dir, f32_dir):
    out_dir = Path(out_dir)
    (out_dir / "layers").mkdir(parents=True, exist_ok=False)
    kv, tensors, buf = read_gguf(src)
    cfg = build_config(kv, tensors)
    bad = check_shapes(cfg, tensors)
    if bad:
        raise SystemExit("shape check failed:\n  " + "\n  ".join(bad))
    types = {GGML_TYPES.get(t["type"], ("?",))[0] for t in tensors}
    if not types <= {"Q8_0", "F32"}:
        raise SystemExit(f"STOP: Q8 pack takes Q8_0 + F32 only, got {sorted(types)}")
    mf, _, f32get = load(f32_dir)
    if mf["config"] != cfg:
        raise SystemExit("config differs from the F32 pack's")
    manifest = {
        "format": "llama-bend-pack-q8/v0",
        "alignment": ALIGN,
        "dtype": "q8_0-rowsplit",
        "source": {"file": str(src), "types": {}},
        "config": cfg,
        "tensors": {},
    }
    pack_hash, pos = hashlib.sha256(), 0
    with open(out_dir / "pack.bin", "wb") as f:
        for t in tensors:
            tn = GGML_TYPES[t["type"]][0]
            pad = b"\0" * ((-pos) % ALIGN)
            f.write(pad), pack_hash.update(pad)
            pos += len(pad)
            if tn == "F32":
                data = np.frombuffer(
                    buf, np.uint8, 4 * int(np.prod(t["shape"])), t["offset"]
                )
                meta = {}
            else:
                rows, cols = t["shape"]
                raw = np.frombuffer(buf, np.uint8, rows * cols // 32 * 34, t["offset"])
                data = np.ascontiguousarray(rowsplit(raw, rows, cols)).reshape(-1)
                meta = {"row_bytes": cols * 34 // 32}
                # lossless: row-split back to GGUF blocks, byte-equal
                r = data.reshape(rows, -1)
                back = np.concatenate(
                    [
                        r[:, cols:].reshape(rows, -1, 2),
                        r[:, :cols].reshape(rows, -1, 32),
                    ],
                    axis=2,
                )
                assert np.array_equal(back.reshape(-1), raw), (
                    f"{t['name']}: re-slice not lossless"
                )
                d = r[:, cols:].copy().view("<f2")
                assert np.isfinite(d).all(), f"{t['name']}: nonfinite f16 scale"
            e = {
                "shape": t["shape"],
                "offset": pos,
                "bytes": len(data),
                "src_type": tn,
                **meta,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            # the dequant is exactly the F32 pack's tensor (raw uint32 bits)
            assert np.array_equal(
                dequant(data, e).view(np.uint32),
                np.asarray(f32get(t["name"])).view(np.uint32),
            ), f"{t['name']}: dequant differs from pack-f32"
            f.write(data.tobytes()), pack_hash.update(data)
            manifest["tensors"][t["name"]] = e
            manifest["source"]["types"][tn] = manifest["source"]["types"].get(tn, 0) + 1
            pos += len(data)
    manifest["pack_bytes"] = pos
    manifest["pack_sha256"] = pack_hash.hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # layers/blk.N.bin: layer N's contiguous range (all layers share one relative layout)
    raw = np.memmap(out_dir / "pack.bin", dtype=np.uint8, mode="r")
    sizes = set()
    for i in range(cfg["n_layers"]):
        ts = [v for k, v in manifest["tensors"].items() if k.startswith(f"blk.{i}.")]
        lo, hi = min(v["offset"] for v in ts), max(v["offset"] + v["bytes"] for v in ts)
        sizes.add(hi - lo)
        (out_dir / "layers" / f"blk.{i}.bin").write_bytes(raw[lo:hi].tobytes())
    assert len(sizes) == 1, sizes
    return manifest, sizes.pop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    m, lb = convert(*sys.argv[1:])
    print(
        f"{len(m['tensors'])} tensors, {m['pack_bytes']} bytes, types {m['source']['types']}, "
        f"layer file {lb} bytes, pack_sha256 {m['pack_sha256']}"
    )
    print(
        "checks: re-slice lossless, scales finite, dequant == pack-f32 bits: all tensors PASS"
    )
