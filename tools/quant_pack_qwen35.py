#!/usr/bin/env python3
"""Qwen3.5 GGUF (mixed quant) -> llama-bend quant pack, passthrough storage.

usage: quant_pack_qwen35.py MODEL.gguf OUT_DIR [--verify]

Doors (fm tag = index.tsv's last column; numbers from tools/formats.tsv):
    0 F32      verbatim f32le
    1 Q8_0     row-split [cols i8][cols/16 f16 d]   (the w2c Q8 layout)
    3 Q4_K     verbatim GGUF block bytes (144 B / 256)
    4 Q6_K     verbatim GGUF block bytes (210 B / 256)
    5 Q3_K     verbatim GGUF block bytes (110 B / 256)
    9 Q5_K     verbatim GGUF block bytes (176 B / 256)
    6 Q2_0     verbatim GGUF block bytes (18 B / 64)
    7 PQ2_0    verbatim GGUF block bytes (34 B / 128)

Q8_0 keeps the exact w2c row-split layout so the audited Q8/F32 engine path is
unchanged (regression). Every K-quant / ternary tensor is stored **byte for
byte** as it sits in the GGUF (no row-split), which is the llama.cpp-identical
contract; the engine dispatches on the format tag.

The Qwen3.5 hybrid has two layer shapes (linear 14 tensors / full 11), so
layers/blk.N.bin differ in size; each is written from its own contiguous range.
The engine reads per-tensor offsets from index.tsv (tools/pack_index.py).

The pack is verified two ways and both are recorded:
  * write-time: stored bytes == GGUF bytes for every verbatim tensor; d/dmin
    finite; and (small tensors fully, large tensors on sampled super-blocks)
    the vendored w0 dequant module reproduces the GGUF block's own decode;
  * readback (--verify): every per-tensor sha256 and the whole-pack sha256.

Peak RAM is ~2x the largest tensor (`token_embd` < 0.6 GiB): no whole-model
F32 materialization.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import q3k
import q4k
import q5k
import q6k
import q20
from formats import FMT
import pack_table
from gguf_convert import GGML_TYPES, build_config, check_shapes, read_gguf

ALIGN = 64
# layer files are a convenience copy of each layer's contiguous pack range; the
# engine reads pack.bin + index.tsv, so a flaky NTFS mount can skip them
# (Q35_PACK_LAYERS=0) without affecting any gate.
WRITE_LAYERS = os.environ.get("Q35_PACK_LAYERS", "1") != "0"
DOORS = {"F32", "Q8_0", "Q4_K", "Q6_K", "Q3_K", "Q5_K", "Q2_0", "PQ2_0", "IQ4_XS"}
AUDIT_FULL = 4_000_000  # dequant-audit tensors <= this many elements fully
AUDIT_BLOCKS = 512      # else compare this many sampled super-blocks per tensor


def tbytes(t):
    tn, blk, bsz = GGML_TYPES[t["type"]]
    n = int(np.prod(t["shape"]))
    return n // blk * bsz


def rowsplit(raw, rows, cols):
    """GGUF Q8_0 block bytes [rows*cols/32*34] -> row-split bytes (w2c layout)."""
    b = raw.reshape(rows, cols // 32, 34)
    return np.concatenate(
        [b[:, :, 2:].reshape(rows, cols), b[:, :, :2].reshape(rows, cols // 16)], axis=1
    )


def unsplit(row, rows, cols):
    r = row.reshape(rows, -1)
    return (
        np.concatenate(
            [r[:, cols:].reshape(rows, -1, 2), r[:, :cols].reshape(rows, -1, 32)],
            axis=2,
        )
        .reshape(-1)
        .copy()
    )


def dequant_rows(data, tn, rows, cols):
    return {
        "Q4_K": lambda: q4k.dequant_rows(data, rows, cols),
        "Q6_K": lambda: q6k.dequant_rows(data, rows, cols),
        "Q3_K": lambda: q3k.dequant_rows(data, rows, cols),
        "Q5_K": lambda: q5k.dequant_rows(data, rows, cols),
        "Q2_0": lambda: q20.dequant_rows(data, rows, cols, 64),
        "PQ2_0": lambda: q20.dequant_rows(data, rows, cols, 128),
    }[tn]()


def ggml_dequant(data, tn):
    return {
        "Q4_K": lambda: q4k.ggml_dequant(data),
        "Q6_K": lambda: q6k.ggml_dequant(data),
        "Q3_K": lambda: q3k.ggml_dequant(data),
        "Q5_K": lambda: q5k.ggml_dequant(data),
        "Q2_0": lambda: q20.ggml_dequant(data, 64),
        "PQ2_0": lambda: q20.ggml_dequant(data, 128),
    }[tn]()


def block_bytes(tn):
    return {
        "Q4_K": q4k.BLOCK_BYTES, "Q6_K": q6k.BLOCK_BYTES, "Q3_K": q3k.BLOCK_BYTES,
        "Q5_K": q5k.BLOCK_BYTES, "Q2_0": q20.BLOCK_BYTES[64],
        "PQ2_0": q20.BLOCK_BYTES[128],
    }[tn]


def qk(tn):
    return {"Q2_0": 64, "PQ2_0": 128}.get(tn, 256)


def check_quant(name, data, tn, rows, cols):
    """Write-time contract check on one stored verbatim tensor: finite scale(s)
    and decode == ggml (full for small tensors, sampled super-blocks for large)."""
    nb = data.size // block_bytes(tn)
    raw = data.reshape(nb, block_bytes(tn))
    # finite d (and dmin for the K family)
    if tn in ("Q4_K", "Q5_K"):
        di = np.concatenate([raw[:, 0:2], raw[:, 2:4]]).copy().view(np.uint16)
    elif tn == "Q6_K":
        di = raw[:, 208:210].copy().view(np.uint16)
    elif tn == "Q3_K":
        di = raw[:, 108:110].copy().view(np.uint16)
    else:  # Q2_0 / PQ2_0
        di = raw[:, 0:2].copy().view(np.uint16)
    assert np.isfinite(di.view(np.float16)).all(), f"{name}: nonfinite scale"
    # decode vs ggml: fully for small tensors, sampled super-blocks otherwise
    if data.size <= AUDIT_FULL:
        sel = np.arange(nb)
    else:
        sel = np.unique(np.linspace(0, nb - 1, AUDIT_BLOCKS).astype(np.int64))
    sample = np.ascontiguousarray(raw[sel]).reshape(-1)
    got = dequant_rows(sample, tn, len(sel), qk(tn))
    want = np.ascontiguousarray(ggml_dequant(sample, tn)).reshape(-1, qk(tn))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (
        f"{name}: decode ({tn}) differs from ggml on {len(sel)}/{nb} blocks"
    )


def convert(src, out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "pack.bin").exists():
        raise SystemExit(f"{out_dir}/pack.bin already exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    if WRITE_LAYERS:
        (out_dir / "layers").mkdir(exist_ok=False)
    kv, tensors, buf = read_gguf(src)
    cfg = build_config(kv, tensors)
    bad = check_shapes(cfg, tensors)
    if bad:
        raise SystemExit("shape check failed:\n  " + "\n  ".join(bad))
    types = {GGML_TYPES[t["type"]][0] for t in tensors}
    if not types <= DOORS:
        raise SystemExit(f"STOP: unsupported types {sorted(types - DOORS)} (doors {sorted(DOORS)})")

    manifest = {
        "format": "llama-bend-pack-q35-quant/v0",
        "alignment": ALIGN,
        "dtype": "mixed-quant",
        "source": {"file": str(src), "types": {}},
        "config": cfg,
        "tensors": {},
    }
    pack_hash, pos = hashlib.sha256(), 0
    with open(out_dir / "pack.bin", "wb") as f:
        for n, t in enumerate(tensors):
            tn = GGML_TYPES[t["type"]][0]
            src_bytes = tbytes(t)
            raw = np.frombuffer(buf, np.uint8, src_bytes, t["offset"])
            meta = {}
            if tn == "F32":
                data = np.ascontiguousarray(raw)
            elif tn == "Q8_0":
                rows, cols = t["shape"]
                data = np.ascontiguousarray(rowsplit(raw, rows, cols)).reshape(-1)
                assert np.array_equal(unsplit(data, rows, cols), raw), (
                    f"{t['name']}: Q8 re-slice not lossless vs GGUF"
                )
                meta["row_bytes"] = cols * 34 // 32
            else:
                rows, cols = t["shape"]
                data = np.ascontiguousarray(raw)
                assert data.tobytes() == bytes(raw), f"{t['name']}: not verbatim"
                check_quant(t["name"], data, tn, rows, cols)
                meta["row_bytes"] = cols // qk(tn) * block_bytes(tn)
            pad = b"\0" * ((-pos) % ALIGN)
            f.write(pad), pack_hash.update(pad)
            pos += len(pad)
            e = {
                "shape": t["shape"],
                "offset": pos,
                "bytes": len(data),
                "src_type": tn,
                "fmt": FMT[tn],
                "sha256": hashlib.sha256(data).hexdigest(),
                **meta,
            }
            f.write(data.tobytes()), pack_hash.update(data)
            manifest["tensors"][t["name"]] = e
            manifest["source"]["types"][tn] = manifest["source"]["types"].get(tn, 0) + 1
            pos += len(data)
    manifest["pack_bytes"] = pos
    manifest["pack_sha256"] = pack_hash.hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    raw = np.memmap(out_dir / "pack.bin", dtype=np.uint8, mode="r")
    sizes = {}
    for i in range(cfg["n_layers"]):
        ts = [v for k, v in manifest["tensors"].items() if k.startswith(f"blk.{i}.")]
        lo, hi = min(v["offset"] for v in ts), max(v["offset"] + v["bytes"] for v in ts)
        sizes[i] = hi - lo
        if WRITE_LAYERS:
            (out_dir / "layers" / f"blk.{i}.bin").write_bytes(raw[lo:hi].tobytes())
    return manifest, sizes


def verify(out_dir):
    """Re-read the pack and re-check every tensor sha256, the whole-file hash,
    and for quant tensors the row_bytes/format consistency + finite scales."""
    d = Path(out_dir)
    m = json.load(open(d / "manifest.json"))
    raw = np.memmap(d / "pack.bin", dtype=np.uint8, mode="r")
    if len(raw) != m["pack_bytes"]:
        raise SystemExit(f"pack size {len(raw)} != manifest {m['pack_bytes']}")
    if hashlib.sha256(raw).hexdigest() != m["pack_sha256"]:
        raise SystemExit("pack_sha256 MISMATCH")
    bad, nchk = [], 0
    for name, t in m["tensors"].items():
        e = raw[t["offset"]: t["offset"] + t["bytes"]]
        if t["offset"] % m["alignment"]:
            bad.append(f"{name}: misaligned")
        if hashlib.sha256(e).hexdigest() != t["sha256"]:
            bad.append(f"{name}: sha256 mismatch")
            continue
        tn = t["src_type"]
        if tn in DOORS and tn != "F32":
            if tn == "Q8_0":
                pass
            else:
                rows, cols = t["shape"]
                if t["row_bytes"] != cols // qk(tn) * block_bytes(tn):
                    bad.append(f"{name}: row_bytes mismatch")
                    continue
                bb = block_bytes(tn)
                if tn in ("Q4_K", "Q5_K"):
                    di = np.concatenate([np.asarray(e).reshape(-1, bb)[:, 0:2],
                                         np.asarray(e).reshape(-1, bb)[:, 2:4]])
                elif tn == "Q6_K":
                    di = np.asarray(e).reshape(-1, bb)[:, 208:210]
                elif tn == "Q3_K":
                    di = np.asarray(e).reshape(-1, bb)[:, 108:110]
                else:
                    di = np.asarray(e).reshape(-1, bb)[:, 0:2]
                if not np.isfinite(di.copy().view(np.float16)).all():
                    bad.append(f"{name}: nonfinite scale")
                nchk += 1
    if bad:
        raise SystemExit("verify FAILED:\n  " + "\n  ".join(bad[:20]))
    ldir = d / "layers"
    if ldir.is_dir():
        for i in range(m["config"]["n_layers"]):
            ts = [v for k, v in m["tensors"].items() if k.startswith(f"blk.{i}.")]
            lo, hi = min(v["offset"] for v in ts), max(v["offset"] + v["bytes"] for v in ts)
            if (ldir / f"blk.{i}.bin").read_bytes() != bytes(raw[lo:hi]):
                raise SystemExit(f"layers/blk.{i}.bin differs from pack range")
    return nchk


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if flags == ["--verify"] and len(pos) == 1:
        c = verify(pos[0])
        print(f"verify PASS: all tensor sha256 + pack_sha256 ok, {c} quant tensors checked")
        raise SystemExit(0)
    if flags or len(pos) != 2:
        raise SystemExit(__doc__)
    m, sizes = convert(pos[0], pos[1])
    print(f"{len(m['tensors'])} tensors, {m['pack_bytes']} bytes, types {m['source']['types']}, "
          f"fmt counts {sorted({e['fmt'] for e in m['tensors'].values()})}, "
          f"pack_sha256 {m['pack_sha256']}")
    print("checks: F32 verbatim, Q8_0 re-slice lossless, K-quant/ternary bytes verbatim + "
          "decode == ggml (small tensors full, large sampled), scales finite")
    c = verify(pos[1])
    print(f"readback verify PASS: all tensor sha256 + pack_sha256 ok, {c} quant tensors checked")
    print(f"table.bin: {len(pack_table.write(pos[1]))} layers")
