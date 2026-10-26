#!/usr/bin/env python3
"""Copy each layer's byte range of pack.bin into <pack-dir>/layers/blk.N.bin.

usage: tools/split_pack.py <pack-dir> [--check]

bend2's File.read_at takes a U32 offset, so later layers need separate files.
Validate the fixed v0 MiniCPM config, tensor metadata and identical relative
layer layouts before writing anything. Verify the source pack and all tensors,
then independently read/hash each split file. Existing matching files are
reused; mismatches fail without overwriting. --check never creates/writes files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from verify_forward import validate_manifest


def hash_bytes(src, size, whole=None):
    digest = hashlib.sha256()
    while size:
        buf = src.read(min(size, 1024 * 1024))
        if not buf:
            raise ValueError(f"{src.name}: short read ({size} bytes missing)")
        digest.update(buf)
        if whole is not None:
            whole.update(buf)
        size -= len(buf)
    return digest.hexdigest()


def verify_file(path, tensors, start, size, expected=None):
    """Read from disk, hashing every tensor and the complete file including gaps."""
    with open(path, "rb") as src:
        if os.fstat(src.fileno()).st_size != size:
            raise ValueError(f"{path}: expected {size} bytes")
        whole, pos = hashlib.sha256(), 0
        for name, t in sorted(tensors.items(), key=lambda item: item[1]["offset"]):
            offset = t["offset"] - start
            hash_bytes(src, offset - pos, whole)
            if hash_bytes(src, t["bytes"], whole) != t["sha256"]:
                raise ValueError(f"{path}: {name} sha256 mismatch")
            pos = offset + t["bytes"]
        hash_bytes(src, size - pos, whole)
    digest = whole.hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: full-file sha256 mismatch")
    return digest


def layer_layouts(m):
    """Preflight every layer and head before even creating the layers directory."""
    t, layouts, rel0 = m["tensors"], [], None
    for i in range(m["config"]["n_layers"]):
        prefix = f"blk.{i}."
        ts = {k: v for k, v in t.items() if k.startswith(prefix)}
        lo = min(v["offset"] for v in ts.values())
        hi = max(v["offset"] + v["bytes"] for v in ts.values())
        rel = {
            k[len(prefix) :]: (v["offset"] - lo, v["bytes"], v["shape"])
            for k, v in ts.items()
        }
        if rel0 is None:
            rel0 = rel
        if rel != rel0:
            raise ValueError(f"blk.{i}: layout differs from blk.0")
        if hi - lo > 2**32:
            raise ValueError(f"blk.{i}: layer exceeds U32 address space")
        layouts.append((i, lo, hi - lo, ts))
    for k in ("output.weight", "output_norm.weight", "token_embd.weight"):
        if t[k]["offset"] + t[k]["bytes"] > 2**32:
            raise ValueError(f"{k}: exceeds U32 address space in pack.bin")
    return layouts, rel0


def main(pack, check=False):
    pack = Path(pack)
    with open(pack / "manifest.json") as f:
        m = json.load(f)
    source = pack / "pack.bin"
    validate_manifest(m, source.stat().st_size)
    layouts, rel = layer_layouts(m)
    digest = verify_file(source, m["tensors"], 0, m["pack_bytes"], m["pack_sha256"])
    print(
        f"pack.bin: {m['pack_bytes']} B, {len(m['tensors'])} tensors sha256 ok, pack_sha256 {digest} ok",
        flush=True,
    )

    pending = []
    with open(source, "rb") as src:
        # Check every existing output before creating any new output.
        for i, lo, size, ts in layouts:
            src.seek(lo)
            expected = hash_bytes(src, size)
            path = pack / "layers" / f"blk.{i}.bin"
            if path.exists() or check:
                verify_file(path, ts, lo, size, expected)
                print(
                    f"blk.{i}: {size} B, {len(ts)} tensors + source-range sha256 ok ({'checked' if check else 'reused'})",
                    flush=True,
                )
            else:
                pending.append((path, lo, size, ts, expected))
        if pending:
            (pack / "layers").mkdir(exist_ok=True)
        for path, lo, size, ts, expected in pending:
            # Exclusive creation also prevents overwriting a file added since preflight.
            with open(path, "xb") as dst:
                src.seek(lo)
                remaining = size
                while remaining:
                    buf = src.read(min(remaining, 1024 * 1024))
                    if not buf:
                        raise ValueError(
                            f"{source}: short read while splitting {path.name}"
                        )
                    dst.write(buf)
                    remaining -= len(buf)
            verify_file(path, ts, lo, size, expected)
            print(
                f"{path.name}: {size} B, {len(ts)} tensors + source-range sha256 ok (written, read back)",
                flush=True,
            )
    print(
        "relative layout:",
        json.dumps({k: v[0] for k, v in sorted(rel.items(), key=lambda kv: kv[1][0])}),
    )
    print(
        f"PASS: original pack + {len(m['tensors'])} tensors; {len(layouts)} independently read layer files"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing split files without writing",
    )
    args = parser.parse_args()
    try:
        main(args.pack, args.check)
    except (ValueError, OSError, KeyError, TypeError) as e:
        parser.exit(1, f"ERROR: {e}\n")
