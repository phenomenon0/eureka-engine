#!/usr/bin/env python3
"""Minimal llama-bend pack reader: re-reads pack.bin via manifest.json and re-verifies every sha256.
Tensors come back in their stored dtype (f32le, or f16le in an f16 pack)."""
import hashlib, json, sys
import numpy as np

def load(d):
    m = json.load(open(f"{d}/manifest.json"))
    raw = np.memmap(f"{d}/pack.bin", dtype=np.uint8, mode="r")
    dt = lambda n: {"f32le": "<f4", "f16le": "<f2"}[m["tensors"][n].get("dtype", "f32le")]
    get = lambda n: raw[m["tensors"][n]["offset"]:][:m["tensors"][n]["bytes"]].view(dt(n)).reshape(m["tensors"][n]["shape"])
    return m, raw, get

if __name__ == "__main__":
    m, raw, get = load(sys.argv[1])
    assert len(raw) == m["pack_bytes"], "pack size mismatch"
    bad = [n for n, t in m["tensors"].items()
           if t["offset"] % m["alignment"] or hashlib.sha256(get(n)).hexdigest() != t["sha256"]]
    whole = hashlib.sha256(raw).hexdigest() == m["pack_sha256"]
    print(f"{len(m['tensors']) - len(bad)}/{len(m['tensors'])} tensor sha256 ok, pack_sha256 {'ok' if whole else 'MISMATCH'}")
    sys.exit(bool(bad) or not whole)
