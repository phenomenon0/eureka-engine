#!/usr/bin/env python3
# cache.py evict|warm PATH...: page-cache control for G3 (read-only opens, no root).
#   evict: posix_fadvise(DONTNEED) on every file (a dir: its files, recursively; symlinks followed)
#   warm:  read every file; a pack dir: manifest/table/layers + only the two pack.bin ranges the
#          engine reads ([0, headn) and token_embd), not the whole (HDD) pack.bin
#   head:  a pack dir as warm, minus layers/ (the 27B's 17 GB of NVMe layers would not stay cached)
import json, os, sys


def files(p):
    if os.path.isdir(p):
        for d, _, fs in os.walk(p, followlinks=True):
            for f in fs:
                yield os.path.join(d, f)
    else:
        yield p


def read(p, a=0, n=None):
    with open(p, "rb", buffering=0) as f:
        f.seek(a)
        left = n if n is not None else os.fstat(f.fileno()).st_size - a
        while left > 0:
            b = f.read(min(left, 64 << 20))
            if not b:
                break
            left -= len(b)


mode, paths = sys.argv[1], sys.argv[2:]
for p in paths:
    if mode == "evict":
        for f in files(p):
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    elif os.path.isdir(p):
        t = json.load(open(f"{p}/manifest.json"))["tensors"]
        dm = json.load(open(f"{p}/manifest.json"))["config"]["d_model"]
        o, n, e = t["output.weight"], t["output_norm.weight"], t["token_embd.weight"]
        head = max(o["offset"] + o["bytes"], n["offset"] + dm * 4)
        for f in files(p):
            if os.path.basename(f) != "pack.bin" and (mode == "warm" or "/layers/" not in f):
                read(f)
        read(f"{p}/pack.bin", 0, head)
        read(f"{p}/pack.bin", e["offset"], e["bytes"])
    else:
        read(p)
    print(f"{mode} {p}")
