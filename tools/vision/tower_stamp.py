#!/usr/bin/env python3
"""tower_stamp.py ROWS.f32 MMPROJ.gguf NX NY [--mmproj-sha256 S]: the llb-tower-rows-v1 stamp ROWS.json beside ROWS.f32
(FOLD-CONTINUE MIMG). Binds the Qwen3.5 tower rows (vision/build/cudavis-cli IMG MMPROJ --out ROWS.f32) to the mmproj
they came from and to their grid; dual.c's "images" requests refuse rows whose stamp is missing, whose rows_sha256 is not
the file's, whose nx/ny differ from the request, or whose mmproj_sha256 is not the engine's LLB_VIS_MMPROJ_SHA256.
--mmproj-sha256 skips re-hashing a known mmproj (a server caches it once)."""
import hashlib, json, os, sys

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def main(a):
    ms = None
    if "--mmproj-sha256" in a:
        i = a.index("--mmproj-sha256"); ms = a[i + 1]; del a[i:i + 2]
    if len(a) != 4:
        sys.exit(__doc__)
    rows, mm, nx, ny = a[0], a[1], int(a[2]), int(a[3])
    if not rows.endswith(".f32") or nx < 1 or ny < 1:
        sys.exit("tower_stamp: ROWS must be a .f32 file, nx/ny >= 1")
    # SENSES-MATRIX: d is the tower's (its mmproj projection: 4096 on the 9B, 1024 / 2560 / 5120 on the 0.8B / 4B / 27B);
    # dual.c refuses rows whose d is not the trunk's width
    sz = os.path.getsize(rows)
    d = sz // (nx * ny * 4) if nx * ny and sz % (nx * ny * 4) == 0 else 0
    if d < 256 or d > 8192 or d % 256:
        sys.exit("tower_stamp: %s is not nx*ny x d F32 (d a multiple of 256, <= 8192)" % rows)
    st = {"kind": "llb-tower-rows-v1", "mmproj_sha256": ms or sha(mm), "mmproj": os.path.basename(mm),
          "nx": nx, "ny": ny, "d": d, "rows_sha256": sha(rows)}
    with open(rows[:-4] + ".json", "w") as f:
        json.dump(st, f, indent=1)
        f.write("\n")
    print(rows[:-4] + ".json", st["rows_sha256"])

if __name__ == "__main__":
    main(sys.argv[1:])
