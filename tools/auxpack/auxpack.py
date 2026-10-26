#!/usr/bin/env python3
"""auxpack.py — AUX1 pack tool (twoheads S6-a, W6/S6 'aux-pack loader generalization').

AUX1 = the named/dtype/shape-table container for dedicated module/aux packs
(the generalization of the MTP1 positional table; precedent: tools/mtp27/mtp_pack.py
layout + host/llb_gguf.c validating-reader discipline). Replaces position-identity
with name-identity so any head/adapter/aux pack can be described, converted and
validated without a new bespoke parser.

Layout (little-endian):
  0:  4B magic "AUX1"
  4:  u32 version (=1)
  8:  u32 n
  12: u32 flags (0)
  16: n * 72B entries, in file order:
        char name[32];   NUL-padded, unique
        u32 dtype;       0 F32, 1 BF16, 2 Q8_0, 3 Q4_K, 4 Q6_K, 5 U32, 6 BLOB
        u32 rank;        0 = opaque bytes (shape checked downstream), 1..4
        u32 ne[4];       dims, ne[0] fastest (rank 0: all zero)
        u64 off;         256-aligned payload offset
        u64 bytes;
  16 + 72n rounded up to 256: payloads, 256-aligned, in entry order.

Commands:
  convert  IN.mtp1 OUT.aux1   MTP1 -> AUX1 (names from tools/mtp27/mtp_pack.py ORDER;
                              entry 16 'mtp.head_slice' Q6_K, entry 17 'mtp.ids' U32)
  mkfixture OUT.aux1          synthetic fixture for auxpack.c --selftest
  mkbads    OUT.aux1          like mkfixture but fixture.a declares ne=[9] (tamper test)
  show     PACK               header dump (mtp1 or aux1)
"""
import json, struct, sys, os

DT = {"F32": 0, "BF16": 1, "Q8_0": 2, "Q4_K": 3, "Q6_K": 4, "U32": 5, "BLOB": 6}
DT_REV = {v: k for k, v in DT.items()}
ENTRY = 72

def align256(x): return (x + 255) // 256 * 256

def wr_aux(path, entries, blobs):
    # entries: list of (name, dtype, rank, ne[4]); blobs parallel
    off = align256(16 + ENTRY * len(entries))
    tab, payloads = [], []
    for name, dt, rank, ne in entries:
        b = blobs[len(tab)]
        tab.append((name, dt, rank, ne, off, len(b)))
        payloads.append(b)
        off = align256(off + len(b))
    with open(path, "wb") as f:
        f.write(b"AUX1" + struct.pack("<III", 1, len(entries), 0))
        for name, dt, rank, ne, o, nb in tab:
            f.write(struct.pack("<32sII4IQQ", name.encode(), dt, rank, *ne, o, nb))
        pad = align256(16 + ENTRY * len(entries)) - (16 + ENTRY * len(entries))
        if pad: f.write(b"\0" * pad)
        for i, b in enumerate(payloads):
            pad = tab[i][4] - f.tell()
            assert pad >= 0, (f.tell(), tab[i][4])
            if pad: f.write(b"\0" * pad)
            f.write(b)
    print(f"AUX1 written: {path} n={len(entries)} size={os.path.getsize(path)}")

def rd_mtp(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"MTP1", "not MTP1"
    n, = struct.unpack_from("<I", d, 4)
    tab = [struct.unpack_from("<QQ", d, 8 + 16 * i) for i in range(n)]
    return d, tab

def cmd_convert(src, dst):
    d, tab = rd_mtp(src)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mtp27"))
    try:
        from mtp_pack import ORDER
    except Exception as e:
        sys.exit(f"convert needs tools/mtp27/mtp_pack.py ORDER: {e}")
    assert len(ORDER) + 2 == len(tab), (len(ORDER), len(tab))
    entries, blobs = [], []
    for (name, ty), (o, nb) in zip(ORDER, tab):
        entries.append((name, DT[ty], 0, [0, 0, 0, 0]))
        blobs.append(d[o:o + nb])
    entries.append(("mtp.head_slice", DT["Q6_K"], 0, [0, 0, 0, 0]))
    blobs.append(d[tab[-2][0]:tab[-2][0] + tab[-2][1]])
    entries.append(("mtp.ids", DT["U32"], 0, [0, 0, 0, 0]))
    blobs.append(d[tab[-1][0]:tab[-1][0] + tab[-1][1]])
    wr_aux(dst, entries, blobs)

def cmd_fixture(dst, bad=False):
    a = struct.pack("<8f", *[1.5, -2.25, 3.0, 4.5, 5.25, 6.0, 7.75, 8.0])
    b = struct.pack("<4I", 11, 22, 33, 44)
    c = bytes(range(128))
    entries = [("fixture.a", DT["F32"], 1, [9 if bad else 8, 0, 0, 0]),
               ("fixture.b", DT["U32"], 1, [4, 0, 0, 0]),
               ("fixture.c", DT["BLOB"], 0, [0, 0, 0, 0])]
    wr_aux(dst, entries, [a, b, c])

def cmd_show(path):
    d = open(path, "rb").read()
    if d[:4] == b"AUX1":
        ver, n, fl = struct.unpack_from("<III", d, 4)
        print(f"AUX1 v{ver} n={n} flags={fl} size={len(d)}")
        for i in range(n):
            name, dt, rank, ne0, ne1, ne2, ne3, o, nb = struct.unpack_from("<32sII4IQQ", d, 16 + ENTRY * i)
            print(f"  [{i:2d}] {name.rstrip(chr(0).encode()).decode():20s} {DT_REV.get(dt, dt):5s} rank={rank} ne=({ne0},{ne1},{ne2},{ne3}) off={o} bytes={nb}")
    elif d[:4] == b"MTP1":
        n, = struct.unpack_from("<I", d, 4)
        print(f"MTP1 n={n} size={len(d)}")
        for i in range(n):
            o, nb = struct.unpack_from("<QQ", d, 8 + 16 * i)
            print(f"  [{i:2d}] off={o} bytes={nb}")
    else:
        sys.exit("unknown magic")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    c = sys.argv[1]
    if c == "convert": cmd_convert(sys.argv[2], sys.argv[3])
    elif c == "mkfixture": cmd_fixture(sys.argv[2])
    elif c == "mkbads": cmd_fixture(sys.argv[2], bad=True)
    elif c == "show": cmd_show(sys.argv[2])
    else: sys.exit(__doc__)
