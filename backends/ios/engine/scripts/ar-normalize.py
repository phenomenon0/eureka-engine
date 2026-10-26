#!/usr/bin/env python3
"""ar-normalize.py A.a B.a: compare two static archives with every member header's mtime (and uid/gid/mode)
zeroed. libtool -static stamps each member with the build time, so the archive's sha256 changes per build while
every object inside is the same; this says whether that is the ONLY difference (exit 0) or not (exit 1)."""
import sys

def norm(path):
    d = bytearray(open(path, 'rb').read())
    assert d[:8] == b'!<arch>\n', path
    i = 8
    while i + 60 <= len(d):
        name = bytes(d[i:i + 16]); size = int(d[i + 48:i + 58].decode().strip())
        d[i + 16:i + 48] = b'0'.ljust(12) + b'0'.ljust(6) + b'0'.ljust(6) + b'0'.ljust(8)  # mtime uid gid mode
        i += 60 + size + (size & 1)
    return bytes(d)

a, b = norm(sys.argv[1]), norm(sys.argv[2])
print('IDENTICAL modulo ar member mtime/uid/gid/mode' if a == b else 'DIFFERENT beyond member headers')
sys.exit(0 if a == b else 1)
