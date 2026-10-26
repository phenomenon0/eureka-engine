#!/usr/bin/env python3
"""Serve a model dir over HTTP with Range support (stdlib http.server ignores Range, so the app could not resume).
  serve-models.py DIR [--port 8765] [--drop-at BYTES]
DIR mirrors models.json paths (pack/pack.bin, kev-head-q8.gguf, ...). --drop-at cuts each file's first response
after BYTES bytes, once per path: a flaky network, to watch the app resume with a Range request."""

import argparse, functools, http.server, os, re, sys

a = argparse.ArgumentParser()
a.add_argument("dir")
a.add_argument("--port", type=int, default=8765)
a.add_argument("--drop-at", type=int, default=0)
args = a.parse_args()
dropped = set()


class H(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        start, end = (
            (int(m[1]), int(m[2]) if m[2] else size - 1) if m else (0, size - 1)
        )
        if start >= size:
            self.send_error(416)
            return None
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        f = open(path, "rb")
        f.seek(start)
        self.left = end - start + 1
        if args.drop_at and self.path not in dropped and size > args.drop_at:
            dropped.add(self.path)
            self.left = args.drop_at - start  # then the connection closes short
            sys.stderr.write(f"drop {self.path} after {self.left} bytes\n")
        return f

    def copyfile(self, src, dst):
        while self.left > 0:
            b = src.read(min(1 << 20, self.left))
            if not b:
                break
            dst.write(b)
            self.left -= len(b)


http.server.ThreadingHTTPServer(
    ("0.0.0.0", args.port), functools.partial(H, directory=args.dir)
).serve_forever()
