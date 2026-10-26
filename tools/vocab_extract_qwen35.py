#!/usr/bin/env python3
# tools/vocab_extract_qwen35.py — extract the Qwen3.5 tokenizer vocabulary from
# GGUF metadata into the compact `tokenizer.bin` the C tokenizer loads.
#
#   vocab_extract_qwen35.py MODEL.gguf OUTDIR [--pack]
#
# Writes OUTDIR/tokenizer.bin (see tools/qwen35_tokenizer.h for the layout) and
# OUTDIR/tokenizer.tsv (a human-readable manifest). Everything the tokenizer
# needs comes from the GGUF: tokens, merges, token_type, special ids, add_bos.
import sys
import os
import struct
import hashlib

from gguf import GGUFReader

MAGIC = b"Q35TOK01"


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 2
    model = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    r = GGUFReader(model)
    f = r.fields
    tokens = f["tokenizer.ggml.tokens"].contents()
    types = list(f["tokenizer.ggml.token_type"].contents())
    merges_raw = f["tokenizer.ggml.merges"].contents()
    n_vocab = len(tokens)
    n_merges = len(merges_raw)
    max_token_len = max(len(t) for t in tokens)

    def getid(key):
        return int(f[key].contents()) if key in f else -1

    bos = getid("tokenizer.ggml.bos_token_id")
    eos = getid("tokenizer.ggml.eos_token_id")
    pad = getid("tokenizer.ggml.padding_token_id")
    add_bos = 1 if ("tokenizer.ggml.add_bos_token" in f and int(f["tokenizer.ggml.add_bos_token"].contents())) else 0

    specials = [i for i in range(n_vocab) if types[i] in (2, 3, 4)]  # unknown/control/user_defined
    # llama.cpp sorts the special-token cache by descending text length
    specials.sort(key=lambda i: -len(tokens[i]))

    with open(os.path.join(outdir, "tokenizer.bin"), "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<IIIiiiII", n_vocab, n_merges, max_token_len,
                             bos, eos, pad, add_bos, len(specials)))
        for s in specials:
            fh.write(struct.pack("<i", s))
        for i, t in enumerate(tokens):
            b = t.encode("utf-8")
            fh.write(struct.pack("<HB", len(b), types[i] & 0xFF))
            fh.write(b)
        for m in merges_raw:
            parts = m.split(" ", 1)
            l = parts[0].encode("utf-8")
            rr = (parts[1] if len(parts) > 1 else "").encode("utf-8")
            fh.write(struct.pack("<H", len(l)))
            fh.write(l)
            fh.write(struct.pack("<H", len(rr)))
            fh.write(rr)

    blob = open(os.path.join(outdir, "tokenizer.bin"), "rb").read()
    with open(os.path.join(outdir, "tokenizer.tsv"), "w") as tsv:
        tsv.write(f"# source\t{model}\n")
        tsv.write(f"# n_vocab\t{n_vocab}\n# n_merges\t{n_merges}\n")
        tsv.write(f"# bos\t{bos}\n# eos\t{eos}\n# pad\t{pad}\n# add_bos\t{add_bos}\n")
        tsv.write(f"# max_token_len\t{max_token_len}\n# n_special\t{len(specials)}\n")
        tsv.write(f"# tokenizer.bin\t{len(blob)} bytes\tsha256 {hashlib.sha256(blob).hexdigest()}\n")
    print(f"wrote {outdir}/tokenizer.bin ({len(blob)} B): n_vocab={n_vocab} n_merges={n_merges} "
          f"bos={bos} eos={eos} pad={pad} add_bos={add_bos} specials={len(specials)} max_len={max_token_len}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
