#!/usr/bin/env python3
"""relayout_gguf.py IN OUT: same tensor bytes, house (unsloth) layout = non-blk tensors by name, then blk.N contiguous per layer
(N numeric, names sorted) so gguf2pack's per-layer u32 span holds; general.name/basename set to the pack's bound trunk id
(identity proven separately by trunk_fingerprint == ledger).  Verifies every tensor's bytes after writing."""
import hashlib, re, sys, gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails
src, dst = sys.argv[1:3]; ID = "Qwen3.5-9B-Base"
r = gguf.GGUFReader(src); arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
new = {"general.name": ID, "general.basename": ID}
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, new.get(f.name, f.contents()), vt, sub_type=st)
def key(t):
    m = re.match(r"blk\.(\d+)\.(.*)", t.name)
    return (1, int(m.group(1)), m.group(2)) if m else (0, 0, t.name)
ts = sorted(r.tensors, key=key)
for t in ts: w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in ts: w.write_tensor_data(t.data)
w.close()
r2 = gguf.GGUFReader(dst); a = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in r.tensors}
b = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in r2.tensors}
print("tensors", len(a), len(b), "bytes_identical", a == b, "first", [t.name for t in r2.tensors[:5]])
h = hashlib.sha256("".join(f"{k}:{a[k]}\n" for k in sorted(a)).encode()).hexdigest(); print("tensor_table_sha256", h)
sys.exit(0 if a == b else 1)
