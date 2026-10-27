# bf16_to_q8.py SRC.gguf OUT.gguf: copy a GGUF, every BF16 tensor re-encoded as Q8_0 (others byte-for-byte)
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
from gguf.quants import quantize

src, out = sys.argv[1:3]
r = GGUFReader(src)
arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
w = GGUFWriter(out, arch)
for f in r.fields.values():
    if f.name.startswith(("GGUF.", "general.architecture")) or not f.types:
        continue
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        et = f.types[1]
        vals = [bytes(f.parts[i]).decode() if et == GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data]
        w.add_key_value(f.name, vals, t, sub_type=et)
    elif t == GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode(), t)
    else:
        w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], t)
n = 0
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape.tolist())]  # numpy order (rows first)
    if t.tensor_type == GGMLQuantizationType.BF16:
        a = np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16
        f32 = a.view(np.float32).reshape(shape)
        q = quantize(f32, GGMLQuantizationType.Q8_0)
        w.add_tensor(t.name, q, raw_dtype=GGMLQuantizationType.Q8_0)
        n += 1
    else:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "bf16->q8_0 tensors:", n)
