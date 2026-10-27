# deq_stream.py IN OUT : deq_gguf.py's F32 copy, streamed one tensor at a time (bounded memory); token_embd keeps its
# type (llama.cpp's get_rows dequantizes it exactly; no activation is quantized on that path), saving disk
import sys, numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.quants import dequantize
r = GGUFReader(sys.argv[1]); a = r.fields["general.architecture"]; arch = bytes(a.parts[a.data[0]]).decode()
w = GGUFWriter(sys.argv[2], arch)
for f in r.fields.values():
    if f.name.startswith(("general.architecture", "GGUF.", "general.file_type")) or not f.types: continue
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        et = f.types[1]; w.add_key_value(f.name, [bytes(f.parts[i]).decode() if et == GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data], t, sub_type=et)
    elif t == GGUFValueType.STRING: w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode(), t)
    else: w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], t)
keep = lambda t: t.name == "token_embd.weight"
for t in r.tensors:
    shp = tuple(int(v) for v in reversed(t.shape))
    if keep(t): w.add_tensor_info(t.name, np.asarray(t.data).shape, np.dtype(np.uint8), int(t.n_bytes), raw_dtype=t.tensor_type)
    else: w.add_tensor_info(t.name, shp, np.dtype(np.float32), int(np.prod(shp)) * 4)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    if keep(t): w.write_tensor_data(np.asarray(t.data))
    else: w.write_tensor_data(dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(tuple(int(v) for v in reversed(t.shape))))
w.close(); print("wrote", sys.argv[2])
