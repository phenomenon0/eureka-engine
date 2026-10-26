#!/usr/bin/env python3
"""Kokoro ONNX with fp16 *storage* and fp32 *compute*: every large float32 initializer is stored as float16 and Cast back
to float32 in-graph (onnxruntime folds the Casts at session load). Half the download, fp32 numerics — unlike the
onnx-community fp16 graph, which overflows to NaN/silence on ~13% of lines (kit/eval). Usage: fp16_storage.py IN OUT"""
import sys, numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
m = onnx.load(sys.argv[1])
g = m.graph
casts, keep = [], []
for init in g.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 1024:
        a = numpy_helper.to_array(init)
        h = numpy_helper.from_array(a.astype(np.float16), init.name + "__f16")
        keep.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    else:
        keep.append(init)
del g.initializer[:]
g.initializer.extend(keep)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.save(m, sys.argv[2])
print(f"{len(casts)} initializers -> fp16 storage")
