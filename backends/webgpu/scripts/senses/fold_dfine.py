# fold_dfine.py <in.onnx> <out.onnx>: ORT-web's wasm/WebGPU builds carry no float64 kernels, and D-FINE's exported sine
# position embedding is a float64 subgraph of constants. Constant-fold it offline with full onnxruntime (basic level =
# constant folding + redundant node elimination only, no EP-specific fusions), then assert no DOUBLE tensors remain.
import sys, onnx, onnxruntime as ort
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.optimized_model_filepath = sys.argv[2]
ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
m = onnx.load(sys.argv[2]); D = onnx.TensorProto.DOUBLE
dbl = [n.name for n in m.graph.node if n.op_type == "Cast" and any(a.name == "to" and a.i == D for a in n.attribute)]
dbl += [i.name for i in m.graph.initializer if i.data_type == D]
print(sys.argv[2], "nodes", len(m.graph.node), "float64 left:", dbl[:5], len(dbl))
