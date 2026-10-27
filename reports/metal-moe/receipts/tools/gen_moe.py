# gen_moe.py SRC.gguf OUT.gguf SPEC: a random-weight qwen35moe GGUF (tokenizer copied from SRC) at any shape.
# SPEC = comma list of key=value over: d L nh nkv hk hv nexp kexp ffe ffs fused seed noshexp (gen_synth.py's layout)
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, out, spec = sys.argv[1:4]
c = dict(d=1024, L=4, nh=8, nkv=2, hk=8, hv=16, nexp=60, kexp=6, ffe=768, ffs=512, fused=0, seed=1234, noshexp=0)
for kv in spec.split(","):
    k, v = kv.split("=")
    c[k] = int(v)
hd, S, rope, convk = 256, 128, 64, 4
rng = np.random.default_rng(c["seed"])
r = GGUFReader(src)
A = "qwen35moe"
inner = c["hv"] * S
C = 2 * c["hk"] * S + inner
w = GGUFWriter(out, A)
skip = ("general.architecture", "qwen35.", "qwen35moe.", "GGUF.", "general.file_type", "general.quantization_version")
for f in r.fields.values():
    if f.name.startswith(skip) or not f.types:
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
d = c["d"]
w.add_uint32(f"{A}.context_length", 262144)
w.add_uint32(f"{A}.embedding_length", d)
w.add_uint32(f"{A}.block_count", c["L"])
w.add_uint32(f"{A}.feed_forward_length", c["ffs"])
w.add_uint32(f"{A}.attention.head_count", c["nh"])
w.add_uint32(f"{A}.attention.head_count_kv", c["nkv"])
w.add_uint32(f"{A}.attention.key_length", hd)
w.add_uint32(f"{A}.attention.value_length", hd)
w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", 1e-6)
w.add_uint32(f"{A}.rope.dimension_count", rope)
w.add_array(f"{A}.rope.dimension_sections", [11, 11, 10, 0])
w.add_float32(f"{A}.rope.freq_base", 10000000.0)
w.add_uint32(f"{A}.ssm.conv_kernel", convk)
w.add_uint32(f"{A}.ssm.state_size", S)
w.add_uint32(f"{A}.ssm.group_count", c["hk"])
w.add_uint32(f"{A}.ssm.time_step_rank", c["hv"])
w.add_uint32(f"{A}.ssm.inner_size", inner)
w.add_uint32(f"{A}.full_attention_interval", 4)
w.add_uint32(f"{A}.expert_count", c["nexp"])
w.add_uint32(f"{A}.expert_used_count", c["kexp"])
w.add_uint32(f"{A}.expert_feed_forward_length", c["ffe"])
w.add_uint32(f"{A}.expert_shared_feed_forward_length", c["ffs"])
V = len(r.fields["tokenizer.ggml.tokens"].data)
def T(name, shape, scale=0.02, base=0.0):
    a = rng.standard_normal(shape, dtype=np.float32)
    a *= np.float32(scale)
    a += np.float32(base)
    w.add_tensor(name, a)
T("token_embd.weight", (V, d), 0.05)
T("output_norm.weight", (d,), 0.1, 1.0)
T("output.weight", (V, d), 0.05)
for i in range(c["L"]):
    p = f"blk.{i}."
    T(p + "attn_norm.weight", (d,), 0.1, 1.0)
    T(p + "post_attention_norm.weight", (d,), 0.1, 1.0)
    if (i + 1) % 4:
        T(p + "attn_qkv.weight", (C, d), 0.05)
        T(p + "attn_gate.weight", (inner, d), 0.05)
        T(p + "ssm_conv1d.weight", (C, convk), 0.3)
        T(p + "ssm_dt.bias", (c["hv"],), 0.5)
        w.add_tensor(p + "ssm_a", (-np.exp(0.5 * rng.standard_normal(c["hv"]))).astype(np.float32))
        T(p + "ssm_alpha.weight", (c["hv"], d), 0.05)
        T(p + "ssm_beta.weight", (c["hv"], d), 0.05)
        T(p + "ssm_norm.weight", (S,), 0.1, 1.0)
        T(p + "ssm_out.weight", (d, inner), 0.05)
    else:
        T(p + "attn_q.weight", (2 * c["nh"] * hd, d), 0.05)
        T(p + "attn_k.weight", (c["nkv"] * hd, d), 0.05)
        T(p + "attn_v.weight", (c["nkv"] * hd, d), 0.05)
        T(p + "attn_q_norm.weight", (hd,), 0.1, 1.0)
        T(p + "attn_k_norm.weight", (hd,), 0.1, 1.0)
        T(p + "attn_output.weight", (d, c["nh"] * hd), 0.05)
    T(p + "ffn_gate_inp.weight", (c["nexp"], d), 0.3)
    if c["fused"]:
        T(p + "ffn_gate_up_exps.weight", (c["nexp"], 2 * c["ffe"], d), 0.05)
    else:
        T(p + "ffn_gate_exps.weight", (c["nexp"], c["ffe"], d), 0.05)
        T(p + "ffn_up_exps.weight", (c["nexp"], c["ffe"], d), 0.05)
    T(p + "ffn_down_exps.weight", (c["nexp"], d, c["ffe"]), 0.05)
    T(p + "ffn_gate_inp_shexp.weight", (d,), 0.3)
    if c["noshexp"]:
        continue
    T(p + "ffn_gate_shexp.weight", (c["ffs"], d), 0.05)
    T(p + "ffn_up_shexp.weight", (c["ffs"], d), 0.05)
    T(p + "ffn_down_shexp.weight", (d, c["ffs"]), 0.05)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, c)
