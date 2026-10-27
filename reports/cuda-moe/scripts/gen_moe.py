# gen_moe.py SRC.gguf OUT.gguf key=value... : a random-weight qwen35moe GGUF (tokenizer from SRC, optionally
# truncated to V tokens), matrices written directly as Q8_0 (random int8 + f16 scale per 32-block), the router,
# shared-expert gate, conv1d and 1-D tensors F32. Tensor order mimics llama.cpp converts: token_embd, then each
# block's tensors name-sorted, then output_norm / output (so the head is not in front: the MoE virtual layout
# moves it). Keys: d L ne k ef sf nh nkv lkh lvh hd S V iv fused(0/1) mtp(0/1) seed rope
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType as Q

src, out = sys.argv[1:3]
c = dict(d=2048, L=4, ne=256, k=8, ef=512, sf=512, nh=16, nkv=2, lkh=16, lvh=32, hd=256, S=128, V=8192, iv=4,
         fused=0, mtp=0, seed=1234, rope=64, convk=4)
for kv in sys.argv[3:]:
    a, b = kv.split("="); c[a] = int(b)
rng = np.random.default_rng(c["seed"])
r = GGUFReader(src)
A = "qwen35moe"
w = GGUFWriter(out, A)
V0 = len(r.fields["tokenizer.ggml.tokens"].data)
V = c["V"] if c["V"] > 0 else V0
skip_prefix = ("general.architecture", "qwen35.", "qwen35moe.", "GGUF.", "general.file_type", "general.quantization_version")
toks = [bytes(r.fields["tokenizer.ggml.tokens"].parts[i]).decode() for i in r.fields["tokenizer.ggml.tokens"].data]
keep = set(toks[:V])
for f in r.fields.values():
    if f.name.startswith(skip_prefix) or not f.types:
        continue
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        et = f.types[1]
        vals = [bytes(f.parts[i]).decode() if et == GGUFValueType.STRING else f.parts[i].tolist()[0] for i in f.data]
        if f.name in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.scores"):
            vals = vals[:V]
        if f.name == "tokenizer.ggml.merges" and V < V0:
            vals = [m for m in vals if len(m.split(" ")) == 2 and all(p in keep for p in m.split(" ")) and m.replace(" ", "") in keep]
        w.add_key_value(f.name, vals, t, sub_type=et)
    elif t == GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode(), t)
    else:
        v = f.parts[f.data[0]].tolist()[0]
        if f.name.endswith("_token_id") and v >= V:
            v = V - 1
        w.add_key_value(f.name, v, t)
d, L, hd, S = c["d"], c["L"], c["hd"], c["S"]
inner = c["lvh"] * S; C = 2 * c["lkh"] * S + inner
Lall = L + c["mtp"]
w.add_uint32(f"{A}.context_length", 262144)
w.add_uint32(f"{A}.embedding_length", d)
w.add_uint32(f"{A}.block_count", Lall)
w.add_uint32(f"{A}.feed_forward_length", max(c["ef"], c["sf"]))
w.add_uint32(f"{A}.attention.head_count", c["nh"])
w.add_uint32(f"{A}.attention.head_count_kv", c["nkv"])
w.add_uint32(f"{A}.attention.key_length", hd)
w.add_uint32(f"{A}.attention.value_length", hd)
w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", 1e-6)
w.add_uint32(f"{A}.rope.dimension_count", c["rope"])
w.add_array(f"{A}.rope.dimension_sections", [11, 11, 10, 0])
w.add_float32(f"{A}.rope.freq_base", 10000000.0)
w.add_uint32(f"{A}.ssm.conv_kernel", c["convk"])
w.add_uint32(f"{A}.ssm.state_size", S)
w.add_uint32(f"{A}.ssm.group_count", c["lkh"])
w.add_uint32(f"{A}.ssm.time_step_rank", c["lvh"])
w.add_uint32(f"{A}.ssm.inner_size", inner)
w.add_uint32(f"{A}.full_attention_interval", c["iv"])
w.add_uint32(f"{A}.expert_count", c["ne"])
w.add_uint32(f"{A}.expert_used_count", c["k"])
w.add_uint32(f"{A}.expert_feed_forward_length", c["ef"])
w.add_uint32(f"{A}.expert_shared_feed_forward_length", c["sf"])
if c["mtp"]:
    w.add_uint32(f"{A}.nextn_predict_layers", c["mtp"])

# plan: (name, numpy shape, kind, std, base); kind q8 = random Q8_0, f32 = random normal F32, ssma = -exp(0.5 N)
plan = [("token_embd.weight", (V, d), "q8", 0.8 / d ** 0.5, 0.0)]
def mat(n, shp, std=None):
    plan.append((n, shp, "q8", std if std is not None else 0.8 / shp[-1] ** 0.5, 0.0))
def f32(n, shp, std, base=0.0):
    plan.append((n, shp, "f32", std, base))
for i in range(Lall):
    p = f"blk.{i}."
    ts = []
    full = i >= L or (i + 1) % c["iv"] == 0
    ts.append((p + "attn_norm.weight", (d,), "f32", 0.1, 1.0))
    ts.append((p + "post_attention_norm.weight", (d,), "f32", 0.1, 1.0))
    if not full:
        ts += [(p + "attn_qkv.weight", (C, d), "q8", 0.8 / d ** 0.5, 0.0), (p + "attn_gate.weight", (inner, d), "q8", 0.8 / d ** 0.5, 0.0),
               (p + "ssm_conv1d.weight", (C, c["convk"]), "f32", 0.3, 0.0), (p + "ssm_dt.bias", (c["lvh"],), "f32", 0.5, 0.0),
               (p + "ssm_a", (c["lvh"],), "ssma", 0, 0), (p + "ssm_alpha.weight", (c["lvh"], d), "q8", 0.8 / d ** 0.5, 0.0),
               (p + "ssm_beta.weight", (c["lvh"], d), "q8", 0.8 / d ** 0.5, 0.0), (p + "ssm_norm.weight", (S,), "f32", 0.1, 1.0),
               (p + "ssm_out.weight", (d, inner), "q8", 0.8 / inner ** 0.5, 0.0)]
    else:
        ts += [(p + "attn_q.weight", (2 * c["nh"] * hd, d), "q8", 0.8 / d ** 0.5, 0.0), (p + "attn_k.weight", (c["nkv"] * hd, d), "q8", 0.8 / d ** 0.5, 0.0),
               (p + "attn_v.weight", (c["nkv"] * hd, d), "q8", 0.8 / d ** 0.5, 0.0), (p + "attn_q_norm.weight", (hd,), "f32", 0.1, 1.0),
               (p + "attn_k_norm.weight", (hd,), "f32", 0.1, 1.0), (p + "attn_output.weight", (d, c["nh"] * hd), "q8", 0.8 / (c["nh"] * hd) ** 0.5, 0.0)]
    ts.append((p + "ffn_gate_inp.weight", (c["ne"], d), "f32", 2.0 / d ** 0.5, 0.0))
    if c["fused"]:
        ts.append((p + "ffn_gate_up_exps.weight", (c["ne"], 2 * c["ef"], d), "q8", 0.8 / d ** 0.5, 0.0))
    else:
        ts.append((p + "ffn_gate_exps.weight", (c["ne"], c["ef"], d), "q8", 0.8 / d ** 0.5, 0.0))
        ts.append((p + "ffn_up_exps.weight", (c["ne"], c["ef"], d), "q8", 0.8 / d ** 0.5, 0.0))
    ts.append((p + "ffn_down_exps.weight", (c["ne"], d, c["ef"]), "q8", 0.8 / c["ef"] ** 0.5, 0.0))
    ts.append((p + "ffn_gate_inp_shexp.weight", (d,), "f32", 1.0 / d ** 0.5, 0.0))
    ts += [(p + "ffn_gate_shexp.weight", (c["sf"], d), "q8", 0.8 / d ** 0.5, 0.0), (p + "ffn_up_shexp.weight", (c["sf"], d), "q8", 0.8 / d ** 0.5, 0.0),
           (p + "ffn_down_shexp.weight", (d, c["sf"]), "q8", 0.8 / c["sf"] ** 0.5, 0.0)]
    if i >= L:  # the MTP block (llama.cpp load_block_mtp)
        ts += [(p + "nextn.eh_proj.weight", (d, 2 * d), "q8", 0.8 / (2 * d) ** 0.5, 0.0), (p + "nextn.enorm.weight", (d,), "f32", 0.1, 1.0),
               (p + "nextn.hnorm.weight", (d,), "f32", 0.1, 1.0), (p + "nextn.shared_head_norm.weight", (d,), "f32", 0.1, 1.0)]
    plan += sorted(ts)
plan += [("output_norm.weight", (d,), "f32", 0.1, 1.0), ("output.weight", (V, d), "q8", 3.0 / d ** 0.5, 0.0)]

def q8_bytes_shape(shp):
    return tuple(shp[:-1]) + (shp[-1] // 32 * 34,)
for n, shp, kind, std, base in plan:
    if kind == "q8":
        bs = q8_bytes_shape(shp)
        w.add_tensor_info(n, bs, np.dtype(np.uint8), int(np.prod(bs)), raw_dtype=Q.Q8_0)
    else:
        w.add_tensor_info(n, shp, np.dtype(np.float32), int(np.prod(shp)) * 4)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for n, shp, kind, std, base in plan:
    if kind == "q8":
        rows = int(np.prod(shp[:-1])); cols = shp[-1]; nb = cols // 32
        a = np.empty((rows, nb, 34), dtype=np.uint8)
        step = max(1, (64 << 20) // cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            q = rng.integers(-127, 128, size=(r1 - r0, nb, 32), dtype=np.int8)
            dsc = (std / 73.3 * rng.uniform(0.8, 1.2, size=(r1 - r0, nb))).astype(np.float16)
            a[r0:r1, :, 0:2] = dsc.view(np.uint8).reshape(r1 - r0, nb, 2)
            a[r0:r1, :, 2:] = q.view(np.uint8)
        w.write_tensor_data(a.reshape(q8_bytes_shape(shp)))
        del a
    elif kind == "ssma":
        w.write_tensor_data((-np.exp(0.5 * rng.standard_normal(shp))).astype(np.float32))
    else:
        w.write_tensor_data((base + std * rng.standard_normal(shp)).astype(np.float32))
w.close()
print("wrote", out, c, "V", V)
