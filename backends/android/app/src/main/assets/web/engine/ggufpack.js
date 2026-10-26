// ggufpack.js (BENDWEB-SHOWCASE): a GGUF file read as the engine's web pack, in place (no conversion pass, no copy).
//   const v = await gguf_pack(read)     // read(offset, length) -> Uint8Array over the .gguf (File.slice, HTTP Range, fs)
//   v.ok, v.problems[]: can the web graph run it, and if not, why (each reason a sentence for the person)
//   v.manifest, v.source(path, offset, length): what q35_open(g, v.source, v.manifest, {verify:false}) reads
// The layout is webpack.py's (bendweb/build2-q8/engine/tools): every layer tensor bin-packed first fit, largest first, into
// 32 MiB blocks at 16-byte alignment; wtab words fm<<28 | block<<21 | offset/16; token_embd (the tied head) = head.bin,
// output_norm = norm.bin; Q8_0 tensors in the quant pack's row-split layout ([cols int8][cols/32 f16 scales] a row),
// every K-quant and F32 tensor byte for byte. So a GGUF of the gated pack's model yields the gated pack's exact bytes
// (tools/ggufpack-check.mjs compares all 73 files).
import { gguf_read } from "./gguf.js";

const BLOCK = 1 << 25;
const FIELDS = ["attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
  "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_a", "ssm_dt.bias",
  "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight",
  "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight"];
const FM = { F32: 0, Q8_0: 1, Q4_K: 3, Q6_K: 4, Q5_K: 9 };   // the codes q35_plan.js decodes
// the one geometry the web graph is built for (q35_plan.js q35_tensors), as GGUF keys
const GEOMETRY = [["block_count", "n_layers", 32], ["attention.head_count", "n_heads", 16], ["attention.head_count_kv", "n_kv_heads", 4],
  ["embedding_length", "d_model", 2560], ["attention.key_length", "head_dim", 256], ["feed_forward_length", "d_ff", 9216],
  ["full_attention_interval", "full_attention_interval", 4], ["ssm.state_size", "ssm_state_size", 128], ["ssm.inner_size", "ssm_inner_size", 4096],
  ["ssm.conv_kernel", "ssm_conv_kernel", 4], ["rope.dimension_count", "rope_dim", 64], ["rope.freq_base", "rope_freq_base", 10000000]];
export const WEB_QUANTS = Object.keys(FM);

/** A short name for what the header says: "Qwen3.5-4B", "Qwen3.5 4B Q4_K_M (Kev)" ... */
export function gguf_describe(g) {
  const kv = g.kv, arch = kv["general.architecture"];
  const ft = { 7: "Q8_0", 15: "Q4_K_M", 14: "Q4_K_S", 17: "Q5_K_M", 16: "Q5_K_S", 18: "Q6_K", 1: "F16", 0: "F32", 30: "IQ4_XS", 25: "IQ4_NL" }[kv["general.file_type"]];
  const types = {};
  for (const t of g.tensors) types[t.type] = (types[t.type] ?? 0) + 1;
  return { arch, name: kv["general.name"] ?? null, size: kv["general.size_label"] ?? null, fileType: ft ?? (kv["general.file_type"] != null ? "type " + kv["general.file_type"] : null),
    types, layers: kv[`${arch}.block_count`], width: kv[`${arch}.embedding_length`], experts: kv[`${arch}.expert_count`] ?? null,
    decision: kv[`${arch}.decision.type`] ?? null, vocab: kv["tokenizer.ggml.tokens"]?.length ?? null, tensors: g.tensors.length };
}

export async function gguf_pack(read, { decision = false } = {}) {
  const g = await gguf_read(read), kv = g.kv, d = gguf_describe(g), problems = [];
  const arch = d.arch, by = new Map(g.tensors.map(t => [t.name, t]));
  if (arch !== "qwen35") {
    problems.push(`It's a "${arch}" model${d.name ? ` (${d.name})` : ""}. The web engine's graph is Qwen3.5's (the qwen35 hybrid: Gated DeltaNet + full attention every 4th layer); a ${arch} model needs its own graph${d.experts ? ` — and its ${d.experts}-expert MoE routing` : ""}.`);
  } else {
    const off = GEOMETRY.filter(([k, , want]) => kv[`qwen35.${k}`] !== want);
    if (off.length) problems.push(`It's Qwen3.5, but not the 4B shape the web graph is built for (${off.map(([k, , want]) => `${k} ${kv["qwen35." + k]}, want ${want}`).join("; ")}). Another size needs its own kernels and graph.`);
  }
  const unsupported = [...new Set(g.tensors.filter(t => !FM.hasOwnProperty(t.type)).map(t => t.type))];
  if (unsupported.length) problems.push(`Its weights use ${unsupported.join(", ")}; the web engine's kernels read ${WEB_QUANTS.join(", ")} only.`);
  const emb = by.get("token_embd.weight");
  // a decision model (Kev) may carry a Q4_K embedding: the showcase's engine copy gathers it (emit/k_embed_q4k.wgsl)
  if (arch === "qwen35" && emb && emb.type !== "Q6_K" && !(decision && emb.type === "Q4_K")) problems.push(`Its embedding (the tied output head) is ${emb.type}; the web graph's gather and head read Q6_K.`);
  if (arch === "qwen35" && by.has("output.weight")) problems.push("It has an untied output head; the web graph reads the tied one.");
  if (d.vocab != null && d.vocab !== 248320) problems.push(`Its vocabulary has ${d.vocab} tokens; the web graph's head is built for 248,320.`);
  if (decision && kv["qwen35.decision.type"] !== "kev") problems.push("It isn't a Kev decision model (no qwen35.decision.type = kev).");
  if (problems.length) return { ok: false, problems, describe: d, gguf: g };

  const n = kv["qwen35.block_count"];
  const items = [];
  for (let i = 0; i < n; i++) FIELDS.forEach((f, fi) => {
    const t = by.get(`blk.${i}.${f}`);
    if (t) items.push({ t, layer: i, field: fi, bytes: t.bytes });
  });
  const order = [...items].sort((a, b) => b.bytes - a.bytes || a.layer - b.layer || a.field - b.field);
  const bins = [];
  for (const it of order) {
    let placed = false;
    for (let bi = 0; bi < bins.length && !placed; bi++) {
      const at = (bins[bi].used + 15) & ~15;
      if (at + it.bytes <= BLOCK) { bins[bi].used = at + it.bytes; bins[bi].items.push(it); it.block = bi; it.at = at; placed = true; }
    }
    if (!placed) { it.block = bins.length; it.at = 0; bins.push({ used: it.bytes, items: [it] }); }
  }
  const wtab = new Uint8Array(4096), wv = new DataView(wtab.buffer);
  for (const it of items) wv.setUint32(it.layer * 128 + 4 * (1 + it.field), ((FM[it.t.type] << 28) | (it.block << 21) | (it.at / 16)) >>> 0, true);
  const nrm = by.get("output_norm.weight");
  const files = [...bins.map((_, i) => ({ path: `blocks/b${i}.bin`, bytes: BLOCK })), { path: "wtab.bin", bytes: 4096 },
    { path: "head.bin", bytes: emb.bytes }, { path: "norm.bin", bytes: nrm.bytes }];
  for (const f of files) f.chunks = new Array(Math.ceil(f.bytes / BLOCK)).fill(null);   // no digests (verify:false); the engine counts them
  const config = { architecture: "qwen35", vocab_size: d.vocab, tied_embeddings: true, eos_token_id: kv["tokenizer.ggml.eos_token_id"] ?? null };
  for (const [k, name] of GEOMETRY) config[name] = kv[`qwen35.${k}`];
  const manifest = { format: "llama-bend-pack-q35-quant-web/v0", dtype: "mixed-quant", source: { file: "(local GGUF)", types: d.types }, config,
    tensors: { "output_norm.weight": { offset: 0, bytes: nrm.bytes, src_type: "F32" },
      "token_embd.weight": { offset: 0, bytes: emb.bytes, src_type: emb.type, row_bytes: emb.type === "Q4_K" ? 1440 : 2100 },
      "output.weight": { src_type: emb.type, alias_of: "token_embd.weight" } },
    web: { blocks: bins.length, block_bytes: BLOCK, pack_sha256: null, files } };

  // a tensor's bytes as the pack stores them (Q8_0: the row split)
  const tensorBytes = async (t) => {
    const raw = await read(t.abs, t.bytes);
    if (t.type !== "Q8_0") return raw;
    const cols = t.dims[0], rows = t.bytes / (cols / 32 * 34), rb = cols / 32 * 34, out = new Uint8Array(t.bytes);
    for (let r = 0; r < rows; r++) {
      const src = r * rb, q = r * rb, s = q + cols;
      for (let b = 0; b < cols / 32; b++) {
        out.set(raw.subarray(src + b * 34 + 2, src + b * 34 + 34), q + b * 32);
        out[s + b * 2] = raw[src + b * 34]; out[s + b * 2 + 1] = raw[src + b * 34 + 1];
      }
    }
    return out;
  };
  let last = null;   // the engine reads a block whole, once: keep the one being read
  const block = async (bi) => {
    if (last?.bi === bi) return last.data;
    const data = new Uint8Array(BLOCK);
    for (const it of bins[bi].items) data.set(await tensorBytes(it.t), it.at);
    last = { bi, data };
    return data;
  };
  const source = async (path, offset = 0, length) => {
    const f = files.find(x => x.path === path);
    if (!f) throw new Error("ggufpack: no file " + path);
    const len = length ?? f.bytes - offset;
    if (path === "wtab.bin") return wtab.slice(offset, offset + len);
    if (path === "head.bin") return read(emb.abs + offset, len);
    if (path === "norm.bin") return read(nrm.abs + offset, len);
    const m = /^blocks\/b(\d+)\.bin$/.exec(path);
    return (await block(Number(m[1]))).slice(offset, offset + len);
  };
  // Kev's decision head: cls.output (Q8_0, 2560 -> n_embd_out) + bias, applied by the worker to the final-normed rows
  const cls = decision ? { weight: by.get("cls.output.weight"), bias: by.get("cls.output.bias"), out: kv["qwen35.embedding_length_out"],
    temperature: kv["qwen35.decision.temperature.choice"] ?? 1, template: kv["tokenizer.chat_template.systemone"] } : null;
  return { ok: true, problems, describe: d, gguf: g, manifest, source, cls, read };
}

/** SHA-256 over the GGUF's vocabulary and merges: equal to the bundled tokenizer's source = the bundled tokenizer serves it. */
export async function gguf_vocab_sha256(g) {
  const enc = new TextEncoder();
  const parts = [...(g.kv["tokenizer.ggml.tokens"] ?? []), "\u0000merges\u0000", ...(g.kv["tokenizer.ggml.merges"] ?? [])];
  const bytes = enc.encode(parts.join("\n"));
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(x => x.toString(16).padStart(2, "0")).join("");
}
