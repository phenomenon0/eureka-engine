// trunk_engine.js (WEB-TRUNK-PORT): the family trunk in the browser. One shape-generic WebGPU graph for the Qwen3.5 hybrid
// (Gated DeltaNet + gated full attention every 4th layer, dense or MoE FFN) read straight from a GGUF: every Qwen3.5 size
// (0.8B, 2B, 4B, 9B, 27B; the 35B-A3B / 122B-A10B / 397B-A17B MoE), Kev, and clef-flash (arch "clef": the Qwen3.5-9B trunk +
// Clef's decision decoder, which this chat path does not run). Weights are uploaded in their GGUF block layout, no conversion,
// in any of TRUNK_QUANTS. The context is any length that fits the device (the KV cache is f16, one buffer a full-attention layer).
//   const v = trunk_check(g)                  // {ok, problems[], shape}: can this graph run it, and if not, why
//   const e = await trunk_open(read, {ctx})   // read(offset, length) -> Uint8Array over the .gguf
//   const {id, ms} = await e.step(token)      // one token in, the greedy next token out (GPU argmax); e.logits() reads them all
//   const {id} = await e.prefill(tokens)      // a whole prompt in a few GPU submissions (no per-token round trip)
//   const {hidden} = await e.step(token, {wantHidden: true, noHead: true})   // CLEF-HEAD: the final-normed hidden state, without the vocab head
// The shipped 4B engine (q35_*.js) is separate and unchanged.
import { gguf_read } from "../engine/gguf.js";
import * as K from "./trunk_kernels.js";
import { iq_grids } from "./trunk_iq_grids.js";

export const TRUNK_CAPACITIES = [512, 2048, 4096, 8192, 32768, 131072, 262144];

export function trunk_shape(g) {
  const kv = g.kv, a = kv["general.architecture"], k = (n) => kv[`${a}.${n}`];
  // block_count includes the NextN / MTP draft block(s) appended after the trunk; the chat path runs the trunk only
  const nextn = k("nextn_predict_layers") ?? 0, n_layer = k("block_count") - nextn, fai = k("full_attention_interval") ?? 4;
  const rec = (k("attention.recurrent_layers") ?? Array.from({ length: n_layer }, (_, i) => (i + 1) % fai !== 0)).slice(0, n_layer).map(Boolean);
  const nexp = k("expert_count") ?? 0;
  return { arch: a, n_layer, nextn, d: k("embedding_length"), ff: k("feed_forward_length"), nh: k("attention.head_count"),
    nkv: k("attention.head_count_kv"), hd: k("attention.key_length"), hdv: k("attention.value_length"),
    S: k("ssm.state_size"), hk: k("ssm.group_count"), hv: k("ssm.time_step_rank"), inner: k("ssm.inner_size"),
    convk: k("ssm.conv_kernel"), rope: k("rope.dimension_count"), base: k("rope.freq_base"),
    eps: k("attention.layer_norm_rms_epsilon"), rec, vocab: kv["tokenizer.ggml.tokens"]?.length,
    eos: kv["tokenizer.ggml.eos_token_id"], ctx_train: k("context_length"),
    moe: nexp > 0, nexp, kexp: k("expert_used_count") ?? 0, ffe: k("expert_feed_forward_length") ?? 0,
    ffs: k("expert_shared_feed_forward_length") ?? 0 };
}

const ARCHS = ["qwen35", "qwen35moe", "clef"];
const pow2 = (n) => n > 0 && (n & (n - 1)) === 0;

/** Can the trunk graph run this file? Every refusal is a sentence for the person.
 *  headless = a decision file whose vocab head (output.weight) the readout reads on the CPU, never the GPU. */
export function trunk_check(g, { headless = false } = {}) {
  const s = trunk_shape(g), problems = [], by = new Map(g.tensors.map(t => [t.name, t]));
  const ok_arch = ARCHS.includes(s.arch);
  if (!ok_arch) problems.push(`It's a "${s.arch}" model; the trunk graph is the Qwen3.5 hybrid (qwen35, qwen35moe, and clef = the Qwen3.5-9B trunk with Clef's head).`);
  else {
    if (s.hd !== s.hdv) problems.push(`Its key and value head sizes differ (${s.hd}, ${s.hdv}).`);
    if (!(s.hd >= 32 && s.hd <= 512 && s.hd % 32 === 0)) problems.push(`Its head size ${s.hd} isn't a multiple of 32 up to 512.`);
    if (!(pow2(s.S) && s.S >= 16 && s.S <= 256)) problems.push(`Its delta-net state size ${s.S} isn't a power of two from 16 to 256.`);
    if (!(s.convk >= 2 && s.convk <= 16)) problems.push(`Its conv kernel ${s.convk} isn't 2 to 16.`);
    if (s.inner !== s.hv * s.S) problems.push(`ssm.inner_size ${s.inner} isn't time_step_rank x state_size.`);
    if (s.hv % s.hk || s.nh % s.nkv) problems.push("Its head counts don't divide (the kernels tile k-heads over v-heads and share kv-heads).");
    if (s.rope > s.hd || s.rope % 2) problems.push(`rope.dimension_count ${s.rope} doesn't fit a ${s.hd}-wide head.`);
    if (s.moe && !(s.kexp >= 1 && s.kexp <= s.nexp && s.nexp <= 1024 && s.ffe > 0))
      problems.push(`Its experts (${s.nexp}, ${s.kexp} used, width ${s.ffe}) aren't a routing the kernels take (up to 1024 experts).`);
    if (s.arch === "qwen35moe" && !s.moe) problems.push("It's a qwen35moe file without expert_count.");
  }
  const head = headless ? /^(blk\.\d+\.|token_embd|output_norm)/ : /^(blk\.\d+\.|token_embd|output)/;
  const blk = (n) => { const m = /^blk\.(\d+)\./.exec(n); return m ? +m[1] : -1; };
  const used = g.tensors.filter(t => head.test(t.name) && blk(t.name) < s.n_layer);   // the chat path's tensors (not dec.* / decision.* / the MTP block)
  const small = /(_norm\.weight|ssm_a|ssm_dt\.bias|ssm_conv1d\.weight)$/;   // read on the CPU as f32 (F32, F16, BF16)
  const bad = [...new Set(used.filter(t => small.test(t.name) ? !["F32", "F16", "BF16"].includes(t.type) : !K.TRUNK_QUANTS.includes(t.type)).map(t => t.type))];
  if (bad.length) problems.push(`Its weights use ${bad.map(b => b === "unknown" ? "a quant type this reader doesn't know" : b).join(", ")}; the trunk kernels read ${K.TRUNK_QUANTS.join(", ")}.`);
  const odd = used.find(t => !small.test(t.name) && K.QUANT_BLOCK[t.type] && t.dims[0] % K.QUANT_BLOCK[t.type]);
  if (odd) problems.push(`${odd.name} is ${odd.dims[0]} wide, not a multiple of its ${odd.type} block (${K.QUANT_BLOCK[odd.type]}).`);
  for (const n of ["token_embd.weight", "output_norm.weight"]) if (!by.has(n)) problems.push(`It has no ${n}.`);
  if (ok_arch && s.d && s.n_layer) for (let i = 0; i < s.n_layer && problems.length < 8; i++) {
    const want = s.rec[i] ? ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_a", "ssm_dt.bias", "ssm_conv1d", "ssm_norm", "ssm_out"]
      : ["attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output"];
    const P = (f) => `blk.${i}.${f}${f.includes(".") || f === "ssm_a" ? "" : ".weight"}`;
    let ffn = ["ffn_gate", "ffn_up", "ffn_down"];
    if (s.moe) {
      ffn = ["ffn_gate_inp", "ffn_down_exps", ...(by.has(P("ffn_gate_up_exps")) ? ["ffn_gate_up_exps"] : ["ffn_gate_exps", "ffn_up_exps"])];
      if (by.has(P("ffn_up_shexp"))) ffn.push("ffn_gate_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp");
    }
    for (const f of [...want, "attn_norm", "post_attention_norm", ...ffn]) if (!by.has(P(f))) { problems.push(`Layer ${i} has no ${P(f)}.`); break; }
  }
  const heavy = by.has("output.weight") ? by.get("output.weight") : by.get("token_embd.weight");
  const bytes = used.reduce((a, t) => a + t.bytes, 0);
  return { ok: problems.length === 0, problems, shape: s, tied: !by.has("output.weight"), head_type: heavy?.type, embed_type: by.get("token_embd.weight")?.type,
    trunk_bytes: bytes, skipped: g.tensors.filter(t => !used.includes(t)).map(t => t.name) };
}

/** KV bytes a token (f16 K and V over every full-attention layer). */
export function trunk_kv_bytes_per_token(check) {
  const s = check.shape; return s.rec.filter(r => !r).length * s.nkv * s.hd * 2 * 2;
}

/** Resident bytes at a context: the trunk weights + KV + states + activations (what trunk_open allocates). */
export function trunk_resident(check, ctx) {
  const s = check.shape, nlin = s.rec.filter(r => r).length, C = 2 * s.hk * s.S + s.inner, MC = Math.ceil(ctx / K.ATTN_CH);
  const ff = s.moe ? Math.max(s.kexp * s.ffe, s.ffs) : s.ff;
  return check.trunk_bytes + trunk_kv_bytes_per_token(check) * ctx + nlin * (s.hv * s.S * s.S + C * (s.convk - 1)) * 4
    + s.nh * MC * (s.hd + 2) * 4 + s.vocab * 4 + 8 * 1048576 + 8 * ff * 4 + (s.moe ? s.kexp * s.d * 4 : 0);
}

const f16 = (h) => { const e = (h >> 10) & 31, m = h & 1023, sg = h & 32768 ? -1 : 1;
  return e === 0 ? sg * m * 2 ** -24 : e === 31 ? (m ? NaN : sg * Infinity) : sg * (1 + m / 1024) * 2 ** (e - 15); };

export async function trunk_open(read, { ctx = 4096, onProgress = () => {}, gguf = null, headless = false } = {}) {
  const g = gguf ?? await gguf_read(read), chk = trunk_check(g, { headless });
  if (!chk.ok) throw Object.assign(new Error(chk.problems.join(" ")), { code: "unsupported", problems: chk.problems });
  const s = chk.shape, by = new Map(g.tensors.map(t => [t.name, t])), D = s.hd;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize, maxStorageBuffersPerShaderStage: Math.max(8, Math.min(10, L.maxStorageBuffersPerShaderStage)),
    maxComputeWorkgroupStorageSize: Math.min(32768, L.maxComputeWorkgroupStorageSize) } });
  let lost = null; device.lost.then(i => { lost = i; });
  device.pushErrorScope("validation");   // a kernel or binding the device rejects fails the open, not the output
  const LIM = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 2 ** 31 - 256) / 256) * 256;   // (not & ~255: 2^31 is past int32)
  // the context: one f16 K and one f16 V buffer a full-attention layer, each under the binding limit
  const kvLayer = s.nkv * D * 2, MC = Math.ceil(ctx / K.ATTN_CH);
  const DIRECT_CHUNKS = 64;   // up to 64 chunks (16k tokens) every attention chunk is dispatched directly; past that, indirectly
  const maxCtx = Math.floor((LIM - 16) / kvLayer / K.ATTN_CH) * K.ATTN_CH;
  if (ctx > maxCtx) { device.destroy(); throw Object.assign(new Error(`A ${ctx}-token context needs ${(ctx * kvLayer / 2 ** 20).toFixed(0)} MiB a KV buffer; this GPU binds at most ${(LIM / 2 ** 20).toFixed(0)} MiB, so ${maxCtx} tokens is the most it can hold.`), { code: "ctx", max_ctx: maxCtx }); }
  const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const bufs = []; let resident = 0;
  const mk = (bytes, usage = SU) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage }); bufs.push(b); resident += b.size; return b; };
  const f32 = (n) => mk(n * 4);

  // ---- weights: each tensor its own buffer (big ones split by rows, or by experts, under the binding limit), raw GGUF bytes
  const total = chk.trunk_bytes; let done = 0;
  const PIECE = 64 << 20;
  async function upload(buf, at, len) {   // file bytes [at, at+len) -> buf[0..]
    for (let o = 0; o < len; o += PIECE) {
      const n = Math.min(PIECE, len - o), b = await read(at + o, n);
      const pad = (4 - (n % 4)) % 4, src = pad ? (() => { const c = new Uint8Array(n + pad); c.set(b); return c; })() : b;
      device.queue.writeBuffer(buf, o, src, 0, src.byteLength);
      done += n; onProgress({ done, total });
      if (lost) throw new Error("WebGPU device lost while loading: " + lost.message);
    }
  }
  async function weight(name) {   // 2-D (or 1-D) -> {parts: [{buf, r0, rows}], K = dims[0], rows, type}
    const t = by.get(name), rows = t.dims[1] ?? 1, rb = t.bytes / rows, per = Math.max(1, Math.floor((LIM - 16) / rb));
    const parts = [];
    for (let r0 = 0; r0 < rows; r0 += per) {
      const n = Math.min(per, rows - r0), buf = mk(n * rb + 16);
      await upload(buf, t.abs + r0 * rb, n * rb); parts.push({ buf, r0, rows: n });
    }
    return { parts, K: t.dims[0], rows, type: t.type };
  }
  async function experts(name) {  // 3-D [K, rows, experts] -> {parts: [{buf, e0, ne}], K, rows, type}
    const t = by.get(name), [Kd, rows, ne] = t.dims, eb = t.bytes / ne, per = Math.floor((LIM - 16) / eb);
    if (per < 1) throw new Error(`${name}: one expert (${(eb / 2 ** 20).toFixed(0)} MiB) is bigger than this GPU's ${(LIM / 2 ** 20).toFixed(0)} MiB binding limit.`);
    const parts = [];
    for (let e0 = 0; e0 < ne; e0 += per) {
      const n = Math.min(per, ne - e0), buf = mk(n * eb + 16);
      await upload(buf, t.abs + e0 * eb, n * eb); parts.push({ buf, e0, ne: n });
    }
    return { parts, K: Kd, rows, type: t.type };
  }
  const f32s = async (name) => {   // a small tensor as f32 (F32, F16, BF16)
    const t = by.get(name), b = (await read(t.abs, t.bytes)).slice();
    if (t.type === "F32") return new Float32Array(b.buffer);
    const h = new Uint16Array(b.buffer), out = new Float32Array(h.length);
    if (t.type === "BF16") { const u = new Uint32Array(out.buffer); for (let i = 0; i < h.length; i++) u[i] = h[i] << 16; }
    else for (let i = 0; i < h.length; i++) out[i] = f16(h[i]);
    return out;
  };
  async function packed(arrs) { const n = arrs.reduce((a, x) => a + x.length, 0), all = new Float32Array(n); let o = 0; for (const x of arrs) { all.set(x, o); o += x.length; }
    const b = f32(n); device.queue.writeBuffer(b, 0, all); return b; }
  const grid = K.TRUNK_QUANTS.some(q => K.NEEDS_GRID.has(q) && g.tensors.some(t => t.type === q)) ? (() => { const a = iq_grids(), b = mk(a.byteLength); device.queue.writeBuffer(b, 0, a); return b; })() : null;
  const withGrid = (type, entries) => K.NEEDS_GRID.has(type) ? [...entries, grid] : entries;

  // ---- pipelines
  const pipes = new Map();
  const pipe = (key, code) => { if (!pipes.has(key)) pipes.set(key, device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })); return pipes.get(key); };
  const uni = (words) => { const b = device.createBuffer({ size: Math.max(16, words.length * 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); bufs.push(b);
    const a = new ArrayBuffer(words.length * 4), v = new DataView(a); words.forEach((w, i) => typeof w === "object" ? v.setFloat32(i * 4, w.f, true) : v.setUint32(i * 4, w >>> 0, true));
    device.queue.writeBuffer(b, 0, a); return b; };
  const fl = (x) => ({ f: x });   // a float uniform word (plain numbers are written as u32)
  const ops = [];
  const op = (p, entries, x, y = 1, z = 1, ind = null) => ops.push({ p, bg: device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) }), x, y, z, ind });
  const grid2 = (rows) => { const groups = Math.ceil(rows / 8), gx = Math.min(groups, 32768); return [gx, Math.ceil(groups / gx)]; };
  const gemv = (w, X, Y, { xoff = 0, yoff = 0, acc = 0 } = {}) => {
    for (const part of w.parts) {
      const [gx, gy] = grid2(part.rows);
      op(pipe("g" + w.type, K.gemv(w.type)), withGrid(w.type, [part.buf, X, Y, uni([part.rows, w.K, xoff, yoff, acc, gx, part.r0, 0])]), gx, gy);
    }
  };
  const gemvId = (w, X, Y, sel, { xs = 0, ys, yoff = 0 }) => {
    for (const part of w.parts) {
      const [gx, gy] = grid2(w.rows);
      op(pipe("i" + w.type, K.gemv_id(w.type)), withGrid(w.type, [part.buf, X, Y, uni([w.rows, w.K, xs, ys, yoff, gx, part.e0, part.ne]), sel]), gx, gy, s.kexp);
    }
  };
  const swiglu = (GU, Y, n, blk, gs, uo) => op(pipe("swiglu", K.swiglu), [GU, Y, uni([n, blk, gs, uo])], Math.ceil(n / 256));
  const norm = (X, Wn, Y, { dim, heads = 1, xoff = 0, xs = 0, yoff = 0, ys = 0, mode = 0 }) => op(pipe("norm", K.rmsnorm), [X, Wn, Y, uni([dim, xoff, xs, yoff, ys, mode, fl(s.eps), 0])], heads);

  // ---- activations and state
  const C = 2 * s.hk * s.S + s.inner;
  const ffw = s.moe ? Math.max(s.kexp * s.ffe, s.ffs) : s.ff;
  const x = f32(s.d), xn = f32(s.d), hn = f32(s.d), qkv = f32(C), z = f32(s.inner), ab = f32(2 * s.hv), cv = f32(C), lo = f32(s.inner);
  const qg = f32(s.nh * 2 * D), kvb = f32(2 * s.nkv * D), qa = f32(s.nh * D), ao = f32(s.nh * D);
  const pmb = f32(s.nh * MC * 2), pab = f32(s.nh * MC * D);
  const gu = f32(2 * ffw), sw = f32(ffw), logits = f32(s.vocab), pm = f32(1048576), pi = f32(1048576);
  const ids = mk(16), step = mk(16), args = mk(16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
  const moe = s.moe ? { rl: f32(s.nexp), sel: mk(s.kexp * 4), wt: f32(s.kexp), eo: f32(s.kexp * s.d), sh: f32(s.d + 4) } : null;
  const states = [];

  // embedding (gather) and head
  const embW = await weight("token_embd.weight");
  for (const part of embW.parts) op(pipe("e" + embW.type, K.embed(embW.type)), withGrid(embW.type, [part.buf, ids, x, uni([s.d, part.r0, part.rows, 0])]), Math.ceil(s.d / 256));
  for (let i = 0; i < s.n_layer; i++) {
    const P = (f) => `blk.${i}.${f}`;
    norm(x, await packed([await f32s(P("attn_norm.weight"))]), xn, { dim: s.d });
    if (s.rec[i]) {
      gemv(await weight(P("attn_qkv.weight")), xn, qkv);
      gemv(await weight(P("attn_gate.weight")), xn, z);
      gemv(await weight(P("ssm_alpha.weight")), xn, ab);
      gemv(await weight(P("ssm_beta.weight")), xn, ab, { yoff: s.hv });
      const convw = await packed([await f32s(P("ssm_conv1d.weight"))]), cst = f32(C * (s.convk - 1)), st = f32(s.hv * s.S * s.S);
      states.push(cst, st);
      op(pipe("conv" + s.convk, K.conv(s.convk)), [qkv, convw, cst, cv, uni([C, 0, 0, 0])], Math.ceil(C / 256));
      const pr = await packed([await f32s(P("ssm_dt.bias")), await f32s(P("ssm_a")), await f32s(P("ssm_norm.weight"))]);
      op(pipe("delta" + s.S, K.delta(s.S)), [cv, ab, pr, st, z, lo, uni([s.hk, s.hv, fl(s.eps), 0])], s.hv);
      gemv(await weight(P("ssm_out.weight")), lo, x, { acc: 1 });
    } else {
      gemv(await weight(P("attn_q.weight")), xn, qg);
      gemv(await weight(P("attn_k.weight")), xn, kvb);
      gemv(await weight(P("attn_v.weight")), xn, kvb, { yoff: s.nkv * D });
      const nw = await packed([await f32s(P("attn_q_norm.weight")), await f32s(P("attn_k_norm.weight"))]);
      const kc = mk(ctx * kvLayer), vc = mk(ctx * kvLayer);
      op(pipe("prep" + D, K.attnprep(D)), [qg, kvb, nw, qa, kc, vc, uni([s.nh, s.nkv, s.rope, fl(s.eps), fl(s.base), ctx, 0, 0]), step], s.nh + s.nkv);
      op(pipe("part" + D, K.attn_part(D)), [qa, kc, vc, pmb, pab, uni([s.nh, s.nkv, MC, fl(1 / Math.sqrt(D)), ctx, 0, 0, 0]), step], MC, s.nh, 1, MC > DIRECT_CHUNKS ? args : null);
      op(pipe("merge" + D, K.attn_merge(D)), [pmb, pab, qg, ao, uni([s.nh, MC, 0, 0]), step], s.nh);
      gemv(await weight(P("attn_output.weight")), ao, x, { acc: 1 });
    }
    norm(x, await packed([await f32s(P("post_attention_norm.weight"))]), hn, { dim: s.d });
    if (!s.moe) {
      gemv(await weight(P("ffn_gate.weight")), hn, gu);
      gemv(await weight(P("ffn_up.weight")), hn, gu, { yoff: s.ff });
      swiglu(gu, sw, s.ff, s.ff, 0, s.ff);
      gemv(await weight(P("ffn_down.weight")), sw, x, { acc: 1 });
    } else {
      const KF = s.kexp * s.ffe;
      gemv(await weight(P("ffn_gate_inp.weight")), hn, moe.rl);
      op(pipe("topk", K.topk), [moe.rl, moe.sel, moe.wt, uni([s.nexp, s.kexp, 0, 0])], 1);
      if (by.has(P("ffn_gate_up_exps.weight"))) {
        gemvId(await experts(P("ffn_gate_up_exps.weight")), hn, gu, moe.sel, { ys: 2 * s.ffe });
        swiglu(gu, sw, KF, s.ffe, 2 * s.ffe, s.ffe);
      } else {
        gemvId(await experts(P("ffn_gate_exps.weight")), hn, gu, moe.sel, { ys: s.ffe });
        gemvId(await experts(P("ffn_up_exps.weight")), hn, gu, moe.sel, { ys: s.ffe, yoff: KF });
        swiglu(gu, sw, KF, KF, 0, KF);
      }
      gemvId(await experts(P("ffn_down_exps.weight")), sw, moe.eo, moe.sel, { xs: s.ffe, ys: s.d });
      const shared = by.has(P("ffn_up_shexp.weight"));
      if (shared) {   // the shared expert runs after the routed ones, so gu / sw are free to reuse
        const fs = by.get(P("ffn_up_shexp.weight")).dims[1];
        gemv(await weight(P("ffn_gate_shexp.weight")), hn, gu);
        gemv(await weight(P("ffn_up_shexp.weight")), hn, gu, { yoff: fs });
        swiglu(gu, sw, fs, fs, 0, fs);
        gemv(await weight(P("ffn_down_shexp.weight")), sw, moe.sh);
        gemv(await weight(P("ffn_gate_inp_shexp.weight")), hn, moe.sh, { yoff: s.d });
      }
      op(pipe("moeadd", K.moe_add), [moe.eo, moe.wt, moe.sh, x, uni([s.d, s.kexp, shared ? 1 : 0, 0])], Math.ceil(s.d / 256));
    }
  }
  norm(x, await packed([await f32s("output_norm.weight")]), hn, { dim: s.d });
  const nBody = ops.length;   // CLEF-HEAD: ops past this one are the vocab head + argmax
  const headW = by.has("output.weight") && !headless ? await weight("output.weight") : embW;
  gemv(headW, hn, logits);
  const nb = Math.ceil(s.vocab / 4096);
  op(pipe("amax", K.argmax), [logits, pm, pi, uni([s.vocab, 0, 0, 0])], nb);
  op(pipe("amax", K.argmax), [logits, pm, pi, uni([nb, 1, 0, 0])], 1);
  await device.queue.onSubmittedWorkDone();
  if (lost) throw new Error("WebGPU device lost: " + lost.message);
  const invalid = await device.popErrorScope();
  if (invalid) { device.destroy(); throw new Error("WebGPU rejected the trunk graph: " + invalid.message); }

  const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const lrb = device.createBuffer({ size: s.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const hrb = device.createBuffer({ size: s.d * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  // per-token records [id, pos, chunks, NH, 1, 0, 0, 0] (32 bytes); each token copies its id, pos and attention dispatch size
  // into place before its pass, so a run of tokens is one submission
  let seq = null, seqCap = 0;
  const BATCH = 64;
  let pos = 0;
  async function run(tokens, { wantLogits = false, wantHidden = false, noHead = false } = {}) {
    const n = tokens.length;
    if (pos + n > ctx) throw new Error(`The context is full (${ctx} tokens).`);
    const t0 = performance.now();
    if (n > seqCap) { seq?.destroy(); seqCap = Math.max(n, 64); seq = device.createBuffer({ size: seqCap * 32, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); }
    const rec = new Uint32Array(n * 8);
    for (let i = 0; i < n; i++) { const p = pos + i; rec.set([tokens[i], p, Math.ceil((p + 1) / K.ATTN_CH), s.nh, 1], i * 8); }
    device.queue.writeBuffer(seq, 0, rec);
    const head = !(noHead && !wantLogits);
    for (let b = 0; b < n; b += BATCH) {
      const enc = device.createCommandEncoder();
      for (let i = b; i < Math.min(n, b + BATCH); i++) {
        enc.copyBufferToBuffer(seq, i * 32, ids, 0, 4); enc.copyBufferToBuffer(seq, i * 32 + 4, step, 0, 4); enc.copyBufferToBuffer(seq, i * 32 + 8, args, 0, 12);
        const pass = enc.beginComputePass(), last = i === n - 1;
        for (const o of last && head ? ops : ops.slice(0, nBody)) {
          pass.setPipeline(o.p); pass.setBindGroup(0, o.bg);
          if (o.ind) pass.dispatchWorkgroupsIndirect(o.ind, 0); else pass.dispatchWorkgroups(o.x, o.y, o.z);
        }
        pass.end();
      }
      if (b + BATCH >= n) {
        if (head) { enc.copyBufferToBuffer(pm, 1048575 * 4, rb, 0, 4); enc.copyBufferToBuffer(pi, 1048575 * 4, rb, 4, 4); }
        if (wantLogits) enc.copyBufferToBuffer(logits, 0, lrb, 0, s.vocab * 4);
        if (wantHidden) enc.copyBufferToBuffer(hn, 0, hrb, 0, s.d * 4);
      }
      device.queue.submit([enc.finish()]);
    }
    let v = null, Lg = null, Hd = null;
    if (head) { await rb.mapAsync(GPUMapMode.READ); v = new DataView(rb.getMappedRange().slice(0)); rb.unmap(); }
    if (wantHidden) { await hrb.mapAsync(GPUMapMode.READ); Hd = new Float32Array(hrb.getMappedRange().slice(0)); hrb.unmap(); }
    if (wantLogits) { await lrb.mapAsync(GPUMapMode.READ); Lg = new Float32Array(lrb.getMappedRange().slice(0)); lrb.unmap(); }
    if (!head) await device.queue.onSubmittedWorkDone();
    if (lost) throw new Error("WebGPU device lost: " + lost.message);
    pos += n;
    return { id: v ? v.getUint32(4, true) : -1, logit: v ? v.getFloat32(0, true) : NaN, ms: performance.now() - t0, logits: Lg, hidden: Hd };
  }
  const stepTok = (token, o) => run([token], o);
  async function reset() {
    pos = 0;
    const enc = device.createCommandEncoder();
    for (const b of states) enc.clearBuffer(b);
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
  }
  await reset();
  return { shape: s, check: chk, ctx, max_ctx: maxCtx, dispatches: ops.length, resident_bytes: resident,
    limits: { binding: LIM, adapter_binding: L.maxStorageBufferBindingSize, adapter_buffer: L.maxBufferSize },
    adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null,
    step: stepTok, prefill: (tokens, o) => run(tokens, o), reset, get pos() { return pos; }, free() { seq?.destroy(); for (const b of bufs) b.destroy(); device.destroy(); } };
}
