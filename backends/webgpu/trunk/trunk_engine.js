// trunk_engine.js (WEB-TRUNK-PORT): the family trunk in the browser. One shape-generic WebGPU graph for the Qwen3.5 hybrid
// (Gated DeltaNet + gated full attention every 4th layer) read straight from a GGUF (any size whose dims the kernels take):
// clef-flash (arch "clef": the Qwen3.5-9B trunk + Clef's decision decoder, which this chat path does not run), Qwen3.5-9B,
// Qwen3.5-0.8B / Kev-0.8B, and in principle the 27B. Weights are uploaded in their GGUF block layout, no conversion.
//   const v = trunk_check(g)                  // {ok, problems[], shape}: can this graph run it, and if not, why
//   const e = await trunk_open(read, {ctx})   // read(offset, length) -> Uint8Array over the .gguf
//   const {id, ms} = await e.step(token)      // one token in, the greedy next token out (GPU argmax); e.logits() reads them all
//   const {hidden} = await e.step(token, {wantHidden: true, noHead: true})   // CLEF-HEAD: the final-normed hidden state, without the vocab head
// The shipped 4B engine (q35_*.js) is separate and unchanged.
import { gguf_read } from "../engine/gguf.js";
import * as K from "./trunk_kernels.js";

export const TRUNK_CAPACITIES = [512, 2048, 4096];

export function trunk_shape(g) {
  const kv = g.kv, a = kv["general.architecture"], k = (n) => kv[`${a}.${n}`];
  const n_layer = k("block_count"), fai = k("full_attention_interval") ?? 4;
  const rec = k("attention.recurrent_layers") ?? Array.from({ length: n_layer }, (_, i) => (i + 1) % fai !== 0);
  return { arch: a, n_layer, d: k("embedding_length"), ff: k("feed_forward_length"), nh: k("attention.head_count"),
    nkv: k("attention.head_count_kv"), hd: k("attention.key_length"), hdv: k("attention.value_length"),
    S: k("ssm.state_size"), hk: k("ssm.group_count"), hv: k("ssm.time_step_rank"), inner: k("ssm.inner_size"),
    convk: k("ssm.conv_kernel"), rope: k("rope.dimension_count"), base: k("rope.freq_base"),
    eps: k("attention.layer_norm_rms_epsilon"), rec, vocab: kv["tokenizer.ggml.tokens"]?.length,
    eos: kv["tokenizer.ggml.eos_token_id"] };
}

/** Can the trunk graph run this file? Every refusal is a sentence for the person. */
export function trunk_check(g) {
  const s = trunk_shape(g), problems = [], by = new Map(g.tensors.map(t => [t.name, t]));
  if (s.arch !== "qwen35" && s.arch !== "clef")
    problems.push(`It's a "${s.arch}" model; the trunk graph is the Qwen3.5 hybrid (qwen35, and clef = the Qwen3.5-9B trunk with Clef's head).`);
  else {
    const need = [["attention.key_length", s.hd, 256], ["attention.value_length", s.hdv, 256], ["ssm.state_size", s.S, 128], ["ssm.conv_kernel", s.convk, 4]];
    const off = need.filter(([, v, w]) => v !== w);
    if (off.length) problems.push(`Its ${off.map(([n, v, w]) => `${n} is ${v} (the kernels take ${w})`).join("; ")}.`);
    if (s.inner !== s.hv * s.S) problems.push(`ssm.inner_size ${s.inner} isn't time_step_rank x state_size.`);
    if (s.hv % s.hk || s.nh % s.nkv) problems.push("Its head counts don't divide (the kernels tile k-heads over v-heads and share kv-heads).");
    if (s.d % 256 || s.ff % 256) problems.push(`Its widths (${s.d}, ${s.ff}) aren't multiples of 256 (one K-quant block).`);
    if (s.rope > 256 || s.rope % 2) problems.push(`rope.dimension_count ${s.rope} doesn't fit a 256-wide head.`);
  }
  const used = g.tensors.filter(t => /^(blk\.\d+\.|token_embd|output)/.test(t.name));   // the chat path's tensors (not dec.* / decision.*)   // the chat path's tensors (not dec.* / decision.*)
  const bad = [...new Set(used.filter(t => !K.TRUNK_QUANTS.includes(t.type)).map(t => t.type))];
  if (bad.length) problems.push(`Its weights use ${bad.map(b => b === "unknown" ? "a quant type this reader doesn't know" : b).join(", ")}; the trunk kernels read ${K.TRUNK_QUANTS.join(", ")}.`);
  for (const n of ["token_embd.weight", "output_norm.weight"]) if (!by.has(n)) problems.push(`It has no ${n}.`);
  if ((s.arch === "qwen35" || s.arch === "clef") && s.d && s.n_layer) for (let i = 0; i < s.n_layer && problems.length < 8; i++) {
    const want = s.rec[i] ? ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_a", "ssm_dt.bias", "ssm_conv1d", "ssm_norm", "ssm_out"]
      : ["attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output"];
    for (const f of [...want, "attn_norm", "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down"]) {
      const n = `blk.${i}.${f}${f.includes(".") || f === "ssm_a" ? "" : ".weight"}`;
      if (!by.has(n)) { problems.push(`Layer ${i} has no ${n}.`); break; }
    }
  }
  const heavy = by.has("output.weight") ? by.get("output.weight") : by.get("token_embd.weight");
  const bytes = used.reduce((a, t) => a + t.bytes, 0);
  return { ok: problems.length === 0, problems, shape: s, tied: !by.has("output.weight"), head_type: heavy?.type, embed_type: by.get("token_embd.weight")?.type,
    trunk_bytes: bytes, skipped: g.tensors.filter(t => !used.includes(t)).map(t => t.name) };
}

/** Resident bytes at a context: the trunk weights + KV + states + activations (what trunk_open allocates). */
export function trunk_resident(check, ctx) {
  const s = check.shape, nfull = s.rec.filter(r => !r).length, nlin = s.n_layer - nfull, C = 2 * s.hk * s.S + s.inner;
  return check.trunk_bytes + nfull * ctx * s.nkv * 256 * 4 * 2 + nlin * (s.hv * s.S * s.S + C * 3) * 4 + s.nh * ctx * 4 + s.vocab * 4 + 8 * 1048576 + 64 * s.ff * 4;
}

export async function trunk_open(read, { ctx = 4096, onProgress = () => {}, gguf = null } = {}) {
  const g = gguf ?? await gguf_read(read), chk = trunk_check(g);
  if (!chk.ok) throw Object.assign(new Error(chk.problems.join(" ")), { code: "unsupported", problems: chk.problems });
  const s = chk.shape, by = new Map(g.tensors.map(t => [t.name, t]));
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize, maxStorageBuffersPerShaderStage: Math.max(8, Math.min(10, L.maxStorageBuffersPerShaderStage)) } });
  let lost = null; device.lost.then(i => { lost = i; });
  const LIM = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 1 << 30) & ~255;
  const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const bufs = []; let resident = 0;
  const mk = (bytes, usage = SU) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage }); bufs.push(b); resident += b.size; return b; };
  const f32 = (n) => mk(n * 4);

  // ---- weights: each tensor its own buffer (big ones split by rows under the binding limit), raw GGUF bytes
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
  const rowBytes = (t) => t.bytes / t.dims[1];
  async function weight(name) {   // -> [{buf, r0, rows}], with K = dims[0], type
    const t = by.get(name), rb = rowBytes(t), rows = t.dims[1] ?? 1, per = Math.max(1, Math.floor((LIM - 16) / rb));
    const parts = [];
    for (let r0 = 0; r0 < rows; r0 += per) {
      const n = Math.min(per, rows - r0), buf = mk(n * rb + 16);
      await upload(buf, t.abs + r0 * rb, n * rb); parts.push({ buf, r0, rows: n });
    }
    return { parts, K: t.dims[0], rows, type: t.type };
  }
  const f32s = async (name) => { const t = by.get(name); return new Float32Array((await read(t.abs, t.bytes)).slice().buffer); };
  async function packed(arrs) { const n = arrs.reduce((a, x) => a + x.length, 0), all = new Float32Array(n); let o = 0; for (const x of arrs) { all.set(x, o); o += x.length; }
    const b = f32(n); device.queue.writeBuffer(b, 0, all); return b; }

  // ---- pipelines
  const pipes = new Map();
  const pipe = (key, code) => { if (!pipes.has(key)) pipes.set(key, device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })); return pipes.get(key); };
  const uni = (words) => { const b = device.createBuffer({ size: Math.max(16, words.length * 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); bufs.push(b);
    const a = new ArrayBuffer(words.length * 4), v = new DataView(a); words.forEach((w, i) => typeof w === "object" ? v.setFloat32(i * 4, w.f, true) : v.setUint32(i * 4, w >>> 0, true));
    device.queue.writeBuffer(b, 0, a); return b; };
  const fl = (x) => ({ f: x });   // a float uniform word (plain numbers are written as u32)
  const ops = [];
  const op = (p, entries, x, y = 1) => ops.push({ p, bg: device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) }), x, y });
  const gemv = (w, X, Y, { xoff = 0, yoff = 0, acc = 0 } = {}) => {
    for (const part of w.parts) {
      const groups = Math.ceil(part.rows / 8), gx = Math.min(groups, 32768), gy = Math.ceil(groups / gx);
      op(pipe("g" + w.type, K.gemv(w.type)), [part.buf, X, Y, uni([part.rows, w.K, xoff, yoff, acc, gx, part.r0, 0])], gx, gy);
    }
  };
  const norm = (X, Wn, Y, { dim, heads = 1, xoff = 0, xs = 0, yoff = 0, ys = 0, mode = 0 }) => op(pipe("norm", K.rmsnorm), [X, Wn, Y, uni([dim, xoff, xs, yoff, ys, mode, fl(s.eps), 0])], heads);

  // ---- activations and state
  const C = 2 * s.hk * s.S + s.inner;
  const x = f32(s.d), xn = f32(s.d), hn = f32(s.d), qkv = f32(C), z = f32(s.inner), ab = f32(2 * s.hv), cv = f32(C), lo = f32(s.inner);
  const qg = f32(s.nh * 512), kvb = f32(2 * s.nkv * 256), qa = f32(s.nh * 256), ao = f32(s.nh * 256), sc = f32(s.nh * ctx);
  const gu = f32(2 * s.ff), sw = f32(s.ff), logits = f32(s.vocab), pm = f32(1048576), pi = f32(1048576);
  const ids = mk(16), step = mk(16);
  const states = [];

  // embedding (gather) and head
  const embW = await weight("token_embd.weight");
  for (const part of embW.parts) op(pipe("e" + embW.type, K.embed(embW.type)), [part.buf, ids, x, uni([s.d, part.r0, part.rows, 0])], Math.ceil(s.d / 256));
  for (let i = 0; i < s.n_layer; i++) {
    const P = (f) => `blk.${i}.${f}`;
    norm(x, await packed([await f32s(P("attn_norm.weight"))]), xn, { dim: s.d });
    if (s.rec[i]) {
      gemv(await weight(P("attn_qkv.weight")), xn, qkv);
      gemv(await weight(P("attn_gate.weight")), xn, z);
      gemv(await weight(P("ssm_alpha.weight")), xn, ab);
      gemv(await weight(P("ssm_beta.weight")), xn, ab, { yoff: s.hv });
      const convw = await packed([await f32s(P("ssm_conv1d.weight"))]), cst = f32(C * 3), st = f32(s.hv * s.S * s.S);
      states.push(cst, st);
      op(pipe("conv", K.conv), [qkv, convw, cst, cv, uni([C, 0, 0, 0])], Math.ceil(C / 256));
      const pr = await packed([await f32s(P("ssm_dt.bias")), await f32s(P("ssm_a")), await f32s(P("ssm_norm.weight"))]);
      op(pipe("delta", K.delta), [cv, ab, pr, st, z, lo, uni([s.hk, s.hv, fl(s.eps), 0])], s.hv);
      gemv(await weight(P("ssm_out.weight")), lo, x, { acc: 1 });
    } else {
      gemv(await weight(P("attn_q.weight")), xn, qg);
      gemv(await weight(P("attn_k.weight")), xn, kvb);
      gemv(await weight(P("attn_v.weight")), xn, kvb, { yoff: s.nkv * 256 });
      const nw = await packed([await f32s(P("attn_q_norm.weight")), await f32s(P("attn_k_norm.weight"))]);
      const kc = f32(ctx * s.nkv * 256), vc = f32(ctx * s.nkv * 256);
      op(pipe("prep", K.attnprep), [qg, kvb, nw, qa, kc, vc, uni([s.nh, s.nkv, s.rope, fl(s.eps), fl(s.base), 0, 0, 0]), step], s.nh + s.nkv);
      op(pipe("attn", K.attn), [qa, kc, vc, sc, qg, ao, uni([s.nh, s.nkv, ctx, 0]), step], s.nh);
      gemv(await weight(P("attn_output.weight")), ao, x, { acc: 1 });
    }
    norm(x, await packed([await f32s(P("post_attention_norm.weight"))]), hn, { dim: s.d });
    gemv(await weight(P("ffn_gate.weight")), hn, gu);
    gemv(await weight(P("ffn_up.weight")), hn, gu, { yoff: s.ff });
    op(pipe("swiglu", K.swiglu), [gu, sw, uni([s.ff, 0, 0, 0])], Math.ceil(s.ff / 256));
    gemv(await weight(P("ffn_down.weight")), sw, x, { acc: 1 });
  }
  norm(x, await packed([await f32s("output_norm.weight")]), hn, { dim: s.d });
  const nBody = ops.length;   // CLEF-HEAD: ops past this one are the vocab head + argmax
  const headW = by.has("output.weight") ? await weight("output.weight") : embW;
  gemv(headW, hn, logits);
  const nb = Math.ceil(s.vocab / 4096);
  op(pipe("amax", K.argmax), [logits, pm, pi, uni([s.vocab, 0, 0, 0])], nb);
  op(pipe("amax", K.argmax), [logits, pm, pi, uni([nb, 1, 0, 0])], 1);
  await device.queue.onSubmittedWorkDone();
  if (lost) throw new Error("WebGPU device lost: " + lost.message);

  const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const lrb = device.createBuffer({ size: s.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const hrb = device.createBuffer({ size: s.d * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let pos = 0;
  async function stepTok(token, { wantLogits = false, wantHidden = false, noHead = false } = {}) {
    if (pos >= ctx) throw new Error(`The context is full (${ctx} tokens).`);
    const t0 = performance.now();
    device.queue.writeBuffer(ids, 0, new Uint32Array([token, 0, 0, 0]));
    device.queue.writeBuffer(step, 0, new Uint32Array([pos, 0, 0, 0]));
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    const head = !(noHead && !wantLogits);
    for (const o of head ? ops : ops.slice(0, nBody)) { pass.setPipeline(o.p); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(o.x, o.y); }
    pass.end();
    if (head) { enc.copyBufferToBuffer(pm, 1048575 * 4, rb, 0, 4); enc.copyBufferToBuffer(pi, 1048575 * 4, rb, 4, 4); }
    if (wantLogits) enc.copyBufferToBuffer(logits, 0, lrb, 0, s.vocab * 4);
    if (wantHidden) enc.copyBufferToBuffer(hn, 0, hrb, 0, s.d * 4);
    device.queue.submit([enc.finish()]);
    let v = null, L = null, Hd = null;
    if (head) { await rb.mapAsync(GPUMapMode.READ); v = new DataView(rb.getMappedRange().slice(0)); rb.unmap(); }
    if (wantHidden) { await hrb.mapAsync(GPUMapMode.READ); Hd = new Float32Array(hrb.getMappedRange().slice(0)); hrb.unmap(); }
    if (wantLogits) { await lrb.mapAsync(GPUMapMode.READ); L = new Float32Array(lrb.getMappedRange().slice(0)); lrb.unmap(); }
    if (lost) throw new Error("WebGPU device lost: " + lost.message);
    pos++;
    return { id: v ? v.getUint32(4, true) : -1, logit: v ? v.getFloat32(0, true) : NaN, ms: performance.now() - t0, logits: L, hidden: Hd };
  }
  async function reset() {
    pos = 0;
    const enc = device.createCommandEncoder();
    for (const b of states) enc.clearBuffer(b);
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
  }
  await reset();
  return { shape: s, check: chk, ctx, dispatches: ops.length, resident_bytes: resident, limits: { binding: LIM, adapter_binding: L.maxStorageBufferBindingSize, adapter_buffer: L.maxBufferSize },
    adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null,
    step: stepTok, reset, get pos() { return pos; }, free() { for (const b of bufs) b.destroy(); device.destroy(); } };
}
