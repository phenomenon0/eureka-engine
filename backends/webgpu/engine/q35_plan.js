// The fixed Qwen3.5 4B geometry, with every weight format/location decoded
// independently from the web pack's wtab. No model-wide quantization guess.
const FORMATS = { 0: "f32", 1: "q8", 3: "q4k", 4: "q6k", 9: "q5k" };
const RB = { q8: c => c * 34 / 32, q4k: c => c * 144 / 256,
  q6k: c => c * 210 / 256, q5k: c => c * 176 / 256 };
const COMMON = { an: [1, 2560], post: [2, 2560], fg: [3, 9216, 2560],
  fu: [4, 9216, 2560], fd: [5, 2560, 9216] };
const LINEAR = { qkv: [6, 8192, 2560], gate: [7, 4096, 2560], sb: [8, 32, 2560],
  sa: [9, 32, 2560], ssa: [10, 32], dt: [11, 32], cv: [12, 32768],
  sn: [13, 128], so: [14, 2560, 4096] };
const FULL = { aq: [15, 8192, 2560], ak: [16, 1024, 2560], av: [17, 1024, 2560],
  aqn: [18, 256], akn: [19, 256], ao: [20, 2560, 4096] };
import { FAST_FORMATS } from "./q35_split.js";
const check = (yes, why) => { if (!yes) throw new Error("q35_plan: " + why); };
export const Q35_HEAD_ROWS = 62080;
export const Q35_CAPACITIES = [16, 32, 128, 256, 4096];

export function q35_capacity(promptTokens, generationSteps) {
  check(Number.isSafeInteger(promptTokens) && promptTokens > 0 &&
    Number.isSafeInteger(generationSteps) && generationSteps > 0, "invalid sequence length");
  const need = promptTokens + generationSteps - 1;
  check(need <= 131072, "native context envelope exceeded");
  const ctx = 2 ** Math.ceil(Math.log2(need));
  check(Q35_CAPACITIES.includes(ctx), `unsupported native capacity ${ctx}; emit and gate its shape first`);
  return ctx;
}

export function q35_tensors(manifest, table) {
  const c = manifest.config;
  for (const [k, v] of Object.entries({ n_layers: 32, n_heads: 16, n_kv_heads: 4,
    d_model: 2560, head_dim: 256, d_ff: 9216, vocab_size: 248320,
    full_attention_interval: 4, ssm_state_size: 128, ssm_inner_size: 4096,
    ssm_conv_kernel: 4, rope_dim: 64, rope_freq_base: 10000000 }))
    check(c?.[k] === v, `unsupported ${k}: ${c?.[k]}`);
  check(manifest.format === "llama-bend-pack-q35-quant-web/v0", "expected quant web pack");
  check(manifest.web?.block_bytes===33554432,'unsupported web block size');
  // BENDWEB-SHOWCASE: the tied embedding/head may also be Q4_K (Kev-4B): row 1440 bytes, gathered by k_embed_q4k.wgsl, head GEMV q4k2560
  const HEAD = { Q6_K: { rowBytes: 2100, embed: ["k_embed.wgsl", "embed"], gemv: "q6k2560" }, Q4_K: { rowBytes: 1440, embed: ["k_embed_q4k.wgsl", "embed4"], gemv: "q4k2560" } };
  const head = HEAD[manifest.tensors?.["token_embd.weight"]?.src_type];
  check(head && manifest.tensors?.["token_embd.weight"]?.row_bytes === head.rowBytes &&
    manifest.tensors?.["token_embd.weight"]?.offset === 0 &&
    manifest.tensors?.["token_embd.weight"]?.bytes === 248320 * head.rowBytes &&
    manifest.tensors?.["output.weight"]?.src_type === manifest.tensors["token_embd.weight"].src_type &&
    manifest.tensors?.["output.weight"]?.alias_of === "token_embd.weight" && c.tied_embeddings === true, "unsupported tied head");
  const bytes = table instanceof ArrayBuffer ? new Uint8Array(table) : table;
  check(ArrayBuffer.isView(bytes) && bytes.byteLength === 4096, "wtab must be 4096 bytes");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const files = new Map(manifest.web.files.map(f => [f.path, f]));
  const normOffset = manifest.tensors["output_norm.weight"]?.offset;
  check(manifest.tensors["output_norm.weight"]?.src_type === 'F32' && Number.isSafeInteger(normOffset) && normOffset >= 0 &&
    normOffset % 4 === 0 && normOffset + 10240 <= files.get('norm.bin')?.bytes, 'invalid output norm');
  check(files.get('head.bin')?.bytes === 248320 * head.rowBytes, 'invalid head file length');
  const layers = [];
  for (let layer = 0; layer < 32; layer++) {
    const full = (layer + 1) % 4 === 0, tensors = {};
    for (const [name, [field, rows, cols]] of Object.entries({ ...COMMON, ...(full ? FULL : LINEAR) })) {
      const word = view.getUint32(layer * 128 + field * 4, true);
      const format = FORMATS[word >>> 28], block = (word >>> 21) & 127, offset = (word & 2097151) * 16;
      check(format && (cols ? format !== "f32" : format === "f32"), `bad format L${layer}.${name}`);
      const rowBytes = cols ? RB[format](cols) : 4, size = rows * rowBytes;
      const path = `blocks/b${block}.bin`, file = files.get(path);
      check(file && offset + size <= file.bytes, `tensor crosses block: L${layer}.${name}`);
      tensors[name] = { name: `L${layer}.${name}`, layer, role: name, word, format, block,
        path, offset, rows, cols: cols ?? 1, rowBytes, bytes: size,
        buffer: cols ? `block${block}` : `L${layer}.w.${name}`,
        entry: cols ? (format === "q8" ? "q8w" : format) + cols : null };
    }
    layers.push({ layer, full, tensors });
  }
  return { layers, files: Object.fromEntries(files), normOffset, head };
}

// Graph variants differ only in embedding chunk and ping-pong bindings. All
// buffers are engine-owned externals, shared by variants on the same device.
// A step reads state[parity], writes state[1-parity], then advances position.
// PIPELINESPLIT: with `split` ({fast(name) -> bool, audit}) the plan is the speculative fast variant: Q4_K/Q5_K/Q6_K GEMVs
// whose tensor passed the load-time scale scan run on k_gemv_fast.wgsl, and a chain of k_split.wgsl links (one a layer, one
// for the head) checks every word those GEMVs read; the last link's word follows the top-2 in the result.
export function q35_plan(manifest, table, { ctx, layers = 32, embedChunk = 0, parity = 0, trace = [], split = null, turbo = false } = {}) {
  check(Q35_CAPACITIES.includes(ctx), `unsupported capacity ${ctx}`);
  check(Number.isInteger(layers) && layers >= 1 && layers <= 32, "layers outside 1..32");
  check(Number.isInteger(embedChunk) && embedChunk >= 0 && embedChunk < 4, "bad embedding chunk");
  check(parity === 0 || parity === 1, "bad ping-pong parity");
  const model = q35_tensors(manifest, table), buffers = {}, steps = [], states = [], hidden = [];
  const T = turbo ? "_t" : "";   // TURBODEFAULT: turbo plans use fused hand-written kernels (k_fuse_t, k_top_t, turbo GEMV) and the hardware-arithmetic k_norms_t for the few unfused norm ops
  const N = `k_norms${T}.wgsl`, A = "k_attention.wgsl", R = "k_recurrent.wgsl",
    V = "k_gemv.wgsl", S = "k_softmax.wgsl", X = "k_step.wgsl", VF = "k_gemv_fastE.wgsl", K = "k_split.wgsl", VT = "k_gemv_turbo.wgsl", TF = "k_fuse_t.wgsl";
  // TURBODEFAULT: turbo = every GEMV on the row-tiled inexact module (no split chain, no replay); canon fast steps run the row-tiled exact E module
  const links = [];
  let link = null;
  const buf = (name, words, options = {}) => {
    const bytes = words * 4;
    if (buffers[name]) { check(buffers[name].bytes === bytes, "conflicting buffer " + name); return name; }
    buffers[name] = { bytes, external: true, persist: true, ...options };
    return name;
  };
  const TILE = 4;
  const op = (module, kind, name, input, out, count, sc = {}, out2 = null) => {
    check(!Object.values(input).includes(out) && !(out2 && Object.values(input).includes(out2)), "aliased output " + out);
    const tile = (module === VT || module === VF) && /^(q|gu$)/.test(name) ? TILE : 1;
    steps.push({ module, kind, name, in: input, out, [kind === "cube" ? "rows" : "len"]: count, sc, ...(tile > 1 ? { tile } : {}), ...(out2 ? { out2 } : {}) });
  };
  const cp = (from, to, words, fromWord = 0, toWord = 0) => steps.push({ kind: "copy", from, to,
    bytes: words * 4, fromOffset: fromWord * 4, toOffset: toWord * 4 });
  for (const { tensors } of model.layers.slice(0, layers)) for (const t of Object.values(tensors))
    buf(t.buffer, (t.format === "f32" ? t.bytes : model.files[t.path].bytes) / 4, { weight: true });
  for (let ch = 0; ch < 4; ch++) buf(`head${ch}`, Q35_HEAD_ROWS * model.head.rowBytes / 4, { weight: true });
  buf("head.normw", 2560, { weight: true });
  buf("hidden0", 4096); hidden.push("hidden0");
  op(model.head.embed[0], "map", model.head.embed[1], { wb: `head${embedChunk}` }, "hidden0", 2560,
    { id: { param: "token" }, chunk_row0: embedChunk * Q35_HEAD_ROWS });
  for (const l of model.layers.slice(0, layers)) {
    const prefix = `L${l.layer}.`, t = l.tensors;
    const b = (s, n) => buf(prefix + s, n);
    const x = `hidden${l.layer}`, out = buf(`hidden${l.layer + 1}`, 4096); hidden.push(out);
    const h = b("h", 4096), fastIn = [];
    let proj;
    const mv = (role, input, dest) => {
      const w = t[role];
      const output = b(dest, w.rows === 2560 ? 4096 : w.rows);
      // split.exactX: x buffers that are not speculated (their GEMVs run the per-leaf exact module and the chain skips them)
      const fast = !!(split && FAST_FORMATS[w.format] && split.fast(w.name) && !split.exactX?.includes(input));
      op(turbo ? VT : fast ? VF : V, "cube", w.entry, { wb: w.buffer, xb: input }, output, w.rows, { base: w.offset });
      if (fast) fastIn.push({ x: input, cols: w.cols, name: w.name });
      return output;
    };
    if (!(turbo && l.layer > 0)) op(N, "cube", "rms2560", { x, w: t.an.buffer }, h, 1);   // turbo: layer N>0's input norm is the previous layer's fused add+norm
    let contribution;
    if (!l.full) {
      const fuseG = turbo && t.sb.entry === "q8w2560" && t.sa.entry === "q8w2560" && t.sb.buffer === t.sa.buffer;   // beta0/alpha0 GEMVs + gate math in one dispatch
      const qkv = mv("qkv", h, "qkv"), z = mv("gate", h, "z");
      const beta0 = fuseG ? null : mv("sb", h, "beta0"), alpha0 = fuseG ? null : mv("sa", h, "alpha0");
      const beta = b("beta", 32), dec = b("dec", 32);
      if (fuseG) op(TF, "cube", "gatesg", { wb: t.sb.buffer, xb: h, dt: t.dt.buffer, sa: t.ssa.buffer }, beta, 1, { bb: t.sb.offset, ba: t.sa.offset }, dec);
      else if (turbo) op(TF, "cube", "gates", { b0: beta0, a0: alpha0, dt: t.dt.buffer, sa: t.ssa.buffer }, beta, 1, {}, dec);   // sigmoid + (vadd, softplus, vmul, exp)
      else {
        const adt = b("adt", 32), sp = b("sp", 32), gp = b("gp", 32);
        op(N, "map", "sigmoid", { x: beta0 }, beta, 32);
        op(N, "map", "vadd", { a: alpha0, b: t.dt.buffer }, adt, 32);
        op(X, "map", "softplus", { x: adt }, sp, 32);
        op(N, "map", "vmul", { a: sp, b: t.ssa.buffer }, gp, 32);
        op(X, "map", "exp", { x: gp }, dec, 32);
      }
      const histories = [b("hist0", 24576), b("hist1", 24576)], ss = [b("state0", 524288), b("state1", 524288)];
      states.push(...histories, ...ss);
      const conv = b("conv", 8192);
      if (turbo) op(TF, "cube", "convh", { xb: qkv, hb: histories[parity], wb: t.cv.buffer }, conv, 32, {}, histories[1 - parity]);   // conv + history shift
      else {
        op(R, "map", "conv", { xb: qkv, hb: histories[parity], wb: t.cv.buffer }, conv, 8192);
        op(R, "map", "history", { xb: qkv, hb: histories[parity] }, histories[1 - parity], 24576);
      }
      const q = b("q", 4096), k = b("k", 4096), v = b("v", 4096);
      if (turbo) {   // q/k l2norm straight from the conv output, duplicated to 32 heads in the same dispatch; v is one copy
        op(TF, "cube", "l2qk", { c: conv }, q, 32, {}, k);
        cp(conv, v, 4096, 4096);
      } else {
        const q0 = b("q0", 2048), k0 = b("k0", 2048), q16 = b("q16", 2048), k16 = b("k16", 2048);
        cp(conv, q0, 2048); cp(conv, k0, 2048, 2048); cp(conv, v, 4096, 4096);
        op(N, "cube", "l2", { x: q0 }, q16, 16, { sc: Math.fround(Math.sqrt(128)) });
        op(N, "cube", "l2", { x: k0 }, k16, 16, { sc: 1 });
        cp(q16, q, 2048); cp(q16, q, 2048, 0, 2048);
        cp(k16, k, 2048); cp(k16, k, 2048, 0, 2048);
      }
      const delta = b("delta", 4096), readout = b("readout", 4096), normed = b("normed", 4096);
      if (turbo) op(TF, "cube", "gdn", { sb: ss[parity], kb: k, db: dec, vb: v, bb: beta, qb: q }, ss[1 - parity], 1024, {}, readout);   // delta + state + readout
      else {
        op(R, "cube", "delta", { sb: ss[parity], kb: k, db: dec, vb: v, bb: beta }, delta, 4096);
        op(R, "map", "state", { sb: ss[parity], kb: k, db: dec, eb: delta }, ss[1 - parity], 524288);
        op(R, "cube", "readout", { sb: ss[1 - parity], qb: q }, readout, 4096);
      }
      op(N, "cube", "gated", { o: readout, w: t.sn.buffer, z }, normed, 32);
      proj = normed;
      contribution = mv("so", normed, "attn");
    } else {
      const qx = mv("aq", h, "qx"), kx = mv("ak", h, "kx"), vx = mv("av", h, "vx");
      const kc = [b("kc0", ctx * 1024), b("kc1", ctx * 1024)], vc = [b("vc0", ctx * 1024), b("vc1", ctx * 1024)];
      states.push(...kc, ...vc);
      const gated = b("gated", 4096);
      if (turbo) {   // 16 dispatches -> 3: q norm+rope | k norm+rope + K/V put | scores..softmax..weighted + output gate (positions <= pos only)
        const q = b("q", 4096);
        op(TF, "cube", "qprep", { qx, w: t.aqn.buffer }, q, 16, { pos: { param: "pos" } });
        op(TF, "cube", "kvput", { cb: kc[parity], vb: vc[parity], kx, vx, w: t.akn.buffer }, kc[1 - parity], ctx * 4, { pos: { param: "pos" } }, vc[1 - parity]);
        op(TF, "cube", "attn", { q, cb: kc[1 - parity], vb: vc[1 - parity], qx }, gated, 16, { pos: { param: "pos" } });
      } else {
      const qn = b("qn", 4096), qg = b("qg", 8192), kn = b("kn", 1024), kr = b("kr", 1024), q = b("q", 4096), gate = b("gate", 4096);
      op(A, "cube", "qnorm", { xb: qx, w: t.aqn.buffer }, qn, 16);
      op(A, "map", "qgrope", { qb: qn, xb: qx }, qg, 8192, { pos: { param: "pos" } });
      op(A, "cube", "knorm", { x: kx, w: t.akn.buffer }, kn, 4);
      op(A, "map", "rope", { xb: kn }, kr, 1024, { pos: { param: "pos" } });
      op(A, "map", "query", { qgb: qg }, q, 4096);
      op(A, "map", "gate", { qgb: qg }, gate, 4096);
      op(A, "map", "put", { cb: kc[parity], xb: kr }, kc[1 - parity], ctx * 1024, { pos: { param: "pos" } });
      op(A, "map", "put", { cb: vc[parity], xb: vx }, vc[1 - parity], ctx * 1024, { pos: { param: "pos" } });
      const scores = b("scores", 16 * ctx), max = b("max", 16), masked = b("masked", 16 * ctx), exp = b("exp", 16 * ctx), sum = b("sum", 16), prob = b("prob", 16 * ctx), attn = b("weighted", 4096);
      op(A, "cube", "scores", { qb: q, cb: kc[1 - parity] }, scores, 16 * ctx, { ctx });
      op(A, "cube", "max" + ctx, { s: scores }, max, 16, { pos: { param: "pos" } });
      op(A, "map", "mask" + ctx, { s: scores }, masked, 16 * ctx, { pos: { param: "pos" } });
      op(A, "map", "exp" + ctx, { s: masked, m: max }, exp, 16 * ctx);
      op(A, "cube", "sum" + ctx, { e: exp }, sum, 16);
      op(A, "map", "prob" + ctx, { e: exp, z: sum }, prob, 16 * ctx);
      op(A, "cube", "weighted" + ctx, { pb: prob, cb: vc[1 - parity] }, attn, 4096);
      op(N, "map", "ogate", { x: attn, g: gate }, gated, 4096);
      }
      proj = gated;
      contribution = mv("ao", gated, "attn");
    }
    const residual = b("residual", 4096), hn = b("ffn.h", 4096);
    if (turbo) op(TF, "cube", "addrms", { a: x, b: contribution, w: t.post.buffer }, residual, 1, {}, hn);   // residual add + post norm
    else {
      op(N, "map", "vadd", { a: x, b: contribution }, residual, 2560);
      op(N, "cube", "rms2560", { x: residual, w: t.post.buffer }, hn, 1);
    }
    const swiglu = b("ffn.swiglu", 9216);
    if (turbo && t.fg.entry === "q4k2560" && t.fu.entry === "q4k2560")   // gate + up + swiglu in one dispatch (both Q4_K, same x)
      op(VT, "cube", "gu", { wg: t.fg.buffer, wu: t.fu.buffer, xb: hn }, swiglu, 9216, { bg: t.fg.offset, bu: t.fu.offset });
    else {
      const fg = mv("fg", hn, "ffn.g"), fu = mv("fu", hn, "ffn.u");
      op(N, "map", "swiglu", { g: fg, u: fu }, swiglu, 9216);
    }
    const dn = mv("fd", swiglu, "ffn.down");
    if (turbo) {   // residual add + the NEXT layer's input norm (the head norm after the last layer)
      const nextW = l.layer + 1 < layers ? model.layers[l.layer + 1].tensors.an.buffer : "head.normw";
      op(TF, "cube", "addrms", { a: residual, b: dn, w: nextW }, out, 1, {}, buf(l.layer + 1 < layers ? `L${l.layer + 1}.h` : "head.h", 4096));
    } else op(N, "map", "vadd", { a: residual, b: dn }, out, 2560);
    if (split && split.links !== false) {
      // a slot whose buffer no fast GEMV reads (an exactX buffer) re-checks a buffer with at least as many real words (the ffn
      // input for a, the down input's first 4096 for b): over-checking can only cause a replay, never a missed word
      const used = (x) => fastIn.some((f) => f.x === x);
      const slots = { ab: [used(h) ? h : hn, 2560], bb: [used(proj) ? proj : swiglu, 4096], cb: [hn, 2560], db: [swiglu, 9216] };
      for (const f of fastIn) check(Object.values(slots).some(([x, n]) => x === f.x && n === f.cols), "split: fast GEMV input not checked " + f.name);
      link ??= buf("xchk.zero", 1, { weight: true });   // never written: 0, the max identity
      const next = b("xchk", 1);
      op(K, "cube", "xchk4", { f: link, ...Object.fromEntries(Object.entries(slots).map(([k, [x]]) => [k, x])) }, next, 1);
      links.push(next); link = next;
    }
  }
  const hn = buf("head.h", 4096), logits = buf("head.logits", 248320);
  if (!turbo) op(N, "cube", "rms2560", { x: `hidden${layers}`, w: "head.normw" }, hn, 1);
  for (let ch = 0; ch < 4; ch++) {
    const out = buf(`head.chunk${ch}`, Q35_HEAD_ROWS);
    op(turbo ? VT : split?.fast(`head${ch}`) ? VF : V, "cube", model.head.gemv, { wb: `head${ch}`, xb: hn }, out, Q35_HEAD_ROWS, { base: 0 });
    cp(out, logits, Q35_HEAD_ROWS, 0, ch * Q35_HEAD_ROWS);
  }
  if (split && split.links !== false) {   // the head's x (2560 words, all four chunks read the same vector)
    link ??= buf("xchk.zero", 1, { weight: true });
    const next = buf("head.xchk", 1);
    op(K, "cube", "xchk1", { f: link, ab: hn }, next, 1);
    links.push(next); link = next;
  }
  const argmax = (z, label) => {
    const nan = buf(label + ".nan", 1), max = buf(label + ".max", 1), idx = buf(label + ".idx", 1), id = buf(label + ".id", 1);
    op(S, "cube", "anan", { z }, nan, 1);
    op(S, "cube", "amx", { z }, max, 1);
    op(S, "cube", "aix", { z, m: max }, idx, 1);
    op(S, "map", "asel", { a: nan, c: idx }, id, 1);
    return id;
  };
  let top;
  if (turbo) {   // TURBODEFAULT: one partial-reduce + one merge dispatch for the top-2 (replaces ten serial dispatches); same `top` words
    const part = buf("top.part", 243 * 8);
    op("k_top_t.wgsl", "cube", "tp1", { z: logits }, part, 243);
    top = buf("top", 4);
    op("k_top_t.wgsl", "cube", "tp2", { p: part }, top, 1);
  } else {
    const id1 = argmax(logits, "top1"), masked = buf("head.without", 248320);
    op(X, "map", "without", { z: logits, ib: id1 }, masked, 248320);
    const id2 = argmax(masked, "top2");
    top = buf("top", 4);
    op(X, "map", "top2", { zb: logits, ab: id1, bb: id2 }, top, 4);
  }
  // turbo fuses ops away: intermediate stage buffers it never materializes cannot be traced (dropped, not an error; canon traces them all)
  const traceNames = trace === true ? hidden : turbo ? trace.filter((n) => buffers[n]) : trace;
  check(Array.isArray(traceNames), "trace must be true or buffer names");
  const result = [], seen = new Set(); let resultWords = 0;
  for (const name of [top, ...(link ? [link, ...(split.audit ? links : [])] : []), ...traceNames]) {
    check(buffers[name], "unknown trace buffer " + name);
    if (seen.has(name)) continue; seen.add(name);
    result.push({ name, offset: resultWords, words: buffers[name].bytes / 4 });
    resultWords += buffers[name].bytes / 4;
  }
  const resultBuf = split ? "result.split" : "result";   // the split variant's readback is longer; both plans share one buffer set
  buf(resultBuf, resultWords);
  for (const r of result) cp(r.name, resultBuf, r.words, 0, r.offset);
  const flag = link ? { final: result.find((r) => r.name === link).offset,
    links: split.audit ? links.map((n) => result.find((r) => r.name === n).offset) : null } : null;
  return { schema: "q35-step-plan/1", ctx, layers, embedChunk, parity, model, buffers, steps, states, hidden, result, flag,
    spec: { lane: "bend-graph/1", buffers: Object.fromEntries(Object.entries(buffers).map(([n, b]) => [n,
      { bytes: b.bytes, external: true, persist: true }])), steps, result: { buffer: resultBuf, bytes: resultWords * 4 } } };
}
