// q35_step.js: the engine step graph, as data. Ours, not canon: it names the engine's entries,
// strides and buffers, and nothing in the compiler series imports it. The same plan is lowered two
// ways, and the gate holds the two to the same bits:
//   run_direct: one entry_run per op, outputs kept as GPUBuffers (the path every kernel gate used);
//   lower:      graph_new(): recorded once, one command encoder and one submit per run.
// A plan step is {op:{mod,kind,name,in,out,rows|len,sc}} or {copy:{src,dst,bytes}}. `sc` values are
// numbers or ctx key names. `result` lists the buffers that are read back (gates only; they are
// assembled in order into one buffer with ordered copies).
import { entry_run } from "./webgpu.js";
export { q35_open, q35_source } from "./q35_engine.js";
export { q35_plan, q35_capacity, q35_tensors } from "./q35_plan.js";

const op = (mod, kind, name, ins, out, shape, sc = {}) => ({ op: { mod, kind, name, in: ins, out, ...shape, sc } });
const copy = (src, dst, words) => ({ copy: { src, dst, bytes: words * 4 } });

// One full-attention layer core for capacity n (a power of two in {16, 32, 256, 4096}):
// qnorm, qgrope, knorm, rope, query, gate, immutable put into K and V, scores, the staged softmax,
// weighted sum, output gate. kc / vc persist across tokens; put's new buffer is copied back.
export function attn_plan(n) {
  const kv = n * 1024, A = "k_attention.wgsl", N = "k_norms.wgsl";
  return {
    name: "attn" + n,
    bufs: {
      qx: { words: 8192 }, qw: { words: 256 }, kx: { words: 1024 }, kw: { words: 256 }, vslot: { words: 1024 },
      kc: { words: kv, persist: true }, vc: { words: kv, persist: true }, kcn: { words: kv }, vcn: { words: kv },
      qn: { words: 4096 }, qg: { words: 8192 }, kn: { words: 1024 }, kr: { words: 1024 }, qi: { words: 4096 },
      gi: { words: 4096 }, s: { words: 16 * n }, m: { words: 16 }, ms: { words: 16 * n }, e: { words: 16 * n },
      z: { words: 16 }, p: { words: 16 * n }, o: { words: 4096 }, gated: { words: 4096 },
    },
    writes: ["qx", "qw", "kx", "kw", "vslot"],
    persist: ["kc", "vc"],
    steps: [
      op(A, "cube", "qnorm", { xb: "qx", w: "qw" }, "qn", { rows: 16 }),
      op(A, "map", "qgrope", { qb: "qn", xb: "qx" }, "qg", { len: 8192 }, { pos: "pos" }),
      op(A, "cube", "knorm", { x: "kx", w: "kw" }, "kn", { rows: 4 }),
      op(A, "map", "rope", { xb: "kn" }, "kr", { len: 1024 }, { pos: "pos" }),
      op(A, "map", "query", { qgb: "qg" }, "qi", { len: 4096 }),
      op(A, "map", "gate", { qgb: "qg" }, "gi", { len: 4096 }),
      op(A, "map", "put", { cb: "kc", xb: "kr" }, "kcn", { len: kv }, { pos: "pos" }),
      copy("kcn", "kc", kv),
      op(A, "map", "put", { cb: "vc", xb: "vslot" }, "vcn", { len: kv }, { pos: "pos" }),
      copy("vcn", "vc", kv),
      op(A, "cube", "scores", { qb: "qi", cb: "kc" }, "s", { rows: 16 * n }, { ctx: n }),
      op(A, "cube", "max" + n, { s: "s" }, "m", { rows: 16 }, { pos: "pos" }),
      op(A, "map", "mask" + n, { s: "s" }, "ms", { len: 16 * n }, { pos: "pos" }),
      op(A, "map", "exp" + n, { s: "ms", m: "m" }, "e", { len: 16 * n }),
      op(A, "cube", "sum" + n, { e: "e" }, "z", { rows: 16 }),
      op(A, "map", "prob" + n, { e: "e", z: "z" }, "p", { len: 16 * n }),
      op(A, "cube", "weighted" + n, { pb: "p", cb: "vc" }, "o", { rows: 4096 }),
      op(N, "map", "ogate", { x: "o", g: "gi" }, "gated", { len: 4096 }),
    ],
    result: ["qn", "qg", "kn", "kr", "qi", "gi", "kcn", "vcn", "s", "m", "ms", "e", "z", "p", "o", "gated"],
  };
}

// One linear-attention (gated delta) layer core: conv1d, history rotation, delta, state update,
// readout. History and the 32x128x128 state persist; the update outputs are copied back.
export function rec_plan(heads = 32) {
  const R = "k_recurrent.wgsl", S = heads * 16384;
  return {
    name: "rec" + heads,
    bufs: {
      cx: { words: 8192 }, cw: { words: 32768 }, q: { words: heads * 128 }, k: { words: heads * 128 }, v: { words: heads * 128 },
      dec: { words: heads }, bet: { words: heads },
      hist: { words: 24576, persist: true }, state: { words: S, persist: true },
      convo: { words: 8192 }, hist2: { words: 24576 }, delta: { words: heads * 128 }, state2: { words: S },
      readout: { words: heads * 128 },
    },
    writes: ["cx", "cw", "q", "k", "v", "dec", "bet"],
    persist: ["hist", "state"],
    steps: [
      op(R, "map", "conv", { xb: "cx", hb: "hist", wb: "cw" }, "convo", { len: 8192 }),
      op(R, "map", "history", { xb: "cx", hb: "hist" }, "hist2", { len: 24576 }),
      copy("hist2", "hist", 24576),
      op(R, "cube", "delta", { sb: "state", kb: "k", db: "dec", vb: "v", bb: "bet" }, "delta", { rows: heads * 128 }),
      op(R, "map", "state", { sb: "state", kb: "k", db: "dec", eb: "delta" }, "state2", { len: S }),
      copy("state2", "state", S),
      op(R, "cube", "readout", { sb: "state", qb: "q" }, "readout", { rows: heads * 128 }),
    ],
    result: ["convo", "hist2", "delta", "state2", "readout"],
  };
}

// A combined step: several plans become one graph with prefixed buffer names. Their steps are
// concatenated in order, so the whole thing is one submit.
export function combine(name, ...plans) {
  const out = { name, bufs: {}, writes: [], persist: [], steps: [], result: [], parts: [] };
  plans.forEach((p, i) => {
    const pre = p.name + ".", r = (b) => pre + b;
    for (const [k, v] of Object.entries(p.bufs)) out.bufs[r(k)] = v;
    out.writes.push(...p.writes.map(r));
    out.persist.push(...p.persist.map(r));
    for (const s of p.steps) {
      if (s.copy) out.steps.push({ copy: { ...s.copy, src: r(s.copy.src), dst: r(s.copy.dst) } });
      else out.steps.push({ op: { ...s.op, in: Object.fromEntries(Object.entries(s.op.in).map(([a, b]) => [a, r(b)])), out: r(s.op.out) } });
    }
    out.result.push(...p.result.map(r));
    out.parts.push(p.name);
  });
  return out;
}

const wordsOf = (plan, b) => plan.bufs[b].words;
export function result_layout(plan) {
  let at = 0;
  return plan.result.map((b) => { const o = { buf: b, off: at, words: wordsOf(plan, b) }; at += o.words; return o; });
}
export const result_words = (plan) => result_layout(plan).reduce((n, r) => n + r.words, 0);

// One FFN block at the 4B's real shapes: rms2560 over the residual, Q4_K gate/up (9216 x 2560), swiglu,
// Q4_K down (2560 x 9216), residual add. Weights are external GPU buffers (caller-owned chunks).
export function ffn_plan() {
  const N = "k_norms.wgsl", V = "k_gemv.wgsl", GV = 9216 * 1440 / 4;
  return {
    name: "ffn",
    bufs: {
      x: { words: 2560 }, w: { words: 2560 }, wg: { words: GV, external: true }, wu: { words: GV, external: true },
      wd: { words: GV, external: true },
      xn: { words: 2560 }, g: { words: 9216 }, u: { words: 9216 }, h: { words: 9216 }, dn: { words: 2560 }, y: { words: 2560 },
    },
    writes: ["x", "w"],
    persist: [],
    steps: [
      op(N, "cube", "rms2560", { x: "x", w: "w" }, "xn", { rows: 1 }),
      op(V, "cube", "q4k2560", { wb: "wg", xb: "xn" }, "g", { rows: 9216 }, { base: 0 }),
      op(V, "cube", "q4k2560", { wb: "wu", xb: "xn" }, "u", { rows: 9216 }, { base: 0 }),
      op(N, "map", "swiglu", { g: "g", u: "u" }, "h", { len: 9216 }),
      op(V, "cube", "q4k9216", { wb: "wd", xb: "h" }, "dn", { rows: 2560 }, { base: 0 }),
      op(N, "map", "vadd", { a: "x", b: "dn" }, "y", { len: 2560 }),
    ],
    result: ["xn", "g", "u", "h", "dn", "y"],
  };
}

// Record the plan as a graph. `mods` maps module names to pipes() results; the graph keeps the
// module name as its serialization key. The result buffer is the assembled readback.
export function lower(graph_new, g, plan, mods, externals = {}, options = {}) {
  const G = graph_new(g, { maxReadBytes: result_words(plan) * 4, ...options });
  for (const [b, v] of Object.entries(plan.bufs))
    G.buf(b, { words: v.words, persist: !!plan.persist.includes(b), ...(v.external ? { buffer: externals[b] } : {}) });
  G.buf("step_result", { words: result_words(plan) });
  for (const s of plan.steps) {
    if (s.copy) { G.copy(s.copy.src, s.copy.dst, { bytes: s.copy.bytes }); continue; }
    const o = s.op, pe = mods[o.mod].by(o.kind, o.name);
    if (!pe) throw new Error(`lower: no entry ${o.kind} ${o.name} in ${o.mod}`);
    const key = (name) => (ctx) => { if (ctx[name] === undefined) throw new Error("ctx lacks " + name); return ctx[name]; };
    const sc = Object.fromEntries(Object.entries(o.sc).map(([k, v]) => [k, typeof v === "string" ? key(v) : v]));
    G.op(pe, { module: o.mod, in: o.in, out: o.out, rows: o.rows, len: o.len, sc });
  }
  for (const r of result_layout(plan)) G.copy(r.buf, "step_result", { bytes: r.words * 4, dst_off: r.off * 4 });
  return G;
}
export const graph_read = (plan) => ["step_result", result_words(plan)];
export function split(plan, bits) {
  return Object.fromEntries(result_layout(plan).map((r) => [r.buf, bits.subarray(r.off, r.off + r.words)]));
}

// Direct execution: one entry_run per op, GPU-resident chaining. State: Map buffer-name -> GPUBuffer,
// persisted by the caller across tokens. `fresh` supplies the typed arrays written this token.
export async function run_direct(g, plan, mods, state, ctx, fresh) {
  const d = g.device, ST = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const bufs = state, outs = {}, owned = [];
  const put = (name, data) => {
    const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: ST | CD | CS });
    d.queue.writeBuffer(b, 0, data);
    return b;
  };
  for (const [name, data] of Object.entries(fresh)) {
    const old = bufs.get(name);
    if (old) old.destroy();
    bufs.set(name, put(name, data));
  }
  for (const s of plan.steps) {
    if (s.copy) {
      // full-buffer copies are pointer moves in the direct path; ranged copies are real copies
      const c = s.copy;
      if (c.bytes !== plan.bufs[c.src].words * 4 || c.bytes !== plan.bufs[c.dst].words * 4) throw new Error("run_direct: ranged copy");
      const dst = bufs.get(c.dst);
      if (dst && plan.persist.includes(c.dst)) dst.destroy();
      bufs.set(c.dst, bufs.get(c.src));
      bufs.delete(c.src);
      continue;
    }
    const o = s.op, pe = mods[o.mod].by(o.kind, o.name);
    const io = { in: {}, sc: {}, rows: o.rows, len: o.len, keep: true };
    for (const [k, b] of Object.entries(o.in)) {
      io.in[k] = bufs.get(b);
      if (!io.in[k]) throw new Error("run_direct: no buffer " + b);
    }
    for (const [k, v] of Object.entries(o.sc)) io.sc[k] = typeof v === "string" ? ctx[v] : v;
    const x = await entry_run(g, pe, io);
    outs[o.out] = x.bits;
    const prior = bufs.get(o.out);
    if (prior && !plan.persist.includes(o.out)) prior.destroy();
    bufs.set(o.out, x.buffer);
  }
  return outs;
}
export function release(state) { for (const b of state.values()) b.destroy(); state.clear(); }
