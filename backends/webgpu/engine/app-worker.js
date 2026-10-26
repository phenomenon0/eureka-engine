// app-worker.js (WEBGPU-E2E): the app's in-tab engine worker. It runs the shipped bend-worker.js unchanged (the 4B
// q35 engine, the 81 tok/s path, and Kev's decision head) and adds one door: a .gguf the 4B graph refuses for its shape
// (the 0.8B pocket, the 9B clef/Qwen3.5 trunk, the 27B stretch, the Q8_0-headed 4Bs) opens on WEB-TRUNK-PORT's
// shape-generic trunk engine (../trunk/trunk_engine.js), behind the same session interface, so the chat loop, the
// tokenizer and the receipts are bend-worker's own. Top-level `let`s of classic worker scripts share one scope:
// `session`, `cached`, `ctx`, `gen`, `post`, `free`, `tokenizer` below are bend-worker.js's.
importScripts("bend-worker.js");
const shippedOpen = open;
let trunk = null;
open = async function (m) {
  try { return await shippedOpen(m); }
  catch (e) {
    // only a shape refusal of a chat .gguf falls through; a decision model, a pack, a wrong tokenizer keep their reasons
    if (m.role === "decision" || m.source?.kind !== "gguf" || e?.code !== "unsupported" || /tokenizer/i.test(String(e?.message))) throw e;
    return trunkOpen(m, e);
  }
};
async function trunkOpen(m, shipped) {
  const t0 = performance.now(), mine = ++gen;
  await free(true);
  ctx = m.ctx ?? 4096;
  trunk ??= await import(here + "../trunk/trunk_engine.js");
  const gg = await import(here + "gguf.js"), gp = await import(here + "ggufpack.js");
  const src = m.source, read = src.file ? fileReader(src.file) : urlReader(new URL(src.url, location.href).href);
  const g = await gg.gguf_read(read), chk = trunk.trunk_check(g);
  if (!chk.ok) throw Object.assign(new Error([...(shipped.problems ?? [shipped.message]), ...chk.problems.map(p => "Trunk engine: " + p)].join(" ")), { code: "unsupported", problems: [...(shipped.problems ?? []), ...chk.problems.map(p => "Trunk engine: " + p)] });
  if ((await gp.gguf_vocab_sha256(g)) !== m.vocabSha) throw Object.assign(new Error("Its tokenizer isn't Qwen3.5's; the web build only carries Qwen3.5's tokenizer."), { code: "unsupported" });
  await tokenizer(toU8(await (await fetch(new URL(m.tokenizer, location.href))).arrayBuffer()));
  post({ t: "progress", phase: "validated", describe: `trunk engine · ${chk.shape.arch} · ${chk.shape.n_layer} layers · d${chk.shape.d}` });
  const e = await trunk.trunk_open(read, { ctx, gguf: g, onProgress: ({ done, total }) => post({ t: "progress", phase: "weights", done, total, of: "trunk" }) });
  if (mine !== gen) { e.free(); return; }
  // the q35 session's shape: step(tok, pos) -> {ms, top2:[{id}]}; the trunk keeps its own position, and reset() is async
  let resetting = null;
  session = {
    numerics: "trunk f32", facts: {},
    async step(tok, pos) {
      if (resetting) { await resetting; resetting = null; }
      if (pos !== e.pos) { await e.reset(); if (pos !== 0) throw new Error("trunk session out of step"); }
      const r = await e.step(tok); return { ms: r.ms, top2: [{ id: r.id }] };
    },
    reset() { resetting = e.reset(); },
    async free() { e.free(); },
    async setNumerics() { return { numerics: "trunk f32" }; },
  };
  cached = [];
  const w0 = performance.now(); await session.step(IM_END, 0); session.reset(); const warm_ms = performance.now() - w0;
  post({ t: "ready", label: m.label ?? "", open_ms: Math.round(performance.now() - t0), warm_ms: Math.round(warm_ms),
    facts: { numerics: "trunk f32", subgroups: false, dispatches: e.dispatches, ctx, resident_bytes: e.resident_bytes, pack_sha256: null, verified: false,
      engine: "trunk", shape: `${chk.shape.arch} · ${chk.shape.n_layer} layers · d${chk.shape.d}`, adapter: e.adapter ?? {} } });
}
