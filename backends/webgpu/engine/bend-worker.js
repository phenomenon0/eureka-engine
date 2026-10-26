// bend-worker.js (BENDWEB-SHOWCASE): the in-tab engine, a dedicated (classic) Worker. It hosts TURBODEFAULT's WebGPU engine
// (q35_step.js: q35_open / step / reset / free, Turbo numerics by default) and the app's C tokenizer (tok.wasm), and runs the
// chat loop the native shell runs: prompt token by token through the same step, then greedy (the engine's own top-1) until
// <|im_end|>. Its only state besides the GPU session: the token ids already in the session's KV/state, so a follow-up turn
// whose prompt extends them only steps the new tokens.
//
//   in:  open {source, ctx, numerics, tokenizer, vocabSha}   chat {id, system, messages, maxNew}   cancel {id}
//        setNumerics {numerics}   decide {id, ...}   free
//   out: progress {phase, done, total, bytes}   ready {facts, open_ms, label}   prompt {id, ids, reused}
//        token {id, text, ms}   done {id, reason, stats}   error {id?, message, fatal?}
//
// source kinds: {kind:"pack", base}         a hosted web pack (manifest.json, wtab/blocks/head/norm, tokenizer.bin), Range reads
//               {kind:"gguf", file | url}   a .gguf read in place as the web pack (ggufpack.js), File.slice or Range reads
//               {kind:"files", files}       a web pack folder picked from disk ({path: File})
importScripts("wire.js");
const here = new URL(".", location.href).href;
const post = (m) => postMessage(m);
const IM_END = 248046, END_OF_TEXT = 248044;

let engine = null, ggufpack = null, device = null, session = null, Tok = null;
let ctx = 4096, cached = [], current = null, tail = Promise.resolve(), gen = 0, label = "", kev = null;

const toU8 = (b) => b instanceof Uint8Array ? b : new Uint8Array(b);
const fileReader = (file) => async (offset, length) => toU8(await file.slice(offset, offset + length).arrayBuffer());
const urlReader = (url) => async (offset, length) => {
  const r = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const b = toU8(await r.arrayBuffer());
  return r.status === 206 ? b : b.slice(offset, offset + length);
};

async function tokenizer(bytes) {
  if (!Tok) Tok = await new Promise((resolve, reject) => {
    self.Module = { locateFile: (p) => here + p, onRuntimeInitialized: () => resolve(self.Module), onAbort: (e) => reject(new Error("tokenizer: " + e)) };
    importScripts(here + "tok.js");
  });
  try { Tok.FS.mkdir("/tok"); } catch {}
  Tok.FS.writeFile("/tok/tokenizer.bin", bytes);
  if (Tok._tok_init() !== 0) throw new Error("tokenizer.bin did not load");
}
function encode(text) {
  const n = Tok.lengthBytesUTF8(text) + 1, p = Tok._malloc(n); Tok.stringToUTF8(text, p, n);
  let cap = 8192, out = Tok._malloc(cap * 4), got = Tok._tok_encode(p, out, cap);
  if (got < 0) { Tok._free(out); cap = -got; out = Tok._malloc(cap * 4); got = Tok._tok_encode(p, out, cap); }
  const ids = Array.from(Tok.HEAP32.subarray(out >> 2, (out >> 2) + got)); Tok._free(p); Tok._free(out); return ids;
}
function piece(id) {
  const cap = 65536, p = Tok._malloc(cap), n = Tok._tok_piece(id, p, cap);
  const b = Tok.HEAPU8.slice(p, p + Math.min(n, cap)); Tok._free(p); return b;
}

// Qwen3.5's chat layout (wire.js renderChat), with one change: an earlier assistant turn keeps the empty think block it was
// generated after, so the next turn's prompt extends the ids already in the session (only the new turn is stepped).
function render(system, messages) {
  let s = system ? `<|im_start|>system\n${system}<|im_end|>\n` : "";
  for (const m of messages) s += m.role === "assistant" ? `<|im_start|>assistant\n<think>\n\n</think>\n\n${m.content}<|im_end|>\n` : `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

async function open(m) {
  const t0 = performance.now(), mine = ++gen;
  await free(true);
  kev = null;
  ctx = m.ctx ?? 4096; label = m.label ?? "";
  engine ??= await import(here + "q35_step.js");
  if (!self.navigator?.gpu) throw Object.assign(new Error("This browser has no WebGPU in workers."), { code: "no-gpu" });
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw Object.assign(new Error("No WebGPU adapter."), { code: "no-gpu" });
  const want = { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 }, req = {};
  for (const [k, v] of Object.entries(want)) req[k] = Math.min(v, adapter.limits[k]);
  const sg = adapter.features.has("subgroups") && m.subgroups !== false;
  device = await adapter.requestDevice({ requiredLimits: req, requiredFeatures: sg ? ["subgroups"] : [] });
  const info = adapter.info ?? {};
  device.lost.then((l) => { if (mine === gen && session) { session = null; post({ t: "error", fatal: true, message: "The GPU device was lost (" + (l.message || l.reason) + "). Load the model again." }); } });
  const src = m.source;
  let manifest, packSrc, tokBytes, verify = true;
  post({ t: "progress", phase: "manifest", done: 0, total: 1 });
  if (src.kind === "pack") {
    const base = new URL(src.base.replace(/\/?$/, "/"), location.href).href;
    manifest = await (await fetch(base + "manifest.json")).json();
    packSrc = base;
    tokBytes = toU8(await (await fetch(base + "tokenizer.bin")).arrayBuffer());
  } else if (src.kind === "files") {
    const f = src.files;
    if (!f["manifest.json"]) throw new Error("That folder has no manifest.json: it isn't a web pack.");
    manifest = JSON.parse(await f["manifest.json"].text());
    packSrc = { read: async (path, offset = 0, length) => { if (!f[path]) throw new Error("the pack folder is missing " + path); return toU8(await f[path].slice(offset, length === undefined ? undefined : offset + length).arrayBuffer()); } };
    tokBytes = f["tokenizer.bin"] ? toU8(await f["tokenizer.bin"].arrayBuffer()) : null;
    if (!tokBytes) throw new Error("That pack folder has no tokenizer.bin.");
  } else if (src.kind === "gguf") {
    ggufpack ??= await import(here + "ggufpack.js");
    const read = src.file ? fileReader(src.file) : urlReader(new URL(src.url, location.href).href);
    const v = await ggufpack.gguf_pack(read, { decision: m.role === "decision" });
    if (!v.ok) throw Object.assign(new Error(v.problems.join(" ")), { code: "unsupported", problems: v.problems, describe: v.describe });
    const sha = await ggufpack.gguf_vocab_sha256(v.gguf);
    if (sha !== m.vocabSha) throw Object.assign(new Error("Its tokenizer isn't Qwen3.5's (vocabulary hash " + sha.slice(0, 12) + "…); the web build only carries Qwen3.5's tokenizer."), { code: "unsupported" });
    manifest = v.manifest; packSrc = v.source; verify = false;   // no chunk digests in a GGUF: its bytes are the person's file
    if (m.role === "decision") kev = await kevHead(v);
    tokBytes = toU8(await (await fetch(new URL(m.tokenizer, location.href))).arrayBuffer());
    post({ t: "progress", phase: "validated", describe: v.describe });
  } else throw new Error("unknown source " + src.kind);
  await tokenizer(tokBytes);
  const blocks = manifest.web.blocks, headChunks = Math.ceil(manifest.tensors["token_embd.weight"].bytes / manifest.web.block_bytes);
  const total = blocks + headChunks;
  const s = await engine.q35_open({ adapter, device }, packSrc, manifest, {
    ctx, numerics: m.numerics ?? "turbo", canon: kev ? false : m.canon !== false, verify, subgroups: sg, trace: kev ? ["head.h"] : m.trace ?? [],
    // Kev's Q8_0 4096-column projections: the turbo GEMV module + that one shape (tools/turbo-build.py), every other entry identical
    modules: kev ? { "k_gemv_turbo.wgsl": await (await fetch(here + `../emit/k_gemv_turbo${sg ? "_sg" : ""}_kev.wgsl`)).text() } : undefined,
    progress: (e) => {
      if (e.phase === "weights") post({ t: "progress", phase: "weights", done: e.block + 1, total, of: "blocks" });
      else if (e.phase === "head") post({ t: "progress", phase: "weights", done: blocks + e.chunk + 1, total, of: "head" });
      else if (e.phase === "compile") post({ t: "progress", phase: "compile", module: e.module });
    },
  });
  if (mine !== gen) { await s.free(); return; }
  session = s; cached = [];
  // a first step compiles the graph variant; do it now so the person's first token doesn't pay for it
  const w0 = performance.now(); await s.step(IM_END, 0); s.reset(); const warm_ms = performance.now() - w0;
  post({ t: "ready", label, open_ms: Math.round(performance.now() - t0), warm_ms: Math.round(warm_ms),
    facts: { numerics: s.facts.numerics, subgroups: s.facts.subgroups, dispatches: s.facts.dispatches, ctx, resident_bytes: s.facts.resident_bytes,
      pack_sha256: s.facts.pack_sha256, verified: verify, adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description } } });
}

function chat({ id, system, messages, maxNew = 768, prompt }) {
  if (!session) { post({ t: "error", id, message: "No model is loaded." }); return; }
  const text = prompt ?? render(system, messages);
  const ids = encode(text);
  if (ids.length + 8 > ctx) { post({ t: "error", id, message: `The conversation is ${ids.length} tokens; the web engine holds ${ctx}. Start a new chat.` }); return; }
  const mine = current = { id, finished: false };
  tail = tail.then(() => loop(mine, ids, Math.min(maxNew, ctx - ids.length)));
}

async function loop(r, ids, maxNew) {
  const t0 = performance.now(), dec = new BendWire.Utf8Stream();
  let reuse = cached.length && cached.length <= ids.length && cached.every((x, i) => ids[i] === x) ? cached.length : 0;
  if (reuse === ids.length) reuse = 0;   // nothing new to feed: the last prompt token must be stepped for its logits
  post({ t: "prompt", id: r.id, ids: ids.length, reused: reuse });
  let prefill = 0, decode = 0, steps = 0, n = 0, firstAt = null;
  const finish = (reason, extra = {}) => {
    if (r.finished) return; r.finished = true; if (current === r) current = null;
    post({ t: "done", id: r.id, reason, ...extra, stats: { numerics: session?.numerics, prompt_tokens: ids.length, reused_tokens: reuse,
      prefill_tokens: ids.length - reuse, prefill_ms: Math.round(prefill), decode_tokens: n, decode_ms: Math.round(decode),
      decode_steps: steps, tokens_per_second: steps ? steps * 1000 / decode : null, first_token_ms: firstAt, wall_ms: Math.round(performance.now() - t0) } });
  };
  r.finish = finish;
  try {
    if (r.finished) return;
    if (!reuse) { session.reset(); cached = []; }
    let pos = reuse, tok = ids[pos];
    for (;;) {
      const out = await session.step(tok, pos);
      cached.push(tok);
      if (r.finished) return;
      if (pos < ids.length - 1) { prefill += out.ms; tok = ids[++pos]; continue; }
      const top = out.top2[0].id;
      if (n === 0) { prefill += out.ms; firstAt = Math.round(performance.now() - t0); } else { decode += out.ms; steps++; }
      if (top === IM_END || top === END_OF_TEXT) { const t = dec.end(); if (t) post({ t: "token", id: r.id, text: t }); finish("stop"); return; }
      n++;
      const text = dec.push(piece(top));
      if (text) post({ t: "token", id: r.id, text, ms: out.ms });
      if (n >= maxNew || pos + 1 >= ctx) { const t = dec.end(); if (t) post({ t: "token", id: r.id, text: t }); finish("max"); return; }
      tok = top; pos++;
    }
  } catch (e) {
    cached = [];
    finish("error", { message: String(e?.message ?? e) });
  }
}

// ---- Kev (decision) -------------------------------------------------------------------------------------------------
// llama.cpp tools/server/server-decision.cpp, COMMON_DECISION_TYPE_KEV, as the phone runs it (/v1/systemone): the prompt is the
// GGUF's systemone template, an option is read at its <|box_end|>, the question at the last token; the final-normed hidden state
// at each goes through cls.output (+ bias) to n_embd_out = 512; score = q[0:256]·k[256:512] / sqrt(256); probabilities =
// softmax(score / temperature.choice); confidence = (p_max - 1/n) / (1 - 1/n).
const f16 = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023; return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15); };
async function kevHead(v) {
  const c = v.cls;
  if (!c.weight || !c.bias || c.weight.type !== "Q8_0" || c.weight.dims[0] !== 2560) throw new Error("Kev's decision head (cls.output) isn't the Q8_0 2560-wide one");
  const rows = c.weight.dims[1], raw = await v.read(c.weight.abs, c.weight.bytes), w = new Float32Array(rows * 2560);
  for (let r = 0; r < rows; r++) for (let b = 0; b < 80; b++) {
    const o = (r * 80 + b) * 34, d = f16(raw[o] | (raw[o + 1] << 8));
    for (let k = 0; k < 32; k++) w[r * 2560 + b * 32 + k] = d * ((raw[o + 2 + k] << 24) >> 24);
  }
  const bias = new Float32Array((await v.read(c.bias.abs, c.bias.bytes)).slice().buffer);
  return { w, bias, rows, temperature: c.temperature, boxEnd: 248050 };
}
const kevText = (t) => String(t).replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
// the criteria object's keys in the order written (transport.ts splices them as text: a JSON object reorders integer-like keys)
function orderedKeys(body, key) {
  const at = body.indexOf(`"${key}":{`); if (at < 0) return null;
  const keys = []; let i = at + key.length + 4;
  while (body[i] !== "}") {
    const start = i; i++; while (body[i] !== '"') i += body[i] === "\\" ? 2 : 1; i++;
    keys.push(JSON.parse(body.slice(start, i))); i++;   // ':'
    while (body[i] !== "," && body[i] !== "}") i++;
    if (body[i] === ",") i++;
  }
  return keys;
}
async function decide({ id, body }) {
  if (!session || !kev) { post({ t: "error", id, message: "The decision model isn't loaded." }); return; }
  const t0 = performance.now(), req = JSON.parse(body), names = Object.keys(req.questions ?? {});
  const qn = names[0], q = req.questions?.[qn];
  if (names.length !== 1 || q?.type !== "choice") { post({ t: "error", id, status: 501, message: "the web decider answers one choice question" }); return; }
  const options = orderedKeys(body, "criteria") ?? Object.keys(q.criteria);
  const prompt = `<|fim_prefix|>${kevText(typeof req.state === "string" ? req.state : JSON.stringify(req.state))}<|fim_middle|>${kevText(q.instructions)}` +
    options.map(o => `<|box_start|>${kevText(o)}<|box_end|>`).join("") + "<|fim_suffix|>";
  const ids = encode(prompt), markers = ids.flatMap((x, i) => x === kev.boxEnd ? [i] : []);
  if (markers.length !== options.length) { post({ t: "error", id, message: "unexpected layout of the decision prompt" }); return; }
  if (ids.length > ctx) { post({ t: "error", id, message: `the decision prompt is ${ids.length} tokens; the web engine holds ${ctx}` }); return; }
  const want = new Set([...markers, ids.length - 1]), rows = new Map();
  let ms = 0;
  session.reset(); cached = [];
  for (let pos = 0; pos < ids.length; pos++) {
    const out = await session.step(ids[pos], pos); ms += out.ms;
    if (want.has(pos)) rows.set(pos, new Float32Array(out.stages["head.h"].buffer.slice(0, 2560 * 4)));
  }
  const embed = (h) => { const e = new Float32Array(kev.rows); for (let r = 0; r < kev.rows; r++) { let a = 0; const o = r * 2560; for (let k = 0; k < 2560; k++) a += kev.w[o + k] * h[k]; e[r] = a + kev.bias[r]; } return e; };
  const half = kev.rows / 2, eq = embed(rows.get(ids.length - 1));
  const scores = markers.map(m => { const ek = embed(rows.get(m)); let dot = 0; for (let i = 0; i < half; i++) dot += eq[i] * ek[half + i]; return dot / Math.sqrt(half); });
  const mx = Math.max(...scores), ex = scores.map(x => Math.exp((x - mx) / kev.temperature)), z = ex.reduce((a, b) => a + b, 0), p = ex.map(x => x / z);
  const best = p.indexOf(Math.max(...p)), n = p.length;
  const answer = { type: "choice", choice: options[best], probabilities: Object.fromEntries(options.map((o, i) => [o, p[i]])), confidence: n < 2 ? 1 : Math.max(0, (p[best] - 1 / n) / (1 - 1 / n)) };
  post({ t: "decided", id, reply: { answers: { [qn]: answer }, usage: { input_tokens: ids.length, output_tokens: 0 } },
    stats: { prompt_tokens: ids.length, engine_ms: Math.round(ms), wall_ms: Math.round(performance.now() - t0), scores } });
}

async function free(keepGen) {
  if (!keepGen) gen++;
  if (current) { current.finished = true; current = null; }
  await tail.catch(() => {});
  const s = session; session = null; cached = [];
  try { await s?.free(); } catch {}
  try { device?.destroy(); } catch {}
  device = null;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.t === "open") await open(m);
    else if (m.t === "chat") chat(m);
    else if (m.t === "decide") await (tail = tail.then(() => decide(m)).catch(e => post({ t: "error", id: m.id, message: String(e?.message ?? e) })));
    else if (m.t === "cancel") { if (current && current.id === m.id) { const r = current; if (r.finish) r.finish("aborted"); else { r.finished = true; current = null; post({ t: "done", id: m.id, reason: "aborted", stats: {} }); } } }
    else if (m.t === "setNumerics") { await tail; const x = await session.setNumerics(m.numerics); post({ t: "numerics", ...x }); }
    else if (m.t === "free") { await free(); post({ t: "freed" }); }
  } catch (err) {
    post({ t: "error", id: m.id, op: m.t, message: String(err?.message ?? err), stack: String(err?.stack ?? ""), code: err?.code, problems: err?.problems, describe: err?.describe });
  }
};
