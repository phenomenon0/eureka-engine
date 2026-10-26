// gpu-shim.js (PHONEFAST): the WebGPU model path inside the app's WebView. When navigator.gpu + an adapter work, it opens
// the WGSL vision tower (wgtower/tower.js, the 4B mmproj, F16) and the turbo web engine (wgtower/q35v_engine.js: the
// bendweb-live kernels plus image-row injection) on the GGUF that models.json selects (default Kev-4B-Q4_K_M.gguf), both read in place from the app's files dir
// (Server.kt /data/), prewarms both (one tiny tower encode + 2 steps compile every pipeline), then answers
// POST /v1/chat/completions in-page (same SSE frames as Server.kt). Anything it can't do (decide, open failure, a WebGPU
// error) stays with / falls back to the native CPU engine: it tells the shell via POST /gpu/state.
import { gpu_device } from "./wgtower/gpu.js";
import { tower_open } from "./wgtower/tower.js";
import { vpre_image } from "./wgtower/vpre.js";
import { gguf_pack } from "./engine/ggufpack.js";
import { q35_open } from "./wgtower/q35v_engine.js";

const N = window.__BEND_NATIVE__, IM_END = 248046, END_OF_TEXT = 248044, PD = 2560;
const post = (p, b) => fetch(p, { method: "POST", headers: { "X-Api-Key": N.key, "Content-Type": "application/json" }, body: JSON.stringify(b) }).catch(() => {});
const log = (line) => { console.info("gpu " + line); post("/gpu/log", { line }); };
const reader = (url) => async (o, n) => {
  for (let k = 0; ; k++) {
    try { const r = await fetch(url, { headers: { Range: `bytes=${o}-${o + n - 1}` } }); if (r.status !== 206) throw new Error(`${url}: HTTP ${r.status}`); return new Uint8Array(await r.arrayBuffer()); }
    catch (e) { if (k >= 4) throw e; await new Promise((r) => setTimeout(r, 300 * (k + 1))); }
  }
};

let E = null, T = null, Tok = null, busy = Promise.resolve();
const tokenizer = async () => {
  const bytes = new Uint8Array(await (await fetch("/data/qwen35-tokenizer.bin")).arrayBuffer());
  const M = await new Promise((resolve, reject) => {
    window.Module = { locateFile: (p) => "/engine/" + p, onRuntimeInitialized: () => resolve(window.Module), onAbort: (e) => reject(new Error("tokenizer: " + e)) };
    const s = document.createElement("script"); s.src = "/engine/tok.js"; s.onerror = () => reject(new Error("tok.js")); document.head.appendChild(s);
  });
  try { M.FS.mkdir("/tok"); } catch {}
  M.FS.writeFile("/tok/tokenizer.bin", bytes);
  if (M._tok_init() !== 0) throw new Error("tokenizer.bin did not load");
  return M;
};
const encode = (text) => {
  const n = Tok.lengthBytesUTF8(text) + 1, p = Tok._malloc(n); Tok.stringToUTF8(text, p, n);
  let cap = 8192, out = Tok._malloc(cap * 4), got = Tok._tok_encode(p, out, cap);
  if (got < 0) { Tok._free(out); cap = -got; out = Tok._malloc(cap * 4); got = Tok._tok_encode(p, out, cap); }
  const ids = Array.from(Tok.HEAP32.subarray(out >> 2, (out >> 2) + got)); Tok._free(p); Tok._free(out); return ids;
};
const piece = (id) => { const p = Tok._malloc(65536), n = Tok._tok_piece(id, p, 65536), b = Tok.HEAPU8.slice(p, p + Math.min(n, 65536)); Tok._free(p); return b; };

// Server.kt prompt(): ChatML no-think, newest photo only, markers in content defused; split around the one image slot.
const IMG = "<|vision_start|><|image_pad|><|vision_end|>";
function render(messages) {
  let lastImg = -1, url = null, s = "";
  messages.forEach((m, i) => { if (Array.isArray(m.content) && m.content.some((p) => p.type === "image_url")) lastImg = i; });
  messages.forEach((m, i) => {
    let text = Array.isArray(m.content) ? m.content.filter((p) => p.type === "text").map((p) => p.text).join("\n") : String(m.content ?? "");
    if (i === lastImg) { const p = m.content.find((p) => p.type === "image_url"); url = p.image_url?.url ?? p.image_url; text = "\u0001" + (text.trim() ? text : "Describe this photo in one or two sentences."); }
    s += `<|im_start|>${m.role ?? "user"}\n${text.replaceAll("<|", "<¦").replaceAll("|>", "¦>").replace("\u0001", IMG)}<|im_end|>\n`;
  });
  s += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  const [pre, post] = url ? s.split("<|image_pad|>") : [s, null];
  return { url, pre, post };
}

// One turn: text before the image, the tower's rows (slot s0+i, rope (s0, s0+i/nx, s0+i%nx)), text after (rope = slot - shift), then greedy.
async function turn(req, onText) {
  const t0 = performance.now(), { url, pre, post: after } = render(req.messages), n = Math.min(req.max_tokens ?? 256, 256);
  const st = { mode: "webgpu", image_tokens: 0, tower_ms: 0 };
  E.reset();
  let slot = 0, last = null, shift = 0, tw = null;
  for (const id of encode(pre)) { last = await E.step(id, slot, { rope: [slot, slot, slot] }); slot++; }
  if (url && !T) { st.image_dropped = true; for (const id of encode(after)) { last = await E.step(id, slot, { rope: [slot, slot, slot] }); slot++; } }
  else if (url) {
    const tv = performance.now(), blob = await (await fetch(url)).blob(), p = await vpre_image(blob);
    tw = await T.encode(p.planar, p.nx, p.ny);
    st.vpre_ms = Math.round(performance.now() - tv - tw.ms); st.tower_ms = Math.round(tw.ms); st.image_tokens = tw.T; st.grid = [tw.nx_t, tw.ny_t];
    const s0 = slot; shift = tw.T - Math.max(tw.nx_t, tw.ny_t);
    for (let i = 0; i < tw.T; i++) { last = await E.step(0, slot, { row: { buffer: tw.embd, offset: i * PD * 4 }, rope: [s0, s0 + Math.floor(i / tw.nx_t), s0 + (i % tw.nx_t)] }); slot++; }
    for (const id of encode(after)) { const r = slot - shift; last = await E.step(id, slot, { rope: [r, r, r] }); slot++; }
    tw.embd.destroy();
  }
  st.prompt_tokens = slot; st.prefill_ms = Math.round(performance.now() - t0);
  const dec = new TextDecoder(); let out = "", k = 0; const td = performance.now();
  for (; k < n; k++) {
    const id = last.top2[0].id;
    if (id === IM_END || id === END_OF_TEXT) break;
    const t = dec.decode(piece(id), { stream: true }); out += t; if (t) onText(t);
    if (k === 0) st.ttft_ms = Math.round(performance.now() - t0);
    if (k + 1 < n) { const r = slot - shift; last = await E.step(id, slot, { rope: [r, r, r] }); slot++; }
  }
  st.n = k; st.decode_tok_s = k > 1 ? +((k - 1) / ((performance.now() - td) / 1000)).toFixed(1) : 0; st.total_ms = Math.round(performance.now() - t0);
  log(`chat ${JSON.stringify(st)}`);
  return { out, st };
}

function serveChat(req) {
  const enc = new TextEncoder();
  if (!req.stream) return busy = busy.then(() => turn(req, () => {})).then(({ out }) => new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: out } }] }), { headers: { "Content-Type": "application/json" } }));
  let ctl; const body = new ReadableStream({ start(c) { ctl = c; } });
  const frame = (o) => ctl.enqueue(enc.encode(`data: ${typeof o === "string" ? o : JSON.stringify(o)}\n\n`));
  busy = busy.then(() => turn(req, (t) => frame({ choices: [{ delta: { content: t } }] })))
    .then(({ st }) => { if (st.image_tokens) frame({ llb: { look: true, image_tokens: st.image_tokens, canvas: st.grid, first_ms: st.ttft_ms, total_ms: st.total_ms, tower_ms_sum: st.tower_ms, llm_ms_sum: st.total_ms - st.tower_ms, engine: "webgpu" } }); frame("[DONE]"); ctl.close(); })
    .catch((e) => { log("chat error " + (e?.message ?? e)); frame({ error: { message: String(e?.message ?? e) } }); ctl.close(); });
  return Promise.resolve(new Response(body, { headers: { "Content-Type": "text/event-stream" } }));
}

async function open() {
  const t0 = performance.now();
  const g = await gpu_device(), d = g.device;
  d.lost.then((l) => { E = null; log("device lost " + (l.message || l.reason)); post("/gpu/state", { state: "failed", reason: "device lost: " + (l.message || l.reason) }); });
  d.addEventListener("uncapturederror", (e) => log("GPU error: " + e.error.message));
  log(`open adapter ${JSON.stringify(g.facts)}`);
  post("/gpu/state", { state: "loading" });
  Tok = await tokenizer();
  // the LLM first (the piece that must fit), then the tower: each stage logs, so an lmkd kill names the stage it died in
  let t = performance.now();
  const v = await gguf_pack(reader(`/data/${N.model || "Kev-4B-Q4_K_M"}.gguf`));
  if (!v.ok) throw new Error(v.problems.join(" "));
  log("open pack ok");
  E = await q35_open(g, v.source, v.manifest, { ctx: 4096, numerics: "turbo", canon: false, split: "exact", verify: false, subgroups: d.features.has("subgroups") });
  const engineOpen = Math.round(performance.now() - t); t = performance.now();
  log(`open engine_ms=${engineOpen} resident=${E.facts.resident_bytes}`);
  if (N.gpuVision !== false) { T = await tower_open(g, reader("/data/mmproj-F16.gguf")); log("open tower done"); }
  const towerOpen = Math.round(performance.now() - t);
  log(`open tower_ms=${towerOpen} engine_ms=${engineOpen} dispatches=${E.facts.dispatches} subgroups=${E.facts.subgroups} resident=${E.facts.resident_bytes}`);
  // prewarm: a 64x64 grey photo through vpre + tower (compiles its 7 pipelines, pages the weights) and 2 engine steps (text + row variant)
  t = performance.now();
  const c = new OffscreenCanvas(64, 64), x = c.getContext("2d"); x.fillStyle = "#808080"; x.fillRect(0, 0, 64, 64);
  let cold = -1, warm = -1;
  E.reset(); await E.step(encode("Hi")[0], 0, { rope: [0, 0, 0] });
  if (T) {
    const p = await vpre_image(await c.convertToBlob({ type: "image/png" })), tw = await T.encode(p.planar, p.nx, p.ny); cold = Math.round(tw.ms);
    const tw2 = await T.encode(p.planar, p.nx, p.ny); warm = Math.round(tw2.ms);
    await E.step(0, 1, { row: { buffer: tw.embd, offset: 0 }, rope: [1, 1, 1] });
    tw.embd.destroy(); tw2.embd.destroy();
  }
  E.reset();
  log(`prewarm tower_cold_ms=${cold} tower_warm_ms=${warm} total_ms=${Math.round(performance.now() - t)} open_total_ms=${Math.round(performance.now() - t0)}`);
  const f = window.fetch.bind(window);
  window.fetch = (u, o = {}) => {
    const path = new URL(typeof u === "string" ? u : u.url, location.href).pathname;
    if (E && path === "/v1/chat/completions" && (o.method ?? "GET").toUpperCase() === "POST") return serveChat(JSON.parse(o.body));
    return f(u, o);
  };
  window.__bendGpu = { turn, encode };
  post("/gpu/state", { state: "live" });
}

if (N && !N.cpuOnly) (async () => {
  try {
    const a = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!a) throw new Error("no WebGPU adapter");
    await open();
  } catch (e) { E = null; log("open failed " + (e?.stack ?? e)); post("/gpu/state", { state: "failed", reason: String(e?.message ?? e) }); }
})();
