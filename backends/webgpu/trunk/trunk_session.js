// trunk_session.js (WEB-TRUNK-PORT): the picker's handle on the trunk engine, the same shape as family.js's Engine
// (open(source, {ctx, label}) -> {open_ms, facts}; chat(history, maxNew) with ontoken({text}) -> {reason, stats}), run on the
// page's thread (the GPU does the work; each step is awaited) with the app's C tokenizer (tok.wasm, Qwen3.5's vocabulary).
import { gguf_read } from "/engine/gguf.js";
import { gguf_vocab_sha256 } from "/engine/ggufpack.js";
import { trunk_open, trunk_check } from "./trunk_engine.js";
const QWEN_VOCAB = "9c144d2eb9c628950b338dd86b2e1bd3526e040a1420362f5f2e5f28d6d9a95c";
let Tok = null;
async function tokenizer() {
  if (Tok) return Tok;
  Tok = await new Promise((resolve, reject) => {
    window.Module = { locateFile: (p) => "/engine/" + p, onRuntimeInitialized: () => resolve(window.Module), onAbort: (e) => reject(new Error("tokenizer: " + e)) };
    const s = document.createElement("script"); s.src = "/engine/tok.js"; s.onerror = () => reject(new Error("tok.js did not load")); document.head.append(s);
  });
  Tok.FS.mkdir("/tok"); Tok.FS.writeFile("/tok/tokenizer.bin", new Uint8Array(await (await fetch("/models/qwen35-tokenizer.bin")).arrayBuffer()));
  if (Tok._tok_init() !== 0) throw new Error("tokenizer.bin did not load");
  return Tok;
}
const encode = (text) => { const T = Tok, n = T.lengthBytesUTF8(text) + 1, p = T._malloc(n); T.stringToUTF8(text, p, n);
  let cap = 8192, o = T._malloc(cap * 4), got = T._tok_encode(p, o, cap);
  if (got < 0) { T._free(o); cap = -got; o = T._malloc(cap * 4); got = T._tok_encode(p, o, cap); }
  const ids = Array.from(T.HEAP32.subarray(o >> 2, (o >> 2) + got)); T._free(p); T._free(o); return ids; };
const piece = (id) => { const p = Tok._malloc(65536), n = Tok._tok_piece(id, p, 65536), b = Tok.HEAPU8.slice(p, p + Math.min(n, 65536)); Tok._free(p); return b; };
// Qwen3.5's chat layout with the empty think block (as family-worker.js renders it)
const render = (messages) => messages.map(m => m.role === "assistant" ? `<|im_start|>assistant\n<think>\n\n</think>\n\n${m.content}<|im_end|>\n` : `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n<think>\n\n</think>\n\n";

export class TrunkEngine {
  constructor() { this.e = null; this.ids = []; this.onprogress = () => {}; this.ontoken = () => {}; this.w = { terminate: () => this.terminate() }; }
  async open(source, { ctx = 4096 } = {}) {
    const t0 = performance.now();
    const read = source.file ? async (o, n) => new Uint8Array(await source.file.slice(o, o + n).arrayBuffer())
      : async (o, n) => { const r = await fetch(source.url, { headers: { range: `bytes=${o}-${o + n - 1}` } }); if (r.status !== 206) throw new Error("range read " + r.status); return new Uint8Array(await r.arrayBuffer()); };
    await tokenizer();
    const g = await gguf_read(read), chk = trunk_check(g);
    if (!chk.ok) throw new Error(chk.problems.join(" "));
    if ((await gguf_vocab_sha256(g)) !== QWEN_VOCAB) throw new Error("Its tokenizer isn't Qwen3.5's; the web build only carries Qwen3.5's tokenizer.");
    this.e = await trunk_open(read, { ctx, gguf: g, onProgress: ({ done, total }) => this.onprogress({ phase: "weights", done, total }) });
    this.ids = [];
    const s = this.e.shape;
    return { open_ms: Math.round(performance.now() - t0), facts: { resident_bytes: this.e.resident_bytes, ctx, numerics: "trunk f32", dispatches: this.e.dispatches, adapter: this.e.adapter,
      shape: `${s.arch} · ${s.n_layer} layers · d${s.d}`, skipped: chk.skipped.length } };
  }
  async chat(history, maxNew = 256) {
    const e = this.e, t0 = performance.now(), ids = encode(render(history));
    let reuse = 0; while (reuse < this.ids.length && reuse < ids.length - 1 && this.ids[reuse] === ids[reuse]) reuse++;
    if (reuse < this.ids.length) { await e.reset(); reuse = 0; this.ids = []; }   // the recurrent state can't rewind: replay from 0
    if (ids.length + maxNew > e.ctx) throw new Error(`The conversation (${ids.length} tokens) and the reply don't fit the ${e.ctx}-token context.`);
    let r, prefill = 0;
    for (let i = reuse; i < ids.length; i++) { r = await e.step(ids[i]); prefill += r.ms; }
    this.ids = ids.slice();
    const stops = new Set([e.shape.eos, 248044, 248046]), td = new TextDecoder();
    let decode = 0, steps = 0, n = 0, reason = "length", firstAt = Math.round(performance.now() - t0);
    while (n < maxNew) {
      const id = r.id;
      if (stops.has(id)) { reason = "stop"; break; }
      this.ontoken({ text: td.decode(piece(id), { stream: true }) }); n++;
      if (n >= maxNew) break;
      r = await e.step(id); this.ids.push(id); decode += r.ms; steps++;
    }
    return { reason, stats: { prefill_tokens: ids.length - reuse, prefill_ms: Math.round(prefill), decode_tokens: n, decode_ms: Math.round(decode), decode_steps: steps,
      tokens_per_second: steps ? steps * 1000 / decode : null, first_token_ms: firstAt, wall_ms: Math.round(performance.now() - t0) } };
  }
  terminate() { this.e?.free(); this.e = null; }
}
