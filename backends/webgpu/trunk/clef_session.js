// clef_session.js (CLEF-HEAD): /v1/systemone in the page. The trunk engine (WebGPU) runs the prompt token by token and hands back
// each final-normed hidden state; clef_head.js (CPU) turns them into one score per option; the answers follow format_answer().
//   const c = new ClefDecider(); await c.open(read)       // read(offset, length) -> Uint8Array over clef-flash's .gguf
//   const r = await c.decide({state, questions})          // {answers, usage: {input_tokens}, stats, debug: {ids, spans, scores, pieces}}
// Needs tok.js loaded on the page (window.Module = the app's Qwen3.5 tokenizer), as smoke.html does.
import { gguf_read } from "/engine/gguf.js";
import { trunk_open } from "./trunk_engine.js";
import { clef_head_open } from "./clef_head.js";
import { clef_questions, clef_pieces, clef_tokens, CLEF_TEMPLATE_SHA256 } from "./clef_prompt.js";

const sha256 = async (s) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)))).map(b => b.toString(16).padStart(2, "0")).join("");

export async function clef_tokenizer(T = window.Module) {
  if (!T.__clefInit) {
    T.FS.mkdir("/tok"); T.FS.writeFile("/tok/tokenizer.bin", new Uint8Array(await (await fetch("/models/qwen35-tokenizer.bin")).arrayBuffer()));
    if (T._tok_init() !== 0) throw new Error("tokenizer.bin did not load");
    T.__clefInit = true;
  }
  return (text) => { const n = T.lengthBytesUTF8(text) + 1, p = T._malloc(n); T.stringToUTF8(text, p, n);
    let cap = 16384, o = T._malloc(cap * 4), got = T._tok_encode(p, o, cap);
    if (got < 0) { T._free(o); cap = -got; o = T._malloc(cap * 4); got = T._tok_encode(p, o, cap); }
    const ids = Array.from(T.HEAP32.subarray(o >> 2, (o >> 2) + got)); T._free(p); T._free(o); return ids; };
}

export class ClefDecider {
  async open(read, { ctx = 4096, onProgress = () => {} } = {}) {
    const t0 = performance.now();
    this.encode = await clef_tokenizer();
    const g = await gguf_read(read), a = g.kv["general.architecture"];
    const tmpl = g.kv["tokenizer.chat_template.systemone"];
    if (!tmpl) throw new Error('It has no "systemone" template, so it is not a decision model.');
    this.template_ok = (await sha256(tmpl)) === CLEF_TEMPLATE_SHA256;
    if (!this.template_ok) throw new Error("Its systemone template isn't the one clef_prompt.js renders (sha256 differs).");
    this.head = await clef_head_open(read, g);
    const th = performance.now();
    this.e = await trunk_open(read, { ctx, gguf: g, onProgress });
    return { open_ms: Math.round(performance.now() - t0), head_load_ms: Math.round(th - t0), arch: a, head_bytes: this.head.bytes, resident_bytes: this.e.resident_bytes };
  }
  async decide(body, { debug = false } = {}) {
    const questions = clef_questions(body), pieces = clef_pieces(body.state, questions), tk = clef_tokens(pieces, this.encode);
    const n = tk.ids.length, d = this.e.shape.d;
    if (n > this.e.ctx) throw new Error(`The prompt (${n} tokens) doesn't fit the ${this.e.ctx}-token context.`);
    const t0 = performance.now();
    await this.e.reset();   // clef's graph has no cache: every request starts from zero state
    const hidden = new Float32Array(n * d);
    for (let i = 0; i < n; i++) hidden.set((await this.e.step(tk.ids[i], { wantHidden: true, noHead: true })).hidden, i * d);
    const t1 = performance.now(), trace = {};
    const scores = await this.head.scores(hidden, tk.ids, tk.spans, { trace });
    const answers = this.head.answers(questions, scores), t2 = performance.now();
    const r = { answers, usage: { input_tokens: n, output_tokens: 0 },
      stats: { trunk_ms: Math.round(t1 - t0), trunk_tok_s: +(n * 1000 / (t1 - t0)).toFixed(1), head_ms: Math.round(t2 - t1) } };
    if (debug) r.debug = { ids: tk.ids, spans: tk.spans, pieces, scores: Array.from(scores), prior: trace.prior, cosine: trace.cosine, residual: trace.residual };
    return r;
  }
  free() { this.e?.free(); this.e = null; }
}
