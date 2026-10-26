// look-core.js: the in-tab describer the app's tab shell calls for a photo turn (src/engines/tab/shell.ts). SmolVLM-256M:
// SigLIP tower -> pixel-shuffle projector -> SmolLM2 decoder, three ONNX graphs on ORT-web (WebGPU, else WASM). Loaded on
// the first photo only, so a text-only session never pays for it. Not bend's Qwen3.5 tower (that is the phone app's).
import T from "./tjs.js";
const M = "HuggingFaceTB/SmolVLM-256M-Instruct";
let ready = null;
// APP-IPHONE-MEDIA: a phone's tab already holds the chat model, and iOS ends a tab that grows past its memory (no error, the
// page just reloads), so a phone reads the photo with the small set: q4 tower + int8 embeddings + q4 decoder, ~170 MB to
// fetch instead of ~550 MB (fp32 tower 357 MB). A computer keeps the measured fp32 set. ?lookset=phone|full overrides.
const PHONE = /iPhone|iPod|Android.+Mobile/i.test(navigator.userAgent);
const set = new URLSearchParams(location.search).get("lookset") ?? (PHONE ? "phone" : "full");
const DTYPES = { full: { vision_encoder: "fp32", embed_tokens: "fp32", decoder_model_merged: "q4" }, phone: { vision_encoder: "q4", embed_tokens: "int8", decoder_model_merged: "q4" },
  wasm: { vision_encoder: "q8", embed_tokens: "q8", decoder_model_merged: "q8" } };
// The first photo downloads the reader: its progress is what the photo bubble says ("Warming vision… N%"), never a silent wait.
export const life = { phase: "cold", progress: 0, loaded: 0, total: 0, set: null, device: null, error: null };
const files = new Map();
function onProgress(e) {
  if (!e?.file || !/\.onnx|\.json/.test(e.file)) return;
  if (e.status === "progress" || e.status === "done") files.set(e.file, { loaded: e.loaded ?? files.get(e.file)?.total ?? 0, total: e.total ?? files.get(e.file)?.total ?? 0 });
  let loaded = 0, total = 0; for (const f of files.values()) { loaded += f.loaded; total += f.total; }
  Object.assign(life, { loaded, total, progress: total ? Math.min(99, Math.round(loaded / total * 100)) : 0 });
}
async function open() {
  const ad = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null, device = ad ? "webgpu" : "wasm";
  Object.assign(life, { phase: "warming", progress: 0, error: null, device, set: device === "webgpu" ? set : "wasm" });
  try {
    const proc = await T.AutoProcessor.from_pretrained(M, { progress_callback: onProgress });
    const model = await T.AutoModelForVision2Seq.from_pretrained(M, { device, dtype: DTYPES[life.set] ?? DTYPES.full, progress_callback: onProgress });
    Object.assign(life, { phase: "warm", progress: 100 });
    return { proc, model, device };
  } catch (e) { Object.assign(life, { phase: "cold", error: String(e?.message ?? e) }); throw e; }
}
export const label = "SmolVLM-256M, in this tab";
/** DEMO-HARDEN: open the reader now (a page's explicit load step). */
export async function prepare() { ready ??= open().catch((e) => { ready = null; throw e; }); const { device } = await ready; return { device, set: life.set }; }
/** DEMO-HARDEN: free the reader's three sessions (GPU buffers + wasm heap). The next describe/choose opens it again from the cache. */
export async function release() {
  const r = ready; ready = null; files.clear(); const m = r && (await r.catch(() => null)); await m?.model?.dispose?.();
  Object.assign(life, { phase: "cold", progress: 0 });
}
/** The set this browser would fetch, and its size in MB (what a page shows on its load step). */
export const plan = (device) => { const s = device === "webgpu" ? set : "wasm"; return { set: s, mb: { full: 575, phone: 180, wasm: 260 }[s] ?? 575 }; };
/** describe(imageUrl) -> { text, device, ms }: a plain description of the photo, greedy, at most 96 tokens. */
export async function describe(url, prompt = "Describe this image in detail: the objects, people, any text, and the setting.") {
  const t0 = performance.now(); ready ??= open().catch((e) => { ready = null; throw e; });
  const { proc, model, device } = await ready, image = await T.load_image(url);
  const text = proc.apply_chat_template([{ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] }], { add_generation_prompt: true });
  const inputs = await proc(text, [image], { do_image_splitting: false });
  const out = await model.generate({ ...inputs, max_new_tokens: 96, do_sample: false, repetition_penalty: 1.1 });
  const [s] = proc.batch_decode(out.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true });
  return { text: s.trim(), device, set: life.set, ms: Math.round(performance.now() - t0) };
}
/** choose(imageUrl, prompt, options) -> { logp: number[], ms, device }: one forward pass, no sampling. The model's own next-token logits
 * over the options (a letter "A", "B", … or a word "Yes", "No", …; the first token of each, bare and space-led, logsumexp'd) at the first answer position. */
export async function choose(url, prompt, options) {
  const t0 = performance.now(); ready ??= open().catch((e) => { ready = null; throw e; });
  const { proc, model, device } = await ready, image = await T.load_image(url);
  const text = proc.apply_chat_template([{ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] }], { add_generation_prompt: true });
  const inputs = await proc(text, [image], { do_image_splitting: false });
  const ids = options.map((l) => [l, " " + l].map((v) => proc.tokenizer.encode(v, { add_special_tokens: false }).at(0)));
  let row = null;
  await model.generate({ ...inputs, max_new_tokens: 1, do_sample: false, logits_processor: [(_, lg) => { row ??= Float32Array.from(lg.data ?? lg); return lg; }] });
  const lse = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); };
  return { logp: ids.map((p) => lse(p.map((i) => row[i]))), ms: Math.round(performance.now() - t0), device };
}
globalThis.__bendLookCore = { describe, choose, prepare, release, plan, label, life };
