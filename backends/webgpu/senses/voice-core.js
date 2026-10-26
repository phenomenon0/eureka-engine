// voice-core.js (WEBGPU-E2E): the app's in-tab ears and voice (src/voice/web.ts). Ears: whisper-base.en on ORT-web
// (transformers.js), WebGPU else WASM. Voice: Kokoro-82M af_bella (kokoro-js), WebGPU else WASM. The same models, paths
// and dtypes the proof pages (ears.html, speak.html) measure. Loaded on the first voice use only; nothing leaves the tab.
import T from "./tjs.js";
import { HF, hfBase } from "./hosts.js";
const ASR = "onnx-community/whisper-base.en", TTS = "onnx-community/Kokoro-82M-v1.0-ONNX";
const base = new URL("../", import.meta.url).href;
let device = null, asr = null, tts = null;
const progress = {};   // DEMO-VOICE: optional per-engine progress callbacks (kind -> fn), set by a page before prepare()
export const onProgress = (kind, fn) => { progress[kind] = fn; };
// DEMO-HARDEN: a page can ask for the light voice (Kokoro q8 on WASM, ~92 MB instead of the fp32 326 MB WebGPU set) and can
// release an engine, so a phone tab holds one model at a time. The app's defaults are unchanged.
const opts = { voiceSet: "full" };
export const configure = (o) => Object.assign(opts, o);
const pick = async () => (device ??= (navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null)) ? "webgpu" : "wasm");
export const timings = {};
async function openEars() {
  const d = await pick(), t0 = performance.now();
  const p = await T.pipeline("automatic-speech-recognition", ASR, { device: d, progress_callback: progress.ears, dtype: d === "webgpu" ? { encoder_model: "fp32", decoder_model_merged: "q4" } : { encoder_model: "q8", decoder_model_merged: "q8" } });
  timings.ears_load_ms = Math.round(performance.now() - t0); return p;
}
async function openVoice() {
  const d = await pick(), t0 = performance.now();
  const K = await import("./vendor/kokoro/kokoro.web.js");
  K.env.wasmPaths = base + "senses/vendor/kokoro/"; K.tfenv.allowRemoteModels = HF; K.tfenv.allowLocalModels = !HF;
  globalThis.__KOKORO_VOICES = hfBase(TTS) + "voices/";   // the vendored bundle reads its voice from here (vendor.sh patch)
  K.tfenv.localModelPath = new URL("models/hf/", base).pathname; K.tfenv.backends.onnx.logLevel = "error";
  const lite = opts.voiceSet === "lite" || d !== "webgpu";
  const t = await K.KokoroTTS.from_pretrained(TTS, { dtype: lite ? "q8" : "fp32", device: lite ? "wasm" : d, progress_callback: progress.voice });
  timings.voice_load_ms = Math.round(performance.now() - t0); return t;
}
/** Pays both cold loads (the first hold pays them otherwise). */
export async function prepare({ ears = true, voice = true } = {}) {
  if (ears) asr ??= openEars().catch((e) => { asr = null; throw e; });
  if (voice) tts ??= openVoice().catch((e) => { tts = null; throw e; });
  await Promise.all([ears && asr, voice && tts]);
  return { device: await pick(), ...timings };
}
/** Free an engine's sessions (GPU buffers + wasm heap); the next use opens it again from the browser's cache. */
export async function release(kind) {
  const take = (p) => (p ? p.then((x) => x, () => null) : Promise.resolve(null));
  if (kind === "ears" || kind === "all") { const p = await take(asr); asr = null; await p?.dispose?.(); }
  if (kind === "voice" || kind === "all") { const t = await take(tts); tts = null; await t?.model?.dispose?.(); }
}
/** Any encoded audio (a recording, a .wav) -> mono Float32 at `rate` Hz. */
export async function decodeTo(rate, buf) {
  const ac = new OfflineAudioContext(1, 1, 44100), a = await ac.decodeAudioData(buf);
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(a.duration * rate)), rate), src = off.createBufferSource();
  src.buffer = a; src.connect(off.destination); src.start(); return (await off.startRendering()).getChannelData(0);
}
/** transcribe(pcm @16 kHz) -> { text, ms, device } */
export async function transcribe(pcm) {
  asr ??= openEars().catch((e) => { asr = null; throw e; });
  const p = await asr, t0 = performance.now(), r = await p(pcm);
  return { text: String(r.text ?? "").trim(), ms: Math.round(performance.now() - t0), device: await pick() };
}
/** synth(text) -> { audio: Float32Array, rate, ms, device } (af_bella unless asked) */
export async function synth(text, voice = "af_bella") {
  tts ??= openVoice().catch((e) => { tts = null; throw e; });
  const t = await tts, t0 = performance.now(), out = await t.generate(text, { voice });
  return { audio: out.audio, rate: out.sampling_rate, ms: Math.round(performance.now() - t0), device: await pick() };
}
export const label = { ears: "whisper-base.en, in this tab", voice: "Kokoro af_bella, in this tab" };
globalThis.__bendVoiceCore = { onProgress, configure, release, prepare, transcribe, synth, decodeTo, timings, label };
