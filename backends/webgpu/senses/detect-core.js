// detect-core.js (WEBGPU-E2E-P2): the app camera's boxes (src/app/desktop/DeskCamera.tsx). D-FINE-n (COCO-80) on ORT-web,
// WebGPU else WASM: the graph, 640² stretch, sigmoid scores and 0.5 threshold the proof page (detect.html) measures.
// Loaded on the first "Boxes" tap only, so a camera that only sends photos never pays for it.
import * as ort from "./vendor/ort/ort.webgpu.min.mjs";
import { DETECT } from "./hosts.js";
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href; ort.env.logLevel = "error";
const SZ = 640;
let ready = null, cx = null;
async function open() {
  const ad = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null, ep = ad ? "webgpu" : "wasm", t0 = performance.now();
  const names = (await (await fetch(new URL("assets/dfine_n_ref.json", import.meta.url))).json()).names;
  const sess = await ort.InferenceSession.create(DETECT, { executionProviders: [ep], graphOptimizationLevel: "all", logSeverityLevel: 3 });
  const out = (re, i) => sess.outputNames.find((n) => re.test(n)) ?? sess.outputNames[i];
  return { sess, ep, names, logits: out(/logit/i, 0), boxes: out(/box/i, 1), load_ms: Math.round(performance.now() - t0) };
}
export const label = "D-FINE-n (COCO-80), in this tab";
/** DEMO-HARDEN: open the session now (a page's explicit load step), so a later detect() only runs. */
export async function prepare() { ready ??= open().catch((e) => { ready = null; throw e; }); const m = await ready; return { ep: m.ep, load_ms: m.load_ms }; }
/** DEMO-HARDEN: free the session (one resident model at a time on the demo pages); the next detect() opens it again. */
export async function release() { const r = ready; ready = null; const m = r && (await r.catch(() => null)); await m?.sess?.release?.(); }
/** detect(video | img | canvas) -> { dets: [{ cls, score, xyxy }] in the source's pixels, ep, ms, load_ms }. */
export async function detect(el, thr = 0.5) {
  ready ??= open().catch((e) => { ready = null; throw e; });
  const m = await ready, t0 = performance.now();
  const W = el.videoWidth || el.naturalWidth || el.width, H = el.videoHeight || el.naturalHeight || el.height;
  cx ??= new OffscreenCanvas(SZ, SZ).getContext("2d", { willReadFrequently: true });
  cx.drawImage(el, 0, 0, SZ, SZ);
  const d = cx.getImageData(0, 0, SZ, SZ).data, P = SZ * SZ, x = new Float32Array(3 * P);
  for (let i = 0; i < P; i++) { x[i] = d[4 * i] / 255; x[P + i] = d[4 * i + 1] / 255; x[2 * P + i] = d[4 * i + 2] / 255; }
  const o = await m.sess.run({ [m.sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, SZ, SZ]) });
  const lg = o[m.logits], L = lg.data, B = o[m.boxes].data, nc = lg.dims[2], hit = [];
  for (let i = 0; i < L.length; i++) { const s = 1 / (1 + Math.exp(-L[i])); if (s >= thr) hit.push([i, s]); }
  hit.sort((a, b) => b[1] - a[1]);
  const dets = hit.slice(0, 100).map(([i, s]) => { const q = (i / nc) | 0, c = i % nc, b = B.subarray(4 * q, 4 * q + 4);
    return { cls: m.names[c], c, score: +s.toFixed(4), xyxy: [(b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H].map((v) => +v.toFixed(1)) }; });
  return { dets, ep: m.ep, ms: Math.round(performance.now() - t0), load_ms: m.load_ms, model: label };
}
globalThis.__bendDetectCore = { detect, prepare, release, label };
