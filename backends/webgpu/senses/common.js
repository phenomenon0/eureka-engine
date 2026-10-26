// common.js: the shared shell for the in-page sense demos. Every page runs entirely in this tab (models are static files
// under /models/; no inference server). ?ep=webgpu|wasm picks the runtime path; the page reports what it really ran on.
export const q = new URLSearchParams(location.search);
export const S = (window.__SENSE__ = { page: document.body.dataset.page, ep: null, status: "init", t: {}, errors: [], notes: [] });
const $ = (id) => document.getElementById(id);
export const now = () => performance.now();

export async function pickEP() {
  const want = q.get("ep") ?? "webgpu";
  let ep = want;
  if (want === "webgpu") {
    const ad = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
    if (!ad) { ep = "wasm"; S.notes.push("WebGPU not available in this browser: fell back to WASM (CPU)"); }
    else S.adapter = ad.info ? { vendor: ad.info.vendor, architecture: ad.info.architecture } : {};
    S.shaderF16 = !!ad?.features.has("shader-f16");
  }
  S.ep = ep; S.want = want; S.crossOriginIsolated = self.crossOriginIsolated;
  return ep;
}

export function shell(title, sub) {
  const ep = q.get("ep") ?? "webgpu", link = (e) => { const u = new URLSearchParams(q); u.set("ep", e); return "?" + u; };
  document.body.insertAdjacentHTML("afterbegin", `<header><a href="./" title="All senses">←</a><h1>${title}</h1><span class="sub">${sub}</span><span class="sp"></span>
    <a class="chip ${ep === "webgpu" ? "on" : ""}" href="${link("webgpu")}">WebGPU</a><a class="chip ${ep === "wasm" ? "on" : ""}" href="${link("wasm")}">WASM (CPU)</a>
    <span class="chip" id="epchip"><span class="dot" id="epdot"></span><span id="eptxt">starting</span></span></header>`);
}
export function ran(ep) { $("epdot").className = "dot ok"; $("eptxt").textContent = "ran on " + (ep === "webgpu" ? "WebGPU" : "WASM (CPU)") + (S.want !== ep ? " (fallback)" : ""); }
export function log(s) { const el = $("log"); if (el) el.textContent += s + "\n"; }
export function status(s, kind = "") { const el = $("status"); if (el) { el.textContent = s; el.className = "status " + kind; } }
export function stats(o) { const el = $("stats"); if (el) el.innerHTML = Object.entries(o).map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join(""); }
export function fail(where, e) {
  const msg = `${where}: ${e?.message ?? e}`; S.errors.push(msg); S.status = "error"; status(msg, "err"); log(msg);
  $("epdot") && ($("epdot").className = "dot bad");
}
export const ms = (x) => (x == null ? "–" : x < 10 ? x.toFixed(1) + " ms" : Math.round(x) + " ms");
export const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };

// Audio helpers: decode any blob/url to mono Float32 at `rate` Hz.
export async function decodeTo(rate, buf) {
  const ac = new OfflineAudioContext(1, 1, 44100), a = await ac.decodeAudioData(buf);
  const off = new OfflineAudioContext(1, Math.ceil(a.duration * rate), rate), src = off.createBufferSource();
  src.buffer = a; src.connect(off.destination); src.start(); return (await off.startRendering()).getChannelData(0);
}
