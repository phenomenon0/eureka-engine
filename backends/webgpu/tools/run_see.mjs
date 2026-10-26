// run_see.mjs URL OUT.png: adapted from WEB-ALLMODAL tools/shoot.mjs (same page, same wait contract).
// Real Chrome, WebGPU on (Vulkan/ANGLE, RTX 3090), waits for window.__SENSE__.bench, screenshots,
// prints the page's own result JSON plus console/page error counts.
import { chromium } from "playwright";
import fs from "node:fs";

const [url, out, waitExpr = "window.__SENSE__ && window.__SENSE__.status === 'ok' && !!window.__SENSE__.bench", tmo = "600000", w = "1280", h = "860"] = process.argv.slice(2);
const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan",
  "--use-angle=vulkan", "--enable-webgpu-developer-features", "--disable-gpu-sandbox", "--no-sandbox",
  "--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"];
const b = await chromium.launch({ channel: "chrome", headless: false, args });
const p = await b.newPage({ viewport: { width: +w, height: +h } });
const cons = [], errs = [], t0 = Date.now();
p.on("console", (m) => cons.push({ type: m.type(), text: m.text().slice(0, 500) }));
p.on("pageerror", (e) => errs.push(String(e).slice(0, 500)));
let timedOut = false;
try {
  await p.goto(url, { waitUntil: "load", timeout: 60000 });
  await p.waitForFunction(waitExpr, null, { timeout: +tmo, polling: 250 });
} catch { timedOut = true; }
await p.waitForTimeout(400);
await p.screenshot({ path: out, fullPage: false });
const result = await p.evaluate(() => {
  const r = window.__SENSE__ ?? window.__smoke ?? null;
  try { return JSON.parse(JSON.stringify(r, (k, v) => (k === "events" ? undefined : v))); } catch { return String(r); }
});
const ua = await p.evaluate(() => navigator.userAgent);
const consoleErrors = cons.filter((c) => c.type === "error");
const doc = {
  url, out, ms: Date.now() - t0, timedOut, ua,
  consoleErrors: consoleErrors.length, pageErrors: errs.length,
  errors: [...consoleErrors.map((c) => c.text), ...errs],
  warnings: cons.filter((c) => c.type === "warning").length,
  result,
};
fs.writeFileSync(out.replace(/\.png$/, ".json"), JSON.stringify(doc, null, 1));
console.log(JSON.stringify(doc, null, 1).slice(0, 4000));
await b.close();
