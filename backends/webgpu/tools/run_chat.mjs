// run_chat.mjs URL REPS OUT: drive the exact Mac trunk smoke page (smoke.html) N times in real Chrome
// with WebGPU on the NVIDIA 3090 (Vulkan/ANGLE). Reads window.__smoke after document.title === "DONE".
// Prints one JSON document: {url, reps, runs:[...], summary:{...}}.
import { chromium } from "playwright";
import crypto from "node:crypto";
import fs from "node:fs";

const [url, repsArg, outPath] = process.argv.slice(2);
const REPS = Number(repsArg ?? 10);
const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan",
  "--use-angle=vulkan", "--enable-webgpu-developer-features", "--disable-gpu-sandbox", "--no-sandbox",
  "--autoplay-policy=no-user-gesture-required"];
const browser = await chromium.launch({ channel: "chrome", headless: false, args });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");
const runs = [];
const t00 = Date.now();
for (let i = 0; i < REPS; i++) {
  const page = await ctx.newPage();
  const cons = [], perrs = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") cons.push({ type: m.type(), text: m.text().slice(0, 500) }); });
  page.on("pageerror", (e) => perrs.push(String(e).slice(0, 500)));
  const t0 = Date.now();
  let timedOut = false;
  try {
    await page.bringToFront();
    await page.goto(url, { waitUntil: "load", timeout: 60000 });
    await page.waitForFunction(() => document.title === "DONE", null, { timeout: 600000, polling: 500 });
  } catch (e) { timedOut = true; perrs.push("driver: " + String(e).slice(0, 300)); }
  let smoke = null;
  try { smoke = await page.evaluate(() => window.__smoke ? JSON.parse(JSON.stringify(window.__smoke)) : null); } catch (e) { perrs.push("evaluate: " + String(e).slice(0, 200)); }
  const ua = await page.evaluate(() => navigator.userAgent).catch(() => null);
  const r = smoke?.result ?? null;
  const run = {
    rep: i, wall_ms: Date.now() - t0, timed_out: timedOut, ua,
    ok: !!r && !r.error && !(smoke?.errors?.length) && !perrs.length && (r.decode_steps ?? 0) > 0,
    errors: [...(smoke?.errors ?? []), ...perrs],
    console: cons,
    result: r ? {
      model: r.model, arch: r.arch, shape: r.shape, ctx: r.ctx, open_ms: r.open_ms,
      resident_bytes: r.resident_bytes, estimate_bytes: r.estimate_bytes, dispatches: r.dispatches,
      limits: r.limits, adapter: r.adapter, skipped_tensors: r.skipped_tensors,
      prompt_tokens: r.prompt_tokens, prefill_ms: r.prefill_ms, prefill_tok_s: r.prefill_tok_s,
      decode_steps: r.decode_steps, decode_ms: r.decode_ms, tokens_per_second: r.tokens_per_second,
      ids: r.ids, text: r.text,
    } : null,
    text_sha256: r?.text ? sha(r.text) : null,
    ids_sha256: r?.ids ? sha(r.ids.join(",")) : null,
    text_len: r?.text?.length ?? null,
  };
  runs.push(run);
  console.log(`rep ${i}: ok=${run.ok} steps=${r?.decode_steps} tok/s=${r?.tokens_per_second} open=${r?.open_ms}ms text_sha=${run.text_sha256?.slice(0, 12)} errs=${run.errors.length}`);
  await page.close().catch(() => {});
  await new Promise((r) => setTimeout(r, 750));
}
// health probe: a fresh page must still get an adapter after 10 loads
let health = { adapter: false, info: null, error: null };
try {
  const hp = await ctx.newPage();
  await hp.goto(new URL("/senses/assets/bus.jpg", url).href, { waitUntil: "load", timeout: 30000 }).catch(() => {});
  health = await hp.evaluate(async () => {
    if (!navigator.gpu) return { adapter: false, info: null, error: "no navigator.gpu" };
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    return { adapter: !!a, info: a?.info ? { vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description } : null, error: null };
  });
  await hp.close();
} catch (e) { health.error = String(e).slice(0, 300); }
await browser.close();

const sorts = (a) => a.slice().sort((x, y) => x - y);
const med = (a) => { const s = sorts(a); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const good = runs.filter((r) => r.ok && r.result);
const tps = good.map((r) => r.result.tokens_per_second);
const steps = good.map((r) => r.result.decode_steps);
const open = good.map((r) => r.result.open_ms);
const doc = {
  url, reps: REPS, started_at: new Date(t00).toISOString(), elapsed_ms: Date.now() - t00,
  runs,
  summary: {
    ok: good.length, failed: runs.length - good.length,
    decode_tok_s: { values: tps, median: tps.length ? +med(tps).toFixed(3) : null, min: tps.length ? Math.min(...tps) : null, max: tps.length ? Math.max(...tps) : null },
    decode_steps: { values: steps, median: steps.length ? med(steps) : null },
    open_ms: { median: open.length ? med(open) : null },
    unique_text_sha: [...new Set(good.map((r) => r.text_sha256))],
    unique_ids_sha: [...new Set(good.map((r) => r.ids_sha256))],
    identical_text: new Set(good.map((r) => r.text_sha256)).size === 1 && good.length === runs.length,
    identical_ids: new Set(good.map((r) => r.ids_sha256)).size === 1 && good.length === runs.length,
    text_sha256: good[0]?.text_sha256 ?? null, ids_sha256: good[0]?.ids_sha256 ?? null,
    text: good[0]?.result?.text ?? null, adapter: good[0]?.result?.adapter ?? null,
    health_after: health, orphans: 0,
  },
};
fs.writeFileSync(outPath, JSON.stringify(doc, null, 1));
console.log("SUMMARY " + JSON.stringify(doc.summary).slice(0, 1200));
