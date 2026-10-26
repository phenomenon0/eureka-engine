// drive-turn.js: in-page driver for wkshot (WebKit) — the e2e.mjs flow without Playwright. Reads window.__drive = {door, prompt, follow}.
(async () => {
  const cfg = Object.assign({ door: "instant", prompt: "In two sentences: why does running a language model locally in the browser matter?" }, window.__drive || {});
  const shots = (window.__shots = window.__shots || []);
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const until = async (fn, ms = 600000) => { const t = Date.now(); for (;;) { const v = fn(); if (v) return v; if (Date.now() - t > ms) throw new Error("timeout"); await sleep(200); } };
  const t0 = performance.now();
  try {
    await sleep(1200);
    const fresh = document.querySelector("button[aria-label='New chat']");   // receipts start from a new chat
    if (fresh && !fresh.disabled) { fresh.click(); await sleep(800); }
    shots.push("1-open"); await sleep(400);
    const send = async (text) => {
      const box = await until(() => document.querySelector("textarea"));
      const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
      box.focus(); set.call(box, text); box.dispatchEvent(new Event("input", { bubbles: true }));
      await sleep(100);
      box.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true }));
    };
    const meter = () => document.querySelector("button[aria-label^='Model:']");
    await send(cfg.prompt);
    const dialog = await until(() => document.querySelector("[role=dialog][aria-label='Choose a model']"), 20000);
    await sleep(700); shots.push("2-doors"); await sleep(400);
    const btn = [...dialog.querySelectorAll("button")].find(b => cfg.door === "gguf-url" ? /by URL/.test(b.textContent) : /Instant demo/.test(b.textContent));
    const tClick = performance.now();
    btn.click();
    await sleep(1500); shots.push("3-loading");
    await until(() => !document.querySelector("[role=dialog][aria-label='Choose a model']") || document.querySelector("[data-phase=error]"));
    const tLive = performance.now();
    await sleep(1200); shots.push("4-streaming");
    await until(() => meter() && /tok\/s/.test(meter().textContent) && !meter().hasAttribute("data-live"), 300000);
    await sleep(800); shots.push("5-answer"); await sleep(500);
    const s = window.__bendTab.state;
    const out = { ua: navigator.userAgent, door: cfg.door, meter: meter().textContent, live_s: (tLive - tClick) / 1000, send_to_done_s: (performance.now() - tClick) / 1000,
      model: s.model && s.model.name, openMs: s.openMs, facts: s.facts, last: s.last };
    if (cfg.follow) {
      await send(cfg.follow); await sleep(1500);
      await until(() => !meter().hasAttribute("data-live"), 300000); await sleep(800); shots.push("6-follow"); await sleep(500);
      out.follow = window.__bendTab.state.last;
    }
    window.__done = out;
  } catch (e) { shots.push("error"); await sleep(500); window.__done = { error: String(e && e.message || e), state: window.__bendTab && window.__bendTab.state && { phase: window.__bendTab.state.phase, error: window.__bendTab.state.error } }; }
})();
