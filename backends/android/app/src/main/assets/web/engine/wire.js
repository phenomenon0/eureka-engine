// wire.js: the pure parts of the in-tab engine (UMD-lite: a classic script in the workers, require()d by node tests).
(function (g) {
  const IM_START = "<|im_start|>", IM_END = "<|im_end|>";
  // the Qwen3.5 chat template as engine-4b's p1/p2 prompts render it: an optional system turn, the history, then the
  // generation prompt with an empty think block
  function renderChat(system, messages) {
    let s = "";
    if (system) s += `${IM_START}system\n${system}${IM_END}\n`;
    for (const m of messages) s += `${IM_START}${m.role}\n${m.content}${IM_END}\n`;
    return s + `${IM_START}assistant\n<think>\n\n</think>\n\n`;
  }
  // the engine's argv: its own --threads, the pack dir, the prompt ids, --steps, --stop
  function engineArgs({ dir = "/pack", ids, maxNew, stop, threads = 8 }) {
    return ["--threads", String(threads), "--", dir, ...ids.map(String), "--steps", String(maxNew), ...(stop == null ? [] : ["--stop", String(stop)])];
  }
  // stdout lines: one id a line, then `wall_ms N`
  function parseOut(line) {
    if (/^\d+$/.test(line)) return { id: Number(line) };
    const m = /^wall_ms (\d+)$/.exec(line);
    return m ? { wall_ms: Number(m[1]) } : {};
  }
  // bytes of successive token pieces -> text, holding back an incomplete UTF-8 tail
  class Utf8Stream {
    constructor() { this.d = new TextDecoder("utf-8"); }
    push(bytes) { return this.d.decode(bytes, { stream: true }); }
    end() { return this.d.decode(); }
  }
  // the first n bytes of the prompt kept apart from what is generated: strip a leading space only on the first piece
  const api = { renderChat, engineArgs, parseOut, Utf8Stream, IM_END_ID: 248046 };
  g.BendWire = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
