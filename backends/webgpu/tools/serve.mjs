#!/usr/bin/env node
// serve.mjs [PORT]: static server for the WebGPU battery pages (byte ranges; COOP/COEP on /senses/).
//   /trunk/   the WGSL trunk engine and its pages (smoke.html = chat, clef.html = decide)
//   /engine/  engine deps (tok.js, tok.wasm, gguf.js, ggufpack.js, q35_*.js)
//   /emit/    the WGSL kernels used by the shipped engine
//   /senses/  see / hear / speak pages (detect.html, ears.html, speak.html, battery.html); senses/vendor/ from scripts/senses/vendor.sh
//   /models/  MODELS_DIR (default ../models): clef-flash-Q4_K_S.gguf, qwen35-tokenizer.bin, det-web/dfine_n.onnx, hf/<repo>/
//   /cases/   CASES_DIR (default ../cases): decide request batteries for clef.html?cases=/cases/<file>.json
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const top = path.join(here, "..");
const port = Number(process.argv[2] ?? 8802), host = "127.0.0.1";
const roots = [
  ["/models/", process.env.MODELS_DIR ?? path.join(top, "models")],
  ["/cases/", process.env.CASES_DIR ?? path.join(top, "cases")],
  ["/senses/", path.join(top, "senses")],
  ["/trunk/", path.join(top, "trunk")],
  ["/engine/", path.join(top, "engine")],
  ["/emit/", path.join(top, "emit")],
];
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".wgsl": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
  ".bin": "application/octet-stream", ".gguf": "application/octet-stream", ".onnx": "application/octet-stream",
  ".woff2": "font/woff2", ".wav": "audio/wav", ".txt": "text/plain; charset=utf-8" };

http.createServer((q, s) => {
  if (q.method !== "GET" && q.method !== "HEAD") { s.statusCode = 405; return s.end(); }
  const rel = decodeURIComponent(new URL(q.url, "http://x").pathname);
  const hit = roots.find(([p]) => rel.startsWith(p));
  if (!hit) { s.statusCode = 404; return s.end("not found"); }
  const [prefix, root] = hit;
  let file = path.resolve(root, "." + rel.slice(prefix.length - 1));
  if (!file.startsWith(path.resolve(root))) { s.statusCode = 403; return s.end(); }
  let st; try { st = fs.statSync(file); if (st.isDirectory()) { file = path.join(file, "index.html"); st = fs.statSync(file); } }
  catch { s.statusCode = 404; return s.end("not found: " + rel); }
  const size = st.size, m = /^bytes=(\d*)-(\d*)$/.exec(q.headers.range ?? "");
  let start = 0, end = size - 1;
  if (m) { start = m[1] === "" ? size - Number(m[2]) : Number(m[1]); end = m[1] !== "" && m[2] !== "" ? Math.min(Number(m[2]), size - 1) : size - 1; }
  if (m && (start > end || start >= size)) { s.statusCode = 416; s.setHeader("content-range", `bytes */${size}`); return s.end(); }
  s.setHeader("content-type", types[path.extname(file).toLowerCase()] ?? "application/octet-stream");
  s.setHeader("accept-ranges", "bytes");
  s.setHeader("content-length", end - start + 1);
  s.setHeader("cache-control", "no-cache");
  if (rel.startsWith("/senses/")) { s.setHeader("cross-origin-opener-policy", "same-origin"); s.setHeader("cross-origin-embedder-policy", "require-corp"); }
  if (m) { s.statusCode = 206; s.setHeader("content-range", `bytes ${start}-${end}/${size}`); }
  if (q.method === "HEAD") return s.end();
  fs.createReadStream(file, { start, end }).pipe(s);
}).listen(port, host, () => console.log(`webgpu: http://${host}:${port}/`));
