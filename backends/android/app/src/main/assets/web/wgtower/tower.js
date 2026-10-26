// tower.js: the Qwen3.5 vision tower (mmproj F16, clip qwen3vl_merger) on WebGPU. The host half of k_tower.wgsl:
//   const T = await tower_open(g, read)              // read(offset, length) -> Uint8Array over the mmproj .gguf
//   const r = await T.encode(planar, nx, ny, opts)   // planar [3][ny][nx] f32 (vpre.js) -> r.embd: GPUBuffer [T][PD] f32
// Every weight is copied up as its GGUF bytes (F16 stays F16). One command buffer per image; opts.dump = Set of stage names
// (vit.c's / llama.cpp's node names) read back for the kcheck gates. The host computes only index tables (bilinear taps,
// RoPE cos/sin) with f32 rounding emulated, as vit.c does.
import { gguf_read } from "../engine/gguf.js";

const U = () => GPUBufferUsage;
const fr = Math.fround;

export async function tower_open(g, read, { progress } = {}) {
  const d = g.device, gg = await gguf_read(read), kv = gg.kv, by = new Map(gg.tensors.map((t) => [t.name, t]));
  if (kv["clip.projector_type"] !== "qwen3vl_merger") throw new Error("tower: projector " + kv["clip.projector_type"] + " is not qwen3vl_merger");
  const h = { E: kv["clip.vision.embedding_length"], H: kv["clip.vision.attention.head_count"], F: kv["clip.vision.feed_forward_length"],
    L: kv["clip.vision.block_count"], P: kv["clip.vision.patch_size"], M: kv["clip.vision.spatial_merge_size"], PD: kv["clip.vision.projection_dim"],
    G: kv["clip.vision.image_size"] / kv["clip.vision.patch_size"], eps: kv["clip.vision.attention.layer_norm_epsilon"] ?? 1e-6 };
  h.D = h.E / h.H;
  if (h.P !== 16 || h.M !== 2 || h.D !== 64 || by.has("v.deepstack.0.fc1.weight")) throw new Error("tower: geometry outside the kernels (patch 16, merge 2, head dim 64, no deepstack)");
  const W = {}; let bytes = 0;
  const want = ["v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias", "v.position_embd.weight", "v.post_ln.weight", "v.post_ln.bias",
    "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"];
  for (let i = 0; i < h.L; i++) for (const s of ["ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
    "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"]) want.push(`v.blk.${i}.${s}`);
  for (const name of want) {
    const t = by.get(name);
    if (!t) throw new Error("tower: missing " + name);
    if (t.type !== (/\.weight(\.1)?$/.test(name) && !/ln|position/.test(name) ? "F16" : "F32")) throw new Error(`tower: ${name} is ${t.type}`);
    const data = await read(t.abs, t.bytes);
    const b = d.createBuffer({ label: "tw:" + name, size: (t.bytes + 3) & ~3, usage: U().STORAGE | U().COPY_DST });
    d.queue.writeBuffer(b, 0, data, 0, t.bytes & ~3);
    W[name] = b; bytes += t.bytes;
    progress?.({ phase: "tower-weights", name, bytes });
  }
  const code = await (await fetch(new URL("./k_tower.wgsl", import.meta.url))).text();
  const module = d.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error("WGSL k_tower: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
  const pipe = {};
  await Promise.all(["gemm", "lnorm", "im2col", "posadd", "addb", "rope", "attn"].map(async (e) => {
    pipe[e] = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: e } });
  }));
  const buf = (words, label) => d.createBuffer({ label, size: Math.max(16, words * 4), usage: U().STORAGE | U().COPY_SRC | U().COPY_DST });
  const up = (arr, label) => { const b = d.createBuffer({ label, size: Math.max(16, (arr.byteLength + 3) & ~3), usage: U().STORAGE | U().COPY_DST | U().COPY_SRC }); d.queue.writeBuffer(b, 0, arr); return b; };

  // one recorded op: entry, uniform words, bindings {slot: GPUBuffer}, grid
  function rec(st, e, p, binds, gx, gy = 1) {
    const ub = d.createBuffer({ size: 32, usage: U().UNIFORM | U().COPY_DST });
    d.queue.writeBuffer(ub, 0, new Uint32Array([...p, 0, 0, 0, 0, 0, 0, 0, 0].slice(0, 8)));
    st.tmp.push(ub);
    const entries = [{ binding: 0, resource: { buffer: ub } }, ...Object.entries(binds).map(([k, b]) => ({ binding: Number(k), resource: { buffer: b } }))];
    const pass = st.enc.beginComputePass();
    pass.setPipeline(pipe[e]);
    pass.setBindGroup(0, d.createBindGroup({ layout: pipe[e].getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    st.dispatches++;
  }
  const grid1 = (n) => { const w = Math.ceil(n / 256); return w <= 65535 ? [w, 1] : [65535, Math.ceil(w / 65535)]; };
  const epsBits = new Uint32Array(new Float32Array([h.eps]).buffer)[0];
  // the kernel vocabulary, exposed for the per-kernel gates (st = {enc, tmp, dispatches})
  const ops = {
    gemm: (st, x, w, b, y, n, k, o, f) => { if (k % 16 || o % 64) throw new Error("gemm shape"); rec(st, "gemm", [n, k, o, f], { 1: x, 2: w, 3: b ?? x, 4: y }, o / 64, Math.ceil(n / 64)); },
    lnorm: (st, x, w, b, y, n, k) => rec(st, "lnorm", [n, k, 0, 0, epsBits], { 1: x, 2: w, 3: b, 4: y }, n),
    im2col: (st, planar, rows, N, nx, ny, f) => rec(st, "im2col", [N, 0, 0, f, nx, ny, nx / 16], { 1: planar, 4: rows }, ...grid1(N * 768)),
    posadd: (st, tab, taps, x, N, px) => rec(st, "posadd", [N, h.E, 0, 0, h.G, 0, px], { 1: tab, 2: taps, 4: x }, ...grid1(N * h.E)),
    addb: (st, b, y, n, k) => rec(st, "addb", [n, k], { 3: b, 4: y }, ...grid1(n * k)),
    rope: (st, cs, sn, qkv, n, off) => rec(st, "rope", [n, h.E, 0, 0, h.H, h.D, off], { 1: cs, 2: sn, 4: qkv }, ...grid1(n * h.H * h.D / 2)),
    attn: (st, qkv, out, n, f) => rec(st, "attn", [n, h.E, 0, f], { 1: qkv, 4: out }, Math.ceil(n / 16), h.H),
  };

  // vit.c pos_resize's taps (ggml_interpolate BILINEAR|ALIGN_CORNERS), f32 arithmetic emulated
  function taps(px, py) {
    const G = h.G, out = new Uint32Array(3 * (px + py)), f = new Float32Array(1), fu = new Uint32Array(f.buffer);
    const axis = (n, at) => {
      const sf = n > 1 && G > 1 ? fr((n - 1) / (G - 1)) : fr(n / G);
      for (let x = 0; x < n; x++) {
        const v = fr(x / sf); let i0 = Math.floor(v); i0 = Math.min(Math.max(i0, 0), G - 1);
        const i1 = Math.min(i0 + 1, G - 1); let fd = fr(v - i0); fd = fd < 0 ? 0 : fd > 1 ? 1 : fd;
        f[0] = fd; out.set([i0, i1, fu[0]], at + 3 * x);
      }
    };
    axis(px, 0); axis(py, 3 * px);
    return out;
  }
  // vit.c rope_tables: theta = pos * 10000^(2 * (-1/nd) * p), cos/sin rounded to f32
  function ropeTables(px, N) {
    const nd = h.D / 2, s0 = h.D / 4, cs = new Float32Array(N * nd), sn = new Float32Array(N * nd), inv = fr(-1 / nd);
    for (let r = 0; r < N; r++) {
      const b = r >> 2, q = r & 3, half = px / 2, y = Math.floor(b / half) * 2 + (q >> 1), x = (b % half) * 2 + (q & 1);
      for (let ic = 0; ic < nd; ic++) {
        const p = ic < s0 ? ic : ic - s0, base = ic < s0 ? y : x;
        const th = fr(base * fr(Math.pow(10000, fr(fr(2 * inv) * p))));
        cs[r * nd + ic] = Math.cos(th); sn[r * nd + ic] = Math.sin(th);
      }
    }
    return { cs, sn };
  }

  async function encode(planar, nx, ny, { dump = new Set(), f16act = true, layers = h.L } = {}) {
    const { E, H, F, L, PD } = h;
    if (nx % 32 || ny % 32) throw new Error(`tower: image ${nx}x${ny} is not a multiple of 32`);
    const px = nx / 16, py = ny / 16, N = px * py, T = N / 4, f16 = f16act ? 1 : 0;
    const st = { enc: d.createCommandEncoder(), tmp: [], dispatches: 0 }, reads = [];
    const B = { planar: planar instanceof GPUBuffer ? planar : up(planar, "tw:planar"), rows: buf(N * 768, "tw:rows"), x: buf(N * E, "tw:x"), cur: buf(N * E, "tw:cur"),
      qkv: buf(N * 3 * E, "tw:qkv"), t2: buf(N * E, "tw:t2"), up: buf(N * Math.max(F, 4 * E), "tw:up"), embd: buf(T * PD, "tw:embd") };
    const own = Object.entries(B).filter(([k]) => k !== "embd" && !(k === "planar" && planar instanceof GPUBuffer)).map(([, b]) => b);
    const tp = up(taps(px, py), "tw:taps"), rt = ropeTables(px, N), cs = up(rt.cs, "tw:cs"), sn = up(rt.sn, "tw:sn");
    own.push(tp, cs, sn);
    const snap = (name, b, words) => {
      if (!dump.has(name)) return;
      const rb = d.createBuffer({ size: words * 4, usage: U().MAP_READ | U().COPY_DST });
      st.enc.copyBufferToBuffer(b, 0, rb, 0, words * 4); reads.push([name, rb, words]);
    };
    const qkvPart = (name, off) => {   // Q or K rows out of qkv [N][3E] (the dump layout [N][H][D])
      if (!dump.has(name)) return;
      const rb = d.createBuffer({ size: N * E * 4, usage: U().MAP_READ | U().COPY_DST });
      for (let r = 0; r < N; r++) st.enc.copyBufferToBuffer(B.qkv, (r * 3 * E + off) * 4, rb, r * E * 4, E * 4);
      reads.push([name, rb, N * E]);
    };
    ops.im2col(st, B.planar, B.rows, N, nx, ny, f16);
    ops.gemm(st, B.rows, W["v.patch_embd.weight"], null, B.x, N, 768, E, 0);
    ops.gemm(st, B.rows, W["v.patch_embd.weight.1"], null, B.x, N, 768, E, 8);
    ops.addb(st, W["v.patch_embd.bias"], B.x, N, E);
    snap("patch_bias", B.x, N * E);
    ops.posadd(st, W["v.position_embd.weight"], tp, B.x, N, px);
    snap("inp_pos_emb", B.x, N * E);
    for (let il = 0; il < layers; il++) {
      const w = (s) => W[`v.blk.${il}.${s}`];
      ops.lnorm(st, B.x, w("ln1.weight"), w("ln1.bias"), B.cur, N, E);
      snap(`ln1-${il}`, B.cur, N * E);
      ops.gemm(st, B.cur, w("attn_qkv.weight"), w("attn_qkv.bias"), B.qkv, N, E, 3 * E, f16 | 2);
      ops.rope(st, cs, sn, B.qkv, N, 0);
      ops.rope(st, cs, sn, B.qkv, N, E);
      qkvPart(`Qcur_rope-${il}`, 0); qkvPart(`Kcur_rope-${il}`, E);
      ops.attn(st, B.qkv, B.t2, N, f16);
      snap(`kqv_out-${il}`, B.t2, N * E);
      if (dump.has(`attn_out-${il}`)) { ops.gemm(st, B.t2, w("attn_out.weight"), w("attn_out.bias"), B.cur, N, E, E, f16 | 2); snap(`attn_out-${il}`, B.cur, N * E); }
      ops.gemm(st, B.t2, w("attn_out.weight"), w("attn_out.bias"), B.x, N, E, E, f16 | 2 | 8);   // x += o . W + b
      snap(`ffn_inp-${il}`, B.x, N * E);
      ops.lnorm(st, B.x, w("ln2.weight"), w("ln2.bias"), B.cur, N, E);
      snap(`ffn_inp_normed-${il}`, B.cur, N * E);
      if (dump.has(`ffn_up_b-${il}`)) { ops.gemm(st, B.cur, w("ffn_up.weight"), w("ffn_up.bias"), B.up, N, E, F, f16 | 2); snap(`ffn_up_b-${il}`, B.up, N * F); }
      ops.gemm(st, B.cur, w("ffn_up.weight"), w("ffn_up.bias"), B.up, N, E, F, f16 | 2 | 4);
      snap(`ffn_gelu-${il}`, B.up, N * F);
      if (dump.has(`ffn_out-${il}`)) { ops.gemm(st, B.up, w("ffn_down.weight"), w("ffn_down.bias"), B.cur, N, F, E, f16 | 2); snap(`ffn_out-${il}`, B.cur, N * E); }
      ops.gemm(st, B.up, w("ffn_down.weight"), w("ffn_down.bias"), B.x, N, F, E, f16 | 2 | 8);
      snap(`layer_out-${il}`, B.x, N * E);
    }
    ops.lnorm(st, B.x, W["v.post_ln.weight"], W["v.post_ln.bias"], B.cur, N, E);
    snap(`norm_b-${L}`, B.cur, N * E);
    if (dump.has("ffn_up_b")) { ops.gemm(st, B.cur, W["mm.0.weight"], W["mm.0.bias"], B.up, T, 4 * E, 4 * E, f16 | 2); snap("ffn_up_b", B.up, T * 4 * E); }
    ops.gemm(st, B.cur, W["mm.0.weight"], W["mm.0.bias"], B.up, T, 4 * E, 4 * E, f16 | 2 | 4);
    snap("ffn_gelu", B.up, T * 4 * E);
    ops.gemm(st, B.up, W["mm.2.weight"], W["mm.2.bias"], B.embd, T, 4 * E, PD, f16 | 2);
    snap("embd", B.embd, T * PD);
    const t0 = performance.now();
    d.queue.submit([st.enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    const dumps = {};
    for (const [name, rb, words] of reads) { await rb.mapAsync(GPUMapMode.READ); dumps[name] = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); }
    for (const b of [...own, ...st.tmp]) b.destroy();
    return { embd: B.embd, T, N, nx_t: nx / 32, ny_t: ny / 32, ms, dispatches: st.dispatches, dumps };
  }
  return { h, W, ops, pipe, bytes, encode, taps, ropeTables, buf, up, newState: () => ({ enc: d.createCommandEncoder(), tmp: [], dispatches: 0 }) };
}
