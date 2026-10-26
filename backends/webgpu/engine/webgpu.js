// bendgpu.js: the WebGPU host runtime (WEBEMIT, the fifth runtime's skeleton).
//
// Metal's host runtime (comp.ts, Gpu: gpu_probe / gpu_load / gpu_kernel / gpu_pass) is the blueprint:
//   probe   -> gpu_open: adapter + device, the limits the lane asks for, the adapter's facts kept
//   load    -> gpu_chunks: a host file as chunk buffers, each <= the binding limit and whole rows
//              (WebGPU binds at most maxStorageBufferBindingSize bytes; a 4B's weights are several)
//   kernel  -> cube_pipe + cube_run: one emitted .wgsl module, one dispatch per chunk, a workgroup a row
//   pass    -> one command buffer for the whole run, one map of the result (Metal's m30_wait)
// Device memory is not the host's: unlike Metal's newBufferWithBytesNoCopy, every byte is copied up.

export async function gpu_open(want = {}) {
  if (!("gpu" in navigator)) throw new Error("no navigator.gpu: this browser has no WebGPU");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const lim = adapter.limits, req = {};
  for (const [k, v] of Object.entries(want)) req[k] = Math.min(v, lim[k]);
  const device = await adapter.requestDevice({ requiredLimits: req });
  const info = adapter.info ?? {};
  device.lost.then((l) => console.error("bendgpu: device lost:", l.message));
  return {
    device, adapter,
    facts: {
      vendor: info.vendor ?? "", architecture: info.architecture ?? "", description: info.description ?? "",
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
    },
  };
}

// gpu_chunks: bytes (an ArrayBuffer, or a ReadableStream of Uint8Array) as storage buffers of at most
// `chunk` bytes, each a whole number of `row` bytes. A stream is copied up as it arrives (a staged
// window per chunk), so the host never holds the whole file: the chunked load of AD3.
export async function gpu_chunks(g, src, total, row, chunk) {
  const lim = Math.min(chunk, g.device.limits.maxStorageBufferBindingSize, g.device.limits.maxBufferSize);
  const per = Math.floor(lim / row) * row;
  if (per === 0) throw new Error(`gpu_chunks: a row of ${row} B is over the binding limit ${lim} B`);
  const out = [];
  for (let at = 0; at < total; at += per) {
    const size = Math.min(per, total - at);
    out.push({ at, size, buf: g.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) });
  }
  if (src instanceof ArrayBuffer || ArrayBuffer.isView(src)) {
    const u8 = src instanceof ArrayBuffer ? new Uint8Array(src) : new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
    for (const c of out) g.device.queue.writeBuffer(c.buf, 0, u8, c.at, c.size);
    return out;
  }
  // a stream: fill one chunk's window, write it, move on
  const reader = src.getReader();
  let ci = 0, fill = 0, win = new Uint8Array(out[0].size);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    let off = 0;
    while (off < value.length) {
      const n = Math.min(value.length - off, win.length - fill);
      win.set(value.subarray(off, off + n), fill);
      fill += n; off += n;
      if (fill === win.length) {
        g.device.queue.writeBuffer(out[ci].buf, 0, win);
        ci += 1; fill = 0;
        if (ci < out.length) win = new Uint8Array(out[ci].size);
      }
    }
  }
  if (ci !== out.length) throw new Error(`gpu_chunks: the stream ended in chunk ${ci} of ${out.length}`);
  return out;
}

// cube_pipe: an emitted lane cube (bend -o x.wgsl) as a compute pipeline; the module's compile
// messages are the first receipt a browser gives.
export async function cube_pipe(g, wgsl) {
  const module = g.device.createShaderModule({ code: wgsl });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
  const pipe = await g.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "cube_main" } });
  return { pipe, messages: info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`) };
}

// cube_run: y = the cube over every row of x (and w, chunked alike); one dispatch a chunk, one
// workgroup a row; returns y as a Float32Array and the GPU-side ms.
export async function cube_run(g, cp, xc, wc, n) {
  const d = g.device, rowB = n * 4;
  const total = xc.reduce((s, c) => s + c.size, 0);
  const ys = xc.map((c) => d.createBuffer({ size: c.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
  const read = d.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(cp.pipe);
  for (let i = 0; i < xc.length; i++) {
    if (wc[i].size !== xc[i].size) throw new Error("cube_run: x and w chunk differently");
    pass.setBindGroup(0, d.createBindGroup({ layout: cp.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: xc[i].buf } },
      { binding: 1, resource: { buffer: wc[i].buf } },
      { binding: 2, resource: { buffer: ys[i] } }] }));
    pass.dispatchWorkgroups(xc[i].size / rowB);
  }
  pass.end();
  for (let i = 0; i < xc.length; i++) enc.copyBufferToBuffer(ys[i], 0, read, xc[i].at, xc[i].size);
  const t0 = performance.now();
  d.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const ms = performance.now() - t0;
  const y = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  for (const b of ys) b.destroy();
  read.destroy();
  return { y, ms };
}

// wgsl_manifest: the first line of an emitted .wgsl (`// bend-wgsl {...}`): what the program holds
// (a cube, a fork tree) and the bang's default arguments.
export function wgsl_manifest(wgsl) {
  const m = /^\/\/ bend-wgsl (.*)$/m.exec(wgsl);
  if (!m) throw new Error("not an emitted bend .wgsl (no manifest line)");
  return JSON.parse(m[1]);
}

// entries_of: a manifest's entries; a bend-wgsl/1 file is read as its fork (and cube) entries.
export function entries_of(man) {
  if (man.entries) return man.entries;
  const es = [];
  if (man.cube) es.push({ kind: "cube", name: "", entry: "cube_main", n: man.cube.n });
  if (man.fork) es.push({ kind: "fork", name: "", leaves: "fork_leaves", join: "fork_join", joinp: false, bytes: [], ...man.fork });
  return es;
}

const st = () => GPUShaderStage.COMPUTE;
const bgl_of = (d, spec) => d.createBindGroupLayout({ entries: spec.map(([binding, type]) =>
  ({ binding, visibility: st(), buffer: { type } })) });

// pipes: one module's pipelines, an explicit bind group layout an entry (the manifest lists every
// binding, so no entry has to use them all), all compiled in parallel (a page-load cost).
export async function pipes(g, wgsl) {
  const d = g.device, man = wgsl_manifest(wgsl);
  const module = d.createShaderModule({ code: wgsl });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
  const out = { module, man, entries: [], messages: info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`) };
  const mk = (bgl, entryPoint) => d.createComputePipelineAsync({
    layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint } });
  await Promise.all(entries_of(man).map(async (e) => {
    const pe = { man: e };
    if (e.kind === "fork") {
      pe.bgl = bgl_of(d, [[0, "read-only-storage"], [1, "storage"], [2, "storage"], [3, "read-only-storage"],
        [4, "storage"], [5, "read-only-storage"], ...e.bytes.map((b) => [b.binding, "read-only-storage"])]);
      [pe.leaves, pe.join] = await Promise.all([mk(pe.bgl, e.leaves), mk(pe.bgl, e.join)]);
    } else if (e.kind === "cube" || e.kind === "map") {
      pe.bgl = bgl_of(d, [...(e.ins ?? [{ binding: 0 }, { binding: 1 }]).map((i) => [i.binding, "read-only-storage"]),
        [(e.out ?? { binding: 2 }).binding, "storage"], ...(e.out2 ? [[e.out2.binding, "storage"]] : []), ...(e.sc ? [[e.sc.binding, "read-only-storage"]] : [])]);
      pe.pipe = await mk(pe.bgl, e.entry);
    }
    out.entries.push(pe);
  }));
  out.by = (kind, name = "") => out.entries.find((pe) => pe.man.kind === kind && pe.man.name === name);
  return out;
}

// writeBuffer takes whole words: a Bytes goes up zero-padded (its length travels as a word)
const up4 = (v) => { const u = new Uint8Array(v.buffer, v.byteOffset, v.byteLength); if (u.length % 4 === 0) return u; const p = new Uint8Array((u.length + 3) & ~3); p.set(u); return p; };
export const grid = (w, MAXG = 65535) => [Math.min(Math.max(w, 1), MAXG), Math.ceil(Math.max(w, 1) / MAXG)];
export const u32bits = (t, v) => t === "f32" ? new Uint32Array(new Float32Array([v]).buffer)[0] : t === "bool" ? (v ? 1 : 0) : v >>> 0;

// entry_run: a cube or a map once. io.in = {name: Float32Array | Uint32Array | Uint8Array (Bytes) | GPUBuffer};
// io.sc = {name: number} (the scalars); io.rows (cube) or io.len (map). Returns the output as a Uint32Array of
// bits (f32 by its bits) and the wall ms of submit + map.
export async function entry_run(g, pe, io) {
  const d = g.device, e = pe.man, ST = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST;
  const mkbuf = (bytes, usage) => d.createBuffer({ size: Math.max(16, (bytes + 3) & ~3), usage });
  const fresh = [];
  const entries = [];
  const lens = {};
  for (const i of e.ins) {
    const v = io.in[i.name];
    if (v === undefined) throw new Error(`entry_run: no input ${i.name}`);
    let buf = v;
    if (!(v instanceof GPUBuffer)) {
      buf = mkbuf(v.byteLength, ST | CD);
      d.queue.writeBuffer(buf, 0, up4(v));
      fresh.push(buf);
    }
    if (i.cls === "bytes") lens[i.name] = v instanceof GPUBuffer ? v.size : v.byteLength;
    entries.push({ binding: i.binding, resource: { buffer: buf } });
  }
  const rows = e.kind === "cube" ? io.rows : io.len;
  const outWords = e.kind === "map" ? io.len : e.per === "elem" ? io.rows * e.len : io.rows;
  const ybuf = mkbuf(outWords * 4, ST | GPUBufferUsage.COPY_SRC);
  entries.push({ binding: e.out.binding, resource: { buffer: ybuf } });
  if (e.sc) {
    const sc = new Uint32Array(1 + e.sc.scalars.length + e.sc.lens.length + 1);
    sc[0] = rows;
    for (const s of e.sc.scalars) sc[s.at] = u32bits(s.ty, io.sc?.[s.name] ?? 0);
    for (const l of e.sc.lens) sc[l.at] = lens[l.name];
    const scb = mkbuf(sc.byteLength, ST | CD);
    d.queue.writeBuffer(scb, 0, sc);
    fresh.push(scb);
    entries.push({ binding: e.sc.binding, resource: { buffer: scb } });
  }
  const read = d.createBuffer({ size: Math.max(16, outWords * 4), usage: GPUBufferUsage.MAP_READ | CD });
  const enc = d.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pe.pipe);
  pass.setBindGroup(0, d.createBindGroup({ layout: pe.bgl, entries }));
  const [gx, gy] = e.kind === "cube" ? grid(io.rows) : grid(Math.ceil(io.len / e.lanes));
  for (let k = 0; k < (io.repeat ?? 1); k++) pass.dispatchWorkgroups(gx, gy);   // repeat: only to time a kernel past the submit latency
  pass.end();
  enc.copyBufferToBuffer(ybuf, 0, read, 0, outWords * 4);
  const t0 = performance.now();
  d.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const ms = performance.now() - t0;
  const bits = new Uint32Array(read.getMappedRange().slice(0, outWords * 4));
  read.unmap();
  for (const b of [...fresh, read, ...io.keep ? [] : [ybuf]]) b.destroy();
  return { bits, ms, buffer: io.keep ? ybuf : undefined };
}

// fork_run: a bang's fork tree, f(d, xs): 2^d lanes in fork_leaves (a workgroup's 256 joined to one
// partial), then fork_join passes over the partials until one is left. The tree is the program's own:
// lane L's bit p is its choice at level p. Words go up and come back as u32 (f32 by its bits); a Bytes
// parameter takes its Uint8Array in `bufs` (by parameter index) and its byte length as the word.
// fork_err set = a Nat went past 2^32 - 1 (the lane's envelope), so no value is returned.
// pe: a pipes() result's fork entry (or, for a bend-wgsl/1 file, the wgsl text, compiled here).
export async function fork_run(g, wgsl_or_pipes, args, bufs = {}, name = "") {
  const ps = typeof wgsl_or_pipes === "string" ? await pipes(g, wgsl_or_pipes) : wgsl_or_pipes;
  const pe = ps.by("fork", name);
  if (!pe) throw new Error("this .wgsl holds no fork tree" + (name ? ` named ${name}` : ""));
  const man = pe.man;
  const d = g.device, T = 256;
  const depth = args[0];
  if (depth > 31) throw new Error(`fork_run: 2^${depth} lanes is past a u32 lane index`);
  const n = 2 ** depth;
  const words = new Uint32Array([depth, ...args.slice(1).map((v, i) =>
    man.params[i + 1] === "bytes" ? bufs[i + 1].byteLength : u32bits(man.params[i + 1], v)), n]);
  const st = GPUBufferUsage.STORAGE, cp = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = (size, usage) => d.createBuffer({ size: Math.max(16, (size + 3) & ~3), usage });
  const argb = buf(words.byteLength, st | cp);
  d.queue.writeBuffer(argb, 0, words);
  const err = buf(4, st | cp);
  d.queue.writeBuffer(err, 0, new Uint32Array([0]));
  const bys = man.bytes.map((b) => {
    const u = bufs[b.param], bb = buf(u.byteLength, st | cp);
    d.queue.writeBuffer(bb, 0, up4(u));
    return { binding: b.binding, resource: { buffer: bb } };
  });
  let w = Math.ceil(n / T), [gx, gy] = grid(w);
  let part = buf(gx * gy * 4, st | cp);
  const none = () => buf(4, st | cp);
  const bg = (entries) => d.createBindGroup({ layout: pe.bgl, entries: [...entries, ...bys] });
  const enc = d.createCommandEncoder();
  let pass = enc.beginComputePass();
  pass.setPipeline(pe.leaves);
  pass.setBindGroup(0, bg([{ binding: 0, resource: { buffer: argb } }, { binding: 1, resource: { buffer: part } },
    { binding: 2, resource: { buffer: err } }, { binding: 3, resource: { buffer: none() } },
    { binding: 4, resource: { buffer: none() } }, { binding: 5, resource: { buffer: none() } }]));
  pass.dispatchWorkgroups(gx, gy);
  pass.end();
  let passes = 1, lb = 0;
  for (let c = w; c > 1; c = w) {
    w = Math.ceil(c / T); [gx, gy] = grid(w);
    lb += 8;
    const out = buf(gx * gy * 4, st | cp), cb = buf(8, st | cp);
    d.queue.writeBuffer(cb, 0, new Uint32Array([c, lb]));
    pass = enc.beginComputePass();
    pass.setPipeline(pe.join);
    pass.setBindGroup(0, bg([{ binding: 0, resource: { buffer: argb } }, { binding: 1, resource: { buffer: none() } },
      { binding: 2, resource: { buffer: err } }, { binding: 3, resource: { buffer: part } },
      { binding: 4, resource: { buffer: out } }, { binding: 5, resource: { buffer: cb } }]));
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    part = out; passes += 1;
  }
  const read = buf(8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  enc.copyBufferToBuffer(part, 0, read, 0, 4);
  enc.copyBufferToBuffer(err, 0, read, 4, 4);
  const t0 = performance.now();
  d.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const ms = performance.now() - t0;
  const [word, bad] = new Uint32Array(read.getMappedRange().slice(0));
  read.unmap();
  const value = bad ? null : man.result === "f32" ? new Float32Array(new Uint32Array([word]).buffer)[0]
    : man.result === "bool" ? word !== 0 : word;
  return { value, bits: bad ? null : word, envelope: bad ? "a Nat went past 2^32 - 1: run it on the C or JS lane" : null, lanes: n, passes, ms };
}
