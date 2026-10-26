// Recorded WebGPU dispatch graphs. Engine-neutral companion to webgpu.js.
import { pipes, grid, u32bits } from "./webgpu.js";

// A graph contains names and sizes, never GPU objects. It can be serialized once and opened on a
// new device. Buffers marked external are supplied by the caller (for example weight chunks).
// Other buffers are owned by graph_open and retained across token replays. A run may write inputs
// before encoding; subsequent dispatches consume earlier outputs in the same command buffer.
export function graph_record() {
  const spec = { lane: "bend-graph/1", buffers: {}, steps: [], result: null };
  const name = (s) => typeof s === "string" && /^[A-Za-z][A-Za-z0-9_.:-]*$/.test(s);
  const check = (yes, why) => { if (!yes) throw new Error("graph_record: " + why); };
  const api = {
    buffer(id, bytes, options = {}) {
      check(name(id) && !Object.hasOwn(spec.buffers, id), "duplicate or invalid buffer " + id);
      check(Number.isSafeInteger(bytes) && bytes > 0 && bytes % 4 === 0, "buffer bytes must be positive and word aligned");
      spec.buffers[id] = { bytes, external: options.external === true, persist: options.persist === true };
      return api;
    },
    dispatch(step) {
      check(step && name(step.module) && ["cube", "map"].includes(step.kind), "invalid dispatch");
      check(name(step.out) && Object.hasOwn(spec.buffers, step.out), "unknown output " + step.out);
      check(step.in && typeof step.in === "object" && !Array.isArray(step.in), "inputs required");
      for (const id of Object.values(step.in)) check(name(id) && Object.hasOwn(spec.buffers, id), "unknown input " + id);
      check(!Object.values(step.in).includes(step.out), "an output may not alias an input binding");
      const count = step.kind === "cube" ? step.rows : step.len;
      check(Number.isSafeInteger(count) && count > 0, "dispatch count must be fixed and positive");
      check(step.sc === undefined || (step.sc && typeof step.sc === "object" && !Array.isArray(step.sc)), "invalid scalars");
      spec.steps.push({ module: step.module, kind: step.kind, name: step.name ?? "", in: { ...step.in },
        out: step.out, ...(step.kind === "cube" ? { rows: count } : { len: count }), sc: { ...(step.sc ?? {}) } });
      return api;
    },
    copy(from, to, bytes, fromOffset = 0, toOffset = 0) {
      check(name(from) && name(to) && from !== to && spec.buffers[from] && spec.buffers[to], "invalid copy buffers");
      check(Number.isSafeInteger(bytes) && bytes > 0 && bytes % 4 === 0
        && Number.isSafeInteger(fromOffset) && fromOffset >= 0 && fromOffset % 4 === 0
        && Number.isSafeInteger(toOffset) && toOffset >= 0 && toOffset % 4 === 0
        && fromOffset + bytes <= spec.buffers[from].bytes && toOffset + bytes <= spec.buffers[to].bytes,
      "copy range outside buffer");
      spec.steps.push({ kind: "copy", from, to, bytes, fromOffset, toOffset });
      return api;
    },
    read(id, bytes = spec.buffers[id]?.bytes) {
      check(name(id) && Object.hasOwn(spec.buffers, id), "unknown result " + id);
      check(Number.isSafeInteger(bytes) && bytes > 0 && bytes <= spec.buffers[id].bytes && bytes % 4 === 0,
        "invalid result byte length");
      spec.result = { buffer: id, bytes };
      return api;
    },
    serialize() {
      check(spec.steps.some((s) => s.kind !== "copy") && spec.result, "dispatch and result required");
      return JSON.stringify(spec);
    },
  };
  return api;
}

// Open a recorded graph against emitted modules. `sources` maps module names to WGSL text or an
// already compiled pipes() result; every entry is checked against its manifest before allocation.
// run({write, params}) accepts typed-array writes by buffer name and scalar parameters by name.
// Runs on one graph instance are deliberately serialized because they share scratch and state.
export async function graph_open(g, sources, serialized, external = {}, options = {}) {
  // maxReadBytes is for validation gates that assemble many stage outputs into one result buffer
  // with ordered copies; the engine keeps the 4096-byte default.
  const maxRead = options.maxReadBytes ?? 4096;
  const spec = typeof serialized === "string" ? JSON.parse(serialized) : structuredClone(serialized);
  if (spec.lane !== "bend-graph/1" || !spec.buffers || !Array.isArray(spec.steps) || !spec.result)
    throw new Error("graph_open: unsupported graph");
  const d = g.device, ST = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const names = [...new Set(spec.steps.filter((s) => s.kind !== "copy").map((s) => s.module))];
  const compiled = new Map(await Promise.all(names.map(async (n) => {
    const src = sources instanceof Map ? sources.get(n) : sources[n];
    if (!src) throw new Error("graph_open: missing emitted module " + n);
    return [n, typeof src === "string" ? await pipes(g, src) : src];
  })));
  for (const [id, b] of Object.entries(spec.buffers))
    if (!Number.isSafeInteger(b.bytes) || b.bytes < 4 || b.bytes % 4 || b.bytes > d.limits.maxBufferSize)
      throw new Error("graph_open: invalid buffer size " + id);
  const prepared = spec.steps.map((s, index) => {
    if (s.kind === "copy") {
      if (!spec.buffers[s.from] || !spec.buffers[s.to] || s.from === s.to || !Number.isSafeInteger(s.bytes)
        || s.bytes <= 0 || s.bytes % 4 || !Number.isSafeInteger(s.fromOffset) || s.fromOffset < 0
        || s.fromOffset % 4 || !Number.isSafeInteger(s.toOffset) || s.toOffset < 0 || s.toOffset % 4
        || s.fromOffset + s.bytes > spec.buffers[s.from].bytes || s.toOffset + s.bytes > spec.buffers[s.to].bytes)
        throw new Error(`graph_open: invalid copy at step ${index}`);
      return { s };
    }
    const pe = compiled.get(s.module)?.by(s.kind, s.name);
    if (!pe || !pe.pipe) throw new Error(`graph_open: missing entry at step ${index}`);
    const e = pe.man, count = s.kind === "cube" ? s.rows : s.len;
    if (!Number.isSafeInteger(count) || count <= 0) throw new Error(`graph_open: invalid count at step ${index}`);
    const words = s.kind === "map" ? count : e.per === "elem" ? count * e.len : count;
    if (words * 4 > spec.buffers[s.out]?.bytes) throw new Error(`graph_open: short output at step ${index}`);
    if (Object.keys(s.in).length !== e.ins.length || e.ins.some((i) => !Object.hasOwn(s.in, i.name)))
      throw new Error(`graph_open: bindings disagree at step ${index}`);
    for (const i of e.ins) if (spec.buffers[s.in[i.name]].bytes > d.limits.maxStorageBufferBindingSize)
      throw new Error(`graph_open: binding limit at step ${index}`);
    if (spec.buffers[s.out].bytes > d.limits.maxStorageBufferBindingSize)
      throw new Error(`graph_open: output binding limit at step ${index}`);
    if (!!e.out2 !== !!s.out2 || (s.out2 && (!spec.buffers[s.out2] || s.out2 === s.out || spec.buffers[s.out2].bytes > d.limits.maxStorageBufferBindingSize)))
      throw new Error(`graph_open: second output disagrees at step ${index}`);   // TURBODEFAULT: fused kernels may write two buffers (manifest out2)
    if (e.sc && e.sc.scalars.some((x) => !Object.hasOwn(s.sc, x.name)))
      throw new Error(`graph_open: missing scalar at step ${index}`);
    if (!e.sc && Object.keys(s.sc).length) throw new Error(`graph_open: unused scalars at step ${index}`);
    const tile = s.tile ?? 1;   // TURBODEFAULT: rows per workgroup of a row-tiled GEMV entry
    if (count % tile) throw new Error(`graph_open: rows ${count} not a multiple of tile ${tile} at step ${index}`);
    const dispatch = s.kind === "cube" ? grid(count / tile) : grid(Math.ceil(count / e.lanes));
    if (dispatch[1] > d.limits.maxComputeWorkgroupsPerDimension)
      throw new Error(`graph_open: dispatch grid limit at step ${index}`);
    return { s, e, pe, count, dispatch };
  });
  if (!Number.isSafeInteger(spec.result.bytes) || spec.result.bytes <= 0 || spec.result.bytes % 4
    || spec.result.bytes > maxRead) throw new Error("graph_open: invalid small readback length");
  if (!spec.buffers[spec.result.buffer] || spec.result.bytes > spec.buffers[spec.result.buffer].bytes)
    throw new Error("graph_open: invalid result buffer");
  const buffers = new Map(), owned = [];
  try {
    for (const [id, b] of Object.entries(spec.buffers)) {
      const buf = b.external ? external[id] : d.createBuffer({ size: Math.max(16, b.bytes), usage: ST | CD | CS });
      if (!(buf instanceof GPUBuffer) || buf.size < b.bytes || !(buf.usage & ST))
        throw new Error("graph_open: missing or short storage buffer " + id);
      buffers.set(id, buf);
      if (!b.external) owned.push(buf);
    }
    if (!(buffers.get(spec.result.buffer).usage & CS)) throw new Error("graph_open: result needs COPY_SRC");
    for (const { s, e } of prepared) {
      if (s.kind === "copy") {
        if (buffers.get(s.from) === buffers.get(s.to)
          || !(buffers.get(s.from).usage & CS) || !(buffers.get(s.to).usage & CD))
          throw new Error("graph_open: copy needs COPY_SRC and COPY_DST");
      } else {
        if (e.ins.some((i) => buffers.get(s.in[i.name]) === buffers.get(s.out) || (s.out2 && buffers.get(s.in[i.name]) === buffers.get(s.out2))))
          throw new Error("graph_open: input and output GPUBuffer alias");
      }
    }
  } catch (e) { for (const b of owned) b.destroy(); throw e; }
  let closed = false, busy = false, destroyed = false, idle = Promise.resolve(), finishRun = null;
  const initialized = new Set(Object.entries(spec.buffers).filter(([, b]) => b.external).map(([id]) => id));
  const release = () => {
    if (closed && !destroyed && !busy) { destroyed = true; for (const b of owned) b.destroy(); }
  };
  const invalidate = () => { closed = true; release(); return idle; };
  d.lost.then(invalidate);
  const resolve = (v, params) => {
    if (v && typeof v === "object" && typeof v.param === "string") v = params[v.param];
    if (typeof v !== "number") throw new Error("graph_open: missing scalar parameter");
    return v;
  };
  return {
    spec,
    async run({ write = {}, params = {} } = {}) {
      if (closed || busy) throw new Error("graph_open: graph closed or busy");
      busy = true;
      idle = new Promise((resolve) => { finishRun = resolve; });
      const fresh = [];
      let submitted = false;
      try {
        const ready = new Set([...initialized, ...Object.keys(write)]);
        for (const { s, e } of prepared) {
          if (s.kind === "copy") {
            if (!ready.has(s.from)) throw new Error("graph_open: copy source has no writer: " + s.from);
            ready.add(s.to);
            continue;
          }
          for (const i of e.ins) if (!ready.has(s.in[i.name]))
            throw new Error("graph_open: input has no writer: " + s.in[i.name]);
          ready.add(s.out);
          if (s.out2) ready.add(s.out2);
        }
        if (!ready.has(spec.result.buffer)) throw new Error("graph_open: result has no writer");
        // Each dispatch gets an immutable aligned slot. One upload precedes the encoder submit;
        // repeated queue.writeBuffer calls on one slot would make all dispatches see the last value.
        const scalarSteps = prepared.filter((p) => p.e?.sc);
        const alignment = d.limits.minStorageBufferOffsetAlignment;
        const scalarBytes = (p) => Math.max(16, (1 + p.e.sc.scalars.length + p.e.sc.lens.length + 1) * 4);
        const stride = scalarSteps.length ? Math.ceil(Math.max(...scalarSteps.map(scalarBytes)) / alignment) * alignment : 0;
        const scalarData = scalarSteps.length ? new Uint8Array(stride * scalarSteps.length) : null;
        const scalarOffset = new Map();
        for (const [index, p] of scalarSteps.entries()) {
          const { s, e, count } = p, offset = index * stride;
          const words = new Uint32Array(scalarData.buffer, offset, scalarBytes(p) / 4);
          words[0] = count;
          for (const x of e.sc.scalars) words[x.at] = u32bits(x.ty, resolve(s.sc[x.name], params));
          for (const x of e.sc.lens) words[x.at] = Math.max(16, spec.buffers[s.in[x.name]].bytes);
          scalarOffset.set(p, offset);
        }
        for (const [id, data] of Object.entries(write)) {
          const b = spec.buffers[id];
          if (!b || !ArrayBuffer.isView(data) || data.byteLength !== b.bytes || !(buffers.get(id).usage & CD))
            throw new Error("graph_open: invalid input write " + id);
        }
        for (const [id, data] of Object.entries(write)) d.queue.writeBuffer(buffers.get(id), 0, data);
        const scb = scalarData && d.createBuffer({ size: scalarData.byteLength, usage: ST | CD });
        if (scb) { d.queue.writeBuffer(scb, 0, scalarData); fresh.push(scb); }
        const enc = d.createCommandEncoder();
        for (const p of prepared) {
          const { s, e, pe, dispatch } = p;
          if (s.kind === "copy") {
            enc.copyBufferToBuffer(buffers.get(s.from), s.fromOffset, buffers.get(s.to), s.toOffset, s.bytes);
            continue;
          }
          const entries = e.ins.map((i) => ({ binding: i.binding, resource: { buffer: buffers.get(s.in[i.name]) } }));
          entries.push({ binding: e.out.binding, resource: { buffer: buffers.get(s.out) } });
          if (e.out2) entries.push({ binding: e.out2.binding, resource: { buffer: buffers.get(s.out2) } });
          if (e.sc) {
            entries.push({ binding: e.sc.binding, resource: {
              buffer: scb, offset: scalarOffset.get(p), size: scalarBytes(p) } });
          }
          const pass = enc.beginComputePass();
          pass.setPipeline(pe.pipe);
          pass.setBindGroup(0, d.createBindGroup({ layout: pe.bgl, entries }));
          pass.dispatchWorkgroups(...dispatch);
          pass.end();
        }
        const read = d.createBuffer({ size: Math.max(16, spec.result.bytes), usage: GPUBufferUsage.MAP_READ | CD });
        fresh.push(read);
        enc.copyBufferToBuffer(buffers.get(spec.result.buffer), 0, read, 0, spec.result.bytes);
        const t0 = performance.now();
        d.queue.submit([enc.finish()]);
        submitted = true;
        await read.mapAsync(GPUMapMode.READ);
        if (closed) throw new Error("graph_open: graph invalidated or device lost");
        const ms = performance.now() - t0;
        const bits = new Uint32Array(read.getMappedRange().slice(0, spec.result.bytes));
        read.unmap();
        for (const id of ready) if (spec.buffers[id].persist || spec.buffers[id].external) initialized.add(id);
        return { bits, ms, submits: 1, dispatches: prepared.filter((p) => p.s.kind !== "copy").length,
          copies: prepared.filter((p) => p.s.kind === "copy").length };
      } finally {
        if (submitted) await d.queue.onSubmittedWorkDone().catch(() => {});
        for (const b of fresh) b.destroy();
        busy = false;
        release();
        finishRun();
      }
    },
    invalidate,
    close: invalidate,
  };
}

// Engine-facing builder for already compiled manifest entries. This matches the step graph's
// recording vocabulary. A scalar callback reads the per-token ctx and becomes a named parameter
// in the serialized graph; callers reopening that JSON supply the parameter values to run().
export function graph_new(g, options = {}) {
  const rec = graph_record(), sources = new Map(), entriesByModule = new Map(), external = {}, scalarFns = [];
  let opened = null, readKey = null, freed = false, operations = 0;
  const mutable = () => { if (opened || freed) throw new Error("graph_new: graph already opened or freed"); };
  const api = {
    buf(id, spec) {
      mutable();
      const bytes = spec.bytes ?? spec.words * 4;
      if (spec.buffer) external[id] = spec.buffer;
      rec.buffer(id, bytes, { external: !!spec.buffer, persist: spec.persist === true });
      return api;
    },
    op(pe, spec) {
      mutable();
      if (!pe?.pipe || !pe.man || !pe.bgl) throw new Error("graph_new: expected a pipes() entry");
      const module = spec.module ?? `entry${operations}`;
      if (!entriesByModule.has(module)) {
        const entries = new Map();
        entriesByModule.set(module, entries);
        sources.set(module, { by: (kind, name) => entries.get(`${kind}:${name}`) });
      }
      const key = `${pe.man.kind}:${pe.man.name}`, entries = entriesByModule.get(module);
      if (entries.has(key) && entries.get(key) !== pe)
        throw new Error("graph_new: conflicting entry under module " + module);
      entries.set(key, pe);
      const sc = {};
      for (const [name, value] of Object.entries(spec.sc ?? {})) {
        if (typeof value === "function") {
          const param = `op${operations}_${name}`;
          scalarFns.push([param, value]);
          sc[name] = { param };
        } else sc[name] = value;
      }
      rec.dispatch({ module, kind: pe.man.kind, name: pe.man.name, in: spec.in, out: spec.out,
        rows: spec.rows, len: spec.len, sc });
      operations++;
      return api;
    },
    copy(src, dst, { src_off = 0, dst_off = 0, bytes }) {
      mutable(); rec.copy(src, dst, bytes, src_off, dst_off); return api;
    },
    async run(ctx = {}, { write = {}, read } = {}) {
      if (freed) throw new Error("graph_new: graph freed");
      if (!opened) {
        if (!Array.isArray(read) || read.length !== 2) throw new Error("graph_new: read [buffer, words] required");
        readKey = JSON.stringify(read);
        rec.read(read[0], read[1] * 4);
        opened = graph_open(g, sources, rec.serialize(), external, options);
      } else if (read && JSON.stringify(read) !== readKey) throw new Error("graph_new: readback shape changed");
      return (await opened).run({ write, params: api.params(ctx) });
    },
    // the named scalar parameters of the serialized graph, for graph_open() on another device
    params(ctx = {}) { return Object.fromEntries(scalarFns.map(([name, fn]) => [name, fn(ctx)])); },
    serialize() { if (!readKey) throw new Error("graph_new: readback not set"); return rec.serialize(); },
    free() { freed = true; return opened?.then((graph) => graph.invalidate()); },
  };
  return api;
}
