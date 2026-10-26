// Device and pack lifetime for q35_plan. Publicly re-exported by q35_step.js.
import { pipes, entry_run, wgsl_manifest } from "./webgpu.js";
import { graph_open } from "./webgpu_graph.js";
import { q35_plan, Q35_HEAD_ROWS } from "./q35_plan.js";
import { XCHK_LIMIT, FAST_FORMATS, SPLIT_MODES, scale_scan, audit_chain } from "./q35_split.js";

const u8 = value => value instanceof ArrayBuffer ? new Uint8Array(value) :
  ArrayBuffer.isView(value) ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength) : null;
const hash = async bytes => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
  .map(x => x.toString(16).padStart(2, "0")).join("");
const unhex = s => Uint8Array.from(s.match(/../g), b => parseInt(b, 16));
async function verifyFile(file, bytes, blockBytes) {
  if (!file || bytes.byteLength !== file.bytes || file.chunks.length !== Math.ceil(file.bytes / blockBytes))
    throw new Error('q35 invalid file metadata');
  const digests=[];
  for(let i=0;i<file.chunks.length;i++) {
    const digest=await hash(bytes.subarray(i*blockBytes,Math.min((i+1)*blockBytes,file.bytes)));
    if(digest!==file.chunks[i])throw new Error(`q35 pack chunk SHA mismatch ${file.path}:${i}`);
    digests.push(...unhex(digest));
  }
  // web/v0's file SHA is SHA256(concatenated binary chunk digests), not
  // SHA256(file bytes). This is the same convention as the OPFS loader.
  if(await hash(Uint8Array.from(digests))!==file.sha256)throw new Error('q35 file digest mismatch '+file.path);
}
async function gpuErrors(device, body) {
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  let result, failure;
  try { result=await body(); } catch(error) { failure=error; }
  const memory=await device.popErrorScope(), validation=await device.popErrorScope();
  if(failure)throw failure;
  if(memory||validation)throw new Error('q35 GPU error: '+(memory??validation).message);
  return result;
}

// A pack source is a base URL or {read(path, offset, length)}. The latter is
// deliberately small: an OPFS adapter can use File.slice().arrayBuffer().
export function q35_source(source) {
  let opfsRoot;
  return async (path, offset = 0, length) => {
    let data;
    if (source?.opfs === true) {
      opfsRoot ??= (async () => {
        let dir=await navigator.storage.getDirectory();
        for(const part of (source.dir??'pack').split('/').filter(Boolean)) dir=await dir.getDirectoryHandle(part);
        return dir;
      })();
      let dir=await opfsRoot;
      const parts=path.split('/'),name=parts.pop();
      for(const part of parts)dir=await dir.getDirectoryHandle(part);
      const file=await (await dir.getFileHandle(name)).getFile();
      data=new Uint8Array(await file.slice(offset,length===undefined?undefined:offset+length).arrayBuffer());
    } else if (typeof source === "string" || source instanceof URL) {
      const base = String(source).replace(/\/?$/, "/");
      const response = await fetch(new URL(path, new URL(base, globalThis.location?.href ?? import.meta.url)),
        offset || length !== undefined ? { headers: { Range: `bytes=${offset}-${length === undefined ? "" : offset + length - 1}` } } : {});
      if (!response.ok) throw new Error(`q35 source ${path}: HTTP ${response.status}`);
      data = new Uint8Array(await response.arrayBuffer());
      if (response.status !== 206 && (offset || length !== undefined)) data = data.slice(offset, length === undefined ? undefined : offset + length);
    } else {
      const value = typeof source === "function" ? await source(path, offset, length) : await source.read(path, offset, length);
      data = u8(value);
    }
    if (!data || (length !== undefined && data.byteLength !== length)) throw new Error(`q35 source ${path}: short or invalid read`);
    return data;
  };
}

// Compile through the trunk's pipes(), timing each asynchronous pipeline via
// a small device facade. The shared runtime/emitter are not modified.
async function compile(g, texts, progress) {
  const compile_ms = [], modules = [];
  const results = await Promise.allSettled(Object.entries(texts).map(async ([name, text]) => {
    const d = g.device, start = performance.now();
    progress?.({ phase: "compile-start", module: name });
    const device = {
      createShaderModule: (...a) => d.createShaderModule(...a),
      createBindGroupLayout: (...a) => d.createBindGroupLayout(...a),
      createPipelineLayout: (...a) => d.createPipelineLayout(...a),
      async createComputePipelineAsync(desc) {
        const at = performance.now();
        const pipeline = await d.createComputePipelineAsync(desc);
        compile_ms.push({ module: name, entry: desc.compute.entryPoint, ms: performance.now() - at });
        return pipeline;
      },
    };
    const mod = await pipes({ ...g, device }, text);
    modules.push({ name, sha256: await hash(new TextEncoder().encode(text)), ms: performance.now() - start, entries: mod.entries.length });
    progress?.({ phase: "compile", module: name, entries: mod.entries.length });
    return [name, mod];
  }));
  const failed = results.filter(r => r.status === "rejected");
  if (failed.length) throw new AggregateError(failed.map(r => r.reason), "q35 pipeline compilation failed: " + failed.map(r=>String(r.reason)).join('; '));
  return { mods: Object.fromEntries(results.map(r => r.value)), compile_ms, modules };
}

function validate(plan, mods, limits) {
  const ready = new Set(Object.entries(plan.buffers).filter(([, b]) => b.weight).map(([n]) => n));
  for (const name of plan.states) ready.add(name);
  for (const [name, b] of Object.entries(plan.buffers)) {
    if (b.bytes > limits.maxBufferSize || b.bytes > limits.maxStorageBufferBindingSize)
      throw new Error(`q35 buffer exceeds actual device limits: ${name} (${b.bytes})`);
  }
  for (const s of plan.steps) {
    if (s.kind === "copy") {
      if (!ready.has(s.from) || s.fromOffset + s.bytes > plan.buffers[s.from].bytes ||
        s.toOffset + s.bytes > plan.buffers[s.to].bytes) throw new Error("q35 invalid copy " + s.from);
      ready.add(s.to); continue;
    }
    const e = mods[s.module]?.by(s.kind, s.name)?.man;
    if (!e) throw new Error(`q35 missing emitted entry ${s.module}:${s.kind}.${s.name}`);
    if (e.lanes > limits.maxComputeInvocationsPerWorkgroup || e.lanes > limits.maxComputeWorkgroupSizeX ||
      e.ins.length + 1 + Number(!!e.sc) + Number(!!e.out2) > limits.maxStorageBuffersPerShaderStage)
      throw new Error(`q35 entry exceeds actual device limits: ${s.module}:${s.name}`);
    if (e.ins.length !== Object.keys(s.in).length || e.ins.some(i => !Object.hasOwn(s.in, i.name)))
      throw new Error(`q35 bindings disagree: ${s.module}:${s.name}`);
    for (const i of e.ins) if (!ready.has(s.in[i.name])) throw new Error("q35 read before write: " + s.in[i.name]);
    for (const scalar of e.sc?.scalars ?? []) if (!Object.hasOwn(s.sc, scalar.name))
      throw new Error("q35 missing scalar " + scalar.name);
    const words = s.kind === "map" ? s.len : e.per === "elem" ? s.rows * e.len : s.rows;
    if (words * 4 > plan.buffers[s.out].bytes) throw new Error("q35 short output " + s.out);
    ready.add(s.out);
    if (s.out2) ready.add(s.out2);
  }
}

export async function q35_open(g, packSrc, manifest, options = {}) {
  const { ctx, progress, mode = "graph", trace = [], layers = 32, split = "spec", splitAudit = false, splitExactX = ["L0.h"] } = options;
  // TURBODEFAULT: numerics = "turbo" (default: row-tiled factored-math GEMV on every projection, no replay, NOT bit-exact; quality-gated)
  // or "canon" (opt-in: the bit-exact engine, the speculative split + exact E/replay path). Switchable at runtime: setNumerics().
  // canon: false at open drops the load-time scale scan and the canon buffers (turbo only; setNumerics("canon") then throws).
  let numerics = options.numerics ?? "turbo";
  if (!["turbo", "canon"].includes(numerics)) throw new Error("q35 numerics must be turbo or canon");
  const canonOK = options.canon !== false;
  if (numerics === "canon" && !canonOK) throw new Error("q35 numerics canon needs canon != false");
  if (!["graph", "direct"].includes(mode)) throw new Error("q35 mode must be graph or direct");
  // PIPELINESPLIT: spec (default) = speculate on the fast pipelines, replay exact when the x-range flag fires; exact = the
  // pre-split engine; replay = always replay (control); fast = never replay (UNSAFE mutant, controls only).
  if (!SPLIT_MODES.includes(split)) throw new Error("q35 split must be one of " + SPLIT_MODES.join("/"));
  // splitLinks: false removes the k_split chain (UNSAFE speed-only ablation): allowed only with split=fast, which never reads it
  const splitLinks = options.splitLinks !== false;
  if (!splitLinks && split !== "fast") throw new Error("q35 splitLinks=false is a speed-only ablation of split=fast");
  const read = q35_source(packSrc), table = await read("wtab.bin", 0, 4096);
  const plan = q35_plan(manifest, table, { ctx, layers, trace });   // the exact (canon, split=exact) plan; also the buffer/state authority
  const tplan = q35_plan(manifest, table, { ctx, layers, trace, turbo: true });
  if (options.verify !== false) await verifyFile(plan.model.files['wtab.bin'], table, manifest.web.block_bytes);
  const sgOK = options.subgroups !== false && g.device.features?.has("subgroups");
  const loadText = async (name) => {
    const given = options.modules?.[name];
    if (typeof given === "string") return given;
    const file = name === "k_gemv_turbo.wgsl" ? (sgOK ? "k_gemv_turbo_sg.wgsl" : "k_gemv_turbo.wgsl") : name === "k_gemv_fastE.wgsl" && options.canonF16 ? "k_gemv_fastEI.wgsl" : name;
    const response = await fetch(new URL(`../emit/${file}`, import.meta.url));
    if (!response.ok) throw new Error("q35 missing module " + file);
    return response.text();
  };
  const namesOf = (p) => [...new Set(p.steps.filter(s => s.kind !== "copy").map(s => s.module))];
  const turboNames = namesOf(tplan), canonNames = [...new Set([...namesOf(plan), "k_gemv_fastE.wgsl", "k_split.wgsl"])];
  const moduleNames = numerics === "turbo" ? turboNames : canonNames;   // the other mode compiles on first setNumerics()
  const texts = {};
  for (const name of moduleNames) texts[name] = await loadText(name);
  // Contract metadata may be supplied by a pack integrator. Reject ABI drift
  // before loading weights when contracts are present.
  for (const contract of options.contracts ?? []) {
    if (!texts[contract.module]) continue;
    if (contract.module_sha256 && await hash(new TextEncoder().encode(texts[contract.module])) !== contract.module_sha256)
      throw new Error("q35 contract module hash differs: " + contract.module);
    const entries = wgsl_manifest(texts[contract.module]).entries;
    for (const e of contract.entries) if (e.manifest && !entries.some(x => JSON.stringify(x) === JSON.stringify(e.manifest)))
      throw new Error(`q35 ABI drift: ${contract.module}:${e.name}`);
  }
  const compiled = await compile(g, texts, progress), d = g.device;
  validate(numerics === "turbo" ? tplan : plan, compiled.mods, d.limits);
  // the speculative variant's buffers (links, the zero word, its longer result) join the exact plan's; shared names agree in size
  const shape = (split === "exact" && !canonOK) ? null : q35_plan(manifest, table, { ctx, layers, trace, split: { fast: () => true, audit: !!splitAudit, exactX: splitExactX, links: splitLinks } });
  const allBuffers = { ...plan.buffers };
  for (const [name, b] of Object.entries(tplan.buffers)) {   // the readback buffer differs in size when a traced stage is fused away in turbo: one buffer, the larger size
    if (allBuffers[name] && allBuffers[name].bytes !== b.bytes) {
      if (name !== "result") throw new Error("q35 turbo buffer size conflict " + name);
      allBuffers[name] = { ...allBuffers[name], bytes: Math.max(allBuffers[name].bytes, b.bytes) };
    }
    allBuffers[name] ??= b;
  }
  for (const [name, b] of Object.entries(shape?.buffers ?? {})) {
    if (allBuffers[name] && allBuffers[name].bytes !== b.bytes) throw new Error("q35 split buffer size conflict " + name);
    allBuffers[name] ??= b;
  }
  const buffers = {}, graphs = new Map(); let closed = false, busy = false, poisoned = false, nextPos = 0, active = Promise.resolve();
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const facts = { ctx, layers, mode, pack_sha256: manifest.web.pack_sha256, wtab_sha256: await hash(table),
    compile_ms: compiled.compile_ms, modules: compiled.modules,
    pipeline_count: compiled.compile_ms.length, dispatches: (numerics === "turbo" ? tplan : plan).steps.filter(s => s.kind !== "copy").length, numerics,
    copies: plan.steps.filter(s => s.kind === "copy").length,
    resident_bytes: Object.values(allBuffers).reduce((n, b) => n + Math.max(16, b.bytes), 0),
    readback_bytes: (numerics === "turbo" ? tplan : plan).spec.result.bytes, subgroups: !!sgOK, device: g.facts, head_source_chunk_sha256: [], max_source_window_bytes: manifest.web.block_bytes };
  // the weight half of the fast-leaf rule, scanned once as the verified bytes stream past: tensor name -> all f16 scales finite
  const fastOK = new Map(), scaleFacts = [];
  const destroy = () => { for (const buffer of Object.values(buffers)) buffer.destroy(); };
  const upload = (name, data) => {
    const b = plan.buffers[name];
    if (!b || data.byteLength !== b.bytes) throw new Error("q35 invalid upload " + name);
    d.queue.writeBuffer(buffers[name], 0, data);
  };
  try { await gpuErrors(d, async () => {
    for (const [name, b] of Object.entries(allBuffers)) buffers[name] = d.createBuffer({ label: "q35:" + name, size: Math.max(16, b.bytes), usage });
    const tensors = plan.model.layers.slice(0, layers).flatMap(l => Object.values(l.tensors));
    const blocks = [...new Set(tensors.map(t => t.block))].sort((a, b) => a - b);
    for (const block of blocks) {
      const path = `blocks/b${block}.bin`, file = plan.model.files[path];
      const data = await read(path, 0, file.bytes);
      if (options.verify !== false) await verifyFile(file, data, manifest.web.block_bytes);
      if (buffers[`block${block}`]) upload(`block${block}`, data);
      if (canonOK && split !== "exact") for (const t of tensors.filter(t => t.block === block && FAST_FORMATS[t.format])) {
        const scan = scale_scan(t.format); scan.feed(data.subarray(t.offset, t.offset + t.bytes), 0);
        fastOK.set(t.name, scan.ok()); scaleFacts.push({ name: t.name, ok: scan.ok(), ...scan.facts() });
      }
      for (const t of tensors.filter(t => t.block === block && t.format === "f32")) upload(t.buffer, data.subarray(t.offset, t.offset + t.bytes));
      progress?.({ phase: "weights", block, blocks: blocks.length });
    }
    const chunkBytes = Q35_HEAD_ROWS * plan.model.head.rowBytes, headFile=plan.model.files['head.bin'], blockBytes=manifest.web.block_bytes;
    const headDigests=[], headScan=[0,1,2,3].map(() => scale_scan("q6k"));
    for (let at=0,index=0;at<headFile.bytes;at+=blockBytes,index++) {
      const data = await read("head.bin", at, Math.min(blockBytes,headFile.bytes-at));
      const digest=await hash(data);
      facts.head_source_chunk_sha256.push(digest);headDigests.push(...unhex(digest));
      if(options.verify !== false && digest!==headFile.chunks[index])throw new Error('q35 pack chunk SHA mismatch head.bin:'+index);
      // Source hash windows and whole-row GPU chunks have different edges.
      // Split uploads at a GPU chunk edge without changing tensor row order.
      for(let offset=0;offset<data.byteLength;) {
        const chunk=Math.floor((at+offset)/chunkBytes),local=(at+offset)%chunkBytes;
        const n=Math.min(data.byteLength-offset,chunkBytes-local);
        d.queue.writeBuffer(buffers[`head${chunk}`],local,data,offset,n);
        if (canonOK && split !== "exact") headScan[chunk].feed(data.subarray(offset,offset+n),local);
        offset+=n;
      }
      progress?.({ phase: "head", chunk:index, chunks:headFile.chunks.length });
    }
    if(options.verify !== false && await hash(Uint8Array.from(headDigests))!==headFile.sha256)throw new Error('q35 file digest mismatch head.bin');
    const normFile=plan.model.files['norm.bin'];
    const norm=await read("norm.bin", 0, normFile.bytes);
    if(options.verify !== false)await verifyFile(normFile,norm,blockBytes);
    upload("head.normw", norm.subarray(plan.model.normOffset,plan.model.normOffset+2560*4));
    if (canonOK && split !== "exact") headScan.forEach((scan, ch) => { fastOK.set(`head${ch}`, scan.ok()); scaleFacts.push({ name: `head${ch}`, ok: scan.ok(), ...scan.facts() }); });
    await d.queue.onSubmittedWorkDone();
  });
  } catch (error) { destroy(); throw error; }
  d.lost.then(() => { closed = true; poisoned = true; destroy(); });
  // splitExactX (default L0.h: the embedding layer's attention input carries exact zeros from zero Q6_K embedding codes on every
  // token, measured; its GEMVs keep the per-leaf exact module instead of forcing a replay every token)
  const splitPlan = { fast: (name) => fastOK.get(name) === true, audit: !!splitAudit, exactX: splitExactX, links: splitLinks };
  if (canonOK && split !== "exact") {
    const fp = q35_plan(manifest, table, { ctx, layers, trace, split: splitPlan });
    facts.split = { mode: split, audit: !!splitAudit, limit: XCHK_LIMIT, dispatches: fp.steps.filter(s => s.kind !== "copy").length,
      copies: fp.steps.filter(s => s.kind === "copy").length, readback_bytes: fp.spec.result.bytes,
      fast_gemvs: fp.steps.filter(s => s.module === "k_gemv_fastE.wgsl").length, exact_gemvs: fp.steps.filter(s => s.module === "k_gemv.wgsl").length,
      links: fp.steps.filter(s => s.module === "k_split.wgsl").length, exactX: splitExactX,
      scale_scanned: scaleFacts.length, scale_bad: scaleFacts.filter(f => !f.ok).map(f => f.name) };
  }
  async function variant(token, pos, fast = false, turbo = false) {
    const embedChunk = Math.floor(token / Q35_HEAD_ROWS), parity = pos % 2, key = `${embedChunk}:${parity}:${turbo ? "turbo" : fast ? "fast" : "exact"}`;
    if (!graphs.has(key)) {
      const p = q35_plan(manifest, table, { ctx, layers, trace, embedChunk, parity, ...(turbo ? { turbo: true } : fast ? { split: splitPlan } : {}) });
      validate(p, compiled.mods, d.limits);
      const graph = mode === "graph" ? await graph_open(g, compiled.mods, p.spec, buffers, { maxReadBytes: p.spec.result.bytes }) : null;
      graphs.set(key, { plan: p, graph });
    }
    return graphs.get(key);
  }
  async function direct(p, params) {
    let submits = 0, dispatches = 0; const at = performance.now();
    for (const s of p.steps) {
      if (s.out2) throw new Error("q35 direct mode does not support two-output fused kernels (turbo); use mode graph or numerics canon");
      if (s.kind === "copy") {
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(buffers[s.from], s.fromOffset, buffers[s.to], s.toOffset, s.bytes);
        d.queue.submit([enc.finish()]); submits++; continue;
      }
      const r = await entry_run(g, compiled.mods[s.module].by(s.kind, s.name), {
        in: Object.fromEntries(Object.entries(s.in).map(([k, n]) => [k, buffers[n]])),
        sc: Object.fromEntries(Object.entries(s.sc).map(([k, v]) => [k, typeof v === "object" ? params[v.param] : v])),
        rows: s.rows, len: s.len, keep: true,
      });
      // Destination can be padded (4096 hidden words, 2560 written). Preserve
      // its initialized +0 tail rather than replacing it with a short buffer.
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(r.buffer, 0, buffers[s.out], 0, r.bits.byteLength);
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone(); r.buffer.destroy();
      submits += 2; dispatches++;
    }
    const rb = d.createBuffer({ size: p.spec.result.bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(buffers[p.spec.result.buffer], 0, rb, 0, p.spec.result.bytes);
      d.queue.submit([enc.finish()]); submits++;
      await rb.mapAsync(GPUMapMode.READ);
      const bits = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap();
      return { bits, ms: performance.now() - at, submits, dispatches, copies: p.steps.length - dispatches };
    } finally { rb.destroy(); }
  }
  const api = {
    facts,
    plan,
    async step(token_id, pos) {
      if (closed || poisoned) throw new Error("q35 step is closed or invalidated; reopen after device loss/error");
      if (busy) throw new Error("q35 step already running");
      if (!Number.isInteger(token_id) || token_id < 0 || token_id >= 248320) throw new Error("q35 token outside vocabulary");
      if (!Number.isInteger(pos) || pos !== nextPos || pos >= ctx) throw new Error(`q35 expected position ${nextPos}, got ${pos} (capacity ${ctx})`);
      busy = true;
      const run = async () => {
        try {
          const params = { token: token_id, pos }, start = performance.now();
          const exec = (v) => gpuErrors(d, () => mode === "graph" ? v.graph.run({ params }) : direct(v.plan, params));
          let v, r, info;
          if (numerics === "turbo") { v = await variant(token_id, pos, false, true); r = await exec(v); info = { mode: "turbo" }; }
          else if (split === "exact") { v = await variant(token_id, pos); r = await exec(v); }
          else {
            // speculate: the fast variant writes only temporaries and the [1-parity] state banks, so a replay of this
            // position on the exact variant reads the same [parity] inputs and overwrites everything the fast pass wrote
            const fv = await variant(token_id, pos, true), fr = await exec(fv);
            if (closed) throw new Error("q35 step invalidated while running");
            const flag = fv.plan.flag ? fr.bits[fv.plan.flag.final] : null;
            if (splitAudit && fv.plan.flag) {
              const bad = audit_chain(fv.plan.flag.links.map(o => fr.bits[o]), flag);
              if (bad) throw new Error("q35 split audit: " + bad);
            }
            const replay = split === "replay" || (split === "spec" && flag >= XCHK_LIMIT);
            info = { mode: split, flag, replayed: replay, fast_ms: fr.ms };
            if (!replay) { v = fv; r = fr; }
            else {
              v = await variant(token_id, pos); const xr = await exec(v);
              info.exact_ms = xr.ms;
              r = { ...xr, ms: fr.ms + xr.ms, submits: fr.submits + xr.submits, dispatches: fr.dispatches + xr.dispatches, copies: fr.copies + xr.copies };
            }
          }
          if (closed) throw new Error("q35 step invalidated while running");
          const top2 = [{ id: r.bits[0], bits: r.bits[1] }, { id: r.bits[2], bits: r.bits[3] }];
          if (top2.some(x => x.id >= 248320 || (x.bits & 0x7f800000) === 0x7f800000)) throw new Error("q35 nonfinite or invalid top-2");
          nextPos++;
          const traced = v.plan.result.slice(1).filter(x => !/xchk$/.test(x.name));   // the split's link words are not stages
          const stages = Object.fromEntries(traced.map(x => [x.name, r.bits.slice(x.offset, x.offset + x.words)]));
          return { prof: r.prof, top2, numerics, ms: r.ms, wall_ms: performance.now() - start, submits: r.submits, dispatches: r.dispatches, copies: r.copies,
            ...(info ? { split: info } : {}), ...(traced.length ? { stages } : {}) };
        } catch (error) { poisoned = true; throw error; }
        finally { busy = false; }
      };
      active = run(); return active;
    },
    get numerics() { return numerics; },
    // runtime mode switch (same model, same buffers/KV state; takes effect on the next step). Canon compiles its modules on first use.
    async setNumerics(next) {
      if (!["turbo", "canon"].includes(next)) throw new Error("q35 numerics must be turbo or canon");
      if (closed || busy || poisoned) throw new Error("q35 cannot switch numerics while closed, busy or invalidated");
      if (next === numerics) return { numerics, compiled_ms: 0 };
      const at = performance.now();
      if (next === "canon" && !canonOK) throw new Error("q35 opened with canon:false");
      const need = (next === "canon" ? canonNames : turboNames).filter(n => !compiled.mods[n]), t = {};
      for (const n of need) t[n] = await loadText(n);
      if (need.length) {
        const more = await compile(g, t, progress);
        Object.assign(compiled.mods, more.mods); compiled.compile_ms.push(...more.compile_ms); compiled.modules.push(...more.modules);
      }
      if (next === "canon") {
        validate(plan, compiled.mods, d.limits);
        if (split !== "exact") validate(q35_plan(manifest, table, { ctx, layers, trace, split: splitPlan }), compiled.mods, d.limits);
      } else validate(tplan, compiled.mods, d.limits);
      numerics = next; facts.numerics = next;
      return { numerics, compiled_ms: performance.now() - at };
    },
    reset() {
      if (closed || busy) throw new Error("q35 cannot reset a closed or busy step");
      const enc = d.createCommandEncoder();
      for (const name of plan.states) enc.clearBuffer(buffers[name]);
      d.queue.submit([enc.finish()]);
      nextPos = 0; poisoned = false;
    },
    async free() {
      closed = true;
      for (const { graph } of graphs.values()) graph?.invalidate();
      await active.catch(() => {});
      await d.queue.onSubmittedWorkDone().catch(() => {});
      destroy(); graphs.clear();
    },
  };
  return api;
}
