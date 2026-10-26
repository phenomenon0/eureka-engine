// Device and pack lifetime for q35_plan. Publicly re-exported by q35_step.js.
import { pipes, entry_run, wgsl_manifest } from "./webgpu.js";
import { graph_open } from "./webgpu_graph.js";
import { q35_plan, Q35_HEAD_ROWS, q35_prefill_plan, Q35_PF_WIDTHS } from "./q35_plan.js";
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
  // BATCHPREFILL: prompt tokens (all but the last) layer-major in chunks of T through the turbo kernels (q35_prefill_plan): no head,
  // no readback, one compute pass a chunk, chunks pipelined (encode the next while the GPU runs this one), one wait at the end.
  // Turbo only. prefillBatched: GEMV entries that run as one batched dispatch a chunk (k_gemv_turbo_b.wgsl, subgroups only).
  // prefillKV: "inplace" (default: k_fuse_pf kvput1, one bank) or "pingpong" (the decode step's whole-cache kvput every token).
  // prefillTokens: the row-independent ops as one dispatch a chunk (k_*_tb twins; default on with the in-place kvput).
  const pfT = options.prefillChunk ?? 32, pfTB = options.prefillBatchWidth ?? 4, pfKV = options.prefillKV ?? "inplace", pfPlans = new Map();
  const pfTok = pfKV === "inplace" && options.prefillTokens !== false;
  // webprefill2: prefillGrid: GEMV entries through their grid-token twins (k_gemv_turbo_grf.wgsl); prefillGu: "r<R>t<T>" gate+up shape
  // (k_gemv_turbo_guf.wgsl) or "" for the per-token gu. Both subgroups only and need prefillTokens; WebKit keeps the per-token GEMVs.
  const pfRecur = !pfTok || options.prefillRecur === false ? false : (options.prefillRecur ?? "sg") === "sg" && sgOK ? "sg" : true;   // webprefill2: convh/gdn one dispatch a chunk ("sg": k_fuse_rcs.wgsl, else / no subgroups: k_fuse_rc.wgsl)
  const pfEmbed = pfTok && options.prefillEmbed !== false;   // webprefill2: embed_tb (k_embed_tb.wgsl), one dispatch per head chunk
  // webprefill2 defaults (bitwise vs the shipped entries: receipts bc5-rtf / bc3-guf / bc2-grf; q6k R x T is NOT bit-identical -> grid)
  const Q35_PF_RT = { q4k2560: "r2t4", q4k4096: "r2t4", q4k9216: "r4t2", q5k2560: "r2t8", q5k4096: "r2t4" };
  // webprefill3: no subgroups (WebKit): the grid-token twins of the shipped NON-subgroup k_gemv_turbo.wgsl entries, gu included
  // (k_gemv_turbo_ngrf.wgsl, bitwise receipts bc7-wk-*; the no-subgroup R x T twins are not bit-identical there). prefillGrid /
  // prefillGu ("g" = gu_g) apply; prefillNsg: false = the per-token GEMVs (as webprefill2). prefillRt stays subgroups only.
  const Q35_PF_NGRID = ["q4k2560", "q4k4096", "q4k9216", "q6k2560", "q6k9216", "q5k2560", "q5k4096"];
  const pfN = !sgOK && pfTok && options.prefillNsg !== false;
  const pfRt = new Map(sgOK && pfTok ? Object.entries(options.prefillRt ?? Q35_PF_RT) : []);   // webprefill2: entry -> "r<R>t<T>"
  const pfGrid = new Set(sgOK && pfTok ? (options.prefillGrid ?? ["q6k2560", "q6k9216"]) : pfN ? (options.prefillGrid ?? Q35_PF_NGRID) : []);
  const pfGu = sgOK && pfTok ? (options.prefillGu ?? "r1t8") : pfN ? (options.prefillGu ?? "g") : "";
  if (pfGu === "g" && !pfN) throw new Error("q35 prefillGu \"g\" is the no-subgroup path only");
  const pfBatched = new Set(sgOK ? (options.prefillBatched ?? ["q5k4096", "q4k9216", "q4k4096"]).filter((x) => !pfGrid.has(x) && !pfRt.has(x)) : []);
  if (pfT % pfTB || !Number.isInteger(pfT) || pfT < 1) throw new Error("q35 prefillChunk must be a multiple of prefillBatchWidth");
  if (!["inplace", "pingpong"].includes(pfKV)) throw new Error("q35 prefillKV must be inplace or pingpong");
  let pfReady = null;
  const pfInit = () => pfReady ??= (async () => {
    const want = [...(pfBatched.size ? ["k_gemv_turbo_b.wgsl"] : []), ...(pfGrid.size ? [pfN ? "k_gemv_turbo_ngrf.wgsl" : "k_gemv_turbo_grf.wgsl"] : []), ...(pfGu ? [pfGu === "g" ? "k_gemv_turbo_ngrf.wgsl" : "k_gemv_turbo_guf.wgsl"] : []), ...(pfRecur ? [pfRecur === "sg" ? "k_fuse_rcs.wgsl" : "k_fuse_rc.wgsl"] : []), ...(pfRt.size ? ["k_gemv_turbo_rtf.wgsl"] : []), ...(pfEmbed ? ["k_embed_tb.wgsl"] : []), ...(pfKV === "inplace" && !pfTok ? ["k_fuse_pf.wgsl"] : []), ...(pfTok ? ["k_fuse_tb.wgsl", "k_norms_tb.wgsl", "k_fuse_pfb.wgsl"] : [])].filter((n, i, a) => !compiled.mods[n] && a.indexOf(n) === i);
    if (want.length) {
      const t = {}; for (const n of want) t[n] = await loadText(n);
      const more = await compile(g, t, progress);
      Object.assign(compiled.mods, more.mods); compiled.compile_ms.push(...more.compile_ms); compiled.modules.push(...more.modules);
    }
    for (const [k, w] of Object.entries(Q35_PF_WIDTHS)) buffers[`pf.${k}`] ??= d.createBuffer({ label: "q35:pf." + k, size: pfT * w * 4, usage });
  })();
  const align = d.limits.minStorageBufferOffsetAlignment;
  function pfPlan(nt, parity0, bank) {   // the chunk plan, its pipelines, grids and static bindings, cached by (nt, parity, bank)
    const key = nt + ":" + parity0 + ":" + bank;
    if (pfPlans.has(key)) return pfPlans.get(key);
    const p = q35_prefill_plan(manifest, table, { ctx, layers, nt, T: pfT, TB: pfTB, parity0, batched: pfBatched, bank, tokb: pfTok, grid: pfGrid, gu: pfGu || null, recur: pfRecur, rt: pfRt, embed: pfEmbed, nsg: pfN });
    const res = ([name, off, words]) => typeof name === "object" ? ("head" in name ? { head: name.head } : { embed: name.embed }) : { buffer: buffers[name], offset: off * 4, size: words === null ? undefined : words * 4 };
    let slots = 0, stride = 0;
    const prep = p.steps.map((s) => {
      const pe = compiled.mods[s.module]?.by(s.kind, s.name);
      if (!pe?.pipe) throw new Error(`q35 prefill: missing entry ${s.module}:${s.kind}.${s.name}`);
      const e = pe.man, tile = s.tile ?? 1;
      if (s.kind === "cube" && s.count % tile) throw new Error("q35 prefill: rows not a multiple of the tile");
      const ins = e.ins.map((i) => { if (!s.bind[i.name]) throw new Error(`q35 prefill: unbound ${s.name}.${i.name}`); return [i.binding, res(s.bind[i.name])]; });
      const outs = [[e.out.binding, res(s.out)], ...(e.out2 ? [[e.out2.binding, res(s.out2)]] : [])];
      if (!!e.out2 !== !!s.out2) throw new Error("q35 prefill: second output disagrees " + s.name);
      const bytes = e.sc ? Math.max(16, (1 + e.sc.scalars.length + e.sc.lens.length + 1) * 4) : 0;
      stride = Math.max(stride, bytes);
      let grid = s.kind === "cube" ? [Math.min(s.count / tile, 65535), Math.ceil(s.count / tile / 65535)] : [Math.min(Math.ceil(s.count / e.lanes), 65535), Math.ceil(Math.ceil(s.count / e.lanes) / 65535)];
      if (s.tokens) { if (grid[1] !== 1 || !(e.tokens || e.gridtok === "row" || e.gurt)) throw new Error("q35 prefill: not a token-batched entry " + s.name); grid = [grid[0], s.tokens]; }   // y = token slot
      return { s, pe, e, ins, outs, slot: e.sc ? slots++ : -1, bytes, grid };
    });
    const v = { plan: p, prep, slots, stride: Math.ceil(Math.max(16, stride) / align) * align };
    pfPlans.set(key, v); return v;
  }
  function pfEncode(v, toks, pos0, pre = null) {
    const sc = new Uint8Array(v.slots * v.stride), scb = d.createBuffer({ size: Math.max(16, sc.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const chunkOf = (t) => Math.floor(toks[t] / Q35_HEAD_ROWS);
    const val = (x) => typeof x === "number" ? x : "pos" in x ? pos0 + x.pos : "tok" in x ? toks[x.tok] : chunkOf(x.row0) * Q35_HEAD_ROWS;
    const enc = d.createCommandEncoder();
    pre?.(enc);
    const pass = enc.beginComputePass();
    for (const q of v.prep) {
      const ent = (b, r) => ({ binding: b, resource: r.head !== undefined ? { buffer: buffers[`head${r.head}`] } : r.embed !== undefined ? { buffer: buffers[`head${chunkOf(r.embed)}`] } : { buffer: r.buffer, offset: r.offset, ...(r.size ? { size: r.size } : {}) } });
      const entries = [...q.ins.map(([b, r]) => ent(b, r)), ...q.outs.map(([b, r]) => ent(b, r))];
      if (q.e.sc) {
        const words = new Uint32Array(sc.buffer, q.slot * v.stride, q.bytes / 4);
        words[0] = q.s.count;
        for (const x of q.e.sc.scalars) words[x.at] = val(q.s.sc[x.name]) >>> 0;
        for (const x of q.e.sc.lens) { const r = q.ins.find(([b]) => b === q.e.ins.find((i) => i.name === x.name).binding)[1]; words[x.at] = r.head !== undefined ? buffers[`head${r.head}`].size : r.embed !== undefined ? buffers[`head${chunkOf(r.embed)}`].size : (r.size ?? r.buffer.size - r.offset); }
        entries.push({ binding: q.e.sc.binding, resource: { buffer: scb, offset: q.slot * v.stride, size: q.bytes } });
      }
      pass.setPipeline(q.pe.pipe);
      pass.setBindGroup(0, d.createBindGroup({ layout: q.pe.bgl, entries }));
      pass.dispatchWorkgroups(...q.grid);
    }
    pass.end();
    d.queue.writeBuffer(scb, 0, sc);
    return { cmd: enc.finish(), scb };
  }
  // the prefill's extra modules and scratch compile with the model (not on the first prompt) when the session opens on turbo
  if (numerics === "turbo" && options.prefill !== false) { await pfInit(); facts.prefill = { chunk: pfT, batch_width: pfTB, kv: pfKV, tokens: pfTok, batched: [...pfBatched], grid: [...pfGrid], gu: pfGu, recur: pfRecur, rt: Object.fromEntries(pfRt), embed: pfEmbed, nsg: pfN }; }
  const api = {
    facts,
    plan,
    // BATCHPREFILL: step tokens[i] at pos0 + i for every i, without logits (no head, nothing read back). Leaves the session exactly
    // where stepping them one by one would (state banks, nextPos); the caller steps the prompt's last token for its logits.
    async prefill(tokens, pos0) {
      if (closed || poisoned) throw new Error("q35 prefill is closed or invalidated; reopen after device loss/error");
      if (busy) throw new Error("q35 step already running");
      if (numerics !== "turbo") throw new Error("q35 prefill runs the turbo graph only");
      if (!Array.isArray(tokens) || !tokens.length || tokens.some((x) => !Number.isInteger(x) || x < 0 || x >= 248320)) throw new Error("q35 prefill: token outside vocabulary");
      if (pos0 !== nextPos || pos0 + tokens.length >= ctx) throw new Error(`q35 prefill expected position ${nextPos}, got ${pos0} (+${tokens.length}, capacity ${ctx})`);
      busy = true;
      const run = async () => {
        const fresh = [];
        try {
          const start = performance.now(); let submitted = 0, dispatches = 0, encode_ms = 0;
          await gpuErrors(d, async () => {
            await pfInit();
            // in place: the whole history goes to the bank the step after the prefill reads; a follow-up turn's earlier rows move there first
            const bank = pfKV === "inplace" ? (pos0 + tokens.length) % 2 : null, from = pos0 % 2;
            const pre = bank !== null && pos0 > 0 && bank !== from ? (enc) => { for (const L of pfPlan(Math.min(tokens.length, pfT), pos0 % 2, bank).plan.kv) for (const c of ["kc", "vc"]) enc.copyBufferToBuffer(buffers[L + c + from], 0, buffers[L + c + bank], 0, pos0 * 4096); } : null;
            for (let i = 0; i < tokens.length; i += pfT) {
              const toks = tokens.slice(i, i + pfT), at = performance.now();
              const v = pfPlan(toks.length, (pos0 + i) % 2, bank), { cmd, scb } = pfEncode(v, toks, pos0 + i, i === 0 ? pre : null);
              encode_ms += performance.now() - at;
              fresh.push(scb); d.queue.submit([cmd]); submitted++; dispatches += v.prep.length;
              if (closed) throw new Error("q35 prefill invalidated while running");
            }
            await d.queue.onSubmittedWorkDone();
          });
          if (closed) throw new Error("q35 prefill invalidated while running");
          nextPos += tokens.length;
          return { tokens: tokens.length, ms: performance.now() - start, encode_ms, submits: submitted, dispatches, chunk: pfT, batch_width: pfTB, kv: pfKV, tokb: pfTok, batched: [...pfBatched], grid: [...pfGrid], gu: pfGu, recur: pfRecur, rt: Object.fromEntries(pfRt), embed: pfEmbed };
        } catch (error) { poisoned = true; throw error; }
        finally { for (const b of fresh) b.destroy(); busy = false; }
      };
      active = run(); return active;
    },
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
