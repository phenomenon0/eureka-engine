// gguf.js (BENDWEB-SHOWCASE): read a GGUF header in the browser (or node) from a byte source, without loading the file.
//   const g = await gguf_read(read)   // read(offset, length) -> Uint8Array (File.slice, an HTTP Range, fs)
//   g.kv: {key: value} (arrays of strings stay arrays; big token arrays included), g.tensors: [{name, dims, type, offset}],
//   g.data: absolute byte offset of the tensor data section.
export const GGML = { 0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
  12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
  22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
  39: "MXFP4" };
// [elements per block, bytes per block]
export const GGML_BLOCK = { F32: [1, 4], F16: [1, 2], BF16: [1, 2], Q4_0: [32, 18], Q4_1: [32, 20], Q5_0: [32, 22], Q5_1: [32, 24],
  Q8_0: [32, 34], Q2_K: [256, 84], Q3_K: [256, 110], Q4_K: [256, 144], Q5_K: [256, 176], Q6_K: [256, 210], IQ4_XS: [256, 136], IQ4_NL: [32, 18],
  IQ2_XXS: [256, 66], IQ2_XS: [256, 74], IQ2_S: [256, 82], IQ3_XXS: [256, 98], IQ3_S: [256, 110], IQ1_S: [256, 50], IQ1_M: [256, 56],
  TQ1_0: [256, 54], TQ2_0: [256, 66], MXFP4: [32, 17] };

export async function gguf_read(read, { first = 1 << 23 } = {}) {
  let buf = await read(0, first), at = 0;
  const td = new TextDecoder();
  const need = async (n) => {   // grow the window until n more bytes are in it (headers carry ~5 MB of vocabulary)
    while (at + n > buf.byteLength) {
      const more = await read(buf.byteLength, Math.max(n, buf.byteLength));
      if (!more.byteLength) throw new Error("GGUF header is truncated");
      const b = new Uint8Array(buf.byteLength + more.byteLength); b.set(buf); b.set(more, buf.byteLength); buf = b;
    }
    return new DataView(buf.buffer, buf.byteOffset + at, n);
  };
  const u32 = async () => { const v = (await need(4)).getUint32(0, true); at += 4; return v; };
  const u64 = async () => { const v = (await need(8)).getBigUint64(0, true); at += 8; if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("GGUF value too large"); return Number(v); };
  const str = async () => { const n = await u64(); await need(n); const s = td.decode(buf.subarray(at, at + n)); at += n; return s; };
  const scalar = async (t) => {
    const size = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 }[t];
    if (t === 8) return str();
    if (!size) throw new Error("GGUF unknown value type " + t);
    const v = await need(size); at += size;
    switch (t) { case 0: return v.getUint8(0); case 1: return v.getInt8(0); case 2: return v.getUint16(0, true); case 3: return v.getInt16(0, true);
      case 4: return v.getUint32(0, true); case 5: return v.getInt32(0, true); case 6: return v.getFloat32(0, true); case 7: return v.getUint8(0) !== 0;
      case 10: return Number(v.getBigUint64(0, true)); case 11: return Number(v.getBigInt64(0, true)); case 12: return v.getFloat64(0, true); }
  };
  const value = async (t) => {
    if (t !== 9) return scalar(t);
    const et = await u32(), n = await u64(), out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = await value(et);
    return out;
  };
  const magic = td.decode((await need(4), buf.subarray(0, 4))); at = 4;
  if (magic !== "GGUF") throw new Error("not a GGUF file (magic " + JSON.stringify(magic) + ")");
  const version = await u32();
  if (version < 2 || version > 3) throw new Error("unsupported GGUF version " + version);
  const nt = await u64(), nk = await u64(), kv = {};
  for (let i = 0; i < nk; i++) { const k = await str(); kv[k] = await value(await u32()); }
  const tensors = [];
  for (let i = 0; i < nt; i++) {
    const name = await str(), nd = await u32(), dims = [];
    for (let d = 0; d < nd; d++) dims.push(await u64());
    const type = GGML[await u32()] ?? "unknown", offset = await u64();
    tensors.push({ name, dims, type, offset });
  }
  const align = kv["general.alignment"] ?? 32;
  const data = Math.ceil(at / align) * align;
  for (const t of tensors) {
    const [per, bytes] = GGML_BLOCK[t.type] ?? [0, 0];
    const n = t.dims.reduce((a, b) => a * b, 1);
    t.bytes = per ? n / per * bytes : null;
    t.abs = data + t.offset;
  }
  return { version, kv, tensors, data };
}
