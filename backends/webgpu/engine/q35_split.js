// q35_split.js (PIPELINESPLIT): the host half of the fast/exact pipeline split, shared by the engine (q35_engine.js / q35_plan.js)
// and the unit suites (run.js `split` jobs).
//   x side (per token, on the GPU): k_split.wgsl xchk4 / xchk1 links; the last link's word < XCHK_LIMIT iff every x word the fast
//     GEMVs read had its exponent field in 67 .. 187 (k_gemv.bend's xin16 rule). Otherwise the position is replayed exact.
//   weight side (static, once at load): every block's f16 scale(s) finite (k_gemv.bend's f16fin on d / dmin, Q6_K's d at +208).
//     A tensor that fails is bound to the exact module in the fast plan.
export const XCHK_LIMIT = 1015021568;           // 121 << 23: t = (|bits| - (67 << 23)) mod 2^32 below it <=> exponent 67 .. 187
export const FAST_FORMATS = { q4k: 144, q5k: 176, q6k: 210 };
export const SPLIT_MODES = ["spec", "exact", "replay", "fast"];
const f16fin = (h) => (h & 0x7c00) !== 0x7c00;

// the rule on one x word (a JS reference of the WGSL; used by the audit and the unit-suite builder)
export const xt = (bits) => (((bits & 0x7fffffff) - 562036736) >>> 0);
export const xok = (bits) => xt(bits) < XCHK_LIMIT;

// A streaming scan of block headers. feed(bytes, at) with consecutive windows of the tensor's bytes (at = the window's offset
// from the tensor's first byte); ok() after the last window. Q4_K / Q5_K: the block's first u32 holds d (low half) and dmin (high
// half); Q6_K: d is the u16 at +208. A header split across two windows is completed from the carried byte.
export function scale_scan(format) {
  const bb = FAST_FORMATS[format];
  if (!bb) throw new Error("scale_scan: not a fast format " + format);
  const fields = format === "q6k" ? [208] : [0, 2];
  let ok = true, blocks = 0, bad = 0, carry = null;
  const at16 = (u8, i) => u8[i] | (u8[i + 1] << 8);
  const check = (h) => { if (!f16fin(h)) { ok = false; bad++; } };
  return {
    feed(u8, at) {
      const end = at + u8.byteLength;
      if (carry) { check(carry.lo | (u8[0] << 8)); carry = null; }
      for (let b = Math.floor(at / bb) * bb; b < end; b += bb) {
        for (const f of fields) {
          const p = b + f;
          if (p < at || p >= end) continue;
          if (f === fields[0]) blocks++;
          if (p + 1 < end) check(at16(u8, p - at)); else carry = { lo: u8[p - at] };
        }
      }
    },
    ok: () => ok && carry === null,
    facts: () => ({ format, blocks, bad }),
  };
}
export function scales_finite(u8, format) { const s = scale_scan(format); s.feed(u8, 0); return s.ok(); }

// the chain audit (debug): link words must never decrease and the final word must be the last link's
export function audit_chain(links, final) {
  for (let i = 1; i < links.length; i++)
    if (links[i] < links[i - 1]) return `dropped flag: link ${i} word ${links[i]} < link ${i - 1} word ${links[i - 1]}`;
  if (links.length && links[links.length - 1] !== final) return `final ${final} != last link ${links[links.length - 1]}`;
  return null;
}
