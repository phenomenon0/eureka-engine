// clef_head.js (CLEF-HEAD): Clef's decision head in plain JS, over the trunk's final-normed hidden states.
// Port of llama.cpp src/models/clef.cpp build_head() (JointSchemaHead): LayerNorm + projections 4096->1024, 2 routing blocks
// (options read the prompt), 4 joint blocks (fields read each other and the prompt), then prior + cosine + scorer MLP per option.
// Weights: dec.* / decision.* (Q8_0 + F32, one BF16 vector), and the rows of output.weight (Q6_K here) of the option tokens.
//   const h = await clef_head_open(read, g)                 // dequantizes the head once (~31M params, 125 MB of f32)
//   const scores = await h.scores(hidden, ids, spans)       // hidden: Float32Array[n_tok * 4096]; one score per option, in span order
//   const answers = h.answers(questions, scores)            // format_answer(): softmax at the GGUF temperature per question
const f16 = (u) => { const s = u & 0x8000 ? -1 : 1, e = (u >> 10) & 31, m = u & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); };
const F16 = new Float32Array(65536); for (let i = 0; i < 65536; i++) F16[i] = f16(i);

function deq(type, b, n) {   // raw GGUF bytes -> Float32Array[n]
  const out = new Float32Array(n), v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (type === "F32") out.set(new Float32Array(b.slice(0, n * 4).buffer));
  else if (type === "BF16") for (let i = 0; i < n; i++) out[i] = new Float32Array(new Uint32Array([v.getUint16(2 * i, true) << 16]).buffer)[0];
  else if (type === "Q8_0") for (let blk = 0, o = 0; o < n; blk++, o += 32) {
    const d = F16[v.getUint16(blk * 34, true)];
    for (let j = 0; j < 32; j++) out[o + j] = d * v.getInt8(blk * 34 + 2 + j);
  } else if (type === "Q6_K") for (let blk = 0, o = 0; o < n; blk++, o += 256) {   // block_q6_K {ql[128], qh[64], scales[16], d}
    const B = blk * 210, d = F16[v.getUint16(B + 208, true)];
    for (let h = 0; h < 2; h++) {
      const ql = B + 64 * h, qh = B + 128 + 32 * h, sc = B + 192 + 8 * h, y = o + 128 * h;
      for (let l = 0; l < 32; l++) {
        const is = l >> 4, H = b[qh + l];
        out[y + l] = d * v.getInt8(sc + is) * (((b[ql + l] & 15) | ((H & 3) << 4)) - 32);
        out[y + l + 32] = d * v.getInt8(sc + is + 2) * (((b[ql + l + 32] & 15) | (((H >> 2) & 3) << 4)) - 32);
        out[y + l + 64] = d * v.getInt8(sc + is + 4) * (((b[ql + l] >> 4) | (((H >> 4) & 3) << 4)) - 32);
        out[y + l + 96] = d * v.getInt8(sc + is + 6) * (((b[ql + l + 32] >> 4) | (((H >> 6) & 3) << 4)) - 32);
      }
    }
  } else throw new Error("clef head: no dequant for " + type);
  return out;
}

// ---- dense helpers, row-major: a matrix is {w: Float32Array[rows*K], rows, K}; X is n vectors of K packed
function mm(W, X, n, bias = null) {   // -> n vectors of W.rows: Y[t] = W X[t] + b
  const { w, rows, K } = W, Y = new Float32Array(n * rows);
  for (let r = 0; r < rows; r++) {
    const o = r * K, b = bias ? bias[r] : 0;
    for (let t = 0; t < n; t++) {
      const x = t * K; let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let k = 0; k < K; k += 4) { s0 += w[o + k] * X[x + k]; s1 += w[o + k + 1] * X[x + k + 1]; s2 += w[o + k + 2] * X[x + k + 2]; s3 += w[o + k + 3] * X[x + k + 3]; }
      Y[t * rows + r] = s0 + s1 + s2 + s3 + b;
    }
  }
  return Y;
}
function layernorm(X, n, D, g, eps) {   // ggml_norm then * w + b
  const Y = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    const o = t * D; let m = 0; for (let i = 0; i < D; i++) m += X[o + i]; m /= D;
    let v = 0; for (let i = 0; i < D; i++) { const d = X[o + i] - m; v += d * d; } v /= D;
    const s = 1 / Math.sqrt(v + eps);
    for (let i = 0; i < D; i++) Y[o + i] = (X[o + i] - m) * s * g.w[i] + g.b[i];
  }
  return Y;
}
const addTo = (A, B) => { for (let i = 0; i < A.length; i++) A[i] += B[i]; return A; };
const gelu = (A) => { for (let i = 0; i < A.length; i++) { const x = A[i]; A[i] = 0.5 * x * (1 + erf(x * Math.SQRT1_2)); } return A; };
function erf(x) {   // |error| < 1.2e-7 (Numerical Recipes erfc)
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851632 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}
const l2 = (x, o, D, eps = 1e-12) => { let s = 0; for (let i = 0; i < D; i++) s += x[o + i] * x[o + i]; return 1 / Math.max(Math.sqrt(s), eps); };
const rows = (X, D, idx) => { const Y = new Float32Array(idx.length * D); idx.forEach((r, i) => Y.set(X.subarray(r * D, r * D + D), i * D)); return Y; };

// build_head_attn(): q [nq, D], kv [nkv, D], H heads, no mask
function attn(q, nq, kv, nkv, A, H) {
  const D = A.wq.rows, dh = D / H, sc = 1 / Math.sqrt(dh);
  const Q = mm(A.wq, q, nq, A.bq), Kx = mm(A.wk, kv, nkv, A.bk), V = mm(A.wv, kv, nkv, A.bv), O = new Float32Array(nq * D), p = new Float32Array(nkv);
  for (let h = 0; h < H; h++) for (let i = 0; i < nq; i++) {
    const qo = i * D + h * dh; let m = -Infinity;
    for (let j = 0; j < nkv; j++) { let s = 0; const ko = j * D + h * dh; for (let k = 0; k < dh; k++) s += Q[qo + k] * Kx[ko + k]; p[j] = s * sc; if (p[j] > m) m = p[j]; }
    let z = 0; for (let j = 0; j < nkv; j++) { p[j] = Math.exp(p[j] - m); z += p[j]; }
    for (let j = 0; j < nkv; j++) { const w = p[j] / z, vo = j * D + h * dh; for (let k = 0; k < dh; k++) O[qo + k] += w * V[vo + k]; }
  }
  return mm(A.wo, O, nq, A.bo);
}

export async function clef_head_open(read, g) {
  const kv = g.kv, a = kv["general.architecture"], by = new Map(g.tensors.map(t => [t.name, t]));
  if (kv[`${a}.decision.type`] !== "clef") throw new Error(`It isn't a Clef decision model (decision.type ${kv[`${a}.decision.type`] ?? "absent"}).`);
  const nR = kv[`${a}.decision.routing_block_count`], nJ = kv[`${a}.decision.block_count`], H = kv[`${a}.decision.head_count`];
  const eps = kv[`${a}.attention.layer_norm_epsilon`], d = kv[`${a}.embedding_length`];
  const temps = Object.fromEntries(Object.entries(kv).filter(([k]) => k.startsWith(`${a}.decision.temperature.`)).map(([k, v]) => [k.slice(`${a}.decision.temperature.`.length), Number(v)]));
  let bytes = 0;
  const vec = async (name) => { const t = by.get(name); if (!t) throw new Error("clef head: no " + name); bytes += t.bytes; return deq(t.type, await read(t.abs, t.bytes), t.dims.reduce((x, y) => x * y, 1)); };
  const mat = async (name) => { const t = by.get(name); return { w: await vec(name), K: t.dims[0], rows: t.dims[1] ?? 1 }; };
  const nrm = async (p) => ({ w: await vec(p + ".weight"), b: await vec(p + ".bias") });
  const att = async (p) => ({ wq: await mat(p + "_q.weight"), bq: await vec(p + "_q.bias"), wk: await mat(p + "_k.weight"), bk: await vec(p + "_k.bias"),
    wv: await mat(p + "_v.weight"), bv: await vec(p + "_v.bias"), wo: await mat(p + "_o.weight"), bo: await vec(p + "_o.bias") });
  const L = [];
  for (let i = 0; i < nR + nJ; i++) {
    const P = `dec.blk.${i}.`, l = {};
    if (i < nR) l.cross_norm_kv = await nrm(P + "cross_attn_norm_kv"); else { l.self_norm = await nrm(P + "attn_norm"); l.self = await att(P + "attn"); }
    l.cross_norm = await nrm(P + "cross_attn_norm"); l.cross = await att(P + "cross_attn"); l.ffn_norm = await nrm(P + "ffn_norm");
    l.up = await mat(P + "ffn_up.weight"); l.up_b = await vec(P + "ffn_up.bias"); l.down = await mat(P + "ffn_down.weight"); l.down_b = await vec(P + "ffn_down.bias");
    L.push(l);
  }
  const M = { hidden_norm: await nrm("decision.hidden_norm"), option_summary_norm: await nrm("decision.option_summary_norm"),
    field_norm: await nrm("decision.field_norm"), option_norm: await nrm("decision.option_norm") };
  for (const p of ["memory", "question", "option_question", "global", "option_context", "option_lexical"]) M[p] = await mat(`decision.proj_${p}.weight`);
  M.scales = await vec("decision.scales"); M.type = await vec("token_types.weight");
  M.scorer = await mat("decision.scorer.weight"); M.scorer_b = await vec("decision.scorer.bias");
  M.scorer_out = await vec("decision.scorer_out.weight"); M.scorer_out_b = (await vec("decision.scorer_out.bias"))[0];
  const D = M.memory.rows, out = by.get(by.has("output.weight") ? "output.weight" : "token_embd.weight"), outRow = out.bytes / out.dims[1];
  const lexCache = new Map();
  async function lexRow(id) {   // a row of output.weight, dequantized (cached)
    if (!lexCache.has(id)) lexCache.set(id, deq(out.type, await read(out.abs + id * outRow, outRow), d));
    return lexCache.get(id);
  }
  const ffn = (X, n, l) => mm(l.down, gelu(mm(l.up, layernorm(X, n, D, l.ffn_norm, eps), n, l.up_b)), n, l.down_b);

  async function scores(hidden, ids, spans, { trace = null } = {}) {
    const n = ids.length, { questions: Qs, options: Os } = spans, nq = Qs.length, no = Os.length;
    if (!nq || !no || Qs.some((_, i) => !Os.some(o => o.question === i))) return new Float32Array(no || 1).fill(NaN);   // status = NaN
    const t0 = performance.now();
    const hid = layernorm(hidden, n, d, M.hidden_norm, eps);
    const memory = mm(M.memory, hid, n);
    const global = hid.subarray((n - 1) * d, n * d);
    const pool = (sp) => { const Y = new Float32Array(sp.length * d); sp.forEach((s, i) => { for (let t = s.start; t < s.end; t++) for (let k = 0; k < d; k++) Y[i * d + k] += hid[t * d + k] / (s.end - s.start); }); return Y; };
    const question = pool(Qs), optctx = pool(Os);
    const lexical = new Float32Array(no * d);
    for (let i = 0; i < no; i++) { const s = Os[i]; for (let t = s.start; t < s.end; t++) { const r = await lexRow(ids[t]); for (let k = 0; k < d; k++) lexical[i * d + k] += r[k] / (s.end - s.start); } }
    const owner = Os.map(o => o.question);
    // one query per option
    let options = addTo(addTo(mm(M.option_context, optctx, no), mm(M.option_lexical, lexical, no)), rows(mm(M.option_question, question, nq), D, owner));
    for (let il = 0; il < nR; il++) {
      const l = L[il];
      options = addTo(options, attn(layernorm(options, no, D, l.cross_norm, eps), no, layernorm(memory, n, D, l.cross_norm_kv, eps), n, l.cross, H));
      options = addTo(options, ffn(options, no, l));
    }
    // one vector per question: its text, a summary of its own options, the end of the prompt and its type
    let fields = mm(M.question, question, nq);
    {
      const summary = new Float32Array(nq * D), sc = 1 / Math.sqrt(D);
      for (let q = 0; q < nq; q++) {
        const mine = []; for (let o = 0; o < no; o++) if (owner[o] === q) { let s = 0; for (let k = 0; k < D; k++) s += options[o * D + k] * fields[q * D + k]; mine.push([o, s * sc]); }
        const m = Math.max(...mine.map(x => x[1])); let z = 0; for (const x of mine) { x[1] = Math.exp(x[1] - m); z += x[1]; }
        for (const [o, w] of mine) for (let k = 0; k < D; k++) summary[q * D + k] += w / z * options[o * D + k];
      }
      addTo(fields, layernorm(summary, nq, D, M.option_summary_norm, eps));
      const gl = mm(M.global, global, 1);
      for (let q = 0; q < nq; q++) for (let k = 0; k < D; k++) fields[q * D + k] += gl[k] + M.type[Qs[q].type * D + k];
    }
    for (let il = nR; il < nR + nJ; il++) {
      const l = L[il];
      let cur = layernorm(fields, nq, D, l.self_norm, eps);
      fields = addTo(fields, attn(cur, nq, cur, nq, l.self, H));
      cur = layernorm(fields, nq, D, l.cross_norm, eps);
      fields = addTo(fields, attn(cur, nq, memory, n, l.cross, H));
      fields = addTo(fields, ffn(fields, nq, l));
    }
    fields = layernorm(fields, nq, D, M.field_norm, eps);
    // prior: the output embeddings of the option against the question and the end of the prompt
    const prior = new Float32Array(no), cosine = new Float32Array(no), resid = new Float32Array(no), out = new Float32Array(no);
    const anchor = new Float32Array(nq * d);
    for (let q = 0; q < nq; q++) { for (let k = 0; k < d; k++) anchor[q * d + k] = question[q * d + k] + global[k]; const s = l2(anchor, q * d, d); for (let k = 0; k < d; k++) anchor[q * d + k] *= s; }
    for (let o = 0; o < no; o++) { const s = l2(lexical, o * d, d); let p = 0; for (let k = 0; k < d; k++) p += lexical[o * d + k] * s * anchor[owner[o] * d + k]; prior[o] = p; }
    // joint: each option against the vector of its question
    const opts = layernorm(options, no, D, M.option_norm, eps), field = rows(fields, D, owner), feats = new Float32Array(no * 4 * D);
    for (let o = 0; o < no; o++) {
      const sf = l2(field, o * D, D), so = l2(opts, o * D, D); let c = 0;
      for (let k = 0; k < D; k++) {
        const f = field[o * D + k], x = opts[o * D + k]; c += f * sf * x * so;
        feats[o * 4 * D + k] = f; feats[o * 4 * D + D + k] = x; feats[o * 4 * D + 2 * D + k] = f * x; feats[o * 4 * D + 3 * D + k] = Math.abs(f - x);
      }
      cosine[o] = c;
    }
    const hsc = gelu(mm(M.scorer, feats, no, M.scorer_b));
    for (let o = 0; o < no; o++) { let s = M.scorer_out_b; for (let k = 0; k < D; k++) s += M.scorer_out[k] * hsc[o * D + k]; resid[o] = s;
      out[o] = prior[o] * M.scales[0] + (cosine[o] * M.scales[1] + resid[o]) * M.scales[2]; }
    if (trace) Object.assign(trace, { head_ms: performance.now() - t0, prior: Array.from(prior), cosine: Array.from(cosine), residual: Array.from(resid) });
    return out;
  }

  // get_temperature(): bucket by option count, then the type alone, else 1
  const temperature = (q) => { const n = q.options.length, b = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    return temps[`${q.type}.${b}`] ?? temps[q.type] ?? 1; };
  function answers(questions, s) {   // format_answer() for one variant (Clef: one joint pass, no reversed variant)
    const res = {}; let i = 0;
    for (const q of questions) {
      const sc = Array.from(s.slice(i, i + q.options.length)); i += q.options.length;
      if (sc.some(Number.isNaN)) throw new Error("the model could not evaluate the decision");
      const T = temperature(q), m = Math.max(...sc), e = sc.map(x => Math.exp((x - m) / T)), z = e.reduce((x, y) => x + y, 0), p = e.map(x => x / z);
      const A = { type: q.type };
      if (q.type === "noul") A.noul = p[q.options.findIndex(o => o.key === "true")];
      else {
        const probabilities = Object.fromEntries(q.options.map((o, j) => [o.key, p[j]])), n = p.length, best = p.indexOf(Math.max(...p));
        if (q.type === "choice") Object.assign(A, { choice: q.options[best].key, probabilities, confidence: n < 2 ? 1 : Math.max(0, (p[best] - 1 / n) / (1 - 1 / n)) });
        else {
          let dist = 0, du = 0; p.forEach((x, j) => { dist += x * Math.abs(j - best); du += Math.abs(j - (n - 1) / 2) / n; });
          Object.assign(A, { score: p.reduce((x, y, j) => x + j * y, 0), legend: Object.fromEntries(q.options.map(o => [o.key, o.description])), probabilities, confidence: Math.max(0, 1 - dist / du) });
        }
      }
      A.raw_scores = sc; A.temperature = T;
      res[q.id] = A;
    }
    return res;
  }
  return { scores, answers, temperatures: temps, bytes, dims: { d, D, H, routing: nR, joint: nJ, eps }, scales: Array.from(M.scales) };
}
