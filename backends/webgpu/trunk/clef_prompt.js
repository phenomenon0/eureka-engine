// clef_prompt.js (CLEF-HEAD): a /v1/systemone request -> the pieces of Clef's joint prompt, each tagged with its decision order.
// Port of server-decision.cpp parse_questions() + fill_task_joint() and of the GGUF's "systemone" Jinja template for the text path
// (no images). The template is rendered by hand, so it is checked against the one this code was written from (CLEF_TEMPLATE_SHA256).
//   const q = clef_questions(body)                 // [{id, type, instructions, options: [{key, description}]}] in the server's order
//   const pieces = clef_pieces(body.state, q)      // [{text, order}]; tokenize each piece on its own (special tokens parsed)
//   const t = clef_tokens(pieces, encode)          // {ids, order, spans: {questions, options}} as clef.cpp's clef_get_spans() reads them
export const ORDER = { NONE: 0, NOUL: 1, CHOICE: 2, SCORE: 3, OPTION: 4 };
export const QTYPE = { noul: 0, choice: 1, score: 2 };   // token_types.weight row = order - ORDER.NOUL
export const CLEF_TEMPLATE_SHA256 = "083fcf6056ae4489f179e34ea299f180887c611c72d3357f895ffb94214d0480";   // sha256 of clef-flash's tokenizer.chat_template.systemone
export const CLEF_MAX_OPTIONS = 255;

const utf8 = new TextEncoder();
const bytecmp = (a, b) => {   // std::string operator<: byte order of the UTF-8
  const x = utf8.encode(a), y = utf8.encode(b), n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
};

/** parse_questions(): ids in request order; choice options sorted by key; noul is [true, false]; score keys are "0".."n-1". */
export function clef_questions(body) {
  if (!body || body.state === undefined || body.state === null) throw new Error('"state" must be provided');
  const Q = body.questions;
  if (!Q || typeof Q !== "object" || Array.isArray(Q) || !Object.keys(Q).length) throw new Error('"questions" must be a non-empty object');
  const out = [];
  for (const [id, q] of Object.entries(Q)) {
    const err = (m) => new Error(`questions.${id}: ${m}`);
    if (!q || typeof q !== "object" || Array.isArray(q)) throw err("must be an object");
    if (q.instructions === undefined || q.instructions === null) throw err('"instructions" must be provided');
    const c = q.criteria, isObj = c && typeof c === "object" && !Array.isArray(c);
    let options;
    if (q.type === "choice") {
      if (!isObj || !Object.keys(c).length) throw err('"criteria" must be a non-empty object');
      options = Object.entries(c).map(([key, description]) => ({ key, description })).sort((a, b) => bytecmp(a.key, b.key));
    } else if (q.type === "score") {
      if (!Array.isArray(c) || c.length < 2 || c.length > 10) throw err('"criteria" must be an array of 2 to 10 levels');
      options = c.map((description, i) => ({ key: String(i), description }));
    } else if (q.type === "noul") {
      if (c !== undefined && c !== null && !isObj) throw err('"criteria" must be an object');
      options = ["true", "false"].map(key => ({ key, description: isObj && key in c ? c[key] : null }));   // noul_true_first
    } else throw err('"type" must be one of: choice, score, noul');
    if (options.length > CLEF_MAX_OPTIONS) throw err(`too many options (${options.length}), this model supports at most ${CLEF_MAX_OPTIONS}`);
    out.push({ id, type: q.type, instructions: q.instructions, options });
  }
  return out;
}

// decision_sort_keys() + decision_replace_text(..., "<<clef:", "<<clef ") over a JSON value
const clean = (v) => {
  if (typeof v === "string") return v.split("<<clef:").join("<<clef ");
  if (Array.isArray(v)) return v.map(clean);
  if (v && typeof v === "object") { const o = {}; for (const k of Object.keys(v).sort(bytecmp)) o[k] = clean(v[k]); return o; }
  return v;
};
// C++ `ostream << double` (6 significant digits, %g)
const fmtG = (x) => {
  if (!Number.isFinite(x)) return x > 0 ? "inf" : x < 0 ? "-inf" : "nan";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const e = Math.floor(Math.log10(Math.abs(Number(x.toPrecision(6)))));
  if (e < -4 || e >= 6) { let [m, ex] = x.toExponential(5).split("e"); m = m.replace(/\.?0+$/, ""); const n = Number(ex); return `${m}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`; }
  return x.toFixed(Math.max(0, 5 - e)).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
};
// jinja value_to_json() with separators (',', ':'): raw UTF-8, the C escapes, \u00XX below 0x20
const ESC = { '"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t" };
export function tojson(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return Number.isInteger(v) && Math.abs(v) < 2 ** 63 ? String(v) : fmtG(v);
  if (typeof v === "string") return '"' + v.replace(/["\\\b\f\n\r\t\u0000-\u001f]/g, c => ESC[c] ?? "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0")) + '"';
  if (Array.isArray(v)) return "[" + v.map(tojson).join(",") + "]";
  return "{" + Object.entries(v).map(([k, x]) => tojson(k) + ":" + tojson(x)).join(",") + "}";
}
const str = (v) => typeof v === "string" ? v : tojson(v);

/** The systemone template, rendered piece by piece (the pieces are what the template separates with {{ sep }}). */
export function clef_pieces(state, questions) {
  state = clean(state);
  const P = [], none = (text) => P.push({ text, order: ORDER.NONE });
  none("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n");
  none(str(state));
  none("\n\nSCHEMA FIELDS:\n");
  questions.forEach((q0, qi) => {
    const q = { ...q0, instructions: clean(q0.instructions), options: q0.options.map(o => ({ key: clean(o.key), description: clean(o.description ?? null) })) };
    none(`\nFIELD ${qi + 1}\nID: ${clean(q.id)}\nTYPE: ${q.type}\nINSTRUCTION: `);
    P.push({ text: str(q.instructions), order: ORDER[q.type.toUpperCase()] });
    none("\nALLOWED OPTIONS:\n");
    q.options.forEach((o, oi) => {
      none(`OPTION ${oi + 1}: `);
      let d = o.description;
      if (q.type === "noul" && d === null) d = o.key === "true" ? "The proposition is true or the answer is yes." : "The proposition is false or the answer is no.";
      P.push({ text: tojson(d === null ? { option_id: o.key } : { description: d, option_id: o.key }), order: ORDER.OPTION });
      none("\n");
    });
    none("END FIELD\n");
  });
  none("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
  return P.filter(p => p.text.length || p.order !== ORDER.NONE);
}

/** Tokenize each piece on its own; the spans are the runs of equal order, as clef_get_spans() reads them. */
export function clef_tokens(pieces, encode) {
  const ids = [], order = [];
  for (const p of pieces) {
    const t = encode(p.text);
    if (p.order !== ORDER.NONE && !t.length) throw new Error("the instructions and the options of a question must not be empty");
    for (const id of t) { ids.push(id); order.push(p.order); }
  }
  const questions = [], options = [];
  for (let i = 0; i < ids.length;) {
    let end = i + 1; while (end < ids.length && order[end] === order[i]) end++;
    if (order[i] >= ORDER.NOUL && order[i] <= ORDER.SCORE) questions.push({ type: order[i] - ORDER.NOUL, start: i, end });
    else if (order[i] === ORDER.OPTION) { if (!questions.length) throw new Error("an option before any question"); options.push({ question: questions.length - 1, start: i, end }); }
    i = end;
  }
  return { ids, order, spans: { questions, options } };
}
