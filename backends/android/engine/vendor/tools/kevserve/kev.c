// kev.c: KEVSERVE request side (docs/kevserve-registration.md; README.md). tools/kevdual/kev.c plus: explicit
// "mode":"decide", unknown modes refused, nesting depth and token overflow refused. sys1dual's dual.c contract with
// Kev-4B as the model: linked into the engine in dual.c's place, called by the Sys1.next / Sys1.done effects (sys1dual/sys1.c).
// stdin: one JSON request per line. stdout: one JSON response per line; a bad request gets {"error":MSG}, the process stays.
//   {"mode":"generate","prompt":TEXT,"n":N} -> {"prompt_ids":[...],"ids":[...],"text":"..."}   (dual.c, unchanged)
//   {["mode":"decide",]"state":..,"questions":{qid:{"type","instructions","criteria"}}} -> {"answers":{qid:ANSWER},"usage":{..}}
// A decide is one exact prefill per question (k4bfull: Kev serves each question as its own row on the shared
// state), queued here: Sys1.next hands out the next question without reading stdin until the request is answered.
// Render = tools/k4bfull/assemble.py, readout + ANSWER bytes = tools/k4bfull/kevhead.c (same f32 order).
// Timings + kevhead's raw line: one JSON line per request on stderr.
// Env: LLB_GGUF (the Kev GGUF), TMPDIR, KEVDUAL_KEEP (dir: keep each question's marker + pointer rows, kevhead format).
#define _POSIX_C_SOURCE 200809L
#include "../../host/llb_gguf.c"
#include "../clef3/json.h"
#include "../qwen35_tokenizer.h"
#include <math.h>
#include <time.h>

enum { D = 2560, N = 512, DP = 256, RB = D / 32 * 34, KMAX = 255, TMAX = 16384, QMAX = 64 };

static void die(const char *s) { fprintf(stderr, "kevdual: %s\n", s); exit(1); }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- render (assemble.py)

static const char *SPECIAL[5] = {"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"};
static Q35Tokenizer tok;
static int sid[5];        // P M BS BE S
static double temp[3];    // qwen35.decision.temperature.{noul,choice,score}
static const char *TYPES[3] = {"noul", "choice", "score"};

typedef struct {
  char *id;
  int type, n, k, ids[TMAX], at[KMAX + 2];  // at = markers..., pointer
  char *keys[KMAX];
} Q;

static int is_box(const jv *v) { return v->type == J_ARR || v->type == J_OBJ; }

static void lstrip_add(sb *o, sb *x) {
  char *p = x->p ? x->p : "";
  while (*p == ' ' || (*p >= '\t' && *p <= '\r')) p++;
  sb_add(o, p);
}

static void render(sb *o, const jv *v, int indent) {  // kev/api.py render
  if (v->type == J_NULL) return;
  if (v->type == J_STR) { sb_add(o, v->str); return; }
  if (v->type == J_NUM) { sb_add(o, v->raw); return; }  // ponytail: integers as written; Python float repr only for plain literals
  if (v->type == J_BOOL) { sb_add(o, v->b ? "True" : "False"); return; }
  for (size_t i = 0; i < v->n; i++) {
    sb x = {0};
    if (i) sb_addc(o, '\n');
    for (int j = 0; j < indent; j++) sb_add(o, "  ");
    if (v->type == J_ARR) {
      render(&x, v->items[i], indent + 1);
      sb_add(o, "- ");
      lstrip_add(o, &x);
    } else if (is_box(v->items[i])) {
      render(&x, v->items[i], indent + 1);
      sb_fmt(o, "%s:\n%s", v->keys[i], x.p ? x.p : "");
    } else {
      render(&x, v->items[i], 0);
      sb_fmt(o, "%s: %s", v->keys[i], x.p ? x.p : "");
    }
    free(x.p);
  }
}

static void escape(sb *o, const char *t) {  // kev/model.py user_tokens: <|x|> -> <¦x¦>
  while (*t) {
    const char *e = t + 2;
    if (t[0] == '<' && t[1] == '|') {
      while ((*e >= 'A' && *e <= 'Z') || (*e >= 'a' && *e <= 'z') || (*e >= '0' && *e <= '9') || *e == '_') e++;
      if (e > t + 2 && e[0] == '|' && e[1] == '>') {
        sb_add(o, "<\xc2\xa6");
        sb_addn(o, t + 2, (size_t)(e - t - 2));
        sb_add(o, "\xc2\xa6>");
        t = e + 2;
        continue;
      }
    }
    sb_addc(o, *t++);
  }
}

// escape(render(v)) or escape(name[: render(desc)]) (option_text) as plain-text ids (llb-tok enc), appended
static int piece(Q *q, const char *name, const jv *v) {
  sb r = {0}, e = {0};
  if (name) sb_add(&r, name);
  if (v && !(v->type == J_NULL || (v->type == J_STR && !*v->str))) {
    if (name) sb_add(&r, ": ");
    render(&r, v, 0);
  }
  escape(&e, r.p ? r.p : "");
  int c = e.p ? q35_tok_encode(&tok, e.p, 0, q->ids + q->n, TMAX - q->n) : 0;
  free(r.p), free(e.p);
  if (c < 0 || c > TMAX - q->n) return 0;  // over cap: the count comes back, nothing copied
  q->n += c;
  return 1;
}

static int push(Q *q, int id) {
  if (q->n >= TMAX) return 0;
  q->ids[q->n++] = id;
  return 1;
}

// one question of the request -> ids, markers + pointer, keys; NULL = ok, else the refusal
static const char *build(Q *q, const jv *state, const char *qid, const jv *b) {
  const jv *t = json_get(b, "type"), *c = json_get(b, "criteria"), *ins = json_get(b, "instructions");
  q->id = strdup(qid);
  q->n = q->k = 0;
  if (!t || t->type != J_STR || !ins) return "each question needs type and instructions";
  q->type = !strcmp(t->str, "noul") ? 0 : !strcmp(t->str, "choice") ? 1 : !strcmp(t->str, "score") ? 2 : -1;
  if (q->type < 0) return "type must be noul, choice or score";
  if (!push(q, sid[0]) || !piece(q, NULL, state) || !push(q, sid[1]) || !piece(q, NULL, ins)) return "request too long";
  size_t K = q->type == 0 ? 2 : c ? c->n : 0;
  if (q->type == 0 && c && c->type != J_OBJ) return "noul criteria must be an object";
  if (q->type == 1 && (!c || c->type != J_OBJ)) return "choice criteria must be an object";
  if (q->type == 2 && (!c || c->type != J_ARR)) return "score criteria must be an array";
  if (K < 1 || K > KMAX) return "1..255 options";
  for (size_t i = 0; i < K; i++) {
    char num[8];
    snprintf(num, sizeof num, "%zu", i);
    const char *key = q->type == 0 ? (i ? "true" : "false") : q->type == 1 ? c->keys[i] : num;
    int ok = !push(q, sid[2]) ? 0
      : q->type == 0 ? piece(q, i ? "yes" : "no", json_get(c, key))
      : q->type == 1 ? piece(q, key, c->items[i]) : piece(q, NULL, c->items[i]);
    if (!ok || !push(q, sid[3])) return "request too long";
    q->at[q->k] = q->n - 1;
    q->keys[q->k++] = strdup(key);
  }
  if (!push(q, sid[4])) return "request too long";
  q->at[q->k] = q->n - 1;
  return NULL;
}

static void temps(const char *path) {  // the f32 qwen35.decision.temperature.<type> keys
  int fd = open(path, O_RDONLY);
  if (fd < 0) die("cannot open LLB_GGUF");
  GgRd r = {fd, 0, 0, malloc(1 << 20), 0, 0};
  gg_rd_get(&r, NULL, 8);
  gg_u64(&r);
  uint64_t nkv = gg_u64(&r);
  for (uint64_t i = 0; i < nkv && !r.bad; i++) {
    char *key = gg_str(&r, 1);
    uint32_t t = gg_u32(&r);
    const char *p = "qwen35.decision.temperature.";
    if (key && t == 6 && !strncmp(key, p, strlen(p))) {
      float f;
      gg_rd_get(&r, &f, 4);
      for (int j = 0; j < 3; j++) if (!strcmp(key + strlen(p), TYPES[j])) temp[j] = f;
    } else gg_val(&r, t);
    free(key);
  }
  free(r.buf);
  close(fd);
  for (int j = 0; j < 3; j++) if (!(temp[j] > 0)) die("GGUF lacks qwen35.decision.temperature.{noul,choice,score}");
}

static void tok_init(const char *dir, const char *gguf) {
  if (q35_tok_load(&tok, dir)) die("sidecar requires tokenizer.bin (llb-tok extract)");
  for (int i = 0; i < 5; i++) {
    int id[2];
    if (q35_tok_encode(&tok, SPECIAL[i], 1, id, 2) != 1) die("special token missing from the vocab");
    sid[i] = id[0];
  }
  temps(gguf);
}

// ---------------------------------------------------------------- readout (kevhead.c)

static uint8_t wq[N * RB];
static float bias[N];

static void head(const float *h, float *y) {  // decide.c's y = W h + b, f32 in index order
  for (int i = 0; i < N; i++) {
    const uint8_t *row = wq + (size_t)i * RB;
    float acc = 0;
    for (int k = 0; k < D / 32; k++) {
      const uint8_t *blk = row + k * 34;
      _Float16 d;
      memcpy(&d, blk, 2);
      for (int j = 0; j < 32; j++) acc += ((float)d * (int8_t)blk[2 + j]) * h[k * 32 + j];
    }
    y[i] = acc + bias[i];
    if (!isfinite(y[i])) die("nonfinite head output");
  }
}

static void head_init(const char *path) {
  GgFile g;
  char err[256];
  if (gg_open(&g, path, err)) die(err);
  const GgT *w = NULL, *b = NULL;
  for (size_t i = 0; i < g.nt; i++) {
    if (!strcmp(g.t[i].name, "cls.output.weight")) w = &g.t[i];
    if (!strcmp(g.t[i].name, "cls.output.bias")) b = &g.t[i];
  }
  if (!w || !b || !w->q8 || w->rb != RB || w->bytes != (uint64_t)N * RB || b->bytes != N * 4) die("cls.output: not Q8_0 2560x512 + F32 512");
  if (gg_pread(g.fd, wq, sizeof wq, w->src) || gg_pread(g.fd, (uint8_t *)bias, sizeof bias, b->src)) die("tensor read failed");
  close(g.fd);
}

static double r4(double x) { return round(x * 1e4) / 1e4; }  // kev/api.py round_prob (4 decimals)

// rows (pos order, all prompt rows of the question) -> kevhead's raw line and answer object
static void answer(const Q *q, float (*h)[D], sb *raw, sb *o) {
  static float yq[N], yk[N];
  static double s[KMAX], p[KMAX];
  int K = q->k;
  head(h[q->at[K]], yq);
  for (int i = 0; i < K; i++) {
    head(h[q->at[i]], yk);
    double acc = 0;
    for (int j = 0; j < DP; j++) acc += (double)yk[DP + j] * yq[j];  // k(h_opt) . q(h_decide)
    s[i] = acc / sqrt((double)DP);
  }
  double T = temp[q->type], mx = s[0], sum = 0, best = 0, ex = 0;
  int arg = 0;
  for (int i = 1; i < K; i++) if (s[i] > mx) mx = s[i];
  for (int i = 0; i < K; i++) sum += p[i] = exp((s[i] - mx) / T);
  for (int i = 0; i < K; i++) { p[i] /= sum; if (p[i] > best) best = p[i], arg = i; ex += i * p[i]; }
  sb_add(raw, "raw");
  for (int i = 0; i < K; i++) sb_fmt(raw, " %s s=%.9g p=%.9g", q->keys[i], s[i], p[i]);
  sb_fmt(o, "{\"type\": \"%s\", ", TYPES[q->type]);
  if (q->type == 0) { sb_fmt(o, "\"noul\": %.4f}", r4(p[1])); return; }
  if (q->type == 1) {
    double conf = K == 1 ? 1 : (best - 1.0 / K) / (1 - 1.0 / K);
    sb_add(o, "\"choice\": ");
    json_quote(o, q->keys[arg]);
    sb_fmt(o, ", \"confidence\": %.4f, \"probabilities\": {", r4(conf));
  } else {
    double dist = 0, dU = 0;
    for (int i = 0; i < K; i++) dist += p[i] * fabs((double)(i - arg)), dU += fabs(i - (K - 1) / 2.0) / K;
    double conf = K == 1 ? 1 : fmax(0, 1 - dist / dU);
    sb_fmt(o, "\"score\": %.4f, \"confidence\": %.4f, \"probabilities\": {", r4(ex), r4(conf));
  }
  for (int i = 0; i < K; i++) {
    if (i) sb_add(o, ", ");
    json_quote(o, q->keys[i]);
    sb_fmt(o, ": %.4f", r4(p[i]));
  }
  sb_add(o, "}}");
}

// ---------------------------------------------------------------- the request loop (dual.c)

static FILE *out, *cap;
static char *capp, *text, hidden_path[4096];
static size_t capn;
static jv *req;
static Q qs[QMAX];
static int mode, steps, seq, prompt_n, prompt_ids[TMAX], nq, qi, total;
static double t_req, t_fwd, fwd_ms[QMAX], head_ms;
static sb ans, raws;

static void reply(const char *s) {
  fputs(s, out);
  fputc('\n', out);
  fflush(out);
}

static void refuse(const char *msg) {
  sb e = {0};
  sb_add(&e, "{\"error\":");
  json_quote(&e, msg);
  sb_addc(&e, '}');
  reply(e.p);
  free(e.p);
  fprintf(stderr, "{\"seq\":%d,\"error\":\"%s\"}\n", seq, msg);
}

static void drop(void) {
  for (int i = 0; i < nq; i++) {
    free(qs[i].id);
    for (int j = 0; j < qs[i].k; j++) free(qs[i].keys[j]);
  }
  json_free(req);
  free(text), free(ans.p), free(raws.p);
  text = NULL, req = NULL, nq = qi = 0, ans = raws = (sb){0};
}

static void init(const char *dir) {
  double t0 = now();
  const char *gguf = getenv("LLB_GGUF"), *tmp = getenv("TMPDIR");
  if (!gguf || !*gguf) die("kevdual needs LLB_GGUF");
  if (!tmp || !*tmp) die("set TMPDIR to owned scratch");
  if (snprintf(hidden_path, sizeof hidden_path, "%s/kevdual-%d.hidden", tmp, (int)getpid()) >= (int)sizeof hidden_path) die("TMPDIR too long");
  tok_init(dir, gguf);
  head_init(gguf);
  out = stdout;
  fprintf(stderr, "kevdual: kev head + tokenizer ready %.3f ms (T noul %.9g choice %.9g score %.9g)\n", (now() - t0) * 1000, temp[0], temp[1], temp[2]);
}

static char *command(const int *ids, int n, int st) {  // argv words for the request's ids + steps
  sb c = {0};
  for (int i = 0; i < n; i++) sb_fmt(&c, "%d ", ids[i]);
  sb_fmt(&c, "--steps %d", st);
  return c.p;
}

static int too_deep(const char *s) {  // the parser recurses per [ / {: refuse deep lines before it runs
  int d = 0, in = 0;
  for (; *s; s++) {
    if (in) { if (*s == '\\' && s[1]) s++; else if (*s == '"') in = 0; }
    else if (*s == '"') in = 1;
    else if ((*s == '[' || *s == '{') && ++d > 64) return 1;
    else if (*s == ']' || *s == '}') d--;
  }
  return 0;
}

// one request line -> 1 (queued) or 0 (refused, answered already)
static int parse(char *line, unsigned ctx) {
  const char *err = NULL;
  text = line;
  if (too_deep(text)) return refuse("JSON nested deeper than 64"), 0;
  req = json_parse(text, &err);
  if (!req || req->type != J_OBJ) return refuse(err ? err : "request must be a JSON object"), 0;
  jv *m = json_get(req, "mode");
  if (m && !(m->type == J_STR && (!strcmp(m->str, "generate") || !strcmp(m->str, "decide"))))
    return refuse("mode must be generate or decide"), 0;
  if (m && !strcmp(m->str, "generate")) {
    jv *p = json_get(req, "prompt"), *k = json_get(req, "n");
    if (!p || p->type != J_STR || !k || k->type != J_NUM || k->num < 1 || k->num > 65536 || k->num != floor(k->num))
      return refuse("generate needs prompt (string) and n (integer 1..65536)"), 0;
    int c = q35_tok_encode(&tok, p->str, 0, prompt_ids, TMAX);
    if (c < 1 || c > TMAX) return refuse("prompt must tokenize to 1..16384 tokens"), 0;
    if ((unsigned)c + (unsigned)k->num - 1 > ctx) return refuse("prompt + n exceeds the KV capacity fixed at launch"), 0;
    mode = 2, steps = (int)k->num, prompt_n = c;
    return 1;
  }
  const jv *state = json_get(req, "state"), *qo = json_get(req, "questions");
  if (!state || !qo || qo->type != J_OBJ || qo->n < 1 || qo->n > QMAX) return refuse("decide needs state and questions (object, 1..64)"), 0;
  for (nq = 0; nq < (int)qo->n; nq++) {
    if (qo->items[nq]->type != J_OBJ) return refuse("each question must be an object"), 0;
    const char *e = build(&qs[nq], state, qo->keys[nq], qo->items[nq]);
    if (e) return nq++, refuse(e), 0;
    if ((unsigned)qs[nq].n > ctx) return nq++, refuse("question exceeds the KV capacity fixed at launch"), 0;
  }
  mode = 1, steps = 1, total = 0;
  return 1;
}

int dual_next(const char *dir, unsigned ctx, char **cmd, const char **hidden, unsigned *n) {
  if (!out) init(dir);
  if (!(mode == 1 && qi < nq)) for (;;) {
    char *line = NULL;
    size_t len = 0;
    if (getline(&line, &len, stdin) < 0) {
      free(line);
      unlink(hidden_path);
      return 0;
    }
    seq++;
    t_req = now();
    head_ms = 0;
    if (parse(line, ctx)) break;
    drop();
  }
  if (mode == 2) *n = (unsigned)prompt_n, *hidden = NULL, *cmd = command(prompt_ids, prompt_n, steps);
  else *n = (unsigned)qs[qi].n, *hidden = hidden_path, *cmd = command(qs[qi].ids, qs[qi].n, 1);
  t_fwd = now();
  capp = NULL, capn = 0;
  cap = open_memstream(&capp, &capn);
  if (!cap) die("open_memstream failed");
  stdout = cap;
  return 1;
}

static float (*rows(const Q *q))[D] {  // the hook's file: every prompt row, [pos, width] + f32 row each
  float (*h)[D] = malloc(sizeof *h * q->n);
  FILE *f = fopen(hidden_path, "rb"), *keep = NULL;
  const char *kd = getenv("KEVDUAL_KEEP");
  if (!h || !f) die("cannot read the hidden rows");
  if (kd && *kd) {
    char p[4200];
    snprintf(p, sizeof p, "%s/%d-%s.hidden", kd, seq, q->id);
    if (!(keep = fopen(p, "wb"))) die("KEVDUAL_KEEP: cannot write");
  }
  for (int i = 0, m = 0; i < q->n; i++) {
    uint32_t hd[2];
    if (fread(hd, 4, 2, f) != 2 || hd[0] != (uint32_t)i || hd[1] != D || fread(h[i], 4, D, f) != D) die("hidden rows: bad dump");
    if (i != q->at[m]) continue;
    for (int j = 0; j < D; j++) if (!isfinite(h[i][j])) die("nonfinite hidden row");
    if (keep && (fwrite(hd, 4, 2, keep) != 2 || fwrite(h[i], 4, D, keep) != D)) die("KEVDUAL_KEEP: write failed");
    m++;
  }
  fclose(f);
  if (keep && fclose(keep)) die("KEVDUAL_KEEP: close failed");
  return h;
}

void dual_done(void) {
  double t_ran = now();
  fflush(cap);
  stdout = out;
  static int ids[65536];  // gen's prints: one id per line
  int k = 0;
  for (char *s = capp, *e; k < 65536; s = e) {
    long v = strtol(s, &e, 10);
    if (e == s) break;
    ids[k++] = (int)v;
  }
  fclose(cap);
  free(capp);
  sb o = {0};
  if (mode == 2) {
    if (k != steps) die("generate: id count differs from n");
    sb_add(&o, "{\"prompt_ids\":[");
    for (int i = 0; i < prompt_n; i++) sb_fmt(&o, i ? ",%d" : "%d", prompt_ids[i]);
    sb_add(&o, "],\"ids\":[");
    for (int i = 0; i < k; i++) sb_fmt(&o, i ? ",%d" : "%d", ids[i]);
    int need = q35_tok_decode(&tok, ids, k, NULL, 0);
    char *t = malloc(need + 1);
    q35_tok_decode(&tok, ids, k, t, need);
    t[need] = 0;
    sb_add(&o, "],\"text\":");
    json_quote(&o, t);
    sb_addc(&o, '}');
    free(t);
    reply(o.p);
    fprintf(stderr, "{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"n\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
      seq, prompt_n, k, (t_ran - t_req) * 1000, (now() - t_req) * 1000);
    free(o.p);
    drop();
    return;
  }
  Q *q = &qs[qi];
  fwd_ms[qi] = (t_ran - t_fwd) * 1000;
  double t_h = now();
  float (*h)[D] = rows(q);
  sb r = {0};
  sb_add(&ans, qi ? "," : "{\"answers\":{");
  json_quote(&ans, q->id);
  sb_addc(&ans, ':');
  answer(q, h, &r, &ans);
  free(h);
  head_ms += (now() - t_h) * 1000;
  sb_add(&raws, qi ? "," : "{");
  json_quote(&raws, q->id);
  sb_addc(&raws, ':');
  json_quote(&raws, r.p);
  free(r.p);
  total += q->n;
  if (++qi < nq) return;
  sb_fmt(&ans, "},\"usage\":{\"input_tokens\":%d}}", total);
  reply(ans.p);
  sb_add(&o, "[");
  for (int i = 0; i < nq; i++) sb_fmt(&o, i ? ",%.3f" : "%.3f", fwd_ms[i]);
  fprintf(stderr, "{\"seq\":%d,\"mode\":\"decide\",\"questions\":%d,\"input_tokens\":%d,\"forward_ms\":%s],\"head_ms\":%.3f,\"total_ms\":%.3f,\"raw\":%s}}\n",
    seq, nq, total, o.p, head_ms, (now() - t_req) * 1000, raws.p);
  free(o.p);
  drop();
}

#ifdef KEV_MAIN
// kevrender TOKDIR GGUF REQUEST.json: per question "qid n markers pointer T ids..." (the CPU render check, G2)
int main(int argc, char **argv) {
  if (argc != 4) die("usage: kevrender TOKDIR GGUF REQUEST.json");
  tok_init(argv[1], argv[2]);
  const char *err = NULL;
  char *text = json_read_file(argv[3]);
  jv *r = text ? json_parse(text, &err) : NULL;
  const jv *qo = json_get(r, "questions");
  if (!qo || qo->type != J_OBJ) die(err ? err : "bad request");
  for (size_t i = 0; i < qo->n; i++) {
    const char *e = build(&qs[0], json_get(r, "state"), qo->keys[i], qo->items[i]);
    if (e) die(e);
    printf("%s %d ", qs[0].id, qs[0].n);
    for (int j = 0; j < qs[0].k; j++) printf(j ? ",%d" : "%d", qs[0].at[j]);
    printf(" %d %.17g", qs[0].at[qs[0].k], temp[qs[0].type]);
    for (int j = 0; j < qs[0].n; j++) printf(" %d", qs[0].ids[j]);
    printf("\n");
  }
  return 0;
}
#endif
