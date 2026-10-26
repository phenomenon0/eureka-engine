// head/jevhead.c — see jevhead.h. Line references are to llama.cpp 9881906
// (~/bend-phone-prep/llama.cpp, tools/server/), the reference implementation.
#include "jevhead.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

// The Kev systemone template, verbatim from the Kev-4B / Kev-0.8B GGUFs (tokenizer.chat_template.systemone).
// jvh_render_prompt implements exactly this template; a GGUF with any other template is refused.
static const char KEV_TEMPLATE[] =
    "<|fim_prefix|>{{ state }}<|fim_middle|>{{ instructions }}{% for o in options %}<|box_start|>{% if type == "
    "'score' %}{% if o.description %}{{ o.description }}{% endif %}{% else %}{% if type != 'noul' %}{{ o.key }}{% "
    "elif o.key == 'true' %}yes{% else %}no{% endif %}{% if o.description %}: {{ o.description }}{% endif %}{% "
    "endif %}<|box_end|>{% endfor %}<|fim_suffix|>";

static const char *TYPE_NAME[3] = {"choice", "score", "noul"};

// ---- head loading ------------------------------------------------------------

static void add_temp(jvh_head *h, const char *name, float v) {
  h->temp_name = (char **)realloc(h->temp_name, sizeof(char *) * (h->n_temp + 1));
  h->temp = (float *)realloc(h->temp, sizeof(float) * (h->n_temp + 1));
  h->temp_name[h->n_temp] = strdup(name);
  h->temp[h->n_temp++] = v;
}

// raw = cls.output.weight bytes ([n_out] rows of ggml type wtype, each n_in long)
static int set_weight(jvh_head *h, int wtype, const uint8_t *raw, char *err, size_t errn) {
  size_t n = (size_t)h->n_out * h->n_in;
  h->wtype = wtype;
  h->W = (float *)malloc(n * sizeof(float));
  if (wtype == GGML_F32) {
    memcpy(h->W, raw, n * 4);
  } else if (wtype == GGML_F16 || wtype == GGML_BF16) {
    for (size_t i = 0; i < n; i++) {
      uint16_t u; memcpy(&u, raw + 2 * i, 2);
      h->W[i] = wtype == GGML_F16 ? gg_f16(u) : gg_bf16(u);
    }
  } else if (wtype == GGML_Q8_0) {
    size_t nb = n / 32;
    h->qs = (int8_t *)malloc(n);
    h->qd = (float *)malloc(nb * sizeof(float));
    for (size_t k = 0; k < nb; k++) {
      const uint8_t *blk = raw + k * 34;
      uint16_t u; memcpy(&u, blk, 2);
      h->qd[k] = gg_f16(u);
      memcpy(h->qs + k * 32, blk + 2, 32);
      for (int j = 0; j < 32; j++) h->W[k * 32 + j] = (float)h->qs[k * 32 + j] * h->qd[k];
    }
  } else {
    snprintf(err, errn, "cls.output.weight: ggml type %d unsupported (F32 F16 BF16 Q8_0)", wtype);
    return -1;
  }
  return 0;
}

static int check_dims(jvh_head *h, char *err, size_t errn) {
  if (h->n_in <= 0 || h->n_out <= 0 || h->n_out % 2 || (h->n_in % 32)) {
    snprintf(err, errn, "bad head shape n_in=%d n_out=%d", h->n_in, h->n_out);
    return -1;
  }
  h->n_ptr = h->n_out / 2;
  return 0;
}

int jvh_load_gguf(jvh_head *h, const char *path, char *err, size_t errn) {
  memset(h, 0, sizeof *h);
  gguf g;
  if (gguf_open(&g, path, err, errn)) return -1;
  int rc = -1;
  char key[256];
  const gg_kv *a = gguf_kv(&g, "general.architecture");
  if (!a || a->type != GG_STR) { snprintf(err, errn, "no general.architecture"); goto out; }
  snprintf(h->arch, sizeof h->arch, "%s", a->str);
  snprintf(key, sizeof key, "%s.decision.type", h->arch);
  const gg_kv *dt = gguf_kv(&g, key);
  if (!dt || dt->type != GG_STR || strcmp(dt->str, "kev")) {
    snprintf(err, errn, "%s is not a Kev decision model (%s = %s)", path, key, dt && dt->str ? dt->str : "absent");
    goto out;
  }
  const gg_kv *tp = gguf_kv(&g, "tokenizer.chat_template.systemone");
  if (!tp || tp->type != GG_STR || strcmp(tp->str, KEV_TEMPLATE)) {
    snprintf(err, errn, "tokenizer.chat_template.systemone differs from the Kev template this head renders");
    goto out;
  }
  snprintf(key, sizeof key, "%s.decision.temperature.", h->arch);
  for (int i = 0; i < g.n_kv; i++)
    if (!strncmp(g.kv[i].key, key, strlen(key)) && g.kv[i].type != GG_STR && g.kv[i].type != GG_ARR)
      add_temp(h, g.kv[i].key + strlen(key), (float)g.kv[i].num);
  const gg_tensor *w = gguf_tensor(&g, "cls.output.weight"), *b = gguf_tensor(&g, "cls.output.bias");
  if (!w || !b || w->n_dims != 2 || b->type != GGML_F32 || b->ne[0] != w->ne[1]) {
    snprintf(err, errn, "missing or malformed cls.output.{weight,bias}");
    goto out;
  }
  h->n_in = (int)w->ne[0];
  h->n_out = (int)w->ne[1];
  if (check_dims(h, err, errn)) goto out;
  snprintf(key, sizeof key, "%s.embedding_length", h->arch);
  const gg_kv *ne = gguf_kv(&g, key);
  if (ne && (int)ne->num != h->n_in) { snprintf(err, errn, "cls.output.weight n_in %d != %s %d", h->n_in, key, (int)ne->num); goto out; }
  size_t wb = ggml_row_bytes(w->type, w->ne[0]) * (size_t)h->n_out;
  if (!wb) { snprintf(err, errn, "cls.output.weight: ggml type %d unsupported", w->type); goto out; }
  uint8_t *raw = (uint8_t *)malloc(wb);
  h->b = (float *)malloc(sizeof(float) * h->n_out);
  if (gguf_read(&g, w, raw, wb) || gguf_read(&g, b, h->b, sizeof(float) * h->n_out)) {
    free(raw); snprintf(err, errn, "short read of cls.output.*"); goto out;
  }
  rc = set_weight(h, w->type, raw, err, errn);
  free(raw);
out:
  gguf_close(&g);
  if (rc) jvh_free(h);
  return rc;
}

static char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *p = (char *)malloc((size_t)sz + 1);
  if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return NULL; }
  p[sz] = 0;
  fclose(f);
  if (n) *n = (size_t)sz;
  return p;
}

int jvh_load_prefix(jvh_head *h, const char *prefix, char *err, size_t errn) {
  memset(h, 0, sizeof *h);
  char path[4096];
  size_t n;
  snprintf(path, sizeof path, "%s.json", prefix);
  char *txt = slurp(path, &n);
  if (!txt) { snprintf(err, errn, "cannot read %s", path); return -1; }
  jv *m = jv_parse(txt, n, err, errn);
  free(txt);
  if (!m) return -1;
  int rc = -1;
  const jv *meta = jv_get(m, "meta"), *tw = jv_get(jv_get(jv_get(m, "tensors"), "cls.output.weight"), "ggml_shape");
  const jv *ty = jv_get(jv_get(jv_get(m, "tensors"), "cls.output.weight"), "type");
  if (!meta || !tw || tw->t != JN_ARR || tw->n != 2 || !ty || ty->t != JN_STR) { snprintf(err, errn, "%s: bad layout", path); goto out; }
  for (size_t i = 0; i < meta->n; i++) {
    const char *k = meta->keys[i], *d = strstr(k, ".decision.");
    if (d && !h->arch[0]) snprintf(h->arch, sizeof h->arch, "%.*s", (int)(d - k), k);
    const char *t = strstr(k, ".decision.temperature.");
    if (t && meta->items[i]->t == JN_DBL) add_temp(h, t + strlen(".decision.temperature."), (float)meta->items[i]->d);
  }
  h->n_in = (int)tw->items[0]->u;
  h->n_out = (int)tw->items[1]->u;
  if (check_dims(h, err, errn)) goto out;
  int wtype = !strcmp(ty->s, "Q8_0") ? GGML_Q8_0 : !strcmp(ty->s, "BF16") ? GGML_BF16 : !strcmp(ty->s, "F16") ? GGML_F16
            : !strcmp(ty->s, "F32") ? GGML_F32 : -1;
  size_t wb = wtype < 0 ? 0 : ggml_row_bytes(wtype, (uint64_t)h->n_in) * (size_t)h->n_out, bn;
  snprintf(path, sizeof path, "%s.cls.output.weight.bin", prefix);
  char *raw = slurp(path, &n);
  if (!raw || !wb || n != wb) { free(raw); snprintf(err, errn, "%s: missing, wrong size or unsupported type %s", path, ty->s); goto out; }
  snprintf(path, sizeof path, "%s.cls.output.bias.bin", prefix);
  h->b = (float *)slurp(path, &bn);
  if (!h->b || bn != sizeof(float) * (size_t)h->n_out) { free(raw); snprintf(err, errn, "%s: missing or wrong size", path); goto out; }
  rc = set_weight(h, wtype, (const uint8_t *)raw, err, errn);
  free(raw);
out:
  jv_free(m);
  if (rc) jvh_free(h);
  return rc;
}

void jvh_free(jvh_head *h) {
  free(h->W); free(h->qs); free(h->qd); free(h->b);
  for (int i = 0; i < h->n_temp; i++) free(h->temp_name[i]);
  free(h->temp_name); free(h->temp);
  memset(h, 0, sizeof *h);
}

// server-decision.cpp:682 get_temperature: "<type>.<bucket>", then "<type>", then 1.0
float jvh_temperature(const jvh_head *h, int qtype, size_t n) {
  const char *bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
  char names[2][64];
  snprintf(names[0], sizeof names[0], "%s.%s", TYPE_NAME[qtype], bucket);
  snprintf(names[1], sizeof names[1], "%s", TYPE_NAME[qtype]);
  for (int k = 0; k < 2; k++)
    for (int i = 0; i < h->n_temp; i++)
      if (!strcmp(h->temp_name[i], names[k])) return h->temp[i];
  return 1.0f;
}

// ---- request parsing (server-decision.cpp:136-205 parse_questions, :227-283 parse_state) -------------

static int qerr(char *err, size_t errn, const char *id, const char *msg) {
  snprintf(err, errn, "questions.%s: %s", id, msg);
  return -1;
}

int jvh_parse_request(const char *text, size_t len, jvh_req *r, char *err, size_t errn) {
  memset(r, 0, sizeof *r);
  r->root = jv_parse(text, len, err, errn);
  if (!r->root) return -1;
  const jv *body = r->root;
  if (body->t != JN_OBJ) { snprintf(err, errn, "request must be a JSON object"); goto bad; }
  // images: llama.cpp loads them but the Kev prompt has no media; this head refuses them rather than guess
  const jv *im = jv_get(body, "images"), *vi = jv_get(body, "videos");
  if (vi && vi->t != JN_NULL && vi->n) { snprintf(err, errn, "\"videos\" is not supported"); goto bad; }
  if (im && im->t != JN_NULL && (im->t != JN_ARR || im->n)) { snprintf(err, errn, "\"images\" are not supported by this head"); goto bad; }
  r->state = jv_get(body, "state");
  if (!r->state || r->state->t == JN_NULL) { snprintf(err, errn, "\"state\" must be provided"); goto bad; }
  {  // chat-message image parts would be stripped (and loaded) by llama.cpp: refuse them
    const jv *msgs = r->state->t == JN_OBJ && jv_get(r->state, "messages") ? jv_get(r->state, "messages") : r->state;
    if (msgs->t == JN_ARR)
      for (size_t i = 0; i < msgs->n; i++) {
        const jv *c = jv_get(msgs->items[i], "content");
        if (c && c->t == JN_ARR)
          for (size_t j = 0; j < c->n; j++) {
            const jv *ty = jv_get(c->items[j], "type");
            if (ty && ty->t == JN_STR && !strcmp(ty->s, "image_url") && jv_get(c->items[j], "image_url")) {
              snprintf(err, errn, "image parts in \"state\" are not supported by this head");
              goto bad;
            }
          }
      }
  }
  const jv *qs = jv_get(body, "questions");
  if (!qs || qs->t != JN_OBJ || !qs->n) { snprintf(err, errn, "\"questions\" must be a non-empty object"); goto bad; }
  r->q = (jvh_q *)calloc(qs->n, sizeof(jvh_q));
  for (size_t k = 0; k < qs->n; k++) {
    const char *id = qs->keys[k];
    const jv *q = qs->items[k];
    jvh_q *o = &r->q[r->nq++];
    o->id = strdup(id);
    if (q->t != JN_OBJ) { qerr(err, errn, id, "must be an object"); goto bad; }
    o->instructions = jv_get(q, "instructions");
    if (!o->instructions || o->instructions->t == JN_NULL) { qerr(err, errn, id, "\"instructions\" must be provided"); goto bad; }
    const jv *ty = jv_get(q, "type"), *cr = jv_get(q, "criteria");
    const char *tn = ty && ty->t == JN_STR ? ty->s : "";
    if (!strcmp(tn, "choice")) {
      o->type = JVH_CHOICE;
      if (!cr || cr->t != JN_OBJ || !cr->n) { qerr(err, errn, id, "\"criteria\" must be a non-empty object"); goto bad; }
      o->opt = (jvh_opt *)calloc(cr->n, sizeof(jvh_opt));
      for (size_t i = 0; i < cr->n; i++) {  // insertion order; Kev does not sort (Clef does)
        o->opt[o->n].key = strdup(cr->keys[i]);
        o->opt[o->n++].desc = cr->items[i];
      }
    } else if (!strcmp(tn, "score")) {
      o->type = JVH_SCORE;
      if (!cr || cr->t != JN_ARR || cr->n < 2 || cr->n > 10) { qerr(err, errn, id, "\"criteria\" must be an array of 2 to 10 levels"); goto bad; }
      o->opt = (jvh_opt *)calloc(cr->n, sizeof(jvh_opt));
      for (size_t i = 0; i < cr->n; i++) {
        char num[24]; snprintf(num, sizeof num, "%zu", i);
        o->opt[o->n].key = strdup(num);
        o->opt[o->n++].desc = cr->items[i];
      }
    } else if (!strcmp(tn, "noul")) {
      o->type = JVH_NOUL;
      if (cr && cr->t != JN_NULL && cr->t != JN_OBJ) { qerr(err, errn, id, "\"criteria\" must be an object"); goto bad; }
      o->opt = (jvh_opt *)calloc(2, sizeof(jvh_opt));
      const char *keys[2] = {"false", "true"};  // Kev: no noul_true_first
      for (int i = 0; i < 2; i++) {
        o->opt[o->n].key = strdup(keys[i]);
        o->opt[o->n++].desc = cr && cr->t == JN_OBJ ? jv_get(cr, keys[i]) : NULL;
      }
    } else {
      qerr(err, errn, id, "\"type\" must be one of: choice, score, noul");
      goto bad;
    }
    if (o->n > 255) {
      char m[96]; snprintf(m, sizeof m, "too many options (%zu), this model supports at most 255", o->n);
      qerr(err, errn, id, m);
      goto bad;
    }
  }
  return 0;
bad:
  jvh_req_free(r);
  return -1;
}

void jvh_req_free(jvh_req *r) {
  for (size_t k = 0; k < r->nq; k++) {
    for (size_t i = 0; i < r->q[k].n; i++) free(r->q[k].opt[i].key);
    free(r->q[k].opt); free(r->q[k].id);
  }
  free(r->q);
  jv_free(r->root);
  memset(r, 0, sizeof *r);
}

// ---- prompt (server-decision.cpp:326-360 decision_kev_render / decision_kev_text, :380-440 render) ---------

static void kev_render(sbuf *o, const jv *v, int indent) {
  switch (v->t) {
    case JN_NULL: return;
    case JN_STR: sb_putn(o, v->s, v->slen); return;
    case JN_BOOL: sb_puts(o, v->b ? "True" : "False"); return;
    case JN_ARR:
      for (size_t i = 0; i < v->n; i++) {
        sbuf t = {0};
        kev_render(&t, v->items[i], indent + 1);
        size_t s = 0;
        while (s < t.n && (t.p[s] == ' ' || t.p[s] == '\t' || t.p[s] == '\n' || t.p[s] == '\r')) s++;
        if (o->n) sb_putc(o, '\n');
        for (int k = 0; k < 2 * indent; k++) sb_putc(o, ' ');
        sb_puts(o, "- ");
        sb_putn(o, t.p ? t.p + s : "", t.n - s);
        free(t.p);
      }
      return;
    case JN_OBJ:
      for (size_t i = 0; i < v->n; i++) {
        const jv *it = v->items[i];
        int nested = it->t == JN_OBJ || it->t == JN_ARR;
        if (o->n) sb_putc(o, '\n');
        for (int k = 0; k < 2 * indent; k++) sb_putc(o, ' ');
        sb_puts(o, v->keys[i]);
        sb_puts(o, nested ? ":\n" : ": ");
        // the nested value is rendered on its own (the C++ concatenates a fresh string), so render into a
        // temporary: its own "out.empty()" test must not see this object's text
        sbuf t = {0};
        kev_render(&t, it, nested ? indent + 1 : 0);
        sb_putn(o, t.p ? t.p : "", t.n);
        free(t.p);
      }
      return;
    default: jv_dump(o, v); return;  // numbers: nlohmann dump()
  }
}

// std::regex_replace(text, "<\\|([A-Za-z0-9_]+)\\|>", "<\xC2\xA6$1\xC2\xA6>")
static char *neutralise(const char *s, size_t n) {
  sbuf o = {0};
  size_t i = 0;
  while (i < n) {
    if (s[i] == '<' && i + 1 < n && s[i + 1] == '|') {
      size_t j = i + 2;
      while (j < n && ((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z') || (s[j] >= '0' && s[j] <= '9') || s[j] == '_')) j++;
      if (j > i + 2 && j + 1 < n && s[j] == '|' && s[j + 1] == '>') {
        sb_puts(&o, "<\xC2\xA6");
        sb_putn(&o, s + i + 2, j - i - 2);
        sb_puts(&o, "\xC2\xA6>");
        i = j + 2;
        continue;
      }
    }
    sb_putc(&o, s[i++]);
  }
  return sb_take(&o);
}

char *jvh_kev_text(const jv *v) {
  sbuf t = {0};
  kev_render(&t, v, 0);
  char *r = neutralise(t.p ? t.p : "", t.n);
  free(t.p);
  return r;
}

static char *kev_text_str(const char *s) {
  jv tmp = {0};
  tmp.t = JN_STR; tmp.s = (char *)s; tmp.slen = strlen(s);
  return jvh_kev_text(&tmp);
}

char *jvh_render_prompt(const jvh_req *r, const jvh_q *q) {
  sbuf o = {0};
  char *st = jvh_kev_text(r->state), *in = jvh_kev_text(q->instructions);
  sb_puts(&o, "<|fim_prefix|>"); sb_puts(&o, st);
  sb_puts(&o, "<|fim_middle|>"); sb_puts(&o, in);
  free(st); free(in);
  for (size_t i = 0; i < q->n; i++) {
    // option["description"] = kev_text(desc) unless null; the template tests its truthiness (non-empty string)
    char *d = q->opt[i].desc && q->opt[i].desc->t != JN_NULL ? jvh_kev_text(q->opt[i].desc) : NULL;
    int has_d = d && d[0];
    sb_puts(&o, "<|box_start|>");
    if (q->type == JVH_SCORE) {
      if (has_d) sb_puts(&o, d);
    } else {
      if (q->type == JVH_CHOICE) { char *k = kev_text_str(q->opt[i].key); sb_puts(&o, k); free(k); }
      else sb_puts(&o, !strcmp(q->opt[i].key, "true") ? "yes" : "no");
      if (has_d) { sb_puts(&o, ": "); sb_puts(&o, d); }
    }
    sb_puts(&o, "<|box_end|>");
    free(d);
  }
  sb_puts(&o, "<|fim_suffix|>");
  return sb_take(&o);
}

// ---- head math ------------------------------------------------------------------

static float round_bf16(float x) {  // ggml fp32 -> bf16, round to nearest even
  uint32_t u; memcpy(&u, &x, 4);
  if ((u & 0x7fffffff) > 0x7f800000) u |= 0x00400000;  // quiet NaN
  else u = (u + (0x7fff + ((u >> 16) & 1))) & 0xffff0000u;
  float f; memcpy(&f, &u, 4);
  return f;
}
static float round_f16(float x) {  // f32 -> f16 -> f32, round to nearest even (normal range is all we need)
  if (x == 0 || !isfinite(x)) return x;
  int e; frexpf(fabsf(x), &e);  // |x| = m * 2^e, m in [0.5, 1)
  int ulp_e = (e - 1 < -14 ? -14 : e - 1) - 10;
  float q = ldexpf(1.0f, ulp_e);
  return rintf(x / q) * q;
}

void jvh_project(const jvh_head *h, const float *x, float *e, int mode) {
  const int n = h->n_in, nb = n / 32;
  if (mode == JVH_CPU && h->wtype == GGML_Q8_0) {
    // ggml-cpu: the activation row is quantized to Q8_0 (d = amax/127 stored as f16; q = round-half-even(x/d)),
    // int8 block dots, acc += isum * (d_w * d_a) over blocks in order
    int8_t *xq = (int8_t *)malloc((size_t)n);
    float *xd = (float *)malloc(sizeof(float) * nb);
    for (int k = 0; k < nb; k++) {
      float amax = 0;
      for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[k * 32 + j]));
      float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
      xd[k] = round_f16(d);
      for (int j = 0; j < 32; j++) xq[k * 32 + j] = (int8_t)rintf(x[k * 32 + j] * id);
    }
    for (int o = 0; o < h->n_out; o++) {
      float acc = 0;
      for (int k = 0; k < nb; k++) {
        int isum = 0;
        const int8_t *wq = h->qs + (size_t)o * n + k * 32, *aq = xq + k * 32;
        for (int j = 0; j < 32; j++) isum += wq[j] * aq[j];
        acc += (float)isum * (h->qd[(size_t)o * nb + k] * xd[k]);
      }
      e[o] = acc + h->b[o];
    }
    free(xq); free(xd);
    return;
  }
  for (int o = 0; o < h->n_out; o++) {
    const float *w = h->W + (size_t)o * n;
    double acc = 0;
    if (mode == JVH_CPU && h->wtype == GGML_BF16)
      for (int i = 0; i < n; i++) acc += (double)w[i] * (double)round_bf16(x[i]);
    else
      for (int i = 0; i < n; i++) acc += (double)w[i] * (double)x[i];
    e[o] = (float)acc + h->b[o];
  }
}

// server-context.cpp:2386-2392: f32 accumulate in index order, then / sqrtf(n_pointer)
void jvh_scores(const jvh_head *h, const float *ep, const float *const *em, size_t n, float *s) {
  const int np = h->n_ptr;
  for (size_t m = 0; m < n; m++) {
    float dot = 0.0f;
    for (int i = 0; i < np; i++) dot += ep[i] * em[m][np + i];
    s[m] = dot / sqrtf((float)np);
  }
}

// server-decision.cpp:735-756 (one variant)
void jvh_probs(const float *s, size_t n, float T, double *p) {
  float smax = s[0];
  for (size_t i = 1; i < n; i++) if (s[i] > smax) smax = s[i];
  double sum = 0;
  for (size_t i = 0; i < n; i++) { p[i] = exp((double)(s[i] - smax) / T); sum += p[i]; }
  for (size_t i = 0; i < n; i++) p[i] = p[i] / sum;  // Kev has one variant: / scores.size() == / 1
}

static size_t argmax_first(const double *p, size_t n) {  // std::max_element: first of equal maxima
  size_t b = 0;
  for (size_t i = 1; i < n; i++) if (p[i] > p[b]) b = i;
  return b;
}

// server-decision.cpp:711-790 format_answer + the TypeSafe confidence formulas
void jvh_answer(sbuf *o, const jvh_q *q, const double *p) {
  const size_t n = q->n;
  sb_puts(o, "{\"type\":");
  jv_dump_str(o, TYPE_NAME[q->type], strlen(TYPE_NAME[q->type]));
  if (q->type == JVH_NOUL) {
    for (size_t i = 0; i < n; i++)
      if (!strcmp(q->opt[i].key, "true")) { sb_puts(o, ",\"noul\":"); jv_dump_double(o, p[i]); }
    sb_putc(o, '}');
    return;
  }
  const size_t best = argmax_first(p, n);
  double conf;
  if (q->type == JVH_CHOICE) {
    sb_puts(o, ",\"choice\":");
    jv_dump_str(o, q->opt[best].key, strlen(q->opt[best].key));
    const double u = 1.0 / n;
    conf = n < 2 ? 1.0 : fmax(0.0, (p[best] - u) / (1.0 - u));
  } else {
    double expected = 0, dist = 0, dist_u = 0;
    for (size_t i = 0; i < n; i++) expected += i * p[i];
    for (size_t i = 0; i < n; i++) {
      dist += p[i] * fabs((double)i - (double)best);
      dist_u += fabs((double)i - (n - 1) / 2.0) / n;
    }
    conf = n < 2 ? 1.0 : fmax(0.0, 1.0 - dist / dist_u);
    sb_puts(o, ",\"score\":");
    jv_dump_double(o, expected);
    sb_puts(o, ",\"legend\":{");
    for (size_t i = 0; i < n; i++) {
      if (i) sb_putc(o, ',');
      jv_dump_str(o, q->opt[i].key, strlen(q->opt[i].key));
      sb_putc(o, ':');
      if (q->opt[i].desc) jv_dump(o, q->opt[i].desc); else sb_puts(o, "null");
    }
    sb_putc(o, '}');
  }
  sb_puts(o, ",\"probabilities\":{");
  for (size_t i = 0; i < n; i++) {
    if (i) sb_putc(o, ',');
    jv_dump_str(o, q->opt[i].key, strlen(q->opt[i].key));
    sb_putc(o, ':');
    jv_dump_double(o, p[i]);
  }
  sb_puts(o, "},\"confidence\":");
  jv_dump_double(o, conf);
  sb_putc(o, '}');
}

void jvh_answer_rows(sbuf *out, const jvh_head *h, const jvh_q *q, const float *rows, int mode, float *s_out, double *p_out) {
  const size_t n = q->n;
  float *e = (float *)malloc(sizeof(float) * h->n_out * (n + 1));
  const float **em = (const float **)malloc(sizeof(float *) * n);
  for (size_t r = 0; r <= n; r++) jvh_project(h, rows + r * h->n_in, e + r * h->n_out, mode);
  for (size_t m = 0; m < n; m++) em[m] = e + (m + 1) * h->n_out;
  float *s = s_out ? s_out : (float *)malloc(sizeof(float) * n);
  double *p = p_out ? p_out : (double *)malloc(sizeof(double) * n);
  jvh_scores(h, e, em, n, s);
  jvh_probs(s, n, jvh_temperature(h, q->type, n), p);
  jvh_answer(out, q, p);
  if (!s_out) free(s);
  if (!p_out) free(p);
  free(e); free(em);
}

int jvh_rows_read(const char *path, int width, uint32_t **pos, float **rows, char *err, size_t errn) {
  FILE *f = fopen(path, "rb");
  if (!f) { snprintf(err, errn, "cannot open %s", path); return -1; }
  int n = 0;
  *pos = NULL; *rows = NULL;
  uint32_t hdr[2];
  while (fread(hdr, 4, 2, f) == 2) {
    if ((int)hdr[1] != width) { snprintf(err, errn, "%s: row width %u != %d", path, hdr[1], width); goto bad; }
    *pos = (uint32_t *)realloc(*pos, sizeof(uint32_t) * (n + 1));
    *rows = (float *)realloc(*rows, sizeof(float) * (size_t)width * (n + 1));
    (*pos)[n] = hdr[0];
    if (fread(*rows + (size_t)n * width, 4, (size_t)width, f) != (size_t)width) { snprintf(err, errn, "%s: truncated", path); goto bad; }
    if (n && (*pos)[n] <= (*pos)[n - 1]) { snprintf(err, errn, "%s: positions not increasing", path); goto bad; }
    n++;
  }
  fclose(f);
  return n;
bad:
  fclose(f);
  free(*pos); free(*rows);
  *pos = NULL; *rows = NULL;
  return -1;
}
