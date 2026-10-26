// kevhead.c: the Kev-4B decide head for dualm.c (KEV-HEAD-METAL). Followed: tools/kevserve/kev.c (the hardened
// tools/kevdual/kev.c: same render and readout bytes, plus the token-overflow refusal); its JSON-nesting guard and
// "mode" check stay in dualm.c's own parser. Render = assemble.py (fim/box markers), readout = kevhead.c (same f32 order).
// The answer's numbers are kev.c's; its JSON is the Metal clef response's (and the capfix llama-server ruler rows'):
// full precision, score with its legend, usage input/output tokens. kev.c's 4-decimal round_prob is not applied.
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "kevhead.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void clef_fail(const char *msg);  // spans.c: the engine's fatal exit

enum { KMAX = 255, TMAX = 16384, QMAX = 64 };

// ---------------------------------------------------------------- GGUF (header, KVs, tensor infos only)

typedef struct {
    int found, type;
    uint64_t dims[4], off;
    int nd;
} GT;

typedef struct {
    char arch[64], dtype[64];
    uint64_t d, nout;
    float temp[3];
    int ntemp;
    GT w, b;
    uint64_t data;  // start of the tensor data
} GInfo;

static const char *TYPES[3] = {"noul", "choice", "score"};

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }
static uint64_t u64(FILE *f, int *ok) { uint64_t v = 0; *ok &= rd(f, &v, 8); return v; }
static uint32_t u32(FILE *f, int *ok) { uint32_t v = 0; *ok &= rd(f, &v, 4); return v; }

static char *gstr(FILE *f, int *ok, int keep) {  // keep = 0: skipped, NULL back
    uint64_t n = u64(f, ok);
    if (!*ok || n > (1u << 30)) return *ok = 0, NULL;
    if (!keep) { *ok &= !fseeko(f, (off_t)n, SEEK_CUR); return NULL; }
    char *s = malloc(n + 1);
    if (!s || !rd(f, s, n)) return free(s), *ok = 0, NULL;
    s[n] = 0;
    return s;
}

static const int GSZ[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

static void gskip(FILE *f, uint32_t t, int *ok) {
    if (t == 8) { gstr(f, ok, 0); return; }
    if (t == 9) {
        uint32_t et = u32(f, ok);
        uint64_t n = u64(f, ok);
        if (!*ok || et > 12) { *ok = 0; return; }
        if (et == 8 || et == 9) for (uint64_t i = 0; i < n && *ok; i++) gskip(f, et, ok);
        else *ok &= !fseeko(f, (off_t)(n * GSZ[et]), SEEK_CUR);
        return;
    }
    if (t > 12) { *ok = 0; return; }
    *ok &= !fseeko(f, GSZ[t], SEEK_CUR);
}

static int ends(const char *k, const char *arch, const char *tail) {  // k == "<arch>.<tail>"
    size_t a = strlen(arch);
    return !strncmp(k, arch, a) && k[a] == '.' && !strcmp(k + a + 1, tail);
}

static int gguf_info(const char *path, GInfo *g) {
    memset(g, 0, sizeof *g);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = 1;
    char magic[4];
    ok &= rd(f, magic, 4) && !memcmp(magic, "GGUF", 4);
    uint32_t ver = u32(f, &ok);
    uint64_t nt = u64(f, &ok), nkv = u64(f, &ok), align = 32;
    ok &= ver >= 2;
    for (uint64_t i = 0; i < nkv && ok; i++) {
        char *k = gstr(f, &ok, 1);
        uint32_t t = u32(f, &ok);
        if (!ok) { free(k); break; }
        if (!strcmp(k, "general.architecture") && t == 8) {
            char *v = gstr(f, &ok, 1);
            if (v) snprintf(g->arch, sizeof g->arch, "%s", v);
            free(v);
        } else if (*g->arch && ends(k, g->arch, "decision.type") && t == 8) {
            char *v = gstr(f, &ok, 1);
            if (v) snprintf(g->dtype, sizeof g->dtype, "%s", v);
            free(v);
        } else if (*g->arch && (ends(k, g->arch, "embedding_length") || ends(k, g->arch, "embedding_length_out")) && (t == 4 || t == 10)) {
            uint64_t v = t == 4 ? u32(f, &ok) : u64(f, &ok);
            if (ends(k, g->arch, "embedding_length")) g->d = v; else g->nout = v;
        } else if (!strcmp(k, "general.alignment") && t == 4) {
            align = u32(f, &ok);
        } else if (*g->arch && t == 6 && !strncmp(k, g->arch, strlen(g->arch)) && !strncmp(k + strlen(g->arch), ".decision.temperature.", 22)) {
            float v;
            ok &= rd(f, &v, 4);
            for (int j = 0; j < 3; j++) if (!strcmp(k + strlen(g->arch) + 22, TYPES[j])) g->temp[j] = v, g->ntemp |= 1 << j;
        } else gskip(f, t, &ok);
        free(k);
    }
    for (uint64_t i = 0; i < nt && ok; i++) {
        char *n = gstr(f, &ok, 1);
        GT x = {0};
        x.nd = (int)u32(f, &ok);
        if (!ok || x.nd < 1 || x.nd > 4) { free(n); ok = 0; break; }
        for (int j = 0; j < x.nd; j++) x.dims[j] = u64(f, &ok);
        x.type = (int)u32(f, &ok);
        x.off = u64(f, &ok);
        x.found = 1;
        if (!strcmp(n, "cls.output.weight")) g->w = x;
        if (!strcmp(n, "cls.output.bias")) g->b = x;
        free(n);
    }
    off_t pos = ftello(f);
    fclose(f);
    if (!ok || !align) return 0;
    g->data = ((uint64_t)pos + align - 1) / align * align;
    return 1;
}

// the Kev head, checked: NULL = complete, else what is missing or malformed
static const char *kev_check(const GInfo *g) {
    if (strcmp(g->dtype, "kev")) return "decision.type is not \"kev\"";
    if (!g->w.found || !g->b.found) return "cls.output.weight/bias missing";
    if (g->ntemp != 7) return "decision.temperature.{noul,choice,score} missing";
    if (!g->d || g->d % 32 || !g->nout || g->nout % 2) return "embedding_length / embedding_length_out missing or odd";
    if (g->w.type != 8 || g->w.nd != 2 || g->w.dims[0] != g->d || g->w.dims[1] != g->nout) return "cls.output.weight is not Q8_0 [embedding_length, embedding_length_out]";
    if (g->b.type != 0 || g->b.nd != 1 || g->b.dims[0] != g->nout) return "cls.output.bias is not F32 [embedding_length_out]";
    for (int j = 0; j < 3; j++) if (!(g->temp[j] > 0)) return "a decision temperature is not positive";
    return NULL;
}

int kev_head_probe(const char *gguf, char *why, size_t why_n) {
    GInfo g;
    if (why && why_n) *why = 0;
    if (!gguf_info(gguf, &g)) return 0;
    const char *e = kev_check(&g);
    if (e && why && (!strcmp(g.dtype, "kev") || g.w.found || g.b.found)) snprintf(why, why_n, "%s", e);  // a partial Kev head: say so
    return !e;
}

// ---------------------------------------------------------------- render (assemble.py)

static const char *SPECIAL[5] = {"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"};
static Q35Tokenizer *tok;
static int sid[5];  // P M BS BE S
static double temp[3];
static int D, N, DP;
static size_t RB;
static uint8_t *wq;
static float *bias;

typedef struct {
    char *id;
    const jv *crit;
    int type, n, k, ids[TMAX], at[KMAX + 2];  // at = markers..., pointer
    char *keys[KMAX];
} Q;

static Q *qs;
static int nq, total, raws_closed;
static sb raws;

static int is_box(const jv *v) { return v->type == J_ARR || v->type == J_OBJ; }

static void lstrip_add(sb *o, sb *x) {
    char *p = x->p ? x->p : "";
    while (*p == ' ' || (*p >= '\t' && *p <= '\r')) p++;
    sb_add(o, p);
}

static void render(sb *o, const jv *v, int indent) {  // kev/api.py render
    if (v->type == J_NULL) return;
    if (v->type == J_STR) { sb_add(o, v->str); return; }
    if (v->type == J_NUM) { sb_add(o, v->raw); return; }  // kev.c ponytail: integers as written
    if (v->type == J_BOOL) { sb_add(o, v->b ? "True" : "False"); return; }
    for (size_t i = 0; i < v->n; i++) {
        sb x = {0};
        if (i) sb_addc(o, '\n');
        for (int j = 0; j < indent; j++) sb_add(o, "  ");
        if (v->type == J_ARR) {
            render(&x, v->items[i], indent + 1);
            sb_add(o, "- ");
            lstrip_add(o, &x);
        } else {  // key, marker, value appended whole: sb_fmt's 512-byte buffer cut long values and blocks
            render(&x, v->items[i], is_box(v->items[i]) ? indent + 1 : 0);
            sb_add(o, v->keys[i]);
            sb_add(o, is_box(v->items[i]) ? ":\n" : ": ");
            sb_add(o, x.p ? x.p : "");
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

// escape(render(v)) or escape(name[: render(desc)]) (option_text) as plain-text ids, appended
static int piece(Q *q, const char *name, const jv *v) {
    sb r = {0}, e = {0};
    if (name) sb_add(&r, name);
    if (v && !(v->type == J_NULL || (v->type == J_STR && !*v->str))) {
        if (name) sb_add(&r, ": ");
        render(&r, v, 0);
    }
    escape(&e, r.p ? r.p : "");
    int c = e.p ? q35_tok_encode(tok, e.p, 0, q->ids + q->n, TMAX - q->n) : 0;
    free(r.p), free(e.p);
    if (c < 0 || c > TMAX - q->n) return 0;  // kevserve: over cap refused
    q->n += c;
    return 1;
}

static int push(Q *q, int id) {
    if (q->n >= TMAX) return 0;
    q->ids[q->n++] = id;
    return 1;
}

// one question -> ids, markers + pointer, keys; NULL = ok, else the refusal (kev.c build, unchanged)
static const char *build(Q *q, const jv *state, const char *qid, const jv *b) {
    const jv *t = json_get(b, "type"), *c = json_get(b, "criteria"), *ins = json_get(b, "instructions");
    q->id = strdup(qid);
    q->n = q->k = 0;
    q->crit = c;
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

void kev_head_open(const char *gguf, Q35Tokenizer *t) {
    GInfo g;
    if (!gguf_info(gguf, &g)) clef_fail("kev: cannot read the GGUF header");
    const char *e = kev_check(&g);
    if (e) clef_fail(e);
    D = (int)g.d, N = (int)g.nout, DP = N / 2, RB = (size_t)D / 32 * 34;
    for (int j = 0; j < 3; j++) temp[j] = g.temp[j];
    tok = t;
    for (int i = 0; i < 5; i++) {
        int id[2];
        if (q35_tok_encode(tok, SPECIAL[i], 1, id, 2) != 1) clef_fail("kev: render marker missing from the vocab");
        sid[i] = id[0];
    }
    wq = malloc(RB * N);
    bias = malloc(sizeof *bias * N);
    qs = calloc(QMAX, sizeof *qs);
    FILE *f = fopen(gguf, "rb");
    if (!wq || !bias || !qs || !f) clef_fail("kev: head allocation/open failed");
    if (fseeko(f, (off_t)(g.data + g.w.off), SEEK_SET) || fread(wq, 1, RB * N, f) != RB * N
        || fseeko(f, (off_t)(g.data + g.b.off), SEEK_SET) || fread(bias, 4, N, f) != (size_t)N) clef_fail("kev: cls.output read failed");
    fclose(f);
    for (int i = 0; i < N; i++) if (!isfinite(bias[i])) clef_fail("kev: nonfinite cls.output.bias");
    fprintf(stderr, "sys1: kev head d %d out %d (T noul %.9g choice %.9g score %.9g)\n", D, N, temp[0], temp[1], temp[2]);
}

void kev_drop(void) {
    for (int i = 0; i < nq; i++) {
        free(qs[i].id);
        for (int j = 0; j < qs[i].k; j++) free(qs[i].keys[j]);
        qs[i].id = NULL, qs[i].k = 0;
    }
    free(raws.p);
    raws = (sb){0};
    nq = total = raws_closed = 0;
}

const char *kev_request(const jv *req, unsigned ctx) {
    const jv *state = json_get(req, "state"), *qo = json_get(req, "questions");
    kev_drop();
    if (!state || !qo || qo->type != J_OBJ || qo->n < 1 || qo->n > QMAX) return "decide needs state and questions (object, 1..64)";
    for (nq = 0; nq < (int)qo->n; nq++) {
        if (qo->items[nq]->type != J_OBJ) return "each question must be an object";
        const char *e = build(&qs[nq], state, qo->keys[nq], qo->items[nq]);
        if (e) return nq++, e;
        if ((unsigned)qs[nq].n > ctx) return nq++, "question exceeds the KV capacity fixed at launch";
    }
    return NULL;
}

int kev_count(void) { return nq; }
int kev_tokens(void) { return total; }
const char *kev_raw(void) {  // the object kev_answer opened, closed on the first read
    if (!raws.p) return "{}";
    if (!raws_closed) sb_addc(&raws, '}'), raws_closed = 1;
    return raws.p;
}

const int *kev_ids(int qi, int *n) {
    *n = qs[qi].n;
    return qs[qi].ids;
}

// ---------------------------------------------------------------- readout (kevhead.c)

static void head(const float *h, float *y) {  // y = W h + b, f32 in index order (decide.c)
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
        if (!isfinite(y[i])) clef_fail("kev: nonfinite head output");
    }
}

// the hook's file (every prompt row, [pos, width] + f32 row each) -> the K+1 marker rows, in marker order
static float *rows(const Q *q, const char *path) {
    float *h = malloc(sizeof *h * D * (q->k + 1)), *skip = malloc(sizeof *skip * D);
    FILE *f = fopen(path, "rb");
    if (!h || !skip || !f) clef_fail("kev: cannot read the hidden rows");
    for (int i = 0, m = 0; i < q->n; i++) {
        uint32_t hd[2];
        float *dst = m <= q->k && i == q->at[m] ? h + (size_t)m * D : skip;
        if (fread(hd, 4, 2, f) != 2 || hd[0] != (uint32_t)i || hd[1] != (uint32_t)D || fread(dst, 4, D, f) != (size_t)D) clef_fail("kev: hidden rows: bad dump");
        if (dst == skip) continue;
        for (int j = 0; j < D; j++) if (!isfinite(dst[j])) clef_fail("kev: nonfinite hidden row");
        m++;
    }
    if (fgetc(f) != EOF) clef_fail("kev: extra hidden rows");
    fclose(f);
    free(skip);
    return h;
}

void kev_answer(int qi, const char *hidden_path, sb *o) {
    const Q *q = &qs[qi];
    int K = q->k;
    float *h = rows(q, hidden_path), *yq = malloc(sizeof *yq * N), *yk = malloc(sizeof *yk * N);
    double s[KMAX], p[KMAX];
    head(h + (size_t)K * D, yq);
    for (int i = 0; i < K; i++) {
        head(h + (size_t)i * D, yk);
        double acc = 0;
        for (int j = 0; j < DP; j++) acc += (double)yk[DP + j] * yq[j];  // k(h_opt) . q(h_decide)
        s[i] = acc / sqrt((double)DP);
    }
    free(h), free(yq), free(yk);
    double T = temp[q->type], mx = s[0], sum = 0, best = 0, ex = 0;
    int arg = 0;
    for (int i = 1; i < K; i++) if (s[i] > mx) mx = s[i];
    for (int i = 0; i < K; i++) sum += p[i] = exp((s[i] - mx) / T);
    for (int i = 0; i < K; i++) {
        p[i] /= sum;
        if (!isfinite(p[i])) clef_fail("kev: nonfinite probability");
        if (p[i] > best) best = p[i], arg = i;
        ex += i * p[i];
    }
    sb r = {0};
    sb_add(&r, "raw");
    for (int i = 0; i < K; i++) sb_fmt(&r, " %s s=%.9g p=%.9g", q->keys[i], s[i], p[i]);
    sb_add(&raws, qi ? "," : "{");
    json_quote(&raws, q->id);
    sb_addc(&raws, ':');
    json_quote(&raws, r.p);
    free(r.p);
    total += q->n;
    // the answer: the Metal clef response's fields, kev.c's values
    if (qi) sb_addc(o, ',');
    json_quote(o, q->id);
    sb_add(o, ":{\"type\":");
    json_quote(o, TYPES[q->type]);
    if (q->type == 0) { sb_fmt(o, ",\"noul\":%.17g}", p[1]); return; }  // P(true): key "true" is option 1
    double conf;
    if (q->type == 1) {
        sb_add(o, ",\"choice\":");
        json_quote(o, q->keys[arg]);
        conf = K == 1 ? 1 : (best - 1.0 / K) / (1 - 1.0 / K);
    } else {
        double dist = 0, dU = 0;
        for (int i = 0; i < K; i++) dist += p[i] * fabs((double)(i - arg)), dU += fabs(i - (K - 1) / 2.0) / K;
        conf = K == 1 ? 1 : fmax(0, 1 - dist / dU);
        sb_fmt(o, ",\"score\":%.17g,\"legend\":{", ex);
        for (int i = 0; i < K; i++) {
            if (i) sb_addc(o, ',');
            json_quote(o, q->keys[i]);
            sb_addc(o, ':');
            json_dump(o, q->crit->items[i]);
        }
        sb_addc(o, '}');
    }
    sb_fmt(o, ",\"confidence\":%.17g,\"probabilities\":{", conf);
    for (int i = 0; i < K; i++) {
        if (i) sb_addc(o, ',');
        json_quote(o, q->keys[i]);
        sb_fmt(o, ":%.17g", p[i]);
    }
    sb_add(o, "}}");
}

#ifdef KEV_HEAD_TEST
// kevhead-test: (1) render  TOKDIR GGUF REQUEST.json -> per question "qid n markers pointer T ids..." (kev.c KEV_MAIN's
// kevrender format, for a byte diff against it); (2) head GGUF -> the head on fixed toy rows (no GPU): answers + raw.
void clef_fail(const char *msg) { fprintf(stderr, "kevhead-test: %s\n", msg); exit(1); }

int main(int argc, char **argv) {
    static Q35Tokenizer tk;
    if (argc == 5 && !strcmp(argv[1], "render")) {
        if (q35_tok_load(&tk, argv[2])) clef_fail("tokenizer.bin");
        kev_head_open(argv[3], &tk);
        const char *err = NULL;
        char *text = json_read_file(argv[4]);
        jv *r = text ? json_parse(text, &err) : NULL;
        const char *e = r ? kev_request(r, TMAX) : err;
        if (e) clef_fail(e);
        for (int i = 0; i < nq; i++) {
            printf("%s %d ", qs[i].id, qs[i].n);
            for (int j = 0; j < qs[i].k; j++) printf(j ? ",%d" : "%d", qs[i].at[j]);
            printf(" %d %.17g", qs[i].at[qs[i].k], temp[qs[i].type]);
            for (int j = 0; j < qs[i].n; j++) printf(" %d", qs[i].ids[j]);
            printf("\n");
        }
        return 0;
    }
    if (argc == 6 && !strcmp(argv[1], "head")) {  // head TOKDIR GGUF REQUEST.json TMPFILE: toy rows h[pos][j] = sin(pos*0.37 + j*0.011)
        if (q35_tok_load(&tk, argv[2])) clef_fail("tokenizer.bin");
        kev_head_open(argv[3], &tk);
        const char *err = NULL;
        char *text = json_read_file(argv[4]);
        jv *r = text ? json_parse(text, &err) : NULL;
        const char *e = r ? kev_request(r, TMAX) : err;
        if (e) clef_fail(e);
        sb o = {0};
        sb_add(&o, "{\"answers\":{");
        float *row = malloc(sizeof *row * D);
        for (int i = 0; i < nq; i++) {
            FILE *f = fopen(argv[5], "wb");
            for (int t = 0; t < qs[i].n; t++) {
                uint32_t hd[2] = {(uint32_t)t, (uint32_t)D};
                for (int j = 0; j < D; j++) row[j] = (float)sin(t * 0.37 + j * 0.011);
                fwrite(hd, 4, 2, f), fwrite(row, 4, D, f);
            }
            fclose(f);
            kev_answer(i, argv[5], &o);
        }
        sb_fmt(&o, "},\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}", total);
        printf("%s\n%s\n", o.p, kev_raw());
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "probe")) {
        char why[256];
        int k = kev_head_probe(argv[2], why, sizeof why);
        printf("kev=%d%s%s\n", k, *why ? " partial: " : "", why);
        return 0;
    }
    clef_fail("usage: kevhead-test render TOKDIR GGUF REQ | head TOKDIR GGUF REQ TMP | probe GGUF");
}
#endif
