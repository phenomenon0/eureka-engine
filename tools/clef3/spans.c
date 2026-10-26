#define _POSIX_C_SOURCE 200809L
#include "spans.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

jmp_buf *clef_trap;
const char *clef_trap_msg;
void clef_fail(const char *s) {
    if (clef_trap) { clef_trap_msg = s; longjmp(*clef_trap, 1); }
    fprintf(stderr, "clef: %s\n", s); exit(1);
}
// numbers as llama-server's template tojson writes them: nlohmann int64 (uint64 read through get<int64_t>),
// anything else a double printed by std::ostream (%.6g), so 4.0 -> 4 and 1234567.0 -> 1.23457e+06
static void tojson_num(sb *s, const jv *v) {
    const char *r = v->raw ? v->raw : "0";
    if (!strpbrk(r, ".eE")) {
        errno = 0; long long i = strtoll(r, NULL, 10);
        if (!errno) { sb_fmt(s, "%lld", i); return; }
        errno = 0; unsigned long long u = strtoull(r, NULL, 10);
        if (!errno && r[0] != '-') { sb_fmt(s, "%lld", (long long) u); return; }
    }
    sb_fmt(s, "%.6g", v->num);
}
static void sorted_json(sb *s, const jv *v) {
    if (v->type == J_NUM) { tojson_num(s, v); return; }
    if (v->type == J_OBJ) {
        size_t *ix = calloc(v->n + 1, sizeof *ix);
        for (size_t i = 0; i < v->n; i++) {
            size_t j = i;
            while (j && strcmp(v->keys[ix[j-1]], v->keys[i]) > 0) { ix[j] = ix[j-1]; j--; }
            ix[j] = i;
        }
        sb_addc(s, '{');
        for (size_t i = 0; i < v->n; i++) {
            if (i) sb_addc(s, ',');
            json_quote(s, v->keys[ix[i]]); sb_addc(s, ':'); sorted_json(s, v->items[ix[i]]);
        }
        sb_addc(s, '}'); free(ix);
    } else if (v->type == J_ARR) {
        sb_addc(s, '[');
        for (size_t i = 0; i < v->n; i++) { if (i) sb_addc(s, ','); sorted_json(s, v->items[i]); }
        sb_addc(s, ']');
    } else json_dump(s, v);
}
static char *render(const jv *v) {
    if (v->type == J_STR) return strdup(v->str);
    sb s = {0}; sorted_json(&s, v); return s.p;
}
static void piece(Q35Tokenizer *tok, Record *r, const char *s) {
    int n = q35_tok_encode(tok, s, 1, NULL, 0);
    if (n < 0 || r->n + n > CLEF_MAX_T * 8) clef_fail("piece token budget exceeded");
    r->ids = realloc(r->ids, (r->n + n + 1) * sizeof(int));
    if (q35_tok_encode(tok, s, 1, r->ids + r->n, n) != n) clef_fail("tokenize failed");
    int cap = (int)strlen(s) + 1; char *decoded = calloc(cap, 1);
    int dn = q35_tok_decode(tok, r->ids + r->n, n, decoded, cap);
    if (dn != cap-1 || memcmp(decoded, s, dn)) r->roundtrip = 0;
    free(decoded);
    if (r->pieces.n) sb_addc(&r->pieces, ',');
    json_quote(&r->pieces, s); sb_add(&r->text, s); r->n += n;
}
static void option(Q35Tokenizer *tok, Record *r, Question *q, const char *id, const jv *desc) {
    if (r->no == CLEF_MAX_O) clef_fail("too many options");
    sb b = {0}; sb_fmt(&b, "OPTION %d: ", q->count + 1); piece(tok, r, b.p); free(b.p);
    Option *o = &r->o[r->no++]; o->id = strdup(id); o->parent = r->nq-1; o->span.start = r->n;
    b = (sb){0}; sb_addc(&b, '{');
    if (desc && desc->type != J_NULL) { sb_add(&b, "\"description\":"); sorted_json(&b, desc); sb_addc(&b, ','); }
    sb_add(&b, "\"option_id\":"); json_quote(&b, id); sb_addc(&b, '}');
    piece(tok, r, b.p); free(b.p); o->span.end = r->n;
    piece(tok, r, "\n"); q->count++;
}
void clef_encode(Q35Tokenizer *tok, const jv *req, Record *r, int max_length) {
    memset(r, 0, sizeof *r); r->roundtrip = 1;
    const jv *qs = json_get(req, "questions"), *state = json_get(req, "state");
    if (!state || !qs || qs->type != J_OBJ || !qs->n || qs->n > CLEF_MAX_Q) clef_fail("invalid state/questions");
    if (json_get(req, "images") || json_get(req, "videos")) clef_fail("multimodal is out of scope");
    piece(tok, r, "\n\nSCHEMA FIELDS:\n");
    for (size_t i = 0; i < qs->n; i++) {
        const jv *jq = qs->items[i], *ty = json_get(jq, "type"), *ins = json_get(jq, "instructions"), *cr = json_get(jq, "criteria");
        if (!ty || ty->type != J_STR) clef_fail("missing question type");
        Question *q = &r->q[r->nq++]; q->id = strdup(qs->keys[i]); q->first = r->no;
        q->type = !strcmp(ty->str, "noul") ? 0 : !strcmp(ty->str, "choice") ? 1 : !strcmp(ty->str, "score") ? 2 : -1;
        if (q->type < 0) clef_fail("unknown question type");
        sb b = {0}; sb_fmt(&b, "\nFIELD %zu\nID: ", i+1); sb_add(&b, q->id);
        sb_add(&b, "\nTYPE: "); sb_add(&b, ty->str); sb_add(&b, "\nINSTRUCTION: "); piece(tok, r, b.p); free(b.p);
        q->span.start = r->n;
        char *text = (!ins || ins->type == J_NULL || (ins->type == J_STR && !*ins->str)) ? strdup(q->id) : render(ins);
        piece(tok, r, text); free(text); q->span.end = r->n;
        if (q->span.start == q->span.end) clef_fail("empty question span");
        piece(tok, r, "\nALLOWED OPTIONS:\n");
        if (!q->type) {
            const char *ids[] = {"true", "false"};
            char *defaults[] = {"The proposition is true or the answer is yes.", "The proposition is false or the answer is no."};
            if (cr && cr->type != J_NULL && cr->type != J_OBJ) clef_fail("invalid noul criteria");
            for (int j = 0; j < 2; j++) {
                jv d = {.type=J_STR, .str=defaults[j]}; const jv *v = json_get(cr, ids[j]); option(tok, r, q, ids[j], v ? v : &d);
            }
        } else {
            if (!cr || cr->type != (q->type == 1 ? J_OBJ : J_ARR) || !cr->n) clef_fail("invalid criteria");
            size_t *ix = calloc(cr->n, sizeof *ix);
            for (size_t j = 0; j < cr->n; j++) {
                size_t k = j;
                while (q->type == 1 && k && strcmp(cr->keys[ix[k-1]], cr->keys[j]) > 0) { ix[k] = ix[k-1]; k--; }
                ix[k] = j;
            }
            for (size_t j = 0; j < cr->n; j++) {
                char id[24]; snprintf(id, sizeof id, "%zu", j);
                option(tok, r, q, q->type == 1 ? cr->keys[ix[j]] : id, cr->items[ix[j]]);
            }
            free(ix);
        }
        piece(tok, r, "END FIELD\n");
    }
    Record *prefix = calloc(1, sizeof *prefix), *st = calloc(1, sizeof *st), *suffix = calloc(1, sizeof *suffix);
    prefix->roundtrip = st->roundtrip = suffix->roundtrip = 1;
    piece(tok, prefix, "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n");
    piece(tok, suffix, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    char *text = render(state); piece(tok, st, text); free(text);
    int fixed = prefix->n + r->n + suffix->n;
    if (max_length < fixed || max_length > CLEF_MAX_T) clef_fail("schema exceeds token budget");
    if (st->n > max_length - fixed) clef_fail("state truncation not enabled in CLI; input exceeds budget");
    int shift = prefix->n + st->n, total = fixed + st->n;
    int *ids = calloc(total, sizeof *ids);
    memcpy(ids, prefix->ids, prefix->n*4); memcpy(ids+prefix->n, st->ids, st->n*4);
    memcpy(ids+shift, r->ids, r->n*4); memcpy(ids+shift+r->n, suffix->ids, suffix->n*4);
    free(r->ids); r->ids = ids; r->n = total;
    for (int i = 0; i < r->nq; i++) { r->q[i].span.start += shift; r->q[i].span.end += shift; }
    for (int i = 0; i < r->no; i++) { r->o[i].span.start += shift; r->o[i].span.end += shift; }
    sb joined = {0}, pieces = {0}; Record *parts[] = {prefix, st, r, suffix};
    for (int i = 0; i < 4; i++) {
        sb_add(&joined, parts[i]->text.p); if (i) sb_addc(&pieces, ','); sb_add(&pieces, parts[i]->pieces.p);
        r->roundtrip &= parts[i]->roundtrip;
    }
    free(r->text.p); free(r->pieces.p); r->text = joined; r->pieces = pieces;
    clef_record_free(prefix); clef_record_free(st); clef_record_free(suffix); free(prefix); free(st); free(suffix);
}
void clef_record_json(const Record *r, sb *b) {
    sb_fmt(b, "{\"input_tokens\":%d,\"roundtrip\":%s,\"input_ids\":[", r->n, r->roundtrip ? "true" : "false");
    for (int i = 0; i < r->n; i++) sb_fmt(b, "%s%d", i ? "," : "", r->ids[i]);
    sb_add(b, "],\"questions\":[");
    for (int i = 0; i < r->nq; i++) {
        const Question *q = &r->q[i]; if (i) sb_addc(b, ','); sb_add(b, "{\"question_id\":"); json_quote(b, q->id);
        sb_fmt(b, ",\"question_type\":%d,\"question_span\":[%d,%d],\"option_ids\":[", q->type, q->span.start, q->span.end);
        for (int j = 0; j < q->count; j++) { if (j) sb_addc(b, ','); json_quote(b, r->o[q->first+j].id); }
        sb_add(b, "],\"option_spans\":[");
        for (int j = 0; j < q->count; j++) { Span s = r->o[q->first+j].span; sb_fmt(b, "%s[%d,%d]", j ? "," : "", s.start, s.end); }
        sb_add(b, "]}");
    }
    sb_add(b, "],\"text\":"); json_quote(b, r->text.p); sb_add(b, ",\"pieces\":["); sb_add(b, r->pieces.p); sb_add(b, "]}");
}
void clef_record_free(Record *r) {
    for (int i = 0; i < r->nq; i++) free(r->q[i].id);
    for (int i = 0; i < r->no; i++) free(r->o[i].id);
    free(r->ids); free(r->text.p); free(r->pieces.p);
}
#ifdef SPANS_MAIN
int main(int argc, char **argv) {
    if (argc != 3) clef_fail("usage: clef-spans TOKENIZER_DIR REQUEST.json");
    Q35Tokenizer tok; if (q35_tok_load(&tok, argv[1])) clef_fail("tokenizer load failed");
    char *text = json_read_file(argv[2]); const char *err = NULL;
    if (!text) clef_fail("request read failed"); jv *req = json_parse(text, &err); if (!req) clef_fail(err);
    Record *r = calloc(1, sizeof *r); clef_encode(&tok, req, r, CLEF_MAX_T); sb b = {0}; clef_record_json(r, &b); puts(b.p);
    clef_record_free(r); free(r); free(b.p); json_free(req); free(text); q35_tok_free(&tok); return 0;
}
#endif
