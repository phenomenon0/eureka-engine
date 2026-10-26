// serve_structured.c — G16: the request side of llb-serve's structured output
// (serve_structured.h; docs/guides/structured-output.md). Everything the engine sees is one GBNF text in
// dual.c's existing "grammar" field, so the engine (G1's mask at the W1-SAMPLER hook) is
// unchanged: JSON schemas are converted here (G1's converter), a user grammar passes through
// verbatim, and the thinking prefix / tool-call grammars are composed here by renaming rules.
// Every composed grammar is compiled once here (G1's compiler) so a bad one is a 400 with
// the compiler's message instead of an engine-side refusal.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "serve_structured.h"
#include "../../host/llb_grammar.h"
#include "../../host/llb_toolcall.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int word(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

// rename every rule (definitions and references) by prefixing pfx; literals, classes,
// {m,n} counts and comments are copied as they are (G1's GBNF lexer: host/llb_grammar_impl.h)
char *st_gbnf_prefix(const char *g, const char *pfx) {
    sb o = {0};
    sb_add(&o, "");
    for (const char *p = g; *p;) {
        char c = *p;
        if (c == '#') {
            const char *e = p;
            while (*e && *e != '\n' && *e != '\r') e++;
            sb_addn(&o, p, (size_t)(e - p));
            p = e;
        } else if (c == '"' || c == '[') {
            char close = c == '"' ? '"' : ']';
            const char *e = p + 1;
            while (*e && *e != close) e += e[0] == '\\' && e[1] ? 2 : 1;
            if (*e) e++;
            sb_addn(&o, p, (size_t)(e - p));
            p = e;
        } else if (c == '{') {
            const char *e = strchr(p, '}');
            e = e ? e + 1 : p + strlen(p);
            sb_addn(&o, p, (size_t)(e - p));
            p = e;
        } else if (word(c)) {
            const char *e = p;
            while (word(*e)) e++;
            sb_add(&o, pfx);
            sb_addn(&o, p, (size_t)(e - p));
            p = e;
        } else sb_addc(&o, c), p++;
    }
    return o.p;
}

// a GBNF "literal": printable ASCII as is, everything else \xHH (bytes; G1 matches bytes)
static void lit(sb *o, const char *s) {
    sb_addc(o, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') sb_fmt(o, "\\%c", *p);
        else if (*p >= 0x20 && *p < 0x7f) sb_addc(o, (char)*p);
        else if (*p >= 0x80) sb_addc(o, (char)*p); // UTF-8 bytes of a name: literal bytes
        else sb_fmt(o, "\\x%02X", *p);
    }
    sb_addc(o, '"');
}

static void cls_ch(sb *o, char c) { sb_fmt(o, "\\x%02X", (unsigned char)c); }

// name-0: any text that does not contain marker, followed by marker (a KMP automaton, one
// rule per matched-prefix length; tail recursion, which the PDA runs without stack growth).
// marker[0] must not occur again in marker (true for "</think>", "\n</parameter>").
void st_gbnf_until(sb *o, const char *name, const char *marker) {
    int k = (int)strlen(marker);
    char m0 = marker[0];
    sb_fmt(o, "%s-0 ::= [^", name);
    cls_ch(o, m0);
    sb_fmt(o, "] %s-0 | ", name);
    char s0[2] = {m0, 0};
    lit(o, s0);
    sb_fmt(o, " %s-1\n", name);
    for (int i = 1; i < k; i++) {
        char mi[2] = {marker[i], 0};
        sb_fmt(o, "%s-%d ::= ", name, i);
        lit(o, mi);
        if (i + 1 < k) sb_fmt(o, " %s-%d", name, i + 1);
        sb_add(o, " | ");
        lit(o, s0);
        sb_fmt(o, " %s-1 | [^", name);
        cls_ch(o, marker[i]);
        cls_ch(o, m0);
        sb_fmt(o, "] %s-0\n", name);
    }
}

// name-0: a parameter's raw text, the way G1's parser delimits it: no "</parameter>",
// "<parameter=" or "</function>" anywhere (the parser ends a value at the first of those).
// It may end anywhere: the parser strips one leading and one trailing "\n", so "Paris" and
// "\nParis\n" are the same value (G16 GPU run: a forced "\n" before </parameter> made a model
// that writes "Paris</parameter>" emit junk instead). One rule per marker-trie node (every
// marker starts with '<' and has no other '<', so a '<' always restarts at the "<" node).
static const char *pv_marks[] = {"</parameter>", "<parameter=", "</function>", NULL};
void st_gbnf_value(sb *o, const char *name) {
    char nodes[64][16];
    int nn = 2;
    nodes[0][0] = nodes[1][0] = 0;
    for (int m = 0; pv_marks[m]; m++)
        for (size_t k = 1; k < strlen(pv_marks[m]); k++) {
            int seen = 0;
            for (int i = 2; i < nn; i++)
                if (strlen(nodes[i]) == k && !strncmp(nodes[i], pv_marks[m], k)) seen = 1;
            if (!seen) snprintf(nodes[nn++], sizeof nodes[0], "%.*s", (int)k, pv_marks[m]);
        }
    int lt = -1;
    for (int i = 2; i < nn; i++)
        if (!strcmp(nodes[i], "<")) lt = i;
    for (int i = 0; i < nn; i++) {
        const char *p = nodes[i];
        size_t pl = strlen(p);
        char kids[8];
        int nk = 0;
        sb_fmt(o, "%s-%d ::= ", name, i);
        sb_add(o, "\"\" | ");
        for (int m = 0; pv_marks[m]; m++) { // the children: chars that extend p along a marker
            size_t ml = strlen(pv_marks[m]);
            if (ml <= pl || strncmp(pv_marks[m], p, pl) || memchr(kids, pv_marks[m][pl], (size_t)nk)) continue;
            char c = pv_marks[m][pl];
            kids[nk++] = c;
            if (ml == pl + 1) continue; // completes a marker: forbidden
            for (int j = 2; j < nn; j++)
                if (strlen(nodes[j]) == pl + 1 && !strncmp(nodes[j], pv_marks[m], pl + 1)) {
                    char cs[2] = {c, 0};
                    lit(o, cs);
                    sb_fmt(o, " %s-%d | ", name, j);
                }
        }
        if (!memchr(kids, '<', (size_t)nk)) sb_fmt(o, "\"<\" %s-%d | ", name, lt);
        sb_fmt(o, "\"\\n\" %s-1 | [^", name);
        cls_ch(o, '<');
        cls_ch(o, '\n');
        for (int k = 0; k < nk; k++) cls_ch(o, kids[k]);
        sb_fmt(o, "] %s-0\n", name);
    }
}

// the Qwen3.5 thinking generation prompt ends with "<think>\n": the reasoning runs free up
// to "</think>", then up to two "\n" (the template writes "\n</think>\n\n"), then the
// user's grammar, renamed into its own namespace
char *st_think_wrap(const char *gbnf) {
    sb o = {0};
    sb_add(&o, "root ::= st-think-0 \"\\n\"{0,2} u-root\n");
    st_gbnf_until(&o, "st-think", "</think>");
    char *u = st_gbnf_prefix(gbnf, "u-");
    sb_add(&o, u);
    sb_addc(&o, '\n');
    free(u);
    return o.p;
}

static char *dump(const jv *v) {
    sb o = {0};
    json_dump(&o, v);
    return o.p;
}

// a parameter's effective type the way the parser (llb_toolcall.h) types its value:
// a type list -> its first non-"null" entry; absent -> NULL (string semantics)
static const char *ptype(const jv *ps) {
    const jv *t = ps && ps->type == J_OBJ ? json_get(ps, "type") : NULL;
    if (t && t->type == J_STR) return t->str;
    if (t && t->type == J_ARR)
        for (size_t i = 0; i < t->n; i++)
            if (t->items[i]->type == J_STR && strcmp(t->items[i]->str, "null")) return t->items[i]->str;
    return NULL;
}

// one parameter's value grammar into rule `rule`: strings raw (the template writes them
// verbatim; an all-string enum/const = those literals), other types = G1's JSON-schema
// grammar of the property (renamed); 0, or -1 + *err
static int pvalue(sb *o, const char *rule, const jv *ps, int *need_str, const char **err) {
    const char *t = ptype(ps);
    const jv *en = ps && ps->type == J_OBJ ? json_get(ps, "enum") : NULL, *cn = ps && ps->type == J_OBJ ? json_get(ps, "const") : NULL;
    if (!t || !strcmp(t, "string")) {
        int lits = (en && en->type == J_ARR && en->n) || (cn && cn->type == J_STR);
        for (size_t i = 0; en && en->type == J_ARR && i < en->n; i++)
            if (en->items[i]->type != J_STR) lits = 0;
        sb_fmt(o, "%s ::= ", rule);
        if (lits && cn && cn->type == J_STR) lit(o, cn->str);
        else if (lits)
            for (size_t i = 0; i < en->n; i++) {
                if (i) sb_add(o, " | ");
                lit(o, en->items[i]->str);
            }
        else {
            sb_add(o, "st-str-0 \"</parameter>\" \"\\n\"?"); // the raw text, its "\n"s included
            *need_str = 1;
            sb_addc(o, '\n');
            return 1; // the rule already ends the value
        }
        sb_addc(o, '\n');
        return 0;
    }
    char e[256], *sj = dump(ps), *g = llb_gram_schema_to_gbnf(sj, e, sizeof e);
    if (!g) { // a keyword G1's converter refuses (minimum, pattern, ...): the bare type
        fprintf(stderr, "structured: tool parameter schema %s: %s; constraining the bare type \"%s\"\n", sj, e, t);
        free(sj);
        sb ts = {0};
        sb_add(&ts, "{\"type\":");
        json_quote(&ts, t);
        sb_addc(&ts, '}');
        g = llb_gram_schema_to_gbnf(ts.p, e, sizeof e);
        free(ts.p);
        if (!g) return *err = "a tool parameter has an unsupported type", -1;
    } else free(sj);
    sb pf = {0};
    sb_fmt(&pf, "%s-", rule);
    char *r = st_gbnf_prefix(g, pf.p);
    sb_fmt(o, "%s ::= %sroot\n", rule, pf.p);
    sb_add(o, r);
    sb_addc(o, '\n');
    free(r), free(g), free(pf.p);
    return 0;
}

// tool_choice "required" / {"function":{"name":only}}: one call (multi = 0) or calls
// separated by "\n" (the template's own glue), each the template's form
//   <tool_call>\n<function=NAME>\n(<parameter=P>\nVALUE\n</parameter>\n)*</function>\n</tool_call>
// with the "\n"s inside a call optional (G1's parser does not need them; a model that drops
// one must not be forced into junk)
// parameters in declared order, the non-required ones optional (as G1's object rule)
char *st_tools_gbnf(const jv *tools, const char *only, int multi, const char **err) {
    sb o = {0}, fs = {0};
    int need_str = 0, nf = 0;
    sb_add(&o, multi ? "root ::= st-call (\"\\n\" st-call)*\n" : "root ::= st-call\n");
    sb_add(&o, "st-call ::= \"<tool_call>\\n\" (");
    for (size_t i = 0; i < tools->n; i++) {
        const jv *f = json_get(tools->items[i], "function"), *nm = f ? json_get(f, "name") : NULL;
        if (only && strcmp(nm->str, only)) continue;
        const jv *pa = json_get(f, "parameters"), *pr = pa && pa->type == J_OBJ ? json_get(pa, "properties") : NULL;
        const jv *rq = pa && pa->type == J_OBJ ? json_get(pa, "required") : NULL;
        sb_fmt(&o, "%sst-f%zu", nf++ ? " | " : " ", i);
        sb_fmt(&fs, "st-f%zu ::= ", i);
        sb fn = {0};
        sb_add(&fn, "<function=");
        sb_add(&fn, nm->str);
        sb_add(&fn, ">");
        lit(&fs, fn.p);
        sb_add(&fs, " \"\\n\"?");
        free(fn.p);
        sb ps = {0};
        for (size_t j = 0; pr && pr->type == J_OBJ && j < pr->n; j++) {
            int req = 0;
            for (size_t q = 0; rq && rq->type == J_ARR && q < rq->n; q++)
                if (rq->items[q]->type == J_STR && !strcmp(rq->items[q]->str, pr->keys[j])) req = 1;
            sb_fmt(&fs, " st-f%zu-p%zu%s", i, j, req ? "" : "?");
            sb rule = {0}, head = {0};
            sb_fmt(&rule, "st-f%zu-p%zu", i, j);
            sb_add(&head, "<parameter=");
            sb_add(&head, pr->keys[j]);
            sb_add(&head, ">");
            sb_fmt(&ps, "%s ::= ", rule.p);
            lit(&ps, head.p);
            sb_fmt(&ps, " \"\\n\"? %s-v", rule.p);
            sb_add(&rule, "-v");
            sb vs = {0};
            int ends = pvalue(&vs, rule.p, pr->items[j], &need_str, err);
            if (ends < 0) {
                free(rule.p), free(head.p), free(vs.p), free(ps.p), free(o.p), free(fs.p);
                return NULL;
            }
            sb_add(&ps, ends ? "\n" : " \"\\n\"? \"</parameter>\" \"\\n\"?\n");
            sb_add(&ps, vs.p);
            free(rule.p), free(head.p), free(vs.p);
        }
        sb_addc(&fs, '\n');
        if (ps.p) sb_add(&fs, ps.p);
        free(ps.p);
    }
    if (!nf) {
        free(o.p), free(fs.p);
        return *err = "tool_choice names a function that is not in tools", NULL;
    }
    sb_add(&o, " ) \"</function>\" \"\\n\"? \"</tool_call>\"\n");
    sb_add(&o, fs.p);
    if (need_str) st_gbnf_value(&o, "st-str");
    free(fs.p);
    return o.p;
}

static int has_tool_history(const jv *ms) {
    for (size_t i = 0; ms && ms->type == J_ARR && i < ms->n; i++) {
        const jv *r = json_get(ms->items[i], "role"), *tc = json_get(ms->items[i], "tool_calls");
        if (r && r->type == J_STR && !strcmp(r->str, "tool")) return 1;
        if (tc && tc->type == J_ARR && tc->n) return 1;
    }
    return 0;
}

static const char *check_tools(const jv *tl) {
    if (tl->type != J_ARR) return "tools must be an array";
    for (size_t i = 0; i < tl->n; i++) {
        const jv *ty = json_get(tl->items[i], "type"), *f = json_get(tl->items[i], "function");
        const jv *nm = f ? json_get(f, "name") : NULL, *pa = f ? json_get(f, "parameters") : NULL;
        if ((ty && (ty->type != J_STR || strcmp(ty->str, "function"))) || !f || f->type != J_OBJ)
            return "every tool must be {\"type\":\"function\",\"function\":{...}}";
        if (!nm || nm->type != J_STR || !*nm->str || strpbrk(nm->str, "<>\n"))
            return "every tool function needs a name (no '<', '>' or newline)";
        if (pa && pa->type != J_OBJ) return "tool function parameters must be a JSON-schema object";
        const jv *pr = pa ? json_get(pa, "properties") : NULL;
        for (size_t j = 0; pr && pr->type == J_OBJ && j < pr->n; j++)
            if (!*pr->keys[j] || strpbrk(pr->keys[j], "<>\n")) return "tool parameter names must not contain '<', '>' or newline";
    }
    return NULL;
}

void st_free(StReq *s) {
    free(s->gbnf), free(s->tools_json), free(s->prompt);
    *s = (StReq){0};
}

const char *st_request(const jv *body, int chat, int thinking, int agp, int flag_on, StReq *s) {
    const jv *rf = json_get(body, "response_format"), *js = json_get(body, "json_schema"), *gr = json_get(body, "grammar");
    const jv *tl = json_get(body, "tools"), *fn = json_get(body, "functions"), *tc = json_get(body, "tool_choice");
    const jv *rft = rf && rf->type == J_OBJ ? json_get(rf, "type") : NULL;
    if (rf && rf->type == J_NULL) rf = NULL;
    if (js && js->type == J_NULL) js = NULL;
    if (gr && gr->type == J_NULL) gr = NULL;
    if (tl && tl->type == J_ARR && !tl->n) tl = NULL; // OpenAI clients send tools: [] for "no tools"
    if (rf && (!rft || rft->type != J_STR)) return "response_format must be {\"type\": \"text\" | \"json_object\" | \"json_schema\", ...}";
    if (rf && !strcmp(rft->str, "text")) rf = NULL;
    int hist = chat && has_tool_history(json_get(body, "messages"));
    if (!rf && !js && !gr && !tl && !fn && !hist) return NULL; // a plain request: W1-SERVER's path, untouched
    if (!flag_on) {
        if (tl || fn || hist) return NULL; // render_chat answers these with W1-SERVER's own 400
        return "response_format / json_schema / grammar need the server started with --structured";
    }
    if (fn) return "functions is the deprecated form: send tools";
    if ((rf != NULL) + (js != NULL) + (gr != NULL) > 1) return "send at most one of response_format, json_schema, grammar";
    if (tl && (rf || js || gr)) return "tools cannot be combined with response_format / json_schema / grammar";
    if (!chat && tl) return "tools need /v1/chat/completions";
    char e[512];
    s->on = 1;
    s->reasoning = chat && agp && thinking;
    // ---- the constraint
    if (rf || js) {
        const jv *sch = js;
        if (rf && !strcmp(rft->str, "json_object")) sch = NULL;
        else if (rf && !strcmp(rft->str, "json_schema")) {
            const jv *w = json_get(rf, "json_schema");
            sch = w && w->type == J_OBJ ? json_get(w, "schema") : NULL;
            if (!sch) return "response_format json_schema needs json_schema.schema";
        } else if (rf) return "response_format type must be text, json_object or json_schema";
        if (sch && sch->type != J_OBJ && sch->type != J_BOOL) return "json_schema must be a JSON-schema object";
        char *st = sch ? dump(sch) : strdup("{\"type\":\"object\"}");
        s->gbnf = llb_gram_schema_to_gbnf(st, e, sizeof e);
        free(st);
        if (!s->gbnf) {
            static __thread char m[600];
            snprintf(m, sizeof m, "json_schema: %s", e);
            return m;
        }
    } else if (gr) {
        if (gr->type != J_STR || !*gr->str) return "grammar must be a GBNF string";
        s->gbnf = strdup(gr->str);
    }
    const char *only = NULL;
    int choice_none = 0, forced = 0;
    if (tl) {
        const char *te = check_tools(tl);
        if (te) return te;
        if (!tc || (tc->type == J_STR && !strcmp(tc->str, "auto"))) {
        } else if (tc->type == J_STR && !strcmp(tc->str, "none")) choice_none = 1;
        else if (tc->type == J_STR && !strcmp(tc->str, "required")) forced = 1;
        else if (tc->type == J_OBJ) {
            const jv *f = json_get(tc, "function"), *n = f ? json_get(f, "name") : NULL;
            if (!n || n->type != J_STR) return "tool_choice must be auto, none, required or {\"type\":\"function\",\"function\":{\"name\":...}}";
            only = n->str, forced = 1;
        } else return "tool_choice must be auto, none, required or {\"type\":\"function\",\"function\":{\"name\":...}}";
        const jv *par = json_get(body, "parallel_tool_calls");
        int multi = !(par && par->type == J_BOOL && !par->b) && !only;
        if (forced) {
            const char *ge = NULL;
            s->gbnf = st_tools_gbnf(tl, only, multi, &ge);
            if (!s->gbnf) return ge;
        }
        s->tools = !choice_none;
        s->forced = forced;
        s->tools_json = dump(tl);
    }
    // ---- the prompt: the tool-capable template port (G1, byte-identical to HF jinja)
    if (tl || hist) {
        sb rq = {0};
        sb_add(&rq, "{\"messages\":");
        const jv *ms = json_get(body, "messages");
        if (ms) json_dump(&rq, ms);
        else sb_add(&rq, "null");
        if (tl && !choice_none) {
            sb_add(&rq, ",\"tools\":");
            json_dump(&rq, tl);
        }
        sb_addc(&rq, '}');
        s->prompt = llb_tc_render(rq.p, agp, thinking, e, sizeof e);
        free(rq.p);
        if (!s->prompt) {
            static __thread char m[600];
            snprintf(m, sizeof m, "%s", e);
            return m;
        }
    }
    // ---- reasoning first, then the constraint; compile once here for a clean 400
    if (s->gbnf && s->reasoning) {
        char *w = st_think_wrap(s->gbnf);
        free(s->gbnf);
        s->gbnf = w;
    }
    if (s->gbnf) {
        LlbGram *g = llb_gram_from_gbnf(s->gbnf, NULL, e, sizeof e);
        if (!g) {
            static __thread char m[600];
            snprintf(m, sizeof m, "grammar: %s", e);
            return m;
        }
        llb_gram_free(g);
    }
    return NULL;
}
