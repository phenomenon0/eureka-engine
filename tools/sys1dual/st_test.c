// tools/sys1dual/st_test.c — G16 tests of serve_structured.c (the request side of
// llb-serve --structured), matched byte by byte with G1's grammar engine.
//   st_test unit                       the built-in checks (PASS/FAIL lines, rc = #FAIL)
//   st_test prefix PFX < G.gbnf        the renamed grammar
//   st_test wrap < G.gbnf              the thinking-wrapped grammar
//   st_test tools TOOLS.json [ONLY|-] [1|0]   the tool_choice grammar (multi = 1)
//   st_test request BODY.json CHAT FLAG       st_request's verdict + grammar + prompt
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "serve_structured.h"
#include "../../host/llb_grammar.h"
#include "../../host/llb_toolcall.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int npass, nfail;
static void ok(int c, const char *what) {
    printf("%s %s\n", c ? "PASS" : "FAIL", what);
    c ? npass++ : nfail++;
}

static char *slurp(FILE *f) {
    sb o = {0};
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) sb_addn(&o, b, n);
    sb_add(&o, "");
    return o.p;
}

// a byte vocabulary: id b = the byte b, id 256 = end of generation
static LlbGVocab *bytevocab(void) {
    static unsigned char bs[256];
    static const unsigned char *ps[257];
    static int ls[257];
    for (int i = 0; i < 256; i++) bs[i] = (unsigned char)i, ps[i] = bs + i, ls[i] = 1;
    ps[256] = (const unsigned char *)"<|eog|>", ls[256] = 7;
    LlbGVocab *v = llb_gv_new(257, ps, ls);
    int e = 256;
    llb_gv_set_eog(v, &e, 1);
    return v;
}
static LlbGVocab *V;

// 1 = s is a complete sentence of the grammar
static int sentence(const char *gbnf, const char *s) {
    char err[256];
    LlbGram *g = llb_gram_from_gbnf(gbnf, NULL, err, sizeof err);
    if (!g) {
        printf("     compile: %s\n", err);
        return -1;
    }
    LlbGMatch *m = llb_gm_new(g, V);
    int r = llb_gm_accept_bytes(m, s, (int)strlen(s)) == 0 && llb_gm_can_end(m);
    llb_gm_free(m);
    llb_gram_free(g);
    return r;
}

static void unit_until(void) {
    sb g = {0};
    sb_add(&g, "root ::= t-0\n");
    st_gbnf_until(&g, "t", "</think>");
    const char *yes[] = {"</think>", "abc</think>", "<</think>", "</thin</think>", "<<<//</think>", "é😀\n</think>",
        "</thinx></think>", "a<b<c</t</think>", NULL};
    const char *no[] = {"", "abc", "</think", "a</think>b", "</think></think>", "x</think>\n", NULL};
    int good = 1;
    for (int i = 0; yes[i]; i++)
        if (sentence(g.p, yes[i]) != 1) good = 0, printf("     should accept %s\n", yes[i]);
    for (int i = 0; no[i]; i++)
        if (sentence(g.p, no[i]) != 0) good = 0, printf("     should reject %s\n", no[i]);
    ok(good, "until(</think>): 8 accepted, 6 rejected");
    // a long free run (tail recursion: no stack growth, bounded time)
    size_t n = 200000;
    char *big = malloc(n + 16);
    for (size_t i = 0; i < n; i++) big[i] = "ab<c/d\n"[i % 7];
    strcpy(big + n, "</think>");
    clock_t t = clock();
    int r = sentence(g.p, big);
    double ms = 1e3 * (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("     200 kB reasoning matched in %.1f ms\n", ms);
    ok(r == 1 && ms < 5000, "until: a 200 kB run matches in bounded time");
    free(big);
    free(g.p);
    sb h = {0};
    sb_add(&h, "root ::= p-0 \"</parameter>\"\n");
    st_gbnf_value(&h, "p");
    const char *vy[] = {"\n</parameter>", "a\nb\n</paramete\n</parameter>", "<<</p</par=<param\n</parameter>", "</func\n</parameter>",
        "é <b>x</b>\n</parameter>", "x</parameter>", "</parameter>", NULL};
    const char *vn[] = {"</parameter>\n</parameter>", "a<parameter=b\n</parameter>", "a</function>\n</parameter>",
        "x\n</parameter>y\n</parameter>", "\n</parameter>\n</parameter>", NULL};
    good = 1;
    for (int i = 0; vy[i]; i++)
        if (sentence(h.p, vy[i]) != 1) good = 0, printf("     value should accept %s\n", vy[i]);
    for (int i = 0; vn[i]; i++)
        if (sentence(h.p, vn[i]) != 0) good = 0, printf("     value should reject %s\n", vn[i]);
    ok(good, "value: ends anywhere (\\n optional); no </parameter>, <parameter=, </function> inside");
    free(h.p);
}

static void unit_prefix(void) {
    const char *g = "# a comment with root and words\nroot ::= item (\",\" item)* [a-z0-9_-]{2,3} \"root-x\"\n"
                    "item ::= \"[\" x-1 \"]\" | [\\]\\\\] # trailing\nx-1 ::= \"\\\"q\\\"\" ws\nws ::= [ \\t]*\n";
    char *p = st_gbnf_prefix(g, "u-");
    int good = strstr(p, "u-root ::= u-item (\",\" u-item)* [a-z0-9_-]{2,3} \"root-x\"") && strstr(p, "u-x-1 ::=") &&
               strstr(p, "u-ws ::= [ \\t]*") && strstr(p, "# a comment with root and words");
    char *w = st_think_wrap(g);
    good = good && sentence(w, "hmm\n</think>\n\n[\"q\"  ],]zz-root-x") == 1 && sentence(w, "[\"q\"],]zz-root-x") == 0;
    ok(good, "prefix: rules renamed, literals / classes / counts / comments kept");
    free(p), free(w);
}

static void unit_wrap(void) {
    char e[256];
    char *js = llb_gram_schema_to_gbnf("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"]}", e, sizeof e);
    char *w = st_think_wrap(js);
    ok(sentence(w, "Let me think.\n</think>\n\n{\"a\": 1}") == 1 && sentence(w, "</think>{\"a\":1}") == 1 &&
           sentence(w, "{\"a\": 1}") == 0 && sentence(w, "x</think>\n\n\n{\"a\": 1}") == 0 &&
           sentence(w, "a</think>b</think>{\"a\":1}") == 0 && sentence(w, "x</think>\n\n{\"a\": \"1\"}") == 0,
        "think wrap: reasoning, </think>, <= 2 newlines, then exactly the schema");
    free(js), free(w);
}

static const char *TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]},"
    "\"days\":{\"type\":\"integer\",\"minimum\":1},\"detail\":{\"type\":[\"boolean\",\"null\"]}},\"required\":[\"city\",\"unit\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"search\",\"description\":\"web\",\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"q\":{\"description\":\"query\"},\"filters\":{\"type\":\"object\",\"properties\":{\"site\":{\"type\":\"string\"}}},"
    "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"maxItems\":3}},\"required\":[\"q\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"now\"}}]";

static const char *CALL1 = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSão Paulo\n</parameter>\n"
                           "<parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
static const char *CALL2 = "<tool_call>\n<function=search>\n<parameter=q>\nbest <b>tags</b>\nline two\n</parameter>\n"
                           "<parameter=filters>\n{\"site\": \"x.org\"}\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n"
                           "</function>\n</tool_call>";
static const char *CALL3 = "<tool_call>\n<function=now>\n</function>\n</tool_call>";

static void unit_tools(void) {
    const char *err = NULL;
    jv *tl = json_parse(TOOLS, &err);
    char *g = st_tools_gbnf(tl, NULL, 1, &err);
    if (!g) {
        ok(0, err);
        return;
    }
    sb two = {0};
    sb_add(&two, CALL1), sb_add(&two, "\n"), sb_add(&two, CALL2), sb_add(&two, "\n"), sb_add(&two, CALL3);
    ok(sentence(g, CALL1) == 1 && sentence(g, CALL2) == 1 && sentence(g, CALL3) == 1 && sentence(g, two.p) == 1,
        "tools grammar: the template's call form (string, enum, int, object, array, no params, 3 calls)");
    const char *bad[] = {
        "<tool_call>\n<function=get_weather>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>", // city missing
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n<parameter=unit>\nkelvin\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=delete_all>\n</function>\n</tool_call>",
        "Sure! <tool_call>\n<function=now>\n</function>\n</tool_call>",
        "<tool_call>\n<function=now>\n</function>\n</tool_call>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>",
        "<tool_call>\n<function=search>\n<parameter=q>\nx\n</parameter>\n<parameter=tags>\n[\"a\",\"b\",\"c\",\"d\"]\n</parameter>\n</function>\n</tool_call>",
        NULL};
    int good = 1;
    for (int i = 0; bad[i]; i++)
        if (sentence(g, bad[i]) != 0) good = 0, printf("     should reject #%d\n", i);
    ok(good, "tools grammar: missing required, bad enum, bad int, unknown function, prose, \\n\\n glue, maxItems: rejected");
    free(g);
    g = st_tools_gbnf(tl, "now", 0, &err);
    ok(g && sentence(g, CALL3) == 1 && sentence(g, CALL1) == 0 && sentence(g, two.p) == 0, "tools grammar: named function, one call");
    free(g);
    ok(!st_tools_gbnf(tl, "nope", 0, &err) && strstr(err, "not in tools"), "tools grammar: unknown named function refused");
    // the template writes the same form: render an assistant turn holding these calls
    sb rq = {0};
    sb_add(&rq, "{\"tools\":");
    sb_add(&rq, TOOLS);
    sb_add(&rq, ",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":["
                "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"São Paulo\",\"unit\":\"celsius\",\"days\":3}}},"
                "{\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"best <b>tags</b>\\\\nline two\\\",\\\"filters\\\":{\\\"site\\\":\\\"x.org\\\"},\\\"tags\\\":[\\\"a\\\",\\\"b\\\"]}\"}},"
                "{\"type\":\"function\",\"function\":{\"name\":\"now\",\"arguments\":{}}}]}]}");
    char e[256];
    char *pr = llb_tc_render(rq.p, 0, 1, e, sizeof e);
    ok(pr && strstr(pr, two.p), "tools grammar == the chat template's own rendering of the same calls");
    if (!pr) printf("     render: %s\n", e);
    // and the parser reads them back typed by the schema
    LlbTcParser *p = llb_tcp_new(TOOLS);
    for (const char *q = two.p; *q; q++) llb_tcp_feed(p, q, 1);
    llb_tcp_finish(p);
    LlbTcEvent ev;
    sb got = {0};
    while (llb_tcp_next(p, &ev))
        if (ev.type == LLB_TCE_TOOL_CALL) sb_fmt(&got, "%s %s|", ev.name, ev.args);
        else if (ev.type != LLB_TCE_CONTENT || ev.len) sb_fmt(&got, "ev%d|", ev.type);
    const char *want = "get_weather {\"city\": \"São Paulo\", \"unit\": \"celsius\", \"days\": 3}|"
                       "search {\"q\": \"best <b>tags</b>\\nline two\", \"filters\": {\"site\": \"x.org\"}, \"tags\": [\"a\", \"b\"]}|now {}|";
    ok(got.p && !strcmp(got.p, want), "parser: the grammar's calls parse back typed (1-byte chunks)");
    // G16 GPU finding: a model that drops the template's inner "\n"s ("Paris</parameter>")
    // must be accepted, not forced into junk; the parser reads the same values
    const char *TIGHT = "<tool_call>\n<function=get_weather><parameter=city>Paris</parameter><parameter=unit>celsius"
                        "</parameter>\n<parameter=days>3</parameter></function></tool_call>";
    char *g1 = st_tools_gbnf(tl, NULL, 1, &err);
    LlbTcParser *p2 = llb_tcp_new(TOOLS);
    llb_tcp_feed(p2, TIGHT, (int)strlen(TIGHT));
    llb_tcp_finish(p2);
    sb got2 = {0};
    while (llb_tcp_next(p2, &ev))
        if (ev.type == LLB_TCE_TOOL_CALL) sb_fmt(&got2, "%s %s|", ev.name, ev.args);
        else if (ev.type != LLB_TCE_CONTENT || ev.len) sb_fmt(&got2, "ev%d|", ev.type);
    ok(sentence(g1, TIGHT) == 1 && got2.p && !strcmp(got2.p, "get_weather {\"city\": \"Paris\", \"unit\": \"celsius\", \"days\": 3}|"),
        "tools grammar: the inner newlines are optional and parse to the same values");
    if (got2.p && strcmp(got2.p, "get_weather {\"city\": \"Paris\", \"unit\": \"celsius\", \"days\": 3}|")) printf("     got %s\n", got2.p);
    free(g1), free(got2.p);
    llb_tcp_free(p2);
    if (!got.p || strcmp(got.p, want)) printf("     got  %s\n     want %s\n", got.p, want);
    llb_tcp_free(p);
    free(got.p), free(pr), free(rq.p), free(two.p);
    json_free(tl);
}

static const char *req(const char *body, int chat, int flag, StReq *s) {
    const char *err = NULL;
    jv *b = json_parse(body, &err);
    *s = (StReq){0};
    const jv *kw = json_get(b, "chat_template_kwargs"), *et = kw ? json_get(kw, "enable_thinking") : NULL;
    const char *r = st_request(b, chat, et && et->type == J_BOOL ? et->b : 1, 1, flag, s);
    static char keep[700];
    if (r) snprintf(keep, sizeof keep, "%s", r), r = keep;
    json_free(b);
    return r;
}

static void unit_request(void) {
    StReq s;
    const char *m = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
    char b[2048];
    snprintf(b, sizeof b, "{%s}", m);
    ok(!req(b, 1, 1, &s) && !s.on && !s.gbnf, "plain chat: not structured (W1-SERVER path)");
    snprintf(b, sizeof b, "{%s,\"response_format\":{\"type\":\"text\"},\"tools\":[]}", m);
    ok(!req(b, 1, 1, &s) && !s.on, "response_format text + tools []: plain");
    snprintf(b, sizeof b, "{%s,\"response_format\":{\"type\":\"json_object\"}}", m);
    const char *e = req(b, 1, 0, &s);
    ok(e && strstr(e, "--structured"), "flag off: response_format is a 400 naming --structured");
    snprintf(b, sizeof b, "{%s,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}", m);
    ok(!req(b, 1, 0, &s) && !s.on, "flag off: tools fall through to W1-SERVER's own 400");
    snprintf(b, sizeof b, "{%s,\"response_format\":{\"type\":\"json_object\"}}", m);
    ok(!req(b, 1, 1, &s) && s.on && s.reasoning && s.gbnf && strstr(s.gbnf, "st-think-0") &&
           sentence(s.gbnf, "ok\n</think>\n\n{\"k\": [1, {}]}") == 1 && sentence(s.gbnf, "ok\n</think>\n\n[1]") == 0,
        "json_object: an object, after the reasoning");
    st_free(&s);
    snprintf(b, sizeof b, "{%s,\"chat_template_kwargs\":{\"enable_thinking\":false},\"response_format\":{\"type\":\"json_schema\","
                          "\"json_schema\":{\"name\":\"x\",\"strict\":true,\"schema\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"maxItems\":2}}}}", m);
    ok(!req(b, 1, 1, &s) && s.on && !s.reasoning && sentence(s.gbnf, "[1, 2]") == 1 && sentence(s.gbnf, "[1, 2, 3]") == 0,
        "json_schema (thinking off): exactly the schema from the first token");
    st_free(&s);
    ok(!req("{\"prompt\":\"x\",\"grammar\":\"root ::= \\\"yes\\\" | \\\"no\\\"\"}", 0, 1, &s) && s.on && !s.reasoning &&
           !strcmp(s.gbnf, "root ::= \"yes\" | \"no\""),
        "completions grammar: GBNF passed through verbatim");
    st_free(&s);
    e = req("{\"prompt\":\"x\",\"grammar\":\"root ::= foo\"}", 0, 1, &s);
    ok(e && strstr(e, "grammar:"), "bad GBNF: 400 with the compiler's message");
    st_free(&s);
    e = req("{\"prompt\":\"x\",\"json_schema\":{\"type\":\"string\",\"pattern\":\"a+\"}}", 0, 1, &s);
    ok(e && strstr(e, "json_schema:"), "unsupported schema keyword: 400 (never widened)");
    st_free(&s);
    e = req("{\"prompt\":\"x\",\"grammar\":\"root ::= \\\"a\\\"\",\"json_schema\":{}}", 0, 1, &s);
    ok(e && strstr(e, "at most one"), "two constraints: 400");
    st_free(&s);
    snprintf(b, sizeof b, "{%s,\"tools\":%s,\"tool_choice\":\"required\",\"chat_template_kwargs\":{\"enable_thinking\":false}}", m, TOOLS);
    ok(!req(b, 1, 1, &s) && s.on && s.tools && s.forced && s.prompt && strstr(s.prompt, "<tools>") &&
           sentence(s.gbnf, CALL3) == 1 && sentence(s.gbnf, "hello") == 0,
        "tools required: the tool template + a forced-call grammar");
    st_free(&s);
    snprintf(b, sizeof b, "{%s,\"tools\":%s}", m, TOOLS);
    ok(!req(b, 1, 1, &s) && s.on && s.tools && !s.forced && !s.gbnf && s.reasoning && s.prompt &&
           strstr(s.prompt, "<|im_start|>assistant\n<think>\n") ,
        "tools auto: the tool template, no grammar, parser on");
    st_free(&s);
    snprintf(b, sizeof b, "{%s,\"tools\":%s,\"tool_choice\":\"none\"}", m, TOOLS);
    ok(!req(b, 1, 1, &s) && s.on && !s.tools && !strstr(s.prompt, "<tools>"), "tools none: tools left out of the prompt");
    st_free(&s);
    snprintf(b, sizeof b, "{%s,\"tools\":%s,\"response_format\":{\"type\":\"json_object\"}}", m, TOOLS);
    e = req(b, 1, 1, &s);
    ok(e && strstr(e, "cannot be combined"), "tools + response_format: 400");
    st_free(&s);
    e = req("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a<b\"}}]}", 1, 1, &s);
    ok(e && strstr(e, "name"), "tool name with '<': 400");
    st_free(&s);
    e = req("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"functions\":[]}", 1, 1, &s);
    ok(e && strstr(e, "deprecated"), "functions: 400 (send tools)");
    st_free(&s);
    ok(!req("{\"messages\":[{\"role\":\"user\",\"content\":\"w?\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"type\":\"function\","
            "\"function\":{\"name\":\"now\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"content\":\"12:00\"}]}", 1, 1, &s) &&
           s.on && s.prompt && strstr(s.prompt, "<tool_response>\n12:00\n</tool_response>"),
        "tool history without tools: rendered by the tool template");
    st_free(&s);
}

int main(int argc, char **argv) {
    V = bytevocab();
    const char *mode = argc > 1 ? argv[1] : "unit";
    if (!strcmp(mode, "unit")) {
        unit_until();
        unit_prefix();
        unit_wrap();
        unit_tools();
        unit_request();
        printf("st_test: %d PASS / %d FAIL\n", npass, nfail);
        return nfail;
    }
    if (!strcmp(mode, "prefix") && argc > 2) {
        char *g = slurp(stdin), *p = st_gbnf_prefix(g, argv[2]);
        fputs(p, stdout);
        return 0;
    }
    if (!strcmp(mode, "wrap")) {
        char *g = slurp(stdin);
        fputs(st_think_wrap(g), stdout);
        return 0;
    }
    if (!strcmp(mode, "tools") && argc > 2) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) return perror(argv[2]), 2;
        const char *err = NULL;
        jv *t = json_parse(slurp(f), &err);
        char *g = t ? st_tools_gbnf(t, argc > 3 && strcmp(argv[3], "-") ? argv[3] : NULL, argc > 4 ? atoi(argv[4]) : 1, &err) : NULL;
        if (!g) return fprintf(stderr, "%s\n", err), 1;
        fputs(g, stdout);
        return 0;
    }
    if (!strcmp(mode, "request") && argc > 4) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) return perror(argv[2]), 2;
        const char *err = NULL;
        jv *b = json_parse(slurp(f), &err);
        StReq s = {0};
        const jv *kw = json_get(b, "chat_template_kwargs"), *et = kw ? json_get(kw, "enable_thinking") : NULL;
        const char *r = st_request(b, atoi(argv[3]), et && et->type == J_BOOL ? et->b : 1, 1, atoi(argv[4]), &s);
        printf("error: %s\non=%d reasoning=%d tools=%d forced=%d\n--- grammar\n%s\n--- prompt\n%s\n", r ? r : "-", s.on, s.reasoning,
            s.tools, s.forced, s.gbnf ? s.gbnf : "-", s.prompt ? s.prompt : "-");
        return 0;
    }
    fprintf(stderr, "usage: see the header\n");
    return 2;
}
