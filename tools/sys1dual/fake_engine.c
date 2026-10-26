// tools/sys1dual/fake_engine.c — G16 test double for the SYS1 engine behind llb-serve: the
// LLB_SYS1=1 + LLB_SYS1_STREAM=1 stdin/stdout protocol of dual.c, with the REAL tokenizer
// (tools/qwen35_tokenizer.c) and G1's REAL grammar mask (host/llb_grammar.h) over the real
// vocabulary, but no model: CPU only, for the server's structured-output tests.
//   generate {"prompt","n","special"?,"grammar"?,"sampling":{"seed"}?}
//     -> {"token":ID,"text":PIECE} per step (UTF-8 tail held back as dual.c), then
//        {"prompt_ids":[...],"ids":[...],"text":"..."}; exactly n steps, like the engine.
//   The "model": FAKE_SCRIPT (JSON lines {"match":SUBSTR,"text":TARGET}); the first entry
//   whose match occurs in the prompt is the target. Unconstrained: the target's tokens
//   (special parsed, so "<|im_end|>" is EOG), then "<|endoftext|><|im_start|>user\n" and
//   filler, as an engine with no stop would. Constrained: each step takes the allowed
//   token with the longest piece that prefixes the rest of the target (teacher forcing);
//   with no such token (or no target) a seeded random allowed token biased to short pieces
//   and to ending (G1's gram_test gen); once the grammar is done, EOG only (as the engine).
// Env: LLB_TOKENIZER (tokenizer.bin; its directory is the tokenizer dir), FAKE_SCRIPT,
// FAKE_LOG (every request line appended).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../clef3/json.h"
#include "../qwen35_tokenizer.h"
#include "../../host/llb_grammar.h"
#include <libgen.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Q35Tokenizer tok;
static LlbGVocab *gv;
static jv *script;
static uint64_t rs;
static double urand(void) {
    rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17;
    return (double)(rs >> 11) / 9007199254740992.0;
}

static int utf8_whole(const char *b, int n) { // dual.c's
    int i = n;
    while (i > 0 && i > n - 4 && ((unsigned char)b[i - 1] & 0xC0) == 0x80) i--;
    if (i == 0) return 0;
    unsigned char c = (unsigned char)b[i - 1];
    int need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    return n - (i - 1) >= need ? n : i - 1;
}

static char spend[64];
static int spn;
static void event(int id) {
    char piece[4096], buf[4200];
    int g = q35_tok_decode(&tok, &id, 1, piece, sizeof piece);
    if (g > (int)sizeof piece) g = sizeof piece;
    memcpy(buf, spend, spn);
    memcpy(buf + spn, piece, g);
    int n = spn + g, w = utf8_whole(buf, n);
    if (n - w > (int)sizeof spend) w = n;
    spn = n - w;
    memcpy(spend, buf + w, spn);
    buf[w] = 0;
    sb e = {0};
    sb_fmt(&e, "{\"token\":%d,\"text\":", id);
    json_quote(&e, buf);
    sb_add(&e, "}\n");
    fputs(e.p, stdout);
    free(e.p);
}

static const char *target_for(const char *prompt) {
    for (size_t i = 0; script && i < script->n; i++) {
        const jv *m = json_get(script->items[i], "match"), *t = json_get(script->items[i], "text");
        if (m && t && strstr(prompt, m->str)) return t->str;
    }
    return "";
}

int main(void) {
    const char *tb = getenv("LLB_TOKENIZER");
    if (!tb) return fprintf(stderr, "fake: set LLB_TOKENIZER\n"), 2;
    char *d = strdup(tb), err[256];
    if (q35_tok_load(&tok, dirname(d))) return fprintf(stderr, "fake: tokenizer load failed\n"), 2;
    gv = llb_gv_load(tb, err, sizeof err);
    if (!gv) return fprintf(stderr, "fake: %s\n", err), 2;
    const char *sp = getenv("FAKE_SCRIPT");
    if (sp) {
        FILE *f = fopen(sp, "rb");
        sb all = {0};
        char *l = NULL;
        size_t cap = 0;
        sb_add(&all, "[");
        int k = 0;
        while (f && getline(&l, &cap, f) > 0)
            if (*l == '{') sb_add(&all, k++ ? "," : ""), sb_add(&all, l);
        sb_add(&all, "]");
        const char *e = NULL;
        script = json_parse(all.p, &e);
        if (!script) return fprintf(stderr, "fake: FAKE_SCRIPT: %s\n", e), 2;
        if (f) fclose(f);
    }
    FILE *lg = getenv("FAKE_LOG") ? fopen(getenv("FAKE_LOG"), "a") : NULL;
    int nv = llb_gv_n_vocab(gv);
    uint32_t *bits = malloc(((size_t)nv + 31) / 32 * 4);
    fprintf(stderr, "sys1: FP32 clef head + tokenizer ready 0.000 ms (fake engine)\n");
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, stdin) > 0) {
        if (lg) fputs(line, lg), fflush(lg);
        const char *e = NULL;
        jv *r = json_parse(line, &e);
        const jv *p = r ? json_get(r, "prompt") : NULL, *n = r ? json_get(r, "n") : NULL;
        if (!p || p->type != J_STR || !n || n->type != J_NUM) {
            printf("{\"error\":\"fake engine: generate requests only\"}\n");
            fflush(stdout);
            json_free(r);
            continue;
        }
        const jv *spc = json_get(r, "special"), *gr = json_get(r, "grammar"), *sa = json_get(r, "sampling");
        const jv *seed = sa ? json_get(sa, "seed") : NULL;
        rs = seed && seed->type == J_NUM ? (uint64_t)seed->num * 2654435761u + 1 : 88172645463325252ull;
        int pids[16384];
        int pc = q35_tok_encode(&tok, p->str, spc && spc->type == J_BOOL && spc->b, pids, 16384);
        LlbGram *g = NULL;
        if (gr && gr->type == J_STR && !(g = llb_gram_from_gbnf(gr->str, NULL, err, sizeof err))) {
            printf("{\"error\":\"bad json_schema / grammar (see stderr)\"}\n");
            fprintf(stderr, "%s\n", err);
            fflush(stdout);
            json_free(r);
            continue;
        }
        const char *tgt = target_for(p->str);
        size_t tl = strlen(tgt), at = 0;
        int steps = (int)n->num, *ids = malloc(sizeof(int) * (size_t)steps), ni = 0;
        spn = 0;
        if (!g) { // unconstrained: the target, then what an engine with no stop runs into
            sb s = {0};
            sb_add(&s, tgt);
            sb_add(&s, "<|endoftext|><|im_start|>user\n");
            for (int i = 0; i < 64; i++) sb_add(&s, " filler");
            int *t = malloc(sizeof(int) * 8192), c = q35_tok_encode(&tok, s.p, 1, t, 8192);
            for (int i = 0; i < steps; i++) ids[ni++] = t[i < c ? i : c - 1], event(ids[ni - 1]), fflush(stdout);
            free(t), free(s.p);
        } else {
            LlbGMatch *m = llb_gm_new(g, gv);
            sb out = {0};
            sb_add(&out, "");
            for (int i = 0; i < steps; i++) {
                int best = -1;
                if (llb_gm_done(m)) best = 248046;
                else {
                    if (!llb_gm_mask(m, bits, nv)) {
                        fprintf(stderr, "fake: grammar dead end\n");
                        return 3; // the real engine err_fails here too
                    }
                    size_t bl = 0;
                    for (int t = 0; at < tl && t < nv; t++) {
                        if (!(bits[t / 32] >> (t % 32) & 1)) continue;
                        const unsigned char *pc2;
                        size_t l = (size_t)llb_gv_piece(gv, t, &pc2);
                        if (l > bl && l <= tl - at && !memcmp(pc2, tgt + at, l)) bl = l, best = t;
                    }
                    if (best >= 0) at += bl;
                    else if (at >= tl && llb_gm_can_end(m)) best = 248046;
                    else {
                        float bv = -INFINITY;
                        for (int t = 0; t < nv; t++) {
                            if (!(bits[t / 32] >> (t % 32) & 1)) continue;
                            const unsigned char *pc2;
                            int l = llb_gv_piece(gv, t, &pc2);
                            float x = (float)(4.0 * urand()) - 0.5f * (float)l + (t >= 248044 && t <= 248046 ? 2.0f + 0.05f * (float)i : 0.0f);
                            // free text ends only at a closer: favour pieces that start one,
                            // more the longer the run (else random strings never terminate)
                            // (and push hard to finish a closer already begun)
                            static const char *cl[] = {"\n</parameter>\n", "</function>", "</think>", "\"", "]", "}", NULL};
                            for (int k = 0; cl[k] && l > 0; k++) {
                                size_t ck = strlen(cl[k]), hv = 0;
                                for (size_t h = ck - 1; h >= 1 && !hv; h--)
                                    if (out.n >= h && !memcmp(out.p + out.n - h, cl[k], h)) hv = h;
                                const char *w = cl[k] + hv;
                                size_t wn = ck - hv, mn = (size_t)l < wn ? (size_t)l : wn;
                                if (!memcmp(pc2, w, mn)) { x += 0.08f * (float)i + (hv ? 6.0f : 0.0f); break; }
                            }
                            if (x > bv) bv = x, best = t;
                        }
                        at = tl; // off the script: random from here on
                    }
                    if (llb_gm_accept(m, best)) return fprintf(stderr, "fake: accept refused\n"), 3;
                    const unsigned char *bp;
                    int bl2 = llb_gv_piece(gv, best, &bp);
                    sb_addn(&out, (const char *)bp, (size_t)bl2);
                }
                ids[ni++] = best;
                event(best);
                fflush(stdout);
            }
            llb_gm_free(m);
            llb_gram_free(g);
            free(out.p);
        }
        sb o = {0};
        sb_add(&o, "{\"prompt_ids\":[");
        for (int i = 0; i < pc; i++) sb_fmt(&o, i ? ",%d" : "%d", pids[i]);
        sb_add(&o, "],\"ids\":[");
        for (int i = 0; i < ni; i++) sb_fmt(&o, i ? ",%d" : "%d", ids[i]);
        sb_add(&o, "],\"text\":");
        char *txt = malloc(1 << 20);
        int tn = q35_tok_decode(&tok, ids, ni, txt, (1 << 20) - 1);
        txt[tn < (1 << 20) - 1 ? tn : (1 << 20) - 1] = 0;
        json_quote(&o, txt);
        sb_add(&o, "}\n");
        fputs(o.p, stdout);
        fflush(stdout);
        free(o.p), free(txt), free(ids);
        json_free(r);
    }
    return 0;
}
