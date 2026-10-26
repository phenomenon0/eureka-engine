// gate 3: stop-string matcher.  Each case: stop list, the pieces (detok bytes per token) fed in
// order, exact expected emitted text and whether a stop fired.  Every intermediate emission must
// also be a prefix of the expected text (nothing that will be trimmed is ever released early) and,
// after each feed, the text released so far must end on a whole UTF-8 character.
#include <stdio.h>
#include <string.h>
#include "qwen35_sampler.h"

typedef struct { const char *name; const char *stops[4]; const char *pieces[6]; const char *expect; int stopped; } Case;

static const Case C[] = {
 {"stop at very start",        {"STOP"},        {"STOP", "abc"},                 "",            1},
 {"stop at very end",          {"STOP"},        {"Hello ", "world", "STOP"},     "Hello world", 1},
 {"straddles token boundary",  {"</s>"},        {"abc<", "/", "s>def"},          "abc",         1},
 {"straddles 3 tokens",        {"<|im_end|>"},  {"hi<|im", "_e", "nd|>junk"},    "hi",          1},
 {"overlapping prefixes",      {"aab"},         {"aa", "a", "b"},                "a",           1},
 {"false start then real",     {"abc"},         {"ab", "x", "abc"},              "abx",         1},
 {"partial at end, no match",  {"abc"},         {"xxab"},                        "xxab",        0},
 {"utf8 2-byte split",         {"END"},         {"\xC3", "\xA9", "xEND"},        "\xC3\xA9x",   1},
 {"utf8 3-byte split in 3",    {"zzz"},         {"a\xE2", "\x82", "\xAC" "b"},   "a\xE2\x82\xAC" "b", 0},
 {"multibyte stop string",     {"\xE4\xB8\x96\xE7\x95\x8C"}, {"\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96", "\xE7\x95\x8C!"}, "\xE4\xBD\xA0\xE5\xA5\xBD", 1},
 {"two stops, earliest wins",  {"foo", "bar"},  {"xxbarfoo"},                    "xx",          1},
 {"two stops, later-listed first in text", {"foo", "bar"}, {"ab", "bar", "foo"}, "ab",          1},
 {"stop inside one big piece", {"STOP"},        {"one two STOP three"},          "one two ",    1},
 {"prefix-of-other stops",     {"ab", "abc"},   {"xabc"},                        "x",           1},
 {"empty stop string ignored", {"", "Q"},       {"aQb"},                         "a",           1},
};

int main(void) {
    int nfail = 0, n = (int)(sizeof C / sizeof C[0]);
    for (int i = 0; i < n; i++) {
        int ns = 0; while (ns < 4 && C[i].stops[ns]) ns++;
        Q35Stop *S = q35stop_new(C[i].stops, ns);
        char got[512]; int gl = 0, bad = 0;
        for (int p = 0; p < 6 && C[i].pieces[p]; p++) {
            int ol; const char *o = q35stop_feed(S, C[i].pieces[p], (int)strlen(C[i].pieces[p]), &ol);
            memcpy(got + gl, o, (size_t)ol); gl += ol;
            if (gl > (int)strlen(C[i].expect) || memcmp(got, C[i].expect, (size_t)gl)) bad = 1;       // released something it should not have
            if (gl > 0 && ((unsigned char)got[gl - 1] & 0xC0) == 0xC0) bad = 1;                       // lead byte released without its tail
            if (gl > 1 && ((unsigned char)got[gl - 2] & 0xE0) == 0xE0 && ((unsigned char)got[gl - 1] & 0xC0) == 0x80) bad = 1;
        }
        if (!q35stop_stopped(S)) { int ol; const char *o = q35stop_flush(S, &ol); memcpy(got + gl, o, (size_t)ol); gl += ol; }
        if (gl != (int)strlen(C[i].expect) || memcmp(got, C[i].expect, (size_t)gl) || q35stop_stopped(S) != C[i].stopped) bad = 1;
        printf("%-42s %s\n", C[i].name, bad ? "FAIL" : "ok");
        nfail += bad; q35stop_free(S);
    }
    printf("gate3: %d cases, %d failing\n", n, nfail);
    return nfail != 0;
}
