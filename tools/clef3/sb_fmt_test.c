// sb_fmt_test.c — SPEED45-FIX regression: sb_fmt must not cut its output at 511 bytes.
// The served engine builds its command line as sb_fmt("%s --sample %s", ids_steps_pos, spec); a cut there
// made the engine exit rc 2 on every prompt of >= 82 tokens that carried a sampling field.
//   cc -D_POSIX_C_SOURCE=200809L -std=c11 -O2 tools/clef3/sb_fmt_test.c tools/clef3/json.c -lm -o /tmp/sb_fmt_test && /tmp/sb_fmt_test
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int fails = 0;
    // a 1133-token command, as the SPEED45 1K warm-up request built it
    sb cmd = {0};
    for (int i = 0; i < 1133; i++) sb_fmt(&cmd, "%d ", 200000 + i);
    sb_fmt(&cmd, "--steps %d --pos %d", 16, 0);
    sb c = {0};
    sb_fmt(&c, "%s --sample %s", cmd.p, "preset=greedy");
    sb_fmt(&c, ",%s=%.9g", "temp", 0.0);
    size_t want = strlen(cmd.p) + strlen(" --sample preset=greedy,temp=0");
    const char *tail = "--steps 16 --pos 0 --sample preset=greedy,temp=0";
    if (strlen(c.p) != want || c.n != want) printf("FAIL long: len %zu n %zu want %zu\n", strlen(c.p), c.n, want), fails++;
    if (strlen(c.p) < strlen(tail) || strcmp(c.p + strlen(c.p) - strlen(tail), tail)) printf("FAIL long: tail lost\n"), fails++;
    // boundary lengths around the old buffer
    for (int len = 505; len <= 520; len++) {
        char *x = malloc(len + 1);
        memset(x, 'a', len);
        x[len] = 0;
        sb t = {0};
        sb_fmt(&t, "%s|", x);
        if (t.n != (size_t)len + 1 || t.p[len] != '|') printf("FAIL boundary %d: n %zu\n", len, t.n), fails++;
        free(x);
        free(t.p);
    }
    // short outputs unchanged
    sb s = {0};
    sb_fmt(&s, "%d,%s", 7, "x");
    if (strcmp(s.p, "7,x")) printf("FAIL short: %s\n", s.p), fails++;
    printf("%s (%d fails)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
