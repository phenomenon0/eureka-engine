// tools/qwen35_toktest.c — exercise the C tokenizer against a corpus.
//   qwen35_toktest <packdir> encode <in.bin>   # [u32 len][bytes]... -> ids per line
//   qwen35_toktest <packdir> decode <ids.bin>  # [u32 n][i32 id]...    -> detok bytes per line
//   qwen35_toktest <packdir> string <text>      -> ids (space separated)
//   qwen35_toktest <packdir> decode_ids <id>... -> detok to stdout
#include "qwen35_tokenizer.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_ids(const int *a, int n) {
    for (int i = 0; i < n; i++) printf("%d%c", a[i], i + 1 == n ? '\n' : ' ');
    if (n == 0) printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <packdir> encode|decode|string|decode_ids ...\n", argv[0]); return 2; }
    Q35Tokenizer t;
    if (q35_tok_load(&t, argv[1])) return 1;
    const char *mode = argv[2];

    if (!strcmp(mode, "string")) {
        int cap = 1 << 20;
        int *ids = (int *)malloc(sizeof(int) * cap);
        int n = q35_tok_encode(&t, argc > 3 ? argv[3] : "", 1, ids, cap);
        if (n > cap) { ids = (int *)realloc(ids, sizeof(int) * n); cap = n; n = q35_tok_encode(&t, argv[3], 1, ids, cap); }
        print_ids(ids, n);
        free(ids);
    } else if (!strcmp(mode, "encode")) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror("in.bin"); return 1; }
        while (1) {
            uint32_t len;
            if (fread(&len, 4, 1, f) != 1) break;
            char *buf = (char *)malloc(len + 1);
            if (fread(buf, 1, len, f) != len) { free(buf); break; }
            buf[len] = 0;
            int cap = 1 << 20;
            int *ids = (int *)malloc(sizeof(int) * cap);
            int n = q35_tok_encode(&t, buf, 1, ids, cap);
            if (n > cap) { free(ids); cap = n; ids = (int *)malloc(sizeof(int) * cap); n = q35_tok_encode(&t, buf, 1, ids, cap); }
            print_ids(ids, n);
            free(ids); free(buf);
        }
        fclose(f);
    } else if (!strcmp(mode, "decode")) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) { perror("ids.bin"); return 1; }
        while (1) {
            uint32_t n;
            if (fread(&n, 4, 1, f) != 1) break;
            int *ids = (int *)malloc(4 * (n ? n : 1));
            if (fread(ids, 4, n, f) != n) { free(ids); break; }
            int need = q35_tok_decode(&t, ids, n, NULL, 0);
            char *out = (char *)malloc((size_t)need + 1);
            int nb = q35_tok_decode(&t, ids, n, out, need);
            uint32_t un = (uint32_t)nb;
            fwrite(&un, 4, 1, stdout);
            fwrite(out, 1, nb, stdout);
            free(out); free(ids);
        }
        fclose(f);
    } else if (!strcmp(mode, "decode_ids")) {
        int n = argc - 3;
        int *ids = (int *)malloc(4 * (n ? n : 1));
        for (int i = 0; i < n; i++) ids[i] = atoi(argv[3 + i]);
        int need = q35_tok_decode(&t, ids, n, NULL, 0);
        char *out = (char *)malloc((size_t)need + 1);
        int nb = q35_tok_decode(&t, ids, n, out, need);
        fwrite(out, 1, nb, stdout);
        free(out); free(ids);
    } else { fprintf(stderr, "unknown mode\n"); return 2; }
    q35_tok_free(&t);
    return 0;
}
