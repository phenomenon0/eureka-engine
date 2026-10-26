// tools/llb/llb-tok.c - the C helper behind `llb`: GGUF header, tokenizer, chat template, engine runner.
//   llb-tok info GGUF                  one line: arch layers ftype size key (key = size-mtime-fnv64 of the first MiB)
//   llb-tok extract GGUF DIR           DIR/tokenizer.bin, byte-identical to tools/vocab_extract_qwen35.py
//   llb-tok enc DIR [TEXT]             ids (TEXT or stdin), plain text: control tokens (<|im_start|>...) NOT parsed
//   llb-tok dec DIR [ID...]            text (ids from args or stdin)
//   llb-tok chat DIR HIST [--text] [--think 0|1|2]   ChatML ids for HIST (records "role\0content\0"), + generation prompt
//   llb-tok run DIR [--ids] [-m NAME] [-P N] [--save F] [--log F] [-v] -- CMD...
//                                      runs the engine CMD, streams text, stops at end-of-text, prints a summary
// Builds with any C compiler: cc -O2 -o llb-tok llb-tok.c
#define _POSIX_C_SOURCE 200809L
#include "../qwen35_tokenizer.c"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void die(const char *a, const char *b) { fprintf(stderr, "llb: %s%s\n", a, b ? b : ""); exit(1); }

// ---- GGUF metadata ---------------------------------------------------------
typedef struct { FILE *f; const char *path; } G;
static void gread(G *g, void *p, size_t n) { if (fread(p, 1, n, g->f) != n) die("truncated GGUF header: ", g->path); }
static uint64_t gu64(G *g) { uint64_t v; gread(g, &v, 8); return v; }
static uint32_t gu32(G *g) { uint32_t v; gread(g, &v, 4); return v; }
static char *gstr(G *g, uint64_t *len) {
    uint64_t n = gu64(g);
    if (n > (1u << 30)) die("bad string in GGUF header: ", g->path);
    char *s = malloc(n + 1); gread(g, s, n); s[n] = 0;
    if (len) *len = n;
    return s;
}
static const int gsize[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
static int64_t gscalar(G *g, uint32_t ty) {  // integer-ish scalar
    uint64_t v = 0;
    gread(g, &v, gsize[ty]);
    if (ty == 1) return (int8_t)v;
    if (ty == 3) return (int16_t)v;
    if (ty == 5) return (int32_t)v;
    return (int64_t)v;
}
static void gskip(G *g, uint32_t ty) {
    if (ty == 8) { free(gstr(g, NULL)); return; }
    if (ty == 9) {
        uint32_t et = gu32(g); uint64_t n = gu64(g);
        if (et == 8) for (uint64_t i = 0; i < n; i++) free(gstr(g, NULL));
        else if (et < 13 && gsize[et]) fseeko(g->f, (off_t)(n * gsize[et]), SEEK_CUR);
        else for (uint64_t i = 0; i < n; i++) gskip(g, et);
        return;
    }
    if (ty > 12) die("bad value type in GGUF header: ", g->path);
    fseeko(g->f, gsize[ty], SEEK_CUR);
}
static void gopen(G *g, const char *path, uint64_t *nkv) {
    g->path = path;
    g->f = fopen(path, "rb");
    if (!g->f) die("cannot open ", path);
    char m[4];
    if (fread(m, 1, 4, g->f) != 4 || memcmp(m, "GGUF", 4)) die("not a GGUF file: ", path);
    uint32_t ver = gu32(g);
    if (ver < 2) die("GGUF version 1 is not supported: ", path);
    gu64(g);  // tensor count
    *nkv = gu64(g);
}

static int cmd_info(const char *path) {
    G g; uint64_t nkv; gopen(&g, path, &nkv);
    char *arch = NULL, *name = NULL; int64_t layers = -1, ftype = -1;
    for (uint64_t i = 0; i < nkv && !(arch && layers >= 0 && ftype >= 0 && name); i++) {
        uint64_t kl; char *k = gstr(&g, &kl); uint32_t ty = gu32(&g);
        if (!strcmp(k, "general.architecture") && ty == 8) arch = gstr(&g, NULL);
        else if (!strcmp(k, "general.name") && ty == 8) name = gstr(&g, NULL);
        else if (!strcmp(k, "general.file_type") && ty < 13 && ty != 6 && ty != 8 && ty != 9) ftype = gscalar(&g, ty);
        else if (arch && kl > strlen(arch) && !strncmp(k, arch, strlen(arch)) && !strcmp(k + strlen(arch), ".block_count") && ty < 13 && ty != 6 && ty != 8 && ty != 9) layers = gscalar(&g, ty);
        else gskip(&g, ty);
        free(k);
    }
    struct stat st; fstat(fileno(g.f), &st);
    fseeko(g.f, 0, SEEK_SET);
    static unsigned char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf, g.f);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= 0x100000001b3ull; }
    for (char *p = name; p && *p; p++) if (*p == ' ' || *p == '\t' || *p == '\n') *p = '_';
    printf("arch=%s layers=%lld ftype=%lld size=%lld key=%lld-%lld-%016llx name=%s\n", arch ? arch : "?", (long long)layers,
           (long long)ftype, (long long)st.st_size, (long long)st.st_size, (long long)st.st_mtime, (unsigned long long)h,
           name ? name : "?");
    fclose(g.f);
    return 0;
}

static int cmd_meta(const char *path, const char *key) {  // print one string/int value
    G g; uint64_t nkv; gopen(&g, path, &nkv);
    for (uint64_t i = 0; i < nkv; i++) {
        char *k = gstr(&g, NULL); uint32_t ty = gu32(&g);
        if (!strcmp(k, key) && ty == 8) { uint64_t n; char *v = gstr(&g, &n); fwrite(v, 1, n, stdout); return 0; }
        if (!strcmp(k, key) && ty < 13 && ty != 6 && ty != 9) { printf("%lld\n", (long long)gscalar(&g, ty)); return 0; }
        gskip(&g, ty); free(k);
    }
    return 1;
}

// ---- tokenizer.bin extraction (layout: tools/vocab_extract_qwen35.py) ------
static int cplen(const char *s, uint64_t n) { int c = 0; for (uint64_t i = 0; i < n; i++) c += ((unsigned char)s[i] & 0xC0) != 0x80; return c; }

static int cmd_extract(const char *path, const char *dir) {
    G g; uint64_t nkv; gopen(&g, path, &nkv);
    char **tok = NULL, **mer = NULL; uint64_t *tlen = NULL, *mlen = NULL, ntok = 0, nmer = 0, ntype = 0;
    char *tpl = NULL;
    int32_t *type = NULL; int32_t bos = -1, eos = -1, pad = -1; uint32_t add_bos = 0;
    for (uint64_t i = 0; i < nkv; i++) {
        char *k = gstr(&g, NULL); uint32_t ty = gu32(&g);
        if (ty == 9 && (!strcmp(k, "tokenizer.ggml.tokens") || !strcmp(k, "tokenizer.ggml.merges"))) {
            uint32_t et = gu32(&g); uint64_t n = gu64(&g);
            if (et != 8) die("unexpected array type in ", path);
            char **a = malloc(sizeof(char *) * n); uint64_t *l = malloc(8 * n);
            for (uint64_t j = 0; j < n; j++) a[j] = gstr(&g, &l[j]);
            if (k[15] == 't') { tok = a; tlen = l; ntok = n; } else { mer = a; mlen = l; nmer = n; }
        } else if (ty == 9 && !strcmp(k, "tokenizer.ggml.token_type")) {
            uint32_t et = gu32(&g); ntype = gu64(&g);
            if (et == 8 || et == 9 || et > 12) die("unexpected token_type array in ", path);
            type = malloc(4 * ntype);
            for (uint64_t j = 0; j < ntype; j++) type[j] = (int32_t)gscalar(&g, et);
        } else if (ty == 8 && !strcmp(k, "tokenizer.chat_template")) tpl = gstr(&g, NULL);
        else if (ty != 8 && ty != 9 && ty != 6 && ty < 13 && !strcmp(k, "tokenizer.ggml.bos_token_id")) bos = (int32_t)gscalar(&g, ty);
        else if (ty != 8 && ty != 9 && ty != 6 && ty < 13 && !strcmp(k, "tokenizer.ggml.eos_token_id")) eos = (int32_t)gscalar(&g, ty);
        else if (ty != 8 && ty != 9 && ty != 6 && ty < 13 && !strcmp(k, "tokenizer.ggml.padding_token_id")) pad = (int32_t)gscalar(&g, ty);
        else if (ty != 8 && ty != 9 && ty != 6 && ty < 13 && !strcmp(k, "tokenizer.ggml.add_bos_token")) add_bos = gscalar(&g, ty) != 0;
        else gskip(&g, ty);
        free(k);
    }
    fclose(g.f);
    if (!tok || !mer || !type || ntype != ntok) die("no BPE tokenizer in ", path);
    uint32_t maxlen = 0;
    for (uint64_t i = 0; i < ntok; i++) { uint32_t c = cplen(tok[i], tlen[i]); if (c > maxlen) maxlen = c; }
    // specials (types 2,3,4), stable-sorted by descending code-point length (Python's sort)
    int32_t *sp = malloc(4 * (ntok + 1)); uint32_t nsp = 0;
    for (uint64_t i = 0; i < ntok; i++) if (type[i] >= 2 && type[i] <= 4) {
        int c = cplen(tok[i], tlen[i]); uint32_t j = nsp++;
        while (j > 0 && cplen(tok[sp[j - 1]], tlen[sp[j - 1]]) < c) { sp[j] = sp[j - 1]; j--; }
        sp[j] = (int32_t)i;
    }
    mkdir(dir, 0755);
    char out[4096]; snprintf(out, sizeof out, "%s/tokenizer.bin.tmp", dir);
    FILE *f = fopen(out, "wb");
    if (!f) die("cannot write ", out);
    uint32_t hdr[8] = {(uint32_t)ntok, (uint32_t)nmer, maxlen, (uint32_t)bos, (uint32_t)eos, (uint32_t)pad, add_bos, nsp};
    fwrite("Q35TOK01", 1, 8, f); fwrite(hdr, 4, 8, f); fwrite(sp, 4, nsp, f);  // little-endian hosts only
    for (uint64_t i = 0; i < ntok; i++) {
        uint16_t l = (uint16_t)tlen[i]; uint8_t t = (uint8_t)type[i];
        fwrite(&l, 2, 1, f); fwrite(&t, 1, 1, f); fwrite(tok[i], 1, tlen[i], f);
    }
    for (uint64_t i = 0; i < nmer; i++) {
        char *s = mer[i], *spc = strchr(s, ' ');
        uint16_t a = spc ? (uint16_t)(spc - s) : (uint16_t)mlen[i], b = spc ? (uint16_t)(mlen[i] - a - 1) : 0;
        fwrite(&a, 2, 1, f); fwrite(s, 1, a, f); fwrite(&b, 2, 1, f); if (b) fwrite(spc + 1, 1, b, f);
    }
    if (fclose(f)) die("cannot write ", out);
    char fin[4096]; snprintf(fin, sizeof fin, "%s/tokenizer.bin", dir);
    if (rename(out, fin)) die("cannot write ", fin);
    snprintf(fin, sizeof fin, "%s/chat.flags", dir);
    FILE *cf = fopen(fin, "w");
    if (!cf) die("cannot write ", fin);
    fprintf(cf, "%d %d %d %d\n", tpl && strstr(tpl, "Reasoning effort is set to xhigh") != NULL,
            tpl && strstr(tpl, "preserve_thinking is undefined or preserve_thinking is true") != NULL,
            tpl && strstr(tpl, "enable_thinking is defined and enable_thinking is false") ? 1 : 2,
            tpl && strstr(tpl, "content.split('</think>')") != NULL);
    fclose(cf);
    return 0;
}

// ---- tokenizer front ---------------------------------------------------------
static char *slurp(FILE *f, size_t *len) {
    size_t cap = 1 << 16, n = 0; char *b = malloc(cap);
    for (size_t r; (r = fread(b + n, 1, cap - n - 1, f)) > 0;) { n += r; if (n + 1 >= cap) b = realloc(b, cap *= 2); }
    b[n] = 0; if (len) *len = n;
    return b;
}
static void load(Q35Tokenizer *t, const char *dir) { if (q35_tok_load(t, dir)) die("no tokenizer in ", dir); }
// User text is plain: control tokens are not parsed (llama.cpp parse_special=false; user-defined ones like <think>
// still split, as there). A prompt that contains "<|im_end|>" stays text and cannot fake a chat turn.
static int encode(Q35Tokenizer *t, const char *text, int **ids) {
    int n = q35_tok_encode(t, text, 0, NULL, 0);
    *ids = malloc(sizeof(int) * (n + 1));
    return q35_tok_encode(t, text, 0, *ids, n);
}
static void print_ids(const int *ids, int n) { for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]); printf("\n"); }

static int cmd_enc(const char *dir, const char *text) {
    Q35Tokenizer t; load(&t, dir);
    char *own = text ? NULL : slurp(stdin, NULL);
    int *ids; int n = encode(&t, text ? text : own, &ids);
    print_ids(ids, n);
    return 0;
}
static int cmd_dec(const char *dir, int argc, char **argv) {
    Q35Tokenizer t; load(&t, dir);
    char *own = NULL, *save = NULL;
    if (argc == 0) own = slurp(stdin, NULL);
    char piece[65536];
    for (int i = 0;; i++) {
        const char *w = own ? strtok_r(i ? NULL : own, " \t\r\n,", &save) : (i < argc ? argv[i] : NULL);
        if (!w) break;
        fwrite(piece, 1, q35_tok_piece(&t, atoi(w), piece, sizeof piece), stdout);
    }
    return 0;
}

// ---- chat template ---------------------------------------------------------
// The GGUF's own Jinja, text roles only (no tools, no vision). Two Qwen variants, told apart at extract time
// (DIR/chat.flags = "effort preserve think split"): Qwen3.5 (0 0 2 1) and Qwen3.8 (1 1 1 0).
//   effort   - a default system block "Reasoning effort is set to xhigh. ..." (thinking on)
//   preserve - every assistant turn keeps its <think> block (else only turns after the last user message)
//   think    - generation prompt: 1 "<think>\n", 2 "<think>\n\n</think>\n\n"
//   split    - the template splits </think> out of the content itself (leading \n stripped); else the client
//              does (reasoning_content) and the template trims the remaining content
static const char *EFFORT = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
typedef struct { char *b, *m; size_t n, cap; } Buf;  // m[i] = 1: byte i is template text (specials parsed there)
static void putm(Buf *o, const char *s, size_t n, char m) {
    if (o->n + n + 1 > o->cap) { o->cap = (o->n + n + 1) * 2; o->b = realloc(o->b, o->cap); o->m = realloc(o->m, o->cap); }
    memcpy(o->b + o->n, s, n); memset(o->m + o->n, m, n); o->n += n; o->b[o->n] = 0;
}
static void put(Buf *o, const char *s, size_t n) { putm(o, s, n, 0); }       // conversation content: plain
static void puts_(Buf *o, const char *s) { putm(o, s, strlen(s), 1); }     // the template's own text
// Encode the rendered chat: a control/unknown special (<|im_end|>...) is matched only where all its bytes are template
// text; user-defined ones (<think>) match anywhere, as llama.cpp parse_special=false does for content. The text between
// matched specials is one BPE fragment, as in llama.cpp (so ids == llama.cpp's whenever the content holds no specials).
static int encode_chat(Q35Tokenizer *t, const Buf *o, int **ids) {
    IVec v = {0}; size_t start = 0;
    for (size_t i = 0; i < o->n;) {
        int sid = -1, sl = 0;
        for (int k = 0; k < t->n_special && sid < 0; k++) {  // longest first
            int l, id = t->special[k], ctl = t->type[id] == 3 || t->type[id] == 2; const char *s = q35_tok_text(t, id, &l);
            if (l && i + l <= o->n && !memcmp(o->b + i, s, l) && (!ctl || !memchr(o->m + i, 0, l))) sid = id, sl = l;
        }
        if (sid < 0) { i++; continue; }
        if (i > start) { char *f = strndup(o->b + start, i - start); bpe_fragment(t, f, (int)(i - start), &v); free(f); }
        iv_push(&v, sid); i += sl; start = i;
    }
    if (o->n > start) { char *f = strndup(o->b + start, o->n - start); bpe_fragment(t, f, (int)(o->n - start), &v); free(f); }
    *ids = v.a;
    return v.n;
}
static int ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static void trim(const char **s, size_t *n) {  // Jinja |trim
    while (*n && ws(**s)) { (*s)++; (*n)--; }
    while (*n && ws((*s)[*n - 1])) (*n)--;
}
static const char *memstr(const char *h, size_t n, const char *x) {
    size_t k = strlen(x);
    for (size_t i = 0; i + k <= n; i++) if (!memcmp(h + i, x, k)) return h + i;
    return NULL;
}
static const char *memrstr(const char *h, size_t n, const char *x) {
    size_t k = strlen(x);
    for (size_t i = n >= k ? n - k + 1 : 0; i-- > 0;) if (!memcmp(h + i, x, k)) return h + i;
    return NULL;
}

static int cmd_chat(const char *dir, const char *hist, int text_only, int think) {
    int effort = 0, preserve = 0, think_dflt = 2, split = 1;
    char fp[4096]; snprintf(fp, sizeof fp, "%s/chat.flags", dir);
    FILE *ff = fopen(fp, "r");
    if (!ff || fscanf(ff, "%d %d %d %d", &effort, &preserve, &think_dflt, &split) != 4) die("no chat template flags in ", dir);
    fclose(ff);
    if (think < 0) think = think_dflt;
    if (think == 2) effort = 0;  // enable_thinking=false drops the effort block
    FILE *f = fopen(hist, "rb");
    if (!f) die("cannot read ", hist);
    size_t n; char *b = slurp(f, &n); fclose(f);
    enum { MAXM = 4096 };
    static const char *role[MAXM], *con[MAXM];
    int nm = 0;
    for (size_t i = 0; i < n && nm < MAXM;) {
        role[nm] = b + i; i += strlen(b + i) + 1;
        if (i >= n) break;
        con[nm] = b + i; i += strlen(b + i) + 1;
        nm++;
    }
    if (nm == 0) die("empty conversation in ", hist);
    Buf o = {0};
    int first = 0;
    if (!strcmp(role[0], "system")) {
        const char *c = con[0]; size_t cn = strlen(c); trim(&c, &cn);
        if (cn || effort || !preserve) {  // Qwen3.5 emits an empty system block; Qwen3.8 drops it
            puts_(&o, "<|im_start|>system\n");
            if (effort) { puts_(&o, EFFORT); if (cn) puts_(&o, "\n\n"); }
            put(&o, c, cn);
            puts_(&o, "<|im_end|>\n");
        }
        first = 1;
    } else if (effort) {
        puts_(&o, "<|im_start|>system\n"); puts_(&o, EFFORT); puts_(&o, "<|im_end|>\n");
    }
    int last_query = -1;
    for (int i = nm - 1; i >= 0; i--) if (!strcmp(role[i], "user")) { last_query = i; break; }
    if (last_query < 0) die("no user message in ", hist);
    for (int i = first; i < nm; i++) {
        const char *c = con[i]; size_t cn = strlen(c); trim(&c, &cn);
        if (!strcmp(role[i], "user")) {
            puts_(&o, "<|im_start|>user\n"); put(&o, c, cn); puts_(&o, "<|im_end|>\n");
        } else if (!strcmp(role[i], "assistant")) {
            // reasoning = text before the first </think> (after its last <think>); content = after the last </think>
            const char *r = "", *th = memstr(c, cn, "</think>"); size_t rn = 0;
            if (th) {
                r = c; rn = (size_t)(th - c);
                while (rn && r[rn - 1] == '\n') rn--;
                const char *op = memrstr(r, rn, "<think>");
                if (op) { rn -= (size_t)(op + 7 - r); r = op + 7; }
                while (rn && *r == '\n') { r++; rn--; }
                trim(&r, &rn);
                const char *last = memrstr(c, cn, "</think>") + 8;
                cn -= (size_t)(last - c); c = last;
                while (cn && *c == '\n') { c++; cn--; }
                if (!split) trim(&c, &cn);
            }
            puts_(&o, "<|im_start|>assistant\n");
            if (preserve || i > last_query) { puts_(&o, "<think>\n"); put(&o, r, rn); puts_(&o, "\n</think>\n\n"); }
            put(&o, c, cn);
            puts_(&o, "<|im_end|>\n");
        } else die("unknown role in conversation: ", role[i]);
    }
    puts_(&o, think == 1 ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    if (text_only) { fwrite(o.b, 1, o.n, stdout); return 0; }
    Q35Tokenizer t; load(&t, dir);
    int *ids; int k = encode_chat(&t, &o, &ids);
    print_ids(ids, k);
    return 0;
}

// ---- engine runner -----------------------------------------------------------
static volatile sig_atomic_t interrupted = 0;
static void on_sig(int s) { (void)s; interrupted = 1; }
static double now(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

typedef struct { char b[1 << 16]; int n; } Line;

static int cmd_run(const char *dir, int argc, char **argv) {
    int ids_mode = 0, verbose = 0, nprompt = 0; const char *name = "model", *save = NULL, *logp = NULL;
    int i = 0;
    for (; i < argc && strcmp(argv[i], "--"); i++) {
        if (!strcmp(argv[i], "--ids")) ids_mode = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) name = argv[++i];
        else if (!strcmp(argv[i], "-P") && i + 1 < argc) nprompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) logp = argv[++i];
        else die("run: unknown option ", argv[i]);
    }
    if (i + 1 >= argc) die("run: no engine command after --", NULL);
    char **cmd = argv + i + 1;
    int spec = 0;  // LLB_SPEC=K in the engine command: step 0 also loads the NextN drafter, so its time is load, not prompt
    for (char **c = cmd; *c; c++) if (!strncmp(*c, "LLB_SPEC=", 9) && atoi(*c + 9) > 0) spec = 1;
    Q35Tokenizer t; load(&t, dir);
    int eot = -1;
    for (int k = 0; k < t.n_vocab; k++) { int l; const char *s = q35_tok_text(&t, k, &l); if (l == 13 && !memcmp(s, "<|endoftext|>", 13)) { eot = k; break; } }
    FILE *logf = logp ? fopen(logp, "w") : NULL;
    FILE *savef = save ? fopen(save, "ab") : NULL;
    if (save && !savef) die("cannot write ", save);
    if (savef) fwrite("assistant", 1, 10, savef);

    int po[2], pe[2];
    if (pipe(po) || pipe(pe)) die("pipe failed", NULL);
    struct sigaction sa = {0}; sa.sa_handler = on_sig; sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL); sigaction(SIGHUP, &sa, NULL);
    double t0 = now();
    pid_t pid = fork();
    if (pid == 0) {
        int dn = open("/dev/null", O_RDONLY); dup2(dn, 0);
        dup2(po[1], 1); dup2(pe[1], 2); close(po[0]); close(pe[0]);
        signal(SIGINT, SIG_DFL); signal(SIGTERM, SIG_DFL);
        execvp(cmd[0], cmd);
        fprintf(stderr, "llb: cannot run the engine %s: %s\n", cmd[0], strerror(errno));
        _exit(127);
    }
    close(po[1]); close(pe[1]);
    Line lo = {{0}, 0}, le = {{0}, 0};
    int open_o = 1, open_e = 1, ngen = 0, stopped = 0, nstep_pf = 0, nstep_dec = 0, out_nl = 1, loaded = 0;
    double ms_pf = 0, ms_dec = 0, t_load = 0, t_first = 0; long wall_ms = -1;
    char piece[65536];
    struct pollfd pf[2] = {{po[0], POLLIN, 0}, {pe[0], POLLIN, 0}};
    while ((open_o || open_e) && !interrupted) {
        pf[0].fd = open_o ? po[0] : -1; pf[1].fd = open_e ? pe[0] : -1;
        if (poll(pf, 2, -1) < 0) { if (errno == EINTR) continue; break; }
        for (int s = 0; s < 2; s++) {
            if (!(pf[s].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            Line *L = s ? &le : &lo;
            ssize_t r = read(pf[s].fd, L->b + L->n, sizeof L->b - 1 - L->n);
            if (r <= 0) { if (s) open_e = 0; else open_o = 0; if (L->n == 0) continue; r = 0; L->b[L->n++] = '\n'; }
            L->n += (int)r;
            char *p = L->b, *e;
            while ((e = memchr(p, '\n', L->b + L->n - p)) || (p == L->b && L->n == (int)sizeof L->b - 1)) {
                if (!e) e = L->b + L->n - 1;
                *e = 0;
                if (!s) {  // engine stdout: ids, then wall_ms
                    if (logf) fprintf(logf, "%s\n", p);
                    if (ids_mode) { printf("%s\n", p); fflush(stdout); }
                    if (!strncmp(p, "wall_ms ", 8)) wall_ms = atol(p + 8);
                    else if (*p >= '0' && *p <= '9' && !stopped) {
                        int id = atoi(p);
                        if (!t_first) t_first = now();
                        if (!ids_mode && (id == t.eos || id == eot)) {
                            stopped = 1; kill(pid, SIGTERM);
                        } else {
                            ngen++;
                            if (!ids_mode) {
                                int k = q35_tok_piece(&t, id, piece, sizeof piece);
                                fwrite(piece, 1, k, stdout); fflush(stdout);
                                if (k) out_nl = piece[k - 1] == '\n';
                                if (savef) fwrite(piece, 1, k, savef);
                            }
                        }
                    }
                } else {  // engine stderr: logits / step / batch lines and messages
                    if (logf) fprintf(logf, "%s\n", p);
                    int pos, cnt; double ms;
                    int is_step = sscanf(p, "step %d %lf", &pos, &ms) == 2;
                    int is_batch = !is_step && sscanf(p, "batch %d %d %lf", &pos, &cnt, &ms) == 3;
                    if (spec && is_step && pos == 0) ms = 0;
                    if ((is_step || is_batch) && !loaded) {
                        loaded = 1; t_load = now() - t0 - ms / 1000;
                        if (!verbose && !ids_mode && isatty(2)) fprintf(stderr, "%.1f s\n", t_load);
                        else if (!verbose) fprintf(stderr, "llb: loaded in %.1f s\n", t_load);
                    }
                    if (is_batch) { ms_pf += ms; nstep_pf += cnt; }
                    else if (is_step && (nprompt == 0 || pos < nprompt)) { ms_pf += ms; nstep_pf++; }
                    else if (is_step) { ms_dec += ms; nstep_dec++; }
                    if (verbose) fprintf(stderr, "%s\n", p);
                    else if (!strncmp(p, "bend: compiling the GPU program", 31))
                        fprintf(stderr, "llb: building the GPU kernel cache next to the engine (one time, about 70 s)\n");
                    else if (!is_step && !is_batch && strncmp(p, "logits ", 7) && strncmp(p, "gguf: ", 6) && strncmp(p, "spec: ", 6) && strncmp(p, "resident ", 9))
                        fprintf(stderr, "%s\n", p);
                }
                p = e + 1;
            }
            L->n -= (int)(p - L->b);
            memmove(L->b, p, L->n);
        }
    }
    if (interrupted || stopped) kill(pid, SIGTERM);
    int st = 0;
    for (int w = 0; w < 500 && waitpid(pid, &st, WNOHANG) == 0; w++) {   // up to 5 s, then SIGKILL
        struct timespec ts = {0, 10000000}; nanosleep(&ts, NULL);
        if (w == 499) { kill(pid, SIGKILL); waitpid(pid, &st, 0); }
    }
    if (savef) { fputc(0, savef); fclose(savef); }
    if (logf) fclose(logf);
    if (!ids_mode && !out_nl && isatty(1)) printf("\n");
    fflush(stdout);
    int code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    if (interrupted) { fprintf(stderr, "llb: interrupted\n"); return 130; }
    if (!stopped && code != 0) {
        fprintf(stderr, "llb: the engine failed (exit %d)%s%s\n", code, logp ? "; full log: " : "", logp ? logp : "");
        return code ? code : 1;
    }
    double pf_s = ms_pf / 1000, dec_s = ms_dec / 1000;
    (void)wall_ms;
    fprintf(stderr, "llb: %s | prompt %d tok, %.1f tok/s | generated %d tok, %.1f tok/s (%.1f ms/tok) | load %.1f s | first token %.1f s | stop: %s\n",
            name, nstep_pf, pf_s > 0 ? nstep_pf / pf_s : 0, ngen, dec_s > 0 ? nstep_dec / dec_s : 0,
            nstep_dec ? ms_dec / nstep_dec : 0, t_load, t_first ? t_first - t0 : 0, stopped ? "end of text" : "token limit");
    return 0;
}

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "";
    if (!strcmp(c, "info") && argc == 3) return cmd_info(argv[2]);
    if (!strcmp(c, "meta") && argc == 4) return cmd_meta(argv[2], argv[3]);
    if (!strcmp(c, "extract") && argc == 4) return cmd_extract(argv[2], argv[3]);
    if (!strcmp(c, "enc") && (argc == 3 || argc == 4)) return cmd_enc(argv[2], argc == 4 ? argv[3] : NULL);
    if (!strcmp(c, "dec") && argc >= 3) return cmd_dec(argv[2], argc - 3, argv + 3);
    if (!strcmp(c, "chat") && argc >= 4) {
        int text = 0, think = -1;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--text")) text = 1;
            else if (!strcmp(argv[i], "--think") && i + 1 < argc) think = atoi(argv[++i]);
        }
        return cmd_chat(argv[2], argv[3], text, think);
    }
    if (!strcmp(c, "run") && argc >= 3) return cmd_run(argv[2], argc - 3, argv + 3);
    fprintf(stderr, "usage: llb-tok info GGUF | extract GGUF DIR | enc DIR [TEXT] | dec DIR [ID...] | chat DIR HIST [--text] [--think N] | run DIR [opts] -- CMD...\n");
    return 2;
}
