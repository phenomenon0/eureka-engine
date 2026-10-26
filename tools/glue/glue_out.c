// glue_out.c: see glue_out.h. The GGUF walk is clef_head_open's (tools/clef3/weights.c)
// reduced to the one tensor; the dequant is the same GGML block layouts.
#define _POSIX_C_SOURCE 200809L
#include "glue_out.h"
#include "../clef3/head.h"
#include "../../host/llb_gguf.c"
#include <math.h>

static int out_fd = -1, out_type, out_rows;
static uint64_t out_off, out_rb;

static float half(const unsigned char *p) {
    _Float16 h;
    memcpy(&h, p, 2);
    return (float)h;
}

void glue_out_open(const char *path) {
    out_fd = open(path, O_RDONLY);
    struct stat st;
    if (out_fd < 0 || fstat(out_fd, &st)) clef_fail("glue: cannot open GGUF");
    GgRd r = {out_fd, 0, 0, malloc(1 << 20), 0, 0};
    char magic[4];
    gg_rd_get(&r, magic, 4);
    uint32_t ver = gg_u32(&r);
    uint64_t nt = gg_u64(&r), nkv = gg_u64(&r), align = 32;
    int found = 0;
    if (memcmp(magic, "GGUF", 4) || ver != 3 || nt > 100000 || nkv > 100000) clef_fail("glue: invalid GGUF header");
    for (uint64_t i = 0; i < nkv && !r.bad; i++) {
        char *key = gg_str(&r, 1);
        uint32_t type = gg_u32(&r);
        uint64_t v = gg_val(&r, type);
        if (key && !strcmp(key, "general.alignment")) align = v;
        free(key);
    }
    for (uint64_t i = 0; i < nt && !r.bad; i++) {
        char *name = gg_str(&r, 1);
        uint32_t nd = gg_u32(&r);
        uint64_t ne[4] = {1, 1, 1, 1};
        if (!name || nd < 1 || nd > 4) clef_fail("glue: invalid tensor header");
        for (unsigned j = 0; j < nd; j++) ne[j] = gg_u64(&r);
        uint32_t type = gg_u32(&r);
        uint64_t off = gg_u64(&r);
        if (!strcmp(name, "output.weight")) {
            if (nd != 2 || ne[0] != 4096 || (type != 0 && type != 8 && type != 14)) clef_fail("glue: output.weight wants 4096 cols, F32/Q8_0/Q6_K");
            out_type = (int)type, out_rows = (int)ne[1], out_off = off, found = 1;
            out_rb = type == 0 ? 4096 * 4 : type == 8 ? 4096 / 32 * 34 : 4096 / 256 * 210;
        }
        free(name);
    }
    if (r.bad || !found || !align || align > 4096) clef_fail("glue: output.weight missing / corrupt GGUF");
    out_off += (r.off + r.p + align - 1) / align * align;
    free(r.buf);
    if (out_off + (uint64_t)out_rows * out_rb > (uint64_t)st.st_size) clef_fail("glue: output.weight outside GGUF");
}

void glue_out_row(int id, float *w) {
    static unsigned char raw[4096 * 4];
    if (out_fd < 0 || id < 0 || id >= out_rows || gg_pread(out_fd, raw, out_rb, out_off + (uint64_t)id * out_rb))
        clef_fail("glue: output row read failed");
    const unsigned char *b = raw;
    if (out_type == 0) memcpy(w, b, 4096 * 4);
    else if (out_type == 8) {
        for (int i = 0; i < 4096; i += 32, b += 34) {
            float d = half(b);
            for (int j = 0; j < 32; j++) w[i + j] = d * (int8_t)b[2 + j];
        }
    } else {
        for (int i = 0; i < 4096; i += 256, b += 210) {
            float d = half(b + 208);
            for (int j = 0; j < 256; j++) {
                int group = j / 128, lane = j % 32, quarter = (j % 128) / 32;
                unsigned low = (b[group * 64 + lane + (quarter % 2) * 32] >> (quarter >= 2 ? 4 : 0)) & 15;
                unsigned high = (b[128 + group * 32 + lane] >> (quarter * 2)) & 3;
                w[i + j] = (d * (int8_t)b[192 + group * 8 + quarter * 2 + lane / 16]) * ((int)(low | (high << 4)) - 32);
            }
        }
    }
    for (int i = 0; i < 4096; i++) if (!isfinite(w[i])) clef_fail("glue: nonfinite output row");
}

// ---- G24 F2/F4: the trunk binding ------------------------------------------------
// tensor_table_sha256 = sha256 over "name:sha256(tensor bytes)\n" for every tensor,
// names sorted (relayout_gguf.py's definition; glue-attach2 Base = 58357be7…). It binds
// content, not a label: a relabeled GGUF with other weights gets another sha.

typedef struct { const GgFile *g; size_t *ord; size_t next, n; pthread_mutex_t mu; char (*hex)[65]; int bad; } TabJob;

static void *tab_worker(void *arg) {
    TabJob *J = arg;
    uint8_t *buf = malloc(GG_UNIT);
    for (;;) {
        pthread_mutex_lock(&J->mu);
        size_t i = J->next++;
        pthread_mutex_unlock(&J->mu);
        if (i >= J->n || !buf) break;
        const GgT *t = &J->g->t[i];
        GgSha s;
        gg_sha_init(&s);
        for (uint64_t at = 0; at < t->bytes;) {
            size_t k = t->bytes - at < GG_UNIT ? (size_t)(t->bytes - at) : GG_UNIT;
            if (gg_pread(J->g->fd, buf, k, t->src + at)) { J->bad = 1; break; }
            gg_sha_update(&s, buf, k), at += k;
        }
        gg_sha_hex(&s, J->hex[i]);
    }
    free(buf);
    return NULL;
}

static const GgFile *tab_g;
static int tab_cmp(const void *a, const void *b) {
    return strcmp(tab_g->t[*(const size_t *)a].name, tab_g->t[*(const size_t *)b].name);
}

int glue_table_sha(const char *path, char out[65]) {
    GgFile g;
    char err[256];
    if (gg_open(&g, path, err)) return fprintf(stderr, "glue: %s\n", err), -1;
    TabJob J = {&g, malloc(g.nt * sizeof(size_t)), 0, g.nt, PTHREAD_MUTEX_INITIALIZER, malloc(g.nt * 65), 0};
    if (!J.ord || !J.hex) return -1;
    int nth = gg_threads();
    pthread_t th[64];
    if (nth > 64) nth = 64;
    for (int i = 0; i < nth; i++) pthread_create(&th[i], NULL, tab_worker, &J);
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    for (size_t i = 0; i < g.nt; i++) J.ord[i] = i;
    tab_g = &g;
    qsort(J.ord, g.nt, sizeof(size_t), tab_cmp);
    GgSha s;
    gg_sha_init(&s);
    for (size_t i = 0; i < g.nt; i++) {
        const char *nm = g.t[J.ord[i]].name;
        gg_sha_update(&s, (const uint8_t *)nm, strlen(nm));
        gg_sha_update(&s, (const uint8_t *)":", 1);
        gg_sha_update(&s, (const uint8_t *)J.hex[J.ord[i]], 64);
        gg_sha_update(&s, (const uint8_t *)"\n", 1);
    }
    gg_sha_hex(&s, out);
    close(g.fd);
    free(J.ord), free(J.hex);
    return J.bad ? -1 : 0;
}

int glue_file_sha(const char *path, char out[65]) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    uint8_t buf[1 << 16];
    GgSha s;
    gg_sha_init(&s);
    for (ssize_t k; (k = read(fd, buf, sizeof buf)) != 0;) {
        if (k < 0) { if (errno == EINTR) continue; close(fd); return -1; }
        gg_sha_update(&s, buf, (size_t)k);
    }
    close(fd);
    gg_sha_hex(&s, out);
    return 0;
}

// The table sha reads every tensor once (342 s for the 9B Q8_0 on the FUSE store), so it is
// cached under DIR/tabsha/<dev>-<ino>-<size>-<mtime_ns>, a line "v1 HEADER_SHA TABLE_SHA".
// A hit also re-hashes the header (KV + tensor infos, bytes [0, first tensor)): a relabel
// that kept size and mtime still misses. glue_rows.py reads and writes the same records.
static int head_sha(const char *path, char out[65]) {
    GgFile g;
    char err[256];
    if (gg_open(&g, path, err)) return fprintf(stderr, "glue: %s\n", err), -1;
    uint64_t lo = UINT64_MAX;
    for (size_t i = 0; i < g.nt; i++) if (g.t[i].src < lo) lo = g.t[i].src;
    uint8_t *buf = malloc(1 << 20);
    GgSha s;
    gg_sha_init(&s);
    int bad = !buf || lo == UINT64_MAX;
    for (uint64_t at = 0; !bad && at < lo;) {
        size_t k = lo - at < (1u << 20) ? (size_t)(lo - at) : (1u << 20);
        if (gg_pread(g.fd, buf, k, at)) bad = 1;
        else gg_sha_update(&s, buf, k), at += k;
    }
    gg_sha_hex(&s, out);
    close(g.fd);
    free(buf);
    return bad ? -1 : 0;
}

int glue_table_sha_cached(const char *path, const char *dir, char out[65], int *hit) {
    struct stat st;
    char key[4096], hs[65], line[256], h2[65], t2[65];
    *hit = 0;
    if (stat(path, &st) || head_sha(path, hs)) return -1;
    int kn = dir && *dir ? snprintf(key, sizeof key, "%s/tabsha/%llu-%llu-%llu-%lld%09ld", dir, (unsigned long long)st.st_dev,
        (unsigned long long)st.st_ino, (unsigned long long)st.st_size, (long long)st.st_mtim.tv_sec, st.st_mtim.tv_nsec) : -1;
    FILE *f = kn > 0 && kn < (int)sizeof key ? fopen(key, "r") : NULL;
    if (f) {
        if (fgets(line, sizeof line, f) && sscanf(line, "v1 %64s %64s", h2, t2) == 2 && !strcmp(h2, hs))
            return fclose(f), memcpy(out, t2, 65), *hit = 1, 0;
        fclose(f);
    }
    if (glue_table_sha(path, out)) return -1;
    if (kn > 0 && kn < (int)sizeof key) {
        char d[4200], tmp[4300];
        snprintf(d, sizeof d, "%s/tabsha", dir);
        mkdir(dir, 0755), mkdir(d, 0755);
        snprintf(tmp, sizeof tmp, "%s.%d", key, (int)getpid());
        if ((f = fopen(tmp, "w"))) fprintf(f, "v1 %s %s\n", hs, out), fclose(f), rename(tmp, key);
    }
    return 0;
}
