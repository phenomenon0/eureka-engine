// tools/qwen35_tokenizer.c — see qwen35_tokenizer.h. Faithful port of the
// llama.cpp BPE/qwen35 path. Host code only (compiled by gcc/nvcc/ObjC++).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "qwen35_unicode_data.inc"
#include "qwen35_tokenizer.h"

#define OUT_OF_RANGE 0xFFFFFFFFu
#define CP_REPLACEMENT 0xFFFDu

// ---- GPT-2 byte <-> unicode -------------------------------------------------
static int gByteToCp[256];
static int gInitialized = 0;

static void byte_maps_init(void) {
    if (gInitialized) return;
    int seen[256] = {0};
    for (int c = 0; c < 256; c++) {
        if ((c >= 0x21 && c <= 0x7E) || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF)) {
            gByteToCp[c] = c; seen[c] = 1;
        }
    }
    int n = 0;
    for (int c = 0; c < 256; c++) if (!seen[c]) gByteToCp[c] = 256 + n++;
    gInitialized = 1;
}

static inline int cp_to_byte(uint32_t cp) {
    for (int c = 0; c < 256; c++) if ((uint32_t)gByteToCp[c] == cp) return c;
    return -1;
}

// ---- utf8 -------------------------------------------------------------------
static inline int utf8_len(unsigned char b) {
    static const int lut[16] = {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4};
    return lut[b >> 4];
}

static int cp_to_utf8(uint32_t cp, char *out) {
    if (cp <= 0x7f) { out[0] = (char)cp; return 1; }
    if (cp <= 0x7ff) { out[0] = (char)(0xc0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3f)); return 2; }
    if (cp <= 0xffff) { out[0] = (char)(0xe0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3f)); out[2] = (char)(0x80 | (cp & 0x3f)); return 3; }
    out[0] = (char)(0xf0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); out[3] = (char)(0x80 | (cp & 0x3f)); return 4;
}

// llama.cpp unicode_cpts_from_utf8: invalid bytes -> U+FFFD, advance 1
static int cpts_from_utf8(const char *s, int n, uint32_t *out) {
    int off = 0, k = 0;
    while (off < n) {
        unsigned char b0 = (unsigned char)s[off];
        uint32_t cp; int len;
        if (!(b0 & 0x80)) { cp = b0; len = 1; }
        else if (!(b0 & 0x40)) { out[k++] = CP_REPLACEMENT; off++; continue; }
        else if (!(b0 & 0x20)) {
            if (off + 1 >= n || (((unsigned char)s[off+1]) & 0xc0) != 0x80) { out[k++] = CP_REPLACEMENT; off++; continue; }
            cp = ((b0 & 0x1f) << 6) | ((unsigned char)s[off+1] & 0x3f); len = 2;
        } else if (!(b0 & 0x10)) {
            if (off + 2 >= n || (((unsigned char)s[off+1]) & 0xc0) != 0x80 || (((unsigned char)s[off+2]) & 0xc0) != 0x80) { out[k++] = CP_REPLACEMENT; off++; continue; }
            cp = ((b0 & 0x0f) << 12) | (((unsigned char)s[off+1] & 0x3f) << 6) | ((unsigned char)s[off+2] & 0x3f); len = 3;
        } else if (!(b0 & 0x08)) {
            if (off + 3 >= n || (((unsigned char)s[off+1]) & 0xc0) != 0x80 || (((unsigned char)s[off+2]) & 0xc0) != 0x80 || (((unsigned char)s[off+3]) & 0xc0) != 0x80) { out[k++] = CP_REPLACEMENT; off++; continue; }
            cp = ((b0 & 0x07) << 18) | (((unsigned char)s[off+1] & 0x3f) << 12) | (((unsigned char)s[off+2] & 0x3f) << 6) | ((unsigned char)s[off+3] & 0x3f); len = 4;
        } else { out[k++] = CP_REPLACEMENT; off++; continue; }
        out[k++] = cp; off += len;
    }
    return k;
}

// ---- qwen35 pre-tokenizer regex split --------------------------------------
// port of llama.cpp unicode_regex_split_custom_qwen35 with a single offset
// covering the whole input; lengths are in codepoints.
static int split_qwen35(const uint32_t *cpts, int ncp, int *lens) {
    int nw = 0;
    size_t pos = 0;
    size_t prev_end = 0;

#define GET_CP(p)  ((p) < (size_t)ncp ? cpts[(p)] : OUT_OF_RANGE)
#define GET_FL(p)  ((p) < (size_t)ncp ? uni_flags_full(cpts[(p)]) : (uint16_t)0)
#define ADD_TOK(end) (__extension__ ({ \
        size_t _e = (size_t)(end); \
        int _len = (int)(_e - prev_end); \
        if (_len > 0) lens[nw++] = _len; \
        prev_end = _e; \
        _len; }))
#define FL_LETTER(f)  (((f) & 0x04) != 0)
#define FL_NUMBER(f)  (((f) & 0x02) != 0)
#define FL_MARK(f)    (((f) & 0x10) != 0)
#define FL_WS(f)      (((f) & 0x100) != 0)

    while (pos < (size_t)ncp) {
        uint32_t cpt = GET_CP(pos);
        uint16_t flags = GET_FL(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < (size_t)ncp) {
            uint32_t n1 = uni_tolower(GET_CP(pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { pos += ADD_TOK(pos + 2); continue; }
            if (pos + 2 < (size_t)ncp) {
                uint32_t n2 = uni_tolower(GET_CP(pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) { pos += ADD_TOK(pos + 3); continue; }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' || cpt == '\n' || FL_NUMBER(flags))) {
            uint16_t f1 = GET_FL(pos + 1);
            if (FL_LETTER(flags) || FL_MARK(flags) || FL_MARK(f1) || FL_LETTER(f1)) {
                pos++;
                while (FL_LETTER(GET_FL(pos)) || FL_MARK(GET_FL(pos))) pos++;
                ADD_TOK(pos);
                continue;
            }
        }

        // \p{N}
        if (FL_NUMBER(flags)) { pos++; ADD_TOK(pos); continue; }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        uint16_t flags2 = (cpt == ' ') ? GET_FL(pos + 1) : flags;
        if (!(FL_WS(flags2) || FL_LETTER(flags2) || FL_MARK(flags2) || FL_NUMBER(flags2)) && flags != 0) {
            pos += (cpt == ' ');
            while (!(FL_WS(flags2) || FL_LETTER(flags2) || FL_MARK(flags2) || FL_NUMBER(flags2)) && flags2 != 0) {
                pos++;
                flags2 = GET_FL(pos);
            }
            uint32_t cpt2 = GET_CP(pos);
            while (cpt2 == '\r' || cpt2 == '\n') { pos++; cpt2 = GET_CP(pos); }
            ADD_TOK(pos);
            continue;
        }

        size_t num_ws = 0, last_rn = 0;
        while (FL_WS(GET_FL(pos + num_ws))) {
            uint32_t cpt2 = GET_CP(pos + num_ws);
            if (cpt2 == '\r' || cpt2 == '\n') last_rn = pos + num_ws + 1;
            num_ws++;
        }

        // \s*[\r\n]+
        if (last_rn > 0) { pos = last_rn; ADD_TOK(pos); continue; }
        // \s+(?!\S)
        if (num_ws > 1 && GET_CP(pos + num_ws) != OUT_OF_RANGE) { pos += num_ws - 1; ADD_TOK(pos); continue; }
        // \s+
        if (num_ws > 0) { pos += num_ws; ADD_TOK(pos); continue; }

        pos++;
        ADD_TOK(pos);
    }
#undef GET_CP
#undef GET_FL
#undef ADD_TOK
#undef FL_LETTER
#undef FL_NUMBER
#undef FL_MARK
#undef FL_WS
    return nw;
}

// ---- hashes -----------------------------------------------------------------
static inline uint32_t fnv1a(const void *p, int n) {
    const unsigned char *s = (const unsigned char *)p;
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < n; i++) { h ^= s[i]; h *= 0x01000193u; }
    return h;
}

static int32_t tok_find(Q35Tokenizer *t, const char *s, int n) {
    uint32_t h = fnv1a(s, n) & t->tok_mask;
    for (;;) {
        int32_t id = t->text_to_id[h];
        if (id < 0) return -1;
        if (t->len[id] == n && memcmp(t->buf + t->off[id], s, n) == 0) return id;
        h = (h + 1) & t->tok_mask;
    }
}

static int bpe_rank(Q35Tokenizer *t, const char *l, int ln, const char *r, int rn) {
    uint32_t h = (fnv1a(l, ln) * 0x9E3779B1u ^ fnv1a(r, rn)) & t->hash_mask;
    for (;;) {
        int32_t rk = t->rank[h];
        if (rk < 0) return -1;
        if (t->ml_len[rk] == ln && t->mr_len[rk] == rn &&
            memcmp(t->mbuf + t->ml_off[rk], l, ln) == 0 &&
            memcmp(t->mbuf + t->mr_off[rk], r, rn) == 0) return rk;
        h = (h + 1) & t->hash_mask;
    }
}

// ---- load -------------------------------------------------------------------
static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }

int q35_tok_load(Q35Tokenizer *t, const char *dir) {
    memset(t, 0, sizeof(*t));
    byte_maps_init();
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return -1; }
    char magic[8];
    if (rd(f, magic, 8) || memcmp(magic, "Q35TOK01", 8) != 0) { fprintf(stderr, "tokenizer: bad magic\n"); fclose(f); return -1; }
    uint32_t n_vocab, n_merges, max_len, n_special;
    int32_t bos, eos, pad, add_bos;
    if (rd(f, &n_vocab, 4) || rd(f, &n_merges, 4) || rd(f, &max_len, 4) ||
        rd(f, &bos, 4) || rd(f, &eos, 4) || rd(f, &pad, 4) || rd(f, &add_bos, 4) ||
        rd(f, &n_special, 4)) { fclose(f); return -1; }
    t->n_vocab = n_vocab; t->n_merges = n_merges; t->max_token_len = max_len;
    t->bos = bos; t->eos = eos; t->pad = pad; t->add_bos = add_bos; t->mask_id = -1;

    t->special = (int32_t *)malloc(sizeof(int32_t) * (n_special ? n_special : 1));
    t->n_special = n_special;
    if (n_special && rd(f, t->special, 4 * n_special)) { fclose(f); return -1; }

    // token blobs
    size_t cap = 0;
    for (int i = 0; i < (int)n_vocab; i++) cap += max_len + 4;
    t->buf = (unsigned char *)malloc(cap);
    t->off = (uint32_t *)malloc(4 * n_vocab);
    t->len = (uint16_t *)malloc(2 * n_vocab);
    t->type = (uint8_t *)malloc(n_vocab);
    size_t boff = 0;
    for (int i = 0; i < (int)n_vocab; i++) {
        uint16_t l; uint8_t ty;
        if (rd(f, &l, 2) || rd(f, &ty, 1)) { fclose(f); return -1; }
        t->off[i] = (uint32_t)boff; t->len[i] = l; t->type[i] = ty;
        if (rd(f, t->buf + boff, l)) { fclose(f); return -1; }
        boff += l;
    }

    // merges
    size_t mcap = 0;
    for (int i = 0; i < (int)n_merges; i++) mcap += 2 * (max_len + 2);
    t->mbuf = (unsigned char *)malloc(mcap);
    t->ml_off = (uint32_t *)malloc(4 * n_merges);
    t->ml_len = (uint16_t *)malloc(2 * n_merges);
    t->mr_off = (uint32_t *)malloc(4 * n_merges);
    t->mr_len = (uint16_t *)malloc(2 * n_merges);
    size_t moff = 0;
    for (int i = 0; i < (int)n_merges; i++) {
        uint16_t ll, rl;
        if (rd(f, &ll, 2)) { fclose(f); return -1; }
        t->ml_off[i] = (uint32_t)moff; t->ml_len[i] = ll;
        if (rd(f, t->mbuf + moff, ll)) { fclose(f); return -1; }
        moff += ll;
        if (rd(f, &rl, 2)) { fclose(f); return -1; }
        t->mr_off[i] = (uint32_t)moff; t->mr_len[i] = rl;
        if (rd(f, t->mbuf + moff, rl)) { fclose(f); return -1; }
        moff += rl;
    }
    fclose(f);

    int bits = 1;
    while ((1u << bits) < n_vocab * 2u) bits++;
    t->tok_mask = (1u << bits) - 1;
    t->text_to_id = (int32_t *)malloc(4 * (t->tok_mask + 1));
    for (uint32_t i = 0; i <= t->tok_mask; i++) t->text_to_id[i] = -1;
    for (int i = 0; i < (int)n_vocab; i++) {
        uint32_t h = fnv1a(t->buf + t->off[i], t->len[i]) & t->tok_mask;
        while (t->text_to_id[h] >= 0) h = (h + 1) & t->tok_mask;
        t->text_to_id[h] = i;
    }

    int mbits = 1;
    while ((1u << mbits) < n_merges * 2u) mbits++;
    t->hash_mask = (1u << mbits) - 1;
    t->rank = (int32_t *)malloc(4 * (t->hash_mask + 1));
    for (uint32_t i = 0; i <= t->hash_mask; i++) t->rank[i] = -1;
    for (int i = 0; i < (int)n_merges; i++) {
        uint32_t h = (fnv1a(t->mbuf + t->ml_off[i], t->ml_len[i]) * 0x9E3779B1u ^
                      fnv1a(t->mbuf + t->mr_off[i], t->mr_len[i])) & t->hash_mask;
        while (t->rank[h] >= 0) h = (h + 1) & t->hash_mask;
        t->rank[h] = i;
    }
    return 0;
}

void q35_tok_free(Q35Tokenizer *t) {
    free(t->buf); free(t->off); free(t->len); free(t->type); free(t->text_to_id);
    free(t->mbuf); free(t->ml_off); free(t->ml_len); free(t->mr_off); free(t->mr_len); free(t->rank);
    free(t->special);
    memset(t, 0, sizeof(*t));
}

const char *q35_tok_text(Q35Tokenizer *t, int id, int *len) {
    if (id < 0 || id >= t->n_vocab) { if (len) *len = 0; return ""; }
    if (len) *len = t->len[id];
    return (const char *)t->buf + t->off[id];
}

// ---- BPE --------------------------------------------------------------------
typedef struct { const char *text; int n; int prev, next; } Sym;
typedef struct {
    int left, right, rank;
    const char *lp; int ln; const char *rp; int rn;
} Bigram;

typedef struct { Bigram *a; int n, cap; } BHeap;

static int bg_less(const Bigram *a, const Bigram *b) {
    return a->rank < b->rank || (a->rank == b->rank && a->left < b->left);
}
static void heap_push(BHeap *h, Bigram b) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap * 2 : 16; h->a = (Bigram *)realloc(h->a, sizeof(Bigram) * h->cap); }
    int i = h->n++;
    h->a[i] = b;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (bg_less(&h->a[i], &h->a[p])) { Bigram tmp = h->a[i]; h->a[i] = h->a[p]; h->a[p] = tmp; i = p; } else break;
    }
}
static Bigram heap_pop(BHeap *h) {
    Bigram top = h->a[0];
    h->a[0] = h->a[--h->n];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < h->n && bg_less(&h->a[l], &h->a[s])) s = l;
        if (r < h->n && bg_less(&h->a[r], &h->a[s])) s = r;
        if (s == i) break;
        Bigram tmp = h->a[i]; h->a[i] = h->a[s]; h->a[s] = tmp; i = s;
    }
    return top;
}

// dynamic id vector
typedef struct { int *a; int n, cap; } IVec;
static void iv_push(IVec *v, int x) {
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 64; v->a = (int *)realloc(v->a, sizeof(int) * v->cap); }
    v->a[v->n++] = x;
}

// BPE one already-byte-encoded word (valid UTF-8, one char == one raw byte)
static void bpe_word(Q35Tokenizer *t, const char *w, int wn, IVec *out) {
    int ns = 0, cap = wn / 1 + 2;
    Sym *sy = (Sym *)malloc(sizeof(Sym) * cap);
    int off = 0;
    while (off < wn) {
        int cl = utf8_len((unsigned char)w[off]);
        if (off + cl > wn) cl = wn - off;
        sy[ns].text = w + off; sy[ns].n = cl;
        sy[ns].prev = ns - 1; sy[ns].next = (off + cl == wn) ? -1 : ns + 1;
        off += cl; ns++;
    }
    BHeap heap = {0};
    for (int i = 1; i < ns; i++) {
        int l = i - 1, r = i;
        int rk = bpe_rank(t, sy[l].text, sy[l].n, sy[r].text, sy[r].n);
        if (rk >= 0) { Bigram b = {l, r, rk, sy[l].text, sy[l].n, sy[r].text, sy[r].n}; heap_push(&heap, b); }
    }
    while (heap.n) {
        Bigram b = heap_pop(&heap);
        Sym *L = &sy[b.left], *R = &sy[b.right];
        if (L->n == 0 || R->n == 0) continue;
        if (L->n != b.ln || memcmp(L->text, b.lp, b.ln) != 0) continue;
        if (R->n != b.rn || memcmp(R->text, b.rp, b.rn) != 0) continue;
        L->n += R->n; R->n = 0;
        L->next = R->next;
        if (R->next >= 0) sy[R->next].prev = b.left;
        // add (L->prev, L) and (L, L->next)
        if (L->prev >= 0) {
            Sym *A = &sy[L->prev];
            int rk = bpe_rank(t, A->text, A->n, L->text, L->n);
            if (rk >= 0) { Bigram nb = {L->prev, b.left, rk, A->text, A->n, L->text, L->n}; heap_push(&heap, nb); }
        }
        if (L->next >= 0) {
            Sym *C = &sy[L->next];
            int rk = bpe_rank(t, L->text, L->n, C->text, C->n);
            if (rk >= 0) { Bigram nb = {b.left, L->next, rk, L->text, L->n, C->text, C->n}; heap_push(&heap, nb); }
        }
    }
    for (int i = 0; i < ns; i++) {
        if (sy[i].n == 0) continue;
        int id = tok_find(t, sy[i].text, sy[i].n);
        if (id >= 0) { iv_push(out, id); continue; }
        for (int j = 0; j < sy[i].n; j++) {
            int bid = tok_find(t, sy[i].text + j, 1);
            if (bid >= 0) iv_push(out, bid);
        }
    }
    free(heap.a);
    free(sy);
}

// split raw fragment into one or more words then BPE each
static void bpe_fragment(Q35Tokenizer *t, const char *s, int n, IVec *out) {
    if (n <= 0) return;
    uint32_t *cpts = (uint32_t *)malloc(sizeof(uint32_t) * (n + 1));
    int ncp = cpts_from_utf8(s, n, cpts);
    int *lens = (int *)malloc(sizeof(int) * (ncp + 1));
    int nw = split_qwen35(cpts, ncp, lens);

    // raw byte encoding of a cpt range, then GPT-2 byte encode
    char *raw = (char *)malloc(n * 4 + 8);
    int cp0 = 0;
    for (int wi = 0; wi < nw; wi++) {
        int cnt = lens[wi];
        int rn = 0;
        for (int i = 0; i < cnt; i++) rn += cp_to_utf8(cpts[cp0 + i], raw + rn);
        cp0 += cnt;
        // GPT-2 byte-encode
        char *enc = (char *)malloc(rn * 2 + 8);
        int en = 0;
        for (int i = 0; i < rn; i++) en += cp_to_utf8((uint32_t)gByteToCp[(unsigned char)raw[i]], enc + en);
        bpe_word(t, enc, en, out);
        free(enc);
    }
    free(raw); free(lens); free(cpts);
}

// ---- special-token partition ------------------------------------------------
typedef struct { int tok; const char *base; int off, len; } Frag;
typedef struct { Frag *a; int n, cap; } FVec;
static void fv_push(FVec *v, Frag f) {
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 16; v->a = (Frag *)realloc(v->a, sizeof(Frag) * v->cap); }
    v->a[v->n++] = f;
}
static Frag mkfrag(int tok, const char *base, int off, int len) { Frag f; f.tok = tok; f.base = base; f.off = off; f.len = len; return f; }

static void partition(Q35Tokenizer *t, const char *text, int parse_special, FVec *out) {
    fv_push(out, mkfrag(-1, text, 0, (int)strlen(text)));
    for (int si = 0; si < t->n_special; si++) {
        int sid = t->special[si];
        int slen; const char *sp = q35_tok_text(t, sid, &slen);
        if (slen == 0) continue;
        if (!parse_special && (t->type[sid] == 3 || t->type[sid] == 2)) continue; // control/unknown
        FVec nxt = {0};
        for (int fi = 0; fi < out->n; fi++) {
            Frag fr = out->a[fi];
            if (fr.tok >= 0) { fv_push(&nxt, fr); continue; }
            int start = fr.off, end = fr.off + fr.len;
            int scan = start;
            while (scan <= end - slen) {
                int m = -1;
                for (int i = scan; i + slen <= end; i++) {
                    if (memcmp(fr.base + i, sp, slen) == 0) { m = i; break; }
                }
                if (m < 0) break;
                if (m > start) fv_push(&nxt, mkfrag(-1, fr.base, start, m - start));
                fv_push(&nxt, mkfrag(sid, NULL, 0, slen));
                start = m + slen;
                scan = start;
            }
            if (start < end) fv_push(&nxt, mkfrag(-1, fr.base, start, end - start));
        }
        free(out->a);
        *out = nxt;
    }
}

// ---- public encode/decode ---------------------------------------------------
int q35_tok_encode(Q35Tokenizer *t, const char *text, int parse_special, int *out, int out_cap) {
    IVec ids = {0};
    FVec frags = {0};
    partition(t, text, parse_special, &frags);
    for (int i = 0; i < frags.n; i++) {
        if (frags.a[i].tok >= 0) { iv_push(&ids, frags.a[i].tok); continue; }
        int n = frags.a[i].len;
        if (n == 0) continue;
        char *tmp = (char *)malloc(n + 1);
        memcpy(tmp, frags.a[i].base + frags.a[i].off, n); tmp[n] = 0;
        bpe_fragment(t, tmp, n, &ids);
        free(tmp);
    }
    free(frags.a);
    int rc = ids.n;
    if (rc <= out_cap) memcpy(out, ids.a, sizeof(int) * rc);
    free(ids.a);
    return rc;
}

int q35_tok_piece(Q35Tokenizer *t, int id, char *out, int out_cap) {
    int len; const char *s = q35_tok_text(t, id, &len);
    if (id < 0 || id >= t->n_vocab) return 0;
    uint8_t ty = t->type[id];
    if (ty == 3 || ty == 2 || ty == 4) { // control / unknown / user_defined -> raw text
        int n = len < out_cap ? len : out_cap;
        memcpy(out, s, n);
        return n;
    }
    if (ty != 1) return 0; // unused/byte -> nothing (llama behavior)
    // GPT-2 byte-decode the (encoded) token text
    uint32_t *cps = (uint32_t *)malloc(sizeof(uint32_t) * (len + 1));
    int ncp = cpts_from_utf8(s, len, cps);
    int n = 0;
    for (int i = 0; i < ncp; i++) {
        int b = cp_to_byte(cps[i]);
        if (b < 0) continue;
        if (n < out_cap) out[n] = (char)b;
        n++;
    }
    free(cps);
    return n;
}

int q35_tok_decode(Q35Tokenizer *t, const int *ids, int n, char *out, int out_cap) {
    int total = 0;
    char tmp[65536];
    for (int i = 0; i < n; i++) {
        int got = q35_tok_piece(t, ids[i], tmp, sizeof tmp);
        for (int j = 0; j < got; j++) { if (total < out_cap) out[total] = tmp[j]; total++; }
    }
    return total;
}
