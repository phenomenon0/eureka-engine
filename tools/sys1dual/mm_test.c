// mm_test.c: G5 unit tests for mm.c (no engine, no network except a loopback refusal).
// build: clang -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra mm_test.c mm.c ../clef3/json.c -o mm_test
// run:   mm_test FIXTURE_DIR   (fixtures: tests/mm/mkfix.py)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "mm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nt, nf;
#define OK(c, ...)                                                                                                     \
    do {                                                                                                               \
        nt++;                                                                                                          \
        if (c) printf("ok %d - ", nt);                                                                                 \
        else printf("NOT OK %d - ", nt), nf++;                                                                         \
        printf(__VA_ARGS__);                                                                                           \
        printf("\n");                                                                                                  \
    } while (0)

static const char *fixdir;

static unsigned char *slurp(const char *name, size_t *n) {
    char p[512];
    snprintf(p, sizeof p, "%s/%s", fixdir, name);
    FILE *f = fopen(p, "rb");
    if (!f) return fprintf(stderr, "missing fixture %s\n", p), exit(2), NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) exit(2);
    fclose(f);
    return b;
}

static char *b64(const unsigned char *b, size_t n) {
    static const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *o = malloc(n / 3 * 4 + 5), *p = o;
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        unsigned v = (unsigned)b[i] << 16 | (unsigned)b[i + 1] << 8 | b[i + 2];
        *p++ = A[v >> 18], *p++ = A[v >> 12 & 63], *p++ = A[v >> 6 & 63], *p++ = A[v & 63];
    }
    if (n - i == 1) {
        unsigned v = (unsigned)b[i] << 16;
        *p++ = A[v >> 18], *p++ = A[v >> 12 & 63], *p++ = '=', *p++ = '=';
    } else if (n - i == 2) {
        unsigned v = (unsigned)b[i] << 16 | (unsigned)b[i + 1] << 8;
        *p++ = A[v >> 18], *p++ = A[v >> 12 & 63], *p++ = A[v >> 6 & 63], *p++ = '=';
    }
    *p = 0;
    return o;
}

static char *data_uri(const char *mime, const char *fix) {
    size_t n;
    unsigned char *b = slurp(fix, &n);
    char *e = b64(b, n), *u;
    if (asprintf(&u, "data:%s;base64,%s", mime, e) < 0) exit(2);
    free(b), free(e);
    return u;
}

// render one content JSON; returns status (0 ok) and fills out/L
static int render(const char *content_json, int is_system, const mm_caps *c, sb *out, mm_list *L, mm_err *e) {
    const char *err = NULL;
    jv *v = json_parse(content_json, &err);
    if (!v) return fprintf(stderr, "bad test json: %s\n", err), exit(2), -1;
    memset(e, 0, sizeof *e);
    int rc = mm_render_content(v, 1, is_system, c, L, out, e);
    json_free(v);
    return rc ? e->status : 0;
}

static int status_of(const char *content_json, const mm_caps *c, mm_err *e) {
    sb o = {0};
    mm_list L = {0};
    int s = render(content_json, 0, c, &o, &L, e);
    free(o.p), mm_list_free(&L);
    return s;
}

static char *img_part(const char *uri) {
    char *j;
    if (asprintf(&j, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"%s\"}}", uri) < 0) exit(2);
    return j;
}

int main(int argc, char **argv) {
    if (argc < 2) return fprintf(stderr, "usage: mm_test FIXTURE_DIR\n"), 2;
    fixdir = argv[1];
    mm_caps c;
    mm_caps_default(&c);
    mm_err e;

    // -- base64: RFC 4648 §10 vectors + strictness
    const char *pl[] = {"", "f", "fo", "foo", "foob", "fooba", "foobar"};
    const char *en[] = {"", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy"};
    for (int i = 0; i < 7; i++) {
        unsigned char *o = NULL;
        size_t n = 0;
        int rc = mm_b64_decode(en[i], strlen(en[i]), &o, &n);
        OK(!rc && n == strlen(pl[i]) && !memcmp(o, pl[i], n), "b64 RFC4648 '%s'", en[i]);
        free(o);
    }
    const char *good[] = {"Zg", "Zm8", "Zm9v\nYmFy", "Zm9v YmE="};
    for (int i = 0; i < 4; i++) {
        unsigned char *o = NULL;
        size_t n = 0;
        OK(!mm_b64_decode(good[i], strlen(good[i]), &o, &n), "b64 accepts unpadded/wrapped '%s'", good[i]);
        free(o);
    }
    const char *bad[] = {"Z", "Zh==", "Zg=a", "Zm9v!", "Zm9v-_", "Zg===", "=Zg=", "Zm8==", "Zm9vZ"};
    for (int i = 0; i < 9; i++) {
        unsigned char *o = NULL;
        size_t n = 0;
        OK(mm_b64_decode(bad[i], strlen(bad[i]), &o, &n) != 0, "b64 rejects '%s'", bad[i]);
    }

    // -- sha256: FIPS 180-2 vectors (one block, two-block padding) + 1 MB
    char h[65];
    mm_sha256_hex((const unsigned char *)"", 0, h);
    OK(!strcmp(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "sha256('')");
    mm_sha256_hex((const unsigned char *)"abc", 3, h);
    OK(!strcmp(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "sha256('abc')");
    const char *s56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    mm_sha256_hex((const unsigned char *)s56, 56, h);
    OK(!strcmp(h, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "sha256(56 bytes)");
    unsigned char *mil = malloc(1000000);
    memset(mil, 'a', 1000000);
    mm_sha256_hex(mil, 1000000, h);
    OK(!strcmp(h, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "sha256(10^6 x 'a')");
    free(mil);

    // -- sniff: type + header dims on real encoder output
    struct { const char *f, *mime; unsigned w, h; } sn[] = {{"a.png", "image/png", 64, 48}, {"a.jpg", "image/jpeg", 64, 48},
        {"prog.jpg", "image/jpeg", 320, 240}, {"coco142472.jpg", "image/jpeg", 640, 480}, {"a.gif", "image/gif", 64, 48},
        {"a-lossy.webp", "image/webp", 64, 48}, {"a-lossless.webp", "image/webp", 64, 48},
        {"a-alpha.webp", "image/webp", 33, 17}, {"one.png", "image/png", 1, 1}, {"wide.png", "image/png", 9000, 8}};
    for (size_t i = 0; i < sizeof sn / sizeof *sn; i++) {
        size_t n;
        unsigned char *b = slurp(sn[i].f, &n);
        unsigned w, hh;
        const char *m = mm_sniff_image(b, n, &w, &hh);
        OK(m && !strcmp(m, sn[i].mime) && w == sn[i].w && hh == sn[i].h, "sniff %s -> %s %ux%u", sn[i].f,
            m ? m : "NULL", w, hh);
        free(b);
    }
    {
        size_t n;
        unsigned char *b = slurp("a.txt", &n);
        unsigned w, hh;
        OK(!mm_sniff_image(b, n, &w, &hh), "sniff a.txt -> not an image");
        free(b);
        b = slurp("a.wav", &n);
        OK(mm_sniff_audio(b, n) && !strcmp(mm_sniff_audio(b, n), "wav"), "sniff a.wav -> wav");
        free(b);
        b = slurp("a.mp3", &n);
        OK(mm_sniff_audio(b, n) && !strcmp(mm_sniff_audio(b, n), "mp3"), "sniff a.mp3 -> mp3");
        free(b);
    }

    // -- render: text-only stays byte-identical to the old text-part concatenation
    {
        sb o = {0};
        mm_list L = {0};
        int s = render("[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\" b\"}]", 0, &c, &o, &L, &e);
        OK(!s && !strcmp(o.p, "a b") && !L.n, "text parts concatenate verbatim, no media");
        free(o.p), mm_list_free(&L);
        o = (sb){0};
        s = render("\"plain\"", 0, &c, &o, &L, &e);
        OK(!s && !strcmp(o.p, "plain"), "string content verbatim");
        free(o.p), mm_list_free(&L);
        o = (sb){0};
        s = render("null", 0, &c, &o, &L, &e);
        OK(!s && !o.p, "null content renders nothing");
    }
    // -- render: one image (the COCO smoke image), position + manifest
    {
        char *u = data_uri("image/jpeg", "coco142472.jpg"), *p = img_part(u), *j;
        if (asprintf(&j, "[{\"type\":\"text\",\"text\":\"Look: \"},%s,{\"type\":\"text\",\"text\":\" Is there a car?\"}]", p) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        int s = render(j, 0, &c, &o, &L, &e);
        OK(!s && !strcmp(o.p, "Look: " MM_IMAGE_TAG " Is there a car?"), "1 image renders the template tag in place");
        OK(L.n == 1 && L.images == 1 && L.v[0].kind == MM_IMAGE && !strcmp(L.v[0].mime, "image/jpeg") && L.v[0].w == 640 &&
               L.v[0].h == 480 && L.v[0].n == 191947 && !strcmp(L.v[0].detail, "auto") && L.v[0].part == 1,
            "1 image manifest: image/jpeg 640x480 191947 B detail=auto part=1");
        OK(L.n == 1 && !strncmp(L.v[0].sha256, "db717381", 8), "1 image sha256 = file sha (db717381…) %s", L.n ? L.v[0].sha256 : "");
        free(o.p), mm_list_free(&L), free(u), free(p), free(j);
    }
    // -- render: two images, different formats, order kept, detail passed
    {
        char *u1 = data_uri("image/png", "a.png"), *u2 = data_uri("image/webp", "a-lossy.webp"), *j;
        if (asprintf(&j, "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"%s\",\"detail\":\"low\"}},{\"type\":\"text\",\"text\":\"vs\"},{\"type\":\"image_url\",\"image_url\":\"%s\"}]", u1, u2) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        int s = render(j, 0, &c, &o, &L, &e);
        OK(!s && !strcmp(o.p, MM_IMAGE_TAG "vs" MM_IMAGE_TAG), "2 images render in order around text");
        OK(L.n == 2 && !strcmp(L.v[0].mime, "image/png") && !strcmp(L.v[0].detail, "low") &&
               !strcmp(L.v[1].mime, "image/webp") && L.total == L.v[0].n + L.v[1].n,
            "2 images manifest: png(detail low) then webp (string image_url form), total bytes summed");
        free(o.p), mm_list_free(&L), free(u1), free(u2), free(j);
    }
    // -- the {"type":"image","image":...} form (HF/llama.cpp style)
    {
        char *u = data_uri("image/gif", "a.gif"), *j;
        if (asprintf(&j, "[{\"type\":\"image\",\"image\":\"%s\"}]", u) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        OK(!render(j, 0, &c, &o, &L, &e) && L.n == 1 && !strcmp(L.v[0].mime, "image/gif"), "type:image/image: form accepted");
        free(o.p), mm_list_free(&L), free(u), free(j);
    }
    // -- content-type handling: the bytes decide; declared must be image/*
    {
        char *u = data_uri("image/jpeg", "a.png"), *p = img_part(u), *j;
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        OK(!render(j, 0, &c, &o, &L, &e) && L.n == 1 && !strcmp(L.v[0].mime, "image/png") &&
               !strcmp(L.v[0].declared, "image/jpeg"),
            "mislabelled data:image/jpeg holding a png: accepted as image/png (declared kept)");
        free(o.p), mm_list_free(&L), free(u), free(p), free(j);
    }
    struct { const char *mime, *fix; int want; const char *what; } ct[] = {
        {"text/plain", "a.png", 415, "data:text/plain -> 415"},
        {"image/png", "a.txt", 415, "image/png label on text bytes -> 415"},
        {"image/png", "a.wav", 415, "image/png label on wav bytes -> 415"},
        {"image/png", "wide.png", 413, "9000x8 png over the 8192 side cap -> 413"},
    };
    for (size_t i = 0; i < sizeof ct / sizeof *ct; i++) {
        char *u = data_uri(ct[i].mime, ct[i].fix), *p = img_part(u), *j;
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        int s = status_of(j, &c, &e);
        OK(s == ct[i].want, "%s (got %d: %s)", ct[i].what, s, e.msg);
        free(u), free(p), free(j);
    }
    // -- size caps
    {
        char *u = data_uri("image/png", "big.bin"), *p = img_part(u), *j;
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        int s = status_of(j, &c, &e);
        OK(s == 413, "7 MiB image over the 6 MiB per-image cap -> 413 before decode (%s)", e.msg);
        mm_caps c2 = c;
        c2.image_bytes = 100;
        free(u), free(p), free(j);
        u = data_uri("image/png", "a.png"), p = img_part(u);
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        s = status_of(j, &c2, &e);
        OK(s == 413, "139 B png over a 100 B cap -> 413 (%s)", e.msg);
        c2 = c, c2.total_bytes = 200;
        free(j);
        if (asprintf(&j, "[%s,%s]", p, p) < 0) exit(2);
        s = status_of(j, &c2, &e);
        OK(s == 413 && !strcmp(e.param, "messages[1].content[1]"), "2 x 139 B over a 200 B total cap -> 413 at content[1] (%s)", e.param);
        c2 = c, c2.max_images = 2;
        free(j);
        if (asprintf(&j, "[%s,%s,%s]", p, p, p) < 0) exit(2);
        s = status_of(j, &c2, &e);
        OK(s == 400, "3 images over max_images 2 -> 400 (%s)", e.msg);
        c2 = c, c2.max_side = 63;
        free(j);
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        OK(status_of(j, &c2, &e) == 413, "64x48 over a 63 side cap -> 413");
        c2 = c, c2.max_pixels = 64 * 48 - 1;
        OK(status_of(j, &c2, &e) == 413, "64x48 over a 3071 pixel cap -> 413");
        free(u), free(p), free(j);
    }
    // -- malformed parts
    struct { const char *j; int want; const char *what; } mf[] = {
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBOR!!\"}}]", 400, "bad base64 -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png,abc\"}}]", 400, "non-base64 data: URI -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64\"}}]", 400, "data: URI without comma -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,\"}}]", 400, "empty data -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{}}]", 400, "image_url without url -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file:///etc/passwd\"}}]", 400, "file: URL -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"/etc/passwd\"}}]", 400, "bare path -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/a.png\"}}]", 400, "remote URL with remote off -> 400"},
        {"[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,iVBORw0KGgo=\",\"detail\":\"ultra\"}}]", 400, "detail=ultra -> 400"},
        {"[{\"type\":\"video\",\"video\":\"x\"}]", 400, "video -> 400"},
        {"[{\"type\":\"input_file\",\"file\":{}}]", 400, "unknown part type -> 400 (template: Unexpected item type)"},
        {"[{\"type\":\"text\",\"text\":5}]", 400, "non-string text -> 400"},
        {"[\"just a string\"]", 400, "non-object part -> 400"},
        {"{\"text\":\"x\"}", 400, "object content -> 400 (template: Unexpected content type)"},
        {"[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\",\"format\":\"aac\"}}]", 415, "audio format aac -> 415"},
        {"[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\"}}]", 400, "audio without format -> 400"},
    };
    for (size_t i = 0; i < sizeof mf / sizeof *mf; i++) {
        int s = status_of(mf[i].j, &c, &e);
        OK(s == mf[i].want, "%s (got %d: %s)", mf[i].what, s, e.msg);
    }
    // -- system messages cannot hold media (the template raises)
    {
        char *u = data_uri("image/png", "a.png"), *p = img_part(u), *j;
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        int s = render(j, 1, &c, &o, &L, &e);
        OK(s == 400 && strstr(e.msg, "System message cannot contain images."), "system image -> 400 (template message)");
        free(o.p), mm_list_free(&L), free(u), free(p), free(j);
    }
    // -- audio: parsed + sniffed; format must match the bytes; engine gate refuses without the ear
    {
        size_t n;
        unsigned char *b = slurp("a.wav", &n);
        char *e64 = b64(b, n), *j;
        if (asprintf(&j, "[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"%s\",\"format\":\"wav\"}},{\"type\":\"text\",\"text\":\"transcribe\"}]", e64) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        int s = render(j, 0, &c, &o, &L, &e);
        OK(!s && L.n == 1 && L.v[0].kind == MM_AUDIO && !strcmp(L.v[0].mime, "audio/wav") && L.v[0].n == n,
            "wav input_audio parsed (audio/wav, %zu B)", n);
        OK(mm_check_engine(&L, &c, &e) && e.status == 400 && strstr(e.msg, "ear"), "audio refused while the ear is not attached");
        mm_caps c2 = c;
        c2.audio = 1;
        OK(!mm_check_engine(&L, &c2, &e), "audio passes the engine gate when --mm audio");
        free(o.p), mm_list_free(&L), free(j);
        if (asprintf(&j, "[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"%s\",\"format\":\"mp3\"}}]", e64) < 0) exit(2);
        OK(status_of(j, &c, &e) == 415, "format mp3 on wav bytes -> 415 (%s)", e.msg);
        free(j), free(e64), free(b);
    }
    // -- engine gate for images
    {
        char *u = data_uri("image/png", "a.png"), *p = img_part(u), *j;
        if (asprintf(&j, "[%s]", p) < 0) exit(2);
        sb o = {0};
        mm_list L = {0};
        render(j, 0, &c, &o, &L, &e);
        OK(mm_check_engine(&L, &c, &e) && e.status == 400 && strstr(e.msg, "no vision path"),
            "image refused when the engine takes no images (default --mm none)");
        mm_caps c2 = c;
        c2.images = 1;
        OK(!mm_check_engine(&L, &c2, &e), "image passes the engine gate with --mm image");
        free(o.p), mm_list_free(&L), free(u), free(p), free(j);
    }
    // -- remote fetch: loopback refused unless private is allowed (no socket opened for the refusal)
    {
        mm_caps c2 = c;
        c2.allow_remote = 1;
        sb o = {0};
        mm_list L = {0};
        int s = render("[{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://127.0.0.1:9/a.png\"}}]", 0, &c2, &o, &L, &e);
        OK(!s && L.n == 1 && L.v[0].url && !L.v[0].bytes, "remote URL accepted for later fetch when remote is on");
        OK(mm_fetch_all(&L, &c2, &e) && e.status == 400 && strstr(e.msg, "private address"), "fetch of 127.0.0.1 refused (%s)", e.msg);
        free(o.p), mm_list_free(&L), o = (sb){0};
        s = render("[{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://user:pw@example.com/a.png\"}}]", 0, &c2, &o, &L, &e);
        OK(!s && mm_fetch_all(&L, &c2, &e) && strstr(e.msg, "credentials"), "URL credentials refused");
        free(o.p), mm_list_free(&L), o = (sb){0};
        s = render("[{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://[::1]:9/a.png\"}}]", 0, &c2, &o, &L, &e);
        OK(!s && mm_fetch_all(&L, &c2, &e) && strstr(e.msg, "private"), "fetch of [::1] refused");
        free(o.p), mm_list_free(&L);
    }
    // -- pad expansion (the engine's side of the contract)
    {
        sb o = {0};
        unsigned rows[2] = {3, 1};
        int rc = mm_expand_pads("a" MM_IMAGE_TAG "b" MM_IMAGE_TAG, rows, 2, &o, &e);
        OK(!rc && !strcmp(o.p, "a<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>b" MM_IMAGE_TAG),
            "expand_pads 3 + 1 rows");
        free(o.p), o = (sb){0};
        OK(mm_expand_pads("a" MM_IMAGE_TAG, rows, 2, &o, &e) == -1, "expand_pads count mismatch refused");
        free(o.p), o = (sb){0};
        unsigned z = 0;
        OK(mm_expand_pads(MM_IMAGE_TAG, &z, 1, &o, &e) == -1, "expand_pads zero rows refused");
        free(o.p);
    }
    // -- EARS-CANON: the ear's chat rule (transcript replaces the tag) and the transcription endpoint's audio object
    {
        sb o = {0};
        const char *tx[] = {"Nor is Mister Quilter's manner less interesting than matter."};
        int rc = mm_replace_audio("<|im_start|>user\n" MM_AUDIO_TAG " summarize<|im_end|>", tx, 1, &o, &e);
        OK(!rc && !strcmp(o.p, "<|im_start|>user\nNor is Mister Quilter's manner less interesting than matter. summarize<|im_end|>"),
            "replace_audio: the transcript is the user turn");
        free(o.p), o = (sb){0};
        OK(mm_replace_audio("a" MM_AUDIO_TAG MM_AUDIO_TAG, tx, 1, &o, &e) == -1 && e.status == 400, "replace_audio count mismatch refused");
        free(o.p), o = (sb){0};
        OK(mm_replace_audio("text only", tx, 0, &o, &e) == 0 && !strcmp(o.p, "text only"), "replace_audio: no tag, no audio = unchanged");
        free(o.p);
        size_t n;
        unsigned char *b = slurp("a.wav", &n);
        char *e64 = b64(b, n), *j;
        if (asprintf(&j, "{\"data\":\"%s\",\"format\":\"wav\"}", e64) < 0) exit(2);
        const char *err = NULL;
        jv *ia = json_parse(j, &err);
        mm_list L = {0};
        OK(ia && !mm_audio_item(ia, "input_audio", &c, &L, &e) && L.n == 1 && L.v[0].kind == MM_AUDIO &&
            !strcmp(L.v[0].mime, "audio/wav") && L.v[0].n == n && strlen(L.v[0].sha256) == 64, "audio_item: wav object staged like a chat part");
        mm_list_free(&L), json_free(ia), free(j);
        if (asprintf(&j, "{\"data\":\"%s\",\"format\":\"mp3\"}", e64) < 0) exit(2);
        ia = json_parse(j, &err);
        OK(mm_audio_item(ia, "input_audio", &c, &L, &e) == -1 && e.status == 415 && !strcmp(e.param, "input_audio"),
            "audio_item: format mp3 on wav bytes -> 415, param input_audio");
        mm_list_free(&L), json_free(ia), free(j);
        ia = json_parse("{\"data\":5}", &err);
        OK(mm_audio_item(ia, "input_audio", &c, &L, &e) == -1 && e.status == 400, "audio_item: missing format -> 400");
        mm_list_free(&L), json_free(ia);
        OK(mm_audio_item(NULL, "input_audio", &c, &L, &e) == -1 && e.status == 400, "audio_item: absent -> 400");
        mm_list_free(&L), free(e64), free(b);
    }
    printf("%d/%d passed\n", nt - nf, nt);
    return nf != 0;
}
