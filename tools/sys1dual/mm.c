// mm.c: G5 — multimodal message content (see mm.h and docs/reference/multimodal-content.md).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "mm.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

void mm_caps_default(mm_caps *c) {
    *c = (mm_caps){
        .image_bytes = 6u << 20, // the 8 MiB request body holds one 6 MiB image as base64
        .audio_bytes = 6u << 20,
        .total_bytes = 24u << 20, // reachable with URL images
        .max_images = 8,
        .max_audio = 1,
        .max_side = 8192,
        .max_pixels = 40LL * 1000 * 1000,
        .fetch_timeout_s = 10,
    };
}

void mm_list_free(mm_list *L) {
    for (size_t i = 0; i < L->n; i++) free(L->v[i].bytes), free(L->v[i].url);
    free(L->v);
    *L = (mm_list){0};
}

static int fail(mm_err *e, int status, const char *param, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    e->status = status;
    snprintf(e->param, sizeof e->param, "%s", param ? param : "");
    vsnprintf(e->msg, sizeof e->msg, fmt, a);
    va_end(a);
    return -1;
}

// ---- base64 (RFC 4648 §4, standard alphabet; '=' padding optional; ASCII whitespace skipped) ----

static int b64v(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

int mm_b64_decode(const char *s, size_t len, unsigned char **out, size_t *n) {
    unsigned char *o = malloc(len / 4 * 3 + 3);
    if (!o) return -1;
    size_t k = 0, q = 0, pad = 0;
    uint32_t acc = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') continue;
        if (ch == '=') {
            pad++;
            continue;
        }
        int v = b64v(ch);
        if (v < 0 || pad) return free(o), -1; // bad char, or data after padding
        acc = acc << 6 | (uint32_t)v;
        if (++q == 4) {
            o[k++] = (unsigned char)(acc >> 16), o[k++] = (unsigned char)(acc >> 8), o[k++] = (unsigned char)acc;
            q = 0, acc = 0;
        }
    }
    // q = data chars in the last quantum: 0 (pad must be 0), 2 (pad 0|2), 3 (pad 0|1); 1 is never valid
    if (q == 1 || (q == 0 && pad) || (q == 2 && pad && pad != 2) || (q == 3 && pad && pad != 1)) return free(o), -1;
    if (q == 2) {
        if (acc & 0xf) return free(o), -1; // non-canonical trailing bits
        o[k++] = (unsigned char)(acc >> 4);
    } else if (q == 3) {
        if (acc & 0x3) return free(o), -1;
        o[k++] = (unsigned char)(acc >> 10), o[k++] = (unsigned char)(acc >> 2);
    }
    *out = o, *n = k;
    return 0;
}

// ---- sniffing: the bytes decide the type; header dimensions for the caps ----

static unsigned be16(const unsigned char *p) { return (unsigned)p[0] << 8 | p[1]; }
static unsigned le16(const unsigned char *p) { return (unsigned)p[1] << 8 | p[0]; }
static unsigned be32(const unsigned char *p) { return (unsigned)p[0] << 24 | (unsigned)p[1] << 16 | (unsigned)p[2] << 8 | p[3]; }
static unsigned le24(const unsigned char *p) { return (unsigned)p[2] << 16 | (unsigned)p[1] << 8 | p[0]; }

const char *mm_sniff_image(const unsigned char *b, size_t n, unsigned *w, unsigned *h) {
    *w = *h = 0;
    if (n >= 24 && !memcmp(b, "\x89PNG\r\n\x1a\n", 8)) {
        if (!memcmp(b + 12, "IHDR", 4)) *w = be32(b + 16), *h = be32(b + 20);
        return "image/png";
    }
    if (n >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff) {
        // walk the marker segments to the first SOFn (C0-CF except C4 DHT, C8 JPG, CC DAC)
        size_t i = 2;
        while (i + 9 < n) {
            if (b[i] != 0xff) break;
            unsigned m = b[i + 1];
            if (m == 0xff) {
                i++;
                continue;
            }
            if (m == 0xd8 || m == 0x01 || (m >= 0xd0 && m <= 0xd7)) {
                i += 2;
                continue;
            }
            unsigned L = be16(b + i + 2);
            if (L < 2) break;
            if (m >= 0xc0 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc) {
                *h = be16(b + i + 5), *w = be16(b + i + 7);
                break;
            }
            if (m == 0xd9 || m == 0xda) break; // EOI / SOS before any SOF
            i += 2 + L;
        }
        return "image/jpeg";
    }
    if (n >= 10 && (!memcmp(b, "GIF87a", 6) || !memcmp(b, "GIF89a", 6))) {
        *w = le16(b + 6), *h = le16(b + 8);
        return "image/gif";
    }
    if (n >= 30 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WEBP", 4)) {
        if (!memcmp(b + 12, "VP8 ", 4) && b[23] == 0x9d && b[24] == 0x01 && b[25] == 0x2a)
            *w = le16(b + 26) & 0x3fff, *h = le16(b + 28) & 0x3fff;
        else if (!memcmp(b + 12, "VP8L", 4) && b[20] == 0x2f) {
            uint32_t v = (uint32_t)b[21] | (uint32_t)b[22] << 8 | (uint32_t)b[23] << 16 | (uint32_t)b[24] << 24;
            *w = (v & 0x3fff) + 1, *h = (v >> 14 & 0x3fff) + 1;
        } else if (!memcmp(b + 12, "VP8X", 4))
            *w = le24(b + 24) + 1, *h = le24(b + 27) + 1;
        return "image/webp";
    }
    return NULL;
}

const char *mm_sniff_audio(const unsigned char *b, size_t n) {
    if (n >= 12 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WAVE", 4)) return "wav";
    if (n >= 4 && !memcmp(b, "fLaC", 4)) return "flac";
    if (n >= 4 && !memcmp(b, "OggS", 4)) return "ogg";
    if (n >= 3 && !memcmp(b, "ID3", 3)) return "mp3";
    if (n >= 2 && b[0] == 0xff && (b[1] & 0xe0) == 0xe0 && (b[1] & 0x06)) return "mp3"; // MPEG frame sync, layer != 0
    return NULL;
}

// ---- SHA-256 (FIPS 180-4) ----

static const uint32_t K256[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138,
    0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70,
    0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa,
    0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define ROR(x, r) ((x) >> (r) | (x) << (32 - (r)))

static void sha_block(uint32_t H[8], const unsigned char *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ w[i - 15] >> 3;
        uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ w[i - 2] >> 10;
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    H[0] += a, H[1] += b, H[2] += c, H[3] += d, H[4] += e, H[5] += f, H[6] += g, H[7] += h;
}

void mm_sha256_hex(const unsigned char *b, size_t n, char out[65]) {
    uint32_t H[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) sha_block(H, b + i);
    unsigned char t[128] = {0};
    size_t r = n - i;
    memcpy(t, b + i, r);
    t[r] = 0x80;
    size_t tl = r + 9 <= 64 ? 64 : 128;
    uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) t[tl - 1 - k] = (unsigned char)(bits >> (8 * k));
    sha_block(H, t);
    if (tl == 128) sha_block(H, t + 64);
    for (int k = 0; k < 8; k++) snprintf(out + 8 * k, 9, "%08x", H[k]);
}

// ---- items ----

static mm_item *push(mm_list *L) {
    if (L->n == L->cap) {
        size_t nc = L->cap ? L->cap * 2 : 4;
        mm_item *v = realloc(L->v, nc * sizeof *v);
        if (!v) return NULL;
        L->v = v, L->cap = nc;
    }
    mm_item *it = &L->v[L->n++];
    memset(it, 0, sizeof *it);
    return it;
}

// sniff + header caps + running total, once the bytes are in hand
static int finish_image(mm_item *it, mm_list *L, const mm_caps *c, const char *pp, mm_err *e) {
    const char *m = mm_sniff_image(it->bytes, it->n, &it->w, &it->h);
    if (!m) return fail(e, 415, pp, "unsupported image format (the bytes are not png, jpeg, webp or gif)");
    snprintf(it->mime, sizeof it->mime, "%s", m);
    if (!it->w || !it->h) return fail(e, 400, pp, "corrupt %s: no image dimensions in the header", m);
    if (it->w > c->max_side || it->h > c->max_side || (long long)it->w * it->h > c->max_pixels)
        return fail(e, 413, pp, "image is %ux%u; the limit is %u per side and %lld pixels", it->w, it->h, c->max_side,
            c->max_pixels);
    if ((L->total += it->n) > c->total_bytes)
        return fail(e, 413, pp, "the request's media total %zu bytes over the %zu-byte limit", L->total, c->total_bytes);
    mm_sha256_hex(it->bytes, it->n, it->sha256);
    return 0;
}

static int decode_capped(const char *b64, size_t len, size_t cap, mm_item *it, const char *what, const char *pp, mm_err *e) {
    if (len / 4 * 3 > cap + 3) // early refusal before allocating
        return fail(e, 413, pp, "%s is about %zu bytes decoded; the limit is %zu", what, len / 4 * 3, cap);
    if (mm_b64_decode(b64, len, &it->bytes, &it->n)) return fail(e, 400, pp, "%s: invalid base64 data", what);
    if (!it->n) return fail(e, 400, pp, "%s: empty data", what);
    if (it->n > cap) return fail(e, 413, pp, "%s is %zu bytes decoded; the limit is %zu", what, it->n, cap);
    return 0;
}

static int add_image(const jv *item, mm_list *L, const mm_caps *c, const char *pp, mm_item *it, mm_err *e) {
    const jv *iu = json_get(item, "image_url"), *im = json_get(item, "image"), *src = iu ? iu : im;
    const char *url = NULL;
    if (src && src->type == J_STR) url = src->str;
    else if (src && src->type == J_OBJ) {
        const jv *u = json_get(src, "url"), *d = json_get(src, "detail");
        if (u && u->type == J_STR) url = u->str;
        if (d && d->type != J_NULL) {
            if (d->type != J_STR || (strcmp(d->str, "auto") && strcmp(d->str, "low") && strcmp(d->str, "high")))
                return fail(e, 400, pp, "image_url.detail must be auto, low or high");
            snprintf(it->detail, sizeof it->detail, "%s", d->str);
        }
    }
    if (!*it->detail) strcpy(it->detail, "auto");
    if (!url) return fail(e, 400, pp, "image part needs image_url.url (a data: URI or an http(s) URL)");
    if (++L->images > c->max_images) return fail(e, 400, pp, "too many images: the limit is %d per request", c->max_images);
    if (!strncasecmp(url, "data:", 5)) {
        const char *comma = strchr(url, ',');
        if (!comma) return fail(e, 400, pp, "malformed data: URI (no comma)");
        size_t hl = (size_t)(comma - url - 5);
        if (hl < 7 || strncasecmp(comma - 7, ";base64", 7))
            return fail(e, 400, pp, "data: URI images must be base64 (data:image/<type>;base64,...)");
        const char *semi = memchr(url + 5, ';', hl);
        snprintf(it->declared, sizeof it->declared, "%.*s", (int)(semi - url - 5), url + 5);
        if (strncasecmp(it->declared, "image/", 6))
            return fail(e, 415, pp, "data: URI media type '%s' is not image/*", it->declared);
        if (decode_capped(comma + 1, strlen(comma + 1), c->image_bytes, it, "image", pp, e)) return -1;
        return finish_image(it, L, c, pp, e);
    }
    if (!strncasecmp(url, "http://", 7) || !strncasecmp(url, "https://", 8)) {
        if (!c->allow_remote)
            return fail(e, 400, pp, "remote image URLs are disabled on this server; send the image inline as a data: URI");
        if (strlen(url) > 2048) return fail(e, 400, pp, "image URL longer than 2048 bytes");
        it->url = strdup(url);
        return 0; // fetched (and checked) by mm_fetch_all once the whole request has parsed
    }
    return fail(e, 400, pp, "image URL scheme not accepted (only data: and http(s):)");
}

static int add_audio(const jv *ia, mm_list *L, const mm_caps *c, const char *pp, mm_item *it, mm_err *e) {
    const jv *d = ia ? json_get(ia, "data") : NULL, *f = ia ? json_get(ia, "format") : NULL;
    if (!d || d->type != J_STR || !f || f->type != J_STR)
        return fail(e, 400, pp, "input_audio needs string data (base64) and format");
    if (strcmp(f->str, "wav") && strcmp(f->str, "mp3") && strcmp(f->str, "flac") && strcmp(f->str, "ogg"))
        return fail(e, 415, pp, "input_audio.format '%s' not accepted (wav, mp3, flac, ogg)", f->str);
    snprintf(it->declared, sizeof it->declared, "%s", f->str);
    if (++L->audio > c->max_audio) return fail(e, 400, pp, "too many audio clips: the limit is %d per request", c->max_audio);
    if (decode_capped(d->str, strlen(d->str), c->audio_bytes, it, "audio", pp, e)) return -1;
    const char *m = mm_sniff_audio(it->bytes, it->n);
    if (!m) return fail(e, 415, pp, "unsupported audio format (the bytes are not wav, mp3, flac or ogg)");
    if (strcmp(m, f->str)) return fail(e, 415, pp, "input_audio.format is '%s' but the bytes are %s", f->str, m);
    snprintf(it->mime, sizeof it->mime, "audio/%s", m);
    if ((L->total += it->n) > c->total_bytes)
        return fail(e, 413, pp, "the request's media total %zu bytes over the %zu-byte limit", L->total, c->total_bytes);
    mm_sha256_hex(it->bytes, it->n, it->sha256);
    return 0;
}

int mm_audio_item(const jv *input_audio, const char *param, const mm_caps *c, mm_list *L, mm_err *e) {
    if (!input_audio || input_audio->type != J_OBJ) return fail(e, 400, param, "input_audio must be an object {data, format}");
    mm_item *it = push(L);
    if (!it) return fail(e, 400, param, "out of memory");
    it->kind = MM_AUDIO;
    return add_audio(input_audio, L, c, param, it, e);
}

int mm_render_content(const jv *content, int msg, int is_system, const mm_caps *c, mm_list *L, sb *out, mm_err *e) {
    char pp[48];
    snprintf(pp, sizeof pp, "messages[%d].content", msg);
    if (!content || content->type == J_NULL) return 0;
    if (content->type == J_STR) return sb_add(out, content->str), 0;
    if (content->type != J_ARR) return fail(e, 400, pp, "Unexpected content type.");
    for (size_t i = 0; i < content->n; i++) {
        const jv *item = content->items[i], *ty = json_get(item, "type");
        snprintf(pp, sizeof pp, "messages[%d].content[%zu]", msg, i);
        if (item->type != J_OBJ) return fail(e, 400, pp, "a content part must be an object");
        const char *t = ty && ty->type == J_STR ? ty->str : "";
        // the template's order: image keys first, then video, then text
        if (json_get(item, "image") || json_get(item, "image_url") || !strcmp(t, "image") || !strcmp(t, "image_url")) {
            if (is_system) return fail(e, 400, pp, "System message cannot contain images.");
            mm_item *it = push(L);
            if (!it) return fail(e, 400, pp, "out of memory");
            it->kind = MM_IMAGE, it->msg = msg, it->part = (int)i;
            if (add_image(item, L, c, pp, it, e)) return -1;
            sb_add(out, MM_IMAGE_TAG);
        } else if (json_get(item, "video") || !strcmp(t, "video") || !strcmp(t, "video_url")) {
            return fail(e, 400, pp, "video content is not supported");
        } else if (json_get(item, "input_audio") || !strcmp(t, "input_audio")) {
            if (is_system) return fail(e, 400, pp, "System message cannot contain audio.");
            mm_item *it = push(L);
            if (!it) return fail(e, 400, pp, "out of memory");
            it->kind = MM_AUDIO, it->msg = msg, it->part = (int)i;
            if (add_audio(json_get(item, "input_audio"), L, c, pp, it, e)) return -1;
            sb_add(out, MM_AUDIO_TAG);
        } else if (json_get(item, "text")) {
            const jv *tx = json_get(item, "text");
            if (tx->type != J_STR) return fail(e, 400, pp, "a text part's text must be a string");
            sb_add(out, tx->str);
        } else
            return fail(e, 400, pp, "Unexpected item type in content.");
    }
    return 0;
}

int mm_check_engine(const mm_list *L, const mm_caps *c, mm_err *e) {
    for (size_t i = 0; i < L->n; i++) {
        char pp[48];
        snprintf(pp, sizeof pp, "messages[%d].content[%d]", L->v[i].msg, L->v[i].part);
        if (L->v[i].kind == MM_IMAGE && !c->images)
            return fail(e, 400, pp, "this model has no vision path: image input is not supported by the served engine");
        if (L->v[i].kind == MM_AUDIO && !c->audio)
            return fail(e, 400, pp, "audio input needs the ear, which is not attached to this server");
    }
    return 0;
}

int mm_expand_pads(const char *s, const unsigned *n_rows, size_t n_images, sb *out, mm_err *e) {
    size_t k = 0, pl = strlen(MM_IMAGE_PAD);
    const char *p;
    while ((p = strstr(s, MM_IMAGE_PAD))) {
        if (k >= n_images) return fail(e, 400, "", "more <|image_pad|> markers than images");
        if (!n_rows[k]) return fail(e, 400, "", "image %zu encoded to zero rows", k);
        sb_addn(out, s, (size_t)(p - s));
        for (unsigned r = 0; r < n_rows[k]; r++) sb_add(out, MM_IMAGE_PAD);
        s = p + pl, k++;
    }
    sb_add(out, s);
    if (k != n_images) return fail(e, 400, "", "%zu <|image_pad|> markers for %zu images", k, n_images);
    return 0;
}

int mm_replace_audio(const char *s, const char *const *text, size_t n_audio, sb *out, mm_err *e) {
    size_t k = 0, tl = strlen(MM_AUDIO_TAG);
    const char *p;
    while ((p = strstr(s, MM_AUDIO_TAG))) {
        if (k >= n_audio) return fail(e, 400, "", "more audio tags than audio clips");
        sb_addn(out, s, (size_t)(p - s));
        sb_add(out, text[k]);
        s = p + tl, k++;
    }
    sb_add(out, s);
    if (k != n_audio) return fail(e, 400, "", "%zu audio tags for %zu audio clips", k, n_audio);
    return 0;
}

// ---- remote fetch: resolve once, refuse private targets, pin curl to that address ----

static int private_addr(const struct sockaddr *sa) {
    if (sa->sa_family == AF_INET) {
        uint32_t a = ntohl(((const struct sockaddr_in *)sa)->sin_addr.s_addr);
        return (a >> 24) == 0 || (a >> 24) == 10 || (a >> 24) == 127 || (a >> 16) == 0xa9fe || (a >> 20) == 0xac1 ||
               (a >> 16) == 0xc0a8 || (a >> 22) == (100u << 2 | 1) || (a >> 28) >= 0xe;
    }
    if (sa->sa_family == AF_INET6) {
        const unsigned char *b = ((const struct sockaddr_in6 *)sa)->sin6_addr.s6_addr;
        static const unsigned char z[16];
        if (!memcmp(b, z, 15) && (b[15] == 0 || b[15] == 1)) return 1; // :: and ::1
        if ((b[0] & 0xfe) == 0xfc || (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) || b[0] == 0xff) return 1;
        if (!memcmp(b, z, 10) && b[10] == 0xff && b[11] == 0xff) { // ::ffff:a.b.c.d
            struct sockaddr_in v4 = {.sin_family = AF_INET};
            memcpy(&v4.sin_addr, b + 12, 4);
            return private_addr((struct sockaddr *)&v4);
        }
        return 0;
    }
    return 1;
}

static int fetch_one(mm_item *it, mm_list *L, const mm_caps *c, const char *pp, mm_err *e) {
    const char *u = it->url, *hs = strstr(u, "://") + 3;
    int https = !strncasecmp(u, "https", 5);
    size_t al = strcspn(hs, "/?#");
    char auth[300], host[256], port[8];
    if (!al || al >= sizeof auth) return fail(e, 400, pp, "image URL has no host");
    memcpy(auth, hs, al), auth[al] = 0;
    if (strchr(auth, '@')) return fail(e, 400, pp, "image URL must not carry credentials");
    char *colon;
    if (auth[0] == '[') {
        char *rb = strchr(auth, ']');
        if (!rb) return fail(e, 400, pp, "bad IPv6 host in image URL");
        snprintf(host, sizeof host, "%.*s", (int)(rb - auth - 1), auth + 1);
        colon = rb[1] == ':' ? rb + 1 : NULL;
    } else {
        colon = strrchr(auth, ':');
        snprintf(host, sizeof host, "%.*s", (int)(colon ? colon - auth : (long)al), auth);
    }
    snprintf(port, sizeof port, "%s", colon && colon[1] ? colon + 1 : https ? "443" : "80");
    struct addrinfo hints = {.ai_socktype = SOCK_STREAM}, *res = NULL;
    if (getaddrinfo(host, port, &hints, &res) || !res) return fail(e, 400, pp, "cannot resolve image host '%s'", host);
    char ip[INET6_ADDRSTRLEN];
    int priv = private_addr(res->ai_addr);
    if (res->ai_family == AF_INET) inet_ntop(AF_INET, &((struct sockaddr_in *)res->ai_addr)->sin_addr, ip, sizeof ip);
    else inet_ntop(AF_INET6, &((struct sockaddr_in6 *)res->ai_addr)->sin6_addr, ip, sizeof ip);
    int v6 = res->ai_family == AF_INET6;
    freeaddrinfo(res);
    if (priv && !c->allow_private)
        return fail(e, 400, pp, "image host '%s' resolves to a private address (%s); refused", host, ip);
    char pin[400], mt[16], mf[32];
    snprintf(pin, sizeof pin, v6 ? "%s:%s:[%s]" : "%s:%s:%s", host, port, ip);
    snprintf(mt, sizeof mt, "%d", c->fetch_timeout_s);
    snprintf(mf, sizeof mf, "%zu", c->image_bytes);
    int fd[2];
    if (pipe(fd)) return fail(e, 400, pp, "fetch: pipe failed");
    pid_t pid = fork();
    if (pid < 0) return close(fd[0]), close(fd[1]), fail(e, 400, pp, "fetch: fork failed");
    if (!pid) {
        dup2(fd[1], 1), close(fd[0]), close(fd[1]);
        int dn = open("/dev/null", 1);
        if (dn >= 0) dup2(dn, 2);
        execlp("curl", "curl", "-sS", "--fail", "--proto", "=http,https", "--max-redirs", "0", "--max-time", mt,
            "--max-filesize", mf, "--resolve", pin, "-o", "-", "--", u, (char *)NULL);
        _exit(127);
    }
    close(fd[1]);
    size_t cap = c->image_bytes + 1, n = 0;
    unsigned char *buf = malloc(cap);
    ssize_t k;
    while (buf && n < cap && (k = read(fd[0], buf + n, cap - n)) > 0) n += (size_t)k;
    close(fd[0]);
    if (n >= cap) kill(pid, SIGKILL);
    int st = 0;
    waitpid(pid, &st, 0);
    if (!buf) return fail(e, 400, pp, "out of memory");
    if (n > c->image_bytes) return free(buf), fail(e, 413, pp, "fetched image is over the %zu-byte limit", c->image_bytes);
    if (!WIFEXITED(st) || WEXITSTATUS(st)) {
        int rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        free(buf);
        if (rc == 63) return fail(e, 413, pp, "fetched image is over the %zu-byte limit", c->image_bytes);
        return fail(e, 400, pp, "could not fetch the image URL (curl rc %d)", rc);
    }
    if (!n) return free(buf), fail(e, 400, pp, "the image URL returned no data (redirects are not followed)");
    it->bytes = buf, it->n = n;
    return finish_image(it, L, c, pp, e);
}

int mm_fetch_all(mm_list *L, const mm_caps *c, mm_err *e) {
    for (size_t i = 0; i < L->n; i++) {
        if (!L->v[i].url || L->v[i].bytes) continue;
        char pp[48];
        snprintf(pp, sizeof pp, "messages[%d].content[%d]", L->v[i].msg, L->v[i].part);
        if (fetch_one(&L->v[i], L, c, pp, e)) return -1;
    }
    return 0;
}
