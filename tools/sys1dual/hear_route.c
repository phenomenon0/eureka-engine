// hear_route.c — POST /v1/hear for llb-serve: WAV in, transcript out.
// #include'd by serve.c just before route(); it uses serve.c's Conn / json_reply / sb / jv directly.
//
// The route is registered unconditionally, so an unconfigured server answers the
// contract's 503 ("hearing is not enabled on this backend"), never a 404.
//
// Design: a proxy to a resident whisper.cpp `whisper-server` on loopback HTTP. You start
// the sidecar yourself (any ggml backend: CUDA, Metal, CPU); this route only speaks HTTP
// to it. Strict FIFO under hear_lock: the whisper context is single-slot.
// This is separate from the own-ear route (--ear, POST /v1/audio/transcriptions).
//
// Config (environment, read once at startup):
//   LLB_HEAR_URL              http://127.0.0.1:PORT/inference  (unset = hearing disabled -> 503)
//   LLB_HEAR_MODEL_ID         whisper-base.en-q5_1             reported as "model" when the request has none
//   LLB_HEAR_MODEL_PATH       (empty)                          reported, not opened
//   LLB_HEAR_MODEL_SHA256     (empty)                          reported; hash it in your launch script
//   LLB_HEAR_ENGINE           whisper.cpp                      reported as llb.engine (e.g. "whisper.cpp (GGML_CUDA)")
//   LLB_HEAR_TIMEOUT_MS       30000                            per-request upstream budget
//   LLB_HEAR_MAX_AUDIO_BYTES  16777216                         decoded WAV cap (413 above; --max-body-mb also applies)

#define HEAR_RESP_MAX (4 << 20)

typedef struct {
    const char *url, *host, *path, *model_id, *model_path, *model_sha, *engine;
    int port, timeout_ms;
    long max_audio_bytes;
} HearCfg;

static HearCfg hear_cfg;
static int hear_cfg_done;
static pthread_mutex_t hear_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *hear_env(const char *k, const char *d) {
    const char *v = getenv(k);
    return v && *v ? v : d;
}

// "http://host:port/path" -> host/port/path. Only plain HTTP (loopback) is expected; https is refused.
static int hear_parse_url(const char *url, char *host, size_t hcap, int *port, char *path, size_t pcap) {
    const char *p = url;
    if (!strncmp(p, "http://", 7)) p += 7;
    else return -1;
    const char *slash = strchr(p, '/');
    const char *colon = strchr(p, ':');
    size_t hlen;
    if (colon && (!slash || colon < slash)) {
        hlen = (size_t)(colon - p);
        *port = atoi(colon + 1);
    } else {
        hlen = slash ? (size_t)(slash - p) : strlen(p);
        *port = 80;
    }
    if (hlen == 0 || hlen >= hcap) return -1;
    memcpy(host, p, hlen);
    host[hlen] = 0;
    if (slash) snprintf(path, pcap, "%s", slash);
    else snprintf(path, pcap, "/inference");
    return (*port > 0 && *port < 65536) ? 0 : -1;
}

static void hear_cfg_init(void) {
    if (hear_cfg_done) return;
    static char host[256], path[512];
    hear_cfg.url = hear_env("LLB_HEAR_URL", "");
    hear_cfg.model_id = hear_env("LLB_HEAR_MODEL_ID", "whisper-base.en-q5_1");
    hear_cfg.model_path = hear_env("LLB_HEAR_MODEL_PATH", "");
    hear_cfg.model_sha = hear_env("LLB_HEAR_MODEL_SHA256", "");
    hear_cfg.engine = hear_env("LLB_HEAR_ENGINE", "whisper.cpp");
    hear_cfg.timeout_ms = atoi(hear_env("LLB_HEAR_TIMEOUT_MS", "30000"));
    hear_cfg.max_audio_bytes = atol(hear_env("LLB_HEAR_MAX_AUDIO_BYTES", "16777216"));
    if (hear_cfg.timeout_ms < 1000) hear_cfg.timeout_ms = 1000;
    if (hear_cfg.max_audio_bytes < 1024) hear_cfg.max_audio_bytes = 1024;
    hear_cfg.host = host;
    hear_cfg.path = path;
    if (*hear_cfg.url && hear_parse_url(hear_cfg.url, host, sizeof host, &hear_cfg.port, path, sizeof path)) {
        logf_("hear: LLB_HEAR_URL is not http://HOST:PORT/PATH — /v1/hear disabled");
        hear_cfg.url = "";
    }
    hear_cfg_done = 1;
    if (*hear_cfg.url) logf_("hear: /v1/hear on: whisper-server at %s:%d%s", host, hear_cfg.port, path);
}

// ---- small helpers ----------------------------------------------------------

static char *hear_err(int code, const char *msg) {
    sb e = {0};
    sb_add(&e, "{\"error\":{\"message\":");
    json_quote(&e, msg);
    sb_fmt(&e, ",\"type\":\"%s\",\"param\":null,\"code\":%d}}",
           code >= 500 ? "unavailable_error" : "invalid_request_error", code);
    return e.p;
}

static int hear_b64_val(const char *s, size_t n) {
    if (n == 0 || n % 4) return 0;
    size_t pad = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                 c == '+' || c == '/';
        if (c == '=') {
            if (i < n - 2 || pad == 2) return 0;
            pad++;
        } else if (!ok || pad) {
            return 0;
        }
    }
    return 1;
}

static int hear_b64_dec(const char *s, size_t n, unsigned char *out, size_t cap, size_t *outn) {
    int val[256];
    for (int i = 0; i < 256; i++) val[i] = -1;
    const char *abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) val[(unsigned char)abc[i]] = i;
    size_t o = 0;
    unsigned int acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; i++) {
        int v = val[(unsigned char)s[i]];
        if (v < 0) {
            if (s[i] == '=') break;
            return -1;
        }
        acc = (acc << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (o >= cap) return -1;
            out[o++] = (unsigned char)((acc >> bits) & 0xFF);
        }
    }
    *outn = o;
    return 0;
}

// RIFF/WAVE: channels, rate, bits, data bytes -> seconds. 0 on anything unrecognised.
static double hear_wav_seconds(const unsigned char *p, size_t n) {
    if (n < 44) return 0.0;
    if (memcmp(p, "RIFF", 4) || memcmp(p + 8, "WAVE", 4)) return 0.0;
    unsigned channels = 0, bits = 0, rate = 0;
    unsigned long long data = 0;
    size_t pos = 12;
    while (pos + 8 <= n) {
        const unsigned char *id = p + pos;
        unsigned long sz = (unsigned long)p[pos + 4] | ((unsigned long)p[pos + 5] << 8) |
                           ((unsigned long)p[pos + 6] << 16) | ((unsigned long)p[pos + 7] << 24);
        if (!memcmp(id, "fmt ", 4) && pos + 8 + 16 <= n) {
            channels = (unsigned)p[pos + 10] | ((unsigned)p[pos + 11] << 8);
            rate = (unsigned)p[pos + 12] | ((unsigned)p[pos + 13] << 8) |
                   ((unsigned)p[pos + 14] << 16) | ((unsigned)p[pos + 15] << 24);
            bits = (unsigned)p[pos + 22] | ((unsigned)p[pos + 23] << 8);
        } else if (!memcmp(id, "data", 4)) {
            data = sz;
            break;
        }
        pos += 8 + sz + (sz & 1);
    }
    unsigned long long bps = (unsigned long long)channels * rate * (bits / 8);
    if (!bps || !data) return 0.0;
    return (double)data / (double)bps;
}

// ---- upstream HTTP ----------------------------------------------------------

static int hear_connect(const char *h, int p, int timeout_ms, char *err, size_t errlen) {
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16];
    snprintf(ps, sizeof ps, "%d", p);
    int g = getaddrinfo(h, ps, &hints, &res);
    if (g) {
        snprintf(err, errlen, "resolve %s: %s", h, gai_strerror(g));
        return -1;
    }
    int fd = socket(res->ai_family, res->ai_socktype | SOCK_CLOEXEC, res->ai_protocol);
    if (fd < 0) {
        snprintf(err, errlen, "socket: %s", strerror(errno));
        freeaddrinfo(res);
        return -1;
    }
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    if (connect(fd, res->ai_addr, res->ai_addrlen)) {
        snprintf(err, errlen, "connect %s:%d: %s", h, p, strerror(errno));
        close(fd);
        freeaddrinfo(res);
        return -1;
    }
    freeaddrinfo(res);
    return fd;
}

// read until EOF or cap; returns bytes read, -1 on error/timeout, -2 when too large
static long hear_recv_all(int fd, char *buf, size_t cap) {
    size_t n = 0;
    for (;;) {
        if (n + 4096 > cap) return -2;
        ssize_t r = read(fd, buf + n, cap - n - 1);
        if (r > 0) { n += (size_t)r; continue; }
        if (r == 0) break;
        if (errno == EINTR) continue;
        return -1;
    }
    buf[n] = 0;
    return (long)n;
}

// body after the header block; de-chunks when Transfer-Encoding: chunked
static char *hear_http_body(char *raw, size_t rawlen, int *status) {
    char *hdr_end = strstr(raw, "\r\n\r\n");
    if (!hdr_end) return NULL;
    if (sscanf(raw, "HTTP/%*d.%*d %d", status) != 1) return NULL;
    char *body = hdr_end + 4;
    size_t blen = rawlen - (size_t)(body - raw);
    char *h = strcasestr(raw, "\r\ntransfer-encoding:");
    int chunked = h && h < hdr_end && strcasestr(h, "chunked") && strcasestr(h, "chunked") < hdr_end;
    if (!chunked) return strndup(body, blen);
    sb out = {0};
    size_t pos = 0;
    while (pos < blen) {
        char *end = NULL;
        long sz = strtol(body + pos, &end, 16);
        if (end == body + pos || sz < 0) break;
        while (end < body + blen && *end != '\n') end++;
        if (end >= body + blen) break;
        pos = (size_t)(end - body) + 1;
        if (sz == 0) break;
        if (pos + (size_t)sz > blen) break;
        sb_addn(&out, body + pos, (size_t)sz);
        pos += (size_t)sz;
        while (pos + 1 < blen && body[pos] == '\r' && body[pos + 1] == '\n') pos += 2;
    }
    return out.p;
}

typedef struct { int status; char *body; double ms; char err[256]; } HearUp;

static void hear_upstream(const unsigned char *wav, size_t wavn, HearUp *up) {
    up->status = 0;
    up->body = NULL;
    up->err[0] = 0;
    char err[256] = {0};
    double t0 = now();
    int fd = hear_connect(hear_cfg.host, hear_cfg.port, hear_cfg.timeout_ms, err, sizeof err);
    if (fd < 0) { snprintf(up->err, sizeof up->err, "%s", err); return; }
    static const char *bnd = "----llbhear7f3c9a1d";
    sb pre = {0};
    sb_fmt(&pre,
           "--%s\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n"
           "--%s\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n0\r\n"
           "--%s\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n"
           "--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
           "Content-Type: audio/wav\r\n\r\n", bnd, bnd, bnd, bnd);
    sb suf = {0};
    sb_fmt(&suf, "\r\n--%s--\r\n", bnd);
    size_t clen = pre.n + wavn + suf.n;
    sb head = {0};
    sb_fmt(&head, "POST %s HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: multipart/form-data; boundary=%s\r\n"
                   "Content-Length: %zu\r\nConnection: close\r\n\r\n",
           hear_cfg.path, hear_cfg.host, hear_cfg.port, bnd, clen);
    int bad = send_all(fd, head.p, head.n) || send_all(fd, pre.p, pre.n) ||
              send_all(fd, (const char *)wav, wavn) || send_all(fd, suf.p, suf.n);
    free(pre.p); free(suf.p); free(head.p);
    if (bad) {
        snprintf(up->err, sizeof up->err, "write to whisper-server failed: %s", strerror(errno));
        close(fd);
        return;
    }
    char *raw = malloc(HEAR_RESP_MAX);
    if (!raw) { snprintf(up->err, sizeof up->err, "out of memory"); close(fd); return; }
    long r = hear_recv_all(fd, raw, HEAR_RESP_MAX);
    close(fd);
    if (r < 0) {
        snprintf(up->err, sizeof up->err, r == -2 ? "whisper-server reply too large"
                                                 : "whisper-server read failed/timed out");
        free(raw);
        return;
    }
    up->ms = (now() - t0) * 1000.0;
    int st = 0;
    char *b = hear_http_body(raw, (size_t)r, &st);
    free(raw);
    if (!b) {
        snprintf(up->err, sizeof up->err, "whisper-server sent no parseable HTTP response");
        return;
    }
    up->status = st;
    up->body = b;
}

// ---- request -----------------------------------------------------------------

// returns malloc'd response body, sets *status. Never NULL.
static char *hear_process(const char *body_s, int *status) {
    hear_cfg_init();
    *status = 200;
    if (!*hear_cfg.url) {
        *status = 503;
        return hear_err(503, "hearing is not enabled on this backend");
    }
    const char *perr = NULL;
    jv *body = json_parse(body_s, &perr);
    if (!body || body->type != J_OBJ) {
        json_free(body);
        *status = 400;
        return hear_err(400, perr ? perr : "body must be a JSON object");
    }
    const jv *au = json_get(body, "audio");
    if (!au) au = json_get(body, "audio_b64");
    const jv *task = json_get(body, "task"), *lang = json_get(body, "language"),
             *model = json_get(body, "model");
    char msg[256];
    int code = 0;
    if (task && task->type != J_STR) code = 400, snprintf(msg, sizeof msg, "task must be a string");
    else if (task && task->type == J_STR && strcmp(task->str, "transcribe")) {
        json_free(body);
        *status = 503;
        return hear_err(503, "hearing translate is not enabled on this backend (transcribe-only build)");
    } else if (!au || au->type != J_STR || !au->str[0])
        code = 400, snprintf(msg, sizeof msg, "audio must be a base64 WAV string or data URL");
    else if (lang && lang->type != J_STR) code = 400, snprintf(msg, sizeof msg, "language must be a string");
    else if (model && model->type != J_STR) code = 400, snprintf(msg, sizeof msg, "model must be a string");
    if (code) {
        json_free(body);
        *status = code;
        return hear_err(code, msg);
    }

    const char *b64 = au->str;
    if (!strncmp(b64, "data:", 5)) {
        const char *comma = strchr(b64, ',');
        if (!comma || !strstr(b64, ";base64")) {
            json_free(body);
            *status = 400;
            return hear_err(400, "audio must be a base64 data URL");
        }
        b64 = comma + 1;
    }
    size_t blen = strlen(b64);
    long max_b64 = (hear_cfg.max_audio_bytes + 2) / 3 * 4 + 8;
    if (blen > (size_t)max_b64) {
        json_free(body);
        *status = 413;
        return hear_err(413, "audio is too large");
    }
    if (!hear_b64_val(b64, blen)) {
        json_free(body);
        *status = 400;
        return hear_err(400, "audio is not valid base64");
    }
    size_t dcap = blen / 4 * 3 + 4;
    unsigned char *wav = malloc(dcap);
    if (!wav) {
        json_free(body);
        *status = 500;
        return hear_err(500, "out of memory");
    }
    size_t wavn = 0;
    if (hear_b64_dec(b64, blen, wav, dcap, &wavn) || wavn < 12) {
        free(wav);
        json_free(body);
        *status = 400;
        return hear_err(400, "audio is not valid base64");
    }
    double secs = hear_wav_seconds(wav, wavn);
    if (secs <= 0.0) {
        free(wav);
        json_free(body);
        *status = 400;
        return hear_err(400, "audio must be a RIFF/WAVE file with a readable data chunk");
    }

    HearUp up = {0};
    pthread_mutex_lock(&hear_lock);
    hear_upstream(wav, wavn, &up);
    pthread_mutex_unlock(&hear_lock);
    free(wav);

    if (!up.body || up.status != 200) {
        char m[512];
        const char *why = up.err[0] ? up.err : "whisper-server returned an error";
        snprintf(m, sizeof m, "%s (upstream %d)", why, up.status);
        free(up.body);
        json_free(body);
        *status = 503;
        return hear_err(503, m);
    }
    const char *e2 = NULL;
    jv *resp = json_parse(up.body, &e2);
    free(up.body);
    if (!resp || resp->type != J_OBJ) {
        json_free(resp);
        json_free(body);
        *status = 503;
        return hear_err(503, "whisper-server produced invalid JSON");
    }
    const jv *text = json_get(resp, "text");
    if (!text || text->type != J_STR) {
        json_free(resp);
        json_free(body);
        *status = 503;
        return hear_err(503, "whisper-server reply had no text");
    }
    char *t = text->str;
    while (*t == ' ') t++;
    size_t tl = strlen(t);
    while (tl && (t[tl - 1] == ' ' || t[tl - 1] == '\n' || t[tl - 1] == '\r')) t[--tl] = 0;
    double rtf = secs > 0 ? (up.ms / 1000.0) / secs : 0.0;

    sb o = {0};
    sb_add(&o, "{\"model\":");
    json_quote(&o, (model && model->type == J_STR) ? model->str : hear_cfg.model_id);
    sb_add(&o, ",\"task\":\"transcribe\",\"text\":");
    json_quote(&o, t);
    sb_add(&o, ",\"language\":");
    json_quote(&o, (lang && lang->type == J_STR) ? lang->str : "en");
    sb_fmt(&o, ",\"audio_seconds\":%.4f,\"usage\":{\"audio_seconds\":%.4f},", secs, secs);
    sb_add(&o, "\"llb\":{\"engine\":");
    json_quote(&o, hear_cfg.engine);
    sb_fmt(&o, ",\"route\":\"/v1/hear\",\"ms\":%.2f,\"rtf\":%.6f,\"audio_bytes\":%zu,", up.ms, rtf, wavn);
    sb_add(&o, "\"model_path\":");
    json_quote(&o, hear_cfg.model_path);
    sb_add(&o, ",\"model_sha256\":");
    json_quote(&o, hear_cfg.model_sha);
    sb_fmt(&o, ",\"upstream\":\"%s:%d%s\"}}", hear_cfg.host, hear_cfg.port, hear_cfg.path);
    json_free(resp);
    json_free(body);
    *status = 200;
    return o.p;
}

static void handle_hear(Conn *c, const char *body_s) {
    int status = 200;
    char *out = hear_process(body_s, &status);
    if (!out) out = hear_err(500, "hearing failed");
    json_reply(c, status, out);
    free(out);
}
