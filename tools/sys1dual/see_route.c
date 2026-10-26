// see_route.c — POST /v1/see for llb-serve: image in, detections out (D-FINE-N, COCO-80).
// #include'd by serve.c just before route(); it uses serve.c's Conn / json_reply / sb / jv directly.
//
// The route is registered unconditionally, so an unconfigured server answers the
// contract's 503 ("vision is not enabled on this backend"), never a 404.
//
// The detector is a child process: tools/see/see_worker.py (onnxruntime, one session for
// the server's lifetime, one JSON line in / one line out). It is forked on the first
// request and restarted on the next request after a timeout or a crash. Strict FIFO
// under see_lock: one ORT session, one request at a time.
//
// Config (environment, read once at startup):
//   LLB_SEE_SCRIPT          path to see_worker.py      (unset = vision disabled -> 503)
//   LLB_SEE_MODEL           path to dfine_n_coco.onnx  (required with LLB_SEE_SCRIPT)
//   LLB_SEE_LABELS          COCO-80 label JSON         (default: coco80.json beside the script)
//   LLB_SEE_PYTHON          python3                    interpreter with onnxruntime + Pillow (or ffmpeg)
//   LLB_SEE_EP              auto|cuda|cpu              default auto; cuda refuses a CPU fallback
//   LLB_SEE_THREADS         4                          ORT intra-op threads
//   LLB_SEE_TIMEOUT_MS      30000                      per-request detector budget
//   LLB_SEE_MAX_IMAGE_BYTES 4194304                    decoded image cap (413 above)

#define SEE_RESP_MAX (1 << 20)

typedef struct {
    const char *script, *python, *model, *labels, *ep;
    int threads, timeout_ms;
    long max_image_bytes;
} SeeCfg;

typedef struct {
    int in_fd, out_fd; // worker stdin / stdout
    pid_t pid;
    long next_id;
} SeeWorker;

static SeeCfg see_cfg;
static int see_cfg_done;
static SeeWorker see_w = {-1, -1, 0, 0};
static pthread_mutex_t see_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *see_env(const char *k, const char *d) {
    const char *v = getenv(k);
    return v && *v ? v : d;
}

static void see_cfg_init(void) {
    if (see_cfg_done) return;
    see_cfg.script = see_env("LLB_SEE_SCRIPT", "");
    see_cfg.python = see_env("LLB_SEE_PYTHON", "python3");
    see_cfg.model = see_env("LLB_SEE_MODEL", "");
    see_cfg.labels = see_env("LLB_SEE_LABELS", "");
    see_cfg.ep = see_env("LLB_SEE_EP", "auto");
    see_cfg.threads = atoi(see_env("LLB_SEE_THREADS", "4"));
    see_cfg.timeout_ms = atoi(see_env("LLB_SEE_TIMEOUT_MS", "30000"));
    see_cfg.max_image_bytes = atol(see_env("LLB_SEE_MAX_IMAGE_BYTES", "4194304"));
    if (see_cfg.threads < 1) see_cfg.threads = 1;
    if (see_cfg.timeout_ms < 1000) see_cfg.timeout_ms = 1000;
    if (see_cfg.max_image_bytes < 1024) see_cfg.max_image_bytes = 1024;
    see_cfg_done = 1;
    if (*see_cfg.script)
        logf_("see: /v1/see on: %s %s --model %s --ep %s (forked on first request)", see_cfg.python, see_cfg.script,
            *see_cfg.model ? see_cfg.model : "(LLB_SEE_MODEL unset)", see_cfg.ep);
}

// ---- small helpers ----------------------------------------------------------

// the one error shape (OpenAI's)
static char *see_err(int code, const char *msg) {
    sb e = {0};
    sb_add(&e, "{\"error\":{\"message\":");
    json_quote(&e, msg);
    sb_fmt(&e, ",\"type\":\"%s\",\"param\":null,\"code\":%d}}",
           code >= 500 ? "unavailable_error" : "invalid_request_error", code);
    return e.p;
}

static int see_b64_val(const char *s, size_t n) {
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

// ---- worker -----------------------------------------------------------------

static void see_worker_stop(void) {
    if (see_w.out_fd >= 0) { close(see_w.out_fd); see_w.out_fd = -1; }
    if (see_w.in_fd >= 0) { close(see_w.in_fd); see_w.in_fd = -1; }
    if (see_w.pid > 0) {
        kill(see_w.pid, SIGTERM);
        waitpid(see_w.pid, NULL, 0);
        see_w.pid = 0;
    }
}

static int see_worker_start(char *err, size_t errlen) {
    see_worker_stop();
    if (!*see_cfg.model) {
        snprintf(err, errlen, "vision is not configured (set LLB_SEE_MODEL)");
        return -1;
    }
    int pin[2], pout[2];
    if (pipe2(pin, O_CLOEXEC) || pipe2(pout, O_CLOEXEC)) {
        snprintf(err, errlen, "cannot create detector pipes: %s", strerror(errno));
        return -1;
    }
    char tbuf[16], mbuf[32]; // formatted before fork: the child only dup2s, closes and execs
    snprintf(tbuf, sizeof tbuf, "%d", see_cfg.threads);
    snprintf(mbuf, sizeof mbuf, "%ld", see_cfg.max_image_bytes);
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, errlen, "cannot fork detector: %s", strerror(errno));
        close(pin[0]), close(pin[1]), close(pout[0]), close(pout[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(pin[0], 0);
        dup2(pout[1], 1);
        close_extra_fds(); // the listen socket, client sockets and engine pipes stay with the server
        if (*see_cfg.labels)
            execlp(see_cfg.python, see_cfg.python, "-u", see_cfg.script, "--model", see_cfg.model, "--labels",
                   see_cfg.labels, "--ep", see_cfg.ep, "--threads", tbuf, "--max-image-bytes", mbuf, (char *)NULL);
        else
            execlp(see_cfg.python, see_cfg.python, "-u", see_cfg.script, "--model", see_cfg.model, "--ep", see_cfg.ep,
                   "--threads", tbuf, "--max-image-bytes", mbuf, (char *)NULL);
        _exit(127);
    }
    close(pin[0]);
    close(pout[1]);
    see_w.in_fd = pin[1];
    see_w.out_fd = pout[0];
    see_w.pid = pid;
    logf_("see: detector pid=%d (loading on first request)", (int)pid);
    return 0;
}

// one line, with a wall-clock budget. 0 = ok, -1 = eof/error, -2 = timeout, -3 = too long
static int see_read_line(int fd, char *buf, size_t cap, int timeout_ms) {
    size_t n = 0;
    double end = now() + timeout_ms / 1000.0;
    for (;;) {
        double left = end - now();
        if (left <= 0) return -2;
        struct pollfd pf = {.fd = fd, .events = POLLIN};
        int pr = poll(&pf, 1, (int)(left * 1000) + 1);
        if (pr < 0) { if (errno == EINTR) continue; return -1; }
        if (pr == 0) return -2;
        ssize_t r = read(fd, buf + n, cap - 1 - n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        n += (size_t)r;
        if (memchr(buf, '\n', n)) {
            buf[n] = 0;
            char *nl = strchr(buf, '\n');
            if (nl) *nl = 0;
            return 0;
        }
        if (n >= cap - 1) return -3;
    }
}

// returns malloc'd response body, sets *status. Never NULL.
static char *see_process(const char *body_s, int *status) {
    see_cfg_init();
    *status = 200;
    if (!*see_cfg.script) {
        *status = 503;
        return see_err(503, "vision is not enabled on this backend");
    }
    const char *perr = NULL;
    jv *body = json_parse(body_s, &perr);
    if (!body || body->type != J_OBJ) {
        json_free(body);
        *status = 400;
        return see_err(400, perr ? perr : "body must be a JSON object");
    }
    const jv *img = json_get(body, "image");
    if (!img) img = json_get(body, "image_b64");
    const jv *task = json_get(body, "task"), *thr = json_get(body, "threshold"),
             *md = json_get(body, "max_detections");
    char msg[256];
    int code = 0;
    double threshold = 0.4;
    long max_det = 100;

    if (task && task->type != J_STR) code = 400, snprintf(msg, sizeof msg, "task must be a string");
    else if (task && task->type == J_STR && strcmp(task->str, "detect") &&
             strcmp(task->str, "describe")) code = 400, snprintf(msg, sizeof msg, "task must be detect or describe");
    else if (task && task->type == J_STR && !strcmp(task->str, "describe")) {
        json_free(body);
        *status = 503;
        return see_err(503, "vision describe is not enabled on this backend (detect-only build)");
    }
    else if (!img || img->type != J_STR || !img->str[0]) code = 400, snprintf(msg, sizeof msg, "image must be a base64 string or data URL");
    else if (thr && (thr->type != J_NUM || thr->num < 0.0 || thr->num > 1.0)) code = 400, snprintf(msg, sizeof msg, "threshold must be a number in [0,1]");
    else if (md && (md->type != J_NUM || md->num < 1 || md->num > 300 || md->num != (long)md->num)) code = 400, snprintf(msg, sizeof msg, "max_detections must be an integer in [1,300]");
    if (code) {
        json_free(body);
        *status = code;
        return see_err(code, msg);
    }
    if (thr) threshold = thr->num;
    if (md) max_det = (long)md->num;

    const char *b64 = img->str;
    if (!strncmp(b64, "data:", 5)) {
        const char *comma = strchr(b64, ',');
        if (!comma || !strstr(b64, ";base64")) {
            json_free(body);
            *status = 400;
            return see_err(400, "image must be a base64 data URL");
        }
        b64 = comma + 1;
    }
    size_t blen = strlen(b64);
    long max_b64 = (see_cfg.max_image_bytes + 2) / 3 * 4 + 8;
    if (blen > (size_t)max_b64) {
        json_free(body);
        *status = 413;
        return see_err(413, "image is too large");
    }
    if (!see_b64_val(b64, blen)) {
        json_free(body);
        *status = 400;
        return see_err(400, "image is not valid base64");
    }

    // one worker, one request at a time
    pthread_mutex_lock(&see_lock);
    char lerr[256] = {0};
    if (see_w.pid <= 0 && see_worker_start(lerr, sizeof lerr)) {
        pthread_mutex_unlock(&see_lock);
        json_free(body);
        *status = 503;
        return see_err(503, lerr[0] ? lerr : "detector did not start");
    }
    long id = ++see_w.next_id;
    sb req = {0};
    sb_fmt(&req, "{\"id\":%ld,\"image_b64\":\"", id);
    sb_add(&req, b64);
    sb_fmt(&req, "\",\"threshold\":%.6g,\"max_detections\":%ld,\"task\":\"detect\"}\n", threshold, max_det);
    json_free(body);

    size_t off = 0;
    while (off < req.n) {
        ssize_t w = write(see_w.in_fd, req.p + off, req.n - off);
        if (w > 0) { off += (size_t)w; continue; }
        if (w < 0 && errno == EINTR) continue;
        break;
    }
    int wrote = off == req.n;
    free(req.p);
    if (!wrote) {
        see_worker_stop();
        pthread_mutex_unlock(&see_lock);
        *status = 503;
        return see_err(503, "detector is not running");
    }

    char *line = malloc(SEE_RESP_MAX);
    if (!line) {
        pthread_mutex_unlock(&see_lock);
        *status = 500;
        return see_err(500, "out of memory");
    }
    int r = see_read_line(see_w.out_fd, line, SEE_RESP_MAX, see_cfg.timeout_ms);
    if (r != 0) {
        see_worker_stop(); // a timeout or a dead child: the next request starts fresh
        pthread_mutex_unlock(&see_lock);
        free(line);
        *status = 503;
        return see_err(503, r == -2 ? "detector timed out" : r == -3 ? "detector reply too large"
                                                                     : "detector is not running");
    }
    pthread_mutex_unlock(&see_lock);

    // protocol line: "<id> <json>"; the JSON after the space is the response body
    char *sp = strchr(line, ' ');
    if (!sp || strtol(line, NULL, 10) != id) {
        free(line);
        *status = 503;
        return see_err(503, "detector protocol desync");
    }
    const char *rbody = sp + 1;
    const char *e2 = NULL;
    jv *resp = json_parse(rbody, &e2);
    if (!resp || resp->type != J_OBJ) {
        free(line);
        json_free(resp);
        *status = 500;
        return see_err(500, "detector produced invalid JSON");
    }
    const jv *er = json_get(resp, "error");
    if (er && er->type == J_OBJ) {
        const jv *ec = json_get(er, "code");
        *status = ec && ec->type == J_NUM ? (int)ec->num : 500;
    } else {
        *status = 200;
    }
    json_free(resp);
    char *out = strdup(rbody);
    free(line);
    return out;
}

static void handle_see(Conn *c, const char *body_s) {
    int status = 200;
    char *out = see_process(body_s, &status);
    if (!out) out = see_err(500, "detector failed");
    json_reply(c, status, out);
    free(out);
}
