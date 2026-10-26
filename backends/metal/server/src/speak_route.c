// speak_route.c — POST /v1/speak: {"text","voice?","style?"} -> 24 kHz WAV.
//
// M2-SPEAK-SCAFFOLD, CPU-only. No engine, no GPU, no libespeak. The route
// fork/execs a configured launcher (voice_p3_misaki.py via speak-p3.sh), waits
// for the WAV file, validates the RIFF header (24 000 Hz mono PCM16), and
// writes the bytes through the sink. The sink owns HTTP framing, so this file
// compiles and runs without serve.c and without a socket.
#define _GNU_SOURCE 1
#include "speak_route.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_TEXT 2000u
#define MAX_REQ_BODY (1u << 20)
#define MAX_WAV (32u << 20)
#define DEFAULT_TIMEOUT_MS 180000

// P3's browser runner writes to ONE jobs/ dir next to voice_p3.py
// (jobs/request.json, jobs/wave-*.f32), so two syntheses in one process would
// corrupt each other. Serialize inside the route; the HTTP server's threads
// still accept and queue connections while one synthesis runs.
static pthread_mutex_t speak_lock = PTHREAD_MUTEX_INITIALIZER;

// The six sealed Kokoro style banks present in voice-intangibles/voices/.
static const char *const VOICES[] = {
    "af_bella", "af_heart", "af_nicole", "af_sky", "am_michael", "bm_george", NULL
};

// "style" is a named prosody preset in this contract, not a Kokoro style
// vector. The P3 vector slots are untrained (accepted_inert) and are not
// exposed to HTTP callers; the drive knobs below are the measured ones.
static const struct {
    const char *name;
    const char *drive; // NULL = native durations/F0, no --drive argument
} STYLES[] = {
    {"neutral", NULL},
    {"slow", "{\"global_speed\":0.8}"},
    {"fast", "{\"global_speed\":1.25}"},
    {"pitch_up", "{\"global_f0_offset_st\":2}"},
    {"pitch_down", "{\"global_f0_offset_st\":-2}"},
    {NULL, NULL},
};

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static unsigned rd16(const unsigned char *p) {
    return (unsigned)p[0] | (unsigned)p[1] << 8;
}

static int ascii_ok(const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        if (*p > 0x7e || (*p < 0x20 && *p != '\n' && *p != '\t')) return 0;
    return 1;
}

static int voice_ok(const char *v) {
    for (int i = 0; VOICES[i]; i++)
        if (!strcmp(v, VOICES[i])) return 1;
    return 0;
}

int speak_route_parse(const char *body, size_t body_len, const char **voice,
                      const char **style, const char **err) {
    static char why[256];
    // `voice` points into the parsed JSON, which is freed before this function
    // returns; the handler then uses it for argv and for X-LLB-Voice. Copy it
    // into per-thread storage so the pointer stays valid. The dangling read was
    // real: 1 of 10 identical requests reached the launcher with voice=""
    // (FileNotFoundError .../voices/.bin on the PC).
    static _Thread_local char voicebuf[64];
    *voice = "af_heart";
    *style = "neutral";
    *err = NULL;
    if (!body) return *err = "empty request body", -1;
    if (body_len > MAX_REQ_BODY) return *err = "request body too large", -1;

    const char *perr = NULL;
    jv *b = json_parse(body, &perr);
    if (!b) {
        snprintf(why, sizeof why, "%s", perr ? perr : "invalid JSON");
        return *err = why, -1;
    }
    int rc = -1;
    const jv *t, *v, *s;
    do {
        if (b->type != J_OBJ) { *err = "request body must be a JSON object"; break; }
        t = json_get(b, "text");
        if (!t || t->type != J_STR) { *err = "text must be a string"; break; }
        size_t n = strlen(t->str);
        if (!n) { *err = "text must not be empty"; break; }
        if (n > MAX_TEXT) { *err = "text exceeds 2000 characters"; break; }
        if (!ascii_ok(t->str)) { *err = "text must be English ASCII (the P3 tokenizer is en-US)"; break; }
        v = json_get(b, "voice");
        if (v && v->type != J_NULL) {
            if (v->type != J_STR) { *err = "voice must be a string"; break; }
            if (!voice_ok(v->str)) {
                snprintf(why, sizeof why, "unsupported voice %s; known: af_bella af_heart af_nicole af_sky am_michael bm_george", v->str);
                *err = why;
                break;
            }
            snprintf(voicebuf, sizeof voicebuf, "%s", v->str);
            *voice = voicebuf;
        }
        s = json_get(b, "style");
        if (s && s->type != J_NULL) {
            if (s->type != J_STR) { *err = "style must be a string"; break; }
            int found = 0;
            for (int i = 0; STYLES[i].name; i++)
                if (!strcmp(s->str, STYLES[i].name)) { *style = STYLES[i].name; found = 1; break; }
            if (!found) {
                snprintf(why, sizeof why, "unsupported style %s; known: neutral slow fast pitch_up pitch_down", s->str);
                *err = why;
                break;
            }
        }
        rc = 0;
    } while (0);
    json_free(b);
    return rc;
}

static const char *style_drive(const char *name) {
    for (int i = 0; STYLES[i].name; i++)
        if (!strcmp(name, STYLES[i].name)) return STYLES[i].drive;
    return NULL;
}

// O_EXCL unique path, so concurrent requests never share scratch files.
static int mkpath(char *dst, size_t cap, const char *dir, const char *ext) {
    for (int i = 0; i < 100; i++) {
        int k = snprintf(dst, cap, "%s/llb-speak-%ld-%d.%s", dir, (long)getpid(), i, ext);
        if (k < 0 || (size_t)k >= cap) return -1;
        int fd = open(dst, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) return close(fd), 0;
        if (errno != EEXIST) return -1;
    }
    return -1;
}

static void tail_log(const char *path, char *why, size_t cap) {
    char buf[513];
    int fd = open(path, O_RDONLY);
    why[0] = 0;
    if (fd < 0) return;
    off_t end = lseek(fd, 0, SEEK_END);
    off_t start = end > 512 ? end - 512 : 0;
    ssize_t got = 0;
    if (lseek(fd, start, SEEK_SET) >= 0) got = read(fd, buf, 512);
    close(fd);
    if (got <= 0) return;
    buf[got] = 0;
    for (ssize_t i = 0; i < got; i++)
        if ((unsigned char)buf[i] > 0x7e || ((unsigned char)buf[i] < 0x20 && buf[i] != '\n')) buf[i] = ' ';
    char *last = buf; // the last non-empty line carries the exception text
    for (char *p = buf; *p; p++)
        if (*p == '\n' && p[1]) last = p + 1;
    size_t k = strlen(last);
    while (k && (last[k - 1] == '\n' || last[k - 1] == ' ')) last[--k] = 0;
    snprintf(why, cap, "%s", last);
}

static int run_child(const speak_cfg *cfg, char *const av[], const char *log,
                     int timeout_ms, char *why, size_t cap) {
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(why, cap, "fork failed: %s", strerror(errno));
        return -1;
    }
    if (!pid) {
        int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            dup2(fd, 2);
            close(fd);
        }
        execv(av[0], av);
        _exit(127);
    }
    double deadline = now_ms() + timeout_ms;
    int st = 0;
    for (;;) {
        pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) break;
        if (r < 0 && errno != EINTR) {
            snprintf(why, cap, "waitpid failed: %s", strerror(errno));
            kill(pid, SIGKILL);
            waitpid(pid, &st, 0);
            return -1;
        }
        if (now_ms() >= deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &st, 0);
            snprintf(why, cap, "speech synthesis timed out after %d ms", timeout_ms);
            return -1;
        }
        usleep(20000);
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        char tail[513] = "";
        tail_log(log, tail, sizeof tail);
        if (WIFEXITED(st))
            snprintf(why, cap, "speech engine failed (exit %d): %s", WEXITSTATUS(st), tail);
        else
            snprintf(why, cap, "speech engine killed by signal %d: %s", WTERMSIG(st), tail);
        return -1;
    }
    (void)cfg;
    return 0;
}

// 24 000 Hz, mono, signed 16-bit little-endian, RIFF/WAVE.
static int wav_check(const unsigned char *p, size_t n, char *why, size_t cap) {
    if (n < 44 || memcmp(p, "RIFF", 4) || memcmp(p + 8, "WAVE", 4)) {
        snprintf(why, cap, "speech engine did not produce a RIFF/WAVE file");
        return -1;
    }
    size_t off = 12;
    while (off + 8 <= n) {
        uint32_t sz = rd32(p + off + 4);
        if (!memcmp(p + off, "fmt ", 4)) {
            if (sz < 16 || off + 24 > n) {
                snprintf(why, cap, "WAV fmt chunk truncated");
                return -1;
            }
            unsigned ch = rd16(p + off + 10), rate = rd32(p + off + 12), bits = rd16(p + off + 22);
            if (ch != 1 || rate != 24000 || bits != 16) {
                snprintf(why, cap, "expected 24000 Hz mono PCM16, got %u Hz %u ch %u bit", rate, ch, bits);
                return -1;
            }
            return 0;
        }
        if (sz > n - off - 8) break;
        off += 8 + (size_t)sz + (sz & 1u);
    }
    snprintf(why, cap, "WAV has no fmt chunk");
    return -1;
}

static void fail(const speak_sink *k, int status, const char *type, const char *msg) {
    sb e = {0};
    sb_add(&e, "{\"error\":{\"message\":");
    json_quote(&e, msg ? msg : "speak failed");
    sb_fmt(&e, ",\"type\":\"%s\",\"param\":null,\"code\":%d}}", type, status);
    k->head(k->ctx, status, "application/json; charset=utf-8", e.n, NULL);
    k->send(k->ctx, e.p, e.n);
    free(e.p);
}

int speak_route_handle(const speak_cfg *cfg, const speak_sink *sink,
                       const char *body, size_t body_len) {
    if (!cfg || !sink || !sink->send || !sink->head || !cfg->argv || !cfg->argv[0]) {
        if (sink && sink->send && sink->head)
            fail(sink, 500, "server_error", "speak route is not configured");
        return -1;
    }
    const char *voice = NULL, *style = NULL, *err = NULL;
    if (speak_route_parse(body, body_len, &voice, &style, &err) < 0) {
        fail(sink, 400, "invalid_request_error", err);
        return 0;
    }
    jv *jb = json_parse(body, NULL);
    const jv *jt = jb ? json_get(jb, "text") : NULL;
    if (!jt || jt->type != J_STR) {
        json_free(jb);
        fail(sink, 400, "invalid_request_error", "text must be a string");
        return 0;
    }
    const char *text = jt->str;

    const char *td = cfg->tmpdir;
    char tdbuf[1024];
    if (!td || !*td) td = getenv("TMPDIR");
    if (!td || !*td) td = "/tmp";
    snprintf(tdbuf, sizeof tdbuf, "%s", td);
    size_t tl = strlen(tdbuf);
    while (tl > 1 && tdbuf[tl - 1] == '/') tdbuf[--tl] = 0;
    td = tdbuf;

    char out[1200] = "", drive[1200] = "", log[1200] = "";
    if (mkpath(out, sizeof out, td, "wav") || mkpath(log, sizeof log, td, "log")) {
        json_free(jb);
        fail(sink, 500, "server_error", "cannot create scratch file");
        return 0;
    }
    const char *drv = style_drive(style);
    if (drv) {
        if (mkpath(drive, sizeof drive, td, "json")) {
            unlink(out);
            unlink(log);
            json_free(jb);
            fail(sink, 500, "server_error", "cannot create scratch file");
            return 0;
        }
        int fd = open(drive, O_WRONLY | O_TRUNC);
        size_t dn = strlen(drv);
        if (fd < 0 || write(fd, drv, dn) != (ssize_t)dn) {
            if (fd >= 0) close(fd);
            unlink(out), unlink(drive), unlink(log);
            json_free(jb);
            fail(sink, 500, "server_error", "cannot write drive preset");
            return 0;
        }
        close(fd);
    }

    char *av[24];
    int an = 0;
    for (int i = 0; cfg->argv[i] && an < 12; i++) av[an++] = (char *)cfg->argv[i];
    av[an++] = (char *)"--text";
    av[an++] = (char *)text;
    av[an++] = (char *)"--out";
    av[an++] = out;
    av[an++] = (char *)"--voice";
    av[an++] = (char *)voice;
    if (drv) {
        av[an++] = (char *)"--drive";
        av[an++] = drive;
    }
    if (cfg->cpu) av[an++] = (char *)"--cpu";
    av[an] = NULL;

    int timeout = cfg->timeout_ms > 0 ? cfg->timeout_ms : DEFAULT_TIMEOUT_MS;
    pthread_mutex_lock(&speak_lock);
    double t0 = now_ms();
    char why[1024] = "";
    int bad = run_child(cfg, av, log, timeout, why, sizeof why);
    double ms = now_ms() - t0;

    unsigned char *wav = NULL;
    size_t wn = 0;
    if (!bad) {
        int fd = open(out, O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) || st.st_size <= 44 || (uint64_t)st.st_size > MAX_WAV) {
            if (fd >= 0) close(fd);
            bad = -1;
            snprintf(why, sizeof why, "speech engine produced no usable WAV");
        } else {
            wn = (size_t)st.st_size;
            wav = malloc(wn);
            size_t got = 0;
            while (wav && got < wn) {
                ssize_t r = read(fd, wav + got, wn - got);
                if (r <= 0) break;
                got += (size_t)r;
            }
            close(fd);
            if (!wav || got != wn) {
                bad = -1;
                snprintf(why, sizeof why, "cannot read the synthesized WAV");
            } else if (wav_check(wav, wn, why, sizeof why)) {
                bad = -1;
            }
        }
    }

    unlink(out);
    unlink(log);
    if (drive[0]) unlink(drive);
    {
        char sidecar[1210];
        snprintf(sidecar, sizeof sidecar, "%s", out);
        size_t sl = strlen(sidecar);
        if (sl > 4 && sl + 2 < sizeof sidecar && !strcmp(sidecar + sl - 4, ".wav")) {
            memcpy(sidecar + sl - 4, ".json", 6);
            unlink(sidecar);
        }
    }
    pthread_mutex_unlock(&speak_lock);

    if (bad) {
        free(wav);
        json_free(jb);
        fail(sink, 503, "unavailable_error", why[0] ? why : "speech synthesis failed");
        return 0;
    }

    char xh[512];
    snprintf(xh, sizeof xh,
             "X-LLB-Sample-Rate: 24000\r\nX-LLB-Voice: %s\r\nX-LLB-Style: %s\r\n"
             "X-LLB-G2P: misaki\r\nX-LLB-Speak-Ms: %.1f\r\n",
             voice, style, ms);
    sink->head(sink->ctx, 200, "audio/wav", wn, xh);
    int rc = sink->send(sink->ctx, wav, wn);
    free(wav);
    json_free(jb);
    return rc ? -1 : 0;
}
