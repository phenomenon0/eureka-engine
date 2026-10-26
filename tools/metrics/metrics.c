// metrics.c: W1-METRICS telemetry (see metrics.h). Prometheus text by hand,
// fixed-bucket histograms, a sliding window per latency for exact percentiles,
// and an optional standalone /metrics listener thread.
#define _POSIX_C_SOURCE 200809L
#include "metrics.h"
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

double llb_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---- percentile ---------------------------------------------------------------

double llb_percentile(const double *s, size_t n, double q) {
    if (!n) return NAN;
    if (q <= 0) return s[0];
    if (q >= 1) return s[n - 1];
    double h = q * (double)(n - 1);
    size_t lo = (size_t)floor(h);
    if (lo + 1 >= n) return s[n - 1];
    return s[lo] + (h - (double)lo) * (s[lo + 1] - s[lo]);
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// ---- histogram + window -------------------------------------------------------

#define MAXB 16
#define WIN 1024

typedef struct {
    const char *name, *help;
    int nb;
    double le[MAXB];          // upper bounds, ascending; +Inf implicit
    unsigned long long c[MAXB + 1];
    unsigned long long count;
    double sum;
} hist;

typedef struct {
    double v[WIN];
    size_t n, head;           // n = filled (<= WIN), head = next write slot
} window;

static void hist_obs(hist *h, double v) {
    int i = 0;
    while (i < h->nb && v > h->le[i]) i++;
    h->c[i]++;
    h->count++;
    h->sum += v;
}

static void win_obs(window *w, double v) {
    w->v[w->head] = v;
    w->head = (w->head + 1) % WIN;
    if (w->n < WIN) w->n++;
}

// ---- global state -------------------------------------------------------------

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static double t_boot;

enum { H_TTFT, H_ITL, H_E2E, H_QUEUE, H_PREFILL, H_NH };
static hist H[H_NH] = {
    {"llb_time_to_first_token_seconds", "Request arrival to first generated token.", 11,
     {.025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60}, {0}, 0, 0},
    {"llb_inter_token_latency_seconds", "Gap between consecutive generated tokens of one request.", 14,
     {.005, .01, .015, .02, .025, .03, .04, .05, .075, .1, .15, .25, .5, 1}, {0}, 0, 0},
    {"llb_request_duration_seconds", "Request arrival to response written.", 12,
     {.05, .1, .25, .5, 1, 2.5, 5, 10, 20, 30, 60, 120}, {0}, 0, 0},
    {"llb_queue_time_seconds", "Request arrival to engine pass start.", 8,
     {.0005, .001, .005, .01, .05, .1, 1, 10}, {0}, 0, 0},
    {"llb_prefill_seconds", "Engine pass start to first generated token (prefill + first sample).", 11,
     {.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30}, {0}, 0, 0},
};

enum { W_TTFT, W_ITL, W_E2E, W_PREFILL_TPS, W_DECODE_TPS, W_N };
static const char *wname[W_N] = {"ttft_seconds", "itl_seconds", "e2e_seconds",
                                 "prefill_tokens_per_second", "decode_tokens_per_second"};
static window Wn[W_N];

#define MAXLAB 24
static struct { char mode[24], outcome[16]; unsigned long long n; } reqs[MAXLAB];
static int nreqs;

static unsigned long long prompt_tok, prefill_tok, cached_tok, gen_tok, decode_steps;
static double prefill_sec, decode_sec;
static unsigned long long cache_hit, cache_partial, cache_miss;
static unsigned long long spec_drafted, spec_accepted;
static double last_prefill_tps, last_decode_tps;
static int queue_depth, in_flight;

static void boot(void) {
    if (!t_boot) t_boot = llb_now();
}

void llb_metrics_reset(void) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < H_NH; i++) {
        memset(H[i].c, 0, sizeof H[i].c);
        H[i].count = 0;
        H[i].sum = 0;
    }
    memset(Wn, 0, sizeof Wn);
    memset(reqs, 0, sizeof reqs);
    nreqs = 0;
    prompt_tok = prefill_tok = cached_tok = gen_tok = decode_steps = 0;
    prefill_sec = decode_sec = 0;
    cache_hit = cache_partial = cache_miss = 0;
    spec_drafted = spec_accepted = 0;
    last_prefill_tps = last_decode_tps = 0;
    queue_depth = in_flight = 0;
    t_boot = llb_now();
    pthread_mutex_unlock(&mu);
}

// ---- per-request --------------------------------------------------------------

void llb_req_begin(llb_req *r) {
    memset(r, 0, sizeof *r);
    r->t_arrive = llb_now();
    pthread_mutex_lock(&mu);
    boot();
    in_flight++;
    pthread_mutex_unlock(&mu);
}

void llb_req_start(llb_req *r, int prompt_n, int cached_n) {
    r->t_start = llb_now();
    r->prompt_n = prompt_n;
    r->cached_n = cached_n;
}

void llb_req_token_at(llb_req *r, double t) {
    if (r->gen_n) {
        pthread_mutex_lock(&mu);
        hist_obs(&H[H_ITL], t - r->t_last);
        win_obs(&Wn[W_ITL], t - r->t_last);
        pthread_mutex_unlock(&mu);
    }
    if (!r->gen_n) r->t_first = t;
    r->t_last = t;
    r->gen_n++;
}

void llb_req_token(llb_req *r) { llb_req_token_at(r, llb_now()); }

void llb_req_spec(llb_req *r, int drafted, int accepted) {
    r->drafted += drafted;
    r->accepted += accepted;
}

// derived numbers shared by the aggregates and the timings block
typedef struct {
    double queue, prompt, predicted, ttft, total;
    int prefill_n, steps;
} derived;

static derived derive(const llb_req *r) {
    derived d = {0};
    double end = r->t_end > 0 ? r->t_end : llb_now();
    d.total = end - r->t_arrive;
    d.queue = r->t_start > 0 ? r->t_start - r->t_arrive : d.total;
    d.prefill_n = r->prompt_n - r->cached_n;
    if (d.prefill_n < 0) d.prefill_n = 0;
    if (r->gen_n) {
        d.ttft = r->t_first - r->t_arrive;
        d.prompt = r->t_start > 0 ? r->t_first - r->t_start : 0;
        d.predicted = r->t_last - r->t_first;
        d.steps = r->gen_n - 1;   // decode steps after the first (prefill-sampled) token
    } else if (r->t_start > 0) {
        d.prompt = end - r->t_start;   // no tokens (e.g. a decide pass): the whole pass
    }
    return d;
}

static void count_req(const char *mode, const char *outcome) {
    for (int i = 0; i < nreqs; i++)
        if (!strcmp(reqs[i].mode, mode) && !strcmp(reqs[i].outcome, outcome)) {
            reqs[i].n++;
            return;
        }
    if (nreqs == MAXLAB) return;
    snprintf(reqs[nreqs].mode, sizeof reqs[nreqs].mode, "%s", mode);
    snprintf(reqs[nreqs].outcome, sizeof reqs[nreqs].outcome, "%s", outcome);
    reqs[nreqs++].n = 1;
}

void llb_req_end(llb_req *r, const char *mode, const char *outcome) {
    r->t_end = llb_now();
    derived d = derive(r);
    pthread_mutex_lock(&mu);
    boot();
    if (in_flight > 0) in_flight--;
    count_req(mode ? mode : "unknown", outcome ? outcome : "ok");
    hist_obs(&H[H_E2E], d.total);
    win_obs(&Wn[W_E2E], d.total);
    if (r->t_start > 0) hist_obs(&H[H_QUEUE], d.queue);
    prompt_tok += (unsigned long long)r->prompt_n;
    cached_tok += (unsigned long long)(r->cached_n < 0 ? 0 : r->cached_n);
    if (r->prompt_n > 0) {
        if (r->cached_n >= r->prompt_n) cache_hit++;
        else if (r->cached_n > 0) cache_partial++;
        else cache_miss++;
    }
    spec_drafted += (unsigned long long)r->drafted;
    spec_accepted += (unsigned long long)r->accepted;
    if (r->gen_n) {
        gen_tok += (unsigned long long)r->gen_n;
        hist_obs(&H[H_TTFT], d.ttft);
        win_obs(&Wn[W_TTFT], d.ttft);
        if (r->t_start > 0) {
            hist_obs(&H[H_PREFILL], d.prompt);
            prefill_tok += (unsigned long long)d.prefill_n;
            prefill_sec += d.prompt;
            if (d.prompt > 0 && d.prefill_n > 0) {
                last_prefill_tps = d.prefill_n / d.prompt;
                win_obs(&Wn[W_PREFILL_TPS], last_prefill_tps);
            }
        }
        decode_steps += (unsigned long long)d.steps;
        decode_sec += d.predicted;
        if (d.steps > 0 && d.predicted > 0) {
            last_decode_tps = d.steps / d.predicted;
            win_obs(&Wn[W_DECODE_TPS], last_decode_tps);
        }
    }
    pthread_mutex_unlock(&mu);
}

int llb_req_timings_json(const llb_req *r, char *buf, size_t cap) {
    derived d = derive(r);
    double pms = d.prompt * 1e3, gms = d.predicted * 1e3;
    return snprintf(buf, cap,
        "{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.3f,\"prompt_per_token_ms\":%.3f,\"prompt_per_second\":%.3f,"
        "\"predicted_n\":%d,\"predicted_ms\":%.3f,\"predicted_per_token_ms\":%.3f,\"predicted_per_second\":%.3f,"
        "\"ttft_ms\":%.3f,\"queue_ms\":%.3f,\"total_ms\":%.3f,\"draft_n\":%d,\"draft_n_accepted\":%d}",
        r->cached_n, d.prefill_n, pms, d.prefill_n ? pms / d.prefill_n : 0, pms > 0 ? d.prefill_n / d.prompt : 0,
        r->gen_n, gms, d.steps ? gms / d.steps : 0, gms > 0 ? d.steps / d.predicted : 0,
        d.ttft * 1e3, d.queue * 1e3, d.total * 1e3, r->drafted, r->accepted);
}

void llb_metrics_queue_depth(int depth) {
    pthread_mutex_lock(&mu);
    queue_depth = depth;
    pthread_mutex_unlock(&mu);
}

void llb_metrics_cache(int hit) {
    pthread_mutex_lock(&mu);
    if (hit) cache_hit++;
    else cache_miss++;
    pthread_mutex_unlock(&mu);
}

// ---- Prometheus text ----------------------------------------------------------

typedef struct { char *p; size_t n, cap; } buf;

static void put(buf *b, const char *fmt, ...) {
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int k = vsnprintf(b->p ? b->p + b->n : NULL, b->p ? b->cap - b->n : 0, fmt, ap);
        va_end(ap);
        if (k < 0) return;
        if (b->p && b->n + (size_t)k < b->cap) {
            b->n += (size_t)k;
            return;
        }
        size_t nc = b->cap ? b->cap * 2 : 8192;
        while (nc < b->n + (size_t)k + 1) nc *= 2;
        char *np = realloc(b->p, nc);
        if (!np) return;
        b->p = np;
        b->cap = nc;
    }
}

// %g-style number in Prometheus syntax (+Inf / NaN spelled out)
static const char *num(char *s, double v) {
    if (isnan(v)) return "NaN";
    if (isinf(v)) return v > 0 ? "+Inf" : "-Inf";
    snprintf(s, 32, "%.9g", v);
    return s;
}

static void meta(buf *b, const char *name, const char *type, const char *help) {
    put(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, type);
}

static void counter(buf *b, const char *name, const char *help, double v) {
    char s[32];
    meta(b, name, "counter", help);
    put(b, "%s %s\n", name, num(s, v));
}

static void gauge(buf *b, const char *name, const char *help, double v) {
    char s[32];
    meta(b, name, "gauge", help);
    put(b, "%s %s\n", name, num(s, v));
}

static void hist_render(buf *b, const hist *h) {
    char s[32], t[32];
    meta(b, h->name, "histogram", h->help);
    unsigned long long cum = 0;
    for (int i = 0; i < h->nb; i++) {
        cum += h->c[i];
        put(b, "%s_bucket{le=\"%s\"} %llu\n", h->name, num(s, h->le[i]), cum);
    }
    cum += h->c[h->nb];
    put(b, "%s_bucket{le=\"+Inf\"} %llu\n%s_sum %s\n%s_count %llu\n", h->name, cum, h->name, num(t, h->sum), h->name, h->count);
}

char *llb_metrics_render(size_t *len) {
    buf b = {0};
    char s[32];
    pthread_mutex_lock(&mu);
    boot();
    meta(&b, "llb_requests_total", "counter", "Requests finished, by mode and outcome (ok|retained|refused|error).");
    for (int i = 0; i < nreqs; i++)
        put(&b, "llb_requests_total{mode=\"%s\",outcome=\"%s\"} %llu\n", reqs[i].mode, reqs[i].outcome, reqs[i].n);
    gauge(&b, "llb_requests_in_flight", "Requests accepted and not yet finished.", in_flight);
    gauge(&b, "llb_queue_depth", "Requests waiting for the engine.", queue_depth);
    counter(&b, "llb_prompt_tokens_total", "Prompt tokens of finished requests (cached + prefilled).", (double)prompt_tok);
    counter(&b, "llb_prefill_tokens_total", "Prompt tokens actually prefilled by the engine.", (double)prefill_tok);
    counter(&b, "llb_prefill_seconds_total", "Engine prefill time (pass start to first token).", prefill_sec);
    counter(&b, "llb_generated_tokens_total", "Generated tokens emitted.", (double)gen_tok);
    counter(&b, "llb_decode_steps_total", "Decode steps after each request's first token.", (double)decode_steps);
    counter(&b, "llb_decode_seconds_total", "Decode time (first to last token).", decode_sec);
    gauge(&b, "llb_prefill_tokens_per_second", "Prefill throughput of the last request that prefilled.", last_prefill_tps);
    gauge(&b, "llb_decode_tokens_per_second", "Decode throughput of the last request with >1 token.", last_decode_tps);
    meta(&b, "llb_cache_requests_total", "counter", "Prompt cache lookups: hit = whole prompt cached, partial = prefix cached, miss = none.");
    put(&b, "llb_cache_requests_total{result=\"hit\"} %llu\n", cache_hit);
    put(&b, "llb_cache_requests_total{result=\"partial\"} %llu\n", cache_partial);
    put(&b, "llb_cache_requests_total{result=\"miss\"} %llu\n", cache_miss);
    counter(&b, "llb_cache_tokens_total", "Prompt tokens served from cache (not prefilled).", (double)cached_tok);
    counter(&b, "llb_spec_drafted_tokens_total", "Speculative tokens drafted.", (double)spec_drafted);
    counter(&b, "llb_spec_accepted_tokens_total", "Speculative tokens accepted.", (double)spec_accepted);
    gauge(&b, "llb_spec_acceptance_ratio", "Accepted / drafted since start (NaN when nothing drafted).",
          spec_drafted ? (double)spec_accepted / (double)spec_drafted : NAN);
    for (int i = 0; i < H_NH; i++) hist_render(&b, &H[i]);
    meta(&b, "llb_latency_window", "gauge", "Exact quantiles over the last 1024 observations of each metric.");
    static const double qs[] = {.5, .9, .95, .99};
    double *tmp = malloc(sizeof(double) * WIN);
    for (int w = 0; w < W_N && tmp; w++) {
        memcpy(tmp, Wn[w].v, sizeof(double) * Wn[w].n);
        qsort(tmp, Wn[w].n, sizeof(double), cmp_d);
        for (size_t k = 0; k < sizeof qs / sizeof *qs; k++)
            put(&b, "llb_latency_window{metric=\"%s\",quantile=\"%g\"} %s\n", wname[w], qs[k],
                num(s, llb_percentile(tmp, Wn[w].n, qs[k])));
        put(&b, "llb_latency_window{metric=\"%s\",quantile=\"count\"} %zu\n", wname[w], Wn[w].n);
    }
    free(tmp);
    gauge(&b, "llb_uptime_seconds", "Seconds since the metrics module started.", llb_now() - t_boot);
    pthread_mutex_unlock(&mu);
    if (len) *len = b.n;
    return b.p;
}

// ---- standalone listener ------------------------------------------------------

static void serve(int c) {
    char req[2048];
    ssize_t k = recv(c, req, sizeof req - 1, 0);
    if (k <= 0) return;
    req[k] = 0;
    const char *ctype = "text/plain; charset=utf-8", *status = "200 OK";
    char *body = NULL;
    size_t blen = 0;
    if (!strncmp(req, "GET /metrics", 12) && (req[12] == ' ' || req[12] == '?')) {
        body = llb_metrics_render(&blen);
        ctype = "text/plain; version=0.0.4; charset=utf-8";
    } else if (!strncmp(req, "GET /health ", 12)) {
        body = strdup("{\"status\":\"ok\"}\n");
        blen = strlen(body);
        ctype = "application/json";
    } else {
        status = "404 Not Found";
        body = strdup("not found\n");
        blen = strlen(body);
    }
    char head[256];
    int h = snprintf(head, sizeof head, "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                     status, ctype, blen);
    if (send(c, head, (size_t)h, MSG_NOSIGNAL) == h && body)
        for (size_t off = 0; off < blen;) {
            ssize_t w = send(c, body + off, blen - off, MSG_NOSIGNAL);
            if (w <= 0) break;
            off += (size_t)w;
        }
    free(body);
}

static void *listen_loop(void *arg) {
    int s = (int)(long)arg;
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR) continue;
            break;
        }
        serve(c);
        close(c);
    }
    return NULL;
}

int llb_metrics_listen(const char *host, int port) {
    pthread_mutex_lock(&mu);
    boot();
    pthread_mutex_unlock(&mu);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host && *host ? host : "127.0.0.1", &a.sin_addr) != 1) {
        errno = EINVAL;
        return -1;
    }
    int s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 16)) {
        int e = errno;
        close(s);
        errno = e;
        return -1;
    }
    pthread_t t;
    if (pthread_create(&t, NULL, listen_loop, (void *)(long)s)) {
        close(s);
        return -1;
    }
    pthread_detach(t);
    return 0;
}
