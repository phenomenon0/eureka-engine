// serve.c: W1-SERVER — the OpenAI-compatible HTTP front of the SYS1-DUAL engine
// (docs: SERVER.md in the W1-SERVER run dir). One process that owns ONE engine
// child (`LLB_SYS1=1 llb-engine ...`, the stdin/stdout JSON-line protocol of
// dual.c) and turns HTTP requests into its generate requests. No engine code
// here: tokenize/prefill/decode/detokenize all stay in the engine; this file is
// sockets, HTTP/1.1 framing, the chat template, OpenAI JSON shapes and SSE.
//   GET  /health                      200 {"status":"ok"} | 503 while loading / engine gone
//   GET  /v1/models                   the one served model
//   POST /v1/completions              text completion (stream or not)
//   POST /v1/chat/completions         chat completion (Qwen3.5 ChatML template)
//   POST /v1/decide                   one clef decide request (state + questions -> the
//                                     clef head's per-question probabilities/choice/score)
//   POST /v1/hear                     one audio clip (base64 WAV) -> transcript (the served
//                                     ear: whisper-server sidecar, whisper.cpp Metal kernels)
//   POST /v1/see                      one image -> D-FINE-n COCO-80 boxes (the senses
//                                     detector the app ships; shells see/detect_cli.py)
// One request runs on the engine at a time (single slot, FIFO by the engine
// lock); the others wait. Streaming = dual.c's LLB_SYS1_STREAM token events.
// G6 (--parallel N, default 1): N slots = N engine children, each its own
// weights + KV (no shared-weight batching yet; see G6 DESIGN.md). A request
// takes a FIFO ticket and gets the first free slot when its ticket is at the
// head of the queue (strict arrival order); among free slots it prefers the one
// whose last prompt shares the longest prefix (prefix-cache affinity), then the
// least recently used. N=1 is the single-slot server exactly.
// Sampling fields (temperature, top_p, seed, stop, ...) are ACCEPTED and passed
// through to the engine request under "sampling" untouched; the engine is
// greedy until the W1-SAMPLER lane wires them (X-LLB-Sampler: greedy).
// Build: tools/sys1dual/build-serve.sh (links tools/clef3/json.c only).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "json.h"
#include "speak_route.h" // M2: POST /v1/speak (separate translation unit)
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_HEAD (64 * 1024)
#define MAX_BODY (8 * 1024 * 1024)

static const char *host = "127.0.0.1", *alias, *api_key;
static int canon = 0;
static int think_default = 1; // --reasoning off: the template default is non-thinking (llama-server --reasoning off)
static int port = 8080, chat_default_n = 256, drain_s = 120;
// M2 /v1/speak: the P3 launcher (a script that pins the venv + Misaki
// frontend) and its deadline. CPU-only by default: the SwiftShader path
// takes no GPU lock. --speak-gpu opts into the flocked WebGPU path.
static const char *speak_cmd = SPEAK_P3_CMD_DEFAULT;
static int speak_cpu = 1, speak_timeout_ms = 180000;
// G6: one slot = one engine child and its pipes. busy/n_req/t_last/last are
// guarded by q_lock; ready/dead are set by the slot's stderr reader and the
// reaper.
typedef struct {
    int idx, busy;
    pid_t pid;
    FILE *in, *out;
    atomic_int ready, dead;
    long n_req;
    int status; // wait status once reaped (pid < 0)
    double t_last;
    char *last; // the last prompt text sent to this slot (affinity)
} Slot;
static Slot *slots;
static int n_slots = 1;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cond = PTHREAD_COND_INITIALIZER;
static unsigned long q_tail, q_head; // FIFO tickets: next to issue, next to serve
static atomic_int ready, engine_dead, inflight, queued, stopping;
static atomic_long req_seq;
static time_t t_start;
static int wake[2];

// hear: one whisper-server sidecar (whisper.cpp) owns the ear. It is a separate
// process on loopback; /v1/hear serializes on hear_lock because the sidecar
// transcribes one clip at a time. The LLM slots are untouched by this verb.
static const char *whisper_bin, *whisper_model, *hear_model = "whisper-base.en";
static int whisper_port = 8137;
static pid_t whisper_pid = -1;
static atomic_int whisper_ready;
static pthread_mutex_t hear_lock = PTHREAD_MUTEX_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void logf_(const char *fmt, ...) {
    char ts[32];
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    struct tm tm;
    localtime_r(&t.tv_sec, &tm);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tm);
    va_list a;
    va_start(a, fmt);
    flockfile(stderr);
    fprintf(stderr, "%s.%03ld serve: ", ts, t.tv_nsec / 1000000);
    vfprintf(stderr, fmt, a);
    fputc('\n', stderr);
    funlockfile(stderr);
    va_end(a);
}

// ---- engine child ----------------------------------------------------------

typedef struct {
    Slot *s;
    FILE *f;
} ErrArg;

static void *engine_err(void *p) {
    ErrArg a = *(ErrArg *)p;
    free(p);
    char *line = NULL, tag[32] = "engine: ";
    if (n_slots > 1) snprintf(tag, sizeof tag, "engine[%d]: ", a.s->idx);
    size_t cap = 0;
    while (getline(&line, &cap, a.f) > 0) {
        // the engine is up once dual.c's init line appears (after the model load)
        if (strstr(line, "tokenizer ready")) {
            a.s->ready = 1;
            int all = 1;
            for (int i = 0; i < n_slots; i++) all &= slots[i].ready || slots[i].dead;
            if (all) ready = 1;
            pthread_mutex_lock(&q_lock);
            pthread_cond_broadcast(&q_cond);
            pthread_mutex_unlock(&q_lock);
        }
        flockfile(stderr);
        fputs(tag, stderr);
        fputs(line, stderr);
        funlockfile(stderr);
    }
    free(line);
    return NULL;
}

static void engine_start(Slot *s, char **argv) {
    int in[2], out[2], err[2];
    if (pipe(in) || pipe(out) || pipe(err)) perror("pipe"), exit(1);
    pid_t engine_pid = s->pid = fork();
    if (engine_pid < 0) perror("fork"), exit(1);
    if (!engine_pid) {
        // children must not hold the other slots' pipe ends (EOF on shutdown)
        for (int i = 0; i < s->idx; i++) fclose(slots[i].in), fclose(slots[i].out);
        dup2(in[0], 0), dup2(out[1], 1), dup2(err[1], 2);
        close(in[0]), close(in[1]), close(out[0]), close(out[1]), close(err[0]), close(err[1]);
        setenv("LLB_SYS1", "1", 1);
        setenv("LLB_SYS1_STREAM", "1", 1);
        // Owner directive: FAST is the product default (chunkf prefill, fast GEMV); the exact
        // per-token / bit-exact tier is explicit opt-in only. --canon forces the CANON tier over
        // any inherited env; without it an explicit env value (LLB_DUAL_FASTPF=0 / LLB_FASTGEMV=0)
        // still wins, so the env opt-in keeps working.
        setenv("LLB_DUAL_FASTPF", canon ? "0" : "512", canon ? 1 : 0);
        setenv("LLB_FASTGEMV", canon ? "0" : "1", canon ? 1 : 0);
        // dual.c / sys1.c refuse these (exact pass only)
        unsetenv("LLB_PREFILL"), unsetenv("LLB_SPEC"), unsetenv("LLB_MTP"), unsetenv("LLB_DUMP_HIDDEN");
        execv(argv[0], argv);
        perror("execv engine");
        _exit(127);
    }
    close(in[0]), close(out[1]), close(err[1]);
    s->in = fdopen(in[1], "w");
    s->out = fdopen(out[0], "r");
    ErrArg *a = malloc(sizeof *a);
    *a = (ErrArg){s, fdopen(err[0], "r")};
    pthread_t t;
    pthread_create(&t, NULL, engine_err, a);
    pthread_detach(t);
}

static int all_dead(void) {
    for (int i = 0; i < n_slots; i++)
        if (!slots[i].dead) return 0;
    return 1;
}

// G6: the FIFO slot queue. Returns a busy slot, or NULL (every engine is dead).
static size_t lcp(const char *a, const char *b) {
    size_t n = 0;
    if (a && b)
        while (a[n] && a[n] == b[n]) n++;
    return n;
}

static Slot *slot_acquire(const char *prompt, unsigned long *ticket) {
    pthread_mutex_lock(&q_lock);
    unsigned long me = *ticket = q_tail++;
    Slot *s = NULL;
    for (;;) {
        if (all_dead()) break;
        if (me == q_head) {
            long best = -1;
            for (int i = 0; i < n_slots; i++) {
                Slot *x = &slots[i];
                if (x->busy || x->dead || !x->ready) continue;
                long k = (long)lcp(x->last, prompt);
                if (!s || k > best || (k == best && x->t_last < s->t_last)) s = x, best = k;
            }
            if (s) break;
        }
        pthread_cond_wait(&q_cond, &q_lock);
    }
    q_head++; // also when giving up, so the tickets behind stay live
    if (s) s->busy = 1;
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_lock);
    return s;
}

static void slot_release(Slot *s, const char *prompt) {
    pthread_mutex_lock(&q_lock);
    s->busy = 0;
    s->n_req++;
    s->t_last = now();
    free(s->last);
    s->last = strdup(prompt);
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_lock);
}

// ---- sockets / HTTP --------------------------------------------------------

static int send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            return -1;
        }
        p += w, n -= (size_t)w;
    }
    return 0;
}

typedef struct {
    int fd, status, stream, prompt_n, gen_n;
    long bytes, id;
    char ip[64], method[16], path[256];
    int broken; // client went away mid-stream (the engine is still drained)
    int slot;   // G6: the slot that ran it (-1 = none)
    int authed;
    char xh[128]; // G6: extra response headers (N>1 only: slot, ticket, queue wait)
} Conn;

static const char *reason(int s) {
    switch (s) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

static void respond(Conn *c, int status, const char *ctype, const char *body, const char *extra) {
    char h[512];
    size_t n = strlen(body);
    int k = snprintf(h, sizeof h,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\nServer: llb-serve\r\n%s%s\r\n",
        status, reason(status), ctype, n, extra ? extra : "", c->xh);
    c->status = status;
    if (send_all(c->fd, h, (size_t)k) || send_all(c->fd, body, n)) c->broken = 1;
    c->bytes += k + (long)n;
}

static void json_reply(Conn *c, int status, const char *body) {
    respond(c, status, "application/json; charset=utf-8", body, NULL);
}

// OpenAI error shape
static void err_reply(Conn *c, int status, const char *type, const char *msg) {
    sb e = {0};
    sb_add(&e, "{\"error\":{\"message\":");
    json_quote(&e, msg);
    sb_fmt(&e, ",\"type\":\"%s\",\"param\":null,\"code\":%d}}", type, status);
    json_reply(c, status, e.p);
    free(e.p);
}

static void sse_open(Conn *c) {
    char h[512];
    snprintf(h, sizeof h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    "Connection: close\r\nX-Accel-Buffering: no\r\nX-LLB-Sampler: greedy\r\nServer: llb-serve\r\n%s\r\n", c->xh);
    c->status = 200;
    if (send_all(c->fd, h, strlen(h))) c->broken = 1;
    c->bytes += (long)strlen(h);
}

static void sse(Conn *c, const char *data) {
    if (c->broken) return;
    sb e = {0};
    sb_add(&e, "data: ");
    sb_add(&e, data);
    sb_add(&e, "\n\n");
    if (send_all(c->fd, e.p, e.n)) c->broken = 1;
    c->bytes += (long)e.n;
    free(e.p);
}

// ---- requests --------------------------------------------------------------

// the OpenAI/llama.cpp sampling knobs: accepted and passed through verbatim
static const char *sampling_keys[] = {
    "temperature", "top_p", "top_k", "min_p", "seed", "stop", "presence_penalty", "frequency_penalty",
    "repeat_penalty", "repetition_penalty", "logit_bias", "logprobs", "top_logprobs", "n_probs", "typical_p", NULL};

typedef struct {
    int chat, stream, n, usage_chunk, ret_tokens, special;
    char *prompt; // rendered text sent to the engine
    char *ids;    // /v1/completions with a token-id prompt: the ids as "1,2,3" (prompt then holds the same, for slot affinity)
} Req;

static char *trim(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    return strndup(s, n);
}

// a message content: a string or [{"type":"text","text":...}] parts
static const char *content_text(const jv *m, sb *tmp, const char **err) {
    const jv *c = json_get(m, "content");
    if (!c || c->type == J_NULL) return "";
    if (c->type == J_STR) return c->str;
    if (c->type != J_ARR) return *err = "message content must be a string or an array of text parts", NULL;
    for (size_t i = 0; i < c->n; i++) {
        const jv *t = json_get(c->items[i], "text");
        const jv *ty = json_get(c->items[i], "type");
        if (!t || t->type != J_STR || (ty && (ty->type != J_STR || strcmp(ty->str, "text"))))
            return *err = "only text content parts are supported (no images/audio on this server yet)", NULL;
        sb_add(tmp, t->str);
    }
    return tmp->p ? tmp->p : "";
}

// the model's Qwen3.5 chat template (flash-chat_template.jinja), the no-tools subset
static const char *render_chat(const jv *body, sb *o) {
    const jv *ms = json_get(body, "messages");
    if (!ms || ms->type != J_ARR || !ms->n) return "messages must be a non-empty array";
    if (json_get(body, "tools") || json_get(body, "functions")) return "tools/function calling is not supported yet";
    int thinking = think_default;
    const jv *kw = json_get(body, "chat_template_kwargs");
    const jv *et = kw ? json_get(kw, "enable_thinking") : NULL;
    if (et && et->type == J_BOOL) thinking = et->b;
    long last_user = -1;
    for (size_t i = 0; i < ms->n; i++) {
        const jv *r = json_get(ms->items[i], "role");
        if (!r || r->type != J_STR) return "every message needs a string role";
        if (!strcmp(r->str, "user")) last_user = (long)i;
        else if (strcmp(r->str, "system") && strcmp(r->str, "assistant"))
            return "role must be system, user or assistant (tool messages are not supported yet)";
        if (!strcmp(r->str, "system") && i) return "System message must be at the beginning.";
    }
    if (last_user < 0) return "No user query found in messages.";
    for (size_t i = 0; i < ms->n; i++) {
        const char *role = json_get(ms->items[i], "role")->str, *err = NULL;
        sb tmp = {0};
        const char *raw = content_text(ms->items[i], &tmp, &err);
        if (!raw) return free(tmp.p), err;
        char *content = trim(raw);
        if (!strcmp(role, "assistant")) {
            char *reasoning = strdup("");
            const jv *rc = json_get(ms->items[i], "reasoning_content");
            char *end = strstr(content, "</think>");
            if (rc && rc->type == J_STR) {
                free(reasoning);
                reasoning = strdup(rc->str);
            } else if (end) {
                // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                char *head = strndup(content, (size_t)(end - content)), *b = head, *t;
                while ((t = strstr(b, "<think>"))) b = t + 7;
                size_t n = strlen(b);
                while (n && b[n - 1] == '\n') n--;
                while (n && *b == '\n') b++, n--;
                free(reasoning);
                reasoning = strndup(b, n);
                free(head);
                char *last = end, *t2;
                while ((t2 = strstr(last + 8, "</think>"))) last = t2;
                char *after = last + 8;
                while (*after == '\n') after++;
                char *nc = strdup(after);
                free(content);
                content = nc;
            }
            char *r2 = trim(reasoning);
            free(reasoning);
            if ((long)i > last_user) {
                sb_add(o, "<|im_start|>assistant\n<think>\n");
                sb_add(o, r2);
                sb_add(o, "\n</think>\n\n");
                sb_add(o, content);
            } else {
                sb_add(o, "<|im_start|>assistant\n");
                sb_add(o, content);
            }
            sb_add(o, "<|im_end|>\n");
            free(r2);
        } else {
            sb_fmt(o, "<|im_start|>%s\n", role);
            sb_add(o, content);
            sb_add(o, "<|im_end|>\n");
        }
        free(content);
        free(tmp.p);
    }
    const jv *agp = json_get(body, "add_generation_prompt");
    if (!agp || agp->type != J_BOOL || agp->b) {
        sb_add(o, "<|im_start|>assistant\n");
        sb_add(o, thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    }
    return NULL;
}

static int num_field(const jv *body, const char *k, int *v, const char **err) {
    const jv *x = json_get(body, k);
    if (!x || x->type == J_NULL) return 0;
    if (x->type != J_NUM || x->num != (double)(long)x->num || x->num < 1 || x->num > 65536)
        return *err = "max_tokens must be an integer 1..65536", -1;
    *v = (int)x->num;
    return 1;
}

static const char *parse_req(const jv *body, int chat, Req *r) {
    if (!body || body->type != J_OBJ) return "request body must be a JSON object";
    const jv *x;
    if ((x = json_get(body, "n")) && !(x->type == J_NUM && x->num == 1)) return "only n=1 is supported";
    if ((x = json_get(body, "stream")) && x->type == J_BOOL) r->stream = x->b;
    if ((x = json_get(body, "stream_options")) && (x = json_get(x, "include_usage")) && x->type == J_BOOL)
        r->usage_chunk = x->b;
    if ((x = json_get(body, "return_tokens")) && x->type == J_BOOL) r->ret_tokens = x->b;
    const char *err = NULL;
    r->n = chat ? chat_default_n : 16; // OpenAI legacy completions default 16
    const char *ks[] = {"n_predict", "max_tokens", "max_completion_tokens"};
    for (int i = 0; i < 3; i++)
        if (num_field(body, ks[i], &r->n, &err) < 0) return err;
    sb p = {0};
    if (chat) {
        r->chat = 1;
        r->special = 1; // the template's <|im_start|>/<|im_end|> are control tokens
        if ((err = render_chat(body, &p))) return free(p.p), err;
    } else {
        x = json_get(body, "prompt");
        if (x && x->type == J_ARR && x->n >= 1 && x->items[0]->type == J_NUM) { // llama.cpp/OpenAI: a token-id prompt
            sb ib = {0};
            for (size_t i = 0; i < x->n; i++) {
                if (x->items[i]->type != J_NUM || x->items[i]->num < 0 || x->items[i]->num >= 248320) return free(ib.p), "prompt ids must be integers in [0, 248320)";
                sb_fmt(&ib, i ? ",%d" : "%d", (int)x->items[i]->num);
            }
            r->ids = ib.p;
            r->prompt = strdup(ib.p);
            return NULL;
        }
        if (x && x->type == J_ARR && x->n == 1) x = x->items[0];
        if (!x || x->type != J_STR) return "prompt must be a string (or a one-element array of strings)";
        if (!*x->str) return "prompt must not be empty";
        sb_add(&p, x->str);
    }
    r->prompt = p.p;
    return NULL;
}

// the dual.c generate request line (sampling knobs passed through, unused yet)
static char *engine_line(const jv *body, const Req *r) {
    sb o = {0};
    sb_add(&o, "{\"mode\":\"generate\",");
    if (r->ids) sb_fmt(&o, "\"prompt_ids\":[%s]", r->ids);
    else {
        sb_add(&o, "\"prompt\":");
        json_quote(&o, r->prompt);
    }
    sb_fmt(&o, ",\"n\":%d", r->n);
    if (r->special) sb_add(&o, ",\"special\":true");
    int any = 0;
    for (int i = 0; sampling_keys[i]; i++) {
        const jv *v = json_get(body, sampling_keys[i]);
        if (!v) continue;
        sb_add(&o, any++ ? "," : ",\"sampling\":{");
        json_quote(&o, sampling_keys[i]);
        sb_addc(&o, ':');
        json_dump(&o, v);
    }
    if (any) sb_addc(&o, '}');
    sb_addc(&o, '}');
    return o.p;
}

typedef struct {
    Conn *c;
    const Req *r;
    char id[48];
    time_t created;
    int nchunks;
    sb text; // streamed text so far
} Gen;

static void chunk_head(Gen *g, sb *o) {
    sb_fmt(o, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", g->id,
        g->r->chat ? "chat.completion.chunk" : "text_completion", (long)g->created);
    json_quote(o, alias);
    sb_add(o, ",\"system_fingerprint\":\"llb-sys1\",\"choices\":[");
}

static void stream_piece(Gen *g, const char *piece) {
    sb o = {0};
    chunk_head(g, &o);
    if (g->r->chat) {
        sb_add(&o, "{\"index\":0,\"delta\":{");
        if (!g->nchunks) sb_add(&o, "\"role\":\"assistant\",");
        sb_add(&o, "\"content\":");
        json_quote(&o, piece);
        sb_add(&o, "},\"logprobs\":null,\"finish_reason\":null}]}");
    } else {
        sb_add(&o, "{\"index\":0,\"text\":");
        json_quote(&o, piece);
        sb_add(&o, ",\"logprobs\":null,\"finish_reason\":null}]}");
    }
    sse(g->c, o.p);
    g->nchunks++;
    free(o.p);
}

static void usage_timings(sb *o, int pn, int gn, double t0, double t1, double t2) {
    double pm = (t1 - t0) * 1000, gm = (t2 - t1) * 1000;
    sb_fmt(o, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},", pn, gn, pn + gn);
    // llama.cpp-style timings, measured at the server: prompt = request sent ->
    // first token event (tokenize + prefill + first argmax); predicted = first ->
    // last token (the gn-1 decode steps).
    sb_fmt(o,
        "\"timings\":{\"prompt_n\":%d,\"prompt_ms\":%.3f,\"prompt_per_token_ms\":%.3f,\"prompt_per_second\":%.3f,"
        "\"predicted_n\":%d,\"predicted_ms\":%.3f,\"predicted_per_token_ms\":%.3f,\"predicted_per_second\":%.3f}",
        pn, pm, pn ? pm / pn : 0, pm > 0 ? pn * 1000 / pm : 0, gn, gm, gn > 1 ? gm / (gn - 1) : 0,
        gn > 1 && gm > 0 ? (gn - 1) * 1000 / gm : 0);
}

static void int_array(sb *o, const jv *a) {
    sb_addc(o, '[');
    for (size_t i = 0; a && i < a->n; i++) sb_fmt(o, i ? ",%s" : "%s", a->items[i]->raw);
    sb_addc(o, ']');
}

// one generate request through the engine (caller holds the slot)
static void run_generate(Slot *s, Conn *c, const jv *body, const Req *r) {
    Gen g = {.c = c, .r = r, .created = time(NULL)};
    snprintf(g.id, sizeof g.id, "%s-%lx%04lx", r->chat ? "chatcmpl" : "cmpl", (long)g.created, c->id & 0xffff);
    char *line = engine_line(body, r);
    double t0 = now(), t1 = 0;
    fputs(line, s->in);
    fputc('\n', s->in);
    free(line);
    if (fflush(s->in)) {
        s->dead = 1;
        err_reply(c, 503, "unavailable_error", "engine is not running");
        return;
    }
    if (r->stream) sse_open(c);
    char *l = NULL;
    size_t cap = 0;
    jv *fin = NULL;
    for (;;) {
        if (getline(&l, &cap, s->out) <= 0) {
            s->dead = 1;
            if (!r->stream) err_reply(c, 503, "unavailable_error", "engine exited");
            else sse(c, "{\"error\":{\"message\":\"engine exited\",\"type\":\"unavailable_error\",\"code\":503}}");
            free(l);
            return;
        }
        const char *err = NULL;
        jv *v = json_parse(l, &err);
        if (!v || v->type != J_OBJ) {
            logf_("engine: unparsable line: %s", l);
            json_free(v);
            continue;
        }
        const jv *tk = json_get(v, "token"), *e = json_get(v, "error");
        if (tk) {
            if (!t1) t1 = now();
            const char *piece = json_get(v, "text")->str;
            sb_add(&g.text, piece);
            if (r->stream && *piece) stream_piece(&g, piece);
            c->gen_n++;
            json_free(v);
            continue;
        }
        if (e) { // refused (prompt too long, ...): nothing ran
            const char *m = e->type == J_STR ? e->str : "engine refused the request";
            if (r->stream) {
                sb o = {0};
                sb_add(&o, "{\"error\":{\"message\":");
                json_quote(&o, m);
                sb_add(&o, ",\"type\":\"invalid_request_error\",\"code\":400}}");
                sse(c, o.p);
                sse(c, "[DONE]");
                free(o.p);
            } else err_reply(c, 400, "invalid_request_error", m);
            json_free(v);
            free(l);
            free(g.text.p);
            return;
        }
        fin = v;
        break;
    }
    free(l);
    double t2 = now();
    if (!t1) t1 = t2;
    const jv *pids = json_get(fin, "prompt_ids"), *ids = json_get(fin, "ids"), *txt = json_get(fin, "text");
    int pn = pids ? (int)pids->n : 0, gn = ids ? (int)ids->n : 0;
    c->prompt_n = pn;
    const char *full = txt && txt->type == J_STR ? txt->str : "";
    const jv *fj = json_get(fin, "finish");
    const char *finish = fj && fj->type == J_STR ? fj->str : "length";
    size_t have = g.text.p ? g.text.n : 0;
    if (strncmp(full, g.text.p ? g.text.p : "", have)) logf_("warning: streamed text is not a prefix of the final text");
    sb o = {0};
    if (r->stream) {
        if (strlen(full) > have) stream_piece(&g, full + have); // held-back UTF-8 tail, if any
        chunk_head(&g, &o);
        if (r->chat) sb_fmt(&o, "{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"%s\"}],", finish);
        else sb_fmt(&o, "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"%s\"}],", finish);
        usage_timings(&o, pn, gn, t0, t1, t2);
        sb_addc(&o, '}');
        sse(c, o.p);
        if (r->usage_chunk) { // OpenAI: a last chunk with empty choices carrying usage
            free(o.p);
            o = (sb){0};
            chunk_head(&g, &o);
            sb_fmt(&o, "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}", pn, gn, pn + gn);
            sse(c, o.p);
        }
        sse(c, "[DONE]");
    } else {
        sb_fmt(&o, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", g.id,
            r->chat ? "chat.completion" : "text_completion", (long)g.created);
        json_quote(&o, alias);
        sb_add(&o, ",\"system_fingerprint\":\"llb-sys1\",\"choices\":[{\"index\":0,");
        if (r->chat) {
            sb_add(&o, "\"message\":{\"role\":\"assistant\",\"content\":");
            json_quote(&o, full);
            sb_add(&o, "},");
        } else {
            sb_add(&o, "\"text\":");
            json_quote(&o, full);
            sb_addc(&o, ',');
        }
        sb_fmt(&o, "\"logprobs\":null,\"finish_reason\":\"%s\"}],", finish);
        usage_timings(&o, pn, gn, t0, t1, t2);
        if (r->ret_tokens) {
            sb_add(&o, ",\"llb\":{\"prompt_ids\":");
            int_array(&o, pids);
            sb_add(&o, ",\"ids\":");
            int_array(&o, ids);
            sb_addc(&o, '}');
        }
        sb_addc(&o, '}');
        respond(c, 200, "application/json; charset=utf-8", o.p, "X-LLB-Sampler: greedy\r\n");
    }
    free(o.p);
    free(g.text.p);
    json_free(fin);
}

static void handle_generate(Conn *c, const char *body_s, int chat) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    Req r = {0};
    if (!body) {
        err_reply(c, 400, "invalid_request_error", err ? err : "invalid JSON");
        return;
    }
    if ((err = parse_req(body, chat, &r))) {
        err_reply(c, 400, "invalid_request_error", err);
        json_free(body);
        free(r.prompt);
    free(r.ids);
        return;
    }
    c->stream = r.stream;
    if (engine_dead || all_dead()) err_reply(c, 503, "unavailable_error", "engine is not running");
    else if (!ready) err_reply(c, 503, "unavailable_error", "Loading model");
    else {
        queued++;
        unsigned long ticket;
        double tq = now();
        Slot *s = slot_acquire(r.prompt, &ticket);
        queued--;
        if (!s) err_reply(c, 503, "unavailable_error", "engine is not running");
        else {
            c->slot = s->idx;
            if (n_slots > 1)
                snprintf(c->xh, sizeof c->xh, "X-LLB-Slot: %d\r\nX-LLB-Ticket: %lu\r\nX-LLB-Queue-Ms: %.3f\r\nX-LLB-Run-Start: %.6f\r\n",
                    s->idx, ticket, (now() - tq) * 1000, now());
            run_generate(s, c, body, &r);
            slot_release(s, r.prompt);
        }
    }
    json_free(body);
    free(r.prompt);
    free(r.ids);
}

// the engine's decide line: the request body itself (the engine needs "model"; the served
// alias is filled in when the caller leaves it out). The body is already JSON.
static char *decide_line(const char *body_s, const jv *body) {
    if (json_get(body, "model")) return trim(body_s);
    char *t = trim(body_s);
    sb o = {0};
    sb_add(&o, "{\"model\":");
    json_quote(&o, alias);
    sb_addc(&o, ',');
    sb_add(&o, t + 1); // past '{', keeps the caller's keys as they are
    free(t);
    return o.p;
}

// POST /v1/decide: one clef decide request through the engine's clef head. The engine
// answers one line, its own response(): {"model","answers":{<q>:{type,choice|score,
// confidence,probabilities}},"usage":{"input_tokens",...}}. Passed through unchanged.
static void handle_decide(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body || body->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", err ? err : "request body must be a JSON object");
        json_free(body);
        return;
    }
    const jv *qs = json_get(body, "questions");
    if (!qs || qs->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", "questions must be an object");
        json_free(body);
        return;
    }
    if (engine_dead || all_dead()) {
        err_reply(c, 503, "unavailable_error", "engine is not running");
    } else if (!ready) {
        err_reply(c, 503, "unavailable_error", "Loading model");
    } else {
        char *line = decide_line(body_s, body);
        queued++;
        unsigned long ticket;
        Slot *s = slot_acquire(line, &ticket);
        queued--;
        if (!s) {
            err_reply(c, 503, "unavailable_error", "engine is not running");
        } else {
            double t0 = now();
            fputs(line, s->in);
            fputc('\n', s->in);
            if (fflush(s->in)) {
                s->dead = 1;
                err_reply(c, 503, "unavailable_error", "engine is not running");
            } else {
                char *l = NULL;
                size_t cap = 0;
                jv *v = NULL;
                // LLB_SYS1_STREAM=1 is set for the chat path, so the engine answers a decide's
                // forward with token events before the decision line: skip them here.
                for (;;) {
                    if (getline(&l, &cap, s->out) <= 0) {
                        s->dead = 1;
                        err_reply(c, 503, "unavailable_error", "engine exited");
                        break;
                    }
                    const char *perr = NULL;
                    v = json_parse(l, &perr);
                    if (!v || v->type != J_OBJ) {
                        err_reply(c, 502, "server_error", perr ? perr : "engine sent an unparsable decide line");
                        break;
                    }
                    if (json_get(v, "token")) {
                        json_free(v);
                        v = NULL;
                        continue;
                    }
                    const jv *e = json_get(v, "error");
                    if (e) {
                        err_reply(c, 400, "invalid_request_error", e->type == J_STR ? e->str : "engine refused the request");
                        break;
                    }
                    const jv *u = json_get(v, "usage"), *it = u ? json_get(u, "input_tokens") : NULL;
                    c->prompt_n = it && it->type == J_NUM ? (int)it->num : 0;
                    json_reply(c, 200, l);
                    logf_("decide: %d input tokens, %.1f ms", c->prompt_n, (now() - t0) * 1000);
                    break;
                }
                json_free(v);
                free(l);
            }
            slot_release(s, line);
        }
        free(line);
    }
    json_free(body);
}

// ---- see: the shipped senses detector --------------------------------------
// POST /v1/see {"image_b64": "<base64 or data: URL>"} -> {"detections":[{"label",
// "score","box":[x1,y1,x2,y2]}],"inference_ms",...}. The detector is the app's own
// D-FINE-n pipeline (see/detect_cli.py on the reflex-up venv: ORT CPU fp32, 640²
// stretch, sigmoid >= .5); the route writes the image to a temp file and shells it.
// Env overrides: LLB_SEE_PY, LLB_SEE_CLI, LLB_SEE_MODEL.

static int b64val(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+' || ch == '-') return 62;
    if (ch == '/' || ch == '_') return 63;
    return -1;
}

// standard base64 (padding optional, whitespace skipped); NULL on a bad character
static unsigned char *b64_decode(const char *s, size_t *out_n) {
    size_t n = strlen(s), k = 0, cap = n / 4 * 3 + 4;
    unsigned char *o = malloc(cap);
    unsigned v = 0;
    int q = 0;
    if (!o) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '=') break;
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
        int d = b64val(ch);
        if (d < 0) { free(o); return NULL; }
        v = (v << 6) | (unsigned)d;
        if (++q == 4) {
            o[k++] = (unsigned char)(v >> 16), o[k++] = (unsigned char)(v >> 8), o[k++] = (unsigned char)v;
            v = 0, q = 0;
        }
    }
    if (q == 2) o[k++] = (unsigned char)(v >> 4);
    else if (q == 3) o[k++] = (unsigned char)(v >> 10), o[k++] = (unsigned char)(v >> 2);
    else if (q == 1) { free(o); return NULL; }
    o[k] = 0;
    *out_n = k;
    return o;
}

static void see_path(char *dst, size_t cap, const char *env, const char *home_rel) {
    const char *e = getenv(env), *h = getenv("HOME");
    if (e && *e) snprintf(dst, cap, "%s", e);
    else snprintf(dst, cap, "%s%s", h ? h : "", home_rel);
}

static void handle_see(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body || body->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", err ? err : "request body must be a JSON object");
        json_free(body);
        return;
    }
    // SCHEMAS.md §3's text-vision shape (prompt in, text out) has no encoder on
    // this backend; answer its documented 503 instead of a wrong (boxes) answer.
    const jv *pr = json_get(body, "prompt");
    if (pr && pr->type == J_STR && pr->str[0]) {
        err_reply(c, 503, "unavailable_error", "vision is not enabled on this backend");
        json_free(body);
        return;
    }
    const jv *im = json_get(body, "image_b64");
    if (!im) im = json_get(body, "image"); // SCHEMAS.md's name; a data: URL is fine
    if (!im || im->type != J_STR || !im->str[0]) {
        err_reply(c, 400, "invalid_request_error", "image_b64 must be a base64 image string");
        json_free(body);
        return;
    }
    const char *b = im->str;
    if (!strncmp(b, "data:", 5)) {
        const char *comma = strchr(b, ',');
        if (!comma || !strstr(b, ";base64")) {
            err_reply(c, 400, "invalid_request_error", "image must be a base64 data: URL");
            json_free(body);
            return;
        }
        b = comma + 1;
    }
    size_t img_n = 0;
    unsigned char *img = b64_decode(b, &img_n);
    if (!img || !img_n) {
        err_reply(c, 400, "invalid_request_error", "image_b64 is not valid base64");
        free(img);
        json_free(body);
        return;
    }
    char tpl[1024], py[1024], cli[1024], model[1024], cmd[8192];
    const char *td = getenv("TMPDIR");
    snprintf(tpl, sizeof tpl, "%s/llb-see-XXXXXX", (td && *td) ? td : "/tmp");
    int fd = mkstemp(tpl);
    if (fd < 0) {
        err_reply(c, 500, "server_error", "cannot create the image temp file");
        free(img);
        json_free(body);
        return;
    }
    size_t off = 0;
    while (off < img_n) {
        ssize_t w = write(fd, img + off, img_n - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    close(fd);
    free(img);
    json_free(body);
    if (off != img_n) {
        unlink(tpl);
        err_reply(c, 500, "server_error", "cannot write the image temp file");
        return;
    }
    see_path(py, sizeof py, "LLB_SEE_PY", "/.eureka/see/.venv/bin/python");
    see_path(cli, sizeof cli, "LLB_SEE_CLI", "/.eureka/see/detect_cli.py");
    see_path(model, sizeof model, "LLB_SEE_MODEL", "/.eureka/see/dfine_n_hf.onnx");
    snprintf(cmd, sizeof cmd, "'%s' '%s' --model '%s' --image '%s' 2>/dev/null", py, cli, model, tpl);
    double t0 = now();
    FILE *f = popen(cmd, "r");
    sb out = {0};
    if (f) {
        char buf[4096];
        size_t k;
        while ((k = fread(buf, 1, sizeof buf, f)) > 0) sb_addn(&out, buf, k);
        int rc = pclose(f);
        if (rc != 0) out.n = 0;
    }
    unlink(tpl);
    while (out.n && (out.p[out.n - 1] == '\n' || out.p[out.n - 1] == '\r')) out.p[--out.n] = 0;
    jv *v = out.n ? json_parse(out.p, &err) : NULL;
    if (!v || v->type != J_OBJ || !json_get(v, "detections")) {
        err_reply(c, 503, "unavailable_error", "the detector is unavailable (see/detect_cli.py failed)");
    } else {
        json_reply(c, 200, out.p);
        logf_("see: %zu image bytes, %.1f ms wall", img_n, (now() - t0) * 1000);
    }
    json_free(v);
    free(out.p);
}

// ---- hear: whisper.cpp sidecar ---------------------------------------------

static int tcp_connect_ip4(int p) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons((unsigned short)p)};
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
    if (connect(fd, (struct sockaddr *)&sa, sizeof sa)) {
        close(fd);
        return -1;
    }
    return fd;
}

static void whisper_start(void) {
    if (!whisper_bin) return;
    if (!whisper_model) {
        logf_("hear: --whisper-model is required with --whisper; /v1/hear will 503");
        return;
    }
    pid_t p = fork();
    if (p < 0) {
        logf_("hear: fork failed: %s; /v1/hear will 503", strerror(errno));
        return;
    }
    if (!p) {
        char ps[16];
        snprintf(ps, sizeof ps, "%d", whisper_port);
        // Metal is whisper.cpp's default ggml backend on Apple Silicon; -t sizes the CPU side.
        execl(whisper_bin, whisper_bin, "--host", "127.0.0.1", "--port", ps, "-m", whisper_model, "-t", "8", (char *)NULL);
        _exit(127);
    }
    whisper_pid = p;
    int up = 0;
    for (int i = 0; i < 300; i++) { // 30 s: model load is ~0.2 s for base.en q5_1
        int fd = tcp_connect_ip4(whisper_port);
        if (fd >= 0) {
            close(fd);
            int st;
            if (waitpid(p, &st, WNOHANG) == p) {
                whisper_pid = -1;
                logf_("hear: whisper-server exited during load (status %d)", st);
                return;
            }
            up = 1;
            break;
        }
        int st;
        if (waitpid(p, &st, WNOHANG) == p) {
            whisper_pid = -1;
            logf_("hear: whisper-server exited during load (status %d)", st);
            return;
        }
        usleep(100000);
    }
    whisper_ready = up;
    logf_("hear: whisper-server pid=%d port=%d model=%s %s", (int)whisper_pid, whisper_port, whisper_model,
          up ? "ready" : "did not come up");
}

static int hear_b64v(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// data:audio/wav;base64,<...> or the bare base64 text; whitespace is ignored
static unsigned char *hear_b64_decode(const char *s, size_t *out_n) {
    if (!strncmp(s, "data:", 5)) {
        const char *comma = strchr(s, ',');
        if (!comma || !strstr(s, ";base64")) return NULL;
        s = comma + 1;
    }
    size_t cap = strlen(s) / 4 * 3 + 4, n = 0;
    unsigned char *o = malloc(cap);
    if (!o) return NULL;
    int q[4], qi = 0;
    for (; *s; s++) {
        int ch = (unsigned char)*s;
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') continue;
        if (ch == '=') break;
        int v = hear_b64v(ch);
        if (v < 0) {
            free(o);
            return NULL;
        }
        q[qi++] = v;
        if (qi == 4) {
            o[n++] = (unsigned char)((q[0] << 2) | (q[1] >> 4));
            o[n++] = (unsigned char)(((q[1] & 15) << 4) | (q[2] >> 2));
            o[n++] = (unsigned char)(((q[2] & 3) << 6) | q[3]);
            qi = 0;
        }
    }
    if (qi == 1) {
        free(o);
        return NULL;
    }
    if (qi == 2) o[n++] = (unsigned char)((q[0] << 2) | (q[1] >> 4));
    if (qi == 3) {
        o[n++] = (unsigned char)((q[0] << 2) | (q[1] >> 4));
        o[n++] = (unsigned char)(((q[1] & 15) << 4) | (q[2] >> 2));
    }
    *out_n = n;
    return o;
}

static unsigned le16(const unsigned char *p) { return (unsigned)p[0] | (unsigned)p[1] << 8; }
static unsigned le32(const unsigned char *p) {
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

// input duration from the RIFF chunks; -1 when this is not a WAV we understand
static double wav_duration(const unsigned char *p, size_t n) {
    if (n < 44 || memcmp(p, "RIFF", 4) || memcmp(p + 8, "WAVE", 4)) return -1;
    size_t off = 12, data = 0;
    unsigned rate = 0, ch = 0, bits = 0;
    while (off + 8 <= n) {
        unsigned sz = le32(p + off + 4);
        size_t avail = n - off - 8;
        if (sz > avail) sz = (unsigned)avail;
        if (!memcmp(p + off, "fmt ", 4) && sz >= 16) {
            ch = le16(p + off + 8 + 2);
            rate = le32(p + off + 8 + 4);
            bits = le16(p + off + 8 + 14);
        } else if (!memcmp(p + off, "data", 4)) {
            data = sz;
        }
        off += 8 + (size_t)sz + (sz & 1);
    }
    if (!rate || !ch || !bits || !data) return -1;
    return (double)data / ((double)rate * ch * (bits / 8.0));
}

// trim + collapse runs of whitespace (whisper's text and segments carry leading
// spaces and newlines; the wire shape carries clean sentences)
static void clean_text(sb *o, const char *s) {
    int sp = 0, any = 0;
    for (; *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            sp = 1;
            continue;
        }
        if (sp && any) sb_addc(o, ' ');
        sp = 0, any = 1;
        sb_addc(o, (char)ch);
    }
}

static int write_temp(const unsigned char *p, size_t n, char *path, size_t pathn) {
    const char *td = getenv("TMPDIR");
    if (!td || !*td) td = "/tmp";
    if ((size_t)snprintf(path, pathn, "%s/llb-hear-XXXXXX", td) >= pathn) return -1;
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, p + off, n - off);
        if (w <= 0) {
            close(fd);
            unlink(path);
            return -1;
        }
        off += (size_t)w;
    }
    close(fd);
    return 0;
}

// POST multipart /inference to the sidecar; returns the JSON body (malloc'd),
// *st_out = HTTP status, *ms_out = wall time around the sidecar call
static char *whisper_infer(const unsigned char *wav, size_t wn, const char *lang, long *ms_out, int *st_out) {
    const char *B = "----llb-hear-2c9a7f";
    sb req = {0};
    sb_fmt(&req, "--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n", B);
    sb_addn(&req, (const char *)wav, wn);
    sb_fmt(&req, "\r\n--%s\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n", B);
    sb_fmt(&req, "--%s\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n0\r\n", B);
    if (lang && *lang)
        sb_fmt(&req, "--%s\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n%s\r\n", B, lang);
    sb_fmt(&req, "--%s--\r\n", B);

    sb head = {0};
    sb_fmt(&head, "POST /inference HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
                  "Content-Type: multipart/form-data; boundary=%s\r\nContent-Length: %zu\r\n"
                  "Connection: close\r\n\r\n", whisper_port, B, req.n);

    double t0 = now();
    int fd = tcp_connect_ip4(whisper_port);
    if (fd < 0) {
        free(req.p), free(head.p);
        return NULL;
    }
    struct timeval tv = {120, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    if (send_all(fd, head.p, head.n) || send_all(fd, req.p, req.n)) {
        close(fd), free(req.p), free(head.p);
        return NULL;
    }
    sb r = {0};
    char buf[16384];
    ssize_t k;
    while ((k = recv(fd, buf, sizeof buf, 0)) > 0) sb_addn(&r, buf, (size_t)k);
    close(fd);
    free(req.p), free(head.p);
    *ms_out = (long)((now() - t0) * 1000);
    if (!r.p) return NULL;
    char *he = strstr(r.p, "\r\n\r\n");
    if (!he) {
        free(r.p);
        return NULL;
    }
    *st_out = atoi(r.p + 9);
    char *body = strdup(he + 4);
    free(r.p);
    return body;
}

static int lang_ok(const char *s) {
    size_t n = strlen(s);
    if (n < 2 || n > 10) return 0;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-')) return 0;
    }
    return 1;
}

// POST /v1/hear: {audio: data URL, language, timestamps} -> the SCHEMAS.md shape.
// The LLM engine is not involved; the sidecar is the ear.
static void handle_hear(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body || body->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", err ? err : "request body must be a JSON object");
        json_free(body);
        return;
    }
    const jv *a = json_get(body, "audio");
    if (!a || a->type != J_STR || !a->str[0]) {
        err_reply(c, 400, "invalid_request_error", "audio must be a base64 data URL string");
        json_free(body);
        return;
    }
    const char *lang = "en";
    const jv *lj = json_get(body, "language");
    if (lj) {
        if (lj->type != J_STR || !lang_ok(lj->str)) {
            err_reply(c, 400, "invalid_request_error", "language must be a short code such as \"en\"");
            json_free(body);
            return;
        }
        lang = lj->str;
    }
    int want_ts = 0;
    const jv *tj = json_get(body, "timestamps");
    if (tj) {
        if (tj->type != J_BOOL) {
            err_reply(c, 400, "invalid_request_error", "timestamps must be true or false");
            json_free(body);
            return;
        }
        want_ts = tj->b;
    }
    const char *model = hear_model;
    const jv *mj = json_get(body, "model");
    if (mj) {
        if (mj->type != J_STR || !mj->str[0]) {
            err_reply(c, 400, "invalid_request_error", "model must be a string");
            json_free(body);
            return;
        }
        model = mj->str;
    }
    if (!whisper_bin || !whisper_model || whisper_pid <= 0 || !whisper_ready) {
        err_reply(c, 503, "unavailable_error", "ears are not available (whisper-server is not running)");
        json_free(body);
        return;
    }
    size_t wn = 0;
    unsigned char *wav = hear_b64_decode(a->str, &wn);
    if (!wav || wn < 44) {
        free(wav);
        err_reply(c, 400, "invalid_request_error", "audio must be base64-encoded WAV bytes");
        json_free(body);
        return;
    }
    double wd = wav_duration(wav, wn);
    if (wd < 0) {
        free(wav);
        err_reply(c, 400, "invalid_request_error", "audio must be a RIFF/WAVE file");
        json_free(body);
        return;
    }
    char path[1024];
    if (write_temp(wav, wn, path, sizeof path)) {
        free(wav);
        err_reply(c, 500, "server_error", "cannot write the audio temp file");
        json_free(body);
        return;
    }

    int st = 0;
    long ms = 0;
    pthread_mutex_lock(&hear_lock);
    char *resp = whisper_infer(wav, wn, lang, &ms, &st);
    pthread_mutex_unlock(&hear_lock);
    unlink(path);
    free(wav);

    if (!resp || st != 200) {
        char msg[128];
        snprintf(msg, sizeof msg, "whisper-server did not answer (HTTP %d)", st);
        err_reply(c, 503, "unavailable_error", msg);
        free(resp);
        json_free(body);
        return;
    }
    const char *perr = NULL;
    jv *w = json_parse(resp, &perr);
    if (!w || w->type != J_OBJ) {
        err_reply(c, 502, "server_error", perr ? perr : "whisper-server sent unparsable JSON");
        json_free(w);
        free(resp);
        json_free(body);
        return;
    }
    const jv *wt = json_get(w, "text");
    if (!wt || wt->type != J_STR) {
        const jv *we = json_get(w, "error");
        err_reply(c, 502, "server_error", we && we->type == J_STR ? we->str : "whisper-server sent no text");
        json_free(w);
        free(resp);
        json_free(body);
        return;
    }
    const jv *wj = json_get(w, "duration");
    double dur = wj && wj->type == J_NUM && wj->num > 0 ? wj->num : wd;

    sb o = {0};
    sb_add(&o, "{\"model\":");
    json_quote(&o, model);
    sb_add(&o, ",\"text\":");
    sb t = {0};
    clean_text(&t, wt->str);
    json_quote(&o, t.p ? t.p : "");
    free(t.p);
    sb_fmt(&o, ",\"duration_s\":%.3f", dur);
    if (want_ts) {
        const jv *segs = json_get(w, "segments");
        sb_add(&o, ",\"segments\":[");
        if (segs && segs->type == J_ARR) {
            int emitted = 0;
            for (size_t i = 0; i < segs->n; i++) {
                const jv *s = segs->items[i];
                if (!s || s->type != J_OBJ) continue;
                const jv *stj = json_get(s, "start"), *enj = json_get(s, "end"), *txj = json_get(s, "text");
                if (emitted++) sb_addc(&o, ',');
                sb_fmt(&o, "{\"start\":%.2f,\"end\":%.2f,\"text\":", stj && stj->type == J_NUM ? stj->num : 0.0,
                       enj && enj->type == J_NUM ? enj->num : 0.0);
                sb tx = {0};
                clean_text(&tx, txj && txj->type == J_STR ? txj->str : "");
                json_quote(&o, tx.p ? tx.p : "");
                free(tx.p);
                sb_addc(&o, '}');
            }
        }
        sb_addc(&o, ']');
    }
    // whisper's encoder sees 1500 audio frames per 30 s window
    sb_fmt(&o, ",\"usage\":{\"audio_tokens\":%d},\"llb\":{\"decode_ms\":%.1f}}", (int)(dur * 50.0 + 0.5), (double)ms);
    json_reply(c, 200, o.p);
    logf_("hear: %.3f s audio, %zu B body, %.1f ms whisper", dur, o.n, (double)ms);
    free(o.p);
    json_free(w);
    free(resp);
    json_free(body);
}

// ---- M2: /v1/speak transport ----------------------------------------------
// speak_route.c never sees a Conn: these two callbacks are the whole coupling.
// send is length-based (a WAV is full of NUL bytes; respond()'s strlen would
// truncate it), and head emits the status line plus the route's extra headers.
static int speak_send(void *ctx, const void *buf, size_t n) {
    Conn *c = ctx;
    if (send_all(c->fd, buf, n)) {
        c->broken = 1;
        return -1;
    }
    c->bytes += (long)n;
    return 0;
}

static void speak_head(void *ctx, int status, const char *ctype, size_t len, const char *extra) {
    Conn *c = ctx;
    char h[640];
    int k = snprintf(h, sizeof h,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n"
        "Server: llb-serve\r\n%s%s\r\n",
        status, reason(status), ctype, len, extra ? extra : "", c->xh);
    c->status = status;
    if (send_all(c->fd, h, (size_t)k)) c->broken = 1;
    c->bytes += k;
}

static void route(Conn *c, const char *body) {
    char *q = strchr(c->path, '?');
    if (q) *q = 0;
    const char *p = c->path;
    int get = !strcmp(c->method, "GET"), post = !strcmp(c->method, "POST");
    if (api_key && strcmp(p, "/health") && strcmp(p, "/v1/health") && !c->authed)
        return err_reply(c, 401, "authentication_error", "Invalid API Key");
    if (!strcmp(p, "/health") || !strcmp(p, "/v1/health")) {
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        if (engine_dead || all_dead()) return err_reply(c, 503, "unavailable_error", "engine is not running");
        if (!ready) return err_reply(c, 503, "unavailable_error", "Loading model");
        return json_reply(c, 200, "{\"status\":\"ok\"}");
    }
    if (n_slots > 1 && !strcmp(p, "/slots")) { // G6: llama-server's slot monitor, the N>1 server only
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        sb o = {0};
        sb_addc(&o, '[');
        pthread_mutex_lock(&q_lock);
        for (int i = 0; i < n_slots; i++) {
            Slot *x = &slots[i];
            sb_fmt(&o, "%s{\"id\":%d,\"state\":\"%s\",\"is_processing\":%s,\"n_requests\":%ld,\"pid\":%d}", i ? "," : "",
                i, x->dead ? "dead" : !x->ready ? "loading" : x->busy ? "processing" : "idle", x->busy ? "true" : "false",
                x->n_req, (int)x->pid);
        }
        sb_fmt(&o, "]");
        pthread_mutex_unlock(&q_lock);
        json_reply(c, 200, o.p);
        free(o.p);
        return;
    }
    if (!strcmp(p, "/v1/models") || !strcmp(p, "/models")) {
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        sb o = {0};
        sb_add(&o, "{\"object\":\"list\",\"data\":[{\"id\":");
        json_quote(&o, alias);
        sb_fmt(&o, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"llama-bend\"}]}", (long)t_start);
        json_reply(c, 200, o.p);
        free(o.p);
        return;
    }
    if (!strcmp(p, "/v1/decide") || !strcmp(p, "/decide")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_decide(c, body);
    }
    if (!strcmp(p, "/v1/hear") || !strcmp(p, "/hear")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_hear(c, body);
    }
    if (!strcmp(p, "/v1/see") || !strcmp(p, "/see")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_see(c, body);
    }
    if (!strcmp(p, "/v1/speak")) { // M2: {text, voice?, style?} -> 24 kHz WAV
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        const char *const av[] = {speak_cmd, NULL};
        speak_cfg cfg = {.argv = av, .cpu = speak_cpu, .timeout_ms = speak_timeout_ms, .tmpdir = getenv("TMPDIR")};
        speak_sink sink = {.ctx = c, .send = speak_send, .head = speak_head};
        speak_route_handle(&cfg, &sink, body, strlen(body));
        return;
    }
    int comp = !strcmp(p, "/v1/completions") || !strcmp(p, "/completions");
    int chat = !strcmp(p, "/v1/chat/completions") || !strcmp(p, "/chat/completions");
    if (comp || chat) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_generate(c, body, chat);
    }
    err_reply(c, 404, "not_found_error", "File Not Found");
}

static char *find_header(char *head, const char *name) {
    size_t n = strlen(name);
    for (char *l = strstr(head, "\r\n"); l; l = strstr(l + 2, "\r\n"))
        if (!strncasecmp(l + 2, name, n) && l[2 + n] == ':') {
            char *v = l + 3 + n;
            while (*v == ' ' || *v == '\t') v++;
            return v;
        }
    return NULL;
}

static void *conn_main(void *arg) {
    Conn *c = arg;
    double t0 = now();
    char *buf = malloc(MAX_HEAD + 1), *body = NULL;
    size_t got = 0;
    char *end = NULL;
    while (!end && got < MAX_HEAD) {
        ssize_t k = recv(c->fd, buf + got, MAX_HEAD - got, 0);
        if (k <= 0) goto done;
        got += (size_t)k;
        buf[got] = 0;
        end = strstr(buf, "\r\n\r\n");
    }
    if (!end) {
        err_reply(c, 413, "invalid_request_error", "headers too large");
        goto done;
    }
    *end = 0;
    if (sscanf(buf, "%15s %255s", c->method, c->path) != 2) {
        err_reply(c, 400, "invalid_request_error", "bad request line");
        goto done;
    }
    if (api_key) { // Authorization: Bearer KEY or X-Api-Key: KEY (the desk supervisor's per-launch key)
        size_t kl = strlen(api_key);
        char *a = find_header(buf, "Authorization"), *x = find_header(buf, "X-Api-Key");
        if (a && !strncasecmp(a, "Bearer ", 7) && !strncmp(a + 7, api_key, kl) && (a[7 + kl] == '\r' || a[7 + kl] == 0)) c->authed = 1;
        if (x && !strncmp(x, api_key, kl) && (x[kl] == '\r' || x[kl] == 0)) c->authed = 1;
    }
    char *te = find_header(buf, "Transfer-Encoding"), *cl = find_header(buf, "Content-Length");
    if (te && !strncasecmp(te, "chunked", 7)) {
        err_reply(c, 411, "invalid_request_error", "chunked request bodies are not supported; send Content-Length");
        goto done;
    }
    long len = cl ? strtol(cl, NULL, 10) : 0;
    if (len < 0 || len > MAX_BODY) {
        err_reply(c, 413, "invalid_request_error", "body too large");
        goto done;
    }
    body = malloc((size_t)len + 1);
    size_t have = got - (size_t)(end + 4 - buf);
    if (have > (size_t)len) have = (size_t)len;
    memcpy(body, end + 4, have);
    while (have < (size_t)len) {
        ssize_t k = recv(c->fd, body + have, (size_t)len - have, 0);
        if (k <= 0) goto done;
        have += (size_t)k;
    }
    body[len] = 0;
    route(c, body);
done:;
    char sl[24] = "";
    if (n_slots > 1 && c->slot >= 0) snprintf(sl, sizeof sl, " slot=%d", c->slot);
    logf_("#%ld %s \"%s %s\" %d %ldB %.1fms prompt=%d gen=%d%s%s%s", c->id, c->ip, c->method, c->path, c->status,
        c->bytes, (now() - t0) * 1000, c->prompt_n, c->gen_n, c->stream ? " stream" : "",
        c->broken ? " client-gone" : "", sl);
    shutdown(c->fd, SHUT_WR);
    close(c->fd);
    free(buf);
    free(body);
    free(c);
    inflight--;
    return NULL;
}

static void on_signal(int s) {
    (void)s;
    stopping = 1;
    if (write(wake[1], "x", 1) < 0) {}
}

static void usage(const char *a0) {
    fprintf(stderr,
        "usage: %s [--host H] [--port P] [--alias NAME] [--chat-max-tokens N] [--drain-seconds S]\n"
        "          [--canon] [--api-key KEY] [--reasoning on|off] [--parallel N]\n"
        "          [--speak-cmd PATH] [--speak-gpu] [--speak-timeout MS] --engine PATH [-- ENGINE ARGS...]\n"
        "          [--whisper PATH --whisper-model PATH [--whisper-port P] [--hear-model NAME]]\n"
        "          --engine PATH [-- ENGINE ARGS...]\n"
        "  --parallel N (-np): N slots = N engine children (each loads its own weights + KV), default 1\n"
        "  --canon: the CANON tier for this launch (exact per-token prefill, bit-exact GEMV); default is fast\n"
        "  --whisper PATH: whisper-server binary for /v1/hear (spawned as a loopback sidecar)\n"
        "  --whisper-model PATH: ggml whisper model (base.en-q5_1 is the pinned one)\n"
        "  --whisper-port P: sidecar port, default 8137\n"
        "  ENGINE ARGS default: --threads 16 -- $SYS1_SIDECAR 0 --steps 4096  (the SYS1-DUAL launch)\n"
        "  env: LLB_GGUF (the model, required by the engine), TMPDIR (owned scratch)\n",
        a0);
    exit(2);
}

int main(int argc, char **argv) {
    const char *engine = NULL;
    char **eargs = NULL;
    int ean = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--")) {
            eargs = argv + i + 1, ean = argc - i - 1;
            break;
        }
        if (!strcmp(a, "--canon")) { canon = 1; continue; }
        if (!strcmp(a, "--speak-gpu")) { speak_cpu = 0; continue; } // M2
        if (i + 1 >= argc) usage(argv[0]);
        if (!strcmp(a, "--host")) host = argv[++i];
        else if (!strcmp(a, "--port")) port = atoi(argv[++i]);
        else if (!strcmp(a, "--alias")) alias = argv[++i];
        else if (!strcmp(a, "--engine")) engine = argv[++i];
        else if (!strcmp(a, "--chat-max-tokens")) chat_default_n = atoi(argv[++i]);
        else if (!strcmp(a, "--drain-seconds")) drain_s = atoi(argv[++i]);
        else if (!strcmp(a, "--api-key")) api_key = argv[++i];
        else if (!strcmp(a, "--reasoning")) think_default = strcmp(argv[++i], "off") != 0;
        else if (!strcmp(a, "--parallel") || !strcmp(a, "-np")) n_slots = atoi(argv[++i]);
        else if (!strcmp(a, "--speak-cmd")) speak_cmd = argv[++i]; // M2
        else if (!strcmp(a, "--speak-timeout")) speak_timeout_ms = atoi(argv[++i]);
        else if (!strcmp(a, "--whisper")) whisper_bin = argv[++i];
        else if (!strcmp(a, "--whisper-model")) whisper_model = argv[++i];
        else if (!strcmp(a, "--whisper-port")) whisper_port = atoi(argv[++i]);
        else if (!strcmp(a, "--hear-model")) hear_model = argv[++i];
        else usage(argv[0]);
    }
    if (!engine || port <= 0 || port > 65535 || chat_default_n < 1 || n_slots < 1 || n_slots > 64 ||
        whisper_port <= 0 || whisper_port > 65535) usage(argv[0]);
    if (!getenv("LLB_GGUF")) fprintf(stderr, "serve: LLB_GGUF is unset (the engine will refuse)\n"), exit(2);
    // print the tier this launch will actually use (the env opt-ins still win without --canon)
    const char *fpe = getenv("LLB_DUAL_FASTPF"), *fge = getenv("LLB_FASTGEMV");
    int pfk = canon ? 0 : (fpe && *fpe ? atoi(fpe) : 512);
    int gemv = canon ? 0 : (fge ? (*fge != 0 && atoi(fge) != 0) : 1);
    fprintf(stderr, "serve: tier %s (prefill %s, gemv %s)\n", pfk > 0 ? "FAST" : "CANON",
            pfk > 0 ? "chunkf non-exact" : "exact per-token", gemv ? "fast" : "bit-exact");
    if (!alias) {
        const char *g = getenv("LLB_GGUF"), *s = strrchr(g, '/');
        alias = s ? s + 1 : g;
    }
    char *defargs[] = {"--threads", "16", "--", getenv("SYS1_SIDECAR"), "0", "--steps", "4096"};
    if (!eargs) {
        if (!defargs[3]) fprintf(stderr, "serve: no engine args and SYS1_SIDECAR unset\n"), exit(2);
        eargs = defargs, ean = 7;
    }
    char **ev = calloc((size_t)ean + 2, sizeof *ev);
    ev[0] = (char *)engine;
    memcpy(ev + 1, eargs, sizeof *ev * (size_t)ean);

    signal(SIGPIPE, SIG_IGN);
    if (pipe(wake)) perror("pipe"), exit(1);
    struct sigaction sa = {.sa_handler = on_signal};
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM, .ai_flags = AI_PASSIVE}, *ai;
    char ps[16];
    snprintf(ps, sizeof ps, "%d", port);
    if (getaddrinfo(host, ps, &hints, &ai)) fprintf(stderr, "serve: cannot resolve %s\n", host), exit(2);
    int ls = socket(ai->ai_family, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (ls < 0 || bind(ls, ai->ai_addr, ai->ai_addrlen) || listen(ls, 64))
        fprintf(stderr, "serve: cannot listen on %s:%d: %s\n", host, port, strerror(errno)), exit(2);
    freeaddrinfo(ai);

    t_start = time(NULL);
    slots = calloc((size_t)n_slots, sizeof *slots);
    for (int i = 0; i < n_slots; i++) {
        slots[i].idx = i;
        engine_start(&slots[i], ev);
    }
    whisper_start();
    if (n_slots == 1)
        logf_("listening on http://%s:%d model=%s engine=%s pid=%d (loading)", host, port, alias, engine, (int)slots[0].pid);
    else
        logf_("listening on http://%s:%d model=%s engine=%s slots=%d (loading)", host, port, alias, engine, n_slots);

    while (!stopping) {
        struct pollfd pf[2] = {{.fd = ls, .events = POLLIN}, {.fd = wake[0], .events = POLLIN}};
        if (poll(pf, 2, 1000) < 0 && errno != EINTR) break;
        for (int i = 0; i < n_slots; i++) {
            Slot *x = &slots[i];
            int st;
            if (x->pid > 0 && waitpid(x->pid, &st, WNOHANG) == x->pid) {
                x->pid = -x->pid;
                x->status = st;
                x->dead = 1;
                if (n_slots == 1) logf_("engine exited (status %d) — serving 503", st);
                else logf_("engine[%d] exited (status %d)%s", i, st, all_dead() ? " — serving 503" : "");
                pthread_mutex_lock(&q_lock);
                pthread_cond_broadcast(&q_cond);
                pthread_mutex_unlock(&q_lock);
            }
        }
        if (all_dead()) engine_dead = 1;
        if (whisper_pid > 0) {
            int st;
            if (waitpid(whisper_pid, &st, WNOHANG) == whisper_pid) {
                whisper_pid = -1;
                whisper_ready = 0;
                logf_("whisper-server exited (status %d) — /v1/hear 503", st);
            }
        }
        if (!(pf[0].revents & POLLIN)) continue;
        struct sockaddr_storage sa2;
        socklen_t sl = sizeof sa2;
        int fd = accept(ls, (struct sockaddr *)&sa2, &sl);
        if (fd < 0) continue;
        Conn *c = calloc(1, sizeof *c);
        c->fd = fd;
        c->slot = -1;
        c->id = ++req_seq;
        if (sa2.ss_family == AF_INET) inet_ntop(AF_INET, &((struct sockaddr_in *)&sa2)->sin_addr, c->ip, sizeof c->ip);
        else inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&sa2)->sin6_addr, c->ip, sizeof c->ip);
        inflight++;
        pthread_t t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&t, &at, conn_main, c)) close(fd), free(c), inflight--;
        pthread_attr_destroy(&at);
    }
    // graceful: stop accepting, drain in-flight requests, then close the
    // engine's stdin (dual.c sees EOF and the engine exits on its own)
    close(ls);
    logf_("shutdown: no new connections; draining %d in-flight (%d queued)", (int)inflight, (int)queued);
    double td = now();
    while (inflight > 0 && now() - td < drain_s) usleep(20000);
    // no request is mid-engine past this point: take every slot (q_head order)
    for (int i = 0; i < n_slots; i++) {
        pthread_mutex_lock(&q_lock);
        while (slots[i].busy) pthread_cond_wait(&q_cond, &q_lock);
        slots[i].busy = 1;
        pthread_mutex_unlock(&q_lock);
    }
    for (int i = 0; i < n_slots; i++) fclose(slots[i].in);
    for (int i = 0; i < n_slots; i++) {
        pid_t pid = slots[i].pid < 0 ? -slots[i].pid : slots[i].pid;
        int st = slots[i].status, rc = slots[i].pid < 0 ? pid : -1;
        char tag[16] = "";
        if (n_slots > 1) snprintf(tag, sizeof tag, "[%d]", i);
        for (double tw = now(); rc != pid && now() - tw < 30; usleep(50000))
            if ((rc = waitpid(pid, &st, WNOHANG)) == pid) break;
        if (rc != pid) {
            logf_("engine%s did not exit on EOF in 30 s: SIGTERM", tag);
            kill(pid, SIGTERM);
            waitpid(pid, &st, 0);
        }
        logf_("shutdown complete: drained in %.1f ms, %d left, engine%s exit %s %d", (now() - td) * 1000, (int)inflight,
            tag, WIFEXITED(st) ? "rc" : "signal", WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
    }
    if (whisper_pid > 0) {
        kill(whisper_pid, SIGTERM);
        int ws;
        waitpid(whisper_pid, &ws, 0);
        logf_("shutdown: whisper-server stopped");
    }
    return 0;
}
