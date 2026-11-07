// serve.c: W1-SERVER — the OpenAI-compatible HTTP front of the SYS1-DUAL engine
// (docs: SERVER.md in the W1-SERVER run dir). One process that owns ONE engine
// child (`LLB_SYS1=1 llb-engine ...`, the stdin/stdout JSON-line protocol of
// dual.c) and turns HTTP requests into its generate requests. No engine code
// here: tokenize/prefill/decode/detokenize all stay in the engine; this file is
// sockets, HTTP/1.1 framing, the chat template, OpenAI JSON shapes and SSE.
//   GET  /health                      200 {"status":"ok"} | 503 while loading / engine gone
//   GET  /v1/models                   the one served model
//   GET  /version                     {"name":"llb-serve","version":...,"git":...} (G31)
//   POST /v1/completions              text completion (stream or not)
//   POST /v1/chat/completions         chat completion (Qwen3.5 ChatML template)
//   POST /v1/decide                   one clef decide task (all its questions from one engine pass) (SERVE-BATCH)
//   {"batch":[...]} on the three POSTs: N requests in one HTTP request, one slot hold (SERVE-BATCH)
// G41 (--tts DIR, default off): speech. A resident Kokoro-82M child (tools/tts/kokoro_tts.py worker, the
// phone's voice layer on Linux: docs in tools/tts/README.md), one render at a time:
//   POST /v1/audio/speech             OpenAI shape {input, voice?, speed?, response_format: wav|pcm} -> audio
//   GET  /v1/audio/voices, /voice/voices   the voice list (VoiceServer's /voice/voices), the selected one
//   POST /voice/select {id}           the default voice from the next request on (hot swap, no reload)
// EARS-CANON (--ear DIR, default off): hearing. A resident llb-ear child (tools/ears: the CUDA AuT tower + the
// Qwen3-ASR-0.6B decoder, each loaded once), one transcription at a time:
//   POST /v1/audio/transcriptions     {input_audio:{data,format:"wav"}, response_format?: json|text|verbose_json} -> text
//   chat: an input_audio part is transcribed by the ear and its transcript replaces the audio tag in the user turn
//         (mm_replace_audio), so the trunk gets a text prompt; requests without audio are untouched.
// Senses (env-configured, always routed; 503 until configured, docs in docs/reference/http-api.md#senses):
//   POST /v1/see                      {image, threshold?, max_detections?} -> D-FINE-N COCO-80 boxes
//                                     (see_route.c: a tools/see/see_worker.py child, onnxruntime; LLB_SEE_*)
//   POST /v1/hear                     {audio: base64 WAV} -> transcript (hear_route.c: proxy to a resident
//                                     whisper.cpp whisper-server on loopback; LLB_HEAR_*)
//   POST /v1/speak                    {text, voice?} -> WAV: the --tts speech route under the senses name
// One request runs on the engine at a time (single slot, FIFO by the engine
// lock); the others wait. Streaming = dual.c's LLB_SYS1_STREAM token events.
// G6 (--parallel N, default 1): N slots = N engine children, each its own
// weights + KV (no shared-weight batching yet; see G6 DESIGN.md). A request
// takes a FIFO ticket and gets the first free slot when its ticket is at the
// head of the queue (strict arrival order); among free slots it prefers the one
// whose last prompt shares the longest prefix (prefix-cache affinity), then the
// least recently used. N=1 is the single-slot server exactly.
// G20 (--banks B, default 1): stage S1. Each engine child holds B sequence
// states (banks: KV + GDN ring/S + its prefix-cache snapshots and resident ids,
// host llb_bank_select) behind ONE copy of the weights. A logical slot is
// (engine, bank), id = engine * B + bank. One request runs on an engine at a
// time; among the free engines' banks the queue head picks the bank whose last
// prompt shares the longest prefix (>= --affinity-min bytes, else 0), then the
// least recently used bank, so up to B clients keep their own warm state.
// Sampling fields (temperature, top_p, seed, stop, ...) are passed through to the
// engine request under "sampling"; dual.c turns the numeric ones into the
// W1-SAMPLER `--sample` word (G15). Stop strings are not applied yet.
// --decide-path (G15, docs/reference/decide-configuration.md; fast bound owner-signed 2026-10-14): the child
// runs LLB_PREFILL=fast + LLB_PREFIX_CACHE=1 with a greedy LLB_SAMPLE default.
// Hidden-row (CLEF decide) requests still take the exact pass: the fast chunks emit
// no per-position rows. NOT the default: fast prefill needs every GEMM in Q4_K/Q5_K/
// Q6_K/Q8_0, and the served 9B clef-flash GGUF has F32 ssm_alpha/ssm_beta, so the
// engine exited on the first fast chunk (G15 receipts). S6-GUARD: llb-serve now reads the
// GGUF's tensor table at start and serves --exact for such a model (one stderr line); the
// engine's pf_fast makes the same check. --exact (the default) = the
// golden oracle / W1-SERVER launch: exact prefill, no prefix cache.
// Multimodal content (G5): image_url / input_audio parts are parsed, sniffed and capped in mm.c,
// rendered as the template's <|vision_start|><|image_pad|><|vision_end|>, staged to TMPDIR and
// handed to the engine as "media":[...] (docs/reference/multimodal-content.md). Off unless --mm says the engine takes it.
// Build: tools/sys1dual/build-serve.sh (links mm.c + tools/clef3/json.c, and for G16's
// --structured serve_structured.c + host/llb_structured.c).
// G16 (--structured, default off): response_format / json_schema / grammar (one GBNF text
// in dual.c's "grammar" field, G1's mask), tools with the Qwen3.5 tool template and the
// streaming tool-call / reasoning parser, and a stop at the first end-of-generation id for
// those requests. Plain requests take the W1-SERVER path byte for byte, flag on or off.
// Docs: docs/guides/structured-output.md.
// G31 polish: HTTP/1.1 keep-alive (non-streamed responses; SSE still closes), one
// OpenAI error shape everywhere (SSE error events included), CORS that allows
// local origins by default (--cors), a bounded SIGTERM shutdown (wedged engines
// are signalled after the drain; a second signal skips the drain), /version.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../clef3/json.h"
#include "../../host/llb_toolcall.h"
#include "serve_structured.h"
#include "mm.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_HEAD (64 * 1024)
static size_t MAX_BODY = 16 * 1024 * 1024; // --max-body-mb (16: a 6 MiB clip is ~8.0 MiB base64 + JSON)
static int max_conns = 256;        // --max-conns (0 = unlimited): above it, 503 + close without a thread
static int send_timeout_s = 30;    // --send-timeout: SO_SNDTIMEO, so a stalled reader cannot pin a slot
static int engine_timeout_s = 180; // --engine-timeout: a busy engine silent this long is killed (watchdog)

// HARDEN: a forked child keeps only stdio. Fixes the inherited listen fd / other slots' pipes.
static void close_extra_fds(void) {
    long m = sysconf(_SC_OPEN_MAX);
    if (m < 0 || m > 65536) m = 65536;
    for (int fd = 3; fd < (int)m; fd++) close(fd);
}
#define IO_TIMEOUT_MS 30000 // a started request must keep arriving (else 408)
#ifndef LLB_SERVE_VERSION
#define LLB_SERVE_VERSION "0.9.0"
#endif
#ifndef LLB_GIT
#define LLB_GIT "unknown"
#endif


static const char *host = "127.0.0.1", *alias;
static int port = 8080, chat_default_n = 256, drain_s = 120, structured_on;
// R5-FIX: LLB_METRICS_PORT with N > 1 slots: engine[i] listens on metrics_base + i (0 = off, or N = 1:
// the env reaches the one engine untouched)
static int metrics_base;
static int keepalive_s = 30, keepalive_max = 1000;
// CORS: NULL = local origins only (default), "*" = any, "none" = no CORS at all
// (no headers, no Origin check: the pre-G31 behaviour), else extra allowed
// origins, comma-separated, on top of the local ones.
static const char *cors = NULL;
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
    pthread_t err_thread;
    double t_last;
    char *last; // the last prompt text sent to this slot (affinity)
    struct Bank *bk; // G20: the engine's banks (n_banks of them)
} Slot;
// G20: one bank = one sequence state inside an engine (guarded by q_lock)
typedef struct Bank {
    long n_req;
    double t_last;
    char *last;
} Bank;
static Slot *slots;
static int n_slots = 1, n_banks = 1;
// BATCH-NATIVE (--native-batch N, 2..8; default 0 = off): a /v1/decide batch goes to the engine as
// {"decide_batch":[up to N item lines]} (one engine forward a step over the N sequences, LLB_BATCH=N);
// each item still gets its own reply line, read and answered as run_decide does
static int native_batch;
// BATCH-NATIVE-2 (--native-cont, with --native-batch N; default off): the batch goes in lines of up to
// 256 good items, and the engine admits the next item into a row as soon as one frees (a step boundary)
static int native_cont;
static long affinity_min = 256;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cond = PTHREAD_COND_INITIALIZER;
static unsigned long q_tail; // FIFO tickets: next to issue, next to serve
static atomic_int ready, engine_dead, inflight, queued, stopping, closing; // closing: past the drain
static atomic_long req_seq;
static time_t t_start;
static int wake[2];
static char **engine_argv;
static atomic_long mm_seq;
static mm_caps mmc; // G5 multimodal content caps + what the engine takes (--mm-*)
// G41: the TTS child (--tts DIR). tts_lock serialises requests on its pipes.
static const char *tts_dir;
static pid_t tts_pid;
static FILE *tts_in, *tts_out;
static atomic_int tts_ready, tts_dead;
static pthread_mutex_t tts_lock = PTHREAD_MUTEX_INITIALIZER;
// EARS-CANON: the ear child (--ear DIR). ear_lock serialises requests on its pipes.
static const char *ear_dir;
// FOLD-CONTINUE MIMG: --vis-tower CLI --vis-mmproj MMPROJ (both, default off = the old behaviour): every image of a
// chat/completions request goes through the Qwen3.5 tower (CLI --batch, outside the engine lock), its rows are stamped
// (llb-tower-rows-v1, bound to the mmproj's sha, which the engine children get as LLB_VIS_MMPROJ_SHA256), its one
// <|image_pad|> placeholder becomes nx*ny pads and the engine line carries "images":[{rows,nx,ny},...] (dual.c), in order.
static const char *vis_tower, *vis_mmproj;
static char vis_mmsha[65];
static atomic_long vis_seq;
static int ear_llm_cpu;
static pid_t ear_pid;
static FILE *ear_in, *ear_out;
static atomic_int ear_ready, ear_dead;
static pthread_mutex_t ear_lock = PTHREAD_MUTEX_INITIALIZER;

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

// every engine has loaded or died. Also re-checked by the reaper: an engine that dies
// before another's ready line (R5: the 2nd engine's metrics bind) left the server "loading"
static void ready_check(void) {
    int all = 1;
    for (int i = 0; i < n_slots; i++) all &= slots[i].ready || slots[i].dead;
    if (all) ready = 1;
}

static void *engine_err(void *p) {
    ErrArg a = *(ErrArg *)p;
    free(p);
    char *line = NULL, tag[32] = "engine: ";
    if (n_slots > 1) snprintf(tag, sizeof tag, "engine[%d]: ", a.s->idx);
    size_t cap = 0;
    while (getline(&line, &cap, a.f) > 0) {
        // the engine is up once dual.c's init line appears (after the model load)
        if (strstr(line, "sys1: FP32 clef head + tokenizer ready")) {
            a.s->ready = 1;
            ready_check();
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
    fclose(a.f);
    return NULL;
}

static int decide_path; // --decide-path (G15); --exact = the default
static int prefill_fast, prefix_cache; // explicit opt-ins, independent of --exact

// S6-GUARD: the load-time pre-check for --decide-path. Reads the GGUF's tensor table (no weights) and
// returns 0 when every GEMM the fast chunks run (host/llb_pf.h pf_mm) has a fast format: Q4_K / Q5_K /
// Q6_K, or Q8_0 with an input width that is a multiple of 128, and every input width a multiple of 256
// (pf_q8). Otherwise 1, with the first offender in why. The 9B clef-flash Q4_K_S fails here on its F32
// ssm_alpha / ssm_beta unless LLB_PF_SSMQ8=1 (SI-FASTDECIDE / SHIP45: the engine's fast tier reads a
// Q8_0 shadow of them); the engine has the same check (pf_fast) as the backstop.
static int gg_rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }
static int gg_skip(FILE *f, uint64_t n) { return fseeko(f, (off_t)n, SEEK_CUR); }
static int gg_skip_val(FILE *f, uint32_t t) {
    static const uint8_t sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    uint64_t n;
    if (t == 8) return gg_rd(f, &n, 8) || gg_skip(f, n);
    if (t == 9) {
        uint32_t et;
        if (gg_rd(f, &et, 4) || gg_rd(f, &n, 8) || et > 12) return -1;
        if (et != 8 && et != 9) return gg_skip(f, n * sz[et]);
        for (uint64_t i = 0; i < n; i++)
            if (gg_skip_val(f, et)) return -1;
        return 0;
    }
    return t > 12 ? -1 : gg_skip(f, sz[t]);
}
static int fast_unsupported(const char *path, char *why, size_t wn) {
    static const char *gemm[] = {"ffn_gate", "ffn_up", "ffn_down", "attn_qkv", "attn_gate", "ssm_alpha",
                                 "ssm_beta", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output"};
    FILE *f = fopen(path, "rb");
    uint32_t magic = 0, ver;
    uint64_t nt, nkv, n;
    if (!f || gg_rd(f, &magic, 4) || magic != 0x46554747u || gg_rd(f, &ver, 4) || gg_rd(f, &nt, 8) || gg_rd(f, &nkv, 8)) {
        snprintf(why, wn, "cannot read the GGUF tensor table");
        if (f) fclose(f);
        return 1;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        uint32_t t;
        if (gg_rd(f, &n, 8) || gg_skip(f, n) || gg_rd(f, &t, 4) || gg_skip_val(f, t)) goto bad;
    }
    for (uint64_t i = 0; i < nt; i++) {
        char name[256];
        uint32_t nd, ty;
        uint64_t ne[4] = {0}, off;
        if (gg_rd(f, &n, 8) || n >= sizeof name || gg_rd(f, name, n) || gg_rd(f, &nd, 4) || nd < 1 || nd > 4 ||
            gg_rd(f, ne, 8 * nd) || gg_rd(f, &ty, 4) || gg_rd(f, &off, 8)) goto bad;
        name[n] = 0;
        char *s = strncmp(name, "blk.", 4) ? NULL : strchr(name + 4, '.');
        if (!s) continue;
        s++;
        size_t sl = strlen(s);
        if (sl < 8 || strcmp(s + sl - 7, ".weight")) continue;
        int hit = 0;
        for (size_t g = 0; g < sizeof gemm / sizeof *gemm; g++)
            hit |= strlen(gemm[g]) == sl - 7 && !strncmp(s, gemm[g], sl - 7);
        if (!hit) continue;
        // ggml types: 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K
        // SI-FASTDECIDE: with LLB_PF_SSMQ8=1 the engine's fast tier shadows F32 ssm_alpha / ssm_beta as Q8_0
        const char *sq = getenv("LLB_PF_SSMQ8");
        int sq8 = ty == 0 && sq && atoi(sq) == 1 && (!strncmp(s, "ssm_alpha.", 10) || !strncmp(s, "ssm_beta.", 9));
        int ok = (ty == 12 || ty == 13 || ty == 14 || ((ty == 8 || sq8) && ne[0] % 128 == 0)) && ne[0] % 256 == 0;
        if (!ok) {
            snprintf(why, wn, "%s is ggml type %u (width %llu)%s", name, ty, (unsigned long long)ne[0],
                     ty == 0 ? " = F32" : ty == 1 ? " = F16" : ty == 30 ? " = BF16" : "");
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
bad:
    snprintf(why, wn, "cannot read the GGUF tensor table");
    fclose(f);
    return 1;
}

static void engine_start(Slot *s, char **argv) {
    int in[2], out[2], err[2];
    if (pipe(in) || pipe(out) || pipe(err)) perror("pipe"), exit(1);
    pid_t engine_pid = s->pid = fork();
    if (engine_pid < 0) perror("fork"), exit(1);
    if (!engine_pid) {
        // children must not hold the other slots' pipe ends (EOF on shutdown)
        /* close_extra_fds below closes all unrelated descriptors safely. */
        dup2(in[0], 0), dup2(out[1], 1), dup2(err[1], 2);
        close(in[0]), close(in[1]), close(out[0]), close(out[1]), close(err[0]), close(err[1]);
        close_extra_fds(); // HARDEN
        setenv("LLB_SYS1", "1", 1);
        setenv("LLB_SYS1_STREAM", "1", 1);
        // dual.c / sys1.c refuse these
        unsetenv("LLB_SPEC"), unsetenv("LLB_MTP"), unsetenv("LLB_DUMP_HIDDEN");
        if (decide_path) { // G15 decide path (docs/reference/decide-configuration.md): fast prefill, prefix cache, greedy default
            setenv("LLB_PREFILL", "fast", 1), setenv("LLB_PREFIX_CACHE", "1", 1);
            setenv("LLB_SAMPLE", "preset=greedy", 0);
        } else {
            if (prefill_fast) setenv("LLB_PREFILL", "fast", 1);
            else unsetenv("LLB_PREFILL");
            if (prefix_cache) setenv("LLB_PREFIX_CACHE", "1", 1);
            else unsetenv("LLB_PREFIX_CACHE");
        }
        if (metrics_base) { // R5-FIX: one listener port per engine (the same port: every engine but one died on bind)
            char mp[16];
            snprintf(mp, sizeof mp, "%d", metrics_base + s->idx);
            setenv("LLB_METRICS_PORT", mp, 1);
        }
        execv(argv[0], argv);
        perror("execv engine");
        _exit(127);
    }
    close(in[0]), close(out[1]), close(err[1]);
    s->in = fdopen(in[1], "w");
    s->out = fdopen(out[0], "r");
    setvbuf(s->out, NULL, _IONBF, 0);
    ErrArg *a = malloc(sizeof *a);
    *a = (ErrArg){s, fdopen(err[0], "r")};
    pthread_create(&s->err_thread, NULL, engine_err, a);
}

// ---- G41: TTS child ---------------------------------------------------------

static void *tts_err(void *p) {
    FILE *f = p;
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        if (strstr(line, "kokoro: ready")) tts_ready = 1;
        flockfile(stderr);
        fputs("tts: ", stderr);
        fputs(line, stderr);
        funlockfile(stderr);
    }
    free(line);
    tts_dead = 1;
    return NULL;
}

// python3 kokoro_tts.py worker --dir DIR; the script defaults to the one beside the llb-serve binary
static void tts_start(int listen_fd) {
    static char script[4096];
    const char *py = getenv("LLB_TTS_PYTHON"), *sc = getenv("LLB_TTS_SCRIPT");
    if (!py) py = "python3";
    if (!sc) {
        ssize_t n = readlink("/proc/self/exe", script, sizeof script - 32);
        if (n <= 0) fprintf(stderr, "serve: --tts: cannot find kokoro_tts.py (set LLB_TTS_SCRIPT)\n"), exit(2);
        script[n] = 0;
        char *sl = strrchr(script, '/');
        strcpy(sl ? sl + 1 : script, "kokoro_tts.py");
        sc = script;
    }
    if (access(sc, R_OK)) fprintf(stderr, "serve: --tts: %s: %s (set LLB_TTS_SCRIPT)\n", sc, strerror(errno)), exit(2);
    int in[2], out[2], err[2];
    if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC) || pipe2(err, O_CLOEXEC)) perror("pipe"), exit(1);
    tts_pid = fork();
    if (tts_pid < 0) perror("fork"), exit(1);
    if (!tts_pid) {
        for (int i = 0; i < n_slots; i++) fclose(slots[i].in), fclose(slots[i].out);
        close(listen_fd);
        dup2(in[0], 0), dup2(out[1], 1), dup2(err[1], 2);
        close(in[0]), close(in[1]), close(out[0]), close(out[1]), close(err[0]), close(err[1]);
        close_extra_fds(); // HARDEN (after dup2: it closes the pipe originals)
        execlp(py, py, sc, "worker", "--dir", tts_dir, (char *)NULL);
        perror("exec tts");
        _exit(127);
    }
    close(in[0]), close(out[1]), close(err[1]);
    tts_in = fdopen(in[1], "w");
    tts_out = fdopen(out[0], "r");
    pthread_t t;
    pthread_create(&t, NULL, tts_err, fdopen(err[0], "r"));
    pthread_detach(t);
    logf_("tts: %s %s worker --dir %s pid=%d (loading)", py, sc, tts_dir, (int)tts_pid);
}

// One request/reply on the TTS child: a JSON line out, a JSON header line back (+ "bytes" raw bytes).
// Returns the parsed header (caller frees) or NULL with *err set; *payload is malloc'd (or NULL).
static jv *tts_call(const char *req, char **payload, size_t *n, const char **err) {
    *payload = NULL, *n = 0;
    if (tts_dead) return *err = "the speech worker is not running", NULL;
    if (!tts_ready) return *err = "Loading voice", NULL;
    pthread_mutex_lock(&tts_lock);
    jv *h = NULL;
    char *line = NULL;
    size_t cap = 0;
    if (fputs(req, tts_in) < 0 || fputc('\n', tts_in) < 0 || fflush(tts_in) || getline(&line, &cap, tts_out) <= 0) {
        tts_dead = 1;
        *err = "the speech worker went away";
    } else if (!(h = json_parse(line, err))) {
        *err = "the speech worker sent a bad reply";
    } else {
        const jv *b = json_get(h, "bytes");
        size_t want = b && b->type == J_NUM && b->num > 0 ? (size_t)b->num : 0;
        if (want) {
            *payload = malloc(want);
            if (fread(*payload, 1, want, tts_out) != want) {
                tts_dead = 1;
                *err = "the speech worker went away mid-reply";
                free(*payload), *payload = NULL;
                json_free(h), h = NULL;
            } else *n = want;
        }
    }
    pthread_mutex_unlock(&tts_lock);
    free(line);
    return h;
}

// ---- EARS-CANON: the ear child ------------------------------------------------

static void *ear_err(void *p) {
    FILE *f = p;
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        if (strstr(line, "ear: ready")) ear_ready = 1;
        flockfile(stderr);
        fputs(strncmp(line, "ear: ", 5) ? "ear: " : "", stderr);
        fputs(line, stderr);
        funlockfile(stderr);
    }
    free(line);
    ear_dead = 1;
    return NULL;
}

// llb-ear worker --llm DIR/Qwen3-ASR-0.6B-Q8_0.gguf --mmproj DIR/mmproj-Qwen3-ASR-0.6B-bf16.gguf; the binary defaults
// to the one beside llb-serve (LLB_EAR_BIN overrides; LLB_EAR_LLM / LLB_EAR_MMPROJ override the model files)
static void ear_start(int listen_fd) {
    static char bin[4096], llm[4096], mmp[4096];
    const char *b = getenv("LLB_EAR_BIN");
    if (!b) {
        ssize_t n = readlink("/proc/self/exe", bin, sizeof bin - 32);
        if (n <= 0) fprintf(stderr, "serve: --ear: cannot find llb-ear (set LLB_EAR_BIN)\n"), exit(2);
        bin[n] = 0;
        char *sl = strrchr(bin, '/');
        strcpy(sl ? sl + 1 : bin, "llb-ear");
        b = bin;
    }
    const char *l = getenv("LLB_EAR_LLM"), *m = getenv("LLB_EAR_MMPROJ");
    if (!l) snprintf(llm, sizeof llm, "%s/Qwen3-ASR-0.6B-Q8_0.gguf", ear_dir), l = llm;
    if (!m) snprintf(mmp, sizeof mmp, "%s/mmproj-Qwen3-ASR-0.6B-bf16.gguf", ear_dir), m = mmp;
    const char *need[] = {b, l, m};
    for (int i = 0; i < 3; i++)
        if (access(need[i], R_OK)) fprintf(stderr, "serve: --ear: %s: %s\n", need[i], strerror(errno)), exit(2);
    int in[2], out[2], err[2];
    if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC) || pipe2(err, O_CLOEXEC)) perror("pipe"), exit(1);
    ear_pid = fork();
    if (ear_pid < 0) perror("fork"), exit(1);
    if (!ear_pid) {
        for (int i = 0; i < n_slots; i++) fclose(slots[i].in), fclose(slots[i].out);
        close(listen_fd);
        dup2(in[0], 0), dup2(out[1], 1), dup2(err[1], 2);
        close(in[0]), close(in[1]), close(out[0]), close(out[1]), close(err[0]), close(err[1]);
        close_extra_fds(); // HARDEN (after dup2: it closes the pipe originals)
        if (ear_llm_cpu) execl(b, b, "worker", "--llm", l, "--mmproj", m, "--llm-cpu", (char *)NULL);
        else execl(b, b, "worker", "--llm", l, "--mmproj", m, (char *)NULL);
        perror("exec ear");
        _exit(127);
    }
    close(in[0]), close(out[1]), close(err[1]);
    ear_in = fdopen(in[1], "w");
    ear_out = fdopen(out[0], "r");
    pthread_t t;
    pthread_create(&t, NULL, ear_err, fdopen(err[0], "r"));
    pthread_detach(t);
    logf_("ear: %s worker --llm %s --mmproj %s%s pid=%d (loading)", b, l, m, ear_llm_cpu ? " --llm-cpu" : "", (int)ear_pid);
}

// One transcription on the ear child (a staged file + its sha256). Returns the parsed reply (caller frees; check "ok")
// or NULL with *err set and *status 503.
static jv *ear_call(const char *path, const char *sha, int n, const char **err, int *status) {
    *status = 503;
    if (ear_dead) return *err = "the ear is not running", NULL;
    if (!ear_ready) return *err = "Loading ear", NULL;
    sb q = {0};
    sb_add(&q, "{\"op\":\"transcribe\",\"path\":");
    json_quote(&q, path);
    sb_add(&q, ",\"sha256\":");
    json_quote(&q, sha);
    if (n > 0) sb_fmt(&q, ",\"n\":%d", n);
    sb_addc(&q, '}');
    pthread_mutex_lock(&ear_lock);
    jv *h = NULL;
    char *line = NULL;
    size_t cap = 0;
    if (fputs(q.p, ear_in) < 0 || fputc('\n', ear_in) < 0 || fflush(ear_in) || getline(&line, &cap, ear_out) <= 0) {
        ear_dead = 1;
        *err = "the ear went away";
    } else if (!(h = json_parse(line, err))) {
        *status = 502, *err = "the ear sent a bad reply";
    }
    pthread_mutex_unlock(&ear_lock);
    free(line), free(q.p);
    return h;
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

typedef struct Waiter { struct Waiter *next; } Waiter;
static Waiter *wait_first, *wait_last;
static int client_gone(int fd) { char ch; ssize_t n = recv(fd, &ch, 1, MSG_PEEK | MSG_DONTWAIT); return n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR); }
static Slot *slot_acquire(const char *prompt, unsigned long *ticket, int *bank, int fd) {
    pthread_mutex_lock(&q_lock);
    *ticket = q_tail++;
    Waiter me = {0}; if (wait_last) wait_last->next = &me; else wait_first = &me; wait_last = &me;
    Slot *s = NULL;
    double t_best = 0;
    *bank = 0;
    for (;;) {
        if (all_dead() || closing || client_gone(fd)) break;
        if (wait_first == &me) {
            long best = -1;
            for (int i = 0; i < n_slots; i++) {
                Slot *x = &slots[i];
                if (x->busy || x->dead || !x->ready) continue;
                for (int b = 0; b < n_banks; b++) {
                    const char *last = n_banks > 1 ? x->bk[b].last : x->last;
                    double tl = n_banks > 1 ? x->bk[b].t_last : x->t_last;
                    long k = (long)lcp(last, prompt);
                    if (n_banks > 1 && k < affinity_min) k = 0; // G20: a shared chat-template head is not affinity
                    if (!s || k > best || (k == best && tl < t_best)) s = x, best = k, t_best = tl, *bank = b;
                }
            }
            if (s) break;
        }
        struct timespec until; clock_gettime(CLOCK_REALTIME, &until);
        until.tv_nsec += 100000000; if (until.tv_nsec >= 1000000000) { until.tv_sec++; until.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&q_cond, &q_lock, &until);
    }
    Waiter **p = &wait_first; while (*p && *p != &me) p = &(*p)->next;
    if (*p) *p = me.next;
    if (wait_last == &me) { wait_last = wait_first; while (wait_last && wait_last->next) wait_last = wait_last->next; }
    if (s) s->busy = 1, s->t_last = now(); // HARDEN: the watchdog baseline
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_lock);
    return s;
}

static void slot_restart(Slot *s) {
    pthread_mutex_lock(&q_lock);
    pid_t pid = s->pid; s->ready = 0;
    if (pid > 0) { kill(pid, SIGKILL); int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} s->pid = -pid; }
    pthread_mutex_unlock(&q_lock);
    pthread_join(s->err_thread, NULL); s->ready = 0;
    fclose(s->in); fclose(s->out);
    pthread_mutex_lock(&q_lock);
    free(s->last); s->last = NULL;
    if (s->bk) for (int b = 0; b < n_banks; b++) { free(s->bk[b].last); s->bk[b].last = NULL; }
    s->dead = 0; engine_start(s, engine_argv); engine_dead = 0;
    pthread_cond_broadcast(&q_cond); pthread_mutex_unlock(&q_lock);
}
static int wait_engine_line(Slot *s, int fd) {
    struct pollfd fds[2] = {{.fd = fileno(s->out), .events = POLLIN}, {.fd = fd, .events = POLLIN}};
    while (!client_gone(fd)) { int n = poll(fds, 2, 100); if (n < 0 && errno != EINTR) return 0; if (n > 0 && (fds[0].revents & (POLLIN | POLLHUP | POLLERR))) return 1; }
    return 0;
}

static void slot_release(Slot *s, int bank, const char *prompt) {
    pthread_mutex_lock(&q_lock);
    s->busy = 0;
    s->n_req++;
    s->t_last = now();
    free(s->last);
    s->last = strdup(prompt);
    if (n_banks > 1) {
        Bank *b = &s->bk[bank];
        b->n_req++;
        b->t_last = s->t_last;
        free(b->last);
        b->last = strdup(prompt);
    }
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_lock);
}

// ---- G24 F2: the glue run-ahead worker -------------------------------------
// --glue-pack DIR --glue-rows glue_rows.py [--glue-python PY] [--glue-cache DIR]: one resident
// `PY -I glue_rows.py worker` (SigLIP + connector, feature/row cache) started beside the engine.
// llb_image {"image":PATH} or {"image_b64":B64} is sent to it as one JSON line; its reply
// {"image":{"rows","n"},...} (stamped rows, bound to the pack's trunk) or {"error":MSG} comes
// back as one line. Requests that already carry {"rows","n"} skip it (the engine checks the stamp).
static const char *glue_pack, *glue_rows, *glue_py = "python3", *glue_cache;
static FILE *glue_in, *glue_out;
static pid_t glue_pid;
static atomic_int glue_ready, glue_dead;
static pthread_mutex_t glue_lock = PTHREAD_MUTEX_INITIALIZER;

static void *glue_err(void *p) {
    FILE *f = p;
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        if (strstr(line, "glue worker ready")) glue_ready = 1;
        flockfile(stderr);
        fputs("glue: ", stderr);
        fputs(line, stderr);
        funlockfile(stderr);
    }
    glue_dead = 1;
    free(line);
    return NULL;
}

static void glue_start(void) {
    int in[2], out[2], err[2];
    if (pipe(in) || pipe(out) || pipe(err)) perror("pipe"), exit(1);
    glue_pid = fork();
    if (glue_pid < 0) perror("fork"), exit(1);
    if (!glue_pid) {
        dup2(in[0], 0), dup2(out[1], 1), dup2(err[1], 2);
        close(in[0]), close(in[1]), close(out[0]), close(out[1]), close(err[0]), close(err[1]);
        close_extra_fds(); // HARDEN
        char *av[] = {(char *)glue_py, "-I", (char *)glue_rows, "worker", "--pack", (char *)glue_pack, "--gguf",
                      getenv("LLB_GGUF"), "--cache", (char *)glue_cache, NULL};
        execvp(av[0], av);
        perror("execvp glue worker");
        _exit(127);
    }
    close(in[0]), close(out[1]), close(err[1]);
    glue_in = fdopen(in[1], "w");
    glue_out = fdopen(out[0], "r");
    pthread_t t;
    pthread_create(&t, NULL, glue_err, fdopen(err[0], "r"));
    pthread_detach(t);
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
    int cancelled; // HARDEN: the client is gone; keep draining to fin, stop streaming
    int slot;   // G6: the slot that ran it (-1 = none)
    char xh[128]; // G6: extra response headers (N>1 only: slot, ticket, queue wait)
    int close;      // G31: close after this response (else keep-alive)
    char cors[320]; // G31: Access-Control-* response headers for an allowed Origin
    char acrh[256]; // G31: the preflight's Access-Control-Request-Headers
    sb *cap;        // SERVE-BATCH: a batch item: the response body is kept here (not sent), status in cap_status
    int cap_status;
} Conn;

static const char *reason(int s) {
    switch (s) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

static void respond_n(Conn *c, int status, const char *ctype, const char *body, size_t n, const char *extra) {
    char h[1024], ka[48];
    if (c->cap) { // SERVE-BATCH: a batch item's response goes into the batch, not onto the socket
        sb_addn(c->cap, body, n);
        c->cap_status = status;
        return;
    }
    if (stopping || c->broken) c->close = 1;
    snprintf(ka, sizeof ka, "keep-alive\r\nKeep-Alive: timeout=%d", keepalive_s);
    snprintf(h, sizeof h,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: %s\r\nServer: llb-serve\r\n%s%s%s\r\n",
        status, reason(status), ctype, n, c->close ? "close" : ka, extra ? extra : "", c->xh, c->cors);
    c->status = status;
    sb o = {0}; // one send: a kept-alive client must not wait on a delayed ACK for the body
    sb_add(&o, h);
    sb_addn(&o, body, n); // binary-safe (G41 audio)
    if (send_all(c->fd, o.p, o.n)) c->broken = 1;
    c->bytes += (long)o.n;
    free(o.p);
}

static void respond(Conn *c, int status, const char *ctype, const char *body, const char *extra) {
    respond_n(c, status, ctype, body, strlen(body), extra);
}

static void json_reply(Conn *c, int status, const char *body) {
    respond(c, status, "application/json; charset=utf-8", body, NULL);
}

// the one error shape (OpenAI's, code = the HTTP status), for replies and SSE events
static void err_json_p(sb *e, int status, const char *type, const char *msg, const char *param) {
    sb_add(e, "{\"error\":{\"message\":");
    json_quote(e, msg);
    sb_fmt(e, ",\"type\":\"%s\",\"param\":", type);
    if (param && *param) json_quote(e, param);
    else sb_add(e, "null");
    sb_fmt(e, ",\"code\":%d}}", status);
}

static void err_json(sb *e, int status, const char *type, const char *msg) { err_json_p(e, status, type, msg, NULL); }

static void err_reply_p(Conn *c, int status, const char *type, const char *msg, const char *param) {
    sb e = {0};
    err_json_p(&e, status, type, msg, param);
    json_reply(c, status, e.p);
    free(e.p);
}

static void err_reply(Conn *c, int status, const char *type, const char *msg) { err_reply_p(c, status, type, msg, NULL); }

static void sse_open(Conn *c) {
    char h[1024];
    snprintf(h, sizeof h, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    "Connection: close\r\nX-Accel-Buffering: no\r\nX-LLB-Sampler: w1-sampler\r\nServer: llb-serve\r\n"
                    "X-LLB-Prefill: %s\r\n%s%s\r\n", (decide_path || prefill_fast) ? "fast" : "exact", c->xh, c->cors);
    c->status = 200;
    c->close = 1; // an SSE body ends at the close
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
    StReq st;     // G16: the structured part (st.on = 0 for a plain request)
    int status;   // error status (0 = 400)
    mm_err mme;   // the multimodal error, when status came from mm.c
    mm_list media; // G5: images/audio in prompt order (one placeholder each in prompt)
    char **paths;  // their scratch files, handed to the engine (unlinked after the request)
    char *img;    // G24 F2: the engine's "image" object made by the glue worker (else llb_image as sent)
    char *glue;   // G24 F2: the worker's reply, echoed in "llb":{"glue":...}
    char *vis;    // MIMG: the engine's "images" array (tower rows), else NULL
    char *visdir; // MIMG: its scratch dir (rows + stamps), removed in media_done
} Req;

static char *trim(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    return strndup(s, n);
}

// the model's Qwen3.5 chat template (flash-chat_template.jinja), the no-tools subset
static const char *render_chat(const jv *body, sb *o, Req *rq) {
    const jv *ms = json_get(body, "messages");
    if (!ms || ms->type != J_ARR || !ms->n) return "messages must be a non-empty array";
    const jv *tl = json_get(body, "tools"); // G16: with --structured, tools: [] = no tools
    if ((tl && !(structured_on && tl->type == J_ARR && !tl->n)) || json_get(body, "functions"))
        return "tools/function calling is not supported yet";
    int thinking = 1;
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
        const char *role = json_get(ms->items[i], "role")->str;
        sb tmp = {0};
        // G5: text parts verbatim, each image -> <|vision_start|><|image_pad|><|vision_end|> (mm.c)
        if (mm_render_content(json_get(ms->items[i], "content"), (int)i, !strcmp(role, "system"), &mmc, &rq->media,
                &tmp, &rq->mme))
            return free(tmp.p), rq->status = rq->mme.status, rq->mme.msg;
        char *content = trim(tmp.p ? tmp.p : "");
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
    if (x->type != J_NUM || x->num < 1 || x->num > 65536 || x->num != (double)(long)x->num)
        return *err = "max_tokens must be an integer 1..65536", -1;
    *v = (int)x->num;
    return 1;
}

// 0 ok (r->img / r->glue set), else the HTTP status with *msg
static int glue_run(const jv *body, Req *r, const char **msg) {
    static char m[512];
    const jv *im = json_get(body, "llb_image");
    if (!im || im->type != J_OBJ || (!json_get(im, "image") && !json_get(im, "image_b64"))) return 0;
    if (!glue_pack) return *msg = "llb_image.image needs the server's --glue-pack", 400;
    if (glue_dead) return *msg = "the glue worker is not running", 503;
    if (!glue_ready) return *msg = "Loading glue worker", 503;
    sb o = {0};
    json_dump(&o, im);
    char *line = NULL;
    size_t cap = 0;
    pthread_mutex_lock(&glue_lock);
    int ok = fprintf(glue_in, "%s\n", o.p) > 0 && !fflush(glue_in) && getline(&line, &cap, glue_out) > 0;
    pthread_mutex_unlock(&glue_lock);
    free(o.p);
    if (!ok) return free(line), glue_dead = 1, *msg = "the glue worker died", 503;
    const char *err = NULL;
    jv *rep = json_parse(line, &err);
    const jv *e = rep ? json_get(rep, "error") : NULL, *img = rep ? json_get(rep, "image") : NULL;
    int st = 0;
    if (!rep || (!e && (!img || img->type != J_OBJ))) *msg = "the glue worker replied garbage", st = 502;
    else if (e) snprintf(m, sizeof m, "glue: %s", e->type == J_STR ? e->str : "refused"), *msg = m, st = 400;
    else {
        sb a = {0}, b = {0};
        json_dump(&a, img), json_dump(&b, rep);
        r->img = a.p, r->glue = b.p;
    }
    json_free(rep);
    free(line);
    return st;
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
    int thinking = 1, agp = 1;
    const jv *kw = json_get(body, "chat_template_kwargs");
    if ((x = kw ? json_get(kw, "enable_thinking") : NULL) && x->type == J_BOOL) thinking = x->b;
    if ((x = json_get(body, "add_generation_prompt")) && x->type == J_BOOL) agp = x->b;
    if ((err = st_request(body, chat, thinking, agp, structured_on, &r->st))) return err;
    if (chat && r->st.prompt) { // G16: tools / tool history: the tool-capable template port
        r->chat = 1;
        r->special = 1;
        sb_add(&p, r->st.prompt);
    } else if (chat) {
        r->chat = 1;
        r->special = 1; // the template's <|im_start|>/<|im_end|> are control tokens
        if ((err = render_chat(body, &p, r))) return free(p.p), err;
        if (mm_check_engine(&r->media, &mmc, &r->mme)) return free(p.p), r->status = r->mme.status, r->mme.msg;
    } else {
        x = json_get(body, "prompt");
        if (x && x->type == J_ARR && x->n == 1) x = x->items[0];
        if (!x || x->type != J_STR) return "prompt must be a string (or a one-element array of strings)";
        if (!*x->str) return "prompt must not be empty";
        sb_add(&p, x->str);
    }
    r->prompt = p.p;
    return NULL;
}

// the dual.c generate request line (sampling knobs passed through, unused yet)
static char *engine_line(const jv *body, const Req *r, int bank) {
    sb o = {0};
    sb_add(&o, "{\"mode\":\"generate\",\"prompt\":");
    json_quote(&o, r->prompt);
    sb_fmt(&o, ",\"n\":%d", r->n);
    if (n_banks > 1) sb_fmt(&o, ",\"bank\":%d", bank); // G20: the engine's sequence state
    if (r->special) sb_add(&o, ",\"special\":true");
    if (r->st.gbnf) { // G16: dual.c's G1 field; constrained from the first generated token
        sb_add(&o, ",\"grammar\":");
        json_quote(&o, r->st.gbnf);
    }
    if (r->media.n) { // G5 engine media hook (docs/reference/multimodal-content.md): one entry per placeholder, in prompt order
        sb_add(&o, ",\"media\":[");
        for (size_t i = 0; i < r->media.n; i++) {
            const mm_item *m = &r->media.v[i];
            sb_fmt(&o, "%s{\"kind\":\"%s\",\"path\":", i ? "," : "", m->kind == MM_IMAGE ? "image" : "audio");
            json_quote(&o, r->paths && r->paths[i] ? r->paths[i] : ""); // HARDEN
            sb_fmt(&o, ",\"mime\":\"%s\",\"bytes\":%zu,\"sha256\":\"%s\"", m->mime, m->n, m->sha256);
            if (m->kind == MM_IMAGE) sb_fmt(&o, ",\"w\":%u,\"h\":%u,\"detail\":\"%s\"", m->w, m->h, m->detail);
            sb_addc(&o, '}');
        }
        sb_addc(&o, ']');
    }
    // G4 F1 (non-standard): connector rows + logit readout, validated by the engine (dual.c)
    const jv *x;
    if (r->vis) sb_add(&o, ",\"images\":"), sb_add(&o, r->vis); // MIMG: tower rows, dual.c "images"
    if (r->img) sb_add(&o, ",\"image\":"), sb_add(&o, r->img);
    else if ((x = json_get(body, "llb_image"))) sb_add(&o, ",\"image\":"), json_dump(&o, x);
    if ((x = json_get(body, "llb_readout"))) sb_add(&o, ",\"readout\":"), json_dump(&o, x);
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
    // G16, a structured request
    LlbTcParser *tp;          // chat: the reasoning / tool-call parser
    int stopped, ncalls;      // an end-of-generation id was seen; tool calls so far
    int n_eog;                // ids before the end-of-generation id (unstructured requests)
    double t_eog;
    sb content, reasoning, calls; // the message parts (calls = the tool_calls JSON items)
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

// ---- G16: the response side of a structured request ---------------------------

// one chat delta ({"content"|"reasoning_content"|"tool_calls": ...}); role on the first
static void stream_delta(Gen *g, const char *inner) {
    sb o = {0};
    chunk_head(g, &o);
    sb_add(&o, "{\"index\":0,\"delta\":{");
    if (!g->nchunks) sb_add(&o, "\"role\":\"assistant\",");
    sb_add(&o, inner);
    sb_add(&o, "},\"logprobs\":null,\"finish_reason\":null}]}");
    sse(g->c, o.p);
    g->nchunks++;
    free(o.p);
}

// drain the parser's events into the message parts (and the stream, as deltas: a tool
// call goes out whole, once its </function> is seen)
static void st_events(Gen *g) {
    LlbTcEvent ev;
    while (llb_tcp_next(g->tp, &ev)) {
        sb d = {0};
        if ((ev.type == LLB_TCE_CONTENT || ev.type == LLB_TCE_REASONING) && ev.len) {
            int rs = ev.type == LLB_TCE_REASONING;
            sb_addn(rs ? &g->reasoning : &g->content, ev.text, ev.len);
            sb_add(&d, rs ? "\"reasoning_content\":" : "\"content\":");
            json_quote(&d, ev.text);
        } else if (ev.type == LLB_TCE_TOOL_CALL) {
            sb call = {0};
            sb_fmt(&call, "{\"id\":\"call_%s_%d\",\"type\":\"function\",\"function\":{\"name\":", g->id, ev.index);
            json_quote(&call, ev.name);
            sb_add(&call, ",\"arguments\":");
            json_quote(&call, ev.args);
            sb_add(&call, "}}");
            if (g->ncalls++) sb_addc(&g->calls, ',');
            sb_add(&g->calls, call.p);
            sb_fmt(&d, "\"tool_calls\":[{\"index\":%d,", ev.index);
            sb_add(&d, call.p + 1);
            sb_addc(&d, ']');
            free(call.p);
        } else if (ev.type == LLB_TCE_ERROR)
            logf_("#%ld tool-call parse: %s (its raw text is kept as content)", g->c->id, ev.text);
        if (d.p && g->r->stream) stream_delta(g, d.p);
        free(d.p);
    }
}

// one decoded piece (before the end-of-generation id)
static void st_piece(Gen *g, const char *piece) {
    if (g->tp) {
        llb_tcp_feed(g->tp, piece, (int)strlen(piece));
        st_events(g);
        return;
    }
    sb_add(&g->content, piece); // completions: raw text
    if (g->r->stream && *piece) stream_piece(g, piece);
}

static void int_array_n(sb *o, const jv *a, size_t lim) {
    sb_addc(o, '[');
    for (size_t i = 0; a && i < a->n && i < lim; i++) sb_fmt(o, i ? ",%s" : "%s", a->items[i]->raw);
    sb_addc(o, ']');
}

// the end of a structured request: finish_reason tool_calls | stop (EOG) | length;
// usage counts the generated tokens up to and including the EOG id
static void st_finish(Gen *g, const jv *fin, double t0, double t1, double t2) {
    Conn *c = g->c;
    const Req *r = g->r;
    const jv *pids = json_get(fin, "prompt_ids"), *ids = json_get(fin, "ids"), *txt = json_get(fin, "text");
    int pn = pids ? (int)pids->n : 0, gn = c->gen_n;
    c->prompt_n = pn;
    if (!g->stopped) { // max_tokens ran out: a held-back UTF-8 tail, if any
        const char *full = txt && txt->type == J_STR ? txt->str : "";
        size_t have = g->text.p ? g->text.n : 0;
        if (strlen(full) > have && !strncmp(full, g->text.p ? g->text.p : "", have)) st_piece(g, full + have);
    }
    if (g->tp) {
        llb_tcp_finish(g->tp);
        st_events(g);
    }
    const char *fr = g->ncalls ? "tool_calls" : g->stopped ? "stop" : "length";
    sb o = {0};
    if (r->stream) {
        chunk_head(g, &o);
        if (r->chat) sb_fmt(&o, "{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
        else sb_fmt(&o, "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
        usage_timings(&o, pn, gn, t0, t1, t2);
        sb_addc(&o, '}');
        sse(c, o.p);
        if (r->usage_chunk) {
            free(o.p);
            o = (sb){0};
            chunk_head(g, &o);
            sb_fmt(&o, "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}", pn, gn, pn + gn);
            sse(c, o.p);
        }
        sse(c, "[DONE]");
    } else {
        sb_fmt(&o, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", g->id,
            r->chat ? "chat.completion" : "text_completion", (long)g->created);
        json_quote(&o, alias);
        sb_add(&o, ",\"system_fingerprint\":\"llb-sys1\",\"choices\":[{\"index\":0,");
        if (r->chat) {
            sb_add(&o, "\"message\":{\"role\":\"assistant\",\"content\":");
            if (g->ncalls && !g->content.n) sb_add(&o, "null");
            else json_quote(&o, g->content.p ? g->content.p : "");
            if (g->reasoning.n) {
                sb_add(&o, ",\"reasoning_content\":");
                json_quote(&o, g->reasoning.p);
            }
            if (g->ncalls) {
                sb_add(&o, ",\"tool_calls\":[");
                sb_add(&o, g->calls.p);
                sb_addc(&o, ']');
            }
            sb_add(&o, "},");
        } else {
            sb_add(&o, "\"text\":");
            json_quote(&o, g->content.p ? g->content.p : "");
            sb_addc(&o, ',');
        }
        sb_fmt(&o, "\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
        usage_timings(&o, pn, gn, t0, t1, t2);
        if (r->ret_tokens) {
            sb_add(&o, ",\"llb\":{\"prompt_ids\":");
            int_array_n(&o, pids, (size_t)-1);
            sb_add(&o, ",\"ids\":");
            int_array_n(&o, ids, (size_t)gn);
            sb_addc(&o, '}');
        }
        sb_addc(&o, '}');
        respond(c, 200, "application/json; charset=utf-8", o.p, // merge-train S11: same headers as the plain path (G15)
            (decide_path || prefill_fast) ? "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: fast\r\n" : "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: exact\r\n");
    }
    free(o.p);
}

static void gen_free(Gen *g) {
    free(g->text.p), free(g->content.p), free(g->reasoning.p), free(g->calls.p);
    if (g->tp) llb_tcp_free(g->tp);
}

// one generate request through the engine (caller holds the slot)
static void run_generate(Slot *s, int bank, Conn *c, const jv *body, const Req *r) {
    Gen g = {.c = c, .r = r, .created = time(NULL)};
    snprintf(g.id, sizeof g.id, "%s-%lx%04lx", r->chat ? "chatcmpl" : "cmpl", (long)g.created, c->id & 0xffff);
    char *line = engine_line(body, r, bank);
    double t0 = now(), t1 = 0;
    fputs(line, s->in);
    fputc('\n', s->in);
    free(line);
    if (fflush(s->in)) {
        s->dead = 1;
        err_reply(c, 503, "unavailable_error", "engine is not running");
        return;
    }
    if (r->st.on && r->chat) { // G16: reasoning / tool-call parsing of the response
        g.tp = llb_tcp_new(r->st.tools_json);
        llb_tcp_set_reasoning(g.tp, r->st.reasoning);
    }
    if (r->stream) sse_open(c);
    char *l = NULL;
    size_t cap = 0;
    jv *fin = NULL;
    for (;;) {
        if (c->broken || !wait_engine_line(s, c->fd)) {
            c->cancelled = 1; free(l); gen_free(&g); slot_restart(s); return;
        }
        if (getline(&l, &cap, s->out) <= 0) {
            s->dead = 1;
            if (!r->stream) err_reply(c, 503, "unavailable_error", "engine exited");
            else {
                sb o = {0};
                err_json(&o, 503, "unavailable_error", "engine exited");
                sse(c, o.p);
                sse(c, "[DONE]");
                free(o.p);
            }
            free(l);
            gen_free(&g);
            return;
        }
        s->t_last = now(); // HARDEN: the watchdog sees engine progress
        const char *err = NULL;
        jv *v = json_parse(l, &err);
        if (!v || v->type != J_OBJ) {
            logf_("engine: unparsable line: %s", l);
            json_free(v);
            continue;
        }
        const jv *tk = json_get(v, "token"), *e = json_get(v, "error");
        if (tk && r->st.on) { // G16: stop at the first end-of-generation id
            if (!t1) t1 = now();
            if (!g.stopped) {
                c->gen_n++;
                int id = (int)tk->num;
                const jv *tx = json_get(v, "text");
                const char *piece = tx && tx->type == J_STR ? tx->str : "";
                if (id == ST_EOG_A || id == ST_EOG_B) g.stopped = 1, g.t_eog = now();
                else sb_add(&g.text, piece), st_piece(&g, piece);
            }
            json_free(v);
            continue;
        }
        if (tk) { // stop the response at the first end-of-generation id, as the Metal server does;
                  // the engine still runs its n steps (it has no early stop), their text is dropped
            if (!t1) t1 = now();
            c->gen_n++;
            if (g.stopped) { json_free(v); continue; }
            int id = (int)tk->num;
            if (id == ST_EOG_A || id == ST_EOG_B) { g.stopped = 1, g.t_eog = now(), g.n_eog = c->gen_n - 1; json_free(v); continue; }
            const jv *tx = json_get(v, "text");
            const char *piece = tx && tx->type == J_STR ? tx->str : "";
            sb_add(&g.text, piece);
            if (r->stream && *piece && !c->cancelled) stream_piece(&g, piece);
            json_free(v);
            continue;
        }
        if (e) { // refused (prompt too long, ...): nothing ran
            const char *m = e->type == J_STR ? e->str : "engine refused the request";
            if (r->stream) {
                sb o = {0};
                err_json(&o, 400, "invalid_request_error", m);
                sse(c, o.p);
                sse(c, "[DONE]");
                free(o.p);
            } else err_reply(c, 400, "invalid_request_error", m);
            json_free(v);
            free(l);
            gen_free(&g);
            return;
        }
        fin = v;
        break;
    }
    free(l);
    double t2 = now();
    if (!t1) t1 = t2;
    if (r->st.on) {
        st_finish(&g, fin, t0, t1, g.stopped ? g.t_eog : t2);
        gen_free(&g);
        json_free(fin);
        return;
    }
    const jv *pids = json_get(fin, "prompt_ids"), *ids = json_get(fin, "ids"), *txt = json_get(fin, "text");
    int pn = pids ? (int)pids->n : 0, gn = ids ? (int)ids->n : 0;
    c->prompt_n = pn;
    const char *full = txt && txt->type == J_STR ? txt->str : "";
    size_t have = g.text.p ? g.text.n : 0;
    if (strncmp(full, g.text.p ? g.text.p : "", have)) logf_("warning: streamed text is not a prefix of the final text");
    const char *fr = "length";
    if (g.stopped) full = g.text.p ? g.text.p : "", gn = g.n_eog, t2 = g.t_eog, fr = "stop";
    sb o = {0};
    if (r->stream) {
        if (strlen(full) > have) stream_piece(&g, full + have); // held-back UTF-8 tail, if any
        chunk_head(&g, &o);
        if (r->chat) sb_fmt(&o, "{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
        else sb_fmt(&o, "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
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
        sb_fmt(&o, "\"logprobs\":null,\"finish_reason\":\"%s\"}],", fr);
        usage_timings(&o, pn, gn, t0, t1, t2);
        const jv *ro = json_get(fin, "readout");
        if (r->ret_tokens || ro || r->glue) {
            sb_add(&o, ",\"llb\":{");
            if (r->ret_tokens) {
                sb_add(&o, "\"prompt_ids\":");
                int_array(&o, pids);
                sb_add(&o, ",\"ids\":");
                int_array(&o, ids);
            }
            if (ro) sb_add(&o, r->ret_tokens ? ",\"readout\":" : "\"readout\":"), json_dump(&o, ro);
            if (r->glue) sb_add(&o, r->ret_tokens || ro ? ",\"glue\":" : "\"glue\":"), sb_add(&o, r->glue);
            sb_addc(&o, '}');
        }
        sb_addc(&o, '}');
        respond(c, 200, "application/json; charset=utf-8", o.p,
                (decide_path || prefill_fast) ? "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: fast\r\n"
                            : "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: exact\r\n");
    }
    free(o.p);
    free(g.text.p);
    json_free(fin);
}

// G5: fetch remote images (outside the engine lock), then write every media item to
// owned scratch ($TMPDIR/llb-serve-mm/<sha256>.<ext>, 0600) for the engine to read
static const char *media_stage(Req *r) {
    if (mm_fetch_all(&r->media, &mmc, &r->mme)) return r->status = r->mme.status, r->mme.msg;
    const char *td = getenv("TMPDIR");
    char dir[512];
    snprintf(dir, sizeof dir, "%s/llb-serve-mm", td && *td ? td : "/tmp");
    mkdir(dir, 0700);
    r->paths = calloc(r->media.n, sizeof *r->paths);
    for (size_t i = 0; i < r->media.n; i++) {
        const mm_item *m = &r->media.v[i];
        const char *ext = strchr(m->mime, '/') + 1;
        if (asprintf(&r->paths[i], "%s/%s-%ld.%s", dir, m->sha256, (long)++mm_seq, ext) < 0)
            return r->paths[i] = NULL, "out of memory";
        int fd = open(r->paths[i], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        size_t k = 0;
        while (fd >= 0 && k < m->n) {
            ssize_t w = write(fd, m->bytes + k, m->n - k);
            if (w <= 0) break;
            k += (size_t)w;
        }
        if (fd < 0 || close(fd) || k != m->n) return r->status = 503, "cannot stage the media in TMPDIR";
    }
    return NULL;
}

static void media_done(Req *r) {
    if (r->visdir) { // MIMG: the tower's rows, stamps and log
        char c[700];
        snprintf(c, sizeof c, "rm -rf '%s'", r->visdir);
        if (system(c)) {}
        free(r->visdir), r->visdir = NULL;
    }
    free(r->vis), r->vis = NULL;
    for (size_t i = 0; r->paths && i < r->media.n; i++)
        if (r->paths[i]) unlink(r->paths[i]), free(r->paths[i]);
    free(r->paths);
    mm_list_free(&r->media);
}

// EARS-CANON: with --ear, every staged audio item of a chat request is transcribed (outside the engine lock) and its
// transcript replaces its tag in the prompt; the audio items then leave the media list, so the engine line is the one
// a text request with that user turn would produce. 0 ok, else the HTTP status with *msg.
static int ear_attach(Req *r, const char **msg) {
    static __thread char m[512];
    if (!ear_dir || !r->media.audio) return 0;
    size_t na = 0;
    char **tx = calloc(r->media.n, sizeof *tx);
    int st = 0;
    for (size_t i = 0; i < r->media.n && !st; i++) {
        const mm_item *it = &r->media.v[i];
        if (it->kind != MM_AUDIO) continue;
        const char *err = NULL;
        jv *h = ear_call(r->paths[i], it->sha256, 0, &err, &st);
        const jv *ok = h ? json_get(h, "ok") : NULL, *t = h ? json_get(h, "text") : NULL;
        if (ok && ok->type == J_BOOL && ok->b && t && t->type == J_STR) tx[na++] = strdup(t->str), st = 0;
        else if (h) {
            const jv *s = json_get(h, "status"), *e = json_get(h, "error");
            st = s && s->type == J_NUM ? (int)s->num : 502;
            snprintf(m, sizeof m, "ear: %s", e && e->type == J_STR ? e->str : "transcription failed");
            *msg = m;
        } else *msg = err;
        json_free(h);
    }
    if (!st) {
        sb o = {0};
        if (mm_replace_audio(r->prompt, (const char *const *)tx, na, &o, &r->mme)) st = r->mme.status, *msg = r->mme.msg, free(o.p);
        else free(r->prompt), r->prompt = o.p;
    }
    for (size_t i = 0; i < na; i++) free(tx[i]);
    free(tx);
    if (st) return st;
    size_t k = 0; // drop the transcribed audio items (their scratch files go now)
    for (size_t i = 0; i < r->media.n; i++) {
        if (r->media.v[i].kind == MM_AUDIO) {
            if (r->paths[i]) unlink(r->paths[i]), free(r->paths[i]);
            free(r->media.v[i].bytes), free(r->media.v[i].url);
            continue;
        }
        r->media.v[k] = r->media.v[i], r->paths[k] = r->paths[i], k++;
    }
    r->media.n = k, r->media.audio = 0;
    return 0;
}

// MIMG: with --vis-tower, every staged image of the request -> the tower's rows (one CLI --batch run, outside the
// engine lock), stamped; its placeholder pad -> nx*ny pads; the image items leave the media list ("media" is not sent
// for them). 0 ok, else the HTTP status with *msg.
static int vis_attach(Req *r, const char **msg) {
    static __thread char m[512];
    size_t ni = 0;
    for (size_t i = 0; i < r->media.n; i++) ni += r->media.v[i].kind == MM_IMAGE;
    if (!vis_tower || !ni) return 0;
    if (ni > 16) return *msg = "at most 16 images per request with --vis-tower", 400;
    size_t np = 0;
    for (const char *q = r->prompt; (q = strstr(q, "<|image_pad|>")); q += 13) np++;
    if (np != ni) return *msg = "the prompt's <|image_pad|> count differs from its images (literal <|image_pad|> in text?)", 400;
    const char *td = getenv("TMPDIR");
    if (asprintf(&r->visdir, "%s/llb-serve-vis-%d-%ld", td && *td ? td : "/tmp", (int)getpid(), (long)++vis_seq) < 0)
        return r->visdir = NULL, *msg = "out of memory", 503;
    char **av = calloc(ni + 5, sizeof *av);
    size_t k = 0;
    av[k++] = (char *)vis_tower, av[k++] = "--batch", av[k++] = (char *)vis_mmproj, av[k++] = r->visdir;
    for (size_t i = 0; i < r->media.n; i++) if (r->media.v[i].kind == MM_IMAGE) av[k++] = r->paths[i];
    av[k] = NULL;
    int pp[2];
    if (pipe(pp)) return free(av), *msg = "cannot run the vision tower", 503;
    pid_t pid = fork();
    if (pid == 0) {
        char lg[600];
        snprintf(lg, sizeof lg, "%s/tower.log", r->visdir);
        mkdir(r->visdir, 0700);
        int fd = open(lg, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        dup2(pp[1], 1), close(pp[0]), close(pp[1]);
        if (fd >= 0) dup2(fd, 2);
        execv(vis_tower, av);
        _exit(127);
    }
    free(av);
    close(pp[1]);
    FILE *f = pid > 0 ? fdopen(pp[0], "r") : NULL;
    unsigned gx[16] = {0}, gy[16] = {0}, got = 0;
    char *ln = NULL;
    size_t lc = 0;
    while (f && getline(&ln, &lc, f) > 0) {
        int kk = -1, w = 0, h = 0;
        char path[4096];
        if (sscanf(ln, "img\t%d\t%4095[^\t]\t%dx%d", &kk, path, &w, &h) == 4 && kk >= 0 && kk < 16 && w > 0 && h > 0)
            gx[kk] = (unsigned)w / 32, gy[kk] = (unsigned)h / 32, got++;
    }
    free(ln);
    if (f) fclose(f); else close(pp[0]);
    int ws = 0;
    if (pid > 0) waitpid(pid, &ws, 0);
    if (pid <= 0 || !WIFEXITED(ws) || WEXITSTATUS(ws) || got != ni) {
        snprintf(m, sizeof m, "vision tower failed (rc %d, %u of %zu images; its log is in the server's stderr)", WIFEXITED(ws) ? WEXITSTATUS(ws) : -1, got, ni);
        char lg[600], b[4096];
        snprintf(lg, sizeof lg, "%s/tower.log", r->visdir);
        FILE *lf = fopen(lg, "r");
        size_t nb = lf ? fread(b, 1, sizeof b - 1, lf) : 0;
        if (lf) fclose(lf);
        b[nb] = 0;
        fprintf(stderr, "serve: vision tower failed: %s\n", b);
        return *msg = m, 502;
    }
    sb js = {0}, pr = {0};
    sb_addc(&js, '[');
    for (unsigned i = 0; i < ni; i++) {
        char rp[700], sp[700], sha[65];
        snprintf(rp, sizeof rp, "%s/single-%u.f32", r->visdir, i);
        snprintf(sp, sizeof sp, "%s/single-%u.json", r->visdir, i);
        FILE *rf = fopen(rp, "rb");
        unsigned char *b = NULL;
        long sz = -1;
        if (rf && !fseek(rf, 0, SEEK_END) && (sz = ftell(rf)) > 0 && !fseek(rf, 0, SEEK_SET) && (b = malloc((size_t)sz)) &&
            fread(b, 1, (size_t)sz, rf) == (size_t)sz && gx[i] && gy[i] && sz % ((long)gx[i] * gy[i] * 4) == 0)
            mm_sha256_hex(b, (size_t)sz, sha);
        else sz = -1;
        // SENSES-MATRIX: the rows' width is the tower's (its mmproj's projection: 4096 on the 9B, 1024 / 2560 / 5120 on
        // the 0.8B / 4B / 27B); the stamp carries it and dual.c refuses rows that are not the trunk's width.
        long rd = sz > 0 ? sz / ((long)gx[i] * gy[i] * 4) : 0;
        if (rd < 256 || rd > 8192 || rd % 256) sz = -1;
        if (rf) fclose(rf);
        free(b);
        FILE *sf = sz > 0 ? fopen(sp, "w") : NULL;
        if (!sf || fprintf(sf, "{\"kind\":\"llb-tower-rows-v1\",\"mmproj_sha256\":\"%s\",\"nx\":%u,\"ny\":%u,\"d\":%ld,\"rows_sha256\":\"%s\"}\n",
                vis_mmsha, gx[i], gy[i], rd, sha) < 0 || fclose(sf)) {
            free(js.p);
            return *msg = "cannot stamp the vision tower's rows", 503;
        }
        sb_fmt(&js, "%s{\"rows\":", i ? "," : "");
        json_quote(&js, rp);
        sb_fmt(&js, ",\"nx\":%u,\"ny\":%u}", gx[i], gy[i]);
    }
    sb_addc(&js, ']');
    unsigned i = 0;
    for (const char *q = r->prompt, *e; ; q = e + 13, i++) {
        if (!(e = strstr(q, "<|image_pad|>"))) { sb_add(&pr, q); break; }
        sb_fmt(&pr, "%.*s", (int)(e - q), q);
        for (unsigned t = 0; t < gx[i] * gy[i]; t++) sb_add(&pr, "<|image_pad|>");
    }
    free(r->prompt), r->prompt = pr.p, r->vis = js.p;
    size_t kk = 0; // drop the image items (their scratch files go now)
    for (size_t j = 0; j < r->media.n; j++) {
        if (r->media.v[j].kind == MM_IMAGE) {
            if (r->paths[j]) unlink(r->paths[j]), free(r->paths[j]);
            free(r->media.v[j].bytes), free(r->media.v[j].url);
            continue;
        }
        r->media.v[kk] = r->media.v[j], r->paths[kk] = r->paths[j], kk++;
    }
    r->media.n = kk;
    return 0;
}

// ---- SERVE-BATCH: /v1/decide and batch requests -------------------------------
// POST /v1/decide: one clef decide request ({"model","state","questions":{...}}, dual.c's decide line)
// passed to the engine as is; the reply is the engine's answer object. Every question of the task is
// answered from ONE engine pass (one prefill; the head reads all question/option spans of its rows).
// {"batch":[item,...]} on /v1/completions, /v1/chat/completions or /v1/decide: N requests in one HTTP
// request. Item i = its own fields plus the batch object's other fields it does not set (shared
// defaults; "batch", "stream" and "custom_id" are not shared). The items take ONE queue ticket and hold
// ONE slot for the whole batch and run back to back, in order, through the same code as the single-item
// path, so item i's body is the single-item response to that merged request. The engine runs one
// sequence per pass, so a batch of tasks/prompts is pipelined here, not fused into one pass (fusing
// would change the prompt and the answers). Reply: {"object":"batch","data":[{"index","custom_id",
// "response":{"status_code","body"}}...],"usage",...}; "stream":true = one SSE event per item as it
// finishes, then a "batch.done" event and [DONE]. /v1/completions also takes "prompt":[s1,s2,...]
// (OpenAI's array form): run as a batch, answered as one text_completion with choices[i].
#define BATCH_MAX 4096
enum { K_COMP, K_CHAT, K_DECIDE };

typedef struct {
    char *body;   // the merged request text
    jv *jb;       // parsed
    const jv *cid; // custom_id (in the batch body)
    Req r;        // completions / chat
    char *line;   // decide: the engine line
    int status;   // 0 = not run yet / ok to run
    sb out;       // the response body (single-item bytes)
    int sent;     // BATCH-NATIVE-2 --native-cont: its stream event is out
} Item;

// item fields first, then the batch's shared fields the item does not set
static char *batch_merge(const jv *top, const jv *it) {
    sb o = {0};
    int k = 0;
    sb_addc(&o, '{');
    for (size_t i = 0; i < it->n; i++) {
        if (!strcmp(it->keys[i], "custom_id")) continue;
        if (k++) sb_addc(&o, ',');
        json_quote(&o, it->keys[i]);
        sb_addc(&o, ':');
        json_dump(&o, it->items[i]);
    }
    for (size_t i = 0; top && i < top->n; i++) {
        const char *key = top->keys[i];
        if (!strcmp(key, "batch") || !strcmp(key, "stream") || !strcmp(key, "custom_id") || json_get(it, key)) continue;
        if (k++) sb_addc(&o, ',');
        json_quote(&o, key);
        sb_addc(&o, ':');
        json_dump(&o, top->items[i]);
    }
    sb_addc(&o, '}');
    return o.p;
}

// the engine line of a decide request (NULL + *err when it is refused here)
static char *decide_line(const jv *b, int bank, const char **err) {
    const jv *m;
    if (!b || b->type != J_OBJ) return *err = "request body must be a JSON object", NULL;
    if ((m = json_get(b, "mode")) && m->type == J_STR && !strcmp(m->str, "generate"))
        return *err = "generate requests go to /v1/completions", NULL;
    if (json_get(b, "bank")) return *err = "bank is chosen by the server", NULL;
    sb o = {0};
    json_dump(&o, b);
    if (n_banks > 1) { // G20: the engine's sequence state
        o.p[--o.n] = 0;
        sb_fmt(&o, "%s\"bank\":%d}", b->n ? "," : "", bank);
    }
    return o.p;
}

static void decide_read(Slot *s, Conn *c);
// one decide request through the engine (caller holds the slot)
static void run_decide(Slot *s, Conn *c, const char *line) {
    fputs(line, s->in);
    fputc('\n', s->in);
    if (fflush(s->in)) {
        s->dead = 1;
        return err_reply(c, 503, "unavailable_error", "engine is not running");
    }
    decide_read(s, c);
}

// BATCH-NATIVE-2 --native-cont: the {"bx_item":k} line before an out-of-order reply; -1 = none
static int bx_item_read(Slot *s, Conn *c) {
    char *l = NULL;
    size_t cap = 0;
    int k = -1;
    for (;;) {
        if (c->broken || !wait_engine_line(s, c->fd)) { c->cancelled = 1; slot_restart(s); break; }
        ssize_t n = getline(&l, &cap, s->out);
        if (n <= 0) break;
        const char *err = NULL;
        jv *v = json_parse(l, &err);
        const jv *b = v && v->type == J_OBJ ? json_get(v, "bx_item") : NULL;
        if (b && b->type == J_NUM) k = (int)b->num;
        int skip = !b && (!v || v->type != J_OBJ || json_get(v, "token"));
        if (!b && !skip) logf_("engine: expected a bx_item line: %s", l);
        json_free(v);
        if (!skip) break;
    }
    free(l);
    return k;
}

// a decide's reply line from the engine (its request already written)
static void decide_read(Slot *s, Conn *c) {
    char *l = NULL;
    size_t cap = 0;
    for (;;) {
        if (!wait_engine_line(s, c->fd)) { c->cancelled = 1; slot_restart(s); break; }
        ssize_t k = getline(&l, &cap, s->out);
        if (k <= 0) {
            s->dead = 1;
            err_reply(c, 503, "unavailable_error", "engine exited");
            break;
        }
        const char *err = NULL;
        jv *v = json_parse(l, &err);
        if (!v || v->type != J_OBJ || json_get(v, "token")) {
            if (!v || v->type != J_OBJ) logf_("engine: unparsable line: %s", l);
            json_free(v);
            continue;
        }
        const jv *e = json_get(v, "error");
        if (e && e->type == J_STR) err_reply(c, 400, "invalid_request_error", e->str); // refused: nothing ran
        else {
            while (k && (l[k - 1] == '\n' || l[k - 1] == '\r')) l[--k] = 0;
            respond(c, 200, "application/json; charset=utf-8", l, "X-LLB-Prefill: exact\r\n"); // decides take the exact pass
        }
        json_free(v);
        break;
    }
    free(l);
}

static int serve_gate(Conn *c) { // 0 = the engine can take a request, else replied
    if (engine_dead || all_dead()) return err_reply(c, 503, "unavailable_error", "engine is not running"), 1;
    if (!ready) return err_reply(c, 503, "unavailable_error", "Loading model"), 1;
    return 0;
}

static void handle_decide(Conn *c, const char *body_s);
static void batch_items(Conn *c, Item *it, int n, int kind, int stream);

static void item_err(Item *t, int status, const char *msg, const char *param) {
    t->status = status;
    err_json_p(&t->out, status, status == 503 ? "unavailable_error" : "invalid_request_error", msg, param);
}

// parse + stage one item (outside the engine lock); t->status set on a refusal
static void item_prep(Item *t, int kind) {
    const char *err = NULL;
    t->jb = json_parse(t->body, &err);
    if (!t->jb) return item_err(t, 400, err ? err : "invalid JSON", NULL);
    if (kind == K_DECIDE) {
        if (!(t->line = decide_line(t->jb, 0, &err))) item_err(t, 400, err, NULL);
        return;
    }
    const jv *x = json_get(t->jb, "stream");
    if (x && x->type == J_BOOL && x->b) return item_err(t, 400, "stream is set on the batch request, not per item", NULL);
    if (!(err = parse_req(t->jb, kind == K_CHAT, &t->r)) && t->r.media.n) err = media_stage(&t->r);
    if (err) return item_err(t, t->r.status ? t->r.status : 400, err, t->r.status ? t->r.mme.param : NULL);
    int est = ear_attach(&t->r, &err); // EARS-CANON: audio -> the ear's transcript in the prompt
    if (!est) est = vis_attach(&t->r, &err); // MIMG: images -> tower rows
    if (est) return item_err(t, est, err, NULL);
    int gst = glue_run(t->jb, &t->r, &err);
    if (gst) item_err(t, gst, err, NULL);
}

static void item_free(Item *t) {
    json_free(t->jb);
    free(t->body), free(t->line), free(t->out.p);
    free(t->r.prompt);
    st_free(&t->r.st);
    media_done(&t->r);
    free(t->r.img), free(t->r.glue);
}

static void item_json(sb *o, const Item *t, int i) {
    sb_fmt(o, "{\"index\":%d,\"custom_id\":", i);
    if (t->cid) json_dump(o, t->cid);
    else sb_add(o, "null");
    sb_fmt(o, ",\"response\":{\"status_code\":%d,\"body\":", t->status);
    sb_add(o, t->out.p ? t->out.p : "null");
    sb_add(o, "}}");
}

// run prepared items: one ticket, one slot hold, in order. stream = an SSE event per item.
// Returns with every item's status/out set (or replied a whole-request error and returned -1).
static int batch_run(Conn *c, Item *it, int n, int kind, int stream, long *pn) {
    int first = -1, last = -1;
    for (int i = 0; i < n; i++)
        if (!it[i].status) last = i, first = first < 0 ? i : first;
    Slot *s = NULL;
    int bank = 0;
    unsigned long ticket = 0;
    if (first >= 0) {
        if (serve_gate(c)) return -1;
        queued++;
        double tq = now();
        s = slot_acquire(kind == K_DECIDE ? "" : it[first].r.prompt, &ticket, &bank, c->fd);
        queued--;
        if (!s) {
            err_reply(c, 503, "unavailable_error", closing ? "server is shutting down" : "engine is not running");
            return -1;
        }
        c->slot = s->idx * n_banks + bank;
        if (n_slots * n_banks > 1)
            snprintf(c->xh, sizeof c->xh, "X-LLB-Slot: %d\r\nX-LLB-Ticket: %lu\r\nX-LLB-Queue-Ms: %.3f\r\nX-LLB-Run-Start: %.6f\r\n",
                c->slot, ticket, (now() - tq) * 1000, now());
    }
    if (stream) sse_open(c);
    int grp_end = 0; // BATCH-NATIVE: the items before it are already sent
    for (int i = 0; i < n; i++) {
        Item *t = &it[i];
        if (s && (c->broken || client_gone(c->fd))) { c->cancelled = 1; slot_restart(s); break; }
        if (!t->status && s->dead) item_err(t, 503, "engine exited", NULL);
        if (!t->status) {
            Conn ic = *c; // the item's own request id; its reply is captured into t->out
            ic.cap = &t->out, ic.id = ++req_seq, ic.gen_n = 0, ic.prompt_n = 0;
            if (kind == K_DECIDE && native_batch > 1 && n_banks == 1) {
                // BATCH-NATIVE: this item opens a group of up to N good items, sent as one line
                if (i >= grp_end) {
                    sb g = {0};
                    sb_add(&g, "{\"decide_batch\":[");
                    int k = 0, *gi = calloc(256, sizeof *gi);
                    for (grp_end = i; grp_end < n && k < (native_cont ? 256 : native_batch); grp_end++) {
                        if (it[grp_end].status) continue;
                        gi[k] = grp_end;
                        if (k++) sb_addc(&g, ',');
                        sb_add(&g, it[grp_end].line);
                    }
                    sb_add(&g, "]}");
                    fputs(g.p, s->in);
                    fputc('\n', s->in);
                    free(g.p);
                    if (fflush(s->in)) s->dead = 1;
                    // --native-cont: the engine answers each item when it is done; route by bx_item and
                    // stream it at once (its event carries its index)
                    for (int r = 0; native_cont && r < k; r++) {
                        int m = s->dead ? -1 : bx_item_read(s, c);
                        if (c->cancelled) break;
                        if (m < 0 || m >= k || it[gi[m]].status) { s->dead = 1; break; }
                        Item *u = &it[gi[m]];
                        Conn uc = *c;
                        uc.cap = &u->out, uc.id = ++req_seq, uc.gen_n = 0, uc.prompt_n = 0;
                        decide_read(s, &uc);
                        if (uc.cancelled) { c->cancelled = 1; break; }
                        u->status = uc.cap_status ? uc.cap_status : 500;
                        if (stream) {
                            sb o = {0};
                            item_json(&o, u, gi[m]);
                            sse(c, o.p);
                            free(o.p);
                            u->sent = 1;
                        }
                    }
                    free(gi);
                    if (c->cancelled) break;
                }
                if (native_cont) {
                    if (!t->status) item_err(t, 503, "engine exited", NULL);
                    if (stream && !t->sent) {
                        sb o = {0};
                        item_json(&o, t, i);
                        sse(c, o.p);
                        free(o.p);
                        t->sent = 1;
                    }
                    continue;
                }
                if (s->dead) err_reply(&ic, 503, "unavailable_error", "engine is not running");
                else decide_read(s, &ic);
            } else if (kind == K_DECIDE) {
                if (n_banks > 1) { // G20: the line carries the bank the batch holds
                    const char *e = NULL;
                    free(t->line);
                    t->line = decide_line(t->jb, bank, &e);
                }
                run_decide(s, &ic, t->line);
            } else run_generate(s, bank, &ic, t->jb, &t->r);
            if (ic.cancelled) { c->cancelled = 1; break; }
            t->status = ic.cap_status ? ic.cap_status : 500;
            c->gen_n += ic.gen_n;
            *pn += ic.prompt_n;
        }
        if (stream && !t->sent) {
            sb o = {0};
            item_json(&o, t, i);
            sse(c, o.p);
            free(o.p);
        }
    }
    if (s) slot_release(s, bank, kind == K_DECIDE ? "" : it[last].r.prompt);
    return 0;
}

static Item *items_from(const jv *top, const jv *arr, const char *key, int kind) {
    Item *it = calloc(arr->n, sizeof *it);
    for (size_t i = 0; i < arr->n; i++) {
        const jv *x = arr->items[i];
        if (key) { // the prompt-array form: item i = {"prompt": element i}
            sb o = {0};
            sb_add(&o, "{\"prompt\":");
            json_dump(&o, x);
            sb_addc(&o, '}');
            const char *pe = NULL;
            jv *one = json_parse(o.p, &pe);
            it[i].body = batch_merge(top, one);
            json_free(one), free(o.p);
        } else if (x->type != J_OBJ) {
            it[i].body = strdup("{}");
            item_err(&it[i], 400, "every batch item must be a JSON object", NULL);
            continue;
        } else {
            it[i].body = batch_merge(top, x);
            it[i].cid = json_get(x, "custom_id");
        }
        item_prep(&it[i], kind);
    }
    return it;
}

static void handle_batch(Conn *c, const jv *body, int kind) {
    const jv *b = json_get(body, "batch"), *x;
    if (b->type != J_ARR || !b->n || b->n > BATCH_MAX)
        return err_reply_p(c, 400, "invalid_request_error", "batch must be an array of 1..4096 request objects", "batch");
    int stream = (x = json_get(body, "stream")) && x->type == J_BOOL && x->b, n = (int)b->n;
    Item *it = items_from(body, b, NULL, kind);
    batch_items(c, it, n, kind, stream);
    for (int i = 0; i < n; i++) item_free(&it[i]);
    free(it);
}

static void batch_items(Conn *c, Item *it, int n, int kind, int stream) {
    double t0 = now();
    long pn = 0;
    if (batch_run(c, it, n, kind, stream, &pn)) return;
    int ok = 0;
    for (int i = 0; i < n; i++) ok += it[i].status == 200;
    sb o = {0};
    if (stream) sb_add(&o, "{\"object\":\"batch.done\",\"model\":");
    else sb_add(&o, "{\"object\":\"batch\",\"model\":");
    json_quote(&o, alias);
    if (!stream) {
        sb_add(&o, ",\"data\":[");
        for (int i = 0; i < n; i++) {
            if (i) sb_addc(&o, ',');
            item_json(&o, &it[i], i);
        }
        sb_addc(&o, ']');
    }
    if (kind != K_DECIDE)
        sb_fmt(&o, ",\"usage\":{\"prompt_tokens\":%ld,\"completion_tokens\":%d,\"total_tokens\":%ld}", pn, c->gen_n, pn + c->gen_n);
    sb_fmt(&o, ",\"llb\":{\"items\":%d,\"ok\":%d,\"failed\":%d,\"wall_ms\":%.3f}}", n, ok, n - ok, (now() - t0) * 1000);
    c->prompt_n = (int)pn;
    if (stream) {
        sse(c, o.p);
        sse(c, "[DONE]");
    } else
        respond(c, 200, "application/json; charset=utf-8", o.p,
            kind == K_DECIDE ? "X-LLB-Prefill: exact\r\n"
            : (decide_path || prefill_fast) ? "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: fast\r\n"
                             : "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: exact\r\n");
    free(o.p);
}

// OpenAI's array prompt on /v1/completions: one text_completion, choices[i] = prompt i (one failing
// prompt fails the request with its error, as a single request would)
static void handle_prompt_array(Conn *c, const jv *body) {
    const jv *p = json_get(body, "prompt"), *x;
    if (p->n > BATCH_MAX) return err_reply_p(c, 400, "invalid_request_error", "at most 4096 prompts", "prompt");
    if ((x = json_get(body, "stream")) && x->type == J_BOOL && x->b)
        return err_reply(c, 400, "invalid_request_error", "stream is not supported with an array of prompts; use {\"batch\":[...],\"stream\":true}");
    for (size_t i = 0; i < p->n; i++)
        if (p->items[i]->type != J_STR) return err_reply(c, 400, "invalid_request_error", "prompt must be a string or an array of strings");
    int n = (int)p->n;
    Item *it = items_from(body, p, "prompt", K_COMP);
    long pn = 0;
    int bad = -1;
    for (int i = 0; i < n && bad < 0; i++)
        if (it[i].status) bad = i;
    if (bad < 0 && !batch_run(c, it, n, K_COMP, 0, &pn))
        for (int i = 0; i < n && bad < 0; i++)
            if (it[i].status != 200) bad = i;
    if (bad >= 0) json_reply(c, it[bad].status, it[bad].out.p);
    else if (!c->status) {
        time_t created = time(NULL);
        sb o = {0};
        sb_fmt(&o, "{\"id\":\"cmpl-%lx%04lx\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", (long)created,
            c->id & 0xffff, (long)created);
        json_quote(&o, alias);
        sb_add(&o, ",\"system_fingerprint\":\"llb-sys1\",\"choices\":[");
        long cn = 0;
        sb llb = {0};
        for (int i = 0; i < n; i++) {
            const char *pe = NULL;
            jv *r = json_parse(it[i].out.p, &pe);
            const jv *ch = r ? json_get(r, "choices") : NULL, *u = r ? json_get(r, "usage") : NULL;
            const jv *c0 = ch && ch->n ? ch->items[0] : NULL, *t = c0 ? json_get(c0, "text") : NULL;
            const jv *fr = c0 ? json_get(c0, "finish_reason") : NULL, *ct = u ? json_get(u, "completion_tokens") : NULL;
            const jv *l = r ? json_get(r, "llb") : NULL;
            sb_fmt(&o, "%s{\"index\":%d,\"text\":", i ? "," : "", i);
            json_quote(&o, t && t->type == J_STR ? t->str : "");
            sb_add(&o, ",\"logprobs\":null,\"finish_reason\":");
            if (fr) json_dump(&o, fr);
            else sb_add(&o, "null");
            sb_addc(&o, '}');
            cn += ct ? (long)ct->num : 0;
            if (l) sb_add(&llb, llb.n ? "," : ""), json_dump(&llb, l);
            json_free(r);
        }
        sb_fmt(&o, "],\"usage\":{\"prompt_tokens\":%ld,\"completion_tokens\":%ld,\"total_tokens\":%ld}", pn, cn, pn + cn);
        if (llb.n) sb_add(&o, ",\"llb\":{\"items\":["), sb_add(&o, llb.p), sb_add(&o, "]}");
        sb_addc(&o, '}');
        c->prompt_n = (int)pn;
        respond(c, 200, "application/json; charset=utf-8", o.p,
            (decide_path || prefill_fast) ? "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: fast\r\n" : "X-LLB-Sampler: w1-sampler\r\nX-LLB-Prefill: exact\r\n");
        free(o.p), free(llb.p);
    }
    for (int i = 0; i < n; i++) item_free(&it[i]);
    free(it);
}

static void handle_decide(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body) return err_reply(c, 400, "invalid_request_error", err ? err : "invalid JSON");
    if (body->type == J_OBJ && json_get(body, "batch")) {
        handle_batch(c, body, K_DECIDE);
        return json_free(body);
    }
    char *line = decide_line(body, 0, &err);
    if (!line) err_reply(c, 400, "invalid_request_error", err);
    else if (!serve_gate(c)) {
        queued++;
        unsigned long ticket;
        int bank;
        double tq = now();
        Slot *s = slot_acquire("", &ticket, &bank, c->fd);
        queued--;
        if (!s) err_reply(c, 503, "unavailable_error", closing ? "server is shutting down" : "engine is not running");
        else {
            c->slot = s->idx * n_banks + bank;
            if (n_slots * n_banks > 1)
                snprintf(c->xh, sizeof c->xh, "X-LLB-Slot: %d\r\nX-LLB-Ticket: %lu\r\nX-LLB-Queue-Ms: %.3f\r\nX-LLB-Run-Start: %.6f\r\n",
                    c->slot, ticket, (now() - tq) * 1000, now());
            if (n_banks > 1) free(line), line = decide_line(body, bank, &err);
            run_decide(s, c, line);
            slot_release(s, bank, "");
        }
    }
    free(line);
    json_free(body);
}

static void handle_generate(Conn *c, const char *body_s, int chat) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    Req r = {0};
    if (!body) {
        err_reply(c, 400, "invalid_request_error", err ? err : "invalid JSON");
        return;
    }
    const jv *bx = body->type == J_OBJ ? json_get(body, "batch") : NULL, *pa; // SERVE-BATCH
    if (bx || (!chat && body->type == J_OBJ && (pa = json_get(body, "prompt")) && pa->type == J_ARR && pa->n > 1)) {
        if (bx) handle_batch(c, body, chat ? K_CHAT : K_COMP);
        else handle_prompt_array(c, body);
        json_free(body);
        return;
    }
    if (!(err = parse_req(body, chat, &r)) && r.media.n) err = media_stage(&r); // fetch URLs, write scratch files
    if (err) {
        err_reply_p(c, r.status ? r.status : 400, "invalid_request_error", err, r.status ? r.mme.param : NULL);
        json_free(body);
        free(r.prompt);
        st_free(&r.st);
        media_done(&r);
        return;
    }
    int gst = ear_attach(&r, &err); // EARS-CANON: the ear runs outside the engine lock too
    if (!gst) gst = vis_attach(&r, &err); // MIMG: the vision tower runs outside the engine lock too
    if (!gst) gst = glue_run(body, &r, &err); // the encoder runs outside the engine lock
    if (gst) {
        err_reply(c, gst, gst == 400 ? "invalid_request_error" : "unavailable_error", err);
        json_free(body);
        free(r.prompt), free(r.img), free(r.glue);
        st_free(&r.st);
        media_done(&r); // the staged media (an ear refusal leaves its scratch file otherwise)
        return;
    }
    c->stream = r.stream;
    if (engine_dead || all_dead()) err_reply(c, 503, "unavailable_error", "engine is not running");
    else if (!ready) err_reply(c, 503, "unavailable_error", "Loading model");
    else {
        queued++;
        unsigned long ticket;
        int bank;
        double tq = now();
        Slot *s = slot_acquire(r.prompt, &ticket, &bank, c->fd);
        queued--;
        if (!s) err_reply(c, 503, "unavailable_error", closing ? "server is shutting down" : "engine is not running");
        else {
            c->slot = s->idx * n_banks + bank;
            if (n_slots * n_banks > 1)
                snprintf(c->xh, sizeof c->xh, "X-LLB-Slot: %d\r\nX-LLB-Ticket: %lu\r\nX-LLB-Queue-Ms: %.3f\r\nX-LLB-Run-Start: %.6f\r\n",
                    c->slot, ticket, (now() - tq) * 1000, now());
            run_generate(s, bank, c, body, &r);
            slot_release(s, bank, r.prompt);
        }
    }
    json_free(body);
    free(r.prompt);
    st_free(&r.st);
    media_done(&r);
    free(r.img), free(r.glue);
}

// ---- G41: speech routes ------------------------------------------------------

static const char *tts_type(int st) { return st == 400 ? "invalid_request_error" : st == 503 ? "unavailable_error" : "server_error"; }

// no reply / a reply that is not ok -> the OpenAI error shape (503 when the worker is loading or gone)
static int tts_failed(Conn *c, jv *h, const char *err) {
    if (!h) {
        err_reply(c, 503, tts_type(503), err);
        return 1;
    }
    const jv *ok = json_get(h, "ok");
    if (ok && ok->type == J_BOOL && ok->b) return 0;
    const jv *st = json_get(h, "status"), *m = json_get(h, "error");
    int s = st && st->type == J_NUM ? (int)st->num : 500;
    err_reply(c, s, tts_type(s), m && m->type == J_STR ? m->str : "speech failed");
    json_free(h);
    return 1;
}

static void handle_speech(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body || body->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", err ? err : "body must be a JSON object");
        json_free(body);
        return;
    }
    const jv *in = json_get(body, "input"), *v = json_get(body, "voice"), *sp = json_get(body, "speed"),
             *f = json_get(body, "response_format");
    if (!in || in->type != J_STR) err = "input must be a string";
    else if (v && v->type != J_STR && v->type != J_NULL) err = "voice must be a string";
    else if (sp && sp->type != J_NUM && sp->type != J_NULL) err = "speed must be a number";
    else if (f && f->type != J_STR && f->type != J_NULL) err = "response_format must be a string";
    if (err) {
        err_reply(c, 400, "invalid_request_error", err);
        json_free(body);
        return;
    }
    sb r = {0};
    sb_add(&r, "{\"op\":\"speak\",\"text\":");
    json_quote(&r, in->str);
    if (v && v->type == J_STR) sb_add(&r, ",\"voice\":"), json_quote(&r, v->str);
    if (sp && sp->type == J_NUM) sb_fmt(&r, ",\"speed\":%.17g", sp->num);
    if (f && f->type == J_STR) sb_add(&r, ",\"format\":"), json_quote(&r, f->str);
    sb_addc(&r, '}');
    char *audio;
    size_t n;
    double t0 = now();
    jv *h = tts_call(r.p, &audio, &n, &err);
    free(r.p);
    json_free(body);
    if (tts_failed(c, h, err)) return free(audio);
    const jv *fmt = json_get(h, "format"), *vo = json_get(h, "voice"), *sha = json_get(h, "sha256"),
             *sy = json_get(h, "synth_ms"), *as = json_get(h, "audio_s");
    int pcm = fmt && fmt->type == J_STR && !strcmp(fmt->str, "pcm");
    char xh[512];
    snprintf(xh, sizeof xh, "X-LLB-TTS-Voice: %s\r\nX-LLB-TTS-Synth-Ms: %.1f\r\nX-LLB-TTS-Ms: %.1f\r\nX-LLB-Audio-Seconds: %.3f\r\nX-LLB-Audio-Sha256: %s\r\n",
        vo && vo->type == J_STR ? vo->str : "?", sy && sy->type == J_NUM ? sy->num : -1, (now() - t0) * 1000,
        as && as->type == J_NUM ? as->num : -1, sha && sha->type == J_STR ? sha->str : "?");
    respond_n(c, 200, pcm ? "audio/pcm; rate=24000; channels=1; encoding=s16le" : "audio/wav", audio ? audio : "", n, xh);
    free(audio);
    json_free(h);
}

// EARS-CANON: POST /v1/audio/transcriptions {"input_audio":{"data":B64,"format":"wav"},"response_format"?,"max_tokens"?}
// (the chat part's audio object, same checks and caps) -> {"text"} | text/plain | verbose_json (+ ids, logits, timings)
static void handle_transcription(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    if (!body || body->type != J_OBJ) {
        err_reply(c, 400, "invalid_request_error", err ? err : "body must be a JSON object {\"input_audio\":{\"data\",\"format\"}}");
        json_free(body);
        return;
    }
    const jv *f = json_get(body, "response_format");
    char fmt[16]; // copied: body is freed before the reply is built
    snprintf(fmt, sizeof fmt, "%s", f && f->type == J_STR ? f->str : "json");
    int n = 0;
    Req r = {0};
    if (f && (f->type != J_STR || (strcmp(fmt, "json") && strcmp(fmt, "text") && strcmp(fmt, "verbose_json"))))
        err = "response_format must be json, text or verbose_json";
    else if (num_field(body, "max_tokens", &n, &err) < 0) {
    } else if (mm_audio_item(json_get(body, "input_audio"), "input_audio", &mmc, &r.media, &r.mme))
        r.status = r.mme.status, err = r.mme.msg;
    else err = media_stage(&r);
    if (err) {
        err_reply_p(c, r.status ? r.status : 400, "invalid_request_error", err, r.status ? r.mme.param : NULL);
        media_done(&r);
        json_free(body);
        return;
    }
    double t0 = now();
    int st;
    jv *h = ear_call(r.paths[0], r.media.v[0].sha256, n, &err, &st);
    char sha[65];
    memcpy(sha, r.media.v[0].sha256, sizeof sha);
    media_done(&r);
    json_free(body);
    const jv *ok = h ? json_get(h, "ok") : NULL;
    if (!ok || ok->type != J_BOOL || !ok->b) {
        const jv *s = h ? json_get(h, "status") : NULL, *e = h ? json_get(h, "error") : NULL;
        if (h) st = s && s->type == J_NUM ? (int)s->num : 502;
        err_reply(c, st, st == 400 ? "invalid_request_error" : st == 503 ? "unavailable_error" : "server_error",
            h ? (e && e->type == J_STR ? e->str : "transcription failed") : err);
        json_free(h);
        return;
    }
    const jv *tx = json_get(h, "text"), *as = json_get(h, "audio_s");
    char xh[256];
    snprintf(xh, sizeof xh, "X-LLB-Ear-Ms: %.1f\r\nX-LLB-Audio-Seconds: %.3f\r\nX-LLB-Audio-Sha256: %s\r\n", (now() - t0) * 1000,
        as && as->type == J_NUM ? as->num : -1, sha);
    sb o = {0};
    if (!strcmp(fmt, "text")) {
        sb_add(&o, tx->str), sb_addc(&o, '\n');
        respond_n(c, 200, "text/plain; charset=utf-8", o.p, o.n, xh);
    } else {
        sb_add(&o, "{\"text\":");
        json_quote(&o, tx->str);
        if (!strcmp(fmt, "verbose_json")) { // OpenAI's task/language/duration, then the ear's receipt under "llb"
            sb_add(&o, ",\"task\":\"transcribe\",\"language\":");
            json_dump(&o, json_get(h, "language"));
            sb_fmt(&o, ",\"duration\":%.4f,\"llb\":{", as && as->type == J_NUM ? as->num : -1);
            int first = 1;
            for (size_t i = 0; i < h->n; i++) {
                if (!strcmp(h->keys[i], "ok") || !strcmp(h->keys[i], "text") || !strcmp(h->keys[i], "language")) continue;
                if (!first) sb_addc(&o, ',');
                first = 0;
                json_quote(&o, h->keys[i]);
                sb_addc(&o, ':');
                json_dump(&o, h->items[i]);
            }
            sb_addc(&o, '}');
        }
        sb_addc(&o, '}');
        respond_n(c, 200, "application/json", o.p, o.n, xh);
    }
    free(o.p);
    json_free(h);
}

// GET voices / POST select: the worker's reply as the body, minus the framing keys
static void handle_voice(Conn *c, const char *req) {
    const char *err = NULL;
    char *payload;
    size_t n;
    jv *h = tts_call(req, &payload, &n, &err);
    free(payload);
    if (tts_failed(c, h, err)) return;
    sb o = {0};
    sb_addc(&o, '{');
    int first = 1;
    for (size_t i = 0; i < h->n; i++) {
        if (!strcmp(h->keys[i], "ok") || !strcmp(h->keys[i], "bytes")) continue;
        if (!first) sb_addc(&o, ',');
        first = 0;
        json_quote(&o, h->keys[i]);
        sb_addc(&o, ':');
        json_dump(&o, h->items[i]);
    }
    sb_addc(&o, '}');
    json_reply(c, 200, o.p);
    free(o.p);
    json_free(h);
}

// /v1/speak {text, voice?, speed?} = /v1/audio/speech {input, voice?, speed?} with a WAV reply (style is ignored)
static void handle_speak(Conn *c, const char *body_s) {
    const char *err = NULL;
    jv *body = json_parse(body_s, &err);
    const jv *t = body && body->type == J_OBJ ? json_get(body, "text") : NULL;
    if (!t || t->type != J_STR) {
        err_reply(c, 400, "invalid_request_error", err ? err : "text must be a string");
        json_free(body);
        return;
    }
    const jv *v = json_get(body, "voice"), *sp = json_get(body, "speed");
    sb r = {0};
    sb_add(&r, "{\"input\":");
    json_quote(&r, t->str);
    if (v) sb_add(&r, ",\"voice\":"), json_dump(&r, v);
    if (sp) sb_add(&r, ",\"speed\":"), json_dump(&r, sp);
    sb_add(&r, ",\"response_format\":\"wav\"}");
    json_free(body);
    handle_speech(c, r.p);
    free(r.p);
}

#include "see_route.c"  // POST /v1/see (D-FINE-N detector child)
#include "hear_route.c" // POST /v1/hear (whisper-server proxy)

static void route(Conn *c, const char *body) {
    char *q = strchr(c->path, '?');
    if (q) *q = 0;
    const char *p = c->path;
    int get = !strcmp(c->method, "GET"), post = !strcmp(c->method, "POST");
    if (!strcmp(c->method, "OPTIONS")) { // CORS preflight (conn_main already refused a foreign Origin)
        char h[1024];
        int k = snprintf(h, sizeof h,
            "HTTP/1.1 204 No Content\r\nConnection: %s\r\nServer: llb-serve\r\nAllow: GET, POST, OPTIONS\r\n%s",
            c->close || stopping ? "close" : "keep-alive", c->cors);
        if (*c->cors)
            k += snprintf(h + k, sizeof h - (size_t)k,
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: %s\r\n"
                "Access-Control-Max-Age: 600\r\n",
                *c->acrh ? c->acrh : "Content-Type, Authorization");
        k += snprintf(h + k, sizeof h - (size_t)k, "\r\n");
        c->status = 204;
        if (stopping) c->close = 1;
        if (send_all(c->fd, h, (size_t)k)) c->broken = 1;
        c->bytes += k;
        return;
    }
    if (!strcmp(p, "/version") || !strcmp(p, "/v1/version")) {
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        sb o = {0};
        sb_add(&o, "{\"name\":\"llb-serve\",\"version\":\"" LLB_SERVE_VERSION "\",\"git\":\"" LLB_GIT "\",\"model\":");
        json_quote(&o, alias);
        sb_fmt(&o, ",\"slots\":%d,\"keepalive_s\":%d,\"cors\":", n_slots, keepalive_s);
        json_quote(&o, cors ? cors : "local");
        sb_add(&o, ",\"api\":\"openai-v1\"}");
        json_reply(c, 200, o.p);
        free(o.p);
        return;
    }
    if (!strcmp(p, "/health") || !strcmp(p, "/v1/health")) {
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        if (engine_dead || all_dead()) return err_reply(c, 503, "unavailable_error", "engine is not running");
        if (!ready) return err_reply(c, 503, "unavailable_error", "Loading model");
        return json_reply(c, 200, "{\"status\":\"ok\"}");
    }
    if (n_slots * n_banks > 1 && !strcmp(p, "/slots")) { // G6: llama-server's slot monitor, the N>1 server only
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        sb o = {0};
        sb_addc(&o, '[');
        pthread_mutex_lock(&q_lock);
        for (int i = 0; i < n_slots; i++) {
            Slot *x = &slots[i];
            if (n_banks == 1) {
                sb_fmt(&o, "%s{\"id\":%d,\"state\":\"%s\",\"is_processing\":%s,\"n_requests\":%ld,\"pid\":%d", i ? "," : "",
                    i, x->dead ? "dead" : !x->ready ? "loading" : x->busy ? "processing" : "idle", x->busy ? "true" : "false",
                    x->n_req, (int)x->pid);
                if (metrics_base) sb_fmt(&o, ",\"metrics_port\":%d", metrics_base + i); // R5-FIX
                sb_addc(&o, '}');
                continue;
            }
            // G20: one entry per bank; the engine's state is shared by its banks
            for (int b = 0; b < n_banks; b++) {
                sb_fmt(&o, "%s{\"id\":%d,\"engine\":%d,\"bank\":%d,\"state\":\"%s\",\"n_requests\":%ld,\"pid\":%d",
                    i + b ? "," : "", i * n_banks + b, i, b,
                    x->dead ? "dead" : !x->ready ? "loading" : x->busy ? "engine_busy" : "idle", x->bk[b].n_req, (int)x->pid);
                if (metrics_base) sb_fmt(&o, ",\"metrics_port\":%d", metrics_base + i); // R5-FIX
                sb_addc(&o, '}');
            }
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
    if (ear_dir && !strcmp(p, "/v1/audio/transcriptions")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_transcription(c, body);
    }
    if (tts_dir && !strcmp(p, "/v1/audio/speech")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_speech(c, body);
    }
    if (tts_dir && (!strcmp(p, "/v1/audio/voices") || !strcmp(p, "/voice/voices"))) {
        if (!get) return err_reply(c, 405, "invalid_request_error", "use GET");
        return handle_voice(c, "{\"op\":\"voices\"}");
    }
    if (tts_dir && !strcmp(p, "/voice/select")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        const char *err = NULL;
        jv *b = json_parse(body, &err);
        const jv *id = b ? json_get(b, "id") : NULL;
        if (!id || id->type != J_STR) {
            json_free(b);
            return err_reply(c, 400, "invalid_request_error", "body must be {\"id\": \"<voice id>\"}");
        }
        sb r = {0};
        sb_add(&r, "{\"op\":\"select\",\"id\":");
        json_quote(&r, id->str);
        sb_addc(&r, '}');
        json_free(b);
        handle_voice(c, r.p);
        free(r.p);
        return;
    }
    if (!strcmp(p, "/v1/speak")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        if (!tts_dir) return err_reply(c, 503, "unavailable_error", "speech is not enabled on this backend");
        return handle_speak(c, body);
    }
    if (!strcmp(p, "/v1/see")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_see(c, body);
    }
    if (!strcmp(p, "/v1/hear")) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_hear(c, body);
    }
    int comp = !strcmp(p, "/v1/completions") || !strcmp(p, "/completions");
    int chat = !strcmp(p, "/v1/chat/completions") || !strcmp(p, "/chat/completions");
    if (comp || chat) {
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_generate(c, body, chat);
    }
    if (!strcmp(p, "/v1/decide") || !strcmp(p, "/decide")) { // SERVE-BATCH
        if (!post) return err_reply(c, 405, "invalid_request_error", "use POST");
        return handle_decide(c, body);
    }
    err_reply(c, 404, "not_found_error", "File Not Found");
}

static char *find_header(char *head, const char *name); // HARDEN: see below

// HARDEN: strict Content-Length. Returns 0 ok, 1 malformed/duplicate, 2 too large.
static int parse_content_length(const char *head, long *out) {
    int n = 0;
    for (const char *q = head; *q; ) {
        const char *e = strchr(q, '\n');
        size_t L = e ? (size_t)(e - q) : strlen(q);
        if (L > 15 && !strncasecmp(q, "content-length:", 15)) n++;
        if (!e) break;
        q = e + 1;
    }
    *out = 0;
    if (n == 0) return 0;
    if (n > 1) return 1;
    char *cl = find_header((char *)head, "Content-Length");
    if (!cl) return 1;
    while (*cl == ' ' || *cl == '\t') cl++;
    errno = 0;
    char *end = NULL;
    long v = strtol(cl, &end, 10);
    if (errno || end == cl) return 1;
    while (*end == ' ' || *end == '\t') end++;
    if (*end != '\r' && *end != '\n' && *end != '\0') return 1;
    if (v < 0) return 1;
    if ((size_t)v > MAX_BODY) return 2;
    *out = v;
    return 0;
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

// G31: wait until fd is readable within ms. idle = no byte of the request read yet:
// a stopping server then gives up at once on a kept-alive connection, and after
// 0.5 s on a fresh one (its request may still be on the way). 1 = readable,
// 0 = timed out, -1 = stopping.
static int wait_in(int fd, int ms, int idle, int fresh) {
    double t_end = now() + ms / 1000.0, t_stop = 0;
    for (;;) {
        if (idle && stopping) {
            if (!fresh) return -1;
            if (!t_stop) t_stop = now() + 0.5;
            if (now() >= t_stop) return -1;
        }
        double left = t_end - now();
        if (left <= 0) return 0;
        struct pollfd pf = {.fd = fd, .events = POLLIN};
        int r = poll(&pf, 1, left > 0.1 ? 100 : (int)(left * 1000) + 1);
        if (r > 0 || (r < 0 && errno != EINTR)) return 1; // recv reports errors / EOF
    }
}

// a header's value (to the end of its line, trailing blanks cut) into out; 0 if absent
static int header_val(char *head, const char *name, char *out, size_t cap) {
    char *v = find_header(head, name);
    size_t n = 0;
    if (v)
        while (v[n] && v[n] != '\r' && v[n] != '\n') n++;
    while (n && (v[n - 1] == ' ' || v[n - 1] == '\t')) n--;
    if (n >= cap) n = cap - 1;
    if (n) memcpy(out, v, n);
    out[n] = 0;
    return v != NULL;
}

// CORS default-local: http(s)://localhost | 127.0.0.1 | [::1] on any port, plus
// the exact origins listed in --cors ("*" = any origin)
static int origin_ok(const char *o) {
    if (cors && !strcmp(cors, "*")) return 1;
    // App webviews: Tauri ships tauri://localhost, a file:// page sends file:// (or "null" if sandboxed).
    if (!strncasecmp(o, "tauri://", 8) || !strncasecmp(o, "file://", 7)) return 1;
    const char *h = !strncmp(o, "http://", 7) ? o + 7 : !strncmp(o, "https://", 8) ? o + 8 : NULL;
    static const char *locals[] = {"localhost", "127.0.0.1", "[::1]"};
    for (int i = 0; h && i < 3; i++) {
        size_t n = strlen(locals[i]);
        if (strncasecmp(h, locals[i], n)) continue;
        const char *r = h + n;
        if (*r == ':' && r[1])
            for (r++; *r >= '0' && *r <= '9';) r++;
        if (!*r) return 1;
    }
    size_t n = strlen(o);
    for (const char *p = cors; p && *p;) {
        const char *e = strchr(p, ',');
        size_t k = e ? (size_t)(e - p) : strlen(p);
        if (k == n && !strncmp(p, o, n)) return 1;
        p += k + (e != NULL);
    }
    return 0;
}

// one connection: HTTP/1.1 requests in a row (keep-alive) until the client or a
// close-type response ends it, it idles keepalive_s, or the server stops
static void *conn_main(void *arg) {
    Conn *c = arg;
    char *buf = malloc(MAX_HEAD + 1), *body = NULL;
    if (!buf) { // HARDEN: OOM must not NULL-deref
        close(c->fd), free(c), inflight--;
        return NULL;
    }
    size_t got = 0; // buffered bytes: the start of the next (pipelined) request, if any
    for (int nreq = 1;; nreq++) {
        if (nreq > 1) { // a fresh Conn for the next request, same socket and peer
            Conn n = {.fd = c->fd, .slot = -1, .id = ++req_seq};
            memcpy(n.ip, c->ip, sizeof n.ip);
            *c = n;
        }
        double t0 = now();
        int seen = got > 0, ka = keepalive_s > 0 && nreq < keepalive_max, forbid = 0;
        char *end = NULL;
        buf[got] = 0;
        end = strstr(buf, "\r\n\r\n");
        while (!end && got < MAX_HEAD) {
            int w = wait_in(c->fd, got || nreq == 1 ? IO_TIMEOUT_MS : keepalive_s * 1000, !got, nreq == 1);
            if (w <= 0) {
                c->close = 1;
                if (got && !w) err_reply(c, 408, "invalid_request_error", "timed out reading the request headers");
                goto done;
            }
            ssize_t k = recv(c->fd, buf + got, MAX_HEAD - got, 0);
            if (k <= 0) {
                c->close = 1;
                goto done;
            }
            seen = 1;
            got += (size_t)k;
            buf[got] = 0;
            end = strstr(buf, "\r\n\r\n");
        }
        c->close = 1; // until the request is framed
        if (!end) {
            err_reply(c, 413, "invalid_request_error", "headers too large");
            goto done;
        }
        *end = 0;
        char ver[16] = "", v[256];
        if (sscanf(buf, "%15s %255s %15s", c->method, c->path, ver) < 2) {
            err_reply(c, 400, "invalid_request_error", "bad request line");
            goto done;
        }
        if (strcmp(ver, "HTTP/1.1")) ka = 0; // HTTP/1.0: one request per connection
        if (header_val(buf, "Connection", v, sizeof v) && strcasestr(v, "close")) ka = 0;
        char *te = find_header(buf, "Transfer-Encoding");
        if (te && !strncasecmp(te, "chunked", 7)) {
            err_reply(c, 411, "invalid_request_error", "chunked request bodies are not supported; send Content-Length");
            goto done;
        }
        long len = 0;
        int clr = parse_content_length(buf, &len);
        if (clr == 1) {
            err_reply(c, 400, "invalid_request_error", "invalid Content-Length (duplicate, non-numeric, or trailing garbage)");
            goto done;
        }
        if (clr == 2) {
            err_reply(c, 413, "invalid_request_error", "body too large");
            goto done;
        }
        if ((!cors || strcmp(cors, "none")) && header_val(buf, "Origin", v, sizeof v)) {
            if (!origin_ok(v)) forbid = 1;
            else if (cors && !strcmp(cors, "*")) snprintf(c->cors, sizeof c->cors, "Access-Control-Allow-Origin: *\r\n");
            else snprintf(c->cors, sizeof c->cors, "Access-Control-Allow-Origin: %s\r\nVary: Origin\r\n", v);
            header_val(buf, "Access-Control-Request-Headers", c->acrh, sizeof c->acrh);
            for (char *q = c->acrh; *q; q++) // echoed back: header-name characters only
                if (!(*q == '-' || *q == ',' || *q == ' ' || (*q >= '0' && *q <= '9') || ((*q | 32) >= 'a' && (*q | 32) <= 'z')))
                    *q = '_';
        }
        size_t hn = (size_t)(end + 4 - buf), have = got - hn;
        if (have > (size_t)len) have = (size_t)len;
        if (have < (size_t)len && header_val(buf, "Expect", v, sizeof v) && !strcasecmp(v, "100-continue") &&
            send_all(c->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25)) {
            c->broken = 1;
            goto done;
        }
        body = malloc((size_t)len + 1);
        memcpy(body, buf + hn, have);
        got -= hn + have; // what follows this body stays buffered for the next request
        memmove(buf, buf + hn + have, got);
        while (have < (size_t)len) {
            int w = wait_in(c->fd, IO_TIMEOUT_MS, 0, 0);
            ssize_t k = w > 0 ? recv(c->fd, body + have, (size_t)len - have, 0) : 0;
            if (k <= 0) {
                if (!w) err_reply(c, 408, "invalid_request_error", "timed out reading the request body");
                goto done;
            }
            have += (size_t)k;
        }
        body[len] = 0;
        c->close = !ka;
        if (forbid)
            err_reply(c, 403, "permission_error",
                "cross-origin request refused: llb-serve allows local origins only (see --cors)");
        else route(c, body);
    done:
        if (seen) {
            char sl[24] = "";
            if (n_slots * n_banks > 1 && c->slot >= 0) snprintf(sl, sizeof sl, " slot=%d", c->slot);
            logf_("#%ld %s \"%s %s\" %d %ldB %.1fms prompt=%d gen=%d%s%s%s%s", c->id, c->ip, c->method, c->path,
                c->status, c->bytes, (now() - t0) * 1000, c->prompt_n, c->gen_n, c->stream ? " stream" : "",
                c->broken ? " client-gone" : "", sl, nreq > 1 ? " reused" : "");
        }
        free(body);
        body = NULL;
        if (c->close || c->broken || stopping) break;
    }
    shutdown(c->fd, SHUT_WR);
    close(c->fd);
    free(buf);
    free(c);
    inflight--;
    return NULL;
}

static void on_signal(int s) {
    (void)s;
    stopping++; // a second SIGTERM/SIGINT skips the drain
    if (write(wake[1], "x", 1) < 0) {}
}

static void usage(const char *a0) {
    fprintf(stderr,
        "usage: %s [--decide-path | --exact] [--prefill exact|fast] [--prefix-cache] [--host H] [--port P] [--alias NAME] [--chat-max-tokens N] [--drain-seconds S]\n"
        "          [--keepalive-seconds S] [--cors none|*|ORIGIN,...] [--tts DIR] [--ear DIR [--ear-llm-cpu]]\n"
        "          [--parallel N] [--banks B] [--native-batch N [--native-cont]] [--heads SPEC] [--discipline strict|normal]\n"
        "          [--affinity-min N] [--structured TOKENIZER.BIN]\n"
        "          [--max-body-mb N] [--mm none|image|audio|image,audio] [--mm-remote off|on|private]\n"
        "          [--mm-max-image-mb N] [--mm-max-images N] [--vis-tower CUDAVIS_CLI --vis-mmproj MMPROJ]\n"
        "          [--glue-pack DIR --glue-rows glue_rows.py --glue-cache DIR [--glue-python PY]]\n"
        "          --engine PATH [-- ENGINE ARGS...]\n"
        "       %s --version\n"
        "  --parallel N (-np): N slots = N engine children (each loads its own weights + KV), default 1\n"
        "  --banks B: B sequence states (KV + GDN state + prefix cache) per engine, one weights copy; default 1\n"
        "  --native-batch N: a /v1/decide batch runs N tasks a forward in the engine (2..8; exact per item); default off\n"
        "  --heads name=gguf[,...]: more decide heads (engine LLB_HEADS); a decide's \"head\":\"name\" selects one per item\n"
        "  --native-cont: with --native-batch, a task joins a free row at a step boundary (continuous batching); default off\n"
        "  --discipline strict|normal: strict = a decide with no \"head\" key runs the registry head \"strict\" (needs\n"
        "      --heads strict=GGUF); normal (default) = head0, the LLB_GGUF head\n"
        "  --affinity-min BYTES: shortest shared prompt prefix that routes to a bank (default 256)\n"
        "  --ear DIR: the ear (llb-ear beside llb-serve; DIR holds Qwen3-ASR-0.6B-Q8_0.gguf + mmproj-Qwen3-ASR-0.6B-bf16.gguf):\n"
        "      POST /v1/audio/transcriptions, and chat input_audio parts become the ear's transcript in the user turn;\n"
        "      --ear-llm-cpu runs its decoder on the CPU (the tower stays on the GPU)\n"
        "  LLB_METRICS_PORT=P: the engine's /metrics listener; with --parallel N, engine[i] listens on P+i\n"
        "  --keepalive-seconds S: idle time a kept-alive connection may wait for its next request,\n"
        "      default 30; 0 = one request per connection (SSE streams always close)\n"
        "  --cors: browser Origins allowed besides http(s)://localhost|127.0.0.1|[::1] and tauri://|file:// (always allowed);\n"
        "      a request with any other Origin is refused (403). '*' = any origin; 'none' = no CORS headers and\n"
        "      no Origin check (the pre-0.9 behaviour)\n"
        "  SIGTERM/SIGINT: stop accepting, drain up to --drain-seconds (default 120), then signal any engine\n"
        "      still busy; a second signal skips the drain\n"
        "  --tts DIR: speech (POST /v1/audio/speech, GET /v1/audio/voices, POST /voice/select) from a Kokoro pack\n"
        "             (tools/tts/kokoro_tts.py fetch DIR); env LLB_TTS_PYTHON (python3), LLB_TTS_SCRIPT (kokoro_tts.py\n"
        "             beside llb-serve), BEND_KOKORO_THREADS (4), BEND_VOICE (default voice); POST /v1/speak {text}\n"
        "             is the same speech under the senses name\n"
        "  POST /v1/see (503 until set): env LLB_SEE_SCRIPT (see_worker.py), LLB_SEE_MODEL (dfine_n_coco.onnx),\n"
        "             LLB_SEE_PYTHON (python3), LLB_SEE_LABELS, LLB_SEE_EP auto|cuda|cpu, LLB_SEE_THREADS (4)\n"
        "  POST /v1/hear (503 until set): env LLB_HEAR_URL=http://127.0.0.1:PORT/inference (a running whisper.cpp\n"
        "             whisper-server), LLB_HEAR_MODEL_ID, LLB_HEAR_MODEL_PATH, LLB_HEAR_MODEL_SHA256, LLB_HEAR_ENGINE\n"
        "  ENGINE ARGS default: --threads 16 -- $SYS1_SIDECAR 0 --steps 4096  (the SYS1-DUAL launch)\n"
        "  --exact (default): exact tier; prefill/cache opt-ins below still apply\n"
        "  --prefill exact|fast: exact default; fast changes numerics, refuses unsupported models\n"
        "  --prefix-cache: enable engine prefix cache (off by default, also under --exact)\n"
        "  --decide-path: LLB_PREFILL=fast + LLB_PREFIX_CACHE=1 + greedy LLB_SAMPLE (DECIDE-PATH.md);\n"
        "                 needs a GGUF whose GEMMs are all Q4_K/Q5_K/Q6_K/Q8_0, checked at start (S6-GUARD):\n"
        "                 any other model serves --exact, with one stderr line. 9B clef-flash Q4_K_S (F32\n"
        "                 ssm_alpha/beta): fast only with env LLB_PF_SSMQ8=1, the labeled fast tier (Q8_0\n"
        "                 shadow; mean KL to exact 0.0169, top-1 90%% at word starts; decides stay exact)\n"
        "  env: LLB_GGUF (the model, required by the engine), TMPDIR (owned scratch)\n",
        a0, a0);
    exit(2);
}

// HARDEN: a busy engine that has produced no line for engine_timeout_s is killed; its
// readers see EOF and answer 503, so one wedged request cannot hold a slot forever.
static void *watchdog_main(void *arg) {
    (void)arg;
    while (!stopping) {
        usleep(500000);
        for (int i = 0; i < n_slots; i++) {
            Slot *x = &slots[i];
            pthread_mutex_lock(&q_lock);
            if (!x->busy || x->dead || x->pid <= 0) { pthread_mutex_unlock(&q_lock); continue; }
            double idle = now() - x->t_last;
            if (engine_timeout_s > 0 && idle > engine_timeout_s) {
                logf_("watchdog: engine[%d] silent %.0fs (limit %ds) — killing pid %d", i, idle, engine_timeout_s, (int)x->pid);
                kill(x->pid, SIGKILL);
                x->dead = 1;
                pthread_cond_broadcast(&q_cond);
            }
            pthread_mutex_unlock(&q_lock);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *engine = NULL;
    char **eargs = NULL;
    mm_caps_default(&mmc);
    int ean = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--version")) return printf("llb-serve %s (git %s)\n", LLB_SERVE_VERSION, LLB_GIT), 0;
        if (!strcmp(a, "--")) {
            eargs = argv + i + 1, ean = argc - i - 1;
            break;
        }
        if (!strcmp(a, "--decide-path")) { decide_path = 1; continue; }
        if (!strcmp(a, "--exact")) { decide_path = 0; continue; }
        if (!strcmp(a, "--prefix-cache")) { prefix_cache = 1; continue; }
        if (!strcmp(a, "--prefill")) {
            if (i + 1 >= argc || (strcmp(argv[i + 1], "exact") && strcmp(argv[i + 1], "fast")))
                fprintf(stderr, "serve: --prefill takes exact|fast\n"), exit(2);
            prefill_fast = !strcmp(argv[++i], "fast");
            continue;
        }
        if (i + 1 >= argc) usage(argv[0]);
        if (!strcmp(a, "--host")) host = argv[++i];
        else if (!strcmp(a, "--port")) port = atoi(argv[++i]);
        else if (!strcmp(a, "--alias")) alias = argv[++i];
        else if (!strcmp(a, "--engine")) engine = argv[++i];
        else if (!strcmp(a, "--chat-max-tokens")) chat_default_n = atoi(argv[++i]);
        else if (!strcmp(a, "--drain-seconds")) drain_s = atoi(argv[++i]);
        else if (!strcmp(a, "--parallel") || !strcmp(a, "-np")) n_slots = atoi(argv[++i]);
        else if (!strcmp(a, "--native-batch")) native_batch = atoi(argv[++i]);
        else if (!strcmp(a, "--heads")) setenv("LLB_HEADS", argv[++i], 1); // BATCH-NATIVE-3: the head registry
        else if (!strcmp(a, "--native-cont")) native_cont = 1;
        else if (!strcmp(a, "--discipline")) { // ENGINE-HEAD27: the decide default head (engine LLB_DISCIPLINE)
            const char *v = argv[++i];
            if (strcmp(v, "strict") && strcmp(v, "normal")) usage(argv[0]);
            setenv("LLB_DISCIPLINE", v, 1);
        }
        else if (!strcmp(a, "--structured")) { // G16: takes the engine's tokenizer.bin (the mask's vocab)
            structured_on = 1;
            setenv("LLB_TOKENIZER", argv[++i], 1);
        }
        else if (!strcmp(a, "--max-body-mb")) MAX_BODY = (size_t)atoi(argv[++i]) << 20;
        else if (!strcmp(a, "--max-conns")) max_conns = atoi(argv[++i]);           // HARDEN
        else if (!strcmp(a, "--send-timeout")) send_timeout_s = atoi(argv[++i]);   // HARDEN
        else if (!strcmp(a, "--engine-timeout")) engine_timeout_s = atoi(argv[++i]); // HARDEN
        else if (!strcmp(a, "--mm")) { // what the engine takes through the media hook: none | image | audio | image,audio
            const char *v = argv[++i];
            mmc.images = !!strstr(v, "image"), mmc.audio = !!strstr(v, "audio");
        } else if (!strcmp(a, "--mm-remote")) { // off (default) | on (public hosts only) | private (any host)
            const char *v = argv[++i];
            mmc.allow_remote = !strcmp(v, "on") || !strcmp(v, "private");
            mmc.allow_private = !strcmp(v, "private");
        }
        else if (!strcmp(a, "--mm-max-image-mb")) mmc.image_bytes = (size_t)atoi(argv[++i]) << 20;
        else if (!strcmp(a, "--mm-max-images")) mmc.max_images = atoi(argv[++i]);
        else if (!strcmp(a, "--banks")) n_banks = atoi(argv[++i]);
        else if (!strcmp(a, "--affinity-min")) affinity_min = atol(argv[++i]);
        else if (!strcmp(a, "--glue-pack")) glue_pack = argv[++i];
        else if (!strcmp(a, "--glue-rows")) glue_rows = argv[++i];
        else if (!strcmp(a, "--glue-python")) glue_py = argv[++i];
        else if (!strcmp(a, "--glue-cache")) glue_cache = argv[++i];
        else if (!strcmp(a, "--keepalive-seconds")) keepalive_s = atoi(argv[++i]);
        else if (!strcmp(a, "--cors")) cors = argv[++i];
        else if (!strcmp(a, "--tts")) tts_dir = argv[++i];
        else if (!strcmp(a, "--ear")) ear_dir = argv[++i];
        else if (!strcmp(a, "--vis-tower")) vis_tower = argv[++i]; // MIMG
        else if (!strcmp(a, "--vis-mmproj")) vis_mmproj = argv[++i];
        else if (!strcmp(a, "--ear-llm-cpu")) ear_llm_cpu = 1;
        else usage(argv[0]);
    }
    if (!engine || port <= 0 || port > 65535 || chat_default_n < 1 || n_slots < 1 || n_slots > 64 || n_banks < 1 || n_banks > 16 ||
        keepalive_s < 0 || keepalive_s > 3600 || (cors && !*cors))
        usage(argv[0]);
    if (glue_pack && (!glue_rows || !glue_cache)) fprintf(stderr, "serve: --glue-pack needs --glue-rows and --glue-cache\n"), exit(2);
    if (!vis_tower != !vis_mmproj) fprintf(stderr, "serve: --vis-tower and --vis-mmproj go together\n"), exit(2);
    if (vis_tower) { // MIMG: images need the exact tier (dual.c refuses them on fast prefill) and --mm image
        if (decide_path) fprintf(stderr, "serve: --vis-tower needs --exact (images run on the exact tier only)\n"), exit(2);
        mmc.images = 1;
        char c[4200];
        snprintf(c, sizeof c, "sha256sum '%s'", vis_mmproj);
        FILE *pf = popen(c, "r");
        if (!pf || fscanf(pf, "%64s", vis_mmsha) != 1 || strlen(vis_mmsha) != 64) fprintf(stderr, "serve: cannot hash --vis-mmproj %s\n", vis_mmproj), exit(2);
        pclose(pf);
        setenv("LLB_VIS_MMPROJ_SHA256", vis_mmsha, 1);
        fprintf(stderr, "serve: vision tower %s, mmproj %s (sha256 %.12s...)\n", vis_tower, vis_mmproj, vis_mmsha);
    } else unsetenv("LLB_VIS_MMPROJ_SHA256");
    if (ear_dir) mmc.audio = 1; // EARS-CANON: the ear takes audio (transcribed before the engine sees the request)
    // G20: the engine reads its bank count from the env (dual.c refuses "bank" >= it)
    if (n_banks > 1) {
        char nb[16];
        snprintf(nb, sizeof nb, "%d", n_banks);
        setenv("LLB_BANKS", nb, 1);
    } else unsetenv("LLB_BANKS");
    // BATCH-NATIVE: the engine's batch width from the env (off: unset, the engine never batches)
    if (native_batch < 0 || native_batch > 8 || (native_batch > 1 && n_banks > 1))
        fprintf(stderr, "serve: --native-batch takes 0..8 and no --banks\n"), exit(2);
    if (native_cont && native_batch < 2) fprintf(stderr, "serve: --native-cont needs --native-batch N\n"), exit(2);
    if (native_cont) setenv("LLB_BATCH_OOO", "1", 1); // the engine answers each item when it is done
    else unsetenv("LLB_BATCH_OOO");
    if (native_batch > 1) {
        char nb[16];
        snprintf(nb, sizeof nb, "%d", native_batch);
        setenv("LLB_BATCH", nb, 1);
    } else unsetenv("LLB_BATCH");
    if (!getenv("LLB_GGUF")) fprintf(stderr, "serve: LLB_GGUF is unset (the engine will refuse)\n"), exit(2);
    const char *mpe = getenv("LLB_METRICS_PORT");
    if (mpe && *mpe && n_slots > 1) { // R5-FIX: engine[i] on LLB_METRICS_PORT + i
        metrics_base = atoi(mpe);
        if (metrics_base <= 0 || metrics_base + n_slots - 1 > 65535 || (port >= metrics_base && port < metrics_base + n_slots))
            fprintf(stderr, "serve: LLB_METRICS_PORT=%s with --parallel %d needs ports %d..%d (not --port %d)\n", mpe, n_slots,
                metrics_base, metrics_base + n_slots - 1, port), exit(2);
    }
    char why[320];
    if (prefill_fast && fast_unsupported(getenv("LLB_GGUF"), why, sizeof why))
        fprintf(stderr, "serve: --prefill fast refused for this model (%s)\n", why), exit(2);
    if (decide_path && fast_unsupported(getenv("LLB_GGUF"), why, sizeof why)) { // S6-GUARD: fall back, never exit
        fprintf(stderr, "serve: --decide-path refused for this model (%s; fast prefill needs Q4_K/Q5_K/Q6_K/Q8_0 GEMMs): serving --exact\n", why);
        decide_path = 0;
    }
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
    engine_argv = ev;
    slots = calloc((size_t)n_slots, sizeof *slots);
    for (int i = 0; i < n_slots; i++) {
        slots[i].idx = i;
        slots[i].bk = calloc((size_t)n_banks, sizeof(Bank));
        engine_start(&slots[i], ev);
    }
    { // HARDEN: the engine watchdog
        pthread_t wd;
        if (!pthread_create(&wd, NULL, watchdog_main, NULL)) pthread_detach(wd);
        else logf_("watchdog: pthread_create failed; no engine timeout");
    }
    if (metrics_base)
        logf_("metrics: engine[i] on http://%s:%d+i/metrics (LLB_METRICS_PORT + slot, %d..%d)",
            getenv("LLB_METRICS_HOST") && *getenv("LLB_METRICS_HOST") ? getenv("LLB_METRICS_HOST") : "127.0.0.1", metrics_base,
            metrics_base, metrics_base + n_slots - 1);
    if (tts_dir) tts_start(ls); // G41: before the glue worker (its pipes are not CLOEXEC)
    if (ear_dir) ear_start(ls); // EARS-CANON: same rule
    if (glue_pack) glue_start(); // G24: one glue worker shared by every slot (glue_lock)
    see_cfg_init();  // senses: read LLB_SEE_* / LLB_HEAR_* once, before any connection thread
    hear_cfg_init();
    if (n_slots == 1)
        logf_("listening on http://%s:%d model=%s engine=%s pid=%d (loading)%s", host, port, alias, engine, (int)slots[0].pid,
            structured_on ? " structured=on" : "");
    else
        logf_("listening on http://%s:%d model=%s engine=%s slots=%d (loading)%s", host, port, alias, engine, n_slots,
            structured_on ? " structured=on" : "");

    while (!stopping) {
        struct pollfd pf[2] = {{.fd = ls, .events = POLLIN}, {.fd = wake[0], .events = POLLIN}};
        if (poll(pf, 2, 1000) < 0 && errno != EINTR) break;
        for (int i = 0; i < n_slots; i++) {
            Slot *x = &slots[i];
            int st;
            pthread_mutex_lock(&q_lock);
            if (x->pid > 0 && waitpid(x->pid, &st, WNOHANG) == x->pid) {
                x->pid = -x->pid;
                x->status = st;
                x->dead = 1;
                ready_check();
                if (n_slots == 1) logf_("engine exited (status %d) — serving 503", st);
                else logf_("engine[%d] exited (status %d)%s", i, st, all_dead() ? " — serving 503" : "");
                pthread_cond_broadcast(&q_cond);
            }
            pthread_mutex_unlock(&q_lock);
        }
        pthread_mutex_lock(&q_lock);
        if (all_dead()) engine_dead = 1;
        pthread_mutex_unlock(&q_lock);
        if (!(pf[0].revents & POLLIN)) continue;
        struct sockaddr_storage sa2;
        socklen_t sl = sizeof sa2;
        int fd = accept(ls, (struct sockaddr *)&sa2, &sl);
        if (fd < 0) continue;
        // G31: no Nagle. Small writes (a kept-alive reply, each SSE token event) go out
        // at once instead of waiting up to ~40 ms on the client's delayed ACK.
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        // HARDEN: a stalled reader must not park a thread (and a slot) forever.
        struct timeval stv = {.tv_sec = send_timeout_s, .tv_usec = 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof stv);
        // HARDEN: bounded connections. Above the cap: answer 503 and close, no thread.
        if (max_conns > 0 && atomic_load(&inflight) >= max_conns) {
            static const char busy_body[] =
                "{\"error\":{\"message\":\"server is at its connection limit\",\"type\":\"unavailable_error\"}}";
            char b[320];
            int n = snprintf(b, sizeof b,
                "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                "Content-Length: %zu\r\nConnection: close\r\n\r\n%s", strlen(busy_body), busy_body);
            send(fd, b, (size_t)n, MSG_NOSIGNAL);
            close(fd);
            continue;
        }
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
    while (inflight > 0 && now() - td < drain_s && stopping < 2) usleep(20000);
    // G31: past the drain nothing new starts (queued requests get 503), and a slot
    // still busy is a wedged engine or an overlong request: SIGTERM its engine (the
    // request's reader sees EOF and frees the slot), then SIGKILL. Bounded.
    closing = 1;
    pthread_mutex_lock(&q_lock);
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_lock);
    for (int round = 0; round < 2; round++) {
        int sig = round ? SIGKILL : SIGTERM, n = 0;
        pthread_mutex_lock(&q_lock);
        for (int i = 0; i < n_slots; i++)
            if (slots[i].busy && slots[i].pid > 0) kill(slots[i].pid, sig), n++;
        pthread_mutex_unlock(&q_lock);
        if (!n) break;
        logf_("shutdown: %d engine(s) still busy after the drain: %s", n, round ? "SIGKILL" : "SIGTERM");
        for (double tw = now(); now() - tw < 10; usleep(20000)) {
            int busy = 0;
            pthread_mutex_lock(&q_lock);
            for (int i = 0; i < n_slots; i++) busy |= slots[i].busy;
            pthread_mutex_unlock(&q_lock);
            if (!busy) break;
        }
    }
    // take every free slot so nothing writes to an engine's stdin past this point
    pthread_mutex_lock(&q_lock);
    for (int i = 0; i < n_slots; i++) slots[i].busy = 1;
    pthread_mutex_unlock(&q_lock);
    if (inflight == 0) for (int i = 0; i < n_slots; i++) fclose(slots[i].in); // HARDEN: never race conn threads
    else logf_("shutdown: %d connection(s) still in flight; leaving engine pipes to the kills", (int)inflight);
    if (glue_pid > 0) fclose(glue_in), waitpid(glue_pid, NULL, 0); // the worker exits on EOF
    for (int i = 0; i < n_slots; i++) {
        pid_t pid = slots[i].pid < 0 ? -slots[i].pid : slots[i].pid;
        int st = slots[i].status, rc = slots[i].pid < 0 ? pid : -1;
        char tag[16] = "";
        if (n_slots > 1) snprintf(tag, sizeof tag, "[%d]", i);
        // an engine still loading (or a second signal) does not wait for EOF
        double eof_s = !slots[i].ready || stopping >= 2 ? 0 : 30;
        for (double tw = now(); rc != pid && now() - tw < eof_s; usleep(50000))
            if ((rc = waitpid(pid, &st, WNOHANG)) == pid) break;
        if (rc != pid && (rc = waitpid(pid, &st, WNOHANG)) != pid) {
            logf_("engine%s did not exit on EOF in %.0f s: SIGTERM", tag, eof_s);
            kill(pid, SIGTERM);
            for (double tw = now(); rc != pid && now() - tw < 10; usleep(50000)) rc = waitpid(pid, &st, WNOHANG);
            if (rc != pid) {
                logf_("engine%s ignored SIGTERM for 10 s: SIGKILL", tag);
                kill(pid, SIGKILL);
                waitpid(pid, &st, 0);
            }
        }
        logf_("shutdown complete: drained in %.1f ms, %d left, engine%s exit %s %d", (now() - td) * 1000, (int)inflight,
            tag, WIFEXITED(st) ? "rc" : "signal", WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
    }
    if (tts_dir && tts_pid > 0) { // after the engines: EOF on its stdin, the worker exits on its own
        pthread_mutex_lock(&tts_lock); // no render in flight past here
        fclose(tts_in);
        int st = 0, rc = -1;
        for (double tw = now(); now() - tw < 10; usleep(50000))
            if ((rc = waitpid(tts_pid, &st, WNOHANG)) == tts_pid) break;
        if (rc != tts_pid) kill(tts_pid, SIGTERM), waitpid(tts_pid, &st, 0);
        logf_("tts exit %s %d", WIFEXITED(st) ? "rc" : "signal", WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
        pthread_mutex_unlock(&tts_lock);
    }
    if (ear_dir && ear_pid > 0) { // EARS-CANON: the same for the ear (it exits on EOF; its stderr reader then ends)
        pthread_mutex_lock(&ear_lock); // no transcription in flight past here
        fclose(ear_in);
        int st = 0, rc = -1;
        for (double tw = now(); now() - tw < 10; usleep(50000))
            if ((rc = waitpid(ear_pid, &st, WNOHANG)) == ear_pid) break;
        if (rc != ear_pid) kill(ear_pid, SIGTERM), waitpid(ear_pid, &st, 0);
        logf_("ear exit %s %d", WIFEXITED(st) ? "rc" : "signal", WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
        pthread_mutex_unlock(&ear_lock);
    }
    if (see_w.pid > 0) { // the /v1/see detector child, if one was started
        pthread_mutex_lock(&see_lock);
        see_worker_stop();
        pthread_mutex_unlock(&see_lock);
    }
    return 0;
}
