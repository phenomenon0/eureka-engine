// speak_route.h — POST /v1/speak: text -> 24 kHz mono PCM16 WAV bytes.
//
// M2-SPEAK-SCAFFOLD (metal-5of5). This route is deliberately engine-free: it
// shells a configured P3 synthesis command and streams the WAV file back. The
// HTTP server owns the socket; this module never sees a Conn, so it compiles
// and unit-tests without serve.c, without the engine, and without a GPU.
//
// Contract (see SCHEMAS.md, verb "speak"):
//   POST /v1/speak
//   {"text": "...", "voice": "af_heart", "style": "neutral"}
//   -> 200 audio/wav  (24 000 Hz, mono, signed 16-bit LE)
//
// G2P: the configured command MUST use the Apache-2.0 Misaki frontend
// (misaki_frontend.py). The espeak-ng CLI is GPL-3.0-or-later and is blocked
// in the shipping path by tools/check_no_gpl_tts.py (X-LIC-ESPEAK-GATE).
#ifndef SPEAK_ROUTE_H
#define SPEAK_ROUTE_H

#include <stddef.h>

// Default launcher, overridable at build time:
//   cc -DSPEAK_P3_CMD_DEFAULT='"/path/to/speak-p3.sh"' ...
#ifndef SPEAK_P3_CMD_DEFAULT
#define SPEAK_P3_CMD_DEFAULT "/usr/local/libexec/eureka/speak-p3.sh"
#endif

typedef struct speak_cfg {
    // NULL-terminated argv prefix, argv[0] = the launcher. execv, never a shell.
    const char *const *argv;
    int cpu;         // pass --cpu (SwiftShader / no GPU lock); 1 = CPU-only
    int timeout_ms;  // synthesis deadline; <=0 -> 180000
    const char *tmpdir; // scratch dir; NULL -> $TMPDIR, then /tmp
} speak_cfg;

// The transport. `head` must emit the status line, Content-Length, and the
// caller's extra headers; `send` must write exactly n bytes (binary-safe).
// Both return 0 on success; `send` returns nonzero when the client is gone.
typedef struct speak_sink {
    void *ctx;
    int (*send)(void *ctx, const void *buf, size_t n);
    void (*head)(void *ctx, int status, const char *ctype, size_t len, const char *extra);
} speak_sink;

// Handles one request body and writes exactly one response through `sink`.
// `body` must be NUL-terminated at body[body_len] (serve.c guarantees this).
// Returns 0 when a response was written, -1 when the sink failed.
int speak_route_handle(const speak_cfg *cfg, const speak_sink *sink,
                       const char *body, size_t body_len);

// Exposed for the test harness: validates a body and, on success, returns the
// chosen voice (static string) and style preset index. 0 = ok, -1 = rejected.
int speak_route_parse(const char *body, size_t body_len, const char **voice,
                      const char **style, const char **err);

#endif
