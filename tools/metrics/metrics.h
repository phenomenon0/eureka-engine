// metrics.h: W1-METRICS production telemetry (docs: METRICS.md in the lane run dir,
// copied to tools/metrics/README.md). Plain C11 + pthreads, no other dependency.
//
// Two layers:
//   1. a per-request timer (llb_req) the request side fills as the request moves
//      arrive -> start (engine pass begins) -> token ... token -> end;
//   2. process-wide aggregates (counters, gauges, histograms, a sliding window
//      for percentiles) that llb_req_end() folds each request into, rendered as
//      Prometheus text by llb_metrics_render().
// Every global update takes one mutex, so a server thread may render while the
// engine thread records. llb_req itself is owned by one thread (no lock).
//
// Wiring today: tools/sys1dual/dual.c (the LLB_SYS1 stdin engine). W1-SERVER calls
// the same functions around its own request handling and serves
// llb_metrics_render() on GET /metrics; with no server, LLB_METRICS_PORT starts
// llb_metrics_listen()'s standalone listener.
#ifndef LLB_METRICS_H
#define LLB_METRICS_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// one request's clocks (CLOCK_MONOTONIC seconds) and counts
typedef struct llb_req {
    double t_arrive;   // request read / accepted (queue time starts)
    double t_start;    // engine pass starts (prefill begins)
    double t_first;    // first generated token emitted (0 = none yet)
    double t_last;     // last generated token emitted
    double t_end;      // response written
    int prompt_n;      // prompt tokens of the request
    int cached_n;      // prompt tokens served from retained/cached KV (not prefilled)
    int gen_n;         // generated tokens emitted
    int drafted;       // speculative tokens drafted (0 when spec is off)
    int accepted;      // speculative tokens accepted
} llb_req;

double llb_now(void);   // CLOCK_MONOTONIC seconds

void llb_req_begin(llb_req *r);                                  // zero + t_arrive = now
void llb_req_start(llb_req *r, int prompt_n, int cached_n);      // engine pass begins now
void llb_req_token_at(llb_req *r, double t);                     // one token emitted at t (records ITL live)
void llb_req_token(llb_req *r);                                  // llb_req_token_at(r, llb_now())
void llb_req_spec(llb_req *r, int drafted, int accepted);        // add spec counts
// fold the request into the aggregates. mode: "generate" | "decide" | any short
// label; outcome: "ok" | "retained" | "refused" | "error". t_end = now.
void llb_req_end(llb_req *r, const char *mode, const char *outcome);

// llama.cpp-style "timings" object for one request (after llb_req_end):
// {"cache_n","prompt_n","prompt_ms","prompt_per_token_ms","prompt_per_second",
//  "predicted_n","predicted_ms","predicted_per_token_ms","predicted_per_second",
//  "ttft_ms","queue_ms","total_ms","draft_n","draft_n_accepted"}
// Returns the length written (snprintf semantics: >= cap means truncated).
int llb_req_timings_json(const llb_req *r, char *buf, size_t cap);

void llb_metrics_queue_depth(int depth);   // requests waiting (server queue)
void llb_metrics_cache(int hit);           // extra cache lookups outside llb_req (KV-prefix lane)

// Prometheus text exposition (format 0.0.4). Returns a malloc'd NUL-terminated
// string the caller frees; *len (optional) = its length.
char *llb_metrics_render(size_t *len);

// standalone listener: a detached thread serving GET /metrics (and /health) on
// host:port. Returns 0 on success, -1 (errno set) if bind/listen failed.
int llb_metrics_listen(const char *host, int port);

// tokstream.c: a FILE* that keeps every byte written (buffer *buf, length *len,
// NUL-terminated, caller frees *buf after fclose) and calls llb_req_token(r) for
// every '\n' as it is written (line-buffered). r == NULL keeps bytes only.
FILE *llb_token_stream(llb_req *r, char **buf, size_t *len);
// complete lines already buffered in f and not yet read (glibc; lower bound)
int llb_stdin_queued(FILE *f);

// percentile of a sorted array, linear interpolation between closest ranks
// (numpy's default method): q in [0,1]; n == 0 -> NaN.
double llb_percentile(const double *sorted, size_t n, double q);

// test hook: reset every aggregate to its startup state
void llb_metrics_reset(void);

#ifdef __cplusplus
}
#endif
#endif
