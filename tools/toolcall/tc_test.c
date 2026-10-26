// tools/toolcall/tc_test.c — driver for host/llb_toolcall.h, used by run_tests.py.
//   tc_test render ADD_GEN ENABLE_THINKING < request.json
//       prompt bytes on stdout, exit 0; on error "ERROR: <msg>" on stdout, exit 3
//   tc_test parse TOOLS_FILE|- REASONING NRANDOM SEED < stream
//       feeds the stream whole, in 1-byte chunks, at every 2-chunk split and in NRANDOM
//       random chunkings; prints the reference event list as JSON (adjacent CONTENT /
//       REASONING deltas coalesced) and "runs= mismatches= utf8_splits=" on stderr;
//       exit 1 if any chunking gives different events or a delta starts mid-code-point.
// Build: gcc -std=c11 -O2 -Wall -Wextra -Werror -Ihost tools/toolcall/tc_test.c -lm
#include "llb_toolcall_impl.h"

static char* slurp(FILE* f, size_t* n) {
  LlbBuf b = {0};
  char t[65536];
  size_t k;
  llb_buf_put(&b, "", 0);
  while ((k = fread(t, 1, sizeof t, f)) > 0) llb_buf_put(&b, t, k);
  *n = b.n;
  return b.p;
}

static const char* ev_name(LlbTcEvType t) {
  return t == LLB_TCE_CONTENT ? "content" : t == LLB_TCE_REASONING ? "reasoning" : t == LLB_TCE_TOOL_CALL ? "tool_call" : "error";
}

// one run over the given cut points; returns the canonical event JSON (malloc'd)
static char* run(const char* tools, int reasoning, const char* s, size_t n, const size_t* cuts, size_t nc, int* utf8bad) {
  LlbTcParser* P = llb_tcp_new(tools);
  if (!P) { fprintf(stderr, "bad tools json\n"); exit(2); }
  llb_tcp_set_reasoning(P, reasoning);
  LlbBuf out = {0}, pend = {0};
  LlbTcEvType pt = LLB_TCE_NONE;
  int first = 1;
  llb_buf_put(&out, "[", 1);
  llb_buf_put(&pend, "", 0);
  size_t at = 0;
  for (size_t c = 0; c <= nc; c += 1) {
    size_t to = c < nc ? cuts[c] : n;
    if (c < nc) llb_tcp_feed(P, s + at, (int)(to - at));
    else {
      llb_tcp_feed(P, s + at, (int)(to - at));
      llb_tcp_finish(P);
    }
    at = to;
    LlbTcEvent e;
    while (llb_tcp_next(P, &e)) {
      if ((e.type == LLB_TCE_CONTENT || e.type == LLB_TCE_REASONING) && e.len && ((unsigned char)e.text[0] & 0xC0) == 0x80) *utf8bad += 1;
      if ((e.type == LLB_TCE_CONTENT || e.type == LLB_TCE_REASONING) && e.type == pt) {
        llb_buf_put(&pend, e.text, e.len);
        continue;
      }
      if (pt != LLB_TCE_NONE) {
        if (!first) llb_buf_puts(&out, ", ");
        first = 0;
        llb_buf_puts(&out, "{\"type\": \"");
        llb_buf_puts(&out, ev_name(pt));
        llb_buf_puts(&out, "\", \"text\": ");
        llb_json_quote(&out, pend.p, pend.n);
        llb_buf_puts(&out, "}");
        pend.n = 0;
        pt = LLB_TCE_NONE;
      }
      if (e.type == LLB_TCE_CONTENT || e.type == LLB_TCE_REASONING) {
        pt = e.type;
        llb_buf_put(&pend, e.text, e.len);
        continue;
      }
      if (!first) llb_buf_puts(&out, ", ");
      first = 0;
      llb_buf_puts(&out, "{\"type\": \"");
      llb_buf_puts(&out, ev_name(e.type));
      if (e.type == LLB_TCE_TOOL_CALL) {
        char ix[32];
        snprintf(ix, sizeof ix, "\", \"index\": %d, \"name\": ", e.index);
        llb_buf_puts(&out, ix);
        llb_json_quote(&out, e.name, strlen(e.name));
        llb_buf_puts(&out, ", \"arguments\": ");
        llb_buf_puts(&out, e.args);
        llb_buf_puts(&out, "}");
      } else {
        llb_buf_puts(&out, "\", \"text\": ");
        llb_json_quote(&out, e.text, e.len);
        llb_buf_puts(&out, "}");
      }
    }
  }
  if (pt != LLB_TCE_NONE) {
    if (!first) llb_buf_puts(&out, ", ");
    llb_buf_puts(&out, "{\"type\": \"");
    llb_buf_puts(&out, ev_name(pt));
    llb_buf_puts(&out, "\", \"text\": ");
    llb_json_quote(&out, pend.p, pend.n);
    llb_buf_puts(&out, "}");
  }
  llb_buf_puts(&out, "]");
  free(pend.p);
  llb_tcp_free(P);
  return out.p;
}

int main(int argc, char** argv) {
  if (argc >= 4 && !strcmp(argv[1], "render")) {
    size_t n;
    char* req = slurp(stdin, &n);
    char err[512];
    char* r = llb_tc_render(req, atoi(argv[2]), atoi(argv[3]), err, sizeof err);
    free(req);
    if (!r) { printf("ERROR: %s", err); return 3; }
    fwrite(r, 1, strlen(r), stdout);
    free(r);
    return 0;
  }
  if (argc >= 6 && !strcmp(argv[1], "parse")) {
    char* tools = NULL;
    if (strcmp(argv[2], "-")) {
      FILE* f = fopen(argv[2], "rb");
      size_t tn;
      if (!f) { perror(argv[2]); return 2; }
      tools = slurp(f, &tn);
      fclose(f);
    }
    int reasoning = atoi(argv[3]), nrand = atoi(argv[4]);
    unsigned long long rng = strtoull(argv[5], NULL, 10) * 2862933555777941757ULL + 3037000493ULL;
    size_t n;
    char* s = slurp(stdin, &n);
    size_t* cuts = calloc(n + 1, sizeof *cuts);
    if (!cuts) abort();
    int utf8bad = 0, runs = 0, bad = 0;
    char* ref = run(tools, reasoning, s, n, NULL, 0, &utf8bad);
    runs += 1;
#define CHECK(nc)                                                       \
    do {                                                                \
      char* got = run(tools, reasoning, s, n, cuts, (nc), &utf8bad);    \
      runs += 1;                                                        \
      if (strcmp(got, ref)) {                                           \
        if (!bad) fprintf(stderr, "MISMATCH:\n  ref %s\n  got %s\n", ref, got); \
        bad += 1;                                                       \
      }                                                                 \
      free(got);                                                        \
    } while (0)
    for (size_t i = 0; i + 1 < n; i += 1) cuts[i] = i + 1;  // 1-byte chunks
    CHECK(n ? n - 1 : 0);
    for (size_t k = 1; k < n; k += 1) {  // every 2-chunk split
      cuts[0] = k;
      CHECK(1);
    }
    for (int r = 0; r < nrand; r += 1) {  // random chunkings, sizes 1..4 or 1..32
      size_t nc = 0, at = 0, mx = r % 2 ? 4 : 32;
      for (;;) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        at += 1 + (size_t)((rng >> 33) % mx);
        if (at >= n) break;
        cuts[nc++] = at;
      }
      CHECK(nc);
    }
    printf("%s\n", ref);
    fprintf(stderr, "runs=%d mismatches=%d utf8_splits=%d\n", runs, bad, utf8bad);
    free(ref);
    free(cuts);
    free(s);
    free(tools);
    return bad || utf8bad ? 1 : 0;
  }
  fprintf(stderr, "usage: tc_test render ADD_GEN THINK < req.json | tc_test parse TOOLS|- REASONING NRAND SEED < stream\n");
  return 2;
}
