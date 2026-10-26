// sys1.c: SYS1-DUAL effect glue, spliced into the engine after the runtime and
// host/llb_host.c (like host/llb_gguf.c). The request logic is dual.c, a
// separate object (tools/sys1dual/build.sh). Here: the per-request switch of
// the CLEF-3 hidden hook (llb_hidden_* in host/llb_host.c; unchanged): on for
// decide (prompt rows 0..n-1 to a fresh file), off for generate.
//
// S5/S6 twoheads W5 P1: dual.c emits `--pos K` on EVERY request (K = session
// start pos: 0 fresh, retained token count for a continuation). P1 SOURCE PASS
// (S6): the word now PASSES THROUGH to the .bend (`cfg` parses it into `sp`,
// threaded to the prompt call); the latch below is kept as the compiled-side
// belt (llb_sys1_pos + llb_sys1_pos_first stay for a P2-style wrap patch; a
// fresh re-emit wraps nothing, so the belt is dormant).
// The pos-capture entry blocks of the srv.one-path call chain (FID_FWD_PASS ..
// FID_PROMPT, patched in llb-engine.c) call llb_sys1_pos_first(rX, <line>): the
// FIRST entry-block call after request start captures K; later calls (prompt
// self-recursion carrying pos+1, downstream chain functions) see the flag
// consumed and keep the running pos (cur). K=0 fresh => old behavior exactly.

int  dual_next(const char* dir, unsigned ctx, char** cmd, const char** hidden, unsigned* n);
void dual_done(void);

#ifdef CID_SYS1_NEXT
int llb_sampler_configure(const char* spec); /* host/llb_host.c (W1-SAMPLER) */
extern int llb_pf_hold; /* host/llb_pf.h (G15): exact path for this request */
/* S5 W5 P1: latched per request by sys1_next_run from the `--pos K` word. */
u32 llb_sys1_pos;
static int llb_sys1_pos_pending;
static int llb_sys1_pos_target = -2; /* -2 = env not read, -1 = first entry */
u32 llb_sys1_pos_first(u32 cur, int blk) {
  if (llb_sys1_pos_target == -2) {
    const char* t = getenv("LLB_POS_TARGET");
    llb_sys1_pos_target = (t && *t) ? (int)strtol(t, NULL, 10) : -1;
    if (getenv("LLB_POS_PROBE"))
      fprintf(stderr, "[llb-pos] probe target=%d K=%u pending=%d\n", llb_sys1_pos_target, llb_sys1_pos, llb_sys1_pos_pending);
  }
  if (llb_sys1_pos_pending && (llb_sys1_pos_target == -1 || blk == llb_sys1_pos_target)) {
    llb_sys1_pos_pending = 0;
    if (getenv("LLB_POS_PROBE")) fprintf(stderr, "[llb-pos] CONSUME blk=%d cur=%u -> %u\n", blk, cur, llb_sys1_pos);
    return llb_sys1_pos;
  }
  if (getenv("LLB_POS_PROBE")) fprintf(stderr, "[llb-pos] pass blk=%d cur=%u\n", blk, cur);
  return cur;
}

Term sys1_next_run(Env e, Term* f, IoWork* w) {
  (void)w;
  u64 len = 0;
  char* dir = io_cstr(e, f[0], &len);
  static bool once;
  if (!once) {
    once = true;
    // the hook's own guard (llb_hidden_init): exact pass only
    // G15: LLB_PREFILL=fast is admitted for generate requests; decide requests hold it (below)
    if (getenv("LLB_SPEC") || getenv("LLB_MTP") || getenv("LLB_DUMP_HIDDEN"))
      err_fail("sys1: unset LLB_SPEC, LLB_MTP, LLB_DUMP_HIDDEN");
    llb_hidden_ready = true;
    memset(llb_hidden_positions, 1, sizeof llb_hidden_positions);
  }
  char* cmd = NULL;
  const char* hidden = NULL;
  unsigned n = 0;
  int got = dual_next(dir, (unsigned)(u32)f[1], &cmd, &hidden, &n);
  free(dir);
  llb_hidden_path = hidden;
  llb_pf_hold = hidden != NULL;  /* G15: hidden rows at every position need the exact pass */
  llb_hidden_tokens = n;
  llb_hidden_written = false;
  /* S6 W5 P1: latch --pos K (belt) and PASS the pair through to cfg. */
  char* cmd_s = NULL;
  llb_sys1_pos = 0;
  llb_sys1_pos_pending = 0;
  if (got && cmd) {
    cmd_s = malloc(strlen(cmd) + 1);
    if (!cmd_s) err_fail("sys1: pos pass alloc");
    const char* p = cmd;
    char* q = cmd_s;
    int sampled = 0;
    while (*p) {
      const char* s = p;
      while (*p && *p != ' ') p++;
      size_t l = (size_t)(p - s);
      if (l == 8 && !memcmp(s, "--sample", 8)) {
        /* W1-SAMPLER: consume `--sample SPEC` (host/llb_sampler.h key=val list) for this request */
        while (*p == ' ') p++;
        const char* v = p;
        while (*p && *p != ' ') p++;
        char spec[1024];
        size_t vl = (size_t)(p - v) < sizeof spec - 1 ? (size_t)(p - v) : sizeof spec - 1;
        memcpy(spec, v, vl);
        spec[vl] = 0;
        if (llb_sampler_configure(spec) != 0) err_fail("sys1: bad --sample spec");
        sampled = 1;
        while (*p == ' ') p++;
        continue;
      }
      if (l == 5 && !memcmp(s, "--pos", 5)) {
        /* lookahead only: latch K, do NOT consume; the pair is copied below */
        const char* v = p;
        while (*v == ' ') v++;
        const char* v2 = v;
        while (*v2 && *v2 != ' ') v2++;
        if (v2 > v) {
          llb_sys1_pos = (u32)strtoul(v, NULL, 10);
          llb_sys1_pos_pending = 1;
        }
      }
      if (q != cmd_s) *q++ = ' ';
      memcpy(q, s, l);
      q += l;
      while (*p == ' ') p++;
    }
    *q = 0;
    if (!sampled) llb_sampler_configure(getenv("LLB_SAMPLE")); /* no --sample: back to the default */
  }
  char* words = cmd_s ? cmd_s : cmd;
  // the words as a List<String> (io_args_run's build), last first
  Term xs = term_pak(CID_NIL, 0);
  if (got && words)
  for (char* end = words + strlen(words); end > words;) {
    char* sp = end;
    while (sp > words && sp[-1] != ' ') sp--;
    xs = io_node(e, CID_CON, io_str(e, sp, (u64)(end - sp)), xs);
    end = sp > words ? sp - 1 : words;
  }
  free(cmd_s);
  free(cmd);
  return xs;
}

static void __attribute__((constructor)) sys1_next_use(void) {
  io_eff(CID_SYS1_NEXT, sys1_next_run, 0);
}
#endif

#ifdef CID_SYS1_DONE
Term sys1_done_run(Env e, Term* f, IoWork* w) {
  (void)e; (void)f; (void)w;
  llb_hidden_path = NULL;
  dual_done();
  return term_pak(CID_UNIT, 0);
}

static void __attribute__((constructor)) sys1_done_use(void) {
  io_eff(CID_SYS1_DONE, sys1_done_run, 0);
}
#endif
