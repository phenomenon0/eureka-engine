// llb_host.c (ENGINE-ONE copy of engine-phone's (engine-ios's + span_plan / LLB_KREPORT), + decide): the in-process
// API (llb_engine.h) driven from a Mac command line: the same wrapper the iOS app links, built for macOS, so the API is
// checked against the Mac CLI on the same model + ids.
//   llb-host <model_dir> <cache_dir> <n_new> id...
//   llb-host <model_dir> <cache_dir> decide request.json...   (ENGINE-ONE) one reply line per request on stdout
//   llb-host <model_dir> <cache_dir> rows <outdir> item...     (ENGINE-DECIDECHUNK) llb_rows; an item file is
//            two lines: the ids, then the hook positions (LLB_DECIDE_FASTPF picks the tier, as for decide)
#include "llb_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* s = malloc((size_t)n + 1);
  size_t got = fread(s, 1, (size_t)n, f);
  s[got] = 0;
  fclose(f);
  return s;
}

// (ENGINE-ONE) llb_decide over request files; replies on stdout in argument order
static int decide(int nreq, char** paths) {
  char** bodies = calloc((size_t)nreq, sizeof *bodies);
  char** replies = calloc((size_t)nreq, sizeof *replies);
  for (int i = 0; i < nreq; i += 1) {
    if (!(bodies[i] = slurp(paths[i]))) {
      fprintf(stderr, "cannot read %s\n", paths[i]);
      return 2;
    }
  }
  llb_dstats st;
  int got = llb_decide((const char* const*)bodies, nreq, replies, &st);
  if (got < 0) {
    fprintf(stderr, "decide: %s\n", llb_error());
    return 1;
  }
  for (int i = 0; i < nreq; i += 1) {
    printf("%s\n", replies[i]);
    free(replies[i]);
    free(bodies[i]);
  }
  fprintf(stderr, "LLBDECIDE answered=%d/%d questions=%d input_tokens=%d compile_ms=%.0f head_open_ms=%.1f "
    "load_ms=%.0f forward_ms=%.1f head_ms=%.3f steps=%d step_ms_sum=%.0f total_ms=%.0f span_mb=%llu "
    "fastpf=%d chunks=%d chunk_tokens=%d chunk_ms_sum=%.0f\n", got, nreq,
    st.questions, st.input_tokens, st.compile_ms, st.head_open_ms, st.load_ms, st.forward_ms, st.head_ms, st.steps,
    st.step_ms_sum, st.total_ms, (unsigned long long)(st.span_bytes >> 20), st.fastpf, st.chunks, st.chunk_tokens,
    st.chunk_ms_sum);
  return got == nreq ? 0 : 3;
}

// (ENGINE-DECIDECHUNK) llb_rows over item files (line 1 ids, line 2 hook positions)
static int rows(const char* outdir, int n, char** paths) {
  uint32_t** ids = calloc((size_t)n, sizeof *ids);
  uint32_t** pos = calloc((size_t)n, sizeof *pos);
  int* ni = calloc((size_t)n, sizeof *ni);
  int* np = calloc((size_t)n, sizeof *np);
  for (int i = 0; i < n; i += 1) {
    char* s = slurp(paths[i]);
    char* nl = s ? strchr(s, '\n') : NULL;
    if (!nl) {
      fprintf(stderr, "cannot read two lines from %s\n", paths[i]);
      return 2;
    }
    *nl = 0;
    for (int pass = 0; pass < 2; pass += 1) {
      char* t = pass ? nl + 1 : s;
      uint32_t* v = calloc(strlen(t) / 2 + 2, sizeof *v);
      int k = 0;
      for (char* e; ; t = e) {
        unsigned long x = strtoul(t, &e, 10);
        if (e == t) {
          break;
        }
        v[k++] = (uint32_t)x;
      }
      if (pass) {
        pos[i] = v;
        np[i] = k;
      } else {
        ids[i] = v;
        ni[i] = k;
      }
    }
    free(s);
  }
  llb_dstats st;
  int got = llb_rows((const uint32_t* const*)ids, ni, (const uint32_t* const*)pos, np, n, outdir, &st);
  if (got < 0) {
    fprintf(stderr, "rows: %s\n", llb_error());
    return 1;
  }
  fprintf(stderr, "LLBROWS written=%d/%d input_tokens=%d compile_ms=%.0f load_ms=%.0f forward_ms=%.1f steps=%d "
    "step_ms_sum=%.0f total_ms=%.0f fastpf=%d chunks=%d chunk_tokens=%d chunk_ms_sum=%.0f\n", got, n, st.input_tokens,
    st.compile_ms, st.load_ms, st.forward_ms, st.steps, st.step_ms_sum, st.total_ms, st.fastpf, st.chunks,
    st.chunk_tokens, st.chunk_ms_sum);
  return got == n ? 0 : 3;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model_dir> <cache_dir> <n_new> id... | <model_dir> <cache_dir> decide req.json...\n",
      argv[0]);
    return 2;
  }
  if (!strcmp(argv[3], "rows") && argc >= 6) {
    if (llb_load(argv[1], argv[2]) != 0) {
      fprintf(stderr, "load: %s\n", llb_error());
      return 1;
    }
    int rc = rows(argv[4], argc - 5, argv + 5);
    llb_free();
    return rc;
  }
  if (!strcmp(argv[3], "decide")) {
    fprintf(stderr, "device: %s\n", llb_device_info());
    if (llb_load(argv[1], argv[2]) != 0) {
      fprintf(stderr, "load: %s\n", llb_error());
      return 1;
    }
    fprintf(stderr, "span_plan: %llu MB\n", (unsigned long long)(llb_span_plan() >> 20));
    int rc = decide(argc - 4, argv + 4);
    llb_free();
    return rc;
  }
  int n_new = atoi(argv[3]), n_ids = argc - 4;
  uint32_t* ids = calloc((size_t)n_ids, sizeof *ids);
  uint32_t* out = calloc((size_t)n_new, sizeof *out);
  for (int i = 0; i < n_ids; i += 1) {
    ids[i] = (uint32_t)strtoul(argv[4 + i], NULL, 10);
  }
  fprintf(stderr, "device: %s\n", llb_device_info());
  if (llb_load(argv[1], argv[2]) != 0) {
    fprintf(stderr, "load: %s\n", llb_error());
    return 1;
  }
  fprintf(stderr, "span_plan: %llu MB\n", (unsigned long long)(llb_span_plan() >> 20));
  if (getenv("LLB_KREPORT")) {  // (ENGINE-PHONE) the threadgroup table, as the device probe prints it
    fprintf(stderr, "kernels: %s\n", llb_kernel_report());
  }
  llb_stats st;
  int got = llb_generate(ids, n_ids, n_new, out, &st);
  if (got < 0) {
    fprintf(stderr, "generate: %s\n", llb_error());
    return 1;
  }
  printf("ids");
  for (int i = 0; i < got; i += 1) {
    printf(" %u", out[i]);
  }
  printf("\n");
  printf("LLBSTATS got=%d compile_ms=%.0f load_ms=%.0f prefill_ms=%.0f decode_steps=%d "
    "decode_ms_per_tok=%.2f total_ms=%.0f span_mb=%llu gguf_mb=%llu\n", got, st.compile_ms,
    st.load_ms, st.prefill_ms, st.steps, st.steps ? st.step_ms_sum / st.steps : 0.0,
    st.total_ms, (unsigned long long)(st.span_bytes >> 20),
    (unsigned long long)(st.gguf_bytes >> 20));
  // (ENGINE-EARLYSTOP) the engine's own "stop <pos> <id>" line (the run ended at a stop id), else ran to n_new
  const char* lg = llb_log();
  const char* sl = NULL;
  for (const char* l = lg; l && *l;) {
    if (!strncmp(l, "stop ", 5)) {
      sl = l;
    }
    const char* nl = strchr(l, '\n');
    l = nl ? nl + 1 : NULL;
  }
  printf("LLBSTOP %s", sl ? "stopped_at=" : "ran_to_n_new\n");
  if (sl) {
    printf("%.*s\n", (int)strcspn(sl + 5, "\n"), sl + 5);
  }
  int again = llb_generate(ids, n_ids, n_new, out, &st);
  printf("second llb_generate: %d (%s)\n", again, llb_error());
  llb_free();
  return 0;
}
