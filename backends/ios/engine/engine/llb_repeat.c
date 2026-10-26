// llb_repeat.c (ENGINE-REPEAT): N requests in ONE process through the in-process API (llb_engine.h), the R13
// reopen sequence as a loop. Each cycle: chat (p2, 31 ids -> 32 new) -> chat (p0, a distinct prompt, 5 ids -> 16)
// -> decide (Kev-4B, the travel request) -> chat (p2 again). Every request is what Bend's lifecycle does:
// llb_load(dir) -> run -> llb_free. Each answer is judged against the recorded references (ids byte for byte,
// the decide reply byte for byte), and after every request the process's memory and threads are printed:
//   phys_footprint (jetsam's number), resident size, virtual size, thread count, open fds.
//
//   llb-repeat <chat_dir> <decide_dir> <cache_dir> <cycles> <p2.ids> <p2.expect> <p0.ids> <p0.expect>
//              <travel.json> <travel.mac.json>
// Lines: "REPEAT ..." per request, "MEM ..." per sample, "RESULT ..." at the end. Exit 0 iff every request answered
// and matched.
#include "llb_engine.h"
#include <mach/mach.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static char* slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path);
    exit(2);
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

static int ids_of(const char* path, uint32_t* out, int cap) {
  char* s = slurp(path);
  int n = 0;
  for (char* t = s; n < cap;) {
    char* e;
    unsigned long v = strtoul(t, &e, 10);
    if (e == t) {
      break;
    }
    out[n++] = (uint32_t)v;
    t = e;
  }
  free(s);
  return n;
}

static char* trim(char* s) {
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) {
    s[--n] = 0;
  }
  return s;
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static void mem(const char* tag) {
  task_vm_info_data_t v;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n);
  thread_act_array_t th;
  mach_msg_type_number_t nth = 0;
  task_threads(mach_task_self(), &th, &nth);
  for (mach_msg_type_number_t i = 0; i < nth; i += 1) {
    mach_port_deallocate(mach_task_self(), th[i]);
  }
  vm_deallocate(mach_task_self(), (vm_address_t)th, nth * sizeof *th);
  int fds = 0;
  for (int fd = 0; fd < 4096; fd += 1) {
    fds += fcntl(fd, F_GETFD) != -1;
  }
  printf("MEM %s footprint_mb=%llu rss_mb=%llu vsize_gb=%.2f threads=%u fds=%d\n", tag,
    (unsigned long long)(v.phys_footprint >> 20), (unsigned long long)(v.resident_size >> 20),
    (double)v.virtual_size / (1ull << 30), nth, fds);
  fflush(stdout);
}

static const char *chat_dir, *decide_dir, *cache;
static int fails;

// (ENGINE-EARLYSTOP) ask: the n_new requested (a maximum); the reply must be want's nw ids exactly
static void chat(int cyc, int step, const char* name, const uint32_t* ids, int n, const uint32_t* want, int nw,
  int ask) {
  uint32_t out[1024];
  llb_stats st;
  double t0 = now_ms();
  int got = -1;
  if (llb_load(chat_dir, cache) == 0) {
    got = llb_generate(ids, n, ask, out, &st);
  }
  double wall = now_ms() - t0;
  int same = got == nw && memcmp(out, want, sizeof *out * (size_t)nw) == 0;
  int k = 0;
  while (k < got && k < nw && out[k] == want[k]) {
    k += 1;
  }
  if (got < 0) {
    printf("REPEAT cycle %d step %d chat %s FAILED: %s\n", cyc, step, name, llb_error());
  } else {
    printf("REPEAT cycle %d step %d chat %s asked %d answered %d ids, PARITY %s (%d/%d identical) wall_ms=%.0f "
      "compile_ms=%.0f load_ms=%.0f prefill_ms=%.0f decode_ms_per_tok=%.2f engine_total_ms=%.0f span_mb=%llu | %s\n",
      cyc, step, name, ask, got, same ? "OK" : "DIFF", k, nw, wall, st.compile_ms, st.load_ms, st.prefill_ms,
      st.steps ? st.step_ms_sum / st.steps : 0.0, st.total_ms, (unsigned long long)(st.span_bytes >> 20),
      llb_release_report());
  }
  fails += !same;
  llb_free();
}

static void decide(int cyc, int step, const char* body, const char* mac) {
  char* reply = NULL;
  llb_dstats st;
  double t0 = now_ms();
  int got = -1;
  if (llb_load(decide_dir, cache) == 0) {
    got = llb_decide(&body, 1, &reply, &st);
  }
  double wall = now_ms() - t0;
  int same = got == 1 && reply && strcmp(trim(reply), mac) == 0;
  if (got < 0) {
    printf("REPEAT cycle %d step %d decide travel FAILED: %s\n", cyc, step, llb_error());
  } else {
    const char* p = reply ? strstr(reply, "\"Amsterdam\":") : NULL;
    printf("REPEAT cycle %d step %d decide travel answered %d/1, MACREF %s, Amsterdam %.6f wall_ms=%.0f "
      "compile_ms=%.0f head_open_ms=%.1f load_ms=%.0f forward_ms=%.1f head_ms=%.2f chunks=%d engine_total_ms=%.0f "
      "span_mb=%llu | %s\n", cyc, step, got, same ? "BYTE-IDENTICAL" : "DIFFERS", p ? atof(p + 12) : -1.0, wall,
      st.compile_ms, st.head_open_ms, st.load_ms, st.forward_ms, st.head_ms, st.chunks, st.total_ms,
      (unsigned long long)(st.span_bytes >> 20), llb_release_report());
    if (!same && reply) {
      printf("REPEAT cycle %d step %d decide reply %s\n", cyc, step, reply);
    }
  }
  free(reply);
  fails += !same;
  llb_free();
}

int main(int argc, char** argv) {
  if (argc != 11 && argc != 13) {
    fprintf(stderr, "usage: %s chat_dir decide_dir cache cycles p2.ids p2.expect p0.ids p0.expect travel.json "
      "travel.mac.json [ps.ids ps.expect]\n", argv[0]);
    return 2;
  }
  chat_dir = argv[1];
  decide_dir = argv[2];
  cache = argv[3];
  int cycles = atoi(argv[4]);
  uint32_t p2[64], e2[64], p0[64], e0[64];
  int n2 = ids_of(argv[5], p2, 64), m2 = ids_of(argv[6], e2, 64);
  int n0 = ids_of(argv[7], p0, 64), m0 = ids_of(argv[8], e0, 64);
  char* body = slurp(argv[9]);
  char* mac = trim(slurp(argv[10]));
  // (ENGINE-EARLYSTOP) optional step 5: a short reply (ps, ending at its stop id) with 768 requested
  uint32_t ps[64], es[64];
  int ns = argc == 13 ? ids_of(argv[11], ps, 64) : 0, ms = argc == 13 ? ids_of(argv[12], es, 64) : 0;
  printf("REPEAT device %s\n", llb_device_info());
  mem("start");
  for (int c = 1; c <= cycles; c += 1) {
    chat(c, 1, "p2", p2, n2, e2, m2, m2);
    mem("after-step-1");
    chat(c, 2, "p0", p0, n0, e0, m0, m0);
    mem("after-step-2");
    decide(c, 3, body, mac);
    mem("after-step-3");
    chat(c, 4, "p2", p2, n2, e2, m2, m2);
    mem("after-step-4");
    if (ns) {
      chat(c, 5, "ps", ps, ns, es, ms, 768);
      mem("after-step-5");
    }
  }
  if (getenv("LLB_REPEAT_PAUSE")) {  // hold the process for footprint / vmmap
    printf("PAUSE pid %d\n", getpid());
    fflush(stdout);
    sleep((unsigned)atoi(getenv("LLB_REPEAT_PAUSE")));
  }
  int per = ns ? 5 : 4;
  printf("RESULT %d cycles x %d requests in one process: %d/%d answered and matched -> %s\n", cycles, per,
    per * cycles - fails, per * cycles, fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
