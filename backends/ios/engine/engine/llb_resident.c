// llb_resident.c (KEVPHONE): the resident session (llb_open) against the one-shot runs it replaces, in ONE process.
//
//   llb-resident <chat_dir> <decide_dir> <cache_dir> <reps> <travel.json> <travel.mac.json> <prompt.txt>...
//
// Each prompt.txt is a rendered chat (the template's markup in the text, as Bend renders it), tokenized here with the
// model's tokenizer (parse_special). For every prompt:
//   1. ONE-SHOT: llb_load -> llb_generate -> llb_free (Bend's per-request path today): the reference ids + timing.
//   2. RESIDENT: llb_open(chat) once, then the prompt `reps` times: ids judged byte for byte against step 1, load_ms
//      must be 0 and the session's load count must not move (no reload on the warm path).
// Then decide: one-shot travel vs travel.mac.json, then llb_open(decide) and `reps` resident decides, each judged
// byte for byte. Then the swap (chat -> decide -> chat on one session slot: each swap is one load, logged), then
// llb_free (the release report) and the memory after it.
// Lines: "RES ..." per request, "MEM ..." per sample, "RESULT ..." at the end. Exit 0 iff all matched and no warm
// request loaded.
#include "llb_engine.h"
#include "qwen35_tokenizer.h"
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static long footprint(void) {
  task_vm_info_data_t v;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n);
  return (long)(v.phys_footprint >> 20);
}

static void mem(const char* tag) {
  printf("MEM %s footprint_mb=%ld | %s\n", tag, footprint(), llb_resident_report());
  fflush(stdout);
}

static long loads_now(void) {  // the session's load count, from its report
  const char* r = strstr(llb_resident_report(), "loads=");
  return r ? atol(r + 6) : -1;
}

static int fails, warm_loads;
#define NNEW 256

typedef struct {
  const char* name;
  uint32_t    ids[8192];
  int         n;
  uint32_t    ref[NNEW];
  int         nref;
} prompt;

static void oneshot(prompt* p, const char* dir, const char* cache) {
  llb_stats st;
  double t0 = now_ms();
  int got = -1;
  if (llb_load(dir, cache) == 0) {
    got = llb_generate(p->ids, p->n, NNEW, p->ref, &st);
  }
  double wall = now_ms() - t0;
  p->nref = got;
  if (got < 0) {
    printf("RES oneshot %s FAILED: %s\n", p->name, llb_error());
    fails += 1;
  } else {
    printf("RES oneshot %s prompt_ids=%d new=%d wall_ms=%.0f load_ms=%.0f prefill_ms=%.0f decode_ms_per_tok=%.2f "
      "compile_ms=%.0f | %s\n", p->name, p->n, got, wall, st.load_ms, st.prefill_ms,
      st.steps ? st.step_ms_sum / st.steps : 0.0, st.compile_ms, llb_release_report());
    if (getenv("RES_SHOW")) {  // the reply as text (the chat model's tokenizer)
      Q35Tokenizer t;
      if (q35_tok_load(&t, dir) == 0) {
        int iv[NNEW];
        for (int k = 0; k < got; k += 1) {
          iv[k] = (int)p->ref[k];
        }
        char text[8192];
        int m = q35_tok_decode(&t, iv, got, text, (int)sizeof text - 1);
        text[m > 0 ? m : 0] = 0;
        printf("RES text %s %s\n", p->name, text);
        q35_tok_free(&t);
      }
    }
  }
  fflush(stdout);
  llb_free();
}

static void resident(prompt* p, const char* dir, const char* cache, int rep) {
  uint32_t out[NNEW];
  llb_stats st;
  long l0 = loads_now();
  double t0 = now_ms();
  int got = -1;
  if (llb_load(dir, cache) == 0) {  // what Bend calls before every run: must not disturb the session
    got = llb_generate(p->ids, p->n, NNEW, out, &st);
  }
  double wall = now_ms() - t0;
  long l1 = loads_now();
  int same = got == p->nref && memcmp(out, p->ref, sizeof *out * (size_t)(got > 0 ? got : 0)) == 0;
  int k = 0;
  while (k < got && k < p->nref && out[k] == p->ref[k]) {
    k += 1;
  }
  int loaded = l1 != l0 || st.load_ms > 0;
  if (got < 0) {
    printf("RES resident %s rep %d FAILED: %s\n", p->name, rep, llb_error());
  } else {
    printf("RES resident %s rep %d prompt_ids=%d new=%d PARITY %s (%d/%d vs one-shot) reload=%s wall_ms=%.0f load_ms=%.0f "
      "prefill_ms=%.0f decode_ms_per_tok=%.2f | %s\n", p->name, rep, p->n, got, same ? "OK" : "DIFF", k, p->nref,
      loaded ? "YES" : "no", wall, st.load_ms, st.prefill_ms, st.steps ? st.step_ms_sum / st.steps : 0.0,
      llb_resident_report());
  }
  fflush(stdout);
  fails += !same;
  warm_loads += loaded;
}

static void decide1(const char* tag, const char* dir, const char* cache, const char* body, const char* mac, int free_after) {
  char* reply = NULL;
  llb_dstats st;
  long l0 = loads_now();
  double t0 = now_ms();
  int got = -1;
  if (llb_load(dir, cache) == 0) {
    got = llb_decide(&body, 1, &reply, &st);
  }
  double wall = now_ms() - t0;
  long l1 = loads_now();
  int same = got == 1 && reply && strcmp(trim(reply), mac) == 0;
  const char* p = reply ? strstr(reply, "\"Amsterdam\":") : NULL;
  if (got < 0) {
    printf("RES %s decide FAILED: %s\n", tag, llb_error());
  } else {
    printf("RES %s decide travel MACREF %s Amsterdam %s reload=%s wall_ms=%.0f load_ms=%.0f forward_ms=%.1f "
      "head_ms=%.2f chunks=%d | %s\n", tag, same ? "BYTE-IDENTICAL" : "DIFFERS", p ? strtok(strdup(p + 12), ",}") : "-",
      free_after ? "n/a" : (l1 != l0 || st.load_ms > 0) ? "YES" : "no", wall, st.load_ms, st.forward_ms, st.head_ms,
      st.chunks, free_after ? llb_release_report() : llb_resident_report());
    if (!same && reply) {
      printf("RES %s decide reply %s\n", tag, reply);
    }
  }
  fflush(stdout);
  fails += !same;
  if (!free_after) {
    warm_loads += l1 != l0 || st.load_ms > 0;
  }
  free(reply);
  if (free_after) {
    llb_free();
  }
}

static void open1(const char* tag, const char* dir, const char* cache, int ctx) {
  llb_stats st;
  double t0 = now_ms();
  int rc = llb_open(dir, cache, ctx, getenv("RES_NOWARMUP") ? LLB_OPEN_NOWARMUP : 0, &st);
  printf("RES open %s rc=%d wall_ms=%.0f load_ms=%.0f compile_ms=%.0f %s| %s\n", tag, rc, now_ms() - t0, st.load_ms,
    st.compile_ms, rc ? llb_error() : "", llb_resident_report());
  fflush(stdout);
  fails += rc != 0;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s chat_dir decide_dir cache reps travel.json travel.mac.json prompt.txt...\n", argv[0]);
    return 2;
  }
  const char *chat = argv[1], *dec = argv[2], *cache = argv[3];
  int reps = atoi(argv[4]);
  char* body = slurp(argv[5]);
  char* mac = trim(slurp(argv[6]));
  int np = argc - 7;
  int ctx = getenv("RES_CTX") ? atoi(getenv("RES_CTX")) : 4096;
  prompt* ps = calloc((size_t)np, sizeof *ps);
  Q35Tokenizer tok;
  if (q35_tok_load(&tok, chat) != 0) {
    fprintf(stderr, "no tokenizer in %s\n", chat);
    return 2;
  }
  for (int i = 0; i < np; i += 1) {
    char* text = slurp(argv[7 + i]);
    const char* b = strrchr(argv[7 + i], '/');
    ps[i].name = b ? b + 1 : argv[7 + i];
    int ids[8192];
    int n = q35_tok_encode(&tok, text, 1, ids, 8192);
    for (int k = 0; k < n; k += 1) {
      ps[i].ids[k] = (uint32_t)ids[k];
    }
    ps[i].n = n;
    free(text);
  }
  printf("RES device %s\n", llb_device_info());
  mem("start");
  // 1. the one-shot references (Bend's path before this lane)
  for (int i = 0; i < np; i += 1) {
    oneshot(&ps[i], chat, cache);
  }
  mem("after-oneshots");
  // 2. resident chat
  open1("chat", chat, cache, ctx);
  mem("chat-resident");
  for (int r = 1; r <= reps; r += 1) {
    for (int i = 0; i < np; i += 1) {
      resident(&ps[i], chat, cache, r);
    }
  }
  mem("after-resident-chat");
  // 3. decide: one-shot, then resident (the llb_load of another model closes the chat session first)
  decide1("oneshot", dec, cache, body, mac, 1);
  mem("after-oneshot-decide");
  open1("decide", dec, cache, ctx);
  for (int r = 1; r <= reps; r += 1) {
    decide1("resident", dec, cache, body, mac, 0);
  }
  mem("after-resident-decide");
  // 4. the swap: chat -> decide -> chat on the one slot, each swap one load (Bend's exclusive lifecycle)
  open1("swap-chat", chat, cache, ctx);
  resident(&ps[0], chat, cache, 0);
  open1("swap-decide", dec, cache, ctx);
  decide1("resident", dec, cache, body, mac, 0);
  open1("swap-chat", chat, cache, ctx);
  resident(&ps[0], chat, cache, 0);
  mem("after-swaps");
  // 5. release
  llb_free();
  printf("RES release %s\n", llb_release_report());
  mem("after-release");
  printf("RESULT resident: %d mismatches, %d warm requests that loaded -> %s\n", fails, warm_loads,
    fails || warm_loads ? "FAIL" : "PASS");
  q35_tok_free(&tok);
  return fails || warm_loads ? 1 : 0;
}
