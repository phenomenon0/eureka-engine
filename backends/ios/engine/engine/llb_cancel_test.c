// Native integration test: ./llb_cancel_test MODEL_DIR CACHE_DIR
// A cancellation during decode must end the job and leave the resident engine reusable.
#include <llb_engine.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

static long delay_ns = 10000000;
static void *cancel_later(void *unused) {
  (void)unused;
  struct timespec delay = { .tv_sec = 0, .tv_nsec = delay_ns };
  nanosleep(&delay, NULL);
  llb_cancel();
  return NULL;
}
int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s model-dir cache-dir\n", argv[0]); return 2; }
  setenv("LLB_EARLYSTOP", "0", 1);
  llb_stats st = {0};
  if (llb_open(argv[1], argv[2], 1024, 0, &st) < 0) { fprintf(stderr, "open: %s\n", llb_error()); return 1; }
  uint32_t ids[] = { 248046, 1, 2, 3, 4 }, out[256], long_ids[600];
  for (int i = 0; i < 600; i++) long_ids[i] = 1;
  llb_cancel_reset();
  pthread_t t;
  pthread_create(&t, NULL, cancel_later, NULL);
  int prefill = llb_generate(long_ids, 600, 1, out, &st);
  pthread_join(t, NULL);
  int prefill_observed = strstr(llb_log(), "cancelled") != NULL;
  llb_cancel_reset();
  delay_ns = 100000000;
  pthread_create(&t, NULL, cancel_later, NULL);
  int prefix_ids = 0;
  int first = llb_generate_px(ids, 5, 2, 256, out, &st, &prefix_ids);
  pthread_join(t, NULL);
  int observed = strstr(llb_log(), "cancelled") != NULL;
  llb_cancel_reset();
  int second = llb_generate_px(ids, 5, 2, 4, out, &st, &prefix_ids);
  int ok = prefill == 0 && prefill_observed && observed && first >= 0 && first < 256 && second == 4;
  fprintf(stderr, "cancel test prefill=%d marker=%d first=%d second=%d observed=%d: %s\n", prefill, prefill_observed, first, second, observed, ok ? "PASS" : llb_error());
  llb_free();
  return ok ? 0 : 1;
}
