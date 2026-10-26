// tools/cam_host.c (PHONECAM): the camera engine library's Mac harness, the same calls the app makes.
//   cam-host <model_dir> <cache_dir> <mmproj> <list> [repeat] [gap_ms] [idsdir]
// list: lines "<id>\t<canvas.ppm (P6)>\t<question>". One resident session (llb_vision_open, llb_open), then every item
// `repeat` times (gap_ms of idle before each); one JSON line per classify on stdout. idsdir: each item's prompt ids
// (pre + pads + post, as the job ran them) written to <idsdir>/<id>.ids (the one-shot CLI reference runs the same ids).
#include "llb_engine.h"
#include "qwen35_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define YES_ID 9175u
#define NO_ID 2665u
#define PAD_ID 248056u
static const char* PRE = "<|im_start|>system\nYou are a careful visual inspector. Look at the photo and answer the "
  "question with a single word: Yes or No.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
static const char* POST_FMT = "<|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

static unsigned char* ppm(const char* path, int* w, int* h) {
  FILE* f = fopen(path, "rb");
  int mx = 0;
  if (!f || fscanf(f, "P6 %d %d %d", w, h, &mx) != 3 || mx != 255 || fgetc(f) == EOF) {
    if (f) fclose(f);
    return NULL;
  }
  size_t n = (size_t)*w * *h * 3;
  unsigned char* p = malloc(n);
  if (fread(p, 1, n, f) != n) { free(p); p = NULL; }
  fclose(f);
  return p;
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: cam-host <model_dir> <cache_dir> <mmproj> <list> [repeat] [gap_ms] [idsdir]\n");
    return 2;
  }
  int repeat = argc > 5 ? atoi(argv[5]) : 1, gap = argc > 6 ? atoi(argv[6]) : 0;
  const char* idsdir = argc > 7 ? argv[7] : NULL;
  Q35Tokenizer tk;
  if (q35_tok_load(&tk, argv[1]) != 0) { fprintf(stderr, "tokenizer\n"); return 1; }
  int pre[64], npre = q35_tok_encode(&tk, PRE, 1, pre, 64);
  uint32_t upre[64];
  printf("{\"pre_ids\":[");
  for (int i = 0; i < npre; i++) { upre[i] = (uint32_t)pre[i]; printf("%s%d", i ? "," : "", pre[i]); }
  printf("],\"n_pre\":%d}\n", npre);
  double t0 = now_ms();
  if (llb_vision_open(argv[3], 2560, 1024) != 0) { fprintf(stderr, "vision_open: %s\n", llb_error()); return 1; }
  double t1 = now_ms();
  llb_stats os = {0};
  if (llb_open(argv[1], argv[2], 4096, 0, &os) != 0) { fprintf(stderr, "open: %s\n%s\n", llb_error(), llb_log()); return 1; }
  printf("{\"vision_open_ms\":%.1f,\"open_ms\":%.1f,\"load_ms\":%.0f,\"compile_ms\":%.0f,\"resident\":\"%s\"}\n",
    t1 - t0, now_ms() - t1, os.load_ms, os.compile_ms, llb_resident_report());
  fflush(stdout);
  FILE* lf = fopen(argv[4], "r");
  if (!lf) { fprintf(stderr, "list\n"); return 1; }
  char line[4096];
  int bad = 0;
  while (fgets(line, sizeof line, lf)) {
    line[strcspn(line, "\n")] = 0;
    char* id = strtok(line, "\t"); char* path = strtok(NULL, "\t"); char* q = strtok(NULL, "\t");
    if (!id || !path || !q) continue;
    int w = 0, h = 0;
    unsigned char* rgb = ppm(path, &w, &h);
    if (!rgb) { fprintf(stderr, "ppm %s\n", path); return 1; }
    char post_s[2048];
    snprintf(post_s, sizeof post_s, POST_FMT, q);
    int post[512], npost = q35_tok_encode(&tk, post_s, 1, post, 512);
    uint32_t upost[512];
    for (int i = 0; i < npost; i++) upost[i] = (uint32_t)post[i];
    for (int r = 0; r < repeat; r++) {
      if (gap > 0) usleep((useconds_t)gap * 1000);
      llb_vstats st = {0};
      double c0 = now_ms();
      int rc = llb_classify_rgb(rgb, w, h, upre, npre, upost, npost, YES_ID, NO_ID, &st);
      double c1 = now_ms();
      if (rc != 0) { printf("{\"id\":\"%s\",\"error\":\"%s\"}\n", id, llb_error()); fprintf(stderr, "%s\n", llb_log()); bad++; break; }
      float s = st.logit_a - st.logit_b;
      printf("{\"id\":\"%s\",\"rep\":%d,\"w\":%d,\"h\":%d,\"n_img\":%d,\"grid\":[%d,%d],\"n_ids\":%d,\"yes_bits\":\"%08x\","
        "\"no_bits\":\"%08x\",\"yes\":%.6f,\"no\":%.6f,\"score\":%.6f,\"verdict\":\"%s\",\"first_id\":%u,"
        "\"prep_ms\":%.2f,\"tower_ms\":%.2f,\"tower_gpu_ms\":%.2f,\"llm_ms\":%.2f,\"total_ms\":%.2f,\"wall_ms\":%.2f}\n",
        id, r, w, h, st.n_img, st.nx, st.ny, st.n_ids, st.bits_a, st.bits_b, st.logit_a, st.logit_b, s,
        s > 0 ? "yes" : "no", st.first_id, st.prep_ms, st.tower_ms, st.tower_gpu_ms, st.llm_ms, st.total_ms, c1 - c0);
      fflush(stdout);
      if (r == 0 && idsdir) {
        char p[1024];
        snprintf(p, sizeof p, "%s/%s.ids", idsdir, id);
        FILE* o = fopen(p, "w");
        for (int i = 0; i < npre; i++) fprintf(o, "%u ", upre[i]);
        for (int i = 0; i < st.n_img; i++) fprintf(o, "%u ", PAD_ID);
        for (int i = 0; i < npost; i++) fprintf(o, "%u%s", upost[i], i + 1 < npost ? " " : "\n");
        fclose(o);
      }
    }
    free(rgb);
  }
  fclose(lf);
  printf("{\"resident\":\"%s\"}\n", llb_resident_report());
  llb_close();
  llb_vision_close();
  printf("{\"release\":\"%s\"}\n", llb_release_report());
  return bad ? 1 : 0;
}
