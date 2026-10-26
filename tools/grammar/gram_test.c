#define _POSIX_C_SOURCE 199309L
// tools/grammar/gram_test.c — the G1 grammar module's test driver (tools/grammar/README.md).
//   gram_test gbnf  G.gbnf                      compile; prints "ok" or the error
//   gram_test schema S.json                     JSON schema -> GBNF on stdout
//   gram_test match G.gbnf < recs               [u32 len][bytes]... -> "1"/"0" per record
//                                               (1 = the whole record is a sentence)
//   gram_test vocab TOKBIN                      every id's piece as [u32 len][bytes]
//   gram_test maskcheck G.gbnf TOKBIN < recs    per prefix record: trie mask vs brute force
//                                               (per-token walk on a second matcher; every
//                                               997th id on a fresh one), and mask timings
//   gram_test gen G.gbnf TOKBIN N SEED          random-logit constrained decode, N runs:
//                                               prints each text as [u32 len][bytes];
//                                               len's high bit = hit the step cap unfinished
#include "../../host/llb_grammar_impl.h"
#include <time.h>

static char* slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* s = malloc((size_t)n + 1);
  if (fread(s, 1, (size_t)n, f) != (size_t)n) exit(2);
  s[n] = 0;
  fclose(f);
  return s;
}

static char* rec(FILE* f, uint32_t* len) {
  if (fread(len, 4, 1, f) != 1) return NULL;
  char* b = malloc(*len + 1);
  if (fread(b, 1, *len, f) != *len) { free(b); return NULL; }
  b[*len] = 0;
  return b;
}

static LlbGram* load_gram(const char* path) {
  char err[256];
  char* src = slurp(path);
  LlbGram* g = llb_gram_from_gbnf(src, NULL, err, sizeof err);
  free(src);
  if (!g) { fprintf(stderr, "%s\n", err); exit(1); }
  return g;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t rng_s;
static double urand(void) {
  rng_s += 0x9E3779B97F4A7C15ull;
  uint64_t z = rng_s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return (double)((z ^ (z >> 31)) >> 11) * 0x1.0p-53;
}

int main(int argc, char** argv) {
  char err[256];
  if (argc < 3) { fprintf(stderr, "usage: see the header\n"); return 2; }
  const char* mode = argv[1];
  if (!strcmp(mode, "gbnf")) {
    char* src = slurp(argv[2]);
    LlbGram* g = llb_gram_from_gbnf(src, NULL, err, sizeof err);
    printf("%s\n", g ? "ok" : err);
    llb_gram_free(g);
    free(src);
    return g ? 0 : 1;
  }
  if (!strcmp(mode, "schema")) {
    char* src = slurp(argv[2]);
    char* gb = llb_gram_schema_to_gbnf(src, err, sizeof err);
    free(src);
    if (!gb) { fprintf(stderr, "%s\n", err); return 1; }
    fputs(gb, stdout);
    free(gb);
    return 0;
  }
  if (!strcmp(mode, "match")) {
    LlbGram* g = load_gram(argv[2]);
    LlbGVocab* v = llb_gv_new(0, NULL, NULL);
    LlbGMatch* m = llb_gm_new(g, v);
    uint32_t len;
    char* b;
    while ((b = rec(stdin, &len))) {
      llb_gm_reset(m);
      int ok = llb_gm_accept_bytes(m, b, (int)len) == 0 && llb_gm_can_end(m);
      printf("%d\n", ok);
      free(b);
    }
    llb_gm_free(m); llb_gv_free(v); llb_gram_free(g);
    return 0;
  }
  if (!strcmp(mode, "vocab")) {
    LlbGVocab* v = llb_gv_load(argv[2], err, sizeof err);
    if (!v) { fprintf(stderr, "%s\n", err); return 1; }
    for (int i = 0; i < llb_gv_n_vocab(v); i += 1) {
      const unsigned char* p;
      uint32_t n = (uint32_t)llb_gv_piece(v, i, &p);
      fwrite(&n, 4, 1, stdout);
      fwrite(p, 1, n, stdout);
    }
    llb_gv_free(v);
    return 0;
  }
  if (!strcmp(mode, "maskcheck") || !strcmp(mode, "gen")) {
    LlbGram* g = load_gram(argv[2]);
    double t0 = now();
    LlbGVocab* v = llb_gv_load(argv[3], err, sizeof err);
    if (!v) { fprintf(stderr, "%s\n", err); return 1; }
    fprintf(stderr, "vocab+trie load %.1f ms (%d ids, %d trie nodes)\n", 1e3 * (now() - t0), v->n, v->tn);
    int NL = 248320;   // the 9B head width: ids past the vocabulary must stay masked
    uint32_t* bits = malloc(sizeof(uint32_t) * (size_t)((NL + 31) / 32));
    LlbGMatch* m = llb_gm_new(g, v);
    if (!strcmp(mode, "maskcheck")) {
      LlbGMatch* bf = llb_gm_new(g, v);
      uint32_t len;
      char* b;
      int bad = 0, nrec = 0;
      while ((b = rec(stdin, &len))) {
        llb_gm_reset(m);
        if (llb_gm_accept_bytes(m, b, (int)len)) { printf("prefix %d rejected\n", nrec++); free(b); continue; }
        double a = now();
        int cnt = llb_gm_mask(m, bits, NL);
        double t_cold = now() - a;
        a = now();
        llb_gm_mask(m, bits, NL);
        double t_warm = now() - a;
        int diff = 0, bcnt = 0;
        llb_gm_reset(bf);
        llb_gm_accept_bytes(bf, b, (int)len);
        for (int t = 0; t < NL; t += 1) {
          // brute force: a second matcher walks each token's bytes on its own (no trie); every
          // 997th id also on a brand-new matcher (no shared memo at all)
          int want = t < v->n && llb_gm_allows(bf, t);
          if (t < v->n && t % 997 == 0) {
            LlbGMatch* fr = llb_gm_new(g, v);
            llb_gm_accept_bytes(fr, b, (int)len);
            int w2 = llb_gm_accept(fr, t) == 0;
            llb_gm_free(fr);
            if (w2 != want && diff++ < 3) printf("  FRESH-DIFF prefix %d tok %d\n", nrec, t);
          }
          int got = (int)(bits[t >> 5] >> (t & 31) & 1);
          bcnt += want;
          if (want != got && diff++ < 3) printf("  DIFF prefix %d tok %d want %d got %d\n", nrec, t, want, got);
        }
        bad += diff != 0;
        printf("prefix %d allowed %d brute %d diffs %d cold %.2f ms warm %.3f ms\n", nrec, cnt, bcnt, diff, 1e3 * t_cold, 1e3 * t_warm);
        nrec += 1;
        free(b);
        if (getenv("MASKCHECK_FAST")) break;
      }
      printf("maskcheck: %d/%d prefixes identical\n", nrec - bad, nrec);
      llb_gm_free(bf);
    } else {
      int N = atoi(argv[4]);
      rng_s = (uint64_t)atoll(argv[5]);
      float* lg = malloc(sizeof(float) * (size_t)NL);
      double tm = 0;
      long nm = 0;
      for (int r = 0; r < N; r += 1) {
        llb_gm_reset(m);
        LlbBuf out = {0};
        llb_buf_put(&out, "", 0);
        int steps = 0;
        while (!llb_gm_done(m) && steps < 1500) {
          for (int i = 0; i < NL; i += 1) lg[i] = (float)(4.0 * urand());
          double a = now();
          int cnt = llb_gm_mask(m, bits, NL);
          tm += now() - a;
          nm += 1;
          if (!cnt) { fprintf(stderr, "dead end at step %d\n", steps); break; }
          llb_gm_apply(bits, lg, NL);
          // a structural bias so random runs terminate: favour short pieces and ending
          int best = -1;
          float bv = -INFINITY;
          for (int i = 0; i < NL; i += 1) {
            if (lg[i] == -INFINITY) continue;
            const unsigned char* p;
            int l = i < v->n ? llb_gv_piece(v, i, &p) : 0;
            float x = lg[i] - 0.5f * (float)l + (v->eog[i] ? 2.0f + 0.05f * (float)steps : 0.0f);
            if (x > bv) { bv = x; best = i; }
          }
          if (llb_gm_accept(m, best)) { fprintf(stderr, "accept refused %d\n", best); return 1; }
          if (!v->eog[best]) {
            const unsigned char* p;
            int l = llb_gv_piece(v, best, &p);
            llb_buf_put(&out, (const char*)p, (size_t)l);
          }
          steps += 1;
        }
        uint32_t n = (uint32_t)out.n | (llb_gm_done(m) ? 0u : 0x80000000u);   // high bit: unfinished
        fwrite(&n, 4, 1, stdout);
        fwrite(out.p, 1, out.n, stdout);
        free(out.p);
      }
      fprintf(stderr, "gen: %ld masks, mean %.3f ms\n", nm, 1e3 * tm / (double)(nm ? nm : 1));
      free(lg);
    }
    llb_gm_free(m); llb_gv_free(v); llb_gram_free(g); free(bits);
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 2;
}
