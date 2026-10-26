// llb_guard_test.c (ENGINE-REPEAT): the R1 guards stay guards. llb_load -> llb_generate (p0) answers; a second
// llb_generate with no llb_load between is refused (one run per llb_load); llb_load again -> answers; llb_free ->
// llb_load -> answers. Exit 0 iff exactly that happens.
#include "llb_engine.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s chat_dir cache\n", argv[0]);
    return 2;
  }
  const uint32_t p0[] = { 760, 6511, 314, 9338, 369 };
  uint32_t out[16];
  llb_stats st;
  int ok = 1, r;
  r = llb_load(argv[1], argv[2]) == 0 ? llb_generate(p0, 5, 16, out, &st) : -1;
  printf("GUARD 1 load+generate -> %d %s\n", r, r == 16 ? "answered" : llb_error());
  ok &= r == 16;
  r = llb_generate(p0, 5, 16, out, &st);
  printf("GUARD 2 generate again, no load -> %d %s\n", r, r < 0 ? llb_error() : "ANSWERED (guard did not fire)");
  ok &= r < 0 && strstr(llb_error(), "one run per llb_load") != NULL;
  r = llb_load(argv[1], argv[2]) == 0 ? llb_generate(p0, 5, 16, out, &st) : -1;
  printf("GUARD 3 load+generate -> %d %s\n", r, r == 16 ? "answered" : llb_error());
  ok &= r == 16;
  llb_free();
  r = llb_load(argv[1], argv[2]) == 0 ? llb_generate(p0, 5, 16, out, &st) : -1;
  printf("GUARD 4 free+load+generate -> %d %s | %s\n", r, r == 16 ? "answered" : llb_error(), llb_release_report());
  ok &= r == 16;
  printf("RESULT guard %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
