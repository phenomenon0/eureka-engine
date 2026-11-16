/* backends/windows/compat/test_shim.c: the POSIX shim's self-test (XPLAT-WIN).
 *
 * Runs the paths the Bend runtime preamble and dual.c take that engine_cpu does not:
 * a MAP_NORESERVE reservation at a high hint committed on first touch, adjacent growth,
 * a kernel read (pread) into untouched reserved pages, a PROT_NONE guard, threads with a
 * mutex + condition variable, a non-blocking pipe woken through select(), poll() on a
 * pipe, a loopback socket pair, fopencookie with stdout assigned to it, getline, setenv,
 * clock_gettime, and an 8 MiB thread stack. One "ok"/"FAIL" line each; exit 0 = all ok.
 * Build (from the repo root):
 *   clang -std=c11 -O2 -D_CRT_DECLARE_NONSTDC_NAMES=0 -c backends/windows/compat/llbw_posix.c -o shim.o
 *   clang -std=c11 -O2 -D_CRT_DECLARE_NONSTDC_NAMES=0 -D_POSIX_C_SOURCE=200809L -I backends/windows/compat \
 *     backends/windows/compat/test_shim.c shim.o -lws2_32 -lwinmm -o test_shim.exe
 */
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int fails;
static FILE *real_out;
static void check(int ok, const char *what) {
  fprintf(real_out, "%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) fails++;
}

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int flag, wake_fd[2];
static void *signaller(void *arg) {
  (void)arg;
  struct timespec t = { 0, 20 * 1000000 };
  nanosleep(&t, NULL);
  pthread_mutex_lock(&mu);
  flag = 1;
  pthread_cond_broadcast(&cv);
  pthread_mutex_unlock(&mu);
  char b = 'w';
  return write(wake_fd[1], &b, 1) == 1 ? (void *)1 : NULL;
}
static int deep(int n) { volatile char pad[4096]; pad[0] = (char)n; return n ? deep(n - 1) + pad[0] : 0; }
static void *stack_user(void *arg) { (void)arg; return (void *)(intptr_t)deep(1500); } /* ~6 MiB of frames */

static char cap[256];
static size_t capn;
static ssize_t cw(void *c, const char *p, size_t n) { (void)c; memcpy(cap + capn, p, n); capn += n; return (ssize_t)n; }

int main(void) {
  real_out = stdout;
  /* 1. an 8 GiB MAP_NORESERVE reservation at 2^45, touched sparsely; then the 8 GiB above it (corpus_grow) */
  uint64_t size = 1ull << 33;
  void *hint = (void *)(1ull << 45);
  char *p = mmap(hint, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
  check(p != MAP_FAILED, "mmap 8 GiB MAP_NORESERVE");
  check(p == hint, "mmap honours the 2^45 hint");
  if (p != MAP_FAILED) {
    for (uint64_t o = 0; o < size; o += 1ull << 30) p[o] = 7, p[o + 4095] = 9;
    int ok = 1;
    for (uint64_t o = 0; o < size; o += 1ull << 30) ok &= p[o] == 7 && p[o + 4095] == 9 && p[o + 4096] == 0;
    check(ok, "first touch commits zero pages (8 touches, 1 GiB apart)");
    char *q = mmap(p + size, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    check(q == p + size, "adjacent growth lands at base + size");
    if (q == p + size) { q[size - 1] = 3; check(q[size - 1] == 3, "touch at the far end of the grown map"); munmap(q, size); }
    /* 2. pread into reserved, never-touched pages (a kernel write: no first-touch fault) */
    const char *path = "test_shim.tmp";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    char buf[100000];
    for (int i = 0; i < (int)sizeof buf; i++) buf[i] = (char)(i * 31 + 1);
    check(fd >= 0 && write(fd, buf, sizeof buf) == (ssize_t)sizeof buf, "open + write a temp file (binary)");
    char *dst = p + (3ull << 30) + 12345;
    ssize_t got = pread(fd, dst, sizeof buf, 0);
    check(got == (ssize_t)sizeof buf && !memcmp(dst, buf, sizeof buf), "pread into untouched reserved pages");
    struct stat st;
    check(fstat(fd, &st) == 0 && st.st_size == (long long)sizeof buf, "fstat st_size (64-bit)");
    close(fd);
    unlink(path);
    /* 3. a PROT_NONE guard inside a reservation stays inaccessible (checked through VirtualQuery's view:
     *    mprotect returned 0, and a later mprotect back to RW makes it usable) */
    check(mprotect(p + (5ull << 30), 16384, PROT_NONE) == 0, "mprotect PROT_NONE guard");
    check(mprotect(p + (5ull << 30), 16384, PROT_READ | PROT_WRITE) == 0 && (p[5ull << 30] = 1) == 1, "mprotect back to RW");
    check(munmap(p, size) == 0, "munmap the reservation");
  }
  /* 4. threads, mutex + condition variable, a pipe woken through select() */
  check(pipe(wake_fd) == 0 && fcntl(wake_fd[0], F_SETFL, O_NONBLOCK) == 0, "pipe + O_NONBLOCK");
  char b;
  check(read(wake_fd[0], &b, 1) < 0 && errno == EAGAIN, "empty non-blocking pipe reads EAGAIN");
  pthread_t t;
  check(pthread_create(&t, NULL, signaller, NULL) == 0, "pthread_create");
  pthread_mutex_lock(&mu);
  while (!flag) pthread_cond_wait(&cv, &mu);
  pthread_mutex_unlock(&mu);
  check(flag == 1, "pthread_cond_wait woken by broadcast");
  fd_set rs;
  FD_ZERO(&rs);
  FD_SET(wake_fd[0], &rs);
  struct timeval tv = { 2, 0 };
  check(select(wake_fd[0] + 1, &rs, NULL, NULL, &tv) == 1 && FD_ISSET(wake_fd[0], &rs), "select wakes on the pipe");
  check(read(wake_fd[0], &b, 1) == 1 && b == 'w', "read the wake byte");
  void *ret = NULL;
  check(pthread_join(t, &ret) == 0 && ret == (void *)1, "pthread_join returns the thread's value");
  struct pollfd pf = { wake_fd[0], POLLIN, 0 };
  check(poll(&pf, 1, 0) == 0, "poll on an empty pipe: not ready");
  /* 5. an 8 MiB thread stack */
  check(pthread_create(&t, NULL, stack_user, NULL) == 0 && pthread_join(t, &ret) == 0, "6 MiB of recursion on a thread");
  /* 6. a loopback socket pair */
  int sv[2];
  check(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "socketpair (loopback TCP)");
  check(send(sv[0], "hi", 2, 0) == 2, "send");
  char r2[3] = { 0 };
  check(recv(sv[1], r2, 2, 0) == 2 && !strcmp(r2, "hi"), "recv");
  close(sv[0]), close(sv[1]);
  /* 7. stdout assigned to a fopencookie stream (dual.c's capture) */
  FILE *c = fopencookie(NULL, "w", (cookie_io_functions_t){ .write = cw });
  FILE *saved = stdout;
  stdout = c;
  printf("%d\n", 42);
  puts("x");
  fputs("y", stdout);
  stdout = saved;
  fclose(c);
  check(capn == 6 && !memcmp(cap, "42\nx\ny", 6), "printf/puts/fputs reach the cookie through stdout");
  /* 8. getline, setenv, clock_gettime */
  FILE *f = fopen("test_shim.txt", "wb");
  fputs("line one\r\nline two\n", f);
  fclose(f);
  f = fopen("test_shim.txt", "r");
  char *line = NULL;
  size_t lc = 0;
  check(getline(&line, &lc, f) == 10 && !strcmp(line, "line one\r\n"), "getline keeps \\r\\n (binary mode)");
  fclose(f);
  free(line);
  unlink("test_shim.txt");
  check(setenv("LLBW_TEST", "1", 1) == 0 && getenv("LLBW_TEST") && !strcmp(getenv("LLBW_TEST"), "1"), "setenv");
  struct timespec a, z;
  clock_gettime(CLOCK_MONOTONIC, &a);
  struct timespec ten = { 0, 10 * 1000000 };
  nanosleep(&ten, NULL);
  clock_gettime(CLOCK_MONOTONIC, &z);
  double ms = (z.tv_sec - a.tv_sec) * 1e3 + (z.tv_nsec - a.tv_nsec) / 1e6;
  check(ms >= 9.0 && ms < 50.0, "nanosleep 10 ms measured by CLOCK_MONOTONIC");
  fprintf(real_out, "%s: %d failed\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
