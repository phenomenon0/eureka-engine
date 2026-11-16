/* llbw_posix.c: the Windows POSIX shim's implementation (XPLAT-WIN).
 *
 * Compiled on Windows only, and WITHOUT -I backends/windows/compat: it needs the
 * real UCRT stdio (the shim's stdio.h routes stdout through here) and winsock2.h,
 * whose names the shim's headers keep away from the emitted C. Declarations, and
 * which parts are real and which are stubs: llbw_posix.h and README.md.
 */
#ifndef _WIN32
#error "llbw_posix.c is the Windows POSIX shim"
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <tlhelp32.h>
#include <mmsystem.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <limits.h>
#include <process.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define LLBW_IMPL 1
#include "llbw_posix.h"

/* the shim's numbers that winsock2.h also defines, under LLBW_ names (llbw_posix.h) */
#define LLBW_POLLIN 0x0001
#define LLBW_POLLPRI 0x0002
#define LLBW_POLLOUT 0x0004
#define LLBW_POLLERR 0x0008
#define LLBW_POLLHUP 0x0010
#define LLBW_POLLNVAL 0x0020
#define LLBW_SOCK_NONBLOCK 0x40000000
#define LLBW_SOCK_CLOEXEC 0x20000000
#define LLBW_MSG_DONTWAIT 0x40000000
#define LLBW_O_NONBLOCK 0x10000000
#define LLBW_FDSET_WORDS ((LLBW_SOCK_BASE + LLBW_SOCK_MAX) / 64)
typedef struct { unsigned long long fds_bits[LLBW_FDSET_WORDS]; } llbw_fd_set;
struct llbw_pollfd { int fd; short events; short revents; };
struct llbw_timeval { long tv_sec; long tv_usec; };
struct llbw_iovec { void *iov_base; size_t iov_len; };

static int errno_from_win(DWORD e) {
  switch (e) {
  case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: case ERROR_INVALID_DRIVE: return ENOENT;
  case ERROR_ACCESS_DENIED: case ERROR_SHARING_VIOLATION: case ERROR_LOCK_VIOLATION: return EACCES;
  case ERROR_INVALID_HANDLE: return EBADF;
  case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: case ERROR_COMMITMENT_LIMIT: return ENOMEM;
  case ERROR_FILE_EXISTS: case ERROR_ALREADY_EXISTS: return EEXIST;
  case ERROR_BROKEN_PIPE: case ERROR_NO_DATA: return EPIPE;
  case ERROR_HANDLE_EOF: return 0;
  case ERROR_DISK_FULL: return ENOSPC;
  case ERROR_NOACCESS: return EFAULT;
  case ERROR_INVALID_ADDRESS: case ERROR_INVALID_PARAMETER: return EINVAL;
  default: return EIO;
  }
}
static int errno_from_wsa(int e) {
  switch (e) {
  case WSAEWOULDBLOCK: return EAGAIN;
  case WSAEINPROGRESS: return EINPROGRESS;
  case WSAEALREADY: return EALREADY;
  case WSAENOTSOCK: return ENOTSOCK;
  case WSAEADDRINUSE: return EADDRINUSE;
  case WSAEADDRNOTAVAIL: return EADDRNOTAVAIL;
  case WSAECONNREFUSED: return ECONNREFUSED;
  case WSAECONNRESET: return ECONNRESET;
  case WSAECONNABORTED: return ECONNABORTED;
  case WSAENOTCONN: return ENOTCONN;
  case WSAETIMEDOUT: return ETIMEDOUT;
  case WSAEINTR: return EINTR;
  case WSAEINVAL: return EINVAL;
  case WSAEMFILE: return EMFILE;
  case WSAENOBUFS: return ENOBUFS;
  case WSAEAFNOSUPPORT: return EAFNOSUPPORT;
  case WSAESHUTDOWN: return EPIPE;
  case WSAEACCES: return EACCES;
  case WSAEFAULT: return EFAULT;
  case WSAEMSGSIZE: return EMSGSIZE;
  case WSAENETUNREACH: return ENETUNREACH;
  case WSAEHOSTUNREACH: return EHOSTUNREACH;
  default: return EIO;
  }
}
static int fail_win(void) { errno = errno_from_win(GetLastError()); return -1; }
static int fail_wsa(void) { errno = errno_from_wsa(WSAGetLastError()); return -1; }

/* ======================================================================== */
/* memory maps: reservations committed on first touch                        */
/* ======================================================================== */

/* One entry per live mmap. lazy = an anonymous map whose reserved pages the
 * vectored handler commits when they are first touched (MAP_NORESERVE). */
typedef struct { uintptr_t base, len; int lazy, view; } MapEnt;
#define MAPS_MAX 8192
static MapEnt maps[MAPS_MAX];
static int maps_n;
static SRWLOCK maps_lock = SRWLOCK_INIT;
#define COMMIT_CHUNK ((uintptr_t)1 << 20)

static int map_find(uintptr_t a) { /* caller holds maps_lock */
  for (int i = 0; i < maps_n; i++)
    if (a >= maps[i].base && a - maps[i].base < maps[i].len) return i;
  return -1;
}
static int map_add(uintptr_t base, uintptr_t len, int lazy, int view) {
  AcquireSRWLockExclusive(&maps_lock);
  int ok = maps_n < MAPS_MAX;
  if (ok) maps[maps_n++] = (MapEnt){ base, len, lazy, view };
  ReleaseSRWLockExclusive(&maps_lock);
  return ok;
}

/* commit the reserved pages of [a, end) that lie in lazy maps */
static int commit_range(uintptr_t a, uintptr_t end) {
  int any = 0;
  while (a < end) {
    MEMORY_BASIC_INFORMATION mi;
    if (!VirtualQuery((void *)a, &mi, sizeof mi)) break;
    uintptr_t rend = (uintptr_t)mi.BaseAddress + mi.RegionSize;
    if (rend > end) rend = end;
    if (mi.State == MEM_RESERVE) {
      AcquireSRWLockShared(&maps_lock);
      int i = map_find(a);
      int lazy = i >= 0 && maps[i].lazy;
      uintptr_t mend = i >= 0 ? maps[i].base + maps[i].len : 0;
      ReleaseSRWLockShared(&maps_lock);
      if (lazy) {
        if (rend > mend) rend = mend;
        if (VirtualAlloc((void *)a, rend - a, MEM_COMMIT, PAGE_READWRITE)) any = 1;
      }
    }
    a = rend;
  }
  return any;
}

void llbw_precommit(const void *p, size_t n) {
  if (!n) return;
  uintptr_t a = (uintptr_t)p & ~(uintptr_t)4095, end = (uintptr_t)p + n;
  commit_range(a, end);
}

static int fault_trace;
static LONG CALLBACK map_fault(EXCEPTION_POINTERS *x) {
  EXCEPTION_RECORD *r = x->ExceptionRecord;
  if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
  uintptr_t a = (uintptr_t)r->ExceptionInformation[1];
  MEMORY_BASIC_INFORMATION mi;
  int known = VirtualQuery((void *)a, &mi, sizeof mi) != 0;
  if (fault_trace && (!known || mi.State != MEM_RESERVE)) {
    /* LLBW_FAULT_TRACE=1: every access violation the first-touch handler does not take, one line */
    HMODULE mod = NULL;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)r->ExceptionAddress, &mod);
    fprintf(stderr, "llbw: access violation (%s) at %p, pc %p (module base %p +0x%llx), state %s\n",
            r->ExceptionInformation[0] == 1 ? "write" : r->ExceptionInformation[0] == 8 ? "exec" : "read",
            (void *)a, r->ExceptionAddress, (void *)mod,
            (unsigned long long)((uintptr_t)r->ExceptionAddress - (uintptr_t)mod),
            !known ? "?" : mi.State == MEM_COMMIT ? "committed" : mi.State == MEM_FREE ? "free" : "reserved");
  }
  if (!known || mi.State != MEM_RESERVE) return EXCEPTION_CONTINUE_SEARCH;
  AcquireSRWLockShared(&maps_lock);
  int i = map_find(a);
  int lazy = i >= 0 && maps[i].lazy;
  uintptr_t mend = i >= 0 ? maps[i].base + maps[i].len : 0;
  ReleaseSRWLockShared(&maps_lock);
  if (!lazy) return EXCEPTION_CONTINUE_SEARCH;
  /* commit forward from the faulting page, to the end of its reserved run, its map, or a chunk */
  uintptr_t pg = a & ~(uintptr_t)4095;
  uintptr_t end = pg + COMMIT_CHUNK, rend = (uintptr_t)mi.BaseAddress + mi.RegionSize;
  if (end > rend) end = rend;
  if (end > mend) end = mend;
  if (!VirtualAlloc((void *)pg, end - pg, MEM_COMMIT, PAGE_READWRITE)) return EXCEPTION_CONTINUE_SEARCH;
  return EXCEPTION_CONTINUE_EXECUTION;
}

static DWORD page_prot(int prot) {
  if (prot & PROT_WRITE) return PAGE_READWRITE;
  if (prot & PROT_READ) return PAGE_READONLY;
  return PAGE_NOACCESS;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  if (len == 0) { errno = EINVAL; return MAP_FAILED; }
  if (flags & MAP_ANONYMOUS) {
    void *p = NULL;
    /* a hint is tried first, as Linux does without MAP_FIXED; the 64 KiB granularity applies */
    if (addr && ((uintptr_t)addr & 0xFFFF) == 0) p = VirtualAlloc(addr, len, MEM_RESERVE, PAGE_READWRITE);
    if (!p && (flags & MAP_FIXED)) { errno = ENOMEM; return MAP_FAILED; }
    if (!p) p = VirtualAlloc(NULL, len, MEM_RESERVE, PAGE_READWRITE);
    if (!p) { errno = ENOMEM; return MAP_FAILED; }
    if (prot == PROT_NONE) {
      /* PROT_NONE: reserved and not lazy, so a touch faults until mprotect opens it */
      if (!map_add((uintptr_t)p, len, 0, 0)) { VirtualFree(p, 0, MEM_RELEASE); errno = ENOMEM; return MAP_FAILED; }
    } else if (!map_add((uintptr_t)p, len, 1, 0)) {
      VirtualFree(p, 0, MEM_RELEASE); errno = ENOMEM; return MAP_FAILED;
    }
    if (flags & MAP_POPULATE && prot != PROT_NONE) llbw_precommit(p, len);
    return p;
  }
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return MAP_FAILED; }
  DWORD fprot = (prot & PROT_WRITE) ? ((flags & MAP_PRIVATE) ? PAGE_WRITECOPY : PAGE_READWRITE) : PAGE_READONLY;
  DWORD acc = (prot & PROT_WRITE) ? ((flags & MAP_PRIVATE) ? FILE_MAP_COPY : FILE_MAP_WRITE) : FILE_MAP_READ;
  unsigned long long end = (unsigned long long)off + len;
  HANDLE m = CreateFileMappingW(h, NULL, fprot, (DWORD)(end >> 32), (DWORD)end, NULL);
  if (!m) { fail_win(); return MAP_FAILED; }
  void *p = NULL;
  if (addr) p = MapViewOfFileEx(m, acc, (DWORD)((unsigned long long)off >> 32), (DWORD)off, len, addr);
  if (!p && !(flags & MAP_FIXED)) p = MapViewOfFileEx(m, acc, (DWORD)((unsigned long long)off >> 32), (DWORD)off, len, NULL);
  DWORD e = GetLastError();
  CloseHandle(m); /* the view keeps the mapping alive */
  if (!p) { errno = errno_from_win(e); return MAP_FAILED; }
  if (!map_add((uintptr_t)p, len, 0, 1)) { UnmapViewOfFile(p); errno = ENOMEM; return MAP_FAILED; }
  return p;
}

int munmap(void *addr, size_t len) {
  uintptr_t a = (uintptr_t)addr;
  AcquireSRWLockExclusive(&maps_lock);
  int i = map_find(a);
  if (i < 0) { ReleaseSRWLockExclusive(&maps_lock); errno = EINVAL; return -1; }
  MapEnt m = maps[i];
  int whole = a == m.base && len >= m.len;
  if (whole) maps[i] = maps[--maps_n];
  ReleaseSRWLockExclusive(&maps_lock);
  if (m.view) return whole ? (UnmapViewOfFile(addr) ? 0 : fail_win()) : 0;
  if (whole) return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : fail_win();
  /* part of a reservation: give the pages back (a later touch commits fresh zero pages) */
  uintptr_t lo = a & ~(uintptr_t)4095, hi = (a + len + 4095) & ~(uintptr_t)4095;
  return VirtualFree((void *)lo, hi - lo, MEM_DECOMMIT) ? 0 : fail_win();
}

int mprotect(void *addr, size_t len, int prot) {
  uintptr_t lo = (uintptr_t)addr & ~(uintptr_t)4095, hi = ((uintptr_t)addr + len + 4095) & ~(uintptr_t)4095;
  DWORD old;
  /* committed with the asked protection: a PROT_NONE guard is then a committed no-access page, which the
   * first-touch handler leaves alone */
  if (VirtualAlloc((void *)lo, hi - lo, MEM_COMMIT, page_prot(prot))) return 0;
  return VirtualProtect((void *)lo, hi - lo, page_prot(prot), &old) ? 0 : fail_win();
}

int madvise(void *addr, size_t len, int advice) {
  if (advice != MADV_DONTNEED) return 0;
  uintptr_t a = (uintptr_t)addr;
  AcquireSRWLockShared(&maps_lock);
  int i = map_find(a);
  MapEnt m = i >= 0 ? maps[i] : (MapEnt){ 0 };
  ReleaseSRWLockShared(&maps_lock);
  if (i < 0) { errno = EINVAL; return -1; }
  uintptr_t lo = a & ~(uintptr_t)4095, hi = (a + len) & ~(uintptr_t)4095;
  if (hi <= lo) return 0;
  /* anonymous: decommit (the next touch reads zeros, as Linux's MADV_DONTNEED gives); a file view: drop the
   * pages from the working set (VirtualUnlock on an unlocked range, as tools/llb_platform.h does) */
  if (m.lazy) return VirtualFree((void *)lo, hi - lo, MEM_DECOMMIT) ? 0 : fail_win();
  VirtualUnlock((void *)lo, hi - lo);
  return 0;
}
int msync(void *addr, size_t len, int flags) { (void)flags; return FlushViewOfFile(addr, len) ? 0 : fail_win(); }
int mlock(const void *addr, size_t len) { return VirtualLock((void *)addr, len) ? 0 : fail_win(); }
int munlock(const void *addr, size_t len) { VirtualUnlock((void *)addr, len); return 0; }

/* ======================================================================== */
/* threads                                                                    */
/* ======================================================================== */

struct llbw_thread {
  HANDLE h;
  void *(*fn)(void *);
  void *arg, *ret;
  volatile LONG refs; /* the thread and its joiner (or detacher) */
};
static DWORD tls_self = TLS_OUT_OF_INDEXES;
static void thread_unref(struct llbw_thread *t) {
  if (InterlockedDecrement(&t->refs) == 0) {
    if (t->h) CloseHandle(t->h);
    free(t);
  }
}
static unsigned __stdcall thread_main(void *p) {
  struct llbw_thread *t = p;
  TlsSetValue(tls_self, t);
  t->ret = t->fn(t->arg);
  thread_unref(t);
  return 0;
}
/* XPLAT-WIN: 8 MiB a thread, glibc's default, rather than Windows' 1 MiB (the runtime recurses on the C stack) */
#define THREAD_STACK ((size_t)8 << 20)
int pthread_create(pthread_t *out, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
  struct llbw_thread *t = calloc(1, sizeof *t);
  if (!t) return EAGAIN;
  t->fn = fn, t->arg = arg, t->refs = 2;
  size_t stack = a && a->stack ? a->stack : THREAD_STACK;
  uintptr_t h = _beginthreadex(NULL, (unsigned)stack, thread_main, t, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
  if (!h) { free(t); return EAGAIN; }
  t->h = (HANDLE)h;
  *out = t;
  if (a && a->detach) pthread_detach(t);
  return 0;
}
int pthread_join(pthread_t t, void **ret) {
  if (!t || !t->h) return EINVAL;
  WaitForSingleObject(t->h, INFINITE);
  if (ret) *ret = t->ret;
  thread_unref(t);
  return 0;
}
int pthread_detach(pthread_t t) { if (!t) return EINVAL; thread_unref(t); return 0; }
pthread_t pthread_self(void) {
  struct llbw_thread *t = TlsGetValue(tls_self);
  if (!t) { /* a thread this shim did not start (main): one record, never freed */
    t = calloc(1, sizeof *t);
    if (t) t->refs = 1 << 30, TlsSetValue(tls_self, t);
  }
  return t;
}
int pthread_equal(pthread_t a, pthread_t b) { return a == b; }
void pthread_exit(void *ret) {
  struct llbw_thread *t = TlsGetValue(tls_self);
  if (t) { t->ret = ret; thread_unref(t); }
  _endthreadex(0);
}
int pthread_attr_init(pthread_attr_t *a) { a->detach = 0, a->stack = 0; return 0; }
int pthread_attr_destroy(pthread_attr_t *a) { (void)a; return 0; }
int pthread_attr_setdetachstate(pthread_attr_t *a, int s) { a->detach = s == PTHREAD_CREATE_DETACHED; return 0; }
int pthread_attr_setstacksize(pthread_attr_t *a, size_t n) { a->stack = n; return 0; }

#define SRW(m) ((PSRWLOCK)&(m)->p)
#define CV(c) ((PCONDITION_VARIABLE)&(c)->p)
int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) { (void)a; InitializeSRWLock(SRW(m)); return 0; }
int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
int pthread_mutex_lock(pthread_mutex_t *m) { AcquireSRWLockExclusive(SRW(m)); return 0; }
int pthread_mutex_trylock(pthread_mutex_t *m) { return TryAcquireSRWLockExclusive(SRW(m)) ? 0 : EBUSY; }
int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(SRW(m)); return 0; }
int pthread_mutexattr_init(pthread_mutexattr_t *a) { a->kind = 0; return 0; }
int pthread_mutexattr_destroy(pthread_mutexattr_t *a) { (void)a; return 0; }
/* a recursive mutex is refused rather than silently made non-recursive */
int pthread_mutexattr_settype(pthread_mutexattr_t *a, int k) { if (k == PTHREAD_MUTEX_RECURSIVE) return ENOTSUP; a->kind = k; return 0; }
int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) { (void)a; InitializeConditionVariable(CV(c)); return 0; }
int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
  return SleepConditionVariableSRW(CV(c), SRW(m), INFINITE, 0) ? 0 : EINVAL;
}
int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *abs) {
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  long long ms = ((long long)abs->tv_sec - now.tv_sec) * 1000 + (abs->tv_nsec - now.tv_nsec) / 1000000;
  if (ms < 0) ms = 0;
  if (SleepConditionVariableSRW(CV(c), SRW(m), ms > 0x7FFFFFFE ? 0x7FFFFFFE : (DWORD)ms, 0)) return 0;
  return GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EINVAL;
}
int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(CV(c)); return 0; }
int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(CV(c)); return 0; }
int pthread_rwlock_init(pthread_rwlock_t *l, const pthread_rwlockattr_t *a) { (void)a; InitializeSRWLock(SRW(l)); return 0; }
int pthread_rwlock_destroy(pthread_rwlock_t *l) { (void)l; return 0; }
/* XPLAT-WIN: pthread_rwlock_unlock cannot tell a reader from a writer on an SRWLOCK, so the
 * shim records the writer: a thread-local count of the write locks it holds */
static __declspec(thread) int rw_writes;
int pthread_rwlock_rdlock(pthread_rwlock_t *l) { AcquireSRWLockShared(SRW(l)); return 0; }
int pthread_rwlock_wrlock(pthread_rwlock_t *l) { AcquireSRWLockExclusive(SRW(l)); rw_writes++; return 0; }
int pthread_rwlock_unlock(pthread_rwlock_t *l) {
  if (rw_writes > 0) { rw_writes--; ReleaseSRWLockExclusive(SRW(l)); }
  else ReleaseSRWLockShared(SRW(l));
  return 0;
}
static BOOL CALLBACK once_run(PINIT_ONCE o, PVOID fn, PVOID *ctx) { (void)o, (void)ctx; ((void (*)(void))fn)(); return TRUE; }
int pthread_once(pthread_once_t *o, void (*fn)(void)) {
  return InitOnceExecuteOnce((PINIT_ONCE)&o->p, once_run, (PVOID)fn, NULL) ? 0 : EINVAL;
}
/* keys: TLS slots; destructors are not run (no caller in this repo passes one that matters) */
int pthread_key_create(pthread_key_t *k, void (*dtor)(void *)) {
  (void)dtor;
  DWORD i = TlsAlloc();
  if (i == TLS_OUT_OF_INDEXES) return EAGAIN;
  *k = i;
  return 0;
}
int pthread_key_delete(pthread_key_t k) { TlsFree((DWORD)k); return 0; }
void *pthread_getspecific(pthread_key_t k) { return TlsGetValue((DWORD)k); }
int pthread_setspecific(pthread_key_t k, const void *v) { return TlsSetValue((DWORD)k, (void *)v) ? 0 : EINVAL; }
int sched_yield(void) { SwitchToThread(); return 0; }

/* ======================================================================== */
/* clocks                                                                     */
/* ======================================================================== */

static LARGE_INTEGER qpf;
int clock_gettime(clockid_t id, struct timespec *ts) {
  if (id == CLOCK_REALTIME) {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    unsigned long long t = ((unsigned long long)ft.dwHighDateTime << 32 | ft.dwLowDateTime) - 116444736000000000ull;
    ts->tv_sec = (time_t)(t / 10000000ull);
    ts->tv_nsec = (long)(t % 10000000ull * 100);
    return 0;
  }
  if (id == CLOCK_PROCESS_CPUTIME_ID || id == CLOCK_THREAD_CPUTIME_ID) {
    FILETIME c, e, k, u;
    BOOL ok = id == CLOCK_PROCESS_CPUTIME_ID ? GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)
                                             : GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u);
    if (!ok) return fail_win();
    unsigned long long t = ((unsigned long long)k.dwHighDateTime << 32 | k.dwLowDateTime)
      + ((unsigned long long)u.dwHighDateTime << 32 | u.dwLowDateTime);
    ts->tv_sec = (time_t)(t / 10000000ull);
    ts->tv_nsec = (long)(t % 10000000ull * 100);
    return 0;
  }
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  if (!qpf.QuadPart) QueryPerformanceFrequency(&qpf);
  long long f = qpf.QuadPart;
  ts->tv_sec = (time_t)(c.QuadPart / f);
  ts->tv_nsec = (long)((c.QuadPart % f) * 1000000000ll / f);
  return 0;
}
int nanosleep(const struct timespec *req, struct timespec *rem) {
  long long ns = (long long)req->tv_sec * 1000000000ll + req->tv_nsec;
  if (rem) rem->tv_sec = 0, rem->tv_nsec = 0;
  if (ns <= 0) { SwitchToThread(); return 0; }
  /* a high-resolution waitable timer: Sleep() rounds up to the timer tick */
  HANDLE t = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (!t) { Sleep((DWORD)((ns + 999999) / 1000000)); return 0; }
  LARGE_INTEGER due;
  due.QuadPart = -(ns / 100 > 0 ? ns / 100 : 1);
  SetWaitableTimer(t, &due, 0, NULL, NULL, FALSE);
  WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
  return 0;
}
int usleep(useconds_t us) { struct timespec t = { us / 1000000, (long)(us % 1000000) * 1000 }; return nanosleep(&t, NULL); }
unsigned int sleep(unsigned int s) { Sleep(s * 1000u); return 0; }
int gettimeofday(struct llbw_timeval *tv, void *tz) {
  (void)tz;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  tv->tv_sec = (long)ts.tv_sec, tv->tv_usec = ts.tv_nsec / 1000;
  return 0;
}
struct tm *gmtime_r(const time_t *t, struct tm *out) { return gmtime_s(out, t) ? NULL : out; }
struct tm *localtime_r(const time_t *t, struct tm *out) { return localtime_s(out, t) ? NULL : out; }

/* ======================================================================== */
/* sockets: Winsock SOCKETs behind descriptors LLBW_SOCK_BASE..              */
/* ======================================================================== */

static SOCKET socks[LLBW_SOCK_MAX];
static unsigned char sock_nb[LLBW_SOCK_MAX];
static SRWLOCK socks_lock = SRWLOCK_INIT;
static int is_sock(int fd) { return fd >= LLBW_SOCK_BASE && fd < LLBW_SOCK_BASE + LLBW_SOCK_MAX && socks[fd - LLBW_SOCK_BASE]; }
static SOCKET sock_of(int fd) { return is_sock(fd) ? socks[fd - LLBW_SOCK_BASE] : INVALID_SOCKET; }
uintptr_t llbw_socket_handle(int fd) { return (uintptr_t)sock_of(fd); }
static int sock_new(SOCKET s) {
  AcquireSRWLockExclusive(&socks_lock);
  int fd = -1;
  for (int i = 0; i < LLBW_SOCK_MAX; i++)
    if (!socks[i]) { socks[i] = s, sock_nb[i] = 0, fd = LLBW_SOCK_BASE + i; break; }
  ReleaseSRWLockExclusive(&socks_lock);
  if (fd < 0) { closesocket(s); errno = EMFILE; }
  return fd;
}
static int sock_set_nb(int fd, int on) {
  u_long v = on != 0;
  if (ioctlsocket(sock_of(fd), FIONBIO, &v)) return fail_wsa();
  sock_nb[fd - LLBW_SOCK_BASE] = (unsigned char)(on != 0);
  return 0;
}
#define SOCK_OR_EBADF(fd) do { if (!is_sock(fd)) { errno = ENOTSOCK; return -1; } } while (0)
int llbw_socket(int domain, int type, int proto) {
  int nb = (type & LLBW_SOCK_NONBLOCK) != 0;
  type &= ~(LLBW_SOCK_NONBLOCK | LLBW_SOCK_CLOEXEC);
  SOCKET s = WSASocketW(domain, type, proto, NULL, 0, WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
  if (s == INVALID_SOCKET) return fail_wsa();
  int fd = sock_new(s);
  if (fd >= 0 && nb && sock_set_nb(fd, 1)) { closesocket(s); socks[fd - LLBW_SOCK_BASE] = 0; return -1; }
  return fd;
}
int llbw_bind(int fd, const void *a, int n) { SOCK_OR_EBADF(fd); return bind(sock_of(fd), a, n) ? fail_wsa() : 0; }
int llbw_listen(int fd, int backlog) { SOCK_OR_EBADF(fd); return listen(sock_of(fd), backlog) ? fail_wsa() : 0; }
int llbw_accept4(int fd, void *a, int *n, int flags) {
  SOCK_OR_EBADF(fd);
  SOCKET s = accept(sock_of(fd), a, n);
  if (s == INVALID_SOCKET) return fail_wsa();
  int c = sock_new(s);
  /* a socket accepted from a non-blocking listener is non-blocking on Windows; POSIX gives it the asked mode */
  if (c >= 0 && sock_set_nb(c, (flags & LLBW_SOCK_NONBLOCK) != 0)) { closesocket(s); socks[c - LLBW_SOCK_BASE] = 0; return -1; }
  return c;
}
int llbw_accept(int fd, void *a, int *n) { return llbw_accept4(fd, a, n, 0); }
int llbw_connect(int fd, const void *a, int n) {
  SOCK_OR_EBADF(fd);
  if (!connect(sock_of(fd), a, n)) return 0;
  int e = WSAGetLastError();
  errno = e == WSAEWOULDBLOCK ? EINPROGRESS : errno_from_wsa(e);
  return -1;
}
static int sock_flags(int flags, int fd, int *restore) {
  *restore = 0;
  if ((flags & LLBW_MSG_DONTWAIT) && !sock_nb[fd - LLBW_SOCK_BASE]) { sock_set_nb(fd, 1); *restore = 1; }
  return flags & ~LLBW_MSG_DONTWAIT;
}
ssize_t llbw_recv(int fd, void *buf, size_t n, int flags) {
  SOCK_OR_EBADF(fd);
  int restore, f = sock_flags(flags, fd, &restore);
  llbw_precommit(buf, n);
  int r = recv(sock_of(fd), buf, n > INT_MAX ? INT_MAX : (int)n, f);
  int e = WSAGetLastError();
  if (restore) sock_set_nb(fd, 0);
  if (r < 0) { errno = errno_from_wsa(e); return -1; }
  return r;
}
ssize_t llbw_send(int fd, const void *buf, size_t n, int flags) {
  SOCK_OR_EBADF(fd);
  int restore, f = sock_flags(flags, fd, &restore);
  int r = send(sock_of(fd), buf, n > INT_MAX ? INT_MAX : (int)n, f);
  int e = WSAGetLastError();
  if (restore) sock_set_nb(fd, 0);
  if (r < 0) { errno = errno_from_wsa(e); return -1; }
  return r;
}
ssize_t llbw_recvfrom(int fd, void *buf, size_t n, int flags, void *a, int *an) {
  SOCK_OR_EBADF(fd);
  int r = recvfrom(sock_of(fd), buf, n > INT_MAX ? INT_MAX : (int)n, flags & ~LLBW_MSG_DONTWAIT, a, an);
  return r < 0 ? fail_wsa() : r;
}
ssize_t llbw_sendto(int fd, const void *buf, size_t n, int flags, const void *a, int an) {
  SOCK_OR_EBADF(fd);
  int r = sendto(sock_of(fd), buf, n > INT_MAX ? INT_MAX : (int)n, flags & ~LLBW_MSG_DONTWAIT, a, an);
  return r < 0 ? fail_wsa() : r;
}
int llbw_shutdown(int fd, int how) { SOCK_OR_EBADF(fd); return shutdown(sock_of(fd), how) ? fail_wsa() : 0; }
int llbw_setsockopt(int fd, int level, int name, const void *v, int n) {
  SOCK_OR_EBADF(fd);
  /* SO_RCVTIMEO / SO_SNDTIMEO: POSIX passes a struct timeval, Winsock a DWORD of milliseconds */
  if (level == SOL_SOCKET && (name == SO_RCVTIMEO || name == SO_SNDTIMEO) && n >= (int)sizeof(struct llbw_timeval)) {
    const struct llbw_timeval *tv = v;
    DWORD ms = (DWORD)(tv->tv_sec * 1000 + tv->tv_usec / 1000);
    return setsockopt(sock_of(fd), level, name, (const char *)&ms, sizeof ms) ? fail_wsa() : 0;
  }
  return setsockopt(sock_of(fd), level, name, v, n) ? fail_wsa() : 0;
}
int llbw_getsockopt(int fd, int level, int name, void *v, int *n) { SOCK_OR_EBADF(fd); return getsockopt(sock_of(fd), level, name, v, n) ? fail_wsa() : 0; }
int llbw_getsockname(int fd, void *a, int *n) { SOCK_OR_EBADF(fd); return getsockname(sock_of(fd), a, n) ? fail_wsa() : 0; }
int llbw_getpeername(int fd, void *a, int *n) { SOCK_OR_EBADF(fd); return getpeername(sock_of(fd), a, n) ? fail_wsa() : 0; }
/* a loopback TCP pair: Windows 10+ has AF_UNIX stream sockets, but not socketpair() */
int llbw_socketpair(int domain, int type, int proto, int fds[2]) {
  (void)domain, (void)proto, (void)type; /* always a stream pair */
  SOCKET l = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP), a = INVALID_SOCKET, b = INVALID_SOCKET;
  struct sockaddr_in in = { 0 };
  int n = sizeof in;
  in.sin_family = AF_INET, in.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (l == INVALID_SOCKET || bind(l, (void *)&in, sizeof in) || getsockname(l, (void *)&in, &n) || listen(l, 1)) goto bad;
  a = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (a == INVALID_SOCKET || connect(a, (void *)&in, sizeof in)) goto bad;
  b = accept(l, NULL, NULL);
  if (b == INVALID_SOCKET) goto bad;
  closesocket(l);
  if ((fds[0] = sock_new(a)) < 0) { closesocket(b); return -1; }
  if ((fds[1] = sock_new(b)) < 0) { close(fds[0]); return -1; }
  return 0;
bad:
  fail_wsa();
  if (l != INVALID_SOCKET) closesocket(l);
  if (a != INVALID_SOCKET) closesocket(a);
  return -1;
}
int llbw_inet_pton(int af, const char *src, void *dst) { return inet_pton(af, src, dst); }
const char *llbw_inet_ntop(int af, const void *src, char *dst, int n) { return inet_ntop(af, src, dst, (size_t)n); }
unsigned short llbw_htons(unsigned short v) { return htons(v); }
unsigned short llbw_ntohs(unsigned short v) { return ntohs(v); }
unsigned int llbw_htonl(unsigned int v) { return htonl(v); }
unsigned int llbw_ntohl(unsigned int v) { return ntohl(v); }
const struct in6_addr llbw_in6addr_any = { { { 0 } } };
const struct in6_addr llbw_in6addr_loopback = { { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } } };

/* ======================================================================== */
/* descriptors                                                                */
/* ======================================================================== */

#define CRT_FD_MAX 8192
static unsigned char fd_nb[CRT_FD_MAX]; /* O_NONBLOCK set by fcntl, for pipes */

static int is_pipe(int fd) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  return h != INVALID_HANDLE_VALUE && GetFileType(h) == FILE_TYPE_PIPE;
}

int llbw_open(const char *path, int flags, ...) {
  int mode = 0;
  if (flags & _O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
  int nb = (flags & LLBW_O_NONBLOCK) != 0;
  flags &= ~LLBW_O_NONBLOCK;
  if (!(flags & _O_TEXT)) flags |= _O_BINARY;
  /* POSIX mode bits: only owner-write maps to anything (a file without it is read-only) */
  int pm = _S_IREAD | ((mode & 0200) || !(flags & _O_CREAT) ? _S_IWRITE : 0);
  int fd = -1;
  /* a directory opens as a read-only handle on POSIX; the CRT refuses it (EACCES), so it is reported as such */
  errno_t e = _sopen_s(&fd, path, flags, _SH_DENYNO, pm);
  if (e) { errno = e; return -1; }
  if (nb && fd < CRT_FD_MAX) fd_nb[fd] = 1;
  return fd;
}

ssize_t read(int fd, void *buf, size_t n) {
  if (is_sock(fd)) return llbw_recv(fd, buf, n, 0);
  if (n > (1u << 30)) n = 1u << 30; /* a short read, as POSIX allows */
  llbw_precommit(buf, n);
  if (fd >= 0 && fd < CRT_FD_MAX && fd_nb[fd] && is_pipe(fd)) {
    DWORD avail = 0;
    if (!PeekNamedPipe((HANDLE)_get_osfhandle(fd), NULL, 0, NULL, &avail, NULL)) {
      if (GetLastError() == ERROR_BROKEN_PIPE) return 0;
      return fail_win();
    }
    if (avail == 0) { errno = EAGAIN; return -1; }
    if (n > avail) n = avail;
  }
  int r = _read(fd, buf, (unsigned)n);
  return r;
}

ssize_t write(int fd, const void *buf, size_t n) {
  if (is_sock(fd)) return llbw_send(fd, buf, n, 0);
  llbw_precommit(buf, n);
  size_t done = 0;
  while (done < n) {
    size_t c = n - done > (1u << 30) ? (1u << 30) : n - done;
    int r = _write(fd, (const char *)buf + done, (unsigned)c);
    if (r < 0) return done ? (ssize_t)done : -1;
    if (r == 0) break;
    done += (size_t)r;
  }
  return (ssize_t)done;
}

int close(int fd) {
  if (is_sock(fd)) {
    AcquireSRWLockExclusive(&socks_lock);
    SOCKET s = socks[fd - LLBW_SOCK_BASE];
    socks[fd - LLBW_SOCK_BASE] = 0;
    ReleaseSRWLockExclusive(&socks_lock);
    return closesocket(s) ? fail_wsa() : 0;
  }
  if (fd >= 0 && fd < CRT_FD_MAX) fd_nb[fd] = 0;
  return _close(fd);
}

off_t lseek(int fd, off_t off, int whence) { return _lseeki64(fd, off, whence); }

/* positional IO: ReadFile / WriteFile with an OVERLAPPED offset (tools/llb_platform.h's form). Unlike
 * POSIX pread this moves the file position; nothing that calls it also reads at the position. */
ssize_t pread(int fd, void *buf, size_t n, off_t off) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
  if (n > (1u << 30)) n = 1u << 30;
  llbw_precommit(buf, n);
  OVERLAPPED o = { 0 };
  o.Offset = (DWORD)((unsigned long long)off & 0xFFFFFFFFu);
  o.OffsetHigh = (DWORD)((unsigned long long)off >> 32);
  DWORD got = 0;
  if (!ReadFile(h, buf, (DWORD)n, &got, &o)) {
    if (GetLastError() == ERROR_HANDLE_EOF) return 0;
    return fail_win();
  }
  return (ssize_t)got;
}
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
  if (n > (1u << 30)) n = 1u << 30;
  OVERLAPPED o = { 0 };
  o.Offset = (DWORD)((unsigned long long)off & 0xFFFFFFFFu);
  o.OffsetHigh = (DWORD)((unsigned long long)off >> 32);
  DWORD put = 0;
  if (!WriteFile(h, buf, (DWORD)n, &put, &o)) return fail_win();
  return (ssize_t)put;
}
ssize_t readv(int fd, const struct llbw_iovec *v, int n) {
  ssize_t t = 0;
  for (int i = 0; i < n; i++) {
    ssize_t r = read(fd, v[i].iov_base, v[i].iov_len);
    if (r < 0) return t ? t : -1;
    t += r;
    if ((size_t)r < v[i].iov_len) break;
  }
  return t;
}
ssize_t writev(int fd, const struct llbw_iovec *v, int n) {
  ssize_t t = 0;
  for (int i = 0; i < n; i++) {
    ssize_t r = write(fd, v[i].iov_base, v[i].iov_len);
    if (r < 0) return t ? t : -1;
    t += r;
    if ((size_t)r < v[i].iov_len) break;
  }
  return t;
}
int ftruncate(int fd, off_t n) { errno_t e = _chsize_s(fd, n); if (e) { errno = e; return -1; } return 0; }
int fsync(int fd) { return _commit(fd); }
int pipe(int fds[2]) { return _pipe(fds, 1 << 16, _O_BINARY | _O_NOINHERIT); }
int dup(int fd) { return _dup(fd); }
int dup2(int a, int b) { return _dup2(a, b) ? -1 : b; }
int isatty(int fd) { return _isatty(fd); }
int unlink(const char *path) { return _unlink(path); }
int rmdir(const char *path) { return _rmdir(path); }
/* X_OK has no meaning to the CRT (it reports it as an invalid parameter): a file that exists counts */
int access(const char *path, int mode) { return _access(path, mode & (R_OK | W_OK)); }
int llbw_mkdir(const char *path, mode_t mode) { (void)mode; return _mkdir(path); }
char *getcwd(char *buf, size_t n) { return _getcwd(buf, (int)n); }
int chdir(const char *path) { return _chdir(path); }
pid_t getpid(void) { return (pid_t)GetCurrentProcessId(); }
pid_t getppid(void) {
  DWORD me = GetCurrentProcessId(), parent = 0;
  HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (s == INVALID_HANDLE_VALUE) return 0;
  PROCESSENTRY32W e = { .dwSize = sizeof e };
  for (BOOL ok = Process32FirstW(s, &e); ok; ok = Process32NextW(s, &e))
    if (e.th32ProcessID == me) { parent = e.th32ParentProcessID; break; }
  CloseHandle(s);
  return (pid_t)parent;
}
pid_t fork(void) { errno = ENOSYS; return -1; }
int execv(const char *path, char *const argv[]) { (void)path, (void)argv; errno = ENOSYS; return -1; }
int execvp(const char *file, char *const argv[]) { (void)file, (void)argv; errno = ENOSYS; return -1; }
pid_t waitpid(pid_t pid, int *status, int options) { (void)pid, (void)status, (void)options; errno = ECHILD; return -1; }
/* /proc/self/exe (the runtime's .gpu cache path) is the module's own path; nothing else is a link here */
ssize_t readlink(const char *path, char *buf, size_t n) {
  if (strcmp(path, "/proc/self/exe") != 0) { errno = EINVAL; return -1; }
  DWORD r = GetModuleFileNameA(NULL, buf, (DWORD)n);
  if (r == 0 || r >= n) { errno = ENAMETOOLONG; return -1; }
  return (ssize_t)r;
}
long sysconf(int name) {
  switch (name) {
  case _SC_NPROCESSORS_ONLN: case _SC_NPROCESSORS_CONF: return (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  case _SC_PAGESIZE: return 4096;
  case _SC_PHYS_PAGES: {
    MEMORYSTATUSEX m = { .dwLength = sizeof m };
    return GlobalMemoryStatusEx(&m) ? (long)(m.ullTotalPhys / 4096) : -1;
  }
  default: errno = EINVAL; return -1;
  }
}
int fcntl(int fd, int cmd, ...) {
  va_list ap;
  va_start(ap, cmd);
  long arg = va_arg(ap, long);
  va_end(ap);
  switch (cmd) {
  case F_GETFD: return 0;
  case F_SETFD: return 0; /* descriptors this shim opens are already not inherited */
  case F_GETFL:
    if (is_sock(fd)) return sock_nb[fd - LLBW_SOCK_BASE] ? LLBW_O_NONBLOCK : 0;
    return fd >= 0 && fd < CRT_FD_MAX && fd_nb[fd] ? LLBW_O_NONBLOCK : 0;
  case F_SETFL:
    if (is_sock(fd)) return sock_set_nb(fd, (arg & LLBW_O_NONBLOCK) != 0);
    if (fd < 0 || fd >= CRT_FD_MAX || _get_osfhandle(fd) == -1) { errno = EBADF; return -1; }
    fd_nb[fd] = (arg & LLBW_O_NONBLOCK) != 0; /* honoured by read() on a pipe */
    return 0;
  default: errno = EINVAL; return -1;
  }
}
char *mkdtemp(char *tmpl) {
  size_t n = strlen(tmpl);
  if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX")) { errno = EINVAL; return NULL; }
  for (int tries = 0; tries < 100; tries++) {
    unsigned r = (unsigned)GetTickCount64() ^ (unsigned)GetCurrentProcessId() * 2654435761u ^ (unsigned)tries * 40503u;
    for (int i = 0; i < 6; i++) tmpl[n - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[(r >> (i * 5)) % 36];
    if (_mkdir(tmpl) == 0) return tmpl;
    if (errno != EEXIST) return NULL;
  }
  return NULL;
}
int mkstemp(char *tmpl) {
  size_t n = strlen(tmpl);
  if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX")) { errno = EINVAL; return -1; }
  for (int tries = 0; tries < 100; tries++) {
    unsigned r = (unsigned)GetTickCount64() ^ (unsigned)GetCurrentProcessId() * 2654435761u ^ (unsigned)tries * 40503u;
    for (int i = 0; i < 6; i++) tmpl[n - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[(r >> (i * 5)) % 36];
    int fd = -1;
    if (!_sopen_s(&fd, tmpl, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYNO, _S_IREAD | _S_IWRITE)) return fd;
    if (errno != EEXIST) return -1;
  }
  return -1;
}

/* struct stat with POSIX timespecs (compat/sys/stat.h); the impl sees the UCRT's own names */
struct llbw_stat {
  unsigned int st_dev, st_ino;
  unsigned short st_mode;
  short st_nlink;
  short st_uid, st_gid;
  unsigned int st_rdev;
  long long st_size;
  struct timespec st_atim, st_mtim, st_ctim;
};
static void stat_fill(struct llbw_stat *o, const struct _stat64 *s) {
  memset(o, 0, sizeof *o);
  o->st_dev = s->st_dev, o->st_ino = s->st_ino, o->st_mode = s->st_mode, o->st_nlink = s->st_nlink;
  o->st_uid = s->st_uid, o->st_gid = s->st_gid, o->st_rdev = s->st_rdev, o->st_size = s->st_size;
  o->st_atim.tv_sec = s->st_atime, o->st_mtim.tv_sec = s->st_mtime, o->st_ctim.tv_sec = s->st_ctime;
}
int llbw_stat(const char *path, struct llbw_stat *st) {
  struct _stat64 s;
  if (_stat64(path, &s)) return -1;
  stat_fill(st, &s);
  return 0;
}
int llbw_fstat(int fd, struct llbw_stat *st) {
  if (is_sock(fd)) { memset(st, 0, sizeof *st); st->st_mode = 0140000; return 0; } /* S_IFSOCK */
  struct _stat64 s;
  if (_fstat64(fd, &s)) return -1;
  stat_fill(st, &s);
  return 0;
}

/* ======================================================================== */
/* poll / select                                                              */
/* ======================================================================== */

/* one descriptor's readiness now: sockets through WSAPoll (one at a time is enough for the few the runtime
 * waits on), pipes through PeekNamedPipe, anything else (a file, a console) ready */
static short fd_ready(int fd, short want) {
  short got = 0;
  if (is_sock(fd)) {
    WSAPOLLFD p = { sock_of(fd), 0, 0 };
    if (want & LLBW_POLLIN) p.events |= POLLRDNORM;
    if (want & LLBW_POLLOUT) p.events |= POLLWRNORM;
    if (WSAPoll(&p, 1, 0) > 0) {
      if (p.revents & (POLLRDNORM | POLLRDBAND)) got |= LLBW_POLLIN;
      if (p.revents & POLLWRNORM) got |= LLBW_POLLOUT;
      if (p.revents & POLLERR) got |= LLBW_POLLERR;
      if (p.revents & POLLHUP) got |= LLBW_POLLHUP | (want & LLBW_POLLIN);
      if (p.revents & POLLNVAL) got |= LLBW_POLLNVAL;
    }
    return got;
  }
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) return LLBW_POLLNVAL;
  if (GetFileType(h) == FILE_TYPE_PIPE) {
    DWORD avail = 0;
    if (want & LLBW_POLLIN) {
      if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) got |= LLBW_POLLHUP | LLBW_POLLIN;
      else if (avail) got |= LLBW_POLLIN;
    }
    if (want & LLBW_POLLOUT) got |= LLBW_POLLOUT;
    return got;
  }
  return want & (LLBW_POLLIN | LLBW_POLLOUT);
}
static void wait_a_little(int round) {
  if (round < 64) SwitchToThread();
  else Sleep(1);
}
int llbw_poll(struct llbw_pollfd *fds, unsigned long n, int timeout_ms) {
  ULONGLONG until = timeout_ms < 0 ? ~0ull : GetTickCount64() + (ULONGLONG)timeout_ms;
  for (int round = 0;; round++) {
    int k = 0;
    for (unsigned long i = 0; i < n; i++) {
      fds[i].revents = fds[i].fd < 0 ? 0 : fd_ready(fds[i].fd, fds[i].events);
      k += fds[i].revents != 0;
    }
    if (k || timeout_ms == 0 || GetTickCount64() >= until) return k;
    wait_a_little(round);
  }
}
#define BIT(s, fd) (((s)->fds_bits[(fd) / 64] >> ((fd) % 64)) & 1)
int llbw_select(int nfds, llbw_fd_set *r, llbw_fd_set *w, llbw_fd_set *x, struct llbw_timeval *tv) {
  if (nfds > LLBW_SOCK_BASE + LLBW_SOCK_MAX) { errno = EINVAL; return -1; }
  int words = (nfds + 63) / 64;
  unsigned long long rin[LLBW_FDSET_WORDS], win[LLBW_FDSET_WORDS];
  for (int i = 0; i < words; i++) rin[i] = r ? r->fds_bits[i] : 0, win[i] = w ? w->fds_bits[i] : 0;
  long long ms = tv ? (long long)tv->tv_sec * 1000 + tv->tv_usec / 1000 : -1;
  ULONGLONG until = ms < 0 ? ~0ull : GetTickCount64() + (ULONGLONG)ms;
  for (int round = 0;; round++) {
    int k = 0;
    for (int i = 0; i < words; i++) {
      unsigned long long want = rin[i] | win[i], rout = 0, wout = 0;
      while (want) {
        int b = __builtin_ctzll(want);
        want &= want - 1;
        int fd = i * 64 + b;
        short ask = (short)((rin[i] >> b & 1 ? LLBW_POLLIN : 0) | (win[i] >> b & 1 ? LLBW_POLLOUT : 0));
        short got = fd_ready(fd, ask);
        if (got & LLBW_POLLNVAL) { errno = EBADF; return -1; }
        if ((rin[i] >> b & 1) && (got & (LLBW_POLLIN | LLBW_POLLHUP | LLBW_POLLERR))) rout |= 1ull << b, k++;
        if ((win[i] >> b & 1) && (got & (LLBW_POLLOUT | LLBW_POLLERR))) wout |= 1ull << b, k++;
      }
      if (r) r->fds_bits[i] = rout;
      if (w) w->fds_bits[i] = wout;
    }
    if (x) for (int i = 0; i < words; i++) x->fds_bits[i] = 0;
    if (k || ms == 0 || GetTickCount64() >= until) return k;
    wait_a_little(round);
  }
}

/* ======================================================================== */
/* signals                                                                    */
/* ======================================================================== */

static int crt_signal(int sig) {
  return sig == SIGINT || sig == SIGTERM || sig == SIGSEGV || sig == SIGFPE || sig == SIGILL || sig == SIGABRT
    || sig == SIGBREAK;
}
void (*llbw_signal(int sig, void (*fn)(int)))(int) {
  if (crt_signal(sig)) return signal(sig, fn);
  return SIG_DFL; /* no such signal on Windows: nothing will ever be delivered */
}
int sigaction(int sig, const struct sigaction *sa, struct sigaction *old) {
  if (old) memset(old, 0, sizeof *old);
  if (!sa || !crt_signal(sig)) return 0;
  void (*fn)(int) = sa->sa_handler;
  if (signal(sig, fn) == SIG_ERR) { errno = EINVAL; return -1; }
  return 0;
}
int sigemptyset(sigset_t *s) { *s = 0; return 0; }
int sigfillset(sigset_t *s) { *s = ~0ull; return 0; }
int sigaddset(sigset_t *s, int sig) { *s |= 1ull << (sig & 63); return 0; }
int sigdelset(sigset_t *s, int sig) { *s &= ~(1ull << (sig & 63)); return 0; }
int sigismember(const sigset_t *s, int sig) { return (int)(*s >> (sig & 63) & 1); }
int sigprocmask(int how, const sigset_t *s, sigset_t *old) { (void)how, (void)s; if (old) *old = 0; return 0; }
int pthread_sigmask(int how, const sigset_t *s, sigset_t *old) { return sigprocmask(how, s, old); }
int sigaltstack(const stack_t *ss, stack_t *old) { (void)ss; if (old) memset(old, 0, sizeof *old); return 0; }
int kill(pid_t pid, int sig) {
  if (pid == 0 || pid == (pid_t)GetCurrentProcessId()) {
    if (sig == 0) return 0;
    if (crt_signal(sig)) return raise(sig);
    if (sig == SIGKILL) TerminateProcess(GetCurrentProcess(), 128 + SIGKILL);
    return 0;
  }
  if (pid < 0) { errno = ENOSYS; return -1; }
  HANDLE h = OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
  if (!h) { errno = ESRCH; return -1; }
  BOOL ok = sig == 0 ? TRUE : TerminateProcess(h, 128 + (unsigned)sig);
  CloseHandle(h);
  return ok ? 0 : fail_win();
}

/* ======================================================================== */
/* stdio: cookie streams, an assignable stdout, getline                      */
/* ======================================================================== */

int llbw_fputc(int c, FILE *f);
typedef struct { cookie_io_functions_t io; void *c; int err; } Cookie;
#define COOKIES_MAX 256
static Cookie *cookies[COOKIES_MAX];
static SRWLOCK cookies_lock = SRWLOCK_INIT;
static FILE *cur_stdout;

static Cookie *cookie_of(FILE *f) {
  if (!f) return NULL;
  Cookie *k = NULL;
  AcquireSRWLockShared(&cookies_lock);
  for (int i = 0; i < COOKIES_MAX; i++)
    if (cookies[i] == (Cookie *)f) { k = cookies[i]; break; }
  ReleaseSRWLockShared(&cookies_lock);
  return k;
}
FILE *fopencookie(void *c, const char *mode, cookie_io_functions_t io) {
  (void)mode;
  Cookie *k = calloc(1, sizeof *k);
  if (!k) { errno = ENOMEM; return NULL; }
  k->io = io, k->c = c;
  AcquireSRWLockExclusive(&cookies_lock);
  int i = 0;
  while (i < COOKIES_MAX && cookies[i]) i++;
  if (i < COOKIES_MAX) cookies[i] = k;
  ReleaseSRWLockExclusive(&cookies_lock);
  if (i == COOKIES_MAX) { free(k); errno = EMFILE; return NULL; }
  return (FILE *)k;
}
static size_t cookie_put(Cookie *k, const char *p, size_t n) {
  size_t done = 0;
  if (!k->io.write) { k->err = 1; return 0; }
  while (done < n) {
    ssize_t r = k->io.write(k->c, p + done, n - done);
    if (r <= 0) { k->err = 1; break; }
    done += (size_t)r;
  }
  return done;
}
FILE **llbw_stdout_ref(void) {
  if (!cur_stdout) cur_stdout = __acrt_iob_func(1);
  return &cur_stdout;
}
int llbw_vfprintf(FILE *f, const char *fmt, va_list ap) {
  Cookie *k = cookie_of(f);
  if (!k) return vfprintf(f, fmt, ap);
  char small[1024], *p = small;
  va_list aq;
  va_copy(aq, ap);
  int n = vsnprintf(small, sizeof small, fmt, ap);
  if (n < 0) { va_end(aq); return n; }
  if ((size_t)n >= sizeof small) {
    p = malloc((size_t)n + 1);
    if (!p) { va_end(aq); return -1; }
    vsnprintf(p, (size_t)n + 1, fmt, aq);
  }
  va_end(aq);
  size_t put = cookie_put(k, p, (size_t)n);
  if (p != small) free(p);
  return put == (size_t)n ? n : -1;
}
int llbw_fprintf(FILE *f, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = llbw_vfprintf(f, fmt, ap); va_end(ap); return r; }
int llbw_vprintf(const char *fmt, va_list ap) { return llbw_vfprintf(*llbw_stdout_ref(), fmt, ap); }
int llbw_printf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = llbw_vprintf(fmt, ap); va_end(ap); return r; }
int llbw_fputs(const char *s, FILE *f) {
  Cookie *k = cookie_of(f);
  if (!k) return fputs(s, f);
  size_t n = strlen(s);
  return cookie_put(k, s, n) == n ? 1 : EOF;
}
int llbw_puts(const char *s) {
  FILE *f = *llbw_stdout_ref();
  if (llbw_fputs(s, f) == EOF) return EOF;
  return llbw_fputc('\n', f) == EOF ? EOF : 1;
}
int llbw_fputc(int c, FILE *f) {
  Cookie *k = cookie_of(f);
  if (!k) return fputc(c, f);
  char b = (char)c;
  return cookie_put(k, &b, 1) == 1 ? (unsigned char)c : EOF;
}
int llbw_putchar(int c) { return llbw_fputc(c, *llbw_stdout_ref()); }
size_t llbw_fwrite(const void *p, size_t sz, size_t n, FILE *f) {
  Cookie *k = cookie_of(f);
  if (!k) return fwrite(p, sz, n, f);
  if (!sz || !n) return 0;
  return cookie_put(k, p, sz * n) / sz;
}
size_t llbw_fread(void *p, size_t sz, size_t n, FILE *f) {
  if (cookie_of(f)) { errno = EBADF; return 0; } /* the cookie streams here are write-only */
  llbw_precommit(p, sz * n);
  return fread(p, sz, n, f);
}
int llbw_fflush(FILE *f) { return cookie_of(f) ? 0 : fflush(f); } /* a cookie stream holds no buffer */
int llbw_fclose(FILE *f) {
  Cookie *k = cookie_of(f);
  if (!k) return fclose(f);
  AcquireSRWLockExclusive(&cookies_lock);
  for (int i = 0; i < COOKIES_MAX; i++) if (cookies[i] == k) cookies[i] = NULL;
  ReleaseSRWLockExclusive(&cookies_lock);
  int r = k->io.close ? k->io.close(k->c) : 0;
  if (cur_stdout == (FILE *)k) cur_stdout = __acrt_iob_func(1);
  free(k);
  return r ? EOF : 0;
}
int llbw_setvbuf(FILE *f, char *buf, int mode, size_t n) { return cookie_of(f) ? 0 : setvbuf(f, buf, mode, n); }
void llbw_setbuf(FILE *f, char *buf) { if (!cookie_of(f)) setbuf(f, buf); }
int llbw_ferror(FILE *f) { Cookie *k = cookie_of(f); return k ? k->err : ferror(f); }
int llbw_fileno(FILE *f) { if (cookie_of(f)) { errno = EBADF; return -1; } return _fileno(f); }

ssize_t getdelim(char **line, size_t *cap, int delim, FILE *f) {
  if (!line || !cap || !f) { errno = EINVAL; return -1; }
  if (!*line || !*cap) { *cap = 256; *line = realloc(*line, *cap); if (!*line) { errno = ENOMEM; return -1; } }
  size_t n = 0;
  _lock_file(f);
  for (;;) {
    int c = _getc_nolock(f);
    if (c == EOF) break;
    if (n + 2 > *cap) {
      size_t nc = *cap * 2;
      char *p = realloc(*line, nc);
      if (!p) { _unlock_file(f); errno = ENOMEM; return -1; }
      *line = p, *cap = nc;
    }
    (*line)[n++] = (char)c;
    if (c == delim) break;
  }
  _unlock_file(f);
  (*line)[n] = 0;
  return n ? (ssize_t)n : -1;
}
ssize_t getline(char **line, size_t *cap, FILE *f) { return getdelim(line, cap, '\n', f); }

typedef struct { char **buf; size_t *len, cap; } MemStream;
static ssize_t mem_write(void *c, const char *p, size_t n) {
  MemStream *m = c;
  if (*m->len + n + 1 > m->cap) {
    size_t nc = m->cap ? m->cap * 2 : 256;
    while (nc < *m->len + n + 1) nc *= 2;
    char *q = realloc(*m->buf, nc);
    if (!q) return -1;
    *m->buf = q, m->cap = nc;
  }
  memcpy(*m->buf + *m->len, p, n);
  *m->len += n;
  (*m->buf)[*m->len] = 0;
  return (ssize_t)n;
}
static int mem_close(void *c) { free(c); return 0; }
FILE *open_memstream(char **buf, size_t *len) {
  MemStream *m = calloc(1, sizeof *m);
  if (!m) return NULL;
  m->buf = buf, m->len = len;
  *buf = calloc(1, 1), *len = 0, m->cap = 1;
  cookie_io_functions_t io = { NULL, mem_write, NULL, mem_close };
  FILE *f = fopencookie(m, "w", io);
  if (!f) free(m);
  return f;
}

/* ======================================================================== */
/* stdlib / string                                                            */
/* ======================================================================== */

int setenv(const char *name, const char *value, int overwrite) {
  if (!name || !*name || strchr(name, '=')) { errno = EINVAL; return -1; }
  if (!overwrite && getenv(name)) return 0;
  /* _putenv_s with "" would delete the variable; POSIX keeps it, empty. The CRT cannot hold an empty
   * value, so the empty string is stored through SetEnvironmentVariable for children and getenv says NULL. */
  if (!*value) { SetEnvironmentVariableA(name, ""); return 0; }
  errno_t e = _putenv_s(name, value);
  if (e) { errno = e; return -1; }
  return 0;
}
int unsetenv(const char *name) { _putenv_s(name, ""); return 0; }
char *realpath(const char *path, char *resolved) {
  char *r = _fullpath(resolved, path, resolved ? 260 : 0);
  if (!r) { errno = ENOENT; return NULL; }
  DWORD a = GetFileAttributesA(r);
  if (a == INVALID_FILE_ATTRIBUTES) { if (!resolved) free(r); errno = ENOENT; return NULL; }
  return r;
}
char *strndup(const char *s, size_t n) {
  size_t k = strnlen(s, n);
  char *p = malloc(k + 1);
  if (!p) return NULL;
  memcpy(p, s, k);
  p[k] = 0;
  return p;
}
void *memmem(const void *h, size_t hn, const void *n, size_t nn) {
  if (!nn) return (void *)h;
  for (const char *p = h, *e = (const char *)h + hn; (size_t)(e - p) >= nn; p++)
    if (*p == *(const char *)n && !memcmp(p, n, nn)) return (void *)p;
  return NULL;
}
char *stpcpy(char *d, const char *s) { size_t n = strlen(s); memcpy(d, s, n + 1); return d + n; }
int strerror_r(int e, char *buf, size_t n) { return strerror_s(buf, n, e); }

/* ======================================================================== */
/* process start                                                              */
/* ======================================================================== */

/* a CRT function handed a bad descriptor or flag returns its error (EBADF, EINVAL), as POSIX does, instead
 * of ending the process through the default invalid-parameter handler */
static void quiet_invalid_parameter(const wchar_t *e, const wchar_t *f, const wchar_t *file, unsigned line, uintptr_t r) {
  (void)e, (void)f, (void)file, (void)line, (void)r;
}
__attribute__((constructor)) static void llbw_start(void) {
  _set_invalid_parameter_handler(quiet_invalid_parameter);
  /* binary everywhere: POSIX has no text mode, and llb-serve reads the engine's stdout byte for byte */
  _set_fmode(_O_BINARY);
  _setmode(0, _O_BINARY);
  _setmode(1, _O_BINARY);
  _setmode(2, _O_BINARY);
  tls_self = TlsAlloc();
  const char *ft = getenv("LLBW_FAULT_TRACE");
  fault_trace = ft && *ft && strcmp(ft, "0");
  AddVectoredExceptionHandler(1, map_fault);
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
  /* 1 ms waits in poll/select need a 1 ms timer tick (Windows' default is 15.6 ms) */
  timeBeginPeriod(1);
}
