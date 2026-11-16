/* llbw_posix.h: the Windows POSIX shim's one declaration header (XPLAT-WIN).
 *
 * backends/windows/compat/ is an include directory, passed with -I, that lets the
 * POSIX C this repo emits and links (the Bend runtime preamble in the emitted
 * llb-engine.c, tools/sys1dual/dual.c, tools/metrics, tools/glue, tools/clef3)
 * compile on Windows (clang, x86_64-pc-windows-msvc, UCRT) WITHOUT editing any of
 * those sources or the Bend compiler. The header names it shadows (pthread.h,
 * unistd.h, sys/mman.h, ...) are thin and all point here; the implementation is
 * llbw_posix.c, which is compiled WITHOUT -I compat so that it sees the real
 * UCRT and Win32 headers. POSIX builds never see this directory.
 *
 * What is real and what is a stub is listed in README.md beside this file.
 * Nothing here pulls in windows.h or winsock2.h: their macros (near, far, ERROR,
 * min, max, IN, OUT, ...) would collide with emitted identifiers, so every type
 * the callers see is declared here with the Win32 ABI it is passed to.
 */
#ifndef LLBW_POSIX_H
#define LLBW_POSIX_H
#ifndef _WIN32
#error "backends/windows/compat is the Windows POSIX shim; do not put it on a POSIX include path"
#endif

/* XPLAT-WIN: the UCRT's own "POSIX names" (int read(int, void *, unsigned), a 32-bit off_t, one-argument
 * mkdir, ...) clash with the POSIX forms declared here, so every file built with this directory is built
 * with -D_CRT_DECLARE_NONSTDC_NAMES=0 (backends/windows/build.ps1 passes it). */
#include <stddef.h>
#include <stdint.h>
#if !defined(_CRT_DECLARE_NONSTDC_NAMES) || _CRT_DECLARE_NONSTDC_NAMES
#error "build with -D_CRT_DECLARE_NONSTDC_NAMES=0 (backends/windows/compat/README.md)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- basic POSIX types the UCRT leaves out under __STDC__ ---------------- */
/* XPLAT-WIN: clang defines __STDC__, so the UCRT declares none of its "POSIX
 * names" (off_t, open, read, ...). off_t is 64-bit here on purpose: the UCRT's
 * own _off_t is a 32-bit long, and the engine passes GGUF offsets past 2 GiB. */
#ifndef LLBW_TYPES_DEFINED
#define LLBW_TYPES_DEFINED
typedef long long ssize_t;
typedef long long off_t;
typedef int pid_t;
typedef unsigned short mode_t;
typedef unsigned int uid_t;
typedef unsigned int gid_t;
typedef unsigned int useconds_t;
typedef int socklen_t;
typedef unsigned short sa_family_t;
typedef unsigned short in_port_t;
typedef unsigned int in_addr_t;
#endif

/* ---- pthread (Win32 threads, SRWLOCK, CONDITION_VARIABLE, InitOnce) ------ */
/* Each object is one pointer-sized word, the size of the Win32 primitive behind it,
 * so the static initialisers are all zero, as SRWLOCK_INIT / CONDITION_VARIABLE_INIT
 * / INIT_ONCE_STATIC_INIT are. A mutex is an exclusive SRWLOCK: not recursive, and
 * it is not owned by a thread (as PTHREAD_MUTEX_NORMAL / DEFAULT allow). */
typedef struct llbw_thread *pthread_t;
typedef struct { int detach; size_t stack; } pthread_attr_t;
typedef struct { void *p; } pthread_mutex_t;
typedef struct { int kind; } pthread_mutexattr_t;
typedef struct { void *p; } pthread_cond_t;
typedef struct { int unused; } pthread_condattr_t;
typedef struct { void *p; } pthread_rwlock_t;
typedef struct { int unused; } pthread_rwlockattr_t;
typedef struct { void *p; } pthread_once_t;
typedef unsigned long pthread_key_t;
#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_COND_INITIALIZER {0}
#define PTHREAD_RWLOCK_INITIALIZER {0}
#define PTHREAD_ONCE_INIT {0}
#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CREATE_DETACHED 1
#define PTHREAD_MUTEX_NORMAL 0
#define PTHREAD_MUTEX_DEFAULT 0
#define PTHREAD_MUTEX_ERRORCHECK 2
#define PTHREAD_MUTEX_RECURSIVE 1

int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg);
int pthread_join(pthread_t t, void **ret);
int pthread_detach(pthread_t t);
pthread_t pthread_self(void);
int pthread_equal(pthread_t a, pthread_t b);
void pthread_exit(void *ret);
int pthread_attr_init(pthread_attr_t *a);
int pthread_attr_destroy(pthread_attr_t *a);
int pthread_attr_setdetachstate(pthread_attr_t *a, int s);
int pthread_attr_setstacksize(pthread_attr_t *a, size_t n);
int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a);
int pthread_mutex_destroy(pthread_mutex_t *m);
int pthread_mutex_lock(pthread_mutex_t *m);
int pthread_mutex_trylock(pthread_mutex_t *m);
int pthread_mutex_unlock(pthread_mutex_t *m);
int pthread_mutexattr_init(pthread_mutexattr_t *a);
int pthread_mutexattr_destroy(pthread_mutexattr_t *a);
int pthread_mutexattr_settype(pthread_mutexattr_t *a, int k);
int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a);
int pthread_cond_destroy(pthread_cond_t *c);
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m);
struct timespec;
int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *abs);
int pthread_cond_signal(pthread_cond_t *c);
int pthread_cond_broadcast(pthread_cond_t *c);
int pthread_rwlock_init(pthread_rwlock_t *l, const pthread_rwlockattr_t *a);
int pthread_rwlock_destroy(pthread_rwlock_t *l);
int pthread_rwlock_rdlock(pthread_rwlock_t *l);
int pthread_rwlock_wrlock(pthread_rwlock_t *l);
int pthread_rwlock_unlock(pthread_rwlock_t *l);
int pthread_once(pthread_once_t *o, void (*fn)(void));
int pthread_key_create(pthread_key_t *k, void (*dtor)(void *));
int pthread_key_delete(pthread_key_t k);
void *pthread_getspecific(pthread_key_t k);
int pthread_setspecific(pthread_key_t k, const void *v);
int sched_yield(void);

/* ---- clocks -------------------------------------------------------------- */
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3
#define CLOCK_MONOTONIC_RAW 4
typedef int clockid_t;
int clock_gettime(clockid_t id, struct timespec *ts);
int nanosleep(const struct timespec *req, struct timespec *rem);
int usleep(useconds_t us);
unsigned int sleep(unsigned int s);

/* ---- file descriptors (unistd.h / fcntl.h) ------------------------------- */
/* XPLAT-WIN: open() always adds _O_BINARY (text mode would turn \r\n into \n and
 * stop a read at 0x1A); pread/pwrite go through ReadFile/WriteFile with an
 * OVERLAPPED offset, at most 1 GiB a call as tools/llb_platform.h does. Sockets
 * get descriptors from LLBW_SOCK_BASE up (see sys/socket.h), so read/write/close
 * know which they hold. */
int llbw_open(const char *path, int flags, ...);
ssize_t read(int fd, void *buf, size_t n);
ssize_t write(int fd, const void *buf, size_t n);
int close(int fd);
off_t lseek(int fd, off_t off, int whence);
ssize_t pread(int fd, void *buf, size_t n, off_t off);
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off);
int ftruncate(int fd, off_t n);
int fsync(int fd);
int pipe(int fds[2]);
int dup(int fd);
int dup2(int a, int b);
int isatty(int fd);
int unlink(const char *path);
int rmdir(const char *path);
int access(const char *path, int mode);
int llbw_mkdir(const char *path, mode_t mode);
char *getcwd(char *buf, size_t n);
int chdir(const char *path);
pid_t getpid(void);
pid_t getppid(void);
pid_t fork(void);
int execv(const char *path, char *const argv[]);
int execvp(const char *file, char *const argv[]);
ssize_t readlink(const char *path, char *buf, size_t n);
long sysconf(int name);
int fcntl(int fd, int cmd, ...);
char *mkdtemp(char *tmpl);
int mkstemp(char *tmpl);
#define F_OK 0
#define X_OK 1
#define W_OK 2
#define R_OK 4
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#define _SC_NPROCESSORS_ONLN 84
#define _SC_NPROCESSORS_CONF 83
#define _SC_PAGESIZE 30
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_PHYS_PAGES 85
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define FD_CLOEXEC 1

/* ---- memory maps (sys/mman.h) -------------------------------------------- */
/* XPLAT-WIN: an anonymous map is a VirtualAlloc reservation whose pages are
 * committed on first touch by a vectored exception handler, which is what
 * MAP_NORESERVE gives on Linux (the runtime reserves 8 GiB of corpus and 2 GiB of
 * stack a worker and touches a fraction; committing it all would pass the commit
 * limit). A file map is CreateFileMapping + MapViewOfFile. */
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE 0x8000
#define MAP_FAILED ((void *)-1)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#define MS_ASYNC 1
#define MS_SYNC 4
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off);
int munmap(void *addr, size_t len);
int mprotect(void *addr, size_t len, int prot);
int madvise(void *addr, size_t len, int advice);
int msync(void *addr, size_t len, int flags);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);
/* commit the reserved pages of [p, p + n) now: a kernel call (ReadFile into a fresh
 * reservation) does not take the first-touch fault, it fails with ERROR_NOACCESS */
void llbw_precommit(const void *p, size_t n);

/* ---- signals (signal.h additions) ---------------------------------------- */
/* XPLAT-WIN: only SIGINT, SIGTERM, SIGSEGV, SIGFPE, SIGILL and SIGABRT exist in
 * the UCRT. sigaction installs those through signal(); the others (SIGHUP,
 * SIGPIPE, SIGBUS, SIGCHLD, ...) have nothing to deliver them on Windows and are
 * accepted and ignored, as is sigaltstack. kill() can stop another process
 * (TerminateProcess) or raise in this one; it cannot deliver a handler elsewhere. */
#define SIGHUP 1
#define SIGQUIT 3
#define SIGTRAP 5
#define SIGKILL 9
#define SIGBUS 10
#define SIGUSR1 30
#define SIGUSR2 31
#define SIGPIPE 13
#define SIGALRM 14
#define SIGCHLD 17
#define SIGCONT 18
#define SIGSTOP 19
#define SIGWINCH 28
#define SA_ONSTACK 0x08000000
#define SA_RESTART 0x10000000
#define SA_NODEFER 0x40000000
#define SA_RESETHAND 0x80000000
#define SA_SIGINFO 0x00000004
#define SIGSTKSZ 65536
#define MINSIGSTKSZ 16384
#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2
typedef unsigned long long sigset_t;
typedef struct { int si_signo, si_code, si_errno; void *si_addr; } siginfo_t;
struct sigaction {
  void (*sa_handler)(int);
  void (*sa_sigaction)(int, siginfo_t *, void *);
  sigset_t sa_mask;
  int sa_flags;
};
typedef struct { void *ss_sp; int ss_flags; size_t ss_size; } stack_t;
int sigaction(int sig, const struct sigaction *sa, struct sigaction *old);
int sigemptyset(sigset_t *s);
int sigfillset(sigset_t *s);
int sigaddset(sigset_t *s, int sig);
int sigdelset(sigset_t *s, int sig);
int sigismember(const sigset_t *s, int sig);
int sigprocmask(int how, const sigset_t *s, sigset_t *old);
int pthread_sigmask(int how, const sigset_t *s, sigset_t *old);
int sigaltstack(const stack_t *ss, stack_t *old);
int kill(pid_t pid, int sig);
void (*llbw_signal(int sig, void (*fn)(int)))(int);

/* ---- child processes (sys/wait.h) ---------------------------------------- */
/* XPLAT-WIN: there is no fork. fork() fails with ENOSYS and waitpid() with ECHILD,
 * so the runtime's --workers N (the only fork caller) refuses on Windows. */
#define WNOHANG 1
#define WUNTRACED 2
#define WIFEXITED(s) (((s) & 0x7f) == 0)
#define WEXITSTATUS(s) (((s) >> 8) & 0xff)
#define WIFSIGNALED(s) (((s) & 0x7f) != 0 && ((s) & 0x7f) != 0x7f)
#define WTERMSIG(s) ((s) & 0x7f)
pid_t waitpid(pid_t pid, int *status, int options);

/* ---- sockets, poll, select ------------------------------------------------ */
/* Everything below collides with winsock2.h (struct pollfd, POLLIN's value, fd_set,
 * struct timeval, the socket names), so llbw_posix.c, which includes winsock2.h,
 * defines LLBW_IMPL and keeps its own LLBW_-prefixed copies of these numbers. */
#define LLBW_SOCK_BASE 4096
#define LLBW_SOCK_MAX 4096
#ifndef LLBW_IMPL
#define AF_UNSPEC 0
#define AF_UNIX 1
#define AF_INET 2
#define AF_INET6 23
#define PF_INET AF_INET
#define PF_INET6 AF_INET6
#define SOCK_STREAM 1
#define SOCK_DGRAM 2
#define SOCK_NONBLOCK 0x40000000
#define SOCK_CLOEXEC 0x20000000
#define IPPROTO_IP 0
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define IPPROTO_IPV6 41
#define IPV6_V6ONLY 27
#define TCP_NODELAY 1
#define SOL_SOCKET 0xffff
#define SO_REUSEADDR 0x0004
#define SO_KEEPALIVE 0x0008
#define SO_BROADCAST 0x0020
#define SO_LINGER 0x0080
#define SO_SNDBUF 0x1001
#define SO_RCVBUF 0x1002
#define SO_SNDTIMEO 0x1005
#define SO_RCVTIMEO 0x1006
#define SO_ERROR 0x1007
#define SO_TYPE 0x1008
#define SOMAXCONN 0x7fffffff
#define MSG_OOB 0x1
#define MSG_PEEK 0x2
#define MSG_DONTWAIT 0x40000000
#define MSG_NOSIGNAL 0
#define SHUT_RD 0
#define SHUT_WR 1
#define SHUT_RDWR 2
#define INADDR_ANY ((in_addr_t)0x00000000)
#define INADDR_LOOPBACK ((in_addr_t)0x7f000001)
#define INADDR_NONE ((in_addr_t)0xffffffff)
#define INET_ADDRSTRLEN 16
#define INET6_ADDRSTRLEN 46
struct sockaddr { sa_family_t sa_family; char sa_data[14]; };
struct in_addr { in_addr_t s_addr; };
struct sockaddr_in { sa_family_t sin_family; in_port_t sin_port; struct in_addr sin_addr; char sin_zero[8]; };
struct in6_addr { union { unsigned char s6_addr[16]; unsigned short s6_addr16[8]; } u6; };
#define s6_addr u6.s6_addr
struct sockaddr_in6 {
  sa_family_t sin6_family; in_port_t sin6_port; unsigned int sin6_flowinfo;
  struct in6_addr sin6_addr; unsigned int sin6_scope_id;
};
struct sockaddr_storage { sa_family_t ss_family; char ss_pad1[6]; long long ss_align; char ss_pad2[112]; };
struct linger { unsigned short l_onoff, l_linger; };
struct iovec { void *iov_base; size_t iov_len; };
extern const struct in6_addr llbw_in6addr_any, llbw_in6addr_loopback;
#define in6addr_any llbw_in6addr_any
#define in6addr_loopback llbw_in6addr_loopback
/* XPLAT-WIN: the BSD names are macros for llbw_* functions, because ws2_32 exports
 * the same names with SOCKET (64-bit) arguments: a function named socket() here
 * could not call Winsock's. */
#define socket llbw_socket
#define bind llbw_bind
#define listen llbw_listen
#define accept llbw_accept
#define accept4 llbw_accept4
#define connect llbw_connect
#define recv llbw_recv
#define send llbw_send
#define recvfrom llbw_recvfrom
#define sendto llbw_sendto
#define shutdown llbw_shutdown
#define setsockopt llbw_setsockopt
#define getsockopt llbw_getsockopt
#define getsockname llbw_getsockname
#define getpeername llbw_getpeername
#define socketpair llbw_socketpair
#define inet_pton llbw_inet_pton
#define inet_ntop llbw_inet_ntop
#define htons llbw_htons
#define ntohs llbw_ntohs
#define htonl llbw_htonl
#define ntohl llbw_ntohl
#define select llbw_select
#define poll llbw_poll
int socket(int domain, int type, int proto);
int bind(int fd, const struct sockaddr *a, socklen_t n);
int listen(int fd, int backlog);
int accept(int fd, struct sockaddr *a, socklen_t *n);
int accept4(int fd, struct sockaddr *a, socklen_t *n, int flags);
int connect(int fd, const struct sockaddr *a, socklen_t n);
ssize_t recv(int fd, void *buf, size_t n, int flags);
ssize_t send(int fd, const void *buf, size_t n, int flags);
ssize_t recvfrom(int fd, void *buf, size_t n, int flags, struct sockaddr *a, socklen_t *an);
ssize_t sendto(int fd, const void *buf, size_t n, int flags, const struct sockaddr *a, socklen_t an);
int shutdown(int fd, int how);
int setsockopt(int fd, int level, int name, const void *v, socklen_t n);
int getsockopt(int fd, int level, int name, void *v, socklen_t *n);
int getsockname(int fd, struct sockaddr *a, socklen_t *n);
int getpeername(int fd, struct sockaddr *a, socklen_t *n);
int llbw_socketpair(int domain, int type, int proto, int fds[2]);
int inet_pton(int af, const char *src, void *dst);
const char *inet_ntop(int af, const void *src, char *dst, socklen_t n);
unsigned short htons(unsigned short v);
unsigned short ntohs(unsigned short v);
unsigned int htonl(unsigned int v);
unsigned int ntohl(unsigned int v);
ssize_t readv(int fd, const struct iovec *v, int n);
ssize_t writev(int fd, const struct iovec *v, int n);

/* ---- poll / select -------------------------------------------------------- */
/* XPLAT-WIN: fd_set is the BSD bit array (the runtime builds one by hand and casts
 * it), not Winsock's counted list. poll and select take sockets (WSAPoll), pipes
 * (PeekNamedPipe) and other descriptors (always ready), and wait by polling the
 * pipes at 1 ms when a pipe is among them. */
#define POLLIN 0x0001
#define POLLPRI 0x0002
#define POLLOUT 0x0004
#define POLLERR 0x0008
#define POLLHUP 0x0010
#define POLLNVAL 0x0020
#define POLLRDNORM 0x0040
#define POLLWRNORM 0x0100
typedef unsigned long nfds_t;
struct pollfd { int fd; short events; short revents; };
int llbw_poll(struct pollfd *fds, nfds_t n, int timeout_ms);
#ifndef FD_SETSIZE
#define FD_SETSIZE (LLBW_SOCK_BASE + LLBW_SOCK_MAX)
#endif
typedef struct { unsigned long long fds_bits[FD_SETSIZE / 64]; } fd_set;
#define FD_ZERO(s) do { for (size_t llbw_i_ = 0; llbw_i_ < FD_SETSIZE / 64; llbw_i_++) (s)->fds_bits[llbw_i_] = 0; } while (0)
#define FD_SET(fd, s) ((s)->fds_bits[(fd) / 64] |= 1ull << ((fd) % 64))
#define FD_CLR(fd, s) ((s)->fds_bits[(fd) / 64] &= ~(1ull << ((fd) % 64)))
#define FD_ISSET(fd, s) (((s)->fds_bits[(fd) / 64] >> ((fd) % 64)) & 1)
struct timeval;
int llbw_select(int nfds, fd_set *r, fd_set *w, fd_set *x, struct timeval *tv);
struct timeval { long tv_sec; long tv_usec; };
#endif /* !LLBW_IMPL */

/* Windows' own SOCKET behind a shim descriptor (~0 if it is not one) */
uintptr_t llbw_socket_handle(int fd);

/* ---- stdio extensions (stdio.h additions) --------------------------------- */
/* XPLAT-WIN: the UCRT cannot make a FILE from callbacks, and its stdout cannot be
 * assigned. dual.c does both (stdout = a fopencookie stream that captures the
 * engine's id lines). So a cookie stream is a shim object that is NOT a UCRT FILE,
 * stdout is a shim variable, and the write side of stdio (printf, fprintf, fputs,
 * fwrite, fflush, fclose, setvbuf, ...) is routed through llbw_* functions that
 * call the cookie's write for a cookie stream and the UCRT for every other FILE.
 * A cookie stream is unbuffered: each call reaches the cookie at once, which is
 * when line buffering would deliver an id line too. */
typedef ssize_t (*llbw_cookie_read_t)(void *c, char *buf, size_t n);
typedef ssize_t (*llbw_cookie_write_t)(void *c, const char *buf, size_t n);
typedef int (*llbw_cookie_seek_t)(void *c, long long *off, int whence);
typedef int (*llbw_cookie_close_t)(void *c);
typedef struct {
  llbw_cookie_read_t read;
  llbw_cookie_write_t write;
  llbw_cookie_seek_t seek;
  llbw_cookie_close_t close;
} cookie_io_functions_t;
typedef cookie_io_functions_t llbw_cookie_fns;

#ifdef __cplusplus
}
#endif
#endif
