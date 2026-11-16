# backends/windows/compat: the Windows POSIX shim

An include directory, passed with `-I`, that lets the POSIX C this repo emits and
links compile and run on Windows without editing it:

- the Bend runtime preamble at the top of every emitted C file (`llb-engine.c`
  and the rest);
- `tools/sys1dual/dual.c`, `tools/metrics`, `tools/glue`, `tools/clef3`;
- `host/llb_gguf.c`, `host/llb_sampler.c`, `tools/llb/llb-tok.c`.

It does not touch the Bend compiler or its patch stack, and POSIX builds never
see it. All of it is Windows-only code (`XPLAT-WIN`).

Toolchain: clang for Windows (LLVM 23 was used, target `x86_64-pc-windows-msvc`)
with the UCRT and the Windows SDK. The headers rely on clang's `#include_next`.
MSVC `cl` cannot use them.

## How to build with it

```
clang -std=c11 -O2 -D_CRT_DECLARE_NONSTDC_NAMES=0 -c backends\windows\compat\llbw_posix.c -o llbw_posix.o
clang <flags> -D_CRT_DECLARE_NONSTDC_NAMES=0 -I backends\windows\compat -c some_posix_file.c -o some_posix_file.o
clang ... some_posix_file.o llbw_posix.o <clang resource dir>\lib\windows\clang_rt.builtins-x86_64.lib -lws2_32 -lwinmm -Wl,/STACK:8388608
```

- **`-D_CRT_DECLARE_NONSTDC_NAMES=0` is required.** Without it the UCRT declares
  its own "POSIX names": `int read(int, void *, unsigned)`, a 32-bit `off_t`, and
  a one-argument `mkdir`. Those clash with the POSIX forms. `llbw_posix.h` stops
  the build with `#error` if the define is missing.
- **`llbw_posix.c` is compiled without `-I compat`.** It needs the real UCRT
  stdio and `winsock2.h`.
- **`clang_rt.builtins`** provides `__extendhfsf2`, used for `_Float16` in
  `tools/clef3` and `tools/glue`.
- **`/STACK:8388608`** sets an 8 MiB main stack, glibc's default. Threads made
  through the shim also get 8 MiB.
- **Run `test_shim.c`, the self-test.** `backends/windows/build.ps1` builds and
  runs it on every build.

## Emitted Bend C also needs `winabi.ps1`

Shim headers alone are not enough for an emitted Bend program. The runtime's
work loop is a chain of segment functions (`WL_FN`) entered by `musttail`, and
each one takes `Env e` (two pointers) by value.

- **Win64 ABI:** any struct over 8 bytes is passed by a hidden pointer to a copy
  in the caller's frame.
- **`musttail`:** the tail call frees that frame, so the callee reads a dead
  copy.

On this machine, `kernels/argmax.bend` emitted and built with the shim died at
once. The CUDA-stack compiler's build said `bend: solo delivery lost`. The pinned
compiler's build said `bend: memory fault (machine stack overflow?)`, with
`Env.mem == NULL`.

`backends/windows/winabi.ps1 -In x.c -Out x.win.c` fixes this. It changes
exactly two lines, `WL_FN` and the `WlFn` table type, to `sysv_abi`, and refuses
to run if either line is not found exactly once. After it, both builds printed
the right answer (`777777`) on 1 and 16 threads.

**Apply it to `llb-engine.c` before compiling it on Windows.** It applies cleanly
to the CUDA emission of this tree.

## What each header gives

| Header | Real on Windows | Stubbed or limited (says so, never silent) |
|---|---|---|
| `pthread.h` | Threads (`_beginthreadex`, 8 MiB stacks), mutex (exclusive SRWLOCK), cond (CONDITION_VARIABLE, timed wait), rwlock, once (InitOnce), keys (TLS) | Recursive mutex type is refused (`ENOTSUP`); key destructors are not run |
| `sched.h` | `sched_yield` (SwitchToThread) | |
| `unistd.h`, `fcntl.h` | `open` (always `_O_BINARY`), `read`, `write`, `close`, `lseek` (64-bit), `pread`/`pwrite` (OVERLAPPED, at most 1 GiB a call), `pipe` (binary, not inherited), `dup`, `dup2`, `unlink`, `rmdir`, `access`, `mkdir(path, mode)`, `getcwd`, `chdir`, `getpid`, `getppid` (toolhelp), `sysconf` (CPU count, page size, physical pages), `readlink("/proc/self/exe")`, `fcntl` `F_GETFL`/`F_SETFL` `O_NONBLOCK`, `mkdtemp`, `mkstemp`, `ftruncate`, `fsync`, `sleep`, `usleep` | `fork`, `execv`, and `execvp` fail with `ENOSYS`. The runtime's `--workers N>1` is the only fork caller. `X_OK` counts as "exists". |
| `sys/mman.h` | `mmap` anon: a VirtualAlloc reservation; a vectored exception handler commits pages on first touch (MAP_NORESERVE). It honours an address hint, including the runtime's 2^45 corpus base and its adjacent growth. `mmap` of a file: CreateFileMapping + MapViewOfFile, read-only, copy-on-write, or shared. `munmap`, `mprotect` (a PROT_NONE guard becomes a committed no-access page), `madvise(MADV_DONTNEED)` (decommit; on a file view, VirtualUnlock), `mlock`, `msync` | `llbw_precommit(p, n)`: a kernel call (ReadFile, recv) into untouched reserved pages fails rather than faulting, so `read`, `pread`, `recv`, and `fread` commit their destination first |
| `signal.h` | `sigaction` for the signals the UCRT has (INT, TERM, SEGV, FPE, ILL, ABRT), through `signal()` | SIGHUP, SIGPIPE, SIGBUS, SIGCHLD, and the rest are accepted and ignored: nothing delivers them on Windows. `sigaltstack` and `sigprocmask` are no-ops. `kill` ends another process (TerminateProcess) or raises in this one. |
| `sys/wait.h` | The W* macros | `waitpid` fails with `ECHILD` (no fork) |
| `sys/socket.h`, `netinet/in.h`, `arpa/inet.h`, `sys/uio.h` | Winsock2 sockets behind int descriptors from 4096 up: `socket`, `bind`, `listen`, `accept`/`accept4`, `connect`, `recv`/`send` (MSG_DONTWAIT), `sendto`/`recvfrom`, `setsockopt` (SO_RCVTIMEO/SNDTIMEO converted from timeval), `getsockopt`, `getsockname`, `getpeername`, `shutdown`, `socketpair` (loopback TCP), `inet_pton`/`ntop`, `hton*`, `readv`/`writev`. `read`, `write`, and `close` work on them. | The BSD names are macros (`#define socket llbw_socket`), because ws2_32 exports the same names with 64-bit SOCKET arguments |
| `poll.h`, `sys/select.h` | `poll` and `select` over sockets (WSAPoll), pipes (PeekNamedPipe), and other descriptors (always ready). `fd_set` is the BSD bit array: the runtime builds one by hand and casts it. | A wait with a pipe in it polls at 1 ms (`timeBeginPeriod(1)`). There is no wakeable pipe wait on Windows. |
| `time.h`, `sys/time.h` | `clock_gettime` (MONOTONIC: QPC; REALTIME: precise system time; process and thread CPU time), `nanosleep` (high-resolution waitable timer), `gettimeofday`, `gmtime_r`, `localtime_r` | |
| `sys/stat.h`, `sys/types.h` | `struct stat` / `stat` / `fstat` over `_stat64` (64-bit `st_size`) with POSIX `st_mtim`-style timespecs; `off_t` is 64-bit; `ssize_t`, `pid_t`, `mode_t` | `st_ino` is 0 and the timestamps have whole seconds, as the UCRT reports them |
| `stdio.h` | `fopencookie`, an assignable `stdout` (dual.c does `stdout = cap`), `getline`/`getdelim`, `open_memstream`, `fseeko`/`ftello` (64-bit). Every write-side call (`printf`, `fprintf`, `fputs`, `puts`, `putchar`, `fwrite`, `fflush`, `fclose`, `setvbuf`, `ferror`, `fileno`) goes through `llbw_*`, which calls a cookie's write for a cookie stream and the UCRT for any other FILE. | A cookie stream is unbuffered and write-only. `_IO_read_ptr`/`_IO_read_end` (glibc FILE internals, read by `tools/metrics/tokstream.c`'s queue-depth metric) map to the UCRT's one opaque field, so that metric reads 0. |
| `stdlib.h`, `string.h`, `strings.h` | `setenv`/`unsetenv` (`_putenv_s`), `realpath`, `strndup`, `memmem`, `stpcpy`, `strerror_r`, `strdup`, `strtok_r`, `strcasecmp` | An empty `setenv` value cannot be stored in the CRT's copy, so `getenv` returns NULL for it |
| `openblas/cblas.h` | `cblas_sgemm`, plain C (`backends/windows/cblas_sgemm.c`), for the CLEF decide head | Sums in its own order, so a head's probabilities can differ from OpenBLAS's in the last bits |

### What happens at start-up

These run once, in a constructor in `llbw_posix.c`:

- `_set_fmode(_O_BINARY)`, and binary mode on stdin, stdout, and stderr. POSIX
  has no text mode, and llb-serve reads the engine's stdout byte for byte.
- A no-op invalid-parameter handler. A bad descriptor then returns `EBADF`, as
  POSIX does, instead of ending the process.
- `WSAStartup`.
- The first-touch exception handler.
- `timeBeginPeriod(1)`.

`LLBW_FAULT_TRACE=1` prints one line for each access violation the first-touch
handler does not take: the address, the PC, and the module offset to feed to
`llvm-symbolizer`.

## Proven on this machine (2026-11-16)

- `test_shim.exe`: **27/27** checks pass. They cover:
  - an 8 GiB NORESERVE map at 2^45, first-touch commit, and adjacent growth;
  - `pread` into untouched pages;
  - PROT_NONE guard;
  - threads with a mutex and condition variable;
  - a non-blocking pipe woken through `select`, and `poll` on a pipe;
  - a 6 MiB recursion on a shim thread;
  - socketpair `send`/`recv`;
  - `printf`, `puts`, and `fputs` into a cookie through an assigned `stdout`;
  - binary `getline`, `setenv`, and `nanosleep`/`CLOCK_MONOTONIC`.
- `engine_cpu.exe`, `llb-tok.exe`, and the request objects (`dual.c`, `metrics`,
  `glue_out`, `clef3`, `llb_sampler`, `llb_gguf`) compile against the shim
  unchanged and run. Numbers are in `WINDOWS-TEST-STATUS.md`.
- An emitted Bend program runs through the shim plus `winabi.ps1`:
  `kernels/argmax.bend` reads a 4 MiB file through the IO effects, runs on the
  CPU pool, and prints the right index on 1 and 16 threads. This was checked with
  both the pinned compiler and the CUDA-stack compiler.
- The emitted `llb-engine.c` (CUDA stack) without `BEND_CUDA` compiles against the
  shim except for one thing: `llb_hidden_*` from the CUDA-only host chain. That is
  expected; see `backends/windows/BUILD.md`.

**Not proven:** a CUDA build (`BEND_CUDA=1`) of the emitted engine against the
shim. The CUDA toolkit was not installed when this was written.
