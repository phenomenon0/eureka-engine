// tools/llb_platform.h — the file, mapping and clock calls of the two CLI sources
// (qwen35_engine.c, gguf2pack.c), on POSIX and on Windows.
//
// POSIX: each wrapper is the call the sources made before this header, with the
// same arguments; behaviour and output are unchanged (receipt: WINDOWS-TEST-STATUS.md).
// Windows (_WIN32): CRT _open/_close/_fstat64, ReadFile/WriteFile at an offset for
// pread/pwrite, CreateFileMappingW + MapViewOfFile for the pack. If the mapping
// fails, llb_map_read reads the file into RAM and returns 0, and the caller skips
// its per-layer page drop. UNTESTED ON WINDOWS as of the commit that adds it.
//
// Define LLB_PLATFORM_NO_MAP before including to leave the mapping calls out
// (gguf2pack: it never maps, and its _POSIX_C_SOURCE hides madvise on glibc).

#ifndef LLB_PLATFORM_H
#define LLB_PLATFORM_H

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#include <direct.h>
#include <io.h>

// Win32 error -> errno, so the callers' strerror(errno) messages stay meaningful
static inline void llb_set_errno_win(void) {
  DWORD e = GetLastError();
  errno = (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) ? ENOENT
        : e == ERROR_ACCESS_DENIED ? EACCES
        : (e == ERROR_DISK_FULL || e == ERROR_HANDLE_DISK_FULL) ? ENOSPC
        : e == ERROR_NOT_ENOUGH_MEMORY || e == ERROR_COMMITMENT_LIMIT ? ENOMEM
        : EIO;
}

static inline int llb_open_read(const char *path) { return _open(path, _O_RDONLY | _O_BINARY); }

// create for writing; excl: fail if it exists (O_EXCL), else truncate (O_TRUNC)
static inline int llb_open_write(const char *path, int excl) {
  return _open(path, _O_WRONLY | _O_CREAT | _O_BINARY | (excl ? _O_EXCL : _O_TRUNC), _S_IREAD | _S_IWRITE);
}

static inline int llb_close(int fd) { return _close(fd); }

// one positional read/write of at most 1 GiB (DWORD); callers loop on short counts.
// Returns bytes moved, 0 at end of file, -1 on error (errno set).
static inline int64_t llb_pread(int fd, void *buf, size_t n, uint64_t off) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
  OVERLAPPED ov;
  memset(&ov, 0, sizeof ov);
  ov.Offset = (DWORD)off, ov.OffsetHigh = (DWORD)(off >> 32);
  DWORD got = 0, want = n > (1u << 30) ? (1u << 30) : (DWORD)n;
  if (!ReadFile(h, buf, want, &got, &ov)) {
    if (GetLastError() == ERROR_HANDLE_EOF) return 0;
    llb_set_errno_win();
    return -1;
  }
  return (int64_t)got;
}

static inline int64_t llb_pwrite(int fd, const void *buf, size_t n, uint64_t off) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
  OVERLAPPED ov;
  memset(&ov, 0, sizeof ov);
  ov.Offset = (DWORD)off, ov.OffsetHigh = (DWORD)(off >> 32);
  DWORD put = 0, want = n > (1u << 30) ? (1u << 30) : (DWORD)n;
  if (!WriteFile(h, buf, want, &put, &ov)) { llb_set_errno_win(); return -1; }
  return (int64_t)put;
}

static inline int64_t llb_fdsize(int fd) {
  struct _stat64 st;
  if (_fstat64(fd, &st)) return -1;
  return (int64_t)st.st_size;
}

static inline int64_t llb_filesize(const char *path) {
  struct _stat64 st;
  if (_stat64(path, &st)) return -1;
  return (int64_t)st.st_size;
}

// NTFS: no sparse file unless FSCTL_SET_SPARSE is set, which this does not do;
// a --head-only pack.bin is full size on disk, zero-filled holes.
static inline int llb_ftruncate(int fd, uint64_t size) {
  errno_t e = _chsize_s(fd, (__int64)size);
  if (e) { errno = e; return -1; }
  return 0;
}

static inline int llb_mkdir(const char *path) { return _mkdir(path); }
static inline int llb_exists(const char *path) { return _access(path, 0) == 0; }

// perror with the Win32 error too (CRT calls set errno, Win32 calls GetLastError)
static inline void llb_perror(const char *what) {
  DWORD e = GetLastError();
  int en = errno;
  fprintf(stderr, "%s: %s (errno %d, GetLastError %lu)\n", what, strerror(en), en, (unsigned long)e);
}

static inline int llb_clock_monotonic(struct timespec *ts) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  ts->tv_sec = (time_t)(c.QuadPart / f.QuadPart);
  ts->tv_nsec = (long)((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
  return 0;
}

#ifndef LLB_PLATFORM_NO_MAP
// Map size bytes of fd read-only. Returns 1 (mapped: llb_drop_pages / llb_unmap
// apply), 0 (mapping failed, the file was read into malloc'd RAM: free() it, no
// page drop), -1 (both failed; errno set).
static inline int llb_map_read(int fd, uint64_t size, const uint8_t **out) {
  HANDLE h = (HANDLE)_get_osfhandle(fd);
  if (h != INVALID_HANDLE_VALUE && size > 0) {
    HANDLE m = CreateFileMappingW(h, NULL, PAGE_READONLY, (DWORD)(size >> 32), (DWORD)size, NULL);
    if (m) {
      void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, (SIZE_T)size);
      CloseHandle(m);  // the view keeps the section alive
      if (p) { *out = (const uint8_t *)p; return 1; }
    }
  }
  fprintf(stderr, "map pack failed (GetLastError %lu); reading it into RAM, per-layer page drop off\n",
          (unsigned long)GetLastError());
  uint8_t *buf = (uint8_t *)malloc(size ? (size_t)size : 1);
  if (!buf) { errno = ENOMEM; return -1; }
  uint64_t off = 0;
  while (off < size) {
    int64_t got = llb_pread(fd, buf + off, (size_t)(size - off), off);
    if (got <= 0) { free(buf); if (got == 0) errno = EIO; return -1; }
    off += (uint64_t)got;
  }
  *out = buf;
  return 0;
}

// Drop the mapped pages from the working set (POSIX: madvise MADV_DONTNEED).
// VirtualUnlock on an unlocked range "releases the pages from the process's
// working set" (MSDN, VirtualUnlock remarks) and then reports ERROR_NOT_LOCKED;
// the clean file-backed pages stay in the standby list, as Linux keeps them in
// the page cache. The return value is ignored, as the POSIX caller ignores madvise's.
static inline void llb_drop_pages(const void *p, size_t size) { VirtualUnlock((LPVOID)p, size); }

static inline int llb_unmap(const void *p, size_t size) { (void)size; return UnmapViewOfFile(p) ? 0 : -1; }
#endif

#else  // POSIX

#include <unistd.h>
#ifndef LLB_PLATFORM_NO_MAP
#include <sys/mman.h>
#endif

static inline int llb_open_read(const char *path) { return open(path, O_RDONLY); }
static inline int llb_open_write(const char *path, int excl) {
  return open(path, O_WRONLY | O_CREAT | (excl ? O_EXCL : O_TRUNC), 0644);
}
static inline int llb_close(int fd) { return close(fd); }
static inline int64_t llb_pread(int fd, void *buf, size_t n, uint64_t off) { return pread(fd, buf, n, (off_t)off); }
static inline int64_t llb_pwrite(int fd, const void *buf, size_t n, uint64_t off) { return pwrite(fd, buf, n, (off_t)off); }
static inline int64_t llb_fdsize(int fd) {
  struct stat st;
  if (fstat(fd, &st)) return -1;
  return (int64_t)st.st_size;
}
static inline int64_t llb_filesize(const char *path) {
  struct stat st;
  if (stat(path, &st)) return -1;
  return (int64_t)st.st_size;
}
static inline int llb_ftruncate(int fd, uint64_t size) { return ftruncate(fd, (off_t)size); }
static inline int llb_mkdir(const char *path) { return mkdir(path, 0755); }
static inline int llb_exists(const char *path) { return !access(path, F_OK); }
static inline void llb_perror(const char *what) { perror(what); }
static inline int llb_clock_monotonic(struct timespec *ts) { return clock_gettime(CLOCK_MONOTONIC, ts); }

#ifndef LLB_PLATFORM_NO_MAP
// POSIX never falls back: a failed mmap returns -1, as the engine's mmap path did.
static inline int llb_map_read(int fd, uint64_t size, const uint8_t **out) {
  void *p = mmap(NULL, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) return -1;
  *out = (const uint8_t *)p;
  return 1;
}
static inline void llb_drop_pages(const void *p, size_t size) { madvise((void *)p, size, MADV_DONTNEED); }
static inline int llb_unmap(const void *p, size_t size) { return munmap((void *)p, size); }
#endif

#endif  // _WIN32
#endif  // LLB_PLATFORM_H
