// File
// ====

#include <sys/stat.h>
#include <sys/uio.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

// path, opened for reading one component at a time below root: every
// step is an openat with O_NOFOLLOW, so a symbolic link anywhere under
// root, to a file or to a directory, fails (ELOOP) rather than being
// followed out of it. A component that is empty, "." or ".." fails with
// EACCES, and so does an end that is not a regular file (EISDIR for a
// directory); O_NONBLOCK keeps a FIFO from parking the helper thread.
// root itself is opened as given: it is the operator's, not the peer's.
static int file_open_under_walk(const char* root, char* at) {
  int dir = open(root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  while (dir >= 0) {
    char* cut  = strchr(at, '/');
    int   last = cut == NULL;
    if (!last) {
      *cut = 0;
    }
    if (*at == 0 || strcmp(at, ".") == 0 || strcmp(at, "..") == 0) {
      close(dir);
      errno = EACCES;
      return -1;
    }
    int fd = openat(dir, at, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY
      | (last ? O_NONBLOCK : O_DIRECTORY));
    int no = errno;
    close(dir);
    errno = no;
    if (fd < 0 || !last) {
      dir = fd;
      at  = cut + 1;
      continue;
    }
    return fd;
  }
  return -1;
}

// Linux 5.6 and later resolve the whole path in one openat2 under
// RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS: no symbolic link anywhere, no
// escape from root. The walk's refusals by shape (an empty, "." or ".."
// component: EACCES) are made first, so both ways refuse the same names
// with the same codes. A kernel or a sandbox without it (ENOSYS, EPERM)
// falls back to the walk, for good.
#if defined(__linux__) && defined(SYS_openat2)
static int file_open_under_no2;

static bool file_open_under_shape(const char* p) {
  for (;;) {
    const char* c = strchr(p, '/');
    size_t      n = c ? (size_t)(c - p) : strlen(p);
    if (n == 0 || (p[0] == '.' && (n == 1 || (n == 2 && p[1] == '.')))) {
      return false;
    }
    if (!c) {
      return true;
    }
    p = c + 1;
  }
}

// The root's descriptor is kept, per thread, for up to a second: a root
// renamed or swapped (a deploy that repoints a link) is seen within it.
typedef struct {
  char* root;
  int   fd;
  u64   until;
} FileTop;

static __thread FileTop file_open_under_top = { NULL, -1, 0 };

static int file_open_under_dir(const char* root) {
  FileTop* t   = &file_open_under_top;
  u64      now = io_tick();
  if (t->fd >= 0 && now < t->until && strcmp(t->root, root) == 0) {
    return t->fd;
  }
  if (t->fd >= 0) {
    close(t->fd);
    free(t->root);
    t->fd = -1;
  }
  int dir = open(root, O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (dir >= 0) {
    t->root  = io_mem(strdup(root));
    t->fd    = dir;
    t->until = now + 1000000000ull;
  }
  return dir;
}

static int file_open_under_two(const char* root, const char* at) {
  if (!file_open_under_shape(at)) {
    errno = EACCES;
    return -1;
  }
  int dir = file_open_under_dir(root);
  if (dir < 0) {
    return -1;
  }
  struct { u64 flags, mode, resolve; } how = {
    O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY | O_NONBLOCK, 0,
    0x08 | 0x04 | 0x02 };  // RESOLVE_BENEATH, NO_SYMLINKS, NO_MAGICLINKS
  return (int)syscall(SYS_openat2, dir, at, &how, sizeof how);
}
#endif

// the end, opened, must be a regular file; st is its fstat
static int file_open_under_end(int fd, struct stat* st) {
  if (fd >= 0 && (fstat(fd, st) != 0 || !S_ISREG(st->st_mode))) {
    int no = S_ISDIR(st->st_mode) ? EISDIR : EACCES;
    close(fd);
    errno = no;
    return -1;
  }
  return fd;
}

static int file_open_under_stat(const char* root, char* at, struct stat* st) {
#if defined(__linux__) && defined(SYS_openat2)
  if (!file_open_under_no2) {
    int fd = file_open_under_two(root, at);
    if (fd >= 0 || (errno != ENOSYS && errno != EPERM)) {
      return file_open_under_end(fd, st);
    }
    file_open_under_no2 = 1;
  }
#endif
  return file_open_under_end(file_open_under_walk(root, at), st);
}

static int file_open_under_fd(const char* root, char* at) {
  struct stat st;
  int fd = file_open_under_stat(root, at, &st);
  if (fd >= 0) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
  }
  return fd;
}

#ifdef CID_FILE_OPEN_UNDER

static void file_open_under_call(IoWork* w) {
  w->made = (intptr_t)io_sys_end(w, file_open_under_fd(w->text, w->data));
}

static Term file_open_under_pack(Env e, IoWork* w) {
  free(w->data);
  free(w->text);
  return w->code != 0 ? io_fail(e, w->code, NULL)
    : io_done(e, io_hand(w->made));
}

Term file_open_under_run(Env e, Term* f, IoWork* w) {
  uint64_t rn = 0;
  w->text = io_cstr(e, f[0], &rn);
  w->data = io_cstr(e, f[1], &w->size);
  w->code = 0;
  if (io_nul(w->text, rn) || io_nul(w->data, w->size)) {
    w->code = EILSEQ;
    return file_open_under_pack(e, w);
  }
  // on the loop: a walk of openat and one fstat, which the dentry cache
  // answers, costs less than a helper's round trip
  return io_now(e, w, file_open_under_call, file_open_under_pack);
}

static void __attribute__((constructor)) file_open_under_use(void) {
  io_eff(CID_FILE_OPEN_UNDER, file_open_under_run, 0);
}

#endif

#ifdef CID_FILE_GET_UNDER

// File.get_under(root, path, small): File.open_under and its fstat in
// one effect, and, for a file under small bytes, its read and close too:
// Done{(size, (mtime, (None, body)))}, or Done{(size, (mtime, (Some{file},
// "")))} with the file open at 0 for a larger one -- or a small one whose
// bytes the page cache does not hold, which is then read as any other.
// The body is read straight into the block the Bytes will be; a file
// that shrank since its fstat comes back shorter than its size.
static Term file_get_under_body(Env e, int fd, u64 n, bool* held) {
  StrParts p = str_alloc(e, n, 2);
  *held = true;
  if (!n || !p.data || err_seen(e.mem)) {
    return str_view_owned(e, p);
  }
  u8* at  = (u8*)(e.mem + term_peek(e, p.data));
  u64 got = 0;
  while (got < n) {
    struct iovec v = { at + got, n - got };
#if defined(__linux__) && defined(RWF_NOWAIT)
    ssize_t k = preadv2(fd, &v, 1, (off_t)got, got == 0 ? RWF_NOWAIT : 0);
#else
    ssize_t k = preadv(fd, &v, 1, (off_t)got);
#endif
    if (k < 0 && errno == EINTR) {
      continue;
    }
    if (k < 0 && got == 0) {
      *held = false;
      p.len = 0;
      break;
    }
    if (k <= 0) {
      break;
    }
    got += (u64)k;
  }
  p.len = *held ? (u32)got : 0;
  return str_view_owned(e, p);
}

// The memo: what get_under last read of a file under small bytes, per
// thread, by root and path, answered again without a syscall for up to
// a second after it was read (as nginx's open_file_cache answers within
// its valid time). Past that the file is opened and fstat'ed again, and
// its bytes are kept when its inode, size and mtime are what they were,
// read again when not. A slot holds one file; a new one takes the slot
// its name hashes to, so the memo never holds more than FILE_MEMO_N
// files, each under small bytes. Only a regular file that opened is
// kept: a refusal is asked of the file system every time.
#define FILE_MEMO_N  256
#define FILE_MEMO_NS 1000000000ull

typedef struct {
  char* key;
  u64   kn;
  u64   at;
  u64   dev, ino, mns;
  u32   size, mtime;
  char* body;
} FileMemo;

static __thread FileMemo* file_memo;

static FileMemo* file_memo_slot(const char* key, u64 kn) {
  if (file_memo == NULL) {
    file_memo = io_mem(calloc(FILE_MEMO_N, sizeof(FileMemo)));
  }
  u64 h = 1469598103934665603ull;
  for (u64 i = 0; i < kn; i++) {
    h = (h ^ (u8)key[i]) * 1099511628211ull;
  }
  return &file_memo[h & (FILE_MEMO_N - 1)];
}

static bool file_memo_is(FileMemo* m, const char* key, u64 kn) {
  return m->key != NULL && m->kn == kn && memcmp(m->key, key, kn) == 0;
}

static u64 file_memo_mns(struct stat* st) {
#if defined(__APPLE__)
  return (u64)st->st_mtimespec.tv_sec * 1000000000ull + (u64)st->st_mtimespec.tv_nsec;
#else
  return (u64)st->st_mtim.tv_sec * 1000000000ull + (u64)st->st_mtim.tv_nsec;
#endif
}

static bool file_memo_same(FileMemo* m, struct stat* st) {
  return m->dev == (u64)st->st_dev && m->ino == (u64)st->st_ino
    && m->size == (u64)st->st_size && m->mns == file_memo_mns(st);
}

static void file_memo_put(FileMemo* m, const char* key, u64 kn, struct stat* st,
  const char* body, u64 at) {
  free(m->key);
  free(m->body);
  m->key  = io_mem(malloc(kn));
  m->body = io_mem(malloc(st->st_size + 1));
  memcpy(m->key, key, kn);
  memcpy(m->body, body, (size_t)st->st_size);
  m->kn    = kn;
  m->at    = at;
  m->dev   = (u64)st->st_dev;
  m->ino   = (u64)st->st_ino;
  m->mns   = file_memo_mns(st);
  m->size  = (u32)st->st_size;
  m->mtime = (u32)st->st_mtime;
}

static Term file_get_under_done(Env e, u32 n, u32 mtime, Term file, Term body) {
  return io_done(e, io_tup(e, n, io_tup(e, mtime, io_tup(e, file, body))));
}

// root and path as the key, "root\0path\0", which is both C strings;
// NULL for a byte past 255 or a NUL in either
static char* file_get_under_key(Env e, Term root, Term path, u64* rn, u64* kn) {
  u32         an = 0, bn = 0;
  char*       ao = NULL;
  char*       bo = NULL;
  const char* a  = io_buf_ptr(e, root, &an, &ao);
  const char* b  = io_buf_ptr(e, path, &bn, &bo);
  char*       k  = NULL;
  if (a && b && !memchr(a, 0, an) && !memchr(b, 0, bn)) {
    k = io_mem(malloc((u64)an + bn + 2));
    memcpy(k, a, an);
    k[an] = 0;
    memcpy(k + an + 1, b, bn);
    k[an + 1 + bn] = 0;
  }
  free(ao);
  free(bo);
  term_sink(e, root);
  term_sink(e, path);
  *rn = an;
  *kn = (u64)an + 1 + bn;
  return k;
}

Term file_get_under_run(Env e, Term* f, IoWork* w) {
  u64   rn = 0, kn = 0;
  char* key = file_get_under_key(e, f[0], f[1], &rn, &kn);
  u64   small = (u64)(u32)f[2];
  (void)w;
  if (key == NULL) {
    return io_fail(e, EILSEQ, NULL);
  }
  char* root = key;
  char* path = key + rn + 1;
  u64       now = io_tick();
  FileMemo* m   = file_memo_slot(key, kn);
  bool      is  = file_memo_is(m, key, kn);
  if (is && now - m->at < FILE_MEMO_NS && m->size < small) {
    free(key);
    return file_get_under_done(e, m->size, m->mtime, term_pak(CID_NONE, 0),
      io_buf(e, m->body, m->size));
  }
  struct stat st;
  char* at = io_mem(strdup(path));
  int   fd = file_open_under_stat(root, at, &st);
  int   no = errno;
  free(at);
  if (fd >= 0 && st.st_size > (off_t)UINT32_MAX) {
    close(fd);
    fd = -1;
    no = EOVERFLOW;
  }
  if (fd < 0) {
    free(key);
    return io_fail(e, (u32)no, NULL);
  }
  u64 n = (u64)st.st_size;
  if (n < small && is && file_memo_same(m, &st)) {
    close(fd);
    free(key);
    m->at = now;
    return file_get_under_done(e, m->size, m->mtime, term_pak(CID_NONE, 0),
      io_buf(e, m->body, m->size));
  }
  bool held = false;
  Term body = n < small ? file_get_under_body(e, fd, n, &held)
    : term_pak(CID_SNIL, 0);
  Term file = term_pak(CID_NONE, 0);
  if (held) {
    close(fd);
    StrParts p = str_peek(e, body);
    if (p.len == n && (n == 0 || str_nar(p) == 2)) {
      file_memo_put(m, key, kn, &st,
        n ? (const char*)(e.mem + term_peek(e, p.data)) + p.off : "", now);
    }
  } else {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
    file = io_box(e, CID_SOME, io_hand(fd));
  }
  free(key);
  return file_get_under_done(e, (u32)n, (u32)st.st_mtime, file, body);
}

static void __attribute__((constructor)) file_get_under_use(void) {
  io_eff(CID_FILE_GET_UNDER, file_get_under_run, 0);
}

#endif
