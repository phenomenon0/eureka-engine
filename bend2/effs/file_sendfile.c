// File
// ====

// File.sendfile(sock, file, off, len, ms): the file's bytes from off, len
// of them, onto the socket, under TCP.send_buf_poll's deadline: a full
// socket parks on the socket and on the clock, and a wait of ms with no
// byte moved fails with ETIMEDOUT. Done once all len are out; a file
// that ends first fails with ENODATA, after what it had went out. The
// file's position does not move.
//
// On a plain socket the kernel copies from the page cache to the socket
// (sendfile(2) on Linux and macOS) and no byte passes through the
// process. Under a wire (TLS) the bytes must, so they are read in blocks
// with pread and written through the wire, one block held at a time; a
// file sendfile(2) refuses (EINVAL, ENOSYS, EOPNOTSUPP) is sent that way
// too.
#include <sys/types.h>
#include <sys/uio.h>
#ifndef ENODATA
#define ENODATA EIO
#endif
#if defined(__linux__)
#include <sys/sendfile.h>
#endif

#define FILE_SENDFILE_BLOCK 65536

// FileSend ::= FileSend(file, off, end, ms, slow, buf, have, at)
typedef struct {
  int   file;
  u64   off;
  u64   end;
  u64   ms;
  bool  slow;
  char* buf;
  u32   have;
  u32   at;
} FileSend;

static Term file_sendfile_pack(Env e, IoWork* w) {
  FileSend* fs = (FileSend*)w->data;
  Term r = w->code != 0 ? io_fail(e, w->code, NULL)
    : io_done(e, term_pak(CID_UNIT, 0));
  int  file = fs->file;
  free(fs->buf);
  free(fs);
  return io_tup(e, io_hand(w->hand), io_tup(e, io_hand(file), r));
}

// one move: bytes moved (> 0), 0 at the end of the file, -1 with errno
// (EAGAIN: wait on *dir)
static ssize_t file_sendfile_slow(int fd, FileSend* fs, short* dir) {
  if (fs->at == fs->have) {
    u64 want = fs->end - fs->off;
    want = want < FILE_SENDFILE_BLOCK ? want : FILE_SENDFILE_BLOCK;
    if (fs->buf == NULL) {
      fs->buf = io_mem(malloc(FILE_SENDFILE_BLOCK));
    }
    ssize_t k = pread(fs->file, fs->buf, want, (off_t)fs->off);
    if (k <= 0) {
      return k;
    }
    fs->have = (u32)k;
    fs->at   = 0;
  }
  ssize_t n = io_wire_write(fd, fs->buf + fs->at, fs->have - fs->at, dir);
  if (n > 0) {
    fs->at += (u32)n;
  }
  return n;
}

static ssize_t file_sendfile_fast(int fd, FileSend* fs, short* dir) {
  *dir = POLLOUT;
  u64 want = fs->end - fs->off;
  want = want < (1u << 30) ? want : (1u << 30);
#if defined(__linux__)
  off_t o = (off_t)fs->off;
  return sendfile(fd, fs->file, &o, (size_t)want);
#elif defined(__APPLE__)
  off_t len = (off_t)want;
  int   r   = sendfile(fs->file, fd, (off_t)fs->off, &len, NULL, 0);
  if (r < 0 && len > 0) {
    return (ssize_t)len;
  }
  return r < 0 ? -1 : (ssize_t)len;
#else
  (void)fd;
  errno = ENOSYS;
  return -1;
#endif
}

static Term file_sendfile_more(Env e, IoWork* w);

// at is the deadline of the wait in progress, 0 when none is
static Term file_sendfile_at(Env e, IoWork* w, u64 at) {
  FileSend* fs  = (FileSend*)w->data;
  int       fd  = (int)w->hand;
  short     dir = POLLOUT;
  while (w->code == 0 && fs->off < fs->end) {
    ssize_t n = fs->slow ? file_sendfile_slow(fd, fs, &dir)
      : file_sendfile_fast(fd, fs, &dir);
    if (n < 0 && !fs->slow
        && (errno == EINVAL || errno == ENOSYS || errno == EOPNOTSUPP)) {
      fs->slow = true;
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && errno == EAGAIN) {
      u64 now = io_tick();
      at = at != 0 ? at : now + fs->ms * 1000000ull;
      if (now >= at) {
        w->code = ETIMEDOUT;
        break;
      }
      return io_wait_on(w, fd, dir, at, file_sendfile_more);
    }
    if (n == 0) {
      w->code = ENODATA;
      break;
    }
    at = 0;
    u64 k = io_sys_end(w, n);
    fs->off += k;
    w->made += (intptr_t)k;
  }
  return file_sendfile_pack(e, w);
}

static Term file_sendfile_more(Env e, IoWork* w) {
  return file_sendfile_at(e, w, io_wait_time(w));
}

Term file_sendfile_run(Env e, Term* f, IoWork* w) {
  FileSend* fs = io_mem(calloc(1, sizeof(FileSend)));
  fs->file = (int)io_hand_v(f[1]);
  fs->off  = (u64)(u32)f[2];
  fs->end  = fs->off + (u64)(u32)f[3];
  fs->ms   = (u64)(u32)f[4];
  fs->slow = io_wire != NULL;
  w->hand  = (intptr_t)io_hand_v(f[0]);
  w->data  = (char*)fs;
  w->made  = 0;
  w->code  = 0;
  return file_sendfile_at(e, w, 0);
}

static void __attribute__((constructor)) file_sendfile_use(void) {
  io_eff(CID_FILE_SENDFILE, file_sendfile_run, 0);
}
