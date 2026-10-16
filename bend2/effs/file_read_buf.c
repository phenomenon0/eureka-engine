// File
// ====

#include <sys/uio.h>

// File.read_bytes into a Bytes: one packed block, a byte a cell.
static void file_read_buf_call(IoWork* w) {
  int fd = (int)w->hand;
  w->size = io_sys_end(w, read(fd, w->data, w->word));
}

static Term file_read_buf_pack(Env e, IoWork* w) {
  Term r = w->code ? io_fail(e, w->code, NULL)
    : io_done(e, io_buf(e, w->data, w->size));
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

// What the page cache already holds is read on the loop, straight into
// the block the Bytes will be (preadv2 with RWF_NOWAIT, at the file's
// position, which it moves as read does); a read that would wait on the
// disk goes to a helper as before. A short count is what read may give.
static bool file_read_buf_now(Env e, IoWork* w, Term* out) {
#if defined(__linux__) && defined(RWF_NOWAIT)
  if (w->word == 0) { return false; }
  StrParts p = str_alloc(e, w->word, 2);
  if (!p.data || err_seen(e.mem)) { term_sink(e, p.data); return false; }
  struct iovec v = { (u8*)(e.mem + term_peek(e, p.data)), w->word };
  ssize_t n = preadv2((int)w->hand, &v, 1, -1, RWF_NOWAIT);
  if (n < 0 && (errno == EAGAIN || errno == EOPNOTSUPP || errno == ENOSYS)) {
    term_sink(e, p.data);
    return false;
  }
  if (n < 0) {
    u32 code = (u32)errno;
    term_sink(e, p.data);
    *out = io_tup(e, io_hand(w->hand), io_fail(e, code, NULL));
    return true;
  }
  p.len = (u32)n;
  *out = io_tup(e, io_hand(w->hand), io_done(e, str_view_owned(e, p)));
  return true;
#else
  (void)e; (void)w; (void)out;
  return false;
#endif
}

Term file_read_buf_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->word = f[1] < INT32_MAX ? f[1] : INT32_MAX;
  Term out;
  if (file_read_buf_now(e, w, &out)) {
    return out;
  }
  w->data = io_mem(malloc(w->word + 1));
  return io_work(w, file_read_buf_call, file_read_buf_pack);
}

static void __attribute__((constructor)) file_read_buf_use(void) {
  io_eff(CID_FILE_READ_BUF, file_read_buf_run, 0);
}
