// TCP
// ===

// TCP.send_bytes from a Bytes. A block of 1-byte cells goes to the
// socket as it lies in the heap; only what a full socket leaves unsent
// is copied, to wait for the park. A Char past 255 is not a byte: EINVAL
// before anything goes out.
static Term tcp_send_buf_pack(Env e, IoWork* w) {
  Term r = w->code != 0 ? io_fail(e, w->code, NULL)
    : io_done(e, term_pak(CID_UNIT, 0));
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

// Sends what is left of w->data; EAGAIN parks until writable.
static Term tcp_send_buf_more(Env e, IoWork* w) {
  int   fd  = (int)w->hand;
  short dir = POLLOUT;
  while (w->code == 0 && (u64)w->made < w->size) {
    ssize_t n = io_wire_write(fd, w->data + w->made, w->size - (u64)w->made, &dir);
    if (n < 0 && errno == EAGAIN) {
      return io_wait_on(w, fd, dir, 0, tcp_send_buf_more);
    }
    w->made += io_sys_end(w, n);
  }
  return tcp_send_buf_pack(e, w);
}

Term tcp_send_buf_run(Env e, Term* f, IoWork* w) {
  int   fd  = (int)io_hand_v(f[0]);
  short dir = POLLOUT;
  u32   len = 0;
  char* own = NULL;
  const char* p = io_buf_ptr(e, f[1], &len, &own);
  w->hand = (intptr_t)fd;
  w->code = p == NULL ? EINVAL : 0;
  w->data = NULL;
  u64 at = 0;
  while (w->code == 0 && at < len) {
    ssize_t n = io_wire_write(fd, p + at, len - at, &dir);
    if (n < 0 && errno == EAGAIN) {
      break;
    }
    at += io_sys_end(w, n);
  }
  if (w->code == 0 && at < len) {
    w->data = io_mem(malloc(len - at));
    memcpy(w->data, p + at, len - at);
  }
  free(own);
  term_sink(e, f[1]);
  if (w->data == NULL) {
    return tcp_send_buf_pack(e, w);
  }
  w->size = len - at;
  w->made = 0;
  return io_wait_on(w, fd, dir, 0, tcp_send_buf_more);
}

static void __attribute__((constructor)) tcp_send_buf_use(void) {
  io_eff(CID_TCP_SEND_BUF, tcp_send_buf_run, 0);
}
