// TCP
// ===

// TCP.poll_bytes into a Bytes, as TCP.recv_buf is TCP.recv_bytes: past
// the deadline None{}, data Some{bytes}, and the empty Bytes the peer's
// close. A recv is tried before any park, unless the socket is plain
// and the poller has reported nothing since a read found it drained:
// then the recv could only say EAGAIN, and it parks at once.
static Term tcp_poll_buf_end(Env e, IoWork* w, Term r) {
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

static Term tcp_poll_buf_more(Env e, IoWork* w);

static Term tcp_poll_buf_at(Env e, IoWork* w, u64 at) {
  int   fd  = (int)w->hand;
  short dir = POLLIN;
  bool  raw = io_wire == NULL;
  if (raw && io_fd_quiet(fd, 1)) {
    w->size = 0;
    w->code = EAGAIN;
  } else {
    w->size = io_sys_end(w, io_wire_read(fd, w->data, (size_t)w->made, &dir));
    if (raw && w->code == 0 && w->size > 0) {
      io_fd_seen(fd, 1, w->size >= (u64)w->made);
    }
  }
  if (w->code == EAGAIN) {
    return io_tick() < at ? io_wait_on(w, fd, dir, at, tcp_poll_buf_more)
      : tcp_poll_buf_end(e, w, io_done(e, term_pak(CID_NONE, 0)));
  }
  return tcp_poll_buf_end(e, w, w->code ? io_fail(e, w->code, NULL)
    : io_done(e, io_box(e, CID_SOME, io_buf(e, w->data, w->size))));
}

static Term tcp_poll_buf_more(Env e, IoWork* w) {
  return tcp_poll_buf_at(e, w, io_wait_time(w));
}

Term tcp_poll_buf_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->made = f[1] < INT32_MAX ? (intptr_t)f[1] : INT32_MAX;
  w->data = io_mem(malloc((size_t)w->made + 1));
  return tcp_poll_buf_at(e, w, io_tick() + (u64)f[2] * 1000000ull);
}

static void __attribute__((constructor)) tcp_poll_buf_use(void) {
  io_eff(CID_TCP_POLL_BUF, tcp_poll_buf_run, 0);
}
