// TCP
// ===

// TCP.send_buf_poll(sock, bytes, ms) is TCP.send_buf with a deadline: a
// full socket parks on the socket and on the clock, and a send that has
// made no progress for ms fails with ETIMEDOUT. Each write that moves
// bytes starts the wait over, so a slow reader is served and one that
// has stopped reading is let go; without it, a peer that never reads
// holds its computation, and whatever that holds, for good. What a full
// socket leaves unsent is copied after an 8-byte lead that keeps ms, so
// a resumed send can re-arm.
static Term tcp_send_buf_poll_pack(Env e, IoWork* w) {
  Term r = w->code != 0 ? io_fail(e, w->code, NULL)
    : io_done(e, term_pak(CID_UNIT, 0));
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

static Term tcp_send_buf_poll_more(Env e, IoWork* w);

// at is the deadline of the wait in progress, 0 when none is
static Term tcp_send_buf_poll_at(Env e, IoWork* w, u64 at) {
  int   fd  = (int)w->hand;
  short dir = POLLOUT;
  u64   ms;
  memcpy(&ms, w->data, sizeof(ms));
  while (w->code == 0 && (u64)w->made < w->size) {
    ssize_t n = io_wire_write(fd, w->data + 8 + w->made, w->size - (u64)w->made, &dir);
    if (n < 0 && errno == EAGAIN) {
      u64 now = io_tick();
      at = at != 0 ? at : now + ms * 1000000ull;
      if (now >= at) {
        w->code = ETIMEDOUT;
        break;
      }
      return io_wait_on(w, fd, dir, at, tcp_send_buf_poll_more);
    }
    at       = 0;
    w->made += io_sys_end(w, n);
  }
  return tcp_send_buf_poll_pack(e, w);
}

static Term tcp_send_buf_poll_more(Env e, IoWork* w) {
  return tcp_send_buf_poll_at(e, w, io_wait_time(w));
}

// The first writes go straight from the heap, as TCP.send_buf's do;
// only a remainder is copied.
Term tcp_send_buf_poll_run(Env e, Term* f, IoWork* w) {
  int   fd  = (int)io_hand_v(f[0]);
  short dir = POLLOUT;
  u32   len = 0;
  u64   ms  = (u64)(u32)f[2], at = 0;
  char* own = NULL;
  const char* p = io_buf_ptr(e, f[1], &len, &own);
  w->hand = (intptr_t)fd;
  w->code = p == NULL ? EINVAL : 0;
  w->data = NULL;
  while (w->code == 0 && at < len) {
    ssize_t n = io_wire_write(fd, p + at, len - at, &dir);
    if (n < 0 && errno == EAGAIN) {
      break;
    }
    at += io_sys_end(w, n);
  }
  if (w->code == 0 && at < len) {
    w->data = io_mem(malloc(8 + len - at));
    memcpy(w->data, &ms, sizeof(ms));
    memcpy(w->data + 8, p + at, len - at);
  }
  free(own);
  term_sink(e, f[1]);
  if (w->data == NULL) {
    return tcp_send_buf_poll_pack(e, w);
  }
  w->size = len - at;
  w->made = 0;
  return tcp_send_buf_poll_at(e, w, 0);
}

static void __attribute__((constructor)) tcp_send_buf_poll_use(void) {
  io_eff(CID_TCP_SEND_BUF_POLL, tcp_send_buf_poll_run, 0);
}
