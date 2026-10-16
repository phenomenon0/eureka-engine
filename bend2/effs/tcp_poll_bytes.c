// TCP
// ===

// TCP.poll_bytes(sock, max, ms) is TCP.poll carrying bytes, one List cell
// each, as TCP.recv_bytes does. A recv is tried before any park, so a
// socket with data already waiting costs no pass; one that finds nothing
// parks on the socket and on the clock, whichever fires first. Past the
// deadline it answers None{}; data answers Some{bytes}, and the empty
// list is the peer's close, as TCP.recv_bytes answers it.
static Term tcp_poll_bytes_end(Env e, IoWork* w, Term r) {
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

static Term tcp_poll_bytes_pack(Env e, IoWork* w) {
  Term xs = term_pak(CID_NIL, 0);
  for (u64 i = w->size; i > 0; i -= 1) {
    xs = io_node(e, CID_CON, ((uint8_t*)w->data)[i - 1], xs);
  }
  return tcp_poll_bytes_end(e, w, io_done(e, io_box(e, CID_SOME, xs)));
}

static Term tcp_poll_bytes_more(Env e, IoWork* w);

static Term tcp_poll_bytes_at(Env e, IoWork* w, u64 at) {
  int   fd  = (int)w->hand;
  short dir = POLLIN;
  w->size = io_sys_end(w, io_wire_read(fd, w->data, (size_t)w->made, &dir));
  if (w->code == EAGAIN) {
    return io_tick() < at ? io_wait_on(w, fd, dir, at, tcp_poll_bytes_more)
      : tcp_poll_bytes_end(e, w, io_done(e, term_pak(CID_NONE, 0)));
  }
  return w->code ? tcp_poll_bytes_end(e, w, io_fail(e, w->code, NULL))
    : tcp_poll_bytes_pack(e, w);
}

static Term tcp_poll_bytes_more(Env e, IoWork* w) {
  return tcp_poll_bytes_at(e, w, io_wait_time(w));
}

Term tcp_poll_bytes_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->made = f[1] < INT32_MAX ? (intptr_t)f[1] : INT32_MAX;
  w->data = io_mem(malloc((size_t)w->made + 1));
  return tcp_poll_bytes_at(e, w, io_tick() + (u64)f[2] * 1000000ull);
}

static void __attribute__((constructor)) tcp_poll_bytes_use(void) {
  io_eff(CID_TCP_POLL_BYTES, tcp_poll_bytes_run, 0);
}
