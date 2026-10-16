// TCP
// ===

// The bytes as they are (0..255), one List cell each, exactly as
// File.read_bytes gives them. TCP.recv hands back a String, and a
// String is built by io_str, which decodes the bytes as UTF-8: on the
// wire that is not a cost but a corruption, since every byte that is
// not valid UTF-8 becomes U+FFFD, three bytes out for one byte in. A
// reader of a binary body needs this one instead.
static Term tcp_recv_bytes_pack(Env e, IoWork* w) {
  Term r;
  if (w->code) {
    r = io_fail(e, w->code, NULL);
  } else {
    Term xs = term_pak(CID_NIL, 0);
    for (u64 i = w->size; i > 0; i -= 1) {
      xs = io_node(e, CID_CON, ((uint8_t*)w->data)[i - 1], xs);
    }
    r = io_done(e, xs);
  }
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

// The loop parked the request until the socket was readable; a recv
// that still finds nothing (the socket is non-blocking) parks again.
static Term tcp_recv_bytes_more(Env e, IoWork* w) {
  int   fd  = (int)w->hand;
  short dir = POLLIN;
  w->size = io_sys_end(w, io_wire_read(fd, w->data, (size_t)w->made, &dir));
  return w->code == EAGAIN ? io_wait_on(w, fd, dir, 0, tcp_recv_bytes_more)
    : tcp_recv_bytes_pack(e, w);
}

Term tcp_recv_bytes_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->made = f[1] < INT32_MAX ? (intptr_t)f[1] : INT32_MAX;
  w->data = io_mem(malloc((size_t)w->made + 1));
  return tcp_recv_bytes_more(e, w);
}

static void __attribute__((constructor)) tcp_recv_bytes_use(void) {
  io_eff(CID_TCP_RECV_BYTES, tcp_recv_bytes_run, IO_READ);
}
