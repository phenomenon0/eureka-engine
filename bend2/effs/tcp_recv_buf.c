// TCP
// ===

// TCP.recv_bytes into a Bytes: the bytes received are copied once into
// one packed block (io_buf), a byte a cell, instead of a List cell each.
static Term tcp_recv_buf_pack(Env e, IoWork* w) {
  Term r = w->code ? io_fail(e, w->code, NULL)
    : io_done(e, io_buf(e, w->data, w->size));
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

// The loop parked the request until the socket was readable; a recv
// that still finds nothing (the socket is non-blocking) parks again.
static Term tcp_recv_buf_more(Env e, IoWork* w) {
  int   fd  = (int)w->hand;
  short dir = POLLIN;
  w->size = io_sys_end(w, io_wire_read(fd, w->data, (size_t)w->made, &dir));
  return w->code == EAGAIN ? io_wait_on(w, fd, dir, 0, tcp_recv_buf_more)
    : tcp_recv_buf_pack(e, w);
}

Term tcp_recv_buf_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->made = f[1] < INT32_MAX ? (intptr_t)f[1] : INT32_MAX;
  w->data = io_mem(malloc((size_t)w->made + 1));
  return tcp_recv_buf_more(e, w);
}

static void __attribute__((constructor)) tcp_recv_buf_use(void) {
  io_eff(CID_TCP_RECV_BUF, tcp_recv_buf_run, IO_READ);
}
