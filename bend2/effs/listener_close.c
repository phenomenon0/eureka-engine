// Listener
// ========

// A listener's wire goes with it: a TLS listener's context would
// otherwise be found by the next listener given the same number.
Term listener_close_run(Env e, Term* f, IoWork* w) {
  int fd = (int)io_hand_v(f[0]);
  io_wire_shut(fd);
  io_fd_gone(fd);
  close(fd);
  return term_pak(CID_UNIT, 0);
}

static void __attribute__((constructor)) listener_close_use(void) {
  io_eff(CID_LISTENER_CLOSE, listener_close_run, 0);
}
