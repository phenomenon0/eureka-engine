// Socket
// ======

Term socket_close_run(Env e, Term* f, IoWork* w) {
  int fd = (int)io_hand_v(f[0]);
  io_wire_shut(fd);
  io_fd_gone(fd);
  close(fd);
  return term_pak(CID_UNIT, 0);
}

static void __attribute__((constructor)) socket_close_use(void) {
  io_eff(CID_SOCKET_CLOSE, socket_close_run, 0);
}
