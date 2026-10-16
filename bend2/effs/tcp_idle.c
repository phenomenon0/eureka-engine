// TCP
// ===

// TCP.idle(sock): whether nothing waits to be read on the socket and the
// peer has not closed it -- the test a pool puts a kept connection to
// before handing it out. One poll with no wait: readable (a byte, the
// peer's FIN, an error) is not idle. Over TLS the same test is made of
// the socket under the session: a record that arrived is not idle.
#include <poll.h>

Term tcp_idle_run(Env e, Term* f, IoWork* w) {
  int           fd = (int)io_hand_v(f[0]);
  struct pollfd p  = { fd, POLLIN, 0 };
  int           n  = poll(&p, 1, 0);
  bool          ok = n == 0;
  return io_tup(e, io_hand(fd), term_pak(ok ? CID_TRUE : CID_FALSE, 0));
}

static void __attribute__((constructor)) tcp_idle_use(void) {
  io_eff(CID_TCP_IDLE, tcp_idle_run, 0);
}
