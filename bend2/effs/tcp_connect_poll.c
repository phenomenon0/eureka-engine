// TCP
// ===

// TCP.connect with a deadline, to a dotted IPv4 address or an IPv6 one:
// the socket is non-blocking for life, a
// connect that is not done at once parks on the socket becoming writable
// and on the clock, and whichever comes first ends it. Writable, SO_ERROR
// says how the connect went; the clock first, it is ETIMEDOUT and the
// socket is closed. w->made holds the descriptor.
#include <netinet/tcp.h>
#include <poll.h>

static Term tcp_connect_poll_end(Env e, IoWork* w, int err) {
  int fd = (int)w->made;
  if (err != 0 && fd >= 0) {
    close(fd);
  }
  free(w->data);
  return err != 0 ? io_fail(e, (u32)err, NULL) : io_done(e, io_hand(fd));
}

static Term tcp_connect_poll_more(Env e, IoWork* w) {
  int           fd = (int)w->made;
  struct pollfd p  = { fd, POLLOUT, 0 };
  if (poll(&p, 1, 0) > 0) {
    int       err = 0;
    socklen_t len = sizeof(err);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0) {
      err = errno;
    }
    return tcp_connect_poll_end(e, w, err);
  }
  u64 at = io_wait_time(w);
  return io_tick() < at ? io_wait_on(w, fd, POLLOUT, at, tcp_connect_poll_more)
    : tcp_connect_poll_end(e, w, ETIMEDOUT);
}

Term tcp_connect_poll_run(Env e, Term* f, IoWork* w) {
  struct sockaddr_storage at;
  socklen_t               len = 0;
  w->data = io_cstr(e, f[0], &w->size);
  w->made = -1;
  if (io_nul(w->data, w->size) || io_sys_sa(w->data, (u32)f[1], &at, &len) < 0) {
    return tcp_connect_poll_end(e, w, EINVAL);
  }
  int fd = socket(at.ss_family, SOCK_STREAM, 0);
  if (fd < 0) {
    return tcp_connect_poll_end(e, w, errno);
  }
  w->made = fd;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) < 0) {
    return tcp_connect_poll_end(e, w, errno);
  }
  if (connect(fd, (struct sockaddr*)&at, len) == 0) {
    return tcp_connect_poll_end(e, w, 0);
  }
  if (errno != EINPROGRESS) {
    return tcp_connect_poll_end(e, w, errno);
  }
  u64 ms = (u64)f[2];
  return io_wait_on(w, fd, POLLOUT, io_tick() + (ms == 0 ? 1 : ms) * 1000000ull,
    tcp_connect_poll_more);
}

static void __attribute__((constructor)) tcp_connect_poll_use(void) {
  io_eff(CID_TCP_CONNECT_POLL, tcp_connect_poll_run, 0);
}
