// TCP
// ===

// TCP.accept_poll(listener, ms) is TCP.accept with a deadline: an accept
// is tried first, and one that finds no connection parks on the listener
// and on the clock, whichever fires first. Past the deadline it answers
// None{}, so a loop that accepts can also look up from time to time --
// at a signal, at a count -- without a second computation to do it.
//
// Only a listener that is no longer one fails it. accept(2) also fails
// for one connection (reset before it was taken, a network error the
// kernel hands over with it) and for a passing shortage (descriptors,
// buffers, memory); those are the peer's or the moment's, not the
// server's, and answering them as a failure is how one burst of
// arrivals stops a server for good. Out of descriptors, the connection
// at the head of the queue is taken on a spare one and closed, so it
// leaves the backlog instead of waking the loop forever; out of
// buffers, the loop steps back for a moment.
#include <netinet/tcp.h>

static Term tcp_accept_poll_more(Env e, IoWork* w);
static int  tcp_accept_poll_spare = -1;

static int tcp_accept_poll_spare_open(void) {
#ifdef SOCK_CLOEXEC
  return socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
  return socket(AF_INET, SOCK_STREAM, 0);
#endif
}

static bool tcp_accept_poll_passing(int code) {
  switch (code) {
    case ECONNABORTED: case EINTR: case EPROTO: case EPERM: case ENOBUFS:
    case ENOMEM: case EMFILE: case ENFILE: case ENETDOWN: case ENOPROTOOPT:
    case EHOSTDOWN: case EHOSTUNREACH: case EOPNOTSUPP: case ENETUNREACH:
#ifdef ENONET
    case ENONET:
#endif
      return true;
  }
  return false;
}

// One accept: non-blocking and close-on-exec from birth, Nagle off.
static int tcp_accept_poll_one(int fd) {
#ifdef __linux__
  int got = accept4(fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
#else
  int got = accept(fd, NULL, NULL);
  if (got >= 0 && (fcntl(got, F_SETFL, fcntl(got, F_GETFL) | O_NONBLOCK) < 0
    || fcntl(got, F_SETFD, FD_CLOEXEC) < 0)) {
    close(got);
    errno = ECONNABORTED;
    got   = -1;
  }
#endif
  if (got >= 0) {
    int one = 1;
    setsockopt(got, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return got;
}

// Out of descriptors: the spare one is given up for the moment it takes
// to accept the waiting connection and close it. Answers whether one
// was shed; when none was, the caller backs off instead.
static bool tcp_accept_poll_shed(int fd) {
  if (tcp_accept_poll_spare < 0) {
    return false;
  }
  close(tcp_accept_poll_spare);
  int got = accept(fd, NULL, NULL);
  if (got >= 0) {
    close(got);
  }
  tcp_accept_poll_spare = tcp_accept_poll_spare_open();
  return got >= 0;
}

// w->size holds the caller's deadline; a back-off parks on the clock
// alone, for at most 20 ms and never past it.
static Term tcp_accept_poll_at(Env e, IoWork* w, u64 at) {
  int fd = (int)w->hand;
  w->size = at;
  for (int tries = 0; tries < 64; tries += 1) {
    int got = tcp_accept_poll_one(fd);
    int why = got < 0 ? errno : 0;
    if (got >= 0) {
      io_fd_fresh(got);
      io_wire_join(fd, got);
      return io_tup(e, io_hand(fd), io_done(e, io_box(e, CID_SOME, io_hand(got))));
    }
    if (why == EAGAIN || why == EWOULDBLOCK) {
      return io_tick() < at ? io_wait_on(w, fd, POLLIN, at, tcp_accept_poll_more)
        : io_tup(e, io_hand(fd), io_done(e, term_pak(CID_NONE, 0)));
    }
    if (!tcp_accept_poll_passing(why)) {
      return io_tup(e, io_hand(fd), io_fail(e, (u32)why, NULL));
    }
    bool out = why == EMFILE || why == ENFILE;
    if ((out && !tcp_accept_poll_shed(fd)) || why == ENOBUFS || why == ENOMEM) {
      u64 now = io_tick(), back = now + 20000000ull;
      return now < at ? io_wait_on(w, -1, 0, back < at ? back : at, tcp_accept_poll_more)
        : io_tup(e, io_hand(fd), io_done(e, term_pak(CID_NONE, 0)));
    }
  }
  return io_tup(e, io_hand(fd), io_done(e, term_pak(CID_NONE, 0)));
}

static Term tcp_accept_poll_more(Env e, IoWork* w) {
  return tcp_accept_poll_at(e, w, w->size);
}

Term tcp_accept_poll_run(Env e, Term* f, IoWork* w) {
  if (tcp_accept_poll_spare < 0) {
    tcp_accept_poll_spare = tcp_accept_poll_spare_open();
  }
  w->hand = (intptr_t)io_hand_v(f[0]);
  return tcp_accept_poll_at(e, w, io_tick() + (u64)f[1] * 1000000ull);
}

static void __attribute__((constructor)) tcp_accept_poll_use(void) {
  io_eff(CID_TCP_ACCEPT_POLL, tcp_accept_poll_run, 0);
}
