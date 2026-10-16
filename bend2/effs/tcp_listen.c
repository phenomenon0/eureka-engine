// TCP
// ===

// TCP.listen(port) binds every interface; TCP.listen_on(host, port) the
// one address named, a dotted IPv4 one or an IPv6 one ("::1"; "::" is
// every interface, IPv4's too where the system allows: IPV6_V6ONLY off);
// EINVAL for anything else.
uint32_t tcp_listen_at(const char* host, uint32_t port, int* out) {
  struct sockaddr_storage at;
  socklen_t               len = 0;
  int                     fam = io_sys_sa(host, port, &at, &len);
  if (fam < 0) {
    return EINVAL;
  }
  int fd = socket(fam, SOCK_STREAM, 0);
  if (fd < 0) {
    return (uint32_t)errno;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  io_sys_dual(fd, fam);
  int bound = bind(fd, (struct sockaddr*)&at, len);
  if (bound < 0 || listen(fd, SOMAXCONN) < 0
    || fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) < 0) {
    uint32_t code = (uint32_t)errno;
    close(fd);
    return code;
  }
  *out = fd;
  return 0;
}

uint32_t tcp_listen(uint32_t port, int* out) {
  return tcp_listen_at("0.0.0.0", port, out);
}

static Term tcp_listen_end(Env e, uint32_t q, int out) {
  return q != 0 ? io_fail(e, q, NULL) : io_done(e, io_hand(out));
}

#ifdef CID_TCP_LISTEN
Term tcp_listen_run(Env e, Term* f, IoWork* w) {
  int out = -1;
  uint32_t q = tcp_listen((uint32_t)f[0], &out);
  return tcp_listen_end(e, q, out);
}

static void __attribute__((constructor)) tcp_listen_use(void) {
  io_eff(CID_TCP_LISTEN, tcp_listen_run, 0);
}
#endif

#ifdef CID_TCP_LISTEN_ON
Term tcp_listen_on_run(Env e, Term* f, IoWork* w) {
  u64   n = 0;
  char* host = io_cstr(e, f[0], &n);
  int   out = -1;
  uint32_t q = io_nul(host, n) ? EINVAL : tcp_listen_at(host, (uint32_t)f[1], &out);
  free(host);
  return tcp_listen_end(e, q, out);
}

static void __attribute__((constructor)) tcp_listen_on_use(void) {
  io_eff(CID_TCP_LISTEN_ON, tcp_listen_on_run, 0);
}
#endif
