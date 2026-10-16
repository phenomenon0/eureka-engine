// TLS
// ===

// TLS.listen(port, cert, key) is TCP.listen with a TLS session under
// every socket it accepts. It installs an IoWire, so nothing else
// changes: TCP.accept, TCP.recv_bytes, TCP.send_bytes, TCP.poll_bytes
// and Socket.close go through the wire already and do not know whether
// what they are carrying is encrypted. A program that never imports
// this never includes <openssl/ssl.h>, and the build links libssl only
// for the ones that do.
//
// The handshake is not done at accept: it runs inside the first read or
// write, driven by the same park the rest of the loop uses. That is why
// a slow or hostile handshake costs a parked computation and not a
// blocked server, and why the connection's idle deadline covers it.
#include <openssl/err.h>
#include <openssl/ssl.h>

// Which listeners are TLS, and which sockets have a session. Both are
// arrays over the descriptor, grown as descriptors get larger, because
// a descriptor is a small dense integer and anything cleverer would be
// a map that is never looked at twice.
static SSL_CTX** tls_ctx;
static u32       tls_ctx_cap;
static SSL**     tls_ssl;
static u32       tls_ssl_cap;
static u8*       tls_up;
static u32       tls_up_cap;

// What tls_up says of a socket: its session is not up yet, is up, has
// failed (so no close_notify is owed or sent), or never existed because
// SSL_new failed (so the socket is refused, never read in the clear).
enum { TLS_NEW = 0, TLS_UP = 1, TLS_BROKE = 2, TLS_NONE = 3 };

#define TLS_SLOT(tab, cap, fd, type)                                       \
  do {                                                                     \
    if ((u32)(fd) >= (cap)) {                                              \
      u32 was = (cap);                                                     \
      (cap) = (cap) != 0 ? (cap) : 256;                                    \
      while ((u32)(fd) >= (cap)) {                                         \
        (cap) *= 2;                                                        \
      }                                                                    \
      (tab) = io_mem(realloc((tab), (cap) * sizeof(type)));                \
      memset((tab) + was, 0, ((cap) - was) * sizeof(type));                \
    }                                                                      \
  } while (0)

static SSL* tls_of(int fd) {
  return fd >= 0 && (u32)fd < tls_ssl_cap ? tls_ssl[fd] : NULL;
}

static u8 tls_state(int fd) {
  return fd >= 0 && (u32)fd < tls_up_cap ? tls_up[fd] : TLS_NEW;
}

// Before every SSL_* call: what SSL_get_error reads is the error queue
// and errno, so a stale entry from another socket, or a stale EAGAIN,
// would be read as this call's answer.
static void tls_pre(void) {
  errno = 0;
  ERR_clear_error();
}

// What OpenSSL is waiting for, said the way the rest of the runtime
// says it: a short read that has to be tried again sets errno to EAGAIN
// and dir to the direction the socket must become ready in, which for a
// read is not always POLLIN.
static ssize_t tls_again(int fd, SSL* ssl, int n, short* dir) {
  int why = SSL_get_error(ssl, n);
  if (why == SSL_ERROR_WANT_READ) {
    *dir  = POLLIN;
    errno = EAGAIN;
    return -1;
  }
  if (why == SSL_ERROR_WANT_WRITE) {
    *dir  = POLLOUT;
    errno = EAGAIN;
    return -1;
  }
  if (why == SSL_ERROR_ZERO_RETURN) {
    return 0;
  }
  tls_up[fd] = TLS_BROKE;
  if (why == SSL_ERROR_SYSCALL) {
    if (errno == 0) {
      errno = ECONNRESET;
    }
    return -1;
  }
  ERR_clear_error();
  errno = EPROTO;
  return -1;
}

// The handshake, run wherever the first byte is wanted. It answers 1
// when the session is up, 0 when it needs the socket to become ready in
// dir, and -1 when it will never come up.
static int tls_hand(int fd, SSL* ssl, short* dir) {
  if (tls_state(fd) == TLS_UP) {
    return 1;
  }
  if (tls_state(fd) == TLS_BROKE) {
    errno = EPROTO;
    return -1;
  }
  tls_pre();
  int n = SSL_accept(ssl);
  if (n == 1) {
    tls_up[fd] = TLS_UP;
    return 1;
  }
  return tls_again(fd, ssl, n, dir) < 0 && errno == EAGAIN ? 0 : -1;
}

// A socket from a TLS listener that has no session is refused rather
// than read or written in the clear.
static bool tls_none(int fd) {
  if (tls_state(fd) != TLS_NONE) {
    return false;
  }
  errno = EPROTO;
  return true;
}

static ssize_t tls_read(int fd, void* buf, size_t len, short* dir) {
  SSL* ssl = tls_of(fd);
  if (ssl == NULL) {
    return tls_none(fd) ? -1 : recv(fd, buf, len, 0);
  }
  int up = tls_hand(fd, ssl, dir);
  if (up != 1) {
    return up == 0 ? (errno = EAGAIN, -1) : -1;
  }
  tls_pre();
  int n = SSL_read(ssl, buf, len > INT_MAX ? INT_MAX : (int)len);
  return n > 0 ? (ssize_t)n : tls_again(fd, ssl, n, dir);
}

static ssize_t tls_write(int fd, const void* buf, size_t len, short* dir) {
  SSL* ssl = tls_of(fd);
  if (ssl == NULL) {
    return tls_none(fd) ? -1 : send(fd, buf, len, 0);
  }
  int up = tls_hand(fd, ssl, dir);
  if (up != 1) {
    return up == 0 ? (errno = EAGAIN, -1) : -1;
  }
  tls_pre();
  int n = SSL_write(ssl, buf, len > INT_MAX ? INT_MAX : (int)len);
  return n > 0 ? (ssize_t)n : tls_again(fd, ssl, n, dir);
}

// A socket that closes without a close_notify leaves the peer unable to
// tell an orderly end from a cut connection, which is the truncation a
// peer is supposed to refuse to guess at. Ours goes out first, once and
// without waiting for theirs: the connection is ending either way, and
// a server that waited would be one a peer could hold open. After a
// fatal error there is no session to end, and none is attempted.
static void tls_shut(int fd) {
  SSL* ssl = tls_of(fd);
  if (ssl != NULL) {
    if (tls_state(fd) == TLS_UP) {
      tls_pre();
      SSL_shutdown(ssl);
      ERR_clear_error();
    }
    SSL_free(ssl);
    tls_ssl[fd] = NULL;
  }
  if (fd >= 0 && (u32)fd < tls_up_cap) {
    tls_up[fd] = TLS_NEW;
  }
  if (fd >= 0 && (u32)fd < tls_ctx_cap && tls_ctx[fd] != NULL) {
    SSL_CTX_free(tls_ctx[fd]);
    tls_ctx[fd] = NULL;
  }
}

// A socket accepted from a TLS listener gets that listener's session
// waiting to be handshaken; one from a plain listener gets nothing and
// stays a plain socket, so a program may serve both.
static void tls_join(int lfd, int fd) {
  SSL_CTX* ctx = lfd >= 0 && (u32)lfd < tls_ctx_cap ? tls_ctx[lfd] : NULL;
  if (ctx == NULL) {
    return;
  }
  TLS_SLOT(tls_up, tls_up_cap, fd, u8);
  tls_pre();
  SSL* ssl = SSL_new(ctx);
  if (ssl == NULL || SSL_set_fd(ssl, fd) != 1) {
    SSL_free(ssl);
    ERR_clear_error();
    shutdown(fd, SHUT_RDWR);
    tls_up[fd] = TLS_NONE;
    return;
  }
  SSL_set_accept_state(ssl);
  TLS_SLOT(tls_ssl, tls_ssl_cap, fd, SSL*);
  tls_ssl[fd] = ssl;
  tls_up[fd]  = TLS_NEW;
}

static IoWire tls_wire = { tls_read, tls_write, tls_join, tls_shut };

// What a listener agrees to by ALPN, in the wire's form (a length byte
// before each name), in the server's order of preference. TLS.listen's
// is http/1.1, what the HTTP/1 engine speaks; TLS.listen_alpn names its
// own (bend-h2's is h2), and a listener's list lives as long as it does.
typedef struct { u32 n; unsigned char at[255]; } TlsAlpn;

static TlsAlpn tls_alpn = { 9, { 8, 'h', 't', 't', 'p', '/', '1', '.', '1' } };

// The server's first choice the client also offers (SSL_select_next_proto
// walks the server's list and falls back to its first when nothing
// matches, which here means no agreement at all)
static int tls_pick(SSL* ssl, const unsigned char** out, unsigned char* outn,
  const unsigned char* in, unsigned int inn, void* arg) {
  TlsAlpn* a = arg != NULL ? (TlsAlpn*)arg : &tls_alpn;
  return SSL_select_next_proto((unsigned char**)out, outn, a->at, a->n, in, inn)
    == OPENSSL_NPN_NEGOTIATED ? SSL_TLSEXT_ERR_OK : SSL_TLSEXT_ERR_NOACK;
}

// The context, or why not, in words that name the file at fault
static SSL_CTX* tls_make(const char* cert, const char* key, TlsAlpn* alpn, uint32_t* err, char* why, u64 wn) {
  SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
  if (ctx == NULL) {
    *err = ENOMEM;
    return NULL;
  }
  // no renegotiation (a CPU lever for the peer); a close without a
  // close_notify is an end rather than an error; buffers go back while
  // a connection idles; TLS 1.2 only with ephemeral keys and AEAD (TLS
  // 1.3's suites are all of that already)
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  SSL_CTX_set_options(ctx, SSL_OP_NO_RENEGOTIATION
#ifdef SSL_OP_IGNORE_UNEXPECTED_EOF
    | SSL_OP_IGNORE_UNEXPECTED_EOF
#endif
    );
  SSL_CTX_set_mode(ctx, SSL_MODE_ENABLE_PARTIAL_WRITE
    | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER | SSL_MODE_RELEASE_BUFFERS);
  SSL_CTX_set_alpn_select_cb(ctx, tls_pick, alpn);
  const char* bad = NULL;
  const char* at  = "";
  if (SSL_CTX_set_cipher_list(ctx, "ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL") != 1) {
    bad = "the cipher list was refused";
  } else if (SSL_CTX_use_certificate_chain_file(ctx, cert) != 1) {
    bad = "no certificate (PEM) could be read from ";
    at  = cert;
  } else if (SSL_CTX_use_PrivateKey_file(ctx, key, SSL_FILETYPE_PEM) != 1) {
    bad = "no private key (PEM) matching the certificate could be read from ";
    at  = key;
  } else if (SSL_CTX_check_private_key(ctx) != 1) {
    bad = "the key does not match the certificate: ";
    at  = key;
  }
  if (bad != NULL) {
    snprintf(why, wn, "%s%s", bad, at);
    ERR_clear_error();
    SSL_CTX_free(ctx);
    *err = EINVAL;
    return NULL;
  }
  return ctx;
}

// The same socket TCP.listen opens, opened here rather than borrowed:
// an effect is only compiled into a program that uses it, so one that
// leans on another's C is one that does not build.
static uint32_t tls_bind(const char* host, uint32_t port, int* out) {
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
  if (bind(fd, (struct sockaddr*)&at, len) < 0
    || listen(fd, SOMAXCONN) < 0
    || fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) < 0) {
    uint32_t code = (uint32_t)errno;
    close(fd);
    return code;
  }
  *out = fd;
  return 0;
}

// a listener: the address, the port, the certificate and key (their
// terms), the ALPN list
static Term tls_listen_go(Env e, const char* host, uint32_t port, Term tc, Term tk, TlsAlpn* alpn) {
  u64   cn = 0, kn = 0;
  char* cert = io_cstr(e, tc, &cn);
  char* key  = io_cstr(e, tk, &kn);
  if (io_nul(cert, cn) || io_nul(key, kn)) {
    free(cert);
    free(key);
    free(alpn);
    return io_fail(e, EINVAL, NULL);
  }
  uint32_t err = 0;
  char     why[1024];
  why[0] = 0;
  SSL_CTX* ctx = tls_make(cert, key, alpn, &err, why, sizeof(why));
  free(cert);
  free(key);
  if (ctx == NULL) {
    free(alpn);
    return io_fail(e, err, why[0] != 0 ? why : NULL);
  }
  int out = -1;
  uint32_t q = tls_bind(host, port, &out);
  if (q != 0) {
    SSL_CTX_free(ctx);
    free(alpn);
    return io_fail(e, q, NULL);
  }
  TLS_SLOT(tls_ctx, tls_ctx_cap, out, SSL_CTX*);
  tls_ctx[out] = ctx;
  io_wire      = &tls_wire;
  return io_done(e, io_hand(out));
}

#ifdef CID_TLS_LISTEN
Term tls_listen_run(Env e, Term* f, IoWork* w) {
  return tls_listen_go(e, "0.0.0.0", (uint32_t)f[0], f[1], f[2], NULL);
}

static void __attribute__((constructor)) tls_listen_use(void) {
  io_eff(CID_TLS_LISTEN, tls_listen_run, 0);
}
#endif

// TLS.listen_on(host, port, cert, key): TLS.listen on the one address
// named, as TCP.listen_on reads it
#ifdef CID_TLS_LISTEN_ON
Term tls_listen_on_run(Env e, Term* f, IoWork* w) {
  u64   hn = 0;
  char* host = io_cstr(e, f[0], &hn);
  if (io_nul(host, hn)) {
    free(host);
    return io_fail(e, EINVAL, NULL);
  }
  Term r = tls_listen_go(e, host, (uint32_t)f[1], f[2], f[3], NULL);
  free(host);
  return r;
}

static void __attribute__((constructor)) tls_listen_on_use(void) {
  io_eff(CID_TLS_LISTEN_ON, tls_listen_on_run, 0);
}
#endif

// TLS.listen_alpn(port, cert, key, names): names is a comma-separated
// list ("h2", "h2,http/1.1"), each 1 to 255 bytes, turned here into the
// wire's form; an empty name or a list past 255 bytes is EINVAL.
// TLS.connect offers its list in the same form.
static TlsAlpn* tls_alpn_of(const char* s, u64 n) {
  TlsAlpn* a = io_mem(calloc(1, sizeof(TlsAlpn)));
  u64 at = 0;
  while (at <= n) {
    u64 end = at;
    while (end < n && s[end] != ',') {
      end += 1;
    }
    u64 len = end - at;
    if (len == 0 || len > 254 || a->n + 1 + len > sizeof(a->at)) {
      free(a);
      return NULL;
    }
    a->at[a->n] = (unsigned char)len;
    memcpy(a->at + a->n + 1, s + at, len);
    a->n += 1 + (u32)len;
    at = end + 1;
  }
  return a;
}

#ifdef CID_TLS_LISTEN_ALPN
Term tls_listen_alpn_run(Env e, Term* f, IoWork* w) {
  u64   an = 0;
  char* names = io_cstr(e, f[3], &an);
  TlsAlpn* alpn = io_nul(names, an) ? NULL : tls_alpn_of(names, an);
  free(names);
  if (alpn == NULL) {
    return io_fail(e, EINVAL, NULL);
  }
  return tls_listen_go(e, "0.0.0.0", (uint32_t)f[0], f[1], f[2], alpn);
}

static void __attribute__((constructor)) tls_listen_alpn_use(void) {
  io_eff(CID_TLS_LISTEN_ALPN, tls_listen_alpn_run, 0);
}
#endif

// TLS.connect(addr, port, name, alpn, ca, ms): a TCP connect to addr (a
// dotted IPv4 address or an IPv6 one) and a TLS client handshake over it, both under
// one deadline of ms. The session is the one TLS.listen's sockets get,
// so the byte effects carry it without knowing; unlike theirs, its
// handshake is done here, so a peer that is not who it says it is fails
// the connect rather than the first read.
//
// The peer is verified: its chain against the system's store (ca "") or
// against the one file of certificates ca names (a pin), and its name
// against name -- a host name checked as RFC 6125 says and sent as SNI,
// or an address literal (IPv4, or IPv6 without brackets) checked against
// the certificate's IP addresses and never sent (RFC 6066 3: SNI names
// hosts, not addresses).
// alpn is what is offered, comma-separated ("" offers nothing); the
// answer carries the protocol the server chose ("" for none). A peer
// that closes without a close_notify is a failed read, never a FIN: a
// body delimited by the close cannot be cut short and pass as whole.
//
// Codes: ETIMEDOUT for the deadline; EACCES for a certificate refused
// (untrusted, expired, another name), with the verifier's reason;
// EPROTO for any other handshake failure; the connect's own otherwise.
#ifdef CID_TLS_CONNECT
#include <netinet/tcp.h>
#include <openssl/x509v3.h>
#include <poll.h>

typedef struct { char* ca; SSL_CTX* ctx; } TlsCli;

static TlsCli tls_cli[8];

// one client context per trust store, kept: loading the system's store
// is the dear part of a connect. Past eight stores a context is made for
// the connect and freed with its session (SSL_free drops the last
// reference).
static SSL_CTX* tls_cli_ctx(const char* ca, bool* own) {
  *own = false;
  for (u32 i = 0; i < 8; i += 1) {
    if (tls_cli[i].ctx != NULL && strcmp(tls_cli[i].ca, ca) == 0) {
      return tls_cli[i].ctx;
    }
  }
  tls_pre();
  SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
  if (ctx == NULL) {
    return NULL;
  }
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  SSL_CTX_set_options(ctx, SSL_OP_NO_RENEGOTIATION);
  SSL_CTX_set_mode(ctx, SSL_MODE_ENABLE_PARTIAL_WRITE
    | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER | SSL_MODE_RELEASE_BUFFERS);
  SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
  int ok = ca[0] == 0 ? SSL_CTX_set_default_verify_paths(ctx)
    : SSL_CTX_load_verify_locations(ctx, ca, NULL);
  if (ok != 1) {
    ERR_clear_error();
    SSL_CTX_free(ctx);
    return NULL;
  }
  for (u32 i = 0; i < 8; i += 1) {
    if (tls_cli[i].ctx == NULL) {
      tls_cli[i].ca  = io_mem(strdup(ca));
      tls_cli[i].ctx = ctx;
      return ctx;
    }
  }
  *own = true;
  return ctx;
}

// While it runs: w->made the descriptor, w->hand the session (0 while
// the TCP connect is under way), w->data the name, w->text the ALPN
// list and the trust store, one after the other (w->size where the
// store begins).
static Term tls_connect_end(Env e, IoWork* w, int err, const char* why) {
  int  fd  = (int)w->made;
  SSL* ssl = (SSL*)w->hand;
  Term r;
  if (err != 0) {
    if (ssl != NULL) {
      SSL_free(ssl);
    }
    if (fd >= 0) {
      close(fd);
    }
    r = io_fail(e, (u32)err, why);
  } else {
    const unsigned char* p = NULL;
    unsigned int         n = 0;
    SSL_get0_alpn_selected(ssl, &p, &n);
    TLS_SLOT(tls_up, tls_up_cap, fd, u8);
    TLS_SLOT(tls_ssl, tls_ssl_cap, fd, SSL*);
    tls_ssl[fd] = ssl;
    tls_up[fd]  = TLS_UP;
    io_wire     = &tls_wire;
    r = io_done(e, io_tup(e, io_hand(fd), io_str(e, p != NULL ? (const char*)p : "", n)));
  }
  ERR_clear_error();
  free(w->data);
  free(w->text);
  return r;
}

static Term tls_connect_more(Env e, IoWork* w);

// the handshake, a step: done, parked on the way the socket must become
// ready, or failed
static Term tls_connect_shake(Env e, IoWork* w, u64 at) {
  int  fd  = (int)w->made;
  SSL* ssl = (SSL*)w->hand;
  tls_pre();
  int n = SSL_do_handshake(ssl);
  if (n == 1) {
    return tls_connect_end(e, w, 0, NULL);
  }
  int why = SSL_get_error(ssl, n);
  if (why == SSL_ERROR_WANT_READ || why == SSL_ERROR_WANT_WRITE) {
    return io_tick() < at
      ? io_wait_on(w, fd, why == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT, at, tls_connect_more)
      : tls_connect_end(e, w, ETIMEDOUT, NULL);
  }
  long v = SSL_get_verify_result(ssl);
  if (v != X509_V_OK) {
    return tls_connect_end(e, w, EACCES, X509_verify_cert_error_string(v));
  }
  unsigned long q = ERR_peek_error();
  const char* text = q != 0 ? ERR_reason_error_string(q) : NULL;
  return tls_connect_end(e, w, why == SSL_ERROR_SYSCALL && q == 0 ? ECONNRESET : EPROTO, text);
}

// the session over the connected socket: the store, the name to verify
// and to send, the offer
static Term tls_connect_open(Env e, IoWork* w, u64 at) {
  int      fd  = (int)w->made;
  bool     own = false;
  SSL_CTX* ctx = tls_cli_ctx(w->text + w->size, &own);
  if (ctx == NULL) {
    return tls_connect_end(e, w, EINVAL, "the trust store did not load");
  }
  tls_pre();
  SSL* ssl = SSL_new(ctx);
  if (own) {
    SSL_CTX_free(ctx);
  }
  if (ssl == NULL || SSL_set_fd(ssl, fd) != 1) {
    SSL_free(ssl);
    return tls_connect_end(e, w, ENOMEM, NULL);
  }
  w->hand = (intptr_t)ssl;
  SSL_set_connect_state(ssl);
  unsigned char ip[16];
  bool literal = inet_pton(AF_INET, w->data, ip) == 1 || inet_pton(AF_INET6, w->data, ip) == 1;
  int ok = literal
    ? X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl), w->data)
    : SSL_set_tlsext_host_name(ssl, w->data) == 1 && SSL_set1_host(ssl, w->data) == 1;
  if (ok != 1) {
    return tls_connect_end(e, w, EINVAL, "the name cannot be verified");
  }
  if (w->text[0] != 0) {
    TlsAlpn* a = tls_alpn_of(w->text, strlen(w->text));
    if (a == NULL) {
      return tls_connect_end(e, w, EINVAL, "the ALPN list is malformed");
    }
    int bad = SSL_set_alpn_protos(ssl, a->at, a->n);
    free(a);
    if (bad != 0) {
      return tls_connect_end(e, w, EINVAL, "the ALPN list is malformed");
    }
  }
  return tls_connect_shake(e, w, at);
}

static Term tls_connect_more(Env e, IoWork* w) {
  u64 at = io_wait_time(w);
  int fd = (int)w->made;
  if (w->hand != 0) {
    return tls_connect_shake(e, w, at);
  }
  struct pollfd p = { fd, POLLOUT, 0 };
  if (poll(&p, 1, 0) <= 0) {
    return io_tick() < at ? io_wait_on(w, fd, POLLOUT, at, tls_connect_more)
      : tls_connect_end(e, w, ETIMEDOUT, NULL);
  }
  int       err = 0;
  socklen_t len = sizeof(err);
  if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0) {
    err = errno;
  }
  return err != 0 ? tls_connect_end(e, w, err, NULL) : tls_connect_open(e, w, at);
}

Term tls_connect_run(Env e, Term* f, IoWork* w) {
  struct sockaddr_storage to;
  socklen_t               tn = 0;
  u64   an = 0, nn = 0, pn = 0, cn = 0;
  char* addr = io_cstr(e, f[0], &an);
  w->data = io_cstr(e, f[2], &nn);
  char* alpn = io_cstr(e, f[3], &pn);
  char* ca   = io_cstr(e, f[4], &cn);
  w->text = io_mem(malloc(pn + cn + 2));
  memcpy(w->text, alpn, pn + 1);
  memcpy(w->text + pn + 1, ca, cn + 1);
  w->size = pn + 1;
  w->made = -1;
  w->hand = 0;
  bool bad = io_nul(addr, an) || io_nul(w->data, nn) || nn == 0 || io_nul(alpn, pn)
    || io_nul(ca, cn) || io_sys_sa(addr, (u32)f[1], &to, &tn) < 0;
  free(addr);
  free(alpn);
  free(ca);
  if (bad) {
    return tls_connect_end(e, w, EINVAL, NULL);
  }
  u64 at = io_tick() + ((u64)f[5] == 0 ? 1 : (u64)f[5]) * 1000000ull;
  int fd = socket(to.ss_family, SOCK_STREAM, 0);
  if (fd < 0) {
    return tls_connect_end(e, w, errno, NULL);
  }
  w->made = fd;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) < 0) {
    return tls_connect_end(e, w, errno, NULL);
  }
  if (connect(fd, (struct sockaddr*)&to, tn) == 0) {
    return tls_connect_open(e, w, at);
  }
  if (errno != EINPROGRESS) {
    return tls_connect_end(e, w, errno, NULL);
  }
  return io_wait_on(w, fd, POLLOUT, at, tls_connect_more);
}

static void __attribute__((constructor)) tls_connect_use(void) {
  io_eff(CID_TLS_CONNECT, tls_connect_run, 0);
}
#endif
