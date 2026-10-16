// DNS
// ===

// DNS.resolve(name): the name's first IPv4 address, dotted, as the
// system's resolver has it (getaddrinfo: /etc/hosts, then DNS). The
// resolver blocks, so it runs on a helper thread and the loop goes on.
// A dotted address is its own answer. A name that does not resolve is
// ENOENT, with the resolver's reason as the text.
#include <netdb.h>

static void dns_resolve_call(IoWork* w) {
  struct addrinfo  hint = { 0 };
  struct addrinfo* got  = NULL;
  hint.ai_family   = AF_INET;
  hint.ai_socktype = SOCK_STREAM;
  int r = getaddrinfo(w->data, NULL, &hint, &got);
  if (r != 0 || got == NULL) {
    w->code = ENOENT;
    w->text = (char*)gai_strerror(r);
    return;
  }
  char out[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &((struct sockaddr_in*)got->ai_addr)->sin_addr, out, sizeof(out));
  freeaddrinfo(got);
  w->code = 0;
  w->size = strlen(out);
  memcpy(w->data, out, w->size + 1);
}

static Term dns_resolve_pack(Env e, IoWork* w) {
  Term r = w->code != 0 ? io_fail(e, w->code, w->text)
    : io_done(e, io_str(e, w->data, w->size));
  free(w->data);
  return r;
}

Term dns_resolve_run(Env e, Term* f, IoWork* w) {
  u64   n    = 0;
  char* name = io_cstr(e, f[0], &n);
  w->text = NULL;
  if (io_nul(name, n) || n == 0 || n > 253) {
    free(name);
    w->data = NULL;
    return io_fail(e, EINVAL, NULL);
  }
  w->data = io_mem(realloc(name, n + INET_ADDRSTRLEN + 1));
  return io_work(w, dns_resolve_call, dns_resolve_pack);
}

static void __attribute__((constructor)) dns_resolve_use(void) {
  io_eff(CID_DNS_RESOLVE, dns_resolve_run, 0);
}
