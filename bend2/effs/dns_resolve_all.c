// DNS
// ===

// DNS.resolve_all(name): every address the system's resolver has for the
// name, IPv6 and IPv4, in the order getaddrinfo answers them (its RFC
// 6724 sort: the order to try them in), each once, at most 16, joined by
// commas: "::1,127.0.0.1". An IPv6 address is in inet_ntop's text form
// (RFC 5952's), a literal is its own answer, and a name with none is
// ENOENT with the resolver's reason. As DNS.resolve, the resolver blocks,
// so it runs on a helper thread and the loop goes on.
#include <netdb.h>

#define DNS_ALL_MAX 16

static void dns_resolve_all_call(IoWork* w) {
  struct addrinfo  hint = { 0 };
  struct addrinfo* got  = NULL;
  hint.ai_family   = AF_UNSPEC;
  hint.ai_socktype = SOCK_STREAM;
  int r = getaddrinfo(w->data, NULL, &hint, &got);
  if (r != 0 || got == NULL) {
    w->code = ENOENT;
    w->text = (char*)gai_strerror(r);
    if (got != NULL) {
      freeaddrinfo(got);
    }
    return;
  }
  char   out[DNS_ALL_MAX * (INET6_ADDRSTRLEN + 1)];
  u64    n = 0;
  u32    k = 0;
  out[0] = 0;
  for (struct addrinfo* a = got; a != NULL && k < DNS_ALL_MAX; a = a->ai_next) {
    char one[INET6_ADDRSTRLEN];
    const void* at = a->ai_family == AF_INET6
      ? (const void*)&((struct sockaddr_in6*)a->ai_addr)->sin6_addr
      : a->ai_family == AF_INET ? (const void*)&((struct sockaddr_in*)a->ai_addr)->sin_addr : NULL;
    if (at == NULL || inet_ntop(a->ai_family, at, one, sizeof(one)) == NULL) {
      continue;
    }
    // once each: the same address as ",one," in ",out,"
    u64  m = strlen(one);
    bool seen = false;
    for (u64 i = 0; i + m <= n && !seen; i += 1) {
      seen = (i == 0 || out[i - 1] == ',') && memcmp(out + i, one, m) == 0
        && (i + m == n || out[i + m] == ',');
    }
    if (seen) {
      continue;
    }
    if (n != 0) {
      out[n++] = ',';
    }
    memcpy(out + n, one, m + 1);
    n += m;
    k += 1;
  }
  freeaddrinfo(got);
  if (n == 0) {
    w->code = ENOENT;
    w->text = "no address of a family the system speaks";
    return;
  }
  free(w->data);
  w->data = io_mem(malloc(n + 1));
  memcpy(w->data, out, n + 1);
  w->code = 0;
  w->size = n;
}

static Term dns_resolve_all_pack(Env e, IoWork* w) {
  Term r = w->code != 0 ? io_fail(e, w->code, w->text)
    : io_done(e, io_str(e, w->data, w->size));
  free(w->data);
  return r;
}

Term dns_resolve_all_run(Env e, Term* f, IoWork* w) {
  u64   n    = 0;
  char* name = io_cstr(e, f[0], &n);
  w->text = NULL;
  if (io_nul(name, n) || n == 0 || n > 253) {
    free(name);
    w->data = NULL;
    return io_fail(e, EINVAL, NULL);
  }
  w->data = name;
  return io_work(w, dns_resolve_all_call, dns_resolve_all_pack);
}

static void __attribute__((constructor)) dns_resolve_all_use(void) {
  io_eff(CID_DNS_RESOLVE_ALL, dns_resolve_all_run, 0);
}
