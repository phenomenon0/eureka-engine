// TCP
// ===

// The bytes as they are (0..255), one List cell each, as
// File.write_bytes takes them; a value past 255 is not a byte and
// fails with EINVAL before anything goes out. TCP.send takes a String
// and re-encodes it as UTF-8, so a binary body sent that way leaves
// mangled: a writer of one needs this instead.
static Term tcp_send_bytes_pack(Env e, IoWork* w) {
  Term r = w->code != 0 ? io_fail(e, w->code, NULL)
    : io_done(e, term_pak(CID_UNIT, 0));
  free(w->data);
  return io_tup(e, io_hand(w->hand), r);
}

// Sends what is left; a full socket (non-blocking, so EAGAIN) parks
// the computation until the socket is writable, and resumes here.
static Term tcp_send_bytes_more(Env e, IoWork* w) {
  int   fd  = (int)w->hand;
  short dir = POLLOUT;
  while (w->code == 0 && (u64)w->made < w->size) {
    ssize_t n = io_wire_write(fd, w->data + w->made, w->size - (u64)w->made, &dir);
    if (n < 0 && errno == EAGAIN) {
      return io_wait_on(w, fd, dir, 0, tcp_send_bytes_more);
    }
    w->made += io_sys_end(w, n);
  }
  return tcp_send_bytes_pack(e, w);
}

Term tcp_send_bytes_run(Env e, Term* f, IoWork* w) {
  u64  cap = 256;
  Term xs  = f[1];
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->code = 0;
  w->size = 0;
  w->made = 0;
  w->data = io_mem(malloc(cap));
  while (term_aux(xs) == CID_CON) {
    Term fb[2];
    spare_free(e, cls_fit(2), ctr_take(e, xs, 2, fb));
    if (w->size == cap) {
      cap *= 2;
      w->data = io_mem(realloc(w->data, cap));
    }
    w->code = fb[0] > 255 ? EINVAL : w->code;
    w->data[w->size++] = (char)fb[0];
    xs = fb[1];
  }
  return w->code ? tcp_send_bytes_pack(e, w) : tcp_send_bytes_more(e, w);
}

static void __attribute__((constructor)) tcp_send_bytes_use(void) {
  io_eff(CID_TCP_SEND_BYTES, tcp_send_bytes_run, 0);
}
