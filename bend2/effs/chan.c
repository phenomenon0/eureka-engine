// Chan
// ====

// Chan.new, send, offer, recv, take and close share this file, and so does
// TCP.poll_buf_or, which waits on a socket and a channel at once: the
// rows live here, and each effect registers when the program uses it.

typedef struct {
  u32   gen;
  u32   next;
  u32   room;
  u32   size;
  u32   head;
  u32   live;
  u32   shut;
  Term*  ring;
  IoQue  wait;
  IoAct* bell;
} ChanRow;

// A channel is Data: its handle is copied and may outlive the row, so it
// names the row by index and generation, a freed row waits on a list and
// comes back one generation up, and a stale copy finds no row (closed).
static ChanRow* chan_rows;
static u32      chan_len;
static u32      chan_idle = ~0u;

#define chan_some(e, v) io_box(e, CID_SOME, v)
#define chan_bool(b)    term_pak((b) ? CID_TRUE : CID_FALSE, 0)

static Term chan_open(u32 room) {
  u32 i = chan_idle;
  if (i != ~0u) {
    chan_idle = chan_rows[i].next;
  } else {
    if (chan_len == 1u << 24) {
      err_fail("more than 16777216 channels at once");
    }
    if ((chan_len & (chan_len - 1)) == 0) {
      chan_rows = io_mem(realloc(chan_rows,
        (chan_len == 0 ? 1 : 2 * chan_len) * sizeof(ChanRow)));
    }
    i = chan_len;
    chan_len += 1;
    chan_rows[i].gen = 0;
  }
  ChanRow* row = &chan_rows[i];
  *row = (ChanRow){ .gen = row->gen + 1, .room = room, .live = 1,
    .ring = room == 0 ? NULL : io_mem(malloc(room * sizeof(Term))) };
  return io_hand(((u64)row->gen << 24) | i);
}

static ChanRow* chan_at(Term t) {
  u64      v   = io_hand_v(t);
  u32      i   = (u32)v & 0xFFFFFF;
  ChanRow* row = i < chan_len ? &chan_rows[i] : NULL;
  return row != NULL && row->live && row->gen == (u32)(v >> 24) ? row : NULL;
}

// Parks the effect's activation on row with item: a sent value, or
// TERM_HOLE for a receiver.
static Term chan_park(ChanRow* row, IoWork* w, Term item) {
  IoAct* a = (IoAct*)w;
  a->item  = item;
  io_push(&row->wait, a);
  return IO_PARK;
}

static Term chan_wake(ChanRow* row, Term x) {
  IoAct* a  = io_pop(&row->wait);
  Term item = a->item;
  a->item   = x;
  io_push(&io_runs, a);
  return item;
}

static Term chan_take(ChanRow* row) {
  Term v = row->ring[row->head];
  row->head = (row->head + 1) % row->room;
  row->size -= 1;
  if (row->wait.head != NULL) {
    Term item = chan_wake(row, chan_bool(true));
    row->ring[(row->head + row->size) % row->room] = item;
    row->size += 1;
  }
  return v;
}

// A bell is an activation parked elsewhere (TCP.poll_buf_or, on its
// socket) until the channel has something for a receiver: a value
// arriving, or the close, wakes it (io_ring) and it looks again. The
// value stays in the channel, so nothing is lost to the wait.
static void chan_ring(ChanRow* row) {
  if (row->bell != NULL) {
    io_ring(row->bell);
    row->bell = NULL;
  }
}

// whether a receive would answer at once
static bool chan_ready(ChanRow* row) {
  return row == NULL || row->size > 0 || row->shut
    || (row->wait.head != NULL && row->wait.head->item != TERM_HOLE);
}

static void chan_free(ChanRow* row) {
  free(row->ring);
  row->live = 0;
  row->next = chan_idle;
  chan_idle = (u32)(row - chan_rows);
}

static void chan_shut(Env e, ChanRow* row) {
  row->shut = 1;
  chan_ring(row);
  while (row->wait.head != NULL) {
    bool rcv = row->wait.head->item == TERM_HOLE;
    Term x = rcv ? term_pak(CID_NONE, 0) : chan_bool(false);
    term_sink(e, chan_wake(row, x));
  }
  if (row->size == 0) {
    chan_free(row);
  }
}

#ifdef CID_CHAN_NEW

Term chan_new_run(Env e, Term* f, IoWork* w) {
  return chan_open((u32)f[0]);
}

static void __attribute__((constructor)) chan_new_use(void) {
  io_eff(CID_CHAN_NEW, chan_new_run, 0);
}

#endif

#ifdef CID_CHAN_SEND

Term chan_send_run(Env e, Term* f, IoWork* w) {
  ChanRow* row = chan_at(f[0]);
  if (row == NULL || row->shut) {
    term_drop(e, f[1]);
    return chan_bool(false);
  }
  if (row->wait.head != NULL && row->wait.head->item == TERM_HOLE) {
    chan_wake(row, chan_some(e, f[1]));
    return chan_bool(true);
  }
  chan_ring(row);
  if (row->size < row->room) {
    row->ring[(row->head + row->size) % row->room] = f[1];
    row->size += 1;
    return chan_bool(true);
  }
  return chan_park(row, w, f[1]);
}

static void __attribute__((constructor)) chan_send_use(void) {
  io_eff(CID_CHAN_SEND, chan_send_run, 0);
}

#endif

#ifdef CID_CHAN_OFFER

// Chan.send that never waits: a value with nowhere to go (the ring full,
// no receiver waiting, or the channel closed) is dropped, and False.
Term chan_offer_run(Env e, Term* f, IoWork* w) {
  ChanRow* row = chan_at(f[0]);
  if (row != NULL && !row->shut && row->wait.head != NULL
      && row->wait.head->item == TERM_HOLE) {
    chan_wake(row, chan_some(e, f[1]));
    return chan_bool(true);
  }
  if (row == NULL || row->shut || row->size == row->room) {
    term_drop(e, f[1]);
    return chan_bool(false);
  }
  row->ring[(row->head + row->size) % row->room] = f[1];
  row->size += 1;
  chan_ring(row);
  return chan_bool(true);
}

static void __attribute__((constructor)) chan_offer_use(void) {
  io_eff(CID_CHAN_OFFER, chan_offer_run, 0);
}

#endif

#ifdef CID_CHAN_TAKE

// Chan.recv that never waits: None{} when nothing is there now.
Term chan_take_run(Env e, Term* f, IoWork* w) {
  ChanRow* row = chan_at(f[0]);
  if (row != NULL && row->size > 0) {
    Term v = chan_take(row);
    if (row->shut && row->size == 0) {
      chan_free(row);
    }
    return chan_some(e, v);
  }
  if (row != NULL && row->wait.head != NULL && row->wait.head->item != TERM_HOLE) {
    return chan_some(e, chan_wake(row, chan_bool(true)));
  }
  return term_pak(CID_NONE, 0);
}

static void __attribute__((constructor)) chan_take_use(void) {
  io_eff(CID_CHAN_TAKE, chan_take_run, 0);
}

#endif

#ifdef CID_TCP_POLL_BUF_OR

// TCP.poll_buf_or(sock, max, ms, chan) is TCP.poll_buf that also ends,
// with rang True and Done{None{}}, when chan has something for a
// receiver: it takes nothing from chan. Bytes win when both are there.
// The deadline and the channel live in text, since a ring moves the
// park's own deadline to now.
typedef struct {
  u64  at;
  Term chan;
} ChanWait;

static Term chan_poll_end(Env e, IoWork* w, bool rang, Term r) {
  free(w->data);
  free(w->text);
  return io_tup(e, io_hand(w->hand), io_tup(e, chan_bool(rang), r));
}

static Term chan_poll_at(Env e, IoWork* w) {
  ChanWait* cw  = (ChanWait*)w->text;
  ChanRow*  row = chan_at(cw->chan);
  int       fd  = (int)w->hand;
  short     dir = POLLIN;
  bool      raw = io_wire == NULL;
  if (row != NULL && row->bell == (IoAct*)w) {
    row->bell = NULL;
  }
  // quiet is the poller's belief, which lags a wake by a pass: with the
  // channel ready the socket is asked, so bytes that are there win
  if (raw && io_fd_quiet(fd, 1) && !chan_ready(row)) {
    w->size = 0;
    w->code = EAGAIN;
  } else {
    w->size = io_sys_end(w, io_wire_read(fd, w->data, (size_t)w->made, &dir));
    if (raw && w->code == 0 && w->size > 0) {
      io_fd_seen(fd, 1, w->size >= (u64)w->made);
    }
  }
  if (w->code != EAGAIN) {
    return chan_poll_end(e, w, false, w->code ? io_fail(e, w->code, NULL)
      : io_done(e, io_box(e, CID_SOME, io_buf(e, w->data, w->size))));
  }
  if (chan_ready(row) || io_tick() >= cw->at) {
    return chan_poll_end(e, w, chan_ready(row), io_done(e, term_pak(CID_NONE, 0)));
  }
  row->bell = (IoAct*)w;
  return io_wait_on(w, fd, dir, cw->at, chan_poll_at);
}

Term tcp_poll_buf_or_run(Env e, Term* f, IoWork* w) {
  ChanWait* cw = io_mem(malloc(sizeof(ChanWait)));
  cw->at   = io_tick() + (u64)f[2] * 1000000ull;
  cw->chan = f[3];
  w->hand  = (intptr_t)io_hand_v(f[0]);
  w->made  = f[1] < INT32_MAX ? (intptr_t)f[1] : INT32_MAX;
  w->data  = io_mem(malloc((size_t)w->made + 1));
  w->text  = (char*)cw;
  return chan_poll_at(e, w);
}

static void __attribute__((constructor)) tcp_poll_buf_or_use(void) {
  io_eff(CID_TCP_POLL_BUF_OR, tcp_poll_buf_or_run, 0);
}

#endif

#ifdef CID_CHAN_RECV

Term chan_recv_run(Env e, Term* f, IoWork* w) {
  ChanRow* row = chan_at(f[0]);
  if (row == NULL) {
    return term_pak(CID_NONE, 0);
  }
  if (row->size > 0) {
    Term v = chan_take(row);
    if (row->shut && row->size == 0) {
      chan_free(row);
    }
    return chan_some(e, v);
  }
  if (row->wait.head != NULL && row->wait.head->item != TERM_HOLE) {
    return chan_some(e, chan_wake(row, chan_bool(true)));
  }
  if (row->shut) {
    chan_free(row);
    return term_pak(CID_NONE, 0);
  }
  return chan_park(row, w, TERM_HOLE);
}

static void __attribute__((constructor)) chan_recv_use(void) {
  io_eff(CID_CHAN_RECV, chan_recv_run, 0);
}

#endif

#ifdef CID_CHAN_CLOSE

Term chan_close_run(Env e, Term* f, IoWork* w) {
  ChanRow* row = chan_at(f[0]);
  if (row != NULL && !row->shut) {
    chan_shut(e, row);
  }
  return term_pak(CID_UNIT, 0);
}

static void __attribute__((constructor)) chan_close_use(void) {
  io_eff(CID_CHAN_CLOSE, chan_close_run, 0);
}

#endif
