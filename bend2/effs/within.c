// IO
// ==

// IO.within.go(ms, op): op, the rest of an act that has not answered at
// once, runs as a computation of its own, at once and before the caller
// goes on, until it answers or waits. An answer before it waits is Some;
// after, Some when it comes before the deadline, and else the caller
// wakes at the deadline with None, and the answer is dropped when it
// comes. Nothing stops act: an effect is the only place a computation
// yields, so one that computes without one holds the loop until it does.
// IoKid ::=
//   | IoKid(act, up, val, at, st)   st: 0 running, 1 answered at once, 2 late
// act's race is the kid itself; up the caller parked on the deadline.
typedef struct IoKid {
  IoAct  act;
  IoAct* up;
  Term   val;
  u64    at;
  u8     st;
} IoKid;

// a kid done with, kept for the next (act.next links them)
static IoAct* io_kids;

static void io_kid_free(IoKid* k) {
  k->act.next = io_kids;
  io_kids     = &k->act;
}

static Term io_within_late(Env e, IoWork* w) {
  ((IoKid*)w->data)->st = 2;
  return term_pak(CID_NONE, 0);
}

static Term io_within_ans(Env e, IoKid* k, Term v) {
  if (io_tick() < k->at) {
    return io_box(e, CID_SOME, v);
  }
  term_drop(e, v);
  return term_pak(CID_NONE, 0);
}

// act answered: kept for the caller still in io_within_run, handed to the
// caller parked on the deadline (taken off it), or dropped when late
static void io_within_end(Env e, IoAct* c, Term v) {
  IoKid* k = (IoKid*)c;
  if (k->st == 2) {
    term_drop(e, v);
    io_kid_free(k);
    return;
  }
  if (k->up == NULL) {
    k->val = v;
    k->st  = 1;
    return;
  }
  IoAct* p = k->up;
#ifdef __linux__
  io_time_drop(p);
#else
  IoAct** at = &io_park.head;
  IoAct*  pv = NULL;
  while (*at != p) {
    pv = *at;
    at = &pv->next;
  }
  *at = p->next;
  if (io_park.last == p) {
    io_park.last = pv;
  }
#endif
  p->item = io_within_ans(e, k, v);
  io_kid_free(k);
  io_push(&io_runs, p);
}

// op is act's first request (Base's IO.within took the steps before it)
Term io_within_go_run(Env e, Term* f, IoWork* w) {
  IoKid* k = (IoKid*)io_kids;
  if (k != NULL) {
    io_kids = k->act.next;
    memset(k, 0, sizeof *k);
  } else {
    k = io_mem(calloc(1, sizeof *k));
  }
  k->at       = io_tick() + (u64)f[0] * 1000000ull;
  k->act.race = k;
  io_live += 1;
  int code = io_step(e, &k->act, f[1]);
  if (code >= 0) {
    io_sync();
    exit(code);
  }
  if (k->st == 1) {
    Term x = io_box(e, CID_SOME, k->val);
    io_kid_free(k);
    return x;
  }
  k->up   = (IoAct*)w;
  w->data = (char*)k;
  return io_wait_on(w, 0, 0, k->at, io_within_late);
}

static void __attribute__((constructor)) io_within_go_use(void) {
  io_race = io_within_end;
  io_eff(CID_IO_WITHIN_GO, io_within_go_run, 0);
}
