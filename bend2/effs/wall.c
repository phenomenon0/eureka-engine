// IO
// ==

// IO.wall: CLOCK_REALTIME in ms since the Unix epoch (a Nat word: 2^48 ms
// is the year 10889)
Term io_wall_run(Env e, Term* f, IoWork* w) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (Term)((u64)ts.tv_sec * 1000ull + (u64)ts.tv_nsec / 1000000ull);
}

static void __attribute__((constructor)) io_wall_use(void) {
  io_eff(CID_IO_WALL, io_wall_run, 0);
}
