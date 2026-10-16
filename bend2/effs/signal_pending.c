// IO
// ==

// chan_bool lives in effs/chan.c since 2.0.26, which only a channel program
// carries, so this effect spells its Bool itself.
#define sig_bool(b) term_pak((b) ? CID_TRUE : CID_FALSE, 0)

// IO.signal_pending(sig) answers whether the signal has arrived since it
// was last asked, and takes the first ask as the request to catch it: the
// handler only sets a flag, so the default (for SIGTERM, to die at once)
// becomes the program's own answer, given when it next looks. Syscalls
// restart across it, so an effect in flight is not failed by a signal.
// IO.signal_seen(sig) answers whether it has arrived at all since it was
// first caught, and clears nothing: every computation that asks after it
// came hears it, whoever asked first (a server's connections, each on
// its own, after its accept loop took the pending flag).
static volatile sig_atomic_t io_sig_seen[65];
static volatile sig_atomic_t io_sig_ever[65];
static volatile sig_atomic_t io_sig_hook[65];

static void io_sig_note(int sig) {
  if (sig > 0 && sig < 65) {
    io_sig_seen[sig] = 1;
    io_sig_ever[sig] = 1;
  }
}

static void io_sig_catch(u32 sig) {
  if (!io_sig_hook[sig]) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = io_sig_note;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction((int)sig, &sa, NULL);
    io_sig_hook[sig] = 1;
  }
}

Term io_signal_pending_run(Env e, Term* f, IoWork* w) {
  u32 sig = (u32)f[0];
  if (sig == 0 || sig >= 65) {
    return sig_bool(false);
  }
  io_sig_catch(sig);
  // read and cleared in one step, so a signal landing between the two
  // is not lost
  return sig_bool(__atomic_exchange_n(&io_sig_seen[sig], 0, __ATOMIC_SEQ_CST) != 0);
}

Term io_signal_seen_run(Env e, Term* f, IoWork* w) {
  u32 sig = (u32)f[0];
  if (sig == 0 || sig >= 65) {
    return sig_bool(false);
  }
  io_sig_catch(sig);
  return sig_bool(__atomic_load_n(&io_sig_ever[sig], __ATOMIC_SEQ_CST) != 0);
}

static void __attribute__((constructor)) io_signal_pending_use(void) {
#ifdef CID_IO_SIGNAL_PENDING
  io_eff(CID_IO_SIGNAL_PENDING, io_signal_pending_run, 0);
#endif
#ifdef CID_IO_SIGNAL_SEEN
  io_eff(CID_IO_SIGNAL_SEEN, io_signal_seen_run, 0);
#endif
}
