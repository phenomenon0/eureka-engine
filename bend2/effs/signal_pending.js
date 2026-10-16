// IO
// ==

// IO.signal_pending(sig) answers whether the signal has arrived since it
// was last asked; the first ask installs the handler that notes it.
// IO.signal_seen(sig) answers whether it has arrived at all since then,
// and clears nothing. io_run never returns to bun's event loop, so a
// process.on handler would never run: the handler is C, as on the C lane
// (bun:ffi's cc builds it the first time a signal is asked for), and it
// only counts; pending is the count moved since the last ask. Where cc
// cannot build (a platform TinyCC lacks), process.on is the fallback.
const io_sig_names = { 1: "SIGHUP", 2: "SIGINT", 10: "SIGUSR1", 12: "SIGUSR2", 15: "SIGTERM" };
const io_sig_js = {};
const io_sig_last = {};
let io_sig_lib;

function io_sig_native() {
  if (io_sig_lib === undefined) {
    io_sig_lib = null;
    const fs = require("fs");
    let dir = null;
    try {
      dir = fs.mkdtempSync(require("path").join(require("os").tmpdir(), "bend-sig-"));
      // no headers: signal() has the BSD semantics (the handler stays,
      // syscalls restart) on glibc and Darwin alike
      fs.writeFileSync(dir + "/sig.c", "typedef void (*io_h)(int);\n"
        + "io_h signal(int, io_h);\n"
        + "static volatile unsigned io_n[65];\n"
        + "static void io_note(int s) { if (s > 0 && s < 65) io_n[s] += 1; }\n"
        + "int io_hook(int s) { return signal(s, io_note) == (io_h)-1 ? -1 : 0; }\n"
        + "unsigned io_count(int s) { return io_n[s]; }\n");
      io_sig_lib = require("bun:ffi").cc({ source: dir + "/sig.c", symbols: {
        io_hook: { args: ["i32"], returns: "i32" },
        io_count: { args: ["i32"], returns: "u32" } } });
    } catch (e) {
      io_sig_lib = null;
    } finally {
      if (dir !== null) {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    }
  }
  return io_sig_lib;
}

// how often the signal has arrived since it was first caught, or -1
function io_sig_count(sig) {
  const s = Number(sig);
  if (!(s in io_sig_last)) {
    const lib = s >= 1 && s <= 64 ? io_sig_native() : null;
    if (lib === null || lib.symbols.io_hook(s) !== 0) {
      if (io_sig_names[s] === undefined) {
        return -1;
      }
      io_sig_js[s] = 0;
      process.on(io_sig_names[s], () => { io_sig_js[s] += 1; });
    }
    io_sig_last[s] = 0;
  }
  return s in io_sig_js ? io_sig_js[s] : io_sig_lib.symbols.io_count(s);
}

function io_signal_pending(sig) {
  const n = io_sig_count(sig);
  if (n < 0) {
    return false;
  }
  const moved = n !== io_sig_last[Number(sig)];
  io_sig_last[Number(sig)] = n;
  return moved;
}

function io_signal_seen(sig) {
  return io_sig_count(sig) > 0;
}
