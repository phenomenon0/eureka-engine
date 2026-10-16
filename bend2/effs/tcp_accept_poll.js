// TCP
// ===

// TCP.accept_poll(listener, ms) is TCP.accept with a deadline: an accept
// is tried first; nothing parks on the listener and on the clock; past
// the deadline it answers None{}. As in the C twin, only a listener that
// is no longer one fails it: an error that belongs to one connection is
// tried past, out of descriptors the waiting connection is shed on a
// spare one, and out of buffers the loop steps back for 20 ms.
let tcp_accept_poll_spare = -1;

function tcp_accept_poll(listener, ms, k) {
  const sys = io_sys();
  const lfd = listener;
  const at = performance.now() + Number(ms);
  const again = sys.mac ? 35 : 11;
  const [emfile, enfile, enobufs, enomem] = [24, 23, sys.mac ? 55 : 105, 12];
  const passing = sys.mac ? [53, 4, 100, 1, 55, 12, 24, 23, 50, 42, 64, 65, 102, 51]
    : [103, 4, 71, 1, 105, 12, 24, 23, 100, 92, 112, 64, 113, 95, 101];
  const none = () => io_tup(listener, io_done({ $: "None" }));
  if (tcp_accept_poll_spare < 0) {
    tcp_accept_poll_spare = sys.socket(2, 1, 0);
  }
  const shed = () => {
    if (tcp_accept_poll_spare < 0) {
      return false;
    }
    sys.close(tcp_accept_poll_spare);
    const fd = sys.accept(lfd, null, null);
    if (fd >= 0) {
      sys.close(fd);
    }
    tcp_accept_poll_spare = sys.socket(2, 1, 0);
    return fd >= 0;
  };
  const go = () => {
    for (let tries = 0; tries < 64; tries += 1) {
      const fd = sys.accept(lfd, null, null);
      if (fd >= 0) {
        if (sys.fcntl(fd, 4, sys.fcntl(fd, 3, 0) | (sys.mac ? 4 : 0x800)) < 0
          || sys.fcntl(fd, 2, 1) < 0) {
          sys.close(fd);
          continue;
        }
        sys.setsockopt(fd, 6, 1, sys.ptr(new Int32Array([1])), 4);
        return io_tup(listener, io_done({ $: "Some", value: fd }));
      }
      const code = sys.errno();
      if (code === again) {
        if (performance.now() >= at) {
          return none();
        }
        io_park_on(lfd, false, k, go, at);
        return undefined;
      }
      if (!passing.includes(code)) {
        return io_tup(listener, io_fail(code));
      }
      const out = code === emfile || code === enfile;
      if ((out && !shed()) || code === enobufs || code === enomem) {
        if (performance.now() >= at) {
          return none();
        }
        io_park_on(undefined, false, k, go, Math.min(at, performance.now() + 20));
        return undefined;
      }
    }
    return none();
  };
  return go();
}
