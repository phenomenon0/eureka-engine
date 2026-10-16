// TCP
// ===

// TCP.send_buf_poll(sock, bytes, ms) is TCP.send_buf with a deadline: a
// full socket parks on the socket and on the clock, and a send that has
// made no progress for ms fails with ETIMEDOUT; each write that moves
// bytes starts the wait over.
function tcp_send_buf_poll(socket, data, ms, k) {
  const sys = io_sys();
  const fd = socket;
  const b = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i += 1) {
    const v = data.charCodeAt(i);
    if (v > 255) {
      return io_tup(socket, io_fail(22));
    }
    b[i] = v;
  }
  const again = sys.mac ? 35 : 11;
  const go = (at, due) => {
    while (at < b.length) {
      const part = b.subarray(at);
      const w = Number(sys.send(fd, sys.ptr(part), part.length, 0));
      if (w < 0) {
        const code = sys.errno();
        if (code !== again) {
          return io_tup(socket, io_fail(code));
        }
        const now = performance.now();
        const by = due ?? now + Number(ms);
        if (now >= by) {
          return io_tup(socket, io_fail(sys.mac ? 60 : 110));
        }
        io_park_on(fd, true, k, () => go(at, by), by);
        return undefined;
      }
      at += w;
      due = undefined;
    }
    return io_tup(socket, io_done({ $: "Unit" }));
  };
  return go(0, undefined);
}
