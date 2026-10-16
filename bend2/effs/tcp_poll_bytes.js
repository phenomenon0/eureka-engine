// TCP
// ===

// TCP.poll_bytes(sock, max, ms) is TCP.poll carrying bytes, one List cell
// each: a recv is tried before any park; nothing parks on the socket and
// on the clock; past the deadline it answers None{}.
function tcp_poll_bytes(socket, max, ms, k) {
  const sys = io_sys();
  const fd = socket;
  const b = new Uint8Array(Math.max(Number(max), 1));
  const at = performance.now() + Number(ms);
  const again = sys.mac ? 35 : 11;
  const go = () => {
    const n = Number(sys.recv(fd, sys.ptr(b), Number(max), 0));
    if (n >= 0) {
      let xs = { $: "Nil" };
      for (let i = n; i > 0; i -= 1) {
        xs = { $: "Con", head: b[i - 1], tail: xs };
      }
      return io_tup(socket, io_done({ $: "Some", value: xs }));
    }
    const code = sys.errno();
    if (code !== again) {
      return io_tup(socket, io_fail(code));
    }
    if (performance.now() >= at) {
      return io_tup(socket, io_done({ $: "None" }));
    }
    io_park_on(fd, false, k, go, at);
    return undefined;
  };
  return go();
}
