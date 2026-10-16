// TCP
// ===

// TCP.poll_bytes into a Bytes, as tcp_recv_buf: a recv is tried before
// any park; nothing parks on the socket and on the clock; past the
// deadline it answers None{}.
function tcp_poll_buf(socket, max, ms, k) {
  const sys = io_sys();
  const fd = socket;
  const b = new Uint8Array(Math.max(Number(max), 1));
  const at = performance.now() + Number(ms);
  const again = sys.mac ? 35 : 11;
  const go = () => {
    const n = Number(sys.recv(fd, sys.ptr(b), Number(max), 0));
    if (n >= 0) {
      let s = "";
      for (let i = 0; i < n; i += 8192) {
        s += String.fromCharCode.apply(null, b.subarray(i, Math.min(n, i + 8192)));
      }
      return io_tup(socket, io_done({ $: "Some", value: s }));
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
