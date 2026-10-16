// TCP
// ===

// TCP.recv_bytes into a Bytes: a string of char codes 0..255, the bytes
// as they are, no text decoder anywhere near the socket.
function tcp_recv_buf(socket, max, k) {
  const sys = io_sys();
  const fd = socket;
  const b = new Uint8Array(Math.max(Number(max), 1));
  const again = sys.mac ? 35 : 11;
  const go = () => {
    const n = Number(sys.recv(fd, sys.ptr(b), Number(max), 0));
    if (n < 0) {
      const code = sys.errno();
      if (code === again) {
        io_park_on(fd, false, k, go);
        return undefined;
      }
      return io_tup(socket, io_fail(code));
    }
    let s = "";
    for (let i = 0; i < n; i += 8192) {
      s += String.fromCharCode.apply(null, b.subarray(i, Math.min(n, i + 8192)));
    }
    return io_tup(socket, io_done(s));
  };
  return go();
}

function tcp_recv_buf_need() {
  return { read: true };
}
