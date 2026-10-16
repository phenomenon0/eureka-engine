// TCP
// ===

// TCP.idle(sock), as tcp_idle.c: a zero-time select on the one
// descriptor; readable (a byte, the peer's FIN, an error) is not idle.
function tcp_idle(socket) {
  const sys = io_sys();
  const fd = socket;
  const len = (fd >> 6 << 3) + 8;
  const set = new Uint8Array(len);
  set[fd >> 3] |= 1 << (fd & 7);
  const tv = new BigInt64Array([0n, 0n]);
  const n = sys.select(fd + 1, sys.ptr(set), null, null, sys.ptr(tv));
  return io_tup(socket, n === 0);
}
