// TCP
// ===

// TCP.connect with a deadline, as tcp_connect.js: EINPROGRESS parks the
// computation on the socket becoming writable and on the clock; writable,
// SO_ERROR says how the connect ended; the clock first, ETIMEDOUT.
function tcp_connect_poll(host, port, ms, k) {
  const sys = io_sys();
  const at = io_sa(host, Number(port));
  if (at === null) {
    return io_fail(22);
  }
  const fd = sys.socket(at.fam, 1, 0);
  if (fd < 0) {
    return io_fail(sys.errno());
  }
  const end = (code) => {
    if (code !== 0) {
      sys.close(fd);
      return io_fail(code);
    }
    return io_done(fd);
  };
  const error = () => {
    const v = new Int32Array([0]);
    const l = new Uint32Array([4]);
    return sys.getsockopt(fd, sys.mac ? 0xffff : 1, sys.mac ? 0x1007 : 4,
      sys.ptr(v), sys.ptr(l)) < 0 ? sys.errno() : v[0];
  };
  // writable now: a zero-time select on the one descriptor
  const ready = () => {
    const len = (fd >> 6 << 3) + 8;
    const set = new Uint8Array(len);
    set[fd >> 3] |= 1 << (fd & 7);
    const tv = new BigInt64Array([0n, 0n]);
    sys.select(fd + 1, null, sys.ptr(set), null, sys.ptr(tv));
    return (set[fd >> 3] & 1 << (fd & 7)) !== 0;
  };
  const set = sys.fcntl(fd, 4, sys.fcntl(fd, 3, 0) | (sys.mac ? 4 : 0x800));
  const ok = set >= 0 && sys.connect(fd, sys.ptr(at.b), at.b.length) >= 0;
  const code = ok ? 0 : sys.errno();
  if (code !== (sys.mac ? 36 : 115)) {
    return end(code);
  }
  const due = performance.now() + Math.max(Number(ms), 1);
  const more = () => {
    if (ready()) {
      return end(error());
    }
    if (performance.now() >= due) {
      return end(sys.mac ? 60 : 110);
    }
    io_park_on(fd, true, k, more, due);
    return undefined;
  };
  io_park_on(fd, true, k, more, due);
  return undefined;
}
