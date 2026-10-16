// TCP
// ===

// The list drained into one buffer, then written as tcp_send writes
// it; a value past 255 fails with EINVAL before anything goes out.
function tcp_send_bytes(socket, data, k) {
  const sys = io_sys();
  const fd = socket;
  let n = 0;
  for (let xs = data; xs.$ === "Con"; xs = xs.tail) {
    n += 1;
  }
  const b = new Uint8Array(n);
  let i = 0;
  for (let xs = data; xs.$ === "Con"; xs = xs.tail) {
    const v = Number(xs.head);
    if (v > 255) {
      return io_tup(socket, io_fail(22));
    }
    b[i++] = v;
  }
  const again = sys.mac ? 35 : 11;
  const go = (at) => {
    while (at < b.length) {
      const part = b.subarray(at);
      const w = Number(sys.send(fd, sys.ptr(part), part.length, 0));
      if (w < 0) {
        const code = sys.errno();
        if (code === again) {
          io_park_on(fd, true, k, () => go(at));
          return undefined;
        }
        return io_tup(socket, io_fail(code));
      }
      at += w;
    }
    return io_tup(socket, io_done({ $: "Unit" }));
  };
  return go(0);
}
