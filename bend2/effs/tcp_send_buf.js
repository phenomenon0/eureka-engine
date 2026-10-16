// TCP
// ===

// TCP.send_bytes from a Bytes: its char codes are the bytes; one past
// 255 fails with EINVAL before anything goes out.
function tcp_send_buf(socket, data, k) {
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
