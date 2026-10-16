// File
// ====

// File.sendfile(sock, file, off, len, ms): the file's bytes from off, len
// of them, onto the socket, each piece within ms of the last; Done once
// all len are out, and ENODATA when the file ended first.
// Here it is a pread of a block at a time and a send of it, parking on a
// full socket as TCP.send_buf_poll does; the file's position does not
// move.
function file_sendfile(socket, file, off, len, ms, k) {
  const sys = io_sys();
  const again = sys.mac ? 35 : 11;
  const b = new Uint8Array(65536);
  let pos = Number(off);
  const end = pos + Number(len);
  let have = 0;
  let at = 0;
  const done = (r) => io_tup(socket, io_tup(file, r));
  const go = (due) => {
    while (pos < end) {
      if (at === have) {
        const want = Math.min(b.length, end - pos);
        const n = Number(sys.pread(file, sys.ptr(b), want, BigInt(pos)));
        if (n < 0) {
          return done(io_fail(sys.errno()));
        }
        if (n === 0) {
          return done(io_fail(sys.mac ? 96 : 61));
        }
        have = n;
        at = 0;
      }
      const part = b.subarray(at, have);
      const w = Number(sys.send(socket, sys.ptr(part), part.length, 0));
      if (w < 0) {
        const code = sys.errno();
        if (code !== again) {
          return done(io_fail(code));
        }
        const now = performance.now();
        const by = due ?? now + Number(ms);
        if (now >= by) {
          return done(io_fail(sys.mac ? 60 : 110));
        }
        io_park_on(socket, true, k, () => go(by), by);
        return undefined;
      }
      at += w;
      pos += w;
      due = undefined;
    }
    return done(io_done({ $: "Unit" }));
  };
  return go(undefined);
}
