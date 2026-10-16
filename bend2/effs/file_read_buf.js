// File
// ====

// File.read_bytes into a Bytes: a string of char codes 0..255.
function file_read_buf(file, max) {
  const sys = io_sys();
  const fd = file;
  const len = Math.min(max, 2147483647);
  const b = new Uint8Array(Math.max(len, 1));
  const n = Number(sys.read(fd, sys.ptr(b), len));
  if (n < 0) {
    return io_tup(file, io_fail(sys.errno()));
  }
  let s = "";
  for (let i = 0; i < n; i += 8192) {
    s += String.fromCharCode.apply(null, b.subarray(i, Math.min(n, i + 8192)));
  }
  return io_tup(file, io_done(s));
}
