// File
// ====

// path under root, as file_open_under.c: each component is lstat'ed
// before the next, so a symbolic link anywhere under root fails with
// ELOOP; an empty, "." or ".." component with EACCES; an end that is
// not a regular file with EISDIR or EACCES. The end is opened with
// O_NOFOLLOW and checked again on the open descriptor. The answer is
// [fd, stat] or a Fail.
function file_open_under_at(root, path) {
  const fs = require("fs");
  const mac = process.platform === "darwin";
  const top = io_bytes(root);
  const rel = io_bytes(path);
  if (top.includes(0) || rel.includes(0)) {
    return io_fail(mac ? 92 : 84);
  }
  const parts = [[]];
  for (const b of rel) {
    b === 47 ? parts.push([]) : parts[parts.length - 1].push(b);
  }
  let at = Buffer.from(top);
  try {
    if (!fs.statSync(at).isDirectory()) {
      return io_fail(20);
    }
    for (let i = 0; i < parts.length; i += 1) {
      const p = parts[i];
      if (p.length === 0 || (p[0] === 46 && (p.length === 1
        || (p.length === 2 && p[1] === 46)))) {
        return io_fail(13);
      }
      at = Buffer.concat([at, Buffer.from([47]), Buffer.from(p)]);
      const st = fs.lstatSync(at);
      if (st.isSymbolicLink()) {
        return io_fail(mac ? 62 : 40);
      }
      if (i < parts.length - 1 && !st.isDirectory()) {
        return io_fail(20);
      }
    }
    const c = fs.constants;
    const fd = fs.openSync(at, c.O_RDONLY | c.O_NOFOLLOW | c.O_NONBLOCK);
    const st = fs.fstatSync(fd);
    if (!st.isFile()) {
      fs.closeSync(fd);
      return io_fail(st.isDirectory() ? 21 : 13);
    }
    return [fd, st];
  } catch (e) {
    return io_fail(Math.abs(e.errno ?? 5));
  }
}

function file_open_under(root, path) {
  const got = file_open_under_at(root, path);
  return Array.isArray(got) ? io_done(got[0]) : got;
}

// File.get_under(root, path, small), as file_open_under.c: the open,
// its size and mtime, and a file under small bytes read whole and
// closed (None and its bytes) or a larger one left open (Some, "").
function file_get_under(root, path, small) {
  const fs = require("fs");
  const got = file_open_under_at(root, path);
  if (!Array.isArray(got)) {
    return got;
  }
  const [fd, st] = got;
  const n = st.size;
  if (n > 4294967295) {
    fs.closeSync(fd);
    return io_fail(process.platform === "darwin" ? 84 : 75);
  }
  const mt = Math.floor(st.mtimeMs / 1000) >>> 0;
  if (n >= Number(small)) {
    return io_done(io_tup(n, mt, { $: "Some", value: fd }, ""));
  }
  const b = new Uint8Array(Math.max(n, 1));
  let k = 0;
  try {
    for (let r = 1; k < n && r > 0; k += r) {
      r = fs.readSync(fd, b, k, n - k, k);
    }
  } catch (e) {
    fs.closeSync(fd);
    return io_fail(Math.abs(e.errno ?? 5));
  }
  fs.closeSync(fd);
  let s = "";
  for (let i = 0; i < k; i += 8192) {
    s += String.fromCharCode.apply(null, b.subarray(i, Math.min(k, i + 8192)));
  }
  return io_done(io_tup(n, mt, { $: "None" }, s));
}
