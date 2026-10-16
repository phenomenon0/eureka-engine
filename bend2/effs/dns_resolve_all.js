// DNS
// ===

// DNS.resolve_all(name), as dns_resolve_all.c: getaddrinfo for stream
// addresses of either family, in its order, each once, at most 16, joined
// by commas, IPv6 ones as RFC 5952 writes them; a literal answers at
// once. The JS lane has no helper threads, so the resolver's wait is the
// loop's.
function dns_resolve_all(name) {
  if (typeof name !== "string" || name.length === 0 || name.length > 253
    || name.includes("\0")) {
    return io_fail(22);
  }
  if (io_addr(name, 0) !== null) {
    return io_done(name);
  }
  const lit = io_ip6(name);
  if (lit !== null) {
    return io_done(io_ip6_show(lit));
  }
  const ffi = require("bun:ffi");
  const mac = process.platform === "darwin";
  if (globalThis.BEND_GAI === undefined) {
    globalThis.BEND_GAI = ffi.dlopen(mac ? "libSystem.dylib" : "libc.so.6", {
      getaddrinfo: { args: ["ptr", "ptr", "ptr", "ptr"], returns: "i32" },
      freeaddrinfo: { args: ["ptr"], returns: "void" },
    }).symbols;
  }
  const gai = globalThis.BEND_GAI;
  const node = new TextEncoder().encode(name + "\0");
  const hint = new Int32Array(12);
  hint[2] = 1;
  const out = new BigUint64Array(1);
  const r = gai.getaddrinfo(ffi.ptr(node), null, ffi.ptr(hint), ffi.ptr(out));
  if (r !== 0 || out[0] === 0n) {
    return io_fail(2);
  }
  const [v4, v6] = [2, mac ? 30 : 10];
  const seen = [];
  for (let a = Number(out[0]); a !== 0 && seen.length < 16; a = Number(ffi.read.ptr(a, 40))) {
    const fam = ffi.read.i32(a, 4);
    const sa = ffi.read.ptr(a, mac ? 32 : 24);
    const ip = fam === v4 ? [4, 5, 6, 7].map((i) => ffi.read.u8(sa, i)).join(".")
      : fam === v6 ? io_ip6_show(new Uint8Array([...new Array(16).keys()].map((i) => ffi.read.u8(sa, 8 + i))))
      : null;
    if (ip !== null && !seen.includes(ip)) {
      seen.push(ip);
    }
  }
  gai.freeaddrinfo(Number(out[0]));
  return seen.length === 0 ? io_fail(2) : io_done(seen.join(","));
}
