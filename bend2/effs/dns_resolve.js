// DNS
// ===

// DNS.resolve(name), as dns_resolve.c: getaddrinfo for an IPv4 stream
// address, the first one dotted. The JS lane has no helper threads, so
// the resolver's wait is the loop's; a dotted address answers at once.
function dns_resolve(name) {
  if (typeof name !== "string" || name.length === 0 || name.length > 253
    || name.includes("\0")) {
    return io_fail(22);
  }
  if (io_addr(name, 0) !== null) {
    return io_done(name);
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
  hint[1] = 2;
  hint[2] = 1;
  const out = new BigUint64Array(1);
  const r = gai.getaddrinfo(ffi.ptr(node), null, ffi.ptr(hint), ffi.ptr(out));
  if (r !== 0 || out[0] === 0n) {
    return io_fail(2);
  }
  const head = Number(out[0]);
  const addr = ffi.read.ptr(head, mac ? 32 : 24);
  const ip = [4, 5, 6, 7].map((i) => ffi.read.u8(addr, i)).join(".");
  gai.freeaddrinfo(head);
  return io_done(ip);
}
