// TLS
// ===

// As tls_listen.js: the JS lane speaks to the kernel through raw
// syscalls and has no TLS under it, so a TLS client is refused as
// plainly as a TLS listener (ENOSYS); a program that wants one is a
// program to build.
function tls_connect(addr, port, name, alpn, ca, ms) {
  return io_fail(38);
}
