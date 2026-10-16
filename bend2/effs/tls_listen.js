// TLS
// ===

// The JS lane speaks to the kernel through raw syscalls and has no TLS
// under it, so this says so rather than pretending: a program that
// wants TLS is a program to build, not to interpret.
function tls_listen(port, cert, key) {
  return io_fail(38);
}

function tls_listen_on(host, port, cert, key) {
  return io_fail(38);
}
