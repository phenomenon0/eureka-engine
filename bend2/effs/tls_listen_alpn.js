// TLS
// ===

// As tls_listen.js: the JS lane has no TLS under it, so a listener that
// agrees by ALPN is refused as plainly as one that does not.
function tls_listen_alpn(port, cert, key, names) {
  return io_fail(38);
}
