// Gguf.read has no JS side: the engine is a CUDA build (host/llb_gguf.c).
function gguf_read(which, off, n) {
  return io_fail(95);  // EOPNOTSUPP
}
