// Sys1.next / Sys1.done have no JS side: the engine is a CUDA build (sys1.c).
function sys1_next(dir, ctx) {
  return [];
}
function sys1_done(ok) {
}
