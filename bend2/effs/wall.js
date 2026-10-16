// IO
// ==

// IO.wall: the wall clock in ms since the Unix epoch
function io_wall() {
  return BigInt(Date.now());
}
