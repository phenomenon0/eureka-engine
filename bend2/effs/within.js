// IO
// ==

// IO.within.go(ms, op), as within.c has it: op runs as a computation of
// its own, driven here so that its answer comes back -- at once as Some,
// or to the caller parked on the deadline, or dropped after it.
function io_within_go(ms, op, k) {
  const io = globalThis.BEND_IO;
  const at = performance.now() + Number(ms);
  let up = null, late = false, got = null;
  const end = (x) => {
    if (late) {
      return;
    }
    if (up === null) {
      got = { $: "Some", value: x };
      return;
    }
    late = true;
    io.waits = io.waits.filter((y) => y !== up);
    io_push(k, performance.now() < at ? { $: "Some", value: x } : { $: "None" }, false);
  };
  const drive = (o) => {
    while (o !== undefined) {
      o = run_loop(o);
      if (o.$ === "Emit") {
        end(o.value);
        return { $: "Emit", value: null };
      }
      if (o.$ === "Halt") {
        return o;
      }
      const q = o;
      const K = (x) => drive(q.kont(x));
      const need = q.need?.() ?? {};
      if (need.time || need.read) {
        io_park_on(need.read ? q.args[0] : undefined, false, K, () => q.run(...q.args, K),
          need.read ? undefined : performance.now() + Number(q.args[0]));
        return undefined;
      }
      const x = q.run(...q.args, K);
      if (x === undefined) {
        return undefined;
      }
      o = q.kont(x);
    }
    return undefined;
  };
  const r = drive(op);
  if (r?.$ === "Halt") {
    io_errs(r.message);
    globalThis.process?.exit(r.code);
  }
  if (got !== null) {
    return got;
  }
  io.live += 1;
  up = { fd: undefined, out: false, k, at, more: () => {
    late = true;
    return { $: "None" };
  } };
  io.waits.push(up);
  return undefined;
}
