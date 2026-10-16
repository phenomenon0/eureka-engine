// Chan
// ====

// Chan.new, send, offer, recv, take and close share this file, and so does
// TCP.poll_buf_or (chan.c says how the bell works).

// A parked receiver holds CHAN_RECV: the C lane parks TERM_HOLE, and a
// program can make neither. A sent value may be null (an erased proof).
const CHAN_RECV = Symbol();

function chan_wake(row, x) {
  const w = row.wait.shift();
  io_push(w.cont, x, false);
  return w.item;
}

function chan_pull(row) {
  const v = row.ring.shift();
  if (row.wait.length > 0) {
    row.ring.push(chan_wake(row, true));
  }
  return v;
}

function chan_ring(row) {
  if (row.bell !== null) {
    row.bell.at = 0;
    row.bell = null;
  }
}

function chan_ready(row) {
  return row.ring.length > 0 || row.shut
    || row.wait.length > 0 && row.wait[0].item !== CHAN_RECV;
}

// A handle is the row (a stale copy keeps it, shut).
function chan_shut(row) {
  row.shut = true;
  chan_ring(row);
  while (row.wait.length > 0) {
    chan_wake(row, row.wait[0].item === CHAN_RECV ? { $: "None" } : false);
  }
}

function chan_new(room) {
  return { room: Number(room), ring: [], wait: [], shut: false, bell: null };
}

function chan_send(handle, value, k) {
  const row = handle;
  if (row.shut) {
    return false;
  }
  if (row.wait.length > 0 && row.wait[0].item === CHAN_RECV) {
    chan_wake(row, { $: "Some", value: value });
    return true;
  }
  chan_ring(row);
  if (row.ring.length < row.room) {
    row.ring.push(value);
    return true;
  }
  row.wait.push({ cont: k, item: value });
  return;
}

function chan_offer(handle, value) {
  const row = handle;
  if (!row.shut && row.wait.length > 0 && row.wait[0].item === CHAN_RECV) {
    chan_wake(row, { $: "Some", value: value });
    return true;
  }
  if (row.shut || row.ring.length >= row.room) {
    return false;
  }
  row.ring.push(value);
  chan_ring(row);
  return true;
}

function chan_take(handle) {
  const row = handle;
  if (row.ring.length > 0) {
    return { $: "Some", value: chan_pull(row) };
  }
  if (row.wait.length > 0 && row.wait[0].item !== CHAN_RECV) {
    return { $: "Some", value: chan_wake(row, true) };
  }
  return { $: "None" };
}

function tcp_poll_buf_or(socket, max, ms, chan, k) {
  const sys = io_sys();
  const fd = socket;
  const row = chan;
  const b = new Uint8Array(Math.max(Number(max), 1));
  const at = performance.now() + Number(ms);
  const again = sys.mac ? 35 : 11;
  let me = null;
  const end = (rang, r) => io_tup(socket, rang, r);
  const go = () => {
    if (row.bell === me) {
      row.bell = null;
    }
    const n = Number(sys.recv(fd, sys.ptr(b), Number(max), 0));
    if (n >= 0) {
      let s = "";
      for (let i = 0; i < n; i += 8192) {
        s += String.fromCharCode.apply(null, b.subarray(i, Math.min(n, i + 8192)));
      }
      return end(false, io_done({ $: "Some", value: s }));
    }
    const code = sys.errno();
    if (code !== again) {
      return end(false, io_fail(code));
    }
    if (chan_ready(row) || performance.now() >= at) {
      return end(chan_ready(row), io_done({ $: "None" }));
    }
    me = io_park_on(fd, false, k, go, at);
    row.bell = me;
    return undefined;
  };
  return go();
}

function chan_recv(handle, k) {
  const row = handle;
  if (row.ring.length > 0) {
    return { $: "Some", value: chan_pull(row) };
  }
  if (row.wait.length > 0 && row.wait[0].item !== CHAN_RECV) {
    return { $: "Some", value: chan_wake(row, true) };
  }
  if (row.shut) {
    return { $: "None" };
  }
  row.wait.push({ cont: k, item: CHAN_RECV });
  return;
}

function chan_close(handle) {
  const row = handle;
  if (!row.shut) {
    chan_shut(row);
  }
  return { $: "Unit" };
}
