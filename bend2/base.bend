# Types
# =====

# Data
# ----

type Empty is Data:

type Unit is Data:
  Unit{}

type Bool is Data:
  False{}
  True{}

type Cmp is Data:
  LT{}
  EQ{}
  GT{}

type Either<a, b, -A: Kind(a), -B: Kind(b)> is Kind(a <&> b):
  Inl{value: A}
  Inr{value: B}

type Sigma<a, b, -A: Kind(a), -B: @-x: A -> Kind(b)> is Kind(a <&> b):
  Tuple{fst: A, snd: B(fst)}

type Nat is Data:
  Zero{}
  Succ{pred: Nat}

type Maybe<a, -A: Kind(a)> is Kind(a):
  None{}
  Some{value: A}

type Result<a, b, -E: Kind(a), -A: Kind(b)> is Kind(a <&> b):
  Fail{error: E}
  Done{value: A}

type List<a, -A: Kind(a)> is Kind(a):
  Nil{}
  Con{head: A, tail: List<a, A>}

law Word:
  for n: Nat
  Data

type Word.Nil is Data:
  WNil{}

type Word.Con<-p: Nat> is Data:
  WCon{head: Bool, tail: Word(p)}

type U32 is Data:
  U32{data: Word(32n)}

type F32 is Data:
  F32{data: Word(32n)}

type F64 is Data:
  F64{data: Word(64n)}

type Char is Data:
  Chr{code: U32}

type String is Data:
  SNil{}
  SCon{head: Char, tail: String}

type Array<-T: Type> is Type:
  ALeaf{value: T}
  ANode{xs: Array<T>, ys: Array<T>}

type Image is Data:
  Pix{color: U32}
  Qua{tl: Image, tr: Image, bl: Image, br: Image}

type Event is Data:
  Key{code: U32, down: Bool}
  Mouse{x: U32, y: U32, button: U32, down: Bool}
  Move{x: U32, y: U32}
  Close{}

type Map<a, -V: Kind(a)> is Kind(a):
  MTip{}
  MLeaf{key: String, val: V}
  MNode{pos: Nat, lo: Map<a, V>, hi: Map<a, V>}

# A String below 256 in every Char: see Bytes under String
def Bytes() -> Data:
  String

# The shapes the Regex types share: a code point range, a class, a span
def Regex.Range() -> Data:
  Sigma<&2, &2, U32, _ => U32>

def Regex.Ranges() -> Data:
  List<&2, Regex.Range()>

def Regex.Span() -> Data:
  Sigma<&2, &2, Nat, _ => Nat>

type Re is Data:
  RLit{c: Char}
  RAny{}
  RSet{neg: Bool, rs: Regex.Ranges()}
  RCat{a: Re, b: Re}
  RAlt{a: Re, b: Re}
  RRep{r: Re, lo: Nat, hi: Maybe<&2, Nat>, greedy: Bool}
  RCap{n: Nat, r: Re}
  RBol{}
  REol{}
  RWordB{neg: Bool}
  REmpty{}

type Inst is Data:
  IChr{c: U32}
  IAny{}
  ISet{neg: Bool, rs: Regex.Ranges()}
  ISplit{x: Nat, y: Nat}
  IJmp{x: Nat}
  ISave{slot: Nat}
  IBol{multi: Bool}
  IEol{multi: Bool}
  IWordB{neg: Bool}
  IMatch{}

type Regex is Data:
  Regex{prog: List<&2, Inst>, ngroups: Nat}

type Match is Data:
  Match{start: Nat, end: Nat, groups: List<&2, Maybe<&2, Regex.Span()>>}

type Regex.Error is Data:
  RErr{pos: Nat, msg: String}

# Parser scratch: one frame per open group (cap = its number), the finished
# alternatives and the current one both newest-first, q = what the last
# item was (0 nothing to repeat, 1 an atom, 2 a fresh greedy repeat)
type Regex.Frame is Data:
  Frame{cap: Maybe<&2, Nat>, alts: List<&2, Re>, cat: List<&2, Re>, q: Nat}

# Class scan: the unread tail, the last single char (a pending range low),
# whether a `-` follows it, the ranges so far; CDone once `]` is read
type Regex.Cls is Data:
  CScan{s: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges()}
  CDone{rs: Regex.Ranges(), rest: String}

# Effects
# -------

# A handle is opaque, a law with no constructor and no field, so no
# program forges one, and linear (Type), so none copies or reuses one.
law File:
  Type

law Socket:
  Type

law Listener:
  Type

law Window:
  Type

law Audio:
  Type

# A channel is Data: the computations it links each hold a copy.
law Chan:
  for -A: Type
  Data

type IO.OP<-R: Type> is Type:
  Emit{value: R}
  Halt{code: U32, message: String}

law Pair:
  for -A: Type
  for -B: Type
  Type

law IO:
  for -A: Type
  Type

type App<-S: Type> is Type:
  App{view: S -> Pair(S, Image), tick: List<Event> -> S -> IO(Maybe<S>)}

# Defs
# ====

# Word
# ----

def Word(n):
  match n:
    case 0n:
      Word.Nil
    case 1n+p:
      Word.Con<p>

# Pair
# ----

def Pair(A, B):
  Sigma<&1, &1, A, _ => B>

def Exists(-A: Type, -B: @-x: A -> Type) -> Type:
  Sigma<&1, &1, A, B>

def Or(-A: Type, -B: Type) -> Type:
  Either<&1, &1, A, B>

def Pair.fst(-A: Type, -B: Type, p: A & B) -> A:
  (a, b) = p
  a

def Pair.snd(-A: Type, -B: Type, p: A & B) -> B:
  (a, b) = p
  b

# Effects
# -------

def IO(A):
  @-R: Type ->
  @k: (A -> IO.OP<R>) ->
  IO.OP<R>

def IO.pure(-A: Type, x: A) -> IO(A):
  R => k => k(x)

def IO.bind(-A: Type, -B: Type, m: IO(A), f: A -> IO(B)) -> IO(B):
  R => k => m(R, x => f(x, R, k))

def IO.print(text: String) -> IO(Unit):
  import "./effs/print.c"
  import "./effs/print.js"

def IO.write(text: String) -> IO(Unit):
  import "./effs/write.c"
  import "./effs/write.js"

def IO.print_err(text: String) -> IO(Unit):
  import "./effs/print_err.c"
  import "./effs/print_err.js"

def IO.get_env(name: String) -> IO(Result<&1, &1, U32 & String, String>):
  import "./effs/get_env.c"
  import "./effs/get_env.js"

def IO.args() -> IO(List<String>):
  import "./effs/args.c"
  import "./effs/args.js"

def IO.die(-A: Type, code: U32, msg: String) -> IO(A):
  R => k => Halt{code, msg}

def IO.pass(-A: Type, r: Result<&1, &1, U32 & String, A>) -> IO(A):
  match r:
    case Done{value}:
      IO.pure(A, value)
    case Fail{(code, message)}:
      IO.die(A, code, message)

def IO.try(-A: Type, act: IO(Result<&1, &1, U32 & String, A>)) -> IO(A):
  IO.bind(Result<&1, &1, U32 & String, A>, A, act, IO.pass(A))

def IO.random_u32() -> IO(Result<&1, &1, U32 & String, U32>):
  import "./effs/random_u32.c"
  import "./effs/random_u32.js"

def IO.spawn(-A: Type, act: IO(A)) -> IO(Unit):
  import "./effs/spawn.c"
  import "./effs/spawn.js"

def IO.sleep(ms: U32) -> IO(Unit):
  import "./effs/sleep.c"
  import "./effs/sleep.js"

def IO.within.go(-A: Type, ms: U32, op: IO.OP<A>) -> IO(Maybe<&1, A>):
  import "./effs/within.c"
  import "./effs/within.js"

def IO.within.at(-A: Type, ms: U32, op: IO.OP<A>) -> IO(Maybe<&1, A>):
  match op:
    case Emit{v}:
      IO.pure(Maybe<&1, A>, Some{v})
    case o:
      IO.within.go(A, ms, o)

# act's steps up to its first effect, at once; then, if it has not
# answered, the rest as a computation of its own: Some{its answer}, or
# None{} when ms after that effect it still waits (a later answer is
# dropped). Nothing preempts act: it yields only at an effect.
def IO.within(-A: Type, ms: U32, act: IO(A)) -> IO(Maybe<&1, A>):
  IO.within.at(A, ms, act(A, x => Emit{x}))

# ms on a monotonic clock (since boot): for durations and deadlines
def IO.now() -> IO(Nat):
  import "./effs/now.c"
  import "./effs/now.js"

# ms since the Unix epoch, the wall clock (it can step): for dates, which
# bend2/time.bend prints (Time.iso, Time.http)
def IO.wall() -> IO(Nat):
  import "./effs/wall.c"
  import "./effs/wall.js"

def IO.signal_pending(sig: U32) -> IO(Bool):
  import "./effs/signal_pending.c"
  import "./effs/signal_pending.js"

# whether the signal has arrived at all since it was first caught (by
# either ask): nothing clears it, so every computation hears it
def IO.signal_seen(sig: U32) -> IO(Bool):
  import "./effs/signal_pending.c"
  import "./effs/signal_pending.js"

def Chan.new(-A: Type, room: U32) -> IO(Chan(A)):
  import "./effs/chan.c"
  import "./effs/chan.js"

def Chan.send(-A: Type, chan: Chan(A), value: A) -> IO(Bool):
  import "./effs/chan.c"
  import "./effs/chan.js"

# Chan.send that never waits: False, and the value dropped, when the
# channel is full or closed
def Chan.offer(-A: Type, chan: Chan(A), value: A) -> IO(Bool):
  import "./effs/chan.c"
  import "./effs/chan.js"

def Chan.recv(-A: Type, chan: Chan(A)) -> IO(Maybe<&1, A>):
  import "./effs/chan.c"
  import "./effs/chan.js"

# Chan.recv that never waits: None{} when nothing is there now
def Chan.take(-A: Type, chan: Chan(A)) -> IO(Maybe<&1, A>):
  import "./effs/chan.c"
  import "./effs/chan.js"

def Chan.close(-A: Type, chan: Chan(A)) -> IO(Unit):
  import "./effs/chan.c"
  import "./effs/chan.js"

def IO.fork.go(-A: Type, act: IO(A), chan: Chan(A)) -> IO(Chan(A)):
  +c = chan
  sent = IO.bind(A, Bool, act, x => Chan.send(A, c, x))
  IO.bind(Unit, Chan(A), IO.spawn(Bool, sent), u => IO.pure(Chan(A), c))

def IO.fork(-A: Type, act: IO(A)) -> IO(Chan(A)):
  IO.bind(Chan(A), Chan(A), Chan.new(A, 1), IO.fork.go(A, act))

def IO.join.go(-A: Type, chan: Chan(A), got: Maybe<&1, A>) -> IO(A):
  match got:
    case None{}:
      IO.die(A, 1, "IO.join: the channel was closed")
    case Some{value}:
      IO.bind(Unit, A, Chan.close(A, chan), u => IO.pure(A, value))

def IO.join(-A: Type, +chan: Chan(A)) -> IO(A):
  IO.bind(Maybe<&1, A>, A, Chan.recv(A, chan), IO.join.go(A, chan))

def File.open(path: String, mode: String) ->
  IO(Result<&1, &1, U32 & String, File>):
  import "./effs/file_open.c"
  import "./effs/file_open.js"

# path read-only below the directory root, never through a symbolic link
# (one openat2 on Linux 5.6+, else an openat per component; the root's
# descriptor is kept up to a second). Contract (wire/world.bend,
# fopen.ok): Done only for a regular file; a missing name, a symbolic
# link, a directory and a path that climbs out of root all fail.
def File.open_under(root: String, path: String) ->
  IO(Result<&1, &1, U32 & String, File>):
  import "./effs/file_open_under.c"
  import "./effs/file_open_under.js"

# File.open_under with its fstat, and a file under small bytes read and
# closed: (size, (mtime in s, (None, its bytes))), or a larger one left
# open at 0 (Some, ""). Contract (wire/world.bend, fget.ok): opens what
# open_under opens; a body is the whole file, under small. On C a small
# file's answer is kept a second per thread and answered again from
# memory (the memo: memo_bounded, memo_answers); the JS twin always asks.
def File.get_under(root: String, path: String, small: U32) ->
  IO(Result<&1, &1, U32 & String, U32 & U32 & Maybe<&1, File> & Bytes()>):
  import "./effs/file_open_under.c"
  import "./effs/file_open_under.js"

def File.read(file: File, max: U32) ->
  IO(File & Result<&1, &1, U32 & String, String>):
  import "./effs/file_read.c"
  import "./effs/file_read.js"

def File.read_bytes(file: File, max: U32) ->
  IO(File & Result<&1, &1, U32 & String, List<&2, U32>>):
  import "./effs/file_read.c"
  import "./effs/file_read.js"

def File.read_at(file: File, offset: U32, max: U32) ->
  IO(File & Result<&1, &1, U32 & String, List<&2, U32>>):
  import "./effs/file_read.c"
  import "./effs/file_read.js"

def File.size(file: File) ->
  IO(File & Result<&1, &1, U32 & String, U32>):
  import "./effs/file_size.c"
  import "./effs/file_size.js"

# A UTF-8 decoder's state between chunks: the bytes of a scalar cut by
# the end of a chunk (pend, low byte first; need of them, 3 at most).
# File.read_text is File.read through it, so the chunks of a file append
# to its whole decode; need answers 4 at the end of the file.
type Utf8.Dec is Data:
  Dec{pend: U32, need: U32}

def Utf8.Dec.new() -> Utf8.Dec:
  Dec{0, 0}

def Utf8.Dec.Raw(H: Type) -> Type:
  H & Result<&1, &1, U32 & String, U32 & U32 & String>

def Utf8.Dec.Out(H: Type) -> Type:
  H & Result<&1, &1, U32 & String, Utf8.Dec & String>

def File.read_text.go(file: File, pend: U32, need: U32, max: U32) ->
  IO(Utf8.Dec.Raw(File)):
  import "./effs/file_read_text_go.c"
  import "./effs/file_read_text_go.js"

def File.read_text.pack(-H: Type, got: Utf8.Dec.Raw(H)) -> IO(Utf8.Dec.Out(H)):
  (file, r) = got
  match r:
    case Fail{e}:
      IO.pure(Utf8.Dec.Out(H), (file, Fail{e}))
    case Done{(pend, (need, text))}:
      IO.pure(Utf8.Dec.Out(H), (file, Done{(Dec{pend, need}, text)}))

def File.read_text(file: File, dec: Utf8.Dec, max: U32) ->
  IO(File & Result<&1, &1, U32 & String, Utf8.Dec & String>):
  Dec{pend, need} = dec
  IO.bind(Utf8.Dec.Raw(File), Utf8.Dec.Out(File),
    File.read_text.go(file, pend, need, max), File.read_text.pack(File))

def File.write(file: File, data: String) ->
  IO(File & Result<&1, &1, U32 & String, Unit>):
  import "./effs/file_write.c"
  import "./effs/file_write.js"

def File.write_bytes(file: File, data: List<&2, U32>) ->
  IO(File & Result<&1, &1, U32 & String, Unit>):
  import "./effs/file_write.c"
  import "./effs/file_write.js"

# File.write_bytes from a Bytes: the block as it lies, no byte a cell,
# every byte written (a Char past 255 is not a byte: EINVAL, nothing
# written)
def File.write_buf(file: File, data: Bytes()) ->
  IO(File & Result<&1, &1, U32 & String, Unit>):
  import "./effs/file_write.c"
  import "./effs/file_write.js"

def File.close(file: File) -> IO(Unit):
  import "./effs/file_close.c"
  import "./effs/file_close.js"

def TCP.listen(port: U32) -> IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tcp_listen.c"
  import "./effs/tcp_listen.js"

def TCP.listen_shared(port: U32) -> IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tcp_listen_shared.c"
  import "./effs/tcp_listen_shared.js"

def TLS.listen(port: U32, cert: String, key: String) ->
  IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tls_listen.c"
  import "./effs/tls_listen.js"

# TLS.listen agreeing by ALPN to the names given, comma-separated, in
# the server's order of preference ("h2", "h2,http/1.1"); TLS.listen
# agrees to http/1.1
def TLS.listen_alpn(port: U32, cert: String, key: String, names: String) ->
  IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tls_listen.c"
  import "./effs/tls_listen_alpn.js"

# TCP.listen, TCP.listen_shared and TLS.listen bound to one address, a
# dotted IPv4 one ("127.0.0.1": this machine alone) or an IPv6 one ("::1";
# "::" every interface, IPv4's too where the system allows); they bind
# 0.0.0.0
def TCP.listen_on(host: String, port: U32) -> IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tcp_listen.c"
  import "./effs/tcp_listen.js"

def TCP.listen_shared_on(host: String, port: U32) -> IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tcp_listen_shared.c"
  import "./effs/tcp_listen_shared.js"

def TLS.listen_on(host: String, port: U32, cert: String, key: String) ->
  IO(Result<&1, &1, U32 & String, Listener>):
  import "./effs/tls_listen.c"
  import "./effs/tls_listen.js"

def TCP.accept(listener: Listener) ->
  IO(Listener & Result<&1, &1, U32 & String, Socket>):
  import "./effs/tcp_accept.c"
  import "./effs/tcp_accept.js"

# Contract (demos/io_http_engine/world.bend, lsn.ok): Some{socket} for a
# connection; None{} once ms have passed with none, or after shedding an
# arrival it had no descriptor for; Fail only for a listener that is no
# longer one, never with a code of the passing set (EMFILE, ENFILE,
# ECONNABORTED, ENOBUFS, ENOMEM, ...), which it answers itself.
def TCP.accept_poll(listener: Listener, ms: U32) ->
  IO(Listener & Result<&1, &1, U32 & String, Maybe<&1, Socket>>):
  import "./effs/tcp_accept_poll.c"
  import "./effs/tcp_accept_poll.js"

def TCP.connect(host: String, port: U32) ->
  IO(Result<&1, &1, U32 & String, Socket>):
  import "./effs/tcp_connect.c"
  import "./effs/tcp_connect.js"

# TCP.connect with a deadline: host a dotted IPv4 address or an IPv6 one
# (DNS.resolve and DNS.resolve_all turn a name into them). Contract (wire/world.bend, cx.ok): to a peer
# that accepts, Done within ms; to a port nothing listens on, Fail with
# ECONNREFUSED; to one that never answers, Fail with ETIMEDOUT (110 on
# Linux, 60 on macOS) after at least ms, and never later than the loop's
# next turn after it; a malformed address is EINVAL at once.
def TCP.connect_poll(host: String, port: U32, ms: U32) ->
  IO(Result<&1, &1, U32 & String, Socket>):
  import "./effs/tcp_connect_poll.c"
  import "./effs/tcp_connect_poll.js"

# Whether nothing waits to be read on a socket and its peer has not
# closed it: one poll with no wait, readable (a byte, the FIN, an error)
# being not idle. Contract (wire/world.bend, idle.ok): False whenever a
# byte or the FIN has arrived and not been read; True for a connection
# with nothing pending. A pool asks it before handing a kept connection
# out (wire/pool.bend).
def TCP.idle(sock: Socket) -> IO(Socket & Bool):
  import "./effs/tcp_idle.c"
  import "./effs/tcp_idle.js"

# The first IPv4 address the system's resolver has for a name, dotted
# (/etc/hosts, then DNS); a dotted address is its own answer, and a name
# with none is ENOENT. The resolver blocks, so it runs off the loop.
def DNS.resolve(name: String) -> IO(Result<&1, &1, U32 & String, String>):
  import "./effs/dns_resolve.c"
  import "./effs/dns_resolve.js"

# Every address the system's resolver has for a name, IPv6 and IPv4, in
# the order it answers them (RFC 6724's, the order to try them in), each
# once, at most 16, joined by commas ("::1,127.0.0.1"); IPv6 ones as RFC
# 5952 writes them; a literal is its own answer; none is ENOENT.
def DNS.resolve_all(name: String) -> IO(Result<&1, &1, U32 & String, String>):
  import "./effs/dns_resolve_all.c"
  import "./effs/dns_resolve_all.js"

# TLS.connect(addr, port, name, alpn, ca, ms): TCP.connect_poll and a
# TLS client handshake under the one deadline; the socket then carries
# the session through the byte effects, as TLS.listen's do. The peer's
# chain is verified against the system's store (ca "") or the file ca
# names (a pin), and its certificate against name (a host name, sent as
# SNI too, or an address, IPv4 or IPv6 without brackets, checked against
# the certificate's IP addresses and never sent as SNI); alpn is offered comma-separated ("" for
# none), and the answer carries the protocol chosen ("" for none). A
# close without close_notify reads as a failure, never as a FIN.
# Contract (wire/world.bend, tx.ok's sibling cx.ok, and tls.ok): Done
# only when the handshake completed and the certificate verified; a
# certificate refused (untrusted, another name) is EACCES; any other
# handshake failure is EPROTO or ECONNRESET; the deadline is ETIMEDOUT.
# The JS lane has no TLS and answers ENOSYS.
def TLS.connect(addr: String, port: U32, name: String, alpn: String, ca: String, ms: U32) ->
  IO(Result<&1, &1, U32 & String, Socket & String>):
  import "./effs/tls_listen.c"
  import "./effs/tls_connect.js"

# The byte pair. A String is decoded as UTF-8, so TCP.recv and TCP.send
# carry text and only text: a byte that is not valid UTF-8 arrives as
# U+FFFD, three bytes for one. A protocol that carries bytes -- any
# body that is not text -- needs these, which are to sockets what
# File.read_bytes and File.write_bytes already are to files.
def TCP.recv_bytes(sock: Socket, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, List<&2, U32>>):
  import "./effs/tcp_recv_bytes.c"
  import "./effs/tcp_recv_bytes.js"

def TCP.send_bytes(sock: Socket, data: List<&2, U32>) ->
  IO(Socket & Result<&1, &1, U32 & String, Unit>):
  import "./effs/tcp_send_bytes.c"
  import "./effs/tcp_send_bytes.js"

def TCP.send(sock: Socket, data: String) ->
  IO(Socket & Result<&1, &1, U32 & String, Unit>):
  import "./effs/tcp_send.c"
  import "./effs/tcp_send.js"

def TCP.recv(sock: Socket, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, String>):
  import "./effs/tcp_recv.c"
  import "./effs/tcp_recv.js"

def TCP.poll(sock: Socket, max: U32, ms: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Maybe<&1, String>>):
  import "./effs/tcp_poll.c"
  import "./effs/tcp_poll.js"

# TCP.poll_buf that also ends when chan has something for a receiver (a
# value, a sender waiting, its close): then True and Done{None{}}, and
# chan keeps it. Bytes win when both are there. One wait at a time may
# watch a channel.
def TCP.poll_buf_or(-A: Type, sock: Socket, max: U32, ms: U32, chan: Chan(A)) ->
  IO(Socket & Bool & Result<&1, &1, U32 & String, Maybe<&1, Bytes()>>):
  import "./effs/chan.c"
  import "./effs/chan.js"

def TCP.poll_bytes(sock: Socket, max: U32, ms: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Maybe<&1, List<&2, U32>>>):
  import "./effs/tcp_poll_bytes.c"
  import "./effs/tcp_poll_bytes.js"

# The buffer quartet: the byte pair and File.read_bytes with a Bytes in
# place of the list, so no byte is ever a cell (see Bytes below).
def TCP.recv_buf(sock: Socket, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Bytes()>):
  import "./effs/tcp_recv_buf.c"
  import "./effs/tcp_recv_buf.js"

def TCP.send_buf(sock: Socket, data: Bytes()) ->
  IO(Socket & Result<&1, &1, U32 & String, Unit>):
  import "./effs/tcp_send_buf.c"
  import "./effs/tcp_send_buf.js"

# TCP.send_buf with a deadline: a send that makes no progress for ms
# fails with ETIMEDOUT, so a peer that stops reading cannot hold it.
# Contract (demos/io_http_engine/world.bend, tx.ok): to a peer that
# reads, each piece within ms of the last, Done once every byte is out
# however long the whole takes; to one that stops reading, Fail with
# ETIMEDOUT after at least ms; to one that resets, Fail. An empty send
# is Done at once.
def TCP.send_buf_poll(sock: Socket, data: Bytes(), ms: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Unit>):
  import "./effs/tcp_send_buf_poll.c"
  import "./effs/tcp_send_buf_poll.js"

# Contract (demos/io_http_engine/world.bend, rx.ok): None{} only once ms
# have passed with nothing to read; Some{bytes} within ms, at most max
# of them, the empty Bytes being the peer's FIN; Fail at any time.
def TCP.poll_buf(sock: Socket, max: U32, ms: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Maybe<&1, Bytes()>>):
  import "./effs/tcp_poll_buf.c"
  import "./effs/tcp_poll_buf.js"

# Contract (demos/io_http_engine/world.bend, fread.ok): at most max bytes
# and at most what is left, none only at the end of the file.
def File.read_buf(file: File, max: U32) ->
  IO(File & Result<&1, &1, U32 & String, Bytes()>):
  import "./effs/file_read_buf.c"
  import "./effs/file_read_buf.js"

# The file's bytes from off, len of them, onto the socket, with
# TCP.send_buf_poll's deadline: the kernel's sendfile on a plain socket,
# a block at a time through the session under TLS. The file's position
# does not move.
# Contract (wire/world.bend, fsend.ok): to a peer that reads, Done once
# the len bytes of the file from off are out, exactly those; when the
# file has fewer, the peer has what it had and the send fails with
# ENODATA; to one that stops reading, Fail with ETIMEDOUT after at least
# ms; to one that resets, Fail.
def File.sendfile(sock: Socket, file: File, off: U32, len: U32, ms: U32) ->
  IO(Socket & File & Result<&1, &1, U32 & String, Unit>):
  import "./effs/file_sendfile.c"
  import "./effs/file_sendfile.js"

# TCP.recv through a Utf8.Dec, as File.read_text: a scalar cut by the end
# of a recv is carried; need answers 4 when the peer has closed.
def TCP.recv_text.go(sock: Socket, pend: U32, need: U32, max: U32) ->
  IO(Utf8.Dec.Raw(Socket)):
  import "./effs/tcp_recv_text_go.c"
  import "./effs/tcp_recv_text_go.js"

def TCP.recv_text(sock: Socket, dec: Utf8.Dec, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Utf8.Dec & String>):
  Dec{pend, need} = dec
  IO.bind(Utf8.Dec.Raw(Socket), Utf8.Dec.Out(Socket),
    TCP.recv_text.go(sock, pend, need, max), File.read_text.pack(Socket))

def UDP.bind(port: U32) -> IO(Result<&1, &1, U32 & String, Socket>):
  import "./effs/udp_bind.c"
  import "./effs/udp_bind.js"

def UDP.send_to(sock: Socket, host: String, port: U32, data: String) ->
  IO(Socket & Result<&1, &1, U32 & String, Unit>):
  import "./effs/udp_send_to.c"
  import "./effs/udp_send_to.js"

def UDP.recv_from(sock: Socket, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, String & U32 & String>):
  import "./effs/udp_recv_from.c"
  import "./effs/udp_recv_from.js"

def UDP.poll(sock: Socket, max: U32) ->
  IO(Socket & Result<&1, &1, U32 & String, Maybe<&1, String & U32 & String>>):
  import "./effs/udp_poll.c"
  import "./effs/udp_poll.js"

def Socket.close(socket: Socket) -> IO(Unit):
  import "./effs/socket_close.c"
  import "./effs/socket_close.js"

def Listener.close(listener: Listener) -> IO(Unit):
  import "./effs/listener_close.c"
  import "./effs/listener_close.js"

def Window.open(title: String, width: U32, height: U32) ->
  IO(Result<&1, &1, U32 & String, Window>):
  import "./effs/window_open.c"
  import "./effs/window_open.js"

def Window.frame(window: Window, image: Image) ->
  IO(Window & Image & List<Event>):
  import "./effs/window_frame.c"
  import "./effs/window_frame.js"

def Window.set_title(window: Window, title: String) -> IO(Window):
  import "./effs/window_set_title.c"
  import "./effs/window_set_title.js"

def Window.close(window: Window) -> IO(Unit):
  import "./effs/window_close.c"
  import "./effs/window_close.js"

def Audio.open(rate: U32) -> IO(Result<&1, &1, U32 & String, Audio>):
  import "./effs/audio.c"
  import "./effs/audio_open.js"

def Audio.write(audio: Audio, samples: List<&2, F32>) -> IO(Audio & U32):
  import "./effs/audio.c"
  import "./effs/audio_write.js"

def Audio.close(audio: Audio) -> IO(Unit):
  import "./effs/audio.c"
  import "./effs/audio_close.js"

# Equal
# -----

law Equal.cong:
  for -A: Type
  for -B: Type
  for -f: A -> B
  for -a: A
  for -b: A
  for  e: {a == b : A}
  {f(a) == f(b) : B}

def Equal.cong(A, B, f, a, b, e):
  %e : {f(a) == f(_) : B}
  {==}

law Equal.sym:
  for -A: Type
  for -a: A
  for -b: A
  for  e: {a == b : A}
  {b == a : A}

def Equal.sym(A, a, b, e):
  %e : {_ == a : A}
  {==}

law Equal.trans:
  for -A : Type
  for -a : A
  for -b : A
  for -c : A
  for ab : {a == b : A}
  for bc : {b == c : A}
  {a == c : A}

def Equal.trans(A, a, b, c, ab, bc):
  %bc : {a == _ : A}
  ab

# Empty
# -----

def Empty.absurd(-A: Type, e: Empty) -> A:
  match e:

# Bool
# ----

def Bool.not(b: Bool) -> Bool:
  match b:
    case False{}:
      True{}
    case True{}:
      False{}

def Bool.and(a: Bool, b: Bool) -> Bool:
  match a:
    case False{}:
      False{}
    case True{}:
      b

def Bool.or(a: Bool, b: Bool) -> Bool:
  match a:
    case False{}:
      b
    case True{}:
      True{}

def Bool.xor(a: Bool, b: Bool) -> Bool:
  match a:
    case False{}:
      b
    case True{}:
      Bool.not(b)

def Bool.cmp(a: Bool, b: Bool) -> Cmp:
  match a b:
    case False{} False{}:
      EQ{}
    case False{} True{}:
      LT{}
    case True{} False{}:
      GT{}
    case True{} True{}:
      EQ{}

def Bool.full_add(a: Bool, b: Bool, c: Bool) -> Bool & Bool:
  match a b c:
    case False{} False{} False{}:
      (False{}, False{})
    case False{} False{} True{}:
      (True{}, False{})
    case False{} True{} False{}:
      (True{}, False{})
    case False{} True{} True{}:
      (False{}, True{})
    case True{} False{} False{}:
      (True{}, False{})
    case True{} False{} True{}:
      (False{}, True{})
    case True{} True{} False{}:
      (False{}, True{})
    case True{} True{} True{}:
      (True{}, True{})

def Bool.pick(-A: Type, c: Bool, a: A, b: A) -> A:
  match c:
    case False{}:
      b
    case True{}:
      a

def Bool.to_u32(b: Bool) -> U32:
  match b:
    case False{}:
      0
    case True{}:
      1

# Cmp
# ---

def Cmp.is_lt(c: Cmp) -> Bool:
  match c:
    case LT{}:
      True{}
    case EQ{}:
      False{}
    case GT{}:
      False{}

def Cmp.is_eq(c: Cmp) -> Bool:
  match c:
    case LT{}:
      False{}
    case EQ{}:
      True{}
    case GT{}:
      False{}

def Cmp.is_gt(c: Cmp) -> Bool:
  match c:
    case LT{}:
      False{}
    case EQ{}:
      False{}
    case GT{}:
      True{}

def Cmp.is_le(c: Cmp) -> Bool:
  match c:
    case LT{}:
      True{}
    case EQ{}:
      True{}
    case GT{}:
      False{}

def Cmp.is_ge(c: Cmp) -> Bool:
  match c:
    case LT{}:
      False{}
    case EQ{}:
      True{}
    case GT{}:
      True{}

# Nat
# ---

def Nat.double(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      2n+Nat.double(p)

def Nat.add(a: Nat, b: Nat) -> Nat:
  match a:
    case 0n:
      b
    case 1n+p:
      1n+Nat.add(p, b)

def Nat.sub(a: Nat, b: Nat) -> Nat:
  match a b:
    case 0n 0n:
      0n
    case 0n 1n+bp:
      0n
    case 1n+ap 0n:
      1n+ap
    case 1n+ap 1n+bp:
      Nat.sub(ap, bp)

def Nat.mul(a: Nat, +b: Nat) -> Nat:
  match a:
    case 0n:
      0n
    case 1n+p:
      Nat.add(b, Nat.mul(p, b))

def Nat.divmod.go(n: Nat, m: Nat, d: Nat, r: Nat) -> Nat & Nat:
  match n:
    case 0n:
      (d, r)
    case 1n+np:
      match m:
        case 0n:
          Nat.divmod.go(np, r, 1n+d, 0n)
        case 1n+mp:
          Nat.divmod.go(np, mp, d, 1n+r)

def Nat.divmod(a: Nat, b: Nat) -> Nat & Nat:
  match b:
    case 0n:
      (0n, a)
    case 1n+bp:
      Nat.divmod.go(a, bp, 0n, 0n)

def Nat.cmp(a: Nat, b: Nat) -> Cmp:
  match a b:
    case 0n 0n:
      EQ{}
    case 0n 1n+bp:
      LT{}
    case 1n+ap 0n:
      GT{}
    case 1n+ap 1n+bp:
      Nat.cmp(ap, bp)

def Nat.is_lt(a: Nat, b: Nat) -> Bool:
  Cmp.is_lt(Nat.cmp(a, b))

def Nat.is_eq(a: Nat, b: Nat) -> Bool:
  Cmp.is_eq(Nat.cmp(a, b))

def Nat.is_ne(a: Nat, b: Nat) -> Bool:
  Bool.not(Cmp.is_eq(Nat.cmp(a, b)))

def Nat.is_le(a: Nat, b: Nat) -> Bool:
  Cmp.is_le(Nat.cmp(a, b))

def Nat.is_gt(a: Nat, b: Nat) -> Bool:
  Cmp.is_gt(Nat.cmp(a, b))

def Nat.is_ge(a: Nat, b: Nat) -> Bool:
  Cmp.is_ge(Nat.cmp(a, b))

def Nat.min(+a: Nat, +b: Nat) -> Nat:
  match a b:
    case 0n _:
      0n
    case 1n+ap 0n:
      0n
    case 1n+ap 1n+bp:
      1n+Nat.min(ap, bp)

def Nat.max(+a: Nat, +b: Nat) -> Nat:
  match a b:
    case 0n _:
      b
    case 1n+ap 0n:
      1n+ap
    case 1n+ap 1n+bp:
      1n+Nat.max(ap, bp)

law Nat.ge_refl:
  for a: Nat
  {Nat.is_ge(a, a) == True{} : Bool}

def Nat.ge_refl(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      Nat.ge_refl(p)

law Nat.max_ge_l:
  for +a: Nat
  for +b: Nat
  {Nat.is_ge(Nat.max(a, b), a) == True{} : Bool}

def Nat.max_ge_l(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      Nat.ge_refl(a)
    case 1n+ap 1n+bp:
      Nat.max_ge_l(ap, bp)

law Nat.max_ge_r:
  for +a: Nat
  for +b: Nat
  {Nat.is_ge(Nat.max(a, b), b) == True{} : Bool}

def Nat.max_ge_r(a, b):
  match a b:
    case 0n _:
      Nat.ge_refl(b)
    case 1n+ap 0n:
      {==}
    case 1n+ap 1n+bp:
      Nat.max_ge_r(ap, bp)

def Nat.div.fin(qr: Nat & Nat) -> Nat:
  (q, r) = qr
  q

def Nat.div(a: Nat, b: Nat) -> Nat:
  Nat.div.fin(Nat.divmod(a, b))

def Nat.mod.fin(qr: Nat & Nat) -> Nat:
  (q, r) = qr
  r

def Nat.mod(a: Nat, b: Nat) -> Nat:
  Nat.mod.fin(Nat.divmod(a, b))

def Nat.pow(+a: Nat, b: Nat) -> Nat:
  match b:
    case 0n:
      1n
    case 1n+p:
      Nat.mul(a, Nat.pow(a, p))

# Maybe
# -----

def Maybe.pure(a, -A: Kind(a), x: A) -> Maybe<a, A>:
  Some{x}

def Maybe.bind(
  a, -A: Kind(a), -B: Kind(a), m: Maybe<a, A>, f: A -> Maybe<a, B>
) -> Maybe<a, B>:
  match m:
    case None{}:
      None{}
    case Some{x}:
      f(x)

def Maybe.default(a, -A: Kind(a), m: Maybe<a, A>, d: A) -> A:
  match m:
    case None{}:
      d
    case Some{x}:
      x

def Maybe.is_some(a, -A: Kind(a), m: Maybe<a, A>) -> Bool:
  match m:
    case None{}:
      False{}
    case Some{x}:
      True{}

def Maybe.is_none(a, -A: Kind(a), m: Maybe<a, A>) -> Bool:
  Bool.not(Maybe.is_some(a, A, m))

def Maybe.map(
  a, -A: Kind(a), -B: Kind(a), f: A -> B, m: Maybe<a, A>
) -> Maybe<a, B>:
  match m:
    case None{}:
      None{}
    case Some{x}:
      Some{f(x)}

def Maybe.or(a, -A: Kind(a), m: Maybe<a, A>, n: Maybe<a, A>) -> Maybe<a, A>:
  match m:
    case None{}:
      n
    case Some{x}:
      Some{x}

# Result
# ------

def Result.pure(a, b, -E: Kind(a), -A: Kind(b), x: A) -> Result<a, b, E, A>:
  Done{x}

def Result.bind(
  a, b, -E: Kind(a), -A: Kind(b), -B: Kind(b), r: Result<a, b, E, A>,
  f: A -> Result<a, b, E, B>
) -> Result<a, b, E, B>:
  match r:
    case Fail{e}:
      Fail{e}
    case Done{x}:
      f(x)

def Result.default(
  a, b, -E: Kind(a), -A: Kind(b), r: Result<a, b, E, A>, d: A
) -> A:
  match r:
    case Fail{e}:
      d
    case Done{x}:
      x

def Result.is_done(
  a, b, -E: Kind(a), -A: Kind(b), r: Result<a, b, E, A>
) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{x}:
      True{}

def Result.is_fail(
  a, b, -E: Kind(a), -A: Kind(b), r: Result<a, b, E, A>
) -> Bool:
  Bool.not(Result.is_done(a, b, E, A, r))

def Result.map(
  a, b, -E: Kind(a), -A: Kind(b), -B: Kind(b), f: A -> B,
  r: Result<a, b, E, A>
) -> Result<a, b, E, B>:
  match r:
    case Fail{e}:
      Fail{e}
    case Done{x}:
      Done{f(x)}

# List
# ----

def List.map(~A: Type, ~B: Type, ~f: A -> B, xs: List<A>) -> List<B>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      f(h) <> List.map(~A, ~B, ~f, t)

def List.length(a, -A: Kind(a), xs: List<a, A>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case h <> t:
      1n+List.length(a, A, t)

def List.append(a, -A: Kind(a), xs: List<a, A>, ys: List<a, A>) -> List<a, A>:
  match xs:
    case Nil{}:
      ys
    case h <> t:
      h <> List.append(a, A, t, ys)

def List.concat(a, -A: Kind(a), xss: List<a, List<a, A>>) -> List<a, A>:
  match xss:
    case Nil{}:
      Nil{}
    case h <> t:
      List.append(a, A, h, List.concat(a, A, t))

def List.reverse.go(a, -A: Kind(a), xs: List<a, A>, acc: List<a, A>) ->
  List<a, A>:
  match xs:
    case Nil{}:
      acc
    case h <> t:
      List.reverse.go(a, A, t, h <> acc)

def List.reverse(a, -A: Kind(a), xs: List<a, A>) -> List<a, A>:
  List.reverse.go(a, A, xs, Nil{})

def List.is_empty(a, -A: Kind(a), xs: List<a, A>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case h <> t:
      False{}

def List.head(a, -A: Kind(a), xs: List<a, A>) -> Maybe<a, A>:
  match xs:
    case Nil{}:
      None{}
    case h <> t:
      Some{h}

def List.tail(a, -A: Kind(a), xs: List<a, A>) -> List<a, A>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      t

def List.last.go(a, -A: Kind(a), xs: List<a, A>, last: A) -> A:
  match xs:
    case Nil{}:
      last
    case h <> t:
      List.last.go(a, A, t, h)

def List.last(a, -A: Kind(a), xs: List<a, A>) -> Maybe<a, A>:
  match xs:
    case Nil{}:
      None{}
    case h <> t:
      Some{List.last.go(a, A, t, h)}

def List.get(a, -A: Kind(a), xs: List<a, A>, n: Nat) -> Maybe<a, A>:
  match xs n:
    case Nil{} _:
      None{}
    case h <> t 0n:
      Some{h}
    case h <> t 1n+p:
      List.get(a, A, t, p)

def List.set(a, -A: Kind(a), xs: List<a, A>, n: Nat, x: A) -> List<a, A>:
  match xs n:
    case Nil{} _:
      Nil{}
    case h <> t 0n:
      x <> t
    case h <> t 1n+p:
      h <> List.set(a, A, t, p, x)

def List.take(a, -A: Kind(a), xs: List<a, A>, n: Nat) -> List<a, A>:
  match xs n:
    case Nil{} _:
      Nil{}
    case h <> t 0n:
      Nil{}
    case h <> t 1n+p:
      h <> List.take(a, A, t, p)

def List.drop(a, -A: Kind(a), xs: List<a, A>, n: Nat) -> List<a, A>:
  match xs n:
    case Nil{} _:
      Nil{}
    case h <> t 0n:
      h <> t
    case h <> t 1n+p:
      List.drop(a, A, t, p)

def List.zip(
  a, -A: Kind(a), b, -B: Kind(b), xs: List<a, A>, ys: List<b, B>
) -> List<&1, A & B>:
  match xs ys:
    case Nil{} _:
      Nil{}
    case h <> t Nil{}:
      Nil{}
    case x <> xt y <> yt:
      (x, y) <> List.zip(a, A, b, B, xt, yt)

def List.range.go(+n: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match n:
    case 0n:
      acc
    case 1n+p:
      List.range.go(p, p <> acc)

def List.range(n: Nat) -> List<&2, Nat>:
  List.range.go(n, Nil{})

def List.replicate(-A: Data, n: Nat, +x: A) -> List<&2, A>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      x <> List.replicate(A, p, x)

def List.filter.put(-A: Data, h: A, r: List<&2, A>, keep: Bool) -> List<&2, A>:
  match keep:
    case False{}:
      r
    case True{}:
      h <> r

def List.filter(~A: Data, ~f: A -> Bool, xs: List<&2, A>) -> List<&2, A>:
  match xs:
    case Nil{}:
      Nil{}
    case +h <> t:
      List.filter.put(A, h, List.filter(~A, ~f, t), f(h))

def List.foldl(
  ~a: Quant, ~A: Kind(a), ~B: Type, ~f: B -> A -> B, xs: List<a, A>, acc: B
) -> B:
  match xs:
    case Nil{}:
      acc
    case h <> t:
      List.foldl(~a, ~A, ~B, ~f, t, f(acc, h))

def List.foldr(
  ~a: Quant, ~A: Kind(a), ~B: Type, ~f: A -> B -> B, xs: List<a, A>, z: B
) -> B:
  match xs:
    case Nil{}:
      z
    case h <> t:
      f(h, List.foldr(~a, ~A, ~B, ~f, t, z))

def List.any(~a: Quant, ~A: Kind(a), ~f: A -> Bool, xs: List<a, A>) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case h <> t:
      Bool.or(f(h), List.any(~a, ~A, ~f, t))

def List.all(~a: Quant, ~A: Kind(a), ~f: A -> Bool, xs: List<a, A>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case h <> t:
      Bool.and(f(h), List.all(~a, ~A, ~f, t))

def List.find.put(-A: Data, h: A, r: Maybe<&2, A>, hit: Bool) -> Maybe<&2, A>:
  match hit:
    case False{}:
      r
    case True{}:
      Some{h}

def List.find(~A: Data, ~f: A -> Bool, xs: List<&2, A>) -> Maybe<&2, A>:
  match xs:
    case Nil{}:
      None{}
    case +h <> t:
      List.find.put(A, h, List.find(~A, ~f, t), f(h))

def List.contains(
  ~A: Data, ~eq: A -> A -> Bool, xs: List<&2, A>, +x: A
) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case h <> t:
      Bool.or(eq(h, x), List.contains(~A, ~eq, t, x))

def List.merge.step(
  -A: Data, acc: List<&2, A>, x: A, xt: List<&2, A>, y: A, yt: List<&2, A>,
  le: Bool
) -> List<&2, A> & List<&2, A> & List<&2, A>:
  match le:
    case False{}:
      (y <> acc, x <> xt, yt)
    case True{}:
      (x <> acc, xt, y <> yt)

# The fuel bounds the steps: a sort passes its length, enough for any merge.
def List.merge.go(
  ~A: Data, ~le: A -> A -> Bool, fuel: Nat,
  st: List<&2, A> & List<&2, A> & List<&2, A>
) -> List<&2, A>:
  match fuel:
    case 0n:
      (acc, xs, ys) = st
      List.reverse.go(&2, A, acc, List.append(&2, A, xs, ys))
    case 1n+f:
      (acc, xs, ys) = st
      match xs ys:
        case Nil{} Nil{}:
          List.reverse.go(&2, A, acc, Nil{})
        case Nil{} y <> yt:
          List.reverse.go(&2, A, acc, y <> yt)
        case x <> xt Nil{}:
          List.reverse.go(&2, A, acc, x <> xt)
        case +x <> xt +y <> yt:
          List.merge.go(~A, ~le, f,
            List.merge.step(A, acc, x, xt, y, yt, le(x, y)))

def List.sort.runs(-A: Data, xs: List<&2, A>) -> List<&2, List<&2, A>>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      [h] <> List.sort.runs(A, t)

def List.sort.pass(
  ~A: Data, ~le: A -> A -> Bool, +n: Nat, runs: List<&2, List<&2, A>>
) -> List<&2, List<&2, A>>:
  match runs:
    case Nil{}:
      Nil{}
    case r <> Nil{}:
      [r]
    case r1 <> r2 <> rest:
      List.merge.go(~A, ~le, n, (Nil{}, r1, r2))
        <> List.sort.pass(~A, ~le, n, rest)

def List.sort.go(
  ~A: Data, ~le: A -> A -> Bool, fuel: Nat, +n: Nat, runs: List<&2, List<&2, A>>
) -> List<&2, A>:
  match fuel:
    case 0n:
      List.concat(&2, A, runs)
    case 1n+f:
      match runs:
        case Nil{}:
          Nil{}
        case r <> Nil{}:
          r
        case r1 <> r2 <> rest:
          List.sort.go(~A, ~le, f, n,
            List.sort.pass(~A, ~le, n, r1 <> r2 <> rest))

def List.sort(~A: Data, ~le: A -> A -> Bool, +xs: List<&2, A>) -> List<&2, A>:
  +n = List.length(&2, A, xs)
  List.sort.go(~A, ~le, n, n, List.sort.runs(A, xs))

def List.for_each(
  ~a: Quant, ~A: Kind(a), ~f: A -> IO(Unit), xs: List<a, A>
) -> IO(Unit):
  match xs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case h <> t:
      IO.bind(Unit, Unit, f(h), u => List.for_each(~a, ~A, ~f, t))

# Word
# ----

def Word.zero(n: Nat) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      WCon{False{}, Word.zero(p)}

def Word.not(n: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match w:
        case WCon{b, t}:
          WCon{Bool.not(b), Word.not(p, t)}

def Word.and(n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          WCon{Bool.and(ab, bb), Word.and(p, at, bt)}

def Word.or(n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          WCon{Bool.or(ab, bb), Word.or(p, at, bt)}

def Word.xor(n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          WCon{Bool.xor(ab, bb), Word.xor(p, at, bt)}

def Word.shl.put(n: Nat, c: Bool, w: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match w:
        case WCon{b, t}:
          WCon{c, Word.shl.put(p, b, t)}

def Word.shl(n: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match w:
        case WCon{b, t}:
          WCon{False{}, Word.shl.put(p, b, t)}

def Word.shl.out.con(-p: Nat, c: Bool, r: Bool & Word(p)) -> Bool & Word(1n+p):
  (hi, t2) = r
  (hi, WCon{c, t2})

def Word.shl.out(n: Nat, c: Bool, w: Word(n)) -> Bool & Word(n):
  match n:
    case 0n:
      (c, WNil{})
    case 1n+p:
      match w:
        case WCon{b, t}:
          Word.shl.out.con(p, c, Word.shl.out(p, b, t))

def Word.shr.pad(n: Nat, w: Word(n)) -> Word(1n+n):
  match n:
    case 0n:
      WCon{False{}, w}
    case 1n+p:
      match w:
        case WCon{b, t}:
          WCon{b, Word.shr.pad(p, t)}

def Word.shr(n: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match w:
        case WCon{b, t}:
          Word.shr.pad(p, t)

def Word.cmp.fin(ab: Bool, bb: Bool, t: Cmp) -> Cmp:
  match t:
    case LT{}:
      LT{}
    case EQ{}:
      Bool.cmp(ab, bb)
    case GT{}:
      GT{}

def Word.cmp(n: Nat, a: Word(n), b: Word(n)) -> Cmp:
  match n:
    case 0n:
      EQ{}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          Word.cmp.fin(ab, bb, Word.cmp(p, at, bt))

def Word.inc(n: Nat, w: Word(n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match w:
        case WCon{False{}, t}:
          WCon{True{}, t}
        case WCon{True{}, t}:
          WCon{False{}, Word.inc(p, t)}

law Word.adc:
  for n: Nat
  for a: Word(n)
  for b: Word(n)
  for f: Bool
  for c: Bool
  Word(n)

def Word.adc.con(p: Nat, at: Word(p), bt: Word(p), f: Bool, sk: Bool & Bool) ->
  Word(1n+p):
  (s, k) = sk
  WCon{s, Word.adc(p, at, bt, f, k)}

def Word.adc(n, a, b, f, c):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      match a b f:
        case WCon{ab, at} WCon{bb, bt} False{}:
          Word.adc.con(p, at, bt, False{}, Bool.full_add(ab, bb, c))
        case WCon{ab, at} WCon{bb, bt} True{}:
          Word.adc.con(p, at, bt, True{}, Bool.full_add(ab, Bool.not(bb), c))

def Word.add(n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  Word.adc(n, a, b, False{}, False{})

def Word.sub(n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  Word.adc(n, a, b, True{}, True{})

law Word.add_comm.arm:
  for p  : Nat
  for -h : Bool
  for at : Word(p)
  for bt : Word(p)
  for k  : Bool
  {WCon{h, Word.adc(p, at, bt, False{}, k)}
    == WCon{h, Word.adc(p, bt, at, False{}, k)} : Word.Con<p>}

law Word.add_comm.go:
  for n: Nat
  for a: Word(n)
  for b: Word(n)
  for c: Bool
  {Word.adc(n, a, b, False{}, c) == Word.adc(n, b, a, False{}, c) : Word(n)}

def Word.add_comm.arm(p, h, at, bt, k):
  Equal.cong(Word(p), Word(1n+p), w => WCon{h, w},
    Word.adc(p, at, bt, False{}, k), Word.adc(p, bt, at, False{}, k),
    Word.add_comm.go(p, at, bt, k))

def Word.add_comm.go(n, a, b, c):
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}:
          {==}
    case 1n+p:
      match a b c:
        case WCon{False{}, at} WCon{False{}, bt} False{}:
          Word.add_comm.arm(p, False{}, at, bt, False{})
        case WCon{False{}, at} WCon{False{}, bt} True{}:
          Word.add_comm.arm(p, True{}, at, bt, False{})
        case WCon{False{}, at} WCon{True{}, bt} False{}:
          Word.add_comm.arm(p, True{}, at, bt, False{})
        case WCon{False{}, at} WCon{True{}, bt} True{}:
          Word.add_comm.arm(p, False{}, at, bt, True{})
        case WCon{True{}, at} WCon{False{}, bt} False{}:
          Word.add_comm.arm(p, True{}, at, bt, False{})
        case WCon{True{}, at} WCon{False{}, bt} True{}:
          Word.add_comm.arm(p, False{}, at, bt, True{})
        case WCon{True{}, at} WCon{True{}, bt} False{}:
          Word.add_comm.arm(p, False{}, at, bt, True{})
        case WCon{True{}, at} WCon{True{}, bt} True{}:
          Word.add_comm.arm(p, True{}, at, bt, True{})

law Word.add_comm:
  for n: Nat
  for a: Word(n)
  for b: Word(n)
  {Word.add(n, a, b) == Word.add(n, b, a) : Word(n)}

def Word.add_comm(n, a, b):
  Word.add_comm.go(n, a, b, False{})

def Word.to_nat(n: Nat, w: Word(n)) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      match w:
        case WCon{False{}, t}:
          Nat.double(Word.to_nat(p, t))
        case WCon{True{}, t}:
          1n+Nat.double(Word.to_nat(p, t))

def Word.mul.go(+n: Nat, m: Nat, a: Word(m), +b: Word(n), acc: Word(n)) ->
  Word(n):
  match m:
    case 0n:
      acc
    case 1n+mp:
      match a:
        case WCon{False{}, at}:
          Word.mul.go(n, mp, at, Word.shl(n, b), acc)
        case WCon{True{}, at}:
          Word.mul.go(n, mp, at, Word.shl(n, b), Word.add(n, acc, b))

def Word.mul(+n: Nat, a: Word(n), b: Word(n)) -> Word(n):
  Word.mul.go(n, n, a, b, Word.zero(n))

# U32
# ---

def U32.inc(a: U32) -> U32:
  match a:
    case U32{x}:
      U32{Word.inc(32n, x)}

def U32.add(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.add(32n, x, y)}

law U32.add_comm:
  for a: U32
  for b: U32
  {U32.add(a, b) == U32.add(b, a) : U32}

def U32.add_comm(a, b):
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Word(32n), U32, w => U32{w}, Word.add(32n, x, y),
        Word.add(32n, y, x), Word.add_comm(32n, x, y))

def U32.sub(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.sub(32n, x, y)}

def U32.mul(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.mul(32n, x, y)}

def U32.not(a: U32) -> U32:
  match a:
    case U32{x}:
      U32{Word.not(32n, x)}

def U32.and(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.and(32n, x, y)}

def U32.or(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.or(32n, x, y)}

def U32.xor(a: U32, b: U32) -> U32:
  match a b:
    case U32{x} U32{y}:
      U32{Word.xor(32n, x, y)}

def U32.shl(a: U32) -> U32:
  match a:
    case U32{x}:
      U32{Word.shl(32n, x)}

def U32.shr(a: U32) -> U32:
  match a:
    case U32{x}:
      U32{Word.shr(32n, x)}

def U32.shln(a: U32, n: Nat) -> U32:
  match n:
    case 0n:
      a
    case 1n+p:
      U32.shl(U32.shln(a, p))

def U32.shrn(a: U32, n: Nat) -> U32:
  match n:
    case 0n:
      a
    case 1n+p:
      U32.shr(U32.shrn(a, p))

def U32.cmp(a: U32, b: U32) -> Cmp:
  match a b:
    case U32{x} U32{y}:
      Word.cmp(32n, x, y)

def U32.is_eq(a: U32, b: U32) -> Bool:
  Cmp.is_eq(U32.cmp(a, b))

def U32.is_ne(a: U32, b: U32) -> Bool:
  Bool.not(Cmp.is_eq(U32.cmp(a, b)))

def U32.is_lt(a: U32, b: U32) -> Bool:
  Cmp.is_lt(U32.cmp(a, b))

def U32.is_le(a: U32, b: U32) -> Bool:
  Cmp.is_le(U32.cmp(a, b))

def U32.is_gt(a: U32, b: U32) -> Bool:
  Cmp.is_gt(U32.cmp(a, b))

def U32.is_ge(a: U32, b: U32) -> Bool:
  Cmp.is_ge(U32.cmp(a, b))

def U32.is_zero(a: U32) -> Bool:
  U32.is_eq(a, 0)

def U32.to_nat(a: U32) -> Nat:
  match a:
    case U32{w}:
      Word.to_nat(32n, w)

def U32.from_nat(n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      U32.inc(U32.from_nat(p))

# U32 and Nat order
# ------------------
#
# U32's order is its value's: Word.cmp decides by the highest bit that
# differs, which is Nat.cmp of the two values (U32.cmp_nat). Nat's order
# is proven here once, and a U32 comparison is read as a Nat one: flipped
# (a < b is not b <= a), chained, and stepped by one (a < b + 1 is a <= b).

def Bool.if(b: Bool, t: Type, f: Type) -> Type:
  match b:
    case True{}:
      t
    case False{}:
      f

# anything, from False == True
def Bool.no(-g: Type, e: {False{} == True{} : Bool}) -> g:
  %e : Bool.if(_, g, Unit)
  Unit{}

# a conjunction that holds, taken apart
law Bool.and_l:
  for a: Bool
  for -b: Bool
  for e: {Bool.and(a, b) == True{} : Bool}
  {a == True{} : Bool}

def Bool.and_l(a, b, e):
  match a:
    case True{}:
      {==}
    case False{}:
      Bool.no({False{} == True{} : Bool}, e)

law Bool.and_r:
  for a: Bool
  for b: Bool
  for e: {Bool.and(a, b) == True{} : Bool}
  {b == True{} : Bool}

def Bool.and_r(a, b, e):
  match a:
    case True{}:
      e
    case False{}:
      Bool.no({b == True{} : Bool}, e)

# a number's bit below the rest: 2x, or 2x + 1
def Nat.bit(b: Bool, x: Nat) -> Nat:
  match b:
    case False{}:
      Nat.double(x)
    case True{}:
      1n+Nat.double(x)

law Nat.cmp_bit:
  for +ab: Bool
  for +bb: Bool
  for +x: Nat
  for +y: Nat
  {Nat.cmp(Nat.bit(ab, x), Nat.bit(bb, y)) == Word.cmp.fin(ab, bb, Nat.cmp(x, y)) : Cmp}

def Nat.cmp_bit(ba, bc, x, y):
  match ba bc x y:
    case False{} False{} 0n 0n:
      {==}
    case False{} True{} 0n 0n:
      {==}
    case True{} False{} 0n 0n:
      {==}
    case True{} True{} 0n 0n:
      {==}
    case False{} False{} 0n 1n+q:
      {==}
    case False{} True{} 0n 1n+q:
      {==}
    case True{} False{} 0n 1n+q:
      {==}
    case True{} True{} 0n 1n+q:
      {==}
    case False{} False{} 1n+p 0n:
      {==}
    case False{} True{} 1n+p 0n:
      {==}
    case True{} False{} 1n+p 0n:
      {==}
    case True{} True{} 1n+p 0n:
      {==}
    case False{} False{} 1n+p 1n+q:
      Nat.cmp_bit(False{}, False{}, p, q)
    case False{} True{} 1n+p 1n+q:
      Nat.cmp_bit(False{}, True{}, p, q)
    case True{} False{} 1n+p 1n+q:
      Nat.cmp_bit(True{}, False{}, p, q)
    case True{} True{} 1n+p 1n+q:
      Nat.cmp_bit(True{}, True{}, p, q)

law Word.to_nat_bit:
  for -p: Nat
  for b: Bool
  for -t: Word(p)
  {Word.to_nat(1n+p, WCon{b, t}) == Nat.bit(b, Word.to_nat(p, t)) : Nat}

def Word.to_nat_bit(p, b, t):
  match b:
    case False{}:
      {==}
    case True{}:
      {==}

def Word.cmp_nat.go(+p: Nat, +ab: Bool, +at: Word(p), +bb: Bool, +bt: Word(p)) ->
  {Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))) ==
    Nat.cmp(Word.to_nat(1n+p, WCon{ab, at}), Word.to_nat(1n+p, WCon{bb, bt})) : Cmp}:
  %Equal.sym(Nat, Word.to_nat(1n+p, WCon{ab, at}), Nat.bit(ab, Word.to_nat(p, at)),
    Word.to_nat_bit(p, ab, at)) :
    {Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))) ==
      Nat.cmp(_, Word.to_nat(1n+p, WCon{bb, bt})) : Cmp}
  %Equal.sym(Nat, Word.to_nat(1n+p, WCon{bb, bt}), Nat.bit(bb, Word.to_nat(p, bt)),
    Word.to_nat_bit(p, bb, bt)) :
    {Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))) ==
      Nat.cmp(Nat.bit(ab, Word.to_nat(p, at)), _) : Cmp}
  Equal.sym(Cmp, Nat.cmp(Nat.bit(ab, Word.to_nat(p, at)), Nat.bit(bb, Word.to_nat(p, bt))),
    Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))),
    Nat.cmp_bit(ab, bb, Word.to_nat(p, at), Word.to_nat(p, bt)))

law Word.cmp_nat:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.cmp(n, a, b) == Nat.cmp(Word.to_nat(n, a), Word.to_nat(n, b)) : Cmp}

def Word.cmp_nat(n, a, b):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{ab, at} WCon{bb, bt}:
          Equal.trans(Cmp, Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)),
            Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))),
            Nat.cmp(Word.to_nat(1n+p, WCon{ab, at}), Word.to_nat(1n+p, WCon{bb, bt})),
            Equal.cong(Cmp, Cmp, x => Word.cmp.fin(ab, bb, x), Word.cmp(p, at, bt),
              Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt)), Word.cmp_nat(p, at, bt)),
            Word.cmp_nat.go(p, ab, at, bb, bt))

# U32's order is its value's
law U32.cmp_nat:
  for a: U32
  for b: U32
  {U32.cmp(a, b) == Nat.cmp(U32.to_nat(a), U32.to_nat(b)) : Cmp}

def U32.cmp_nat(a, b):
  match a b:
    case U32{x} U32{y}:
      Word.cmp_nat(32n, x, y)

law U32.lt_nat:
  for a: U32
  for b: U32
  {U32.is_lt(a, b) == Nat.is_lt(U32.to_nat(a), U32.to_nat(b)) : Bool}

def U32.lt_nat(a, b):
  Equal.cong(Cmp, Bool, Cmp.is_lt, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)),
    U32.cmp_nat(a, b))

law U32.le_nat:
  for a: U32
  for b: U32
  {U32.is_le(a, b) == Nat.is_le(U32.to_nat(a), U32.to_nat(b)) : Bool}

def U32.le_nat(a, b):
  Equal.cong(Cmp, Bool, Cmp.is_le, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)),
    U32.cmp_nat(a, b))

law U32.eq_nat:
  for a: U32
  for b: U32
  {U32.is_eq(a, b) == Nat.is_eq(U32.to_nat(a), U32.to_nat(b)) : Bool}

def U32.eq_nat(a, b):
  Equal.cong(Cmp, Bool, Cmp.is_eq, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)),
    U32.cmp_nat(a, b))

# Nat's order: a < b is not b <= a
law Nat.lt_flip:
  for a: Nat
  for b: Nat
  {Bool.not(Nat.is_lt(a, b)) == Nat.is_le(b, a) : Bool}

def Nat.lt_flip(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Nat.lt_flip(p, q)

law Nat.le_flip:
  for a: Nat
  for b: Nat
  {Bool.not(Nat.is_le(a, b)) == Nat.is_lt(b, a) : Bool}

def Nat.le_flip(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Nat.le_flip(p, q)

def Nat.lt_succ.z(p: Nat) -> {Nat.is_lt(1n+p, 1n) == Nat.is_le(1n+p, 0n) : Bool}:
  match p:
    case 0n:
      {==}
    case 1n+r:
      {==}

# a < b + 1 is a <= b
law Nat.lt_succ:
  for a: Nat
  for b: Nat
  {Nat.is_lt(a, 1n+b) == Nat.is_le(a, b) : Bool}

def Nat.lt_succ(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Nat.lt_succ.z(p)
    case 1n+p 1n+q:
      Nat.lt_succ(p, q)

law Nat.is_le_trans:
  for a: Nat
  for b: Nat
  for c: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  for bc: {Nat.is_le(b, c) == True{} : Bool}
  {Nat.is_le(a, c) == True{} : Bool}

def Nat.is_le_trans(a, b, c, ab, bc):
  match a b c:
    case 0n _ 0n:
      {==}
    case 0n _ 1n+r:
      {==}
    case 1n+p 0n _:
      Bool.no({Nat.is_le(1n+p, c) == True{} : Bool}, ab)
    case 1n+p 1n+q 0n:
      Bool.no({Nat.is_le(1n+p, 0n) == True{} : Bool}, bc)
    case 1n+p 1n+q 1n+r:
      Nat.is_le_trans(p, q, r, ab, bc)

law Nat.is_lt_le_trans:
  for a: Nat
  for b: Nat
  for c: Nat
  for ab: {Nat.is_lt(a, b) == True{} : Bool}
  for bc: {Nat.is_le(b, c) == True{} : Bool}
  {Nat.is_lt(a, c) == True{} : Bool}

def Nat.is_lt_le_trans(a, b, c, ab, bc):
  match a b c:
    case 0n 0n _:
      Bool.no({Nat.is_lt(0n, c) == True{} : Bool}, ab)
    case 1n+p 0n _:
      Bool.no({Nat.is_lt(1n+p, c) == True{} : Bool}, ab)
    case 0n 1n+q 0n:
      Bool.no({Nat.is_lt(0n, 0n) == True{} : Bool}, bc)
    case 0n 1n+q 1n+r:
      {==}
    case 1n+p 1n+q 0n:
      Bool.no({Nat.is_lt(1n+p, 0n) == True{} : Bool}, bc)
    case 1n+p 1n+q 1n+r:
      Nat.is_lt_le_trans(p, q, r, ab, bc)

# a <= b and not a < b: a is b
law Nat.is_le_eq:
  for a: Nat
  for b: Nat
  for le: {Nat.is_le(a, b) == True{} : Bool}
  for nlt: {Nat.is_lt(a, b) == False{} : Bool}
  {Nat.is_eq(a, b) == True{} : Bool}

def Nat.is_le_eq(a, b, le, nlt):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Bool.no({Nat.is_eq(0n, 1n+q) == True{} : Bool}, Equal.sym(Bool, True{}, False{}, nlt))
    case 1n+p 0n:
      Bool.no({Nat.is_eq(1n+p, 0n) == True{} : Bool}, le)
    case 1n+p 1n+q:
      Nat.is_le_eq(p, q, le, nlt)

# a Bool that is not True is False; a disjunction with a True in it holds
law Bool.not_true:
  for a: Bool
  for e: {Bool.not(a) == True{} : Bool}
  {a == False{} : Bool}

def Bool.not_true(a, e):
  match a:
    case False{}:
      {==}
    case True{}:
      Bool.no({True{} == False{} : Bool}, e)

law Bool.or_true:
  for a: Bool
  {Bool.or(a, True{}) == True{} : Bool}

def Bool.or_true(a):
  match a:
    case False{}:
      {==}
    case True{}:
      {==}

# U32's order, read as Nat's
law U32.lt_flip:
  for +a: U32
  for +b: U32
  {Bool.not(U32.is_lt(a, b)) == U32.is_le(b, a) : Bool}

def U32.lt_flip(a, b):
  %Equal.sym(Bool, U32.is_lt(a, b), Nat.is_lt(U32.to_nat(a), U32.to_nat(b)), U32.lt_nat(a, b)) :
    {Bool.not(_) == U32.is_le(b, a) : Bool}
  %Equal.sym(Bool, U32.is_le(b, a), Nat.is_le(U32.to_nat(b), U32.to_nat(a)), U32.le_nat(b, a)) :
    {Bool.not(Nat.is_lt(U32.to_nat(a), U32.to_nat(b))) == _ : Bool}
  Nat.lt_flip(U32.to_nat(a), U32.to_nat(b))

law U32.le_flip:
  for +a: U32
  for +b: U32
  {Bool.not(U32.is_le(a, b)) == U32.is_lt(b, a) : Bool}

def U32.le_flip(a, b):
  %Equal.sym(Bool, U32.is_le(a, b), Nat.is_le(U32.to_nat(a), U32.to_nat(b)), U32.le_nat(a, b)) :
    {Bool.not(_) == U32.is_lt(b, a) : Bool}
  %Equal.sym(Bool, U32.is_lt(b, a), Nat.is_lt(U32.to_nat(b), U32.to_nat(a)), U32.lt_nat(b, a)) :
    {Bool.not(Nat.is_le(U32.to_nat(a), U32.to_nat(b))) == _ : Bool}
  Nat.le_flip(U32.to_nat(a), U32.to_nat(b))

# c < b is c <= a, when b is a + 1
law U32.lt_succ:
  for +c: U32
  for +a: U32
  for +b: U32
  for e: {U32.to_nat(b) == 1n+U32.to_nat(a) : Nat}
  {U32.is_lt(c, b) == U32.is_le(c, a) : Bool}

def U32.lt_succ(c, a, b, e):
  %Equal.sym(Bool, U32.is_lt(c, b), Nat.is_lt(U32.to_nat(c), U32.to_nat(b)), U32.lt_nat(c, b)) :
    {_ == U32.is_le(c, a) : Bool}
  %Equal.sym(Bool, U32.is_le(c, a), Nat.is_le(U32.to_nat(c), U32.to_nat(a)), U32.le_nat(c, a)) :
    {Nat.is_lt(U32.to_nat(c), U32.to_nat(b)) == _ : Bool}
  %Equal.sym(Nat, U32.to_nat(b), 1n+U32.to_nat(a), e) :
    {Nat.is_lt(U32.to_nat(c), _) == Nat.is_le(U32.to_nat(c), U32.to_nat(a)) : Bool}
  Nat.lt_succ(U32.to_nat(c), U32.to_nat(a))

law U32.le_trans:
  for +a: U32
  for +b: U32
  for +c: U32
  for ab: {U32.is_le(a, b) == True{} : Bool}
  for bc: {U32.is_le(b, c) == True{} : Bool}
  {U32.is_le(a, c) == True{} : Bool}

def U32.le_trans(a, b, c, ab, bc):
  %Equal.sym(Bool, U32.is_le(a, c), Nat.is_le(U32.to_nat(a), U32.to_nat(c)), U32.le_nat(a, c)) :
    {_ == True{} : Bool}
  Nat.is_le_trans(U32.to_nat(a), U32.to_nat(b), U32.to_nat(c),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(a), U32.to_nat(b)), U32.is_le(a, b), True{},
      Equal.sym(Bool, U32.is_le(a, b), Nat.is_le(U32.to_nat(a), U32.to_nat(b)), U32.le_nat(a, b)), ab),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(b), U32.to_nat(c)), U32.is_le(b, c), True{},
      Equal.sym(Bool, U32.is_le(b, c), Nat.is_le(U32.to_nat(b), U32.to_nat(c)), U32.le_nat(b, c)), bc))

law U32.lt_le_trans:
  for +a: U32
  for +b: U32
  for +c: U32
  for ab: {U32.is_lt(a, b) == True{} : Bool}
  for bc: {U32.is_le(b, c) == True{} : Bool}
  {U32.is_lt(a, c) == True{} : Bool}

def U32.lt_le_trans(a, b, c, ab, bc):
  %Equal.sym(Bool, U32.is_lt(a, c), Nat.is_lt(U32.to_nat(a), U32.to_nat(c)), U32.lt_nat(a, c)) :
    {_ == True{} : Bool}
  Nat.is_lt_le_trans(U32.to_nat(a), U32.to_nat(b), U32.to_nat(c),
    Equal.trans(Bool, Nat.is_lt(U32.to_nat(a), U32.to_nat(b)), U32.is_lt(a, b), True{},
      Equal.sym(Bool, U32.is_lt(a, b), Nat.is_lt(U32.to_nat(a), U32.to_nat(b)), U32.lt_nat(a, b)), ab),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(b), U32.to_nat(c)), U32.is_le(b, c), True{},
      Equal.sym(Bool, U32.is_le(b, c), Nat.is_le(U32.to_nat(b), U32.to_nat(c)), U32.le_nat(b, c)), bc))

# lo <= c < lo + 1: c is lo
law U32.eq_of_lt_succ:
  for +c: U32
  for +lo: U32
  for +hi: U32
  for ge: {Bool.not(U32.is_lt(c, lo)) == True{} : Bool}
  for lt: {U32.is_lt(c, hi) == True{} : Bool}
  for e: {U32.to_nat(hi) == 1n+U32.to_nat(lo) : Nat}
  {U32.is_eq(c, lo) == True{} : Bool}

def U32.eq_of_lt_succ(c, lo, hi, ge, lt, e):
  %Equal.sym(Bool, U32.is_eq(c, lo), Nat.is_eq(U32.to_nat(c), U32.to_nat(lo)), U32.eq_nat(c, lo)) :
    {_ == True{} : Bool}
  Nat.is_le_eq(U32.to_nat(c), U32.to_nat(lo),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(c), U32.to_nat(lo)), U32.is_le(c, lo), True{},
      Equal.sym(Bool, U32.is_le(c, lo), Nat.is_le(U32.to_nat(c), U32.to_nat(lo)), U32.le_nat(c, lo)),
      Equal.trans(Bool, U32.is_le(c, lo), U32.is_lt(c, hi), True{},
        Equal.sym(Bool, U32.is_lt(c, hi), U32.is_le(c, lo), U32.lt_succ(c, lo, hi, e)), lt)),
    Equal.trans(Bool, Nat.is_lt(U32.to_nat(c), U32.to_nat(lo)), U32.is_lt(c, lo), False{},
      Equal.sym(Bool, U32.is_lt(c, lo), Nat.is_lt(U32.to_nat(c), U32.to_nat(lo)), U32.lt_nat(c, lo)),
      Bool.not_true(U32.is_lt(c, lo), ge)))

# nothing is below zero
def Nat.lt_zero(-g: Type, k: Nat, e: {Nat.is_lt(k, 0n) == True{} : Bool}) -> g:
  match k:
    case 0n:
      Bool.no(g, e)
    case 1n+j:
      Bool.no(g, e)

def U32.log2.go(+k: Nat, +n: U32, +d: Nat) -> Nat:
  match k:
    case 0n:
      d
    case 1n+p:
      U32.log2.go(p, U32.shr(n),
        Nat.add(d, U32.to_nat(Bool.to_u32(U32.is_lt(1, n)))))

def U32.log2(+n: U32) -> Nat:
  U32.log2.go(32n, n, 0n)

def U32.divmod.go.fin(-p: Nat, q: Word(p), s: U32, b: U32, g: Bool) ->
  Word(1n+p) & U32:
  match g:
    case True{}:
      (WCon{True{}, q}, U32.sub(s, b))
    case False{}:
      (WCon{False{}, q}, s)

def U32.divmod.go.shl(-p: Nat, q: Word(p), +b: U32, ts: Bool & Word(32n)) ->
  Word(1n+p) & U32:
  (t, s) = ts
  +s2 = {U32{s} : U32}
  U32.divmod.go.fin(p, q, s2, b, Bool.or(t, U32.is_ge(s2, b)))

def U32.divmod.go.rec(-p: Nat, a0: Bool, +b: U32, qr: Word(p) & U32) ->
  Word(1n+p) & U32:
  (q, r) = qr
  U32{rw} = r
  U32.divmod.go.shl(p, q, b, Word.shl.out(32n, a0, rw))

def U32.divmod.go(m: Nat, a: Word(m), +b: U32) -> Word(m) & U32:
  match m a:
    case 0n WNil{}:
      (WNil{}, 0)
    case 1n+p WCon{a0, hi}:
      U32.divmod.go.rec(p, a0, b, U32.divmod.go(p, hi, b))

def U32.div.fin(qr: Word(32n) & U32) -> U32:
  (q, r) = qr
  U32{q}

def U32.div.if(aw: Word(32n), b: U32, z: Bool) -> U32:
  match z:
    case True{}:
      0
    case False{}:
      U32.div.fin(U32.divmod.go(32n, aw, b))

def U32.div(a: U32, +b: U32) -> U32:
  U32{aw} = a
  U32.div.if(aw, b, U32.is_zero(b))

def U32.mod.fin(qr: Word(32n) & U32) -> U32:
  (q, r) = qr
  r

def U32.mod.if(aw: Word(32n), b: U32, z: Bool) -> U32:
  match z:
    case True{}:
      U32{aw}
    case False{}:
      U32.mod.fin(U32.divmod.go(32n, aw, b))

def U32.mod(a: U32, +b: U32) -> U32:
  U32{aw} = a
  U32.mod.if(aw, b, U32.is_zero(b))

def U32.min(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_lt(a, b), a, b)

def U32.max(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_lt(a, b), b, a)

def U32.clamp(x: U32, lo: U32, hi: U32) -> U32:
  U32.min(U32.max(x, lo), hi)

def U32.pow(+a: U32, n: Nat) -> U32:
  match n:
    case 0n:
      1
    case 1n+p:
      U32.mul(a, U32.pow(a, p))

def U32.is_even(a: U32) -> Bool:
  U32.is_zero(U32.and(a, 1))

# F32
# ---

law U32.to_f32:
  for a: U32
  F32

law U32.to_f64:
  for a: U32
  F64

law F32.to_u32:
  for a: F32
  U32

law F32.add:
  for a: F32
  for b: F32
  F32

law F32.sub:
  for a: F32
  for b: F32
  F32

law F32.mul:
  for a: F32
  for b: F32
  F32

law F32.div:
  for a: F32
  for b: F32
  F32

law F32.mod:
  for a: F32
  for b: F32
  F32

law F32.pow:
  for a: F32
  for b: F32
  F32

law F32.atan2:
  for a: F32
  for b: F32
  F32

law F32.neg:
  for a: F32
  F32

law F32.abs:
  for a: F32
  F32

law F32.sqrt:
  for a: F32
  F32

law F32.exp:
  for a: F32
  F32

law F32.log:
  for a: F32
  F32

law F32.log2:
  for a: F32
  F32

law F32.log10:
  for a: F32
  F32

law F32.sin:
  for a: F32
  F32

law F32.cos:
  for a: F32
  F32

law F32.tan:
  for a: F32
  F32

law F32.asin:
  for a: F32
  F32

law F32.acos:
  for a: F32
  F32

law F32.atan:
  for a: F32
  F32

law F32.sinh:
  for a: F32
  F32

law F32.cosh:
  for a: F32
  F32

law F32.tanh:
  for a: F32
  F32

law F32.floor:
  for a: F32
  F32

law F32.ceil:
  for a: F32
  F32

law F32.trunc:
  for a: F32
  F32

law F32.is_eq:
  for a: F32
  for b: F32
  Bool

law F32.is_ne:
  for a: F32
  for b: F32
  Bool

law F32.is_lt:
  for a: F32
  for b: F32
  Bool

law F32.is_le:
  for a: F32
  for b: F32
  Bool

law F32.is_gt:
  for a: F32
  for b: F32
  Bool

law F32.is_ge:
  for a: F32
  for b: F32
  Bool

law F32.show:
  for +a: F32
  String

law F32.bits:
  for a: F32
  U32

law F32.read:
  for s: String
  Maybe<&2, F32>

def F32.min(+a: F32, +b: F32) -> F32:
  Bool.pick(F32, F32.is_lt(a, b), a, b)

def F32.max(+a: F32, +b: F32) -> F32:
  Bool.pick(F32, F32.is_lt(a, b), b, a)

def F32.clamp(x: F32, lo: F32, hi: F32) -> F32:
  F32.min(F32.max(x, lo), hi)

def F32.lerp(+a: F32, b: F32, t: F32) -> F32:
  F32.add(a, F32.mul(F32.sub(b, a), t))

def F32.square(+a: F32) -> F32:
  F32.mul(a, a)

def F32.hypot(+x: F32, +y: F32) -> F32:
  F32.sqrt(F32.add(F32.mul(x, x), F32.mul(y, y)))

def F32.round(a: F32) -> F32:
  F32.floor(F32.add(a, 0.5))

def F32.pi() -> F32:
  3.14159265

def F32.from_nat(n: Nat) -> F32:
  U32.to_f32(U32.from_nat(n))

def F32.to_nat(a: F32) -> Nat:
  U32.to_nat(F32.to_u32(a))

# U64
# ---
#
# The 64-bit word as a number: the same Word(64n) slot F64 uses, with integer
# operations. The compiler lowers these natively (the u64_ op family), so a
# call is one machine operation, not the width-64 bit walk. The add_comm law
# and the signed twin live in the bend-u64 / bend-i64 packages until they
# earn their place here.

type U64 is Data:
  U64{data: Word(64n)}

def U64.inc(a: U64) -> U64:
  match a:
    case U64{x}:
      U64{Word.inc(64n, x)}

def U64.add(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.add(64n, x, y)}

law U64.add_comm:
  for a: U64
  for b: U64
  {U64.add(a, b) == U64.add(b, a) : U64}

def U64.add_comm(a, b):
  match a b:
    case U64{x} U64{y}:
      Equal.cong(Word(64n), U64, w => U64{w}, Word.add(64n, x, y),
        Word.add(64n, y, x), Word.add_comm(64n, x, y))

def U64.sub(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.sub(64n, x, y)}

def U64.mul(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.mul(64n, x, y)}

def U64.not(a: U64) -> U64:
  match a:
    case U64{x}:
      U64{Word.not(64n, x)}

def U64.and(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.and(64n, x, y)}

def U64.or(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.or(64n, x, y)}

def U64.xor(a: U64, b: U64) -> U64:
  match a b:
    case U64{x} U64{y}:
      U64{Word.xor(64n, x, y)}

def U64.shl(a: U64) -> U64:
  match a:
    case U64{x}:
      U64{Word.shl(64n, x)}

def U64.shr(a: U64) -> U64:
  match a:
    case U64{x}:
      U64{Word.shr(64n, x)}

def U64.shln(a: U64, n: Nat) -> U64:
  match n:
    case 0n:
      a
    case 1n+p:
      U64.shl(U64.shln(a, p))

def U64.shrn(a: U64, n: Nat) -> U64:
  match n:
    case 0n:
      a
    case 1n+p:
      U64.shr(U64.shrn(a, p))

def U64.cmp(a: U64, b: U64) -> Cmp:
  match a b:
    case U64{x} U64{y}:
      Word.cmp(64n, x, y)

def U64.is_eq(a: U64, b: U64) -> Bool:
  Cmp.is_eq(U64.cmp(a, b))

def U64.is_ne(a: U64, b: U64) -> Bool:
  Bool.not(Cmp.is_eq(U64.cmp(a, b)))

def U64.is_lt(a: U64, b: U64) -> Bool:
  Cmp.is_lt(U64.cmp(a, b))

def U64.is_le(a: U64, b: U64) -> Bool:
  Cmp.is_le(U64.cmp(a, b))

def U64.is_gt(a: U64, b: U64) -> Bool:
  Cmp.is_gt(U64.cmp(a, b))

def U64.is_ge(a: U64, b: U64) -> Bool:
  Cmp.is_ge(U64.cmp(a, b))

def U64.is_zero(a: U64) -> Bool:
  match a:
    case U64{x}:
      Cmp.is_eq(Word.cmp(64n, x, Word.zero(64n)))

# I64
# ---
#
# The signed twin of U64: the same Word(64n) slot read as two's complement.
# add, sub, mul and the bitwise ops are bit-identical to the unsigned ones,
# so they share their lowering; the signed layer is negation (0 - x), the
# comparison that biases both sides by the sign bit before the unsigned
# compare, and the arithmetic shift that replicates the sign. The compiler
# lowers all of it natively (the i64_ family), so a signed compare is one
# machine compare, not a bias and a width-64 bit walk.

type I64 is Data:
  I64{data: Word(64n)}

def I64.sign_mask() -> Word(64n):
  Word.not(64n, Word.shr(64n, Word.not(64n, Word.zero(64n))))

def I64.inc(a: I64) -> I64:
  match a:
    case I64{x}:
      I64{Word.inc(64n, x)}

def I64.add(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.add(64n, x, y)}

law I64.add_comm:
  for a: I64
  for b: I64
  {I64.add(a, b) == I64.add(b, a) : I64}

# The signed twin of U32.add_comm, and the same citation: two's complement
# means I64.add IS Word.add(64n) -- the proof is the unsigned one, repacked.
def I64.add_comm(a, b):
  match a b:
    case I64{x} I64{y}:
      Equal.cong(Word(64n), I64, w => I64{w}, Word.add(64n, x, y),
        Word.add(64n, y, x), Word.add_comm(64n, x, y))

def I64.sub(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.sub(64n, x, y)}

def I64.mul(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.mul(64n, x, y)}

def I64.neg(a: I64) -> I64:
  match a:
    case I64{x}:
      I64{Word.sub(64n, Word.zero(64n), x)}

def I64.not(a: I64) -> I64:
  match a:
    case I64{x}:
      I64{Word.not(64n, x)}

def I64.and(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.and(64n, x, y)}

def I64.or(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.or(64n, x, y)}

def I64.xor(a: I64, b: I64) -> I64:
  match a b:
    case I64{x} I64{y}:
      I64{Word.xor(64n, x, y)}

def I64.shl(a: I64) -> I64:
  match a:
    case I64{x}:
      I64{Word.shl(64n, x)}

def I64.shr(a: I64) -> I64:
  match a:
    case I64{x}:
      I64{Word.shr(64n, x)}

def I64.shln(a: I64, n: Nat) -> I64:
  match n:
    case 0n:
      a
    case 1n+p:
      I64.shl(I64.shln(a, p))

def I64.shrn(a: I64, n: Nat) -> I64:
  match n:
    case 0n:
      a
    case 1n+p:
      I64.shr(I64.shrn(a, p))

def I64.is_neg(a: I64) -> Bool:
  match a:
    case I64{x}:
      Cmp.is_ge(Word.cmp(64n, x, I64.sign_mask()))

def I64.shr.s.put(neg: Bool, a: I64) -> I64:
  match neg:
    case True{}:
      I64.or(a, I64{I64.sign_mask()})
    case False{}:
      a

def I64.shr.s(+a: I64) -> I64:
  I64.shr.s.put(I64.is_neg(a), I64.shr(a))

def I64.shr.s.n(a: I64, n: Nat) -> I64:
  match n:
    case 0n:
      a
    case 1n+p:
      I64.shr.s(I64.shr.s.n(a, p))

def I64.cmp(a: I64, b: I64) -> Cmp:
  match a b:
    case I64{x} I64{y}:
      Word.cmp(64n, Word.xor(64n, x, I64.sign_mask()),
        Word.xor(64n, y, I64.sign_mask()))

def I64.is_eq(a: I64, b: I64) -> Bool:
  Cmp.is_eq(I64.cmp(a, b))

def I64.is_ne(a: I64, b: I64) -> Bool:
  Bool.not(Cmp.is_eq(I64.cmp(a, b)))

def I64.is_lt(a: I64, b: I64) -> Bool:
  Cmp.is_lt(I64.cmp(a, b))

def I64.is_le(a: I64, b: I64) -> Bool:
  Cmp.is_le(I64.cmp(a, b))

def I64.is_gt(a: I64, b: I64) -> Bool:
  Cmp.is_gt(I64.cmp(a, b))

def I64.is_ge(a: I64, b: I64) -> Bool:
  Cmp.is_ge(I64.cmp(a, b))

def I64.is_zero(a: I64) -> Bool:
  match a:
    case I64{x}:
      Cmp.is_eq(Word.cmp(64n, x, Word.zero(64n)))

# F64
# ---

law F32.to_f64:
  for a: F32
  F64

law F64.to_f32:
  for a: F64
  F32

law F64.to_u32:
  for a: F64
  U32

law F64.add:
  for a: F64
  for b: F64
  F64

law F64.sub:
  for a: F64
  for b: F64
  F64

law F64.mul:
  for a: F64
  for b: F64
  F64

law F64.div:
  for a: F64
  for b: F64
  F64

law F64.mod:
  for a: F64
  for b: F64
  F64

law F64.pow:
  for a: F64
  for b: F64
  F64

law F64.atan2:
  for a: F64
  for b: F64
  F64

law F64.neg:
  for a: F64
  F64

law F64.abs:
  for a: F64
  F64

law F64.sqrt:
  for a: F64
  F64

law F64.exp:
  for a: F64
  F64

law F64.log:
  for a: F64
  F64

law F64.log2:
  for a: F64
  F64

law F64.log10:
  for a: F64
  F64

law F64.sin:
  for a: F64
  F64

law F64.cos:
  for a: F64
  F64

law F64.tan:
  for a: F64
  F64

law F64.asin:
  for a: F64
  F64

law F64.acos:
  for a: F64
  F64

law F64.atan:
  for a: F64
  F64

law F64.sinh:
  for a: F64
  F64

law F64.cosh:
  for a: F64
  F64

law F64.tanh:
  for a: F64
  F64

law F64.floor:
  for a: F64
  F64

law F64.ceil:
  for a: F64
  F64

law F64.trunc:
  for a: F64
  F64

law F64.is_eq:
  for a: F64
  for b: F64
  Bool

law F64.is_ne:
  for a: F64
  for b: F64
  Bool

law F64.is_lt:
  for a: F64
  for b: F64
  Bool

law F64.is_le:
  for a: F64
  for b: F64
  Bool

law F64.is_gt:
  for a: F64
  for b: F64
  Bool

law F64.is_ge:
  for a: F64
  for b: F64
  Bool

law F64.show:
  for +a: F64
  String

law F64.bits:
  for a: F64
  Nat

law F64.read:
  for s: String
  Maybe<&2, F64>

# Bool.pick at F64, monomorphic: a double never rides a boxed parameter
def F64.pick(c: Bool, a: F64, b: F64) -> F64:
  match c:
    case False{}:
      b
    case True{}:
      a

def F64.min(+a: F64, +b: F64) -> F64:
  F64.pick(F64.is_lt(a, b), a, b)

def F64.max(+a: F64, +b: F64) -> F64:
  F64.pick(F64.is_lt(a, b), b, a)

def F64.clamp(x: F64, lo: F64, hi: F64) -> F64:
  F64.min(F64.max(x, lo), hi)

def F64.lerp(+a: F64, b: F64, t: F64) -> F64:
  F64.add(a, F64.mul(F64.sub(b, a), t))

def F64.square(+a: F64) -> F64:
  F64.mul(a, a)

def F64.hypot(+x: F64, +y: F64) -> F64:
  F64.sqrt(F64.add(F64.mul(x, x), F64.mul(y, y)))

def F64.round(a: F64) -> F64:
  F64.floor(F64.add(a, 0.5d))

def F64.pi() -> F64:
  3.141592653589793d

def F64.from_nat(n: Nat) -> F64:
  U32.to_f64(U32.from_nat(n))

def F64.to_nat(a: F64) -> Nat:
  U32.to_nat(F64.to_u32(a))

# Char
# ----

def Char.cmp(a: Char, b: Char) -> (Char & Char) & Cmp:
  match a b:
    case Chr{+x} Chr{+y}:
      ((Chr{x}, Chr{y}), U32.cmp(x, y))

def Char.to_u32(c: Char) -> U32:
  match c:
    case Chr{x}:
      x

def Char.from_u32(x: U32) -> Char:
  Chr{x}

def Char.is_eq(a: Char, b: Char) -> Bool:
  match a b:
    case Chr{x} Chr{y}:
      U32.is_eq(x, y)

def Char.is_digit(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.and(U32.is_ge(x, 48), U32.is_le(x, 57))

def Char.is_upper(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.and(U32.is_ge(x, 65), U32.is_le(x, 90))

def Char.is_lower(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.and(U32.is_ge(x, 97), U32.is_le(x, 122))

def Char.is_alpha(+c: Char) -> Bool:
  Bool.or(Char.is_upper(c), Char.is_lower(c))

def Char.is_space(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.or(U32.is_eq(x, 32), Bool.and(U32.is_ge(x, 9), U32.is_le(x, 13)))

def Char.is_line_break(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.or(Bool.and(U32.is_ge(x, 10), U32.is_le(x, 13)),
        Bool.or(Bool.and(U32.is_ge(x, 28), U32.is_le(x, 30)),
        Bool.or(U32.is_eq(x, 133),
        Bool.or(U32.is_eq(x, 8232), U32.is_eq(x, 8233)))))

def Char.to_upper(+c: Char) -> Char:
  Chr{U32.sub(Char.to_u32(c), U32.mul(Bool.to_u32(Char.is_lower(c)), 32))}

def Char.to_lower(+c: Char) -> Char:
  Chr{U32.add(Char.to_u32(c), U32.mul(Bool.to_u32(Char.is_upper(c)), 32))}

# String
# ------

def Utf8.Dec.eof(dec: Utf8.Dec) -> Bool:
  Dec{pend, need} = dec
  U32.is_eq(need, 4)

# File.fold_text folds step over a file's File.read_text chunks (max bytes
# each, fuel chunks at most) and closes it; a read error dies as IO.try.
def File.fold_text.end(-S: Type, eof: Bool, file: File, dec: Utf8.Dec, st: S,
  again: File -> Utf8.Dec -> S -> IO(S)) -> IO(S):
  match eof:
    case True{}:
      IO.bind(Unit, S, File.close(file), u => IO.pure(S, st))
    case False{}:
      again(file, dec, st)

def File.fold_text.go(-S: Type, step: S -> String -> S, st: S,
  again: File -> Utf8.Dec -> S -> IO(S), got: Utf8.Dec.Out(File)) -> IO(S):
  (file, r) = got
  match r:
    case Fail{(code, msg)}:
      IO.die(S, code, msg)
    case Done{(+dec, text)}:
      File.fold_text.end(S, Utf8.Dec.eof(dec), file, dec, step(st, text), again)

def File.fold_text(~S: Type, ~step: S -> String -> S, fuel: Nat, file: File,
  dec: Utf8.Dec, +max: U32, st: S) -> IO(S):
  match fuel:
    case 0n:
      IO.die(S, 1, "File.fold_text: out of fuel")
    case 1n+left:
      IO.bind(Utf8.Dec.Out(File), S, File.read_text(file, dec, max),
        File.fold_text.go(S, step, st,
          f => d => s => File.fold_text(~S, ~step, left, f, d, max, s)))

def String.append(a: String, b: String) -> String:
  match a:
    case SNil{}:
      b
    case SCon{h, t}:
      SCon{h, String.append(t, b)}

law String.order:
  for a: String
  for b: String
  Cmp

def String.order.step(a: String, b: String, c: Cmp) -> Cmp:
  match c:
    case LT{}:
      LT{}
    case EQ{}:
      String.order(a, b)
    case GT{}:
      GT{}

def String.order(a, b):
  match a b:
    case SNil{} SNil{}:
      EQ{}
    case SNil{} SCon{h, t}:
      LT{}
    case SCon{h, t} SNil{}:
      GT{}
    case SCon{Chr{x}, xt} SCon{Chr{y}, yt}:
      String.order.step(xt, yt, U32.cmp(x, y))

law String.cmp:
  for a: String
  for b: String
  (String & String) & Cmp

def String.cmp(a, b):
  +x = a
  +y = b
  ((x, y), String.order(x, y))

def String.eq(a: String, b: String) -> Bool:
  Cmp.is_eq(String.order(a, b))

def String.length.go(s: String, n: Nat) -> Nat:
  match s:
    case SNil{}:
      n
    case SCon{h, t}:
      String.length.go(t, 1n+n)

def String.length(s: String) -> Nat:
  String.length.go(s, 0n)

def String.is_empty(s: String) -> Bool:
  match s:
    case SNil{}:
      True{}
    case SCon{h, t}:
      False{}

def String.reverse.go(s: String, acc: String) -> String:
  match s:
    case SNil{}:
      acc
    case SCon{h, t}:
      String.reverse.go(t, SCon{h, acc})

def String.reverse(s: String) -> String:
  String.reverse.go(s, SNil{})

def String.is_lt(a: String, b: String) -> Bool:
  Cmp.is_lt(String.order(a, b))

def String.is_le(a: String, b: String) -> Bool:
  Cmp.is_le(String.order(a, b))

def String.is_gt(a: String, b: String) -> Bool:
  Cmp.is_gt(String.order(a, b))

def String.is_ge(a: String, b: String) -> Bool:
  Cmp.is_ge(String.order(a, b))

law String.starts_with:
  for s: String
  for p: String
  Bool

def String.starts_with.if(t: String, pt: String, same: Bool) -> Bool:
  match same:
    case False{}:
      False{}
    case True{}:
      String.starts_with(t, pt)

def String.starts_with(s, p):
  match s p:
    case SNil{} SNil{}:
      True{}
    case SNil{} SCon{h, t}:
      False{}
    case SCon{h, t} SNil{}:
      True{}
    case SCon{x, xt} SCon{y, yt}:
      String.starts_with.if(xt, yt, Char.is_eq(x, y))

law String.contains:
  for +s: String
  for +p: String
  Bool

def String.contains.if(t: String, p: String, here: Bool) -> Bool:
  match here:
    case False{}:
      String.contains(t, p)
    case True{}:
      True{}

def String.contains(s, p):
  match s:
    case SNil{}:
      String.is_empty(p)
    case SCon{h, t}:
      String.contains.if(t, p, String.starts_with(SCon{h, t}, p))

def String.take(s: String, n: Nat) -> String:
  match s n:
    case SNil{} _:
      SNil{}
    case SCon{h, t} 0n:
      SNil{}
    case SCon{h, t} 1n+p:
      SCon{h, String.take(t, p)}

def String.drop(s: String, n: Nat) -> String:
  match s n:
    case SNil{} _:
      SNil{}
    case SCon{h, t} 0n:
      SCon{h, t}
    case SCon{h, t} 1n+p:
      String.drop(t, p)

def String.ends_with(+s: String, +p: String) -> Bool:
  String.eq(String.drop(s, Nat.sub(String.length(s), String.length(p))), p)

def String.get(s: String, n: Nat) -> Maybe<&2, Char>:
  match s n:
    case SNil{} _:
      None{}
    case SCon{h, t} 0n:
      Some{h}
    case SCon{h, t} 1n+p:
      String.get(t, p)

def String.to_list(s: String) -> List<&2, Char>:
  match s:
    case SNil{}:
      Nil{}
    case SCon{h, t}:
      h <> String.to_list(t)

def String.from_list(cs: List<&2, Char>) -> String:
  match cs:
    case Nil{}:
      SNil{}
    case h <> t:
      SCon{h, String.from_list(t)}

def String.concat(xs: List<&2, String>) -> String:
  match xs:
    case Nil{}:
      SNil{}
    case h <> t:
      String.append(h, String.concat(t))

def String.join.go(xs: List<&2, String>, h: String, +sep: String) -> String:
  match xs:
    case Nil{}:
      h
    case h2 <> t:
      String.append(h, String.append(sep, String.join.go(t, h2, sep)))

def String.join(xs: List<&2, String>, +sep: String) -> String:
  match xs:
    case Nil{}:
      SNil{}
    case h <> t:
      String.join.go(t, h, sep)

def String.split.push(c: Char, ps: List<&2, String>) -> List<&2, String>:
  match ps:
    case Nil{}:
      [SCon{c, SNil{}}]
    case h <> t:
      SCon{c, h} <> t

# Accumulators keep the reference split stack bounded on large input.
# The current field and completed fields are reversed independently.
law String.split.acc:
  for s: String
  for +sep: Char
  for +space: Bool
  for word: String
  for words: List<&2, String>
  List<&2, String>

def String.split.test(c: Char, sep: Char, space: Bool) -> Bool:
  match space:
    case False{}:
      Char.is_eq(c, sep)
    case True{}:
      Char.is_space(c)

def String.split.step(t: String, sep: Char, space: Bool, h: Char,
  word: String, words: List<&2, String>, cut: Bool) -> List<&2, String>:
  match cut:
    case False{}:
      String.split.acc(t, sep, space, SCon{h, word}, words)
    case True{}:
      String.split.acc(t, sep, space, "", String.reverse(word) <> words)

def String.split.acc(s, sep, space, word, words):
  match s:
    case SNil{}:
      List.reverse(&2, String, String.reverse(word) <> words)
    case SCon{+h, t}:
      String.split.step(t, sep, space, h, word, words,
        String.split.test(h, sep, space))

def String.split(s: String, +sep: Char) -> List<&2, String>:
  String.split.acc(s, sep, False{}, "", [])

def String.lines(s: String) -> List<&2, String>:
  String.split(s, '\n')

def String.repeat.go(n: Nat, +s: String) -> String:
  match n:
    case 0n:
      SNil{}
    case 1n+p:
      String.append(s, String.repeat.go(p, s))

def String.repeat(+s: String, n: Nat) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{h, t}:
      String.repeat.go(n, SCon{h, t})

def String.to_upper(s: String) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{h, t}:
      SCon{Char.to_upper(h), String.to_upper(t)}

def String.to_lower(s: String) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{h, t}:
      SCon{Char.to_lower(h), String.to_lower(t)}

law String.trim_start:
  for s: String
  String

def String.trim_start.if(h: Char, t: String, space: Bool) -> String:
  match space:
    case False{}:
      SCon{h, t}
    case True{}:
      String.trim_start(t)

def String.trim_start(s):
  match s:
    case SNil{}:
      SNil{}
    case SCon{+h, t}:
      String.trim_start.if(h, t, Char.is_space(h))

def String.trim_end(s: String) -> String:
  String.reverse(String.trim_start(String.reverse(s)))

def String.trim(s: String) -> String:
  String.trim_end(String.trim_start(s))

def String.take_end(+s: String, k: Nat) -> String:
  String.drop(s, Nat.sub(String.length(s), k))

def String.drop_end(+s: String, k: Nat) -> String:
  String.take(s, Nat.sub(String.length(s), k))

def String.get_end.if(s: String, i: Nat, ok: Bool) -> Maybe<&2, Char>:
  match ok:
    case False{}:
      None{}
    case True{}:
      String.get(s, i)

# s[-k]: k = 0 or k > length is None
def String.get_end(+s: String, +k: Nat) -> Maybe<&2, Char>:
  +n = String.length(s)
  String.get_end.if(s, Nat.sub(n, k), Bool.and(Nat.is_ge(k, 1n), Nat.is_le(k, n)))

def String.slice(s: String, +i: Nat, j: Nat) -> String:
  String.take(String.drop(s, i), Nat.sub(j, i))

def String.cut(+s: String, +n: Nat) -> String & String:
  (String.take(s, n), String.drop(s, n))

def String.clone(+s: String) -> String & String:
  (s, s)

def String.copy(s: String) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{h, t}:
      SCon{h, String.copy(t)}

law String.find.go:
  for +s: String
  for +p: String
  for i: Nat
  Maybe<&2, Nat>

def String.find.if(t: String, p: String, i: Nat, here: Bool) -> Maybe<&2, Nat>:
  match here:
    case False{}:
      String.find.go(t, p, 1n+i)
    case True{}:
      Some{i}

def String.find.go(s, p, i):
  match s:
    case SNil{}:
      Bool.pick(Maybe<&2, Nat>, String.is_empty(p), Some{i}, None{})
    case SCon{h, t}:
      String.find.if(t, p, i, String.starts_with(SCon{h, t}, p))

# first match start; the empty needle matches at 0
def String.find(s: String, p: String) -> Maybe<&2, Nat>:
  String.find.go(s, p, 0n)

law String.find_last.go:
  for +s: String
  for +p: String
  for i: Nat
  for best: Maybe<&2, Nat>
  Maybe<&2, Nat>

def String.find_last.if(
  t: String, p: String, +i: Nat, best: Maybe<&2, Nat>, here: Bool
) -> Maybe<&2, Nat>:
  match here:
    case False{}:
      String.find_last.go(t, p, 1n+i, best)
    case True{}:
      String.find_last.go(t, p, 1n+i, Some{i})

def String.find_last.go(s, p, i, best):
  match s:
    case SNil{}:
      Bool.pick(Maybe<&2, Nat>, String.is_empty(p), Some{i}, best)
    case SCon{h, t}:
      String.find_last.if(t, p, i, best, String.starts_with(SCon{h, t}, p))

# last match start (overlapping starts count); the empty needle matches at length
def String.find_last(s: String, p: String) -> Maybe<&2, Nat>:
  String.find_last.go(s, p, 0n, None{})

law String.count.go:
  for +s: String
  for +p: String
  for +n: Nat
  Nat

def String.count.if(t: String, +p: String, n: Nat, here: Bool) -> Nat:
  match here:
    case False{}:
      String.count.go(t, p, n)
    case True{}:
      String.count.go(String.drop(t, Nat.sub(String.length(p), 1n)), p, 1n+n)

def String.count.go(s, p, n):
  match s:
    case SNil{}:
      Bool.pick(Nat, String.is_empty(p), 1n+n, n)
    case SCon{h, t}:
      String.count.if(t, p, n, String.starts_with(SCon{h, t}, p))

# nonoverlapping matches; the empty needle counts length+1
def String.count(s: String, p: String) -> Nat:
  String.count.go(s, p, 0n)

law String.replace:
  for +s: String
  for +old: String
  for +new: String
  String

def String.replace.if(
  +h: Char, +t: String, +old: String, +new: String, here: Bool
) -> String:
  match here:
    case False{}:
      SCon{h, String.replace(t, old, new)}
    case True{}:
      +m = String.length(old)
      String.append(new, String.append(String.take(SCon{h, t}, Nat.sub(1n, m)),
        String.replace(String.drop(SCon{h, t}, Nat.max(m, 1n)), old, new)))

# nonoverlapping, left to right; empty old inserts new at every boundary
def String.replace(s, old, new):
  match s:
    case SNil{}:
      Bool.pick(String, String.is_empty(old), new, SNil{})
    case SCon{h, t}:
      String.replace.if(h, t, old, new, String.starts_with(SCon{h, t}, old))

law String.split_on.go:
  for +s: String
  for +sep: String
  List<&2, String>

def String.split_on.if(h: Char, t: String, +sep: String, here: Bool) ->
  List<&2, String>:
  match here:
    case False{}:
      String.split.push(h, String.split_on.go(t, sep))
    case True{}:
      SNil{} <> String.split_on.go(
        String.drop(t, Nat.sub(String.length(sep), 1n)), sep)

def String.split_on.go(s, sep):
  match s:
    case SNil{}:
      [SNil{}]
    case SCon{h, t}:
      String.split_on.if(h, t, sep, String.starts_with(SCon{h, t}, sep))

def String.split_on.start(s: String, sep: String, empty: Bool) ->
  List<&2, String>:
  match empty:
    case False{}:
      String.split_on.go(s, sep)
    case True{}:
      [s]

# substring separator, empties kept; unlike Python the empty separator is
# total: it never matches, so the result is [s]
def String.split_on(s: String, +sep: String) -> List<&2, String>:
  String.split_on.start(s, sep, String.is_empty(sep))

def String.words.cut(s: String) -> List<&2, String>:
  String.split.acc(s, ' ', True{}, "", [])

def String.words.keep(w: String) -> Bool:
  Bool.not(String.is_empty(w))

# runs between ASCII whitespace (U+0020, U+0009..U+000D); no empties
def String.words(s: String) -> List<&2, String>:
  List.filter(~String, ~String.words.keep, String.words.cut(s))

def String.partition.at(+s: String, +sep: String, i: Maybe<&2, Nat>) ->
  String & (String & String):
  match i:
    case None{}:
      (s, (SNil{}, SNil{}))
    case Some{+k}:
      (String.take(s, k), (sep, String.drop(s, Nat.add(k, String.length(sep)))))

def String.partition.find(s: String, sep: String, empty: Bool) ->
  Maybe<&2, Nat>:
  match empty:
    case False{}:
      String.find(s, sep)
    case True{}:
      None{}

# (before, sep, after) at the first match; absent, or the empty separator
# (which Python rejects), gives (s, "", "")
def String.partition(+s: String, +sep: String) -> String & (String & String):
  String.partition.at(s, sep, String.partition.find(s, sep, String.is_empty(sep)))

law String.splitlines:
  for s: String
  List<&2, String>

def String.splitlines.lf(t: String) -> String:
  match t:
    case SNil{}:
      SNil{}
    case SCon{+h, +t2}:
      Bool.pick(String, Char.is_eq(h, '\n'), t2, SCon{h, t2})

def String.splitlines.cr(t: String, cr: Bool) -> String:
  match cr:
    case False{}:
      t
    case True{}:
      String.splitlines.lf(t)

def String.splitlines.if(h: Char, t: String, brk: Bool) -> List<&2, String>:
  match brk:
    case False{}:
      String.split.push(h, String.splitlines(t))
    case True{}:
      SNil{} <> String.splitlines(String.splitlines.cr(t, Char.is_eq(h, '\r')))

# Python's boundaries (LF CR CRLF VT FF FS GS RS NEL LS PS); no trailing empty
def String.splitlines(s):
  match s:
    case SNil{}:
      Nil{}
    case SCon{+h, t}:
      String.splitlines.if(h, t, Char.is_line_break(h))

def String.capitalize(s: String) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{h, t}:
      SCon{Char.to_upper(h), String.to_lower(t)}

# width counts code points, not columns
def String.pad_start(+s: String, w: Nat, c: Char) -> String:
  String.append(
    String.from_list(List.replicate(Char, Nat.sub(w, String.length(s)), c)), s)

def String.pad_end(+s: String, w: Nat, c: Char) -> String:
  String.append(
    s, String.from_list(List.replicate(Char, Nat.sub(w, String.length(s)), c)))

def String.zfill.if(h: Char, t: String, w: Nat, sign: Bool) -> String:
  match sign:
    case False{}:
      String.pad_start(SCon{h, t}, w, '0')
    case True{}:
      SCon{h, String.pad_start(t, Nat.sub(w, 1n), '0')}

# zeros after a leading '+' or '-'
def String.zfill(s: String, w: Nat) -> String:
  match s:
    case SNil{}:
      String.pad_start(SNil{}, w, '0')
    case SCon{+h, t}:
      String.zfill.if(h, t, w, Bool.or(Char.is_eq(h, '+'), Char.is_eq(h, '-')))

# Python's str.is* family, ASCII only like case and trim: a non-ASCII code
# point is never alpha, digit, space, cased or printable. Nonempty and every
# char passes f
def String.nonempty_all(~f: Char -> Bool, s: String) -> Bool:
  match s:
    case SNil{}:
      False{}
    case SCon{h, t}:
      Bool.and(f(h), List.all(~&2, ~Char, ~f, String.to_list(t)))

# Char.is_alpha with a linear argument, the shape ~f takes
def Char.is_letter(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Char.is_alpha(Chr{x})

def String.is_alpha(s: String) -> Bool:
  String.nonempty_all(~Char.is_letter, s)

def String.is_digit(s: String) -> Bool:
  String.nonempty_all(~Char.is_digit, s)

# ASCII aliases of is_digit (Python's isdecimal and isnumeric)
def String.is_decimal(s: String) -> Bool:
  String.is_digit(s)

def String.is_numeric(s: String) -> Bool:
  String.is_digit(s)

def Char.is_alnum(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.or(Char.is_alpha(Chr{x}), Char.is_digit(Chr{x}))

def String.is_alnum(s: String) -> Bool:
  String.nonempty_all(~Char.is_alnum, s)

# U+0020 and U+0009..U+000D, as trim; Python adds U+001C..U+001F
def String.is_space(s: String) -> Bool:
  String.nonempty_all(~Char.is_space, s)

# a cased char exists and none has the other case
def String.is_lower(+s: String) -> Bool:
  Bool.and(List.any(~&2, ~Char, ~Char.is_lower, String.to_list(s)),
    String.eq(String.to_lower(s), s))

def String.is_upper(+s: String) -> Bool:
  Bool.and(String.is_lower(String.to_lower(s)), String.eq(String.to_upper(s), s))

def Char.is_ascii(c: Char) -> Bool:
  U32.is_lt(Char.to_u32(c), 128)

def Char.is_printable(c: Char) -> Bool:
  U32.is_lt(U32.sub(Char.to_u32(c), 32), 95)

# the empty string is ASCII and printable, as in Python
def String.is_ascii(s: String) -> Bool:
  List.all(~&2, ~Char, ~Char.is_ascii, String.to_list(s))

# U+0020..U+007E
def String.is_printable(s: String) -> Bool:
  List.all(~&2, ~Char, ~Char.is_printable, String.to_list(s))

def Char.is_word(c: Char) -> Bool:
  match c:
    case Chr{+x}:
      Bool.or(Char.is_alnum(Chr{x}), U32.is_eq(x, 95))

# [A-Za-z_][A-Za-z0-9_]*; keywords are identifiers, as in Python
def String.is_identifier(+s: String) -> Bool:
  Bool.and(String.nonempty_all(~Char.is_word, s),
    Bool.not(String.is_digit(String.take(s, 1n))))

def String.is_title.go(s: String, +prev: Bool) -> Bool:
  match s:
    case SNil{}:
      True{}
    case SCon{+h, t}:
      Bool.and(Bool.or(Bool.not(Char.is_alpha(h)), Bool.xor(Char.is_upper(h), prev)),
        String.is_title.go(t, Char.is_alpha(h)))

# a letter exists; uppercase only after a non-letter, lowercase only after one
def String.is_title(+s: String) -> Bool:
  Bool.and(List.any(~&2, ~Char, ~Char.is_letter, String.to_list(s)),
    String.is_title.go(s, False{}))

def String.remove_prefix(+s: String, +p: String) -> String:
  Bool.pick(String, String.starts_with(s, p), String.drop(s, String.length(p)), s)

def String.remove_suffix(+s: String, +p: String) -> String:
  Bool.pick(String, String.ends_with(s, p), String.drop_end(s, String.length(p)), s)

# CPython's rule: the odd pad cell goes left only when the width is odd too
def String.center(+s: String, +w: Nat, +c: Char) -> String:
  +n = String.length(s)
  +m = Nat.sub(w, n)
  String.pad_end(String.pad_start(s, Nat.add(n, Nat.add(Nat.div(m, 2n),
    Nat.mod(Nat.mul(m, w), 2n))), c), w, c)

def String.swapcase(s: String) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{+h, t}:
      SCon{Bool.pick(Char, Char.is_upper(h), Char.to_lower(h), Char.to_upper(h)),
        String.swapcase(t)}

def String.title.go(s: String, prev: Bool) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{+h, t}:
      SCon{Bool.pick(Char, prev, Char.to_lower(h), Char.to_upper(h)),
        String.title.go(t, Char.is_alpha(h))}

# uppercase after any non-letter ("they're" -> "They'Re"), lowercase elsewhere
def String.title(s: String) -> String:
  String.title.go(s, False{})

# on ASCII, casefold is to_lower
def String.casefold(s: String) -> String:
  String.to_lower(s)

def String.expandtabs.go(s: String, +n: Nat, +col: Nat) -> String:
  match s:
    case SNil{}:
      SNil{}
    case SCon{+h, t}:
      +tab = Char.is_eq(h, '\t')
      +k = Bool.pick(Nat, tab, Nat.sub(n, Nat.mod(col, n)), 1n)
      String.append(
        Bool.pick(String, tab, String.pad_start(SNil{}, k, ' '), SCon{h, SNil{}}),
        String.expandtabs.go(t, n, Bool.pick(Nat,
          Bool.or(Char.is_eq(h, '\n'), Char.is_eq(h, '\r')), 0n, Nat.add(col, k))))

# a tab pads to the next multiple of n (none when n is 0); '\n' and '\r'
# restart the column, as in Python; a column is a code point
def String.expandtabs(s: String, n: Nat) -> String:
  String.expandtabs.go(s, n, 0n)

# split_on scanning from the right: "aaa" on "aa" is ["a", ""]
def String.rsplit.push(ws: List<&2, String>, w: String) -> List<&2, String>:
  String.reverse(w) <> ws

def String.rsplit(s: String, sep: String) -> List<&2, String>:
  List.foldl(~&2, ~String, ~List<&2, String>, ~String.rsplit.push,
    String.split_on(String.reverse(s), String.reverse(sep)), [])

def String.rpartition.at(+s: String, +sep: String, i: Maybe<&2, Nat>) ->
  String & (String & String):
  match i:
    case None{}:
      (SNil{}, (SNil{}, s))
    case Some{k}:
      String.partition.at(s, sep, Some{k})

# (before, sep, after) at the last match; absent, or the empty separator,
# gives ("", "", s)
def String.rpartition(+s: String, +sep: String) -> String & (String & String):
  String.rpartition.at(s, sep, Bool.pick(Maybe<&2, Nat>, String.is_empty(sep),
    None{}, String.find_last(s, sep)))

def String.hash.byte(h: U32, b: U32) -> U32:
  U32.mul(U32.xor(h, b), 16777619)

def String.hash.cell(h: U32, +x: U32) -> U32:
  String.hash.byte(String.hash.byte(String.hash.byte(String.hash.byte(h,
    U32.and(x, 255)), U32.and(U32.shrn(x, 8n), 255)),
    U32.and(U32.shrn(x, 16n), 255)), U32.shrn(x, 24n))

def String.hash.go(s: String, h: U32) -> U32:
  match s:
    case SNil{}:
      h
    case SCon{Chr{x}, t}:
      String.hash.go(t, String.hash.cell(h, x))

# FNV-1a (32-bit) over four little-endian bytes per code point; a stable
# Bend spec, not Python's hash
def String.hash(s: String) -> U32:
  String.hash.go(s, 2166136261)

# Bytes
# -----

# A byte buffer is a String whose every Char is below 256. The runtime
# packs such a String a byte a cell in one block, cuts it by view and
# appends to it in place, so it is what List<&2, U32> is on the wire at a
# byte a byte instead of a cell a byte. An ASCII literal is its own bytes;
# TCP.send_buf refuses a Char past 255 (EINVAL) rather than encode it.
def Bytes.len(b: Bytes()) -> Nat:
  String.length(b)

def Bytes.byte(m: Maybe<&2, Char>) -> U32:
  match m:
    case None{}:
      256
    case Some{Chr{c}}:
      c

# the byte at i; past the end, 256, which no byte is
def Bytes.get(b: Bytes(), i: Nat) -> U32:
  Bytes.byte(String.get(b, i))

# n bytes from at, a view: nothing is copied
def Bytes.slice(b: Bytes(), at: Nat, n: Nat) -> Bytes():
  String.take(String.drop(b, at), n)

def Bytes.drop(b: Bytes(), n: Nat) -> Bytes():
  String.drop(b, n)

# in place when a is the buffer's one owner and has room
def Bytes.append(a: Bytes(), b: Bytes()) -> Bytes():
  String.append(a, b)

def Bytes.concat(xs: List<&2, Bytes()>) -> Bytes():
  String.concat(xs)

def Bytes.starts_with(b: Bytes(), p: Bytes()) -> Bool:
  String.starts_with(b, p)

# the first offset of p in b, as "\r\n\r\n" ends a head
def Bytes.find(b: Bytes(), p: Bytes()) -> Maybe<&2, Nat>:
  String.find(b, p)

def Bytes.from_list(xs: List<&2, U32>) -> Bytes():
  match xs:
    case Nil{}:
      SNil{}
    case h <> t:
      SCon{Chr{h}, Bytes.from_list(t)}

def Bytes.to_list(b: Bytes()) -> List<&2, U32>:
  match b:
    case SNil{}:
      Nil{}
    case SCon{Chr{c}, t}:
      c <> Bytes.to_list(t)

law Bytes.list:
  for xs: List<&2, U32>
  {Bytes.to_list(Bytes.from_list(xs)) == xs : List<&2, U32>}

def Bytes.list(xs):
  match xs:
    case Nil{}:
      {==}
    case h <> t:
      Equal.cong(List<&2, U32>, List<&2, U32>, ys => h <> ys,
        Bytes.to_list(Bytes.from_list(t)), t, Bytes.list(t))

# the bytes of two buffers joined are the bytes of each, joined
law Bytes.to_list_append:
  for +a: Bytes()
  for +b: Bytes()
  {Bytes.to_list(Bytes.append(a, b))
    == List.append(&2, U32, Bytes.to_list(a), Bytes.to_list(b)) : List<&2, U32>}

def Bytes.to_list_append(a, b):
  match a:
    case SNil{}:
      {==}
    case SCon{Chr{c}, t}:
      Equal.cong(List<&2, U32>, List<&2, U32>, ys => c <> ys,
        Bytes.to_list(Bytes.append(t, b)),
        List.append(&2, U32, Bytes.to_list(t), Bytes.to_list(b)),
        Bytes.to_list_append(t, b))

law Bytes.len.shift:
  for +s: String
  for +m: Nat
  {String.length.go(s, 1n+m) == 1n+String.length.go(s, m) : Nat}

def Bytes.len.shift(s, m):
  match s:
    case SNil{}:
      {==}
    case SCon{h, t}:
      Bytes.len.shift(t, 1n+m)

law Bytes.len.from:
  for +s: String
  for +m: Nat
  {String.length.go(s, m) == Nat.add(m, String.length.go(s, 0n)) : Nat}

def Bytes.len.from(s, m):
  match m:
    case 0n:
      {==}
    case 1n+p:
      Equal.trans(Nat, String.length.go(s, 1n+p), 1n+String.length.go(s, p),
        1n+Nat.add(p, String.length.go(s, 0n)), Bytes.len.shift(s, p),
        Equal.cong(Nat, Nat, y => 1n+y, String.length.go(s, p),
          Nat.add(p, String.length.go(s, 0n)), Bytes.len.from(s, p)))

law Bytes.len.go:
  for +a: String
  for +b: String
  for +m: Nat
  {String.length.go(String.append(a, b), m)
    == Nat.add(String.length.go(a, m), String.length.go(b, 0n)) : Nat}

def Bytes.len.go(a, b, m):
  match a:
    case SNil{}:
      Bytes.len.from(b, m)
    case SCon{h, t}:
      Bytes.len.go(t, b, 1n+m)

law Bytes.len_append:
  for +a: Bytes()
  for +b: Bytes()
  {Bytes.len(Bytes.append(a, b)) == Nat.add(Bytes.len(a), Bytes.len(b)) : Nat}

def Bytes.len_append(a, b):
  Bytes.len.go(a, b, 0n)

law Bytes.get.take:
  for +s: String
  for +k: Nat
  for +m: Nat
  {String.get(String.take(s, 1n+Nat.add(k, m)), k) == String.get(s, k)
    : Maybe<&2, Char>}

def Bytes.get.take(s, k, m):
  match s k:
    case SNil{} _:
      {==}
    case SCon{h, t} 0n:
      {==}
    case SCon{h, t} 1n+q:
      Bytes.get.take(t, q, m)

law Bytes.get.drop:
  for +s: String
  for +i: Nat
  for +k: Nat
  {String.get(String.drop(s, i), k) == String.get(s, Nat.add(i, k))
    : Maybe<&2, Char>}

def Bytes.get.drop(s, i, k):
  match s i:
    case SNil{} _:
      {==}
    case SCon{h, t} 0n:
      {==}
    case SCon{h, t} 1n+p:
      Bytes.get.drop(t, p, k)

# a byte of a slice is the byte at its offset, for every k it holds
law Bytes.get_slice:
  for +b: Bytes()
  for +i: Nat
  for +k: Nat
  for +m: Nat
  {Bytes.get(Bytes.slice(b, i, 1n+Nat.add(k, m)), k)
    == Bytes.get(b, Nat.add(i, k)) : U32}

def Bytes.get_slice(b, i, k, m):
  Equal.cong(Maybe<&2, Char>, U32, Bytes.byte,
    String.get(Bytes.slice(b, i, 1n+Nat.add(k, m)), k),
    String.get(b, Nat.add(i, k)),
    Equal.trans(Maybe<&2, Char>,
      String.get(String.take(String.drop(b, i), 1n+Nat.add(k, m)), k),
      String.get(String.drop(b, i), k), String.get(b, Nat.add(i, k)),
      Bytes.get.take(String.drop(b, i), k, m), Bytes.get.drop(b, i, k)))

# The scanner's reads. Natively get, span and the finds borrow the buffer:
# the call reads the packed cells where they lie, and neither shares nor
# drops the buffer, builds a view or a Maybe; push writes into an unshared
# buffer's spare room. The bodies below are what the laws are about.

# b with the byte x on its end
def Bytes.push(b: Bytes(), x: U32) -> Bytes():
  String.append(b, SCon{Chr{x}, SNil{}})

law Bytes.push_list:
  for +b: Bytes()
  for +x: U32
  {Bytes.to_list(Bytes.push(b, x)) == List.append(&2, U32, Bytes.to_list(b), x <> Nil{})
    : List<&2, U32>}

def Bytes.push_list(b, x):
  Bytes.to_list_append(b, SCon{Chr{x}, SNil{}})

# lo <= c < hi, the two tests a byte class is written with
def Bytes.span.in(+c: U32, lo: U32, hi: U32) -> Bool:
  Bool.and(Bool.not(U32.is_lt(c, lo)), Bool.not(U32.is_le(hi, c)))

law Bytes.span.go:
  for s: String
  for +lo: U32
  for +hi: U32
  Nat

def Bytes.span.at(ok: Bool, t: String, +lo: U32, +hi: U32) -> Nat:
  match ok:
    case False{}:
      0n
    case True{}:
      1n+Bytes.span.go(t, lo, hi)

def Bytes.span.go(s, lo, hi):
  match s:
    case SNil{}:
      0n
    case SCon{Chr{c}, t}:
      Bytes.span.at(Bytes.span.in(c, lo, hi), t, lo, hi)

# how many bytes from i are in [lo, hi): a run of digits, of a token, of
# anything but a delimiter. It reads one byte past the run and no further.
def Bytes.span(b: Bytes(), i: Nat, lo: U32, hi: U32) -> Nat:
  Bytes.span.go(String.drop(b, i), lo, hi)

law Bytes.span.go_le:
  for +s: String
  for +lo: U32
  for +hi: U32
  {Nat.is_le(Bytes.span.go(s, lo, hi), String.length.go(s, 0n)) == True{} : Bool}

law Bytes.span.at_le.go:
  for ok: Bool
  for +t: String
  for +lo: U32
  for +hi: U32
  {Nat.is_le(Bytes.span.at(ok, t, lo, hi), 1n+String.length.go(t, 0n)) == True{} : Bool}

def Bytes.span.at_le.go(ok, t, lo, hi):
  match ok:
    case False{}:
      {==}
    case True{}:
      Bytes.span.go_le(t, lo, hi)

law Bytes.span.at_le:
  for ok: Bool
  for +t: String
  for +lo: U32
  for +hi: U32
  {Nat.is_le(Bytes.span.at(ok, t, lo, hi), String.length.go(t, 1n)) == True{} : Bool}

def Bytes.span.at_le(ok, t, lo, hi):
  Equal.trans(Bool, Nat.is_le(Bytes.span.at(ok, t, lo, hi), String.length.go(t, 1n)),
    Nat.is_le(Bytes.span.at(ok, t, lo, hi), 1n+String.length.go(t, 0n)), True{},
    Equal.cong(Nat, Bool, y => Nat.is_le(Bytes.span.at(ok, t, lo, hi), y),
      String.length.go(t, 1n), 1n+String.length.go(t, 0n), Bytes.len.shift(t, 0n)),
    Bytes.span.at_le.go(ok, t, lo, hi))

def Bytes.span.go_le(s, lo, hi):
  match s:
    case SNil{}:
      {==}
    case SCon{Chr{c}, t}:
      Bytes.span.at_le(Bytes.span.in(c, lo, hi), t, lo, hi)

# the run never passes the end: span <= len - i
law Bytes.span_le:
  for +b: Bytes()
  for +i: Nat
  for +lo: U32
  for +hi: U32
  {Nat.is_le(Bytes.span(b, i, lo, hi), Bytes.len(Bytes.drop(b, i))) == True{} : Bool}

def Bytes.span_le(b, i, lo, hi):
  Bytes.span.go_le(String.drop(b, i), lo, hi)

# A byte in a span's range: lo <= c and c < hi; in a range, in any range
# around it
law Bytes.span.in_lo:
  for +c: U32
  for +lo: U32
  for +hi: U32
  for +e: {Bytes.span.in(c, lo, hi) == True{} : Bool}
  {U32.is_le(lo, c) == True{} : Bool}

def Bytes.span.in_lo(c, lo, hi, e):
  %U32.lt_flip(c, lo) : {_ == True{} : Bool}
  Bool.and_l(Bool.not(U32.is_lt(c, lo)), Bool.not(U32.is_le(hi, c)), e)

law Bytes.span.in_hi:
  for +c: U32
  for +lo: U32
  for +hi: U32
  for +e: {Bytes.span.in(c, lo, hi) == True{} : Bool}
  {U32.is_lt(c, hi) == True{} : Bool}

def Bytes.span.in_hi(c, lo, hi, e):
  %U32.le_flip(hi, c) : {_ == True{} : Bool}
  Bool.and_r(Bool.not(U32.is_lt(c, lo)), Bool.not(U32.is_le(hi, c)), e)

law Bytes.span.in_of:
  for +c: U32
  for +lo: U32
  for +hi: U32
  for l: {U32.is_le(lo, c) == True{} : Bool}
  for h: {U32.is_lt(c, hi) == True{} : Bool}
  {Bytes.span.in(c, lo, hi) == True{} : Bool}

def Bytes.span.in_of(c, lo, hi, l, h):
  %Equal.sym(Bool, Bool.not(U32.is_lt(c, lo)), U32.is_le(lo, c), U32.lt_flip(c, lo)) :
    {Bool.and(_, Bool.not(U32.is_le(hi, c))) == True{} : Bool}
  %Equal.sym(Bool, Bool.not(U32.is_le(hi, c)), U32.is_lt(c, hi), U32.le_flip(hi, c)) :
    {Bool.and(U32.is_le(lo, c), _) == True{} : Bool}
  %Equal.sym(Bool, U32.is_le(lo, c), True{}, l) : {Bool.and(_, U32.is_lt(c, hi)) == True{} : Bool}
  h

law Bytes.span.in_widen:
  for +c: U32
  for +lo: U32
  for +hi: U32
  for +lo2: U32
  for +hi2: U32
  for +e: {Bytes.span.in(c, lo, hi) == True{} : Bool}
  for l: {U32.is_le(lo2, lo) == True{} : Bool}
  for h: {U32.is_le(hi, hi2) == True{} : Bool}
  {Bytes.span.in(c, lo2, hi2) == True{} : Bool}

def Bytes.span.in_widen(c, lo, hi, lo2, hi2, e, l, h):
  Bytes.span.in_of(c, lo2, hi2, U32.le_trans(lo2, lo, c, l, Bytes.span.in_lo(c, lo, hi, e)),
    U32.lt_le_trans(c, hi, hi2, Bytes.span.in_hi(c, lo, hi, e), h))

# every byte a span counts is in its range
def Bytes.span.go_get.at(+c: U32, +t: String, +lo: U32, +hi: U32, +k: Nat, ok: Bool,
  ih: @+j: Nat -> {Nat.is_lt(j, Bytes.span.go(t, lo, hi)) == True{} : Bool} ->
    {Bytes.span.in(Bytes.get(t, j), lo, hi) == True{} : Bool}) ->
  {Bytes.span.in(c, lo, hi) == ok : Bool} ->
  {Nat.is_lt(k, Bytes.span.at(ok, t, lo, hi)) == True{} : Bool} ->
  {Bytes.span.in(Bytes.get(SCon{Chr{c}, t}, k), lo, hi) == True{} : Bool}:
  match k ok:
    case _ False{}:
      e => lt => Nat.lt_zero({Bytes.span.in(Bytes.get(SCon{Chr{c}, t}, k), lo, hi) == True{} : Bool}, k, lt)
    case 0n True{}:
      e => lt => e
    case 1n+j True{}:
      e => lt => ih(j, lt)

law Bytes.span.go_get:
  for +s: String
  for +lo: U32
  for +hi: U32
  for +k: Nat
  for lt: {Nat.is_lt(k, Bytes.span.go(s, lo, hi)) == True{} : Bool}
  {Bytes.span.in(Bytes.get(s, k), lo, hi) == True{} : Bool}

def Bytes.span.go_get(s, lo, hi, k, lt):
  match s:
    case SNil{}:
      Nat.lt_zero({Bytes.span.in(Bytes.get(SNil{}, k), lo, hi) == True{} : Bool}, k, lt)
    case SCon{Chr{+c}, t}:
      Bytes.span.go_get.at(c, t, lo, hi, k, Bytes.span.in(c, lo, hi),
        j => l => Bytes.span.go_get(t, lo, hi, j, l))({==}, lt)

# the bytes from i that Bytes.span counts are each in [lo, hi)
law Bytes.span_get:
  for +b: Bytes()
  for +i: Nat
  for +lo: U32
  for +hi: U32
  for +k: Nat
  for lt: {Nat.is_lt(k, Bytes.span(b, i, lo, hi)) == True{} : Bool}
  {Bytes.span.in(Bytes.get(b, Nat.add(i, k)), lo, hi) == True{} : Bool}

def Bytes.span_get(b, i, lo, hi, k, lt):
  %Equal.cong(Maybe<&2, Char>, U32, Bytes.byte, String.get(String.drop(b, i), k),
    String.get(b, Nat.add(i, k)), Bytes.get.drop(b, i, k)) :
    {Bytes.span.in(_, lo, hi) == True{} : Bool}
  Bytes.span.go_get(String.drop(b, i), lo, hi, k, lt)

law Bytes.find_byte.go:
  for s: String
  for +x: U32
  Nat

def Bytes.find_byte.at(hit: Bool, t: String, +x: U32) -> Nat:
  match hit:
    case True{}:
      0n
    case False{}:
      1n+Bytes.find_byte.go(t, x)

def Bytes.find_byte.go(s, x):
  match s:
    case SNil{}:
      0n
    case SCon{Chr{c}, t}:
      Bytes.find_byte.at(U32.is_eq(c, x), t, x)

# the first offset of the byte x at or after i; absent, the end
# (Bytes.len(b), from any i up to it)
def Bytes.find_byte(b: Bytes(), +i: Nat, x: U32) -> Nat:
  Nat.add(i, Bytes.find_byte.go(String.drop(b, i), x))

# w, x, y or z: the test a byte class of four names is written with
def Bytes.find_any.in(+c: U32, w: U32, x: U32, y: U32, z: U32) -> Bool:
  Bool.or(U32.is_eq(c, w), Bool.or(U32.is_eq(c, x), Bool.or(U32.is_eq(c, y), U32.is_eq(c, z))))

law Bytes.find_any.go:
  for s: String
  for +w: U32
  for +x: U32
  for +y: U32
  for +z: U32
  Nat

def Bytes.find_any.at(hit: Bool, t: String, +w: U32, +x: U32, +y: U32, +z: U32) -> Nat:
  match hit:
    case True{}:
      0n
    case False{}:
      1n+Bytes.find_any.go(t, w, x, y, z)

def Bytes.find_any.go(s, w, x, y, z):
  match s:
    case SNil{}:
      0n
    case SCon{Chr{c}, t}:
      Bytes.find_any.at(Bytes.find_any.in(c, w, x, y, z), t, w, x, y, z)

# the first offset at or after i of any of the four bytes w, x, y, z (name
# one twice for fewer), as a delimiter, a quote or a line end ends a
# field; absent, the end
def Bytes.find_any(b: Bytes(), +i: Nat, w: U32, x: U32, y: U32, z: U32) -> Nat:
  Nat.add(i, Bytes.find_any.go(String.drop(b, i), w, x, y, z))

# the low eight bits of the cell at i, 0 past the end
def Bytes.word_le.at(+b: Bytes(), i: Nat) -> U32:
  U32.and(Bytes.get(b, i), 255)

# the four bytes from i as one U32, little end first, a byte past the end 0:
# the word a hash or a checksum reads, in one borrowed read natively
def Bytes.word_le(+b: Bytes(), +i: Nat) -> U32:
  U32.or(U32.or(Bytes.word_le.at(b, i), U32.shln(Bytes.word_le.at(b, 1n+i), 8n)),
    U32.or(U32.shln(Bytes.word_le.at(b, 2n+i), 16n), U32.shln(Bytes.word_le.at(b, 3n+i), 24n)))

# Regex
# -----

# CPython 3.11 `re.ASCII` is the pin: leftmost-first alternation, greedy and
# lazy repeats, a capture keeps its last value, `$` also matches before a
# final newline, `.` skips only newline. Offsets are code-point indices.
# An unbounded repeat of a nullable body is rejected at parse (NullableRepeat).
# Errors carry `pos`: the code-point offset where the parser gave up.

def Regex.max_rep() -> Nat:
  256n

def Regex.max_prog() -> Nat:
  4096n

def Regex.Res(T: Data) -> Data:
  Result<&2, &2, Regex.Error, T>

def Regex.digits() -> Regex.Ranges():
  [(48, 57)]

def Regex.words() -> Regex.Ranges():
  [(48, 57), (65, 90), (95, 95), (97, 122)]

def Regex.spaces() -> Regex.Ranges():
  [(9, 13), (32, 32)]

# Complement of an ascending, disjoint range list over the code points
def Regex.ranges.not(rs: Regex.Ranges(), lo: U32) -> Regex.Ranges():
  match rs:
    case Nil{}:
      [(lo, 1114111)]
    case (a, b) <> rest:
      (lo, U32.sub(a, 1)) <> Regex.ranges.not(rest, U32.add(b, 1))

def Regex.nullable(r: Re) -> Bool:
  match r:
    case RLit{c}:
      False{}
    case RAny{}:
      False{}
    case RSet{neg, rs}:
      False{}
    case RCat{a, b}:
      Bool.and(Regex.nullable(a), Regex.nullable(b))
    case RAlt{a, b}:
      Bool.or(Regex.nullable(a), Regex.nullable(b))
    case RRep{r, lo, hi, g}:
      Bool.or(Nat.is_eq(lo, 0n), Regex.nullable(r))
    case RCap{n, r}:
      Regex.nullable(r)
    case _:
      True{}

# Anchors take no quantifier (a group around one does, as in CPython)
def Regex.parse.q(r: Re) -> Nat:
  match r:
    case RBol{}:
      0n
    case REol{}:
      0n
    case RWordB{neg}:
      0n
    case _:
      1n

# The parser is a stack machine over the pattern: every step reads one
# char, so the fuel is the pattern length. State = unread tail, groups
# opened so far, open frames (the root frame at the bottom). An error
# carries the unread length; `Regex.parse.run` turns it into an offset
def Regex.parse.Stack() -> Data:
  List<&2, Regex.Frame>

def Regex.parse.State() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, String, _ => Sigma<&2, &2, Nat, _ => Regex.parse.Stack()>>>

def Regex.parse.Item() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, Re, _ => String>>

def Regex.parse.Tree() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, Re, _ => Nat>>

def Regex.parse.Scan() -> Data:
  Regex.Res(Regex.Cls)

def Regex.parse.Num() -> Type:
  Nat & Nat & String

def Regex.parse.Code() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, U32, _ => String>>

# Index of x in ks, from i; 0 when absent
def Regex.parse.index(+x: U32, ks: List<&2, U32>, +i: Nat) -> Nat:
  match ks:
    case Nil{}:
      0n
    case k <> rest:
      Bool.pick(Nat, U32.is_eq(x, k), i, Regex.parse.index(x, rest, 1n+i))

# Newest-first items fold into source order: [c, b, a] -> RCat{a, RCat{b, c}}
def Regex.parse.rcat(acc: Re, x: Re) -> Re:
  RCat{x, acc}

def Regex.parse.ralt(acc: Re, x: Re) -> Re:
  RAlt{x, acc}

def Regex.parse.cat(rs: List<&2, Re>) -> Re:
  match rs:
    case Nil{}:
      REmpty{}
    case x <> rest:
      List.foldl(~&2, ~Re, ~Re, ~Regex.parse.rcat, rest, x)

def Regex.parse.wrap(cap: Maybe<&2, Nat>, r: Re) -> Re:
  match cap:
    case None{}:
      r
    case Some{k}:
      RCap{k, r}

def Regex.parse.fin(fr: Regex.Frame) -> Re:
  match fr:
    case Frame{cap, alts, cat, q}:
      Regex.parse.wrap(cap, List.foldl(~&2, ~Re, ~Re, ~Regex.parse.ralt, alts, Regex.parse.cat(cat)))

def Regex.parse.push.q(t: String, n: Nat, stack: Regex.parse.Stack(), r: Re, q2: Nat) -> Regex.parse.State():
  match stack:
    case Frame{cap, alts, cat, q} <> rest:
      Done{(t, (n, Frame{cap, alts, r <> cat, q2} <> rest))}
    case Nil{}:
      Fail{RErr{String.length(t), "unbalanced parenthesis"}}

def Regex.parse.push(t: String, n: Nat, stack: Regex.parse.Stack(), +r: Re) -> Regex.parse.State():
  Regex.parse.push.q(t, n, stack, r, Regex.parse.q(r))

def Regex.parse.atom(n: Nat, stack: Regex.parse.Stack(), r: Regex.parse.Item()) -> Regex.parse.State():
  match r:
    case Fail{e}:
      Fail{e}
    case Done{(re, t)}:
      Regex.parse.push(t, n, stack, re)

def Regex.parse.open.cap(t: String, +n: Nat, stack: Regex.parse.Stack()) -> Regex.parse.State():
  Done{(t, (1n+n, Frame{Some{1n+n}, [], [], 0n} <> stack))}

# `(?:` opens a plain group; any other `(?` extension is out of scope
def Regex.parse.open.ext(+k: Nat, t: String, n: Nat, stack: Regex.parse.Stack()) -> Regex.parse.State():
  match t:
    case SCon{Chr{+x}, t2}:
      Bool.pick(Regex.parse.State(), U32.is_eq(x, 58),
        Done{(t2, (n, Frame{None{}, [], [], 0n} <> stack))},
        Fail{RErr{k, "unsupported group extension"}})
    case SNil{}:
      Fail{RErr{k, "unsupported group extension"}}

def Regex.parse.open(t: String, +n: Nat, +stack: Regex.parse.Stack()) -> Regex.parse.State():
  match t:
    case SCon{Chr{+x}, +t2}:
      Bool.pick(Regex.parse.State(), U32.is_eq(x, 63),
        Regex.parse.open.ext(Nat.add(2n, String.length(t2)), t2, n, stack),
        Regex.parse.open.cap(SCon{Chr{x}, t2}, n, stack))
    case SNil{}:
      Regex.parse.open.cap(SNil{}, n, stack)

def Regex.parse.rep.if(k: Nat, st: Regex.parse.State(), bad: Bool) -> Regex.parse.State():
  match bad:
    case True{}:
      Fail{RErr{k, "NullableRepeat"}}
    case False{}:
      st

# Repeat the last atom; an unbounded repeat needs a body that eats
def Regex.parse.rep(+t: String, n: Nat, stack: Regex.parse.Stack(), lo: Nat, +hi: Maybe<&2, Nat>) -> Regex.parse.State():
  match stack:
    case Frame{cap, alts, +r <> cat, 1n} <> rest:
      Regex.parse.rep.if(1n+String.length(t),
        Done{(t, (n, Frame{cap, alts, RRep{r, lo, hi, True{}} <> cat, 2n} <> rest))},
        Bool.and(Bool.not(Maybe.is_some(&2, Nat, hi)), Regex.nullable(r)))
    case Frame{cap, alts, cat, 2n} <> rest:
      Fail{RErr{1n+String.length(t), "multiple repeat"}}
    case _:
      Fail{RErr{1n+String.length(t), "nothing to repeat"}}

def Regex.parse.Bounds() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, Sigma<&2, &2, Nat, _ => Maybe<&2, Nat>>, _ => String>>

# Decimal digits, capped at 1000 so a silly count costs nothing
def Regex.parse.num(t: String, +acc: Nat, +cnt: Nat) -> Regex.parse.Num():
  match t:
    case SNil{}:
      (acc, (cnt, SNil{}))
    case SCon{Chr{+x}, +t2}:
      Bool.pick(Regex.parse.Num(), Char.is_digit(Chr{x}),
        Regex.parse.num(t2, Nat.min(1000n, Nat.add(Nat.mul(acc, 10n), U32.to_nat(U32.sub(x, 48)))), 1n+cnt),
        (acc, (cnt, SCon{Chr{x}, t2})))

def Regex.parse.bounds.ok(+k: Nat, +lo: Nat, +hi: Maybe<&2, Nat>, t: String) -> Regex.parse.Bounds():
  Bool.pick(Regex.parse.Bounds(), Nat.is_gt(Nat.max(lo, Maybe.default(&2, Nat, hi, 0n)), Regex.max_rep()),
    Fail{RErr{k, "repeat count exceeds max_rep"}},
    Bool.pick(Regex.parse.Bounds(), Nat.is_gt(lo, Maybe.default(&2, Nat, hi, lo)),
      Fail{RErr{k, "min repeat greater than max repeat"}},
      Done{((lo, hi), t)}))

def Regex.parse.bounds.hi(+k: Nat, lo: Nat, r: Regex.parse.Num()) -> Regex.parse.Bounds():
  match r:
    case (hi, (cnt, SCon{Chr{+y}, t2})):
      Bool.pick(Regex.parse.Bounds(), U32.is_eq(y, 125),
        Regex.parse.bounds.ok(k, lo, Bool.pick(Maybe<&2, Nat>, Nat.is_eq(cnt, 0n), None{}, Some{hi}), t2),
        Fail{RErr{k, "bad repeat"}})
    case (hi, (cnt, SNil{})):
      Fail{RErr{k, "bad repeat"}}

def Regex.parse.bounds.lo(+k: Nat, r: Regex.parse.Num()) -> Regex.parse.Bounds():
  match r:
    case (lo, (0n, rest)):
      Fail{RErr{k, "bad repeat"}}
    case (+lo, (cnt, SCon{Chr{+y}, +t2})):
      Bool.pick(Regex.parse.Bounds(), U32.is_eq(y, 125),
        Regex.parse.bounds.ok(k, lo, Some{lo}, t2),
        Bool.pick(Regex.parse.Bounds(), U32.is_eq(y, 44),
          Regex.parse.bounds.hi(k, lo, Regex.parse.num(t2, 0n, 0n)),
          Fail{RErr{k, "bad repeat"}}))
    case (lo, (cnt, SNil{})):
      Fail{RErr{k, "bad repeat"}}

# `{m}` `{m,}` `{m,n}`; anything else (so `{,n}`) is a bad repeat
def Regex.parse.bounds(+t: String) -> Regex.parse.Bounds():
  Regex.parse.bounds.lo(1n+String.length(t), Regex.parse.num(t, 0n, 0n))

def Regex.parse.braces(n: Nat, stack: Regex.parse.Stack(), b: Regex.parse.Bounds()) -> Regex.parse.State():
  match b:
    case Fail{e}:
      Fail{e}
    case Done{((lo, hi), t)}:
      Regex.parse.rep(t, n, stack, lo, hi)

def Regex.parse.hexval(+x: U32) -> U32:
  Bool.pick(U32, Char.is_digit(Chr{x}), U32.sub(x, 48),
    Bool.pick(U32, Bool.and(U32.is_ge(x, 97), U32.is_le(x, 102)), U32.sub(x, 87),
      Bool.pick(U32, Bool.and(U32.is_ge(x, 65), U32.is_le(x, 70)), U32.sub(x, 55), 16)))

def Regex.parse.hex.if(k: Nat, v: U32, r: Regex.parse.Code()) -> Regex.parse.Code():
  Bool.pick(Regex.parse.Code(), U32.is_lt(v, 16), r, Fail{RErr{k, "bad escape"}})

# Exactly d hex digits
def Regex.parse.hex(+k: Nat, d: Nat, t: String, acc: U32) -> Regex.parse.Code():
  match d:
    case 0n:
      Done{(acc, t)}
    case 1n+e:
      match t:
        case SNil{}:
          Fail{RErr{k, "bad escape"}}
        case SCon{Chr{+x}, t2}:
          Regex.parse.hex.if(k, Regex.parse.hexval(x),
            Regex.parse.hex(k, e, t2, U32.add(U32.mul(acc, 16), Regex.parse.hexval(x))))

# A code point escape must be a scalar value
def Regex.parse.esc.code(k: Nat, r: Regex.parse.Code()) -> Regex.parse.Item():
  match r:
    case Fail{e}:
      Fail{e}
    case Done{(+c, t)}:
      Bool.pick(Regex.parse.Item(),
        Bool.or(U32.is_gt(c, 1114111), Bool.and(U32.is_ge(c, 55296), U32.is_le(c, 57343))),
        Fail{RErr{k, "bad escape"}}, Done{(RLit{Chr{c}}, t)})

def Regex.parse.esc.table() -> List<&2, Sigma<&2, &2, U32, _ => Re>>:
  [(110, RLit{Chr{10}}), (114, RLit{Chr{13}}), (116, RLit{Chr{9}}), (102, RLit{Chr{12}}),
   (118, RLit{Chr{11}}), (48, RLit{Chr{0}}), (98, RWordB{False{}}), (66, RWordB{True{}}),
   (100, RSet{False{}, Regex.digits()}), (68, RSet{True{}, Regex.digits()}),
   (119, RSet{False{}, Regex.words()}), (87, RSet{True{}, Regex.words()}),
   (115, RSet{False{}, Regex.spaces()}), (83, RSet{True{}, Regex.spaces()})]

def Regex.parse.esc.find(+x: U32, tbl: List<&2, Sigma<&2, &2, U32, _ => Re>>) -> Maybe<&2, Re>:
  match tbl:
    case Nil{}:
      None{}
    case (c, r) <> rest:
      Bool.pick(Maybe<&2, Re>, U32.is_eq(x, c), Some{r}, Regex.parse.esc.find(x, rest))

# Off the table: \x \u \U take hex digits; a digit is a backreference, a
# letter is unknown, anything else is the char itself
def Regex.parse.esc.go(+k: Nat, hex: Nat, +x: U32, t: String, found: Maybe<&2, Re>) -> Regex.parse.Item():
  match hex found:
    case _ Some{r}:
      Done{(r, t)}
    case 0n None{}:
      Bool.pick(Regex.parse.Item(), Char.is_digit(Chr{x}), Fail{RErr{k, "backreferences are not supported"}},
        Bool.pick(Regex.parse.Item(), Char.is_alpha(Chr{x}), Fail{RErr{k, "bad escape"}},
          Done{(RLit{Chr{x}}, t)}))
    case 1n None{}:
      Regex.parse.esc.code(k, Regex.parse.hex(k, 2n, t, 0))
    case 2n None{}:
      Regex.parse.esc.code(k, Regex.parse.hex(k, 4n, t, 0))
    case 3n+_ None{}:
      Regex.parse.esc.code(k, Regex.parse.hex(k, 8n, t, 0))

# The tail after a backslash
def Regex.parse.esc(t: String) -> Regex.parse.Item():
  match t:
    case SNil{}:
      Fail{RErr{1n, "bad escape"}}
    case SCon{Chr{+x}, +t2}:
      Regex.parse.esc.go(Nat.add(2n, String.length(t2)), Regex.parse.index(x, [120, 117, 85], 1n), x, t2,
        Regex.parse.esc.find(x, Regex.parse.esc.table()))

def Regex.parse.cls.flush(pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges()) -> Regex.Ranges():
  match pend dash:
    case Some{+p} True{}:
      (45, 45) <> ((p, p) <> rs)
    case Some{+p} False{}:
      (p, p) <> rs
    case None{} _:
      rs

def Regex.parse.cls.range(+t: String, rs: Regex.Ranges(), +lo: U32, +c: U32) -> Regex.parse.Scan():
  Bool.pick(Regex.parse.Scan(), U32.is_le(lo, c),
    Done{CScan{t, None{}, False{}, (lo, c) <> rs}},
    Fail{RErr{String.length(t), "bad character range"}})

# One char joins the class: closes a pending range, or waits as the next low
def Regex.parse.cls.item(t: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges(), +c: U32) -> Regex.parse.Scan():
  match pend dash:
    case Some{lo} True{}:
      Regex.parse.cls.range(t, rs, lo, c)
    case Some{+p} False{}:
      Done{CScan{t, Some{c}, False{}, (p, p) <> rs}}
    case None{} _:
      Done{CScan{t, Some{c}, False{}, rs}}

def Regex.parse.cls.union(+t: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges(), rs2: Regex.Ranges()) -> Regex.parse.Scan():
  match dash:
    case True{}:
      Fail{RErr{String.length(t), "bad character range"}}
    case False{}:
      Done{CScan{t, None{}, False{}, List.append(&2, Regex.Range(), rs2, Regex.parse.cls.flush(pend, False{}, rs))}}

# An escape inside a class: a char, a set (`\D` as its complement), `\b` is backspace
def Regex.parse.cls.add(pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges(), r: Regex.parse.Item()) -> Regex.parse.Scan():
  match r:
    case Fail{e}:
      Fail{e}
    case Done{(RLit{Chr{c}}, t)}:
      Regex.parse.cls.item(t, pend, dash, rs, c)
    case Done{(RSet{False{}, rs2}, t)}:
      Regex.parse.cls.union(t, pend, dash, rs, rs2)
    case Done{(RSet{True{}, rs2}, t)}:
      Regex.parse.cls.union(t, pend, dash, rs, Regex.ranges.not(rs2, 0))
    case Done{(RWordB{False{}}, t)}:
      Regex.parse.cls.item(t, pend, dash, rs, 8)
    case Done{(re, +t)}:
      Fail{RErr{String.length(t), "bad escape"}}

def Regex.parse.cls.dash(t: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges()) -> Regex.parse.Scan():
  match pend dash:
    case Some{lo} True{}:
      Regex.parse.cls.range(t, rs, lo, 45)
    case Some{p} False{}:
      Done{CScan{t, Some{p}, True{}, rs}}
    case None{} _:
      Done{CScan{t, Some{45}, False{}, rs}}

def Regex.parse.cls.char(kind: Nat, +x: U32, t: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges()) -> Regex.parse.Scan():
  match kind:
    case 0n:
      Regex.parse.cls.item(t, pend, dash, rs, x)
    case 1n:
      Done{CDone{Regex.parse.cls.flush(pend, dash, rs), t}}
    case 2n:
      Regex.parse.cls.add(pend, dash, rs, Regex.parse.esc(t))
    case 3n+_:
      Regex.parse.cls.dash(t, pend, dash, rs)

def Regex.parse.cls.step(s: String, pend: Maybe<&2, U32>, dash: Bool, rs: Regex.Ranges()) -> Regex.parse.Scan():
  match s:
    case SNil{}:
      Fail{RErr{0n, "unterminated character set"}}
    case SCon{Chr{+x}, t}:
      Regex.parse.cls.char(Regex.parse.index(x, [93, 92, 45], 1n), x, t, pend, dash, rs)

def Regex.parse.cls(fuel: Nat, st: Regex.parse.Scan()) -> Regex.parse.Scan():
  match fuel st:
    case _ Fail{e}:
      Fail{e}
    case _ Done{CDone{rs, rest}}:
      Done{CDone{rs, rest}}
    case 0n Done{CScan{s, pend, dash, rs}}:
      Fail{RErr{0n, "unterminated character set"}}
    case 1n+f Done{CScan{s, pend, dash, rs}}:
      Regex.parse.cls(f, Regex.parse.cls.step(s, pend, dash, rs))

# A leading `]` is a member
def Regex.parse.set.lead(t: String) -> Regex.Cls:
  match t:
    case SCon{Chr{+x}, +t2}:
      Bool.pick(Regex.Cls, U32.is_eq(x, 93),
        CScan{t2, Some{93}, False{}, []}, CScan{SCon{Chr{x}, t2}, None{}, False{}, []})
    case SNil{}:
      CScan{SNil{}, None{}, False{}, []}

def Regex.parse.set.head(t: String) -> Bool & String:
  match t:
    case SCon{Chr{+x}, +t2}:
      Bool.pick(Bool & String, U32.is_eq(x, 94), (True{}, t2), (False{}, SCon{Chr{x}, t2}))
    case SNil{}:
      (False{}, SNil{})

def Regex.parse.set.fin(neg: Bool, r: Regex.parse.Scan()) -> Regex.parse.Item():
  match r:
    case Fail{e}:
      Fail{e}
    case Done{CDone{rs, rest}}:
      Done{(RSet{neg, rs}, rest)}
    case Done{CScan{s, pend, dash, rs}}:
      Fail{RErr{0n, "unterminated character set"}}

def Regex.parse.set.go(fuel: Nat, h: Bool & String) -> Regex.parse.Item():
  (neg, t) = h
  Regex.parse.set.fin(neg, Regex.parse.cls(fuel, Done{Regex.parse.set.lead(t)}))

# The tail after `[`
def Regex.parse.set(+t: String) -> Regex.parse.Item():
  Regex.parse.set.go(String.length(t), Regex.parse.set.head(t))

def Regex.parse.char(kind: Nat, +x: U32, t: String, n: Nat, stack: Regex.parse.Stack()) -> Regex.parse.State():
  match kind:
    case 0n:
      Regex.parse.push(t, n, stack, RLit{Chr{x}})
    case 1n:
      Regex.parse.open(t, n, stack)
    case 2n:
      match stack:
        case fr <> Nil{}:
          Fail{RErr{1n+String.length(t), "unbalanced parenthesis"}}
        case fr <> rest:
          Regex.parse.push.q(t, n, rest, Regex.parse.fin(fr), 1n)
        case Nil{}:
          Fail{RErr{1n+String.length(t), "unbalanced parenthesis"}}
    case 3n:
      match stack:
        case Frame{cap, alts, cat, q} <> rest:
          Done{(t, (n, Frame{cap, Regex.parse.cat(cat) <> alts, [], 0n} <> rest))}
        case Nil{}:
          Fail{RErr{String.length(t), "unbalanced parenthesis"}}
    case 4n:
      Regex.parse.rep(t, n, stack, 0n, None{})
    case 5n:
      Regex.parse.rep(t, n, stack, 1n, None{})
    # `?` after a fresh repeat makes it lazy, else it is the {0,1} repeat
    case 6n:
      match stack:
        case Frame{cap, alts, RRep{r, lo, hi, g} <> cat, 2n} <> rest:
          Done{(t, (n, Frame{cap, alts, RRep{r, lo, hi, False{}} <> cat, 0n} <> rest))}
        case other:
          Regex.parse.rep(t, n, other, 0n, Some{1n})
    case 7n:
      Regex.parse.braces(n, stack, Regex.parse.bounds(t))
    case 8n:
      Regex.parse.atom(n, stack, Regex.parse.set(t))
    case 9n:
      Regex.parse.push(t, n, stack, RAny{})
    case 10n:
      Regex.parse.push(t, n, stack, RBol{})
    case 11n:
      Regex.parse.push(t, n, stack, REol{})
    case 12n+_:
      Regex.parse.atom(n, stack, Regex.parse.esc(t))

def Regex.parse.step(st: Regex.parse.State()) -> Regex.parse.State():
  match st:
    case Fail{e}:
      Fail{e}
    case Done{(SNil{}, (n, stack))}:
      Done{(SNil{}, (n, stack))}
    case Done{(SCon{Chr{+x}, t}, (n, stack))}:
      Regex.parse.char(Regex.parse.index(x, [40, 41, 124, 42, 43, 63, 123, 91, 46, 94, 36, 92], 1n), x, t, n, stack)

def Regex.parse.go(fuel: Nat, st: Regex.parse.State()) -> Regex.parse.Tree():
  match fuel:
    case 0n:
      match st:
        case Fail{e}:
          Fail{e}
        case Done{(s, (n, stack))}:
          match stack:
            case fr <> Nil{}:
              Done{(Regex.parse.fin(fr), n)}
            case _:
              Fail{RErr{0n, "missing )"}}
    case 1n+f:
      Regex.parse.go(f, Regex.parse.step(st))

def Regex.parse.pos(len: Nat, r: Regex.parse.Tree()) -> Regex.parse.Tree():
  match r:
    case Fail{RErr{k, msg}}:
      Fail{RErr{Nat.sub(len, k), msg}}
    case Done{v}:
      Done{v}

# The tree and its group count
def Regex.parse.run(+p: String) -> Regex.parse.Tree():
  +len = String.length(p)
  Regex.parse.pos(len, Regex.parse.go(len, Done{(p, (0n, [Frame{None{}, [], [], 0n}]))}))

def Regex.parse.re(r: Regex.parse.Tree()) -> Regex.Res(Re):
  match r:
    case Fail{e}:
      Fail{e}
    case Done{(re, n)}:
      Done{re}

def Regex.parse(p: String) -> Regex.Res(Re):
  Regex.parse.re(Regex.parse.run(p))

# Flags: (i, (m, s))
def Regex.Flags() -> Data:
  Sigma<&2, &2, Bool, _ => Sigma<&2, &2, Bool, _ => Bool>>

def Regex.flags.kind(s: String) -> Nat:
  match s:
    case SNil{}:
      0n
    case SCon{Chr{+x}, t}:
      Regex.parse.index(x, [105, 109, 115], 1n)

# `kind` classifies the head of `f`: 0 unknown, 1 i, 2 m, 3 s
def Regex.flags(f: String, kind: Nat, fl: Regex.Flags()) -> Regex.Res(Regex.Flags()):
  match f kind fl:
    case SNil{}, _, fl:
      Done{fl}
    case SCon{c, t}, 0n, _:
      Fail{RErr{0n, "unknown flag"}}
    case SCon{c, +t}, 1n, (False{}, r):
      Regex.flags(t, Regex.flags.kind(t), (True{}, r))
    case SCon{c, +t}, 2n, (i, (False{}, s)):
      Regex.flags(t, Regex.flags.kind(t), (i, (True{}, s)))
    case SCon{c, +t}, 3n+_, (i, (m, False{})):
      Regex.flags(t, Regex.flags.kind(t), (i, (m, True{})))
    case SCon{c, t}, _, _:
      Fail{RErr{0n, "duplicate flag"}}

# Program size of a tree, saturating just past max_prog
def Regex.size.rep(+s: Nat, +lo: Nat, hi: Maybe<&2, Nat>) -> Nat:
  match hi:
    case None{}:
      Nat.add(Nat.mul(lo, s), 2n+s)
    case Some{h}:
      Nat.add(Nat.mul(lo, s), Nat.mul(Nat.sub(h, lo), 1n+s))

def Regex.size(r: Re) -> Nat:
  match r:
    case RCat{a, b}:
      Nat.add(Regex.size(a), Regex.size(b))
    case RAlt{a, b}:
      2n+Nat.add(Regex.size(a), Regex.size(b))
    case RRep{r, lo, hi, g}:
      Nat.min(1n+Regex.max_prog(), Regex.size.rep(Regex.size(r), lo, hi))
    case RCap{n, r}:
      2n+Regex.size(r)
    case REmpty{}:
      0n
    case _:
      1n

# Bounded repeats unfold into copies: `x{2,4}` = `x x (x (x)?)?`, `x{2,}` =
# `x x x*`. After this only `x*` and `x?` remain (lo = 0)
def Regex.expand.opt(+e: Re, k: Nat, +g: Bool) -> Re:
  match k:
    case 0n:
      REmpty{}
    case 1n+p:
      RRep{RCat{e, Regex.expand.opt(e, p, g)}, 0n, Some{1n}, g}

def Regex.expand.times(n: Nat, +e: Re, acc: Re) -> Re:
  match n:
    case 0n:
      acc
    case 1n+p:
      Regex.expand.times(p, e, RCat{e, acc})

def Regex.expand.rep(+e: Re, +lo: Nat, hi: Maybe<&2, Nat>, +g: Bool) -> Re:
  match hi:
    case None{}:
      Regex.expand.times(lo, e, RRep{e, 0n, None{}, g})
    case Some{h}:
      Regex.expand.times(lo, e, Regex.expand.opt(e, Nat.sub(h, lo), g))

def Regex.expand(r: Re) -> Re:
  match r:
    case RCat{a, b}:
      RCat{Regex.expand(a), Regex.expand(b)}
    case RAlt{a, b}:
      RAlt{Regex.expand(a), Regex.expand(b)}
    case RRep{r, lo, hi, g}:
      Regex.expand.rep(Regex.expand(r), lo, hi, g)
    case RCap{n, r}:
      RCap{n, Regex.expand(r)}
    case other:
      other

# Case folding (flag i): add the ASCII letters of the other case
def Regex.emit.swap(+a: U32, +b: U32, lo: U32, hi: U32, acc: Regex.Ranges()) -> Regex.Ranges():
  +x = U32.max(a, lo)
  +y = U32.min(b, hi)
  List.filter.put(Regex.Range(), (U32.xor(x, 32), U32.xor(y, 32)), acc, U32.is_le(x, y))

def Regex.emit.fold(rs: Regex.Ranges()) -> Regex.Ranges():
  match rs:
    case Nil{}:
      []
    case (+a, +b) <> rest:
      (a, b) <> Regex.emit.swap(a, b, 65, 90, Regex.emit.swap(a, b, 97, 122, Regex.emit.fold(rest)))

def Regex.emit.ranges(fl: Regex.Flags(), rs: Regex.Ranges()) -> Regex.Ranges():
  (i, r) = fl
  match i:
    case True{}:
      Regex.emit.fold(rs)
    case False{}:
      rs

def Regex.emit.lit(fl: Regex.Flags(), +c: U32) -> Inst:
  (i, r) = fl
  match i:
    case True{}:
      ISet{False{}, Regex.emit.fold([(c, c)])}
    case False{}:
      IChr{c}

def Regex.emit.any(fl: Regex.Flags()) -> Inst:
  (i, (m, s)) = fl
  match s:
    case True{}:
      ISet{True{}, []}
    case False{}:
      IAny{}

def Regex.emit.multi(fl: Regex.Flags()) -> Bool:
  (i, (m, s)) = fl
  m

def Regex.emit.split(g: Bool, x: Nat, y: Nat) -> Inst:
  match g:
    case True{}:
      ISplit{x, y}
    case False{}:
      ISplit{y, x}

# Code for an expanded tree placed at address `pc`
def Regex.emit(r: Re, +fl: Regex.Flags(), +pc: Nat) -> List<&2, Inst>:
  match r:
    case RLit{Chr{c}}:
      [Regex.emit.lit(fl, c)]
    case RAny{}:
      [Regex.emit.any(fl)]
    case RSet{neg, rs}:
      [ISet{neg, Regex.emit.ranges(fl, rs)}]
    case RCat{+a, b}:
      List.append(&2, Inst, Regex.emit(a, fl, pc), Regex.emit(b, fl, Nat.add(pc, Regex.size(a))))
    case RAlt{+a, +b}:
      +q = Nat.add(2n+pc, Regex.size(a))
      ISplit{1n+pc, q} <> List.append(&2, Inst, Regex.emit(a, fl, 1n+pc), IJmp{Nat.add(q, Regex.size(b))} <> Regex.emit(b, fl, q))
    case RRep{+r, lo, None{}, g}:
      Regex.emit.split(g, 1n+pc, Nat.add(2n+pc, Regex.size(r))) <> List.append(&2, Inst, Regex.emit(r, fl, 1n+pc), [IJmp{pc}])
    case RRep{+r, lo, Some{h}, g}:
      Regex.emit.split(g, 1n+pc, Nat.add(1n+pc, Regex.size(r))) <> Regex.emit(r, fl, 1n+pc)
    case RCap{+n, r}:
      ISave{Nat.mul(2n, n)} <> List.append(&2, Inst, Regex.emit(r, fl, 1n+pc), [ISave{1n+Nat.mul(2n, n)}])
    case RBol{}:
      [IBol{Regex.emit.multi(fl)}]
    case REol{}:
      [IEol{Regex.emit.multi(fl)}]
    case RWordB{neg}:
      [IWordB{neg}]
    case REmpty{}:
      []

def Regex.compile.prog(fl: Regex.Flags(), re: Re, n: Nat, big: Bool) -> Regex.Res(Regex):
  match big:
    case True{}:
      Fail{RErr{0n, "program exceeds max_prog"}}
    case False{}:
      Done{Regex{ISave{0n} <> List.append(&2, Inst, Regex.emit(Regex.expand(re), fl, 1n), [ISave{1n}, IMatch{}]), n}}

def Regex.compile.tree(fr: Regex.Res(Regex.Flags()), t: Regex.parse.Tree()) -> Regex.Res(Regex):
  match fr t:
    case Fail{e} _:
      Fail{e}
    case Done{fl} Fail{e}:
      Fail{e}
    case Done{fl} Done{(+re, n)}:
      Regex.compile.prog(fl, re, n, Nat.is_gt(3n+Regex.size(re), Regex.max_prog()))

# Flags: any of `i` (ASCII case fold) `m` (multiline anchors) `s` (dot
# matches newline), each at most once
def Regex.compile(p: String, +f: String) -> Regex.Res(Regex):
  Regex.compile.tree(Regex.flags(f, Regex.flags.kind(f), (False{}, (False{}, False{}))), Regex.parse.run(p))

# Reference Pike VM. A thread is a pc and its capture slots; at each
# position the live threads are kept in priority order, so the first
# thread to reach IMatch is the leftmost-first answer and cuts every
# thread below it. Slow but total: O(n·m²) on every lane
def Regex.exec.Caps() -> Data:
  List<&2, Maybe<&2, Nat>>

def Regex.exec.Thread() -> Data:
  Sigma<&2, &2, Nat, _ => Regex.exec.Caps()>

def Regex.exec.Threads() -> Data:
  List<&2, Regex.exec.Thread()>

# Where the VM stands: (position, (previous char, subject from here))
def Regex.exec.Ctx() -> Data:
  Sigma<&2, &2, Nat, _ => Sigma<&2, &2, Maybe<&2, U32>, _ => String>>

# Closure state: (work stack, (pcs seen at this position, live threads reversed))
def Regex.exec.CState() -> Data:
  Sigma<&2, &2, Regex.exec.Threads(), _ => Sigma<&2, &2, List<&2, Nat>, _ => Regex.exec.Threads()>>

# Run state: (threads to close at the next position, best match so far)
def Regex.exec.State() -> Data:
  Sigma<&2, &2, Regex.exec.Threads(), _ => Maybe<&2, Regex.exec.Caps()>>

def Regex.exec.inset(+y: U32, rs: Regex.Ranges()) -> Bool:
  match rs:
    case Nil{}:
      False{}
    case (a, b) <> rest:
      Bool.or(Bool.and(U32.is_le(a, y), U32.is_le(y, b)), Regex.exec.inset(y, rest))

def Regex.exec.peek(s: String) -> Maybe<&2, U32>:
  match s:
    case SNil{}:
      None{}
    case SCon{Chr{x}, t}:
      Some{x}

def Regex.exec.is_word(m: Maybe<&2, U32>) -> Bool:
  match m:
    case None{}:
      False{}
    case Some{x}:
      Regex.exec.inset(x, Regex.words())

def Regex.exec.is_nl(m: Maybe<&2, U32>) -> Bool:
  match m:
    case None{}:
      False{}
    case Some{x}:
      U32.is_eq(x, 10)

def Regex.exec.bol(multi: Bool, ctx: Regex.exec.Ctx()) -> Bool:
  match ctx:
    case (0n, r):
      True{}
    case (i, (prev, tail)):
      Bool.and(multi, Regex.exec.is_nl(prev))

# `$`: at the end, before a final newline, or (multiline) before any newline
def Regex.exec.eol(multi: Bool, ctx: Regex.exec.Ctx()) -> Bool:
  match ctx:
    case (i, (prev, SNil{})):
      True{}
    case (i, (prev, SCon{Chr{x}, SNil{}})):
      U32.is_eq(x, 10)
    case (i, (prev, SCon{Chr{x}, t})):
      Bool.and(multi, U32.is_eq(x, 10))

# CPython: on the empty string neither `\b` nor `\B` holds
def Regex.exec.wordb(neg: Bool, ctx: Regex.exec.Ctx()) -> Bool:
  match ctx:
    case (0n, (prev, SNil{})):
      False{}
    case (i, (prev, tail)):
      Bool.xor(neg, Bool.xor(Regex.exec.is_word(prev), Regex.exec.is_word(Regex.exec.peek(tail))))

def Regex.exec.pos(ctx: Regex.exec.Ctx()) -> Nat:
  (i, r) = ctx
  i

def Regex.exec.save(caps: Regex.exec.Caps(), slot: Nat, +i: Nat) -> Regex.exec.Caps():
  match caps slot:
    case Nil{} _:
      Nil{}
    case h <> t 0n:
      Some{i} <> t
    case h <> t 1n+p:
      h <> Regex.exec.save(t, p, i)

# One closure step at `pc`: `dup` = already seen here, `inst` = prog[pc]
def Regex.exec.step(ctx: Regex.exec.Ctx(), +pc: Nat, +caps: Regex.exec.Caps(), work: Regex.exec.Threads(), seen: List<&2, Nat>, acc: Regex.exec.Threads(), dup: Bool, inst: Maybe<&2, Inst>) -> Regex.exec.CState():
  match dup inst:
    case True{} _:
      (work, (seen, acc))
    case False{} Some{ISplit{x, y}}:
      ((x, caps) <> ((y, caps) <> work), (pc <> seen, acc))
    case False{} Some{IJmp{x}}:
      ((x, caps) <> work, (pc <> seen, acc))
    case False{} Some{ISave{+slot}}:
      (List.filter.put(Regex.exec.Thread(), (1n+pc, Regex.exec.save(caps, slot, Regex.exec.pos(ctx))), work, Nat.is_lt(slot, List.length(&2, Maybe<&2, Nat>, caps))), (pc <> seen, acc))
    case False{} Some{IBol{multi}}:
      (List.filter.put(Regex.exec.Thread(), (1n+pc, caps), work, Regex.exec.bol(multi, ctx)), (pc <> seen, acc))
    case False{} Some{IEol{multi}}:
      (List.filter.put(Regex.exec.Thread(), (1n+pc, caps), work, Regex.exec.eol(multi, ctx)), (pc <> seen, acc))
    case False{} Some{IWordB{neg}}:
      (List.filter.put(Regex.exec.Thread(), (1n+pc, caps), work, Regex.exec.wordb(neg, ctx)), (pc <> seen, acc))
    case False{} Some{i}:
      (work, (pc <> seen, (pc, caps) <> acc))
    case False{} None{}:
      (work, (pc <> seen, acc))

# ε-closure over a work stack; each pc is expanded at most once per
# position, so 3·(1+|prog|) steps always drain the stack
def Regex.exec.close(fuel: Nat, +prog: List<&2, Inst>, +ctx: Regex.exec.Ctx(), st: Regex.exec.CState()) -> Regex.exec.CState():
  match fuel st:
    case 0n, st:
      st
    case 1n+f, (Nil{}, r):
      (Nil{}, r)
    case 1n+f, ((+pc, caps) <> work, (+seen, acc)):
      Regex.exec.close(f, prog, ctx, Regex.exec.step(ctx, pc, caps, work, seen, acc, List.contains(~Nat, ~Nat.is_eq, seen, pc), List.get(&2, Inst, prog, pc)))

def Regex.exec.closed(st: Regex.exec.CState()) -> Regex.exec.Threads():
  (work, (seen, acc)) = st
  List.reverse(&2, Regex.exec.Thread(), acc)

# Threads alive at this position, in priority order: the closure of the
# survivors plus, while no match is known, a fresh thread at the bottom
def Regex.exec.alive(+prog: List<&2, Inst>, ctx: Regex.exec.Ctx(), work: Regex.exec.Threads(), start: Regex.exec.Threads(), best: Bool) -> Regex.exec.Threads():
  Regex.exec.closed(Regex.exec.close(Nat.mul(3n, 1n+List.length(&2, Inst, prog)), prog, ctx, (List.append(&2, Regex.exec.Thread(), work, Bool.pick(Regex.exec.Threads(), best, [], start)), ([], []))))

def Regex.exec.eat(inst: Inst, c: Maybe<&2, U32>) -> Bool:
  match inst c:
    case IChr{x} Some{y}:
      U32.is_eq(x, y)
    case IAny{} Some{y}:
      Bool.not(U32.is_eq(y, 10))
    case ISet{neg, rs} Some{+y}:
      Bool.xor(neg, Regex.exec.inset(y, rs))
    case _ _:
      False{}

def Regex.exec.fetch(+prog: List<&2, Inst>, ts: Regex.exec.Threads()) -> Maybe<&2, Inst>:
  match ts:
    case Nil{}:
      None{}
    case (pc, caps) <> rest:
      List.get(&2, Inst, prog, pc)

# Feed the char to the live threads in priority order. IMatch wins and cuts
# the rest, unless `cut` is off (fullmatch before the end); `inst` = prog[head pc]
def Regex.exec.adv(clist: Regex.exec.Threads(), +prog: List<&2, Inst>, cut: Bool, +c: Maybe<&2, U32>, inst: Maybe<&2, Inst>, work: Regex.exec.Threads()) -> Maybe<&2, Regex.exec.Caps()> & Regex.exec.Threads():
  match clist cut inst:
    case Nil{}, _, _:
      (None{}, work)
    case (pc, caps) <> +rest, False{}, Some{IMatch{}}:
      Regex.exec.adv(rest, prog, False{}, c, Regex.exec.fetch(prog, rest), work)
    case (pc, caps) <> rest, _, Some{IMatch{}}:
      (Some{caps}, work)
    case (pc, caps) <> +rest, k, Some{i}:
      Regex.exec.adv(rest, prog, k, c, Regex.exec.fetch(prog, rest), List.filter.put(Regex.exec.Thread(), (1n+pc, caps), work, Regex.exec.eat(i, c)))
    case (pc, caps) <> +rest, k, None{}:
      Regex.exec.adv(rest, prog, k, c, Regex.exec.fetch(prog, rest), work)

def Regex.exec.feed(+prog: List<&2, Inst>, full: Bool, ne: Bool, +c: Maybe<&2, U32>, +clist: Regex.exec.Threads()) -> Maybe<&2, Regex.exec.Caps()> & Regex.exec.Threads():
  Regex.exec.adv(clist, prog, Bool.and(Bool.not(ne), Bool.or(Bool.not(full), Maybe.is_none(&2, U32, c))), c, Regex.exec.fetch(prog, clist), [])

def Regex.exec.post(best: Maybe<&2, Regex.exec.Caps()>, r: Maybe<&2, Regex.exec.Caps()> & Regex.exec.Threads()) -> Regex.exec.State():
  (found, acc) = r
  (List.reverse(&2, Regex.exec.Thread(), acc), Maybe.or(&2, Regex.exec.Caps(), found, best))

def Regex.exec.groups(caps: Regex.exec.Caps()) -> List<&2, Maybe<&2, Regex.Span()>>:
  match caps:
    case Some{a} <> (Some{b} <> gs):
      Some{(a, b)} <> Regex.exec.groups(gs)
    case x <> (y <> gs):
      None{} <> Regex.exec.groups(gs)
    case _:
      []

def Regex.exec.match(caps: Regex.exec.Caps()) -> Maybe<&2, Match>:
  match caps:
    case Some{a} <> (Some{b} <> gs):
      Some{Match{a, b, Regex.exec.groups(gs)}}
    case _:
      None{}

def Regex.exec.fin(st: Regex.exec.State()) -> Maybe<&2, Match>:
  match st:
    case (work, Some{caps}):
      Regex.exec.match(caps)
    case (work, None{}):
      None{}

# Walks the subject and gives where it stopped; the first `at` chars are
# skipped (they still set `prev` for `^` `\b`). `anchored` drops the fresh
# thread after `at`; `ne` refuses an empty match at `at` without cutting
# the threads below it
def Regex.exec.run(s: String, at: Nat, +prog: List<&2, Inst>, +full: Bool, +anchored: Bool, ne: Bool, +start: Regex.exec.Threads(), +i: Nat, prev: Maybe<&2, U32>, st: Regex.exec.State()) -> Maybe<&2, Match> & Nat:
  match s at st:
    case SCon{Chr{x}, t}, 1n+a, st:
      Regex.exec.run(t, a, prog, full, anchored, ne, start, 1n+i, Some{x}, st)
    case s, 0n, (Nil{}, Some{caps}):
      (Regex.exec.match(caps), i)
    case SCon{Chr{+x}, +t}, 0n, (work, +best):
      Regex.exec.run(t, 0n, prog, full, anchored, False{}, Bool.pick(Regex.exec.Threads(), anchored, [], start), 1n+i, Some{x}, Regex.exec.post(best, Regex.exec.feed(prog, full, ne, Some{x}, Regex.exec.alive(prog, (i, (prev, SCon{Chr{x}, t})), work, start, Maybe.is_some(&2, Regex.exec.Caps(), best)))))
    case SNil{}, _, (work, +best):
      (Regex.exec.fin(Regex.exec.post(best, Regex.exec.feed(prog, full, ne, None{}, Regex.exec.alive(prog, (i, (prev, SNil{})), work, start, Maybe.is_some(&2, Regex.exec.Caps(), best))))), i)

# Enters one char before `at` (a view, not a walk): it is only `prev` for
# `^` `\b`. `at` past the end clamps, as Python's pos
def Regex.exec.walk(re: Regex, +s: String, at: Nat, full: Bool, anchored: Bool, ne: Bool) -> Maybe<&2, Match> & Nat:
  match re:
    case Regex{prog, n}:
      +a = Nat.min(at, String.length(s))
      +k = Nat.sub(a, 1n)
      Regex.exec.run(String.drop(s, k), Nat.sub(a, k), prog, full, anchored, ne, [(0n, List.replicate(Maybe<&2, Nat>, Nat.mul(2n, 1n+n), None{}))], k, None{}, ([], None{}))

def Regex.exec.go(re: Regex, s: String, at: Nat, full: Bool, anchored: Bool, ne: Bool) -> Maybe<&2, Match>:
  Pair.fst(Maybe<&2, Match>, Nat, Regex.exec.walk(re, s, at, full, anchored, ne))

# First match starting at or after `at` (code-point offset)
def Regex.exec(re: Regex, s: String, at: Nat) -> Maybe<&2, Match>:
  Regex.exec.go(re, s, at, False{}, False{}, False{})

# Match starting exactly at `at`
def Regex.match_at(re: Regex, s: String, at: Nat) -> Maybe<&2, Match>:
  Regex.exec.go(re, s, at, False{}, True{}, False{})

# Match covering the whole subject
def Regex.fullmatch(re: Regex, s: String) -> Maybe<&2, Match>:
  Regex.exec.go(re, s, 0n, True{}, True{}, False{})

# Derived API. `find_all` `split` `replace` run on one budget, never reset
# per result: a search from `at` must afford (1 + length - at) * weight, the
# worst case of one exec whatever the lane, then pays only the span it
# walked; the output is paid from what is left before it is built. Out of
# budget is an error, never a partial answer
def Regex.find(re: Regex, s: String) -> Maybe<&2, Match>:
  Regex.exec(re, s, 0n)

def Regex.is_match(re: Regex, s: String) -> Bool:
  Maybe.is_some(&2, Match, Regex.find(re, s))

def Regex.group.get(s: String, g: Maybe<&2, Maybe<&2, Regex.Span()>>) -> Maybe<&2, String>:
  match g:
    case Some{Some{(+a, b)}}:
      Some{String.slice(s, a, b)}
    case _:
      None{}

# Group k of a match as text (0 = the whole match); None if it took no
# part. A Match keeps no source: `s` must be the subject it came from
def Regex.group(s: String, m: Match, k: Nat) -> Maybe<&2, String>:
  match m k:
    case Match{+a, b, gs} 0n:
      Some{String.slice(s, a, b)}
    case Match{a, b, gs} 1n+p:
      Regex.group.get(s, List.get(&2, Maybe<&2, Regex.Span()>, gs, p))

# Steps one position can cost: every inst once, each with a capture copy
# and its class probes
def Regex.weight.go(prog: List<&2, Inst>, +k: U32, acc: U32) -> U32:
  match prog:
    case Nil{}:
      acc
    case ISet{neg, rs} <> t:
      Regex.weight.go(t, k, U32.add(acc, U32.add(k, U32.from_nat(List.length(&2, Regex.Range(), rs)))))
    case i <> t:
      Regex.weight.go(t, k, U32.add(acc, k))

def Regex.weight(re: Regex) -> U32:
  match re:
    case Regex{prog, n}:
      Regex.weight.go(prog, U32.from_nat(3n+Nat.double(n)), 1)

# a * b, saturating so a cost never wraps under the budget
def Regex.cost(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, Bool.and(U32.is_gt(b, 0), U32.is_gt(a, U32.div(4294967295, b))), 4294967295, U32.mul(a, b))

def Regex.scan.Out() -> Data:
  Sigma<&2, &2, U32, _ => List<&2, Match>>

# Scan state: (the next search (at, ne), None once done; (budget left, matches reversed))
def Regex.scan.St() -> Data:
  Result<&2, &2, Regex.Error, Sigma<&2, &2, Maybe<&2, Sigma<&2, &2, Nat, _ => Bool>>, _ => Regex.scan.Out()>>

def Regex.scan.got(left: U32, acc: List<&2, Match>, m: Maybe<&2, Match>) -> Regex.scan.St():
  match m:
    case None{}:
      Done{(None{}, (left, acc))}
    case Some{Match{+a, +b, gs}}:
      Done{(Some{(b, Nat.is_eq(a, b))}, (left, Match{a, b, gs} <> acc))}

def Regex.scan.spent(left: U32, acc: List<&2, Match>, w: U32, at: Nat, r: Maybe<&2, Match> & Nat) -> Regex.scan.St():
  (m, stop) = r
  Regex.scan.got(U32.sub(left, Regex.cost(U32.from_nat(Nat.sub(1n+stop, at)), w)), acc, m)

def Regex.scan.pay(re: Regex, s: String, +at: Nat, ne: Bool, left: U32, acc: List<&2, Match>, w: U32, short: Bool) -> Regex.scan.St():
  match short:
    case True{}:
      Fail{RErr{at, "budget exhausted"}}
    case False{}:
      Regex.scan.spent(left, acc, w, at, Regex.exec.walk(re, s, at, False{}, False{}, ne))

# Python finditer order: after a match search on from its end; after an
# empty one search at the same place with `ne`, so the empty match is
# refused there. Each search ends further right or sets `ne`, so
# 3 + 2 * length of them always reach the end
def Regex.scan.go(fuel: Nat, +re: Regex, +s: String, +len: Nat, +w: U32, st: Regex.scan.St()) -> Result<&2, &2, Regex.Error, Regex.scan.Out()>:
  match fuel st:
    case _, Fail{e}:
      Fail{e}
    case 1n+f, Done{(Some{(+at, ne)}, (+left, acc))}:
      +cost = Regex.cost(U32.from_nat(Nat.sub(1n+len, at)), w)
      Regex.scan.go(f, re, s, len, w, Regex.scan.pay(re, s, at, ne, left, acc, w, U32.is_lt(left, cost)))
    case _, Done{(nx, (left, acc))}:
      Done{(left, List.reverse(&2, Match, acc))}

# Every match in order, and the budget left to pay for the output
def Regex.scan(+re: Regex, +s: String, limit: U32) -> Result<&2, &2, Regex.Error, Regex.scan.Out()>:
  +len = String.length(s)
  Regex.scan.go(3n+Nat.double(len), re, s, len, Regex.weight(re), Done{(Some{(0n, False{})}, (limit, []))})

def Regex.scan.matched(ms: List<&2, Match>) -> Nat:
  match ms:
    case Nil{}:
      0n
    case Match{a, b, gs} <> t:
      Nat.add(Nat.sub(b, a), Regex.scan.matched(t))

def Regex.scan.paid.if(ms: List<&2, Match>, short: Bool, len: Nat) -> Regex.Res(List<&2, Match>):
  match short:
    case True{}:
      Fail{RErr{len, "budget exhausted"}}
    case False{}:
      Done{ms}

# Pays for the output before it is built: `per` for each match, plus one
# for each unmatched code point when `gaps`
def Regex.scan.paid(r: Result<&2, &2, Regex.Error, Regex.scan.Out()>, gaps: Bool, per: Nat, +len: Nat) -> Regex.Res(List<&2, Match>):
  match r:
    case Fail{e}:
      Fail{e}
    case Done{(+left, +ms)}:
      +c = Regex.cost(U32.from_nat(List.length(&2, Match, ms)), U32.from_nat(per))
      Regex.scan.paid.if(ms, Bool.or(U32.is_lt(left, c), U32.is_lt(U32.sub(left, c), U32.from_nat(Bool.pick(Nat, gaps, Nat.sub(len, Regex.scan.matched(ms)), 0n)))), len)

def Regex.ngroups(re: Regex) -> Nat:
  match re:
    case Regex{prog, n}:
      n

# Every match as a Match record (Python `findall` gives strings instead);
# a record costs 1 + ngroups
def Regex.find_all(+re: Regex, +s: String, limit: U32) -> Regex.Res(List<&2, Match>):
  Regex.scan.paid(Regex.scan(re, s, limit), False{}, 1n+Regex.ngroups(re), String.length(s))

# The unmatched gaps around the matches, empty ones kept
def Regex.split.gaps(ms: List<&2, Match>, +s: String, +prev: Nat) -> List<&2, String>:
  match ms:
    case Nil{}:
      [String.drop(s, prev)]
    case Match{a, b, gs} <> t:
      String.slice(s, prev, a) <> Regex.split.gaps(t, s, b)

def Regex.split.fin(r: Regex.Res(List<&2, Match>), s: String) -> Regex.Res(List<&2, String>):
  match r:
    case Fail{e}:
      Fail{e}
    case Done{ms}:
      Done{Regex.split.gaps(ms, s, 0n)}

# The gaps only: separators and their groups are left out (Python `split`
# keeps the groups); an output code point costs 1
def Regex.split(+re: Regex, +s: String, limit: U32) -> Regex.Res(List<&2, String>):
  Regex.split.fin(Regex.scan.paid(Regex.scan(re, s, limit), True{}, 0n, String.length(s)), s)

def Regex.replace.fin(r: Regex.Res(List<&2, Match>), s: String, by: String) -> Regex.Res(String):
  match r:
    case Fail{e}:
      Fail{e}
    case Done{ms}:
      Done{String.join(Regex.split.gaps(ms, s, 0n), by)}

# Every match replaced by the literal text `by`; an output code point costs 1
def Regex.replace(+re: Regex, +s: String, +by: String, limit: U32) -> Regex.Res(String):
  Regex.replace.fin(Regex.scan.paid(Regex.scan(re, s, limit), True{}, String.length(by), String.length(s)), s, by)

# Text
# ----

def Nat.show.put(qr: Nat & Nat) -> Char & Nat:
  (q, r) = qr
  (Chr{U32.from_nat(Nat.add(48n, r))}, q)

law Nat.show.go:
  for f   : Nat
  for n   : Nat
  for acc : String
  String

def Nat.show.fin(g: Nat, acc: String, dq: Char & Nat) -> String:
  (d, q) = dq
  match q:
    case 0n:
      SCon{d, acc}
    case 1n+p:
      Nat.show.go(g, 1n+p, SCon{d, acc})

def Nat.show.go(f, n, acc):
  match f:
    case 0n:
      acc
    case 1n+g:
      Nat.show.fin(g, acc, Nat.show.put(Nat.divmod(n, 10n)))

def Nat.show(n: Nat) -> String:
  +m = n
  Nat.show.fin(m, SNil{}, Nat.show.put(Nat.divmod(m, 10n)))

law Nat.read.go:
  for s   : String
  for +acc : Nat
  Maybe<&2, Nat>

def Nat.read.max() -> Nat:
  Nat.mul(U32.to_nat(16777215), U32.to_nat(16777217))

def Nat.read.fit(acc: Nat, qr: Nat & Nat) -> Bool:
  (q, r) = qr
  Bool.not(Nat.is_lt(q, acc))

def Nat.read.if(t: String, acc: Nat, d: U32, ok: Bool) -> Maybe<&2, Nat>:
  match ok:
    case True{}:
      Nat.read.go(t, Nat.add(Nat.mul(acc, 10n), U32.to_nat(d)))
    case False{}:
      None{}

def Nat.read.go(s, acc):
  match s:
    case SNil{}:
      Some{acc}
    case SCon{Chr{x}, t}:
      +d = U32.sub(x, 48)
      Nat.read.if(t, acc, d, Bool.and(U32.is_lt(d, 10), Nat.read.fit(acc,
        Nat.divmod(Nat.sub(Nat.read.max(), U32.to_nat(d)), 10n))))

def Nat.read(s: String) -> Maybe<&2, Nat>:
  match s:
    case SNil{}:
      None{}
    case SCon{h, t}:
      Nat.read.go(SCon{h, t}, 0n)

law U32.show.go:
  for f   : Nat
  for +n  : U32
  for acc : String
  String

def U32.show.fin(g: Nat, acc: String, +n: U32, z: Bool) -> String:
  match z:
    case True{}:
      acc
    case False{}:
      U32.show.go(g, U32.div(n, 10),
        SCon{Chr{U32.add(48, U32.mod(n, 10))}, acc})

def U32.show.go(f, n, acc):
  match f:
    case 0n:
      acc
    case 1n+g:
      U32.show.fin(g, acc, n, U32.is_zero(n))

def U32.show.if(a: U32, z: Bool) -> String:
  match z:
    case True{}:
      SCon{Chr{48}, SNil{}}
    case False{}:
      U32.show.go(10n, a, SNil{})

def U32.show(a: U32) -> String:
  +b = a
  U32.show.if(b, U32.is_zero(b))

law U32.read.go:
  for s   : String
  for +acc : U32
  Maybe<&2, U32>

def U32.read.if(t: String, n: U32, ok: Bool) -> Maybe<&2, U32>:
  match ok:
    case True{}:
      U32.read.go(t, n)
    case False{}:
      None{}

def U32.read.go(s, acc):
  match s:
    case SNil{}:
      Some{acc}
    case SCon{Chr{x}, t}:
      +n = U32.add(U32.mul(acc, 10), U32.sub(x, 48))
      U32.read.if(t, n, U32.is_eq(U32.div(n, 10), acc))

def U32.read(s: String) -> Maybe<&2, U32>:
  match s:
    case SNil{}:
      None{}
    case SCon{h, t}:
      U32.read.go(SCon{h, t}, 0)

law String.format.go:
  for s: String
  for +args: List<&2, String>
  for +i: Nat
  for +man: Bool
  Maybe<&2, String>

def String.format.put(a: String, r: Maybe<&2, String>) -> Maybe<&2, String>:
  match r:
    case None{}:
      None{}
    case Some{x}:
      Some{String.append(a, x)}

def String.format.sub(v: Maybe<&2, String>, r: Maybe<&2, String>) ->
  Maybe<&2, String>:
  match v:
    case None{}:
      None{}
    case Some{x}:
      String.format.put(x, r)

# the field after '{': i counts the {} seen, man is whether a {n} was
def String.format.field(+args: List<&2, String>, +i: Nat, +man: Bool,
  parts: String & (String & String)) -> Maybe<&2, String>:
  (+spec, (sep, rest)) = parts
  +auto = String.is_empty(spec)
  +k = U32.read(spec)
  String.format.sub(
    Bool.pick(Maybe<&2, String>,
      Bool.and(Bool.not(String.is_empty(sep)), Bool.pick(Bool, auto, Bool.not(man),
        Bool.and(Nat.is_eq(i, 0n), Maybe.is_some(&2, U32, k)))),
      List.get(&2, String, args, Bool.pick(Nat, auto, i,
        U32.to_nat(Maybe.default(&2, U32, k, 0)))), None{}),
    String.format.go(rest, args, Bool.pick(Nat, auto, 1n+i, i), Bool.or(man, Bool.not(auto))))

def String.format.step(+h: Char, t: String, args: List<&2, String>, i: Nat,
  man: Bool, skip: Nat, lit: Bool, open: Bool) -> Maybe<&2, String>:
  match lit:
    case True{}:
      String.format.put(SCon{h, SNil{}},
        String.format.go(String.drop(t, skip), args, i, man))
    case False{}:
      match open:
        case True{}:
          String.format.field(args, i, man, String.partition(t, "}"))
        case False{}:
          None{}

def String.format.go(s, args, i, man):
  match s:
    case SNil{}:
      Some{SNil{}}
    case SCon{+h, +t}:
      +open = Char.is_eq(h, '{')
      +brace = Bool.or(open, Char.is_eq(h, '}'))
      +dbl = Bool.and(brace, String.starts_with(t, SCon{h, SNil{}}))
      String.format.step(h, t, args, i, man, Bool.pick(Nat, dbl, 1n, 0n),
        Bool.or(dbl, Bool.not(brace)), open)

# Python's str.format, v1: {} takes the next arg, {n} the nth, {{ and }} are
# the braces. None on a lone brace, an index past args, {} mixed with {n}, or
# anything else in a field: no names, no !r, no :spec mini-language (future)
def String.format(s: String, args: List<&2, String>) -> Maybe<&2, String>:
  String.format.go(s, args, 0n, False{})

def Bool.show(b: Bool) -> String:
  match b:
    case False{}:
      "False"
    case True{}:
      "True"

def Char.show(c: Char) -> String:
  SCon{c, SNil{}}

def Maybe.show(~a: Quant, ~A: Kind(a), ~f: A -> String, m: Maybe<a, A>) ->
  String:
  match m:
    case None{}:
      "None"
    case Some{x}:
      "Some(" ++ f(x) ++ ")"

def List.show.go(~a: Quant, ~A: Kind(a), ~f: A -> String, xs: List<a, A>) ->
  String:
  match xs:
    case Nil{}:
      "]"
    case h <> t:
      ", " ++ f(h) ++ List.show.go(~a, ~A, ~f, t)

def List.show(~a: Quant, ~A: Kind(a), ~f: A -> String, xs: List<a, A>) ->
  String:
  match xs:
    case Nil{}:
      "[]"
    case h <> t:
      "[" ++ f(h) ++ List.show.go(~a, ~A, ~f, t)

# Array
# -----

def Array.size.node(-T: Type, ys: Array<T>, r: Array<T> & U32) ->
  Array<T> & U32:
  (xs2, n) = r
  (ANode{xs2, ys}, U32.shl(n))

def Array.size(-T: Type, a: Array<T>) -> Array<T> & U32:
  match a:
    case ALeaf{x}:
      (ALeaf{x}, 1)
    case ANode{xs, ys}:
      Array.size.node(T, ys, Array.size(T, xs))

def Array.swap.lo(-T: Type, ys: Array<T>, r: Array<T> & T) -> Array<T> & T:
  (nxs, old) = r
  (ANode{nxs, ys}, old)

def Array.swap.hi(-T: Type, xs: Array<T>, r: Array<T> & T) -> Array<T> & T:
  (nys, old) = r
  (ANode{xs, nys}, old)

law Array.swap.go:
  for -T: Type
  for  a: Array<T>
  for  n: U32
  for +i: U32
  for  v: T
  Array<T> & T

def Array.swap.if(
  -T: Type, xs: Array<T>, ys: Array<T>, +h: U32, i: U32, v: T, z: Bool
) -> Array<T> & T:
  match z:
    case True{}:
      Array.swap.lo(T, ys, Array.swap.go(T, xs, h, i, v))
    case False{}:
      Array.swap.hi(T, xs, Array.swap.go(T, ys, h, U32.sub(i, h), v))

def Array.swap.go(T, a, n, i, v):
  match a:
    case ALeaf{x}:
      (ALeaf{v}, x)
    case ANode{xs, ys}:
      +h = U32.shr(n)
      Array.swap.if(T, xs, ys, h, i, v, U32.is_lt(i, h))

def Array.swap.at(-T: Type, i: U32, v: T, an: Array<T> & U32) -> Array<T> & T:
  (a, +n) = an
  Array.swap.go(T, a, n, U32.and(i, U32.sub(n, 1)), v)

def Array.swap(-T: Type, a: Array<T>, +i: U32, v: T) -> Array<T> & T:
  Array.swap.at(T, i, v, Array.size(T, a))

def Array.clone.node(
  -T: Data, cx: Array<T> & Array<T>, cy: Array<T> & Array<T>
) -> Array<T> & Array<T>:
  (xa, xb) = cx
  (ya, yb) = cy
  (ANode{xa, ya}, ANode{xb, yb})

def Array.clone(-T: Data, a: Array<T>) -> Array<T> & Array<T>:
  match a:
    case ALeaf{+x}:
      (ALeaf{x}, ALeaf{x})
    case ANode{xs, ys}:
      Array.clone.node(T, Array.clone(T, xs), Array.clone(T, ys))

# A fork is two handles to one array in O(1). Both read and write the
# same cells, so racing writes go through the atomics; a match on a
# handle copies its part, as a clone does. An atomic takes an Array<U32>
# (fadd an Array<F32>) and returns the array with the old value.

@unsafe
def Array.fork(-T: Type, +a: Array<T>) -> Array<T> & Array<T>:
  (a, a)

@unsafe
def Array.join(-T: Type, a: Array<T>, b: Array<T>) -> Array<T>:
  a

law Array.atomic.add:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.min:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.max:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.and:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.or:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.xor:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.exch:
  for  a: Array<U32>
  for +i: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.cas:
  for  a: Array<U32>
  for +i: U32
  for +x: U32
  for +v: U32
  Array<U32> & U32

law Array.atomic.fadd:
  for  a: Array<F32>
  for +i: U32
  for +v: F32
  Array<F32> & F32

def Array.new(-T: Data, +d: Nat, +v: T) -> Array<T>:
  match d:
    case 0n:
      ALeaf{v}
    case 1n+p:
      ANode{[v : T^p], [v : T^p]}

def Array.set.fin(-T: Type, r: Array<T> & T) -> Array<T>:
  (na, old) = r
  na

def Array.set(-T: Type, a: Array<T>, i: U32, v: T) -> Array<T>:
  Array.set.fin(T, Array.swap(T, a, i, v))

law Array.get.go:
  for -T: Data
  for  a: Array<T>
  for  n: U32
  for +i: U32
  Array<T> & T

def Array.get.if(
  -T: Data, xs: Array<T>, ys: Array<T>, +h: U32, i: U32, z: Bool
) -> Array<T> & T:
  match z:
    case True{}:
      Array.swap.lo(T, ys, Array.get.go(T, xs, h, i))
    case False{}:
      Array.swap.hi(T, xs, Array.get.go(T, ys, h, U32.sub(i, h)))

def Array.get.go(T, a, n, i):
  match a:
    case ALeaf{+x}:
      (ALeaf{x}, x)
    case ANode{xs, ys}:
      +h = U32.shr(n)
      Array.get.if(T, xs, ys, h, i, U32.is_lt(i, h))

def Array.get.at(-T: Data, i: U32, an: Array<T> & U32) -> Array<T> & T:
  (a, +n) = an
  Array.get.go(T, a, n, U32.and(i, U32.sub(n, 1)))

def Array.get(-T: Data, a: Array<T>, +i: U32) -> Array<T> & T:
  Array.get.at(T, i, Array.size(T, a))

# Array.get4: slots i..i+3, each wrapped as by Array.get (natively one
# range check for the four, so C can widen the loads).
def Array.get4.d(-T: Data, x0: T, x1: T, x2: T, r: Array<T> & T) ->
  Array<T> & (T & (T & (T & T))):
  (a, x3) = r
  (a, (x0, (x1, (x2, x3))))

def Array.get4.c(-T: Data, +i: U32, x0: T, x1: T, r: Array<T> & T) ->
  Array<T> & (T & (T & (T & T))):
  (a, x2) = r
  Array.get4.d(T, x0, x1, x2, Array.get(T, a, U32.add(i, 3)))

def Array.get4.b(-T: Data, +i: U32, x0: T, r: Array<T> & T) ->
  Array<T> & (T & (T & (T & T))):
  (a, x1) = r
  Array.get4.c(T, i, x0, x1, Array.get(T, a, U32.add(i, 2)))

def Array.get4.a(-T: Data, +i: U32, r: Array<T> & T) ->
  Array<T> & (T & (T & (T & T))):
  (a, x0) = r
  Array.get4.b(T, i, x0, Array.get(T, a, U32.add(i, 1)))

def Array.get4(-T: Data, a: Array<T>, +i: U32) ->
  Array<T> & (T & (T & (T & T))):
  Array.get4.a(T, i, Array.get(T, a, i))

def Array.to_list.go(~T: Type, a: Array<T>, acc: List<T>) -> List<T>:
  match a:
    case ALeaf{x}:
      x <> acc
    case ANode{xs, ys}:
      Array.to_list.go(~T, xs, Array.to_list.go(~T, ys, acc))

def Array.to_list(~T: Type, a: Array<T>) -> List<T>:
  Array.to_list.go(~T, a, Nil{})

def Array.map.go(
  ~T: Data, ~U: Data, ~f: T -> U, +n: Nat, +i: U32, b: Array<U>,
  r: Array<T> & T
) -> Array<U>:
  match n:
    case 0n:
      b
    case 1n+p:
      (a, x) = r
      +j = U32.inc(i)
      Array.map.go(~T, ~U, ~f, p, j, Array.set(U, b, i, f(x)),
        Array.get(T, a, j))

def Array.map.at(
  ~T: Data, ~U: Data, ~f: T -> U, v: U, an: Array<T> & U32
) -> Array<U>:
  (a, +n) = an
  Array.map.go(~T, ~U, ~f, U32.to_nat(U32.sub(n, 1)), 1,
    Array.new(U, U32.log2(n), v), Array.get(T, a, 1))

def Array.map.seed(~T: Data, ~U: Data, ~f: T -> U, r: Array<T> & T) -> Array<U>:
  (a, x) = r
  Array.map.at(~T, ~U, ~f, f(x), Array.size(T, a))

def Array.map(~T: Data, ~U: Data, ~f: T -> U, a: Array<T>) -> Array<U>:
  Array.map.seed(~T, ~U, ~f, Array.get(T, a, 0))

# Map
# ---

def Map.bit.u(x: U32, k: Nat) -> Bool:
  U32.is_ne(U32.and(U32.shrn(x, k), 1), 0)

def Map.bit.chr(c: Char, off: Nat) -> Char & Bool:
  match c off:
    case Chr{x} 0n:
      (Chr{x}, True{})
    case Chr{+x} 1n+b:
      (Chr{x}, Map.bit.u(x, Nat.sub(31n, b)))

def Map.bit.go.chr(t: String, r: Char & Bool) -> String & Bool:
  (c2, b) = r
  (SCon{c2, t}, b)

def Map.bit.go.rec(c: Char, r: String & Bool) -> String & Bool:
  (t2, b) = r
  (SCon{c, t2}, b)

def Map.bit.go(key: String, ci: Nat, off: Nat) -> String & Bool:
  match key:
    case SNil{}:
      (SNil{}, False{})
    case SCon{c, t}:
      match ci:
        case 0n:
          Map.bit.go.chr(t, Map.bit.chr(c, off))
        case 1n+j:
          Map.bit.go.rec(c, Map.bit.go(t, j, off))

def Map.bit.at(key: String, co: Nat & Nat) -> String & Bool:
  (ci, off) = co
  Map.bit.go(key, ci, off)

def Map.bit(key: String, pos: Nat) -> String & Bool:
  Map.bit.at(key, Nat.divmod(pos, 33n))

law Map.msb.u:
  for n: Nat
  for +x: U32
  Nat

def Map.msb.u.if(p: Nat, x2: U32, z: Bool) -> Nat:
  match z:
    case True{}:
      0n
    case False{}:
      Nat.add(1n, Map.msb.u(p, U32.shr(x2)))

def Map.msb.u(n, x):
  match n:
    case 0n:
      0n
    case 1n+p:
      Map.msb.u.if(p, x, U32.is_zero(x))

def Map.diff.chr(x: U32) -> Nat:
  Nat.sub(33n, Map.msb.u(32n, x))

def Map.diff.step(x: Char, y: Char) -> Nat & Bool:
  match x y:
    case Chr{+cx} Chr{+cy}:
      (Map.diff.chr(U32.xor(cx, cy)), U32.is_eq(cx, cy))

law Map.diff:
  for a: String
  for b: String
  Nat

def Map.diff.fin(xt: String, yt: String, rc: Nat & Bool) -> Nat:
  (r, c) = rc
  match c:
    case True{}:
      Nat.add(33n, Map.diff(xt, yt))
    case False{}:
      r

def Map.diff(a, b):
  match a b:
    case SNil{} SNil{}:
      0n
    case SNil{} SCon{h, t}:
      0n
    case SCon{h, t} SNil{}:
      0n
    case SCon{x, xt} SCon{y, yt}:
      Map.diff.fin(xt, yt, Map.diff.step(x, y))

def Map.new(a, -V: Kind(a)) -> Map<a, V>:
  MTip{}

law Map.put:
  for -a  : Quant
  for -V  : Kind(a)
  for m   : Map<a, V>
  for key : String
  for x   : V
  Map<a, V>

def Map.put.bit(
  a, -V: Kind(a), x: V, p2: Nat, lo: Map<a, V>, hi: Map<a, V>, kb: String & Bool
) -> Map<a, V>:
  (key2, b) = kb
  match b:
    case False{}:
      MNode{p2, Map.put(a, V, lo, key2, x), hi}
    case True{}:
      MNode{p2, lo, Map.put(a, V, hi, key2, x)}

def Map.put(a, V, m, key, x):
  match m:
    case MTip{}:
      MLeaf{key, x}
    case MLeaf{k, v}:
      MLeaf{k, x}
    case MNode{+pos, lo, hi}:
      Map.put.bit(a, V, x, pos, lo, hi, Map.bit(key, pos))

def Map.ins.splice.bit(
  a, -V: Kind(a), x: V, rest: Map<a, V>, pb: Nat, kb: String & Bool
) -> Map<a, V>:
  (key2, b) = kb
  match b:
    case False{}:
      MNode{pb, MLeaf{key2, x}, rest}
    case True{}:
      MNode{pb, rest, MLeaf{key2, x}}

def Map.ins.splice(
  a, -V: Kind(a), +p: Nat, key: String, x: V, rest: Map<a, V>
) -> Map<a, V>:
  Map.ins.splice.bit(a, V, x, rest, p, Map.bit(key, p))

law Map.ins:
  for -a  : Quant
  for -V  : Kind(a)
  for m   : Map<a, V>
  for key : String
  for x   : V
  for +p  : Nat
  Map<a, V>

def Map.ins.deep(
  a, -V: Kind(a), x: V, lo: Map<a, V>, hi: Map<a, V>, pb: Nat, qb: Nat,
  kb: String & Bool
) -> Map<a, V>:
  (key2, b) = kb
  match b:
    case False{}:
      MNode{qb, Map.ins(a, V, lo, key2, x, pb), hi}
    case True{}:
      MNode{qb, lo, Map.ins(a, V, hi, key2, x, pb)}

def Map.ins.if(
  a, -V: Kind(a), key: String, x: V, lo: Map<a, V>, hi: Map<a, V>, +p2: Nat,
  pb: Nat, t: Bool
) -> Map<a, V>:
  match t:
    case False{}:
      Map.ins.splice(a, V, pb, key, x, MNode{p2, lo, hi})
    case True{}:
      Map.ins.deep(a, V, x, lo, hi, pb, p2, Map.bit(key, p2))

def Map.ins(a, V, m, key, x, p):
  match m:
    case MTip{}:
      MLeaf{key, x}
    case MLeaf{k, v}:
      Map.ins.splice(a, V, p, key, x, MLeaf{k, v})
    case MNode{+pos, lo, hi}:
      Map.ins.if(a, V, key, x, lo, hi, pos, p, Nat.is_lt(pos, p))

def Map.lo(
  a, -V: Kind(a), -R: Type, p2: Nat, hi: Map<a, V>, r0: Map<a, V> & R
) -> Map<a, V> & R:
  (lo2, r) = r0
  (MNode{p2, lo2, hi}, r)

def Map.hi(
  a, -V: Kind(a), -R: Type, p2: Nat, lo: Map<a, V>, r0: Map<a, V> & R
) -> Map<a, V> & R:
  (hi2, r) = r0
  (MNode{p2, lo, hi2}, r)

law Map.seek:
  for -a  : Quant
  for -V  : Kind(a)
  for m   : Map<a, V>
  for key : String
  Map<a, V> & String & Maybe<&2, String>

def Map.seek.bit(
  a, -V: Kind(a), lo: Map<a, V>, hi: Map<a, V>, p2: Nat, kb: String & Bool
) -> Map<a, V> & String & Maybe<&2, String>:
  (key2, b) = kb
  match b:
    case False{}:
      Map.lo(a, V, String & Maybe<&2, String>, p2, hi, Map.seek(a, V, lo, key2))
    case True{}:
      Map.hi(a, V, String & Maybe<&2, String>, p2, lo, Map.seek(a, V, hi, key2))

def Map.seek(a, V, m, key):
  match m:
    case MTip{}:
      (MTip{}, key, None{})
    case MLeaf{+k, v}:
      (MLeaf{k, v}, key, Some{k})
    case MNode{+pos, lo, hi}:
      Map.seek.bit(a, V, lo, hi, pos, Map.bit(key, pos))

def Map.set.fin.go(
  a, -V: Kind(a), m: Map<a, V>, key: String, x: V, r: (String & String) & Cmp
) -> Map<a, V>:
  ((keyb2, k2), c) = r
  match c:
    case LT{}:
      Map.ins(a, V, m, key, x, Map.diff(keyb2, k2))
    case EQ{}:
      Map.put(a, V, m, key, x)
    case GT{}:
      Map.ins(a, V, m, key, x, Map.diff(keyb2, k2))

def Map.set.fin(
  a, -V: Kind(a), m: Map<a, V>, key: String, x: V, keyb: String, k: String
) -> Map<a, V>:
  Map.set.fin.go(a, V, m, key, x, String.cmp(keyb, k))

def Map.set.go(
  a, -V: Kind(a), x: V, r: Map<a, V> & String & Maybe<&2, String>
) -> Map<a, V>:
  (m2, key2, found) = r
  match found:
    case None{}:
      MLeaf{key2, x}
    case Some{k}:
      +ka = {key2 : String}
      Map.set.fin(a, V, m2, ka, x, ka, k)

def Map.set(a, -V: Kind(a), m: Map<a, V>, key: String, x: V) -> Map<a, V>:
  Map.set.go(a, V, x, Map.seek(a, V, m, key))

def Map.has.leaf(a, -V: Kind(a), v: V, r: (String & String) & Cmp) ->
  Map<a, V> & Bool:
  ((key2, k2), c) = r
  (MLeaf{k2, v}, Cmp.is_eq(c))

law Map.has:
  for -a  : Quant
  for -V  : Kind(a)
  for m   : Map<a, V>
  for key : String
  Map<a, V> & Bool

def Map.has.bit(
  a, -V: Kind(a), lo: Map<a, V>, hi: Map<a, V>, p2: Nat, kb: String & Bool
) -> Map<a, V> & Bool:
  (key2, b) = kb
  match b:
    case False{}:
      Map.lo(a, V, Bool, p2, hi, Map.has(a, V, lo, key2))
    case True{}:
      Map.hi(a, V, Bool, p2, lo, Map.has(a, V, hi, key2))

def Map.has(a, V, m, key):
  match m:
    case MTip{}:
      (MTip{}, False{})
    case MLeaf{k, v}:
      Map.has.leaf(a, V, v, String.cmp(key, k))
    case MNode{+pos, lo, hi}:
      Map.has.bit(a, V, lo, hi, pos, Map.bit(key, pos))

def Map.get.leaf(-V: Data, d: V, +v: V, r: (String & String) & Cmp) ->
  Map<&2, V> & V:
  ((key2, k2), c) = r
  match c:
    case LT{}:
      (MLeaf{k2, v}, d)
    case EQ{}:
      (MLeaf{k2, v}, v)
    case GT{}:
      (MLeaf{k2, v}, d)

law Map.get:
  for -V  : Data
  for d   : V
  for m   : Map<&2, V>
  for key : String
  Map<&2, V> & V

def Map.get.bit(
  -V: Data, d: V, lo: Map<&2, V>, hi: Map<&2, V>, p2: Nat, kb: String & Bool
) -> Map<&2, V> & V:
  (key2, b) = kb
  match b:
    case False{}:
      Map.lo(&2, V, V, p2, hi, Map.get(V, d, lo, key2))
    case True{}:
      Map.hi(&2, V, V, p2, lo, Map.get(V, d, hi, key2))

def Map.get(V, d, m, key):
  match m:
    case MTip{}:
      (MTip{}, d)
    case MLeaf{k, v}:
      Map.get.leaf(V, d, v, String.cmp(key, k))
    case MNode{+pos, lo, hi}:
      Map.get.bit(V, d, lo, hi, pos, Map.bit(key, pos))

def Map.pop.lo(
  a, -V: Kind(a), pos: Nat, hi: Map<a, V>, r0: Map<a, V> & Maybe<a, V>
) -> Map<a, V> & Maybe<a, V>:
  (lo, r) = r0
  match lo:
    case MTip{}:
      (hi, r)
    case lo2:
      (MNode{pos, lo2, hi}, r)

def Map.pop.hi(
  a, -V: Kind(a), pos: Nat, lo: Map<a, V>, r0: Map<a, V> & Maybe<a, V>
) -> Map<a, V> & Maybe<a, V>:
  (hi, r) = r0
  match hi:
    case MTip{}:
      (lo, r)
    case hi2:
      (MNode{pos, lo, hi2}, r)

def Map.pop.leaf(a, -V: Kind(a), v: V, r: (String & String) & Cmp) ->
  Map<a, V> & Maybe<a, V>:
  ((key2, k2), c) = r
  match c:
    case LT{}:
      (MLeaf{k2, v}, None{})
    case EQ{}:
      (MTip{}, Some{v})
    case GT{}:
      (MLeaf{k2, v}, None{})

law Map.pop:
  for -a  : Quant
  for -V  : Kind(a)
  for m   : Map<a, V>
  for key : String
  Map<a, V> & Maybe<a, V>

def Map.pop.bit(
  a, -V: Kind(a), lo: Map<a, V>, hi: Map<a, V>, p2: Nat, kb: String & Bool
) -> Map<a, V> & Maybe<a, V>:
  (key2, b) = kb
  match b:
    case False{}:
      Map.pop.lo(a, V, p2, hi, Map.pop(a, V, lo, key2))
    case True{}:
      Map.pop.hi(a, V, p2, lo, Map.pop(a, V, hi, key2))

def Map.pop(a, V, m, key):
  match m:
    case MTip{}:
      (MTip{}, None{})
    case MLeaf{k, v}:
      Map.pop.leaf(a, V, v, String.cmp(key, k))
    case MNode{+pos, lo, hi}:
      Map.pop.bit(a, V, lo, hi, pos, Map.bit(key, pos))

def Map.del.fin(a, -V: Kind(a), r: Map<a, V> & Maybe<a, V>) -> Map<a, V>:
  (m2, x) = r
  m2

def Map.del(a, -V: Kind(a), m: Map<a, V>, key: String) -> Map<a, V>:
  Map.del.fin(a, V, Map.pop(a, V, m, key))

def Map.to_list.go(
  a, -V: Kind(a), m: Map<a, V>, acc: List<a, Sigma<&2, a, String, _ => V>>
) -> List<a, Sigma<&2, a, String, _ => V>>:
  match m:
    case MTip{}:
      acc
    case MLeaf{k, v}:
      (k, v) <> acc
    case MNode{pos, lo, hi}:
      Map.to_list.go(a, V, lo, Map.to_list.go(a, V, hi, acc))

def Map.to_list(a, -V: Kind(a), m: Map<a, V>) ->
  List<a, Sigma<&2, a, String, _ => V>>:
  Map.to_list.go(a, V, m, Nil{})

def Map.keys.go(a, -V: Kind(a), m: Map<a, V>, acc: List<&2, String>) ->
  List<&2, String>:
  match m:
    case MTip{}:
      acc
    case MLeaf{k, v}:
      k <> acc
    case MNode{pos, lo, hi}:
      Map.keys.go(a, V, lo, Map.keys.go(a, V, hi, acc))

def Map.keys(a, -V: Kind(a), m: Map<a, V>) -> List<&2, String>:
  Map.keys.go(a, V, m, Nil{})

def Map.from_list.go(
  a, -V: Kind(a), kvs: List<a, Sigma<&2, a, String, _ => V>>, m: Map<a, V>
) -> Map<a, V>:
  match kvs:
    case Nil{}:
      m
    case (k, v) <> t:
      Map.from_list.go(a, V, t, Map.set(a, V, m, k, v))

def Map.from_list(a, -V: Kind(a), kvs: List<a, Sigma<&2, a, String, _ => V>>) ->
  Map<a, V>:
  Map.from_list.go(a, V, kvs, MTip{})

def Map.union(a, -V: Kind(a), m: Map<a, V>, n: Map<a, V>) -> Map<a, V>:
  Map.from_list.go(a, V, Map.to_list(a, V, n), m)

def Map.size(a, -V: Kind(a), m: Map<a, V>) -> Nat:
  match m:
    case MTip{}:
      0n
    case MLeaf{k, v}:
      1n
    case MNode{pos, lo, hi}:
      Nat.add(Map.size(a, V, lo), Map.size(a, V, hi))

def Map.values.go(a, -V: Kind(a), m: Map<a, V>, acc: List<a, V>) -> List<a, V>:
  match m:
    case MTip{}:
      acc
    case MLeaf{k, v}:
      v <> acc
    case MNode{pos, lo, hi}:
      Map.values.go(a, V, lo, Map.values.go(a, V, hi, acc))

def Map.values(a, -V: Kind(a), m: Map<a, V>) -> List<a, V>:
  Map.values.go(a, V, m, Nil{})

# Set
# ---

def Set() -> Data:
  Map<&2, Unit>

def Set.new() -> Set():
  Map.new(&2, Unit)

def Set.add(s: Set(), key: String) -> Set():
  Map.set(&2, Unit, s, key, Unit{})

def Set.has(s: Set(), key: String) -> Set() & Bool:
  Map.has(&2, Unit, s, key)

def Set.del(s: Set(), key: String) -> Set():
  Map.del(&2, Unit, s, key)

def Set.size(s: Set()) -> Nat:
  Map.size(&2, Unit, s)

def Set.to_list(s: Set()) -> List<&2, String>:
  Map.keys(&2, Unit, s)

def Set.from_list.go(keys: List<&2, String>, s: Set()) -> Set():
  match keys:
    case Nil{}:
      s
    case h <> t:
      Set.from_list.go(t, Set.add(s, h))

def Set.from_list(keys: List<&2, String>) -> Set():
  Set.from_list.go(keys, Set.new())

# Image
# -----

def Image.sink(img: Image) -> Unit:
  Unit{}

def Image.drop.join(a: Unit, b: Unit, c: Unit, d: Unit) -> Unit:
  match a b c d:
    case Unit{} Unit{} Unit{} Unit{}:
      Unit{}

# Forks k levels, then a dead parameter sinks the rest in one loop.
def Image.free(+k: Nat, img: Image) -> Unit:
  match k:
    case 0n:
      Image.sink(img)
    case 1n+e:
      match img:
        case Pix{c}:
          Unit{}
        case Qua{tl, tr, bl, br}:
          a b c d = Image.free(e, tl) Image.free(e, tr) Image.free(e, bl)
            Image.free(e, br)
          Image.drop.join(a, b, c, d)

def Image.drop(img: Image) -> Unit:
  Image.free(7n, img)

# App
# ---

def App.next(
  -S: Type, window: Window, rest: Window -> S -> IO(Unit), next: Maybe<S>
) -> IO(Unit):
  match next:
    case None{}:
      Window.close(window)
    case Some{state}:
      rest(window, state)

def App.turn(
  -S: Type, shown: Window & Image & List<Event>,
  tick: List<Event> -> S -> IO(Maybe<S>), state: S,
  rest: Window -> S -> IO(Unit)
) -> IO(Unit):
  (window, image, events) = shown
  do IO<Unit>:
    Unit <- IO.pure(Unit, Image.drop!(image))
    IO.bind(Maybe<S>, Unit, tick(events, state), App.next(S, window, rest))

def App.draw(
  -S: Type, window: Window, drawn: S & Image,
  tick: List<Event> -> S -> IO(Maybe<S>), rest: Window -> S -> IO(Unit)
) -> IO(Unit):
  (state, image) = drawn
  do IO<Unit>:
    shown : Window & Image & List<Event> <- Window.frame(window, image)
    App.turn(S, shown, tick, state, rest)

def App.step(
  -S: Type, app: App<S>, window: Window, state: S,
  rest: Window -> S -> IO(Unit)
) -> IO(Unit):
  App{view, tick} = app
  App.draw(S, window, view(state), tick, rest)

def App.loop(~S: Type, ~app: App<S>, fuel: Nat, window: Window, state: S) ->
  IO(Unit):
  match fuel:
    case 0n:
      Window.close(window)
    case 1n+f:
      App.step(S, app, window, state, w => s => App.loop(~S, ~app, f, w, s))

def App.run(
  ~S: Type, ~app: App<S>, title: String, width: U32, height: U32, state: S
) -> IO(Unit):
  do IO<Unit>:
    window : Window <- IO.try(Window, Window.open(title, width, height))
    App.loop(~S, ~app, U32.to_nat(4294967295), window, state)

def App.more(-S: Type, rest: S -> IO(Maybe<S>), next: Maybe<S>) ->
  IO(Maybe<S>):
  match next:
    case None{}:
      IO.pure(Maybe<S>, None{})
    case Some{state}:
      rest(state)

def App.fold(
  -S: Type, app: App<S>, events: List<Event>, state: S,
  rest: S -> IO(Maybe<S>)
) -> IO(Maybe<S>):
  App{view, tick} = app
  IO.bind(Maybe<S>, Maybe<S>, tick(events, state), App.more(S, rest))

def App.play(~S: Type, ~app: App<S>, frames: List<List<Event>>, state: S) ->
  IO(Maybe<S>):
  match frames:
    case Nil{}:
      IO.pure(Maybe<S>, Some{state})
    case events <> rest:
      App.fold(S, app, events, state, s => App.play(~S, ~app, rest, s))
