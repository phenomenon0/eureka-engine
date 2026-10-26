# Borrowed reads must retain exact bits, keep owners live, and not change
# get/set's consuming API or explicit Array.fork aliasing semantics.
import Base

def mk() -> Array<F32>:
  Array.set(F32, Array.set(F32, Array.set(F32,
    Array.new(F32, 2n, 1.0), 1, 2.0), 2, 3.0), 3, 4.0)

@unsafe
def borrowed_group(+a: Array<F32>, +i: U32) -> F32:
  ((Array.peek_f32(a, i) + Array.peek_f32(a, (i + 1 : U32)) : F32) +
    (Array.peek_f32(a, (i + 2 : U32)) + Array.peek_f32(a, (i + 3 : U32)) : F32) : F32)

# Keep construction at the entry boundary, like the engine's caller.
def group_once(a: Array<F32>, i: U32) -> F32:
  borrowed_group(a, i)

def peek_once(a: Array<F32>, i: U32) -> F32:
  Array.peek_f32(a, i)

def temporary_read(a: Array<F32>) -> F32:
  Array.peek_f32(Array.set(F32, a, 2, 9.0), 2)

def owning_group.fin(r: Array<F32> & (F32 & (F32 & (F32 & F32)))) -> F32:
  (rest, (x0, (x1, (x2, x3)))) = r
  ((x0 + x1 : F32) + (x2 + x3 : F32) : F32)

def owning_group(a: Array<F32>, i: U32) -> F32:
  owning_group.fin(Array.get4(F32, a, i))

def get_value(r: Array<F32> & F32) -> F32:
  (rest, value) = r
  value

# The first scalar must be materialized before the later mutation.
@unsafe
def read_then_set(+a: Array<F32>) -> F32:
  before = Array.peek_f32(a, 0)
  (before + get_value(Array.get(F32, Array.set(F32, a, 0, 7.0), 0)) : F32)

# Mutation through a forked handle must remain visible to the other handle.
def alias_read(a: Array<F32>, b: Array<F32>) -> F32:
  before = Array.peek_f32(b, 0)
  (before + get_value(Array.get(F32, a, 0)) : F32)

def alias_write(pair: Array<F32> & Array<F32>) -> F32:
  (a, b) = pair
  alias_read(Array.set(F32, a, 0, 7.0), b)

# Nested reads must finish before a later argument mutates the shared array.
def alias_order(pair: Array<F32> & Array<F32>) -> F32:
  (a, b) = pair
  F32.add(Array.peek_f32(a, 0), Array.peek_f32(Array.set(F32, b, 0, 7.0), 0))

# A peek whose index also reads the handle exercises the held-argument path.
@unsafe
def dependent_index(+a: Array<F32>) -> F32:
  Array.peek_f32(a, F32.to_u32(Array.peek_f32(a, 0)))

@unsafe
def read_loop(n: Nat, +a: Array<F32>, +i: U32) -> F32:
  match n:
    case 0n:
      0.0
    case 1n+p:
      (Array.peek_f32(a, i) + read_loop(p, a, (i + 1 : U32)) : F32)

def loop_once(a: Array<F32>) -> F32:
  read_loop(32n, a, 0)

# Held call sites must lend across helpers rather than bump counts per read.
@unsafe
def reuse_group(+a: Array<F32>) -> F32:
  (borrowed_group(a, 0) + borrowed_group(a, 0) : F32)

@unsafe
def reuse_loop(+a: Array<F32>) -> F32:
  (read_loop(32n, a, 0) + read_loop(32n, a, 0) : F32)

# A dying owner must stay alive until both forked readers have returned.
@unsafe
def fork_reads(+a: Array<F32>) -> F32:
  x y = read_loop(32n, a, 0) read_loop(32n, a, 0)
  (x + y : F32)

def show(x: F32) -> IO(Unit):
  IO.print(U32.show(F32.bits(x)))

def cancellation() -> Array<F32>:
  Array.set(F32, Array.set(F32, Array.set(F32,
    Array.new(F32, 2n, 16777216.0), 1, 1.0), 2, F32.neg(16777216.0)), 3, F32.neg(1.0))

def word_read(b: Bytes(), i: Nat) -> U32:
  Bytes.word_le(b, i)

# Every case is a bang so --gpu on checks the device path, not just the fork.
def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- show(group_once!(mk(), 0))
    Unit <- show(owning_group!(mk(), 0))
    Unit <- show(group_once!(mk(), 3))
    Unit <- show(temporary_read!(mk()))
    Unit <- show(read_then_set!(mk()))
    Unit <- show(alias_write!(Array.fork(F32, mk())))
    Unit <- show(alias_order!(Array.fork(F32, Array.new(F32, 0n, 1.0))))
    Unit <- show(dependent_index!(mk()))
    Unit <- show(loop_once!(mk()))
    Unit <- show(reuse_group!(mk()))
    Unit <- show(reuse_loop!(mk()))
    Unit <- show(fork_reads!(mk()))
    Unit <- show(group_once!(cancellation(), 0))
    Unit <- show(owning_group!(cancellation(), 0))
    Unit <- show(peek_once!(Array.new(F32, 0n, F32.neg(0.0)), 4294967295))
    # Aligned, unaligned, partial-tail, and out-of-range word loads.
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 0n)))
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 1n)))
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 4n)))
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 6n)))
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 8n)))
    Unit <- IO.print(U32.show(word_read!("abcdefgh", 100n)))
    # Alignment is relative to the backing buffer, not the sliced view.
    Unit <- IO.print(U32.show(word_read!(Bytes.slice("abcdefgh", 1n, 7n), 0n)))
    Unit <- IO.print(U32.show(word_read!(Bytes.slice("abcdefgh", 1n, 7n), 3n)))
    IO.print(U32.show(word_read!(Bytes.slice("abcdefgh", 2n, 4n), 1n)))

#|1092616192
#|1092616192
#|1092616192
#|1091567616
#|1090519040
#|1096810496
#|1090519040
#|1073741824
#|1117782016
#|1101004800
#|1126170624
#|1126170624
#|0
#|0
#|2147483648
#|1684234849
#|1701077858
#|1751606885
#|26727
#|0
#|0
#|1701077858
#|1751606885
#|6710628
