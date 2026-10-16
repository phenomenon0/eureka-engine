# Time: the dates of IO.wall's clock, in UTC (no zones, no leap seconds).
#
#   import <bend2>/time.bend as Time
#   Time.iso(951827696n)    # "2000-02-29T12:34:56Z"  (seconds since the epoch)
#   Time.iso.ms(t)          # "2000-02-29T12:34:56.789Z"  (ms, as IO.wall answers)
#   Time.http(951827696n)   # "Tue, 29 Feb 2000 12:34:56 GMT"  (IMF-fixdate)
#   Time.date(11016n)       # (2000n, (2n, 29n)): a day from 1970-01-01
#   Time.day(2000n, 2n, 29n) # Some{11016n}, None for a day no month has
#
# The calendar is demos/io_http_engine/cond.bend's, whose LAWS.bend proves
# it (date_round_trip, and the dated vectors): a year counted from 1970
# with the Gregorian leap rule, 1461-day blocks of four years to 2097,
# then whole years and whole months walked. Every number is a Nat, so
# no date overflows. tests/base/time.bend walks the round trip.
import Base

# The calendar
# ============

# year 1970 + k is a leap year by the Gregorian rule
def leap.c(+y: Nat) -> Bool:
  Bool.not(Nat.is_eq(Nat.mod(y, 100n), 0n)) || Nat.is_eq(Nat.mod(y, 400n), 0n)

def leap.at(q: Bool, +y: Nat) -> Bool:
  match q:
    case True{}:
      leap.c(y)
    case False{}:
      False{}

def leap(k: Nat) -> Bool:
  +y = Nat.add(k, 1970n)
  leap.at(Nat.is_eq(Nat.mod(y, 4n), 0n), y)

def ylen(k: Nat) -> Nat:
  Bool.pick(Nat, leap(k), 366n, 365n)

# the months' lengths, January first
def mlens(lp: Bool) -> List<&2, Nat>:
  match lp:
    case True{}:
      [31n, 29n, 31n, 30n, 31n, 30n, 31n, 31n, 30n, 31n, 30n, 31n]
    case False{}:
      [31n, 28n, 31n, 30n, 31n, 30n, 31n, 31n, 30n, 31n, 30n, 31n]

# the days before year k
def before(+k: Nat) -> Nat:
  match k:
    case 0n:
      0n
    case 1n+j:
      Nat.add(before(j), ylen(j))

# day rem of a year as a month and a day of it (both from 0): each month
# taken whole while rem is past it (l its length, t the months after it)
def mon.up(jd: Nat & Nat) -> Nat & Nat:
  (j, d) = jd
  (1n+j, d)

def mon(t: List<&2, Nat>, +l: Nat, fits: Bool, +rem: Nat) -> Nat & Nat:
  match t:
    case Nil{}:
      (0n, rem)
    case Con{+l2, t2}:
      match fits:
        case True{}:
          (0n, rem)
        case False{}:
          +r2 = Nat.sub(rem, l)
          mon.up(mon(t2, l2, Nat.is_lt(r2, l2), r2))

def mon.of(ls: List<&2, Nat>, +rem: Nat) -> Nat & Nat:
  match ls:
    case Nil{}:
      (0n, rem)
    case Con{+l, t}:
      mon(t, l, Nat.is_lt(rem, l), rem)

# day rem from year k on: each year taken whole while rem is past it (yl
# its length); f bounds the walk
def cal(f: Nat, fits: Bool, +rem: Nat, +k: Nat, +yl: Nat) -> Nat & Nat & Nat:
  match f:
    case 0n:
      (k, (0n, rem))
    case 1n+g:
      match fits:
        case True{}:
          (k, mon.of(mlens(leap(k)), rem))
        case False{}:
          +r2 = Nat.sub(rem, yl)
          +y2 = ylen(1n+k)
          cal(g, Nat.is_lt(r2, y2), r2, 1n+k, y2)

# whole blocks of four years first: each of the first 32 has 1461 days
def blk(f: Nat, ok: Bool, +q: Nat, +k: Nat, +rem: Nat) -> Nat & Nat:
  match f:
    case 0n:
      (k, rem)
    case 1n+g:
      match ok:
        case True{}:
          +r2 = Nat.sub(rem, 1461n)
          blk(g, Nat.is_lt(1n+q, 32n) && Nat.is_le(1461n, r2), 1n+q, 1n+1n+1n+1n+k, r2)
        case False{}:
          (k, rem)

def civil.from(kr: Nat & Nat) -> Nat & Nat & Nat:
  (+k, +r) = kr
  +y = ylen(k)
  cal(1n+r, Nat.is_lt(r, y), r, k, y)

# a day as its year from 1970, month from 0 and day of the month from 0
def civil(+days: Nat) -> Nat & Nat & Nat:
  civil.from(blk(32n, Nat.is_le(1461n, days), 0n, 0n, days))

# ... and back: the days before a day of a month, when the month has it
def mdays.fit(ok: Bool, d: Nat) -> Maybe<&2, Nat>:
  match ok:
    case True{}:
      Some{d}
    case False{}:
      None{}

def mdays.add(+l: Nat, m: Maybe<&2, Nat>) -> Maybe<&2, Nat>:
  match m:
    case None{}:
      None{}
    case Some{x}:
      Some{Nat.add(l, x)}

def mdays(ls: List<&2, Nat>, j: Nat, +d: Nat) -> Maybe<&2, Nat>:
  match ls:
    case Nil{}:
      None{}
    case Con{+l, t}:
      match j:
        case 0n:
          mdays.fit(Nat.is_lt(d, l), d)
        case 1n+i:
          mdays.add(l, mdays(t, i, d))

# Dates
# =====

def date.of(kjd: Nat & Nat & Nat) -> Nat & Nat & Nat:
  (k, jd) = kjd
  (j, d) = jd
  (Nat.add(k, 1970n), (1n+j, 1n+d))

# a day (from 1970-01-01) as its year, month (1 to 12) and day (1 to 31)
def date(+days: Nat) -> Nat & Nat & Nat:
  date.of(civil(days))

def day.at(ok: Bool, +k: Nat, +m: Nat, +d: Nat) -> Maybe<&2, Nat>:
  match ok:
    case True{}:
      mdays.add(before(k), mdays(mlens(leap(k)), Nat.sub(m, 1n), Nat.sub(d, 1n)))
    case False{}:
      None{}

# the day a date is, from 1970-01-01: None before 1970, or for a day its
# month does not have
def day(+y: Nat, +m: Nat, +d: Nat) -> Maybe<&2, Nat>:
  day.at(Nat.is_le(1970n, y) && Nat.is_le(1n, m) && Nat.is_le(1n, d), Nat.sub(y, 1970n), m, d)

# Text
# ====

def pad(+w: Nat, +n: Nat) -> String:
  String.pad_start(Nat.show(n), w, '0')

def nth(xs: List<&2, String>, i: Nat) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{x, t}:
      match i:
        case 0n:
          x
        case 1n+j:
          nth(t, j)

# a second of a day as HH:MM:SS
def hms(+s: Nat) -> String:
  pad(2n, Nat.div(s, 3600n)) ++ ":" ++ pad(2n, Nat.mod(Nat.div(s, 60n), 60n)) ++ ":" ++ pad(2n, Nat.mod(s, 60n))

def iso.ymd(ymd: Nat & Nat & Nat) -> String:
  (y, md) = ymd
  (m, d) = md
  pad(4n, y) ++ "-" ++ pad(2n, m) ++ "-" ++ pad(2n, d)

def iso.of(ds: Nat & Nat) -> String:
  (days, s) = ds
  iso.ymd(date(days)) ++ "T" ++ hms(s)

# seconds since the epoch as ISO 8601 in UTC: 2000-02-29T12:34:56Z
def iso(t: Nat) -> String:
  iso.of(Nat.divmod(t, 86400n)) ++ "Z"

# ms since the epoch (IO.wall's) as ISO 8601 in UTC: ...T12:34:56.789Z
def iso.ms(+t: Nat) -> String:
  iso.of(Nat.divmod(Nat.div(t, 1000n), 86400n)) ++ "." ++ pad(3n, Nat.mod(t, 1000n)) ++ "Z"

def wkc() -> List<&2, String>:
  ["Thu, ", "Fri, ", "Sat, ", "Sun, ", "Mon, ", "Tue, ", "Wed, "]

def monc() -> List<&2, String>:
  ["Jan ", "Feb ", "Mar ", "Apr ", "May ", "Jun ", "Jul ", "Aug ", "Sep ", "Oct ", "Nov ", "Dec "]

def http.ymd(ymd: Nat & Nat & Nat) -> String:
  (y, md) = ymd
  (m, d) = md
  pad(2n, d) ++ " " ++ nth(monc(), Nat.sub(m, 1n)) ++ pad(4n, y)

def http.of(ds: Nat & Nat) -> String:
  (+days, s) = ds
  nth(wkc(), Nat.mod(days, 7n)) ++ http.ymd(date(days)) ++ " " ++ hms(s) ++ " GMT"

# seconds since the epoch as an IMF-fixdate (RFC 9110 5.6.7): the HTTP
# Date, Last-Modified and Expires form, Tue, 29 Feb 2000 12:34:56 GMT
def http(t: Nat) -> String:
  http.of(Nat.divmod(t, 86400n))
